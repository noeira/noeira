# +--------------------------------------------------------------------------+ #
# | Retargeted AMASS for the G1 → `TrajectoryStore` rows
# +--------------------------------------------------------------------------+ #
"""`fleaven/Retargeted_AMASS_for_robotics` for the Unitree G1 29-DoF, read
natively — §10.7 of `docs/BFM_ZERO_NEXT_LEVEL.md`.

**THE RETARGETING IS NOT RUN HERE.** That dump is AMASS already retargeted
to this exact robot, CC-BY-4.0, 6.95 GB, **17 717 clips / 59.4 hours** —
24× the 2.45 h of LAFAN1 the checkpoint was trained on, which is the
ceiling every inner limit of §10.2 sits under.

THE FILE FORMAT, from the dataset's own `g1/visualize.py` (the only
normative statement of it — the README does not give the layout):

    d.qpos        = data[step, :7 + 29]     # 36 columns
    d.qpos[3]     = data[step, 6]           # w first
    d.qpos[4:7]   = data[step, 3:6]
    jpos[:, 2]   += 0.793                   # in `read_rtj`
    fr            = int(fpath[-12:-9])      # the fps is IN THE FILE NAME

So one `[N, 36]` float64 `.npy` per clip:

  * `0:3`  root position, with **z stored relative to the nominal pelvis
    height**: `AMASS_ROOT_Z` (0.793 m) is added back here. Without it every
    clip walks along the floor plane with its pelvis at z ≈ 0, which
    `privileged[0]` reports as a height of zero and the reward vocabulary
    then reads as a permanent crouch;
  * `3:7`  the root quaternion **XYZW** — MuJoCo's `qpos` and our pipeline
    both want WXYZ, so it is rotated into place, not passed through;
  * `7:36` the 29 joint angles, in the G1's own actuator order (legs 12,
    waist 3, left arm 7, right arm 7). Both wrists' pitch and yaw are
    identically zero in every clip, which is the retargeting's own
    signature — SMPL has no wrist detail — and is what pins this ordering.

⚠ **THE FPS IS IN THE NAME AND IT IS NOT CONSTANT.** 120 Hz on 12 222
clips, 100 on 4 546, 60 on 871, 250 on 56, 150 on 9 and **59 on 13**.
`amass_fps_of_name` reproduces the dataset's own parse (`[-12:-9]`, a
leading `_` meaning a two-digit rate) rather than assuming 120, because a
59 Hz clip read as 120 Hz is resampled to half its duration at double its
velocities — plausible, wrong, and invisible in the store.

⚠ **CLIPS TOO SHORT TO CONVERT EXIST.** 68 are under one second and the
shortest is 0.01 s — a single frame. The reference's joint-velocity tail
reads index `n − 3`, so `convert_frames` raises under three frames;
`AMASS_MIN_SECONDS` is the importer's own floor above that.

⚠ **THE SUB-DATASETS CARRY THEIR OWN LICENCES.** CC-BY-4.0 covers the
retargeting, not the 25 sources; each ships a `license.txt` and a
`citation.bib` beside its clips, and `amass_license_files` lists them so
the importer can copy them next to the store it writes.

WHAT THIS MODULE IS NOT. It does not re-transcribe the observation
pipeline: steps 2–5 are `data/lafan.mojo`'s `convert_frames`, the code
`tests/robots/test_lafan_import_vs_oracle.mojo` holds against the
reference dump. Only step 1 — this file's layout — is new, and
`tests/data/test_amass.mojo` gates it against a clip whose numbers were
read out of the dump by hand.
"""

from noeira.data.lafan import LafanRows, convert_frames
from noeira.envs.robots import UnitreeG1
from noeira.envs.robots.unitree_g1_pd import G1_N_DOF
from noeira.io.npy import load_npy_2d_f64, npy_header_of


comptime AMASS_REPO = "fleaven/Retargeted_AMASS_for_robotics"
comptime AMASS_COLS: Int = 36             # 3 root pos + 4 root quat + 29 dof
comptime AMASS_ROOT_Z: Float64 = 0.793    # `read_rtj`: jpos[:, 2] += 0.793
comptime AMASS_NPY_HEADER: Int = 128      # v1.0 header of an [N, 36] f8 array
comptime AMASS_MIN_SECONDS: Float64 = 1.0
comptime AMASS_SUFFIX = "_jpos.npy"


struct AmassClip(Movable):
    """One clip in the form `convert_frames` takes."""
    var name: String                 # the repo-relative path, minus `g1/`
    var fps: Int
    var n: Int
    var root_pos: List[Float32]      # n * 3, pelvis height restored
    var root_quat: List[Float32]     # n * 4, WXYZ
    var dof: List[Float32]           # n * 29

    def __init__(
        out self, var name: String, fps: Int, n: Int,
        var root_pos: List[Float32], var root_quat: List[Float32],
        var dof: List[Float32],
    ):
        self.name = name^
        self.fps = fps
        self.n = n
        self.root_pos = root_pos^
        self.root_quat = root_quat^
        self.dof = dof^

    def __init__(out self, *, deinit move: Self):
        self.name = move.name^
        self.fps = move.fps
        self.n = move.n
        self.root_pos = move.root_pos^
        self.root_quat = move.root_quat^
        self.dof = move.dof^


def amass_fps_of_name(path: String) raises -> Int:
    """The dataset's own fps parse: `fr = fpath[-12:-9]`, `int(fr[1:])` when
    `fr[0] == '_'`.

    Written against the three-and-two-digit cases the dump actually holds
    (250/150/120/100 and 60/59) rather than a regex over any digits, so a
    name this parse cannot read raises instead of guessing 120.
    """
    var nb = path.byte_length()
    if nb < 12:
        raise Error("amass: '" + path + "' is too short to carry an fps tag")
    var fr = String(path[byte=nb - 12:nb - 9])
    var digits = fr
    if String(fr[byte=0:1]) == "_":
        digits = String(fr[byte=1:])
    for i in range(digits.byte_length()):
        var c = digits.as_bytes()[i]
        if c < 48 or c > 57:
            raise Error(
                "amass: '" + path + "' has `" + fr + "` where the fps tag"
                " belongs — the dump's own parse is name[-12:-9]"
            )
    var fps = Int(digits)
    if fps <= 0 or fps > 1000:
        raise Error("amass: '" + path + "' claims " + String(fps) + " fps")
    return fps


def amass_frames_of_size(size: Int) -> Int:
    """Frames in an `[N, 36]` float64 `.npy` of `size` bytes.

    The index the importer plans from carries sizes, not shapes, so the
    budget is computed from this and the fps tag WITHOUT downloading
    anything. Every clip in the dump has the 128-byte v1.0 header (checked
    per clip on load, where the real header is parsed).
    """
    var payload = size - AMASS_NPY_HEADER
    if payload <= 0:
        return 0
    return payload // (AMASS_COLS * 8)


def amass_seconds_of_size(path: String, size: Int) raises -> Float64:
    """The clip's duration from its listed size and its name. See above."""
    return Float64(amass_frames_of_size(size)) / Float64(amass_fps_of_name(path))


def amass_subset_of(path: String) -> String:
    """`g1/KIT/3/walk_...npy` -> `KIT`; the licence unit, and the selector."""
    var parts = path.split("/")
    for i in range(len(parts)):
        if String(parts[i]) == "g1" and i + 1 < len(parts):
            return String(parts[i + 1])
    if len(parts) >= 2:
        return String(parts[0])
    return String("")


def amass_is_clip(path: String) -> Bool:
    return path.endswith(AMASS_SUFFIX)


def load_amass_clip(path: String, var name: String) raises -> AmassClip:
    """One `.npy` in the form `convert_frames` takes.

    The three conversions the dataset's `visualize.py` does on the way into
    MuJoCo are done here and nowhere else: `+ AMASS_ROOT_Z` on the height,
    XYZW → WXYZ on the quaternion, and the fps off the name.

    ⚠ THE FPS COMES FROM `name`, NOT FROM `path`. It is a property of the
    clip, and `path` is wherever the bytes happen to sit — a cache entry, a
    `--source-dir` copy, a temporary. The two agree in the Hub cache, which
    is exactly why reading it off the local path worked and was still wrong.
    """
    var fps = amass_fps_of_name(name)
    var a = load_npy_2d_f64(path, AMASS_COLS)
    var n = a.rows
    var root_pos = List[Float32](length=n * 3, fill=Float32(0))
    var root_quat = List[Float32](length=n * 4, fill=Float32(0))
    var dof = List[Float32](length=n * G1_N_DOF, fill=Float32(0))
    for f in range(n):
        var b = f * AMASS_COLS
        root_pos[f * 3 + 0] = Float32(a.values[b + 0])
        root_pos[f * 3 + 1] = Float32(a.values[b + 1])
        root_pos[f * 3 + 2] = Float32(a.values[b + 2] + AMASS_ROOT_Z)
        root_quat[f * 4 + 0] = Float32(a.values[b + 6])    # w
        root_quat[f * 4 + 1] = Float32(a.values[b + 3])    # x
        root_quat[f * 4 + 2] = Float32(a.values[b + 4])    # y
        root_quat[f * 4 + 3] = Float32(a.values[b + 5])    # z
        for j in range(G1_N_DOF):
            dof[f * G1_N_DOF + j] = Float32(a.values[b + 7 + j])
    return AmassClip(name^, fps, n, root_pos^, root_quat^, dof^)


def convert_amass_clip(
    clip: AmassClip, mut env: UnitreeG1[DType.float64]
) raises -> LafanRows:
    """The clip's 50 Hz store rows — `data/lafan.mojo`'s steps 2–5 verbatim."""
    return convert_frames(
        clip.fps, clip.n, clip.root_pos, clip.root_quat, clip.dof, env
    )
