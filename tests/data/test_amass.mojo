"""Step 1 of the AMASS import — the part no other gate covers.

    pixi run mojo run -I . tests/data/test_amass.mojo

WHY THIS EXISTS
===============
The AMASS importer is two pieces. Steps 2–5 — forward kinematics, the
velocity filters, the 50 Hz resample, the 463-D observation — are
`data/lafan.mojo`'s `convert_frames`, which
`tests/robots/test_lafan_import_vs_oracle.mojo` holds against the
reference's own dump. Step 1 is new: `[N, 36]` float64 → a root pose and 29
joint angles, by a convention stated in one place only, the dataset's own
`g1/visualize.py`.

**EVERY WAY STEP 1 CAN BE WRONG IS SILENT.** Read the quaternion as WXYZ
when it is XYZW and you get a valid unit rotation. Drop the `+ 0.793` and
you get a robot whose pelvis is on the floor, which `privileged[0]`
reports as a plausible permanent crouch. Slice the DoF one element over and
you get a pose. Read a 59 Hz clip as 120 Hz and you get half the duration
at double the velocities. None of those produce a NaN, an exception, a
non-unit quaternion or an out-of-range angle — the store is well-formed and
means something else.

So the gate is written against FIXTURES WHOSE ANSWER IS KNOWN BY
CONSTRUCTION rather than against a clip from the dump: a `.npy` written
here, holding values chosen so that a transposed, mis-sliced or
mis-ordered read lands on a different number. `test_amass_vs_mujoco.mojo`
is the other half — the same convention through MuJoCo on real clips.

The `.npy` reader gets the same treatment. Its refusals are the interesting
part: `fortran_order: True` and a big-endian `descr` are both READABLE as
C-order little-endian, and both read as garbage that has the right shape.
"""

from std.memory import bitcast
from std.os.path import exists

from noeira.data.amass import (
    AMASS_COLS, AMASS_ROOT_Z, amass_frames_of_size, amass_fps_of_name,
    amass_is_clip, amass_subset_of, convert_amass_clip, load_amass_clip,
)
from noeira.envs.robots import UnitreeG1
from noeira.envs.robots.unitree_g1_pd import G1_N_DOF
from noeira.io.fileio import remove_file, write_file_atomic
from noeira.io.npy import load_npy_2d_f64, load_npy_f64, npy_header_of


struct Tally:
    var checks: Int
    var fails: Int

    def __init__(out self):
        self.checks = 0
        self.fails = 0

    def truth(mut self, ok: Bool, msg: String):
        self.checks += 1
        if ok:
            print("  ok:", msg)
        else:
            self.fails += 1
            print("  FAIL:", msg)

    def near(mut self, got: Float64, want: Float64, tol: Float64, msg: String):
        var d = got - want
        if d < 0:
            d = -d
        self.truth(
            d <= tol,
            msg + " (got " + String(got) + ", want " + String(want) + ")",
        )


# ── writing .npy files by hand ──────────────────────────────────────────────
def _f64_bytes(v: Float64) -> List[UInt8]:
    var bits = bitcast[DType.uint64](Float64(v))
    var o = List[UInt8]()
    for i in range(8):
        o.append(UInt8((bits >> UInt64(8 * i)) & UInt64(0xFF)))
    return o^


def _f32_bytes(v: Float32) -> List[UInt8]:
    var bits = bitcast[DType.uint32](Float32(v))
    var o = List[UInt8]()
    for i in range(4):
        o.append(UInt8((bits >> UInt32(8 * i)) & UInt32(0xFF)))
    return o^


def _npy(
    descr: String, shape: String, var payload: List[UInt8],
    version: Int = 1, fortran: Bool = False,
) -> List[UInt8]:
    """A `.npy` whose header numpy itself would write, padded to 64 bytes."""
    var dict = (
        "{'descr': '" + descr + "', 'fortran_order': "
        + (String("True") if fortran else String("False"))
        + ", 'shape': " + shape + ", }"
    )
    var pre = 10 if version == 1 else 12
    var hlen = dict.byte_length() + 1
    while (pre + hlen) % 64 != 0:
        hlen += 1
    var o = List[UInt8]()
    o.append(0x93)
    for c in String("NUMPY").as_bytes():
        o.append(c)
    o.append(UInt8(version))
    o.append(0)
    if version == 1:
        o.append(UInt8(hlen & 0xFF))
        o.append(UInt8((hlen >> 8) & 0xFF))
    else:
        o.append(UInt8(hlen & 0xFF))
        o.append(UInt8((hlen >> 8) & 0xFF))
        o.append(UInt8((hlen >> 16) & 0xFF))
        o.append(UInt8((hlen >> 24) & 0xFF))
    for c in dict.as_bytes():
        o.append(c)
    for _ in range(hlen - dict.byte_length() - 1):
        o.append(32)       # space padding, as numpy pads
    o.append(10)           # the header ends with \n
    for i in range(len(payload)):
        o.append(payload[i])
    return o^


def main() raises:
    var t = Tally()
    var dir = String("/tmp")
    var p = dir + "/noeira_test_amass.npy"

    # ── 1. the reader: a [3, 2] float64 array, values chosen so that a
    #       transposed read lands on a different element ──────────────────
    print("\n[1] the .npy reader")
    var pay = List[UInt8]()
    for k in range(6):
        for b in _f64_bytes(Float64(k) + 0.5):
            pay.append(b)
    write_file_atomic(p, _npy(String("<f8"), String("(3, 2)"), pay.copy()))
    var h = npy_header_of(p)
    t.truth(h.rows == 3 and h.cols == 2, "shape (3, 2) reads as rows 3, cols 2")
    t.truth(h.item_size == 8 and h.is_float, "`<f8` is an 8-byte float")
    t.truth(h.data_offset % 64 == 0, "the data starts 64-byte aligned")
    var v = load_npy_f64(p)
    t.truth(len(v) == 6, "6 elements")
    var row_major = True
    for k in range(6):
        if v[k] != Float64(k) + 0.5:
            row_major = False
    t.truth(row_major, "the elements are in C order, not transposed")

    # v2.0 header: a 4-byte header length, same payload
    write_file_atomic(p, _npy(String("<f8"), String("(3, 2)"), pay.copy(), 2))
    var h2 = npy_header_of(p)
    t.truth(h2.rows == 3 and h2.cols == 2, "a v2.0 header reads the same shape")

    # float32 is widened, not reinterpreted
    var pay32 = List[UInt8]()
    for k in range(4):
        for b in _f32_bytes(Float32(k) - 1.25):
            pay32.append(b)
    write_file_atomic(p, _npy(String("<f4"), String("(2, 2)"), pay32^))
    var v32 = load_npy_f64(p)
    t.near(v32[0], -1.25, 1e-12, "`<f4` is widened to float64")
    t.near(v32[3], 1.75, 1e-12, "`<f4` last element")

    # a 1-D shape has a trailing comma and is not a syntax error
    var pay1 = List[UInt8]()
    for k in range(3):
        for b in _f64_bytes(Float64(k) * 2.0):
            pay1.append(b)
    write_file_atomic(p, _npy(String("<f8"), String("(3,)"), pay1^))
    var h1 = npy_header_of(p)
    t.truth(h1.rows == 3 and h1.cols == 1, "a 1-D `(3,)` shape reads as 3 x 1")

    # ── 2. the refusals. Each of these is READABLE and reads as garbage. ──
    print("\n[2] the reader's refusals")
    write_file_atomic(p, _npy(String("<f8"), String("(3, 2)"), pay.copy(), 1, True))
    var raised = False
    try:
        _ = npy_header_of(p)
    except:
        raised = True
    t.truth(raised, "`fortran_order: True` raises — a C-order read transposes it")

    write_file_atomic(p, _npy(String(">f8"), String("(3, 2)"), pay.copy()))
    raised = False
    try:
        _ = npy_header_of(p)
    except:
        raised = True
    t.truth(raised, "a big-endian `>f8` raises rather than reading byte-swapped")

    write_file_atomic(p, _npy(String("|O"), String("(3, 2)"), pay.copy()))
    raised = False
    try:
        _ = npy_header_of(p)
    except:
        raised = True
    t.truth(raised, "an object dtype `|O` raises (it is a pickle)")

    # a payload one element short: the header promises more than the file holds
    var short = List[UInt8]()
    for i in range(len(pay) - 8):
        short.append(pay[i])
    write_file_atomic(p, _npy(String("<f8"), String("(3, 2)"), short^))
    raised = False
    try:
        _ = npy_header_of(p)
    except:
        raised = True
    t.truth(raised, "a truncated payload raises rather than returning a prefix")

    # the width is the caller's contract
    write_file_atomic(p, _npy(String("<f8"), String("(3, 2)"), pay.copy()))
    raised = False
    try:
        _ = load_npy_2d_f64(p, 36)
    except:
        raised = True
    t.truth(raised, "`load_npy_2d_f64(.., 36)` refuses a [3, 2] array")

    var bad = List[UInt8]()
    for c in String("not a numpy file at all, really not").as_bytes():
        bad.append(c)
    write_file_atomic(p, bad^)
    raised = False
    try:
        _ = npy_header_of(p)
    except:
        raised = True
    t.truth(raised, "a file with the wrong magic raises")

    # ── 3. the fps tag, which is only in the NAME ────────────────────────
    print("\n[3] the fps tag")
    # the six rates the dump actually holds
    t.truth(amass_fps_of_name(String("g1/KIT/3/walk_poses_120_jpos.npy")) == 120, "120")
    t.truth(amass_fps_of_name(String("g1/KIT/3/walk_poses_100_jpos.npy")) == 100, "100")
    t.truth(amass_fps_of_name(String("g1/KIT/3/walk_poses_250_jpos.npy")) == 250, "250")
    t.truth(amass_fps_of_name(String("g1/KIT/3/walk_poses_150_jpos.npy")) == 150, "150")
    t.truth(
        amass_fps_of_name(String("g1/ACCAD/x/B1_poses_60_jpos.npy")) == 60,
        "a two-digit `_60_` is 60, not 0 and not 120",
    )
    t.truth(
        amass_fps_of_name(String("g1/GRAB/x/s1_poses_59_jpos.npy")) == 59,
        "`_59_` is 59 — 13 clips in the dump have it",
    )
    raised = False
    try:
        _ = amass_fps_of_name(String("g1/KIT/3/walk_poses_jpos.npy"))
    except:
        raised = True
    t.truth(raised, "a name with no fps tag raises rather than defaulting to 120")

    # ── 4. duration from the LISTED SIZE, which is what `--plan` budgets on
    print("\n[4] frames from the file size")
    # the real listed size of `g1/ACCAD/Female1General_c3d/A1 - Stand_poses_120_jpos.npy`
    t.truth(
        amass_frames_of_size(103808) == 360,
        "103808 bytes of [N, 36] f8 is 360 frames (A1 - Stand, measured)",
    )
    t.truth(
        amass_frames_of_size(128) == 0 and amass_frames_of_size(0) == 0,
        "a header-only or empty file is 0 frames, not negative",
    )
    t.truth(amass_subset_of(String("g1/KIT/3/x_poses_120_jpos.npy")) == "KIT", "the subset is the dir under g1/")
    t.truth(
        amass_is_clip(String("g1/KIT/license.txt")) == False
        and amass_is_clip(String("g1/KIT/3/x_poses_120_jpos.npy")),
        "`_jpos.npy` separates clips from licences",
    )

    # ── 5. the layout: a clip whose every field is distinguishable ───────
    print("\n[5] the [N, 36] layout")
    # 4 frames. Root position counts 1,2,3; the quaternion is a 90 deg yaw
    # written XYZW so that reading it WXYZ gives a DIFFERENT rotation; the
    # DoF are 100 + j so a slice off by one is visible in every element.
    comptime SQ: Float64 = 0.7071067811865476
    var frames = 4
    var cpay = List[UInt8]()
    for f in range(frames):
        var row = List[Float64](length=AMASS_COLS, fill=0.0)
        row[0] = 1.0 + Float64(f)
        row[1] = 2.0 + Float64(f)
        row[2] = 3.0 + Float64(f)       # relative to the nominal height
        row[3] = 0.0                     # x
        row[4] = 0.0                     # y
        row[5] = SQ                      # z
        row[6] = SQ                      # w  <- LAST, this is XYZW
        for j in range(G1_N_DOF):
            row[7 + j] = 100.0 + Float64(j)
        for k in range(AMASS_COLS):
            for b in _f64_bytes(row[k]):
                cpay.append(b)
    write_file_atomic(p, _npy(String("<f8"), String("(4, 36)"), cpay^))
    var clip = load_amass_clip(p, String("g1/TEST/x_poses_120_jpos.npy"))
    t.truth(clip.n == 4 and clip.fps == 120, "4 frames at 120 fps")
    t.near(Float64(clip.root_pos[0]), 1.0, 1e-6, "root x is column 0")
    t.near(Float64(clip.root_pos[1]), 2.0, 1e-6, "root y is column 1")
    t.near(
        Float64(clip.root_pos[2]), 3.0 + AMASS_ROOT_Z, 1e-6,
        "root z is column 2 PLUS the nominal pelvis height",
    )
    t.near(Float64(clip.root_pos[3 * 3 + 0]), 4.0, 1e-6, "frame 3's root x")
    # the quaternion must come out WXYZ: (w, x, y, z) = (SQ, 0, 0, SQ)
    t.near(Float64(clip.root_quat[0]), SQ, 1e-6, "quat[0] is w — column 6, not column 3")
    t.near(Float64(clip.root_quat[1]), 0.0, 1e-6, "quat[1] is x — column 3")
    t.near(Float64(clip.root_quat[2]), 0.0, 1e-6, "quat[2] is y — column 4")
    t.near(Float64(clip.root_quat[3]), SQ, 1e-6, "quat[3] is z — column 5")
    var dof_ok = True
    for f in range(frames):
        for j in range(G1_N_DOF):
            if Float64(clip.dof[f * G1_N_DOF + j]) != 100.0 + Float64(j):
                dof_ok = False
    t.truth(dof_ok, "all 29 DoF are columns 7..35, in order, every frame")

    # ── 6. a clip too short to convert is REFUSED, not silently emptied ──
    print("\n[6] the short-clip floor")
    var spay = List[UInt8]()
    for _ in range(2 * AMASS_COLS):
        for b in _f64_bytes(0.0):
            spay.append(b)
    write_file_atomic(p, _npy(String("<f8"), String("(2, 36)"), spay^))
    var two = load_amass_clip(p, String("g1/TEST/x_poses_120_jpos.npy"))
    t.truth(two.n == 2, "a 2-frame clip loads")
    var env = UnitreeG1[DType.float64]()
    _ = env.reset()
    raised = False
    try:
        _ = convert_amass_clip(two, env)
    except:
        raised = True
    t.truth(
        raised,
        "converting it RAISES — the reference's velocity tail reads index n - 3"
        " and 68 clips in the dump are under a second",
    )

    if exists(p):
        remove_file(p)
    print("\n===", t.checks - t.fails, "/", t.checks, "passed ===")
    if t.fails != 0:
        raise Error("test_amass: " + String(t.fails) + " failed")
