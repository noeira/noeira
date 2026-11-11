# +--------------------------------------------------------------------------+ #
# | Retargeted robot trajectories (hhtools CSV) → SuperMimic's motion library
# +--------------------------------------------------------------------------+ #
"""A robot trajectory as RoboParty's hhtools exports it, for any robot, into
the 50 Hz reference rows SuperMimic trains on and a `TrajectoryStore` of them.

THE FILE (hhtools 0.1.6, `hhtools/io/robot_csv.py`): comment lines `# key:
value`, then a header row and one row per frame,

    time, root_x, root_y, root_z          METRES, world, Z up
          root_qx, root_qy, root_qz, root_qw   the root's quaternion, XYZW
          dof_<joint name> × n            RADIANS

Among the comments, `sample_rate` (frames per second) and
`playback_mesh_z_lift`.

Columns are selected BY NAME: the joints wanted are the robot's, in the
robot's order, each as `dof_<name>`; a file that lacks one, or carries a
`dof_` column the robot does not have, is refused. The rows come out in the
robot's joint order whatever the file's.

⚠ `playback_mesh_z_lift` IS TAKEN BACK OUT of the root height by default.
hhtools grounds a clip twice: its solver shifts the clip so that the lowest
sole over the clip is on the floor, then its export adds this constant so
that the lowest point of the FIRST frame is. Where the first frame's feet are
above the clip's lowest sole, the export's shift pushes later frames under
the floor (measured on 85 BONES-SEED clips retargeted to Berkeley Humanoid
Lite: soles up to 28 mm under, against 3.8 mm with the shift removed; the
standing feet then float by about 10 mm).

THE RESAMPLE is `data/bones_seed.resample_mjlab_rows`, SuperMimic's: linear
on positions and joints, slerp on the root, `np.gradient` velocities.

THE STORE has `bones_seed_import`'s shape, so the same loaders read it:
    ref        float32 (N_rows, 13 + 2n)  [root pos, root quat WXYZ, q, root
                                          lin vel, root ang vel, qd]
    motion_id  int32   (N_rows, 1)
    fps        float64 (1,)               50
and one episode per kept clip, its name in the task table.

NO FEASIBILITY FILTER: the SEED filter (`bones_seed_filter`) is written on
the G1's bodies. A clip is refused here only when it cannot be read, has a
non-finite value, or is shorter than `min_frames` at 50 Hz.

⚠ A CLIP RETARGETED FROM A LICENSED DATASET keeps that licence: BONES-SEED
retargets stay on private storage, like the SEED stores.
"""

from std.os.path import exists

from noeira.core.bytes import string_from_bytes
from noeira.data.bones_seed import SEED_OUT_FPS, resample_mjlab_rows
from noeira.data.column import ColumnSpec
from noeira.data.store import TrajectoryStoreWriter
from noeira.io.fileio import read_file_bytes, remove_file, rename_over, write_text_atomic


comptime RETARGET_REF_COLUMN = "ref"
comptime RETARGET_MOTION_ID_COLUMN = "motion_id"
comptime RETARGET_DOF_PREFIX = "dof_"


struct RetargetClip(Movable):
    """One trajectory in the robot's joint order: `raw` is `n × (7 + ndof)`
    float64 — root position (m), root quaternion WXYZ, joint angles (rad).
    `z_lift` is the file's `playback_mesh_z_lift` (0 when absent), `lift_removed`
    whether `raw`'s root height has it taken out."""
    var name: String
    var fps: Float64
    var n: Int
    var ndof: Int
    var z_lift: Float64
    var lift_removed: Bool
    var raw: List[Float64]

    def __init__(
        out self, var name: String, fps: Float64, n: Int, ndof: Int, z_lift: Float64, lift_removed: Bool,
        var raw: List[Float64],
    ):
        self.name = name^
        self.fps = fps
        self.n = n
        self.ndof = ndof
        self.z_lift = z_lift
        self.lift_removed = lift_removed
        self.raw = raw^

    def __init__(out self, *, deinit move: Self):
        self.name = move.name^
        self.fps = move.fps
        self.n = move.n
        self.ndof = move.ndof
        self.z_lift = move.z_lift
        self.lift_removed = move.lift_removed
        self.raw = move.raw^


def _fields(line: StringSlice) -> List[String]:
    var out = List[String]()
    for f in line.split(","):
        out.append(String(String(f).strip()))
    return out^


def _col(cols: List[String], name: String, clip: String) raises -> Int:
    for i in range(len(cols)):
        if cols[i] == name:
            return i
    raise Error("retarget_csv: '" + clip + "' has no column '" + name + "'")


def parse_retarget_csv(
    text: String, var name: String, joint_names: List[String], remove_lift: Bool = True,
) raises -> RetargetClip:
    """One hhtools CSV's text → `RetargetClip` in `joint_names`' order."""
    var ndof = len(joint_names)
    var fps = 0.0
    var z_lift = 0.0
    var cols = List[String]()
    var c_pos = List[Int]()
    var c_quat = List[Int]()
    var c_dof = List[Int]()
    var raw = List[Float64]()
    var n = 0
    var row_w = 7 + ndof
    for line in text.split("\n"):
        var s = String(String(line).strip())
        if s.byte_length() == 0:
            continue
        if s.startswith("#"):
            var colon = s.find(":")
            if colon > 0:
                var key = String(String(s[byte = 1 : colon]).strip())
                var value = String(String(s[byte = colon + 1 :]).strip())
                if key == "sample_rate":
                    fps = Float64(value)
                elif key == "playback_mesh_z_lift":
                    z_lift = Float64(value)
            continue
        var f = _fields(s)
        if len(cols) == 0:
            cols = f^
            c_pos = [_col(cols, "root_x", name), _col(cols, "root_y", name), _col(cols, "root_z", name)]
            # the store's quaternion is WXYZ
            c_quat = [
                _col(cols, "root_qw", name), _col(cols, "root_qx", name), _col(cols, "root_qy", name),
                _col(cols, "root_qz", name),
            ]
            for j in range(ndof):
                c_dof.append(_col(cols, String(RETARGET_DOF_PREFIX) + joint_names[j], name))
            var have = 0
            for i in range(len(cols)):
                if cols[i].startswith(RETARGET_DOF_PREFIX):
                    have += 1
            if have != ndof:
                raise Error(
                    "retarget_csv: '" + name + "' has " + String(have) + " joint columns, the robot has "
                    + String(ndof)
                )
            continue
        if len(f) != len(cols):
            raise Error(
                "retarget_csv: '" + name + "' frame " + String(n) + " has " + String(len(f)) + " fields, the header "
                + String(len(cols))
            )
        for k in range(3):
            raw.append(Float64(f[c_pos[k]]))
        for k in range(4):
            raw.append(Float64(f[c_quat[k]]))
        for j in range(ndof):
            raw.append(Float64(f[c_dof[j]]))
        n += 1
    if len(cols) == 0:
        raise Error("retarget_csv: '" + name + "' has no header row")
    if n == 0:
        raise Error("retarget_csv: '" + name + "' has no frames")
    if fps <= 0.0:
        raise Error("retarget_csv: '" + name + "' has no `# sample_rate`")
    for i in range(n * row_w):
        var v = raw[i]
        if v != v or v > 1e9 or v < -1e9:
            raise Error("retarget_csv: '" + name + "' has a non-finite value in frame " + String(i // row_w))
    if remove_lift:
        for i in range(n):
            raw[i * row_w + 2] -= z_lift
    return RetargetClip(name^, fps, n, ndof, z_lift, remove_lift, raw^)


def load_retarget_csv(
    path: String, var name: String, joint_names: List[String], remove_lift: Bool = True,
) raises -> RetargetClip:
    return parse_retarget_csv(string_from_bytes(read_file_bytes(path)), name^, joint_names, remove_lift)


struct RetargetImportSummary(Movable):
    var clips: Int
    var kept: Int
    var refused: Int
    var too_short: Int
    var rows_kept: Int
    var lift_min: Float64
    var lift_max: Float64

    def __init__(out self):
        self.clips = 0
        self.kept = 0
        self.refused = 0
        self.too_short = 0
        self.rows_kept = 0
        self.lift_min = 0.0
        self.lift_max = 0.0

    def __init__(out self, *, deinit move: Self):
        self.clips = move.clips
        self.kept = move.kept
        self.refused = move.refused
        self.too_short = move.too_short
        self.rows_kept = move.rows_kept
        self.lift_min = move.lift_min
        self.lift_max = move.lift_max


def retarget_clip_name(path: String) -> String:
    """The file's stem."""
    var slash = path.rfind("/")
    var base = String(path[byte = slash + 1 :]) if slash >= 0 else path.copy()
    if base.endswith(".csv"):
        return String(base[byte = 0 : base.byte_length() - 4])
    return base^


def import_retarget_clips(
    paths: List[String], joint_names: List[String], out_path: String, manifest_path: String, env_id: String,
    remove_lift: Bool = True, min_frames: Int = 25, verbose: Bool = True,
) raises -> RetargetImportSummary:
    """Every CSV of `paths`, in that order: read, resample to 50 Hz, append to
    the store at `out_path` (written to `<out_path>.part` and renamed at the
    end). The manifest (`name,frames,keep,reason,z_lift`) lists every input."""
    var s = RetargetImportSummary()
    var ndof = len(joint_names)
    var width = 13 + 2 * ndof
    var part = out_path + ".part"
    if exists(part):
        remove_file(part)
    var cols = List[ColumnSpec]()
    cols.append(ColumnSpec(String(RETARGET_REF_COLUMN), DType.float32, width))
    cols.append(ColumnSpec(String(RETARGET_MOTION_ID_COLUMN), DType.int32, 1))
    var w = TrajectoryStoreWriter(part, cols^, env_id.copy(), 0, String(""))
    var man = String("name,frames,keep,reason,z_lift\n")
    for ci in range(len(paths)):
        var name = retarget_clip_name(paths[ci])
        s.clips += 1
        var frames: Int
        var lift: Float64
        var rows64: List[Float64]
        try:
            var clip = load_retarget_csv(paths[ci], name.copy(), joint_names, remove_lift)
            lift = clip.z_lift
            var out = resample_mjlab_rows(clip.raw, clip.n, ndof, clip.fps)
            frames = out[0]
            rows64 = out[1].copy()
        except e:
            if verbose:
                print("  [refused] " + name + ": " + String(e))
            s.refused += 1
            man += name + ",0,0,refused,0\n"
            continue
        if frames < min_frames:
            s.too_short += 1
            man += name + "," + String(frames) + ",0,too_short," + String(lift) + "\n"
            continue
        var rows = List[Float32](length=frames * width, fill=Float32(0))
        for i in range(frames * width):
            rows[i] = Float32(rows64[i])
        var ids = List[Int32](length=frames, fill=Int32(s.kept))
        w.append[DType.float32](
            String(RETARGET_REF_COLUMN),
            rows.unsafe_ptr().unsafe_bitcast[Scalar[DType.float32]]().as_unsafe_any_origin(),
            frames,
        )
        w.append[DType.int32](
            String(RETARGET_MOTION_ID_COLUMN),
            ids.unsafe_ptr().unsafe_bitcast[Scalar[DType.int32]]().as_unsafe_any_origin(),
            frames,
        )
        w.end_episode()
        w.add_task(s.kept, name.copy())
        if s.kept == 0 or lift < s.lift_min:
            s.lift_min = lift
        if s.kept == 0 or lift > s.lift_max:
            s.lift_max = lift
        s.kept += 1
        s.rows_kept += frames
        man += name + "," + String(frames) + ",1,," + String(lift) + "\n"
        if verbose and ((ci + 1) % 500 == 0 or ci == len(paths) - 1):
            print(
                "  [" + String(ci + 1) + "/" + String(len(paths)) + "] kept " + String(s.kept) + ", "
                + String(Float64(s.rows_kept) / SEED_OUT_FPS / 3600.0) + " h"
            )
    if s.kept == 0:
        raise Error("retarget_csv: no clip could be imported; no store written")
    var fps = List[Float64](length=1, fill=SEED_OUT_FPS)
    w.write_vector[DType.float64](
        String("fps"), fps.unsafe_ptr().unsafe_bitcast[Scalar[DType.float64]]().as_unsafe_any_origin(), 1
    )
    w.close()
    rename_over(part, out_path.copy())
    write_text_atomic(manifest_path.copy(), man)
    return s^
