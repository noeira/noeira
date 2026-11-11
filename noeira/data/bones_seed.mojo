# +--------------------------------------------------------------------------+ #
# | BONES-SEED G1 retargets → 50 Hz reference motions, natively
# +--------------------------------------------------------------------------+ #
"""Step 1 of the BONES-SEED import, and the resample SuperMimic trains on.
M1 of `docs/SUPERMIMIC_PORT_PLAN.md`.

⚠ THE DATA IS CONFIDENTIAL (the SEED licence: no redistribution, private
compute only). Nothing here ships a SEED byte; the gate
(`tests/data/test_bones_seed.mojo`) runs on synthetic CSVs from
`tools/supermimic/make_synthetic_seed.py`, and on a real slice only on
private storage.

THE FILE FORMAT (from the SuperMimic reproduction's `motion_convert.py` and
its experiment log; it follows SONIC's `convert_soma_csv_to_motion_lib.py`):
one CSV per motion in `g1.tar.gz`, 120 fps, a header row, then

    Frame, root_translateX/Y/Z      CENTIMETRES
           root_rotateX/Y/Z         Euler 'xyz' in DEGREES -- scipy's lower
                                    case, i.e. EXTRINSIC: R = Rz · Ry · Rx
           <joint>_dof × 29         DEGREES, in the G1 MJCF joint order

Columns are selected BY NAME, as the reference does, so an unknown column
anywhere is ignored; the joint columns (every name ending `_joint_dof`) must
be exactly the 29 G1 joints in order, or the clip is refused. A permuted
header reads as a valid pose with two joints exchanged, so refusing is the
only safe answer.

TWO CONSUMERS, TWO RESAMPLES. They do not agree and must not be merged:
  * BFM-Zero (D1 on SEED) takes `SeedClip` through `data/lafan.mojo`'s
    `convert_frames` (`data/bones_seed_store.mojo`): BFM-Zero's float32
    motion library, Gaussian-filtered velocities, its slerp artefact;
  * SuperMimic takes `resample_mjlab`: mjlab's `csv_to_npz` conventions as
    the reproduction re-states them (float64; linear blend; slerp with a
    linear branch above dot 0.9995, renormalised; `np.gradient` velocities;
    world angular velocity by a central SO(3) difference, ends repeated).
    The result is the compact library's 71-d row: the pelvis's world pose
    and velocities are the free joint's, so no FK is needed here.

Float64 throughout. `Float64(String)` can be one ULP low (see the gate);
the tolerances there are stated in ULPs for that reason.
"""

from std.math import acos, atan2, cos, floor, sin, sqrt

from noeira.core.bytes import string_from_bytes
from noeira.envs.robots.unitree_g1_pd import G1_N_DOF, g1_dof_name
from noeira.io.fileio import read_file_bytes


comptime SEED_FPS: Float64 = 120.0
comptime SEED_OUT_FPS: Float64 = 50.0
comptime SEED_HEADER_PREFIX = "Frame,root_translateX"
comptime SEED_JOINT_SUFFIX = "_joint_dof"
comptime SEED_REF_DIM: Int = 71    # 3 pos + 4 quat + 29 q + 3 v + 3 ω + 29 qd
comptime SEED_RAW_DIM: Int = 36    # 3 pos + 4 quat (WXYZ) + 29 q
comptime _SLERP_LINEAR_DOT: Float64 = 0.9995
comptime _DEG2RAD: Float64 = 3.141592653589793 / 180.0


struct SeedClip(Movable):
    """One SEED CSV in SI units: `raw` is `n × 36` float64 — root position
    (m), root quaternion WXYZ (scipy's sign, not canonicalised), 29 joint
    angles (rad)."""
    var name: String
    var fps: Float64
    var n: Int
    var raw: List[Float64]

    def __init__(out self, var name: String, fps: Float64, n: Int, var raw: List[Float64]):
        self.name = name^
        self.fps = fps
        self.n = n
        self.raw = raw^

    def __init__(out self, *, deinit move: Self):
        self.name = move.name^
        self.fps = move.fps
        self.n = move.n
        self.raw = move.raw^


struct MimicMotion(Movable):
    """A clip resampled to `fps`: `frames` is `n × 71` float64, the compact
    library row `[root pos, root quat WXYZ, q, root lin vel (world),
    root ang vel (world), qd]`."""
    var fps: Float64
    var n: Int
    var frames: List[Float64]

    def __init__(out self, fps: Float64, n: Int, var frames: List[Float64]):
        self.fps = fps
        self.n = n
        self.frames = frames^

    def __init__(out self, *, deinit move: Self):
        self.fps = move.fps
        self.n = move.n
        self.frames = move.frames^


# ── quaternions (WXYZ) ──────────────────────────────────────────────────────
@always_inline
def _qmul(
    aw: Float64, ax: Float64, ay: Float64, az: Float64,
    bw: Float64, bx: Float64, by: Float64, bz: Float64,
) -> SIMD[DType.float64, 4]:
    """Hamilton product a ⊗ b, WXYZ in and out."""
    return SIMD[DType.float64, 4](
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def euler_xyz_deg_to_quat_wxyz(
    x_deg: Float64, y_deg: Float64, z_deg: Float64
) -> SIMD[DType.float64, 4]:
    """scipy's `Rotation.from_euler("xyz", [x, y, z], degrees=True)`, WXYZ.

    Lower-case 'xyz' is EXTRINSIC: rotate about the fixed x, then the fixed
    y, then the fixed z, so q = q_z ⊗ q_y ⊗ q_x. scipy composes the
    elementary quaternions in that order and does NOT flip w to be
    non-negative; neither does this (the sign reaches the store's root
    quaternion, so it is matched, not normalised away).
    """
    var hx = x_deg * _DEG2RAD * 0.5
    var hy = y_deg * _DEG2RAD * 0.5
    var hz = z_deg * _DEG2RAD * 0.5
    var qyx = _qmul(cos(hy), 0.0, sin(hy), 0.0, cos(hx), sin(hx), 0.0, 0.0)
    return _qmul(cos(hz), 0.0, 0.0, sin(hz), qyx[0], qyx[1], qyx[2], qyx[3])


# ── the CSV ─────────────────────────────────────────────────────────────────
def _split_fields(line: StringSlice) -> List[String]:
    var out = List[String]()
    for f in line.split(","):
        out.append(String(String(f).strip()))
    return out^


def _find(cols: List[String], name: String) raises -> Int:
    for i in range(len(cols)):
        if cols[i] == name:
            return i
    raise Error("bones_seed: no column '" + name + "'")


def parse_seed_csv(text: String, var name: String, fps: Float64 = SEED_FPS) raises -> SeedClip:
    """One SEED CSV's text → `SeedClip`. Raises on a missing column, a joint
    header that is not the 29 G1 joints in order, or a ragged row."""
    var lines = text.split("\n")
    if len(lines) == 0 or not String(lines[0]).startswith(SEED_HEADER_PREFIX):
        raise Error("bones_seed: '" + name + "' does not start with the SEED header")
    var cols = _split_fields(lines[0])
    var c_pos = [
        _find(cols, "root_translateX"), _find(cols, "root_translateY"),
        _find(cols, "root_translateZ"),
    ]
    var c_rot = [
        _find(cols, "root_rotateX"), _find(cols, "root_rotateY"),
        _find(cols, "root_rotateZ"),
    ]
    var c_dof = List[Int]()
    for i in range(len(cols)):
        if cols[i].endswith(SEED_JOINT_SUFFIX):
            c_dof.append(i)
    if len(c_dof) != G1_N_DOF:
        raise Error(
            "bones_seed: '" + name + "' has " + String(len(c_dof))
            + " joint columns, the G1 has " + String(G1_N_DOF)
        )
    for j in range(G1_N_DOF):
        var want = String(g1_dof_name(j)) + "_dof"
        if cols[c_dof[j]] != want:
            raise Error(
                "bones_seed: '" + name + "' joint column " + String(j) + " is '"
                + cols[c_dof[j]] + "', the G1 order wants '" + want + "'"
            )

    var raw = List[Float64]()
    var n = 0
    for li in range(1, len(lines)):
        var s = String(lines[li]).strip()
        if s.byte_length() == 0:
            continue
        var f = _split_fields(s)
        if len(f) != len(cols):
            raise Error(
                "bones_seed: '" + name + "' row " + String(li) + " has "
                + String(len(f)) + " fields, the header " + String(len(cols))
            )
        for k in range(3):
            raw.append(Float64(f[c_pos[k]]) / 100.0)
        var q = euler_xyz_deg_to_quat_wxyz(
            Float64(f[c_rot[0]]), Float64(f[c_rot[1]]), Float64(f[c_rot[2]])
        )
        for k in range(4):
            raw.append(q[k])
        for j in range(G1_N_DOF):
            raw.append(Float64(f[c_dof[j]]) * _DEG2RAD)
        n += 1
    if n == 0:
        raise Error("bones_seed: '" + name + "' has no frames")
    return SeedClip(name^, fps, n, raw^)


def load_seed_csv(path: String, var name: String, fps: Float64 = SEED_FPS) raises -> SeedClip:
    return parse_seed_csv(string_from_bytes(read_file_bytes(path)), name^, fps)


# ── the name filter (SONIC's keywords + the paper's) ────────────────────────
comptime _N_KEYWORDS: Int = 47


def _keyword(i: Int) -> StaticString:
    """`filter_bones_seed.KEYWORDS` of the reproduction, IN ITS SORTED ORDER:
    the first match names the reason, so the order is part of the answer.
    Substring matches on the lower-cased name, traps included (`sit_` drops
    `deposit_object`, `chair` drops `chairless_squat`) -- that is the filter
    the paper's counts were matched with."""
    var k: List[StaticString] = [
        "230m", "50cm_box_", "_bed_", "_box_", "_ride_", "_sit_", "_stair",
        "acrobatics_", "bed", "bike", "box_", "box_dips", "box_hspu",
        "box_jump", "cartwheel", "chair", "climb", "com_up_50cm", "crutch",
        "fall_from", "fence", "form_box", "handstand", "handstand_ff_",
        "jump_off_50cm", "jump_over_obstacle_", "jump_to_shoulder_roll",
        "kozak_dance", "ladder", "lift_crate_come_up_", "look_over",
        "monkey_jump", "off_1m", "off_50cm", "on_1m", "on_box",
        "push_obstacle", "safety_roll", "scooter", "seat", "sit_", "sitting",
        "stair", "step_on", "stepdown", "table", "walking_on_edge",
    ]
    return k[i]


def seed_name_filter(name: String) -> String:
    """"" to keep the clip, else `name:<keyword>` (the reproduction's reason)."""
    var n = name.lower()
    for i in range(_N_KEYWORDS):
        var k = String(_keyword(i))
        if k in n:
            return "name:" + k
    return String("")


# ── mjlab's resample (SuperMimic's reference motions) ───────────────────────
@always_inline
def _norm4(q: SIMD[DType.float64, 4]) -> Float64:
    return sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])


def _rotvec_wxyz(q_in: SIMD[DType.float64, 4]) -> SIMD[DType.float64, 4]:
    """`rotvec_from_quat`: angle·axis of a WXYZ quaternion (lane 3 unused)."""
    var q = q_in / _norm4(q_in)
    if q[0] < 0.0:
        q = -q
    var s = sqrt(q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    var angle = 2.0 * atan2(s, q[0])
    var scale = angle / s if s > 1e-9 else 2.0
    return SIMD[DType.float64, 4](q[1] * scale, q[2] * scale, q[3] * scale, 0.0)


def resample_mjlab(clip: SeedClip, out_fps: Float64 = SEED_OUT_FPS) raises -> MimicMotion:
    """`motion_convert.resample_motion` of the reproduction, on a SEED G1 clip."""
    var out = resample_mjlab_rows(clip.raw, clip.n, G1_N_DOF, clip.fps, out_fps)
    return MimicMotion(out_fps, out[0], out[1].copy())


def resample_mjlab_rows(
    raw: List[Float64], n_in: Int, ndof: Int, in_fps: Float64, out_fps: Float64 = SEED_OUT_FPS,
) raises -> Tuple[Int, List[Float64]]:
    """`motion_convert.resample_motion` of the reproduction, line for line,
    for any hinge count: `raw` is `n_in × (7 + ndof)` (root position, root
    quaternion WXYZ, joint angles); the result is `(n_out, n_out × (13 + 2·ndof))`
    rows `[root pos, root quat WXYZ, q, root lin vel (world), root ang vel
    (world), qd]`."""
    var input_dt = 1.0 / in_fps
    var output_dt = 1.0 / out_fps
    var duration = Float64(n_in - 1) * input_dt
    var n_out = Int(floor(duration / output_dt + 1e-6)) + 1
    var RAW = 7 + ndof
    var D = 13 + 2 * ndof
    var CV = 7 + ndof          # root linear velocity
    var CW = 10 + ndof         # root angular velocity
    var CQD = 13 + ndof        # joint velocities
    var frames = List[Float64](length=n_out * D, fill=0.0)

    for i in range(n_out):
        var phase = (Float64(i) * output_dt) / duration if duration > 0 else 0.0
        var x = phase * Float64(n_in - 1)
        var i0 = Int(floor(x))
        if i0 < 0:
            i0 = 0
        if i0 > n_in - 1:
            i0 = n_in - 1
        var i1 = min(i0 + 1, n_in - 1)
        var b = x - Float64(i0)
        var r0 = i0 * RAW
        var r1 = i1 * RAW
        var o = i * D
        for k in range(3):
            frames[o + k] = raw[r0 + k] * (1.0 - b) + raw[r1 + k] * b
        for j in range(ndof):
            frames[o + 7 + j] = raw[r0 + 7 + j] * (1.0 - b) + raw[r1 + 7 + j] * b
        var q0 = SIMD[DType.float64, 4](
            raw[r0 + 3], raw[r0 + 4], raw[r0 + 5], raw[r0 + 6]
        )
        var q1 = SIMD[DType.float64, 4](
            raw[r1 + 3], raw[r1 + 4], raw[r1 + 5], raw[r1 + 6]
        )
        q0 = q0 / _norm4(q0)
        q1 = q1 / _norm4(q1)
        var dot = (q0 * q1).reduce_add()
        if dot < 0.0:
            q1 = -q1
            dot = -dot
        var theta = acos(min(max(dot, -1.0), 1.0))
        var w0: Float64
        var w1: Float64
        if dot > _SLERP_LINEAR_DOT:
            w0 = 1.0 - b
            w1 = b
        else:
            var st = sin(theta)
            w0 = sin((1.0 - b) * theta) / st
            w1 = sin(b * theta) / st
        var q = w0 * q0 + w1 * q1
        q = q / _norm4(q)
        for k in range(4):
            frames[o + 3 + k] = q[k]

    # Velocities: `np.gradient` (central inside, one-sided at the ends) on
    # the root position and the joints, a central SO(3) difference on the
    # root quaternion with its first and last samples repeated.
    if n_out >= 2:
        for i in range(n_out):
            var lo = i - 1 if i > 0 else 0
            var hi = i + 1 if i < n_out - 1 else n_out - 1
            var den = output_dt * Float64(hi - lo)
            for k in range(3):
                frames[i * D + CV + k] = (frames[hi * D + k] - frames[lo * D + k]) / den
            for j in range(ndof):
                frames[i * D + CQD + j] = (frames[hi * D + 7 + j] - frames[lo * D + 7 + j]) / den
    if n_out >= 3:
        for i in range(1, n_out - 1):
            var a = (i + 1) * D + 3
            var c = (i - 1) * D + 3
            var rel = _qmul(
                frames[a], frames[a + 1], frames[a + 2], frames[a + 3],
                frames[c], -frames[c + 1], -frames[c + 2], -frames[c + 3],
            )
            var w = _rotvec_wxyz(rel) / (2.0 * output_dt)
            for k in range(3):
                frames[i * D + CW + k] = w[k]
        for k in range(3):
            frames[CW + k] = frames[D + CW + k]
            frames[(n_out - 1) * D + CW + k] = frames[(n_out - 2) * D + CW + k]
    return (n_out, frames^)
