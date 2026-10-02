"""LAFAN1 for the G1 (`lafan_29dof.pkl`) → `TrajectoryStore` rows, natively —
the reference's motion library transcribed, no Python. G1b of
`docs/BFM_ZERO_G1_REPRODUCTION.md` §13.

What `tools/g1/lafan_reference_dump.py` + `lafan_to_store.py` produce by
running BFM-Zero's own code (`MotionLibRobot`, `Humanoid_Batch.fk_batch`,
`get_motion_state`, `compute_humanoid_observations_max`), this module
produces from the pickle alone, and `tests/robots/test_lafan_import_vs_oracle.mojo`
holds the two stores against each other column by column. The Python
path is kept as the ORACLE, the way `tools/act/lerobot_v3_to_store.py` is
for the LeRobot importer.

THE PIPELINE, per clip (float32 throughout, as the reference; the ops are
written in the reference's order so float32 rounding lands the same way).
Step 1 is LAFAN-specific and lives in `convert_clip`; steps 2–5 are
`convert_frames`, which `data/amass.mojo` calls with its own step 1:

  1. frames at the clip's own fps (30): root quaternion from the root
     axis-angle by `axis_angle_to_quaternion` (pytorch3d's, wxyz; the small-
     angle branch below 1e-6), joint angle j = sum of `pose_aa[f, 1+j, :]`
     (`dof_pos = pose.sum(-1)`), root position `root_trans_offset` as is
     (`fix_height` is `no_fix`, `target_heading` None, `max_len` −1);
  2. forward kinematics of the 30 skeleton bodies through OUR engine
     (`UnitreeG1.set_state`, equal to the reference's torch FK to 5e-7,
     G1), the virtual head from the torso (`+0.35 m` along its z, the
     torso's rotation), every world quaternion re-signed by the
     reference's `matrix_to_quaternion` rule (the largest-magnitude
     component positive — pytorch3d's candidate selection);
  3. velocities at 30 fps: `np.gradient` along frames in float32 (central
     interior, one-sided ends, `/ dt` last) then scipy's
     `gaussian_filter1d(σ=2, mode='nearest')` — radius 8, float64 weights
     and accumulation, cast to float32 (bit-exact against scipy, probed);
     angular velocity from `quat_mul_norm(q[t+1], conj(q[t]))` →
     `quat_angle_axis` (angle = arccos(clamp(2w² − 1)), axis normalised
     with floor 1e-9) × angle / dt, identity (zero) at the last frame, the
     same filter; joint velocities by forward differences with the last
     row duplicated;
  4. resampling to 50 Hz (`get_motion_state` at `arange(ceil(len / 0.02)) ·
     0.02`, `len = (n − 1) / fps`): `_calc_frame_blend` in float32 — phase
     = t / len clipped, idx0 = trunc(phase · (n − 1)), blend = clip((t −
     idx0 · dt) / dt); positions, velocities, joint angles and velocities
     blended linearly `(1 − b)·x0 + b·x1`; every body quaternion by the
     reference's `slerp`, which returns the UNNORMALISED MIDPOINT below
     `sin(half-angle) < 1e-3` for any blend and a non-unit result just
     above it (§10) — reproduced, not fixed;
  5. the two observations per row: `state` = [dof − default, dof_vel,
     `quat_rotate_inverse(root, (0,0,−1))` (their `a − b + c`), root
     angular velocity, world frame, unscaled]; `privileged` =
     `compute_humanoid_observations_max` — heading by `calc_heading`
     (`my_quat_rotate(q, x)` → atan2) and `quat_from_angle_axis(−heading, z)`
     (normalised), local positions with the root dropped, `quat_mul` (the
     reference's own arrangement of the product) → tangent-normal, local
     velocities, all through `my_quat_rotate`.

Columns are those of `lafan_to_store.py`: qpos 36 (root pos, root quat
WXYZ, dof), qvel 35 (root vel and angular velocity WORLD frame, dof vel),
state 64, privileged 463, body_pos / body_quat (XYZW) / body_vel /
body_ang_vel for 31 bodies, motion_id. Side tables `default_dof_pos` and
`env_dt` through `write_vector`; the clip names as the store's task table
(`task=<clip>\t"<name>"` in the manifest), which the Python consumers read
when there is no `motion_key` dataset.

⚠ WHAT DIFFERS FROM THE ORACLE, BY CONSTRUCTION (measured on all 40 clips,
gate on two). The reference's world rotations come from a chain of float32
3×3 matrix products; ours from the engine's float64 kinematics cast to
float32 — the same rotation to ~1e-7, so body positions agree to 2.4e-6 and
linear velocities to 3e-5. The root pose, the joint angles and their
velocities are bit-exact (they never pass through a rotation chain; the
root quaternion takes the reference's own float32 path). That 1e-7 decides
two branches of the reference's code for a small fraction of body-frames:
the slerp's midpoint threshold (5.6e-4 apart on up to 3 % of the body
quaternions of a clip, 0 sign flips) and the arccos of `_compute_angular_
velocity` near a unit w (1.6e-2 on up to 23 % of the angular-velocity
elements of a fast clip; the reference's own function moves that much
under 1e-7 quaternion noise). The observations inherit the second. The
gate bounds every column range by its mechanism and counts the extent.

⚠ EVERY PRODUCT THAT FEEDS AN ADDITION GOES THROUGH `_m`. numpy and torch
round `a*b` before the add; LLVM fuses Mojo's `a*b + c` into one rounding,
and `Float32(Float64(a)*Float64(b))` is shrunk back and fused too. Only a
call boundary keeps the two roundings — `_m` is `@no_inline`. Without it
the blend of a 256-second time was off by 4.5e-4 (docs §13).
"""

from std.math import sqrt, exp, atan2, sin, cos, acos

from noeira.nn.constants import DT
from noeira.io.pickle import JoblibPickle
from noeira.envs.robots import UnitreeG1
from noeira.envs.robots.unitree_g1_xml import UnitreeG1Model, TORSO_BODY_IDX
from noeira.envs.robots.unitree_g1_pd import G1_N_DOF, g1_default_pos
from noeira.envs.robots.unitree_g1_priv_obs import g1_skeleton_body, G1_HEAD_OFFSET_Z


comptime LAFAN_N_BODIES: Int = 31        # 30 skeleton + head
comptime LAFAN_N_SKEL: Int = 30
comptime LAFAN_ENV_DT: Float64 = 0.02
comptime LAFAN_STATE_DIM: Int = 64
comptime LAFAN_PRIV_DIM: Int = 463
comptime LAFAN_NQ: Int = 36
comptime LAFAN_NV: Int = 35
comptime GAUSS_SIGMA: Float64 = 2.0
comptime GAUSS_RADIUS: Int = 8            # int(4 * sigma + 0.5)
comptime SLERP_MIDPOINT_TOL: Float32 = 1e-3


# ── float32 quaternion helpers, the reference's formulas ────────────────────
# Quaternions are (x, y, z, w) unless a name says wxyz.

@always_inline
def _f32(x: Float64) -> Float32:
    return Float32(x)


@no_inline
def _m(a: Float32, b: Float32) -> Float32:
    """The IEEE float32 product, rounded ONCE — numpy's and torch's `a * b`.

    Every product that feeds an addition in the transcribed formulas goes
    through here. numpy materialises each op, so its `a*b + c` is two
    roundings; LLVM contracts the same expression in Mojo into one fused
    multiply-add. On `(time − idx0·dt)/dt` the ulp of a 256-second time
    that the fusion keeps became 4.5e-4 of blend and 3.6e-5 of position
    (measured at walk1_subject1 row 12808). `Float32(Float64(a) *
    Float64(b))` does NOT help: instcombine shrinks it back to a float
    multiply and the backend fuses that. The call boundary is what stops
    the contraction — hence `@no_inline`, and the cost is a call per
    product in a conversion that runs once.
    """
    return a * b


def aa_to_quat_wxyz(ax: Float32, ay: Float32, az: Float32) -> Array[Float32, 4]:
    """PyTorch3D `axis_angle_to_quaternion` in float32: (cos(θ/2), axis·sin(θ/2)/θ),
    with the small-angle series below 1e-6."""
    var angle = sqrt((_m(ax, ax) + _m(ay, ay)) + _m(az, az))
    var half = angle * Float32(0.5)
    var s: Float32
    if angle < Float32(1e-6):
        s = Float32(0.5) - _m(angle, angle) / Float32(48.0)
    else:
        s = sin(half) / angle
    var out = Array[Float32, 4](fill=Float32(0))
    out[0] = cos(half)
    out[1] = ax * s
    out[2] = ay * s
    out[3] = az * s
    return out^


def quat_to_matrix_wxyz(r: Float32, i: Float32, j: Float32, k: Float32) -> Array[Float32, 9]:
    """PyTorch3D `quaternion_to_matrix` (row-major 3×3)."""
    var two_s = Float32(2.0) / (((_m(r, r) + _m(i, i)) + _m(j, j)) + _m(k, k))
    var m = Array[Float32, 9](fill=Float32(0))
    m[0] = Float32(1) - _m(two_s, _m(j, j) + _m(k, k))
    m[1] = _m(two_s, _m(i, j) - _m(k, r))
    m[2] = _m(two_s, _m(i, k) + _m(j, r))
    m[3] = _m(two_s, _m(i, j) + _m(k, r))
    m[4] = Float32(1) - _m(two_s, _m(i, i) + _m(k, k))
    m[5] = _m(two_s, _m(j, k) - _m(i, r))
    m[6] = _m(two_s, _m(i, k) - _m(j, r))
    m[7] = _m(two_s, _m(j, k) + _m(i, r))
    m[8] = Float32(1) - _m(two_s, _m(i, i) + _m(j, j))
    return m^


def _sqrt_pos(x: Float32) -> Float32:
    if x > Float32(0):
        return sqrt(x)
    return Float32(0)


def matrix_to_quat_wxyz(m: Array[Float32, 9]) -> Array[Float32, 4]:
    """PyTorch3D `matrix_to_quaternion`: four candidates, the one whose
    diagonal magnitude is largest, divided by `2·max(q_abs, 0.1)`."""
    var m00 = m[0]
    var m01 = m[1]
    var m02 = m[2]
    var m10 = m[3]
    var m11 = m[4]
    var m12 = m[5]
    var m20 = m[6]
    var m21 = m[7]
    var m22 = m[8]
    var qa = Array[Float32, 4](fill=Float32(0))
    qa[0] = _sqrt_pos(Float32(1) + m00 + m11 + m22)
    qa[1] = _sqrt_pos(Float32(1) + m00 - m11 - m22)
    qa[2] = _sqrt_pos(Float32(1) - m00 + m11 - m22)
    qa[3] = _sqrt_pos(Float32(1) - m00 - m11 + m22)
    var best = 0
    for c in range(1, 4):
        if qa[c] > qa[best]:
            best = c
    var cand = Array[Float32, 4](fill=Float32(0))
    if best == 0:
        cand[0] = qa[0] * qa[0]
        cand[1] = m21 - m12
        cand[2] = m02 - m20
        cand[3] = m10 - m01
    elif best == 1:
        cand[0] = m21 - m12
        cand[1] = qa[1] * qa[1]
        cand[2] = m10 + m01
        cand[3] = m02 + m20
    elif best == 2:
        cand[0] = m02 - m20
        cand[1] = m10 + m01
        cand[2] = qa[2] * qa[2]
        cand[3] = m12 + m21
    else:
        cand[0] = m10 - m01
        cand[1] = m20 + m02
        cand[2] = m21 + m12
        cand[3] = qa[3] * qa[3]
    var den = qa[best]
    if den < Float32(0.1):
        den = Float32(0.1)
    den = Float32(2.0) * den
    var out = Array[Float32, 4](fill=Float32(0))
    for c in range(4):
        out[c] = cand[c] / den
    return out^


def ref_quat_mul(a: Array[Float32, 4], b: Array[Float32, 4]) -> Array[Float32, 4]:
    """The reference's `quat_mul` (w last), its own arrangement of the product."""
    var x1 = a[0]
    var y1 = a[1]
    var z1 = a[2]
    var w1 = a[3]
    var x2 = b[0]
    var y2 = b[1]
    var z2 = b[2]
    var w2 = b[3]
    var ww = _m(z1 + x1, x2 + y2)
    var yy = _m(w1 - y1, w2 + z2)
    var zz = _m(w1 + y1, w2 - z2)
    var xx = (ww + yy) + zz
    var qq = _m(Float32(0.5), xx + _m(z1 - x1, x2 - y2))
    var out = Array[Float32, 4](fill=Float32(0))
    out[3] = (qq - ww) + _m(z1 - y1, y2 - z2)
    out[0] = (qq - xx) + _m(x1 + w1, x2 + w2)
    out[1] = (qq - yy) + _m(w1 - x1, y2 + z2)
    out[2] = (qq - zz) + _m(z1 + y1, w2 - x2)
    return out^


def ref_normalize4(q: Array[Float32, 4]) -> Array[Float32, 4]:
    """`normalize(x, eps=1e-9)`: x / max(‖x‖, 1e-9)."""
    var n = sqrt(((_m(q[0], q[0]) + _m(q[1], q[1])) + _m(q[2], q[2])) + _m(q[3], q[3]))
    if n < Float32(1e-9):
        n = Float32(1e-9)
    var out = Array[Float32, 4](fill=Float32(0))
    for c in range(4):
        out[c] = q[c] / n
    return out^


def ref_quat_rotate(q: Array[Float32, 4], vx: Float32, vy: Float32, vz: Float32) -> Array[Float32, 3]:
    """`my_quat_rotate` / `quat_rotate`: a + b + c with a = v(2w² − 1),
    b = 2w (q_v × v), c = 2 q_v (q_v · v)."""
    var w = q[3]
    var s = _m(_m(Float32(2.0), w), w) - Float32(1.0)
    var cx = _m(_m(_m(q[1], vz) - _m(q[2], vy), w), Float32(2.0))
    var cy = _m(_m(_m(q[2], vx) - _m(q[0], vz), w), Float32(2.0))
    var cz = _m(_m(_m(q[0], vy) - _m(q[1], vx), w), Float32(2.0))
    var d = _m((_m(q[0], vx) + _m(q[1], vy)) + _m(q[2], vz), Float32(2.0))
    var out = Array[Float32, 3](fill=Float32(0))
    out[0] = (_m(vx, s) + cx) + _m(q[0], d)
    out[1] = (_m(vy, s) + cy) + _m(q[1], d)
    out[2] = (_m(vz, s) + cz) + _m(q[2], d)
    return out^


def ref_quat_rotate_inverse(q: Array[Float32, 4], vx: Float32, vy: Float32, vz: Float32) -> Array[Float32, 3]:
    """`quat_rotate_inverse`: a − b + c."""
    var w = q[3]
    var s = _m(_m(Float32(2.0), w), w) - Float32(1.0)
    var cx = _m(_m(_m(q[1], vz) - _m(q[2], vy), w), Float32(2.0))
    var cy = _m(_m(_m(q[2], vx) - _m(q[0], vz), w), Float32(2.0))
    var cz = _m(_m(_m(q[0], vy) - _m(q[1], vx), w), Float32(2.0))
    var d = _m((_m(q[0], vx) + _m(q[1], vy)) + _m(q[2], vz), Float32(2.0))
    var out = Array[Float32, 3](fill=Float32(0))
    out[0] = (_m(vx, s) - cx) + _m(q[0], d)
    out[1] = (_m(vy, s) - cy) + _m(q[1], d)
    out[2] = (_m(vz, s) - cz) + _m(q[2], d)
    return out^


def ref_heading_inv(q: Array[Float32, 4]) -> Array[Float32, 4]:
    """`calc_heading_quat_inv`: heading = atan2 of the rotated x axis;
    `quat_from_angle_axis(−heading, z)` then `quat_unit`."""
    var d = ref_quat_rotate(q, Float32(1), Float32(0), Float32(0))
    var heading = atan2(d[1], d[0])
    var theta = (-heading) / Float32(2.0)
    var out = Array[Float32, 4](fill=Float32(0))
    out[2] = sin(theta)   # normalize(z) · sin(θ/2) — the axis is unit already
    out[3] = cos(theta)
    return ref_normalize4(out)


def ref_slerp(q0: Array[Float32, 4], q1: Array[Float32, 4], t: Float32) -> Array[Float32, 4]:
    """`torch_utils.slerp`, quirks included: the unnormalised midpoint below
    sin(half-angle) < 1e-3, `q0` when |cos| >= 1."""
    var c = ((_m(q0[0], q1[0]) + _m(q0[1], q1[1])) + _m(q0[2], q1[2])) + _m(q0[3], q1[3])
    var q1s = Array[Float32, 4](fill=Float32(0))
    for k in range(4):
        q1s[k] = q1[k]
    if c < Float32(0):
        for k in range(4):
            q1s[k] = -q1[k]
        c = -c
    var half_theta = acos(c)
    var sin_half = sqrt(Float32(1.0) - _m(c, c))
    var ra = sin(_m(Float32(1.0) - t, half_theta)) / sin_half
    var rb = sin(_m(t, half_theta)) / sin_half
    var out = Array[Float32, 4](fill=Float32(0))
    for k in range(4):
        out[k] = _m(ra, q0[k]) + _m(rb, q1s[k])
    if sin_half < SLERP_MIDPOINT_TOL:
        for k in range(4):
            out[k] = _m(Float32(0.5), q0[k]) + _m(Float32(0.5), q1s[k])
    if c >= Float32(1.0):
        for k in range(4):
            out[k] = q0[k]
    return out^


# ── the clip ────────────────────────────────────────────────────────────────
struct LafanClip(Movable):
    var name: String
    var fps: Int
    var n: Int
    var root_trans: List[Float32]   # n * 3
    var pose_aa: List[Float32]      # n * 30 * 3

    def __init__(out self, var name: String, fps: Int, n: Int, var root_trans: List[Float32], var pose_aa: List[Float32]):
        self.name = name^
        self.fps = fps
        self.n = n
        self.root_trans = root_trans^
        self.pose_aa = pose_aa^

    def __init__(out self, *, deinit move: Self):
        self.name = move.name^
        self.fps = move.fps
        self.n = move.n
        self.root_trans = move.root_trans^
        self.pose_aa = move.pose_aa^


def load_lafan_clips(path: String) raises -> List[LafanClip]:
    """Every clip of the pickle in its key order — the order the reference's
    motion library and the oracle store use."""
    var p = JoblibPickle.load(path)
    var keys = p.dict_keys(p.root)
    var out = List[LafanClip]()
    for i in range(len(keys)):
        var c = p.get(p.root, keys[i])
        var pa = p.get(c, String("pose_aa"))
        var sh = p.shape_of(pa)
        if len(sh) != 3 or sh[1] != 30 or sh[2] != 3:
            raise Error("lafan: clip '" + keys[i] + "' pose_aa is not (n, 30, 3)")
        var rt = p.get(c, String("root_trans_offset"))
        var fps = p.int_of(p.get(c, String("fps")))
        out.append(LafanClip(keys[i].copy(), fps, sh[0], p.array_f32(rt), p.array_f32(pa)))
    return out^


# ── the rows of one clip ────────────────────────────────────────────────────
struct LafanRows(Movable):
    """The store's columns for one clip, row-major, float32."""
    var n_rows: Int
    var qpos: List[Float32]         # 36
    var qvel: List[Float32]         # 35
    var state: List[Float32]        # 64
    var privileged: List[Float32]   # 463
    var body_pos: List[Float32]     # 93
    var body_quat: List[Float32]    # 124 (xyzw)
    var body_vel: List[Float32]     # 93
    var body_ang_vel: List[Float32] # 93

    def __init__(out self, n_rows: Int):
        self.n_rows = n_rows
        self.qpos = List[Float32](length=n_rows * LAFAN_NQ, fill=Float32(0))
        self.qvel = List[Float32](length=n_rows * LAFAN_NV, fill=Float32(0))
        self.state = List[Float32](length=n_rows * LAFAN_STATE_DIM, fill=Float32(0))
        self.privileged = List[Float32](length=n_rows * LAFAN_PRIV_DIM, fill=Float32(0))
        self.body_pos = List[Float32](length=n_rows * LAFAN_N_BODIES * 3, fill=Float32(0))
        self.body_quat = List[Float32](length=n_rows * LAFAN_N_BODIES * 4, fill=Float32(0))
        self.body_vel = List[Float32](length=n_rows * LAFAN_N_BODIES * 3, fill=Float32(0))
        self.body_ang_vel = List[Float32](length=n_rows * LAFAN_N_BODIES * 3, fill=Float32(0))

    def __init__(out self, *, deinit move: Self):
        self.n_rows = move.n_rows
        self.qpos = move.qpos^
        self.qvel = move.qvel^
        self.state = move.state^
        self.privileged = move.privileged^
        self.body_pos = move.body_pos^
        self.body_quat = move.body_quat^
        self.body_vel = move.body_vel^
        self.body_ang_vel = move.body_ang_vel^


def _gaussian_weights() -> Array[Float64, 2 * GAUSS_RADIUS + 1]:
    """scipy's `_gaussian_kernel1d(sigma, 0, radius)`: exp(−x²/2σ²) normalised, float64."""
    var w = Array[Float64, 2 * GAUSS_RADIUS + 1](fill=0.0)
    var s = 0.0
    for k in range(-GAUSS_RADIUS, GAUSS_RADIUS + 1):
        var v = exp(-Float64(k * k) / (2.0 * GAUSS_SIGMA * GAUSS_SIGMA))
        w[k + GAUSS_RADIUS] = v
        s += v
    for k in range(2 * GAUSS_RADIUS + 1):
        w[k] = w[k] / s
    return w^


def _gradient_then_filter(mut x: List[Float32], n: Int, width: Int, dt: Float32) -> List[Float32]:
    """`np.gradient(x, axis=0) / dt` in float32, then `gaussian_filter1d(σ 2,
    mode nearest)` with float64 accumulation, cast to float32. `x` is
    (n, width) row-major."""
    var g = List[Float32](length=n * width, fill=Float32(0))
    for c in range(width):
        if n == 1:
            g[c] = Float32(0)
            continue
        g[c] = (x[width + c] - x[c]) / dt
        g[(n - 1) * width + c] = (x[(n - 1) * width + c] - x[(n - 2) * width + c]) / dt
        for t in range(1, n - 1):
            g[t * width + c] = ((x[(t + 1) * width + c] - x[(t - 1) * width + c]) / Float32(2.0)) / dt
    var w = _gaussian_weights()
    var out = List[Float32](length=n * width, fill=Float32(0))
    for t in range(n):
        for c in range(width):
            var acc = 0.0
            for k in range(-GAUSS_RADIUS, GAUSS_RADIUS + 1):
                var j = t + k
                if j < 0:
                    j = 0
                if j > n - 1:
                    j = n - 1
                acc += w[k + GAUSS_RADIUS] * Float64(g[j * width + c])
            out[t * width + c] = Float32(acc)
    return out^


def _filter_only(x: List[Float32], n: Int, width: Int) -> List[Float32]:
    var w = _gaussian_weights()
    var out = List[Float32](length=n * width, fill=Float32(0))
    for t in range(n):
        for c in range(width):
            var acc = 0.0
            for k in range(-GAUSS_RADIUS, GAUSS_RADIUS + 1):
                var j = t + k
                if j < 0:
                    j = 0
                if j > n - 1:
                    j = n - 1
                acc += w[k + GAUSS_RADIUS] * Float64(x[j * width + c])
            out[t * width + c] = Float32(acc)
    return out^


def _q4(xs: List[Float32], base: Int) -> Array[Float32, 4]:
    var q = Array[Float32, 4](fill=Float32(0))
    for k in range(4):
        q[k] = xs[base + k]
    return q^


def convert_clip(clip: LafanClip, mut env: UnitreeG1[DType.float64], motion_id: Int) raises -> LafanRows:
    """Steps 1–5 of the module docstring for one clip.

    Step 1 — the pickle's `(root_trans_offset, pose_aa)` to a root pose and
    29 joint angles — is the ONLY part of the pipeline that knows LAFAN's
    format; it is done here and steps 2–5 are `convert_frames`, which
    `data/amass.mojo` reuses. The arithmetic is unchanged and in the same
    order, so `tests/robots/test_lafan_import_vs_oracle.mojo` still holds
    the whole path against the reference dump.
    """
    var n = clip.n
    var root_pos = List[Float32](length=n * 3, fill=Float32(0))
    var root_quat = List[Float32](length=n * 4, fill=Float32(0))
    var fdof = List[Float32](length=n * G1_N_DOF, fill=Float32(0))
    for f in range(n):
        var rq = aa_to_quat_wxyz(clip.pose_aa[f * 90 + 0], clip.pose_aa[f * 90 + 1], clip.pose_aa[f * 90 + 2])
        for k in range(4):
            root_quat[f * 4 + k] = rq[k]
        for j in range(G1_N_DOF):
            var b = f * 90 + (1 + j) * 3
            fdof[f * G1_N_DOF + j] = clip.pose_aa[b] + clip.pose_aa[b + 1] + clip.pose_aa[b + 2]
        for c in range(3):
            root_pos[f * 3 + c] = clip.root_trans[f * 3 + c]
    _ = motion_id
    return convert_frames(clip.fps, n, root_pos, root_quat, fdof, env)


def convert_frames(
    fps: Int,
    n: Int,
    root_pos: List[Float32],
    root_quat: List[Float32],
    fdof: List[Float32],
    mut env: UnitreeG1[DType.float64],
) raises -> LafanRows:
    """Steps 2–5 for `n` frames at `fps`, whatever produced them.

    `root_pos` is `n × 3`, `root_quat` is `n × 4` WXYZ and `fdof` is
    `n × 29`, all float32 — the form both LAFAN (from `pose_aa`) and the
    retargeted AMASS dump (stored that way) reduce to.

    ⚠ `root_quat` IS THE QUATERNION THE ROOT'S ROTATION MATRIX IS BUILT
    FROM, unnormalised. The engine renormalises a free joint's quaternion in
    float64; the reference does not, and the one ulp between them moves the
    slerp's midpoint branch on 18 rows of LAFAN clip 0 (measured). So the
    root body takes this quaternion and every other body takes the engine's.
    """
    var dt32 = Float32(1.0 / Float64(fps))
    comptime NB = LAFAN_N_BODIES
    if n < 3:
        raise Error(
            "lafan: a clip of " + String(n) + " frames cannot be converted —"
            " the reference's joint-velocity tail reads index n − 3"
        )

    # ── 2: frames -> world poses of the 31 bodies (xyzw, re-signed) ──────
    var fpos = List[Float32](length=n * NB * 3, fill=Float32(0))
    var frot = List[Float32](length=n * NB * 4, fill=Float32(0))
    var qp = List[Float64](length=LAFAN_NQ, fill=0.0)
    var qv = List[Float64](length=LAFAN_NV, fill=0.0)
    for f in range(n):
        var rq = Array[Float32, 4](fill=Float32(0))
        for k in range(4):
            rq[k] = root_quat[f * 4 + k]
        qp[0] = Float64(root_pos[f * 3 + 0])
        qp[1] = Float64(root_pos[f * 3 + 1])
        qp[2] = Float64(root_pos[f * 3 + 2])
        qp[3] = Float64(rq[0])
        qp[4] = Float64(rq[1])
        qp[5] = Float64(rq[2])
        qp[6] = Float64(rq[3])
        for j in range(G1_N_DOF):
            qp[7 + j] = Float64(fdof[f * G1_N_DOF + j])
        env.set_state(qp, qv)
        for s in range(LAFAN_N_SKEL):
            var b = g1_skeleton_body(s)
            for c in range(3):
                fpos[(f * NB + s) * 3 + c] = Float32(env.d.xpos.data[b * 3 + c])
            # our xquat is (x, y, z, w); re-sign through the reference's matrix rule.
            # ⚠ THE ROOT TAKES THE CALLER'S QUATERNION, not the engine's — see
            # this function's docstring: one ulp of renormalisation moves the
            # slerp's branch on 18 rows of LAFAN clip 0 (measured).
            var m: Array[Float32, 9]
            if s == 0:
                m = quat_to_matrix_wxyz(rq[0], rq[1], rq[2], rq[3])
            else:
                m = quat_to_matrix_wxyz(
                    Float32(env.d.xquat.data[b * 4 + 3]), Float32(env.d.xquat.data[b * 4 + 0]),
                    Float32(env.d.xquat.data[b * 4 + 1]), Float32(env.d.xquat.data[b * 4 + 2]),
                )
            var qw = matrix_to_quat_wxyz(m)
            frot[(f * NB + s) * 4 + 0] = qw[1]
            frot[(f * NB + s) * 4 + 1] = qw[2]
            frot[(f * NB + s) * 4 + 2] = qw[3]
            frot[(f * NB + s) * 4 + 3] = qw[0]
        # the head: torso pos + R_torso (0, 0, 0.35), torso's rotation
        var tb = TORSO_BODY_IDX
        var tq = Array[Float32, 4](fill=Float32(0))
        for k in range(4):
            tq[k] = frot[(f * NB + 15) * 4 + k]   # skeleton index 15 = torso_link
        var off = ref_quat_rotate(tq, Float32(0), Float32(0), Float32(G1_HEAD_OFFSET_Z))
        for c in range(3):
            fpos[(f * NB + 30) * 3 + c] = Float32(env.d.xpos.data[tb * 3 + c]) + off[c]
        for k in range(4):
            frot[(f * NB + 30) * 4 + k] = tq[k]

    # ── 3: velocities at the clip's fps ─────────────────────────────────────
    var fvel = _gradient_then_filter(fpos, n, NB * 3, dt32)
    var fang_raw = List[Float32](length=n * NB * 3, fill=Float32(0))
    for f in range(n - 1):
        for b in range(NB):
            var q0 = _q4(frot, (f * NB + b) * 4)
            var q1 = _q4(frot, ((f + 1) * NB + b) * 4)
            var q0c = Array[Float32, 4](fill=Float32(0))
            q0c[0] = -q0[0]
            q0c[1] = -q0[1]
            q0c[2] = -q0[2]
            q0c[3] = q0[3]
            var d = ref_normalize4(ref_quat_mul(q1, q0c))
            var s = _m(_m(Float32(2.0), d[3]), d[3]) - Float32(1.0)
            if s > Float32(1.0):
                s = Float32(1.0)
            if s < Float32(-1.0):
                s = Float32(-1.0)
            var angle = acos(s)
            var an = sqrt((_m(d[0], d[0]) + _m(d[1], d[1])) + _m(d[2], d[2]))
            if an < Float32(1e-9):
                an = Float32(1e-9)
            for c in range(3):
                fang_raw[(f * NB + b) * 3 + c] = _m(d[c] / an, angle) / dt32
    var fang = _filter_only(fang_raw, n, NB * 3)
    var fdvel = List[Float32](length=n * G1_N_DOF, fill=Float32(0))
    for f in range(n - 1):
        for j in range(G1_N_DOF):
            fdvel[f * G1_N_DOF + j] = (fdof[(f + 1) * G1_N_DOF + j] - fdof[f * G1_N_DOF + j]) / dt32
    # `cat([dof_vel, dof_vel[:, -2:-1]])`: the appended last row is the
    # SECOND-TO-LAST difference (index n − 3 of n − 1 differences), not the
    # last one — the reference's slice, reproduced.
    if n >= 3:
        for j in range(G1_N_DOF):
            fdvel[(n - 1) * G1_N_DOF + j] = fdvel[(n - 3) * G1_N_DOF + j]

    # ── 4: resample to 50 Hz ────────────────────────────────────────────────
    var length64 = 1.0 / Float64(fps) * Float64(n - 1)
    var length32 = Float32(length64)
    var n_rows = Int(ceil_div_rows(Float64(length32), LAFAN_ENV_DT))
    var rows = LafanRows(n_rows)
    var default = Array[Float32, G1_N_DOF](fill=Float32(0))
    for j in range(G1_N_DOF):
        default[j] = Float32(g1_default_pos(j))
    for r in range(n_rows):
        var time = _m(Float32(r), Float32(LAFAN_ENV_DT))
        var phase = time / length32
        if phase < Float32(0):
            phase = Float32(0)
        if phase > Float32(1):
            phase = Float32(1)
        var idx0 = Int(_m(phase, Float32(n - 1)))
        var idx1 = idx0 + 1
        if idx1 > n - 1:
            idx1 = n - 1
        var blend = (time - _m(Float32(idx0), dt32)) / dt32
        if blend < Float32(0):
            blend = Float32(0)
        if blend > Float32(1):
            blend = Float32(1)
        var omb = Float32(1.0) - blend
        # bodies
        var bp = Array[Float32, NB * 3](fill=Float32(0))
        var bq = Array[Float32, NB * 4](fill=Float32(0))
        var bv = Array[Float32, NB * 3](fill=Float32(0))
        var bw = Array[Float32, NB * 3](fill=Float32(0))
        for b in range(NB):
            for c in range(3):
                var i0 = (idx0 * NB + b) * 3 + c
                var i1 = (idx1 * NB + b) * 3 + c
                bp[b * 3 + c] = _m(omb, fpos[i0]) + _m(blend, fpos[i1])
                bv[b * 3 + c] = _m(omb, fvel[i0]) + _m(blend, fvel[i1])
                bw[b * 3 + c] = _m(omb, fang[i0]) + _m(blend, fang[i1])
            var q = ref_slerp(_q4(frot, (idx0 * NB + b) * 4), _q4(frot, (idx1 * NB + b) * 4), blend)
            for k in range(4):
                bq[b * 4 + k] = q[k]
        var dof = Array[Float32, G1_N_DOF](fill=Float32(0))
        var dvel = Array[Float32, G1_N_DOF](fill=Float32(0))
        for j in range(G1_N_DOF):
            dof[j] = _m(omb, fdof[idx0 * G1_N_DOF + j]) + _m(blend, fdof[idx1 * G1_N_DOF + j])
            dvel[j] = _m(omb, fdvel[idx0 * G1_N_DOF + j]) + _m(blend, fdvel[idx1 * G1_N_DOF + j])

        # ── 5: the row ──────────────────────────────────────────────────────
        var rb = r * LAFAN_NQ
        rows.qpos[rb + 0] = bp[0]
        rows.qpos[rb + 1] = bp[1]
        rows.qpos[rb + 2] = bp[2]
        rows.qpos[rb + 3] = bq[3]   # WXYZ
        rows.qpos[rb + 4] = bq[0]
        rows.qpos[rb + 5] = bq[1]
        rows.qpos[rb + 6] = bq[2]
        for j in range(G1_N_DOF):
            rows.qpos[rb + 7 + j] = dof[j]
        var vb = r * LAFAN_NV
        for c in range(3):
            rows.qvel[vb + c] = bv[c]
            rows.qvel[vb + 3 + c] = bw[c]
        for j in range(G1_N_DOF):
            rows.qvel[vb + 6 + j] = dvel[j]
        for b in range(NB):
            for c in range(3):
                rows.body_pos[(r * NB + b) * 3 + c] = bp[b * 3 + c]
                rows.body_vel[(r * NB + b) * 3 + c] = bv[b * 3 + c]
                rows.body_ang_vel[(r * NB + b) * 3 + c] = bw[b * 3 + c]
            for k in range(4):
                rows.body_quat[(r * NB + b) * 4 + k] = bq[b * 4 + k]
        # state 64
        var sb = r * LAFAN_STATE_DIM
        var rootq = Array[Float32, 4](fill=Float32(0))
        for k in range(4):
            rootq[k] = bq[k]
        var g = ref_quat_rotate_inverse(rootq, Float32(0), Float32(0), Float32(-1))
        for j in range(G1_N_DOF):
            rows.state[sb + j] = dof[j] - default[j]
            rows.state[sb + G1_N_DOF + j] = dvel[j]
        for c in range(3):
            rows.state[sb + 58 + c] = g[c]
            rows.state[sb + 61 + c] = bw[c]
        # privileged 463
        var pb = r * LAFAN_PRIV_DIM
        var h = ref_heading_inv(rootq)
        rows.privileged[pb + 0] = bp[2]
        for b in range(1, NB):
            var lp = ref_quat_rotate(h, bp[b * 3 + 0] - bp[0], bp[b * 3 + 1] - bp[1], bp[b * 3 + 2] - bp[2])
            for c in range(3):
                rows.privileged[pb + 1 + (b - 1) * 3 + c] = lp[c]
        for b in range(NB):
            var q = Array[Float32, 4](fill=Float32(0))
            for k in range(4):
                q[k] = bq[b * 4 + k]
            var lr = ref_quat_mul(h, q)
            var tan = ref_quat_rotate(lr, Float32(1), Float32(0), Float32(0))
            var nrm = ref_quat_rotate(lr, Float32(0), Float32(0), Float32(1))
            for c in range(3):
                rows.privileged[pb + 91 + b * 6 + c] = tan[c]
                rows.privileged[pb + 91 + b * 6 + 3 + c] = nrm[c]
            var lv = ref_quat_rotate(h, bv[b * 3 + 0], bv[b * 3 + 1], bv[b * 3 + 2])
            var lw = ref_quat_rotate(h, bw[b * 3 + 0], bw[b * 3 + 1], bw[b * 3 + 2])
            for c in range(3):
                rows.privileged[pb + 277 + b * 3 + c] = lv[c]
                rows.privileged[pb + 370 + b * 3 + c] = lw[c]
    return rows^


def ceil_div_rows(length: Float64, dt: Float64) -> Int:
    """`int(np.ceil(length / dt))` with `length` the float32 motion length
    widened to float64, as the dump did."""
    var q = length / dt
    var k = Int(q)
    if Float64(k) < q:
        k += 1
    return k
