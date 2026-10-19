"""OSC_POSE, one lane, inside a kernel — L4b. THE implementation; the CPU
controller in `osc_pose.mojo` is a one-lane caller of these functions.

    osc_reset_gpu(...)          # once per episode, after the state is current
    osc_set_goal_gpu(...)       # on a POLICY step
    osc_run_gpu(...)            # every substep -> writes `ctrl`
    osc_gripper_gpu(...)        # every substep -> writes the two finger ctrls

## ⚠⚠ WHY THE MATRICES LIVE IN A DEVICE BUFFER AND NOT IN LOCALS

The obvious shape is `Array[Scalar[T], 49]` for M and friends, indexed by the
loop counters of a Gauss-Jordan. **That is the Metal miscompute trap**: a
per-thread array indexed by a RUNTIME value reads back wrong, silently, with
no crash — measured at 16 elements in `equality_tendon.mojo` and again at
THREE in `_capsule_box_second_pos`, where a value arrived correctly and read
back `-0.0`. So every matrix here is a slice of a `[BATCH, OSC_WORK_WORDS]`
GLOBAL tensor addressed `env, base + i * n + j`, which is what every other
physics3d kernel does with `scratch.M`.

It costs 456 floats per lane (3.6 KB at 1024 lanes, float32) and buys a
kernel that is safe to index.

## THE RECORD, THE STATE, THE WORK

* `refs` — `[OSC_REF_WORDS]`, SHARED across lanes, built once on the host by
  `build_osc_refs`: which velocity index / qpos address / joint / actuator
  each of the seven arm joints is, the torque limits, the grip site and its
  body, the two gripper actuators and their ctrl ranges, and the six gains.
  A model record, in the same spirit as `Model.joints`.
* `state` — `[BATCH, OSC_STATE_WORDS]`, PER LANE and PER EPISODE: the goal
  position and orientation, the joint configuration the nullspace term pulls
  toward, the gripper's ramp position, and a singular-matrix flag.
* `work` — `[BATCH, OSC_WORK_WORDS]`, per lane, scratch within one call.

## ⚠ A KERNEL CANNOT RAISE, SO SINGULARITY IS A FLAG

`OSC_IDX_SINGULAR` is set when a pivot falls below `SINGULAR_PIVOT` and the
torques are then left at zero for that lane. The host wrapper reads the word
and raises; a batched driver should treat it as a lane to reset. Silently
returning a pseudo-inverse is what robosuite does and is exactly the
behaviour this refuses — see `osc_pose.mojo`'s header.

## WHAT IS READ, AND WHEN IT MUST BE FRESH

`osc_run_gpu` reads `site_xpos`, `xquat`, `qpos`, `qvel`, `subtree_com`,
`cdof`, `M` and `bias` **at the current state** — robosuite calls
`sim.forward()` at the top of `update()` and so must a caller here, every
substep, before this. `osc_refresh_dynamics_gpu` is that call: FK, body
velocities, subtree_com, cdof, CRBA, RNE, in the integrator's own order.

⚠ `M` IS THE RIGID-BODY MASS MATRIX WITHOUT THE ARMATURE — our CRBA leaves
it out and the integrator adds it in a separate pass. `mj_fullM`, which
robosuite reads, HAS it. The arm block is built with `dof_armature` added to
its diagonal here, from the joint record `refs` names.
"""

from std.math import sqrt, sin, cos

from layout import TileTensor, TensorLayout
from max.gpu.host import DeviceContext

from ..fields import Data, Model, DynamicsScratch, DimsLike
from ..kinematics.forward_kinematics import (
    forward_kinematics, compute_body_velocities,
)
from .subtree_com import compute_subtree_com
from .cdof import compute_cdof
from .mass_matrix import compute_mass_matrix
from .rne import compute_bias_forces_rne

from ..gpu.constants import (
    MODEL_BODY_SIZE, MODEL_JOINT_SIZE, MODEL_META_SIZE, MODEL_SITE_SIZE,
    MODEL_META_IDX_NJOINT,
    JOINT_IDX_TYPE, JOINT_IDX_BODY_ID, JOINT_IDX_DOF_ADR, JOINT_IDX_ARMATURE,
    BODY_IDX_PARENT, BODY_IDX_ROOTID, BODY_IDX_WELDID,
    SITE_IDX_BODY, SITE_IDX_QUAT_X, SITE_IDX_QUAT_Y, SITE_IDX_QUAT_Z,
    SITE_IDX_QUAT_W,
)
from ..joint_types import JNT_FREE, JNT_BALL
from ..kinematics.quat_math import gpu_quat_mul


# ── shapes ─────────────────────────────────────────────────────────────────
comptime OSC_ACTION_DIM: Int = 7
"""LIBERO's action: six pose deltas then the gripper.

⚠ IT LIVES HERE, BESIDE THE KERNEL THAT INDEXES BY IT. `osc_gripper_gpu` reads
`actions[ACT_DIM - 1]`, so this width is part of the kernel ABI; it used to be
declared in `osc_pose.mojo` (which re-exports it) and a batched caller needed
the same number without wanting the host controller."""

comptime OSC_ARM: Int = 7
"""The arm's dof count. OSC_POSE is a 6-dof task on a 7-dof arm — the
nullspace term is what the seventh buys, so this is not a free parameter."""
comptime OSC_WRENCH: Int = 6

comptime SINGULAR_PIVOT: Float64 = 1e-12


# ── `refs`: the model record, built once on the host ───────────────────────
comptime OSC_REF_DOF: Int = 0            # 7: velocity index per arm joint
comptime OSC_REF_QADR: Int = 7           # 7: qpos address per arm joint
comptime OSC_REF_JOINT: Int = 14         # 7: joint index (for its armature)
comptime OSC_REF_TMIN: Int = 21          # 7
comptime OSC_REF_TMAX: Int = 28          # 7
comptime OSC_REF_ACT: Int = 35           # 7: actuator index to write ctrl into
comptime OSC_REF_SITE: Int = 42
comptime OSC_REF_SITE_BODY: Int = 43
comptime OSC_REF_GRIP_ACT0: Int = 44
comptime OSC_REF_GRIP_ACT1: Int = 45
comptime OSC_REF_GRIP_BIAS0: Int = 46
comptime OSC_REF_GRIP_W0: Int = 47
comptime OSC_REF_GRIP_BIAS1: Int = 48
comptime OSC_REF_GRIP_W1: Int = 49
comptime OSC_REF_KP: Int = 50
comptime OSC_REF_KD: Int = 51
comptime OSC_REF_OUT_POS: Int = 52
comptime OSC_REF_OUT_ORI: Int = 53
comptime OSC_REF_NS_KP: Int = 54
comptime OSC_REF_NS_KV: Int = 55
comptime OSC_REF_GRIP_SPEED: Int = 56
comptime OSC_REF_WORDS: Int = 57


# ── `state`: per lane, per episode ─────────────────────────────────────────
comptime OSC_IDX_GOAL_POS: Int = 0       # 3
comptime OSC_IDX_GOAL_MAT: Int = 3       # 9, row-major
comptime OSC_IDX_Q0: Int = 12            # 7: the nullspace target
comptime OSC_IDX_GRIP0: Int = 19         # the ramp's two internal positions
comptime OSC_IDX_GRIP1: Int = 20
comptime OSC_IDX_SINGULAR: Int = 21
comptime OSC_IDX_READY: Int = 22
"""0 until `osc_reset_gpu` has run. `osc_run_gpu` leaves a lane's torques at
zero while it is 0 rather than driving toward an unwritten goal."""
comptime OSC_STATE_WORDS: Int = 24


# ── `work`: per lane, within one call ──────────────────────────────────────
comptime OSC_W_JAC: Int = 0                          # 6 x 7
comptime OSC_W_MBLK: Int = OSC_W_JAC + 42            # 7 x 7 (+ armature)
comptime OSC_W_MINV: Int = OSC_W_MBLK + 49           # 7 x 7
comptime OSC_W_AUG: Int = OSC_W_MINV + 49            # 7 x 14, Gauss-Jordan
comptime OSC_W_LAMF: Int = OSC_W_AUG + 98            # 6 x 6
comptime OSC_W_LAMP: Int = OSC_W_LAMF + 36           # 3 x 3
comptime OSC_W_LAMO: Int = OSC_W_LAMP + 9            # 3 x 3
comptime OSC_W_JBAR: Int = OSC_W_LAMO + 9            # 7 x 6
comptime OSC_W_NUL: Int = OSC_W_JBAR + 42            # 7 x 7
comptime OSC_W_TMP: Int = OSC_W_NUL + 49             # 7 x 7, products
comptime OSC_W_TMP2: Int = OSC_W_TMP + 49            # 7 x 7
comptime OSC_W_VEC: Int = OSC_W_TMP2 + 49            # 16: forces, errors
comptime OSC_WORK_WORDS: Int = OSC_W_VEC + 16


# ── flat-buffer linear algebra, all indices runtime, all in GLOBAL memory ──


@always_inline
def _inv_into[
    DTYPE: DType, L_WORK: TensorLayout,
](
    work: TileTensor[DTYPE, L_WORK, MutAnyOrigin],
    env: Int, src: Int, dst: Int, n: Int,
) -> Bool:
    """`dst = inv(src)`, Gauss-Jordan with partial pivoting, through the
    `OSC_W_AUG` workspace. False on a pivot below `SINGULAR_PIVOT`.

    ⚠ THE SAME ALGORITHM AS THE HOST USED TO SPELL, and now the only one:
    `osc_pose.OscPose` calls this function on a one-lane CPU tensor. A second
    spelling of a matrix inverse is exactly the drift this tree keeps paying
    for."""
    var cols = 2 * n
    for i in range(n):
        for j in range(n):
            work[env, OSC_W_AUG + i * cols + j] = work[env, src + i * n + j]
        for j in range(n):
            work[env, OSC_W_AUG + i * cols + n + j] = (
                Scalar[DTYPE](1) if i == j else Scalar[DTYPE](0)
            )
    for c in range(n):
        var piv = c
        var best = rebind[Scalar[DTYPE]](work[env, OSC_W_AUG + c * cols + c])
        if best < Scalar[DTYPE](0):
            best = -best
        for r in range(c + 1, n):
            var v = rebind[Scalar[DTYPE]](work[env, OSC_W_AUG + r * cols + c])
            if v < Scalar[DTYPE](0):
                v = -v
            if v > best:
                best = v
                piv = r
        if best < Scalar[DTYPE](SINGULAR_PIVOT):
            return False
        if piv != c:
            for j in range(cols):
                var t = rebind[Scalar[DTYPE]](work[env, OSC_W_AUG + c * cols + j])
                work[env, OSC_W_AUG + c * cols + j] = work[env, OSC_W_AUG + piv * cols + j]
                work[env, OSC_W_AUG + piv * cols + j] = t
        var p = rebind[Scalar[DTYPE]](work[env, OSC_W_AUG + c * cols + c])
        for j in range(cols):
            work[env, OSC_W_AUG + c * cols + j] = (
                rebind[Scalar[DTYPE]](work[env, OSC_W_AUG + c * cols + j]) / p
            )
        for r in range(n):
            if r == c:
                continue
            var f = rebind[Scalar[DTYPE]](work[env, OSC_W_AUG + r * cols + c])
            if f == Scalar[DTYPE](0):
                continue
            for j in range(cols):
                work[env, OSC_W_AUG + r * cols + j] = (
                    rebind[Scalar[DTYPE]](work[env, OSC_W_AUG + r * cols + j])
                    - f * rebind[Scalar[DTYPE]](work[env, OSC_W_AUG + c * cols + j])
                )
    for i in range(n):
        for j in range(n):
            work[env, dst + i * n + j] = work[env, OSC_W_AUG + i * cols + n + j]
    return True


@always_inline
def _mul_into[
    DTYPE: DType, L_WORK: TensorLayout,
](
    work: TileTensor[DTYPE, L_WORK, MutAnyOrigin],
    env: Int, a: Int, b: Int, dst: Int, n: Int, k: Int, m: Int,
):
    """`dst[n x m] = a[n x k] @ b[k x m]`. `dst` must not alias `a` or `b`."""
    for i in range(n):
        for j in range(m):
            var s = Scalar[DTYPE](0)
            for t in range(k):
                s += (
                    rebind[Scalar[DTYPE]](work[env, a + i * k + t])
                    * rebind[Scalar[DTYPE]](work[env, b + t * m + j])
                )
            work[env, dst + i * m + j] = s


@always_inline
def _mul_t_into[
    DTYPE: DType, L_WORK: TensorLayout,
](
    work: TileTensor[DTYPE, L_WORK, MutAnyOrigin],
    env: Int, a: Int, b: Int, dst: Int, n: Int, k: Int, m: Int,
):
    """`dst[n x m] = a[k x n]^T @ b[k x m]` — the transpose is READ, never
    materialised."""
    for i in range(n):
        for j in range(m):
            var s = Scalar[DTYPE](0)
            for t in range(k):
                s += (
                    rebind[Scalar[DTYPE]](work[env, a + t * n + i])
                    * rebind[Scalar[DTYPE]](work[env, b + t * m + j])
                )
            work[env, dst + i * m + j] = s


@always_inline
def _mul_bt_into[
    DTYPE: DType, L_WORK: TensorLayout,
](
    work: TileTensor[DTYPE, L_WORK, MutAnyOrigin],
    env: Int, a: Int, b: Int, dst: Int, n: Int, k: Int, m: Int,
):
    """`dst[n x m] = a[n x k] @ b[m x k]^T`."""
    for i in range(n):
        for j in range(m):
            var s = Scalar[DTYPE](0)
            for t in range(k):
                s += (
                    rebind[Scalar[DTYPE]](work[env, a + i * k + t])
                    * rebind[Scalar[DTYPE]](work[env, b + j * k + t])
                )
            work[env, dst + i * m + j] = s


@always_inline
def _is_ancestor[
    DTYPE: DType, L_BODIES: TensorLayout,
](
    bodies: TileTensor[DTYPE, L_BODIES, MutAnyOrigin],
    body: Int, maybe: Int,
) -> Bool:
    """Is `maybe` `body` or one of its ancestors? Body ids are in tree order,
    so the walk terminates in at most the tree's depth."""
    var b = body
    while b > 0:
        if b == maybe:
            return True
        b = Int(rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_PARENT]))
    return b == maybe


# ── the site's world orientation, as a row-major 3x3 in `work` ─────────────


@always_inline
def _site_mat_into[
    DTYPE: DType, L_WORK: TensorLayout, L_B4: TensorLayout, L_SITES: TensorLayout,
](
    work: TileTensor[DTYPE, L_WORK, MutAnyOrigin],
    xquat: TileTensor[DTYPE, L_B4, MutAnyOrigin],
    sites: TileTensor[DTYPE, L_SITES, MutAnyOrigin],
    env: Int, site: Int, site_body: Int, dst: Int,
):
    """`site_xmat` — `xquat[site body] * site quat`, the product
    `sensors/touch.mojo` forms, as a rotation matrix."""
    var q = gpu_quat_mul[DTYPE](
        rebind[Scalar[DTYPE]](xquat[env, site_body * 4 + 0]),
        rebind[Scalar[DTYPE]](xquat[env, site_body * 4 + 1]),
        rebind[Scalar[DTYPE]](xquat[env, site_body * 4 + 2]),
        rebind[Scalar[DTYPE]](xquat[env, site_body * 4 + 3]),
        rebind[Scalar[DTYPE]](sites[site, SITE_IDX_QUAT_X]),
        rebind[Scalar[DTYPE]](sites[site, SITE_IDX_QUAT_Y]),
        rebind[Scalar[DTYPE]](sites[site, SITE_IDX_QUAT_Z]),
        rebind[Scalar[DTYPE]](sites[site, SITE_IDX_QUAT_W]),
    )
    var two = Scalar[DTYPE](2)
    var one = Scalar[DTYPE](1)
    work[env, dst + 0] = one - two * (q[1] * q[1] + q[2] * q[2])
    work[env, dst + 1] = two * (q[0] * q[1] - q[2] * q[3])
    work[env, dst + 2] = two * (q[0] * q[2] + q[1] * q[3])
    work[env, dst + 3] = two * (q[0] * q[1] + q[2] * q[3])
    work[env, dst + 4] = one - two * (q[0] * q[0] + q[2] * q[2])
    work[env, dst + 5] = two * (q[1] * q[2] - q[0] * q[3])
    work[env, dst + 6] = two * (q[0] * q[2] - q[1] * q[3])
    work[env, dst + 7] = two * (q[1] * q[2] + q[0] * q[3])
    work[env, dst + 8] = one - two * (q[0] * q[0] + q[1] * q[1])


@always_inline
def axisangle_mat_into[
    DTYPE: DType, L_WORK: TensorLayout,
](
    work: TileTensor[DTYPE, L_WORK, MutAnyOrigin],
    env: Int, dst: Int,
    vx: Scalar[DTYPE], vy: Scalar[DTYPE], vz: Scalar[DTYPE],
):
    """`T.quat2mat(T.axisangle2quat(v))` — Rodrigues, in the working dtype.

    ⚠ robosuite casts that quaternion to float32 before forming the matrix
    (`transform_utils.quat2mat`), so its goal rotation carries ~1e-7 of
    relative error that this does not reproduce. Measured cost on a recorded
    demo: 1.5e-5 m over 80 policy steps, the whole ours-vs-MuJoCo residual.
    """
    # ⚠ THE `comptime assert` IS WHAT LETS `sin`/`cos` TAKE A GENERIC
    # `Scalar[DTYPE]` — without it the compiler has no evidence the type is
    # floating point and refuses the call. `quat_math` opens the same way.
    comptime assert (
        DTYPE.is_floating_point()
    ), "DTYPE must be a floating point type"
    var angle = sqrt(vx * vx + vy * vy + vz * vz)
    var zero = Scalar[DTYPE](0)
    var one = Scalar[DTYPE](1)
    if angle == zero:
        for i in range(3):
            for j in range(3):
                work[env, dst + i * 3 + j] = one if i == j else zero
        return
    var kx = vx / angle
    var ky = vy / angle
    var kz = vz / angle
    var s = Scalar[DTYPE](sin(angle))
    var c = one - Scalar[DTYPE](cos(angle))
    work[env, dst + 0] = one + c * (-ky * ky - kz * kz)
    work[env, dst + 1] = -s * kz + c * kx * ky
    work[env, dst + 2] = s * ky + c * kx * kz
    work[env, dst + 3] = s * kz + c * kx * ky
    work[env, dst + 4] = one + c * (-kx * kx - kz * kz)
    work[env, dst + 5] = -s * kx + c * ky * kz
    work[env, dst + 6] = -s * ky + c * kx * kz
    work[env, dst + 7] = s * kx + c * ky * kz
    work[env, dst + 8] = one + c * (-kx * kx - ky * ky)


# ── the three entry points ────────────────────────────────────────────────


@always_inline
def osc_reset_gpu[
    DTYPE: DType, L_STATE: TensorLayout, L_WORK: TensorLayout, L_REFS: TensorLayout,
    L_QPOS: TensorLayout, L_B4: TensorLayout, L_SX: TensorLayout, L_SITES: TensorLayout,
](
    state: TileTensor[DTYPE, L_STATE, MutAnyOrigin],
    work: TileTensor[DTYPE, L_WORK, MutAnyOrigin],
    refs: TileTensor[DTYPE, L_REFS, MutAnyOrigin],
    qpos: TileTensor[DTYPE, L_QPOS, MutAnyOrigin],
    xquat: TileTensor[DTYPE, L_B4, MutAnyOrigin],
    site_xpos: TileTensor[DTYPE, L_SX, MutAnyOrigin],
    sites: TileTensor[DTYPE, L_SITES, MutAnyOrigin],
    env: Int,
):
    """`Robot.reset` -> `_load_controller`: the nullspace target is the
    CURRENT joint configuration and the goal is the CURRENT eef pose
    (`OperationalSpaceController.__init__`'s last lines). The state must be
    forward-kinematically current — `osc_refresh_dynamics_gpu` first."""
    var site = Int(rebind[Scalar[DTYPE]](refs[OSC_REF_SITE]))
    var site_body = Int(rebind[Scalar[DTYPE]](refs[OSC_REF_SITE_BODY]))
    for k in range(3):
        state[env, OSC_IDX_GOAL_POS + k] = site_xpos[env, site * 3 + k]
    _site_mat_into[DTYPE, L_WORK, L_B4, L_SITES](
        work, xquat, sites, env, site, site_body, OSC_W_TMP
    )
    for k in range(9):
        state[env, OSC_IDX_GOAL_MAT + k] = work[env, OSC_W_TMP + k]
    for a in range(OSC_ARM):
        var qa = Int(rebind[Scalar[DTYPE]](refs[OSC_REF_QADR + a]))
        state[env, OSC_IDX_Q0 + a] = qpos[env, qa]
    state[env, OSC_IDX_GRIP0] = Scalar[DTYPE](0)
    state[env, OSC_IDX_GRIP1] = Scalar[DTYPE](0)
    state[env, OSC_IDX_SINGULAR] = Scalar[DTYPE](0)
    state[env, OSC_IDX_READY] = Scalar[DTYPE](1)


@always_inline
def osc_set_goal_gpu[
    DTYPE: DType, L_STATE: TensorLayout, L_WORK: TensorLayout, L_REFS: TensorLayout,
    L_ACT: TensorLayout, L_B4: TensorLayout, L_SX: TensorLayout, L_SITES: TensorLayout,
](
    state: TileTensor[DTYPE, L_STATE, MutAnyOrigin],
    work: TileTensor[DTYPE, L_WORK, MutAnyOrigin],
    refs: TileTensor[DTYPE, L_REFS, MutAnyOrigin],
    actions: TileTensor[DTYPE, L_ACT, MutAnyOrigin],
    xquat: TileTensor[DTYPE, L_B4, MutAnyOrigin],
    site_xpos: TileTensor[DTYPE, L_SX, MutAnyOrigin],
    sites: TileTensor[DTYPE, L_SITES, MutAnyOrigin],
    env: Int,
):
    """`set_goal(action)` — on a POLICY step, after the state is current.

    `scale_action` maps `clip(a, -1, 1)` to `±[0.05 m, 0.5 rad]`;
    `goal_pos = ee_pos + δp`; `goal_R = R(δθ) @ ee_R`.

    ⚠ THE ROTATION GOAL MOVES ONLY WHEN SOME δθ IS EXACTLY NON-ZERO.
    robosuite's `bools` list is `math.isclose(elem, 0.0)`, which has no
    absolute tolerance: 1e-12 counts as a rotation and 0.0 does not, and
    when every component is 0.0 the PREVIOUS goal orientation is kept. A
    demo that commands no rotation for twenty steps is therefore holding
    the orientation it had twenty steps ago, not the current one.
    """
    var site = Int(rebind[Scalar[DTYPE]](refs[OSC_REF_SITE]))
    var site_body = Int(rebind[Scalar[DTYPE]](refs[OSC_REF_SITE_BODY]))
    var out_pos = rebind[Scalar[DTYPE]](refs[OSC_REF_OUT_POS])
    var out_ori = rebind[Scalar[DTYPE]](refs[OSC_REF_OUT_ORI])
    var one = Scalar[DTYPE](1)
    var zero = Scalar[DTYPE](0)

    var rx = rebind[Scalar[DTYPE]](actions[env, 3])
    var ry = rebind[Scalar[DTYPE]](actions[env, 4])
    var rz = rebind[Scalar[DTYPE]](actions[env, 5])
    if rx > one:
        rx = one
    if rx < -one:
        rx = -one
    if ry > one:
        ry = one
    if ry < -one:
        ry = -one
    if rz > one:
        rz = one
    if rz < -one:
        rz = -one
    rx *= out_ori
    ry *= out_ori
    rz *= out_ori
    if rx != zero or ry != zero or rz != zero:
        _site_mat_into[DTYPE, L_WORK, L_B4, L_SITES](
            work, xquat, sites, env, site, site_body, OSC_W_TMP
        )
        axisangle_mat_into[DTYPE, L_WORK](work, env, OSC_W_TMP2, rx, ry, rz)
        # goal = R(delta) @ ee_R, into LAMP's nine words (free here)
        _mul_into[DTYPE, L_WORK](work, env, OSC_W_TMP2, OSC_W_TMP, OSC_W_LAMP, 3, 3, 3)
        for k in range(9):
            state[env, OSC_IDX_GOAL_MAT + k] = work[env, OSC_W_LAMP + k]
    for k in range(3):
        var a = rebind[Scalar[DTYPE]](actions[env, k])
        if a > one:
            a = one
        if a < -one:
            a = -one
        state[env, OSC_IDX_GOAL_POS + k] = (
            rebind[Scalar[DTYPE]](site_xpos[env, site * 3 + k]) + a * out_pos
        )


@always_inline
def osc_run_gpu[
    DTYPE: DType, L_STATE: TensorLayout, L_WORK: TensorLayout, L_REFS: TensorLayout,
    L_CTRL: TensorLayout,
    L_QPOS: TensorLayout, L_NV: TensorLayout, L_B4: TensorLayout, L_SX: TensorLayout, L_B3: TensorLayout,
    L_CDOF: TensorLayout, L_M: TensorLayout, L_JOINTS: TensorLayout, L_BODIES: TensorLayout,
    L_SITES: TensorLayout, L_MMETA: TensorLayout,
](
    state: TileTensor[DTYPE, L_STATE, MutAnyOrigin],
    work: TileTensor[DTYPE, L_WORK, MutAnyOrigin],
    refs: TileTensor[DTYPE, L_REFS, MutAnyOrigin],
    ctrl: TileTensor[DTYPE, L_CTRL, MutAnyOrigin],
    qpos: TileTensor[DTYPE, L_QPOS, MutAnyOrigin],
    qvel: TileTensor[DTYPE, L_NV, MutAnyOrigin],
    xquat: TileTensor[DTYPE, L_B4, MutAnyOrigin],
    site_xpos: TileTensor[DTYPE, L_SX, MutAnyOrigin],
    subtree_com: TileTensor[DTYPE, L_B3, MutAnyOrigin],
    cdof: TileTensor[DTYPE, L_CDOF, MutAnyOrigin],
    mass: TileTensor[DTYPE, L_M, MutAnyOrigin],
    bias: TileTensor[DTYPE, L_NV, MutAnyOrigin],
    joints: TileTensor[DTYPE, L_JOINTS, MutAnyOrigin],
    bodies: TileTensor[DTYPE, L_BODIES, MutAnyOrigin],
    sites: TileTensor[DTYPE, L_SITES, MutAnyOrigin],
    mmeta: TileTensor[DTYPE, L_MMETA, MutAnyOrigin],
    env: Int,
    nv: Int,
):
    """`run_controller` + `clip_torques`, one lane, into `ctrl`.

    Writes the seven arm actuators and nothing else — the gripper is
    `osc_gripper_gpu` and any other actuator is the caller's business.
    """
    if state[env, OSC_IDX_READY] == Scalar[DTYPE](0):
        return
    var zero = Scalar[DTYPE](0)
    var site = Int(rebind[Scalar[DTYPE]](refs[OSC_REF_SITE]))
    var site_body = Int(rebind[Scalar[DTYPE]](refs[OSC_REF_SITE_BODY]))
    var kp = rebind[Scalar[DTYPE]](refs[OSC_REF_KP])
    var kd = rebind[Scalar[DTYPE]](refs[OSC_REF_KD])

    # ── ONE WALK OVER THE JOINTS: the 6 x 7 arm Jacobian block AND the
    # site's spatial velocity ──────────────────────────────────────────────
    # `jac_point`'s loop, verbatim in structure: a dof contributes only when
    # its joint is the site's welded body or an ancestor of it. ⚠ THE
    # ANCESTOR TEST IS WHAT MAKES THE VELOCITY RIGHT. `get_site_xvelp` is
    # `jacp @ qvel` over the FULL dof vector, and `jacp` is ZERO on every dof
    # that does not move the site — summing `cdof` over all `nv` instead
    # would add the free objects' motion to the end-effector's velocity, a
    # 0.9 m/s term on a settling bowl.
    var root = Int(rebind[Scalar[DTYPE]](bodies[site_body, BODY_IDX_ROOTID]))
    var offx = (
        rebind[Scalar[DTYPE]](site_xpos[env, site * 3 + 0])
        - rebind[Scalar[DTYPE]](subtree_com[env, root * 3 + 0])
    )
    var offy = (
        rebind[Scalar[DTYPE]](site_xpos[env, site * 3 + 1])
        - rebind[Scalar[DTYPE]](subtree_com[env, root * 3 + 1])
    )
    var offz = (
        rebind[Scalar[DTYPE]](site_xpos[env, site * 3 + 2])
        - rebind[Scalar[DTYPE]](subtree_com[env, root * 3 + 2])
    )
    var wbody = Int(rebind[Scalar[DTYPE]](bodies[site_body, BODY_IDX_WELDID]))
    for r in range(OSC_WRENCH):
        for a in range(OSC_ARM):
            work[env, OSC_W_JAC + r * OSC_ARM + a] = zero
        # the site's spatial velocity accumulates in VEC+6..11
        work[env, OSC_W_VEC + 6 + r] = zero
    if wbody != 0:
        var njnt = Int(rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_NJOINT]))
        for j in range(njnt):
            var jb = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_BODY_ID]))
            if not _is_ancestor[DTYPE, L_BODIES](bodies, wbody, jb):
                continue
            var jt = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_TYPE]))
            var adr = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_DOF_ADR]))
            var ndof = 1
            if jt == JNT_FREE:
                ndof = 6
            elif jt == JNT_BALL:
                ndof = 3
            for dd in range(ndof):
                var i = adr + dd
                var angx = rebind[Scalar[DTYPE]](cdof[env, i * 6 + 0])
                var angy = rebind[Scalar[DTYPE]](cdof[env, i * 6 + 1])
                var angz = rebind[Scalar[DTYPE]](cdof[env, i * 6 + 2])
                var c0 = (
                    rebind[Scalar[DTYPE]](cdof[env, i * 6 + 3])
                    + angy * offz - angz * offy
                )
                var c1 = (
                    rebind[Scalar[DTYPE]](cdof[env, i * 6 + 4])
                    + angz * offx - angx * offz
                )
                var c2 = (
                    rebind[Scalar[DTYPE]](cdof[env, i * 6 + 5])
                    + angx * offy - angy * offx
                )
                var v = rebind[Scalar[DTYPE]](qvel[env, i])
                work[env, OSC_W_VEC + 6 + 0] = rebind[Scalar[DTYPE]](work[env, OSC_W_VEC + 6 + 0]) + c0 * v
                work[env, OSC_W_VEC + 6 + 1] = rebind[Scalar[DTYPE]](work[env, OSC_W_VEC + 6 + 1]) + c1 * v
                work[env, OSC_W_VEC + 6 + 2] = rebind[Scalar[DTYPE]](work[env, OSC_W_VEC + 6 + 2]) + c2 * v
                work[env, OSC_W_VEC + 6 + 3] = rebind[Scalar[DTYPE]](work[env, OSC_W_VEC + 6 + 3]) + angx * v
                work[env, OSC_W_VEC + 6 + 4] = rebind[Scalar[DTYPE]](work[env, OSC_W_VEC + 6 + 4]) + angy * v
                work[env, OSC_W_VEC + 6 + 5] = rebind[Scalar[DTYPE]](work[env, OSC_W_VEC + 6 + 5]) + angz * v
                # is this dof one of the arm's seven? then it is a J column
                for a in range(OSC_ARM):
                    if Int(rebind[Scalar[DTYPE]](refs[OSC_REF_DOF + a])) != i:
                        continue
                    work[env, OSC_W_JAC + 0 * OSC_ARM + a] = c0
                    work[env, OSC_W_JAC + 1 * OSC_ARM + a] = c1
                    work[env, OSC_W_JAC + 2 * OSC_ARM + a] = c2
                    work[env, OSC_W_JAC + 3 * OSC_ARM + a] = angx
                    work[env, OSC_W_JAC + 4 * OSC_ARM + a] = angy
                    work[env, OSC_W_JAC + 5 * OSC_ARM + a] = angz

    # ── the 7 x 7 mass block, WITH the armature (see the header) ──────────
    for a in range(OSC_ARM):
        var ia = Int(rebind[Scalar[DTYPE]](refs[OSC_REF_DOF + a]))
        var ja = Int(rebind[Scalar[DTYPE]](refs[OSC_REF_JOINT + a]))
        for b in range(OSC_ARM):
            var ib = Int(rebind[Scalar[DTYPE]](refs[OSC_REF_DOF + b]))
            var v = rebind[Scalar[DTYPE]](mass[env, ia * nv + ib])
            if a == b:
                v += rebind[Scalar[DTYPE]](joints[ja, JOINT_IDX_ARMATURE])
            work[env, OSC_W_MBLK + a * OSC_ARM + b] = v
    if not _inv_into[DTYPE, L_WORK](work, env, OSC_W_MBLK, OSC_W_MINV, OSC_ARM):
        state[env, OSC_IDX_SINGULAR] = Scalar[DTYPE](1)
        return

    # ── the operational-space inertias ────────────────────────────────────
    # lambda_full_inv = J Minv J^T  (6 x 6), and its position / orientation
    # blocks taken the way `opspace_matrices` takes them: from the SEPARATE
    # products of the 3 x 7 halves, not as sub-blocks of the 6 x 6.
    _mul_into[DTYPE, L_WORK](work, env, OSC_W_JAC, OSC_W_MINV, OSC_W_TMP, 6, OSC_ARM, OSC_ARM)
    _mul_bt_into[DTYPE, L_WORK](work, env, OSC_W_TMP, OSC_W_JAC, OSC_W_LAMF, 6, OSC_ARM, 6)
    # position block: rows 0..2 of (J Minv) against rows 0..2 of J
    for i in range(3):
        for j in range(3):
            work[env, OSC_W_LAMP + i * 3 + j] = work[env, OSC_W_LAMF + i * 6 + j]
            work[env, OSC_W_LAMO + i * 3 + j] = work[
                env, OSC_W_LAMF + (3 + i) * 6 + (3 + j)
            ]
    if not _inv_into[DTYPE, L_WORK](work, env, OSC_W_LAMF, OSC_W_TMP2, 6):
        state[env, OSC_IDX_SINGULAR] = Scalar[DTYPE](1)
        return
    for k in range(36):
        work[env, OSC_W_LAMF + k] = work[env, OSC_W_TMP2 + k]
    if not _inv_into[DTYPE, L_WORK](work, env, OSC_W_LAMP, OSC_W_TMP2, 3):
        state[env, OSC_IDX_SINGULAR] = Scalar[DTYPE](1)
        return
    for k in range(9):
        work[env, OSC_W_LAMP + k] = work[env, OSC_W_TMP2 + k]
    if not _inv_into[DTYPE, L_WORK](work, env, OSC_W_LAMO, OSC_W_TMP2, 3):
        state[env, OSC_IDX_SINGULAR] = Scalar[DTYPE](1)
        return
    for k in range(9):
        work[env, OSC_W_LAMO + k] = work[env, OSC_W_TMP2 + k]

    # Jbar = Minv J^T lambda_full  (7 x 6);  N = I - Jbar J  (7 x 7)
    _mul_bt_into[DTYPE, L_WORK](work, env, OSC_W_MINV, OSC_W_JAC, OSC_W_TMP, OSC_ARM, OSC_ARM, 6)
    _mul_into[DTYPE, L_WORK](work, env, OSC_W_TMP, OSC_W_LAMF, OSC_W_JBAR, OSC_ARM, 6, 6)
    _mul_into[DTYPE, L_WORK](work, env, OSC_W_JBAR, OSC_W_JAC, OSC_W_TMP, OSC_ARM, 6, OSC_ARM)
    for i in range(OSC_ARM):
        for j in range(OSC_ARM):
            work[env, OSC_W_NUL + i * OSC_ARM + j] = (
                (Scalar[DTYPE](1) if i == j else zero)
                - rebind[Scalar[DTYPE]](work[env, OSC_W_TMP + i * OSC_ARM + j])
            )

    # ── the desired wrench ────────────────────────────────────────────────
    # The site's spatial velocity is already in VEC+6..11, from the joint walk.

    # position error -> force
    for k in range(3):
        var e = (
            rebind[Scalar[DTYPE]](state[env, OSC_IDX_GOAL_POS + k])
            - rebind[Scalar[DTYPE]](site_xpos[env, site * 3 + k])
        )
        work[env, OSC_W_VEC + k] = (
            e * kp - rebind[Scalar[DTYPE]](work[env, OSC_W_VEC + 6 + k]) * kd
        )
    # orientation error: 0.5 * sum_i current[:, i] x desired[:, i], COLUMNS
    _site_mat_into[DTYPE, L_WORK, L_B4, L_SITES](
        work, xquat, sites, env, site, site_body, OSC_W_TMP
    )
    var ex = zero
    var ey = zero
    var ez = zero
    for i in range(3):
        var cx = rebind[Scalar[DTYPE]](work[env, OSC_W_TMP + 0 * 3 + i])
        var cy = rebind[Scalar[DTYPE]](work[env, OSC_W_TMP + 1 * 3 + i])
        var cz = rebind[Scalar[DTYPE]](work[env, OSC_W_TMP + 2 * 3 + i])
        var dx = rebind[Scalar[DTYPE]](state[env, OSC_IDX_GOAL_MAT + 0 * 3 + i])
        var dy = rebind[Scalar[DTYPE]](state[env, OSC_IDX_GOAL_MAT + 1 * 3 + i])
        var dz = rebind[Scalar[DTYPE]](state[env, OSC_IDX_GOAL_MAT + 2 * 3 + i])
        ex += cy * dz - cz * dy
        ey += cz * dx - cx * dz
        ez += cx * dy - cy * dx
    var half = Scalar[DTYPE](0.5)
    work[env, OSC_W_VEC + 3] = (
        half * ex * kp - rebind[Scalar[DTYPE]](work[env, OSC_W_VEC + 6 + 3]) * kd
    )
    work[env, OSC_W_VEC + 4] = (
        half * ey * kp - rebind[Scalar[DTYPE]](work[env, OSC_W_VEC + 6 + 4]) * kd
    )
    work[env, OSC_W_VEC + 5] = (
        half * ez * kp - rebind[Scalar[DTYPE]](work[env, OSC_W_VEC + 6 + 5]) * kd
    )

    # ── decouple, project, add bias and the nullspace term ────────────────
    # wrench = [Lp F; Lo T], into VEC+6 (the velocity is spent)
    for i in range(3):
        var s = zero
        var t = zero
        for j in range(3):
            s += (
                rebind[Scalar[DTYPE]](work[env, OSC_W_LAMP + i * 3 + j])
                * rebind[Scalar[DTYPE]](work[env, OSC_W_VEC + j])
            )
            t += (
                rebind[Scalar[DTYPE]](work[env, OSC_W_LAMO + i * 3 + j])
                * rebind[Scalar[DTYPE]](work[env, OSC_W_VEC + 3 + j])
            )
        work[env, OSC_W_VEC + 6 + i] = s
        work[env, OSC_W_VEC + 6 + 3 + i] = t
    for a in range(OSC_ARM):
        var s = zero
        for r in range(OSC_WRENCH):
            s += (
                rebind[Scalar[DTYPE]](work[env, OSC_W_JAC + r * OSC_ARM + a])
                * rebind[Scalar[DTYPE]](work[env, OSC_W_VEC + 6 + r])
            )
        var ia = Int(rebind[Scalar[DTYPE]](refs[OSC_REF_DOF + a]))
        work[env, OSC_W_TMP + a] = s + rebind[Scalar[DTYPE]](bias[env, ia])
    # pose_torques = M (kp (q0 - q) - kv qd); tau += N^T pose_torques
    var ns_kp = rebind[Scalar[DTYPE]](refs[OSC_REF_NS_KP])
    var ns_kv = rebind[Scalar[DTYPE]](refs[OSC_REF_NS_KV])
    for a in range(OSC_ARM):
        var qa = Int(rebind[Scalar[DTYPE]](refs[OSC_REF_QADR + a]))
        var ia = Int(rebind[Scalar[DTYPE]](refs[OSC_REF_DOF + a]))
        work[env, OSC_W_TMP + OSC_ARM + a] = (
            ns_kp
            * (
                rebind[Scalar[DTYPE]](state[env, OSC_IDX_Q0 + a])
                - rebind[Scalar[DTYPE]](qpos[env, qa])
            )
            - ns_kv * rebind[Scalar[DTYPE]](qvel[env, ia])
        )
    for a in range(OSC_ARM):
        var s = zero
        for b in range(OSC_ARM):
            s += (
                rebind[Scalar[DTYPE]](work[env, OSC_W_MBLK + a * OSC_ARM + b])
                * rebind[Scalar[DTYPE]](work[env, OSC_W_TMP + OSC_ARM + b])
            )
        work[env, OSC_W_TMP2 + a] = s
    for a in range(OSC_ARM):
        var s = zero
        for b in range(OSC_ARM):
            # N^T: row a of N^T is column a of N
            s += (
                rebind[Scalar[DTYPE]](work[env, OSC_W_NUL + b * OSC_ARM + a])
                * rebind[Scalar[DTYPE]](work[env, OSC_W_TMP2 + b])
            )
        var tau = rebind[Scalar[DTYPE]](work[env, OSC_W_TMP + a]) + s
        var lo = rebind[Scalar[DTYPE]](refs[OSC_REF_TMIN + a])
        var hi = rebind[Scalar[DTYPE]](refs[OSC_REF_TMAX + a])
        if tau < lo:
            tau = lo
        if tau > hi:
            tau = hi
        var ai = Int(rebind[Scalar[DTYPE]](refs[OSC_REF_ACT + a]))
        ctrl[env, ai] = tau


@always_inline
def osc_gripper_gpu[
    DTYPE: DType, L_STATE: TensorLayout, L_REFS: TensorLayout, L_CTRL: TensorLayout,
    L_ACT: TensorLayout, ACT_DIM: Int,
](
    state: TileTensor[DTYPE, L_STATE, MutAnyOrigin],
    refs: TileTensor[DTYPE, L_REFS, MutAnyOrigin],
    ctrl: TileTensor[DTYPE, L_CTRL, MutAnyOrigin],
    actions: TileTensor[DTYPE, L_ACT, MutAnyOrigin],
    env: Int,
):
    """`PandaGripper.format_action` + `Manipulator.grip_action`, ONE SIM STEP.

    ⚠ PER SUBSTEP, NOT PER POLICY STEP. `Robot.control` runs every substep
    and only `set_goal` is gated on `policy_step`, so the ramp advances 0.01
    twenty-five times per action — 0.25 per policy step, four policy steps
    from neutral to an end and eight end to end."""
    var speed = rebind[Scalar[DTYPE]](refs[OSC_REF_GRIP_SPEED])
    var a = rebind[Scalar[DTYPE]](actions[env, ACT_DIM - 1])
    var zero = Scalar[DTYPE](0)
    var one = Scalar[DTYPE](1)
    var sgn = zero
    if a > zero:
        sgn = one
    elif a < zero:
        sgn = -one
    var c0 = rebind[Scalar[DTYPE]](state[env, OSC_IDX_GRIP0]) - speed * sgn
    var c1 = rebind[Scalar[DTYPE]](state[env, OSC_IDX_GRIP1]) + speed * sgn
    if c0 > one:
        c0 = one
    if c0 < -one:
        c0 = -one
    if c1 > one:
        c1 = one
    if c1 < -one:
        c1 = -one
    state[env, OSC_IDX_GRIP0] = c0
    state[env, OSC_IDX_GRIP1] = c1
    var g0 = Int(rebind[Scalar[DTYPE]](refs[OSC_REF_GRIP_ACT0]))
    var g1 = Int(rebind[Scalar[DTYPE]](refs[OSC_REF_GRIP_ACT1]))
    ctrl[env, g0] = (
        rebind[Scalar[DTYPE]](refs[OSC_REF_GRIP_BIAS0])
        + rebind[Scalar[DTYPE]](refs[OSC_REF_GRIP_W0]) * c0
    )
    ctrl[env, g1] = (
        rebind[Scalar[DTYPE]](refs[OSC_REF_GRIP_BIAS1])
        + rebind[Scalar[DTYPE]](refs[OSC_REF_GRIP_W1]) * c1
    )


# ── the host side: the record, and the dynamics refresh ───────────────────


def build_osc_refs(
    dof: List[Int], qadr: List[Int], joint: List[Int],
    tmin: List[Float64], tmax: List[Float64], act: List[Int],
    site: Int, site_body: Int,
    grip_act0: Int, grip_act1: Int,
    g0_min: Float64, g0_max: Float64, g1_min: Float64, g1_max: Float64,
    kp: Float64, damping_ratio: Float64,
    out_pos: Float64, out_ori: Float64,
    ns_kp: Float64, grip_speed: Float64,
) raises -> List[Float64]:
    """The shared `refs` record. `kd = 2 sqrt(kp) * damping_ratio` and
    `ns_kv = 2 sqrt(ns_kp)` are derived HERE so no caller can pass a pair
    that is not critically damped by accident — robosuite derives both the
    same way and neither is a free number in `osc_pose.json`."""
    if (
        len(dof) != OSC_ARM or len(qadr) != OSC_ARM or len(joint) != OSC_ARM
        or len(tmin) != OSC_ARM or len(tmax) != OSC_ARM or len(act) != OSC_ARM
    ):
        raise Error(
            "osc: every per-joint list must be " + String(OSC_ARM) + " long"
        )
    var out = List[Float64]()
    for _ in range(OSC_REF_WORDS):
        out.append(0.0)
    for a in range(OSC_ARM):
        out[OSC_REF_DOF + a] = Float64(dof[a])
        out[OSC_REF_QADR + a] = Float64(qadr[a])
        out[OSC_REF_JOINT + a] = Float64(joint[a])
        out[OSC_REF_TMIN + a] = tmin[a]
        out[OSC_REF_TMAX + a] = tmax[a]
        out[OSC_REF_ACT + a] = Float64(act[a])
    out[OSC_REF_SITE] = Float64(site)
    out[OSC_REF_SITE_BODY] = Float64(site_body)
    out[OSC_REF_GRIP_ACT0] = Float64(grip_act0)
    out[OSC_REF_GRIP_ACT1] = Float64(grip_act1)
    out[OSC_REF_GRIP_BIAS0] = 0.5 * (g0_max + g0_min)
    out[OSC_REF_GRIP_W0] = 0.5 * (g0_max - g0_min)
    out[OSC_REF_GRIP_BIAS1] = 0.5 * (g1_max + g1_min)
    out[OSC_REF_GRIP_W1] = 0.5 * (g1_max - g1_min)
    out[OSC_REF_KP] = kp
    out[OSC_REF_KD] = 2.0 * sqrt(kp) * damping_ratio
    out[OSC_REF_OUT_POS] = out_pos
    out[OSC_REF_OUT_ORI] = out_ori
    out[OSC_REF_NS_KP] = ns_kp
    out[OSC_REF_NS_KV] = 2.0 * sqrt(ns_kp)
    out[OSC_REF_GRIP_SPEED] = grip_speed
    return out^


def osc_refresh_dynamics[
    target: StaticString, DTYPE: DType, D: DimsLike, BATCH: Int = 1,
    PARALLEL: Bool = False,
](
    mut d: Data[DTYPE, D, BATCH],
    mut m: Model[DTYPE, D],
    mut scratch: DynamicsScratch[DTYPE, D, BATCH],
    ctx: Optional[DeviceContext] = None,
) raises:
    """`sim.forward()`, as much of it as OSC_POSE reads: FK products, body
    velocities, `subtree_com`, `cdof`, the CRBA mass matrix and the RNE bias.

    ⚠⚠ THIS IS THE FIRST HALF OF THE INTEGRATOR'S OWN PIPELINE, AND IT RUNS
    AGAIN INSIDE IT. robosuite pays the same cost — `BaseController.update`
    calls `sim.forward()` and then `mj_step` recomputes everything at the
    same state — so a controller that skipped it would not be the benchmark's.
    The state does not change between this and the integrator's own FK, so
    a "warm start" flag on the integrator would make the second pass free;
    that is an optimisation with its own A/B, not a correctness question, and
    it is deliberately not taken here.

    ⚠ THE ARMATURE IS NOT ADDED TO `scratch.M` BY THIS. The integrator adds
    it in a pass of its own and `osc_run_gpu` adds it to the arm block it
    extracts; adding it here would double it in whichever runs second.
    """
    forward_kinematics[target, DTYPE, D, BATCH](d, m, ctx)
    compute_body_velocities[target, DTYPE, BATCH=BATCH, PARALLEL=PARALLEL](d, m, ctx)
    compute_subtree_com[target, DTYPE, BATCH=BATCH](d, m, ctx)
    compute_cdof[target, DTYPE, BATCH=BATCH, PARALLEL=PARALLEL](d, m, scratch, ctx)
    compute_mass_matrix[
        target, DTYPE, BATCH=BATCH, PARALLEL=PARALLEL, TREEWALK=True
    ](d, m, scratch, ctx)
    compute_bias_forces_rne[target, DTYPE, BATCH=BATCH, PARALLEL=PARALLEL](
        d, m, scratch, ctx
    )
