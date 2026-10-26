"""Stateful Euler integrator over per-field tensors (migration P2 pilot).

`EulerIntegrator` is the stateful replacement for the stateless
`EulerIntegrator.step_gpu` + caller-provided workspace slab: the struct OWNS
its `DynamicsScratch` (allocated once, reused every step) and sequences the
single-source stage ports into a full contact-free step:

    FK -> subtree_com -> cdof -> CRBA -> +armature -> LDL factor -> RNE ->
    fnet assembly (qfrc - bias - damping - stiffness - frictionloss) ->
    LDL solve -> qacc writeback -> finalize (implicit-damping re-solve +
    velocity/position integration + quaternion renormalize)

Assembly/finalize arithmetic is verbatim from the legacy Euler
`step_kernel` (:744) / `step_finalize_kernel` (:2140). Body velocities
(xvel/xangvel, consumed by env obs and future fluid forces) run right after
FK, matching legacy step order. Deliberately NOT yet ported (raise on use):
fluid forces (density/viscosity > 0), contacts, limits, constraint solving
— the P4 scope.

vs legacy: 9 small per-stage kernel launches instead of 2 fused monoliths —
each stage is independently gated; fusion is a later NVIDIA perf lever."""

from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceContext
from max.gpu.sync import barrier
from layout import TileTensor, TensorLayout
from noeira.core.tiles import RowMajor, Tile

from std.math import sqrt
from ..kinematics.quat_math import quat_integrate, quat_normalize, atan2_device
from ..sensors.eval import (
    sensor_pos,
    sensor_vel,
    sensor_acc,
)
from ..kinematics.forward_kinematics import (
    forward_kinematics,
    compute_body_velocities,
)
from ..dynamics.subtree_com import compute_subtree_com
from ..dynamics.sensor_lengths import compute_tendon_lengths
from ..dynamics.cdof import compute_cdof
from ..dynamics.mass_matrix import compute_mass_matrix
from std.time import perf_counter_ns
from ..fields.scratch import Scratch, cap
from ..dynamics.ldl import (
    _dof_ancestors,
    ldl_factor,
    ldl_solve,
    compute_m_inv,
    _ldl_factor_env,
    _ldl_solve_env,
    _ldl_factor_tree_env,
    _ldl_solve_tree_env,
)
from ..constraints.limits import solve_limits
from ..constraints.friction_dof import solve_friction
from ..constraints.contact_solve import solve_contacts
from ..solver.newton_solve import solve_newton
from ..solver.warmstart import save_qacc_warmstart
from ..solver.je_budget import je_ws_size
from ..solver.cg_solve import solve_cg
from ..solver.island_pgs_solve import solve_island_pgs
from ..collision.broadphase_sap import detect_contacts_auto
from ..types import ConeType
from ..dynamics.rne import compute_bias_forces_rne
from ..dynamics.rne_post import compute_rne_post
from ..dynamics.fluid_forces import compute_fluid_forces
from ..dynamics.gravcomp import compute_gravcomp_forces
from ..joint_types import JNT_FREE, JNT_BALL, JNT_HINGE, JNT_SLIDE
from ..fields import (
    Data,
    Model,
    DynamicsScratch,
    ContactScratch,
    Dims,
    DimsLike,
    AsStatic,
    DYN1,
    DYN2,
    rl1,
    rl2,
)
from ..gpu.constants import (
    MJ_MAXVAL,
    MODEL_JOINT_SIZE,
    MODEL_TREE_SIZE,
    MODEL_META_IDX_TIMESTEP,
    MODEL_META_IDX_EULERDAMP_DISABLED,
    MODEL_META_IDX_DENSITY,
    MODEL_META_IDX_VISCOSITY,
    JOINT_IDX_TYPE,
    JOINT_IDX_QPOS_ADR,
    JOINT_IDX_DOF_ADR,
    JOINT_IDX_ARMATURE,
    JOINT_IDX_DAMPING,
    JOINT_IDX_STIFFNESS,
    JOINT_IDX_SPRINGREF,
    JOINT_IDX_FRICTIONLOSS,
    MODEL_META_IDX_NTREE,
)

comptime EU_TPB: Int = 64


# ── armature: M diagonal += armature (verbatim step_kernel 6b) ────────────

# ⚠ A STAGE PROBE FOR THE CPU EULER STEP, off and free by default (the twin of
# `newton_solve._CPU_PROBE`). On, `step["cpu"]` prints one `[eprobe]` line per
# step with nanoseconds per stage: FK, body velocities, subtree COM, cdof,
# CRBA, armature, LDL factor, M^-1, RNE, fnet assembly, LDL solve, writeback,
# collision, the constraint solve, warm-start save, finalize. Fold with awk.
comptime _EULER_PROBE: Bool = False


@always_inline
def _armature_env[
    DTYPE: DType,
    D: DimsLike,
    L_JOINTS: TensorLayout,
    L_M: TensorLayout](
    env: Int,
    dims: D,
    joints: TileTensor[DTYPE, L_JOINTS, MutAnyOrigin],
    M: TileTensor[DTYPE, L_M, MutAnyOrigin],
):
    var nv = dims.get_nv()
    var njoint = dims.get_njoint()
    for j in range(njoint):
        var jnt_type = Int(joints[j, JOINT_IDX_TYPE])
        var dof_adr = Int(joints[j, JOINT_IDX_DOF_ADR])
        var arm = joints[j, JOINT_IDX_ARMATURE]
        var diag_add = arm
        if jnt_type == JNT_FREE:
            for d in range(6):
                M[env, (dof_adr + d) * nv + (dof_adr + d)] += diag_add
        elif jnt_type == JNT_BALL:
            for d in range(3):
                M[env, (dof_adr + d) * nv + (dof_adr + d)] += diag_add
        else:
            M[env, dof_adr * nv + dof_adr] += diag_add


# ── fnet assembly: qfrc - bias - damping - stiffness - friction ───────────
# (verbatim step_kernel 9 + 8b; fluid 8c NOT ported — guarded at step())
@always_inline
def _fnet_passive_env[
    DTYPE: DType,
    D: DimsLike,
    L_QPOS: TensorLayout,
    L_QVEL: TensorLayout,
    L_JOINTS: TensorLayout](
    env: Int,
    dims: D,
    qpos: TileTensor[DTYPE, L_QPOS, MutAnyOrigin],
    qvel: TileTensor[DTYPE, L_QVEL, MutAnyOrigin],
    qfrc: TileTensor[DTYPE, L_QVEL, MutAnyOrigin],
    joints: TileTensor[DTYPE, L_JOINTS, MutAnyOrigin],
    bias: TileTensor[DTYPE, L_QVEL, MutAnyOrigin],
    fnet: TileTensor[DTYPE, L_QVEL, MutAnyOrigin],
):
    var nv = dims.get_nv()
    var njoint = dims.get_njoint()
    # f_net = qfrc - bias
    for i in range(nv):
        var qfrc_v = rebind[Scalar[DTYPE]](qfrc[env, i])
        var bias_val = rebind[Scalar[DTYPE]](bias[env, i])
        fnet[env, i] = qfrc_v - bias_val

    # Damping: f -= damping * qvel (explicit part)
    for j in range(njoint):
        var jnt_type_d = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_TYPE]))
        var dof_adr_d = Int(
            rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_DOF_ADR])
        )
        var damp_d = rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_DAMPING])
        # ⚠ `!= 0`, NOT `> 0` (AUD-42). `mj_springdamper` gates the damper on
        # `damping != 0 || !mju_isZero(poly)` (engine_passive.c:722-735), so a
        # NEGATIVE damping injects energy on purpose. `> 0` silently made such
        # a joint undamped, which looks like a stable model and is a different
        # one.
        if damp_d != Scalar[DTYPE](0):
            var nd = 1
            if jnt_type_d == JNT_FREE:
                nd = 6
            elif jnt_type_d == JNT_BALL:
                nd = 3
            for d in range(nd):
                var v = rebind[Scalar[DTYPE]](qvel[env, dof_adr_d + d])
                var cur = rebind[Scalar[DTYPE]](fnet[env, dof_adr_d + d])
                fnet[env, dof_adr_d + d] = cur - damp_d * v

    # Stiffness + frictionloss
    for j in range(njoint):
        var jnt_type = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_TYPE]))
        var dof_adr = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_DOF_ADR]))
        var qpos_adr = Int(
            rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_QPOS_ADR])
        )
        var stiff = rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_STIFFNESS])
        var sref = rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_SPRINGREF])
        var floss = rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_FRICTIONLOSS])
        if stiff > Scalar[DTYPE](0):
            if jnt_type == JNT_BALL:
                # ⚠ AUD-43. MuJoCo: `mju_subQuat(dif, quat, qpos_spring)` with
                # `qpos_spring` = the identity for a ball joint, then
                # `torque = -stiffness * dif` (engine_passive.c:696-707).
                # `dif` is the rotation's axis-angle vector — NOT the
                # quaternion components, which is what this loop used to
                # subtract `springref` from. `mju_quat2Vel` with the `> pi`
                # wrap; qpos holds (w, x, y, z).
                var qw = rebind[Scalar[DTYPE]](qpos[env, qpos_adr + 0])
                var qx = rebind[Scalar[DTYPE]](qpos[env, qpos_adr + 1])
                var qy = rebind[Scalar[DTYPE]](qpos[env, qpos_adr + 2])
                var qz = rebind[Scalar[DTYPE]](qpos[env, qpos_adr + 3])
                var qn = sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
                if qn > Scalar[DTYPE](0):
                    qw /= qn
                    qx /= qn
                    qy /= qn
                    qz /= qn
                var s2 = sqrt(qx * qx + qy * qy + qz * qz)
                var angle = Scalar[DTYPE](2) * atan2_device[DTYPE](s2, qw)
                if angle > Scalar[DTYPE](3.141592653589793):
                    angle -= Scalar[DTYPE](6.283185307179586)
                if s2 > Scalar[DTYPE](0):
                    var sc = angle / s2
                    var c0 = rebind[Scalar[DTYPE]](fnet[env, dof_adr + 0])
                    var c1 = rebind[Scalar[DTYPE]](fnet[env, dof_adr + 1])
                    var c2 = rebind[Scalar[DTYPE]](fnet[env, dof_adr + 2])
                    fnet[env, dof_adr + 0] = c0 - stiff * qx * sc
                    fnet[env, dof_adr + 1] = c1 - stiff * qy * sc
                    fnet[env, dof_adr + 2] = c2 - stiff * qz * sc
            elif jnt_type == JNT_FREE:
                # A free-joint spring pulls the body to its XML pose
                # (`qpos_spring = qpos0`); the pose is not in this routine's
                # inputs, and the parser refuses the model (AUD-43).
                pass
            else:
                var qpos_d = rebind[Scalar[DTYPE]](qpos[env, qpos_adr])
                var cur = rebind[Scalar[DTYPE]](fnet[env, dof_adr])
                fnet[env, dof_adr] = cur - stiff * (qpos_d - sref)
        # frictionloss is NOT a passive force. It used to be applied here as an
        # explicit Coulomb force with a 1e-4 velocity deadband, which cannot
        # arrest motion — it overshoots zero and settles into a period-2 limit
        # cycle, so a joint that should stop dead spins forever (dm_control's
        # finger spinner: MuJoCo 1e-17 rad/s, ours a bit-constant +-0.0329).
        # MuJoCo solves it as a CONSTRAINT ROW (`mjCNSTR_FRICTION_DOF`) whose
        # force is bounded by frictionloss rather than fixed at it; that lives
        # in `constraints/friction_dof.mojo` and runs beside the limit rows in
        # every solver. `floss` is read there, not here.
        _ = floss


# ── qacc writeback: state qacc + qacc_constrained = qacc_ws ───────────────
@always_inline
def _qacc_writeback_env[
    DTYPE: DType,
    D: DimsLike,
    L_QACC_WS: TensorLayout](
    env: Int,
    dims: D,
    qacc_ws: TileTensor[DTYPE, L_QACC_WS, MutAnyOrigin],
    qacc: TileTensor[DTYPE, L_QACC_WS, MutAnyOrigin],
    qacc_constrained: TileTensor[DTYPE, L_QACC_WS, MutAnyOrigin],
):
    var nv = dims.get_nv()
    for i in range(nv):
        var qacc_val = rebind[Scalar[DTYPE]](qacc_ws[env, i])
        qacc[env, i] = qacc_val
        qacc_constrained[env, i] = qacc_val


# ── the integrate half of finalize, shared by both damping branches ───────
#
# ⚠ EXTRACTED, NOT COPIED. `mj_EulerSkip` has two branches — explicit when
# `mjDSBL_EULERDAMP` is set (or no dof is damped) and implicit otherwise —
# and they differ ONLY in how `qacc_final` is obtained. Writing the velocity
# and position update once means the two cannot drift; this tree has been
# bitten by a rule written inline twice before.
@always_inline
def _finalize_integrate_env[
    DTYPE: DType,
    DIMS: DimsLike,
    L_QPOS: TensorLayout,
    L_QVEL: TensorLayout,
    L_JOINTS: TensorLayout,
](
    env: Int,
    dt: Scalar[DTYPE],
    dims: DIMS,
    qpos: TileTensor[DTYPE, L_QPOS, MutAnyOrigin],
    qvel: TileTensor[DTYPE, L_QVEL, MutAnyOrigin],
    qacc: TileTensor[DTYPE, L_QVEL, MutAnyOrigin],
    joints: TileTensor[DTYPE, L_JOINTS, MutAnyOrigin],
    qacc_ws: TileTensor[DTYPE, L_QVEL, MutAnyOrigin],
):
    var nv = dims.get_nv()
    var njoint = dims.get_njoint()
    # Step 5: v_new = v_old + dt * qacc_final.
    #
    # ⚠ THE BOUND IS `mjMAXVAL`, NOT A STABILITY BUDGET. MuJoCo's only
    # velocity guard is `mj_checkVel`, which WARNS and resets the state
    # when a dof goes NaN/inf or past 1e10; it never rescales one. A
    # tighter saturation here is a silent physics change that only shows
    # up on the models fast enough to reach it.
    for i in range(nv):
        var old_qvel = rebind[Scalar[DTYPE]](qvel[env, i])
        var qacc_final = rebind[Scalar[DTYPE]](qacc_ws[env, i])
        qacc[env, i] = qacc_final
        var qvel_new = old_qvel + qacc_final * dt
        var qvel_max = Scalar[DTYPE](MJ_MAXVAL)
        if qvel_new != qvel_new:  # NaN guard
            qvel_new = Scalar[DTYPE](0.0)
        elif qvel_new > qvel_max:
            qvel_new = qvel_max
        elif qvel_new < -qvel_max:
            qvel_new = -qvel_max
        qvel[env, i] = qvel_new

    # Integrate position (quaternion-aware for FREE; BALL not handled, as
    # in the legacy finalize)
    for j in range(njoint):
        var jnt_type = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_TYPE]))
        var jnt_qpos_adr = Int(
            rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_QPOS_ADR])
        )
        var jnt_dof_adr = Int(
            rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_DOF_ADR])
        )

        if jnt_type == JNT_FREE:
            for d in range(3):
                var qp = rebind[Scalar[DTYPE]](qpos[env, jnt_qpos_adr + d])
                var qv = rebind[Scalar[DTYPE]](qvel[env, jnt_dof_adr + d])
                qpos[env, jnt_qpos_adr + d] = qp + qv * dt
            # FREE-joint qpos stores [tx,ty,tz, qw,qx,qy,qz] — w FIRST at +3
            var qw = rebind[Scalar[DTYPE]](qpos[env, jnt_qpos_adr + 3])
            var qx = rebind[Scalar[DTYPE]](qpos[env, jnt_qpos_adr + 4])
            var qy = rebind[Scalar[DTYPE]](qpos[env, jnt_qpos_adr + 5])
            var qz = rebind[Scalar[DTYPE]](qpos[env, jnt_qpos_adr + 6])
            var wx = rebind[Scalar[DTYPE]](qvel[env, jnt_dof_adr + 3])
            var wy = rebind[Scalar[DTYPE]](qvel[env, jnt_dof_adr + 4])
            var wz = rebind[Scalar[DTYPE]](qvel[env, jnt_dof_adr + 5])
            var result = quat_integrate(qx, qy, qz, qw, wx, wy, wz, dt)
            var norm = quat_normalize(
                result[0], result[1], result[2], result[3]
            )
            qpos[env, jnt_qpos_adr + 3] = norm[3]  # qw
            qpos[env, jnt_qpos_adr + 4] = norm[0]  # qx
            qpos[env, jnt_qpos_adr + 5] = norm[1]  # qy
            qpos[env, jnt_qpos_adr + 6] = norm[2]  # qz

        # ⚠⚠ THIS BRANCH DID NOT EXIST, so a `<joint type="ball">` NEVER
        # MOVED. Its three DOFs accumulated velocity that nothing applied to
        # `qpos`, while the quaternion stayed at whatever the reset left —
        # a joint that is free in the mass matrix and frozen on screen.
        # `kinematics/integrate_pos.mojo` has carried the correct body since
        # it was written and has no callers; the integrators each roll their
        # own qpos loop and only FREE and HINGE/SLIDE were ever transcribed.
        #
        # ⚠ MuJoCo FALLS THROUGH from FREE into BALL (`mj_integratePos`) —
        # the free joint's rotation IS this update on shifted addresses, which
        # is why the two are the same four lines and must stay that way.
        # qpos holds the quaternion w FIRST; `quat_math` takes and returns
        # (x, y, z, w).
        elif jnt_type == JNT_BALL:
            var bqw = rebind[Scalar[DTYPE]](qpos[env, jnt_qpos_adr + 0])
            var bqx = rebind[Scalar[DTYPE]](qpos[env, jnt_qpos_adr + 1])
            var bqy = rebind[Scalar[DTYPE]](qpos[env, jnt_qpos_adr + 2])
            var bqz = rebind[Scalar[DTYPE]](qpos[env, jnt_qpos_adr + 3])
            var bwx = rebind[Scalar[DTYPE]](qvel[env, jnt_dof_adr + 0])
            var bwy = rebind[Scalar[DTYPE]](qvel[env, jnt_dof_adr + 1])
            var bwz = rebind[Scalar[DTYPE]](qvel[env, jnt_dof_adr + 2])
            var bres = quat_integrate(bqx, bqy, bqz, bqw, bwx, bwy, bwz, dt)
            var bnorm = quat_normalize(bres[0], bres[1], bres[2], bres[3])
            qpos[env, jnt_qpos_adr + 0] = bnorm[3]  # qw
            qpos[env, jnt_qpos_adr + 1] = bnorm[0]  # qx
            qpos[env, jnt_qpos_adr + 2] = bnorm[1]  # qy
            qpos[env, jnt_qpos_adr + 3] = bnorm[2]  # qz

        elif jnt_type == JNT_HINGE or jnt_type == JNT_SLIDE:
            var qp = rebind[Scalar[DTYPE]](qpos[env, jnt_qpos_adr])
            var qv = rebind[Scalar[DTYPE]](qvel[env, jnt_dof_adr])
            qpos[env, jnt_qpos_adr] = qp + qv * dt


# ── finalize: implicit-damping re-solve + integrate (verbatim :2140) ──────


@always_inline
def _finalize_damping_env[
    DTYPE: DType,
    DIMS: DimsLike,
    L_JOINTS: TensorLayout,
    L_M: TensorLayout,
](
    env: Int,
    dt: Scalar[DTYPE],
    dims: DIMS,
    joints: TileTensor[DTYPE, L_JOINTS, MutAnyOrigin],
    M: TileTensor[DTYPE, L_M, MutAnyOrigin],
):
    """`M_hat = M + dt * diag(damping)`, in place — `mj_Euler`'s implicit
    damping (engine_forward.c, `eulerdamp`). ONE spelling, shared by the dense
    and the tree-ordered finalize."""
    var nv = dims.get_nv()
    # ⚠⚠ TWO PASSES, BECAUSE MuJoCo DETECTS AND APPLIES BY DIFFERENT TESTS
    # (AUD-42). `mj_EulerSkip` (engine_forward.c) first scans for ANY dof with
    # `dof_damping[i] > 0` — strictly positive — and takes the explicit branch
    # if it finds none. Having found one, the `qH += h*diag(B)` loop that
    # follows runs over EVERY dof with no test at all, so a negative damping
    # in a model that also has a positive one IS added. A single `damp > 0`
    # per joint collapses both tests into the wrong one and drops it.
    var any_positive = False
    for j in range(dims.get_njoint()):
        if rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_DAMPING]) > Scalar[
            DTYPE
        ](0):
            any_positive = True
            break
    if not any_positive:
        return
    for j in range(dims.get_njoint()):
        var jnt_type = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_TYPE]))
        var dof_adr = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_DOF_ADR]))
        var damp = rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_DAMPING])
        if damp != Scalar[DTYPE](0):
            var nd = 1
            if jnt_type == JNT_FREE:
                nd = 6
            elif jnt_type == JNT_BALL:
                nd = 3
            for d in range(nd):
                M[env, (dof_adr + d) * nv + (dof_adr + d)] += dt * damp


@always_inline
def _finalize_tree_env[
    DTYPE: DType,
    DIMS: DimsLike,
    L_QPOS: TensorLayout,
    L_QVEL: TensorLayout,
    L_JOINTS: TensorLayout,
    L_M: TensorLayout,
    L_P: TensorLayout,
](
    env: Int,
    dt: Scalar[DTYPE],
    dims: DIMS,
    qpos: TileTensor[DTYPE, L_QPOS, MutAnyOrigin],
    qvel: TileTensor[DTYPE, L_QVEL, MutAnyOrigin],
    qacc: TileTensor[DTYPE, L_QVEL, MutAnyOrigin],
    joints: TileTensor[DTYPE, L_JOINTS, MutAnyOrigin],
    M: TileTensor[DTYPE, L_M, MutAnyOrigin],
    L: TileTensor[DTYPE, L_M, MutAnyOrigin],
    D: TileTensor[DTYPE, L_QVEL, MutAnyOrigin],
    fnet: TileTensor[DTYPE, L_QVEL, MutAnyOrigin],
    qacc_ws: TileTensor[DTYPE, L_QVEL, MutAnyOrigin],
    qacc_constrained: TileTensor[DTYPE, L_QVEL, MutAnyOrigin],
    dofp: TileTensor[DTYPE, L_P, MutAnyOrigin],
    eulerdamp_off: Bool,
):
    """`_finalize_env` on the tree-ordered LDL (CPU, `NTREE > 0`).

    The dense twin below was the one caller of the forward-elimination LDL
    that §13.10 did not switch — it calls `_ldl_factor_env` directly, not
    through the dispatcher — and formed `M * qacc` as a dense nv^2 matvec:
    34 us of humanoid_CMU's 208 and 71 of dog's 663 against MuJoCo's
    `mj_Euler` at 4.6 / 6.8 (PERFORMANCE.md §13.14, item 3). Here the
    right-hand side walks M's tree sparsity (row i is nonzero at i's
    ancestors, plus the symmetric entries — `mj_mulM`'s shape) and the
    factorisation and solve are `mj_factorI` / `mj_solveLD` on our storage.
    Not bit-exact against the dense twin: a different rounding of the same
    `M_hat^-1 (M qacc)`, gated by the Euler-model MuJoCo tests."""
    var nv = dims.get_nv()
    if eulerdamp_off:
        for i in range(nv):
            qacc_ws[env, i] = rebind[Scalar[DTYPE]](qacc_constrained[env, i])
        _finalize_integrate_env(
            env, dt, dims, qpos, qvel, qacc, joints, qacc_ws
        )
        return
    comptime V_CAP = cap[DIMS.NV]()
    comptime A_CAP = V_CAP * V_CAP
    var par = Scratch[Int, V_CAP](nv, uninitialized=0)
    for i in range(nv):
        par[i] = Int(dofp[i])
    # rhs = M * qacc_constrained over M's sparsity: the diagonal, then each
    # (i, ancestor j) entry once for both rows it belongs to. Ancestor lists
    # and pointers in place of the `par` chase through `LayoutTensor`
    # (PERFORMANCE.md §13.26); the accumulation order per entry — diagonal,
    # own ancestors parent-first, then descendants ascending — is the walk's,
    # so this is the same arithmetic.
    var dep = Scratch[Int, V_CAP](nv, uninitialized=0)
    var anc = Scratch[Int, A_CAP](nv * nv, uninitialized=0)
    _dof_ancestors[V_CAP, A_CAP](nv, par, dep, anc)
    var Mp = M.ptr.unsafe_offset(env * nv * nv)
    var fq = Scratch[Scalar[DTYPE], V_CAP](nv, uninitialized=Scalar[DTYPE](0))
    var qc = Scratch[Scalar[DTYPE], V_CAP](nv, uninitialized=Scalar[DTYPE](0))
    var fp = fq.unsafe_ptr()
    var qp = qc.unsafe_ptr()
    for i in range(nv):
        qp[unsafe_offset=i] = rebind[Scalar[DTYPE]](qacc_constrained[env, i])
        fp[unsafe_offset=i] = Mp[unsafe_offset=i * nv + i] * qp[unsafe_offset=i]
    for i in range(nv):
        var qi = qp[unsafe_offset=i]
        var ri = i * nv
        var acc = fp[unsafe_offset=i]
        for a in range(dep[i] - 1, -1, -1):
            var j = anc[ri + a]
            var mij = Mp[unsafe_offset=ri + j]
            acc = acc + mij * qp[unsafe_offset=j]
            fp[unsafe_offset=j] = fp[unsafe_offset=j] + mij * qi
        fp[unsafe_offset=i] = acc
    for i in range(nv):
        fnet[env, i] = fp[unsafe_offset=i]
    _finalize_damping_env[DTYPE](env, dt, dims, joints, M)
    _ldl_factor_tree_env(env, dims, M, L, D, dofp)
    _ldl_solve_tree_env(env, dims, L, D, fnet, qacc_ws, dofp)
    _finalize_integrate_env(
        env, dt, dims, qpos, qvel, qacc, joints, qacc_ws
    )


@always_inline
def _finalize_env[
    DTYPE: DType,
    DIMS: DimsLike,
    L_QPOS: TensorLayout,
    L_QVEL: TensorLayout,
    L_JOINTS: TensorLayout,
    L_M: TensorLayout,
    L_TREE: TensorLayout](
    env: Int,
    dt: Scalar[DTYPE],
    dims: DIMS,
    qpos: TileTensor[DTYPE, L_QPOS, MutAnyOrigin],
    qvel: TileTensor[DTYPE, L_QVEL, MutAnyOrigin],
    qacc: TileTensor[DTYPE, L_QVEL, MutAnyOrigin],
    joints: TileTensor[DTYPE, L_JOINTS, MutAnyOrigin],
    M: TileTensor[DTYPE, L_M, MutAnyOrigin],
    L: TileTensor[DTYPE, L_M, MutAnyOrigin],
    D: TileTensor[DTYPE, L_QVEL, MutAnyOrigin],
    fnet: TileTensor[DTYPE, L_QVEL, MutAnyOrigin],
    qacc_ws: TileTensor[DTYPE, L_QVEL, MutAnyOrigin],
    qacc_constrained: TileTensor[DTYPE, L_QVEL, MutAnyOrigin],
    # ⚠ M's DIAGONAL BLOCKS, for the `M_hat` re-factorisation below. Threaded
    # and not yet read — see `dynamics/ldl.mojo`'s header. ⚠⚠ NOTE THIS IS A
    # SECOND DENSE LDL PER STEP, distinct from the `ldl_factor` in `step`:
    # `M_hat = M + dt*diag(damping)` adds to the DIAGONAL only, so it has the
    # same block structure and the same O(nv^3) cost.
    trees: TileTensor[DTYPE, L_TREE, MutAnyOrigin],
    # ⚠⚠ `<option><flag eulerdamp="disable"/></option>` — mjDSBL_EULERDAMP.
    # `mj_EulerSkip` puts its ENTIRE damping scan behind this flag and, when
    # it is set, integrates velocity explicitly: `qvel += h * qacc`, with the
    # `(M + h*diag(B))` solve below skipped, not merely fed a zero B. The two
    # are far apart whenever `h*B` is comparable to `M`'s diagonal — on
    # tetheria (`h = 0.01`, `dof_damping = 0.1`, `M_ii ~ 1.6e-03`) the solve
    # returns 61.5% of the explicit velocity, which was that model's whole
    # remaining residual.
    eulerdamp_off: Bool,
):
    var nv = dims.get_nv()
    var njoint = dims.get_njoint()
    if eulerdamp_off:
        # Explicit velocity integration — `mj_EulerSkip`'s `if (!dof_damping)`
        # branch, which copies `d->qacc` and never touches `qH`.
        for i in range(nv):
            qacc_ws[env, i] = rebind[Scalar[DTYPE]](qacc_constrained[env, i])
        _finalize_integrate_env(
            env, dt, dims, qpos, qvel, qacc, joints, qacc_ws
        )
        return
    # Step 1: rhs = M * qacc_constrained (into fnet)
    for i in range(nv):
        var sum = Scalar[DTYPE](0)
        for j in range(nv):
            var M_ij = rebind[Scalar[DTYPE]](M[env, i * nv + j])
            var qacc_j = rebind[Scalar[DTYPE]](qacc_constrained[env, j])
            sum += M_ij * qacc_j
        fnet[env, i] = sum

    # Step 2: M_hat = M + dt*D (damping to diagonal)
    _finalize_damping_env[DTYPE](env, dt, dims, joints, M)

    # Step 3+4: re-factor M_hat, solve qacc_final = M_hat^{-1} * rhs
    _ldl_factor_env(env, dims, M, L, D, trees)
    _ldl_solve_env(env, dims, L, D, fnet, qacc_ws, trees)

    _finalize_integrate_env(
        env, dt, dims, qpos, qvel, qacc, joints, qacc_ws
    )


# ── launchable kernels ────────────────────────────────────────────────────
def _armature_kernel[
    DTYPE: DType, NV: Int, NJOINT: Int, BATCH: Int
](
    joints: Tile[DTYPE, NJOINT, MODEL_JOINT_SIZE],
    M: Tile[DTYPE, BATCH, NV * NV],
):
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    _armature_env[DTYPE](env, Dims[nv=NV, njoint=NJOINT](), joints, M)


def _fnet_passive_kernel[
    DTYPE: DType, NQ: Int, NV: Int, NJOINT: Int, BATCH: Int
](
    qpos: Tile[DTYPE, BATCH, NQ],
    qvel: Tile[DTYPE, BATCH, NV],
    qfrc: Tile[DTYPE, BATCH, NV],
    joints: Tile[DTYPE, NJOINT, MODEL_JOINT_SIZE],
    bias: Tile[DTYPE, BATCH, NV],
    fnet: Tile[DTYPE, BATCH, NV],
):
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    _fnet_passive_env[DTYPE](
        env, Dims[nq=NQ, nv=NV, njoint=NJOINT](), qpos, qvel, qfrc, joints, bias, fnet
    )


def _qacc_writeback_kernel[
    DTYPE: DType, NV: Int, BATCH: Int
](
    qacc_ws: Tile[DTYPE, BATCH, NV],
    qacc: Tile[DTYPE, BATCH, NV],
    qacc_constrained: Tile[DTYPE, BATCH, NV],
):
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    _qacc_writeback_env[DTYPE](
        env, Dims[nv=NV](), qacc_ws, qacc, qacc_constrained
    )


def _finalize_kernel[
    DTYPE: DType, NQ: Int, NV: Int, NJOINT: Int, BATCH: Int
](
    dt: Scalar[DTYPE],
    qpos: Tile[DTYPE, BATCH, NQ],
    qvel: Tile[DTYPE, BATCH, NV],
    qacc: Tile[DTYPE, BATCH, NV],
    joints: Tile[DTYPE, NJOINT, MODEL_JOINT_SIZE],
    M: Tile[DTYPE, BATCH, NV * NV],
    L: Tile[DTYPE, BATCH, NV * NV],
    D: Tile[DTYPE, BATCH, NV],
    fnet: Tile[DTYPE, BATCH, NV],
    qacc_ws: Tile[DTYPE, BATCH, NV],
    qacc_constrained: Tile[DTYPE, BATCH, NV],
    trees: Tile[DTYPE, NV * MODEL_TREE_SIZE],
    # ⚠ A SCALAR, NOT A `Bool`, so the launch does not need a second
    # argument kind — every other model constant reaching a kernel here
    # arrives the same way.
    eulerdamp_off: Scalar[DTYPE],
):
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    _finalize_env[DTYPE](
        env, dt, Dims[nq=NQ, nv=NV, njoint=NJOINT](), qpos, qvel, qacc, joints, M, L, D, fnet, qacc_ws,
        qacc_constrained, trees, eulerdamp_off != Scalar[DTYPE](0),
    )


# ── the stateful integrator ───────────────────────────────────────────────
# ⚠ THE FINALIZE IN FOUR LAUNCHES, NOT ONE — a measurement's consequence.
# `_finalize_kernel` ran the whole finalize on ONE THREAD PER ENV: a dense
# nv^2 matvec, the damping diagonal, `_ldl_factor_env` and `_ldl_solve_env`.
# On the RTX 5090 that launch read 1028 us at the k=13 park scene (nv=84) —
# 22% of the Euler step, second to Newton, growing like nv^3 — where the
# cooperative LDL factor the step already launches costs 73 us on the same
# matrix and the solve 115 (PERFORMANCE.md §13.39). So: a block-per-env
# right-hand-side kernel (row-parallel matvec, then the damping diagonal
# over joints), then the SAME `ldl_factor` / `ldl_solve` dispatchers the
# step uses — `scratch.M` is `M_hat` in place by then, `scratch.fnet` the rhs,
# `scratch.qacc_ws` the result, exactly the fields the solve reads and writes
# — then a per-env integrate kernel. Every piece is arithmetic that already
# runs on the other path (the dense matvec's ascending row sum, the
# damping's in-place add, the factor and solve the dispatchers own), so the
# result is the old kernel's bit for bit; `test_tape_gpu_parity` and
# `test_device_placement` gate it. False = the old single launch.
comptime EULER_FINALIZE_SPLIT: Bool = True


def _finalize_rhs_kernel[
    DTYPE: DType, NV: Int, NJOINT: Int, BATCH: Int
](
    dt: Scalar[DTYPE],
    joints: Tile[DTYPE, NJOINT, MODEL_JOINT_SIZE],
    M: Tile[DTYPE, BATCH, NV * NV],
    fnet: Tile[DTYPE, BATCH, NV],
    qacc_constrained: Tile[DTYPE, BATCH, NV],
):
    """One block per env, `NV` threads: `fnet = M * qacc_constrained` one row
    per thread (the dense path's ascending sum, so the same bits), then
    `M_hat = M + dt * diag(damping)` with the joints split over threads —
    every joint owns its dofs' diagonal entries, so no two threads write one
    slot. The barrier between them is what keeps the matvec reading `M`."""
    var env = Int(block_idx.x)
    var i = Int(thread_idx.x)
    if env >= BATCH:
        return
    comptime NVn = NV if NV > 0 else 1
    if i < NV:
        var sum = Scalar[DTYPE](0)
        for j in range(NV):
            sum += rebind[Scalar[DTYPE]](M[env, i * NV + j]) * rebind[
                Scalar[DTYPE]
            ](qacc_constrained[env, j])
        fnet[env, i] = sum
    barrier()
    # The same two tests as the per-env builder above (AUD-42): detect on
    # `> 0` over ALL joints, then apply to every joint whose damping is
    # non-zero. The scan is whole-array in each thread rather than strided —
    # it is a read-only reduction over NJOINT scalars and the strided form
    # would need a barrier to share its answer.
    var any_positive = False
    for j in range(NJOINT):
        if rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_DAMPING]) > Scalar[
            DTYPE
        ](0):
            any_positive = True
            break
    for j in range(i, NJOINT, NVn):
        if not any_positive:
            break
        var jnt_type = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_TYPE]))
        var dof_adr = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_DOF_ADR]))
        var damp = rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_DAMPING])
        if damp != Scalar[DTYPE](0):
            var nd = 1
            if jnt_type == JNT_FREE:
                nd = 6
            elif jnt_type == JNT_BALL:
                nd = 3
            for d in range(nd):
                M[env, (dof_adr + d) * NV + (dof_adr + d)] += dt * damp


def _finalize_integrate_kernel[
    DTYPE: DType, NQ: Int, NV: Int, NJOINT: Int, BATCH: Int
](
    dt: Scalar[DTYPE],
    qpos: Tile[DTYPE, BATCH, NQ],
    qvel: Tile[DTYPE, BATCH, NV],
    qacc: Tile[DTYPE, BATCH, NV],
    joints: Tile[DTYPE, NJOINT, MODEL_JOINT_SIZE],
    qacc_ws: Tile[DTYPE, BATCH, NV],
    qacc_constrained: Tile[DTYPE, BATCH, NV],
    # 1 = `eulerdamp` disabled: integrate the constrained acceleration as is,
    # as `_finalize_env`'s first branch does.
    use_constrained: Scalar[DTYPE],
):
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    if use_constrained != Scalar[DTYPE](0):
        for i in range(NV):
            qacc_ws[env, i] = rebind[Scalar[DTYPE]](qacc_constrained[env, i])
    _finalize_integrate_env(
        env, dt, Dims[nq=NQ, nv=NV, njoint=NJOINT](), qpos, qvel, qacc, joints, qacc_ws
    )


struct EulerIntegrator[
    DTYPE: DType,
    D: DimsLike,
    CONE_TYPE: Int = ConeType.ELLIPTIC,
    BATCH: Int = 1,
    SOLVER: StaticString = "pgs",
    PARALLEL_GPU: Bool = False,
    CRBA_TREEWALK: Bool = False,
    RNE_POST: Bool = False,
    MAX_CONDIM: Int = 3,
    NOSLIP_ITER: Int = 0,
](Movable):
    """Owns its scratch; steps contact-free dynamics on either target. See
    module docstring for what is deliberately not yet ported.
    PARALLEL_GPU=True: the GPU FK / body-velocity / cdof / CRBA /
    LDL-factor / M^-1 / RNE stages run their cooperative within-env (_mt)
    kernels (bit-exact vs serial; other stages stay serial). CPU ignores
    it. CRBA_TREEWALK=True: the CRBA runs the tree-walk algorithm
    (O(NV·depth)) instead of the dense O(NV²·NBODY) one —
    float-tolerance-equal, NOT bit-exact vs dense. ⚠ IT APPLIES ON BOTH
    TARGETS NOW; it used to be GPU-only, which left every CPU caller on the
    dense kernel — 12.6× slower on Sawyer (NV=15, NBODY=34)."""

    var scratch: DynamicsScratch[Self.DTYPE, Self.D, Self.BATCH]
    # Blocked-Newton Jacobian spill size — 0 unless `Je` overflows threadgroup
    # memory. Computed HERE (not by the caller) because this struct already
    # carries every dimension it depends on, and via `je_budget` so the buffer
    # and the kernel that indexes it cannot drift apart.
    comptime JE_WS = je_ws_size[
        Self.DTYPE, Self.D.NV, Self.D.NJOINT, Self.D.NTENDON, Self.D.NEQUALITY,
        Self.D.MAX_CONTACTS, Self.MAX_CONDIM, CONE_TYPE=Self.CONE_TYPE,
    ]()

    var cscratch: ContactScratch[Self.DTYPE, Self.D, Self.BATCH, Self.JE_WS]

    def __init__(out self) raises:
        """Dimensions from the comptime provider; raises on a dynamic one.

        ⚠ THE DIMS OVERLOAD BELOW IS WHAT A RUNTIME-LOADED MODEL NEEDS. The
        `step` body has been dimension-agnostic since 3a — it reads `d.dims`
        and builds `RuntimeLayout`s — so the ONLY thing that stood between
        this integrator and a `DynDims` model was this constructor, which
        allocates its scratch through the nullary path and therefore through
        `comptime_value()`. Same dual-constructor shape as `Model`, `Data`,
        `SpecFields` and both scratches (3a/3b).
        """
        self = Self(Self.D.comptime_value())

    def __init__(out self, dims: Self.D) raises:
        """Dimensions passed in, and ALLOCATED FROM — the runtime path."""
        self.scratch = DynamicsScratch[Self.DTYPE, Self.D, Self.BATCH](dims)
        self.cscratch = ContactScratch[
            Self.DTYPE, Self.D, Self.BATCH, Self.JE_WS
        ](dims)

    def prepare_gpu(mut self, ctx: DeviceContext) raises:
        """Allocate device buffers for the scratch (once, before stepping)."""
        self.scratch.upload_all(ctx)
        self.cscratch.upload_all(ctx)

    def step[
        target: StaticString, CONTACTS: Bool = True
    ](
        mut self,
        mut d: Data[Self.DTYPE, Self.D, Self.BATCH],
        mut m: Model[Self.DTYPE, Self.D],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """One full contact-free Euler step."""
        var dt = m.meta.data[MODEL_META_IDX_TIMESTEP]
        # `<option><flag eulerdamp="disable"/></option>`. Read HERE rather
        # than inside `_finalize_env` so both targets get it from the same
        # place and the GPU kernel takes it as a launch argument.
        var eulerdamp_off = (
            m.meta.data[MODEL_META_IDX_EULERDAMP_DISABLED]
            != Scalar[Self.DTYPE](0)
        )

        var _e_fk: Int = 0
        var _e_bodyvel: Int = 0
        var _e_subtree: Int = 0
        var _e_cdof: Int = 0
        var _e_crba: Int = 0
        var _e_armature: Int = 0
        var _e_ldlf: Int = 0
        var _e_minv: Int = 0
        var _e_rne: Int = 0
        var _e_fnet: Int = 0
        var _e_ldls: Int = 0
        var _e_wb: Int = 0
        var _e_coll: Int = 0
        var _e_newton: Int = 0
        var _e_warm: Int = 0
        var _e_fin: Int = 0
        var _e_last: Int = 0
        comptime if _EULER_PROBE:
            _e_last = Int(perf_counter_ns())
        forward_kinematics[target, Self.DTYPE, BATCH=Self.BATCH, PARALLEL = Self.PARALLEL_GPU](d, m, ctx)
        # ⚠⚠ `subtree_com` IS BUILT HERE, BEFORE THE POSITION-STAGE SENSORS,
        # AND IT USED TO BE BUILT AFTER THEM. It is `mj_comPos`, part of
        # `mj_fwdPosition` and therefore upstream of `mj_sensorPos`
        # (engine_forward.c:1786, 1797) — and the `subtreecom` sensor reads it
        # directly. Evaluated at the old point it would have reported the
        # PREVIOUS STEP's centre of mass: a number that tracks the model,
        # lags it by one step, and is wrong by an amount nobody would notice
        # on a slow-moving body.
        #
        # ⚠ NOTHING ELSE MOVES. It consumes only `xipos` and the body masses,
        # both of which the line above has just written, and every downstream
        # consumer (cdof, the acceleration-stage sensors) reads it later in
        # the step either way — so this reorder changes the VALUE of exactly
        # one thing, and that thing was stale.
        compute_subtree_com[target, Self.DTYPE, BATCH=Self.BATCH](d, m, ctx)
        comptime if _EULER_PROBE:
            var _e_now0 = Int(perf_counter_ns())
            _e_subtree += _e_now0 - _e_last
            _e_last = _e_now0
        # ── mj_tendon, for the sensors only ───────────────────────────────
        # ⚠ INSIDE THE SAME CPU/BATCH=1 GUARD AS THE SENSOR PASSES, AND
        # GUARDED AGAIN ON A CONSUMER. `compute_tendon_lengths` returns at its
        # first lines unless a served `<tendonpos>` row exists, so the cost is
        # zero on every model in this tree but the two hands that declare one.
        # Filling `d.ten_length` unconditionally would be a second polyline
        # walk per tendon per step — 700 of them on ms_human_700 — for a
        # quantity the dynamics already computes where it needs it.
        # ⚠ CPU ONLY, ANY BATCH — AND THE `BATCH == 1` HALF OF THAT GUARD WAS
        # A HOLE. The sensor table marked `<tendonpos>` SERVED while this pass
        # ran for env 0 alone, so a batched model read NaN from a slot the
        # contract says is computed. `compute_tendon_lengths` loops envs now.
        # The `target` half stands: the pass would need per-thread stack for a
        # Jacobian it immediately discards (see its docstring), so a
        # GPU-batched `<tendonpos>` is still NaN, and still loud.
        comptime if target == "cpu":
            compute_tendon_lengths[Self.DTYPE, Self.D, Self.BATCH](
                d, m, self.scratch
            )
        # ── mj_sensorPos ──────────────────────────────────────────────────
        # MuJoCo evaluates the position-stage sensors here, immediately after
        # `mj_fwdPosition` (engine_forward.c:1797).
        #
        # ⚠ BOTH TARGETS, ANY BATCH, AND THE DISPATCH IS INSIDE THE PASS.
        # `sensors/eval.mojo` walks the table through kernels that take
        # `(tensor..., env)`; the CPU leg loops envs and the GPU leg launches
        # one thread per env (AUD-53).
        #
        # ⚠ NO `RNE_POST` GUARD HERE, AND THAT IS DELIBERATE. `RNE_POST` is a
        # SEPARATE axis from the stage: the three sensors that read
        # `cacc`/`cfrc_int` go uncomputed when it is off, and everything else
        # still runs. It used to RAISE from inside `step`, which broke the
        # seven manipulation envs — Jaco declares force/torque sensors and
        # runs `RNE_POST=False`, and the raise aborted their reset hook.
        # `Data` fills `sensordata` with NaN instead, so an uncomputed slot is
        # loud where it is READ and inert where it is not.
        sensor_pos[target, Self.DTYPE, Self.D, Self.BATCH](d, m, ctx)
        comptime if _EULER_PROBE:
            var _e_now = Int(perf_counter_ns())
            _e_fk += _e_now - _e_last
            _e_last = _e_now
        compute_body_velocities[target, Self.DTYPE, BATCH=Self.BATCH, PARALLEL = Self.PARALLEL_GPU](d, m, ctx)
        # ── mj_sensorVel ──────────────────────────────────────────────────
        # `mj_fwdVelocity`'s point (engine_forward.c:1814). velocimeter, gyro
        # and subtreelinvel read `xvel`/`xangvel`, which exist as of the line
        # above and are overwritten on the next step.
        sensor_vel[target, Self.DTYPE, Self.D, Self.BATCH](d, m, ctx)
        comptime if _EULER_PROBE:
            var _e_now = Int(perf_counter_ns())
            _e_bodyvel += _e_now - _e_last
            _e_last = _e_now
        compute_cdof[target, Self.DTYPE, BATCH=Self.BATCH, PARALLEL = Self.PARALLEL_GPU](d, m, self.scratch, ctx)
        comptime if _EULER_PROBE:
            var _e_now = Int(perf_counter_ns())
            _e_cdof += _e_now - _e_last
            _e_last = _e_now
        compute_mass_matrix[target, Self.DTYPE, BATCH=Self.BATCH, PARALLEL = Self.PARALLEL_GPU, TREEWALK = Self.CRBA_TREEWALK](d, m, self.scratch, ctx)
        comptime if _EULER_PROBE:
            var _e_now = Int(perf_counter_ns())
            _e_crba += _e_now - _e_last
            _e_last = _e_now

        comptime L_JOINT = RowMajor[Self.D.NJOINT, MODEL_JOINT_SIZE]
        comptime L_M = RowMajor[Self.BATCH, Self.D.NV * Self.D.NV]
        comptime L_TREE = RowMajor[Self.D.NV * MODEL_TREE_SIZE]
        comptime L_NV = RowMajor[Self.BATCH, Self.D.NV]
        comptime L_QPOS = RowMajor[Self.BATCH, Self.D.NQ]
        comptime BLOCKS = (Self.BATCH + EU_TPB - 1) // EU_TPB

        comptime if target == "cpu":
            var dm = d.dims
            var rl_JOINT = rl2(dm.get_njoint(), MODEL_JOINT_SIZE)
            var rl_M = rl2(Self.BATCH, dm.get_nv() * dm.get_nv())
            var joints_v = m.joints.tt_dyn["cpu", DYN2](rl_JOINT)
            var M_v = self.scratch.M.tt_dyn["cpu", DYN2](rl_M)
            for e in range(Self.BATCH):
                _armature_env[
                    Self.DTYPE](e, dm, joints_v, M_v)
        else:
            ctx.value().enqueue_function[
                _armature_kernel[Self.DTYPE, Self.D.NV, Self.D.NJOINT, Self.BATCH]
            ](
                m.joints.tt["gpu", Self.D.NJOINT, MODEL_JOINT_SIZE](),
                self.scratch.M.tt["gpu", Self.BATCH, Self.D.NV * Self.D.NV](),
                grid_dim=(BLOCKS,),
                block_dim=(EU_TPB,),
            )

        comptime if _EULER_PROBE:
            var _e_now = Int(perf_counter_ns())
            _e_armature += _e_now - _e_last
            _e_last = _e_now
        ldl_factor[target, Self.DTYPE, BATCH=Self.BATCH, PARALLEL = Self.PARALLEL_GPU](m, self.scratch, ctx)
        comptime if _EULER_PROBE:
            var _e_now = Int(perf_counter_ns())
            _e_ldlf += _e_now - _e_last
            _e_last = _e_now
        # ⚠ `M^-1` IS FORMED ONLY FOR A READER. MuJoCo's Newton never builds
        # it: `mj_diagApprox` prices rows with the model-time `*_invweight0`,
        # and `mj_projectConstraint` solves against `qLD`, only for a dual
        # solver or noslip. Ours formed the full dense inverse every step —
        # O(nv^3), measured at 24-46% of every step past 20 dofs
        # (`PERFORMANCE.md` §13) — to feed a diagonal round-trip the row
        # builders no longer do. What still reads it: PGS/CG/island (their
        # whole `A R` machinery), `noslip_pyramidal` (M^-1 J^T per row), and
        # the connect/weld equality rows (`w_MinvJ`). The first two are
        # comptime facts; the last is a model count, read at RUNTIME so the
        # studio's dynamic leg makes the same decision as the static one.
        # ⚠ `CONTACTS` IS PART OF THE PREDICATE. With `CONTACTS=False` the seam
        # below runs the standalone `solve_limits` / `solve_friction` stages,
        # which are Gauss-Seidel over `J M^-1 J^T` and READ the inverse — the
        # bisect that found this was `test_limit_solref_per_joint`, whose
        # limit row went inert (153 of qacc) the moment the matrix was stale.
        comptime if CONTACTS and Self.SOLVER == "newton":
            # The Newton reads `M^-1` for equality rows, and its noslip does
            # unless the CPU tree path solves against the LDL factor instead
            # (`noslip._minv_apply`; the predicate is the LDL dispatcher's).
            var need_minv = d.dims.get_nequality() > 0
            comptime if Self.NOSLIP_ITER > 0:
                var tree_cpu = False
                comptime if target == "cpu":
                    tree_cpu = Int(m.meta.data[MODEL_META_IDX_NTREE]) > 0
                if not tree_cpu:
                    need_minv = True
            if need_minv:
                compute_m_inv[target, Self.DTYPE, BATCH=Self.BATCH, PARALLEL = Self.PARALLEL_GPU](m, self.scratch, ctx)
        else:
            compute_m_inv[target, Self.DTYPE, BATCH=Self.BATCH, PARALLEL = Self.PARALLEL_GPU](m, self.scratch, ctx)
        comptime if _EULER_PROBE:
            var _e_now = Int(perf_counter_ns())
            _e_minv += _e_now - _e_last
            _e_last = _e_now
        compute_bias_forces_rne[target, Self.DTYPE, BATCH=Self.BATCH, PARALLEL = Self.PARALLEL_GPU](d, m, self.scratch, ctx)
        comptime if _EULER_PROBE:
            var _e_now = Int(perf_counter_ns())
            _e_rne += _e_now - _e_last
            _e_last = _e_now

        comptime if target == "cpu":
            var dm = d.dims
            var rl_QPOS = rl2(Self.BATCH, dm.get_nq())
            var rl_NV = rl2(Self.BATCH, dm.get_nv())
            var rl_JOINT = rl2(dm.get_njoint(), MODEL_JOINT_SIZE)
            var qpos_v = d.qpos.tt_dyn["cpu", DYN2](rl_QPOS)
            var qvel_v = d.qvel.tt_dyn["cpu", DYN2](rl_NV)
            var qfrc_v = d.qfrc.tt_dyn["cpu", DYN2](rl_NV)
            var joints_v2 = m.joints.tt_dyn["cpu", DYN2](rl_JOINT)
            var bias_v = self.scratch.bias.tt_dyn["cpu", DYN2](rl_NV)
            var fnet_v = self.scratch.fnet.tt_dyn["cpu", DYN2](rl_NV)
            for e in range(Self.BATCH):
                _fnet_passive_env[
                    Self.DTYPE](e, dm, qpos_v, qvel_v, qfrc_v, joints_v2, bias_v, fnet_v)
        else:
            ctx.value().enqueue_function[
                _fnet_passive_kernel[
                    Self.DTYPE, Self.D.NQ, Self.D.NV, Self.D.NJOINT, Self.BATCH
                ]
            ](
                d.qpos.tt["gpu", Self.BATCH, Self.D.NQ](),
                d.qvel.tt["gpu", Self.BATCH, Self.D.NV](),
                d.qfrc.tt["gpu", Self.BATCH, Self.D.NV](),
                m.joints.tt["gpu", Self.D.NJOINT, MODEL_JOINT_SIZE](),
                self.scratch.bias.tt["gpu", Self.BATCH, Self.D.NV](),
                self.scratch.fnet.tt["gpu", Self.BATCH, Self.D.NV](),
                grid_dim=(BLOCKS,),
                block_dim=(EU_TPB,),
            )

        # 8c. Fluid drag into fnet (no-op unless meta density/viscosity > 0).
        compute_fluid_forces[target, Self.DTYPE, BATCH=Self.BATCH](d, m, self.scratch, ctx)
        # ⚠ AFTER the fluid call, not before: `mj_passive` adds `qfrc_fluid`
        # into `qfrc_passive` and only then `qfrc_gravcomp`
        # (engine_passive.c:1000-1022). No-op unless the model declares
        # `<body gravcomp>`.
        compute_gravcomp_forces[target, Self.DTYPE, BATCH=Self.BATCH](d, m, self.scratch, ctx)

        comptime if _EULER_PROBE:
            var _e_now = Int(perf_counter_ns())
            _e_fnet += _e_now - _e_last
            _e_last = _e_now
        ldl_solve[target, Self.DTYPE, BATCH=Self.BATCH, PARALLEL = Self.PARALLEL_GPU](m, self.scratch, ctx)
        comptime if _EULER_PROBE:
            var _e_now = Int(perf_counter_ns())
            _e_ldls += _e_now - _e_last
            _e_last = _e_now

        comptime if target == "cpu":
            var dm = d.dims
            var rl_NV = rl2(Self.BATCH, dm.get_nv())
            var qacc_ws_v = self.scratch.qacc_ws.tt_dyn["cpu", DYN2](rl_NV)
            var qacc_v = d.qacc.tt_dyn["cpu", DYN2](rl_NV)
            var qacc_c_v = self.scratch.qacc_constrained.tt_dyn["cpu", DYN2](rl_NV)
            for e in range(Self.BATCH):
                _qacc_writeback_env[Self.DTYPE](
                    e, dm, qacc_ws_v, qacc_v, qacc_c_v
                )
        else:
            ctx.value().enqueue_function[
                _qacc_writeback_kernel[Self.DTYPE, Self.D.NV, Self.BATCH]
            ](
                self.scratch.qacc_ws.tt["gpu", Self.BATCH, Self.D.NV](),
                d.qacc.tt["gpu", Self.BATCH, Self.D.NV](),
                self.scratch.qacc_constrained.tt["gpu", Self.BATCH, Self.D.NV](),
                grid_dim=(BLOCKS,),
                block_dim=(EU_TPB,),
            )

        comptime if _EULER_PROBE:
            var _e_now = Int(perf_counter_ns())
            _e_wb += _e_now - _e_last
            _e_last = _e_now
        # Constraint seam (order matches the legacy PGS solver): contact
        # detection -> contact PGS -> joint limits, all updating
        # scratch.qacc_constrained. Equality/tendons join here later.
        comptime if CONTACTS:
            # Contact solve runs limits INSIDE (legacy PGS position: between
            # the normal and friction phases, PGS_ITERATIONS iterations) —
            # the standalone limits stage below is for CONTACTS=False only.
            # Auto broadphase = legacy production (SAP for NGEOM >= 16,
            # O(N^2) otherwise; same routing as the legacy step kernel's
            # detect_contacts_auto_gpu call).
            detect_contacts_auto[target, Self.DTYPE, BATCH=Self.BATCH](d, m, ctx)
            comptime if _EULER_PROBE:
                var _e_now = Int(perf_counter_ns())
                _e_coll += _e_now - _e_last
                _e_last = _e_now
            comptime assert (
                Self.SOLVER == "pgs"
                or Self.SOLVER == "newton"
                or Self.SOLVER == "cg"
                or Self.SOLVER == "island"
            ), (
                "EulerIntegrator: SOLVER must be 'pgs', 'newton', 'cg',"
                " or 'island'"
            )
            comptime if Self.SOLVER == "newton":
                solve_newton[target, Self.DTYPE, CONE_TYPE=Self.CONE_TYPE, BATCH=Self.BATCH, MAX_CONDIM=Self.MAX_CONDIM, NOSLIP_ITER=Self.NOSLIP_ITER, JE_WS=Self.JE_WS](d, m, self.scratch, self.cscratch, ctx)
            else:
                comptime if Self.SOLVER == "cg":
                    solve_cg[target, Self.DTYPE, CONE_TYPE=Self.CONE_TYPE, BATCH=Self.BATCH](d, m, self.scratch, self.cscratch, ctx)
                else:
                    comptime if Self.SOLVER == "island":
                        solve_island_pgs[target, Self.DTYPE, CONE_TYPE=Self.CONE_TYPE, BATCH=Self.BATCH](d, m, self.scratch, self.cscratch, ctx)
                    else:
                        solve_contacts[target, Self.DTYPE, CONE_TYPE=Self.CONE_TYPE, BATCH=Self.BATCH](d, m, self.scratch, self.cscratch, ctx)
        else:
            solve_limits[target, Self.DTYPE, BATCH=Self.BATCH](d, m, self.scratch, ctx)
            # Dry-friction dof rows. With CONTACTS=False no solver runs, so
            # this is the only place they can be applied; with contacts the
            # solvers call `_friction_env` themselves beside their limit rows.
            solve_friction[target, Self.DTYPE, BATCH=Self.BATCH](d, m, self.scratch, ctx)

        # `qacc_warmstart = qacc` — the tail of `mj_forward`
        # (engine_forward.c:1087). ⚠ HERE AND NOT AFTER THE INTEGRATOR: MuJoCo
        # saves the CONSTRAINT SOLVER's acceleration, and anything the
        # integrator does to `qacc` afterwards never reaches `qacc_warmstart`.
        comptime if _EULER_PROBE:
            var _e_now = Int(perf_counter_ns())
            _e_newton += _e_now - _e_last
            _e_last = _e_now
        save_qacc_warmstart[target, Self.DTYPE, BATCH=Self.BATCH](
            d, self.scratch, ctx
        )

        # `mj_sensorAcc` sits exactly here in MuJoCo: after fwdConstraint,
        # before the integrator. Every input the stage needs (FK products,
        # solved contact forces, scratch.qacc_constrained) is valid at this
        # point and stale one line later — `_finalize_env` overwrites
        # d.qacc with the implicit-damping re-solve and moves qpos/qvel on.
        comptime if _EULER_PROBE:
            var _e_now = Int(perf_counter_ns())
            _e_warm += _e_now - _e_last
            _e_last = _e_now
        comptime if Self.RNE_POST:
            compute_rne_post[target, Self.DTYPE, BATCH=Self.BATCH](d, m, self.scratch, ctx)

            # ⚠ AND THE FK PRODUCTS THAT GO WITH THEM — defect 19.
            #
            # `cacc`/`cfrc_int` are only half of an acceleration-stage sensor:
            # the other half is the site pose they are transported to and
            # rotated into. Those live in `site_xpos`/`xquat`, which
            # `Phyics3dEnv._fields_fk` moves to the POST-integration state
            # after the substep loop (correctly — the position/velocity-stage
            # observation dims need it, and dm_control's `mj_step1` does the
            # same). Reading them at observation time therefore mixed
            # pre-integration `cacc` with post-integration geometry.
            #
            # MuJoCo never has this problem because it evaluates the stage
            # HERE and stores the finished sensor value. We cannot do that
            # generically — the sensor set is per-CONFIG, not per-engine — so
            # the inputs are frozen instead, at the same instant and under the
            # same `RNE_POST` gate that writes `cacc`. A model without the
            # stage pays nothing.
            comptime N_SITE_ACC = Self.BATCH * Self.D.NSITE * 3
            comptime N_QUAT_ACC = Self.BATCH * Self.D.NBODY * 4
            comptime if target == "cpu":
                comptime if N_SITE_ACC > 0:
                    for i in range(N_SITE_ACC):
                        d.site_xpos_acc.data[i] = d.site_xpos.data[i]
                for i in range(N_QUAT_ACC):
                    d.xquat_acc.data[i] = d.xquat.data[i]
            else:
                # Device-to-device, both buffers owned and the same length by
                # construction. Mirrored on GPU rather than skipped: a snapshot
                # that only exists on one target is the same silent-divergence
                # shape this whole fix exists to remove.
                # ⚠ `as_unsafe_any_origin()` — `copy_from_device` wants
                # `MutAnyOrigin` and the buffer's pointer carries
                # `origin_of(dev._value)`. Same spelling as
                # `dreamerv3/param_sync.mojo`, which is the established caller.
                #
                # ⚠ COMPILES BUT IS NOT RUNTIME-GATED: no model with
                # `RNE_POST` runs on GPU today (dog and quadruped are both
                # CPU-only, because the batched facade carries no `act`). It is
                # written rather than skipped so the device path cannot silently
                # diverge the day one does, and it is flagged here rather than
                # left to look tested.
                var c = ctx.value()
                comptime if N_SITE_ACC > 0:
                    d.site_xpos_acc.copy_from_device(
                        c,
                        d.site_xpos.dev.value().unsafe_ptr().as_unsafe_any_origin(),
                        N_SITE_ACC,
                    )
                d.xquat_acc.copy_from_device(
                    c,
                    d.xquat.dev.value().unsafe_ptr().as_unsafe_any_origin(),
                    N_QUAT_ACC,
                )

            # ── mj_sensorAcc ──────────────────────────────────────────────
            # ⚠⚠ THIS IS WHAT THE COMMENT ABOVE SAID WE COULD NOT DO. It reads
            # "MuJoCo never has this problem because it evaluates the stage
            # HERE and stores the finished sensor value. We cannot do that
            # generically — the sensor set is per-CONFIG, not per-engine."
            # That stopped being true when `<sensor>` became part of the
            # model (AUD-23): the sensor set now IS per-engine, so the stage
            # is evaluated here like MuJoCo's and `d.sensordata` holds the
            # answer.
            #
            # The `*_acc` snapshot above is still needed and is not dead: the
            # hand-written env hooks that predate the framework read it, and
            # they are what the framework is replacing rather than what it has
            # already replaced.
            #
            # ⚠ INSIDE THE `RNE_POST` GATE, because `cacc`/`cfrc_int` are what
            # this stage reads and only that gate writes them. A model with
            # acceleration-stage sensors and `RNE_POST=False` would otherwise
            # read zeros — see `_sensors_need_a_pass`.

        # ── mj_sensorAcc ──────────────────────────────────────────────────
        # ⚠⚠ OUTSIDE THE `RNE_POST` BLOCK, AND THAT IS THE FIX FOR A REAL
        # BREAKAGE. MuJoCo classes TOUCH as an acceleration-stage sensor, so
        # the first version put this inside the gate — and hopper, which
        # declares two touch sensors and runs `RNE_POST=False`, stopped
        # stepping entirely. Touch reads contacts and the live `site_xpos`;
        # only the accelerometer and the force/torque pair read `cacc` /
        # `cfrc_int`. The flag is passed down so the pass skips exactly those
        # three when the RNE has not run, and computes the rest.
        #
        # This still sits at MuJoCo's point (engine_forward.c:1832): after the
        # constraint solve, before `_finalize_env` moves qpos/qvel on.
        sensor_acc[target, Self.DTYPE, Self.D, Self.BATCH](
            d, m, Self.RNE_POST, ctx
        )

        comptime if target == "cpu":
            var dm = d.dims
            var rl_QPOS = rl2(Self.BATCH, dm.get_nq())
            var rl_NV = rl2(Self.BATCH, dm.get_nv())
            var rl_JOINT = rl2(dm.get_njoint(), MODEL_JOINT_SIZE)
            var rl_M = rl2(Self.BATCH, dm.get_nv() * dm.get_nv())
            var qpos_v3 = d.qpos.tt_dyn["cpu", DYN2](rl_QPOS)
            var qvel_v3 = d.qvel.tt_dyn["cpu", DYN2](rl_NV)
            var qacc_v3 = d.qacc.tt_dyn["cpu", DYN2](rl_NV)
            var joints_v3 = m.joints.tt_dyn["cpu", DYN2](rl_JOINT)
            var M_v3 = self.scratch.M.tt_dyn["cpu", DYN2](rl_M)
            var L_v3 = self.scratch.L.tt_dyn["cpu", DYN2](rl_M)
            var D_v3 = self.scratch.D.tt_dyn["cpu", DYN2](rl_NV)
            var fnet_v3 = self.scratch.fnet.tt_dyn["cpu", DYN2](rl_NV)
            var qacc_ws_v3 = self.scratch.qacc_ws.tt_dyn["cpu", DYN2](rl_NV)
            var qacc_c_v3 = self.scratch.qacc_constrained.tt_dyn["cpu", DYN2](rl_NV)
            var rl_TREE = rl1(dm.get_nv() * MODEL_TREE_SIZE)
            var trees_v3 = m.trees.tt_dyn["cpu", DYN1](rl_TREE)
            # Tree-ordered when the parser built the tables (the LDL
            # dispatchers' own rule), dense otherwise.
            var use_tree3 = Int(m.meta.data[MODEL_META_IDX_NTREE]) > 0
            var P_v3 = m.dof_parentid.tt_dyn["cpu", DYN1](
                rl1(dm.get_nv() if dm.get_nv() > 0 else 1)
            )
            for e in range(Self.BATCH):
                if use_tree3:
                    _finalize_tree_env[Self.DTYPE](
                        e, dt, dm, qpos_v3, qvel_v3, qacc_v3, joints_v3, M_v3,
                        L_v3, D_v3, fnet_v3, qacc_ws_v3, qacc_c_v3, P_v3,
                        eulerdamp_off,
                    )
                else:
                    _finalize_env[
                        Self.DTYPE](
                        e, dt, dm, qpos_v3, qvel_v3, qacc_v3, joints_v3, M_v3, L_v3,
                        D_v3, fnet_v3, qacc_ws_v3, qacc_c_v3, trees_v3,
                        eulerdamp_off,
                    )
            comptime if _EULER_PROBE:
                var _e_now = Int(perf_counter_ns())
                _e_fin += _e_now - _e_last
                _e_last = _e_now
            comptime if _EULER_PROBE:
                print("[eprobe]", "fk", _e_fk, "bodyvel", _e_bodyvel, "subtree", _e_subtree, "cdof", _e_cdof, "crba", _e_crba, "armature", _e_armature, "ldlf", _e_ldlf, "minv", _e_minv, "rne", _e_rne, "fnet", _e_fnet, "ldls", _e_ldls, "wb", _e_wb, "coll", _e_coll, "newton", _e_newton, "warm", _e_warm, "fin", _e_fin)
        elif EULER_FINALIZE_SPLIT:
            var c = ctx.value()
            if not eulerdamp_off:
                c.enqueue_function[
                    _finalize_rhs_kernel[
                        Self.DTYPE, Self.D.NV, Self.D.NJOINT, Self.BATCH
                    ]
                ](
                    dt,
                    m.joints.tt["gpu", Self.D.NJOINT, MODEL_JOINT_SIZE](),
                    self.scratch.M.tt["gpu", Self.BATCH, Self.D.NV * Self.D.NV](),
                    self.scratch.fnet.tt["gpu", Self.BATCH, Self.D.NV](),
                    self.scratch.qacc_constrained.tt["gpu", Self.BATCH, Self.D.NV](),
                    grid_dim=(Self.BATCH,),
                    block_dim=(Self.D.NV if Self.D.NV > 0 else 1,),
                )
                # `scratch.M` is `M_hat` now; the step's own factor and solve,
                # on the fields they already read and write.
                ldl_factor[target, Self.DTYPE, BATCH=Self.BATCH, PARALLEL = Self.PARALLEL_GPU](m, self.scratch, ctx)
                ldl_solve[target, Self.DTYPE, BATCH=Self.BATCH, PARALLEL = Self.PARALLEL_GPU](m, self.scratch, ctx)
            c.enqueue_function[
                _finalize_integrate_kernel[
                    Self.DTYPE, Self.D.NQ, Self.D.NV, Self.D.NJOINT, Self.BATCH
                ]
            ](
                dt,
                d.qpos.tt["gpu", Self.BATCH, Self.D.NQ](),
                d.qvel.tt["gpu", Self.BATCH, Self.D.NV](),
                d.qacc.tt["gpu", Self.BATCH, Self.D.NV](),
                m.joints.tt["gpu", Self.D.NJOINT, MODEL_JOINT_SIZE](),
                self.scratch.qacc_ws.tt["gpu", Self.BATCH, Self.D.NV](),
                self.scratch.qacc_constrained.tt["gpu", Self.BATCH, Self.D.NV](),
                Scalar[Self.DTYPE](1) if eulerdamp_off
                else Scalar[Self.DTYPE](0),
                grid_dim=(BLOCKS,),
                block_dim=(EU_TPB,),
            )
        else:
            ctx.value().enqueue_function[
                _finalize_kernel[
                    Self.DTYPE, Self.D.NQ, Self.D.NV, Self.D.NJOINT, Self.BATCH
                ]
            ](
                dt,
                d.qpos.tt["gpu", Self.BATCH, Self.D.NQ](),
                d.qvel.tt["gpu", Self.BATCH, Self.D.NV](),
                d.qacc.tt["gpu", Self.BATCH, Self.D.NV](),
                m.joints.tt["gpu", Self.D.NJOINT, MODEL_JOINT_SIZE](),
                self.scratch.M.tt["gpu", Self.BATCH, Self.D.NV * Self.D.NV](),
                self.scratch.L.tt["gpu", Self.BATCH, Self.D.NV * Self.D.NV](),
                self.scratch.D.tt["gpu", Self.BATCH, Self.D.NV](),
                self.scratch.fnet.tt["gpu", Self.BATCH, Self.D.NV](),
                self.scratch.qacc_ws.tt["gpu", Self.BATCH, Self.D.NV](),
                self.scratch.qacc_constrained.tt["gpu", Self.BATCH, Self.D.NV](),
                m.trees.tt["gpu", Self.D.NV * MODEL_TREE_SIZE](),
                Scalar[Self.DTYPE](1) if eulerdamp_off
                else Scalar[Self.DTYPE](0),
                grid_dim=(BLOCKS,),
                block_dim=(EU_TPB,),
            )
