"""Stateful full-Implicit integrator over per-field tensors (Stage-I).

`ImplicitIntegrator` is the fields-native port of the legacy
`ImplicitIntegrator`: like Euler/ImplicitFast up through the mass matrix, but
it forms the FULL non-symmetric

    M_hat = M + armature - dt * qDeriv

where `qDeriv = d(qfrc_bias)/d(qvel)` includes BOTH the passive damping
diagonal (`qDeriv[i,i] = -damping[i]`) AND the dense RNE velocity derivative
(Coriolis/centrifugal, `qderiv`). Because `M_hat` is non-symmetric it
uses LU (`lu`), not LDL. Damping is also explicit in the force
(`fnet -= damping*qvel`), exactly like the legacy step — that is the standard
implicit linearization, not double counting.

Pipeline (contact-free path shown; the constraint seam mirrors euler):
    FK -> body vel -> subtree_com -> cdof -> CRBA -> +armature ->
    qDeriv (damping diag + RNE deriv) -> M_hat = M - dt*qDeriv ->
    LU factor -> M^-1 (for constraints) -> RNE bias -> fnet assembly ->
    LU solve -> qacc writeback -> [constraint seam] ->
    finalize (v += dt*qacc ; integrate qpos, quat-aware)

Unlike Euler/ImplicitFast there is NO post-constraint `dt*D` re-solve: the
implicit terms already live in `M_hat` (this matches the legacy CPU Implicit
step; the legacy GPU path re-used ImplicitFast's finalize, an asymmetry this
single-source port drops in favour of CPU==GPU consistency).

Reuses euler' tested per-stage helpers (armature / fnet / qacc
writeback); the M_hat-forming, damping-diagonal, and implicit-finalize
kernels are new here. Deliberately NOT ported yet (raise on use): fluid
forces (density/viscosity > 0)."""

from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceContext
from layout import TileTensor, TensorLayout
from noeira.core.tiles import RowMajor, Tile

from ..kinematics.quat_math import quat_integrate, quat_normalize, quat_rotate, quat_mul
from ..kinematics.forward_kinematics import (
    forward_kinematics,
    compute_body_velocities,
)
from ..dynamics.subtree_com import compute_subtree_com
from ..dynamics.cdof import compute_cdof
from ..dynamics.mass_matrix import compute_mass_matrix
from ..dynamics.rne import compute_bias_forces_rne
from ..dynamics.fluid_forces import compute_fluid_forces
from ..dynamics.gravcomp import compute_gravcomp_forces
from ..dynamics.lu import (
    lu_factor,
    lu_solve,
    compute_m_inv_from_lu,
)
from ..dynamics.ldl import ldl_factor
from ..dynamics.qderiv import compute_rne_vel_derivative
from ..constraints.limits import solve_limits
from ..constraints.contact_solve import solve_contacts
from ..solver.newton_solve import solve_newton
from ..solver.warmstart import save_qacc_warmstart
from ..solver.je_budget import je_ws_size
from ..solver.cg_solve import solve_cg
from ..solver.island_pgs_solve import solve_island_pgs
from ..collision.broadphase_sap import detect_contacts_auto
from ..types import ConeType
from ..joint_types import JNT_FREE, JNT_BALL, JNT_HINGE, JNT_SLIDE
from ..fields import (
    AsStatic,
    Dims,
    Dims,
    Dims,
    DimsLike,
    DimsLike,
    DimsLike,
    Data,
    Model,
    DynamicsScratch,
    ContactScratch,
    ImplicitScratch,
    Dims,
    DimsLike,
    DYN2,
    rl2,
)
from .euler import (
    _armature_env,
    _armature_kernel,
    _fnet_passive_env,
    _fnet_passive_kernel,
    _qacc_writeback_env,
    _qacc_writeback_kernel,
)
from ..gpu.constants import (
    MODEL_BODY_SIZE,
    BODY_IDX_MASS,
    BODY_IDX_IXX,
    BODY_IDX_IYY,
    BODY_IDX_IZZ,
    BODY_IDX_PARENT,
    BODY_IDX_IQUAT_X,
    BODY_IDX_IQUAT_Y,
    BODY_IDX_IQUAT_Z,
    BODY_IDX_IQUAT_W,
    JOINT_IDX_BODY_ID,
    MJ_MAXVAL,
    MODEL_JOINT_SIZE,
    MODEL_META_IDX_TIMESTEP,
    MODEL_META_IDX_DENSITY,
    MODEL_META_IDX_VISCOSITY,
    MODEL_META_IDX_NJOINT,
    MODEL_META_IDX_NTREE,
    JOINT_IDX_TYPE,
    JOINT_IDX_QPOS_ADR,
    JOINT_IDX_DOF_ADR,
    JOINT_IDX_DAMPING,
    METADATA_SIZE,
    META_IDX_ACTDAMP_LIVE,
    ACTDAMP_TRN_SIZE,
    ACTDAMP_IDX_N,
    ACTDAMP_IDX_DOF_0,
    ACTDAMP_IDX_PAIR_0,
    TENDON_MAX_WRAPS,
)

comptime IM_TPB: Int = 64


# ── qDeriv damping diagonal init: zero, then qDeriv[i,i] = -damping[i] ─────
@always_inline
def _qderiv_damping_env[
    DTYPE: DType,
    D: DimsLike,
    L_JOINTS: TensorLayout,
    L_ACTD: TensorLayout,
    L_ACTDL: TensorLayout,
    L_ATRN: TensorLayout,
    L_AACT: TensorLayout,
    L_META: TensorLayout,
    L_QDERIV: TensorLayout](
    env: Int,
    dims: D,
    joints: TileTensor[DTYPE, L_JOINTS, MutAnyOrigin],
    njoint: Int,
    actd: TileTensor[DTYPE, L_ACTD, MutAnyOrigin],
    actd_live: TileTensor[DTYPE, L_ACTDL, MutAnyOrigin],
    atrn: TileTensor[DTYPE, L_ATRN, MutAnyOrigin],
    aact: TileTensor[DTYPE, L_AACT, MutAnyOrigin],
    meta: TileTensor[DTYPE, L_META, MutAnyOrigin],
    qderiv: TileTensor[DTYPE, L_QDERIV, MutAnyOrigin],
):
    var nv = dims.get_nv()
    for i in range(nv * nv):
        qderiv[env, i] = 0
    for j in range(njoint):
        var jnt_type = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_TYPE]))
        var dof_adr = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_DOF_ADR]))
        var damp = rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_DAMPING])
        var nd = 1
        if jnt_type == JNT_FREE:
            nd = 6
        elif jnt_type == JNT_BALL:
            nd = 3
        for d in range(nd):
            var dof = dof_adr + d
            qderiv[env, dof * nv + dof] = -damp

    # ── d qfrc_actuator / d qvel — `mjd_actuator_vel` ────────────────────
    #
    # ⚠⚠ THE TERM WITHOUT WHICH THIS INTEGRATOR IS NOT IMPLICIT FOR SERVOS.
    # MuJoCo's `mjd_smooth_vel` is actuator + passive + (optional) RNE; only
    # the last two were here, so a model whose damping is ENTIRELY actuator
    # `kv` — spot's `dof_damping` is 0 — got an M_hat identical to M and an
    # integrator that was implicit in name only.
    #
    # `dof_actdamp` is `sum_a kv_a * trn_a^2`, the diagonal of
    # `J^T diag(kv) J`, banked at build time because for a JOINT transmission
    # it is constant in qpos and exact. Subtracted, because qDeriv holds
    # d(force)/d(vel) and the servo term is `-kv*vel`.
    # ⚠⚠ THE LIVE ARRAY WHEN IT WAS FILLED, THE MODEL'S WHEN IT WAS NOT.
    # MuJoCo's `mjd_actuator_vel` SKIPS an actuator whose force is clamped by
    # its `forcerange` — a saturated servo's force is pinned at the bound and
    # no longer depends on velocity — and whether it is saturated changes
    # every step, so no model-time array can carry it. `apply_actions_fields`
    # writes `d.dof_actdamp` and raises `META_IDX_ACTDAMP_LIVE`; a step taken
    # with no actuation call at all leaves the flag down, and then the
    # model-time value IS right because nothing can be saturated.
    #
    # Measured on rby1, whose 24 servos are `forcerange="-270 270"` and
    # saturate at qpos0: MuJoCo's `qDeriv` diagonal is -5 there (joint damping
    # alone) against our -405, while its two `<velocity>` wheels — no
    # forcerange, never clamped — read -4005 in both.
    var live = rebind[Scalar[DTYPE]](
        meta[env, META_IDX_ACTDAMP_LIVE]
    ) != Scalar[DTYPE](0)
    for i in range(nv):
        var a = (
            rebind[Scalar[DTYPE]](actd_live[env, i]) if live
            else rebind[Scalar[DTYPE]](actd[i, 0])
        )
        qderiv[env, i * nv + i] -= a

    # ── and the OFF-DIAGONAL of the same `moment^T kv moment` ────────────
    #
    # ⚠⚠ THE HALF THAT WAS MISSING. `dof_actdamp` above is only the DIAGONAL;
    # `mjd_actuator_vel` adds the whole outer product. For a JOINT
    # transmission the two are identical — one dof, one entry — so this loop
    # writes nothing on every legged model in the tree. It is a TENDON
    # transmission that has the rest, and `hello_robot_stretch`'s
    # `arm_extend` is one: 4 dofs, `kv = 10`, 12 off-diagonal terms. Its
    # step-1 residual against MuJoCo was **4.406e-05** with them dropped and
    # 1.823e-10 when both engines were forced to an integrator that does not
    # use `qDeriv` at all.
    #
    # ⚠ `p == q` IS SKIPPED, NOT ADDED. That entry is `dof_actdamp`'s and
    # adding it here would double every servo's damping — including the ~1200
    # single-dof ones this loop is otherwise a no-op for.
    #
    # ⚠ THE LIVE GATE IS PER ACTUATOR, the diagonal's is per dof, and they
    # are raised under the SAME test in `apply_actions_fields`. With the flag
    # down every actuator counts, because a step that never actuated cannot
    # have saturated one — the same reading `actd` gets above.
    var nact_d = dims.get_nact()
    for a in range(nact_d):
        if live and rebind[Scalar[DTYPE]](aact[env, a]) == 0:
            continue
        var ao = a * ACTDAMP_TRN_SIZE
        var n_d = Int(rebind[Scalar[DTYPE]](atrn[ao + ACTDAMP_IDX_N, 0]))
        if n_d < 2:
            continue
        for p in range(n_d):
            var dp = Int(
                rebind[Scalar[DTYPE]](atrn[ao + ACTDAMP_IDX_DOF_0 + p, 0])
            )
            if dp < 0 or dp >= nv:
                continue
            for q in range(n_d):
                if q == p:
                    continue
                var dq = Int(
                    rebind[Scalar[DTYPE]](atrn[ao + ACTDAMP_IDX_DOF_0 + q, 0])
                )
                if dq < 0 or dq >= nv:
                    continue
                # ⚠ ALREADY `kv * mom_p * mom_q`, AND ALREADY ZERO where
                # MuJoCo's sparse `qDeriv` has no slot for the pair. Both the
                # product and the sparsity filter are model-time — see
                # `ACTDAMP_TRN_SIZE` and `_dofs_are_ancestor_related`.
                qderiv[env, dp * nv + dq] -= rebind[Scalar[DTYPE]](
                    atrn[
                        ao + ACTDAMP_IDX_PAIR_0 + p * TENDON_MAX_WRAPS + q, 0
                    ]
                )


# ── M_hat: M -= dt * qDeriv (full, non-symmetric) ─────────────────────────
@always_inline
def _msub_qderiv_env[
    DTYPE: DType,
    D: DimsLike,
    L_M: TensorLayout](
    env: Int,
    dt: Scalar[DTYPE],
    dims: D,
    M: TileTensor[DTYPE, L_M, MutAnyOrigin],
    qderiv: TileTensor[DTYPE, L_M, MutAnyOrigin],
):
    var nv = dims.get_nv()
    for i in range(nv * nv):
        var cur = rebind[Scalar[DTYPE]](M[env, i])
        var qd = rebind[Scalar[DTYPE]](qderiv[env, i])
        M[env, i] = cur - dt * qd


# ── rhs = M * qacc_constrained, and adopting the re-solved acceleration ───
# =============================================================================
# implicitfast: the standalone free body's local 6x6 solve (AUD-44)
# =============================================================================
#
# MuJoCo 3.11 (commit f0fa3d82) replaced midpoint integration of free bodies
# with the GYROSCOPIC DERIVATIVE: `implicitfast` skips the RNE velocity
# derivative globally (it is unsymmetric), but for a body that is a whole
# kinematic tree of its own — one free joint, no children — the 6x6 block of
# `M - h*qDeriv` is decoupled from every other dof, so the bias derivative can
# be added back there and the block solved locally with an unsymmetric LU
# (`engine_forward.c:1738-1758`, `engine_derivative.c:723-896`). Without it a
# tumbling free body under implicitfast gains/loses energy that MuJoCo's
# does not: 8.2e-5 of qvel per step on a 0.1 x 0.05 x 0.02 box spinning at
# (1, 5, 2) rad/s, 6.8e-3 of qpos after 500 steps.


@always_inline
def _q2m[
    DTYPE: DType
](
    qx: Scalar[DTYPE], qy: Scalar[DTYPE], qz: Scalar[DTYPE], qw: Scalar[DTYPE]
) -> Array[Scalar[DTYPE], 9]:
    """Rotation matrix of a unit quaternion (x, y, z, w), row-major."""
    var m = Array[Scalar[DTYPE], 9](fill=Scalar[DTYPE](0))
    var c0 = quat_rotate[DTYPE](qx, qy, qz, qw, Scalar[DTYPE](1), Scalar[DTYPE](0), Scalar[DTYPE](0))
    var c1 = quat_rotate[DTYPE](qx, qy, qz, qw, Scalar[DTYPE](0), Scalar[DTYPE](1), Scalar[DTYPE](0))
    var c2 = quat_rotate[DTYPE](qx, qy, qz, qw, Scalar[DTYPE](0), Scalar[DTYPE](0), Scalar[DTYPE](1))
    m[0] = c0[0]
    m[3] = c0[1]
    m[6] = c0[2]
    m[1] = c1[0]
    m[4] = c1[1]
    m[7] = c1[2]
    m[2] = c2[0]
    m[5] = c2[1]
    m[8] = c2[2]
    return m^


@always_inline
def _free_body_block_env[
    DTYPE: DType,
    D: DimsLike,
    L_M: TensorLayout,
    L_NV: TensorLayout,
    L_JOINTS: TensorLayout,
    L_BODIES: TensorLayout,
    L_X3: TensorLayout,
    L_X4: TensorLayout,
](
    env: Int,
    dt: Scalar[DTYPE],
    dims: D,
    joints: TileTensor[DTYPE, L_JOINTS, MutAnyOrigin],
    bodies: TileTensor[DTYPE, L_BODIES, MutAnyOrigin],
    xpos: TileTensor[DTYPE, L_X3, MutAnyOrigin],
    xipos: TileTensor[DTYPE, L_X3, MutAnyOrigin],
    xquat: TileTensor[DTYPE, L_X4, MutAnyOrigin],
    qvel: TileTensor[DTYPE, L_NV, MutAnyOrigin],
    mhat: TileTensor[DTYPE, L_M, MutAnyOrigin],
    fnet: TileTensor[DTYPE, L_NV, MutAnyOrigin],
    qacc_ws: TileTensor[DTYPE, L_NV, MutAnyOrigin],
):
    """For every standalone free body: `A = M_hat_block + h*B_bias`, then
    `qacc_ws[block] = A^-1 fnet[block]` (`mjd_freeMhat` + `mju_solveLU6`).

    `mhat` is `M - h*qDeriv` as `_msub_qderiv_env` left it (the global LU
    factor lives in `scratch.L`, so the matrix is intact), `fnet` the re-solve
    rhs `M*qacc_constrained`, and `qacc_ws` the global solve's answer, whose
    six entries for this body are overwritten. The rows of a standalone body
    touch no other dof, so the rest of `qacc_ws` is unaffected.
    """
    var nv = dims.get_nv()
    var njoint = dims.get_njoint()
    var nbody = dims.get_nbody()
    comptime ZERO = Scalar[DTYPE](0)

    for j in range(njoint):
        if Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_TYPE])) != JNT_FREE:
            continue
        var body = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_BODY_ID]))
        var adr = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_DOF_ADR]))
        # `mj_isFreeBody`: exactly one joint on the body, no child body
        var njb = 0
        for jj in range(njoint):
            if Int(rebind[Scalar[DTYPE]](joints[jj, JOINT_IDX_BODY_ID])) == body:
                njb += 1
        if njb != 1:
            continue
        var has_child = False
        for b in range(1, nbody):
            if Int(rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_PARENT])) == body:
                has_child = True
        if has_child:
            continue
        if adr + 6 > nv:
            continue

        # A = M_hat block
        var A = Array[Scalar[DTYPE], 36](fill=ZERO)
        for r in range(6):
            for c in range(6):
                A[6 * r + c] = rebind[Scalar[DTYPE]](
                    mhat[env, (adr + r) * nv + adr + c]
                )

        # freeBias_vel_blocks
        var mass = rebind[Scalar[DTYPE]](bodies[body, BODY_IDX_MASS])
        var i0 = rebind[Scalar[DTYPE]](bodies[body, BODY_IDX_IXX])
        var i1 = rebind[Scalar[DTYPE]](bodies[body, BODY_IDX_IYY])
        var i2 = rebind[Scalar[DTYPE]](bodies[body, BODY_IDX_IZZ])
        var qx = rebind[Scalar[DTYPE]](xquat[env, body * 4 + 0])
        var qy = rebind[Scalar[DTYPE]](xquat[env, body * 4 + 1])
        var qz = rebind[Scalar[DTYPE]](xquat[env, body * 4 + 2])
        var qw = rebind[Scalar[DTYPE]](xquat[env, body * 4 + 3])
        var R = _q2m[DTYPE](qx, qy, qz, qw)
        var xi = quat_mul[DTYPE](
            qx, qy, qz, qw,
            rebind[Scalar[DTYPE]](bodies[body, BODY_IDX_IQUAT_X]),
            rebind[Scalar[DTYPE]](bodies[body, BODY_IDX_IQUAT_Y]),
            rebind[Scalar[DTYPE]](bodies[body, BODY_IDX_IQUAT_Z]),
            rebind[Scalar[DTYPE]](bodies[body, BODY_IDX_IQUAT_W]),
        )
        var Xi = _q2m[DTYPE](xi[0], xi[1], xi[2], xi[3])
        var s0 = rebind[Scalar[DTYPE]](xipos[env, body * 3 + 0]) - rebind[Scalar[DTYPE]](xpos[env, body * 3 + 0])
        var s1 = rebind[Scalar[DTYPE]](xipos[env, body * 3 + 1]) - rebind[Scalar[DTYPE]](xpos[env, body * 3 + 1])
        var s2 = rebind[Scalar[DTYPE]](xipos[env, body * 3 + 2]) - rebind[Scalar[DTYPE]](xpos[env, body * 3 + 2])
        var v0 = rebind[Scalar[DTYPE]](qvel[env, adr + 3])
        var v1 = rebind[Scalar[DTYPE]](qvel[env, adr + 4])
        var v2 = rebind[Scalar[DTYPE]](qvel[env, adr + 5])
        # world-frame angular velocity w = R * qvel_rot
        var w0 = R[0] * v0 + R[1] * v1 + R[2] * v2
        var w1 = R[3] * v0 + R[4] * v1 + R[5] * v2
        var w2 = R[6] * v0 + R[7] * v1 + R[8] * v2
        # Iw = Xi diag(I) Xi^T
        var XiI = Array[Scalar[DTYPE], 9](fill=ZERO)
        for i in range(3):
            XiI[3 * i + 0] = Xi[3 * i + 0] * i0
            XiI[3 * i + 1] = Xi[3 * i + 1] * i1
            XiI[3 * i + 2] = Xi[3 * i + 2] * i2
        var Iw = Array[Scalar[DTYPE], 9](fill=ZERO)
        Iw[0] = XiI[0] * Xi[0] + XiI[1] * Xi[1] + XiI[2] * Xi[2]
        Iw[4] = XiI[3] * Xi[3] + XiI[4] * Xi[4] + XiI[5] * Xi[5]
        Iw[8] = XiI[6] * Xi[6] + XiI[7] * Xi[7] + XiI[8] * Xi[8]
        Iw[1] = XiI[0] * Xi[3] + XiI[1] * Xi[4] + XiI[2] * Xi[5]
        Iw[3] = Iw[1]
        Iw[2] = XiI[0] * Xi[6] + XiI[1] * Xi[7] + XiI[2] * Xi[8]
        Iw[6] = Iw[2]
        Iw[5] = XiI[3] * Xi[6] + XiI[4] * Xi[7] + XiI[5] * Xi[8]
        Iw[7] = Iw[5]
        # ws = w x s ; Iww = Iw * w
        var ws0 = w1 * s2 - w2 * s1
        var ws1 = w2 * s0 - w0 * s2
        var ws2 = w0 * s1 - w1 * s0
        var Iww0 = Iw[0] * w0 + Iw[1] * w1 + Iw[2] * w2
        var Iww1 = Iw[3] * w0 + Iw[4] * w1 + Iw[5] * w2
        var Iww2 = Iw[6] * w0 + Iw[7] * w1 + Iw[8] * w2
        # K = s w^T - (w.s) I + [ws]_x
        var wds = w0 * s0 + w1 * s1 + w2 * s2
        var K = Array[Scalar[DTYPE], 9](fill=ZERO)
        K[0] = s0 * w0 - wds
        K[1] = s0 * w1 - ws2
        K[2] = s0 * w2 + ws1
        K[3] = s1 * w0 + ws2
        K[4] = s1 * w1 - wds
        K[5] = s1 * w2 - ws0
        K[6] = s2 * w0 - ws1
        K[7] = s2 * w1 + ws0
        K[8] = s2 * w2 - wds
        # lin = K * R
        var lin = Array[Scalar[DTYPE], 9](fill=ZERO)
        for r in range(3):
            for c in range(3):
                lin[3 * r + c] = K[3 * r + 0] * R[c] + K[3 * r + 1] * R[3 + c] + K[3 * r + 2] * R[6 + c]
        # C = -mass [s]_x K + [w]_x Iw - [Iww]_x, column by column
        var C = Array[Scalar[DTYPE], 9](fill=ZERO)
        for c in range(3):
            var sxk0 = s1 * K[6 + c] - s2 * K[3 + c]
            var sxk1 = s2 * K[c] - s0 * K[6 + c]
            var sxk2 = s0 * K[3 + c] - s1 * K[c]
            var wxi0 = w1 * Iw[6 + c] - w2 * Iw[3 + c]
            var wxi1 = w2 * Iw[c] - w0 * Iw[6 + c]
            var wxi2 = w0 * Iw[3 + c] - w1 * Iw[c]
            var t0 = ZERO
            var t1 = ZERO
            var t2 = ZERO
            if c == 1:
                t0 = Iww2
            elif c == 2:
                t0 = -Iww1
            if c == 0:
                t1 = -Iww2
            elif c == 2:
                t1 = Iww0
            if c == 0:
                t2 = Iww1
            elif c == 1:
                t2 = -Iww0
            C[c] = -mass * sxk0 + wxi0 + t0
            C[3 + c] = -mass * sxk1 + wxi1 + t1
            C[6 + c] = -mass * sxk2 + wxi2 + t2
        # rot = R^T C R
        var tmp = Array[Scalar[DTYPE], 9](fill=ZERO)
        for r in range(3):
            for c in range(3):
                tmp[3 * r + c] = R[r] * C[c] + R[3 + r] * C[3 + c] + R[6 + r] * C[6 + c]
        var rot = Array[Scalar[DTYPE], 9](fill=ZERO)
        for r in range(3):
            for c in range(3):
                rot[3 * r + c] = tmp[3 * r + 0] * R[c] + tmp[3 * r + 1] * R[3 + c] + tmp[3 * r + 2] * R[6 + c]
        # A -= h * d(qfrc_smooth)/d(qvel): qfrc_smooth carries -qfrc_bias
        var h_mass = -dt * mass
        for r in range(3):
            for c in range(3):
                A[6 * r + 3 + c] += h_mass * lin[3 * r + c]
                A[6 * (3 + r) + 3 + c] += dt * rot[3 * r + c]

        # solve A x = fnet[block] — Gaussian elimination with partial pivoting
        var x = Array[Scalar[DTYPE], 6](fill=ZERO)
        for r in range(6):
            x[r] = rebind[Scalar[DTYPE]](fnet[env, adr + r])
        var singular = False
        for k in range(6):
            var piv = k
            var best = abs(A[6 * k + k])
            for r in range(k + 1, 6):
                if abs(A[6 * r + k]) > best:
                    best = abs(A[6 * r + k])
                    piv = r
            if best < Scalar[DTYPE](1e-300):
                singular = True
                break
            if piv != k:
                for c in range(6):
                    var t = A[6 * k + c]
                    A[6 * k + c] = A[6 * piv + c]
                    A[6 * piv + c] = t
                var tx = x[k]
                x[k] = x[piv]
                x[piv] = tx
            for r in range(k + 1, 6):
                var f = A[6 * r + k] / A[6 * k + k]
                if f == ZERO:
                    continue
                for c in range(k, 6):
                    A[6 * r + c] -= f * A[6 * k + c]
                x[r] -= f * x[k]
        if singular:
            continue
        for k in range(5, -1, -1):
            var acc = x[k]
            for c in range(k + 1, 6):
                acc -= A[6 * k + c] * x[c]
            x[k] = acc / A[6 * k + k]
        for r in range(6):
            qacc_ws[env, adr + r] = x[r]


def _free_body_block_kernel[
    DTYPE: DType, NV: Int, NJOINT: Int, NBODY: Int, BATCH: Int
](
    dt: Scalar[DTYPE],
    joints: Tile[DTYPE, NJOINT, MODEL_JOINT_SIZE],
    bodies: Tile[DTYPE, NBODY, MODEL_BODY_SIZE],
    xpos: Tile[DTYPE, BATCH, NBODY * 3],
    xipos: Tile[DTYPE, BATCH, NBODY * 3],
    xquat: Tile[DTYPE, BATCH, NBODY * 4],
    qvel: Tile[DTYPE, BATCH, NV],
    mhat: Tile[DTYPE, BATCH, NV * NV],
    fnet: Tile[DTYPE, BATCH, NV],
    qacc_ws: Tile[DTYPE, BATCH, NV],
):
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    _free_body_block_env[DTYPE](
        env, dt, Dims[nv=NV, njoint=NJOINT, nbody=NBODY](),
        joints, bodies, xpos, xipos, xquat, qvel, mhat, fnet, qacc_ws,
    )


@always_inline
def _mrhs_env[
    DTYPE: DType,
    D: DimsLike,
    L_M: TensorLayout,
    L_NV: TensorLayout](
    env: Int,
    dims: D,
    M: TileTensor[DTYPE, L_M, MutAnyOrigin],
    qacc_c: TileTensor[DTYPE, L_NV, MutAnyOrigin],
    fnet: TileTensor[DTYPE, L_NV, MutAnyOrigin],
):
    """`fnet = M * qacc_constrained` — the force MuJoCo re-solves against.

    ⚠⚠ THIS MUST RUN WHILE `M` IS STILL THE PLAIN MASS MATRIX, before
    `_msub_qderiv_env` turns it into M_hat in place. `qfrc_smooth +
    qfrc_constraint` is exactly `M * qacc_constrained`: the constraint solver
    reports an ACCELERATION, and multiplying it back by the same `M` it was
    solved against recovers the total force without needing the solver to
    hand out `qfrc_constraint` separately.
    """
    var nv = dims.get_nv()
    for i in range(nv):
        var acc = Scalar[DTYPE](0)
        for j in range(nv):
            acc += rebind[Scalar[DTYPE]](M[env, i * nv + j]) * rebind[
                Scalar[DTYPE]
            ](qacc_c[env, j])
        fnet[env, i] = acc


@always_inline
def _adopt_qacc_env[
    DTYPE: DType,
    D: DimsLike,
    L_NV: TensorLayout](
    env: Int,
    dims: D,
    qacc_ws: TileTensor[DTYPE, L_NV, MutAnyOrigin],
    qacc_c: TileTensor[DTYPE, L_NV, MutAnyOrigin],
):
    """`qacc_constrained <- qacc_ws`, the M_hat re-solve's answer."""
    var nv = dims.get_nv()
    for i in range(nv):
        qacc_c[env, i] = rebind[Scalar[DTYPE]](qacc_ws[env, i])


# ── implicit finalize: v += dt*qacc ; integrate qpos (quat-aware) ─────────
@always_inline
def _implicit_finalize_env[
    DTYPE: DType,
    D: DimsLike,
    L_QPOS: TensorLayout,
    L_QVEL: TensorLayout,
    L_JOINTS: TensorLayout](
    env: Int,
    dt: Scalar[DTYPE],
    dims: D,
    qpos: TileTensor[DTYPE, L_QPOS, MutAnyOrigin],
    qvel: TileTensor[DTYPE, L_QVEL, MutAnyOrigin],
    qacc: TileTensor[DTYPE, L_QVEL, MutAnyOrigin],
    joints: TileTensor[DTYPE, L_JOINTS, MutAnyOrigin],
    qacc_constrained: TileTensor[DTYPE, L_QVEL, MutAnyOrigin],
):
    var nv = dims.get_nv()
    var njoint = dims.get_njoint()
    # Velocity update straight from the (constrained) implicit qacc — NO
    # dt*D re-solve (M_hat already carries the implicit terms).
    #
    # ⚠ THE BOUND IS `mjMAXVAL`, NOT A STABILITY BUDGET — see
    # `MJ_MAXVAL`. MuJoCo resets on a bad velocity, it never rescales one.
    for i in range(nv):
        var qacc_final = rebind[Scalar[DTYPE]](qacc_constrained[env, i])
        qacc[env, i] = qacc_final
        var qvel_new = rebind[Scalar[DTYPE]](qvel[env, i]) + qacc_final * dt
        var qvel_max = Scalar[DTYPE](MJ_MAXVAL)
        if qvel_new != qvel_new:  # NaN guard
            qvel_new = Scalar[DTYPE](0.0)
        elif qvel_new > qvel_max:
            qvel_new = qvel_max
        elif qvel_new < -qvel_max:
            qvel_new = -qvel_max
        qvel[env, i] = qvel_new

    # Position integration (verbatim from euler finalize).
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
            var qw = rebind[Scalar[DTYPE]](qpos[env, jnt_qpos_adr + 3])
            var qx = rebind[Scalar[DTYPE]](qpos[env, jnt_qpos_adr + 4])
            var qy = rebind[Scalar[DTYPE]](qpos[env, jnt_qpos_adr + 5])
            var qz = rebind[Scalar[DTYPE]](qpos[env, jnt_qpos_adr + 6])
            var wx = rebind[Scalar[DTYPE]](qvel[env, jnt_dof_adr + 3])
            var wy = rebind[Scalar[DTYPE]](qvel[env, jnt_dof_adr + 4])
            var wz = rebind[Scalar[DTYPE]](qvel[env, jnt_dof_adr + 5])
            var result = quat_integrate(qx, qy, qz, qw, wx, wy, wz, dt)
            var norm = quat_normalize(result[0], result[1], result[2], result[3])
            qpos[env, jnt_qpos_adr + 3] = norm[3]
            qpos[env, jnt_qpos_adr + 4] = norm[0]
            qpos[env, jnt_qpos_adr + 5] = norm[1]
            qpos[env, jnt_qpos_adr + 6] = norm[2]
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
            qpos[env, jnt_qpos_adr + 0] = bnorm[3]
            qpos[env, jnt_qpos_adr + 1] = bnorm[0]
            qpos[env, jnt_qpos_adr + 2] = bnorm[1]
            qpos[env, jnt_qpos_adr + 3] = bnorm[2]

        elif jnt_type == JNT_HINGE or jnt_type == JNT_SLIDE:
            var qp = rebind[Scalar[DTYPE]](qpos[env, jnt_qpos_adr])
            var qv = rebind[Scalar[DTYPE]](qvel[env, jnt_dof_adr])
            qpos[env, jnt_qpos_adr] = qp + qv * dt


# ── launchable kernels ────────────────────────────────────────────────────
def _qderiv_damping_kernel[
    DTYPE: DType, NV: Int, NJOINT: Int, NACT: Int, BATCH: Int
](
    joints: Tile[DTYPE, NJOINT, MODEL_JOINT_SIZE],
    njoint_arg: Int64,
    actd: Tile[DTYPE, NV, 1],
    actd_live: Tile[DTYPE, BATCH, NV],
    atrn: Tile[DTYPE, NACT * ACTDAMP_TRN_SIZE, 1],
    aact: Tile[DTYPE, BATCH, NACT],
    meta: Tile[DTYPE, BATCH, METADATA_SIZE],
    qderiv: Tile[DTYPE, BATCH, NV * NV],
):
    # Mojo 1.0: `Int`/`UInt` are not `DevicePassable`; the kernel takes
    # a fixed-width `Int64` and re-binds the original name here.
    var njoint = Int(njoint_arg)
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    _qderiv_damping_env[DTYPE](
        env, Dims[nv=NV, njoint=NJOINT, nact=NACT](), joints, njoint, actd,
        actd_live, atrn, aact, meta, qderiv
    )


def _msub_qderiv_kernel[
    DTYPE: DType, NV: Int, BATCH: Int
](
    dt: Scalar[DTYPE],
    M: Tile[DTYPE, BATCH, NV * NV],
    qderiv: Tile[DTYPE, BATCH, NV * NV],
):
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    _msub_qderiv_env[DTYPE](env, dt, Dims[nv=NV](), M, qderiv)


def _mrhs_kernel[
    DTYPE: DType, NV: Int, BATCH: Int
](
    M: Tile[DTYPE, BATCH, NV * NV],
    qacc_c: Tile[DTYPE, BATCH, NV],
    fnet: Tile[DTYPE, BATCH, NV],
):
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    _mrhs_env[DTYPE](env, Dims[nv=NV](), M, qacc_c, fnet)


def _adopt_qacc_kernel[
    DTYPE: DType, NV: Int, BATCH: Int
](
    qacc_ws: Tile[DTYPE, BATCH, NV],
    qacc_c: Tile[DTYPE, BATCH, NV],
):
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    _adopt_qacc_env[DTYPE](env, Dims[nv=NV](), qacc_ws, qacc_c)


def _implicit_finalize_kernel[
    DTYPE: DType, NQ: Int, NV: Int, NJOINT: Int, BATCH: Int
](
    dt: Scalar[DTYPE],
    qpos: Tile[DTYPE, BATCH, NQ],
    qvel: Tile[DTYPE, BATCH, NV],
    qacc: Tile[DTYPE, BATCH, NV],
    joints: Tile[DTYPE, NJOINT, MODEL_JOINT_SIZE],
    qacc_constrained: Tile[DTYPE, BATCH, NV],
):
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    _implicit_finalize_env[DTYPE](
        env, dt, Dims[nq=NQ, nv=NV, njoint=NJOINT](), qpos, qvel, qacc, joints, qacc_constrained
    )


# ── the stateful integrator ───────────────────────────────────────────────
struct ImplicitIntegrator[
    DTYPE: DType,
    D: DimsLike,
    CONE_TYPE: Int = ConeType.ELLIPTIC,
    BATCH: Int = 1,
    SOLVER: StaticString = "pgs",
    PARALLEL_GPU: Bool = False,
    CRBA_TREEWALK: Bool = False,
    SKIP_RNE_DERIV: Bool = False,
    # ⚠⚠ THIS DID NOT EXIST, AND ITS ABSENCE WAS SILENT. `solve_newton`
    # defaults it to 3 and `_contact_solve_env` CLAMPS each contact's own
    # condim down to it, so every model stepped by an implicit integrator
    # solved its contacts at condim 3 whatever the file declared — torsional
    # and rolling friction dropped, with no diagnostic. `EulerIntegrator` has
    # carried the parameter since the elliptic cone was generalised; this twin
    # was never given it. Measured on apptronik_apollo, whose soles are
    # `<pair condim="6">`: worst |d(qpos)| against MuJoCo after ONE step
    # 1.856e-03, against 5.551e-17 for the same model with its pairs edited
    # down to condim 3 — i.e. the entire divergence was the dropped rows.
    MAX_CONDIM: Int = 3,
    # ⚠⚠ AND NEITHER DID THIS ONE, WITH THE SAME SHAPE OF SILENCE. `mj_solNoSlip`
    # runs behind `solve_newton`'s `NOSLIP_ITER`, which defaults to 0 — so an
    # implicitfast model asking for `<option noslip_iterations>` had the pass
    # dropped and nothing said so. `EulerIntegrator` and `RK4Integrator` have
    # both carried the parameter since the pass was ported; this twin did not,
    # exactly as it did not carry `MAX_CONDIM` above. 50 of the 131 loadable
    # models in this tree are implicitfast.
    #
    # ⚠ IT IS AN ENABLE, NOT A COUNT (2026-08-25). > 0 emits the pass; the
    # number of sweeps is `opt.noslip_iterations` from model meta. So passing
    # 1 here is "build it in", not "run one sweep".
    NOSLIP_ITER: Int = 0,
](Movable):
    """Owns its scratch (dynamics + contact + implicit); steps full-implicit
    dynamics on either target. See module docstring for the algorithm and
    what is not yet ported. PARALLEL_GPU / CRBA_TREEWALK behave as in
    EulerIntegrator for the shared FK/CRBA/RNE stages; the LU + qDeriv
    stages are serial per-env."""

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

    var iscratch: ImplicitScratch[Self.DTYPE, Self.D, Self.BATCH]

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
        self.iscratch = ImplicitScratch[Self.DTYPE, Self.D, Self.BATCH](dims)

    def prepare_gpu(mut self, ctx: DeviceContext) raises:
        self.scratch.upload_all(ctx)
        self.cscratch.upload_all(ctx)
        self.iscratch.upload_all(ctx)

    def step[
        target: StaticString, CONTACTS: Bool = True
    ](
        mut self,
        mut d: Data[Self.DTYPE, Self.D, Self.BATCH],
        mut m: Model[Self.DTYPE, Self.D],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """One full implicit step."""
        var dt = m.meta.data[MODEL_META_IDX_TIMESTEP]
        var njoint = Int(m.meta.data[MODEL_META_IDX_NJOINT])

        # ── kinematics + composite inertia + mass matrix (as euler) ──────
        forward_kinematics[target, Self.DTYPE, BATCH=Self.BATCH, PARALLEL = Self.PARALLEL_GPU](d, m, ctx)
        compute_body_velocities[target, Self.DTYPE, BATCH=Self.BATCH, PARALLEL = Self.PARALLEL_GPU](d, m, ctx)
        compute_subtree_com[target, Self.DTYPE, BATCH=Self.BATCH](d, m, ctx)
        compute_cdof[target, Self.DTYPE, BATCH=Self.BATCH, PARALLEL = Self.PARALLEL_GPU](d, m, self.scratch, ctx)
        compute_mass_matrix[target, Self.DTYPE, BATCH=Self.BATCH, PARALLEL = Self.PARALLEL_GPU, TREEWALK = Self.CRBA_TREEWALK](d, m, self.scratch, ctx)

        comptime L_JOINT = RowMajor[Self.D.NJOINT, MODEL_JOINT_SIZE]
        comptime L_ACTD = RowMajor[Self.D.NV, 1]
        # ⚠ FLOORED AT 1. `CAP_NACT` is 0 on a model with no actuators and a
        # zero-extent layout is a null view — the `site_xpos` trap. The
        # tensor is allocated with the same floor.
        comptime NACT_G = Self.D.CAP_NACT if Self.D.CAP_NACT > 0 else 1
        comptime L_M = RowMajor[Self.BATCH, Self.D.NV * Self.D.NV]
        comptime L_NV = RowMajor[Self.BATCH, Self.D.NV]
        comptime L_QPOS = RowMajor[Self.BATCH, Self.D.NQ]
        comptime BLOCKS = (Self.BATCH + IM_TPB - 1) // IM_TPB

        # ── armature: M diag += armature ─────────────────────────────────
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
                block_dim=(IM_TPB,),
            )


        # ── LU factor the PLAIN M (+ M^-1 for the constraint solver) ────
        #
        # ⚠⚠ PLAIN M, NOT M_hat, AND THE ORDER IS THE WHOLE POINT. MuJoCo's
        # `mj_step` is `mj_forward` then `mj_implicit`: the constraint rows are
        # built AND SOLVED against the plain mass matrix, and only then does
        # the integrator form `M_hat = M - dt*qDeriv` and RE-SOLVE
        # `qacc = M_hat^-1 (qfrc_smooth + qfrc_constraint)`
        # (`engine_forward.c:1983` then `:2003`).
        #
        # This used to form M_hat first and hand `M_hat^-1` to the solver, so
        # every constraint row was solved against a mass matrix the reference
        # never uses. With no active rows the two orderings agree exactly —
        # which is why spot's implicitfast first step matched to 2.851622 —
        # and they diverge as soon as a row carries force. Measured on
        # sharpa_wave, whose 22 dof-friction rows are live from step 0: the
        # thumb's acceleration came out -1.86249 against MuJoCo's -1.56997,
        # and the one-dof algebra says exactly that: with R = 29.23 recovered
        # from MuJoCo's own efc_force, `a = a0*R/(K+R)` gives -1.902 at
        # K = 1/M and -1.8629 at K = 1/M_hat.
        lu_factor[target, Self.DTYPE, BATCH=Self.BATCH](self.scratch, ctx)
        compute_m_inv_from_lu[target, Self.DTYPE, BATCH=Self.BATCH](self.scratch, ctx)

        # ── RNE bias forces ──────────────────────────────────────────────
        compute_bias_forces_rne[target, Self.DTYPE, BATCH=Self.BATCH, PARALLEL = Self.PARALLEL_GPU](d, m, self.scratch, ctx)

        # ── fnet = qfrc - bias - damping*qvel - stiffness - friction ─────
        comptime if target == "cpu":
            var dm = d.dims
            var rl_QPOS = rl2(Self.BATCH, dm.get_nq())
            var rl_NV = rl2(Self.BATCH, dm.get_nv())
            var rl_JOINT = rl2(dm.get_njoint(), MODEL_JOINT_SIZE)
            var qpos_v = d.qpos.tt_dyn["cpu", DYN2](rl_QPOS)
            var qvel_v = d.qvel.tt_dyn["cpu", DYN2](rl_NV)
            var qfrc_v = d.qfrc.tt_dyn["cpu", DYN2](rl_NV)
            var joints_v = m.joints.tt_dyn["cpu", DYN2](rl_JOINT)
            var bias_v = self.scratch.bias.tt_dyn["cpu", DYN2](rl_NV)
            var fnet_v = self.scratch.fnet.tt_dyn["cpu", DYN2](rl_NV)
            for e in range(Self.BATCH):
                _fnet_passive_env[
                    Self.DTYPE](e, dm, qpos_v, qvel_v, qfrc_v, joints_v, bias_v, fnet_v)
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
                block_dim=(IM_TPB,),
            )

        # Fluid drag into fnet (no-op unless meta density/viscosity > 0).
        compute_fluid_forces[target, Self.DTYPE, BATCH=Self.BATCH](d, m, self.scratch, ctx)
        # ⚠ AFTER the fluid call, not before: `mj_passive` adds `qfrc_fluid`
        # into `qfrc_passive` and only then `qfrc_gravcomp`
        # (engine_passive.c:1000-1022). No-op unless the model declares
        # `<body gravcomp>`.
        compute_gravcomp_forces[target, Self.DTYPE, BATCH=Self.BATCH](d, m, self.scratch, ctx)

        # ── LU solve: qacc_ws = M^-1 fnet (the SMOOTH acceleration) ─────
        lu_solve[target, Self.DTYPE, BATCH=Self.BATCH](self.scratch, ctx)

        # ── qacc writeback: qacc + qacc_constrained = qacc_ws ────────────
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
                block_dim=(IM_TPB,),
            )


        # ── the tree LDL of the PLAIN M, for the noslip ──────────────────
        #
        # ⚠⚠ THE NOSLIP DOES NOT READ `M^-1` ANY MORE. On the CPU with a
        # kinematic tree, `noslip._minv_apply` solves against `scratch.L` /
        # `scratch.D` (`mj_solveLD` on `qLD`, engine_core_smooth.c) — and the
        # Euler and RK4 steps fill those with `ldl_factor` before their
        # Newton, so nothing there noticed. This step only ever LU-factored
        # M (`lu_factor` above) and left `scratch.L/D` holding whatever the
        # last caller wrote — nothing, on a fresh Data — so every implicitfast
        # model with contacts and `NOSLIP_ITER > 0` solved its noslip against
        # an unfactored slab. On the Menagerie board that was unitree_g1
        # 5.9e-17 -> 1.0e-02 and kinova_gen3 2.4e-15 -> 4.5e-02 after ONE
        # step, and a g1 that bounced off the floor in the studio; the bench
        # models never saw it because none of them says `implicitfast`.
        #
        # MuJoCo keeps the two factors apart: `qLD` is M's (made in
        # `mj_makeM`/`mj_factorM` for every integrator) and the implicit
        # `qH = M - h*qDeriv` is factored separately (engine_forward.c:1812).
        # Same here: factor M now, while `scratch.M` IS the plain M — it is
        # turned into M_hat in place further down. Gated exactly like the
        # noslip's own `tree_ok`, so no other leg pays for it.
        #
        # ⚠⚠ AFTER THE SMOOTH `lu_solve`, NOT BESIDE `lu_factor`. `lu_factor`
        # keeps ITS factor in the same `scratch.L` / `scratch.D` slabs
        # (`dynamics/lu.mojo`); factoring here first overwrote the LU the
        # smooth solve above still had to read, and kinova_gen3 went from
        # 4.5e-02 to 3.3e-01. From this line to the M_hat re-factor below
        # nothing reads the LU of the plain M, so the slabs are the LDL's.
        comptime if CONTACTS and Self.SOLVER == "newton" and Self.NOSLIP_ITER > 0:
            comptime if target == "cpu":
                if Int(m.meta.data[MODEL_META_IDX_NTREE]) > 0:
                    ldl_factor[target, Self.DTYPE, BATCH=Self.BATCH, PARALLEL = Self.PARALLEL_GPU](m, self.scratch, ctx)

        # ── constraint seam (mirrors euler; uses M^-1 of the PLAIN M) ───
        comptime if CONTACTS:
            detect_contacts_auto[target, Self.DTYPE, BATCH=Self.BATCH](d, m, ctx)
            comptime assert (
                Self.SOLVER == "pgs"
                or Self.SOLVER == "newton"
                or Self.SOLVER == "cg"
                or Self.SOLVER == "island"
            ), (
                "ImplicitIntegrator: SOLVER must be 'pgs', 'newton',"
                " 'cg', or 'island'"
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
                        # ⚠ `solve_cg`, `solve_island_pgs` and `solve_contacts`
                        # DO NOT TAKE `MAX_CONDIM` — they have no such
                        # parameter, and `solve_contacts` calls
                        # `_contact_solve_env` (which does) without one. So
                        # those three are condim-3-only, on BOTH integrators;
                        # `EulerIntegrator` forwards the parameter to
                        # `solve_newton` alone for the same reason. The studio
                        # only ever builds `newton`, which is why this is the
                        # call that had to change first.
                        solve_contacts[target, Self.DTYPE, CONE_TYPE=Self.CONE_TYPE, BATCH=Self.BATCH](d, m, self.scratch, self.cscratch, ctx)
        else:
            solve_limits[target, Self.DTYPE, BATCH=Self.BATCH](d, m, self.scratch, ctx)

        # `qacc_warmstart = qacc` — the tail of `mj_forward`
        # (engine_forward.c:1087). ⚠ HERE AND NOT AFTER THE INTEGRATOR: MuJoCo
        # saves the CONSTRAINT SOLVER's acceleration, and anything the
        # integrator does to `qacc` afterwards never reaches `qacc_warmstart`.
        save_qacc_warmstart[target, Self.DTYPE, BATCH=Self.BATCH](
            d, self.scratch, ctx
        )

        # ── the implicit re-solve: qacc = M_hat^-1 * (M * qacc_constrained)
        #
        # `M * qacc_constrained` IS `qfrc_smooth + qfrc_constraint` — the
        # solver hands back an acceleration, and multiplying by the same M it
        # was solved against recovers the force MuJoCo re-solves with. It has
        # to happen while `M` is still plain, which is why it comes before the
        # qDeriv block below rather than after it.
        comptime if target == "cpu":
            var dm_r = d.dims
            var rl_M_r = rl2(Self.BATCH, dm_r.get_nv() * dm_r.get_nv())
            var rl_NV_r = rl2(Self.BATCH, dm_r.get_nv())
            var M_r = self.scratch.M.tt_dyn["cpu", DYN2](rl_M_r)
            var qc_r = self.scratch.qacc_constrained.tt_dyn["cpu", DYN2](rl_NV_r)
            var fnet_r = self.scratch.fnet.tt_dyn["cpu", DYN2](rl_NV_r)
            for e in range(Self.BATCH):
                _mrhs_env[Self.DTYPE](e, dm_r, M_r, qc_r, fnet_r)
        else:
            ctx.value().enqueue_function[
                _mrhs_kernel[Self.DTYPE, Self.D.NV, Self.BATCH]
            ](
                self.scratch.M.tt["gpu", Self.BATCH, Self.D.NV * Self.D.NV](),
                self.scratch.qacc_constrained.tt["gpu", Self.BATCH, Self.D.NV](),
                self.scratch.fnet.tt["gpu", Self.BATCH, Self.D.NV](),
                grid_dim=(BLOCKS,),
                block_dim=(IM_TPB,),
            )

        # ── qDeriv = damping diagonal, then subtract RNE velocity deriv ──
        comptime if target == "cpu":
            var dm = d.dims
            var rl_JOINT = rl2(dm.get_njoint(), MODEL_JOINT_SIZE)
            var rl_M = rl2(Self.BATCH, dm.get_nv() * dm.get_nv())
            var joints_v = m.joints.tt_dyn["cpu", DYN2](rl_JOINT)
            var qd_v = self.iscratch.qderiv.tt_dyn["cpu", DYN2](rl_M)
            var rl_ACTD = rl2(dm.get_nv(), 1)
            var actd_v = m.dof_actdamp.tt_dyn["cpu", DYN2](rl_ACTD)
            var rl_ACTDL = rl2(Self.BATCH, dm.get_nv())
            var actdl_v = d.dof_actdamp.tt_dyn["cpu", DYN2](rl_ACTDL)
            # ⚠ `nact` FROM THE **DATA** PROVIDER, not from a comptime member
            # — on the runtime leg those are `DIM_POISON`. `_at_least_one`
            # because a model with no actuators still binds this tensor.
            var _na = dm.get_nact()
            if _na < 1:
                _na = 1
            var rl_ATRN = rl2(_na * ACTDAMP_TRN_SIZE, 1)
            var atrn_v = m.actdamp_trn.tt_dyn["cpu", DYN2](rl_ATRN)
            var rl_AACT = rl2(Self.BATCH, _na)
            var aact_v = d.actdamp_act.tt_dyn["cpu", DYN2](rl_AACT)
            var rl_META = rl2(Self.BATCH, METADATA_SIZE)
            var meta_v = d.meta.tt_dyn["cpu", DYN2](rl_META)
            for e in range(Self.BATCH):
                _qderiv_damping_env[
                    Self.DTYPE](e, dm, joints_v, njoint, actd_v, actdl_v,
                                atrn_v, aact_v, meta_v, qd_v)
        else:
            ctx.value().enqueue_function[
                _qderiv_damping_kernel[
                    Self.DTYPE, Self.D.NV, Self.D.NJOINT, NACT_G,
                    Self.BATCH
                ]
            ](
                m.joints.tt["gpu", Self.D.NJOINT, MODEL_JOINT_SIZE](),
                Int64(njoint),
                m.dof_actdamp.tt["gpu", Self.D.NV, 1](),
                d.dof_actdamp.tt["gpu", Self.BATCH, Self.D.NV](),
                m.actdamp_trn.tt["gpu", NACT_G * ACTDAMP_TRN_SIZE, 1](),
                d.actdamp_act.tt["gpu", Self.BATCH, NACT_G](),
                d.meta.tt["gpu", Self.BATCH, METADATA_SIZE](),
                self.iscratch.qderiv.tt["gpu", Self.BATCH, Self.D.NV * Self.D.NV](),
                grid_dim=(BLOCKS,),
                block_dim=(IM_TPB,),
            )
        # ⚠⚠ THIS ONE FLAG IS THE DIFFERENCE BETWEEN `implicit` AND
        # `implicitfast`, and it is MuJoCo's own: `mj_implicitSkip` calls
        # `mjd_smooth_vel(m, d, flg_bias)` with 1 for `implicit` and 0 for
        # `implicitfast` (`engine_forward.c:1794` vs `:1806`). `flg_bias`
        # gates exactly this term — the dense RNE velocity derivative
        # (Coriolis/centrifugal).
        #
        # Skipping it is not only cheaper. Without it qDeriv is SYMMETRIC, so
        # MuJoCo factorises `implicitfast` with its ordinary Cholesky and
        # keeps LU for `implicit`. We use LU for both: correct either way, and
        # a second factorisation path is a second thing to keep in step for a
        # speed difference the studio does not need. Noted so the choice reads
        # as a decision rather than an oversight.
        comptime if not Self.SKIP_RNE_DERIV:
            compute_rne_vel_derivative[target, Self.DTYPE, Self.BATCH](d, m, self.scratch, self.iscratch, ctx)

        # ── M_hat = M - dt * qDeriv ──────────────────────────────────────
        comptime if target == "cpu":
            var dm = d.dims
            var rl_M = rl2(Self.BATCH, dm.get_nv() * dm.get_nv())
            var M_v = self.scratch.M.tt_dyn["cpu", DYN2](rl_M)
            var qd_v = self.iscratch.qderiv.tt_dyn["cpu", DYN2](rl_M)
            for e in range(Self.BATCH):
                _msub_qderiv_env[Self.DTYPE](
                    e, dt, dm, M_v, qd_v
                )
        else:
            ctx.value().enqueue_function[
                _msub_qderiv_kernel[Self.DTYPE, Self.D.NV, Self.BATCH]
            ](
                dt,
                self.scratch.M.tt["gpu", Self.BATCH, Self.D.NV * Self.D.NV](),
                self.iscratch.qderiv.tt["gpu", Self.BATCH, Self.D.NV * Self.D.NV](),
                grid_dim=(BLOCKS,),
                block_dim=(IM_TPB,),
            )

        # M_hat is formed; factor it and re-solve. `compute_m_inv_from_lu` is
        # deliberately NOT re-run: nothing downstream of here reads `m_inv`,
        # and M_hat^-1 is not what the constraint rows were solved against.
        lu_factor[target, Self.DTYPE, BATCH=Self.BATCH](self.scratch, ctx)
        lu_solve[target, Self.DTYPE, BATCH=Self.BATCH](self.scratch, ctx)

        # ── implicitfast: standalone free bodies get the gyroscopic derivative
        # through a local unsymmetric 6x6 solve (AUD-44, 3.11 f0fa3d82). Under
        # full `implicit` the RNE derivative already carries it.
        comptime if Self.SKIP_RNE_DERIV:
            comptime if target == "cpu":
                var dm_f = d.dims
                var nb_f = dm_f.get_nbody()
                var rl_M_f = rl2(Self.BATCH, dm_f.get_nv() * dm_f.get_nv())
                var rl_NV_f = rl2(Self.BATCH, dm_f.get_nv())
                var rl_J_f = rl2(dm_f.get_njoint(), MODEL_JOINT_SIZE)
                var rl_B_f = rl2(nb_f, MODEL_BODY_SIZE)
                var rl_X3_f = rl2(Self.BATCH, nb_f * 3)
                var rl_X4_f = rl2(Self.BATCH, nb_f * 4)
                var joints_f = m.joints.tt_dyn["cpu", DYN2](rl_J_f)
                var bodies_f = m.bodies.tt_dyn["cpu", DYN2](rl_B_f)
                var xpos_f = d.xpos.tt_dyn["cpu", DYN2](rl_X3_f)
                var xipos_f = d.xipos.tt_dyn["cpu", DYN2](rl_X3_f)
                var xquat_f = d.xquat.tt_dyn["cpu", DYN2](rl_X4_f)
                var qvel_f = d.qvel.tt_dyn["cpu", DYN2](rl_NV_f)
                var M_f = self.scratch.M.tt_dyn["cpu", DYN2](rl_M_f)
                var fnet_f = self.scratch.fnet.tt_dyn["cpu", DYN2](rl_NV_f)
                var qws_f = self.scratch.qacc_ws.tt_dyn["cpu", DYN2](rl_NV_f)
                for e in range(Self.BATCH):
                    _free_body_block_env[Self.DTYPE](
                        e, dt, dm_f, joints_f, bodies_f, xpos_f, xipos_f,
                        xquat_f, qvel_f, M_f, fnet_f, qws_f,
                    )
            else:
                ctx.value().enqueue_function[
                    _free_body_block_kernel[
                        Self.DTYPE, Self.D.NV, Self.D.NJOINT, Self.D.NBODY,
                        Self.BATCH
                    ]
                ](
                    dt,
                    m.joints.tt["gpu", Self.D.NJOINT, MODEL_JOINT_SIZE](),
                    m.bodies.tt["gpu", Self.D.NBODY, MODEL_BODY_SIZE](),
                    d.xpos.tt["gpu", Self.BATCH, Self.D.NBODY * 3](),
                    d.xipos.tt["gpu", Self.BATCH, Self.D.NBODY * 3](),
                    d.xquat.tt["gpu", Self.BATCH, Self.D.NBODY * 4](),
                    d.qvel.tt["gpu", Self.BATCH, Self.D.NV](),
                    self.scratch.M.tt["gpu", Self.BATCH, Self.D.NV * Self.D.NV](),
                    self.scratch.fnet.tt["gpu", Self.BATCH, Self.D.NV](),
                    self.scratch.qacc_ws.tt["gpu", Self.BATCH, Self.D.NV](),
                    grid_dim=(BLOCKS,),
                    block_dim=(IM_TPB,),
                )

        comptime if target == "cpu":
            var dm_a = d.dims
            var rl_NV_a = rl2(Self.BATCH, dm_a.get_nv())
            var qws_a = self.scratch.qacc_ws.tt_dyn["cpu", DYN2](rl_NV_a)
            var qc_a = self.scratch.qacc_constrained.tt_dyn["cpu", DYN2](rl_NV_a)
            for e in range(Self.BATCH):
                _adopt_qacc_env[Self.DTYPE](e, dm_a, qws_a, qc_a)
        else:
            ctx.value().enqueue_function[
                _adopt_qacc_kernel[Self.DTYPE, Self.D.NV, Self.BATCH]
            ](
                self.scratch.qacc_ws.tt["gpu", Self.BATCH, Self.D.NV](),
                self.scratch.qacc_constrained.tt["gpu", Self.BATCH, Self.D.NV](),
                grid_dim=(BLOCKS,),
                block_dim=(IM_TPB,),
            )

        # ── implicit finalize: v += dt*qacc ; integrate qpos ─────────────
        comptime if target == "cpu":
            var dm = d.dims
            var rl_QPOS = rl2(Self.BATCH, dm.get_nq())
            var rl_NV = rl2(Self.BATCH, dm.get_nv())
            var rl_JOINT = rl2(dm.get_njoint(), MODEL_JOINT_SIZE)
            var qpos_v = d.qpos.tt_dyn["cpu", DYN2](rl_QPOS)
            var qvel_v = d.qvel.tt_dyn["cpu", DYN2](rl_NV)
            var qacc_v = d.qacc.tt_dyn["cpu", DYN2](rl_NV)
            var joints_v = m.joints.tt_dyn["cpu", DYN2](rl_JOINT)
            var qacc_c_v = self.scratch.qacc_constrained.tt_dyn["cpu", DYN2](rl_NV)
            for e in range(Self.BATCH):
                _implicit_finalize_env[
                    Self.DTYPE](e, dt, dm, qpos_v, qvel_v, qacc_v, joints_v, qacc_c_v)
        else:
            ctx.value().enqueue_function[
                _implicit_finalize_kernel[
                    Self.DTYPE, Self.D.NQ, Self.D.NV, Self.D.NJOINT, Self.BATCH
                ]
            ](
                dt,
                d.qpos.tt["gpu", Self.BATCH, Self.D.NQ](),
                d.qvel.tt["gpu", Self.BATCH, Self.D.NV](),
                d.qacc.tt["gpu", Self.BATCH, Self.D.NV](),
                m.joints.tt["gpu", Self.D.NJOINT, MODEL_JOINT_SIZE](),
                self.scratch.qacc_constrained.tt["gpu", Self.BATCH, Self.D.NV](),
                grid_dim=(BLOCKS,),
                block_dim=(IM_TPB,),
            )
