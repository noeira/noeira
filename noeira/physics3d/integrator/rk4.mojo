"""Stateful RK4 integrator over per-field tensors (migration P2).

`RK4Integrator` is the stateful replacement for the stateless
`RK4Integrator.step_gpu` + caller-provided workspace slab: the struct OWNS
its `DynamicsScratch` + `Rk4Scratch` and sequences the single-source
per-stage functions into a full contact-free RK4 step (MuJoCo
mj_RungeKutta tableau, exactly as the legacy `rk4_stage_kernel` /
`rk4_combine_kernel` compute it):

    save (q0, v0)                                     [stage-0 setup]
    for stage s in 0..3:
        s==1: A[0]=qacc_c; q = pos(q0, v0,  dt/2); v = v0 + dt/2*A[0]
        s==2: A[1]=qacc_c; C1 = v0 + dt/2*A[0];
              q = pos(q0, C1, dt/2); v = v0 + dt/2*A[1]
        s==3: A[2]=qacc_c; C2 = v0 + dt/2*A[1];
              q = pos(q0, C2, dt);   v = v0 + dt*A[2]
        dynamics chain: FK -> body velocities -> subtree_com -> cdof ->
            CRBA -> +armature (NO implicit damping for RK4) -> LDL factor
            (+ M_inv) -> RNE -> fnet (qfrc - bias - damping - stiffness -
            frictionloss, all EXPLICIT) -> LDL solve -> qacc writeback
            (d.qacc + qacc_constrained)
    combine: qacc = (A[0] + 2*A[1] + 2*A[2] + A[3])/6 with A[3] read from
        qacc_constrained; C[3] = v0 + dt*A[2];
        v_comb = (v0 + 2*C1 + 2*C2 + C[3])/6 (NaN guard + clamp, stored in
        the A0 slot exactly like legacy); qvel = v0 + qacc*dt (NaN guard +
        clamp); qpos = pos(q0, v_comb, dt) quaternion-aware.

Staging/combine arithmetic is verbatim from the legacy `rk4_stage_kernel`
(:1232) / `rk4_combine_kernel` (:2140) / `_integrate_pos_gpu` (:151);
armature/fnet/writeback env bodies are shared with `euler` (the
legacy RK4 stage steps 6b/9/9b are byte-identical to the Euler ones —
verified against both sources). Unlike Euler's finalize, RK4 does NO
implicit-damping re-solve: damping is explicit in fnet, and the combine
integrates directly.

Deliberately NOT yet ported (raise / absent by design):
- fluid forces (density/viscosity > 0) — raise on use, like Euler fields.
- contacts AND limits: legacy RK4 handles both inside the per-stage
  constraint-SOLVER launch (`step_gpu` = 4 x (stage + solver) + combine);
  the contact-free legacy path (stage kernels only, no solver launches)
  never touches limits — so unlike `EulerIntegrator` there is no
  CONTACTS param and no standalone limits stage here. Contacts-in-RK4
  (solver seam per stage) is a later slice.
"""

from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceContext
from layout import TileTensor, TensorLayout
from noeira.core.tiles import RowMajor, Tile

from ..kinematics.quat_math import quat_integrate, quat_normalize
from ..kinematics.forward_kinematics import (
    forward_kinematics,
    compute_body_velocities,
)
from ..dynamics.subtree_com import compute_subtree_com
from ..dynamics.cdof import compute_cdof
from ..dynamics.mass_matrix import compute_mass_matrix
from ..dynamics.ldl import (
    ldl_factor,
    ldl_solve,
    compute_m_inv,
)
from ..types import ConeType
from ..dynamics.rne import compute_bias_forces_rne
from ..dynamics.fluid_forces import compute_fluid_forces
from ..dynamics.gravcomp import compute_gravcomp_forces
from ..dynamics.actuation import apply_actions_fields
from ..dynamics.pose_transmission import apply_pose_transmission
from ..constraints.contact_solve import solve_contacts
from ..solver.newton_solve import solve_newton
from ..solver.warmstart import save_qacc_warmstart
from ..solver.je_budget import je_ws_size
from ..solver.cg_solve import solve_cg
from ..solver.island_pgs_solve import solve_island_pgs
from ..collision.broadphase_sap import detect_contacts_auto
from ..joint_types import JNT_FREE, JNT_BALL, JNT_HINGE, JNT_SLIDE
from ..fields import (
    AsStatic,
    Dims,
    Dims,
    DimsLike,
    DimsLike,
    DimsLike,
    Data,
    Model,
    DynamicsScratch,
    ContactScratch,
    Rk4Scratch,
    SpecFields,
    Dims,
    DimsLike,
    DYN2,
    rl2,
)
from ..gpu.constants import (
    MODEL_META_IDX_NTREE,
    MJ_MAXVAL,
    MODEL_JOINT_SIZE,
    MODEL_META_IDX_TIMESTEP,
    MODEL_META_IDX_DENSITY,
    MODEL_META_IDX_VISCOSITY,
    JOINT_IDX_TYPE,
    JOINT_IDX_QPOS_ADR,
    JOINT_IDX_DOF_ADR,
)
from .euler import (
    EU_TPB,
    _armature_env,
    _armature_kernel,
    _fnet_passive_env,
    _fnet_passive_kernel,
    _qacc_writeback_env,
    _qacc_writeback_kernel,
)


# ── position integration: qpos = q0 + vel * dt, quaternion-aware ──────────
# (verbatim legacy _integrate_pos_gpu :151; num_joints -> comptime NJOINT,
#  ws rk4 regions -> per-field q0/vel tensors, state qpos -> qpos tensor)
@always_inline
def _rk4_integrate_pos_env[
    DTYPE: DType,
    D: DimsLike,
    L_JOINTS: TensorLayout,
    L_Q0: TensorLayout,
    L_VEL: TensorLayout](
    env: Int,
    dt: Scalar[DTYPE],
    dims: D,
    joints: TileTensor[DTYPE, L_JOINTS, MutAnyOrigin],
    q0: TileTensor[DTYPE, L_Q0, MutAnyOrigin],
    vel: TileTensor[DTYPE, L_VEL, MutAnyOrigin],
    qpos: TileTensor[DTYPE, L_Q0, MutAnyOrigin],
):
    var njoint = dims.get_njoint()
    for j in range(njoint):
        var jnt_type = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_TYPE]))
        var qpos_adr = Int(
            rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_QPOS_ADR])
        )
        var dof_adr = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_DOF_ADR]))

        if jnt_type == JNT_FREE:
            # Position: simple addition
            for d in range(3):
                var q0_d = rebind[Scalar[DTYPE]](q0[env, qpos_adr + d])
                var v_d = rebind[Scalar[DTYPE]](vel[env, dof_adr + d])
                qpos[env, qpos_adr + d] = q0_d + v_d * dt
            # Quaternion: exponential map integration.
            # MuJoCo qpos layout: [tx, ty, tz, qw, qx, qy, qz]
            # Our internal convention: (x, y, z, w)
            var qw = rebind[Scalar[DTYPE]](q0[env, qpos_adr + 3])
            var qx = rebind[Scalar[DTYPE]](q0[env, qpos_adr + 4])
            var qy = rebind[Scalar[DTYPE]](q0[env, qpos_adr + 5])
            var qz = rebind[Scalar[DTYPE]](q0[env, qpos_adr + 6])
            var wx = rebind[Scalar[DTYPE]](vel[env, dof_adr + 3])
            var wy = rebind[Scalar[DTYPE]](vel[env, dof_adr + 4])
            var wz = rebind[Scalar[DTYPE]](vel[env, dof_adr + 5])
            var result = quat_integrate(qx, qy, qz, qw, wx, wy, wz, dt)
            var norm = quat_normalize(
                result[0], result[1], result[2], result[3]
            )
            # Write back in MuJoCo qpos layout: [qw, qx, qy, qz]
            qpos[env, qpos_adr + 3] = norm[3]  # qw
            qpos[env, qpos_adr + 4] = norm[0]  # qx
            qpos[env, qpos_adr + 5] = norm[1]  # qy
            qpos[env, qpos_adr + 6] = norm[2]  # qz

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
            var bqw = rebind[Scalar[DTYPE]](q0[env, qpos_adr + 0])
            var bqx = rebind[Scalar[DTYPE]](q0[env, qpos_adr + 1])
            var bqy = rebind[Scalar[DTYPE]](q0[env, qpos_adr + 2])
            var bqz = rebind[Scalar[DTYPE]](q0[env, qpos_adr + 3])
            var bwx = rebind[Scalar[DTYPE]](vel[env, dof_adr + 0])
            var bwy = rebind[Scalar[DTYPE]](vel[env, dof_adr + 1])
            var bwz = rebind[Scalar[DTYPE]](vel[env, dof_adr + 2])
            var bres = quat_integrate(bqx, bqy, bqz, bqw, bwx, bwy, bwz, dt)
            var bnorm = quat_normalize(bres[0], bres[1], bres[2], bres[3])
            qpos[env, qpos_adr + 0] = bnorm[3]
            qpos[env, qpos_adr + 1] = bnorm[0]
            qpos[env, qpos_adr + 2] = bnorm[1]
            qpos[env, qpos_adr + 3] = bnorm[2]

        elif jnt_type == JNT_HINGE or jnt_type == JNT_SLIDE:
            var q0_val = rebind[Scalar[DTYPE]](q0[env, qpos_adr])
            var v_val = rebind[Scalar[DTYPE]](vel[env, dof_adr])
            qpos[env, qpos_adr] = q0_val + v_val * dt


# ── per-stage setup (verbatim rk4_stage_kernel pre-stage block :1316) ─────
@always_inline
def _rk4_stage_setup_env[
    DTYPE: DType,
    STAGE: Int,
    D: DimsLike,
    L_JOINTS: TensorLayout,
    L_QPOS: TensorLayout,
    L_QVEL: TensorLayout](
    env: Int,
    dt: Scalar[DTYPE],
    dims: D,
    joints: TileTensor[DTYPE, L_JOINTS, MutAnyOrigin],
    qpos: TileTensor[DTYPE, L_QPOS, MutAnyOrigin],
    qvel: TileTensor[DTYPE, L_QVEL, MutAnyOrigin],
    qacc_constrained: TileTensor[DTYPE, L_QVEL, MutAnyOrigin],
    q0: TileTensor[DTYPE, L_QPOS, MutAnyOrigin],
    v0: TileTensor[DTYPE, L_QVEL, MutAnyOrigin],
    A0: TileTensor[DTYPE, L_QVEL, MutAnyOrigin],
    A1: TileTensor[DTYPE, L_QVEL, MutAnyOrigin],
    A2: TileTensor[DTYPE, L_QVEL, MutAnyOrigin],
    C1: TileTensor[DTYPE, L_QVEL, MutAnyOrigin],
    C2: TileTensor[DTYPE, L_QVEL, MutAnyOrigin],
):
    var nq = dims.get_nq()
    var nv = dims.get_nv()
    var half_dt = dt * Scalar[DTYPE](0.5)

    comptime if STAGE == 0:
        # Save initial state to scratch
        for i in range(nq):
            q0[env, i] = qpos[env, i]
        for i in range(nv):
            v0[env, i] = qvel[env, i]
    elif STAGE == 1:
        # Save A[0] from qacc_constrained
        for i in range(nv):
            A0[env, i] = qacc_constrained[env, i]
        # Set intermediate state: qpos = q0 + dt/2 * v0 (C[0] = v0)
        # qvel = v0 + dt/2 * A[0]
        _rk4_integrate_pos_env[DTYPE](
            env, half_dt, dims, joints, q0, v0, qpos
        )
        for i in range(nv):
            var v0_i = rebind[Scalar[DTYPE]](v0[env, i])
            var a0_i = rebind[Scalar[DTYPE]](A0[env, i])
            qvel[env, i] = v0_i + half_dt * a0_i
    elif STAGE == 2:
        # Save A[1] from qacc_constrained
        for i in range(nv):
            A1[env, i] = qacc_constrained[env, i]
        # C[1] = v0 + dt/2 * A[0] — save for combine
        for i in range(nv):
            var v0_i = rebind[Scalar[DTYPE]](v0[env, i])
            var a0_i = rebind[Scalar[DTYPE]](A0[env, i])
            C1[env, i] = v0_i + half_dt * a0_i
        # Set intermediate state: qpos = q0 + dt/2 * C[1]
        # qvel = v0 + dt/2 * A[1]
        _rk4_integrate_pos_env[DTYPE](
            env, half_dt, dims, joints, q0, C1, qpos
        )
        for i in range(nv):
            var v0_i = rebind[Scalar[DTYPE]](v0[env, i])
            var a1_i = rebind[Scalar[DTYPE]](A1[env, i])
            qvel[env, i] = v0_i + half_dt * a1_i
    elif STAGE == 3:
        # Save A[2] from qacc_constrained
        for i in range(nv):
            A2[env, i] = qacc_constrained[env, i]
        # C[2] = v0 + dt/2 * A[1] — save for combine
        for i in range(nv):
            var v0_i = rebind[Scalar[DTYPE]](v0[env, i])
            var a1_i = rebind[Scalar[DTYPE]](A1[env, i])
            C2[env, i] = v0_i + half_dt * a1_i
        # Set intermediate state: qpos = q0 + dt * C[2]
        # qvel = v0 + dt * A[2]
        _rk4_integrate_pos_env[DTYPE](
            env, dt, dims, joints, q0, C2, qpos
        )
        for i in range(nv):
            var v0_i = rebind[Scalar[DTYPE]](v0[env, i])
            var a2_i = rebind[Scalar[DTYPE]](A2[env, i])
            qvel[env, i] = v0_i + dt * a2_i


# ── combine: RK4 weights + integrate (verbatim rk4_combine_kernel :2140) ──
@always_inline
def _rk4_combine_env[
    DTYPE: DType,
    D: DimsLike,
    L_JOINTS: TensorLayout,
    L_QPOS: TensorLayout,
    L_QVEL: TensorLayout](
    env: Int,
    dt: Scalar[DTYPE],
    dims: D,
    joints: TileTensor[DTYPE, L_JOINTS, MutAnyOrigin],
    qpos: TileTensor[DTYPE, L_QPOS, MutAnyOrigin],
    qvel: TileTensor[DTYPE, L_QVEL, MutAnyOrigin],
    qacc: TileTensor[DTYPE, L_QVEL, MutAnyOrigin],
    qacc_constrained: TileTensor[DTYPE, L_QVEL, MutAnyOrigin],
    q0: TileTensor[DTYPE, L_QPOS, MutAnyOrigin],
    v0: TileTensor[DTYPE, L_QVEL, MutAnyOrigin],
    A0: TileTensor[DTYPE, L_QVEL, MutAnyOrigin],
    A1: TileTensor[DTYPE, L_QVEL, MutAnyOrigin],
    A2: TileTensor[DTYPE, L_QVEL, MutAnyOrigin],
    C1: TileTensor[DTYPE, L_QVEL, MutAnyOrigin],
    C2: TileTensor[DTYPE, L_QVEL, MutAnyOrigin],
):
    var nv = dims.get_nv()
    comptime ONE_SIXTH = Scalar[DTYPE](1.0 / 6.0)
    comptime ONE_THIRD = Scalar[DTYPE](1.0 / 3.0)

    # Read A[3] from qacc_constrained (stage 3 just ran)
    # Compute qacc_combined = (A[0] + 2*A[1] + 2*A[2] + A[3]) / 6
    # Compute C[3] = v0 + dt * A[2]
    # v_combined = (C[0] + 2*C[1] + 2*C[2] + C[3]) / 6  where C[0] = v0

    # First pass: compute qacc_combined, v_combined, update qvel/qacc.
    # Store v_combined in the A0 slot (no longer needed after this) —
    # exactly like the legacy combine kernel.
    for i in range(nv):
        var a0_i = rebind[Scalar[DTYPE]](A0[env, i])
        var a1_i = rebind[Scalar[DTYPE]](A1[env, i])
        var a2_i = rebind[Scalar[DTYPE]](A2[env, i])
        var a3_i = rebind[Scalar[DTYPE]](qacc_constrained[env, i])
        var v0_i = rebind[Scalar[DTYPE]](v0[env, i])
        var c1_i = rebind[Scalar[DTYPE]](C1[env, i])
        var c2_i = rebind[Scalar[DTYPE]](C2[env, i])

        # Combined acceleration
        var qacc_i = (
            ONE_SIXTH * a0_i
            + ONE_THIRD * a1_i
            + ONE_THIRD * a2_i
            + ONE_SIXTH * a3_i
        )

        # C[3] = v0 + dt * A[2]
        var c3_i = v0_i + dt * a2_i

        # Combined velocity — stored in A0 for position integration.
        # NaN guard: if any stage produced NaN qacc, c1/c2/c3 are NaN and
        # the position integration below would spread it.
        #
        # ⚠ THE SATURATION BOUND IS `mjMAXVAL` — see `MJ_MAXVAL`. It is a
        # numeric backstop on an already-diverged sim, not a speed limit.
        var v_combined_i = (
            ONE_SIXTH * v0_i
            + ONE_THIRD * c1_i
            + ONE_THIRD * c2_i
            + ONE_SIXTH * c3_i
        )
        var vpos_max = Scalar[DTYPE](MJ_MAXVAL)
        if v_combined_i != v_combined_i:  # NaN guard: no position change
            v_combined_i = Scalar[DTYPE](0.0)
        elif v_combined_i > vpos_max:
            v_combined_i = vpos_max
        elif v_combined_i < -vpos_max:
            v_combined_i = -vpos_max
        A0[env, i] = v_combined_i

        # Integrate: qvel = v0 + qacc * dt (NaN guard + `mjMAXVAL`)
        var qvel_new = v0_i + qacc_i * dt
        var qvel_max = Scalar[DTYPE](MJ_MAXVAL)
        if qvel_new != qvel_new:  # NaN guard: reset to zero
            qvel_new = Scalar[DTYPE](0.0)
        elif qvel_new > qvel_max:
            qvel_new = qvel_max
        elif qvel_new < -qvel_max:
            qvel_new = -qvel_max
        qvel[env, i] = qvel_new
        qacc[env, i] = qacc_i

    # Second pass: integrate position using v_combined (quaternion-aware);
    # v_combined lives in the A0 slot.
    _rk4_integrate_pos_env[DTYPE](
        env, dt, dims, joints, q0, A0, qpos
    )


# ── launchable kernels ────────────────────────────────────────────────────
def _rk4_stage_setup_kernel[
    DTYPE: DType, NQ: Int, NV: Int, NJOINT: Int, BATCH: Int, STAGE: Int
](
    dt: Scalar[DTYPE],
    joints: Tile[DTYPE, NJOINT, MODEL_JOINT_SIZE],
    qpos: Tile[DTYPE, BATCH, NQ],
    qvel: Tile[DTYPE, BATCH, NV],
    qacc_constrained: Tile[DTYPE, BATCH, NV],
    q0: Tile[DTYPE, BATCH, NQ],
    v0: Tile[DTYPE, BATCH, NV],
    A0: Tile[DTYPE, BATCH, NV],
    A1: Tile[DTYPE, BATCH, NV],
    A2: Tile[DTYPE, BATCH, NV],
    C1: Tile[DTYPE, BATCH, NV],
    C2: Tile[DTYPE, BATCH, NV],
):
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    _rk4_stage_setup_env[DTYPE, STAGE](
        env, dt, Dims[nq=NQ, nv=NV, njoint=NJOINT](), joints, qpos, qvel, qacc_constrained,
        q0, v0, A0, A1, A2, C1, C2,
    )


def _rk4_combine_kernel[
    DTYPE: DType, NQ: Int, NV: Int, NJOINT: Int, BATCH: Int
](
    dt: Scalar[DTYPE],
    joints: Tile[DTYPE, NJOINT, MODEL_JOINT_SIZE],
    qpos: Tile[DTYPE, BATCH, NQ],
    qvel: Tile[DTYPE, BATCH, NV],
    qacc: Tile[DTYPE, BATCH, NV],
    qacc_constrained: Tile[DTYPE, BATCH, NV],
    q0: Tile[DTYPE, BATCH, NQ],
    v0: Tile[DTYPE, BATCH, NV],
    A0: Tile[DTYPE, BATCH, NV],
    A1: Tile[DTYPE, BATCH, NV],
    A2: Tile[DTYPE, BATCH, NV],
    C1: Tile[DTYPE, BATCH, NV],
    C2: Tile[DTYPE, BATCH, NV],
):
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    _rk4_combine_env[DTYPE](
        env, dt, Dims[nq=NQ, nv=NV, njoint=NJOINT](), joints, qpos, qvel, qacc, qacc_constrained,
        q0, v0, A0, A1, A2, C1, C2,
    )


# ── the stateful integrator ───────────────────────────────────────────────
struct RK4Integrator[
    DTYPE: DType,
    D: DimsLike,
    CONE_TYPE: Int = ConeType.ELLIPTIC,
    BATCH: Int = 1,
    SOLVER: StaticString = "pgs",
    PARALLEL_GPU: Bool = False,
    CRBA_TREEWALK: Bool = False,
    MAX_CONDIM: Int = 3,
    NOSLIP_ITER: Int = 0,
](Movable):
    """Owns its scratch; steps RK4 dynamics on either target. With
    CONTACTS=True (default), each stage is followed by contact detection +
    the PGS contact solve (joint limits inside, legacy position) — matching
    the legacy per-stage solver launch. CONTACTS=False = unconstrained
    stages (the original contact-free pilot gates).
    PARALLEL_GPU=True: the GPU FK / body-velocity / cdof / CRBA /
    LDL-factor / M^-1 / RNE stages run their cooperative within-env (_mt)
    kernels (bit-exact vs serial; other stages stay serial). CPU ignores
    it. CRBA_TREEWALK=True: the CRBA runs the tree-walk algorithm
    (O(NV·depth)) instead of the dense O(NV²·NBODY) one —
    float-tolerance-equal, NOT bit-exact vs dense; mirrors the legacy
    USE_TREEWALK_MM selection (rk4_integrator.mojo:1540). ⚠ IT APPLIES ON
    BOTH TARGETS NOW; it used to be GPU-only, which left every CPU caller
    on the dense kernel — 12.6× slower on Sawyer (NV=15, NBODY=34)."""

    var scratch: DynamicsScratch[Self.DTYPE, Self.D, Self.BATCH]

    var rk4: Rk4Scratch[Self.DTYPE, Self.D, Self.BATCH]
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
        self.rk4 = Rk4Scratch[Self.DTYPE, Self.D, Self.BATCH](dims)
        self.cscratch = ContactScratch[
            Self.DTYPE, Self.D, Self.BATCH, Self.JE_WS
        ](dims)

    def prepare_gpu(mut self, ctx: DeviceContext) raises:
        """Allocate device buffers for the scratch (once, before stepping)."""
        self.scratch.upload_all(ctx)
        self.rk4.upload_all(ctx)
        self.cscratch.upload_all(ctx)

    def _stage_dynamics[
        target: StaticString, CONTACTS: Bool = True
    ](
        mut self,
        mut d: Data[Self.DTYPE, Self.D, Self.BATCH],
        mut m: Model[Self.DTYPE, Self.D],
        ctx: Optional[DeviceContext],
    ) raises:
        """One RK4 stage's forward-dynamics chain (same order as the legacy
        rk4_stage_kernel: FK -> body velocities -> subtree_com -> cdof ->
        CRBA -> +armature (no implicit damping) -> LDL factor + M_inv ->
        RNE -> fnet passive (all explicit) -> LDL solve -> qacc writeback).
        Legacy also runs contact DETECTION here; contact-free slice skips
        it (detection writes only contact state, never qpos/qvel/qacc)."""
        forward_kinematics[target, Self.DTYPE, BATCH=Self.BATCH, PARALLEL = Self.PARALLEL_GPU](d, m, ctx)
        compute_body_velocities[target, Self.DTYPE, BATCH=Self.BATCH, PARALLEL = Self.PARALLEL_GPU](d, m, ctx)
        compute_subtree_com[target, Self.DTYPE, BATCH=Self.BATCH](d, m, ctx)
        compute_cdof[target, Self.DTYPE, BATCH=Self.BATCH, PARALLEL = Self.PARALLEL_GPU](d, m, self.scratch, ctx)
        compute_mass_matrix[target, Self.DTYPE, BATCH=Self.BATCH, PARALLEL = Self.PARALLEL_GPU, TREEWALK = Self.CRBA_TREEWALK](d, m, self.scratch, ctx)

        comptime L_JOINT = RowMajor[Self.D.NJOINT, MODEL_JOINT_SIZE]
        comptime L_M = RowMajor[Self.BATCH, Self.D.NV * Self.D.NV]
        comptime L_NV = RowMajor[Self.BATCH, Self.D.NV]
        comptime L_QPOS = RowMajor[Self.BATCH, Self.D.NQ]
        comptime BLOCKS = (Self.BATCH + EU_TPB - 1) // EU_TPB

        # 6b. Armature ONLY (no implicit damping for RK4) — shared env body
        # with Euler (legacy step 6b is byte-identical in both kernels).
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

        ldl_factor[target, Self.DTYPE, BATCH=Self.BATCH, PARALLEL = Self.PARALLEL_GPU](m, self.scratch, ctx)
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
        # ⚠ `CONTACTS` IS PART OF THE PREDICATE — see the Euler twin: the
        # `CONTACTS=False` seam runs `solve_limits` / `solve_friction`, which
        # read the inverse.
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
        compute_bias_forces_rne[target, Self.DTYPE, BATCH=Self.BATCH, PARALLEL = Self.PARALLEL_GPU](d, m, self.scratch, ctx)

        # 9 + 9b. fnet = qfrc - bias - damping - stiffness - frictionloss
        # (shared env body with Euler; legacy RK4 steps 9/9b are
        # byte-identical to Euler's).
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

        # Fluid drag into fnet, per RK4 stage (no-op unless meta
        # density/viscosity > 0).
        compute_fluid_forces[target, Self.DTYPE, BATCH=Self.BATCH](d, m, self.scratch, ctx)
        # ⚠ AFTER the fluid call, not before: `mj_passive` adds `qfrc_fluid`
        # into `qfrc_passive` and only then `qfrc_gravcomp`
        # (engine_passive.c:1000-1022). No-op unless the model declares
        # `<body gravcomp>`.
        compute_gravcomp_forces[target, Self.DTYPE, BATCH=Self.BATCH](d, m, self.scratch, ctx)

        ldl_solve[target, Self.DTYPE, BATCH=Self.BATCH, PARALLEL = Self.PARALLEL_GPU](m, self.scratch, ctx)

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

    def _stage_setup[
        target: StaticString, STAGE: Int
    ](
        mut self,
        dt: Scalar[Self.DTYPE],
        mut d: Data[Self.DTYPE, Self.D, Self.BATCH],
        mut m: Model[Self.DTYPE, Self.D],
        ctx: Optional[DeviceContext],
    ) raises:
        comptime L_JOINT = RowMajor[Self.D.NJOINT, MODEL_JOINT_SIZE]
        comptime L_NV = RowMajor[Self.BATCH, Self.D.NV]
        comptime L_QPOS = RowMajor[Self.BATCH, Self.D.NQ]
        comptime BLOCKS = (Self.BATCH + EU_TPB - 1) // EU_TPB

        comptime if target == "cpu":
            var dm = d.dims
            var rl_JOINT = rl2(dm.get_njoint(), MODEL_JOINT_SIZE)
            var rl_QPOS = rl2(Self.BATCH, dm.get_nq())
            var rl_NV = rl2(Self.BATCH, dm.get_nv())
            var joints_v = m.joints.tt_dyn["cpu", DYN2](rl_JOINT)
            var qpos_v = d.qpos.tt_dyn["cpu", DYN2](rl_QPOS)
            var qvel_v = d.qvel.tt_dyn["cpu", DYN2](rl_NV)
            var qacc_c_v = self.scratch.qacc_constrained.tt_dyn["cpu", DYN2](rl_NV)
            var q0_v = self.rk4.q0.tt_dyn["cpu", DYN2](rl_QPOS)
            var v0_v = self.rk4.v0.tt_dyn["cpu", DYN2](rl_NV)
            var A0_v = self.rk4.A0.tt_dyn["cpu", DYN2](rl_NV)
            var A1_v = self.rk4.A1.tt_dyn["cpu", DYN2](rl_NV)
            var A2_v = self.rk4.A2.tt_dyn["cpu", DYN2](rl_NV)
            var C1_v = self.rk4.C1.tt_dyn["cpu", DYN2](rl_NV)
            var C2_v = self.rk4.C2.tt_dyn["cpu", DYN2](rl_NV)
            for e in range(Self.BATCH):
                _rk4_stage_setup_env[
                    Self.DTYPE,
                    STAGE](
                    e, dt, dm, joints_v, qpos_v, qvel_v, qacc_c_v,
                    q0_v, v0_v, A0_v, A1_v, A2_v, C1_v, C2_v,
                )
        else:
            ctx.value().enqueue_function[
                _rk4_stage_setup_kernel[
                    Self.DTYPE, Self.D.NQ, Self.D.NV, Self.D.NJOINT, Self.BATCH,
                    STAGE,
                ]
            ](
                dt,
                m.joints.tt["gpu", Self.D.NJOINT, MODEL_JOINT_SIZE](),
                d.qpos.tt["gpu", Self.BATCH, Self.D.NQ](),
                d.qvel.tt["gpu", Self.BATCH, Self.D.NV](),
                self.scratch.qacc_constrained.tt["gpu", Self.BATCH, Self.D.NV](),
                self.rk4.q0.tt["gpu", Self.BATCH, Self.D.NQ](),
                self.rk4.v0.tt["gpu", Self.BATCH, Self.D.NV](),
                self.rk4.A0.tt["gpu", Self.BATCH, Self.D.NV](),
                self.rk4.A1.tt["gpu", Self.BATCH, Self.D.NV](),
                self.rk4.A2.tt["gpu", Self.BATCH, Self.D.NV](),
                self.rk4.C1.tt["gpu", Self.BATCH, Self.D.NV](),
                self.rk4.C2.tt["gpu", Self.BATCH, Self.D.NV](),
                grid_dim=(BLOCKS,),
                block_dim=(EU_TPB,),
            )

    def step[
        target: StaticString, CONTACTS: Bool = True
    ](
        mut self,
        mut d: Data[Self.DTYPE, Self.D, Self.BATCH],
        mut m: Model[Self.DTYPE, Self.D],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """One full RK4 step (4 stages [+ per-stage contact/limit solve] +
        combine), with `d.qfrc` — the driver's actuator + applied force —
        HELD FIXED across the four stages.

        ⚠ THAT IS NOT WHAT MuJoCo DOES for an actuator whose force depends
        on the pose: `mj_RungeKutta` runs `mj_forwardSkip` at every stage,
        which re-evaluates `qfrc_actuator` at the stage's `qpos`/`qvel`. A
        `<motor>` on a joint is `gear*ctrl`, the same at every stage, so
        this entry is exact for it; a `<position>` servo, a spatial-tendon
        transmission or a body-fixed thrust (`<general site=...>` on a free
        body — crazyflie) is not. Use `step_actuated` for those: PERFORMANCE.md
        §13.34 measured the difference at 3.3e-13 per step on crazyflie,
        amplified to 2e-8 by fifty steps of tumbling flight, and a Python
        replica of `mj_RungeKutta` with the actuator frozen at stage 0
        matched THIS entry to 1.3e-23.
        """
        var dt = m.meta.data[MODEL_META_IDX_TIMESTEP]

        comptime for s in range(4):
            self._stage_setup[target, s](dt, d, m, ctx)
            self._stage_dynamics[target, CONTACTS](d, m, ctx)
            self._stage_constraints[target, CONTACTS](d, m, ctx)
        self._finish[target](dt, d, m, ctx)

    def step_actuated[
        target: StaticString,
        CONTACTS: Bool = True,
        NORMALIZED: Bool = False,
        DS: DimsLike = Self.D,
    ](
        mut self,
        mut d: Data[Self.DTYPE, Self.D, Self.BATCH],
        mut m: Model[Self.DTYPE, Self.D],
        sf: SpecFields[Self.DTYPE, DS],
        actions: List[Float64],
        act: List[Scalar[Self.DTYPE]],
        timestep: Float64,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """`step`, with the actuator forces RE-EVALUATED at stages 1-3 from
        each stage's own `qpos`/`qvel` — what `mj_RungeKutta` does.

        The caller has already applied `actions` at the start state (stage
        0 uses the `d.qfrc` it left, exactly as `step` does); here every
        later stage zeroes `d.qfrc` and re-runs `apply_actions_fields` +
        `apply_pose_transmission` at the stage state, so a servo reads the
        stage's `qpos`, a spatial tendon its stage moment arm, and a site
        wrench the stage's body orientation. `_stage_setup` writes the stage
        `qpos`/`qvel` before this runs, and `_stage_dynamics` recomputes FK
        a moment later from the same `qpos`, so the FK the transmission
        refreshes is not wasted work on a different pose.

        ⚠ `act` IS NOT ADVANCED HERE. `apply_actions_fields` integrates a
        dyntype's activation (filter, PID integral, slew) by `timestep` on
        every call, and the driver's call at stage 0 already did that for
        this step; the stages work on a COPY, so a dyntype actuator sees its
        start-of-step activation at every stage. MuJoCo carries `act` in the
        RK state and evaluates it at the stage (`act + h*A*act_dot`); no RK4
        model in the tree has a dyntype (crazyflie's rotors are `<general>`
        with none), so the difference is not measurable here yet.

        CPU, BATCH=1 (the actuation entry points are CPU and single-env).
        `NORMALIZED` is the env's `NORMALIZED_ACTIONS`.
        """
        comptime assert target == "cpu", "step_actuated: CPU only"
        comptime assert Self.BATCH == 1, "step_actuated: BATCH must be 1"
        var dt = m.meta.data[MODEL_META_IDX_TIMESTEP]
        var nv = d.dims.get_nv()
        # `apply_actions_fields` is typed on `Data[DTYPE, D2, 1]`; with
        # BATCH asserted 1 the two types are the same bytes, and the cast is
        # the no-op the assert makes it.
        var d1 = Pointer(to=d).unsafe_bitcast[Data[Self.DTYPE, Self.D, 1]]()

        comptime for s in range(4):
            self._stage_setup[target, s](dt, d, m, ctx)
            comptime if s > 0:
                var act_stage = act.copy()
                for i in range(nv):
                    d.qfrc.data[i] = Scalar[Self.DTYPE](0)
                apply_actions_fields[Self.DTYPE, NORMALIZED=NORMALIZED](
                    sf, d1[], actions, act_stage, timestep
                )
                apply_pose_transmission[Self.DTYPE](
                    sf, m, d, self.scratch, actions, act_stage, timestep
                )
            self._stage_dynamics[target, CONTACTS](d, m, ctx)
            self._stage_constraints[target, CONTACTS](d, m, ctx)
        self._finish[target](dt, d, m, ctx)

    def _stage_constraints[
        target: StaticString, CONTACTS: Bool
    ](
        mut self,
        mut d: Data[Self.DTYPE, Self.D, Self.BATCH],
        mut m: Model[Self.DTYPE, Self.D],
        ctx: Optional[DeviceContext],
    ) raises:
        """Per-stage constraint solve (legacy: solver launch after every
        stage kernel; corrects qacc_constrained before the next stage's
        A[k] snapshot / the combine)."""
        comptime if CONTACTS:
            # Auto broadphase = legacy production (the legacy stage
            # kernel calls detect_contacts_auto_gpu): SAP for
            # NGEOM >= 16, O(N^2) otherwise — routing is bit-identical
            # for every existing gate model (all NGEOM < 16).
            detect_contacts_auto[target, Self.DTYPE, BATCH=Self.BATCH](d, m, ctx)
            comptime assert (
                Self.SOLVER == "pgs"
                or Self.SOLVER == "newton"
                or Self.SOLVER == "cg"
                or Self.SOLVER == "island"
            ), (
                "RK4Integrator: SOLVER must be 'pgs', 'newton',"
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
                        solve_contacts[target, Self.DTYPE, CONE_TYPE=Self.CONE_TYPE, BATCH=Self.BATCH](d, m, self.scratch, self.cscratch, ctx)

    def _finish[
        target: StaticString
    ](
        mut self,
        dt: Scalar[Self.DTYPE],
        mut d: Data[Self.DTYPE, Self.D, Self.BATCH],
        mut m: Model[Self.DTYPE, Self.D],
        ctx: Optional[DeviceContext],
    ) raises:
        """After the fourth stage: save the warmstart, then the RK4 combine."""
        # `qacc_warmstart = qacc` — the tail of `mj_forward`
        # (engine_forward.c:1087). ⚠ HERE AND NOT AFTER THE INTEGRATOR: MuJoCo
        # saves the CONSTRAINT SOLVER's acceleration, and anything the
        # integrator does to `qacc` afterwards never reaches `qacc_warmstart`.
        save_qacc_warmstart[target, Self.DTYPE, BATCH=Self.BATCH](
            d, self.scratch, ctx
        )

        comptime L_JOINT = RowMajor[Self.D.NJOINT, MODEL_JOINT_SIZE]
        comptime L_NV = RowMajor[Self.BATCH, Self.D.NV]
        comptime L_QPOS = RowMajor[Self.BATCH, Self.D.NQ]
        comptime BLOCKS = (Self.BATCH + EU_TPB - 1) // EU_TPB

        comptime if target == "cpu":
            var dm = d.dims
            var rl_JOINT = rl2(dm.get_njoint(), MODEL_JOINT_SIZE)
            var rl_QPOS = rl2(Self.BATCH, dm.get_nq())
            var rl_NV = rl2(Self.BATCH, dm.get_nv())
            var joints_v = m.joints.tt_dyn["cpu", DYN2](rl_JOINT)
            var qpos_v = d.qpos.tt_dyn["cpu", DYN2](rl_QPOS)
            var qvel_v = d.qvel.tt_dyn["cpu", DYN2](rl_NV)
            var qacc_v = d.qacc.tt_dyn["cpu", DYN2](rl_NV)
            var qacc_c_v = self.scratch.qacc_constrained.tt_dyn["cpu", DYN2](rl_NV)
            var q0_v = self.rk4.q0.tt_dyn["cpu", DYN2](rl_QPOS)
            var v0_v = self.rk4.v0.tt_dyn["cpu", DYN2](rl_NV)
            var A0_v = self.rk4.A0.tt_dyn["cpu", DYN2](rl_NV)
            var A1_v = self.rk4.A1.tt_dyn["cpu", DYN2](rl_NV)
            var A2_v = self.rk4.A2.tt_dyn["cpu", DYN2](rl_NV)
            var C1_v = self.rk4.C1.tt_dyn["cpu", DYN2](rl_NV)
            var C2_v = self.rk4.C2.tt_dyn["cpu", DYN2](rl_NV)
            for e in range(Self.BATCH):
                _rk4_combine_env[
                    Self.DTYPE](
                    e, dt, dm, joints_v, qpos_v, qvel_v, qacc_v, qacc_c_v,
                    q0_v, v0_v, A0_v, A1_v, A2_v, C1_v, C2_v,
                )
        else:
            ctx.value().enqueue_function[
                _rk4_combine_kernel[
                    Self.DTYPE, Self.D.NQ, Self.D.NV, Self.D.NJOINT, Self.BATCH
                ]
            ](
                dt,
                m.joints.tt["gpu", Self.D.NJOINT, MODEL_JOINT_SIZE](),
                d.qpos.tt["gpu", Self.BATCH, Self.D.NQ](),
                d.qvel.tt["gpu", Self.BATCH, Self.D.NV](),
                d.qacc.tt["gpu", Self.BATCH, Self.D.NV](),
                self.scratch.qacc_constrained.tt["gpu", Self.BATCH, Self.D.NV](),
                self.rk4.q0.tt["gpu", Self.BATCH, Self.D.NQ](),
                self.rk4.v0.tt["gpu", Self.BATCH, Self.D.NV](),
                self.rk4.A0.tt["gpu", Self.BATCH, Self.D.NV](),
                self.rk4.A1.tt["gpu", Self.BATCH, Self.D.NV](),
                self.rk4.A2.tt["gpu", Self.BATCH, Self.D.NV](),
                self.rk4.C1.tt["gpu", Self.BATCH, Self.D.NV](),
                self.rk4.C2.tt["gpu", Self.BATCH, Self.D.NV](),
                grid_dim=(BLOCKS,),
                block_dim=(EU_TPB,),
            )
