"""Dry-friction (frictionloss) DOF constraints — MuJoCo `mjCNSTR_FRICTION_DOF`.

Replaces an EXPLICIT Coulomb force that could not arrest motion. The previous
implementation lived in `integrator/euler.mojo`'s `_fnet_passive_env` and read

    if v > 1e-4:  fnet -= floss
    elif v < -1e-4: fnet += floss

An explicit force of fixed magnitude cannot bring a velocity to rest: it
overshoots zero every step and settles into a PERIOD-2 LIMIT CYCLE, so a joint
that should stop dead instead oscillates forever. Measured on dm_control's
finger spinner (spun to 2 rad/s, no contact, no ctrl): MuJoCo decays to
1e-17 rad/s while ours locked bit-constant at +-0.0329 rad/s with qacc -+6.577.
It was invisible in the observation because the chatter period is exactly two
substeps and the env samples every `FRAME_SKIP`=2. The fixed point is
`v = floss / (2*M/dt - damping)`.

MuJoCo instead adds ONE CONSTRAINT ROW per frictional dof
(`mj_instantiateFriction`, engine_core_constraint.c:660-691), solved together
with limits and contacts, which can hold a joint exactly at rest because the
constraint force is solved for rather than prescribed:

    J        = e_i                    (a single dof)
    pos      = 0, margin = 0
    K        = 0                      (engine_core_constraint.c:1426-1428)
      => aref = -B*vel                (no position term)
    imp      = getimpedance(dof_solimp, pos=0) = dof_solimp[0] = dmin,
               since x = (pos-margin)/width = 0 lands on the saturated branch
    R        = (1-imp)/imp * dof_invweight0[i]        (mj_diagApprox :1120)
    B        = 2 / (dmax * timeconst)                 (          :1438-1441)
    force    = clamp(-jar/R, -frictionloss, +frictionloss)   (   :2307-2334)

where `jar = J*qacc - aref = qacc_i + B*vel_i`. That last line is the whole
difference from a joint limit: a limit clamps its multiplier to [0, inf), a
friction row clamps to the symmetric BOX [-floss, +floss], and a friction row
is ALWAYS present rather than conditional on a violation.

⚠ `dof_solref` / `dof_solimp` are MuJoCo's DEFAULTS here, not parsed. They are
distinct from the LIMIT parameters already in the model meta, and conflating
the two would be a real bug: walker and humanoid set
`solimplimit="0 .99 .01"` while leaving `solimpfriction` at the default
(0.9, 0.95, ...), so reusing `MODEL_META_IDX_SOLIMP_LIMIT_*` here would give
them a dmin of 0 and friction forces ~1e4x too soft. MJCF spells the friction
ones `solreffriction` / `solimpfriction`; no model in the repo sets either, and
`full_parser` RAISES if one appears, so this cannot go silently wrong. When a
model does need them, give them their own meta slots beside the limit ones.

Solved as an acceleration-level PGS sweep over `scratch.qacc_constrained`,
exactly like `limits.mojo`, and called from the same four sites (`contact_solve`
PGS, `newton_solve`, `cg_solve`, `island_pgs_solve`) plus the CONTACTS=False
standalone path.
"""

from std.math import abs
from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceContext
from layout import TileTensor, TensorLayout
from noeira.core.tiles import RowMajor, Tile

from ..joint_types import JNT_FREE, JNT_BALL
from ..fields import (
    Data,
    Model,
    DynamicsScratch,
    Dims,
    DimsLike,
    AsStatic,
    Scratch,
    cap,
    DYN1,
    DYN2,
    rl1,
    rl2,
)
from .constraint_data import refsafe_timeconst
from ..gpu.constants import (
    MODEL_META_IDX_TIMESTEP,
    MODEL_JOINT_SIZE,
    JOINT_IDX_TYPE,
    JOINT_IDX_DOF_ADR,
    JOINT_IDX_FRICTIONLOSS,
)

comptime FRIC_TPB: Int = 64

# MuJoCo defaults for dof_solref / dof_solimp (mjModel defaults; MJCF
# `solreffriction` / `solimpfriction`). Only dmin, dmax and timeconst are
# reachable: `pos` is identically 0 for a friction row, so the impedance sits
# on the saturated branch at dmin and width/midpoint/power never apply.
comptime DOF_SOLREF_TIMECONST: Float64 = 0.02
comptime DOF_SOLIMP_DMIN: Float64 = 0.9
comptime DOF_SOLIMP_DMAX: Float64 = 0.95

# engine_core_constraint.c:1284-1287
comptime MJF_MINIMP: Float64 = 0.0001
comptime MJF_MAXIMP: Float64 = 0.9999


@always_inline
def _max_one[N: Int]() -> Int:
    return N if N > 0 else 1


@always_inline
def _friction_env[
    DTYPE: DType,
    NUM_ITERATIONS: Int,
    D: DimsLike,
    L_QVEL: TensorLayout,
    L_JOINTS: TensorLayout,
    L_DOF_INVWEIGHT0: TensorLayout,
    L_M_INV: TensorLayout,
](
    env: Int,
    dims: D,
    qvel: TileTensor[DTYPE, L_QVEL, MutAnyOrigin],
    joints: TileTensor[DTYPE, L_JOINTS, MutAnyOrigin],
    # ⚠ REFSAFE needs the timestep, and `_friction_env` has no `meta`.
    # Passed as a scalar rather than threading the whole meta tensor: the
    # only thing this row type needs from the model options is `2*dt`.
    timestep: Scalar[DTYPE],
    dof_invweight0: TileTensor[DTYPE, L_DOF_INVWEIGHT0, MutAnyOrigin],
    m_inv: TileTensor[DTYPE, L_M_INV, MutAnyOrigin],
    qacc_constrained: TileTensor[DTYPE, L_QVEL, MutAnyOrigin],
):
    """Build + PGS-solve one friction row per frictional dof, for one env."""
    var nv = dims.get_nv()
    var njoint = dims.get_njoint()
    comptime FRIC_CAP = cap[D.NV]()

    var fric_dof = Scratch[Int, FRIC_CAP](nv, uninitialized=0)
    var fric_loss = Scratch[Scalar[DTYPE], FRIC_CAP](nv, uninitialized=0)
    var K_fric = Scratch[Scalar[DTYPE], FRIC_CAP](nv, uninitialized=0)
    var lambda_fric = Scratch[Scalar[DTYPE], FRIC_CAP](nv, uninitialized=0)
    var fric_bias = Scratch[Scalar[DTYPE], FRIC_CAP](nv, uninitialized=0)
    var fric_inv_K = Scratch[Scalar[DTYPE], FRIC_CAP](nv, uninitialized=0)
    for i in range(nv):
        fric_dof[i] = 0
        fric_loss[i] = Scalar[DTYPE](0)
        K_fric[i] = Scalar[DTYPE](1)
        lambda_fric[i] = Scalar[DTYPE](0)
        fric_bias[i] = Scalar[DTYPE](0)
        fric_inv_K[i] = Scalar[DTYPE](1)

    # One row per DOF with frictionloss > 0. MuJoCo stores frictionloss per
    # DOF; we store it per JOINT, so a free/ball joint's value expands across
    # its dofs exactly as damping and stiffness already do in `_fnet_passive`.
    var num_fric = 0
    for j in range(njoint):
        var floss = rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_FRICTIONLOSS])
        if floss <= Scalar[DTYPE](0):
            continue
        var jtype = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_TYPE]))
        var dof_adr = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_DOF_ADR]))
        var nd = 1
        if jtype == JNT_FREE:
            nd = 6
        elif jtype == JNT_BALL:
            nd = 3
        for d in range(nd):
            if num_fric >= nv:
                break
            var dof = dof_adr + d
            fric_dof[num_fric] = dof
            fric_loss[num_fric] = floss
            var kd = rebind[Scalar[DTYPE]](m_inv[env, dof * nv + dof])
            if kd < Scalar[DTYPE](1e-10):
                kd = Scalar[DTYPE](1e-10)
            K_fric[num_fric] = kd
            num_fric += 1

    if num_fric == 0:
        return

    # `pos` is identically 0, so the impedance is the saturated branch value
    # dmin (getimpedance, engine_core_constraint.c:1361-1365) — there is no
    # penetration to interpolate over.
    var imp = Scalar[DTYPE](DOF_SOLIMP_DMIN)
    if imp < Scalar[DTYPE](MJF_MINIMP):
        imp = Scalar[DTYPE](MJF_MINIMP)
    elif imp > Scalar[DTYPE](MJF_MAXIMP):
        imp = Scalar[DTYPE](MJF_MAXIMP)
    var dmax = Scalar[DTYPE](DOF_SOLIMP_DMAX)
    if dmax < Scalar[DTYPE](MJF_MINIMP):
        dmax = Scalar[DTYPE](MJF_MINIMP)
    elif dmax > Scalar[DTYPE](MJF_MAXIMP):
        dmax = Scalar[DTYPE](MJF_MAXIMP)
    # K = 0 for a friction row, so only B survives: aref = -B*vel.
    # ⚠ THE HARDCODED DEFAULT IS STILL SUBJECT TO REFSAFE (defect 23).
    # MuJoCo clamps `solreffriction[0]` to 2*timestep exactly as it clamps
    # solref (engine_core_constraint.c:2039), and that applies to the DEFAULT
    # 0.02 we substitute here just as much as to a declared value.
    var f_tc = refsafe_timeconst[DTYPE](
        Scalar[DTYPE](DOF_SOLREF_TIMECONST), timestep
    )
    var B_damp = Scalar[DTYPE](2.0) / (dmax * f_tc)

    comptime MINVJ_FRIC_CAP = cap[D.NV]() * cap[D.NV]()
    var fric_MinvJ = Scratch[Scalar[DTYPE], MINVJ_FRIC_CAP](
        nv * nv, uninitialized=0
    )
    for f in range(num_fric):
        var dof = fric_dof[f]
        # bias = -aref = +B*vel, so residual = a + bias + R*lambda = jar + R*l.
        fric_bias[f] = B_damp * rebind[Scalar[DTYPE]](qvel[env, dof])
        var diag = rebind[Scalar[DTYPE]](dof_invweight0[dof])
        if diag < Scalar[DTYPE](1e-10):
            diag = K_fric[f]  # Fallback, as in limits.mojo
        var R_f = (Scalar[DTYPE](1.0) - imp) / imp * diag
        fric_inv_K[f] = Scalar[DTYPE](1.0) / (K_fric[f] + R_f)
        for i in range(nv):
            fric_MinvJ[f * nv + i] = rebind[Scalar[DTYPE]](
                m_inv[env, i * nv + dof]
            )

    # PGS iterations (acceleration-level), identical in shape to the limit
    # sweep apart from the BOX clamp.
    for _ in range(NUM_ITERATIONS):
        var max_delta: Scalar[DTYPE] = 0
        for f in range(num_fric):
            var a_f = rebind[Scalar[DTYPE]](
                qacc_constrained[env, fric_dof[f]]
            )
            var R_f = Scalar[DTYPE](1.0) / fric_inv_K[f] - K_fric[f]
            var residual = a_f + fric_bias[f] + R_f * lambda_fric[f]
            var delta = -residual * fric_inv_K[f]
            var old_lam = lambda_fric[f]
            var lam = old_lam + rebind[Scalar[DTYPE]](delta)
            # THE box clamp — |force| <= frictionloss. This is what lets the
            # row hold a dof exactly at rest instead of overshooting it.
            if lam > fric_loss[f]:
                lam = fric_loss[f]
            elif lam < -fric_loss[f]:
                lam = -fric_loss[f]
            lambda_fric[f] = lam
            var actual = lam - old_lam
            var abs_d = abs(actual)
            if abs_d > max_delta:
                max_delta = abs_d
            for i in range(nv):
                qacc_constrained[env, i] += fric_MinvJ[f * nv + i] * actual
        if max_delta < Scalar[DTYPE](1e-4):
            break


def _friction_fields_kernel[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NJOINT: Int,
    BATCH: Int,
    NUM_ITERATIONS: Int,
](
    qvel: Tile[DTYPE, BATCH, NV],
    joints: Tile[DTYPE, NJOINT, MODEL_JOINT_SIZE],
    dof_invweight0: Tile[DTYPE, NV],
    m_inv: Tile[DTYPE, BATCH, NV * NV],
    qacc_constrained: Tile[DTYPE, BATCH, NV],
    # A SCALAR, not the meta tensor: the friction rows need exactly `2*dt` from
    # the model options, and a scalar kernel argument is the cheap, capture-safe
    # way to carry it.
    timestep: Scalar[DTYPE],
):
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    _friction_env[DTYPE, NUM_ITERATIONS](
        env, Dims[nq=NQ, nv=NV, njoint=NJOINT](), qvel, joints, timestep, dof_invweight0, m_inv, qacc_constrained
    )


def solve_friction[

    target: StaticString,
    DTYPE: DType,
    D: DimsLike,
    BATCH: Int = 1,
    NUM_ITERATIONS: Int = 50,
    # Appended, not grouped with NEXCLUDE — see `fields.Model`.
](
    mut d: Data[DTYPE, D, BATCH],
    mut m: Model[DTYPE, D],
    mut scratch: DynamicsScratch[DTYPE, D, BATCH],
    ctx: Optional[DeviceContext] = None,
) raises:
    """Solve dry-friction dof rows into `scratch.qacc_constrained`, both
    targets. Mirrors `solve_limits`; used on the CONTACTS=False path, where no
    solver runs and would otherwise call `_friction_env` itself."""
    comptime L_NV = RowMajor[BATCH, D.NV]
    comptime L_JOINT = RowMajor[D.NJOINT, MODEL_JOINT_SIZE]
    comptime L_DW = RowMajor[D.NV]
    comptime L_M = RowMajor[BATCH, D.NV * D.NV]

    # REFSAFE (defect 23): the friction rows use the HARDCODED default
    # timeconst, and MuJoCo clamps that to 2*timestep like any other. Read once
    # here; both targets take it as a scalar.
    var ts_v = m.meta.data[MODEL_META_IDX_TIMESTEP]

    comptime if target == "cpu":
        var dm = d.dims
        var rl_NV = rl2(BATCH, dm.get_nv())
        var rl_JOINT = rl2(dm.get_njoint(), MODEL_JOINT_SIZE)
        var rl_DW = rl1(dm.get_nv())
        var rl_M = rl2(BATCH, dm.get_nv() * dm.get_nv())
        var qvel_v = d.qvel.tt_dyn["cpu", DYN2](rl_NV)
        var joints_v = m.joints.tt_dyn["cpu", DYN2](rl_JOINT)
        var dw_v = m.dof_invweight0.tt_dyn["cpu", DYN1](rl_DW)
        var mi_v = scratch.m_inv.tt_dyn["cpu", DYN2](rl_M)
        var qc_v = scratch.qacc_constrained.tt_dyn["cpu", DYN2](rl_NV)
        for e in range(BATCH):
            _friction_env[DTYPE, NUM_ITERATIONS](
                e, dm, qvel_v, joints_v, ts_v, dw_v, mi_v, qc_v
            )
    else:
        var c = ctx.value()
        comptime BLOCKS = (BATCH + FRIC_TPB - 1) // FRIC_TPB
        c.enqueue_function[
            _friction_fields_kernel[
                DTYPE, D.NQ, D.NV, D.NJOINT, BATCH, NUM_ITERATIONS
            ]
        ](
            d.qvel.tt["gpu", BATCH, D.NV](),
            m.joints.tt["gpu", D.NJOINT, MODEL_JOINT_SIZE](),
            m.dof_invweight0.tt["gpu", D.NV](),
            scratch.m_inv.tt["gpu", BATCH, D.NV * D.NV](),
            scratch.qacc_constrained.tt["gpu", BATCH, D.NV](),
            ts_v,
            grid_dim=(BLOCKS,),
            block_dim=(FRIC_TPB,),
        )
