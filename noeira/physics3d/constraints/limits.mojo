"""Joint-limit constraints over per-field tensors (migration P4 opener,
single-source). Per-field port of `detect_and_solve_limits_gpu`
(constraints/constraint_builder_gpu.mojo:809) — arithmetic verbatim:
detect HINGE/SLIDE range violations, MuJoCo impedance (solref/solimp from
the meta record), acceleration-level PGS updating `scratch.qacc_constrained`
with `scratch.m_inv` columns.

Operands (7): qpos, qvel (data) + joints, meta, dof_invweight0 (model) +
m_inv, qacc_constrained (scratch). The legacy `dt` argument is dropped —
it is unused in the legacy body. This is the first consumer of the
constraint seam (writes qacc_constrained between the unconstrained solve
and the finalize); contacts/equality/tendons follow at P4."""

from std.math import abs, pow
from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceContext
from layout import TileTensor, TensorLayout
from noeira.core.tiles import RowMajor, Tile

from ..joint_types import JNT_HINGE, JNT_SLIDE
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
from ..gpu.constants import (
    MODEL_META_IDX_TIMESTEP,
    MODEL_JOINT_SIZE,
    MODEL_META_SIZE,
    JOINT_IDX_TYPE,
    JOINT_IDX_DOF_ADR,
    JOINT_IDX_QPOS_ADR,
    JOINT_IDX_RANGE_MIN,
    JOINT_IDX_RANGE_MAX,
    JOINT_IDX_MARGIN,
    # ⚠ PER-JOINT limit solver params (defect 22). The `MODEL_META_IDX_SOL*
    # _LIMIT_*` slots are deliberately NOT imported any more: `fields_build`
    # fills them from JOINT 0 for the whole model, which is the free root on
    # dog — unlimited, so the only joint that can never own a limit row.
    JOINT_IDX_SOLREF_LIMIT_0,
    JOINT_IDX_SOLREF_LIMIT_1,
    JOINT_IDX_SOLIMP_LIMIT_0,
    JOINT_IDX_SOLIMP_LIMIT_1,
    JOINT_IDX_SOLIMP_LIMIT_2,
    JOINT_IDX_SOLIMP_LIMIT_3,
    JOINT_IDX_SOLIMP_LIMIT_4,
)

from .constraint_data import solref_spring_damper

comptime LIM_TPB: Int = 64


@always_inline
def _max_one[N: Int]() -> Int:
    return N if N > 0 else 1


@always_inline
def _limits_env[
    DTYPE: DType,
    NUM_ITERATIONS: Int,
    D: DimsLike,
    L_QPOS: TensorLayout,
    L_QVEL: TensorLayout,
    L_JOINTS: TensorLayout,
    L_META: TensorLayout,
    L_DOF_INVWEIGHT0: TensorLayout,
    L_M_INV: TensorLayout,
](
    env: Int,
    dims: D,
    qpos: TileTensor[DTYPE, L_QPOS, MutAnyOrigin],
    qvel: TileTensor[DTYPE, L_QVEL, MutAnyOrigin],
    joints: TileTensor[DTYPE, L_JOINTS, MutAnyOrigin],
    meta: TileTensor[DTYPE, L_META, MutAnyOrigin],
    dof_invweight0: TileTensor[DTYPE, L_DOF_INVWEIGHT0, MutAnyOrigin],
    m_inv: TileTensor[DTYPE, L_M_INV, MutAnyOrigin],
    qacc_constrained: TileTensor[DTYPE, L_QVEL, MutAnyOrigin],
):
    """Detect + PGS-solve active joint limits for one env (verbatim from
    detect_and_solve_limits_gpu)."""
    var nv = dims.get_nv()
    var njoint = dims.get_njoint()
    comptime LIM_CAP = 2 * cap[D.NJOINT]()

    var limit_dof = Scratch[Int, LIM_CAP](2 * njoint, uninitialized=0)
    # ⚠ THE OWNING JOINT, because solref/solimp are PER-JOINT (defect 22).
    # These used to come from `meta`, which `fields_build` filled from JOINT 0
    # for the whole model on the assumption of "uniform joint solimp across all
    # current models". False on dog: 73 of its 74 joints are limited and all use
    # solreflimit [0.01 1], while joint 0 is the FREE ROOT — unlimited, so the
    # one joint whose parameters were broadcast is the only one that can never
    # form a limit row. It carried the model defaults [0.02 1], making every
    # limit 3.68x too soft (K 2770.08 where MuJoCo's efc_KBIP reads 10203.04).
    var limit_jnt = Scratch[Int, LIM_CAP](2 * njoint, uninitialized=0)
    var limit_sign = Scratch[Scalar[DTYPE], LIM_CAP](2 * njoint, uninitialized=0)
    var limit_dist_arr = Scratch[Scalar[DTYPE], LIM_CAP](
        2 * njoint, uninitialized=0
    )
    var K_limit = Scratch[Scalar[DTYPE], LIM_CAP](2 * njoint, uninitialized=0)
    var lambda_limit = Scratch[Scalar[DTYPE], LIM_CAP](
        2 * njoint, uninitialized=0
    )
    for i in range(2 * njoint):
        limit_dof[i] = 0
        limit_jnt[i] = 0
        limit_sign[i] = Scalar[DTYPE](0)
        limit_dist_arr[i] = Scalar[DTYPE](0)
        K_limit[i] = Scalar[DTYPE](1)
        lambda_limit[i] = Scalar[DTYPE](0)

    var num_limits = 0
    for j in range(njoint):
        var jtype = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_TYPE]))
        if jtype != JNT_HINGE and jtype != JNT_SLIDE:
            continue
        var dof = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_DOF_ADR]))
        var qpos_adr = Int(
            rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_QPOS_ADR])
        )
        var rmin = rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_RANGE_MIN])
        var rmax = rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_RANGE_MAX])
        if rmin < Scalar[DTYPE](-1e9) or rmax > Scalar[DTYPE](1e9):
            continue
        # `jnt_margin` (AUD-03): the row is built when `dist < margin` and
        # everything downstream sees `dist - margin` (MuJoCo's `pos - margin`,
        # engine_core_constraint.c:1394-1425, :2109, :3255).
        var jmargin = rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_MARGIN])
        var pos = rebind[Scalar[DTYPE]](qpos[env, qpos_adr])
        var dist_lo = pos - rmin - jmargin
        if dist_lo < Scalar[DTYPE](0) and num_limits < 2 * njoint:
            limit_dof[num_limits] = dof
            limit_jnt[num_limits] = j
            limit_sign[num_limits] = Scalar[DTYPE](1)
            limit_dist_arr[num_limits] = dist_lo
            K_limit[num_limits] = rebind[Scalar[DTYPE]](
                m_inv[env, dof * nv + dof]
            )
            if K_limit[num_limits] < Scalar[DTYPE](1e-10):
                K_limit[num_limits] = Scalar[DTYPE](1e-10)
            num_limits += 1
        var dist_hi = rmax - pos - jmargin
        if dist_hi < Scalar[DTYPE](0) and num_limits < 2 * njoint:
            limit_dof[num_limits] = dof
            limit_jnt[num_limits] = j
            limit_sign[num_limits] = Scalar[DTYPE](-1)
            limit_dist_arr[num_limits] = dist_hi
            K_limit[num_limits] = rebind[Scalar[DTYPE]](
                m_inv[env, dof * nv + dof]
            )
            if K_limit[num_limits] < Scalar[DTYPE](1e-10):
                K_limit[num_limits] = Scalar[DTYPE](1e-10)
            num_limits += 1

    if num_limits == 0:
        return

    comptime MJ_MINIMP = Scalar[DTYPE](0.0001)
    comptime MJ_MAXIMP = Scalar[DTYPE](0.9999)

    var lim_bias = Scratch[Scalar[DTYPE], LIM_CAP](2 * njoint, uninitialized=0)
    var lim_inv_K = Scratch[Scalar[DTYPE], LIM_CAP](2 * njoint, uninitialized=0)
    comptime MINVJ_LIM_CAP = 2 * cap[D.NJOINT]() * cap[D.NV]()
    var lim_MinvJ = Scratch[Scalar[DTYPE], MINVJ_LIM_CAP](
        2 * njoint * nv, uninitialized=0
    )
    for l in range(num_limits):
        # PER-ROW solref/solimp, read from the joint that OWNS this row
        # (defect 22). MuJoCo carries `jnt_solref[j]` / `jnt_solimp[j]` and
        # builds each limit row from its own joint's values; these were already
        # parsed and written into the joint record by `fields_build` and then
        # read by nothing.
        var lj = limit_jnt[l]
        var lr_tc = rebind[Scalar[DTYPE]](joints[lj, JOINT_IDX_SOLREF_LIMIT_0])
        var lr_dr = rebind[Scalar[DTYPE]](joints[lj, JOINT_IDX_SOLREF_LIMIT_1])
        var li_dmin = rebind[Scalar[DTYPE]](
            joints[lj, JOINT_IDX_SOLIMP_LIMIT_0]
        )
        var li_dmax = rebind[Scalar[DTYPE]](
            joints[lj, JOINT_IDX_SOLIMP_LIMIT_1]
        )
        var li_width = rebind[Scalar[DTYPE]](
            joints[lj, JOINT_IDX_SOLIMP_LIMIT_2]
        )
        var li_midpoint = rebind[Scalar[DTYPE]](
            joints[lj, JOINT_IDX_SOLIMP_LIMIT_3]
        )
        var li_power = rebind[Scalar[DTYPE]](
            joints[lj, JOINT_IDX_SOLIMP_LIMIT_4]
        )
        if li_width < Scalar[DTYPE](1e-6):
            li_width = Scalar[DTYPE](1e-6)
        # Clamp BOTH ends to [mjMINIMP, mjMAXIMP] as MuJoCo does before
        # interpolating (engine_core_constraint.c:1284-1287) — see the same fix
        # in contact_solve.mojo for why the dmin floor is the one that matters.
        if li_dmin < MJ_MINIMP:
            li_dmin = MJ_MINIMP
        elif li_dmin > MJ_MAXIMP:
            li_dmin = MJ_MAXIMP
        if li_dmax < MJ_MINIMP:
            li_dmax = MJ_MINIMP
        elif li_dmax > MJ_MAXIMP:
            li_dmax = MJ_MAXIMP
        if li_power < Scalar[DTYPE](1):
            li_power = Scalar[DTYPE](1)
        # solref -> (K, B), including MuJoCo's DIRECT form for a NEGATIVE
        # solref. See `constraints/constraint_data.solref_spring_damper` — the
        # formula lived in twelve copy-pasted sites until 2026-08-03.
        var (l_K_spring, l_B_damp) = solref_spring_damper[DTYPE](
            lr_tc, lr_dr, li_dmax,
            rebind[Scalar[DTYPE]](meta[MODEL_META_IDX_TIMESTEP]),
        )

        var penetration = -limit_dist_arr[l]
        if penetration < Scalar[DTYPE](0):
            penetration = Scalar[DTYPE](0)
        var imp_lim: Scalar[DTYPE]
        if li_dmin == li_dmax or li_width <= Scalar[DTYPE](0):
            imp_lim = Scalar[DTYPE](0.5) * (li_dmin + li_dmax)
        else:
            var x_lim = penetration / li_width
            var y_lim: Scalar[DTYPE]
            if x_lim <= Scalar[DTYPE](0):
                y_lim = Scalar[DTYPE](0)
            elif x_lim >= Scalar[DTYPE](1):
                y_lim = Scalar[DTYPE](1)
            elif li_power == Scalar[DTYPE](1):
                y_lim = x_lim
            elif x_lim <= li_midpoint:
                var a = Scalar[DTYPE](1) / pow(
                    li_midpoint, li_power - Scalar[DTYPE](1)
                )
                y_lim = a * pow(x_lim, li_power)
            else:
                var b = Scalar[DTYPE](1) / pow(
                    Scalar[DTYPE](1) - li_midpoint, li_power - Scalar[DTYPE](1)
                )
                y_lim = Scalar[DTYPE](1) - b * pow(
                    Scalar[DTYPE](1) - x_lim, li_power
                )
            imp_lim = li_dmin + y_lim * (li_dmax - li_dmin)
        if imp_lim < Scalar[DTYPE](1e-6):
            imp_lim = Scalar[DTYPE](1e-6)
        var v_limit = limit_sign[l] * rebind[Scalar[DTYPE]](
            qvel[env, limit_dof[l]]
        )
        lim_bias[l] = rebind[Scalar[DTYPE]](
            -l_K_spring * imp_lim * penetration + l_B_damp * v_limit
        )
        var diag_lim = rebind[Scalar[DTYPE]](dof_invweight0[limit_dof[l]])
        if diag_lim < Scalar[DTYPE](1e-10):
            diag_lim = K_limit[l]  # Fallback
        var R_lim = (Scalar[DTYPE](1.0) - imp_lim) / imp_lim * diag_lim
        lim_inv_K[l] = Scalar[DTYPE](1.0) / (K_limit[l] + R_lim)
        var ldof = limit_dof[l]
        var lsign = limit_sign[l]
        for i in range(nv):
            lim_MinvJ[l * nv + i] = (
                rebind[Scalar[DTYPE]](m_inv[env, i * nv + ldof]) * lsign
            )

    # PGS iterations (acceleration-level)
    for _ in range(NUM_ITERATIONS):
        var max_lim_delta: Scalar[DTYPE] = 0
        for l in range(num_limits):
            var a_limit = (
                limit_sign[l] * qacc_constrained[env, limit_dof[l]]
            )
            var R_lim = Scalar[DTYPE](1.0) / lim_inv_K[l] - K_limit[l]
            var residual_l = a_limit + lim_bias[l] + R_lim * lambda_limit[l]
            var delta_l = -residual_l * lim_inv_K[l]
            var old_lam = lambda_limit[l]
            lambda_limit[l] = lambda_limit[l] + rebind[Scalar[DTYPE]](delta_l)
            if lambda_limit[l] < Scalar[DTYPE](0):
                lambda_limit[l] = Scalar[DTYPE](0)
            var actual_l = lambda_limit[l] - old_lam
            var abs_l = abs(actual_l)
            if abs_l > max_lim_delta:
                max_lim_delta = abs_l
            for i in range(nv):
                qacc_constrained[env, i] += lim_MinvJ[l * nv + i] * actual_l
        if max_lim_delta < Scalar[DTYPE](1e-4):
            break


def _limits_fields_kernel[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NJOINT: Int,
    BATCH: Int,
    NUM_ITERATIONS: Int,
](
    qpos: Tile[DTYPE, BATCH, NQ],
    qvel: Tile[DTYPE, BATCH, NV],
    joints: Tile[DTYPE, NJOINT, MODEL_JOINT_SIZE],
    meta: Tile[DTYPE, MODEL_META_SIZE],
    dof_invweight0: Tile[DTYPE, NV],
    m_inv: Tile[DTYPE, BATCH, NV * NV],
    qacc_constrained: Tile[DTYPE, BATCH, NV],
):
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    _limits_env[DTYPE, NUM_ITERATIONS](
        env, Dims[nq=NQ, nv=NV, njoint=NJOINT](), qpos, qvel, joints, meta, dof_invweight0, m_inv, qacc_constrained
    )


def solve_limits[

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
    """Detect + solve joint limits into `scratch.qacc_constrained`, both
    targets, one body. NUM_ITERATIONS=50 matches the Newton solver's
    SOLVER_ITER_GPU."""
    comptime L_QPOS = RowMajor[BATCH, D.NQ]
    comptime L_NV = RowMajor[BATCH, D.NV]
    comptime L_JOINT = RowMajor[D.NJOINT, MODEL_JOINT_SIZE]
    comptime L_META = RowMajor[MODEL_META_SIZE]
    comptime L_DW = RowMajor[D.NV]
    comptime L_M = RowMajor[BATCH, D.NV * D.NV]

    comptime if target == "cpu":
        var dm = d.dims
        var rl_QPOS = rl2(BATCH, dm.get_nq())
        var rl_NV = rl2(BATCH, dm.get_nv())
        var rl_JOINT = rl2(dm.get_njoint(), MODEL_JOINT_SIZE)
        var rl_META = rl1(MODEL_META_SIZE)
        var rl_DW = rl1(dm.get_nv())
        var rl_M = rl2(BATCH, dm.get_nv() * dm.get_nv())
        var qpos_v = d.qpos.tt_dyn["cpu", DYN2](rl_QPOS)
        var qvel_v = d.qvel.tt_dyn["cpu", DYN2](rl_NV)
        var joints_v = m.joints.tt_dyn["cpu", DYN2](rl_JOINT)
        var meta_v = m.meta.tt_dyn["cpu", DYN1](rl_META)
        var dw_v = m.dof_invweight0.tt_dyn["cpu", DYN1](rl_DW)
        var mi_v = scratch.m_inv.tt_dyn["cpu", DYN2](rl_M)
        var qc_v = scratch.qacc_constrained.tt_dyn["cpu", DYN2](rl_NV)
        for e in range(BATCH):
            _limits_env[DTYPE, NUM_ITERATIONS](
                e, dm, qpos_v, qvel_v, joints_v, meta_v, dw_v, mi_v, qc_v
            )
    else:
        var c = ctx.value()
        comptime BLOCKS = (BATCH + LIM_TPB - 1) // LIM_TPB
        c.enqueue_function[
            _limits_fields_kernel[
                DTYPE, D.NQ, D.NV, D.NJOINT, BATCH, NUM_ITERATIONS
            ]
        ](
            d.qpos.tt["gpu", BATCH, D.NQ](),
            d.qvel.tt["gpu", BATCH, D.NV](),
            m.joints.tt["gpu", D.NJOINT, MODEL_JOINT_SIZE](),
            m.meta.tt["gpu", MODEL_META_SIZE](),
            m.dof_invweight0.tt["gpu", D.NV](),
            scratch.m_inv.tt["gpu", BATCH, D.NV * D.NV](),
            scratch.qacc_constrained.tt["gpu", BATCH, D.NV](),
            grid_dim=(BLOCKS,),
            block_dim=(LIM_TPB,),
        )
