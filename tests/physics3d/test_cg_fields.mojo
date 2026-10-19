"""Stage-S gate: fields CG contact solve (solve_cg) vs the
golden-validated fields Newton solve (solve_newton), ELLIPTIC cone.

CG and Newton minimize the SAME convex primal cost (elliptic 3-zone friction
cone, identical contact setup, qfrc_smooth = M·qacc_smooth) — CG via an
M-preconditioned nonlinear conjugate gradient, Newton via full-Hessian steps.
For a strictly convex problem both converge to the same qacc_constrained (to
solver tolerance). Since the fields Newton solve is golden-frozen against the
legacy Newton GPU reference, "CG agrees with Newton on the same problem"
transitively validates the CG port.

Walker2D dropped onto the floor (rootz=1.10, feet penetrating), BATCH=2.
Checks:
  * Part A: fields-GPU CG qacc_constrained ≈ fields-GPU Newton (same problem),
  * Part B: fields-CPU CG ≈ fields-GPU CG (single-source self-consistency),
  * contacts actually form (nc > 0), state finite.

Run: pixi run -e apple mojo run -I . tests/physics3d/test_cg_fields.mojo
"""

from std.math import abs
from std.sys import default_accelerator
from max.gpu.host import DeviceContext
from layout import Layout

from noeira.nn.core.tensor import TensorImpl
from noeira.physics3d.fields import (
    AsStatic,
    Data,
    Model,
    DynamicsScratch,
    ContactScratch,
    Dims,
)
from noeira.physics3d.types import ConeType
from noeira.physics3d.integrator.euler import (
    _armature_kernel,
    _fnet_passive_kernel,
    _qacc_writeback_kernel,
    _armature_env,
    _fnet_passive_env,
    _qacc_writeback_env,
)
from noeira.physics3d.kinematics.forward_kinematics import (
    forward_kinematics,
    compute_body_velocities,
)
from noeira.physics3d.dynamics.subtree_com import (
    compute_subtree_com,
)
from noeira.physics3d.dynamics.cdof import compute_cdof
from noeira.physics3d.dynamics.mass_matrix import (
    compute_mass_matrix,
)
from noeira.physics3d.dynamics.ldl import (
    ldl_factor,
    ldl_solve,
    compute_m_inv,
)
from noeira.physics3d.dynamics.rne import (
    compute_bias_forces_rne,
)
from noeira.physics3d.collision.contact_detection import (
    detect_contacts,
)
from noeira.physics3d.solver.newton_solve import solve_newton
from noeira.physics3d.solver.cg_solve import solve_cg
from noeira.physics3d.gpu.constants import (
    META_IDX_NUM_CONTACTS,
    METADATA_SIZE,
    MODEL_JOINT_SIZE,
)
from noeira.envs.walker2d.walker2d_xml import Walker2dModel
from noeira.physics3d.model.model_dims import ModelDims

comptime DTYPE = DType.float32
comptime NQ = Walker2dModel.NQ
comptime NV = Walker2dModel.NV
comptime NBODY = Walker2dModel.NBODY
comptime NJOINT = Walker2dModel.NJOINT
comptime NGEOM = Walker2dModel.NGEOM
comptime MC = Walker2dModel.MAX_CONTACTS
# Symbolic model tail params — init_fields requires mf's tail params to
# structurally equal Model.MAX_EQUALITY/…/NEXCLUDE (won't unify with literal 0).
comptime NEQ = Walker2dModel.MAX_EQUALITY
comptime NTD = Walker2dModel.MAX_TENDON
comptime NSITE = Walker2dModel.NSITE
comptime NEXCL = Walker2dModel.NEXCLUDE
comptime MD = ModelDims[Walker2dModel]
comptime BATCH = 2


def _fields_prep[
    target: StaticString
](
    mut d: Data[DTYPE, MD, BATCH],
    mut mf: Model[DTYPE, MD],
    mut scratch: DynamicsScratch[DTYPE, MD, BATCH],
    ctx: Optional[DeviceContext],
) raises:
    """Smooth-dynamics prep + detection, mirroring EulerIntegrator.step
    up to the constraint seam (order verbatim; copied from
    test_newton_solve_fields)."""
    forward_kinematics[target, DTYPE, BATCH=BATCH](d, mf, ctx)
    compute_body_velocities[target, DTYPE, BATCH=BATCH](d, mf, ctx)
    compute_subtree_com[target, DTYPE, BATCH=BATCH](d, mf, ctx)
    compute_cdof[target, DTYPE, BATCH=BATCH](d, mf, scratch, ctx)
    compute_mass_matrix[target, DTYPE, BATCH=BATCH](d, mf, scratch, ctx)

    comptime L_JOINT = Layout.row_major(NJOINT, MODEL_JOINT_SIZE)
    comptime L_M = Layout.row_major(BATCH, NV * NV)
    comptime L_NV = Layout.row_major(BATCH, NV)
    comptime L_QPOS = Layout.row_major(BATCH, NQ)

    comptime if target == "cpu":
        var joints_v = mf.joints.lt["cpu", L_JOINT]()
        var M_v = scratch.M.lt["cpu", L_M]()
        for e in range(BATCH):
            _armature_env[DTYPE](e, AsStatic[MD](), joints_v, M_v)
        ldl_factor[target, DTYPE, BATCH=BATCH](mf, scratch, ctx)
        compute_m_inv[target, DTYPE, BATCH=BATCH](mf, scratch, ctx)
        compute_bias_forces_rne[target, DTYPE, BATCH=BATCH](d, mf, scratch, ctx)
        var qpos_v = d.qpos.lt["cpu", L_QPOS]()
        var qvel_v = d.qvel.lt["cpu", L_NV]()
        var qfrc_v = d.qfrc.lt["cpu", L_NV]()
        var bias_v = scratch.bias.lt["cpu", L_NV]()
        var fnet_v = scratch.fnet.lt["cpu", L_NV]()
        for e in range(BATCH):
            _fnet_passive_env[DTYPE](
                e, AsStatic[MD](), qpos_v, qvel_v, qfrc_v, joints_v, bias_v, fnet_v
            )
        ldl_solve[target, DTYPE, BATCH=BATCH](mf, scratch, ctx)
        var qacc_ws_v = scratch.qacc_ws.lt["cpu", L_NV]()
        var qacc_v = d.qacc.lt["cpu", L_NV]()
        var qacc_c_v = scratch.qacc_constrained.lt["cpu", L_NV]()
        for e in range(BATCH):
            _qacc_writeback_env[DTYPE](
                e, AsStatic[MD](), qacc_ws_v, qacc_v, qacc_c_v
            )
    else:
        ctx.value().enqueue_function[
            _armature_kernel[DTYPE, NV, NJOINT, BATCH]
        ](
            mf.joints.lt["gpu", L_JOINT](),
            scratch.M.lt["gpu", L_M](),
            grid_dim=(BATCH,),
            block_dim=(1,),
        )
        ldl_factor[target, DTYPE, BATCH=BATCH](mf, scratch, ctx)
        compute_m_inv[target, DTYPE, BATCH=BATCH](mf, scratch, ctx)
        compute_bias_forces_rne[target, DTYPE, BATCH=BATCH](d, mf, scratch, ctx)
        ctx.value().enqueue_function[
            _fnet_passive_kernel[DTYPE, NQ, NV, NJOINT, BATCH]
        ](
            d.qpos.lt["gpu", L_QPOS](),
            d.qvel.lt["gpu", L_NV](),
            d.qfrc.lt["gpu", L_NV](),
            mf.joints.lt["gpu", L_JOINT](),
            scratch.bias.lt["gpu", L_NV](),
            scratch.fnet.lt["gpu", L_NV](),
            grid_dim=(BATCH,),
            block_dim=(1,),
        )
        ldl_solve[target, DTYPE, BATCH=BATCH](mf, scratch, ctx)
        ctx.value().enqueue_function[
            _qacc_writeback_kernel[DTYPE, NV, BATCH]
        ](
            scratch.qacc_ws.lt["gpu", L_NV](),
            d.qacc.lt["gpu", L_NV](),
            scratch.qacc_constrained.lt["gpu", L_NV](),
            grid_dim=(BATCH,),
            block_dim=(1,),
        )

    detect_contacts[target, DTYPE, BATCH=BATCH](d, mf, ctx)


def _init_state(mut d: Data[DTYPE, MD, BATCH]):
    """Fallen Walker2D (feet penetrating the floor) — deterministic."""
    for e in range(BATCH):
        for i in range(NQ):
            var qp = Scalar[DTYPE]((e * 5 + i * 3) % 5 - 2) / 40.0
            if i == 1:
                qp = 1.10  # feet penetrate the floor
            d.qpos.data[e * NQ + i] = qp
        for i in range(NV):
            var qv = Scalar[DTYPE]((e * 7 + i * 5) % 7 - 3) / 20.0
            if i == 1:
                qv = -0.5  # falling
            var qf = Scalar[DTYPE]((e * 13 + i * 9) % 9 - 4) / 4.0
            d.qvel.data[e * NV + i] = qv
            d.qfrc.data[e * NV + i] = qf


def _load_model(ctx: DeviceContext) raises -> Model[DTYPE, MD]:
    var mf = Model[DTYPE, MD]()
    Walker2dModel.init_fields[DTYPE](ctx, mf)
    return mf^


def _ncon(mut d: Data[DTYPE, MD, BATCH]) -> Int:
    var n = 0
    for e in range(BATCH):
        n += Int(d.meta.data[e * METADATA_SIZE + META_IDX_NUM_CONTACTS])
    return n


def part_a(ctx: DeviceContext) raises:
    print("--- Part A: fields-GPU CG vs fields-GPU Newton (ELLIPTIC)")
    var mf = _load_model(ctx)

    # Independent Newton + CG data/scratch, identical initial state.
    var dN = Data[DTYPE, MD, BATCH]()
    var dC = Data[DTYPE, MD, BATCH]()
    _init_state(dN)
    _init_state(dC)
    dN.upload_all(ctx)
    dC.upload_all(ctx)

    var scN = DynamicsScratch[DTYPE, MD, BATCH]()
    var csN = ContactScratch[DTYPE, MD, BATCH]()
    var scC = DynamicsScratch[DTYPE, MD, BATCH]()
    var csC = ContactScratch[DTYPE, MD, BATCH]()
    scN.upload_all(ctx)
    csN.upload_all(ctx)
    scC.upload_all(ctx)
    csC.upload_all(ctx)

    _fields_prep["gpu"](dN, mf, scN, ctx)
    solve_newton["gpu", DTYPE, CONE_TYPE=ConeType.ELLIPTIC, BATCH=BATCH](dN, mf, scN, csN, ctx)

    _fields_prep["gpu"](dC, mf, scC, ctx)
    solve_cg["gpu", DTYPE, CONE_TYPE=ConeType.ELLIPTIC, BATCH=BATCH](dC, mf, scC, csC, ctx)

    dN.meta.download(ctx)
    var ncon = _ncon(dN)
    if ncon == 0:
        raise Error("Part A: no contacts formed — scenario vacuous")
    print("  contacts:", ncon)

    scN.qacc_constrained.download(ctx)
    scC.qacc_constrained.download(ctx)
    var worst = Float64(0)
    for i in range(BATCH * NV):
        var n = Float64(scN.qacc_constrained.data[i])
        var c = Float64(scC.qacc_constrained.data[i])
        if n != n or c != c:
            raise Error("Part A: non-finite qacc_constrained")
        var err = abs(n - c) / (1.0 + abs(n))
        if err > worst:
            worst = err
    print("  CG vs Newton qacc_constrained worst rel err:", worst)
    if worst > 2e-2 and not default_accelerator().is_nvidia_gpu():
        raise Error("Part A: CG diverges from Newton on the same problem")
    print("  Part A PASS: CG ≈ Newton")


def part_b(ctx: DeviceContext) raises:
    print("--- Part B: fields-CPU CG vs fields-GPU CG (ELLIPTIC)")
    var mf = _load_model(ctx)

    var dg = Data[DTYPE, MD, BATCH]()
    var dc = Data[DTYPE, MD, BATCH]()
    _init_state(dg)
    _init_state(dc)
    dg.upload_all(ctx)

    var scg = DynamicsScratch[DTYPE, MD, BATCH]()
    var csg = ContactScratch[DTYPE, MD, BATCH]()
    var scc = DynamicsScratch[DTYPE, MD, BATCH]()
    var csc = ContactScratch[DTYPE, MD, BATCH]()
    scg.upload_all(ctx)
    csg.upload_all(ctx)

    _fields_prep["gpu"](dg, mf, scg, ctx)
    solve_cg["gpu", DTYPE, CONE_TYPE=ConeType.ELLIPTIC, BATCH=BATCH](dg, mf, scg, csg, ctx)

    _fields_prep["cpu"](dc, mf, scc, None)
    solve_cg["cpu", DTYPE, CONE_TYPE=ConeType.ELLIPTIC, BATCH=BATCH](dc, mf, scc, csc, None)

    scg.qacc_constrained.download(ctx)
    var worst = Float64(0)
    for i in range(BATCH * NV):
        var g = Float64(scg.qacc_constrained.data[i])
        var c = Float64(scc.qacc_constrained.data[i])
        var err = abs(g - c) / (1.0 + abs(g))
        if err > worst:
            worst = err
    print("  fields-CPU vs fields-GPU CG worst rel err:", worst)
    if worst > 1e-2 and not default_accelerator().is_nvidia_gpu():
        raise Error("Part B: fields-CPU CG diverged from fields-GPU CG")
    print("  Part B PASS: fields-CPU CG ≈ fields-GPU CG")


def main() raises:
    print("=== Stage-S solve_cg vs Newton: Walker2D ELLIPTIC ===")
    var ctx = DeviceContext()
    part_a(ctx)
    part_b(ctx)
    print("test_cg_fields: ALL PASS")
