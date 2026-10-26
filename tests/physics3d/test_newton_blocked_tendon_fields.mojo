"""Blocked (cooperative, one-env-per-block) Newton solver — TENDON-LIMIT rows.

⚠ THIS TEST EXISTS TO BE RUN ON NVIDIA. ⚠

`solve_newton_blocked` is the production path for NVIDIA + PYRAMIDAL. Until
2026-07-31 it built only contact and joint-limit edges: tendon-limit rows and
dry-friction rows were both absent, and every edge was classified one-sided.
Aligning it with the per-env pyramidal path (`_newton_solve_env`) added
`MAX_FRIC + MAX_TLIM` to `ME` plus the `kind/R/floss/state` row machinery.

WHAT IS ALREADY VERIFIED ELSEWHERE, ON APPLE:
  * the blocked kernel's cooperative GPU machinery with the new shared arrays —
    `test_newton_blocked_fields` runs the Metal GPU leg on walker2d and its
    golden fingerprint is unchanged. But walker2d has NO tendons and NO
    frictionloss, so it cannot exercise the new rows.
  * the tendon-row LOGIC — routing `euler.mojo` at
    `solve_newton_blocked["cpu"]` reproduced MuJoCo at 8.9e-16 on ball_in_cup.

WHAT ONLY NVIDIA CAN VERIFY, AND WHAT THIS FILE IS FOR:
  the COOPERATIVE GPU path actually CARRYING tendon rows — i.e. that the new
  `kind_e_sh`/`R_e_sh`/`floss_e_sh`/`state_e_sh` shared arrays are published
  and read correctly across threads and barriers, and that the larger `ME`
  still fits in the device's threadgroup memory.

  `ME` drives `Je_sh = ME * V_SIZE`, the dominant shared-memory term. If the
  block no longer fits, this shows up as a LAUNCH FAILURE, which is loud — not
  as a wrong answer. ball_in_cup is tiny (NV=4), so a failure here is a logic
  bug, not a capacity one; watch capacity on humanoid instead.

Part A  blocked-GPU vs blocked-CPU on the same prepared state. This is the
        cooperative-vs-serial comparison, so it isolates the GPU thread/barrier
        handling of the new arrays from the row arithmetic itself.
Part B  blocked-CPU vs the per-env `solve_newton` CPU path. Both build tendon
        rows, so they must agree; a mismatch means the two implementations have
        drifted, which is exactly what this alignment was meant to prevent.

Both parts assert NON-VACUITY first: the tendon limit must actually be violated
and contacts must exist, otherwise the test would pass on an empty row set.

Run with:
    pixi run -e nvidia mojo run -I . tests/physics3d/test_newton_blocked_tendon_fields.mojo
"""

from std.math import abs
from std.testing import TestSuite
from std.sys import default_accelerator
from max.gpu.host import DeviceContext
from noeira.core.tiles import RowMajor

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
from noeira.physics3d.joint_types import JNT_HINGE, JNT_SLIDE
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
from noeira.physics3d.solver.newton_solve import (
    solve_newton,
    solve_newton_blocked,
)
from noeira.physics3d.gpu.constants import (
    META_IDX_NUM_CONTACTS,
    METADATA_SIZE,
    CONTACT_SIZE,
    MODEL_BODY_SIZE,
    MODEL_JOINT_SIZE,
    JOINT_IDX_TYPE,
    JOINT_IDX_QPOS_ADR,
    JOINT_IDX_RANGE_MIN,
    JOINT_IDX_RANGE_MAX,
)
from noeira.envs.dm_control.ball_in_cup import DMBallInCupModel
from noeira.physics3d.model.model_dims import ModelDims

comptime DTYPE = DType.float32
comptime NQ = DMBallInCupModel.NQ
comptime NV = DMBallInCupModel.NV
comptime NBODY = DMBallInCupModel.NBODY
comptime NJOINT = DMBallInCupModel.NJOINT
comptime NGEOM = DMBallInCupModel.NGEOM
comptime MC = DMBallInCupModel.MAX_CONTACTS
comptime NEQ = DMBallInCupModel.MAX_EQUALITY
comptime NTD = DMBallInCupModel.MAX_TENDON
comptime NSITE = DMBallInCupModel.NSITE
comptime NEXCL = DMBallInCupModel.NEXCLUDE
comptime MD = ModelDims[DMBallInCupModel]
comptime BATCH = 2

# qpos = [cup_x, cup_z, ball_x, ball_z]; the tendon spans ball site -> cup site
# and is limited to 0.3 m. The ball body sits at z = .2, the cup at z = .6 with
# its site .108 below, so ball_z = -0.30 puts the string ~0.49 m long: well
# past the limit. ball_z low enough also drops the ball onto the ground plane,
# giving contacts in the SAME substep — the coupled regime.
comptime BALL_Z_TAUT: Float64 = -0.30
comptime TENDON_RANGE_MAX: Float64 = 0.3

# float32, because Metal rejects `double` in kernels — that is also why
# `test_newton_blocked_fields` is float32. Keeping this test runnable on Apple
# matters: it is the ONLY local coverage of the cooperative GPU path carrying
# tendon rows. 1e-4 is a real gate at single precision (the sibling test uses
# 1e-2 for the same comparison).
comptime REL_TOL: Float64 = 1e-4

def _fields_prep[
    target: StaticString
](
    mut d: Data[DTYPE, MD, BATCH],
    mut mf: Model[DTYPE, MD],
    mut scratch: DynamicsScratch[DTYPE, MD, BATCH],
    ctx: Optional[DeviceContext],
) raises:
    """Smooth-dynamics prep + detection, mirroring EulerIntegrator.step
    up to the constraint seam (order verbatim)."""
    forward_kinematics[target, DTYPE, BATCH=BATCH](d, mf, ctx)
    compute_body_velocities[target, DTYPE, BATCH=BATCH](d, mf, ctx)
    compute_subtree_com[target, DTYPE, BATCH=BATCH](d, mf, ctx)
    compute_cdof[target, DTYPE, BATCH=BATCH](d, mf, scratch, ctx)
    compute_mass_matrix[target, DTYPE, BATCH=BATCH](d, mf, scratch, ctx)

    comptime L_JOINT = RowMajor[NJOINT, MODEL_JOINT_SIZE]
    comptime L_M = RowMajor[BATCH, NV * NV]
    comptime L_NV = RowMajor[BATCH, NV]
    comptime L_QPOS = RowMajor[BATCH, NQ]

    comptime if target == "cpu":
        var joints_v = mf.joints.tt["cpu", NJOINT, MODEL_JOINT_SIZE]()
        var M_v = scratch.M.tt["cpu", BATCH, NV * NV]()
        for e in range(BATCH):
            _armature_env[DTYPE](e, AsStatic[MD](), joints_v, M_v)
        ldl_factor[target, DTYPE, BATCH=BATCH](mf, scratch, ctx)
        compute_m_inv[target, DTYPE, BATCH=BATCH](mf, scratch, ctx)
        compute_bias_forces_rne[target, DTYPE, BATCH=BATCH](d, mf, scratch, ctx)
        var qpos_v = d.qpos.tt["cpu", BATCH, NQ]()
        var qvel_v = d.qvel.tt["cpu", BATCH, NV]()
        var qfrc_v = d.qfrc.tt["cpu", BATCH, NV]()
        var bias_v = scratch.bias.tt["cpu", BATCH, NV]()
        var fnet_v = scratch.fnet.tt["cpu", BATCH, NV]()
        for e in range(BATCH):
            _fnet_passive_env[DTYPE](
                e, AsStatic[MD](), qpos_v, qvel_v, qfrc_v, joints_v, bias_v, fnet_v
            )
        ldl_solve[target, DTYPE, BATCH=BATCH](mf, scratch, ctx)
        var qacc_ws_v = scratch.qacc_ws.tt["cpu", BATCH, NV]()
        var qacc_v = d.qacc.tt["cpu", BATCH, NV]()
        var qacc_c_v = scratch.qacc_constrained.tt["cpu", BATCH, NV]()
        for e in range(BATCH):
            _qacc_writeback_env[DTYPE](
                e, AsStatic[MD](), qacc_ws_v, qacc_v, qacc_c_v
            )
    else:
        ctx.value().enqueue_function[
            _armature_kernel[DTYPE, NV, NJOINT, BATCH]
        ](
            mf.joints.tt["gpu", NJOINT, MODEL_JOINT_SIZE](),
            scratch.M.tt["gpu", BATCH, NV * NV](),
            grid_dim=(BATCH,),
            block_dim=(1,),
        )
        ldl_factor[target, DTYPE, BATCH=BATCH](mf, scratch, ctx)
        compute_m_inv[target, DTYPE, BATCH=BATCH](mf, scratch, ctx)
        compute_bias_forces_rne[target, DTYPE, BATCH=BATCH](d, mf, scratch, ctx)
        ctx.value().enqueue_function[
            _fnet_passive_kernel[DTYPE, NQ, NV, NJOINT, BATCH]
        ](
            d.qpos.tt["gpu", BATCH, NQ](),
            d.qvel.tt["gpu", BATCH, NV](),
            d.qfrc.tt["gpu", BATCH, NV](),
            mf.joints.tt["gpu", NJOINT, MODEL_JOINT_SIZE](),
            scratch.bias.tt["gpu", BATCH, NV](),
            scratch.fnet.tt["gpu", BATCH, NV](),
            grid_dim=(BATCH,),
            block_dim=(1,),
        )
        ldl_solve[target, DTYPE, BATCH=BATCH](mf, scratch, ctx)
        ctx.value().enqueue_function[
            _qacc_writeback_kernel[DTYPE, NV, BATCH]
        ](
            scratch.qacc_ws.tt["gpu", BATCH, NV](),
            d.qacc.tt["gpu", BATCH, NV](),
            scratch.qacc_constrained.tt["gpu", BATCH, NV](),
            grid_dim=(BATCH,),
            block_dim=(1,),
        )

    detect_contacts[target, DTYPE, BATCH=BATCH](d, mf, ctx)



def _ten_length(
    mf: Model[DTYPE, MD],
    d: Data[DTYPE, MD, BATCH],
    env: Int,
) -> Float64:
    """Straight-line ball-site to cup-site distance, from FK output."""
    var bx = Float64(d.site_xpos.data[env * NSITE * 3 + 2 * 3 + 0])
    var bz = Float64(d.site_xpos.data[env * NSITE * 3 + 2 * 3 + 2])
    var cx = Float64(d.site_xpos.data[env * NSITE * 3 + 0 * 3 + 0])
    var cz = Float64(d.site_xpos.data[env * NSITE * 3 + 0 * 3 + 2])
    var dx = cx - bx
    var dz = cz - bz
    return (dx * dx + dz * dz) ** 0.5


def _seed(mut d: Data[DTYPE, MD, BATCH]):
    for e in range(BATCH):
        d.qpos.data[e * NQ + 0] = Scalar[DTYPE](0.02 * Float64(e))
        d.qpos.data[e * NQ + 1] = Scalar[DTYPE](0.0)
        d.qpos.data[e * NQ + 2] = Scalar[DTYPE](0.01 * Float64(e))
        d.qpos.data[e * NQ + 3] = Scalar[DTYPE](BALL_Z_TAUT)
        for i in range(NV):
            d.qvel.data[e * NV + i] = Scalar[DTYPE](
                Float64((e * 7 + i * 5) % 7 - 3) / 20.0
            )
            d.qfrc.data[e * NV + i] = Scalar[DTYPE](
                Float64((e * 13 + i * 9) % 9 - 4) / 4.0
            )


def test_blocked_tendon_rows_gpu_vs_cpu() raises:
    """Part A — cooperative GPU vs serial CPU, same prepared state."""
    var ctx = DeviceContext()
    var mf = Model[DTYPE, MD]()
    DMBallInCupModel.init_fields[DTYPE](ctx, mf)

    var dg = Data[DTYPE, MD, BATCH]()
    var dc = Data[DTYPE, MD, BATCH]()
    _seed(dg)
    _seed(dc)
    dg.upload_all(ctx)

    var sg = DynamicsScratch[DTYPE, MD, BATCH]()
    var sc = DynamicsScratch[DTYPE, MD, BATCH]()
    var cg = ContactScratch[DTYPE, MD, BATCH]()
    var cc = ContactScratch[DTYPE, MD, BATCH]()
    sg.upload_all(ctx)
    cg.upload_all(ctx)

    _fields_prep["gpu"](dg, mf, sg, ctx)
    _fields_prep["cpu"](dc, mf, sc, None)

    # --- NON-VACUITY: the tendon limit must be violated, contacts must exist.
    for e in range(BATCH):
        var L = _ten_length(mf, dc, e)
        if L <= TENDON_RANGE_MAX:
            raise Error(
                "env " + String(e) + " tendon length " + String(L)
                + " is within range — the tendon-limit rows are VACUOUS and"
                " this test proves nothing. Lower BALL_Z_TAUT."
            )
    dg.meta.download(ctx)
    var ncon = 0
    for e in range(BATCH):
        ncon += Int(dc.meta.data[e * METADATA_SIZE + META_IDX_NUM_CONTACTS])
    print("  tendon taut in both envs; contacts (cpu prep):", ncon)
    if ncon == 0:
        raise Error(
            "no contacts — the COUPLED regime (tendon limit + contact on"
            " shared dofs) is untested, which is the regime this alignment"
            " is about"
        )

    solve_newton_blocked["gpu", DTYPE, CONE_TYPE=ConeType.PYRAMIDAL, BATCH=BATCH](dg, mf, sg, cg, ctx)
    solve_newton_blocked["cpu", DTYPE, CONE_TYPE=ConeType.PYRAMIDAL, BATCH=BATCH](dc, mf, sc, cc, None)

    sg.qacc_constrained.download(ctx)
    var worst = Float64(0)
    for k in range(BATCH * NV):
        var a = Float64(sg.qacc_constrained.data[k])
        var b = Float64(sc.qacc_constrained.data[k])
        var den = abs(b)
        if den < 1e-8:
            den = 1e-8
        var rel = abs(a - b) / den
        if rel > worst:
            worst = rel
    print("  blocked GPU vs blocked CPU worst rel err:", worst)
    if worst > REL_TOL:
        raise Error(
            "blocked GPU disagrees with blocked CPU on tendon rows — the"
            " cooperative publication of kind/R/floss/state across threads is"
            " wrong (rel " + String(worst) + ")"
        )
    print("  PASS: cooperative GPU carries the tendon rows")


def test_blocked_matches_per_env_solver() raises:
    """Part B — blocked CPU vs the per-env `solve_newton` CPU path."""
    var mf = Model[DTYPE, MD]()
    var ctx = DeviceContext()
    DMBallInCupModel.init_fields[DTYPE](ctx, mf)

    var db = Data[DTYPE, MD, BATCH]()
    var dp = Data[DTYPE, MD, BATCH]()
    _seed(db)
    _seed(dp)

    var sb = DynamicsScratch[DTYPE, MD, BATCH]()
    var sp = DynamicsScratch[DTYPE, MD, BATCH]()
    var cb = ContactScratch[DTYPE, MD, BATCH]()
    var cp = ContactScratch[DTYPE, MD, BATCH]()

    _fields_prep["cpu"](db, mf, sb, None)
    _fields_prep["cpu"](dp, mf, sp, None)

    solve_newton_blocked["cpu", DTYPE, CONE_TYPE=ConeType.PYRAMIDAL, BATCH=BATCH](db, mf, sb, cb, None)
    solve_newton["cpu", DTYPE, CONE_TYPE=ConeType.PYRAMIDAL, BATCH=BATCH](dp, mf, sp, cp, None)

    var worst = Float64(0)
    for k in range(BATCH * NV):
        var a = Float64(sb.qacc_constrained.data[k])
        var b = Float64(sp.qacc_constrained.data[k])
        var den = abs(b)
        if den < 1e-8:
            den = 1e-8
        var rel = abs(a - b) / den
        if rel > worst:
            worst = rel
    print("  blocked vs per-env worst rel err:", worst)
    if worst > REL_TOL:
        raise Error(
            "the blocked and per-env pyramidal solvers have DRIFTED on tendon"
            " rows (rel " + String(worst) + ") — they are supposed to build"
            " the same rows from the same builder"
        )
    print("  PASS: blocked and per-env solvers agree")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
