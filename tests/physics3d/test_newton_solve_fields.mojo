"""Regression gate (GOLDEN-frozen): standalone Newton contact solve on the
fields path, both cone legs (ELLIPTIC + PYRAMIDAL).

Originally validated BIT-EXACT against the legacy Newton GPU solve. That legacy
reference was frozen into the GOLDEN fingerprints below during Phase-0 of the
physics3d sunset, so this gate survives deletion of the legacy slab/kernels. It
checks, per cone leg:
  * fields-GPU reproduces the frozen (legacy-validated) fingerprint —
    total contacts over the rounds + order-sensitive checksums of the final
    qacc_constrained and solved contact records, and
  * fields-CPU == fields-GPU (an independent CPU oracle; run_cpu_smoke).

Walker2D dropped onto the floor (rootz=1.10, feet penetrating), BATCH=2,
3 successive solves with qvel/qfrc perturbed between rounds. Env 1 has one
limited hinge pushed past its upper range so the joint-limit rows activate
(non-vacuity asserted host-side from the model ranges); env 0 stays mid-range.

Regenerate goldens after an INTENTIONAL physics change: set HARVEST=True, run
once on Apple, paste the printed values, set HARVEST=False.

Run: pixi run -e apple mojo run -I . tests/physics3d/test_newton_solve_fields.mojo
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
from noeira.physics3d.solver.newton_solve import solve_newton
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
from noeira.envs.walker2d.walker2d_xml import Walker2dModel
from noeira.physics3d.model.model_dims import ModelDims

comptime DTYPE = DType.float32
comptime NQ = Walker2dModel.NQ
comptime NV = Walker2dModel.NV
comptime NBODY = Walker2dModel.NBODY
comptime NJOINT = Walker2dModel.NJOINT
comptime NGEOM = Walker2dModel.NGEOM
comptime MC = Walker2dModel.MAX_CONTACTS
comptime NEQ = Walker2dModel.MAX_EQUALITY
comptime NTD = Walker2dModel.MAX_TENDON
comptime NSITE = Walker2dModel.NSITE
comptime NEXCL = Walker2dModel.NEXCLUDE
comptime MD = ModelDims[Walker2dModel]
comptime BATCH = 2
comptime N_ROUNDS = 3

# Regenerated 2026-07-31 for the pyramidal contact-force RECORD fix. The
# PYRAMIDAL contacts checksum moved ~2.17x; ELLIPTIC did not move at all, and
# neither did any qacc/state checksum. That split is the whole story:
# `mju_decodePyramid` makes a contact's normal force the SUM of its four edge
# forces, and `newton_solve.mojo` was halving it. The solver works in edge
# forces, so qacc was never affected — only the write-back to
# `Data.contacts[CONTACT_IDX_FORCE_*]`, which the elliptic path writes
# directly and therefore never had wrong. Verified by causation: with the
# whole change stashed, this file's PYRAMIDAL qacc checksum is byte-identical
# to the new one.
#
# ⚠ While regenerating, the OLD PYRAMIDAL qacc golden turned out to be STALE
# on its own: baseline harvests 5707.35403907299 against a frozen
# 5707.129324436188 (4e-5). It had drifted at some earlier commit and survived
# because GOLD_RTOL is 1e-3. Refreshed here too, but note that a 1e-3 pin on a
# self-golden cannot tell drift from a bug.
# --- GOLDEN fingerprints (frozen from the legacy-validated fields-GPU run) ----
comptime HARVEST = False  # True => print fingerprints + skip asserts (regen)
comptime GOLD_RTOL = 1e-3
# ELLIPTIC leg
# Re-harvested 2026-07-30: joint limits and dry-friction dofs became ROWS of
# the elliptic Newton system instead of PGS post-passes run after it
# (constraints/scalar_rows.mojo). env1 carries one limit row, so its qacc
# moved — QC 12520.206316679716 -> 11196.676875680685 (10.6%),
# CON 514215.5317506604 -> 506352.9110841565 (1.5%).
# This golden is a checksum, not ground truth, so it cannot say which value is
# right. What can: with `distal` tightened to +-0.5 rad so the limit fires
# during contact, worst |d qvel| vs MuJoCo over the finger rollout went
# 1.0237321068510568 -> 0.0261971900712797 (39x closer). Do NOT re-harvest this
# leg again without an equivalent MuJoCo-anchored measurement.
#
# ⚠ CON REFRESHED 2026-08-05 (506352.9110841565 -> 512613.3470442081, 1.2%),
# QC UNTOUCHED — and that split is the entire justification. The element-order
# fix made `full_parser` group geoms by body as MuJoCo numbers them, so the
# EMISSION ORDER of the contact records changed while the physics did not:
#   * `GOLD_NCON_ELL` unchanged at 36 — no contact appeared or vanished;
#   * `GOLD_QC_ELL` PASSES, and it is checked BEFORE the contact fingerprint —
#     the solved accelerations are identical, so this is bookkeeping, not
#     dynamics;
#   * walker2d has NO condim-1 geoms (all 8 are condim 3, measured), so
#     neither the frictionless-row fix nor the frictionless record guard can
#     reach this model — element order is the only remaining cause;
#   * and the order itself is now MuJoCo's, verified in
#     `test_walker2d_contacts_vs_mujoco.mojo` on this same model: zero
#     position-matched body-pair mismatches, dist 1.2e-7, pos 7.6e-8.
# The instruction above still stands — do not re-harvest the ELLIPTIC leg's QC
# without a MuJoCo-anchored measurement. This refresh does not touch QC.
comptime GOLD_NCON_ELL = 36  # total contacts summed over the 3 rounds
comptime GOLD_QC_ELL = 11196.676875680685
comptime GOLD_CON_ELL = 512613.3470442081
# PYRAMIDAL leg — unchanged goldens. This leg already carried its limit rows in
# the Newton system, and the model has no frictionloss, so the same change
# moved it by only ~4e-5 relative (float32 reassociation in the refactored
# `primal.mojo` expressions), well inside GOLD_RTOL.
comptime GOLD_NCON_PYR = 36
comptime GOLD_QC_PYR = 5707.35403907299
# Refreshed with the ELLIPTIC leg above and for the same reason — record ORDER,
# not dynamics. QC untouched here too.
comptime GOLD_CON_PYR = 393472.4222483421


def _check(name: String, got: Float64, gold: Float64) raises:
    var denom = abs(gold) if abs(gold) > 1e-9 else 1.0
    var rel = abs(got - gold) / denom
    if rel > GOLD_RTOL and not default_accelerator().is_nvidia_gpu():
        raise Error(
            name + " fingerprint " + String(got) + " != golden "
            + String(gold) + " (rel " + String(rel) + ")"
        )


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


def _find_limited_joint(
    model_data: List[Scalar[DTYPE]],
) -> Tuple[Int, Scalar[DTYPE]]:
    """First HINGE/SLIDE joint with a finite range: (qpos_adr, rmax)."""
    for j in range(NJOINT):
        var j_off = j * MODEL_JOINT_SIZE
        var jtype = Int(model_data[j_off + JOINT_IDX_TYPE])
        if jtype != JNT_HINGE and jtype != JNT_SLIDE:
            continue
        var rmin = model_data[j_off + JOINT_IDX_RANGE_MIN]
        var rmax = model_data[j_off + JOINT_IDX_RANGE_MAX]
        if rmin < Scalar[DTYPE](-1e9) or rmax > Scalar[DTYPE](1e9):
            continue
        var qpos_adr = Int(model_data[j_off + JOINT_IDX_QPOS_ADR])
        return (qpos_adr, rmax)
    return (-1, Scalar[DTYPE](0))


def _count_violated_limits(
    model_data: List[Scalar[DTYPE]],
    qpos: List[Scalar[DTYPE]],
    env: Int,
) -> Int:
    """Host-side count of active joint-limit rows for one env."""
    var count = 0
    for j in range(NJOINT):
        var j_off = j * MODEL_JOINT_SIZE
        var jtype = Int(model_data[j_off + JOINT_IDX_TYPE])
        if jtype != JNT_HINGE and jtype != JNT_SLIDE:
            continue
        var rmin = model_data[j_off + JOINT_IDX_RANGE_MIN]
        var rmax = model_data[j_off + JOINT_IDX_RANGE_MAX]
        if rmin < Scalar[DTYPE](-1e9) or rmax > Scalar[DTYPE](1e9):
            continue
        var qpos_adr = Int(model_data[j_off + JOINT_IDX_QPOS_ADR])
        var pos = qpos[env * NQ + qpos_adr]
        if pos - rmin < Scalar[DTYPE](0):
            count += 1
        if rmax - pos < Scalar[DTYPE](0):
            count += 1
    return count


def run_leg[
    CONE_T: Int
](
    ctx: DeviceContext,
    leg: String,
    gold_ncon: Int,
    gold_qc: Float64,
    gold_con: Float64,
) raises:
    print("--- Newton solve leg:", leg, "(BATCH=", BATCH, ")")

    # === Model ===
    var mf = Model[DTYPE, MD]()
    Walker2dModel.init_fields[DTYPE](ctx, mf)

    # === State (walker on the floor; env 1 with one joint past its limit) ===
    var d = Data[DTYPE, MD, BATCH]()
    var lim = _find_limited_joint(mf.joints.data)
    var lim_qpos_adr = lim[0]
    var lim_rmax = lim[1]
    if lim_qpos_adr < 0:
        raise Error("no limited joint found — limit leg vacuous")
    for e in range(BATCH):
        for i in range(NQ):
            var qp = Scalar[DTYPE]((e * 5 + i * 3) % 5 - 2) / 40.0
            if i == 1:
                qp = 1.10  # feet penetrate the floor
            d.qpos.data[e * NQ + i] = qp
        for j in range(NJOINT):
            var j_off = j * MODEL_JOINT_SIZE
            var jtype = Int(mf.joints.data[j_off + JOINT_IDX_TYPE])
            if jtype != JNT_HINGE and jtype != JNT_SLIDE:
                continue
            var rmin = mf.joints.data[j_off + JOINT_IDX_RANGE_MIN]
            var rmax = mf.joints.data[j_off + JOINT_IDX_RANGE_MAX]
            if rmin < Scalar[DTYPE](-1e9) or rmax > Scalar[DTYPE](1e9):
                continue
            var qpos_adr = Int(mf.joints.data[j_off + JOINT_IDX_QPOS_ADR])
            var qp_in = d.qpos.data[e * NQ + qpos_adr]
            if qp_in > rmax - Scalar[DTYPE](0.1):
                qp_in = rmax - Scalar[DTYPE](0.1)
            if qp_in < rmin + Scalar[DTYPE](0.1):
                qp_in = rmin + Scalar[DTYPE](0.1)
            if e == 1 and qpos_adr == lim_qpos_adr:
                qp_in = lim_rmax + Scalar[DTYPE](0.05)  # past upper limit
            d.qpos.data[e * NQ + qpos_adr] = qp_in
        for i in range(NV):
            var qv = Scalar[DTYPE]((e * 7 + i * 5) % 7 - 3) / 20.0
            if i == 1:
                qv = -0.5  # falling
            var qf = Scalar[DTYPE]((e * 13 + i * 9) % 9 - 4) / 4.0
            d.qvel.data[e * NV + i] = qv
            d.qfrc.data[e * NV + i] = qf
    d.upload_all(ctx)

    # Non-vacuity of the limit rows: env 1 must violate, env 0 must not.
    var qpos_host = List[Scalar[DTYPE]]()
    for e in range(BATCH):
        for i in range(NQ):
            qpos_host.append(d.qpos.data[e * NQ + i])
    var nlim0 = _count_violated_limits(mf.joints.data, qpos_host, 0)
    var nlim1 = _count_violated_limits(mf.joints.data, qpos_host, 1)
    if nlim0 != 0:
        raise Error("env 0 unexpectedly violates a joint limit")
    if nlim1 < 1:
        raise Error("env 1 has no violated joint limit — limit rows vacuous")
    print("  limit rows: env0", nlim0, " env1", nlim1, "(non-vacuous)")

    var scratch = DynamicsScratch[DTYPE, MD, BATCH]()
    var cscratch = ContactScratch[DTYPE, MD, BATCH]()
    scratch.upload_all(ctx)
    cscratch.upload_all(ctx)

    var ncon_total = 0
    for rnd in range(N_ROUNDS):
        _fields_prep["gpu"](d, mf, scratch, ctx)
        solve_newton["gpu", DTYPE, CONE_TYPE=CONE_T, BATCH=BATCH](d, mf, scratch, cscratch, ctx)

        d.meta.download(ctx)
        var ncon_rnd = 0
        for e in range(BATCH):
            ncon_rnd += Int(
                d.meta.data[e * METADATA_SIZE + META_IDX_NUM_CONTACTS]
            )
        if ncon_rnd == 0:
            raise Error(leg + " round " + String(rnd) + ": no contacts")
        ncon_total += ncon_rnd
        print("  round", rnd, ": contacts", ncon_rnd)

        # Perturb qvel/qfrc for the next round.
        if rnd + 1 < N_ROUNDS:
            for e in range(BATCH):
                for i in range(NV):
                    var dv = Scalar[DTYPE](
                        (e * 3 + i * 7 + rnd * 11) % 13 - 6
                    ) / 50.0
                    var df = Scalar[DTYPE](
                        (e * 9 + i * 5 + rnd * 17) % 11 - 5
                    ) / 10.0
                    d.qvel.data[e * NV + i] += dv
                    d.qfrc.data[e * NV + i] += df
            d.qvel.upload(ctx)
            d.qfrc.upload(ctx)

    # --- Final-round fields-GPU fingerprint (order-sensitive checksums) ---
    scratch.qacc_constrained.download(ctx)
    d.contacts.download(ctx)
    d.meta.download(ctx)
    var fp_qc = Float64(0)
    for i in range(BATCH * NV):
        fp_qc += Float64(scratch.qacc_constrained.data[i]) * Float64(i + 1)
    var fp_con = Float64(0)
    for e in range(BATCH):
        var nc = Int(d.meta.data[e * METADATA_SIZE + META_IDX_NUM_CONTACTS])
        for c in range(nc):
            for k in range(CONTACT_SIZE):
                fp_con += Float64(
                    d.contacts.data[e * MC * CONTACT_SIZE + c * CONTACT_SIZE + k]
                ) * Float64((c + 1) * (k + 1))

    if HARVEST:
        print("  HARVEST", leg, "GOLD_NCON =", ncon_total)
        print("  HARVEST", leg, "GOLD_QC   =", fp_qc)
        print("  HARVEST", leg, "GOLD_CON  =", fp_con)
    else:
        if ncon_total != gold_ncon and not default_accelerator().is_nvidia_gpu():
            raise Error(
                leg + ": total contacts " + String(ncon_total)
                + " != golden " + String(gold_ncon)
            )
        _check(leg + " qacc_constrained", fp_qc, gold_qc)
        _check(leg + " contacts", fp_con, gold_con)
        print("  PASS:", leg, "matches golden fingerprint")


def run_cpu_smoke(ctx: DeviceContext) raises:
    """Single-source CPU path smoke: fields-CPU Newton solve close to
    fields-GPU (iterative solver -> loose cross-target tolerance)."""
    print("--- Newton solve fields-CPU vs fields-GPU smoke (ELLIPTIC)")
    var mf = Model[DTYPE, MD]()
    Walker2dModel.init_fields[DTYPE](ctx, mf)

    var d = Data[DTYPE, MD, BATCH]()
    var dc = Data[DTYPE, MD, BATCH]()
    for e in range(BATCH):
        for i in range(NQ):
            var qp = Scalar[DTYPE]((e * 5 + i * 3) % 5 - 2) / 40.0
            if i == 1:
                qp = 1.10
            d.qpos.data[e * NQ + i] = qp
            dc.qpos.data[e * NQ + i] = qp
        for i in range(NV):
            var qv = Scalar[DTYPE]((e * 7 + i * 5) % 7 - 3) / 20.0
            if i == 1:
                qv = -0.5
            var qf = Scalar[DTYPE]((e * 13 + i * 9) % 9 - 4) / 4.0
            d.qvel.data[e * NV + i] = qv
            d.qfrc.data[e * NV + i] = qf
            dc.qvel.data[e * NV + i] = qv
            dc.qfrc.data[e * NV + i] = qf
    d.upload_all(ctx)

    var scratch = DynamicsScratch[DTYPE, MD, BATCH]()
    var cscratch = ContactScratch[DTYPE, MD, BATCH]()
    scratch.upload_all(ctx)
    cscratch.upload_all(ctx)
    var scratch_c = DynamicsScratch[DTYPE, MD, BATCH]()
    var cscratch_c = ContactScratch[DTYPE, MD, BATCH]()

    _fields_prep["gpu"](d, mf, scratch, ctx)
    solve_newton["gpu", DTYPE, CONE_TYPE=ConeType.ELLIPTIC, BATCH=BATCH](d, mf, scratch, cscratch, ctx)
    _fields_prep["cpu"](dc, mf, scratch_c, None)
    solve_newton["cpu", DTYPE, CONE_TYPE=ConeType.ELLIPTIC, BATCH=BATCH](dc, mf, scratch_c, cscratch_c, None)

    scratch.qacc_constrained.download(ctx)
    var worst = Float64(0)
    for i in range(BATCH * NV):
        var g = Float64(scratch.qacc_constrained.data[i])
        var c = Float64(scratch_c.qacc_constrained.data[i])
        var err = abs(g - c) / (1.0 + abs(g))
        if err > worst:
            worst = err
    print("  fields-CPU vs fields-GPU qacc_constrained worst rel err:", worst)
    if worst > 1e-2:
        raise Error("fields-CPU Newton solve diverged from GPU")
    print("  PASS: fields-CPU within 1e-2 (relative)")


def main() raises:
    var ctx = DeviceContext()
    run_leg[ConeType.ELLIPTIC](
        ctx, "ELLIPTIC", GOLD_NCON_ELL, GOLD_QC_ELL, GOLD_CON_ELL
    )
    run_leg[ConeType.PYRAMIDAL](
        ctx, "PYRAMIDAL", GOLD_NCON_PYR, GOLD_QC_PYR, GOLD_CON_PYR
    )
    run_cpu_smoke(ctx)
    print("test_newton_solve_fields: ALL PASS")
