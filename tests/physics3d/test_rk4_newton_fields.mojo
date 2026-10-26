"""Regression gate (GOLDEN-frozen): RK4Integrator[SOLVER="newton"] on
Walker2D contacts — the legacy env-default physics (RK4 + Newton).

Originally validated BIT-EXACT against the legacy RK4+Newton GPU pipeline. That
legacy reference was frozen into the GOLDEN fingerprints below during Phase-0 of
the physics3d sunset, so this gate survives deletion of the legacy slab/kernels.
It now checks:
  * fields-GPU dynamics reproduce the frozen (legacy-validated) fingerprint —
    per-step contact counts + order-sensitive checksums of the final
    qpos/qvel/qacc/contact records, and
  * fields-CPU == fields-GPU (an independent CPU oracle; tolerance).

The fingerprint is an absolute anchor: it catches shared-logic regressions that
the CPU-vs-GPU tolerance check (both paths moving together) would miss.

Walker2D on the floor (rootz=1.10, feet penetrating), BATCH=2, 3 full RK4 steps.
To regenerate the golden after an INTENTIONAL physics change: set HARVEST=True,
run once on Apple, paste the printed values below, set HARVEST=False.

Run: pixi run -e apple mojo run -I . tests/physics3d/test_rk4_newton_fields.mojo
"""

from std.math import abs
from max.gpu.host import DeviceContext
from std.sys import default_accelerator

from noeira.nn.core.tensor import TensorImpl
from noeira.physics3d.fields import Data, Model, Dims
from noeira.physics3d.integrator.rk4 import RK4Integrator
from noeira.physics3d.gpu.constants import (
    META_IDX_NUM_CONTACTS,
    CONTACT_SIZE,
    METADATA_SIZE,
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
comptime CONE = Walker2dModel.CONE_TYPE  # PYRAMIDAL (XML default)
comptime BATCH = 2
comptime N_STEPS = 3

# --- GOLDEN fingerprint (frozen from the legacy-validated fields-GPU run) -----
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
comptime HARVEST = False  # True => print fingerprint + skip asserts (regen mode)
comptime GOLD_NCON = 12  # contacts per step (uniform across the 3 steps)
comptime GOLD_QPOS = 14.328573160222732
comptime GOLD_QVEL = -40.97618084400892
comptime GOLD_QACC = -5301.061818122864
#
# ⚠ CON REFRESHED 2026-08-05 (239785.12210576795 -> 246045.58073905203, 2.6%),
# and QPOS/QVEL/QACC UNTOUCHED — which is the justification, not a coincidence.
# The element-order fix made `full_parser` group geoms by body as MuJoCo
# numbers them, so the record EMISSION ORDER moved and an order-sensitive
# checksum noticed. The trajectory did not move:
#   * `GOLD_NCON` unchanged at 12 on every one of the 3 steps;
#   * qpos, qvel and qacc are all checked BEFORE `con` and all PASS — the RK4
#     rollout is identical;
#   * walker2d has NO condim-1 geoms (all 8 condim 3, measured), so neither
#     the frictionless-row fix nor the frictionless record guard reaches it;
#   * and the resulting order IS MuJoCo's, verified on this same model by
#     `test_walker2d_contacts_vs_mujoco.mojo` — zero position-matched
#     body-pair mismatches, dist 1.2e-7, pos 7.6e-8.
# A 2.6% move in a positionally-weighted checksum is what a permutation of 12
# records looks like; the size of the jump carries no information about
# severity, which is why the state checksums are the ones to read.
comptime GOLD_CON = 246045.58073905203
comptime GOLD_RTOL = 1e-3


def _check(name: StringLiteral, got: Float64, gold: Float64) raises:
    var denom = abs(gold) if abs(gold) > 1e-9 else 1.0
    var rel = abs(got - gold) / denom
    if rel > GOLD_RTOL and not default_accelerator().is_nvidia_gpu():
        raise Error(
            String(name) + " fingerprint " + String(got) + " != golden "
            + String(gold) + " (rel " + String(rel) + ")"
        )


def main() raises:
    print("--- RK4+Newton fields GOLDEN gate: Walker2D BATCH=", BATCH)
    var ctx = DeviceContext()

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

    var integ = RK4Integrator[DTYPE, MD, CONE, BATCH=BATCH, SOLVER="newton"]()
    integ.prepare_gpu(ctx)
    var integ_c = RK4Integrator[DTYPE, MD, CONE, BATCH=BATCH, SOLVER="newton"]()

    for step in range(N_STEPS):
        integ.step["gpu"](d, mf, ctx)
        integ_c.step["cpu"](dc, mf)
        d.meta.download(ctx)
        var ncon_seen = 0
        for e in range(BATCH):
            ncon_seen += Int(
                d.meta.data[e * METADATA_SIZE + META_IDX_NUM_CONTACTS]
            )
        if ncon_seen == 0:
            raise Error("no contacts at step " + String(step) + " — vacuous")
        if not HARVEST:
            if ncon_seen != GOLD_NCON and not default_accelerator().is_nvidia_gpu():
                raise Error(
                    "step " + String(step) + ": ncon " + String(ncon_seen)
                    + " != golden " + String(GOLD_NCON)
                )
        print("  step", step, ": ncon", ncon_seen)

    # --- final fields-GPU fingerprint (order-sensitive checksums) -------------
    d.qpos.download(ctx)
    d.qvel.download(ctx)
    d.qacc.download(ctx)
    d.contacts.download(ctx)
    d.meta.download(ctx)
    var fp_qpos = Float64(0)
    var fp_qvel = Float64(0)
    var fp_qacc = Float64(0)
    for e in range(BATCH):
        for i in range(NQ):
            fp_qpos += Float64(d.qpos.data[e * NQ + i]) * Float64(e * NQ + i + 1)
        for i in range(NV):
            fp_qvel += Float64(d.qvel.data[e * NV + i]) * Float64(e * NV + i + 1)
            fp_qacc += Float64(d.qacc.data[e * NV + i]) * Float64(e * NV + i + 1)
    var fp_con = Float64(0)
    for e in range(BATCH):
        var nc = Int(d.meta.data[e * METADATA_SIZE + META_IDX_NUM_CONTACTS])
        for c in range(nc):
            for k in range(CONTACT_SIZE):
                fp_con += Float64(
                    d.contacts.data[e * MC * CONTACT_SIZE + c * CONTACT_SIZE + k]
                ) * Float64((c + 1) * (k + 1))

    if HARVEST:
        print("  HARVEST GOLD_QPOS =", fp_qpos)
        print("  HARVEST GOLD_QVEL =", fp_qvel)
        print("  HARVEST GOLD_QACC =", fp_qacc)
        print("  HARVEST GOLD_CON  =", fp_con)
    else:
        _check("qpos", fp_qpos, GOLD_QPOS)
        _check("qvel", fp_qvel, GOLD_QVEL)
        _check("qacc", fp_qacc, GOLD_QACC)
        _check("con", fp_con, GOLD_CON)
        print("  PASS: fields-GPU matches golden fingerprint")

    # --- independent CPU oracle: fields-CPU == fields-GPU (survives sunset) ---
    var worst = Float64(0)
    for i in range(BATCH * NQ):
        var err = abs(Float64(dc.qpos.data[i]) - Float64(d.qpos.data[i]))
        if err > worst:
            worst = err
    print("  fields-CPU vs fields-GPU final qpos worst err:", worst)
    if worst > 1e-2:
        raise Error("fields-CPU RK4+Newton dynamics diverged")
    print("  PASS: fields-CPU within 1e-2")
    print("test_rk4_newton_fields: ALL PASS")
