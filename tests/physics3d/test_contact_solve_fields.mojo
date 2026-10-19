"""Regression gate (GOLDEN-frozen): full Euler step WITH CONTACTS on the fields
path, Walker2D on the floor.

Originally validated BIT-EXACT against the legacy Euler+PGS contact pipeline.
That legacy reference was frozen into the GOLDEN fingerprints below during
Phase-0 of the physics3d sunset, so this gate survives deletion of the legacy
slab/kernels. It checks:
  * fields-GPU reproduces the frozen (legacy-validated) fingerprint —
    per-step contact counts + order-sensitive checksums of the final
    qpos/qvel/qacc/contact records, and
  * fields-CPU == fields-GPU (independent CPU oracle; tolerance).

Walker2D dropped onto the floor (rootz=1.10, feet penetrating), BATCH=2,
3 consecutive full steps (detection -> serialized contact PGS -> limits ->
re-homes it). Regenerate goldens after an INTENTIONAL physics change:
HARVEST=True, run on Apple, paste, HARVEST=False.

Run: pixi run -e apple mojo run -I . tests/physics3d/test_contact_solve_fields.mojo
"""

from std.math import abs
from max.gpu.host import DeviceContext
from std.sys import default_accelerator

from noeira.nn.core.tensor import TensorImpl
from noeira.physics3d.fields import Data, Model, Dims
from noeira.physics3d.integrator.euler import EulerIntegrator
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
comptime CONE = Walker2dModel.CONE_TYPE
comptime BATCH = 2
comptime N_STEPS = 3

# --- GOLDEN fingerprints (frozen from the legacy-validated fields-GPU run) ----
comptime HARVEST = False  # True => print fingerprints + skip asserts (regen)
comptime GOLD_RTOL = 1e-3
# ⚠ RE-RECORDED when the pyramidal edge builder stopped leaving a STALE
# JACOBIAN on non-touching contacts. The builder's early-out zeroed only the
# two slots named `ws_Jt1_idx`/`ws_Jt2_idx`, which alias pyramid edges 0 and 1;
# edges 2 and 3 live further up the same region and kept the PREVIOUS step's
# Jacobian, so a contact that had separated still pushed a row into the Newton
# solve and still got a force written into its record.
# The mechanism was isolated, not inferred: zeroing D and bias as well leaves
# the fingerprint bit-identical (everything flows through Je), and zeroing only
# edges 0-1 reproduces the old value exactly.
# ⚠ THESE NUMBERS ARE NOT MuJoCo-ANCHORED. The originals were frozen from the
# legacy Euler+PGS pipeline, which carried the same defect; the case for the
# new values is that a geom which is not touching must contribute neither a
# Jacobian nor a contact force. A MuJoCo-anchored Walker2D contact-FORCE gate
# is still missing — every MuJoCo-anchored test in tests/physics3d passes
# unchanged either way, which is exactly why this went unnoticed.
# The contact-record fingerprint moves 1.5-1.7%; qpos/qvel/qacc move <2e-6,
# i.e. the defect was almost entirely in the RECORD, not the dynamics.
comptime GOLD_NCON = 12  # contacts per step (uniform across the 3 steps)
comptime GOLD_QPOS = 14.316056373878382
comptime GOLD_QVEL = -38.348557934165
comptime GOLD_QACC = -5676.547218322754
comptime GOLD_CON = 420119.10779341473


def _check(name: String, got: Float64, gold: Float64) raises:
    var denom = abs(gold) if abs(gold) > 1e-9 else 1.0
    var rel = abs(got - gold) / denom
    if rel > GOLD_RTOL and not default_accelerator().is_nvidia_gpu():
        raise Error(
            name + " fingerprint " + String(got) + " != golden "
            + String(gold) + " (rel " + String(rel) + ")"
        )


def main() raises:
    print("--- Euler WITH CONTACTS fields GOLDEN gate: Walker2D BATCH=", BATCH)
    var ctx = DeviceContext()

    var mf = Model[DTYPE, MD]()
    Walker2dModel.init_fields[DTYPE](ctx, mf)

    var d = Data[DTYPE, MD, BATCH]()
    var dc = Data[DTYPE, MD, BATCH]()
    for e in range(BATCH):
        for i in range(NQ):
            var qp = Scalar[DTYPE]((e * 5 + i * 3) % 5 - 2) / 40.0
            if i == 1:
                qp = 1.10  # feet penetrate the floor
            d.qpos.data[e * NQ + i] = qp
            dc.qpos.data[e * NQ + i] = qp
        for i in range(NV):
            var qv = Scalar[DTYPE]((e * 7 + i * 5) % 7 - 3) / 20.0
            if i == 1:
                qv = -0.5  # falling
            var qf = Scalar[DTYPE]((e * 13 + i * 9) % 9 - 4) / 4.0
            d.qvel.data[e * NV + i] = qv
            d.qfrc.data[e * NV + i] = qf
            dc.qvel.data[e * NV + i] = qv
            dc.qfrc.data[e * NV + i] = qf
    d.upload_all(ctx)

    var integ = EulerIntegrator[DTYPE, MD, CONE, BATCH]()
    integ.prepare_gpu(ctx)
    var integ_c = EulerIntegrator[DTYPE, MD, CONE, BATCH]()

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

    # --- final fields-GPU fingerprint (order-sensitive checksums) ---
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

    # --- independent CPU oracle: fields-CPU == fields-GPU ---
    var worst = Float64(0)
    for i in range(BATCH * NQ):
        var err = abs(Float64(dc.qpos.data[i]) - Float64(d.qpos.data[i]))
        if err > worst:
            worst = err
    print("  fields-CPU vs fields-GPU final qpos worst err:", worst)
    if worst > 1e-2:
        raise Error("fields-CPU contact dynamics diverged from GPU")
    print("  PASS: fields-CPU within 1e-2")
    print("test_contact_solve_fields: ALL PASS")
