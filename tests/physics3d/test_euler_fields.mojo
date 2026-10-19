"""Regression gate (GOLDEN-frozen): stateful per-field Euler integrator, FULL
STEP, unconstrained-but-limit-aware dynamics on Walker2D.

Originally validated BIT-EXACT against the legacy Euler step (step_kernel +
limits + finalize). That legacy reference was frozen into the GOLDEN
fingerprints below during Phase-0 of the physics3d sunset, so this gate survives
deletion of the legacy slab/kernels. It checks:
  * fields-GPU reproduces the frozen (legacy-validated) fingerprint —
    order-sensitive checksums of the final qpos/qvel/qacc/xvel/xangvel, and
  * fields-CPU == fields-GPU (independent CPU oracle; tolerance).

Walker2D, BATCH=3 (distinct qpos/qvel/qfrc per env), 3 consecutive steps; the
thigh config violates walker2d's hinge range so the limit path is exercised.
Regenerate goldens after an INTENTIONAL physics change: set HARVEST=True, run
once on Apple, paste the printed values, set HARVEST=False.

Run: pixi run -e apple mojo run -I . tests/physics3d/test_euler_fields.mojo
"""

from std.math import abs
from max.gpu.host import DeviceContext
from std.sys import default_accelerator

from noeira.nn.core.tensor import TensorImpl
from noeira.physics3d.fields import Data, Model, Dims
from noeira.physics3d.integrator.euler import EulerIntegrator
from noeira.envs.walker2d.walker2d_xml import Walker2dModel
from noeira.physics3d.model.model_dims import ModelDims

comptime DTYPE = DType.float32
comptime NQ = Walker2dModel.NQ
comptime NV = Walker2dModel.NV
comptime NBODY = Walker2dModel.NBODY
comptime NJOINT = Walker2dModel.NJOINT
comptime NGEOM = Walker2dModel.NGEOM
comptime MAX_CONTACTS = Walker2dModel.MAX_CONTACTS
comptime NEQ = Walker2dModel.MAX_EQUALITY
comptime NTD = Walker2dModel.MAX_TENDON
comptime NSITE = Walker2dModel.NSITE
comptime NEXCL = Walker2dModel.NEXCLUDE
comptime MD = ModelDims[Walker2dModel]
comptime BATCH = 3
comptime N_STEPS = 3

# --- GOLDEN fingerprints (frozen from the legacy-validated fields-GPU run) ----
comptime HARVEST = False  # True => print fingerprints + skip asserts (regen)
comptime GOLD_RTOL = 1e-3
comptime GOLD_QPOS = 42.2036153235822
comptime GOLD_QVEL = -12.21184940263629
comptime GOLD_QACC = -1929.7593653202057
# Re-harvested 2026-07-29. qpos/qvel/qacc are unchanged — the state trajectory
# is identical — but `xvel` was recomputed when the hinge angular-to-linear
# coupling bug was fixed (commit f0d35e2c), so the frozen value here described
# the OLD, wrong xvel. The new one is not taken on trust: walker2d's xvel is
# gated directly against MuJoCo to 1.2e-10 in
# `tests/physics3d/test_body_velocities_vs_mujoco.mojo`, on this same model.
comptime GOLD_XVEL = 16.63215710222721
comptime GOLD_XANG = -46.879036627709866


def _check(name: String, got: Float64, gold: Float64) raises:
    var denom = abs(gold) if abs(gold) > 1e-9 else 1.0
    var rel = abs(got - gold) / denom
    if rel > GOLD_RTOL and not default_accelerator().is_nvidia_gpu():
        raise Error(
            name + " fingerprint " + String(got) + " != golden "
            + String(gold) + " (rel " + String(rel) + ")"
        )


def main() raises:
    print("--- Euler full-step GOLDEN gate: walker2d BATCH=", BATCH)
    var ctx = DeviceContext()

    var mf = Model[DTYPE, MD]()
    Walker2dModel.init_fields[DTYPE](ctx, mf)

    var d = Data[DTYPE, MD, BATCH]()
    var dc = Data[DTYPE, MD, BATCH]()
    for e in range(BATCH):
        for i in range(NQ):
            var qp = Scalar[DTYPE]((e * 7 + i * 3) % 5 - 2) / 20.0
            if i == 1:
                qp = 1.25  # rootz standing height
            d.qpos.data[e * NQ + i] = qp
            dc.qpos.data[e * NQ + i] = qp
        for i in range(NV):
            var qv = Scalar[DTYPE]((e * 11 + i * 5) % 7 - 3) / 10.0
            var qf = Scalar[DTYPE]((e * 13 + i * 9) % 9 - 4) / 2.0
            d.qvel.data[e * NV + i] = qv
            d.qfrc.data[e * NV + i] = qf
            dc.qvel.data[e * NV + i] = qv
            dc.qfrc.data[e * NV + i] = qf
    d.upload_all(ctx)

    var integ = EulerIntegrator[DTYPE, MD, BATCH=BATCH]()
    integ.prepare_gpu(ctx)
    var integ_c = EulerIntegrator[DTYPE, MD, BATCH=BATCH]()

    for step in range(N_STEPS):
        integ.step["gpu", False](d, mf, ctx)  # CONTACTS=False: unconstrained
        integ_c.step["cpu", False](dc, mf)
        print("  step", step, ": ok")

    # --- final fields-GPU fingerprint (order-sensitive checksums) ---
    d.qpos.download(ctx)
    d.qvel.download(ctx)
    d.qacc.download(ctx)
    d.xvel.download(ctx)
    d.xangvel.download(ctx)
    var fp_qpos = Float64(0)
    var fp_qvel = Float64(0)
    var fp_qacc = Float64(0)
    for e in range(BATCH):
        for i in range(NQ):
            fp_qpos += Float64(d.qpos.data[e * NQ + i]) * Float64(e * NQ + i + 1)
        for i in range(NV):
            fp_qvel += Float64(d.qvel.data[e * NV + i]) * Float64(e * NV + i + 1)
            fp_qacc += Float64(d.qacc.data[e * NV + i]) * Float64(e * NV + i + 1)
    var fp_xvel = Float64(0)
    var fp_xang = Float64(0)
    for i in range(BATCH * NBODY * 3):
        fp_xvel += Float64(d.xvel.data[i]) * Float64(i + 1)
        fp_xang += Float64(d.xangvel.data[i]) * Float64(i + 1)

    if HARVEST:
        print("  HARVEST GOLD_QPOS =", fp_qpos)
        print("  HARVEST GOLD_QVEL =", fp_qvel)
        print("  HARVEST GOLD_QACC =", fp_qacc)
        print("  HARVEST GOLD_XVEL =", fp_xvel)
        print("  HARVEST GOLD_XANG =", fp_xang)
    else:
        _check("qpos", fp_qpos, GOLD_QPOS)
        _check("qvel", fp_qvel, GOLD_QVEL)
        _check("qacc", fp_qacc, GOLD_QACC)
        _check("xvel", fp_xvel, GOLD_XVEL)
        _check("xangvel", fp_xang, GOLD_XANG)
        print("  PASS: fields-GPU matches golden fingerprint")

    # --- independent CPU oracle: fields-CPU == fields-GPU ---
    var worst = Float64(0)
    for i in range(BATCH * NQ):
        var err = abs(Float64(dc.qpos.data[i]) - Float64(d.qpos.data[i]))
        if err > worst:
            worst = err
    for i in range(BATCH * NV):
        var err = abs(Float64(dc.qvel.data[i]) - Float64(d.qvel.data[i]))
        if err > worst:
            worst = err
    print("  fields-CPU vs fields-GPU after", N_STEPS, "steps, worst err:", worst)
    if worst > 1e-3:
        raise Error("fields-CPU tolerance exceeded")
    print("  PASS: fields-CPU within 1e-3")
    print("test_euler_fields: ALL PASS")
