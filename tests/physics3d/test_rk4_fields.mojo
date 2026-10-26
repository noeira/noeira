"""Regression gate (GOLDEN-frozen): stateful per-field RK4 integrator, FULL
STEP, contact-free AND constraint-free dynamics on Walker2D.

Originally validated BIT-EXACT against the legacy RK4 step (4 stage kernels +
combine, minus the solver launches = unconstrained RK4). That legacy reference
was frozen into the GOLDEN fingerprints below during Phase-0 of the physics3d
sunset, so this gate survives deletion of the legacy slab/kernels. It checks:
  * fields-GPU reproduces the frozen (legacy-validated) fingerprint —
    order-sensitive checksums of the final qpos/qvel/qacc,
  * fields-CPU == fields-GPU (independent CPU oracle; tolerance), and
  * the final pose stays strictly inside all joint ranges (limits inactive,
    so the contact/limit-free comparison is provably valid).

Walker2D, BATCH=3 (distinct qpos/qvel/qfrc per env), free-flight (rootz=2.0).
Model build routes through the offset-free init_fields (Stage E). Regenerate
goldens after an INTENTIONAL physics
change: set HARVEST=True, run once on Apple, paste the values, set HARVEST=False.

Run: pixi run -e apple mojo run -I . tests/physics3d/test_rk4_fields.mojo
"""

from std.math import abs
from max.gpu.host import DeviceContext
from std.sys import default_accelerator

from noeira.nn.core.tensor import TensorImpl
from noeira.physics3d.fields import Data, Model, Dims
from noeira.physics3d.integrator.rk4 import RK4Integrator
from noeira.physics3d.gpu.constants import (
    MODEL_JOINT_SIZE,
    JOINT_IDX_TYPE,
    JOINT_IDX_QPOS_ADR,
    JOINT_IDX_RANGE_MIN,
    JOINT_IDX_RANGE_MAX,
)
from noeira.physics3d.joint_types import JNT_HINGE, JNT_SLIDE
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
comptime GOLD_QPOS = -19.34484850493027
comptime GOLD_QVEL = 9.900580829940736
comptime GOLD_QACC = 1344.4496380127966


def _check(name: String, got: Float64, gold: Float64) raises:
    var denom = abs(gold) if abs(gold) > 1e-9 else 1.0
    var rel = abs(got - gold) / denom
    if rel > GOLD_RTOL and not default_accelerator().is_nvidia_gpu():
        raise Error(
            name + " fingerprint " + String(got) + " != golden "
            + String(gold) + " (rel " + String(rel) + ")"
        )


def _init_qpos(e: Int, i: Int) -> Scalar[DTYPE]:
    """Free-flight walker2d pose, strictly inside all joint ranges."""
    var ef = Scalar[DTYPE](e)
    if i == 0:  # rootx (unlimited slide)
        return Scalar[DTYPE](0.05) * ef - Scalar[DTYPE](0.05)
    elif i == 1:  # rootz: 2.0 -> torso ~2m up, no floor contact possible
        return Scalar[DTYPE](2.0)
    elif i == 2:  # rooty (unlimited hinge)
        return Scalar[DTYPE](0.04) * (ef - Scalar[DTYPE](1.0))
    elif i == 3:  # thigh
        return Scalar[DTYPE](-0.30) - Scalar[DTYPE](0.05) * ef
    elif i == 4:  # leg
        return Scalar[DTYPE](-0.50) + Scalar[DTYPE](0.03) * ef
    elif i == 5:  # foot
        return Scalar[DTYPE](-0.20) + Scalar[DTYPE](0.04) * ef
    elif i == 6:  # thigh_left
        return Scalar[DTYPE](-0.40) + Scalar[DTYPE](0.05) * ef
    elif i == 7:  # leg_left
        return Scalar[DTYPE](-0.35) - Scalar[DTYPE](0.04) * ef
    else:  # foot_left
        return Scalar[DTYPE](-0.15) - Scalar[DTYPE](0.03) * ef


def main() raises:
    print("--- RK4 full-step GOLDEN gate: walker2d BATCH=", BATCH)
    var ctx = DeviceContext()

    var mf = Model[DTYPE, MD]()
    Walker2dModel.init_fields[DTYPE](ctx, mf)

    var d = Data[DTYPE, MD, BATCH]()
    var dc = Data[DTYPE, MD, BATCH]()
    for e in range(BATCH):
        for i in range(NQ):
            var qp = _init_qpos(e, i)
            d.qpos.data[e * NQ + i] = qp
            dc.qpos.data[e * NQ + i] = qp
        for i in range(NV):
            var qv = Scalar[DTYPE]((e * 11 + i * 5) % 7 - 3) / 20.0
            var qf = Scalar[DTYPE]((e * 13 + i * 9) % 9 - 4) / 4.0
            d.qvel.data[e * NV + i] = qv
            d.qfrc.data[e * NV + i] = qf
            dc.qvel.data[e * NV + i] = qv
            dc.qfrc.data[e * NV + i] = qf
    d.upload_all(ctx)

    var integ = RK4Integrator[DTYPE, MD, BATCH=BATCH]()
    integ.prepare_gpu(ctx)
    var integ_c = RK4Integrator[DTYPE, MD, BATCH=BATCH]()

    for step in range(N_STEPS):
        integ.step["gpu", False](d, mf, ctx)
        integ_c.step["cpu", False](dc, mf)
        print("  step", step, ": ok")

    d.qpos.download(ctx)
    d.qvel.download(ctx)
    d.qacc.download(ctx)

    # No joint-limit violations in the final pose (limits provably inactive).
    for j in range(NJOINT):
        var jo = j * MODEL_JOINT_SIZE
        var jt = Int(mf.joints.data[jo + JOINT_IDX_TYPE])
        if jt != JNT_HINGE and jt != JNT_SLIDE:
            continue
        var rmin = mf.joints.data[jo + JOINT_IDX_RANGE_MIN]
        var rmax = mf.joints.data[jo + JOINT_IDX_RANGE_MAX]
        if not (rmin < rmax):
            continue  # unlimited joint
        var qadr = Int(mf.joints.data[jo + JOINT_IDX_QPOS_ADR])
        for e in range(BATCH):
            var qp = d.qpos.data[e * NQ + qadr]
            if qp <= rmin or qp >= rmax:
                raise Error(
                    "joint " + String(j) + " env " + String(e)
                    + " violates its range — pose selection broken"
                )
    print("  final qpos strictly inside all joint ranges (limits inactive)")

    # --- final fields-GPU fingerprint (order-sensitive checksums) ---
    var fp_qpos = Float64(0)
    var fp_qvel = Float64(0)
    var fp_qacc = Float64(0)
    for e in range(BATCH):
        for i in range(NQ):
            fp_qpos += Float64(d.qpos.data[e * NQ + i]) * Float64(e * NQ + i + 1)
        for i in range(NV):
            fp_qvel += Float64(d.qvel.data[e * NV + i]) * Float64(e * NV + i + 1)
            fp_qacc += Float64(d.qacc.data[e * NV + i]) * Float64(e * NV + i + 1)

    if HARVEST:
        print("  HARVEST GOLD_QPOS =", fp_qpos)
        print("  HARVEST GOLD_QVEL =", fp_qvel)
        print("  HARVEST GOLD_QACC =", fp_qacc)
    else:
        _check("qpos", fp_qpos, GOLD_QPOS)
        _check("qvel", fp_qvel, GOLD_QVEL)
        _check("qacc", fp_qacc, GOLD_QACC)
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
    print("test_rk4_fields: ALL PASS")
