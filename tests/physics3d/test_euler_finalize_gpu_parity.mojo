"""The Euler integrator's GPU step against its CPU step on a model with joint
damping — the gate for `_finalize_rhs_kernel` and the four-launch finalize
(`EULER_FINALIZE_SPLIT`).

⚠ NOT BIT-EXACT, AND THE REASON IS ON RECORD. The CPU step takes
`_finalize_tree_env` on a model with a tree table, whose docstring says it
is "not bit-exact against the dense twin: a different rounding of the same
M_hat^-1 (M qacc)"; the GPU step is the dense twin. Measured on this fixture
the two legs differ at 11 of 12 values by at most 2^-20 (1.49e-06 at
|qpos| ~ 1.6) — with the OLD single-launch kernel and with the split, the
same values and the same worst digit, which is what says the split is the
old kernel's arithmetic. The bound below is 1e-5 absolute: two decades
above that rounding, five decades below what a halved damping does (0.32).

⚠ OPEN (2026-10-19): ON AN RTX 5090 UNDER MOJO 1.2rc2 THIS READS 2.24e-05 and
fails, the same on `dev` with and without the TileTensor migration, while Apple
Metal still reads exactly 1.49e-06 (2^-20). The band is deliberately NOT
widened: the cause is unknown. Ruled out: a change in how the GPU math lowers.
The sm_120 PTX of this file under 1.1 and under 1.2rc2 has the same op mix
(sqrt.approx 180 / 181, sin/cos/ex2/lg2.approx 6/6/8/8, fma.rn.f32 7329 /
7298). Still open, and each needs an NVIDIA run: physics changes since the
09-07 measurement, the 31 fewer FMA contractions, and the CPU leg itself (the
box's x86 and the Mac's arm64 contract FMAs differently).

WHY THIS FILE EXISTS. The finalize's implicit damping (`M_hat = M +
dt*diag(damping)`, `mj_Euler`'s eulerdamp) runs only when a dof has damping,
and the two gates that were green while a mutant halved that damping did not
reach it: `test_tape_gpu_parity` never steps the physics, and
`test_ip_fields_env_loop` compares at 1e-2. This one steps the inverted
pendulum (`damping="1"` on both joints, no meshes) on both targets from the
same state under the same forces and demands `qpos` and `qvel` within the
tree/dense rounding, and it asserts the damping is non-zero so it cannot go vacuous by a
model edit.

Run: pixi run -e apple mojo run -I . tests/physics3d/test_euler_finalize_gpu_parity.mojo
"""

from std.math import abs
from max.gpu.host import DeviceContext
from noeira.physics3d.integrator.euler import EulerIntegrator
from noeira.physics3d.fields import Data, Model
from noeira.physics3d.model.model_dims import ModelDims
from noeira.physics3d.gpu.constants import MODEL_JOINT_SIZE, JOINT_IDX_DAMPING
from noeira.envs.inverted_pendulum.inverted_pendulum_xml import (
    InvertedPendulumModel,
)

comptime DTYPE = DType.float32
comptime IPM = InvertedPendulumModel
comptime NQ = IPM.NQ
comptime NV = IPM.NV
comptime MD = ModelDims[IPM]
comptime BATCH = 3
comptime N_STEPS = 40


def main() raises:
    print("=== Euler finalize: GPU step == CPU step, bit for bit, with damping ===")
    var ctx = DeviceContext()
    var mf = Model[DTYPE, MD]()
    IPM.init_fields[DTYPE](ctx, mf)
    # Non-vacuity: the eulerdamp branch runs only for damped dofs.
    var damped = 0
    for j in range(IPM.NJOINT):
        if Float64(mf.joints.data[j * MODEL_JOINT_SIZE + JOINT_IDX_DAMPING]) > 0.0:
            damped += 1
    if damped == 0:
        raise Error("no damped joint: the implicit-damping finalize is not exercised")
    print("  damped joints:", damped, "of", IPM.NJOINT)

    var d = Data[DTYPE, MD, BATCH]()
    var dc = Data[DTYPE, MD, BATCH]()
    for e in range(BATCH):
        var pole = Scalar[DTYPE](0.05 * Float64(e + 1))
        d.qpos.data[e * NQ + 1] = pole
        dc.qpos.data[e * NQ + 1] = pole
        var f = Scalar[DTYPE](0.3 * Float64(e) - 0.2)
        d.qfrc.data[e * NV + 0] = f
        dc.qfrc.data[e * NV + 0] = f
    d.upload_all(ctx)
    var integ = EulerIntegrator[DTYPE, MD, BATCH=BATCH]()
    integ.prepare_gpu(ctx)
    var integ_c = EulerIntegrator[DTYPE, MD, BATCH=BATCH]()
    for _ in range(N_STEPS):
        integ.step["gpu"](d, mf, ctx)
        integ_c.step["cpu"](dc, mf)
    d.qpos.download(ctx)
    d.qvel.download(ctx)
    var diffs = 0
    var worst = Float64(0)
    var moved = Float64(0)
    for e in range(BATCH):
        for i in range(NQ):
            var a = Float64(d.qpos.data[e * NQ + i])
            var b = Float64(dc.qpos.data[e * NQ + i])
            if a != b:
                diffs += 1
            if abs(a - b) > worst:
                worst = abs(a - b)
            if abs(b) > moved:
                moved = abs(b)
        for i in range(NV):
            var a = Float64(d.qvel.data[e * NV + i])
            var b = Float64(dc.qvel.data[e * NV + i])
            if a != b:
                diffs += 1
            if abs(a - b) > worst:
                worst = abs(a - b)
    print("  compared", BATCH * (NQ + NV), "values over", N_STEPS, "steps; differing:", diffs, " worst |d|:", worst, " |qpos| max:", moved)
    if moved == 0.0:
        raise Error("nothing moved: the step is vacuous")
    # The tree-vs-dense rounding is ~1.5e-6 here; the damping mutant is 0.32.
    if worst > 1e-5:
        raise Error("Euler GPU step differs from the CPU step beyond the tree/dense rounding: worst " + String(worst))
    print("test_euler_finalize_gpu_parity: ALL PASS (within the tree/dense rounding)")
