"""End-to-end ENV loop on the fields path: InvertedPendulum closed-loop
balancing (obs -> controller -> action -> qfrc -> physics -> obs).

Originally gated bit-exact against the legacy GPU slab pipeline. That legacy
reference was frozen during Phase-0 of the physics3d sunset, so this gate now:
  * runs the closed loop on the fields path (GPU) with a deterministic PD
    controller and checks the behavior is sane (pole stays up, bounded, ends
    upright) — a device-robust behavioral anchor,
  * checks the final obs + episode returns against a frozen GOLDEN fingerprint
    (Apple-gated; f32 GPU accumulation drifts slightly on NVIDIA), and
  * checks fields-CPU closed loop == fields-GPU (independent CPU oracle).

InvertedPendulum: contact-free (contype=0), damping=1, slide limits +-1, single
motor (gear=100, ctrlrange +-3), obs = qpos||qvel (OBS_DIM=4), FRAME_SKIP=2.
Regenerate goldens after an INTENTIONAL physics change: HARVEST=True, run on
Apple, paste, HARVEST=False.

Run: pixi run -e apple mojo run -I . tests/physics3d/test_ip_fields_env_loop.mojo
"""

from std.math import abs
from std.sys import default_accelerator
from max.gpu.host import DeviceContext

from noeira.nn.core.tensor import TensorImpl
from noeira.physics3d.integrator.euler import EulerIntegrator
from noeira.physics3d.fields import Data, Model, Dims, DimsLike
from noeira.envs.phyics3d_obs import extract_obs_qpos_qvel
from noeira.physics3d.model.model_dims import ModelDims
from noeira.envs.inverted_pendulum.inverted_pendulum_xml import (
    InvertedPendulumModel,
)

comptime DTYPE = DType.float32
comptime IPM = InvertedPendulumModel
comptime NQ = IPM.NQ  # 2
comptime NV = IPM.NV  # 2
comptime NBODY = IPM.NBODY
comptime NJOINT = IPM.NJOINT
comptime NGEOM = IPM.NGEOM
comptime MC = IPM.MAX_CONTACTS
comptime NSITE = IPM.NSITE
comptime NEQ = IPM.MAX_EQUALITY
comptime NTEN = IPM.MAX_TENDON
comptime NEXCL = IPM.NEXCLUDE
comptime MD = ModelDims[IPM]
comptime BATCH = 2
comptime OBS_DIM = NQ + NV  # obs_qpos_skip=0
comptime FRAME_SKIP = 2
comptime N_CTRL_STEPS = 60
comptime GEAR: Float64 = 100.0
comptime CTRL_MAX: Float64 = 3.0

# --- GOLDEN fingerprints (frozen from the legacy-validated fields-GPU run) ----
comptime HARVEST = False  # True => print fingerprints + skip asserts (regen)
comptime GOLD_RTOL = 1e-3
comptime GOLD_R0 = 60.0  # episode return env0 (|pole|<0.2 per step)
comptime GOLD_R1 = 60.0
comptime GOLD_OBS = 0.10425555426627398  # checksum of the final obs across both envs


@always_inline
def _controller(obs: Array[Float64, OBS_DIM]) -> Float64:
    """Deterministic PD balancing controller on [x, theta, xd, thd]."""
    var u = 0.3 * obs[0] + 0.8 * obs[2] + 6.0 * obs[1] + 1.5 * obs[3]
    if u > CTRL_MAX:
        u = CTRL_MAX
    elif u < -CTRL_MAX:
        u = -CTRL_MAX
    return u


def main() raises:
    print(
        "--- IP closed-loop env on fields GPU, BATCH=", BATCH,
        " ctrl steps=", N_CTRL_STEPS, "x", FRAME_SKIP, "substeps ---",
    )
    var ctx = DeviceContext()

    var mf = Model[DTYPE, MD]()
    IPM.init_fields[DTYPE](ctx, mf)

    var pole0 = List[Float64]()
    pole0.append(0.05)
    pole0.append(-0.12)

    var d = Data[DTYPE, MD, BATCH]()
    for e in range(BATCH):
        d.qpos.data[e * NQ + 1] = Scalar[DTYPE](pole0[e])
    d.upload_all(ctx)

    var integ = EulerIntegrator[DTYPE, MD, BATCH=BATCH]()
    integ.prepare_gpu(ctx)
    var obs_t = TensorImpl[DTYPE].alloc(BATCH * OBS_DIM)
    obs_t.upload(ctx)

    var total_reward = List[Float64](length=BATCH, fill=0.0)
    var max_angle = List[Float64](length=BATCH, fill=0.0)
    for step in range(N_CTRL_STEPS):
        extract_obs_qpos_qvel["gpu", DTYPE, OBS_QPOS_SKIP=0, BATCH=BATCH](d, obs_t, ctx)
        obs_t.download(ctx)
        for e in range(BATCH):
            var obs_arr = Array[Float64, OBS_DIM](uninitialized=True)
            for i in range(OBS_DIM):
                obs_arr[i] = Float64(obs_t.data[e * OBS_DIM + i])
            var u = _controller(obs_arr)
            d.qfrc.data[e * NV + 0] = Scalar[DTYPE](GEAR * u)
            d.qfrc.data[e * NV + 1] = Scalar[DTYPE](0)

            var angle = Float64(obs_t.data[e * OBS_DIM + 1])
            if angle > -0.2 and angle < 0.2:
                total_reward[e] += 1.0
            var a_abs = angle if angle > 0 else -angle
            if a_abs > max_angle[e]:
                max_angle[e] = a_abs
        d.qfrc.upload(ctx)
        for _ in range(FRAME_SKIP):
            integ.step["gpu"](d, mf, ctx)

    var final0 = Float64(obs_t.data[0 * OBS_DIM + 1])
    var final1 = Float64(obs_t.data[1 * OBS_DIM + 1])
    print(
        "  episode returns (|pole|<0.2):", total_reward[0], total_reward[1],
        "/", N_CTRL_STEPS, " max|angle|:", max_angle[0], max_angle[1],
        " final angle:", final0, final1,
    )
    # Behavioral sanity (device-robust): pole never falls, ends near upright.
    var f0 = final0 if final0 > 0 else -final0
    var f1 = final1 if final1 > 0 else -final1
    if max_angle[0] > 0.5 or max_angle[1] > 0.5 or f0 > 0.15 or f1 > 0.15:
        raise Error("controller failed to keep the pole up — dynamics wrong")
    print("  PASS: both envs kept the pole up (bounded + upright at end)")

    # --- GOLDEN fingerprint of the fields-GPU closed loop (Apple-gated) ---
    var fp_obs = Float64(0)
    for i in range(BATCH * OBS_DIM):
        fp_obs += Float64(obs_t.data[i]) * Float64(i + 1)
    if HARVEST:
        print("  HARVEST GOLD_R0  =", total_reward[0])
        print("  HARVEST GOLD_R1  =", total_reward[1])
        print("  HARVEST GOLD_OBS =", fp_obs)
    elif not default_accelerator().is_nvidia_gpu():
        if total_reward[0] != GOLD_R0 or total_reward[1] != GOLD_R1:
            raise Error("episode returns differ from golden")
        var denom = abs(GOLD_OBS) if abs(GOLD_OBS) > 1e-9 else 1.0
        if abs(fp_obs - GOLD_OBS) / denom > GOLD_RTOL:
            raise Error(
                "final obs fingerprint " + String(fp_obs) + " != golden "
                + String(GOLD_OBS)
            )
        print("  PASS: fields-GPU matches golden fingerprint")

    # --- independent CPU oracle: fields-CPU closed loop == fields-GPU ---
    var dc = Data[DTYPE, MD, BATCH]()
    for e in range(BATCH):
        dc.qpos.data[e * NQ + 1] = Scalar[DTYPE](pole0[e])
    var integ_c = EulerIntegrator[DTYPE, MD, BATCH=BATCH]()
    var obs_c = TensorImpl[DTYPE].alloc(BATCH * OBS_DIM)
    for _ in range(N_CTRL_STEPS):
        extract_obs_qpos_qvel["cpu", DTYPE, OBS_QPOS_SKIP=0, BATCH=BATCH](dc, obs_c)
        for e in range(BATCH):
            var obs_arr = Array[Float64, OBS_DIM](uninitialized=True)
            for i in range(OBS_DIM):
                obs_arr[i] = Float64(obs_c.data[e * OBS_DIM + i])
            var u = _controller(obs_arr)
            dc.qfrc.data[e * NV + 0] = Scalar[DTYPE](GEAR * u)
            dc.qfrc.data[e * NV + 1] = Scalar[DTYPE](0)
        for _ in range(FRAME_SKIP):
            integ_c.step["cpu"](dc, mf)
    var worst = Float64(0)
    d.qpos.download(ctx)
    for e in range(BATCH):
        for i in range(NQ):
            var err = abs(
                Float64(dc.qpos.data[e * NQ + i])
                - Float64(d.qpos.data[e * NQ + i])
            )
            if err > worst:
                worst = err
    print("  fields-CPU closed loop vs fields-GPU, final qpos worst err:", worst)
    if worst > 1e-2:
        raise Error("fields-CPU closed loop diverged from GPU")
    print("  PASS: fields-CPU closed loop within 1e-2")
    print("test_ip_fields_env_loop: ALL PASS")
