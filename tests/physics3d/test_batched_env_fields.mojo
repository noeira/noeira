"""GPU-batched fields env gate (GOLDEN-frozen): Phyics3dBatchedEnv on
Walker2D (contacts), pinned to the PGS + serial + dense config.

Originally validated BIT-EXACT against the legacy Phyics3dEnv slab pipeline
(reset/pre_step/apply_actions/RK4+PGS/cfrc_ext/cvel/extract-obs + selective
reset). That legacy reference was frozen during Phase-0 of the physics3d sunset,
so this gate now drives the facade alone and checks:
  * the facade reset -> N steps -> selective reset -> steps loop runs and
    produces contacts (non-vacuous), and
  * obs / qpos fingerprints + total contact count match a frozen GOLDEN
    (Apple-gated; f32 GPU accumulation drifts slightly on NVIDIA).

The production config (newton + parallel + treewalk) is gated separately in
test_batched_env_fields_production.mojo. Regenerate goldens after an INTENTIONAL
physics change: HARVEST=True, run on Apple, paste, HARVEST=False.

Run: pixi run -e apple mojo run -I . tests/physics3d/test_batched_env_fields.mojo
"""

from std.math import abs
from max.gpu.host import DeviceContext
from std.sys import default_accelerator

from noeira.nn.constants import DT
from noeira.envs.phyics3d_batched_env import Phyics3dBatchedEnv
from noeira.envs.walker2d.walker2d_xml import Walker2dModel
from noeira.envs.walker2d.walker2d_config import Walker2dConfig
from noeira.physics3d.gpu.constants import META_IDX_NUM_CONTACTS, METADATA_SIZE

comptime NQ = Walker2dModel.NQ
comptime NV = Walker2dModel.NV
comptime NBODY = Walker2dModel.NBODY
comptime MC = Walker2dModel.MAX_CONTACTS
comptime OBS_DIM = Walker2dModel.OBS_DIM
comptime ACT_DIM = Walker2dModel.ACTION_DIM
comptime BATCH = 2
comptime N_STEPS = 20
comptime RESET_SEED = UInt64(123)

# Pinned to PGS + serial + dense (the config this gate's legacy reference used).
comptime FieldsEnv = Phyics3dBatchedEnv[
    Walker2dModel,
    Walker2dConfig,
    BATCH,
    TERMINATE_ON_UNHEALTHY=True,
    SOLVER="pgs",
    PARALLEL_GPU=False,
    CRBA_TREEWALK=False,
]

# --- GOLDEN fingerprints (frozen from the legacy-validated fields run) --------
comptime HARVEST = False  # True => print fingerprints + skip asserts (regen)
comptime GOLD_RTOL = 1e-3
comptime GOLD_NCON = 39  # total contacts over N_STEPS
comptime GOLD_OBS = -152.1723847836256  # final obs checksum
comptime GOLD_QPOS = -12.877537071704865  # final qpos checksum
comptime GOLD_RST_OBS = -134.76768249977613  # post-selective-reset obs checksum


def _action_val(t: Int, e: Int, j: Int) -> Scalar[DT]:
    return Scalar[DT]((t * 5 + e * 3 + j * 7) % 9 - 4) / 8.0


def _chk(name: String, got: Float64, gold: Float64) raises:
    var denom = abs(gold) if abs(gold) > 1e-9 else 1.0
    if abs(got - gold) / denom > GOLD_RTOL and not default_accelerator().is_nvidia_gpu():
        raise Error(name + " " + String(got) + " != golden " + String(gold))


def main() raises:
    print("--- Batched fields env GOLDEN gate: Walker2D (PGS/serial) ---")
    var ctx = DeviceContext()

    var env = FieldsEnv(ctx)
    env.reset_batch[BATCH](Optional(ctx), RESET_SEED)

    var h_act = ctx.enqueue_create_host_buffer[DT](BATCH * ACT_DIM)
    var h_obs = ctx.enqueue_create_host_buffer[DT](BATCH * OBS_DIM)
    ctx.synchronize()

    var total_ncon = 0
    for t in range(N_STEPS):
        for e in range(BATCH):
            for j in range(ACT_DIM):
                h_act[e * ACT_DIM + j] = _action_val(t, e, j)
        ctx.enqueue_copy(env._action, h_act)
        env.step_batch[BATCH](Optional(ctx), 0)
        env.d.meta.download(ctx)
        for e in range(BATCH):
            total_ncon += Int(
                env.d.meta.data[e * METADATA_SIZE + META_IDX_NUM_CONTACTS]
            )
    if total_ncon == 0:
        raise Error("no contacts over the run — gate is vacuous")
    print("  stepping ok over", N_STEPS, "steps, total contacts:", total_ncon)

    ctx.enqueue_copy(h_obs, env._obs)
    ctx.synchronize()
    var fp_obs = Float64(0)
    for i in range(BATCH * OBS_DIM):
        fp_obs += Float64(h_obs[i]) * Float64(i + 1)
    env.d.qpos.download(ctx)
    var fp_qpos = Float64(0)
    for i in range(BATCH * NQ):
        fp_qpos += Float64(env.d.qpos.data[i]) * Float64(i + 1)

    # ── selective reset: force env 0 done, env 1 live ─────────────────────
    var h_done = ctx.enqueue_create_host_buffer[DT](BATCH)
    ctx.synchronize()
    h_done[0] = Scalar[DT](1.0)
    for e in range(1, BATCH):
        h_done[e] = Scalar[DT](0.0)
    ctx.enqueue_copy(env._done, h_done)
    env.selective_reset_batch[BATCH](Optional(ctx), 0)
    ctx.enqueue_copy(h_obs, env._obs)
    ctx.synchronize()
    var fp_rst_obs = Float64(0)
    for i in range(BATCH * OBS_DIM):
        fp_rst_obs += Float64(h_obs[i]) * Float64(i + 1)
    print("  selective reset ok (env 0 reset, env 1 live)")

    # two more steps to prove the reset state feeds the physics correctly
    for t in range(2):
        for e in range(BATCH):
            for j in range(ACT_DIM):
                h_act[e * ACT_DIM + j] = _action_val(100 + t, e, j)
        ctx.enqueue_copy(env._action, h_act)
        env.step_batch[BATCH](Optional(ctx), 0)
    print("  post-reset stepping ok")

    if HARVEST:
        print("  HARVEST GOLD_NCON    =", total_ncon)
        print("  HARVEST GOLD_OBS     =", fp_obs)
        print("  HARVEST GOLD_QPOS    =", fp_qpos)
        print("  HARVEST GOLD_RST_OBS =", fp_rst_obs)
    else:
        if total_ncon != GOLD_NCON and not default_accelerator().is_nvidia_gpu():
            raise Error(
                "total contacts " + String(total_ncon) + " != golden "
                + String(GOLD_NCON)
            )
        _chk("obs", fp_obs, GOLD_OBS)
        _chk("qpos", fp_qpos, GOLD_QPOS)
        _chk("reset-obs", fp_rst_obs, GOLD_RST_OBS)
        print("  PASS: facade matches golden fingerprint")

    print("test_batched_env_fields: ALL PASS")
