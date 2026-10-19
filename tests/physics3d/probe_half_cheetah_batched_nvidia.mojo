"""NVIDIA crash probe: HalfCheetah batched env ALONE, no agent.

⚠ NOT BUILT ANYWHERE YET. Written on an Apple laptop while a physics sweep was
running, so it has not been compiled on either target. Treat it as a draft
until it builds — an unbuilt generic is uncompiled code.

WHY ENV-ONLY IS THE RIGHT FIRST SPLIT. The reported crash is at
`Step 7520/50000` with `Train: 0`, and `sac_half_cheetah_training_gpu.mojo`
sets `learning_starts=10_000`. So NOT ONE GRADIENT UPDATE HAD RUN. That rules
out, without needing to test them:

    * the SAC train step and its losses,
    * the grouped multi-tensor Adam apply (NVIDIA-only),
    * `USE_TRAIN_CUDA_GRAPH=True` — the train graph is captured at the first
      train step, which never happened.

What WAS running: env stepping, replay insertion, and the RemoteLogger
(`diag_every=1000`, so ~7 flushes by then). This probe removes the last two.

    survives 20k steps  =>  physics is clean; suspect replay / logger / driver
    crashes             =>  physics, and `solve_newton_blocked` is the prime
                            suspect (see below)

⚠ `solve_newton_blocked` IS NVIDIA-ONLY AND APPLE CANNOT REACH IT. The
production dispatch in `newton_solve.mojo` is

    comptime if CONE_TYPE == ConeType.PYRAMIDAL:
        if default_accelerator().is_nvidia_gpu():
            solve_newton_blocked[...]

so on this laptop that `if` is always False and the one-thread-per-env kernel
runs instead. HalfCheetah is PYRAMIDAL, so on NVIDIA it takes the cooperative
blocked kernel that no Apple run has ever executed in production dispatch.
Every engine defect fixed during the dm_control port was gated against Apple
and CPU. That is the structural reason an NVIDIA-only regression can sit here
undetected, and it is why this probe exists rather than another code reading.

RUN IT UNDER compute-sanitizer. A bare crash gives an AsyncRT stack that says
nothing; memcheck names the offending kernel and access:

    pixi run -e nvidia compute-sanitizer --tool memcheck \
        mojo run -I . tests/physics3d/probe_half_cheetah_batched_nvidia.mojo

It is slow (10-50x), which is why N_ENVS and STEPS below are small. Raise
STEPS until it reproduces; the reported crash needed ~7520/32 = ~235 driver
iterations, and this probe's step counter is per-iteration, so ~300 should be
plenty.

⚠ SOLVER_STEPS lets you flip to the NON-blocked kernel WITHOUT touching the
engine: `CONE_TYPE=ELLIPTIC` skips the `PYRAMIDAL` branch entirely. If the
crash disappears under ELLIPTIC, it is the blocked kernel, and that is a
one-line bisect rather than an afternoon.
"""

from max.gpu.host import DeviceContext
from std.math import sin

from noeira.nn.constants import DT
from noeira.envs.phyics3d_batched_env import Phyics3dBatchedEnv
from noeira.envs.half_cheetah import HalfCheetahModel, HalfCheetahConfig
from noeira.physics3d.gpu.constants import (
    METADATA_SIZE,
    META_IDX_NUM_CONTACTS,
)

# Small on purpose — compute-sanitizer is 10-50x slow. The production script
# uses 32; keep this at 32 too so the contact patterns match, and cut STEPS.
comptime N_ENVS = 32
comptime STEPS = 400
comptime REPORT_EVERY = 25

comptime EnvT = Phyics3dBatchedEnv[
    HalfCheetahModel, HalfCheetahConfig, N_ENVS, TERMINATE_ON_UNHEALTHY=False
]


def main() raises:
    comptime ACT_DIM = HalfCheetahModel.ACTION_DIM
    comptime OBS_DIM = HalfCheetahModel.OBS_DIM
    comptime NQ = HalfCheetahModel.NQ

    print("HalfCheetah batched env probe — N_ENVS =", N_ENVS,
          " STEPS =", STEPS)
    print("  no agent, no replay, no logger — env stepping only")

    with DeviceContext() as ctx:
        var env = EnvT(ctx)
        env.reset_batch[N_ENVS](Optional(ctx), UInt64(42))

        var h_act = ctx.enqueue_create_host_buffer[DT](N_ENVS * ACT_DIM)
        var h_obs = ctx.enqueue_create_host_buffer[DT](N_ENVS * OBS_DIM)
        ctx.synchronize()

        var max_ncon = 0
        for t in range(STEPS):
            # Deterministic wiggle, decorrelated per lane. Not a policy — the
            # point is to drive the cheetah into varied contact states, which
            # is what a state-dependent fault needs.
            for e in range(N_ENVS):
                for j in range(ACT_DIM):
                    h_act[e * ACT_DIM + j] = Scalar[DT](
                        0.9 * sin(
                            Float64(t) * 0.13
                            + Float64(e) * 0.7
                            + Float64(j) * 1.7
                        )
                    )
            ctx.enqueue_copy(env._action, h_act)
            env.step_batch[N_ENVS](Optional(ctx), 0)

            # ⚠ SYNC EVERY STEP. A device fault surfaces at the next sync, so
            # without this the reported step number is meaningless — the crash
            # would be attributed to whichever later call happened to sync.
            ctx.enqueue_copy(h_obs, env._obs)
            env.d.meta.download(ctx)
            ctx.synchronize()

            var step_ncon = 0
            for e in range(N_ENVS):
                step_ncon += Int(
                    env.d.meta.data[e * METADATA_SIZE + META_IDX_NUM_CONTACTS]
                )
            if step_ncon > max_ncon:
                max_ncon = step_ncon

            if t % REPORT_EVERY == 0:
                # qpos[0] is rootx; if it is NaN the physics diverged rather
                # than faulted, which is a different bug with a different fix.
                env.d.qpos.download(ctx)
                ctx.synchronize()
                print(
                    "  t", t, " ncon(batch)", step_ncon,
                    " rootx[0]", Float64(env.d.qpos.data[0]),
                    " obs[0]", Float64(h_obs[0]),
                )

        print("SURVIVED", STEPS, "steps. peak ncon over the batch:", max_ncon)
        print("  => the env/physics path is not the crash; move the suspicion")
        print("     to replay insertion, the driver, or the RemoteLogger.")
