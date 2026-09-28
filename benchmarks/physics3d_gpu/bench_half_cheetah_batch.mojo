"""HalfCheetah env-step throughput on the GPU vs the number of parallel envs.

The GPU row of the community-meeting physics table: `Phyics3dBatchedEnv`
(Euler + Newton, as the XML and MuJoCo) stepping N HalfCheetahs at once, for
N = 1, 16, 64, 256, 1024, 4096. Same protocol as the CPU harness
(`benchmarks/physics3d_cpu/harness.mojo`) where it can be: reset, ctrl = 0.1
on every actuator, WARMUP untimed steps, STEPS steps timed between two
device synchronisations.

What one step is here: a full ENV step — FRAME_SKIP (5) physics steps plus
the env's observation / reward / done extraction and the selective reset an
RL driver runs every step. So `physics_steps_per_s = env_steps_per_s * 5` is
a conservative number next to `mj_step` or MuJoCo Warp's `step` (physics
only). The action upload (H2D, async) is inside the timed loop too, as it is
in training.

Two modes per N: EAGER (every kernel launched each step) and GRAPH (the
physics `step_batch` captured once into a CUDA graph and replayed, as the
off-policy driver's `USE_ENV_CUDA_GRAPH` does; action upload and selective
reset stay eager). Both start from the same reset with the same seeds, so a
correct graph reproduces the eager run bit for bit: each row prints a
checksum of the observations after the timed loop to prove it. The graph
needs the CUDA interceptor preloaded — run the binary through `pixi run`.

    pixi run -e nvidia mojo build -I . benchmarks/physics3d_gpu/bench_half_cheetah_batch.mojo -o bench_hc_batch
    ./bench_hc_batch [warmup=100] [steps=1000]
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.sys import argv
from std.time import perf_counter_ns

from noeira.cuda import CUDAGraph, maybe_capture_replay
from noeira.nn.constants import DT
from noeira.envs.phyics3d_batched_env import Phyics3dBatchedEnv
from noeira.envs.half_cheetah import HalfCheetahModel, HalfCheetahConfig


comptime ACT = HalfCheetahConfig.ACTION_DIM
comptime FRAME_SKIP = HalfCheetahConfig.FRAME_SKIP


def bench[
    N: Int, USE_GRAPH: Bool
](ctx: DeviceContext, warmup: Int, steps: Int) raises:
    var env = Phyics3dBatchedEnv[
        HalfCheetahModel, HalfCheetahConfig, N, TERMINATE_ON_UNHEALTHY=False
    ](ctx)
    env.reset_batch[N](ctx=ctx, rng_seed=UInt64(7))

    var act_host = ctx.enqueue_create_host_buffer[DT](N * ACT)
    ctx.synchronize()
    var ah = act_host.unsafe_ptr()
    for k in range(N * ACT):
        ah[k] = Scalar[DT](0.1)
    var act_dev = DeviceBuffer[DT](ctx, env.action_ptr(), N * ACT, owning=False)

    var graph: Optional[CUDAGraph] = None

    @always_inline
    @parameter
    def physics() raises capturing:
        # RNG-free (the seed is unused by the physics step), so it is safe to
        # bake into a graph.
        env.step_batch[N](ctx=ctx, rng_seed=UInt64(1))

    @always_inline
    def one_step(it: Int) raises capturing:
        ctx.enqueue_copy(act_dev, act_host)
        comptime if USE_GRAPH:
            maybe_capture_replay[physics](graph, ctx)
        else:
            physics()
        env.selective_reset_batch[N](ctx=ctx, rng_seed=UInt64(it + 1) * 7)

    for it in range(warmup):
        one_step(it)
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for it in range(steps):
        one_step(warmup + it)
    ctx.synchronize()
    var dt = Float64(perf_counter_ns() - t0) / 1e9
    var env_sps = Float64(N * steps) / dt

    # Checksum of the final observations (eager == graph when capture is right).
    comptime OBS = HalfCheetahConfig.OBS_DIM
    var obs_host = ctx.enqueue_create_host_buffer[DT](N * OBS)
    ctx.enqueue_copy(
        obs_host, DeviceBuffer[DT](ctx, env.obs_ptr(), N * OBS, owning=False)
    )
    ctx.synchronize()
    var checksum: Float64 = 0.0
    for k in range(N * OBS):
        checksum += Float64(obs_host.unsafe_ptr()[k])
    var mode = String("graph") if USE_GRAPH else String("eager")
    if USE_GRAPH and graph and graph.value().is_disabled():
        mode = String("graph-DISABLED(ran eager)")
    print(
        "RESULT side=noeira-gpu mode=" + mode,
        "model=half_cheetah n_envs=" + String(N),
        "steps=" + String(steps),
        "wall_s=" + String(dt),
        "us_per_batch_step=" + String(dt / Float64(steps) * 1e6),
        "env_steps_per_s=" + String(Int(env_sps)),
        "physics_steps_per_s=" + String(Int(env_sps * Float64(FRAME_SKIP))),
        "obs_checksum=" + String(checksum),
    )


def main() raises:
    var args = argv()
    var warmup = Int(String(args[1])) if len(args) > 1 else 100
    var steps = Int(String(args[2])) if len(args) > 2 else 1000
    var ctx = DeviceContext()
    print("device:", ctx.name(), "| warmup", warmup, "| steps", steps)
    bench[1, False](ctx, warmup, steps)
    bench[1, True](ctx, warmup, steps)
    bench[16, False](ctx, warmup, steps)
    bench[16, True](ctx, warmup, steps)
    bench[64, False](ctx, warmup, steps)
    bench[64, True](ctx, warmup, steps)
    bench[256, False](ctx, warmup, steps)
    bench[256, True](ctx, warmup, steps)
    bench[1024, False](ctx, warmup, steps)
    bench[1024, True](ctx, warmup, steps)
    bench[4096, False](ctx, warmup, steps)
    bench[4096, True](ctx, warmup, steps)
