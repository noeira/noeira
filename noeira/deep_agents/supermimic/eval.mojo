"""SuperMimic's benchmark: the paper's Appendix F protocol, on the device env.

The reproduction's `eval/benchmark.py`, the protocol its headline numbers
use (BONES-SEED full set: 0.930 normalised return, 145 / 129,491 failures):

    every clip played once from its first frame, deterministic policy, no
    observation / action / reset noise, no domain randomisation (foot
    friction 1), no episode cap (the clip's end decides); a clip FAILS when
    the training termination fires; normalised return = Σ step rewards /
    (0.1 · steps), 1.0 is perfect tracking (0.1 = the per-step maximum).

Clips are queued per lane: lane l plays clips l, l + N, l + 2N, … (the
reproduction's `forced_frame` queue), each starting at its clip's first
frame and, as in training, first targeting frame + 1.

`summarize` gives the mean and step-weighted normalised return, the failure
count and the per-step mean of each weighted reward term (max 0.02 for the
four body terms, 0.01 for the two root terms).
"""

from max.gpu.host import DeviceContext

from noeira.deep_agents.supermimic.config import SMConfigT
from noeira.deep_agents.supermimic.env_gpu import ST_ENDING, ST_TERMINATED, SMReplayGpuOf, SMTrackingEnvGpu
from noeira.deep_agents.supermimic.heads import SM_N_TERMS
from noeira.deep_agents.supermimic.lanes import talloc
from noeira.deep_agents.supermimic.motion_library import SMMotionLibrary
from noeira.deep_agents.supermimic.robot import SMDims, TrackingBatchedSim, TrackingRobot
from noeira.deep_agents.supermimic.trainer import SuperMimicTrainer
from noeira.nn.constants import DT


comptime SM_MAX_STEP_REWARD: Float64 = 0.1       # dt · Σ weights of the positive terms


def sm_term_name(k: Int) -> String:
    """`heads.T_*` order."""
    var names: List[String] = [
        "root_pos", "root_ori", "body_pos", "body_ori", "body_lin_vel", "body_ang_vel", "action_rate", "joint_limit",
    ]
    return names[k]


@fieldwise_init
struct SMClipResult(Copyable, Movable):
    var clip: Int
    var frames: Int
    var steps: Int
    var norm_return: Float64
    var failed: Bool
    var terms: List[Float64]          # per-step mean of each weighted term


@fieldwise_init
struct SMBenchmarkSummary(Copyable, Movable):
    var clips: Int
    var failed: Int
    var mean_norm_return: Float64
    var step_weighted_norm_return: Float64
    var terms: List[Float64]          # step-weighted per-step means


def sm_summarize(results: List[SMClipResult]) -> SMBenchmarkSummary:
    var n = len(results)
    var failed = 0
    var mean = 0.0
    var w = 0.0
    var steps = 0
    var terms = List[Float64](length=SM_N_TERMS, fill=0.0)
    for i in range(n):
        failed += 1 if results[i].failed else 0
        mean += results[i].norm_return
        w += results[i].norm_return * Float64(results[i].steps)
        steps += results[i].steps
        for k in range(SM_N_TERMS):
            terms[k] += results[i].terms[k] * Float64(results[i].steps)
    for k in range(SM_N_TERMS):
        terms[k] /= Float64(max(steps, 1))
    return SMBenchmarkSummary(n, failed, mean / Float64(max(n, 1)), w / Float64(max(steps, 1)), terms^)


def sm_benchmark_gpu[C: SMConfigT, BS: TrackingBatchedSim](
    ctx: DeviceContext, mut trainer: SuperMimicTrainer[C], var lib: SMMotionLibrary[C.ROBOT], var sim: BS,
    max_clips: Int = 0, verbose: Bool = True,
) raises -> List[SMClipResult]:
    """Every clip of `lib` (the first `max_clips` if > 0), the protocol above,
    with the trainer's policy over the device env's lanes."""
    var n_clips = lib.num_clips if max_clips <= 0 else min(max_clips, lib.num_clips)
    var env = sm_benchmark_env[C.ROBOT, BS](ctx, lib^, sim^)
    var rb = SMReplayGpuOf[C.ROBOT](ctx, BS.N, 4)          # the env writes rows; nobody samples them
    return sm_benchmark_run[C, BS](ctx, trainer, env, rb, n_clips, verbose)


def sm_benchmark_env[R: TrackingRobot, BS: TrackingBatchedSim](
    ctx: DeviceContext, var lib: SMMotionLibrary[R], var sim: BS,
) raises -> SMTrackingEnvGpu[R, BS]:
    """The protocol's env: no noise, no domain randomisation, no cap."""
    return SMTrackingEnvGpu[R, BS](
        ctx, lib^, sim^, dr=False, obs_noise=False, reset_noise=False, action_noise=0.0, max_episode_steps=0,
    )


def sm_benchmark_run[C: SMConfigT, BS: TrackingBatchedSim](
    ctx: DeviceContext, mut trainer: SuperMimicTrainer[C], mut env: SMTrackingEnvGpu[C.ROBOT, BS],
    mut rb: SMReplayGpuOf[C.ROBOT], n_clips: Int, verbose: Bool = True,
) raises -> List[SMClipResult]:
    """The first `n_clips` clips of `env`'s library, the protocol above, over
    an env from `sm_benchmark_env`. Re-runnable: it restarts every lane."""
    comptime R = C.ROBOT
    comptime N = BS.N          # any lane count: the policy acts on N lanes (training's C.LANES need not match)
    var current = List[Int](length=N, fill=0)
    var nxt = List[Int](length=N, fill=0)
    var active = List[Bool](length=N, fill=False)
    var frames0 = List[Int](length=N, fill=0)
    for l in range(N):
        current[l] = l
        nxt[l] = l + N
        active[l] = l < n_clips
        frames0[l] = env.lib.clip_start[min(l, n_clips - 1)]
    env.restart(ctx, frames0)
    for l in range(N):
        env.next_frame[l] = env.lib.clip_start[min(nxt[l], n_clips - 1)]
    var ret = List[Float64](length=N, fill=0.0)
    var steps = List[Int](length=N, fill=0)
    var tsum = List[Float64](length=N * SM_N_TERMS, fill=0.0)
    var results = List[SMClipResult]()
    var goals = talloc["gpu"](N * SMDims[R].GOAL, Optional(ctx))
    var remaining = n_clips
    var next_report = 1000
    while remaining > 0:
        env.policy_inputs(ctx, goals)
        var a = trainer.act[N](env.hist, goals, resident=True)
        _ = env.step(ctx, a, rb)
        env.out_terms.download(ctx)
        for l in range(N):
            if not active[l]:
                continue
            ret[l] += env.last_reward[l]
            steps[l] += 1
            for k in range(SM_N_TERMS):
                tsum[l * SM_N_TERMS + k] += Float64(env.out_terms.data[l * SM_N_TERMS + k])
            var st = env.h_out_i[2 * l + 1]
            if (st & ST_ENDING) == 0:
                continue
            var c = current[l]
            var terms = List[Float64](length=SM_N_TERMS, fill=0.0)
            for k in range(SM_N_TERMS):
                terms[k] = tsum[l * SM_N_TERMS + k] / Float64(max(steps[l], 1))
            results.append(SMClipResult(
                c, env.lib.clip_len[c], steps[l], ret[l] / (SM_MAX_STEP_REWARD * Float64(max(steps[l], 1))),
                (st & ST_TERMINATED) != 0, terms^,
            ))
            remaining -= 1
            # the lane was just reset to its queued clip: advance the queue
            current[l] = nxt[l]
            nxt[l] += N
            active[l] = current[l] < n_clips
            env.next_frame[l] = env.lib.clip_start[min(nxt[l], n_clips - 1)]
            ret[l] = 0.0
            steps[l] = 0
            for k in range(SM_N_TERMS):
                tsum[l * SM_N_TERMS + k] = 0.0
        if verbose and len(results) >= next_report:
            print("  evaluated", len(results), "/", n_clips)
            next_report += 1000
    return results^


struct SMInRunBench[C: SMConfigT, BS: TrackingBatchedSim](Movable):
    """The protocol on a FIXED clip set during training, every `every`
    iterations: an unbiased learning curve beside `avg_reward`, whose
    episodes the reset sampler chooses. Its own env (a second simulator over
    its own small library), so training's lanes, replay and domain
    randomisation are untouched. Logged as `bench_norm_return` (mean) and
    `bench_failed` (fraction of clips). Its simulator may have fewer lanes
    than training's (the policy acts on any lane count): the 5090 has ~2.7 GB
    left beside a paper-width run on the full fp32 library."""
    var env: SMTrackingEnvGpu[Self.C.ROBOT, Self.BS]
    var rb: SMReplayGpuOf[Self.C.ROBOT]
    var every: Int

    def __init__(out self, ctx: DeviceContext, var lib: SMMotionLibrary[Self.C.ROBOT], var sim: Self.BS, every: Int) raises:
        self.env = sm_benchmark_env[Self.C.ROBOT, Self.BS](ctx, lib^, sim^)
        self.rb = SMReplayGpuOf[Self.C.ROBOT](ctx, Self.BS.N, 4)
        self.every = every

    def __init__(out self, *, deinit move: Self):
        self.env = move.env^
        self.rb = move.rb^
        self.every = move.every

    def due(self, it: Int, total: Int) -> Bool:
        return self.every > 0 and (it % self.every == 0 or it == total)

    def run(mut self, ctx: DeviceContext, mut trainer: SuperMimicTrainer[Self.C]) raises -> SMBenchmarkSummary:
        var res = sm_benchmark_run[Self.C, Self.BS](ctx, trainer, self.env, self.rb, self.env.lib.num_clips, verbose=False)
        return sm_summarize(res)
