"""PPO ON A TASK FAMILY — so101-nexus's recipe on our GPU-batched env.

    examples/tasks/ppo_tower_gpu.mojo   the so101_tower entry (project so101-tower)

    pixi run -e nvidia mojo build -I . [-D TASK_PPO_LANES_4096] \\
        examples/tasks/ppo_tower_gpu.mojo -o ppo_tower
    ./ppo_tower so101_tower_lift_brick --steps 30000000

WHY PPO, AND WHY THIS SHAPE (`noeira-docs/SO101_PIXEL_RL_PLAN.md`):
so101-nexus (references/so101-nexus-main, `examples/ppo_warp.py`) solves
PickLift on the SO-101 with CleanRL-style PPO at 1024 MuJoCo Warp worlds in
30M steps / 24.5 min (4 of 5 seeds), and PickAndPlace only with demo-seeding
plus a success bonus. Its three decisive settings are reproduced here:

- FIXED-HORIZON episodes (`TERMINATE_ON_UNHEALTHY=False`): terminating on
  success ends the reward stream, and PPO then farms the shaping instead of
  finishing.
- The CleanRL update budget: rollout 16 per env, 32 minibatches, 10 epochs,
  grad-norm clip 0.5, advantage normalisation (the trainer's own), no KL
  stop.
- A strong entropy warm start with a nonzero floor (0.03 -> 0.005, linear),
  the learning rate annealed to 0 with it, and STAGGERED resets (each lane's
  episode clock starts at a random phase).

plus its observation and reward normalisation (CleanRL's `NormalizeObservation`
/ `NormalizeReward`: running mean/std, clip ±10), done here on the host because
the on-policy trainer already stages obs through host memory.

⚠⚠ A DIVERGED LANE IS RESET AND KEPT OUT OF THE STATISTICS. The first 5090
run (lift, absolute actions, 2026-09-26) reached 72 % at 9.2M steps and fell
to 1 % by 10.5M with a HEALTHY update (KL 0.009, explained variance 0.9): a
few lanes' physics had blown up (arm qpos ~1e7, brick 1e5 m) under bang-bang
targets, and the running variance — cumulative, never forgotten — reached
1e14 on the joint words, so every healthy lane's normalised joints read ~0
and the policy lost its own arm. so101-nexus guards the same failure on
MuJoCo Warp (`_finite`). Here a lane whose raw obs has a non-finite word or
one beyond `OBS_BOUND` (or a reward beyond `REW_BOUND`) is marked done (the
env resets it), its reward is zeroed, its observation is left out of the
running statistics, and the count is logged as `diverged`.

THE REWARD is the family's potential-based mode by default
(`--reward potential`, `tasks/shaping.reward_mode_words`): the change of the
staged potential, the full budget while the goal holds, an optional one-time
`--success-bonus`. `--reward legacy` is the raw per-step terms.

THE METRIC is so101-nexus's: the success rate of the COMPLETED episodes of
the training rollout (the goal held at ANY step — `META_IDX_GOAL_HELD`), over
a window of the last `SUCCESS_WINDOW` episodes. A greedy evaluation is a
later addition.

⚠ TWO ACTION SPACES (`--action`):
- `absolute` (default): the family's own — joint targets normalised to each
  actuator's ctrlrange. A Gaussian at std 1 would throw every target across
  the whole range each step, hence `--log-std-init -1.0` (std 0.37); the first
  Metal smoke then showed a first-update KL ~1.0 with 85 % clipped, because a
  small mean shift at a small std IS a large KL.
- `delta`: so101-nexus's `pd_joint_delta_pos` — the target is the CURRENT
  joint position plus `a * DELTA_SCALE` (0.05 rad per arm joint, 0.2 for the
  gripper, per control step), clamped to the ctrlrange, then normalised into
  the env's absolute action. Anchored on the joints (read from the raw
  observation), so there is no hidden target state. Pair with
  `--log-std-init 0` (nexus's std 1).
"""

from std.math import abs, sqrt
from std.random import random_float64, seed as seed_rng
from std.sys import is_defined
from std.time import perf_counter_ns

from max.gpu.host import DeviceBuffer, DeviceContext

from noeira.core.run import RunContext, register_run
from noeira.core.run_session import RunLogger, finish_run, run_logger
from noeira.deep_agents.ppo import PPOAgent
from noeira.deep_agents.primitives.gaussian_head import GaussianHead
from noeira.envs.phyics3d_batched_env import Phyics3dBatchedEnv
from noeira.envs.phyics3d_env import Phyics3dEnvConfig
from noeira.io.artifact_sink import sink_for_run
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.constants import DT
from noeira.nn.core.ptr import mptr
from noeira.nn.primitives.activations import Tanh
from noeira.nn.primitives.linear import Linear
from noeira.physics3d.gpu.constants import (
    METADATA_SIZE, MODEL_CURRICULUM_SIZE, META_IDX_GOAL_HELD,
    META_IDX_STEP_COUNT, META_IDX_REWARD_MODE,
)
from noeira.physics3d.model import ModelDefLike
from noeira.physics3d.parser.runtime_load import parse_model_runtime
from noeira.tasks.eval import region_sites, region_rects, region_half_heights
from noeira.tasks.family import scene_path
from noeira.tasks.gpu_eval import region_table_words
from noeira.tasks.posed_reset import task_meta_words
from noeira.tasks.shaping import reward_mode_words
from noeira.tasks.spec import load_family

# ⚠ LANES AT BUILD TIME (Mojo has no integer define): 1024 by default,
# so101-nexus's count; `-D TASK_PPO_LANES_4096` for our physics' better
# throughput (22k vs 9.8k env-steps/s on the 5090), `_256` for a smoke run.
comptime N_ENVS = (
    4096 if is_defined["TASK_PPO_LANES_4096"]()
    else (256 if is_defined["TASK_PPO_LANES_256"]() else 1024)
)
comptime ROLLOUT = 16
comptime N_MINIBATCHES = 32
comptime MINIBATCH = N_ENVS * ROLLOUT // N_MINIBATCHES
comptime N_EPOCHS = 10
comptime HIDDEN = 256
comptime ACT_DIM = 6
comptime SUCCESS_WINDOW = 1024
comptime OBS_CLIP = 10.0
comptime REW_CLIP = 10.0
comptime GAMMA = 0.99
comptime OBS_BOUND = 1.0e3
"""A raw observation word beyond this (or non-finite) marks the lane DIVERGED."""
comptime REW_BOUND = 1.0e3
comptime DELTA_ARM = 0.05
comptime DELTA_GRIPPER = 0.2

comptime EnvT[M: ModelDefLike, C: Phyics3dEnvConfig] = Phyics3dBatchedEnv[
    M, C, N_ENVS, TERMINATE_ON_UNHEALTHY=False,
]
comptime OBS_DIM[M: ModelDefLike, C: Phyics3dEnvConfig] = EnvT[M, C].OBS_DIM

comptime ActorNet[OBS: Int] = Sequential[
    Linear[OBS, HIDDEN], Tanh[HIDDEN],
    Linear[HIDDEN, HIDDEN], Tanh[HIDDEN],
    GaussianHead[HIDDEN, ACT_DIM],
]
comptime CriticNet[OBS: Int] = Sequential[
    Linear[OBS, HIDDEN], Tanh[HIDDEN],
    Linear[HIDDEN, HIDDEN], Tanh[HIDDEN],
    Linear[HIDDEN, 1],
]
comptime AgentT[OBS: Int] = PPOAgent[
    "gpu", ActorNet[OBS], CriticNet[OBS], OBS, ACT_DIM, ROLLOUT, MINIBATCH,
    N_EPOCHS, N_ENVS,
]


struct RunningMeanStd(Movable):
    """CleanRL's running mean / variance (parallel Welford over batches)."""

    var mean: List[Float64]
    var var_: List[Float64]
    var count: Float64

    def __init__(out self, dim: Int):
        self.mean = List[Float64](length=dim, fill=0.0)
        self.var_ = List[Float64](length=dim, fill=1.0)
        self.count = 1e-4

    def update(
        mut self, x: Pointer[Scalar[DT], MutAnyOrigin], n_rows: Int, dim: Int,
        skip: List[Bool] = List[Bool](),
    ):
        """Rows with `skip[i]` set are left out (diverged lanes)."""
        var bm = List[Float64](length=dim, fill=0.0)
        var bv = List[Float64](length=dim, fill=0.0)
        var n = 0
        for i in range(n_rows):
            if len(skip) > 0 and skip[i]:
                continue
            n += 1
            for k in range(dim):
                bm[k] += Float64(x[unsafe_offset = i * dim + k])
        if n == 0:
            return
        for k in range(dim):
            bm[k] /= Float64(n)
        for i in range(n_rows):
            if len(skip) > 0 and skip[i]:
                continue
            for k in range(dim):
                var d = Float64(x[unsafe_offset = i * dim + k]) - bm[k]
                bv[k] += d * d
        for k in range(dim):
            bv[k] /= Float64(n)
        var tot = self.count + Float64(n)
        for k in range(dim):
            var delta = bm[k] - self.mean[k]
            var m_a = self.var_[k] * self.count
            var m_b = bv[k] * Float64(n)
            var m2 = m_a + m_b + delta * delta * self.count * Float64(n) / tot
            self.mean[k] += delta * Float64(n) / tot
            self.var_[k] = m2 / tot
        self.count = tot

    def normalize_into(
        self,
        src: Pointer[Scalar[DT], MutAnyOrigin],
        dst: Pointer[Scalar[DT], MutAnyOrigin],
        n: Int,
        dim: Int,
        clip: Float64,
    ):
        for i in range(n):
            for k in range(dim):
                var v = (Float64(src[unsafe_offset = i * dim + k]) - self.mean[k]) / sqrt(
                    self.var_[k] + 1e-8
                )
                if v > clip:
                    v = clip
                elif v < -clip:
                    v = -clip
                dst[unsafe_offset = i * dim + k] = Scalar[DT](v)

    def save(self, path: String) raises:
        var s = String("count ") + String(self.count) + "\n"
        s += "mean"
        for k in range(len(self.mean)):
            s += " " + String(self.mean[k])
        s += "\nvar"
        for k in range(len(self.var_)):
            s += " " + String(self.var_[k])
        s += "\n"
        with open(path, "w") as f:
            f.write(s)


def _arg(args: List[String], key: String, default: String) raises -> String:
    for i in range(len(args) - 1):
        if args[i] == key:
            return args[i + 1]
    return default


def run_ppo[M: ModelDefLike, C: Phyics3dEnvConfig](
    args: List[String],
    family_path: String,
    family: String,
    project: String,
    driver: String,
    default_task: String,
    shape_w_goal: Float64,
    shape_w_reach: Float64,
    goal_margin: Float64,
    reach_margin: Float64,
) raises:
    comptime OBS = OBS_DIM[M, C]
    comptime assert N_ENVS * ROLLOUT % N_MINIBATCHES == 0

    # ── flags ────────────────────────────────────────────────────────────
    var task = default_task
    if len(args) > 1 and not args[1].startswith("--"):
        task = args[1]
    var total_steps = Int(_arg(args, "--steps", "30000000"))
    var seed = Int(_arg(args, "--seed", "1"))
    var lr0 = Float64(_arg(args, "--lr", "0.0003"))
    var ent0 = Float64(_arg(args, "--ent-coef", "0.03"))
    var ent1 = Float64(_arg(args, "--ent-coef-final", "0.005"))
    var log_std0 = Float64(_arg(args, "--log-std-init", "-1.0"))
    var reward = _arg(args, "--reward", "potential")
    var bonus = Float64(_arg(args, "--success-bonus", "0"))
    var ckpt_every = Int(_arg(args, "--checkpoint-every", "5000000"))
    var anneal_steps = Int(_arg(args, "--anneal-steps", String(total_steps)))
    var action_mode = _arg(args, "--action", "absolute")
    if action_mode != "absolute" and action_mode != "delta":
        raise Error("ppo task: --action absolute|delta, got " + action_mode)
    if reward != "potential" and reward != "legacy":
        raise Error("ppo task: --reward potential|legacy, got " + reward)
    var rw = reward_mode_words(reward == "potential", bonus)
    seed_rng(seed)

    var n_updates_total = anneal_steps // (N_ENVS * ROLLOUT)
    print("=" * 70)
    print("PPO on", family, "—", task)
    print("  lanes", N_ENVS, "| rollout", ROLLOUT, "| batch", N_ENVS * ROLLOUT,
          "| minibatch", MINIBATCH, "x", N_MINIBATCHES, "| epochs", N_EPOCHS)
    print("  obs", OBS, "| act", ACT_DIM, "| hidden", HIDDEN, "| horizon",
          C.MAX_STEPS)
    print("  steps", total_steps, "| lr", lr0, "| ent", ent0, "->", ent1,
          "over", anneal_steps, "steps | log_std init", log_std0)
    print("  reward", reward, "| success bonus", bonus, "| seed", seed,
          "| action", action_mode)
    print("=" * 70)

    # ── the task's words (the eval's / the bench's set-up) ───────────────
    var f = load_family(family_path)
    var fmd = parse_model_runtime(scene_path(f))
    var rsites = region_sites(f, fmd.site_names)
    var rects = region_rects(f)
    var rheights = region_half_heights(f)
    var cw = region_table_words(
        rsites[0], rects[0][0], rects[0][1], rects[0][2], rects[0][3],
        rheights[0],
    )
    var mw = task_meta_words(
        task, family, shape_w_goal, shape_w_reach, goal_margin, reach_margin,
    )
    # The actuators' joints (qpos addresses) and ctrl ranges, for `--action
    # delta`: target = clamp(q + a * scale), normalised as the env expects.
    var jadr = List[Int]()
    var acc = 0
    for i in range(len(fmd.joints)):
        jadr.append(acc)
        acc += fmd.joints[i].nq
    var a_qa = List[Int]()
    var a_lo = List[Float64]()
    var a_hi = List[Float64]()
    for i in range(ACT_DIM):
        a_qa.append(jadr[fmd.actuators[i].joint_id])
        a_lo.append(fmd.actuators[i].ctrl_min)
        a_hi.append(fmd.actuators[i].ctrl_max)

    var run = RunContext(
        project=project, driver=driver, slug=String("ppo-") + task,
        env=String("family:") + family, task=task, seed=seed, device="gpu",
    )
    print("  run", run.dir)
    var logger = run_logger(run, buffer_size=64)
    logger.set_config("algorithm", "PPO")
    logger.set_config("family", family)
    logger.set_config("task", task)
    logger.set_config("n_envs", String(N_ENVS))
    logger.set_config("rollout", String(ROLLOUT))
    logger.set_config("minibatch", String(MINIBATCH))
    logger.set_config("epochs", String(N_EPOCHS))
    logger.set_config("hidden", String(HIDDEN))
    logger.set_config("lr", String(lr0))
    logger.set_config("ent_coef", String(ent0))
    logger.set_config("ent_coef_final", String(ent1))
    logger.set_config("log_std_init", String(log_std0))
    logger.set_config("reward", reward)
    logger.set_config("action", action_mode)
    logger.set_config("success_bonus", String(bonus))
    logger.set_config("horizon", String(C.MAX_STEPS))
    logger.set_config("obs_norm", "running, clip 10")
    logger.set_config("reward_norm", "discounted-return std, clip 10")
    register_run(run, logger)
    var artifacts = sink_for_run(run.id, run.dir)
    var logger_ptr = Pointer(to=logger).as_unsafe_any_origin()

    with DeviceContext() as ctx:
        var agent = AgentT[OBS](
            ctx=ctx,
            actor_lr=Scalar[DT](lr0),
            critic_lr=Scalar[DT](lr0),
            gamma=Scalar[DT](GAMMA),
            gae_lambda=Scalar[DT](0.95),
            clip_eps=Scalar[DT](0.2),
            entropy_coef=Scalar[DT](ent0),
            action_scale=Scalar[DT](1.0),
            log_std_init=Scalar[DT](log_std0),
            window_size=100,
            initial_episode_fill=Scalar[DT](0.0),
            max_grad_norm=Scalar[DT](0.5),
        )
        agent.trainer.actor.children[4].set_log_std_init["gpu"](
            Scalar[DT](log_std0), ctx
        )
        var env = EnvT[M, C](ctx)
        for i in range(MODEL_CURRICULUM_SIZE):
            env.mf.curriculum.data[i] = Scalar[DT](cw[i])
        env.mf.curriculum.upload(ctx)
        for e in range(N_ENVS):
            var mb = e * METADATA_SIZE
            for k in range(METADATA_SIZE):
                env.d.meta.data[mb + k] = Scalar[DT](0)
            for k in range(len(mw[0])):
                env.d.meta.data[mb + mw[0][k]] = Scalar[DT](mw[1][k])
            for k in range(len(rw)):
                env.d.meta.data[mb + META_IDX_REWARD_MODE + k] = Scalar[DT](rw[k])
        env.d.meta.upload(ctx)
        ctx.synchronize()
        env.reset_batch[N_ENVS](ctx=ctx, rng_seed=UInt64(seed))
        ctx.synchronize()
        # Staggered resets: each lane's episode clock at a random phase, so
        # the batch does not truncate in lockstep (so101-nexus
        # `--stagger-resets`).
        env.d.meta.download(ctx)
        ctx.synchronize()
        for e in range(N_ENVS):
            env.d.meta.data[e * METADATA_SIZE + META_IDX_STEP_COUNT] = Scalar[DT](
                Int(random_float64() * Float64(C.MAX_STEPS))
            )
        env.d.meta.upload(ctx)
        ctx.synchronize()

        # ── host scratch ─────────────────────────────────────────────────
        var raw_h = ctx.enqueue_create_host_buffer[DT](N_ENVS * OBS)
        var cur_n = ctx.enqueue_create_host_buffer[DT](N_ENVS * OBS)
        var next_n = ctx.enqueue_create_host_buffer[DT](N_ENVS * OBS)
        var act_h = ctx.enqueue_create_host_buffer[DT](N_ENVS * ACT_DIM)
        var rew_h = ctx.enqueue_create_host_buffer[DT](N_ENVS)
        var rew_n = ctx.enqueue_create_host_buffer[DT](N_ENVS)
        var done_h = ctx.enqueue_create_host_buffer[DT](N_ENVS)
        var rets = ctx.enqueue_create_host_buffer[DT](N_ENVS)
        var env_act = ctx.enqueue_create_host_buffer[DT](N_ENVS * ACT_DIM)
        var arm_q = List[Float64](length=N_ENVS * ACT_DIM, fill=0.0)
        ctx.synchronize()
        var obs_dev = DeviceBuffer[DT](ctx, env.obs_ptr(), N_ENVS * OBS, owning=False)
        var act_dev = DeviceBuffer[DT](ctx, env.action_ptr(), N_ENVS * ACT_DIM, owning=False)
        var rew_dev = DeviceBuffer[DT](ctx, env.reward_ptr(), N_ENVS, owning=False)
        var done_dev = DeviceBuffer[DT](ctx, env.done_ptr(), N_ENVS, owning=False)

        var obs_rms = RunningMeanStd(OBS)
        var ret_rms = RunningMeanStd(1)
        var ret_acc = List[Float64](length=N_ENVS, fill=0.0)
        var raw_ret = List[Float64](length=N_ENVS, fill=0.0)
        var succ = List[Bool](length=N_ENVS, fill=False)
        var hist_succ = List[Bool]()
        var hist_ret = List[Float64]()
        var n_episodes = 0
        var diverged = List[Bool](length=N_ENVS, fill=False)
        var n_diverged = 0

        # the first observation
        ctx.enqueue_copy(raw_h, obs_dev)
        ctx.synchronize()
        var rp = mptr(raw_h.unsafe_ptr())
        # ⚠ THE DELTA MODE READS THE JOINTS OUT OF THE RAW OBSERVATION at the
        # actuators' qpos addresses (the family's obs starts with qpos). Proved
        # here against the env's own qpos rather than assumed.
        env.d.qpos.download(ctx)
        ctx.synchronize()
        comptime NQ_M = M.NQ
        for e in range(N_ENVS):
            for j in range(ACT_DIM):
                var a = Float64(rp[unsafe_offset = e * OBS + a_qa[j]])
                var b = Float64(env.d.qpos.data[e * NQ_M + a_qa[j]])
                if abs(a - b) > 1e-5:
                    raise Error(
                        "ppo task: obs[" + String(a_qa[j]) + "] of lane "
                        + String(e) + " is " + String(a) + ", qpos is "
                        + String(b) + " — the observation does not start with"
                        " qpos, and --action delta would anchor on garbage"
                    )
        for e in range(N_ENVS):
            for j in range(ACT_DIM):
                arm_q[e * ACT_DIM + j] = Float64(rp[unsafe_offset = e * OBS + a_qa[j]])
        obs_rms.update(rp, N_ENVS, OBS)
        obs_rms.normalize_into(rp, mptr(cur_n.unsafe_ptr()), N_ENVS, OBS, OBS_CLIP)

        var step = 0
        var it = 0
        var n_updates = 0
        var t0 = perf_counter_ns()
        var next_ckpt = ckpt_every
        var ckpt_path = run.checkpoint_path(String("last"))
        while step < total_steps:
            # 1. act on the normalised observation
            agent.trainer.select_action_batched(
                mptr(cur_n.unsafe_ptr()), mptr(act_h.unsafe_ptr()), step,
            )
            if action_mode == "delta":
                var ap = mptr(act_h.unsafe_ptr())
                var ep = mptr(env_act.unsafe_ptr())
                for e in range(N_ENVS):
                    for j in range(ACT_DIM):
                        var sc = DELTA_GRIPPER if j == ACT_DIM - 1 else DELTA_ARM
                        var tgt = arm_q[e * ACT_DIM + j] + Float64(
                            ap[unsafe_offset = e * ACT_DIM + j]
                        ) * sc
                        if tgt < a_lo[j]:
                            tgt = a_lo[j]
                        elif tgt > a_hi[j]:
                            tgt = a_hi[j]
                        var mid = 0.5 * (a_lo[j] + a_hi[j])
                        var half = 0.5 * (a_hi[j] - a_lo[j])
                        ep[unsafe_offset = e * ACT_DIM + j] = Scalar[DT](
                            (tgt - mid) / half
                        )
                ctx.enqueue_copy(act_dev, env_act)
            else:
                ctx.enqueue_copy(act_dev, act_h)
            # 2. step; the post-step obs, reward, done, meta (goal bit)
            env.step_batch[N_ENVS](ctx=ctx, rng_seed=UInt64(it + 1))
            ctx.enqueue_copy(raw_h, obs_dev)
            ctx.enqueue_copy(rew_h, rew_dev)
            ctx.enqueue_copy(done_h, done_dev)
            env.d.meta.download(ctx)
            ctx.synchronize()
            var raw_p = mptr(raw_h.unsafe_ptr())
            var dh0 = mptr(done_h.unsafe_ptr())
            var rh0 = mptr(rew_h.unsafe_ptr())
            var n_bad = 0
            for e in range(N_ENVS):
                var bad = False
                var rv = Float64(rh0[unsafe_offset=e])
                if not (rv == rv) or abs(rv) > REW_BOUND:
                    bad = True
                for k in range(OBS):
                    var v = Float64(raw_p[unsafe_offset = e * OBS + k])
                    if not (v == v) or abs(v) > OBS_BOUND:
                        bad = True
                        break
                diverged[e] = bad
                if bad:
                    n_bad += 1
                    dh0[unsafe_offset=e] = Scalar[DT](1)
                    rh0[unsafe_offset=e] = Scalar[DT](0)
            n_diverged += n_bad
            obs_rms.update(raw_p, N_ENVS, OBS, diverged)
            obs_rms.normalize_into(
                raw_p, mptr(next_n.unsafe_ptr()), N_ENVS, OBS, OBS_CLIP,
            )
            if n_bad > 0:
                # the terminal obs of a diverged lane is garbage: zero it
                var nn = mptr(next_n.unsafe_ptr())
                for e in range(N_ENVS):
                    if diverged[e]:
                        for k in range(OBS):
                            nn[unsafe_offset = e * OBS + k] = Scalar[DT](0)
                # the env resets on its OWN done buffer: write the forced
                # dones back before `selective_reset_batch`
                ctx.enqueue_copy(done_dev, done_h)
            # 3. reward normalisation (CleanRL NormalizeReward) + tallies
            var rh = mptr(rew_h.unsafe_ptr())
            var dh = mptr(done_h.unsafe_ptr())
            var rt = mptr(rets.unsafe_ptr())
            for e in range(N_ENVS):
                var r = Float64(rh[unsafe_offset=e])
                ret_acc[e] = ret_acc[e] * GAMMA + r
                rt[unsafe_offset=e] = Scalar[DT](ret_acc[e])
                raw_ret[e] += r
                if env.d.meta.data[e * METADATA_SIZE + META_IDX_GOAL_HELD] > Scalar[DT](0.5):
                    succ[e] = True
            ret_rms.update(rt, N_ENVS, 1)
            var rscale = 1.0 / sqrt(ret_rms.var_[0] + 1e-8)
            var rn = mptr(rew_n.unsafe_ptr())
            for e in range(N_ENVS):
                var v = Float64(rh[unsafe_offset=e]) * rscale
                if v > REW_CLIP:
                    v = REW_CLIP
                elif v < -REW_CLIP:
                    v = -REW_CLIP
                rn[unsafe_offset=e] = Scalar[DT](v)
                if dh[unsafe_offset=e] > Scalar[DT](0.5):
                    hist_succ.append(succ[e])
                    hist_ret.append(raw_ret[e])
                    n_episodes += 1
                    succ[e] = False
                    raw_ret[e] = 0.0
                    ret_acc[e] = 0.0
            # 4. record (normalised obs / reward); done cuts GAE
            agent.trainer.record_batch_cpu(
                mptr(cur_n.unsafe_ptr()), mptr(rew_n.unsafe_ptr()),
                mptr(next_n.unsafe_ptr()), mptr(done_h.unsafe_ptr()),
            )
            # 5. reset the finished lanes; the obs they restart from
            env.selective_reset_batch[N_ENVS](
                ctx=ctx, rng_seed=UInt64(seed * 7919 + it + 1)
            )
            ctx.enqueue_copy(raw_h, obs_dev)
            ctx.synchronize()
            var rp2 = mptr(raw_h.unsafe_ptr())
            for e in range(N_ENVS):
                for j in range(ACT_DIM):
                    arm_q[e * ACT_DIM + j] = Float64(
                        rp2[unsafe_offset = e * OBS + a_qa[j]]
                    )
            obs_rms.normalize_into(
                rp2, mptr(cur_n.unsafe_ptr()), N_ENVS, OBS, OBS_CLIP,
            )
            step += N_ENVS
            it += 1
            # 6. the update at the rollout boundary, then the schedules
            if agent.trainer.train_step(step):
                n_updates += 1
                var frac = 1.0 - Float64(n_updates) / Float64(max(n_updates_total, 1))
                if frac < 0.0:
                    frac = 0.0
                agent.trainer.actor_opt.set_lr(Scalar[DT](lr0 * frac))
                agent.trainer.critic_opt.set_lr(Scalar[DT](lr0 * frac))
                agent.trainer.actor_train.set_entropy_coef(
                    Scalar[DT](ent1 + (ent0 - ent1) * frac)
                )
                # the window's success rate and return, per update
                var n = len(hist_succ)
                var lo = n - SUCCESS_WINDOW if n > SUCCESS_WINDOW else 0
                var ns = 0
                var rsum = 0.0
                for k in range(lo, n):
                    if hist_succ[k]:
                        ns += 1
                    rsum += hist_ret[k]
                var nw = n - lo
                var rate = Float64(ns) / Float64(nw) if nw > 0 else 0.0
                var mret = rsum / Float64(nw) if nw > 0 else 0.0
                var secs = Float64(perf_counter_ns() - t0) / 1e9
                logger.log_scalar("success_rate", rate, step)
                logger.log_scalar("episode_return", mret, step)
                logger.log_scalar("episodes", Float64(n_episodes), step)
                logger.log_scalar("diverged", Float64(n_diverged), step)
                logger.log_scalar("sps", Float64(step) / secs, step)
                logger.log_scalar("lr", lr0 * frac, step)
                logger.log_scalar("ent_coef", ent1 + (ent0 - ent1) * frac, step)
                agent.trainer.flush_metrics_through_logger[RunLogger](
                    logger_ptr, step
                )
                if n_updates % 10 == 0:
                    print("  step", step, "| success", rate, "over", nw,
                          "ep | return", mret, "| episodes", n_episodes,
                          "| diverged", n_diverged,
                          "|", Int(Float64(step) / secs), "steps/s")
            if step >= next_ckpt:
                agent.trainer.save_state(ckpt_path)
                obs_rms.save(run.dir + "/obs_norm.txt")
                next_ckpt += ckpt_every
        agent.trainer.save_state(ckpt_path)
        obs_rms.save(run.dir + "/obs_norm.txt")
        var n = len(hist_succ)
        var lo = n - SUCCESS_WINDOW if n > SUCCESS_WINDOW else 0
        var ns = 0
        for k in range(lo, n):
            if hist_succ[k]:
                ns += 1
        var rate = Float64(ns) / Float64(n - lo) if n - lo > 0 else 0.0
        var secs = Float64(perf_counter_ns() - t0) / 1e9
        print("=" * 70)
        print("  done:", step, "steps in", secs, "s |", n_episodes,
              "episodes | final success", rate, "over", n - lo)
        print("  checkpoint", ckpt_path)
        print("=" * 70)
        finish_run(run, logger, artifacts,
                   String("success_rate=") + String(rate))
        _ = logger
