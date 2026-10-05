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
- `target`: SimToolReal's arm rule — the same step added to the lane's
  PREVIOUS TARGET, not to the measured joint (`delta_action.target_step`,
  why in its module header). The target is hidden state the policy
  integrates, so the build needs `-D TASK_PPO_TARGET_OBS` (the policy sees
  `target - q`). `--target-lead L` bounds the target to L rad from `q`.
"""

from std.builtin.sort import sort
from std.math import abs, sqrt, log, cos
from std.random import random_float64, seed as seed_rng
from std.sys import is_defined
from std.time import perf_counter_ns

from max.gpu.host import DeviceBuffer, DeviceContext

from noeira.core.run import RunContext, register_run
from noeira.core.run_session import RunLogger, finish_run, run_logger
from noeira.deep_agents.ppo import PPOAgent
from noeira.deep_agents.training.driver_onpolicy import onpolicy_update_device
from noeira.cuda import CUDAGraph
# re-exported for the family's other readers (pixel DAgger, the SO-101 probes)
from noeira.deep_agents.training.obs_norm import RunningMeanStd
from noeira.tasks.ppo_family_device import (
    FamilyDeviceRollout, FamilyStepConfig,
)
from noeira.deep_agents.primitives.gaussian_head import GaussianHead
from noeira.envs.phyics3d_batched_env import Phyics3dBatchedEnv
from noeira.envs.phyics3d_env import Phyics3dEnvConfig
from noeira.io.artifact_sink import sink_for_run
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.constants import DT
from noeira.nn.core.ptr import mptr
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.call import call_forward, call_vjp
from noeira.deep_agents.demos.file import read_demo_file
from std.random import random_ui64
from noeira.nn.primitives.activations import Tanh
from noeira.nn.primitives.linear import Linear
from noeira.physics3d.gpu.constants import (
    METADATA_SIZE, MODEL_CURRICULUM_SIZE, META_IDX_GOAL_HELD,
    META_IDX_STEP_COUNT, META_IDX_REWARD_MODE, META_IDX_TASK_PARAM_0,
)
from noeira.physics3d.model import ModelDefLike
from noeira.physics3d.parser.runtime_load import parse_model_runtime
from noeira.tasks.eval import region_sites, region_rects, region_half_heights
from noeira.tasks.family import scene_path
from noeira.tasks.gpu_eval import region_table_words
from noeira.tasks.posed_reset import task_meta_words
from noeira.tasks.shaping import reward_mode_words
from noeira.tasks.delta_action import (
    DELTA_ACT, DELTA_ARM, DELTA_GRIPPER, delta_scale, delta_target, ServoLag,
    ACT_HIST, TARGET_OBS, target_step,
)
# `ACT_HIST` (`-D TASK_PPO_ACT_HIST=K`): the last K executed actions after the
# env's observation — see `delta_action.ACT_HIST`. A run trained without it is
# widened for `--init` by `tools/tasks/pad_ppo_obs.py` (zero rows: the same
# policy to start).
from noeira.tasks.spec import load_family

# ⚠ LANES AT BUILD TIME (Mojo has no integer define): 1024 by default,
# so101-nexus's count; `-D TASK_PPO_LANES_4096` for our physics' better
# throughput (22k vs 9.8k env-steps/s on the 5090), `_256` for a smoke run.
comptime N_ENVS = (
    4096 if is_defined["TASK_PPO_LANES_4096"]()
    else (256 if is_defined["TASK_PPO_LANES_256"]() else 1024)
)
# `-D TASK_PPO_TRAIN_GRAPH`: the K-epoch update through the trainer's device
# path, one minibatch step captured into a CUDA graph and replayed
# (`onpolicy_update_device`) — bit-identical to the eager update
# (`tests/nn/test_ppo_train_graph_parity.mojo`). The lr / entropy schedules
# change after every update and are kernel arguments, so the graph is dropped
# after them and re-captured at the next update. Run through `pixi run`.
comptime TRAIN_GRAPH = is_defined["TASK_PPO_TRAIN_GRAPH"]()
# `-D TASK_PPO_DEVICE_ROLLOUT`: the rollout on the GPU, every per-step option
# included (`ppo_family_device.FamilyDeviceRollout`): no host round trip per
# control step; episode records and counters read back at update cadence.
# Not bit-identical to the host loop (device RNG, Float32 servo model and
# statistics). `-D TASK_PPO_ENV_GRAPH` (with it) captures the env step and
# reset. The greedy evaluation stays on the host.
comptime DEVICE_ROLLOUT = is_defined["TASK_PPO_DEVICE_ROLLOUT"]()
comptime ENV_GRAPH = is_defined["TASK_PPO_ENV_GRAPH"]()
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


def _delta_to_env(
    ap: Pointer[Scalar[DT], MutAnyOrigin],
    ep: Pointer[Scalar[DT], MutAnyOrigin],
    ref arm_q: List[Float64],
    ref a_lo: List[Float64],
    ref a_hi: List[Float64],
    mut lag: ServoLag,
    d_arm: Float64 = DELTA_ARM,
    d_grip: Float64 = DELTA_GRIPPER,
):
    """`--action delta`: target = clamp(q + a * scale), through the servo
    model (`delta_action.ServoLag`; the identity when off), normalised onto
    the env's absolute action (`(target - mid) / half`). One tick: the
    targets of `_delta_targets`, then `_targets_to_env`."""
    var tg = List[Float64](length=N_ENVS * ACT_DIM, fill=0.0)
    _delta_targets(ap, tg, arm_q, a_lo, a_hi, d_arm, d_grip)
    _targets_to_env(tg, ep, a_lo, a_hi, lag)


def _delta_targets(
    ap: Pointer[Scalar[DT], MutAnyOrigin],
    mut tg: List[Float64],
    ref arm_q: List[Float64],
    ref a_lo: List[Float64],
    ref a_hi: List[Float64],
    d_arm: Float64 = DELTA_ARM,
    d_grip: Float64 = DELTA_GRIPPER,
):
    """The policy step's joint targets, clamp(q + a * scale), per lane —
    held for the `--repeat` ticks the action lasts."""
    for e in range(N_ENVS):
        for j in range(ACT_DIM):
            tg[e * ACT_DIM + j] = delta_target(
                arm_q[e * ACT_DIM + j],
                Float64(ap[unsafe_offset = e * ACT_DIM + j]),
                j, a_lo[j], a_hi[j], d_arm, d_grip,
            )


def _target_targets(
    ap: Pointer[Scalar[DT], MutAnyOrigin],
    mut tg: List[Float64],
    mut tprev: List[Float64],
    ref arm_q: List[Float64],
    ref a_lo: List[Float64],
    ref a_hi: List[Float64],
    d_arm: Float64,
    d_grip: Float64,
    lead: Float64,
):
    """`--action target`: the policy step's joint targets from each lane's
    PREVIOUS target (`delta_action.target_step`), which they then become."""
    for e in range(N_ENVS):
        for j in range(ACT_DIM):
            var k = e * ACT_DIM + j
            tg[k] = target_step(
                tprev[k], arm_q[k], Float64(ap[unsafe_offset=k]), j,
                a_lo[j], a_hi[j], d_arm, d_grip, lead,
            )
            tprev[k] = tg[k]


def _target_reset(
    mut tprev: List[Float64], ref arm_q: List[Float64], lane: Int,
):
    """A new episode: lane `lane`'s target starts on its joints."""
    for j in range(ACT_DIM):
        tprev[lane * ACT_DIM + j] = arm_q[lane * ACT_DIM + j]


def _targets_to_env(
    ref tg: List[Float64],
    ep: Pointer[Scalar[DT], MutAnyOrigin],
    ref a_lo: List[Float64],
    ref a_hi: List[Float64],
    mut lag: ServoLag,
):
    """One TICK: the held targets through the servo model (it advances per
    tick, not per policy step), normalised onto the env's absolute action."""
    for e in range(N_ENVS):
        for j in range(ACT_DIM):
            var tgt = lag.apply(e, j, tg[e * ACT_DIM + j])
            var mid = 0.5 * (a_lo[j] + a_hi[j])
            var half = 0.5 * (a_hi[j] - a_lo[j])
            ep[unsafe_offset = e * ACT_DIM + j] = Scalar[DT]((tgt - mid) / half)
    lag.advance()


def _lag_reset(
    mut lag: ServoLag, ref arm_q: List[Float64], lane: Int,
):
    """A fresh draw of lane `lane`'s servo model, settled on its joints."""
    lag.reset_lane(
        lane, arm_q, lane * ACT_DIM, random_float64(), random_float64(),
        random_float64(),
    )


def _hist_push(
    mut hist: List[Float64], ap: Pointer[Scalar[DT], MutAnyOrigin],
):
    """Shift every lane's action history by one and put the EXECUTED action
    (clipped to [-1, 1], as `delta_target` reads it) in slot 0."""
    comptime if ACT_HIST > 0:
        comptime W = ACT_HIST * ACT_DIM
        for e in range(N_ENVS):
            for k in range(W - 1, ACT_DIM - 1, -1):
                hist[e * W + k] = hist[e * W + k - ACT_DIM]
            for j in range(ACT_DIM):
                var v = Float64(ap[unsafe_offset = e * ACT_DIM + j])
                hist[e * W + j] = 1.0 if v > 1.0 else (-1.0 if v < -1.0 else v)


def _hist_clear(mut hist: List[Float64], lane: Int):
    comptime W = ACT_HIST * ACT_DIM
    for k in range(W):
        hist[lane * W + k] = 0.0


def _augment[E_OBS: Int](
    raw: Pointer[Scalar[DT], MutAnyOrigin],
    ref hist: List[Float64],
    ref tprev: List[Float64],
    ref a_qa: List[Int],
    aug: Pointer[Scalar[DT], MutAnyOrigin],
):
    """The policy's observation: each lane's env row, then its history,
    then (`TARGET_OBS`) its target's lead over the joints, `target - q`."""
    comptime W = ACT_HIST * ACT_DIM
    comptime A = E_OBS + W + TARGET_OBS
    for e in range(N_ENVS):
        for k in range(E_OBS):
            aug[unsafe_offset = e * A + k] = raw[unsafe_offset = e * E_OBS + k]
        for k in range(W):
            aug[unsafe_offset = e * A + E_OBS + k] = Scalar[DT](hist[e * W + k])
        comptime if TARGET_OBS > 0:
            for j in range(ACT_DIM):
                var q = Float64(raw[unsafe_offset = e * E_OBS + a_qa[j]])
                aug[unsafe_offset = e * A + E_OBS + W + j] = Scalar[DT](
                    tprev[e * ACT_DIM + j] - q
                )


def _bc_pretrain[OBS: Int](
    mut agent: AgentT[OBS],
    ctx: DeviceContext,
    demo_paths: String,
    action_mode: String,
    ref a_qa: List[Int],
    ref a_lo: List[Float64],
    ref a_hi: List[Float64],
    mut obs_rms: RunningMeanStd,
    updates: Int,
    lr: Float64,
    d_arm: Float64 = DELTA_ARM,
    d_grip: Float64 = DELTA_GRIPPER,
    repeat: Int = 1,
) raises:
    """Behaviour-clone the actor's MEAN on the SUCCESSFUL episodes of the
    scripted teacher's `.demo` files — so101-nexus's fix for pick-and-place
    (`bc_ppo_warp.py`: pure PPO never finds the place event).

    The observation statistics are seeded from the demo rows first, so PPO
    starts normalising in the demonstrated distribution. Labels in `delta`
    mode are the teacher's TARGETS expressed as deltas from the current
    joints, `(target - q) / scale`, clipped to [-1, 1] (the teacher's `act`
    is the family's normalised absolute target); in `absolute` mode they are
    `act` itself. MSE on the mean only; the log-std is the caller's to set.

    On the run's cadence and inputs: every `repeat`-th demo row (the demos
    are recorded per tick), labels at the run's scales (`d_arm` / `d_grip`),
    and with `TASK_PPO_ACT_HIST` the history words rebuilt from the episode's
    own earlier labels (zero at its start) — the demos hold the env's
    observation only.
    """
    comptime MB = MINIBATCH
    comptime A2 = 2 * ACT_DIM
    comptime W = ACT_HIST * ACT_DIM
    comptime E = OBS - W - TARGET_OBS
    var X = List[Scalar[DT]]()
    var Y = List[Scalar[DT]]()
    var n_rows = 0
    var n_eps = 0
    for path in demo_paths.split(","):
        var d = read_demo_file(String(path))
        if d.obs_dim != E or d.act_dim != ACT_DIM:
            raise Error(
                "ppo bc: " + String(path) + " has obs " + String(d.obs_dim)
                + " / act " + String(d.act_dim) + ", the env " + String(E)
                + " / " + String(ACT_DIM)
            )
        for ep in range(d.n_episodes()):
            if not d.ep_success[ep]:
                continue
            n_eps += 1
            var hist = List[Float64](length=W, fill=0.0)
            var r = d.ep_start[ep]
            while r < d.ep_start[ep] + d.ep_len[ep]:
                for k in range(E):
                    X.append(Scalar[DT](d.obs[r * E + k]))
                for k in range(W):
                    X.append(Scalar[DT](hist[k]))
                for _ in range(TARGET_OBS):
                    X.append(Scalar[DT](0))
                var lab = List[Float64](length=ACT_DIM, fill=0.0)
                for j in range(ACT_DIM):
                    var a = Float64(d.act[r * ACT_DIM + j])
                    if action_mode == "delta":
                        var mid = 0.5 * (a_lo[j] + a_hi[j])
                        var half = 0.5 * (a_hi[j] - a_lo[j])
                        var tgt = mid + a * half
                        var q = Float64(d.obs[r * E + a_qa[j]])
                        a = (tgt - q) / delta_scale(j, d_arm, d_grip)
                    if a > 1.0:
                        a = 1.0
                    elif a < -1.0:
                        a = -1.0
                    lab[j] = a
                    Y.append(Scalar[DT](a))
                # the history the policy would have: this label, then the
                # earlier ones
                for k in range(W - 1, ACT_DIM - 1, -1):
                    hist[k] = hist[k - ACT_DIM]
                comptime if W > 0:
                    for j in range(ACT_DIM):
                        hist[j] = lab[j]
                n_rows += 1
                r += repeat
    if n_rows < MB:
        raise Error("ppo bc: only " + String(n_rows) + " demo rows")
    print("  bc: ", n_eps, "successful episodes,", n_rows, "rows from",
          demo_paths)
    obs_rms.update(mptr(X.unsafe_ptr()), n_rows, OBS)
    var Xn = List[Scalar[DT]](length=n_rows * OBS, fill=Scalar[DT](0))
    obs_rms.normalize_into(
        mptr(X.unsafe_ptr()), mptr(Xn.unsafe_ptr()), n_rows, OBS, OBS_CLIP
    )

    # ⚠ HOST-FILLED TENSORS ARE `alloc`ed (host `.data`) and uploaded ONCE —
    # `Tensor.make["gpu"]` allocates the device buffer only, with an empty
    # host list; the per-update copies are then `upload_resident`.
    var obs_t = Tensor.alloc(MB * OBS)
    obs_t.upload(ctx)
    var g_t = Tensor.alloc(MB * A2)
    g_t.upload(ctx)
    var ao_t = Tensor.make["gpu"](MB * A2, ctx)
    var og_t = Tensor.make["gpu"](MB * OBS, ctx)
    agent.trainer.actor_opt.set_lr(Scalar[DT](lr))
    var loss_acc = 0.0
    for u in range(updates):
        var idx = List[Int](capacity=MB)
        for _ in range(MB):
            idx.append(Int(random_ui64(0, UInt64(n_rows - 1))))
        for b in range(MB):
            for k in range(OBS):
                obs_t.data[b * OBS + k] = Xn[idx[b] * OBS + k]
        obs_t.upload_resident(ctx)
        agent.trainer.actor_opt.zero_grad["gpu", M=ActorNet[OBS]](
            agent.trainer.actor, ctx
        )
        call_forward["gpu", MB](
            agent.trainer.actor, TensorRefs[1](obs_t), ao_t, ctx
        )
        ao_t.download(ctx)
        var loss = 0.0
        for b in range(MB):
            for j in range(A2):
                g_t.data[b * A2 + j] = Scalar[DT](0)
            for j in range(ACT_DIM):
                var diff = Float64(ao_t.data[b * A2 + j]) - Float64(
                    Y[idx[b] * ACT_DIM + j]
                )
                loss += diff * diff
                g_t.data[b * A2 + j] = Scalar[DT](
                    2.0 * diff / Float64(MB * ACT_DIM)
                )
        loss_acc += loss / Float64(MB * ACT_DIM)
        g_t.upload_resident(ctx)
        call_vjp["gpu", MB](
            agent.trainer.actor, TensorRefs[1](obs_t), g_t,
            TensorRefs[1](og_t), ctx,
        )
        _ = agent.trainer.actor_opt.clip_grads["gpu", M=ActorNet[OBS]](
            agent.trainer.actor, Scalar[DT](1.0), ctx
        )
        agent.trainer.actor_opt.step["gpu", M=ActorNet[OBS]](
            agent.trainer.actor, ctx
        )
        if (u + 1) % 250 == 0:
            print("  bc update", u + 1, "/", updates, "| mse", loss_acc / 250.0)
            loss_acc = 0.0


def _gauss() -> Float64:
    """A standard normal draw (Box-Muller on the host RNG)."""
    var u1 = random_float64()
    if u1 < 1e-12:
        u1 = 1e-12
    return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.141592653589793 * random_float64())


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
    comptime E_OBS = OBS_DIM[M, C]
    """The env's observation words; the policy's `OBS` adds the history."""
    comptime OBS = E_OBS + ACT_HIST * ACT_DIM + TARGET_OBS
    comptime assert N_ENVS * ROLLOUT % N_MINIBATCHES == 0
    comptime assert ACT_DIM == DELTA_ACT, "the delta action is six words"

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
    var init_dir = _arg(args, "--init", "")
    var bc_demos = _arg(args, "--bc-demos", "")
    var bc_updates = Int(_arg(args, "--bc-updates", "2000"))
    var bc_lr = Float64(_arg(args, "--bc-lr", "0.001"))
    var bc_log_std = Float64(_arg(args, "--bc-log-std", "-1.0"))
    var eval_rounds = Int(_arg(args, "--eval-rounds", "4"))
    var exec_noise = Float64(_arg(args, "--exec-noise", "0"))
    var log_every = max(Int(_arg(args, "--log-every", "10")), 1)
    # the real servos' response (`delta_action.ServoLag`): `--lag-tau lo,hi`
    # ms and `--lag-delay lo,hi` ticks, drawn per episode; off by default
    var lag_tau = _arg(args, "--lag-tau", "")
    var lag_delay = _arg(args, "--lag-delay", "")
    # the real arm's speed cap and elbow stop (`ServoLag.set_limits`)
    var lag_vmax = _arg(args, "--lag-vmax", "")
    var elbow_max = Float64(_arg(args, "--elbow-max", "0"))
    # per-joint servo dynamics, offsets, control-period jitter
    # (`ServoLag.set_per_joint` / `set_offset` / `set_period`; why in its
    # header): "lo,hi;..." x 6 joints (offsets in rad), "lo,hi" ms
    var lag_tau_j = _arg(args, "--lag-tau-j", "")
    var lag_delay_j = _arg(args, "--lag-delay-j", "")
    var lag_vmax_j = _arg(args, "--lag-vmax-j", "")
    var lag_off_j = _arg(args, "--lag-offset-j", "")
    var lag_period = _arg(args, "--lag-period", "")
    # the delta action's per-step scales (rad at a = 1): the defaults are
    # so101-nexus's; under `--lag-*` the real servos need larger ones
    var d_arm = Float64(_arg(args, "--delta-arm", String(DELTA_ARM)))
    var d_grip = Float64(_arg(args, "--delta-gripper", String(DELTA_GRIPPER)))
    # ⚠ `--repeat K`: the policy acts every K ticks (31.25 / K Hz) and its
    # delta targets are HELD for the K ticks, the servo model running per
    # tick underneath. Why: at 31 Hz under the real servos' 1-2 tick delay
    # and ~50 ms response, every action lands while the last two or three are
    # still in flight, and the lagged cube-in-bowl teachers plateaued at
    # 30-37 % (the stiff sim: 79.5 %); at 10 Hz (K = 3) most of an action is
    # done within its own step — the regime the real expert (it waits for the
    # arm to settle) and Squint (10 Hz) work in. Rewards are summed over the
    # K ticks; the steps counted are TICKS.
    var repeat = max(Int(_arg(args, "--repeat", "1")), 1)
    # ⚠ `--smooth-penalty W`: each policy step pays W x the mean squared
    # change of the five ARM action words (as executed) from the lane's last
    # step (not on an episode's first). Why: the 31 Hz lag students flip the
    # sign of their arm actions on ~25-30 % of ticks, in sim as on the real
    # arm, where it shook the clamped tower and blurred the wrist camera; a
    # deploy-side EMA of 0.5 halved the flips and RAISED the probe's success
    # (7/16 -> 11/16) — smoothness is not bought with skill here.
    var smooth_w = Float64(_arg(args, "--smooth-penalty", "0"))
    # ⚠ `--dive-penalty W`: each tick with the arm reaching down past the desk
    # (shoulder_lift > 1.35 AND elbow_flex < -1.35 rad) costs W. No sim grasp
    # is made there (they close at shoulder_lift -0.1..0.5, elbow 0.35..1.4),
    # yet the smooth student spent 12.6 % of its probe ticks there: the sim's
    # rigid desk absorbs the push; on the real arm it drove into the desk and
    # rocked the clamped tower.
    var dive_w = Float64(_arg(args, "--dive-penalty", "0"))
    # `--target-lead L` (`--action target`): the target stays within L rad of
    # the measured joint (`delta_action.target_step`); 0 = unbounded
    var target_lead = Float64(_arg(args, "--target-lead", "0"))
    if action_mode != "absolute" and action_mode != "delta" and action_mode != "target":
        raise Error("ppo task: --action absolute|delta|target, got " + action_mode)
    # a per-step action (`delta` or `target`): through `ServoLag`, held over
    # `--repeat`, with the action history
    var stepped = action_mode != "absolute"
    if (action_mode == "target") != (TARGET_OBS > 0):
        raise Error("ppo task: --action target needs -D TASK_PPO_TARGET_OBS"
                    " and the define needs --action target (the target is"
                    " hidden state the policy must see)")
    if action_mode == "target" and bc_demos.byte_length() > 0:
        raise Error("ppo task: --bc-demos labels absolute or delta actions,"
                    " not --action target")
    if C.MAX_STEPS % repeat != 0:
        raise Error("ppo task: --repeat " + String(repeat) + " must divide the horizon "
                    + String(C.MAX_STEPS) + " (episodes end on a policy step)")
    if repeat > 1 and not stepped:
        raise Error("ppo task: --repeat is for --action delta|target")
    if ACT_HIST > 0 and not stepped:
        raise Error("ppo task: TASK_PPO_ACT_HIST is for --action delta|target")
    if reward != "potential" and reward != "legacy":
        raise Error("ppo task: --reward potential|legacy, got " + reward)
    var rw = reward_mode_words(reward == "potential", bonus)
    seed_rng(seed)

    var n_updates_total = anneal_steps // (N_ENVS * ROLLOUT * repeat)
    print("=" * 70)
    print("PPO on", family, "—", task)
    print("  lanes", N_ENVS, "| rollout", ROLLOUT, "| batch", N_ENVS * ROLLOUT,
          "| minibatch", MINIBATCH, "x", N_MINIBATCHES, "| epochs", N_EPOCHS)
    print("  obs", OBS, "(env", E_OBS, "+ last", ACT_HIST, "actions +",
          TARGET_OBS, "target lead) | act",
          ACT_DIM, "| hidden", HIDDEN, "| horizon",
          C.MAX_STEPS)
    print("  steps", total_steps, "| lr", lr0, "| ent", ent0, "->", ent1,
          "over", anneal_steps, "steps | log_std init", log_std0)
    print("  reward", reward, "| success bonus", bonus, "| seed", seed,
          "| action", action_mode, "| repeat", repeat, "ticks per policy step",
          "| target lead", target_lead)
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
        project=project, driver=driver,
        # an evaluation of a trained policy (`--steps 0`) is a run too, but
        # it must not read as a training run that logged one point
        slug=String("eval-ppo-" if total_steps == 0 else "ppo-") + task,
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
    logger.set_config("bc_demos", bc_demos)
    logger.set_config("bc_updates", String(bc_updates))
    logger.set_config("success_bonus", String(bonus))
    logger.set_config("exec_noise", String(exec_noise))
    logger.set_config("lag_tau_ms", lag_tau)
    logger.set_config("lag_delay_ticks", lag_delay)
    logger.set_config("lag_vmax", lag_vmax)
    logger.set_config("elbow_max", String(elbow_max))
    logger.set_config("lag_tau_j", lag_tau_j)
    logger.set_config("lag_delay_j", lag_delay_j)
    logger.set_config("lag_vmax_j", lag_vmax_j)
    logger.set_config("lag_offset_j", lag_off_j)
    logger.set_config("lag_period_ms", lag_period)
    logger.set_config("delta_arm", String(d_arm))
    logger.set_config("delta_gripper", String(d_grip))
    logger.set_config("act_hist", String(ACT_HIST))
    logger.set_config("target_obs", String(TARGET_OBS))
    logger.set_config("target_lead", String(target_lead))
    logger.set_config("repeat", String(repeat))
    logger.set_config("smooth_penalty", String(smooth_w))
    logger.set_config("dive_penalty", String(dive_w))
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
        if init_dir.byte_length() > 0:
            agent.trainer.load_state(init_dir + "/checkpoints/last.ckpt")
            print("  init: actor + critic from", init_dir)
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
            # a multiple of --repeat: every episode ends on a policy step
            env.d.meta.data[e * METADATA_SIZE + META_IDX_STEP_COUNT] = Scalar[DT](
                Int(random_float64() * Float64(C.MAX_STEPS // repeat)) * repeat
            )
        env.d.meta.upload(ctx)
        ctx.synchronize()

        # ── host scratch ─────────────────────────────────────────────────
        var raw_h = ctx.enqueue_create_host_buffer[DT](N_ENVS * E_OBS)
        var aug = List[Scalar[DT]](length=N_ENVS * OBS, fill=Scalar[DT](0))
        var hist = List[Float64](length=N_ENVS * ACT_HIST * ACT_DIM, fill=0.0)
        var cur_n = ctx.enqueue_create_host_buffer[DT](N_ENVS * OBS)
        var next_n = ctx.enqueue_create_host_buffer[DT](N_ENVS * OBS)
        var act_h = ctx.enqueue_create_host_buffer[DT](N_ENVS * ACT_DIM)
        var rew_h = ctx.enqueue_create_host_buffer[DT](N_ENVS)
        var rew_n = ctx.enqueue_create_host_buffer[DT](N_ENVS)
        var done_h = ctx.enqueue_create_host_buffer[DT](N_ENVS)
        var rets = ctx.enqueue_create_host_buffer[DT](N_ENVS)
        var env_act = ctx.enqueue_create_host_buffer[DT](N_ENVS * ACT_DIM)
        var arm_q = List[Float64](length=N_ENVS * ACT_DIM, fill=0.0)
        # `--action target`: each lane's last commanded target
        var tprev = List[Float64](length=N_ENVS * ACT_DIM, fill=0.0)
        # the policy step's held targets, and per lane over its ticks: the
        # summed reward, whether it ended, its terminal observation
        var tg = List[Float64](length=N_ENVS * ACT_DIM, fill=0.0)
        var rsum = List[Float64](length=N_ENVS, fill=0.0)
        var dmac = List[Bool](length=N_ENVS, fill=False)
        var term = List[Scalar[DT]](length=N_ENVS * E_OBS, fill=Scalar[DT](0))
        # `--smooth-penalty`: each lane's last executed action, and whether
        # its episode has one yet
        var a_prev = List[Float64](length=N_ENVS * ACT_DIM, fill=0.0)
        var has_prev = List[Bool](length=N_ENVS, fill=False)
        var spen = List[Float64](length=N_ENVS, fill=0.0)
        var spen_acc = 0.0
        var spen_n = 0
        var dive_ticks = 0
        var all_ticks = 0
        var lag = ServoLag.parse(
            N_ENVS, lag_tau, lag_delay, Float64(C.FRAME_SKIP) * M.TIMESTEP
        )
        lag.set_limits(lag_vmax, elbow_max)
        lag.set_per_joint(lag_tau_j, lag_delay_j, lag_vmax_j)
        lag.set_offset(lag_off_j)
        lag.set_period(lag_period)
        if lag.on:
            print("  servo lag: tau", lag_tau, "ms, delay", lag_delay,
                  "ticks (per episode, per lane) | arm speed cap", lag_vmax,
                  "rad/s | elbow max", elbow_max)
        if lag.per_joint or lag.has_off or lag.dt_hi > 0.0:
            print("  servo per joint: tau", lag_tau_j, "ms | delay", lag_delay_j,
                  "| cap", lag_vmax_j, "rad/s | offset", lag_off_j,
                  "rad | period", lag_period, "ms")
        ctx.synchronize()
        var obs_dev = DeviceBuffer[DT](ctx, env.obs_ptr(), N_ENVS * E_OBS, owning=False)
        var act_dev = DeviceBuffer[DT](ctx, env.action_ptr(), N_ENVS * ACT_DIM, owning=False)
        var rew_dev = DeviceBuffer[DT](ctx, env.reward_ptr(), N_ENVS, owning=False)
        var done_dev = DeviceBuffer[DT](ctx, env.done_ptr(), N_ENVS, owning=False)

        var obs_rms = RunningMeanStd(OBS)
        if init_dir.byte_length() > 0:
            obs_rms.load(init_dir + "/obs_norm.txt")
            print("  init: observation statistics from", init_dir)
        if bc_demos.byte_length() > 0:
            _bc_pretrain[OBS](
                agent, ctx, bc_demos, action_mode, a_qa, a_lo, a_hi, obs_rms,
                bc_updates, bc_lr, d_arm, d_grip, repeat,
            )
            agent.trainer.actor_opt.set_lr(Scalar[DT](lr0))
            agent.trainer.actor.children[4].set_log_std_init["gpu"](
                Scalar[DT](bc_log_std), ctx
            )
            print("  bc: done; PPO starts from the cloned mean, log-std",
                  bc_log_std)
        var ret_rms = RunningMeanStd(1)
        var ret_acc = List[Float64](length=N_ENVS, fill=0.0)
        var raw_ret = List[Float64](length=N_ENVS, fill=0.0)
        var succ = List[Bool](length=N_ENVS, fill=False)
        var hist_succ = List[Bool]()
        var hist_ret = List[Float64]()
        var n_episodes = 0
        var diverged = List[Bool](length=N_ENVS, fill=False)
        var n_diverged = 0

        comptime assert not ENV_GRAPH or DEVICE_ROLLOUT, (
            "TASK_PPO_ENV_GRAPH needs TASK_PPO_DEVICE_ROLLOUT"
        )
        var dev: Optional[FamilyDeviceRollout[N_ENVS, E_OBS]] = None
        var env_graph: Optional[CUDAGraph] = None
        var reset_graph: Optional[CUDAGraph] = None
        var meta_ptr = mptr(env.d.meta.dev.value().unsafe_ptr())
        comptime if DEVICE_ROLLOUT:
            var mode = 0 if action_mode == "absolute" else (
                1 if action_mode == "delta" else 2
            )
            dev = FamilyDeviceRollout[N_ENVS, E_OBS](
                ctx,
                FamilyStepConfig(
                    mode, repeat, d_arm, d_grip, target_lead, exec_noise,
                    smooth_w, dive_w, GAMMA, OBS_CLIP, REW_CLIP, OBS_BOUND,
                    REW_BOUND,
                ),
                UInt64(seed),
                a_qa, a_lo, a_hi, lag,
            )
            dev.value().stats_from_host(obs_rms.mean, obs_rms.var_, obs_rms.count)
            agent.trainer.enable_device_rollout(
                UInt64(seed) * UInt64(2654435761) + UInt64(1)
            )

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
                var a = Float64(rp[unsafe_offset = e * E_OBS + a_qa[j]])
                var b = Float64(env.d.qpos.data[e * NQ_M + a_qa[j]])
                if abs(a - b) > 1e-5:
                    raise Error(
                        "ppo task: obs[" + String(a_qa[j]) + "] of lane "
                        + String(e) + " is " + String(a) + ", qpos is "
                        + String(b) + " — the observation does not start with"
                        " qpos, and --action delta would anchor on garbage"
                    )
        comptime if DEVICE_ROLLOUT:
            dev.value().start(env.obs_ptr())
        else:
            for e in range(N_ENVS):
                for j in range(ACT_DIM):
                    arm_q[e * ACT_DIM + j] = Float64(rp[unsafe_offset = e * E_OBS + a_qa[j]])
                _lag_reset(lag, arm_q, e)
                _target_reset(tprev, arm_q, e)
            _augment[E_OBS](rp, hist, tprev, a_qa, mptr(aug.unsafe_ptr()))
            obs_rms.update(mptr(aug.unsafe_ptr()), N_ENVS, OBS)
            obs_rms.normalize_into(mptr(aug.unsafe_ptr()), mptr(cur_n.unsafe_ptr()), N_ENVS, OBS, OBS_CLIP)

        var step = 0
        var it = 0
        var n_updates = 0
        var train_graph: Optional[CUDAGraph] = None
        var t0 = perf_counter_ns()
        var next_ckpt = ckpt_every
        var ckpt_path = run.checkpoint_path(String("last"))
        while step < total_steps:
            comptime if DEVICE_ROLLOUT:
                dev.value().step[USE_ENV_GRAPH=ENV_GRAPH](
                    agent.trainer, env, meta_ptr, env_graph, reset_graph
                )
                if dev.value().ring_full():
                    dev.value().drain(hist_succ, hist_ret, n_episodes)
            else:
                # 1. act on the normalised observation
                agent.trainer.select_action_batched(
                    mptr(cur_n.unsafe_ptr()), mptr(act_h.unsafe_ptr()), step,
                )
                # ⚠ `--exec-noise`: the EXECUTED action is perturbed, the RECORDED
                # one is not (the trainer keeps its own sample) — noise of the
                # environment, not of the policy, so the policy meets states off
                # its own trajectories and learns to recover from them. Why: the
                # cube-in-bowl teacher CHATTERS where its pixel student hovers
                # (per step |teacher - student| ~0.9 on four joints, the teacher
                # opening in 13 % of those steps) — states it never visits.
                if exec_noise > 0.0:
                    var ah = mptr(act_h.unsafe_ptr())
                    for k in range(N_ENVS * ACT_DIM):
                        var v = Float64(ah[unsafe_offset=k]) + exec_noise * _gauss()
                        if v > 1.0:
                            v = 1.0
                        elif v < -1.0:
                            v = -1.0
                        ah[unsafe_offset=k] = Scalar[DT](v)
                if smooth_w > 0.0:
                    var ap = mptr(act_h.unsafe_ptr())
                    for e in range(N_ENVS):
                        var d2 = 0.0
                        for j in range(ACT_DIM - 1):
                            var v = Float64(ap[unsafe_offset = e * ACT_DIM + j])
                            v = 1.0 if v > 1.0 else (-1.0 if v < -1.0 else v)
                            if has_prev[e]:
                                var dd = v - a_prev[e * ACT_DIM + j]
                                d2 += dd * dd
                            a_prev[e * ACT_DIM + j] = v
                        spen[e] = smooth_w * d2 / Float64(ACT_DIM - 1)
                        has_prev[e] = True
                        spen_acc += spen[e]
                        spen_n += 1
                if action_mode == "delta":
                    _delta_targets(
                        mptr(act_h.unsafe_ptr()), tg, arm_q, a_lo, a_hi, d_arm, d_grip,
                    )
                elif action_mode == "target":
                    _target_targets(
                        mptr(act_h.unsafe_ptr()), tg, tprev, arm_q, a_lo, a_hi,
                        d_arm, d_grip, target_lead,
                    )
                _hist_push(hist, mptr(act_h.unsafe_ptr()))
                # 2. `repeat` ticks under the held targets: per lane the summed
                # reward, the goal bit, and — for a lane that ends (or diverges)
                # before the last tick — its TERMINAL row, kept (its later ticks
                # are stepped and discarded; it resets with the others below)
                for e in range(N_ENVS):
                    rsum[e] = 0.0
                    dmac[e] = False
                    diverged[e] = False
                var n_bad = 0
                for tick in range(repeat):
                    if stepped:
                        _targets_to_env(tg, mptr(env_act.unsafe_ptr()), a_lo, a_hi, lag)
                        ctx.enqueue_copy(act_dev, env_act)
                    else:
                        ctx.enqueue_copy(act_dev, act_h)
                    env.step_batch[N_ENVS](
                        ctx=ctx, rng_seed=UInt64(it * repeat + tick + 1)
                    )
                    ctx.enqueue_copy(raw_h, obs_dev)
                    ctx.enqueue_copy(rew_h, rew_dev)
                    ctx.enqueue_copy(done_h, done_dev)
                    env.d.meta.download(ctx)
                    ctx.synchronize()
                    var rp_t = mptr(raw_h.unsafe_ptr())
                    var dh_t = mptr(done_h.unsafe_ptr())
                    var rh_t = mptr(rew_h.unsafe_ptr())
                    for e in range(N_ENVS):
                        if dmac[e]:
                            continue
                        var bad = False
                        var rv = Float64(rh_t[unsafe_offset=e])
                        if not (rv == rv) or abs(rv) > REW_BOUND:
                            bad = True
                        for k in range(E_OBS):
                            var v = Float64(rp_t[unsafe_offset = e * E_OBS + k])
                            if not (v == v) or abs(v) > OBS_BOUND:
                                bad = True
                                break
                        if bad:
                            diverged[e] = True
                            n_bad += 1
                            dmac[e] = True
                        else:
                            rsum[e] += rv
                            if dive_w > 0.0:
                                all_ticks += 1
                                var q1 = Float64(rp_t[unsafe_offset = e * E_OBS + a_qa[1]])
                                var q2 = Float64(rp_t[unsafe_offset = e * E_OBS + a_qa[2]])
                                if q1 > 1.35 and q2 < -1.35:
                                    rsum[e] -= dive_w
                                    dive_ticks += 1
                            if env.d.meta.data[e * METADATA_SIZE + META_IDX_GOAL_HELD] > Scalar[DT](0.5):
                                succ[e] = True
                            if dh_t[unsafe_offset=e] > Scalar[DT](0.5):
                                dmac[e] = True
                        # ⚠ AT WHATEVER TICK it ends — the last one included. A
                        # copy only for ticks before the last left a lane that
                        # ended on the last tick (the usual case) reading the row
                        # from its PREVIOUS early end: after one physics blow-up,
                        # that lane's diverged row re-entered the observation
                        # statistics, UNFLAGGED, at each of its later episode
                        # ends — the joint means went to +-593, the variances to
                        # 1e11, and the 10 Hz lift_real stages collapsed from ~60 %
                        # to 0 at their first diverged lane (18ea9d25, b5ccffa3).
                        if dmac[e]:
                            for k in range(E_OBS):
                                term[e * E_OBS + k] = rp_t[unsafe_offset = e * E_OBS + k]
                n_diverged += n_bad
                # the policy step's transition: the last tick's rows, a lane that
                # ended early its terminal row; the summed reward; done if it ended
                var raw_p = mptr(raw_h.unsafe_ptr())
                var dh0 = mptr(done_h.unsafe_ptr())
                var rh0 = mptr(rew_h.unsafe_ptr())
                for e in range(N_ENVS):
                    if smooth_w > 0.0:
                        rsum[e] -= spen[e]
                    rh0[unsafe_offset=e] = Scalar[DT](rsum[e])
                    dh0[unsafe_offset=e] = Scalar[DT](1) if dmac[e] else Scalar[DT](0)
                if repeat > 1:
                    for e in range(N_ENVS):
                        if dmac[e]:
                            for k in range(E_OBS):
                                raw_p[unsafe_offset = e * E_OBS + k] = term[e * E_OBS + k]
                _augment[E_OBS](raw_p, hist, tprev, a_qa, mptr(aug.unsafe_ptr()))
                obs_rms.update(mptr(aug.unsafe_ptr()), N_ENVS, OBS, diverged)
                obs_rms.normalize_into(
                    mptr(aug.unsafe_ptr()), mptr(next_n.unsafe_ptr()), N_ENVS, OBS, OBS_CLIP,
                )
                if n_bad > 0:
                    # the terminal obs of a diverged lane is garbage: zero it
                    var nn = mptr(next_n.unsafe_ptr())
                    for e in range(N_ENVS):
                        if diverged[e]:
                            for k in range(OBS):
                                nn[unsafe_offset = e * OBS + k] = Scalar[DT](0)
                if n_bad > 0 or repeat > 1:
                    # the env resets on its OWN done buffer: write the policy
                    # step's dones (forced, or from an earlier tick) back before
                    # `selective_reset_batch`
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
                var dh2 = mptr(done_h.unsafe_ptr())
                for e in range(N_ENVS):
                    for j in range(ACT_DIM):
                        arm_q[e * ACT_DIM + j] = Float64(
                            rp2[unsafe_offset = e * E_OBS + a_qa[j]]
                        )
                    if dh2[unsafe_offset=e] > Scalar[DT](0.5):
                        _lag_reset(lag, arm_q, e)
                        _target_reset(tprev, arm_q, e)
                        _hist_clear(hist, e)
                        has_prev[e] = False
                _augment[E_OBS](rp2, hist, tprev, a_qa, mptr(aug.unsafe_ptr()))
                obs_rms.normalize_into(
                    mptr(aug.unsafe_ptr()), mptr(cur_n.unsafe_ptr()), N_ENVS, OBS, OBS_CLIP,
                )
            step += N_ENVS * repeat
            it += 1
            # 6. the update at the rollout boundary, then the schedules
            var updated: Bool
            comptime if TRAIN_GRAPH or DEVICE_ROLLOUT:
                # the device rollout's pool is on the device: always the
                # device update (captured under TRAIN_GRAPH)
                updated = onpolicy_update_device[
                    type_of(agent.trainer), TRAIN_GRAPH
                ](
                    agent.trainer, Optional(ctx), step, train_graph,
                    quiet=n_updates > 0,
                )
            else:
                updated = agent.trainer.train_step(step)
            if updated:
                n_updates += 1
                var frac = 1.0 - Float64(n_updates) / Float64(max(n_updates_total, 1))
                if frac < 0.0:
                    frac = 0.0
                agent.trainer.actor_opt.set_lr(Scalar[DT](lr0 * frac))
                agent.trainer.critic_opt.set_lr(Scalar[DT](lr0 * frac))
                agent.trainer.actor_train.set_entropy_coef(
                    Scalar[DT](ent1 + (ent0 - ent1) * frac)
                )
                # The new lr / entropy are kernel arguments: re-capture.
                train_graph = None
                comptime if DEVICE_ROLLOUT:
                    # the device's episode records and counters, now
                    dev.value().drain(hist_succ, hist_ret, n_episodes)
                    var st = dev.value().read_window_stats()
                    n_diverged = Int(st[0])
                    spen_acc = st[1]
                    spen_n = Int(st[2])
                    dive_ticks = Int(st[3])
                    all_ticks = Int(st[4])
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
                # ⚠ EVERY `--log-every` UPDATES, not every update: a 60M run
                # at 1024 lanes is 3 662 updates x 16 series, more points than
                # a chart can show. The trainer's metrics are ACCUMULATORS
                # drained at each flush, so a sparser flush logs the window's
                # MEAN, not a sample of it.
                if n_updates % log_every == 0:
                    logger.log_scalar("success_rate", rate, step)
                    logger.log_scalar("episode_return", mret, step)
                    logger.log_scalar("episodes", Float64(n_episodes), step)
                    logger.log_scalar("diverged", Float64(n_diverged), step)
                    logger.log_scalar("sps", Float64(step) / secs, step)
                    if spen_n > 0:
                        logger.log_scalar("smooth_penalty_mean", spen_acc / Float64(spen_n), step)
                    logger.log_scalar("lr", lr0 * frac, step)
                    logger.log_scalar("ent_coef", ent1 + (ent0 - ent1) * frac, step)
                    agent.trainer.flush_metrics_through_logger[RunLogger](
                        logger_ptr, step
                    )
                if n_updates % 10 == 0:
                    if spen_n > 0:
                        print("  smooth penalty per step", spen_acc / Float64(spen_n))
                        spen_acc = 0.0
                        spen_n = 0
                    if all_ticks > 0:
                        print("  dive fraction", Float64(dive_ticks) / Float64(all_ticks))
                        dive_ticks = 0
                        all_ticks = 0
                    comptime if DEVICE_ROLLOUT:
                        dev.value().reset_window_stats()
                    print("  step", step, "| success", rate, "over", nw,
                          "ep | return", mret, "| episodes", n_episodes,
                          "| diverged", n_diverged,
                          "|", Int(Float64(step) / secs), "steps/s")
            if step >= next_ckpt:
                agent.trainer.save_state(ckpt_path)
                comptime if DEVICE_ROLLOUT:
                    dev.value().stats_to_host(
                        obs_rms.mean, obs_rms.var_, obs_rms.count
                    )
                obs_rms.save(run.dir + "/obs_norm.txt")
                next_ckpt += ckpt_every
        comptime if DEVICE_ROLLOUT:
            # the last records, and the statistics the eval normalises with
            dev.value().drain(hist_succ, hist_ret, n_episodes)
            dev.value().stats_to_host(obs_rms.mean, obs_rms.var_, obs_rms.count)
        if total_steps > 0:
            agent.trainer.save_state(ckpt_path)
            obs_rms.save(run.dir + "/obs_norm.txt")

        # ── the greedy evaluation: the actor's MEAN, frozen statistics,
        # held-out placements (reset seeds the training never drew) ─────
        var eval_ok = 0
        var eval_n = 0
        for rnd in range(eval_rounds):
            env.reset_batch[N_ENVS](
                ctx=ctx, rng_seed=UInt64(1_000_003 + seed * 101 + rnd)
            )
            ctx.enqueue_copy(raw_h, obs_dev)
            ctx.synchronize()
            var held = List[Bool](length=N_ENVS, fill=False)
            # ⚠ WHEN it first succeeds (step index, -1 never): is the horizon
            # the limit? A lagged arm is slower; successes bunched at the end
            # of the episode say the clock, not the skill, caps the score.
            var t_held = List[Int](length=N_ENVS, fill=-1)
            # ⚠ WHERE EPISODES STOP, not only whether they succeed: the brick
            # (term 0's `a`) rising, reaching over `b`, and the closest
            # horizontal gap. A success rate cannot tell "never grasps" from
            # "carries and drops at the rim".
            var z0 = List[Float64](length=N_ENVS, fill=0.0)
            var rise = List[Float64](length=N_ENVS, fill=0.0)
            var hmin = List[Float64](length=N_ENVS, fill=1e9)
            var over = List[Bool](length=N_ENVS, fill=False)
            var held_end = List[Bool](length=N_ENVS, fill=False)
            # the brick's rise and horizontal gap at the LAST step, to sort
            # the failures (carried / inside the rim / beside it / elsewhere)
            var dz_end = List[Float64](length=N_ENVS, fill=0.0)
            var h_end = List[Float64](length=N_ENVS, fill=0.0)
            # the start state per lane (after the first step), for the
            # per-episode CSV: brick x, y, quat (x y z w — `xquat`'s order); bowl x, y
            var start = List[Float64](length=N_ENVS * 8, fill=0.0)
            for t in range(C.MAX_STEPS - 1):
                var rq = mptr(raw_h.unsafe_ptr())
                # the policy acts every `repeat` ticks; its targets are held
                # between, the servo model advancing per tick
                if t % repeat == 0:
                    for e in range(N_ENVS):
                        for j in range(ACT_DIM):
                            arm_q[e * ACT_DIM + j] = Float64(
                                rq[unsafe_offset = e * E_OBS + a_qa[j]]
                            )
                        if t == 0:
                            _lag_reset(lag, arm_q, e)
                            _target_reset(tprev, arm_q, e)
                            _hist_clear(hist, e)
                    _augment[E_OBS](rq, hist, tprev, a_qa, mptr(aug.unsafe_ptr()))
                    obs_rms.normalize_into(
                        mptr(aug.unsafe_ptr()), mptr(cur_n.unsafe_ptr()), N_ENVS, OBS, OBS_CLIP
                    )
                    agent.trainer.select_greedy_action_batched(
                        mptr(cur_n.unsafe_ptr()), mptr(act_h.unsafe_ptr())
                    )
                    if action_mode == "delta":
                        _delta_targets(
                            mptr(act_h.unsafe_ptr()), tg, arm_q, a_lo, a_hi,
                            d_arm, d_grip,
                        )
                    elif action_mode == "target":
                        _target_targets(
                            mptr(act_h.unsafe_ptr()), tg, tprev, arm_q, a_lo,
                            a_hi, d_arm, d_grip, target_lead,
                        )
                    _hist_push(hist, mptr(act_h.unsafe_ptr()))
                if stepped:
                    _targets_to_env(tg, mptr(env_act.unsafe_ptr()), a_lo, a_hi, lag)
                    ctx.enqueue_copy(act_dev, env_act)
                else:
                    ctx.enqueue_copy(act_dev, act_h)
                env.step_batch[N_ENVS](ctx=ctx, rng_seed=UInt64(t + 1))
                ctx.enqueue_copy(raw_h, obs_dev)
                env.d.meta.download(ctx)
                ctx.synchronize()
                env.d.xpos.download(ctx)
                if t == 0:
                    env.d.xquat.download(ctx)
                ctx.synchronize()
                for e in range(N_ENVS):
                    var hb = env.d.meta.data[e * METADATA_SIZE + META_IDX_GOAL_HELD] > Scalar[DT](0.5)
                    if hb:
                        held[e] = True
                        if t_held[e] < 0:
                            t_held[e] = t
                    held_end[e] = hb
                    var mb = e * METADATA_SIZE + META_IDX_TASK_PARAM_0
                    var ia = Int(env.d.meta.data[mb + 1])
                    var ib = Int(env.d.meta.data[mb + 2])
                    var xb = e * M.NBODY * 3
                    var ex = Float64(env.d.xpos.data[xb + ia * 3] - env.d.xpos.data[xb + ib * 3])
                    var ey = Float64(env.d.xpos.data[xb + ia * 3 + 1] - env.d.xpos.data[xb + ib * 3 + 1])
                    var za = Float64(env.d.xpos.data[xb + ia * 3 + 2])
                    if t == 0:
                        z0[e] = za
                        var qb = e * M.NBODY * 4 + ia * 4
                        start[e * 8 + 0] = Float64(env.d.xpos.data[xb + ia * 3])
                        start[e * 8 + 1] = Float64(env.d.xpos.data[xb + ia * 3 + 1])
                        for c in range(4):
                            start[e * 8 + 2 + c] = Float64(env.d.xquat.data[qb + c])
                        start[e * 8 + 6] = Float64(env.d.xpos.data[xb + ib * 3])
                        start[e * 8 + 7] = Float64(env.d.xpos.data[xb + ib * 3 + 1])
                    var dz = za - z0[e]
                    if dz > rise[e]:
                        rise[e] = dz
                    var h = sqrt(ex * ex + ey * ey)
                    if h < hmin[e]:
                        hmin[e] = h
                    if dz > 0.02 and h < 0.045:
                        over[e] = True
                    dz_end[e] = dz
                    h_end[e] = h
            var ok = 0
            var n_lift = 0
            var n_over = 0
            var n_end = 0
            var n_h45 = 0
            var n_h80 = 0
            var n_h150 = 0
            for e in range(N_ENVS):
                if held[e]:
                    ok += 1
                if rise[e] > 0.02:
                    n_lift += 1
                if over[e]:
                    n_over += 1
                if held_end[e]:
                    n_end += 1
                if hmin[e] < 0.045:
                    n_h45 += 1
                if hmin[e] < 0.08:
                    n_h80 += 1
                if hmin[e] < 0.15:
                    n_h150 += 1
            # ⚠ ENDINGS OF THE FAILED EPISODES, by the brick's final pose:
            # still up (> 2 cm: carried, hovering, or perched on the rim), inside the
            # bowl's 5.2 cm inner rim yet not `Near` (the 3D test wants the
            # centre within 4.5 cm), resting beside the bowl (< 10 cm), or
            # elsewhere. Cube-in-bowl's geometry; other tasks read it loosely.
            var f_up = 0
            var f_rim_in = 0
            var f_beside = 0
            var f_else = 0
            for e in range(N_ENVS):
                if held_end[e]:
                    continue
                if dz_end[e] > 0.02:
                    f_up += 1
                elif h_end[e] < 0.052:
                    f_rim_in += 1
                elif h_end[e] < 0.10:
                    f_beside += 1
                else:
                    f_else += 1
            # one row per episode: where it started, how far it got
            var csv = String(
                "lane,brick_x,brick_y,qx,qy,qz,qw,bowl_x,bowl_y,"
                + "rise_max,h_min,over,success,held_end,dz_end,h_end,t_held\n"
            )
            for e in range(N_ENVS):
                csv += String(e)
                for c in range(8):
                    csv += "," + String(start[e * 8 + c])
                csv += "," + String(rise[e]) + "," + String(hmin[e])
                csv += "," + ("1" if over[e] else "0")
                csv += "," + ("1" if held[e] else "0")
                csv += "," + ("1" if held_end[e] else "0")
                csv += "," + String(dz_end[e]) + "," + String(h_end[e])
                csv += "," + String(t_held[e]) + "\n"
            var csv_path = run.dir + "/eval_lanes_round" + String(rnd) + ".csv"
            with open(csv_path, "w") as f:
                f.write(csv)
            print("  greedy eval round", rnd, "per-episode rows:", csv_path)
            # the successes' first-success step: quartiles, and the share in
            # the horizon's last quarter
            var ts = List[Int]()
            for e in range(N_ENVS):
                if t_held[e] >= 0:
                    ts.append(t_held[e])
            if len(ts) > 0:
                sort(ts)
                var late = 0
                for k in range(len(ts)):
                    if ts[k] >= (3 * C.MAX_STEPS) // 4:
                        late += 1
                var dt_s = Float64(C.FRAME_SKIP) * M.TIMESTEP
                print("  greedy eval round", rnd, "first success at step (of",
                      C.MAX_STEPS, ") p25", ts[len(ts) // 4], "p50",
                      ts[len(ts) // 2], "p90", ts[(9 * len(ts)) // 10],
                      "| p50", Float64(ts[len(ts) // 2]) * dt_s, "s | in the last quarter",
                      late, "of", len(ts))
            print("  greedy eval round", rnd, "endings of the", N_ENVS - n_end,
                  "not held at the end | brick > 2 cm up (carried, hovering or on the rim)", f_up,
                  "| resting inside the rim, not Near", f_rim_in,
                  "| beside (< 10 cm)", f_beside, "| elsewhere", f_else)
            print("  greedy eval round", rnd, ":", ok, "/", N_ENVS,
                  "| held at the end", n_end, "| lifted >2cm", n_lift,
                  "| over b while lifted", n_over,
                  "| closest horizontal <4.5cm", n_h45, "<8cm", n_h80,
                  "<15cm", n_h150)
            eval_ok += ok
            eval_n += N_ENVS
        var eval_rate = Float64(eval_ok) / Float64(eval_n) if eval_n > 0 else 0.0
        if eval_n > 0:
            print("  GREEDY SUCCESS", eval_ok, "/", eval_n, "=", eval_rate)
            logger.log_scalar("eval_success_rate", eval_rate, step)
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
                   String("success_rate=") + String(rate)
                   + " eval_success_rate=" + String(eval_rate))
        _ = logger
