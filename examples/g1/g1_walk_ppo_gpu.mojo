"""PPO on the G1 walker — velocity tracking with a trained stop (G1_WALKER_PLAN L0c).

    pixi run -e nvidia mojo build -I . -D PPO_VEC_ENV_GRAPH -D PPO_VEC_TRAIN_GRAPH \\
        examples/g1/g1_walk_ppo_gpu.mojo -o g1_walk_ppo
    pixi run -e nvidia ./g1_walk_ppo --steps 200000000 --seed 1 --out runs/g1_walk_s1

The env is `UnitreeG1WalkBatched` (`noeira/envs/robots/unitree_g1_walk*.mojo`):
70-D observation, 29 PD-offset actions, commands / pushes / noise / random
resets inside its hooks. The driver is `run_ppo_vec`, with the last action
appended to the observation (`HIST = 1`) and the action-rate penalty.

Recipe (plan §5): 1024 lanes, rollout 24, 4 minibatches, 5 epochs, lr 3e-4
fixed, gamma 0.99, lambda 0.95, clip 0.2, entropy 0.005, grad norm 1.0,
actor and critic 512-256-128 Swish (brax's default activation, which
Playground's G1 run uses) — defined in `g1_walk_policy.mojo`.

⚠ ACTION SCALE IS THE CLAMP, NOT A GAIN. The driver clamps the Gaussian
sample to +-`G1_WALK_ACTION_CLIP` (2) and writes it raw; the env maps
`target = q_default + 0.25 a` and clamps the same way.

⚠ THE ACTION-RATE WEIGHT IS PER STEP: RoboParty's -0.02 per second
(IsaacLab scales every term by the step dt) is 4e-4 on `|a_t - a_{t-1}|^2`
in action units. The default is a quarter of that, 1e-4: our per-step
env reward is ~0.002-0.035, and at log-std -1 the exploration noise alone
has |da|^2 ~ 7.6 (0.0008 per step here). Run s1 had 4e-4 at log-std 0
(0.023 per step of pure noise: dying paid); run s2 had it off and learned
a 25 Hz bang-bang stand.

⚠ EPISODE CLOCKS ARE STAGGERED after the first reset (each lane's step
counter drawn in [0, MAX_STEPS)), so the time-limit resets do not arrive in
one wave every 1000 steps.

Outputs in `--out`: `ckpt` (trainer v3 storage checkpoint), `obs_norm.txt`
(the observation normaliser), `metrics.csv`.
"""

from std.os import makedirs
from std.os.path import exists
from std.random import seed as seed_rng, random_float64
from std.sys import argv, is_defined

from max.gpu.host import DeviceContext

from noeira.core.logger import CsvLogger
from noeira.deep_agents.ppo import PPOAgent
from noeira.deep_agents.training.obs_norm import RunningMeanStd
from noeira.deep_agents.training.ppo_vec_driver import (
    PPOVecConfig, run_ppo_vec,
)
from noeira.envs.robots.unitree_g1_walk import UnitreeG1WalkBatched
from noeira.envs.robots.unitree_g1_walk_config import G1_WALK_MAX_STEPS
from noeira.physics3d.gpu.constants import METADATA_SIZE, META_IDX_STEP_COUNT
from noeira.nn.constants import DT

from g1_walk_policy import WalkActor, WalkCritic, WALK_OBS, WALK_ACTION_SCALE


comptime N_ENVS = 1024
comptime ROLLOUT = 24
comptime N_MINIBATCHES = 4
comptime MINIBATCH = N_ENVS * ROLLOUT // N_MINIBATCHES
comptime EPOCHS = 5
comptime HIST = 1
comptime ENV_GRAPH = is_defined["PPO_VEC_ENV_GRAPH"]()
comptime TRAIN_GRAPH = is_defined["PPO_VEC_TRAIN_GRAPH"]()
comptime EnvT = UnitreeG1WalkBatched[N_ENVS, True]
comptime ACT = EnvT.ACT_DIM
comptime OBS = EnvT.OBS_DIM + HIST * ACT
# One definition of the networks: the eval and the room load them from here.
comptime Actor = WalkActor
comptime Critic = WalkCritic


def _arg(name: String, default: String) raises -> String:
    var a = argv()
    for i in range(len(a) - 1):
        if String(a[i]) == name:
            return String(a[i + 1])
    return default


def main() raises:
    comptime assert OBS == WALK_OBS, "g1_walk_policy.WALK_OBS disagrees with the env"
    var steps = Int(_arg("--steps", "200000000"))
    var seed = Int(_arg("--seed", "1"))
    var lr = Float64(_arg("--lr", "3e-4"))
    var ent = Float64(_arg("--ent", "0.005"))
    var act_rate = Float64(_arg("--act-rate", "1e-4"))
    var log_std = Float64(_arg("--log-std", "-1.0"))
    var ckpt_every = Int(_arg("--ckpt-every", "20000000"))
    var out = _arg("--out", "runs/g1_walk_s" + String(seed))
    if not exists(out):
        makedirs(out)
    seed_rng(seed)
    print("PPO (run_ppo_vec) on the G1 walker | lanes", N_ENVS, "| obs", OBS,
          "| act", ACT, "| rollout", ROLLOUT, "| minibatch", MINIBATCH, "x",
          N_MINIBATCHES, "| epochs", EPOCHS, "| lr", lr, "| ent", ent,
          "| act-rate", act_rate, "| log-std", log_std, "| env graph",
          ENV_GRAPH, "| train graph", TRAIN_GRAPH, "| steps", steps,
          "| seed", seed, "| out", out)
    with DeviceContext() as ctx:
        var agent = PPOAgent[
            "gpu", Actor, Critic, OBS, ACT, ROLLOUT, MINIBATCH, EPOCHS, N_ENVS
        ](
            ctx=ctx, actor_lr=Scalar[DT](lr), critic_lr=Scalar[DT](lr),
            gamma=0.99, gae_lambda=0.95, clip_eps=0.2,
            entropy_coef=Scalar[DT](ent),
            action_scale=Scalar[DT](WALK_ACTION_SCALE),
            log_std_init=Scalar[DT](log_std), max_grad_norm=1.0,
        )
        # ⚠ `log_std_init` ABOVE IS NOT APPLIED: `PPOTrainer.make` drops it
        # (`_ = log_std_init`, "the caller's responsibility"). Runs s1-s6
        # (2026-10-05) all trained at the GaussianHead's default 0 whatever
        # `--log-std` said. The head is the actor's 7th child.
        agent.trainer.actor.children[6].set_log_std_init["gpu"](
            Scalar[DT](log_std), ctx
        )
        var env = EnvT(ctx)
        env.reset_batch[N_ENVS](ctx=ctx, rng_seed=UInt64(seed))
        ctx.synchronize()
        # stagger the episode clocks
        env.d.meta.download(ctx)
        ctx.synchronize()
        for e in range(N_ENVS):
            env.d.meta.data[e * METADATA_SIZE + META_IDX_STEP_COUNT] = Scalar[DT](
                Int(random_float64() * Float64(G1_WALK_MAX_STEPS - 1))
            )
        env.d.meta.upload(ctx)
        ctx.synchronize()
        var rms = RunningMeanStd(OBS)
        var logger = CsvLogger(out + "/metrics.csv")
        var cfg = PPOVecConfig(
            total_steps=steps, lr=lr, ent0=ent, ent1=ent, anneal=False,
            act_rate_w=act_rate, seed=seed, ckpt_every=ckpt_every,
            print_every=20,
        )
        var res = run_ppo_vec[HIST=HIST, ENV_GRAPH=ENV_GRAPH, TRAIN_GRAPH=TRAIN_GRAPH](
            agent, env, ctx, cfg, rms, logger, out + "/ckpt",
            out + "/obs_norm.txt",
        )
        print("RESULT steps", res.steps, "| seconds", res.seconds,
              "| episodes", res.episodes, "| return (last 100)", res.mean_return,
              "| length", res.mean_length, "| diverged", res.diverged,
              "|", Int(Float64(res.steps) / res.seconds), "steps/s")
