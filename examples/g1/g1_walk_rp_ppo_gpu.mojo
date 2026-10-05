"""PPO on the G1 walker with RoboParty's recipe (G1_WALKER_PLAN §10, L0d).

    pixi run -e nvidia mojo build -I . -D PPO_VEC_ENV_GRAPH -D PPO_VEC_TRAIN_GRAPH \\
        examples/g1/g1_walk_rp_ppo_gpu.mojo -o g1_walk_rp_ppo
    pixi run -e nvidia ./g1_walk_rp_ppo --steps 60000000 --seed 1 --out runs/rp_s1
    pixi run -e nvidia ./g1_walk_rp_ppo ... --init runs/rp_s1 --out runs/rp_s1b   # resume

`--init` loads the actor + critic and the observation statistics (Adam's
moments are not checkpointed; the step count restarts at 0).

The env is `UnitreeG1WalkRPBatched` (`unitree_g1_walk_rp_config.mojo`: their
26 reward terms, terminations, resets, pushes, gain DR). The policy's
observation is `run_ppo_vec`'s frame history:

    [10 x (actor 67 + last action 29) | 10 x privileged 15]      = 1110

and the ACTOR reads only the first 960 words (`Slice` at its head) — the
privileged critic without a trainer change. Their PPO (`rpo_interrupt_agent_cfg.py`):
lr 1e-3 adaptive (KL 0.01), gamma 0.99, lambda 0.95, clip 0.2, entropy
0.005, 5 epochs x 4 minibatches of 24-step rollouts, grad norm 1.0, init
noise std 1.0, observation normalisation on, reward normalisation OFF
(rsl_rl has none), action rate and smoothness at -0.02 / s each.

Their networks: 512-256-128 ELU, actor and critic. The mirror loss is on at
their 0.2 (`--mirror`, `deep_agents/ppo/mirror.mojo`; maps `rp_mirror_maps`).
Not theirs yet: value-loss clipping, mirror DATA AUGMENTATION (§10 rows 5, 7).
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
from noeira.envs.robots.unitree_g1_walk_rp import UnitreeG1WalkRPBatched
from noeira.envs.robots.unitree_g1_walk_rp_config import (
    G1R_MAX_STEPS, G1R_OBS_ACTOR, G1R_ACTION_CLIP,
)
from noeira.physics3d.gpu.constants import METADATA_SIZE, META_IDX_STEP_COUNT
from noeira.nn.constants import DT

from g1_walk_rp_policy import (
    RPActor, RPCritic, RP_OBS, RP_FRAMES, RP_HIST, RP_HEAD_CHILD, rp_mirror_maps,
    RP_EA, RP_ACTOR_LINVEL,
)


comptime N_ENVS = 1024
comptime ROLLOUT = 24
comptime N_MINIBATCHES = 4
comptime MINIBATCH = N_ENVS * ROLLOUT // N_MINIBATCHES
comptime EPOCHS = 5
comptime ENV_GRAPH = is_defined["PPO_VEC_ENV_GRAPH"]()
comptime TRAIN_GRAPH = is_defined["PPO_VEC_TRAIN_GRAPH"]()
comptime EnvT = UnitreeG1WalkRPBatched[N_ENVS, True]
comptime ACT = EnvT.ACT_DIM


def _arg(name: String, default: String) raises -> String:
    var a = argv()
    for i in range(len(a) - 1):
        if String(a[i]) == name:
            return String(a[i + 1])
    return default


def main() raises:
    var steps = Int(_arg("--steps", "60000000"))
    var seed = Int(_arg("--seed", "1"))
    var lr = Float64(_arg("--lr", "1e-3"))
    var ent = Float64(_arg("--ent", "0.005"))
    var act_rate = Float64(_arg("--act-rate", "4e-4"))
    var act_smooth = Float64(_arg("--act-smooth", "4e-4"))
    var log_std = Float64(_arg("--log-std", "0.0"))
    var adaptive = _arg("--adaptive-kl", "1") == "1"
    var mirror = Float64(_arg("--mirror", "0.2"))
    var norm_rew = _arg("--norm-reward", "0") == "1"
    var rew_bound = Float64(_arg("--rew-bound", "1000"))
    var ckpt_every = Int(_arg("--ckpt-every", "20000000"))
    var out = _arg("--out", "runs/g1_walk_rp_s" + String(seed))
    var init = _arg("--init", "")
    if not exists(out):
        makedirs(out)
    seed_rng(seed)
    print("PPO (run_ppo_vec) on the G1 walker, RoboParty recipe | lanes", N_ENVS,
          "| obs", RP_OBS, "(actor", RP_FRAMES * (RP_EA + RP_HIST * ACT),
          ", linvel in actor", RP_ACTOR_LINVEL,
          ") | act", ACT, "| frames", RP_FRAMES, "| rollout", ROLLOUT,
          "| minibatch", MINIBATCH, "x", N_MINIBATCHES, "| epochs", EPOCHS,
          "| lr", lr, "adaptive", adaptive, "| ent", ent, "| act-rate", act_rate,
          "| act-smooth", act_smooth, "| mirror", mirror, "| norm-reward", norm_rew,
          "| rew-bound", rew_bound, "| log-std", log_std, "| graphs",
          ENV_GRAPH, TRAIN_GRAPH, "| steps", steps, "| seed", seed, "| out", out)
    with DeviceContext() as ctx:
        var agent = PPOAgent[
            "gpu", RPActor, RPCritic, RP_OBS, ACT, ROLLOUT, MINIBATCH, EPOCHS, N_ENVS
        ](
            ctx=ctx, actor_lr=Scalar[DT](lr), critic_lr=Scalar[DT](lr),
            gamma=0.99, gae_lambda=0.95, clip_eps=0.2,
            entropy_coef=Scalar[DT](ent),
            action_scale=Scalar[DT](G1R_ACTION_CLIP), max_grad_norm=1.0,
        )
        # ⚠ `log_std_init` is NOT applied by `make`; set it on the head
        agent.trainer.actor.children[RP_HEAD_CHILD].set_log_std_init["gpu"](
            Scalar[DT](log_std), ctx
        )
        if init.byte_length() > 0:
            # after the log-std: the checkpoint's own log-std wins
            # (`load_state` writes resident — the optimiser's arena keeps
            # the buffers it adopted)
            agent.trainer.load_state(init + "/ckpt")
            print("  init: actor + critic from", init + "/ckpt")
        if mirror > 0.0:
            var mm = rp_mirror_maps()
            agent.trainer.actor_train.inner.enable_mirror["gpu"](
                mm[0], mm[1], mm[2], mm[3], mirror, ctx
            )
        var env = EnvT(ctx)
        env.reset_batch[N_ENVS](ctx=ctx, rng_seed=UInt64(seed))
        ctx.synchronize()
        env.d.meta.download(ctx)
        ctx.synchronize()
        for e in range(N_ENVS):
            env.d.meta.data[e * METADATA_SIZE + META_IDX_STEP_COUNT] = Scalar[DT](
                Int(random_float64() * Float64(G1R_MAX_STEPS - 1))
            )
        env.d.meta.upload(ctx)
        ctx.synchronize()
        var rms = RunningMeanStd(RP_OBS)
        if init.byte_length() > 0:
            rms.load(init + "/obs_norm.txt")
        var logger = CsvLogger(out + "/metrics.csv")
        var cfg = PPOVecConfig(
            total_steps=steps, lr=lr, ent0=ent, ent1=ent, anneal=False,
            norm_obs=True, norm_reward=norm_rew, rew_bound=rew_bound,
            keep_ckpts=True,
            act_rate_w=act_rate, act_smooth_w=act_smooth,
            adaptive_kl=adaptive, desired_kl=0.01,
            seed=seed, ckpt_every=ckpt_every, print_every=20,
        )
        var res = run_ppo_vec[
            HIST=RP_HIST, ENV_GRAPH=ENV_GRAPH, TRAIN_GRAPH=TRAIN_GRAPH,
            FRAMES=RP_FRAMES, E_ACTOR=RP_EA,
        ](agent, env, ctx, cfg, rms, logger, out + "/ckpt", out + "/obs_norm.txt")
        print("RESULT steps", res.steps, "| seconds", res.seconds,
              "| episodes", res.episodes, "| return (last 100)", res.mean_return,
              "| length", res.mean_length, "| diverged", res.diverged,
              "|", Int(Float64(res.steps) / res.seconds), "steps/s")
