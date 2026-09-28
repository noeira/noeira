"""CleanRL PPO (discrete) on LunarLander-v3 — wall clock to a mean return >= 200.

PyTorch reference for `examples/lunar_lander/ppo_lunar_lander_time_to_solve.mojo`.
It is CleanRL's `ppo.py` (references/RL-Algorithms/cleanrl-master/cleanrl/ppo.py)
with the SAME hyper-parameters as the Mojo run (RL Baselines3 Zoo's tuned PPO
LunarLander set) and these changes:

  * defaults: 16 envs x 1024 steps, 256 minibatches of 64, 4 epochs, gamma
    0.999, GAE lambda 0.98, lr 3e-4 constant (`anneal_lr` off), no value-loss
    clipping, entropy 0.01, grad-norm clip 0.5 — what noeira's PPO implements;
  * Gymnasium 1.x: `AutoresetMode.SAME_STEP` (the pre-1.0 behaviour `ppo.py`
    was written for), episode returns summed here instead of parsed from
    `final_info`, and a time-limit truncation bootstraps with
    `gamma * V(final_obs)` (SB3's handling, and noeira's) instead of being
    treated as a terminal;
  * stops the first time the mean return over the last 100 episodes reaches
    200, prints a `RESULT` line in the Mojo example's format, then a greedy
    20-episode eval and an optional `--save-model` state_dict;
  * no tensorboard / wandb / tyro.

The clock starts after the envs and the model are built, like the Mojo run.

    python benchmarks/cleanrl_ppo_lunar_lander.py --seed 1            # cuda if available
    python benchmarks/cleanrl_ppo_lunar_lander.py --seed 1 --no-cuda  # cpu
"""

import argparse
import random
import time
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--no-cuda", action="store_true")
    p.add_argument("--torch-deterministic", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--env-id", default="LunarLander-v3")
    p.add_argument("--total-timesteps", type=int, default=3_000_000)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--num-steps", type=int, default=1024)
    p.add_argument("--anneal-lr", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--gamma", type=float, default=0.999)
    p.add_argument("--gae-lambda", type=float, default=0.98)
    p.add_argument("--num-minibatches", type=int, default=256)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--norm-adv", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--clip-vloss", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--target-kl", type=float, default=None)
    p.add_argument("--target-return", type=float, default=200.0)
    p.add_argument("--window", type=int, default=100)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--save-model", default="")
    args = p.parse_args()
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size
    return args


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, n_actions), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


def greedy_eval(agent, env_id, episodes, seed, device):
    env = gym.make(env_id)
    returns = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, total = False, 0.0
        while not done:
            with torch.no_grad():
                logits = agent.actor(torch.tensor(obs, dtype=torch.float32, device=device))
            obs, r, term, trunc, _ = env.step(int(logits.argmax()))
            total += float(r)
            done = term or trunc
        returns.append(total)
    env.close()
    return float(np.mean(returns))


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [lambda: gym.make(args.env_id) for _ in range(args.num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    agent = Agent(obs_dim, int(envs.single_action_space.n)).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    print("=" * 70)
    print(f"CleanRL PPO {args.env_id} — time to mean return >= {args.target_return:.0f}")
    print(f"  torch {torch.__version__} | gymnasium {gym.__version__} | device {device}"
          + (f" ({torch.cuda.get_device_name()})" if device.type == "cuda" else ""))
    print(f"  seed {args.seed} | envs {args.num_envs} x rollout {args.num_steps} | minibatch "
          f"{args.minibatch_size} | epochs {args.update_epochs} | budget {args.total_timesteps}")
    print("=" * 70)

    obs = torch.zeros((args.num_steps, args.num_envs, obs_dim)).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs)).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    ep_ret = np.zeros(args.num_envs)
    recent = deque(maxlen=args.window)
    episodes = 0
    solved = False

    global_step = 0
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    start_time = time.time()

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs_np, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            done_np = np.logical_or(terminations, truncations)
            ep_ret += reward
            reward = reward.astype(np.float32)

            # Time-limit truncation: bootstrap from the pre-reset final obs.
            boot = np.nonzero(truncations & ~terminations)[0]
            if len(boot):
                final = np.stack([infos["final_obs"][i] for i in boot])
                with torch.no_grad():
                    v = agent.get_value(torch.Tensor(final).to(device)).flatten().cpu().numpy()
                reward[boot] += args.gamma * v

            for i in np.nonzero(done_np)[0]:
                recent.append(ep_ret[i])
                ep_ret[i] = 0.0
                episodes += 1
                if len(recent) == args.window and np.mean(recent) >= args.target_return:
                    solved = True

            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs_np).to(device)
            next_done = torch.Tensor(done_np.astype(np.float32)).to(device)
            if solved:
                break
        if solved:
            break

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        b_obs = obs.reshape((-1, obs_dim))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(args.batch_size)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                mb_inds = b_inds[start:start + args.minibatch_size]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef)
                    v_loss = 0.5 * torch.max(v_loss_unclipped, (v_clipped - b_returns[mb_inds]) ** 2).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                loss = pg_loss - args.ent_coef * entropy.mean() + v_loss * args.vf_coef
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        mean_recent = float(np.mean(recent)) if recent else float("nan")
        print(f"step {global_step} | episodes {episodes} | mean return (last {len(recent)}) "
              f"{mean_recent:.1f} | SPS {int(global_step / (time.time() - start_time))}", flush=True)

    if device.type == "cuda":
        torch.cuda.synchronize()
    wall_s = time.time() - start_time
    envs.close()
    mean_recent = float(np.mean(recent)) if recent else float("nan")
    if solved:
        print(f"target mean return reached: {mean_recent:.1f} | step {global_step} | episodes {episodes}")
    eval_mean = greedy_eval(agent, args.env_id, args.eval_episodes, 10_000 + args.seed, device)
    print("=" * 70)
    print(f"RESULT backend=CleanRL-{device.type} seed={args.seed} solved={solved} wall_s={wall_s:.3f} "
          f"env_steps={global_step} episodes={episodes} mean_return_last100={mean_recent:.2f} "
          f"greedy_eval_mean_{args.eval_episodes}ep={eval_mean:.2f}")
    if args.save_model:
        torch.save(agent.state_dict(), args.save_model)
        print("checkpoint:", args.save_model)
    print("=" * 70)


if __name__ == "__main__":
    main()
