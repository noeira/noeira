"""CleanRL SAC on HalfCheetah — wall clock for a fixed env-step budget.

PyTorch reference for `examples/half_cheetah/sac_half_cheetah_benchmark.mojo`.
It is CleanRL's `sac_continuous_action.py`
(references/RL-Algorithms/cleanrl-master/cleanrl/sac_continuous_action.py)
with its default hyper-parameters and these changes:

  * `--policy-frequency 1` by default: noeira's SAC updates the actor every
    step (CleanRL's default 2 delays it and does 2 updates every 2 steps,
    the same count);
  * one gymnasium env (CleanRL's `num_envs` is 1 too) instead of a
    SyncVectorEnv, so the final observation of a truncated episode is the
    one `step` returned; terminations are stored as dones, truncations
    bootstrap (CleanRL's `handle_timeout_termination=False` + terminations);
  * a numpy ring buffer instead of `cleanrl_utils.buffers.ReplayBuffer`
    (same sampling: uniform over the filled part);
  * a fixed env-step budget, a `RESULT` line in the Mojo example's format and
    a 10-episode deterministic eval (the tanh-squashed mean) at the end;
  * no tensorboard / wandb / tyro.

The clock starts after the env, the networks and the buffer are built.

    python benchmarks/cleanrl_sac_half_cheetah.py --seed 1            # cuda
    python benchmarks/cleanrl_sac_half_cheetah.py --seed 1 --no-cuda  # cpu
"""

import argparse
import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--no-cuda", action="store_true")
    p.add_argument("--torch-deterministic", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--env-id", default="HalfCheetah-v5")
    p.add_argument("--total-timesteps", type=int, default=200_000)
    p.add_argument("--buffer-size", type=int, default=int(1e6))
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-starts", type=int, default=5_000)
    p.add_argument("--policy-lr", type=float, default=3e-4)
    p.add_argument("--q-lr", type=float, default=1e-3)
    p.add_argument("--policy-frequency", type=int, default=1)
    p.add_argument("--target-network-frequency", type=int, default=1)
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--autotune", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--print-every", type=int, default=10_000)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--save-model", default="")
    return p.parse_args()


class SoftQNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim + act_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, low, high):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, act_dim)
        self.fc_logstd = nn.Linear(256, act_dim)
        self.register_buffer("action_scale", torch.tensor((high - low) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor((high + low) / 2.0, dtype=torch.float32))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = torch.tanh(self.fc_logstd(x))
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


class ReplayBuffer:
    def __init__(self, size, obs_dim, act_dim, device):
        self.obs = np.zeros((size, obs_dim), np.float32)
        self.next_obs = np.zeros((size, obs_dim), np.float32)
        self.actions = np.zeros((size, act_dim), np.float32)
        self.rewards = np.zeros(size, np.float32)
        self.dones = np.zeros(size, np.float32)
        self.size, self.pos, self.full, self.device = size, 0, False, device

    def add(self, o, no, a, r, d):
        i = self.pos
        self.obs[i], self.next_obs[i], self.actions[i], self.rewards[i], self.dones[i] = o, no, a, r, d
        self.pos = (i + 1) % self.size
        self.full = self.full or self.pos == 0

    def sample(self, n):
        idx = np.random.randint(0, self.size if self.full else self.pos, size=n)
        t = lambda x: torch.as_tensor(x[idx], device=self.device)
        return t(self.obs), t(self.next_obs), t(self.actions), t(self.rewards), t(self.dones)


def evaluate(actor, env_id, episodes, seed, device):
    env = gym.make(env_id)
    returns = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, total = False, 0.0
        while not done:
            with torch.no_grad():
                _, _, mean = actor.get_action(torch.as_tensor(obs, dtype=torch.float32, device=device)[None])
            obs, r, term, trunc, _ = env.step(mean[0].cpu().numpy())
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

    env = gym.make(args.env_id)
    env.action_space.seed(args.seed)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(np.prod(env.action_space.shape))
    low, high = env.action_space.low, env.action_space.high

    actor = Actor(obs_dim, act_dim, low, high).to(device)
    qf1 = SoftQNetwork(obs_dim, act_dim).to(device)
    qf2 = SoftQNetwork(obs_dim, act_dim).to(device)
    qf1_target = SoftQNetwork(obs_dim, act_dim).to(device)
    qf2_target = SoftQNetwork(obs_dim, act_dim).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)
    if args.autotune:
        target_entropy = -float(act_dim)
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha
    rb = ReplayBuffer(args.buffer_size, obs_dim, act_dim, device)

    print("=" * 70)
    print(f"CleanRL SAC {args.env_id} — wall clock for {args.total_timesteps} env steps")
    print(f"  torch {torch.__version__} | gymnasium {gym.__version__} | device {device}"
          + (f" ({torch.cuda.get_device_name()})" if device.type == "cuda" else ""))
    print(f"  seed {args.seed} | batch {args.batch_size} | learning_starts {args.learning_starts} "
          f"| policy_frequency {args.policy_frequency}")
    print("=" * 70)

    obs, _ = env.reset(seed=args.seed)
    ep_ret, recent = 0.0, []
    start_time = time.time()
    for global_step in range(args.total_timesteps):
        if global_step < args.learning_starts:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                a, _, _ = actor.get_action(torch.as_tensor(obs, dtype=torch.float32, device=device)[None])
            action = a[0].cpu().numpy()
        next_obs, reward, term, trunc, _ = env.step(action)
        rb.add(obs, next_obs, action, reward, float(term))
        ep_ret += float(reward)
        obs = next_obs
        if term or trunc:
            recent = (recent + [ep_ret])[-10:]
            ep_ret = 0.0
            obs, _ = env.reset()

        if global_step > args.learning_starts:
            o, no, a, r, d = rb.sample(args.batch_size)
            with torch.no_grad():
                next_a, next_log_pi, _ = actor.get_action(no)
                min_q_next = torch.min(qf1_target(no, next_a), qf2_target(no, next_a)) - alpha * next_log_pi
                next_q = r + (1 - d) * args.gamma * min_q_next.view(-1)
            qf1_a = qf1(o, a).view(-1)
            qf2_a = qf2(o, a).view(-1)
            qf_loss = F.mse_loss(qf1_a, next_q) + F.mse_loss(qf2_a, next_q)
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                for _ in range(args.policy_frequency):
                    pi, log_pi, _ = actor.get_action(o)
                    actor_loss = ((alpha * log_pi) - torch.min(qf1(o, pi), qf2(o, pi))).mean()
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()
                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(o)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()
                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            if global_step % args.target_network_frequency == 0:
                for net, tgt in ((qf1, qf1_target), (qf2, qf2_target)):
                    for p_, tp in zip(net.parameters(), tgt.parameters()):
                        tp.data.copy_(args.tau * p_.data + (1 - args.tau) * tp.data)

        if (global_step + 1) % args.print_every == 0:
            el = time.time() - start_time
            mr = float(np.mean(recent)) if recent else float("nan")
            print(f"[step {global_step + 1}] mean_ret(10)= {mr:.1f} elapsed= {el:.1f} s "
                  f"SPS {int((global_step + 1) / el)} alpha {alpha:.3f}", flush=True)

    if device.type == "cuda":
        torch.cuda.synchronize()
    wall_s = time.time() - start_time
    env.close()
    train_mean = float(np.mean(recent)) if recent else float("nan")
    eval_mean = evaluate(actor, args.env_id, args.eval_episodes, 10_000 + args.seed, device)
    print("=" * 70)
    print(f"RESULT backend=CleanRL-{device.type} n_envs=1 seed={args.seed} env_steps={args.total_timesteps} "
          f"wall_s={wall_s:.3f} sps={int(args.total_timesteps / wall_s)} "
          f"train_return_last10={train_mean:.2f} greedy_eval_{args.eval_episodes}ep={eval_mean:.2f}")
    if args.save_model:
        torch.save(actor.state_dict(), args.save_model)
        print("checkpoint:", args.save_model)
    print("=" * 70)


if __name__ == "__main__":
    main()
