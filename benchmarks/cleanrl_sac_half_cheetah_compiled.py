"""CleanRL SAC on HalfCheetah, tuned PyTorch: torch.compile + CUDA graphs.

The best-effort PyTorch twin of `cleanrl_sac_half_cheetah.py` (same networks,
hyper-parameters, env loop and budget), restructured the way LeanRL
(github.com/pytorch-labs/LeanRL, `sac_continuous_action_torchcompile.py`)
restructures CleanRL so that it can be compiled and captured:

  * the replay buffer lives on the device: one row per transition
    (obs, next_obs, action, reward, done), written with one host->device copy
    per env step; the minibatch indices are still drawn by numpy on the host
    (as the eager script does) and gathered on the device;
  * the whole update (critic, actor, alpha, Polyak) is ONE function with no
    host sync: alpha stays a device tensor, Polyak is `torch._foreach_lerp_`,
    distribution argument validation is off;
  * `--compile`: torch.compile (default mode) on the update and the policy;
  * `--cudagraphs`: the update and the policy each captured in one CUDA graph
    (capturable Adam), replayed every step.

With neither flag it is the restructured script run eagerly, which separates
what the restructuring buys from what compile and graphs buy. Only
`--policy-frequency 1` and `--target-network-frequency 1` (the meeting's
settings) are supported.

The clock covers the training loop, torch.compile's JIT and the graph
capture included (`jit_s` reports the first calls, which hold them).

    python benchmarks/cleanrl_sac_half_cheetah_compiled.py --seed 1 --compile --cudagraphs
"""

import argparse
import random
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from cleanrl_sac_half_cheetah import Actor, SoftQNetwork, evaluate
from torch_cuda_graph import CudaGraphed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--env-id", default="HalfCheetah-v5")
    p.add_argument("--total-timesteps", type=int, default=200_000)
    p.add_argument("--buffer-size", type=int, default=int(1e6))
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-starts", type=int, default=5_000)
    p.add_argument("--policy-lr", type=float, default=3e-4)
    p.add_argument("--q-lr", type=float, default=1e-3)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--cudagraphs", action="store_true")
    p.add_argument("--print-every", type=int, default=10_000)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--save-model", default="")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.distributions.Distribution.set_default_validate_args(False)
    device = torch.device("cuda")
    mode = "+".join(m for m, on in (("compile", args.compile), ("cudagraphs", args.cudagraphs)) if on) or "eager"

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
    q_params = list(qf1.parameters()) + list(qf2.parameters())
    tgt_params = list(qf1_target.parameters()) + list(qf2_target.parameters())
    cap = args.cudagraphs
    q_optimizer = optim.Adam(q_params, lr=args.q_lr, capturable=cap)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr, capturable=cap)
    target_entropy = -float(act_dim)
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha = log_alpha.detach().exp()
    a_optimizer = optim.Adam([log_alpha], lr=args.q_lr, capturable=cap)

    width = 2 * obs_dim + act_dim + 2
    data = torch.zeros((args.buffer_size, width), device=device)
    o_, no_, a_ = slice(0, obs_dim), slice(obs_dim, 2 * obs_dim), slice(2 * obs_dim, 2 * obs_dim + act_dim)

    def update(idx):
        b = data[idx]
        o, no, a, r, d = b[:, o_], b[:, no_], b[:, a_], b[:, -2], b[:, -1]
        with torch.no_grad():
            next_a, next_log_pi, _ = actor.get_action(no)
            min_q_next = torch.min(qf1_target(no, next_a), qf2_target(no, next_a)) - alpha * next_log_pi
            next_q = r + (1 - d) * args.gamma * min_q_next.view(-1)
        qf_loss = F.mse_loss(qf1(o, a).view(-1), next_q) + F.mse_loss(qf2(o, a).view(-1), next_q)
        q_optimizer.zero_grad()
        qf_loss.backward()
        q_optimizer.step()

        pi, log_pi, _ = actor.get_action(o)
        actor_loss = ((alpha * log_pi) - torch.min(qf1(o, pi), qf2(o, pi))).mean()
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()
        with torch.no_grad():
            _, log_pi, _ = actor.get_action(o)
        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()
        a_optimizer.zero_grad()
        alpha_loss.backward()
        a_optimizer.step()
        with torch.no_grad():
            alpha.copy_(log_alpha.exp())
            torch._foreach_lerp_(tgt_params, q_params, args.tau)
        return qf_loss.detach(), actor_loss.detach()

    def policy(obs):
        a, _, _ = actor.get_action(obs)
        return a

    update_fn, policy_fn = update, policy
    if args.compile:
        update_fn, policy_fn = torch.compile(update_fn), torch.compile(policy_fn)
    if args.cudagraphs:
        update_fn, policy_fn = CudaGraphed(update_fn), CudaGraphed(policy_fn)

    print("=" * 70)
    print(f"CleanRL SAC {args.env_id} [{mode}] — wall clock for {args.total_timesteps} env steps")
    print(f"  torch {torch.__version__} | gymnasium {gym.__version__} | device {device} "
          f"({torch.cuda.get_device_name()})")
    print(f"  seed {args.seed} | batch {args.batch_size} | learning_starts {args.learning_starts} "
          f"| replay on the device")
    print("=" * 70)

    row = np.zeros(width, np.float32)
    obs, _ = env.reset(seed=args.seed)
    ep_ret, recent = 0.0, []
    jit_s, t_env, n_policy, n_update = 0.0, 0.0, 0, 0
    start_time = time.time()
    for global_step in range(args.total_timesteps):
        if global_step < args.learning_starts:
            action = env.action_space.sample()
        else:
            t0 = time.time()
            x = torch.as_tensor(obs, dtype=torch.float32)[None]
            with torch.no_grad():
                a = policy_fn(x if args.cudagraphs else x.to(device))
            action = a[0].cpu().numpy()
            if n_policy < 4:
                jit_s += time.time() - t0
            n_policy += 1
        t0 = time.time()
        next_obs, reward, term, trunc, _ = env.step(action)
        t_env += time.time() - t0
        row[o_], row[no_], row[a_], row[-2], row[-1] = obs, next_obs, action, reward, float(term)
        data[global_step].copy_(torch.from_numpy(row))
        ep_ret += float(reward)
        obs = next_obs
        if term or trunc:
            recent = (recent + [ep_ret])[-10:]
            ep_ret = 0.0
            t0 = time.time()
            obs, _ = env.reset()
            t_env += time.time() - t0

        if global_step > args.learning_starts:
            t0 = time.time()
            idx = torch.from_numpy(np.random.randint(0, global_step + 1, size=args.batch_size))
            update_fn(idx if args.cudagraphs else idx.to(device))
            if n_update < 4:
                torch.cuda.synchronize()
                jit_s += time.time() - t0
            n_update += 1

        if (global_step + 1) % args.print_every == 0:
            el = time.time() - start_time
            mr = float(np.mean(recent)) if recent else float("nan")
            print(f"[step {global_step + 1}] mean_ret(10)= {mr:.1f} elapsed= {el:.1f} s "
                  f"SPS {int((global_step + 1) / el)} alpha {alpha.item():.3f}", flush=True)

    torch.cuda.synchronize()
    wall_s = time.time() - start_time
    env.close()
    train_mean = float(np.mean(recent)) if recent else float("nan")
    eval_mean = evaluate(actor, args.env_id, args.eval_episodes, 10_000 + args.seed, device)
    print("=" * 70)
    print(f"RESULT backend=CleanRL-cuda-{mode} n_envs=1 seed={args.seed} env_steps={args.total_timesteps} "
          f"wall_s={wall_s:.3f} sps={int(args.total_timesteps / wall_s)} jit_s={jit_s:.1f} "
          f"env_s={t_env:.1f} train_return_last10={train_mean:.2f} "
          f"greedy_eval_{args.eval_episodes}ep={eval_mean:.2f}")
    if args.save_model:
        torch.save(actor.state_dict(), args.save_model)
        print("checkpoint:", args.save_model)
    print("=" * 70)


if __name__ == "__main__":
    main()
