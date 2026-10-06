"""CleanRL PPO (discrete) on LunarLander-v3, tuned PyTorch: torch.compile + CUDA graphs.

The best-effort PyTorch twin of `cleanrl_ppo_lunar_lander.py` (same agent,
hyper-parameters, Gymnasium envs and stopping rule), restructured the way
LeanRL (github.com/pytorch-labs/LeanRL, `ppo_continuous_action_torchcompile.py`)
restructures CleanRL so that it can be compiled and captured:

  * the policy step (actor + critic + sample) is one function returning one
    [3, num_envs] tensor, read back with ONE device->host copy per env step;
    the rollout is recorded in host numpy arrays and GAE runs there;
  * the rollout is uploaded once per update; each minibatch step (gather,
    loss, backward, grad clip, Adam) is ONE function with no host sync,
    distribution argument validation off;
  * `--compile`: torch.compile (default mode) on the policy step and the
    minibatch step;
  * `--cudagraphs`: each of them captured in one CUDA graph (capturable
    Adam), replayed every call;
  * `--async-envs`: the 16 Box2D envs in subprocesses (`AsyncVectorEnv`)
    instead of one process (`SyncVectorEnv`, CleanRL's default).

With neither --compile nor --cudagraphs it is the restructured script run
eagerly. Only the meeting's settings are supported (no value-loss clipping,
no target KL, no lr annealing).

The clock covers the training loop, torch.compile's JIT and the graph
capture included (`jit_s` reports the first calls, which hold them).

    python benchmarks/cleanrl_ppo_lunar_lander_compiled.py --seed 1 --compile --cudagraphs
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

from cleanrl_ppo_lunar_lander import Agent, greedy_eval
from torch_cuda_graph import CudaGraphed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--env-id", default="LunarLander-v3")
    p.add_argument("--total-timesteps", type=int, default=3_000_000)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--num-envs", type=int, default=16)
    p.add_argument("--num-steps", type=int, default=1024)
    p.add_argument("--gamma", type=float, default=0.999)
    p.add_argument("--gae-lambda", type=float, default=0.98)
    p.add_argument("--num-minibatches", type=int, default=256)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--target-return", type=float, default=200.0)
    p.add_argument("--window", type=int, default=100)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--cudagraphs", action="store_true")
    p.add_argument("--async-envs", action="store_true")
    p.add_argument("--save-model", default="")
    args = p.parse_args()
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    args.num_iterations = args.total_timesteps // args.batch_size
    return args


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.distributions.Distribution.set_default_validate_args(False)
    device = torch.device("cuda")
    mode = "+".join(m for m, on in (("compile", args.compile), ("cudagraphs", args.cudagraphs)) if on) or "eager"
    if args.async_envs:
        mode += "+async"

    vec = gym.vector.AsyncVectorEnv if args.async_envs else gym.vector.SyncVectorEnv
    envs = vec([lambda: gym.make(args.env_id) for _ in range(args.num_envs)],
               autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    agent = Agent(obs_dim, int(envs.single_action_space.n)).to(device)
    params = list(agent.parameters())
    optimizer = optim.Adam(params, lr=args.learning_rate, eps=1e-5, capturable=args.cudagraphs)

    N, E, MB = args.num_steps, args.num_envs, args.minibatch_size
    obs_h = np.zeros((N, E, obs_dim), np.float32)
    out_h = np.zeros((N, 3, E), np.float32)          # action, logprob, value
    rew_h = np.zeros((N, E), np.float32)
    done_h = np.zeros((N, E), np.float32)
    b_obs = torch.zeros((N * E, obs_dim), device=device)
    b_act = torch.zeros(N * E, dtype=torch.long, device=device)
    b_logp = torch.zeros(N * E, device=device)
    b_adv = torch.zeros(N * E, device=device)
    b_ret = torch.zeros(N * E, device=device)
    perm = torch.zeros((args.num_minibatches, MB), dtype=torch.long, device=device)

    def policy(x):
        logits = agent.actor(x)
        probs = Categorical(logits=logits)
        a = probs.sample()
        return torch.stack([a.float(), probs.log_prob(a), agent.critic(x).view(-1)])

    def train_step(mb):
        o, a = b_obs[mb], b_act[mb]
        probs = Categorical(logits=agent.actor(o))
        newlogprob, entropy = probs.log_prob(a), probs.entropy()
        newvalue = agent.critic(o).view(-1)
        ratio = (newlogprob - b_logp[mb]).exp()
        mb_adv = b_adv[mb]
        mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
        pg_loss = torch.max(-mb_adv * ratio,
                            -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)).mean()
        v_loss = 0.5 * ((newvalue - b_ret[mb]) ** 2).mean()
        loss = pg_loss - args.ent_coef * entropy.mean() + v_loss * args.vf_coef
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(params, args.max_grad_norm)
        optimizer.step()
        return loss.detach()

    policy_fn, train_fn = policy, train_step
    if args.compile:
        policy_fn, train_fn = torch.compile(policy_fn), torch.compile(train_fn)
    if args.cudagraphs:
        policy_fn, train_fn = CudaGraphed(policy_fn), CudaGraphed(train_fn)

    print("=" * 70)
    print(f"CleanRL PPO {args.env_id} [{mode}] — time to mean return >= {args.target_return:.0f}")
    print(f"  torch {torch.__version__} | gymnasium {gym.__version__} | device {device} "
          f"({torch.cuda.get_device_name()})")
    print(f"  seed {args.seed} | envs {E} x rollout {N} | minibatch {MB} | epochs "
          f"{args.update_epochs} | budget {args.total_timesteps}")
    print("=" * 70)

    ep_ret = np.zeros(E)
    recent = deque(maxlen=args.window)
    episodes, solved, global_step = 0, False, 0
    jit_s, t_env, t_update, n_policy, n_train = 0.0, 0.0, 0.0, 0, 0
    next_obs, _ = envs.reset(seed=args.seed)
    next_done = np.zeros(E, np.float32)
    start_time = time.time()

    for iteration in range(1, args.num_iterations + 1):
        for step in range(N):
            global_step += E
            obs_h[step] = next_obs
            done_h[step] = next_done
            t0 = time.time()
            x = torch.from_numpy(obs_h[step])
            with torch.no_grad():
                out = policy_fn(x if args.cudagraphs else x.to(device))
            out_h[step] = out.cpu().numpy()
            if n_policy < 4:
                jit_s += time.time() - t0
            n_policy += 1

            t0 = time.time()
            next_obs, reward, terminations, truncations, infos = envs.step(out_h[step, 0].astype(np.int64))
            t_env += time.time() - t0
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

            rew_h[step] = reward
            next_done = done_np.astype(np.float32)
            if solved:
                break
        if solved:
            break

        t0 = time.time()
        with torch.no_grad():
            next_value = agent.get_value(torch.from_numpy(next_obs).to(device)).view(-1).cpu().numpy()
        values = out_h[:, 2]
        adv = np.zeros((N, E), np.float32)
        lastgaelam = np.zeros(E, np.float32)
        for t in reversed(range(N)):
            if t == N - 1:
                nextnonterminal, nextvalues = 1.0 - next_done, next_value
            else:
                nextnonterminal, nextvalues = 1.0 - done_h[t + 1], values[t + 1]
            delta = rew_h[t] + args.gamma * nextvalues * nextnonterminal - values[t]
            adv[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
        ret = adv + values

        b_obs.copy_(torch.from_numpy(obs_h.reshape(-1, obs_dim)))
        b_act.copy_(torch.from_numpy(out_h[:, 0].reshape(-1).astype(np.int64)))
        b_logp.copy_(torch.from_numpy(out_h[:, 1].reshape(-1)))
        b_adv.copy_(torch.from_numpy(adv.reshape(-1)))
        b_ret.copy_(torch.from_numpy(ret.reshape(-1)))
        b_inds = np.arange(args.batch_size)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            perm.copy_(torch.from_numpy(b_inds.reshape(args.num_minibatches, MB)))
            for k in range(args.num_minibatches):
                t1 = time.time()
                train_fn(perm[k])
                if n_train < 4:
                    torch.cuda.synchronize()
                    jit_s += time.time() - t1
                n_train += 1
        torch.cuda.synchronize()
        t_update += time.time() - t0

        mean_recent = float(np.mean(recent)) if recent else float("nan")
        print(f"step {global_step} | episodes {episodes} | mean return (last {len(recent)}) "
              f"{mean_recent:.1f} | SPS {int(global_step / (time.time() - start_time))}", flush=True)

    torch.cuda.synchronize()
    wall_s = time.time() - start_time
    envs.close()
    mean_recent = float(np.mean(recent)) if recent else float("nan")
    if solved:
        print(f"target mean return reached: {mean_recent:.1f} | step {global_step} | episodes {episodes}")
    eval_mean = greedy_eval(agent, args.env_id, args.eval_episodes, 10_000 + args.seed, device)
    print("=" * 70)
    print(f"RESULT backend=CleanRL-cuda-{mode} seed={args.seed} solved={solved} wall_s={wall_s:.3f} "
          f"env_steps={global_step} sps={int(global_step / wall_s)} jit_s={jit_s:.1f} "
          f"env_s={t_env:.1f} update_s={t_update:.1f} episodes={episodes} "
          f"mean_return_last100={mean_recent:.2f} greedy_eval_mean_{args.eval_episodes}ep={eval_mean:.2f}")
    if args.save_model:
        torch.save(agent.state_dict(), args.save_model)
        print("checkpoint:", args.save_model)
    print("=" * 70)


if __name__ == "__main__":
    main()
