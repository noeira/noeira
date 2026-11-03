#!/usr/bin/env python3
# +--------------------------------------------------------------------------+ #
# | SuperMimic M5 oracle: the reproduction's imagined rollout + policy grad
# +--------------------------------------------------------------------------+ #
"""Reference tensors for `tests/deep_agents/supermimic/test_supermimic_imag.mojo`.

    uv run --no-project --python 3.12 --with torch --with mujoco==3.12.0 --with numpy \\
        python tools/supermimic/imag_oracle.py --out /tmp/sm_imag

The reproduction's `imagination/rollout.py` (`imagine`), `compiled.py`
(`ImagStep`), `objective.py` (`policy_objective`, `gae_targets`),
`policy/nets.py` (`Policy`, `Critic`) and the world model, VERBATIM in torch
float64, then `loss = −J / H` backpropagated into the policy (the trainer's
actor half, `policy_trainer.py`).

What is NOT the reproduction's: the motion library and the replay start
batch. `imagine` reads a library through four calls (`goal_window`,
`last_frame_of`, `ref`, `bodies`) and `wm_trainer.reference_at`; a small
random library of two clips stands in, its body states from the
reproduction's own FK on our model. Its replay buffer / library / distributed
modules are stubbed by name so their imports (mjlab's env stack) never run.

Weights and inputs are rounded to float32 first (the Mojo side is float32).
`obs_noise=False`: the noise is data, not part of the gradient path.
Files `i_*`; `i_names.txt` lists every net's params in OUR walk order
(weights transposed to [IN, OUT]).
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path

import mujoco
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
N, H = 8, 32
HW, TH = 32, 16
ENC = (24, 20, 16, 12)
DEC = (28, 22, 18, 14)
CRIT = (16, 12)
GAMMA = 0.99
L_CLIP = 60


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def f32(x):
    return np.asarray(x, dtype=np.float32).astype(np.float64)


def unit(q):
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def stub(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    parent, _, leaf = name.rpartition(".")
    if parent in sys.modules:
        setattr(sys.modules[parent], leaf, m)
    return m


class _NS:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=17)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    ho = _load_by_path("sm_heads_oracle", HERE / "heads_oracle.py")
    fkmod, hm = ho.reference_modules()
    R = ho.REF
    ho._pkg("supermimic.data")
    ho._pkg("supermimic.training")
    ho._pkg("supermimic.policy")
    ho._pkg("supermimic.imagination")
    stub("supermimic.data.replay_buffer", ReplayBuffer=object, WindowBatch=object, StartBatch=object)
    stub("supermimic.envs.mdp.motion_library", MotionLibrary=object)
    stub("supermimic.training.distributed", allreduce_grads=lambda *a, **k: None)
    sspec = ho._load("supermimic.world_model.state_spec", R / "world_model" / "state_spec.py")
    ho._load("supermimic.world_model.dynamics", R / "world_model" / "dynamics.py")
    wmmod = ho._load("supermimic.world_model.world_model", R / "world_model" / "world_model.py")
    wmt = ho._load("supermimic.world_model.wm_trainer", R / "world_model" / "wm_trainer.py")
    nets = ho._load("supermimic.policy.nets", R / "policy" / "nets.py")
    ho._load("supermimic.imagination.compiled", R / "imagination" / "compiled.py")
    rollout = ho._load("supermimic.imagination.rollout", R / "imagination" / "rollout.py")
    objective = ho._load("supermimic.imagination.objective", R / "imagination" / "objective.py")
    _ = sspec, wmt

    rng = np.random.default_rng(a.seed)
    m = mujoco.MjModel.from_xml_path(str(ho.G1_XML))
    fk = fkmod.G1Kinematics(m, dtype=torch.float64)
    tracked = ho._literal(R / "envs" / "tracking_env_cfg.py", "TRACKED_BODIES")
    q0 = f32(rng.uniform(-0.3, 0.3, 29))
    heads = hm.AnalyticHeads(fk, tracked, "torso_link", "pelvis", torch.tensor(ho.soft_limits(m)), torch.tensor(q0))

    # ── the random reference library: two clips of L_CLIP frames ──
    L = 2 * L_CLIP
    t_ = np.arange(L_CLIP) / 50.0
    lib_s = np.zeros((L, 71))
    for c in range(2):
        sl = slice(c * L_CLIP, (c + 1) * L_CLIP)
        lib_s[sl, 0] = 0.3 * t_ + c
        lib_s[sl, 1] = 0.1 * np.sin(t_)
        lib_s[sl, 2] = 0.78 + 0.02 * np.sin(3 * t_)
        yaw = 0.4 * t_ + c
        lib_s[sl, 3] = np.cos(yaw / 2)
        lib_s[sl, 6] = np.sin(yaw / 2)
        ph = rng.uniform(0, 6.28, 29)
        lib_s[sl, 7:36] = 0.3 * np.sin(2 * t_[:, None] + ph[None, :])
        lib_s[sl, 36] = 0.3
        lib_s[sl, 41] = 0.4
        lib_s[sl, 42:71] = 0.6 * np.cos(2 * t_[:, None] + ph[None, :])
    lib_s = f32(lib_s)
    with torch.no_grad():
        lo = fk(torch.tensor(lib_s[:, :36]), torch.tensor(lib_s[:, 36:]))
    ti = fk.body_indices(tracked)
    pel = fk.body_index("pelvis")
    lib_goal = f32(rng.normal(0, 0.5, (L, 680)))
    clip_end = np.where(np.arange(L) < L_CLIP, L_CLIP - 1, L - 1)

    class Lib:
        default_joint_pos = torch.tensor(q0)

        def goal_window(self, frame, k):
            return torch.tensor(lib_goal)[frame]

        def last_frame_of(self, frame):
            return torch.tensor(clip_end)[frame]

        def ref(self, frame):
            return _NS(root_pos_w=lo["pos"][frame, pel], root_quat_w=lo["quat"][frame, pel])

        def bodies(self, frame):
            return _NS(pos_w=lo["pos"][frame][:, ti], quat_w=lo["quat"][frame][:, ti],
                       lin_vel_w=lo["lin_vel"][frame][:, ti], ang_vel_w=lo["ang_vel"][frame][:, ti])

    lib = Lib()

    # ── networks, every layer random, float32-rounded ──
    wm = wmmod.WorldModel(wmmod.WorldModelCfg(hidden=(HW, HW, HW), term_hidden=(TH, TH)), heads, "cpu").double()
    pcfg = nets.PolicyCfg(encoder_hidden=ENC, decoder_hidden=DEC)
    policy = nets.Policy(pcfg).double()
    critic = nets.Critic(nets.CriticCfg(hidden=CRIT), pcfg=pcfg).double()

    def linears(mlp):
        return [x for x in mlp.net if isinstance(x, torch.nn.Linear)]

    groups = [
        ("dynamics", linears(wm.dynamics.mlp), 0.2),
        ("term_head", linears(wm.term_head.mlp), 1.0),
        ("encoder", linears(policy.encoders["motion"].mlp), 1.0),
        ("decoder", linears(policy.decoder.mlp), 0.3),
        ("critic", linears(critic.mlp), 1.0),
    ]
    with torch.no_grad():
        for _, lins, last_scale in groups:
            for i, lin in enumerate(lins):
                sc = last_scale if i == len(lins) - 1 else 1.0
                lin.weight.copy_(torch.tensor(f32(sc * rng.normal(0, 1.0 / np.sqrt(lin.weight.shape[1]), lin.weight.shape))))
                lin.bias.copy_(torch.tensor(f32(rng.normal(0, 0.05, lin.bias.shape))))
        # a random termination head kills every lane within a few steps, which
        # would leave the late steps out of J: bias it to ~2 % a step
        linears(wm.term_head.mlp)[-1].bias.fill_(-4.0)

    import copy

    fresh = (copy.deepcopy(wm), copy.deepcopy(policy), copy.deepcopy(critic))

    # ── the start batch ──
    frame = rng.integers(0, L - 1, N)
    frame[0] = L_CLIP - 5          # a lane whose clip ends inside the horizon
    s0 = lib_s[frame].copy()
    s0[:, 0:3] += rng.normal(0, 0.03, (N, 3))
    s0[:, 3:7] = unit(s0[:, 3:7] + rng.normal(0, 0.03, (N, 4)))
    s0[:, 7:36] += rng.normal(0, 0.05, (N, 29))
    s0 = f32(s0)
    hist_states = f32(np.stack([lib_s[np.maximum(frame - 10 + i, 0)] for i in range(10)]))
    hist_actions = f32(rng.normal(0, 0.3, (10, N, 29)))
    prev_action = f32(rng.normal(0, 0.3, (N, 29)))
    xi = np.zeros((N, 33))
    xi[:, 0] = rng.uniform(0.3, 1.2, N)
    xi[:, 1:30] = rng.normal(0, 0.01, (N, 29))
    xi[:, 30:] = rng.normal(0, 0.05, (N, 3))
    xi = f32(xi)
    start = _NS(state=torch.tensor(s0), frame=torch.tensor(frame), prev_action=torch.tensor(prev_action),
                xi=torch.tensor(xi), hist_states=torch.tensor(hist_states), hist_actions=torch.tensor(hist_actions))

    # ── the actor half of PolicyTrainer.update ──
    wm.requires_grad_(False)
    critic.requires_grad_(False)
    ro = rollout.imagine(wm, policy, lib, start, H, 10, obs_noise=False)
    Hs, n_, _ = ro.states.shape
    feat = wm.spec.canonical(ro.states.reshape(Hs * n_, -1)).reshape(Hs, n_, -1)
    xi_norm = wm.spec.normalize_xi(start.xi)
    v_H = critic(feat[-1], ro.goals[-1], xi_norm, ro.e_root[-1])
    J = objective.policy_objective(ro, v_H, GAMMA)
    loss = -J / H
    policy.zero_grad()
    loss.backward()
    with torch.no_grad():
        hist0 = torch.stack([hm.analytic_observation(start.hist_states[i], start.hist_actions[i], start.xi, torch.tensor(q0)) for i in range(10)])

    # per-step data the Mojo side is handed: goals and refs at frames_all
    last = clip_end[frame]
    frames_all = np.minimum(frame[None, :] + np.arange(H + 1)[:, None], last[None, :])
    refs = np.concatenate([
        np.concatenate([lo["pos"][frames_all][..., ti, :], lo["quat"][frames_all][..., ti, :],
                        lo["lin_vel"][frames_all][..., ti, :], lo["ang_vel"][frames_all][..., ti, :]], -1).reshape(H + 1, N, -1),
        lo["pos"][frames_all][..., pel, :], lo["quat"][frames_all][..., pel, :]], -1)
    assert refs.shape[-1] == 189

    def sv(name, x):
        x = x.detach().numpy() if torch.is_tensor(x) else np.asarray(x)
        np.save(a.out / f"i_{name}.npy", np.ascontiguousarray(np.asarray(x, dtype=np.float64).reshape(-1, x.shape[-1] if x.ndim > 1 else 1)))

    sv("state", s0)
    sv("prev_action", prev_action)
    sv("xi", xi)
    sv("q0", q0[None, :])
    sv("hist0", hist0)
    sv("goals", torch.tensor(lib_goal)[frames_all])
    sv("refs", refs)
    sv("states", ro.states)
    sv("actions", ro.actions)
    sv("rewards", ro.rewards[..., None])
    sv("dhat", ro.dhat[..., None])
    sv("survival", ro.survival[..., None])
    sv("e_root", ro.e_root)
    sv("v_h", v_H[:, None])
    np.save(a.out / "i_j.npy", np.array([[float(J.detach()), float((ro.survival[:-1] * ro.rewards).sum(0).mean().detach())]]))

    names, vals, grads = [], [], []
    for gname, lins, _ in groups:
        for i, lin in enumerate(lins):
            names.append((f"{gname}.{i}.weight", lin.weight.numel()))
            names.append((f"{gname}.{i}.bias", lin.bias.numel()))
            vals += [lin.weight.detach().numpy().T.ravel(), lin.bias.detach().numpy().ravel()]
            gw = lin.weight.grad.numpy().T.ravel() if lin.weight.grad is not None else np.zeros(lin.weight.numel())
            gb = lin.bias.grad.numpy().ravel() if lin.bias.grad is not None else np.zeros(lin.bias.numel())
            grads += [gw, gb]
    np.save(a.out / "i_weights.npy", np.concatenate(vals)[None, :])
    np.save(a.out / "i_grads.npy", np.concatenate(grads)[None, :])
    (a.out / "i_names.txt").write_text("".join(f"{nm}\t{k}\n" for nm, k in names))
    print(f"  N={N} H={H}: J {float(J.detach()):.6f}, undiscounted return "
          f"{float((ro.survival[:-1] * ro.rewards).sum(0).mean()):.4f}, survival_H {float(ro.survival[-1].mean()):.3f}, "
          f"mean reward {float(ro.rewards.mean()):.5f}, |a| {float(ro.actions.abs().mean()):.3f}")
    trainer_run(ho, R, fresh, lib, start, a.out)
    print(f"-> {a.out}")


def trainer_run(ho, R, fresh, lib, start, out: Path) -> None:
    """`PolicyTrainer.update` VERBATIM on copies of the same initial networks
    and the same start batch (a stub buffer returns it). `torch.randperm` and
    `gae_targets` are wrapped to RECORD their draws / inputs / outputs, so the
    Mojo side can replay the critic's minibatch order exactly. Files `u_*`."""
    ptmod = ho._load("supermimic.imagination.policy_trainer", R / "imagination" / "policy_trainer.py")
    wm, policy, critic = fresh
    perms, gae_io = [], []
    real_randperm = torch.randperm

    def rec_randperm(*args, **kw):
        p_ = real_randperm(*args, **kw)
        perms.append(p_.clone())
        return p_

    real_gae = ptmod.gae_targets

    def rec_gae(rewards, dhat_next, values, gamma, lam):
        t_ = real_gae(rewards, dhat_next, values, gamma, lam)
        gae_io.append((values.clone(), t_.clone()))
        return t_

    class Buf:
        def sample_starts(self, b, h):
            return start

    cfg = ptmod.PolicyTrainerCfg(horizon=H, batch_size=N, compile=None, obs_noise=False)
    tr = ptmod.PolicyTrainer(wm, policy, critic, lib, cfg)
    torch.manual_seed(5)
    torch.randperm = rec_randperm
    ptmod.gae_targets = rec_gae
    try:
        met = tr.update(Buf())
    finally:
        torch.randperm = real_randperm
        ptmod.gae_targets = real_gae
    assert len(perms) == cfg.critic_epochs and len(gae_io) == 1

    def lin(mlp):
        return [x for x in mlp.net if isinstance(x, torch.nn.Linear)]

    def flat(lins):
        return np.concatenate([np.concatenate([l.weight.detach().numpy().T.ravel(), l.bias.detach().numpy().ravel()]) for l in lins])[None, :]

    np.save(out / "u_policy.npy", flat(lin(policy.encoders["motion"].mlp) + lin(policy.decoder.mlp)))
    np.save(out / "u_critic.npy", flat(lin(critic.mlp)))
    np.save(out / "u_perms.npy", torch.stack(perms).numpy().astype(np.float64).reshape(1, -1))
    np.save(out / "u_values.npy", gae_io[0][0].numpy().reshape(-1, 1))
    np.save(out / "u_targets.npy", gae_io[0][1].numpy().reshape(-1, 1))
    keys = ("imag_objective", "imag_return_undisc", "policy_grad_norm", "critic_loss", "critic_value_mean", "critic_target_mean")
    np.save(out / "u_metrics.npy", np.array([[met[k] for k in keys]]))
    print("  [update] " + ", ".join(f"{k} {met[k]:.6g}" for k in keys))


if __name__ == "__main__":
    main()
