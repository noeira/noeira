#!/usr/bin/env python3
# +--------------------------------------------------------------------------+ #
# | SuperMimic with an object: the interface's imagined rollout, in torch
# +--------------------------------------------------------------------------+ #
"""Reference tensors for `tests/deep_agents/supermimic/test_supermimic_obj_imag.mojo`, section [4].

    uv run --no-project --python 3.12 --with torch --with mujoco==3.12.0 --with numpy \\
        python tools/supermimic/imag_oracle.py --out /tmp/sm_imag
    uv run --no-project --python 3.12 --with torch --with mujoco==3.12.0 --with numpy \\
        python tools/supermimic/obj_imag_oracle.py --tracker /tmp/sm_imag --out /tmp/sm_obj_imag

There is no reference implementation of the object interface. This is an
INDEPENDENT transcription of `object_imagination.sm_imagine_obj`'s formulas
(`docs/OBJECT_INTERACTION_PLAN.md` §4, step O1.4), in torch float64, with
autograd for the gradient the Mojo side derives by hand:

  * the tracker's side is the reproduction's own modules (world model, heads,
    policy), rebuilt from the tracker oracle's dump (`--tracker`): the same
    weights, start batch, goals and reference rows the Mojo test loads;
  * the object's state spec is `obj_oracle.ObjectSpec` (the O1.3 oracle);
  * the reward's three terms, the 33 features, the object encoder, the mix
    `l = z + gain · e`, the interface's critic and the loop are written here.

Files `o_*`: the object's inputs (obj0, octx, orefs, rw, meta = [gain, γ]),
the three new networks' weights in OUR walk order (weights transposed to
[IN, OUT]) with `o_sizes.txt`, the rollout's tensors, J, and the gradient of
−J / H with respect to the object encoder.
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
HO = 24                 # the object network's width
OE = (20, 16)           # the object encoder
OCRIT = (16, 12)        # the interface's critic
GAMMA = 0.99
GAIN = 2.0
DT = 0.02
# wide kernels, so no lane sits in a tail: (w, σ) for position, orientation, hands
RW = ((1.0, 0.8), (0.5, 2.0), (1.0, 1.2))
HANDS = ("left_wrist_yaw_link", "right_wrist_yaw_link")


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


def qconj(q):
    return q * torch.tensor([1.0, -1.0, -1.0, -1.0])


def mlp(sizes):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(torch.nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(torch.nn.SiLU())
    return torch.nn.Sequential(*layers).double()


def linears(seq):
    return [x for x in seq if isinstance(x, torch.nn.Linear)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tracker", type=Path, required=True, help="imag_oracle.py's output directory")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=23)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    ho = _load_by_path("sm_heads_oracle", HERE / "heads_oracle.py")
    fkmod, hm = ho.reference_modules()
    R = ho.REF
    ho._pkg("supermimic.data")
    ho._pkg("supermimic.training")
    ho._pkg("supermimic.policy")
    stub("supermimic.data.replay_buffer", ReplayBuffer=object, WindowBatch=object, StartBatch=object)
    stub("supermimic.envs.mdp.motion_library", MotionLibrary=object)
    stub("supermimic.training.distributed", allreduce_grads=lambda *a, **k: None)
    sspec = ho._load("supermimic.world_model.state_spec", R / "world_model" / "state_spec.py")
    ho._load("supermimic.world_model.dynamics", R / "world_model" / "dynamics.py")
    wmmod = ho._load("supermimic.world_model.world_model", R / "world_model" / "world_model.py")
    nets = ho._load("supermimic.policy.nets", R / "policy" / "nets.py")
    oo = _load_by_path("sm_obj_oracle", HERE / "obj_oracle.py")
    osp = oo.ObjectSpec(sspec)

    # ── the tracker's side, from its oracle's dump ──
    T = a.tracker

    def ld(name):
        return np.load(T / f"i_{name}.npy")

    q0 = ld("q0").reshape(29)
    m = mujoco.MjModel.from_xml_path(str(ho.G1_XML))
    fk = fkmod.G1Kinematics(m, dtype=torch.float64)
    tracked = ho._literal(R / "envs" / "tracking_env_cfg.py", "TRACKED_BODIES")
    heads = hm.AnalyticHeads(fk, tracked, "torso_link", "pelvis", torch.tensor(ho.soft_limits(m)), torch.tensor(q0))
    wm = wmmod.WorldModel(wmmod.WorldModelCfg(hidden=(HW, HW, HW), term_hidden=(TH, TH)), heads, "cpu").double()
    pcfg = nets.PolicyCfg(encoder_hidden=ENC, decoder_hidden=DEC)
    policy = nets.Policy(pcfg).double()

    def tl(x):
        return [y for y in x.net if isinstance(y, torch.nn.Linear)]

    groups = {
        "dynamics": tl(wm.dynamics.mlp), "term_head": tl(wm.term_head.mlp),
        "encoder": tl(policy.encoders["motion"].mlp), "decoder": tl(policy.decoder.mlp),
    }
    flat = np.load(T / "i_weights.npy").reshape(-1)
    at = 0
    with torch.no_grad():
        for line in (T / "i_names.txt").read_text().splitlines():
            if not line:
                continue
            nm, k = line.split("\t")
            k = int(k)
            g, i, kind = nm.split(".")
            if g in groups:
                lin = groups[g][int(i)]
                if kind == "weight":
                    lin.weight.copy_(torch.tensor(flat[at:at + k].reshape(lin.weight.shape[1], lin.weight.shape[0]).T.copy()))
                else:
                    lin.bias.copy_(torch.tensor(flat[at:at + k].copy()))
            at += k
    assert at == flat.size
    for p in list(wm.parameters()) + list(policy.parameters()):
        p.requires_grad_(False)

    s0 = torch.tensor(ld("state"))
    prev_action = torch.tensor(ld("prev_action"))
    xi = torch.tensor(ld("xi"))
    hist = list(torch.tensor(ld("hist0")).reshape(10, N, -1))
    goals = torch.tensor(ld("goals")).reshape(H + 1, N, -1)
    refs = torch.tensor(ld("refs")).reshape(H + 1, N, 189)
    NT = len(tracked)
    rb = refs[..., : 13 * NT].reshape(H + 1, N, NT, 13)

    def ref_at(t):
        return hm.ReferenceFrame(rb[t][..., 0:3], rb[t][..., 3:7], rb[t][..., 7:10], rb[t][..., 10:13], refs[t][..., 13 * NT:13 * NT + 3], refs[t][..., 13 * NT + 3:])

    # ── the object's inputs ──
    rng = np.random.default_rng(a.seed)
    obj0 = np.zeros((N, 13))
    obj0[:, 0:3] = s0.numpy()[:, 0:3] + rng.uniform(-0.2, 0.2, (N, 3)) + np.array([0.4, 0.0, 0.0])
    obj0[:, 3:7] = unit(rng.normal(size=(N, 4)))
    obj0[:, 7:13] = rng.uniform(-0.3, 0.3, (N, 6))
    obj0 = f32(obj0)
    octx = f32(np.stack([rng.uniform(0.2, 1.9, N), rng.uniform(0.5, 1.3, N)], -1))
    orefs = np.zeros((H + 1, N, 19))
    orefs[..., 0:3] = obj0[None, :, 0:3] + rng.uniform(-0.1, 0.1, (H + 1, N, 3)) + 0.004 * np.arange(H + 1)[:, None, None]
    orefs[..., 3:7] = unit(obj0[None, :, 3:7] + rng.uniform(-0.2, 0.2, (H + 1, N, 4)))
    orefs[..., 7:13] = rng.uniform(-0.3, 0.3, (H + 1, N, 6))
    orefs[..., 13:19] = rng.uniform(-0.3, 0.3, (H + 1, N, 6))
    orefs = f32(orefs)
    rw = f32(np.array([[DT * w, 1.0 / (sd * sd)] for w, sd in RW]).reshape(-1))

    # ── the three new networks ──
    wm_in = 71 + 29 + 33
    om = mlp([wm_in + 15 + 2, HO, HO, HO, 70 + 12])
    enc = mlp([33 + 930, OE[0], OE[1], 32])
    critic = mlp([71 + 680 + 33 + 4 + 33 + 2, OCRIT[0], OCRIT[1], 1])
    with torch.no_grad():
        for net, last in ((om, 0.02), (enc, 1.0), (critic, 1.0)):
            ls = linears(net)
            for i, lin in enumerate(ls):
                sc = last if i == len(ls) - 1 else 1.0
                lin.weight.copy_(torch.tensor(f32(sc * rng.normal(0, 1.0 / np.sqrt(lin.weight.shape[1]), lin.weight.shape))))
                lin.bias.copy_(torch.tensor(f32(sc * rng.normal(0, 0.05, lin.bias.shape))))
        # The object network's last layer: its 70 robot rows stay small (a larger correction throws the
        # robot, survival goes to 0 and J stops depending on the policy), its 12 OBJECT rows are 25 times
        # larger, so the object's motion depends on the action and every path through the object's own
        # state carries a visible share of the gradient (at 0.02 three planted bugs on those paths passed).
        # The object encoder reads its 33 features 20 times louder than its 930 history entries, and the
        # gain is 2: the path "an earlier action moves the state, the state moves the features, the
        # features move a later action" is second order in the encoder's influence, and at gain 0.5 with
        # plain weights a planted bug on it (the features' cotangent dropped) still passed.
        first = linears(enc)[0]
        first.weight[:, :33].mul_(20.0)
        first.weight.copy_(torch.tensor(f32(first.weight.numpy())))
        last = linears(om)[-1]
        last.weight[70:].mul_(25.0)
        last.bias[70:].mul_(25.0)
        last.weight.copy_(torch.tensor(f32(last.weight.numpy())))
        last.bias.copy_(torch.tensor(f32(last.bias.numpy())))
    for p in list(om.parameters()) + list(critic.parameters()):
        p.requires_grad_(False)

    # ── the interface's formulas ──
    hand_idx = [fk.body_index(nm) for nm in HANDS]
    rwt = torch.tensor(rw)
    dec_scale = policy.decoder.scale
    goal_scale = policy.encoders["motion"].scale

    def features(s, o, oref):
        fa = osp.feat(s, o)
        fb = osp.feat(s, oref[..., :13])
        return torch.cat([fa, fb, (fb[..., 0:3] - fa[..., 0:3]) / 0.1], -1)

    def obj_reward(s_next, o, oref):
        """(3, n): position, orientation, hands."""
        qo = sspec.quat_normalize(o[..., 3:7])
        qr = sspec.quat_normalize(oref[..., 3:7])
        e_p = ((o[..., 0:3] - oref[..., 0:3]) ** 2).sum(-1)
        d = sspec.quat_mul(qr, qconj(qo))
        ang = 2.0 * torch.atan2(d[..., 1:4].norm(dim=-1), d[..., 0].abs())
        e_q = ang * ang
        pos = fk(s_next[..., :36], s_next[..., 36:])["pos"]
        e_h = 0.0
        for i in range(2):
            rel = sspec.quat_apply_inverse(qo, pos[..., hand_idx[i], :] - o[..., 0:3])
            e_h = e_h + ((rel - oref[..., 13 + 3 * i:16 + 3 * i]) ** 2).sum(-1)
        e_h = e_h / 2.0
        return torch.stack([rwt[0] * torch.exp(-e_p * rwt[1]), rwt[2] * torch.exp(-e_q * rwt[3]), rwt[4] * torch.exp(-e_h * rwt[5])])

    # ── the rollout ──
    orefs_t = torch.tensor(orefs)
    octx_t = torch.tensor(octx)
    ctx_n = osp.ctx_norm(octx_t)
    xi_n = wm.spec.normalize_xi(xi)
    q0_t = torch.tensor(q0)
    s = s0
    o = torch.tensor(obj0)
    a_prev = prev_action
    c = torch.ones(N)
    states, objs, actions, rewards, oterms, dhats, survs = [s], [o], [], [], [], [torch.zeros(N)], [c]
    e_roots = [heads.root_error_features(s, ref_at(0))]
    for t in range(H):
        ref = ref_at(t)
        obs_flat = hm.flatten_history_term_major(torch.stack(hist, 0))
        z = policy.encoders["motion"](goals[t])
        e = enc(torch.cat([features(s, o, orefs_t[t]), obs_flat * dec_scale], -1))
        act = policy.decoder(z + GAIN * e, obs_flat)
        feat = wm.spec.canonical(s)
        delta = wm.dynamics(feat, act, xi_n)
        dlo = om(torch.cat([feat, act, xi_n, osp.feat(s, o), ctx_n], -1))
        s_next = wm.spec.integrate(s, delta + dlo[..., :70])
        o_next = osp.integrate(s, o, dlo[..., 70:])
        r_track = heads.reward(s_next, act, a_prev, ref, heads.robot_bodies(s_next))["total"]
        ot = obj_reward(s_next, o_next, orefs_t[t])
        r = r_track + ot.sum(0)
        d = torch.sigmoid(wm.termination_logits(s_next, ref))
        c = c * (1.0 - d)
        hist = hist[1:] + [hm.analytic_observation(s_next, act, xi, q0_t)]
        a_prev = act
        s, o = s_next, o_next
        states.append(s)
        objs.append(o)
        actions.append(act)
        rewards.append(r)
        oterms.append(ot.T)
        dhats.append(d)
        survs.append(c)
        e_roots.append(heads.root_error_features(s, ref_at(t + 1)))
    v_h = critic(torch.cat([wm.spec.canonical(s), goals[H] * goal_scale, xi_n, e_roots[H], features(s, o, orefs_t[H]), ctx_n], -1)).squeeze(-1)
    surv = torch.stack(survs)
    rew = torch.stack(rewards)
    disc = GAMMA ** torch.arange(H, dtype=torch.float64)
    J = ((disc[:, None] * surv[:-1] * rew).sum(0) + GAMMA ** H * surv[-1] * v_h).mean()
    loss = -J / H
    loss.backward()

    def sv(name, x, cols):
        x = x.detach().numpy() if torch.is_tensor(x) else np.asarray(x)
        np.save(a.out / f"o_{name}.npy", np.ascontiguousarray(np.asarray(x, dtype=np.float64).reshape(-1, cols)))

    sv("obj0", obj0, 13)
    sv("octx", octx, 2)
    sv("orefs", orefs, 19)
    sv("rw", rw, 6)
    sv("meta", np.array([GAIN, GAMMA]), 2)
    sv("states", torch.stack(states), 71)
    sv("objs", torch.stack(objs), 13)
    sv("actions", torch.stack(actions), 29)
    sv("rewards", rew, 1)
    sv("oterms", torch.stack(oterms), 3)
    sv("dhat", torch.stack(dhats), 1)
    sv("survival", surv, 1)
    sv("v_h", v_h, 1)
    sv("j", np.array([float(J.detach()), float((surv[:-1] * rew).sum(0).mean().detach())]), 2)
    vals, sizes = [], []
    for nm, net in (("object", om), ("obj_encoder", enc), ("obj_critic", critic)):
        for i, lin in enumerate(linears(net)):
            vals += [lin.weight.detach().numpy().T.ravel(), lin.bias.detach().numpy().ravel()]
            sizes += [(f"{nm}.{i}.weight", lin.weight.numel()), (f"{nm}.{i}.bias", lin.bias.numel())]
    np.save(a.out / "o_weights.npy", np.concatenate(vals)[None, :])
    (a.out / "o_sizes.txt").write_text("".join(f"{nm}\t{k}\n" for nm, k in sizes))
    grads = []
    for lin in linears(enc):
        grads += [lin.weight.grad.numpy().T.ravel(), lin.bias.grad.numpy().ravel()]
    g = np.concatenate(grads)
    np.save(a.out / "o_grads.npy", g[None, :])
    print(f"  N={N} H={H}: J {float(J.detach()):.6f}, survival_H {float(surv[-1].mean()):.3f}, mean reward {float(rew.mean()):.5f} "
          f"(object terms {float(torch.stack(oterms).sum(-1).mean()):.5f}), |g| {float(np.linalg.norm(g)):.3e}, "
          f"box moved {float((objs[-1][:, :3] - objs[0][:, :3]).norm(dim=-1).mean()):.3f} m")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
