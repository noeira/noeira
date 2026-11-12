#!/usr/bin/env python3
# +--------------------------------------------------------------------------+ #
# | SuperMimic with an object: oracle for the object's state spec and the joint loss
# +--------------------------------------------------------------------------+ #
"""Reference tensors for `tests/deep_agents/supermimic/test_supermimic_obj.mojo`.

    uv run --no-project --python 3.12 --with torch --with mujoco==3.12.0 --with numpy \\
        python tools/supermimic/obj_oracle.py --out /tmp/sm_obj

There is no reference implementation of the object block (it is noeira's:
`noeira/deep_agents/supermimic/object_spec.mojo`, `object_model.mojo`). This
file is a SECOND transcription of its formulas, written in torch float64 from
the module docstrings, with autograd for every gradient; the robot's side is
the reproduction's `StateSpec` and `WorldModel` run verbatim, as in
`wm_oracle.py` (whose helpers this script reuses), and the quaternion algebra
is the reproduction's.

    o (13)     [p 3, q 4 WXYZ, v 3 world, w 3 own frame]
    feat (15)  [R_h^-1 (p - p_root), rot6d(h^-1 (x) q), R_h^-1 v, w / 2],  h = heading of the ROBOT's root
    step       v' = v + R_h dv_h, w' = w + dw_b, p' = p + dt v' + R_h dp_h, q' = normalise(q (x) exp(dt w' + dth_b))
               delta (12) = [dv_h, dw_b, dp_h, dth_b] x (0.3, 0.5, 0.01, 0.02)
    error (12) [dp, log(q_true^-1 (x) q), dv, dw] / (0.05, 0.1, 0.5, 1)
    ctx (2)    [(mass - 1.05) / 0.55, (friction - 0.9) / 0.4]

[A] features, integration and error on 48 lanes with autograd for random
    cotangents; a third of the lanes reaches the small-angle series (as
    `wm_oracle.py` [A]). Files `a_*`.
[B] the joint k = 8 window loss on B = 16 lanes: the reproduction's world
    model (H = 32, TH = 16) plus the object network (HO = 24), every layer
    random,

        dl = dynamics(x);  dlo = object([x, feat(s, o), ctx])
        s' = integrate(s, dl + dlo[:70]);  o' = integrate_o(s, o, dlo[70:])
        loss = (1/k) sum_j mean_b [ sum e^2 + w_obj sum eo^2 ] + (1/k) sum_j mean_b BCE,   w_obj = 0.7

    and its gradient for all eleven linears. Weights and inputs are rounded
    to float32 first. Files `b_*`; `b_names.txt` lists the params in OUR walk
    order (the world model's, then the object network's) and layout (weights
    transposed to [IN, OUT]).
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
H, TH, HO, K, B = 32, 16, 24, 8, 16
W_OBJ = 0.7
DT = 0.02


def _load_sibling(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def unit(q):
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def f32(x):
    return np.asarray(x, dtype=np.float32).astype(np.float64)


class ObjectSpec:
    """The object block's formulas, on the reproduction's quaternion helpers (`rot`)."""

    def __init__(self, rot):
        self.r = rot
        self.out_scale = torch.tensor([0.3] * 3 + [0.5] * 3 + [0.01] * 3 + [0.02] * 3)
        self.err_scale = torch.tensor([0.05] * 3 + [0.1] * 3 + [0.5] * 3 + [1.0] * 3)

    def heading(self, s):
        return self.r.heading_quat(self.r.quat_normalize(s[..., 3:7]))

    def feat(self, s, o):
        r = self.r
        hq = self.heading(s)
        qo = r.quat_normalize(o[..., 3:7])
        rel = r.quat_apply_inverse(hq, o[..., 0:3] - s[..., 0:3])
        r6 = r.rot6d_from_quat(r.quat_mul(r.quat_apply_inverse_quat(hq), qo))
        vh = r.quat_apply_inverse(hq, o[..., 7:10])
        return torch.cat([rel, r6, vh, o[..., 10:13] / 2.0], -1)

    def integrate(self, s, o, delta):
        r = self.r
        d = delta * self.out_scale
        dv_h, dw_b, dp_h, dth_b = d[..., 0:3], d[..., 3:6], d[..., 6:9], d[..., 9:12]
        hq = self.heading(s)
        qo = r.quat_normalize(o[..., 3:7])
        v1 = o[..., 7:10] + r.quat_apply(hq, dv_h)
        w1 = o[..., 10:13] + dw_b
        p1 = o[..., 0:3] + DT * v1 + r.quat_apply(hq, dp_h)
        q1 = r.quat_normalize(r.quat_mul(qo, r.quat_from_rotvec(DT * w1 + dth_b)))
        return torch.cat([p1, q1, v1, w1], -1)

    def error(self, op, ot):
        r = self.r
        q1 = r.quat_normalize(op[..., 3:7])
        q2 = r.quat_normalize(ot[..., 3:7])
        rv = r.rotvec_from_quat(r.quat_mul(r.quat_apply_inverse_quat(q2), q1))
        return torch.cat([op[..., 0:3] - ot[..., 0:3], rv, op[..., 7:10] - ot[..., 7:10], op[..., 10:13] - ot[..., 10:13]], -1) / self.err_scale

    @staticmethod
    def ctx_norm(c):
        return torch.stack([(c[..., 0] - 1.05) / 0.55, (c[..., 1] - 0.9) / 0.4], -1)


def random_objects(s, rng):
    """Object states near the robots of `s` (n, 71): within ~1 m, any attitude, moving; the quaternion is NOT unit."""
    n = s.shape[0]
    o = np.zeros((n, 13))
    o[:, 0:3] = s[:, 0:3] + rng.normal(0, 0.5, (n, 3))
    o[:, 3:7] = unit(rng.normal(size=(n, 4))) * rng.uniform(0.8, 1.25, (n, 1))
    o[:, 7:10] = rng.normal(0, 0.7, (n, 3))
    o[:, 10:13] = rng.normal(0, 2.0, (n, 3))
    return o


def part_a(wo, osp, m, out: Path, rng):
    n = 48
    s = wo.random_states(wo_ho, m, n, rng)
    o = random_objects(s, rng)
    delta = rng.normal(0, 1.0, (n, 12))
    third = n // 3
    # quat_from_rotvec's series: no spin, a tiny rotation increment
    o[2 * third:, 10:13] = 0.0
    delta[2 * third:, 3:6] = 0.0
    delta[2 * third:, 9:12] = rng.normal(0, 1e-3, (n - 2 * third, 3))
    # rotvec_from_quat's series: a truth within ~1e-7 rad of the prediction
    o_true = random_objects(s, rng)
    near = unit(o[2 * third:, 3:7])
    o_true[2 * third:, 3:7] = unit(near + rng.normal(0, 5e-8, near.shape))
    t = lambda x: torch.tensor(x, requires_grad=True)  # noqa: E731
    ts, to, td, tot = t(s), t(o), t(delta), torch.tensor(o_true)
    feat = osp.feat(ts, to)
    nxt = osp.integrate(ts, to, td)
    err = osp.error(to, tot)
    g_f, g_n, g_e = rng.normal(size=feat.shape), rng.normal(size=nxt.shape), rng.normal(size=err.shape)
    gf = torch.autograd.grad((feat * torch.tensor(g_f)).sum(), (ts, to), retain_graph=True)
    gn = torch.autograd.grad((nxt * torch.tensor(g_n)).sum(), (ts, to, td), retain_graph=True)
    ge = torch.autograd.grad((err * torch.tensor(g_e)).sum(), (to,), retain_graph=True)
    save = lambda name, x: np.save(out / f"a_{name}.npy", np.ascontiguousarray(np.asarray(x, dtype=np.float64).reshape(n, -1)))  # noqa: E731
    for name, x in (("s", s), ("o", o), ("delta", delta), ("o_true", o_true), ("feat", feat.detach()), ("next", nxt.detach()),
                    ("err", err.detach()), ("g_feat", g_f), ("g_next", g_n), ("g_err", g_e),
                    ("gs_feat", gf[0]), ("go_feat", gf[1]), ("gs_integ", gn[0]), ("go_integ", gn[1]), ("gd_integ", gn[2]),
                    ("go_err", ge[0])):
        save(name, x)
    # the heading's share must be there to be gated: report its size
    print(f"  [A] {n} lanes: feat / integrate / error + VJPs; |d feat / d root quat| max {np.abs(gf[0][:, 3:7].numpy()).max():.3f}, "
          f"|d integrate / d root quat| max {np.abs(gn[0][:, 3:7].numpy()).max():.3f}")


def part_b(wo, osp, mods, m, out: Path, rng):
    ho = wo_ho
    fkmod, hm, wmmod = mods
    fk = fkmod.G1Kinematics(m, dtype=torch.float64)
    tracked = ho._literal(ho.REF / "envs" / "tracking_env_cfg.py", "TRACKED_BODIES")
    heads = hm.AnalyticHeads(fk, tracked, "torso_link", "pelvis", torch.zeros(29, 2), torch.zeros(29))
    cfg = wmmod.WorldModelCfg(hidden=(H, H, H), term_hidden=(TH, TH))
    wm = wmmod.WorldModel(cfg, heads, "cpu").double()
    wm_in = 71 + 29 + 33
    obj = torch.nn.Sequential(
        torch.nn.Linear(wm_in + 15 + 2, HO), torch.nn.SiLU(), torch.nn.Linear(HO, HO), torch.nn.SiLU(),
        torch.nn.Linear(HO, HO), torch.nn.SiLU(), torch.nn.Linear(HO, 70 + 12),
    ).double()
    lin_wm = [mod for mod in wm.dynamics.mlp.net if isinstance(mod, torch.nn.Linear)] + [
        mod for mod in wm.term_head.mlp.net if isinstance(mod, torch.nn.Linear)]
    lin_obj = [mod for mod in obj if isinstance(mod, torch.nn.Linear)]
    linears = lin_wm + lin_obj
    with torch.no_grad():
        for lin in linears:
            fan_in = lin.weight.shape[1]
            lin.weight.copy_(torch.tensor(f32(rng.normal(0, 1.0 / np.sqrt(fan_in), lin.weight.shape))))
            lin.bias.copy_(torch.tensor(f32(rng.normal(0, 0.1, lin.bias.shape))))
        # keep both outputs in a sane range so the 8-step chains stay physical
        for last in (lin_wm[3], lin_obj[3]):
            last.weight.mul_(0.2)
            last.weight.copy_(torch.tensor(f32(last.weight.numpy())))
    names = [f"dynamics.{j}" for j in range(4)] + [f"term_head.{j}" for j in range(3)] + [f"object.{j}" for j in range(4)]

    s0 = wo.random_states(ho, m, B, rng)
    s0[:, 3:7] = unit(s0[:, 3:7])
    states = [s0]
    o0 = random_objects(s0, rng)
    o0[:, 3:7] = unit(o0[:, 3:7])
    objs = [o0]
    for _ in range(K):
        nx = states[-1].copy()
        nx[:, 0:3] += 0.02 * nx[:, 36:39] + rng.normal(0, 0.005, (B, 3))
        nx[:, 3:7] = unit(nx[:, 3:7] + rng.normal(0, 0.02, (B, 4)))
        nx[:, 7:36] += 0.02 * nx[:, 42:71] + rng.normal(0, 0.01, (B, 29))
        nx[:, 36:] += rng.normal(0, 0.1, (B, 35))
        states.append(nx)
        no = objs[-1].copy()
        no[:, 0:3] += 0.02 * no[:, 7:10] + rng.normal(0, 0.005, (B, 3))
        no[:, 3:7] = unit(no[:, 3:7] + rng.normal(0, 0.02, (B, 4)))
        no[:, 7:] += rng.normal(0, 0.1, (B, 6))
        objs.append(no)
    states = f32(np.stack(states))                       # (k+1, B, 71)
    objs = f32(np.stack(objs))                           # (k+1, B, 13)
    a_exec = f32(rng.normal(0, 0.5, (K, B, 29)))
    xi = np.zeros((B, 33))
    xi[:, 0] = rng.uniform(0.3, 1.2, B)
    xi[:, 1:30] = rng.normal(0, 0.01, (B, 29))
    xi[:, 30:] = rng.normal(0, 0.05, (B, 3))
    xi = f32(xi)
    octx = f32(np.stack([rng.uniform(0.1, 2.0, B), rng.uniform(0.5, 1.3, B)], 1))
    done = (rng.random((K, B)) < 0.15).astype(np.float64)
    ref_pos = f32(np.concatenate([rng.normal(0, 1, (K, B, 2)), rng.uniform(0.6, 0.9, (K, B, 1))], -1))
    ref_quat = f32(unit(np.array([1.0, 0, 0, 0]) + rng.normal(0, 0.2, (K, B, 4))))

    T = torch.tensor
    s, o = T(states[0]), T(objs[0])
    xi_n, ctx_n = wm.spec.normalize_xi(T(xi)), osp.ctx_norm(T(octx))
    tot_s = tot_o = tot_b = 0.0
    nse = []
    onse = []
    for j in range(K):
        a = T(a_exec[j])
        x = torch.cat([wm.spec.canonical(s), a, xi_n], -1)
        dl = wm.dynamics.mlp(x)
        dlo = obj(torch.cat([x, osp.feat(s, o), ctx_n], -1))
        s1 = wm.spec.integrate(s, dl + dlo[:, :70])
        o1 = osp.integrate(s, o, dlo[:, 70:])
        e = wm.spec.error_vector(s1, T(states[j + 1]))
        eo = osp.error(o1, T(objs[j + 1]))
        ref = hm.ReferenceFrame(None, None, None, None, T(ref_pos[j]), T(ref_quat[j]))
        logits = wm.term_head(heads.termination_features(s1[..., 0:3], s1[..., 3:7], ref))
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, T(done[j]), reduction="none")
        tot_s = tot_s + (e * e).sum(-1).mean()
        tot_o = tot_o + (eo * eo).sum(-1).mean()
        tot_b = tot_b + bce.mean()
        nse.append(float((e * e).mean(-1).mean()))
        onse.append(float((eo * eo).mean(-1).mean()))
        s, o = s1, o1
    loss = tot_s / K + W_OBJ * tot_o / K + tot_b / K
    for lin in linears:
        lin.zero_grad()
    loss.backward()
    grads = [(lin.weight.grad.numpy().T.copy(), lin.bias.grad.numpy().copy()) for lin in linears]
    vals = [(lin.weight.detach().numpy().T.copy(), lin.bias.detach().numpy().copy()) for lin in linears]

    def flat(pairs):
        return np.concatenate([np.concatenate([w.ravel(), b.ravel()]) for w, b in pairs])[None, :]

    np.save(out / "b_weights.npy", flat(vals))
    np.save(out / "b_grads.npy", flat(grads))
    lines = []
    for nm, (w, b) in zip(names, vals):
        lines.append(f"{nm}.weight\t{w.size}")
        lines.append(f"{nm}.bias\t{b.size}")
    (out / "b_names.txt").write_text("\n".join(lines) + "\n")
    np.save(out / "b_states.npy", states.reshape((K + 1) * B, 71))
    np.save(out / "b_objs.npy", objs.reshape((K + 1) * B, 13))
    np.save(out / "b_a_exec.npy", a_exec.reshape(K * B, 29))
    np.save(out / "b_xi.npy", xi)
    np.save(out / "b_octx.npy", octx)
    np.save(out / "b_done.npy", done.reshape(K * B, 1))
    np.save(out / "b_ref_root.npy", np.concatenate([ref_pos, ref_quat], -1).reshape(K * B, 7))
    np.save(out / "b_loss.npy", np.array([[float(loss), float(tot_s / K), float(tot_b / K), float(tot_o / K),
                                           nse[0], nse[-1], onse[0], onse[-1], W_OBJ]]))
    g_obj = np.concatenate([np.concatenate([w.ravel(), b.ravel()]) for w, b in grads[7:]])
    g_wm = np.concatenate([np.concatenate([w.ravel(), b.ravel()]) for w, b in grads[:7]])
    print(f"  [B] k={K} B={B} H={H} TH={TH} HO={HO}: loss {float(loss):.6f} (robot {float(tot_s / K):.4f}, object {float(tot_o / K):.4f}, "
          f"bce {float(tot_b / K):.4f}); |grad| world model {np.linalg.norm(g_wm):.3f}, object net {np.linalg.norm(g_obj):.3f}")


def main() -> None:
    global wo_ho
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=17)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    wo = _load_sibling("sm_wm_oracle", "wm_oracle.py")
    ho = wo._heads_oracle()
    wo_ho = ho
    fkmod, hm = ho.reference_modules()
    sspec = ho._load("supermimic.world_model.state_spec", ho.REF / "world_model" / "state_spec.py")
    ho._load("supermimic.world_model.dynamics", ho.REF / "world_model" / "dynamics.py")
    wmmod = ho._load("supermimic.world_model.world_model", ho.REF / "world_model" / "world_model.py")
    m = mujoco.MjModel.from_xml_path(str(ho.G1_XML))
    rng = np.random.default_rng(a.seed)
    osp = ObjectSpec(sspec)       # the state spec module carries the reproduction's quaternion helpers
    part_a(wo, osp, m, a.out, rng)
    part_b(wo, osp, (fkmod, hm, wmmod), m, a.out, rng)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
