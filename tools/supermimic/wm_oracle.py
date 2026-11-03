#!/usr/bin/env python3
# +--------------------------------------------------------------------------+ #
# | SuperMimic M4 oracle: the reproduction's StateSpec and WorldModel loss
# +--------------------------------------------------------------------------+ #
"""Reference tensors for `tests/deep_agents/supermimic/test_supermimic_wm.mojo`.

    uv run --no-project --python 3.12 --with torch --with mujoco==3.12.0 --with numpy \\
        python tools/supermimic/wm_oracle.py --out /tmp/sm_wm

The reproduction's `world_model/state_spec.py`, `dynamics.py` and
`world_model.py` run VERBATIM in torch float64 (loaded by file path, as in
`heads_oracle.py`, whose loaders this script reuses).

[A] `StateSpec.canonical`, `integrate`, `error_vector` on 48 lanes, with
    autograd for random cotangents. A third of the lanes is shaped to reach
    the small-angle series: `quat_from_rotvec` (|dt·w' + dθ| < 1e-4) and
    `rotvec_from_quat` (|xyz| < 1e-6, a prediction within 1e-7 rad of the
    truth). Files `a_*`.
[B] `WorldModel.window_loss` on k = 8 windows of B = 16 lanes with small
    widths (H = 32, TH = 16) and random weights in every layer (the
    dynamics' last layer is zero at init, which would zero every gradient
    behind it). Weights and inputs are rounded to float32 first, so the
    Mojo side (float32) and this one (float64) start from the same numbers.
    Then `clip_grad_norm_(1.0)` + one `Adam(lr=2e-4)` step. Files `b_*`;
    `b_names.txt` lists the params in OUR walk order and layout (weights
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


def _heads_oracle():
    spec = importlib.util.spec_from_file_location("sm_heads_oracle", HERE / "heads_oracle.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sm_heads_oracle"] = mod
    spec.loader.exec_module(mod)
    return mod


H, TH, K, B = 32, 16, 8, 16


def unit(q):
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def random_states(ho, m, n, rng):
    qpos, qvel = ho.states(m, n, rng)
    return np.concatenate([qpos, qvel], 1)


def part_a(ho, ss, m, out: Path, rng):
    n = 48
    s = random_states(ho, m, n, rng)
    delta = rng.normal(0, 1.0, (n, 70))
    third = n // 3
    # quat_from_rotvec's series: no body rate, tiny rotation increment
    s[2 * third:, 39:42] = 0.0
    delta[2 * third:, 3:6] = 0.0
    delta[2 * third:, 38:41] = rng.normal(0, 1e-3, (n - 2 * third, 3))   # × 0.02 → ~2e-5 rad
    # rotvec_from_quat's series: a truth within ~1e-7 rad of the prediction
    s_true = random_states(ho, m, n, rng)
    near = s[2 * third:, 3:7] / np.linalg.norm(s[2 * third:, 3:7], axis=1, keepdims=True)
    s_true[2 * third:, 3:7] = unit(near + rng.normal(0, 5e-8, near.shape))
    t = lambda x: torch.tensor(x, requires_grad=True)  # noqa: E731
    ts, td, tst = t(s), t(delta), torch.tensor(s_true)
    feat = ss.canonical(ts)
    nxt = ss.integrate(ts, td)
    err = ss.error_vector(ts, tst)
    g_f, g_n, g_e = rng.normal(size=feat.shape), rng.normal(size=nxt.shape), rng.normal(size=err.shape)
    grads = {}
    for name, y, g, wrt in (("canon", feat, g_f, (ts,)), ("integ", nxt, g_n, (ts, td)), ("err", err, g_e, (ts,))):
        gr = torch.autograd.grad((y * torch.tensor(g)).sum(), wrt, retain_graph=True)
        grads[name] = [x.numpy() for x in gr]
    save = lambda name, x: np.save(out / f"a_{name}.npy", np.ascontiguousarray(np.asarray(x, dtype=np.float64).reshape(n, -1)))  # noqa: E731
    save("s", s)
    save("delta", delta)
    save("s_true", s_true)
    save("feat", feat.detach())
    save("next", nxt.detach())
    save("err", err.detach())
    save("g_feat", g_f)
    save("g_next", g_n)
    save("g_err", g_e)
    save("gs_canon", grads["canon"][0])
    save("gs_integ", grads["integ"][0])
    save("gd_integ", grads["integ"][1])
    save("gs_err", grads["err"][0])
    print(f"  [A] {n} lanes: canonical / integrate / error_vector + VJPs")


def f32(x):
    return np.asarray(x, dtype=np.float32).astype(np.float64)


def part_b(ho, mods, m, out: Path, rng):
    fkmod, hm, wmmod = mods
    fk = fkmod.G1Kinematics(m, dtype=torch.float64)
    tracked = ho._literal(ho.REF / "envs" / "tracking_env_cfg.py", "TRACKED_BODIES")
    heads = hm.AnalyticHeads(fk, tracked, "torso_link", "pelvis", torch.zeros(29, 2), torch.zeros(29))
    cfg = wmmod.WorldModelCfg(hidden=(H, H, H), term_hidden=(TH, TH))
    wm = wmmod.WorldModel(cfg, heads, "cpu").double()
    # every layer random (the dynamics' last is zero at init)
    linears = [mod for mod in wm.dynamics.mlp.net if isinstance(mod, torch.nn.Linear)] + [
        mod for mod in wm.term_head.mlp.net if isinstance(mod, torch.nn.Linear)]
    with torch.no_grad():
        for i, lin in enumerate(linears):
            fan_in = lin.weight.shape[1]
            lin.weight.copy_(torch.tensor(f32(rng.normal(0, 1.0 / np.sqrt(fan_in), lin.weight.shape))))
            lin.bias.copy_(torch.tensor(f32(rng.normal(0, 0.1, lin.bias.shape))))
        # keep the dynamics output in a sane range so the 8-step chain stays physical
        linears[3].weight.mul_(0.2)
        linears[3].weight.copy_(torch.tensor(f32(linears[3].weight.numpy())))
    names = [f"dynamics.{j}" for j in range(4)] + [f"term_head.{j}" for j in range(3)]

    s0 = random_states(ho, m, B, rng)
    s0[:, 3:7] = unit(s0[:, 3:7])
    states = [s0]
    for _ in range(K):
        nx = states[-1].copy()
        nx[:, 0:3] += 0.02 * nx[:, 36:39] + rng.normal(0, 0.005, (B, 3))
        nx[:, 3:7] = unit(nx[:, 3:7] + rng.normal(0, 0.02, (B, 4)))
        nx[:, 7:36] += 0.02 * nx[:, 42:71] + rng.normal(0, 0.01, (B, 29))
        nx[:, 36:] += rng.normal(0, 0.1, (B, 35))
        states.append(nx)
    states = f32(np.stack(states))                       # (k+1, B, 71)
    a_exec = f32(rng.normal(0, 0.5, (K, B, 29)))
    xi = np.zeros((B, 33))
    xi[:, 0] = rng.uniform(0.3, 1.2, B)
    xi[:, 1:30] = rng.normal(0, 0.01, (B, 29))
    xi[:, 30:] = rng.normal(0, 0.05, (B, 3))
    xi = f32(xi)
    done = (rng.random((K, B)) < 0.15).astype(np.float64)
    ref_pos = f32(np.concatenate([rng.normal(0, 1, (K, B, 2)), rng.uniform(0.6, 0.9, (K, B, 1))], -1))
    ref_quat = f32(unit(np.array([1.0, 0, 0, 0]) + rng.normal(0, 0.2, (K, B, 4))))

    refs = [hm.ReferenceFrame(None, None, None, None, torch.tensor(ref_pos[j]), torch.tensor(ref_quat[j])) for j in range(K)]
    loss, met = wm.window_loss(torch.tensor(states), torch.tensor(a_exec), torch.tensor(xi), torch.tensor(done), refs)
    wm.zero_grad()
    loss.backward()
    grads = [(lin.weight.grad.numpy().T.copy(), lin.bias.grad.numpy().copy()) for lin in linears]
    vals = [(lin.weight.detach().numpy().T.copy(), lin.bias.detach().numpy().copy()) for lin in linears]
    gn = torch.nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
    opt = torch.optim.Adam(wm.parameters(), lr=2e-4)
    opt.step()
    stepped = [(lin.weight.detach().numpy().T.copy(), lin.bias.detach().numpy().copy()) for lin in linears]

    def flat(pairs):
        return np.concatenate([np.concatenate([w.ravel(), b.ravel()]) for w, b in pairs])[None, :]

    np.save(out / "b_weights.npy", flat(vals))
    np.save(out / "b_grads.npy", flat(grads))
    np.save(out / "b_stepped.npy", flat(stepped))
    lines = []
    for nm, (w, b) in zip(names, vals):
        lines.append(f"{nm}.weight\t{w.size}")
        lines.append(f"{nm}.bias\t{b.size}")
    (out / "b_names.txt").write_text("\n".join(lines) + "\n")
    np.save(out / "b_states.npy", states.reshape((K + 1) * B, 71))
    np.save(out / "b_a_exec.npy", a_exec.reshape(K * B, 29))
    np.save(out / "b_xi.npy", xi)
    np.save(out / "b_done.npy", done.reshape(K * B, 1))
    np.save(out / "b_ref_root.npy", np.concatenate([ref_pos, ref_quat], -1).reshape(K * B, 7))
    np.save(out / "b_loss.npy", np.array([[float(loss), float(met["state_loss"]), float(met["term_bce"]),
                                           float(met["nse_1step"]), float(met["nse_kstep"]), float(gn)]]))
    print(f"  [B] k={K} B={B} H={H} TH={TH}: loss {float(loss):.6f} (state {float(met['state_loss']):.4f}, "
          f"bce {float(met['term_bce']):.4f}), grad norm {float(gn):.4f}, done rate {done.mean():.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    ho = _heads_oracle()
    fkmod, hm = ho.reference_modules()
    sspec = ho._load("supermimic.world_model.state_spec", ho.REF / "world_model" / "state_spec.py")
    ho._load("supermimic.world_model.dynamics", ho.REF / "world_model" / "dynamics.py")
    # world_model.py imports StateSpec etc. only; no trainer, no env
    wmmod = ho._load("supermimic.world_model.world_model", ho.REF / "world_model" / "world_model.py")
    m = mujoco.MjModel.from_xml_path(str(ho.G1_XML))
    rng = np.random.default_rng(a.seed)
    ss = sspec.StateSpec(sspec.StateSpecCfg(), "cpu", dtype=torch.float64)
    part_a(ho, ss, m, a.out, rng)
    part_b(ho, (fkmod, hm, wmmod), m, a.out, rng)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
