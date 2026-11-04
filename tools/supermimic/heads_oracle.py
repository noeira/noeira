#!/usr/bin/env python3
# +--------------------------------------------------------------------------+ #
# | SuperMimic M3 oracle: the reproduction's torch FK (and heads) on OUR G1
# +--------------------------------------------------------------------------+ #
"""Reference tensors for `tests/deep_agents/test_supermimic_heads.mojo`.

    uv run --no-project --python 3.12 --with torch --with mujoco==3.12.0 --with numpy \\
        python tools/supermimic/heads_oracle.py --out /tmp/sm_heads

Needs torch, which pixi does not ship: the command above runs it in uv's
throwaway environment. MuJoCo is pinned to the runtime pixi ships (3.12.0).

TWO REFERENCES, NOT ONE
=======================
* MuJoCo itself on our XML (`noeira/envs/robots/assets/unitree_g1.xml`):
  `mj_kinematics` for `xpos`/`xquat`, `mj_comPos` + `mj_comVel` +
  `mj_objectVelocity(mjOBJ_XBODY, flg_local=0)` for each body's world velocity at its
  origin. Independent of the reproduction and of us.
* The reproduction's `G1Kinematics` (`references/SuperMimic-Reproduced/src/
  supermimic/kinematics/g1_fk.py`), run VERBATIM in float64 on our MjModel,
  and torch autograd through it for the VJP. Its quaternion helpers are
  mjlab's (`references/mjlab-8bb2bc9`, the commit the reproduction pins),
  loaded by file path under stub packages so neither package's `__init__`
  (warp, mjlab's env stack) is imported.

The script checks the two against each other before writing anything.

Written (all float64, 2-D `.npy`, B rows):
    qpos [B, 36], qvel [B, 35]            random states (see `states`)
    mj_pos, mj_quat, mj_lin, mj_ang       MuJoCo, [B, 40·3 / 40·4]
    t_pos, t_quat, t_lin, t_ang           the reproduction's FK
    c_pos, c_quat, c_lin, c_ang           random cotangents
    g_qpos [B, 36], g_qvel [B, 35]        autograd of Σ <c, FK(qpos, qvel)>
and the same set prefixed `jp_` for the model with random hinge anchors,
whose `jnt_pos` (hinges, in joint order) is `jp_jnt_pos.npy` [29, 3].
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

REPO = Path(__file__).resolve().parents[2]
REF = REPO / "references" / "SuperMimic-Reproduced" / "src" / "supermimic"
MJLAB = REPO / "references" / "mjlab-8bb2bc9" / "src" / "mjlab"
G1_XML = REPO / "noeira" / "envs" / "robots" / "assets" / "unitree_g1.xml"


def _pkg(name: str) -> None:
    if name not in sys.modules:
        m = types.ModuleType(name)
        m.__path__ = []
        sys.modules[name] = m


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    parent, _, leaf = name.rpartition(".")
    setattr(sys.modules[parent], leaf, mod)
    return mod


def reference_modules():
    for p in ("mjlab", "mjlab.utils", "mjlab.utils.lab_api"):
        _pkg(p)
    _load("mjlab.utils.lab_api.math", MJLAB / "utils" / "lab_api" / "math.py")
    for p in ("supermimic", "supermimic.kinematics", "supermimic.envs", "supermimic.envs.mdp", "supermimic.world_model"):
        _pkg(p)
    _load("supermimic.kinematics.rotations", REF / "kinematics" / "rotations.py")
    fk = _load("supermimic.kinematics.g1_fk", REF / "kinematics" / "g1_fk.py")
    _load("supermimic.envs.mdp.rewards", REF / "envs" / "mdp" / "rewards.py")
    heads = _load("supermimic.world_model.heads", REF / "world_model" / "heads.py")
    return fk, heads


def _literal(path: Path, name: str):
    """A module-level literal (a tuple of names) read with `ast`, so the module
    itself (mjlab's env stack) is never imported."""
    import ast

    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


def states(m: mujoco.MjModel, b: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Random generalized states: root anywhere within a few metres, a root
    quaternion that is NOT unit (|q| in [0.8, 1.2]: the FK normalises it, and
    the VJP must carry that), joints uniform over their range, velocities of
    a few rad/s."""
    qpos = np.zeros((b, m.nq))
    qpos[:, 0:2] = rng.uniform(-3, 3, (b, 2))
    qpos[:, 2] = rng.uniform(0.2, 1.0, b)
    q = rng.normal(size=(b, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    qpos[:, 3:7] = q * rng.uniform(0.8, 1.2, (b, 1))
    for j in range(1, m.njnt):
        lo, hi = m.jnt_range[j]
        qpos[:, m.jnt_qposadr[j]] = rng.uniform(lo, hi, b)
    qvel = np.zeros((b, m.nv))
    qvel[:, 0:3] = rng.normal(0, 1.0, (b, 3))
    qvel[:, 3:6] = rng.normal(0, 2.0, (b, 3))
    qvel[:, 6:] = rng.normal(0, 3.0, (b, m.nv - 6))
    return qpos, qvel


def mujoco_fk(m: mujoco.MjModel, qpos: np.ndarray, qvel: np.ndarray):
    d = mujoco.MjData(m)
    b, nb = qpos.shape[0], m.nbody - 1
    pos, quat = np.zeros((b, nb, 3)), np.zeros((b, nb, 4))
    lin, ang = np.zeros((b, nb, 3)), np.zeros((b, nb, 3))
    v6 = np.zeros(6)
    for i in range(b):
        d.qpos[:] = qpos[i]
        d.qvel[:] = qvel[i]
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        mujoco.mj_comVel(m, d)
        pos[i] = d.xpos[1:]
        quat[i] = d.xquat[1:]
        for k in range(1, m.nbody):
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_XBODY, k, v6, 0)  # XBODY = the link ORIGIN; BODY would be the CoM
            ang[i, k - 1] = v6[0:3]
            lin[i, k - 1] = v6[3:6]
    return pos, quat, lin, ang


def dump(fkmod, m: mujoco.MjModel, out: Path, prefix: str, b: int, rng: np.random.Generator) -> None:
    qpos, qvel = states(m, b, rng)
    nb = m.nbody - 1
    mj = mujoco_fk(m, qpos, qvel)
    fk = fkmod.G1Kinematics(m, dtype=torch.float64)
    tq = torch.tensor(qpos, requires_grad=True)
    tv = torch.tensor(qvel, requires_grad=True)
    o = fk(tq, tv)
    t = [o["pos"], o["quat"], o["lin_vel"], o["ang_vel"]]
    # The two references must agree before either is trusted.
    for name, x, y in zip(("pos", "quat", "lin", "ang"), mj, t):
        e = np.abs(x - y.detach().numpy()).max()
        print(f"  [{prefix or 'asset'}] reproduction FK vs MuJoCo, {name}: max abs err {e:.3e}")
        assert e < 1e-10, name
    cot = [rng.normal(size=x.shape) for x in mj]
    loss = sum((torch.tensor(c) * y).sum() for c, y in zip(cot, t))
    loss.backward()

    def save(name, x):
        np.save(out / f"{prefix}{name}.npy", np.ascontiguousarray(x.reshape(b, -1)))

    save("qpos", qpos)
    save("qvel", qvel)
    for nm, x in zip(("mj_pos", "mj_quat", "mj_lin", "mj_ang"), mj):
        save(nm, x)
    for nm, x in zip(("t_pos", "t_quat", "t_lin", "t_ang"), t):
        save(nm, x.detach().numpy())
    for nm, x in zip(("c_pos", "c_quat", "c_lin", "c_ang"), cot):
        save(nm, x)
    save("g_qpos", tq.grad.numpy())
    save("g_qvel", tv.grad.numpy())
    print(f"  [{prefix or 'asset'}] {b} states, {nb} bodies")


SOFT_LIMIT_FACTOR = 0.9  # mjlab's G1 (g1_constants.py, soft_joint_pos_limit_factor)


def soft_limits(m: mujoco.MjModel) -> np.ndarray:
    """mjlab's `soft_joint_pos_limits` (entity.py) for the 29 hinges."""
    r = m.jnt_range[1:]
    mean = (r[:, 0] + r[:, 1]) / 2
    rng_ = r[:, 1] - r[:, 0]
    return np.stack([mean - 0.5 * rng_ * SOFT_LIMIT_FACTOR, mean + 0.5 * rng_ * SOFT_LIMIT_FACTOR], -1)


def dump_heads(fkmod, hm, m: mujoco.MjModel, out: Path, b: int, rng: np.random.Generator) -> None:
    """`AnalyticHeads.reward`, `termination_features` and `analytic_observation`
    verbatim on our model, and autograd of Σ g_r·r + <g_f, feat> + <g_o, obs>
    with respect to (s, a, a_prev). Files `h_*`.

    Lanes come in three kinds, a third each, so every branch is reached:
    near the reference (every exp term well above zero), far from it, and
    within ~1e-9 of it (the quaternion error's Taylor branch, |angle| ≤ 1e-6)."""
    tracked = _literal(REF / "envs" / "tracking_env_cfg.py", "TRACKED_BODIES")
    anchor = _literal(REF / "envs" / "tracking_env_cfg.py", "ANCHOR_BODY")
    names = _literal(REF / "imagination" / "compiled.py", "REWARD_TERM_NAMES")
    fk = fkmod.G1Kinematics(m, dtype=torch.float64)
    q0 = rng.uniform(-0.3, 0.3, 29)
    soft = soft_limits(m)
    heads = hm.AnalyticHeads(fk, tracked, anchor, "pelvis", torch.tensor(soft), torch.tensor(q0))
    ref_qpos, ref_qvel = states(m, b, rng)
    ref_qpos[:, 3:7] /= np.linalg.norm(ref_qpos[:, 3:7], axis=1, keepdims=True)  # a library frame is unit
    s_ref = np.concatenate([ref_qpos, ref_qvel], 1)
    scale = np.ones((b, 1))
    third = b // 3
    scale[:third] = 0.15        # near
    scale[third:2 * third] = 1.0  # far
    scale[2 * third:] = 1e-9    # within the Taylor branch
    s = s_ref.copy()
    s[:, 0:3] += rng.normal(0, 0.3, (b, 3)) * scale
    s[:, 3:7] += rng.normal(0, 0.3, (b, 4)) * scale
    s[:, 3:7] *= (1.0 + rng.uniform(-0.01, 0.01, (b, 1)) * (scale > 1e-6))  # slightly non-unit roots
    s[:, 7:36] += rng.normal(0, 0.8, (b, 29)) * scale
    s[:, 36:] += rng.normal(0, 2.0, (b, 35)) * scale
    act = rng.normal(0, 0.5, (b, 29))
    act_prev = act + rng.normal(0, 0.3, (b, 29))
    xi = rng.normal(0, 0.05, (b, 33))

    with torch.no_grad():
        o = fk(torch.tensor(s_ref[:, :36]), torch.tensor(s_ref[:, 36:]))
    ti = fk.body_indices(tracked)
    pel = fk.body_index("pelvis")
    ref = hm.ReferenceFrame(o["pos"][:, ti], o["quat"][:, ti], o["lin_vel"][:, ti], o["ang_vel"][:, ti], o["pos"][:, pel], o["quat"][:, pel])

    ts = torch.tensor(s, requires_grad=True)
    ta = torch.tensor(act, requires_grad=True)
    tap = torch.tensor(act_prev, requires_grad=True)
    txi = torch.tensor(xi)
    terms = heads.reward(ts, ta, tap, ref)
    feat = heads.termination_features(ts[..., 0:3], ts[..., 3:7], ref)
    obs = hm.analytic_observation(ts, ta, txi, torch.tensor(q0))
    g_r = rng.normal(size=b)
    g_f = rng.normal(size=(b, 2))
    g_o = rng.normal(size=(b, 93))
    loss = (torch.tensor(g_r) * terms["total"]).sum() + (torch.tensor(g_f) * feat).sum() + (torch.tensor(g_o) * obs).sum()
    loss.backward()
    for name, g in (("s", ts.grad), ("a", ta.grad), ("a_prev", tap.grad)):
        assert torch.isfinite(g).all(), name

    # the reference in the Mojo layout: per tracked body pos | quat | lin | ang, then root pos | quat
    nt = len(tracked)
    ref_flat = np.concatenate([
        np.concatenate([ref.body_pos_w, ref.body_quat_w, ref.body_lin_vel_w, ref.body_ang_vel_w], -1).reshape(b, nt * 13),
        ref.root_pos_w, ref.root_quat_w], -1)

    def save(name, x):
        x = np.asarray(x.detach().numpy() if torch.is_tensor(x) else x, dtype=np.float64)
        np.save(out / f"h_{name}.npy", np.ascontiguousarray(x.reshape(x.shape[0], -1)))

    save("s", s)
    save("a", act)
    save("a_prev", act_prev)
    save("xi", xi)
    save("ref", ref_flat)
    save("q0", q0[None, :])
    save("soft", soft)
    save("tracked", ti.numpy()[None, :].astype(np.float64))
    save("terms", torch.stack([terms[k] for k in names], -1))
    save("total", terms["total"][:, None])
    save("feat", feat)
    save("obs", obs)
    save("g_r", g_r[:, None])
    save("g_feat", g_f)
    save("g_obs", g_o)
    save("g_s", ts.grad)
    save("g_a", ta.grad)
    save("g_a_prev", tap.grad)
    t = torch.stack([terms[k] for k in names], -1).detach().numpy()
    print(f"  [heads] {b} lanes, anchor {anchor}, {nt} tracked; mean terms near/far/taylor:")
    for lo, hi, nm in ((0, third, "near"), (third, 2 * third, "far"), (2 * third, b, "taylor")):
        print(f"    {nm:7s}", " ".join(f"{v:+.4f}" for v in t[lo:hi].mean(0)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--xml", type=Path, default=G1_XML,
                    help="the G1 model (default ours; noeira/envs/robots/assets/unitree_g1_mjlab.xml for mjlab's)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.set_default_dtype(torch.float64)
    fkmod, headsmod = reference_modules()
    rng = np.random.default_rng(a.seed)
    m = mujoco.MjModel.from_xml_path(str(a.xml))
    dump(fkmod, m, a.out, "", a.batch, rng)
    dump_heads(fkmod, headsmod, m, a.out, a.batch, rng)
    # ⚠ EVERY HINGE ANCHOR OF OUR G1 (and of mjlab's) IS AT ITS BODY ORIGIN
    # (`jnt_pos` = 0), so the FK's `P_i − anchor` lever arm is identically
    # zero there and a wrong sign in its VJP passes (a planted mutant did).
    # The `jp_` dump is the same model with random anchors; the Mojo test
    # writes `jp_jnt_pos.npy` into its tree.
    m.jnt_pos[1:] = rng.uniform(-0.05, 0.05, m.jnt_pos[1:].shape)
    np.save(a.out / "jp_jnt_pos.npy", np.ascontiguousarray(m.jnt_pos[1:]))
    dump(fkmod, m, a.out, "jp_", a.batch, rng)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
