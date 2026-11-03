#!/usr/bin/env python3
# +--------------------------------------------------------------------------+ #
# | SuperMimic M2 oracle: the reproduction's motion library and reset sampler
# +--------------------------------------------------------------------------+ #
"""Reference tensors for `tests/deep_agents/supermimic/test_supermimic_lib.mojo`.

    uv run --no-project --python 3.12 --with torch --with mujoco==3.12.0 --with numpy \\
        python tools/supermimic/lib_oracle.py --out /tmp/sm_lib

* `MotionLibrary` (`envs/mdp/motion_library.py`) VERBATIM in compact mode on a
  synthetic packed library (`build_library.py`'s `{"ref", "names", "starts",
  "lengths"}`) of four clips, 73 / 120 / 50 / 31 frames, so bins end short.
  Its FK is the reproduction's `G1Kinematics` on OUR model (its
  `load_g1_model` is pointed at our XML). Files `l_*`.
* `MultiClipMotionCommand._sample_frames` and `_finalize_episodes`
  (`envs/mdp/commands.py`) VERBATIM on a command object built without its
  env (`__new__` + the attributes those two methods read). `torch.multinomial`
  is wrapped to record the bin distribution and inject the bins; `torch.rand`
  injects the uniform draws. Files `s_*` (sampler) and `f_*` (finalize).
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
LENS = (73, 120, 50, 31)


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def f32(x):
    return np.asarray(x, dtype=np.float32).astype(np.float64)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=19)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    ho = _load_by_path("sm_heads_oracle", HERE / "heads_oracle.py")
    fkmod, hm = ho.reference_modules()
    R = ho.REF
    m = mujoco.MjModel.from_xml_path(str(ho.G1_XML))
    ho._pkg("supermimic.data")
    mc = types.ModuleType("supermimic.data.motion_convert")
    mc.load_g1_model = lambda *args, **kw: m
    sys.modules["supermimic.data.motion_convert"] = mc
    mlib = ho._load("supermimic.envs.mdp.motion_library", R / "envs" / "mdp" / "motion_library.py")
    mgr = types.ModuleType("mjlab.managers")
    mgr.CommandTerm = object
    mgr.CommandTermCfg = object
    sys.modules["mjlab.managers"] = mgr
    cmdmod = ho._load("supermimic.envs.mdp.commands", R / "envs" / "mdp" / "commands.py")
    tracked = ho._literal(R / "envs" / "tracking_env_cfg.py", "TRACKED_BODIES")
    rng = np.random.default_rng(a.seed)

    # ── the packed library ──
    rows = []
    for c, n in enumerate(LENS):
        t = np.arange(n) / 50.0
        r = np.zeros((n, 71))
        r[:, 0] = 0.4 * t + c
        r[:, 1] = 0.2 * np.sin(t + c)
        r[:, 2] = 0.75 + 0.03 * np.sin(2 * t)
        yaw = 0.5 * t + 0.7 * c
        roll = 0.1 * np.sin(3 * t)
        r[:, 3] = np.cos(yaw / 2) * np.cos(roll / 2)
        r[:, 4] = np.cos(yaw / 2) * np.sin(roll / 2)
        r[:, 5] = np.sin(yaw / 2) * np.sin(roll / 2)
        r[:, 6] = np.sin(yaw / 2) * np.cos(roll / 2)
        ph = rng.uniform(0, 6.28, 29)
        r[:, 7:36] = 0.4 * np.sin(1.5 * t[:, None] + ph[None, :])
        r[:, 36:39] = rng.normal(0, 0.4, 3)
        r[:, 39:42] = rng.normal(0, 0.6, 3)
        r[:, 42:71] = 0.6 * np.cos(1.5 * t[:, None] + ph[None, :])
        rows.append(r)
    ref = f32(np.concatenate(rows))
    starts = np.cumsum((0,) + LENS[:-1]).tolist()
    pt = a.out / "l_library.pt"
    torch.save({"ref": torch.tensor(ref), "names": [f"clip{c}" for c in range(len(LENS))], "starts": starts, "lengths": list(LENS)}, pt)
    q0 = f32(rng.uniform(-0.3, 0.3, 29))
    lib = mlib.MotionLibrary([pt], tracked, torch.tensor(q0), device="cpu", dtype=torch.float64, compact=True, store_dtype=torch.float64)
    F = lib.num_frames
    frames = torch.arange(F)
    goal = lib.goal_window(frames, 10)
    b = lib.bodies(frames)
    r = lib.ref(frames)
    refrow = torch.cat([torch.cat([b.pos_w, b.quat_w, b.lin_vel_w, b.ang_vel_w], -1).reshape(F, -1), r.root_pos_w, r.root_quat_w], -1)
    sv = lambda name, x: np.save(a.out / f"{name}.npy", np.ascontiguousarray(np.asarray(x.detach().numpy() if torch.is_tensor(x) else x, dtype=np.float64).reshape(len(x), -1)))  # noqa: E731
    sv("l_ref", ref)
    sv("l_q0", q0[None, :])
    sv("l_lens", np.array(LENS, dtype=np.float64)[:, None])
    sv("l_bins", torch.stack([lib.bin_start, lib.bin_end, lib.bin_clip], -1))
    sv("l_last", lib.last_frame_of(frames))
    sv("l_bin_of", lib.bin_of_frame(frames))
    sv("l_goal", goal)
    sv("l_refrow", refrow)
    print(f"  [library] {F} frames, {lib.num_clips} clips, {lib.num_bins} bins; goal {tuple(goal.shape)}, ref rows {tuple(refrow.shape)}")

    # ── the sampler ──
    torch.set_default_dtype(torch.float32)          # the command's tensors are float32 in training
    cfg = cmdmod.MultiClipMotionCommandCfg(npz_files=[], anchor_body_name="torso_link", body_names=tracked)
    cmd = cmdmod.MultiClipMotionCommand.__new__(cmdmod.MultiClipMotionCommand)
    cmd.cfg = cfg
    cmd.library = lib
    cmd.device = torch.device("cpu")
    nb = lib.num_bins
    cmd.bin_return = torch.tensor(rng.uniform(0.0, cfg.g_max, nb), dtype=torch.float32)
    cmd.bin_memory = torch.tensor(unit_mem := rng.uniform(0.5, 1.5, nb) / nb, dtype=torch.float32)
    cmd.metrics = {"sampling_entropy": torch.zeros(8), "episode_return_to_go": torch.zeros(8)}
    br0, bm0 = cmd.bin_return.clone(), cmd.bin_memory.clone()
    n_draw = 3 * nb
    inj_b = torch.tensor([i % nb for i in range(n_draw)])
    inj_u = torch.tensor(rng.uniform(0, 1, n_draw), dtype=torch.float32)
    seen_p = []
    real_mn, real_rand = torch.multinomial, torch.rand
    torch.multinomial = lambda p, n, replacement=False: (seen_p.append(p.clone()), inj_b)[1]
    torch.rand = lambda n, device=None: inj_u
    try:
        fr = cmd._sample_frames(n_draw)
    finally:
        torch.multinomial, torch.rand = real_mn, real_rand
    sv("s_return0", br0)
    sv("s_memory0", bm0)
    sv("s_p", seen_p[0])
    sv("s_memory1", cmd.bin_memory)
    sv("s_b", inj_b)
    sv("s_u", inj_u)
    sv("s_frames", fr)
    _ = unit_mem

    # ── finalize ──
    E = 6
    cmd.max_ep_steps = 502
    cmd._gamma_pow = cfg.gamma ** torch.arange(cmd.max_ep_steps, dtype=torch.float64)
    cmd.ep_rewards = torch.zeros(E, cmd.max_ep_steps)
    cmd.ep_bins = torch.zeros(E, cmd.max_ep_steps, dtype=torch.long)
    lens = [40, 230, 180, 7, 150, 120]
    term = [True, False, False, True, False, False]
    cend = [False, True, False, False, False, False]   # env 2, 4 and 5 are TRUNCATED
    for e in range(E):
        cmd.ep_rewards[e, : lens[e]] = torch.tensor(rng.uniform(0.0, 0.15, lens[e]), dtype=torch.float32)
        cmd.ep_bins[e, : lens[e]] = torch.tensor(rng.integers(0, nb, lens[e]))
    cmd.ep_len = torch.tensor(lens)
    cmd.clip_end = torch.tensor(cend)
    cmd.bootstrap_fn = None
    cmd._env = types.SimpleNamespace(termination_manager=types.SimpleNamespace(terminated=torch.tensor(term)))
    cmd.bin_return = torch.tensor(rng.uniform(0.0, cfg.g_max, nb), dtype=torch.float32)
    f0 = cmd.bin_return.clone()
    ids = torch.arange(E)
    sv("f_rewards", cmd.ep_rewards[:, :max(lens)])
    sv("f_bins", cmd.ep_bins[:, :max(lens)])
    cmd._finalize_episodes(ids)
    sv("f_lens", np.array(lens, dtype=np.float64)[:, None])
    sv("f_flags", np.stack([np.array(term, dtype=np.float64), np.array(cend, dtype=np.float64)], -1))
    sv("f_return0", f0)
    sv("f_return1", cmd.bin_return)
    print(f"  [sampler] {n_draw} draws over {nb} bins; finalize {E} episodes ({sum(1 for t_, c_ in zip(term, cend) if not t_ and not c_)} truncated)")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
