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
* `ReplayBuffer` (`data/replay_buffer.py`) VERBATIM, fed a recorded sequence
  of `add` / `end_episode` calls (3 envs, a 40-row ring that wraps, episodes
  of 3–25 steps each closed by a terminal row); `multinomial` is wrapped to
  capture the window and start masks and to pick (and record) eligible rows.
  Files `r_*`.
* realmimic's `_finalize_episodes` with `bin_estimator="count"` (its default;
  `references/realmimic/src/supermimic/envs/mdp/commands.py`) VERBATIM on six
  episodes (two terminated, one at its clip's end, three timed out, one of
  them shorter than the 50-step continuation window), bins starting at 0, 3 or
  25 visits so both the running mean and the EMA floor fold. Files `c_*`.
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


def count_part(ho, lib, tracked, sv, rng) -> None:
    """realmimic's count estimator on six episodes (files `c_*`)."""
    rmod = _load_by_path("realmimic_commands", ho.REPO / "references" / "realmimic" / "src" / "supermimic" / "envs" / "mdp" / "commands.py")
    torch.set_default_dtype(torch.float32)
    cfg = rmod.MultiClipMotionCommandCfg(npz_files=[], anchor_body_name="torso_link", body_names=tracked)
    assert cfg.bin_estimator == "count" and cfg.bootstrap_steps == 50, (cfg.bin_estimator, cfg.bootstrap_steps)
    cmd = rmod.MultiClipMotionCommand.__new__(rmod.MultiClipMotionCommand)
    cmd.cfg = cfg
    cmd.library = lib
    cmd.device = torch.device("cpu")
    nb = lib.num_bins
    E = 6
    cmd.max_ep_steps = 502
    cmd._gamma_pow = cfg.gamma ** torch.arange(cmd.max_ep_steps, dtype=torch.float64)
    cmd.ep_rewards = torch.zeros(E, cmd.max_ep_steps)
    cmd.ep_bins = torch.zeros(E, cmd.max_ep_steps, dtype=torch.long)
    lens = [40, 230, 180, 7, 150, 30]
    term = [True, False, False, True, False, False]
    cend = [False, True, False, False, False, False]   # 2, 4 and 5 time out; 5 is shorter than the 50-step window
    for e in range(E):
        cmd.ep_rewards[e, : lens[e]] = torch.tensor(rng.uniform(0.0, 0.15, lens[e]), dtype=torch.float32)
        cmd.ep_bins[e, : lens[e]] = torch.tensor(rng.integers(0, nb, lens[e]))
    cmd.ep_len = torch.tensor(lens)
    cmd.clip_end = torch.tensor(cend)
    cmd.bootstrap_fn = None
    cmd._env = types.SimpleNamespace(termination_manager=types.SimpleNamespace(terminated=torch.tensor(term)))
    cmd.unphysical = torch.zeros(E, dtype=torch.bool)
    cmd.unphysical_episodes = 0
    cmd.metrics = {"episode_return_to_go": torch.zeros(E)}
    counts0 = rng.choice(np.array([0.0, 3.0, 25.0]), nb)
    cmd.bin_count = torch.tensor(counts0, dtype=torch.float32)
    cmd.bin_return = torch.tensor(np.where(counts0 == 0, 0.0, rng.uniform(0.0, cfg.g_max, nb)), dtype=torch.float32)
    sv("c_rewards", cmd.ep_rewards[:, :max(lens)])
    sv("c_bins", cmd.ep_bins[:, :max(lens)])
    sv("c_lens", np.array(lens, dtype=np.float64)[:, None])
    sv("c_flags", np.stack([np.array(term, dtype=np.float64), np.array(cend, dtype=np.float64)], -1))
    sv("c_return0", cmd.bin_return.clone())
    sv("c_count0", cmd.bin_count.clone())
    cmd._finalize_episodes(torch.arange(E))
    sv("c_return1", cmd.bin_return)
    sv("c_count1", cmd.bin_count)
    touched = int((cmd.bin_count != torch.tensor(counts0, dtype=torch.float32)).sum())
    print(f"  [count] realmimic's estimator: {E} episodes, {touched} of {nb} bins folded")

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
    replay_part(R, a.out, rng)
    count_part(ho, lib, tracked, sv, rng)          # last: the earlier parts keep their random streams
    print(f"-> {a.out}")


def replay_part(R, out: Path, rng) -> None:
    rb_mod = _load_by_path("sm_replay_buffer", R / "data" / "replay_buffer.py")
    E, T, K, HIST = 3, 40, 8, 10
    rb = rb_mod.ReplayBuffer(E, T, 71, 29, 33, "cpu")
    calls = []   # one row per add: e, r, done, frame, terminal, ep_start, end_after, then s | a | a_exec | xi
    for e in range(E):
        for _ in range(7):
            L = int(rng.integers(3, 26))
            xi = rng.normal(0, 1, 33).astype(np.float32)
            for t in range(L + 1):
                term = t == L
                s_ = rng.normal(0, 1, 71).astype(np.float32)
                a_ = np.zeros(29, np.float32) if term else rng.normal(0, 1, 29).astype(np.float32)
                ae = np.zeros(29, np.float32) if term else (a_ + 0.1 * rng.normal(0, 1, 29)).astype(np.float32)
                r_ = 0.0 if term else float(np.float32(rng.uniform(0, 0.1)))
                d_ = float(term and rng.random() < 0.6)
                fr = int(rng.integers(0, 274))
                st = t == 0
                ids = torch.tensor([e])
                rb.add(ids, s=torch.tensor(s_)[None], a=torch.tensor(a_)[None], a_exec=torch.tensor(ae)[None],
                       xi=torch.tensor(xi)[None], r=torch.tensor([r_]), done=torch.tensor([d_]), frame=torch.tensor([fr]),
                       clip=torch.tensor([0]), terminal=torch.tensor([term]), ep_start=torch.tensor([st]))
                if term:
                    rb.end_episode(ids)
                calls.append(np.concatenate([[e, r_, d_, fr, float(term), float(st), float(term)], s_, a_, ae, xi]))
    np.save(out / "r_calls.npy", np.stack(calls).astype(np.float64))
    masks, picks = [], []
    real_mn = torch.multinomial

    def pick(p_, n, replacement=False):
        masks.append(p_.clone())
        elig = np.nonzero(p_.numpy() > 0)[0]
        idx = torch.tensor(rng.choice(elig, n))
        picks.append(idx.clone())
        return idx

    torch.multinomial = pick
    try:
        w = rb.sample_windows(12, K)
        st = rb.sample_starts(12, HIST)
    finally:
        torch.multinomial = real_mn
    sv = lambda name, x: np.save(out / f"{name}.npy", np.ascontiguousarray(np.asarray(x.detach().numpy() if torch.is_tensor(x) else x, dtype=np.float64).reshape(len(x), -1)))  # noqa: E731
    sv("r_wmask", masks[0][:, None])
    sv("r_smask", masks[1][:, None])
    sv("r_wpick", picks[0][:, None])
    sv("r_spick", picks[1][:, None])
    sv("r_w_states", w.states.reshape(-1, 71))
    sv("r_w_a_exec", w.a_exec.reshape(-1, 29))
    sv("r_w_xi", w.xi)
    sv("r_w_done", w.done.reshape(-1, 1))
    sv("r_w_frame", w.frame.reshape(-1, 1))
    sv("r_s_state", st.state)
    sv("r_s_prev_action", st.prev_action)
    sv("r_s_hist_states", st.hist_states.reshape(-1, 71))
    sv("r_s_hist_actions", st.hist_actions.reshape(-1, 29))
    print(f"  [replay] {len(calls)} adds over {E} envs x {T} rows; {int(masks[0].sum())} window rows, {int(masks[1].sum())} start rows")


if __name__ == "__main__":
    main()
