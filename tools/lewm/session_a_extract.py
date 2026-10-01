#!/usr/bin/env python3
# +--------------------------------------------------------------------------+ #
# | LeWM box session A — the 50 eval pairs, the stats, and the fixture
# +--------------------------------------------------------------------------+ #
"""Run ON THE BOX, inside the stable-worldmodel 0.1.1 venv that ran the
reference eval (`tools/lewm/box_session_a.sh` calls it). Writes everything the
laptop needs from the 47 GB dataset into `--out` (target < 300 MB), in the
`tools/act/dump_act_reference.py` format (`<name>.bin` float32 + manifest.txt,
read by `noeira.deep_agents.act.refload.RefDump`), plus one small HDF5 in the
dataset's own schema for `noeira.nn.datasets.LewmPushTExpert`.

## What is written (docs/LEWM_REOPEN_PLAN.md §5)

* `pairs.*` — THE 50 EVAL PAIRS column R played: the `random_episode_indices`
  `references/le-wm-main/eval.py` PRINTED (`--eval-log`) are authoritative.
  They are also recomputed here from eval.py's sampling code (seed 42, goal
  offset 25) over the raw columns, and a mismatch aborts: either our reading
  of eval.py or of the file is wrong, and a different 50 would make R vs M a
  comparison of two samples, not two stacks. Written per pair: row, episode,
  start step, start/goal `state` + `proprio`, the 25 recorded expert actions
  between them, start/goal frames (uint8 values stored as float32).
* `stats.*` — per-column mean / scale for `action`, `proprio`, `state`, both
  ways the reference computes them: eval's `sklearn StandardScaler`
  (population std — what the PLANNER normalises with) and training's
  `get_column_normalizer` (torch `std`, unbiased — what the MODEL was
  trained on). NaN rows dropped, as both do.
* `dyn.*` — `state`, `action`, `proprio`, `ep_len`, `ep_offset` for
  `--dyn-episodes` episodes (no pixels): P4's dynamics replay — set our
  PushT to a recorded state, apply the recorded actions, compare against the
  recorded states.
* `pusht_fixture_pix.h5` — `--pix-episodes` whole episodes WITH pixels, in
  the dataset's schema: real frames for the P2 gates and P4's renderer check.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from tools.act.dump_act_reference import Dump  # noqa: E402

GOAL_OFFSET = 25  # config/eval/pusht.yaml eval.goal_offset_steps
NUM_EVAL = 50     # eval.num_eval
SEED = 42         # seed


def eval_rows(episode_idx: np.ndarray, step_idx: np.ndarray) -> np.ndarray:
    """`references/le-wm-main/eval.py`, "sample the episodes and the starting
    indices" through `np.sort(...)`, over the raw columns. swm's
    `get_col_data` returns a column in file row order, so row index = file
    row; the cross-check against eval.py's printed indices holds us to it."""
    ep_indices = np.unique(episode_idx)
    episode_len = np.array(
        [np.max(step_idx[episode_idx == ep]) + 1 for ep in ep_indices]
    )
    max_start_idx = episode_len - GOAL_OFFSET - 1
    max_start = {ep: max_start_idx[i] for i, ep in enumerate(ep_indices)}
    max_start_per_row = np.array([max_start[ep] for ep in episode_idx])
    valid_indices = np.nonzero(step_idx <= max_start_per_row)[0]
    g = np.random.default_rng(SEED)
    pick = g.choice(len(valid_indices) - 1, size=NUM_EVAL, replace=False)
    return np.sort(valid_indices[pick])


def printed_rows(log: Path) -> np.ndarray | None:
    """The `print(random_episode_indices)` line(s) of eval.py's stdout."""
    text = log.read_text()
    m = re.search(r"valid starting points found for evaluation\.\s*\n(\[[^\]]*\])", text)
    if not m:
        return None
    return np.array([int(x) for x in m.group(1).strip("[]").split()])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True, help="pusht_expert_train.h5")
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval-log", help="eval.py stdout, for the row cross-check")
    ap.add_argument("--dyn-episodes", type=int, default=2000)
    ap.add_argument("--pix-episodes", type=int, default=8)
    args = ap.parse_args()

    h5 = Path(args.h5)
    out = Path(args.out)
    dump = Dump(out)

    f = h5py.File(h5, "r")
    ep_col = "episode_idx" if "episode_idx" in f else "ep_idx"
    episode = f[ep_col][:]
    step = f["step_idx"][:]
    ep_len = f["ep_len"][:]
    ep_offset = f["ep_offset"][:]
    state, action, proprio = f["state"], f["action"], f["proprio"]

    # ---- the 50 pairs: eval.py's printed rows, recomputed as a check -----
    rows = eval_rows(episode, step)
    if args.eval_log:
        got = printed_rows(Path(args.eval_log))
        if got is None:
            sys.exit("eval log has no printed random_episode_indices")
        if not np.array_equal(got, rows):
            sys.exit(f"ROW MISMATCH vs eval.py's printed indices:\n{got}\n{rows}")
        print(f"  pairs: recomputed rows == eval.py's printed indices ({len(rows)})")
    else:
        print("  ⚠ pairs: no --eval-log, rows NOT cross-checked against eval.py")

    starts, goals = rows, rows + GOAL_OFFSET
    assert np.all(episode[goals] == episode[starts]), "goal row left the episode"
    assert np.all(step[goals] == step[starts] + GOAL_OFFSET)
    dump.add("pairs.row", rows.astype(np.float64))
    dump.add("pairs.episode", episode[starts].astype(np.float64))
    dump.add("pairs.start_step", step[starts].astype(np.float64))
    for name, col in (("state", state), ("proprio", proprio)):
        dump.add(f"pairs.start_{name}", col[starts])
        dump.add(f"pairs.goal_{name}", col[goals])
    dump.add("pairs.expert_actions",
             np.stack([action[r:r + GOAL_OFFSET] for r in starts]))
    dump.add("pairs.start_pixels", np.stack([f["pixels"][r] for r in starts]))
    dump.add("pairs.goal_pixels", np.stack([f["pixels"][r] for r in goals]))

    # ---- normalisation stats, both ways the reference computes them ------
    for name in ("action", "proprio", "state"):
        x = f[name][:].astype(np.float64)
        x = x[~np.isnan(x).any(axis=1)]
        dump.add(f"stats.{name}_mean", x.mean(0))
        dump.add(f"stats.{name}_scale_eval", x.std(0, ddof=0))   # StandardScaler
        dump.add(f"stats.{name}_std_train", x.std(0, ddof=1))    # torch .std()

    # ---- dynamics replay set (no pixels) ---------------------------------
    g = np.random.default_rng(0)
    eps = np.sort(g.choice(len(ep_len), size=min(args.dyn_episodes, len(ep_len)),
                           replace=False))
    sl = [slice(int(ep_offset[e]), int(ep_offset[e] + ep_len[e])) for e in eps]
    dump.add("dyn.episode", eps.astype(np.float64))
    dump.add("dyn.ep_len", ep_len[eps].astype(np.float64))
    lens = ep_len[eps].astype(np.int64)
    dump.add("dyn.ep_offset", np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.float64))
    for name, col in (("state", state), ("action", action), ("proprio", proprio)):
        dump.add(f"dyn.{name}", np.concatenate([col[s] for s in sl]))
    dump.close()

    # ---- a few whole episodes WITH pixels, in the dataset's schema --------
    pe = eps[: args.pix_episodes]
    with h5py.File(out / "pusht_fixture_pix.h5", "w") as o:
        lens = ep_len[pe].astype(np.int32)
        offs = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.int64)
        o["ep_len"] = lens
        o["ep_offset"] = offs
        for key in ("pixels", "action", "proprio", "state", ep_col, "step_idx"):
            o[key] = np.concatenate(
                [f[key][int(ep_offset[e]):int(ep_offset[e] + ep_len[e])] for e in pe]
            )
    f.close()
    print(f"  fixture: {len(eps)} dynamics episodes, {len(pe)} pixel episodes -> {out}")


if __name__ == "__main__":
    main()
