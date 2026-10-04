#!/usr/bin/env python3
# +--------------------------------------------------------------------------+ #
# | PushT dynamics oracle: the dataset's actions replayed in pymunk
# +--------------------------------------------------------------------------+ #
"""docs/LEWM_REOPEN_PLAN.md P4, G4a. Run where stable-worldmodel 0.0.6 is
installed (box session A's venv): pymunk is the ORACLE here, never part of
the stack.

    $VENV/bin/python tools/lewm/pusht_replay_oracle.py \
        --fixture ~/.cache/noeira/lewm_pusht/session_a/out/fixture --out /tmp/pusht_oracle

For N_STARTS starts sampled inside the fixture's 2000 dynamics episodes:
`env._set_state(state[r])`, then the recorded raw actions r..r+K-1 through
`env.step`, recording the 7-d state after each step. Writes
`oracle.starts` (episode, row), `oracle.replay` (N, K+1, 7) — entry 0 is the
state right after `_set_state` — in the RefDump format.

Why an oracle at all: the dataset's `state` has NO block velocity, and
`_set_state` leaves the block's velocity at whatever it was (zero here) and
advances the physics one substep. So even pymunk cannot replay the dataset
exactly from a mid-episode row: its error against the recorded states is the
FLOOR any engine reaches from this initialisation. Our native PushT is judged
against that floor (and against pymunk's replay directly), not against zero.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from tools.act.dump_act_reference import Dump  # noqa: E402
from tools.lewm.convert_ref_to_ours import read_dump  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-starts", type=int, default=400)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import gymnasium as gym
    import stable_worldmodel  # noqa: F401  (registers swm/PushT-v1)
    import importlib.metadata as md
    print("  stable-worldmodel", md.version("stable-worldmodel"))

    fx = read_dump(Path(args.fixture))
    state = fx["dyn.state"].astype(np.float64)
    action = fx["dyn.action"].astype(np.float64)
    ep_len = fx["dyn.ep_len"].astype(np.int64)
    ep_off = fx["dyn.ep_offset"].astype(np.int64)
    K = args.steps

    rng = np.random.default_rng(args.seed)
    eligible = np.nonzero(ep_len > K + 1)[0]
    eps = rng.choice(eligible, size=args.n_starts, replace=True)
    starts = np.array([ep_off[e] + rng.integers(0, ep_len[e] - K - 1) for e in eps])

    env = gym.make("swm/PushT-v1")
    env.reset(seed=args.seed)
    pt = env.unwrapped
    replay = np.zeros((len(starts), K + 1, 7))
    for i, r in enumerate(starts):
        pt.block.velocity = (0.0, 0.0)  # what a fresh env holds; not in `state`
        pt.block.angular_velocity = 0.0
        pt._set_state(state[r])
        replay[i, 0] = pt._get_obs()
        for k in range(K):
            obs, *_ = pt.step(action[r + k].astype(np.float32))
            replay[i, k + 1] = obs["state"]

    out = Dump(Path(args.out))
    out.add("oracle.starts", np.stack([eps, starts], 1).astype(np.float64))
    out.add("oracle.replay", replay)
    out.close()

    # the floor, printed for the record: pymunk vs the recorded states
    for k in (1, 5, 25, 50):
        if k > K:
            continue
        rec = np.stack([state[r + k] for r in starts])
        d = replay[:, k]
        pos = np.linalg.norm(d[:, :4] - rec[:, :4], axis=1)
        ang = np.abs(d[:, 4] - rec[:, 4])
        ang = np.minimum(ang, 2 * np.pi - ang)
        print(f"  pymunk vs recorded after {k:2d} steps: |d(agent,block)| median "
              f"{np.median(pos):.3g} px, p90 {np.percentile(pos, 90):.3g}; "
              f"|d angle| median {np.median(ang):.3g} rad")


if __name__ == "__main__":
    main()
