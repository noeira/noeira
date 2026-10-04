#!/usr/bin/env python3
"""Sample dataset frames + their states from box session A's pixel fixture,
for the renderer gate (G4b, docs/LEWM_REOPEN_PLAN.md P4).

    pixi run -e act-ref python tools/lewm/sample_fixture_frames.py \
        --fixture ~/.cache/noeira/lewm_pusht/session_a/out/fixture --out /tmp/lewm_frames

Writes (RefDump format): `frames.pixels` (N, 224, 224, 3) uint8 values as
float32, `frames.next_pixels` (the frame one env step later, same episode),
`frames.state` (N, 7) — the dataset's own 7-d state for each frame.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from tools.act.dump_act_reference import Dump  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--out", default="/tmp/lewm_frames")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    f = h5py.File(Path(args.fixture) / "pusht_fixture_pix.h5", "r")
    ep_len, ep_off = f["ep_len"][:], f["ep_offset"][:]
    rng = np.random.default_rng(args.seed)
    rows = []
    for _ in range(args.n):
        e = rng.integers(len(ep_len))
        rows.append(int(ep_off[e] + rng.integers(0, ep_len[e] - 1)))
    rows = np.unique(rows)  # h5py fancy indexing: strictly increasing
    d = Dump(Path(args.out))
    d.add("frames.pixels", f["pixels"][rows].astype(np.float32))
    d.add("frames.next_pixels", f["pixels"][rows + 1].astype(np.float32))
    d.add("frames.state", f["state"][rows].astype(np.float32))
    d.close()


if __name__ == "__main__":
    main()
