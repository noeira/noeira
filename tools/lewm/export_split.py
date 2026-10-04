#!/usr/bin/env python3
# +--------------------------------------------------------------------------+ #
# | LeWM train / validation split — exported from the reference's own code
# +--------------------------------------------------------------------------+ #
"""`train.py`'s 90 / 10 split of the PushT clips, as index lists the Mojo
trainer reads (docs/LEWM_REOPEN_PLAN.md P6.2):

    python tools/lewm/export_split.py --h5 .../pusht_expert_train.h5 --out <dir>

`train.py` does `spt.data.random_split(dataset, [0.9, 0.1],
generator=torch.Generator().manual_seed(cfg.seed))` (seed 3072) over the
stable-worldmodel `HDF5Dataset` of clips (num_steps 4 = history 3 + 1,
frameskip 5). That function is called here on `range(N)` with the same
generator, so the indices are exactly the reference's — and N is checked
against stable-worldmodel's own dataset length, so index k means the same
window to both (episode-major, every start with a full 20-frame span:
`LewmPushTExpert.clip_*`).

Writes `split.train` and `split.val` (float32, exact below 2^24) in the
`Dump` format `RefDump` reads.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from tools.act.dump_act_reference import Dump  # noqa: E402

SPAN = 4 * 5


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--h5", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=3072)
    args = ap.parse_args()

    with h5py.File(args.h5, "r") as f:
        ep_len = f["ep_len"][:]
    n = int(sum(int(x) - SPAN + 1 for x in ep_len if x >= SPAN))
    print(f"clips (ours): {n}")

    import stable_pretraining as spt
    try:
        import stable_worldmodel as swm
        os.environ.setdefault("STABLEWM_HOME", str(Path(args.h5).parent))
        ds = swm.data.HDF5Dataset(
            name=Path(args.h5).stem, num_steps=4, frameskip=5,
            keys_to_load=["pixels", "action", "proprio", "state"],
            keys_to_cache=["action", "proprio", "state"], transform=None,
        )
        assert len(ds) == n, f"stable-worldmodel has {len(ds)} clips, ours {n}"
        print(f"clips (stable-worldmodel HDF5Dataset): {len(ds)} — same")
    except ImportError:
        print("stable_worldmodel not importable: the clip count is NOT cross-checked")

    gen = torch.Generator().manual_seed(args.seed)
    train, val = spt.data.random_split(range(n), lengths=[0.9, 0.1], generator=gen)
    tr = np.asarray(train.indices, dtype=np.float64)
    va = np.asarray(val.indices, dtype=np.float64)
    assert len(tr) + len(va) == n and len(np.intersect1d(tr, va)) == 0
    dump = Dump(Path(args.out))
    dump.add("split.train", tr)
    dump.add("split.val", va)
    dump.close()
    print(f"train {len(tr)}  val {len(va)}  (first train indices {tr[:5].astype(int).tolist()})")


if __name__ == "__main__":
    main()
