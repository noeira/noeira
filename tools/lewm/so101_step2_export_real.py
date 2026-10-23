"""Real teleop frames for step 2 of the SO-101 world-model plan.

noeira-docs/SO101_LEWM_PLAN.md §3c step 2; consumed by
`so101_frozen_encoder_step2.py` (as `real_step2.h5`).

    pixi run python -I tools/lewm/so101_step2_export_real.py STORE \\
        TWIN.demo.map.txt OUT.h5 tools/lewm

STORE is the undistorted teleop store (`act_so101_import_dataset.mojo
--undistort`, 240 x 320). Keeps only the rows step 2 encodes: the twin rows
(`lewm_so101_twin.mojo`'s map: twin ep -> real ep, first row, rows; every
STRIDE ticks) and the pre-close labelled rows (`so101_preclose_labels`), with
their indices, so the box does not need the 7 GB store.
"""
import sys, h5py, numpy as np
sys.path.insert(0, sys.argv[4])
from so101_preclose_labels import store_labels
STRIDE = 5
store, mapf, out = sys.argv[1:4]
f = h5py.File(store, "r")
tw = [l.split() for l in open(mapf) if not l.startswith("#") and l.strip()]
twin_rows, twin_ep, twin_tick = [], [], []
for te, re_, r0, n in tw:
    for t in range(0, int(n), STRIDE):
        twin_rows.append(int(r0) + t); twin_ep.append(int(te)); twin_tick.append(t)
row, ep, tick, y, kind = store_labels(f)
allr = np.unique(np.concatenate([twin_rows, row]))
img = f["images"]
with h5py.File(out, "w") as g:
    g["rows"] = allr
    g.create_dataset("images", data=np.stack([img[r] for r in allr]), compression="gzip", compression_opts=4)
    g["qpos"] = f["qpos"][:][allr]
    g["twin_rows"], g["twin_ep"], g["twin_tick"] = np.array(twin_rows), np.array(twin_ep), np.array(twin_tick)
    g["lab_rows"], g["lab_ep"], g["lab_tick"], g["lab_y"], g["lab_kind"] = row, ep, tick, y, kind
print(out, len(allr), "rows |", len(twin_rows), "twin |", len(row), "labelled")
