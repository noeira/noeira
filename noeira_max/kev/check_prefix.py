"""`hidden_shared` (the prefix computed once, then the rows graph) against `hidden` (each row whole, one prefix-graph
call per row), same token rows.

    PYTHONPATH=<this dir> python check_prefix.py <kev9b-gptq-q4g32 dir>

Synthetic token ids shaped like one room decision: a ~100-token state shared by a long command row and four short
question rows. Per row: max |h_shared - h_whole| / max |h_whole| and the mean per-token cosine, over the WHOLE row
(prefix states included). Then the time of each path for that request.
"""
import sys
import time

import numpy as np

from qwen35_backbone import Qwen35Backbone

rng = np.random.default_rng(0)
prefix = rng.integers(1000, 30000, 104).tolist()
suffixes = [rng.integers(1000, 30000, n).tolist() for n in (760, 62, 48, 40, 36)]

t = time.perf_counter()
bb = Qwen35Backbone(sys.argv[1])
print(f"built + compiled in {time.perf_counter() - t:.0f} s")

whole = bb.hidden([prefix + s for s in suffixes])
hp, hs = bb.hidden_shared(prefix, suffixes)
worst = 0.0
for i, s in enumerate(suffixes):
    g, ref = np.concatenate([hp, hs[i]]), whole[i]
    rel = float(np.abs(g - ref).max() / np.abs(ref).max())
    cos = float(np.mean(np.sum(g * ref, -1) / (np.linalg.norm(g, axis=-1) * np.linalg.norm(ref, axis=-1))))
    worst = max(worst, rel)
    print(f"  row {i} suffix {len(s):4d}  rel max err {rel:.4f}  mean cos {cos:.6f}")

seq = sys.argv[2] if len(sys.argv) > 2 else "ws"  # the call order; alternating shapes is what crashed the 1-graph form
for c in seq * 4:
    (bb.hidden([prefix + s for s in suffixes]) if c == "w" else bb.hidden_shared(prefix, suffixes))
print("alternated", seq * 4, "OK")
for name, fn in (("whole, per row", lambda: bb.hidden([prefix + s for s in suffixes])),
                 ("shared prefix", lambda: bb.hidden_shared(prefix, suffixes))):
    fn()
    n = 5
    t = time.perf_counter()
    for _ in range(n):
        fn()
    print(f"{name:14s} {(time.perf_counter() - t) / n * 1e3:7.1f} ms")
print("worst rel err", f"{worst:.4f}")
