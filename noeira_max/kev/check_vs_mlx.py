"""The MAX backbone against Kev-on-MLX (mlx-lm's qwen3_5), same checkpoint, same token rows.

    PYTHONPATH=<kev checkout>:<this dir> python check_vs_mlx.py <kev9b-gptq-q4g32 dir> <trap or calib file>

Per row: max |h_max - h_mlx| / max |h_mlx| over the final hidden states, and the mean per-token cosine. Then the
MAX call's time for one decision-sized batch (state + the request and gate questions, as gptq_kev.py encodes them).
"""
import sys
import time

import mlx.core as mx
import numpy as np

from kev.checkpoint import LoadOptions, load
from kev.model import rows_of

import gptq_kev
from qwen35_backbone import Qwen35Backbone

ckpt, data = sys.argv[1], sys.argv[2]
tok, m = load(ckpt, "cpu", LoadOptions(backend="mlx"))
recs = gptq_kev.records([data])[:4]
rows = []
for rec in recs:
    S, _, rs = rows_of(m.encode(tok, rec))
    rows += [S + r["ids"] for r in rs]
print(f"{len(rows)} rows, lengths {[len(r) for r in rows]}")

t = time.perf_counter()
bb = Qwen35Backbone(ckpt)
print(f"MAX backbone built + compiled in {time.perf_counter() - t:.0f} s")

got = bb.hidden(rows)
worst = 0.0
for i, r in enumerate(rows):
    ref = np.array(m.text(mx.array([r], dtype=mx.int32))[0].astype(mx.float32))
    g = got[i]
    rel = float(np.abs(g - ref).max() / np.abs(ref).max())
    cos = float(np.mean(np.sum(g * ref, -1) / (np.linalg.norm(g, axis=-1) * np.linalg.norm(ref, axis=-1))))
    worst = max(worst, rel)
    print(f"  row {i} L={len(r):4d}  rel max err {rel:.4f}  mean cos {cos:.6f}")

batch = rows[:3]  # one record: state + 3 question rows
for _ in range(2):
    bb.hidden(batch)
n = 10
t = time.perf_counter()
for _ in range(n):
    bb.hidden(batch)
print(f"MAX hidden(): {(time.perf_counter() - t) / n * 1e3:.1f} ms for {sum(map(len, batch))} tokens in {len(batch)} rows")
