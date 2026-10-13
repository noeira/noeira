#!/usr/bin/env python3
# +--------------------------------------------------------------------------+ #
# | GPTQ on the Q4_0 grid in numpy: the fixture noeira's GPTQ is gated against
# +--------------------------------------------------------------------------+ #
"""Write `tests/fixtures/quant/gptq_q4_0.safetensors`: weights, Hessians, and
their GPTQ codes, scales and objectives, computed in numpy.

    pixi run python tools/quant/gptq_q4_0_reference.py --out tests/fixtures/quant
    pixi run mojo run -I . tests/nn/quant/test_gptq.mojo

## The algorithm (`noeira/nn/quant/gptq.mojo` documents the same steps)

    H (fp32): dead columns (diag 0) -> diag 1 and their weights zeroed;
    H += 0.01 * mean(diag H) * I, in fp32
    U = upper Cholesky factor of H^-1, in fp64 (chol, inverse, chol), -> fp32
    blocks of 128 columns, fp32:
        per column c: at c % 32 == 0 the group's scale from the current
        weights (the RTN grid of `q4_0_rtn_reference.py`), the code, then
        e = (w[:, c] - deq) / U[c, c];  w[:, c:end] -= e U[c, c:end]
        after the block: w[:, end:] -= E @ U[block, end:]

This is the GPTQ pipeline that quantized Kev for MAX, transcribed from MLX to
numpy; the agreement of this transcription with it on this file's cases is
recorded in the results doc. Both run the loop in fp32, so they agree up to
the order of floating-point operations (see `test_gptq.mojo`).

## The cases

    a  64 x 160: a partial last block (32 columns); H from 96 rows, rank
       deficient before the damping; 3 dead inputs
    b  96 x 384: three blocks; H from 1024 rows whose columns are scaled
       over 100x (a few outlier channels, as in a transformer's
       activations); 2 dead inputs; 4 weight rows 10x the others

Per case: w [n, k], h [k, k] (undamped fp32), q, d (GPTQ), and obj = [GPTQ,
RTN] = tr((W - What) H (W - What)^T) in fp64 for both quantizers.
"""
import argparse
from pathlib import Path

import numpy as np

from q4_0_rtn_reference import GS, bf16_round, rtn_q4_0, save_safetensors

BLOCK = 128


def scale(g):
    """[N, GS] -> [N] Q4_0 scales: first max magnitude / -8, 1e-8 for 0, bf16."""
    idx = np.argmax(np.abs(g), axis=1)
    d = g[np.arange(g.shape[0]), idx] / np.float32(-8.0)
    d = np.where(d == 0, np.float32(1e-8), d).astype(np.float32)
    return bf16_round(d)


def hinv_upper(h32):
    h = h32.astype(np.float64)
    l = np.linalg.cholesky(h)
    li = np.linalg.inv(l)
    return np.linalg.cholesky(li.T @ li).T.astype(np.float32)


def gptq_q4_0(w, h):
    n, k = w.shape
    w = w.astype(np.float32).copy()
    h = h.astype(np.float32).copy()
    dead = np.diag(h) == 0
    h[dead, dead] = 1.0
    w[:, dead] = 0.0
    damp = np.float32(0.01) * np.mean(np.diag(h), dtype=np.float32)
    h = (h + damp * np.eye(k, dtype=np.float32)).astype(np.float32)
    u = hinv_upper(h)
    q = np.zeros((n, k), np.uint8)
    d = np.zeros((n, k // GS), np.float32)
    for i1 in range(0, k, BLOCK):
        i2 = min(i1 + BLOCK, k)
        err = np.zeros((n, i2 - i1), np.float32)
        dg = None
        for c in range(i1, i2):
            if c % GS == 0:
                dg = scale(w[:, c:c + GS])
                d[:, c // GS] = dg
            x = w[:, c].copy()
            code = np.clip(np.round(x / dg) + np.float32(8.0), 0.0, 15.0).astype(np.float32)
            q[:, c] = code.astype(np.uint8)
            e = ((x - (code - np.float32(8.0)) * dg) / u[c, c]).astype(np.float32)
            w[:, c:i2] = (w[:, c:i2] - e[:, None] * u[c, c:i2][None, :]).astype(np.float32)
            err[:, c - i1] = e
        if i2 < k:
            w[:, i2:] = (w[:, i2:] - err @ u[i1:i2, i2:]).astype(np.float32)
    return q, d, int(dead.sum()), float(damp)


def objective(w, q, d, h):
    deq = (q.astype(np.float64) - 8.0) * np.repeat(d.astype(np.float64), GS, axis=1)
    delta = w.astype(np.float64) - deq
    return float(np.sum((delta @ h.astype(np.float64)) * delta))


def case(rng, n, k, rows, dead, col_spread, outlier_rows):
    x = rng.standard_normal((rows, k)).astype(np.float32)
    x *= np.geomspace(1.0, col_spread, k).astype(np.float32)[rng.permutation(k)]
    x[:, rng.choice(k, dead, replace=False)] = 0.0
    h = (x.T @ x).astype(np.float32)
    w = (rng.standard_normal((n, k)) * 0.05).astype(np.float32)
    w[rng.choice(n, outlier_rows, replace=False)] *= 10.0
    return w, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(20261013)
    cases = {
        "a": case(rng, 64, 160, 96, 3, 10.0, 0),
        "b": case(rng, 96, 384, 1024, 2, 100.0, 4),
    }
    tensors = {}
    for name, (w, h) in cases.items():
        q, d, n_dead, damp = gptq_q4_0(w, h)
        qr, dr = rtn_q4_0(w)
        og, orr = objective(w, q, d, h), objective(w, qr, dr, h)
        tensors.update({f"{name}.w": w, f"{name}.h": h, f"{name}.q": q, f"{name}.d": d,
                        f"{name}.obj": np.array([og, orr], np.float32)})
        print(f"case {name}: w {w.shape}, {n_dead} dead, damp {damp:.6g}; objective GPTQ {og:.6g} "
              f"RTN {orr:.6g} (GPTQ / RTN {og / orr:.3f}); codes differing from RTN "
              f"{int(np.sum(q != qr))} of {q.size}, scales {int(np.sum(d != dr))} of {d.size}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    save_safetensors(out / "gptq_q4_0.safetensors", tensors)
    print(f"wrote {out / 'gptq_q4_0.safetensors'}")


if __name__ == "__main__":
    main()
