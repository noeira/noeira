#!/usr/bin/env python3
# +--------------------------------------------------------------------------+ #
# | The Q4_0 RTN grid in numpy: the fixture noeira's quantizer must reproduce
# +--------------------------------------------------------------------------+ #
"""Write `tests/fixtures/quant/q4_0_rtn.safetensors`: a fixed weight and its
round-to-nearest Q4_0 codes and scales, computed in numpy.

    pixi run python tools/quant/q4_0_rtn_reference.py --out tests/fixtures/quant
    pixi run mojo run -I . tests/nn/quant/test_q4_0.mojo     # bit for bit

## The grid (llama.cpp's Q4_0, scale rounded to bf16, as MAX's GEMM reads it)

    per group of 32 consecutive weights of a row ([out, in], grouped along in):
    d = the element of largest magnitude, the FIRST one on a tie, / -8
    d = 1e-8 where that is 0; then d rounded to bf16, ties to even
    q = clip(round(w / d) + 8, 0, 15), numpy's round (ties to even)

This is the grid of the GPTQ pipeline that quantized Kev for MAX (its `--rtn`
mode). That pipeline is written in MLX; this numpy transcription equals it
on every code and scale of this file's matrix (checked once, on MLX's CPU
and Metal).

## The matrix

25 rows of 128 (4 groups each). Rows 0-15 and 23 are seeded Gaussians at
scales from 1e-3 to 30. The others are built to hit each rule's edge:

    16  exact rounding ties: w / d = -(t + 0.5) for t = -8 .. 7
    17  a magnitude tie of opposite signs (+0.5 at 3, -0.5 at 10): the FIRST wins
    18  an all-zero group (d = 0 -> 1e-8 -> a bf16 that flushes to 0 in fp16)
    19  magnitudes ~1e-5 (d below fp16's normal range)
    20  magnitudes ~1e3
    21  d / -8 whose bf16 rounding is an exact tie (both parities)
    22  +1 then -1: the second divides to +8 and clips to 15
    24  exact ties at d = 0.953125, whose fp32 reciprocal is off by enough
        that `w * (1 / d)` rounds 8 of the 16 ties the other way: the grid
        divides

Each edge is load-bearing: a first-max -> last-max change, round half away
from zero, a truncating bf16 cast or a reciprocal multiply each changes
codes or scales of this file (checked when it was made).
"""
import argparse
import json
import struct
from pathlib import Path

import numpy as np

GS = 32


def bf16_round(x):
    """fp32 -> nearest bf16 (ties to even) -> fp32. No NaNs reach this."""
    b = np.asarray(x, np.float32).view(np.uint32).astype(np.uint64)
    b = (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000
    return b.astype(np.uint32).view(np.float32)


def rtn_q4_0(w):
    """[N, K] fp32 -> (q [N, K] uint8 in 0..15, d [N, K / 32] fp32)."""
    n, k = w.shape
    g = w.reshape(n, k // GS, GS).astype(np.float32)
    idx = np.argmax(np.abs(g), axis=2)  # numpy: the FIRST maximum
    d = np.take_along_axis(g, idx[..., None], axis=2)[..., 0] / np.float32(-8.0)
    d = np.where(d == 0, np.float32(1e-8), d).astype(np.float32)
    d = bf16_round(d)
    q = np.clip(np.round(g / d[..., None]) + np.float32(8.0), 0.0, 15.0)
    return q.reshape(n, k).astype(np.uint8), d.astype(np.float32)


def matrix():
    rng = np.random.default_rng(20261012)
    n, k = 25, 128
    w = np.zeros((n, k), np.float32)
    scales = np.geomspace(1e-3, 30.0, 16)
    for r in range(16):
        w[r] = rng.standard_normal(k).astype(np.float32) * np.float32(scales[r])
    # 16: exact ties. The group's max is +1 (d = -0.125, bf16-exact); the
    # other 31 elements are (t + 0.5) * 0.125, so w / d = -(t + 0.5).
    for g in range(k // GS):
        grp = np.zeros(GS, np.float32)
        grp[0] = 1.0
        for j in range(1, GS):
            t = ((j + g) % 16) - 8
            grp[j] = np.float32((t + 0.5) * 0.125)
        w[16, g * GS:(g + 1) * GS] = grp
    # 17: a magnitude tie of opposite signs, then small values.
    w[17] = rng.standard_normal(k).astype(np.float32) * np.float32(0.01)
    for g in range(k // GS):
        w[17, g * GS + 3] = 0.5
        w[17, g * GS + 10] = -0.5
    # 18: one all-zero group among random ones.
    w[18] = rng.standard_normal(k).astype(np.float32) * np.float32(0.05)
    w[18, GS:2 * GS] = 0.0
    # 19 / 20: below fp16's normal range for d, and large.
    w[19] = rng.standard_normal(k).astype(np.float32) * np.float32(1e-5)
    w[20] = rng.standard_normal(k).astype(np.float32) * np.float32(1e3)
    # 21: a group max m whose m / -8 sits exactly halfway between two bf16s
    # (low 16 bits 0x8000), with an even and an odd upper half.
    w[21] = rng.standard_normal(k).astype(np.float32) * np.float32(0.01)
    for g, upper in enumerate((0x3E80, 0x3E81, 0xBE80, 0xBE81)):
        d_exact = np.array([(upper << 16) | 0x8000], np.uint32).view(np.float32)[0]
        w[21, g * GS] = np.float32(d_exact * np.float32(-8.0))
    # 22: +1 first, -1 later: -1 / (1 / -8) = +8 -> 16 -> clipped to 15.
    w[22] = rng.standard_normal(k).astype(np.float32) * np.float32(0.05)
    for g in range(k // GS):
        w[22, g * GS + 2] = 1.0
        w[22, g * GS + 20] = -1.0
    w[23] = rng.standard_normal(k).astype(np.float32) * np.float32(0.3)
    # 24: max -7.625 (d = 0.953125, bf16 0x3F74), the rest (t + 0.5) * d:
    # w / d = t + 0.5 exactly, while w * fl(1 / d) lands off the tie.
    for g in range(k // GS):
        grp = np.zeros(GS, np.float32)
        grp[0] = -7.625
        for j in range(1, GS):
            t = ((3 * j + g) % 16) - 8
            grp[j] = np.float32((t + 0.5) * 0.953125)
        w[24, g * GS:(g + 1) * GS] = grp
    return w


def save_safetensors(path, tensors):
    """A minimal safetensors writer: {name: np.ndarray} in insertion order."""
    header = {}
    blobs = []
    off = 0
    for name, a in tensors.items():
        a = np.ascontiguousarray(a)
        dt = {np.dtype(np.float32): "F32", np.dtype(np.uint8): "U8"}[a.dtype]
        b = a.tobytes()
        header[name] = {"dtype": dt, "shape": list(a.shape), "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * ((8 - len(h) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for b in blobs:
            f.write(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    w = matrix()
    q, d = rtn_q4_0(w)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    save_safetensors(out / "q4_0_rtn.safetensors", {"w": w, "q": q, "d": d})
    ties = int(np.sum(np.abs(w / np.repeat(d, GS, axis=1) - np.round(w / np.repeat(d, GS, axis=1))) == 0.5))
    print(f"wrote {out / 'q4_0_rtn.safetensors'}: w {w.shape}, q {q.shape}, d {d.shape}; "
          f"{ties} exact rounding ties, codes {np.bincount(q.ravel(), minlength=16).tolist()}")


if __name__ == "__main__":
    main()
