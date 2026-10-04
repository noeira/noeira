"""Counter-hash dropout masks: keep(seed, ctr, idx) with no state and no cache.

A mask element is a pure function of (a per-instance seed, a per-forward
counter, the element's flat index): the forward draws it, the backward
REDRAWS it — nothing is stored between them — and the CPU, Metal and CUDA
paths draw the same bits, so a GPU-vs-CPU gate can compare dropout runs
exactly. Used by `HashDropout` and `ScaledDotProductAttention`'s attention
dropout.

The hash is splitmix64's finaliser over the three words mixed with odd
constants; the top 24 bits make a float32 uniform in [0, 1), kept when
>= p (inverted dropout scales the kept elements by 1 / (1 - p)).
"""

from std.random import random_ui64


@always_inline
def hash_keep(seed: UInt64, ctr: UInt64, idx: UInt64, p: Float32) -> Bool:
    var x = seed ^ (ctr * UInt64(0x9E3779B97F4A7C15)) ^ (idx * UInt64(0xD1B54A32D192ED03))
    x ^= x >> 30
    x *= UInt64(0xBF58476D1CE4E5B9)
    x ^= x >> 27
    x *= UInt64(0x94D049BB133111EB)
    x ^= x >> 31
    var u = Float32(x >> 40) * Float32(1.0 / 16777216.0)
    return u >= p


def new_dropout_seed() -> UInt64:
    """A per-instance seed from the process RNG (`std.random`): distinct for
    every dropout site, reproducible under `std.random.seed`."""
    return random_ui64(0, UInt64.MAX)
