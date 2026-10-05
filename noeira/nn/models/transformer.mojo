"""Shared transformer building blocks (pre-LN GPT/ViT-style) for nn.storage.

Storage-surface port of `nn/models/transformer.mojo`. Pure `comptime`
composition over storage leaves + combinators — no structs, no kernels — so it
inherits correct forward / vjp / walkers from its parts. The ONLY change vs the
legacy file is the import path for `ReLU`/`GELUTanh` (storage keeps the activation
aliases in `primitives/activations.mojo`, not standalone `relu.mojo`/`gelu.mojo`).

All per-token sublayers (QKV/out projection, FFN, LayerNorm) are wrapped in
`Tokenwise[seq_len, ...]` so weights are shared across positions; attention
itself spans the full sequence. `causal=False` → bidirectional (ViT);
`causal=True` → decoder (GPT). Input/output per sample: seq_len * dim.

  - `MultiHeadAttentionXL` / `MultiHeadAttention`
  - `MultiHeadAttentionFused` (no `QKVToMajor`: the attention reads the QKV
    projection's own layout)
  - `TransformerFFN`
  - `TransformerBlock` / `TransformerBlockFused`
"""

from noeira.nn.constants import DT
from ..primitives.linear import Linear
from ..primitives.layer_norm import LayerNorm
from ..primitives.activations import GELUTanh
from ..primitives.attention import (
    ScaledDotProductAttention, ScaledDotProductAttentionQKV,
)
from ..primitives.qkv_to_major import QKVToMajor
from ..primitives.feed_forward_gelu import FeedForwardGELU
from ..combinators.sequential import Sequential
from ..combinators.residual import Residual
from ..combinators.tokenwise import Tokenwise


# MultiHeadAttentionXL: per-token QKV proj → QKVToMajor (token-major → qkv-major,
# so the causal mask hits the right tokens) → attention → per-token out proj.
# The QKV/out projections + SDPA run at the inner width n_heads*head_dim (can
# exceed `dim`), then project back to `dim`. `MultiHeadAttention` is the special
# case head_dim = dim/n_heads (inner == dim) — a strict generalization.
comptime MultiHeadAttentionXL[
    dim: Int, n_heads: Int, head_dim: Int, seq_len: Int,
    causal: Bool = False, use_max: Bool = True, ADT: DType = DT,
] = Sequential[
    Tokenwise[seq_len, Linear[dim, 3 * n_heads * head_dim, ADT]],
    QKVToMajor[seq_len, n_heads * head_dim, ADT],
    ScaledDotProductAttention[
        n_heads * head_dim, n_heads, seq_len, causal, use_max, ADT
    ],
    Tokenwise[seq_len, Linear[n_heads * head_dim, dim, ADT]],
]


comptime MultiHeadAttention[
    dim: Int, n_heads: Int, seq_len: Int, causal: Bool = False,
    use_max: Bool = True, ADT: DType = DT,
] = MultiHeadAttentionXL[
    dim, n_heads, dim // n_heads, seq_len, causal, use_max, ADT
]


# MultiHeadAttentionFused: `MultiHeadAttention` without the `QKVToMajor`
# child — `ScaledDotProductAttentionQKV` takes the QKV projection's token-major
# output, and its fused GPU path reads that layout directly (one permutation
# pass and its backward fewer per block). Same parameters as
# `MultiHeadAttention`, but the out projection is child [2], not [3], so a
# checkpoint of one does not load into the other.
comptime MultiHeadAttentionFused[
    dim: Int, n_heads: Int, seq_len: Int, causal: Bool = False,
    use_max: Bool = True, ADT: DType = DT,
] = Sequential[
    Tokenwise[seq_len, Linear[dim, 3 * dim, ADT]],
    ScaledDotProductAttentionQKV[dim, n_heads, seq_len, causal, use_max, ADT],
    Tokenwise[seq_len, Linear[dim, dim, ADT]],
]


# TransformerFFN: per-token Linear → GELU → per-token Linear.
# GELU is pointwise, so applying it to the flat (BATCH, seq_len*ff_dim) tensor
# is identical to per-token — no Tokenwise wrapper needed.
comptime TransformerFFN[
    seq_len: Int, dim: Int, ff_dim: Int, ADT: DType = DT,
] = Sequential[
    Tokenwise[seq_len, Linear[dim, ff_dim, ADT]],
    GELUTanh[seq_len * ff_dim, ADT],
    Tokenwise[seq_len, Linear[ff_dim, dim, ADT]],
]


# TransformerBlock: pre-LN encoder/decoder layer.
#   y = x + MHA(LN(x));  z = y + FFN(LN(y)).  IN_DIM == OUT_DIM == seq_len*dim.
comptime TransformerBlock[
    dim: Int, n_heads: Int, seq_len: Int, ff_dim: Int, causal: Bool = False,
    use_max: Bool = True, ADT: DType = DT,
] = Sequential[
    Residual[
        Sequential[
            Tokenwise[seq_len, LayerNorm[dim, ADT]],
            MultiHeadAttention[dim, n_heads, seq_len, causal, use_max, ADT],
        ]
    ],
    Residual[
        Sequential[
            Tokenwise[seq_len, LayerNorm[dim, ADT]],
            TransformerFFN[seq_len, dim, ff_dim, ADT],
        ]
    ],
]


# TransformerBlockFused: `TransformerBlock` on `MultiHeadAttentionFused` and
# `FeedForwardGELU` (the FFN as one module: cuBLASLt GELU epilogues on NVIDIA).
# Same parameters as `TransformerBlock`; different child paths.
comptime TransformerBlockFused[
    dim: Int, n_heads: Int, seq_len: Int, ff_dim: Int, causal: Bool = False,
    use_max: Bool = True, ADT: DType = DT,
] = Sequential[
    Residual[
        Sequential[
            Tokenwise[seq_len, LayerNorm[dim, ADT]],
            MultiHeadAttentionFused[dim, n_heads, seq_len, causal, use_max, ADT],
        ]
    ],
    Residual[
        Sequential[
            Tokenwise[seq_len, LayerNorm[dim, ADT]],
            FeedForwardGELU[seq_len, dim, ff_dim, ADT],
        ]
    ],
]
