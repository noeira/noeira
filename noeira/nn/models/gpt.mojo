"""GPT compositions for nn.storage — plain GPT, the nanoGPT dropout variants,
the weight-tied GPT, and the GPT construction ops (init-time weight surgery +
the one-time weight-tie wiring).

Storage-surface port of `nn/models/gpt.mojo`. The forward-graph aliases (`GPT`,
`GPTDrop`, `GPTDropTied`) are pure `comptime` compositions of storage leaves +
combinators (only change vs legacy: `GELU` imported from
`primitives/activations.mojo`). The two construction ops walk the child tree by
position and are adapted to the storage surface:

  - **c_proj scaled init** (`gpt_scale_residual_proj`): divide each residual
    *output* projection (attention-out + FFN-out) weight by 1/√(2L) after the
    generic INIT (GPT-2 / nanoGPT). Legacy was GPU-only via `mptr(... .dev
    .unsafe_ptr())`; here it takes a `target` and runs on CPU (host loop over
    `.val.data`) OR GPU (one scale kernel over `.val.dev`, the DeviceBuffer
    marshalled straight into the kernel — no `mptr`).

  - **weight-tie wiring** (`gpt_wire_tie`): point the `TiedLinear` head at the
    embedding's value + grad cells (nanoGPT's `lm_head.weight = wte.weight`).
    Uses `TiedLinear.tie_to_ptr` with wildcard-`Pointer` VALUES built from the
    embedding cells — these hold no tracked borrow of `net`, so building them
    into locals releases the structural borrow before the mutable head wiring
    (a `ref`-arg `tie_to` would trip exclusivity since owner + head are both
    children of one `net`). Call ONCE after the model reaches its final home
    (and after any load). After wiring, the standard `Trainer.train_*` loop
    trains the tied model with NO per-step tying code: the shared weight gets
    one gradient (embedding lookup-vjp + head matmul-vjp both `+=` into it) and
    one optimizer update (reflection sees it only via the embedding; the head
    owns no `Param`). See `primitives/tied_linear.mojo`.

GPTDropTied child tree (for the surgery walks):
    [0] Tokenwise[Embedding]      ← embedding W (the shared weight)
    [1] BiasAdd
    [2] Dropout
    [3] Repeat[TransformerBlockDrop]
          .children[L] = block L (Sequential):
            [0] Residual(LN + MHADrop) ; .inner.children[1] = MHADrop =
                  Seq[Tok[Lin d,3d], QKVToMajor, Attn, Tok[Lin d,d]@[3], Dropout]
            [1] Residual(LN + FFNDrop) ; .inner.children[1] = FFNDrop =
                  Seq[Tok[Lin d,ff], GELU, Tok[Lin ff,d]@[2], Dropout]
    [4] Tokenwise[LayerNorm]
    [5] Tokenwise[TiedLinear]     ← LM head (borrows [0]) ; = N-1
"""

from std.math import sqrt
from std.memory import Pointer
from max.gpu import global_idx
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT, TPB
from ..core.tensor import Tensor
from ..primitives.linear import Linear
from ..primitives.tied_linear import TiedLinear
from ..primitives.layer_norm import LayerNorm
from ..primitives.activations import GELUTanh
from ..primitives.embedding import Embedding
from ..primitives.bias_add import BiasAdd
from ..primitives.attention import ScaledDotProductAttention
from ..primitives.dropout import Dropout
from ..primitives.qkv_to_major import QKVToMajor
from ..combinators.sequential import Sequential
from ..combinators.residual import Residual
from ..combinators.repeat import Repeat
from ..combinators.tokenwise import Tokenwise
from .transformer import TransformerBlock


# ──────────────────────────────────────────────────────────────────────
# GPT (plain, no dropout)
# ──────────────────────────────────────────────────────────────────────


# GPT: token Embedding → learnable position BiasAdd → N×TransformerBlock
#      (causal) → final LayerNorm → LM head. Input: seq_len one-hots of width
#      vocab; output: seq_len * vocab logits.
comptime GPT[
    vocab: Int,
    seq_len: Int,
    embed_dim: Int,
    n_heads: Int,
    n_layers: Int,
    ff_mult: Int = 4,
    causal: Bool = True,
    use_max: Bool = True,
    ADT: DType = DT,
] = Sequential[
    Tokenwise[seq_len, Embedding[vocab, embed_dim, ADT]],
    BiasAdd[seq_len * embed_dim, ADT],
    Repeat[
        n_layers,
        TransformerBlock[
            embed_dim, n_heads, seq_len, ff_mult * embed_dim, causal, use_max,
            ADT,
        ],
    ],
    Tokenwise[seq_len, LayerNorm[embed_dim, ADT]],
    Tokenwise[seq_len, Linear[embed_dim, vocab, ADT]],
]


# ──────────────────────────────────────────────────────────────────────
# Transformer + Dropout variants (nanoGPT-style)
# ──────────────────────────────────────────────────────────────────────
#
# Three nanoGPT dropout points: (1) after token+position embedding, (2) after
# MHA's output projection, (3) after the FFN's output projection. Within a
# block MHA-dropout uses seed_base+1, FFN-dropout seed_base+2, input dropout
# seed_base+0 (Repeat reuses one block type → seeds correlated across depth,
# same as legacy; the per-forward counter still refreshes them each step).


comptime MultiHeadAttentionDrop[
    dim: Int, n_heads: Int, seq_len: Int, causal: Bool,
    dropout_p: Float64, seed: UInt64, use_max: Bool = True, ADT: DType = DT,
] = Sequential[
    Tokenwise[seq_len, Linear[dim, 3 * dim, ADT]],
    QKVToMajor[seq_len, dim, ADT],
    ScaledDotProductAttention[dim, n_heads, seq_len, causal, use_max, ADT],
    Tokenwise[seq_len, Linear[dim, dim, ADT]],
    Dropout[seq_len * dim, dropout_p, seed, ADT],
]


comptime TransformerFFNDrop[
    seq_len: Int, dim: Int, ff_dim: Int, dropout_p: Float64, seed: UInt64,
    ADT: DType = DT,
] = Sequential[
    Tokenwise[seq_len, Linear[dim, ff_dim, ADT]],
    GELUTanh[seq_len * ff_dim, ADT],
    Tokenwise[seq_len, Linear[ff_dim, dim, ADT]],
    Dropout[seq_len * dim, dropout_p, seed, ADT],
]


comptime TransformerBlockDrop[
    dim: Int, n_heads: Int, seq_len: Int, ff_dim: Int, causal: Bool,
    dropout_p: Float64, seed_base: UInt64, use_max: Bool = True,
    ADT: DType = DT,
] = Sequential[
    Residual[
        Sequential[
            Tokenwise[seq_len, LayerNorm[dim, ADT]],
            MultiHeadAttentionDrop[
                dim, n_heads, seq_len, causal, dropout_p,
                seed_base + UInt64(1), use_max, ADT,
            ],
        ]
    ],
    Residual[
        Sequential[
            Tokenwise[seq_len, LayerNorm[dim, ADT]],
            TransformerFFNDrop[
                seq_len, dim, ff_dim, dropout_p, seed_base + UInt64(2), ADT
            ],
        ]
    ],
]


# GPTDrop: GPT with the three dropout points (input, MHA-out, FFN-out).
comptime GPTDrop[
    vocab: Int,
    seq_len: Int,
    embed_dim: Int,
    n_heads: Int,
    n_layers: Int,
    ff_mult: Int = 4,
    causal: Bool = True,
    dropout_p: Float64 = 0.2,
    seed_base: UInt64 = UInt64(0xC0FFEE),
    use_max: Bool = True,
    ADT: DType = DT,
] = Sequential[
    Tokenwise[seq_len, Embedding[vocab, embed_dim, ADT]],
    BiasAdd[seq_len * embed_dim, ADT],
    Dropout[seq_len * embed_dim, dropout_p, seed_base, ADT],
    Repeat[
        n_layers,
        TransformerBlockDrop[
            embed_dim, n_heads, seq_len, ff_mult * embed_dim, causal,
            dropout_p, seed_base, use_max, ADT,
        ],
    ],
    Tokenwise[seq_len, LayerNorm[embed_dim, ADT]],
    Tokenwise[seq_len, Linear[embed_dim, vocab, ADT]],
]


# GPTDropTied: GPTDrop with the LM head WEIGHT-TIED to the token embedding
# (nanoGPT's `lm_head.weight = wte.weight`). The final projection is a bias-less
# `TiedLinear` that borrows the embedding's `[vocab, embed]` table (used
# transposed) instead of owning a separate `Linear[embed, vocab]`. After `make`,
# call `gpt_wire_tie` once; then the standard `Trainer.train_*` loop trains it
# with no per-step tying code.
comptime GPTDropTied[
    vocab: Int,
    seq_len: Int,
    embed_dim: Int,
    n_heads: Int,
    n_layers: Int,
    ff_mult: Int = 4,
    causal: Bool = True,
    dropout_p: Float64 = 0.2,
    seed_base: UInt64 = UInt64(0xC0FFEE),
    use_max: Bool = True,
    ADT: DType = DT,
] = Sequential[
    Tokenwise[seq_len, Embedding[vocab, embed_dim, ADT]],
    BiasAdd[seq_len * embed_dim, ADT],
    Dropout[seq_len * embed_dim, dropout_p, seed_base, ADT],
    Repeat[
        n_layers,
        TransformerBlockDrop[
            embed_dim, n_heads, seq_len, ff_mult * embed_dim, causal,
            dropout_p, seed_base, use_max, ADT,
        ],
    ],
    Tokenwise[seq_len, LayerNorm[embed_dim, ADT]],
    Tokenwise[seq_len, TiedLinear[embed_dim, vocab, ADT]],
]


# ──────────────────────────────────────────────────────────────────────
# GPT construction ops — concrete `GPTDropTied[...]`, full param list explicit.
# (Mojo can't reverse-infer a parametric alias's args from the expanded
# Sequential type, so callers pass GPTDropTied's full param list.)
# ──────────────────────────────────────────────────────────────────────


def _scale_kernel(
    buf: Pointer[Scalar[DT], MutAnyOrigin],
    # ⚠ Int32, NOT Int — `Int`/`UInt` are not `DevicePassable`. A bare
    # `Int` still compiles in `pixi run build` and fails only where the
    # kernel is LAUNCHED; keep the `Int32(...)` casts at the call sites.
    n: Int32,
    s: Scalar[DT],
):
    """`buf[i] *= s`. One thread per element."""
    var i = Int(global_idx.x)
    if i < Int(n):
        buf[unsafe_offset=i] = buf[unsafe_offset=i] * s


def _gpt_scale_weight[target: StaticString, N: Int](
    mut w: Tensor, s: Scalar[DT], ctx: Optional[DeviceContext]
) raises:
    """Scale one `[N]` weight slab in place by `s` (CPU loop or GPU kernel).

    Takes the weight by `mut` so the caller can pass an interior reference
    inline — the reference lives only for the duration of this call, which is
    what keeps two such slabs from being borrowed simultaneously.

    Args:
        w: Weight storage cell to scale in place.
        s: Scale factor.
        ctx: Device context; required when `target` is not "cpu".
    """
    comptime if target == "cpu":
        for i in range(N):
            w.data[i] = w.data[i] * s
    else:
        var c = ctx.value()
        c.enqueue_function[_scale_kernel](
            w.dev.value(), Int32(N), s,
            grid_dim=(N + TPB - 1) // TPB, block_dim=TPB,
        )


def gpt_scale_residual_proj[
    target: StaticString,
    vocab: Int,
    seq_len: Int,
    embed_dim: Int,
    n_heads: Int,
    n_layers: Int,
    ff_mult: Int,
    causal: Bool,
    dropout_p: Float64,
    seed_base: UInt64,
    use_max: Bool,
    ADT: DType = DT,
](
    mut net: GPTDropTied[
        vocab, seq_len, embed_dim, n_heads, n_layers,
        ff_mult, causal, dropout_p, seed_base, use_max, ADT,
    ],
    ctx: Optional[DeviceContext] = None,
) raises:
    """NanoGPT/GPT-2 scaled init: divide each residual output projection
    (attention-out + FFN-out) weight by 1/√(2L). Call once after `make`.
    CPU (host loop) or GPU (scale kernel) per `target`."""
    var s = Scalar[DT](1.0 / sqrt(Float64(2 * n_layers)))
    comptime DD = embed_dim * embed_dim  # attn-out Linear[D, D]
    comptime FD = (ff_mult * embed_dim) * embed_dim  # FFN-out  Linear[F, D]
    # Each weight is scaled through its own call so that only ONE interior
    # reference into `net` is live at a time: nightly invalidates an interior
    # `ref` as soon as a second one is formed from the same root, which the
    # previous "bind a_w and f_w, then use both" shape tripped over.
    for L in range(n_layers):
        # attn-out c_proj: block.children[0] (Residual) .inner.children[1]
        # (MHADrop) .children[3] (Tok[Lin d,d]) .inner.weight
        _gpt_scale_weight[target, DD](
            net.children[3].children[L].children[0]
            .inner.children[1].children[3].inner.weight.val,
            s,
            ctx,
        )
        # FFN-out c_proj: block.children[1] (Residual) .inner.children[1]
        # (FFNDrop) .children[2] (Tok[Lin ff,d]) .inner.weight
        _gpt_scale_weight[target, FD](
            net.children[3].children[L].children[1]
            .inner.children[1].children[2].inner.weight.val,
            s,
            ctx,
        )


def gpt_wire_tie[
    target: StaticString,
    vocab: Int,
    seq_len: Int,
    embed_dim: Int,
    n_heads: Int,
    n_layers: Int,
    ff_mult: Int,
    causal: Bool,
    dropout_p: Float64,
    seed_base: UInt64,
    use_max: Bool,
    ADT: DType = DT,
](
    mut net: GPTDropTied[
        vocab, seq_len, embed_dim, n_heads, n_layers,
        ff_mult, causal, dropout_p, seed_base, use_max, ADT,
    ],
) raises:
    """Point the `TiedLinear` LM head at the embedding's value + grad cells.
    Call ONCE after the model settles in its final home (and after any load).
    Idempotent. `target` is accepted for call-site symmetry with the rest of
    the model API (the tie is target-agnostic — the borrowed `Tensor` cells
    carry both CPU + device storage)."""
    comptime LM_IDX = GPTDropTied[
        vocab, seq_len, embed_dim, n_heads, n_layers,
        ff_mult, causal, dropout_p, seed_base, use_max, ADT,
    ].N - 1
    # Build wildcard Pointer VALUES to the embedding (child 0) val/grad cells;
    # these hold no tracked borrow of `net`, so the structural borrow is
    # released before the mutable head wiring below.
    var val_p = rebind[Pointer[Tensor, MutAnyOrigin]](
        Pointer(to=net.children[0].inner.weight.val)
    )
    var grd_p = rebind[Pointer[Tensor, MutAnyOrigin]](
        Pointer(to=net.children[0].inner.weight.grd)
    )
    net.children[LM_IDX].inner.tie_to_ptr(val_p, grd_p)
