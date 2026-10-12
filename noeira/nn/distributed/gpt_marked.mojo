"""GPTMarked — `GPTDropTied` with a gradient mark after every transformer block.

The same model, layer for layer and parameter for parameter, with each block
wrapped in `GradReady[..., ACTIVE]` so an overlapped DDP step knows where in
the backward each block's gradients are final. With `ACTIVE = False` every
block's vjp compiles to the plain block's.

The two construction ops of `models/gpt.mojo` take the concrete
`GPTDropTied`, whose field paths differ from this type by one `.inner` per
block, so they are restated here for this type. ⚠ A rule written twice can
drift: `tests/nn/distributed/test_gpt_marked.mojo` gates this file against
`gpt.mojo`, bit for bit on the initial weights and on a few training steps.

Not covered by a mark (final only at the end of the backward, so reduced in
the last bucket): the token embedding (the tied head also writes its
gradient, at the START of the backward), the positional bias and the final
LayerNorm. Together they are ~1 % of the GPT's parameters.
"""

from std.math import sqrt
from std.memory import Pointer
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.primitives.tied_linear import TiedLinear
from noeira.nn.primitives.layer_norm import LayerNorm
from noeira.nn.primitives.embedding import Embedding
from noeira.nn.primitives.bias_add import BiasAdd
from noeira.nn.primitives.dropout import Dropout
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.combinators.repeat import Repeat
from noeira.nn.combinators.tokenwise import Tokenwise
from noeira.nn.models.gpt import TransformerBlockDrop, _gpt_scale_weight

from .grad_marks import GradReady


comptime GPTMarked[
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
    ACTIVE: Bool = True,
] = Sequential[
    Tokenwise[seq_len, Embedding[vocab, embed_dim]],
    BiasAdd[seq_len * embed_dim],
    Dropout[seq_len * embed_dim, dropout_p, seed_base],
    Repeat[
        n_layers,
        GradReady[
            TransformerBlockDrop[
                embed_dim, n_heads, seq_len, ff_mult * embed_dim, causal,
                dropout_p, seed_base, use_max,
            ],
            ACTIVE,
        ],
    ],
    Tokenwise[seq_len, LayerNorm[embed_dim]],
    Tokenwise[seq_len, TiedLinear[embed_dim, vocab]],
]


def gpt_marked_scale_residual_proj[
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
    ACTIVE: Bool,
](
    mut net: GPTMarked[
        vocab, seq_len, embed_dim, n_heads, n_layers,
        ff_mult, causal, dropout_p, seed_base, use_max, ACTIVE,
    ],
    ctx: Optional[DeviceContext] = None,
) raises:
    """`gpt_scale_residual_proj` for this type: each residual output
    projection weight (attention-out, FFN-out) divided by sqrt(2L)."""
    var s = Scalar[DT](1.0 / sqrt(Float64(2 * n_layers)))
    comptime DD = embed_dim * embed_dim
    comptime FD = (ff_mult * embed_dim) * embed_dim
    for L in range(n_layers):
        # gpt.mojo's paths with `.inner` after the block (the GradReady).
        _gpt_scale_weight[target, DD](
            net.children[3].children[L].inner.children[0]
            .inner.children[1].children[2].inner.weight.val,
            s,
            ctx,
        )
        _gpt_scale_weight[target, FD](
            net.children[3].children[L].inner.children[1]
            .inner.children[1].children[0].fc2.weight.val,
            s,
            ctx,
        )


def gpt_marked_wire_tie[
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
    ACTIVE: Bool,
](
    mut net: GPTMarked[
        vocab, seq_len, embed_dim, n_heads, n_layers,
        ff_mult, causal, dropout_p, seed_base, use_max, ACTIVE,
    ],
) raises:
    """`gpt_wire_tie` for this type: the LM head reads the embedding's value
    and gradient cells."""
    comptime LM_IDX = GPTMarked[
        vocab, seq_len, embed_dim, n_heads, n_layers,
        ff_mult, causal, dropout_p, seed_base, use_max, ACTIVE,
    ].N - 1
    var val_p = rebind[Pointer[Tensor, MutAnyOrigin]](
        Pointer(to=net.children[0].inner.weight.val)
    )
    var grd_p = rebind[Pointer[Tensor, MutAnyOrigin]](
        Pointer(to=net.children[0].inner.weight.grd)
    )
    net.children[LM_IDX].inner.tie_to_ptr(val_p, grd_p)
