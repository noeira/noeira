# +--------------------------------------------------------------------------+ #
# | SmolVLA — the SigLIP vision tower
# +--------------------------------------------------------------------------+ #
"""The vision half of SmolVLM2-500M, as shipped inside `lerobot/smolvla_base`.

Shapes are not guessed. They are read from the checkpoint's own manifest
(`tools/vla/smolvla_base_manifest.tsv`, 500 tensors) and from the backbone's
`config.json`:

    hidden 768 · 12 layers · 12 heads (head_dim 64) · ff 3072
    image 512x512 · patch 16 -> 32x32 = 1024 tokens

197 of the checkpoint's 500 tensors live here:
2 patch-embedding + 1 position-embedding + 12x16 per-layer + 2 post-LayerNorm.

## Why the projections are separate

`MultiHeadAttentionXL` projects q, k and v with ONE `Linear[dim, 3*inner]`.
SigLIP ships `q_proj`/`k_proj`/`v_proj` as three tensors, and the text tower
this shares a file with is GQA (q 960-wide, k/v 320-wide) where fusing is not
even arithmetically possible. Fusing here would also break `TorchNameMap`'s
one-of-ours <-> one-of-theirs contract: three file tensors into one `Param` is
not expressible in that table.

So the projections are four separate `Linear`s around `CrossAttention`, which is
already a PARAM-FREE attention core over pre-projected q/k/v. One decision
serves this tower, the GQA text tower and the expert's cross-attention.

⚠ **q, k, v and o are all `[768, 768]`.** They are mutually substitutable by
size, so a positional or careless load swaps two of them and the model stays the
right shape while computing something else. That is exactly the confusion
`deep_agents/act/refload.mojo` was written to make impossible, and the reason
this is a NAMED struct rather than a `Sequential` of anonymous children: the
walked names are `attn.q.…`/`attn.k.…`, not `0.1.…`/`0.2.…`.

⚠ **GELU here is the tanh approximation, and that is correct.** SigLIP is
`hidden_act = "gelu_pytorch_tanh"`, i.e. `nn.primitives.activations.GELUTanh`
(`GELUTanhOp`) — so this matches by construction. Do not
"fix" it to the erf form.
"""

from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.initializer import Initializer
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.module import Module
from noeira.nn.core.param import ParamVisitor
from noeira.nn.core.walkers import join_name
from noeira.nn.core.amp import AMPPolicy, NoAMP
from noeira.nn.combinators.compute_graph import ComputeGraph
from noeira.nn.combinators.graph_decl import InputSlot, Node
from noeira.nn.combinators.tokenwise import Tokenwise
from noeira.nn.primitives.linear import Linear
from noeira.nn.primitives.cross_attention import CrossAttention
from noeira.nn.primitives.layer_norm import LayerNorm
from noeira.nn.primitives.conv2d import Conv2D
from noeira.nn.primitives.bias_add import BiasAdd
from noeira.nn.primitives.transpose_2d import Transpose2D
from noeira.nn.primitives.activations import GELUTanh
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.combinators.residual import Residual
from noeira.nn.combinators.repeat import Repeat


# ── the checkpoint's numbers, in one place ───────────────────────────────
comptime SIGLIP_DIM: Int = 768
comptime SIGLIP_HEADS: Int = 12  # head_dim = 64
comptime SIGLIP_FF: Int = 3072
comptime SIGLIP_LAYERS: Int = 12
comptime SIGLIP_PATCH: Int = 16
comptime SIGLIP_IMG: Int = 512
comptime SIGLIP_GRID: Int = SIGLIP_IMG // SIGLIP_PATCH  # 32
comptime SIGLIP_TOKENS: Int = SIGLIP_GRID * SIGLIP_GRID  # 1024
comptime SIGLIP_EPS: Scalar[DT] = 1e-6
"""⚠ SigLIP's `layer_norm_eps`. `nn`'s `LayerNorm` defaults to 1e-5, which is a
different model — invisible to shapes and NaNs, visible in a parity check.
`SigLIPLN` below pins it."""

comptime SigLIPLN[D: Int] = LayerNorm[D, DT, SIGLIP_EPS]


struct SigLIPAttention[SEQ: Int, DIM: Int, HEADS: Int](Module):
    """Self-attention with four separate projections: `q`, `k`, `v`, `o`.

    A `ComputeGraph` rather than a hand-rolled forward, so the fan-out of `x`
    into three projections gets its gradient ACCUMULATED rather than
    last-write-wins — the graph already does that, and it is the one part of
    this that is easy to get silently wrong.
    """

    comptime ARITY: Int = 1
    comptime SEQ_DIM: Int = Self.SEQ * Self.DIM
    comptime IN_DIMS = Array[Int, 1](fill=Self.SEQ_DIM)
    comptime OUT_DIM: Int = Self.SEQ_DIM

    comptime Proj = Tokenwise[Self.SEQ, Linear[Self.DIM, Self.DIM]]

    comptime Graph = ComputeGraph[
        InputSlot["x", Self.SEQ_DIM],
        Node["q", Self.Proj, "x"],
        Node["k", Self.Proj, "x"],
        Node["v", Self.Proj, "x"],
        Node[
            "a",
            CrossAttention[Self.DIM, Self.HEADS, Self.SEQ, Self.SEQ, False],
            "q", "k", "v",
        ],
        Node["o", Self.Proj, "a"],
    ]

    var graph: Self.Graph

    def __init__(out self):
        comptime assert Self.DIM % Self.HEADS == 0, (
            "SigLIPAttention: DIM must be divisible by HEADS"
        )
        self.graph = Self.Graph()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "SigLIPAttention: target must be 'cpu' or 'gpu'"
        )
        var a = Self()
        a.graph = Self.Graph.make[target, INIT](ctx)
        return a^

    def set_attr[ATTR: StaticString](mut self, value: Scalar[DT]):
        """Forward runtime attrs into the graph — `fused_attention` has to
        reach the `CrossAttention` leaf at node "a" from the tower's root.
        ⚠ Without this the `Module` default (`pass`) swallows it here and the
        deploy's switch silently does nothing, the way Dropout's `training`
        once did (`_a_trait_default_of_pass_makes_a_missing_override_silent`);
        `test_vision_fused_toggle.mojo` proves it reaches."""
        self.graph.set_attr[ATTR](value)

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[Self.ARITY, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        self.graph.set_input["x", B](inputs[0], ctx)
        self.graph.forward[B, target, POLICY=POLICY](out, ctx)

    def vjp[
        target: StaticString, B: Int, ofi: MutOrigin, ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[Self.ARITY, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[Self.ARITY, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref gx = grad_inputs[0]
        comptime DN = B * Self.SEQ_DIM
        self.graph.vjp[B, target, POLICY=POLICY](grad_output, ctx)
        comptime if target == "cpu":
            gx.ensure(DN)
            for i in range(DN):
                gx.data[i] = self.graph.grad_input["x"]().data[i]
        else:
            var cc = ctx.value()
            gx.ensure_gpu(cc, DN)
            # Size-exact sub-buffer copy: the caller's grad slot is reused
            # across nodes and may be larger than DN, and a whole-buffer copy
            # errors on the mismatch. Same fix as `DecoderBlock.vjp`.
            var src = self.graph.grad_input["x"]().dev.value(
            ).create_sub_buffer[DT](0, DN)
            var dst = gx.dev.value().create_sub_buffer[DT](0, DN)
            cc.enqueue_copy(dst, src)

    def for_each_param[
        target: StaticString, V: ParamVisitor
    ](mut self, mut visitor: V, ctx: Optional[DeviceContext],
      prefix: String = String("")) raises:
        self.graph.for_each_param[target](visitor, ctx, prefix)

    def for_each_state[
        target: StaticString, V: ParamVisitor
    ](mut self, mut visitor: V, ctx: Optional[DeviceContext],
      prefix: String = String("")) raises:
        self.graph.for_each_state[target](visitor, ctx, prefix)

    def zero_grad[
        target: StaticString
    ](mut self, ctx: Optional[DeviceContext]) raises:
        self.graph.zero_grad[target](ctx)

    def polyak_from[
        target: StaticString
    ](
        mut self, mut src: Self, tau: Scalar[DT], ctx: Optional[DeviceContext]
    ) raises:
        self.graph.polyak_from[target](src.graph, tau, ctx)


# ── the layer, and the stack of twelve ───────────────────────────────────
# Pre-LN, exactly `TransformerBlock`'s topology, with the fused-QKV attention
# swapped for the four-projection one above. The FFN *is* `TransformerFFN`
# (Linear -> GELU -> Linear, per-token), so it is reused rather than restated.
comptime SigLIPFFN[SEQ: Int, DIM: Int, FF: Int] = Sequential[
    Tokenwise[SEQ, Linear[DIM, FF]],
    GELUTanh[SEQ * FF],
    Tokenwise[SEQ, Linear[FF, DIM]],
]

comptime SigLIPLayer[SEQ: Int, DIM: Int, HEADS: Int, FF: Int] = Sequential[
    Residual[
        Sequential[
            Tokenwise[SEQ, SigLIPLN[DIM]],
            SigLIPAttention[SEQ, DIM, HEADS],
        ]
    ],
    Residual[
        Sequential[
            Tokenwise[SEQ, SigLIPLN[DIM]],
            SigLIPFFN[SEQ, DIM, FF],
        ]
    ],
]


# ── patch + position embedding ───────────────────────────────────────────
# `Conv2D` with K == S == 16 and no padding IS the patch embedding: each 16x16
# stride-16 window is one patch, and the OC axis is the embedding. It emits
# NCHW, i.e. CHANNEL-major [768, 32, 32]; the encoder wants TOKEN-major
# [1024, 768], hence the transpose.
#
# The position table is [1024, 768] — one vector per patch slot, added to every
# sample. Flattened token-major that is exactly a `BiasAdd` over the whole
# 1024*768 activation, and its `bias` Param maps to `position_embedding.weight`
# with NO transpose: both sides are row-major position-major.
comptime SigLIPEmbeddings[
    IMG: Int, PATCH: Int, DIM: Int, GRID: Int, TOKENS: Int
] = Sequential[
    Conv2D[3, DIM, PATCH, PATCH, 0, IMG, IMG],
    Transpose2D[DIM, TOKENS],
    BiasAdd[TOKENS * DIM],
]


# ── the tower ────────────────────────────────────────────────────────────
comptime SigLIPVisionTower[
    IMG: Int = SIGLIP_IMG,
    PATCH: Int = SIGLIP_PATCH,
    DIM: Int = SIGLIP_DIM,
    HEADS: Int = SIGLIP_HEADS,
    FF: Int = SIGLIP_FF,
    LAYERS: Int = SIGLIP_LAYERS,
    GRID: Int = SIGLIP_GRID,
    TOKENS: Int = SIGLIP_TOKENS,
] = Sequential[
    SigLIPEmbeddings[IMG, PATCH, DIM, GRID, TOKENS],
    Repeat[LAYERS, SigLIPLayer[TOKENS, DIM, HEADS, FF]],
    Tokenwise[TOKENS, SigLIPLN[DIM]],
]
