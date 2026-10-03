"""LeWM, reference-exact: the published model's architecture, layer for layer.

docs/LEWM_REOPEN_PLAN.md P2. The reference is `stable_worldmodel` 0.1.1
(`references/stable-worldmodel-0.1.1/stable_worldmodel/wm/lewm/`) with a
HuggingFace `ViTModel` encoder, and the published checkpoint
`quentinll/lewm-pusht` loads into these modules tensor for tensor
(`ref_load.mojo`). Every module here is gated against torch on that
checkpoint (`tests/experimental/lewm/ref/`).

The first port (`encoder.mojo` & co., removed 2026-10-03 once the
reference model had replaced it — P5 / P6) was its own reading of the paper
and differed from the reference in ways that changed every number:

| piece | reference (here) | the first port |
|---|---|---|
| ViT LayerNorm eps | 1e-12 (HF) | 1e-5 |
| GELU (ViT FFN, projectors, predictor FFN) | exact (erf) | tanh |
| action embedder | Linear(A,A) -> Linear(A,4E)+SiLU -> Linear(4E,E) | Linear(A,32)+SiLU -> Linear(32,2E)+SiLU -> Linear |
| predictor block | AdaLN-zero + an affine LN INSIDE attention and FFN | AdaLN-zero only |
| predictor tail | final affine LN | none |
| BN running var | unbiased update (torch) | biased |

⚠ ONE DEVIATION LEFT: the reference's attention `to_qkv` has NO bias; ours is
a biased `Linear`. Loaded from the checkpoint the bias is zero, so forward and
every reference gradient are exact; in training `ref_trainer` zeroes its
gradient, so it stays 0.
"""

from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.initializer import Initializer
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.param import ParamVisitor
from noeira.nn.core.amp import AMPPolicy, NoAMP
from noeira.nn.core.module import Module
from noeira.nn import (
    Sequential,
    Repeat,
    Checkpointed,
    RepeatConditional,
    Residual,
    Linear,
    LinearSwish,
    BatchNorm1D,
    GELU,
    LayerNorm,
    LayerNormNoAffine,
    BiasAdd,
    LearnedTokens,
    Slice,
    Tokenwise,
    ComputeGraph,
    InputSlot,
    Node,
    Scale,
    Add,
    MSEPerSample,
    SIGReg,
    SiLU,
    ZeroLinear,
    Modulate,
    Gate,
)
from noeira.nn.models.vit import PatchEmbed
from noeira.nn.models.transformer import MultiHeadAttention, MultiHeadAttentionXL
from noeira.nn.primitives.attention import ScaledDotProductAttention
from noeira.nn.primitives.qkv_to_major import QKVToMajor
from noeira.nn.primitives.hash_dropout import HashDropout
from noeira.nn.primitives.batch_norm_1d import BN_DEFAULT_MOM, BN_DEFAULT_EPS


comptime HF_VIT_LN_EPS: Scalar[DT] = 1e-12
"""`ViTConfig.layer_norm_eps` default."""

comptime TORCH_LN_EPS: Scalar[DT] = 1e-5
"""`nn.LayerNorm` default: the predictor's inner and final LayerNorms."""


# ── ViT encoder (HF ViTModel, pre-LN) ─────────────────────────────────────

comptime FFNExact[SEQ: Int, DIM: Int, FF: Int] = Sequential[
    Tokenwise[SEQ, Linear[DIM, FF]],
    GELU[SEQ * FF],
    Tokenwise[SEQ, Linear[FF, DIM]],
]
"""Linear -> exact GELU -> Linear, per token (HF `ViTMLP`; the predictor's
`FeedForward` minus its leading LayerNorm)."""

comptime ViTBlockHF[DIM: Int, HEADS: Int, SEQ: Int, FF: Int] = Sequential[
    Residual[
        Sequential[
            Tokenwise[SEQ, LayerNorm[DIM, DT, HF_VIT_LN_EPS]],
            MultiHeadAttention[DIM, HEADS, SEQ, False],
        ]
    ],
    Residual[
        Sequential[
            Tokenwise[SEQ, LayerNorm[DIM, DT, HF_VIT_LN_EPS]],
            FFNExact[SEQ, DIM, FF],
        ]
    ],
]
"""HF `ViTLayer`: x += attn(LN_before(x)); x += mlp(LN_after(x)). q/k/v are
separate biased Linears in HF, fused here as [q|k|v] (the loader concatenates)."""

comptime ProjectorRef[IN: Int, HID: Int, OUT: Int] = Sequential[
    Linear[IN, HID],
    BatchNorm1D[HID, BN_DEFAULT_MOM, BN_DEFAULT_EPS, DT, True],
    GELU[HID],
    Linear[HID, OUT],
]
"""`module.MLP(norm_fn=BatchNorm1d)`: the encoder `projector` and `pred_proj`."""

comptime LeWMEncoderRef[
    IN_CH: Int,
    IMG: Int,
    PATCH: Int,
    HIDDEN: Int,
    HEADS: Int,
    LAYERS: Int,
    EMB: Int,
    PROJ_H: Int,
] = Sequential[
    PatchEmbed[
        IN_CH, IMG, IMG, PATCH, HIDDEN, (IMG // PATCH) * (IMG // PATCH)
    ],
    LearnedTokens[(IMG // PATCH) * (IMG // PATCH), 1, HIDDEN, True, 0.02],
    BiasAdd[((IMG // PATCH) * (IMG // PATCH) + 1) * HIDDEN],
    Repeat[
        LAYERS,
        Checkpointed[
            ViTBlockHF[
                HIDDEN, HEADS, (IMG // PATCH) * (IMG // PATCH) + 1, 4 * HIDDEN
            ]
        ],
    ],
    Tokenwise[
        (IMG // PATCH) * (IMG // PATCH) + 1, LayerNorm[HIDDEN, DT, HF_VIT_LN_EPS]
    ],
    Slice[((IMG // PATCH) * (IMG // PATCH) + 1) * HIDDEN, 0, HIDDEN],
    ProjectorRef[HIDDEN, PROJ_H, EMB],
]
"""Image (CHW, ImageNet-normalised) -> (B, EMB): patch Conv2D -> prepend CLS ->
+ position embedding (no interpolation at the trained size) -> LAYERS x ViT
block -> final LN -> CLS token -> projector.

Each ViT block is `Checkpointed` (names unchanged), OFF unless a trainer sets
`checkpoint`: on, the encoder keeps the 12 block inputs and one block's
internals at a time, and recomputes each block's forward in the vjp. The
blocks are deterministic (no BN, no dropout), so the gradients are the same."""


# ── action embedder ───────────────────────────────────────────────────────

comptime ActionEmbedderRef[T: Int, ACT: Int, EMB: Int, MLP_SCALE: Int = 4] = (
    Sequential[
        Tokenwise[T, Linear[ACT, ACT]],
        Tokenwise[T, LinearSwish[ACT, MLP_SCALE * EMB]],
        Tokenwise[T, Linear[MLP_SCALE * EMB, EMB]],
    ]
)
"""`module.Embedder` (smoothed_dim = input_dim, mlp_scale 4): Conv1d(k=1) =
a per-token Linear, then Linear -> SiLU -> Linear. ONE SiLU."""


# ── predictor block ───────────────────────────────────────────────────────

comptime PRED_DROPOUT = 0.1
"""`config/train/lewm.yaml` `predictor.dropout`: every dropout site of the
predictor's blocks (`emb_dropout` is 0). Off unless the trainer switches
`dropout` on: the torch gates, validation and planning run without it."""

comptime AttnDropRef[EMB: Int, HEADS: Int, HEAD_DIM: Int, H: Int, P: Float64] = Sequential[
    Tokenwise[H, Linear[EMB, 3 * HEADS * HEAD_DIM]],
    QKVToMajor[H, HEADS * HEAD_DIM],
    ScaledDotProductAttention[HEADS * HEAD_DIM, HEADS, H, True, True, DT, P],
    Tokenwise[H, Linear[HEADS * HEAD_DIM, EMB]],
    HashDropout[H * EMB, P],
]
"""`module.Attention`: causal SDPA with `dropout_p` on the weights, then
`to_out` = Linear + Dropout. Same parameter paths as `MultiHeadAttentionXL`
(`.0.0` qkv, `.3.0` out)."""

comptime FFNDropRef[SEQ: Int, DIM: Int, FF: Int, P: Float64] = Sequential[
    Tokenwise[SEQ, Linear[DIM, FF]],
    GELU[SEQ * FF],
    HashDropout[SEQ * FF, P],
    Tokenwise[SEQ, Linear[FF, DIM]],
    HashDropout[SEQ * DIM, P],
]
"""`module.FeedForward` minus its LayerNorm: Linear -> GELU -> Dropout ->
Linear -> Dropout (the second Linear is `.3.0`)."""


struct ConditionalTransformerBlockRef[
    EMB: Int, HEADS: Int, H: Int, FF: Int, HEAD_DIM: Int
](Module):
    """`module.ConditionalBlock`: AdaLN-zero, with the reference's affine
    LayerNorm INSIDE `Attention` and inside `FeedForward`:

        sh1,sc1,g1,sh2,sc2,g2 = Linear(SiLU(c)).chunk(6)     (6 ZeroLinears)
        x = x + g1 * Attn( LN_a( LN1(x)*(1+sc1)+sh1 ) )      LN1: no affine, 1e-6
        x = x + g2 * FFN ( LN_f( LN2(x)*(1+sc2)+sh2 ) )      LN_a/LN_f: affine, 1e-5

    Attention: causal, HEADS x HEAD_DIM inner (16 x 64 = 1024 for the paper).
    FFN: Linear -> exact GELU -> Linear. Dropout (`PRED_DROPOUT`) on the
    attention weights, after `to_out`, after the GELU and after the FFN —
    off unless `set_attr["dropout"](1)`. Same wrapper as
    `nn.primitives.ConditionalTransformerBlock` (a Module around an internal
    ComputeGraph); only the graph differs — Mojo has no conditional type alias,
    so the variant is a sibling struct, not a flag."""

    comptime ARITY: Int = 2
    comptime SEQ_DIM = Self.H * Self.EMB
    comptime IN_DIMS = Array[Int, 2](fill=Self.SEQ_DIM)
    comptime OUT_DIM = Self.SEQ_DIM

    comptime Mod6 = Tokenwise[Self.H, ZeroLinear[Self.EMB, Self.EMB]]
    comptime LN = Tokenwise[Self.H, LayerNormNoAffine[Self.EMB]]
    comptime LNA = Tokenwise[Self.H, LayerNorm[Self.EMB, DT, TORCH_LN_EPS]]

    comptime Graph = ComputeGraph[
        InputSlot["x", Self.SEQ_DIM],
        InputSlot["c", Self.SEQ_DIM],
        Node["cs", SiLU[Self.SEQ_DIM], "c"],
        Node["sh1", Self.Mod6, "cs"],
        Node["sc1", Self.Mod6, "cs"],
        Node["g1", Self.Mod6, "cs"],
        Node["sh2", Self.Mod6, "cs"],
        Node["sc2", Self.Mod6, "cs"],
        Node["g2", Self.Mod6, "cs"],
        Node["ln1", Self.LN, "x"],
        Node["mod1", Modulate[Self.SEQ_DIM], "ln1", "sc1", "sh1"],
        Node["ln_a", Self.LNA, "mod1"],
        Node[
            "attn",
            AttnDropRef[Self.EMB, Self.HEADS, Self.HEAD_DIM, Self.H, PRED_DROPOUT],
            "ln_a",
        ],
        Node["x1", Gate[Self.SEQ_DIM], "x", "g1", "attn"],
        Node["ln2", Self.LN, "x1"],
        Node["mod2", Modulate[Self.SEQ_DIM], "ln2", "sc2", "sh2"],
        Node["ln_f", Self.LNA, "mod2"],
        Node["mlp", FFNDropRef[Self.H, Self.EMB, Self.FF, PRED_DROPOUT], "ln_f"],
        Node["x2", Gate[Self.SEQ_DIM], "x1", "g2", "mlp"],
    ]

    var graph: Self.Graph

    def __init__(out self):
        self.graph = Self.Graph()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "ConditionalTransformerBlockRef: target must be 'cpu' or 'gpu'"
        )
        var b = Self()
        b.graph = Self.Graph.make[target=target, INIT=INIT](ctx)
        b.graph.set_attr["dropout"](Scalar[DT](0))  # see PRED_DROPOUT
        return b^

    def set_attr[ATTR: StaticString](mut self, value: Scalar[DT]):
        """Into the block's graph (`dropout`; nothing else listens here)."""
        self.graph.set_attr[ATTR](value)

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[2, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref x = inputs[0]
        ref c = inputs[1]
        self.graph.set_input["x", B](x, ctx)
        self.graph.set_input["c", B](c, ctx)
        self.graph.forward[B, target, POLICY=POLICY](out, ctx)

    def vjp[
        target: StaticString, B: Int, ofi: MutOrigin, ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[2, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[2, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime total = B * Self.SEQ_DIM
        self.graph.vjp[B, target, POLICY=POLICY](grad_output, ctx)
        ref gx = grad_inputs[0]
        ref gc = grad_inputs[1]
        comptime if target == "cpu":
            gx.ensure(total)
            gc.ensure(total)
            for q in range(total):
                gx.data[q] = self.graph.grad_input["x"]().data[q]
            for q in range(total):
                gc.data[q] = self.graph.grad_input["c"]().data[q]
        else:
            var c = ctx.value()
            gx.ensure_gpu(c, total)
            gc.ensure_gpu(c, total)
            var gx_src = self.graph.grad_input["x"]().dev.value(
            ).create_sub_buffer[DT](0, total)
            var gx_dst = gx.dev.value().create_sub_buffer[DT](0, total)
            c.enqueue_copy(gx_dst, gx_src)
            var gc_src = self.graph.grad_input["c"]().dev.value(
            ).create_sub_buffer[DT](0, total)
            var gc_dst = gc.dev.value().create_sub_buffer[DT](0, total)
            c.enqueue_copy(gc_dst, gc_src)

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


# ── the training objective ────────────────────────────────────────────────

comptime LeWMLossGraphRef[
    IN_CH: Int,
    IMG: Int,
    PATCH: Int,
    HIDDEN: Int,
    ENC_HEADS: Int,
    ENC_LAYERS: Int,
    EMB: Int,
    PROJ_H: Int,
    T: Int,
    ACT: Int,
    H: Int,
    N_PREDS: Int,
    PRED_HEADS: Int,
    PRED_DIM_HEAD: Int,
    PRED_FF: Int,
    DEPTH: Int,
    SIG_PROJ: Int,
    SIG_KNOTS: Int,
] = ComputeGraph[
    InputSlot["pixels", T * IN_CH * IMG * IMG],
    InputSlot["actions", T * ACT],
    Node[
        "emb",
        Tokenwise[
            T,
            LeWMEncoderRef[
                IN_CH, IMG, PATCH, HIDDEN, ENC_HEADS, ENC_LAYERS, EMB, PROJ_H
            ],
        ],
        "pixels",
    ],
    Node["act_emb", ActionEmbedderRef[T, ACT, EMB], "actions"],
    Node["ctx_x", Slice[T * EMB, 0, H * EMB], "emb"],
    Node["ctx_a", Slice[T * EMB, 0, H * EMB], "act_emb"],
    Node["tgt", Slice[T * EMB, N_PREDS * EMB, (N_PREDS + H) * EMB], "emb"],
    Node["x_pe", BiasAdd[H * EMB], "ctx_x"],
    Node[
        "pred_raw",
        RepeatConditional[
            DEPTH,
            ConditionalTransformerBlockRef[
                EMB, PRED_HEADS, H, PRED_FF, PRED_DIM_HEAD
            ],
        ],
        "x_pe",
        "ctx_a",
    ],
    Node["pred_ln", Tokenwise[H, LayerNorm[EMB, DT, TORCH_LN_EPS]], "pred_raw"],
    Node["pred", Tokenwise[H, ProjectorRef[EMB, PROJ_H, EMB]], "pred_ln"],
    Node["pl", MSEPerSample[H * EMB], "pred", "tgt"],
    Node["sig", SIGReg[EMB, T, SIG_PROJ, SIG_KNOTS], "emb"],
    Node["sig_s", Scale[1], "sig"],
    Node["loss", Add[1], "pl", "sig_s"],
]
"""`train.py:lejepa_forward`: loss = mean((pred - emb[:, n_preds:])²) +
λ·SIGReg(emb), per sample (B, 1); λ = `sig_s.multiplier`. Predictor context =
the first H frames (+ the learned position embedding `x_pe`), conditioned on
the first H action embeddings; NO stop-gradient on the target.

⚠ `MSEPerSample` is per-sample mean; the trainer's 1/B seed makes the batch
mean = torch's `.mean()` over (B, H, EMB). `PROJ_H` serves both projectors
(2048 for the paper)."""
