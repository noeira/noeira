"""EfficientZeroV2 **Atari** networks (storage nn) — the spatial-latent model.

Full-parity port of the official EZv2 Atari backbone
(`references/EfficientZeroV2-main/ez/agents/models/base_model.py`,
`ez/config/exp/atari.yaml`). Unlike the MLP / pixel MuZero configs in this repo
— which collapse the encoder to a **vector** latent — EZv2 Atari keeps a
**spatial** latent ``[64, 6, 6]`` (``num_channels=64``) all the way through the
representation, dynamics and heads. ``state_norm=False`` (no MinMaxNorm on the
latent), ``num_blocks=1``, ``action_embedding=True`` (dim 16),
``reduced_channels=16``, value-prefix LSTM reward, 601-atom value/reward support.

Implementation lever (see ``docs/EZV2_ATARI_PARITY.md`` §B): the spatial latent
rides through the planner / replay / MCTS adapters as a **flat** ``LATENT = 64·6·6
= 2304`` vector — every ``Conv2D`` here already consumes a CHW-flat input via its
``H``/``W`` comptime params (channel-concat of two ``[C,H,W]`` maps with equal
``H,W`` is exactly flat vector concat). So ``GumbelGPUMCTS`` /
``GPUMCTSSequenceReplay`` / the ``MZ*GPU`` adapters stay untouched; only
``LATENT`` changes (128 → 2304).

Spatial-shape convention (matches ``Conv2D`` / the ResNet composites):
    OH = (H + 2*P - K) // S + 1

Note on bias: the reference convs are ``bias=False`` (every conv is BN-followed,
so the bias is redundant — BN re-centres). Storage ``Conv2D`` carries a bias; for
the one conv whose BN is dropped (the no-BN downsample skip below) this is a
benign extra parameter, logged in the parity doc.

Storage migration: the dynamics ComputeGraph drops the legacy leading
output-count param (storage infers OUT_DIM from the last node) and the
``EZDynNetAtari`` wrapper threads ``ctx`` through ``forward``/``vjp`` (no
``TargetStorage``). It KEEPS ``set_attr`` because this backbone is BatchNorm-based
(unlike the BN-free Connect-Four spatial nets) — BN train/eval must propagate.
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT, TPB, LAYOUT_NCHW
from noeira.nn.core.initializer import Initializer
from noeira.nn.core.module import Module
from noeira.nn.core.tensor import Tensor, TensorImpl
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.param import ParamVisitor
from noeira.nn.core.amp import AMPPolicy, NoAMP
from noeira.nn.core.walkers import join_name
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.combinators.parallel import Parallel
from noeira.nn.combinators.projected_residual import ProjectedResidual
from noeira.nn.combinators.compute_graph import ComputeGraph
from noeira.nn.combinators.graph_decl import InputSlot, Node
from noeira.nn.primitives.conv2d import Conv2D
from noeira.nn.primitives.batch_norm_2d import BatchNorm2D
from noeira.nn.primitives.activations import ReLU
from noeira.nn.primitives.avg_pool_2d import AvgPool2D
from noeira.nn.primitives.linear import Linear
from noeira.nn.primitives.slice import Slice
from noeira.nn.primitives.concat import Concat
from noeira.nn.primitives.add import Add
from noeira.nn.primitives.broadcast_tokens import BroadcastTokens
from noeira.nn.primitives.layer_norm import LayerNorm
from noeira.nn.primitives.lstm_cell import LSTMCell
from noeira.nn.primitives.batch_norm_1d import BatchNorm1D
from noeira.nn.core.call import call_forward, call_vjp
from noeira.nn.models.resnet import ResBlockConv2DBN
from noeira.deep_agents.dreamerv3.zero_init import (
    scale_output_module, scale_output_graph,
)
from noeira.core.tiles import Tile


# ──────────────────────────────────────────────────────────────────────
# Downsample ResNet block — 3×3-stride-2 skip with NO BatchNorm
# ──────────────────────────────────────────────────────────────────────
#
# The EZv2 `DownSample.downsample_block` is a post-activated ResidualBlock with
# `downsample = conv3x3(IC→OC, stride=2)` (base_model.py:37, layer.py:11-35):
#
#   main:  Conv3×3-s2 → BN → ReLU → Conv3×3-s1 → BN
#   skip:  Conv3×3-s2                         (NO BN — differs from our
#                                              `ResBlockDownsampleBN`, whose skip
#                                              is 1×1-s2 + BN)
#   y   :  ReLU(main(x) + skip(x))
#
# Canonical K=3, P=1, S=2: both paths map H → (H-1)//2 + 1.
comptime EZDownBlockNoBN[
    IC: Int, OC: Int, H: Int, W: Int,
    ADT: DType = DT,
    LAYOUT: Int = LAYOUT_NCHW,
] = Sequential[
    ProjectedResidual[
        Sequential[
            Conv2D[IC, OC, 3, 2, 1, H, W, ADT, LAYOUT],
            BatchNorm2D[
                OC, (H - 1) // 2 + 1, (W - 1) // 2 + 1, ADT=ADT, LAYOUT=LAYOUT
            ],
            ReLU[OC * ((H - 1) // 2 + 1) * ((W - 1) // 2 + 1), ADT],
            Conv2D[
                OC, OC, 3, 1, 1,
                (H - 1) // 2 + 1, (W - 1) // 2 + 1, ADT, LAYOUT,
            ],
            BatchNorm2D[
                OC, (H - 1) // 2 + 1, (W - 1) // 2 + 1, ADT=ADT, LAYOUT=LAYOUT
            ],
        ],
        # skip: bare 3×3-stride-2 conv, no BN (reference `downsample=conv2`).
        Conv2D[IC, OC, 3, 2, 1, H, W, ADT, LAYOUT],
    ],
    ReLU[OC * ((H - 1) // 2 + 1) * ((W - 1) // 2 + 1), ADT],
]


# ──────────────────────────────────────────────────────────────────────
# h (Atari) — representation: stacked RGB obs → spatial latent [64, 6, 6]
# ──────────────────────────────────────────────────────────────────────
#
# `DownSample(in=IN_CH, out=C)` (base_model.py:14-60) then the
# `RepresentationNetwork`'s `num_blocks` identity ResBlocks (base_model.py:86-99,
# num_blocks=1 for Atari). For Atari `IN_CH = n_stack·3 = 12` (RGB), input 96×96,
# `C = num_channels = 64`. No `state_norm` ⇒ no MinMaxNorm tail.
#
# Spatial collapse (K=3,P=1 throughout; s2 conv/pool halve, s1 preserve):
#   96 →[conv1 s2] 48 →[resblocks1 s1] 48 →[down s2] 24 →[resblocks2 s1] 24
#      →[pool1 s2] 12 →[resblocks3 s1] 12 →[pool2 s2] 6 →[rep resblock s1] 6
#   ⇒ [64, 6, 6] = 2304.
comptime EZRepNetResNetAtari[
    IN_CH: Int, C: Int,
    ADT: DType = DT,
    LAYOUT: Int = LAYOUT_NCHW,
] = Sequential[
    # ── DownSample ───────────────────────────────────────────────────
    # conv1: Conv(IN_CH→C/2, k3,s2,p1) → BN → ReLU      (96 → 48)
    Conv2D[IN_CH, C // 2, 3, 2, 1, 96, 96, ADT, LAYOUT],
    BatchNorm2D[C // 2, 48, 48, ADT=ADT, LAYOUT=LAYOUT],
    ReLU[(C // 2) * 48 * 48, ADT],
    # resblocks1: 1× identity ResBlock(C/2) at 48×48
    ResBlockConv2DBN[C // 2, 3, 1, 48, 48, ADT=ADT, LAYOUT=LAYOUT],
    # downsample_block: ResBlock(C/2→C, s2) at 48 → 24 (no-BN skip)
    EZDownBlockNoBN[C // 2, C, 48, 48, ADT, LAYOUT],
    # resblocks2: 1× identity ResBlock(C) at 24×24
    ResBlockConv2DBN[C, 3, 1, 24, 24, ADT=ADT, LAYOUT=LAYOUT],
    # pooling1: AvgPool(k3,s2,p1) 24 → 12
    AvgPool2D[C, 3, 2, 1, 24, 24, ADT=ADT, LAYOUT=LAYOUT],
    # resblocks3: 1× identity ResBlock(C) at 12×12
    ResBlockConv2DBN[C, 3, 1, 12, 12, ADT=ADT, LAYOUT=LAYOUT],
    # pooling2: AvgPool(k3,s2,p1) 12 → 6
    AvgPool2D[C, 3, 2, 1, 12, 12, ADT=ADT, LAYOUT=LAYOUT],
    # ── RepresentationNetwork: num_blocks=1 identity ResBlock(C) at 6×6 ──
    ResBlockConv2DBN[C, 3, 1, 6, 6, ADT=ADT, LAYOUT=LAYOUT],
]


# ──────────────────────────────────────────────────────────────────────
# Atari spatial-latent fixed geometry (num_channels=64, 6×6, reduced=16)
# ──────────────────────────────────────────────────────────────────────
comptime EZ_C = 64                     # num_channels
comptime EZ_HW = 6                     # spatial side of the latent
comptime EZ_LATENT = EZ_C * EZ_HW * EZ_HW          # 2304 = [64,6,6]
comptime EZ_REDC = 16                  # reduced_channels (heads + action embed)
comptime EZ_PLANE = EZ_HW * EZ_HW                   # 36  = [1,6,6]
comptime EZ_EMB = EZ_REDC * EZ_HW * EZ_HW          # 576 = [16,6,6]
comptime EZ_CAT = (EZ_C + EZ_REDC) * EZ_HW * EZ_HW # 2880 = [80,6,6]


# ──────────────────────────────────────────────────────────────────────
# Action → embedding plane.  onehot(ACT) → normalized scalar → [1,6,6]
# plane (broadcast) → Conv1×1(1→16) → LayerNorm([16,6,6]) → ReLU.
#
# Reference (`base_model.py:134-150`, action_embedding=True, dim=16) fills a
# single plane with `action_idx / action_space_size`, then conv1×1/LN/relu.
# Here the onehot→scalar map is a learnable `Linear[ACT,1]` (DEVIATION: the
# reference uses a FIXED `a/A` ramp — a learnable per-action scalar is a
# superset; the frozen-ramp init is a documented follow-up). The scalar is
# then broadcast to the 36-cell plane by `BroadcastTokens`.
# ──────────────────────────────────────────────────────────────────────
comptime EZActionPlane[ACT: Int, ADT: DType = DT] = Sequential[
    Linear[ACT, 1, ADT],
    BroadcastTokens[EZ_PLANE, 1, ADT],     # scalar → [1,6,6] = 36
]


# ──────────────────────────────────────────────────────────────────────
# g (Atari) — convolutional dynamics DAG (ComputeGraph).
#   input  [z(2304) | onehot(ACT)]   (the flat MZDyn adapter contract)
#   output [z'(2304) | reward_logits(BINS)]
#
# z → state [64,6,6]; action → [16,6,6] embed; channel-concat → [80,6,6] →
# Conv3×3(80→64)→BN → += z (residual to input state) → ReLU → 1×ResBlock(64)
# = z'. Reward branch off z': Conv1×1(64→16)→BN→ReLU→MLP[32]→BINS (a fused,
# stateless stand-in for the value-prefix LSTM — see docs/EZV2_ATARI_PARITY.md
# §D; honoring the LSTM needs a shared-planner (h,c)-pool rewrite).
# Storage ComputeGraph: OUT_DIM is inferred from the last node ("out"), so there
# is no leading output-count param (vs. the legacy `ComputeGraph[OUT, ...]`).
# ──────────────────────────────────────────────────────────────────────
comptime EZDynAtariGraph[ACT: Int, BINS: Int] = ComputeGraph[
    InputSlot["in", EZ_LATENT + ACT],
    # split the flat input into the latent state and the action onehot
    Node["z",    Slice[EZ_LATENT + ACT, 0, EZ_LATENT],            "in"],
    Node["aoh",  Slice[EZ_LATENT + ACT, EZ_LATENT, EZ_LATENT + ACT], "in"],
    # action → [1,6,6] plane → Conv1×1(1→16) → LN([16,6,6]) → ReLU
    Node["apl",  EZActionPlane[ACT],                              "aoh"],
    Node["aemb", Sequential[
                     Conv2D[1, EZ_REDC, 1, 1, 0, EZ_HW, EZ_HW],
                     LayerNorm[EZ_EMB],
                     ReLU[EZ_EMB],
                 ],                                               "apl"],
    # channel-concat state(64) ⊕ action-embed(16) = [80,6,6]
    Node["cat",  Concat[EZ_LATENT, EZ_EMB],                       "z", "aemb"],
    # Conv3×3(80→64) → BN  (P=1 keeps 6×6)
    Node["c1",   Sequential[
                     Conv2D[EZ_C + EZ_REDC, EZ_C, 3, 1, 1, EZ_HW, EZ_HW],
                     BatchNorm2D[EZ_C, EZ_HW, EZ_HW],
                 ],                                               "cat"],
    # residual to the INPUT state, then ReLU
    Node["res",  Add[EZ_LATENT],                                  "c1", "z"],
    Node["rl",   ReLU[EZ_LATENT],                                 "res"],
    # 1× post-activated ResBlock(64) → next state z'
    Node["zp",   ResBlockConv2DBN[EZ_C, 3, 1, EZ_HW, EZ_HW],      "rl"],
    # reward branch off z': Conv1×1(64→16)→BN→ReLU→MLP[32]→BINS
    Node["rew",  Sequential[
                     Conv2D[EZ_C, EZ_REDC, 1, 1, 0, EZ_HW, EZ_HW],
                     BatchNorm2D[EZ_REDC, EZ_HW, EZ_HW],
                     ReLU[EZ_EMB],
                     Linear[EZ_EMB, 32],
                     ReLU[32],
                     Linear[32, BINS],
                 ],                                               "zp"],
    # pack [z' | reward_logits]
    Node["out",  Concat[EZ_LATENT, BINS],                         "zp", "rew"],
]


# GPU LayoutTensor copy for the vjp input-grad copy-back (dst, src both real
# graph-pool / grad-pool buffers).
def _ezdyn_copy_kernel[N: Int, ADT: DType = DT](
    dst: Tile[ADT, N],
    src: Tile[ADT, N],
):
    var idx = Int(global_idx.x)
    if idx < N:
        dst[idx] = rebind[Scalar[ADT]](src[idx])


# ──────────────────────────────────────────────────────────────────────
# EZDynNetAtari — single-input Module wrapper over the dynamics graph.
# forward feeds the slot then runs the graph; vjp runs the graph backward then
# copies the slot's accumulated input-grad into grad_inputs[0] (the unroll's
# BPTT needs ∂/∂[z|action]). Param/state walks delegate to the graph. KEEPS
# set_attr (BatchNorm train/eval). Storage: no TargetStorage (ctx arg-threaded).
# Contract: IN_DIMS[0]=LATENT+ACT, OUT_DIM=LATENT+BINS.
# ──────────────────────────────────────────────────────────────────────
struct EZDynNetAtari[ACT: Int, BINS: Int](Module):
    comptime ARITY: Int = 1
    comptime LATENT = EZ_LATENT
    comptime IN_DIM = Self.LATENT + Self.ACT
    comptime IN_DIMS = Array[Int, 1](fill=Self.IN_DIM)
    comptime OUT_DIM = Self.LATENT + Self.BINS
    comptime Graph = EZDynAtariGraph[Self.ACT, Self.BINS]

    var graph: Self.Graph

    def __init__(out self):
        comptime assert Self.Graph.OUT_DIM == Self.OUT_DIM, (
            "EZDynNetAtari: graph OUT_DIM must equal LATENT+BINS"
        )
        self.graph = Self.Graph()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "EZDynNetAtari: target must be 'cpu' or 'gpu'"
        )
        var s = Self()
        s.graph = Self.Graph.make[target, INIT](ctx)
        return s^

    def forward[
        target: StaticString,
        B: Int,
        o: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        inputs: TensorRefs[Self.ARITY, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        self.graph.set_input["in", B](inputs[0], ctx)
        self.graph.forward[B, target, POLICY=POLICY](out, ctx)

    def vjp[
        target: StaticString,
        B: Int,
        ofi: MutOrigin,
        ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[Self.ARITY, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[Self.ARITY, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        self.graph.vjp[B, target, POLICY=POLICY](grad_output, ctx)
        comptime N = B * Self.IN_DIM
        comptime if target == "cpu":
            ref gin = self.graph.grad_input["in"]()
            ref dst = grad_inputs[0]
            for i in range(N):
                dst.data[i] = gin.data[i]
        else:
            var c = ctx.value()
            comptime nb = (N + TPB - 1) // TPB
            c.enqueue_function[_ezdyn_copy_kernel[N]](
                grad_inputs[0].tt["gpu", N](),
                self.graph.grad_input["in"]().tt["gpu", N](),
                grid_dim=nb, block_dim=TPB,
            )

    def for_each_param[
        target: StaticString,
        V: ParamVisitor,
    ](
        mut self,
        mut visitor: V,
        ctx: Optional[DeviceContext],
        prefix: String = String(""),
    ) raises:
        self.graph.for_each_param[target, V](
            visitor, ctx, join_name(prefix, String("graph"))
        )

    def for_each_state[
        target: StaticString,
        V: ParamVisitor,
    ](
        mut self,
        mut visitor: V,
        ctx: Optional[DeviceContext],
        prefix: String = String(""),
    ) raises:
        self.graph.for_each_state[target, V](
            visitor, ctx, join_name(prefix, String("graph"))
        )

    def set_attr[ATTR: StaticString](mut self, value: Scalar[DT]):
        self.graph.set_attr[ATTR](value)


# ──────────────────────────────────────────────────────────────────────
# f (Atari) — convolutional prediction: z → [policy_logits | value_logits]
#
# `ValuePolicyNetwork` (base_model.py:166-213): a shared ResBlock(64), then two
# conv heads — Conv1×1(64→16)→BN→ReLU→flatten 576→MLP[32]→(ACT policy / BINS
# value categorical). `init_zero=True` zeros the last layer of each head (stable
# zero outputs at init) — see `ez_atari_init_zero_pred`. Activation in the head
# MLP is ELU in the reference; nn has no ELU, so ReLU is substituted (logged).
# Output packing `[policy(ACT) | value(BINS)]` matches `MZPredNet`, so the
# planner's prediction adapter slices it unchanged.
# ──────────────────────────────────────────────────────────────────────
comptime EZPredNetAtari[ACT: Int, BINS: Int, ADT: DType = DT] = Sequential[
    # shared num_blocks=1 ResBlock(64)
    ResBlockConv2DBN[EZ_C, 3, 1, EZ_HW, EZ_HW, ADT=ADT],
    Parallel[
        # policy head: Conv1×1(64→16)→BN→ReLU→MLP[32]→ACT
        Sequential[
            Conv2D[EZ_C, EZ_REDC, 1, 1, 0, EZ_HW, EZ_HW, ADT],
            BatchNorm2D[EZ_REDC, EZ_HW, EZ_HW, ADT=ADT],
            ReLU[EZ_EMB, ADT],
            Linear[EZ_EMB, 32, ADT],
            ReLU[32, ADT],
            Linear[32, ACT, ADT],
        ],
        # value head: Conv1×1(64→16)→BN→ReLU→MLP[32]→BINS
        Sequential[
            Conv2D[EZ_C, EZ_REDC, 1, 1, 0, EZ_HW, EZ_HW, ADT],
            BatchNorm2D[EZ_REDC, EZ_HW, EZ_HW, ADT=ADT],
            ReLU[EZ_EMB, ADT],
            Linear[EZ_EMB, 32, ADT],
            ReLU[32, ADT],
            Linear[32, BINS, ADT],
        ],
    ],
]


# ──────────────────────────────────────────────────────────────────────
# init_zero (EZv2 `init_zero=True`) — scale each head's OUTPUT layer at init.
#
# base_model.py zeros the last Linear of the value, policy, and reward heads
# so the model starts with neutral value/reward predictions and a uniform
# policy prior (stable MCTS targets before the heads have learned). We reuse
# the DreamerV3 `scale_output_*` visitors (scale=0.0 == exact zero-init).
#
# Param names follow the STORAGE combinator naming (Sequential→`.{i}`, Parallel→
# `.{i}` by branch INDEX [NOT `.a`/`.b` as in legacy nn], ComputeGraph node→
# `.{node}`, Linear leaf→`.weight`/`.bias`):
#   • EZPredNetAtari = Sequential[ResBlock(0), Parallel(1)[policy=0, value=1]];
#     each head's output Linear is Sequential child 5 → `1.0.5.*` / `1.1.5.*`.
#   • EZDynAtariGraph reward branch is node `rew` (a 6-child Sequential); its
#     output Linear is child 5 → `rew.5.*`.
# Only the OUTPUT layer is scaled — scaling the whole head would choke the
# hidden layers' gradient (see zero_init.mojo).
# ──────────────────────────────────────────────────────────────────────
def ez_atari_init_zero_pred[
    target: StaticString, ACT: Int, BINS: Int, ADT: DType = DT
](
    mut pred: EZPredNetAtari[ACT, BINS, ADT=ADT],
    ctx: Optional[DeviceContext] = None,
    scale: Scalar[DT] = Scalar[DT](0.0),
) raises:
    """Zero (or `scale`) the policy + value head output Linears of the
    prediction net. scale=0.0 → uniform policy + neutral value at init."""
    scale_output_module[target, EZPredNetAtari[ACT, BINS, ADT=ADT]](
        pred, "1.0.5.weight", "1.0.5.bias", scale, ctx
    )
    scale_output_module[target, EZPredNetAtari[ACT, BINS, ADT=ADT]](
        pred, "1.1.5.weight", "1.1.5.bias", scale, ctx
    )


def ez_atari_init_zero_dyn[
    target: StaticString, ACT: Int, BINS: Int
](
    mut dyn: EZDynNetAtari[ACT, BINS],
    ctx: Optional[DeviceContext] = None,
    scale: Scalar[DT] = Scalar[DT](0.0),
) raises:
    """Zero (or `scale`) the reward head output Linear of the dynamics net
    (the fused value-prefix stand-in). scale=0.0 → neutral reward at init."""
    scale_output_graph[target](
        dyn.graph, "rew.5.weight", "rew.5.bias", scale, ctx
    )


# ══════════════════════════════════════════════════════════════════════
# VALUE-PREFIX path (EZv2 `value_prefix=True`, Atari only).  Stage 3 — see
# docs/EZV2_ATARI_PARITY.md. Reverses deliberate-deviation B1: the fused
# stateless reward branch above is replaced by a stateful LSTM reward head
# whose (h,c) is carried across the unroll and reset every LSTM_HORIZON
# steps, predicting a cumulative *value prefix* instead of a per-step reward.
#
# This whole block is dormant unless a caller opts in (the planner's
# `LSTM_HIDDEN` gate, the driver's value_prefix flag). The non-VP nets above
# are untouched and remain the default.
# ══════════════════════════════════════════════════════════════════════

# atari.yaml: lstm_hidden_size=512, lstm_horizon_len=5.
comptime EZ_LSTM_HIDDEN = 512
comptime EZ_RHID = 2 * EZ_LSTM_HIDDEN   # packed [h | c] carry width = 1024
comptime EZ_LSTM_HORIZON = 5


# ──────────────────────────────────────────────────────────────────────
# g (Atari, VP) — z'-ONLY dynamics graph: [z(2304) | onehot(ACT)] → z'(2304).
# Identical to `EZDynAtariGraph` MINUS the fused reward branch (`rew`/`out`
# nodes): the last node is `zp`, so storage infers OUT_DIM = EZ_LATENT. The
# reward is produced separately by the stateful `EZRewardLSTMAtari`.
# ──────────────────────────────────────────────────────────────────────
comptime EZDynZGraph[ACT: Int, ADT: DType = DT] = ComputeGraph[
    InputSlot["in", EZ_LATENT + ACT, ADT=ADT],
    Node["z",    Slice[EZ_LATENT + ACT, 0, EZ_LATENT, ADT],            "in"],
    Node["aoh",  Slice[EZ_LATENT + ACT, EZ_LATENT, EZ_LATENT + ACT, ADT], "in"],
    Node["apl",  EZActionPlane[ACT, ADT],                              "aoh"],
    Node["aemb", Sequential[
                     Conv2D[1, EZ_REDC, 1, 1, 0, EZ_HW, EZ_HW, ADT],
                     LayerNorm[EZ_EMB, ADT],
                     ReLU[EZ_EMB, ADT],
                 ],                                               "apl"],
    Node["cat",  Concat[EZ_LATENT, EZ_EMB, ADT=ADT],                  "z", "aemb"],
    Node["c1",   Sequential[
                     Conv2D[EZ_C + EZ_REDC, EZ_C, 3, 1, 1, EZ_HW, EZ_HW, ADT],
                     BatchNorm2D[EZ_C, EZ_HW, EZ_HW, ADT=ADT],
                 ],                                               "cat"],
    Node["res",  Add[EZ_LATENT, ADT],                                 "c1", "z"],
    Node["rl",   ReLU[EZ_LATENT, ADT],                                "res"],
    Node["zp",   ResBlockConv2DBN[EZ_C, 3, 1, EZ_HW, EZ_HW, ADT=ADT],     "rl"],
]


# ──────────────────────────────────────────────────────────────────────
# EZDynZNetAtari — single-input Module wrapper over the z'-only graph.
# Mirrors EZDynNetAtari but OUT_DIM = LATENT (no reward slot). Used by the
# VP unroll (training) and folded into EZDynVPNetAtari (search).
# ──────────────────────────────────────────────────────────────────────
struct EZDynZNetAtari[ACT: Int, ADT: DType = DT](Module):
    comptime ARITY: Int = 1
    comptime ACT_DT = Self.ADT
    comptime LATENT = EZ_LATENT
    comptime IN_DIM = Self.LATENT + Self.ACT
    comptime IN_DIMS = Array[Int, 1](fill=Self.IN_DIM)
    comptime OUT_DIM = Self.LATENT
    comptime Graph = EZDynZGraph[Self.ACT, ADT=Self.ADT]

    var graph: Self.Graph

    def __init__(out self):
        comptime assert Self.Graph.OUT_DIM == Self.OUT_DIM, (
            "EZDynZNetAtari: graph OUT_DIM must equal LATENT"
        )
        self.graph = Self.Graph()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        var s = Self()
        s.graph = Self.Graph.make[target, INIT](ctx)
        return s^

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        inputs: TensorRefs[Self.ARITY, o, Self.ACT_DT],
        mut out: TensorImpl[Self.ACT_DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        self.graph.set_input["in", B](inputs[0], ctx)
        self.graph.forward[B, target, POLICY=POLICY](out, ctx)

    def vjp[
        target: StaticString, B: Int, ofi: MutOrigin, ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[Self.ARITY, ofi, Self.ACT_DT],
        mut grad_output: TensorImpl[Self.ACT_DT],
        grad_inputs: TensorRefs[Self.ARITY, ogi, Self.ACT_DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        self.graph.vjp[B, target, POLICY=POLICY](grad_output, ctx)
        comptime N = B * Self.IN_DIM
        comptime if target == "cpu":
            ref gin = self.graph.grad_input["in"]()
            ref dst = grad_inputs[0]
            for i in range(N):
                dst.data[i] = gin.data[i]
        else:
            var c = ctx.value()
            comptime nb = (N + TPB - 1) // TPB
            c.enqueue_function[_ezdyn_copy_kernel[N, Self.ACT_DT]](
                grad_inputs[0].tt["gpu", N](),
                self.graph.grad_input["in"]().tt["gpu", N](),
                grid_dim=nb, block_dim=TPB,
            )

    def for_each_param[target: StaticString, V: ParamVisitor](
        mut self, mut visitor: V, ctx: Optional[DeviceContext],
        prefix: String = String(""),
    ) raises:
        self.graph.for_each_param[target, V](
            visitor, ctx, join_name(prefix, String("graph"))
        )

    def for_each_state[target: StaticString, V: ParamVisitor](
        mut self, mut visitor: V, ctx: Optional[DeviceContext],
        prefix: String = String(""),
    ) raises:
        self.graph.for_each_state[target, V](
            visitor, ctx, join_name(prefix, String("graph"))
        )

    def set_attr[ATTR: StaticString](mut self, value: Scalar[DT]):
        self.graph.set_attr[ATTR](value)


# ──────────────────────────────────────────────────────────────────────
# EZRewardLSTMAtari — stateful value-prefix reward head (`SupportLSTMNetwork`,
# base_model.py:234). ARITY=2 conceptually (z', [h;c]) but it is RECURRENT, so
# — like LSTMCell — the Module.forward/vjp stubs RAISE and callers use the
# step API instead (reward_step_forward / reward_step_backward), owning the
# (h,c) carry buffers. The conv stem + post-MLP are plain sub-Modules driven
# per step (re-forward-in-reverse, matching blocks.mojo's dyn handling).
#
#   stem : Conv1×1(64→16)→BN→ReLU                 [LATENT 2304] → [EZ_EMB 576]
#   cell : LSTMCell[576, 512]                      x576,(h,c) → (h',c')
#   head : BN1D→ReLU→Linear[512,32]→ReLU→Linear[32,BINS]   h'512 → BINS
#
# Caller-owned step buffers (CPU): h,c each sized 2·B·HIDDEN (slab0=prev,
# slab1=out); cache sized B·CACHE_SIZE.
# ──────────────────────────────────────────────────────────────────────
struct EZRewardLSTMAtari[BINS: Int](Module):
    comptime ARITY: Int = 2
    comptime LATENT = EZ_LATENT
    comptime HIDDEN = EZ_LSTM_HIDDEN
    comptime RHID = EZ_RHID
    comptime IN_DIMS = Self._build_in_dims()
    comptime OUT_DIM = Self.BINS + Self.RHID    # [vp_logits | h' | c']

    @staticmethod
    def _build_in_dims() -> Array[Int, 2]:
        var d = Array[Int, 2](fill=0)
        d[0] = Self.LATENT
        d[1] = Self.RHID
        return d^

    comptime Stem = Sequential[
        Conv2D[EZ_C, EZ_REDC, 1, 1, 0, EZ_HW, EZ_HW],
        BatchNorm2D[EZ_REDC, EZ_HW, EZ_HW],
        ReLU[EZ_EMB],
    ]
    comptime Cell = LSTMCell[EZ_EMB, EZ_LSTM_HIDDEN]
    comptime Head = Sequential[
        BatchNorm1D[EZ_LSTM_HIDDEN],
        ReLU[EZ_LSTM_HIDDEN],
        Linear[EZ_LSTM_HIDDEN, 32],
        ReLU[32],
        Linear[32, Self.BINS],
    ]
    comptime CACHE_SIZE = Self.Cell.CACHE_SIZE

    var stem: Self.Stem
    var cell: Self.Cell
    var head: Self.Head
    # single-step scratch (lazy by B)
    var stem_out: Tensor   # [B·EZ_EMB]  stem fwd output / cell x
    var hin: Tensor        # [B·HIDDEN]  head input (= h_t)
    var dhin: Tensor       # [B·HIDDEN]  grad wrt head input (= dh from head)
    var dstem: Tensor      # [B·EZ_EMB]  grad wrt stem output (cell dx)

    def __init__(out self):
        self.stem = Self.Stem()
        self.cell = Self.Cell()
        self.head = Self.Head()
        self.stem_out = Tensor()
        self.hin = Tensor()
        self.dhin = Tensor()
        self.dstem = Tensor()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "EZRewardLSTMAtari: target must be 'cpu' or 'gpu'"
        )
        var s = Self()
        s.stem = Self.Stem.make[target, INIT](ctx)
        s.cell = Self.Cell.make[target, INIT](ctx)
        s.head = Self.Head.make[target, INIT](ctx)
        return s^

    # ----- Module conformance: recurrent → use the step API (stubs raise) ---
    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP,
    ](
        mut self, inputs: TensorRefs[Self.ARITY, o], mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        raise Error(
            "EZRewardLSTMAtari is recurrent — use reward_step_forward /"
            " reward_step_backward, not Module.forward"
        )

    def vjp[
        target: StaticString, B: Int, ofi: MutOrigin, ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self, forward_input: TensorRefs[Self.ARITY, ofi],
        mut grad_output: Tensor, grad_inputs: TensorRefs[Self.ARITY, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        raise Error(
            "EZRewardLSTMAtari is recurrent — use reward_step_backward,"
            " not Module.vjp"
        )

    # ----- reflection: recurse into the three sub-modules -------------------
    def for_each_param[target: StaticString, V: ParamVisitor](
        mut self, mut visitor: V, ctx: Optional[DeviceContext],
        prefix: String = String(""),
    ) raises:
        self.stem.for_each_param[target, V](
            visitor, ctx, join_name(prefix, String("stem")))
        self.cell.for_each_param[target, V](
            visitor, ctx, join_name(prefix, String("cell")))
        self.head.for_each_param[target, V](
            visitor, ctx, join_name(prefix, String("head")))

    def for_each_state[target: StaticString, V: ParamVisitor](
        mut self, mut visitor: V, ctx: Optional[DeviceContext],
        prefix: String = String(""),
    ) raises:
        self.stem.for_each_state[target, V](
            visitor, ctx, join_name(prefix, String("stem")))
        self.cell.for_each_state[target, V](
            visitor, ctx, join_name(prefix, String("cell")))
        self.head.for_each_state[target, V](
            visitor, ctx, join_name(prefix, String("head")))

    def set_attr[ATTR: StaticString](mut self, value: Scalar[DT]):
        # BN train/eval toggle propagates to the stem + head (cell has no BN).
        self.stem.set_attr[ATTR](value)
        self.head.set_attr[ATTR](value)

    # ----- recurrent step API ---------------------------------------------
    # h, c: caller-owned, sized 2·B·HIDDEN (slab0 = prev, slab1 = out).
    # cache: caller-owned, sized B·CACHE_SIZE (repopulated each forward).
    def reward_step_forward[target: StaticString, B: Int](
        mut self,
        mut zprime: Tensor,
        mut h: Tensor,
        mut c: Tensor,
        mut cache: Tensor,
        mut out_vp: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """One reward step: stem(z') → LSTM(·, h_prev, c_prev) → head(h_t) =
        value-prefix logits. Reads h/c slab0, writes h/c slab1 + cache + out_vp."""
        comptime H = Self.HIDDEN
        self.stem_out.ensure[target](B * EZ_EMB, ctx)
        self.hin.ensure[target](B * H, ctx)
        out_vp.ensure[target](B * Self.BINS, ctx)
        call_forward[target, B](
            self.stem, TensorRefs[1](zprime), self.stem_out, ctx)
        self.cell.step_forward[target, B](
            self.stem_out, h, c, cache, ctx,
            x_off=0, h_prev_off=0, c_prev_off=0,
            h_t_off=B * H, c_t_off=B * H, cache_off=0,
        )
        # head input = h_t (slab1)
        comptime if target == "cpu":
            for i in range(B * H):
                self.hin.data[i] = h.data[B * H + i]
        else:
            var dctx = ctx.value()
            var h_sub = h.dev.value().create_sub_buffer[DT](B * H, B * H)
            # hin may be over-sized from a prior larger-B call (the fused dyn's
            # reward head is shared between B=1 search and B=N train) — copy into
            # an exactly-B*H sub-view so enqueue_copy's src/dst sizes match.
            var hin_sub = self.hin.dev.value().create_sub_buffer[DT](0, B * H)
            dctx.enqueue_copy(hin_sub, h_sub)
        call_forward[target, B](
            self.head, TensorRefs[1](self.hin), out_vp, ctx)

    def reward_step_backward[target: StaticString, B: Int](
        mut self,
        mut zprime: Tensor,        # re-forwarded stem input for this step
        mut grad_out_vp: Tensor,   # [B·BINS] grad from the VP loss
        mut h: Tensor,             # slab0=h_prev, slab1=h_t (re-forwarded)
        mut c: Tensor,
        mut cache: Tensor,         # re-forwarded
        mut dh_carry: Tensor,      # [B·HIDDEN] grad wrt h_t from step k+1 (0 at end)
        mut dc_carry: Tensor,      # [B·HIDDEN] grad wrt c_t from step k+1 (0 at end)
        mut grad_zprime: Tensor,   # [B·LATENT] OUT grad wrt z'
        mut dh_prev: Tensor,       # [B·HIDDEN] OUT grad wrt h_{k-1} (next carry)
        mut dc_prev: Tensor,       # [B·HIDDEN] OUT grad wrt c_{k-1}
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """BPTT one reward step. MUST be preceded by a reward_step_forward with
        the SAME (h_prev, c_prev) to repopulate stem/head caches + cache buf."""
        comptime H = Self.HIDDEN
        self.dhin.ensure[target](B * H, ctx)
        self.dstem.ensure[target](B * EZ_EMB, ctx)
        grad_zprime.ensure[target](B * Self.LATENT, ctx)
        # head: grad_out_vp → dhin (grad wrt h_t)
        call_vjp[target, B](
            self.head, TensorRefs[1](self.hin), grad_out_vp,
            TensorRefs[1](self.dhin), ctx)
        # combine with the recurrent carry: dh_total = dhin + dh_carry
        comptime if target == "cpu":
            for i in range(B * H):
                self.dhin.data[i] += dh_carry.data[i]
        else:
            var dctx = ctx.value()
            comptime nb = (B * H + TPB - 1) // TPB
            dctx.enqueue_function[_ez_add_inplace_kernel[B * H]](
                self.dhin.tt["gpu", B * H](),
                dh_carry.tt["gpu", B * H](),
                grid_dim=nb, block_dim=TPB,
            )
        # cell BPTT: dh=dhin(=dh_total), dc=dc_carry → dx(dstem), dh_prev, dc_prev
        self.cell.step_backward[target, B](
            self.dhin, dc_carry, self.stem_out, h, c, cache,
            self.dstem, dh_prev, dc_prev, ctx,
            dh_off=0, dc_off=0, x_off=0,
            h_prev_off=0, c_prev_off=0, cache_off=0,
            dx_off=0, dh_prev_off=0, dc_prev_off=0,
        )
        # stem: dstem → grad_zprime
        call_vjp[target, B](
            self.stem, TensorRefs[1](zprime), self.dstem,
            TensorRefs[1](grad_zprime), ctx)


def ez_atari_init_zero_reward[
    target: StaticString, BINS: Int
](
    mut rew: EZRewardLSTMAtari[BINS],
    ctx: Optional[DeviceContext] = None,
    scale: Scalar[DT] = Scalar[DT](0.0),
) raises:
    """Zero (or `scale`) the value-prefix head's OUTPUT Linear (head child 4 →
    `4.*`). scale=0.0 → neutral value-prefix at init (matches init_zero=True)."""
    scale_output_module[target, EZRewardLSTMAtari[BINS].Head](
        rew.head, "4.weight", "4.bias", scale, ctx
    )


# GPU elementwise a += b (reward-head carry combine).
def _ez_add_inplace_kernel[N: Int](
    a: Tile[DT, N],
    b: Tile[DT, N],
):
    var idx = Int(global_idx.x)
    if idx < N:
        a[idx] = rebind[Scalar[DT]](a[idx]) + rebind[Scalar[DT]](b[idx])


# GPU pack [z'(LATENT) | vp(BINS)] per batch row → out[B·(LATENT+BINS)].
def _ez_pack_zvp_kernel[B: Int, LATENT: Int, BINS: Int](
    dst: Tile[DT, B * (LATENT + BINS)],
    zp: Tile[DT, B * LATENT],
    vp: Tile[DT, B * BINS],
):
    comptime OUT = LATENT + BINS
    var gid = Int(global_idx.x)
    if gid >= B * OUT:
        return
    var b = gid // OUT
    var i = gid % OUT
    if i < LATENT:
        dst[gid] = rebind[Scalar[DT]](zp[b * LATENT + i])
    else:
        dst[gid] = rebind[Scalar[DT]](vp[b * BINS + (i - LATENT)])


# ──────────────────────────────────────────────────────────────────────
# EZDynVPNetAtari — FUSED value-prefix dynamics for SEARCH (decision B1.1).
# A drop-in `[z|act] → [z'|vp_logits]` dynamics (IN=LATENT+ACT, OUT=LATENT+BINS,
# identical contract to EZDynNetAtari) so it plugs into the EXISTING
# GumbelGPUMCTS / MZ adapters / replay with NO orchestrator changes. It OWNS the
# z'-only dynamics + the stateful LSTM reward head (so training drives the same
# weights via the public `.dynz` / `.rew` fields and their step API). In SEARCH
# the reward head runs with ZERO (h,c) per node (stateless ≡ horizon-1); the
# search-side (h,c) carry + prefix-diff is the deferred parity fast-follow (see
# docs/EZV2_ATARI_PARITY.md decision B1.1). `vjp` raises — search is forward-only
# and training drives `dynz`/`rew` directly. Reflection/set_attr recurse both.
# ──────────────────────────────────────────────────────────────────────
struct EZDynVPNetAtari[ACT: Int, BINS: Int](Module):
    comptime ARITY: Int = 1
    comptime LATENT = EZ_LATENT
    comptime IN_DIM = Self.LATENT + Self.ACT
    comptime IN_DIMS = Array[Int, 1](fill=Self.IN_DIM)
    comptime OUT_DIM = Self.LATENT + Self.BINS
    comptime HIDDEN = EZ_LSTM_HIDDEN

    var dynz: EZDynZNetAtari[Self.ACT]
    var rew: EZRewardLSTMAtari[Self.BINS]
    # search scratch (lazy by B): z', zero (h,c) carry, cache, vp logits
    var _zp: Tensor
    var _h0: Tensor
    var _c0: Tensor
    var _cache: Tensor
    var _vp: Tensor

    def __init__(out self):
        self.dynz = EZDynZNetAtari[Self.ACT]()
        self.rew = EZRewardLSTMAtari[Self.BINS]()
        self._zp = Tensor()
        self._h0 = Tensor()
        self._c0 = Tensor()
        self._cache = Tensor()
        self._vp = Tensor()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        var s = Self()
        s.dynz = EZDynZNetAtari[Self.ACT].make[target, INIT](ctx)
        s.rew = EZRewardLSTMAtari[Self.BINS].make[target, INIT](ctx)
        return s^

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        inputs: TensorRefs[Self.ARITY, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime H = Self.HIDDEN
        comptime LAT = Self.LATENT
        self._zp.ensure[target](B * LAT, ctx)
        self._h0.ensure[target](2 * B * H, ctx)
        self._c0.ensure[target](2 * B * H, ctx)
        self._cache.ensure[target](B * EZRewardLSTMAtari[Self.BINS].CACHE_SIZE, ctx)
        self._vp.ensure[target](B * Self.BINS, ctx)
        out.ensure[target](B * Self.OUT_DIM, ctx)
        # z' = g_z([z | act])
        self.dynz.forward[target, B](inputs, self._zp, ctx)
        # zero the (h,c) carry (stateless-in-search), then reward LSTM step
        comptime if target == "cpu":
            for i in range(2 * B * H):
                self._h0.data[i] = Scalar[DT](0.0)
                self._c0.data[i] = Scalar[DT](0.0)
        else:
            self._h0.dev.value().enqueue_fill(Scalar[DT](0.0))
            self._c0.dev.value().enqueue_fill(Scalar[DT](0.0))
        self.rew.reward_step_forward[target, B](
            self._zp, self._h0, self._c0, self._cache, self._vp, ctx)
        # pack [z' | vp_logits]
        comptime if target == "cpu":
            for b in range(B):
                for i in range(LAT):
                    out.data[b * Self.OUT_DIM + i] = self._zp.data[b * LAT + i]
                for i in range(Self.BINS):
                    out.data[b * Self.OUT_DIM + LAT + i] = self._vp.data[b * Self.BINS + i]
        else:
            var c = ctx.value()
            comptime nb = (B * Self.OUT_DIM + TPB - 1) // TPB
            c.enqueue_function[_ez_pack_zvp_kernel[B, LAT, Self.BINS]](
                out.tt["gpu", B * Self.OUT_DIM](),
                self._zp.tt["gpu", B * LAT](),
                self._vp.tt["gpu", B * Self.BINS](),
                grid_dim=nb, block_dim=TPB,
            )

    def vjp[
        target: StaticString, B: Int, ofi: MutOrigin, ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self, forward_input: TensorRefs[Self.ARITY, ofi],
        mut grad_output: Tensor, grad_inputs: TensorRefs[Self.ARITY, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        raise Error(
            "EZDynVPNetAtari.forward is search-only; training drives .dynz/.rew"
            " via ezv2_unroll_train_step_*_vp"
        )

    def for_each_param[target: StaticString, V: ParamVisitor](
        mut self, mut visitor: V, ctx: Optional[DeviceContext],
        prefix: String = String(""),
    ) raises:
        self.dynz.for_each_param[target, V](
            visitor, ctx, join_name(prefix, String("dynz")))
        self.rew.for_each_param[target, V](
            visitor, ctx, join_name(prefix, String("rew")))

    def for_each_state[target: StaticString, V: ParamVisitor](
        mut self, mut visitor: V, ctx: Optional[DeviceContext],
        prefix: String = String(""),
    ) raises:
        self.dynz.for_each_state[target, V](
            visitor, ctx, join_name(prefix, String("dynz")))
        self.rew.for_each_state[target, V](
            visitor, ctx, join_name(prefix, String("rew")))

    def set_attr[ATTR: StaticString](mut self, value: Scalar[DT]):
        self.dynz.set_attr[ATTR](value)
        self.rew.set_attr[ATTR](value)
