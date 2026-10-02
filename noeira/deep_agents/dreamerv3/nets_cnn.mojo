"""DreamerV3 CNN encoder / decoder for pixel observations (P1 of the pixel arc).

The MLP `DreamerEncoder`/`DreamerDecoder` in `nets.mojo` only handle vector obs.
Pixel obs (CarRacing: stacked grayscale frames) need a convolutional encoder
(image → tokens) and a transposed-conv decoder (feature → image). Like the rest
of the Dreamer nets these are pure `comptime` `Sequential` aliases over existing
primitives — `Conv2D` (down) and the new `Conv2DTranspose` (up) — so they get
forward / vjp / param-walkers for free.

In this storage framework a tensor is always a flat `[B, DIM]` slab; `Conv2D` /
`Conv2DTranspose` interpret the flat `DIM` as `[C,H,W]` (or `[H,W,C]` for NHWC)
via index math. So there is NO reshape/flatten module between the conv stack and
the Linear bottleneck — the flat conv output IS the Linear input, and the flat
Linear output IS the deconv input (as long as the LAYOUT is consistent).

Architecture = the DreamerV3 image backbone: a fixed 4-layer stride-2 stack
(kernel 4, stride 2, pad 1) with channels `BASE·{1,2,4,8}`. Each stride-2 conv
HALVES an even spatial dim; each transposed conv DOUBLES it. So:

    encoder  C×H×W → BASE×H/2 → 2B×H/4 → 4B×H/8 → 8B×H/16 → Linear → tokens[TOKEN]
    decoder  feat[FEATIN] → Linear → 8B×H/16 → 4B×H/8 → 2B×H/4 → B×H/2 → C×H×W

`H` (== `W`) MUST be divisible by 16 (so every intermediate dim stays even).
For CarRacing pick a 16-divisible resolution: 64 (→ minres 4) or 96 (→ minres 6).

Norm: the reference (`rssm.py`) inserts a norm (default RMSNorm) after each conv,
applied over the CHANNEL axis (per spatial location), then the activation
(Conv→Norm→act). We match that with `ConvRMSNorm[C, HW]` (channel-wise RMSNorm
for NCHW maps; γ size = channels) in every _ConvDown/_ConvUp and the decoder's
initial projection. The encoder's final Linear (→tokens) and the decoder's
output deconv stay raw, mirroring the reference's unnormalized output layers.

Activation `A` defaults to `GELUTanhOp` (matching `nets.mojo`); the training config
passes `SwishOp` (SiLU). The final encoder Linear and final decoder deconv are
LINEAR (no activation) — raw tokens / raw pixel logits.
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from noeira.nn.combinators.sequential import Sequential
from noeira.nn.primitives.conv2d import Conv2D
from noeira.nn.primitives.conv2d_transpose import Conv2DTranspose
from noeira.nn.primitives.linear import Linear
from noeira.nn.primitives.block_linear import BlockLinear
from noeira.nn.primitives.rms_norm import RMSNorm
from noeira.nn.primitives.conv_rms_norm import ConvRMSNorm
from noeira.nn.primitives.max_pool_2d import MaxPool2D
from noeira.nn.primitives.upsample2x import Upsample2x
from noeira.nn.primitives.elementwise import Elementwise
from noeira.nn.primitives.ops.gelu_tanh_op import GELUTanhOp
from noeira.nn.primitives.ops.center_half_op import CenterHalfOp
from noeira.nn.core.element_op import ElementOp
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.module import Module
from noeira.nn.core.param import ParamVisitor
from noeira.nn.core.initializer import Initializer
from noeira.nn.core.walkers import join_name
from noeira.nn.core.amp import AMPPolicy, NoAMP
from noeira.nn.constants import DT, TPB, LAYOUT_NCHW


# ── stride-2 building blocks (k=4, s=2, p=1) ──────────────────────────────────
# Each conv is followed by channel-wise RMSNorm then activation (Conv→Norm→act),
# matching the DreamerV3 reference (`rssm.py`: norm over the channel axis after
# every conv). `ConvRMSNorm` assumes NCHW (γ size = out-channels) — keep LAYOUT
# at NCHW for these blocks.
# Down: even H,W → H/2, W/2.  OH = (H + 2 - 4)//2 + 1 = H//2 (even H).
comptime _ConvDown[
    IC: Int, OC: Int, H: Int, W: Int, A: ElementOp, LAYOUT: Int
] = Sequential[
    Conv2D[IC, OC, 4, 2, 1, H, W, DT, LAYOUT],
    ConvRMSNorm[OC, (H // 2) * (W // 2)],
    Elementwise[OC * (H // 2) * (W // 2), A],
]

# Up: H,W → 2H, 2W.  OHt = (H-1)*2 - 2 + 4 = 2H.
comptime _ConvUp[
    IC: Int, OC: Int, H: Int, W: Int, A: ElementOp, LAYOUT: Int
] = Sequential[
    Conv2DTranspose[IC, OC, 4, 2, 1, H, W, 0, LAYOUT],
    ConvRMSNorm[OC, (2 * H) * (2 * W)],
    Elementwise[OC * (2 * H) * (2 * W), A],
]


# ── Encoder: image[C·H·W] → 4× stride-2 conv → Linear → tokens[TOKEN] ─────────
# The leading `Elementwise[..., CenterHalfOp]` centers the [0,1] obs to
# [-0.5, 0.5] before the first conv — the DreamerV3 reference's `imgs/255 - 0.5`
# (our envs already emit [0,1], so it reduces to `- 0.5`). The decoder target
# stays in [0,1] (the reference centers only the encoder input).
comptime DreamerEncoderCNN[
    C: Int, H: Int, W: Int, BASE: Int, TOKEN: Int,
    A: ElementOp = GELUTanhOp, LAYOUT: Int = LAYOUT_NCHW,
] = Sequential[
    Elementwise[C * H * W, CenterHalfOp],                      # [0,1] → [-0.5,0.5]
    _ConvDown[C, BASE, H, W, A, LAYOUT],                       # H   → H/2
    _ConvDown[BASE, 2 * BASE, H // 2, W // 2, A, LAYOUT],      # H/2 → H/4
    _ConvDown[2 * BASE, 4 * BASE, H // 4, W // 4, A, LAYOUT],  # H/4 → H/8
    _ConvDown[4 * BASE, 8 * BASE, H // 8, W // 8, A, LAYOUT],  # H/8 → H/16
    Linear[8 * BASE * (H // 16) * (W // 16), TOKEN],
]


# ── Raw-token encoder (REFERENCE parity): tokens = the flattened final conv
# map, NO Linear bottleneck. The reference posterior consumes the raw conv
# features (`rssm.py Encoder`: `x.reshape((B, -1))` → tokens, ~9-16k dims);
# the Linear-to-TOKEN squeeze above is a noeira economy that STARVES the
# posterior — prime suspect in the slow WM maturation on pixel Pong (obs_loss
# still ~4.4 after 3k updates vs the reference solving Pong outright in that
# budget). Pass TOKEN = 8·BASE·(H/16)·(W/16) wherever the arch needs it.
comptime DreamerEncoderCNNRaw[
    C: Int, H: Int, W: Int, BASE: Int,
    A: ElementOp = GELUTanhOp, LAYOUT: Int = LAYOUT_NCHW,
] = Sequential[
    Elementwise[C * H * W, CenterHalfOp],                      # [0,1] → [-0.5,0.5]
    _ConvDown[C, BASE, H, W, A, LAYOUT],                       # H   → H/2
    _ConvDown[BASE, 2 * BASE, H // 2, W // 2, A, LAYOUT],      # H/2 → H/4
    _ConvDown[2 * BASE, 4 * BASE, H // 4, W // 4, A, LAYOUT],  # H/4 → H/8
    _ConvDown[4 * BASE, 8 * BASE, H // 8, W // 8, A, LAYOUT],  # H/8 → H/16
]


# ── Decoder: feature[FEATIN] → Linear → 4× transposed-conv → image[C·H·W] ─────
comptime DreamerDecoderCNN[
    FEATIN: Int, C: Int, H: Int, W: Int, BASE: Int,
    A: ElementOp = GELUTanhOp, LAYOUT: Int = LAYOUT_NCHW,
] = Sequential[
    Linear[FEATIN, 8 * BASE * (H // 16) * (W // 16)],
    ConvRMSNorm[8 * BASE, (H // 16) * (W // 16)],
    Elementwise[8 * BASE * (H // 16) * (W // 16), A],
    _ConvUp[8 * BASE, 4 * BASE, H // 16, W // 16, A, LAYOUT],  # H/16 → H/8
    _ConvUp[4 * BASE, 2 * BASE, H // 8, W // 8, A, LAYOUT],    # H/8  → H/4
    _ConvUp[2 * BASE, BASE, H // 4, W // 4, A, LAYOUT],        # H/4  → H/2
    Conv2DTranspose[BASE, C, 4, 2, 1, H // 2, W // 2, 0, LAYOUT],  # H/2 → H (raw)
]


# ═══════════════════════════════════════════════════════════════════════════
# REFERENCE-GEOMETRY (Phase B-2) encoder / decoder — `strided: False` port of
# `rssm.py` Encoder/Decoder: kernel-5 stride-1 convs, 2×2 max-pool downsample
# (encoder) / nearest-×2 upsample (decoder), channel schedule depth·[2,3,4,4],
# and the decoder's `bspace=8` two-branch input stem. Stage order matches the
# reference exactly: encoder conv → POOL → norm → act; decoder UP → conv →
# norm → act.
# ═══════════════════════════════════════════════════════════════════════════


# ── k5-s1 conv + 2×2 max-pool downsample stage (reference encoder stage) ──
comptime _ConvPoolDown[
    IC: Int, OC: Int, H: Int, W: Int, A: ElementOp, LAYOUT: Int
] = Sequential[
    Conv2D[IC, OC, 5, 1, 2, H, W, DT, LAYOUT],    # SAME: H → H
    MaxPool2D[OC, 2, 2, 0, H, W, LAYOUT],         # H → H/2
    ConvRMSNorm[OC, (H // 2) * (W // 2)],
    Elementwise[OC * (H // 2) * (W // 2), A],
]

# ── nearest-×2 upsample + k5-s1 conv stage (reference decoder stage) ──
comptime _UpConv[
    IC: Int, OC: Int, H: Int, W: Int, A: ElementOp, LAYOUT: Int
] = Sequential[
    Upsample2x[IC, H, W],                                  # H → 2H
    Conv2D[IC, OC, 5, 1, 2, 2 * H, 2 * W, DT, LAYOUT],     # SAME
    ConvRMSNorm[OC, (2 * H) * (2 * W)],
    Elementwise[OC * (2 * H) * (2 * W), A],
]


# ── Encoder (reference `strided: False`): channels BASE·{2,3,4,4}, raw
# flattened tokens (no bottleneck). TOKEN = 4·BASE·(H/16)·(W/16) — at BASE 64,
# IMG 96 that is 256·36 = 9216, the reference's exact token width. ──
comptime DreamerEncoderCNNPool[
    C: Int, H: Int, W: Int, BASE: Int,
    A: ElementOp = GELUTanhOp, LAYOUT: Int = LAYOUT_NCHW,
] = Sequential[
    Elementwise[C * H * W, CenterHalfOp],                          # [0,1] → [-0.5,0.5]
    _ConvPoolDown[C, 2 * BASE, H, W, A, LAYOUT],                   # H   → H/2
    _ConvPoolDown[2 * BASE, 3 * BASE, H // 2, W // 2, A, LAYOUT],  # H/2 → H/4
    _ConvPoolDown[3 * BASE, 4 * BASE, H // 4, W // 4, A, LAYOUT],  # H/4 → H/8
    _ConvPoolDown[4 * BASE, 4 * BASE, H // 8, W // 8, A, LAYOUT],  # H/8 → H/16
]


# ── bspace decoder stem GPU kernels (split / merge / add) ──────────────────
def _stem_split_k[
    B: Int, D: Int, S: Int
](
    feat: LayoutTensor[DT, Layout.row_major(B * (D + S)), MutAnyOrigin],
    xd: LayoutTensor[DT, Layout.row_major(B * D), MutAnyOrigin],
    xs: LayoutTensor[DT, Layout.row_major(B * S), MutAnyOrigin],
):
    """feat[b] = [deter | stoch] → xd[b], xs[b]. One thread per feat element."""
    var i = Int(global_idx.x)
    if i < B * (D + S):
        var b = i // (D + S)
        var k = i % (D + S)
        if k < D:
            xd[b * D + k] = feat[i]
        else:
            xs[b * S + (k - D)] = feat[i]


def _stem_merge_k[
    B: Int, D: Int, S: Int
](
    gxd: LayoutTensor[DT, Layout.row_major(B * D), MutAnyOrigin],
    gxs: LayoutTensor[DT, Layout.row_major(B * S), MutAnyOrigin],
    gfeat: LayoutTensor[DT, Layout.row_major(B * (D + S)), MutAnyOrigin],
):
    """Inverse of `_stem_split_k` (disjoint scatter — no accumulation)."""
    var i = Int(global_idx.x)
    if i < B * (D + S):
        var b = i // (D + S)
        var k = i % (D + S)
        if k < D:
            gfeat[i] = gxd[b * D + k]
        else:
            gfeat[i] = gxs[b * S + (k - D)]


def _stem_add_k[
    N: Int
](
    a: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    b: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
    dst: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin],
):
    var i = Int(global_idx.x)
    if i < N:
        dst[i] = a[i] + b[i]


struct DreamerDecoderStem[
    DETER: Int, SC: Int, UNITS: Int, U: Int,
    A: ElementOp = GELUTanhOp, BSPACE: Int = 8,
](Module):
    """The reference decoder's `bspace` input stem.

    (`rssm.py Decoder`, bspace=8): feat = [deter | stoch] →

        x0 = BlockLinear[DETER → U, groups BSPACE](deter)
        x1 = Linear[SC → 2·UNITS] → RMSNorm → act → Linear[→ U](stoch)
        out = x0 + x1                                  (norm+act follow OUTSIDE)

    In NCHW the reference's einops rearrange `(g h w c) -> h w (g c)` is a
    NO-OP: BlockLinear's block-major output [g][h·w·c] read as NCHW IS
    "block g fills channels [g·cs, (g+1)·cs)" — the same block-to-channel
    structure, zero permutation.

    vjp RECOMPUTES the forward chain from `forward_input` (the dreamer4
    bridge convention) so child caches (RMSNorm) are consistent."""

    comptime ARITY: Int = 1
    comptime IN_DIMS = Array[Int, 1](fill=Self.DETER + Self.SC)
    comptime OUT_DIM: Int = Self.U
    comptime U2 = 2 * Self.UNITS

    var bl: BlockLinear[Self.DETER, Self.U, Self.BSPACE]
    var l1: Linear[Self.SC, Self.U2]
    var n1: RMSNorm[Self.U2]
    var el: Elementwise[Self.U2, Self.A]
    var l2: Linear[Self.U2, Self.U]
    # forward/recompute scratch
    var xd: Tensor
    var xs: Tensor
    var h1: Tensor
    var h2: Tensor
    var h3: Tensor
    var x0: Tensor
    var x1: Tensor
    # backward scratch
    var g3: Tensor
    var g2: Tensor
    var g1: Tensor
    var gxd: Tensor
    var gxs: Tensor

    def __init__(out self):
        comptime assert Self.U % Self.BSPACE == 0, (
            "DreamerDecoderStem: U must be divisible by BSPACE"
        )
        comptime assert Self.DETER % Self.BSPACE == 0, (
            "DreamerDecoderStem: DETER must be divisible by BSPACE"
        )
        self.bl = BlockLinear[Self.DETER, Self.U, Self.BSPACE]()
        self.l1 = Linear[Self.SC, Self.U2]()
        self.n1 = RMSNorm[Self.U2]()
        self.el = Elementwise[Self.U2, Self.A]()
        self.l2 = Linear[Self.U2, Self.U]()
        self.xd = Tensor()
        self.xs = Tensor()
        self.h1 = Tensor()
        self.h2 = Tensor()
        self.h3 = Tensor()
        self.x0 = Tensor()
        self.x1 = Tensor()
        self.g3 = Tensor()
        self.g2 = Tensor()
        self.g1 = Tensor()
        self.gxd = Tensor()
        self.gxs = Tensor()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        var s = Self()
        s.bl = BlockLinear[Self.DETER, Self.U, Self.BSPACE].make[target, INIT](
            ctx
        )
        s.l1 = Linear[Self.SC, Self.U2].make[target, INIT](ctx)
        s.n1 = RMSNorm[Self.U2].make[target, INIT](ctx)
        s.el = Elementwise[Self.U2, Self.A].make[target, INIT](ctx)
        s.l2 = Linear[Self.U2, Self.U].make[target, INIT](ctx)
        return s^

    def _split[
        target: StaticString, B: Int, o: MutOrigin
    ](
        mut self, inputs: TensorRefs[1, o], ctx: Optional[DeviceContext]
    ) raises:
        """feat → (xd, xs) member scratch."""
        ref feat = inputs[0]
        comptime F = Self.DETER + Self.SC
        comptime if target == "cpu":
            self.xd.ensure(B * Self.DETER)
            self.xs.ensure(B * Self.SC)
            for b in range(B):
                for k in range(Self.DETER):
                    self.xd.data[b * Self.DETER + k] = feat.data[b * F + k]
                for k in range(Self.SC):
                    self.xs.data[b * Self.SC + k] = feat.data[
                        b * F + Self.DETER + k
                    ]
        else:
            var c = ctx.value()
            self.xd.ensure_gpu(c, B * Self.DETER)
            self.xs.ensure_gpu(c, B * Self.SC)
            comptime nb = (B * F + TPB - 1) // TPB
            c.enqueue_function[_stem_split_k[B, Self.DETER, Self.SC]](
                feat.lt["gpu", Layout.row_major(B * F)](),
                self.xd.lt["gpu", Layout.row_major(B * Self.DETER)](),
                self.xs.lt["gpu", Layout.row_major(B * Self.SC)](),
                grid_dim=nb,
                block_dim=TPB,
            )

    def _run_branches[
        target: StaticString, B: Int, POLICY: AMPPolicy
    ](mut self, ctx: Optional[DeviceContext]) raises:
        """(xd, xs) → x0, h1..h3, x1 member scratch."""
        self.bl.forward[target, B, POLICY=POLICY](
            TensorRefs[1](self.xd), self.x0, ctx
        )
        self.l1.forward[target, B, POLICY=POLICY](
            TensorRefs[1](self.xs), self.h1, ctx
        )
        self.n1.forward[target, B, POLICY=POLICY](
            TensorRefs[1](self.h1), self.h2, ctx
        )
        self.el.forward[target, B, POLICY=POLICY](
            TensorRefs[1](self.h2), self.h3, ctx
        )
        self.l2.forward[target, B, POLICY=POLICY](
            TensorRefs[1](self.h3), self.x1, ctx
        )

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[1, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        self._split[target, B](inputs, ctx)
        self._run_branches[target, B, POLICY](ctx)
        comptime if target == "cpu":
            out.ensure(B * Self.U)
            for i in range(B * Self.U):
                out.data[i] = self.x0.data[i] + self.x1.data[i]
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B * Self.U)
            comptime nb = (B * Self.U + TPB - 1) // TPB
            c.enqueue_function[_stem_add_k[B * Self.U]](
                self.x0.lt["gpu", Layout.row_major(B * Self.U)](),
                self.x1.lt["gpu", Layout.row_major(B * Self.U)](),
                out.lt["gpu", Layout.row_major(B * Self.U)](),
                grid_dim=nb,
                block_dim=TPB,
            )

    def vjp[
        target: StaticString,
        B: Int,
        ofi: MutOrigin,
        ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[1, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[1, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        # Recompute the forward chain (refreshes child caches consistently).
        self._split[target, B](forward_input, ctx)
        self._run_branches[target, B, POLICY](ctx)
        # out = x0 + x1  →  d_x0 = d_x1 = grad_output.
        self.l2.vjp[target, B, POLICY=POLICY](
            TensorRefs[1](self.h3), grad_output, TensorRefs[1](self.g3), ctx
        )
        self.el.vjp[target, B, POLICY=POLICY](
            TensorRefs[1](self.h2), self.g3, TensorRefs[1](self.g2), ctx
        )
        self.n1.vjp[target, B, POLICY=POLICY](
            TensorRefs[1](self.h1), self.g2, TensorRefs[1](self.g1), ctx
        )
        self.l1.vjp[target, B, POLICY=POLICY](
            TensorRefs[1](self.xs), self.g1, TensorRefs[1](self.gxs), ctx
        )
        self.bl.vjp[target, B, POLICY=POLICY](
            TensorRefs[1](self.xd), grad_output, TensorRefs[1](self.gxd), ctx
        )
        # Merge (gxd | gxs) → grad_inputs[0] (disjoint scatter).
        ref gfeat = grad_inputs[0]
        comptime F = Self.DETER + Self.SC
        comptime if target == "cpu":
            gfeat.ensure(B * F)
            for b in range(B):
                for k in range(Self.DETER):
                    gfeat.data[b * F + k] = self.gxd.data[b * Self.DETER + k]
                for k in range(Self.SC):
                    gfeat.data[b * F + Self.DETER + k] = self.gxs.data[
                        b * Self.SC + k
                    ]
        else:
            var c = ctx.value()
            gfeat.ensure_gpu(c, B * F)
            comptime nb = (B * F + TPB - 1) // TPB
            c.enqueue_function[_stem_merge_k[B, Self.DETER, Self.SC]](
                self.gxd.lt["gpu", Layout.row_major(B * Self.DETER)](),
                self.gxs.lt["gpu", Layout.row_major(B * Self.SC)](),
                gfeat.lt["gpu", Layout.row_major(B * F)](),
                grid_dim=nb,
                block_dim=TPB,
            )

    def for_each_param[
        target: StaticString, V: ParamVisitor
    ](mut self, mut visitor: V, ctx: Optional[DeviceContext],
      prefix: String = String("")) raises:
        self.bl.for_each_param[target](visitor, ctx, join_name(prefix, "bl"))
        self.l1.for_each_param[target](visitor, ctx, join_name(prefix, "l1"))
        self.n1.for_each_param[target](visitor, ctx, join_name(prefix, "n1"))
        self.l2.for_each_param[target](visitor, ctx, join_name(prefix, "l2"))

    def for_each_state[
        target: StaticString, V: ParamVisitor
    ](mut self, mut visitor: V, ctx: Optional[DeviceContext],
      prefix: String = String("")) raises:
        self.bl.for_each_state[target](visitor, ctx, join_name(prefix, "bl"))
        self.l1.for_each_state[target](visitor, ctx, join_name(prefix, "l1"))
        self.n1.for_each_state[target](visitor, ctx, join_name(prefix, "n1"))
        self.l2.for_each_state[target](visitor, ctx, join_name(prefix, "l2"))

    def zero_grad[
        target: StaticString
    ](mut self, ctx: Optional[DeviceContext]) raises:
        self.bl.zero_grad[target](ctx)
        self.l1.zero_grad[target](ctx)
        self.n1.zero_grad[target](ctx)
        self.l2.zero_grad[target](ctx)

    def polyak_from[
        target: StaticString
    ](
        mut self, mut src: Self, tau: Scalar[DT], ctx: Optional[DeviceContext]
    ) raises:
        self.bl.polyak_from[target](src.bl, tau, ctx)
        self.l1.polyak_from[target](src.l1, tau, ctx)
        self.n1.polyak_from[target](src.n1, tau, ctx)
        self.l2.polyak_from[target](src.l2, tau, ctx)


# ── Decoder (reference `strided: False`, bspace=8): stem → norm → act →
# 3× [up2 → conv k5 → norm → act] (channels 4B→4B→3B→2B) → up2 → conv k5 → C
# (raw pixel logits; the trainer's RECON_SIGMOID applies the sigmoid). ──
comptime DreamerDecoderCNNPool[
    FEATIN: Int, DETER: Int, C: Int, H: Int, W: Int, BASE: Int, UNITS: Int,
    A: ElementOp = GELUTanhOp, LAYOUT: Int = LAYOUT_NCHW,
] = Sequential[
    DreamerDecoderStem[
        DETER, FEATIN - DETER, UNITS, 4 * BASE * (H // 16) * (W // 16), A
    ],
    ConvRMSNorm[4 * BASE, (H // 16) * (W // 16)],
    Elementwise[4 * BASE * (H // 16) * (W // 16), A],
    _UpConv[4 * BASE, 4 * BASE, H // 16, W // 16, A, LAYOUT],  # H/16 → H/8
    _UpConv[4 * BASE, 3 * BASE, H // 8, W // 8, A, LAYOUT],    # H/8  → H/4
    _UpConv[3 * BASE, 2 * BASE, H // 4, W // 4, A, LAYOUT],    # H/4  → H/2
    Upsample2x[2 * BASE, H // 2, W // 2],                      # H/2  → H
    Conv2D[2 * BASE, C, 5, 1, 2, H, W, DT, LAYOUT],            # imgout (raw)
]
