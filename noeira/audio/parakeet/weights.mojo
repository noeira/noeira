# +--------------------------------------------------------------------------+ #
# | Parakeet-TDT — the weights, in `nn` modules
# +--------------------------------------------------------------------------+ #
"""`ParakeetTDT` — every parameter of `nvidia/parakeet-tdt-0.6b-v3`, held in `nn`
leaves (`Linear`, `LayerNorm`, `BatchNorm1D`, `Embedding`, `Param`) and walked
under names of our own (`names.mojo` maps them to the checkpoint's).

Weights and not a composed `Module`, for SmolVLA's reason (`smolvla/text.mojo`):
the forward needs to reach between a block's pieces — the relative-position
term enters the attention logits, the padding mask enters three times — so the
loop that drives these lives in `encoder.mojo` and `decode.mojo`.

What each published tensor becomes:

- A `nn.Linear` is a `Linear`. So is a KERNEL-1 convolution (the conformer's two
  pointwise `Conv1d`s, the subsampling's two pointwise `Conv2d`s): the same
  matrix, applied per position.
- A convolution in which every output channel reads ONE input channel — the
  first subsampling conv (one input channel) and every depthwise conv — is a
  `PkConv[C, TAPS]`: `[C, TAPS]` weights and a `[C]` bias.
- An LSTM layer is two `Linear`s, `ih` and `hh`: torch's gates are exactly
  `W_ih x + b_ih + W_hh h + b_hh`, with the gates stacked `i, f, g, o`.
- `BatchNorm1d` is a `BatchNorm1D` used as a container: inference reads its
  `gamma` / `beta` and the two running statistics (State, walked separately).

⚠ `Linear` always has a bias and most of this model's linears have none (every
attention, feed-forward and conformer conv weight). Those biases are `TN_ZEROS`
in the name map: filled with zeros on load, and that is the function the
checkpoint computes.

`make` FREEZES by default (no gradient buffers): the model is 627 M parameters,
2.5 GB in fp32, and a gradient per parameter would double that on a 16 GB
laptop. Fine-tuning unfreezes what it trains.
"""

from max.gpu.host import DeviceContext

from noeira.nn.core.initializer import Zero
from noeira.nn.core.param import Param, ParamVisitor, ParamWalkable
from noeira.nn.core.walkers import join_name, for_each_param_auto, zero_grad_auto
from noeira.nn.core.torch_names import LoadTorchNamed
from noeira.nn.peft.freeze import FreezeParams
from noeira.nn.primitives.linear import Linear
from noeira.nn.primitives.layer_norm import LayerNorm
from noeira.nn.primitives.batch_norm_1d import BatchNorm1D
from noeira.nn.primitives.embedding import Embedding
from noeira.io.safetensors import SafeTensors
from .config import (
    PK_D, PK_FF, PK_LAYERS, PK_CONV_K, PK_SUB_CH, PK_SUB_FLAT, PK_DEC,
    PK_DEC_LAYERS, PK_VOCAB, PK_JOINT, PK_FILE_TENSORS, PK_FILE_WEIGHTS,
)
from .names import parakeet_name_map


struct PkConv[C: Int, TAPS: Int](Movable):
    """A convolution whose every output channel reads one input channel:
    `weight` is `[C, TAPS]` (torch's `[C, 1, K]` or `[C, 1, KH, KW]`, same
    order), `bias` is `[C]`."""

    var weight: Param["weight", True, Self.C * Self.TAPS]
    var bias: Param["bias", False, Self.C]

    def __init__(out self):
        self.weight = Param["weight", True, Self.C * Self.TAPS]()
        self.bias = Param["bias", False, Self.C]()

    @staticmethod
    def make[target: StaticString](
        ctx: Optional[DeviceContext] = None
    ) raises -> Self:
        var c = Self()
        c.weight = Param["weight", True, Self.C * Self.TAPS].make[target](ctx)
        c.bias = Param["bias", False, Self.C].make[target](ctx)
        return c^

    def walk[target: StaticString, V: ParamVisitor](
        mut self, mut v: V, ctx: Optional[DeviceContext], prefix: String
    ) raises:
        for_each_param_auto[Self, V, target](self, v, ctx, prefix)

    def zero_grad[target: StaticString](
        mut self, ctx: Optional[DeviceContext]
    ) raises:
        zero_grad_auto[Self, target](self, ctx)


struct PkSubsampling(Movable):
    """`mel [F, 128]` -> `[T, 1024]`, T = F / 8: a 3x3 stride-2 conv to 256
    channels, then twice (depthwise 3x3 stride-2, pointwise), then a linear
    over the 256 x 16 values left per frame."""

    var conv0: PkConv[PK_SUB_CH, 9]
    var dw1: PkConv[PK_SUB_CH, 9]
    var pw1: Linear[PK_SUB_CH, PK_SUB_CH]
    var dw2: PkConv[PK_SUB_CH, 9]
    var pw2: Linear[PK_SUB_CH, PK_SUB_CH]
    var linear: Linear[PK_SUB_FLAT, PK_D]

    def __init__(out self):
        self.conv0 = PkConv[PK_SUB_CH, 9]()
        self.dw1 = PkConv[PK_SUB_CH, 9]()
        self.pw1 = Linear[PK_SUB_CH, PK_SUB_CH]()
        self.dw2 = PkConv[PK_SUB_CH, 9]()
        self.pw2 = Linear[PK_SUB_CH, PK_SUB_CH]()
        self.linear = Linear[PK_SUB_FLAT, PK_D]()

    @staticmethod
    def make[target: StaticString](
        ctx: Optional[DeviceContext] = None
    ) raises -> Self:
        var s = Self()
        s.conv0 = PkConv[PK_SUB_CH, 9].make[target](ctx)
        s.dw1 = PkConv[PK_SUB_CH, 9].make[target](ctx)
        s.pw1 = Linear[PK_SUB_CH, PK_SUB_CH].make[target, Zero](ctx)
        s.dw2 = PkConv[PK_SUB_CH, 9].make[target](ctx)
        s.pw2 = Linear[PK_SUB_CH, PK_SUB_CH].make[target, Zero](ctx)
        s.linear = Linear[PK_SUB_FLAT, PK_D].make[target, Zero](ctx)
        return s^

    def walk[target: StaticString, V: ParamVisitor](
        mut self, mut v: V, ctx: Optional[DeviceContext], prefix: String
    ) raises:
        self.conv0.walk[target](v, ctx, join_name(prefix, String("conv0")))
        self.dw1.walk[target](v, ctx, join_name(prefix, String("dw1")))
        self.pw1.for_each_param[target](v, ctx, join_name(prefix, String("pw1")))
        self.dw2.walk[target](v, ctx, join_name(prefix, String("dw2")))
        self.pw2.for_each_param[target](v, ctx, join_name(prefix, String("pw2")))
        self.linear.for_each_param[target](
            v, ctx, join_name(prefix, String("linear"))
        )

    def zero_grad[target: StaticString](
        mut self, ctx: Optional[DeviceContext]
    ) raises:
        """⚠ The same six children as `walk`, in the same order."""
        self.conv0.zero_grad[target](ctx)
        self.dw1.zero_grad[target](ctx)
        self.pw1.zero_grad[target](ctx)
        self.dw2.zero_grad[target](ctx)
        self.pw2.zero_grad[target](ctx)
        self.linear.zero_grad[target](ctx)


struct PkConformerLayer(Movable):
    """One FastConformer block:

        x += 0.5 * ff1(norm_ff1(x))
        x += attention(norm_att(x))            relative positions, `bias_u` / `bias_v`
        x += conv(norm_conv(x))                pw1 -> GLU -> dw (k 9) -> bn -> silu -> pw2
        x += 0.5 * ff2(norm_ff2(x))
        x  = norm_out(x)
    """

    var norm_ff1: LayerNorm[PK_D]
    var ff1_a: Linear[PK_D, PK_FF]
    var ff1_b: Linear[PK_FF, PK_D]
    var norm_att: LayerNorm[PK_D]
    var q: Linear[PK_D, PK_D]
    var k: Linear[PK_D, PK_D]
    var v: Linear[PK_D, PK_D]
    var o: Linear[PK_D, PK_D]
    var rel_k: Linear[PK_D, PK_D]
    var bias_u: Param["bias_u", False, PK_D]
    """`[HEADS, HEAD_DIM]`: added to the query for the content term."""
    var bias_v: Param["bias_v", False, PK_D]
    """`[HEADS, HEAD_DIM]`: added to the query for the position term."""
    var norm_conv: LayerNorm[PK_D]
    var pw1: Linear[PK_D, 2 * PK_D]
    var dw: PkConv[PK_D, PK_CONV_K]
    var bn: BatchNorm1D[PK_D]
    var pw2: Linear[PK_D, PK_D]
    var norm_ff2: LayerNorm[PK_D]
    var ff2_a: Linear[PK_D, PK_FF]
    var ff2_b: Linear[PK_FF, PK_D]
    var norm_out: LayerNorm[PK_D]

    def __init__(out self):
        self.norm_ff1 = LayerNorm[PK_D]()
        self.ff1_a = Linear[PK_D, PK_FF]()
        self.ff1_b = Linear[PK_FF, PK_D]()
        self.norm_att = LayerNorm[PK_D]()
        self.q = Linear[PK_D, PK_D]()
        self.k = Linear[PK_D, PK_D]()
        self.v = Linear[PK_D, PK_D]()
        self.o = Linear[PK_D, PK_D]()
        self.rel_k = Linear[PK_D, PK_D]()
        self.bias_u = Param["bias_u", False, PK_D]()
        self.bias_v = Param["bias_v", False, PK_D]()
        self.norm_conv = LayerNorm[PK_D]()
        self.pw1 = Linear[PK_D, 2 * PK_D]()
        self.dw = PkConv[PK_D, PK_CONV_K]()
        self.bn = BatchNorm1D[PK_D]()
        self.pw2 = Linear[PK_D, PK_D]()
        self.norm_ff2 = LayerNorm[PK_D]()
        self.ff2_a = Linear[PK_D, PK_FF]()
        self.ff2_b = Linear[PK_FF, PK_D]()
        self.norm_out = LayerNorm[PK_D]()

    @staticmethod
    def make[target: StaticString](
        ctx: Optional[DeviceContext] = None
    ) raises -> Self:
        var l = Self()
        l.norm_ff1 = LayerNorm[PK_D].make[target, Zero](ctx)
        l.ff1_a = Linear[PK_D, PK_FF].make[target, Zero](ctx)
        l.ff1_b = Linear[PK_FF, PK_D].make[target, Zero](ctx)
        l.norm_att = LayerNorm[PK_D].make[target, Zero](ctx)
        l.q = Linear[PK_D, PK_D].make[target, Zero](ctx)
        l.k = Linear[PK_D, PK_D].make[target, Zero](ctx)
        l.v = Linear[PK_D, PK_D].make[target, Zero](ctx)
        l.o = Linear[PK_D, PK_D].make[target, Zero](ctx)
        l.rel_k = Linear[PK_D, PK_D].make[target, Zero](ctx)
        l.bias_u = Param["bias_u", False, PK_D].make[target](ctx)
        l.bias_v = Param["bias_v", False, PK_D].make[target](ctx)
        l.norm_conv = LayerNorm[PK_D].make[target, Zero](ctx)
        l.pw1 = Linear[PK_D, 2 * PK_D].make[target, Zero](ctx)
        l.dw = PkConv[PK_D, PK_CONV_K].make[target](ctx)
        l.bn = BatchNorm1D[PK_D].make[target, Zero](ctx)
        l.pw2 = Linear[PK_D, PK_D].make[target, Zero](ctx)
        l.norm_ff2 = LayerNorm[PK_D].make[target, Zero](ctx)
        l.ff2_a = Linear[PK_D, PK_FF].make[target, Zero](ctx)
        l.ff2_b = Linear[PK_FF, PK_D].make[target, Zero](ctx)
        l.norm_out = LayerNorm[PK_D].make[target, Zero](ctx)
        return l^

    def walk[target: StaticString, V: ParamVisitor](
        mut self, mut v: V, ctx: Optional[DeviceContext], prefix: String
    ) raises:
        self.norm_ff1.for_each_param[target](v, ctx, join_name(prefix, String("norm_ff1")))
        self.ff1_a.for_each_param[target](v, ctx, join_name(prefix, String("ff1_a")))
        self.ff1_b.for_each_param[target](v, ctx, join_name(prefix, String("ff1_b")))
        self.norm_att.for_each_param[target](v, ctx, join_name(prefix, String("norm_att")))
        self.q.for_each_param[target](v, ctx, join_name(prefix, String("q")))
        self.k.for_each_param[target](v, ctx, join_name(prefix, String("k")))
        self.v.for_each_param[target](v, ctx, join_name(prefix, String("v")))
        self.o.for_each_param[target](v, ctx, join_name(prefix, String("o")))
        self.rel_k.for_each_param[target](v, ctx, join_name(prefix, String("rel_k")))
        self.bias_u.visit_with[target, V](v, ctx, join_name(prefix, String("bias_u")))
        self.bias_v.visit_with[target, V](v, ctx, join_name(prefix, String("bias_v")))
        self.norm_conv.for_each_param[target](v, ctx, join_name(prefix, String("norm_conv")))
        self.pw1.for_each_param[target](v, ctx, join_name(prefix, String("pw1")))
        self.dw.walk[target](v, ctx, join_name(prefix, String("dw")))
        self.bn.for_each_param[target](v, ctx, join_name(prefix, String("bn")))
        self.pw2.for_each_param[target](v, ctx, join_name(prefix, String("pw2")))
        self.norm_ff2.for_each_param[target](v, ctx, join_name(prefix, String("norm_ff2")))
        self.ff2_a.for_each_param[target](v, ctx, join_name(prefix, String("ff2_a")))
        self.ff2_b.for_each_param[target](v, ctx, join_name(prefix, String("ff2_b")))
        self.norm_out.for_each_param[target](v, ctx, join_name(prefix, String("norm_out")))

    def walk_state[target: StaticString, V: ParamVisitor](
        mut self, mut v: V, ctx: Optional[DeviceContext], prefix: String
    ) raises:
        """The BatchNorm running statistics: the block's only State."""
        self.bn.for_each_state[target](v, ctx, join_name(prefix, String("bn")))

    def zero_grad[target: StaticString](
        mut self, ctx: Optional[DeviceContext]
    ) raises:
        """⚠ The same twenty children as `walk`, in the same order."""
        self.norm_ff1.zero_grad[target](ctx)
        self.ff1_a.zero_grad[target](ctx)
        self.ff1_b.zero_grad[target](ctx)
        self.norm_att.zero_grad[target](ctx)
        self.q.zero_grad[target](ctx)
        self.k.zero_grad[target](ctx)
        self.v.zero_grad[target](ctx)
        self.o.zero_grad[target](ctx)
        self.rel_k.zero_grad[target](ctx)
        self.bias_u.zero_grad[target](ctx)
        self.bias_v.zero_grad[target](ctx)
        self.norm_conv.zero_grad[target](ctx)
        self.pw1.zero_grad[target](ctx)
        self.dw.zero_grad[target](ctx)
        self.bn.zero_grad[target](ctx)
        self.pw2.zero_grad[target](ctx)
        self.norm_ff2.zero_grad[target](ctx)
        self.ff2_a.zero_grad[target](ctx)
        self.ff2_b.zero_grad[target](ctx)
        self.norm_out.zero_grad[target](ctx)


struct PkLSTMLayer(Movable):
    """One LSTM layer as torch stores it: `gates = ih(x) + hh(h)`, the four
    gates stacked `i, f, g, o` along the 4 x 640 output."""

    var ih: Linear[PK_DEC, 4 * PK_DEC]
    var hh: Linear[PK_DEC, 4 * PK_DEC]

    def __init__(out self):
        self.ih = Linear[PK_DEC, 4 * PK_DEC]()
        self.hh = Linear[PK_DEC, 4 * PK_DEC]()

    @staticmethod
    def make[target: StaticString](
        ctx: Optional[DeviceContext] = None
    ) raises -> Self:
        var l = Self()
        l.ih = Linear[PK_DEC, 4 * PK_DEC].make[target, Zero](ctx)
        l.hh = Linear[PK_DEC, 4 * PK_DEC].make[target, Zero](ctx)
        return l^

    def walk[target: StaticString, V: ParamVisitor](
        mut self, mut v: V, ctx: Optional[DeviceContext], prefix: String
    ) raises:
        self.ih.for_each_param[target](v, ctx, join_name(prefix, String("ih")))
        self.hh.for_each_param[target](v, ctx, join_name(prefix, String("hh")))

    def zero_grad[target: StaticString](
        mut self, ctx: Optional[DeviceContext]
    ) raises:
        self.ih.zero_grad[target](ctx)
        self.hh.zero_grad[target](ctx)


struct ParakeetTDT(ParamWalkable):
    """The whole model's parameters. `ParakeetTDT.make["cpu"]()` then
    `load["cpu"](path)`; the forward is in `encoder.mojo` / `decode.mojo`."""

    var sub: PkSubsampling
    var layers: List[PkConformerLayer]
    var enc_proj: Linear[PK_D, PK_DEC]
    var embedding: Embedding[PK_VOCAB, PK_DEC]
    var lstm: List[PkLSTMLayer]
    var dec_proj: Linear[PK_DEC, PK_DEC]
    var joint: Linear[PK_DEC, PK_JOINT]

    def __init__(out self):
        self.sub = PkSubsampling()
        self.layers = List[PkConformerLayer]()
        self.enc_proj = Linear[PK_D, PK_DEC]()
        self.embedding = Embedding[PK_VOCAB, PK_DEC]()
        self.lstm = List[PkLSTMLayer]()
        self.dec_proj = Linear[PK_DEC, PK_DEC]()
        self.joint = Linear[PK_DEC, PK_JOINT]()

    @staticmethod
    def make[target: StaticString, FROZEN: Bool = True](
        ctx: Optional[DeviceContext] = None
    ) raises -> Self:
        """Allocate every parameter (zeros; `load` fills them). `FROZEN` frees
        each piece's gradient buffers as soon as the piece exists, so the peak
        is the values plus ONE block's gradients, not twice the model."""
        var m = Self()
        var f = FreezeParams()
        m.sub = PkSubsampling.make[target](ctx)
        comptime if FROZEN:
            m.sub.walk[target](f, ctx, String("sub"))
        for i in range(PK_LAYERS):
            m.layers.append(PkConformerLayer.make[target](ctx))
            comptime if FROZEN:
                m.layers[i].walk[target](f, ctx, String("layers"))
        m.enc_proj = Linear[PK_D, PK_DEC].make[target, Zero](ctx)
        m.embedding = Embedding[PK_VOCAB, PK_DEC].make[target, Zero](ctx)
        for _ in range(PK_DEC_LAYERS):
            m.lstm.append(PkLSTMLayer.make[target](ctx))
        m.dec_proj = Linear[PK_DEC, PK_DEC].make[target, Zero](ctx)
        m.joint = Linear[PK_DEC, PK_JOINT].make[target, Zero](ctx)
        comptime if FROZEN:
            m.for_each_param[target](f, ctx)
        return m^

    def for_each_param[target: StaticString, V: ParamVisitor](
        mut self,
        mut visitor: V,
        ctx: Optional[DeviceContext],
        prefix: String = String(""),
    ) raises:
        self.sub.walk[target](visitor, ctx, join_name(prefix, String("sub")))
        for i in range(len(self.layers)):
            self.layers[i].walk[target](
                visitor, ctx, join_name(prefix, String("layers." + String(i)))
            )
        self.enc_proj.for_each_param[target](
            visitor, ctx, join_name(prefix, String("enc_proj"))
        )
        self.embedding.for_each_param[target](
            visitor, ctx, join_name(prefix, String("embedding"))
        )
        for i in range(len(self.lstm)):
            self.lstm[i].walk[target](
                visitor, ctx, join_name(prefix, String("lstm." + String(i)))
            )
        self.dec_proj.for_each_param[target](
            visitor, ctx, join_name(prefix, String("dec_proj"))
        )
        self.joint.for_each_param[target](
            visitor, ctx, join_name(prefix, String("joint"))
        )

    def for_each_state[target: StaticString, V: ParamVisitor](
        mut self,
        mut visitor: V,
        ctx: Optional[DeviceContext],
        prefix: String = String(""),
    ) raises:
        for i in range(len(self.layers)):
            self.layers[i].walk_state[target](
                visitor, ctx, join_name(prefix, String("layers." + String(i)))
            )

    def zero_grad[target: StaticString](
        mut self, ctx: Optional[DeviceContext]
    ) raises:
        """⚠ The same children as `for_each_param`, in the same order."""
        self.sub.zero_grad[target](ctx)
        for i in range(len(self.layers)):
            self.layers[i].zero_grad[target](ctx)
        self.enc_proj.zero_grad[target](ctx)
        self.embedding.zero_grad[target](ctx)
        for i in range(len(self.lstm)):
            self.lstm[i].zero_grad[target](ctx)
        self.dec_proj.zero_grad[target](ctx)
        self.joint.zero_grad[target](ctx)

    def load[target: StaticString](
        mut self, path: String, ctx: Optional[DeviceContext] = None
    ) raises:
        """Fill every parameter and running statistic from the published
        `model.safetensors`, and check the accounting in BOTH directions:

        - every parameter of ours has a map entry and every entry was claimed
          (a walk and a map that drifted apart leave weights at zero);
        - every fp32 tensor of the file was read: the only tensors left
          unclaimed are the 24 `num_batches_tracked` counters.
        """
        var ld = LoadTorchNamed[""](SafeTensors(path), parakeet_name_map())
        self.for_each_param[target](ld, ctx)
        self.for_each_state[target](ld, ctx)
        ld.report(String("parakeet"))
        if len(ld.loaded) + len(ld.zeroed) != ld.map.size():
            raise Error(
                "parakeet: filled " + String(len(ld.loaded) + len(ld.zeroed))
                + " of the map's " + String(ld.map.size()) + " entries — the"
                " walk never emitted the others, so those weights stay zero"
            )
        if len(ld.file.names) != PK_FILE_TENSORS or len(ld.loaded) != PK_FILE_WEIGHTS:
            raise Error(
                "parakeet: read " + String(len(ld.loaded)) + " tensors of a file"
                " of " + String(len(ld.file.names)) + ", expected "
                + String(PK_FILE_WEIGHTS) + " of " + String(PK_FILE_TENSORS)
                + " — not the checkpoint this map was written against"
            )
        var left = ld.unclaimed()
        for i in range(len(left)):
            if not left[i].endswith("num_batches_tracked"):
                raise Error(
                    "parakeet: the file's '" + left[i] + "' is named by no map"
                    " entry — that published weight would never be read"
                )
        if len(left) != PK_LAYERS:
            raise Error(
                "parakeet: " + String(len(left)) + " unread tensors, expected the "
                + String(PK_LAYERS) + " num_batches_tracked counters"
            )
