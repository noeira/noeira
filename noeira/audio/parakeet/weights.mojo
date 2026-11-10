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

`make` allocates NOTHING: it builds the tree of leaves, and `load` allocates each
parameter as it fills it. A parameter with no gradient buffer is frozen
(`Param.is_frozen`), so a loaded model is frozen by construction — 627 M
parameters are 2.5 GB in fp32 and a gradient each would double that on a 16 GB
laptop. Fine-tuning unfreezes what it trains.

## On a device

`load["gpu"]` puts every parameter on the device: the encoder (subsampling, the
24 blocks, the encoder projection: 99 % of the weights) and the prediction
network and joint the greedy loop runs one token at a time (`decode.mojo`).
Two things about how, both measured:

- **One arena per block, not one buffer per parameter.** On Metal every kernel
  launch costs ~22 us plus ~0.22 us for EACH live device buffer in the process
  (MAX marks every tracked buffer as used, per launch). With a buffer per
  parameter the encoder has ~1,200 of them and every launch costs ~190 us —
  the whole encode was launch overhead (162 ms of it for 912 launches). A
  sub-buffer of an arena is free: 1 arena + 1,200 sub-buffers measures 21.7 us.
  So each block's parameters and running statistics are slices of one device
  buffer (`arenas`; one more for the subsampling and projection, one for the
  decoder: 26), each slice starting on a multiple of `PK_ARENA_ALIGN`
  elements — MAX's GEMM loads its operands with 16-byte vectors and faults on a
  slice that starts off that boundary.
- **No host copies.** A parameter uploaded the ordinary way exists three times:
  the host list, the pinned staging buffer `Tensor` keeps, and the device
  buffer — 7.5 GB here, on a machine whose host and device share 16 GB. Each
  parameter's two host copies are freed as soon as its values have landed
  (after a synchronize), so the peak is the device copy plus one tensor. The
  consequence: a device-loaded encoder cannot be read, saved or edited from
  the host. Load on the CPU for that.

## Half-precision weights (`load["gpu", "fp16"]`, NVIDIA)

On an Orin the encoder's GEMM time is the STREAMING of its weights — 2.4 GB of
fp32 per clip, 35-37 ms whatever the row count — so the lever is their size.
With `WEIGHTS = "fp16"` the eleven GEMM weights of each block (the two
feed-forwards, the five attention projections, the two pointwise convolutions:
96 % of the parameters) are converted to float16 on the host and stored ONLY
in that form, in a float16 arena per block (`PkConformerLayer.half`, one
buffer per `PK_GEMM_*` slot). Everything else — norms, biases, the depthwise
convolution, the subsampling, the decoder — stays fp32, and so do the
activations: the GEMM's other operand is cast per call (`encoder.mojo`).
The `Linear` leaves of those eleven weights are then EMPTY (no host copy, no
device buffer): this is a deployment form, not one to fine-tune or save.
"""

from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from max.gpu.host import DeviceBuffer
from noeira.nn.core.param import Param, ParamVisitor, ParamVisitorRT, ParamWalkable
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.walkers import join_name, for_each_param_auto, zero_grad_auto
from noeira.nn.core.torch_names import LoadTorchNamed
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


def _drop_host(mut t: Tensor):
    """Free a tensor's host list and pinned staging buffer; the device buffer
    and the logical length stay. ⚠ Only after a synchronize: the staging buffer
    is the source of an upload that may still be in flight."""
    t.hbuf = None
    t.hcap = 0
    t._h2d_done = None
    t.data = List[Scalar[DT]]()


comptime PK_HALF = DType.float16

# The eleven GEMM weights of a block, in the order `PkConformerLayer.walk`
# visits them — which is the order their half-precision copies are stored in.
comptime PK_GEMM_FF1_A: Int = 0
comptime PK_GEMM_FF1_B: Int = 1
comptime PK_GEMM_Q: Int = 2
comptime PK_GEMM_K: Int = 3
comptime PK_GEMM_V: Int = 4
comptime PK_GEMM_O: Int = 5
comptime PK_GEMM_REL_K: Int = 6
comptime PK_GEMM_PW1: Int = 7
comptime PK_GEMM_PW2: Int = 8
comptime PK_GEMM_FF2_A: Int = 9
comptime PK_GEMM_FF2_B: Int = 10
comptime PK_GEMM_SLOTS: Int = 11


def pk_gemm_slot(name: String) -> Int:
    """The `PK_GEMM_*` slot of a walked parameter name, or -1 when it is not
    one of a block's GEMM weights. ⚠ The slot order must be the walk's:
    `PkLoad` raises when a weight arrives out of it."""
    if not name.startswith("layers."):
        return -1
    var slots: List[String] = [
        ".ff1_a.weight", ".ff1_b.weight", ".q.weight", ".k.weight",
        ".v.weight", ".o.weight", ".rel_k.weight", ".pw1.weight",
        ".pw2.weight", ".ff2_a.weight", ".ff2_b.weight",
    ]
    for i in range(len(slots)):
        if name.endswith(slots[i]):
            return i
    return -1


comptime PK_ARENA_ALIGN: Int = 32
"""Elements (128 bytes): every arena slice starts on a multiple of this."""


def _arena_span(n: Int) -> Int:
    """Arena elements a parameter of `n` values takes. ⚠ The ONE place the
    rule lives: `PkCount` sizes an arena with it and `PkLoad` places with it."""
    return ((n + PK_ARENA_ALIGN - 1) // PK_ARENA_ALIGN) * PK_ARENA_ALIGN


comptime PK_COUNT_ALL: Int = 0
comptime PK_COUNT_GEMM: Int = 1
comptime PK_COUNT_OTHER: Int = 2


struct PkCount(Movable, ParamVisitor, ParamVisitorRT):
    """Sums the arena elements of the parameters it visits: all of them, only
    the blocks' GEMM weights, or everything but those."""

    var elements: Int
    var which: Int

    def __init__(out self, which: Int = PK_COUNT_ALL):
        self.elements = 0
        self.which = which

    def visit_rt[target: StaticString](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, n: Int, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        var gemm = pk_gemm_slot(name) >= 0
        if self.which == PK_COUNT_GEMM and not gemm:
            return
        if self.which == PK_COUNT_OTHER and gemm:
            return
        self.elements += _arena_span(n)

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        self.visit_rt[target](name, param, grad, m, v, N, apply_decay, ctx)


struct PkLoad(Movable, ParamVisitor, ParamVisitorRT):
    """`LoadTorchNamed`; on a device, each parameter lands in the next slice of
    the current arena (`begin`) and its host copies are freed."""

    var ld: LoadTorchNamed[""]
    var arena: Optional[DeviceBuffer[DT]]
    var capacity: Int
    var offset: Int
    var half_arena: Optional[DeviceBuffer[PK_HALF]]
    """Set by `begin_half`: the blocks' GEMM weights go there, as float16."""
    var half_capacity: Int
    var half_offset: Int
    var half_out: List[DeviceBuffer[PK_HALF]]
    """The float16 slices placed since `begin_half`, in slot order."""

    def __init__(out self, var ld: LoadTorchNamed[""]):
        self.ld = ld^
        self.arena = None
        self.capacity = 0
        self.offset = 0
        self.half_arena = None
        self.half_capacity = 0
        self.half_offset = 0
        self.half_out = List[DeviceBuffer[PK_HALF]]()

    def begin(mut self, arena: DeviceBuffer[DT], capacity: Int):
        """The arena the next device parameters are sliced from."""
        self.arena = arena
        self.capacity = capacity
        self.offset = 0

    def begin_half(mut self, arena: DeviceBuffer[PK_HALF], capacity: Int):
        """The float16 arena the next block's GEMM weights are sliced from."""
        self.half_arena = arena
        self.half_capacity = capacity
        self.half_offset = 0
        self.half_out = List[DeviceBuffer[PK_HALF]]()

    def _place_half(
        mut self, name: String, mut param: Tensor, n: Int, slot: Int,
        c: DeviceContext,
    ) raises:
        """`param`'s host values, as float16, into the next float16 slice; the
        fp32 host copy is freed and NO fp32 device copy is made."""
        if slot != len(self.half_out):
            raise Error(
                "parakeet: GEMM weight '" + name + "' arrived as number "
                + String(len(self.half_out)) + " of its block, its slot is "
                + String(slot) + " — the walk order and PK_GEMM_* have drifted"
            )
        if self.half_offset + _arena_span(n) > self.half_capacity:
            raise Error("parakeet: '" + name + "' does not fit its float16 arena")
        var hb = c.enqueue_create_host_buffer[PK_HALF](n)
        c.synchronize()
        var limit = Float32(65504.0)
        for i in range(n):
            var v = param.data[i]
            if v > limit or v < -limit:
                raise Error("parakeet: '" + name + "' has a weight beyond float16's range")
            hb[i] = v.cast[PK_HALF]()
        var slice = self.half_arena.value().create_sub_buffer[PK_HALF](self.half_offset, n)
        c.enqueue_copy(slice, hb)
        c.synchronize()
        self.half_out.append(slice)
        self.half_offset += _arena_span(n)
        param.data = List[Scalar[DT]]()

    def visit_rt[target: StaticString](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, n: Int, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        comptime if target == "gpu":
            # Host fill only (no context, so no buffer of its own) ...
            self.ld.visit_rt["cpu"](name, param, grad, m, v, n, apply_decay, None)
            if len(param.data) < n:
                # Not filled: unmapped or missing. `report` raises on it.
                return
            if self.half_arena:
                var slot = pk_gemm_slot(name)
                if slot >= 0:
                    self._place_half(name, param, n, slot, ctx.value())
                    return
            if self.offset + _arena_span(n) > self.capacity:
                raise Error("parakeet: '" + name + "' does not fit the arena sized for its block")
            # ... then into the arena slice, and the host copies go.
            var c = ctx.value()
            param.dev = self.arena.value().create_sub_buffer[DT](self.offset, n)
            param.n = n
            param.upload_resident(c)
            c.synchronize()
            _drop_host(param)
            self.offset += _arena_span(n)
        else:
            self.ld.visit_rt[target](name, param, grad, m, v, n, apply_decay, ctx)

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        self.visit_rt[target](name, param, grad, m, v, N, apply_decay, ctx)


struct PkConv[C: Int, TAPS: Int](Movable):
    """A convolution whose every output channel reads one input channel:
    `weight` is `[C, TAPS]` (torch's `[C, 1, K]` or `[C, 1, KH, KW]`, same
    order), `bias` is `[C]`."""

    var weight: Param["weight", True, Self.C * Self.TAPS]
    var bias: Param["bias", False, Self.C]

    def __init__(out self):
        self.weight = Param["weight", True, Self.C * Self.TAPS]()
        self.bias = Param["bias", False, Self.C]()

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
    var half: List[DeviceBuffer[PK_HALF]]
    """Half-precision load only: the block's eleven GEMM weights as float16
    device buffers, indexed by `PK_GEMM_*` (their `Linear` leaves are empty)."""

    def __init__(out self):
        self.half = List[DeviceBuffer[PK_HALF]]()
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
    var arenas: List[DeviceBuffer[DT]]
    """Device only: the buffers the encoder's parameters are slices of."""
    var half_arenas: List[DeviceBuffer[PK_HALF]]
    """Half-precision load only: one float16 arena per block."""

    def __init__(out self):
        self.arenas = List[DeviceBuffer[DT]]()
        self.half_arenas = List[DeviceBuffer[PK_HALF]]()
        self.sub = PkSubsampling()
        self.layers = List[PkConformerLayer]()
        self.enc_proj = Linear[PK_D, PK_DEC]()
        self.embedding = Embedding[PK_VOCAB, PK_DEC]()
        self.lstm = List[PkLSTMLayer]()
        self.dec_proj = Linear[PK_DEC, PK_DEC]()
        self.joint = Linear[PK_DEC, PK_JOINT]()

    @staticmethod
    def make[target: StaticString](
        ctx: Optional[DeviceContext] = None
    ) raises -> Self:
        """The tree of leaves, with no parameter allocated: `load[target]`
        allocates each one as it fills it (module docstring). `target` and
        `ctx` are accepted so the call reads like every other `make`."""
        var m = Self()
        for _ in range(PK_LAYERS):
            m.layers.append(PkConformerLayer())
        for _ in range(PK_DEC_LAYERS):
            m.lstm.append(PkLSTMLayer())
        return m^

    def walk_encoder[target: StaticString, V: ParamVisitor](
        mut self, mut visitor: V, ctx: Optional[DeviceContext], prefix: String
    ) raises:
        """The parameters that live on `target`: subsampling, blocks, projection."""
        self.sub.walk[target](visitor, ctx, join_name(prefix, String("sub")))
        for i in range(len(self.layers)):
            self.layers[i].walk[target](
                visitor, ctx, join_name(prefix, String("layers." + String(i)))
            )
        self.enc_proj.for_each_param[target](
            visitor, ctx, join_name(prefix, String("enc_proj"))
        )

    def walk_decoder[target: StaticString, V: ParamVisitor](
        mut self, mut visitor: V, ctx: Optional[DeviceContext], prefix: String
    ) raises:
        """The prediction network and the joint."""
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

    def for_each_param[target: StaticString, V: ParamVisitor](
        mut self,
        mut visitor: V,
        ctx: Optional[DeviceContext],
        prefix: String = String(""),
    ) raises:
        self.walk_encoder[target](visitor, ctx, prefix)
        self.walk_decoder[target](visitor, ctx, prefix)

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

    def load[target: StaticString, WEIGHTS: StaticString = "fp32"](
        mut self, path: String, ctx: Optional[DeviceContext] = None
    ) raises:
        """Fill every parameter and running statistic from the published
        `model.safetensors`, and check the accounting in BOTH directions:

        - every parameter of ours has a map entry and every entry was claimed
          (a walk and a map that drifted apart leave weights at zero);
        - every fp32 tensor of the file was read: the only tensors left
          unclaimed are the 24 `num_batches_tracked` counters.
        """
        comptime assert WEIGHTS == "fp32" or WEIGHTS == "fp16", (
            "ParakeetTDT.load: WEIGHTS is 'fp32' or 'fp16'"
        )
        comptime assert WEIGHTS == "fp32" or target == "gpu", (
            "ParakeetTDT.load: half-precision weights are a device form"
        )
        var w = PkLoad(LoadTorchNamed[""](SafeTensors(path), parakeet_name_map()))
        comptime if target == "gpu":
            var c = ctx.value()
            # Subsampling + projection share the first arena; then one per block.
            var c0 = PkCount()
            self.sub.walk["cpu"](c0, None, String("sub"))
            self.enc_proj.for_each_param["cpu"](c0, None, String("enc_proj"))
            var a0 = c.enqueue_create_buffer[DT](c0.elements)
            a0.enqueue_fill(0.0)
            w.begin(a0, c0.elements)
            self.sub.walk[target](w, ctx, String("sub"))
            self.enc_proj.for_each_param[target](w, ctx, String("enc_proj"))
            self.arenas.append(a0)
            for i in range(len(self.layers)):
                var name = String("layers." + String(i))
                # In half precision the GEMM weights are counted, and placed,
                # in the block's float16 arena and nowhere else.
                var ci = PkCount(PK_COUNT_OTHER if WEIGHTS == "fp16" else PK_COUNT_ALL)
                self.layers[i].walk["cpu"](ci, None, name)
                self.layers[i].walk_state["cpu"](ci, None, name)
                var ai = c.enqueue_create_buffer[DT](ci.elements)
                ai.enqueue_fill(0.0)
                w.begin(ai, ci.elements)
                comptime if WEIGHTS == "fp16":
                    var ch = PkCount(PK_COUNT_GEMM)
                    self.layers[i].walk["cpu"](ch, None, name)
                    var ah = c.enqueue_create_buffer[PK_HALF](ch.elements)
                    ah.enqueue_fill(0.0)
                    w.begin_half(ah, ch.elements)
                    self.half_arenas.append(ah)
                self.layers[i].walk[target](w, ctx, name)
                self.layers[i].walk_state[target](w, ctx, name)
                comptime if WEIGHTS == "fp16":
                    if len(w.half_out) != PK_GEMM_SLOTS:
                        raise Error(
                            "parakeet: block " + String(i) + " placed "
                            + String(len(w.half_out)) + " float16 weights, expected "
                            + String(PK_GEMM_SLOTS)
                        )
                    self.layers[i].half = w.half_out.copy()
                self.arenas.append(ai)
            # The prediction network and the joint: one more arena.
            var cd = PkCount()
            self.walk_decoder["cpu"](cd, None, String(""))
            var ad = c.enqueue_create_buffer[DT](cd.elements)
            ad.enqueue_fill(0.0)
            w.begin(ad, cd.elements)
            self.walk_decoder[target](w, ctx, String(""))
            self.arenas.append(ad)
        else:
            self.walk_encoder[target](w, ctx, String(""))
            self.for_each_state[target](w, ctx)
            self.walk_decoder[target](w, ctx, String(""))
        w.ld.report(String("parakeet"))
        if len(w.ld.loaded) + len(w.ld.zeroed) != w.ld.map.size():
            raise Error(
                "parakeet: filled " + String(len(w.ld.loaded) + len(w.ld.zeroed))
                + " of the map's " + String(w.ld.map.size()) + " entries — the"
                " walk never emitted the others, so those weights stay zero"
            )
        if len(w.ld.file.names) != PK_FILE_TENSORS or len(w.ld.loaded) != PK_FILE_WEIGHTS:
            raise Error(
                "parakeet: read " + String(len(w.ld.loaded)) + " tensors of a file"
                " of " + String(len(w.ld.file.names)) + ", expected "
                + String(PK_FILE_WEIGHTS) + " of " + String(PK_FILE_TENSORS)
                + " — not the checkpoint this map was written against"
            )
        var left = w.ld.unclaimed()
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
