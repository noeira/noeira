# +--------------------------------------------------------------------------+ #
# | SmolVLA — one decoder layer's weights, for both towers
# +--------------------------------------------------------------------------+ #
"""`DecoderLayerWeights` — the parameters of one SmolVLA decoder layer.

**The VLM text layer and the expert layer are the same topology at different
widths**, so they are the same struct. Only five numbers differ:

    stream  W    FF    QW   KVW  KV_IN   what it is
    ------  ---  ----  ---  ---  -----   ---------------------------------
    VLM     960  2560  960  320  960     text tower layer
    expert  720  2048  960  320  720     expert, self-attention  (even i)
    expert  720  2048  960  320  320     expert, cross-attention (odd i)

`KV_IN` is the whole story: a layer projects K/V either from its own stream or,
for the expert's odd layers, from the VLM's cached 320-wide K/V. `QW` is 960 for
both because the expert computes queries in the VLM's head geometry and projects
back down through `o` — which is what lets one attention serve two streams of
different widths.

## Why weights and not a `Module`

The fused loop needs each layer's K/V *between* the projection and the attention,
to write into the prefix cache. A composed `Sequential`/`Repeat` tower hands back
only its final activation, so there is nowhere to reach in. These structs own the
parameters and expose them by name; the loop that drives them lives in
`fused.mojo`.

## LoRA

`LORA_R > 0` gives each of the seven linears a `LoRAAdapter` of that rank
(`<name>_lora`, beside the linear it adapts), walked right after it as
`<name>.lora_A` / `<name>.lora_B`. `fused.mojo` calls `lora_fwd` / `lora_vjp`
after every linear's `forward` / `vjp`; at `LORA_R = 0`, the default, the
adapters are never built or walked and both calls compile to nothing, so the
type, its walk and every checkpoint are what they were.

`walk[..., WITH_LORA=False]` walks the base parameters only: what the
checkpoint name maps describe (`SmolVLAPolicy.load`) and what freezing the
base for LoRA training freezes. The default walk, the optimizers' and the
checkpoints', includes the adapters.
"""

from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.initializer import Initializer
from noeira.nn.core.param import ParamVisitor
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.walkers import join_name
from noeira.nn.primitives.linear import Linear
from noeira.nn.primitives.rms_norm import RMSNorm
from noeira.nn.peft.lora import LoRAAdapter

comptime SMOLLM_RMS_EPS: Scalar[DT] = 1e-5
"""⚠ SmolLM2's `rms_norm_eps`. `nn`'s `RMSNorm` defaults to **1e-4**, ten times
larger — a real difference in the output that no shape or NaN check can see.
Every norm in both towers is pinned to this."""

comptime SmolNorm[D: Int] = RMSNorm[D, SMOLLM_RMS_EPS]

# The seven adapted projections, for `lora_fwd` / `lora_vjp`.
comptime LORA_Q = 0
comptime LORA_K = 1
comptime LORA_V = 2
comptime LORA_O = 3
comptime LORA_GATE = 4
comptime LORA_UP = 5
comptime LORA_DOWN = 6


struct DecoderMLP[W: Int, FF: Int, LORA_R: Int = 0](Movable):
    """`down(silu(gate(x)) * up(x))` — three separate `Linear`s, as shipped,
    and with `LORA_R > 0` an adapter beside each (module docstring)."""

    comptime AR = Self.LORA_R if Self.LORA_R > 0 else 1
    """The adapters' rank type parameter; never built at `LORA_R = 0`."""

    var gate: Linear[Self.W, Self.FF]
    var up: Linear[Self.W, Self.FF]
    var down: Linear[Self.FF, Self.W]
    var gate_lora: LoRAAdapter[Self.W, Self.FF, Self.AR]
    var up_lora: LoRAAdapter[Self.W, Self.FF, Self.AR]
    var down_lora: LoRAAdapter[Self.FF, Self.W, Self.AR]

    def __init__(out self):
        self.gate = Linear[Self.W, Self.FF]()
        self.up = Linear[Self.W, Self.FF]()
        self.down = Linear[Self.FF, Self.W]()
        self.gate_lora = LoRAAdapter[Self.W, Self.FF, Self.AR]()
        self.up_lora = LoRAAdapter[Self.W, Self.FF, Self.AR]()
        self.down_lora = LoRAAdapter[Self.FF, Self.W, Self.AR]()

    def __init__(out self, *, deinit move: Self):
        self.gate = move.gate^
        self.up = move.up^
        self.down = move.down^
        self.gate_lora = move.gate_lora^
        self.up_lora = move.up_lora^
        self.down_lora = move.down_lora^

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        var m = Self()
        m.gate = Linear[Self.W, Self.FF].make[target, INIT](ctx)
        m.up = Linear[Self.W, Self.FF].make[target, INIT](ctx)
        m.down = Linear[Self.FF, Self.W].make[target, INIT](ctx)
        comptime if Self.LORA_R > 0:
            m.gate_lora = LoRAAdapter[Self.W, Self.FF, Self.AR].make[target](ctx)
            m.up_lora = LoRAAdapter[Self.W, Self.FF, Self.AR].make[target](ctx)
            m.down_lora = LoRAAdapter[Self.FF, Self.W, Self.AR].make[target](ctx)
        return m^

    def walk[
        target: StaticString, V: ParamVisitor, WITH_LORA: Bool = True
    ](mut self, mut v: V, ctx: Optional[DeviceContext], prefix: String) raises:
        comptime LORA = Self.LORA_R > 0 and WITH_LORA
        var g = join_name(prefix, String("gate"))
        var u = join_name(prefix, String("up"))
        var d = join_name(prefix, String("down"))
        self.gate.for_each_param[target](v, ctx, g)
        comptime if LORA:
            self.gate_lora.walk[target](v, ctx, g)
        self.up.for_each_param[target](v, ctx, u)
        comptime if LORA:
            self.up_lora.walk[target](v, ctx, u)
        self.down.for_each_param[target](v, ctx, d)
        comptime if LORA:
            self.down_lora.walk[target](v, ctx, d)

    def set_lora_alpha(mut self, alpha: Float64):
        """PEFT's `lora_alpha` for the three adapters; nothing at R = 0."""
        self.gate_lora.alpha = alpha
        self.up_lora.alpha = alpha
        self.down_lora.alpha = alpha

    def zero_grad[
        target: StaticString
    ](mut self, ctx: Optional[DeviceContext]) raises:
        """⚠ The SAME three children as `walk`, in the same order. A component
        that is walked but not zeroed accumulates across steps forever; one
        that is zeroed but not walked never trains. Neither raises."""
        self.gate.zero_grad[target](ctx)
        self.up.zero_grad[target](ctx)
        self.down.zero_grad[target](ctx)
        comptime if Self.LORA_R > 0:
            self.gate_lora.zero_grad[target](ctx)
            self.up_lora.zero_grad[target](ctx)
            self.down_lora.zero_grad[target](ctx)

    def walk_state[
        target: StaticString, V: ParamVisitor
    ](mut self, mut v: V, ctx: Optional[DeviceContext], prefix: String
    ) raises:
        """The State walk the checkpoint runs right after the Param walk.

        ⚠ These three leaves hold no State — `Linear` has weight and bias and
        nothing running — so this is a no-op TODAY. It exists so the walk is
        complete by construction: a leaf that later grows a State field would
        otherwise be saved without it, and a checkpoint missing a running
        statistic reloads to a model that behaves differently and says
        nothing.
        """
        self.gate.for_each_state[target](
            v, ctx, join_name(prefix, String("gate"))
        )
        self.up.for_each_state[target](
            v, ctx, join_name(prefix, String("up"))
        )
        self.down.for_each_state[target](
            v, ctx, join_name(prefix, String("down"))
        )


struct DecoderLayerWeights[
    W: Int, FF: Int, QW: Int, KVW: Int, KV_IN: Int, LORA_R: Int = 0
](Movable):
    """One expert layer. `KV_IN` is the only difference between the two kinds:
    `W` (720) for a self-attention layer, `KVW` (320) for a cross-attention one,
    which reads the VLM's cached K/V instead of its own stream. `LORA_R > 0`
    adds an adapter beside each of the seven linears (module docstring)."""

    comptime AR = Self.LORA_R if Self.LORA_R > 0 else 1

    var input_layernorm: SmolNorm[Self.W]
    var q: Linear[Self.W, Self.QW]
    var k: Linear[Self.KV_IN, Self.KVW]
    var v: Linear[Self.KV_IN, Self.KVW]
    var o: Linear[Self.QW, Self.W]
    var post_attention_layernorm: SmolNorm[Self.W]
    var mlp: DecoderMLP[Self.W, Self.FF, Self.LORA_R]
    var q_lora: LoRAAdapter[Self.W, Self.QW, Self.AR]
    var k_lora: LoRAAdapter[Self.KV_IN, Self.KVW, Self.AR]
    var v_lora: LoRAAdapter[Self.KV_IN, Self.KVW, Self.AR]
    var o_lora: LoRAAdapter[Self.QW, Self.W, Self.AR]

    def __init__(out self):
        self.input_layernorm = SmolNorm[Self.W]()
        self.q = Linear[Self.W, Self.QW]()
        self.k = Linear[Self.KV_IN, Self.KVW]()
        self.v = Linear[Self.KV_IN, Self.KVW]()
        self.o = Linear[Self.QW, Self.W]()
        self.post_attention_layernorm = SmolNorm[Self.W]()
        self.mlp = DecoderMLP[Self.W, Self.FF, Self.LORA_R]()
        self.q_lora = LoRAAdapter[Self.W, Self.QW, Self.AR]()
        self.k_lora = LoRAAdapter[Self.KV_IN, Self.KVW, Self.AR]()
        self.v_lora = LoRAAdapter[Self.KV_IN, Self.KVW, Self.AR]()
        self.o_lora = LoRAAdapter[Self.QW, Self.W, Self.AR]()

    def __init__(out self, *, deinit move: Self):
        self.input_layernorm = move.input_layernorm^
        self.q = move.q^
        self.k = move.k^
        self.v = move.v^
        self.o = move.o^
        self.post_attention_layernorm = move.post_attention_layernorm^
        self.mlp = move.mlp^
        self.q_lora = move.q_lora^
        self.k_lora = move.k_lora^
        self.v_lora = move.v_lora^
        self.o_lora = move.o_lora^

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        var l = Self()
        l.input_layernorm = SmolNorm[Self.W].make[target, INIT](ctx)
        l.q = Linear[Self.W, Self.QW].make[target, INIT](ctx)
        l.k = Linear[Self.KV_IN, Self.KVW].make[target, INIT](ctx)
        l.v = Linear[Self.KV_IN, Self.KVW].make[target, INIT](ctx)
        l.o = Linear[Self.QW, Self.W].make[target, INIT](ctx)
        l.post_attention_layernorm = SmolNorm[Self.W].make[target, INIT](ctx)
        l.mlp = DecoderMLP[Self.W, Self.FF, Self.LORA_R].make[target, INIT](ctx)
        comptime if Self.LORA_R > 0:
            l.q_lora = LoRAAdapter[Self.W, Self.QW, Self.AR].make[target](ctx)
            l.k_lora = LoRAAdapter[Self.KV_IN, Self.KVW, Self.AR].make[target](ctx)
            l.v_lora = LoRAAdapter[Self.KV_IN, Self.KVW, Self.AR].make[target](ctx)
            l.o_lora = LoRAAdapter[Self.QW, Self.W, Self.AR].make[target](ctx)
        return l^

    def walk[
        target: StaticString, V: ParamVisitor, WITH_LORA: Bool = True
    ](mut self, mut vis: V, ctx: Optional[DeviceContext],
      prefix: String) raises:
        comptime LORA = Self.LORA_R > 0 and WITH_LORA
        self.input_layernorm.for_each_param[target](
            vis, ctx, join_name(prefix, String("input_layernorm"))
        )
        var sa = join_name(prefix, String("self_attn"))
        var nq = join_name(sa, String("q"))
        var nk = join_name(sa, String("k"))
        var nv = join_name(sa, String("v"))
        var no = join_name(sa, String("o"))
        self.q.for_each_param[target](vis, ctx, nq)
        comptime if LORA:
            self.q_lora.walk[target](vis, ctx, nq)
        self.k.for_each_param[target](vis, ctx, nk)
        comptime if LORA:
            self.k_lora.walk[target](vis, ctx, nk)
        self.v.for_each_param[target](vis, ctx, nv)
        comptime if LORA:
            self.v_lora.walk[target](vis, ctx, nv)
        self.o.for_each_param[target](vis, ctx, no)
        comptime if LORA:
            self.o_lora.walk[target](vis, ctx, no)
        self.post_attention_layernorm.for_each_param[target](
            vis, ctx, join_name(prefix, String("post_attention_layernorm"))
        )
        self.mlp.walk[target, V, WITH_LORA](
            vis, ctx, join_name(prefix, String("mlp"))
        )

    def set_lora_alpha(mut self, alpha: Float64):
        """PEFT's `lora_alpha` for the seven adapters; nothing at R = 0."""
        self.q_lora.alpha = alpha
        self.k_lora.alpha = alpha
        self.v_lora.alpha = alpha
        self.o_lora.alpha = alpha
        self.mlp.set_lora_alpha(alpha)

    def zero_grad[
        target: StaticString
    ](mut self, ctx: Optional[DeviceContext]) raises:
        """⚠ The SAME seven children as `walk`. See `DecoderMLP.zero_grad`."""
        self.input_layernorm.zero_grad[target](ctx)
        self.q.zero_grad[target](ctx)
        self.k.zero_grad[target](ctx)
        self.v.zero_grad[target](ctx)
        self.o.zero_grad[target](ctx)
        self.post_attention_layernorm.zero_grad[target](ctx)
        self.mlp.zero_grad[target](ctx)
        comptime if Self.LORA_R > 0:
            self.q_lora.zero_grad[target](ctx)
            self.k_lora.zero_grad[target](ctx)
            self.v_lora.zero_grad[target](ctx)
            self.o_lora.zero_grad[target](ctx)

    # ── LoRA hooks, called by `fused.mojo` after each linear ───────────
    def lora_fwd[
        WHICH: Int, target: StaticString, T: Int, o: MutOrigin
    ](
        mut self,
        inputs: TensorRefs[1, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """Projection WHICH's adapter (`LORA_Q` .. `LORA_DOWN`) adds its
        share onto `out`, with the linear's own arguments. Nothing at
        `LORA_R = 0`."""
        comptime if Self.LORA_R > 0:
            comptime if WHICH == LORA_Q:
                self.q_lora.forward_add[target, T](inputs, out, ctx)
            elif WHICH == LORA_K:
                self.k_lora.forward_add[target, T](inputs, out, ctx)
            elif WHICH == LORA_V:
                self.v_lora.forward_add[target, T](inputs, out, ctx)
            elif WHICH == LORA_O:
                self.o_lora.forward_add[target, T](inputs, out, ctx)
            elif WHICH == LORA_GATE:
                self.mlp.gate_lora.forward_add[target, T](inputs, out, ctx)
            elif WHICH == LORA_UP:
                self.mlp.up_lora.forward_add[target, T](inputs, out, ctx)
            else:
                comptime assert WHICH == LORA_DOWN, "lora_fwd: WHICH is 0..6"
                self.mlp.down_lora.forward_add[target, T](inputs, out, ctx)

    def lora_vjp[
        WHICH: Int, target: StaticString, T: Int, ofi: MutOrigin,
        ogi: MutOrigin,
    ](
        mut self,
        forward_input: TensorRefs[1, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[1, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """Projection WHICH's adapter: dA, dB, and its share added onto the
        linear's grad_input, with the linear's own arguments right after its
        `vjp`. Nothing at `LORA_R = 0`."""
        comptime if Self.LORA_R > 0:
            comptime if WHICH == LORA_Q:
                self.q_lora.vjp_add[target, T](forward_input, grad_output, grad_inputs, ctx)
            elif WHICH == LORA_K:
                self.k_lora.vjp_add[target, T](forward_input, grad_output, grad_inputs, ctx)
            elif WHICH == LORA_V:
                self.v_lora.vjp_add[target, T](forward_input, grad_output, grad_inputs, ctx)
            elif WHICH == LORA_O:
                self.o_lora.vjp_add[target, T](forward_input, grad_output, grad_inputs, ctx)
            elif WHICH == LORA_GATE:
                self.mlp.gate_lora.vjp_add[target, T](forward_input, grad_output, grad_inputs, ctx)
            elif WHICH == LORA_UP:
                self.mlp.up_lora.vjp_add[target, T](forward_input, grad_output, grad_inputs, ctx)
            else:
                comptime assert WHICH == LORA_DOWN, "lora_vjp: WHICH is 0..6"
                self.mlp.down_lora.vjp_add[target, T](forward_input, grad_output, grad_inputs, ctx)

    def walk_state[
        target: StaticString, V: ParamVisitor
    ](mut self, mut vis: V, ctx: Optional[DeviceContext], prefix: String
    ) raises:
        """⚠ The SAME seven children as `walk`. See `DecoderMLP.walk_state`."""
        self.input_layernorm.for_each_state[target](
            vis, ctx, join_name(prefix, String("input_layernorm"))
        )
        var sa = join_name(prefix, String("self_attn"))
        self.q.for_each_state[target](vis, ctx, join_name(sa, String("q")))
        self.k.for_each_state[target](vis, ctx, join_name(sa, String("k")))
        self.v.for_each_state[target](vis, ctx, join_name(sa, String("v")))
        self.o.for_each_state[target](vis, ctx, join_name(sa, String("o")))
        self.post_attention_layernorm.for_each_state[target](
            vis, ctx, join_name(prefix, String("post_attention_layernorm"))
        )
        self.mlp.walk_state[target](vis, ctx, join_name(prefix, String("mlp")))
