"""QLoRA: a frozen Q4_0 linear (`QLinear`) with a trainable LoRA adapter
(`nn/peft/lora.mojo`) beside it.

    y = deq(W) x  +  s · (x A) B          W: Q4_0 blocks, frozen; A, B fp32

`QLoRALinear` is `LoRALinear` with the `Linear` replaced by a `QLinear`, for
models composed from modules; SmolVLA's hand-written text layers get the same
pairing from `DecoderLayerWeights[..., LORA_R, Q4=True]`, whose `proj_fwd` /
`proj_vjp` run the `QLinear` and then the adapter's hook. The adapter never
reads the base weight, so it is the same code over an fp32 or an int4 base:
its gradients (dA, dB) depend only on the layer's input and output gradient,
and the base adds its own input gradient (`QLinear.vjp`, through the
dequantized weight).

## Activations stay fp32

The plan sketched QLoRA with bf16-flow activations, because MAX's int4 GEMM
takes and returns bf16. `QLinear` already keeps fp32 at its boundary and
casts inside (results doc, Q4.2), and SmolVLA's towers are fp32 end to end,
so the pairing is fp32 too. A bf16-flow tower would drop `QLinear`'s two cast
kernels; it is a change to the whole prefill, not to this layer.

## No merge

Folding s · A B into Q4_0 blocks would re-quantize them: the merged layer
would not compute the adapted function. A QLoRA deploy keeps the adapter
unmerged (one rank-R product per linear).
"""

from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from ..core.tensor import Tensor
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import ParamVisitor
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP
from ..peft.lora import LoRAAdapter
from .qlinear import QLinear


struct QLoRALinear[
    IN_: Int, OUT_: Int, R: Int, PAD_R: Int = 16, RSLORA: Bool = False
](Module):
    """`QLinear[IN_, OUT_]` (frozen by construction) + `LoRAAdapter`. The
    params walk as the adapter's `lora_A` and `lora_B` under the layer's
    prefix: the int4 base has no `Param`."""

    comptime ARITY = 1
    comptime IN_DIMS = Array[Int, 1](fill=Self.IN_)
    comptime OUT_DIM = Self.OUT_
    comptime Base = QLinear[Self.IN_, Self.OUT_]
    comptime Adapter = LoRAAdapter[
        Self.IN_, Self.OUT_, Self.R, Self.PAD_R, Self.RSLORA
    ]

    var base: Self.Base
    var adapter: Self.Adapter

    def __init__(out self):
        self.base = Self.Base()
        self.adapter = Self.Adapter()

    @staticmethod
    def display_label() -> String:
        return String("QLoRALinear")

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        """An empty int4 base (load it with `base.set_weight` or
        `base.set_blocks`) and the adapter at PEFT's initialisation with
        α = R (`set_attr["lora_alpha"]` changes it)."""
        var l = Self()
        l.base = Self.Base.make[target, INIT](ctx)
        l.adapter = Self.Adapter.make[target](ctx)
        return l^

    def set_attr[ATTR: StaticString](mut self, value: Scalar[DT]):
        comptime if ATTR == "lora_alpha":
            self.adapter.alpha = Float64(value)

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[1, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        self.base.forward[target, B, o, POLICY](inputs, out, ctx)
        self.adapter.forward_add[target, B](inputs, out, ctx)

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
        """The base's grad_input (it overwrites), then the adapter's share
        and its dA, dB (it adds)."""
        self.base.vjp[target, B, ofi, ogi, POLICY](
            forward_input, grad_output, grad_inputs, ctx
        )
        self.adapter.vjp_add[target, B](
            forward_input, grad_output, grad_inputs, ctx
        )

    def for_each_param[
        target: StaticString, V: ParamVisitor
    ](
        mut self,
        mut visitor: V,
        ctx: Optional[DeviceContext],
        prefix: String = String(""),
    ) raises:
        self.adapter.walk[target](visitor, ctx, prefix)

    def zero_grad[
        target: StaticString
    ](mut self, ctx: Optional[DeviceContext]) raises:
        self.adapter.zero_grad[target](ctx)

    def release_buffers(mut self):
        self.base.release_buffers()
        self.adapter.release_buffers()
