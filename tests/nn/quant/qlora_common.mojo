"""Shared by the QLoRA gates (CPU and GPU), one SmolVLA text shape at a time.

    [B = 0]  `QLoRALinear` at PEFT's initialisation (B = 0) against the bare
             `QLinear` holding the same blocks: forward and grad_input
             BIT-identical (the adapter adds an exact 0)
    [vs LoRA over deq(W)]  `QLoRALinear` with a nonzero adapter against
             `LoRALinear` whose frozen `Linear` holds deq(W)ᵀ and a zero
             bias, the same A, B and α: forward, grad_input, dA, dB

dA and dB depend only on the input and the output gradient, never on the
base weight, so they are the same computation on both sides. The forward and
grad_input differ by the base GEMM alone: fp32 GEMMs on both sides on the CPU
and Metal, MAX's int4 GEMM (bf16 activations) against an fp32 one on NVIDIA.
"""

from std.math import abs
from std.random import seed
from max.gpu.host import DeviceContext

from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Deterministic
from noeira.nn.quant.qlinear import QLinear
from noeira.nn.quant.qlora import QLoRALinear
from noeira.nn.peft.lora import LoRALinear
from tests.nn.quant.qlinear_common import gaussian, put, to_host, rel


struct QLoRAVerdict(Copyable, Movable):
    var b0_fwd_diff: Int
    """Values of the B = 0 forward that differ from the bare QLinear's."""
    var b0_dx_diff: Int
    var fwd: Float64
    var dx: Float64
    var da: Float64
    var db: Float64
    var grad_mag: Float64
    """min(max |dA|, max |dB|) of the reference: the dA / dB rows are vacuous
    if it is 0."""

    def __init__(out self):
        self.b0_fwd_diff = 0
        self.b0_dx_diff = 0
        self.fwd = 0
        self.dx = 0
        self.da = 0
        self.db = 0
        self.grad_mag = 0


def _ndiff(ref a: List[Float32], ref b: List[Float32]) -> Int:
    var n = 0
    for i in range(len(a)):
        if a[i] != b[i]:
            n += 1
    return n


def check_qlora[
    target: StaticString, IN: Int, OUT: Int, T: Int, R: Int
](ctx: Optional[DeviceContext], label: String) raises -> QLoRAVerdict:
    var v = QLoRAVerdict()
    seed(IN * 13 + OUT + R)
    var w = gaussian(OUT * IN, 0.05)
    var xs = gaussian(T * IN, 1.0)
    var gs = gaussian(T * OUT, 1.0)
    var x = Tensor()
    var g = Tensor()
    put(x, xs, target, ctx)
    put(g, gs, target, ctx)

    # ── B = 0 against the bare QLinear ─────────────────────────────────
    var ql = QLinear[IN, OUT].make[target, Deterministic](ctx)
    ql.set_weight[target](w, ctx)
    var qa = QLoRALinear[IN, OUT, R].make[target, Deterministic](ctx)
    qa.base.set_weight[target](w, ctx)
    var y0 = Tensor()
    var y1 = Tensor()
    ql.forward[target, T](TensorRefs[1](x), y0, ctx)
    qa.forward[target, T](TensorRefs[1](x), y1, ctx)
    var gx0 = Tensor()
    var gx1 = Tensor()
    ql.vjp[target, T](TensorRefs[1](x), g, TensorRefs[1](gx0), ctx)
    qa.vjp[target, T](TensorRefs[1](x), g, TensorRefs[1](gx1), ctx)
    v.b0_fwd_diff = _ndiff(to_host(y1, T * OUT, ctx), to_host(y0, T * OUT, ctx))
    v.b0_dx_diff = _ndiff(to_host(gx1, T * IN, ctx), to_host(gx0, T * IN, ctx))

    # ── a nonzero adapter, against LoRA over deq(W) ────────────────────
    comptime RP = QLoRALinear[IN, OUT, R].Adapter.R_PAD
    var a_vals = List[Float32](length=IN * RP, fill=0)
    var b_vals = List[Float32](length=RP * OUT, fill=0)
    var ga = gaussian(IN * R, 0.05)
    var gb = gaussian(R * OUT, 0.05)
    for i in range(IN):
        for j in range(R):
            a_vals[i * RP + j] = ga[i * R + j]
    for j in range(R):
        for o in range(OUT):
            b_vals[j * OUT + o] = gb[j * OUT + o]
    put(qa.adapter.lora_A.val, a_vals, target, ctx)
    put(qa.adapter.lora_B.val, b_vals, target, ctx)
    qa.adapter.alpha = 32.0
    var deq = qa.base.dequantized()
    var la = LoRALinear[IN, OUT, R].make[target, Deterministic](ctx)
    var wl = List[Float32](length=IN * OUT, fill=0)
    for o in range(OUT):
        for i in range(IN):
            wl[i * OUT + o] = deq[o * IN + i]
    var zb = List[Float32](length=OUT, fill=0)
    put(la.base.weight.val, wl, target, ctx)
    put(la.base.bias.val, zb, target, ctx)
    la.base.weight.val.version += 1
    put(la.adapter.lora_A.val, a_vals, target, ctx)
    put(la.adapter.lora_B.val, b_vals, target, ctx)
    la.adapter.alpha = 32.0
    qa.zero_grad[target](ctx)
    la.zero_grad[target](ctx)

    var yq = Tensor()
    var yl = Tensor()
    qa.forward[target, T](TensorRefs[1](x), yq, ctx)
    la.forward[target, T](TensorRefs[1](x), yl, ctx)
    var gq = Tensor()
    var gl = Tensor()
    qa.vjp[target, T](TensorRefs[1](x), g, TensorRefs[1](gq), ctx)
    la.vjp[target, T](TensorRefs[1](x), g, TensorRefs[1](gl), ctx)
    v.fwd = rel(to_host(yq, T * OUT, ctx), to_host(yl, T * OUT, ctx), T * OUT)
    v.dx = rel(to_host(gq, T * IN, ctx), to_host(gl, T * IN, ctx), T * IN)
    var da_ref = to_host(la.adapter.lora_A.grd, IN * RP, ctx)
    var db_ref = to_host(la.adapter.lora_B.grd, RP * OUT, ctx)
    v.da = rel(to_host(qa.adapter.lora_A.grd, IN * RP, ctx), da_ref, IN * RP)
    v.db = rel(to_host(qa.adapter.lora_B.grd, RP * OUT, ctx), db_ref, RP * OUT)
    var ma = Float64(0)
    var mb = Float64(0)
    for e in da_ref:
        ma = max(ma, abs(Float64(e)))
    for e in db_ref:
        mb = max(mb, abs(Float64(e)))
    v.grad_mag = min(ma, mb)
    print("    ", label, "[", IN, "->", OUT, "] T =", T, "R =", R,
          ": B = 0 vs QLinear, values differing: forward", v.b0_fwd_diff,
          "dx", v.b0_dx_diff, "| vs LoRA over deq(W): forward", v.fwd,
          "dx", v.dx, "dA", v.da, "dB", v.db)
    return v^
