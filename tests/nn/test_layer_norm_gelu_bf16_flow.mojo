"""LayerNorm + GELU(=Elementwise) bf16-FLOW (AMP "Step B") gate.

Both leaves are fp32-INTERNAL: only their I/O activations flow at bf16 (the stats
/ affine / cache for LayerNorm, the OP math for Elementwise, stay fp32). On Apple
Metal bf16 IS supported for plain casts (these leaves do no bf16 GEMM), so we can
assert a SANE round-trip: bf16-flow output vs the fp32 reference within bf16
precision (~1-2%), not just "no crash".

Two checks per leaf:
 1. NoAMP guard: the fp32 `LayerNorm[D]` / `GELUTanh[D]` GPU forward is the reference.
 2. bf16-flow: `LayerNorm[D, bfloat16]` / `GELUTanh[D, bfloat16]` (= `Elementwise[D,
    GELUTanhOp, bfloat16]`) fwd+vjp on GPU compiles + runs, and forward matches the
    fp32 reference to bf16 tolerance (a bf16-vs-fp32 round-trip rel err).

Run: pixi run -e apple mojo run -I . tests/nn/test_layer_norm_gelu_bf16_flow.mojo
"""

from std.math import abs
from std.sys import has_accelerator
from std.testing import assert_true
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor, TensorImpl
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Deterministic
from noeira.nn.primitives.layer_norm import LayerNorm
from noeira.nn.primitives.activations import GELUTanh

comptime DIM = 16
comptime B = 8
comptime BF = DType.bfloat16


def _rel_err(ref_v: Tensor, got: TensorImpl[BF], n: Int) -> Float64:
    """max relative error of the bf16-flow `got` vs the fp32 `ref_v`."""
    var m: Float64 = 0.0
    for i in range(n):
        var r = Float64(ref_v.data[i])
        var g = Float64(got.data[i].cast[DT]())
        var denom = abs(r) if abs(r) > 1e-4 else 1e-4
        var e = abs(r - g) / denom
        if e > m:
            m = e
    return m


def test_layer_norm_bf16_flow() raises:
    print("LayerNorm bf16-flow (fp32-internal) ...")
    var c = DeviceContext()

    # ── fp32 reference (GPU forward) ──
    var refm = LayerNorm[DIM].make["gpu", Deterministic](Optional(c))
    var xr = Tensor.alloc(B * DIM)
    for i in range(B * DIM):
        xr.data[i] = Scalar[DT]((i % 13) - 6) * 0.18
    xr.upload(c)
    var outr = Tensor.alloc(B * DIM)
    refm.forward["gpu", B](TensorRefs[1](xr), outr, Optional(c))
    outr.download(c)

    # ── bf16-flow (stage the same input at bf16 via CPU alloc + upload) ──
    var m = LayerNorm[DIM, BF].make["gpu", Deterministic](Optional(c))
    var x = TensorImpl[BF].alloc(B * DIM)
    for i in range(B * DIM):
        x.data[i] = xr.data[i].cast[BF]()
    x.upload(c)
    var out = TensorImpl[BF].alloc(B * DIM)
    var go = TensorImpl[BF].alloc(B * DIM)
    var gi = TensorImpl[BF].alloc(B * DIM)
    out.upload(c)
    go.upload(c)
    gi.upload(c)
    m.forward["gpu", B](TensorRefs[1, _, BF](x), out, Optional(c))
    m.zero_grad["gpu"](Optional(c))
    m.vjp["gpu", B](
        TensorRefs[1, _, BF](x), go, TensorRefs[1, _, BF](gi), Optional(c)
    )
    c.synchronize()
    out.download(c)

    var re = _rel_err(outr, out, B * DIM)
    print("  bf16-flow fwd+vjp ran; forward rel err vs fp32:", re)
    assert_true(re < 0.05, "LayerNorm bf16-flow forward within bf16 tolerance")
    print("  ok")


def test_gelu_bf16_flow() raises:
    print("GELU(=Elementwise) bf16-flow (fp32-internal) ...")
    var c = DeviceContext()

    # ── fp32 reference ──
    var refm = GELUTanh[DIM].make["gpu", Deterministic](Optional(c))
    var xr = Tensor.alloc(B * DIM)
    for i in range(B * DIM):
        xr.data[i] = Scalar[DT]((i % 11) - 5) * 0.25
    xr.upload(c)
    var outr = Tensor.alloc(B * DIM)
    refm.forward["gpu", B](TensorRefs[1](xr), outr, Optional(c))
    outr.download(c)

    # ── bf16-flow (stage the same input at bf16 via CPU alloc + upload) ──
    var m = GELUTanh[DIM, BF].make["gpu", Deterministic](Optional(c))
    var x = TensorImpl[BF].alloc(B * DIM)
    for i in range(B * DIM):
        x.data[i] = xr.data[i].cast[BF]()
    x.upload(c)
    var out = TensorImpl[BF].alloc(B * DIM)
    var go = TensorImpl[BF].alloc(B * DIM)
    var gi = TensorImpl[BF].alloc(B * DIM)
    out.upload(c)
    go.upload(c)
    gi.upload(c)
    m.forward["gpu", B](TensorRefs[1, _, BF](x), out, Optional(c))
    m.vjp["gpu", B](
        TensorRefs[1, _, BF](x), go, TensorRefs[1, _, BF](gi), Optional(c)
    )
    c.synchronize()
    out.download(c)

    var re = _rel_err(outr, out, B * DIM)
    print("  bf16-flow fwd+vjp ran; forward rel err vs fp32:", re)
    assert_true(re < 0.05, "GELU bf16-flow forward within bf16 tolerance")
    print("  ok")


def main() raises:
    print("=" * 60)
    print("LayerNorm + GELU bf16-FLOW gate (fp32-internal)")
    print("=" * 60)
    comptime if not has_accelerator():
        print("No accelerator — skipping (bf16-flow is a GPU gate)")
        return
    test_layer_norm_bf16_flow()
    test_gelu_bf16_flow()
    print("ALL LAYER_NORM + GELU bf16-FLOW GATES PASSED")
