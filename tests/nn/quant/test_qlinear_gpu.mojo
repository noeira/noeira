"""QLinear on a GPU: against a Linear holding deq(W), and its scratch.

    pixi run -e apple  mojo run -I . tests/nn/quant/test_qlinear_gpu.mojo   # Metal: dequantize path
    pixi run -e jetson mojo run -I . --target-accelerator sm_87 \\
        tests/nn/quant/test_qlinear_gpu.mojo                                # Orin: MAX's int4 GEMM

  [1] SmolVLA's four text shapes at T = 135 (the prefill's tokens) and 16:
      forward and grad_input against `Linear` over the same deq(W). Metal and
      `NN_Q4_PATH=deq` run fp32 GEMMs on both sides: 1e-5. NVIDIA's forward
      is MAX's int4 GEMM with bf16 activations, and both grad_inputs are TF32
      cuBLAS GEMMs at these sizes: the plan's 1e-2
  [2] scratch: a second forward and vjp allocate nothing (every scratch
      buffer keeps its device address), and two layers sharing one
      dequantize scratch use that buffer and still compute the same outputs
"""

from std.sys import has_nvidia_gpu_accelerator
from std.testing import assert_true
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Deterministic
from noeira.nn.quant.qlinear import QLinear, Q4_PATH
from tests.nn.quant.qlinear_common import check_shape, gaussian, put, to_host, rel

comptime INT4 = has_nvidia_gpu_accelerator() and Q4_PATH != "deq"
comptime TOL = 1e-2 if has_nvidia_gpu_accelerator() else 1e-5


def addr(ref t: Tensor) -> Int:
    return Int(t.dev.value().unsafe_ptr()) if t.dev else 0


def shapes(ctx: DeviceContext) raises:
    var c = Optional[DeviceContext](ctx)
    var worst_f = Float64(0)
    var worst_d = Float64(0)
    var vs = List[Float64]()
    var a0 = check_shape["gpu", 960, 960, 135](c, String("q / o   "))
    var a1 = check_shape["gpu", 960, 320, 135](c, String("k / v   "))
    var a2 = check_shape["gpu", 960, 2560, 135](c, String("gate/up "))
    var a3 = check_shape["gpu", 2560, 960, 135](c, String("down    "))
    var b0 = check_shape["gpu", 960, 960, 16](c, String("q / o   "))
    var b3 = check_shape["gpu", 2560, 960, 16](c, String("down    "))
    for v in [a0.copy(), a1.copy(), a2.copy(), a3.copy(), b0.copy(), b3.copy()]:
        worst_f = max(worst_f, max(v.fwd, v.fwd_vs_exact))
        worst_d = max(worst_d, v.dx)
    print("  [1] worst forward", worst_f, "| worst dx", worst_d, "(band", TOL,
          ", int4 GEMM:", INT4, ")")
    assert_true(worst_f < TOL and worst_d < TOL, "QLinear differs from Linear over deq(W)")


def scratch(ctx: DeviceContext) raises:
    comptime IN = 960
    comptime OUT = 2560
    comptime T = 135
    var c = Optional[DeviceContext](ctx)
    var w = gaussian(OUT * IN, 0.05)
    var ql = QLinear[IN, OUT].make["gpu", Deterministic](c)
    ql.set_weight["gpu"](w, c)
    var xs = gaussian(T * IN, 1.0)
    var gs = gaussian(T * OUT, 1.0)
    var x = Tensor()
    var g = Tensor()
    put(x, xs, "gpu", c)
    put(g, gs, "gpu", c)
    var y = Tensor()
    var gx = Tensor()
    ql.forward["gpu", T](TensorRefs[1](x), y, c)
    ql.vjp["gpu", T](TensorRefs[1](x), g, TensorRefs[1](gx), c)
    ctx.synchronize()
    var before = List[Int]()
    before.append(addr(ql.deq))
    before.append(Int(ql.xb.dev.value().unsafe_ptr()) if ql.xb.dev else 0)
    before.append(Int(ql.yb.dev.value().unsafe_ptr()) if ql.yb.dev else 0)
    before.append(addr(y))
    before.append(addr(gx))
    var y1 = to_host(y, T * OUT, c)
    for _ in range(3):
        ql.forward["gpu", T](TensorRefs[1](x), y, c)
        ql.vjp["gpu", T](TensorRefs[1](x), g, TensorRefs[1](gx), c)
    ctx.synchronize()
    var after = List[Int]()
    after.append(addr(ql.deq))
    after.append(Int(ql.xb.dev.value().unsafe_ptr()) if ql.xb.dev else 0)
    after.append(Int(ql.yb.dev.value().unsafe_ptr()) if ql.yb.dev else 0)
    after.append(addr(y))
    after.append(addr(gx))
    var moved = 0
    for i in range(len(before)):
        if before[i] != after[i]:
            moved += 1
    var y2 = to_host(y, T * OUT, c)
    var same = rel(y2, y1, T * OUT)

    # Two layers on one shared dequantize scratch, sized for the larger.
    var shared = ctx.enqueue_create_buffer[DT](OUT * IN)
    var q2 = QLinear[IN, OUT].make["gpu", Deterministic](c)
    q2.set_weight["gpu"](w, c)
    q2.share_scratch(shared, OUT * IN)
    var w3 = gaussian(960 * 960, 0.05)
    var q3 = QLinear[960, 960].make["gpu", Deterministic](c)
    q3.set_weight["gpu"](w3, c)
    q3.share_scratch(shared, OUT * IN)
    var y3 = Tensor()
    var gx3 = Tensor()
    var gx2 = Tensor()
    var g3 = Tensor()
    var gs3 = gaussian(T * 960, 1.0)
    put(g3, gs3, "gpu", c)
    q2.vjp["gpu", T](TensorRefs[1](x), g, TensorRefs[1](gx2), c)
    q3.vjp["gpu", T](TensorRefs[1](x), g3, TensorRefs[1](gx3), c)
    q2.forward["gpu", T](TensorRefs[1](x), y3, c)
    ctx.synchronize()
    var sp = Int(shared.unsafe_ptr())
    var uses = (addr(q2.deq) == sp) and (addr(q3.deq) == sp)
    var gx_ref = to_host(gx, T * IN, c)
    var gx2_h = to_host(gx2, T * IN, c)
    var y3_h = to_host(y3, T * OUT, c)
    var d_shared = max(rel(gx2_h, gx_ref, T * IN), rel(y3_h, y1, T * OUT))
    print("  [2] scratch: device addresses moved after the first call:", moved,
          "of", len(before), "| outputs repeat:", same,
          "| shared buffer used by both layers:", uses,
          "| outputs on the shared scratch vs own:", d_shared)
    assert_true(moved == 0, "a scratch buffer was reallocated after the first call")
    assert_true(same == 0.0, "a repeated forward changed its output")
    assert_true(uses, "the shared scratch is not the one the layers use")
    assert_true(d_shared == 0.0, "the shared scratch changed an output")


def main() raises:
    var ctx = DeviceContext()
    print("QLinear on", ctx.name(), "(int4 GEMM:", INT4, ")")
    shapes(ctx)
    scratch(ctx)
    print("PASSED")
