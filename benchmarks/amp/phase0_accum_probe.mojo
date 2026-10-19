"""Phase-0 probe: does linalg.matmul accumulate bf16 inputs in fp32?

Decisive test — a wide all-ones GEMM (1 x K) @ (K x 1):
  - fp32 accumulator -> 4096.0 (exactly representable in bf16 on store).
  - bf16 SERIAL accumulator -> stagnates ~256 (bf16 can't represent
    consecutive integers past 256; adding 1 to a >256 running sum rounds away).

A result of ~4096 => accumulation is fp32 (or exact) = GOOD (AMP prereq met).
A result of ~256  => bf16 serial accumulator = BAD (need a fp32-output epilogue).

Three cases:
  (1) bf16 in -> bf16 out   (the CURRENT AMP path in linear.mojo)
  (2) bf16 in -> fp32 out   (does accum follow the OUTPUT dtype?)
  (3) fp32 in -> fp32 out   (control; must be exactly 4096)
"""

from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from linalg.matmul import matmul as max_matmul

from noeira.nn.core.tensor import Tensor, TensorImpl

comptime BF16 = DType.bfloat16
comptime K = 4096


def _ones_bf16(ctx: DeviceContext, n: Int) raises -> TensorImpl[BF16]:
    var t = TensorImpl[BF16].alloc(n)
    for i in range(n):
        t.data[i] = Scalar[BF16](1)
    t.upload(ctx)
    return t^


def _ones_f32(ctx: DeviceContext, n: Int) raises -> Tensor:
    var t = Tensor.alloc(n)
    for i in range(n):
        t.data[i] = Scalar[DType.float32](1)
    t.upload(ctx)
    return t^


def case1_bf16_bf16(ctx: DeviceContext) raises:
    var x = _ones_bf16(ctx, K)
    var w = _ones_bf16(ctx, K)
    var o = TensorImpl[BF16]()
    o.ensure_gpu(ctx, 1)
    var x_v = TileTensor(x.dev.value(), row_major[1, K]())
    var w_v = TileTensor(w.dev.value(), row_major[K, 1]())
    var o_v = TileTensor(o.dev.value(), row_major[1, 1]())
    max_matmul[target="gpu"](o_v, x_v, w_v, ctx)
    o.download(ctx)
    print("(1) bf16 in -> bf16 out :", Float64(o.data[0].cast[DType.float64]()))


def case2_bf16_f32(ctx: DeviceContext) raises:
    var x = _ones_bf16(ctx, K)
    var w = _ones_bf16(ctx, K)
    var o = Tensor()
    o.ensure_gpu(ctx, 1)
    var x_v = TileTensor(x.dev.value(), row_major[1, K]())
    var w_v = TileTensor(w.dev.value(), row_major[K, 1]())
    var o_v = TileTensor(o.dev.value(), row_major[1, 1]())
    max_matmul[target="gpu"](o_v, x_v, w_v, ctx)
    o.download(ctx)
    print("(2) bf16 in -> fp32 out :", Float64(o.data[0].cast[DType.float64]()))


def case3_f32_f32(ctx: DeviceContext) raises:
    var x = _ones_f32(ctx, K)
    var w = _ones_f32(ctx, K)
    var o = Tensor()
    o.ensure_gpu(ctx, 1)
    var x_v = TileTensor(x.dev.value(), row_major[1, K]())
    var w_v = TileTensor(w.dev.value(), row_major[K, 1]())
    var o_v = TileTensor(o.dev.value(), row_major[1, 1]())
    max_matmul[target="gpu"](o_v, x_v, w_v, ctx)
    o.download(ctx)
    print("(3) fp32 in -> fp32 out :", Float64(o.data[0].cast[DType.float64]()))


def main() raises:
    print("K =", K, "  expect 4096 if fp32-accum, ~256 if bf16-serial-accum")
    var ctx = DeviceContext()
    case3_f32_f32(ctx)
    case1_bf16_bf16(ctx)
    case2_bf16_f32(ctx)
