"""Phase 3 probe: does DType.bfloat16 work end-to-end on Apple Metal?

If this test compiles + runs + reports max-diff <= 2^-7 (bf16 ULP),
Phase 3 can use the cast-around-matmul pattern. If it fails, Phase 3
becomes NVIDIA-only.

Steps:
1. Allocate fp32 host buffer with known values.
2. Allocate fp32 device buffer + bf16 device buffer.
3. Launch `_fp32_to_bf16_kernel` reading fp32, writing bf16.
4. Launch `_bf16_to_fp32_kernel` reading bf16, writing fp32.
5. Copy fp32 result back; verify max-diff <= 1/128 (~7-bit mantissa).
"""

from std.math import abs as fabs
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.testing import assert_true
from noeira.core.tiles import RowMajor, Tile, tile


comptime FP32 = DType.float32
comptime BF16 = DType.bfloat16


def _fp32_to_bf16_kernel[
    N: Int,
](
    src: Tile[FP32, N],
    dst: Tile[BF16, N],
):
    var i = Int(global_idx.x)
    if i < N:
        var x = rebind[Scalar[FP32]](src[i])
        dst[i] = x.cast[BF16]()


def _bf16_to_fp32_kernel[
    N: Int,
](
    src: Tile[BF16, N],
    dst: Tile[FP32, N],
):
    var i = Int(global_idx.x)
    if i < N:
        var x = rebind[Scalar[BF16]](src[i])
        dst[i] = x.cast[FP32]()


def test_bf16_roundtrip() raises:
    comptime N = 256

    var ctx = DeviceContext()

    var host_in  = ctx.enqueue_create_host_buffer[FP32](N)
    var host_out = ctx.enqueue_create_host_buffer[FP32](N)
    ctx.synchronize()

    # Mix of small/large/negative/zero values — bf16 should handle all of these.
    for i in range(N):
        host_in.unsafe_ptr()[i] = Scalar[FP32](0.001) * Scalar[FP32](Float32(i) - Float32(N // 2))

    var dev_fp32_in  = ctx.enqueue_create_buffer[FP32](N)
    var dev_bf16     = ctx.enqueue_create_buffer[BF16](N)
    var dev_fp32_out = ctx.enqueue_create_buffer[FP32](N)
    ctx.enqueue_copy(dev_fp32_in, host_in)

    comptime layout_fp32 = RowMajor[N]
    comptime layout_bf16 = RowMajor[N]

    var lt_in   = tile[N](dev_fp32_in)
    var lt_bf16 = tile[N](dev_bf16)
    var lt_out  = tile[N](dev_fp32_out)

    comptime TPB = 64
    comptime BLOCKS = (N + TPB - 1) // TPB

    comptime down = _fp32_to_bf16_kernel[N]
    ctx.enqueue_function[down](
        lt_in, lt_bf16, grid_dim=BLOCKS, block_dim=TPB,
    )

    comptime up = _bf16_to_fp32_kernel[N]
    ctx.enqueue_function[up](
        lt_bf16, lt_out, grid_dim=BLOCKS, block_dim=TPB,
    )

    ctx.enqueue_copy(host_out, dev_fp32_out)
    ctx.synchronize()

    # bf16 has 7 mantissa bits — relative error up to ~1/256, but for our
    # small values the absolute error should stay under 1/128 of the input.
    var max_diff: Scalar[FP32] = 0.0
    var max_rel:  Scalar[FP32] = 0.0
    for i in range(N):
        var orig = host_in.unsafe_ptr()[i]
        var rt   = host_out.unsafe_ptr()[i]
        var diff = fabs(orig - rt)
        if diff > max_diff:
            max_diff = diff
        if fabs(orig) > 1e-6:
            var rel = diff / fabs(orig)
            if rel > max_rel:
                max_rel = rel

    print("bf16 round-trip on Apple Metal:")
    print("  N           = " + String(N))
    print("  max-abs-err = " + String(max_diff))
    print("  max-rel-err = " + String(max_rel))

    # bf16 has ~3 decimal digits of precision -> relative error up to ~4e-3.
    assert_true(max_rel < 0.01,
        "bf16 round-trip rel error " + String(max_rel) + " > 0.01 — Apple bf16 broken?")


def main() raises:
    print("=" * 60)
    print("nn Phase 3 probe — DType.bfloat16 on Apple Metal")
    print("=" * 60)
    test_bf16_roundtrip()
    print("=" * 60)
    print("PROBE PASSED — bf16 is usable on Apple")
    print("=" * 60)
