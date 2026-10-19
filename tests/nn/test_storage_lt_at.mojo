"""Tensor.lt_at — offset sub-view correctness + create_sub_buffer overhead bench.

`lt_at[target, layout](offset)` is the sanctioned replacement for
`LayoutTensor[..MutAnyOrigin](buf.dev.value().unsafe_ptr() + offset)` used by the
stacked-ensemble agents (MBPO dynamics bounds, rollout buffers). It must:
  1. slice the CORRECT [offset : offset+DIM] region (per-member),
  2. NOT corrupt neighbouring members,
on CPU (Span slice) and GPU (create_sub_buffer).

It also benchmarks the create_sub_buffer overhead vs the old unsafe_ptr+offset
view-build, in the realistic "build the per-member view → launch a kernel" loop —
the exact shape the MBPO sites use. (Apple Metal here = ballpark; NVIDIA is the
real perf surface — run there before drawing perf conclusions.)

Run:
  pixi run mojo run -I . tests/nn/test_storage_lt_at.mojo
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.time import perf_counter_ns
from std.testing import assert_true
from noeira.core.tiles import Tile, tile

from noeira.nn.constants import DT, TPB
from noeira.nn.core.tensor import Tensor


def _add_const_kernel[
    DIM: Int
](
    v: Tile[DT, DIM],
    c: Scalar[DT],
):
    var i = Int(global_idx.x)
    if i < DIM:
        v[i] = rebind[Scalar[DT]](v[i]) + c


def main() raises:
    print("=" * 64)
    print("Tensor.lt_at — offset sub-view correctness + create_sub_buffer bench")
    print("=" * 64)
    comptime N = 4
    comptime DIM = 8
    comptime TOTAL = N * DIM

    var gpu_ok = False
    with DeviceContext() as ctx:
        # ---- GPU correctness: each member's [DIM] sub-view gets its own const,
        # neighbours untouched (proves offset slicing + no cross-corruption) ----
        var tg = Tensor.alloc_gpu(ctx, TOTAL)  # zero-filled on device
        comptime nblk = (DIM + TPB - 1) // TPB
        for m in range(N):
            var v = tg.tt_at["gpu", DIM](m * DIM)
            ctx.enqueue_function[_add_const_kernel[DIM]](
                v, Scalar[DT](m + 1), grid_dim=nblk, block_dim=TPB
            )
        tg.download(ctx)
        gpu_ok = True
        for m in range(N):
            for j in range(DIM):
                if tg.data[m * DIM + j] != Scalar[DT](m + 1):
                    gpu_ok = False
        print("  GPU lt_at offset correctness:", "OK" if gpu_ok else "FAIL")

        # ---- Bench: old unsafe_ptr+offset vs new lt_at, build-view → launch ----
        comptime BN = 8        # members per round
        comptime BDIM = 64
        comptime BTOTAL = BN * BDIM
        comptime ITERS = 3000
        comptime WARMUP = 300
        comptime bnblk = (BDIM + TPB - 1) // TPB
        comptime tiny = Scalar[DT](1e-9)
        var tb = Tensor.alloc_gpu(ctx, BTOTAL)

        # OLD: LayoutTensor over unsafe_ptr()+offset
        for _ in range(WARMUP):
            for m in range(BN):
                var v = tile[BDIM](tb.dev.value().unsafe_ptr().unsafe_offset(m * BDIM).as_unsafe_any_origin())
                ctx.enqueue_function[_add_const_kernel[BDIM]](
                    v, tiny, grid_dim=bnblk, block_dim=TPB
                )
        ctx.synchronize()
        var t0 = perf_counter_ns()
        for _ in range(ITERS):
            for m in range(BN):
                var v = tile[BDIM](tb.dev.value().unsafe_ptr().unsafe_offset(m * BDIM).as_unsafe_any_origin())
                ctx.enqueue_function[_add_const_kernel[BDIM]](
                    v, tiny, grid_dim=bnblk, block_dim=TPB
                )
        ctx.synchronize()
        var t1 = perf_counter_ns()

        # NEW: lt_at (create_sub_buffer)
        for _ in range(WARMUP):
            for m in range(BN):
                var v = tb.tt_at["gpu", BDIM](m * BDIM)
                ctx.enqueue_function[_add_const_kernel[BDIM]](
                    v, tiny, grid_dim=bnblk, block_dim=TPB
                )
        ctx.synchronize()
        var t2 = perf_counter_ns()
        for _ in range(ITERS):
            for m in range(BN):
                var v = tb.tt_at["gpu", BDIM](m * BDIM)
                ctx.enqueue_function[_add_const_kernel[BDIM]](
                    v, tiny, grid_dim=bnblk, block_dim=TPB
                )
        ctx.synchronize()
        var t3 = perf_counter_ns()

        var launches = Float64(ITERS * BN)
        var old_ns = Float64(t1 - t0) / launches
        var new_ns = Float64(t3 - t2) / launches
        print("  --- bench (build view + kernel launch; per-launch) ---")
        print("    old unsafe_ptr+offset :", old_ns, "ns/launch")
        print("    new lt_at(sub_buffer) :", new_ns, "ns/launch")
        print("    create_sub_buffer overhead :", new_ns - old_ns, "ns/launch")
        print("    (Apple Metal ballpark — confirm on NVIDIA before concluding)")

    assert_true(gpu_ok, "lt_at offset sub-view correctness")
    print("LT_AT OK")
