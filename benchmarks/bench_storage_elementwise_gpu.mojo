"""Group D (D4) microbench: storage Elementwise GPU — scalar (one elem/thread,
current production) vs vec4 (width-4 load/compute/store). Self-contained A/B.

The GPU elementwise kernel is scalar today (CPU path is already SIMD). Per the
D0 lesson, vectorization only helps if the scalar baseline is NOT already
bandwidth-saturated — which splits by op cost:
  • cheap op (ReLU = max(0,x)): pure data movement → headroom check; expect wash
    if naive already near peak HBM.
  • transcendental op (Tanh): compute per element is heavy → may be compute-bound,
    where wider loads give ILP/occupancy headroom.

Reported GB/s on ReLU is the honest "is there headroom" signal (compare to the
GPU's peak HBM). Shapes: typical RL activation tensors [BATCH, HIDDEN] folded.

Run (NVIDIA): pixi run -e nvidia mojo run -I . benchmarks/bench_storage_elementwise_gpu.mojo
Run (Apple):  pixi run -e apple  mojo run -I . benchmarks/bench_storage_elementwise_gpu.mojo
"""

from std.math import tanh
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.time import perf_counter_ns
from noeira.core.tiles import Tile, tile

comptime DT = DType.float32
comptime TPB = 256
comptime VEC = 4


# ── ReLU (cheap) ──────────────────────────────────────────────────────────
def _relu_scalar[M: Int](
    x: Tile[DT, M],
    o: Tile[DT, M],
):
    var i = Int(global_idx.x)
    if i < M:
        var v = rebind[Scalar[DT]](x[i])
        o[i] = max(v, Scalar[DT](0))


def _relu_vec[M: Int](
    x: Tile[DT, M],
    o: Tile[DT, M],
):
    var i = Int(global_idx.x) * VEC
    if i < M:
        var v = x.ptr.load[width=VEC](i)
        o.ptr.store(i, max(v, SIMD[DT, VEC](0)))


# ── Tanh (transcendental) ─────────────────────────────────────────────────
def _tanh_scalar[M: Int](
    x: Tile[DT, M],
    o: Tile[DT, M],
):
    var i = Int(global_idx.x)
    if i < M:
        o[i] = tanh(rebind[Scalar[DT]](x[i]))


def _tanh_vec[M: Int](
    x: Tile[DT, M],
    o: Tile[DT, M],
):
    var i = Int(global_idx.x) * VEC
    if i < M:
        o.ptr.store(i, tanh(x.ptr.load[width=VEC](i)))


def _time[
    M: Int, KIND: Int, WARMUP: Int, ITERS: Int
](ctx: DeviceContext, label: StaticString) raises:
    # KIND: 0 relu_scalar, 1 relu_vec, 2 tanh_scalar, 3 tanh_vec
    var x = ctx.enqueue_create_buffer[DT](M)
    var o = ctx.enqueue_create_buffer[DT](M)
    _ = x.enqueue_fill(Scalar[DT](0.5)); _ = o.enqueue_fill(Scalar[DT](0))
    var xl = tile[M](x)
    var ol = tile[M](o)
    comptime nb_s = (M + TPB - 1) // TPB
    comptime nb_v = (M // VEC + TPB - 1) // TPB
    var us = Float64(0)

    comptime if KIND == 0:
        comptime for _ in range(WARMUP):
            ctx.enqueue_function[_relu_scalar[M]](xl, ol, grid_dim=nb_s, block_dim=TPB)
        ctx.synchronize()
        var t0 = perf_counter_ns()
        comptime for _ in range(ITERS):
            ctx.enqueue_function[_relu_scalar[M]](xl, ol, grid_dim=nb_s, block_dim=TPB)
        ctx.synchronize()
        us = Float64(perf_counter_ns() - t0) / Float64(ITERS) / 1000.0
    elif KIND == 1:
        comptime for _ in range(WARMUP):
            ctx.enqueue_function[_relu_vec[M]](xl, ol, grid_dim=nb_v, block_dim=TPB)
        ctx.synchronize()
        var t0 = perf_counter_ns()
        comptime for _ in range(ITERS):
            ctx.enqueue_function[_relu_vec[M]](xl, ol, grid_dim=nb_v, block_dim=TPB)
        ctx.synchronize()
        us = Float64(perf_counter_ns() - t0) / Float64(ITERS) / 1000.0
    elif KIND == 2:
        comptime for _ in range(WARMUP):
            ctx.enqueue_function[_tanh_scalar[M]](xl, ol, grid_dim=nb_s, block_dim=TPB)
        ctx.synchronize()
        var t0 = perf_counter_ns()
        comptime for _ in range(ITERS):
            ctx.enqueue_function[_tanh_scalar[M]](xl, ol, grid_dim=nb_s, block_dim=TPB)
        ctx.synchronize()
        us = Float64(perf_counter_ns() - t0) / Float64(ITERS) / 1000.0
    else:
        comptime for _ in range(WARMUP):
            ctx.enqueue_function[_tanh_vec[M]](xl, ol, grid_dim=nb_v, block_dim=TPB)
        ctx.synchronize()
        var t0 = perf_counter_ns()
        comptime for _ in range(ITERS):
            ctx.enqueue_function[_tanh_vec[M]](xl, ol, grid_dim=nb_v, block_dim=TPB)
        ctx.synchronize()
        us = Float64(perf_counter_ns() - t0) / Float64(ITERS) / 1000.0

    var gb = 2.0 * Float64(M) * 4.0 / 1e9   # read + write
    print("  ", label, " M=", M, " | ", us, "us/iter ",
          gb / (us / 1e6) / 1e3, "TB/s")


def _ab[M: Int, WARMUP: Int, ITERS: Int](ctx: DeviceContext) raises:
    _time[M, 0, WARMUP, ITERS](ctx, "relu_scalar")
    _time[M, 1, WARMUP, ITERS](ctx, "relu_vec4  ")
    _time[M, 2, WARMUP, ITERS](ctx, "tanh_scalar")
    _time[M, 3, WARMUP, ITERS](ctx, "tanh_vec4  ")


def main() raises:
    var ctx = DeviceContext()
    print("Elementwise GPU — scalar vs vec4 [fp32] (D4). ReLU=cheap (BW headroom),")
    print("Tanh=transcendental (compute-bound?). GB/s near peak HBM = no headroom.")
    print("=" * 70)
    _ab[256 * 256, 5, 200](ctx)        # 64K  — small
    _ab[512 * 1024, 5, 200](ctx)       # 512K — SAC/PPO MLP hidden
    _ab[4096 * 1024, 5, 100](ctx)      # 4M   — large batch
    _ab[16 * 1024 * 1024, 5, 50](ctx)  # 16M  — bandwidth regime
    print("=" * 70)
    print("vec/scalar ~1.0 on ReLU at peak GB/s = bandwidth-bound (keep scalar);")
    print("tanh_vec > tanh_scalar = compute/ILP win on transcendental ops.")
