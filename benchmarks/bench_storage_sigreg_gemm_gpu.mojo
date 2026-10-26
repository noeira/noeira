"""Group A (re-audit) microbench: storage SIGReg projection GEMMs —
hand-rolled scalar GEMM (one thread per output element, scalar K-loop, the
production code before this change) vs `max_matmul` (tensor-core GEMM).
Self-contained A/B in one process, forward AND backward.

SIGReg (Epps-Pulley Gaussianity loss, the LeWM JEPA anti-collapse term) has two
matmul-shaped kernels that were hand-rolled scalar contractions:

  forward  `_sr_project`:  Z[B*T, P]    = X[B*T, D]    @ A[D, P]          (K = D)
  backward `_sr_matmul_a`: gIn[B*T, D]  = dLdz[B*T, P] @ A[D, P]ᵀ          (K = P)

Both are clean GEMMs with contiguous operands (the input slab [B, T*D] is
[B*T, D] row-major; cache_z [B, T*P] is [B*T, P]); backward consumes A as
`transpose_b=True`. The hand-rolled forward additionally reads `a_t[d, p]` with
stride P (uncoalesced) inside the scalar K-loop. This is the Group A pattern
(A1/A4: embedding/LSTM saw 6-12× on NVIDIA) — measure-first whether it lands.

Shapes = real LeWM configs + one scaled batch:
  pusht  B16·T6, D=96,  P=1024  (M=B·T=96)   — paper value, P ≫ D
  pong   B16·T6, D=128, P=256   (M=96)
  scaled B256·T6, D=128, P=512  (M=1536)     — larger M, tensor-core regime

Run (NVIDIA): pixi run -e nvidia mojo run -I . benchmarks/bench_storage_sigreg_gemm_gpu.mojo
Run (Apple):  pixi run -e apple  mojo run -I . benchmarks/bench_storage_sigreg_gemm_gpu.mojo
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.time import perf_counter_ns
from layout import TileTensor, row_major
from noeira.core.tiles import Tile, tile
from linalg.matmul import matmul as max_matmul

comptime DT = DType.float32
comptime TPB = 128


# ── naive hand-rolled scalar GEMMs (production before this change) ─────────
def _sr_project[
    BATCH: Int, T: Int, D: Int, P: Int
](
    input_t: Tile[DT, BATCH, T * D],
    a_t: Tile[DT, D, P],
    cache_t: Tile[DT, BATCH, T * P],
):
    var idx = Int(global_idx.x)
    if idx >= BATCH * T * P:
        return
    var p_idx = idx % P
    var t_idx = (idx // P) % T
    var b = idx // (T * P)
    var z = Scalar[DT](0)
    for d in range(D):
        var xi = rebind[Scalar[DT]](input_t[b, t_idx * D + d])
        var aval = rebind[Scalar[DT]](a_t[d, p_idx])
        z += xi * aval
    cache_t[b, t_idx * P + p_idx] = z


def _sr_matmul_a[
    BATCH: Int, T: Int, D: Int, P: Int
](
    dLdz_t: Tile[DT, BATCH, T * P],
    a_t: Tile[DT, D, P],
    grad_input_t: Tile[DT, BATCH, T * D],
):
    var idx = Int(global_idx.x)
    if idx >= BATCH * T * D:
        return
    var d_idx = idx % D
    var t_idx = (idx // D) % T
    var b = idx // (T * D)
    var acc = Scalar[DT](0)
    for p in range(P):
        var dL = rebind[Scalar[DT]](dLdz_t[b, t_idx * P + p])
        var aval = rebind[Scalar[DT]](a_t[d_idx, p])
        acc += aval * dL
    grad_input_t[b, t_idx * D + d_idx] = acc


# ── forward A/B:  Z[M, P] = X[M, D] @ A[D, P],  M = B·T ─────────────────────
def _time_fwd[
    B: Int, T: Int, D: Int, P: Int, USE_MM: Bool, WARMUP: Int, ITERS: Int
](ctx: DeviceContext, label: StaticString) raises:
    comptime M = B * T
    var x = ctx.enqueue_create_buffer[DT](B * T * D)
    var a = ctx.enqueue_create_buffer[DT](D * P)
    var z = ctx.enqueue_create_buffer[DT](B * T * P)
    _ = x.enqueue_fill(Scalar[DT](0.01)); _ = a.enqueue_fill(Scalar[DT](0.02))
    _ = z.enqueue_fill(Scalar[DT](0.0))
    var us = Float64(0)
    comptime if USE_MM:
        var xv = TileTensor(x, row_major[M, D]())
        var av = TileTensor(a, row_major[D, P]())
        var zv = TileTensor(z, row_major[M, P]())
        comptime for _ in range(WARMUP):
            max_matmul[target="gpu"](zv, xv, av, ctx)
        ctx.synchronize()
        var t0 = perf_counter_ns()
        comptime for _ in range(ITERS):
            max_matmul[target="gpu"](zv, xv, av, ctx)
        ctx.synchronize()
        us = Float64(perf_counter_ns() - t0) / Float64(ITERS) / 1000.0
    else:
        var xl = tile[B, T * D](x)
        var al = tile[D, P](a)
        var zl = tile[B, T * P](z)
        comptime nb = (B * T * P + TPB - 1) // TPB
        comptime for _ in range(WARMUP):
            ctx.enqueue_function[_sr_project[B, T, D, P]](
                xl, al, zl, grid_dim=nb, block_dim=TPB)
        ctx.synchronize()
        var t0 = perf_counter_ns()
        comptime for _ in range(ITERS):
            ctx.enqueue_function[_sr_project[B, T, D, P]](
                xl, al, zl, grid_dim=nb, block_dim=TPB)
        ctx.synchronize()
        us = Float64(perf_counter_ns() - t0) / Float64(ITERS) / 1000.0
    var tflop = 2.0 * Float64(M) * Float64(P) * Float64(D) / (us * 1e6) / 1e3
    print("  ", label, " M=", M, " D=", D, " P=", P, " | ",
          us, "us/iter ", tflop, "TFLOP/s")


# ── backward A/B:  gIn[M, D] = dLdz[M, P] @ A[D, P]ᵀ,  M = B·T ──────────────
def _time_bwd[
    B: Int, T: Int, D: Int, P: Int, USE_MM: Bool, WARMUP: Int, ITERS: Int
](ctx: DeviceContext, label: StaticString) raises:
    comptime M = B * T
    var dl = ctx.enqueue_create_buffer[DT](B * T * P)
    var a = ctx.enqueue_create_buffer[DT](D * P)
    var gi = ctx.enqueue_create_buffer[DT](B * T * D)
    _ = dl.enqueue_fill(Scalar[DT](0.01)); _ = a.enqueue_fill(Scalar[DT](0.02))
    _ = gi.enqueue_fill(Scalar[DT](0.0))
    var us = Float64(0)
    comptime if USE_MM:
        var dlv = TileTensor(dl, row_major[M, P]())
        var av = TileTensor(a, row_major[D, P]())
        var giv = TileTensor(gi, row_major[M, D]())
        comptime for _ in range(WARMUP):
            max_matmul[transpose_b=True, target="gpu"](giv, dlv, av, ctx)
        ctx.synchronize()
        var t0 = perf_counter_ns()
        comptime for _ in range(ITERS):
            max_matmul[transpose_b=True, target="gpu"](giv, dlv, av, ctx)
        ctx.synchronize()
        us = Float64(perf_counter_ns() - t0) / Float64(ITERS) / 1000.0
    else:
        var dll = tile[B, T * P](dl)
        var al = tile[D, P](a)
        var gil = tile[B, T * D](gi)
        comptime nb = (B * T * D + TPB - 1) // TPB
        comptime for _ in range(WARMUP):
            ctx.enqueue_function[_sr_matmul_a[B, T, D, P]](
                dll, al, gil, grid_dim=nb, block_dim=TPB)
        ctx.synchronize()
        var t0 = perf_counter_ns()
        comptime for _ in range(ITERS):
            ctx.enqueue_function[_sr_matmul_a[B, T, D, P]](
                dll, al, gil, grid_dim=nb, block_dim=TPB)
        ctx.synchronize()
        us = Float64(perf_counter_ns() - t0) / Float64(ITERS) / 1000.0
    var tflop = 2.0 * Float64(M) * Float64(D) * Float64(P) / (us * 1e6) / 1e3
    print("  ", label, " M=", M, " P=", P, " D=", D, " | ",
          us, "us/iter ", tflop, "TFLOP/s")


def _ab[
    B: Int, T: Int, D: Int, P: Int, WARMUP: Int, ITERS: Int
](ctx: DeviceContext) raises:
    _time_fwd[B, T, D, P, False, WARMUP, ITERS](ctx, "fwd naive   ")
    _time_fwd[B, T, D, P, True, WARMUP, ITERS](ctx, "fwd max_mm  ")
    _time_bwd[B, T, D, P, False, WARMUP, ITERS](ctx, "bwd naive   ")
    _time_bwd[B, T, D, P, True, WARMUP, ITERS](ctx, "bwd max_mm  ")


def main() raises:
    var ctx = DeviceContext()
    print("SIGReg projection GEMMs — naive scalar vs max_matmul [fp32] (A-redux)")
    print("fwd Z=X@A (K=D), bwd gIn=dLdz@Aᵀ (K=P). max_mm/naive >1 = tensor-core win.")
    print("=" * 70)
    print("pusht  B16 T6 D96 P1024:")
    _ab[16, 6, 96, 1024, 10, 300](ctx)
    print("pong   B16 T6 D128 P256:")
    _ab[16, 6, 128, 256, 10, 300](ctx)
    print("scaled B256 T6 D128 P512:")
    _ab[256, 6, 128, 512, 10, 200](ctx)
    print("=" * 70)
    print("max_mm > naive = the conversion is a win (expect multiplicative at P=1024).")
