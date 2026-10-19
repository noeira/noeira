"""Group B (B2/B3) microbench: storage SpaceTimeTranspose + QKVToMajor GPU —
naive (one thread per element, div/mod per lane) vs vectorized run-copy
(width-VEC load/store over the contiguous innermost D/DIM run, div/mod once
per VEC). Self-contained A/B in one process.

Unlike B1's element-level transpose, both ops permute at a COARSE grid level
with the innermost dimension (D for STT, DIM for QKV) contiguous in BOTH the
source and destination layouts. For D/DIM ≥ warp size the naive scalar kernel
is therefore already coalesced; the only waste is per-element index math. The
vec variant tests whether that math (or sub-VEC memory transactions) is the
bottleneck — i.e. whether these are real wins or A3-style no-ops.

  STT:  out[b,(s*T+t)*D+d] = in[b,(t*S+s)*D+d]      (T,S,D) → (S,T,D)
  QKV:  out[b,g*SEQ*DIM+t*DIM+d] = in[b,t*3*DIM+g*DIM+d]   (token-major→qkv-major)

Shapes: Dreamer4 space/time grids (T,S,D); ViT/attention QKV (SEQ,DIM).

Run (NVIDIA): pixi run -e nvidia mojo run -I . benchmarks/bench_storage_permute_gpu.mojo
Run (Apple):  pixi run -e apple  mojo run -I . benchmarks/bench_storage_permute_gpu.mojo
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.time import perf_counter_ns
from noeira.core.tiles import Tile, tile

comptime DT = DType.float32
comptime TPB = 128
comptime VEC = 4


# ── SpaceTimeTranspose ────────────────────────────────────────────────────
def _stt_naive[
    BATCH: Int, T: Int, S: Int, D: Int
](
    src: Tile[DT, BATCH, T * S * D],
    dst: Tile[DT, BATCH, T * S * D],
):
    var gid = Int(global_idx.x)
    comptime total = BATCH * T * S * D
    if gid >= total:
        return
    var b = gid // (T * S * D)
    var rem = gid % (T * S * D)
    var d = rem % D
    var pos = rem // D
    var s = pos // T
    var t = pos % T
    dst[b, rem] = rebind[Scalar[DT]](src[b, (t * S + s) * D + d])


def _stt_vec[
    BATCH: Int, T: Int, S: Int, D: Int
](
    src: Tile[DT, BATCH, T * S * D],
    dst: Tile[DT, BATCH, T * S * D],
):
    # one thread per VEC-chunk of the contiguous d-run (D % VEC == 0)
    var gid = Int(global_idx.x)
    comptime total = BATCH * T * S * D // VEC
    if gid >= total:
        return
    comptime DV = D // VEC
    var b = gid // (T * S * DV)
    var rem = gid % (T * S * DV)
    var dv = rem % DV
    var pos = rem // DV
    var s = pos // T
    var t = pos % T
    var src_base = b * (T * S * D) + (t * S + s) * D + dv * VEC
    var dst_base = b * (T * S * D) + pos * D + dv * VEC
    var v = src.ptr.load[width=VEC](src_base)
    dst.ptr.store(dst_base, v)


# ── QKVToMajor ────────────────────────────────────────────────────────────
def _qkv_naive[
    BATCH: Int, SEQ: Int, DIM: Int
](
    src: Tile[DT, BATCH, 3 * SEQ * DIM],
    dst: Tile[DT, BATCH, 3 * SEQ * DIM],
):
    var gid = Int(global_idx.x)
    comptime total = BATCH * 3 * SEQ * DIM
    if gid >= total:
        return
    var b = gid // (3 * SEQ * DIM)
    var o = gid % (3 * SEQ * DIM)
    var g = o // (SEQ * DIM)
    var rem = o % (SEQ * DIM)
    var t = rem // DIM
    var d = rem % DIM
    dst[b, o] = rebind[Scalar[DT]](src[b, t * 3 * DIM + g * DIM + d])


def _qkv_vec[
    BATCH: Int, SEQ: Int, DIM: Int
](
    src: Tile[DT, BATCH, 3 * SEQ * DIM],
    dst: Tile[DT, BATCH, 3 * SEQ * DIM],
):
    var gid = Int(global_idx.x)
    comptime total = BATCH * 3 * SEQ * DIM // VEC
    if gid >= total:
        return
    comptime DV = DIM // VEC
    var b = gid // (3 * SEQ * DV)
    var o = gid % (3 * SEQ * DV)
    var g = o // (SEQ * DV)
    var rem = o % (SEQ * DV)
    var t = rem // DV
    var dv = rem % DV
    var src_base = b * (3 * SEQ * DIM) + t * 3 * DIM + g * DIM + dv * VEC
    var dst_base = b * (3 * SEQ * DIM) + (g * SEQ * DIM + t * DIM) + dv * VEC
    var v = src.ptr.load[width=VEC](src_base)
    dst.ptr.store(dst_base, v)


def _time_stt[
    BATCH: Int, T: Int, S: Int, D: Int, V: Bool, WARMUP: Int, ITERS: Int
](ctx: DeviceContext, label: StaticString) raises:
    comptime N = BATCH * T * S * D
    var s = ctx.enqueue_create_buffer[DT](N)
    var d = ctx.enqueue_create_buffer[DT](N)
    _ = s.enqueue_fill(Scalar[DT](0.01)); _ = d.enqueue_fill(Scalar[DT](0.0))
    var sl = tile[BATCH, T * S * D](s)
    var dl = tile[BATCH, T * S * D](d)
    var us = Float64(0)
    comptime if V:
        comptime nb = (N // VEC + TPB - 1) // TPB
        comptime for _ in range(WARMUP):
            ctx.enqueue_function[_stt_vec[BATCH, T, S, D]](sl, dl, grid_dim=nb, block_dim=TPB)
        ctx.synchronize()
        var t0 = perf_counter_ns()
        comptime for _ in range(ITERS):
            ctx.enqueue_function[_stt_vec[BATCH, T, S, D]](sl, dl, grid_dim=nb, block_dim=TPB)
        ctx.synchronize()
        us = Float64(perf_counter_ns() - t0) / Float64(ITERS) / 1000.0
    else:
        comptime nb = (N + TPB - 1) // TPB
        comptime for _ in range(WARMUP):
            ctx.enqueue_function[_stt_naive[BATCH, T, S, D]](sl, dl, grid_dim=nb, block_dim=TPB)
        ctx.synchronize()
        var t0 = perf_counter_ns()
        comptime for _ in range(ITERS):
            ctx.enqueue_function[_stt_naive[BATCH, T, S, D]](sl, dl, grid_dim=nb, block_dim=TPB)
        ctx.synchronize()
        us = Float64(perf_counter_ns() - t0) / Float64(ITERS) / 1000.0
    var gb = 2.0 * Float64(N) * 4.0 / 1e9
    print("  ", label, " B=", BATCH, " T=", T, " S=", S, " D=", D, " | ",
          us, "us/iter ", gb / (us / 1e6) / 1e3, "TB/s")


def _time_qkv[
    BATCH: Int, SEQ: Int, DIM: Int, V: Bool, WARMUP: Int, ITERS: Int
](ctx: DeviceContext, label: StaticString) raises:
    comptime N = BATCH * 3 * SEQ * DIM
    var s = ctx.enqueue_create_buffer[DT](N)
    var d = ctx.enqueue_create_buffer[DT](N)
    _ = s.enqueue_fill(Scalar[DT](0.01)); _ = d.enqueue_fill(Scalar[DT](0.0))
    var sl = tile[BATCH, 3 * SEQ * DIM](s)
    var dl = tile[BATCH, 3 * SEQ * DIM](d)
    var us = Float64(0)
    comptime if V:
        comptime nb = (N // VEC + TPB - 1) // TPB
        comptime for _ in range(WARMUP):
            ctx.enqueue_function[_qkv_vec[BATCH, SEQ, DIM]](sl, dl, grid_dim=nb, block_dim=TPB)
        ctx.synchronize()
        var t0 = perf_counter_ns()
        comptime for _ in range(ITERS):
            ctx.enqueue_function[_qkv_vec[BATCH, SEQ, DIM]](sl, dl, grid_dim=nb, block_dim=TPB)
        ctx.synchronize()
        us = Float64(perf_counter_ns() - t0) / Float64(ITERS) / 1000.0
    else:
        comptime nb = (N + TPB - 1) // TPB
        comptime for _ in range(WARMUP):
            ctx.enqueue_function[_qkv_naive[BATCH, SEQ, DIM]](sl, dl, grid_dim=nb, block_dim=TPB)
        ctx.synchronize()
        var t0 = perf_counter_ns()
        comptime for _ in range(ITERS):
            ctx.enqueue_function[_qkv_naive[BATCH, SEQ, DIM]](sl, dl, grid_dim=nb, block_dim=TPB)
        ctx.synchronize()
        us = Float64(perf_counter_ns() - t0) / Float64(ITERS) / 1000.0
    var gb = 2.0 * Float64(N) * 4.0 / 1e9
    print("  ", label, " B=", BATCH, " SEQ=", SEQ, " DIM=", DIM, " | ",
          us, "us/iter ", gb / (us / 1e6) / 1e3, "TB/s")


def main() raises:
    var ctx = DeviceContext()
    print("SpaceTimeTranspose GPU — naive vs vec4 run-copy [fp32] (B2)")
    print("=" * 66)
    _time_stt[32, 16, 64, 256, False, 5, 100](ctx, "naive")
    _time_stt[32, 16, 64, 256, True, 5, 100](ctx, "vec4 ")
    _time_stt[16, 8, 256, 512, False, 5, 50](ctx, "naive")
    _time_stt[16, 8, 256, 512, True, 5, 50](ctx, "vec4 ")
    _time_stt[16, 32, 32, 512, False, 5, 50](ctx, "naive")
    _time_stt[16, 32, 32, 512, True, 5, 50](ctx, "vec4 ")
    print("=" * 66)
    print("QKVToMajor GPU — naive vs vec4 run-copy [fp32] (B3)")
    print("=" * 66)
    _time_qkv[128, 196, 384, False, 5, 100](ctx, "naive")
    _time_qkv[128, 196, 384, True, 5, 100](ctx, "vec4 ")
    _time_qkv[64, 256, 512, False, 5, 50](ctx, "naive")
    _time_qkv[64, 256, 512, True, 5, 50](ctx, "vec4 ")
    _time_qkv[256, 196, 768, False, 5, 50](ctx, "naive")
    _time_qkv[256, 196, 768, True, 5, 50](ctx, "vec4 ")
    print("=" * 66)
    print("vec/naive speedup; ~1.0 = naive already coalesced+bandwidth-bound (A3-like).")
