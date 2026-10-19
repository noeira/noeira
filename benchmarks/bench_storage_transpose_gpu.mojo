"""Group B (B1) microbench: storage Transpose2D GPU —
naive scattered global access (one thread per element, strided read) vs
shared-memory tiled transpose (32x32 tile, padded to avoid bank conflicts,
coalesced read AND write). Self-contained A/B in one process.

Per-sample A×B (row-major) → B×A (row-major):
    dst[b, j*A + i] = src[b, i*B + j]

The naive kernel writes dst coalesced but reads src with stride B (uncoalesced);
the tiled kernel stages a 32×32 block in shared memory so both halves coalesce.

Shapes: ViT PatchEmbed turns Conv2D channel-major (embed_dim, n_patches) into
attention patch-major (n_patches, embed_dim). embed_dim∈{192,384,768},
n_patches≈196 (14×14). Plus a large square case as a tiling stress test.

Run (NVIDIA): pixi run -e nvidia mojo run -I . benchmarks/bench_storage_transpose_gpu.mojo
Run (Apple):  pixi run -e apple  mojo run -I . benchmarks/bench_storage_transpose_gpu.mojo
"""

from max.gpu import global_idx, thread_idx, block_idx, block_dim
from max.gpu.sync import barrier
from max.gpu.memory import AddressSpace
from max.gpu.host import DeviceContext
from std.time import perf_counter_ns
from noeira.core.tiles import Tile, tile, shared_tile

comptime DT = DType.float32
comptime TPB = 128
comptime TILE = 32
comptime BR = 8           # BLOCK_ROWS: 32x8 block, 4 elems/thread (variant C)


# ── naive: one thread per element, strided read of src ────────────────────
def _t2d_naive[
    BATCH: Int, A: Int, B: Int
](
    src: Tile[DT, BATCH, A * B],
    dst: Tile[DT, BATCH, A * B],
):
    var gid = Int(global_idx.x)
    comptime total = BATCH * A * B
    if gid >= total:
        return
    var b = gid // (A * B)
    var o = gid % (A * B)          # dst position = j*A + i
    var j = o // A
    var i = o % A
    dst[b, o] = rebind[Scalar[DT]](src[b, i * B + j])


# ── tiled: 32x32 shared-mem tile, coalesced read + write ──────────────────
def _t2d_tiled[
    BATCH: Int, A: Int, B: Int
](
    src: Tile[DT, BATCH, A * B],
    dst: Tile[DT, BATCH, A * B],
):
    # Tile in src-space: rows = i (A dim), cols = j (B dim).
    var tile = shared_tile[DT, TILE, TILE + 1]()

    var b = Int(block_idx.z)
    var cy = Int(block_idx.y) * TILE        # tile origin in i (A)
    var cx = Int(block_idx.x) * TILE        # tile origin in j (B)
    var tx = Int(thread_idx.x)
    var ty = Int(thread_idx.y)

    # load: i=cy+ty, j=cx+tx → consecutive tx = consecutive j (contiguous read)
    var i = cy + ty
    var j = cx + tx
    if i < A and j < B:
        tile[ty, tx] = rebind[Scalar[DT]](src[b, i * B + j])
    barrier()

    # write: out col = i (contiguous, stride 1 in dst row j), out row = j
    var i2 = cy + tx
    var j2 = cx + ty
    if i2 < A and j2 < B:
        dst[b, j2 * A + i2] = rebind[Scalar[DT]](tile[tx, ty])


# ── tiled + BLOCK_ROWS: 32x8 block, 4 elems/thread (canonical NVIDIA) ──────
def _t2d_tiled_br[
    BATCH: Int, A: Int, B: Int
](
    src: Tile[DT, BATCH, A * B],
    dst: Tile[DT, BATCH, A * B],
):
    var tile = shared_tile[DT, TILE, TILE + 1]()

    var b = Int(block_idx.z)
    var cy = Int(block_idx.y) * TILE        # tile origin in i (A)
    var cx = Int(block_idx.x) * TILE        # tile origin in j (B)
    var tx = Int(thread_idx.x)              # [0, TILE)
    var ty = Int(thread_idx.y)              # [0, BR)

    var j = cx + tx
    comptime for r in range(0, TILE, BR):
        var i = cy + ty + r
        if i < A and j < B:
            tile[ty + r, tx] = rebind[Scalar[DT]](src[b, i * B + j])
    barrier()

    var i2 = cy + tx                        # out col (coalesced)
    comptime for r in range(0, TILE, BR):
        var j2 = cx + ty + r                # out row
        if i2 < A and j2 < B:
            dst[b, j2 * A + i2] = rebind[Scalar[DT]](tile[tx, ty + r])


def _time[
    BATCH: Int, A: Int, B: Int, MODE: Int, WARMUP: Int, ITERS: Int
](ctx: DeviceContext, label: StaticString) raises:
    comptime AB = A * B
    var s = ctx.enqueue_create_buffer[DT](BATCH * AB)
    var d = ctx.enqueue_create_buffer[DT](BATCH * AB)
    _ = s.enqueue_fill(Scalar[DT](0.01))
    _ = d.enqueue_fill(Scalar[DT](0.0))
    var sl = tile[BATCH, AB](s)
    var dl = tile[BATCH, AB](d)
    var us = Float64(0)

    comptime gx = (B + TILE - 1) // TILE
    comptime gy = (A + TILE - 1) // TILE
    comptime nb = (BATCH * AB + TPB - 1) // TPB

    comptime if MODE == 0:
        comptime for _ in range(WARMUP):
            ctx.enqueue_function[_t2d_naive[BATCH, A, B]](
                sl, dl, grid_dim=nb, block_dim=TPB
            )
        ctx.synchronize()
        var t0 = perf_counter_ns()
        comptime for _ in range(ITERS):
            ctx.enqueue_function[_t2d_naive[BATCH, A, B]](
                sl, dl, grid_dim=nb, block_dim=TPB
            )
        ctx.synchronize()
        us = Float64(perf_counter_ns() - t0) / Float64(ITERS) / 1000.0
    elif MODE == 1:
        comptime for _ in range(WARMUP):
            ctx.enqueue_function[_t2d_tiled[BATCH, A, B]](
                sl, dl, grid_dim=(gx, gy, BATCH), block_dim=(TILE, TILE)
            )
        ctx.synchronize()
        var t0 = perf_counter_ns()
        comptime for _ in range(ITERS):
            ctx.enqueue_function[_t2d_tiled[BATCH, A, B]](
                sl, dl, grid_dim=(gx, gy, BATCH), block_dim=(TILE, TILE)
            )
        ctx.synchronize()
        us = Float64(perf_counter_ns() - t0) / Float64(ITERS) / 1000.0
    else:
        comptime for _ in range(WARMUP):
            ctx.enqueue_function[_t2d_tiled_br[BATCH, A, B]](
                sl, dl, grid_dim=(gx, gy, BATCH), block_dim=(TILE, BR)
            )
        ctx.synchronize()
        var t0 = perf_counter_ns()
        comptime for _ in range(ITERS):
            ctx.enqueue_function[_t2d_tiled_br[BATCH, A, B]](
                sl, dl, grid_dim=(gx, gy, BATCH), block_dim=(TILE, BR)
            )
        ctx.synchronize()
        us = Float64(perf_counter_ns() - t0) / Float64(ITERS) / 1000.0

    # bandwidth: read AB + write AB elements per sample, 4 bytes each
    var gb = 2.0 * Float64(BATCH) * Float64(AB) * 4.0 / 1e9
    print(
        "  ", label, " B=", BATCH, " A=", A, " (", B, "patches) | ",
        us, "us/iter ", gb / (us / 1e6) / 1e3, "TB/s",
    )


def _ab[
    BATCH: Int, A: Int, B: Int, WARMUP: Int, ITERS: Int
](ctx: DeviceContext) raises:
    _time[BATCH, A, B, 0, WARMUP, ITERS](ctx, "naive   ")
    _time[BATCH, A, B, 1, WARMUP, ITERS](ctx, "tiled32 ")
    _time[BATCH, A, B, 2, WARMUP, ITERS](ctx, "tiled_br")


def _verify[BATCH: Int, A: Int, B: Int, MODE: Int](ctx: DeviceContext) raises:
    """Index-fill correctness: src[b,k]=b*1000+k; dst[b,j*A+i] must equal
    src[b,i*B+j]. Confirms both tiled variants match the analytic map."""
    comptime AB = A * B
    var s = ctx.enqueue_create_buffer[DT](BATCH * AB)
    var d = ctx.enqueue_create_buffer[DT](BATCH * AB)
    with s.map_to_host() as hs:
        for b in range(BATCH):
            for k in range(AB):
                hs[b * AB + k] = Scalar[DT](b * 1000 + k)
    _ = d.enqueue_fill(Scalar[DT](-1.0))
    var sl = tile[BATCH, AB](s)
    var dl = tile[BATCH, AB](d)
    comptime gx = (B + TILE - 1) // TILE
    comptime gy = (A + TILE - 1) // TILE
    comptime if MODE == 1:
        ctx.enqueue_function[_t2d_tiled[BATCH, A, B]](
            sl, dl, grid_dim=(gx, gy, BATCH), block_dim=(TILE, TILE)
        )
    else:
        ctx.enqueue_function[_t2d_tiled_br[BATCH, A, B]](
            sl, dl, grid_dim=(gx, gy, BATCH), block_dim=(TILE, BR)
        )
    ctx.synchronize()
    var bad = 0
    with d.map_to_host() as hd:
        for b in range(BATCH):
            for i in range(A):
                for j in range(B):
                    var got = hd[b * AB + j * A + i]
                    var want = Scalar[DT](b * 1000 + i * B + j)
                    if got != want:
                        bad += 1
    print("  verify mode=", MODE, " B=", BATCH, " A=", A, " B=", B,
          " mismatches=", bad)


def main() raises:
    var ctx = DeviceContext()
    print("Transpose2D GPU — naive scattered vs shared-mem tiled [fp32] (B1)")
    print("=" * 66)
    # correctness (both tiled variants vs analytic) on tile-edge shapes
    _verify[3, 37, 50, 1](ctx)
    _verify[3, 37, 50, 2](ctx)
    _verify[2, 196, 65, 1](ctx)
    _verify[2, 196, 65, 2](ctx)
    print("-" * 66)
    # ViT PatchEmbed: (embed_dim, n_patches)
    _ab[256, 192, 196, 5, 100](ctx)
    _ab[256, 384, 196, 5, 100](ctx)
    _ab[256, 768, 196, 5, 100](ctx)
    # large square stress
    _ab[64, 512, 512, 5, 50](ctx)
    _ab[16, 1024, 1024, 5, 50](ctx)
    print("=" * 66)
    print("tiled32 = current B1 (32x32, 1elem/thread); tiled_br = 32x8, 4elem/thread.")
    print("tiled_br > tiled32 = occupancy/ILP win → promote to B1'.")
