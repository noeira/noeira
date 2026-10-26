"""O5 spike: fused implicit-GEMM Conv2D forward (no materialized `col`).

The Conv2D forward is `out[BS,OC] = im2col(x)[BS,COL] @ Wᵀ[OC,COL] + bias`, today
done as THREE ops: `_im2col_kernel` (materialize col[BS,COL]) → `max_matmul` →
scatter/bias. On NVIDIA im2col is bandwidth-bound (~2 TB/s), so the only lever is
removing the `col` buffer traffic entirely. O5 = an implicit GEMM: a tiled GEMM
whose A operand (col) is GATHERED from `input` on the fly (via the O1 shape-static
index table) inside the tile loop, so col is never written or re-read.

This spike measures the crossover honestly:
  baseline = _im2col_kernel + max_matmul[transpose_b] + bias-add   (3 ops)
  O5       = _implicit_gemm_fwd                                    (1 fused kernel)

Expectation: the hand-rolled TILE×TILE SIMT GEMM will NOT match max_matmul's
tensor-core throughput; O5's only saving is the col write (~BS·COL·4 B) + a launch.
So O5-by-hand likely WINS only when max_matmul is on a slow small-N path, and
LOSES where max_matmul shines → conclusion informs whether O5 needs Modular's
structured/`max` conv kernels rather than a hand-roll. Correctness is verified on
a small shape vs a CPU im2col+matmul reference.

O6 (MEC) — the productive variant. Instead of removing the lowering buffer, MEC
SHRINKS it: lower a compact `L[B·Ow, Hp·K·IC]` (K-fold width dup, not K²) and run
`Oh` overlapping-band GEMMs via `linalg.bmm.batched_matmul` (tensor-core, Modular-
maintained — NOT a hand-roll). The bands are a STRIDED 3D `TileTensor` view over `L`
(batch-stride `S·K·IC`), read in place. So MEC keeps the library GEMM (unlike O5) and
cuts the dominant im2col traffic ~`Kh/s`× (3× on S=1, 1.5× on S=2). See
`docs/CONV2D_KERNEL_OPTIMIZATION.md` §8. Δ vs CPU ref is ~1e-5 (fp32 GEMM reduction
reorder, not bit-exact — promote on convergence parity, like O2).

Run (NVIDIA = perf truth):
    pixi run -e nvidia mojo run -I . benchmarks/bench_conv2d_implicit_gemm_gpu.mojo
Run (Apple = parity + signal):
    pixi run -e apple  mojo run -I . benchmarks/bench_conv2d_implicit_gemm_gpu.mojo
"""

from max.gpu import global_idx, thread_idx, block_idx, block_dim
from max.gpu.sync import barrier
from max.gpu.memory import AddressSpace
from max.gpu.host import DeviceContext, DeviceBuffer
from std.time import perf_counter_ns
from layout import TileTensor, row_major, Idx
from noeira.core.tiles import Tile, tile, shared_tile
from layout.tile_layout import Layout as TileLayout
from linalg.matmul import matmul as max_matmul
from linalg.bmm import batched_matmul

from noeira.nn.primitives.conv2d import _im2col_cpu, _im2col_kernel

comptime DT = DType.float32
comptime IT = DType.int32
comptime TPB = 128
comptime TILE = 16


# ── O5: fused implicit-GEMM forward ───────────────────────────────────────
# C[m,n] = Σ_k A[m,k]·W[n,k] + bias[n], m∈[0,BS) n∈[0,OC) k∈[0,COL).
# A[m,k] = gather(input): m→(b,s), k→ck, off=table[s*COL+ck] (−1=pad). W direct.
def _implicit_gemm_fwd[
    IC: Int, K: Int, S: Int, P: Int, H: Int, W: Int, OH: Int, OW: Int,
    IN_FLAT: Int, COL: Int, SO: Int, BS: Int, OC: Int,
](
    input: Tile[DT, BS // SO * IN_FLAT],
    gather: Tile[IT, SO * COL],
    weight: Tile[DT, OC, COL],
    bias: Tile[DT, OC],
    output: Tile[DT, BS, OC],
):
    var a_sh = shared_tile[DT, TILE, TILE]()
    var b_sh = shared_tile[DT, TILE, TILE]()

    var ty = Int(thread_idx.y)
    var tx = Int(thread_idx.x)
    var row = Int(block_idx.y) * TILE + ty  # m (bs)
    var col = Int(block_idx.x) * TILE + tx  # n (oc)
    var acc: Scalar[DT] = 0

    comptime nkt = (COL + TILE - 1) // TILE
    for kt in range(nkt):
        # A-tile: a_sh[ty,tx] = A[row, kt*TILE+tx] = gather(input)
        var k_a = kt * TILE + tx
        var av: Scalar[DT] = 0
        if row < BS and k_a < COL:
            var b = row // SO
            var s = row % SO
            var off = Int(rebind[Scalar[IT]](gather[s * COL + k_a]))
            if off >= 0:
                av = rebind[Scalar[DT]](input[b * IN_FLAT + off])
        a_sh[ty, tx] = av
        # B-tile: b_sh[ty,tx] = Wᵀ[kt*TILE+ty, col] = W[col, kt*TILE+ty]
        var k_b = kt * TILE + ty
        var bv: Scalar[DT] = 0
        if col < OC and k_b < COL:
            bv = rebind[Scalar[DT]](weight[col, k_b])
        b_sh[ty, tx] = bv
        barrier()
        for e in range(TILE):
            acc += rebind[Scalar[DT]](a_sh[ty, e]) * rebind[Scalar[DT]](
                b_sh[e, tx]
            )
        barrier()

    if row < BS and col < OC:
        output[row, col] = acc + rebind[Scalar[DT]](bias[col])


# ── baseline helpers ──────────────────────────────────────────────────────
def _bias_add[BS: Int, OC: Int](
    output: Tile[DT, BS, OC],
    bias: Tile[DT, OC],
):
    var idx = Int(global_idx.x)
    if idx >= BS * OC:
        return
    var row = idx // OC
    var oc = idx % OC
    output[row, oc] = rebind[Scalar[DT]](output[row, oc]) + rebind[
        Scalar[DT]
    ](bias[oc])


def _build_gather[
    IC: Int, K: Int, S: Int, P: Int, H: Int, W: Int, OH: Int, OW: Int,
    COL: Int, SO: Int,
](ctx: DeviceContext) raises -> DeviceBuffer[IT]:
    var g = ctx.enqueue_create_buffer[IT](SO * COL)
    with g.map_to_host() as hg:
        for s in range(SO):
            var oh = s // OW
            var ow = s % OW
            for ck in range(COL):
                var ic = ck // (K * K)
                var rem = ck % (K * K)
                var kh = rem // K
                var kw = rem % K
                var ih = oh * S + kh - P
                var iw = ow * S + kw - P
                var v: Int = -1
                if ih >= 0 and ih < H and iw >= 0 and iw < W:
                    v = ic * H * W + ih * W + iw
                hg[s * COL + ck] = Scalar[IT](v)
    return g


# ── correctness (small shape) ─────────────────────────────────────────────
def _verify[
    BATCH: Int, IC: Int, K: Int, S: Int, P: Int, H: Int, W: Int,
    OH: Int, OW: Int, IN_FLAT: Int, COL: Int, SO: Int, BS: Int, OC: Int,
](ctx: DeviceContext) raises -> Float64:
    var inp = ctx.enqueue_create_buffer[DT](BATCH * IN_FLAT)
    var wbuf = ctx.enqueue_create_buffer[DT](OC * COL)
    var bbuf = ctx.enqueue_create_buffer[DT](OC)
    var obuf = ctx.enqueue_create_buffer[DT](BS * OC)

    var in_host = List[Scalar[DT]](length=BATCH * IN_FLAT, fill=Scalar[DT](0))
    var w_host = List[Scalar[DT]](length=OC * COL, fill=Scalar[DT](0))
    var b_host = List[Scalar[DT]](length=OC, fill=Scalar[DT](0))
    with inp.map_to_host() as hi:
        for i in range(BATCH * IN_FLAT):
            var v = Scalar[DT](Float64((i % 97) - 48) * 0.05)
            hi[i] = v
            in_host[i] = v
    with wbuf.map_to_host() as hw:
        for i in range(OC * COL):
            var v = Scalar[DT](Float64((i % 53) - 26) * 0.03)
            hw[i] = v
            w_host[i] = v
    with bbuf.map_to_host() as hb:
        for i in range(OC):
            var v = Scalar[DT](Float64(i) * 0.01)
            hb[i] = v
            b_host[i] = v
    _ = obuf.enqueue_fill(Scalar[DT](-999.0))

    var g = _build_gather[IC, K, S, P, H, W, OH, OW, COL, SO](ctx)
    var inl = tile[BATCH * IN_FLAT](inp)
    var gl = tile[SO * COL](g)
    var wl = tile[OC, COL](wbuf)
    var bl = tile[OC](bbuf)
    var ol = tile[BS, OC](obuf)
    comptime gx = (OC + TILE - 1) // TILE
    comptime gy = (BS + TILE - 1) // TILE
    ctx.enqueue_function[
        _implicit_gemm_fwd[IC, K, S, P, H, W, OH, OW, IN_FLAT, COL, SO, BS, OC]
    ](inl, gl, wl, bl, ol, grid_dim=(gx, gy), block_dim=(TILE, TILE))
    ctx.synchronize()

    # CPU ref: im2col per sample → out[bs,oc] = Σ_ck col[s,ck]·W[oc,ck] + bias
    var col_s = List[Scalar[DT]](length=SO * COL, fill=Scalar[DT](0))
    var max_abs = Float64(0)
    with obuf.map_to_host() as ho:
        for b in range(BATCH):
            _im2col_cpu[IC, K, S, P, H, W, OH, OW](in_host, b * IN_FLAT, col_s)
            for s in range(SO):
                for oc in range(OC):
                    var acc = Scalar[DT](0)
                    for ck in range(COL):
                        acc += col_s[s * COL + ck] * w_host[oc * COL + ck]
                    acc += b_host[oc]
                    var got = ho[(b * SO + s) * OC + oc]
                    var d = Float64(got - acc)
                    if d < 0:
                        d = -d
                    if d > max_abs:
                        max_abs = d
    return max_abs


# ── timing ────────────────────────────────────────────────────────────────
def _time_baseline[
    BATCH: Int, IC: Int, K: Int, S: Int, P: Int, H: Int, W: Int,
    OH: Int, OW: Int, IN_FLAT: Int, COL: Int, SO: Int, BS: Int, OC: Int,
    WARMUP: Int, ITERS: Int,
](ctx: DeviceContext) raises -> Float64:
    var inp = ctx.enqueue_create_buffer[DT](BATCH * IN_FLAT)
    var colb = ctx.enqueue_create_buffer[DT](BS * COL)
    var wbuf = ctx.enqueue_create_buffer[DT](OC * COL)
    var bbuf = ctx.enqueue_create_buffer[DT](OC)
    var obuf = ctx.enqueue_create_buffer[DT](BS * OC)
    _ = inp.enqueue_fill(Scalar[DT](0.01))
    _ = wbuf.enqueue_fill(Scalar[DT](0.01))
    _ = bbuf.enqueue_fill(Scalar[DT](0.0))
    var inl = tile[BATCH, IN_FLAT](inp)
    var coll = tile[BS, COL](colb)
    var col_tt = TileTensor(colb, row_major[BS, COL]())
    var w_tt = TileTensor(wbuf, row_major[OC, COL]())
    var outp_tt = TileTensor(obuf, row_major[BS, OC]())
    var bl = tile[OC](bbuf)
    var ol = tile[BS, OC](obuf)
    comptime nb_col = (BS * COL + TPB - 1) // TPB
    comptime nb_bias = (BS * OC + TPB - 1) // TPB

    @__parameter
    def one() raises:
        ctx.enqueue_function[
            _im2col_kernel[
                BATCH, IC, K, S, P, H, W, OH, OW, IN_FLAT, COL, SO, BS
            ]
        ](inl, coll, grid_dim=nb_col, block_dim=TPB)
        max_matmul[transpose_b=True, target="gpu"](outp_tt, col_tt, w_tt, ctx)
        ctx.enqueue_function[_bias_add[BS, OC]](
            ol, bl, grid_dim=nb_bias, block_dim=TPB
        )

    comptime for _ in range(WARMUP):
        one()
    ctx.synchronize()
    var t0 = perf_counter_ns()
    comptime for _ in range(ITERS):
        one()
    ctx.synchronize()
    return Float64(perf_counter_ns() - t0) / Float64(ITERS) / 1000.0


def _time_o5[
    BATCH: Int, IC: Int, K: Int, S: Int, P: Int, H: Int, W: Int,
    OH: Int, OW: Int, IN_FLAT: Int, COL: Int, SO: Int, BS: Int, OC: Int,
    WARMUP: Int, ITERS: Int,
](ctx: DeviceContext) raises -> Float64:
    var inp = ctx.enqueue_create_buffer[DT](BATCH * IN_FLAT)
    var wbuf = ctx.enqueue_create_buffer[DT](OC * COL)
    var bbuf = ctx.enqueue_create_buffer[DT](OC)
    var obuf = ctx.enqueue_create_buffer[DT](BS * OC)
    _ = inp.enqueue_fill(Scalar[DT](0.01))
    _ = wbuf.enqueue_fill(Scalar[DT](0.01))
    _ = bbuf.enqueue_fill(Scalar[DT](0.0))
    var g = _build_gather[IC, K, S, P, H, W, OH, OW, COL, SO](ctx)
    var inl = tile[BATCH * IN_FLAT](inp)
    var gl = tile[SO * COL](g)
    var wl = tile[OC, COL](wbuf)
    var bl = tile[OC](bbuf)
    var ol = tile[BS, OC](obuf)
    comptime gx = (OC + TILE - 1) // TILE
    comptime gy = (BS + TILE - 1) // TILE

    @__parameter
    def one() raises:
        ctx.enqueue_function[
            _implicit_gemm_fwd[
                IC, K, S, P, H, W, OH, OW, IN_FLAT, COL, SO, BS, OC
            ]
        ](inl, gl, wl, bl, ol, grid_dim=(gx, gy), block_dim=(TILE, TILE))

    comptime for _ in range(WARMUP):
        one()
    ctx.synchronize()
    var t0 = perf_counter_ns()
    comptime for _ in range(ITERS):
        one()
    ctx.synchronize()
    return Float64(perf_counter_ns() - t0) / Float64(ITERS) / 1000.0


# ── MEC (O6): compact lowering + batched (strided-band) GEMM ───────────────
# L[B·Ow, Hp·K·IC]  (K-fold dup in width, Hp=H+2P).  The band for output-row oh
# is the contiguous column slice [oh·S·K·IC : +K·K·IC] of L — fed to
# batched_matmul as a STRIDED 3D view (batch=Oh, no extra copy).  k-order within
# a band = (kh,kw,ic), so the weight is repacked from im2col's (ic,kh,kw).
def _mec_lower_kernel[
    BATCH: Int, IC: Int, K: Int, S: Int, P: Int, H: Int, W: Int, OW: Int,
    IN_FLAT: Int, HP: Int,
](
    input: Tile[DT, BATCH * IN_FLAT],
    lout: Tile[DT, BATCH * OW * HP * K * IC],
):
    var idx = Int(global_idx.x)
    comptime LW = HP * K * IC  # L row width
    comptime LN = BATCH * OW * LW
    if idx >= LN:
        return
    var r = idx // LW  # r = b*OW + ow
    var j = idx % LW
    var b = r // OW
    var ow = r % OW
    var hp = j // (K * IC)
    var rem = j % (K * IC)
    var kw = rem // IC
    var ic = rem % IC
    var ih = hp - P
    var iw = ow * S + kw - P
    var v: Scalar[DT] = 0
    if ih >= 0 and ih < H and iw >= 0 and iw < W:
        v = rebind[Scalar[DT]](input[b * IN_FLAT + ic * H * W + ih * W + iw])
    lout[idx] = v


# C[Oh, B·Ow, OC] → out[BS, OC] + bias.  bs = b*SO + oh*OW + ow ; r = b*OW + ow.
def _mec_scatter_bias[
    BATCH: Int, OC: Int, OH: Int, OW: Int, BS: Int,
](
    cmat: Tile[DT, OH * BATCH * OW * OC],
    bias: Tile[DT, OC],
    output: Tile[DT, BS * OC],
):
    var idx = Int(global_idx.x)
    if idx >= BS * OC:
        return
    comptime SO = OH * OW
    comptime BOW = BATCH * OW
    var oc = idx % OC
    var bs = idx // OC
    var b = bs // SO
    var s = bs % SO
    var oh = s // OW
    var ow = s % OW
    var r = b * OW + ow
    var cidx = (oh * BOW + r) * OC + oc
    output[idx] = rebind[Scalar[DT]](cmat[cidx]) + rebind[Scalar[DT]](bias[oc])


# Strided 3D view over L for the batched_matmul A operand: A[oh, r, k] =
# L[r·(HP·K·IC) + oh·(S·K·IC) + k].  shape (Oh, B·Ow, COL), strides
# (S·K·IC, HP·K·IC, 1) — the overlapping bands, read in place (no copy).
def _time_mec[
    BATCH: Int, IC: Int, K: Int, S: Int, P: Int, H: Int, W: Int,
    OH: Int, OW: Int, IN_FLAT: Int, COL: Int, SO: Int, BS: Int, OC: Int,
    WARMUP: Int, ITERS: Int,
](ctx: DeviceContext) raises -> Float64:
    comptime HP = H + 2 * P
    comptime BOW = BATCH * OW
    comptime LN = BOW * HP * K * IC
    var inp = ctx.enqueue_create_buffer[DT](BATCH * IN_FLAT)
    var lbuf = ctx.enqueue_create_buffer[DT](LN)
    var wbuf = ctx.enqueue_create_buffer[DT](OH * OC * COL)  # replicated weight
    var cbuf = ctx.enqueue_create_buffer[DT](OH * BOW * OC)
    var bbuf = ctx.enqueue_create_buffer[DT](OC)
    var obuf = ctx.enqueue_create_buffer[DT](BS * OC)
    _ = inp.enqueue_fill(Scalar[DT](0.01))
    _ = wbuf.enqueue_fill(Scalar[DT](0.01))
    _ = bbuf.enqueue_fill(Scalar[DT](0.0))
    var inl = tile[BATCH * IN_FLAT](inp)
    var ll = tile[LN](lbuf)
    var bl = tile[OC](bbuf)
    var ol = tile[BS * OC](obuf)
    var a_lay = TileLayout(
        (Idx[OH], Idx[BOW], Idx[COL]),
        (Idx[S * K * IC], Idx[HP * K * IC], Idx[1]),
    )
    var a_tt = TileTensor(lbuf, a_lay)
    var w_tt = TileTensor(wbuf, row_major[OH, OC, COL]())
    var c_tt = TileTensor(cbuf, row_major[OH, BOW, OC]())
    var cl = tile[OH * BOW * OC](cbuf)
    comptime nb_l = (LN + TPB - 1) // TPB
    comptime nb_sc = (BS * OC + TPB - 1) // TPB

    @__parameter
    def one() raises:
        ctx.enqueue_function[
            _mec_lower_kernel[BATCH, IC, K, S, P, H, W, OW, IN_FLAT, HP]
        ](inl, ll, grid_dim=nb_l, block_dim=TPB)
        batched_matmul[transpose_b=True, target="gpu"](
            c_tt, a_tt, w_tt, context=ctx
        )
        ctx.enqueue_function[_mec_scatter_bias[BATCH, OC, OH, OW, BS]](
            cl, bl, ol, grid_dim=nb_sc, block_dim=TPB
        )

    comptime for _ in range(WARMUP):
        one()
    ctx.synchronize()
    var t0 = perf_counter_ns()
    comptime for _ in range(ITERS):
        one()
    ctx.synchronize()
    return Float64(perf_counter_ns() - t0) / Float64(ITERS) / 1000.0


def _verify_mec[
    BATCH: Int, IC: Int, K: Int, S: Int, P: Int, H: Int, W: Int,
    OH: Int, OW: Int, IN_FLAT: Int, COL: Int, SO: Int, BS: Int, OC: Int,
](ctx: DeviceContext) raises -> Float64:
    comptime HP = H + 2 * P
    comptime BOW = BATCH * OW
    comptime LN = BOW * HP * K * IC
    var inp = ctx.enqueue_create_buffer[DT](BATCH * IN_FLAT)
    var lbuf = ctx.enqueue_create_buffer[DT](LN)
    var wbuf = ctx.enqueue_create_buffer[DT](OH * OC * COL)
    var cbuf = ctx.enqueue_create_buffer[DT](OH * BOW * OC)
    var bbuf = ctx.enqueue_create_buffer[DT](OC)
    var obuf = ctx.enqueue_create_buffer[DT](BS * OC)

    var in_host = List[Scalar[DT]](length=BATCH * IN_FLAT, fill=Scalar[DT](0))
    var w_host = List[Scalar[DT]](length=OC * COL, fill=Scalar[DT](0))  # im2col order
    var b_host = List[Scalar[DT]](length=OC, fill=Scalar[DT](0))
    with inp.map_to_host() as hi:
        for i in range(BATCH * IN_FLAT):
            var v = Scalar[DT](Float64((i % 97) - 48) * 0.05)
            hi[i] = v
            in_host[i] = v
    # logical weight in im2col (ic,kh,kw) order
    for i in range(OC * COL):
        w_host[i] = Scalar[DT](Float64((i % 53) - 26) * 0.03)
    with bbuf.map_to_host() as hb:
        for i in range(OC):
            var v = Scalar[DT](Float64(i) * 0.01)
            hb[i] = v
            b_host[i] = v
    # repack → wbuf[oh, oc, k], k=(kh*K+kw)*IC+ic, replicated across Oh.
    with wbuf.map_to_host() as hw:
        for oc in range(OC):
            for k in range(COL):
                var kh = k // (K * IC)
                var rem = k % (K * IC)
                var kw = rem // IC
                var ic = rem % IC
                var ck = (ic * K + kh) * K + kw  # im2col index
                var wv = w_host[oc * COL + ck]
                for oh in range(OH):
                    hw[(oh * OC + oc) * COL + k] = wv
    _ = obuf.enqueue_fill(Scalar[DT](-999.0))

    var inl = tile[BATCH * IN_FLAT](inp)
    var ll = tile[LN](lbuf)
    var bl = tile[OC](bbuf)
    var ol = tile[BS * OC](obuf)
    var a_lay = TileLayout(
        (Idx[OH], Idx[BOW], Idx[COL]),
        (Idx[S * K * IC], Idx[HP * K * IC], Idx[1]),
    )
    var a_tt = TileTensor(lbuf, a_lay)
    var w_tt = TileTensor(wbuf, row_major[OH, OC, COL]())
    var c_tt = TileTensor(cbuf, row_major[OH, BOW, OC]())
    var cl = tile[OH * BOW * OC](cbuf)
    comptime nb_l = (LN + TPB - 1) // TPB
    comptime nb_sc = (BS * OC + TPB - 1) // TPB
    ctx.enqueue_function[
        _mec_lower_kernel[BATCH, IC, K, S, P, H, W, OW, IN_FLAT, HP]
    ](inl, ll, grid_dim=nb_l, block_dim=TPB)
    batched_matmul[transpose_b=True, target="gpu"](c_tt, a_tt, w_tt, context=ctx)
    ctx.enqueue_function[_mec_scatter_bias[BATCH, OC, OH, OW, BS]](
        cl, bl, ol, grid_dim=nb_sc, block_dim=TPB
    )
    ctx.synchronize()

    var col_s = List[Scalar[DT]](length=SO * COL, fill=Scalar[DT](0))
    var max_abs = Float64(0)
    with obuf.map_to_host() as ho:
        for b in range(BATCH):
            _im2col_cpu[IC, K, S, P, H, W, OH, OW](in_host, b * IN_FLAT, col_s)
            for s in range(SO):
                for oc in range(OC):
                    var acc = Scalar[DT](0)
                    for ck in range(COL):
                        acc += col_s[s * COL + ck] * w_host[oc * COL + ck]
                    acc += b_host[oc]
                    var got = ho[(b * SO + s) * OC + oc]
                    var d = Float64(got - acc)
                    if d < 0:
                        d = -d
                    if d > max_abs:
                        max_abs = d
    return max_abs


def run_shape[
    BATCH: Int, IC: Int, K: Int, S: Int, P: Int, H: Int, W: Int, OC: Int,
    WARMUP: Int, ITERS: Int,
](ctx: DeviceContext, label: StaticString) raises:
    comptime OH = (H + 2 * P - K) // S + 1
    comptime OW = (W + 2 * P - K) // S + 1
    comptime IN_FLAT = IC * H * W
    comptime COL = IC * K * K
    comptime SO = OH * OW
    comptime BS = BATCH * SO
    var base = _time_baseline[
        BATCH, IC, K, S, P, H, W, OH, OW, IN_FLAT, COL, SO, BS, OC, WARMUP, ITERS
    ](ctx)
    var o5 = _time_o5[
        BATCH, IC, K, S, P, H, W, OH, OW, IN_FLAT, COL, SO, BS, OC, WARMUP, ITERS
    ](ctx)
    var mec = _time_mec[
        BATCH, IC, K, S, P, H, W, OH, OW, IN_FLAT, COL, SO, BS, OC, WARMUP, ITERS
    ](ctx)
    print(
        "  ", label, " BS=", BS, " OC=", OC, " COL=", COL,
        " | baseline=", base, "us  O5=", o5, "us  MEC(O6)=", mec,
        "us  | O5×=", base / o5, " MEC×=", base / mec,
    )


def main() raises:
    var ctx = DeviceContext()
    print("O5 spike: fused implicit-GEMM Conv2D forward [fp32]")
    print("=" * 70)
    # correctness on a small shape (B=2, IC=8, 3x3 s1 p1, 6x7, OC=16)
    var err = _verify[2, 8, 3, 1, 1, 6, 7, 6, 7, 8 * 6 * 7, 8 * 9, 42, 84, 16](ctx)
    print("  O5  correctness: max|Δ| vs CPU im2col+matmul =", err)
    # MEC correctness on the same small shape (validates the strided-band view).
    var errm = _verify_mec[
        2, 8, 3, 1, 1, 6, 7, 6, 7, 8 * 6 * 7, 8 * 9, 42, 84, 16
    ](ctx)
    print("  MEC correctness: max|Δ| vs CPU im2col+matmul =", errm)
    # MEC on a strided (S=2) shape too — band stride = S·K·IC ≠ K·IC.
    var errm2 = _verify_mec[
        2, 8, 3, 2, 1, 12, 12, 6, 6, 8 * 12 * 12, 8 * 9, 36, 72, 16
    ](ctx)
    print("  MEC correctness (S=2): max|Δ| =", errm2)
    print("-" * 70)
    # C4 res tower (hot): IC=64, 3x3 s1 p1, 6x7, OC=64
    run_shape[256, 64, 3, 1, 1, 6, 7, 64, 5, 100](ctx, "C4 res    ")
    # EZv2 deep block spatial 6x6: IC=64, 3x3 s1 p1, OC=64
    run_shape[256, 64, 3, 1, 1, 6, 6, 64, 5, 100](ctx, "EZ deep6  ")
    # Atari mid: IC=32, 4x4 s2 p0, 20x20, OC=64
    run_shape[64, 32, 4, 2, 0, 20, 20, 64, 5, 50](ctx, "Atari mid ")
    print("-" * 70)
    print("  EZv2-Atari REP hot shapes (the §8.3 census 1ms+ im2col kernels):")
    # resblocks1 (the BIGGEST im2col): Conv2D[32,32,3,1,1,48,48], S=1
    run_shape[64, 32, 3, 1, 1, 48, 48, 32, 5, 20](ctx, "EZ rep48  ")
    # down main2 / resblocks2: Conv2D[64,64,3,1,1,24,24], S=1
    run_shape[64, 64, 3, 1, 1, 24, 24, 64, 5, 30](ctx, "EZ rep24  ")
    # conv1 stem: Conv2D[12,32,3,2,1,96,96], S=2 (the 1.5× MEC case)
    run_shape[32, 12, 3, 2, 1, 96, 96, 32, 5, 20](ctx, "EZ stem96 ")
    print("=" * 70)
    print("MEC(O6) = compact lowering + batched_matmul (tensor-core, strided band).")
    print("  MEC×>1 → MEC beats im2col+max_matmul (3× less lowering on S=1).")
    print("O5(fused) = hand SIMT GEMM; O5×<1 expected (loses to max_matmul).")
    print("NVIDIA is the perf truth; Apple here is parity (Δ) + a directional signal.")
