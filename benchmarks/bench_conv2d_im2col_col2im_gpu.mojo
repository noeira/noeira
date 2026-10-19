"""Conv2D im2col / col2im kernel A/B — perf + correctness, in isolation.

Self-contained A/B harness (one process) for the two gather/scatter kernels that
flank the Conv2D GEMM in `noeira/nn/primitives/conv2d.mojo`. Modeled on
`bench_storage_transpose_gpu.mojo`. Tracker: `docs/CONV2D_KERNEL_OPTIMIZATION.md`.

Each GPU variant is verified against the CPU reference (`_im2col_cpu` /
`_col2im_cpu`, imported from conv2d.mojo — the source of truth) on an index-fill
pattern, then timed. A variant ships to conv2d.mojo only at `mismatches=0` and a
measured NVIDIA win.

Variants:
  im2col  v0 = baseline (current conv2d.mojo kernel, copied here) — NCHW
          v1 = O1 shape-static gather-index table (2 divmods + 1 cached lookup)
          v2 = O4 vectorized v4 col write
          v3 = NHWC ceiling test (channels-last input, COL→(kh,kw,ic), ic
               innermost → coalesced reads in bursts of IC vs v0's bursts of K).
               v0→v3 isolates the NCHW→NHWC layout question for the forward
               gather: if v3 ≈ v0 on NVIDIA, L2 already hides the strides and a
               channels_last refactor is not worth it (same verdict O1 reached).
  col2im  v0 = baseline (gather, [BS,COL] d_col — uncoalesced)
          v1 = O2 transpose ([COL,BS] d_col → coalesced) + O3 comptime S==1
          v2 = O2+O3+O4 vectorized

Adding a variant: write `_im2col_vN` / `_col2im_vN`, add a `comptime if VARIANT`
arm in the time/verify dispatch, register it in `main()`. (v3 uses a dedicated
`_im2col_nhwc_{verify,time}` pair since its NHWC layout needs its own reference.)

Run (NVIDIA = perf truth):
    pixi run -e nvidia mojo run -I . benchmarks/bench_conv2d_im2col_col2im_gpu.mojo
Run (Apple = parity only):
    pixi run -e apple  mojo run -I . benchmarks/bench_conv2d_im2col_col2im_gpu.mojo
"""

from max.gpu import global_idx, thread_idx, block_idx, block_dim
from max.gpu.host import DeviceContext, DeviceBuffer
from std.time import perf_counter_ns
from noeira.core.tiles import Tile, tile

from noeira.nn.primitives.conv2d import _im2col_cpu, _col2im_cpu

comptime DT = DType.float32
comptime IT = DType.int32
comptime TPB = 128


# ══════════════════════════════════════════════════════════════════════════
# im2col kernels
# ══════════════════════════════════════════════════════════════════════════
# v0 — baseline (verbatim from conv2d.mojo `_im2col_kernel`, fp32 only).
def _im2col_v0[
    BATCH: Int, IC: Int, K: Int, S: Int, P: Int, H: Int, W: Int,
    OH: Int, OW: Int, IN_FLAT: Int, COL: Int, SO: Int, BS: Int,
](
    input: Tile[DT, BATCH, IN_FLAT],
    col: Tile[DT, BS, COL],
):
    var idx = Int(global_idx.x)
    if idx >= BS * COL:
        return
    var row = idx // COL
    var ck = idx % COL
    var b = row // SO
    var s = row % SO
    var oh = s // OW
    var ow = s % OW
    var ic = ck // (K * K)
    var rem = ck % (K * K)
    var kh = rem // K
    var kw = rem % K
    var ih = oh * S + kh - P
    var iw = ow * S + kw - P
    if ih < 0 or ih >= H or iw < 0 or iw >= W:
        col[row, ck] = Scalar[DT](0)
    else:
        col[row, ck] = rebind[Scalar[DT]](input[b, ic * H * W + ih * W + iw])


# v1 — O1 shape-static gather table. `gather[SO*COL]` holds the within-sample
# input offset (`ic*H*W + ih*W + iw`) or -1 (padding), built once on the host.
# Per thread: `b = idx // (SO*COL)`, `g = idx % (SO*COL)`, one table load, one
# branchless select. col[BS,COL] and input[BATCH,IN_FLAT] are addressed flat
# (element (row,ck) is at row*COL+ck == idx; offset is b*IN_FLAT + table value).
def _im2col_v1[
    BATCH: Int, IC: Int, K: Int, S: Int, P: Int, H: Int, W: Int,
    OH: Int, OW: Int, IN_FLAT: Int, COL: Int, SO: Int, BS: Int,
](
    input: Tile[DT, BATCH * IN_FLAT],
    gather: Tile[IT, SO * COL],
    col: Tile[DT, BS * COL],
):
    var idx = Int(global_idx.x)
    if idx >= BS * COL:
        return
    var b = idx // (SO * COL)
    var g = idx % (SO * COL)
    var off = Int(rebind[Scalar[IT]](gather[g]))
    if off < 0:
        col[idx] = Scalar[DT](0)
    else:
        col[idx] = rebind[Scalar[DT]](input[b * IN_FLAT + off])


# v2 — O4 vectorize the baseline: each thread does 4 consecutive `ck` (same row)
# → the col WRITE is a v4 store (col[row,ck..ck+3] contiguous). The input gather
# stays scalar (4 reads — the input offsets aren't contiguous in general). Tests
# whether vectorizing im2col's (already-coalesced) write side buys anything.
def _im2col_v2[
    BATCH: Int, IC: Int, K: Int, S: Int, P: Int, H: Int, W: Int,
    OH: Int, OW: Int, IN_FLAT: Int, COL: Int, SO: Int, BS: Int,
](
    input: Tile[DT, BATCH, IN_FLAT],
    col: Tile[DT, BS, COL],
):
    var base = Int(global_idx.x) * 4
    if base >= BS * COL:
        return
    var row = base // COL
    var ck0 = base % COL
    var b = row // SO
    var s = row % SO
    var oh = s // OW
    var ow = s % OW
    var cptr = col.ptr
    var iptr = input.ptr
    var in_base = b * IN_FLAT
    if ck0 + 4 <= COL:
        var v = SIMD[DT, 4](0)
        for L in range(4):
            var ck = ck0 + L
            var ic = ck // (K * K)
            var rem = ck % (K * K)
            var kh = rem // K
            var kw = rem % K
            var ih = oh * S + kh - P
            var iw = ow * S + kw - P
            if ih >= 0 and ih < H and iw >= 0 and iw < W:
                v[L] = iptr[in_base + ic * H * W + ih * W + iw]
        cptr.store[alignment=4](base, v)
    else:
        for L in range(4):
            var idx = base + L
            if idx >= BS * COL:
                break
            var ck = idx % COL
            var ic = ck // (K * K)
            var rem = ck % (K * K)
            var kh = rem // K
            var kw = rem % K
            var ih = oh * S + kh - P
            var iw = ow * S + kw - P
            if ih >= 0 and ih < H and iw >= 0 and iw < W:
                cptr[idx] = iptr[in_base + ic * H * W + ih * W + iw]
            else:
                cptr[idx] = Scalar[DT](0)


# v3 — NHWC CEILING TEST (NCHW→NHWC layout question). Input is channels-LAST
# `[BATCH, H, W, IC]` (flat IN_FLAT=H*W*IC) and COL is reordered to (kh, kw, ic)
# so `ic` is the innermost gather dim — CONTIGUOUS in NHWC. Consecutive threads →
# consecutive `ic` → coalesced reads in bursts of IC (=32/64) vs v0's NCHW bursts
# of K (=3). The write side is identical to v0 (contiguous col). So v0→v3 isolates
# *exactly* the input-layout coalescing effect — the upper bound on what a
# framework-wide channels_last refactor could buy the forward gather. (Reordering
# COL permutes the weight columns to match: GEMM-identical, free; the bench only
# times the gather.) If v3 ≈ v0 on NVIDIA, L2 was already hiding the strides and
# NHWC is not worth the refactor — same verdict O1 reached.
def _im2col_v3[
    BATCH: Int, IC: Int, K: Int, S: Int, P: Int, H: Int, W: Int,
    OH: Int, OW: Int, IN_FLAT: Int, COL: Int, SO: Int, BS: Int,
](
    input: Tile[DT, BATCH, IN_FLAT],
    col: Tile[DT, BS, COL],
):
    var idx = Int(global_idx.x)
    if idx >= BS * COL:
        return
    var row = idx // COL
    var ck = idx % COL
    var b = row // SO
    var s = row % SO
    var oh = s // OW
    var ow = s % OW
    # COL order (kh, kw, ic) — ic innermost (contiguous in NHWC input).
    var khkw = ck // IC
    var ic = ck % IC
    var kh = khkw // K
    var kw = khkw % K
    var ih = oh * S + kh - P
    var iw = ow * S + kw - P
    if ih < 0 or ih >= H or iw < 0 or iw >= W:
        col[row, ck] = Scalar[DT](0)
    else:
        # NHWC: input[b, ih, iw, ic] = (ih*W + iw)*IC + ic  within sample.
        col[row, ck] = rebind[Scalar[DT]](input[b, (ih * W + iw) * IC + ic])


# ══════════════════════════════════════════════════════════════════════════
# col2im kernels
# ══════════════════════════════════════════════════════════════════════════
# v0 — baseline (verbatim from conv2d.mojo `_dx_col2im_kernel`, fp32). Reads
# d_col[BS,COL] with col_idx fixed across adjacent threads → stride-COL,
# uncoalesced.
def _col2im_v0[
    BATCH: Int, IC: Int, K: Int, S: Int, P: Int, H: Int, W: Int,
    OH: Int, OW: Int, IN_FLAT: Int, COL: Int, SO: Int, BS: Int,
](
    d_col: Tile[DT, BS, COL],
    grad_input: Tile[DT, BATCH, IN_FLAT],
):
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= BATCH * IN_FLAT:
        return
    var b = idx // IN_FLAT
    var in_pos = idx % IN_FLAT
    var hw = H * W
    var ic = in_pos // hw
    var rem = in_pos % hw
    var ih = rem // W
    var iw = rem % W
    var acc: Scalar[DT] = 0
    for kh in range(K):
        var oh_num = ih + P - kh
        if oh_num < 0 or oh_num % S != 0:
            continue
        var oh = oh_num // S
        if oh >= OH:
            continue
        for kw in range(K):
            var ow_num = iw + P - kw
            if ow_num < 0 or ow_num % S != 0:
                continue
            var ow = ow_num // S
            if ow >= OW:
                continue
            var row = b * SO + oh * OW + ow
            var col_idx = (ic * K + kh) * K + kw
            acc += rebind[Scalar[DT]](d_col[row, col_idx])
    grad_input[b, in_pos] = acc


# v1 — O2 transpose ([COL,BS] d_col, so adjacent threads (row+1) read adjacent
# addresses → coalesced) + O3 comptime S==1 fast path (drops `% S` / `// S`).
def _col2im_v1[
    BATCH: Int, IC: Int, K: Int, S: Int, P: Int, H: Int, W: Int,
    OH: Int, OW: Int, IN_FLAT: Int, COL: Int, SO: Int, BS: Int,
](
    d_colT: Tile[DT, COL, BS],
    grad_input: Tile[DT, BATCH, IN_FLAT],
):
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= BATCH * IN_FLAT:
        return
    var b = idx // IN_FLAT
    var in_pos = idx % IN_FLAT
    var hw = H * W
    var ic = in_pos // hw
    var rem = in_pos % hw
    var ih = rem // W
    var iw = rem % W
    var acc: Scalar[DT] = 0
    comptime if S == 1:
        for kh in range(K):
            var oh = ih + P - kh
            if oh < 0 or oh >= OH:
                continue
            for kw in range(K):
                var ow = iw + P - kw
                if ow < 0 or ow >= OW:
                    continue
                var row = b * SO + oh * OW + ow
                var col_idx = (ic * K + kh) * K + kw
                acc += rebind[Scalar[DT]](d_colT[col_idx, row])
    else:
        for kh in range(K):
            var oh_num = ih + P - kh
            if oh_num < 0 or oh_num % S != 0:
                continue
            var oh = oh_num // S
            if oh >= OH:
                continue
            for kw in range(K):
                var ow_num = iw + P - kw
                if ow_num < 0 or ow_num % S != 0:
                    continue
                var ow = ow_num // S
                if ow >= OW:
                    continue
                var row = b * SO + oh * OW + ow
                var col_idx = (ic * K + kh) * K + kw
                acc += rebind[Scalar[DT]](d_colT[col_idx, row])
    grad_input[b, in_pos] = acc


# v2 — O2 transpose + O3 S==1 + O4 vectorize: each thread does 4 consecutive
# in_pos (4 consecutive iw → 4 contiguous `row` in d_colT[COL,BS]) so each (kh,kw)
# read is a v4 load and the write is a v4 store. Fast path when the 4 share
# (b,ic,ih) and the 4-wide ow window is fully in-bounds; scalar fallback at the W
# edge / tail. Same signature as v1 (uses `.ptr` for flat vectorized access), so
# the dispatch only swaps the kernel + grid. (S!=1 → no vec, scalar per lane.)
def _col2im_v2[
    BATCH: Int, IC: Int, K: Int, S: Int, P: Int, H: Int, W: Int,
    OH: Int, OW: Int, IN_FLAT: Int, COL: Int, SO: Int, BS: Int,
](
    d_colT: Tile[DT, COL, BS],
    grad_input: Tile[DT, BATCH, IN_FLAT],
):
    var hw = H * W
    var base = Int(block_dim.x * block_idx.x + thread_idx.x) * 4
    if base >= BATCH * IN_FLAT:
        return
    var b = base // IN_FLAT
    var in_pos = base % IN_FLAT
    var ic = in_pos // hw
    var rem = in_pos % hw
    var ih = rem // W
    var iw = rem % W
    var dptr = d_colT.ptr
    var gptr = grad_input.ptr

    comptime if S == 1:
        if iw + 4 <= W and in_pos + 4 <= IN_FLAT:
            # fast path: 4 lanes share (b, ic, ih); rows are contiguous → v4
            var acc = SIMD[DT, 4](0)
            for kh in range(K):
                var oh = ih + P - kh
                if oh < 0 or oh >= OH:
                    continue
                var base_row = b * SO + oh * OW
                for kw in range(K):
                    var col_idx = (ic * K + kh) * K + kw
                    var ow0 = iw + P - kw
                    if ow0 >= 0 and ow0 + 4 <= OW:
                        acc += dptr.load[width=4, alignment=4](
                            col_idx * BS + base_row + ow0
                        )
                    else:
                        for L in range(4):
                            var o = ow0 + L
                            if o >= 0 and o < OW:
                                acc[L] = acc[L] + dptr[
                                    col_idx * BS + base_row + o
                                ]
            gptr.store[alignment=4](base, acc)
        else:
            # boundary group straddles a W-row / tail → scalar per lane
            for L in range(4):
                var idx = base + L
                if idx >= BATCH * IN_FLAT:
                    break
                var ip = idx % IN_FLAT
                var iww = ip % W
                var ihh = (ip % hw) // W
                var icc = ip // hw
                var bb = idx // IN_FLAT
                var a: Scalar[DT] = 0
                for kh in range(K):
                    var oh = ihh + P - kh
                    if oh < 0 or oh >= OH:
                        continue
                    for kw in range(K):
                        var ow = iww + P - kw
                        if ow < 0 or ow >= OW:
                            continue
                        var ci = (icc * K + kh) * K + kw
                        a += dptr[ci * BS + bb * SO + oh * OW + ow]
                gptr[idx] = a
    else:
        for L in range(4):
            var idx = base + L
            if idx >= BATCH * IN_FLAT:
                break
            var ip = idx % IN_FLAT
            var iww = ip % W
            var ihh = (ip % hw) // W
            var icc = ip // hw
            var bb = idx // IN_FLAT
            var a: Scalar[DT] = 0
            for kh in range(K):
                var oh_num = ihh + P - kh
                if oh_num < 0 or oh_num % S != 0:
                    continue
                var oh = oh_num // S
                if oh >= OH:
                    continue
                for kw in range(K):
                    var ow_num = iww + P - kw
                    if ow_num < 0 or ow_num % S != 0:
                        continue
                    var ow = ow_num // S
                    if ow >= OW:
                        continue
                    var ci = (icc * K + kh) * K + kw
                    a += dptr[ci * BS + bb * SO + oh * OW + ow]
            gptr[idx] = a


# ══════════════════════════════════════════════════════════════════════════
# host helpers
# ══════════════════════════════════════════════════════════════════════════
def _build_gather[
    IC: Int, K: Int, S: Int, P: Int, H: Int, W: Int, OH: Int, OW: Int,
    COL: Int, SO: Int
](ctx: DeviceContext) raises -> DeviceBuffer[IT]:
    """Shape-static O1 table: gather[s*COL+ck] = within-sample input offset, or
    -1 for padding. Mirrors `_im2col_v0`'s decode but with no batch term."""
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


# ══════════════════════════════════════════════════════════════════════════
# im2col: verify + time
# ══════════════════════════════════════════════════════════════════════════
def _im2col_verify[
    BATCH: Int, IC: Int, K: Int, S: Int, P: Int, H: Int, W: Int,
    OH: Int, OW: Int, IN_FLAT: Int, COL: Int, SO: Int, BS: Int, VARIANT: Int,
](ctx: DeviceContext) raises -> Int:
    var inp = ctx.enqueue_create_buffer[DT](BATCH * IN_FLAT)
    var col = ctx.enqueue_create_buffer[DT](BS * COL)
    # index-fill input: distinct per (b, pos) so any wrong gather shows up
    var in_host = List[Scalar[DT]](length=BATCH * IN_FLAT, fill=Scalar[DT](0))
    with inp.map_to_host() as hi:
        for i in range(BATCH * IN_FLAT):
            var v = Scalar[DT](Float64((i % 997) + 1) * 0.5)
            hi[i] = v
            in_host[i] = v
    _ = col.enqueue_fill(Scalar[DT](-123.0))
    comptime nb = (BS * COL + TPB - 1) // TPB

    comptime if VARIANT == 0:
        var inl = tile[BATCH, IN_FLAT](inp)
        var cl = tile[BS, COL](col)
        ctx.enqueue_function[
            _im2col_v0[BATCH, IC, K, S, P, H, W, OH, OW, IN_FLAT, COL, SO, BS]
        ](inl, cl, grid_dim=nb, block_dim=TPB)
    elif VARIANT == 1:
        var g = _build_gather[IC, K, S, P, H, W, OH, OW, COL, SO](ctx)
        var inl = tile[BATCH * IN_FLAT](inp)
        var gl = tile[SO * COL](g)
        var cl = tile[BS * COL](col)
        ctx.enqueue_function[
            _im2col_v1[BATCH, IC, K, S, P, H, W, OH, OW, IN_FLAT, COL, SO, BS]
        ](inl, gl, cl, grid_dim=nb, block_dim=TPB)
    else:
        comptime nb2 = ((BS * COL + 3) // 4 + TPB - 1) // TPB
        var inl = tile[BATCH, IN_FLAT](inp)
        var cl = tile[BS, COL](col)
        ctx.enqueue_function[
            _im2col_v2[BATCH, IC, K, S, P, H, W, OH, OW, IN_FLAT, COL, SO, BS]
        ](inl, cl, grid_dim=nb2, block_dim=TPB)
    ctx.synchronize()

    # CPU reference (per sample) into a full [BS, COL] expectation
    var col_sample = List[Scalar[DT]](length=SO * COL, fill=Scalar[DT](0))
    var bad = 0
    with col.map_to_host() as hc:
        for b in range(BATCH):
            _im2col_cpu[IC, K, S, P, H, W, OH, OW](in_host, b * IN_FLAT, col_sample)
            for j in range(SO * COL):
                var got = hc[b * SO * COL + j]
                if got != col_sample[j]:
                    bad += 1
    return bad


def _im2col_time[
    BATCH: Int, IC: Int, K: Int, S: Int, P: Int, H: Int, W: Int,
    OH: Int, OW: Int, IN_FLAT: Int, COL: Int, SO: Int, BS: Int, VARIANT: Int,
    WARMUP: Int, ITERS: Int,
](ctx: DeviceContext, label: StaticString) raises:
    var inp = ctx.enqueue_create_buffer[DT](BATCH * IN_FLAT)
    var col = ctx.enqueue_create_buffer[DT](BS * COL)
    _ = inp.enqueue_fill(Scalar[DT](0.01))
    _ = col.enqueue_fill(Scalar[DT](0.0))
    comptime nb = (BS * COL + TPB - 1) // TPB
    var us = Float64(0)

    comptime if VARIANT == 0:
        var inl = tile[BATCH, IN_FLAT](inp)
        var cl = tile[BS, COL](col)
        comptime for _ in range(WARMUP):
            ctx.enqueue_function[
                _im2col_v0[BATCH, IC, K, S, P, H, W, OH, OW, IN_FLAT, COL, SO, BS]
            ](inl, cl, grid_dim=nb, block_dim=TPB)
        ctx.synchronize()
        var t0 = perf_counter_ns()
        comptime for _ in range(ITERS):
            ctx.enqueue_function[
                _im2col_v0[BATCH, IC, K, S, P, H, W, OH, OW, IN_FLAT, COL, SO, BS]
            ](inl, cl, grid_dim=nb, block_dim=TPB)
        ctx.synchronize()
        us = Float64(perf_counter_ns() - t0) / Float64(ITERS) / 1000.0
    elif VARIANT == 1:
        var g = _build_gather[IC, K, S, P, H, W, OH, OW, COL, SO](ctx)
        var inl = tile[BATCH * IN_FLAT](inp)
        var gl = tile[SO * COL](g)
        var cl = tile[BS * COL](col)
        comptime for _ in range(WARMUP):
            ctx.enqueue_function[
                _im2col_v1[BATCH, IC, K, S, P, H, W, OH, OW, IN_FLAT, COL, SO, BS]
            ](inl, gl, cl, grid_dim=nb, block_dim=TPB)
        ctx.synchronize()
        var t0 = perf_counter_ns()
        comptime for _ in range(ITERS):
            ctx.enqueue_function[
                _im2col_v1[BATCH, IC, K, S, P, H, W, OH, OW, IN_FLAT, COL, SO, BS]
            ](inl, gl, cl, grid_dim=nb, block_dim=TPB)
        ctx.synchronize()
        us = Float64(perf_counter_ns() - t0) / Float64(ITERS) / 1000.0
    else:
        comptime nb2 = ((BS * COL + 3) // 4 + TPB - 1) // TPB
        var inl = tile[BATCH, IN_FLAT](inp)
        var cl = tile[BS, COL](col)
        comptime for _ in range(WARMUP):
            ctx.enqueue_function[
                _im2col_v2[BATCH, IC, K, S, P, H, W, OH, OW, IN_FLAT, COL, SO, BS]
            ](inl, cl, grid_dim=nb2, block_dim=TPB)
        ctx.synchronize()
        var t0 = perf_counter_ns()
        comptime for _ in range(ITERS):
            ctx.enqueue_function[
                _im2col_v2[BATCH, IC, K, S, P, H, W, OH, OW, IN_FLAT, COL, SO, BS]
            ](inl, cl, grid_dim=nb2, block_dim=TPB)
        ctx.synchronize()
        us = Float64(perf_counter_ns() - t0) / Float64(ITERS) / 1000.0

    var gb = 2.0 * Float64(BS) * Float64(COL) * 4.0 / 1e9
    print("  ", label, " | ", us, "us/iter ", gb / (us / 1e6) / 1e3, "TB/s")


# ── NHWC im2col (v3): own channels-last reference + timer ──────────────────
# Separate from the shared NCHW path because v3 uses an NHWC input layout AND a
# (kh,kw,ic) COL order, so it can't be checked against `_im2col_cpu` (NCHW,
# ic,kh,kw). The index-fill input is self-consistent: the GPU gather and the CPU
# reference both decode the SAME NHWC buffer the same way.
def _im2col_nhwc_verify[
    BATCH: Int, IC: Int, K: Int, S: Int, P: Int, H: Int, W: Int,
    OH: Int, OW: Int, IN_FLAT: Int, COL: Int, SO: Int, BS: Int,
](ctx: DeviceContext) raises -> Int:
    var inp = ctx.enqueue_create_buffer[DT](BATCH * IN_FLAT)
    var col = ctx.enqueue_create_buffer[DT](BS * COL)
    var in_host = List[Scalar[DT]](length=BATCH * IN_FLAT, fill=Scalar[DT](0))
    with inp.map_to_host() as hi:
        for i in range(BATCH * IN_FLAT):
            var v = Scalar[DT](Float64((i % 997) + 1) * 0.5)
            hi[i] = v
            in_host[i] = v
    _ = col.enqueue_fill(Scalar[DT](-123.0))
    comptime nb = (BS * COL + TPB - 1) // TPB
    var inl = tile[BATCH, IN_FLAT](inp)
    var cl = tile[BS, COL](col)
    ctx.enqueue_function[
        _im2col_v3[BATCH, IC, K, S, P, H, W, OH, OW, IN_FLAT, COL, SO, BS]
    ](inl, cl, grid_dim=nb, block_dim=TPB)
    ctx.synchronize()

    var bad = 0
    with col.map_to_host() as hc:
        for b in range(BATCH):
            for s in range(SO):
                var oh = s // OW
                var ow = s % OW
                for ck in range(COL):
                    var khkw = ck // IC
                    var ic = ck % IC
                    var kh = khkw // K
                    var kw = khkw % K
                    var ih = oh * S + kh - P
                    var iw = ow * S + kw - P
                    var exp = Scalar[DT](0)
                    if ih >= 0 and ih < H and iw >= 0 and iw < W:
                        exp = in_host[b * IN_FLAT + (ih * W + iw) * IC + ic]
                    if hc[(b * SO + s) * COL + ck] != exp:
                        bad += 1
    return bad


def _im2col_nhwc_time[
    BATCH: Int, IC: Int, K: Int, S: Int, P: Int, H: Int, W: Int,
    OH: Int, OW: Int, IN_FLAT: Int, COL: Int, SO: Int, BS: Int,
    WARMUP: Int, ITERS: Int,
](ctx: DeviceContext, label: StaticString) raises:
    var inp = ctx.enqueue_create_buffer[DT](BATCH * IN_FLAT)
    var col = ctx.enqueue_create_buffer[DT](BS * COL)
    _ = inp.enqueue_fill(Scalar[DT](0.01))
    _ = col.enqueue_fill(Scalar[DT](0.0))
    comptime nb = (BS * COL + TPB - 1) // TPB
    var inl = tile[BATCH, IN_FLAT](inp)
    var cl = tile[BS, COL](col)
    comptime for _ in range(WARMUP):
        ctx.enqueue_function[
            _im2col_v3[BATCH, IC, K, S, P, H, W, OH, OW, IN_FLAT, COL, SO, BS]
        ](inl, cl, grid_dim=nb, block_dim=TPB)
    ctx.synchronize()
    var t0 = perf_counter_ns()
    comptime for _ in range(ITERS):
        ctx.enqueue_function[
            _im2col_v3[BATCH, IC, K, S, P, H, W, OH, OW, IN_FLAT, COL, SO, BS]
        ](inl, cl, grid_dim=nb, block_dim=TPB)
    ctx.synchronize()
    var us = Float64(perf_counter_ns() - t0) / Float64(ITERS) / 1000.0
    var gb = 2.0 * Float64(BS) * Float64(COL) * 4.0 / 1e9
    print("  ", label, " | ", us, "us/iter ", gb / (us / 1e6) / 1e3, "TB/s")


# ══════════════════════════════════════════════════════════════════════════
# col2im: verify + time
# ══════════════════════════════════════════════════════════════════════════
def _col2im_verify[
    BATCH: Int, IC: Int, K: Int, S: Int, P: Int, H: Int, W: Int,
    OH: Int, OW: Int, IN_FLAT: Int, COL: Int, SO: Int, BS: Int, VARIANT: Int,
](ctx: DeviceContext) raises -> Int:
    var dcol = ctx.enqueue_create_buffer[DT](BS * COL)
    var gin = ctx.enqueue_create_buffer[DT](BATCH * IN_FLAT)
    # logical d_col[b,s,ck] index-fill; lay out per variant ([BS,COL] vs [COL,BS])
    var dcol_host = List[Scalar[DT]](length=BS * COL, fill=Scalar[DT](0))  # [BS,COL]
    with dcol.map_to_host() as hd:
        for b in range(BATCH):
            for s in range(SO):
                for ck in range(COL):
                    var row = b * SO + s
                    var v = Scalar[DT](Float64(((row * COL + ck) % 991) + 1) * 0.25)
                    dcol_host[row * COL + ck] = v
                    comptime if VARIANT == 0:
                        hd[row * COL + ck] = v          # [BS, COL]
                    else:
                        hd[ck * BS + row] = v           # [COL, BS] transpose
    _ = gin.enqueue_fill(Scalar[DT](0.0))
    comptime nb = (BATCH * IN_FLAT + TPB - 1) // TPB

    comptime if VARIANT == 0:
        var dl = tile[BS, COL](dcol)
        var gl = tile[BATCH, IN_FLAT](gin)
        ctx.enqueue_function[
            _col2im_v0[BATCH, IC, K, S, P, H, W, OH, OW, IN_FLAT, COL, SO, BS]
        ](dl, gl, grid_dim=nb, block_dim=TPB)
    elif VARIANT == 1:
        var dl = tile[COL, BS](dcol)
        var gl = tile[BATCH, IN_FLAT](gin)
        ctx.enqueue_function[
            _col2im_v1[BATCH, IC, K, S, P, H, W, OH, OW, IN_FLAT, COL, SO, BS]
        ](dl, gl, grid_dim=nb, block_dim=TPB)
    else:
        comptime nb2 = ((BATCH * IN_FLAT + 3) // 4 + TPB - 1) // TPB
        var dl = tile[COL, BS](dcol)
        var gl = tile[BATCH, IN_FLAT](gin)
        ctx.enqueue_function[
            _col2im_v2[BATCH, IC, K, S, P, H, W, OH, OW, IN_FLAT, COL, SO, BS]
        ](dl, gl, grid_dim=nb2, block_dim=TPB)
    ctx.synchronize()

    # CPU reference: scatter-add per sample into d_in
    var d_in_ref = List[Scalar[DT]](length=BATCH * IN_FLAT, fill=Scalar[DT](0))
    var dcol_sample = List[Scalar[DT]](length=SO * COL, fill=Scalar[DT](0))
    for b in range(BATCH):
        for j in range(SO * COL):
            dcol_sample[j] = dcol_host[b * SO * COL + j]
        _col2im_cpu[IC, K, S, P, H, W, OH, OW](dcol_sample, d_in_ref, b * IN_FLAT)

    var bad = 0
    with gin.map_to_host() as hg:
        for i in range(BATCH * IN_FLAT):
            # tolerance: fp32 sum order differs between GPU and CPU ref
            var diff = Float64(hg[i] - d_in_ref[i])
            if diff < 0:
                diff = -diff
            if diff > 1e-3:
                bad += 1
    return bad


def _col2im_time[
    BATCH: Int, IC: Int, K: Int, S: Int, P: Int, H: Int, W: Int,
    OH: Int, OW: Int, IN_FLAT: Int, COL: Int, SO: Int, BS: Int, VARIANT: Int,
    WARMUP: Int, ITERS: Int,
](ctx: DeviceContext, label: StaticString) raises:
    var dcol = ctx.enqueue_create_buffer[DT](BS * COL)
    var gin = ctx.enqueue_create_buffer[DT](BATCH * IN_FLAT)
    _ = dcol.enqueue_fill(Scalar[DT](0.01))
    _ = gin.enqueue_fill(Scalar[DT](0.0))
    comptime nb = (BATCH * IN_FLAT + TPB - 1) // TPB
    var us = Float64(0)

    comptime if VARIANT == 0:
        var dl = tile[BS, COL](dcol)
        var gl = tile[BATCH, IN_FLAT](gin)
        comptime for _ in range(WARMUP):
            ctx.enqueue_function[
                _col2im_v0[BATCH, IC, K, S, P, H, W, OH, OW, IN_FLAT, COL, SO, BS]
            ](dl, gl, grid_dim=nb, block_dim=TPB)
        ctx.synchronize()
        var t0 = perf_counter_ns()
        comptime for _ in range(ITERS):
            ctx.enqueue_function[
                _col2im_v0[BATCH, IC, K, S, P, H, W, OH, OW, IN_FLAT, COL, SO, BS]
            ](dl, gl, grid_dim=nb, block_dim=TPB)
        ctx.synchronize()
        us = Float64(perf_counter_ns() - t0) / Float64(ITERS) / 1000.0
    elif VARIANT == 1:
        var dl = tile[COL, BS](dcol)
        var gl = tile[BATCH, IN_FLAT](gin)
        comptime for _ in range(WARMUP):
            ctx.enqueue_function[
                _col2im_v1[BATCH, IC, K, S, P, H, W, OH, OW, IN_FLAT, COL, SO, BS]
            ](dl, gl, grid_dim=nb, block_dim=TPB)
        ctx.synchronize()
        var t0 = perf_counter_ns()
        comptime for _ in range(ITERS):
            ctx.enqueue_function[
                _col2im_v1[BATCH, IC, K, S, P, H, W, OH, OW, IN_FLAT, COL, SO, BS]
            ](dl, gl, grid_dim=nb, block_dim=TPB)
        ctx.synchronize()
        us = Float64(perf_counter_ns() - t0) / Float64(ITERS) / 1000.0
    else:
        comptime nb2 = ((BATCH * IN_FLAT + 3) // 4 + TPB - 1) // TPB
        var dl = tile[COL, BS](dcol)
        var gl = tile[BATCH, IN_FLAT](gin)
        comptime for _ in range(WARMUP):
            ctx.enqueue_function[
                _col2im_v2[BATCH, IC, K, S, P, H, W, OH, OW, IN_FLAT, COL, SO, BS]
            ](dl, gl, grid_dim=nb2, block_dim=TPB)
        ctx.synchronize()
        var t0 = perf_counter_ns()
        comptime for _ in range(ITERS):
            ctx.enqueue_function[
                _col2im_v2[BATCH, IC, K, S, P, H, W, OH, OW, IN_FLAT, COL, SO, BS]
            ](dl, gl, grid_dim=nb2, block_dim=TPB)
        ctx.synchronize()
        us = Float64(perf_counter_ns() - t0) / Float64(ITERS) / 1000.0

    var gb = 2.0 * Float64(BS) * Float64(COL) * 4.0 / 1e9
    print("  ", label, " | ", us, "us/iter ", gb / (us / 1e6) / 1e3, "TB/s")


# ══════════════════════════════════════════════════════════════════════════
# driver — one shape, all variants
# ══════════════════════════════════════════════════════════════════════════
def run_shape[
    BATCH: Int, IC: Int, K: Int, S: Int, P: Int, H: Int, W: Int,
    WARMUP: Int, ITERS: Int,
](ctx: DeviceContext, label: StaticString) raises:
    comptime OH = (H + 2 * P - K) // S + 1
    comptime OW = (W + 2 * P - K) // S + 1
    comptime IN_FLAT = IC * H * W
    comptime COL = IC * K * K
    comptime SO = OH * OW
    comptime BS = BATCH * SO
    print(label, " IC=", IC, " K=", K, " S=", S, " P=", P, " ", H, "x", W,
          " | BS=", BS, " COL=", COL)

    var b0 = _im2col_verify[
        BATCH, IC, K, S, P, H, W, OH, OW, IN_FLAT, COL, SO, BS, 0](ctx)
    var b1 = _im2col_verify[
        BATCH, IC, K, S, P, H, W, OH, OW, IN_FLAT, COL, SO, BS, 1](ctx)
    var b2 = _im2col_verify[
        BATCH, IC, K, S, P, H, W, OH, OW, IN_FLAT, COL, SO, BS, 2](ctx)
    var b3 = _im2col_nhwc_verify[
        BATCH, IC, K, S, P, H, W, OH, OW, IN_FLAT, COL, SO, BS](ctx)
    print("  im2col verify: v0=", b0, " v1=", b1, " v2=", b2, " v3(NHWC)=", b3,
          " mismatches")
    _im2col_time[
        BATCH, IC, K, S, P, H, W, OH, OW, IN_FLAT, COL, SO, BS, 0, WARMUP, ITERS
    ](ctx, "im2col v0 baseline NCHW")
    _im2col_time[
        BATCH, IC, K, S, P, H, W, OH, OW, IN_FLAT, COL, SO, BS, 1, WARMUP, ITERS
    ](ctx, "im2col v1 gather-table")
    _im2col_time[
        BATCH, IC, K, S, P, H, W, OH, OW, IN_FLAT, COL, SO, BS, 2, WARMUP, ITERS
    ](ctx, "im2col v2 vec-store(O4)")
    _im2col_nhwc_time[
        BATCH, IC, K, S, P, H, W, OH, OW, IN_FLAT, COL, SO, BS, WARMUP, ITERS
    ](ctx, "im2col v3 NHWC ceiling ")

    var c0 = _col2im_verify[
        BATCH, IC, K, S, P, H, W, OH, OW, IN_FLAT, COL, SO, BS, 0](ctx)
    var c1 = _col2im_verify[
        BATCH, IC, K, S, P, H, W, OH, OW, IN_FLAT, COL, SO, BS, 1](ctx)
    var c2 = _col2im_verify[
        BATCH, IC, K, S, P, H, W, OH, OW, IN_FLAT, COL, SO, BS, 2](ctx)
    print("  col2im verify: v0=", c0, " v1=", c1, " v2=", c2, " mismatches")
    _col2im_time[
        BATCH, IC, K, S, P, H, W, OH, OW, IN_FLAT, COL, SO, BS, 0, WARMUP, ITERS
    ](ctx, "col2im v0 baseline    ")
    _col2im_time[
        BATCH, IC, K, S, P, H, W, OH, OW, IN_FLAT, COL, SO, BS, 1, WARMUP, ITERS
    ](ctx, "col2im v1 transpose+S1")
    _col2im_time[
        BATCH, IC, K, S, P, H, W, OH, OW, IN_FLAT, COL, SO, BS, 2, WARMUP, ITERS
    ](ctx, "col2im v2 +vec O4     ")
    print("-" * 70)


def main() raises:
    var ctx = DeviceContext()
    print("Conv2D im2col/col2im A/B — perf + correctness [fp32]")
    print("=" * 70)
    # C4 residual tower (the hot shape): IC=64, 3x3 s1 p1, 6x7
    run_shape[256, 64, 3, 1, 1, 6, 7, 5, 100](ctx, "C4 res")
    # Atari/DQN mid: IC=32, 4x4 s2 p0, 20x20 (strided, S!=1 path)
    run_shape[64, 32, 4, 2, 0, 20, 20, 5, 50](ctx, "Atari mid")
    # Atari/DQN stem: IC=4, 8x8 s4 p0, 84x84 (large K)
    run_shape[64, 4, 8, 4, 0, 84, 84, 5, 50](ctx, "Atari stem")
    # EZv2-Atari rep net hot shapes (the 1ms forward im2col from the profile) —
    # this is where the NHWC v3 ceiling matters most: IC=32/64, large spatial.
    run_shape[64, 32, 3, 1, 1, 48, 48, 5, 30](ctx, "EZv2 rep48")
    run_shape[64, 64, 3, 1, 1, 24, 24, 5, 30](ctx, "EZv2 rep24")
    print("=" * 70)
    print("All `mismatches=0` required before promoting a variant to conv2d.mojo.")
    print("Perf truth = NVIDIA; Apple is parity-only.")
