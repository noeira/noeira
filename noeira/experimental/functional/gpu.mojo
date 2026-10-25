"""The functional tape's GPU kernels. Every size is a RUNTIME argument, so one
compiled kernel serves every shape a tape records — the property that lets
one binary run any architecture on the GPU as it does on the CPU.

Kernels take raw device pointers (a `DeviceBuffer` converts at the launch)
and `Int64` sizes, as `fill_dev` does. Each thread owns one OUTPUT element
and loops over what it sums, so nothing needs atomics and a gradient is the
same bits on every run.

Every backward kernel has an `ACC` form: the FIRST contribution to an
activation's gradient is written (`ACC=False`), later ones added. That
replaces a zero-fill per node per step and lets a GEMM write its gradient
in place. Parameter gradients always accumulate.

GEMMs: on NVIDIA, `cublas_gemm` (runtime dims, transposes and beta built
in), at the comptime nn's precision for the same shape: `cublas_tf32(m, n,
k)` — TF32 from 2^24 multiply-adds, fp32 below — evaluated at RUN time over
both instantiations. (Always-fp32 made a B=4096 MLP step 1.8x the comptime
one on an RTX 5090, mostly SIMT sgemm vs TF32 tensor-core GEMMs.) Elsewhere (Metal), MAX's `matmul` over runtime-extent views —
the form `mm.mojo` already uses on Apple — with two exceptions:

* `transpose_b` only at N >= 2: MAX's GPU matmul silently miscomputes
  with `transpose_b=True` at N = 1 (Conv2D OC=1). It has no `transpose_a`,
  so dB's `aᵀ` is materialised by `_transpose_kernel`.
* a dimension of 1 (a `[B, 1]` critic head, a `[B, 1]` input) runs on
  `_naive_mm_kernel`, which also covers the cases MAX never sees here.

`ctx` is a `Dev` (`backend.mojo`): MAX's context, or the metal_rt launcher.
With `ctx.own` (always on a metal_rt tape) the Metal GEMMs are the tape's
own `_tiled_mm_kernel` / `_naive_mm_kernel`, never MAX's matmul.
"""

from std.sys import default_accelerator
from max.gpu import global_idx, thread_idx, block_idx
from max.gpu.sync import barrier
from max.gpu.memory import AddressSpace
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major
from noeira.core.tiles import shared_tile
from linalg.matmul import matmul as max_matmul

from noeira.nn.constants import DT
from noeira.nn.core.element_op import ElementOp
from noeira.nn.core.cublas_gemm import cublas_gemm, cublas_tf32
from noeira.nn.core.cublaslt_gemm import cublaslt_gemm, EPI_BIAS
from .opcodes import *
from .backend import Dev
from std.memory import stack_allocation


comptime P = Pointer[Scalar[DT], MutAnyOrigin]
comptime TPB = 256
comptime REDUCE_TPB = 256


@always_inline
def _grid(n: Int) -> Int:
    return (n + TPB - 1) // TPB


# ── elementwise binary ops, broadcast ────────────────────────────────────
def _bin_fwd_kernel[OP: Int](
    o: P, a: P, b: P,
    r_: Int64, c_: Int64, ra_: Int64, ca_: Int64, rb_: Int64, cb_: Int64,
):
    var c = Int(c_)
    var t = Int(global_idx.x)
    if t >= Int(r_) * c:
        return
    var ra = Int(ra_)
    var ca = Int(ca_)
    var rb = Int(rb_)
    var cb = Int(cb_)
    var i = t // c
    var j = t % c
    var ka = (i if ra > 1 else 0) * ca + (j if ca > 1 else 0)
    var kb = (i if rb > 1 else 0) * cb + (j if cb > 1 else 0)
    o[unsafe_offset=t] = bin_op[OP](a[unsafe_offset=ka], b[unsafe_offset=kb])


@always_inline
def bin_term[OP: Int, WHICH: Int](
    g: Scalar[DT], x: Scalar[DT], y: Scalar[DT]
) -> Scalar[DT]:
    """Input `WHICH`'s share of the output gradient `g` at one element, `x`
    and `y` being a's and b's values there."""
    comptime if OP == OP_ADD:
        return g
    elif OP == OP_SUB:
        comptime if WHICH == 0:
            return g
        else:
            return -g
    elif OP == OP_MUL:
        comptime if WHICH == 0:
            return g * y
        else:
            return g * x
    else:
        comptime if WHICH == 0:
            return g / y
        else:
            return -(g * x / (y * y))


@always_inline
def _bin_bwd_sum[OP: Int, WHICH: Int](
    go: P, a: P, b: P, t: Int, i_lo: Int, i_hi: Int,
    r: Int, c: Int, ra: Int, ca: Int, rb: Int, cb: Int,
) -> Scalar[DT]:
    """Element `t` of input `WHICH`'s gradient, summed over output rows
    [i_lo, i_hi) of the ones it was broadcast to (and every column it was)."""
    var rx = ra if WHICH == 0 else rb
    var cx = ca if WHICH == 0 else cb
    var ix = t // cx
    var jx = t % cx
    var i0 = ix if rx == r else i_lo
    var i1 = ix + 1 if rx == r else i_hi
    var j0 = jx if cx == c else 0
    var j1 = jx + 1 if cx == c else c
    var acc: Scalar[DT] = 0
    for i in range(i0, i1):
        for j in range(j0, j1):
            var ka = (i if ra > 1 else 0) * ca + (j if ca > 1 else 0)
            var kb = (i if rb > 1 else 0) * cb + (j if cb > 1 else 0)
            acc += bin_term[OP, WHICH](
                go[unsafe_offset=i * c + j], a[unsafe_offset=ka],
                b[unsafe_offset=kb],
            )
    return acc


def _bin_bwd_kernel[OP: Int, WHICH: Int, ACC: Bool](
    gx: P, go: P, a: P, b: P,
    r_: Int64, c_: Int64, ra_: Int64, ca_: Int64, rb_: Int64, cb_: Int64,
):
    """One thread per element of input `WHICH` (0 = a, 1 = b): its gradient
    summed over every output element it was broadcast to; written, or added
    when `ACC`."""
    var ra = Int(ra_)
    var ca = Int(ca_)
    var rb = Int(rb_)
    var cb = Int(cb_)
    var nx = ra * ca if WHICH == 0 else rb * cb
    var t = Int(global_idx.x)
    if t >= nx:
        return
    var r = Int(r_)
    var s = _bin_bwd_sum[OP, WHICH](
        go, a, b, t, 0, r, r, Int(c_), ra, ca, rb, cb
    )
    comptime if ACC:
        gx[unsafe_offset=t] += s
    else:
        gx[unsafe_offset=t] = s


comptime ROW_CHUNK = 64
"""Rows per partial sum when an input was broadcast along the rows (a bias
`[1, C]`): `r / ROW_CHUNK` x `C` threads instead of `C` threads each walking
every row — the bias gradient at batch 4096 was 512 threads doing 4096
serial adds."""


def _bin_bwd_partial_kernel[OP: Int, WHICH: Int](
    part: P, go: P, a: P, b: P,
    r_: Int64, c_: Int64, ra_: Int64, ca_: Int64, rb_: Int64, cb_: Int64,
):
    """`part[chunk, t]` = element t's gradient over the chunk's rows."""
    var ra = Int(ra_)
    var ca = Int(ca_)
    var rb = Int(rb_)
    var cb = Int(cb_)
    var r = Int(r_)
    var nx = ra * ca if WHICH == 0 else rb * cb
    var nch = (r + ROW_CHUNK - 1) // ROW_CHUNK
    var g = Int(global_idx.x)
    if g >= nch * nx:
        return
    var chunk = g // nx
    var t = g % nx
    var lo = chunk * ROW_CHUNK
    var hi = min(lo + ROW_CHUNK, r)
    part[unsafe_offset=g] = _bin_bwd_sum[OP, WHICH](
        go, a, b, t, lo, hi, r, Int(c_), ra, ca, rb, cb
    )


def _chunk_final_kernel[ACC: Bool](gx: P, part: P, nch_: Int64, nx_: Int64):
    """`gx[t] (+)= sum over chunks of part[chunk, t]`, chunks in order."""
    var nx = Int(nx_)
    var t = Int(global_idx.x)
    if t >= nx:
        return
    var s: Scalar[DT] = 0
    for ch in range(Int(nch_)):
        s += part[unsafe_offset=ch * nx + t]
    comptime if ACC:
        gx[unsafe_offset=t] += s
    else:
        gx[unsafe_offset=t] = s


def gpu_binary[OP: Int](
    ctx: Dev, o: DeviceBuffer[DT], a: DeviceBuffer[DT],
    b: DeviceBuffer[DT], r: Int, c: Int, ra: Int, ca: Int, rb: Int, cb: Int,
) raises:
    ctx.enqueue_function[_bin_fwd_kernel[OP]](
        o, a, b, Int64(r), Int64(c), Int64(ra), Int64(ca), Int64(rb),
        Int64(cb), grid_dim=_grid(r * c), block_dim=TPB,
    )


def gpu_binary_bwd[OP: Int, WHICH: Int, ACC: Bool](
    ctx: Dev, gx: DeviceBuffer[DT], go: DeviceBuffer[DT],
    a: DeviceBuffer[DT], b: DeviceBuffer[DT],
    r: Int, c: Int, ra: Int, ca: Int, rb: Int, cb: Int,
    part: DeviceBuffer[DT],
) raises:
    """`part` >= `bin_bwd_part_size(…)` elements (unused unless the input
    was broadcast along more than `ROW_CHUNK` rows)."""
    var rx = ra if WHICH == 0 else rb
    var nx = ra * ca if WHICH == 0 else rb * cb
    if rx == 1 and r > ROW_CHUNK:
        var nch = (r + ROW_CHUNK - 1) // ROW_CHUNK
        ctx.enqueue_function[_bin_bwd_partial_kernel[OP, WHICH]](
            part, go, a, b, Int64(r), Int64(c), Int64(ra), Int64(ca),
            Int64(rb), Int64(cb), grid_dim=_grid(nch * nx), block_dim=TPB,
        )
        ctx.enqueue_function[_chunk_final_kernel[ACC]](
            gx, part, Int64(nch), Int64(nx), grid_dim=_grid(nx), block_dim=TPB
        )
        return
    ctx.enqueue_function[_bin_bwd_kernel[OP, WHICH, ACC]](
        gx, go, a, b, Int64(r), Int64(c), Int64(ra), Int64(ca), Int64(rb),
        Int64(cb), grid_dim=_grid(nx), block_dim=TPB,
    )


def bin_bwd_part_size(r: Int, nx: Int) -> Int:
    return ((r + ROW_CHUNK - 1) // ROW_CHUNK) * nx


# ── tensor ⊕ constant ────────────────────────────────────────────────────
def _scalar_fwd_kernel[OP: Int](o: P, a: P, n: Int64, c: Scalar[DT]):
    var k = Int(global_idx.x)
    if k >= Int(n):
        return
    var x = a[unsafe_offset=k]
    comptime if OP == OP_SCALE:
        o[unsafe_offset=k] = x * c
    elif OP == OP_SHIFT:
        o[unsafe_offset=k] = x + c
    elif OP == OP_DIVC:
        o[unsafe_offset=k] = x / c
    else:
        o[unsafe_offset=k] = c - x


def _scalar_bwd_kernel[OP: Int, ACC: Bool](
    ga: P, go: P, n: Int64, c: Scalar[DT]
):
    var k = Int(global_idx.x)
    if k >= Int(n):
        return
    var g = go[unsafe_offset=k]
    var d: Scalar[DT]
    comptime if OP == OP_SCALE:
        d = g * c
    elif OP == OP_SHIFT:
        d = g
    elif OP == OP_DIVC:
        d = g / c
    else:
        d = -g
    comptime if ACC:
        ga[unsafe_offset=k] += d
    else:
        ga[unsafe_offset=k] = d


def gpu_scalar[OP: Int](
    ctx: Dev, o: DeviceBuffer[DT], a: DeviceBuffer[DT], n: Int,
    c: Scalar[DT],
) raises:
    ctx.enqueue_function[_scalar_fwd_kernel[OP]](
        o, a, Int64(n), c, grid_dim=_grid(n), block_dim=TPB
    )


def gpu_scalar_bwd[OP: Int, ACC: Bool](
    ctx: Dev, ga: DeviceBuffer[DT], go: DeviceBuffer[DT], n: Int,
    c: Scalar[DT],
) raises:
    ctx.enqueue_function[_scalar_bwd_kernel[OP, ACC]](
        ga, go, Int64(n), c, grid_dim=_grid(n), block_dim=TPB
    )


# ── activations: the comptime nn's `ElementOp`s ──────────────────────────
def _ew_fwd_kernel[OP: ElementOp](y: P, x: P, n: Int64):
    var k = Int(global_idx.x)
    if k < Int(n):
        y[unsafe_offset=k] = OP.forward_scalar(x[unsafe_offset=k])


def _ew_bwd_kernel[OP: ElementOp, ACC: Bool](
    ga: P, go: P, x: P, y: P, n: Int64
):
    var k = Int(global_idx.x)
    if k >= Int(n):
        return
    var cache = y[unsafe_offset=k] if OP.owns_cache else x[unsafe_offset=k]
    var d = OP.backward_scalar(cache, go[unsafe_offset=k])
    comptime if ACC:
        ga[unsafe_offset=k] += d
    else:
        ga[unsafe_offset=k] = d


def gpu_ew[OP: ElementOp](
    ctx: Dev, y: DeviceBuffer[DT], x: DeviceBuffer[DT], n: Int
) raises:
    ctx.enqueue_function[_ew_fwd_kernel[OP]](
        y, x, Int64(n), grid_dim=_grid(n), block_dim=TPB
    )


def gpu_ew_bwd[OP: ElementOp, ACC: Bool](
    ctx: Dev, ga: DeviceBuffer[DT], go: DeviceBuffer[DT],
    x: DeviceBuffer[DT], y: DeviceBuffer[DT], n: Int,
) raises:
    ctx.enqueue_function[_ew_bwd_kernel[OP, ACC]](
        ga, go, x, y, Int64(n), grid_dim=_grid(n), block_dim=TPB
    )


# ── fused linear: bias + activation epilogue, activation gate ────────────
def _bias_act_kernel[OP: ElementOp, HAS_ACT: Bool](
    o: P, bias: P, z: P, m_: Int64, n_: Int64
):
    """`o = act(o + bias)` in place over a GEMM's [m, n] output, saving the
    pre-activation `z` when the activation's backward needs its INPUT — the
    arithmetic of `LinearAct`'s `_bias_act_cache_kernel` (fp32 path)."""
    var n = Int(n_)
    var t = Int(global_idx.x)
    if t >= Int(m_) * n:
        return
    var zz = o[unsafe_offset=t] + bias[unsafe_offset=t % n]
    comptime if HAS_ACT:
        comptime if not OP.owns_cache:
            z[unsafe_offset=t] = zz
        o[unsafe_offset=t] = OP.forward_scalar(zz)
    else:
        o[unsafe_offset=t] = zz


def gpu_bias_act[OP: ElementOp, HAS_ACT: Bool](
    ctx: Dev, o: DeviceBuffer[DT], bias: DeviceBuffer[DT],
    z: DeviceBuffer[DT], m: Int, n: Int,
) raises:
    ctx.enqueue_function[_bias_act_kernel[OP, HAS_ACT]](
        o, bias, z, Int64(m), Int64(n), grid_dim=_grid(m * n), block_dim=TPB
    )


def _act_gate_kernel[OP: ElementOp](g: P, y: P, z: P, n: Int64):
    """`g = act'(·) ⊙ g` in place: the gradient of the PRE-activation, from
    the output's. `y` for an output-cache op, `z` otherwise."""
    var k = Int(global_idx.x)
    if k >= Int(n):
        return
    var cache = y[unsafe_offset=k] if OP.owns_cache else z[unsafe_offset=k]
    g[unsafe_offset=k] = OP.backward_scalar(cache, g[unsafe_offset=k])


def gpu_act_gate[OP: ElementOp](
    ctx: Dev, g: DeviceBuffer[DT], y: DeviceBuffer[DT],
    z: DeviceBuffer[DT], n: Int,
) raises:
    ctx.enqueue_function[_act_gate_kernel[OP]](
        g, y, z, Int64(n), grid_dim=_grid(n), block_dim=TPB
    )


def gpu_linear_lt_bias(
    ctx: Dev, o: DeviceBuffer[DT], x: DeviceBuffer[DT],
    w: DeviceBuffer[DT], bias: DeviceBuffer[DT], m: Int, n: Int, k: Int,
) raises:
    """NVIDIA: `o = x @ w + bias` in ONE launch, the bias in cuBLASLt's
    epilogue — `Linear`'s forward, at its precision rule."""
    comptime if default_accelerator().is_nvidia_gpu():
        if cublas_tf32(m, n, k):
            cublaslt_gemm[False, False, DT, DT, True](
                ctx.ctx, o, x, w, m, n, k, EPI_BIAS, bias=Int(bias.unsafe_ptr())
            )
        else:
            cublaslt_gemm[False, False, DT, DT, False](
                ctx.ctx, o, x, w, m, n, k, EPI_BIAS, bias=Int(bias.unsafe_ptr())
            )
    else:
        raise Error("gpu_linear_lt_bias: NVIDIA only")


# ── reductions ───────────────────────────────────────────────────────────
def _reduce_kernel[MEAN: Bool](o: P, a: P, n_: Int64):
    """ONE block: strided partial sums, then a fixed-order tree in shared
    memory — the same result every run."""
    var n = Int(n_)
    var tid = Int(thread_idx.x)
    var sh = shared_tile[DT, REDUCE_TPB]()
    var s: Scalar[DT] = 0
    var k = tid
    while k < n:
        s += a[unsafe_offset=k]
        k += REDUCE_TPB
    sh[tid] = s
    barrier()
    var active = REDUCE_TPB
    while active > 1:
        active //= 2
        if tid < active:
            sh[tid] = rebind[Scalar[DT]](sh[tid]) + rebind[Scalar[DT]](
                sh[tid + active]
            )
        barrier()
    if tid == 0:
        var v = rebind[Scalar[DT]](sh[0])
        comptime if MEAN:
            v = v / Scalar[DT](n)
        o[unsafe_offset=0] = v


def _reduce_bwd_kernel[ACC: Bool](ga: P, go: P, n: Int64, div: Int64):
    var k = Int(global_idx.x)
    if k >= Int(n):
        return
    var d = go[unsafe_offset=0] / Scalar[DT](Int(div))
    comptime if ACC:
        ga[unsafe_offset=k] += d
    else:
        ga[unsafe_offset=k] = d


def gpu_reduce[MEAN: Bool](
    ctx: Dev, o: DeviceBuffer[DT], a: DeviceBuffer[DT], n: Int
) raises:
    ctx.enqueue_function[_reduce_kernel[MEAN]](
        o, a, Int64(n), grid_dim=1, block_dim=REDUCE_TPB
    )


def gpu_reduce_bwd[ACC: Bool](
    ctx: Dev, ga: DeviceBuffer[DT], go: DeviceBuffer[DT], n: Int,
    div: Int,
) raises:
    ctx.enqueue_function[_reduce_bwd_kernel[ACC]](
        ga, go, Int64(n), Int64(div), grid_dim=_grid(n), block_dim=TPB
    )


def _sum_rows_kernel(o: P, a: P, r_: Int64, c_: Int64):
    var c = Int(c_)
    var j = Int(global_idx.x)
    if j >= c:
        return
    var s: Scalar[DT] = 0
    for i in range(Int(r_)):
        s += a[unsafe_offset=i * c + j]
    o[unsafe_offset=j] = s


def _sum_rows_bwd_kernel[ACC: Bool](ga: P, go: P, r_: Int64, c_: Int64):
    var c = Int(c_)
    var t = Int(global_idx.x)
    if t >= Int(r_) * c:
        return
    comptime if ACC:
        ga[unsafe_offset=t] += go[unsafe_offset=t % c]
    else:
        ga[unsafe_offset=t] = go[unsafe_offset=t % c]


def gpu_sum_rows(
    ctx: Dev, o: DeviceBuffer[DT], a: DeviceBuffer[DT], r: Int, c: Int
) raises:
    ctx.enqueue_function[_sum_rows_kernel](
        o, a, Int64(r), Int64(c), grid_dim=_grid(c), block_dim=TPB
    )


def gpu_sum_rows_bwd[ACC: Bool](
    ctx: Dev, ga: DeviceBuffer[DT], go: DeviceBuffer[DT], r: Int,
    c: Int,
) raises:
    ctx.enqueue_function[_sum_rows_bwd_kernel[ACC]](
        ga, go, Int64(r), Int64(c), grid_dim=_grid(r * c), block_dim=TPB
    )


# ── layout ───────────────────────────────────────────────────────────────
def _transpose_kernel[ACC: Bool](o: P, a: P, r_: Int64, c_: Int64):
    """`o[c, r] (+)= a[r, c]ᵀ`, one thread per element of `a`."""
    var r = Int(r_)
    var c = Int(c_)
    var t = Int(global_idx.x)
    if t >= r * c:
        return
    var i = t // c
    var j = t % c
    comptime if ACC:
        o[unsafe_offset=j * r + i] += a[unsafe_offset=t]
    else:
        o[unsafe_offset=j * r + i] = a[unsafe_offset=t]


def gpu_transpose[ACC: Bool = False](
    ctx: Dev, o: DeviceBuffer[DT], a: DeviceBuffer[DT], r: Int, c: Int
) raises:
    ctx.enqueue_function[_transpose_kernel[ACC]](
        o, a, Int64(r), Int64(c), grid_dim=_grid(r * c), block_dim=TPB
    )


def _axpy_kernel(dst: P, src: P, n: Int64, alpha: Scalar[DT]):
    """`dst += alpha · src` (alpha 1: a gradient accumulate; -lr: SGD)."""
    var k = Int(global_idx.x)
    if k < Int(n):
        dst[unsafe_offset=k] += alpha * src[unsafe_offset=k]


def gpu_axpy(
    ctx: Dev, dst: DeviceBuffer[DT], src: DeviceBuffer[DT], n: Int,
    alpha: Scalar[DT],
) raises:
    ctx.enqueue_function[_axpy_kernel](
        dst, src, Int64(n), alpha, grid_dim=_grid(n), block_dim=TPB
    )


def _copy_kernel(dst: P, src: P, n: Int64):
    var k = Int(global_idx.x)
    if k < Int(n):
        dst[unsafe_offset=k] = src[unsafe_offset=k]


def gpu_copy(
    ctx: Dev, dst: DeviceBuffer[DT], src: DeviceBuffer[DT], n: Int
) raises:
    ctx.enqueue_function[_copy_kernel](
        dst, src, Int64(n), grid_dim=_grid(n), block_dim=TPB
    )


# ── GEMM ─────────────────────────────────────────────────────────────────
def _naive_mm_kernel[TA: Bool, TB: Bool, ACC: Bool](
    o: P, a: P, b: P, m_: Int64, n_: Int64, k_: Int64
):
    """`o[M, N] (+)= op(a) @ op(b)`: op(a) = a[M, K] or a[K, M]ᵀ, op(b) =
    b[K, N] or b[N, K]ᵀ. One thread per output element."""
    var m = Int(m_)
    var n = Int(n_)
    var kk = Int(k_)
    var t = Int(global_idx.x)
    if t >= m * n:
        return
    var i = t // n
    var j = t % n
    var s: Scalar[DT] = 0
    for k in range(kk):
        var av = a[unsafe_offset=k * m + i] if TA else a[unsafe_offset=i * kk + k]
        var bv = b[unsafe_offset=j * kk + k] if TB else b[unsafe_offset=k * n + j]
        s += av * bv
    comptime if ACC:
        o[unsafe_offset=t] += s
    else:
        o[unsafe_offset=t] = s


def _naive_mm[TA: Bool, TB: Bool, ACC: Bool](
    ctx: Dev, o: DeviceBuffer[DT], a: DeviceBuffer[DT],
    b: DeviceBuffer[DT], m: Int, n: Int, k: Int,
) raises:
    ctx.enqueue_function[_naive_mm_kernel[TA, TB, ACC]](
        o, a, b, Int64(m), Int64(n), Int64(k), grid_dim=_grid(m * n),
        block_dim=TPB,
    )


comptime MM_T = 64  # output tile edge
comptime MM_K = 16  # k-slab per shared-memory stage
comptime MM_SUB = 16  # threads per tile edge (each owns 4 x 4 outputs)


def _tiled_mm_kernel[TA: Bool, TB: Bool, ACC: Bool](
    o: P, a: P, b: P, m_: Int64, n_: Int64, k_: Int64
):
    """`_naive_mm_kernel`'s product, 64 x 64 output tiles per threadgroup of
    16 x 16 threads, each thread 4 x 4 outputs (rows ty + 16 r, columns
    tx + 16 c), A and B staged through shared memory 16 k at a time. Each
    output still sums k = 0, 1, … in order: deterministic. Written for the
    metal_rt tape, which cannot call MAX's matmul."""
    var m = Int(m_)
    var n = Int(n_)
    var kk = Int(k_)
    var tx = Int(thread_idx.x)
    var ty = Int(thread_idx.y)
    var tid = ty * MM_SUB + tx
    var row0 = Int(block_idx.y) * MM_T
    var col0 = Int(block_idx.x) * MM_T
    var As = stack_allocation[
        MM_K * MM_T, Scalar[DT], address_space=AddressSpace.SHARED
    ]()
    var Bs = stack_allocation[
        MM_K * MM_T, Scalar[DT], address_space=AddressSpace.SHARED
    ]()
    var acc0 = SIMD[DT, 4](0)
    var acc1 = SIMD[DT, 4](0)
    var acc2 = SIMD[DT, 4](0)
    var acc3 = SIMD[DT, 4](0)
    var k0 = 0
    while k0 < kk:
        comptime for q in range(MM_K * MM_T // (MM_SUB * MM_SUB)):
            var e = tid + q * MM_SUB * MM_SUB
            # A tile: [MM_T rows] x [MM_K k], stored k-major
            var ia: Int
            var ka: Int
            comptime if TA:
                ka = e // MM_T
                ia = e % MM_T
            else:
                ia = e // MM_K
                ka = e % MM_K
            var gi = row0 + ia
            var gk = k0 + ka
            var av: Scalar[DT] = 0
            if gi < m and gk < kk:
                comptime if TA:
                    av = a[unsafe_offset=gk * m + gi]
                else:
                    av = a[unsafe_offset=gi * kk + gk]
            As[unsafe_offset=ka * MM_T + ia] = av
            # B tile: [MM_K k] x [MM_T cols]
            var jb: Int
            var kb: Int
            comptime if TB:
                jb = e // MM_K
                kb = e % MM_K
            else:
                kb = e // MM_T
                jb = e % MM_T
            var gj = col0 + jb
            var gk2 = k0 + kb
            var bv: Scalar[DT] = 0
            if gj < n and gk2 < kk:
                comptime if TB:
                    bv = b[unsafe_offset=gj * kk + gk2]
                else:
                    bv = b[unsafe_offset=gk2 * n + gj]
            Bs[unsafe_offset=kb * MM_T + jb] = bv
        barrier()
        var kend = min(MM_K, kk - k0)
        for t in range(kend):
            var bvec = SIMD[DT, 4](
                Bs[unsafe_offset=t * MM_T + tx],
                Bs[unsafe_offset=t * MM_T + tx + 16],
                Bs[unsafe_offset=t * MM_T + tx + 32],
                Bs[unsafe_offset=t * MM_T + tx + 48],
            )
            acc0 += As[unsafe_offset=t * MM_T + ty] * bvec
            acc1 += As[unsafe_offset=t * MM_T + ty + 16] * bvec
            acc2 += As[unsafe_offset=t * MM_T + ty + 32] * bvec
            acc3 += As[unsafe_offset=t * MM_T + ty + 48] * bvec
        barrier()
        k0 += MM_K
    comptime for r in range(4):
        var i = row0 + ty + 16 * r
        if i < m:
            var accr = acc0
            comptime if r == 1:
                accr = acc1
            elif r == 2:
                accr = acc2
            elif r == 3:
                accr = acc3
            comptime for c in range(4):
                var j = col0 + tx + 16 * c
                if j < n:
                    comptime if ACC:
                        o[unsafe_offset=i * n + j] += accr[c]
                    else:
                        o[unsafe_offset=i * n + j] = accr[c]


comptime MM_TILED_MIN = MM_T * MM_T * 8
"""Below 8 output tiles the tiled kernel leaves most of the GPU idle (one
threadgroup per tile); the naive one-thread-per-output kernel runs instead."""


def _own_mm[TA: Bool, TB: Bool, ACC: Bool](
    ctx: Dev, o: DeviceBuffer[DT], a: DeviceBuffer[DT],
    b: DeviceBuffer[DT], m: Int, n: Int, k: Int,
) raises:
    """The tape's own GEMM (no MAX): tiled for large outputs, naive else."""
    if m * n >= MM_TILED_MIN and k > 1:
        ctx.enqueue_function[_tiled_mm_kernel[TA, TB, ACC]](
            o, a, b, Int64(m), Int64(n), Int64(k),
            grid_dim=((n + MM_T - 1) // MM_T, (m + MM_T - 1) // MM_T),
            block_dim=(MM_SUB, MM_SUB),
        )
    else:
        _naive_mm[TA, TB, ACC](ctx, o, a, b, m, n, k)


def _max_mm[transpose_b: Bool = False](
    ctx: Dev, o: DeviceBuffer[DT], a: DeviceBuffer[DT],
    b: DeviceBuffer[DT], m: Int, n: Int, k: Int,
) raises:
    """`o[m, n] = a[m, k] @ b[k, n]` (`b[n, k]ᵀ` when `transpose_b`) on MAX's
    matmul, runtime-extent views. The output view needs a MUTABLE origin, so
    it is built over a local copy of the handle (same device memory).
    ⚠ `transpose_b` only with n >= 2 (MAX miscomputes N = 1)."""
    var o_mut = o
    var ov = TileTensor(o_mut, row_major(m, n))
    var av = TileTensor(a, row_major(m, k))
    comptime if transpose_b:
        var bv = TileTensor(b, row_major(n, k))
        max_matmul[transpose_b=True, target="gpu"](ov, av, bv, ctx.ctx)
    else:
        var bv = TileTensor(b, row_major(k, n))
        max_matmul[target="gpu"](ov, av, bv, ctx.ctx)


def _cublas[TA: Bool, TB: Bool](
    ctx: Dev, o: DeviceBuffer[DT], a: DeviceBuffer[DT],
    b: DeviceBuffer[DT], m: Int, n: Int, k: Int, beta: Float32,
) raises:
    """`cublas_gemm` with the comptime nn's precision rule applied at run
    time. A layer's three GEMMs share m·n·k, so they share a precision."""
    if cublas_tf32(m, n, k):
        cublas_gemm[TA, TB, True](ctx.ctx, o, a, b, m, n, k, beta)
    else:
        cublas_gemm[TA, TB, False](ctx.ctx, o, a, b, m, n, k, beta)


@always_inline
def _degenerate(m: Int, n: Int, k: Int) -> Bool:
    return n == 1 or k == 1 or m == 1


def gpu_matmul(
    ctx: Dev, o: DeviceBuffer[DT], a: DeviceBuffer[DT],
    b: DeviceBuffer[DT], m: Int, n: Int, k: Int,
) raises:
    """`o[m, n] = a[m, k] @ b[k, n]`."""
    comptime if default_accelerator().is_nvidia_gpu():
        _cublas[False, False](ctx, o, a, b, m, n, k, 0.0)
    else:
        if ctx.own:
            _own_mm[False, False, False](ctx, o, a, b, m, n, k)
        elif _degenerate(m, n, k):
            _naive_mm[False, False, False](ctx, o, a, b, m, n, k)
        else:
            _max_mm(ctx, o, a, b, m, n, k)


def gpu_matmul_bwd_a[ACC: Bool](
    ctx: Dev, ga: DeviceBuffer[DT], go: DeviceBuffer[DT],
    b: DeviceBuffer[DT], m: Int, n: Int, k: Int, tmp_o: DeviceBuffer[DT],
) raises:
    """`ga[m, k] (+)= go[m, n] @ b[k, n]ᵀ`. `tmp_o` >= m·k (used only when
    accumulating through MAX's matmul, which cannot)."""
    comptime if default_accelerator().is_nvidia_gpu():
        _cublas[False, True](
            ctx, ga, go, b, m, k, n, Float32(1.0) if ACC else Float32(0.0)
        )
    else:
        if ctx.own:
            _own_mm[False, True, ACC](ctx, ga, go, b, m, k, n)
        elif _degenerate(m, k, n):
            _naive_mm[False, True, ACC](ctx, ga, go, b, m, k, n)
        else:
            comptime if ACC:
                _max_mm[transpose_b=True](ctx, tmp_o, go, b, m, k, n)
                gpu_axpy(ctx, ga, tmp_o, m * k, 1.0)
            else:
                _max_mm[transpose_b=True](ctx, ga, go, b, m, k, n)


def gpu_matmul_bwd_b[ACC: Bool](
    ctx: Dev, gb: DeviceBuffer[DT], a: DeviceBuffer[DT],
    go: DeviceBuffer[DT], m: Int, n: Int, k: Int,
    tmp_t: DeviceBuffer[DT], tmp_o: DeviceBuffer[DT],
) raises:
    """`gb[k, n] (+)= a[m, k]ᵀ @ go[m, n]`. `tmp_t` >= m·k (aᵀ: MAX's matmul
    has no transpose_a), `tmp_o` >= k·n."""
    comptime if default_accelerator().is_nvidia_gpu():
        _cublas[True, False](
            ctx, gb, a, go, k, n, m, Float32(1.0) if ACC else Float32(0.0)
        )
    else:
        if ctx.own:
            _own_mm[True, False, ACC](ctx, gb, a, go, k, n, m)
        elif _degenerate(k, n, m):
            _naive_mm[True, False, ACC](ctx, gb, a, go, k, n, m)
        else:
            gpu_transpose(ctx, tmp_t, a, m, k)  # aᵀ: [k, m]
            comptime if ACC:
                _max_mm(ctx, tmp_o, tmp_t, go, k, n, m)
                gpu_axpy(ctx, gb, tmp_o, k * n, 1.0)
            else:
                _max_mm(ctx, gb, tmp_t, go, k, n, m)
