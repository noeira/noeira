# +--------------------------------------------------------------------------+ #
# | LFM3 — the device kernels of the text backbone (forward)
# +--------------------------------------------------------------------------+ #
"""One kernel per operation of `model.mojo`, fp32 activations, bf16 weights
read in place (`_bf`: a bf16 is the top half of an fp32, so widening is a
shift). Runtime lengths (`Int64` arguments), 2-D / 3-D grids so that no kernel
divides by a runtime value (a 64-bit divide is software on a GPU — Kokoro's
im2col lesson).

Each forward here is written so its VJP can sit beside it: no fused op hides
an intermediate the backward would need (the attention keeps its
probabilities, the short conv its inputs).
"""

from std.math import exp, log, sqrt
from std.memory import bitcast
from max.gpu import global_idx, thread_idx, block_idx
from max.gpu.primitives import block, warp

from noeira.nn.constants import DT


comptime TPB: Int = 256
comptime Ptr = Pointer[Scalar[DT], MutAnyOrigin]
comptime HPtr = Pointer[Scalar[DType.uint16], MutAnyOrigin]
comptime IPtr = Pointer[Scalar[DType.int32], MutAnyOrigin]
comptime NEG_INF: Float32 = -3.0e38


@always_inline
def _bf(x: Scalar[DType.uint16]) -> Scalar[DT]:
    return bitcast[DT](UInt32(x) << 16)


@always_inline
def _div(a: Int, b: Int) -> Int:
    """`a // b` for non-negative values below 2^32, in 32-bit arithmetic."""
    return Int(UInt32(a) // UInt32(b))


@always_inline
def _sigmoid(x: Scalar[DT]) -> Scalar[DT]:
    return Scalar[DT](1) / (Scalar[DT](1) + exp(-x))


def bf16_to_f32_kernel(src: HPtr, dst: Ptr, n_n: Int64):
    var i = Int(global_idx.x)
    if i < Int(n_n):
        dst[unsafe_offset=i] = _bf(src[unsafe_offset=i])


def embed_kernel(ids: IPtr, table: HPtr, dst: Ptr, d_n: Int64):
    """`out[t, :] = table[ids[t], :]`. Grid `(ceil(D / TPB), T)`."""
    var d = Int(block_idx.x) * TPB + Int(thread_idx.x)
    var t = Int(block_idx.y)
    var dim = Int(d_n)
    if d < dim:
        dst[unsafe_offset=t * dim + d] = _bf(table[unsafe_offset=Int(ids[unsafe_offset=t]) * dim + d])


def rms_kernel(x: Ptr, y: Ptr, w: HPtr, dim_n: Int64, eps: Float32):
    """`y[r, :] = w * x[r, :] / sqrt(mean(x[r, :]^2) + eps)`, a block per row;
    `w` has `dim` entries (a per-head norm: rows are heads). `y` may be `x`."""
    var dim = Int(dim_n)
    var row = Int(block_idx.x) * dim
    var tid = Int(thread_idx.x)
    var s = Scalar[DT](0)
    var d = tid
    while d < dim:
        var v = x[unsafe_offset=row + d]
        s += v * v
        d += TPB
    var tot = block.sum[block_size=TPB, broadcast=True](val=s)
    var inv = Scalar[DT](1) / sqrt(tot / Scalar[DT](dim) + Scalar[DT](eps))
    d = tid
    while d < dim:
        y[unsafe_offset=row + d] = _bf(w[unsafe_offset=d]) * (x[unsafe_offset=row + d] * inv)
        d += TPB


def add_kernel(dst: Ptr, src: Ptr, n_n: Int64):
    var i = Int(global_idx.x)
    if i < Int(n_n):
        dst[unsafe_offset=i] = dst[unsafe_offset=i] + src[unsafe_offset=i]


def short_conv_kernel(bcx: Ptr, w: HPtr, z: Ptr, d_n: Int64, k_n: Int64):
    """`z[t, d] = C[t, d] * sum_k w[d, k] * B[t-K+1+k, d] * x[t-K+1+k, d]`, zeros
    before the row's start; `bcx[t] = [B | C | x]` (3D wide). Grid `(ceil(D / TPB), T)`."""
    var d = Int(block_idx.x) * TPB + Int(thread_idx.x)
    var t = Int(block_idx.y)
    var dim = Int(d_n)
    var kk = Int(k_n)
    if d >= dim:
        return
    var acc = Scalar[DT](0)
    for k in range(kk):
        var s = t - (kk - 1) + k
        if s >= 0:
            var base = s * 3 * dim
            acc += _bf(w[unsafe_offset=d * kk + k]) * bcx[unsafe_offset=base + d] * bcx[unsafe_offset=base + 2 * dim + d]
    z[unsafe_offset=t * dim + d] = bcx[unsafe_offset=t * 3 * dim + dim + d] * acc


def rope_kernel(x: Ptr, cos_t: Ptr, sin_t: Ptr, heads_n: Int64, half_n: Int64):
    """Rotate-half RoPE in place on `x[T, heads, 2 half]`: pairs (i, i + half),
    `cos_t` / `sin_t` `[T, half]`. Grid `(ceil(heads half / TPB), T)`."""
    var half = Int(half_n)
    var heads = Int(heads_n)
    var idx = Int(block_idx.x) * TPB + Int(thread_idx.x)
    var t = Int(block_idx.y)
    if idx >= heads * half:
        return
    var h = _div(idx, half)
    var i = idx - h * half
    var base = t * heads * 2 * half + h * 2 * half
    var c = cos_t[unsafe_offset=t * half + i]
    var s = sin_t[unsafe_offset=t * half + i]
    var a = x[unsafe_offset=base + i]
    var b = x[unsafe_offset=base + half + i]
    x[unsafe_offset=base + i] = a * c - b * s
    x[unsafe_offset=base + half + i] = b * c + a * s


def scores_kernel(
    q: Ptr, k: Ptr, sc: Ptr, groups: IPtr, t_n: Int64, heads_n: Int64, kv_heads_n: Int64,
    hd_n: Int64, scale: Float32, mode_n: Int64, state_n: Int64, window_n: Int64,
):
    """`sc[h, i, j] = q[i, h] . k[j, h / rep] * scale`, or NEG_INF where `i` may
    not see `j`. Mode 0 (`set`, full attention): `i < state or j < state or
    group[i] == group[j]`; mode 1 (sliding): `i - window < j <= i`; mode 2:
    causal. Grid `(ceil(T / TPB), T, heads)`."""
    var t = Int(t_n)
    var j = Int(block_idx.x) * TPB + Int(thread_idx.x)
    var i = Int(block_idx.y)
    var h = Int(block_idx.z)
    if j >= t:
        return
    var heads = Int(heads_n)
    var hd = Int(hd_n)
    var mode = Int(mode_n)
    var sees: Bool
    if mode == 0:
        var s = Int(state_n)
        sees = i < s or j < s or groups[unsafe_offset=i] == groups[unsafe_offset=j]
    elif mode == 1:
        sees = j <= i and j > i - Int(window_n)
    else:
        sees = j <= i
    var out = h * t * t + i * t + j
    if not sees:
        sc[unsafe_offset=out] = Scalar[DT](NEG_INF)
        return
    var kvh = _div(h, _div(heads, Int(kv_heads_n)))
    var qb = i * heads * hd + h * hd
    var kb = j * Int(kv_heads_n) * hd + kvh * hd
    var acc = Scalar[DT](0)
    for d in range(hd):
        acc += q[unsafe_offset=qb + d] * k[unsafe_offset=kb + d]
    sc[unsafe_offset=out] = acc * Scalar[DT](scale)


def softmax_rows_kernel(sc: Ptr, t_n: Int64):
    """Each row of `sc[rows, T]` to its softmax, in place; a block per row."""
    var t = Int(t_n)
    var row = Int(block_idx.x) * t
    var tid = Int(thread_idx.x)
    var m = Scalar[DT](NEG_INF)
    var j = tid
    while j < t:
        m = max(m, sc[unsafe_offset=row + j])
        j += TPB
    var top = block.max[block_size=TPB, broadcast=True](val=m)
    var s = Scalar[DT](0)
    j = tid
    while j < t:
        var v = sc[unsafe_offset=row + j]
        var e = exp(v - top) if v > Scalar[DT](NEG_INF) else Scalar[DT](0)
        sc[unsafe_offset=row + j] = e
        s += e
        j += TPB
    var tot = block.sum[block_size=TPB, broadcast=True](val=s)
    var inv = Scalar[DT](1) / tot
    j = tid
    while j < t:
        sc[unsafe_offset=row + j] = sc[unsafe_offset=row + j] * inv
        j += TPB


def attn_out_kernel(
    p: Ptr, v: Ptr, gate: Ptr, dst: Ptr, t_n: Int64, heads_n: Int64, kv_heads_n: Int64, hd_n: Int64,
):
    """`out[i, h, d] = sigmoid(gate[i, h]) * sum_j p[h, i, j] v[j, h / rep, d]`.
    Grid `(ceil(heads hd / TPB), T)`."""
    var t = Int(t_n)
    var heads = Int(heads_n)
    var hd = Int(hd_n)
    var col = Int(block_idx.x) * TPB + Int(thread_idx.x)
    var i = Int(block_idx.y)
    if col >= heads * hd:
        return
    var h = _div(col, hd)
    var d = col - h * hd
    var kvh = _div(h, _div(heads, Int(kv_heads_n)))
    var kvw = Int(kv_heads_n) * hd
    var pb = h * t * t + i * t
    var acc = Scalar[DT](0)
    for j in range(t):
        acc += p[unsafe_offset=pb + j] * v[unsafe_offset=j * kvw + kvh * hd + d]
    dst[unsafe_offset=i * heads * hd + col] = acc * _sigmoid(gate[unsafe_offset=i * heads + h])


def swiglu_kernel(a: Ptr, b: Ptr, dst: Ptr, n_n: Int64):
    """`out = silu(a) * b`."""
    var i = Int(global_idx.x)
    if i < Int(n_n):
        var x = a[unsafe_offset=i]
        dst[unsafe_offset=i] = x * _sigmoid(x) * b[unsafe_offset=i]


def gather_rows_kernel(x: Ptr, rows: IPtr, dst: Ptr, d_n: Int64):
    """`out[s, :] = x[rows[s], :]`. Grid `(ceil(D / TPB), S)`."""
    var d = Int(block_idx.x) * TPB + Int(thread_idx.x)
    var s = Int(block_idx.y)
    var dim = Int(d_n)
    if d < dim:
        dst[unsafe_offset=s * dim + d] = x[unsafe_offset=Int(rows[unsafe_offset=s]) * dim + d]


def lm_head_kernel(h: Ptr, table: HPtr, logits: Ptr, v_n: Int64, d_n: Int64):
    """`logits[s, v] = h[s] . table[v]` (the tied head, bf16 rows read in place):
    a warp per vocabulary row, 8 rows per block. Grid `(ceil(V / 8), S)`."""
    var tid = Int(thread_idx.x)
    var vrow = Int(block_idx.x) * 8 + (tid >> 5)
    var lane = tid & 31
    var s = Int(block_idx.y)
    var dim = Int(d_n)
    var acc = Scalar[DT](0)
    if vrow < Int(v_n):
        var hb = s * dim
        var tb = vrow * dim
        var d = lane
        while d < dim:
            acc += h[unsafe_offset=hb + d] * _bf(table[unsafe_offset=tb + d])
            d += 32
    acc = warp.sum(acc)
    if lane == 0 and vrow < Int(v_n):
        logits[unsafe_offset=s * Int(v_n) + vrow] = acc


def log_softmax_rows_kernel(x: Ptr, v_n: Int64):
    """Each row of `x[S, V]` to its log-softmax, in place; a block per row."""
    var v = Int(v_n)
    var row = Int(block_idx.x) * v
    var tid = Int(thread_idx.x)
    var m = Scalar[DT](NEG_INF)
    var j = tid
    while j < v:
        m = max(m, x[unsafe_offset=row + j])
        j += TPB
    var top = block.max[block_size=TPB, broadcast=True](val=m)
    var s = Scalar[DT](0)
    j = tid
    while j < v:
        s += exp(x[unsafe_offset=row + j] - top)
        j += TPB
    var tot = block.sum[block_size=TPB, broadcast=True](val=s)
    var lz = top + log(tot)
    j = tid
    while j < v:
        x[unsafe_offset=row + j] = x[unsafe_offset=row + j] - lz
        j += TPB

