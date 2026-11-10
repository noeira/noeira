# +--------------------------------------------------------------------------+ #
# | LFM3 — the device kernels of the backward (VJPs of kernels.mojo)
# +--------------------------------------------------------------------------+ #
"""The vector-Jacobian products of the forward's operations, for LoRA training
(`train.mojo`). The base weights are frozen: nothing here produces a weight
gradient, only the gradient with respect to each operation's input. Same
conventions as `kernels.mojo` (fp32, runtime lengths, 2-D / 3-D grids, 32-bit
divisions). RoPE's VJP is the forward with `-sin` (`rope_kernel`, negated
tables), so it has no kernel of its own.
"""

from std.math import exp, sqrt
from max.gpu import global_idx, thread_idx, block_idx
from max.gpu.primitives import block

from noeira.nn.constants import DT
from .kernels import TPB, Ptr, HPtr, _bf, _sigmoid, _div


def rms_bwd_kernel(x: Ptr, w: HPtr, dy: Ptr, dx: Ptr, dim_n: Int64, eps: Float32):
    """`y = w * x * r`, `r = 1 / sqrt(mean(x^2) + eps)`, per row:
    `dx = r * (w dy) - x r^3 mean(x * w dy)`. A block per row."""
    var dim = Int(dim_n)
    var row = Int(block_idx.x) * dim
    var tid = Int(thread_idx.x)
    var ss = Scalar[DT](0)
    var sd = Scalar[DT](0)
    var d = tid
    while d < dim:
        var xv = x[unsafe_offset=row + d]
        ss += xv * xv
        sd += xv * _bf(w[unsafe_offset=d]) * dy[unsafe_offset=row + d]
        d += TPB
    var tss = block.sum[block_size=TPB, broadcast=True](val=ss)
    var tsd = block.sum[block_size=TPB, broadcast=True](val=sd)
    var r = Scalar[DT](1) / sqrt(tss / Scalar[DT](dim) + Scalar[DT](eps))
    var c = r * r * r * tsd / Scalar[DT](dim)
    d = tid
    while d < dim:
        dx[unsafe_offset=row + d] = r * _bf(w[unsafe_offset=d]) * dy[unsafe_offset=row + d] - x[unsafe_offset=row + d] * c
        d += TPB


def swiglu_bwd_kernel(a: Ptr, b: Ptr, dout: Ptr, da: Ptr, db: Ptr, n_n: Int64):
    """`out = silu(a) b`: `da = dout b silu'(a)`, `db = dout silu(a)`,
    `silu'(a) = s (1 + a (1 - s))`, `s = sigmoid(a)`."""
    var i = Int(global_idx.x)
    if i < Int(n_n):
        var x = a[unsafe_offset=i]
        var s = _sigmoid(x)
        var g = dout[unsafe_offset=i]
        da[unsafe_offset=i] = g * b[unsafe_offset=i] * s * (Scalar[DT](1) + x * (Scalar[DT](1) - s))
        db[unsafe_offset=i] = g * x * s


def short_conv_bwd_kernel(bcx: Ptr, w: HPtr, dz: Ptr, dbcx: Ptr, t_n: Int64, d_n: Int64, k_n: Int64):
    """`z[t] = C[t] * sum_k w[k] u[t - K + 1 + k]`, `u = B x` (per channel):
    `dC[t] = dz[t] conv(u)[t]`, `du[s] = sum_k w[k] (dz C)[s + K - 1 - k]`,
    `dB = du x`, `dx = du B`. Each (s, d) writes its own three entries — the
    conv's scatter is written as a gather, no atomics. Grid `(ceil(D / TPB), T)`."""
    var d = Int(block_idx.x) * TPB + Int(thread_idx.x)
    var s = Int(block_idx.y)
    var dim = Int(d_n)
    var t = Int(t_n)
    var kk = Int(k_n)
    if d >= dim:
        return
    var row = s * 3 * dim
    # dC at t = s: needs conv(u)[s]
    var conv = Scalar[DT](0)
    for k in range(kk):
        var src = s - (kk - 1) + k
        if src >= 0:
            var b2 = src * 3 * dim
            conv += _bf(w[unsafe_offset=d * kk + k]) * bcx[unsafe_offset=b2 + d] * bcx[unsafe_offset=b2 + 2 * dim + d]
    # du[s]: every output t = s + K - 1 - k that read u[s]
    var du = Scalar[DT](0)
    for k in range(kk):
        var tt = s + (kk - 1) - k
        if tt < t:
            du += _bf(w[unsafe_offset=d * kk + k]) * dz[unsafe_offset=tt * dim + d] * bcx[unsafe_offset=tt * 3 * dim + dim + d]
    dbcx[unsafe_offset=row + d] = du * bcx[unsafe_offset=row + 2 * dim + d]          # dB
    dbcx[unsafe_offset=row + dim + d] = dz[unsafe_offset=s * dim + d] * conv          # dC
    dbcx[unsafe_offset=row + 2 * dim + d] = du * bcx[unsafe_offset=row + d]           # dx


def gate_bwd_kernel(a_pre: Ptr, g: Ptr, da: Ptr, da_pre: Ptr, dg: Ptr, heads_n: Int64, hd_n: Int64):
    """`a = sigmoid(g) a_pre` per head: `da_pre = sigmoid(g) da`,
    `dg = sigmoid'(g) sum_d da a_pre`. A block per (token, head) — grid `T heads`,
    block `hd` (128) threads."""
    var heads = Int(heads_n)
    var hd = Int(hd_n)
    var ih = Int(block_idx.x)  # i * heads + h
    var d = Int(thread_idx.x)
    var s = _sigmoid(g[unsafe_offset=ih])
    var idx = ih * hd + d
    var v = da[unsafe_offset=idx]
    da_pre[unsafe_offset=idx] = s * v
    var tot = block.sum[block_size=128, broadcast=True](val=v * a_pre[unsafe_offset=idx])
    if d == 0:
        dg[unsafe_offset=ih] = s * (Scalar[DT](1) - s) * tot


def attn_dp_kernel(da_pre: Ptr, v: Ptr, p: Ptr, ds: Ptr, t_n: Int64, heads_n: Int64, kv_heads_n: Int64, hd_n: Int64):
    """`dP[h, i, j] = da_pre[i, h] . v[j, h / rep]` where `P > 0` (0 elsewhere:
    a masked score has no gradient). Grid `(ceil(T / TPB), T, heads)`."""
    var t = Int(t_n)
    var j = Int(block_idx.x) * TPB + Int(thread_idx.x)
    var i = Int(block_idx.y)
    var h = Int(block_idx.z)
    if j >= t:
        return
    var heads = Int(heads_n)
    var hd = Int(hd_n)
    var o = h * t * t + i * t + j
    if p[unsafe_offset=o] == Scalar[DT](0):
        ds[unsafe_offset=o] = Scalar[DT](0)
        return
    var kvh = _div(h, _div(heads, Int(kv_heads_n)))
    var ab = i * heads * hd + h * hd
    var vb = j * Int(kv_heads_n) * hd + kvh * hd
    var acc = Scalar[DT](0)
    for d in range(hd):
        acc += da_pre[unsafe_offset=ab + d] * v[unsafe_offset=vb + d]
    ds[unsafe_offset=o] = acc


def softmax_bwd_rows_kernel(p: Ptr, ds: Ptr, t_n: Int64, scale: Float32):
    """In place on `ds` (holding dP): `dS = scale P (dP - sum_j P dP)` — the
    softmax's VJP, with the score scale folded in. A block per row."""
    var t = Int(t_n)
    var row = Int(block_idx.x) * t
    var tid = Int(thread_idx.x)
    var s = Scalar[DT](0)
    var j = tid
    while j < t:
        s += p[unsafe_offset=row + j] * ds[unsafe_offset=row + j]
        j += TPB
    var tot = block.sum[block_size=TPB, broadcast=True](val=s)
    j = tid
    while j < t:
        ds[unsafe_offset=row + j] = Scalar[DT](scale) * p[unsafe_offset=row + j] * (ds[unsafe_offset=row + j] - tot)
        j += TPB


def attn_dq_kernel(ds: Ptr, k: Ptr, dq: Ptr, t_n: Int64, heads_n: Int64, kv_heads_n: Int64, hd_n: Int64):
    """`dq[i, h] = sum_j dS[h, i, j] k[j, h / rep]`. Grid `(ceil(heads hd / TPB), T)`."""
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
    var sb = h * t * t + i * t
    var acc = Scalar[DT](0)
    for j in range(t):
        acc += ds[unsafe_offset=sb + j] * k[unsafe_offset=j * kvw + kvh * hd + d]
    dq[unsafe_offset=i * heads * hd + col] = acc


def attn_dkv_kernel(
    ds: Ptr, p: Ptr, q: Ptr, da_pre: Ptr, dk: Ptr, dv: Ptr,
    t_n: Int64, heads_n: Int64, kv_heads_n: Int64, hd_n: Int64,
):
    """`dk[j, g] = sum_{h in g} sum_i dS[h, i, j] q[i, h]`,
    `dv[j, g] = sum_{h in g} sum_i P[h, i, j] da_pre[i, h]`.
    Grid `(ceil(kv_heads hd / TPB), T)`."""
    var t = Int(t_n)
    var heads = Int(heads_n)
    var kvn = Int(kv_heads_n)
    var hd = Int(hd_n)
    var col = Int(block_idx.x) * TPB + Int(thread_idx.x)
    var j = Int(block_idx.y)
    if col >= kvn * hd:
        return
    var g = _div(col, hd)
    var d = col - g * hd
    var rep = _div(heads, kvn)
    var ak = Scalar[DT](0)
    var av = Scalar[DT](0)
    for r in range(rep):
        var h = g * rep + r
        for i in range(t):
            var o = h * t * t + i * t + j
            var qi = i * heads * hd + h * hd + d
            ak += ds[unsafe_offset=o] * q[unsafe_offset=qi]
            av += p[unsafe_offset=o] * da_pre[unsafe_offset=qi]
    dk[unsafe_offset=j * kvn * hd + col] = ak
    dv[unsafe_offset=j * kvn * hd + col] = av


def lm_head_bwd_kernel(dl: Ptr, table: HPtr, dh: Ptr, v_n: Int64, d_n: Int64):
    """`dh[s, d] = sum_v dl[s, v] table[v, d]` — the tied head's transpose.
    Grid `(ceil(D / TPB), S)`."""
    var d = Int(block_idx.x) * TPB + Int(thread_idx.x)
    var s = Int(block_idx.y)
    var dim = Int(d_n)
    var v = Int(v_n)
    if d >= dim:
        return
    var acc = Scalar[DT](0)
    for k in range(v):
        acc += dl[unsafe_offset=s * v + k] * _bf(table[unsafe_offset=k * dim + d])
    dh[unsafe_offset=s * dim + d] = acc


def scatter_rows_kernel(src: Ptr, rows: Pointer[Scalar[DType.int32], MutAnyOrigin], dst: Ptr, d_n: Int64):
    """`dst[rows[s], :] += src[s, :]`. Grid `(ceil(D / TPB), S)` — rows distinct."""
    var d = Int(block_idx.x) * TPB + Int(thread_idx.x)
    var s = Int(block_idx.y)
    var dim = Int(d_n)
    if d < dim:
        var o = Int(rows[unsafe_offset=s]) * dim + d
        dst[unsafe_offset=o] = dst[unsafe_offset=o] + src[unsafe_offset=s * dim + d]


def axpy_kernel(dst: Ptr, src: Ptr, n_n: Int64, alpha: Float32):
    var i = Int(global_idx.x)
    if i < Int(n_n):
        dst[unsafe_offset=i] = dst[unsafe_offset=i] + Scalar[DT](alpha) * src[unsafe_offset=i]
