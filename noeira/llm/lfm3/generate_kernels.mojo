# +--------------------------------------------------------------------------+ #
# | LFM3 generation — the device kernels of a forward on a cache
# +--------------------------------------------------------------------------+ #
"""What `generate.mojo` adds to the backbone's kernels (`kernels.mojo`): a chunk
of new tokens at position `p` reads the tokens before it from a cache instead of
recomputing them.

- The short conv reads its `K - 1` earlier rows of `[B | C | x]` from a history
  (`conv_cached_kernel`); `conv_history_kernel` rolls that history forward.
- Attention: `put_kv_kernel` writes the chunk's keys / values into a heads-major
  cache `[kv_heads, capacity, head_dim]`; `cached_attention_kernel` runs each
  query against the keys it may see — causal, and on the sliding layers the
  last `window` positions — in one block per (query, head).
"""

from std.math import exp
from std.memory import stack_allocation
from max.gpu import thread_idx, block_idx
from max.gpu.sync import barrier
from max.gpu.memory import AddressSpace
from max.gpu.primitives import block

from noeira.nn.constants import DT
from .kernels import TPB, Ptr, HPtr, NEG_INF, _bf, _div, _sigmoid


comptime MAX_KEYS: Int = 8192
"""The most keys one query reads (`cached_attention_kernel` holds its scores in
shared memory: 32 KB). The cache's capacity is capped to it."""


def conv_cached_kernel(bcx: Ptr, hist: Ptr, w: HPtr, z: Ptr, d_n: Int64, k_n: Int64):
    """`short_conv_kernel` over a chunk whose earlier rows come from `hist`
    (`[K-1, 3D]`: the `[B | C | x]` rows of the K-1 tokens before the chunk,
    zeros before the conversation's start). The same products in the same
    order: a chunk gives what one pass over the whole sequence gives.
    Grid `(ceil(D / TPB), n)`."""
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
        else:
            var hb = (kk - 1 + s) * 3 * dim
            acc += _bf(w[unsafe_offset=d * kk + k]) * hist[unsafe_offset=hb + d] * hist[unsafe_offset=hb + 2 * dim + d]
    z[unsafe_offset=t * dim + d] = bcx[unsafe_offset=t * 3 * dim + dim + d] * acc


def conv_history_kernel(bcx: Ptr, hist: Ptr, dst: Ptr, n_n: Int64, w_n: Int64, k_n: Int64):
    """`dst` = the last K-1 rows of `[hist ; bcx]` (`hist` K-1 rows, the chunk
    `n` rows, each `w` = 3D wide): the history after the chunk. A separate
    buffer — the chunk may be shorter than the history, so in place would read
    rows it already overwrote. Grid `(ceil(w / TPB), K-1)`."""
    var c = Int(block_idx.x) * TPB + Int(thread_idx.x)
    var r = Int(block_idx.y)
    var wide = Int(w_n)
    if c >= wide:
        return
    var k1 = Int(k_n) - 1
    var src = Int(n_n) + r  # an index into [hist ; bcx]
    if src < k1:
        dst[unsafe_offset=r * wide + c] = hist[unsafe_offset=src * wide + c]
    else:
        dst[unsafe_offset=r * wide + c] = bcx[unsafe_offset=(src - k1) * wide + c]


def put_kv_kernel(src: Ptr, cache: Ptr, p_n: Int64, cap_n: Int64, kvh_n: Int64, hd_n: Int64):
    """`cache[kv, p + i, d] = src[i, kv, d]`: the chunk's rows into the
    heads-major cache at their absolute positions. Grid `(ceil(kvh hd / TPB), n)`."""
    var hd = Int(hd_n)
    var col = Int(block_idx.x) * TPB + Int(thread_idx.x)
    var i = Int(block_idx.y)
    var kvh = Int(kvh_n)
    if col >= kvh * hd:
        return
    var kv = _div(col, hd)
    var d = col - kv * hd
    cache[unsafe_offset=(kv * Int(cap_n) + Int(p_n) + i) * hd + d] = src[unsafe_offset=i * kvh * hd + col]


def cached_attention_kernel(
    q: Ptr, kc: Ptr, vc: Ptr, gate: Ptr, dst: Ptr, off_n: Int64, nh_n: Int64, nkv_n: Int64, hd_n: Int64,
    cap_n: Int64, window_n: Int64, scale: Float32,
):
    """Query `i` of the chunk (absolute position `off + i`), head `h`, against
    the cached keys it sees: `j <= off + i`, and `j > off + i - window` when
    `window > 0` (a sliding layer). `dst[i, h, :] = sigmoid(gate[i, h]) *
    softmax_j(scale q . k_j) . v_j`. `q` `[n, nh, hd]` after its norm and RoPE;
    `kc` / `vc` `[nkv, cap, hd]`. One block per (query, head); grid `(n, nh)`."""
    var i = Int(block_idx.x)
    var h = Int(block_idx.y)
    var tid = Int(thread_idx.x)
    var nh = Int(nh_n)
    var hd = Int(hd_n)
    var cap = Int(cap_n)
    var window = Int(window_n)
    var qi = Int(off_n) + i
    var lo = 0
    if window > 0 and qi - window + 1 > 0:
        lo = qi - window + 1
    var nk = qi + 1 - lo
    var kvh = _div(h, _div(nh, Int(nkv_n)))
    var qb = (i * nh + h) * hd
    var kb = kvh * cap * hd
    var sh = stack_allocation[MAX_KEYS, Scalar[DT], address_space=AddressSpace.SHARED]()
    var m = Scalar[DT](NEG_INF)
    var j = tid
    while j < nk:
        var kr = kb + (lo + j) * hd
        var acc = Scalar[DT](0)
        for d in range(hd):
            acc += q[unsafe_offset=qb + d] * kc[unsafe_offset=kr + d]
        var s = acc * Scalar[DT](scale)
        sh[j] = s
        m = max(m, s)
        j += TPB
    var top = block.max[block_size=TPB, broadcast=True](val=m)
    var tot_t = Scalar[DT](0)
    j = tid
    while j < nk:
        var e = exp(sh[j] - top)
        sh[j] = e
        tot_t += e
        j += TPB
    var tot = block.sum[block_size=TPB, broadcast=True](val=tot_t)
    barrier()  # every score's exp written before any thread reads them all
    if tid < hd:
        var o = Scalar[DT](0)
        var vb = kb + lo * hd + tid
        for jj in range(nk):
            o += sh[jj] * vc[unsafe_offset=vb + jj * hd]
        var g = _sigmoid(gate[unsafe_offset=i * nh + h])
        dst[unsafe_offset=(i * nh + h) * hd + tid] = g * (o / tot)


def masked_softmax_offset_kernel(sc: Ptr, n_n: Int64, l_n: Int64, off_n: Int64, window_n: Int64, scale: Float32):
    """Row `(h, i)` of the raw scores `sc[h, i, :L] = q_i . k_j` (a chunk's `n`
    queries against the cache's `L` keys) to attention probabilities in place:
    query `i` is at position `off + i` and sees `j <= off + i`, and `j > off + i
    - window` when `window > 0`; 0 elsewhere. A block per row; grid `(n, heads)`."""
    var n = Int(n_n)
    var l = Int(l_n)
    var i = Int(block_idx.x)
    var h = Int(block_idx.y)
    var row = (h * n + i) * l
    var tid = Int(thread_idx.x)
    var qi = Int(off_n) + i
    var window = Int(window_n)
    var lo = 0
    if window > 0 and qi - window + 1 > 0:
        lo = qi - window + 1
    var m = Scalar[DT](NEG_INF)
    var j = tid
    while j < l:
        var v = Scalar[DT](NEG_INF)
        if j >= lo and j <= qi:
            v = sc[unsafe_offset=row + j] * Scalar[DT](scale)
        sc[unsafe_offset=row + j] = v
        m = max(m, v)
        j += TPB
    var top = block.max[block_size=TPB, broadcast=True](val=m)
    var s = Scalar[DT](0)
    j = tid
    while j < l:
        var v = sc[unsafe_offset=row + j]
        var e = exp(v - top) if v > Scalar[DT](NEG_INF) else Scalar[DT](0)
        sc[unsafe_offset=row + j] = e
        s += e
        j += TPB
    var inv = Scalar[DT](1) / block.sum[block_size=TPB, broadcast=True](val=s)
    j = tid
    while j < l:
        sc[unsafe_offset=row + j] = sc[unsafe_offset=row + j] * inv
        j += TPB
