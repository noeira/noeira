# +--------------------------------------------------------------------------+ #
# | LFM3-VL — the device kernels of the vision tower and its projector
# +--------------------------------------------------------------------------+ #
"""The kernels `vision.mojo` adds to the text backbone's (`kernels.mojo`).

TOKEN ORDER. A picture's patches are in MERGE-BLOCK order, the processor's
(`Lfm3VlImageProcessor.patchify`): the 2x2 blocks of patches row by row, and
inside a block its four patches row by row. Token `n` of a `gh x gw` grid is

    n = ((bh * gw/2 + bw) * 2 + mh) * 2 + mw,   patch row = 2 bh + mh, col = 2 bw + mw

so the projector's 2x2 merge is four CONSECUTIVE tokens, and the conv (a
raster-grid operation) maps through `_token`. Written once here (`_row_col`,
`_token`) for the patchifier and the conv.
"""

from std.math import erf, sqrt
from max.gpu import global_idx, thread_idx, block_idx
from max.gpu.primitives import block

from noeira.nn.constants import DT
from .kernels import TPB, Ptr, HPtr, IPtr, _bf, _div


comptime U8Ptr = Pointer[Scalar[DType.uint8], MutAnyOrigin]


@always_inline
def _row_col(n: Int, gw: Int) -> Tuple[Int, Int]:
    """Patch row and column of token `n` (merge-block order, merge 2)."""
    var mw = n & 1
    var mh = (n >> 1) & 1
    var blk = n >> 2
    var bh = _div(blk, gw >> 1)
    var bw = blk - bh * (gw >> 1)
    return (bh * 2 + mh, bw * 2 + mw)


@always_inline
def _token(r: Int, c: Int, gw: Int) -> Int:
    """The token at patch row `r`, column `c` (merge-block order, merge 2)."""
    return (((r >> 1) * (gw >> 1) + (c >> 1)) * 2 + (r & 1)) * 2 + (c & 1)


def patchify_kernel(img: U8Ptr, dst: Ptr, w_n: Int64, gw_n: Int64, patch_n: Int64, tp_n: Int64, debug: Int64):
    """`dst[n, (c T + t) P^2 + py P + px] = (img[y, x, c] - 127.5) / 127.5`
    for every temporal copy `t` (a still picture repeats), `img` [H, W, 3]
    uint8; transformers' fused rescale-and-normalise, fp32. Grid
    `(ceil(3 T P^2 / TPB), N)`. `debug` 1: the tokens in RASTER order (a wrong
    patch order, for the falsification)."""
    var p = Int(patch_n)
    var tp = Int(tp_n)
    var e = Int(block_idx.x) * TPB + Int(thread_idx.x)
    var n = Int(block_idx.y)
    var per_c = tp * p * p
    if e >= 3 * per_c:
        return
    var c = _div(e, per_c)
    var rest = e - c * per_c
    var inner = rest - _div(rest, p * p) * (p * p)   # the temporal copy is dropped: every copy is the picture
    var py = _div(inner, p)
    var px = inner - py * p
    var gw = Int(gw_n)
    var r: Int
    var col: Int
    if debug == 1:
        r = _div(n, gw)
        col = n - r * gw
    else:
        var rc = _row_col(n, gw)
        r = rc[0]
        col = rc[1]
    var y = r * p + py
    var x = col * p + px
    var u = Float32(img[unsafe_offset=(y * Int(w_n) + x) * 3 + c])
    dst[unsafe_offset=n * 3 * per_c + e] = (u - Float32(127.5)) / Float32(127.5)


def conv2d_gated_kernel(bcx: Ptr, w: HPtr, z: Ptr, d_n: Int64, gh_n: Int64, gw_n: Int64, debug: Int64):
    """The gated depthwise conv of a vision conv layer: `bcx[n] = [B | C | x]`
    (3D wide), `z[n, d] = C[n, d] * sum_{ky, kx} w[d, ky, kx] (B x)[r + ky - 1,
    c + kx - 1, d]` over the patch grid, zero outside it (padding 1, a 3x3
    kernel `w` [D, 1, 3, 3]). Grid `(ceil(D / TPB), N)`. `debug` 2: the
    neighbours taken in TOKEN order, not on the grid (for the falsification)."""
    var d = Int(block_idx.x) * TPB + Int(thread_idx.x)
    var n = Int(block_idx.y)
    var dim = Int(d_n)
    if d >= dim:
        return
    var gh = Int(gh_n)
    var gw = Int(gw_n)
    var rc = _row_col(n, gw)
    if debug == 2:
        var rr = _div(n, gw)
        rc = (rr, n - rr * gw)
    var acc = Scalar[DT](0)
    for ky in range(3):
        var r = rc[0] + ky - 1
        if r < 0 or r >= gh:
            continue
        for kx in range(3):
            var c = rc[1] + kx - 1
            if c < 0 or c >= gw:
                continue
            var m = _token(r, c, gw) if debug != 2 else r * gw + c
            var base = m * 3 * dim
            acc += _bf(w[unsafe_offset=d * 9 + ky * 3 + kx]) * bcx[unsafe_offset=base + d] * bcx[unsafe_offset=base + 2 * dim + d]
    z[unsafe_offset=n * dim + d] = bcx[unsafe_offset=n * 3 * dim + dim + d] * acc


def layer_norm_kernel(x: Ptr, y: Ptr, w: HPtr, b: HPtr, dim_n: Int64, eps: Float32):
    """`y[r, :] = (x[r, :] - mean) / sqrt(var + eps) * w + b` (biased variance,
    torch's `LayerNorm`), a block per row."""
    var dim = Int(dim_n)
    var row = Int(block_idx.x) * dim
    var tid = Int(thread_idx.x)
    var s = Scalar[DT](0)
    var d = tid
    while d < dim:
        s += x[unsafe_offset=row + d]
        d += TPB
    var mean = block.sum[block_size=TPB, broadcast=True](val=s) / Scalar[DT](dim)
    var q = Scalar[DT](0)
    d = tid
    while d < dim:
        var v = x[unsafe_offset=row + d] - mean
        q += v * v
        d += TPB
    var inv = Scalar[DT](1) / sqrt(block.sum[block_size=TPB, broadcast=True](val=q) / Scalar[DT](dim) + Scalar[DT](eps))
    d = tid
    while d < dim:
        y[unsafe_offset=row + d] = (x[unsafe_offset=row + d] - mean) * inv * _bf(w[unsafe_offset=d]) + _bf(b[unsafe_offset=d])
        d += TPB


def bias_kernel(x: Ptr, b: HPtr, cols_n: Int64, n_n: Int64):
    """`x[r, c] += b[c]` over `n` elements of `x` [rows, cols]."""
    var i = Int(global_idx.x)
    if i < Int(n_n):
        var cols = Int(cols_n)
        x[unsafe_offset=i] = x[unsafe_offset=i] + _bf(b[unsafe_offset=i - _div(i, cols) * cols])


def gelu_kernel(x: Ptr, n_n: Int64):
    """Exact GELU in place: `x / 2 (1 + erf(x / sqrt 2))` (torch's default)."""
    var i = Int(global_idx.x)
    if i < Int(n_n):
        var v = x[unsafe_offset=i]
        x[unsafe_offset=i] = Scalar[DT](0.5) * v * (Scalar[DT](1) + erf(v * Scalar[DT](0.7071067811865476)))


def put_rows_kernel(src: Ptr, rows: IPtr, dst: Ptr, d_n: Int64):
    """`dst[rows[s], :] = src[s, :]`: the image features over their tokens'
    text embeddings. Grid `(ceil(D / TPB), S)`."""
    var d = Int(block_idx.x) * TPB + Int(thread_idx.x)
    var s = Int(block_idx.y)
    var dim = Int(d_n)
    if d < dim:
        dst[unsafe_offset=Int(rows[unsafe_offset=s]) * dim + d] = src[unsafe_offset=s * dim + d]
