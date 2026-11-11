# +--------------------------------------------------------------------------+ #
# | RF-DETR — the kernels its port needs beyond SAM 3's (`sam3.ops`)
# +--------------------------------------------------------------------------+ #
"""Token-major throughout (`[row, channel]`, a feature map's row = y W + x), as
`noeira.perception.sam3`: a 1x1 conv is a linear, a 3x3 conv nine GEMMs on
shifted copies (`conv3x3_tm`), a depthwise 3x3 its own kernel. The shared
pieces — the safetensors weights, linear, LayerNorm, GELU, `mha` — are
`sam3.ops`'.
"""

from std.math import exp, floor
from max.gpu import thread_idx, block_idx, global_idx
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.nn.constants import DT
from noeira.perception.sam3.ops import TPB, Ptr, blocks, gemm_tb, add_kernel, _d
from noeira.perception.sam3.neck import shift_kernel, tap_weights_kernel


comptime IPtr = Pointer[Scalar[DType.int32], MutAnyOrigin]

# The Seg-Small geometry (384 input, patch 12, 2 x 2 windows).
comptime IMG: Int = 384
comptime PATCH: Int = 12
comptime GRID: Int = IMG // PATCH
"""32."""
comptime NTOK: Int = GRID * GRID
comptime NWIN: Int = 2
comptime WIN: Int = GRID // NWIN
"""16: a window is 16 x 16 tokens."""
comptime WTOK: Int = WIN * WIN + 1
"""257: a window's tokens and its CLS copy."""
comptime BB_DIM: Int = 384


def patchify_kernel(px: Ptr, dst: Ptr):
    """`dst[n, c 144 + ky 12 + kx] = px[c, 12 r + ky, 12 col + kx]`, n = r 32 + col
    (the patch conv's weight order). Grid `(ceil(432 / TPB), NTOK)`."""
    var e = Int(block_idx.x) * TPB + Int(thread_idx.x)
    var n = Int(block_idx.y)
    if e >= 3 * PATCH * PATCH:
        return
    var c = e // (PATCH * PATCH)
    var rem = e - c * PATCH * PATCH
    var ky = rem // PATCH
    var kx = rem - ky * PATCH
    var r = n // GRID
    var col = n - r * GRID
    dst[unsafe_offset=n * 3 * PATCH * PATCH + e] = px[unsafe_offset=(c * IMG + r * PATCH + ky) * IMG + col * PATCH + kx]


@always_inline
def _win_raster(m: Int) -> Int:
    """Window-order token `m` (window w = m // 256 row-major in the 2 x 2, then
    its 16 x 16 tokens row-major) -> its raster index."""
    var w = m // (WIN * WIN)
    var i = m - w * WIN * WIN
    var wr = w // NWIN
    var wc = w - wr * NWIN
    return (wr * WIN + i // WIN) * GRID + wc * WIN + (i % WIN)


def window_embed_kernel(tok: Ptr, cls: Ptr, pos: Ptr, dst: Ptr):
    """`dst [4 x 257, D]`: per window its CLS (+ pos[0]) then its 256 tokens
    (+ their pos[1 + raster]). Grid `(ceil(D / TPB), 4 x 257)`."""
    var d = Int(block_idx.x) * TPB + Int(thread_idx.x)
    var row = Int(block_idx.y)
    if d >= BB_DIM:
        return
    var w = row // WTOK
    var j = row - w * WTOK
    if j == 0:
        dst[unsafe_offset=row * BB_DIM + d] = cls[unsafe_offset=d] + pos[unsafe_offset=d]
        return
    var n = _win_raster(w * WIN * WIN + j - 1)
    dst[unsafe_offset=row * BB_DIM + d] = tok[unsafe_offset=n * BB_DIM + d] + pos[unsafe_offset=(1 + n) * BB_DIM + d]


def unwindow_kernel(src: Ptr, dst: Ptr, dst_cols: Int64, col_off: Int64):
    """`dst[n, col_off + d] = src[window row of raster token n, d]` (CLS rows
    dropped): a tap back in raster order, into its slice of the projector's
    concatenation. Grid `(ceil(D / TPB), NTOK)`."""
    var d = Int(block_idx.x) * TPB + Int(thread_idx.x)
    var m = Int(block_idx.y)
    if d >= BB_DIM:
        return
    var w = m // (WIN * WIN)
    var n = _win_raster(m)
    dst[unsafe_offset=n * Int(dst_cols) + Int(col_off) + d] = src[unsafe_offset=(m + w + 1) * BB_DIM + d]


def scale_add_kernel(x: Ptr, a: Ptr, lam: Ptr, cols_n: Int64):
    """`x[r, c] += lam[c] a[r, c]` (DINOv2's layer scale). Grid `(rows, ceil(cols / TPB))`."""
    var c = Int(block_idx.y) * TPB + Int(thread_idx.x)
    var cols = Int(cols_n)
    if c < cols:
        var i = Int(block_idx.x) * cols + c
        x[unsafe_offset=i] = x[unsafe_offset=i] + lam[unsafe_offset=c] * a[unsafe_offset=i]


def silu_kernel(x: Ptr, n_n: Int64):
    var i = Int(global_idx.x)
    if i < Int(n_n):
        var v = x[unsafe_offset=i]
        x[unsafe_offset=i] = v / (Scalar[DT](1) + exp(-v))


def copy_cols_kernel(src: Ptr, src_cols: Int64, src_off: Int64, dst: Ptr, dst_cols: Int64, dst_off: Int64, width: Int64):
    """`dst[r, dst_off + c] = src[r, src_off + c]`, c < width. Grid `(rows, ceil(width / TPB))`."""
    var c = Int(block_idx.y) * TPB + Int(thread_idx.x)
    if c < Int(width):
        var r = Int(block_idx.x)
        dst[unsafe_offset=r * Int(dst_cols) + Int(dst_off) + c] = src[unsafe_offset=r * Int(src_cols) + Int(src_off) + c]


def gather_rows_kernel(src: Ptr, idx: IPtr, dst: Ptr, cols_n: Int64):
    """`dst[i, :] = src[idx[i], :]`. Grid `(n, ceil(cols / TPB))`."""
    var c = Int(block_idx.y) * TPB + Int(thread_idx.x)
    var cols = Int(cols_n)
    if c < cols:
        var i = Int(block_idx.x)
        dst[unsafe_offset=i * cols + c] = src[unsafe_offset=Int(idx[unsafe_offset=i]) * cols + c]


def add_scalar_kernel(x: Ptr, b: Ptr, n_n: Int64):
    var i = Int(global_idx.x)
    if i < Int(n_n):
        x[unsafe_offset=i] = x[unsafe_offset=i] + b[unsafe_offset=0]


def msda_kernel(value: Ptr, offs: Ptr, attw: Ptr, rb: Ptr, dst: Ptr, w_n: Int64, h_n: Int64, swap_xy: Int64):
    """Multi-scale deformable attention, one level, 16 heads x 16, 2 points:
    `out[q, 16 h + d] = sum_p softmax_p(attw[q, 2 h + p]) bilinear(value[:, 16 h + d],
    ref_xy + off[q, h, p] / 2 * ref_wh * 0.5)` — `MSDeformAttn` with
    `grid_sample(bilinear, zeros, align_corners=False)`: pixel `loc W - 0.5`, a
    neighbour outside the map counts 0. Grid `(queries, 16)`, 16 threads.
    `swap_xy` (falsification only): the offsets' x and y swapped."""
    comptime HEADS = 16
    comptime HDIM = 16
    comptime CH = HEADS * HDIM
    var q = Int(block_idx.x)
    var h = Int(block_idx.y)
    var d = Int(thread_idx.x)
    var W = Int(w_n)
    var H = Int(h_n)
    var rx = rb[unsafe_offset=q * 4]
    var ry = rb[unsafe_offset=q * 4 + 1]
    var rw = rb[unsafe_offset=q * 4 + 2]
    var rh = rb[unsafe_offset=q * 4 + 3]
    var a0 = attw[unsafe_offset=q * 2 * HEADS + h * 2]
    var a1 = attw[unsafe_offset=q * 2 * HEADS + h * 2 + 1]
    var mx = max(a0, a1)
    var e0 = exp(a0 - mx)
    var e1 = exp(a1 - mx)
    var s = e0 + e1
    var acc = Scalar[DT](0)
    for p in range(2):
        var ox = offs[unsafe_offset=q * 4 * HEADS + (h * 2 + p) * 2]
        var oy = offs[unsafe_offset=q * 4 * HEADS + (h * 2 + p) * 2 + 1]
        if swap_xy != 0:
            var t = ox
            ox = oy
            oy = t
        var lx = rx + ox / Scalar[DT](2) * rw * Scalar[DT](0.5)
        var ly = ry + oy / Scalar[DT](2) * rh * Scalar[DT](0.5)
        var px = lx * Scalar[DT](W) - Scalar[DT](0.5)
        var py = ly * Scalar[DT](H) - Scalar[DT](0.5)
        var fx0 = floor(px)
        var fy0 = floor(py)
        var x0 = Int(fx0)
        var y0 = Int(fy0)
        var tx = px - fx0
        var ty = py - fy0
        var v = Scalar[DT](0)
        for dy in range(2):
            var yy = y0 + dy
            var wy = ty if dy == 1 else Scalar[DT](1) - ty
            for dx in range(2):
                var xx = x0 + dx
                var wx = tx if dx == 1 else Scalar[DT](1) - tx
                if xx >= 0 and xx < W and yy >= 0 and yy < H:
                    v += wx * wy * value[unsafe_offset=(yy * W + xx) * CH + h * HDIM + d]
        var wp = (e0 if p == 0 else e1) / s
        acc += v * wp
    dst[unsafe_offset=q * CH + h * HDIM + d] = acc


def dwconv3x3_kernel(x: Ptr, w: Ptr, b: Ptr, dst: Ptr, h_n: Int64, w_n: Int64, c_n: Int64):
    """Depthwise 3x3, padding 1, token-major: `dst[n, c] = b[c] + sum w[c, ky, kx]
    x[(y + ky - 1, x + kx - 1), c]` (zeros outside). Grid `(H W, ceil(C / TPB))`."""
    var c = Int(block_idx.y) * TPB + Int(thread_idx.x)
    var n = Int(block_idx.x)
    var C = Int(c_n)
    if c >= C:
        return
    var W = Int(w_n)
    var H = Int(h_n)
    var y = _d(n, W)
    var xx = n - y * W
    var s = b[unsafe_offset=c]
    for ky in range(3):
        var yy = y + ky - 1
        if yy < 0 or yy >= H:
            continue
        for kx in range(3):
            var x2 = xx + kx - 1
            if x2 < 0 or x2 >= W:
                continue
            s += w[unsafe_offset=c * 9 + ky * 3 + kx] * x[unsafe_offset=(yy * W + x2) * C + c]
    dst[unsafe_offset=n * C + c] = s


def upsample_bilinear_kernel(src: Ptr, dst: Ptr, hi_n: Int64, wi_n: Int64, ho_n: Int64, wo_n: Int64, c_n: Int64):
    """`F.interpolate(bilinear, align_corners=False)` token-major: source pixel
    `max((o + 0.5) in / out - 0.5, 0)`, its floor and the next (clamped).
    Grid `(Ho Wo, ceil(C / TPB))`."""
    var c = Int(block_idx.y) * TPB + Int(thread_idx.x)
    var n = Int(block_idx.x)
    var C = Int(c_n)
    if c >= C:
        return
    var Hi = Int(hi_n)
    var Wi = Int(wi_n)
    var Wo = Int(wo_n)
    var oy = _d(n, Wo)
    var ox = n - oy * Wo
    var sy = max((Scalar[DT](oy) + Scalar[DT](0.5)) * (Scalar[DT](Hi) / Scalar[DT](Int(ho_n))) - Scalar[DT](0.5), Scalar[DT](0))
    var sx = max((Scalar[DT](ox) + Scalar[DT](0.5)) * (Scalar[DT](Wi) / Scalar[DT](Wo)) - Scalar[DT](0.5), Scalar[DT](0))
    var y0 = Int(sy)
    var x0 = Int(sx)
    var y1 = min(y0 + 1, Hi - 1)
    var x1 = min(x0 + 1, Wi - 1)
    var ly = sy - Scalar[DT](y0)
    var lx = sx - Scalar[DT](x0)
    var top = src[unsafe_offset=(y0 * Wi + x0) * C + c] * (Scalar[DT](1) - lx) + src[unsafe_offset=(y0 * Wi + x1) * C + c] * lx
    var bot = src[unsafe_offset=(y1 * Wi + x0) * C + c] * (Scalar[DT](1) - lx) + src[unsafe_offset=(y1 * Wi + x1) * C + c] * lx
    dst[unsafe_offset=n * C + c] = top * (Scalar[DT](1) - ly) + bot * ly


def conv_taps(ctx: DeviceContext, w: DeviceBuffer[DT], cout: Int, cin: Int) raises -> DeviceBuffer[DT]:
    """A [cout, cin, 3, 3] conv weight as [9, cout, cin], one GEMM operand per tap."""
    var t = ctx.enqueue_create_buffer[DT](9 * cout * cin)
    ctx.enqueue_function[tap_weights_kernel](
        w, t, Int64(cout), Int64(cin), Int64(0), grid_dim=(blocks(cin), cout, 9), block_dim=TPB,
    )
    return t^


def conv3x3_tm(
    ctx: DeviceContext, x: DeviceBuffer[DT], h: Int, w: Int, cin: Int, cout: Int, taps: DeviceBuffer[DT],
) raises -> DeviceBuffer[DT]:
    """A 3x3 conv, padding 1, no bias, on token-major `x [h w, cin]`: per tap a
    shifted copy through one GEMM, summed."""
    var n = h * w
    var out = ctx.enqueue_create_buffer[DT](n * cout)
    var xs = ctx.enqueue_create_buffer[DT](n * cin)
    var part = ctx.enqueue_create_buffer[DT](n * cout)
    for t in range(9):
        ctx.enqueue_function[shift_kernel](
            x, xs, Int64(h), Int64(w), Int64(cin), Int64(t // 3 - 1), Int64(t % 3 - 1),
            grid_dim=(n, blocks(cin)), block_dim=TPB,
        )
        if t == 0:
            gemm_tb(ctx, out, xs, taps.create_sub_buffer[DT](0, cout * cin), n, cout, cin)
        else:
            gemm_tb(ctx, part, xs, taps.create_sub_buffer[DT](t * cout * cin, cout * cin), n, cout, cin)
            ctx.enqueue_function[add_kernel](out, part, Int64(n * cout), grid_dim=blocks(n * cout), block_dim=TPB)
    return out^
