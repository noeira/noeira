# +--------------------------------------------------------------------------+ #
# | SAM 3 — the vision neck (a 4-level FPN off the backbone's 72x72 grid)
# +--------------------------------------------------------------------------+ #
"""`Sam3Neck`: `Sam3VisionNeck` from `detector_model.vision_encoder.neck.*`.

    var neck = Sam3Neck(ctx, sam3_checkpoint())
    var levels = neck.forward(backbone_out)      # 4 x [h w, 256], token-major (HWC)

From the backbone's [72 x 72, 1024], one level per scale factor:

    4.0   ConvT(1024->512, 2, s2), GELU, ConvT(512->256, 2, s2)   -> 288 x 288
    2.0   ConvT(1024->512, 2, s2)                                  -> 144 x 144
    1.0   (as is)                                                  ->  72 x 72
    0.5   MaxPool(2, s2)                                           ->  36 x 36

then `proj1` (1x1 conv to 256) and `proj2` (3x3 conv, padding 1), both with
biases. Each level's sine position encoding (`Sam3SinePositionEmbedding`,
normalised, 128 features per axis, temperature 1e4) depends on its shape alone:
`position_encoding(h, w)`.

Layout: everything here is TOKEN-MAJOR, `[row w + col, channel]` (torch's
CHW transposed), so a 1x1 conv is a linear and a transposed conv a GEMM plus a
scatter. A 3x3 conv is nine GEMMs on shifted copies (an im2col of the 288x288
level would be 764 MB).

`debug` (falsification only): bit 1, the 3x3 taps flipped (a convolution, not
torch's correlation); bit 2, the transposed conv's dy and dx swapped.
"""

from std.math import cos, sin, pow
from max.gpu import thread_idx, block_idx
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major
from linalg.matmul import matmul as max_matmul
from std.sys import default_accelerator

from noeira.nn.constants import DT
from noeira.nn.core.cublas_gemm import cublas_gemm
from .ops import Sam3Weights, TPB, Ptr, blocks, linear, gemm_tb, add_kernel, gelu_kernel, SAM3_TF32


comptime NECK_PREFIX = "detector_model.vision_encoder.neck."
comptime FPN: Int = 256
comptime IN_DIM: Int = 1024
comptime GRID: Int = 72


def gemm_nn(ctx: DeviceContext, dst: DeviceBuffer[DT], a: DeviceBuffer[DT], b: DeviceBuffer[DT], m: Int, n: Int, k: Int) raises:
    """`dst[m, n] = a[m, k] @ b[k, n]`."""
    comptime if default_accelerator().is_nvidia_gpu():
        cublas_gemm[False, False, SAM3_TF32](ctx, dst, a, b, m, n, k, 0.0)
    else:
        var out_buf = dst
        var av = TileTensor(a, row_major(m, k))
        var ov = TileTensor(out_buf, row_major(m, n))
        var bv = TileTensor(b, row_major(k, n))
        max_matmul[transpose_b=False, target="gpu"](ov, av, bv, ctx)


def convt_scatter_kernel(y: Ptr, b: Ptr, dst: Ptr, w_n: Int64, cout_n: Int64, swap: Int64):
    """`dst[(2r + dy) 2W + 2c + dx, co] = y[r W + c, co 4 + dy 2 + dx] + b[co]`:
    a 2x2 stride-2 transposed conv's GEMM output onto its doubled grid. Grid
    `(ceil(4 cout / TPB), H W)`."""
    var e = Int(block_idx.x) * TPB + Int(thread_idx.x)
    var n = Int(block_idx.y)
    var cout = Int(cout_n)
    if e >= 4 * cout:
        return
    var co = e // 4
    var q = e - co * 4
    var dy = q // 2
    var dx = q - dy * 2
    if swap != 0:
        var t = dy
        dy = dx
        dx = t
    var w = Int(w_n)
    var r = n // w
    var c = n - r * w
    dst[unsafe_offset=((2 * r + dy) * 2 * w + 2 * c + dx) * cout + co] = y[unsafe_offset=n * 4 * cout + e] + b[unsafe_offset=co]


def shift_kernel(x: Ptr, dst: Ptr, h_n: Int64, w_n: Int64, c_n: Int64, dy_n: Int64, dx_n: Int64):
    """`dst[r W + c, :] = x[(r + dy) W + c + dx, :]`, zero outside the grid (a 3x3
    conv's tap). Grid `(ceil(C / TPB), H W)`."""
    var ch = Int(block_idx.x) * TPB + Int(thread_idx.x)
    var n = Int(block_idx.y)
    var cc = Int(c_n)
    if ch >= cc:
        return
    var w = Int(w_n)
    var r = n // w + Int(dy_n)
    var c = n % w + Int(dx_n)
    var v = Scalar[DT](0)
    if r >= 0 and r < Int(h_n) and c >= 0 and c < w:
        v = x[unsafe_offset=(r * w + c) * cc + ch]
    dst[unsafe_offset=n * cc + ch] = v


def tap_weights_kernel(w: Ptr, dst: Ptr, co_n: Int64, ci_n: Int64):
    """`dst[t, co, ci] = w[co, ci, ky, kx]`, t = 3 ky + kx. Grid `(ceil(ci / TPB), co, 9)`."""
    var ci = Int(block_idx.x) * TPB + Int(thread_idx.x)
    var co = Int(block_idx.y)
    var t = Int(block_idx.z)
    var cin = Int(ci_n)
    if ci < cin:
        dst[unsafe_offset=(t * Int(co_n) + co) * cin + ci] = w[unsafe_offset=(co * cin + ci) * 9 + t]


def fill_rows_kernel(dst: Ptr, b: Ptr, cols_n: Int64, n_n: Int64):
    """`dst[r, c] = b[c]`."""
    var i = Int(block_idx.x) * TPB + Int(thread_idx.x)
    if i < Int(n_n):
        dst[unsafe_offset=i] = b[unsafe_offset=i % Int(cols_n)]


def maxpool2_kernel(x: Ptr, dst: Ptr, w_n: Int64, c_n: Int64):
    """2x2 stride-2 max pool, token-major. Grid `(ceil(C / TPB), H/2 W/2)`."""
    var ch = Int(block_idx.x) * TPB + Int(thread_idx.x)
    var n = Int(block_idx.y)
    var cc = Int(c_n)
    if ch >= cc:
        return
    var w = Int(w_n)
    var wo = w // 2
    var r = (n // wo) * 2
    var c = (n % wo) * 2
    var a = max(x[unsafe_offset=(r * w + c) * cc + ch], x[unsafe_offset=(r * w + c + 1) * cc + ch])
    var b = max(x[unsafe_offset=((r + 1) * w + c) * cc + ch], x[unsafe_offset=((r + 1) * w + c + 1) * cc + ch])
    dst[unsafe_offset=n * cc + ch] = max(a, b)


@fieldwise_init
struct FpnLevel(Copyable, Movable):
    var x: DeviceBuffer[DT]
    """[h w, 256], token-major."""
    var h: Int
    var w: Int


struct Sam3Neck(Movable):
    var wts: Sam3Weights
    var taps: List[DeviceBuffer[DT]]
    """Each level's 3x3 weights rearranged per tap, [9, 256, 256]."""
    var debug: Int

    def __init__(out self, var ctx: DeviceContext, path: String) raises:
        self.wts = Sam3Weights(ctx^, path, NECK_PREFIX)
        self.debug = 0
        self.taps = List[DeviceBuffer[DT]]()
        for i in range(4):
            var w = self.wts.w(String("fpn_layers.") + String(i) + ".proj2.weight")
            var t = self.wts.ctx.enqueue_create_buffer[DT](9 * FPN * FPN)
            self.wts.ctx.enqueue_function[tap_weights_kernel](
                w, t, Int64(FPN), Int64(FPN), grid_dim=(blocks(FPN), FPN, 9), block_dim=TPB,
            )
            self.taps.append(t^)

    def new(self, n: Int) raises -> DeviceBuffer[DT]:
        return self.wts.ctx.enqueue_create_buffer[DT](n)

    def _convt(mut self, name: String, x: DeviceBuffer[DT], h: Int, w: Int, cin: Int) raises -> DeviceBuffer[DT]:
        var wt = self.wts.w(name + ".weight")
        var cout = len(wt) // (cin * 4)
        var y = self.new(h * w * cout * 4)
        gemm_nn(self.wts.ctx, y, x, wt, h * w, cout * 4, cin)
        var out = self.new(4 * h * w * cout)
        self.wts.ctx.enqueue_function[convt_scatter_kernel](
            y, self.wts.w(name + ".bias"), out, Int64(w), Int64(cout), Int64(self.debug & 2),
            grid_dim=(blocks(4 * cout), h * w), block_dim=TPB,
        )
        return out^

    def _conv3(mut self, level: Int, x: DeviceBuffer[DT], h: Int, w: Int) raises -> DeviceBuffer[DT]:
        ref ctx = self.wts.ctx
        var n = h * w
        var out = self.new(n * FPN)
        ctx.enqueue_function[fill_rows_kernel](
            out, self.wts.w(String("fpn_layers.") + String(level) + ".proj2.bias"), Int64(FPN), Int64(n * FPN),
            grid_dim=blocks(n * FPN), block_dim=TPB,
        )
        var xs = self.new(n * FPN)
        var part = self.new(n * FPN)
        for t in range(9):
            ctx.enqueue_function[shift_kernel](
                x, xs, Int64(h), Int64(w), Int64(FPN), Int64(t // 3 - 1), Int64(t % 3 - 1),
                grid_dim=(blocks(FPN), n), block_dim=TPB,
            )
            var tw = 8 - t if (self.debug & 1) != 0 else t
            gemm_tb(ctx, part, xs, self.taps[level].create_sub_buffer[DT](tw * FPN * FPN, FPN * FPN), n, FPN, FPN)
            ctx.enqueue_function[add_kernel](out, part, Int64(n * FPN), grid_dim=blocks(n * FPN), block_dim=TPB)
        return out^

    def forward(mut self, backbone_out: DeviceBuffer[DT]) raises -> List[FpnLevel]:
        """[72 x 72, 1024] -> the four levels (scale 4, 2, 1, 0.5)."""
        var levels = List[FpnLevel]()
        for i in range(4):
            var p = String("fpn_layers.") + String(i) + "."
            var x: DeviceBuffer[DT]
            var h = GRID
            var w = GRID
            var c = IN_DIM
            if i == 0:
                var a = self._convt(p + "scale_layers.0", backbone_out, h, w, c)
                h *= 2
                w *= 2
                c = IN_DIM // 2
                self.wts.ctx.enqueue_function[gelu_kernel](a, Int64(h * w * c), grid_dim=blocks(h * w * c), block_dim=TPB)
                x = self._convt(p + "scale_layers.2", a, h, w, c)
                h *= 2
                w *= 2
                c = IN_DIM // 4
            elif i == 1:
                x = self._convt(p + "scale_layers.0", backbone_out, h, w, c)
                h *= 2
                w *= 2
                c = IN_DIM // 2
            elif i == 2:
                x = backbone_out
            else:
                x = self.new(GRID * GRID // 4 * IN_DIM)
                self.wts.ctx.enqueue_function[maxpool2_kernel](
                    backbone_out, x, Int64(GRID), Int64(IN_DIM), grid_dim=(blocks(IN_DIM), GRID * GRID // 4), block_dim=TPB,
                )
                h //= 2
                w //= 2
            var p1 = linear(self.wts, p + "proj1", x, h * w, c)
            var p2 = self._conv3(i, p1, h, w)
            levels.append(FpnLevel(p2^, h, w))
        return levels^


def position_encoding(h: Int, w: Int) -> List[Float32]:
    """`Sam3SinePositionEmbedding(128, normalize=True)` for an h x w level, TOKEN-
    major [h w, 256]: channels 0..127 from the row, 128..255 from the column;
    within each, even = sin, odd = cos of `pos / 1e4^(2 (j // 2) / 128)`, pos =
    `i / (n + 1e-6) 2 pi` for the 1-based index i (fp32, as torch computes it)."""
    comptime NPF = FPN // 2
    var dim_t = List[Float32](length=NPF, fill=0.0)
    for j in range(NPF):
        dim_t[j] = Float32(pow(10000.0, Float64(Float32(2 * (j // 2)) / Float32(NPF))))
    var two_pi = Float32(6.283185307179586)
    var out = List[Float32](length=h * w * FPN, fill=0.0)
    for r in range(h):
        var ye = Float32(r + 1) / (Float32(h) + Float32(1e-6)) * two_pi
        for c in range(w):
            var xe = Float32(c + 1) / (Float32(w) + Float32(1e-6)) * two_pi
            var base = (r * w + c) * FPN
            for j in range(NPF):
                var py = ye / dim_t[j]
                var px = xe / dim_t[j]
                out[base + j] = Float32(sin(Float64(py))) if j % 2 == 0 else Float32(cos(Float64(py)))
                out[base + NPF + j] = Float32(sin(Float64(px))) if j % 2 == 0 else Float32(cos(Float64(px)))
    return out^
