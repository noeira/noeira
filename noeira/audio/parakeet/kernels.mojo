# +--------------------------------------------------------------------------+ #
# | Parakeet-TDT — the encoder's device kernels
# +--------------------------------------------------------------------------+ #
"""Everything in the encoder that is not a `Linear` or a `LayerNorm`, as device
kernels: the subsampling convolutions, the relative-position attention, the
conformer convolution, and the elementwise glue.

Each kernel is the transcription of the CPU loop of the same name in
`encoder.mojo`, one thread per OUTPUT element, with the compiled maximum in its
type and the valid length as a runtime argument. They are written to be read
against those loops, not tuned: the encoder's time is in its GEMMs.

Conventions shared with the CPU path:

- buffers are channel-last (`[time, freq, channel]`, `[time, channel]`);
- a kernel that takes the valid length writes ZERO beyond it (that IS the
  padding mask), except the attention scores, whose padded entries are simply
  never read. ⚠ It is an `Int64` argument: a plain `Int` is not device-passable.
"""

from std.math import exp, sqrt
from max.gpu import global_idx

from noeira.nn.constants import DT
from noeira.core.tiles import Tile


def conv3x3_s2_first_kernel[
    H_IN: Int, W_IN: Int, H_OUT: Int, W_OUT: Int, C: Int
](
    src: Tile[DT, H_IN, W_IN],
    w: Tile[DT, C, 9],
    b: Tile[DT, C],
    dst: Tile[DT, H_OUT, W_OUT, C],
    valid_n: Int64,
):
    """One input channel -> C: 3x3, stride 2, padding 1, then ReLU."""
    var idx = Int(global_idx.x)
    var valid = Int(valid_n)
    if idx >= H_OUT * W_OUT * C:
        return
    var c = idx % C
    var x = (idx // C) % W_OUT
    var t = idx // (C * W_OUT)
    if t >= valid:
        dst[t, x, c] = Scalar[DT](0)
        return
    var acc = rebind[Scalar[DT]](b[c])
    for kh in range(3):
        var sy = 2 * t - 1 + kh
        if sy < 0 or sy >= H_IN:
            continue
        for kw in range(3):
            var sx = 2 * x - 1 + kw
            if sx < 0 or sx >= W_IN:
                continue
            acc += rebind[Scalar[DT]](src[sy, sx]) * rebind[Scalar[DT]](
                w[c, kh * 3 + kw]
            )
    dst[t, x, c] = acc if acc > Scalar[DT](0) else Scalar[DT](0)


def dw3x3_s2_kernel[
    H_IN: Int, W_IN: Int, H_OUT: Int, W_OUT: Int, C: Int
](
    src: Tile[DT, H_IN, W_IN, C],
    w: Tile[DT, C, 9],
    b: Tile[DT, C],
    dst: Tile[DT, H_OUT, W_OUT, C],
    valid_n: Int64,
):
    """Depthwise 3x3, stride 2, padding 1, channel-last."""
    var idx = Int(global_idx.x)
    var valid = Int(valid_n)
    if idx >= H_OUT * W_OUT * C:
        return
    var c = idx % C
    var x = (idx // C) % W_OUT
    var t = idx // (C * W_OUT)
    if t >= valid:
        dst[t, x, c] = Scalar[DT](0)
        return
    var acc = rebind[Scalar[DT]](b[c])
    for kh in range(3):
        var sy = 2 * t - 1 + kh
        if sy < 0 or sy >= H_IN:
            continue
        for kw in range(3):
            var sx = 2 * x - 1 + kw
            if sx < 0 or sx >= W_IN:
                continue
            acc += rebind[Scalar[DT]](src[sy, sx, c]) * rebind[Scalar[DT]](
                w[c, kh * 3 + kw]
            )
    dst[t, x, c] = acc


def relu_rows_kernel[ROWS: Int, ROW: Int](t: Tile[DT, ROWS, ROW], valid_n: Int64):
    """ReLU in place on the first `valid` rows, zero on the others."""
    var idx = Int(global_idx.x)
    var valid = Int(valid_n)
    if idx >= ROWS * ROW:
        return
    var r = idx // ROW
    var k = idx % ROW
    if r >= valid:
        t[r, k] = Scalar[DT](0)
        return
    var v = rebind[Scalar[DT]](t[r, k])
    if v < Scalar[DT](0):
        t[r, k] = Scalar[DT](0)


def flatten_tfc_kernel[T: Int, F: Int, C: Int](
    src: Tile[DT, T, F, C], dst: Tile[DT, T, C * F]
):
    """`[T, F, C]` -> `[T, C * F]`, channel-major as the reference flattens."""
    var idx = Int(global_idx.x)
    if idx >= T * F * C:
        return
    var c = idx % C
    var f = (idx // C) % F
    var t = idx // (C * F)
    dst[t, c * F + f] = rebind[Scalar[DT]](src[t, f, c])


def silu_kernel[N: Int](t: Tile[DT, N]):
    var idx = Int(global_idx.x)
    if idx >= N:
        return
    var v = rebind[Scalar[DT]](t[idx])
    t[idx] = v / (Scalar[DT](1) + exp(-v))


def add_scaled_kernel[N: Int](x: Tile[DT, N], y: Tile[DT, N], scale: Scalar[DT]):
    """`x += scale * y`."""
    var idx = Int(global_idx.x)
    if idx >= N:
        return
    x[idx] = rebind[Scalar[DT]](x[idx]) + scale * rebind[Scalar[DT]](y[idx])


def rel_scores_kernel[T: Int, P: Int, D: Int, H: Int, L: Int](
    q: Tile[DT, T, D],
    k: Tile[DT, T, D],
    relk: Tile[DT, L, P, D],
    bias_u: Tile[DT, D],
    bias_v: Tile[DT, D],
    scores: Tile[DT, H, T, T],
    valid_n: Int64,
    layer_n: Int64,
):
    """`scores[h, i, j] = ((q_i + u) . k_j + (q_i + v) . r_{i-j}) / sqrt(D / H)`
    for valid `i`, `j`; offset `i - j` is row `j - i + T - 1` of this layer's
    slice of `relk` (the projected positional table of all `L` layers)."""
    comptime HD = D // H
    var idx = Int(global_idx.x)
    var valid = Int(valid_n)
    if idx >= H * T * T:
        return
    var j = idx % T
    var i = (idx // T) % T
    var h = idx // (T * T)
    if i >= valid or j >= valid:
        return
    var ho = h * HD
    var rb = j - i + T - 1
    var layer = Int(layer_n)
    var ac = Scalar[DT](0)
    var bd = Scalar[DT](0)
    for d in range(HD):
        var qd = rebind[Scalar[DT]](q[i, ho + d])
        ac += (qd + rebind[Scalar[DT]](bias_u[ho + d])) * rebind[Scalar[DT]](
            k[j, ho + d]
        )
        bd += (qd + rebind[Scalar[DT]](bias_v[ho + d])) * rebind[Scalar[DT]](
            relk[layer, rb, ho + d]
        )
    var scale = Scalar[DT](1) / sqrt(Scalar[DT](HD))
    scores[h, i, j] = ac * scale + bd * scale


def rel_apply_kernel[T: Int, D: Int, H: Int](
    scores: Tile[DT, H, T, T],
    v: Tile[DT, T, D],
    dst: Tile[DT, T, D],
    valid_n: Int64,
):
    """`dst[i, h, :] = softmax_j(scores[h, i, :valid]) @ v[:valid, h, :]`."""
    comptime HD = D // H
    var idx = Int(global_idx.x)
    var valid = Int(valid_n)
    if idx >= T * D:
        return
    var col = idx % D
    var i = idx // D
    if i >= valid:
        dst[i, col] = Scalar[DT](0)
        return
    var h = col // HD
    var top = Scalar[DT](-3.0e38)
    for j in range(valid):
        var s = rebind[Scalar[DT]](scores[h, i, j])
        if s > top:
            top = s
    var total = Scalar[DT](0)
    var acc = Scalar[DT](0)
    for j in range(valid):
        var e = exp(rebind[Scalar[DT]](scores[h, i, j]) - top)
        total += e
        acc += e * rebind[Scalar[DT]](v[j, col])
    dst[i, col] = acc / total


def glu_rows_kernel[T: Int, D: Int](
    src: Tile[DT, T, 2 * D], dst: Tile[DT, T, D], valid_n: Int64
):
    """`dst[t, c] = src[t, c] * sigmoid(src[t, D + c])`, zero beyond `valid`."""
    var idx = Int(global_idx.x)
    var valid = Int(valid_n)
    if idx >= T * D:
        return
    var c = idx % D
    var t = idx // D
    if t >= valid:
        dst[t, c] = Scalar[DT](0)
        return
    var g = rebind[Scalar[DT]](src[t, D + c])
    dst[t, c] = rebind[Scalar[DT]](src[t, c]) / (Scalar[DT](1) + exp(-g))


def dw1d_bn_silu_kernel[T: Int, D: Int, K: Int, EPS: Scalar[DT]](
    src: Tile[DT, T, D],
    w: Tile[DT, D, K],
    gamma: Tile[DT, D],
    beta: Tile[DT, D],
    mean: Tile[DT, D],
    variance: Tile[DT, D],
    dst: Tile[DT, T, D],
    valid_n: Int64,
):
    """Depthwise Conv1d over time (kernel K, padding K // 2), BatchNorm with the
    running statistics, SiLU. Zero beyond `valid`."""
    var idx = Int(global_idx.x)
    var valid = Int(valid_n)
    if idx >= T * D:
        return
    var c = idx % D
    var t = idx // D
    if t >= valid:
        dst[t, c] = Scalar[DT](0)
        return
    var acc = Scalar[DT](0)
    for j in range(K):
        var tt = t - K // 2 + j
        if tt < 0 or tt >= valid:
            continue
        acc += rebind[Scalar[DT]](src[tt, c]) * rebind[Scalar[DT]](w[c, j])
    var n = (acc - rebind[Scalar[DT]](mean[c])) / sqrt(
        rebind[Scalar[DT]](variance[c]) + EPS
    ) * rebind[Scalar[DT]](gamma[c]) + rebind[Scalar[DT]](beta[c])
    dst[t, c] = n / (Scalar[DT](1) + exp(-n))
