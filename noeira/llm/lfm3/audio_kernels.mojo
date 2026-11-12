# +--------------------------------------------------------------------------+ #
# | LFM3-Omni — the device kernels of the audio downsampler
# +--------------------------------------------------------------------------+ #
"""The kernels `audio.mojo` adds to the text backbone's (`kernels.mojo`):
`Lfm3OmniDownsampler`'s three stride-2 3x3 convolutions over the (mel, time)
plane and the flatten before its last linear.

LAYOUT. The reference holds `[C, mel, time]` (NCHW, the spectrogram's mel bins
as rows); here every activation is POSITION-major, `[mel, time, C]`, so the 1x1
convolutions are plain linears (`[positions, C_in] @ W^T`, the GEMM every
other layer uses). The flatten is where the two orders meet: the reference's
`flatten(1, 2)` makes feature `c * 16 + m` of time step `t`
(`audio_flatten_kernel`).

Every convolution is kernel 3, stride 2, padding 1: output row `y` reads input
rows `2 y - 1 .. 2 y + 1` (zeros outside), so `n` rows become `ceil(n / 2)`.
Grids are 3-D, `(ceil(C / TPB), W_out, H_out)`: no runtime division.
"""

from max.gpu import global_idx, thread_idx, block_idx

from noeira.nn.constants import DT
from .kernels import TPB, Ptr, HPtr, _bf, _div, _sigmoid


def audio_conv_in_kernel(mel: Ptr, w: HPtr, b: HPtr, dst: Ptr, h_n: Int64, w_n: Int64, c_n: Int64):
    """`conv_in` + SiLU: one input channel (the spectrogram `[H mel, W time]`,
    mel major) to `C` channels, `dst[(y W2 + x) C + c] = silu(b[c] + sum
    w[c, ky, kx] mel[2y-1+ky, 2x-1+kx])`. `w` is `[C, 1, 3, 3]`."""
    var c = Int(block_idx.x) * TPB + Int(thread_idx.x)
    var x = Int(block_idx.y)
    var y = Int(block_idx.z)
    var ch = Int(c_n)
    if c >= ch:
        return
    var hh = Int(h_n)
    var ww = Int(w_n)
    var w2 = (ww + 1) // 2
    var acc = _bf(b[unsafe_offset=c])
    for ky in range(3):
        var r = 2 * y - 1 + ky
        if r < 0 or r >= hh:
            continue
        for kx in range(3):
            var s = 2 * x - 1 + kx
            if s < 0 or s >= ww:
                continue
            acc += _bf(w[unsafe_offset=c * 9 + ky * 3 + kx]) * mel[unsafe_offset=r * ww + s]
    dst[unsafe_offset=(y * w2 + x) * ch + c] = acc * _sigmoid(acc)


def audio_dw_conv_kernel(src: Ptr, w: HPtr, b: HPtr, dst: Ptr, h_n: Int64, w_n: Int64, c_n: Int64):
    """A depthwise 3x3 stride-2 convolution (+ bias, no activation) on
    `[H, W, C]` position-major activations: `dst[(y W2 + x) C + c] = b[c] +
    sum w[c, ky, kx] src[((2y-1+ky) W + 2x-1+kx) C + c]`."""
    var c = Int(block_idx.x) * TPB + Int(thread_idx.x)
    var x = Int(block_idx.y)
    var y = Int(block_idx.z)
    var ch = Int(c_n)
    if c >= ch:
        return
    var hh = Int(h_n)
    var ww = Int(w_n)
    var w2 = (ww + 1) // 2
    var acc = _bf(b[unsafe_offset=c])
    for ky in range(3):
        var r = 2 * y - 1 + ky
        if r < 0 or r >= hh:
            continue
        for kx in range(3):
            var s = 2 * x - 1 + kx
            if s < 0 or s >= ww:
                continue
            acc += _bf(w[unsafe_offset=c * 9 + ky * 3 + kx]) * src[unsafe_offset=(r * ww + s) * ch + c]
    dst[unsafe_offset=(y * w2 + x) * ch + c] = acc


def bias_silu_kernel(x: Ptr, b: HPtr, cols_n: Int64, n_n: Int64):
    """`x[r, c] = silu(x[r, c] + b[c])` over `n` elements of `x` [rows, cols]."""
    var i = Int(global_idx.x)
    if i < Int(n_n):
        var cols = Int(cols_n)
        var v = x[unsafe_offset=i] + _bf(b[unsafe_offset=i - _div(i, cols) * cols])
        x[unsafe_offset=i] = v * _sigmoid(v)


def audio_flatten_kernel(src: Ptr, dst: Ptr, h_n: Int64, w_n: Int64, c_n: Int64, mel_major: Int64):
    """`[H mel, W time, C]` -> `[W, C H]`, feature `c H + m` (the reference's
    `flatten(1, 2)` of `[C, H, W]`). `mel_major` (falsification): feature `m C + c`.
    Grid `(ceil(C / TPB), W, H)`."""
    var c = Int(block_idx.x) * TPB + Int(thread_idx.x)
    var t = Int(block_idx.y)
    var m = Int(block_idx.z)
    var ch = Int(c_n)
    if c >= ch:
        return
    var hh = Int(h_n)
    var ww = Int(w_n)
    var f = m * ch + c if mel_major != 0 else c * hh + m
    dst[unsafe_offset=t * ch * hh + f] = src[unsafe_offset=(m * ww + t) * ch + c]
