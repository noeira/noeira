# +--------------------------------------------------------------------------+ #
# | Kokoro-82M on the GPU
# +--------------------------------------------------------------------------+ #
"""The CPU forward of `prosody.mojo` / `decoder.mojo`, op for op, on a device.

    var k = KokoroDevice.load(DeviceContext())                 # ~/.cache/noeira/kokoro
    var s = k.speak(es, "Bonjour.", "ff_siwis")                 # or k.synthesize(phonemes, voice)

Same weights (`KokoroWeights`, torch layouts, weight norm folded), same
channels-last sequences (`[length, channels]`), same operations in the same
order — so each stage can be gated against the same reference dumps
(`tests/audio/test_kokoro_gpu.mojo`).

## Shapes are RUNTIME values

Every length (tokens T, frames F, samples) is a run-time argument: the GEMMs go
through cuBLAS (NVIDIA) or MAX's matmul with runtime extents (elsewhere) at the
sentence's real size, and every kernel takes its lengths as `Int64`. Nothing is
padded, so nothing needs a mask: the instance norms see exactly the valid frames.
Buffers are allocated per stage from MAX's caching allocator.

## What stays on the host

The frames per phoneme (read back: they decide every later shape, and the
rounding must be the reference's), the harmonic source and its STFT (`decoder.mojo`;
a few ms), and the inverse STFT. Everything with weights runs on the device.

## Convolution weights are stored tap-major

A `Conv1d` weight is `[COUT, CIN, K]` in torch; on the device it is reordered
at load to `[COUT, K, CIN]`, and `im2col` writes each window tap-major
(`cols[r, j CIN + c]`). Then neighbouring threads read neighbouring channels of
the channels-last input. With torch's order they read input rows `CIN` floats
apart, and `im2col` was 46 % of all GPU time on the Orin (profiled 2026-11-10).
Transposed convolutions and the depthwise `pool` keep torch's layout.

## Index arithmetic is 32-bit, or absent

A GPU has no 64-bit integer divide: Mojo's `Int` `//` and `%` in a kernel
compile to a software routine. With them, `im2col` cost ~0.54 ms a call
whatever its memory order (profiled). So `im2col` runs on a 3-D grid (channel
block, tap, row: no division at all), and the elementwise kernels split their
flat index with 32-bit unsigned arithmetic (`_u`).

## Precision

fp32 throughout by default. `-D KOKORO_TF32=1` lets cuBLAS run the vocoder's large
GEMMs in TF32 (NVIDIA): faster, gated against its own bound.
"""

from std.math import exp, sqrt, sin
from std.memory import stack_allocation
from std.sys import default_accelerator
from std.sys.defines import is_defined
from max.gpu import global_idx, thread_idx, block_idx
from max.gpu.host import DeviceContext, DeviceBuffer
from max.gpu.memory import AddressSpace
from max.gpu.primitives import block, warp
from max.gpu.sync import barrier
from layout import TileTensor, row_major
from linalg.matmul import matmul as max_matmul

from noeira.nn.constants import DT
from noeira.nn.core.cublas_gemm import cublas_gemm
from noeira.nn.core.fill import fill_dev
from noeira.nn.core.tensor import Tensor
from noeira.audio.espeak import EspeakNG
from .decoder import (
    kk_harmonic_source, kk_stft, kk_istft, KokoroNoise, KK_SAMPLES_PER_FRAME,
)
from .g2p import kokoro_language, kokoro_default_g2p, kokoro_phonemes
from .paths import kokoro_dir
from .prosody import KK_CONTEXT, KK_MAX_DUR
from .tts import KokoroTTS, KokoroSpeech, kokoro_noise, kokoro_vocab
from .weights import KokoroWeights


comptime KD_TF32: Bool = is_defined["KOKORO_TF32"]()
comptime TPB: Int = 256
comptime Ptr = Pointer[Scalar[DT], MutAnyOrigin]
comptime IPtr = Pointer[Scalar[DType.int32], MutAnyOrigin]
comptime ALIGN: Int = 32
comptime IM2COL_BUDGET: Int = 1 << 22
"""Floats per im2col chunk (16 MB): the vocoder's last stage at 10 s of speech
would need 270 MB in one piece."""


def _blocks(n: Int) -> Int:
    return (n + TPB - 1) // TPB


@always_inline
def _u(x: Int64) -> UInt32:
    return UInt32(x)


@always_inline
def _gid() -> UInt32:
    return UInt32(global_idx.x)


def _tanh(x: Scalar[DT]) -> Scalar[DT]:
    """Through `exp`: a libm `tanh` is not guaranteed on a device."""
    return Scalar[DT](1) - Scalar[DT](2) / (exp(Scalar[DT](2) * x) + Scalar[DT](1))


def _sigmoid(x: Scalar[DT]) -> Scalar[DT]:
    return Scalar[DT](1) / (Scalar[DT](1) + exp(-x))


# ═══════════════════════════════════════════════════════════════════════════
# Kernels (one thread per output element unless said otherwise)
# ═══════════════════════════════════════════════════════════════════════════


def _bias_rows_kernel(y: Ptr, b: Ptr, rows_n: Int64, cols_n: Int64):
    var cols = _u(cols_n)
    var i = _gid()
    if i < _u(rows_n) * cols:
        y[unsafe_offset=Int(i)] += b[unsafe_offset=Int(i % cols)]


def _im2col_kernel(
    x: Ptr, cols: Ptr, l_in_n: Int64, c_in_n: Int64, k_n: Int64,
    row0_n: Int64, rows_n: Int64, stride_n: Int64, pad_n: Int64, dil_n: Int64,
):
    """`cols[r, j CIN + c] = x[(row0 + r) stride - pad + j dil, c]`, 0 outside:
    tap-major, as the device's conv weights are (module docstring). Grid
    `(ceil(CIN / TPB), K, rows)`: channel, tap and row come from the block
    and thread indices — no division."""
    var c_in = Int(c_in_n)
    var c = Int(block_idx.x) * TPB + Int(thread_idx.x)
    if c >= c_in:
        return
    var j = Int(block_idx.y)
    var r = Int(block_idx.z)
    var dst = r * c_in * Int(k_n) + j * c_in + c
    var t = (Int(row0_n) + r) * Int(stride_n) - Int(pad_n) + j * Int(dil_n)
    if t >= 0 and t < Int(l_in_n):
        cols[unsafe_offset=dst] = x[unsafe_offset=t * c_in + c]
    else:
        cols[unsafe_offset=dst] = Scalar[DT](0)


def _conv_t_gather_kernel(
    tmp: Ptr, b: Ptr, y: Ptr, l_in_n: Int64, c_out_n: Int64, k_n: Int64,
    l_out_n: Int64, stride_n: Int64, pad_n: Int64,
):
    """ConvTranspose1d's scatter as a gather: `y[o, c] = b[c] + sum_j
    tmp[t, c K + j]` over the `t` with `t stride - pad + j = o`."""
    var c_out = _u(c_out_n)
    var k = Int(k_n)
    var stride = _u(stride_n)
    var i = _gid()
    if i >= _u(l_out_n) * c_out:
        return
    var o = i / c_out
    var c = Int(i % c_out)
    var acc = b[unsafe_offset=c]
    # The taps that land on `o`: j = (o + pad) mod stride, then every stride-th.
    var op = o + _u(pad_n)
    var j = Int(op % stride)
    var t = Int(op / stride)
    while j < k and t >= 0:
        if t < Int(l_in_n):
            acc += tmp[unsafe_offset=t * Int(c_out) * k + c * k + j]
        j += Int(stride)
        t -= 1
    y[unsafe_offset=Int(i)] = acc


def _dw_t_x2_kernel(x: Ptr, w: Ptr, b: Ptr, y: Ptr, l_in_n: Int64, c_n: Int64):
    """Depthwise ConvTranspose1d (k 3, stride 2, padding 1, output_padding 1)."""
    var cn = _u(c_n)
    var l_in = Int(l_in_n)
    var i = _gid()
    if i >= 2 * _u(l_in_n) * cn:
        return
    var o = Int(i / cn)
    var c = Int(i % cn)
    var acc = b[unsafe_offset=c]
    for k in range(3):
        var num = o + 1 - k
        if num < 0 or (num & 1) != 0:
            continue
        var t = num >> 1
        if t < l_in:
            acc += x[unsafe_offset=t * Int(cn) + c] * w[unsafe_offset=c * 3 + k]
    y[unsafe_offset=Int(i)] = acc


def _matvec_kernel(a: Ptr, b: Ptr, dst: Ptr, m_n: Int64, k_n: Int64):
    """`dst[m] = a[m, k] . b[k]`: a warp per row, 8 rows per block of 256.
    The GEMM's N = 1 case (a Conv1d with ONE output channel, Kokoro's pitch and
    energy projections), which `max_matmul` on the GPU silently miscomputes."""
    var tid = Int(thread_idx.x)
    var row = Int(block_idx.x) * 8 + (tid >> 5)
    var lane = tid & 31
    var k = Int(k_n)
    var acc = Scalar[DT](0)
    if row < Int(m_n):
        var base = row * k
        var j = lane
        while j < k:
            acc += a[unsafe_offset=base + j] * b[unsafe_offset=j]
            j += 32
    acc = warp.sum(acc)
    if lane == 0 and row < Int(m_n):
        dst[unsafe_offset=row] = acc


comptime STAT_ROWS: Int = 256
"""Rows of time per block in the instance-norm statistics."""


def _chan_partial_kernel(x: Ptr, mean: Ptr, part: Ptr, length_n: Int64, c_n: Int64, second_n: Int64):
    """Partial sums over time of `x[length, C]`, per channel and per chunk of
    `STAT_ROWS` rows: of `x` (first pass) or of `(x - mean[c])^2` (second).
    Grid `(ceil(C / 32), chunks)`, block 256 = 32 channels (coalesced) x 8 row
    lanes; `part[chunk, c]`."""
    var length = Int(length_n)
    var cn = Int(c_n)
    var tid = Int(thread_idx.x)
    var tx = tid & 31
    var ty = tid >> 5
    var ch = Int(block_idx.x) * 32 + tx
    var chunk = Int(block_idx.y)
    var sh = stack_allocation[256, Scalar[DT], address_space=AddressSpace.SHARED]()
    var acc = Scalar[DT](0)
    if ch < cn:
        var m = mean[unsafe_offset=ch] if Int(second_n) == 1 else Scalar[DT](0)
        var t = chunk * STAT_ROWS + ty
        var end = (chunk + 1) * STAT_ROWS
        if end > length:
            end = length
        while t < end:
            var v = x[unsafe_offset=t * cn + ch] - m
            acc += v * v if Int(second_n) == 1 else v
            t += 8
    sh[unsafe_offset=tid] = acc
    barrier()
    if ty == 0 and ch < cn:
        var s = Scalar[DT](0)
        for r in range(8):
            s += sh[unsafe_offset=r * 32 + tx]
        part[unsafe_offset=chunk * cn + ch] = s


def _chan_finish_kernel(part: Ptr, mean: Ptr, inv: Ptr, chunks_n: Int64, length_n: Int64, c_n: Int64, second_n: Int64):
    """Sum the chunks: the mean (first pass), or 1/sqrt(biased var + 1e-5)."""
    var cn = Int(c_n)
    var ch = Int(global_idx.x)
    if ch >= cn:
        return
    var s = Scalar[DT](0)
    for q in range(Int(chunks_n)):
        s += part[unsafe_offset=q * cn + ch]
    var v = s / Scalar[DT](Int(length_n))
    if Int(second_n) == 1:
        inv[unsafe_offset=ch] = Scalar[DT](1) / sqrt(v + Scalar[DT](1e-5))
    else:
        mean[unsafe_offset=ch] = v


def _adain_apply_kernel(x: Ptr, mean: Ptr, inv: Ptr, gb: Ptr, length_n: Int64, c_n: Int64):
    """`x = (1 + gamma[c]) (x - mean[c]) inv[c] + beta[c]`, `gb = [gamma | beta]`."""
    var cn = _u(c_n)
    var i = _gid()
    if i >= _u(length_n) * cn:
        return
    var c = Int(i % cn)
    var ii = Int(i)
    var v = (x[unsafe_offset=ii] - mean[unsafe_offset=c]) * inv[unsafe_offset=c]
    x[unsafe_offset=ii] = (Scalar[DT](1) + gb[unsafe_offset=c]) * v + gb[unsafe_offset=Int(cn) + c]


def _row_norm_kernel(x: Ptr, gamma: Ptr, beta: Ptr, dim_n: Int64, eps: Float32, affine_n: Int64):
    """LayerNorm over each row of `x[rows, dim]` in place (one block per row,
    two passes); `gamma` / `beta` applied when `affine_n` is 1."""
    var dim = Int(dim_n)
    var row = Int(block_idx.x) * dim
    var tid = Int(thread_idx.x)
    var s = Scalar[DT](0)
    var d = tid
    while d < dim:
        s += x[unsafe_offset=row + d]
        d += TPB
    var mean = block.sum[block_size=TPB, broadcast=True](val=s) / Scalar[DT](dim)
    var s2 = Scalar[DT](0)
    d = tid
    while d < dim:
        var v = x[unsafe_offset=row + d] - mean
        s2 += v * v
        d += TPB
    var var_ = block.sum[block_size=TPB, broadcast=True](val=s2) / Scalar[DT](dim)
    var inv = Scalar[DT](1) / sqrt(var_ + Scalar[DT](eps))
    d = tid
    while d < dim:
        var v = (x[unsafe_offset=row + d] - mean) * inv
        if Int(affine_n) == 1:
            v = v * gamma[unsafe_offset=d] + beta[unsafe_offset=d]
        x[unsafe_offset=row + d] = v
        d += TPB


def _modulate_kernel(src: Ptr, gb: Ptr, dst: Ptr, rows_n: Int64, dim_n: Int64, dst_cols_n: Int64):
    """`dst[r, d] = (1 + gb[d]) src[r, d] + gb[dim + d]` (AdaLayerNorm's affine),
    into the first `dim` columns of a `[rows, dst_cols]` buffer."""
    var dim = _u(dim_n)
    var i = _gid()
    if i >= _u(rows_n) * dim:
        return
    var r = Int(i / dim)
    var d = Int(i % dim)
    dst[unsafe_offset=r * Int(dst_cols_n) + d] = (
        (Scalar[DT](1) + gb[unsafe_offset=d]) * src[unsafe_offset=Int(i)] + gb[unsafe_offset=Int(dim) + d]
    )


def _leaky_kernel(x: Ptr, n_n: Int64, slope: Float32):
    var i = Int(global_idx.x)
    if i < Int(n_n):
        var v = x[unsafe_offset=i]
        if v < Scalar[DT](0):
            x[unsafe_offset=i] = v * Scalar[DT](slope)


def _snake_kernel(x: Ptr, alpha: Ptr, n_n: Int64, c_n: Int64):
    """`x + sin^2(a x) / a`, `a` per channel."""
    var i = _gid()
    if i < _u(n_n):
        var a = alpha[unsafe_offset=Int(i % _u(c_n))]
        var ii = Int(i)
        var v = x[unsafe_offset=ii]
        var s = sin(a * v)
        x[unsafe_offset=ii] = v + (Scalar[DT](1) / a) * (s * s)


def _axpy_kernel(y: Ptr, x: Ptr, n_n: Int64, a: Float32):
    """`y += a x`."""
    var i = Int(global_idx.x)
    if i < Int(n_n):
        y[unsafe_offset=i] += Scalar[DT](a) * x[unsafe_offset=i]


def _sum_scale_kernel(y: Ptr, x: Ptr, n_n: Int64, s: Float32):
    """`y = (y + x) s`."""
    var i = Int(global_idx.x)
    if i < Int(n_n):
        y[unsafe_offset=i] = (y[unsafe_offset=i] + x[unsafe_offset=i]) * Scalar[DT](s)


def _scale_kernel(y: Ptr, n_n: Int64, s: Float32):
    var i = Int(global_idx.x)
    if i < Int(n_n):
        y[unsafe_offset=i] = y[unsafe_offset=i] * Scalar[DT](s)


def _gelu_new_kernel(x: Ptr, n_n: Int64):
    var i = Int(global_idx.x)
    if i < Int(n_n):
        var v = x[unsafe_offset=i]
        x[unsafe_offset=i] = Scalar[DT](0.5) * v * (
            Scalar[DT](1) + _tanh(Scalar[DT](0.7978845608028654) * (v + Scalar[DT](0.044715) * v * v * v))
        )


def _copy_cols_kernel(src: Ptr, dst: Ptr, rows_n: Int64, src_cols_n: Int64, dst_cols_n: Int64, col0_n: Int64):
    """`dst[r, col0 + j] = src[r, j]`."""
    var sc = _u(src_cols_n)
    var i = _gid()
    if i >= _u(rows_n) * sc:
        return
    var r = Int(i / sc)
    var j = Int(i % sc)
    dst[unsafe_offset=r * Int(dst_cols_n) + Int(col0_n) + j] = src[unsafe_offset=Int(i)]


def _bcast_cols_kernel(vec: Ptr, dst: Ptr, rows_n: Int64, n_n: Int64, dst_cols_n: Int64, col0_n: Int64):
    """`dst[r, col0 + j] = vec[j]` on every row."""
    var n = _u(n_n)
    var i = _gid()
    if i >= _u(rows_n) * n:
        return
    var r = Int(i / n)
    var j = Int(i % n)
    dst[unsafe_offset=r * Int(dst_cols_n) + Int(col0_n) + j] = vec[unsafe_offset=j]


def _gather_rows_kernel(src: Ptr, idx: IPtr, dst: Ptr, rows_n: Int64, cols_n: Int64):
    """`dst[f] = src[idx[f]]`: the alignment, token rows repeated per frame."""
    var cols = _u(cols_n)
    var i = _gid()
    if i >= _u(rows_n) * cols:
        return
    var f = Int(i / cols)
    dst[unsafe_offset=Int(i)] = src[unsafe_offset=Int(idx[unsafe_offset=f]) * Int(cols) + Int(i % cols)]


def _embed_kernel(table: Ptr, ids: IPtr, dst: Ptr, rows_n: Int64, dim_n: Int64):
    var dim = _u(dim_n)
    var i = _gid()
    if i >= _u(rows_n) * dim:
        return
    dst[unsafe_offset=Int(i)] = table[unsafe_offset=Int(ids[unsafe_offset=Int(i / dim)]) * Int(dim) + Int(i % dim)]


def _albert_embed_kernel(word: Ptr, pos: Ptr, typ: Ptr, ids: IPtr, dst: Ptr, rows_n: Int64, dim_n: Int64):
    """word[id] + token type 0 + position."""
    var dim = Int(dim_n)
    var i = Int(global_idx.x)
    if i >= Int(rows_n) * dim:
        return
    var r = i // dim
    var d = i % dim
    dst[unsafe_offset=i] = (
        word[unsafe_offset=Int(ids[unsafe_offset=r]) * dim + d] + typ[unsafe_offset=d]
        + pos[unsafe_offset=r * dim + d]
    )


def _upsample_x2_kernel(x: Ptr, y: Ptr, l_n: Int64, c_n: Int64):
    """Nearest x2 along time."""
    var cn = _u(c_n)
    var i = _gid()
    if i >= 2 * _u(l_n) * cn:
        return
    y[unsafe_offset=Int(i)] = x[unsafe_offset=Int(((i / cn) >> 1) * cn + i % cn)]


def _reflect_left_kernel(src: Ptr, dst: Ptr, l_n: Int64, c_n: Int64):
    """ReflectionPad1d((1, 0)): `dst[0] = src[1]`, `dst[t + 1] = src[t]`."""
    var cn = _u(c_n)
    var i = _gid()
    if i >= (_u(l_n) + 1) * cn:
        return
    var o = i / cn
    var c = i % cn
    var t = UInt32(1) if o == 0 else o - 1
    dst[unsafe_offset=Int(i)] = src[unsafe_offset=Int(t * cn + c)]


def _attn_scores_kernel(q: Ptr, k: Ptr, sc: Ptr, t_n: Int64, heads_n: Int64, hd_n: Int64, scale: Float32):
    """`sc[h, i, j] = q_i . k_j * scale` for one head of `hd` columns."""
    var t = Int(t_n)
    var hd = Int(hd_n)
    var dim = Int(heads_n) * hd
    var idx = Int(global_idx.x)
    if idx >= Int(heads_n) * t * t:
        return
    var j = idx % t
    var i = (idx // t) % t
    var h = idx // (t * t)
    var acc = Scalar[DT](0)
    for d in range(hd):
        acc += q[unsafe_offset=i * dim + h * hd + d] * k[unsafe_offset=j * dim + h * hd + d]
    sc[unsafe_offset=idx] = acc * Scalar[DT](scale)


def _attn_apply_kernel(sc: Ptr, v: Ptr, ctx: Ptr, t_n: Int64, heads_n: Int64, hd_n: Int64):
    """`ctx[i, col] = softmax_j(sc[h, i, :]) @ v[:, col]`, `h = col / hd`."""
    var t = Int(t_n)
    var hd = Int(hd_n)
    var dim = Int(heads_n) * hd
    var idx = Int(global_idx.x)
    if idx >= t * dim:
        return
    var i = idx // dim
    var col = idx % dim
    var base = (col // hd) * t * t + i * t
    var top = sc[unsafe_offset=base]
    for j in range(1, t):
        var s = sc[unsafe_offset=base + j]
        if s > top:
            top = s
    var total = Scalar[DT](0)
    var acc = Scalar[DT](0)
    for j in range(t):
        var e = exp(sc[unsafe_offset=base + j] - top)
        total += e
        acc += e * v[unsafe_offset=j * dim + col]
    ctx[unsafe_offset=idx] = acc / total


def _lstm_step_kernel(
    gxf: Ptr, gxr: Ptr, whf: Ptr, whr: Ptr,
    bif: Ptr, bhf: Ptr, bir: Ptr, bhr: Ptr,
    y: Ptr, cst: Ptr, s_n: Int64, steps_n: Int64, h_n: Int64,
):
    """One step of a bidirectional LSTM, both directions: block `dir H + unit`,
    four warps — warp `g` computes gate `g`'s recurrent dot product
    (`W_hh[g H + unit] . h_prev`, coalesced), thread 0 updates the cell.
    `h_prev` is the previous step's row of `y` (forward: t - 1; reverse: t + 1)."""
    var h = Int(h_n)
    var steps = Int(steps_n)
    var s = Int(s_n)
    var direction = Int(block_idx.x) // h
    var unit = Int(block_idx.x) % h
    var tid = Int(thread_idx.x)
    var g = tid // 32
    var lane = tid % 32
    var t = s if direction == 0 else steps - 1 - s
    var tp = t - 1 if direction == 0 else t + 1
    var whh = whf if direction == 0 else whr
    var sh = stack_allocation[4, Scalar[DT], address_space=AddressSpace.SHARED]()
    var acc = Scalar[DT](0)
    if s > 0:
        var row = (g * h + unit) * h
        var hp = tp * 2 * h + direction * h
        var k = lane
        while k < h:
            acc += whh[unsafe_offset=row + k] * y[unsafe_offset=hp + k]
            k += 32
    acc = warp.sum(acc)
    if lane == 0:
        sh[unsafe_offset=g] = acc
    barrier()
    if tid == 0:
        var gx = gxf if direction == 0 else gxr
        var bi = bif if direction == 0 else bir
        var bh = bhf if direction == 0 else bhr
        # Four named gates, not a per-thread array indexed at run time (which
        # Metal has miscompiled here before).
        var base = t * 4 * h + unit
        var pi = gx[unsafe_offset=base] + bi[unsafe_offset=unit] + bh[unsafe_offset=unit] + sh[unsafe_offset=0]
        var pf = gx[unsafe_offset=base + h] + bi[unsafe_offset=h + unit] + bh[unsafe_offset=h + unit] + sh[unsafe_offset=1]
        var pg = gx[unsafe_offset=base + 2 * h] + bi[unsafe_offset=2 * h + unit] + bh[unsafe_offset=2 * h + unit] + sh[unsafe_offset=2]
        var po = gx[unsafe_offset=base + 3 * h] + bi[unsafe_offset=3 * h + unit] + bh[unsafe_offset=3 * h + unit] + sh[unsafe_offset=3]
        var c_prev = cst[unsafe_offset=direction * h + unit] if s > 0 else Scalar[DT](0)
        var c = _sigmoid(pf) * c_prev + _sigmoid(pi) * _tanh(pg)
        cst[unsafe_offset=direction * h + unit] = c
        y[unsafe_offset=t * 2 * h + direction * h + unit] = _sigmoid(po) * _tanh(c)


# ═══════════════════════════════════════════════════════════════════════════
# The model on a device
# ═══════════════════════════════════════════════════════════════════════════


def _is_conv1d_weight(w: KokoroWeights, i: Int) -> Bool:
    """A `Conv1d` weight (stored tap-major on the device): rank 3, and neither
    a transposed convolution (`ups`) nor the depthwise `pool`."""
    if w.rank(i) != 3 or not w.names[i].endswith(".weight"):
        return False
    return w.names[i].find(".ups.") < 0 and w.names[i].find(".pool.") < 0


@fieldwise_init
struct DSeq(Copyable, Movable):
    """A device sequence `[rows, cols]`, row-major."""

    var buf: DeviceBuffer[DT]
    var rows: Int
    var cols: Int


struct KokoroDevice(Movable):
    var ctx: DeviceContext
    var w: KokoroWeights
    """Names, shapes and the small host values the host stages read."""
    var arena: DeviceBuffer[DT]
    var dev: List[DeviceBuffer[DT]]
    """Tensor `i` of `w`, on the device (a slice of `arena`)."""
    var vocab: Dict[String, Int]
    var dir: String

    def __init__(
        out self, var ctx: DeviceContext, var w: KokoroWeights,
        var arena: DeviceBuffer[DT], var dev: List[DeviceBuffer[DT]],
        var vocab: Dict[String, Int], var dir: String,
    ):
        self.ctx = ctx^
        self.w = w^
        self.arena = arena^
        self.dev = dev^
        self.vocab = vocab^
        self.dir = dir^

    @staticmethod
    def load(var ctx: DeviceContext, var dir: String = String("")) raises -> Self:
        """Every tensor into one device arena (32-element aligned slices),
        then the large host copies freed: the host keeps only what its own
        stages read (the source module's tiny linear)."""
        if dir.byte_length() == 0:
            dir = kokoro_dir()
        var w = KokoroWeights.load(dir + "/kokoro-v1_0.safetensors")
        var vocab = kokoro_vocab(dir)
        var n = 0
        for i in range(len(w.tensors)):
            n += ((w.tensors[i].n + ALIGN - 1) // ALIGN) * ALIGN
        var arena = ctx.enqueue_create_buffer[DT](n)
        var staging = ctx.enqueue_create_host_buffer[DT](n)
        ctx.synchronize()
        var dev = List[DeviceBuffer[DT]]()
        var off = 0
        for i in range(len(w.tensors)):
            var m = w.tensors[i].n
            if _is_conv1d_weight(w, i):
                # [COUT, CIN, K] -> [COUT, K, CIN]
                var co = w.dim(i, 0)
                var ci = w.dim(i, 1)
                var kk = w.dim(i, 2)
                for o in range(co):
                    for c in range(ci):
                        for j in range(kk):
                            staging[off + o * kk * ci + j * ci + c] = w.tensors[i].data[o * ci * kk + c * kk + j]
            else:
                for k in range(m):
                    staging[off + k] = w.tensors[i].data[k]
            dev.append(arena.create_sub_buffer[DT](off, m))
            off += ((m + ALIGN - 1) // ALIGN) * ALIGN
        ctx.enqueue_copy(arena, staging)
        ctx.synchronize()
        _ = staging^
        for i in range(len(w.tensors)):
            if w.tensors[i].n > 4096:
                w.tensors[i].data = List[Scalar[DT]]()
        return Self(ctx^, w^, arena^, dev^, vocab^, dir^)

    # ── plumbing ─────────────────────────────────────────────────────────

    def t(mut self, name: String) raises -> DeviceBuffer[DT]:
        return self.dev[self.w.at(name)]

    def new(self, n: Int) raises -> DeviceBuffer[DT]:
        return self.ctx.enqueue_create_buffer[DT](n if n > 0 else 1)

    def zeros(self, n: Int) raises -> DeviceBuffer[DT]:
        var b = self.new(n)
        fill_dev(b, n, Scalar[DT](0), self.ctx)
        return b^

    def upload(self, vals: List[Float32]) raises -> DeviceBuffer[DT]:
        var n = len(vals)
        var hb = self.ctx.enqueue_create_host_buffer[DT](n)
        self.ctx.synchronize()
        for i in range(n):
            hb[i] = vals[i]
        var b = self.new(n)
        self.ctx.enqueue_copy(b, hb)
        self.ctx.synchronize()
        return b^

    def upload_ids(self, vals: List[Int]) raises -> DeviceBuffer[DType.int32]:
        var n = len(vals)
        var hb = self.ctx.enqueue_create_host_buffer[DType.int32](n)
        self.ctx.synchronize()
        for i in range(n):
            hb[i] = Int32(vals[i])
        var b = self.ctx.enqueue_create_buffer[DType.int32](n)
        self.ctx.enqueue_copy(b, hb)
        self.ctx.synchronize()
        return b^

    def download(self, s: DSeq) raises -> Tensor:
        """`s` as a host `Tensor` (`[rows, cols]`)."""
        var n = s.rows * s.cols
        var hb = self.ctx.enqueue_create_host_buffer[DT](n)
        self.ctx.enqueue_copy(hb, s.buf.create_sub_buffer[DT](0, n))
        self.ctx.synchronize()
        var t = Tensor.alloc(n)
        for i in range(n):
            t.data[i] = hb[i]
        return t^

    def copy(self, s: DSeq) raises -> DSeq:
        var n = s.rows * s.cols
        var b = self.new(n)
        self.ctx.enqueue_copy(b, s.buf.create_sub_buffer[DT](0, n))
        return DSeq(b, s.rows, s.cols)

    def gemm[TB: Bool, TF32: Bool = False](
        self, dst: DeviceBuffer[DT], a: DeviceBuffer[DT], b: DeviceBuffer[DT],
        m: Int, n: Int, k: Int,
    ) raises:
        """`dst[m, n] = a[m, k] @ (b[n, k]^T if TB else b[k, n])`."""
        comptime if default_accelerator().is_nvidia_gpu():
            cublas_gemm[False, TB, TF32](self.ctx, dst, a, b, m, n, k, 0.0)
        else:
            if n == 1:
                # ⚠ `max_matmul` on the GPU returns WRONG values at N = 1 (Metal
                # and CUDA alike; nn's Conv2D works around it the same way). At
                # N = 1, `b[1, k]` and `b[k, 1]` are the same memory.
                self.ctx.enqueue_function[_matvec_kernel](
                    a, b, dst, Int64(m), Int64(k), grid_dim=(m + 7) // 8, block_dim=256,
                )
                return
            var out_buf = dst  # a mutable handle: the view written through
            var av = TileTensor(a, row_major(m, k))
            var ov = TileTensor(out_buf, row_major(m, n))
            comptime if TB:
                var bv = TileTensor(b, row_major(n, k))
                max_matmul[transpose_b=True, target="gpu"](ov, av, bv, self.ctx)
            else:
                var bv = TileTensor(b, row_major(k, n))
                max_matmul[target="gpu"](ov, av, bv, self.ctx)

    def bias(self, y: DSeq, b: DeviceBuffer[DT]) raises:
        var n = y.rows * y.cols
        self.ctx.enqueue_function[_bias_rows_kernel](
            y.buf, b, Int64(y.rows), Int64(y.cols), grid_dim=_blocks(n), block_dim=TPB,
        )

    # ── layers ───────────────────────────────────────────────────────────

    def linear[TF32: Bool = False](mut self, prefix: String, x: DSeq) raises -> DSeq:
        """`nn.Linear` at `prefix` (weight `[out, in]`)."""
        var wi = self.w.at(prefix + ".weight")
        var n_out = self.w.dim(wi, 0)
        var y = DSeq(self.new(x.rows * n_out), x.rows, n_out)
        self.gemm[True, TF32](y.buf, x.buf, self.dev[wi], x.rows, n_out, x.cols)
        self.bias(y, self.t(prefix + ".bias"))
        return y^

    def conv1d[TF32: Bool = False](
        mut self, prefix: String, x: DSeq, stride: Int = 1, padding: Int = 0,
        dilation: Int = 1, has_bias: Bool = True,
    ) raises -> DSeq:
        """`nn.Conv1d` at `prefix` (weight `[COUT, CIN, K]`) over `x[l_in, CIN]`."""
        var wi = self.w.at(prefix + ".weight")
        var c_out = self.w.dim(wi, 0)
        var c_in = self.w.dim(wi, 1)
        var k = self.w.dim(wi, 2)
        var l_out = (x.rows + 2 * padding - dilation * (k - 1) - 1) // stride + 1
        var y = DSeq(self.new(l_out * c_out), l_out, c_out)
        if k == 1 and stride == 1 and padding == 0:
            self.gemm[True, TF32](y.buf, x.buf, self.dev[wi], l_out, c_out, c_in)
        else:
            var width = c_in * k
            var chunk = IM2COL_BUDGET // width
            if chunk < 1:
                chunk = 1
            if chunk > l_out:
                chunk = l_out
            var cols = self.new(chunk * width)
            var row0 = 0
            while row0 < l_out:
                var rows = chunk if row0 + chunk <= l_out else l_out - row0
                self.ctx.enqueue_function[_im2col_kernel](
                    x.buf, cols, Int64(x.rows), Int64(c_in), Int64(k), Int64(row0), Int64(rows),
                    Int64(stride), Int64(padding), Int64(dilation),
                    grid_dim=((c_in + TPB - 1) // TPB, k, rows), block_dim=TPB,
                )
                self.gemm[True, TF32](
                    y.buf.create_sub_buffer[DT](row0 * c_out, rows * c_out), cols,
                    self.dev[wi], rows, c_out, width,
                )
                row0 += rows
        if has_bias:
            self.bias(y, self.t(prefix + ".bias"))
        return y^

    def conv_transpose1d[TF32: Bool = False](
        mut self, prefix: String, x: DSeq, stride: Int, padding: Int,
    ) raises -> DSeq:
        """`nn.ConvTranspose1d` at `prefix` (weight `[CIN, COUT, K]`): one GEMM
        for every input position's contribution, then a gather."""
        var wi = self.w.at(prefix + ".weight")
        var c_in = self.w.dim(wi, 0)
        var c_out = self.w.dim(wi, 1)
        var k = self.w.dim(wi, 2)
        var l_out = (x.rows - 1) * stride - 2 * padding + k
        var tmp = self.new(x.rows * c_out * k)
        self.gemm[False, TF32](tmp, x.buf, self.dev[wi], x.rows, c_out * k, c_in)
        var y = DSeq(self.new(l_out * c_out), l_out, c_out)
        self.ctx.enqueue_function[_conv_t_gather_kernel](
            tmp, self.t(prefix + ".bias"), y.buf, Int64(x.rows), Int64(c_out), Int64(k),
            Int64(l_out), Int64(stride), Int64(padding),
            grid_dim=_blocks(l_out * c_out), block_dim=TPB,
        )
        return y^

    def leaky(self, x: DSeq, slope: Float32) raises:
        var n = x.rows * x.cols
        self.ctx.enqueue_function[_leaky_kernel](x.buf, Int64(n), slope, grid_dim=_blocks(n), block_dim=TPB)

    def adain(mut self, prefix: String, x: DSeq, style: DSeq) raises:
        """`AdaIN1d` at `prefix`, in place: instance norm over time, then the
        style's scale and shift."""
        var gb = self.linear(prefix + ".fc", style)
        var mean = self.new(x.cols)
        var inv = self.new(x.cols)
        var chunks = (x.rows + STAT_ROWS - 1) // STAT_ROWS
        var part = self.new(chunks * x.cols)
        for second in range(2):
            self.ctx.enqueue_function[_chan_partial_kernel](
                x.buf, mean, part, Int64(x.rows), Int64(x.cols), Int64(second),
                grid_dim=((x.cols + 31) // 32, chunks), block_dim=256,
            )
            self.ctx.enqueue_function[_chan_finish_kernel](
                part, mean, inv, Int64(chunks), Int64(x.rows), Int64(x.cols), Int64(second),
                grid_dim=(x.cols + TPB - 1) // TPB, block_dim=TPB,
            )
        var n = x.rows * x.cols
        self.ctx.enqueue_function[_adain_apply_kernel](
            x.buf, mean, inv, gb.buf, Int64(x.rows), Int64(x.cols),
            grid_dim=_blocks(n), block_dim=TPB,
        )

    def row_norm(mut self, x: DSeq, eps: Float32, gamma: String = String(""), beta: String = String("")) raises:
        var affine = gamma.byte_length() > 0
        var g = self.t(gamma) if affine else x.buf
        var b = self.t(beta) if affine else x.buf
        self.ctx.enqueue_function[_row_norm_kernel](
            x.buf, g, b, Int64(x.cols), eps, Int64(1 if affine else 0),
            grid_dim=x.rows, block_dim=TPB,
        )

    def bilstm(mut self, prefix: String, x: DSeq) raises -> DSeq:
        """`nn.LSTM(bidirectional=True)` at `prefix`: `[steps, 2H]`."""
        var wh = self.w.at(prefix + ".weight_hh_l0")
        var h4 = self.w.dim(wh, 0)
        var h = self.w.dim(wh, 1)
        var steps = x.rows
        var gxf = self.new(steps * h4)
        var gxr = self.new(steps * h4)
        self.gemm[True](gxf, x.buf, self.t(prefix + ".weight_ih_l0"), steps, h4, x.cols)
        self.gemm[True](gxr, x.buf, self.t(prefix + ".weight_ih_l0_reverse"), steps, h4, x.cols)
        var y = DSeq(self.new(steps * 2 * h), steps, 2 * h)
        var cst = self.new(2 * h)
        var whf = self.dev[wh]
        var whr = self.t(prefix + ".weight_hh_l0_reverse")
        var bif = self.t(prefix + ".bias_ih_l0")
        var bhf = self.t(prefix + ".bias_hh_l0")
        var bir = self.t(prefix + ".bias_ih_l0_reverse")
        var bhr = self.t(prefix + ".bias_hh_l0_reverse")
        for s in range(steps):
            self.ctx.enqueue_function[_lstm_step_kernel](
                gxf, gxr, whf, whr, bif, bhf, bir, bhr, y.buf, cst,
                Int64(s), Int64(steps), Int64(h),
                grid_dim=2 * h, block_dim=128,
            )
        return y^

    def adain_resblk(mut self, prefix: String, x: DSeq, style: DSeq) raises -> DSeq:
        """`AdainResBlk1d` at `prefix` (`ops.kk_adain_resblk`)."""
        var upsample = self.w.has(prefix + ".pool.weight")
        var learned_sc = self.w.has(prefix + ".conv1x1.weight")
        var r = self.copy(x)
        self.adain(prefix + ".norm1", r, style)
        self.leaky(r, 0.2)
        var pooled: DSeq
        if upsample:
            var pw = self.t(prefix + ".pool.weight")
            var pb = self.t(prefix + ".pool.bias")
            pooled = DSeq(self.new(2 * r.rows * r.cols), 2 * r.rows, r.cols)
            self.ctx.enqueue_function[_dw_t_x2_kernel](
                r.buf, pw, pb, pooled.buf, Int64(r.rows), Int64(r.cols),
                grid_dim=_blocks(2 * r.rows * r.cols), block_dim=TPB,
            )
        else:
            pooled = r^
        var a = self.conv1d(prefix + ".conv1", pooled, 1, 1)
        self.adain(prefix + ".norm2", a, style)
        self.leaky(a, 0.2)
        var res = self.conv1d(prefix + ".conv2", a, 1, 1)
        var up: DSeq
        if upsample:
            up = DSeq(self.new(2 * x.rows * x.cols), 2 * x.rows, x.cols)
            self.ctx.enqueue_function[_upsample_x2_kernel](
                x.buf, up.buf, Int64(x.rows), Int64(x.cols),
                grid_dim=_blocks(2 * x.rows * x.cols), block_dim=TPB,
            )
        else:
            up = x.copy()
        var sc: DSeq
        if learned_sc:
            sc = self.conv1d(prefix + ".conv1x1", up, 1, 0, 1, False)
        else:
            sc = up^
        var n = res.rows * res.cols
        self.ctx.enqueue_function[_sum_scale_kernel](
            res.buf, sc.buf, Int64(n), Float32(0.7071067811865476),
            grid_dim=_blocks(n), block_dim=TPB,
        )
        return res^

    def adain_resblock1[TF32: Bool](
        mut self, prefix: String, x: DSeq, style: DSeq, kernel: Int,
    ) raises:
        """`AdaINResBlock1` at `prefix`, in place (`decoder._adain_resblock1`)."""
        var n = x.rows * x.cols
        for i in range(3):
            var dil = 2 * i + 1
            var sfx = String(i)
            var xt = self.copy(x)
            self.adain(prefix + ".adain1." + sfx, xt, style)
            self.ctx.enqueue_function[_snake_kernel](
                xt.buf, self.t(prefix + ".alpha1." + sfx), Int64(n), Int64(x.cols),
                grid_dim=_blocks(n), block_dim=TPB,
            )
            var y = self.conv1d[TF32](prefix + ".convs1." + sfx, xt, 1, (kernel * dil - dil) // 2, dil)
            self.adain(prefix + ".adain2." + sfx, y, style)
            self.ctx.enqueue_function[_snake_kernel](
                y.buf, self.t(prefix + ".alpha2." + sfx), Int64(n), Int64(x.cols),
                grid_dim=_blocks(n), block_dim=TPB,
            )
            var z = self.conv1d[TF32](prefix + ".convs2." + sfx, y, 1, (kernel - 1) // 2, 1)
            self.ctx.enqueue_function[_axpy_kernel](
                x.buf, z.buf, Int64(n), Float32(1.0), grid_dim=_blocks(n), block_dim=TPB,
            )

    # ── the forward, stage by stage (as prosody.mojo / decoder.mojo) ─────

    def albert(mut self, ids: DeviceBuffer[DType.int32], t: Int) raises -> DSeq:
        comptime D = 768
        comptime E = 128
        comptime HEADS = 12
        var emb = DSeq(self.new(t * E), t, E)
        self.ctx.enqueue_function[_albert_embed_kernel](
            self.t("bert.embeddings.word_embeddings.weight"),
            self.t("bert.embeddings.position_embeddings.weight"),
            self.t("bert.embeddings.token_type_embeddings.weight"),
            ids, emb.buf, Int64(t), Int64(E), grid_dim=_blocks(t * E), block_dim=TPB,
        )
        self.row_norm(emb, 1e-12, String("bert.embeddings.LayerNorm.weight"), String("bert.embeddings.LayerNorm.bias"))
        var x = self.linear("bert.encoder.embedding_hidden_mapping_in", emb)
        var layer = String("bert.encoder.albert_layer_groups.0.albert_layers.0.")
        var sc = self.new(HEADS * t * t)
        for _ in range(12):
            var q = self.linear(layer + "attention.query", x)
            var k = self.linear(layer + "attention.key", x)
            var v = self.linear(layer + "attention.value", x)
            self.ctx.enqueue_function[_attn_scores_kernel](
                q.buf, k.buf, sc, Int64(t), Int64(HEADS), Int64(D // HEADS), Float32(0.125),
                grid_dim=_blocks(HEADS * t * t), block_dim=TPB,
            )
            var ctx_ = DSeq(self.new(t * D), t, D)
            self.ctx.enqueue_function[_attn_apply_kernel](
                sc, v.buf, ctx_.buf, Int64(t), Int64(HEADS), Int64(D // HEADS),
                grid_dim=_blocks(t * D), block_dim=TPB,
            )
            var proj = self.linear(layer + "attention.dense", ctx_)
            self.ctx.enqueue_function[_axpy_kernel](
                proj.buf, x.buf, Int64(t * D), Float32(1.0), grid_dim=_blocks(t * D), block_dim=TPB,
            )
            self.row_norm(proj, 1e-12, layer + "attention.LayerNorm.weight", layer + "attention.LayerNorm.bias")
            var mid = self.linear(layer + "ffn", proj)
            self.ctx.enqueue_function[_gelu_new_kernel](
                mid.buf, Int64(t * 2048), grid_dim=_blocks(t * 2048), block_dim=TPB,
            )
            var ffo = self.linear(layer + "ffn_output", mid)
            self.ctx.enqueue_function[_axpy_kernel](
                ffo.buf, proj.buf, Int64(t * D), Float32(1.0), grid_dim=_blocks(t * D), block_dim=TPB,
            )
            self.row_norm(ffo, 1e-12, layer + "full_layer_layer_norm.weight", layer + "full_layer_layer_norm.bias")
            x = ffo^
        return x^

    def durations(
        mut self, ids: List[Int], ref_s: List[Float32], speed: Float32,
        mut taps: List[DSeq],
    ) raises -> Tuple[DSeq, List[Int], List[Float32]]:
        """PL-BERT, the duration encoder and LSTM, the frames per phoneme.
        Returns `(dur_enc [T, 640], frames per token, durations before
        rounding)`; `taps` gets bert, d_en, dur_enc, dur_lstm (for the gate)."""
        comptime H = 512
        comptime S = 128
        var t = len(ids)
        if t > KK_CONTEXT:
            raise Error("kokoro: " + String(t) + " tokens, the model's context is " + String(KK_CONTEXT))
        var ids_d = self.upload_ids(ids)
        var bert = self.albert(ids_d, t)
        var d_en = self.linear("bert_encoder", bert)
        var prosody = List[Float32]()
        for i in range(S):
            prosody.append(ref_s[S + i])
        var style = DSeq(self.upload(prosody), 1, S)
        var x = DSeq(self.new(t * (H + S)), t, H + S)
        self.ctx.enqueue_function[_copy_cols_kernel](
            d_en.buf, x.buf, Int64(t), Int64(H), Int64(H + S), Int64(0),
            grid_dim=_blocks(t * H), block_dim=TPB,
        )
        self.ctx.enqueue_function[_bcast_cols_kernel](
            style.buf, x.buf, Int64(t), Int64(S), Int64(H + S), Int64(H),
            grid_dim=_blocks(t * S), block_dim=TPB,
        )
        for blk in range(3):
            var lstm = String("predictor.text_encoder.lstms." + String(2 * blk))
            var ada = String("predictor.text_encoder.lstms." + String(2 * blk + 1))
            var lo = self.bilstm(lstm, x)
            var gb = self.linear(ada + ".fc", style)
            self.row_norm(lo, 1e-5)
            self.ctx.enqueue_function[_modulate_kernel](
                lo.buf, gb.buf, x.buf, Int64(t), Int64(H), Int64(H + S),
                grid_dim=_blocks(t * H), block_dim=TPB,
            )
        var dur_lstm = self.bilstm("predictor.lstm", x)
        var logits = self.download(self.linear("predictor.duration_proj.linear_layer", dur_lstm))
        var frames = List[Int]()
        var dur = List[Float32]()
        for i in range(t):
            var total = Float32(0.0)
            for d in range(KK_MAX_DUR):
                total += 1.0 / (1.0 + exp(-logits.data[i * KK_MAX_DUR + d]))
            var v = total / speed
            dur.append(v)
            var fl = Int(v)
            var frac = v - Float32(fl)
            var r = fl + 1 if frac > 0.5 else (fl if frac < 0.5 else (fl if fl % 2 == 0 else fl + 1))
            frames.append(r if r >= 1 else 1)
        taps.append(bert^)
        taps.append(d_en^)
        taps.append(x.copy())
        taps.append(dur_lstm^)
        return (x^, frames^, dur^)

    def acoustic(
        mut self, ids: List[Int], ref_s: List[Float32], dur_enc: DSeq,
        frames: List[Int], mut taps: List[DSeq],
    ) raises -> Tuple[DSeq, DSeq, DSeq]:
        """Text encoder, alignment, pitch and energy. Returns `(asr [F, 512],
        f0 [2F, 1], n [2F, 1])`; `taps` gets t_en and en."""
        comptime S = 128
        var t = len(ids)
        var ids_d = self.upload_ids(ids)
        var x = DSeq(self.new(t * 512), t, 512)
        self.ctx.enqueue_function[_embed_kernel](
            self.t("text_encoder.embedding.weight"), ids_d, x.buf, Int64(t), Int64(512),
            grid_dim=_blocks(t * 512), block_dim=TPB,
        )
        for i in range(3):
            var cnn = String("text_encoder.cnn." + String(i))
            var y = self.conv1d(cnn + ".0", x, 1, 2)
            self.row_norm(y, 1e-5, cnn + ".1.gamma", cnn + ".1.beta")
            self.leaky(y, 0.2)
            x = y^
        var t_en = self.bilstm("text_encoder.lstm", x)

        var idx = List[Int]()
        for i in range(t):
            for _ in range(frames[i]):
                idx.append(i)
        var fr = len(idx)
        var idx_d = self.upload_ids(idx)
        var en = DSeq(self.new(fr * dur_enc.cols), fr, dur_enc.cols)
        self.ctx.enqueue_function[_gather_rows_kernel](
            dur_enc.buf, idx_d, en.buf, Int64(fr), Int64(dur_enc.cols),
            grid_dim=_blocks(fr * dur_enc.cols), block_dim=TPB,
        )
        var asr = DSeq(self.new(fr * 512), fr, 512)
        self.ctx.enqueue_function[_gather_rows_kernel](
            t_en.buf, idx_d, asr.buf, Int64(fr), Int64(512),
            grid_dim=_blocks(fr * 512), block_dim=TPB,
        )
        var prosody = List[Float32]()
        for i in range(S):
            prosody.append(ref_s[S + i])
        var style = DSeq(self.upload(prosody), 1, S)
        var shared = self.bilstm("predictor.shared", en)
        var curves = List[DSeq]()
        for which in ["F0", "N"]:
            var w = String(which)
            var a = self.adain_resblk("predictor." + w + ".0", shared, style)
            var b = self.adain_resblk("predictor." + w + ".1", a, style)
            var c = self.adain_resblk("predictor." + w + ".2", b, style)
            curves.append(self.conv1d("predictor." + w + "_proj", c))
        taps.append(t_en^)
        taps.append(en^)
        return (asr^, curves[0].copy(), curves[1].copy())

    def decoder_blocks(
        mut self, asr: DSeq, f0: DSeq, n: DSeq, ref_s: List[Float32],
        mut taps: List[DSeq],
    ) raises -> DSeq:
        """`Decoder.forward` up to the generator; `taps` gets encode and the
        four decode blocks. Returns `[2F, 512]`."""
        var fr = asr.rows
        var dec_style = List[Float32]()
        for i in range(128):
            dec_style.append(ref_s[i])
        var style = DSeq(self.upload(dec_style), 1, 128)
        var f0c = self.conv1d("decoder.F0_conv", f0, 2, 1)
        var nc = self.conv1d("decoder.N_conv", n, 2, 1)
        var x = DSeq(self.new(fr * 514), fr, 514)
        self.ctx.enqueue_function[_copy_cols_kernel](
            asr.buf, x.buf, Int64(fr), Int64(512), Int64(514), Int64(0), grid_dim=_blocks(fr * 512), block_dim=TPB,
        )
        self.ctx.enqueue_function[_copy_cols_kernel](
            f0c.buf, x.buf, Int64(fr), Int64(1), Int64(514), Int64(512), grid_dim=_blocks(fr), block_dim=TPB,
        )
        self.ctx.enqueue_function[_copy_cols_kernel](
            nc.buf, x.buf, Int64(fr), Int64(1), Int64(514), Int64(513), grid_dim=_blocks(fr), block_dim=TPB,
        )
        var cur = self.adain_resblk("decoder.encode", x, style)
        taps.append(cur.copy())
        var asr_res = self.conv1d("decoder.asr_res.0", asr)
        for b in range(4):
            var cat = DSeq(self.new(fr * 1090), fr, 1090)
            self.ctx.enqueue_function[_copy_cols_kernel](
                cur.buf, cat.buf, Int64(fr), Int64(1024), Int64(1090), Int64(0), grid_dim=_blocks(fr * 1024), block_dim=TPB,
            )
            self.ctx.enqueue_function[_copy_cols_kernel](
                asr_res.buf, cat.buf, Int64(fr), Int64(64), Int64(1090), Int64(1024), grid_dim=_blocks(fr * 64), block_dim=TPB,
            )
            self.ctx.enqueue_function[_copy_cols_kernel](
                f0c.buf, cat.buf, Int64(fr), Int64(1), Int64(1090), Int64(1088), grid_dim=_blocks(fr), block_dim=TPB,
            )
            self.ctx.enqueue_function[_copy_cols_kernel](
                nc.buf, cat.buf, Int64(fr), Int64(1), Int64(1090), Int64(1089), grid_dim=_blocks(fr), block_dim=TPB,
            )
            cur = self.adain_resblk("decoder.decode." + String(b), cat, style)
            taps.append(cur.copy())
        return cur^

    def generate(
        mut self, var cur: DSeq, har: DSeq, ref_s: List[Float32], mut taps: List[DSeq],
    ) raises -> DSeq:
        """The generator after its source: `cur [2F, 512]` and the source's
        spectrum `har [120F + 1, 22]` -> `conv_post [120F + 1, 22]`. `taps`
        gets ups 0, stage 0 (after its leaky ReLU), ups 1, stage 1."""
        var dec_style = List[Float32]()
        for i in range(128):
            dec_style.append(ref_s[i])
        var style = DSeq(self.upload(dec_style), 1, 128)
        var gen = String("decoder.generator.")
        for i in range(2):
            self.leaky(cur, 0.1)
            var src: DSeq
            if i == 0:
                src = self.conv1d[KD_TF32](gen + "noise_convs.0", har, 6, 3)
            else:
                src = self.conv1d[KD_TF32](gen + "noise_convs.1", har)
            self.adain_resblock1[KD_TF32](gen + "noise_res." + String(i), src, style, 7 if i == 0 else 11)
            var up = self.conv_transpose1d[KD_TF32](gen + "ups." + String(i), cur, 10 if i == 0 else 6, 5 if i == 0 else 3)
            taps.append(self.copy(up))
            var xin: DSeq
            if i == 1:
                xin = DSeq(self.new((up.rows + 1) * up.cols), up.rows + 1, up.cols)
                self.ctx.enqueue_function[_reflect_left_kernel](
                    up.buf, xin.buf, Int64(up.rows), Int64(up.cols),
                    grid_dim=_blocks((up.rows + 1) * up.cols), block_dim=TPB,
                )
            else:
                xin = up^
            if src.rows != xin.rows:
                raise Error("kokoro: the source has " + String(src.rows) + " steps at a stage of " + String(xin.rows))
            var nn = xin.rows * xin.cols
            self.ctx.enqueue_function[_axpy_kernel](
                xin.buf, src.buf, Int64(nn), Float32(1.0), grid_dim=_blocks(nn), block_dim=TPB,
            )
            var mean = DSeq(self.zeros(nn), xin.rows, xin.cols)
            var kernels: List[Int] = [3, 7, 11]
            for j in range(3):
                var branch = self.copy(xin)
                self.adain_resblock1[KD_TF32](gen + "resblocks." + String(i * 3 + j), branch, style, kernels[j])
                self.ctx.enqueue_function[_axpy_kernel](
                    mean.buf, branch.buf, Int64(nn), Float32(1.0), grid_dim=_blocks(nn), block_dim=TPB,
                )
            self.ctx.enqueue_function[_scale_kernel](
                mean.buf, Int64(nn), Float32(1.0 / 3.0), grid_dim=_blocks(nn), block_dim=TPB,
            )
            cur = mean^
            if i == 0:
                var tap0 = self.copy(cur)
                self.leaky(tap0, 0.1)
                taps.append(tap0^)
        self.leaky(cur, 0.01)
        taps.append(self.copy(cur))
        return self.conv1d[KD_TF32](gen + "conv_post", cur, 1, 3)

    # ── end to end ───────────────────────────────────────────────────────

    def synthesize(
        mut self, phonemes: String, voice: String, speed: Float32 = 1.0, seed: UInt64 = 0,
    ) raises -> KokoroSpeech:
        """`KokoroTTS.synthesize` on the device (host: durations, source, STFTs)."""
        var host = KokoroTTS()
        host.vocab = self.vocab.copy()
        host.dir = self.dir
        var ids = host.phoneme_ids(phonemes)
        var ref_s = host.style_row(voice, phonemes.count_codepoints())
        var taps = List[DSeq]()
        var p = self.durations(ids, ref_s, speed, taps)
        var a = self.acoustic(ids, ref_s, p[0], p[1], taps)
        var x = self.decoder_blocks(a[0], a[1], a[2], ref_s, taps)
        var f0 = self.download(a[1])
        var fr = a[0].rows
        var noise = kokoro_noise(fr * KK_SAMPLES_PER_FRAME, seed)
        var source = kk_harmonic_source(self.w, f0, 2 * fr, noise)
        var har_h = Tensor()
        var hf = kk_stft(source, har_h)
        var har = DSeq(self.upload(har_h.data), hf, 22)
        var post = self.download(self.generate(x^, har, ref_s, taps))
        var audio = kk_istft(post, hf)
        return KokoroSpeech(audio^, p[1].copy(), fr, phonemes)

    def speak(
        mut self, mut es: EspeakNG, text: String, voice: String,
        speed: Float32 = 1.0, seed: UInt64 = 0, mode: Int = -1,
    ) raises -> KokoroSpeech:
        var language = kokoro_language(voice)
        var m = mode if mode >= 0 else kokoro_default_g2p(language)
        var ps = kokoro_phonemes(es, text, language, m, self.vocab)
        if ps.byte_length() == 0:
            raise Error("kokoro: nothing to say in '" + text + "'")
        return self.synthesize(ps, voice, speed, seed)
