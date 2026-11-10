# +--------------------------------------------------------------------------+ #
# | Kokoro-82M — the operations its forward is made of (CPU)
# +--------------------------------------------------------------------------+ #
"""Linear, LayerNorm and bidirectional LSTM over `KokoroWeights`, on the host.

Weights are in torch's layout (`weights.mojo`): a `nn.Linear` is `[out, in]`
and is applied as `x @ W^T` (`gemm_cpu[False, True]`); an LSTM is
`weight_ih_l0` `[4H, IN]`, `weight_hh_l0` `[4H, H]`, two biases, gates stacked
`i, f, g, o`, and the same four tensors with `_reverse` for the other
direction.

Shapes are RUNTIME values here: this is the exact CPU path, written to be
compared with the reference, and Accelerate's GEMM takes its dimensions as
arguments. (On a CPU without Accelerate `gemm_cpu` is a plain triple loop:
correct, and slow.) The device path, with compiled maxima and masks as in
Parakeet, comes after the stages are gated.
"""

from std.math import exp, sqrt, tanh

from noeira.nn.core.cpu_gemm import gemm_cpu
from noeira.nn.core.tensor import Tensor
from .weights import KokoroWeights


def kk_linear(
    mut w: KokoroWeights, prefix: String, x: Tensor, rows: Int, mut y: Tensor,
) raises:
    """`y[rows, out] = x[rows, in] @ W^T + b` for the `nn.Linear` at `prefix`."""
    var wi = w.at(prefix + ".weight")
    var n_out = w.dim(wi, 0)
    var n_in = w.dim(wi, 1)
    y.ensure(rows * n_out)
    gemm_cpu[False, True](
        rows, n_out, n_in, 1.0, x, n_in, w.tensors[wi], n_in, 0.0, y, n_out
    )
    var bi = w.at(prefix + ".bias")
    for r in range(rows):
        for j in range(n_out):
            y.data[r * n_out + j] += w.tensors[bi].data[j]


def kk_layer_norm(
    mut x: Tensor, rows: Int, dim: Int, eps: Float32,
):
    """Normalise each of the `rows` rows of `x[rows, dim]` in place (biased
    variance), without an affine."""
    for r in range(rows):
        var base = r * dim
        var total = Float32(0.0)
        for d in range(dim):
            total += x.data[base + d]
        var mean = total / Float32(dim)
        var sq = Float32(0.0)
        for d in range(dim):
            var diff = x.data[base + d] - mean
            sq += diff * diff
        var inv = Float32(1.0) / sqrt(sq / Float32(dim) + eps)
        for d in range(dim):
            x.data[base + d] = (x.data[base + d] - mean) * inv


def kk_layer_norm_affine(
    mut w: KokoroWeights, gamma_name: String, beta_name: String,
    mut x: Tensor, rows: Int, eps: Float32,
) raises:
    """`nn.LayerNorm` in place over the last axis of `x[rows, dim]`."""
    var gi = w.at(gamma_name)
    var bi = w.at(beta_name)
    var dim = w.dim(gi, 0)
    kk_layer_norm(x, rows, dim, eps)
    for r in range(rows):
        for d in range(dim):
            x.data[r * dim + d] = x.data[r * dim + d] * w.tensors[gi].data[d] + w.tensors[bi].data[d]


def _sigmoid(v: Float32) -> Float32:
    return 1.0 / (1.0 + exp(-v))


def kk_bilstm(
    mut w: KokoroWeights, prefix: String, x: Tensor, steps: Int, mut y: Tensor,
) raises -> Int:
    """`nn.LSTM(bidirectional=True, batch_first=True)` at `prefix` over
    `x[steps, IN]`: `y[steps, 2H]`, the forward direction's hidden state in the
    first `H` columns and the reverse direction's in the last `H`. Returns `2H`.
    Both directions start from a zero state; the reverse one starts at the
    last step."""
    var probe = w.at(prefix + ".weight_hh_l0")
    var h4 = w.dim(probe, 0)
    var h = w.dim(probe, 1)
    y.ensure(steps * 2 * h)
    var gates_x = Tensor.alloc(steps * h4)
    var gates_h = Tensor.alloc(h4)
    var hid = Tensor.alloc(h)
    var cell = Tensor.alloc(h)
    for direction in range(2):
        var sfx = String("") if direction == 0 else String("_reverse")
        var wih = w.at(prefix + ".weight_ih_l0" + sfx)
        var whh = w.at(prefix + ".weight_hh_l0" + sfx)
        var bih = w.at(prefix + ".bias_ih_l0" + sfx)
        var bhh = w.at(prefix + ".bias_hh_l0" + sfx)
        var n_in = w.dim(wih, 1)
        # Every step's input contribution at once: [steps, 4H].
        gemm_cpu[False, True](
            steps, h4, n_in, 1.0, x, n_in, w.tensors[wih], n_in, 0.0, gates_x, h4
        )
        for i in range(h):
            hid.data[i] = 0.0
            cell.data[i] = 0.0
        for s in range(steps):
            var t = s if direction == 0 else steps - 1 - s
            gemm_cpu[False, True](
                1, h4, h, 1.0, hid, h, w.tensors[whh], h, 0.0, gates_h, h4
            )
            var gx = t * h4
            for i in range(h):
                var gi = _sigmoid(
                    gates_x.data[gx + i] + w.tensors[bih].data[i]
                    + gates_h.data[i] + w.tensors[bhh].data[i]
                )
                var gf = _sigmoid(
                    gates_x.data[gx + h + i] + w.tensors[bih].data[h + i]
                    + gates_h.data[h + i] + w.tensors[bhh].data[h + i]
                )
                var gg = tanh(
                    gates_x.data[gx + 2 * h + i] + w.tensors[bih].data[2 * h + i]
                    + gates_h.data[2 * h + i] + w.tensors[bhh].data[2 * h + i]
                )
                var go = _sigmoid(
                    gates_x.data[gx + 3 * h + i] + w.tensors[bih].data[3 * h + i]
                    + gates_h.data[3 * h + i] + w.tensors[bhh].data[3 * h + i]
                )
                var c = gf * cell.data[i] + gi * gg
                cell.data[i] = c
                y.data[t * 2 * h + direction * h + i] = go * tanh(c)
            for i in range(h):
                hid.data[i] = y.data[t * 2 * h + direction * h + i]
    return 2 * h


# ── 1-D convolutions and the style-conditioned normalisation ───────────────
# Activations are channels-LAST here, `[length, channels]` (torch's are
# `[channels, length]`): a convolution is then one GEMM over the rows.


def kk_conv1d(
    mut w: KokoroWeights, prefix: String, x: Tensor, l_in: Int, mut y: Tensor,
    stride: Int = 1, padding: Int = 0, dilation: Int = 1, has_bias: Bool = True,
) raises -> Int:
    """`nn.Conv1d` at `prefix` (weight `[COUT, CIN, K]`) over `x[l_in, CIN]`:
    `y[l_out, COUT]`. Returns `l_out`. The window of every output position is
    gathered into a row (`[l_out, CIN * K]`, CIN-major as the weight is) and
    multiplied by the weight's transpose."""
    var wi = w.at(prefix + ".weight")
    var c_out = w.dim(wi, 0)
    var c_in = w.dim(wi, 1)
    var k = w.dim(wi, 2)
    var l_out = (l_in + 2 * padding - dilation * (k - 1) - 1) // stride + 1
    var width = c_in * k
    y.ensure(l_out * c_out)
    if k == 1 and stride == 1 and padding == 0:
        gemm_cpu[False, True](l_out, c_out, c_in, 1.0, x, c_in, w.tensors[wi], c_in, 0.0, y, c_out)
    else:
        var cols = Tensor.alloc(l_out * width)
        for o in range(l_out):
            var base = o * width
            for j in range(k):
                var t = o * stride - padding + j * dilation
                if t < 0 or t >= l_in:
                    continue
                for c in range(c_in):
                    cols.data[base + c * k + j] = x.data[t * c_in + c]
        gemm_cpu[False, True](l_out, c_out, width, 1.0, cols, width, w.tensors[wi], width, 0.0, y, c_out)
    if has_bias:
        var bi = w.at(prefix + ".bias")
        for o in range(l_out):
            for c in range(c_out):
                y.data[o * c_out + c] += w.tensors[bi].data[c]
    return l_out


def kk_conv_transpose_depthwise_x2(
    mut w: KokoroWeights, prefix: String, x: Tensor, l_in: Int, mut y: Tensor,
) raises -> Int:
    """`nn.ConvTranspose1d(C, C, 3, stride=2, groups=C, padding=1,
    output_padding=1)` at `prefix` (weight `[C, 1, 3]`): `y[2 l_in, C]`.
    Input `t` reaches outputs `2t - 1 + k`, `k = 0..2`."""
    var wi = w.at(prefix + ".weight")
    var bi = w.at(prefix + ".bias")
    var c_n = w.dim(wi, 0)
    var l_out = 2 * l_in
    y.ensure(l_out * c_n)
    for o in range(l_out):
        for c in range(c_n):
            y.data[o * c_n + c] = w.tensors[bi].data[c]
    for t in range(l_in):
        for k in range(3):
            var o = 2 * t - 1 + k
            if o < 0 or o >= l_out:
                continue
            for c in range(c_n):
                y.data[o * c_n + c] += x.data[t * c_n + c] * w.tensors[wi].data[c * 3 + k]
    return l_out


def kk_adain(
    mut w: KokoroWeights, prefix: String, mut x: Tensor, length: Int,
    style: Tensor,
) raises:
    """`AdaIN1d` at `prefix`, in place over `x[length, C]`: each channel is
    normalised over TIME (instance norm, biased variance, eps 1e-5; the
    checkpoint has no affine of its own for it), then `(1 + gamma) * . + beta`
    with `[gamma | beta] = fc(style)`."""
    var fi = w.at(prefix + ".fc.weight")
    var c_n = w.dim(fi, 0) // 2
    var gb = Tensor.alloc(2 * c_n)
    kk_linear(w, prefix + ".fc", style, 1, gb)
    for c in range(c_n):
        var total = Float32(0.0)
        for t in range(length):
            total += x.data[t * c_n + c]
        var mean = total / Float32(length)
        var sq = Float32(0.0)
        for t in range(length):
            var d = x.data[t * c_n + c] - mean
            sq += d * d
        var inv = Float32(1.0) / sqrt(sq / Float32(length) + Float32(1e-5))
        var g = Float32(1.0) + gb.data[c]
        var b = gb.data[c_n + c]
        for t in range(length):
            x.data[t * c_n + c] = g * ((x.data[t * c_n + c] - mean) * inv) + b


def kk_leaky_relu(mut x: Tensor, n: Int, slope: Float32):
    for i in range(n):
        if x.data[i] < 0.0:
            x.data[i] = x.data[i] * slope


def kk_adain_resblk(
    mut w: KokoroWeights, prefix: String, x: Tensor, l_in: Int, style: Tensor,
    mut y: Tensor,
) raises -> Int:
    """`AdainResBlk1d` at `prefix` over `x[l_in, DIN]`: `y[l_out, DOUT]`,
    `l_out = 2 l_in` when the block upsamples (it has a `pool`), else `l_in`.

        residual = conv2(lrelu(norm2(conv1(pool(lrelu(norm1(x)))))))
        shortcut = conv1x1(nearest_x2(x))        each only where it exists
        y = (residual + shortcut) / sqrt(2)
    """
    var c1 = w.at(prefix + ".conv1.weight")
    var d_out = w.dim(c1, 0)
    var d_in = w.dim(c1, 1)
    var upsample = w.has(prefix + ".pool.weight")
    var learned_sc = w.has(prefix + ".conv1x1.weight")
    var l_out = 2 * l_in if upsample else l_in

    var r = Tensor.alloc(l_in * d_in)
    for i in range(l_in * d_in):
        r.data[i] = x.data[i]
    kk_adain(w, prefix + ".norm1", r, l_in, style)
    kk_leaky_relu(r, l_in * d_in, 0.2)
    var pooled = Tensor()
    if upsample:
        _ = kk_conv_transpose_depthwise_x2(w, prefix + ".pool", r, l_in, pooled)
    else:
        pooled = r^
    var a = Tensor()
    _ = kk_conv1d(w, prefix + ".conv1", pooled, l_out, a, 1, 1)
    kk_adain(w, prefix + ".norm2", a, l_out, style)
    kk_leaky_relu(a, l_out * d_out, 0.2)
    var res = Tensor()
    _ = kk_conv1d(w, prefix + ".conv2", a, l_out, res, 1, 1)

    var up: Tensor
    if upsample:
        up = Tensor.alloc(l_out * d_in)
        for t in range(l_in):
            for c in range(d_in):
                up.data[(2 * t) * d_in + c] = x.data[t * d_in + c]
                up.data[(2 * t + 1) * d_in + c] = x.data[t * d_in + c]
    else:
        up = Tensor.alloc(l_in * d_in)
        for i in range(l_in * d_in):
            up.data[i] = x.data[i]
    var sc = Tensor()
    if learned_sc:
        _ = kk_conv1d(w, prefix + ".conv1x1", up, l_out, sc, 1, 0, 1, False)
    else:
        sc = up^
    y.ensure(l_out * d_out)
    var rsqrt2 = Float32(0.7071067811865476)
    for i in range(l_out * d_out):
        y.data[i] = (res.data[i] + sc.data[i]) * rsqrt2
    return l_out


def kk_conv_transpose1d(
    mut w: KokoroWeights, prefix: String, x: Tensor, l_in: Int, mut y: Tensor,
    stride: Int, padding: Int,
) raises -> Int:
    """`nn.ConvTranspose1d` at `prefix` (weight `[CIN, COUT, K]`) over
    `x[l_in, CIN]`: `y[l_out, COUT]`, `l_out = (l_in - 1) stride - 2 padding + K`.
    One GEMM gives every input position's `[COUT, K]` contribution; position
    `t`, tap `k` lands on output `t stride - padding + k`."""
    var wi = w.at(prefix + ".weight")
    var bi = w.at(prefix + ".bias")
    var c_in = w.dim(wi, 0)
    var c_out = w.dim(wi, 1)
    var k = w.dim(wi, 2)
    var l_out = (l_in - 1) * stride - 2 * padding + k
    var width = c_out * k
    var tmp = Tensor.alloc(l_in * width)
    gemm_cpu[False, False](l_in, width, c_in, 1.0, x, c_in, w.tensors[wi], width, 0.0, tmp, width)
    y.ensure(l_out * c_out)
    for o in range(l_out):
        for c in range(c_out):
            y.data[o * c_out + c] = w.tensors[bi].data[c]
    for t in range(l_in):
        for j in range(k):
            var o = t * stride - padding + j
            if o < 0 or o >= l_out:
                continue
            for c in range(c_out):
                y.data[o * c_out + c] += tmp.data[t * width + c * k + j]
    return l_out
