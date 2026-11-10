# +--------------------------------------------------------------------------+ #
# | Kokoro-82M — the decoder and the iSTFTNet vocoder
# +--------------------------------------------------------------------------+ #
"""Frames to waveform: `Decoder.forward` and its `Generator`.

    F0, N      = stride-2 convs of the pitch and energy curves        [F, 1] each
    x          = encode([asr | F0 | N])                               [F, 1024]
    x          = decode_0..3([x | asr_res | F0 | N])                  [2F, 512] (the last doubles)
    waveform   = generator(x, style, pitch curve)                     [600 F]

The generator (iSTFTNet with a harmonic source):

    source     nine sines at the upsampled pitch and its harmonics, a random
               start phase per harmonic, noise; merged by a linear + tanh
    har        STFT of the source (n_fft 20, hop 5): magnitude | phase, 22 rows
    2 stages   x = leaky_relu(x, 0.1); x = transposed conv (x10, then x6)
               x += residual block(conv(har))          the source, at this scale
               x = mean of three residual blocks (kernels 3, 7, 11)
    post       leaky_relu, conv to 22: exp(.) of 11 = magnitude, sin(.) of 11 = phase
    inverse STFT -> 24 kHz

Sequences are channels-last (`[length, channels]`), as in `ops.mojo`.

⚠ THE SOURCE IS RANDOM. `KokoroNoise` holds the two draws the waveform depends
on — nine uniform start phases, and a Gaussian per (sample, harmonic). The
gate fills it with the reference's recorded values; a caller that wants speech
fills it from any generator (the choice changes the texture of the noise, not
the words).

⚠ AND THE SOURCE IS NOT REPRODUCIBLE BEYOND ~1e-3, EVEN WITH THE SAME DRAWS. Its
sines are `sin` of a phase accumulated over the whole utterance — thousands of
radians — in float32, where one step is ~5e-4: the reference carries its own
rounding there (a float64 evaluation of the same formulas differs from it by
9e-4 of a 0.15 peak). The vocoder amplifies any source difference (its output
goes through an `exp`), so a waveform cannot be matched sample for sample by
ANY reimplementation. Hence `kk_generate` takes the source's spectrum as an
argument: the gate gives it the reference's own and requires a tight match
from there, and gates `kk_harmonic_source` separately, within that noise.
"""

from std.math import exp, sin, cos, sqrt, tanh, atan2, floor, pi

from noeira.nn.core.tensor import Tensor
from .ops import (
    kk_linear, kk_conv1d, kk_adain, kk_adain_resblk, kk_leaky_relu,
    kk_conv_transpose1d,
)
from .prosody import KokoroAcoustic, KK_STYLE
from .weights import KokoroWeights


comptime KK_SAMPLE_RATE: Int = 24000
comptime KK_HARMONICS: Int = 9
comptime KK_N_FFT: Int = 20
comptime KK_HOP: Int = 5
comptime KK_BINS: Int = KK_N_FFT // 2 + 1
comptime KK_UPSAMPLE: Int = 300
"""Source samples per pitch-curve step: 10 x 6 x the STFT hop of 5."""
comptime KK_SAMPLES_PER_FRAME: Int = 600


struct KokoroNoise(Movable):
    """The source module's random draws for one utterance of `samples` samples."""

    var phase: List[Float32]
    """`[9]` uniform in [0, 1): each harmonic's start phase (the fundamental's
    is not used)."""
    var sines: List[Float32]
    """`[samples, 9]` standard normal: the noise added to the sines."""

    def __init__(out self):
        self.phase = List[Float32]()
        self.sines = List[Float32]()


struct KokoroDecoded(Movable):
    """The decoder's stages (the gate reads them) and the waveform."""

    var frames: Int
    var encode: Tensor
    var decode: List[Tensor]
    var source: List[Float32]
    """`[600 F]`: the harmonic source."""
    var har: Tensor
    """`[120 F + 1, 22]`: its spectrum, 11 magnitudes then 11 phases."""
    var ups: List[Tensor]
    var res: List[Tensor]
    """Each stage after its leaky ReLU: what the next convolution reads."""
    var post: Tensor
    """`[120 F + 1, 22]`"""
    var audio: List[Float32]

    def __init__(out self):
        self.frames = 0
        self.encode = Tensor()
        self.decode = List[Tensor]()
        self.source = List[Float32]()
        self.har = Tensor()
        self.ups = List[Tensor]()
        self.res = List[Tensor]()
        self.post = Tensor()
        self.audio = List[Float32]()


def _snake(mut x: Tensor, length: Int, ref alpha: Tensor):
    """`x + sin^2(a x) / a`, `a` per channel."""
    var c_n = alpha.n
    for t in range(length):
        for c in range(c_n):
            var a = alpha.data[c]
            var v = x.data[t * c_n + c]
            var s = sin(a * v)
            x.data[t * c_n + c] = v + (1.0 / a) * (s * s)


def _adain_resblock1(
    mut w: KokoroWeights, prefix: String, mut x: Tensor, length: Int,
    style: Tensor, kernel: Int,
) raises:
    """`AdaINResBlock1` at `prefix`, in place over `x[length, C]`: three times
    (AdaIN, Snake, conv with dilation 1 / 3 / 5, AdaIN, Snake, conv), each
    added back to its input."""
    var probe = w.at(prefix + ".convs1.0.weight")
    var c_n = w.dim(probe, 0)
    var n = length * c_n
    var xt = Tensor.alloc(n)
    var y = Tensor()
    for i in range(3):
        var dil = 2 * i + 1
        var sfx = String(i)
        for k in range(n):
            xt.data[k] = x.data[k]
        kk_adain(w, prefix + ".adain1." + sfx, xt, length, style)
        _snake(xt, length, w.tensors[w.at(prefix + ".alpha1." + sfx)])
        _ = kk_conv1d(w, prefix + ".convs1." + sfx, xt, length, y, 1, (kernel * dil - dil) // 2, dil)
        kk_adain(w, prefix + ".adain2." + sfx, y, length, style)
        _snake(y, length, w.tensors[w.at(prefix + ".alpha2." + sfx)])
        _ = kk_conv1d(w, prefix + ".convs2." + sfx, y, length, xt, 1, (kernel - 1) // 2, 1)
        for k in range(n):
            x.data[k] += xt.data[k]


def _linear_resample(ref src: List[Float32], n_in: Int, n_out: Int, scale: Float32) -> List[Float32]:
    """`F.interpolate(mode="linear", align_corners=False)` for one channel:
    output `i` reads source position `scale (i + 0.5) - 0.5`, clamped at 0."""
    var out = List[Float32](length=n_out, fill=0.0)
    for i in range(n_out):
        var pos = scale * (Float32(i) + 0.5) - 0.5
        if pos < 0.0:
            pos = 0.0
        var i0 = Int(floor(pos))
        if i0 > n_in - 1:
            i0 = n_in - 1
        var i1 = i0 + 1 if i0 + 1 < n_in else n_in - 1
        var lam = pos - Float32(i0)
        out[i] = (1.0 - lam) * src[i0] + lam * src[i1]
    return out^


def kk_harmonic_source(
    mut w: KokoroWeights, ref f0: Tensor, steps: Int, ref noise: KokoroNoise,
) raises -> List[Float32]:
    """`SourceModuleHnNSF` on the pitch curve (`steps` values, each held for
    300 samples): the merged harmonic excitation, `[300 steps]`."""
    comptime HN = KK_HARMONICS
    comptime UP = KK_UPSAMPLE
    var n = steps * UP
    if len(noise.phase) != HN or len(noise.sines) != n * HN:
        raise Error("kokoro: the noise does not match the utterance's " + String(n) + " samples")
    var lin_w = w.at("decoder.generator.m_source.l_linear.weight")
    var lin_b = w.at("decoder.generator.m_source.l_linear.bias")
    var merged = List[Float32](length=n, fill=0.0)
    for i in range(n):
        merged[i] = w.tensors[lin_b].data[0]
    var rad = List[Float32](length=n, fill=0.0)
    var two_pi = Float32(2.0 * pi)
    for h in range(HN):
        # Instantaneous frequency in cycles per sample, modulo 1 ...
        for i in range(n):
            var f = f0.data[i // UP] * Float32(h + 1) / Float32(KK_SAMPLE_RATE)
            rad[i] = f - floor(f)
        if h > 0:
            rad[0] += noise.phase[h]
        # ... averaged down to one value per pitch step, accumulated there,
        # and interpolated back up: the phase of every sample.
        var down = _linear_resample(rad, n, steps, Float32(UP))
        var acc = Float32(0.0)
        for s in range(steps):
            acc += down[s]
            down[s] = acc * two_pi * Float32(UP)
        var phase = _linear_resample(down, steps, n, Float32(1.0) / Float32(UP))
        var wh = w.tensors[lin_w].data[h]
        for i in range(n):
            var voiced = f0.data[i // UP] > 10.0
            var amp = Float32(0.003) if voiced else Float32(0.1) / 3.0
            var wave = amp * noise.sines[i * HN + h]
            if voiced:
                wave += sin(phase[i]) * Float32(0.1)
            merged[i] += wh * wave
    for i in range(n):
        merged[i] = tanh(merged[i])
    return merged^


def _hann_periodic(n: Int) -> List[Float32]:
    var win = List[Float32](length=n, fill=0.0)
    for i in range(n):
        win[i] = Float32(0.5 - 0.5 * cos(2.0 * pi * Float64(i) / Float64(n)))
    return win^


def kk_stft(ref x: List[Float32], mut out: Tensor) -> Int:
    """`torch.stft(n_fft=20, hop=5, window=hann, center=True)` (reflect
    padding) as `out[frames, 22]`: 11 magnitudes then 11 phases. Returns the
    frames, `len(x) // 5 + 1`."""
    comptime N = KK_N_FFT
    comptime B = KK_BINS
    var n = len(x)
    var frames = n // KK_HOP + 1
    var half = N // 2
    var win = _hann_periodic(N)
    out.ensure(frames * 2 * B)
    var buf = List[Float32](length=N, fill=0.0)
    for m in range(frames):
        for j in range(N):
            var idx = m * KK_HOP - half + j
            if idx < 0:
                idx = -idx
            elif idx >= n:
                idx = 2 * (n - 1) - idx
            buf[j] = x[idx] * win[j]
        for k in range(B):
            var re = 0.0
            var im = 0.0
            for j in range(N):
                var ang = 2.0 * pi * Float64(k * j) / Float64(N)
                re += Float64(buf[j]) * cos(ang)
                im -= Float64(buf[j]) * sin(ang)
            out.data[m * 2 * B + k] = Float32(sqrt(re * re + im * im))
            out.data[m * 2 * B + B + k] = Float32(atan2(im, re))
    return frames


def kk_istft(ref post: Tensor, frames: Int) -> List[Float32]:
    """`torch.istft` of `exp(post[:, :11]) * e^(i sin(post[:, 11:]))` with the
    same window: `[5 (frames - 1)]` samples."""
    comptime N = KK_N_FFT
    comptime B = KK_BINS
    var half = N // 2
    var win = _hann_periodic(N)
    var total = KK_HOP * (frames - 1) + N
    var acc = List[Float64](length=total, fill=0.0)
    var env = List[Float64](length=total, fill=0.0)
    for m in range(frames):
        var base = m * 2 * B
        for j in range(N):
            # Inverse real FFT: DC, 2 x the middle bins, the Nyquist bin's real part.
            var v = 0.0
            for k in range(B):
                var mag = exp(Float64(post.data[base + k]))
                var ph = Float64(sin(post.data[base + B + k]))
                var ang = 2.0 * pi * Float64(k * j) / Float64(N)
                if k == 0:
                    v += mag * cos(ph)
                elif k == B - 1:
                    v += mag * cos(ph) * cos(ang)
                else:
                    v += 2.0 * mag * cos(ph + ang)
            v /= Float64(N)
            acc[m * KK_HOP + j] += v * Float64(win[j])
            env[m * KK_HOP + j] += Float64(win[j]) * Float64(win[j])
    var n_out = KK_HOP * (frames - 1)
    var out = List[Float32](length=n_out, fill=0.0)
    for i in range(n_out):
        out[i] = Float32(acc[half + i] / env[half + i])
    return out^


def kk_decoder_blocks(
    mut w: KokoroWeights, ref a: KokoroAcoustic, ref ref_s: List[Float32],
    mut d: KokoroDecoded,
) raises -> Tensor:
    """`Decoder.forward` up to the generator: fills `d.encode` / `d.decode` and
    returns what the generator reads, `[2 frames, 512]`."""
    comptime S = KK_STYLE
    var fr = a.frames
    d.frames = fr
    var style = Tensor.alloc(S)
    for i in range(S):
        style.data[i] = ref_s[i]
    var f0c = Tensor()
    var nc = Tensor()
    _ = kk_conv1d(w, "decoder.F0_conv", a.f0, 2 * fr, f0c, 2, 1)
    _ = kk_conv1d(w, "decoder.N_conv", a.n, 2 * fr, nc, 2, 1)
    var x = Tensor.alloc(fr * 514)
    for t in range(fr):
        for c in range(512):
            x.data[t * 514 + c] = a.asr.data[t * 512 + c]
        x.data[t * 514 + 512] = f0c.data[t]
        x.data[t * 514 + 513] = nc.data[t]
    _ = kk_adain_resblk(w, "decoder.encode", x, fr, style, d.encode)
    var asr_res = Tensor()
    _ = kk_conv1d(w, "decoder.asr_res.0", a.asr, fr, asr_res)
    var cur = Tensor.alloc(fr * 1024)
    for i in range(fr * 1024):
        cur.data[i] = d.encode.data[i]
    for b in range(4):
        var cat = Tensor.alloc(fr * 1090)
        for t in range(fr):
            for c in range(1024):
                cat.data[t * 1090 + c] = cur.data[t * 1024 + c]
            for c in range(64):
                cat.data[t * 1090 + 1024 + c] = asr_res.data[t * 64 + c]
            cat.data[t * 1090 + 1088] = f0c.data[t]
            cat.data[t * 1090 + 1089] = nc.data[t]
        var y = Tensor()
        var length = kk_adain_resblk(w, "decoder.decode." + String(b), cat, fr, style, y)
        var width = 1024 if b < 3 else 512
        cur = Tensor.alloc(length * width)
        for i in range(length * width):
            cur.data[i] = y.data[i]
        d.decode.append(y^)
    return cur^


def kk_generate(
    mut w: KokoroWeights, mut d: KokoroDecoded, var cur: Tensor,
    ref ref_s: List[Float32], har: Tensor, har_frames: Int,
) raises:
    """The generator after its source: `cur` (`[2 frames, 512]`) and the
    source's spectrum `har` (`[har_frames, 22]`) -> `d.ups`, `d.res`, `d.post`
    and `d.audio`."""
    comptime S = KK_STYLE
    var style = Tensor.alloc(S)
    for i in range(S):
        style.data[i] = ref_s[i]
    var gen = String("decoder.generator.")
    var channels = 512
    var length = 2 * d.frames
    for i in range(2):
        kk_leaky_relu(cur, length * channels, 0.1)
        var src = Tensor()
        var src_len: Int
        if i == 0:
            src_len = kk_conv1d(w, gen + "noise_convs.0", har, har_frames, src, 6, 3)
        else:
            src_len = kk_conv1d(w, gen + "noise_convs.1", har, har_frames, src)
        _adain_resblock1(w, gen + "noise_res." + String(i), src, src_len, style, 7 if i == 0 else 11)
        var up = Tensor()
        var up_len = kk_conv_transpose1d(w, gen + "ups." + String(i), cur, length, up, 10 if i == 0 else 6, 5 if i == 0 else 3)
        channels = channels // 2
        var tap = Tensor.alloc(up_len * channels)
        for k in range(up_len * channels):
            tap.data[k] = up.data[k]
        d.ups.append(tap^)
        var stage_len = up_len
        var xin: Tensor
        if i == 1:
            # ReflectionPad1d((1, 0)): one sample on the left, a copy of sample 1.
            stage_len = up_len + 1
            xin = Tensor.alloc(stage_len * channels)
            for c in range(channels):
                xin.data[c] = up.data[1 * channels + c]
            for k in range(up_len * channels):
                xin.data[channels + k] = up.data[k]
        else:
            xin = up^
        if src_len != stage_len:
            raise Error("kokoro: the source has " + String(src_len) + " steps at a stage of " + String(stage_len))
        for k in range(stage_len * channels):
            xin.data[k] += src.data[k]
        var mean = Tensor.alloc(stage_len * channels)
        var kernels: List[Int] = [3, 7, 11]
        for j in range(3):
            var branch = Tensor.alloc(stage_len * channels)
            for k in range(stage_len * channels):
                branch.data[k] = xin.data[k]
            _adain_resblock1(w, gen + "resblocks." + String(i * 3 + j), branch, stage_len, style, kernels[j])
            for k in range(stage_len * channels):
                mean.data[k] += branch.data[k]
        for k in range(stage_len * channels):
            mean.data[k] = mean.data[k] / 3.0
        cur = mean^
        length = stage_len
        if i == 0:
            var tap0 = Tensor.alloc(length * channels)
            for k in range(length * channels):
                tap0.data[k] = cur.data[k] if cur.data[k] >= 0.0 else cur.data[k] * Float32(0.1)
            d.res.append(tap0^)

    kk_leaky_relu(cur, length * channels, 0.01)
    var tap1 = Tensor.alloc(length * channels)
    for k in range(length * channels):
        tap1.data[k] = cur.data[k]
    d.res.append(tap1^)
    _ = kk_conv1d(w, gen + "conv_post", cur, length, d.post, 1, 3)
    d.audio = kk_istft(d.post, length)


def kk_decode(
    mut w: KokoroWeights, ref a: KokoroAcoustic, ref ref_s: List[Float32],
    ref noise: KokoroNoise,
) raises -> KokoroDecoded:
    """The decoder and the vocoder: `a` (per-frame encodings, pitch, energy)
    and the DECODER half of the style row -> the waveform (`d.audio`)."""
    var d = KokoroDecoded()
    var x = kk_decoder_blocks(w, a, ref_s, d)
    d.source = kk_harmonic_source(w, a.f0, 2 * a.frames, noise)
    var har = Tensor()
    var har_frames = kk_stft(d.source, har)
    kk_generate(w, d, x^, ref_s, har, har_frames)
    d.har = har^
    return d^
