# +--------------------------------------------------------------------------+ #
# | Log-mel front end — waveform to normalized log-mel frames
# +--------------------------------------------------------------------------+ #
"""The speech front end Parakeet (FastConformer) consumes, on the CPU.

    var fe = LogMel[]()                                   # 16 kHz, 512 / 400 / 160, 128 mels
    var feats = List[Float32](length=801 * 128, fill=0.0)
    var frames = fe.extract[801](audio, feats)            # valid frames; rows beyond are 0

Reference: `transformers`' `ParakeetFeatureExtractor` (itself a port of NeMo's
`AudioToMelSpectrogramPreprocessor`), gated by
`tests/audio/test_log_mel_reference.mojo`:

    y[0] = x[0];  y[t] = x[t] - 0.97 x[t-1]               pre-emphasis
    STFT: n_fft 512, hop 160, hann(400, periodic=False) zero-padded to 512 and
          centred, the signal zero-padded by 256 on both sides (`center=True`,
          `pad_mode="constant"`)
    power = re^2 + im^2  ->  slaney mel filterbank (librosa defaults)  ->  log(. + 2^-24)
    per mel bin, over the VALID frames: (v - mean) / (std + 1e-5), std with n - 1

## Fixed maximum, runtime length

`extract[MAX_FRAMES]` is the compile-time pattern of docs/NN_MODEL_PORTS_PLAN.md
(§9.3): the output is always `[MAX_FRAMES, N_MELS]`, the return value is the
number of valid frames, and every row beyond it is zero — which is exactly what
the reference produces for a clip padded inside a longer batch, mask included.

⚠ A clip of `n` samples has `n // HOP` valid frames, not `1 + n // HOP`. The
STFT does produce that extra frame; the extractor masks it, and the encoder's
length arithmetic starts from the masked count.

Arithmetic is float64 throughout and rounded once at the end. The reference
works in float32, so agreement is a bound (stated in the gate), not equality.
"""

from std.math import cos, sin, log, exp, sqrt, pi


comptime LOG_ZERO_GUARD: Float64 = 5.9604644775390625e-08
"""2^-24, added to the mel power before the log."""
comptime NORM_EPS: Float64 = 1e-5


def _hz_to_mel(f: Float64) -> Float64:
    """Slaney's scale (librosa `htk=False`): linear below 1 kHz, log above."""
    var f_sp = 200.0 / 3.0
    if f >= 1000.0:
        return 1000.0 / f_sp + log(f / 1000.0) / (log(6.4) / 27.0)
    return f / f_sp


def _mel_to_hz(m: Float64) -> Float64:
    var f_sp = 200.0 / 3.0
    var min_log_mel = 1000.0 / f_sp
    if m >= min_log_mel:
        return 1000.0 * exp((log(6.4) / 27.0) * (m - min_log_mel))
    return f_sp * m


def slaney_mel_filterbank(sr: Int, n_fft: Int, n_mels: Int) -> List[Float64]:
    """`librosa.filters.mel(sr, n_fft, n_mels, fmin=0, fmax=sr/2, norm="slaney")`
    as `[n_mels, n_fft // 2 + 1]`, row-major: triangles on Slaney's mel scale,
    each scaled by `2 / (its width in Hz)` so every filter has the same area.

    The values are librosa's float32 weights, bit for bit (504 of 504 non-zero
    weights at the defaults). ⚠ That takes TWO roundings: librosa stores the
    triangle in a float32 array and scales it in place, so the weight is
    `f32(f64(f32(triangle)) * scale)`. Rounding once leaves a third of them one
    ulp off."""
    var n_bins = n_fft // 2 + 1
    var m_lo = _hz_to_mel(0.0)
    var m_step = (_hz_to_mel(Float64(sr) / 2.0) - m_lo) / Float64(n_mels + 1)
    var mel_f = List[Float64](length=n_mels + 2, fill=0.0)
    for i in range(n_mels + 2):
        mel_f[i] = _mel_to_hz(Float64(i) * m_step + m_lo)
    # The grid's last point IS fmax (numpy's `linspace` pins its endpoint).
    # Left to the arithmetic it can land an ulp above, and the top filter then
    # gives the Nyquist bin a weight of 1e-13 where the reference has zero.
    mel_f[n_mels + 1] = Float64(sr) / 2.0
    var bin_hz = 1.0 / (Float64(n_fft) * (1.0 / Float64(sr)))
    var w = List[Float64](length=n_mels * n_bins, fill=0.0)
    for i in range(n_mels):
        var rise = mel_f[i + 1] - mel_f[i]
        var fall = mel_f[i + 2] - mel_f[i + 1]
        var enorm = 2.0 / (mel_f[i + 2] - mel_f[i])
        for k in range(n_bins):
            var fk = Float64(k) * bin_hz
            var lower = (fk - mel_f[i]) / rise
            var upper = (mel_f[i + 2] - fk) / fall
            var v = lower if lower < upper else upper
            if v > 0.0:
                w[i * n_bins + k] = Float64(Float32(Float64(Float32(v)) * enorm))
    return w^


def hann_window(n: Int, periodic: Bool) -> List[Float64]:
    """`torch.hann_window(n, periodic=periodic)`: its float32 values.

    torch builds the window in float32 — `arange(n) * f32(2 pi / d)`, cosine,
    `* -0.5 + 0.5` — and the float32 ANGLE is what matters: computed in float64
    and rounded at the end, only a third of the 400 values match and the rest
    are up to 3 ulp off. With the float32 angle, 391 of 400 are identical and
    the others are 1 ulp off (torch's vectorized cosine)."""
    var denom = Float64(n) if periodic else Float64(n - 1)
    var scale = Float32(2.0 * pi / denom)
    var w = List[Float64](length=n, fill=0.0)
    for i in range(n):
        var angle = Float32(i) * scale
        var c = Float32(cos(Float64(angle)))
        var v: Float32 = c * -0.5
        v += 0.5
        w[i] = Float64(v)
    return w^


struct LogMel[
    N_FFT: Int = 512,
    WIN: Int = 400,
    HOP: Int = 160,
    N_MELS: Int = 128,
    SR: Int = 16000,
](Movable):
    """Waveform -> normalized log-mel frames, at a fixed maximum frame count.

    The tables (`window`, `filt`, `preemphasis`) are plain fields: a model whose
    front end differs in a constant sets it after construction, and the gate
    perturbs them to show it can fail.
    """

    comptime BINS: Int = Self.N_FFT // 2 + 1

    var preemphasis: Float64
    var window: List[Float64]
    """`[N_FFT]`: hann(WIN, periodic=False), zero-padded and centred."""
    var filt: List[Float64]
    """`[N_MELS, BINS]`, the float32 values the reference multiplies by."""
    var _lo: List[Int]
    var _hi: List[Int]
    """Per mel bin, the `[lo, hi)` range of FFT bins its triangle covers."""
    var _cos: List[Float64]
    var _sin: List[Float64]
    var _rev: List[Int]
    var _re: List[Float64]
    var _im: List[Float64]
    var _y: List[Float64]
    var log_mel: List[Float64]
    """`[frames, N_MELS]` of the last `extract`: the log-mel BEFORE the
    normalization. `exp(.) - 2^-24` is the mel power, which is what tells a
    bin with signal from a near-silent one (the gate bounds them separately)."""

    def __init__(out self):
        comptime assert Self.N_FFT & (Self.N_FFT - 1) == 0, "N_FFT must be a power of two"
        comptime assert Self.WIN <= Self.N_FFT, "WIN must fit in N_FFT"
        self.preemphasis = 0.97
        self.window = List[Float64](length=Self.N_FFT, fill=0.0)
        var h = hann_window(Self.WIN, False)
        var left = (Self.N_FFT - Self.WIN) // 2
        for i in range(Self.WIN):
            self.window[left + i] = h[i]
        self.filt = slaney_mel_filterbank(Self.SR, Self.N_FFT, Self.N_MELS)
        self._lo = List[Int](length=Self.N_MELS, fill=0)
        self._hi = List[Int](length=Self.N_MELS, fill=0)
        self._cos = List[Float64](length=Self.N_FFT // 2, fill=0.0)
        self._sin = List[Float64](length=Self.N_FFT // 2, fill=0.0)
        for k in range(Self.N_FFT // 2):
            var a = 2.0 * pi * Float64(k) / Float64(Self.N_FFT)
            self._cos[k] = cos(a)
            self._sin[k] = sin(a)
        self._rev = List[Int](length=Self.N_FFT, fill=0)
        var bits = 0
        while (1 << bits) < Self.N_FFT:
            bits += 1
        for i in range(Self.N_FFT):
            var r = 0
            for b in range(bits):
                if (i >> b) & 1 == 1:
                    r |= 1 << (bits - 1 - b)
            self._rev[i] = r
        self._re = List[Float64](length=Self.N_FFT, fill=0.0)
        self._im = List[Float64](length=Self.N_FFT, fill=0.0)
        self._y = List[Float64]()
        self.log_mel = List[Float64]()
        self.refresh_filter_ranges()

    def refresh_filter_ranges(mut self):
        """Recompute each filter's non-zero bin range. Call after editing `filt`."""
        for m in range(Self.N_MELS):
            var lo = Self.BINS
            var hi = 0
            for k in range(Self.BINS):
                if self.filt[m * Self.BINS + k] != 0.0:
                    if k < lo:
                        lo = k
                    hi = k + 1
            self._lo[m] = lo if lo < hi else 0
            self._hi[m] = hi

    def _fft(mut self):
        """In-place radix-2 forward FFT of (`_re`, `_im`), already bit-reversed."""
        var size = 2
        while size <= Self.N_FFT:
            var half = size // 2
            var step = Self.N_FFT // size
            var i = 0
            while i < Self.N_FFT:
                for k in range(half):
                    var wr = self._cos[k * step]
                    var wi = -self._sin[k * step]
                    var a = i + k
                    var b = a + half
                    var tr = self._re[b] * wr - self._im[b] * wi
                    var ti = self._re[b] * wi + self._im[b] * wr
                    self._re[b] = self._re[a] - tr
                    self._im[b] = self._im[a] - ti
                    self._re[a] += tr
                    self._im[a] += ti
                i += size
            size *= 2

    def extract[MAX_FRAMES: Int](
        mut self, ref audio: List[Float32], mut out: List[Float32]
    ) raises -> Int:
        """Fill `out` (`[MAX_FRAMES, N_MELS]`, row-major) from mono `audio` at
        `SR`, and return the number of valid frames (`len(audio) // HOP`). Rows
        beyond it are zero."""
        var n = len(audio)
        var frames = n // Self.HOP
        if len(out) != MAX_FRAMES * Self.N_MELS:
            raise Error("LogMel.extract: out must hold MAX_FRAMES * N_MELS values")
        if frames > MAX_FRAMES:
            raise Error(
                "LogMel.extract: the clip has " + String(frames)
                + " frames, the maximum compiled is " + String(MAX_FRAMES)
            )
        if frames < 2:
            raise Error("LogMel.extract: the clip is shorter than two frames")

        if len(self._y) < n:
            self._y = List[Float64](length=n, fill=0.0)
        self._y[0] = Float64(audio[0])
        for t in range(1, n):
            self._y[t] = Float64(audio[t]) - self.preemphasis * Float64(audio[t - 1])
        if len(self.log_mel) < frames * Self.N_MELS:
            self.log_mel = List[Float64](length=frames * Self.N_MELS, fill=0.0)

        var half = Self.N_FFT // 2
        for t in range(frames):
            var start = t * Self.HOP - half
            for j in range(Self.N_FFT):
                var idx = start + j
                var v = 0.0
                if idx >= 0 and idx < n:
                    v = self._y[idx] * self.window[j]
                self._re[self._rev[j]] = v
                self._im[self._rev[j]] = 0.0
            self._fft()
            for k in range(Self.BINS):
                self._re[k] = self._re[k] * self._re[k] + self._im[k] * self._im[k]
            for m in range(Self.N_MELS):
                var acc = 0.0
                for k in range(self._lo[m], self._hi[m]):
                    acc += self.filt[m * Self.BINS + k] * self._re[k]
                self.log_mel[t * Self.N_MELS + m] = log(acc + LOG_ZERO_GUARD)

        for m in range(Self.N_MELS):
            var mean = 0.0
            for t in range(frames):
                mean += self.log_mel[t * Self.N_MELS + m]
            mean /= Float64(frames)
            var sq = 0.0
            for t in range(frames):
                var d = self.log_mel[t * Self.N_MELS + m] - mean
                sq += d * d
            var inv = 1.0 / (sqrt(sq / Float64(frames - 1)) + NORM_EPS)
            for t in range(frames):
                out[t * Self.N_MELS + m] = Float32(
                    (self.log_mel[t * Self.N_MELS + m] - mean) * inv
                )
        for i in range(frames * Self.N_MELS, MAX_FRAMES * Self.N_MELS):
            out[i] = 0.0
        return frames
