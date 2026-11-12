# +--------------------------------------------------------------------------+ #
# | LFM3-Audio log-mel front end — waveform to the audio encoder's features
# +--------------------------------------------------------------------------+ #
"""The front end LFM3-Omni / d1-Omni's audio encoder consumes, on the CPU
(`Lfm3OmniFeatureExtractor`, transformers' LFM3 code vendored in the
checkpoint; docs/NN_MODEL_PORTS_PLAN.md §12.6).

    var fe = Lfm3LogMel()                          # 16 kHz, n_fft 400, hop 160, 128 mels
    var feats = fe.extract(audio)                  # [128, F] row-major: the model's `input_features[0]`
    var f = fe.frames(len(audio))                  # F; the encoder makes ceil(F / 8) tokens

    the clip right-padded with zeros to whole frames: n -> n + (-(n - 400) mod 160)
    STFT: n_fft 400, hop 160, hann(400, PERIODIC), NOT centred (frame t = samples
          [160 t, 160 t + 400)), power = re^2 + im^2
    mel = filters^T power, transformers' `mel_filter_bank(201, 128, 0, 8000, 16000,
          norm="slaney", mel_scale="slaney")` (float64, rounded to float32 once)
    log10(mel), clamped below at -12, then (x + 4) / 4

NOT Parakeet's front end (`mel.mojo`): no pre-emphasis, no centring, a periodic
window, log10 with a floor, and a fixed affine normalisation instead of a
per-bin one. n_fft 400 is not a power of two: the transform is a mixed-radix
Cooley-Tukey FFT over its factors (400 = 2^4 5^2, `_fft`), ~7 200 complex
multiply-adds a frame instead of a direct DFT's 80 000 (100 frames a second of
audio; the direct DFT was 37 ms a second of audio on the Orin's CPU).

The low mel filters cover no FFT bin at this resolution (all-zero rows, as in
the reference: transformers warns about it); their mel power is exactly 0, the
log -inf, the clamp -12, the feature -2.

Arithmetic in float64, rounded once at the end; the reference's STFT is
float32, so agreement is a bound (the gate states it).
"""

from std.math import cos, sin, log, exp, pi

from .mel import hann_window


def _slaney_hz_to_mel(f: Float64) -> Float64:
    """transformers' `hertz_to_mel(mel_scale="slaney")`."""
    if f >= 1000.0:
        return 15.0 + log(f / 1000.0) * (27.0 / log(6.4))
    return 3.0 * f / 200.0


def _slaney_mel_to_hz(m: Float64) -> Float64:
    if m >= 15.0:
        return 1000.0 * exp((log(6.4) / 27.0) * (m - 15.0))
    return 200.0 * m / 3.0


def hf_mel_filter_bank(n_bins: Int, n_mels: Int, sr: Int) -> List[Float64]:
    """transformers' `mel_filter_bank(n_bins, n_mels, 0, sr / 2, sr,
    norm="slaney", mel_scale="slaney")` as `[n_mels, n_bins]` row-major, each
    weight its float32 value (the reference casts the float64 bank once).

    Not librosa's (`mel.slaney_mel_filterbank`): the FFT bins are
    `linspace(0, sr // 2, n_bins)` and the triangle is scaled in float64."""
    var m_lo = _slaney_hz_to_mel(0.0)
    var m_hi = _slaney_hz_to_mel(Float64(sr // 2))
    var pts = List[Float64](length=n_mels + 2, fill=0.0)
    for i in range(n_mels + 2):
        # numpy's linspace: start + i * step, the endpoint pinned
        var mel = m_hi if i == n_mels + 1 else m_lo + Float64(i) * ((m_hi - m_lo) / Float64(n_mels + 1))
        pts[i] = _slaney_mel_to_hz(mel)
    var w = List[Float64](length=n_mels * n_bins, fill=0.0)
    for i in range(n_mels):
        var enorm = 2.0 / (pts[i + 2] - pts[i])
        for k in range(n_bins):
            var fk = Float64(sr // 2) if k == n_bins - 1 else Float64(k) * (Float64(sr // 2) / Float64(n_bins - 1))
            var down = -(pts[i] - fk) / (pts[i + 1] - pts[i])
            var up = (pts[i + 2] - fk) / (pts[i + 2] - pts[i + 1])
            var v = down if down < up else up
            if v > 0.0:
                w[i * n_bins + k] = Float64(Float32(v * enorm))
    return w^


struct Lfm3LogMel(Movable):
    """Waveform -> LFM3-Audio's log-mel features (module note). The tables are
    plain fields so the gate can perturb them and see it fail."""

    var n_fft: Int
    var hop: Int
    var n_mels: Int
    var sr: Int
    var log_floor: Float64
    var offset: Float64
    var divisor: Float64
    var stride: Int
    """Frames per encoder token (the three stride-2 convolutions)."""
    var window: List[Float64]
    """`[n_fft]`: torch's float32 `hann_window(400)` (periodic)."""
    var filt: List[Float64]
    """`[n_mels, bins]`."""
    var _lo: List[Int]
    var _hi: List[Int]
    var _cos: List[Float64]
    var _sin: List[Float64]
    """`cos / sin(2 pi j / n_fft)`, j < n_fft: the FFT's twiddles."""
    var _factors: List[Int]
    """`n_fft`'s prime factors, smallest first (the FFT's radices)."""

    def __init__(out self):
        self.n_fft = 400
        self.hop = 160
        self.n_mels = 128
        self.sr = 16000
        self.log_floor = -12.0
        self.offset = 4.0
        self.divisor = 4.0
        self.stride = 8
        self.window = hann_window(self.n_fft, True)
        self.filt = hf_mel_filter_bank(self.n_fft // 2 + 1, self.n_mels, self.sr)
        self._lo = List[Int]()
        self._hi = List[Int]()
        self._cos = List[Float64](length=self.n_fft, fill=0.0)
        self._sin = List[Float64](length=self.n_fft, fill=0.0)
        for j in range(self.n_fft):
            var a = 2.0 * pi * Float64(j) / Float64(self.n_fft)
            self._cos[j] = cos(a)
            self._sin[j] = sin(a)
        self._factors = List[Int]()
        var rest = self.n_fft
        var p = 2
        while rest > 1:
            while rest % p == 0:
                self._factors.append(p)
                rest //= p
            p += 1
        self.refresh_filter_ranges()

    def bins(self) -> Int:
        return self.n_fft // 2 + 1

    def refresh_filter_ranges(mut self):
        """Each filter's non-zero bin range. Call after editing `filt`."""
        var nb = self.bins()
        self._lo = List[Int](length=self.n_mels, fill=0)
        self._hi = List[Int](length=self.n_mels, fill=0)
        for m in range(self.n_mels):
            var lo = nb
            var hi = 0
            for k in range(nb):
                if self.filt[m * nb + k] != 0.0:
                    lo = min(lo, k)
                    hi = k + 1
            self._lo[m] = lo if lo < hi else 0
            self._hi[m] = hi

    def padded(self, n: Int) -> Int:
        """The clip's length after the right padding to whole frames."""
        if n < self.n_fft:
            return self.n_fft
        var r = (n - self.n_fft) % self.hop
        return n + (self.hop - r if r != 0 else 0)

    def frames(self, n: Int) -> Int:
        """Mel frames of a clip of `n` samples."""
        return 1 + (self.padded(n) - self.n_fft) // self.hop

    def tokens(self, n: Int) -> Int:
        """`<|audio|>` tokens of a clip of `n` samples: ceil(frames / 8)."""
        var f = self.frames(n)
        return (f + self.stride - 1) // self.stride

    def _fft(
        self, ref xr: List[Float64], ref xi: List[Float64], off: Int, stride: Int, n: Int,
        mut yr: List[Float64], mut yi: List[Float64], yoff: Int, level: Int,
        mut tr: List[Float64], mut ti: List[Float64],
    ):
        """`y[yoff : yoff + n]` = the forward DFT (`exp(-2 pi i k j / n)`) of
        `x[off], x[off + stride], ...` (n values), decimation in time over
        `_factors[level:]`: p sub-transforms of n / p points (every p-th value),
        then the butterflies `Y[k + q m] = sum_r w_n^(r (k + q m)) Y_r[k]`.
        `tr` / `ti`: scratch of `n_fft`."""
        if n == 1:
            yr[yoff] = xr[off]
            yi[yoff] = xi[off]
            return
        var p = self._factors[level]
        var m = n // p
        for r in range(p):
            self._fft(xr, xi, off + r * stride, stride * p, m, yr, yi, yoff + r * m, level + 1, tr, ti)
        var step = self.n_fft // n  # w_n^e = w_N^(e N / n)
        for k in range(m):
            for q in range(p):
                var kk = k + q * m
                var sr = 0.0
                var si = 0.0
                for r in range(p):
                    var e = (r * kk) % n * step
                    var wr = self._cos[e]
                    var wi = -self._sin[e]
                    var ar = yr[yoff + r * m + k]
                    var ai = yi[yoff + r * m + k]
                    sr += ar * wr - ai * wi
                    si += ar * wi + ai * wr
                tr[q] = sr
                ti[q] = si
            for q in range(p):
                yr[yoff + k + q * m] = tr[q]
                yi[yoff + k + q * m] = ti[q]

    def extract(self, ref audio: List[Float32]) raises -> List[Float32]:
        """`[n_mels, frames]` row-major (mel bin major, the encoder's layout)."""
        var n = len(audio)
        if n == 0:
            raise Error("Lfm3LogMel.extract: an empty clip")
        var f = self.frames(n)
        var nb = self.bins()
        var nf = self.n_fft
        var out = List[Float32](length=self.n_mels * f, fill=0.0)
        var x = List[Float64](length=nf, fill=0.0)
        var xi = List[Float64](length=nf, fill=0.0)
        var yr = List[Float64](length=nf, fill=0.0)
        var yi = List[Float64](length=nf, fill=0.0)
        var tr = List[Float64](length=nf, fill=0.0)
        var ti = List[Float64](length=nf, fill=0.0)
        var pw = List[Float64](length=nb, fill=0.0)
        for t in range(f):
            var s0 = t * self.hop
            for j in range(nf):
                var i = s0 + j
                x[j] = Float64(audio[i]) * self.window[j] if i < n else 0.0
            self._fft(x, xi, 0, 1, nf, yr, yi, 0, 0, tr, ti)
            for k in range(nb):
                pw[k] = yr[k] * yr[k] + yi[k] * yi[k]
            for m in range(self.n_mels):
                var acc = 0.0
                for k in range(self._lo[m], self._hi[m]):
                    acc += self.filt[m * nb + k] * pw[k]
                var lg = self.log_floor
                if acc > 0.0:
                    lg = max(log(acc) / log(10.0), self.log_floor)
                out[m * f + t] = Float32((lg + self.offset) / self.divisor)
        return out^
