# +--------------------------------------------------------------------------+ #
# | Kaldi filterbank features — the front end of the speaker model
# +--------------------------------------------------------------------------+ #
"""`kaldi_fbank(samples)`: Kaldi's `fbank` as WeSpeaker asks for it
(`wespeaker/cli/speaker.py`), and `subtract_mean` (the clip's mean removed
from every frame).

    var x = pcm_int16_range(wav.samples)             # Float32, in [-32768, 32767]
    var feats = kaldi_fbank(x)                       # [frames, 80], row-major
    subtract_mean(feats, 80)

THE RECIPE, per frame of 400 samples every 160 (16 kHz; a clip shorter than
400 samples has no frame; the samples after the last whole frame are dropped):

1. subtract the frame's own mean;
2. pre-emphasis 0.97, backwards: `f[j] -= 0.97 f[j-1]`, and `f[0] -= 0.97 f[0]`;
3. a Hamming window, `0.54 - 0.46 cos(2 pi j / 399)`;
4. zero-pad to 512, FFT, power spectrum;
5. 80 triangular filters, equally spaced on the mel scale `1127 ln(1 + f/700)`
   from 20 Hz to 8000 Hz, over the FFT bins 0..255 (Kaldi leaves the Nyquist
   bin out);
6. the natural log, with a floor of float32's epsilon.

No dither. Samples are in the int16 RANGE, as Kaldi takes them: the scale adds
a constant to every log energy, which `subtract_mean` removes, but the floor
of step 6 sits where Kaldi's sits only on that scale.

⚠ NOT `noeira.audio.mel` (Parakeet's front end): that one is centred, has a
Hann window, Slaney's mel scale and a per-feature normalisation. Two models,
two recipes; this one is gated against Kaldi's own code (`kaldi-native-fbank`,
`tests/audio/test_speaker_reference.mojo`).
"""

from std.math import cos, sin, log, pi


comptime FB_SR: Int = 16000
comptime FB_WIN: Int = 400
comptime FB_HOP: Int = 160
comptime FB_FFT: Int = 512
comptime FB_MELS: Int = 80
comptime FB_LOW_HZ: Float64 = 20.0
comptime FB_PREEMPH: Float64 = 0.97
comptime FB_FLOOR: Float64 = 1.1920928955078125e-07
"""float32's epsilon: Kaldi's floor under the log."""


def pcm_int16_range(ref samples: List[Int16]) -> List[Float32]:
    """16-bit samples as the floats Kaldi takes: the same numbers."""
    var out = List[Float32](capacity=len(samples))
    for s in samples:
        out.append(Float32(Int(s)))
    return out^


def kaldi_mel(hz: Float64) -> Float64:
    return 1127.0 * log(1.0 + hz / 700.0)


def _fft512(mut re: List[Float64], mut im: List[Float64], ref cs: List[Float64], ref sn: List[Float64]):
    """In-place radix-2 FFT of 512 points (bit-reversal, then butterflies)."""
    var n = FB_FFT
    var j = 0
    for i in range(1, n):
        var bit = n >> 1
        while j & bit != 0:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            var tr = re[i]
            re[i] = re[j]
            re[j] = tr
            var ti = im[i]
            im[i] = im[j]
            im[j] = ti
    var size = 2
    while size <= n:
        var half = size // 2
        var step = n // size
        var i = 0
        while i < n:
            for k in range(half):
                var wr = cs[k * step]
                var wi = -sn[k * step]
                var a = i + k
                var b = a + half
                var tr = re[b] * wr - im[b] * wi
                var ti = re[b] * wi + im[b] * wr
                re[b] = re[a] - tr
                im[b] = im[a] - ti
                re[a] += tr
                im[a] += ti
            i += size
        size *= 2


def kaldi_fbank(ref samples: List[Float32], preemphasis: Float64 = FB_PREEMPH, hamming: Bool = True) -> List[Float32]:
    """`[frames, FB_MELS]`, row-major; `frames = 1 + (n - 400) // 160`, 0 for a
    clip shorter than a frame. `preemphasis` and `hamming` are the recipe's;
    the gate changes them to show it can fail."""
    var n = len(samples)
    var out = List[Float32]()
    if n < FB_WIN:
        return out^
    var frames = 1 + (n - FB_WIN) // FB_HOP
    # the tables
    var window = List[Float64](capacity=FB_WIN)
    for j in range(FB_WIN):
        var a = 2.0 * pi * Float64(j) / Float64(FB_WIN - 1)
        window.append(0.54 - 0.46 * cos(a) if hamming else 0.5 - 0.5 * cos(a))
    var cs = List[Float64](capacity=FB_FFT)
    var sn = List[Float64](capacity=FB_FFT)
    for k in range(FB_FFT):
        cs.append(cos(2.0 * pi * Float64(k) / Float64(FB_FFT)))
        sn.append(sin(2.0 * pi * Float64(k) / Float64(FB_FFT)))
    var bins = FB_FFT // 2    # 256: the Nyquist bin is not used
    var bin_hz = Float64(FB_SR) / Float64(FB_FFT)
    var mel_low = kaldi_mel(FB_LOW_HZ)
    var mel_high = kaldi_mel(Float64(FB_SR) / 2.0)
    var delta = (mel_high - mel_low) / Float64(FB_MELS + 1)
    var filt = List[Float64](length=FB_MELS * bins, fill=0.0)
    var first = List[Int](length=FB_MELS, fill=bins)
    var last = List[Int](length=FB_MELS, fill=0)
    for m in range(FB_MELS):
        var left = mel_low + Float64(m) * delta
        var centre = left + delta
        var right = centre + delta
        for i in range(bins):
            var mel = kaldi_mel(bin_hz * Float64(i))
            if mel > left and mel < right:
                var wgt = (mel - left) / (centre - left) if mel <= centre else (right - mel) / (right - centre)
                filt[m * bins + i] = wgt
                if i < first[m]:
                    first[m] = i
                last[m] = i + 1
    out = List[Float32](length=frames * FB_MELS, fill=0.0)
    var re = List[Float64](length=FB_FFT, fill=0.0)
    var im = List[Float64](length=FB_FFT, fill=0.0)
    var f = List[Float64](length=FB_WIN, fill=0.0)
    for t in range(frames):
        var start = t * FB_HOP
        var mean = 0.0
        for j in range(FB_WIN):
            f[j] = Float64(samples[start + j])
            mean += f[j]
        mean /= Float64(FB_WIN)
        for j in range(FB_WIN):
            f[j] -= mean
        var j = FB_WIN - 1
        while j > 0:
            f[j] -= preemphasis * f[j - 1]
            j -= 1
        f[0] -= preemphasis * f[0]
        for k in range(FB_FFT):
            re[k] = f[k] * window[k] if k < FB_WIN else 0.0
            im[k] = 0.0
        _fft512(re, im, cs, sn)
        for k in range(bins):
            re[k] = re[k] * re[k] + im[k] * im[k]
        for m in range(FB_MELS):
            var e = 0.0
            for k in range(first[m], last[m]):
                e += filt[m * bins + k] * re[k]
            out[t * FB_MELS + m] = Float32(log(e if e > FB_FLOOR else FB_FLOOR))
    return out^


def subtract_mean(mut feats: List[Float32], dims: Int = FB_MELS):
    """Each column's mean over the clip, removed (WeSpeaker's `cmn`)."""
    var frames = len(feats) // dims
    if frames == 0:
        return
    for d in range(dims):
        var m = 0.0
        for t in range(frames):
            m += Float64(feats[t * dims + d])
        m /= Float64(frames)
        for t in range(frames):
            feats[t * dims + d] = Float32(Float64(feats[t * dims + d]) - m)
