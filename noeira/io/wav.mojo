# +--------------------------------------------------------------------------+ #
# | WAV — 16-bit PCM, read and written
# +--------------------------------------------------------------------------+ #
"""RIFF/WAVE, 16-bit integer PCM: what speech APIs take and return.

    var a = load_wav("hello.wav")          # WavAudio
    print(a.sample_rate, a.channels, a.duration_s())
    var mono16k = a.to_mono().resample(16000)
    save_wav("out.wav", mono16k)

Speech-to-text endpoints accept a WAV upload, and text-to-speech ones return
one (`noeira/ai/speech.mojo`), so this is the audio container the AI layer
speaks. It is deliberately the one format: 16-bit little-endian PCM, any
channel count, any rate.

## Scope

Read: `fmt ` tag 1 (PCM) at 16 bits, and tag 0xFFFE (WAVE_FORMAT_EXTENSIBLE)
when its sub-format is PCM at 16 bits — macOS tools write that header for
anything past stereo. Unknown chunks (`LIST`, `fact`, …) are skipped.
Not supported: 8/24/32-bit PCM, float PCM, compressed tags.

⚠ A STREAMED WAV CAN CARRY A BOGUS `data` SIZE. A TTS server that writes the
header before it knows the length puts 0 or 0xFFFFFFFF there. The reader
clamps the chunk to the bytes actually present instead of raising, since the
samples are all there and the header is the part that lied.
"""

from noeira.io.fileio import read_file_bytes, write_file_atomic


struct WavAudio(Copyable, Movable):
    """Interleaved 16-bit samples: frame `f`, channel `c` is
    `samples[f * channels + c]`."""

    var sample_rate: Int
    var channels: Int
    var samples: List[Int16]

    def __init__(out self, sample_rate: Int, channels: Int, var samples: List[Int16]):
        self.sample_rate = sample_rate
        self.channels = channels
        self.samples = samples^

    def frames(self) -> Int:
        return len(self.samples) // self.channels if self.channels > 0 else 0

    def duration_s(self) -> Float64:
        if self.sample_rate <= 0:
            return 0.0
        return Float64(self.frames()) / Float64(self.sample_rate)

    def to_mono(self) -> WavAudio:
        """Average the channels. A mono input comes back as a copy."""
        if self.channels <= 1:
            return self.copy()
        var out = List[Int16](capacity=self.frames())
        for f in range(self.frames()):
            var acc = 0
            for c in range(self.channels):
                acc += Int(self.samples[f * self.channels + c])
            out.append(Int16(acc // self.channels))
        return WavAudio(self.sample_rate, 1, out^)

    def resample(self, rate: Int) raises -> WavAudio:
        """Linear-interpolation resample.

        ⚠ NO ANTI-ALIAS FILTER. Good enough to hand a 48 kHz microphone
        capture to a speech model at 16 kHz — every STT model resamples
        internally anyway and speech has little energy above 8 kHz — and not
        good enough for music.
        """
        if rate <= 0:
            raise Error("wav: resample to a non-positive rate")
        if rate == self.sample_rate or self.frames() == 0:
            var same = self.copy()
            same.sample_rate = rate
            return same^
        var n_in = self.frames()
        var n_out = Int(Float64(n_in) * Float64(rate) / Float64(self.sample_rate))
        var step = Float64(self.sample_rate) / Float64(rate)
        var out = List[Int16](capacity=n_out * self.channels)
        for i in range(n_out):
            var x = Float64(i) * step
            var i0 = Int(x)
            var i1 = i0 + 1 if i0 + 1 < n_in else n_in - 1
            var t = x - Float64(i0)
            for c in range(self.channels):
                var a = Float64(self.samples[i0 * self.channels + c])
                var b = Float64(self.samples[i1 * self.channels + c])
                out.append(Int16(Int(a + (b - a) * t)))
        return WavAudio(rate, self.channels, out^)

    def to_float32(self) -> List[Float32]:
        """Samples in [-1, 1), interleaved — the form a model consumes."""
        var out = List[Float32](capacity=len(self.samples))
        for i in range(len(self.samples)):
            out.append(Float32(self.samples[i]) / 32768.0)
        return out^


def wav_from_float32(
    ref x: List[Float32], sample_rate: Int, channels: Int = 1
) -> WavAudio:
    """Clamp to [-1, 1] and quantise to 16 bits."""
    var out = List[Int16](capacity=len(x))
    for i in range(len(x)):
        var v = x[i]
        if v > 1.0:
            v = 1.0
        elif v < -1.0:
            v = -1.0
        out.append(Int16(Int(v * 32767.0)))
    return WavAudio(sample_rate, channels, out^)


# ═══════════════════════════════════════════════════════════════════════════
# Codec
# ═══════════════════════════════════════════════════════════════════════════


def _le16(ref b: List[UInt8], off: Int) -> Int:
    return Int(b[off]) | (Int(b[off + 1]) << 8)


def _le32(ref b: List[UInt8], off: Int) -> Int:
    return (
        Int(b[off])
        | (Int(b[off + 1]) << 8)
        | (Int(b[off + 2]) << 16)
        | (Int(b[off + 3]) << 24)
    )


def _tag(ref b: List[UInt8], off: Int) -> String:
    var s = String("")
    for i in range(4):
        s += chr(Int(b[off + i]))
    return s^


def _put16(mut out: List[UInt8], v: Int):
    out.append(UInt8(v & 0xFF))
    out.append(UInt8((v >> 8) & 0xFF))


def _put32(mut out: List[UInt8], v: Int):
    out.append(UInt8(v & 0xFF))
    out.append(UInt8((v >> 8) & 0xFF))
    out.append(UInt8((v >> 16) & 0xFF))
    out.append(UInt8((v >> 24) & 0xFF))


def _put_tag(mut out: List[UInt8], tag: StaticString):
    var t = tag.as_bytes()
    for i in range(4):
        out.append(t[i])


def decode_wav(ref data: List[UInt8]) raises -> WavAudio:
    if len(data) < 12 or _tag(data, 0) != "RIFF" or _tag(data, 8) != "WAVE":
        raise Error("wav: not a RIFF/WAVE stream")
    var pos = 12
    var rate = 0
    var channels = 0
    var have_fmt = False
    while pos + 8 <= len(data):
        var tag = _tag(data, pos)
        var size = _le32(data, pos + 4)
        var body = pos + 8
        if tag == "fmt ":
            if size < 16 or body + 16 > len(data):
                raise Error("wav: truncated fmt chunk")
            var fmt_tag = _le16(data, body)
            channels = _le16(data, body + 2)
            rate = _le32(data, body + 4)
            var bits = _le16(data, body + 14)
            if fmt_tag == 0xFFFE and size >= 40:
                # WAVE_FORMAT_EXTENSIBLE: the real tag is the first two bytes
                # of the sub-format GUID.
                fmt_tag = _le16(data, body + 24)
            if fmt_tag != 1 or bits != 16:
                raise Error(
                    "wav: only 16-bit integer PCM is supported (format tag "
                    + String(fmt_tag) + ", " + String(bits) + " bits)"
                )
            if channels <= 0:
                raise Error("wav: zero channels")
            have_fmt = True
        elif tag == "data":
            if not have_fmt:
                raise Error("wav: data chunk before fmt chunk")
            var avail = len(data) - body
            var n = size if size > 0 and size <= avail else avail
            n -= n % (2 * channels)
            var samples = List[Int16](capacity=n // 2)
            for i in range(n // 2):
                samples.append(Int16(_le16(data, body + 2 * i)))
            return WavAudio(rate, channels, samples^)
        # Chunks are padded to an even size.
        pos = body + size + (size & 1)
    raise Error("wav: no data chunk")


def encode_wav(ref audio: WavAudio) -> List[UInt8]:
    var n_bytes = 2 * len(audio.samples)
    var out = List[UInt8](capacity=44 + n_bytes)
    _put_tag(out, "RIFF")
    _put32(out, 36 + n_bytes)
    _put_tag(out, "WAVE")
    _put_tag(out, "fmt ")
    _put32(out, 16)
    _put16(out, 1)  # PCM
    _put16(out, audio.channels)
    _put32(out, audio.sample_rate)
    _put32(out, audio.sample_rate * audio.channels * 2)  # byte rate
    _put16(out, audio.channels * 2)  # block align
    _put16(out, 16)
    _put_tag(out, "data")
    _put32(out, n_bytes)
    for i in range(len(audio.samples)):
        _put16(out, Int(audio.samples[i]) & 0xFFFF)
    return out^


def load_wav(path: String) raises -> WavAudio:
    var b = read_file_bytes(path)
    return decode_wav(b)


def save_wav(path: String, ref audio: WavAudio) raises:
    var b = encode_wav(audio)
    write_file_atomic(path, b)
