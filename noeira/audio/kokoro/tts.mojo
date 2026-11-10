# +--------------------------------------------------------------------------+ #
# | Kokoro-82M — phonemes in, speech out
# +--------------------------------------------------------------------------+ #
"""`KokoroTTS` — text-to-speech with `hexgrad/Kokoro-82M`, no Python, on the CPU.

    var tts = KokoroTTS.load()                       # ~/.cache/noeira/kokoro
    var s = tts.synthesize("bɔ̃ʒˈuʁ.", "ff_siwis")    # PHONEMES, and a voice
    save_wav("out.wav", wav_from_float32(s.audio, 24000))

The input is a PHONEME string in Kokoro's alphabet (IPA plus a few letters;
`config.json`'s `vocab`), not text: text -> phonemes is espeak-ng's job and
its binding is the next piece (docs/NN_MODEL_PORTS_PLAN.md §11.3, decided: A).

What `synthesize` does is `KModel.forward`: ids = 0, the phonemes' ids, 0
(characters outside the vocabulary are dropped, as upstream); the voice's
style row for this many phoneme CHARACTERS; durations; pitch and energy; the
decoder and the vocoder. 24 kHz, 600 samples per predicted frame.

The vocoder's source is random (`decoder.mojo`): `seed` picks the draw, so the
same seed gives the same waveform, and no seed gives the reference's — its
generator is torch's. The words do not depend on it.
"""

from std.math import cos, log, sqrt, pi

from noeira.io.json import load_json
from noeira.io.safetensors import SafeTensors
from noeira.nn.random.hash_mask import hash_u64
from .decoder import kk_decode, KokoroNoise, KK_HARMONICS, KK_SAMPLES_PER_FRAME
from .paths import kokoro_dir
from .prosody import kk_durations, kk_acoustic, KK_CONTEXT
from .weights import KokoroWeights


comptime KK_VOICE_ROWS: Int = 510
comptime KK_VOICE_WIDTH: Int = 256


struct KokoroSpeech(Movable):
    var audio: List[Float32]
    """24 kHz mono, in [-1, 1]."""
    var pred_dur: List[Int]
    """Frames per token (40 frames per second), boundary tokens included."""
    var frames: Int

    def __init__(out self, var audio: List[Float32], var pred_dur: List[Int], frames: Int):
        self.audio = audio^
        self.pred_dur = pred_dur^
        self.frames = frames


def _unit(seed: UInt64, stream: UInt64, idx: Int) -> Float64:
    """Uniform in (0, 1), a function of (seed, stream, idx)."""
    return (Float64(hash_u64(seed, stream, UInt64(idx)) >> 11) + 0.5) / 9007199254740992.0


def kokoro_noise(samples: Int, seed: UInt64) -> KokoroNoise:
    """The source module's draws for `samples` samples: nine uniform start
    phases and a standard normal per (sample, harmonic), from `seed`."""
    var n = KokoroNoise()
    for h in range(KK_HARMONICS):
        n.phase.append(Float32(_unit(seed, 1, h)))
    var count = samples * KK_HARMONICS
    for i in range(count):
        var u1 = _unit(seed, 2, i)
        var u2 = _unit(seed, 3, i)
        n.sines.append(Float32(sqrt(-2.0 * log(u1)) * cos(2.0 * pi * u2)))
    return n^


struct KokoroTTS(Movable):
    var dir: String
    var w: KokoroWeights
    var vocab: Dict[String, Int]

    def __init__(out self):
        self.dir = String("")
        self.w = KokoroWeights()
        self.vocab = Dict[String, Int]()

    @staticmethod
    def load(var dir: String = String("")) raises -> Self:
        """Load the converted checkpoint and the phoneme vocabulary from `dir`
        (default: `~/.cache/noeira/kokoro`, what
        `tools/kokoro/convert_kokoro_checkpoint.py` writes)."""
        if dir == "":
            dir = kokoro_dir()
        var t = Self()
        t.w = KokoroWeights.load(dir + "/kokoro-v1_0.safetensors")
        var doc = load_json(dir + "/config.json")
        var vocab = doc.field(doc.root(), "vocab")
        for i in range(doc.size(vocab)):
            t.vocab[doc.key_at(vocab, i)] = doc.integer(doc.at(vocab, i))
        if len(t.vocab) == 0:
            raise Error("kokoro: " + dir + "/config.json has no phoneme vocabulary")
        t.dir = dir^
        return t^

    def phoneme_ids(self, phonemes: String) raises -> List[Int]:
        """`0, the id of each phoneme character that has one, 0`."""
        var ids = List[Int]()
        ids.append(0)
        for cp in phonemes.codepoint_slices():
            var key = String(cp)
            if key in self.vocab:
                ids.append(self.vocab[key])
        ids.append(0)
        return ids^

    def style_row(self, voice: String, n_phonemes: Int) raises -> List[Float32]:
        """Row `n_phonemes - 1` of `voice`'s style table: 256 values."""
        if n_phonemes < 1 or n_phonemes > KK_VOICE_ROWS:
            raise Error("kokoro: a voice has styles for 1 to 510 phonemes, not " + String(n_phonemes))
        var st = SafeTensors(self.dir + "/voices/" + voice + ".safetensors")
        var pack = st.read_f32("pack")
        if len(pack) != KK_VOICE_ROWS * KK_VOICE_WIDTH:
            raise Error("kokoro: voice '" + voice + "' is not a [510, 256] table")
        var row = List[Float32]()
        var base = (n_phonemes - 1) * KK_VOICE_WIDTH
        for i in range(KK_VOICE_WIDTH):
            row.append(pack[base + i])
        return row^

    def synthesize(
        mut self, phonemes: String, voice: String, speed: Float32 = 1.0,
        seed: UInt64 = 0,
    ) raises -> KokoroSpeech:
        """Speak `phonemes` with `voice`. `speed` divides the durations."""
        var ids = self.phoneme_ids(phonemes)
        if len(ids) > KK_CONTEXT:
            raise Error("kokoro: " + String(len(ids)) + " tokens, the model's context is " + String(KK_CONTEXT))
        var ref_s = self.style_row(voice, phonemes.count_codepoints())
        var p = kk_durations(self.w, ids, ref_s, speed)
        var a = kk_acoustic(self.w, ids, ref_s, p)
        var noise = kokoro_noise(p.frames * KK_SAMPLES_PER_FRAME, seed)
        var d = kk_decode(self.w, a, ref_s, noise)
        return KokoroSpeech(d.audio.copy(), p.pred_dur.copy(), p.frames)
