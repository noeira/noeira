# +--------------------------------------------------------------------------+ #
# | Parakeet-TDT — waveform in, text out
# +--------------------------------------------------------------------------+ #
"""`ParakeetSTT` — speech-to-text with `nvidia/parakeet-tdt-0.6b-v3`, no Python.

    var stt = ParakeetSTT[801].load()              # the Hub cache's snapshot
    var r = stt.transcribe(audio)                  # 16 kHz mono float32 in [-1, 1)
    print(r.text)

`F_MAX` is the compiled maximum in mel frames (100 per second, plus one):
801 = 8 s, which holds a spoken command with margin. A longer clip raises; a
second, longer instantiation is the answer for dictation, not a larger default
(the encoder's cost follows the compiled length, not the clip's).

`pad_s` adds silence on both sides before the front end. ⚠ It is not cosmetic:
the model transcribes a command cut tight markedly worse ("walk to the fridge"
-> "Yeah." with no margin; docs/NN_MODEL_PORTS_PLAN.md §10.2). A caller whose
capture already has half a second of margin passes 0.

`TARGET` is "cpu" (default) or "gpu": `ParakeetSTT[801, "gpu"].load()` runs
the encoder on the device (Metal or CUDA) and the token-by-token decoder on
the host. fp32 on both; the other precisions come next (same plan, §9.4).
"""

from max.gpu.host import DeviceContext

from noeira.audio.mel import LogMel
from .config import PK_MELS, PK_VOCAB
from .decode import PkDecodeWork, PkTrace, pk_greedy_decode
from .encoder import PkEncoderWork, pk_encode
from .paths import parakeet_snapshot_dir
from .tokenizer import ParakeetTokenizer
from .weights import ParakeetTDT


comptime PK_SAMPLE_RATE: Int = 16000
comptime PK_HOP: Int = 160


struct PkResult(Movable):
    var text: String
    var trace: PkTrace
    var mel_frames: Int
    var encoder_frames: Int

    def __init__(
        out self, var text: String, var trace: PkTrace, mel_frames: Int,
        encoder_frames: Int,
    ):
        self.text = text^
        self.trace = trace^
        self.mel_frames = mel_frames
        self.encoder_frames = encoder_frames


struct ParakeetSTT[F_MAX: Int = 801, TARGET: StaticString = "cpu"](Movable):
    var ctx: Optional[DeviceContext]
    var model: ParakeetTDT
    var tokenizer: ParakeetTokenizer
    var front: LogMel[]
    var enc: PkEncoderWork[Self.F_MAX]
    var dec: PkDecodeWork
    var _padded: List[Float32]

    def __init__(out self):
        self.ctx = None
        self.model = ParakeetTDT()
        self.tokenizer = ParakeetTokenizer()
        self.front = LogMel[]()
        self.enc = PkEncoderWork[Self.F_MAX]()
        self.dec = PkDecodeWork()
        self._padded = List[Float32]()

    @staticmethod
    def load(var dir: String = String("")) raises -> Self:
        """Load `model.safetensors` and `tokenizer.json` from `dir` (default:
        the Hub cache's snapshot of the published repository)."""
        if dir == "":
            dir = parakeet_snapshot_dir()
        var s = Self()
        comptime if Self.TARGET == "gpu":
            s.ctx = DeviceContext()
        s.model = ParakeetTDT.make[Self.TARGET](s.ctx)
        s.model.load[Self.TARGET](dir + "/model.safetensors", s.ctx)
        s.enc = PkEncoderWork[Self.F_MAX].make[Self.TARGET](s.ctx)
        s.tokenizer = ParakeetTokenizer.load(dir + "/tokenizer.json", PK_VOCAB)
        return s^

    @staticmethod
    def max_seconds() -> Float64:
        return Float64(Self.F_MAX * PK_HOP) / Float64(PK_SAMPLE_RATE)

    def transcribe(
        mut self, ref audio: List[Float32], pad_s: Float64 = 0.0
    ) raises -> PkResult:
        """Transcribe mono 16 kHz `audio`, with `pad_s` seconds of silence
        added on both sides."""
        var pad = Int(pad_s * Float64(PK_SAMPLE_RATE))
        var n_mel: Int
        if pad > 0:
            var n = len(audio) + 2 * pad
            self._padded = List[Float32](length=n, fill=0.0)
            for i in range(len(audio)):
                self._padded[pad + i] = audio[i]
            n_mel = self.front.extract[Self.F_MAX](self._padded, self.enc.feats.data)
        else:
            n_mel = self.front.extract[Self.F_MAX](audio, self.enc.feats.data)
        var valid = pk_encode[Self.F_MAX, Self.TARGET](
            self.model, self.enc, n_mel, self.ctx
        )
        var trace = pk_greedy_decode(
            self.model, self.enc.proj, valid, PkEncoderWork[Self.F_MAX].T, self.dec
        )
        var ids = trace.text_ids()
        var text = self.tokenizer.decode(ids)
        return PkResult(text^, trace^, n_mel, valid)
