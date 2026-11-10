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

`F_SHORT > 0` compiles a SECOND, shorter shape and uses it for every clip that
fits: `ParakeetSTT[801, "gpu", 401]` runs a clip of up to 4 s at 51 encoder
frames instead of 101. Off (0) by default, because where it was expected to
pay it does not yet: on an Orin the 240 GEMMs take 35.3 ms at 51 rows and
37.3 ms at 101 — at these row counts a GEMM's time is the STREAMING OF ITS
WEIGHTS (2.4 GB of fp32 per clip at ~60 GB/s), not its arithmetic — and on
Metal and the CPU the GEMMs already follow the valid rows. It becomes useful
once the weights are smaller (half precision, int4) and the rows matter
again. The transcripts do not depend on the shape (the NVIDIA transcript gate
runs the two-shape build).

`WEIGHTS = "fp16"` (NVIDIA, with `TARGET = "gpu"`) stores the encoder blocks'
GEMM weights in half precision — 96 % of the parameters, 1.3 GB on the device
instead of 2.5 — with fp32 activations; see `weights.mojo` for why that is the
lever on an Orin. "fp32" (default) is the reference form.

`pad_s` adds silence on both sides before the front end. ⚠ It is not cosmetic:
the model transcribes a command cut tight markedly worse ("walk to the fridge"
-> "Yeah." with no margin; docs/NN_MODEL_PORTS_PLAN.md §10.2). A caller whose
capture already has half a second of margin passes 0.

`TARGET` is "cpu" (default) or "gpu": `ParakeetSTT[801, "gpu"].load()` runs
the encoder and the token-by-token decoder on the device (Metal or CUDA); only
the mel front end stays on the host. fp32 weights on both (on NVIDIA the large
GEMMs run in TF32, nn's default there); the other precisions come next (same
plan, §9.4).
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


def _transcribe_at[F: Int, target: StaticString, WEIGHTS: StaticString](
    mut model: ParakeetTDT, mut front: LogMel[], mut enc: PkEncoderWork[F],
    mut dec: PkDecodeWork, ref tokenizer: ParakeetTokenizer,
    ref audio: List[Float32], ctx: Optional[DeviceContext],
) raises -> PkResult:
    """Front end, encoder, decoder and tokenizer at the compiled shape `F`."""
    comptime T = PkEncoderWork[F].T
    var n_mel = front.extract[F](audio, enc.feats.data)
    var valid = pk_encode[F, target, WEIGHTS](model, enc, n_mel, ctx)
    var trace = pk_greedy_decode[target, T](model, enc.proj, valid, T, dec, ctx)
    var ids = trace.text_ids()
    var text = tokenizer.decode(ids)
    return PkResult(text^, trace^, n_mel, valid)


struct ParakeetSTT[
    F_MAX: Int = 801, TARGET: StaticString = "cpu", F_SHORT: Int = 0,
    WEIGHTS: StaticString = "fp32",
](Movable):
    comptime FS: Int = Self.F_SHORT if Self.F_SHORT > 0 else 8
    """The short shape's frames; a token size when there is none (never run)."""

    var ctx: Optional[DeviceContext]
    var model: ParakeetTDT
    var tokenizer: ParakeetTokenizer
    var front: LogMel[]
    var enc: PkEncoderWork[Self.F_MAX]
    var enc_short: PkEncoderWork[Self.FS]
    var dec: PkDecodeWork
    var _padded: List[Float32]

    def __init__(out self):
        comptime assert Self.F_SHORT < Self.F_MAX, "F_SHORT must be shorter than F_MAX"
        self.ctx = None
        self.model = ParakeetTDT()
        self.tokenizer = ParakeetTokenizer()
        self.front = LogMel[]()
        self.enc = PkEncoderWork[Self.F_MAX]()
        self.enc_short = PkEncoderWork[Self.FS]()
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
        s.model.load[Self.TARGET, Self.WEIGHTS](dir + "/model.safetensors", s.ctx)
        s.enc = PkEncoderWork[Self.F_MAX].make[Self.TARGET, Self.WEIGHTS](s.ctx)
        comptime if Self.F_SHORT > 0:
            s.enc_short = PkEncoderWork[Self.FS].make[Self.TARGET, Self.WEIGHTS](s.ctx)
        s.dec = PkDecodeWork.make[Self.TARGET](s.ctx)
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
        if pad > 0:
            var n = len(audio) + 2 * pad
            self._padded = List[Float32](length=n, fill=0.0)
            for i in range(len(audio)):
                self._padded[pad + i] = audio[i]
            return self._run(self._padded.copy())
        return self._run(audio.copy())

    def _run(mut self, var audio: List[Float32]) raises -> PkResult:
        """At the short shape when there is one and the clip fits, else at `F_MAX`."""
        comptime if Self.F_SHORT > 0:
            if len(audio) // PK_HOP <= Self.F_SHORT:
                return _transcribe_at[Self.FS, Self.TARGET, Self.WEIGHTS](
                    self.model, self.front, self.enc_short, self.dec,
                    self.tokenizer, audio, self.ctx,
                )
        return _transcribe_at[Self.F_MAX, Self.TARGET, Self.WEIGHTS](
            self.model, self.front, self.enc, self.dec, self.tokenizer, audio,
            self.ctx,
        )
