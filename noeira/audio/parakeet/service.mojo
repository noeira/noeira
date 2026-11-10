# +--------------------------------------------------------------------------+ #
# | Parakeet as a SpeechToTextService: in this process, on its own thread
# +--------------------------------------------------------------------------+ #
"""`ParakeetService` is what `SpeechToText` (the HTTP client) is, without the
server: `start(audio)`, `poll` once a frame, `result()`, `cancel()` — the model
on a worker thread (`noeira.ai.local_service`).

    var stt = ParakeetService[].from_spec("native")       # loads in the background
    stt.start(audio)                                       # any rate, any channels
    while not stt.poll(): ...
    print(stt.result().text)

    comptime Loop = G1VoiceLoopOf[ParakeetService[1101, "gpu", 401, "fp16"], KokoroService]

The parameters are `ParakeetSTT`'s (see `stt.mojo`). `F_MAX` defaults to 1101
mel frames = 11 s, because `G1VoiceLoopOf` cuts a segment at 10 s
(`max_seg`) and Parakeet's own default, 8 s, would refuse the longest ones.
On the Orin use `WEIGHTS = "fp16"` (NVIDIA only) and a short shape.

Spec: `native` (the Hub cache's snapshot) or `native:<dir>`. Each clip gets
`pad_s` = 0.5 s of silence on both sides: a command cut tight transcribes
markedly worse (docs/NN_MODEL_PORTS_PLAN.md §10.2).

What the base model does NOT take: a language (it detects it; `set_language`
is kept for the caller's log only) and a vocabulary (`set_hint` says False).
"""

from noeira.ai.local_service import LocalCall, LocalModel, LocalWorker
from noeira.ai.speech import SpeechToTextService, Transcript
from noeira.core.concurrent.thread import sleep_us
from noeira.core.concurrent.worker import BackgroundThread
from noeira.io.wav import WavAudio
from .stt import ParakeetSTT, PK_SAMPLE_RATE


comptime PS_REQ_SLOT: Int = 2 << 20
"""Bytes per request: 16-bit samples at 16 kHz, ~65 s."""
comptime PS_REP_SLOT: Int = 64 * 1024


struct ParakeetModel[
    F_MAX: Int, TARGET: StaticString, F_SHORT: Int, WEIGHTS: StaticString,
](LocalModel):
    var dir: String
    var stt: Optional[ParakeetSTT[Self.F_MAX, Self.TARGET, Self.F_SHORT, Self.WEIGHTS]]

    def __init__(out self, var dir: String):
        self.dir = dir^
        self.stt = None

    def load(mut self) raises:
        self.stt = ParakeetSTT[Self.F_MAX, Self.TARGET, Self.F_SHORT, Self.WEIGHTS].load(self.dir)

    def warm(mut self) raises:
        var silence = List[Float32](length=PK_SAMPLE_RATE, fill=0.0)
        _ = self.stt.value().transcribe(silence, 0.5)

    def serve(mut self, ref request: List[UInt8]) raises -> List[UInt8]:
        """`[pad ms: u32 LE][16 kHz mono int16 LE samples]` -> UTF-8 text."""
        if len(request) < 4 or (len(request) - 4) % 2 != 0:
            raise Error("parakeet: a request is [pad ms u32][int16 samples]")
        var pad_ms = Int(request[0]) | (Int(request[1]) << 8) | (Int(request[2]) << 16) | (Int(request[3]) << 24)
        var n = (len(request) - 4) // 2
        var audio = List[Float32](capacity=n)
        for i in range(n):
            var u = UInt16(request[4 + 2 * i]) | (UInt16(request[5 + 2 * i]) << 8)
            audio.append(Float32(Int16(u)) / 32768.0)
        var r = self.stt.value().transcribe(audio, Float64(pad_ms) / 1000.0)
        var out = List[UInt8](capacity=r.text.byte_length())
        for x in r.text.as_bytes():
            out.append(x)
        return out^


struct ParakeetService[
    F_MAX: Int = 1101, TARGET: StaticString = "gpu", F_SHORT: Int = 0,
    WEIGHTS: StaticString = "fp32",
](Movable, SpeechToTextService):
    var pad_s: Float64
    var language: String
    """Kept for the caller; the base model detects the language itself."""
    var call: LocalCall
    var bg: BackgroundThread[LocalWorker[ParakeetModel[Self.F_MAX, Self.TARGET, Self.F_SHORT, Self.WEIGHTS]]]

    def __init__(out self, var dir: String = String("")) raises:
        """Starts the worker; the model loads on it in the background."""
        var call = LocalCall(PS_REQ_SLOT, PS_REP_SLOT)
        var bg = BackgroundThread(LocalWorker(
            ParakeetModel[Self.F_MAX, Self.TARGET, Self.F_SHORT, Self.WEIGHTS](dir^), call,
        ))
        self.pad_s = 0.5
        self.language = String("")
        self.call = call^
        self.bg = bg^

    @staticmethod
    def from_spec(spec: String) raises -> Self:
        """`native` or `native:<checkpoint dir>`."""
        if spec == "native":
            return Self()
        if spec.startswith("native:"):
            return Self(String(spec[byte=7:]))
        raise Error("stt: ParakeetService takes 'native' or 'native:<dir>', not '" + spec + "'")

    def state(self) -> Int:
        return self.call.state()

    def load_ms(self) -> Int:
        return self.call.load_ms()

    def set_language(mut self, lang: String):
        self.language = lang

    def set_hint(mut self, vocab: String) -> Bool:
        return False

    def warm_up(mut self):
        pass

    def start(mut self, ref audio: WavAudio) raises:
        """Down-mixed and resampled to 16 kHz mono here (cheap), the rest on
        the worker."""
        var mono = audio.to_mono()
        var a16 = mono.resample(PK_SAMPLE_RATE) if mono.sample_rate != PK_SAMPLE_RATE else mono^
        if Float64(len(a16.samples)) / Float64(PK_SAMPLE_RATE) + 2.0 * self.pad_s > ParakeetSTT[Self.F_MAX].max_seconds():
            raise Error("stt: a clip of " + String(Float64(len(a16.samples)) / Float64(PK_SAMPLE_RATE))
                        + " s (+ padding) is longer than this build's "
                        + String(ParakeetSTT[Self.F_MAX].max_seconds()) + " s")
        var pad_ms = Int(self.pad_s * 1000.0)
        var b = List[UInt8](capacity=4 + 2 * len(a16.samples))
        for k in range(4):
            b.append(UInt8((pad_ms >> (8 * k)) & 0xFF))
        for s in a16.samples:
            var u = UInt16(s)
            b.append(UInt8(u & 0xFF))
            b.append(UInt8(u >> 8))
        self.call.begin(b, String("speech-to-text (parakeet)"))

    def poll(mut self, timeout_ms: Int = 0) raises -> Bool:
        if self.call.poll():
            return True
        var waited = 0
        while waited < timeout_ms * 1000:
            _ = sleep_us(500)
            waited += 500
            if self.call.poll():
                return True
        return False

    def result(mut self) raises -> Transcript:
        var b = self.call.take(String("speech-to-text (parakeet)"))
        return Transcript(String(unsafe_from_utf8=Span(b)), self.call.latency_ms())

    def cancel(mut self) raises:
        self.call.cancel()
