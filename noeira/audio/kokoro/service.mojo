# +--------------------------------------------------------------------------+ #
# | Kokoro as a TextToSpeechService: in this process, on its own thread
# +--------------------------------------------------------------------------+ #
"""`KokoroService` is what `TextToSpeech` (the HTTP client) is, without the
server: a caller that polls once a frame gets the same `start` / `poll` /
`result_wav_bytes` / `cancel`, and the model runs on a worker thread
(`noeira.ai.local_service`).

    var tts = KokoroService.from_spec("native", "ff_siwis")    # loads in the background
    tts.start("Je vais au frigo.")
    while not tts.poll(): ...                                   # once a frame
    var wav = tts.result_wav_bytes()                            # 24 kHz mono WAV

    comptime Loop = G1VoiceLoopOf[SpeechToText, KokoroService]  # --tts native

The model is `KokoroDevice` (the GPU path: Metal, CUDA, the Orin; in a
`-D KOKORO_TF32=1` build its vocoder runs in TF32 through cuDNN) and the text
goes through espeak-ng in the same thread (`EspeakNG`, needs
`tools/kokoro/fetch_espeak_ng.py` once). Spec: `native` (the checkpoint in
`~/.cache/noeira/kokoro`) or `native:<dir>`.

Loading takes ~1 s plus one untimed reply; a `start` before that is queued and
served once the model is up. `state()` says where it is.
"""

from max.gpu.host import DeviceContext

from noeira.ai.local_service import LocalCall, LocalModel, LocalWorker, SVC_READY
from noeira.ai.speech import TextToSpeechService
from noeira.audio.espeak import EspeakNG
from noeira.core.concurrent.thread import sleep_us
from noeira.core.concurrent.worker import BackgroundThread
from noeira.io.wav import encode_wav, wav_from_float32
from .device import KokoroDevice


comptime KS_REQ_SLOT: Int = 16 * 1024
"""Bytes per request: the voice, the speed and the text."""
comptime KS_REP_SLOT: Int = 4 << 20
"""Bytes per reply: a WAV of up to ~87 s at 24 kHz (Kokoro's 510-token context
is ~30 s)."""


struct KokoroModel(LocalModel):
    var dir: String
    var tts: Optional[KokoroDevice]
    var es: Optional[EspeakNG]

    def __init__(out self, var dir: String):
        self.dir = dir^
        self.tts = None
        self.es = None

    def load(mut self) raises:
        self.es = EspeakNG.open()
        self.tts = KokoroDevice.load(DeviceContext(), self.dir)

    def warm(mut self) raises:
        _ = self.tts.value().speak(self.es.value(), String("Hello."), String("af_heart"))

    def serve(mut self, ref request: List[UInt8]) raises -> List[UInt8]:
        """`voice \\n speed (thousandths) \\n text` -> a WAV file's bytes."""
        var parts = String(unsafe_from_utf8=Span(request)).split("\n", 2)
        if len(parts) != 3:
            raise Error("kokoro: a request is 'voice\\nspeed\\ntext'")
        var milli = 0
        for c in parts[1].as_bytes():
            if c < 48 or c > 57:
                raise Error("kokoro: bad speed '" + String(parts[1]) + "'")
            milli = milli * 10 + Int(c - 48)
        var s = self.tts.value().speak(
            self.es.value(), String(parts[2]), String(parts[0]), Float32(milli) / 1000.0,
        )
        var wav = wav_from_float32(s.audio, 24000)
        return encode_wav(wav)


struct KokoroService(Movable, TextToSpeechService):
    var voice: String
    var speed: Float32
    var call: LocalCall
    var bg: BackgroundThread[LocalWorker[KokoroModel]]

    def __init__(out self, var voice: String, var dir: String = String("")) raises:
        """Starts the worker; the model loads on it in the background."""
        var call = LocalCall(KS_REQ_SLOT, KS_REP_SLOT)
        var bg = BackgroundThread(LocalWorker(KokoroModel(dir^), call))
        self.voice = voice^
        self.speed = 1.0
        self.call = call^
        self.bg = bg^

    @staticmethod
    def from_spec(spec: String, voice: String) raises -> KokoroService:
        """`native` or `native:<checkpoint dir>`."""
        if spec == "native":
            return KokoroService(voice)
        if spec.startswith("native:"):
            return KokoroService(voice, String(spec[byte=7:]))
        raise Error("tts: KokoroService takes 'native' or 'native:<dir>', not '" + spec + "'")

    def state(self) -> Int:
        """`SVC_LOADING`, `SVC_READY` or `SVC_FAILED` (`noeira.ai.local_service`)."""
        return self.call.state()

    def load_ms(self) -> Int:
        return self.call.load_ms()

    def latency_ms(self) -> Float64:
        """The model's own time for the last reply."""
        return self.call.latency_ms()

    def warm_up(mut self):
        """Nothing to do: the model loads and warms up on its thread from the
        constructor on."""
        pass

    def start(mut self, text: String) raises:
        var milli = Int(self.speed * 1000.0 + 0.5)
        var req = self.voice + "\n" + String(milli) + "\n" + text
        var b = List[UInt8](capacity=req.byte_length())
        for x in req.as_bytes():
            b.append(x)
        self.call.begin(b, String("text-to-speech (kokoro)"))

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

    def result_wav_bytes(mut self) raises -> List[UInt8]:
        return self.call.take(String("text-to-speech (kokoro)"))

    def cancel(mut self) raises:
        self.call.cancel()
