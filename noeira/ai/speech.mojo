# +--------------------------------------------------------------------------+ #
# | Speech — speech-to-text and text-to-speech over HTTP
# +--------------------------------------------------------------------------+ #
"""Whisper-class transcription and TTS as API calls.

    var stt = SpeechToText.huggingface()            # HF_TOKEN, whisper-large-v3-turbo
    var t = stt.transcribe_file("command.wav")
    print(t.text, t.latency_ms)

    var tts = TextToSpeech.openai()                 # OPENAI_API_KEY
    save_wav("reply.wav", tts.speak("Cube placed."))

## Two upload shapes

| Factory | Shape | Key |
|---|---|---|
| `openai(model)` | multipart `/audio/transcriptions` | `OPENAI_API_KEY` |
| `groq(model)` | same, Groq's Whisper (very fast) | `GROQ_API_KEY` |
| `openai_compatible(url, model)` | same — LOCAL: speaches / faster-whisper-server, LocalAI | optional |
| `huggingface(model)` | raw audio body, `{"text": ...}` back | `HF_TOKEN` |

`TextToSpeech` speaks OpenAI's `/audio/speech`, which Kokoro-FastAPI and
speaches also serve locally, so the same migration path holds.

⚠ SEND 16 kHz MONO. Every Whisper resamples to 16 kHz mono internally, so a
48 kHz stereo capture is 6x the upload for the same transcript — and upload
is most of the latency on a slow link. `transcribe(audio)` converts first.
"""

from noeira.io.fileio import read_file_bytes
from noeira.io.http import HttpClient
from noeira.io.json import J_STRING, JsonWriter
from noeira.io.wav import WavAudio, decode_wav, encode_wav

from noeira.ai.keys import api_key
from noeira.ai.transport import warm_up, ApiCall, CALL_IDLE, MultipartForm


comptime STT_MULTIPART = 0
comptime STT_RAW = 1


@fieldwise_init
struct Transcript(Copyable, Movable):
    var text: String
    var latency_ms: Float64


struct SpeechToText(Movable):
    var kind: Int
    var url: String
    """The full endpoint, e.g. `https://api.openai.com/v1/audio/transcriptions`."""
    var model: String
    var language: String
    """ISO-639-1 hint ("en", "fr"); "" lets the model detect it. A hint
    avoids a wrong-language transcript of a two-word command."""
    var prompt: String
    """Text that biases the decoder's vocabulary — Whisper's own
    `prompt` field. Give it the words you expect and it will prefer them
    among homophones: a French speaker saying "cours" was transcribed
    "cool" twice in a row, and listing the command vocabulary is what
    separates them. It is a hint, not a grammar: anything may still come
    back."""
    var retries: Int
    var _http: HttpClient
    var _call: ApiCall

    def __init__(
        out self, kind: Int, var url: String, var model: String, var key: String
    ) raises:
        self.kind = kind
        self.url = url^
        self.model = model^
        self.language = String("")
        self.prompt = String("")
        self.retries = 2
        self._http = HttpClient(120000, 10000)
        self._call = ApiCall()
        if key.byte_length() > 0:
            self._http.bearer(key)

    def __init__(out self, *, deinit move: Self):
        self.kind = move.kind
        self.url = move.url^
        self.model = move.model^
        self.language = move.language^
        self.prompt = move.prompt^
        self.retries = move.retries
        self._http = move._http^
        self._call = move._call^

    @staticmethod
    def openai(var model: String = String("whisper-1")) raises -> SpeechToText:
        var key = api_key(["OPENAI_API_KEY"])
        return SpeechToText(
            STT_MULTIPART, String("https://api.openai.com/v1/audio/transcriptions"),
            model^, key^,
        )

    @staticmethod
    def groq(var model: String = String("whisper-large-v3-turbo")) raises -> SpeechToText:
        var key = api_key(["GROQ_API_KEY"])
        return SpeechToText(
            STT_MULTIPART,
            String("https://api.groq.com/openai/v1/audio/transcriptions"),
            model^, key^,
        )

    @staticmethod
    def openai_compatible(
        var base_url: String, var model: String, var key: String = String("")
    ) raises -> SpeechToText:
        """`base_url` up to `/v1`, e.g. `http://localhost:8000/v1`."""
        return SpeechToText(
            STT_MULTIPART, base_url + "/audio/transcriptions", model^, key^
        )

    @staticmethod
    def huggingface(
        var model: String = String("openai/whisper-large-v3-turbo")
    ) raises -> SpeechToText:
        var key = api_key(["HF_TOKEN", "HUGGINGFACE_API_KEY"])
        return SpeechToText(
            STT_RAW,
            "https://router.huggingface.co/hf-inference/models/" + model,
            model.copy(), key^,
        )

    def start_wav_bytes(mut self, ref wav: List[UInt8]) raises:
        """Upload in the background; `poll` it, then `result`."""
        var what = String("speech-to-text (") + self.model + ")"
        if self.kind == STT_RAW:
            self._call.begin(
                self._http, self.url, wav.copy(), String("audio/wav"), what^, self.retries
            )
            return
        var form = MultipartForm()
        form.field("model", self.model)
        form.field("response_format", "json")
        if self.language.byte_length() > 0:
            form.field("language", self.language)
        if self.prompt.byte_length() > 0:
            form.field("prompt", self.prompt)
        form.file("file", "audio.wav", "audio/wav", wav)
        var body = form.finish()
        self._call.begin(
            self._http, self.url, body^, form.content_type(), what^, self.retries
        )

    def start(mut self, ref audio: WavAudio) raises:
        """Down-mix and resample to 16 kHz mono, then `start_wav_bytes`."""
        var mono = audio.to_mono()
        var a16 = mono.resample(16000) if mono.sample_rate != 16000 else mono^
        var wav = encode_wav(a16)
        self.start_wav_bytes(wav)

    def poll(mut self, timeout_ms: Int = 0) raises -> Bool:
        return self._call.poll(self._http, timeout_ms)

    def result(mut self) raises -> Transcript:
        var reply = self._call.wait(self._http)
        self._call.state = CALL_IDLE
        var doc = reply.json()
        var t = doc.field(doc.root(), "text")
        if doc.kind_of(t) != J_STRING:
            raise Error("speech-to-text: no 'text' in the reply")
        return Transcript(doc.string(t), reply.latency_ms)

    def warm_up(mut self):
        """Open the TLS connection now, before a real-time loop — see
        `noeira.ai.transport.warm_up`."""
        warm_up(self._http, self.url)

    def cancel(mut self) raises:
        self._call.cancel(self._http)
        self._call.state = CALL_IDLE

    def transcribe_wav_bytes(mut self, ref wav: List[UInt8]) raises -> Transcript:
        self.start_wav_bytes(wav)
        return self.result()

    def transcribe(mut self, ref audio: WavAudio) raises -> Transcript:
        """Down-mix and resample to 16 kHz mono, then transcribe."""
        self.start(audio)
        return self.result()

    def transcribe_file(mut self, path: String) raises -> Transcript:
        var b = read_file_bytes(path)
        var audio = decode_wav(b)
        return self.transcribe(audio)


struct TextToSpeech(Movable):
    var url: String
    var model: String
    var voice: String
    var instructions: String
    """Speaking style for models that take one ("calm, short sentences");
    "" sends none."""
    var retries: Int
    var _http: HttpClient
    var _call: ApiCall

    def __init__(
        out self, var url: String, var model: String, var voice: String, var key: String
    ) raises:
        self.url = url^
        self.model = model^
        self.voice = voice^
        self.instructions = String("")
        self.retries = 2
        self._http = HttpClient(120000, 10000)
        self._call = ApiCall()
        if key.byte_length() > 0:
            self._http.bearer(key)

    def __init__(out self, *, deinit move: Self):
        self.url = move.url^
        self.model = move.model^
        self.voice = move.voice^
        self.instructions = move.instructions^
        self.retries = move.retries
        self._http = move._http^
        self._call = move._call^

    @staticmethod
    def openai(
        var model: String = String("gpt-4o-mini-tts"),
        var voice: String = String("alloy"),
    ) raises -> TextToSpeech:
        var key = api_key(["OPENAI_API_KEY"])
        return TextToSpeech(
            String("https://api.openai.com/v1/audio/speech"), model^, voice^, key^
        )

    @staticmethod
    def openai_compatible(
        var base_url: String,
        var model: String,
        var voice: String,
        var key: String = String(""),
    ) raises -> TextToSpeech:
        """Kokoro-FastAPI: `("http://localhost:8880/v1", "kokoro", "af_heart")`."""
        return TextToSpeech(base_url + "/audio/speech", model^, voice^, key^)

    def start(mut self, text: String) raises:
        """Synthesise in the background; `poll` it, then `result`."""
        var w = JsonWriter()
        w.begin_object()
        w.member("model", self.model)
        w.member("voice", self.voice)
        w.member("input", text)
        w.member("response_format", String("wav"))
        if self.instructions.byte_length() > 0:
            w.member("instructions", self.instructions)
        w.end_object()
        var body = w.done()
        var b = List[UInt8](capacity=body.byte_length())
        for i in range(body.byte_length()):
            b.append(body.as_bytes()[i])
        self._call.begin(
            self._http, self.url, b^, String("application/json"),
            String("text-to-speech (") + self.model + ")", self.retries,
        )

    def poll(mut self, timeout_ms: Int = 0) raises -> Bool:
        return self._call.poll(self._http, timeout_ms)

    def result_wav_bytes(mut self) raises -> List[UInt8]:
        var reply = self._call.wait(self._http)
        self._call.state = CALL_IDLE
        return reply^.take_body()

    def result(mut self) raises -> WavAudio:
        var b = self.result_wav_bytes()
        return decode_wav(b)

    def warm_up(mut self):
        """Open the TLS connection now, before a real-time loop — see
        `noeira.ai.transport.warm_up`."""
        warm_up(self._http, self.url)

    def cancel(mut self) raises:
        self._call.cancel(self._http)
        self._call.state = CALL_IDLE

    def speak_wav_bytes(mut self, text: String) raises -> List[UInt8]:
        self.start(text)
        return self.result_wav_bytes()

    def speak(mut self, text: String) raises -> WavAudio:
        self.start(text)
        return self.result()
