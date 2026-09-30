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
from noeira.ai.transport import MultipartForm, post_api, post_json_api


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
    var retries: Int
    var _http: HttpClient

    def __init__(
        out self, kind: Int, var url: String, var model: String, var key: String
    ) raises:
        self.kind = kind
        self.url = url^
        self.model = model^
        self.language = String("")
        self.retries = 2
        self._http = HttpClient(120000, 10000)
        if key.byte_length() > 0:
            self._http.bearer(key)

    def __init__(out self, *, deinit move: Self):
        self.kind = move.kind
        self.url = move.url^
        self.model = move.model^
        self.language = move.language^
        self.retries = move.retries
        self._http = move._http^

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

    def transcribe_wav_bytes(mut self, ref wav: List[UInt8]) raises -> Transcript:
        var what = String("speech-to-text (") + self.model + ")"
        if self.kind == STT_RAW:
            var reply = post_api(
                self._http, self.url, wav, String("audio/wav"), what^, self.retries
            )
            var doc = reply.json()
            var t = doc.field(doc.root(), "text")
            if doc.kind_of(t) != J_STRING:
                raise Error("speech-to-text: no 'text' in the reply")
            return Transcript(doc.string(t), reply.latency_ms)
        var form = MultipartForm()
        form.field("model", self.model)
        form.field("response_format", "json")
        if self.language.byte_length() > 0:
            form.field("language", self.language)
        form.file("file", "audio.wav", "audio/wav", wav)
        var body = form.finish()
        var reply = post_api(
            self._http, self.url, body, form.content_type(), what^, self.retries
        )
        var doc = reply.json()
        return Transcript(doc.string(doc.field(doc.root(), "text")), reply.latency_ms)

    def transcribe(mut self, ref audio: WavAudio) raises -> Transcript:
        """Down-mix and resample to 16 kHz mono, then transcribe."""
        var mono = audio.to_mono()
        var a16 = mono.resample(16000) if mono.sample_rate != 16000 else mono^
        var wav = encode_wav(a16)
        return self.transcribe_wav_bytes(wav)

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

    def __init__(
        out self, var url: String, var model: String, var voice: String, var key: String
    ) raises:
        self.url = url^
        self.model = model^
        self.voice = voice^
        self.instructions = String("")
        self.retries = 2
        self._http = HttpClient(120000, 10000)
        if key.byte_length() > 0:
            self._http.bearer(key)

    def __init__(out self, *, deinit move: Self):
        self.url = move.url^
        self.model = move.model^
        self.voice = move.voice^
        self.instructions = move.instructions^
        self.retries = move.retries
        self._http = move._http^

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

    def speak_wav_bytes(mut self, text: String) raises -> List[UInt8]:
        var w = JsonWriter()
        w.begin_object()
        w.member("model", self.model)
        w.member("voice", self.voice)
        w.member("input", text)
        w.member("response_format", String("wav"))
        if self.instructions.byte_length() > 0:
            w.member("instructions", self.instructions)
        w.end_object()
        var reply = post_json_api(
            self._http, self.url, w.done(), String("text-to-speech (") + self.model + ")",
            self.retries,
        )
        return reply^.take_body()

    def speak(mut self, text: String) raises -> WavAudio:
        var b = self.speak_wav_bytes(text)
        return decode_wav(b)
