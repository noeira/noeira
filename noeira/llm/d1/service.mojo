# +--------------------------------------------------------------------------+ #
# | d1 as a DecisionService: in this process, on its own thread
# +--------------------------------------------------------------------------+ #
"""`D1Service` is what `JevClient` is, without the server: `start(state_json,
questions)` / `poll` once a frame / `result() -> JevAnswers` / `cancel`, with
d1-2B (`D1`, the LFM3 backbone on the device) on a worker thread
(`noeira.ai.local_service`). Its answers are the System One JSON hosted Jev
returns, so `JevAnswers` reads them unchanged.

    var dec = D1Service.from_spec("native")          # loads in the background (~5 s)
    dec.start(state_json, questions)                 # JevQuestions
    while not dec.poll(): ...
    var a = dec.result()                             # a.choice("destination"), a.confidence(...)

    comptime Loop = G1VoiceLoopOf[SpeechToText, TextToSpeech, D1Service]

Pictures (LFM3-VL's tower, `noeira.llm.lfm3.vision`): `start_seeing(state,
is_json, questions, pictures)`, `Picture`s RGB24 with sides multiples of 32
(640x480, 512x384, ...; anything else FAILS the call with the reason). One
640x480 picture + 3 questions ~0.3 s on the Orin (bf16).

Spec: `native` (`~/.cache/noeira/local-ai/d1/d1-2B`, the Orin's copy) or
`native:<dir>` — with the vision tower (~0.4 GB more on the device);
`native-text` / `native-text:<dir>` without it (pictures then fail). Build with
`-D LFM3_BF16=1` on NVIDIA for the tensor-core GEMMs (75 tokens ~95 ms, 250
~115 ms on the Orin).

The request, one message on the ring: `[0|1 state is JSON] state \0 questions`
then, with pictures, `\0 [u32 count] [u32 height][u32 width][RGB bytes]...`
(`Picture.append_to`); a request of up to DS_REQ_SLOT bytes.
"""

from std.os import getenv
from max.gpu.host import DeviceContext

from noeira.ai.jev import DecisionService, JevAnswers, JevQuestions
from noeira.ai.picture import Picture, picture_from
from noeira.ai.local_service import LocalCall, LocalModel, LocalWorker
from noeira.core.concurrent.thread import sleep_us
from noeira.core.concurrent.worker import BackgroundThread
from noeira.io.json import JsonWriter, parse_json
from .system_one import D1


comptime DS_REQ_SLOT: Int = 12 << 20
"""Two 1280x720 frames, or one at d1's pixel cap (1.6 MP, 4.9 MB), with room."""
comptime DS_REP_SLOT: Int = 64 * 1024


struct D1Model(LocalModel):
    var dir: String
    var vision: Bool
    var d1: Optional[D1]

    def __init__(out self, var dir: String, vision: Bool = True):
        self.dir = dir^
        self.vision = vision
        self.d1 = None

    def load(mut self) raises:
        self.d1 = D1.load(DeviceContext(), self.dir, self.vision)

    def warm(mut self) raises:
        var q = String('{"x": {"type": "noul", "instructions": "Is this a test?"}}')
        _ = self.d1.value().system_one(String("warm-up"), False, q)
        if self.vision:  # the tower's kernels and GEMM plans, so the first real picture is not the slow one
            var grey = List[UInt8](length=256 * 256 * 3, fill=128)
            var pics = List[Picture]()
            pics.append(Picture(grey^, 256, 256))
            _ = self.d1.value().system_one(String("warm-up"), False, q, pics)

    def serve(mut self, ref request: List[UInt8]) raises -> List[UInt8]:
        """`[0|1 state is JSON] state \\0 questions JSON [\\0 pictures]` -> the answers JSON."""
        var n = len(request)
        var cut = _find0(request, 1)
        if n < 2 or cut < 1:
            raise Error("d1: a request is '<0|1>state\\0questions[\\0pictures]'")
        var is_json = request[0] == 49
        var cut2 = _find0(request, cut + 1)
        var state = _text(request, 1, cut)
        var questions = _text(request, cut + 1, cut2 if cut2 >= 0 else n)
        var pics = List[Picture]()
        if cut2 >= 0:
            var at = cut2 + 5
            if at > n:
                raise Error("d1: a truncated picture count")
            var count = 0
            for k in range(4):
                count |= Int(request[cut2 + 1 + k]) << (8 * k)
            for _ in range(count):
                pics.append(picture_from(request, at))
            if at != n:
                raise Error("d1: " + String(n - at) + " bytes after the pictures")
        var out = self.d1.value().system_one(state, is_json, questions, pics)
        var b = List[UInt8](capacity=out.byte_length())
        for x in out.as_bytes():
            b.append(x)
        return b^


def _find0(ref b: List[UInt8], start: Int) -> Int:
    for i in range(start, len(b)):
        if b[i] == 0:
            return i
    return -1


def _text(ref b: List[UInt8], a: Int, e: Int) -> String:
    var t = List[UInt8](capacity=max(e - a, 0))
    for i in range(a, e):
        t.append(b[i])
    return String(unsafe_from_utf8=Span(t))


struct D1Service(Movable, DecisionService):
    var call: LocalCall
    var bg: BackgroundThread[LocalWorker[D1Model]]

    def __init__(out self, var dir: String = String(""), vision: Bool = True) raises:
        if dir.byte_length() == 0:
            dir = getenv("HOME") + "/.cache/noeira/local-ai/d1/d1-2B"
        var call = LocalCall(DS_REQ_SLOT, DS_REP_SLOT)
        var bg = BackgroundThread(LocalWorker(D1Model(dir^, vision), call))
        self.call = call^
        self.bg = bg^

    @staticmethod
    def from_spec(spec: String) raises -> D1Service:
        """`native` / `native:<checkpoint dir>` (with the vision tower), or
        `native-text` / `native-text:<dir>` (text only)."""
        if spec == "native":
            return D1Service()
        if spec.startswith("native:"):
            return D1Service(String(spec[byte=7:]))
        if spec == "native-text":
            return D1Service(String(""), False)
        if spec.startswith("native-text:"):
            return D1Service(String(spec[byte=12:]), False)
        raise Error("decision: D1Service takes 'native[-text]' or 'native[-text]:<dir>', not '" + spec + "'")

    def state(self) -> Int:
        return self.call.state()

    def load_ms(self) -> Int:
        return self.call.load_ms()

    def warm_up(mut self):
        pass

    def _begin(
        mut self, state: String, is_json: Bool, ref questions: JevQuestions, ref pictures: List[Picture],
    ) raises:
        var w = JsonWriter()
        questions.write_to_json(w)
        var q = w.done()
        var size = 2 + state.byte_length() + q.byte_length()
        if len(pictures) > 0:
            size += 5
            for pic in pictures:
                size += pic.framed_bytes()
        var b = List[UInt8](capacity=size)
        b.append(UInt8(49) if is_json else UInt8(48))
        for x in state.as_bytes():
            if x == 0:
                raise Error("decision (d1): a NUL byte in the state")
            b.append(x)
        b.append(0)
        for x in q.as_bytes():
            b.append(x)
        if len(pictures) > 0:
            b.append(0)
            var count = len(pictures)
            for k in range(4):
                b.append(UInt8((count >> (8 * k)) & 0xFF))
            for pic in pictures:
                pic.append_to(b)
        self.call.begin(b, String("decision (d1)"))

    def start(mut self, state_json: String, ref questions: JevQuestions) raises:
        var none = List[Picture]()
        self._begin(state_json, True, questions, none)

    def start_text(mut self, state_text: String, ref questions: JevQuestions) raises:
        var none = List[Picture]()
        self._begin(state_text, False, questions, none)

    def start_seeing(
        mut self, state: String, state_is_json: Bool, ref questions: JevQuestions, ref pictures: List[Picture],
    ) raises:
        """`start` with pictures (the request carries them; the tower reads them
        on the worker). A picture the tower does not take FAILS the call
        (`result` raises with the reason)."""
        self._begin(state, state_is_json, questions, pictures)

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

    def result(mut self) raises -> JevAnswers:
        var b = self.call.take(String("decision (d1)"))
        return JevAnswers(parse_json(b^), self.call.latency_ms())

    def cancel(mut self) raises:
        self.call.cancel()
