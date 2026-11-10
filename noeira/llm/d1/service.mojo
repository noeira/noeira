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

Spec: `native` (`~/.cache/noeira/local-ai/d1/d1-2B`, the Orin's copy) or
`native:<dir>`. Build with `-D LFM3_BF16=1` on NVIDIA for the tensor-core GEMMs
(75 tokens ~95 ms, 250 ~250 ms on the Orin).
"""

from std.os import getenv
from max.gpu.host import DeviceContext

from noeira.ai.jev import DecisionService, JevAnswers, JevQuestions
from noeira.ai.local_service import LocalCall, LocalModel, LocalWorker
from noeira.core.concurrent.thread import sleep_us
from noeira.core.concurrent.worker import BackgroundThread
from noeira.io.json import JsonWriter, parse_json
from .system_one import D1


comptime DS_REQ_SLOT: Int = 256 * 1024
comptime DS_REP_SLOT: Int = 64 * 1024


struct D1Model(LocalModel):
    var dir: String
    var d1: Optional[D1]

    def __init__(out self, var dir: String):
        self.dir = dir^
        self.d1 = None

    def load(mut self) raises:
        self.d1 = D1.load(DeviceContext(), self.dir)

    def warm(mut self) raises:
        _ = self.d1.value().system_one(
            String("warm-up"), False, String('{"x": {"type": "noul", "instructions": "Is this a test?"}}'),
        )

    def serve(mut self, ref request: List[UInt8]) raises -> List[UInt8]:
        """`[0|1 state is JSON] state \\0 questions JSON` -> the answers JSON."""
        var text = String(unsafe_from_utf8=Span(request))
        var cut = text.find("\x00")
        if cut < 1:
            raise Error("d1: a request is '<0|1>state\\0questions'")
        var is_json = text.as_bytes()[0] == 49
        var state = String(text[byte=1:cut])
        var questions = String(text[byte=cut + 1:])
        var out = self.d1.value().system_one(state, is_json, questions)
        var b = List[UInt8](capacity=out.byte_length())
        for x in out.as_bytes():
            b.append(x)
        return b^


struct D1Service(Movable, DecisionService):
    var call: LocalCall
    var bg: BackgroundThread[LocalWorker[D1Model]]

    def __init__(out self, var dir: String = String("")) raises:
        if dir.byte_length() == 0:
            dir = getenv("HOME") + "/.cache/noeira/local-ai/d1/d1-2B"
        var call = LocalCall(DS_REQ_SLOT, DS_REP_SLOT)
        var bg = BackgroundThread(LocalWorker(D1Model(dir^), call))
        self.call = call^
        self.bg = bg^

    @staticmethod
    def from_spec(spec: String) raises -> D1Service:
        """`native` or `native:<checkpoint dir>`."""
        if spec == "native":
            return D1Service()
        if spec.startswith("native:"):
            return D1Service(String(spec[byte=7:]))
        raise Error("decision: D1Service takes 'native' or 'native:<dir>', not '" + spec + "'")

    def state(self) -> Int:
        return self.call.state()

    def load_ms(self) -> Int:
        return self.call.load_ms()

    def warm_up(mut self):
        pass

    def _begin(mut self, state: String, is_json: Bool, ref questions: JevQuestions) raises:
        var w = JsonWriter()
        questions.write_to_json(w)
        var q = w.done()
        var req = String("1" if is_json else "0") + state + "\x00" + q
        var b = List[UInt8](capacity=req.byte_length())
        for x in req.as_bytes():
            b.append(x)
        self.call.begin(b, String("decision (d1)"))

    def start(mut self, state_json: String, ref questions: JevQuestions) raises:
        self._begin(state_json, True, questions)

    def start_text(mut self, state_text: String, ref questions: JevQuestions) raises:
        self._begin(state_text, False, questions)

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
