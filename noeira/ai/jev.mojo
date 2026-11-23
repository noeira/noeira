# +--------------------------------------------------------------------------+ #
# | Jev — TypeSafe's System One API: typed, calibrated decisions
# +--------------------------------------------------------------------------+ #
"""Ask closed questions about a state; get probabilities back, not prose.

    var jev = JevClient.from_env()               # JEV_API_KEY / TYPESAFE_API_KEY
    var q = JevQuestions()
    q.noul("holding", "Is the gripper holding the cube?")
    q.choice("next", "What should the arm do next?",
             ["grasp", "lift", "place", "retry"],
             ["close the jaws on the cube", "raise the cube",
              "lower it onto the target", "re-approach from above"])
    q.score("risk", "How likely is a collision?", ["none", "low", "high"])
    var a = jev.decide('{"gripper_gap_mm": 12, "cube_in_jaws": true}', q)
    if a.confidence("next") > 0.8:
        act(a.choice("next"))

`POST https://api.typesafe.ai/v1/systemone` with `{model, state, questions}`;
the answer is one typed distribution per question. Output tokens are free and
there is one decoding position, so a call is ~100 ms — fast enough for a
decision loop at a few Hz, which is why it sits next to the LLM client here.

## What it is, in a robotics stack (docs/SYSTEM_ONE_ASSESSMENT.md)

A **System 2 component** despite the name: a text-only classifier over a
CLOSED option set at 1–10 Hz. Right for "which skill next", "did the grasp
succeed", "is this state unsafe", termination / escalation signals. Not a
control policy: it emits labels, not joint targets, and its own docs list
arithmetic, counting and numeric comparison as failure modes — so put
thresholds IN THE STATE as booleans or words, not raw numbers to compare.

Typed answers (d1-2B in this process, `D1Service`; docs/NN_MODEL_PORTS_PLAN.md
§12.9): `choices` (every option that applies), `read` (a text, number or date
read from the state, with its span), `estimate` (a judged number with an
interval), `box` / `boxes` (where something is in the one picture of
`start_seeing`). Their answers carry `found` / `confidence` the same way.

⚠ `confidence` IS THE PRODUCT. Act above a threshold, escalate (ask the LLM,
ask a human, stop) below it; a choice read without its confidence throws the
calibration away.
"""

from std.os import getenv

from noeira.ai.picture import Picture
from noeira.io.http import HttpClient
from noeira.io.json import J_NULL, J_NUMBER, J_OBJECT, J_STRING, JsonDoc, JsonWriter, parse_json

from noeira.ai.keys import api_key, find_api_key
from noeira.ai.transport import warm_up, ApiCall, CALL_IDLE


struct JevQuestions(Copyable, Movable):
    """The `questions` map, built one question at a time. Ids are keys, so
    they must be unique; the LAST duplicate wins server-side."""

    var ids: List[String]
    var _json: List[String]
    """Each question's value object, already serialised."""

    def __init__(out self):
        self.ids = List[String]()
        self._json = List[String]()

    def noul(
        mut self,
        var id: String,
        var instructions: String,
        var when_true: String = String(""),
        var when_false: String = String(""),
    ) raises:
        """Yes/no; the answer is P(true). `when_true` / `when_false` optionally
        say what each side means."""
        var w = JsonWriter()
        w.begin_object()
        w.member("type", String("noul"))
        w.member("instructions", instructions)
        if when_true.byte_length() > 0 or when_false.byte_length() > 0:
            w.key("criteria")
            w.begin_object()
            w.member("true", when_true)
            w.member("false", when_false)
            w.end_object()
        w.end_object()
        self.ids.append(id^)
        self._json.append(w.done())

    def choice(
        mut self,
        var id: String,
        var instructions: String,
        options: List[String],
        descriptions: List[String] = List[String](),
    ) raises:
        """Pick one of `options` (up to 255). `descriptions[i]` says what
        belongs under `options[i]`; omitted, the option name stands for
        itself. Add an "other" option when none may apply — without one the
        model must pick something."""
        if len(descriptions) != 0 and len(descriptions) != len(options):
            raise Error("jev choice '" + id + "': one description per option")
        var w = JsonWriter()
        w.begin_object()
        w.member("type", String("choice"))
        w.member("instructions", instructions)
        w.key("criteria")
        w.begin_object()
        for i in range(len(options)):
            w.member(options[i], descriptions[i] if len(descriptions) > 0 else options[i])
        w.end_object()
        w.end_object()
        self.ids.append(id^)
        self._json.append(w.done())

    def score(
        mut self, var id: String, var instructions: String, levels: List[String]
    ) raises:
        """A graded answer over 2–10 ordered `levels` (level 0 first). The
        answer's `score` is the expected level, a float."""
        if len(levels) < 2 or len(levels) > 10:
            raise Error("jev score '" + id + "': 2 to 10 levels")
        var w = JsonWriter()
        w.begin_object()
        w.member("type", String("score"))
        w.member("instructions", instructions)
        w.key("criteria")
        w.begin_array()
        for i in range(len(levels)):
            w.string(levels[i])
        w.end_array()
        w.end_object()
        self.ids.append(id^)
        self._json.append(w.done())

    def choices(
        mut self,
        var id: String,
        var instructions: String,
        options: List[String],
        descriptions: List[String] = List[String](),
    ) raises:
        """Every option that applies (a `choice` with `multiple: true`): each
        option its own yes/no, the answer's `choices` those at 0.5 or more and
        each one's probability (`JevAnswers.choices`, `probability`)."""
        if len(options) == 0:
            raise Error("jev choices '" + id + "': at least one option")
        if len(descriptions) != 0 and len(descriptions) != len(options):
            raise Error("jev choices '" + id + "': one description per option")
        var w = JsonWriter()
        w.begin_object()
        w.member("type", String("choice"))
        w.key("multiple")
        w.boolean(True)
        w.member("instructions", instructions)
        w.key("criteria")
        w.begin_object()
        for i in range(len(options)):
            w.member(options[i], descriptions[i] if len(descriptions) > 0 else options[i])
        w.end_object()
        w.end_object()
        self.ids.append(id^)
        self._json.append(w.done())

    def read(mut self, var id: String, var instructions: String, kind: String = "text", unit: String = "") raises:
        """A value read FROM THE STATE: `kind` "text", "number" or "date". The
        answer is the span it was read from and, for a number or a date, its
        value parsed from it (`JevAnswers.text`, `number`, `date`, `span`,
        `found`); nothing when it is absent (`found` < 0.5). `unit`: a
        number's, when the question names one."""
        if kind != "text" and kind != "number" and kind != "date":
            raise Error("jev read '" + id + "': kind is text, number or date, not " + kind)
        var w = JsonWriter()
        w.begin_object()
        w.member("type", kind)
        w.member("instructions", instructions)
        if unit.byte_length() > 0:
            w.member("unit", unit)
        w.end_object()
        self.ids.append(id^)
        self._json.append(w.done())

    def estimate(
        mut self, var id: String, var instructions: String, low: Float64, high: Float64, step: Float64 = 0.0,
        unit: String = "",
    ) raises:
        """A number ESTIMATED over [`low`, `high`] (`step` apart when given,
        at most 128 values): the answer's median and its p10-p90 interval
        (`JevAnswers.number`, `interval`). Not read from the state: judged."""
        if not low < high:
            raise Error("jev estimate '" + id + "': low below high")
        var w = JsonWriter()
        w.begin_object()
        w.member("type", String("number"))
        w.member("instructions", instructions)
        w.key("range")
        w.begin_array()
        w.number(low)
        w.number(high)
        w.end_array()
        if step > 0.0:
            w.member("step", step)
        if unit.byte_length() > 0:
            w.member("unit", unit)
        w.end_object()
        self.ids.append(id^)
        self._json.append(w.done())

    def box(mut self, var id: String, var instructions: String) raises:
        """Where something is in THE picture (`start_seeing`, exactly one):
        its four edges as fractions of the picture, or none (`JevAnswers.box`)."""
        var w = JsonWriter()
        w.begin_object()
        w.member("type", String("box"))
        w.member("instructions", instructions)
        w.end_object()
        self.ids.append(id^)
        self._json.append(w.done())

    def boxes(mut self, var id: String, var instructions: String, max: Int = 0) raises:
        """Every instance of something in THE picture, in reading order
        (`JevAnswers.boxes`, `count`, `at_least`). `max` (1 to 16 on d1-2B):
        the most read; 0 for the model's default (16). Each block costs ~20
        tokens, so a small `max` is cheaper."""
        var w = JsonWriter()
        w.begin_object()
        w.member("type", String("boxes"))
        w.member("instructions", instructions)
        if max > 0:
            w.member("max", max)
        w.end_object()
        self.ids.append(id^)
        self._json.append(w.done())

    def write_to_json(self, mut w: JsonWriter) raises:
        w.begin_object()
        for i in range(len(self.ids)):
            w.key(self.ids[i])
            w.raw(self._json[i])
        w.end_object()


struct JevAnswers(Movable):
    var doc: JsonDoc
    var model: String
    """The concrete version that answered ("jev-1.13.0"), not the alias."""
    var input_tokens: Int
    var latency_ms: Float64

    def __init__(out self, var doc: JsonDoc, latency_ms: Float64) raises:
        self.doc = doc^
        self.latency_ms = latency_ms
        var root = self.doc.root()
        var m = self.doc.field(root, "model")
        self.model = self.doc.string(m) if self.doc.kind_of(m) == J_STRING else String("")
        var n = self.doc.field(self.doc.field(root, "usage"), "input_tokens")
        self.input_tokens = self.doc.integer(n) if self.doc.kind_of(n) == J_NUMBER else 0

    def __init__(out self, *, deinit move: Self):
        self.doc = move.doc^
        self.model = move.model^
        self.input_tokens = move.input_tokens
        self.latency_ms = move.latency_ms

    def _answer(self, id: String) raises -> Int:
        var a = self.doc.field(self.doc.field(self.doc.root(), "answers"), id)
        if a < 0:
            raise Error("jev: no answer for question '" + id + "'")
        return a

    def _num(self, id: String, field: String) raises -> Float64:
        var n = self.doc.field(self._answer(id), field)
        if n < 0:
            raise Error("jev: answer '" + id + "' has no '" + field + "'")
        return self.doc.number(n)

    def noul(self, id: String) raises -> Float64:
        """P(true) for a yes/no question."""
        return self._num(id, String("noul"))

    def choice(self, id: String) raises -> String:
        return self.doc.string(self.doc.field(self._answer(id), "choice"))

    def score(self, id: String) raises -> Float64:
        """Expected level of a score question (0 = first level)."""
        return self._num(id, String("score"))

    def confidence(self, id: String) raises -> Float64:
        """For a choice/score. A noul carries none: its probability is the
        confidence, so this returns max(p, 1 - p) there."""
        var a = self._answer(id)
        var c = self.doc.field(a, "confidence")
        if c >= 0:
            return self.doc.number(c)
        var p = self.noul(id)
        return p if p > 0.5 else 1.0 - p

    def choices(self, id: String) raises -> List[String]:
        """A multiple choice's options at 0.5 or more, in the question's order."""
        var c = self.doc.field(self._answer(id), "choices")
        if c < 0:
            raise Error("jev: answer '" + id + "' is not a multiple choice")
        var out = List[String]()
        for i in range(self.doc.size(c)):
            out.append(self.doc.string(self.doc.at(c, i)))
        return out^

    def found(self, id: String) raises -> Float64:
        """A read's or a box's calibrated P(it is there); a value or box is
        given when this is 0.5 or more."""
        return self._num(id, String("found"))

    def has_value(self, id: String) raises -> Bool:
        """A read's or an estimate's value is there (not null): a number read
        can be found yet unparsed ("3.1 m" has no number by d1's rules)."""
        var a = self._answer(id)
        var t = self.doc.field(a, "type")
        var v = self.doc.field(a, self.doc.string(t))
        return v >= 0 and self.doc.kind_of(v) != J_NULL

    def text(self, id: String) raises -> String:
        """The text a read points at ("" when nothing was found)."""
        var t = self.doc.field(self._answer(id), "text")
        if t < 0 or self.doc.kind_of(t) != J_STRING:
            return String("")
        return self.doc.string(t)

    def number(self, id: String) raises -> Float64:
        """A number read's parsed value, or an estimate's median. Raises when
        there is none (`has_value`)."""
        var n = self.doc.field(self._answer(id), "number")
        if n < 0 or self.doc.kind_of(n) != J_NUMBER:
            raise Error("jev: answer '" + id + "' has no number")
        return self.doc.number(n)

    def date(self, id: String) raises -> String:
        """A date read's ISO 8601 value ("" when none)."""
        var d = self.doc.field(self._answer(id), "date")
        if d < 0 or self.doc.kind_of(d) != J_STRING:
            return String("")
        return self.doc.string(d)

    def unit(self, id: String) raises -> String:
        """A number's unit: the question's, else the one written in the span ("" for none)."""
        var u = self.doc.field(self._answer(id), "unit")
        if u < 0 or self.doc.kind_of(u) != J_STRING:
            return String("")
        return self.doc.string(u)

    def span(self, id: String) raises -> Tuple[Int, Int]:
        """A read's `[start, end)` in the state's text, in characters; (-1, -1) for none."""
        var s = self.doc.field(self._answer(id), "span")
        if s < 0 or self.doc.kind_of(s) == J_NULL:
            return (-1, -1)
        return (self.doc.integer(self.doc.at(s, 0)), self.doc.integer(self.doc.at(s, 1)))

    def interval(self, id: String) raises -> Tuple[Float64, Float64]:
        """An estimate's p10-p90 interval."""
        var s = self.doc.field(self._answer(id), "interval")
        if s < 0:
            raise Error("jev: answer '" + id + "' is not an estimate")
        return (self.doc.number(self.doc.at(s, 0)), self.doc.number(self.doc.at(s, 1)))

    def box(self, id: String) raises -> JevBox:
        """A `box` question's box (`JevBox.drawn` False when there is none)."""
        var a = self._answer(id)
        var b = self.doc.field(a, "box")
        var found = self.doc.number(self.doc.field(a, "found"))
        var conf = self.doc.number(self.doc.field(a, "confidence"))
        if b < 0 or self.doc.kind_of(b) == J_NULL:
            return JevBox(False, 0.0, 0.0, 0.0, 0.0, found, conf)
        return _box_of(self.doc, b, found, conf)

    def boxes(self, id: String) raises -> List[JevBox]:
        """A `boxes` question's boxes, in reading order."""
        var l = self.doc.field(self._answer(id), "boxes")
        if l < 0:
            raise Error("jev: answer '" + id + "' is not a boxes answer")
        var out = List[JevBox]()
        for i in range(self.doc.size(l)):
            var e = self.doc.at(l, i)
            out.append(_box_of(self.doc, self.doc.field(e, "box"), self.doc.number(self.doc.field(e, "found")),
                               self.doc.number(self.doc.field(e, "confidence"))))
        return out^

    def count(self, id: String) raises -> Int:
        """How many boxes a `boxes` answer drew."""
        return self.doc.integer(self.doc.field(self._answer(id), "count"))

    def at_least(self, id: String) raises -> Bool:
        """Every block read was present: there may be more than `count`."""
        return self.doc.boolean(self.doc.field(self._answer(id), "at_least"))

    def probability(self, id: String, option: String) raises -> Float64:
        """P(option) for a choice, or P(level) for a score (`"0"`, `"1"`…)."""
        var probs = self.doc.field(self._answer(id), "probabilities")
        var p = self.doc.field(probs, option)
        if p < 0:
            return 0.0
        return self.doc.number(p)


@fieldwise_init
struct JevBox(Copyable, Movable):
    """A box in a picture, edges as fractions of it (0 = top / left)."""

    var drawn: Bool
    var top: Float64
    var bottom: Float64
    var left: Float64
    var right: Float64
    var found: Float64
    """Calibrated P(it is there); for `boxes`, P(at least this many)."""
    var confidence: Float64
    """Calibrated P(this box is right: IoU >= 0.5 with the truth)."""


def _box_of(ref doc: JsonDoc, b: Int, found: Float64, conf: Float64) raises -> JevBox:
    return JevBox(True, doc.number(doc.field(b, "top")), doc.number(doc.field(b, "bottom")),
                  doc.number(doc.field(b, "left")), doc.number(doc.field(b, "right")), found, conf)


trait DecisionService(Movable & Deinitable):
    """System One decisions, driven without blocking: `start` (a JSON state) or
    `start_text`, `poll` each frame until True, then `result`. The HTTP client
    (`JevClient`: hosted Jev, or any Jev-compatible server) and the model in
    this process (`noeira.llm.d1.service.D1Service`) both serve it, so a caller
    (`G1VoiceLoopOf`) picks one at BUILD time.

    ⚠ `poll` RETURNING TRUE MEANS THE CALL IS OVER, NOT THAT IT WORKED:
    `result` is what raises."""

    @staticmethod
    def from_spec(spec: String) raises -> Self:
        ...

    def loading(self) -> Bool:
        """True while the backend is still loading or warming up (a model in
        this process); a call made now waits for that first. As
        `noeira.ai.speech.SpeechToTextService.loading`."""
        ...

    def load_failed(self) -> Bool:
        """The backend's model failed to load or warm up: every call will fail.
        Default False (an HTTP client has nothing to load). ⚠ A SERVICE WITH A
        MODEL IN THIS PROCESS MUST OVERRIDE IT (`LocalCall.load_failed`): the
        default made the room demo say "every model ready" with no d1."""
        return False

    def warm_up(mut self):
        ...

    def start(mut self, state_json: String, ref questions: JevQuestions) raises:
        ...

    def start_text(mut self, state_text: String, ref questions: JevQuestions) raises:
        ...

    def start_seeing(
        mut self, state: String, state_is_json: Bool, ref questions: JevQuestions, ref pictures: List[Picture],
    ) raises:
        """`start` with what the state shows (`Picture`s, RGB24): the decision
        reads them too. Default: RAISES — a service that cannot see says so
        rather than answering without the pictures (`D1Service` can)."""
        raise Error("decision: this service takes no pictures (D1Service with a vision checkpoint does)")

    def poll(mut self, timeout_ms: Int = 0) raises -> Bool:
        ...

    def result(mut self) raises -> JevAnswers:
        ...

    def cancel(mut self) raises:
        ...


struct JevClient(Movable, DecisionService):
    var url: String
    var model: String
    var retries: Int
    var _http: HttpClient
    var _call: ApiCall

    def __init__(
        out self,
        var key: String,
        var model: String = String("jev-latest"),
        var url: String = String("https://api.typesafe.ai/v1/systemone"),
    ) raises:
        self.url = url^
        self.model = model^
        self.retries = 2
        # `JEV_TIMEOUT_MS`: a local server can be far slower than the API —
        # Kev-9B on a Jetson Orin takes ~31 s for the G1 demo's seven questions.
        var timeout_ms = 30000
        var t = getenv("JEV_TIMEOUT_MS")
        if t.byte_length() > 0:
            timeout_ms = atol(t)
        self._http = HttpClient(timeout_ms, 10000)
        # A local server without auth gets no header at all, never a bare
        # `Bearer ` (see `keys.mojo`).
        if key.byte_length() > 0:
            self._http.bearer(key)
        self._call = ApiCall()

    def __init__(out self, *, deinit move: Self):
        self.url = move.url^
        self.model = move.model^
        self.retries = move.retries
        self._http = move._http^
        self._call = move._call^

    @staticmethod
    def from_env(var model: String = String("jev-latest")) raises -> JevClient:
        """`JEV_URL` (environment or `.env`) points every caller at a
        Jev-compatible server instead — `laya-serve` on this machine:
        `http://127.0.0.1:8766/v1/systemone`. Its key is `LAYA_API_KEY`,
        optional; the TypeSafe key is never sent there."""
        var local = find_api_key(["JEV_URL"])
        if local.byte_length() > 0:
            return JevClient(find_api_key(["LAYA_API_KEY"]), model^, local^)
        var key = api_key(["JEV_API_KEY", "TYPESAFE_API_KEY"])
        return JevClient(key^, model^)

    @staticmethod
    def from_spec(spec: String) raises -> JevClient:
        """"" = `from_env` (hosted Jev, or `JEV_URL`); anything else is the URL
        of a Jev-compatible server (`LAYA_API_KEY` if set)."""
        if spec.byte_length() == 0:
            return JevClient.from_env()
        return JevClient(find_api_key(["LAYA_API_KEY"]), String("jev-latest"), spec)

    def request_body(
        self, state: String, state_is_json: Bool, ref questions: JevQuestions
    ) raises -> String:
        var w = JsonWriter()
        w.begin_object()
        w.member("model", self.model)
        w.key("state")
        if state_is_json:
            w.raw(state)
        else:
            w.string(state)
        w.key("questions")
        questions.write_to_json(w)
        w.end_object()
        return w.done()

    def decide(mut self, state_json: String, ref questions: JevQuestions) raises -> JevAnswers:
        """`state_json` is a JSON value (normally an object) describing the
        situation; questions refer to its fields by name."""
        self.start(state_json, questions)
        return self.result()

    def decide_text(mut self, state_text: String, ref questions: JevQuestions) raises -> JevAnswers:
        """Same, with free text as the state (a transcript, a message)."""
        self.start_text(state_text, questions)
        return self.result()

    # ── background: decide while the loop keeps running ───────────────
    #
    #     jev.start(state, q)
    #     while not jev.poll():       # never blocks
    #         step_sim(); draw()
    #     var a = jev.result()

    def start(mut self, state_json: String, ref questions: JevQuestions) raises:
        self._begin(self.request_body(state_json, True, questions))

    def start_text(mut self, state_text: String, ref questions: JevQuestions) raises:
        self._begin(self.request_body(state_text, False, questions))

    def _begin(mut self, body: String) raises:
        var b = List[UInt8](capacity=body.byte_length())
        for i in range(body.byte_length()):
            b.append(body.as_bytes()[i])
        self._call.begin(
            self._http, self.url, b^, String("application/json"),
            String("jev decide"), self.retries,
        )

    def poll(mut self, timeout_ms: Int = 0) raises -> Bool:
        """True once the answer is in (or the call failed — `result` says)."""
        return self._call.poll(self._http, timeout_ms)

    def result(mut self) raises -> JevAnswers:
        var reply = self._call.wait(self._http)
        self._call.state = CALL_IDLE
        return JevAnswers(reply.json(), reply.latency_ms)

    def loading(self) -> Bool:
        return False    # an HTTP client: nothing to load

    def warm_up(mut self):
        """Open the TLS connection now, before a real-time loop — see
        `noeira.ai.transport.warm_up`."""
        warm_up(self._http, self.url)

    def cancel(mut self) raises:
        self._call.cancel(self._http)
        self._call.state = CALL_IDLE
