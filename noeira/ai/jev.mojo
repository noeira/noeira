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

⚠ `confidence` IS THE PRODUCT. Act above a threshold, escalate (ask the LLM,
ask a human, stop) below it; a choice read without its confidence throws the
calibration away.
"""

from noeira.io.http import HttpClient
from noeira.io.json import J_NUMBER, J_OBJECT, J_STRING, JsonDoc, JsonWriter, parse_json

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

    def probability(self, id: String, option: String) raises -> Float64:
        """P(option) for a choice, or P(level) for a score (`"0"`, `"1"`…)."""
        var probs = self.doc.field(self._answer(id), "probabilities")
        var p = self.doc.field(probs, option)
        if p < 0:
            return 0.0
        return self.doc.number(p)


struct JevClient(Movable):
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
        self._http = HttpClient(30000, 10000)
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

    def warm_up(mut self):
        """Open the TLS connection now, before a real-time loop — see
        `noeira.ai.transport.warm_up`."""
        warm_up(self._http, self.url)

    def cancel(mut self) raises:
        self._call.cancel(self._http)
        self._call.state = CALL_IDLE
