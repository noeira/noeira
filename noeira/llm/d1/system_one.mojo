# +--------------------------------------------------------------------------+ #
# | d1 — System One in Mojo: a state and closed questions in, answers out
# +--------------------------------------------------------------------------+ #
"""`D1.load(ctx, dir)` then `system_one(state, questions)`: d1-2B's own API
(`modeling_d1.system_one`), the same JSON in and out as hosted Jev's
(`noeira.ai.jev`: `JevQuestions` / `JevAnswers`), without Python or a server.

    var d1 = D1.load(DeviceContext(), dir)
    var answers = d1.system_one(state_json, True, questions_json)   # {"answers": {...}, "usage": {...}}
    var seen = d1.system_one(state_json, True, questions_json, pictures)   # with pictures (d1's `media`)
    var heard = d1.system_one(state_json, True, questions_json, clips=clips)  # d1-Omni: heard clips (16 kHz mono)

One forward per row of up to 16 questions (`prompt.pack_rows`), the answers
read at each question's slots: the best log-probability among each option's
token forms (`prompt.question_slots`), softmaxed — divided first by the
calibration's temperature for an untyped question, raw for a typed one, whose
own table calibrates it. Answers: `noul` P(yes); `choice` the likeliest option,
its confidence and every probability; `score` the MEDIAN level (d1's default
`score_readout`), its confidence, probabilities and legend; a `multiple`
choice each option's P(yes) through the `multiple` table and the options at
0.5 or more; an estimate the median and p10-p90 of its grid distribution at
the `estimate` table's levels (`answers.mojo`) (docs/NN_MODEL_PORTS_PLAN.md
§12.9). Reads and boxes are laid out but refused until they are read (T4-T5).

Pictures (d1's `media`, LFM3-VL's tower: `noeira.llm.lfm3.vision`): their
markup heads the user turn, before the state (`<|vision_start|>`, one
`<|image|>` per 2x2 block of patches, `<|vision_end|>`, each picture in turn),
so under `set` every question sees them; the tower runs once per call and its
features fill the `<|image|>` tokens of every row.

Clips (d1-Omni-2B, `D1.load(..., audio=True)`; `noeira.llm.lfm3.audio`): each
clip's markup (`<|vision_start|>`, one `<|audio|>` per 80 ms, `<|vision_end|>`,
d1's processor's) heads the user turn the same way, AFTER the pictures' (d1's
`media` keeps the caller's order; here pictures come first, then clips).

`NOEIRA_D1_TRACE=<dir>` (read at construction): each call prints where its time
went (the CPU's part: the JSON, the tokens, the rows; the forwards with their
read-out, which wait for the GPU; the answer's JSON) and saves its request to
`<dir>/d1_req_<n>.bin` as `D1Service` frames it (`[0|1] state \0 questions`),
so `benchmarks/bench_d1_replay_gpu.mojo` can time the same requests alone
(docs/NN_MODEL_PORTS_PLAN.md §12.8.7, L1). Pictures and clips are not saved.
"""

from std.os import getenv
from std.time import perf_counter_ns

from std.math import exp
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.io.fileio import write_file_atomic
from noeira.io.json import JsonWriter, load_json, parse_json, J_STRING
from noeira.nn.constants import DT
from noeira.llm.lfm3 import Lfm3Text, Lfm3Vision, Lfm3Audio, Picture
from noeira.text.bpe import ByteLevelBPE
from .prompt import (
    D1Question, D1Record, D1Row, D1Slot, Q_NOUL, Q_CHOICE, Q_SCORE, Q_MULTI, Q_READ, Q_ESTIMATE, Q_BOX, Q_BOXES,
    SLOT_CODES, parse_questions,
    py_json_dumps, prefix_text, suffix_layout, pack_rows, question_slots, typed,
)
from .calibration import D1Calibration
from .answers import answer_estimate


struct D1(Movable):
    var lm: Lfm3Text
    var tok: ByteLevelBPE
    var temperature: Float64
    """`calibration.json`'s default temperature (per-bucket ones are refused: d1-2B has none)."""
    var calibration: D1Calibration
    """`calibration.json`: the temperature and the typed answers' tables."""
    var record: D1Record
    """What the checkpoint's `config.json` records (`d1_codes`, `d1_numbers`, ...); d1-2B's by default."""
    var set_size: Int
    var last_rows: List[D1Row]
    """The rows of the last call (the gate compares them with d1's)."""
    var last_probs: List[List[List[Float64]]]
    """The last call's distributions, per question and per slot, as d1's `run` reads them before
    answering: an untyped question's tempered softmax, a typed slot's raw one (the gates compare them)."""
    var vision: Optional[Lfm3Vision]
    """The picture encoder (`D1.load(..., vision=True)`); None: text only."""
    var audio: Optional[Lfm3Audio]
    """The audio encoder (`D1.load(..., audio=True)`, an omni checkpoint); None: no clips."""
    var trace: String
    """`NOEIRA_D1_TRACE`: the folder requests are saved to, "" = no trace (the module note)."""
    var traced: Int

    def __init__(
        out self, var lm: Lfm3Text, var tok: ByteLevelBPE, temperature: Float64, var vision: Optional[Lfm3Vision] = None,
        var audio: Optional[Lfm3Audio] = None,
    ):
        self.vision = vision^
        self.audio = audio^
        self.lm = lm^
        self.tok = tok^
        self.temperature = temperature
        self.calibration = D1Calibration()
        self.calibration.default = temperature
        self.record = D1Record.d1_2b()
        self.set_size = self.record.set_size
        self.last_rows = List[D1Row]()
        self.last_probs = List[List[List[Float64]]]()
        self.trace = getenv("NOEIRA_D1_TRACE")
        self.traced = 0

    @staticmethod
    def load(
        var ctx: DeviceContext, dir: String, vision: Bool = False, audio: Bool = False, w8a8: Bool = False,
    ) raises -> D1:
        """`vision`: load the picture encoder too (~0.4 GB more on the device);
        `audio`: the audio encoder (d1-Omni, ~0.2 GB); `w8a8`: the backbone's
        linears in int8, run as int8 GEMMs on int8 activations (NVIDIA;
        `noeira/llm/lfm3/model.mojo`, W8A8), the head's table and the
        operators' output projections kept bf16 (where int8 cost the answers
        most)."""
        var vis = Optional[Lfm3Vision](None)
        if vision:
            vis = Optional(Lfm3Vision.load(ctx, dir))
        var aud = Optional[Lfm3Audio](None)
        if audio:
            aud = Optional(Lfm3Audio.load(ctx, dir))
        var lm = Lfm3Text.load(ctx^, dir, int8=w8a8, int8_table=False, int8_operator_out=False)
        lm.w8a8 = w8a8
        var tok = ByteLevelBPE.load(dir + "/tokenizer.json")
        var cal = D1Calibration.load(dir + "/calibration.json")
        var d1 = D1(lm^, tok^, cal.default, vis^, aud^)
        d1.calibration = cal^
        d1.record = D1Record.load(dir + "/config.json")
        d1.set_size = d1.record.set_size
        return d1^

    def system_one(
        mut self, state: String, state_is_json: Bool, questions_json: String,
        pictures: List[Picture] = List[Picture](), clips: List[List[Float32]] = List[List[Float32]](),
    ) raises -> String:
        """`state`: text, or a JSON value (`state_is_json`); `questions_json`:
        `{name: {type, instructions, criteria}}`; `pictures`: what the state
        shows (needs `D1.load(..., vision=True)`); `clips`: what was heard, 16
        kHz mono samples in [-1, 1] (needs `D1.load(..., audio=True)`). Returns
        d1's answer JSON."""
        var t0 = perf_counter_ns()
        if self.trace != "":
            self._save(state, state_is_json, questions_json)
        var block: String
        if state_is_json:
            var sd = parse_json(_bytes(state))
            block = py_json_dumps(sd, sd.root())
        else:
            block = state
        var qd = parse_json(_bytes(questions_json))
        var qs = parse_questions(qd, qd.root(), self.record)
        if len(qs) == 0:
            raise Error("d1: no questions")
        for ref q in qs:
            if q.kind == Q_READ or q.kind == Q_BOX or q.kind == Q_BOXES:
                raise Error("d1: '" + q.name + "': text, number, date, box and boxes answers are laid out but not "
                            + "yet read (docs/NN_MODEL_PORTS_PLAN.md §12.9, T4-T5)")
        var media = String("")
        var feats = Optional[DeviceBuffer[DT]](None)
        var image_token = -1
        if len(pictures) > 0:
            if not self.vision:
                raise Error("d1: pictures need the vision tower (D1.load(..., vision=True))")
            ref vis = self.vision.value()
            for pic in pictures:
                media += vis.markup(pic, "<|image|>", "<|vision_start|>", "<|vision_end|>")
            feats = Optional(vis.encode_all(pictures))
            image_token = vis.cfg.image_token
        var afeats = Optional[DeviceBuffer[DT]](None)
        var audio_token = -1
        if len(clips) > 0:
            if not self.audio:
                raise Error("d1: clips need the audio encoder (D1.load(..., audio=True), an omni checkpoint)")
            ref aud = self.audio.value()
            for ref c in clips:
                media += aud.markup(len(c), "<|audio|>", "<|vision_start|>", "<|vision_end|>")
            afeats = Optional(aud.encode_all(clips))
            audio_token = aud.cfg.audio_token
        var prefix = prefix_text(self.tok, block, media)
        var suffixes = List[String]()
        var offsets = List[List[Int]]()
        for q in qs:
            var lay = suffix_layout(self.tok, q, self.record)
            suffixes.append(lay[0].copy())
            offsets.append(lay[1].copy())
        var rows = pack_rows(self.tok, prefix, suffixes, self.set_size, offsets)
        var slots_of = List[List[D1Slot]]()
        var probs = List[List[List[Float64]]]()
        """Per question, per slot, the slot's distribution (`SlotReader._padded`)."""
        for ref q in qs:
            slots_of.append(question_slots(self.tok, q, self.record))
            probs.append(List[List[Float64]]())
        var t1 = perf_counter_ns()
        var read = 0
        var v = self.lm.cfg.vocab
        for row in rows:
            read += len(row.ids)
            var taps = List[DeviceBuffer[DT]]()
            var h = self.lm.forward(
                row.ids, row.groups, row.state_len, taps, image_feats=feats.copy(), image_token=image_token,
                audio_feats=afeats.copy(), audio_token=audio_token,
            )
            var logz = self.lm.slot_logz(h, len(row.ids), row.slots)
            for s in range(len(row.slots)):
                var qi = row.slot_q[s]
                ref slot = slots_of[qi][len(probs[qi])]
                if slot.kind != SLOT_CODES:
                    raise Error("d1: a pointer slot is not read yet (§12.9 T4)")
                # a typed answer's slot is read raw: its own table calibrates it (`SlotReader._readout`)
                var t = 1.0 if typed(qs[qi]) else self.temperature
                var scores = List[Float64]()
                for ref g in slot.groups:
                    var best = Float64(-1e300)
                    for id in g:
                        best = max(best, Float64(logz[s * v + id]))
                    scores.append(best / t)
                probs[qi].append(_softmax(scores))
        var t2 = perf_counter_ns()
        var nrows = len(rows)
        self.last_rows = rows^
        self.last_probs = probs.copy()
        var w = JsonWriter()
        w.begin_object()
        w.key("answers")
        w.begin_object()
        for i in range(len(qs)):
            if qs[i].kind == Q_MULTI:
                _answer_multiple(w, qs[i], probs[i], self.calibration)
            elif qs[i].kind == Q_ESTIMATE:
                answer_estimate(w, qs[i], probs[i], self.calibration)
            else:
                _answer(w, qs[i], probs[i][0])
        w.end_object()
        w.key("usage")
        w.begin_object()
        w.member("input_tokens", read)
        w.member("output_tokens", 0)
        w.end_object()
        w.end_object()
        var out = w.done()
        if self.trace != "":
            print("  [d1 trace] request", self.traced - 1, ":", read, "tokens in", nrows, "row(s): CPU",
                  Int((t1 - t0) // 1_000_000), "ms, forwards + read-out", Int((t2 - t1) // 1_000_000), "ms, answer",
                  Int((perf_counter_ns() - t2) // 1_000_000), "ms")
        return out^

    def _save(mut self, state: String, state_is_json: Bool, questions_json: String):
        """The request as `D1Service` frames it, to `<trace>/d1_req_<n>.bin`. Never raises: a trace that cannot
        be written says so and the call goes on."""
        var b = List[UInt8](capacity=state.byte_length() + questions_json.byte_length() + 2)
        b.append(UInt8(49) if state_is_json else UInt8(48))
        for x in state.as_bytes():
            b.append(x)
        b.append(0)
        for x in questions_json.as_bytes():
            b.append(x)
        var path = self.trace + "/d1_req_" + String(self.traced) + ".bin"
        self.traced += 1
        try:
            write_file_atomic(path, b)
        except e:
            print("  [d1 trace] not saved:", path, e)


def _bytes(s: String) -> List[UInt8]:
    var b = List[UInt8](capacity=s.byte_length())
    for x in s.as_bytes():
        b.append(x)
    return b^


def _softmax(scores: List[Float64]) -> List[Float64]:
    var m = scores[0]
    for s in scores:
        m = max(m, s)
    var out = List[Float64]()
    var tot = 0.0
    for s in scores:
        var e = exp(s - m)
        out.append(e)
        tot += e
    for i in range(len(out)):
        out[i] /= tot
    return out^


def _answer_multiple(mut w: JsonWriter, q: D1Question, slots: List[List[Float64]], cal: D1Calibration) raises:
    """`api.answer` for a multiple choice: each option's raw P(yes) through the
    `multiple` table; the choices are the options at 0.5 or more."""
    var p = List[Float64]()
    for ref s in slots:
        p.append(cal.through("multiple", s[0]))
    w.key(q.name)
    w.begin_object()
    w.member("type", String("choice"))
    w.key("choices")
    w.begin_array()
    for i in range(len(p)):
        if p[i] >= 0.5:
            w.string(q.labels[i])
    w.end_array()
    w.key("probabilities")
    w.begin_object()
    for i in range(len(p)):
        w.member(q.labels[i], p[i])
    w.end_object()
    w.end_object()


def _answer(mut w: JsonWriter, q: D1Question, p: List[Float64]) raises:
    w.key(q.name)
    w.begin_object()
    if q.kind == Q_NOUL:
        w.member("type", String("noul"))
        w.member("noul", p[0])
        w.end_object()
        return
    var best = 0
    for i in range(len(p)):
        if p[i] > p[best]:
            best = i
    if q.kind == Q_CHOICE:
        w.member("type", String("choice"))
        w.member("choice", q.labels[best])
        w.member("confidence", p[best])
        w.key("probabilities")
        w.begin_object()
        for i in range(len(p)):
            w.member(q.labels[i], p[i])
        w.end_object()
        w.end_object()
        return
    var mass = 0.0
    var level = len(p) - 1
    for i in range(len(p)):
        mass += p[i]
        if mass >= 0.5:
            level = i
            break
    w.member("type", String("score"))
    w.member("score", Float64(level))
    w.member("confidence", p[best])
    w.key("probabilities")
    w.begin_object()
    for i in range(len(p)):
        w.member(String(i), p[i])
    w.end_object()
    w.key("legend")
    w.begin_object()
    for i in range(len(q.labels)):
        w.member(String(i), q.labels[i])
    w.end_object()
    w.end_object()
