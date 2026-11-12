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
read at each question's last token: the best log-probability among each
option's token forms (`prompt.readout_ids`), divided by the calibration's
temperature, softmaxed. Answers: `noul` P(yes); `choice` the likeliest option,
its confidence and every probability; `score` the MEDIAN level (d1's default
`score_readout`), its confidence, probabilities and legend.

Pictures (d1's `media`, LFM3-VL's tower: `noeira.llm.lfm3.vision`): their
markup heads the user turn, before the state (`<|vision_start|>`, one
`<|image|>` per 2x2 block of patches, `<|vision_end|>`, each picture in turn),
so under `set` every question sees them; the tower runs once per call and its
features fill the `<|image|>` tokens of every row.

Clips (d1-Omni-2B, `D1.load(..., audio=True)`; `noeira.llm.lfm3.audio`): each
clip's markup (`<|vision_start|>`, one `<|audio|>` per 80 ms, `<|vision_end|>`,
d1's processor's) heads the user turn the same way, AFTER the pictures' (d1's
`media` keeps the caller's order; here pictures come first, then clips).
"""

from std.math import exp
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.io.json import JsonWriter, load_json, parse_json, J_STRING
from noeira.nn.constants import DT
from noeira.llm.lfm3 import Lfm3Text, Lfm3Vision, Lfm3Audio, Picture
from noeira.text.bpe import ByteLevelBPE
from .prompt import (
    D1Question, D1Row, Q_NOUL, Q_CHOICE, Q_SCORE, parse_questions, py_json_dumps, prefix_text, suffix_text,
    pack_rows, readout_ids,
)


struct D1(Movable):
    var lm: Lfm3Text
    var tok: ByteLevelBPE
    var temperature: Float64
    """`calibration.json`'s default temperature (per-bucket ones are refused: d1-2B has none)."""
    var set_size: Int
    var last_rows: List[D1Row]
    """The rows of the last call (the gate compares them with d1's)."""
    var vision: Optional[Lfm3Vision]
    """The picture encoder (`D1.load(..., vision=True)`); None: text only."""
    var audio: Optional[Lfm3Audio]
    """The audio encoder (`D1.load(..., audio=True)`, an omni checkpoint); None: no clips."""

    def __init__(
        out self, var lm: Lfm3Text, var tok: ByteLevelBPE, temperature: Float64, var vision: Optional[Lfm3Vision] = None,
        var audio: Optional[Lfm3Audio] = None,
    ):
        self.vision = vision^
        self.audio = audio^
        self.lm = lm^
        self.tok = tok^
        self.temperature = temperature
        self.set_size = 16
        self.last_rows = List[D1Row]()

    @staticmethod
    def load(var ctx: DeviceContext, dir: String, vision: Bool = False, audio: Bool = False) raises -> D1:
        """`vision`: load the picture encoder too (~0.4 GB more on the device);
        `audio`: the audio encoder (d1-Omni, ~0.2 GB)."""
        var vis = Optional[Lfm3Vision](None)
        if vision:
            vis = Optional(Lfm3Vision.load(ctx, dir))
        var aud = Optional[Lfm3Audio](None)
        if audio:
            aud = Optional(Lfm3Audio.load(ctx, dir))
        var lm = Lfm3Text.load(ctx^, dir)
        var tok = ByteLevelBPE.load(dir + "/tokenizer.json")
        var cal = load_json(dir + "/calibration.json")
        var t = cal.number(cal.field(cal.root(), "default"))
        var per = cal.field(cal.root(), "temperatures")
        if per >= 0 and cal.size(per) > 0:
            raise Error("d1: per-bucket temperatures are not implemented")
        return D1(lm^, tok^, t, vis^, aud^)

    def system_one(
        mut self, state: String, state_is_json: Bool, questions_json: String,
        pictures: List[Picture] = List[Picture](), clips: List[List[Float32]] = List[List[Float32]](),
    ) raises -> String:
        """`state`: text, or a JSON value (`state_is_json`); `questions_json`:
        `{name: {type, instructions, criteria}}`; `pictures`: what the state
        shows (needs `D1.load(..., vision=True)`); `clips`: what was heard, 16
        kHz mono samples in [-1, 1] (needs `D1.load(..., audio=True)`). Returns
        d1's answer JSON."""
        var block: String
        if state_is_json:
            var sd = parse_json(_bytes(state))
            block = py_json_dumps(sd, sd.root())
        else:
            block = state
        var qd = parse_json(_bytes(questions_json))
        var qs = parse_questions(qd, qd.root())
        if len(qs) == 0:
            raise Error("d1: no questions")
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
        for q in qs:
            suffixes.append(suffix_text(self.tok, q))
        var rows = pack_rows(self.tok, prefix, suffixes, self.set_size)
        var probs = List[List[Float64]]()
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
                ref q = qs[row.first + s]
                var scores = List[Float64]()
                for g in readout_ids(self.tok, q):
                    var best = Float64(-1e300)
                    for id in g:
                        best = max(best, Float64(logz[s * v + id]))
                    scores.append(best / self.temperature)
                probs.append(_softmax(scores))
        self.last_rows = rows^
        var w = JsonWriter()
        w.begin_object()
        w.key("answers")
        w.begin_object()
        for i in range(len(qs)):
            _answer(w, qs[i], probs[i])
        w.end_object()
        w.key("usage")
        w.begin_object()
        w.member("input_tokens", read)
        w.member("output_tokens", 0)
        w.end_object()
        w.end_object()
        return w.done()


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
