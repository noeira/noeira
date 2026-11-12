# +--------------------------------------------------------------------------+ #
# | d1-Omni HEARING a clip vs Parakeet -> d1 READING its transcript
# +--------------------------------------------------------------------------+ #
"""Step 3 of d1-Omni (docs/NN_MODEL_PORTS_PLAN.md §12.6): the same room clips and
questions through the two voice paths, all in Mojo on one device.

    python3 tools/d1/make_room_clips.py                 # on the Mac: the clips + expected answers
    pixi run -e nvidia mojo build -D LFM3_BF16=1 -I . benchmarks/bench_d1_heard_vs_transcribed_gpu.mojo \
        -o build/bench_d1_heard
    ./build/bench_d1_heard [clips_dir] [d1-Omni dir] [d1-2B dir] [out.tsv]

Columns (per clip, the destination / addressed answers and the warm latency):

    omni heard        d1-Omni, the clip as audio (`system_one(..., clips=[clip])`)
    omni parakeet     d1-Omni, Parakeet's transcript as the state's "heard" (no audio)
    2B parakeet       d1-2B, the same transcript: the deployed cascade
    omni truth / 2B truth   the manifest's text: what a perfect transcript gives

Latency: the heard column is d1-Omni's whole call (mel, encoder, backbone); the
cascade is Parakeet (0.5 s of padding both sides, as `ParakeetService` does) +
d1-2B's call. The models are loaded one at a time (16 GB shared on the Orin):
Parakeet transcribes every clip, then d1-Omni runs, then d1-2B. Each pass warms
up on the first clip before it times anything.

Scores (expected answers from the manifest, `make_room_clips.py`):
- destination: d1's choice equals the expected place (nowhere included);
- addressed: P(yes) >= 0.5 equals the label (clips labelled `?` left out);
- action: what the robot would DO — walk to the chosen place if P(addressed) >=
  0.5 and the choice is not nowhere, else stay — equals the expected place;
- p(expected): the mean probability of the expected place.

The state puts the robot in the middle of the room, away from every place
(step 1's state had it AT the door, which failed every "go to the door").
"""

from std.os import getenv
from std.sys import argv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from noeira.io.json import parse_json, json_quote
from noeira.io.wav import load_wav
from noeira.audio.parakeet.stt import ParakeetSTT
from noeira.llm.d1 import D1
from noeira.llm.lfm3.model import LF_BF16


comptime ROBOT = """{"at": "the middle of the living room", "facing": "kitchen", "doing": "standing"}"""
comptime PLACES = """{"fridge": "3.1 m ahead, slightly left", "door": "3.8 m to the left", "table": "2.0 m to the right", "sofa": "4.5 m behind"}"""
comptime QUESTIONS = """{"addressed": {"type": "noul", "instructions": "Is the speaker asking the robot to do something?"}, "destination": {"type": "choice", "instructions": "Where should the robot walk?", "criteria": {"fridge": "walk to the fridge", "door": "walk to the door", "table": "walk to the table", "sofa": "walk to the sofa", "nowhere": "do not walk anywhere"}}, "urgency": {"type": "score", "instructions": "How urgent is the request?", "criteria": ["not urgent", "somewhat urgent", "very urgent"]}}"""
comptime HEARD_AUDIO = "the attached recording"
comptime PK_PAD_S: Float64 = 0.5

comptime N_COLS = 5
comptime C_OMNI_HEARD = 0
comptime C_OMNI_PK = 1
comptime C_2B_PK = 2
comptime C_OMNI_TRUTH = 3
comptime C_2B_TRUTH = 4


def col_name(c: Int) -> String:
    if c == C_OMNI_HEARD:
        return "omni heard"
    if c == C_OMNI_PK:
        return "omni parakeet"
    if c == C_2B_PK:
        return "2B parakeet"
    if c == C_OMNI_TRUTH:
        return "omni truth"
    return "2B truth"


def state_json(heard: String) raises -> String:
    return (
        String('{"robot": ') + ROBOT + ', "places": ' + PLACES + ', "heard": ' + json_quote(heard)
        + ', "recently": []}'
    )


def read_text(path: String) raises -> String:
    var f = open(path, "r")
    var s = f.read()
    f.close()
    return s^


@fieldwise_init
struct Clip(Copyable, Movable):
    var name: String
    var lang: String
    var text: String
    var addressed: String
    var dest: String


@fieldwise_init
struct Answer(Copyable, Movable):
    var choice: String
    var p_expected: Float64
    var p_addressed: Float64
    var ms: Float64


def read_manifest(dir: String) raises -> List[Clip]:
    var out = List[Clip]()
    # ⚠ split("\n"), not splitlines(): the fields are TAB-separated
    for line in read_text(dir + "/manifest.tsv").split("\n"):
        var s = String(line)
        if s == "":
            continue
        var p = s.split("\t")
        if len(p) < 6:
            raise Error("manifest line without labels (make_room_clips.py): " + s)
        out.append(Clip(String(p[0]), String(p[1]), String(p[3]), String(p[4]), String(p[5])))
    return out^


def load_clip(dir: String, name: String) raises -> List[Float32]:
    var wav = load_wav(dir + "/" + name)
    if wav.sample_rate != 16000 or wav.channels != 1:
        raise Error(name + ": need 16 kHz mono")
    return wav.to_float32()


def read_answer(json: String, expected: String, ms: Float64) raises -> Answer:
    var b = List[UInt8](capacity=json.byte_length())
    for x in json.as_bytes():
        b.append(x)
    var d = parse_json(b^)
    var a = d.field(d.root(), "answers")
    var dest = d.field(a, "destination")
    var probs = d.field(dest, "probabilities")
    return Answer(
        d.string(d.field(dest, "choice")),
        d.number(d.field(probs, expected)),
        d.number(d.field(d.field(a, "addressed"), "noul")),
        ms,
    )


def transcribe_all(dir: String, clips: List[Clip], mut texts: List[String], mut ms: List[Float64]) raises:
    """Parakeet (fp16 weights, the 4 s / 8 s shapes) over every clip; freed on return."""
    var stt = ParakeetSTT[801, "gpu", 401, "fp16"].load()
    _ = stt.transcribe(load_clip(dir, clips[0].name), PK_PAD_S)  # warm-up
    for ref c in clips:
        var audio = load_clip(dir, c.name)
        var t0 = perf_counter_ns()
        var r = stt.transcribe(audio, PK_PAD_S)
        ms.append(Float64(perf_counter_ns() - t0) / 1e6)
        texts.append(r.text)


def run_d1(
    model_dir: String, omni: Bool, dir: String, clips: List[Clip], pk: List[String],
    mut res: List[List[Answer]],
) raises:
    """d1-Omni (`omni`: heard, parakeet, truth) or d1-2B (parakeet, truth) over every clip."""
    var d1 = D1.load(DeviceContext(), model_dir, audio=omni)
    var none = List[List[Float32]]()
    var warm = List[List[Float32]]()
    warm.append(load_clip(dir, clips[0].name))
    if omni:
        _ = d1.system_one(state_json(HEARD_AUDIO), True, QUESTIONS, clips=warm^)
    else:
        _ = d1.system_one(state_json(clips[0].text), True, QUESTIONS)
    for i in range(len(clips)):
        ref c = clips[i]
        if omni:
            var one = List[List[Float32]]()
            one.append(load_clip(dir, c.name))
            var t0 = perf_counter_ns()
            var out = d1.system_one(state_json(HEARD_AUDIO), True, QUESTIONS, clips=one^)
            res[C_OMNI_HEARD].append(read_answer(out, c.dest, Float64(perf_counter_ns() - t0) / 1e6))
        for k in range(2):
            var heard = pk[i] if k == 0 else c.text
            var t0 = perf_counter_ns()
            var out = d1.system_one(state_json(heard), True, QUESTIONS, clips=none.copy())
            var a = read_answer(out, c.dest, Float64(perf_counter_ns() - t0) / 1e6)
            var col = (C_OMNI_PK if k == 0 else C_OMNI_TRUTH) if omni else (C_2B_PK if k == 0 else C_2B_TRUTH)
            res[col].append(a^)


def walks_to(a: Answer) -> String:
    if a.p_addressed >= 0.5 and a.choice != "nowhere":
        return a.choice
    return "nowhere"


def words(s: String) -> String:
    """Lower case, no punctuation, single spaces: a transcript compared word for word."""
    var t = s.lower()
    for p in [".", ",", "!", "?", ";", ":"]:
        t = t.replace(p, " ")
    var out = String("")
    for w in t.split(" "):
        if w.byte_length() > 0:
            out += (" " if out.byte_length() > 0 else "") + String(w)
    return out^


def pct(n: Int, d: Int) -> String:
    if d == 0:
        return "-"
    return String(n) + "/" + String(d)


def median(var xs: List[Float64]) -> Float64:
    if len(xs) == 0:
        return 0.0
    for i in range(1, len(xs)):  # insertion sort: tens of values
        var j = i
        while j > 0 and xs[j - 1] > xs[j]:
            var t = xs[j]
            xs[j] = xs[j - 1]
            xs[j - 1] = t
            j -= 1
    return xs[len(xs) // 2]


def pad(s: String, w: Int) -> String:
    var out = s
    while len(out.codepoints()) < w:
        out += " "
    return out^


def main() raises:
    var home = getenv("HOME")
    var args = argv()
    var dir = String(args[1]) if len(args) > 1 else home + "/.cache/noeira/d1/room_clips"
    var omni_dir = String(args[2]) if len(args) > 2 else home + "/.cache/noeira/local-ai/d1/d1-Omni-2B"
    var b2_dir = String(args[3]) if len(args) > 3 else home + "/.cache/noeira/local-ai/d1/d1-2B"
    var out_path = String(args[4]) if len(args) > 4 else dir + "/results_" + ("bf16" if LF_BF16 else "fp32") + ".tsv"
    var clips = read_manifest(dir)
    print("d1-Omni heard vs Parakeet -> d1 written:", len(clips), "clips,", "bf16" if LF_BF16 else "fp32")

    var pk = List[String]()
    var pk_ms = List[Float64]()
    transcribe_all(dir, clips, pk, pk_ms)
    var pk_wrong = 0
    for i in range(len(clips)):
        if words(pk[i]) != words(clips[i].text):
            pk_wrong += 1
    print("  parakeet done:", pk_wrong, "of", len(clips), "transcripts differ from the text (case and punctuation ignored)")

    var res = List[List[Answer]]()
    for _ in range(N_COLS):
        res.append(List[Answer]())
    run_d1(omni_dir, True, dir, clips, pk, res)
    print("  d1-Omni done")
    run_d1(b2_dir, False, dir, clips, pk, res)
    print("  d1-2B done")

    # per clip: a TSV for the record, and the lines where the columns disagree
    var tsv = String("clip\tlang\ttext\tparakeet\texpected_addressed\texpected_dest\tparakeet_ms")
    for c in range(N_COLS):
        var n = col_name(c).replace(" ", "_")
        tsv += "\t" + n + "_choice\t" + n + "_p_expected\t" + n + "_p_addressed\t" + n + "_ms"
    tsv += "\n"
    print("\n  clip       expected   heard / omni-pk / 2B-pk / omni-truth / 2B-truth (chosen place, * = walks there)")
    for i in range(len(clips)):
        ref c = clips[i]
        tsv += c.name + "\t" + c.lang + "\t" + c.text + "\t" + pk[i] + "\t" + c.addressed + "\t" + c.dest + "\t" + String(pk_ms[i])
        var line = String("  ") + c.name + "  " + c.dest + " (" + c.addressed + ")  "
        for col in range(N_COLS):
            ref a = res[col][i]
            tsv += "\t" + a.choice + "\t" + String(a.p_expected) + "\t" + String(a.p_addressed) + "\t" + String(a.ms)
            var mark = "*" if walks_to(a) != "nowhere" else ""
            line += ("" if col == 0 else " / ") + a.choice + mark + ("" if a.choice == c.dest else "(x)")
        tsv += "\n"
        print(line + "   '" + c.text + "'" + ("" if words(pk[i]) == words(c.text) else " -> '" + pk[i] + "'"))
    var f = open(out_path, "w")
    f.write(tsv)
    f.close()

    # summary per column and language
    var langs: List[String] = ["fr", "en", "all"]
    print("\n  column            lang   destination  addressed  action   mean p(expected)  median ms")
    for col in range(N_COLS):
        for ref lg in langs:
            var nd = 0
            var ok_d = 0
            var na = 0
            var ok_a = 0
            var ok_act = 0
            var pe = 0.0
            var ms = List[Float64]()
            for i in range(len(clips)):
                ref c = clips[i]
                if lg != "all" and c.lang != lg:
                    continue
                ref a = res[col][i]
                nd += 1
                if a.choice == c.dest:
                    ok_d += 1
                if walks_to(a) == c.dest:
                    ok_act += 1
                pe += a.p_expected
                if c.addressed != "?":
                    na += 1
                    if (a.p_addressed >= 0.5) == (c.addressed == "yes"):
                        ok_a += 1
                var t = a.ms
                if col == C_2B_PK or col == C_OMNI_PK:
                    t += pk_ms[i]  # the cascade: transcript first
                ms.append(t)
            print(
                "  " + pad(col_name(col), 18) + pad(lg, 7) + pad(pct(ok_d, nd), 13) + pad(pct(ok_a, na), 11)
                + pad(pct(ok_act, nd), 9) + pad(String(round(pe / Float64(max(nd, 1)), 3)), 18)
                + String(round(median(ms^), 1))
            )
    print("\n  latency: parakeet column times include Parakeet (median", round(median(pk_ms.copy()), 1), "ms); truth columns are d1 alone")
    print("  wrote", out_path)
