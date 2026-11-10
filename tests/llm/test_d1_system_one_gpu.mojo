"""d1's System One in Mojo (`noeira.llm.d1`) against d1-2B's own code.

    ./test_d1_system_one_gpu [--prompt-only] [d1 dir] [ref dir]
      defaults: ~/.cache/noeira/local-ai/d1/d1-2B (or ~/.cache/noeira/d1 with --prompt-only)
                and ~/.cache/noeira/d1/ref

The three cases of `tools/d1/dump_d1_reference.py`, asked again here:

1. PROMPTS (the tokenizer only; `--prompt-only` stops here, e.g. on the Mac):
   the state prefix and every question's suffix equal d1's text EXACTLY
   (`prompt.json`), and the packed rows equal d1's — ids, each token's
   question, the state's length, the answer slots (`row<r>_*`).
2. ANSWERS (the model; the Orin): `system_one`'s answers equal `answers.json` —
   the same choice and score level, every probability within 1e-4, the same
   input-token count.
"""

from std.os import getenv
from std.sys import argv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from noeira.io.json import load_json, parse_json, JsonDoc, J_NUMBER, J_STRING, J_OBJECT
from noeira.io.npy import load_npy_f64
from noeira.text.bpe import ByteLevelBPE
from noeira.llm.d1 import D1, parse_questions, pack_rows
from noeira.llm.d1.prompt import prefix_text, suffix_text, py_json_dumps
from noeira.llm.lfm3.model import LF_BF16


comptime PROB_BOUND: Float64 = 3e-3 if LF_BF16 else 1e-4
"""fp32: measured <= 1.8e-6. LFM3_BF16: measured <= 8.1e-4 (the Orin,
2026-11-10), against d1's own fp32 answers; the picks are equal either way."""

comptime ROOM_STATE = """{"robot": {"at": "door", "facing": "kitchen", "doing": "standing"}, "places": {"fridge": "3.1 m ahead, slightly left", "door": "here", "table": "2.0 m to the right", "sofa": "4.5 m behind"}, "heard": "Can you go to the fridge and wait there?", "recently": ["walked to the door", "heard: hello robot"]}"""
comptime ROOM_Q = """{"addressed": {"type": "noul", "instructions": "Is the speaker asking the robot to do something?"}, "destination": {"type": "choice", "instructions": "Where should the robot walk?", "criteria": {"fridge": "walk to the fridge", "door": "walk to the door", "table": "walk to the table", "sofa": "walk to the sofa", "nowhere": "do not walk anywhere"}}, "urgency": {"type": "score", "instructions": "How urgent is the request?", "criteria": ["not urgent", "somewhat urgent", "very urgent"]}}"""
comptime FR_STATE = "Le robot entend : « va vers le canapé, s'il te plaît »."
comptime FR_Q = """{"move": {"type": "choice", "instructions": "Quelle action le robot doit-il faire ?", "criteria": {"canape": "marcher vers le canapé", "porte": "marcher vers la porte", "accroupi": "s'accroupir", "rien": "ne rien faire"}}}"""
comptime MANY_STATE = "The user says: please raise your right hand and then turn left slowly."


def many_questions() -> String:
    var texts: List[String] = [
        "Does the user ask for an arm movement?", "Does the user mention the left hand?",
        "Does the user ask the robot to turn?", "Is a speed mentioned?", "Is the request polite?",
        "Does the user ask the robot to walk?", "Is a place named?", "Does the user ask for two actions?",
        "Is the right hand mentioned?", "Does the user ask the robot to sit?", "Is the user angry?",
        "Does the user ask to stop?", "Is the order of actions given?", "Does the user ask a question?",
        "Is a direction given?", "Does the user greet the robot?", "Does the user ask to squat?",
        "Is the user talking to someone else?",
    ]
    var s = String("{")
    for i in range(len(texts)):
        s += String('"q') + ("0" if i < 10 else "") + String(i) + '": {"type": "noul", "instructions": "' + texts[i] + '"}'
        s += ", " if i + 1 < len(texts) else ""
    return s + "}"


def check(mut fails: Int, ok: Bool, what: String):
    if ok:
        print("  PASS ", what)
    else:
        print("  FAIL ", what)
        fails += 1


def _bytes(s: String) -> List[UInt8]:
    var b = List[UInt8]()
    for x in s.as_bytes():
        b.append(x)
    return b^


def same_ints(a: List[Int], b: List[Float64]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != Int(b[i]):
            return False
    return True


def main() raises:
    var args = argv()
    var prompt_only = False
    var pos = List[String]()
    for i in range(1, len(args)):
        if String(args[i]) == "--prompt-only":
            prompt_only = True
        else:
            pos.append(String(args[i]))
    var home = getenv("HOME")
    var model_dir = pos[0] if len(pos) > 0 else (home + "/.cache/noeira/d1" if prompt_only else home + "/.cache/noeira/local-ai/d1/d1-2B")
    var ref_dir = pos[1] if len(pos) > 1 else home + "/.cache/noeira/d1/ref"
    var names: List[String] = ["room_en", "single_fr", "many_noul"]
    var states: List[String] = [ROOM_STATE, FR_STATE, MANY_STATE]
    var is_json: List[Bool] = [True, False, False]
    var questions: List[String] = [ROOM_Q, FR_Q, many_questions()]
    var fails = 0

    print("d1 System One in Mojo vs d1-2B's own code", "(LFM3_BF16)" if LF_BF16 else "(fp32)")
    var tok = ByteLevelBPE.load(model_dir + "/tokenizer.json")
    # 1. prompts and rows
    for c in range(3):
        var pj = load_json(ref_dir + "/" + names[c] + "/prompt.json")
        var block: String
        if is_json[c]:
            var sd = parse_json(_bytes(states[c]))
            block = py_json_dumps(sd, sd.root())
        else:
            block = states[c]
        var prefix = prefix_text(tok, block)
        check(fails, prefix == pj.string(pj.field(pj.root(), "prefix")), names[c] + ": the state prefix, text exact")
        var qd = parse_json(_bytes(questions[c]))
        var qs = parse_questions(qd, qd.root())
        var sufs = pj.field(pj.root(), "suffixes")
        var suffixes = List[String]()
        var ok = pj.size(sufs) == len(qs)
        for i in range(len(qs)):
            var s = suffix_text(tok, qs[i])
            suffixes.append(s)
            if ok and s != pj.string(pj.at(sufs, i)):
                ok = False
                print("     question", i, ":", repr(s))
        check(fails, ok, names[c] + ": " + String(len(qs)) + " question suffixes, text exact")
        var rows = pack_rows(tok, prefix, suffixes)
        var rows_ok = True
        for r in range(len(rows)):
            var base = ref_dir + "/" + names[c] + "/row" + String(r)
            var meta = load_json(base + "_meta.json")
            var sl = meta.field(meta.root(), "slots")
            var slots_ok = meta.size(sl) == len(rows[r].slots)
            for k in range(len(rows[r].slots)):
                if slots_ok and meta.integer(meta.at(meta.at(sl, k), 0)) != rows[r].slots[k]:
                    slots_ok = False
            var this_ok = (same_ints(rows[r].ids, load_npy_f64(base + "_ids.npy"))
                           and same_ints(rows[r].groups, load_npy_f64(base + "_question.npy"))
                           and rows[r].state_len == meta.integer(meta.field(meta.root(), "state_len"))
                           and slots_ok)
            if not this_ok:
                print("     row", r, ":", len(rows[r].ids), "ids, state", rows[r].state_len)
            rows_ok = rows_ok and this_ok
        check(fails, rows_ok, names[c] + ": " + String(len(rows)) + " packed row(s): ids, groups, state length, slots exact")
    if prompt_only:
        if fails > 0:
            raise Error(String(fails) + " FAILED")
        print("ALL PASS (prompts only)")
        return

    # 2. answers
    var t0 = perf_counter_ns()
    var d1 = D1.load(DeviceContext(), model_dir)
    print("  d1 loaded in", (perf_counter_ns() - t0) // 1_000_000, "ms; temperature", d1.temperature)
    for c in range(3):
        var t1 = perf_counter_ns()
        var got_s = d1.system_one(states[c], is_json[c], questions[c])
        var ms = Float64(perf_counter_ns() - t1) / 1e6
        var got = parse_json(_bytes(got_s))
        var want = load_json(ref_dir + "/" + names[c] + "/answers.json")
        var ga = got.field(got.root(), "answers")
        var wa = want.field(want.root(), "answers")
        var worst = 0.0
        var same = got.size(ga) == want.size(wa)
        for i in range(want.size(wa)):
            var name = want.key_at(wa, i)
            var w = want.at(wa, i)
            var g = got.field(ga, name)
            if g < 0:
                same = False
                continue
            var kind = want.string(want.field(w, "type"))
            if kind == "noul":
                worst = max(worst, abs(got.number(got.field(g, "noul")) - want.number(want.field(w, "noul"))))
                continue
            if kind == "choice" and got.string(got.field(g, "choice")) != want.string(want.field(w, "choice")):
                same = False
            if kind == "score" and got.number(got.field(g, "score")) != want.number(want.field(w, "score")):
                same = False
            var wp = want.field(w, "probabilities")
            var gp = got.field(g, "probabilities")
            for k in range(want.size(wp)):
                var gk = got.field(gp, want.key_at(wp, k))
                if gk < 0:
                    same = False
                    continue
                worst = max(worst, abs(got.number(gk) - want.number(want.at(wp, k))))
        var wu = want.integer(want.field(want.field(want.root(), "usage"), "input_tokens"))
        var gu = got.integer(got.field(got.field(got.root(), "usage"), "input_tokens"))
        check(fails, same and worst < PROB_BOUND and wu == gu,
              names[c] + ": " + String(want.size(wa)) + " answers, same picks, worst probability |d| = " + String(worst)
              + ", " + String(gu) + " tokens (" + String(ms) + " ms)")
    if fails > 0:
        raise Error(String(fails) + " FAILED")
    print("ALL PASS")
