"""Fine-tuning d1 in Mojo, end to end: LoRA on the LFM3 backbone moves an answer.

    build on the Orin: build-jetson tests/llm/test_d1_finetune_smoke_gpu.mojo
    ./test_d1_finetune_smoke_gpu [d1 dir]

The room state of the reference (`room_en`): d1 answers "Where should the robot
walk?" with fridge (0.97). Fresh adapters (PEFT init: B = 0, so the first
answers must be EXACTLY the base model's), then AdamW on ONE row whose labels
say sofa for that question and keep the other two answers' picks. After the
steps: the loss has fallen, the destination is sofa, and the yes/no and the
score keep their picks. A smoke test of the training loop (`train.mojo`), not a
recipe: the gradients themselves are gated against torch in
`test_lfm3_backward_gpu.mojo`.
"""

from std.os import getenv
from std.sys import argv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from noeira.io.json import parse_json
from noeira.llm.d1 import D1, parse_questions, pack_rows, readout_ids
from noeira.llm.d1.prompt import prefix_text, suffix_text, py_json_dumps
from noeira.llm.lfm3.train import LoraAdamW, train_step


comptime STATE = """{"robot": {"at": "door", "facing": "kitchen", "doing": "standing"}, "places": {"fridge": "3.1 m ahead, slightly left", "door": "here", "table": "2.0 m to the right", "sofa": "4.5 m behind"}, "heard": "Can you go to the fridge and wait there?", "recently": ["walked to the door", "heard: hello robot"]}"""
comptime QUESTIONS = """{"addressed": {"type": "noul", "instructions": "Is the speaker asking the robot to do something?"}, "destination": {"type": "choice", "instructions": "Where should the robot walk?", "criteria": {"fridge": "walk to the fridge", "door": "walk to the door", "table": "walk to the table", "sofa": "walk to the sofa", "nowhere": "do not walk anywhere"}}, "urgency": {"type": "score", "instructions": "How urgent is the request?", "criteria": ["not urgent", "somewhat urgent", "very urgent"]}}"""
comptime STEPS = 12


def _bytes(s: String) -> List[UInt8]:
    var b = List[UInt8]()
    for x in s.as_bytes():
        b.append(x)
    return b^


def answers(mut d1: D1) raises -> Tuple[String, String, Float64, Float64]:
    """(destination, P(yes) as text, its confidence, the score level)."""
    var out = parse_json(_bytes(d1.system_one(STATE, True, QUESTIONS)))
    var a = out.field(out.root(), "answers")
    var dest = out.field(a, "destination")
    return (out.string(out.field(dest, "choice")),
            String(out.number(out.field(out.field(a, "addressed"), "noul"))),
            out.number(out.field(dest, "confidence")),
            out.number(out.field(out.field(a, "urgency"), "score")))


def check(mut fails: Int, ok: Bool, what: String):
    if ok:
        print("  PASS ", what)
    else:
        print("  FAIL ", what)
        fails += 1


def main() raises:
    var args = argv()
    var home = getenv("HOME")
    var dir = String(args[1]) if len(args) > 1 else home + "/.cache/noeira/local-ai/d1/d1-2B"
    var fails = 0
    var d1 = D1.load(DeviceContext(), dir)
    var base = answers(d1)
    print("  base: destination", base[0], "(", base[2], "), P(yes)", base[1], ", urgency level", base[3])

    d1.lm.init_lora(16, 32.0, seed=0)
    var fresh = answers(d1)
    check(fails, fresh[0] == base[0] and fresh[2] == base[2] and fresh[1] == base[1],
          "fresh adapters (B = 0) leave every answer exactly as the base model's")

    # the training row and its labels: yes, SOFA, level 0
    var sd = parse_json(_bytes(STATE))
    var qd = parse_json(_bytes(QUESTIONS))
    var qs = parse_questions(qd, qd.root())
    var prefix = prefix_text(d1.tok, py_json_dumps(sd, sd.root()))
    var suffixes = List[String]()
    for q in qs:
        suffixes.append(suffix_text(d1.tok, q))
    var rows = pack_rows(d1.tok, prefix, suffixes)
    var targets = List[Int]()
    targets.append(readout_ids(d1.tok, qs[0])[0][0])   # yes
    targets.append(readout_ids(d1.tok, qs[1])[3][0])   # sofa: option 3's code
    targets.append(readout_ids(d1.tok, qs[2])[0][0])   # level 0
    var opt = LoraAdamW(d1.lm, lr=1e-3)
    var first = 0.0
    var last = 0.0
    var t0 = perf_counter_ns()
    for step in range(STEPS):
        var loss = train_step(d1.lm, opt, rows[0].ids, rows[0].groups, rows[0].state_len, rows[0].slots, targets)
        if step == 0:
            first = loss
        last = loss
        print("    step", step, "loss", loss)
    d1.lm.ctx.synchronize()
    print("  ", STEPS, "steps in", (perf_counter_ns() - t0) // 1_000_000, "ms (fp32,", len(rows[0].ids), "tokens)")
    var tuned = answers(d1)
    print("  tuned: destination", tuned[0], "(", tuned[2], "), P(yes)", tuned[1], ", urgency level", tuned[3])
    check(fails, last < 0.25 * first, String("the loss fell: ") + String(first) + " -> " + String(last))
    check(fails, tuned[0] == "sofa", String("the destination moved to the label: ") + tuned[0])
    check(fails, tuned[3] == base[3] and Float64(tuned[1]) > 0.5, "the other two answers keep their picks")
    if fails > 0:
        raise Error(String(fails) + " FAILED")
    print("ALL PASS")
