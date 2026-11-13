# +--------------------------------------------------------------------------+ #
# | BB-1: does the robot's reply follow its state? (LFM3-1.7B, in process)
# +--------------------------------------------------------------------------+ #
"""The brain's grounded-answers bench (docs/ROBOT_BRAIN_PLAN.md, step B1).

    pixi run -e nvidia mojo build -D LFM3_BF16=1 -I . benchmarks/bench_brain_grounded_gpu.mojo -o build/bench_brain
    ./build/bench_brain [out.tsv] [LFM3-1.7B dir] [tests/brain] [one prompt's name]

Every BB-1 item (`tools/brain/make_bb1.py`: a question a person asks, in French
or English, whose answer is in the robot's state) is asked of the reply model
under four system prompts:

    fixed       the room demo's hand-written sentence (step 4's prompt): no state
    lines       `reply_view(ws)`: the state as short fact lines; language not given
    sentences   `reply_view(ws, REPLY_VIEW_SENTENCES)`: the same facts as sentences
    lines+lang  `lines` with `ws.language` set to the question's (what the room
                demo does: the recogniser's language is pinned)

as the voice loop asks it: one user message, Liquid's sampler with its fixed
seed, 120 tokens at most, int8 weights (`Lfm3ChatService`'s `native`), and the
reply taken as the robot would SPEAK it (`answer_of`: after a stray
`</think>`; `spoken_text`: no emoji). Each reply is scored by the item's rule
(`noeira.brain.bb1.reply_passes`).

Printed per prompt: passes per category, the `task_*` ones (B2: what the
executive's record lets the robot answer) and the `held_*` ones apart (questions
never looked at while the view's wording was chosen); the mutant rows (`where_mut`: two
places swapped in the state) both ways, on the swapped room's truth (the
reply follows the state) and on the room's truth before the swap (must now
fail); replies in the language of their question; raw replies that carried a
`</think>` or an emoji before cleaning; the system prompt's length in tokens
and the time to the first token and to the whole reply. `out.tsv` gets every
reply (`tests/brain/bb1_rescore.mojo` scores such a file again, without the
model, after a rule changes).
"""

from std.os import getenv
from std.sys import argv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from noeira.brain import Body, reply_view, REPLY_VIEW_LINES, REPLY_VIEW_SENTENCES
from noeira.brain.bb1 import BB1Set, reply_passes, reply_in_language
from noeira.llm.lfm3 import Lfm3Chat, ChatSettings, answer_of
from noeira.text.spoken import spoken_text


comptime FIXED = (
    "You are a small humanoid robot in a furnished room with a fridge,"
    " a door, a table and a sofa. Reply in ONE short sentence, in the"
    " language you were addressed in. You can walk to those places,"
    " turn, squat and raise your arms, but you cannot pick things up,"
    " open the door or sit down. Be warm and brief."
)
"""`projects/g1/scripts/room_demo.mojo`'s `chat_sys` before B1, verbatim."""

comptime CATS = "where_dir,where_dist,nearest,doing,before,can_yes,can_no,unknown,self,chat,task_next,task_state,held_can_no,held_can_yes,held_unknown,where_mut"


def ms(t0: Int) -> Float64:
    return Float64(perf_counter_ns() - t0) / 1e6


def flat(s: String) -> String:
    return s.replace("\t", " ").replace("\n", " ")


def main() raises:
    var args = argv()
    var out_path = String(args[1]) if len(args) > 1 else String("")
    var dir = String(args[2]) if len(args) > 2 else getenv("HOME") + "/.cache/noeira/local-ai/lfm3/LFM3-1.7B"
    var set_dir = String(args[3]) if len(args) > 3 else String("tests/brain")
    var only = String(args[4]) if len(args) > 4 else String("")
    var set = BB1Set.load(set_dir)
    print("BB-1:", len(set.items), "items over", len(set.scenarios), "scenarios; LFM3-1.7B int8, Liquid's sampler")
    var chat = Lfm3Chat.load(DeviceContext(), dir, 4096, True)
    var settings = ChatSettings()
    settings.max_tokens = 120

    var cats = List[String]()
    for c in String(CATS).split(","):
        cats.append(String(c))
    var arms: List[String] = ["fixed", "lines", "sentences", "lines+lang"]
    var tsv = String("arm\tid\tsid\tcat\tlang\tpass\talt_pass\tsys_tokens\tfirst_ms\ttotal_ms\tquestion\treply\traw\n")

    for a in range(len(arms)):
        if only != "" and arms[a] != only:
            continue
        var n_pass = List[Int](length=len(cats), fill=0)
        var n_all = List[Int](length=len(cats), fill=0)
        var alt_pass = 0
        var in_lang = 0
        var thinks = 0
        var cleaned = 0
        var sys_tokens = 0
        var first = 0.0
        var total = 0.0
        for item in set.items:
            ref sc = set.scenarios[set.scenario(item.sid)]
            var ws = sc.state(Body.g1_room("Noé"), item.lang if a == 3 else String(""))
            var system = String(FIXED)
            if a == 1 or a == 3:
                system = reply_view(ws, REPLY_VIEW_LINES).text()
            elif a == 2:
                system = reply_view(ws, REPLY_VIEW_SENTENCES).text()
            chat.reset(system)
            var n_sys = len(chat.prompt_ids())
            chat.user(item.question)
            var t0 = perf_counter_ns()
            chat.start(settings)
            var t_first = ms(t0)
            while chat.next():
                pass
            var t_all = ms(t0)
            var raw = chat.reply_text()
            var answer = answer_of(raw)
            var reply = spoken_text(answer)
            if raw.find("</think>") >= 0:
                thinks += 1
            if reply != answer:
                cleaned += 1
            if reply_in_language(reply, item.lang):
                in_lang += 1
            var ok = reply_passes(reply, item.must, item.forbid)
            var alt_ok = False
            var ci = 0
            for k in range(len(cats)):
                if cats[k] == item.cat:
                    ci = k
            n_all[ci] += 1
            if ok:
                n_pass[ci] += 1
            if item.cat == "where_mut":
                alt_ok = reply_passes(reply, item.alt_must, item.alt_forbid)
                if alt_ok:
                    alt_pass += 1
            sys_tokens += n_sys
            first += t_first
            total += t_all
            tsv += (arms[a] + "\t" + item.id + "\t" + item.sid + "\t" + item.cat + "\t" + item.lang + "\t"
                    + ("1" if ok else "0") + "\t" + ("1" if alt_ok else "0") + "\t" + String(n_sys) + "\t"
                    + String(Int(t_first)) + "\t" + String(Int(t_all)) + "\t" + flat(item.question) + "\t" + flat(reply) + "\t" + flat(raw) + "\n")
        var n = len(set.items)
        var grounded = 0
        var grounded_all = 0
        var held = 0
        var held_all = 0
        var tasks = 0
        var tasks_all = 0
        var line = String("")
        for k in range(len(cats)):
            line += "  " + cats[k] + " " + String(n_pass[k]) + "/" + String(n_all[k])
            if cats[k].startswith("held_"):
                held += n_pass[k]
                held_all += n_all[k]
            elif cats[k].startswith("task_"):
                tasks += n_pass[k]
                tasks_all += n_all[k]
            elif cats[k] != "where_mut":
                grounded += n_pass[k]
                grounded_all += n_all[k]
        var mi = len(cats) - 1
        print(arms[a] + ":", grounded, "/", grounded_all, "; tasks", tasks, "/", tasks_all, "; held out", held, "/", held_all, "|" + line)
        print("   mutant rows: right for the swapped room", n_pass[mi], "/", n_all[mi],
              "; still right for the room before the swap", alt_pass, "/", n_all[mi])
        print("   in the question's language", in_lang, "/", n, "; raw replies with </think>", thinks, ", with emoji or markup", cleaned)
        print("   system prompt", sys_tokens // n, "tokens; first token", Int(first / Float64(n)), "ms; reply",
              Int(total / Float64(n)), "ms (means)")
    if out_path != "":
        with open(out_path, "w") as f:
            f.write(tsv)
        print("replies written to", out_path)
