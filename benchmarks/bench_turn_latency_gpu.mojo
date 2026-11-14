# +--------------------------------------------------------------------------+ #
# | A spoken turn's reply: where its time goes, what the prefix cache takes back
# +--------------------------------------------------------------------------+ #
"""The reply model's side of a turn (docs/NN_MODEL_PORTS_PLAN.md §12.8, S0 and S1).

    pixi run -e nvidia mojo build -D LFM3_BF16=1 -I . benchmarks/bench_turn_latency_gpu.mojo -o build/bench_turn
    ./build/bench_turn [LFM3-1.7B dir] [tests/brain] [items, 0 = all] [reps]

1. A chunk's cost by its length, int8 weights, after a 300-token prompt: 1 to
   4 tokens (the GEMV: the weights read once for the rows), then 5 to 64 (every
   weight dequantized for the GEMM). This is the floor of a turn's first token
   (the user's sentence is a chunk of 20 to 40 tokens), and what verifying k
   drafted tokens in one pass would cost (speculative decoding, S4). The LM
   head for 1 to 4 rows is timed apart.

2. BB-1's questions (`tests/brain/bb1_items.tsv`) asked as the voice loop asks
   them: the system prompt is the robot's state (`reply_view`, the question's
   language), one user message, Liquid's sampler, 120 tokens at most. Four ways:

       off      no prefix cache: each prompt computed whole (what the room demo
                paid once the prompt came from the state)
       same     the cache on, the set's order: a scenario's questions follow each
                other, so the state does not change between two turns
       mixed    the cache on, each question after one of ANOTHER scenario: the
                state changed since the last reply
       ahead    `mixed`, with the prompt given before the question (`prepare`,
                not timed: the voice loop calls it while the person speaks)

   Per way: the prompt's tokens and how many were found in the cache, the time
   to the first token, the reply's time, its tokens and the time a token; and
   BB-1's score, with the replies that differ from `off`'s (a reused prefix is
   not bit-identical to a recomputed one: `Lfm3Chat.keep_system`).

The decision model's side is `benchmarks/bench_d1_system_one_gpu.mojo` (time
against the row's tokens) and the voice loop's own notes (`(jev 457 + 641 ms,
512 + 890 tokens)`).
"""

from std.os import getenv
from std.sys import argv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from noeira.brain import Body, reply_view, REPLY_VIEW_LINES
from noeira.brain.bb1 import BB1Set, reply_passes
from noeira.llm.lfm3 import Lfm3Chat, ChatSettings, answer_of, forward_cached
from noeira.text.spoken import spoken_text


def ms(t0: Int) -> Float64:
    return Float64(perf_counter_ns() - t0) / 1e6


def median(var xs: List[Float64]) -> Float64:
    for i in range(1, len(xs)):
        var j = i
        while j > 0 and xs[j - 1] > xs[j]:
            var t = xs[j]
            xs[j] = xs[j - 1]
            xs[j - 1] = t
            j -= 1
    return xs[len(xs) // 2]


def r1(x: Float64) -> String:
    return String(round(x, 1))


def chunk_costs(mut chat: Lfm3Chat, system: String, reps: Int) raises:
    """Part 1: a chunk of n tokens on a cache that already holds `system`."""
    chat.keep_system = False
    chat.reset(system)
    chat.user("Bonjour ! Où est la table, et qu'est-ce que tu es en train de faire ?")
    var prompt = chat.prompt_ids()
    chat.cache.reset()
    _ = forward_cached(chat.lm, prompt, chat.cache)
    chat.lm.ctx.synchronize()
    var base = chat.cache.mark(chat.lm)
    print("  a chunk after", base.pos, "tokens (forward only, int8):")
    var sizes: List[Int] = [1, 2, 3, 4, 5, 8, 16, 32, 64]
    for n in sizes:
        var ids = List[Int]()
        for i in range(n):
            ids.append(prompt[(7 + i) % len(prompt)])
        var times = List[Float64]()
        for r in range(reps + 2):
            chat.cache.rewind(chat.lm, base)
            chat.lm.ctx.synchronize()
            var t0 = perf_counter_ns()
            _ = forward_cached(chat.lm, ids, chat.cache)
            chat.lm.ctx.synchronize()
            if r >= 2:
                times.append(ms(t0))
        var m = median(times^)
        print("     ", n, "tokens:", r1(m), "ms =", r1(m / Float64(n)), "ms a token")
    var v = chat.lm.cfg.vocab
    for n in range(1, 5):
        var rows = chat.lm.slot_rows(n)
        var logits = chat.lm.new(n * v)
        var times = List[Float64]()
        for r in range(reps + 2):
            var t0 = perf_counter_ns()
            chat.lm.lm_head(rows, logits, n)
            chat.lm.ctx.synchronize()
            if r >= 2:
                times.append(ms(t0))
        print("      the LM head for", n, "rows:", r1(median(times^)), "ms")
    chat.reset(String(""))


def main() raises:
    var args = argv()
    var dir = String(args[1]) if len(args) > 1 else getenv("HOME") + "/.cache/noeira/local-ai/lfm3/LFM3-1.7B"
    var set_dir = String(args[2]) if len(args) > 2 else String("tests/brain")
    var limit = Int(String(args[3])) if len(args) > 3 else 0
    var reps = Int(String(args[4])) if len(args) > 4 else 10
    var set = BB1Set.load(set_dir)
    var chat = Lfm3Chat.load(DeviceContext(), dir, 4096, True)
    var settings = ChatSettings()
    settings.max_tokens = 120
    print("A turn's reply: LFM3-1.7B int8, Liquid's sampler;", len(set.items), "BB-1 items over", len(set.scenarios), "scenarios")

    # the prompts, once: the demo's view, in the question's language
    var systems = List[String]()
    for item in set.items:
        ref sc = set.scenarios[set.scenario(item.sid)]
        var ws = sc.state(Body.g1_room("Noé"), item.lang)
        systems.append(reply_view(ws, REPLY_VIEW_LINES).text())

    chunk_costs(chat, systems[0], reps)

    # the two orders: the set's, and one question of each scenario in turn
    var same = List[Int]()
    for i in range(len(set.items)):
        same.append(i)
    var by_sc = List[List[Int]]()
    for s in range(len(set.scenarios)):
        by_sc.append(List[Int]())
    for i in range(len(set.items)):
        by_sc[set.scenario(set.items[i].sid)].append(i)
    var mixed = List[Int]()
    var k = 0
    while len(mixed) < len(set.items):
        for s in range(len(by_sc)):
            if k < len(by_sc[s]):
                mixed.append(by_sc[s][k])
        k += 1
    var n = len(set.items) if limit <= 0 else min(limit, len(set.items))

    var ways: List[String] = ["off", "same", "mixed", "ahead"]
    var off_reply = List[String]()
    for i in range(len(set.items)):
        off_reply.append(String(""))
    for w in range(len(ways)):
        chat.keep_system = w > 0
        chat.reset(String(""))
        var sys_tok = 0
        var cached = 0
        var first = 0.0
        var total = 0.0
        var out_tok = 0
        var per_tok = 0.0
        var passes = 0
        var differ = 0
        for j in range(n):
            var i = same[j] if w <= 1 else mixed[j]
            ref item = set.items[i]
            if w == 3:
                chat.prepare(systems[i])
                chat.lm.ctx.synchronize()
            var t0 = perf_counter_ns()
            chat.reset(systems[i])
            chat.user(item.question)
            chat.start(settings)
            var t_first = ms(t0)
            while chat.next():
                pass
            var t_all = ms(t0)
            var reply = spoken_text(answer_of(chat.reply_text()))
            sys_tok += chat.head_len
            cached += chat.reused
            first += t_first
            total += t_all
            out_tok += len(chat.reply)
            if len(chat.reply) > 1:
                per_tok += (t_all - t_first) / Float64(len(chat.reply) - 1)
            if reply_passes(reply, item.must, item.forbid):
                passes += 1
            if w == 0:
                off_reply[i] = reply.copy()
            elif reply != off_reply[i] and off_reply[i] != "":
                differ += 1
        var f = Float64(n)
        print("  " + ways[w] + ": prompt", sys_tok // n, "tokens,", cached // n, "cached; first token", r1(first / f), "ms; reply",
              r1(total / f), "ms,", r1(Float64(out_tok) / f), "tokens out,", r1(per_tok / f), "ms a token; BB-1", passes, "/", n,
              "; replies that differ from off:", differ)
