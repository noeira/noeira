# +--------------------------------------------------------------------------+ #
# | LFM3 chat speed: prefill, a decoded token's parts, the first word of a turn
# +--------------------------------------------------------------------------+ #
"""Where a chat token's time goes (docs/NN_MODEL_PORTS_PLAN.md §12.7 step 3).

    pixi run -e nvidia mojo build -D LFM3_BF16=1 -I . benchmarks/bench_lfm3_chat_gpu.mojo -o build/bench_lfm3_chat
    ./build/bench_lfm3_chat [LFM3-1.7B dir] [reps]

1. Decode, one token, split: the forward's ENQUEUE (host time to issue it), the
   forward to completion, the LM head + logits download, the host choice
   (penalty + greedy or top-k). Median of `reps` tokens at a ~130-token context.
2. Prefill: chunks of 128, 512 and 1024 tokens on an empty cache.
3. A turn's first token after a long conversation: the time from `start` to the
   first token with ~1000 tokens already in the cache (the new turn's tokens only
   are computed).
4. A real reply, sampled with Liquid's settings, thinking OFF then ON: the time
   to the first token of the ANSWER (after `</think>` when thinking), the
   reply's length and tokens/s.
"""

from std.os import getenv
from std.sys import argv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from noeira.llm.lfm3 import Lfm3Chat, ChatSettings, forward_cached, last_logits, choose
from noeira.llm.lfm3.model import LF_BF16, LF_INT8


comptime SYSTEM = (
    "You are Noé, a small humanoid robot in a living room. You speak with the people around you, briefly and "
    "kindly, in the language they use. Room: the fridge is 3.1 m ahead, the door 3.8 m to the left, the table "
    "2.0 m to the right, the sofa 4.5 m behind. You are standing in the middle of the room."
)


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
    return String(round(x, 2))


def main() raises:
    var args = argv()
    var dir = String(args[1]) if len(args) > 1 else getenv("HOME") + "/.cache/noeira/local-ai/lfm3/LFM3-1.7B"
    var reps = Int(String(args[2])) if len(args) > 2 else 40
    print("LFM3 chat speed", "(LFM3_BF16)" if LF_BF16 else "(fp32)", "(LFM3_INT8)" if LF_INT8 else "")
    var t_load = perf_counter_ns()
    var chat = Lfm3Chat.load(DeviceContext(), dir, 2048)
    var v = chat.lm.cfg.vocab
    print("  loaded in", Int(ms(t_load)), "ms: weights", r1(Float64(chat.lm.bf16_bytes()) / 1e9), "GB bf16 +",
          r1(Float64(chat.lm.q8_bytes()) / 1e9), "GB int8")

    # 1. one decoded token, in parts
    chat.reset(SYSTEM)
    chat.user("Bonjour ! Raconte-moi une histoire sur un robot qui apprend à danser.")
    var prompt = chat.prompt_ids()
    chat.cache.reset()
    _ = forward_cached(chat.lm, prompt, chat.cache)
    chat.lm.ctx.synchronize()
    var enq = List[Float64]()
    var fwd = List[Float64]()
    var head = List[Float64]()
    var host = List[Float64]()
    var tok = 1000  # any id: the time does not depend on it
    var greedy = ChatSettings.greedy_with(1.1, 1)
    var sampled = ChatSettings()
    var rng = UInt64(7)
    for r in range(reps + 3):
        var one: List[Int] = [tok]
        var t0 = perf_counter_ns()
        var h = forward_cached(chat.lm, one, chat.cache)
        var e = ms(t0)
        chat.lm.ctx.synchronize()
        var f = ms(t0)
        var t1 = perf_counter_ns()
        var lg = last_logits(chat.lm, h, 1)
        var hd = ms(t1)
        var t2 = perf_counter_ns()
        tok = choose(lg, chat.seen, sampled if r % 2 == 0 else greedy, rng)
        var c = ms(t2)
        if r >= 3:
            enq.append(e)
            fwd.append(f)
            head.append(hd)
            host.append(c)
    var tot = median(fwd.copy()) + median(head.copy()) + median(host.copy())
    print("  decode at", chat.cache.pos, "tokens: forward enqueue", r1(median(enq^)), "ms, forward done", r1(median(fwd^)),
          "ms, head + download", r1(median(head^)), "ms, host choice", r1(median(host^)), "ms =", r1(tot), "ms a token")

    # 2. prefill
    var sizes: List[Int] = [128, 512, 1024]
    for s in sizes:
        var ids = List[Int]()
        for i in range(s):
            ids.append(prompt[i % len(prompt)])
        var times = List[Float64]()
        for _ in range(3):
            chat.cache.reset()
            var t0 = perf_counter_ns()
            var h = forward_cached(chat.lm, ids, chat.cache)
            _ = last_logits(chat.lm, h, s)
            times.append(ms(t0))
        print("  prefill", s, "tokens:", r1(median(times^)), "ms")

    # 3. a turn's first token after a long conversation
    chat.reset(SYSTEM)
    var filler = String("")
    for i in range(40):
        filler += "Le robot numéro " + String(i) + " range la cuisine et parle avec les enfants. "
    chat.user(filler)
    chat.start(ChatSettings.greedy_with(1.1, 1))
    var before = chat.cache.pos
    chat.user("Et maintenant, où est le canapé ?")
    var t0 = perf_counter_ns()
    chat.start(ChatSettings.greedy_with(1.1, 1))
    print("  new turn after", before, "tokens in the conversation:", len(chat.reply), "token in", r1(ms(t0)), "ms")

    # 4. real replies, thinking off then on
    var questions: List[String] = ["Bonjour ! Qui es-tu, et qu'est-ce que tu sais faire ?", "Can you tell me a short story about the sofa?"]
    for think in range(2):
        for ref q in questions:
            chat.thinking = think == 1
            chat.reset(SYSTEM)
            chat.user(q)
            var s = ChatSettings()
            s.max_tokens = 400
            var t = perf_counter_ns()
            chat.start(s)
            var first = ms(t)
            var answer_at = first if think == 0 else -1.0
            var answer_tok = 0 if think == 0 else -1
            while chat.next():
                if answer_at < 0.0 and chat.reply_text().find("</think>") >= 0:
                    answer_at = ms(t)
                    answer_tok = len(chat.reply)
            var total = ms(t)
            print("  thinking", "ON " if think == 1 else "OFF", ":", len(chat.reply), "tokens in", r1(total), "ms (",
                  r1(Float64(len(chat.reply)) / (total / 1000.0)), "tokens/s ), first token", r1(first), "ms, answer starts at token",
                  answer_tok, "after", r1(answer_at), "ms")
            print("     '" + chat.reply_text().replace("\n", " ") + "'")
