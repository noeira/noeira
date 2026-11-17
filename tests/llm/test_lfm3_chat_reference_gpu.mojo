# +--------------------------------------------------------------------------+ #
# | LFM3-1.7B chatting in Mojo vs Liquid's own code: template, cache, greedy tokens
# +--------------------------------------------------------------------------+ #
"""The gate of LFM3 generation (docs/NN_MODEL_PORTS_PLAN.md §12.7 step 2), against
`tools/lfm3/dump_lfm3_chat_reference.py`'s dumps (Liquid's code, fp32, the Orin).

    pixi run -e nvidia mojo build [-D LFM3_BF16=1] -I . tests/llm/test_lfm3_chat_reference_gpu.mojo -o build/t
    ./build/t [model_dir] [ref_dir]

Per case (a system prompt naming the room, a user turn, 48 greedy tokens):
1. the template: `Lfm3Chat.reset` + `user` make the reference's prompt tokens,
   exactly; with the thinking marker flipped they must NOT (the check can fail);
2. the prefill on an empty cache, stage by stage: the embedding, 24 layers,
   the final norm;
3. teacher-forced decoding: the reference's tokens fed one at a time on the
   cache, the logits at every step against the reference's, and the greedy
   choice (with the case's repetition penalty) equal to the reference's token;
4. free-running through the chat API (`start` / `next`, greedy): the same 48
   tokens. In bf16 a step whose reference top-2 margin is under TIE may
   legitimately flip; the comparison stops there and says so;
5. (two-turn case) turn 2 appended on turn 1's cache: its tokens are the
   reference's, then 48 more greedy tokens and their logits.

On the first case, the prefix cache (`Lfm3Chat.keep_system`, `prefix_checks`):
a system prompt reused whole, reused up to the point where it changed, and
computed ahead (`prepare`), each against the same request with the cache off;
four mutants.

Then, on one sequence longer than the sliding window (both cases' conversations,
~400 tokens): the cached path fed in uneven chunks (100, 1, 1, 60, 1, the rest)
gives the rows one causal pass gives (`Lfm3Text.forward(causal_full=True)`, the
forward the d1 gates hold), and three mutants of the cache must break that by
10x the bound: chunk positions from 0, no sliding window, the conv history
never rolled (fp32 build; in bf16 the two paths are each ~2e-2 from the truth,
see CACHE_BOUND). The reference's own conversations stay under 256 tokens, so
only this check reaches the window.
"""

from std.os import getenv
from std.sys import argv
from std.time import perf_counter_ns
from std.math import exp, log
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.io.json import load_json
from noeira.io.npy import load_npy_f64
from noeira.llm.lfm3 import Lfm3Chat, ChatSettings, Lfm3Cache, forward_cached, last_logits, choose
from noeira.llm.lfm3.model import LF_BF16, LF_INT8, LF_W8A8
from noeira.nn.constants import DT


comptime STAGE_BOUND: Float64 = 0.4 if LF_W8A8 else (0.2 if LF_INT8 else (3e-2 if LF_BF16 else 1e-4))
"""Max |d| / max |reference| over a prefill stage. W8A8 (`-D LFM3_W8A8=1`, with LFM3_INT8: the chunks on int8
activations, docs/NN_MODEL_PORTS_PLAN.md §12.8.9): measured 0.29 at the final norm (the Orin, 2026-11-16).
W8A8's bounds hold the code to the size of its measured error; whether that error costs the replies is BB-1's
question (`benchmarks/bench_turn_latency_gpu.mojo`), not this file's."""
comptime LOGIT_BOUND: Float64 = 0.2 if LF_W8A8 else (0.1 if LF_INT8 else (3e-2 if LF_BF16 else 1e-4))
"""Max |d| / max |reference| over one step's logits."""
comptime TIE: Float64 = 1.0 if LF_BF16 else 0.0
"""A reference top-2 margin (in logits, after the penalty) under which bf16 may flip the token."""
comptime CACHE_BOUND: Float64 = 0.3 if LF_W8A8 else (0.1 if LF_INT8 else (6e-2 if LF_BF16 else 1e-4))
"""Cached chunks vs one causal pass, max |d| / max |rows|. In bf16 these are two
bf16 computations, each ~2e-2 from the fp32 truth at the final norm (the prefill
stage above), so up to twice that apart (4.5e-2 measured): the check is decisive
in fp32 only, and the mutants run there."""
comptime PREFIX_BOUND: Float64 = 0.15 if LF_W8A8 else (3e-2 if (LF_INT8 or LF_BF16) else 1e-4)
"""A reply's first logits on a reused prompt prefix vs the same prompt computed
whole, max |d| / max |logits| (`prefix_checks`). Tighter than CACHE_BOUND: both
sides are chunks of eight tokens or more, on the same route. Measured on the
Orin: 2.5e-6 fp32, 7.1e-3 bf16, 4.9e-3 int8; the four mutants move the logits by
0.094 to 0.17 in every build. W8A8: up to 0.11 — a fully reused prompt's last
token takes the one-token route (int8 weights, fp32 activations), the cold one
the W8A8 chunk, so the two sides differ by W8A8's own error; the mutants (cache
logic, not precision) are judged in the other builds."""
comptime FALSIFY: Float64 = 10.0
comptime KL_BOUND: Float64 = 0.03 if LF_W8A8 else (0.01 if LF_INT8 else (5e-3 if LF_BF16 else 1e-6))
"""Mean KL(reference || ours) per teacher-forced step, nats, over the raw logits.
Measured: int8 0.0009-0.002; W8A8 0.008 / 0.020 / 0.010 on the three cases."""
comptime TOP1_MIN: Float64 = 0.85 if LF_W8A8 else (0.9 if LF_INT8 else 1.0)
"""Teacher-forced steps whose greedy choice is the reference's (penalty included).
int8 changes the numbers: its gate is this agreement and the KL, not equal
tokens (bf16 near-ties aside, the fp32 / bf16 builds must agree everywhere)."""


def kl_ref(ref_row: List[Float64], off: Int, got: List[Float32]) -> Float64:
    """KL(softmax(reference) || softmax(ours)) over one step's logits."""
    var v = len(got)
    var mr = -1e300
    var mg = -1e300
    for i in range(v):
        mr = max(mr, ref_row[off + i])
        mg = max(mg, Float64(got[i]))
    var zr = 0.0
    var zg = 0.0
    for i in range(v):
        zr += exp(ref_row[off + i] - mr)
        zg += exp(Float64(got[i]) - mg)
    var lzr = mr + log(zr)
    var lzg = mg + log(zg)
    var kl = 0.0
    for i in range(v):
        var lr = ref_row[off + i] - lzr
        kl += exp(lr) * (lr - (Float64(got[i]) - lzg))
    return kl


def rel_err(got: List[Float32], want: List[Float64], off: Int = 0, n: Int = -1) raises -> Float64:
    var count = len(want) - off if n < 0 else n
    if len(got) != count:
        raise Error("size " + String(len(got)) + " against the reference's " + String(count))
    var d = 0.0
    var m = 0.0
    for i in range(count):
        d = max(d, abs(Float64(got[i]) - want[off + i]))
        m = max(m, abs(want[off + i]))
    return d / m if m > 0.0 else d


def rel_err32(got: List[Float32], want: List[Float32]) raises -> Float64:
    if len(got) != len(want):
        raise Error("size " + String(len(got)) + " against " + String(len(want)))
    var d = 0.0
    var m = 0.0
    for i in range(len(want)):
        d = max(d, abs(Float64(got[i]) - Float64(want[i])))
        m = max(m, abs(Float64(want[i])))
    return d / m if m > 0.0 else d


def ints(v: List[Float64]) -> List[Int]:
    var out = List[Int](capacity=len(v))
    for x in v:
        out.append(Int(x))
    return out^


def check(mut fails: Int, ok: Bool, what: String):
    if ok:
        print("  PASS ", what)
    else:
        print("  FAIL ", what)
        fails += 1


def same(a: List[Int], b: List[Int]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


def penalised(want: List[Float64], off: Int, v: Int, seen: List[Bool], penalty: Float64) -> List[Float32]:
    """The reference's step logits (row at `off`) with the penalty applied, as `choose` applies it."""
    var out = List[Float32](capacity=v)
    var pen = Float32(penalty)
    for i in range(v):
        var x = Float32(want[off + i])
        if penalty != 1.0 and seen[i]:
            x = x / pen if x > 0.0 else x * pen
        out.append(x)
    return out^


def margin(x: List[Float32]) -> Float64:
    """top-1 minus top-2."""
    var a = Float32(-3.0e38)
    var b = Float32(-3.0e38)
    for y in x:
        if y > a:
            b = a
            a = y
        elif y > b:
            b = y
    return Float64(a - b)


def seen_of(ids: List[Int], v: Int) -> List[Bool]:
    var s = List[Bool](length=v, fill=False)
    for id in ids:
        s[id] = True
    return s^


def free_run(
    mut chat: Lfm3Chat, settings: ChatSettings, want: List[Int], want_logits: List[Float64], seen0: List[Bool],
    what: String, mut fails: Int,
) raises -> Bool:
    """`start` + `next` until done; the tokens must be `want` up to a bf16 near-tie. True if all equal."""
    var v = chat.lm.cfg.vocab
    var seen = seen0.copy()
    chat.start(settings)
    var worst = 0.0
    var s = 0
    var tie_at = -1
    while True:
        if s < len(want):
            worst = max(worst, rel_err(chat.last_logits, want_logits, s * v, v))
        var got = chat.reply[len(chat.reply) - 1]
        if s < len(want) and got != want[s]:
            var m = margin(penalised(want_logits, s * v, v, seen, settings.repetition_penalty))
            if m < TIE:
                tie_at = s
            else:
                print("        step", s, ": token", got, "against the reference's", want[s], "(reference margin", m, ")")
            break
        if s < len(want):
            seen[want[s]] = True
        s += 1
        if not chat.next():
            break
    var all_equal = same(chat.reply, want)
    if tie_at >= 0:
        print("        near-tie at step", tie_at, ": the free run stops comparing there")
    # int8: a free run may leave the reference's path (its gate is the teacher-forced agreement and KL)
    check(fails, all_equal or tie_at >= 0 or LF_INT8, what + ": " + String(s) + " tokens as the reference's, logits <= "
          + String(worst) + " (bound " + String(LOGIT_BOUND) + "): '" + chat.reply_text() + "'")
    check(fails, worst <= LOGIT_BOUND, what + ": logits on the way")
    return all_equal


def cached_rows(mut chat: Lfm3Chat, seq: List[Int], debug: Int) raises -> List[Float32]:
    """Every row of `seq` through the cache, in uneven chunks."""
    chat.cache.reset()
    chat.cache.debug = debug
    var sizes: List[Int] = [100, 1, 1, 60, 1]
    var out = List[Float32]()
    var i = 0
    var k = 0
    var dm = chat.lm.cfg.hidden
    while i < len(seq):
        var n = sizes[k] if k < len(sizes) else len(seq) - i
        n = min(n, len(seq) - i)
        var chunk = List[Int]()
        for j in range(i, i + n):
            chunk.append(seq[j])
        var h = forward_cached(chat.lm, chunk, chat.cache)
        out += chat.lm.download(h, n * dm)
        i += n
        k += 1
    chat.cache.debug = 0
    return out^


def int8_mutant_kl(model_dir: String, dir: String) raises -> Float64:
    """Mean KL over the first case's teacher-forced steps with the weights on a
    4-bit grid: the KL bound must catch a coarser quantization. Its own scope:
    the second model is freed on return."""
    var chat = Lfm3Chat.load(DeviceContext(), model_dir, 1024, True, 7)
    var v = chat.lm.cfg.vocab
    var prompt = ints(load_npy_f64(dir + "/prompt_ids.npy"))
    var gen = ints(load_npy_f64(dir + "/gen_ids.npy"))
    var glog = load_npy_f64(dir + "/gen_logits.npy")
    var h = forward_cached(chat.lm, prompt, chat.cache)
    var tot = kl_ref(glog, 0, last_logits(chat.lm, h, len(prompt)))
    var steps = min(16, len(gen))
    for s in range(1, steps):
        var one: List[Int] = [gen[s - 1]]
        h = forward_cached(chat.lm, one, chat.cache)
        tot += kl_ref(glog, s * v, last_logits(chat.lm, h, 1))
    return tot / Float64(steps)


def _first_logits(mut chat: Lfm3Chat, system: String, turn1: String) raises -> List[Float32]:
    """One request: `reset`, the user's turn, six greedy tokens; the logits
    that chose the first one."""
    chat.reset(system)
    chat.user(turn1)
    chat.start(ChatSettings.greedy_with(1.0, 6))
    var first = chat.last_logits.copy()
    while chat.next():
        pass
    return first^


def _differ(a: List[Float32], b: List[Float32]) -> Int:
    var n = 0
    for i in range(len(a)):
        if a[i] != b[i]:
            n += 1
    return n


def _changed_middle(system: String) -> String:
    """`system` with a sentence put in at the first space past its middle: the
    beginning is the same tokens, the rest moves."""
    var b = system.as_bytes()
    var at = len(b) // 2
    while at < len(b) and b[at] != 32:
        at += 1
    return String(system[byte=0:at]) + " (Il pleut dehors ce matin.)" + String(system[byte=at:])


def prefix_checks(mut chat: Lfm3Chat, system: String, turn1: String, mut fails: Int) raises -> Int:
    """`keep_system`, the prefix cache matched by tokens (docs/NN_MODEL_PORTS_PLAN.md
    §12.8, S1). Against the same request with the cache off (one chunk on an
    empty cache):

    1. cold, the prompt is still one chunk: bit-identical;
    2. the same prompt again: all of it found in the cache, within PREFIX_BOUND
       (its rows were computed in another chunk); a third time, bit-identical
       to the second;
    3. a prompt changed in its middle: its beginning kept up to a mark, the
       rest computed, within PREFIX_BOUND of that prompt computed cold;
    4. `prepare`, then the request: all of the prompt found, within PREFIX_BOUND;
    5. four mutants, each over PREFIX_BOUND: a rewind that leaves the conv
       histories, a mark holding the history of its chunk's end, a rewind one
       mark too far (tokens that differ reused), a rewind that keeps the marks
       past it (a later prompt answered on another prompt's rows).

    Leaves `keep_system` off and the cache empty."""
    var other = _changed_middle(system)
    chat.keep_system = False
    var one_pass = _first_logits(chat, system, turn1)
    var other_cold = _first_logits(chat, other, turn1)

    chat.keep_system = True
    var cold = _first_logits(chat, system, turn1)
    var cold_reused = chat.reused
    var head = chat.head_len
    check(fails, _differ(cold, one_pass) == 0 and cold_reused == 0 and len(chat.marks) == (head - 1) // 32 + 1,
          "prefix cache, cold: the prompt (" + String(head) + " tokens) is one chunk, bit-identical to no cache; "
          + String(len(chat.marks)) + " marks kept")

    var again = _first_logits(chat, system, turn1)
    var e2 = rel_err32(again, one_pass)
    check(fails, chat.reused == head and e2 <= PREFIX_BOUND, "prefix cache, the same prompt: " + String(chat.reused) + " of "
          + String(head) + " tokens found, logits within " + String(e2) + " of no cache (bound " + String(PREFIX_BOUND) + ")")
    var third = _first_logits(chat, system, turn1)
    var bitwise = _differ(third, again)
    check(fails, bitwise == 0, "prefix cache, a third time: bit-identical to the second (" + String(bitwise) + " differ)")

    var part = _first_logits(chat, other, turn1)
    var kept = chat.reused
    var e3 = rel_err32(part, other_cold)
    check(fails, kept >= 32 and kept < head and kept % 32 == 0 and e3 <= PREFIX_BOUND,
          "prefix cache, a prompt changed in its middle: " + String(kept) + " of " + String(chat.head_len)
          + " tokens kept, logits within " + String(e3) + " of that prompt computed cold")

    chat.prepare(system)
    var prep_kept = chat.reused
    var prep_pos = chat.cache.pos
    var after = _first_logits(chat, system, turn1)
    var e4 = rel_err32(after, one_pass)
    check(fails, prep_kept == kept and prep_pos == head and chat.reused == head and e4 <= PREFIX_BOUND,
          "prefix cache, prepare then the request: the prompt computed ahead from token " + String(prep_kept)
          + ", then all " + String(chat.reused) + " found; logits within " + String(e4))

    # mutants (not in W8A8: PREFIX_BOUND there is W8A8's own error, the size of a mutant's; see PREFIX_BOUND)
    comptime if LF_W8A8:
        print("  (the prefix-cache mutants run in the bf16 and int8 builds)")
        chat.keep_system = False
        chat.reset(String(""))
        return bitwise
    chat.cache.debug = 8
    var m8 = rel_err32(_first_logits(chat, system, turn1), again)
    chat.cache.debug = 0
    check(fails, m8 > PREFIX_BOUND, "mutant caught, a rewind that keeps the conv histories: logits moved by " + String(m8)
          + " (> " + String(PREFIX_BOUND) + ")")

    chat.reset(String(""))  # nothing of the prompt left in the cache
    chat.cache.debug = 16
    _ = _first_logits(chat, system, turn1)  # its marks are taken wrongly
    chat.cache.debug = 0
    var m16 = rel_err32(_first_logits(chat, system, turn1), one_pass)
    check(fails, m16 > PREFIX_BOUND, "mutant caught, a mark holding its chunk's last history: logits moved by " + String(m16))

    chat.reset(String(""))
    _ = _first_logits(chat, system, turn1)
    chat.debug_prefix = 1
    var mp1 = rel_err32(_first_logits(chat, other, turn1), other_cold)
    var far = chat.reused
    chat.debug_prefix = 0
    check(fails, far > kept and mp1 > PREFIX_BOUND, "mutant caught, a rewind one mark too far (" + String(far)
          + " tokens reused, " + String(kept) + " are the same): logits moved by " + String(mp1))

    chat.reset(String(""))
    _ = _first_logits(chat, system, turn1)
    chat.debug_prefix = 2
    _ = _first_logits(chat, other, turn1)
    var mp2 = rel_err32(_first_logits(chat, system, turn1), one_pass)
    chat.debug_prefix = 0
    check(fails, mp2 > PREFIX_BOUND, "mutant caught, a rewind that keeps the marks past it: logits moved by " + String(mp2))

    chat.keep_system = False
    chat.reset(String(""))
    return bitwise


def main() raises:
    var home = getenv("HOME")
    var args = argv()
    var model_dir = String(args[1]) if len(args) > 1 else home + "/.cache/noeira/local-ai/lfm3/LFM3-1.7B"
    var ref_dir = String(args[2]) if len(args) > 2 else home + "/.cache/noeira/lfm3/chat_ref"
    var fails = 0
    print("LFM3-1.7B generation vs Liquid's own code", "(LFM3_BF16)" if LF_BF16 else "(fp32)",
          "(LFM3_INT8)" if LF_INT8 else "", "(LFM3_W8A8)" if LF_W8A8 else "")
    var t0 = perf_counter_ns()
    var chat = Lfm3Chat.load(DeviceContext(), model_dir, 1024)
    print("  loaded in", (perf_counter_ns() - t0) // 1_000_000, "ms")
    comptime if LF_INT8:
        # the int8 path is the one running: no bf16 copy of the linears is left to fall back on
        check(fails, chat.lm.q8_on and chat.lm.bf16_bytes() < 50_000_000,
              "int8: weights " + String(chat.lm.q8_bytes()) + " B int8, " + String(chat.lm.bf16_bytes()) + " B bf16 (norms, conv)")
    var v = chat.lm.cfg.vocab
    var dm = chat.lm.cfg.hidden
    var names: List[String] = ["fr_two_turns", "en_penalty", "fr_thinking"]
    var long_seq = List[Int]()
    var decode_ms = List[Float64]()

    for c in range(len(names)):
        var dir = ref_dir + "/" + names[c]
        print(" ", names[c])
        var meta = load_json(dir + "/meta.json")
        var root = meta.root()
        var system = meta.string(meta.field(root, "system"))
        var turns_n = meta.field(root, "turns")
        var turn1 = meta.string(meta.at(turns_n, 0))
        var thinking = meta.boolean(meta.field(root, "thinking"))
        var penalty = meta.number(meta.field(root, "repetition_penalty"))
        var prompt = ints(load_npy_f64(dir + "/prompt_ids.npy"))
        var gen = ints(load_npy_f64(dir + "/gen_ids.npy"))
        var glog = load_npy_f64(dir + "/gen_logits.npy")
        var steps = len(gen)

        # 1. the template
        chat.thinking = thinking
        chat.reset(system)
        chat.user(turn1)
        check(fails, same(chat.prompt_ids(), prompt), "template: " + String(len(prompt)) + " prompt tokens equal")
        chat.thinking = not thinking
        chat.reset(system)
        chat.user(turn1)
        check(fails, not same(chat.prompt_ids(), prompt), "template: the other thinking marker does NOT match")
        chat.thinking = thinking

        # 2. the prefill, stage by stage
        chat.cache.reset()
        var taps = List[DeviceBuffer[DT]]()
        var p = len(prompt)
        var h = forward_cached(chat.lm, prompt, chat.cache, taps, True)
        var worst = rel_err(chat.lm.download(taps[0], p * dm), load_npy_f64(dir + "/prefill_embed.npy"))
        for l in range(len(taps) - 1):
            worst = max(worst, rel_err(chat.lm.download(taps[l + 1], p * dm), load_npy_f64(dir + "/prefill_layer" + String(l) + ".npy")))
        var ef = rel_err(chat.lm.download(h, p * dm), load_npy_f64(dir + "/prefill_final.npy"))
        check(fails, worst <= STAGE_BOUND and ef <= STAGE_BOUND,
              "prefill: embedding + 24 layers <= " + String(worst) + ", final norm " + String(ef) + " (bound " + String(STAGE_BOUND) + ")")

        # 3. teacher-forced: the reference's tokens fed on the cache
        var seen = seen_of(prompt, v)
        var lw = 0.0
        var tok_bad = 0
        var ties = 0
        var kl_sum = 0.0
        var kl_max = 0.0
        for s in range(steps):
            if s > 0:
                var one: List[Int] = [gen[s - 1]]
                var t1 = perf_counter_ns()
                h = forward_cached(chat.lm, one, chat.cache)
                var lg = last_logits(chat.lm, h, 1)
                decode_ms.append(Float64(perf_counter_ns() - t1) / 1e6)
                lw = max(lw, rel_err(lg, glog, s * v, v))
                var kl = kl_ref(glog, s * v, lg)
                kl_sum += kl
                kl_max = max(kl_max, kl)
                var rng = UInt64(1)
                var pick = choose(lg, seen, ChatSettings.greedy_with(penalty, steps), rng)
                if pick != gen[s]:
                    if margin(penalised(glog, s * v, v, seen, penalty)) < TIE:
                        ties += 1
                    else:
                        tok_bad += 1
            else:
                var lg = last_logits(chat.lm, h, p)
                lw = max(lw, rel_err(lg, glog, 0, v))
                var kl = kl_ref(glog, 0, lg)
                kl_sum += kl
                kl_max = max(kl_max, kl)
                var rng = UInt64(1)
                if choose(lg, seen, ChatSettings.greedy_with(penalty, steps), rng) != gen[0]:
                    tok_bad += 1
            seen[gen[s]] = True
        check(fails, lw <= LOGIT_BOUND, "teacher-forced: " + String(steps) + " steps' logits <= " + String(lw) + " (bound " + String(LOGIT_BOUND) + ")")
        var agree = Float64(steps - tok_bad - ties) / Float64(steps)
        comptime if LF_INT8:
            check(fails, agree >= TOP1_MIN, "teacher-forced: " + String(steps - tok_bad - ties) + " / " + String(steps)
                  + " greedy choices are the reference's (min " + String(TOP1_MIN) + ")")
        else:
            check(fails, tok_bad == 0, "teacher-forced: every greedy choice is the reference's (" + String(ties) + " bf16 near-ties allowed)")
        check(fails, kl_sum / Float64(steps) <= KL_BOUND, "teacher-forced: mean KL(reference || ours) " + String(kl_sum / Float64(steps))
              + " nats, max " + String(kl_max) + " (bound on the mean " + String(KL_BOUND) + ")")

        # 4a. the system prompt kept (`keep_system`): prefilled on its own, then
        # rewound to for the next request — what the robot's service does.
        # Before the free run: it resets the chat, and turn 2 continues turn 1
        if c == 0:
            var first = prefix_checks(chat, system, turn1, fails)
            # the reference's turn 2 replayed as history (`assistant`): the same tokens
            if meta.size(turns_n) > 1:
                var t2r = ints(load_npy_f64(dir + "/turn2_ids.npy"))
                chat.keep_system = False
                chat.reset(system)
                chat.user(turn1)
                chat.assistant(chat.tok.decode(gen))
                chat.user(meta.string(meta.at(turns_n, 1)))
                var want_ids = prompt.copy()
                want_ids += gen.copy()
                want_ids += t2r.copy()
                check(fails, same(chat.prompt_ids(), want_ids), "history replayed (user, assistant, user): the template's "
                      + String(len(want_ids)) + " tokens")
            _ = first

        # 4. free-running through the API
        chat.reset(system)
        chat.user(turn1)
        var seen0 = seen_of(chat.prompt_ids(), v)
        var ok1 = free_run(chat, ChatSettings.greedy_with(penalty, steps), gen, glog, seen0, "free run, turn 1", fails)
        if c < 2:  # the two thinking-off conversations make the long sequence
            long_seq += prompt.copy()
            long_seq += gen.copy()

        # 5. turn 2 on turn 1's cache
        if meta.size(turns_n) > 1:
            var t2 = ints(load_npy_f64(dir + "/turn2_ids.npy"))
            var gen2 = ints(load_npy_f64(dir + "/turn2_gen_ids.npy"))
            var glog2 = load_npy_f64(dir + "/turn2_gen_logits.npy")
            if not ok1:
                print("        turn 2 skipped: turn 1 did not end on the reference's tokens")
                continue
            chat.user(meta.string(meta.at(turns_n, 1)))
            var want_feed: List[Int] = [gen[len(gen) - 1]]
            want_feed += t2.copy()
            check(fails, same(chat.prompt_ids(), want_feed), "turn 2: the appended tokens are the template's ("
                  + String(len(t2)) + " + the last reply token)")
            var all_ids = prompt.copy()
            all_ids += gen.copy()
            all_ids += t2.copy()
            _ = free_run(chat, ChatSettings.greedy_with(penalty, len(gen2)), gen2, glog2, seen_of(all_ids, v), "free run, turn 2 on the cache", fails)
            long_seq += t2.copy()
            long_seq += gen2.copy()

    # the cache against one causal pass, past the sliding window
    print("  cache = recompute over", len(long_seq), "tokens (window", chat.lm.cfg.window, ")")
    var taps = List[DeviceBuffer[DT]]()
    var full = chat.lm.forward(long_seq, List[Int](), 0, taps, causal_full=True)
    var want = chat.lm.download(full, len(long_seq) * dm)
    var e0 = rel_err32(cached_rows(chat, long_seq, 0), want)
    check(fails, e0 <= CACHE_BOUND, "chunks 100/1/1/60/1/rest on the cache = one causal pass: " + String(e0) + " (bound " + String(CACHE_BOUND) + ")")
    comptime if not LF_BF16:
        var mutants: List[Int] = [1, 2, 4]
        var mnames: List[String] = ["positions from 0 in each chunk", "no sliding window", "conv history never rolled"]
        for i in range(len(mutants)):
            var em = rel_err32(cached_rows(chat, long_seq, mutants[i]), want)
            check(fails, em > FALSIFY * CACHE_BOUND, "mutant caught, " + mnames[i] + ": " + String(em))
    else:
        print("  (the cache mutants run in the fp32 build: in bf16 the window mutant is only 2x the two paths' gap)")

    comptime if LF_INT8:
        var mkl = int8_mutant_kl(model_dir, ref_dir + "/" + names[0])
        check(fails, mkl > FALSIFY * KL_BOUND, "mutant caught, a 4-bit grid (levels 7): mean KL " + String(mkl)
              + " > " + String(FALSIFY * KL_BOUND))

    # rough speed (step 3 measures properly)
    if len(decode_ms) > 0:
        for i in range(1, len(decode_ms)):  # insertion sort
            var j = i
            while j > 0 and decode_ms[j - 1] > decode_ms[j]:
                var t = decode_ms[j]
                decode_ms[j] = decode_ms[j - 1]
                decode_ms[j - 1] = t
                j -= 1
        print("  decode, teacher-forced: median", decode_ms[len(decode_ms) // 2], "ms a token (logits downloaded)")
    print("")
    if fails > 0:
        print("FAILED:", fails)
        raise Error("lfm3 chat gate failed")
    print("ALL PASS")
