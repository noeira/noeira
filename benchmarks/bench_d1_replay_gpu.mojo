# +--------------------------------------------------------------------------+ #
# | d1's requests from a demo run, timed again with nothing else running
# +--------------------------------------------------------------------------+ #
"""Is d1 slower inside the room demo than alone, and by how much, on the SAME
requests? (docs/NN_MODEL_PORTS_PLAN.md §12.8.7, L1: 0.40 ms a token alone, ~0.57
in the demo's logs.)

    NOEIRA_D1_TRACE=/tmp/d1req ./build/g1room_all ... --voice ...   # saves d1_req_<n>.bin, prints each call's split
    ./build/bench_d1_replay /tmp/d1req [reps] [d1 dir] [answers dir]

Each saved request (`D1.system_one`'s trace: `[0|1] state \\0 questions`) is
answered `reps` times by `D1` in this process, on its own: the median, and the
split the trace prints (run it with `NOEIRA_D1_TRACE` set to a scratch folder to
see it). Text only: the demo's decisions are blind unless `--see`.

Then the same request without the w1 / w3 band split (`Lfm3Text.debug` bit 8,
`noeira/llm/lfm3/model.mojo`'s note): its median, and the largest difference
between the two answers' numbers (every probability and confidence). A split
that cut the rows wrong shows there; a split that does what it says moves them
by the GEMM's own rounding only. FAILS (exit 1) above `MAX_DIFF`. `answers dir`: each
request's two answers saved there (`ans_<n>_split.json`, `ans_<n>_nosplit.json`), to
compare with an fp32 build's (without `-D LFM3_BF16=1`, where they are the same).
"""

from std.os import getenv
from std.os.path import exists
from std.sys import argv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from noeira.io.fileio import read_file_bytes, write_text_atomic
from noeira.llm.d1.system_one import D1


def _text(ref b: List[UInt8], a: Int, e: Int) -> String:
    var t = List[UInt8](capacity=max(e - a, 0))
    for i in range(a, e):
        t.append(b[i])
    return String(unsafe_from_utf8=Span(t))


comptime MAX_DIFF: Float64 = 8e-3
"""The bf16 build's own spread against the fp32 build on the room demo's requests (814 to 1488 tokens): up to
7.1e-3, with or without the split (the Orin, 2026-11-14; `test_d1_system_one_gpu`'s 8.1e-4 is on short
prompts). Measured there, the split's answers are no further from fp32 than the single call's (2.5e-3 against
4.2e-3 at worst) and every pick is the same. A split that cut the rows wrong moved them by 0.89."""


def numbers(s: String) -> List[Float64]:
    """Every number literal in a JSON text, in order."""
    var out = List[Float64]()
    var b = s.as_bytes()
    var i = 0
    var n = len(b)
    while i < n:
        var c = b[i]
        var starts = (c >= 48 and c <= 57) or (c == 45 and i + 1 < n and b[i + 1] >= 48 and b[i + 1] <= 57)
        var inword = i > 0 and ((b[i - 1] >= 65 and b[i - 1] <= 90) or (b[i - 1] >= 97 and b[i - 1] <= 122) or b[i - 1] == 95)
        if starts and not inword:
            var j = i + 1
            while j < n and ((b[j] >= 48 and b[j] <= 57) or b[j] == 46 or b[j] == 101 or b[j] == 69 or b[j] == 45 or b[j] == 43):
                j += 1
            try:
                out.append(Float64(String(s[byte=i:j])))
            except:
                pass
            i = j
        else:
            i += 1
    return out^


def median(var xs: List[Float64]) -> Float64:
    for i in range(1, len(xs)):
        var j = i
        while j > 0 and xs[j - 1] > xs[j]:
            var t = xs[j]
            xs[j] = xs[j - 1]
            xs[j - 1] = t
            j -= 1
    return xs[len(xs) // 2]


def main() raises:
    var args = argv()
    if len(args) < 2:
        raise Error("usage: bench_d1_replay <trace dir> [reps] [d1 dir]")
    var src = String(args[1])
    var reps = Int(String(args[2])) if len(args) > 2 else 5
    var dir = String(args[3]) if len(args) > 3 and String(args[3]) != "-" else getenv("HOME") + "/.cache/noeira/local-ai/d1/d1-2B"
    var save = String(args[4]) if len(args) > 4 else String("")
    var d1 = D1.load(DeviceContext(), dir)
    _ = d1.system_one(String("warm-up"), False, String('{"x": {"type": "noul", "instructions": "Is this a test?"}}'))
    var n = 0
    print("request\ttokens\tmedian_ms\tms_per_token\tno_split_ms\tmax_diff")
    var worst = 0.0
    while exists(src + "/d1_req_" + String(n) + ".bin"):
        var b = read_file_bytes(src + "/d1_req_" + String(n) + ".bin")
        var cut = -1
        for i in range(1, len(b)):
            if b[i] == 0:
                cut = i
                break
        if cut < 0:
            raise Error("a request without its NUL: d1_req_" + String(n))
        var state = _text(b, 1, cut)
        var q = _text(b, cut + 1, len(b))
        var is_json = b[0] == 49
        var ts = List[Float64]()
        var tokens = 0
        for _ in range(reps):
            var t0 = perf_counter_ns()
            _ = d1.system_one(state, is_json, q)
            ts.append(Float64(perf_counter_ns() - t0) / 1e6)
            tokens = 0
            for r in d1.last_rows:
                tokens += len(r.ids)
        var m = median(ts^)
        var with_split = d1.system_one(state, is_json, q)
        d1.lm.debug |= 8
        var ts0 = List[Float64]()
        var without = String("")
        for _ in range(reps):
            var t0 = perf_counter_ns()
            without = d1.system_one(state, is_json, q)
            ts0.append(Float64(perf_counter_ns() - t0) / 1e6)
        d1.lm.debug &= ~8
        if save != "":
            write_text_atomic(save + "/ans_" + String(n) + "_split.json", with_split)
            write_text_atomic(save + "/ans_" + String(n) + "_nosplit.json", without)
        var a = numbers(with_split)
        var c = numbers(without)
        var diff = 0.0 if len(a) == len(c) else 1e9
        if len(a) == len(c):
            for i in range(len(a)):
                diff = max(diff, abs(a[i] - c[i]))
        worst = max(worst, diff)
        print(String(n) + "\t" + String(tokens) + "\t" + String(round(m, 1)) + "\t" + String(round(m / Float64(max(tokens, 1)), 3))
              + "\t" + String(round(median(ts0^), 1)) + "\t" + String(diff))
        n += 1
    if n == 0:
        raise Error("no d1_req_<n>.bin in " + src)
    print("band split", "ON" if d1.lm.band_split else "OFF (not an Orin)", "| largest answer difference with and without:", worst,
          "PASS" if worst <= MAX_DIFF else "FAIL")
    if worst > MAX_DIFF:
        raise Error("the split changes the answers")
