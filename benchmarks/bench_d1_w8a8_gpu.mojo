# +--------------------------------------------------------------------------+ #
# | d1 in W8A8 against bf16, on the room demo's own requests
# +--------------------------------------------------------------------------+ #
"""L5 (docs/NN_MODEL_PORTS_PLAN.md §12.8.8): is d1 with int8 weights AND int8
activations (`Lfm3Text.w8a8`) faster, and does it decide the same?

    pixi run -e jetson build-jetson -D LFM3_BF16=1 benchmarks/bench_d1_w8a8_gpu.mojo -o build/bench_d1_w8a8
    ./build/bench_d1_w8a8 [reps] <trace dir> [trace dir ...]

The traces are `D1.system_one`'s (`NOEIRA_D1_TRACE=<dir>` on a demo run:
`d1_req_<n>.bin`, `[0|1] state \\0 questions`). One d1 is loaded in bf16, then
its linears are quantized to int8 beside the bf16 weights, as
`D1.load(w8a8=True)` loads them (`quantize_int8(table=False,
operator_out=False)`: the head's table and the operators' output projections
stay bf16). Each request is answered in bf16 (what the demo ran before), then
in each of `W8A8_VARIANTS` (comma-separated, default `w8,0,128`), `reps`
times each:

  - `w8`: int8 weights dequantized for the bf16 GEMM (the weights' error alone);
  - a number: W8A8 — `0` as the demo runs it, `16` / `32` / `64` a group of
    linears on bf16 activations (`Lfm3Text.w8a8_bf16_act` 1 / 2 / 4, the
    diagnosis that chose which stay bf16), `128` the MLP unfused (`debug`).

Per request: tokens, the median ms of each, and against bf16 the largest
difference of any probability / confidence and the picks that differ (a
choice's label, a score's level, a yes / no on either side of 0.5). The last
lines sum it all, and with both `0` and `128`, whether the fused MLP gives the
unfused one's numbers to the bit (it computes the same floats in the same
order). `W8A8_SAVE=<dir>`: the answers of every request where a pick moved.
Nothing else here passes or fails: the numbers go to the plan.
"""

from std.os.path import exists
from std.os import getenv
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


def picks(s: String) -> List[String]:
    """Each answer's pick, in order: a choice's label, a score's level, and a
    yes / no's side of 0.5 (`"noul": p`)."""
    var out = List[String]()
    var at = 0
    while True:
        var c = s.find("\"choice\"", at)
        var sc = s.find("\"score\"", at)
        var nl = s.find("\"noul\"", at)
        var first = -1
        for v in [c, sc, nl]:
            if v >= 0 and (first < 0 or v < first):
                first = v
        if first < 0:
            break
        var colon = s.find(":", first)
        if colon < 0:
            break
        var j = colon + 1
        while j < s.byte_length() and s.as_bytes()[j] == 32:
            j += 1
        var e = j
        if s.as_bytes()[j] == 34:  # a string
            e = s.find("\"", j + 1) + 1
        else:
            while e < s.byte_length() and s.as_bytes()[e] != 44 and s.as_bytes()[e] != 125:
                e += 1
        var v = String(s[byte=j:e])
        if first == c:
            out.append("choice " + v)
        elif first == sc:
            out.append("score " + v)
        else:
            try:
                out.append("yes" if Float64(v) >= 0.5 else "no")
            except:
                out.append("noul " + v)
        at = e
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


def answer(mut d1: D1, state: String, is_json: Bool, q: String, reps: Int) raises -> Tuple[String, Float64]:
    var ts = List[Float64]()
    var a = String("")
    for _ in range(reps):
        var t0 = perf_counter_ns()
        a = d1.system_one(state, is_json, q)
        ts.append(Float64(perf_counter_ns() - t0) / 1e6)
    return (a, median(ts^))


def compare(a: String, b: String, mut diff: Float64, mut moved: Int, mut total: Int, mut what: String):
    var x = numbers(a)
    var y = numbers(b)
    if len(x) != len(y):
        diff = 1e9
    else:
        for i in range(len(x)):
            diff = max(diff, abs(x[i] - y[i]))
    var pa = picks(a)
    var pb = picks(b)
    total += len(pa)
    for i in range(min(len(pa), len(pb))):
        if pa[i] != pb[i]:
            moved += 1
            what += " [" + pa[i] + " -> " + pb[i] + "]"
    moved += abs(len(pa) - len(pb))


def main() raises:
    var args = argv()
    if len(args) < 3:
        raise Error("usage: bench_d1_w8a8 <reps> <trace dir> [trace dir ...]")
    var reps = Int(String(args[1]))
    var dir = getenv("HOME") + "/.cache/noeira/local-ai/d1/d1-2B"
    # `W8A8_VARIANTS`: the columns after bf16, comma-separated: `w8` (int8 weights, bf16 GEMM) or a number, W8A8
    # with 16 / 32 / 64 (`w8a8_bf16_act` 1 / 2 / 4: a group of linears on bf16 activations) and 128 (`debug`:
    # the MLP unfused). `W8A8_SAVE`: a
    # folder for the answers of every request where a pick moved (`ans_<k>_bf16.json`, `ans_<k>_<variant>.json`).
    var spec = getenv("W8A8_VARIANTS")
    if spec == "":
        spec = String("w8,0,128")
    var save = getenv("W8A8_SAVE")
    var variants = List[String]()
    for v in spec.split(","):
        variants.append(String(v))
    var d1 = D1.load(DeviceContext(), dir)
    # what `D1.load(w8a8=True)` loads: the table and the operators' output projections stay bf16
    d1.lm.quantize_int8(table=False, operator_out=False)
    _ = d1.system_one(String("warm-up"), False, String('{"x": {"type": "noul", "instructions": "Is this a test?"}}'))
    var head = String("request\ttokens\tbf16_ms")
    for v in variants:
        head += "\t" + v + "_ms\t" + v + "_diff\t" + v + "_moved"
    print(head + "\twhat")
    var nv = len(variants)
    var sums = List[Float64](length=nv + 1, fill=0.0)
    var worst = List[Float64](length=nv, fill=0.0)
    var moved = List[Int](length=nv, fill=0)
    var picks_n = 0
    var reqs = 0
    var i0 = -1
    var i128 = -1
    for vi in range(nv):
        if variants[vi] == "0":
            i0 = vi
        if variants[vi] == "128":
            i128 = vi
    var fused_diff = 0.0
    for ai in range(2, len(args)):
        var src = String(args[ai])
        var n = 0
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
            n += 1
            if state == "warm-up":
                continue
            d1.lm.q8_on = False
            d1.lm.w8a8 = False
            d1.lm.debug = 0
            var rb = answer(d1, state, is_json, q, reps)
            var tokens = 0
            for r in d1.last_rows:
                tokens += len(r.ids)
            var line = src + "/" + String(n - 1) + "\t" + String(tokens) + "\t" + String(round(rb[1], 1))
            var what = String("")
            sums[0] += rb[1]
            var any_moved = False
            var outs = List[String]()
            for vi in range(nv):
                d1.lm.q8_on = True
                d1.lm.w8a8 = variants[vi] != "w8"
                var bits = 0 if variants[vi] == "w8" else Int(variants[vi])
                d1.lm.debug = bits & 128
                d1.lm.w8a8_bf16_act = (bits >> 4) & 7
                var rv = answer(d1, state, is_json, q, reps)
                var dv = 0.0
                var mv = 0
                var tv = 0
                var wv = String("")
                compare(rb[0], rv[0], dv, mv, tv, wv)
                line += "\t" + String(round(rv[1], 1)) + "\t" + String(round(dv, 4)) + "\t" + String(mv) + "/" + String(tv)
                if mv > 0:
                    what += " " + variants[vi] + ":" + wv
                    any_moved = True
                sums[vi + 1] += rv[1]
                worst[vi] = max(worst[vi], dv)
                moved[vi] += mv
                if vi == 0:
                    picks_n += tv
                outs.append(rv[0])
            d1.lm.q8_on = False
            d1.lm.w8a8 = False
            d1.lm.debug = 0
            d1.lm.w8a8_bf16_act = 0
            if i0 >= 0 and i128 >= 0:
                var a = numbers(outs[i0])
                var c = numbers(outs[i128])
                if len(a) != len(c):
                    fused_diff = 1e9
                for i in range(min(len(a), len(c))):
                    fused_diff = max(fused_diff, abs(a[i] - c[i]))
            print(line + "\t" + what)
            if save != "" and any_moved:
                var key = String(reqs)
                write_text_atomic(save + "/ans_" + key + "_bf16.json", rb[0])
                for vi in range(nv):
                    write_text_atomic(save + "/ans_" + key + "_" + variants[vi] + ".json", outs[vi])
            reqs += 1
    print("requests", reqs, "| picks", picks_n)
    var tl = String("time, sum ms: bf16 ") + String(round(sums[0], 1))
    for vi in range(nv):
        tl += " | " + variants[vi] + " " + String(round(sums[vi + 1], 1))
    print(tl)
    for vi in range(nv):
        print(variants[vi] + ": largest difference against bf16", worst[vi], "| picks moved", moved[vi])
    if i0 >= 0 and i128 >= 0:
        print("the fused MLP (0) against the unfused one (128): largest difference", fused_diff,
              "SAME" if fused_diff == 0.0 else "DIFFERENT")
