"""d1's System One in Mojo (`noeira.llm.d1`), timed on the requests of
`tools/d1/bench_d1_cases.py` — whose llama.cpp column it is compared with.

    python3 tools/d1/bench_d1_cases.py            # writes the cases (and times llama.cpp if it runs)
    build-jetson benchmarks/bench_d1_system_one_gpu.mojo -D LFM3_BF16=1 -o build/bench_d1_bf16
    ./build/bench_d1_bf16 [d1 dir] [cases dir] [reps]
      defaults: ~/.cache/noeira/local-ai/d1/d1-2B, ~/.cache/noeira/d1/bench, 20

Each case: one warm call, then `reps` timed calls of `D1.system_one` (the whole
call: prompt rendering, tokenising, packing, the forward, the readout and the
answer JSON). Prints p50 / min / max and the input tokens the model read.
"""

from std.os import getenv
from std.sys import argv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from noeira.io.json import parse_json
from noeira.llm.d1 import D1
from noeira.llm.lfm3.model import LF_BF16


def _bytes(s: String) -> List[UInt8]:
    var b = List[UInt8](capacity=s.byte_length())
    for x in s.as_bytes():
        b.append(x)
    return b^


def main() raises:
    var home = getenv("HOME")
    var args = argv()
    var model_dir = String(args[1]) if len(args) > 1 else home + "/.cache/noeira/local-ai/d1/d1-2B"
    var cases_dir = String(args[2]) if len(args) > 2 else home + "/.cache/noeira/d1/bench"
    var reps = Int(String(args[3])) if len(args) > 3 else 20
    print("d1 System One in Mojo", "(LFM3_BF16)" if LF_BF16 else "(fp32)", "-", reps, "timed calls per case")
    var d1 = D1.load(DeviceContext(), model_dir)
    for line in open(cases_dir + "/list.txt", "r").read().split("\n"):
        var name = String(line.strip())
        if name.byte_length() == 0:
            continue
        var state = open(cases_dir + "/" + name + ".state", "r").read()
        var q = open(cases_dir + "/" + name + ".q", "r").read()
        var out = d1.system_one(state, True, q)
        var doc = parse_json(_bytes(out))
        var tokens = Int(doc.number(doc.field(doc.field(doc.root(), "usage"), "input_tokens")))
        var nq = doc.size(doc.field(doc.root(), "answers"))
        var ms = List[Float64]()
        for _ in range(reps):
            var t = perf_counter_ns()
            _ = d1.system_one(state, True, q)
            ms.append(Float64(perf_counter_ns() - t) / 1e6)
        sort(ms)
        print("  ", name, " ", nq, "q ", tokens, "tokens  p50", ms[len(ms) // 2], "ms  min", ms[0], " max", ms[len(ms) - 1])
