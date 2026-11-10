"""LFM3's text backbone (`noeira.llm.lfm3.Lfm3Text`) against d1-2B's own code,
layer by layer (docs/NN_MODEL_PORTS_PLAN.md §12.5, step 3).

    build on the Orin: build-jetson tests/llm/test_lfm3_d1_reference_gpu.mojo
    ./test_lfm3_d1_reference_gpu [d1 dir] [ref dir]
      defaults: ~/.cache/noeira/local-ai/d1/d1-2B and ~/.cache/noeira/d1/ref

The reference is `tools/d1/dump_d1_reference.py`: d1's System One in fp32,
the rows it packs (state + questions under the `set` mask). For every row of
the three cases: the embedding, each of the 24 layers' outputs, the final norm,
and the slots' log-softmax over the vocabulary — max |d| / max |reference| of
each, against the fp32 bound. Every language-model tensor must be used.

Falsification on the first row: the full layers causal instead of `set`, the
full RoPE instead of the zero tail, the output gate off — each must move the
slots' log-probabilities far beyond the bound.
"""

from std.os import getenv
from std.sys import argv
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.io.json import load_json
from noeira.io.npy import load_npy_f64
from noeira.llm.lfm3 import Lfm3Text
from noeira.llm.lfm3.model import LF_TF32
from noeira.nn.constants import DT


comptime STAGE_BOUND: Float64 = 1e-4
comptime FALSIFY_MIN: Float64 = 10.0 * STAGE_BOUND
"""A defect must land ten times past the bound the gate holds the port to.
Not more: the zero tail touches only frequency slots 32-63, whose angles are
below 1e-3 rad per position, so rotating them too moves the slots by ~3e-3 on a
250-token row (the real error is ~1e-6)."""


def rel_err(got: List[Float32], want: List[Float64]) raises -> Float64:
    if len(got) != len(want):
        raise Error("size " + String(len(got)) + " against the reference's " + String(len(want)))
    var d = 0.0
    var m = 0.0
    for i in range(len(want)):
        d = max(d, abs(Float64(got[i]) - want[i]))
        m = max(m, abs(want[i]))
    return d / m if m > 0.0 else d


def check(mut fails: Int, ok: Bool, what: String):
    if ok:
        print("  PASS ", what)
    else:
        print("  FAIL ", what)
        fails += 1


struct Row(Movable):
    var ids: List[Int]
    var groups: List[Int]
    var state_len: Int
    var slots: List[Int]
    var path: String

    def __init__(out self, path: String) raises:
        self.path = path
        var a = load_npy_f64(path + "_ids.npy")
        var g = load_npy_f64(path + "_question.npy")
        self.ids = List[Int]()
        self.groups = List[Int]()
        for i in range(len(a)):
            self.ids.append(Int(a[i]))
            self.groups.append(Int(g[i]))
        var meta = load_json(path + "_meta.json")
        self.state_len = meta.integer(meta.field(meta.root(), "state_len"))
        self.slots = List[Int]()
        var sl = meta.field(meta.root(), "slots")
        for q in range(meta.size(sl)):
            var own = meta.at(sl, q)
            for k in range(meta.size(own)):
                self.slots.append(meta.integer(meta.at(own, k)))


def main() raises:
    var args = argv()
    var home = getenv("HOME")
    var model_dir = String(args[1]) if len(args) > 1 else home + "/.cache/noeira/local-ai/d1/d1-2B"
    var ref_dir = String(args[2]) if len(args) > 2 else home + "/.cache/noeira/d1/ref"
    print("LFM3 text backbone vs d1-2B's own code (", model_dir, ")")
    var t0 = perf_counter_ns()
    var lm = Lfm3Text.load(DeviceContext(), model_dir)
    print("  loaded", len(lm.names), "tensors in", (perf_counter_ns() - t0) // 1_000_000, "ms;",
          len(lm.cfg.layer_types), "layers")
    var fails = 0
    var cases: List[String] = ["room_en", "single_fr", "many_noul"]
    var rows_per: List[Int] = [1, 1, 2]
    var first = String("")
    for ci in range(len(cases)):
        for r in range(rows_per[ci]):
            var path = ref_dir + "/" + cases[ci] + "/row" + String(r)
            if first == "":
                first = path
            var row = Row(path)
            var taps = List[DeviceBuffer[DT]]()
            var h = lm.forward(row.ids, row.groups, row.state_len, taps, keep_taps=True)
            var t = len(row.ids)
            var dm = lm.cfg.hidden
            var worst = rel_err(lm.download(taps[0], t * dm), load_npy_f64(path + "_embed.npy"))
            var worst_at = String("embed")
            for l in range(len(lm.cfg.layer_types)):
                var e = rel_err(lm.download(taps[l + 1], t * dm), load_npy_f64(path + "_layer" + String(l) + ".npy"))
                if e > worst:
                    worst = e
                    worst_at = String("layer ") + String(l)
            var ef = rel_err(lm.download(h, t * dm), load_npy_f64(path + "_final.npy"))
            var lz = lm.slot_logz(h, t, row.slots)
            var el = rel_err(lz, load_npy_f64(path + "_slot_logz.npy"))
            print("  ", cases[ci], "row", r, ":", t, "tokens, state", row.state_len, ",", len(row.slots), "slots")
            check(fails, worst < STAGE_BOUND, String("embedding and 24 layers, worst ") + worst_at + "  rel err = " + String(worst))
            check(fails, ef < STAGE_BOUND, String("final norm  rel err = ") + String(ef))
            check(fails, el < STAGE_BOUND, String("slots' log-softmax over the vocabulary  rel err = ") + String(el))

    var un = lm.unused()
    check(fails, len(un) == 0, String("every language-model tensor used (") + String(len(un)) + " unused"
          + ((": " + un[0]) if len(un) > 0 else String("")) + ")")

    # falsification, first row
    var row = Row(first)
    var want = load_npy_f64(first + "_slot_logz.npy")
    var names: List[String] = ["full layers causal instead of `set`", "full RoPE instead of the zero tail", "output gate off"]
    var bits: List[Int] = [1, 2, 4]
    for i in range(3):
        lm.debug = bits[i]
        var taps = List[DeviceBuffer[DT]]()
        var h = lm.forward(row.ids, row.groups, row.state_len, taps)
        var e = rel_err(lm.slot_logz(h, len(row.ids), row.slots), want)
        check(fails, e > FALSIFY_MIN, String("defect caught: ") + names[i] + "  rel err = " + String(e))
    lm.debug = 0

    # timing, warm
    var taps = List[DeviceBuffer[DT]]()
    var t1 = perf_counter_ns()
    var h = lm.forward(row.ids, row.groups, row.state_len, taps)
    var lz = lm.slot_logz(h, len(row.ids), row.slots)
    print("  forward + slots,", len(row.ids), "tokens, warm:", Float64(perf_counter_ns() - t1) / 1e6, "ms",
          "(LFM3_TF32)" if LF_TF32 else "(fp32)")
    _ = lz^

    if fails > 0:
        raise Error(String(fails) + " FAILED")
    print("ALL PASS")

