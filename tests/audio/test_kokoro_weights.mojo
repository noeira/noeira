# +--------------------------------------------------------------------------+ #
# | Kokoro's weights as the port holds them, against torch
# +--------------------------------------------------------------------------+ #
"""Gates `noeira/audio/kokoro/weights.mojo` on torch's own tensors.

    pixi run -e detect python tools/kokoro/convert_kokoro_checkpoint.py
    pixi run mojo run -I . tests/audio/test_kokoro_weights.mojo [kokoro_dir]

The converter writes, beside the safetensors file, one line per tensor as the
port must hold it: name, shape, and a position-weighted float64 sum. The 89
weight-normalised convolutions are listed FOLDED, by torch's `_weight_norm`.
This loads the checkpoint with `KokoroWeights.load` and requires, for every
line: the tensor exists under that name, with that shape, and its own
position-weighted sum agrees to `SUM_BOUND` (relative to the sum of |w|; the
fold is float32 here and in torch, in a different order).

Both directions: every line has a tensor, and every tensor has a line.

Non-vacuity, for the fold: the reference also carries the sum of each pair's
UNFOLDED `weight_v`. Our folded tensor must NOT match that one — if it did,
the comparison could not tell a loader that forgot the gains from one that
applied them.
"""

from std.sys import argv

from noeira.audio.kokoro.paths import kokoro_dir
from noeira.audio.kokoro.weights import KokoroWeights, KK_TABLE_TENSORS, KK_WEIGHT_NORMED


comptime SUM_BOUND: Float64 = 1.0e-6


def _abs(x: Float64) -> Float64:
    return x if x >= 0.0 else -x


def main() raises:
    var args = argv()
    var dir = String(args[1]) if len(args) >= 2 else kokoro_dir()
    print("Kokoro weights vs torch (" + dir + ")")
    var w = KokoroWeights.load(dir + "/kokoro-v1_0.safetensors")

    var f = open(dir + "/kokoro-v1_0.check.tsv", "r")
    var text = f.read()
    f.close()

    var fails = 0
    var lines = 0
    var folded_seen = 0
    var told_apart = 0
    var worst = 0.0
    var seen = List[Bool](length=len(w.names), fill=False)
    # ⚠ `split("\n")`: `splitlines()` also breaks at tabs.
    for line in text.split("\n"):
        var s = String(line)
        if s == "":
            continue
        var parts = s.split("\t")
        if len(parts) < 3:
            raise Error("malformed reference line: " + s)
        lines += 1
        var name = String(parts[0])
        if not w.has(name):
            print("  FAIL  no tensor named '" + name + "'")
            fails += 1
            continue
        var i = w.at(name)
        seen[i] = True
        var want_shape = String("[") + String(parts[1]).replace(",", ", ") + "]"
        if w.shape_str(i) != want_shape:
            print("  FAIL  " + name + " is " + w.shape_str(i) + ", torch has " + want_shape)
            fails += 1
            continue
        var got = 0.0
        var mass = 0.0
        for k in range(w.numel(i)):
            var v = Float64(w.tensors[i].data[k]) * Float64(1 + k % 7)
            got += v
            mass += _abs(v)
        var want = atof(parts[2])
        var err = _abs(got - want) / mass if mass > 0.0 else _abs(got - want)
        if err > worst:
            worst = err
        if err > SUM_BOUND:
            print("  FAIL  " + name + "  weighted sum " + String(got) + ", torch " + String(want))
            fails += 1
        if len(parts) >= 4 and String(parts[3]) != "":
            folded_seen += 1
            var unfolded = atof(parts[3])
            var miss = _abs(got - unfolded) / mass if mass > 0.0 else 0.0
            if miss > SUM_BOUND:
                told_apart += 1

    var orphans = 0
    for i in range(len(seen)):
        if not seen[i]:
            orphans += 1
            if orphans <= 3:
                print("  FAIL  '" + w.names[i] + "' is in the table and not in torch's list")
    fails += orphans

    var counts_ok = lines == KK_TABLE_TENSORS and folded_seen == KK_WEIGHT_NORMED
    print("  " + ("PASS" if counts_ok else "FAIL") + "  " + String(lines) + " tensors compared, "
          + String(folded_seen) + " of them folded pairs (expected " + String(KK_TABLE_TENSORS)
          + " and " + String(KK_WEIGHT_NORMED) + ")")
    if not counts_ok:
        fails += 1
    print("  " + ("PASS" if fails == 0 else "FAIL") + "  names, shapes and weighted sums: worst relative error "
          + String(worst) + " (bound " + String(SUM_BOUND) + ")")
    # A pair whose gains happen to equal its norms folds to itself; nearly all must differ.
    var apart_ok = told_apart * 10 >= folded_seen * 9 and folded_seen > 0
    print("  " + ("PASS" if apart_ok else "FAIL") + "  non-vacuity: " + String(told_apart) + " of "
          + String(folded_seen) + " folded weights differ from their unfolded weight_v")
    if not apart_ok:
        fails += 1

    print("")
    if fails == 0:
        print("ALL PASS")
    else:
        print(String(fails) + " FAILURES")
        raise Error("kokoro weights gate failed")
