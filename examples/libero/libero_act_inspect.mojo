"""ACT ON LIBERO — WHAT ONE CHECKPOINT PREDICTS FOR A STORE ROW, POSITION BY POSITION.

    pixi run -e apple  mojo run -I . examples/libero/libero_act_inspect.mojo --act <run_id> --store build/demos/<store>.h5
    pixi run -e nvidia mojo run -I . examples/libero/libero_act_inspect.mojo --act runs/<id>/checkpoints --store build/demos/libero_goal.rendered.h5 --ckpt last

Every ACT evaluation on the 5090 (2026-09-19/20) executed chunks whose 40
positions were IDENTICAL to three decimals — the `--trace-lane` rows of one
`--act-exec 10` window read `0.202 -0.021 -0.047` ten times — so the policy
was driving with the chunk's MEAN per state: the approach at half the
demonstrators' speed (their x over two seconds averages to 0.26), the pull at
the handle blended with the approach to ~0, the run's mean |a| a half of the
store's. Yet the same fits reported a held-out L1 of 0.41, near the
per-position floor (a k=5 neighbour chunk scores 0.396), while a predictor
forced to one vector per state cannot do better than 0.52. The trainer's
validation forward therefore varies across positions and the eval's
`predict` does not — on the box. The two differ in the latent multiplier,
the training flag, and the BATCH instantiation (16 vs 20 lanes), and the
learned query embedding's gradient kernel is a block reduction ending in a
conditional read-modify-write store, the shape NVIDIA has silently dropped
before in this tree (`feedback_nvidia_rmw_store_drop_in_reduction_kernels`).

This driver settles it without a physics step: it loads a checkpoint into
the EVAL's instantiation (`LiberoActTrainer[20, "gpu"]`, the batch the lanes
use), fills up to 20 rows of one store episode through the dataset's own
`fill_at` (the normalised words the trainer saw), runs `predict` (the eval's
path: latent scaled to zero, eval mode), and prints per row the predicted
chunk's spread across its 40 positions beside the recorded chunk's, and the
L1 of the prediction against the recorded chunk both per position and as a
flat vector. A checkpoint trained on Metal read here on Metal says what the
architecture does; the box's checkpoint read here on the box says whether
its `predict` is flat; the same box checkpoint read on Metal says whether
the WEIGHTS carry position structure at all (a dead query embedding does
not, wherever it is read).

`--latent posterior` reads the decoder with `z = mu(qpos, true chunk)` instead
of the zero the eval uses (`ACTTrainer.predict_with_posterior`): a chunk that
varies here and not under `predict` puts the whole shape in the latent.

Flags: `--act RUN_ID|DIR` (a run id, found under `runs/` or
`projects/*/runs/`, or a directory holding best.ckpt + norm.json), `--store PATH`, `--ckpt best|last`,
`--ep E` (episode, default 0), `--stride S` (rows at steps 0, S, 2S, ...;
default 10), `--rows N` (≤ 20, default 12). Normalised units throughout.
"""

from std.os.path import exists, isdir
from std.sys import argv
from std.math import sqrt
from max.gpu.host import DeviceContext

from noeira.core.run import resolve_checkpoint
from noeira.nn.constants import DT
from noeira.envs.libero.act import (
    LiberoActTrainer, LiberoActDataset, LIBERO_ACT_QPOS, LIBERO_ACT_ADIM,
    LIBERO_ACT_K, LIBERO_ACT_IMG_ELEMS, LIBERO_ACT_DIM,
)

comptime B = 20
"""The eval's LANES — the instantiation whose `predict` drove the lanes."""
comptime K = LIBERO_ACT_K
comptime AA = LIBERO_ACT_ADIM
comptime AQ = LIBERO_ACT_QPOS
comptime AIMG = LIBERO_ACT_IMG_ELEMS
comptime DIM = LIBERO_ACT_DIM


def _node_spread[NAME: StaticString, ROWS: Int, COLS: Int](
    mut tr: LiberoActTrainer[B, "gpu"], ctx: DeviceContext
) raises:
    """Sample 0 of node NAME read as [ROWS, COLS]: the mean over columns of the
    std across rows (how much the rows differ), and the tensor's RMS."""
    ref t = tr.graph.node_output[NAME]()
    t.download_enqueue(ctx)
    ctx.synchronize()
    t.download_finalize()
    var rms = 0.0
    for j in range(ROWS * COLS):
        var v = Float64(t.data[j])
        rms += v * v
    rms = sqrt(rms / Float64(ROWS * COLS))
    var spread = 0.0
    for c in range(COLS):
        var m = 0.0
        for r in range(ROWS):
            m += Float64(t.data[r * COLS + c])
        m /= Float64(ROWS)
        var ss = 0.0
        for r in range(ROWS):
            var d = Float64(t.data[r * COLS + c]) - m
            ss += d * d
        spread += sqrt(ss / Float64(ROWS))
    spread /= Float64(COLS)
    print("  node", NAME, "[" + String(ROWS) + " x " + String(COLS) + "]: spread across rows",
          _fd(spread, 5), "| rms", _fd(rms, 5),
          "| row 0 / row 1 / row 39, col 0:", _fd(Float64(t.data[0]), 4),
          _fd(Float64(t.data[COLS]), 4), _fd(Float64(t.data[(ROWS - 1) * COLS]), 4))


def _fd(x: Float64, d: Int) -> String:
    var scale = 1.0
    for _ in range(d):
        scale *= 10.0
    var r = Float64(Int(x * scale + (0.5 if x >= 0 else -0.5))) / scale
    var out = String(r)
    var dot = out.find(".")
    if dot < 0:
        out += "."
        dot = out.byte_length() - 1
    while out.byte_length() - dot - 1 < d:
        out += "0"
    return out^


def _pad(s: String, n: Int) -> String:
    var out = String(s)
    while out.byte_length() < n:
        out = " " + out
    return out^


def _pos_std(ref c: List[Scalar[DT]], base: Int, word: Int, n: Int) -> Float64:
    """Spread of word `word` across the first `n` positions of the chunk at `base`."""
    var m = 0.0
    for t in range(n):
        m += Float64(c[base + t * AA + word])
    m /= Float64(n)
    var ss = 0.0
    for t in range(n):
        var d = Float64(c[base + t * AA + word]) - m
        ss += d * d
    return sqrt(ss / Float64(n))


def main() raises:
    var args = argv()
    var act_dir = String("")
    var store = String("")
    var ckpt = String("best")
    var ep = 0
    var stride = 10
    var rows = 12
    var no_load = False
    var latent = String("prior")
    var i = 1
    while i < len(args):
        var s = String(args[i])
        if s == "--act" and i + 1 < len(args):
            act_dir = String(args[i + 1])
            i += 1
        elif s == "--store" and i + 1 < len(args):
            store = String(args[i + 1])
            i += 1
        elif s == "--ckpt" and i + 1 < len(args):
            ckpt = String(args[i + 1])
            i += 1
        elif s == "--ep" and i + 1 < len(args):
            ep = Int(String(args[i + 1]))
            i += 1
        elif s == "--stride" and i + 1 < len(args):
            stride = Int(String(args[i + 1]))
            i += 1
        elif s == "--rows" and i + 1 < len(args):
            rows = Int(String(args[i + 1]))
            i += 1
        elif s == "--no-load":
            no_load = True
        elif s == "--latent" and i + 1 < len(args):
            latent = String(args[i + 1])
            if latent != "prior" and latent != "posterior" and latent != "sample":
                raise Error("--latent must be prior (z=0), posterior or sample (z ~ N(0,I))")
            i += 1
        else:
            raise Error("libero act inspect: unknown argument '" + s + "' (--act DIR,"
                        " --store PATH, --ckpt best|last, --ep E, --stride S, --rows N)")
        i += 1
    if act_dir == "" or store == "":
        raise Error("libero act inspect: --act DIR and --store PATH are required")
    var path = act_dir + "/" + ckpt + ".ckpt"
    if not isdir(act_dir) and not no_load:
        # a RUN ID: its `checkpoints/<best|last>.ckpt` (or a .ckpt file as is)
        path = resolve_checkpoint(act_dir, ckpt)
    if not exists(path) and not no_load:
        raise Error("libero act inspect: no " + path)
    if rows < 1 or rows > B:
        raise Error("--rows must be in [1, " + String(B) + "]")

    var ctx = DeviceContext()
    var ds = LiberoActDataset(String(store), seed=0)
    if ep < 0 or ep >= ds.n_episodes():
        raise Error("--ep " + String(ep) + " outside the store's " + String(ds.n_episodes()) + " episodes")
    var ep_len = ds.store.episodes.length_of(ep)
    print("==============================================================================")
    print("ACT inspect —", path, "| store", store, "| episode", ep, "(" + String(ep_len) + " rows)",
          "| batch", B, "(the eval's LANES)")
    print("==============================================================================")

    var qpos = List[Scalar[DT]](length=B * AQ, fill=Scalar[DT](0))
    var images = List[Scalar[DT]](length=B * AIMG, fill=Scalar[DT](0))
    var actions = List[Scalar[DT]](length=B * K * AA, fill=Scalar[DT](0))
    var valid = List[Scalar[DT]](length=B * K, fill=Scalar[DT](1))
    var steps = List[Int]()
    for r in range(B):
        var st = (r if r < rows else 0) * stride
        if st >= ep_len:
            st = ep_len - 1
        steps.append(st)
        ds.fill_at[K](r, ep, st, qpos, images, actions, valid)
    # the recorded chunks, kept: predict needs an `actions` argument but nothing
    # downstream of the zeroed latent reads it
    var truth = List[Scalar[DT]](length=B * K * AA, fill=Scalar[DT](0))
    for j in range(B * K * AA):
        truth[j] = actions[j]
    var dummy = List[Scalar[DT]](length=B * K * AA, fill=Scalar[DT](0))
    var pred = List[Scalar[DT]](length=B * K * AA, fill=Scalar[DT](0))

    var tr = LiberoActTrainer[B, "gpu"].make(ctx=ctx)
    if no_load:
        print("  --no-load: a FRESHLY INITIALISED model (Kaiming), nothing loaded")
    else:
        tr.load(path)
    if latent == "posterior":
        print("  --latent posterior: z = mu(qpos, TRUE chunk); the eval uses z = 0")
        tr.predict_with_posterior(qpos, images, truth, valid, pred)
    elif latent == "sample":
        print("  --latent sample: z ~ N(0, I), one prior draw per row; the eval uses z = 0")
        tr.predict_prior_sample(qpos, images, dummy, valid, pred)
    else:
        tr.predict(qpos, images, dummy, valid, pred)
    # the position path, node by node: the learned queries, the decoder's
    # output, its norm, the action head
    _node_spread["qpe", K, DIM](tr, ctx)
    _node_spread["hs", K, DIM](tr, ctx)
    _node_spread["hsn", K, DIM](tr, ctx)
    _node_spread["ahat", K, AA](tr, ctx)

    print("  normalised units; x y z = words 0 1 2; spread = std over the chunk's VALID positions")
    print("  row  step | pred x @ pos 0    10    20    39 | true x @ 0    10    20    39 |"
          " spread pred x y z | true x y z | L1 per-pos | L1 flat")
    var sum_pred_spread = 0.0
    var sum_true_spread = 0.0
    var sum_l1_pos = 0.0
    var sum_l1_flat = 0.0
    var probe = [0, 10, 20, 39]
    for r in range(rows):
        var base = r * K * AA
        var n = 0
        for t in range(K):
            if Float64(valid[r * K + t]) > 0.5:
                n += 1
        if n < 2:
            continue
        var line = String("  ") + _pad(String(r), 3) + " " + _pad(String(steps[r]), 5) + " |"
        for p in probe:
            var pp = p if p < n else n - 1
            line += " " + _pad(_fd(Float64(pred[base + pp * AA]), 2), 6)
        line += " |"
        for p in probe:
            var pp = p if p < n else n - 1
            line += " " + _pad(_fd(Float64(truth[base + pp * AA]), 2), 6)
        line += " |"
        var ps = 0.0
        var ts = 0.0
        for w in range(3):
            var a = _pos_std(pred, base, w, n)
            var b = _pos_std(truth, base, w, n)
            ps += a
            ts += b
            line += " " + _fd(a, 3)
        line += " |"
        for w in range(3):
            line += " " + _fd(_pos_std(truth, base, w, n), 3)
        # L1 of the prediction vs the recorded chunk, and of its flat version
        var l1p = 0.0
        var l1f = 0.0
        for w in range(AA):
            var m = 0.0
            for t in range(n):
                m += Float64(pred[base + t * AA + w])
            m /= Float64(n)
            for t in range(n):
                var tv = Float64(truth[base + t * AA + w])
                l1p += abs(Float64(pred[base + t * AA + w]) - tv)
                l1f += abs(m - tv)
        l1p /= Float64(n * AA)
        l1f /= Float64(n * AA)
        line += " | " + _fd(l1p, 3) + " | " + _fd(l1f, 3)
        print(line)
        sum_pred_spread += ps / 3.0
        sum_true_spread += ts / 3.0
        sum_l1_pos += l1p
        sum_l1_flat += l1f
    var nr = Float64(rows)
    print("  mean over rows: pred spread", _fd(sum_pred_spread / nr, 4), "| true spread",
          _fd(sum_true_spread / nr, 4), "| L1 per-pos", _fd(sum_l1_pos / nr, 4),
          "| L1 flat", _fd(sum_l1_flat / nr, 4))
    if sum_pred_spread / nr < 0.02 * (sum_true_spread / nr):
        print("  ⚠ FLAT: the predicted chunks do not vary across positions (< 2% of the"
              " recorded spread) — the policy drives with the chunk's mean per state")
    else:
        print("  the predicted chunks vary across positions")
