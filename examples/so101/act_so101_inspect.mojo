"""ACT ON THE SO-101 — WHAT ONE CHECKPOINT PREDICTS FOR A STORE ROW, POSITION BY POSITION.

    pixi run -e nvidia mojo run -I . examples/so101/act_so101_inspect.mojo \
        --act <run_id> \
        --store ~/.cache/noeira/act_so101/so101-tower__cube-in-bowl_240x320.h5

The SO-101 twin of `examples/libero/libero_act_inspect.mojo` (read its header
for the why). Every ACT fit before 2026-09-21 predicted ONE action for all K
chunk positions — the decoder's zero targets were a flat attractor — and the
open-loop eval cannot show it, because the temporal ensemble averages the
chunks it would expose. This reads a checkpoint through `predict` (the
deployment's path: latent zero, eval mode) on store rows and prints, per row,
the predicted chunk's spread across its K positions beside the recorded
chunk's, the per-position vs flat L1, and the query / decoder / action-head
nodes' spread. A spread under a few % of the recorded one is the flat
failure: do not arm the robot on it.

Flags: `--act RUN_ID|DIR` (a run id, found under `runs/` or
`projects/*/runs/`, or a directory holding best.ckpt + norm.json), `--store PATH`, `--ckpt best|last`,
`--ep E` (default 0), `--stride S` (default 30 = 1 s), `--rows N` (≤ 16,
default 12), `--latent prior|posterior|sample`, `--no-load`. Normalised units.
"""

from std.os.path import exists, isdir
from std.sys import argv
from std.math import sqrt
from max.gpu.host import DeviceContext

from noeira.core.run import resolve_checkpoint
from noeira.nn.constants import DT
from noeira.deep_agents.act.config import (
    RUN_DEC_LAYERS, RUN_DIM, RUN_ENC_LAYERS, RUN_FF, RUN_HEADS, RUN_K,
    RUN_LATENT, SO101_ADIM, SO101_IMG_H, SO101_IMG_W, SO101_N_CAM, SO101_QPOS,
)
from noeira.deep_agents.act.data import ACTDataset
from noeira.deep_agents.act.trainer import ACTTrainer

comptime B = 16
"""The training batch; `predict` is batch-independent in eval mode."""
comptime K = RUN_K
comptime AA = SO101_ADIM
comptime AQ = SO101_QPOS
comptime DIM = RUN_DIM
comptime AIMG = SO101_N_CAM * 3 * SO101_IMG_H * SO101_IMG_W
comptime T = ACTTrainer[
    SO101_QPOS, SO101_ADIM, SO101_N_CAM, SO101_IMG_H, SO101_IMG_W, K, DIM,
    RUN_HEADS, RUN_FF, RUN_LATENT, RUN_ENC_LAYERS, RUN_DEC_LAYERS, B, 0.1, "gpu",
]


def _node_spread[NAME: StaticString, ROWS: Int, COLS: Int](
    mut tr: T, ctx: DeviceContext
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
    var stride = 30
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
            raise Error("so101 act inspect: unknown argument '" + s + "' (--act DIR,"
                        " --store PATH, --ckpt best|last, --ep E, --stride S, --rows N)")
        i += 1
    if act_dir == "" or store == "":
        raise Error("so101 act inspect: --act DIR and --store PATH are required")
    var path = act_dir + "/" + ckpt + ".ckpt"
    if not isdir(act_dir) and not no_load:
        # a RUN ID: its `checkpoints/<best|last>.ckpt` (or a .ckpt file as is)
        path = resolve_checkpoint(act_dir, ckpt)
    if not exists(path) and not no_load:
        raise Error("so101 act inspect: no " + path)
    if rows < 1 or rows > B:
        raise Error("--rows must be in [1, " + String(B) + "]")

    var ctx = DeviceContext()
    var ds = ACTDataset[SO101_QPOS, SO101_ADIM, SO101_N_CAM, SO101_IMG_H, SO101_IMG_W](
        String(store), seed=0
    )
    if ep < 0 or ep >= ds.n_episodes():
        raise Error("--ep " + String(ep) + " outside the store's " + String(ds.n_episodes()) + " episodes")
    var ep_len = ds.store.episodes.length_of(ep)
    print("==============================================================================")
    print("ACT inspect —", path, "| store", store, "| episode", ep, "(" + String(ep_len) + " rows)",
          "| batch", B)
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

    var tr = T.make(ctx=ctx)
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

    print("  normalised units; the probed word is joint 0 (shoulder pan); spread = mean over joints 0-2")
    print("  row  step | pred j0 @ pos 0  K/4  K/2  K-1 | true j0 @ 0  K/4  K/2  K-1 |"
          " spread pred j0 j1 j2 | true j0 j1 j2 | L1 per-pos | L1 flat")
    var sum_pred_spread = 0.0
    var sum_true_spread = 0.0
    var sum_l1_pos = 0.0
    var sum_l1_flat = 0.0
    var probe = [0, K // 4, K // 2, K - 1]
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
