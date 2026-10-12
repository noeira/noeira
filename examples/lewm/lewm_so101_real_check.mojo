"""S2 — the SO-101 world model on REAL frames, offline (no arm).

noeira-docs/SO101_LEWM_PLAN.md S2 (2), the prediction test. A model trained
on sim renders (`lewm_so101_train.mojo`) predicts the next latent of real
teleop windows — the follower's two cameras, undistorted to the sim pinhole
and area-resized like the renders, the leader's commanded-target changes as
the actions — and the same on sim windows through the same code:

    pred loss real  vs  pred loss sim        (eval mode, mean over windows)

Gate: real within ~2x sim. It also prints the real actions' per-joint mean
and std in the SIM normaliser's units: teleop has no target-mode clamp, so
actions far outside the sim distribution are a confound to read first.

    pixi run -e apple mojo build -I . -D SO101_WM_B=16 \\
        examples/lewm/lewm_so101_real_check.mojo -o build/so101_real_check
    build/so101_real_check --dump <run>/epoch_7 --stats <run>/action_stats.txt \\
        --real ~/.cache/noeira/act_so101/so101-tower__cube-in-bowl-printed_240x320_undist.h5 \\
        [--sim teacher.rendered.h5] [--batches 40]

v2 (`-D SO101_WM_PROPRIO=1`, a `-D SO101_WM_PROPRIO=1` training run): the
windows carry proprio, normalised with the run's `prop_stats.txt` (`--pstats`,
default beside `--stats`).

Per data set, mean over `--batches` batches (eval mode):

    pred       the prediction loss
    copy-last  the context latents themselves as the prediction (frame t for
               t + 1): a predictor that does not beat it has learnt no
               dynamics (v1 epoch 7: 1-step error ~ copy in the S3 loop)
    shuffled   each window given the NEXT window's actions: x pred = how much
               the predictor reads them (v1: x 1.26 sim, x 1.10 real)
    no-prop    (v2) the proprio words at their mean (z = 0): how much the
               model leans on them instead of the pictures
"""

from std.sys import argv
from std.sys.defines import get_defined_int
from std.math import sqrt
from std.random import seed
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.experimental.lewm.ref_trainer import LeWMRefTrainer, WMWindowLoss
from noeira.experimental.lewm.so101_trainer import LeWMProprioTrainer
from noeira.experimental.lewm.so101_data import So101WMData, ACT_IN, CAMS, JOINTS, PROP, T


comptime R = 112
comptime B = get_defined_int["SO101_WM_B", 16]()
comptime PROPRIO = get_defined_int["SO101_WM_PROPRIO", 0]() == 1
comptime TR = LeWMRefTrainer["gpu", B, CAMS * 3, R, ACT_IN]
comptime TRP = LeWMProprioTrainer["gpu", B, CAMS * 3, R, ACT_IN, PROP]
comptime PIX = T * CAMS * 3 * R * R
comptime ACT = T * ACT_IN
comptime PRO = T * PROP


def _arg(args: List[String], key: String, default: String) -> String:
    for i in range(len(args) - 1):
        if args[i] == key:
            return args[i + 1]
    return default


struct _Losses(Copyable, Movable):
    var pred: Float64
    var copy_last: Float64
    var shuffled: Float64
    var no_prop: Float64

    def __init__(out self):
        self.pred = 0.0
        self.copy_last = 0.0
        self.shuffled = 0.0
        self.no_prop = 0.0


def _losses[TRT: WMWindowLoss](
    mut tr: TRT, data: So101WMData[R], batches: Int,
) raises -> _Losses:
    """The four losses (see the header) on the SAME windows, averaged over
    `batches` batches (train + val starts alternately)."""
    var pix = List[Scalar[DT]](length=B * PIX, fill=Scalar[DT](0))
    var act = List[Scalar[DT]](length=B * ACT, fill=Scalar[DT](0))
    var pro = List[Scalar[DT]](length=B * PRO, fill=Scalar[DT](0))
    var zero = List[Scalar[DT]](length=B * PRO, fill=Scalar[DT](0))
    var out = _Losses()
    for k in range(batches):
        var starts = data.sample(k % 2 == 1 and len(data.val_starts) > 0, B)
        data.fill(starts, pix, act)
        data.fill_prop(starts, pro)
        var l = tr.window_losses(pix, act, pro)
        out.pred += l[0]
        out.copy_last += l[1]
        var rolled = act.copy()
        for i in range(len(act)):
            rolled[i] = act[(i + ACT) % len(act)]
        out.shuffled += tr.window_losses(pix, rolled, pro)[0]
        comptime if PROPRIO:
            out.no_prop += tr.window_losses(pix, act, zero)[0]
    var n = Float64(batches)
    out.pred /= n
    out.copy_last /= n
    out.shuffled /= n
    out.no_prop /= n
    return out^


def _report(label: String, l: _Losses):
    var line = String("  ") + label + " pred " + String(Float32(l.pred))
    line += " | copy-last " + String(Float32(l.copy_last)) + " (pred / copy " + String(Float32(l.pred / l.copy_last)) + ")"
    line += " | actions shuffled x" + String(Float32(l.shuffled / l.pred))
    comptime if PROPRIO:
        line += " | proprio zeroed x" + String(Float32(l.no_prop / l.pred))
    print(line)


def _action_spread(data: So101WMData[R], label: String):
    """The data's target changes in the (loaded) normaliser's units."""
    var line = String("  ") + label + " actions in sim units, per joint (mean / std):"
    for j in range(JOINTS):
        var m = 0.0
        var s = 0.0
        for r in range(data.n_rows):
            m += (Float64(data.dtarget[r * JOINTS + j]) - data.a_mean[j]) / data.a_std[j]
        m /= Float64(data.n_rows)
        for r in range(data.n_rows):
            var z = (Float64(data.dtarget[r * JOINTS + j]) - data.a_mean[j]) / data.a_std[j]
            s += (z - m) ** 2
        line += " " + String(Float32(m)) + "/" + String(Float32(sqrt(s / Float64(data.n_rows))))
    print(line)


def _check[TRT: WMWindowLoss](
    mut tr: TRT, real: String, sim: String, stats: String, pstats: String, batches: Int,
) raises:
    var rp = List[String]()
    rp.append(real)
    var rd = So101WMData[R](rp, teleop=True)
    rd.load_action_stats(stats)
    if PROPRIO:
        rd.load_prop_stats(pstats)
    print("  real:", rd.n_rows, "rows,", rd.n_episodes, "episodes,",
          len(rd.train_starts) + len(rd.val_starts), "windows")
    _action_spread(rd, String("real"))
    var lr_ = _losses(tr, rd, batches)
    _report(String("real"), lr_)
    if sim.byte_length() > 0:
        var sp = List[String]()
        sp.append(sim)
        var sd = So101WMData[R](sp)
        sd.load_action_stats(stats)
        if PROPRIO:
            sd.load_prop_stats(pstats)
        _action_spread(sd, String("sim"))
        var ls = _losses(tr, sd, batches)
        _report(String("sim "), ls)
        print("  real / sim pred:", Float32(lr_.pred / ls.pred), "(gate: <= ~2)")


def main() raises:
    var args = List[String]()
    for a in argv():
        args.append(String(a))
    var dump = _arg(args, "--dump", "")
    var stats = _arg(args, "--stats", "")
    var real = _arg(args, "--real", "")
    var sim = _arg(args, "--sim", "")
    var batches = Int(_arg(args, "--batches", "40"))
    if dump.byte_length() == 0 or stats.byte_length() == 0 or real.byte_length() == 0:
        raise Error("--dump, --stats and --real are required")
    var pstats = _arg(args, "--pstats", String(stats[byte = 0 : stats.rfind("/") + 1]) + "prop_stats.txt")
    seed(7)
    var c = DeviceContext()
    var ctx = Optional(c)
    comptime if PROPRIO:
        var tr = TRP(ctx, dropout=False)
        var n = tr.load(dump)
        tr.set_eval()
        print("S2 prediction check | v2 (proprio) model", dump, "(", n, "tensors ) | batch", B, "x", batches)
        _check(tr, real, sim, stats, pstats, batches)
    else:
        var tr = TR(ctx, dropout=False)
        var n = tr.load(dump)
        tr.set_eval()
        print("S2 prediction check | v1 model", dump, "(", n, "tensors ) | batch", B, "x", batches)
        _check(tr, real, sim, stats, pstats, batches)
