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
"""

from std.sys import argv
from std.sys.defines import get_defined_int
from std.math import sqrt
from std.random import seed
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.experimental.lewm.ref_trainer import LeWMRefTrainer
from noeira.experimental.lewm.so101_data import So101WMData, ACT_IN, CAMS, JOINTS


comptime R = 112
comptime B = get_defined_int["SO101_WM_B", 16]()
comptime TR = LeWMRefTrainer["gpu", B, CAMS * 3, R, ACT_IN]


def _arg(args: List[String], key: String, default: String) -> String:
    for i in range(len(args) - 1):
        if args[i] == key:
            return args[i + 1]
    return default


def _pred_loss(
    mut tr: TR, data: So101WMData[R], batches: Int,
) raises -> Tuple[Float64, Float64]:
    """Mean and std over `batches` random windows (train + val starts)."""
    var pix = List[Scalar[DT]](length=B * TR.PIX, fill=Scalar[DT](0))
    var act = List[Scalar[DT]](length=B * TR.ACT, fill=Scalar[DT](0))
    var vals = List[Float64]()
    for k in range(batches):
        var starts = data.sample(k % 2 == 1 and len(data.val_starts) > 0, B)
        data.fill(starts, pix, act)
        vals.append(tr.loss_of(pix, act).pred_loss)
    var m = 0.0
    for v in vals:
        m += v
    m /= Float64(len(vals))
    var s = 0.0
    for v in vals:
        s += (v - m) ** 2
    return (m, sqrt(s / Float64(len(vals))))


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
    seed(7)
    var c = DeviceContext()
    var ctx = Optional(c)
    var tr = TR(ctx, dropout=False)
    var n = tr.load(dump)
    tr.set_eval()
    print("S2 prediction check | model", dump, "(", n, "tensors ) | batch", B, "x", batches)

    var rp = List[String]()
    rp.append(real)
    var rd = So101WMData[R](rp, teleop=True)
    rd.load_action_stats(stats)
    print("  real:", rd.n_rows, "rows,", rd.n_episodes, "episodes,",
          len(rd.train_starts) + len(rd.val_starts), "windows")
    _action_spread(rd, String("real"))
    var lr_ = _pred_loss(tr, rd, batches)
    print("  real pred loss:", lr_[0], "+/-", lr_[1])
    if sim.byte_length() > 0:
        var sp = List[String]()
        sp.append(sim)
        var sd = So101WMData[R](sp)
        sd.load_action_stats(stats)
        _action_spread(sd, String("sim"))
        var ls = _pred_loss(tr, sd, batches)
        print("  sim pred loss:", ls[0], "+/-", ls[1])
        print("  real / sim:", lr_[0] / ls[0], "(gate: <= ~2)")
