"""Train a LeWM world model on the SO-101 tower's rendered teacher rollouts.

noeira-docs/SO101_LEWM_PLAN.md S1. The reference LeWM stack
(`experimental/lewm/ref_*`, the published PushT model's architecture and
recipe) re-shaped for the rig:

    input    the two cameras stacked as 6 channels at R × R (R = 112,
             patch 14: 64 tokens), ImageNet-normalised
    actions  4 frames 5 ticks apart; each frame's block = the 5 per-tick
             changes of the commanded target × 6 joints = 30, z-scored
             (`so101_data.So101WMData`)
    recipe   AdamW (decay on every parameter), predictor dropout 0.1,
             SIGReg λ 0.09, clip 1.0, warmup-cosine LR from `--lr`

v1 has no proprioception (the arm is only in the pictures); S3 found its
predictor barely reads its actions. v2 (`-D SO101_WM_PROPRIO=1`) adds each
frame's q, qd and target lead to the latent (`ref_model.LeWMLossGraphProprio`).

    pixi run -e nvidia mojo run -I . examples/lewm/lewm_so101_train.mojo \\
        --stores a.rendered.h5,b.rendered.h5 --epochs 10 --out runs/so101_wm

Writes `<out>/epoch_<k>/` (the converted-dump format `load_ref` reads),
`<out>/action_stats.txt` (per-joint mean and std of the target change, the
normaliser the critic must reuse) and `<out>/log.csv`.
"""

from std.sys import argv
from std.sys.defines import get_defined_int
from std.os import makedirs
from std.time import perf_counter_ns
from std.random import seed
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.experimental.lewm.ref_trainer import LeWMRefTrainer, RefStepStats, recipe_lr
from noeira.experimental.lewm.so101_trainer import LeWMProprioTrainer
from noeira.experimental.lewm.so101_data import So101WMData, T, ACT_IN, CAMS, JOINTS, PROP


comptime R = 112
comptime B = get_defined_int["SO101_WM_B", 128]()
"""Batch (`-D SO101_WM_B=8` for a laptop smoke)."""
comptime PROPRIO = get_defined_int["SO101_WM_PROPRIO", 0]() == 1
"""v2 (`-D SO101_WM_PROPRIO=1`): proprioception in the latent
(`LeWMProprioTrainer`; writes `prop_stats.txt` too)."""
comptime TR = LeWMRefTrainer["gpu", B, CAMS * 3, R, ACT_IN]
comptime TRP = LeWMProprioTrainer["gpu", B, CAMS * 3, R, ACT_IN, PROP]
comptime VAL_BATCHES = 20


def _arg(args: List[String], key: String, default: String) -> String:
    for i in range(len(args) - 1):
        if args[i] == key:
            return args[i + 1]
    return default


def main() raises:
    var args = List[String]()
    for a in argv():
        args.append(String(a))
    var stores = _arg(args, "--stores", "")
    if stores.byte_length() == 0:
        raise Error("--stores a.h5[,b.h5] is required")
    var epochs = Float64(_arg(args, "--epochs", "10"))
    var out = _arg(args, "--out", "runs/so101_wm")
    var lr = Float64(_arg(args, "--lr", "5e-5"))
    var max_steps = Int(_arg(args, "--max-steps", "0"))
    var log_every = Int(_arg(args, "--log-every", "50"))
    seed(Int(_arg(args, "--seed", "3072")))

    var paths = List[String]()
    for p in stores.split(","):
        paths.append(String(p))
    var t0 = perf_counter_ns()
    var data = So101WMData[R](paths)
    print("SO-101 WM data:", data.n_rows, "rows,", data.n_episodes, "episodes |",
          len(data.train_starts), "train /", len(data.val_starts), "val windows | load",
          Float64(perf_counter_ns() - t0) / 1e9, "s")
    makedirs(out, exist_ok=True)
    with open(out + "/action_stats.txt", "w") as f:
        var s = String("# per-joint change of the commanded target (rad): mean std\n")
        for j in range(JOINTS):
            s += String(data.a_mean[j]) + " " + String(data.a_std[j]) + "\n"
        f.write(s)

    if PROPRIO:
        with open(out + "/prop_stats.txt", "w") as f:
            f.write(data.prop_stats_text())
    var per_epoch = len(data.train_starts) // B
    var total = Int(epochs * Float64(per_epoch))
    if max_steps > 0:
        total = min(total, max_steps)
    print("training:", total, "steps of", B, "|", per_epoch, "per epoch | lr", lr,
          "| proprio", PROPRIO)
    var c = DeviceContext()
    var ctx = Optional(c)
    var log = _Log(out, per_epoch, log_every)
    comptime if PROPRIO:
        _train_v2(data, log, total, lr, ctx)
    else:
        _train_v1(data, log, total, lr, ctx)
    print("done:", total, "steps |", out)


struct _Log(Movable):
    """log.csv, the console line and the epoch bookkeeping, shared by both
    trainers' loops."""

    var out: String
    var per_epoch: Int
    var log_every: Int
    var text: String
    var t_start: Int
    var acc_pred: Float64
    var acc_n: Int

    def __init__(out self, out_dir: String, per_epoch: Int, log_every: Int):
        self.out = out_dir
        self.per_epoch = per_epoch
        self.log_every = log_every
        self.text = String("step,epoch,lr,loss,pred,sigreg,grad_norm,val_pred,secs\n")
        self.t_start = perf_counter_ns()
        self.acc_pred = 0.0
        self.acc_n = 0

    def end_epoch(self, step: Int, total: Int) -> Bool:
        return (step + 1) % self.per_epoch == 0 or step + 1 == total

    def dump_dir(self, step: Int) -> String:
        return self.out + "/epoch_" + String((step + 1 + self.per_epoch - 1) // self.per_epoch - 1)

    def after(
        mut self, step: Int, total: Int, lr_s: Float64, st: RefStepStats, val_pred: Float64
    ) raises:
        self.acc_pred += st.pred_loss
        self.acc_n += 1
        if (step + 1) % self.log_every != 0 and not self.end_epoch(step, total):
            return
        var secs = Float64(perf_counter_ns() - self.t_start) / 1e9
        var mean = self.acc_pred / Float64(self.acc_n)
        var line = String(step + 1) + "," + String(Float64(step + 1) / Float64(self.per_epoch)) + ","
        line += String(lr_s) + "," + String(st.loss) + "," + String(mean) + ","
        line += String(st.sigreg_loss) + "," + String(st.grad_norm) + "," + String(val_pred) + ","
        line += String(secs) + "\n"
        self.text += line
        with open(self.out + "/log.csv", "w") as f:
            f.write(self.text)
        print("step", step + 1, "| pred", mean, "| sigreg", st.sigreg_loss,
              "| norm", st.grad_norm, "| val pred", val_pred, "|",
              secs / Float64(step + 1), "s/step")
        self.acc_pred = 0.0
        self.acc_n = 0


def _train_v1(
    data: So101WMData[R], mut log: _Log, total: Int, lr: Float64, ctx: Optional[DeviceContext]
) raises:
    var tr = TR(ctx, lr=lr, wd=1e-3, max_norm=1.0, sigreg_lambda=0.09, dropout=True)
    tr.set_train()
    var pix = List[Scalar[DT]](length=B * TR.PIX, fill=Scalar[DT](0))
    var act = List[Scalar[DT]](length=B * TR.ACT, fill=Scalar[DT](0))
    for step in range(total):
        var lr_s = recipe_lr(step, total, lr)
        tr.set_lr(lr_s)
        data.fill(data.sample(False, B), pix, act)
        var st = tr.train_step(pix, act)
        var val_pred = -1.0
        if log.end_epoch(step, total):
            if len(data.val_starts) > 0:
                tr.set_eval()
                val_pred = 0.0
                for _ in range(VAL_BATCHES):
                    data.fill(data.sample(True, B), pix, act)
                    val_pred += tr.loss_of(pix, act).pred_loss
                val_pred /= Float64(VAL_BATCHES)
                tr.set_train()
            _ = tr.save_dump(log.dump_dir(step))
        log.after(step, total, lr_s, st, val_pred)


def _train_v2(
    data: So101WMData[R], mut log: _Log, total: Int, lr: Float64, ctx: Optional[DeviceContext]
) raises:
    var tr = TRP(ctx, lr=lr, wd=1e-3, max_norm=1.0, sigreg_lambda=0.09, dropout=True)
    tr.set_train()
    var pix = List[Scalar[DT]](length=B * TRP.PIX, fill=Scalar[DT](0))
    var act = List[Scalar[DT]](length=B * TRP.ACT, fill=Scalar[DT](0))
    var pro = List[Scalar[DT]](length=B * TRP.PRO, fill=Scalar[DT](0))
    for step in range(total):
        var lr_s = recipe_lr(step, total, lr)
        tr.set_lr(lr_s)
        var starts = data.sample(False, B)
        data.fill(starts, pix, act)
        data.fill_prop(starts, pro)
        var st = tr.train_step(pix, act, pro)
        var val_pred = -1.0
        if log.end_epoch(step, total):
            if len(data.val_starts) > 0:
                tr.set_eval()
                val_pred = 0.0
                for _ in range(VAL_BATCHES):
                    var vs = data.sample(True, B)
                    data.fill(vs, pix, act)
                    data.fill_prop(vs, pro)
                    val_pred += tr.loss_of(pix, act, pro).pred_loss
                val_pred /= Float64(VAL_BATCHES)
                tr.set_train()
            _ = tr.save_dump(log.dump_dir(step))
        log.after(step, total, lr_s, st, val_pred)
