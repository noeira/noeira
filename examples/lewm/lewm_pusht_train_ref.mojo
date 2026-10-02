"""P6.2 — train the reference-exact LeWM on PushT the way `train.py` does.

docs/LEWM_REOPEN_PLAN.md P6. `ref_trainer.LeWMRefTrainer` (gated against
torch: G6a five steps, P6.1 1000 steps at batch 128) driven like le-wm-main's
`train.py`:

  * data: the reference's own 90 / 10 split (`tools/lewm/export_split.py`:
    spt's `random_split`, seed 3072); every epoch the train clips reshuffled,
    batches of 128, the last partial batch dropped; read on a thread
    (`batch_loader.mojo`), actions z-scored with the normaliser of the parity
    run (`run.action_mean|std`: train.py's float32 mean / unbiased std);
  * optimizer: AdamW lr 5e-5, wd 1e-3 on every parameter, clip 1.0; the LR
    `recipe_lr` (spt's `LinearWarmupCosineAnnealingLR` defaults, stepped
    every step) over epochs x batches — the schedule depends on `--epochs`;
  * the predictor's dropout 0.1 on; SIGReg resampled every step; BN train;
  * validation after each epoch on the validation split in eval mode (BN
    running stats, no dropout), as Lightning's `model.eval()`;
  * a checkpoint after each epoch: `<out>/epoch_<k>/` in the dump format the
    column-M runner reads (`--dump`), and `<out>/train_log.csv`.

Not the recipe: float32 / TF32 instead of bf16; our shuffle RNG is not torch's.

    pixi run -e nvidia mojo run -I . examples/lewm/lewm_pusht_train_ref.mojo \\
        --h5 .../pusht_expert_train.h5 --split /workspace/lewm_split \\
        --init /workspace/lewm_parity128 --stats /workspace/lewm_parity128 \\
        --out /workspace/lewm_train --epochs 10
"""

from std.sys import argv
from std.time import perf_counter_ns
from std.random import seed, random_ui64
from std.os import makedirs
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.deep_agents.act.refload import RefDump
from noeira.experimental.lewm.ref_trainer import LeWMRefTrainer, recipe_lr, RefStepStats
from noeira.experimental.lewm.batch_loader import LewmBatchLoaderThread


comptime B = 128


def _shuffled(idx: List[Scalar[DT]], s: UInt64) -> List[Int]:
    """Fisher-Yates over the split's clip indices, seeded per epoch."""
    seed(Int(s))
    var out = List[Int](capacity=len(idx))
    for v in idx:
        out.append(Int(v))
    for i in range(len(out) - 1, 0, -1):
        var j = Int(random_ui64(0, UInt64(i)))
        var t = out[i]
        out[i] = out[j]
        out[j] = t
    return out^


def main() raises:
    var h5 = String("/workspace/lewm_session_a/stablewm/pusht_expert_train.h5")
    var split = String("/workspace/lewm_split")
    var init = String("/workspace/lewm_parity128")
    var stats_dir = String("/workspace/lewm_parity128")
    var out = String("/workspace/lewm_train")
    var epochs = 10
    var run_seed = 0
    var val_batches = 0  # 0 = the whole validation split
    var log_every = 100
    var max_steps = -1   # debug: stop an epoch early
    var args = argv()
    var i = 1
    while i < len(args):
        var a = String(args[i])
        if a == "--h5":
            h5 = String(args[i + 1]); i += 1
        elif a == "--split":
            split = String(args[i + 1]); i += 1
        elif a == "--init":
            init = String(args[i + 1]); i += 1
        elif a == "--stats":
            stats_dir = String(args[i + 1]); i += 1
        elif a == "--out":
            out = String(args[i + 1]); i += 1
        elif a == "--epochs":
            epochs = Int(String(args[i + 1])); i += 1
        elif a == "--seed":
            run_seed = Int(String(args[i + 1])); i += 1
        elif a == "--val-batches":
            val_batches = Int(String(args[i + 1])); i += 1
        elif a == "--log-every":
            log_every = Int(String(args[i + 1])); i += 1
        elif a == "--max-steps":
            max_steps = Int(String(args[i + 1])); i += 1
        else:
            raise Error("unknown argument " + a)
        i += 1

    makedirs(out, exist_ok=True)
    var sp = RefDump(split)
    var train_idx = sp.get(String("split.train"))
    var val_idx = sp.get(String("split.val"))
    var st = RefDump(stats_dir)
    var a_mean = st.get(String("run.action_mean"))
    var a_std = st.get(String("run.action_std"))
    var am: List[Float32] = [Float32(a_mean[0]), Float32(a_mean[1])]
    var asd: List[Float32] = [Float32(a_std[0]), Float32(a_std[1])]
    var steps_per_epoch = len(train_idx) // B
    if max_steps > 0:
        steps_per_epoch = min(steps_per_epoch, max_steps)
    var total = epochs * steps_per_epoch
    var n_val = len(val_idx) // B
    if val_batches > 0:
        n_val = min(n_val, val_batches)

    var c = DeviceContext()
    var ctx = Optional(c)
    var tr = LeWMRefTrainer["gpu", B](ctx, lr=5e-5, wd=1e-3, max_norm=1.0, dropout=True)
    var n = tr.load(init)
    var stages: List[Pointer[Scalar[DType.uint8], MutAnyOrigin]] = [tr.staging(0), tr.staging(1)]
    print("LeWM training (P6.2):", epochs, "epochs x", steps_per_epoch, "steps x batch", B,
          "=", total, "steps; train", len(train_idx), "clips, val", n_val, "batches")
    print("  init", init, "(", n, "tensors );  action stats", am[0], am[1], asd[0], asd[1])

    var log = String("step,epoch,lr,loss,pred_loss,sigreg_loss,grad_norm,wall_s\n")
    var vlog = String("epoch,val_loss,val_pred_loss,val_sigreg_loss\n")
    var step = 0
    var t_all = perf_counter_ns()
    for ep in range(epochs):
        var t_ep = perf_counter_ns()
        var clips = _shuffled(train_idx, UInt64(run_seed * 1000003 + ep + 1))
        var use = List[Int](capacity=steps_per_epoch * B)
        for k in range(steps_per_epoch * B):
            use.append(clips[k])
        var loader = LewmBatchLoaderThread[B](h5, use, am, asd, stages)
        tr.set_train()
        loader.request(0)
        var acc = List[Float64](length=4, fill=0.0)
        var n_acc = 0
        for s in range(steps_per_epoch):
            _ = loader.wait(s)
            if s + 1 < steps_per_epoch:
                loader.request(s + 1)
            var lr = recipe_lr(step, total)
            tr.set_lr(lr)
            tr.submit_staged(loader.actions(s), s % 2)
            var r = tr.finish()
            acc[0] += r.loss
            acc[1] += r.pred_loss
            acc[2] += r.sigreg_loss
            acc[3] += r.grad_norm
            n_acc += 1
            step += 1
            if step % log_every == 0 or s == steps_per_epoch - 1:
                var wall = Float64(perf_counter_ns() - t_all) / 1e9
                var line = String(step) + "," + String(ep) + "," + String(lr)
                for k in range(4):
                    line += "," + String(acc[k] / Float64(n_acc))
                line += "," + String(wall)
                log += line + "\n"
                var eta = wall / Float64(step) * Float64(total - step) / 3600.0
                print("  step", step, "/", total, " epoch", ep, " lr", Float32(lr),
                      " loss", Float32(acc[0] / Float64(n_acc)), " pred", Float32(acc[1] / Float64(n_acc)),
                      " sigreg", Float32(acc[2] / Float64(n_acc)), " norm", Float32(acc[3] / Float64(n_acc)),
                      " ", Float32(wall / Float64(step)), "s/step  eta", Float32(eta), "h")
                with open(out + "/train_log.csv", "w") as f:
                    f.write(log)
                for k in range(4):
                    acc[k] = 0.0
                n_acc = 0
        loader.stop()

        # validation: eval mode, the reference's held-out clips
        var vuse = List[Int](capacity=n_val * B)
        for k in range(n_val * B):
            vuse.append(Int(val_idx[k]))
        var vloader = LewmBatchLoaderThread[B](h5, vuse, am, asd, stages)
        tr.set_eval()
        vloader.request(0)
        var v = List[Float64](length=3, fill=0.0)
        for s in range(n_val):
            _ = vloader.wait(s)
            if s + 1 < n_val:
                vloader.request(s + 1)
            var r = tr.eval_staged(vloader.actions(s), s % 2)
            v[0] += r.loss
            v[1] += r.pred_loss
            v[2] += r.sigreg_loss
        vloader.stop()
        tr.set_train()
        for k in range(3):
            v[k] /= Float64(max(1, n_val))
        vlog += String(ep) + "," + String(v[0]) + "," + String(v[1]) + "," + String(v[2]) + "\n"
        with open(out + "/val_log.csv", "w") as f:
            f.write(vlog)
        var nd = tr.save_dump(out + "/epoch_" + String(ep))
        print("  epoch", ep, "done in", Float32(Float64(perf_counter_ns() - t_ep) / 60e9), "min;",
              " val loss", Float32(v[0]), " pred", Float32(v[1]), " sigreg", Float32(v[2]),
              "; saved", nd, "tensors to", out + "/epoch_" + String(ep))
    print("done:", step, "steps in", Float32(Float64(perf_counter_ns() - t_all) / 3600e9), "h")
