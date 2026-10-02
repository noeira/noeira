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
    column-M runner reads (`--dump`), and `<out>/train_log.csv`;
  * RESUME: every `--save-every` steps and at each epoch's end the full
    training state (params, BN stats, Adam's moments and step state, the
    global step and the position in the epoch) goes to `<out>/resume_a` or
    `resume_b` — the one `<out>/latest` does NOT name — and only then is
    `latest` rewritten (atomically). A crash mid-save leaves the previous
    state intact. `--resume` continues from `latest` (mid-epoch: the epoch's
    shuffle is recomputed from the seed and the done batches skipped); with
    no `latest` it starts fresh, so a supervisor can always pass it. What a
    resume does not restore: the SIGReg and dropout random streams (fresh
    draws, same distribution);
  * guards: a non-finite loss stops the run (resume from the last good
    state); every save first checks free disk (`--min-free-gb`, default 3).

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
from std.os.path import exists
from std.math import isfinite
from noeira.io.proc import run_capture
from noeira.io.fileio import write_file_atomic
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


def _free_gb(dir: String) raises -> Float64:
    """Free space on `dir`'s filesystem (`df -Pk`)."""
    var outp = run_capture(String("df -Pk ") + dir)
    var lines = outp.split("\n")
    var cols = List[String]()
    for c in lines[1].split(" "):
        if c.byte_length() > 0:
            cols.append(String(c))
    return Float64(Int(cols[3])) / (1024.0 * 1024.0)


def _check_disk(dir: String, min_gb: Float64) raises:
    var g = _free_gb(dir)
    if g < min_gb:
        raise Error("only " + String(g) + " GB free on " + dir + " (need " + String(min_gb) + "): not saving")


def _read(path: String) raises -> String:
    with open(path, "r") as f:
        return f.read()


def _save_resume[BB: Int](
    mut tr: LeWMRefTrainer["gpu", BB], odir: String, step: Int, ep: Int, s_next: Int,
    min_gb: Float64,
) raises:
    """Into the slot `latest` does not name, then flip `latest`."""
    _check_disk(odir, min_gb)
    var cur = String("")
    if exists(odir + "/latest"):
        cur = _read(odir + "/latest")
    var slot = String("resume_b") if cur == "resume_a" else String("resume_a")
    var n = tr.save_resume(odir + "/" + slot)
    with open(odir + "/" + slot + "/progress.txt", "w") as f:
        f.write(String(step) + "\n" + String(ep) + "\n" + String(s_next) + "\n")
    var b = List[UInt8]()
    for ch in slot.as_bytes():
        b.append(ch)
    write_file_atomic(odir + "/latest", b)
    print("  resume state ->", slot, "(step", step, ", epoch", ep, ", next batch", s_next, ";", n, "moment tensors )")


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
    var resume = False
    var save_every = 2000
    var min_gb = 3.0
    var abort_at = -1        # test: raise at this global step (a "crash")
    var verify_resume = False  # test: re-save right after a resume
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
        elif a == "--resume":
            resume = True
        elif a == "--save-every":
            save_every = Int(String(args[i + 1])); i += 1
        elif a == "--abort-at":
            abort_at = Int(String(args[i + 1])); i += 1
        elif a == "--verify-resume":
            verify_resume = True
        elif a == "--min-free-gb":
            min_gb = Float64(String(args[i + 1])); i += 1
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
    var step = 0
    var ep0 = 0
    var s0 = 0
    var log = String("step,epoch,lr,loss,pred_loss,sigreg_loss,grad_norm,wall_s\n")
    var vlog = String("epoch,val_loss,val_pred_loss,val_sigreg_loss\n")
    if resume and exists(out + "/latest"):
        var slot = out + "/" + _read(out + "/latest")
        var nm = tr.load_resume(slot)
        var pr = _read(slot + "/progress.txt").split("\n")
        step = Int(String(pr[0]))
        ep0 = Int(String(pr[1]))
        s0 = Int(String(pr[2]))
        if exists(out + "/train_log.csv"):
            # keep the lines up to the resumed step: what a crashed run
            # logged past its last save is replayed, not appended twice
            var kept = String("")
            var first_line = True
            for l in _read(out + "/train_log.csv").split("\n"):
                if l.byte_length() == 0:
                    continue
                if first_line:
                    kept += String(l) + "\n"
                    first_line = False
                    continue
                if Int(String(l.split(",")[0])) <= step:
                    kept += String(l) + "\n"
            log = kept
        if exists(out + "/val_log.csv"):
            vlog = _read(out + "/val_log.csv")
        print("  RESUMED from", slot, ": step", step, ", epoch", ep0, ", next batch", s0, ";", nm, "moment tensors")
        if verify_resume:
            # what was loaded, saved again: must be byte-identical to the slot
            _ = tr.save_resume(out + "/resume_verify")
    elif resume:
        print("  --resume: no", out + "/latest", "— starting fresh")
    var stages: List[Pointer[Scalar[DType.uint8], MutAnyOrigin]] = [tr.staging(0), tr.staging(1)]
    print("LeWM training (P6.2):", epochs, "epochs x", steps_per_epoch, "steps x batch", B,
          "=", total, "steps; train", len(train_idx), "clips, val", n_val, "batches")
    print("  init", init, "(", n, "tensors );  action stats", am[0], am[1], asd[0], asd[1])

    var t_all = perf_counter_ns()
    var step_start = step
    for ep in range(ep0, epochs):
        var t_ep = perf_counter_ns()
        var first = s0 if ep == ep0 else 0
        var clips = _shuffled(train_idx, UInt64(run_seed * 1000003 + ep + 1))
        var use = List[Int](capacity=(steps_per_epoch - first) * B)
        for k in range(first * B, steps_per_epoch * B):
            use.append(clips[k])
        var loader = LewmBatchLoaderThread[B](h5, use, am, asd, stages)
        tr.set_train()
        if first < steps_per_epoch:
            loader.request(0)
        var acc = List[Float64](length=4, fill=0.0)
        var n_acc = 0
        for s in range(first, steps_per_epoch):
            var k = s - first  # the loader's batch index
            _ = loader.wait(k)
            if s + 1 < steps_per_epoch:
                loader.request(k + 1)
            var lr = recipe_lr(step, total)
            tr.set_lr(lr)
            tr.submit_staged(loader.actions(k), k % 2)
            var r = tr.finish()
            if step == abort_at:
                raise Error("--abort-at " + String(abort_at) + ": simulated crash")
            if not isfinite(r.loss):
                raise Error("non-finite loss at step " + String(step) + ": stopping (resume from the last state)")
            acc[0] += r.loss
            acc[1] += r.pred_loss
            acc[2] += r.sigreg_loss
            acc[3] += r.grad_norm
            n_acc += 1
            step += 1
            if step % log_every == 0 or s == steps_per_epoch - 1:
                var wall = Float64(perf_counter_ns() - t_all) / 1e9
                var line = String(step) + "," + String(ep) + "," + String(lr)
                for j in range(4):
                    line += "," + String(acc[j] / Float64(n_acc))
                line += "," + String(wall)
                log += line + "\n"
                var eta = wall / Float64(max(1, step - step_start)) * Float64(total - step) / 3600.0
                print("  step", step, "/", total, " epoch", ep, " lr", Float32(lr),
                      " loss", Float32(acc[0] / Float64(n_acc)), " pred", Float32(acc[1] / Float64(n_acc)),
                      " sigreg", Float32(acc[2] / Float64(n_acc)), " norm", Float32(acc[3] / Float64(n_acc)),
                      " ", Float32(wall / Float64(max(1, step - step_start))), "s/step  eta", Float32(eta), "h")
                with open(out + "/train_log.csv", "w") as f:
                    f.write(log)
                for j in range(4):
                    acc[j] = 0.0
                n_acc = 0
            if save_every > 0 and step % save_every == 0 and s + 1 < steps_per_epoch:
                _save_resume(tr, out, step, ep, s + 1, min_gb)
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
        _check_disk(out, min_gb)
        var nd = tr.save_dump(out + "/epoch_" + String(ep))
        _save_resume(tr, out, step, ep + 1, 0, min_gb)
        print("  epoch", ep, "done in", Float32(Float64(perf_counter_ns() - t_ep) / 60e9), "min;",
              " val loss", Float32(v[0]), " pred", Float32(v[1]), " sigreg", Float32(v[2]),
              "; saved", nd, "tensors to", out + "/epoch_" + String(ep))
    print("done:", step, "steps in", Float32(Float64(perf_counter_ns() - t_all) / 3600e9), "h")
