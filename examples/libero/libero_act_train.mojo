"""ACT ON LIBERO'S DEMONSTRATIONS — the first image policy of the port. L7b.

    ACT_STORE=build/demos/libero_goal.rendered.h5 pixi run -e nvidia libero-act-train
    ACT_STORE=build/demos/libero_goal.rendered.smoke.h5 ACT_STEPS=20 ACT_NO_MONITOR=1 \\
        pixi run -e apple libero-act-train                          # a Mac smoke

`envs/libero/act.mojo` is the shape (9 proprio words, 7 OSC_POSE words, two
128x128 cameras, K = 2 s = 40 steps at 20 Hz, the `RUN_*` transformer); this
file is the SO-101 GPU driver's loop on the LIBERO store — `ACTDataset` over
`qpos`/`action`/`images`, the ImageNet ResNet18, validation on held-out
episodes, `best`/`last` checkpoints with `norm.json` beside them under the
run's own directory.

## ⚠⚠ WHICH STORE, AND WHY THE DEFAULT IS THE RENDERED ONE

`build/demos/libero_goal.rendered.h5` (`libero-demo-rerender`) holds OUR
tracer's frame of `state[r]` beside `action[r]`: what the closed-loop eval
renders, paired the way the eval pairs it. `build/demos/libero_goal.h5`
(`libero-demo-import`) holds robosuite's OpenGL frame of `state[r + 1]`
beside `action[r]` — LIBERO's own pairing in LIBERO's own pixels. A checkpoint
from the recorded store is the arm that prices the pixel-domain gap in success
points; it is chosen with `ACT_STORE`, and `norm.json` records which store
fitted the checkpoint so the eval can print it.

## ⚠⚠ THE BASELINES ARE PRINTED BESIDE THE VALIDATION L1

ACT's model-selection metric is the L1 on NORMALISED actions, so two numbers
that need no network are computed over every held-out row in the same units:
the L1 of predicting ZERO (LIBERO's null action — what the batched eval scores
0/200 with) and of predicting the TRAINING MEAN. A checkpoint that does not
beat both has learned nothing the eval could use, and the run says so instead
of reporting a loss that looks like progress. (They ignore the chunk's padding
slots, which the real loss counts in its denominator — a few percent on a
600-step episode with K = 40.)

## The split, and what a validation number is here

`ACTDataset` shuffles EPISODES and holds out 20% — over 500 demonstrations of
10 tasks that is ~100 episodes, roughly ten per task. Its role is model
selection: the success rate comes from `libero_eval_batched.mojo --act` on
LIBERO's frozen inits, which are not demonstrations at all.

## Environment variables

| | |
|---|---|
| `ACT_STORE` | the `.h5` to train on; default `envs/libero/act.LIBERO_ACT_STORE_RENDERED` |
| `ACT_STEPS` | optimizer steps without a rebuild (default 50 000) |
| `ACT_LR` | the learning rate (default `LIBERO_ACT_LR` = 1e-5, the paper's). An overfit at 1e-3 on a few episodes is the test of whether the graph can REPRESENT a per-position chunk at all (2026-09-20: the box fits' chunks were flat across the 40 positions after training even with unit-scale queries) |
| `ACT_SHAPE` | weight of the chunk-shape term (L1 on first differences along the chunk; default `ACT_SHAPE_WEIGHT` = 0, the paper's loss). See `deep_agents/act/config.mojo` |
| `ACT_SEED` | the run's seed (default 7): the episode split, the batch order and the weight init all follow it. Two fits at different seeds bound the run-to-run spread of the closed-loop rate; the default reproduces every fit before 2026-09-22 |
| `ACT_PATIENCE` | validations without improvement before the early stop (default 10); `0` disables it. ⚠ On LIBERO the validation L1 bottoms at the DEMONSTRATOR NOISE FLOOR (task+phase oracle 0.386, the fits 0.41-0.42) 17-28k steps in, and its minimum there is noise: the fit's training L1 is still falling (0.25) and its closed-loop rate still moving. ACT's recipe trains thousands of epochs and the per-task drawer fit at the val minimum scored 1/20 against the multi-task 6/20; `ACT_PATIENCE=0 ACT_STEPS=300000` with `--act-ckpt last` at eval is the recipe's schedule |
| `ACT_KL` | the KL weight (default `LIBERO_ACT_KL` = 10, the paper's). ⚠ Both 5090 fits collapsed the CVAE at 10 — `train/kl` 59 -> 0.05 — so the latent carried nothing and every chunk was the conditional median: half the demonstrations' action scale, 5/200. The paper's ablation says the CVAE is what absorbs demonstrator variability; a lower weight is the lever, and it is here so a sweep needs no rebuild |
| `ACT_PRETRAINED` | defaults to `hub` (ImageNet ResNet18, no PyTorch); `random` opts out |
| `ACT_NO_FREEZE_BN` | leave BatchNorm trainable — the ablation |
| `ACT_PROJECT` | the project the run is filed under (default `libero`; a missing project directory files it under the flat `runs/` root) |
| `ACT_NO_MONITOR` | keep a smoke run off the dashboard |

⚠ Run it from the project root (`io/hdf5` resolves libhdf5 relative to it).
On Apple, `mojo build` needs `-Xlinker -ld_classic`; `mojo run` does not.
"""

from std.os import getenv, makedirs
from std.random import seed as seed_rng
from std.os.path import exists
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.core.dotenv import load_dotenv
from noeira.core.logger import RemoteLogger
from noeira.core.run import RunContext, register_run
from noeira.deep_agents.act.config import act_pretrained_spec, ACT_SHAPE_WEIGHT
from noeira.deep_agents.act.norm_file import act_norm_from
from noeira.deep_agents.act.trainer import ACTWindowMetrics
from noeira.deep_agents.training.checkpoint import announce_checkpoint
from noeira.io.artifact_sink import ArtifactSink, close_sink, sink_for_run
from noeira.envs.libero.act import (
    LiberoActDataset, LiberoActDeviceDataset, LiberoActTrainer,
    LIBERO_ACT_QPOS, LIBERO_ACT_ADIM, LIBERO_ACT_N_CAM, LIBERO_ACT_IMG_H,
    LIBERO_ACT_IMG_W, LIBERO_ACT_IMG_ELEMS, LIBERO_ACT_K, LIBERO_ACT_DIM,
    LIBERO_ACT_HEADS, LIBERO_ACT_FF, LIBERO_ACT_LATENT, LIBERO_ACT_N_ENC,
    LIBERO_ACT_N_DEC, LIBERO_ACT_LR, LIBERO_ACT_KL, LIBERO_ACT_STORE_RENDERED,
    LIBERO_ACT_OH, LIBERO_ACT_OW, LIBERO_ACT_FEAT_CH,
)


comptime QPOS = LIBERO_ACT_QPOS
comptime ADIM = LIBERO_ACT_ADIM
comptime K = LIBERO_ACT_K
comptime IMG_ELEMS = LIBERO_ACT_IMG_ELEMS
comptime BATCH = 16
"""The SO-101 run's batch (paper: 8): 500 demonstrations x 2 cameras is
plenty of frames per step for a 5090."""
comptime DEFAULT_STEPS = 50000
"""~16 epochs over ~400 training episodes (~51 000 frames) at batch 16."""
comptime GPU_DATA = True
"""Draw and normalise batches on the device (`ACTDeviceDataset`): the whole
image column is uploaded once as uint8 — 63 728 rows x 98 304 bytes = 6.3 GB
for the full suite. Set False on a device where that does not fit; the host
sampler streams rows from HDF5 instead."""
comptime VAL_EVERY = 1000
comptime VAL_BATCHES = 64
comptime PATIENCE = 10
comptime VAL_SEED: UInt64 = 0x5DEECE66D
comptime LOG_EVERY = 50

comptime T = LiberoActTrainer[BATCH, "gpu"]
comptime DDS = LiberoActDeviceDataset


def store_path() -> String:
    var env = getenv("ACT_STORE")
    if env.byte_length() > 0:
        return env^
    return String(LIBERO_ACT_STORE_RENDERED)


def _baselines(
    ref ds: LiberoActDataset,
) raises -> Tuple[Float64, Float64, Int]:
    """(L1 of the zero action, L1 of the training-mean action, rows) over every
    held-out row, in the trainer's normalised units — `|(a - mean)/std|`
    averaged over the action's words. The training mean normalises to exactly
    0, so its L1 is the mean |z| of the held-out rows; the zero action
    normalises to `z0 = (0 - mean)/std`, so its L1 is the mean |z - z0| =
    |a|/std.

    ⚠ IT WAS `|z0|` — the distance from the zero action to the MEAN action,
    0.20 on libero_goal — and the first box run printed a 0.42 fit as losing
    to it. The error of a constant prediction is measured against the
    targets, not against another constant."""
    var l1_zero = 0.0
    var l1_mean = 0.0
    var rows = 0
    for i in range(len(ds.val_eps)):
        var e = ds.val_eps[i]
        var off = ds.store.episodes.start_of(e)
        var ln = ds.store.episodes.length_of(e)
        for r in range(off, off + ln):
            for k in range(ADIM):
                var z = (Float64(ds.action_raw[r * ADIM + k])
                         - Float64(ds.action_mean[k])) / Float64(ds.action_std[k])
                var z0 = (0.0 - Float64(ds.action_mean[k])) / Float64(ds.action_std[k])
                l1_mean += abs(z)
                l1_zero += abs(z - z0)
            rows += 1
    var n = Float64(rows * ADIM) if rows > 0 else 1.0
    return (l1_zero / n, l1_mean / n, rows)


def main() raises:
    var path = store_path()
    if not exists(path):
        print("MISSING STORE: " + path)
        print("  build it: pixi run libero-demo-rerender   (or libero-demo-import"
              " for the recorded frames)")
        raise Error("store not found")
    var steps = Int(DEFAULT_STEPS)
    var env_steps = getenv("ACT_STEPS")
    if env_steps.byte_length() > 0:
        steps = Int(env_steps)
        if steps < 1:
            raise Error("ACT_STEPS must be >= 1, got " + env_steps)
    var lr = Float64(LIBERO_ACT_LR)
    var env_lr = getenv("ACT_LR")
    if env_lr.byte_length() > 0:
        lr = Float64(env_lr)
        if lr <= 0.0:
            raise Error("ACT_LR must be > 0, got " + env_lr)
    var shape_weight = Float64(ACT_SHAPE_WEIGHT)
    var env_shape = getenv("ACT_SHAPE")
    if env_shape.byte_length() > 0:
        shape_weight = Float64(env_shape)
        if shape_weight < 0.0:
            raise Error("ACT_SHAPE must be >= 0, got " + env_shape)
    var patience = Int(PATIENCE)
    var env_pat = getenv("ACT_PATIENCE")
    if env_pat.byte_length() > 0:
        patience = Int(env_pat)
        if patience < 0:
            raise Error("ACT_PATIENCE must be >= 0 (0 = no early stop), got " + env_pat)
    var kl_weight = Float64(LIBERO_ACT_KL)
    var env_kl = getenv("ACT_KL")
    if env_kl.byte_length() > 0:
        kl_weight = Float64(env_kl)
        if kl_weight < 0.0:
            raise Error("ACT_KL must be >= 0, got " + env_kl)
    var project = getenv("ACT_PROJECT")
    if project.byte_length() == 0:
        project = String("libero")

    var ctx = DeviceContext()
    print("=" * 78)
    print("ACT on LIBERO — training")
    print("=" * 78)
    print("  device  " + String(ctx.name()))
    print("  store   " + path)
    print("  model   K=" + String(K) + " dim=" + String(LIBERO_ACT_DIM)
          + " heads=" + String(LIBERO_ACT_HEADS) + " ff=" + String(LIBERO_ACT_FF)
          + " latent=" + String(LIBERO_ACT_LATENT) + " enc=" + String(LIBERO_ACT_N_ENC)
          + " dec=" + String(LIBERO_ACT_N_DEC))
    print("  kl      " + String(kl_weight) + ("" if env_kl.byte_length() == 0
          else " (ACT_KL; the declaration's is " + String(LIBERO_ACT_KL) + ")"))
    print("  shape   " + String(shape_weight) + (" (the paper: 0)" if env_shape.byte_length() == 0 else " (ACT_SHAPE; the chunk-shape term, first differences along the chunk)"))
    print("  vision  ResNet18 cut after layer3: " + String(LIBERO_ACT_OH) + "x"
          + String(LIBERO_ACT_OW) + " tokens x " + String(LIBERO_ACT_FEAT_CH)
          + " ch per camera")
    print("  data    " + String(LIBERO_ACT_N_CAM) + " cameras at "
          + String(LIBERO_ACT_IMG_H) + "x" + String(LIBERO_ACT_IMG_W)
          + ", qpos " + String(QPOS) + ", action " + String(ADIM)
          + ", batch " + String(BATCH))

    var run_seed = 7
    var env_seed = getenv("ACT_SEED")
    if env_seed.byte_length() > 0:
        run_seed = Int(env_seed)
    seed_rng(run_seed)
    print("  seed    " + String(run_seed) + ("" if env_seed.byte_length() == 0 else " (ACT_SEED)"))
    var ds = LiberoActDataset(String(path), seed=UInt64(run_seed))
    print("  split   " + String(len(ds.train_eps)) + " train / "
          + String(len(ds.val_eps)) + " val episodes of "
          + String(ds.n_episodes()) + " (" + String(ds.n_rows()) + " rows)")
    var base = _baselines(ds)
    print("  held-out L1 baselines (normalised): zero action "
          + String(base[0]) + " | training mean " + String(base[1]) + " over "
          + String(base[2]) + " rows")
    if base[2] == 0:
        raise Error("no held-out rows — the split left nothing to validate on")

    # ── metrics, run, artifacts ──────────────────────────────────────────
    var env_vars = load_dotenv()
    var no_monitor = getenv("ACT_NO_MONITOR")
    var monitor_url = (
        String("") if no_monitor.byte_length() > 0
        else env_vars.get("NOEIRA_CLOUD_URL", "")
    )
    var run = RunContext(
        project=project,
        driver=String("examples/libero/libero_act_train.mojo"),
        slug=String("act-libero-goal"),
        env=String("builtin:libero_goal"),
        dataset=path,
        device=String(ctx.name()),
    )
    print("  run     " + run.dir)
    var logger = RemoteLogger(
        server_url=monitor_url,
        run_name=run.name(),
        run_id=run.id,
        buffer_size=64,
        api_key=env_vars.get("NOEIRA_CLOUD_API_KEY", ""),
    )
    logger.set_config("algorithm", "ACT")
    logger.set_config("suite", "libero_goal")
    logger.set_config("store", path)
    logger.set_config("chunk_k", String(K))
    logger.set_config("hidden_dim", String(LIBERO_ACT_DIM))
    logger.set_config("dim_feedforward", String(LIBERO_ACT_FF))
    logger.set_config("batch", String(BATCH))
    logger.set_config("lr", String(lr))
    logger.set_config("kl_weight", String(kl_weight))
    logger.set_config("shape_weight", String(shape_weight))
    logger.set_config("steps", String(steps))
    logger.set_config("train_episodes", String(len(ds.train_eps)))
    logger.set_config("val_episodes", String(len(ds.val_eps)))
    logger.set_config("baseline_l1_zero", String(base[0]))
    logger.set_config("baseline_l1_mean", String(base[1]))
    print("  metrics " + (
        "streaming to " + monitor_url if logger.is_active()
        else ("OFF (ACT_NO_MONITOR)" if no_monitor.byte_length() > 0
              else "local only (set NOEIRA_CLOUD_URL in .env)")))

    var tr = T.make(
        shape_weight=Scalar[DT](shape_weight),
        lr=Scalar[DT](lr),
        kl_weight=Scalar[DT](kl_weight),
        max_grad_norm=Scalar[DT](0.0),
        ctx=ctx,
    )
    var dev_ds = DDS()
    comptime if GPU_DATA:
        var u0 = perf_counter_ns()
        dev_ds = DDS.upload_from[BATCH](ds, ctx, seed=UInt64(run_seed))
        print("  device dataset  " + String(Float64(perf_counter_ns() - u0) / 1e9)
              + " s to upload " + String(Float64(dev_ds.n_rows)
                                         * Float64(IMG_ELEMS) / 1e9)
              + " GB uint8 (once)")

    var pretrained = act_pretrained_spec()
    if pretrained.byte_length() > 0:
        var freeze = getenv("ACT_NO_FREEZE_BN").byte_length() == 0
        var n_loaded = tr.load_backbone_auto(pretrained, freeze_norm=freeze)
        print("  backbone  ImageNet weights, " + String(n_loaded)
              + " tensors, BatchNorm " + ("FROZEN" if freeze else "TRAINABLE"))
        logger.set_config("backbone_init", "imagenet")
    else:
        print("  backbone  RANDOM (ACT_PRETRAINED=random)")
        logger.set_config("backbone_init", "random")
    register_run(run, logger)
    var artifacts: Optional[ArtifactSink] = None
    if monitor_url.byte_length() > 0:
        artifacts = sink_for_run(run.id, run.dir)

    # norm.json beside the checkpoints, BEFORE the first one
    var best_ckpt = run.checkpoint_path(String("best"))
    var last_ckpt = run.checkpoint_path(String("last"))
    var ckpt_dir = String(best_ckpt[byte = 0 : best_ckpt.rfind("/")])
    makedirs(ckpt_dir, exist_ok=True)
    var norm_path = ckpt_dir + "/norm.json"
    var cams = List[String]()
    cams.append(String("agentview"))
    cams.append(String("eye_in_hand"))
    var norm = act_norm_from(
        ds.qpos_raw, ds.action_raw, ds.n_rows(), ds.n_episodes(),
        ds.qpos_mean, ds.qpos_std, ds.action_mean, ds.action_std,
        cams, LIBERO_ACT_IMG_H, LIBERO_ACT_IMG_W, path,
    )
    norm.save(norm_path)
    announce_checkpoint(norm_path, artifacts, run.dir)
    print("  norm    " + norm_path)

    var train_frames = 0
    for i in range(len(ds.train_eps)):
        train_frames += ds.store.episodes.length_of(ds.train_eps[i])
    var steps_per_epoch = max(1, train_frames // BATCH)
    print("  run     " + String(steps) + " steps, " + String(steps_per_epoch)
          + " per epoch (" + String(train_frames) + " train frames)")
    print("")

    # host-path buffers (unused under GPU_DATA)
    var qpos = List[Scalar[DT]](unsafe_uninit_length=BATCH * QPOS)
    var images = List[Scalar[DT]](unsafe_uninit_length=BATCH * IMG_ELEMS)
    var actions = List[Scalar[DT]](unsafe_uninit_length=BATCH * K * ADIM)
    var valid = List[Scalar[DT]](unsafe_uninit_length=BATCH * K)

    var names = List[String]()
    names.append(String("l1_loss"))
    names.append(String("kl_loss"))
    names.append(String("loss"))
    names.append(String("grad_norm"))
    names.append(String("epoch"))
    var val_names = List[String]()
    val_names.append(String("val_l1_loss"))
    val_names.append(String("val_kl_loss"))
    val_names.append(String("step_s"))
    val_names.append(String("val_l1_loss_best"))

    var best_val = Float64(1e30)
    var best_step = -1
    var stale = 0
    var t_run0 = perf_counter_ns()
    var train_ns = 0
    var train_steps = 0
    var last_l1 = 0.0
    var acc_l1 = 0.0
    var acc_kl = 0.0
    var acc_loss = 0.0
    var acc_gn = 0.0
    var acc_n = 0
    var probes = List[Int]()
    probes.append(1)
    probes.append(5)
    probes.append(20)
    probes.append(100)
    probes.append(300)

    for s in range(steps):
        var t0 = perf_counter_ns()
        comptime if GPU_DATA:
            tr.train_step_device_accum(dev_ds)
        else:
            ds.sample_batch[K, BATCH](False, qpos, images, actions, valid)
            var r = tr.train_step(qpos, images, actions, valid)
            last_l1 = r.l1
            acc_l1 += r.l1
            acc_kl += r.kl
            acc_loss += r.loss
            acc_gn += r.grad_norm
            acc_n += 1
        train_ns += perf_counter_ns() - t0
        train_steps += 1

        var is_probe = False
        for i in range(len(probes)):
            if s == probes[i]:
                is_probe = True
        if is_probe:
            var rate = Float64(train_ns) / Float64(train_steps) / 1e9
            var pl1 = last_l1
            comptime if GPU_DATA:
                var pw = tr.train_metrics(False)
                if pw.n > 0:
                    pl1 = pw.l1
            print("  step " + String(s) + "  train l1 " + String(pl1) + "  |  "
                  + String(rate) + " s/step, ~"
                  + String(Int(rate * Float64(steps) / 60.0)) + " min for "
                  + String(steps) + " steps")
            train_ns = 0
            train_steps = 0

        var window_full = False
        comptime if GPU_DATA:
            window_full = (s + 1) % LOG_EVERY == 0
        else:
            window_full = acc_n == LOG_EVERY
        if window_full:
            var vals = List[Float64]()
            comptime if GPU_DATA:
                var w = tr.train_metrics()
                last_l1 = w.l1
                vals.append(w.l1)
                vals.append(w.kl)
                vals.append(w.loss)
                vals.append(w.grad_norm)
            else:
                vals.append(acc_l1 / Float64(acc_n))
                vals.append(acc_kl / Float64(acc_n))
                vals.append(acc_loss / Float64(acc_n))
                vals.append(acc_gn / Float64(acc_n))
                acc_l1 = 0.0
                acc_kl = 0.0
                acc_loss = 0.0
                acc_gn = 0.0
                acc_n = 0
            vals.append(Float64(s) / Float64(steps_per_epoch))
            logger.log_scalars(names, vals, s)

        if s % VAL_EVERY == 0 or s == steps - 1:
            # pinned sampler stream: every validation scores the same batches
            var saved_rng = ds.rng
            ds.rng = VAL_SEED
            var saved_off = dev_ds.offset_host
            comptime if GPU_DATA:
                dev_ds.set_offset(ctx, VAL_SEED)
            var vl1 = 0.0
            var vkl = 0.0
            comptime if GPU_DATA:
                for _ in range(VAL_BATCHES):
                    tr.eval_step_device_accum(dev_ds, True)
                var w = tr.val_metrics()
                vl1 = w.l1
                vkl = w.kl
            else:
                for _ in range(VAL_BATCHES):
                    ds.sample_batch[K, BATCH](True, qpos, images, actions, valid)
                    var v = tr.eval_step(qpos, images, actions, valid)
                    vl1 += v.l1
                    vkl += v.kl
                vl1 /= Float64(VAL_BATCHES)
                vkl /= Float64(VAL_BATCHES)
            ds.rng = saved_rng
            comptime if GPU_DATA:
                dev_ds.set_offset(ctx, saved_off)

            var train_line = ACTWindowMetrics(0.0, 0.0, 0.0, 0.0, 0)
            comptime if GPU_DATA:
                train_line = tr.train_metrics(False)
                if train_line.n == 0:
                    train_line.l1 = last_l1
            else:
                train_line.l1 = last_l1
            var sps = (Float64(train_ns) / Float64(train_steps) / 1e9
                       if train_steps > 0 else 0.0)
            train_ns = 0
            train_steps = 0
            print("  step " + String(s) + " (epoch " + String(s // steps_per_epoch)
                  + ")  train l1 " + String(train_line.l1) + "  kl "
                  + String(train_line.kl) + "  |  val l1 " + String(vl1)
                  + " (zero " + String(base[0]) + ", mean " + String(base[1])
                  + ")  |  " + String(sps) + " s/step, ~"
                  + String(Int(sps * Float64(steps - s) / 60.0)) + " min left")
            tr.save(last_ckpt)
            announce_checkpoint(last_ckpt, artifacts, run.dir)
            if vl1 < best_val:
                best_val = vl1
                best_step = s
                stale = 0
                tr.save(best_ckpt)
                announce_checkpoint(best_ckpt, artifacts, run.dir)
            else:
                stale += 1
            var vvals = List[Float64]()
            vvals.append(vl1)
            vvals.append(vkl)
            vvals.append(sps)
            vvals.append(best_val)
            logger.log_scalars(val_names, vvals, s)
            logger.flush()
            if patience > 0 and stale >= patience:
                print("  early stop: " + String(stale) + " validations with no"
                      " improvement on " + String(best_val) + " (step "
                      + String(best_step) + ")")
                break

    var beats = best_val < base[0] and best_val < base[1]
    var outcome = (String("best_val_l1=") + String(best_val) + " best_step="
                   + String(best_step) + " baseline_zero=" + String(base[0])
                   + " baseline_mean=" + String(base[1])
                   + (" BEATS_BASELINES" if beats else " DOES_NOT_BEAT_BASELINES"))
    run.set_outcome(outcome)
    logger.finish(String("done"), outcome)
    logger.close()
    close_sink(artifacts)
    run.close()

    print("")
    print("  wall clock " + String(Float64(perf_counter_ns() - t_run0) / 6e10)
          + " min for " + String(steps) + " steps")
    print("  best validation l1 " + String(best_val) + " at step "
          + String(best_step) + "  |  zero action " + String(base[0])
          + "  |  training mean " + String(base[1]))
    print("  best -> " + best_ckpt)
    print("  last -> " + last_ckpt)
    print("  norm -> " + norm_path)
    print("  run  -> " + run.kv_path())
    if not beats:
        print("  ⚠ the best checkpoint does NOT beat both baselines — it has"
              " learned nothing the eval could use.")
    print("  next: pixi run -e nvidia libero-eval-batched -- --act " + ckpt_dir)
