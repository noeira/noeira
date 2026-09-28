# +--------------------------------------------------------------------------+ #
# | ACT on the SO-ARM101 — CPU training
# +--------------------------------------------------------------------------+ #
"""Train ACT on the LeRobot v3 dataset, on CPU.

    pixi run mojo run -I . examples/so101/act_so101_import_dataset.mojo \\
        --repo DenisLabs/record-test_20260825_094319 --height 240 --width 320
    pixi run mojo build -I . -Xlinker -ld_classic -o /tmp/act_train \\
        examples/so101/act_so101_train_cpu.mojo
    /tmp/act_train                       # ImageNet backbone by default

Set `ACT_STORE=<path/to/store.h5>` to train on a different recording — the
50-episode store, say. Nothing below is pinned to the row or episode count.

`ACT_PRETRAINED` picks the backbone. `load_backbone_auto` reads the STRING, not
what is at the path:

| `hub` (or `1`) | fetch `timm/resnet18.tv_in1k` — no PyTorch, no `-e act-ref`, no dump step, cached after the first run |
| `*.safetensors` | that file, under torchvision's names |
| anything else | a `tools/act/dump_resnet18_imagenet.py` DIRECTORY |
| `random` / `none` / `0` | a RANDOM backbone, deliberately |
| unset | **`hub`** — the ImageNet weights, fetched once and cached |

The dump is not deprecated: it is the oracle the Hub file is gated against
(`tests/nn/test_safetensors_resnet18_torch.mojo`). The startup line says which
of the four happened — worth reading, since a random backbone is legal, silent,
and exactly what "4 training episodes will overfit" below is about.

⚠ `-Xlinker -ld_classic` is required: the fully-expanded graph type mangles to a
symbol longer than Apple's new linker accepts. The source is healthy; `mojo run`
JITs past it and never invokes ld.

## ⚠ This is a REDUCED configuration, and deliberately so

The paper's settings (`hidden_dim 512`, `dim_feedforward 3200`, `chunk_size 100`,
480x640 x4 cameras, 2000 epochs) are a ~5-hour run on an RTX 2080 Ti. On CPU the
ResNet18 forward alone dominates, and the full configuration is not a run you
would wait for. The dimensions below are what fits a CPU session while keeping
every structural property intact — the CVAE, both transformer stacks, the
chunking, the masking. `config.mojo` carries the paper's values; M8 of the plan
is the GPU path where they become usable.

## ⚠ And 4 training episodes will overfit

1997 frames, a from-scratch ResNet18, and no augmentation. Validation L1 is
expected to bottom out early and then rise while training L1 keeps falling. That
is the honest outcome at this data scale and it is what the curve below is for —
the deliverable here is a correct, trainable ACT, not a deployable policy. The
reference used 50 demonstrations per task.
"""

from std.time import perf_counter_ns

from noeira.nn.constants import DT
from noeira.deep_agents.act.config import (
    act_pretrained_spec,
    SO101_ADIM,
    SO101_IMG_H,
    SO101_IMG_W,
    SO101_N_CAM,
    SO101_QPOS,
)
from noeira.deep_agents.act.data import ACTDataset
from noeira.deep_agents.act.trainer import ACTTrainer
from noeira.core.dotenv import load_dotenv
from noeira.core.logger import RemoteLogger
from noeira.core.run import RunContext, register_run
from noeira.deep_agents.training.checkpoint import announce_checkpoint
from noeira.io.artifact_sink import ArtifactSink, close_sink, sink_for_run

from std.python import Python, PythonObject


# ── shape (must match the store; see the header) ─────────────────────────
comptime QPOS = SO101_QPOS
comptime ADIM = SO101_ADIM
comptime N_CAM = SO101_N_CAM
comptime IMG_H = SO101_IMG_H
comptime IMG_W = SO101_IMG_W

# ── reduced model ────────────────────────────────────────────────────────
comptime K = 20  # paper: 100
comptime DIM = 64  # paper: 512
comptime HEADS = 4  # paper: 8
comptime FF = 256  # paper: 3200
comptime LATENT = 32  # paper: 32 (unchanged — it is small already)
comptime N_ENC = 1  # paper: 4
comptime N_DEC = 1  # paper: 7, but output-equivalent to 1 (see config.mojo)
comptime BATCH = 4  # paper: 8

comptime DEFAULT_STEPS = 400
"""Override with `ACT_STEPS` — no rebuild. Same knob as the GPU example."""
comptime VAL_EVERY = 50
comptime PATIENCE = 10
"""Stop after this many validations with no improvement on the best. 0 = never.

Sized from the run that motivated it: 50 episodes, best val L1 0.4076 at epoch
15.5, then **26 consecutive validations all worse** (mean 0.4376) while train
L1 fell another 1.7x. That is 33 epochs and ~2.7 GPU-hours after the last
checkpoint that would ever be written. 10 is loose enough to ride out the
noise in a 256-sample validation and tight enough that the waste is bounded at
~10 epochs.

⚠ This bounds WASTE, not quality. It cannot make a run better — the best
checkpoint is already on disk when it fires. If a run stops early and you think
it was still learning, the fix is more data or regularization, not more
patience."""
comptime VAL_SEED: UInt64 = 0x5DEECE66D
"""Validation draws the SAME batches every pass. Left random, `best_val` is the
minimum of a noisy estimate and selects the luckiest draw rather than the best
model — and successive points on the curve are not comparable."""
comptime LOG_EVERY = 10
comptime VAL_BATCHES = 4
comptime LR = 1e-4
comptime KL_WEIGHT = 10.0

comptime T = ACTTrainer[
    QPOS, ADIM, N_CAM, IMG_H, IMG_W, K, DIM, HEADS, FF, LATENT, N_ENC, N_DEC,
    BATCH,
]
comptime IMG_ELEMS = N_CAM * 3 * IMG_H * IMG_W


def store_path() raises -> String:
    """`$ACT_STORE` if set, else the recording the header names.

    The default is a specific recording, not a pattern: an example that
    silently picked up whichever store happened to be newest in the cache
    would report numbers nobody could attribute to a dataset. Point
    `ACT_STORE` at another store to train on it.
    """
    var os = Python.import_module("os")
    var env = String(
        os.environ.get(PythonObject("ACT_STORE"), PythonObject(""))
    )
    if env.byte_length() > 0:
        return env
    var home = String(os.path.expanduser(PythonObject("~")))
    return (
        home
        + "/.cache/noeira/act_so101/"
        + "DenisLabs__record-test_20260825_094319_"
        + String(IMG_H) + "x" + String(IMG_W) + ".h5"
    )


def main() raises:
    var path = store_path()
    var os = Python.import_module("os")
    if not Bool(os.path.exists(PythonObject(path))):
        print("MISSING STORE: " + path)
        print("build it with tools/act/lerobot_v3_to_store.py — see the header")
        raise Error("store not found")

    var steps = Int(DEFAULT_STEPS)
    var env_steps = String(
        os.environ.get(PythonObject("ACT_STEPS"), PythonObject(""))
    )
    if env_steps.byte_length() > 0:
        steps = Int(env_steps)
        if steps < 1:
            raise Error("ACT_STEPS must be >= 1, got " + env_steps)

    print("ACT / SO-ARM101 — CPU training")
    print("  store   " + path)
    print(
        "  model   K=" + String(K) + " dim=" + String(DIM)
        + " heads=" + String(HEADS) + " ff=" + String(FF)
        + " enc=" + String(N_ENC) + " dec=" + String(N_DEC)
        + "  (REDUCED — see the header)"
    )
    print(
        "  data    " + String(N_CAM) + " cameras at " + String(IMG_H) + "x"
        + String(IMG_W) + ", batch " + String(BATCH)
    )

    var ds = ACTDataset[QPOS, ADIM, N_CAM, IMG_H, IMG_W](String(path), seed=7)
    print(
        "  split   " + String(len(ds.train_eps)) + " train / "
        + String(len(ds.val_eps)) + " val episodes of "
        + String(ds.n_episodes())
    )
    print("")

    # Same metric stream as the GPU example — see its header. Inert without
    # `NOEIRA_CLOUD_URL` in `.env`, and forced inert by `ACT_NO_MONITOR=1`.
    var env_vars = load_dotenv()
    var no_monitor = String(
        os.environ.get(PythonObject("ACT_NO_MONITOR"), PythonObject(""))
    )
    var monitor_url = (
        String("") if no_monitor.byte_length() > 0
        else env_vars.get("NOEIRA_CLOUD_URL", "")
    )
    var run = RunContext(
        project=String("so101"),
        driver=String("examples/so101/act_so101_train_cpu.mojo"),
        slug=String("act-so101-cpu"),
        env=String("builtin:so_arm101"),
        dataset=String(path),
        device=String("cpu"),
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
    logger.set_config("robot", "SO-ARM101")
    logger.set_config("target", "cpu")
    logger.set_config("store", path)
    logger.set_config("chunk_k", String(K))
    logger.set_config("hidden_dim", String(DIM))
    logger.set_config("dim_feedforward", String(FF))
    logger.set_config("enc_layers", String(N_ENC))
    logger.set_config("dec_layers", String(N_DEC))
    logger.set_config("batch", String(BATCH))
    logger.set_config("lr", String(LR))
    logger.set_config("kl_weight", String(KL_WEIGHT))
    logger.set_config("steps", String(steps))
    logger.set_config("train_episodes", String(len(ds.train_eps)))
    print(
        "  metrics " + (
            "streaming to " + monitor_url if logger.is_active()
            else (
                "OFF (ACT_NO_MONITOR)" if no_monitor.byte_length() > 0
                else "local only (set NOEIRA_CLOUD_URL in .env)"
            )
        )
    )

    var tr = T.make(lr=Scalar[DT](LR), kl_weight=Scalar[DT](KL_WEIGHT))

    # ── ImageNet-pretrained backbone ─────────────────────────────────────
    # The paper's backbone is `resnet18(weights=IMAGENET1K_V1)`; ours is random
    # at step 0 and has to learn vision from 12,411 frames, which is what the
    # first 50-episode run overfit doing. Absent the variable the run is
    # unchanged, and it says which happened rather than leaving it to be
    # inferred from the loss curve.
    #
    #   ACT_PRETRAINED=hub    fetch timm/resnet18.tv_in1k -- NO PYTHON, no
    #                         `-e act-ref`, no dump step. Cached after the
    #                         first run.
    #   ACT_PRETRAINED=<dir>  a tools/act/dump_resnet18_imagenet.py dump, the
    #                         original path and the oracle the Hub file is
    #                         gated against.
    # ⚠ Resolved by `act_pretrained_spec`, not read here: the DEFAULT is
    # `hub`, and a default spelled out at two call sites is how the CPU and
    # GPU runs come to train different models.
    var pretrained = act_pretrained_spec()
    if pretrained.byte_length() > 0:
        # `ACT_NO_FREEZE_BN=1` loads the weights and leaves BatchNorm trainable
        # — the ablation, not a normal setting. Both references pass
        # `norm_layer=FrozenBatchNorm2d` at the same call that loads the
        # weights, and unfrozen BatchNorm EMAs the ImageNet statistics away in
        # a few hundred steps while never reading them.
        var no_freeze = String(
            os.environ.get(PythonObject("ACT_NO_FREEZE_BN"), PythonObject(""))
        )
        var freeze = no_freeze.byte_length() == 0
        var n_loaded = tr.load_backbone_auto(pretrained, freeze_norm=freeze)
        print(
            "  backbone  ImageNet weights, " + String(n_loaded)
            + " tensors, BatchNorm "
            + ("FROZEN" if freeze else "TRAINABLE (ACT_NO_FREEZE_BN)")
        )
        logger.set_config("backbone_init", "imagenet")
        logger.set_config(
            "backbone_norm", "frozen" if freeze else "trainable"
        )
    else:
        print(
            "  backbone  RANDOM (ACT_PRETRAINED=random was set explicitly)"
        )
        logger.set_config("backbone_init", "random")
        logger.set_config("backbone_norm", "trainable")

    # Registered after the last `set_config`: the registration carries the
    # config and `project=` to the dashboard.
    register_run(run, logger)
    # `best` and `last` are uploaded as they are written. None without a
    # monitor, and `ACT_NO_MONITOR` must silence the uploads too.
    var artifacts: Optional[ArtifactSink] = None
    if monitor_url.byte_length() > 0:
        artifacts = sink_for_run(run.id, run.dir)

    var qpos = List[Scalar[DT]](unsafe_uninit_length=BATCH * QPOS)
    var images = List[Scalar[DT]](unsafe_uninit_length=BATCH * IMG_ELEMS)
    var actions = List[Scalar[DT]](unsafe_uninit_length=BATCH * K * ADIM)
    var valid = List[Scalar[DT]](unsafe_uninit_length=BATCH * K)

    var best_val = Float64(1e30)
    var best_step = -1
    var stale = 0
    """Validations since the best. See PATIENCE."""
    var best_ckpt = run.checkpoint_path(String("best"))
    var last_ckpt = run.checkpoint_path(String("last"))

    var train_frames = 0
    for i in range(len(ds.train_eps)):
        train_frames += ds.store.episodes.length_of(ds.train_eps[i])
    var steps_per_epoch = train_frames // BATCH
    if steps_per_epoch < 1:
        steps_per_epoch = 1
    print(
        "  run     " + String(steps) + " steps, " + String(steps_per_epoch)
        + " per epoch (" + String(train_frames) + " train frames)"
    )
    print("")

    # Train-step time only — see the GPU example. A wall-clock interval folds
    # in the validation pass, and at step 0 it folds in EVERYTHING.
    var train_ns = 0
    var data_ns = 0
    var train_steps = 0

    var acc_l1 = Float64(0.0)
    var acc_kl = Float64(0.0)
    var acc_n = 0

    var names = List[String]()
    names.append(String("l1_loss"))
    names.append(String("kl_loss"))
    names.append(String("epoch"))

    var val_names = List[String]()
    val_names.append(String("val_l1_loss"))
    val_names.append(String("val_kl_loss"))
    val_names.append(String("step_s"))
    val_names.append(String("val_l1_loss_best"))

    for s in range(steps):
        # Split, because "0.176 s/step" does not say WHICH half. `sample_batch`
        # is host-side, single-threaded and serial with the device: per sample
        # it converts N_CAM*3*H*W uint8 to float32 one element at a time
        # (divide, subtract, multiply) and reads a row from HDF5. At batch 16
        # that is 7.4M elements before the GPU sees anything, and if it is the
        # larger half then no kernel work will fix the step time.
        var step_t0 = perf_counter_ns()
        ds.sample_batch[K, BATCH](False, qpos, images, actions, valid)
        var t_data = perf_counter_ns()
        var r = tr.train_step(qpos, images, actions, valid)
        var t_end = perf_counter_ns()
        data_ns += t_data - step_t0
        train_ns += t_end - step_t0
        train_steps += 1

        acc_l1 += r.l1
        acc_kl += r.kl
        acc_n += 1
        if acc_n == LOG_EVERY:
            var vals = List[Float64]()
            vals.append(acc_l1 / Float64(acc_n))
            vals.append(acc_kl / Float64(acc_n))
            vals.append(Float64(s) / Float64(steps_per_epoch))
            logger.log_scalars(names, vals, s)
            acc_l1 = 0.0
            acc_kl = 0.0
            acc_n = 0

        if s % VAL_EVERY == 0 or s == steps - 1:
            # Validation L1 is the reference's model-selection metric
            # (`imitate_episodes.py` keeps the checkpoint with the lowest
            # validation loss), so that is what the best checkpoint tracks.
            # ⚠ The stream is PINNED and restored, so every pass scores the
            # SAME batches — see VAL_SEED.
            var saved_rng = ds.rng
            ds.rng = VAL_SEED
            var vl1 = Float64(0.0)
            var vkl = Float64(0.0)
            for _ in range(VAL_BATCHES):
                ds.sample_batch[K, BATCH](True, qpos, images, actions, valid)
                var v = tr.eval_step(qpos, images, actions, valid)
                vl1 += v.l1
                vkl += v.kl
            vl1 /= Float64(VAL_BATCHES)
            vkl /= Float64(VAL_BATCHES)
            ds.rng = saved_rng

            var sps = (
                Float64(train_ns) / Float64(train_steps) / 1e9
                if train_steps > 0 else 0.0
            )
            var sps_data = (
                Float64(data_ns) / Float64(train_steps) / 1e9
                if train_steps > 0 else 0.0
            )
            train_ns = 0
            data_ns = 0
            train_steps = 0
            var eta = sps * Float64(steps - s) / 60.0

            print(
                "  step " + String(s)
                + " (epoch " + String(s // steps_per_epoch) + ")"
                + "  train l1 " + String(r.l1)
                + "  kl " + String(r.kl)
                + "  |  val l1 " + String(vl1)
                + "  |  " + String(sps) + " s/step ("
                + String(Int(100.0 * sps_data / (sps + 1e-12)))
                + "% data), ~" + String(Int(eta)) + " min left"
            )
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

            if PATIENCE > 0 and stale >= PATIENCE:
                print(
                    "  early stop: " + String(stale) + " validations with no"
                    " improvement on " + String(best_val) + " (step "
                    + String(best_step) + "). The best checkpoint is written;"
                )
                print(
                    "    more steps will not produce a better one. See"
                    " PATIENCE."
                )
                break

    # The verdict goes to `run.kv` AND the dashboard: `finish` before `close`,
    # and the sink drained after the loop so the last `best` is uploaded.
    var outcome = (
        String("best_val_l1=") + String(best_val)
        + " best_step=" + String(best_step)
    )
    run.set_outcome(outcome)
    logger.finish(String("done"), outcome)
    logger.close()
    close_sink(artifacts)
    run.close()

    print("")
    print(
        "  best validation l1 " + String(best_val) + " at step "
        + String(best_step) + " (epoch "
        + String(best_step // steps_per_epoch) + ")"
    )
    print("  best -> " + best_ckpt)
    print("  last -> " + last_ckpt)
    print("  run  -> " + run.kv_path())
    print("")
    print(
        "  ⚠ validation L1 rising while training L1 falls is what a"
        " from-scratch"
    )
    print(
        "    backbone does on " + String(len(ds.train_eps))
        + " training episodes with no augmentation."
    )
    print(
        "    The best checkpoint above is the one to evaluate; see"
        " act_so101_openloop_eval.mojo."
    )
