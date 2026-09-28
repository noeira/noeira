# +--------------------------------------------------------------------------+ #
# | ACT on the SO-ARM101 — GPU training
# +--------------------------------------------------------------------------+ #
"""Train ACT on the LeRobot v3 dataset, on GPU.

## Launching a run on a fresh NVIDIA box, start to finish

Nothing below assumes a previous checkout. Every path is created by a step
above it; `$ACT_STORE` and `$ACT_PRETRAINED` name files that do not exist until
steps 3 and 4 write them.

```bash
# ── 0. the repo and its environments ──────────────────────────────────────
git clone <this repo> noeira && cd noeira
pixi install -e nvidia      # CUDA toolkit, cuDNN, nsight — tens of GB
pixi install -e act-ref     # PyTorch, ~3 GB. ONLY for step 4b — skip it
                            # entirely if you take 4a, which is the default.

# ── 1. disk ───────────────────────────────────────────────────────────────
#   pixi envs   ~20 GB (nvidia) + ~3 GB (act-ref, only on the 4b route)
#   HF cache     0.7 GB   dataset snapshot
#   store        7.2 GB   the converted .h5
#   resnet18      47 MB   ImageNet weights, either route
#   checkpoints  0.3 GB   best + last, ~130 MB each
# Budget 35 GB and do not cut it fine: a pixi fetch that runs out of space
# fails mid-solve and leaves the environment unusable.
df -h .

# ── 1b. a PROJECT's dataset from the platform (instead of 2-3 below) ──────
# Needs `.env` (step 6) first. The project definition, then its recording:
#   pixi run -e nvidia project-pull so101-tower
#   pixi run -e nvidia dataset-pull -- --project so101-tower --dataset cube-in-bowl
#   pixi run -e nvidia mojo run -I . examples/so101/act_so101_import_dataset.mojo \\
#       --project so101-tower --dataset cube-in-bowl
#   export ACT_PROJECT=so101-tower
#   export ACT_STORE=~/.cache/noeira/act_so101/so101-tower__cube-in-bowl_240x320.h5
# Then skip to step 4.

# ── 2. HuggingFace auth — ONLY if the dataset repo is private ─────────────
pixi run -e nvidia hf auth login          # or: export HF_TOKEN=hf_...

# ── 3. the dataset -> a TrajectoryStore (downloads + converts) ────────────
# Pure Mojo: needs `curl` and `ffmpeg` on PATH and nothing else. It downloads
# the repo if it is not already in ~/.cache/noeira or the huggingface_hub
# cache, then decodes and resizes every frame.
pixi run -e nvidia mojo run -I . \\
    examples/so101/act_so101_import_dataset.mojo \\
    --repo DenisLabs/record-test_20260828_092736 --height 240 --width 320
export ACT_STORE=~/.cache/noeira/act_so101/DenisLabs__record-test_20260828_092736_240x320.h5
# The Python converter it replaced is still there and produces a byte-identical
# store, if you would rather use it:
#   pixi run -e nvidia python tools/act/lerobot_v3_to_store.py \\
#       --repo DenisLabs/record-test_20260828_092736 --height 240 --width 320

# ── 4. ImageNet ResNet18 weights (47 MB) ─────────────────────────────────
# Two routes to the same weights; only the variable below changes.
# `load_backbone_auto` dispatches on the STRING: `hub` (or `1`) fetches, a
# `*.safetensors` value is read as that file, anything else is a dump DIRECTORY.
#
#   4a. the Hub file — NO PyTorch, no `-e act-ref`, no dump step. THIS IS THE
#       DEFAULT: with ACT_PRETRAINED unset you get it, downloaded on the first
#       run and cached afterwards. Nothing to export.
#       `ACT_PRETRAINED=random` opts out and trains a from-scratch backbone.
#
#   4b. the torchvision dump — the original path, and the oracle 4a is gated
#       against (`tests/nn/test_safetensors_resnet18_torch.mojo` compares all
#       11,190,912 values). The ONLY step in this list that needs PyTorch.
# pixi run -e act-ref python tools/act/dump_resnet18_imagenet.py \\
#     --out ~/.cache/noeira/act_so101/resnet18_imagenet
# export ACT_PRETRAINED=~/.cache/noeira/act_so101/resnet18_imagenet

# ── 5. check both before spending GPU hours on them ──────────────────────
pixi run -e nvidia mojo run -I . tests/deep_agents/act/test_act_dataset.mojo
# ⚠ Both backbone gates read the 4b dump. On the 4a route there is nothing to
# check here — the Hub file is verified against that dump, so verifying it
# needs the dump too. Run them on a box that has one, not on the rented GPU.
pixi run -e nvidia mojo run -I . \\
    tests/deep_agents/act/test_act_pretrained_backbone.mojo
# The Hub file itself IS checkable without PyTorch — pinned hash and every
# tensor the backbone loads. Run this one on the GPU box; it is the only
# backbone gate that can run there.
pixi run -e nvidia mojo run -I . tests/nn/test_resnet18_hub_weights.mojo

# ── 6. metrics (optional) ────────────────────────────────────────────────
# `.env` in the project root, read by noeira/core/dotenv.mojo:
#     NOEIRA_CLOUD_URL=https://...
#     NOEIRA_CLOUD_API_KEY=...
# Absent, the run is unaffected. It is NOT copied by git — put it there by
# hand on the new box or the run streams nothing and says so.

# ── 7. build (~6 min) and run ────────────────────────────────────────────
pixi run -e nvidia mojo build -I . -o /tmp/act_train_gpu \\
    examples/so101/act_so101_train_gpu.mojo
/tmp/act_train_gpu
```

The startup lines name the store, the backbone initialization (`ImageNet
weights, N tensors` or `RANDOM`), and whether metrics are streaming. Read all
three: each is an environment variable whose absence is silent and legal, so a
run configured wrong looks exactly like a run configured right until the curve
comes back different.

⚠ **Run it from the project root.** `noeira/io/hdf5` resolves libhdf5 through
a path relative to the working directory (`.pixi/envs/<env>/lib/`), so the
binary aborts with `symbol not found: H5PLprepend` anywhere else. It also reads
`.env` from the working directory.

### Environment variables, all optional

| | |
|---|---|
| `ACT_PROJECT` | the project the run is filed under (default `so101`). It must exist on the box — `project-pull` it first — or the run is refused rather than filed outside it |
| `ACT_STORE` | the `.h5` to train on; default is the 5-episode recording |
| `ACT_PRETRAINED` | **defaults to `hub`** — the ImageNet backbone, fetched with no PyTorch and cached. A `dump_resnet18_imagenet.py` directory uses the torchvision dump; `random` trains a from-scratch backbone |
| `ACT_STEPS` | step count, **without a rebuild** — the graph takes ~6 min to compile, so "run it longer" must not mean "build it again" |
| `ACT_SEED` | the run's seed (default 7): the weight init, the episode split and the batch draws. ⚠ One run per recipe cannot rank recipes: the so101_tower vision student scored 32 / 56% on the SAME recipe (a data subset), so compare >= 3 seeds per arm |
| `ACT_NO_MONITOR` | force the logger inert with the keys present; what a smoke run should use so it does not land in the dashboard beside a real one |
| `ACT_CKPT` | read by `act_so101_openloop_eval.mojo`, not by this file |

On Apple add `-Xlinker -ld_classic`: the fully-expanded graph type mangles to a
symbol longer than the new linker accepts. Healthy source, toolchain limit;
`mojo run` JITs and never invokes ld. NVIDIA needs no flag.

### Resuming, and what a killed run leaves

`runs/<id>/checkpoints/last.ckpt` is rewritten at every validation and
`best.ckpt` whenever validation improves, so a kill loses at most `VAL_EVERY`
steps of progress and never the best model. The run prints its directory at
startup and `run.kv` beside the checkpoints records what the run was and how
it ended.

⚠⚠ THEY USED TO LIVE IN `/tmp` UNDER A `comptime` CONSTANT, which meant every
run of this driver overwrote the previous run's checkpoints by construction —
`docs/PROJECT_LAYER_PLAN.md` §1.1 quotes this file as the example. They are now
under the run's own directory and no two runs can collide. ⚠ That directory is
still on the box: copy it somewhere durable before releasing a rented one,
until P3 uploads artifacts during the run.

## What this configuration is, and what it is not

| | here | paper |
|---|---|---|
| chunk `K` | 60 | 100 |
| `hidden_dim` | 256 | 512 |
| `dim_feedforward` | 1024 | 3200 |
| encoder layers | 4 | 4 |
| decoder layers | 1 | 7 (output-equivalent — see `config.mojo`) |
| heads / latent | 8 / 32 | 8 / 32 |
| batch | 16 | 8 |
| images | 2 x 240x320 | 4 x 480x640 |

**`K = 60` is the paper's horizon, not a reduction.** ALOHA runs at 50 Hz and
chunks 100 steps — 2.0 seconds. This data is 30 fps, so the same two seconds is
60 frames. Copying the number 100 would have copied the wrong quantity.

`enc_layers` matches the paper. `hidden_dim` and `dim_feedforward` do not, and
that gap is a STEP-TIME choice, not a build-time one.

⚠ This paragraph used to say the graph type expands with each of these and that
COMPILE time bounds the file. **Measured 2026-08-31, that is wrong for both**
(`target="gpu"`, `N_ENC=4`, `BATCH=16`, cold builds, `-Xlinker -ld_classic`):

| `hidden_dim` | `dim_feedforward` | build |
|---|---|---|
| 128 | 1024 | 150 s |
| 256 | 1024 | 155 s |
| 256 | 3200 (the paper's) | 153 s |

A 3% spread against ~15% run-to-run noise. Doubling the width and tripling the
feedforward cost NOTHING to build. Neither does `enc_layers`/`dec_layers` —
`RepeatConditional[N, Layer]` instantiates ONE layer type and repeats it at
runtime, so depth is free too (measured flat from N=4 to N=32 on the combinator
in isolation). What compile time DOES scale with is the number of distinct
module POSITIONS, which is why the backbone dominates: swapping ResNet18 for the
5-conv `Stub` was 136 s -> 77 s. That, not the transformer dims, is why the ACT
GPU gate exists in its stub-backbone form — the original reason recorded here,
that a full-backbone build never finished on CUDA, is consistent with it and
still stands (it is a backbone cost, not a `hidden_dim` one).

⚠⚠ Measured on APPLE/Metal. CUDA is a third codegen path and is NOT covered by
these numbers — re-measure before quoting them on the 5090.

So raise them if STEP time allows. The build will not object; a 5090 running out
of memory or wall-clock still might.

`batch = 16` rather than the paper's 8: 15,447 frames at 2 cameras is 32 images
per step, which is still small for a 5090, and halving the step count per epoch
is free wall-clock. It is a lever, not a finding.

## Two things this run does that the CPU example does not

**Validation is DETERMINISTIC.** The sampler's RNG is pinned to a fixed seed
around every validation pass, so each one draws the SAME batches. Otherwise
`best_val` selects on a lucky draw — over hundreds of validations from only 10
held-out episodes, the minimum of a noisy estimate is mostly noise, and the
checkpoint it keeps is not the best model.

**A `last` checkpoint is written at every validation**, beside the `best` one.
A multi-hour run that is killed, or an OOM at hour three, otherwise leaves
nothing to resume from or evaluate.

## Metrics

Streams to the monitoring server named by `NOEIRA_CLOUD_URL` / `NOEIRA_CLOUD_API_KEY`
in `.env` (`noeira/core/dotenv.mojo`), the same path
`examples/half_cheetah/sac_half_cheetah_training_gpu.mojo` uses. With neither
set the logger is inert and the run is unaffected — training must not depend on
a monitoring server being reachable. `ACT_NO_MONITOR=1` forces it inert with
the keys present, which is what a smoke run should use so it does not land in
the dashboard beside a real one.

    train/l1  train/kl  train/loss  train/grad_norm  train/epoch   every 50 steps
    val/l1  val/kl  best/val_l1  perf/s_per_step                   every 1000

Training metrics are the MEAN over their window, not one batch's value: at
batch 16 a single step's L1 is noisy enough to hide the trend, and 100,000
individual points is not a curve anyone reads. The buffer is flushed at every
validation rather than only when full — the point of streaming a multi-hour run
is watching it while it runs.

`train/l1` is the quantity to watch against `val/l1`; `train/grad_norm` is the
one that tells you if the run is about to diverge, and it is worth having
precisely because gradient clipping is OFF (below).

## ⚠ `max_grad_norm = 0.0`

The clip walks every gradient slab through the host (`trainer.mojo:_SumSq`) —
a synchronisation point per parameter, the wrong shape for a hot loop. Zero is
also what the reference effectively does: it parses `--clip_max_norm 0.1` and
never applies it.

## What to expect

40 training episodes / 10 held out, ~12,400 training frames, no augmentation.
`ACT_AUGMENT=light|default` turns on image augmentation
(`deep_agents/act/augment.mojo`, DR plan Phase 1) for TRAINING batches, on
either data path (device gather kernels, or the host sampler under
`-D ACT_HOST_DATA`); validation is never augmented.

The FIRST 50-episode run, with a random backbone, is the baseline to beat:
best val L1 **0.4076 at epoch 15.5**, then 26 consecutive validations all worse
while train L1 fell another 1.7x. 0.159 s/step, ~4.9 h if left to 100,000 steps
— which is why `PATIENCE` exists.

A pretrained backbone should push the turn later and lower; that it does is not
yet measured. `best/val_l1` against the 0.4076 line is the comparison, and
`backbone_init` in the run config distinguishes the two runs in the dashboard.

⚠ Watch `train/kl`. In the random-backbone run it collapsed 12.1 -> 0.0002 —
posterior collapse at `kl_weight = 10`, meaning the CVAE encoder matched the
prior exactly, `z` carried no information, and ACT ran as a plain chunk
predictor with a decorative encoder. The paper's own ablation says the CVAE is
what matters for HUMAN-collected demonstrations, which is what this dataset is.
Whether the reference collapses the same way at this scale is unmeasured.

The `best` checkpoint is what `act_so101_openloop_eval.mojo` should be pointed
at; it reports a `hold` baseline so "it produces plausible actions" cannot be
mistaken for "it learned something".
"""
from std.os import getenv
from std.random import seed as seed_rng
from std.sys import is_defined
from std.os.path import exists
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.deep_agents.act.config import (
    act_pretrained_spec,
    RUN_DEC_LAYERS,
    RUN_DIM,
    RUN_ENC_LAYERS,
    RUN_FF,
    RUN_HEADS,
    RUN_K,
    RUN_LATENT,
    RUN_LR,
    SO101_ADIM,
    SO101_IMG_H,
    SO101_IMG_W,
    SO101_N_CAM,
    SO101_QPOS,
)
from noeira.deep_agents.act.data import ACTDataset, IMAGES_RESIDENT_MAX_BYTES
from noeira.deep_agents.act.data_gpu import ACTDeviceDataset
from noeira.deep_agents.act.augment import ImageAugConfig
from noeira.deep_agents.act.trainer import (
    ACTTrainer,
    ACTWindowMetrics,
)
from noeira.core.dotenv import load_dotenv
from noeira.core.logger import RemoteLogger
from noeira.core.project import project_exists
from noeira.deep_agents.act.norm_file import act_norm_from
from noeira.io.json import load_json
from std.os import makedirs
from noeira.core.run import RunContext, register_run
from noeira.deep_agents.training.checkpoint import announce_checkpoint
from noeira.io.artifact_sink import ArtifactSink, close_sink, sink_for_run



# ── shape (must match the store; see the header) ─────────────────────────
comptime QPOS = SO101_QPOS
comptime ADIM = SO101_ADIM
comptime N_CAM = SO101_N_CAM
comptime IMG_H = SO101_IMG_H
comptime IMG_W = SO101_IMG_W

# ── model (see the table in the header) ──────────────────────────────────
# ⚠ FROM `act.config`'s RUN_* block, because these dims ARE the checkpoint's
# parameter shapes and `act_so101_openloop_eval.mojo` must agree with them
# exactly. They drifted once; one definition now.
comptime K = RUN_K  # 2.0 s at 30 fps — the paper's horizon, not its frame count
comptime DIM = RUN_DIM  # paper: 512
comptime HEADS = RUN_HEADS  # paper: 8
comptime FF = RUN_FF  # paper: 3200
comptime LATENT = RUN_LATENT  # paper: 32 (unchanged — it is small already)
comptime N_ENC = RUN_ENC_LAYERS  # paper: 4
comptime N_DEC = RUN_DEC_LAYERS  # paper: 7, output-equivalent to 1
comptime BATCH = 16  # paper: 8

comptime DEFAULT_STEPS = 100000
"""~100 epochs over 40 episodes at batch 16 (966 steps/epoch). Override with
`ACT_STEPS`; the graph takes minutes to compile and extending a run must not
require rebuilding it."""
# GPU_DATA — draw and normalize batches ON THE DEVICE (`ACTDeviceDataset`)
# instead of on the host. Measured on the profile script, RTX 5090: the
# iteration went 144.8 -> 61.6 ms and GPU busy 38% -> 88%, because the host
# path spent 16.1 ms per step converting 7.4M uint8 to float32 single-threaded
# with the GPU idle, then pushed 29.5 MB across the bus.
#
# ⚠ It does NOT draw the same batches as the host sampler — Philox on the
# device against a xorshift on the host — so a run with this on is not
# step-for-step comparable with an older log. The LOSS CURVE should be
# statistically the same; individual steps will not match.
#
# ⚠ Startup uploads the whole store as uint8 (7.1 GB for 50 episodes, ~13 s).
# A store that does not fit the device is trained on the host path instead:
# build with `-D ACT_HOST_DATA` (the images then STREAM from the store a row
# at a time, `ACTDataset`'s own residency rule). Measured need: the so101_tower
# vision student's DAgger store — 200 expert episodes + one round of expert
# relabels, ~61k rows = 28 GB of uint8 — does not fit a 32 GB 5090 beside the
# model, and capping the expert half to make room REPLACED data instead of
# aggregating it (round 1 fell from 32% to 16% cube-in-bowl). See
# `docs/ACT_GPU_DATA_PATH.md` for the windowed design that would keep the
# device path.
comptime GPU_DATA = not is_defined["ACT_HOST_DATA"]()


# USE_CUDA_GRAPH — capture the per-step device kernel sequence into a CUDA
# graph and replay it (NVIDIA only; a compile-time no-op elsewhere).
#
# ⚠ DEFAULT OFF, AND IT DOES NOT WORK YET. Measured on the 5090: capture
# BEGINS and then dies on an 18.75 MB allocation, "graph capturing in progress,
# no driver fallback". Not out of memory — MAX has 113 GB free but serves only
# from an empty `graphFreeList` while capturing, so ANY per-call device
# allocation is fatal. Probably not split-K either: 18.75 MB is exactly
# 16*64*60*80 fp32, i.e. the BACKBONE at post-maxpool resolution. See
# `docs/ACT_GPU_DATA_PATH.md`. This switch exists so that when the allocation
# is found and removed, testing it is a one-line change.
#
# ⚠ Turning this on ABORTS the run when capture fails. Deliberate: a silent
# fallback would leave `ds.offset_host` advanced by a step the device threw
# away, and a desynced RNG mirror is worse than a crash.
#
# Requires GPU_DATA: a captured step cannot contain the host sampler.
comptime USE_CUDA_GRAPH = False
# (the GPU_DATA requirement is asserted at the top of `main`)

# Eager steps before the first captured one. `maybe_capture_replay` warms the
# stream with one run of its own, which is not the same as warming every LAZY
# DEVICE ALLOCATION the graph performs: anything still allocating on step 2
# would allocate INSIDE the capture region and abort it. Cheap insurance.
comptime CAPTURE_WARMUP = 8

comptime VAL_EVERY = 1000
comptime VAL_BATCHES = 64
"""1024 validation samples per pass, from a pinned RNG — see the header.

Was 16 (256 samples). The held-out split is **3,036 frames**, so that scored
**8%** of it, and the LeRobot baseline — which evaluates the ENTIRE held-out
set — kept improving through epochs where our curve had gone flat. A fixed
sample has no noise BETWEEN passes (the RNG is pinned), but 8% of the holdout
is a weak estimate of the holdout, and `PATIENCE` stops the run on it.

64 batches is 34% and costs ~3 s per validation. Full coverage would want a
deterministic sweep rather than 190 random draws with replacement, which is a
different change to `ACTDataset`."""
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
comptime LR = RUN_LR
"""Paper: 1e-5, with a pretrained backbone it barely moves. This backbone is
random at step 0 and has to learn vision from scratch, so it needs the higher
rate — the same reasoning as the single-lr deviation in `ACT_PORT.md`."""
comptime KL_WEIGHT = 10.0  # paper: 10

comptime LOG_EVERY = 50
"""Training metrics stream as the MEAN over this window, not as one batch's
value. At batch 16 a single step's L1 is noisy enough to hide the trend, and
100,000 individual points is not a curve anyone reads — 2,000 windowed ones
is. Validation goes at `VAL_EVERY`, unaveraged: it is already a mean over
`VAL_BATCHES` fixed batches."""

comptime DDS = ACTDeviceDataset[QPOS, ADIM, N_CAM, IMG_H, IMG_W]

comptime T = ACTTrainer[
    QPOS, ADIM, N_CAM, IMG_H, IMG_W, K, DIM, HEADS, FF, LATENT, N_ENC, N_DEC,
    BATCH, 0.1, "gpu",
]
comptime IMG_ELEMS = N_CAM * 3 * IMG_H * IMG_W


def _store_cameras(store: String) raises -> List[String]:
    """Slot-ordered camera keys from the importer's `<store>.json`, or none."""
    var out = List[String]()
    if not store.endswith(".h5"):
        return out^
    var side = String(store[byte=0 : store.byte_length() - 3]) + ".json"
    if not exists(side):
        return out^
    var doc = load_json(side)
    var cams = doc.field(doc.root(), String("cameras"))
    if cams >= 0:
        for i in range(doc.size(cams)):
            out.append(doc.string(doc.at(cams, i)))
    return out^


def act_project() raises -> String:
    """`$ACT_PROJECT`, else `so101` — the project this run is filed under.

    ⚠⚠ A NAMED PROJECT MUST EXIST ON THIS BOX. `RunContext` falls back to the
    flat `runs/` root when the project directory is missing, which is right for
    a driver that never named one and wrong for a run someone pointed at
    `so101-tower`: its checkpoints would land outside the project, and
    `project-promote` would not find them. So the default keeps the fallback,
    and an explicit name refuses instead.
    """
    var env = getenv("ACT_PROJECT")
    if env.byte_length() == 0:
        return String("so101")
    if not project_exists(env):
        raise Error(
            "ACT_PROJECT=" + env + " but there is no projects/" + env
            + "/project.kv on this box — run `pixi run project-pull " + env
            + "` first, or the run would be filed outside the project"
        )
    return env^


def store_path() raises -> String:
    """`$ACT_STORE` if set, else the recording the header names.

    The default is a specific recording, not a pattern: an example that
    silently picked up whichever store happened to be newest in the cache
    would report numbers nobody could attribute to a dataset. Point
    `ACT_STORE` at another store to train on it.
    """
    var env = getenv("ACT_STORE")
    if env.byte_length() > 0:
        return env^
    var home = getenv("HOME")
    if home == "":
        raise Error("$HOME is unset; set ACT_STORE to the store path")
    return (
        home
        + "/.cache/noeira/act_so101/"
        + "DenisLabs__record-test_20260825_094319_"
        + String(IMG_H) + "x" + String(IMG_W) + ".h5"
    )


def main() raises:
    comptime assert not USE_CUDA_GRAPH or GPU_DATA, (
        "USE_CUDA_GRAPH requires GPU_DATA — a captured step cannot contain host"
        " work, and the host sampler is host work."
    )

    # ⚠ FIRST, before the store: a wrong project must fail in a second, not
    # after minutes of loading 7 GB onto the device.
    var project = act_project()
    var path = store_path()
    if not exists(path):
        print("MISSING STORE: " + path)
        print("build it with examples/so101/act_so101_import_dataset.mojo"
      " — see the header")
        raise Error("store not found")

    var steps = Int(DEFAULT_STEPS)
    var env_steps = getenv("ACT_STEPS")
    if env_steps.byte_length() > 0:
        steps = Int(env_steps)
        if steps < 1:
            raise Error("ACT_STEPS must be >= 1, got " + env_steps)

    var ctx = DeviceContext()
    print("ACT / SO-ARM101 — GPU training")
    print("  device  " + String(ctx.name()))
    print("  store   " + path)
    print(
        "  model   K=" + String(K) + " dim=" + String(DIM)
        + " heads=" + String(HEADS) + " ff=" + String(FF)
        + " enc=" + String(N_ENC) + " dec=" + String(N_DEC)
    )
    print(
        "  data    " + String(N_CAM) + " cameras at " + String(IMG_H) + "x"
        + String(IMG_W) + ", batch " + String(BATCH)
    )

    # `ACT_RESIDENT_GB`: how large an image column the HOST keeps in RAM
    # (default `IMAGES_RESIDENT_MAX_BYTES`, 2 GiB); above it the images stream
    # from the store a row at a time. ⚠ ON THE HOST PATH (`-D ACT_HOST_DATA`)
    # THIS DECIDES THE SPEED: streaming a deflated 61k-row store measured
    # ~0.6 s/step on the 5090 box (one core decompressing 460 KB rows), where
    # a resident column pays only the host normalisation. Unused by the device
    # path, which uploads the column whole either way.
    var resident_gb = getenv("ACT_RESIDENT_GB")
    var max_img = IMAGES_RESIDENT_MAX_BYTES
    if resident_gb.byte_length() > 0:
        max_img = Int(Float64(resident_gb) * Float64(1 << 30))
    var run_seed = 7
    var env_seed = getenv("ACT_SEED")
    if env_seed.byte_length() > 0:
        run_seed = Int(env_seed)
    seed_rng(run_seed)
    print("  seed    " + String(run_seed)
          + ("" if env_seed.byte_length() == 0 else " (ACT_SEED)"))
    var ds = ACTDataset[QPOS, ADIM, N_CAM, IMG_H, IMG_W](
        String(path), seed=UInt64(run_seed), max_image_bytes=max_img
    )
    print(
        "  split   " + String(len(ds.train_eps)) + " train / "
        + String(len(ds.val_eps)) + " val episodes of "
        + String(ds.n_episodes())
    )
    print("")

    # ── remote metrics ───────────────────────────────────────────────────
    # Configured from `.env` (`NOEIRA_CLOUD_URL`, `NOEIRA_CLOUD_API_KEY`). With
    # neither set the logger is inert and the run is unaffected — a training
    # run must not depend on a monitoring server being reachable.
    # `ACT_NO_MONITOR=1` forces it inert. A three-step smoke should not show
    # up in the dashboard next to a six-hour run, and blanking the shared
    # `.env` to achieve that is how a real run silently loses its metrics.
    var env_vars = load_dotenv()
    var no_monitor = getenv("ACT_NO_MONITOR")
    var monitor_url = (
        String("") if no_monitor.byte_length() > 0
        else env_vars.get("NOEIRA_CLOUD_URL", "")
    )
    # ⚠⚠ `run_name="ACT SO-ARM101 (GPU)"` WAS IDENTICAL ON EVERY RUN of this
    # driver — the dashboard showed a column of indistinguishable rows, which
    # is pain 2 of `docs/PROJECT_LAYER_PLAN.md` exactly. The run's id is unique
    # and sortable; grouping comes back with project scoping in P2.
    print("  project " + project)
    var run = RunContext(
        project=project,
        driver=String("examples/so101/act_so101_train_gpu.mojo"),
        slug=String("act-so101"),
        env=String("builtin:so_arm101"),
        dataset=path,
        device=String(ctx.name()),
        seed=run_seed,
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
    logger.set_config("target", "gpu")
    logger.set_config("device", String(ctx.name()))
    logger.set_config("store", path)
    logger.set_config("chunk_k", String(K))
    logger.set_config("hidden_dim", String(DIM))
    logger.set_config("dim_feedforward", String(FF))
    logger.set_config("heads", String(HEADS))
    logger.set_config("latent", String(LATENT))
    logger.set_config("enc_layers", String(N_ENC))
    logger.set_config("dec_layers", String(N_DEC))
    logger.set_config("batch", String(BATCH))
    logger.set_config("lr", String(LR))
    logger.set_config("kl_weight", String(KL_WEIGHT))
    logger.set_config("cameras", String(N_CAM))
    logger.set_config("image", String(IMG_H) + "x" + String(IMG_W))
    logger.set_config("steps", String(steps))
    logger.set_config("train_episodes", String(len(ds.train_eps)))
    logger.set_config("val_episodes", String(len(ds.val_eps)))
    # `ACT_AUGMENT=off|light|default` — an unknown value raises rather than
    # running an un-augmented experiment under an augmented name.
    var aug = ImageAugConfig.parse(getenv("ACT_AUGMENT"))
    # Both samplers augment: the device one in its gather kernels, the host
    # one (`-D ACT_HOST_DATA`) through the same draw and per-pixel rules
    # (`act/augment.mojo`). Validation is never augmented on either.
    comptime if not GPU_DATA:
        ds.set_augment(aug)
    logger.set_config("augment", String(aug))
    print("  augment " + String(aug))
    print(
        "  metrics " + (
            "streaming to " + monitor_url if logger.is_active()
            else (
                "OFF (ACT_NO_MONITOR)" if no_monitor.byte_length() > 0
                else "local only (set NOEIRA_CLOUD_URL in .env)"
            )
        )
    )

    var tr = T.make(
        lr=Scalar[DT](LR),
        kl_weight=Scalar[DT](KL_WEIGHT),
        max_grad_norm=Scalar[DT](0.0),  # see the header
        ctx=ctx,
    )

    # The device copy of the dataset. Timed and printed: ~7.1 GB of HDF5 read
    # plus H2D is a real one-time cost, and a run that hides it inside the
    # first step's timing would be lying about both.
    var dev_ds = DDS()
    comptime if GPU_DATA:
        var u0 = perf_counter_ns()
        dev_ds = DDS.upload_from[BATCH](ds, ctx, seed=UInt64(run_seed))
        dev_ds.set_augment(aug)
        var u1 = perf_counter_ns()
        print(
            "  device dataset    " + String(Float64(u1 - u0) / 1e9) + " s to"
            " upload " + String(Float64(dev_ds.n_rows)
                                * Float64(IMG_ELEMS) / 1e9)
            + " GB uint8 (once)"
        )

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
    # ⚠ Resolved by `act_pretrained_spec`, not read here — see its docstring.
    # The default is `hub`; `ACT_PRETRAINED=random` opts out.
    var pretrained = act_pretrained_spec()
    if pretrained.byte_length() > 0:
        # `ACT_NO_FREEZE_BN=1` loads the weights and leaves BatchNorm trainable
        # — the ablation, not a normal setting. Both references pass
        # `norm_layer=FrozenBatchNorm2d` at the same call that loads the
        # weights, and unfrozen BatchNorm EMAs the ImageNet statistics away in
        # a few hundred steps while never reading them.
        var no_freeze = getenv("ACT_NO_FREEZE_BN")
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

    # ⚠⚠ REGISTERED AFTER THE LAST `set_config`, AND IT WAS NEVER CALLED. The
    # registration is what carries `project=` to the dashboard: without it
    # this driver's runs reached the monitor with no project, so they appeared
    # under "All runs" and on no project's page.
    register_run(run, logger)

    # ⚠⚠ THE ARTIFACT UPLINK — `best` and `last` leave the box while the run
    # is going, so a rented GPU that dies at hour three does not take the
    # weights with it. `sink_for_run` returns None when `.env` names no
    # monitor, and every `announce_checkpoint` is a no-op on a None.
    #
    # ⚠ `ACT_NO_MONITOR` MUST SILENCE THIS TOO. `sink_for_run` reads `.env`
    # itself, so without this branch a run told to stay off the dashboard
    # would still upload ~130 MB per validation to it.
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

    # ⚠⚠ norm.json BESIDE THE CHECKPOINTS, written before the first one. A
    # checkpoint carries no normalization statistics; a deployment used to
    # need this whole store for them. See `deep_agents/act/norm_file.mojo`.
    var norm_path = run.checkpoint_path(String("best"))
    norm_path = String(norm_path[byte=0 : norm_path.rfind("/")]) + "/norm.json"
    var norm = act_norm_from(
        ds.qpos_raw, ds.action_raw, ds.n_rows(), ds.n_episodes(),
        ds.qpos_mean, ds.qpos_std, ds.action_mean, ds.action_std,
        _store_cameras(path), IMG_H, IMG_W, path,
    )
    makedirs(String(norm_path[byte=0 : norm_path.rfind("/")]), exist_ok=True)
    norm.save(norm_path)
    announce_checkpoint(norm_path, artifacts, run.dir)
    print("  norm    " + norm_path)
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

    var t_run0 = perf_counter_ns()

    # ⚠ TIMED AROUND THE TRAINING STEP ONLY. Measuring wall clock between
    # report points folds the validation pass into the rate, and at step 0 —
    # where the interval is zero steps wide — it folded in 16 validation
    # batches plus every first-launch kernel compilation and called the result
    # "1.38 s/step", then multiplied it by 100,000 and printed 38 hours. A
    # progress line that cannot be trusted on its first appearance is worse
    # than none.
    var train_ns = 0
    var data_ns = 0
    var train_steps = 0

    # Log-spaced early probes so the real rate is known in seconds rather than
    # after the first validation 1000 steps in. No validation, no checkpoint —
    # just the number that decides whether this run is worth leaving alone.
    var probes = List[Int]()
    probes.append(1)
    probes.append(5)
    probes.append(20)
    probes.append(100)
    probes.append(300)

    # Windowed training means — see LOG_EVERY.
    var acc_l1 = Float64(0.0)
    var acc_kl = Float64(0.0)
    var acc_loss = Float64(0.0)
    var acc_gn = Float64(0.0)
    var acc_n = 0
    var last_l1 = Float64(0.0)
    """Most recent training l1, for the progress lines only — never logged.

    ⚠ It means different things on the two paths, which is the honest option
    rather than a hidden one. On the host path it is the LAST STEP's l1, as
    before. Under GPU_DATA there is no per-step l1 — that is the point — so it
    is the last WINDOW MEAN, and the progress lines prefer a peek at the live
    window when one has steps in it. Reads 0 only before anything has been
    measured."""
    var last_kl = Float64(0.0)
    """Companion to `last_l1`, host path only — the device path's progress line
    takes both from the peeked window."""

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
    val_names.append(String("data_s"))
    val_names.append(String("gpu_s"))
    val_names.append(String("val_l1_loss_best"))

    # Capture prerequisites: adopt the arena eagerly and force the padded /
    # bf16 weight caches to refresh every forward. WITHOUT the second, a replay
    # never runs the host version-bump at the end of `Adam.step`, the caches
    # decide they are still current, and the model trains against its
    # capture-time weights while the optimizer updates memory nobody reads —
    # a flat loss curve that reads as a bad learning rate.
    if USE_CUDA_GRAPH:
        tr.prepare_device_capture()
        print(
            "  cuda graph        ON (capture after " + String(CAPTURE_WARMUP)
            + " eager steps; weight caches forced to refresh every forward)"
        )
        print("")
    var announced_graph = False

    for s in range(steps):
        # Split, because "0.176 s/step" does not say WHICH half. `sample_batch`
        # is host-side, single-threaded and serial with the device: per sample
        # it converts N_CAM*3*H*W uint8 to float32 one element at a time
        # (divide, subtract, multiply) and reads a row from HDF5. At batch 16
        # that is 7.4M elements before the GPU sees anything, and if it is the
        # larger half then no kernel work will fix the step time.
        var step_t0 = perf_counter_ns()
        comptime if not GPU_DATA:
            ds.sample_batch[K, BATCH](False, qpos, images, actions, valid)
        var t_data = perf_counter_ns()
        # ⚠ Under GPU_DATA the draw happens INSIDE the step, so `perf/s_data`
        # reads ~0 and the sampler's cost is inside `perf/s_per_step`. The
        # split is not comparable across the two settings; the total is.
        # ⚠ Two shapes, deliberately. Under GPU_DATA the step folds its four
        # logged scalars into DEVICE accumulators and returns nothing: a
        # per-step value would cost the two synchronizations and four D2Hs
        # that path exists to delete. Off it, the host path returns them and
        # the host accumulators below take over. `acc_n` is what says which
        # one is live — it stays 0 under GPU_DATA.
        comptime if GPU_DATA:
            # ⚠ A RUNTIME `if` on a COMPTIME flag, deliberately. `comptime if`
            # PRUNES: with capture off the captured path would never be
            # elaborated, and a type error in it would surface only on the day
            # someone flips the flag — which is exactly the day this is
            # supposed to be one flag away from working. A runtime branch on a
            # constant costs a dead-code warning and nothing else.
            if USE_CUDA_GRAPH and s >= CAPTURE_WARMUP:
                tr.train_step_device_captured(dev_ds)
            else:
                tr.train_step_device_accum(dev_ds)
        else:
            var r = tr.train_step(qpos, images, actions, valid)
            last_l1 = r.l1
            last_kl = r.kl
            acc_l1 += r.l1
            acc_kl += r.kl
            acc_loss += r.loss
            acc_gn += r.grad_norm
            acc_n += 1
        var t_end = perf_counter_ns()
        data_ns += t_data - step_t0
        train_ns += t_end - step_t0
        train_steps += 1

        # ⚠ Printed once, and worth printing. A capture that recorded NOTHING
        # — the closure enqueued on a stream other than the one being captured
        # — is indistinguishable from a working one until the loss stops
        # moving, and replaying an empty graph is a training loop running at
        # full speed doing nothing.
        if USE_CUDA_GRAPH:
            if not announced_graph and tr.has_captured_graph():
                announced_graph = True
                print(
                    "  [CUDA graph] captured "
                    + String(tr.captured_graph_nodes())
                    + " nodes at step " + String(s)
                )

        var is_probe = False
        for i in range(len(probes)):
            if s == probes[i]:
                is_probe = True
        if is_probe:
            var rate = Float64(train_ns) / Float64(train_steps) / 1e9
            var rate_d = Float64(data_ns) / Float64(train_steps) / 1e9
            # PEEKED, not flushed — a progress line must not empty the window
            # the logger is filling.
            var pl1 = last_l1
            comptime if GPU_DATA:
                var pw = tr.train_metrics(False)
                if pw.n > 0:
                    pl1 = pw.l1
            print(
                "  step " + String(s) + "  train l1 " + String(pl1)
                + "  |  " + String(rate) + " s/step (" + String(rate_d)
                + " data), ~" + String(Int(rate * Float64(steps) / 60.0))
                + " min for " + String(steps) + " steps"
            )
            train_ns = 0
            data_ns = 0
            train_steps = 0

        # The window means, from wherever they were accumulated. On the device
        # path this is the ONLY D2H in `LOG_EVERY` steps — four `[2]`-buffer
        # reads — against four downloads and two full device drains PER STEP
        # before.
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
            # Validation L1 is the reference's model-selection metric
            # (`imitate_episodes.py` keeps the checkpoint with the lowest
            # validation loss), so that is what the best checkpoint tracks.
            #
            # ⚠ The sampler's stream is PINNED and restored around the pass, so
            # every validation scores the same batches. Left random, `best_val`
            # would be the minimum of a noisy estimate over ~100 passes — which
            # selects the luckiest draw, not the best model — and successive
            # points on the curve would not be comparable to each other.
            # ⚠ Validation is NOT captured, on purpose. It flips train/eval
            # mode — host attribute writes — and runs a different number of
            # forwards; capturing it would bake eval mode into a graph the
            # training loop then replays. The captured TRAIN graph is
            # unaffected: it replays the kernels recorded under training mode
            # regardless of what the attributes say now, and this block always
            # restores training mode on the way out.
            var saved_rng = ds.rng
            ds.rng = VAL_SEED
            # ⚠ The device sampler needs the SAME pinning, for the same
            # reason. Its stream is a device-resident Philox offset, mirrored
            # on the host so save/restore costs no D2H.
            var saved_off = dev_ds.offset_host
            comptime if GPU_DATA:
                dev_ds.set_offset(ctx, VAL_SEED)
            var vl1 = Float64(0.0)
            var vkl = Float64(0.0)
            comptime if GPU_DATA:
                # Same trade as the training step: the pass folds into the
                # device validation window and is drained ONCE, so 64 eval
                # batches cost one D2H instead of 64 downloads behind 64
                # synchronizations.
                for _ in range(VAL_BATCHES):
                    tr.eval_step_device_accum(dev_ds, True)
                var w = tr.val_metrics()
                vl1 = w.l1
                vkl = w.kl
            else:
                for _ in range(VAL_BATCHES):
                    ds.sample_batch[K, BATCH](
                        True, qpos, images, actions, valid
                    )
                    var v = tr.eval_step(qpos, images, actions, valid)
                    vl1 += v.l1
                    vkl += v.kl
                vl1 /= Float64(VAL_BATCHES)
                vkl /= Float64(VAL_BATCHES)
            ds.rng = saved_rng
            comptime if GPU_DATA:
                dev_ds.set_offset(ctx, saved_off)

            # The training numbers printed alongside. Under GPU_DATA there is
            # no per-step value to quote, so this is the partial window since
            # the last `LOG_EVERY` flush — PEEKED, so the logger's window is
            # left intact.
            var train_line = ACTWindowMetrics(0.0, 0.0, 0.0, 0.0, 0)
            comptime if GPU_DATA:
                train_line = tr.train_metrics(False)
                if train_line.n == 0:
                    train_line.l1 = last_l1
            else:
                # The host path still has a per-step value, so it prints the
                # same thing it always did.
                train_line.l1 = last_l1
                train_line.kl = last_kl

            # The mean over the training steps since the last report — never
            # a wall-clock interval, which would include this validation pass
            # and the checkpoint writes.
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
                + "  train l1 " + String(train_line.l1)
                + "  kl " + String(train_line.kl)
                + "  |  val l1 " + String(vl1)
                + "  |  " + String(sps) + " s/step ("
                + String(Int(100.0 * sps_data / (sps + 1e-12)))
                + "% data), ~" + String(Int(eta)) + " min left"
            )
            # Written EVERY pass: a run killed at hour three otherwise leaves
            # nothing to evaluate or resume from.
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
            vvals.append(sps_data)
            vvals.append(sps - sps_data)
            vvals.append(best_val)
            logger.log_scalars(val_names, vvals, s)
            # Flushed at every validation rather than only when the buffer
            # fills: the point of streaming a multi-hour run is watching it
            # while it runs, and a partly-full buffer is invisible.
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

    # ⚠ THE RUN RECORDS ITS OWN VERDICT — `outcome=` is what pain 1 asks for,
    # and no naming convention supplies it. `best/val_l1` against the 0.4076
    # line is this configuration's comparison.
    var outcome = (
        String("best_val_l1=") + String(best_val)
        + " best_step=" + String(best_step)
    )
    run.set_outcome(outcome)
    # ⚠ `finish` BEFORE `close`, WITH THE OUTCOME. `close()` finishes with an
    # EMPTY outcome if the driver did not, so the verdict used to reach
    # `run.kv` and never the dashboard.
    logger.finish(String("done"), outcome)
    logger.close()
    # ⚠ AFTER THE LOOP, NOT IN IT: this drains the uploads still queued, and
    # a `best` written at the last validation is exactly the one that matters.
    close_sink(artifacts)
    run.close()

    print("")
    print(
        "  wall clock " + String(Float64(perf_counter_ns() - t_run0) / 6e10)
        + " min for " + String(steps) + " steps"
    )
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
        + " training episodes with no augmentation. WHERE it turns is the"
    )
    print(
        "    number worth having. Evaluate the BEST checkpoint; see"
        " act_so101_openloop_eval.mojo."
    )
