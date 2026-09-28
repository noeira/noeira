# +--------------------------------------------------------------------------+ #
# | SmolVLA on the SO-ARM101 — fine-tuning the published checkpoint
# +--------------------------------------------------------------------------+ #
"""Fine-tune `lerobot/smolvla_base` on a LeRobot v3 recording, on GPU.

**No robot is required.** Training is entirely offline — a store of recorded
frames in, gradients out. The arm is needed to *deploy* a policy, not to fit
one, and the offline claim this run can actually make is a HELD-OUT loss that
falls: train on some episodes, measure on episodes never trained on.

## Launching a run on a fresh NVIDIA box, start to finish

```bash
git clone <this repo> noeira && cd noeira
pixi install -e nvidia

# ── 1. disk ───────────────────────────────────────────────────────────────
#   pixi env    ~20 GB
#   checkpoint  0.9 GB   lerobot/smolvla_base model.safetensors
#   HF dataset  ~2 GB    the recording's snapshot
#   store        27 GB   the converted .h5 at 480x640 (see step 3)
# Budget 55 GB.
df -h .

# ── 2. auth, ONLY if the dataset repo is private ──────────────────────────
pixi run -e nvidia hf auth login          # or: export HF_TOKEN=hf_...

# ── 3. the dataset -> a TrajectoryStore ───────────────────────────────────
# ⚠ 480x640, NOT the 240x320 the ACT example uses. SmolVLA resizes to 512x512
# itself with `resize_with_pad`, and importing at a reduced size would resample
# twice — `test_store_vs_camera_frame.mojo` gates the training path against the
# deployment path at 640x480 and both must see the same pixels.
pixi run -e nvidia mojo run -I . \\
    examples/so101/act_so101_import_dataset.mojo \\
    --repo DenisLabs/record-test_20260828_092736 --height 480 --width 640
export SMOLVLA_STORE=~/.cache/noeira/act_so101/DenisLabs__record-test_20260828_092736_480x640.h5

# ── 4. normalisation statistics — fetched automatically ──────────────────
# `meta/stats.json` is pulled from the SAME dataset repo as the store, so
# there is nothing to export. SmolVLA reads it rather than recomputing: ours
# and lerobot's differ by exactly sqrt(N/(N-1)) — sample vs population std —
# and training on one while the checkpoint was fit with the other is a silent
# scale error. `SMOLVLA_STATS` overrides it with a local file.

# ── 5. check the pieces before spending GPU hours ────────────────────────
pixi run -e nvidia test-vla                    # structural, no download
pixi run -e nvidia test-vla-gpu                # kernels at real shapes
pixi run -e nvidia test-vla-weights            # the real checkpoint loads

# ── 6. build, SMOKE, then run ────────────────────────────────────────────
pixi run -e nvidia mojo build -I . -o /tmp/smolvla_finetune \\
    examples/so101/smolvla_so101_finetune.mojo

# ⚠ THREE STEPS FIRST. Nothing in this port has ever run at the published
# depth with RECORD on — the tape is ~320 MB of per-layer activations that no
# gate has allocated — and the first thing to learn is whether it starts and
# what a step costs, not whether the loss falls.
SMOLVLA_STEPS=3 SMOLVLA_NO_MONITOR=1 /tmp/smolvla_finetune

# then the real thing. 2000 x 64 = 128k observations, about seven passes over
# the recording; the reference recipe is 20 000 x 64. Read the s/step the
# smoke run printed before choosing a number.
/tmp/smolvla_finetune
```

⚠ **Run it from the project root** — `noeira/io/hdf5` resolves libhdf5 through
a path relative to the working directory, and the tokenised instruction table
is read from `tools/vla/`.

### Environment variables

| | |
|---|---|
| `SMOLVLA_STORE` | the `.h5` to train on. **Required** — there is no default, because a fine-tune attributed to the wrong recording is worse than one that refuses to start |
| `SMOLVLA_STATS` | a local `meta/stats.json`. Optional — by default it is fetched from `SMOLVLA_REPO` |
| `SMOLVLA_REPO` | the dataset repo the statistics come from; defaults to the recording named above |
| `SMOLVLA_TASKS` | the tokenised instruction table; defaults to the checked-in one for this recording |
| `SMOLVLA_STEPS` | optimizer steps, without a rebuild. ⚠ **A SHORT RUN IS NOT A GENTLE RUN** — see below |
| `SMOLVLA_ACCUM` | observations per optimizer step (default 64, the reference's batch size); a multiple of the build's `B` |
| `SMOLVLA_LR` | default 1e-4 |
| `SMOLVLA_CKPT` | checkpoint path prefix (`<prefix>_best.ckpt` / `_last.ckpt`); default the run's own `runs/<id>/checkpoints/best.ckpt` / `last.ckpt` |
| `SMOLVLA_INIT` | a `*_best.ckpt` to start from, applied ON TOP of the base checkpoint |
| `SMOLVLA_VAL_EPISODES` | episodes held out at the END of the recording; default one fifth. **`0` trains on every episode** — see below |
| `SMOLVLA_PROFILE` | set to anything: per-phase wall times (host images / prefix / suffix forward / backward) in the log line, at the cost of three extra drains per observation |
| `SMOLVLA_VISION_CACHE` | path of the vision cache; default `<store>.vision.bin`; `off` recomputes the tower every observation (the pre-cache path, kept for A/B) |
| `SMOLVLA_NO_MONITOR` | force the metrics logger inert |
| `SMOLVLA_PROJECT` | the project the run is filed under (default `so101`); must exist on the box. `project-promote` promotes into the run's project |

⚠ **The held-out fifth is never trained on, and the deployed checkpoint is the
model that never saw it.** The reference recipe trains on all 50 episodes. If
a recording was made position by position — the published SO-101 dataset is
5 cube positions x 10 episodes, in order — the last fifth is one whole cube
position, and a policy fine-tuned here reaches next to it and misses. So the
procedure is two runs: one with the default split, to choose the step count
from a curve that is actually held out; then `SMOLVLA_VAL_EPISODES=0` at that
step count, for the checkpoint that goes on the arm. With `0` the "held-out"
groups are drawn from the TRAINING rows and the line says so — that curve is
a sanity number (finite, falling), not a generalisation measure, and `best`
selected by it is not a held-out best. Deploy `last.ckpt` from that run.

## What this run is, and what it is not

**The trainable set is the shipped default minus one flag.**
`train_expert_only = True` and — here — `train_state_proj = False`: the action
expert, the action projections, and nothing else. `state_proj` IS supported
(`TRAIN_STATE_PROJ`, gated by `test_state_proj_grad.mojo`) and costs a full
backward through sixteen frozen VLM layers to train one 32x960 matrix. Turn it
on when the cheaper regime has been shown to work, not before.

⚠ **The batch is `B` observations per forward, accumulated `accum // B` times
per update.** `B` is a comptime of this file (8; see its note) and is
established by `test_train_step_batched.mojo`: a batch of four distinct
observations reproduces four separate B = 1 runs, per row for the prefill
and summed for every trainable gradient. `Linear.vjp` accumulates natively
across the micro-batches.

⚠ **The loss denominator is the GROUP's, not the micro-batch's.** Each
accumulation group is sampled FIRST, its total valid-timestep count summed,
and every micro-batch's `flow_mse` is given that total — so the accumulated
gradient is the mean over the whole group rather than a sum of per-micro-batch
means. Passing each micro-batch its own count would weight a chunk near an
episode boundary more heavily than a full one, by exactly the ratio of their
valid counts.

⚠ **The held-out groups are FIXED, noise and timestep included.** See
`VAL_SEED`. A validation curve is only readable if every pass scores the
identical problem, and for flow matching the problem includes its `t`.

⚠ **Validation runs the backward it does not need.** `SmolVLATrainStep.run`
does forward and backward together, so a validation pass costs about twice what
it should and leaves gradients that the next training step's
`zero_trainable_grads` discards. Correct, wasteful, and named here rather than
left to be discovered in a profile.

⚠ **A short run is not a low-learning-rate run.** The reference's scheduler
AUTO-SCALES: a run shorter than `num_decay_steps` compresses both the warmup
and the cosine decay to fit it. `SMOLVLA_STEPS=3` therefore gets a 1-step
warmup and runs at ~5e-05, essentially peak — not the gentle ramp a 3-step
"smoke test" sounds like. The smoke run is for checking that it STARTS and
what a step costs; the loss it prints is a full schedule crammed into three
steps and means nothing.

### Checkpoints

`runs/<id>/checkpoints/best.ckpt` (project `so101`) is written whenever the
held-out loss improves and `last.ckpt` at every validation, so a killed run
loses at most `VAL_EVERY` steps and never the best model. Both are uploaded
through the run's artifact sink when `.env` names a monitor; `SMOLVLA_CKPT`
overrides the location with `<prefix>_best.ckpt` / `<prefix>_last.ckpt`.

⚠ **Only the TRAINABLE set is saved**: the expert, the four action
projections, and Adam's moments for them. The SigLIP tower, the sixteen VLM
layers, the connector and the token embedding are frozen and already on disk
as `lerobot/smolvla_base`. Saving them again would triple the file and create
a second copy that could silently disagree with the base it was fine-tuned
from — so `SMOLVLA_INIT` loads the base FIRST and this on top.

⚠ The moments ride along (`save_moments=True`), which makes a resume EXACT
rather than a restart with a cold optimizer — and a cold optimizer's first
step is precisely what damages a pretrained model (see `VAL_SEED` and the
warmup note). ~98 M parameters is 393 MB of weights, ~1.2 GB with moments.

### The vision cache

Under `train_state_proj = False` everything a FRAME contributes to the prefix
— SigLIP over two 512x512 images, the shuffle, the connector, the sqrt(960)
— is a constant of that frame, and it was recomputed on every visit. Measured
with `SMOLVLA_PROFILE` on a 5090: host images 44.8 ms + prefix 26.0 ms of a
125 ms observation. So the image segment is cached: 491 KB per row, 9.5 GB
for the recording, in `<store>.vision.bin` beside the store
(`SMOLVLA_VISION_CACHE` overrides the path; `off` disables). The first run
builds it — every row through the tower once, resumable if killed — and
every observation after that uploads a row and runs the tower from the
language tokens on. See `vision_cache.mojo`.

⚠ **Exact.** The rows are the fp32 numbers `build_prefix` produces, and
`test_vision_cache.mojo` asserts the prefix and the prefill output are
bit-identical through either door. On top of that, this file recomputes two
rows at startup and refuses a cache whose rows differ — a cache built from a
different store, or with different frozen weights, is a plausible policy
trained on the wrong pictures, and nothing downstream could tell.

⚠ The 26 ms "prefix" was the whole tower; what remains per observation is
the sixteen VLM layers over 140 tokens, which the profile line now shows on
its own.
"""

from std.math import cos, pi
from std.os import getenv
from std.os.path import exists
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.ptr import mptr
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.checkpoint import CheckpointScalars
from noeira.nn.core.initializer import Deterministic
from noeira.nn.optimizer.adam import Adam
from noeira.nn.primitives.linear import Linear
from noeira.core.run import RunContext, register_run
from noeira.core.project import project_exists
from noeira.core.run_session import finish_run, run_logger
from noeira.io.artifact_sink import ArtifactSink, sink_for_run
from noeira.deep_agents.training.checkpoint import announce_checkpoint
from noeira.io.hf import hf_download_file, HF_MODEL, HF_DATASET
from noeira.io.hdf5 import H5Dataset

from noeira.deep_agents.smolvla.policy import SmolVLAPolicy
from noeira.deep_agents.smolvla.recording import SO101_N_LANG, SO101_TASKS
from noeira.deep_agents.smolvla.normalize import SmolVLAStats
from noeira.deep_agents.smolvla.tasks import TaskTokens
from noeira.deep_agents.smolvla.dataset import SmolVLABatchSampler
from noeira.deep_agents.smolvla.observation import fill_store_images
from noeira.deep_agents.smolvla.train_step import SmolVLATrainStep
from noeira.deep_agents.smolvla.vision_cache import VisionCache
from noeira.deep_agents.smolvla.finetune import (
    zero_trainable_grads, adam_step_trainables, save_trainables,
    load_trainables, adopt_trainables, clip_trainables,
)
from noeira.deep_agents.smolvla.flow_loss import (
    build_xt_ut, sample_noise, sample_times,
)
from noeira.deep_agents.smolvla.heads import (
    SMOLVLA_ACTION_DIM, SMOLVLA_EXPERT_W, SMOLVLA_STATE_DIM,
)
from noeira.deep_agents.smolvla.text import (
    SMOLLM_DIM, SMOLLM_LAYERS, SMOLLM_KV_W,
)
from noeira.deep_agents.smolvla.expert import EXPERT_FF
from noeira.deep_agents.smolvla.vision import SIGLIP_LAYERS

# ── the recording ────────────────────────────────────────────────────────
comptime SDIM = 6                # the SO-101's joints
comptime ADIM_REAL = 6
comptime N_CAM = 2
comptime SRC_H = 480
comptime SRC_W = 640
comptime N_LANG = SO101_N_LANG
"""⚠ Pinned by the instruction table, not chosen — and read from
`deep_agents/smolvla/recording.mojo`, which the deployment reads too. `test_tasks.mojo` refuses a
table whose tasks tokenise to different lengths, because a comptime N_LANG
cannot hold two — a multi-task fine-tune needs a padding decision first."""

comptime CHUNK = 50
comptime STEPS_EULER = 10        # inference only; training denoises ONCE
comptime B = 8
"""⚠ Observations per FORWARD — the batch the expert, the prefill and the
backward run over at once. A comptime, so changing it is a rebuild.

Why not 1: at B = 1 the step is LAUNCH-BOUND. With the vision cache and the
element-indexed attention backward, a 5090 spent 33 ms per observation on
~1 500 kernel launches of a few microseconds' work each — prefill 9.9 ms,
suffix forward 6.2, backward 16.8, of which 112 `Linear.vjp` calls at ~85 us.
The same launches serve B observations when they are batched, which is what
the reference does at 64. `test_train_step_batched.mojo` gates that B rows
reproduce B separate runs.

Why not 64: the tape. The recording driver keeps every layer's activations
for the backward, and that scales with B — about 320 MB at B = 1, so ~2.6 GB
at 8 and ~20 GB at 64 on top of the 3.2 GB policy. 8 fits a 5090 with room;
raise it once the number below says the launches still dominate."""
comptime PAD = SMOLVLA_ACTION_DIM

comptime REPO = String("lerobot/smolvla_base")
comptime DEFAULT_TASKS = String(SO101_TASKS)
comptime DEFAULT_DATA_REPO = String("DenisLabs/record-test_20260828_092736")

comptime DEFAULT_STEPS = 2000
"""⚠ 2000 x 64 is ~128 k observations, about seven passes over the recording.
The reference recipe is 20 000 x 64. Raise it with `SMOLVLA_STEPS` once a
step time is known."""
comptime DEFAULT_ACCUM = 64
"""Observations per OPTIMIZER step — the reference's `batch_size`. Must be a
multiple of `B`; `accum // B` forwards of `B` accumulate into one update."""
comptime PEAK_LR = 1.0e-4
comptime DECAY_LR = 2.5e-6
comptime WARMUP_STEPS = 1000
comptime DECAY_STEPS = 30000
comptime BETA1 = Scalar[DT](0.9)
comptime BETA2 = Scalar[DT](0.95)
comptime EPS = Scalar[DT](1.0e-8)
comptime WD = Scalar[DT](1.0e-10)
comptime CLIP_NORM = Scalar[DT](10.0)
"""⚠ Every one of these is `configuration_smolvla.py`'s, not a default.

`beta2` is **0.95**, not Adam's usual 0.999 — a much shorter second-moment
memory. And `scheduler_warmup_steps` is 1,000.

The first run of this file had neither, and it showed: base training loss
0.570, held-out 2.765 after ONE optimizer step, 1.055 after three. Adam's
first step moves EVERY parameter by about ±lr regardless of gradient
magnitude — `m/sqrt(v)` is ±1 when the moments are fresh — so 100 M
parameters of a pretrained checkpoint all shift at once and the step damages
the model before it improves it. That is what a warmup is for, and skipping it
does not fail, it just wastes the checkpoint you started from."""
comptime VAL_EVERY = 200
comptime VAL_GROUPS = 8
comptime LOG_EVERY = 10
comptime VAL_SEED: UInt64 = 0x5DEECE66D
"""⚠ The seed the held-out groups are drawn with, ONCE, before training.

An observation here is (frame, instruction, pose, chunk, noise, t) and the
loss depends strongly on t, so pinning only the ROW sampler is not enough —
the noise and the timestep come from the global RNG and would be redrawn every
pass. The first run of this file did exactly that and printed 2.034 -> 1.008
across two Adam steps at lr 1e-4: a redraw, not learning, and unfalsifiable
either way. The groups are now built once and reused verbatim."""

comptime Pol = SmolVLAPolicy[
    N_CAM, N_LANG, CHUNK, STEPS_EULER, B, SMOLLM_LAYERS, SIGLIP_LAYERS, True
]
comptime Step = SmolVLATrainStep[
    CHUNK, ADIM_REAL, PAD, SMOLVLA_EXPERT_W, B, SMOLLM_LAYERS, EXPERT_FF,
    SMOLLM_DIM,
]
comptime Sampler = SmolVLABatchSampler[SDIM, ADIM_REAL, PAD, CHUNK, B]
comptime IMG_ELEMS = N_CAM * 3 * SRC_H * SRC_W
comptime AN = B * CHUNK * PAD
comptime VSEG = Pol.Prefix.IMG_SEG
"""One vision-cache row: N_CAM x 64 tokens x 960, fp32 — 491 520 bytes."""
comptime VC_FLUSH_EVERY = 200
comptime VC_PRINT_EVERY = 1000


def lr_at(step: Int, total: Int) -> Float64:
    """`CosineDecayWithWarmupSchedulerConfig`, transcribed.

    ⚠ Including its AUTO-SCALING: when a run is shorter than `DECAY_STEPS` the
    reference rescales both the warmup and the decay to fit, so a 2,000-step
    run gets a 66-step warmup rather than never leaving it. A transcription
    that dropped that would spend a short run entirely in the ramp and report
    that fine-tuning does not work.
    """
    var warm = WARMUP_STEPS
    var dec = DECAY_STEPS
    if total < DECAY_STEPS:
        var scale = Float64(total) / Float64(DECAY_STEPS)
        warm = Int(Float64(WARMUP_STEPS) * scale)
        dec = total
    if warm < 1:
        warm = 1
    if step < warm:
        # linear from peak/(warm+1) up to peak
        var frac = 1.0 - Float64(step) / Float64(warm)
        var mult = (1.0 / Float64(warm + 1) - 1.0) * frac + 1.0
        return PEAK_LR * mult
    var st = step if step < dec else dec
    var cd = 0.5 * (1.0 + cos(pi * Float64(st) / Float64(dec)))
    var alpha = DECAY_LR / PEAK_LR
    return PEAK_LR * ((1.0 - alpha) * cd + alpha)


def _need(name: String) raises -> String:
    var v = getenv(name)
    if v.byte_length() == 0:
        raise Error(
            "$" + name + " is not set. This example refuses a default: a"
            " fine-tune attributed to the wrong recording or the wrong"
            " statistics is worse than one that will not start. See the"
            " header."
        )
    return v^


struct Phases(Movable):
    """Where an observation's wall time goes, summed over `n` observations
    (`B` per `run_one`; every number `report` prints is per OBSERVATION, so
    the batch's amortisation shows directly).

    `img` is host work (decode + resize + upload, ends synchronised); `step`
    is everything after it, prefix through loss. `prefix` is only filled
    under `SMOLVLA_PROFILE`, when `run_one` drains after `build_prefix`; the
    suffix forward and backward come from `SmolVLATrainStep.ns_fwd/ns_bwd`,
    filled under the same flag.
    """

    var img: Int
    var step: Int
    var prefix: Int
    var n: Int

    def __init__(out self):
        self.img = 0
        self.step = 0
        self.prefix = 0
        self.n = 0

    def __init__(out self, *, deinit move: Self):
        self.img = move.img
        self.step = move.step
        self.prefix = move.prefix
        self.n = move.n

    def report(self, profile: Bool, ref st: Step, ref pol: Pol) -> String:
        var tot = Float64(self.img + self.step)
        if tot <= 0.0 or self.n == 0:
            return String("")
        if not profile:
            return (
                "host-images " + String(100.0 * Float64(self.img) / tot)
                + "%  gpu-step " + String(100.0 * Float64(self.step) / tot)
                + "%"
            )
        # Per observation, in ms. `other` is the step's remainder: the
        # action/noise/time uploads, `build_xt_ut`, `set_times`, and the
        # loss download — small, and named so a surprise there is visible.
        var k = 1.0e-6 / Float64(self.n)
        var other = self.step - self.prefix - st.ns_fwd - st.ns_bwd
        var s = (
            "per obs: images/cache-read " + String(Float64(self.img) * k)
            + " ms, prefix " + String(Float64(self.prefix) * k)
            + " ms, suffix fwd " + String(Float64(st.ns_fwd) * k)
            + " ms, backward " + String(Float64(st.ns_bwd) * k)
            + " ms, other " + String(Float64(other) * k)
            + " ms  (total " + String(tot * k) + " ms)"
        )
        # The backward by op class — the expert's own stage timers. Their
        # sum is the backward less the heads' vjps and the drains.
        ref pr = pol.denoiser.prof
        comptime D = Pol.Den
        s += (
            "\n           backward by op: glue " + String(Float64(pr[D.PR_GLUE]) * k)
            + ", mlp.down " + String(Float64(pr[D.PR_MLP_DOWN]) * k)
            + ", swiglu " + String(Float64(pr[D.PR_GLU]) * k)
            + ", mlp.up+gate " + String(Float64(pr[D.PR_MLP_UPGATE]) * k)
            + ", norms " + String(Float64(pr[D.PR_NORM]) * k)
            + ", o " + String(Float64(pr[D.PR_O]) * k)
            + ", attention " + String(Float64(pr[D.PR_ATTN]) * k)
            + ", kv-repeat " + String(Float64(pr[D.PR_REP]) * k)
            + ", rope " + String(Float64(pr[D.PR_ROPE]) * k)
            + ", q/k/v " + String(Float64(pr[D.PR_QKV]) * k) + " ms"
        )
        return s


struct Group(Movable):
    """One accumulation group, drawn BEFORE any forward runs.

    ⚠ It carries its own NOISE and TIMESTEP, not just its rows. A
    flow-matching observation is (frame, instruction, pose, chunk, noise, t)
    and the loss depends strongly on t — so a validation pass that redrew
    them would score the same frames as a different problem. The first run of
    this file did exactly that and reported 2.034 -> 1.008 across two Adam
    steps at lr 1e-4, which is a redraw and cannot be learning.

    ⚠ The whole group is sampled first so `total_valid` can be its sum. Each
    micro-batch's `flow_mse` is then given that total, which makes the
    accumulated gradient the mean over the group. Giving each micro-batch its
    own count instead would weight a chunk near an episode boundary more
    heavily than a full one, by the ratio of their valid counts — a silent,
    data-dependent reweighting of the loss.
    """

    var rows: List[Int]
    var tasks: List[Int]
    var raw_state: List[Float32]
    var actions: List[Scalar[DT]]
    var valid: List[Scalar[DT]]
    var noise: List[Scalar[DT]]
    var times: List[Float64]
    var total_valid: Int

    def __init__(out self):
        self.rows = List[Int]()
        self.tasks = List[Int]()
        self.raw_state = List[Float32]()
        self.actions = List[Scalar[DT]]()
        self.valid = List[Scalar[DT]]()
        self.noise = List[Scalar[DT]]()
        self.times = List[Float64]()
        self.total_valid = 0

    def __init__(out self, *, deinit move: Self):
        self.rows = move.rows^
        self.tasks = move.tasks^
        self.raw_state = move.raw_state^
        self.actions = move.actions^
        self.valid = move.valid^
        self.noise = move.noise^
        self.times = move.times^
        self.total_valid = move.total_valid


def draw_group(
    mut sam: Sampler, micro: Int, lo: Int, hi: Int,
    mut state_t: Tensor, mut acts_t: Tensor, mut valid_t: Tensor,
) raises -> Group:
    """`micro` forwards of `B` observations each — `accum` observations."""
    var gr = Group()
    for _ in range(micro):
        var tk = List[Int]()
        var rw = List[Int]()
        var rs = List[Float32]()
        var nv = sam.sample(state_t, acts_t, valid_t, tk, rw, rs, lo, hi)
        gr.total_valid += nv
        for b in range(B):
            gr.rows.append(rw[b])
            gr.tasks.append(tk[b])
            for j in range(SDIM):
                gr.raw_state.append(rs[b * SDIM + j])
        for i in range(AN):
            gr.actions.append(acts_t.data[i])
        for i in range(B * CHUNK):
            gr.valid.append(valid_t.data[i])
        var nz = Tensor.alloc(AN)
        sample_noise(nz, AN)
        for i in range(AN):
            gr.noise.append(nz.data[i])
        var tl = sample_times(B)
        for b in range(B):
            gr.times.append(tl[b])
    # ⚠ The bound `flow_mse` used to carry, at the only place that knows
    # the group's size. A total above it means a micro-batch reported more
    # valid timesteps than it has slots; below 1 means every timestep in the
    # group is padding and the loss has no terms.
    if gr.total_valid <= 0 or gr.total_valid > micro * B * CHUNK:
        raise Error(
            "draw_group: total_valid " + String(gr.total_valid)
            + " is outside (0, " + String(micro * B * CHUNK) + "]"
        )
    return gr^


def main() raises:
    print("=" * 74)
    print("SmolVLA fine-tune — SO-ARM101")
    print("=" * 74)

    var store_path = _need(String("SMOLVLA_STORE"))
    # ⚠ Fetched from the dataset repo unless overridden, so the statistics
    # and the store cannot come from different snapshots by accident — which
    # would be a scale error with no symptom but a worse policy.
    var stats_path = getenv("SMOLVLA_STATS")
    if stats_path.byte_length() == 0:
        var data_repo = getenv("SMOLVLA_REPO")
        if data_repo.byte_length() == 0:
            # ⚠⚠ THE FALLBACK'S PROMISE ONLY HOLDS FOR ITS OWN RECORDING.
            # Fetching `meta/stats.json` from the default repo guarantees the
            # statistics and the store agree ONLY when the store WAS built from
            # that repo. For any other store — a project recording, above all —
            # it silently normalises one dataset with another's scale: every
            # joint shifted and rescaled, no error, a worse policy. So a store
            # that is not the default recording must NAME its statistics.
            var slug = DEFAULT_DATA_REPO.replace("/", "__")
            if slug not in store_path:
                raise Error(
                    "smolvla finetune: " + store_path + " is not "
                    + DEFAULT_DATA_REPO + ", so its statistics cannot be"
                    " fetched from there.\n  Set SMOLVLA_STATS to this"
                    " recording's stats.json — for a project recording,"
                    " tools/vla/smolvla_stats_from_act_norm.py builds one"
                    " from the ACT run's exact norm.json."
                )
            data_repo = DEFAULT_DATA_REPO
        stats_path = hf_download_file(
            data_repo, String("meta/stats.json"), HF_DATASET
        )
    var tasks_path = getenv("SMOLVLA_TASKS")
    if tasks_path.byte_length() == 0:
        tasks_path = DEFAULT_TASKS

    var steps = DEFAULT_STEPS
    var e_steps = getenv("SMOLVLA_STEPS")
    if e_steps.byte_length() > 0:
        steps = Int(e_steps)
    var accum = DEFAULT_ACCUM
    var e_accum = getenv("SMOLVLA_ACCUM")
    if e_accum.byte_length() > 0:
        accum = Int(e_accum)
    if accum <= 0 or accum % B != 0:
        raise Error(
            "SMOLVLA_ACCUM=" + String(accum) + " must be a positive multiple"
            " of this build's B = " + String(B)
            + " (observations per forward)"
        )
    var micro = accum // B
    var e_val = getenv("SMOLVLA_VAL_EPISODES")
    var profile = getenv("SMOLVLA_PROFILE").byte_length() > 0

    var ctx = DeviceContext()
    print("  device  " + String(ctx.name()))
    print("  store   " + store_path)
    print("  stats   " + stats_path)
    print("  steps   " + String(steps) + " x " + String(accum)
          + " observations (" + String(micro) + " x B=" + String(B)
          + ")   peak lr " + String(PEAK_LR) + ", warmup "
          + String(WARMUP_STEPS) + " (auto-scaled), beta2 " + String(BETA2))

    var tasks = TaskTokens(tasks_path)
    var n_lang = tasks.n_lang()
    if n_lang != N_LANG:
        raise Error(
            "the instruction table tokenises to " + String(n_lang)
            + " tokens and this build is pinned to " + String(N_LANG)
            + " — rebuild with N_LANG = " + String(n_lang)
        )
    print("  tasks   " + String(tasks.size()) + " instruction(s), "
          + String(n_lang) + " tokens   P = " + String(Pol.P))

    print("  loading lerobot/smolvla_base ...")
    var weights = hf_download_file(REPO, String("model.safetensors"), HF_MODEL)
    var pol = Pol.make["gpu", Deterministic](Optional(ctx))
    pol.load["gpu"](weights, Optional(ctx))
    pol.load_stats(stats_path)
    print("  policy  loaded")

    # `SMOLVLA_PROJECT` files the run under a project (`project-promote`
    # promotes into the RUN's project); unset keeps the historical `so101`.
    # An explicit name that is not on this box refuses, as `ACT_PROJECT` does.
    var project = getenv("SMOLVLA_PROJECT")
    if project.byte_length() == 0:
        project = String("so101")
    elif not project_exists(project):
        raise Error(
            "SMOLVLA_PROJECT=" + project + " but there is no projects/"
            + project + "/project.kv on this box — run `pixi run project-pull "
            + project + "` first"
        )
    var run = RunContext(
        project=project,
        driver=String("examples/so101/smolvla_so101_finetune.mojo"),
        slug=String("smolvla-so101"),
        env=String("builtin:so_arm101"),
        dataset=store_path,
        device=String(ctx.name()),
    )
    print("  run     " + run.dir)
    # The run's own directory unless `SMOLVLA_CKPT` names a prefix.
    var ckpt = getenv("SMOLVLA_CKPT")
    var last_ckpt = run.checkpoint_path(String("last"))
    var best_ckpt = run.checkpoint_path(String("best"))
    if ckpt.byte_length() > 0:
        last_ckpt = ckpt + "_last.ckpt"
        best_ckpt = ckpt + "_best.ckpt"

    var sam = Sampler(store_path, SmolVLAStats.from_stats_json(stats_path))
    var n_ep = sam.store.n_episodes()
    var n_rows = sam.n_rows()
    var n_val_ep = n_ep // 5
    if n_val_ep < 1:
        n_val_ep = 1
    if e_val.byte_length() > 0:
        n_val_ep = Int(e_val)
    if n_val_ep < 0 or n_val_ep >= n_ep:
        raise Error(
            "SMOLVLA_VAL_EPISODES=" + String(n_val_ep) + " but the store has "
            + String(n_ep) + " episodes — 0 trains on all of them, and at"
            " least one must remain to train on"
        )
    # ⚠ A CONTIGUOUS tail, not every fifth episode. Frames inside one episode
    # are near-duplicates of their neighbours, so an interleaved split puts a
    # training frame 33 ms from each held-out one and the held-out loss
    # measures memorisation instead of generalisation.
    var held_out = n_val_ep > 0
    var split = (
        sam.store.episodes.start_of(n_ep - n_val_ep) if held_out else n_rows
    )
    # ⚠ With nothing held out the curve is scored on TRAINING rows. It still
    # has to be finite and to fall — that much is a check on the plumbing —
    # but it says nothing about generalisation, and every print of it below
    # is labelled so it cannot be read as the other kind of number.
    var val_lo = split if held_out else 0
    var val_label = String("HELD-OUT") if held_out else String(
        "TRAIN-SUBSET (nothing held out)"
    )
    if held_out:
        print(
            "  data    " + String(n_rows) + " rows, " + String(n_ep)
            + " episodes — train [0, " + String(split) + "), held out ["
            + String(split) + ", " + String(n_rows) + ")  ("
            + String(n_val_ep) + " episodes)"
        )
    else:
        print(
            "  data    " + String(n_rows) + " rows, " + String(n_ep)
            + " episodes — ALL trained on; the curve below is a training"
            " subset, NOT held out (SMOLVLA_VAL_EPISODES=0)"
        )

    # `SMOLVLA_NO_MONITOR` points the logger at no `.env` at all, so its
    # remote half is inert; `metrics.csv` is still written.
    var no_mon = getenv("SMOLVLA_NO_MONITOR")
    var logger = run_logger(
        run,
        buffer_size=64,
        env_path=String("") if no_mon.byte_length() > 0 else String(".env"),
    )
    logger.set_config("algorithm", "SmolVLA")
    logger.set_config("robot", "SO-ARM101")
    logger.set_config("regime", "train_expert_only; state_proj FROZEN")
    logger.set_config("device", String(ctx.name()))
    logger.set_config("store", store_path)
    logger.set_config("chunk", String(CHUNK))
    logger.set_config("accum", String(accum))
    logger.set_config("peak_lr", String(PEAK_LR))
    logger.set_config("beta2", String(BETA2))
    logger.set_config("warmup", String(WARMUP_STEPS))
    register_run(run, logger)
    # ⚠ `SMOLVLA_NO_MONITOR` must silence the uploads too: `sink_for_run`
    # reads `.env` itself.
    var artifacts: Optional[ArtifactSink] = None
    if no_mon.byte_length() == 0:
        artifacts = sink_for_run(run.id, run.dir)

    var st = Step.make["gpu"](Optional(ctx))
    st.profile = profile
    # ⚠ The denoiser's own stage timers — the backward split by op class.
    # Same flag, same caveat: on, it drains at every stage boundary.
    pol.denoiser.profile = profile
    var opt = Adam(
        lr=Scalar[DT](lr_at(0, steps)), beta1=BETA1, beta2=BETA2, eps=EPS,
        wd=WD,
    )
    var sp_frozen = Linear[SMOLVLA_STATE_DIM, SMOLLM_DIM].make[
        "gpu", Deterministic
    ](Optional(ctx))
    """⚠ A stand-in for the walks' `state_proj` argument. TRAIN_STATE_PROJ is
    False so neither walk touches it, and the policy's real one is never handed
    to the optimizer. That is what "frozen" means here — not a flag inside the
    optimizer but an object it never sees."""

    # ⚠ AFTER the base checkpoint, never instead of it. This file carries the
    # trainable set only; applying it to a fresh policy would leave the vision
    # tower and the VLM at their initialiser and produce finite actions from a
    # random prefix, with nothing to say so.
    var init_from = getenv("SMOLVLA_INIT")
    if init_from.byte_length() > 0:
        var init_sc = load_trainables[
            "gpu", SMOLLM_LAYERS, SMOLVLA_EXPERT_W, EXPERT_FF, SMOLLM_DIM,
            SMOLLM_KV_W, PAD,
        ](
            init_from, pol.expert, pol.action_in, pol.time_mlp_in,
            pol.time_mlp_out, pol.action_out, sp_frozen, Optional(ctx),
        )
        # The moments came back with the weights; their step count must too,
        # or the bias correction restarts at step 1 against settled moments.
        opt.take_step_state(init_sc, "opt")
        print("  resumed from " + init_from + " (Adam t = " + String(opt.t) + ")")

    # ⚠ AFTER the base checkpoint and any resume, BEFORE the first step:
    # adopting rebinds every Param to a slice of one arena, so it must happen
    # once the values are final. It buys the grouped update AND the global
    # grad-norm clip, which cannot be done component-by-component.
    adopt_trainables[
        "gpu", SMOLLM_LAYERS, SMOLVLA_EXPERT_W, EXPERT_FF, SMOLLM_DIM,
        SMOLLM_KV_W, PAD,
    ](
        opt, pol.expert, pol.action_in, pol.time_mlp_in, pol.time_mlp_out,
        pol.action_out, sp_frozen, Optional(ctx),
    )
    print("  arena   " + String(opt.arena.total)
          + " trainable elements, grouped update + global clip at "
          + String(CLIP_NORM))

    var images = Tensor.alloc(N_CAM * 3 * 512 * 512)
    var scratch = List[Float32]()
    var state_t = Tensor.alloc(B * PAD)
    var acts_t = Tensor.alloc(AN)
    var valid_t = Tensor.alloc(B * CHUNK)
    var noise_t = Tensor.alloc(AN)
    var times_t = Tensor.alloc(B)
    var x_t = Tensor.alloc(AN)
    var u_t = Tensor.alloc(AN)
    var row = List[Scalar[DType.uint8]](unsafe_uninit_length=IMG_ELEMS)
    var img_col = sam.store.open_column[DType.uint8](String("images"))

    # ── the vision cache: the frozen half of every observation, once ────
    var vc_path = getenv("SMOLVLA_VISION_CACHE")
    var vcache = VisionCache()
    if vc_path != "off" and vc_path != "0":
        if vc_path.byte_length() == 0:
            vc_path = store_path + ".vision.bin"
        vcache = open_or_build_vision_cache(
            vc_path, n_rows, pol, img_col, row, images, scratch, ctx
        )
    else:
        if B != 1:
            raise Error(
                "SMOLVLA_VISION_CACHE=off needs a B = 1 build: the image path"
                " runs the tower on one frame set. This build is B = "
                + String(B) + "."
            )
        print("  vcache  OFF — the tower runs on every observation")
    var seg = List[Float32]()

    # ⚠ The held-out groups are drawn ONCE, here, and reused by every
    # validation — rows, chunks, NOISE and TIMESTEPS all fixed. Pinning only
    # the row sampler leaves the noise and t coming from the global RNG, and
    # then the curve moves because the problem changed rather than because
    # the policy did.
    var keep_rng = sam.rng
    sam.rng = VAL_SEED
    var vgroups = List[Group]()
    for _ in range(VAL_GROUPS):
        vgroups.append(
            draw_group(sam, micro, val_lo, n_rows, state_t, acts_t, valid_t)
        )
    sam.rng = keep_rng
    print("  val     " + String(VAL_GROUPS) + " fixed groups x "
          + String(accum) + " observations, drawn once from rows ["
          + String(val_lo) + ", " + String(n_rows) + ")  " + val_label)

    # ── the BASELINE, before a single update ─────────────────────────────
    # ⚠ Without this every held-out number is post-update and "the loss fell"
    # has nothing to fall FROM. It is also the only number that says anything
    # about the published checkpoint on this recording, which is the thing a
    # fine-tune has to beat.
    var ph = Phases()
    var base_sum = 0.0
    for vi in range(VAL_GROUPS):
        for m in range(micro):
            base_sum += run_one(
                m, vgroups[vi], sam, tasks, pol, st, img_col, row, images,
                scratch, acts_t, valid_t, noise_t, times_t, x_t, u_t, ctx,
                ph, vcache, seg,
            )
    var base_val = base_sum / Float64(VAL_GROUPS)
    var best_val = base_val
    print("  BASELINE " + val_label + " (lerobot/smolvla_base, 0 updates): "
          + String(base_val))
    var bn = List[String]()
    var bv = List[Float64]()
    bn.append(String("val_loss"))
    bv.append(base_val)
    logger.log_scalars(bn, bv, -1)
    # ⚠ The baseline ran a BACKWARD it does not need and left gradients in
    # every trainable `.grd`. The loop's first `zero_trainable_grads` clears
    # them — but only because it is the first statement in the loop. Moving
    # the zero after the forward would fold the baseline's gradient into
    # step 0's update.

    var t0 = perf_counter_ns()

    for s in range(steps):
        # ⚠ Host-side, because `Adam.attach_warmup_schedule` is a NO-OP off
        # the GPU-arena path — "a no-op stub for CPU/non-adopted (use host
        # set_lr + a host schedule there)". Calling it here would have looked
        # like a schedule and done nothing.
        opt.set_lr(Scalar[DT](lr_at(s, steps)))
        zero_trainable_grads[
            "gpu", SMOLLM_LAYERS, SMOLVLA_EXPERT_W, EXPERT_FF, SMOLLM_DIM,
            SMOLLM_KV_W, PAD,
        ](
            opt, pol.expert, pol.action_in, pol.time_mlp_in,
            pol.time_mlp_out, pol.action_out, sp_frozen, Optional(ctx),
        )
        var gr = draw_group(sam, micro, 0, split, state_t, acts_t, valid_t)
        var loss = 0.0
        for m in range(micro):
            loss += run_one(
                m, gr, sam, tasks, pol, st, img_col, row, images, scratch,
                acts_t, valid_t, noise_t, times_t, x_t, u_t, ctx, ph,
                vcache, seg,
            )
        # ⚠ Clip BEFORE the step, over the JOINT norm of the whole trainable
        # set — `optimizer_grad_clip_norm = 10` in the reference. Clipping the
        # five components separately would let a joint norm of sqrt(5)x the
        # limit through.
        var gnorm = clip_trainables["gpu"](opt, CLIP_NORM, Optional(ctx))
        adam_step_trainables[
            "gpu", SMOLLM_LAYERS, SMOLVLA_EXPERT_W, EXPERT_FF, SMOLLM_DIM,
            SMOLLM_KV_W, PAD,
        ](
            opt, pol.expert, pol.action_in, pol.time_mlp_in,
            pol.time_mlp_out, pol.action_out, sp_frozen, Optional(ctx),
        )

        if s % LOG_EVERY == 0:
            var el = Float64(perf_counter_ns() - t0) / 1.0e9
            print(
                "  step " + String(s) + "   train " + String(loss)
                + "   " + String(el / Float64(s + 1)) + " s/step"
                + "   lr " + String(opt.get_lr())
                + "   |g| " + String(gnorm)
                + "   " + ph.report(profile, st, pol)
            )
            var names = List[String]()
            var vals = List[Float64]()
            names.append(String("loss"))
            vals.append(loss)
            logger.log_scalars(names, vals, s)

        if s % VAL_EVERY == 0 or s == steps - 1:
            # ⚠ The seed is PINNED, so every validation scores the identical
            # held-out observations. It is restored afterwards so training
            # does not replay the same batches for ever.
            var vsum = 0.0
            for vi in range(VAL_GROUPS):
                for m in range(micro):
                    vsum += run_one(
                        m, vgroups[vi], sam, tasks, pol, st, img_col, row,
                        images, scratch, acts_t, valid_t, noise_t, times_t,
                        x_t, u_t, ctx, ph, vcache, seg,
                    )
            var vloss = vsum / Float64(VAL_GROUPS)
            print(
                "  step " + String(s) + "   " + val_label + " " + String(vloss)
                + "   vs baseline " + String(base_val) + "  ("
                + String(100.0 * (vloss - base_val) / base_val) + "%)"
            )
            var vn = List[String]()
            var vv = List[Float64]()
            vn.append(String("val_loss"))
            vv.append(vloss)
            logger.log_scalars(vn, vv, s)

            # ⚠ `last` every time, `best` only on an improvement. A kill then
            # loses at most VAL_EVERY steps and never the best model — which
            # matters here because the curve PLATEAUS and then drifts up, so
            # the final weights are not the ones worth keeping.
            var opt_sc = CheckpointScalars()
            opt.put_step_state(opt_sc, "opt")
            save_trainables[
                "gpu", SMOLLM_LAYERS, SMOLVLA_EXPERT_W, EXPERT_FF, SMOLLM_DIM,
                SMOLLM_KV_W, PAD,
            ](
                last_ckpt, pol.expert, pol.action_in,
                pol.time_mlp_in, pol.time_mlp_out, pol.action_out, sp_frozen,
                True, Optional(ctx), scalars=opt_sc,
            )
            announce_checkpoint(last_ckpt, artifacts, run.dir)
            if vloss < best_val:
                best_val = vloss
                save_trainables[
                    "gpu", SMOLLM_LAYERS, SMOLVLA_EXPERT_W, EXPERT_FF,
                    SMOLLM_DIM, SMOLLM_KV_W, PAD,
                ](
                    best_ckpt, pol.expert, pol.action_in,
                    pol.time_mlp_in, pol.time_mlp_out, pol.action_out,
                    sp_frozen, True, Optional(ctx), scalars=opt_sc,
                )
                announce_checkpoint(best_ckpt, artifacts, run.dir)
                print("      saved " + best_ckpt)
            # ⚠ Validation ran `run_one`, which does a BACKWARD it does not
            # need — `SmolVLATrainStep.run` does both. The gradients it leaves
            # are discarded by the next step's `zero_trainable_grads`, above.
            # Correct, and about twice the cost it should be.

    finish_run(
        run, logger, artifacts,
        String("best_val_loss=") + String(best_val)
        + " base_val_loss=" + String(base_val)
        + " held_out=" + String(held_out),
    )
    print("")
    print("  best " + val_label + " " + String(best_val) + "  vs baseline "
          + String(base_val) + "  ("
          + String(100.0 * (best_val - base_val) / base_val) + "%)")
    print("  weights: " + best_ckpt + "  /  " + last_ckpt)
    print("  run:     " + run.kv_path())
    if not held_out:
        print("  ⚠ nothing was held out: `best` was picked by a TRAINING"
              " subset. Deploy last.ckpt, at the step count the held-out"
              " run chose.")
    print("done")


def fresh_segment(
    g: Int,
    mut pol: Pol,
    mut img_col: H5Dataset,
    mut row: List[Scalar[DType.uint8]],
    mut images: Tensor,
    mut scratch: List[Float32],
    mut seg: List[Float32],
    ctx: DeviceContext,
) raises:
    """Row `g` through the tower: what a vision-cache row IS. Used to build
    the cache and, at startup, to check two of its rows against the store."""
    img_col.read_range[DType.uint8](g, g + 1, mptr(row))
    fill_store_images["gpu", N_CAM](
        row, SRC_W, SRC_H, images, scratch, Optional(ctx)
    )
    pol.embed_images["gpu"](images, Optional(ctx))
    pol.image_segment["gpu"](seg, Optional(ctx))


def open_or_build_vision_cache(
    var path: String,
    n_rows: Int,
    mut pol: Pol,
    mut img_col: H5Dataset,
    mut row: List[Scalar[DType.uint8]],
    mut images: Tensor,
    mut scratch: List[Float32],
    ctx: DeviceContext,
) raises -> VisionCache:
    """Open `path`, building or resuming it first if it is not complete, then
    check two rows against a fresh forward — bit for bit.

    ⚠ The check is what makes an existing file trustworthy. The header pins
    only the row count and the segment width, and two stores of the same
    recording length have the same header. Rows 0 and n/2 through the tower
    cost a quarter of a second and rule out a cache built from another store,
    with other frozen weights, or by a build that was interrupted and then
    edited — every one of which trains a policy on the wrong pictures without
    an error anywhere else.
    """
    var vc: VisionCache
    if exists(path):
        vc = VisionCache.open(path, n_rows, VSEG)
        if vc.complete():
            print("  vcache  " + vc.path + "  (" + String(n_rows)
                  + " rows, complete)")
        else:
            print("  vcache  " + vc.path + "  resuming at row "
                  + String(vc.rows_done) + " of " + String(n_rows))
    else:
        vc = VisionCache.create(path, n_rows, VSEG)
        print("  vcache  " + vc.path + "  building " + String(n_rows)
              + " rows x " + String(VSEG * 4) + " bytes = "
              + String(Float64(n_rows) * Float64(VSEG) * 4.0 / 1.0e9)
              + " GB")

    var seg = List[Float32]()
    if not vc.complete():
        var t0 = perf_counter_ns()
        var start = vc.rows_done
        for g in range(start, n_rows):
            fresh_segment(g, pol, img_col, row, images, scratch, seg, ctx)
            vc.write_row(g, seg)
            if (g + 1) % VC_FLUSH_EVERY == 0:
                vc.flush_progress()
            if (g + 1) % VC_PRINT_EVERY == 0 or g + 1 == n_rows:
                var el = Float64(perf_counter_ns() - t0) / 1.0e9
                var done = g + 1 - start
                var rate = Float64(done) / el
                print(
                    "          row " + String(g + 1) + " / " + String(n_rows)
                    + "   " + String(rate) + " rows/s   ETA "
                    + String(Float64(n_rows - g - 1) / rate / 60.0) + " min"
                )
        vc.flush_progress()
        print("  vcache  built in "
              + String(Float64(perf_counter_ns() - t0) / 60.0e9) + " min")

    # ── two rows against the store, bit for bit ──────────────────────────
    var probe = List[Int]()
    probe.append(0)
    probe.append(n_rows // 2)
    var cached = List[Float32]()
    for k in range(len(probe)):
        var g = probe[k]
        fresh_segment(g, pol, img_col, row, images, scratch, seg, ctx)
        vc.read_row(g, cached)
        var diff = 0
        for i in range(VSEG):
            if seg[i] != cached[i]:
                diff += 1
        if diff != 0:
            raise Error(
                "vision cache " + vc.path + ": row " + String(g) + " differs"
                " from a fresh forward in " + String(diff) + " of "
                + String(VSEG) + " floats — it was built from another store"
                " or with other frozen weights. Delete it (or set"
                " SMOLVLA_VISION_CACHE=off) and rerun."
            )
    print("  vcache  rows 0 and " + String(n_rows // 2)
          + " match a fresh forward bit for bit")
    return vc^


def run_one(
    m: Int,
    ref gr: Group,
    mut sam: Sampler,
    ref tasks: TaskTokens,
    mut pol: Pol,
    mut st: Step,
    mut img_col: H5Dataset,
    mut row: List[Scalar[DType.uint8]],
    mut images: Tensor,
    mut scratch: List[Float32],
    mut acts_t: Tensor,
    mut valid_t: Tensor,
    mut noise_t: Tensor,
    mut times_t: Tensor,
    mut x_t: Tensor,
    mut u_t: Tensor,
    ctx: DeviceContext,
    mut ph: Phases,
    mut vcache: VisionCache,
    mut seg: List[Float32],
) raises -> Float64:
    """One observation: its prefix, its interpolant, one denoising step.

    ⚠ The timers split HOST from DEVICE, and they can only be read that way
    because each phase ENDS in a synchronisation: `fill_store_images`
    finishes with `upload_resident`, and `run` finishes with `mean_err`, which
    downloads. Subtracting host timers across a run of pure enqueues would
    measure the enqueues (`_a_per_call_sweep_is_an_upper_bound_on_a_step`).
    Under `SMOLVLA_PROFILE` the prefix gets its own drain, and `run` two more,
    so the GPU time splits into prefix / suffix forward / backward.

    With the vision cache the "images" phase is a 491 KB file read and the
    "prefix" phase is the row's upload plus the VLM prefill — the tower is
    not run at all.
    """
    var t_img = perf_counter_ns()
    # ⚠ One instruction per ROW, B*N_LANG ids — the rows of a batch may come
    # from different tasks once a recording has more than one. `run_tail`
    # gathers per row when it is handed B instructions.
    var lang = List[Int]()
    var rs = List[Float32]()
    for b in range(B):
        var ids = tasks.for_index(gr.tasks[m * B + b])
        for t in range(len(ids)):
            lang.append(ids[t])
        for j in range(SDIM):
            rs.append(gr.raw_state[(m * B + b) * SDIM + j])
    var t_step = t_img
    if vcache.active:
        for b in range(B):
            vcache.read_row(gr.rows[m * B + b], seg, b * VSEG)
        ph.img += Int(perf_counter_ns() - t_img)
        t_step = perf_counter_ns()
        pol.build_prefix_from_segment["gpu"](seg, lang, rs, Optional(ctx))
    else:
        var g = gr.rows[m * B]
        img_col.read_range[DType.uint8](g, g + 1, mptr(row))
        fill_store_images["gpu", N_CAM](
            row, SRC_W, SRC_H, images, scratch, Optional(ctx)
        )
        ph.img += Int(perf_counter_ns() - t_img)
        t_step = perf_counter_ns()
        pol.build_prefix["gpu"](images, lang, rs, Optional(ctx))
    if st.profile:
        # ⚠ The drain that makes `prefix` a wall time. Without it this timer
        # would end at the last enqueue, and `run`'s first drain would be
        # charged with the whole SigLIP tower and sixteen VLM layers.
        ctx.synchronize()
        ph.prefix += Int(perf_counter_ns() - t_step)

    for i in range(AN):
        acts_t.data[i] = gr.actions[m * AN + i]
    for i in range(B * CHUNK):
        valid_t.data[i] = gr.valid[m * B * CHUNK + i]
    acts_t.upload_resident(ctx)
    valid_t.upload_resident(ctx)

    noise_t.ensure(AN)
    for i in range(AN):
        noise_t.data[i] = gr.noise[m * AN + i]
    noise_t.upload_resident(ctx)
    var tl = List[Float64]()
    times_t.ensure(B)
    for b in range(B):
        tl.append(gr.times[m * B + b])
        times_t.data[b] = Scalar[DT](tl[b])
    times_t.upload_resident(ctx)
    build_xt_ut["gpu", B, CHUNK * PAD](
        noise_t, acts_t, times_t, x_t, u_t, Optional(ctx)
    )
    st.set_times["gpu"](tl, Optional(ctx))
    var l = st.run["gpu", Pol.P](
        pol.expert, pol.cache, pol.denoiser, pol.action_in, pol.time_mlp_in,
        pol.time_mlp_out, pol.action_out, x_t, u_t, valid_t, gr.total_valid,
        Optional(ctx),
    )
    ph.step += Int(perf_counter_ns() - t_step)
    ph.n += B
    return l
