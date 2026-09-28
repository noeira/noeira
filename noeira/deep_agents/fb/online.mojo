"""`FBOnlineAgent` — Forward-Backward trained ONLINE, off-policy, on a batched
GPU env. `docs/BFM_ZERO_SHOT_RL.md` §18.3 step A3.

## Why this exists

The offline `FBTrainer` runs (§13, M2) recover ~19 / 5 / 5 % of the SAC experts
in their own dataset and the curve is flat past 300 k steps. §16.2 names the
constraint: FB run offline is bounded by the COVERAGE of a frozen dataset, and
the field's answer (FB-CPR, BFM-Zero) is no dataset at all — `pi_z` rolls out
in many parallel envs and generates its own coverage. This agent is that
change and ONLY that change: no discriminator, no style critic, no BC term by
default, so that a result is attributable to the coverage effect and nothing
else (§16.7's "isolate one change at a time").

It wraps the unchanged `FBTrainer` (one step body, already capture-safe) in
the `OffPolicyAgentGpu` surface that `run_offpolicy_train_batched` drives.
Nothing in the FB step is re-implemented here; what is new is what an online
loop needs around it:

  * a DEVICE replay ring `[CAP] x (obs | act | z | terminated | boundary)`,
    written by `record_batch_gpu` and gathered by the kernels FB already has.
    `next_obs` is NOT a column: the write order makes it `r_obs` one env step
    later, and `boundary` marks the 0.2 % of rows where that derivation does
    not hold (docs §12.23). That is 3256 B per transition instead of 5360;
  * a per-lane `z` held for `z_hold` env steps then resampled from a mixture
    of the sphere and a FIFO `ZBuffer` of recently trained `z` — BFM-Zero's
    rollout rule (`use_mix_rollout`, buffer of 10 k, hold 150);
  * `relabel_ratio`: 80 % of each minibatch's STORED `z` is overwritten by a
    fresh draw from the training mixture (uniform sphere / `B(s+)`), the
    remaining 20 % keeps the `z` the action was taken under;
  * `s+` drawn INDEPENDENTLY from the ring — the invariant `loss.mojo` states
    and cannot enforce.

## Capture safety — the two counters that must live on device

The driver captures `train_device_kernels` ONCE past `learning_starts` and
replays it. Two host quantities would be baked in at that point and silently
frozen forever:

  * the replay FILL. The uniform index draw reads `size` from a 1-element
    device buffer written by `record_batch_gpu` (`_set_size_kernel`), exactly
    as `data/replay_gpu.mojo` does — its header calls the host form "the
    catastrophic-divergence bug": sampling pinned to the warmup rows for the
    rest of the run, loss still descending.
  * every Philox OFFSET used inside the step: the index draws, the relabel
    uniforms, the mixture Gaussians, and `FBTrainer`'s own smoothing noise.
    All read a device offset and bump it with a kernel in-sequence.

`select_action_batched` and `record_batch_gpu` are EAGER (the driver never
captures them), so their RNG offsets stay host scalars.

⚠ `train_device_kernels_on(gctx)` is NOT overridden: `FBTrainer` enqueues on
its own stored context and cannot be redirected without changing it. Use the
default backend (`NOEIRA_GRAPH_BACKEND=stream`); the trait default RAISES on
the device-graph backend rather than recording a partial step.

## What is deliberately NOT here

  * Observation normalisation. BFM-Zero runs a `BatchNorm1d` on every input;
    the offline sweep's `obsnorm` arm is what decides whether it matters, and
    a running normaliser under a TD bootstrap is a moving target that needs
    its own gate. Inputs are raw, as every §13 number was.
  * The discriminator / `Q_D` (A4). By design — see the top of this file.
  * Termination masking in the measure target. `FBTrainer` has none, because
    dm_control tasks never terminate; `terminated` is stored in the ring so a
    terminating env (the task family) can add it without a ring change. A
    terminated row is also flagged as a sampling boundary, so a terminating
    env never trains on a pair that crosses the termination even before the
    mask exists.
"""

from max.gpu import global_idx, thread_idx
from std.math import sqrt, abs
from std.random.philox import Random as PhiloxRandom
from layout import Layout, LayoutTensor
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.core.logger import Logger, NoOpLogger
from noeira.nn.constants import DT, TPB
from noeira.nn.core.module import Module
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.call import call_forward
from noeira.nn.core.initializer import Initializer, Xavier
from noeira.nn.core.ptr import mptr
from noeira.nn.random.box_muller import (
    box_muller_normal_gpu,
    advance_rng_offset_kernel,
)
from noeira.data.resident import IDX_DT
from noeira.data.replay_gpu import (
    _uniform_indices_dev_kernel,
    _incr_offset_kernel,
    _set_size_kernel,
)

from ..data.n_step_replay import GPUNStepBuffer
from ..training.episode_tracker import EpisodeTracker
from ..training.device_mean_accum import DeviceMeanAccum
from ..training.driver_offpolicy import OffPolicyAgentGpu
from ..training.blocks.action_select import warmup_uniform_batched
from .trainer import FBTrainer
from .obs_ema import ObsEma
from .loss import fb_measure_loss_into, fb_ortho_loss_into
from .kernels import (
    gather_rows_kernel,
    gather_rows_into_kernel,
    derive_tail_kernel,
    gather_idx_kernel,
    pack2_kernel,
    project_sphere_kernel,
    gaussian_dev_t,
    uniform01_kernel,
    uniform01_dev_kernel,
    mean_sq_t,
    mean_t,
    ensure_t,
    _blocks,
)


# ══════════════════════════════════════════════════════════════════════
# Kernels
# ══════════════════════════════════════════════════════════════════════


def ring_store_kernel[
    OBS: Int, ACT: Int, D: Int, CAP: Int, LANES: Int
](
    obs_src: Pointer[Scalar[DT], MutAnyOrigin],
    act_src: Pointer[Scalar[DT], MutAnyOrigin],
    term_src: Pointer[Scalar[DT], MutAnyOrigin],
    z_src: Pointer[Scalar[DT], MutAnyOrigin],
    r_obs: Pointer[Scalar[DT], MutAnyOrigin],
    r_act: Pointer[Scalar[DT], MutAnyOrigin],
    r_term: Pointer[Scalar[DT], MutAnyOrigin],
    r_bnd: Pointer[Scalar[DT], MutAnyOrigin],
    r_age: Pointer[Scalar[DT], MutAnyOrigin],
    r_z: Pointer[Scalar[DT], MutAnyOrigin],
    pos: Int32,
    bnd: Int32,
    age: Int32,
):
    """Append `LANES` transitions at rows `(pos + lane) % CAP`, one launch.

    ⚠ `next_obs` is NOT stored (docs §12.23). The write order is the whole
    reason it need not be: lane `l` of step `s` lands at `(pos + l) % CAP`
    and `pos` advances by exactly `LANES` per step, so

        next_obs(row) == r_obs[(row + LANES) % CAP]

    for every row whose successor is the same lane's next step. That is every
    row except the last one before an episode boundary, which is what `r_bnd`
    marks — one float per row instead of a second `OBS`-wide copy of every
    observation (5360 B -> 3256 B per transition, the largest single item in
    the run's memory budget).

    `bnd` is a SCALAR because every lane resets on the same step (the
    driver's scheduled `_rsi_reset`); a per-lane termination is OR-ed in from
    `term_src` so a terminating env needs no further change here.

    Element-parallel over the concatenated row width so a wide `z` (128) does
    not serialise inside a per-lane thread. `pos` is a host scalar because
    `record_batch_gpu` is eager — the driver never captures it.
    """
    comptime W = OBS + ACT + D + 3
    var t = Int(global_idx.x)
    if t >= LANES * W:
        return
    var lane = t // W
    var k = t % W
    var row = (Int(pos) + lane) % CAP
    if k < OBS:
        r_obs[unsafe_offset=row * OBS + k] = obs_src[unsafe_offset=lane * OBS + k]
    elif k < OBS + ACT:
        var j = k - OBS
        r_act[unsafe_offset=row * ACT + j] = act_src[unsafe_offset=lane * ACT + j]
    elif k < OBS + ACT + D:
        var j = k - OBS - ACT
        r_z[unsafe_offset=row * D + j] = z_src[unsafe_offset=lane * D + j]
    elif k == OBS + ACT + D:
        r_term[unsafe_offset=row] = term_src[unsafe_offset=lane]
    elif k == OBS + ACT + D + 1:
        # steps since this lane's reset, capped by the caller. What the
        # DERIVED tail needs to know how far back it may look (§12.36).
        r_age[unsafe_offset=row] = age.cast[DT]()
    else:
        var terminated = term_src[unsafe_offset=lane] != Scalar[DT](0.0)
        var b = (Int(bnd) != 0) or terminated
        r_bnd[unsafe_offset=row] = Scalar[DT](1.0) if b else Scalar[DT](0.0)


def ring_indices_kernel[
    BATCH: Int, CAP: Int, LANES: Int
](
    indices: LayoutTensor[IDX_DT, Layout.row_major(BATCH), MutAnyOrigin],
    size_buf: LayoutTensor[DType.int32, Layout.row_major(1), MutAnyOrigin],
    pos_buf: LayoutTensor[DType.int32, Layout.row_major(1), MutAnyOrigin],
    r_bnd: Pointer[Scalar[DT], MutAnyOrigin],
    seed: UInt64,
    offset_buf: LayoutTensor[DType.uint64, Layout.row_major(1), MutAnyOrigin],
):
    """The `s` draw, once `s'` is derived rather than stored (§12.23).

    `_uniform_indices_dev_kernel` cannot be used for this draw any more,
    because two rows of the ring are no longer interchangeable:

      * the newest `LANES` rows have NO successor yet — `(row + LANES) % CAP`
        is either unwritten (before the first wrap) or a row from `CAP` steps
        ago (after it). They are excluded from the draw;
      * a row flagged in `r_bnd` has a successor that is a POST-RESET
        observation, not its own next state. It is remapped to the same
        lane's previous step, which is 500 steps from the nearest boundary
        and therefore always safe.

    The exclusion is a WINDOW, not a prefix, so the draw is over `j` in
    `[0, size - LANES)` and the row is `(base + j) % CAP` with
    `base = (pos - size) mod CAP` the oldest stored row — 0 before the ring
    wraps, `pos` after it.

    ⚠ `CAP` need NOT be a multiple of `LANES`. Each step writes the block
    `[pos, pos + LANES)` mod `CAP` and `pos` advances by exactly `LANES`, so
    the blocks tile the circle contiguously whatever the remainder: every row
    is rewritten exactly once per `CAP` rows written, `(row + LANES) % CAP`
    is always the same lane's next step, and `base` is always the oldest row.
    (`CAP = 2_000_000` with 1024 lanes is not a multiple, and is correct.)

    The remap costs ~0.2 % of draws a duplicated predecessor (one boundary
    per lane per `T_EPISODE` = 500 steps). That is a sampling bias, not a
    corrupted pair, and it is the price of the whole change.

    Both `size` and `pos` are read from DEVICE buffers: this draw runs inside
    the captured training step, and a host scalar here is the
    "catastrophic-divergence bug" `replay_gpu.mojo` names — sampling frozen
    to the capture-time fill for the rest of the run.
    """
    var i = Int(global_idx.x)
    if i >= BATCH:
        return
    var size = Int(size_buf[0])
    var n = size - LANES
    if n < 1:
        n = 1
    var base = Int(pos_buf[0]) - size
    if base < 0:
        base += CAP
    var offset_base = rebind[UInt64](offset_buf[0])
    var philox = PhiloxRandom(seed=seed + UInt64(i), offset=offset_base)
    var u = Float32(philox.step_uniform()[0])
    var j = Int(u * Float32(n))
    if j >= n:
        j = n - 1
    if j < 0:
        j = 0
    var row = base + j
    if row >= CAP:
        row -= CAP
    if r_bnd[unsafe_offset=row] != Scalar[DT](0.0):
        # step back one env step within the same lane; forward if the window
        # has no room behind (only reachable during warmup, when n < 2*LANES)
        var j2 = j - LANES
        if j2 < 0:
            j2 = j + LANES
            if j2 >= n:
                j2 = j
        j = j2
        row = base + j
        if row >= CAP:
            row -= CAP
    indices[i] = Scalar[IDX_DT](row)


def ring_next_idx_kernel[BATCH: Int, CAP: Int, LANES: Int](
    idx: Pointer[Scalar[IDX_DT], MutAnyOrigin],
    dst: Pointer[Scalar[IDX_DT], MutAnyOrigin],
):
    """`dst[i] = (idx[i] + LANES) % CAP` — the successor row of a sampled row.

    Separate from the draw so the SAME index array feeds both the `s` gather
    and the `s'` gather: `s` and `s'` must come from one index, or they are
    two unrelated transitions that happen to be one step apart.
    """
    var i = Int(global_idx.x)
    if i >= BATCH:
        return
    var r = Int(idx[unsafe_offset=i]) + LANES
    if r >= CAP:
        r -= CAP
    dst[unsafe_offset=i] = Scalar[IDX_DT](r)


def z_lane_resample_kernel[D: Int, LANES: Int, ZBUF: Int](
    z: Pointer[Scalar[DT], MutAnyOrigin],
    gauss: Pointer[Scalar[DT], MutAnyOrigin],
    zbuf: Pointer[Scalar[DT], MutAnyOrigin],
    zbuf_fill: LayoutTensor[DType.int32, Layout.row_major(1), MutAnyOrigin],
    pick: Pointer[Scalar[DT], MutAnyOrigin],
    iter: Int32,
    hold: Int32,
    zbuf_frac: Scalar[DT],
    force: Int32,
):
    """Per lane: if this lane's `z` is due, replace it.

    A lane is due every `hold` iterations, STAGGERED by lane so the whole
    population does not switch task on the same step (`force != 0` makes
    every lane due — used once to initialise). With probability `zbuf_frac`
    the new `z` is a row of the ZBuffer (a `z` recently trained on, so the
    rollout exercises the region the losses are shaping); otherwise a
    Gaussian, to be projected by `project_sphere_kernel` right after.

    ⚠ NOT renormalised here. The projection runs unconditionally on every
    lane after this kernel, whether or not it was resampled — the invariant
    `z_sampler.mojo` states is "every producer renormalises", and a lane
    that skipped the projection because it was not due is one refactor from
    a lane that skipped it because of a bug.
    """
    var e = Int(global_idx.x)
    if e >= LANES:
        return
    var h = Int(hold)
    if h < 1:
        h = 1
    var stagger = (e * h) // LANES
    var due = (Int(force) != 0) or (((Int(iter) + stagger) % h) == 0)
    if not due:
        return
    var fill = Int(zbuf_fill[0])
    var base = e * D
    if fill > 0 and pick[unsafe_offset=2 * e] < zbuf_frac:
        var src = Int(pick[unsafe_offset=2 * e + 1] * Scalar[DT](fill))
        if src >= fill:
            src = fill - 1
        if src < 0:
            src = 0
        for k in range(D):
            z[unsafe_offset=base + k] = zbuf[unsafe_offset=src * D + k]
    else:
        for k in range(D):
            z[unsafe_offset=base + k] = gauss[unsafe_offset=base + k]


def z_relabel_kernel[D: Int, BATCH: Int](
    z: Pointer[Scalar[DT], MutAnyOrigin],
    gauss: Pointer[Scalar[DT], MutAnyOrigin],
    b_states: Pointer[Scalar[DT], MutAnyOrigin],
    pick: Pointer[Scalar[DT], MutAnyOrigin],
    keep_frac: Scalar[DT],
    uniform_frac: Scalar[DT],
    keep_from: Int32,
):
    """BFM-Zero's `relabel_ratio`, one thread per ROW of the gathered batch.
    Rows below `keep_from` NEVER keep: they were gathered from an expert
    store that carries no z, so their `z` slot is stale and must be drawn.

    `z` arrives holding the STORED `z` of each transition. With probability
    `keep_frac` a row keeps it; otherwise it is overwritten by the training
    mixture — `uniform_frac` of the time a Gaussian (→ sphere), else the
    `B(s+)` embedding of a random row of the batch. Three uniforms per row in
    `pick`; drawn outside so this kernel holds no RNG state.
    """
    var i = Int(global_idx.x)
    if i >= BATCH:
        return
    var base = i * D
    if i >= Int(keep_from) and pick[unsafe_offset=3 * i] < keep_frac:
        return
    if pick[unsafe_offset=3 * i + 1] < uniform_frac:
        for k in range(D):
            z[unsafe_offset=base + k] = gauss[unsafe_offset=base + k]
    else:
        var src = Int(pick[unsafe_offset=3 * i + 2] * Scalar[DT](BATCH))
        if src >= BATCH:
            src = BATCH - 1
        if src < 0:
            src = 0
        for k in range(D):
            z[unsafe_offset=base + k] = b_states[unsafe_offset=src * D + k]


def zbuf_push_kernel[D: Int, BATCH: Int, ZBUF: Int](
    zbuf: Pointer[Scalar[DT], MutAnyOrigin],
    z: Pointer[Scalar[DT], MutAnyOrigin],
    head: LayoutTensor[DType.int32, Layout.row_major(1), MutAnyOrigin],
):
    """Append the batch's `z` rows to the FIFO at `(head + i) % ZBUF`. The
    head is read from device (captured path); `zbuf_advance_kernel` bumps it
    AFTER this launch so every thread reads the same head."""
    var t = Int(global_idx.x)
    if t >= BATCH * D:
        return
    var i = t // D
    var k = t % D
    var row = (Int(head[0]) + i) % ZBUF
    zbuf[unsafe_offset=row * D + k] = z[unsafe_offset=t]


def zbuf_advance_kernel[BATCH: Int, ZBUF: Int](
    head: LayoutTensor[DType.int32, Layout.row_major(1), MutAnyOrigin],
    fill: LayoutTensor[DType.int32, Layout.row_major(1), MutAnyOrigin],
):
    if Int(thread_idx.x) != 0:
        return
    head[0] = Int32((Int(head[0]) + BATCH) % ZBUF)
    var f = Int(fill[0]) + BATCH
    if f > ZBUF:
        f = ZBUF
    fill[0] = Int32(f)


def explore_action_kernel[N: Int](
    action: Pointer[Scalar[DT], MutAnyOrigin],
    pi: Pointer[Scalar[DT], MutAnyOrigin],
    noise: Pointer[Scalar[DT], MutAnyOrigin],
    std: Scalar[DT],
    scale: Scalar[DT],
):
    """`action = clamp(pi + std·n, ±1) · scale` — BFM-Zero's `actor_std`
    exploration on a deterministic actor. `std = 0` is the greedy path."""
    var t = Int(global_idx.x)
    if t >= N:
        return
    var v = pi[unsafe_offset=t] + std * noise[unsafe_offset=t]
    if v > Scalar[DT](1.0):
        v = Scalar[DT](1.0)
    elif v < Scalar[DT](-1.0):
        v = Scalar[DT](-1.0)
    action[unsafe_offset=t] = v * scale


# ══════════════════════════════════════════════════════════════════════
# The agent
# ══════════════════════════════════════════════════════════════════════


def z_pin_kernel[D: Int, LANES: Int](
    z_lane: Pointer[Scalar[DT], MutAnyOrigin],
    z_pin: Pointer[Scalar[DT], MutAnyOrigin],
    mask: Pointer[Scalar[DT], MutAnyOrigin],
):
    """`z_lane[l] = z_pin[l]` where `mask[l] > 0` — the driver-pinned lanes."""
    var i = Int(global_idx.x)
    if i >= LANES * D:
        return
    var l = i // D
    if mask[unsafe_offset=l] > Scalar[DT](0):
        z_lane[unsafe_offset=i] = z_pin[unsafe_offset=i]


struct FBOnlineAgent[
    FNET: Module,
    BNET: Module,
    ANET: Module,
    OBS: Int,
    ACT: Int,
    D: Int,
    BATCH: Int,
    CAP: Int,
    LANES: Int,
    ZBUF: Int = 10_000,
    EXPERT_ROWS: Int = 0,
    DERIVED_TAIL: Int = 0,
](OffPolicyAgentGpu):
    """`OffPolicyAgentGpu` conformer around `FBTrainer[..., "gpu"]`.

    `LANES` is the env count the rollout state (`z` per lane) is sized for;
    the driver's `N_ENVS` must equal it, asserted at each entry point.

    `EXPERT_ROWS` (A3.5, §18.7.2): when > 0 and an expert store is attached
    with `attach_expert`, the first `EXPERT_ROWS` rows of every training
    batch — `s`, `a`, `s'`, and `s+` — are gathered from that store and the
    remaining `BATCH - EXPERT_ROWS` from the online ring. BC applies to the
    expert rows ONLY (`FBTrainer.fill_bc_mask`); their `z` is always drawn
    fresh (no stored z to keep). Without an attached store the agent samples
    the ring alone, whatever `EXPERT_ROWS` says — one binary, both arms.
    """

    comptime AGENT_TRAIN_TARGET: StaticString = "gpu"
    comptime AGENT_OBS_DIM: Int = Self.OBS
    comptime AGENT_ACT_DIM: Int = Self.ACT
    comptime TrainerT = FBTrainer[
        Self.FNET, Self.BNET, Self.ANET,
        Self.OBS, Self.ACT, Self.D, Self.BATCH, "gpu",
    ]
    comptime A_IN: Int = Self.OBS + Self.D
    comptime RING_ROWS: Int = Self.BATCH - Self.EXPERT_ROWS
    # ⚠ The ring stores `STORE_OBS` columns; the batch row is `OBS` wide and
    # the last `DERIVED_TAIL` of it is DERIVED from the same lane's earlier
    # rows (docs §12.36), the way `next_obs` is derived from the next one
    # (§12.23). At `DERIVED_TAIL == 0` — every caller but the G1 — the two are
    # equal and every path below is what it was.
    comptime STORE_OBS: Int = Self.OBS - Self.DERIVED_TAIL

    var t: Self.TrainerT
    var ctx: Optional[DeviceContext]
    var tracker: EpisodeTracker

    # ── replay ring (device only — no host mirror of CAP x D floats) ────
    var r_obs: Tensor
    var r_act: Tensor
    var r_z: Tensor
    var r_term: Tensor
    var r_bnd: Tensor          # CAP, 1 = this row's successor is post-reset
    var r_age: Tensor          # CAP, steps since this lane's reset (capped)
    var tail_spec: Optional[DeviceBuffer[DType.int32]]
    var age_next: Int          # set by `set_age`, consumed by the next record
    var act_norm: Scalar[DT]   # the derived tail's action scaling ...
    var act_clip: Scalar[DT]   # ... and its clip, both env-specific
    var size: Int
    var pos: Int
    var bnd_next: Bool         # set by `set_boundary`, consumed by the next record
    var size_dev: Optional[DeviceBuffer[DType.int32]]
    var pos_dev: Optional[DeviceBuffer[DType.int32]]
    var samp_off: Optional[DeviceBuffer[DType.uint64]]
    var idx_s: Optional[DeviceBuffer[IDX_DT]]
    var idx_sn: Optional[DeviceBuffer[IDX_DT]]
    var idx_sp: Optional[DeviceBuffer[IDX_DT]]

    # ── expert store (A3.5), device-resident; attached after `make` ───────
    var exp_obs: Tensor
    var exp_act: Tensor
    var exp_nxt: Optional[DeviceBuffer[IDX_DT]]
    var exp_size_dev: Optional[DeviceBuffer[DType.int32]]
    var idx_e: Optional[DeviceBuffer[IDX_DT]]
    var idx_esp: Optional[DeviceBuffer[IDX_DT]]
    var idx_en: Optional[DeviceBuffer[IDX_DT]]
    var _expert_n: Int

    # ── training-side RNG scratch (device offsets: captured path) ───────
    var rng_dev: Optional[DeviceBuffer[DType.uint64]]
    var gauss: Tensor      # BATCH * D
    var pick: Tensor       # BATCH * 3

    # ── ZBuffer ─────────────────────────────────────────────────────────
    var zbuf: Tensor       # ZBUF * D
    var zbuf_head: Optional[DeviceBuffer[DType.int32]]
    var zbuf_fill: Optional[DeviceBuffer[DType.int32]]

    # ── rollout (eager) ─────────────────────────────────────────────────
    var z_lane: Tensor     # LANES * D
    var _gauss_lane: Tensor  # LANES * D
    var _pick_lane: Tensor   # LANES * 2
    var _ain: Tensor       # LANES * A_IN
    var _pi: Tensor        # LANES * ACT
    var _noise: Tensor     # LANES * ACT
    var _act_iter: Int
    var _roll_seed: UInt64
    var _roll_off: UInt64
    var _warmup_seed: UInt64
    var _warmup_off: UInt64

    # ── hyperparameters ─────────────────────────────────────────────────
    var learning_starts: Int
    var action_scale: Scalar[DT]
    var expl_std: Float64
    var z_hold: Int
    var zbuf_frac: Float64
    var keep_frac: Float64
    var uniform_frac: Float64
    var _train_seed: UInt64

    # ── bookkeeping ─────────────────────────────────────────────────────
    var _update_count: Int
    var _total_train_steps: Int
    var _mean_abs_action_dev: DeviceMeanAccum
    var _z_lane_resamples: Int
    # ── observation normaliser (opt-in, `normalize_obs`): BFM-Zero's
    # BatchNorm1d run the reference's way — stats updated from every
    # training batch's `s` and `s'`, applied to `s`, `s'`, `s+`, the expert
    # rows and the rollout observation. Raw rows stay in the ring.
    var obs_ema: ObsEma[Self.OBS]
    var _obs_scratch: Tensor   # LANES * OBS, the normalised rollout obs
    # ── driver-pinned lane z (opt-in, `enable_z_pin`): BFM-Zero's expert
    # rollouts drive half the lanes along a tracking-z SEQUENCE for 250
    # steps (`fb/agent.py::maybe_update_rollout_context`). The driver
    # writes `z_pin` / `z_pin_mask` on device every step; `_resample_lanes`
    # applies them AFTER its own draw, so a pinned lane's stored z (what
    # the ring records and D's negatives carry) is the pinned one.
    var z_pin: Tensor          # LANES * D
    var z_pin_mask: Tensor     # LANES, 1 = pinned
    var z_pin_on: Bool

    def __init__(out self):
        self.t = Self.TrainerT()
        self.ctx = None
        self.tracker = EpisodeTracker(
            window=List[Scalar[DT]](), window_size=0, idx=0,
            current_return=Scalar[DT](0.0), ep_count=0,
        )
        self.r_obs = Tensor()
        self.r_act = Tensor()
        self.r_z = Tensor()
        self.r_term = Tensor()
        self.r_bnd = Tensor()
        self.r_age = Tensor()
        self.tail_spec = None
        self.age_next = 0
        self.act_norm = Scalar[DT](1.0)
        self.act_clip = Scalar[DT](1e30)
        self.size = 0
        self.pos = 0
        self.bnd_next = False
        self.size_dev = None
        self.pos_dev = None
        self.samp_off = None
        self.idx_s = None
        self.idx_sn = None
        self.idx_sp = None
        self.exp_obs = Tensor()
        self.exp_act = Tensor()
        self.exp_nxt = None
        self.exp_size_dev = None
        self.idx_e = None
        self.idx_esp = None
        self.idx_en = None
        self._expert_n = 0
        self.rng_dev = None
        self.gauss = Tensor()
        self.pick = Tensor()
        self.zbuf = Tensor()
        self.zbuf_head = None
        self.zbuf_fill = None
        self.z_lane = Tensor()
        self._gauss_lane = Tensor()
        self._pick_lane = Tensor()
        self._ain = Tensor()
        self._pi = Tensor()
        self._noise = Tensor()
        self._act_iter = 0
        self._roll_seed = UInt64(0xF0B0_0011)
        self._roll_off = UInt64(0)
        self._warmup_seed = UInt64(0xF0B0_0022)
        self._warmup_off = UInt64(0)
        self.learning_starts = 0
        self.action_scale = Scalar[DT](1.0)
        self.expl_std = 0.2
        self.z_hold = 150
        self.zbuf_frac = 0.5
        self.keep_frac = 0.2
        self.uniform_frac = 0.5
        self._train_seed = UInt64(0xF0B0_0033)
        self._update_count = 0
        self._total_train_steps = 0
        self._mean_abs_action_dev = DeviceMeanAccum()
        self._z_lane_resamples = 0
        self.obs_ema = ObsEma[Self.OBS]()
        self._obs_scratch = Tensor()
        self.z_pin = Tensor()
        self.z_pin_mask = Tensor()
        self.z_pin_on = False

    @staticmethod
    def make[
        INIT: Initializer = Xavier
    ](
        ctx: DeviceContext,
        *,
        lr: Float64 = 3e-4,
        lr_b: Float64 = -1.0,
        gamma: Float64 = 0.98,
        tau: Float64 = 0.01,
        ortho_weight: Float64 = 1.0,
        max_grad_norm: Float64 = 1.0,
        bc_weight: Float64 = 0.0,
        act_l2_weight: Float64 = 100.0,
        act_l2_margin: Float64 = 0.8,
        learning_starts: Int = 10_000,
        action_scale: Float64 = 1.0,
        expl_std: Float64 = 0.2,
        z_hold: Int = 150,
        zbuf_frac: Float64 = 0.5,
        keep_frac: Float64 = 0.2,
        uniform_frac: Float64 = 0.5,
        window_size: Int = 100,
        initial_episode_fill: Float64 = 0.0,
        seed: UInt64 = UInt64(0x5EED_0B),
        normalize_obs: Bool = False,
    ) raises -> Self:
        """Defaults are BFM-Zero's rollout / relabel settings on top of
        `FBTrainer.make`'s (`gamma` 0.98, `tau` 0.01, Adam 3e-4).

        `bc_weight = 0`, `act_l2_weight = 100`, `act_l2_margin = 0.8`. Two
        online walker runs (2026-09-07) bracket this: no penalty went
        bang-bang from the first flush (mean|a| 0.82 → 0.88, 82–90 %
        saturated, walk/run below random); a plain L2 at 1.0 went NULL
        (replay mean|a| 0.19, eval 0.10, walker lying still). Both are the
        same trap — whatever action regime the actor is in, the ring fills
        with it, `F` fits it, and it becomes self-consistent — and the
        plain L2 lost because the value term had been normalised by |F·z|
        (the BC-only scale, wrongly extended to it) — see
        `FBTrainer.act_l2_margin`. Now the value gradient is RAW and the
        hinge caps the corner. The weight is deliberately STRONG: under Adam
        only its ratio to the value gradient matters, too strong merely
        hardens the wall at the margin (the interior stays the value term's),
        too weak is run 1 again — on random data w = 2 moved mean|a| by 0.01.
        Read `gA value -> total` at the first flush: the actor-gradient RMS
        before and after the penalty; a total far above the value means the
        wall is active, equal means no action reached it.
        """
        comptime assert Self.CAP >= Self.BATCH, (
            "FBOnlineAgent: CAP must be >= BATCH"
        )
        comptime assert Self.ZBUF >= Self.BATCH, (
            "FBOnlineAgent: ZBUF must hold at least one training batch"
        )
        comptime assert Self.EXPERT_ROWS >= 0 and Self.EXPERT_ROWS < Self.BATCH, (
            "FBOnlineAgent: EXPERT_ROWS must be in [0, BATCH)"
        )
        if learning_starts < Self.BATCH:
            raise Error(
                "FBOnlineAgent.make: learning_starts must be >= BATCH — the"
                " first captured step samples from the ring at that fill"
            )
        var a = Self()
        var octx = Optional[DeviceContext](ctx)
        a.ctx = octx
        if normalize_obs:
            a.obs_ema = ObsEma[Self.OBS].make(ctx)
            ensure_t["gpu"](a._obs_scratch, Self.LANES * Self.OBS, octx)
        a.t = Self.TrainerT.make[INIT](
            lr=lr, gamma=gamma, tau=tau, ortho_weight=ortho_weight,
            ctx=octx, seed=seed + 13, max_grad_norm=max_grad_norm,
            bc_weight=bc_weight, lr_b=lr_b, act_l2_weight=act_l2_weight,
            act_l2_margin=act_l2_margin,
        )
        a.t.ensure_sized()
        a.tracker = EpisodeTracker.new(
            window_size=window_size,
            initial_fill=Scalar[DT](initial_episode_fill),
        )
        a.learning_starts = learning_starts
        a.action_scale = Scalar[DT](action_scale)
        a.expl_std = expl_std
        a.z_hold = z_hold
        a.zbuf_frac = zbuf_frac
        a.keep_frac = keep_frac
        a.uniform_frac = uniform_frac
        a._train_seed = seed
        a._roll_seed = seed + 101
        a._warmup_seed = seed + 202

        # Ring: device only. A host mirror of CAP x (OBS + ACT + D + 2)
        # floats at CAP = 1 M would be ~350 MB of host RAM nothing reads.
        # `next_obs` is derived, not stored — see `ring_store_kernel`.
        a.r_obs.ensure_gpu(ctx, Self.CAP * Self.STORE_OBS)
        a.r_act.ensure_gpu(ctx, Self.CAP * Self.ACT)
        a.r_z.ensure_gpu(ctx, Self.CAP * Self.D)
        a.r_term.ensure_gpu(ctx, Self.CAP)
        a.r_bnd.ensure_gpu(ctx, Self.CAP)
        # Every row starts flagged. This is NOT what keeps the warmup safe —
        # `learning_starts` and the draw's window bound are — but it means a
        # ring inspected before it has been written reads as "no valid
        # successor" rather than as a transition into uninitialised memory.
        a.r_bnd.dev.value().enqueue_fill(Scalar[DT](1.0))
        a.r_age.ensure_gpu(ctx, Self.CAP)
        a.r_age.dev.value().enqueue_fill(Scalar[DT](0.0))
        var sz = ctx.enqueue_create_buffer[DType.int32](1)
        sz.enqueue_fill(Int32(0))
        a.size_dev = sz^
        var pz = ctx.enqueue_create_buffer[DType.int32](1)
        pz.enqueue_fill(Int32(0))
        a.pos_dev = pz^
        var so = ctx.enqueue_create_buffer[DType.uint64](1)
        so.enqueue_fill(UInt64(0))
        a.samp_off = so^
        a.idx_s = ctx.enqueue_create_buffer[IDX_DT](Self.BATCH)
        a.idx_sn = ctx.enqueue_create_buffer[IDX_DT](Self.BATCH)
        a.idx_sp = ctx.enqueue_create_buffer[IDX_DT](Self.BATCH)

        var ro = ctx.enqueue_create_buffer[DType.uint64](1)
        ro.enqueue_fill(UInt64(0))
        a.rng_dev = ro^
        ensure_t["gpu"](a.gauss, Self.BATCH * Self.D, octx)
        ensure_t["gpu"](a.pick, Self.BATCH * 3, octx)

        a.zbuf.ensure_gpu(ctx, Self.ZBUF * Self.D)
        var zh = ctx.enqueue_create_buffer[DType.int32](1)
        zh.enqueue_fill(Int32(0))
        a.zbuf_head = zh^
        var zf = ctx.enqueue_create_buffer[DType.int32](1)
        zf.enqueue_fill(Int32(0))
        a.zbuf_fill = zf^

        ensure_t["gpu"](a.z_lane, Self.LANES * Self.D, octx)
        ensure_t["gpu"](a._gauss_lane, Self.LANES * Self.D, octx)
        ensure_t["gpu"](a._pick_lane, Self.LANES * 2, octx)
        ensure_t["gpu"](a._ain, Self.LANES * Self.A_IN, octx)
        ensure_t["gpu"](a._pi, Self.LANES * Self.ACT, octx)
        ensure_t["gpu"](a._noise, Self.LANES * Self.ACT, octx)

        a._mean_abs_action_dev = DeviceMeanAccum.make["gpu"](ctx=octx)

        # Every lane gets a z before the first action — forced resample, then
        # the unconditional projection.
        a._resample_lanes(force=True)
        ctx.synchronize()
        return a^

    # ── expert store (A3.5) ──────────────────────────────────────────────

    def attach_expert(
        mut self,
        var obs: Tensor,
        var act: Tensor,
        var nxt: DeviceBuffer[IDX_DT],
        n_rows: Int,
    ) raises:
        """Hand the agent a device-resident expert store: `obs` `[n_rows, OBS]`
        and `act` `[n_rows, ACT]` already uploaded, `nxt` the episode-safe
        next-row table (`nxt[r] = r` on an episode's last row — the same rule
        `fb_train_gpu.mojo` builds). ⚠ Call BEFORE the driver: it writes the
        BC mask in place, which must precede any capture.

        ⚠ `obs` must be the SAME representation the batched env emits (for
        walker, dm_control's 24-D vector via `Phyics3dEnv.obs_at`), not the
        store's `[qpos | qvel]`; a wrong layout trains fine and evaluates to
        noise."""
        comptime assert Self.EXPERT_ROWS > 0, (
            "FBOnlineAgent.attach_expert: EXPERT_ROWS is 0 on this agent"
        )
        if n_rows < Self.EXPERT_ROWS:
            raise Error("attach_expert: fewer rows than EXPERT_ROWS")
        if not obs.dev or not act.dev:
            raise Error("attach_expert: obs/act must be uploaded to device")
        var c = self.ctx.value()
        self.exp_obs = obs^
        self.exp_act = act^
        self.exp_nxt = nxt^
        var sz = c.enqueue_create_buffer[DType.int32](1)
        sz.enqueue_fill(Int32(n_rows))
        self.exp_size_dev = sz^
        self.idx_e = c.enqueue_create_buffer[IDX_DT](Self.EXPERT_ROWS)
        self.idx_esp = c.enqueue_create_buffer[IDX_DT](Self.EXPERT_ROWS)
        self.idx_en = c.enqueue_create_buffer[IDX_DT](Self.EXPERT_ROWS)
        self._expert_n = n_rows
        self.t.fill_bc_mask(Self.EXPERT_ROWS)
        c.synchronize()

    def has_expert(self) -> Bool:
        return self._expert_n > 0

    def _gather_ring[ROWS: Int](mut self, row0: Int) raises:
        """Two independent uniform draws over the ring's fill, gathered into
        batch rows `[row0, row0 + ROWS)` of `s`, `a`, `s'`, `z` and `s+`.

        The `s` draw goes through `ring_indices_kernel` (windowed + boundary
        remapped) because `s'` is DERIVED from the same index; the `s+` draw
        stays a plain uniform over the fill, because a goal state needs no
        successor and every stored row is a legitimate goal.
        """
        var c = self.ctx.value()
        var size_lt = LayoutTensor[DType.int32, Layout.row_major(1)](
            self.size_dev.value()
        )
        var pos_lt = LayoutTensor[DType.int32, Layout.row_major(1)](
            self.pos_dev.value()
        )
        var off_lt = LayoutTensor[DType.uint64, Layout.row_major(1)](
            self.samp_off.value()
        )
        var is_lt = LayoutTensor[IDX_DT, Layout.row_major(ROWS)](
            self.idx_s.value()
        )
        var isp_lt = LayoutTensor[IDX_DT, Layout.row_major(ROWS)](
            self.idx_sp.value()
        )
        comptime nb = _blocks(ROWS)
        c.enqueue_function[ring_indices_kernel[ROWS, Self.CAP, Self.LANES]](
            is_lt, size_lt, pos_lt,
            mptr(self.r_bnd.dev.value().unsafe_ptr()),
            self._train_seed, off_lt,
            grid_dim=nb, block_dim=TPB,
        )
        c.enqueue_function[_incr_offset_kernel[ROWS]](
            off_lt, grid_dim=1, block_dim=1,
        )
        c.enqueue_function[_uniform_indices_dev_kernel[ROWS]](
            isp_lt, size_lt, self._train_seed, off_lt,
            grid_dim=nb, block_dim=TPB,
        )
        c.enqueue_function[_incr_offset_kernel[ROWS]](
            off_lt, grid_dim=1, block_dim=1,
        )
        var ip_s = mptr(self.idx_s.value().unsafe_ptr())
        var ip_sn = mptr(self.idx_sn.value().unsafe_ptr())
        var ip_sp = mptr(self.idx_sp.value().unsafe_ptr())
        c.enqueue_function[ring_next_idx_kernel[ROWS, Self.CAP, Self.LANES]](
            ip_s, ip_sn, grid_dim=nb, block_dim=TPB,
        )
        # the STORED head; the derived tail follows below
        c.enqueue_function[
            gather_rows_into_kernel[Self.STORE_OBS, Self.OBS, ROWS]
        ](
            mptr(self.r_obs.dev.value().unsafe_ptr()), ip_s,
            mptr(self.t.bs.dev.value().unsafe_ptr()).unsafe_offset(row0 * Self.OBS),
            grid_dim=_blocks(ROWS * Self.STORE_OBS), block_dim=TPB,
        )
        c.enqueue_function[gather_rows_kernel[Self.ACT, ROWS]](
            mptr(self.r_act.dev.value().unsafe_ptr()), ip_s,
            mptr(self.t.ba.dev.value().unsafe_ptr()).unsafe_offset(row0 * Self.ACT),
            grid_dim=_blocks(ROWS * Self.ACT), block_dim=TPB,
        )
        c.enqueue_function[
            gather_rows_into_kernel[Self.STORE_OBS, Self.OBS, ROWS]
        ](
            mptr(self.r_obs.dev.value().unsafe_ptr()), ip_sn,
            mptr(self.t.bsn.dev.value().unsafe_ptr()).unsafe_offset(row0 * Self.OBS),
            grid_dim=_blocks(ROWS * Self.STORE_OBS), block_dim=TPB,
        )
        c.enqueue_function[gather_rows_kernel[Self.D, ROWS]](
            mptr(self.r_z.dev.value().unsafe_ptr()), ip_s,
            mptr(self.t.bz.dev.value().unsafe_ptr()).unsafe_offset(row0 * Self.D),
            grid_dim=_blocks(ROWS * Self.D), block_dim=TPB,
        )
        c.enqueue_function[
            gather_rows_into_kernel[Self.STORE_OBS, Self.OBS, ROWS]
        ](
            mptr(self.r_obs.dev.value().unsafe_ptr()), ip_sp,
            mptr(self.t.bsp.dev.value().unsafe_ptr()).unsafe_offset(row0 * Self.OBS),
            grid_dim=_blocks(ROWS * Self.STORE_OBS), block_dim=TPB,
        )

        # ── the DERIVED tail (§12.36) ────────────────────────────────
        # `bs` and `bsn` feed `f` / `critic`, which take the whole row.
        # `bsp` feeds `b` ONLY, whose filter is the stored head — but its tail
        # is filled anyway: leaving a slice of a live tensor holding whatever
        # the last step left there is the kind of thing that is correct until
        # someone widens a filter.
        comptime if Self.DERIVED_TAIL > 0:
            var sp = self.tail_spec.value().unsafe_ptr()
            var ro = mptr(self.r_obs.dev.value().unsafe_ptr())
            var ra = mptr(self.r_act.dev.value().unsafe_ptr())
            var rg = mptr(self.r_age.dev.value().unsafe_ptr())
            comptime TK = derive_tail_kernel[
                ROWS, Self.CAP, Self.LANES, Self.STORE_OBS, Self.ACT,
                Self.DERIVED_TAIL, Self.OBS,
            ]
            comptime nbt = _blocks(ROWS * Self.DERIVED_TAIL)
            c.enqueue_function[TK](
                ro, ra, rg, ip_s, sp, self.act_norm, self.act_clip,
                mptr(self.t.bs.dev.value().unsafe_ptr()), Int32(row0),
                grid_dim=nbt, block_dim=TPB,
            )
            c.enqueue_function[TK](
                ro, ra, rg, ip_sn, sp, self.act_norm, self.act_clip,
                mptr(self.t.bsn.dev.value().unsafe_ptr()), Int32(row0),
                grid_dim=nbt, block_dim=TPB,
            )
            c.enqueue_function[TK](
                ro, ra, rg, ip_sp, sp, self.act_norm, self.act_clip,
                mptr(self.t.bsp.dev.value().unsafe_ptr()), Int32(row0),
                grid_dim=nbt, block_dim=TPB,
            )

    def _gather_expert(mut self) raises:
        """Batch rows `[0, EXPERT_ROWS)` of `s`, `a`, `s'`, `s+` from the
        expert store. `z` is left for the relabel to draw.

        ⚠ NOT COMPATIBLE WITH `DERIVED_TAIL > 0`, and this refuses rather than
        producing something plausible. The derived tail is lane-strided — row
        `r - k * LANES` is the same lane `k` steps earlier — and an expert
        store is CONTIGUOUS, so the same arithmetic reads unrelated rows. A
        store also has no actions, so `last_action` and the history's
        `actions` key have no source at all. The G1 run uses
        `EXPERT_ROWS = 0` (its expert data reaches the losses through the
        window encoding and the discriminator, both of which filter to the
        stored head), so this combination has never been needed.
        """
        comptime assert Self.DERIVED_TAIL == 0, (
            "FBOnlineAgent: EXPERT_ROWS > 0 with DERIVED_TAIL > 0 is not"
            " supported — the tail's back-steps are lane-strided and an"
            " expert store is contiguous, and a store carries no actions."
            " See `_gather_expert`."
        )
        comptime E = Self.EXPERT_ROWS
        var c = self.ctx.value()
        var size_lt = LayoutTensor[DType.int32, Layout.row_major(1)](
            self.exp_size_dev.value()
        )
        var off_lt = LayoutTensor[DType.uint64, Layout.row_major(1)](
            self.samp_off.value()
        )
        var ie_lt = LayoutTensor[IDX_DT, Layout.row_major(E)](self.idx_e.value())
        var iesp_lt = LayoutTensor[IDX_DT, Layout.row_major(E)](self.idx_esp.value())
        comptime nb = _blocks(E)
        c.enqueue_function[_uniform_indices_dev_kernel[E]](
            ie_lt, size_lt, self._train_seed + 41, off_lt,
            grid_dim=nb, block_dim=TPB,
        )
        c.enqueue_function[_incr_offset_kernel[E]](off_lt, grid_dim=1, block_dim=1)
        c.enqueue_function[_uniform_indices_dev_kernel[E]](
            iesp_lt, size_lt, self._train_seed + 41, off_lt,
            grid_dim=nb, block_dim=TPB,
        )
        c.enqueue_function[_incr_offset_kernel[E]](off_lt, grid_dim=1, block_dim=1)
        var ip_e = mptr(self.idx_e.value().unsafe_ptr())
        var ip_esp = mptr(self.idx_esp.value().unsafe_ptr())
        var ip_en = mptr(self.idx_en.value().unsafe_ptr())
        c.enqueue_function[gather_idx_kernel[E]](
            mptr(self.exp_nxt.value().unsafe_ptr()), ip_e, ip_en,
            grid_dim=nb, block_dim=TPB,
        )
        c.enqueue_function[gather_rows_kernel[Self.OBS, E]](
            mptr(self.exp_obs.dev.value().unsafe_ptr()), ip_e,
            mptr(self.t.bs.dev.value().unsafe_ptr()),
            grid_dim=_blocks(E * Self.OBS), block_dim=TPB,
        )
        c.enqueue_function[gather_rows_kernel[Self.ACT, E]](
            mptr(self.exp_act.dev.value().unsafe_ptr()), ip_e,
            mptr(self.t.ba.dev.value().unsafe_ptr()),
            grid_dim=_blocks(E * Self.ACT), block_dim=TPB,
        )
        c.enqueue_function[gather_rows_kernel[Self.OBS, E]](
            mptr(self.exp_obs.dev.value().unsafe_ptr()), ip_en,
            mptr(self.t.bsn.dev.value().unsafe_ptr()),
            grid_dim=_blocks(E * Self.OBS), block_dim=TPB,
        )
        c.enqueue_function[gather_rows_kernel[Self.OBS, E]](
            mptr(self.exp_obs.dev.value().unsafe_ptr()), ip_esp,
            mptr(self.t.bsp.dev.value().unsafe_ptr()),
            grid_dim=_blocks(E * Self.OBS), block_dim=TPB,
        )

    # ── rollout ──────────────────────────────────────────────────────────

    def _resample_lanes(mut self, force: Bool) raises:
        """Draw the per-lane mixture inputs and run the resample + projection.
        Eager path: host RNG offsets."""
        var c = self.ctx.value()
        comptime NG = Self.LANES * Self.D
        comptime NP = Self.LANES * 2
        box_muller_normal_gpu[NG](
            c, mptr(self._gauss_lane.dev.value().unsafe_ptr()),
            self._roll_seed, self._roll_off,
        )
        self._roll_off += UInt64(NG + (NG % 2))
        c.enqueue_function[uniform01_kernel[NP]](
            mptr(self._pick_lane.dev.value().unsafe_ptr()),
            self._roll_seed + 7, self._roll_off,
            grid_dim=_blocks(NP), block_dim=TPB,
        )
        self._roll_off += UInt64(2 * NP)
        c.enqueue_function[
            z_lane_resample_kernel[Self.D, Self.LANES, Self.ZBUF]
        ](
            mptr(self.z_lane.dev.value().unsafe_ptr()),
            mptr(self._gauss_lane.dev.value().unsafe_ptr()),
            mptr(self.zbuf.dev.value().unsafe_ptr()),
            LayoutTensor[DType.int32, Layout.row_major(1)](
                self.zbuf_fill.value()
            ),
            mptr(self._pick_lane.dev.value().unsafe_ptr()),
            Int32(self._act_iter),
            Int32(self.z_hold),
            Scalar[DT](self.zbuf_frac),
            Int32(1) if force else Int32(0),
            grid_dim=_blocks(Self.LANES), block_dim=TPB,
        )
        c.enqueue_function[project_sphere_kernel[Self.D, Self.LANES]](
            mptr(self.z_lane.dev.value().unsafe_ptr()),
            Scalar[DT](sqrt(Float64(Self.D))),
            grid_dim=_blocks(Self.LANES), block_dim=TPB,
        )
        if self.z_pin_on:
            c.enqueue_function[z_pin_kernel[Self.D, Self.LANES]](
                mptr(self.z_lane.dev.value().unsafe_ptr()),
                mptr(self.z_pin.dev.value().unsafe_ptr()),
                mptr(self.z_pin_mask.dev.value().unsafe_ptr()),
                grid_dim=_blocks(Self.LANES * Self.D), block_dim=TPB,
            )
        self._z_lane_resamples += 1

    def enable_z_pin(mut self) raises:
        """Allocate the pin buffers (mask zero: nothing pinned until the
        driver writes it). Before any capture."""
        ensure_t["gpu"](self.z_pin, Self.LANES * Self.D, self.ctx)
        ensure_t["gpu"](self.z_pin_mask, Self.LANES, self.ctx)
        for i in range(Self.LANES * Self.D):
            self.z_pin.data[i] = Scalar[DT](0)
        for i in range(Self.LANES):
            self.z_pin_mask.data[i] = Scalar[DT](0)
        self.z_pin.upload(self.ctx.value())
        self.z_pin_mask.upload(self.ctx.value())
        self.z_pin_on = True

    def _policy_into(
        mut self,
        obs_ptr: Pointer[Scalar[DT], MutAnyOrigin],
        action_ptr: Pointer[Scalar[DT], MutAnyOrigin],
        std: Float64,
    ) raises:
        """`action = clamp(pi_z(obs, z_lane) + std·n) · scale` for all lanes."""
        var c = self.ctx.value()
        comptime NA = Self.LANES * Self.ACT
        var src = obs_ptr
        if self.obs_ema.enabled:
            self.obs_ema.apply_into[Self.LANES](obs_ptr, self._obs_scratch)
            src = mptr(self._obs_scratch.dev.value().unsafe_ptr())
        c.enqueue_function[pack2_kernel[Self.OBS, Self.D, Self.LANES]](
            src,
            mptr(self.z_lane.dev.value().unsafe_ptr()),
            mptr(self._ain.dev.value().unsafe_ptr()),
            grid_dim=_blocks(Self.LANES * Self.A_IN), block_dim=TPB,
        )
        call_forward["gpu", Self.LANES](
            self.t.actor.online, TensorRefs[1, MutAnyOrigin](self._ain),
            self._pi, self.ctx,
        )
        if std > 0.0:
            box_muller_normal_gpu[NA](
                c, mptr(self._noise.dev.value().unsafe_ptr()),
                self._roll_seed + 3, self._roll_off,
            )
            self._roll_off += UInt64(NA + (NA % 2))
        c.enqueue_function[explore_action_kernel[NA]](
            action_ptr,
            mptr(self._pi.dev.value().unsafe_ptr()),
            mptr(self._noise.dev.value().unsafe_ptr()),
            Scalar[DT](std), self.action_scale,
            grid_dim=_blocks(NA), block_dim=TPB,
        )

    def select_action_batched[
        N_ENVS: Int
    ](
        mut self,
        obs: LayoutTensor[
            DT, Layout.row_major(N_ENVS, Self.AGENT_OBS_DIM), MutAnyOrigin
        ],
        action: LayoutTensor[
            DT, Layout.row_major(N_ENVS, Self.AGENT_ACT_DIM), MutAnyOrigin
        ],
        ao_scratch: LayoutTensor[
            DT, Layout.row_major(N_ENVS, 2 * Self.AGENT_ACT_DIM), MutAnyOrigin
        ],
        alp_scratch: LayoutTensor[
            DT, Layout.row_major(N_ENVS, Self.AGENT_ACT_DIM + 1), MutAnyOrigin
        ],
        step_idx: Int,
    ) raises:
        comptime assert N_ENVS == Self.LANES, (
            "FBOnlineAgent: the driver's N_ENVS must equal the agent's LANES"
        )
        _ = ao_scratch
        _ = alp_scratch
        # The lane z advances on EVERY iteration, warmup included, so the
        # stored z under random actions is a real draw and not a constant.
        self._resample_lanes(force=False)
        self._act_iter += 1
        if step_idx < self.learning_starts:
            warmup_uniform_batched["gpu", N_ENVS, Self.ACT](
                action, self.action_scale, self.ctx,
                self._warmup_seed, self._warmup_off,
            )
            return
        self._policy_into(obs.ptr, action.ptr, self.expl_std)

    def select_greedy_action_batched[
        N_ENVS: Int,
    ](
        mut self,
        ctx: Optional[DeviceContext],
        obs: LayoutTensor[
            DT, Layout.row_major(N_ENVS, Self.AGENT_OBS_DIM), MutAnyOrigin
        ],
        action: LayoutTensor[
            DT, Layout.row_major(N_ENVS, Self.AGENT_ACT_DIM), MutAnyOrigin
        ],
        ao_scratch: LayoutTensor[
            DT, Layout.row_major(N_ENVS, 2 * Self.AGENT_ACT_DIM), MutAnyOrigin
        ],
    ) raises:
        """Greedy under each lane's CURRENT `z` — no noise, no resample.
        ⚠ The env reward this scores is the env's own task; FB is not trained
        on it. Useful as a coverage signal, not as a zero-shot number — that
        needs `z_from_reward`, see `examples/fb/fb_eval_walker_online.mojo`."""
        comptime assert N_ENVS == Self.LANES, (
            "FBOnlineAgent: the driver's N_ENVS must equal the agent's LANES"
        )
        _ = ctx
        _ = ao_scratch
        self._policy_into(obs.ptr, action.ptr, 0.0)

    def select_greedy_action(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
    ) raises:
        """Single-row greedy action under lane 0's `z`. Host-list surface for
        `run_offpolicy_eval`; allocates per call — never on a hot path."""
        var c = self.ctx.value()
        var zl = Tensor.alloc(Self.LANES * Self.D)
        zl.ensure_gpu(c, Self.LANES * Self.D)
        c.enqueue_copy(zl.dev.value(), self.z_lane.dev.value())
        zl.download(c)
        var x = Tensor.alloc(Self.A_IN)
        for k in range(Self.OBS):
            x.data[k] = obs[k]
        if self.obs_ema.enabled:
            self.obs_ema.sync_host()
            self.obs_ema.apply_host_tensor(x)
        for k in range(Self.D):
            x.data[Self.OBS + k] = zl.data[k]
        x.upload(c)
        var y = Tensor.alloc_gpu(c, Self.ACT)
        call_forward["gpu", 1](
            self.t.actor.online, TensorRefs[1, MutAnyOrigin](x), y, self.ctx
        )
        y.download(c)
        for j in range(Self.ACT):
            var v = y.data[j]
            if v > Scalar[DT](1.0):
                v = Scalar[DT](1.0)
            elif v < Scalar[DT](-1.0):
                v = Scalar[DT](-1.0)
            action_out[j] = v * self.action_scale

    # ── record ───────────────────────────────────────────────────────────

    def record(
        mut self,
        ref obs: List[Scalar[DT]],
        ref action: List[Scalar[DT]],
        reward: Scalar[DT],
        ref next_obs: List[Scalar[DT]],
        done: Scalar[DT],
    ) raises:
        raise Error(
            "FBOnlineAgent.record: single-env host path not supported — this"
            " agent is driven by run_offpolicy_train_batched on a GPU env"
        )

    def _replay_add(
        mut self,
        ref obs: List[Scalar[DT]],
        ref action: List[Scalar[DT]],
        reward: Scalar[DT],
        ref next_obs: List[Scalar[DT]],
        done: Scalar[DT],
    ) raises:
        raise Error("FBOnlineAgent._replay_add: host path not supported")

    def _tracker_ptr(self) -> Pointer[EpisodeTracker, MutAnyOrigin]:
        return rebind[Pointer[EpisodeTracker, MutAnyOrigin]](
            Pointer(to=self.tracker)
        )

    def record_batch_gpu[
        N_ENVS: Int
    ](
        mut self,
        ctx: DeviceContext,
        prev_obs_dev: DeviceBuffer[DT],
        action_dev: DeviceBuffer[DT],
        reward_dev: DeviceBuffer[DT],
        obs_dev: DeviceBuffer[DT],
        done_dev: DeviceBuffer[DT],
    ) raises:
        """Append the lanes' transitions, each with the `z` its action was
        taken under. `reward` is NOT stored: FB never reads it.

        ⚠ `obs_dev` (the successor) is not stored either — it is recovered
        from the NEXT step's `prev_obs`, which is the same array. What is
        stored in its place is one boundary flag per row; see
        `ring_store_kernel` and `set_boundary`.
        """
        comptime assert N_ENVS == Self.LANES, (
            "FBOnlineAgent: the driver's N_ENVS must equal the agent's LANES"
        )
        _ = reward_dev
        _ = obs_dev
        comptime W = Self.STORE_OBS + Self.ACT + Self.D + 3
        ctx.enqueue_function[
            ring_store_kernel[
                Self.STORE_OBS, Self.ACT, Self.D, Self.CAP, Self.LANES
            ]
        ](
            mptr(prev_obs_dev.unsafe_ptr()),
            mptr(action_dev.unsafe_ptr()),
            mptr(done_dev.unsafe_ptr()),
            mptr(self.z_lane.dev.value().unsafe_ptr()),
            mptr(self.r_obs.dev.value().unsafe_ptr()),
            mptr(self.r_act.dev.value().unsafe_ptr()),
            mptr(self.r_term.dev.value().unsafe_ptr()),
            mptr(self.r_bnd.dev.value().unsafe_ptr()),
            mptr(self.r_age.dev.value().unsafe_ptr()),
            mptr(self.r_z.dev.value().unsafe_ptr()),
            Int32(self.pos),
            Int32(1) if self.bnd_next else Int32(0),
            Int32(self.age_next),
            grid_dim=_blocks(Self.LANES * W), block_dim=TPB,
        )
        self.bnd_next = False
        self.pos = (self.pos + Self.LANES) % Self.CAP
        self.size = self.size + Self.LANES
        if self.size > Self.CAP:
            self.size = Self.CAP
        # Device mirrors of the fill and the write head — what the captured
        # index draw reads. A host scalar baked into the graph would pin
        # sampling to the capture-time ring for the rest of the run.
        ctx.enqueue_function[_set_size_kernel](
            LayoutTensor[DType.int32, Layout.row_major(1)](
                self.size_dev.value()
            ),
            Int32(self.size),
            grid_dim=1, block_dim=1,
        )
        ctx.enqueue_function[_set_size_kernel](
            LayoutTensor[DType.int32, Layout.row_major(1)](
                self.pos_dev.value()
            ),
            Int32(self.pos),
            grid_dim=1, block_dim=1,
        )

    def set_boundary(mut self, b: Bool):
        """Tell the NEXT `record_batch_gpu` that the transition it records is
        the last one before the env is reset, so its successor row holds a
        post-reset observation rather than its own next state.

        Called by the driver immediately before `record_batch_gpu`, with
        `(s + 1) % T_EPISODE == 0` — the reset runs at the START of a step,
        so it is step `s` whose successor is broken, not step `s + 1`.

        A driver that never calls this records no boundaries at all, which is
        correct for an env that only terminates (the per-lane `done` flag is
        OR-ed in by the store kernel) and WRONG for one that resets on a
        schedule. Both G1 drivers call it.
        """
        self.bnd_next = b

    def set_age(mut self, a: Int):
        """Steps since this lane's reset, for the NEXT `record_batch_gpu`.

        Only read when `DERIVED_TAIL > 0`: it is what lets the gather know how
        far back the derived tail may look before a reset truncates it
        (docs §12.36). Every lane resets together under the G1 driver's
        scheduled `_rsi_reset`, so one scalar covers the batch — as `bnd` does.

        Cap it at whatever the tail's deepest back-step needs; larger values
        are equivalent and only waste range.
        """
        self.age_next = a

    def attach_tail_spec(mut self, ref spec: List[Int32]) raises:
        """The `DERIVED_TAIL x 4` table `derive_tail_kernel` reads.

        The agent never interprets it — the env builds it (§12.36). Uploaded
        once, resident for the run, so the captured training step never sees a
        moving pointer (`_tensor_upload_reallocates_every_call`).
        """
        comptime if Self.DERIVED_TAIL == 0:
            raise Error(
                "attach_tail_spec: this agent derives no tail"
                " (DERIVED_TAIL == 0)"
            )
        if len(spec) != Self.DERIVED_TAIL * 4:
            raise Error(
                "attach_tail_spec: expected " + String(Self.DERIVED_TAIL * 4)
                + " int32 (4 per derived column), got " + String(len(spec))
            )
        var c = self.ctx.value()
        var d = c.enqueue_create_buffer[DType.int32](Self.DERIVED_TAIL * 4)
        var h = c.enqueue_create_host_buffer[DType.int32](Self.DERIVED_TAIL * 4)
        for i in range(Self.DERIVED_TAIL * 4):
            h[i] = spec[i]
        c.enqueue_copy(d, h)
        c.synchronize()
        self.tail_spec = d^

    def record_batch_gpu_nstep[
        N_ENVS: Int, NS: Int
    ](
        mut self,
        ctx: DeviceContext,
        mut nstep_buf: GPUNStepBuffer[
            NS, Self.AGENT_OBS_DIM, Self.AGENT_ACT_DIM, N_ENVS,
        ],
        prev_obs_dev: DeviceBuffer[DT],
        action_dev: DeviceBuffer[DT],
        reward_dev: DeviceBuffer[DT],
        obs_dev: DeviceBuffer[DT],
        done_dev: DeviceBuffer[DT],
    ) raises:
        raise Error(
            "FBOnlineAgent: n-step is meaningless for FB (no reward in the"
            " loss); run the driver with NS=1"
        )

    # ── the training step ────────────────────────────────────────────────

    def _sample_batch(mut self) raises:
        """Assemble `FBTrainer`'s owned batch: expert rows first (if a store
        is attached), the ring for the rest, then the observation
        normaliser if enabled. All device; capture-safe. The branches are on
        `_expert_n` and `obs_ema.enabled`, both fixed before any capture."""
        self._gather_batch()
        self._normalize_batch()

    def _gather_batch(mut self) raises:
        comptime if Self.EXPERT_ROWS > 0:
            if self._expert_n > 0:
                self._gather_expert()
                self._gather_ring[Self.RING_ROWS](Self.EXPERT_ROWS)
                return
        self._gather_ring[Self.BATCH](0)

    def _normalize_batch(mut self) raises:
        """The reference's order: the batch's `s` and `s'` update the running
        statistics (train mode), then `s`, `s'`, `s+` are normalised with the
        updated statistics (eval mode). `a` and `z` are untouched."""
        if not self.obs_ema.enabled:
            return
        self.obs_ema.update[Self.BATCH](self.t.bs)
        self.obs_ema.update[Self.BATCH](self.t.bsn)
        self.obs_ema.apply[Self.BATCH](self.t.bs)
        self.obs_ema.apply[Self.BATCH](self.t.bsn)
        self.obs_ema.apply[Self.BATCH](self.t.bsp)

    def _relabel_z(mut self) raises:
        """`bz` holds the stored z; keep `keep_frac` of rows, overwrite the
        rest from the mixture over `B(bsp)`; project every row."""
        var c = self.ctx.value()
        self.t.embed_sp()
        gaussian_dev_t["gpu", Self.BATCH * Self.D](
            self.gauss, self._train_seed + 5, self.rng_dev.value(), self.ctx
        )
        var off_lt = LayoutTensor[DType.uint64, Layout.row_major(1), MutAnyOrigin](
            mptr(self.rng_dev.value().unsafe_ptr())
        )
        comptime NP = Self.BATCH * 3
        c.enqueue_function[uniform01_dev_kernel[NP]](
            mptr(self.pick.dev.value().unsafe_ptr()),
            self._train_seed + 9, off_lt,
            grid_dim=_blocks(NP), block_dim=TPB,
        )
        c.enqueue_function[advance_rng_offset_kernel[2 * NP]](
            off_lt, grid_dim=1, block_dim=1
        )
        var keep_from = Int32(Self.EXPERT_ROWS) if self._expert_n > 0 else Int32(0)
        c.enqueue_function[z_relabel_kernel[Self.D, Self.BATCH]](
            mptr(self.t.bz.dev.value().unsafe_ptr()),
            mptr(self.gauss.dev.value().unsafe_ptr()),
            mptr(self.t.b_sp.dev.value().unsafe_ptr()),
            mptr(self.pick.dev.value().unsafe_ptr()),
            Scalar[DT](self.keep_frac), Scalar[DT](self.uniform_frac),
            keep_from,
            grid_dim=_blocks(Self.BATCH), block_dim=TPB,
        )
        c.enqueue_function[project_sphere_kernel[Self.D, Self.BATCH]](
            mptr(self.t.bz.dev.value().unsafe_ptr()),
            Scalar[DT](sqrt(Float64(Self.D))),
            grid_dim=_blocks(Self.BATCH), block_dim=TPB,
        )

    def _push_zbuf(mut self) raises:
        var c = self.ctx.value()
        var head = LayoutTensor[DType.int32, Layout.row_major(1)](
            self.zbuf_head.value()
        )
        var fill = LayoutTensor[DType.int32, Layout.row_major(1)](
            self.zbuf_fill.value()
        )
        c.enqueue_function[zbuf_push_kernel[Self.D, Self.BATCH, Self.ZBUF]](
            mptr(self.zbuf.dev.value().unsafe_ptr()),
            mptr(self.t.bz.dev.value().unsafe_ptr()),
            head,
            grid_dim=_blocks(Self.BATCH * Self.D), block_dim=TPB,
        )
        c.enqueue_function[zbuf_advance_kernel[Self.BATCH, Self.ZBUF]](
            head, fill, grid_dim=1, block_dim=1,
        )

    def _train_kernels(mut self) raises:
        """The pure device sequence: sample → relabel → FB step → ZBuffer."""
        self._sample_batch()
        self._relabel_z()
        _ = self.t.train_step(want_loss=False)
        self._push_zbuf()
        # mean|a| over the POLICY'S rows only — the expert rows are not its.
        comptime if Self.EXPERT_ROWS > 0:
            if self._expert_n > 0:
                comptime lr = Layout.row_major(Self.RING_ROWS * Self.ACT)
                self._mean_abs_action_dev.accumulate_gpu_abs_lt[
                    Self.RING_ROWS * Self.ACT
                ](self.t.ba.lt_at["gpu", lr](Self.EXPERT_ROWS * Self.ACT))
                return
        comptime lba = Layout.row_major(Self.BATCH * Self.ACT)
        self._mean_abs_action_dev.accumulate_gpu_abs_lt[Self.BATCH * Self.ACT](
            self.t.ba.lt["gpu", lba]()
        )

    def train_step(mut self, step_idx: Int) raises -> Bool:
        if step_idx < self.learning_starts or self.size < Self.BATCH:
            return False
        self._train_kernels()
        self._update_count += 1
        self._total_train_steps += 1
        return True

    def train_device_kernels(mut self) raises:
        """Capture body. The driver enters only past `learning_starts_count`,
        and `make` requires `learning_starts >= BATCH`, so the ring is never
        sampled below one batch."""
        self._train_kernels()

    def note_train_update(mut self):
        self._update_count += 1
        self._total_train_steps += 1

    def learning_starts_count(self) -> Int:
        return self.learning_starts

    def total_train_steps(self) -> Int:
        return self._total_train_steps

    # ── diagnostics ──────────────────────────────────────────────────────

    def peek_losses(
        mut self,
        mut measure: Float64, mut ortho: Float64, mut actor: Float64,
        mut f_norm: Float64, mut b_norm: Float64,
        mut fb_quad: Float64, mut fb_anchor: Float64, mut m_mean: Float64,
        mut q_fb_abs: Float64,
    ) raises:
        """The last step's losses, RECOMPUTED from the trainer's live buffers.

        `train_step(want_loss=False)` skips the reductions and the D2H; the
        activations it leaves behind (`f1o`, `b_sn`, `m_target`,
        `acc_lam`) are the pre-update ones of that step, so re-running
        the two loss reductions over them returns exactly what
        `want_loss=True` would have — without ever putting a sync inside the
        captured sequence. The gradient outputs go to the scratch the next
        step overwrites. D2Hs several buffers: FLUSH CADENCE ONLY.
        """
        measure = 0.0
        ortho = 0.0
        actor = 0.0
        f_norm = 0.0
        b_norm = 0.0
        fb_quad = 0.0
        fb_anchor = 0.0
        m_mean = 0.0
        q_fb_abs = 0.0
        if self.t.steps == 0:
            return
        var c = self.ctx.value()
        var q1 = Float64(0)
        var a1 = Float64(0)
        var q2 = Float64(0)
        var a2 = Float64(0)
        var l1 = fb_measure_loss_into["gpu", Self.D, Self.BATCH](
            self.t.ws1, self.t.f1o, self.t.b_sn, self.t.m_target,
            self.t.g_f1, self.t.g_bsn1, q1, a1, True, self.ctx,
        )
        var l2 = fb_measure_loss_into["gpu", Self.D, Self.BATCH](
            self.t.ws2, self.t.f2o, self.t.b_sn, self.t.m_target,
            self.t.g_f2, self.t.g_bsn2, q2, a2, True, self.ctx,
        )
        measure = 0.5 * (l1 + l2)
        # The MEAN over the two F heads, and nothing else — these read directly
        # against `train_log.txt`'s `fb_offdiag` / `fb_diag` at the same
        # timestep. ⚠ There used to be a SECOND 0.5 here, "halving into the
        # reference's scale", and it was wrong: `agent.py:243` reduces
        # `diff` over BOTH heads (`Ms` is `num_parallel x B x B`) while
        # dividing by `off_diag_sum = B*(B-1)`, a SINGLE matrix's count — so
        # their leading 0.5 already IS the per-head mean at num_parallel 2.
        # Likewise `agent.py:244`'s `* Ms.shape[0]` is `num_parallel` = 2, not
        # BATCH. Both of our metrics were logged at exactly HALF scale, which
        # is what §12.21 read as "the magnitudes do not match" (docs §12.27).
        fb_quad = 0.5 * (q1 + q2)
        fb_anchor = 0.5 * (a1 + a2)
        # `M1` in the reference's log: the mean ENTRY of `M = F·B(s+)^T`. It
        # sits at ~49 and flat there from 2.3 M steps to 200 M, which makes it
        # the cheapest check that our M is not drifting in scale.
        m_mean = mean_t["gpu", Self.BATCH * Self.BATCH](
            self.t.ws1.m, self.t.acc, self.ctx
        )
        ortho = fb_ortho_loss_into["gpu", Self.D, Self.BATCH](
            self.t.wso, self.t.b_sn, self.t.g_bsn_o, True, self.ctx,
        )
        self.t.acc_lam.download(c)
        actor = -Float64(self.t.acc_lam.data[0])
        # `scale_reg`'s weight. Beside `actor` (= -mean(Q_fb)) this makes the
        # ratio `mean|Q_fb| / |mean Q_fb|` readable — the factor by which the
        # CPR style term was under-weighted before §12.15.
        self.t.acc_mag.download(c)
        q_fb_abs = Float64(self.t.acc_mag.data[0])
        var fn2 = mean_sq_t["gpu", Self.BATCH * Self.D](
            self.t.f1o, self.t.acc, self.ctx
        )
        var bn2 = mean_sq_t["gpu", Self.BATCH * Self.D](
            self.t.b_sn, self.t.acc, self.ctx
        )
        f_norm = sqrt(fn2 * Float64(Self.D))
        b_norm = sqrt(bn2 * Float64(Self.D))

    def flush_metrics_through_logger[
        L: Logger
    ](
        mut self,
        logger: Optional[Pointer[L, MutAnyOrigin]],
        step: Int,
    ) raises:
        var measure = Float64(0)
        var ortho = Float64(0)
        var actor = Float64(0)
        var fnorm = Float64(0)
        var bnorm = Float64(0)
        var fbq = Float64(0)
        var fba = Float64(0)
        var mmean = Float64(0)
        var qabs = Float64(0)
        self.peek_losses(
            measure, ortho, actor, fnorm, bnorm, fbq, fba, mmean, qabs
        )
        var gf1 = Float64(0)
        var gf2 = Float64(0)
        var gb = Float64(0)
        self.t.read_grad_norms(gf1, gf2, gb)
        var gv = Float64(0)
        var gt = Float64(0)
        self.t.read_actor_grad_split(gv, gt)
        var maa = Float64(self._mean_abs_action_dev.read["gpu"]())
        self._mean_abs_action_dev.reset["gpu"]()
        var n_upd = self._update_count
        self._update_count = 0
        if Bool(logger):
            var names = List[String]()
            var vals = List[Float64]()
            names.append(String("fb_measure_loss")); vals.append(measure)
            names.append(String("fb_ortho_loss")); vals.append(ortho)
            names.append(String("policy_loss")); vals.append(actor)
            names.append(String("fb_offdiag")); vals.append(fbq)
            names.append(String("fb_diag")); vals.append(fba)
            names.append(String("fb_m1")); vals.append(mmean)
            names.append(String("fb_q_abs_mean")); vals.append(qabs)
            names.append(String("f_norm")); vals.append(fnorm)
            names.append(String("b_norm")); vals.append(bnorm)
            names.append(String("b_norm_deficit"))
            vals.append(sqrt(Float64(Self.D)) - bnorm)
            names.append(String("fb_ortho_q"))
            vals.append(ortho + bnorm * bnorm)
            names.append(String("f1_grad_norm")); vals.append(gf1)
            names.append(String("f2_grad_norm")); vals.append(gf2)
            names.append(String("b_grad_norm")); vals.append(gb)
            names.append(String("action_abs_mean")); vals.append(maa)
            names.append(String("actor_grad_value")); vals.append(gv)
            names.append(String("actor_grad_total")); vals.append(gt)
            names.append(String("buffer_size")); vals.append(Float64(self.size))
            names.append(String("expert_rows"))
            vals.append(Float64(Self.EXPERT_ROWS if self._expert_n > 0 else 0))
            names.append(String("train_steps"))
            vals.append(Float64(self._total_train_steps))
            names.append(String("n_updates"))
            vals.append(Float64(n_upd))
            logger.value()[].log_scalars(names, vals, step)
        print(
            "   [fb] step", step, " measure", measure, " ortho", ortho,
            " actor", actor, " |F|", fnorm, " |B|", bnorm, " gF", gf1,
            " mean|a|", maa, " gA", gv, "->", gt, " replay", self.size,
        )

    # ── checkpoint ───────────────────────────────────────────────────────

    def save_state(mut self, path: String) raises:
        """`FBTrainer`'s own format — the offline eval scripts load it — plus
        the `.norm` sidecar when the normaliser is on (`ObsNorm`'s format,
        so `ObsNorm.try_load` applies it at eval)."""
        self.t.save_state(path)
        if self.obs_ema.enabled:
            self.obs_ema.save(path + ".norm")

    def load_state(mut self, path: String) raises:
        self.t.load_state(path)
        if self.obs_ema.enabled:
            self.obs_ema.load(path + ".norm")
