"""The G1 tracking protocol, natively — no Python, no `references/`.

`tools/g1/bfm_zero_tracking_oracle.py` is the ORACLE this is gated against
(`examples/g1/bfm_zero_eval_tracking.mojo` checks every metric of every
segment it scores, every run). It is not the supported path: the protocol
needs nothing but the store, so the trainer can score itself IN THE LOOP
rather than the run being scored by hand hours after it ended.

What used to be Python and is now here:

  `Protocol.n_segments`  -> `g1_n_segments`, pure store arithmetic. The oracle
                            prefers the released CSV's count when it has one
                            and falls back to exactly this; its own comment
                            records that the two agree on all 40 clips.
  `Episode.first_row`    -> `g1_segment_row`
  `Episode.init_state`   -> `G1RsiTable` already stores qpos verbatim and qvel
                            with the root angular velocity rotated into the
                            body frame, which IS `init_state`. Reused, not
                            reimplemented.
  `compute_metrics`      -> `g1_track_metrics`. `distance` = mean_t ‖q−q*‖₂,
                            `proximity` = the bound-2 / margin-2 ramp, `emd` =
                            `core/assignment.mojo` (the OT plan between equal
                            uniform clouds is a permutation, so the oracle's
                            `linear_sum_assignment` and ours must agree
                            EXACTLY, not approximately).

⚠ ONE definition of the dims and ONE rollout loop, used by both the standalone
eval and the trainer. They used to be two copies with a comment asking the
reader to keep `H`/`L`/`HB`/`D` in sync by hand, and a mismatch only showed up
as a checkpoint that would not load.
"""

from std.math import sqrt, abs

from noeira.nn.constants import DT
from noeira.nn.core.module import Module
from noeira.nn.core.tensor import Tensor
from noeira.core.cont_action import ContAction
from noeira.core.assignment import emd_uniform
from noeira.deep_agents.fb.trainer import FBTrainer
from noeira.deep_agents.fb.obs_norm import ObsNorm
from noeira.envs.robots import UnitreeG1
from noeira.envs.robots.unitree_g1_history import (
    G1_HIST_DIM, G1_HIST_LEN, G1_N_ACT, G1_LAST_ACTION_DIM,
    g1_hist_key_offset, g1_hist_key_dim, g1_hist_key_state_offset,
    G1ActorObs,
)
from noeira.envs.robots.unitree_g1_pd import G1_NORMALIZE_TO, G1_ACTION_CLIP
from noeira.envs.robots.unitree_g1_xml import (
    UnitreeG1Model, UNITREE_G1_STATE_DIM, UNITREE_G1_PRIV_DIM,
    UNITREE_G1_OBS_DIM,
)
from noeira.envs.robots.unitree_g1_rsi import G1RsiTable, G1_RSI_NQ, G1_RSI_NV


# ── the G1 network dims, in ONE place ─────────────────────────────────────
# The trainer and the standalone eval must agree on these EXACTLY or a
# checkpoint will not load into the eval — they used to be declared in both
# files with a comment asking the reader to keep them in sync by hand.
#
# BFM-Zero's own tower is `hidden_dim 2048, hidden_layers 6`
# (`released/new_model/config.yaml`), 299 M parameters across F x2, Q_D x2,
# actor, B and D. Measured budget on a 32 GiB card (docs §12.21):
#
#     weights + target + Adam   4.46 GiB
#     activations (rough)       2.62 GiB
#     replay ring, expert and tracking tables   11.22 GiB
#     ----------------------------------------------------
#     ~18.3 GiB of 32
#
# ⚠ The REPLAY RING is the big term, not the model: 9.98 GiB at CAP = 2 M,
# because every transition stores `obs` TWICE (2*527 + 29 + 256 + 1 floats =
# 5360 B). The reference's own CAP of 5.12 M would be 25.6 GiB of ring alone
# and does NOT fit beside this tower — that scale-down is forced by the card.
#
# ⚠ 2048/6 DOES NOT FIT ON A 32 GB CARD. Measured, not estimated (§12.22):
# `NOEIRA_ALLOC_TRACE=1` shows 30.24 GiB allocated and STILL CLIMBING when it
# died, on a 31.8 GiB card. Scaling that MEASURED breakdown:
#
#     H=2048 L=6  CAP 2.0M   32.73 GiB   -1.73   OOM
#     H=2048 L=6  CAP 1.5M   30.24 GiB   +0.76   OOM (it wanted more)
#     H=1536 L=4  CAP 2.0M   23.58 GiB   +7.42   <- this
#     H=1024 L=3  CAP 2.0M      ~13 GiB           (runs 1-6)
#
# ⚠ THE ids IN THAT TRACE ARE NOT ALL DISTINCT BUFFERS. `_alloc_trace`'s id is
# `Pointer(to=self)`, stable only for a MODULE FIELD; stack temporaries reuse
# addresses, so "48 allocations under one id" was 48 different buffers, all of
# ONE size — and `ensure_gpu` is a no-op at the same size, so it cannot have
# been a realloc. Aggregating by max-per-id UNDERCOUNTS by ~4.7 GiB and makes a
# capacity wall look like a leak. It is not one.
comptime G1_D: Int = 256          # z_dim
comptime G1_H: Int = 2048         # hidden_dim    — the reference's own (config.json f.hidden_dim)
comptime G1_L: Int = 6            # hidden_layers — §12.30's null is VOID, see §12.40
comptime G1_HB: Int = 256         # backward map hidden
comptime G1_HD: Int = 1024        # discriminator hidden


# ── the protocol's constants (oracle lines 95-97) ─────────────────────────
comptime G1_SEG_ROWS: Int = 499      # ceil((300 - 1) / 30 / 0.02)
comptime G1_SEG_STRIDE: Int = 500    # 10 s at 50 Hz
comptime G1_PROX_BOUND: Float64 = 2.0
comptime G1_PROX_MARGIN: Float64 = 2.0


def g1_n_segments(ep_len: Int) -> Int:
    """Every 500-row start whose 499 rows fit inside the clip."""
    if ep_len < G1_SEG_ROWS:
        return 0
    return (ep_len - G1_SEG_ROWS) // G1_SEG_STRIDE + 1


def g1_segment_pick(n_avail: Int, n_take: Int, k: Int) -> Int:
    """The `k`-th of `n_take` segments spread evenly over `n_avail`.

    ⚠ Scoring the FIRST `n_take` segments of every clip is a BIASED sample of
    the clip, and the bias is not small: the same checkpoint reads 1.398 over
    40 segments (one per clip) and 1.587 over 200 — the later windows of a
    motion are harder than its opening seconds. The reference evaluates all
    862 10 s windows (`humanoidverse_tracking_eval.csv`), so a truncated eval
    that starts every clip from row 0 is not comparable to its numbers at any
    `n_take` (docs §12.29).

    Stratified MIDPOINT sampling: `((2k+1)·n_avail) // (2·n_take)`. It is the
    identity when `n_take == n_avail`, so full coverage still scores exactly
    every segment, and at `n_take == 1` it picks the MIDDLE of the clip rather
    than its first ten seconds.

    Written here, once, because both the in-loop eval in
    `bfm_zero_train_gpu.mojo` and the oracle-gated standalone eval select
    segments — and a selection rule that drifts between them would make the
    two report different numbers for the same checkpoint.
    """
    if n_take >= n_avail:
        return k
    return ((2 * k + 1) * n_avail) // (2 * n_take)


def g1_segment_row(ep_offset: Int, seg: Int) -> Int:
    """The store row this segment starts at."""
    return ep_offset + seg * G1_SEG_STRIDE


@fieldwise_init
struct G1TrackScore(Copyable):
    """One segment's numbers, or a mean over several."""

    var distance: Float64
    var emd: Float64
    var proximity: Float64
    var n: Int


def g1_track_metrics[
    ACT: Int
](ref ach: List[Float64], ref tgt: List[Float64], T: Int) raises -> G1TrackScore:
    """`compute_metrics`, formula for formula, on `(T, ACT)` joint angles."""
    var sum_d = 0.0
    var sum_p = 0.0
    for j in range(T):
        var s = 0.0
        for k in range(ACT):
            var d = ach[j * ACT + k] - tgt[j * ACT + k]
            s += d * d
        var dist = sqrt(s)
        sum_d += dist
        # `inb + ((bound + margin - dist)/margin)·(~inb)·(~outb)`
        if dist <= G1_PROX_BOUND:
            sum_p += 1.0
        elif dist <= G1_PROX_BOUND + G1_PROX_MARGIN:
            sum_p += (G1_PROX_BOUND + G1_PROX_MARGIN - dist) / G1_PROX_MARGIN
    var tf = Float64(T)
    return G1TrackScore(
        sum_d / tf, emd_uniform(ach, tgt, T, ACT), sum_p / tf, 1
    )


def g1_project_z[Dz: Int](mut z: Tensor, row: Int):
    """Onto the radius-sqrt(d) sphere."""
    var s = 0.0
    for k in range(Dz):
        var v = Float64(z.data[row * Dz + k])
        s += v * v
    var scale = sqrt(Float64(Dz)) / sqrt(s + 1e-12)
    for k in range(Dz):
        z.data[row * Dz + k] = Scalar[DT](
            Float64(z.data[row * Dz + k]) * scale
        )


def g1_score_segment[
    FNET: Module, BNET: Module, ANET: Module,
    OBS: Int, ACT: Int, D: Int, BATCH: Int,
](
    mut t: FBTrainer[FNET, BNET, ANET, OBS, ACT, D, BATCH, "cpu"],
    mut env: UnitreeG1[DType.float64],
    ref rsi: G1RsiTable,
    ref st: List[Scalar[DType.float32]],
    ref pv: List[Scalar[DType.float32]],
    ref qpos_col: List[Scalar[DType.float32]],
    ref norm: Optional[ObsNorm[OBS]],
    r0: Int,
    mut ach: List[Float64],
    mut tgt: List[Float64],
    mut b_in: Tensor,
    mut b_out: Tensor,
    mut z_seg: Tensor,
    mut obs_t: Tensor,
    mut z1: Tensor,
    mut act_out: Tensor,
    render: Bool = False,
    frame_delay_ms: Int = 16,
    z_horizon: Int = 1,
) raises -> G1TrackScore:
    """`tracking_inference` for ONE segment starting at store row `r0`.

    `render=True` draws each step through the env's own renderer. That flag
    exists so the VIEWER IS THIS FUNCTION — `bfm_zero_policy_viewer.mojo` opens
    a window and calls exactly the loop the EMD came out of. A viewer with its
    own copy of the rollout would be a second place for the z schedule, the
    history rules and the action chain to drift, and it would then be showing
    something the number never measured (`_a_rule_written_inline_twice_drifts`).
    The caller owns `init_renderer` / `close`; this draws AND pumps. The
    pump matters: `check_renderer_quit` is the renderer's only
    `poll_event` site, and a segment is 499 frames, so a caller that
    polled between segments left the window unserviced for ten seconds
    at a time. Events cannot reach the physics — they move the camera —
    so the score is identical with `render` on or off.

    `z_t = project(sum_{h=1}^{z_horizon} B(row t+h))` — at the default
    `z_horizon=1` a SINGLE row, which is neither the reference's look-ahead sum
    nor the mean of eight our training rollouts use (12.45). Then reset to row 0
    and `T-1` mean-action steps.
    """
    comptime T = G1_SEG_ROWS
    comptime NQ = UnitreeG1Model.NQ
    comptime NV = UnitreeG1Model.NV

    for j in range(T):
        for k in range(UNITREE_G1_STATE_DIM):
            b_in.data[j * OBS + k] = Scalar[DT](
                st[(r0 + j) * UNITREE_G1_STATE_DIM + k]
            )
        for k in range(UNITREE_G1_PRIV_DIM):
            b_in.data[j * OBS + UNITREE_G1_STATE_DIM + k] = Scalar[DT](
                pv[(r0 + j) * UNITREE_G1_PRIV_DIM + k]
            )
    # ⚠ `b_in` is OBS-wide but only the stored head means anything here: `b`'s
    # reference filter is `state + privileged_state` (§12.34). Zero the derived
    # tail rather than leave it holding the previous segment's numbers — `b`
    # ignores it today, and a live tensor full of stale values is one widened
    # filter away from being read.
    comptime if OBS > UNITREE_G1_OBS_DIM:
        for j in range(T):
            for k in range(UNITREE_G1_OBS_DIM, OBS):
                b_in.data[j * OBS + k] = Scalar[DT](0.0)
    if norm:
        norm.value().apply_rows(b_in, T)
    t.backward_embed[T](b_in, b_out)
    # The tracking prompt. `z_horizon=1` is a SINGLE look-ahead row and is the
    # default because every number recorded before 12.45 was measured with it;
    # changing the default would silently reprice the whole history.
    #
    # BFM-Zero's own prompt is a look-ahead SUM, `z_t = sum_{t'=t}^{t+H} B(s_t')`
    # (arXiv 2511.04131), and our TRAINING rollouts use the mean of eight — so
    # H=1 agrees with neither. The sum runs to `T-1` and stops there rather than
    # wrapping or clamping to a repeated final row: a segment's last steps
    # genuinely have less future to aim at, and inventing rows for them would
    # make the tail of every segment score against a motion the clip does not
    # contain. The projection is what makes the SUM and the MEAN the same
    # prompt, so this does not need a 1/H.
    var zh = z_horizon if z_horizon > 0 else 1
    for j in range(T - 1):
        for k in range(D):
            z_seg.data[j * D + k] = b_out.data[(j + 1) * D + k]
        for h in range(1, zh):
            var r = j + 1 + h
            if r > T - 1:
                break
            for k in range(D):
                z_seg.data[j * D + k] = (
                    z_seg.data[j * D + k] + b_out.data[r * D + k]
                )
        g1_project_z[D](z_seg, j)

    # reset: the RSI row IS `init_state()` (qvel already body-frame)
    var qp = List[Float64](length=NQ, fill=0.0)
    var qv = List[Float64](length=NV, fill=0.0)
    var base = r0 * (G1_RSI_NQ + G1_RSI_NV)
    for i in range(NQ):
        qp[i] = Float64(rsi.rows.data[base + i])
    for i in range(NV):
        qv[i] = Float64(rsi.rows.data[base + G1_RSI_NQ + i])
    env.set_state(qp, qv)

    for i in range(NQ):
        qp[i] = Float64(env.d.qpos.data[i])
    var nrec = 0
    # latched: once Escape / the close box has fired, stop PACING and
    # drawing so the caller gets control back in well under a second,
    # while the rollout still runs to T and returns a complete score
    var quit_seen = False
    # ⚠ TWO RECORD SITES, AND THEY MUST STAY IN STEP — the `nrec != T` raise
    # below is the guard that caught exactly that bug once already.
    for k in range(ACT):
        ach[nrec * ACT + k] = qp[7 + k]
    nrec += 1
    # ── `last_action` + `history_actor`, the actor's other 401 (§12.34) ──
    # Host-side and single-lane here, but the SAME rules as the batched
    # rollout: zero at reset, the reset observation is never pushed, the push
    # happens AFTER the step's history is read and BEFORE `last_action` is
    # updated — so the newest `actions` entry is the action applied one step
    # earlier, not the one about to be applied.
    var aobs = G1ActorObs()
    for step in range(T - 1):
        var o = env.get_obs_list()
        aobs.fill[OBS=OBS](o, obs_t)
        if norm:
            norm.value().apply_row(obs_t)
        for k in range(D):
            z1.data[k] = z_seg.data[step * D + k]
        t.act[1](obs_t, z1, act_out)
        # the env's action dim is the model's, not the caller's `ACT` — they
        # are the same number, but only one of them is what `step` accepts
        var a = ContAction[UnitreeG1Model.ACTION_DIM]()
        for k in range(UnitreeG1Model.ACTION_DIM):
            var v = Float64(act_out.data[k])
            if v > 1.0:
                v = 1.0
            elif v < -1.0:
                v = -1.0
            a.data[k] = v
        # push BEFORE `last_a` is updated: the newest `actions` entry is the
        # PREVIOUS step's action (the oracle's `_push` then `last_action =`)
        comptime if OBS > UNITREE_G1_OBS_DIM:
            aobs.push(o, act_out)
        _ = env.step(a)
        if render and not quit_seen:
            # ⚠ THE PUMP IS HERE, NOT ONCE PER SEGMENT. `check_renderer_quit`
            # is the renderer's ONLY `poll_event` site — `render_frame` draws but
            # never services the queue. A segment is 499 frames (≈10 s at 50 Hz),
            # so pumping only between segments left the window unserviced for ten
            # seconds at a time: macOS stops presenting a window whose events go
            # unread, `SDL_WaitAndAcquireGPUSwapchainTexture` then hands back NULL
            # and `end_frame` submits an EMPTY command buffer — the compositor
            # keeps showing whatever was in the surface. That is the colour
            # flashing, and it is also why Escape did nothing mid-segment.
            # Pumping cannot touch the rollout: it moves the CAMERA, never `d`.
            quit_seen = env.check_renderer_quit()
            env.render_frame()
            env.renderer_delay(frame_delay_ms)
            # SPACE holds the picture. The renderer's own pause flag is
            # DISPLAY-ONLY — it draws a badge and freezes a frame counter, it
            # does not gate the caller — so without this loop Space drew
            # "PAUSED" over a robot that kept walking. Holding here cannot
            # change the score: the rollout is a fixed sequence of T steps and
            # this only delays the next one.
            while (
                env.renderer_paused() and not quit_seen
                and env.is_renderer_open()
            ):
                quit_seen = env.check_renderer_quit()
                env.render_frame()
                env.renderer_delay(frame_delay_ms)
        for i in range(NQ):
            qp[i] = Float64(env.d.qpos.data[i])
        for k in range(ACT):
            ach[nrec * ACT + k] = qp[7 + k]
        nrec += 1
    if nrec != T:
        raise Error(
            "achieved rows " + String(nrec) + " != T " + String(T)
            + " — a record site lost its accumulation"
        )

    for j in range(T):
        for k in range(ACT):
            tgt[j * ACT + k] = Float64(qpos_col[(r0 + j) * NQ + 7 + k])
    return g1_track_metrics[ACT](ach, tgt, T)
