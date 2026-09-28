"""EfficientZeroV2 BATCHED **value-prefix** self-play driver (GPU × N CPU envs).

The value-prefix sibling of `selfplay_gpu_batched.mojo` (decision B1 / B1.1, see
`docs/EZV2_ATARI_PARITY.md`): identical batched collection + PER + wide-reanalyze
+ device-obs-ring pipeline, but `dyn` is the CONCRETE fused `EZDynVPNetAtari[ACT,
BINS]` (drop-in `[z|act]→[z'|vp]` search dynamics owning `dyn.dynz` + the LSTM
`dyn.rew`), and training runs `ezv2_unroll_train_step_gpu_vp` on those sub-modules
with a 6th (reward) optimizer + cumulative `value_prefix_from_rewards` targets +
the same PER + on-device obs slab. A generic-`DYN` comptime gate is impossible
(Mojo type-checks both `comptime if` branches), so this is a concrete-`dyn` copy.
Search-side `(h,c)` carry stays deferred (B1.1: zero-(h,c) reward head per node).

⚠️ TOOLCHAIN STATUS (2026-06-24): this driver currently triggers a BACKEND
codegen ICE (SIGILL, no source diagnostic) on the Apple/Metal toolchain. It is
NOT a logic error — every component is independently verified green: the VP train
step `ezv2_unroll_train_step_gpu_vp` with full PER + device-obs at B=4
(`test_ezv2_atari_value_prefix_unroll_gpu`), a real Gumbel search through the
fused `EZDynVPNetAtari` at B=1 and B=2 + train
(`test_ezv2_atari_value_prefix_integration_gpu`), and the entire SINGLE-env VP
driver `run_ezv2_gumbel_selfplay_gpu_vp` (`test_ezv2_atari_value_prefix_driver_
gpu`). Only this ~680-line aggregate ICEs — the larger fused-VP-dyn
monomorphization across the whole batched function exceeds a codegen limit (the
non-VP `run_ezv2_gumbel_selfplay_gpu_batched` of the same shape compiles; this is
the repo's known compile-bloat behavior, docs/ATARI_AUDIT / compile-RAM notes).
Use the single-env VP driver as the verified runnable path; expect this batched
driver to compile on the NVIDIA toolchain / a future Mojo. A function-split
refactor (extract the collection / reanalyze blocks into helpers) is the likely
fix on Apple. Smoke: `test_ezv2_atari_value_prefix_batched_gpu` (ICEs today).

Original (non-VP) recipe notes still apply:

Stage 6c-3 — the Atari-parity driver. The batched sibling of `selfplay_gpu.mojo`
(single-env), assembling the full EfficientZero-V2 recipe:

  * **N_ENVS CPU-emulated envs** stepped in lockstep through a
    `BatchedCpuDiscreteEnv` (host obs/action/reward/done) — the RGB-96 Atari
    `AtariEnv[2]` is CPU-only, so unlike the GPU MuZero batched driver only the
    search-input obs is H2D'd and the improved policy/value are D2H'd; actions
    and rewards stay host. Random no-op starts (`noop_max`) live in the env.
  * **Batched Gumbel search** over the on-device h/g/f nets (one launch covers
    all N_ENVS roots) via the MuZero GPU adapters.
  * **Prioritized replay** (`PrioritizedMCTSSequenceReplay`): each train step
    samples ∝ priorityᵅ, the unroll is weighted by per-sample IS weights, and
    fresh value-error priorities are written back (α=β=1, atari.yaml).
  * **EZv2 unroll train step** (MuZero BPTT + SimSiam consistency) on the
    resident GPU nets with any `Optimizer` (SGD for Atari) + warmup→const LR.
  * **Reanalyze** through the LIVE nets (like the single-env EZv2 driver — no
    lagging target nets, sidestepping the BatchNorm-running-stat hard-copy bug);
    `reanalyze_batch ≈ B` + `reanalyze_every = 1` gives the EZ ratio-1.0 regime.
    Runs on a SEPARATE wider planner (`REANA_W` roots/search) so a ratio-1.0
    re-target is `ceil(reanalyze_batch/REANA_W)` wide searches, not
    `reanalyze_batch/N_ENVS` narrow N_ENVS-root ones — the bottleneck fix (far
    fewer launches/syncs, real GPU occupancy). `REANA_W` defaults to `N_ENVS`
    (back-compat); set it to e.g. 64 for the Atari run.
  * **UTD 1:1** (`train_per_iter = N_ENVS` default) + batched greedy eval.

DEVIATION (documented, revisit if Pong diverges): the n-step value target
bootstraps from the **search-root value** (what the planner returns), whereas
EZ bootstraps the **value head** at s_{k+5} (`efficient_inference only_value`).
Reanalyze refreshes the stored root value with the current net, which is the
main coverage lever; the head-vs-root bootstrap gap is logged for Stage 8.

Run (GPU env required): see `tests/deep_agents/test_ezv2_atari_batched_smoke.mojo`.
"""

from std.math import exp, log
from std.memory import unsafe_memcpy, alloc, dealloc
from std.time import perf_counter_ns
from layout import Layout, LayoutTensor
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.nn.constants import DT
from noeira.nn.core.module import Module
from noeira.nn.optimizer.adam import Adam
from noeira.nn.optimizer.optimizer import Optimizer
from noeira.core.logger import Logger, NoOpLogger
from noeira.planners.tree_search import (
    GumbelGPUMCTS, SinglePlayer,
    RepresentationGPU, DynamicsGPU, PredictionGPU,
)

from ..training.batched_env import BatchedEnv
from .blocks import ezv2_unroll_train_step_gpu_vp
from .nets_atari import EZDynVPNetAtari, EZDynZNetAtari, EZ_LSTM_HIDDEN
from ..zero.mcts_adapters_mz import MZRepGPU, MZDynGPU, MZPredGPU
from ..zero.prioritized_sequence_replay_mcts import PrioritizedMCTSSequenceReplay
from ..zero.nstep_targets import value_prefix_from_rewards
from ..zero.temperature import visit_temperature
from ..muzero.selfplay_gpu_batched import _sample_action


def _a(n: Int) -> Pointer[Scalar[DT], MutAnyOrigin]:
    return alloc[Scalar[DT]]({count = n}).unsafe_leak().as_unsafe_any_origin()


def _ai(n: Int) -> Pointer[Int, MutAnyOrigin]:
    return alloc[Int]({count = n}).unsafe_leak().as_unsafe_any_origin()


def run_ezv2_gumbel_selfplay_gpu_batched_vp[
    ENV: BatchedEnv,
    REP: Module,
    PRED: Module,
    PROJM: Module,
    PREDH: Module,
    N_ENVS: Int,
    OBS: Int,
    ACT: Int,
    LATENT: Int,
    BINS: Int,
    NUM_SIMS: Int,
    MAX_NODES: Int,
    MAX_K: Int,
    CAP: Int,
    B: Int,
    K: Int,
    N: Int,
    REANA_W: Int = N_ENVS,
    OBS_STORE_DT: DType = DT,
    HORIZON: Int = 5,                   # lstm_horizon_len (value-prefix reset)
    L: Logger = NoOpLogger,
](
    ctx: DeviceContext,
    mut env: ENV,
    mut rep: REP,
    mut dyn: EZDynVPNetAtari[ACT, BINS],   # fused VP dynamics (search + train)
    mut pred: PRED,
    mut proj: PROJM,
    mut predh: PREDH,
    mut orep: Adam,
    mut odyn: Adam,                        # steps dyn.dynz
    mut orew: Adam,                        # steps dyn.rew (LSTM value-prefix head)
    mut opred: Adam,
    mut oproj: Adam,
    mut opredh: Adam,
    iterations: Int,
    learning_starts: Int = 256,         # in STORED STEPS
    train_per_iter: Int = N_ENVS,       # UTD 1:1 (grad steps == env steps)
    lr: Scalar[DT] = Scalar[DT](0.0),
    lr_warmup_iters: Int = 0,
    gamma: Scalar[DT] = Scalar[DT](0.997),
    v_min: Scalar[DT] = Scalar[DT](-300.0),
    v_max: Scalar[DT] = Scalar[DT](300.0),
    seed: UInt64 = 0,
    max_ep_steps: Int = 27000,
    value_coef: Scalar[DT] = Scalar[DT](0.25),
    consistency_coef: Scalar[DT] = Scalar[DT](2.0),
    temperature_decay_steps: Int = 0,
    reanalyze_every: Int = 0,
    reanalyze_batch: Int = B,
    eval_every: Int = 0,
    eval_episodes: Int = 5,
    eval_horizon: Int = 0,
    eval_env: Optional[Pointer[ENV, MutAnyOrigin]] = None,
    diag_every: Int = 0,
    report_every: Int = 0,
    logger: Optional[Pointer[L, MutAnyOrigin]] = None,
    verbose: Bool = False,
    diag_sync: Bool = False,
) raises -> Float64:
    comptime assert ENV.OBS_DIM == OBS, "batched EZv2: ENV.OBS_DIM must equal OBS"
    comptime DYNZ = EZDynZNetAtari[ACT]

    var planner = GumbelGPUMCTS[
        N_ENVS, ACT, LATENT, BINS, MAX_NODES, MAX_K, NUM_SIMS, SinglePlayer
    ](
        ctx, gamma=Float64(gamma), v_min=Float64(v_min), v_max=Float64(v_max),
        qnorm_per_node=False,
    )
    # Reanalyze runs a SEPARATE, WIDER planner (REANA_W roots/search) so the
    # ratio-1.0 re-target covers `reanalyze_batch` positions in
    # ceil(reanalyze_batch/REANA_W) wide searches instead of reanalyze_batch/N_ENVS
    # narrow N_ENVS-root ones — far fewer kernel launches + syncs and much better
    # GPU occupancy. Reuses the SAME nets/adapters (forward["gpu", B] is width-
    # generic; the nets' activation buffers lazy-grow to REANA_W on first use).
    # REANA_W == N_ENVS (default) is the same monomorphization as `planner` (no
    # extra compile), just a second runtime instance.
    var reana_planner = GumbelGPUMCTS[
        REANA_W, ACT, LATENT, BINS, MAX_NODES, MAX_K, NUM_SIMS, SinglePlayer
    ](
        ctx, gamma=Float64(gamma), v_min=Float64(v_min), v_max=Float64(v_max),
        qnorm_per_node=False,
    )
    var rep_a = MZRepGPU[OBS, LATENT, REP].make(rep)
    # the fused EZDynVPNetAtari is a drop-in dynamics → adapter unchanged.
    var dyn_a = MZDynGPU[LATENT, ACT, BINS, EZDynVPNetAtari[ACT, BINS]].make(dyn)
    var pred_a = MZPredGPU[LATENT, ACT, BINS, PRED].make(pred)

    # Device-resident obs ring (uint8 pixels): the training obs slab is gathered
    # on-device straight into scratch.d_obs, so neither the host dequant-build nor
    # the ~680 MB/step slab H2D happen (the pixel-obs bottleneck). PER + targets
    # stay on host.
    var rb = PrioritizedMCTSSequenceReplay[OBS, ACT, CAP, OBS_STORE_DT](
        ctx, seed=seed ^ UInt64(0xABCDEF),
        alpha=Scalar[DT](1.0), beta=Scalar[DT](1.0),
    )

    # search-input device obs (H2D'd from the CPU env each step) + host mirrors
    var d_obs = ctx.enqueue_create_buffer[DT](N_ENVS * OBS)
    var h_pol = _a(N_ENVS * ACT)
    var h_val = _a(N_ENVS)

    # training batch slabs (time-major). obs is NOT staged on host any more — the
    # device gather fills scratch.d_obs directly; only the small label slabs are
    # host (H2D'd by the train step). The per-(k,b) ring-slot index array drives
    # the gather.
    var t_act = List[Scalar[DT]](length=K * B, fill=0)
    var t_pol = List[Scalar[DT]](length=(K + 1) * B * ACT, fill=0)
    var t_val = List[Scalar[DT]](length=(K + 1) * B, fill=0)
    var t_rew = List[Scalar[DT]](length=K * B, fill=0)
    var t_cmask = _a(K * B)
    var t_isw = List[Scalar[DT]](length=B, fill=0)              # PER importance-sampling weights
    var t_prio = List[Scalar[DT]](length=B, fill=0)            # PER value-error priorities (writeback)
    var t_slots = List[Int](length=B, fill=0)         # PER sampled ring slots
    var l_parts = _a(4)
    var t_obs_dummy = List[Scalar[DT]](length=1, fill=0)      # unused obs_seq arg (obs_on_device=True)
    # gather slot index arrays: training [(K+1)*B] + reanalyze [REANA_W]
    var h_obs_slots = List[Int32](length=(K + 1) * B, fill=0)
    var d_obs_slots = ctx.enqueue_create_buffer[DType.int32]((K + 1) * B)
    var h_reana_slots = List[Int32](length=REANA_W, fill=0)
    var d_reana_slots = ctx.enqueue_create_buffer[DType.int32](REANA_W)

    # device obs slab the replay gathers into + the VP train step reads
    # (obs_dev). The value-prefix train step uses its own local scratch, so no
    # EZV2UnrollScratch is needed here.
    var d_train_obs = ctx.enqueue_create_buffer[DT]((K + 1) * B * OBS)

    # reanalyze scratch — REANA_W wide. obs gathered on-device into d_reana (no
    # host read_obs / H2D); improved policy/value D2H'd into host mirrors.
    var d_reana = ctx.enqueue_create_buffer[DT](REANA_W * OBS)
    var h_pol_w = List[Scalar[DT]](length=REANA_W * ACT, fill=0)
    var h_val_w = List[Scalar[DT]](length=REANA_W, fill=0)

    # per-env episode accumulators. obs uses a manually grown raw buffer (NOT a
    # List): List.resize/append on a 110592-wide obs reallocs+copies the whole
    # accumulated buffer every step → O(episode_len²) (the bottleneck). The raw
    # buffer doubles its capacity (amortized O(1)) and is REUSED across episodes
    # (cursor reset to 0, capacity retained → no reallocs after warmup). The
    # small label fields (act/rew/pol/val/tp/legal) stay Lists — a few appends/
    # step is negligible.
    var eo_buf = List[Pointer[Scalar[DT], MutAnyOrigin]]()
    var eo_cap = List[Int]()                  # capacity in ELEMENTS
    var e_act = List[List[Scalar[DT]]]()
    var e_rew = List[List[Scalar[DT]]]()
    var e_pol = List[List[Scalar[DT]]]()
    var e_val = List[List[Scalar[DT]]]()
    var e_tp = List[List[Scalar[DT]]]()
    var e_legal = List[List[Scalar[DT]]]()
    var ep_len = List[Int]()
    var ep_return = List[Float64]()
    for _ in range(N_ENVS):
        eo_buf.append(_a(512 * OBS))          # ~512 steps to start; doubles
        eo_cap.append(512 * OBS)
        e_act.append(List[Scalar[DT]]())
        e_rew.append(List[Scalar[DT]]())
        e_pol.append(List[Scalar[DT]]())
        e_val.append(List[Scalar[DT]]())
        e_tp.append(List[Scalar[DT]]())
        e_legal.append(List[Scalar[DT]]())
        ep_len.append(0)
        ep_return.append(0.0)

    var rng = seed ^ UInt64(0x123456789)
    var mcts_seed = UInt32(seed & UInt64(0xFFFF))
    var eval_seed = UInt32((seed ^ UInt64(0xE7A1B2C3)) & UInt64(0xFFFF))
    var last_loss = 0.0
    var train_steps = 0
    var ep_returns = List[Float64]()

    # ── coarse per-section wall timers (ns) — most blocks end on a sync so the
    #    accumulated time includes the real GPU wait, not just enqueue. Printed
    #    each `report_every` (or every 100 if verbose) to localize the wall. ──
    var ts_search = 0.0   # self-play Gumbel search + D2H + sync
    var ts_collect = 0.0  # host obs/label accumulation
    var ts_env = 0.0      # CPU Atari step + selective reset (emulation)
    var ts_store = 0.0    # store finished episodes (quantize + ring H2D)
    var ts_train = 0.0    # prioritized sample (gather) + EZv2 train step
    var ts_reana = 0.0    # reanalyze wide searches
    # finer splits to localize the host hotspot
    var ts_t_sample = 0.0   # PER sample_training_batch_seq_per_gpu (host)
    var ts_t_step = 0.0     # ezv2_unroll_train_step_gpu
    var ts_re_host = 0.0    # reanalyze host (sample_position + gather + update_targets)
    var ts_re_search = 0.0  # reanalyze search + D2H + sync
    # per-phase host-enqueue breakdown of the train step (accumulated in blocks):
    # [0]setup/H2D [1]fwd-scan [2]target-prepass [3]reverse-scan [4]rep-vjp+opt
    # [5]finalize/sync ; reverse-scan sub-splits: [6]pred.fwd [7]pred.vjp
    # [8]consistency-branch [9]dyn.fwd [10]dyn.vjp
    # extra diag slots (diag_sync=True only): [11]/[13] pred/dyn pre-vjp drain,
    # [12]/[14] dyn/pred vjp GPU-drain, [15] fwd-scan GPU-drain, [16] target-
    # pre-pass GPU-drain; [7]/[10] then = pure host enqueue.
    var phase_ns_a = alloc[Float64]({count = 18})
    var phase_ns = phase_ns_a.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    try:
        for i in range(18):
            phase_ns[unsafe_offset=i] = 0.0

        env.reset_batch[N_ENVS](ctx=ctx, rng_seed=seed)

        for it in range(iterations):
            # ── 1. H2D the CPU env's live obs, batched Gumbel search ──
            var _t0 = perf_counter_ns()
            ctx.enqueue_copy(d_obs, env.obs_ptr())
            var obs_t = LayoutTensor[DT, Layout.row_major(N_ENVS, OBS),
                MutAnyOrigin](d_obs.unsafe_ptr().as_unsafe_any_origin())
            planner.search_gpu[
                type_of(rep_a),
                type_of(dyn_a),
                type_of(pred_a),
            ](ctx, rep_a, dyn_a, pred_a, obs_t,
              apply_legal=False, k_actual=MAX_K, rng_seed=mcts_seed)
            mcts_seed += UInt32(1)

            # ── 2. D2H improved policy + root value (obs already host) ──
            ctx.enqueue_copy(h_pol, planner.policies_view())
            ctx.enqueue_copy(h_val, planner.root_value_view())
            ctx.synchronize()
            ts_search += Float64(perf_counter_ns() - _t0)

            # ── 3. per env: sample, record the labelled step, stage the action ──
            _t0 = perf_counter_ns()
            var temp = visit_temperature(it, temperature_decay_steps)
            var obs_host = env.obs_ptr()
            var act_host = env.action_ptr()
            for e in range(N_ENVS):
                var action = _sample_action(h_pol, e * ACT, ACT, temp, rng)
                # write the OBS-wide observation into the per-env raw buffer at the
                # step cursor (ep_len[e]); grow capacity by doubling if needed
                # (amortized O(1), no per-step full-buffer realloc). bulk unsafe_memcpy.
                var off = ep_len[e] * OBS
                if off + OBS > eo_cap[e]:
                    var newcap = eo_cap[e] * 2
                    if newcap < off + OBS:
                        newcap = off + OBS
                    var nb = _a(newcap)
                    unsafe_memcpy(dest=nb, src=eo_buf[e], count=off)
                    eo_buf[e].unsafe_free()
                    eo_buf[e] = nb
                    eo_cap[e] = newcap
                unsafe_memcpy(dest=eo_buf[e].unsafe_offset(off), src=obs_host.unsafe_offset(e * OBS), count=OBS)
                e_act[e].append(Scalar[DT](action))
                for a in range(ACT):
                    e_pol[e].append(h_pol[unsafe_offset=e * ACT + a])
                    e_legal[e].append(Scalar[DT](1.0))
                e_val[e].append(h_val[unsafe_offset=e])
                e_tp[e].append(Scalar[DT](0.0))
                act_host[unsafe_offset=e] = Scalar[DT](action)
            ts_collect += Float64(perf_counter_ns() - _t0)

            # ── 4. step the CPU envs (host action → host reward/done/term) ──
            _t0 = perf_counter_ns()
            env.step_batch[N_ENVS](ctx=ctx, rng_seed=seed + UInt64(it + 1))
            ts_env += Float64(perf_counter_ns() - _t0)
            var rew_host = env.reward_ptr()
            var done_host = env.done_ptr()
            var term_host = env.terminated_ptr()

            # ── 5. accumulate, store + reset finished episodes ──
            var _t_store0 = perf_counter_ns()
            for e in range(N_ENVS):
                e_rew[e].append(rew_host[unsafe_offset=e])
                ep_return[e] += Float64(rew_host[unsafe_offset=e])
                ep_len[e] += 1
                var done = done_host[unsafe_offset=e] > Scalar[DT](0.5)
                var terminated = term_host[unsafe_offset=e] > Scalar[DT](0.5)
                if done or ep_len[e] >= max_ep_steps:
                    # eo_buf[e] is a raw growable pixel buffer; store_episode takes a
                    # List, so copy the resident obs into one (episodes end rarely).
                    var eo_l = List[Scalar[DT]](length=ep_len[e] * OBS, fill=0)
                    for i in range(ep_len[e] * OBS):
                        eo_l[i] = eo_buf[e][unsafe_offset=i]
                    rb.store_episode(
                        eo_l,
                        e_act[e],
                        e_rew[e],
                        e_pol[e],
                        e_val[e],
                        e_tp[e],
                        e_legal[e],
                        ep_len[e],
                        truncated=not terminated,
                    )
                    ep_returns.append(ep_return[e])
                    # reset cursors: obs buffer reused (capacity retained), labels cleared
                    e_act[e].clear(); e_rew[e].clear()
                    e_pol[e].clear(); e_val[e].clear(); e_tp[e].clear()
                    e_legal[e].clear()
                    ep_len[e] = 0
                    ep_return[e] = 0.0
            ts_store += Float64(perf_counter_ns() - _t_store0)

            var _t_rst0 = perf_counter_ns()
            env.selective_reset_batch[N_ENVS](ctx=ctx, rng_seed=seed + UInt64(it + 1))
            ts_env += Float64(perf_counter_ns() - _t_rst0)

            var trained = rb.num_steps() >= learning_starts and rb.num_episodes() > 0

            # ── 6. train (prioritized sample → weighted EZv2 unroll → writeback) ──
            var _t_train0 = perf_counter_ns()
            if trained:
                if lr > Scalar[DT](0.0):
                    var tstep = train_steps
                    var cur_lr = lr
                    if lr_warmup_iters > 0 and tstep < lr_warmup_iters:
                        cur_lr = lr * Scalar[DT](
                            Float64(tstep + 1) / Float64(lr_warmup_iters)
                        )
                    orep.set_lr(cur_lr); odyn.set_lr(cur_lr); orew.set_lr(cur_lr)
                    opred.set_lr(cur_lr); oproj.set_lr(cur_lr); opredh.set_lr(cur_lr)
                for _ in range(train_per_iter):
                    # CPU draws prioritized slots + targets, then gathers the obs
                    # slab on-device straight into d_train_obs (no host build/H2D).
                    var _tsamp = perf_counter_ns()
                    # The sample writes the PAPER PER priority |ν − z| into
                    # `t_prio` (it owns both ν = stored root search value and
                    # z = n-step target). The train step used to overwrite it
                    # with the value-head soft-CE — not the paper signal.
                    rb.sample_training_batch_seq_per_gpu[B, K, N](
                        ctx, gamma, d_train_obs,
                        d_obs_slots, h_obs_slots,
                        t_act, t_pol, t_val, t_rew, t_isw, t_slots,
                        cons_mask=t_cmask,
                        out_prio=Optional(
                            t_prio.unsafe_ptr().as_unsafe_any_origin()
                        ),
                    )
                    # per-step rewards → cumulative value prefixes (reset every HORIZON)
                    value_prefix_from_rewards[K, HORIZON](t_rew, B)
                    ts_t_sample += Float64(perf_counter_ns() - _tsamp)
                    var _tstep = perf_counter_ns()
                    last_loss = Float64(
                        ezv2_unroll_train_step_gpu_vp[
                            REP, DYNZ, PRED, PROJM, PREDH,
                            B, K, OBS, ACT, LATENT, BINS, EZ_LSTM_HIDDEN, HORIZON,
                        ](
                            ctx, rep, dyn.dynz, dyn.rew, pred, proj, predh,
                            orep, odyn, orew, opred, oproj, opredh,
                            t_obs_dummy, t_act, t_pol, t_val, t_rew,
                            v_min, v_max, value_coef, consistency_coef,
                            cons_mask=t_cmask, loss_parts=l_parts,
                            is_weights=Optional(
                                t_isw.unsafe_ptr().as_unsafe_any_origin()
                            ),
                            obs_dev=Optional(
                                d_train_obs.unsafe_ptr().as_unsafe_any_origin()
                            ),
                        )
                    )
                    ts_t_step += Float64(perf_counter_ns() - _tstep)
                    rb.update_priorities(t_slots, t_prio, B)
                    train_steps += 1
            ts_train += Float64(perf_counter_ns() - _t_train0)

            # ── per-batch loss diagnostics → logger ──
            if logger and diag_every > 0 and trained and (it + 1) % diag_every == 0:
                var dn = List[String]()
                var dv = List[Float64]()
                dn.append(String("loss")); dv.append(last_loss)
                dn.append(String("policy_loss")); dv.append(Float64(l_parts[unsafe_offset=0]))
                dn.append(String("value_loss")); dv.append(Float64(l_parts[unsafe_offset=1]))
                dn.append(String("reward_loss")); dv.append(Float64(l_parts[unsafe_offset=2]))
                dn.append(String("consistency_loss")); dv.append(Float64(l_parts[unsafe_offset=3]))
                logger.value()[].log_scalars(dn, dv, it + 1)

            # ── reanalyze through the LIVE nets — WIDE searches (ratio≈1.0 when
            #    reanalyze_batch≈B). Each chunk re-targets REANA_W positions in ONE
            #    search + ONE sync, so the whole reanalyze_batch needs only
            #    ceil(reanalyze_batch/REANA_W) wide searches. ──
            var _t_reana0 = perf_counter_ns()
            if reanalyze_every > 0 and trained and (it + 1) % reanalyze_every == 0:
                var n_chunks = (reanalyze_batch + REANA_W - 1) // REANA_W
                if n_chunks < 1:
                    n_chunks = 1
                for _c in range(n_chunks):
                    var _trh = perf_counter_ns()
                    var rpos_e = List[Int]()
                    var rpos_o = List[Int]()
                    for _ in range(REANA_W):
                        var rpos = rb.sample_position()
                        rpos_e.append(rpos[0])
                        rpos_o.append(rpos[1])
                    # gather the REANA_W positions' obs on-device into d_reana
                    rb.gather_obs_for_positions[REANA_W](
                        ctx, d_reana, d_reana_slots, h_reana_slots,
                        rpos_e, rpos_o,
                    )
                    ts_re_host += Float64(perf_counter_ns() - _trh)
                    var _trs = perf_counter_ns()
                    var reana_t = LayoutTensor[DT, Layout.row_major(REANA_W, OBS),
                        MutAnyOrigin](d_reana.unsafe_ptr().as_unsafe_any_origin())
                    reana_planner.search_gpu[
                        type_of(rep_a),
                        type_of(dyn_a),
                        type_of(pred_a),
                    ](ctx, rep_a, dyn_a, pred_a, reana_t,
                      apply_legal=False, k_actual=MAX_K, rng_seed=mcts_seed)
                    mcts_seed += UInt32(1)
                    ctx.enqueue_copy(h_pol_w.unsafe_ptr(), reana_planner.policies_view())
                    ctx.enqueue_copy(h_val_w.unsafe_ptr(), reana_planner.root_value_view())
                    ctx.synchronize()
                    ts_re_search += Float64(perf_counter_ns() - _trs)
                    var _trh2 = perf_counter_ns()
                    for e in range(REANA_W):
                        var pol_e = List[Scalar[DT]](length=ACT, fill=0)
                        for a in range(ACT):
                            pol_e[a] = h_pol_w[e * ACT + a]
                        rb.update_targets(rpos_e[e], rpos_o[e], pol_e, h_val_w[e])
                    ts_re_host += Float64(perf_counter_ns() - _trh2)
            ts_reana += Float64(perf_counter_ns() - _t_reana0)

            # ── batched greedy eval (CPU env) ──
            if eval_every > 0 and eval_env and (it + 1) % eval_every == 0:
                var cap = (
                    eval_horizon if eval_horizon > 0
                    else max_ep_steps * (eval_episodes + 1)
                )
                var avg = _ez_eval_greedy_cpu_batched[
                    ENV, REP, EZDynVPNetAtari[ACT, BINS], PRED,
                    N_ENVS, OBS, ACT, LATENT, BINS,
                    MAX_NODES, MAX_K, NUM_SIMS,
                ](
                    ctx, eval_env.value(), planner, rep_a, dyn_a, pred_a,
                    d_obs, h_pol, eval_episodes, cap, eval_seed,
                )
                eval_seed += UInt32(cap + 1)
                print("  [eval] step", it + 1, "greedy_return", avg)
                if logger:
                    logger.value()[].log_scalar(String("eval_return"), avg, it + 1)

            if verbose and (it + 1) % 100 == 0:
                var avg = 0.0
                var cnt = 0
                var lo = len(ep_returns) - 10
                if lo < 0:
                    lo = 0
                for ee in range(lo, len(ep_returns)):
                    avg += ep_returns[ee]
                    cnt += 1
                if cnt > 0:
                    avg /= Float64(cnt)
                print("iter", it + 1, "env_steps", (it + 1) * N_ENVS,
                      "loss", last_loss, "eps", rb.num_episodes(),
                      "avg_return(10)", avg)
                # cumulative wall breakdown (s) — localizes the bottleneck across
                # search / collect(host obs) / env(emulation) / store / train / reana.
                print("  [time s] search", ts_search / 1e9, "collect",
                      ts_collect / 1e9, "env", ts_env / 1e9, "store",
                      ts_store / 1e9, "train", ts_train / 1e9, "reana",
                      ts_reana / 1e9)
                # finer splits: train = sample(host) + step ; reana = host + search
                print("    train: sample", ts_t_sample / 1e9, "step",
                      ts_t_step / 1e9, "| reana: host", ts_re_host / 1e9,
                      "search", ts_re_search / 1e9)
                # train-step host-enqueue phases (s)
                print("    step phases: setup", phase_ns[unsafe_offset=0] / 1e9, "fwd",
                      phase_ns[unsafe_offset=1] / 1e9, "tgt", phase_ns[unsafe_offset=2] / 1e9, "rev",
                      phase_ns[unsafe_offset=3] / 1e9, "repvjp+opt", phase_ns[unsafe_offset=4] / 1e9,
                      "finalize/sync", phase_ns[unsafe_offset=5] / 1e9)
                # reverse-scan per-model-call splits (s) — which nn call eats `rev`
                print("    rev calls: pred.fwd", phase_ns[unsafe_offset=6] / 1e9, "pred.vjp",
                      phase_ns[unsafe_offset=7] / 1e9, "cons", phase_ns[unsafe_offset=8] / 1e9, "dyn.fwd",
                      phase_ns[unsafe_offset=9] / 1e9, "dyn.vjp", phase_ns[unsafe_offset=10] / 1e9)

            if logger and report_every > 0 and trained and (it + 1) % report_every == 0:
                var ravg = 0.0
                var rcnt = 0
                var rlo = len(ep_returns) - 10
                if rlo < 0:
                    rlo = 0
                for ee in range(rlo, len(ep_returns)):
                    ravg += ep_returns[ee]
                    rcnt += 1
                if rcnt > 0:
                    ravg /= Float64(rcnt)
                var rn = List[String]()
                var rv = List[Float64]()
                rn.append(String("avg_reward")); rv.append(ravg)
                rn.append(String("episodes")); rv.append(Float64(rb.num_episodes()))
                rn.append(String("buffer_size")); rv.append(Float64(rb.num_steps()))
                logger.value()[].log_scalars(rn, rv, it + 1)

        # ── full-run cumulative timing summary (always prints, so the totals
        #    aren't read off a mid-run snapshot) — wall split + per-iter avg. ──
        if verbose:
            var total = (
                ts_search + ts_collect + ts_env + ts_store + ts_train + ts_reana
            )
            var n = Float64(iterations) if iterations > 0 else 1.0
            print("=" * 60)
            print("FULL-RUN timing over", iterations, "iters (cumulative s):")
            print("  search", ts_search / 1e9, "collect", ts_collect / 1e9,
                  "env", ts_env / 1e9, "store", ts_store / 1e9,
                  "train", ts_train / 1e9, "reana", ts_reana / 1e9)
            print("  train = sample", ts_t_sample / 1e9, "+ step", ts_t_step / 1e9,
                  "| reana = host", ts_re_host / 1e9, "+ search", ts_re_search / 1e9)
            print("  step phases (s): setup", phase_ns[unsafe_offset=0] / 1e9, "fwd",
                  phase_ns[unsafe_offset=1] / 1e9, "tgt", phase_ns[unsafe_offset=2] / 1e9, "rev",
                  phase_ns[unsafe_offset=3] / 1e9, "repvjp+opt", phase_ns[unsafe_offset=4] / 1e9,
                  "finalize/sync", phase_ns[unsafe_offset=5] / 1e9)
            print("  rev calls (s): pred.fwd", phase_ns[unsafe_offset=6] / 1e9, "pred.vjp",
                  phase_ns[unsafe_offset=7] / 1e9, "cons", phase_ns[unsafe_offset=8] / 1e9, "dyn.fwd",
                  phase_ns[unsafe_offset=9] / 1e9, "dyn.vjp", phase_ns[unsafe_offset=10] / 1e9)
            if diag_sync:
                print("  DIAG (diag_sync): pred.vjp host", phase_ns[unsafe_offset=7] / 1e9,
                      "pred.vjp GPU", phase_ns[unsafe_offset=14] / 1e9,
                      "| dyn.vjp host", phase_ns[unsafe_offset=10] / 1e9,
                      "dyn.vjp GPU", phase_ns[unsafe_offset=12] / 1e9)
                # pre-vjp drains = GPU work of the forward/cons/tiny-kernel ops
                # enqueued before each vjp; leftover = pure host enqueue of the
                # ~90 tiny element-wise kernels/step (rev minus everything timed).
                var rev_timed = (phase_ns[unsafe_offset=6] + phase_ns[unsafe_offset=7] + phase_ns[unsafe_offset=8]
                                 + phase_ns[unsafe_offset=9] + phase_ns[unsafe_offset=10] + phase_ns[unsafe_offset=11]
                                 + phase_ns[unsafe_offset=12] + phase_ns[unsafe_offset=13] + phase_ns[unsafe_offset=14])
                print("  DIAG drains: pred pre", phase_ns[unsafe_offset=11] / 1e9,
                      "dyn pre", phase_ns[unsafe_offset=13] / 1e9,
                      "| rev untimed (host enqueue of tiny kernels)",
                      (phase_ns[unsafe_offset=3] - rev_timed) / 1e9)
                print("  DIAG fwd/tgt GPU: fwd-scan (rep×1+dyn×K)",
                      phase_ns[unsafe_offset=15] / 1e9,
                      "target-prepass (rep×K+proj×K)", phase_ns[unsafe_offset=16] / 1e9)
            print("  TOTAL timed", total / 1e9, "s  (", (total / 1e9) / n,
                  "s/iter )")
            print("=" * 60)

        t_cmask.unsafe_free(); l_parts.unsafe_free()

        h_pol.unsafe_free(); h_val.unsafe_free()
        for e in range(N_ENVS):
            eo_buf[e].unsafe_free()
    finally:
        dealloc(phase_ns_a^)
    return last_loss


def _ez_eval_greedy_cpu_batched[
    ENV: BatchedEnv,
    REP: Module,
    DYN: Module,
    PRED: Module,
    N_ENVS: Int,
    OBS: Int,
    ACT: Int,
    LATENT: Int,
    BINS: Int,
    MAX_NODES: Int,
    MAX_K: Int,
    NUM_SIMS: Int,
    RA: RepresentationGPU,
    DA: DynamicsGPU,
    PA: PredictionGPU,
](
    ctx: DeviceContext,
    eval_env: Pointer[ENV, MutAnyOrigin],
    mut planner: GumbelGPUMCTS[
        N_ENVS, ACT, LATENT, BINS, MAX_NODES, MAX_K, NUM_SIMS, SinglePlayer
    ],
    mut rep_a: RA,
    mut dyn_a: DA,
    mut pred_a: PA,
    d_obs: DeviceBuffer[DT],
    h_pol: Pointer[Scalar[DT], MutAnyOrigin],
    target_episodes: Int,
    max_steps: Int,
    rng_seed: UInt32,
) raises -> Float64:
    """Greedy (argmax improved-policy) batched eval over the CPU eval env. Runs
    the N_ENVS eval envs until `target_episodes` complete (or `max_steps`),
    summing the FIRST completed return per env. Read-only w.r.t. the trained
    nets (forward only)."""
    eval_env[].reset_batch[N_ENVS](ctx=ctx, rng_seed=UInt64(rng_seed))
    var ret_sum = 0.0
    var done_count = 0
    var cur_ret = List[Float64]()
    var counted = List[Bool]()
    for _ in range(N_ENVS):
        cur_ret.append(0.0)
        counted.append(False)
    var es = rng_seed
    for _step in range(max_steps):
        if done_count >= target_episodes:
            break
        ctx.enqueue_copy(d_obs, eval_env[].obs_ptr())
        # `d_obs` is an immutable param; planner ABI wants `MutAnyOrigin`.
        var obs_t = LayoutTensor[DT, Layout.row_major(N_ENVS, RA.OBS_DIM),
            MutAnyOrigin](
            d_obs.unsafe_ptr().as_unsafe_any_origin().unsafe_mut_cast[True]()
        )
        planner.search_gpu[RA, DA, PA](ctx, rep_a, dyn_a, pred_a, obs_t,
          apply_legal=False, k_actual=MAX_K, rng_seed=es)
        es += UInt32(1)
        ctx.enqueue_copy(h_pol, planner.policies_view())
        ctx.synchronize()
        var act_host = eval_env[].action_ptr()
        for e in range(N_ENVS):
            var best = 0
            for a in range(1, ACT):
                if Float64(h_pol[unsafe_offset=e * ACT + a]) > Float64(h_pol[unsafe_offset=e * ACT + best]):
                    best = a
            act_host[unsafe_offset=e] = Scalar[DT](best)
        eval_env[].step_batch[N_ENVS](ctx=ctx, rng_seed=UInt64(es))
        var rew_host = eval_env[].reward_ptr()
        var done_host = eval_env[].done_ptr()
        for e in range(N_ENVS):
            if not counted[e]:
                cur_ret[e] += Float64(rew_host[unsafe_offset=e])
                if done_host[unsafe_offset=e] > Scalar[DT](0.5):
                    ret_sum += cur_ret[e]
                    counted[e] = True
                    done_count += 1
        eval_env[].selective_reset_batch[N_ENVS](ctx=ctx, rng_seed=UInt64(es))
    var n = done_count if done_count > 0 else 1
    return ret_sum / Float64(n)
