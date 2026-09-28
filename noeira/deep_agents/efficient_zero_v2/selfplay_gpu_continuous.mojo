"""EfficientZeroV2 continuous self-play driver (GPU) — the Pendulum lighthouse.

The continuous sibling of `selfplay_gpu.mojo`. Same single-env CPU-collect /
GPU-train shape, but two pieces specialize for continuous control:

  * **Search** runs on the `SampledGumbelGPUMCTS` orchestrator (Gumbel-Top-k +
    sequential halving over *sampled continuous actions*) using the on-device
    ``h/g/f`` nets via `MZRepGPU`/`MZDynGPU`/`MZContPredGPU` — the prediction
    adapter emits the squashed-Gaussian ``[μ_raw | σ_raw | value]`` head. The
    planner samples ``K_ROOT`` candidate action vectors at the root, scores them
    by tree backups, and writes the chosen action vector directly.
  * **Training** runs `ezv2_unroll_train_step_continuous_gpu` (MuZero BPTT +
    consistency + squashed-Gaussian policy NLL) on the resident GPU nets.

The squashed-Gaussian hyperparameters (``max_action``, ``min_std``,
``soft_clamp``, ``init_std``, ``ent_scale``) are threaded into **both** the
planner sampler and the training loss so the policy parameterization matches
end-to-end. The replay (`MCTSContSequenceReplay`) stores the chosen action
**vector** per step; that same vector is the dynamics input *and* the
behavior-clone policy target. ``N_ENVS == 1`` keeps the CPU-env ↔ device-obs
round-trip trivial. Returns the last loss.
"""

from std.memory import alloc
from layout import Layout, LayoutTensor
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.call import call_forward, call_vjp
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.nn.constants import DT
from noeira.nn.core.module import Module
from noeira.nn.optimizer.adam import Adam
from noeira.nn.core.hard_copy import hard_copy
from noeira.nn.core.initializer import Kaiming
from noeira.core.env_traits import BoxContinuousActionEnv
from noeira.core.logger import Logger, NoOpLogger
from noeira.planners.tree_search import SampledGumbelGPUMCTS, SinglePlayer

from .blocks_continuous import ezv2_unroll_train_step_continuous_gpu
from .unroll_scratch import EZV2UnrollContScratch
from ..zero.mcts_adapters_mz import MZRepGPU, MZDynGPU, MZContPredGPU
from ..zero.sequence_replay_mcts_continuous import MCTSContSequenceReplay
from ..zero.mz_diagnostics import append_value_diagnostics


def _a(n: Int) -> Pointer[Scalar[DT], MutAnyOrigin]:
    return alloc[Scalar[DT]]({count = n}).unsafe_leak().as_unsafe_any_origin()


def run_ezv2_sampled_selfplay_gpu[
    ENV: BoxContinuousActionEnv,
    REP: Module,
    DYN: Module,
    PRED: Module,
    PROJM: Module,
    PREDH: Module,
    OBS: Int,
    ACT_DIM: Int,
    LATENT: Int,
    BINS: Int,
    NUM_SIMS: Int,
    MAX_NODES: Int,
    K_ROOT: Int,
    K_NON_ROOT: Int,
    CAP: Int,
    B: Int,
    K: Int,
    N: Int,
    L: Logger = NoOpLogger,
](
    ctx: DeviceContext,
    mut env: ENV,
    mut rep: REP,
    mut dyn: DYN,
    mut pred: PRED,
    mut proj: PROJM,
    mut predh: PREDH,
    mut orep: Adam,
    mut odyn: Adam,
    mut opred: Adam,
    mut oproj: Adam,
    mut opredh: Adam,
    iterations: Int,
    learning_starts: Int = 256,
    train_per_iter: Int = 1,
    gamma: Scalar[DT] = Scalar[DT](0.99),
    v_min: Scalar[DT] = Scalar[DT](-50.0),
    v_max: Scalar[DT] = Scalar[DT](2.0),
    seed: UInt64 = 0,
    max_ep_steps: Int = 200,
    value_coef: Scalar[DT] = Scalar[DT](0.25),
    consistency_coef: Scalar[DT] = Scalar[DT](2.0),
    policy_coef: Scalar[DT] = Scalar[DT](1.0),
    max_action: Scalar[DT] = Scalar[DT](2.0),
    min_std: Scalar[DT] = Scalar[DT](0.5),
    std_magnification: Scalar[DT] = Scalar[DT](3.0),
    soft_clamp: Scalar[DT] = Scalar[DT](5.0),
    init_std: Scalar[DT] = Scalar[DT](1.0),
    ent_scale: Scalar[DT] = Scalar[DT](0.05),
    c_visit: Scalar[DT] = Scalar[DT](50.0),
    c_scale: Scalar[DT] = Scalar[DT](0.1),
    target_sync_interval: Int = 200,
    reanalyze_interval: Int = 1,
    reanalyze_warmup: Int = 500,
    reanalyze_batch: Int = 4,
    eval_every: Int = 0,
    eval_episodes: Int = 5,
    diag_every: Int = 0,
    report_every: Int = 0,
    logger: Optional[Pointer[L, MutAnyOrigin]] = None,
    verbose: Bool = False,
) raises -> Float64:
    comptime N_ENVS = 1

    # ── GPU sampled-Gumbel planner + on-device net adapters ──
    # reward head is binned over [v_min, v_max] in the training unroll, so the
    # planner decodes reward over the same range (override the ±0.732 default).
    var planner = SampledGumbelGPUMCTS[
        N_ENVS, ACT_DIM, LATENT, BINS, MAX_NODES, K_ROOT, K_NON_ROOT, NUM_SIMS
    ](
        ctx,
        gamma=Float64(gamma),
        v_min=Float64(v_min),
        v_max=Float64(v_max),
        reward_min=Float64(v_min),
        reward_max=Float64(v_max),
        max_action=Float64(max_action),
        min_std=Float64(min_std),
        std_magnification=Float64(std_magnification),
        soft_clamp=Float64(soft_clamp),
        init_std=Float64(init_std),
        c_visit=Float64(c_visit),
        c_scale=Float64(c_scale),
    )
    var rep_a = MZRepGPU[OBS, LATENT, REP].make(rep)
    var dyn_a = MZDynGPU[LATENT, ACT_DIM, BINS, DYN].make(dyn)
    var pred_a = MZContPredGPU[LATENT, ACT_DIM, BINS, PRED].make(pred)

    # ── lagging target nets (rep/dyn/pred only — proj/predh are consistency-
    #    only, never used in search) + their adapters, for stable reanalyze.
    var rep_t = REP.make["gpu", Kaiming](Optional(ctx))
    var dyn_t = DYN.make["gpu", Kaiming](Optional(ctx))
    var pred_t = PRED.make["gpu", Kaiming](Optional(ctx))
    hard_copy["gpu", M=REP](rep, rep_t, Optional(ctx))
    hard_copy["gpu", M=DYN](dyn, dyn_t, Optional(ctx))
    hard_copy["gpu", M=PRED](pred, pred_t, Optional(ctx))
    var rep_ta = MZRepGPU[OBS, LATENT, REP].make(rep_t)
    var dyn_ta = MZDynGPU[LATENT, ACT_DIM, BINS, DYN].make(dyn_t)
    var pred_ta = MZContPredGPU[LATENT, ACT_DIM, BINS, PRED].make(pred_t)

    var rb = MCTSContSequenceReplay[OBS, ACT_DIM, CAP](
        seed=seed ^ UInt64(0xABCDEF)
    )

    # ── device obs buffer (single env) + host mirrors ──
    var d_obs = ctx.enqueue_create_buffer[DT](N_ENVS * OBS)
    var h_obs = List[Scalar[DT]](length=N_ENVS * OBS, fill=0)
    var h_act = List[Scalar[DT]](length=N_ENVS * ACT_DIM, fill=0)
    var h_val = List[Scalar[DT]](length=N_ENVS, fill=0)
    # reanalyze scratch (root obs read from replay → device → fresh target search)
    var h_ra_obs = List[Scalar[DT]](length=N_ENVS * OBS, fill=0)
    var h_ra_act = List[Scalar[DT]](length=N_ENVS * ACT_DIM, fill=0)
    var h_ra_val = List[Scalar[DT]](length=N_ENVS, fill=0)
    var train_steps = 0

    # ── training batch slabs (time-major), obs is full [K+1, B, OBS] ──
    var t_obs_seq = List[Scalar[DT]](length=(K + 1) * B * OBS, fill=0)
    var t_act = List[Scalar[DT]](length=K * B * ACT_DIM, fill=0)
    var t_pol = List[Scalar[DT]](length=(K + 1) * B * ACT_DIM, fill=0)
    var t_val = List[Scalar[DT]](length=(K + 1) * B, fill=0)
    var t_rew = List[Scalar[DT]](length=K * B, fill=0)
    var t_cmask = _a(K * B)   # consistency episode-boundary mask

    # ── persistent GPU train-step scratch (allocated once, reused per step) ──
    comptime CPRED_OUT = 2 * ACT_DIM + BINS

    # logger scratch: per-component loss split + root-prediction probe (D2H).
    var l_parts = _a(4)
    var d_diag_obs = ctx.enqueue_create_buffer[DT](B * OBS)
    var d_diag_z = ctx.enqueue_create_buffer[DT](B * LATENT)
    var d_diag_pred = ctx.enqueue_create_buffer[DT](B * CPRED_OUT)
    var h_diag_pred = _a(B * CPRED_OUT)

    var train_scratch = EZV2UnrollContScratch[
        B, K, OBS, ACT_DIM, LATENT, BINS, PROJM.OUT_DIM
    ].make(ctx)

    var e_obs = List[Scalar[DT]]()
    var e_act = List[Scalar[DT]]()
    var e_rew = List[Scalar[DT]]()
    var e_val = List[Scalar[DT]]()
    var ep_len = 0

    var mcts_seed = UInt32(0)
    var last_loss = 0.0
    var ep_returns = List[Float64]()

    var cur = env.reset_obs_list()
    var cur_f = List[Float64]()
    for j in range(OBS):
        cur_f.append(Float64(cur[j]))
    var ep_return = 0.0

    for it in range(iterations):
        # ── GPU sampled-Gumbel search over the current obs ──
        for j in range(OBS):
            h_obs[j] = Scalar[DT](cur_f[j])
        ctx.enqueue_copy(d_obs, h_obs.unsafe_ptr())
        var obs_t = LayoutTensor[DT, Layout.row_major(N_ENVS, OBS),
            MutAnyOrigin](d_obs.unsafe_ptr().as_unsafe_any_origin())
        planner.search_gpu[
            type_of(rep_a),
            type_of(dyn_a),
            type_of(pred_a),
        ](ctx, rep_a, dyn_a, pred_a, obs_t,
          deterministic=False, rng_seed=mcts_seed)
        mcts_seed += UInt32(1)

        ctx.enqueue_copy(h_act.unsafe_ptr(), planner.chosen_actions_view())
        ctx.enqueue_copy(h_val.unsafe_ptr(), planner.root_value_view())
        ctx.synchronize()

        var root_v = Float64(h_val[0])

        # ── record obs, the chosen action vector, root value ──
        for j in range(OBS):
            e_obs.append(Scalar[DT](cur_f[j]))
        var action_list = List[Scalar[DT]]()
        for d in range(ACT_DIM):
            var av = h_act[d]
            e_act.append(av)
            action_list.append(av)
        e_val.append(Scalar[DT](root_v))

        var stepped = env.step_continuous_vec[DT](action_list)
        var reward = Float64(stepped[1])
        var done = stepped[2]
        e_rew.append(Scalar[DT](reward))
        ep_return += reward
        ep_len += 1

        cur_f = List[Float64]()
        for j in range(OBS):
            cur_f.append(Float64(stepped[0][j]))

        if done or ep_len >= max_ep_steps:
            # Time-limit cut is NOT a terminal — bootstrap past it. Pendulum
            # never terminates naturally, so EVERY episode is truncated; the
            # old terminal-0 label was an *optimistic* corruption near each
            # episode end (0 > any real all-negative-reward value).
            rb.store_episode(
                e_obs,
                e_act,
                e_rew,
                e_val,
                ep_len,
                truncated=not env.was_terminated(),
            )
            ep_returns.append(ep_return)
            e_obs.clear(); e_act.clear(); e_rew.clear(); e_val.clear()
            ep_len = 0
            ep_return = 0.0
            cur = env.reset_obs_list()
            cur_f = List[Float64]()
            for j in range(OBS):
                cur_f.append(Float64(cur[j]))

        # ── GPU train step ──
        if it >= learning_starts and rb.num_episodes() > 0:
            for _ in range(train_per_iter):
                rb.sample_training_batch_seq[B, K, N](
                    gamma, t_obs_seq, t_act, t_pol, t_val, t_rew,
                    cons_mask=t_cmask,
                )
                last_loss = Float64(
                    ezv2_unroll_train_step_continuous_gpu[
                        REP, DYN, PRED, PROJM, PREDH,
                        B, K, OBS, ACT_DIM, LATENT, BINS,
                    ](
                        ctx, train_scratch, rep, dyn, pred, proj, predh,
                        orep, odyn, opred, oproj, opredh,
                        t_obs_seq, t_act, t_pol, t_val, t_rew,
                        v_min, v_max, value_coef, consistency_coef,
                        policy_coef, max_action, min_std, soft_clamp,
                        init_std, ent_scale,
                        cons_mask=t_cmask,
                        loss_parts=l_parts,
                    )
                )
                train_steps += 1

                # ── sync lagging target nets ──
                if train_steps % target_sync_interval == 0:
                    hard_copy["gpu", M=REP](rep, rep_t, Optional(ctx))
                    hard_copy["gpu", M=DYN](dyn, dyn_t, Optional(ctx))
                    hard_copy["gpu", M=PRED](pred, pred_t, Optional(ctx))

                # ── reanalyze: refresh stale (action, value) targets on old
                #    positions with a fresh TARGET-net search (post-warmup). ──
                if (
                    train_steps >= reanalyze_warmup
                    and train_steps % reanalyze_interval == 0
                    and rb.num_episodes() > 0
                ):
                    for _ra in range(reanalyze_batch):
                        var pos = rb.sample_position()
                        rb.read_obs(pos[0], pos[1], h_ra_obs)
                        ctx.enqueue_copy(d_obs, h_ra_obs.unsafe_ptr())
                        var ra_obs_t = LayoutTensor[
                            DT, Layout.row_major(N_ENVS, OBS), MutAnyOrigin
                        ](d_obs.unsafe_ptr().as_unsafe_any_origin())
                        planner.search_gpu[
                            type_of(rep_ta),
                            type_of(dyn_ta),
                            type_of(pred_ta),
                        ](ctx, rep_ta, dyn_ta, pred_ta, ra_obs_t,
                          deterministic=False, rng_seed=mcts_seed)
                        mcts_seed += UInt32(1)
                        ctx.enqueue_copy(
                            h_ra_act, planner.chosen_actions_view()
                        )
                        ctx.enqueue_copy(h_ra_val.unsafe_ptr(), planner.root_value_view())
                        ctx.synchronize()
                        rb.update_targets(
                            pos[0], pos[1], h_ra_act, h_ra_val[0]
                        )

        # ── per-batch diagnostics → logger (root pred re-forwarded on device) ──
        if (
            logger
            and diag_every > 0
            and it >= learning_starts
            and rb.num_episodes() > 0
            and (it + 1) % diag_every == 0
        ):
            # storage forward on owned scratch Tensors; D2H pred into h_diag_pred.
            var obs_sc = Tensor()
            obs_sc.ensure_gpu(ctx, B * OBS)
            ctx.enqueue_copy(obs_sc.dev.value(), t_obs_seq.unsafe_ptr())
            var z_sc = Tensor()
            z_sc.ensure_gpu(ctx, B * LATENT)
            call_forward["gpu", B](rep, TensorRefs[REP.ARITY](obs_sc), z_sc, Optional(ctx))
            var pred_sc = Tensor()
            pred_sc.ensure_gpu(ctx, B * CPRED_OUT)
            call_forward["gpu", B](
                pred, TensorRefs[PRED.ARITY](z_sc), pred_sc, Optional(ctx)
            )
            ctx.enqueue_copy(h_diag_pred, pred_sc.dev.value())
            ctx.synchronize()
            var dn = List[String]()
            var dv = List[Float64]()
            dn.append(String("loss")); dv.append(last_loss)
            dn.append(String("policy_loss")); dv.append(Float64(l_parts[unsafe_offset=0]))
            dn.append(String("value_loss")); dv.append(Float64(l_parts[unsafe_offset=1]))
            dn.append(String("reward_loss")); dv.append(Float64(l_parts[unsafe_offset=2]))
            dn.append(String("consistency_loss"))
            dv.append(Float64(l_parts[unsafe_offset=3]))
            append_value_diagnostics[CPRED_OUT, 2 * ACT_DIM, BINS, B](
                h_diag_pred, t_val, v_min, v_max, dn, dv
            )
            logger.value()[].log_scalars(dn, dv, it + 1)

        # ── greedy eval (deterministic argmax-visit candidate) ──
        if eval_every > 0 and (it + 1) % eval_every == 0:
            var eval_sum = 0.0
            for _ in range(eval_episodes):
                var eo = env.reset_obs_list()
                var eo_f = List[Float64]()
                for j in range(OBS):
                    eo_f.append(Float64(eo[j]))
                var eret = 0.0
                for _step in range(max_ep_steps):
                    for j in range(OBS):
                        h_obs[j] = Scalar[DT](eo_f[j])
                    ctx.enqueue_copy(d_obs, h_obs.unsafe_ptr())
                    var eobs_t = LayoutTensor[DT,
                        Layout.row_major(N_ENVS, OBS), MutAnyOrigin](
                            d_obs.unsafe_ptr().as_unsafe_any_origin())
                    planner.search_gpu[
                        type_of(rep_a),
                        type_of(dyn_a),
                        type_of(pred_a),
                    ](ctx, rep_a, dyn_a, pred_a, eobs_t,
                      deterministic=True, rng_seed=mcts_seed)
                    mcts_seed += UInt32(1)
                    ctx.enqueue_copy(h_act.unsafe_ptr(), planner.chosen_actions_view())
                    ctx.synchronize()
                    var ea = List[Scalar[DT]]()
                    for d in range(ACT_DIM):
                        ea.append(h_act[d])
                    var es = env.step_continuous_vec[DT](ea)
                    eret += Float64(es[1])
                    eo_f = List[Float64]()
                    for j in range(OBS):
                        eo_f.append(Float64(es[0][j]))
                    if es[2]:
                        break
                eval_sum += eret
            var eval_avg = eval_sum / Float64(eval_episodes)
            print("  [eval] step", it + 1, "greedy_return", eval_avg)
            if logger:
                logger.value()[].log_scalar(
                    String("eval_return"), eval_avg, it + 1
                )
            e_obs.clear(); e_act.clear(); e_rew.clear(); e_val.clear()
            ep_len = 0
            ep_return = 0.0
            cur = env.reset_obs_list()
            cur_f = List[Float64]()
            for j in range(OBS):
                cur_f.append(Float64(cur[j]))

        if verbose and (it + 1) % 500 == 0:
            var avg = 0.0
            var cnt = 0
            var lo = len(ep_returns) - 10
            if lo < 0:
                lo = 0
            for e in range(lo, len(ep_returns)):
                avg += ep_returns[e]
                cnt += 1
            if cnt > 0:
                avg /= Float64(cnt)
            print(
                "step", it + 1, "loss", last_loss,
                "eps", rb.num_episodes(), "avg_return(10)", avg,
            )

        # ── report_every: episode-return / replay status to the logger ──
        if (
            logger
            and report_every > 0
            and it >= learning_starts
            and (it + 1) % report_every == 0
        ):
            var ravg = 0.0
            var rcnt = 0
            var rlo = len(ep_returns) - 10
            if rlo < 0:
                rlo = 0
            for e in range(rlo, len(ep_returns)):
                ravg += ep_returns[e]
                rcnt += 1
            if rcnt > 0:
                ravg /= Float64(rcnt)
            var rn = List[String]()
            var rv = List[Float64]()
            rn.append(String("avg_reward")); rv.append(ravg)
            rn.append(String("episodes")); rv.append(Float64(rb.num_episodes()))
            rn.append(String("buffer_size")); rv.append(Float64(rb.num_steps()))
            logger.value()[].log_scalars(rn, rv, it + 1)

    t_cmask.unsafe_free()
    l_parts.unsafe_free(); h_diag_pred.unsafe_free()
    # keep the target nets (held only via Pointer in the adapters) alive
    # through the whole rollout — the analyzer can't see the indirection.
    _ = rep_t^
    _ = dyn_t^
    _ = pred_t^
    return last_loss
