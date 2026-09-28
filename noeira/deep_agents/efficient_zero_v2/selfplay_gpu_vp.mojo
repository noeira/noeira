"""EfficientZeroV2 **value-prefix** discrete self-play driver (GPU).

A concrete-`dyn` variant of `run_ezv2_gumbel_selfplay_gpu` (`selfplay_gpu.mojo`)
for the EZ `value_prefix=True` recipe (decision B1 / B1.1, see
`docs/EZV2_ATARI_PARITY.md`). Identical collection + replay + search loop, with
TWO differences:

  * `dyn` is the CONCRETE fused `EZDynVPNetAtari[ACT, BINS]` (not a generic
    `DYN: Module`). It is used directly as the SEARCH dynamics (a drop-in
    `[z|act]→[z'|vp]` net — the LSTM reward head runs with zero (h,c) per node,
    decision B1.1) AND it owns `dyn.dynz` (z'-only) + `dyn.rew` (LSTM head),
    whose shared weights are trained by the value-prefix BPTT step. A generic
    driver cannot do this: Mojo type-checks both `comptime if` branches, so a
    `DYN: Module` body cannot touch the concrete `.dynz`/`.rew` fields — hence a
    dedicated concrete-typed driver.
  * Training runs `ezv2_unroll_train_step_gpu_vp` on `dyn.dynz` / `dyn.rew` with
    a 6th (reward-head) optimizer `orew`, after converting the per-step reward
    targets to cumulative value prefixes (`value_prefix_from_rewards`, reset
    every `HORIZON`).

Single-env (`N_ENVS == 1`), mirroring `selfplay_gpu.mojo`. Returns the last loss.
"""

from std.math import exp, log
from std.memory import alloc
from layout import Layout, LayoutTensor
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.call import call_forward, call_vjp
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.nn.constants import DT
from noeira.nn.core.module import Module
from noeira.nn.optimizer.adam import Adam
from noeira.core.env_traits import BoxDiscreteActionEnv
from noeira.core.logger import Logger, NoOpLogger
from noeira.planners.tree_search import GumbelGPUMCTS, SinglePlayer

from .blocks import ezv2_unroll_train_step_gpu_vp
from .nets_atari import EZDynVPNetAtari, EZDynZNetAtari, EZ_LSTM_HIDDEN
from ..zero.mcts_adapters_mz import MZRepGPU, MZDynGPU, MZPredGPU
from ..zero.sequence_replay_mcts import MCTSSequenceReplay
from ..zero.nstep_targets import value_prefix_from_rewards
from ..zero.temperature import visit_temperature
from ..zero.mz_diagnostics import append_mz_train_diagnostics


def _a(n: Int) -> Pointer[Scalar[DT], MutAnyOrigin]:
    return alloc[Scalar[DT]]({count = n}).unsafe_leak().as_unsafe_any_origin()


def run_ezv2_gumbel_selfplay_gpu_vp[
    ENV: BoxDiscreteActionEnv,
    REP: Module,
    PRED: Module,
    PROJM: Module,
    PREDH: Module,
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
    HORIZON: Int = 5,            # lstm_horizon_len (value-prefix reset period)
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
    mut odyn: Adam,             # steps dyn.dynz (z'-only dynamics)
    mut orew: Adam,             # steps dyn.rew (LSTM value-prefix reward head)
    mut opred: Adam,
    mut oproj: Adam,
    mut opredh: Adam,
    iterations: Int,
    learning_starts: Int = 256,
    train_per_iter: Int = 1,
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
    reanalyze_batch: Int = 1,
    eval_every: Int = 0,
    eval_episodes: Int = 5,
    diag_every: Int = 0,
    report_every: Int = 0,
    logger: Optional[Pointer[L, MutAnyOrigin]] = None,
    verbose: Bool = False,
) raises -> Float64:
    comptime N_ENVS = 1
    comptime PRED_OUT = ACT + BINS
    comptime DYNZ = EZDynZNetAtari[ACT]

    # ── GPU Gumbel planner + on-device net adapters ──
    var planner = GumbelGPUMCTS[
        N_ENVS, ACT, LATENT, BINS, MAX_NODES, MAX_K, NUM_SIMS, SinglePlayer
    ](
        ctx, gamma=Float64(gamma), v_min=Float64(v_min), v_max=Float64(v_max),
        qnorm_per_node=False,
    )
    var rep_a = MZRepGPU[OBS, LATENT, REP].make(rep)
    # The fused EZDynVPNetAtari is a drop-in dynamics → the adapter is unchanged.
    var dyn_a = MZDynGPU[LATENT, ACT, BINS, EZDynVPNetAtari[ACT, BINS]].make(dyn)
    var pred_a = MZPredGPU[LATENT, ACT, BINS, PRED].make(pred)

    var rb = MCTSSequenceReplay[OBS, ACT, CAP](seed=seed ^ UInt64(0xABCDEF))

    # ── device obs buffer (single env) + host mirrors ──
    var d_obs = ctx.enqueue_create_buffer[DT](N_ENVS * OBS)
    var h_obs = List[Scalar[DT]](length=N_ENVS * OBS, fill=0)
    var h_pol = List[Scalar[DT]](length=N_ENVS * ACT, fill=0)
    var h_val = List[Scalar[DT]](length=N_ENVS, fill=0)

    # ── training batch slabs (time-major), obs is full [K+1, B, OBS] ──
    var t_obs_seq = List[Scalar[DT]](length=(K + 1) * B * OBS, fill=0)
    var t_act = List[Scalar[DT]](length=K * B, fill=0)
    var t_pol = List[Scalar[DT]](length=(K + 1) * B * ACT, fill=0)
    var t_val = List[Scalar[DT]](length=(K + 1) * B, fill=0)
    var t_rew = List[Scalar[DT]](length=K * B, fill=0)
    var t_cmask = _a(K * B)

    # logger scratch: per-component loss split + root-prediction probe (D2H).
    var l_parts = _a(4)
    var h_diag_pred = _a(B * PRED_OUT)

    var e_obs = List[Scalar[DT]]()
    var e_act = List[Scalar[DT]]()
    var e_rew = List[Scalar[DT]]()
    var e_pol = List[Scalar[DT]]()
    var e_val = List[Scalar[DT]]()
    var e_tp = List[Scalar[DT]]()
    var e_legal = List[Scalar[DT]]()
    var ep_len = 0

    var rng = seed ^ UInt64(0x123456789)
    var mcts_seed = UInt32(0)
    var last_loss = 0.0
    var ep_returns = List[Float64]()

    var cur = env.reset_obs_list()
    var cur_f = List[Float64]()
    for j in range(OBS):
        cur_f.append(Float64(cur[j]))
    var ep_return = 0.0

    for it in range(iterations):
        # ── GPU Gumbel search over the current obs (through EZDynVPNetAtari) ──
        for j in range(OBS):
            h_obs[j] = Scalar[DT](cur_f[j])
        ctx.enqueue_copy(d_obs, h_obs.unsafe_ptr())
        var obs_t = LayoutTensor[DT, Layout.row_major(N_ENVS, OBS),
            MutAnyOrigin](d_obs.unsafe_ptr().as_unsafe_any_origin())
        planner.search_gpu[
            type_of(rep_a), type_of(dyn_a), type_of(pred_a),
        ](ctx, rep_a, dyn_a, pred_a, obs_t,
          apply_legal=False, k_actual=MAX_K, rng_seed=mcts_seed)
        mcts_seed += UInt32(1)

        ctx.enqueue_copy(h_pol.unsafe_ptr(), planner.policies_view())
        ctx.enqueue_copy(h_val.unsafe_ptr(), planner.root_value_view())
        ctx.synchronize()

        var root_v = Float64(h_val[0])

        # ── sample from the improved policy, tempered π^(1/T) ──
        var temp = visit_temperature(it, temperature_decay_steps)
        var w = Array[Float64, ACT](fill=0.0)
        var wsum = 0.0
        for a in range(ACT):
            var p = Float64(h_pol[a])
            if temp != 1.0 and p > 0.0:
                p = exp(log(p) / temp)
            w[a] = p
            wsum += p
        rng = rng ^ (rng << 13); rng = rng ^ (rng >> 7); rng = rng ^ (rng << 17)
        var r = Float64(rng % UInt64(1_000_000)) / 1_000_000.0 * wsum
        var cum = 0.0
        var action = ACT - 1
        for a in range(ACT):
            cum += w[a]
            if r <= cum:
                action = a
                break

        for j in range(OBS):
            e_obs.append(Scalar[DT](cur_f[j]))
        e_act.append(Scalar[DT](action))
        for a in range(ACT):
            e_pol.append(Scalar[DT](Float64(h_pol[a])))
            e_legal.append(Scalar[DT](1.0))
        e_val.append(Scalar[DT](root_v))
        e_tp.append(Scalar[DT](0.0))

        var stepped = env.step_obs(action)
        var reward = Float64(stepped[1])
        var done = stepped[2]
        e_rew.append(Scalar[DT](reward))
        ep_return += reward
        ep_len += 1

        cur_f = List[Float64]()
        for j in range(OBS):
            cur_f.append(Float64(stepped[0][j]))

        if done or ep_len >= max_ep_steps:
            rb.store_episode(
                e_obs, e_act, e_rew, e_pol, e_val, e_tp, e_legal, ep_len,
                truncated=not env.was_terminated(),
            )
            ep_returns.append(ep_return)
            e_obs.clear(); e_act.clear(); e_rew.clear()
            e_pol.clear(); e_val.clear(); e_tp.clear(); e_legal.clear()
            ep_len = 0
            ep_return = 0.0
            cur = env.reset_obs_list()
            cur_f = List[Float64]()
            for j in range(OBS):
                cur_f.append(Float64(cur[j]))

        # ── GPU value-prefix train step ──
        if it >= learning_starts and rb.num_episodes() > 0:
            if lr > Scalar[DT](0.0):
                var tstep = it - learning_starts
                var cur_lr = lr
                if lr_warmup_iters > 0 and tstep < lr_warmup_iters:
                    cur_lr = lr * Scalar[DT](
                        Float64(tstep + 1) / Float64(lr_warmup_iters)
                    )
                orep.set_lr(cur_lr); odyn.set_lr(cur_lr); orew.set_lr(cur_lr)
                opred.set_lr(cur_lr); oproj.set_lr(cur_lr); opredh.set_lr(cur_lr)
            for _ in range(train_per_iter):
                rb.sample_training_batch_seq[B, K, N](
                    gamma, t_obs_seq, t_act, t_pol, t_val, t_rew,
                    cons_mask=t_cmask,
                )
                # per-step rewards → cumulative value prefixes (reset every HORIZON)
                value_prefix_from_rewards[K, HORIZON](t_rew, B)
                last_loss = Float64(
                    ezv2_unroll_train_step_gpu_vp[
                        REP, DYNZ, PRED, PROJM, PREDH,
                        B, K, OBS, ACT, LATENT, BINS, EZ_LSTM_HIDDEN, HORIZON,
                    ](
                        ctx, rep, dyn.dynz, dyn.rew, pred, proj, predh,
                        orep, odyn, orew, opred, oproj, opredh,
                        t_obs_seq, t_act, t_pol, t_val, t_rew,
                        v_min, v_max, value_coef, consistency_coef,
                        cons_mask=t_cmask, loss_parts=l_parts,
                    )
                )

        # ── per-batch diagnostics → logger ──
        if (
            logger and diag_every > 0 and it >= learning_starts
            and rb.num_episodes() > 0 and (it + 1) % diag_every == 0
        ):
            var obs_sc = Tensor()
            obs_sc.ensure_gpu(ctx, B * OBS)
            ctx.enqueue_copy(obs_sc.dev.value(), t_obs_seq.unsafe_ptr())
            var z_sc = Tensor()
            z_sc.ensure_gpu(ctx, B * LATENT)
            call_forward["gpu", B](rep, TensorRefs[REP.ARITY](obs_sc), z_sc, Optional(ctx))
            var pred_sc = Tensor()
            pred_sc.ensure_gpu(ctx, B * PRED_OUT)
            call_forward["gpu", B](
                pred, TensorRefs[PRED.ARITY](z_sc), pred_sc, Optional(ctx))
            ctx.enqueue_copy(h_diag_pred, pred_sc.dev.value())
            ctx.synchronize()
            var dn = List[String]()
            var dv = List[Float64]()
            dn.append(String("loss")); dv.append(last_loss)
            dn.append(String("policy_loss")); dv.append(Float64(l_parts[unsafe_offset=0]))
            dn.append(String("value_loss")); dv.append(Float64(l_parts[unsafe_offset=1]))
            dn.append(String("value_prefix_loss")); dv.append(Float64(l_parts[unsafe_offset=2]))
            dn.append(String("consistency_loss")); dv.append(Float64(l_parts[unsafe_offset=3]))
            append_mz_train_diagnostics[ACT, BINS, B](
                h_diag_pred, t_pol, t_val, v_min, v_max, dn, dv)
            logger.value()[].log_scalars(dn, dv, it + 1)

        # ── reanalyze: refresh stored positions with the current nets ──
        if (
            reanalyze_every > 0 and it >= learning_starts
            and (it + 1) % reanalyze_every == 0 and rb.num_episodes() > 0
        ):
            for _ra in range(reanalyze_batch):
                var rpos = rb.sample_position()
                rb.read_obs(rpos[0], rpos[1], h_obs)
                ctx.enqueue_copy(d_obs, h_obs.unsafe_ptr())
                var robs_t = LayoutTensor[DT, Layout.row_major(N_ENVS, OBS),
                    MutAnyOrigin](d_obs.unsafe_ptr().as_unsafe_any_origin())
                planner.search_gpu[
                    type_of(rep_a), type_of(dyn_a), type_of(pred_a),
                ](ctx, rep_a, dyn_a, pred_a, robs_t,
                  apply_legal=False, k_actual=MAX_K, rng_seed=mcts_seed)
                mcts_seed += UInt32(1)
                ctx.enqueue_copy(h_pol.unsafe_ptr(), planner.policies_view())
                ctx.enqueue_copy(h_val.unsafe_ptr(), planner.root_value_view())
                ctx.synchronize()
                rb.update_targets(rpos[0], rpos[1], h_pol, h_val[0])

        # ── greedy eval (argmax of the Gumbel improved policy) ──
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
                        type_of(rep_a), type_of(dyn_a), type_of(pred_a),
                    ](ctx, rep_a, dyn_a, pred_a, eobs_t,
                      apply_legal=False, k_actual=MAX_K, rng_seed=mcts_seed)
                    mcts_seed += UInt32(1)
                    ctx.enqueue_copy(h_pol.unsafe_ptr(), planner.policies_view())
                    ctx.synchronize()
                    var best = 0
                    for a in range(1, ACT):
                        if Float64(h_pol[a]) > Float64(h_pol[best]):
                            best = a
                    var es = env.step_obs(best)
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
                    String("eval_return"), eval_avg, it + 1)
            e_obs.clear(); e_act.clear(); e_rew.clear()
            e_pol.clear(); e_val.clear(); e_tp.clear(); e_legal.clear()
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
            logger and report_every > 0 and it >= learning_starts
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
    return last_loss
