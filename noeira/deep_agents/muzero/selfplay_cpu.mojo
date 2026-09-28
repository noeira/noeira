"""MuZero single-player self-play driver (CPU) — the CartPole lighthouse loop.

Ties every Phase-B piece together on one device (CPU), reusing the validated
CPU MCTS (learned dynamics) and CPU BPTT unroll — no GPU/CPU param sync:

  per env step:  obs → GenericCPUMCTS.search (MZ CPU adapters) → visit policy π_t
                 + root value v_t → sample a_t ∼ π_t → env.step → r_{t+1}, done →
                 append (o_t, a_t, r_{t+1}, π_t, v_t, to_play=0) to the episode
  on done:       store_episode → MCTSSequenceReplay
  every step≥learning_starts: sample_training_batch → mz_unroll_train_step_cpu

Single-player (CartPole): ``to_play`` is always 0, so the n-step sign flips are
no-ops. Actions are sampled ∝ visits^(1/T) with the legacy piecewise temperature
schedule (1.0 → 0.5 → 0.25 over ``temperature_decay_steps``; 0 = fixed T=1);
the root value stored is the search value (`mcts.root_value()`), the MuZero
bootstrap target. Returns the last training loss.
"""

from noeira.nn.core.ptr import untracked
from std.math import exp, log
from std.memory import alloc

from noeira.nn.constants import DT
from noeira.nn.core.module import Module
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.call import call_forward
from noeira.nn.optimizer.adam import Adam
from noeira.core.env_traits import BoxDiscreteActionEnv
from noeira.core.logger import Logger, NoOpLogger
from ..zero.mz_diagnostics import append_mz_train_diagnostics
from noeira.planners.tree_search import (
    GenericCPUMCTS,
    MuZeroPUCT,
    DirichletNoise,
    SinglePlayer,
)

from .blocks import mz_unroll_train_step_cpu
from ..zero.mcts_adapters_mz_cpu import MZRepCPU, MZDynCPU, MZPredCPU
from ..zero.sequence_replay_mcts import MCTSSequenceReplay
from ..zero.temperature import visit_temperature


def _a(n: Int) -> Pointer[Scalar[DT], MutAnyOrigin]:
    """Category-B raw batch/episode scratch feeding the raw-pointer replay +
    unroll-input boundary (not the nn surface)."""
    return alloc[Scalar[DT]]({count = n}).unsafe_leak().as_unsafe_any_origin()


def run_muzero_selfplay_cpu[
    ENV: BoxDiscreteActionEnv,
    REP: Module,
    DYN: Module,
    PRED: Module,
    OBS: Int,
    ACT: Int,
    LATENT: Int,
    BINS: Int,
    NUM_SIMS: Int,
    MAX_NODES: Int,
    CAP: Int,
    B: Int,
    K: Int,
    N: Int,
    BATCH_SIMS: Int = 8,
    VIRTUAL_LOSS: Int = 3,
    L: Logger = NoOpLogger,
](
    mut env: ENV,
    mut rep: REP,
    mut dyn: DYN,
    mut pred: PRED,
    mut orep: Adam,
    mut odyn: Adam,
    mut opred: Adam,
    iterations: Int,
    learning_starts: Int = 256,
    train_per_iter: Int = 1,
    gamma: Scalar[DT] = Scalar[DT](0.997),
    v_min: Scalar[DT] = Scalar[DT](-10.0),
    v_max: Scalar[DT] = Scalar[DT](10.0),
    seed: UInt64 = 0,
    max_ep_steps: Int = 500,
    value_coef: Scalar[DT] = Scalar[DT](0.25),
    max_grad_norm: Float64 = 0.0,
    temperature_decay_steps: Int = 0,
    reanalyze_every: Int = 0,
    eval_every: Int = 0,
    eval_episodes: Int = 5,
    diag_every: Int = 0,
    report_every: Int = 0,
    logger: Optional[Pointer[L, MutAnyOrigin]] = None,
    verbose: Bool = False,
) raises -> Float64:
    # PUCT defaults (c_base=19652, c_init=1.25) — matches legacy
    # MuZeroMLPConfig.PUCT. NOTE: ``MuZeroPUCT[1.25]`` would bind c_base=1.25
    # (the *first* positional param), not c_init, badly distorting exploration.
    # BATCH_SIMS=8 + VIRTUAL_LOSS=3 diversify root exploration to counter the
    # spiky DirichletNoise[0.25,0.25] prior (legacy batch_sims=8 / virtual_loss=3).
    var mcts = GenericCPUMCTS[
        ACT, LATENT, NUM_SIMS, MAX_NODES,
        MuZeroPUCT[19652.0, 1.25], DirichletNoise[0.25, 0.25], SinglePlayer,
        BATCH_SIMS, VIRTUAL_LOSS,
    ](gamma=Float64(gamma))
    var rb = MCTSSequenceReplay[OBS, ACT, CAP](seed=seed ^ UInt64(0xABCDEF))

    var rep_a = MZRepCPU[OBS, LATENT, REP](
        net=untracked(Pointer(to=rep))
    )
    var dyn_a = MZDynCPU[LATENT, ACT, BINS, DYN](
        net=untracked(Pointer(to=dyn)), v_min=v_min, v_max=v_max
    )
    var pred_a = MZPredCPU[LATENT, ACT, BINS, PRED](
        net=untracked(Pointer(to=pred)),
        v_min=v_min, v_max=v_max
    )

    # training batch slabs (time-major) — owned Lists (RAII), filled by the
    # replay's List API and read by the List-input unroll. No raw pointers.
    var t_obs0 = List[Scalar[DT]](length=B * OBS, fill=0)
    var t_act = List[Scalar[DT]](length=K * B, fill=0)
    var t_pol = List[Scalar[DT]](length=(K + 1) * B * ACT, fill=0)
    var t_val = List[Scalar[DT]](length=(K + 1) * B, fill=0)
    var t_rew = List[Scalar[DT]](length=K * B, fill=0)

    # reanalyze scratch (owned Lists; read_obs/update_targets take Lists)
    var r_obs = List[Scalar[DT]](length=OBS, fill=0)
    var r_pol = List[Scalar[DT]](length=ACT, fill=0)

    # logger scratch: per-component loss split (optional `loss_parts` output of
    # the unroll, kept as a raw buffer — `Optional[List]` would copy on .value())
    # + root-prediction probe buffer for the diagnostics helper.
    var l_parts = _a(3)            # [policy, value, reward] means
    var d_pred = _a(B * (ACT + BINS))

    # episode accumulation buffers
    var e_obs = List[Scalar[DT]]()
    var e_act = List[Scalar[DT]]()
    var e_rew = List[Scalar[DT]]()
    var e_pol = List[Scalar[DT]]()
    var e_val = List[Scalar[DT]]()
    var e_tp = List[Scalar[DT]]()
    var e_legal = List[Scalar[DT]]()    # all-legal (single-player); reanalyze ch.
    var ep_len = 0

    var rng = seed ^ UInt64(0x123456789)
    var last_loss = 0.0
    var ep_returns = List[Float64]()

    var cur = env.reset_obs_list()
    var cur_f = List[Float64]()
    for j in range(OBS):
        cur_f.append(Float64(cur[j]))
    var ep_return = 0.0

    for it in range(iterations):
        # ── search ──
        var policy = mcts.search[type_of(rep_a), type_of(dyn_a), type_of(pred_a)](
            rep_a, dyn_a, pred_a, cur_f, add_noise=True
        )
        var root_v = mcts.root_value()

        # ── sample action ∝ visits^(1/T) ──
        # The *stored* policy target stays the untempered visit distribution.
        var temp = visit_temperature(it, temperature_decay_steps)
        var w = Array[Float64, ACT](fill=0.0)
        var wsum = 0.0
        for a in range(ACT):
            var p = policy[a]
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

        # ── record step (o_t, a_t, π_t, v_t, to_play=0) ──
        for j in range(OBS):
            e_obs.append(Scalar[DT](cur_f[j]))
        e_act.append(Scalar[DT](action))
        for a in range(ACT):
            e_pol.append(Scalar[DT](policy[a]))
            e_legal.append(Scalar[DT](1.0))     # CartPole: every action legal
        e_val.append(Scalar[DT](root_v))
        e_tp.append(Scalar[DT](0.0))

        # ── env step → r_{t+1}, done ──
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
            # Time-limit cut (env truncation or our max_ep_steps cap) is NOT a
            # terminal: the replay must bootstrap past it, not target value 0.
            rb.store_episode(
                e_obs, e_act, e_rew, e_pol, e_val, e_tp, e_legal,
                ep_len,
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

        # ── train ──
        if it >= learning_starts and rb.num_episodes() > 0:
            for _ in range(train_per_iter):
                rb.sample_training_batch[B, K, N](
                    gamma, t_obs0, t_act, t_pol, t_val, t_rew
                )
                last_loss = Float64(
                    mz_unroll_train_step_cpu[
                        REP, DYN, PRED, B, K, OBS, ACT, LATENT, BINS
                    ](
                        rep, dyn, pred, orep, odyn, opred,
                        t_obs0, t_act, t_pol, t_val, t_rew,
                        v_min, v_max, value_coef, max_grad_norm,
                        loss_parts=l_parts,
                    )
                )

        # ── per-batch diagnostics → logger (root pred re-forwarded on host) ──
        if (
            logger
            and diag_every > 0
            and it >= learning_starts
            and rb.num_episodes() > 0
            and (it + 1) % diag_every == 0
        ):
            # storage forward (h then f) for the root-pred probe: copy the raw
            # batch obs into an owned Tensor, run, copy the pred output back into
            # the raw diag buffer for `append_mz_train_diagnostics`.
            var obs_t = Tensor.alloc(B * OBS)
            for i in range(B * OBS):
                obs_t.data[i] = t_obs0[i]
            var z_t = Tensor.alloc(B * LATENT)
            call_forward["cpu", B](rep, TensorRefs[REP.ARITY](obs_t), z_t, None)
            var pred_t = Tensor.alloc(B * (ACT + BINS))
            call_forward["cpu", B](pred, TensorRefs[PRED.ARITY](z_t), pred_t, None)
            for i in range(B * (ACT + BINS)):
                d_pred[unsafe_offset=i] = pred_t.data[i]
            var dn = List[String]()
            var dv = List[Float64]()
            dn.append(String("loss")); dv.append(last_loss)
            dn.append(String("policy_loss")); dv.append(Float64(l_parts[unsafe_offset=0]))
            dn.append(String("value_loss")); dv.append(Float64(l_parts[unsafe_offset=1]))
            dn.append(String("reward_loss")); dv.append(Float64(l_parts[unsafe_offset=2]))
            append_mz_train_diagnostics[ACT, BINS, B](
                d_pred, t_pol, t_val, v_min, v_max, dn, dv
            )
            logger.value()[].log_scalars(dn, dv, it + 1)

        # ── reanalyze: refresh one stored position with a fresh search ──
        # The n-step targets bootstrap from STORED root values; without
        # refresh, never-evicted early episodes keep teaching weak-net values
        # and stale visit policies forever (legacy ran use_reanalyze=True).
        if (
            reanalyze_every > 0
            and it >= learning_starts
            and (it + 1) % reanalyze_every == 0
            and rb.num_episodes() > 0
        ):
            var rpos = rb.sample_position()
            rb.read_obs(rpos[0], rpos[1], r_obs)
            var ro = List[Float64]()
            for j in range(OBS):
                ro.append(Float64(r_obs[j]))
            var rpolicy = mcts.search[
                type_of(rep_a), type_of(dyn_a), type_of(pred_a)
            ](rep_a, dyn_a, pred_a, ro, add_noise=True)
            for a in range(ACT):
                r_pol[a] = Scalar[DT](rpolicy[a])
            rb.update_targets(
                rpos[0], rpos[1], r_pol, Scalar[DT](mcts.root_value())
            )

        # ── periodic GREEDY eval (noise off, argmax visits) ──
        # Training return above is exploratory (∝-visit sampling + root
        # Dirichlet noise) and understates the policy. This measures the
        # deterministic policy. Interrupts the in-progress training episode,
        # so keep ``eval_every`` large.
        if eval_every > 0 and (it + 1) % eval_every == 0:
            var eval_sum = 0.0
            for _ in range(eval_episodes):
                var eo = env.reset_obs_list()
                var eo_f = List[Float64]()
                for j in range(OBS):
                    eo_f.append(Float64(eo[j]))
                var eret = 0.0
                for _step in range(max_ep_steps):
                    var ep = mcts.search[
                        type_of(rep_a), type_of(dyn_a), type_of(pred_a)
                    ](rep_a, dyn_a, pred_a, eo_f, add_noise=False)
                    var best = 0
                    for a in range(1, ACT):
                        if ep[a] > ep[best]:
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
                    String("eval_return"), eval_avg, it + 1
                )
            # restart a clean training episode (eval clobbered ``env``)
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

    l_parts.unsafe_free(); d_pred.unsafe_free()
    return last_loss
