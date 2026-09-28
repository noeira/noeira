"""MuZero single-player self-play driver (GPU) — CPU-search / GPU-train hybrid.

Mirrors `run_muzero_selfplay_cpu` step-for-step, but the K-step BPTT unroll runs
on the **device** (`mz_unroll_train_step_gpu`) while the MCTS search still runs on
the CPU (`GenericCPUMCTS` over the learned model — the search is host control-flow
heavy and not worth porting). The two halves are bridged by a **CPU mirror** of
the three nets: after every GPU train step the freshly-updated device params are
downloaded into the mirror (`mz_sync_gpu_to_cpu`: collect/inject visitors,
exact, no checkpoint round-trip), so the next search plans with
up-to-date weights. The replay buffer is host-side (`MCTSSequenceReplay`) and the
training batch is a host slab H2D-copied inside the unroll step — the same data
path as the CPU driver; only the gradient math moves to the GPU.

Single-player (CartPole): `to_play` is always 0, so the n-step sign flips are
no-ops. Actions are sampled ∝ MCTS visit counts (exploration); the stored root
value is the search value (`mcts.root_value()`). Returns the last training loss.

The mirror sync is the cost of the hybrid: it is done once per *iteration* (after
the inner `train_per_iter` steps), not per inner step, since the search only runs
once per iteration. `learning_starts` warmup needs no sync — the mirror is synced
once before the loop so it matches the device nets from step 0.
"""

from noeira.nn.core.param import walk_params
from noeira.nn.core.ptr import untracked
from std.math import exp, log
from std.memory import alloc

from noeira.nn.constants import DT
from noeira.nn.core.module import Module
from noeira.nn.core.initializer import Kaiming
from noeira.nn.optimizer.adam import Adam
from noeira.nn.core.hard_copy import _CollectVisitor, _InjectVisitor
from noeira.core.env_traits import BoxDiscreteActionEnv
from noeira.planners.tree_search import (
    GenericCPUMCTS,
    MuZeroPUCT,
    DirichletNoise,
    SinglePlayer,
)
from max.gpu.host import DeviceContext

from noeira.core.logger import Logger, NoOpLogger
from .blocks import mz_unroll_train_step_gpu, MZScratch
from .selfplay_gpu_device import _mz_emit_train_diag
from ..zero.mcts_adapters_mz_cpu import MZRepCPU, MZDynCPU, MZPredCPU
from ..zero.sequence_replay_mcts import MCTSSequenceReplay
from ..zero.temperature import visit_temperature


def _a(n: Int) -> Pointer[Scalar[DT], MutAnyOrigin]:
    """Category-B raw batch scratch feeding the raw-pointer replay + unroll
    inputs (not the nn surface)."""
    return alloc[Scalar[DT]]({count = n}).unsafe_leak().as_unsafe_any_origin()


def mz_sync_gpu_to_cpu[M: Module](
    mut gpu: M, mut cpu: M, ctx: DeviceContext
) raises:
    """Download device params into a CPU mirror in place via the storage
    hard_copy collect/inject visitors (exact, no checkpoint text round-trip).
    The CPU net's param buffers are overwritten, so any MCTS adapter holding
    `Pointer(to=cpu_net)` sees the updated weights with no rebind."""
    var c = _CollectVisitor()
    walk_params["gpu"](gpu, c, Optional(ctx))
    var inj = _InjectVisitor(c.names.copy(), c.vals.copy())
    walk_params["cpu"](cpu, inj, None)


def run_muzero_selfplay_gpu[
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
    ctx: DeviceContext,
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
    var mcts = GenericCPUMCTS[
        ACT, LATENT, NUM_SIMS, MAX_NODES,
        MuZeroPUCT[19652.0, 1.25], DirichletNoise[0.25, 0.25], SinglePlayer,
        BATCH_SIMS, VIRTUAL_LOSS,
    ](gamma=Float64(gamma))
    var rb = MCTSSequenceReplay[OBS, ACT, CAP](seed=seed ^ UInt64(0xABCDEF))

    # CPU mirror of the device nets — the search reads these; synced from GPU
    # once now (so the mirror matches the device init) and after every train.
    var rep_c = REP.make["cpu", Kaiming]()
    var dyn_c = DYN.make["cpu", Kaiming]()
    var pred_c = PRED.make["cpu", Kaiming]()
    mz_sync_gpu_to_cpu(rep, rep_c, ctx)
    mz_sync_gpu_to_cpu(dyn, dyn_c, ctx)
    mz_sync_gpu_to_cpu(pred, pred_c, ctx)

    var rep_a = MZRepCPU[OBS, LATENT, REP](
        net=untracked(Pointer(to=rep_c))
    )
    var dyn_a = MZDynCPU[LATENT, ACT, BINS, DYN](
        net=untracked(Pointer(to=dyn_c)),
        v_min=v_min, v_max=v_max
    )
    var pred_a = MZPredCPU[LATENT, ACT, BINS, PRED](
        net=untracked(Pointer(to=pred_c)),
        v_min=v_min, v_max=v_max
    )

    # training batch slabs (time-major) — owned Lists (RAII). Filled by the
    # replay's List API; the GPU unroll H2Ds them (list.unsafe_ptr() inside).
    var t_obs0 = List[Scalar[DT]](length=B * OBS, fill=0)
    var t_act = List[Scalar[DT]](length=K * B, fill=0)
    var t_pol = List[Scalar[DT]](length=(K + 1) * B * ACT, fill=0)
    var t_val = List[Scalar[DT]](length=(K + 1) * B, fill=0)
    var t_rew = List[Scalar[DT]](length=K * B, fill=0)

    # persistent GPU unroll scratch — allocated once, reused every train step
    # (per-step `enqueue_create_buffer` in the hot loop explodes disk on NVIDIA)
    var scratch = MZScratch[B, K, OBS, ACT, LATENT, BINS].make(ctx)

    # logger scratch: per-component loss split + root-prediction probe (D2H).
    var l_parts = _a(3)
    var d_diag_obs = ctx.enqueue_create_buffer[DT](B * OBS)
    var d_diag_z = ctx.enqueue_create_buffer[DT](B * LATENT)
    var d_diag_pred = ctx.enqueue_create_buffer[DT](B * (ACT + BINS))
    var h_diag_pred = _a(B * (ACT + BINS))

    # reanalyze scratch (owned Lists; read_obs/update_targets take Lists)
    var r_obs = List[Scalar[DT]](length=OBS, fill=0)
    var r_pol = List[Scalar[DT]](length=ACT, fill=0)

    # episode accumulation buffers
    var e_obs = List[Scalar[DT]]()
    var e_act = List[Scalar[DT]]()
    var e_rew = List[Scalar[DT]]()
    var e_pol = List[Scalar[DT]]()
    var e_val = List[Scalar[DT]]()
    var e_tp = List[Scalar[DT]]()
    var e_legal = List[Scalar[DT]]()
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
        # ── search (CPU mirror) ──
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
            e_legal.append(Scalar[DT](1.0))
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

        # ── train (GPU unroll) ──
        if it >= learning_starts and rb.num_episodes() > 0:
            for _ in range(train_per_iter):
                rb.sample_training_batch[B, K, N](
                    gamma, t_obs0, t_act, t_pol, t_val, t_rew
                )
                last_loss = Float64(
                    mz_unroll_train_step_gpu[
                        REP, DYN, PRED, B, K, OBS, ACT, LATENT, BINS
                    ](
                        ctx, rep, dyn, pred, orep, odyn, opred,
                        scratch,
                        t_obs0, t_act, t_pol, t_val, t_rew,
                        v_min, v_max, value_coef, max_grad_norm,
                        loss_parts=l_parts,
                    )
                )
            # refresh CPU mirror so the next search plans with fresh weights
            mz_sync_gpu_to_cpu(rep, rep_c, ctx)
            mz_sync_gpu_to_cpu(dyn, dyn_c, ctx)
            mz_sync_gpu_to_cpu(pred, pred_c, ctx)

        # ── per-batch diagnostics → logger (root pred re-forwarded on GPU) ──
        if (
            logger
            and diag_every > 0
            and it >= learning_starts
            and rb.num_episodes() > 0
            and (it + 1) % diag_every == 0
        ):
            _mz_emit_train_diag[REP, PRED, B, OBS, ACT, BINS, L](
                ctx, rep, pred, d_diag_obs, d_diag_z, d_diag_pred,
                h_diag_pred, t_obs0, t_pol, t_val, l_parts,
                v_min, v_max, last_loss, it + 1, logger.value(),
            )

        # ── reanalyze: refresh one stored position with a fresh search ──
        # The n-step targets bootstrap from STORED root values; without
        # refresh, never-evicted early episodes keep teaching weak-net values
        # and stale visit policies forever. Runs on the CPU mirror (just
        # synced above), like the per-step search.
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

    l_parts.unsafe_free(); h_diag_pred.unsafe_free()
    # keep the CPU mirrors alive past the loop — the MCTS adapters hold
    # `Pointer(to=rep_c)`, which the lifetime analyzer can't see.
    _ = rep_c^
    _ = dyn_c^
    _ = pred_c^
    return last_loss
