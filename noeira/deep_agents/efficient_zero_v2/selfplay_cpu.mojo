"""EfficientZeroV2 discrete self-play driver (CPU) — the CartPole lighthouse.

The MuZero CPU self-play loop (`muzero/selfplay_cpu.mojo`) with the EZv2 training
step: every piece of data collection is identical (CPU MCTS over the learned
model `h/g/f`, visit-policy action sampling, sequence replay), but the update
calls ``ezv2_unroll_train_step_cpu`` — MuZero BPTT **plus** the SimSiam
temporal-consistency objective. The replay sampler is the obs-sequence variant
(`sample_training_batch_seq`) so the consistency targets see the real future
observations.

Per decision D1 the CPU path uses the **vanilla** ``GenericCPUMCTS`` (PUCT), not
Gumbel — the GPU path gets the Gumbel planner. The projector/predictor nets carry
BatchNorm but are consistency-only (never used at MCTS inference), so no BN
train/eval toggle is needed here. Returns the last training loss.
"""

from noeira.nn.core.ptr import untracked
from std.math import exp, log, sqrt
from std.memory import alloc

from noeira.nn.constants import DT
from noeira.nn.core.module import Module
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.call import call_forward, call_vjp
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

from .blocks import ezv2_unroll_train_step_cpu
from ..zero.mcts_adapters_mz_cpu import MZRepCPU, MZDynCPU, MZPredCPU
from ..zero.sequence_replay_mcts import MCTSSequenceReplay
from ..zero.temperature import visit_temperature


def _a(n: Int) -> Pointer[Scalar[DT], MutAnyOrigin]:
    return alloc[Scalar[DT]]({count = n}).unsafe_leak().as_unsafe_any_origin()


def _mean_perdim_std[ROWS: Int, DIM: Int](
    x: Pointer[Scalar[DT], MutAnyOrigin],
) -> Float64:
    """Mean over the ``DIM`` features of the per-feature std across the ``ROWS``
    rows. Collapse signal: → 0 means every row maps to ~the same vector (the
    representation can no longer distinguish observations)."""
    var acc = 0.0
    for d in range(DIM):
        var m = 0.0
        for b in range(ROWS):
            m += Float64(x[unsafe_offset=b * DIM + d])
        m /= Float64(ROWS)
        var v = 0.0
        for b in range(ROWS):
            var diff = Float64(x[unsafe_offset=b * DIM + d]) - m
            v += diff * diff
        v /= Float64(ROWS)
        acc += sqrt(v)
    return acc / Float64(DIM)


def _collapse_diag[
    REP: Module, PROJM: Module, B: Int, OBS: Int, LATENT: Int,
](
    mut rep: REP,
    mut proj: PROJM,
    obs0: List[Scalar[DT]],   # [B, OBS]
    z_buf: Pointer[Scalar[DT], MutAnyOrigin],  # [B, LATENT] scratch
    p_buf: Pointer[Scalar[DT], MutAnyOrigin],  # [B, PROJ] scratch
) raises -> Tuple[Float64, Float64]:
    """Probe representation collapse on a batch of real observations. Returns
    ``(latent_std, proj_norm_std)``:
      * ``latent_std`` — mean per-dim std of ``z = h(obs0)`` across the batch.
        Healthy > 0; → 0 is full latent collapse.
      * ``proj_norm_std`` — same metric on the **L2-normalized** projector output
        ``g_proj(z)`` (the standard SimSiam collapse metric). For PROJ features
        uniformly spread on the sphere this sits near ``1/√PROJ``; → 0 means the
        projections collapsed to a single direction.
    Re-forwards rep/proj (clobbers their caches — safe: the next train step
    re-forwards before any vjp)."""
    comptime PROJ = PROJM.OUT_DIM
    # storage forward on owned Tensors; copy outputs back into the raw scratch
    # so the metric helpers below read them unchanged.
    var obs_t = Tensor.alloc(B * OBS)
    for i in range(B * OBS):
        obs_t.data[i] = obs0[i]
    var z_tn = Tensor.alloc(B * LATENT)
    call_forward["cpu", B](rep, TensorRefs[REP.ARITY](obs_t), z_tn, None)
    for i in range(B * LATENT):
        z_buf[unsafe_offset=i] = z_tn.data[i]
    var latent_std = _mean_perdim_std[B, LATENT](z_buf)

    var p_tn = Tensor.alloc(B * PROJ)
    call_forward["cpu", B](proj, TensorRefs[PROJM.ARITY](z_tn), p_tn, None)
    for i in range(B * PROJ):
        p_buf[unsafe_offset=i] = p_tn.data[i]
    # L2-normalize each row before measuring spread (SimSiam convention).
    for b in range(B):
        var nrm = 0.0
        for d in range(PROJ):
            nrm += Float64(p_buf[unsafe_offset=b * PROJ + d]) * Float64(p_buf[unsafe_offset=b * PROJ + d])
        nrm = sqrt(nrm) + 1e-12
        for d in range(PROJ):
            p_buf[unsafe_offset=b * PROJ + d] = Scalar[DT](Float64(p_buf[unsafe_offset=b * PROJ + d]) / nrm)
    var proj_norm_std = _mean_perdim_std[B, PROJ](p_buf)
    return (latent_std, proj_norm_std)


def run_ezv2_selfplay_cpu[
    ENV: BoxDiscreteActionEnv,
    REP: Module,
    DYN: Module,
    PRED: Module,
    PROJM: Module,
    PREDH: Module,
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
    gamma: Scalar[DT] = Scalar[DT](0.997),
    v_min: Scalar[DT] = Scalar[DT](-10.0),
    v_max: Scalar[DT] = Scalar[DT](10.0),
    seed: UInt64 = 0,
    max_ep_steps: Int = 500,
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
    var mcts = GenericCPUMCTS[
        ACT, LATENT, NUM_SIMS, MAX_NODES,
        MuZeroPUCT[19652.0, 1.25], DirichletNoise[0.25, 0.25], SinglePlayer,
        BATCH_SIMS, VIRTUAL_LOSS,
    ](gamma=Float64(gamma))
    var rb = MCTSSequenceReplay[OBS, ACT, CAP](seed=seed ^ UInt64(0xABCDEF))

    var rep_a = MZRepCPU[OBS, LATENT, REP](net=untracked(Pointer(to=rep)))
    var dyn_a = MZDynCPU[LATENT, ACT, BINS, DYN](
        net=untracked(Pointer(to=dyn)), v_min=v_min, v_max=v_max
    )
    var pred_a = MZPredCPU[LATENT, ACT, BINS, PRED](
        net=untracked(Pointer(to=pred)), v_min=v_min, v_max=v_max
    )

    # training batch slabs (time-major) — owned Lists (RAII), fed to the List
    # replay+unroll APIs. obs is the full [K+1, B, OBS] sequence.
    var t_obs_seq = List[Scalar[DT]](length=(K + 1) * B * OBS, fill=0)
    var t_act = List[Scalar[DT]](length=K * B, fill=0)
    var t_pol = List[Scalar[DT]](length=(K + 1) * B * ACT, fill=0)
    var t_val = List[Scalar[DT]](length=(K + 1) * B, fill=0)
    var t_rew = List[Scalar[DT]](length=K * B, fill=0)
    var t_cmask = _a(K * B)   # consistency mask — cons_mask Optional[Pointer]

    # reanalyze scratch (owned Lists; read_obs/update_targets take Lists)
    var r_obs = List[Scalar[DT]](length=OBS, fill=0)
    var r_pol = List[Scalar[DT]](length=ACT, fill=0)

    # collapse-diagnostic scratch (latent + projection probe on obs0)
    var d_z = _a(B * LATENT)
    var d_p = _a(B * PROJM.OUT_DIM)
    # logger scratch: per-component loss breakdown + root prediction probe
    var l_parts = _a(4)            # [policy, value, reward, consistency] means
    var d_pred = _a(B * (ACT + BINS))

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
        var policy = mcts.search[type_of(rep_a), type_of(dyn_a), type_of(pred_a)](
            rep_a, dyn_a, pred_a, cur_f, add_noise=True
        )
        var root_v = mcts.root_value()

        # sample ∝ visits^(1/T); stored policy target stays untempered.
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

        for j in range(OBS):
            e_obs.append(Scalar[DT](cur_f[j]))
        e_act.append(Scalar[DT](action))
        for a in range(ACT):
            e_pol.append(Scalar[DT](policy[a]))
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
            # Time-limit cut is NOT a terminal — bootstrap past it.
            rb.store_episode(
                e_obs,
                e_act,
                e_rew,
                e_pol,
                e_val,
                e_tp,
                e_legal,
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

        if it >= learning_starts and rb.num_episodes() > 0:
            for _ in range(train_per_iter):
                rb.sample_training_batch_seq[B, K, N](
                    gamma, t_obs_seq, t_act, t_pol, t_val, t_rew,
                    cons_mask=t_cmask,
                )
                last_loss = Float64(
                    ezv2_unroll_train_step_cpu[
                        REP, DYN, PRED, PROJM, PREDH,
                        B, K, OBS, ACT, LATENT, BINS,
                    ](
                        rep, dyn, pred, proj, predh,
                        orep, odyn, opred, oproj, opredh,
                        t_obs_seq, t_act, t_pol, t_val, t_rew,
                        v_min, v_max, value_coef, consistency_coef,
                        cons_mask=t_cmask,
                        loss_parts=l_parts,
                    )
                )

        # ── per-batch diagnostics: collapse probe + logger metric emission ──
        if (
            diag_every > 0
            and it >= learning_starts
            and rb.num_episodes() > 0
            and (it + 1) % diag_every == 0
        ):
            var cd = _collapse_diag[REP, PROJM, B, OBS, LATENT](
                rep, proj, t_obs_seq, d_z, d_p
            )
            if verbose:
                print(
                    "  [collapse] step", it + 1,
                    "latent_std", cd[0], "proj_norm_std", cd[1],
                )
            if logger:
                # root prediction (reuse d_z = h(obs0) from the probe).
                var z_in = Tensor.alloc(B * LATENT)
                for i in range(B * LATENT):
                    z_in.data[i] = d_z[unsafe_offset=i]
                var pred_tn = Tensor.alloc(B * (ACT + BINS))
                call_forward["cpu", B](pred, TensorRefs[PRED.ARITY](z_in), pred_tn, None)
                for i in range(B * (ACT + BINS)):
                    d_pred[unsafe_offset=i] = pred_tn.data[i]
                var dn = List[String]()
                var dv = List[Float64]()
                dn.append(String("loss")); dv.append(last_loss)
                dn.append(String("policy_loss")); dv.append(Float64(l_parts[unsafe_offset=0]))
                dn.append(String("value_loss")); dv.append(Float64(l_parts[unsafe_offset=1]))
                dn.append(String("reward_loss")); dv.append(Float64(l_parts[unsafe_offset=2]))
                dn.append(String("consistency_loss"))
                dv.append(Float64(l_parts[unsafe_offset=3]))
                dn.append(String("latent_std")); dv.append(cd[0])
                dn.append(String("proj_norm_std")); dv.append(cd[1])
                append_mz_train_diagnostics[ACT, BINS, B](
                    d_pred, t_pol, t_val, v_min, v_max, dn, dv
                )
                logger.value()[].log_scalars(dn, dv, it + 1)

        # ── reanalyze: refresh `reanalyze_batch` stored positions per trigger ──
        # Lifting `reanalyze_batch` above 1 raises coverage so a larger fraction
        # of the buffer carries fresh-net targets (the EfficientZero coverage
        # lever; mirrors the GPU driver).
        if (
            reanalyze_every > 0
            and it >= learning_starts
            and (it + 1) % reanalyze_every == 0
            and rb.num_episodes() > 0
        ):
            for _ra in range(reanalyze_batch):
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

    t_cmask.unsafe_free()
    d_z.unsafe_free(); d_p.unsafe_free()
    l_parts.unsafe_free(); d_pred.unsafe_free()
    return last_loss
