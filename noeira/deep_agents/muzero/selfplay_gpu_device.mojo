"""MuZero single-player self-play drivers (GPU) — fully on-device search + train.

Unlike `selfplay_gpu.mojo` (the CPU-search / GPU-train HYBRID with its per-step
checkpoint-string mirror sync), these drivers keep the three nets resident on
the device for BOTH halves: the MCTS search runs on the GPU orchestrators over
the on-device ``h/g/f`` via the MuZero GPU adapters (`MZRepGPU` / `MZDynGPU` /
`MZPredGPU`), and the K-step BPTT unroll trains the same nets in place
(`mz_unroll_train_step_gpu`). No CPU mirror, no sync — the only host↔device
traffic in the collection loop is the per-step root obs up and the visit
policy / root value down (the EZv2 GPU driver's proven shape).

Two planner flavors:

  * ``run_muzero_selfplay_gpu_device`` — **vanilla MuZero**: `GenericGPUMCTS`
    with `MuZeroPUCT[19652, 1.25]` + root `DirichletNoise[0.25, 0.25]`, exactly
    the algorithm of the converged CPU lighthouse. Greedy eval uses a SECOND
    planner instance with `NoNoise` (noise is a comptime trait, not a runtime
    flag — the AlphaZero eval pattern).
  * ``run_muzero_gumbel_selfplay_gpu`` — **Gumbel MuZero**: `GumbelGPUMCTS`
    (Gumbel-Top-k root sampling + sequential halving). The stored policy target
    is the *improved policy*; greedy eval is its argmax (Gumbel noise is part
    of the algorithm, no separate eval planner needed).

Both carry the full CartPole-500 fix stack: ∝ visits^(1/T) action sampling with
the legacy temperature schedule, truncation-aware episode storage (time-limit
cut ≠ terminal), and per-iteration reanalyze (fresh search on a stored position
→ overwrite its policy/value targets). ``v_min``/``v_max`` are the h-space
support shared by the planner decode and the two-hot training targets.

NOTE: `GenericGPUMCTS` requires ``NUM_SIMS % BATCH_SIMS == 0`` (the CPU search
handles a remainder round; the GPU one does not) — asserted at compile time.
"""

from std.math import exp, log
from std.memory import alloc
from layout import Layout, LayoutTensor
from max.gpu.host import DeviceContext, DeviceBuffer

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
    GenericGPUMCTS,
    GumbelGPUMCTS,
    MuZeroPUCT,
    DirichletNoise,
    NoNoise,
    SinglePlayer,
)

from .blocks import mz_unroll_train_step_gpu, MZScratch
from ..zero.mcts_adapters_mz import MZRepGPU, MZDynGPU, MZPredGPU
from ..zero.sequence_replay_mcts import MCTSSequenceReplay
from ..zero.temperature import visit_temperature


def _a(n: Int) -> Pointer[Scalar[DT], MutAnyOrigin]:
    """Raw scratch for optional unroll outputs (loss_parts) + diag host buffers
    — function-local, the unroll's optional-output params are Optional[Pointer]."""
    return alloc[Scalar[DT]]({count = n}).unsafe_leak().as_unsafe_any_origin()


def _mz_emit_train_diag[
    REP: Module, PRED: Module,
    B: Int, OBS: Int, ACT: Int, BINS: Int, L: Logger,
](
    ctx: DeviceContext,
    mut rep: REP,
    mut pred: PRED,
    d_obs: DeviceBuffer[DT],
    d_z: DeviceBuffer[DT],
    d_pred: DeviceBuffer[DT],
    h_pred: Pointer[Scalar[DT], MutAnyOrigin],
    t_obs0: List[Scalar[DT]],
    t_pol: List[Scalar[DT]],
    t_val: List[Scalar[DT]],
    l_parts: Pointer[Scalar[DT], MutAnyOrigin],
    v_min: Scalar[DT],
    v_max: Scalar[DT],
    last_loss: Float64,
    step: Int,
    logger: Pointer[L, MutAnyOrigin],
) raises:
    """Re-forward the root prediction on device, D2H, and emit the MuZero
    per-batch metric set (loss + policy/value/reward split + head-fit diagnostics)
    to ``logger``. Shared by both GPU device drivers. The forward runs on owned
    storage Tensors; the legacy d_obs/d_z/d_pred device scratch is unused now."""
    comptime LAT = REP.OUT_DIM
    comptime PRED_OUT = ACT + BINS
    _ = d_obs; _ = d_z; _ = d_pred
    var obs_sc = Tensor()
    obs_sc.ensure_gpu(ctx, B * OBS)
    # H2D the host batch obs into owned scratch (sanctioned staging boundary).
    ctx.enqueue_copy(obs_sc.dev.value(), t_obs0.unsafe_ptr())
    var z_sc = Tensor()
    z_sc.ensure_gpu(ctx, B * LAT)
    call_forward["gpu", B](rep, TensorRefs[REP.ARITY](obs_sc), z_sc, Optional(ctx))
    var pred_sc = Tensor()
    pred_sc.ensure_gpu(ctx, B * PRED_OUT)
    call_forward["gpu", B](pred, TensorRefs[PRED.ARITY](z_sc), pred_sc, Optional(ctx))
    ctx.enqueue_copy(h_pred, pred_sc.dev.value())
    ctx.synchronize()
    var dn = List[String]()
    var dv = List[Float64]()
    dn.append(String("loss")); dv.append(last_loss)
    dn.append(String("policy_loss")); dv.append(Float64(l_parts[unsafe_offset=0]))
    dn.append(String("value_loss")); dv.append(Float64(l_parts[unsafe_offset=1]))
    dn.append(String("reward_loss")); dv.append(Float64(l_parts[unsafe_offset=2]))
    append_mz_train_diagnostics[ACT, BINS, B](
        h_pred, t_pol, t_val, v_min, v_max, dn, dv
    )
    logger[].log_scalars(dn, dv, step)


def run_muzero_selfplay_gpu_device[
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
    # SERIAL search by default. The GPU batched-leaf path (BATCH_SIMS>1 +
    # virtual loss) selects every leaf of a round against a FROZEN tree —
    # unlike the CPU search, which expands during selection so later sims in
    # a round descend through earlier sims' children. The GPU variant
    # re-expands duplicate edges and double-counts their shallow values:
    # measured ~+1.3 systematic root-value bias and occasional argmax flips
    # vs CPU on identical nets (test_mz_search_gpu_cpu_parity), which broke
    # the 60k CartPole run (greedy stuck ~120 while noisy training hit 250).
    # At BATCH_SIMS=1/VLOSS=0 the GPU search is bit-near-identical to the
    # converged CPU search (same test). Raise only with that tradeoff known.
    BATCH_SIMS: Int = 1,
    VIRTUAL_LOSS: Int = 0,
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
    comptime assert NUM_SIMS % BATCH_SIMS == 0, (
        "GenericGPUMCTS needs NUM_SIMS to be a multiple of BATCH_SIMS"
    )
    comptime N_ENVS = 1

    # ── on-device planners + net adapters ──
    # Collection planner: root Dirichlet noise on (exploration). Eval planner:
    # NoNoise — noise is a comptime trait on GenericGPUMCTS, not a runtime arg.
    var planner = GenericGPUMCTS[
        N_ENVS, ACT, LATENT, BINS, MAX_NODES, NUM_SIMS, BATCH_SIMS,
        MuZeroPUCT[19652.0, 1.25], DirichletNoise[0.25, 0.25], SinglePlayer,
        0, VIRTUAL_LOSS,
    ](ctx, gamma=Float64(gamma), v_min=Float64(v_min), v_max=Float64(v_max))
    var eval_planner = GenericGPUMCTS[
        N_ENVS, ACT, LATENT, BINS, MAX_NODES, NUM_SIMS, BATCH_SIMS,
        MuZeroPUCT[19652.0, 1.25], NoNoise, SinglePlayer,
        0, VIRTUAL_LOSS,
    ](ctx, gamma=Float64(gamma), v_min=Float64(v_min), v_max=Float64(v_max))
    var rep_a = MZRepGPU[OBS, LATENT, REP].make(rep)
    var dyn_a = MZDynGPU[LATENT, ACT, BINS, DYN].make(dyn)
    var pred_a = MZPredGPU[LATENT, ACT, BINS, PRED].make(pred)

    var rb = MCTSSequenceReplay[OBS, ACT, CAP](seed=seed ^ UInt64(0xABCDEF))

    # ── device obs buffer (single env) + host mirrors (owned Lists; h_obs/h_pol
    # feed the List read_obs/update_targets, and H2D/D2H via .unsafe_ptr()) ──
    var d_obs = ctx.enqueue_create_buffer[DT](N_ENVS * OBS)
    var h_obs = List[Scalar[DT]](length=N_ENVS * OBS, fill=0)
    var h_pol = List[Scalar[DT]](length=N_ENVS * ACT, fill=0)
    var h_val = List[Scalar[DT]](length=N_ENVS, fill=0)

    # ── training batch slabs (time-major) — owned Lists (RAII) ──
    var t_obs0 = List[Scalar[DT]](length=B * OBS, fill=0)
    var t_act = List[Scalar[DT]](length=K * B, fill=0)
    var t_pol = List[Scalar[DT]](length=(K + 1) * B * ACT, fill=0)
    var t_val = List[Scalar[DT]](length=(K + 1) * B, fill=0)
    var t_rew = List[Scalar[DT]](length=K * B, fill=0)

    # persistent GPU unroll scratch (no per-step enqueue_create_buffer)
    var scratch = MZScratch[B, K, OBS, ACT, LATENT, BINS].make(ctx)

    # logger scratch: per-component loss split + root-prediction probe (D2H).
    var l_parts = _a(3)
    var d_diag_obs = ctx.enqueue_create_buffer[DT](B * OBS)
    var d_diag_z = ctx.enqueue_create_buffer[DT](B * LATENT)
    var d_diag_pred = ctx.enqueue_create_buffer[DT](B * (ACT + BINS))
    var h_diag_pred = _a(B * (ACT + BINS))

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
    var mcts_seed = UInt32(seed & UInt64(0xFFFF))
    var last_loss = 0.0
    var ep_returns = List[Float64]()

    var cur = env.reset_obs_list()
    var cur_f = List[Float64]()
    for j in range(OBS):
        cur_f.append(Float64(cur[j]))
    var ep_return = 0.0

    for it in range(iterations):
        # ── GPU search over the current obs ──
        for j in range(OBS):
            h_obs[j] = Scalar[DT](cur_f[j])
        ctx.enqueue_copy(d_obs, h_obs.unsafe_ptr())
        var obs_t = LayoutTensor[DT, Layout.row_major(N_ENVS, OBS),
            MutAnyOrigin](d_obs.unsafe_ptr().as_unsafe_any_origin())
        planner.search_gpu[
            type_of(rep_a),
            type_of(dyn_a),
            type_of(pred_a),
        ](ctx, rep_a, dyn_a, pred_a, obs_t, rng_seed=mcts_seed)
        mcts_seed += UInt32(1)

        ctx.enqueue_copy(h_pol.unsafe_ptr(), planner.policies_out)
        ctx.enqueue_copy(h_val.unsafe_ptr(), planner.root_value_out)
        ctx.synchronize()

        var root_v = Float64(h_val[0])

        # ── sample action ∝ visits^(1/T) ──
        # The *stored* policy target stays the untempered visit distribution.
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

        # ── record step (o_t, a_t, π_t, v_t, to_play=0) ──
        for j in range(OBS):
            e_obs.append(Scalar[DT](cur_f[j]))
        e_act.append(Scalar[DT](action))
        for a in range(ACT):
            e_pol.append(h_pol[a])
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

        # ── train (GPU unroll on the SAME resident nets — no mirror sync) ──
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

        # ── per-batch diagnostics → logger (root pred re-forwarded) ──
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
        # Re-search the stored obs with the current on-device nets (noisy
        # collection planner, consistent with the stored originals) and
        # overwrite the stored policy + root value.
        if (
            reanalyze_every > 0
            and it >= learning_starts
            and (it + 1) % reanalyze_every == 0
            and rb.num_episodes() > 0
        ):
            var rpos = rb.sample_position()
            rb.read_obs(rpos[0], rpos[1], h_obs)
            ctx.enqueue_copy(d_obs, h_obs.unsafe_ptr())
            var robs_t = LayoutTensor[DT, Layout.row_major(N_ENVS, OBS),
                MutAnyOrigin](d_obs.unsafe_ptr().as_unsafe_any_origin())
            planner.search_gpu[
                type_of(rep_a),
                type_of(dyn_a),
                type_of(pred_a),
            ](ctx, rep_a, dyn_a, pred_a, robs_t, rng_seed=mcts_seed)
            mcts_seed += UInt32(1)
            ctx.enqueue_copy(h_pol.unsafe_ptr(), planner.policies_out)
            ctx.enqueue_copy(h_val.unsafe_ptr(), planner.root_value_out)
            ctx.synchronize()
            rb.update_targets(rpos[0], rpos[1], h_pol, h_val[0])

        # ── periodic GREEDY eval (NoNoise planner, argmax visits) ──
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
                    eval_planner.search_gpu[
                        type_of(rep_a),
                        type_of(dyn_a),
                        type_of(pred_a),
                    ](ctx, rep_a, dyn_a, pred_a, eobs_t, rng_seed=mcts_seed)
                    mcts_seed += UInt32(1)
                    ctx.enqueue_copy(h_pol.unsafe_ptr(), eval_planner.policies_out)
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
    return last_loss


def run_muzero_gumbel_selfplay_gpu[
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
    MAX_K: Int,
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
    """Gumbel MuZero: same loop as `run_muzero_selfplay_gpu_device` but the
    search is `GumbelGPUMCTS` (Gumbel-Top-k + sequential halving). The stored
    policy target is the **improved policy**; greedy eval is its argmax with
    the same planner (Gumbel root sampling is the algorithm's exploration —
    there is no separate Dirichlet noise to switch off)."""
    comptime N_ENVS = 1

    var planner = GumbelGPUMCTS[
        N_ENVS, ACT, LATENT, BINS, MAX_NODES, MAX_K, NUM_SIMS, SinglePlayer
    ](
        ctx, gamma=Float64(gamma), v_min=Float64(v_min), v_max=Float64(v_max),
        # Tree-GLOBAL sigma(Q) normalization: per-node rescale is degenerate
        # at small ACT (CartPole ACT=2 -> qn in {0,1} exactly -> confident-
        # noise one-hot targets, no convergence). See qnorm_per_node doc.
        qnorm_per_node=False,
    )
    var rep_a = MZRepGPU[OBS, LATENT, REP].make(rep)
    var dyn_a = MZDynGPU[LATENT, ACT, BINS, DYN].make(dyn)
    var pred_a = MZPredGPU[LATENT, ACT, BINS, PRED].make(pred)

    var rb = MCTSSequenceReplay[OBS, ACT, CAP](seed=seed ^ UInt64(0xABCDEF))

    var d_obs = ctx.enqueue_create_buffer[DT](N_ENVS * OBS)
    var h_obs = List[Scalar[DT]](length=N_ENVS * OBS, fill=0)
    var h_pol = List[Scalar[DT]](length=N_ENVS * ACT, fill=0)
    var h_val = List[Scalar[DT]](length=N_ENVS, fill=0)

    var t_obs0 = List[Scalar[DT]](length=B * OBS, fill=0)
    var t_act = List[Scalar[DT]](length=K * B, fill=0)
    var t_pol = List[Scalar[DT]](length=(K + 1) * B * ACT, fill=0)
    var t_val = List[Scalar[DT]](length=(K + 1) * B, fill=0)
    var t_rew = List[Scalar[DT]](length=K * B, fill=0)

    var scratch = MZScratch[B, K, OBS, ACT, LATENT, BINS].make(ctx)

    # logger scratch: per-component loss split + root-prediction probe (D2H).
    var l_parts = _a(3)
    var d_diag_obs = ctx.enqueue_create_buffer[DT](B * OBS)
    var d_diag_z = ctx.enqueue_create_buffer[DT](B * LATENT)
    var d_diag_pred = ctx.enqueue_create_buffer[DT](B * (ACT + BINS))
    var h_diag_pred = _a(B * (ACT + BINS))

    var e_obs = List[Scalar[DT]]()
    var e_act = List[Scalar[DT]]()
    var e_rew = List[Scalar[DT]]()
    var e_pol = List[Scalar[DT]]()
    var e_val = List[Scalar[DT]]()
    var e_tp = List[Scalar[DT]]()
    var e_legal = List[Scalar[DT]]()
    var ep_len = 0

    var rng = seed ^ UInt64(0x123456789)
    var mcts_seed = UInt32(seed & UInt64(0xFFFF))
    var last_loss = 0.0
    var ep_returns = List[Float64]()

    var cur = env.reset_obs_list()
    var cur_f = List[Float64]()
    for j in range(OBS):
        cur_f.append(Float64(cur[j]))
    var ep_return = 0.0

    for it in range(iterations):
        # ── GPU Gumbel search over the current obs ──
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
            e_pol.append(h_pol[a])
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

        # ── train (GPU unroll, resident nets) ──
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

        # ── per-batch diagnostics → logger (root pred re-forwarded) ──
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
        if (
            reanalyze_every > 0
            and it >= learning_starts
            and (it + 1) % reanalyze_every == 0
            and rb.num_episodes() > 0
        ):
            var rpos = rb.sample_position()
            rb.read_obs(rpos[0], rpos[1], h_obs)
            ctx.enqueue_copy(d_obs, h_obs.unsafe_ptr())
            var robs_t = LayoutTensor[DT, Layout.row_major(N_ENVS, OBS),
                MutAnyOrigin](d_obs.unsafe_ptr().as_unsafe_any_origin())
            planner.search_gpu[
                type_of(rep_a),
                type_of(dyn_a),
                type_of(pred_a),
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
                        type_of(rep_a),
                        type_of(dyn_a),
                        type_of(pred_a),
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

    l_parts.unsafe_free(); h_diag_pred.unsafe_free()
    return last_loss
