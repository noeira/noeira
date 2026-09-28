"""Arena-gated AlphaZero self-play with symmetry augmentation — full AlphaZero.

Extends `run_alphazero_selfplay` with the two production pieces:

  * **Best/learner split + Arena gating.** A frozen *best* net generates all
    self-play (MCTS expansion + the policy/value priors); a *learner* net trains
    on that data. Every `arena_every` iterations the learner plays the best in
    the Arena (`candidate_winrate_mcts` — both nets at full MCTS strength, both
    colors, random openings); if it wins a clear majority of decisive games
    (`should_promote`) it is copied over the best (`hard_copy_params`) and
    becomes the new generator. The gate runs MCTS (not argmax) because that is
    the condition the winner is used in — self-play generation is always net+MCTS
    — and because it exercises the value head, which argmax ignores. This
    accept/reject loop stops a transiently-worse learner from poisoning the
    self-play distribution.

  * **Symmetry augmentation.** Each recorded `(obs, π)` sample is replicated
    under the board's symmetry group (`AUG`, e.g. TicTacToe's 8 D4 symmetries);
    the scalar value target `z` is symmetry-invariant and copied. Multiplies
    effective data and bakes in the game's invariances.

The passed-in `net` is the *best* and holds the final (best) weights on return.
`AUG = IdentityAugmenter` recovers the un-augmented behaviour.

**Telemetry.** Two `GPUEvaluator` opponents (`OPP1`, `OPP2`, e.g. random +
minimax) and a `Logger` (`L`) can be plugged in. Two cadences feed the logger:

  * **`report_every` (expensive).** The best net is evaluated (both colors) via
    full MCTS against each enabled opponent, a one-line progress summary is
    printed, and the eval win/draw/loss + win-rate (plus loss, replay size,
    promotions) are flushed. MCTS eval is slow, so this is necessarily coarse.

  * **`diag_every` (cheap).** Per-batch *training* diagnostics — policy CE,
    policy/target entropy, target max-prob, the policy KL gap, value MSE/mean,
    and value-target stats (`append_az_train_diagnostics`) — flushed against the
    same train axis. Computed from the last train batch's net output (`pred`
    graph node, one small D2H) + its targets, so it can run far more often than
    the eval, giving the dense curves the legacy `train_selfplay_gpu` logged.

With the defaults (`OPP*=RandomOpponent`, `L=NoOpLogger`, `report_every=0`,
`diag_every=0`) both are no-ops and the run matches the original silent
behaviour.
"""

from noeira.nn.core.param import walk_params
from std.math import exp, log, tanh
from std.memory import Pointer
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor

from noeira.nn.constants import DT
from noeira.nn.core.module import Module
from noeira.nn.core.tensor import Tensor
from noeira.nn.optimizer.adam import Adam
from noeira.nn.optimizer.grad_clip import clip_grad_norm
from noeira.nn.core.initializer import Zero, Kaiming
from noeira.nn.core.hard_copy import hard_copy
from noeira.nn.combinators.compute_graph import ComputeGraph
from noeira.nn.combinators.graph_decl import (
    InputSlot, Node, ExternalNode,
)
from noeira.core.env_traits import GPUTwoPlayerDiscreteEnv
from noeira.core.logger import Logger, NoOpLogger
from noeira.planners.tree_search import (
    GenericGPUMCTS,
    AlphaGoPUCT,
    DirichletNoise,
    SelfPlay,
)

from .loss_ops import AZLossOp
from .arena import candidate_winrate_mcts, should_promote
from .eval import eval_mcts_vs_opponent, EvalResult
from ..zero.mcts_adapters import AZPredGPU, AZEnvGPU
from ..zero.example_replay import MCTSExampleReplay
from ..zero.symmetries import BoardAugmenter
from ..zero.evaluators import GPUEvaluator, RandomOpponent


@fieldwise_init
struct ArenaRunResult(Copyable, Movable):
    """Summary of an arena-gated run: last mean train loss + how many times the
    learner was accepted over the best."""

    var last_loss: Float64
    var promotions: Int


def append_az_train_diagnostics[
    ACT: Int, BATCH: Int
](
    pred: List[Scalar[DT]],
    tgt: List[Scalar[DT]],
    mut names: List[String],
    mut values: List[Float64],
):
    """Append the AlphaZero per-batch training diagnostics (legacy parity) to
    `names`/`values`, computed on the host from the last train batch.

    `pred` is the net output `[policy_logits(ACT) | raw_value(1)]` (BATCH·W);
    `tgt` is the packed self-play target `[mcts_policy(ACT) | z(1)]` (BATCH·W).
    Both are plain host buffers (the GPU caller D2H-copies the `pred` graph node
    first; the CPU graph writes host buffers directly).

    Mirrors `deep_agents/alphazero/alphazero.mojo`'s diag block — the value head
    is `tanh`-squashed before the MSE (AlphaZero value ∈ [-1, 1]), the policy is
    soft-CE against the MCTS visit distribution. Metrics emitted:

      * ``policy_ce`` / ``policy_entropy`` — fit + sharpness of the policy head.
      * ``target_entropy`` / ``target_max_prob`` — are the MCTS targets sharp or
        near-uniform? (Uniform ⇒ search is not discriminating.)
      * ``policy_ce_minus_target_entropy`` — the policy KL gap, the real
        fit-quality number (0 ⇒ head matches the search distribution).
      * ``value_mse`` / ``value_mean`` — value-head fit + its mean output.
      * ``value_target_mean`` / ``value_target_pos_frac`` — what the value head
        is being asked to learn (label balance).

    NaN/inf entries are dropped by the logger's own guard, so no clamping here.
    """
    comptime W = ACT + 1
    var n = Float64(BATCH)
    var ce_sum: Float64 = 0.0
    var ent_sum: Float64 = 0.0
    var vmse_sum: Float64 = 0.0
    var vmean_sum: Float64 = 0.0
    var tent_sum: Float64 = 0.0
    var tmax_sum: Float64 = 0.0
    var vt_sum: Float64 = 0.0
    var vt_pos = 0
    for b in range(BATCH):
        var base = b * W

        # Policy: softmax(logits) → CE vs target π, plus pred-policy entropy.
        var maxl = Float64(pred[base])
        for a in range(1, ACT):
            var v = Float64(pred[base + a])
            if v > maxl:
                maxl = v
        var sume: Float64 = 0.0
        for a in range(ACT):
            sume += exp(Float64(pred[base + a]) - maxl)
        var ce: Float64 = 0.0
        var ent: Float64 = 0.0
        for a in range(ACT):
            var prob = exp(Float64(pred[base + a]) - maxl) / sume
            var t = Float64(tgt[base + a])
            if t > 1e-8:
                # Clamp, don't skip: skipping under-reports CE exactly when
                # the head is sharp (target mass on near-zero predictions),
                # which made CE read ~0 < target entropy — impossible for a
                # true cross-entropy.
                var p_cl = prob if prob > 1e-12 else 1e-12
                ce -= t * log(p_cl)
            if prob > 1e-8:
                ent -= prob * log(prob)
        ce_sum += ce
        ent_sum += ent

        # Value: tanh-squashed head vs z target.
        var tv = tanh(Float64(pred[base + ACT]))
        var z = Float64(tgt[base + ACT])
        vmse_sum += (tv - z) * (tv - z)
        vmean_sum += tv

        # MCTS target distribution stats (sharpness + label balance).
        var tent: Float64 = 0.0
        var tmax: Float64 = 0.0
        for a in range(ACT):
            var tp = Float64(tgt[base + a])
            if tp > 1e-8:
                tent -= tp * log(tp)
            if tp > tmax:
                tmax = tp
        tent_sum += tent
        tmax_sum += tmax
        vt_sum += z
        if z > 0.5:
            vt_pos += 1

    var policy_ce = ce_sum / n
    var target_entropy = tent_sum / n
    names.append(String("policy_ce"))
    values.append(policy_ce)
    names.append(String("policy_entropy"))
    values.append(ent_sum / n)
    names.append(String("target_entropy"))
    values.append(target_entropy)
    names.append(String("target_max_prob"))
    values.append(tmax_sum / n)
    names.append(String("policy_ce_minus_target_entropy"))
    values.append(policy_ce - target_entropy)
    names.append(String("value_mse"))
    values.append(vmse_sum / n)
    names.append(String("value_mean"))
    values.append(vmean_sum / n)
    names.append(String("value_target_mean"))
    values.append(vt_sum / n)
    names.append(String("value_target_pos_frac"))
    values.append(Float64(vt_pos) / n)


def _eval_both_colors[
    ENV: GPUTwoPlayerDiscreteEnv,
    NET: Module,
    OPP: GPUEvaluator,
    N_GAMES: Int,
    NUM_SIMS: Int,
    MAX_NODES: Int,
    MAX_PLIES: Int,
](
    ctx: DeviceContext, mut net: NET, seed: UInt64, open_plies: Int = 0
) raises -> EvalResult:
    """Aggregate the net's **MCTS** record vs `OPP` over `N_GAMES` games as
    *each* color (P0 then P1), so first-move advantage cancels — the legacy eval
    convention. The agent plays at full search strength (`eval_mcts_vs_opponent`)
    so the numbers reflect the deployed agent, not the bare policy head.
    `open_plies > 0` diversifies openings so a deterministic opponent yields a
    real winrate instead of one canonical line ×N (see eval_mcts_vs_opponent).
    """
    var p0 = eval_mcts_vs_opponent[
        ENV, NET, OPP, N_GAMES, NUM_SIMS, MAX_NODES, MAX_PLIES
    ](ctx, net, agent_player=0, seed=seed, open_plies=open_plies)
    var p1 = eval_mcts_vs_opponent[
        ENV, NET, OPP, N_GAMES, NUM_SIMS, MAX_NODES, MAX_PLIES
    ](ctx, net, agent_player=1, seed=seed + 33333, open_plies=open_plies)
    return EvalResult(
        wins=p0.wins + p1.wins,
        draws=p0.draws + p1.draws,
        losses=p0.losses + p1.losses,
    )


def run_alphazero_selfplay_arena[
    ENV: GPUTwoPlayerDiscreteEnv,
    NET: Module,
    AUG: BoardAugmenter,
    N_ENVS: Int,
    NUM_SIMS: Int,
    MAX_NODES: Int,
    BATCH: Int,
    CAP: Int,
    MAX_TRAJ: Int,
    ARENA_GAMES: Int = 32,  # arena games per color (comptime: sizes buffers)
    RESULT_IDX: Int = 10,  # vestigial (MCTS arena/eval attribute by
    #                               reward+turn); kept for caller API stability
    MAX_PLIES: Int = 9,
    OPP1: GPUEvaluator = RandomOpponent,  # primary eval opponent (do_eval)
    OPP2: GPUEvaluator = RandomOpponent,  # secondary eval opponent (do_eval2)
    L: Logger = NoOpLogger,  # metrics sink (NoOp = silent)
    EVAL_GAMES: Int = 64,  # games per color in each periodic eval
    TEMP_MOVES: Int = 4,  # plies sampled ∝ visits per game
    #                                        before switching to greedy (opening
    #                                        diversity); scale to game length
    BATCH_SIMS: Int = 1,  # MCTS leaves expanded per round
    #                                        (virtual-loss batching). >1 cuts net
    #                                        forwards by this factor: NUM_SIMS /
    #                                        BATCH_SIMS rounds. Must divide
    #                                        NUM_SIMS and be ≤ ACT to avoid
    #                                        forced within-round collisions.
](
    ctx: DeviceContext,
    mut net: NET,  # the BEST net — holds final weights on return
    iterations: Int,
    learning_starts: Int = 0,
    train_per_iter: Int = 1,
    lr: Scalar[DT] = Scalar[DT](0.01),
    seed: UInt64 = 0,
    arena_every: Int = 100,
    arena_open_plies: Int = 2,
    promote_threshold: Float64 = 0.55,
    report_every: Int = 0,  # 0 → fall back to arena_every (0 = no reports)
    diag_every: Int = 0,  # cheap per-batch train diagnostics every N
    #                               moves (train-axis), decoupled from the
    #                               expensive periodic eval; 0 = off
    do_eval: Bool = True,  # MCTS-eval the best vs OPP1 each report
    do_eval2: Bool = False,  # also MCTS-eval vs OPP2 each report
    verbose: Bool = True,  # print a per-report progress line
    logger: Optional[Pointer[L, MutAnyOrigin]] = None,
    max_grad_norm: Float64 = 0.0,  # global grad-norm clip (0 = off). The 5-block
    #                               ResNet at lr=2e-3 spikes grad norms → policy
    #                               CE climbs back past uniform → arena
    #                               regression; 1.0 is the legacy AlphaZero.jl
    #                               value and the #1 stability fix.
    weight_decay: Float64 = 0.0,  # decoupled (AdamW) weight decay (0 = off ≡
    #                               plain Adam; 1e-4 matches the legacy config).
    eval_open_plies: Int = 0,  # uniform-random legal plies opening each
    #                               periodic-eval game (both sides). 0 = the
    #                               canonical single-line "perfect-play gate";
    #                               ≥2 = diversified openings → real winrate.
) raises -> ArenaRunResult:
    # NOTE on units: one loop pass advances every one of `N_ENVS` games by a
    # single self-play *move* (not a full game). `iterations` is therefore the
    # number of self-play moves; `report_every` / `arena_every` are in moves. A
    # report prints the moves done, the cumulative finished-game count, the last
    # train loss, promotions, and the MCTS eval vs each opponent. This differs
    # from the legacy batch-then-train driver, whose "iter" was a whole
    # collect+train+eval round — hence the deliberate `move` / `games` labels.
    comptime OBS = NET.IN_DIMS[0]
    comptime ACT = NET.OUT_DIM - 1
    comptime W = NET.OUT_DIM  # ACT + 1
    comptime STATE = ENV.STATE_SIZE
    comptime NSYM = AUG.NUM_SYMMETRIES
    comptime MCTS = GenericGPUMCTS[
        N_ENVS,
        ACT,
        OBS,
        1,
        MAX_NODES,
        NUM_SIMS,
        BATCH_SIMS,
        AlphaGoPUCT[1.0],
        DirichletNoise[0.25, 0.25],
        SelfPlay,
        STATE_SIZE=STATE,
    ]
    comptime Graph = ComputeGraph[
        InputSlot["obs", OBS],
        ExternalNode["pred", NET, "obs"],
        InputSlot["tgt", W],
        Node["loss", AZLossOp[ACT], "pred", "tgt"],
    ]

    var octx = Optional[DeviceContext](ctx)
    # Learner net (trains) initialised to the best net's weights.
    var learner = NET.make["gpu", Kaiming](octx)
    hard_copy["gpu"](net, learner, octx)

    # Decoupled weight decay (AdamW) — `wd=0` ⇒ plain Adam (bit-identical, so
    # TicTacToe is unaffected). Per-param decay flags come from the net's Param
    # decay bits. The global grad-norm clip is applied separately each step via
    # `clip_grad_norm` (max_grad_norm <= 0 ⇒ no-op).
    var opt = Adam(lr=lr, wd=Scalar[DT](weight_decay))
    var graph = Graph.make["gpu", Zero](octx)
    var mcts = MCTS(ctx, gamma=1.0, v_min=-1.0, v_max=1.0)
    var replay = MCTSExampleReplay[OBS, W, CAP]()

    # ── Env / MCTS device buffers (category-B; RAII DeviceBuffer) ──
    var states = ctx.enqueue_create_buffer[DT](N_ENVS * STATE)
    var obs_dev = ctx.enqueue_create_buffer[DT](N_ENVS * OBS)
    var legal_dev = ctx.enqueue_create_buffer[DT](N_ENVS * ACT)
    var rew = ctx.enqueue_create_buffer[DT](N_ENVS)
    var done = ctx.enqueue_create_buffer[DT](N_ENVS)
    var term = ctx.enqueue_create_buffer[DT](N_ENVS)
    var obs_next = ctx.enqueue_create_buffer[DT](N_ENVS * OBS)
    var legal_next = ctx.enqueue_create_buffer[DT](N_ENVS * ACT)
    var ep_steps_dev = ctx.enqueue_create_buffer[DT](N_ENVS)  # per-env ply count

    # ── Host staging for the MCTS root obs / policy + step + temperature ──
    var obs_h = ctx.enqueue_create_host_buffer[DT](N_ENVS * OBS)
    var pol_h = ctx.enqueue_create_host_buffer[DT](N_ENVS * ACT)
    var done_h = ctx.enqueue_create_host_buffer[DT](N_ENVS)
    var rew_h = ctx.enqueue_create_host_buffer[DT](N_ENVS)
    var ep_steps_h = ctx.enqueue_create_host_buffer[DT](N_ENVS)  # per-env ply
    ctx.synchronize()

    # ── Host trajectory storage (per-env in-progress game) — owned Lists. ──
    var traj_obs = List[Scalar[DT]](length=N_ENVS * MAX_TRAJ * OBS, fill=0)
    var traj_pol = List[Scalar[DT]](length=N_ENVS * MAX_TRAJ * ACT, fill=0)
    var traj_len = List[Int](length=N_ENVS, fill=0)
    var tmp_tgt = List[Scalar[DT]](length=W, fill=0)
    var aug_obs = List[Scalar[DT]](length=OBS, fill=0)
    var aug_pol = List[Scalar[DT]](length=ACT, fill=0)

    # ── Train-batch storage Tensors (the nn surface) ──
    var obs_t = Tensor.alloc(BATCH * OBS)
    var tgt_t = Tensor.alloc(BATCH * W)
    var loss_t = Tensor.alloc_gpu(ctx, BATCH)
    var grad_t = Tensor.alloc(BATCH)
    for i in range(BATCH):
        grad_t.data[i] = Scalar[DT](1.0) / Scalar[DT](BATCH)
    grad_t.upload(ctx)  # constant 1/BATCH grad seed, uploaded once

    # ── Initialize all games ──
    ENV.reset_kernel_gpu[N_ENVS, STATE](ctx, states)
    ENV.extract_obs_kernel_gpu[N_ENVS, STATE, OBS](
        ctx, states, obs_dev, legal_dev
    )
    ctx.synchronize()

    var last_loss: Float64 = 0.0
    var promotions = 0
    var total_games = 0  # cumulative finished self-play games

    # Effective reporting cadence: explicit `report_every`, else piggy-back on
    # the arena cadence (0 ⇒ no periodic eval/print/log at all).
    var rep = report_every if report_every > 0 else arena_every
    if verbose:
        print(
            "AlphaZero self-play:",
            iterations,
            "moves,",
            N_ENVS,
            "envs,",
            NUM_SIMS,
            "sims/move | eval(MCTS)1=",
            OPP1.NAME,
            "eval2=",
            OPP2.NAME if do_eval2 else String("off"),
            "| report_every=",
            rep,
            "moves",
        )

    for it in range(iterations):
        # 1. MCTS search with the BEST net (eval mode for any BatchNorm).
        net.set_attr["training"](Scalar[DT](0.0))
        var pred = AZPredGPU[OBS, ACT, NET].make(net)
        var env_ad = AZEnvGPU[ENV, STATE, OBS, ACT]()
        var root_obs = LayoutTensor[DT, Layout.row_major(N_ENVS, OBS)](obs_dev)
        var root_legal = LayoutTensor[DT, Layout.row_major(N_ENVS * ACT)](
            legal_dev
        )
        mcts.search_gpu_alphazero[type_of(pred), type_of(env_ad)](
            ctx,
            pred,
            env_ad,
            root_obs,
            states,
            root_legal,
            rng_seed=seed + UInt64(it),
        )

        # 2. Pull root obs + visit-count policy, record into trajectory.
        ctx.enqueue_copy(obs_h, obs_dev)
        ctx.enqueue_copy(pol_h, mcts.policies_out)
        ctx.synchronize()
        for e in range(N_ENVS):
            # Current ply of env e's game = ep_steps for the temperature kernel.
            ep_steps_h[e] = Scalar[DT](traj_len[e])
            var k = traj_len[e]
            if k < MAX_TRAJ:
                var ob = (e * MAX_TRAJ + k) * OBS
                for j in range(OBS):
                    traj_obs[ob + j] = obs_h[e * OBS + j]
                var pb = (e * MAX_TRAJ + k) * ACT
                for a in range(ACT):
                    traj_pol[pb + a] = pol_h[e * ACT + a]
                traj_len[e] = k + 1

        # 2b. Apply the temperature schedule to the actual move: re-derive
        #     mcts.actions_out from the live visit tree — sample ∝ visits for the
        #     first TEMP_MOVES plies (per env), greedy after. The visit-count
        #     policy *target* was already recorded above from the pre-temp
        #     policies_out, so the one-hot policy this overwrites for greedy
        #     plies is harmless. The kernel is legal-mask aware.
        ctx.enqueue_copy(ep_steps_dev, ep_steps_h)
        var ep_t = LayoutTensor[DT, Layout.row_major(N_ENVS)](ep_steps_dev)
        var legal_t = LayoutTensor[DT, Layout.row_major(N_ENVS * ACT)](
            legal_dev
        )
        mcts.extract_actions_temp[TEMP_MOVES](
            ctx,
            ep_t,
            legal_t,
            rng_seed=UInt32((seed + UInt64(it)) & 0xFFFFFFFF),
            temp_min=0.0,
        )

        # 3. Step every game by its chosen action.
        ENV.step_kernel_gpu[N_ENVS, STATE, OBS](
            ctx,
            states,
            mcts.actions_out,
            rew,
            done,
            term,
            obs_next,
            legal_next,
        )
        ctx.enqueue_copy(done_h, done)
        ctx.enqueue_copy(rew_h, rew)
        ctx.synchronize()

        # 4. Flush finished games with value target z, augmented by symmetry.
        for e in range(N_ENVS):
            if done_h[e] > 0.5:
                total_games += 1
                var L = traj_len[e]
                var win = Float64(rew_h[e]) > 0.5
                for k in range(L):
                    var z: Float64 = 0.0
                    if win:
                        z = 1.0 if ((L - 1 - k) % 2 == 0) else -1.0
                    var ob = (e * MAX_TRAJ + k) * OBS
                    var pb = (e * MAX_TRAJ + k) * ACT
                    # Record guard: DROP plies with a non-finite search
                    # policy — a single NaN target column permanently NaNs
                    # that policy-head weight row via the soft-CE gradient.
                    # See selfplay_arena_gumbel.mojo (which also counts the
                    # drops in its sp[] diagnostics). Should never fire now
                    # that promotion copies BN running stats with the
                    # weights (hard_copy_params + hard_copy_states).
                    var pol_ok = True
                    for a in range(ACT):
                        var pv = Float64(traj_pol[pb + a])
                        if pv - pv != 0.0:
                            pol_ok = False
                            break
                    if not pol_ok:
                        continue
                    for s in range(NSYM):
                        AUG.augment_obs[OBS](traj_obs, ob, s, aug_obs)
                        AUG.augment_policy[ACT](traj_pol, pb, s, aug_pol)
                        for a in range(ACT):
                            tmp_tgt[a] = aug_pol[a]
                        tmp_tgt[ACT] = Scalar[DT](z)
                        replay.record(aug_obs, 0, tmp_tgt, 0)
                traj_len[e] = 0

        # 5. Reset finished games, refresh obs/legal.
        ENV.selective_reset_kernel_gpu[N_ENVS, STATE](
            ctx, states, done, rng_seed=seed + UInt64(it)
        )
        ENV.extract_obs_kernel_gpu[N_ENVS, STATE, OBS](
            ctx, states, obs_dev, legal_dev
        )
        ctx.synchronize()

        # 6. Train the LEARNER (train mode for BatchNorm).
        if len(replay) >= BATCH and it >= learning_starts:
            learner.set_attr["training"](Scalar[DT](1.0))
            for _t in range(train_per_iter):
                replay.sample_batch_tensors[BATCH](obs_t, tgt_t)
                obs_t.upload(ctx)
                tgt_t.upload(ctx)
                learner.zero_grad["gpu"](octx)
                graph.set_input["obs", BATCH](obs_t, octx)
                graph.set_input["tgt", BATCH](tgt_t, octx)
                graph.forward[BATCH, "gpu"](loss_t, octx, learner)
                graph.vjp[BATCH, "gpu"](grad_t, octx, learner)
                # Global grad-norm clip (max_grad_norm <= 0 ⇒ no-op), then step.
                _ = clip_grad_norm["gpu", NET](
                    learner, Scalar[DT](max_grad_norm), octx
                )
                opt.begin_step()
                walk_params["gpu"](learner, opt, octx)
            loss_t.download(ctx)
            var ml: Float64 = 0.0
            for b in range(BATCH):
                ml += Float64(loss_t.data[b])
            last_loss = ml / Float64(BATCH)

            # 6b. Dense per-batch training diagnostics (legacy parity), on the
            #     train axis and decoupled from the expensive periodic eval. The
            #     graph's "pred" node still holds the last train batch's net
            #     output; the matching targets are in `tgt_t`. One small D2H.
            if Bool(logger) and diag_every > 0 and (it + 1) % diag_every == 0:
                ref pred_node = graph.node_output["pred"]()
                pred_node.download(ctx)
                var dnames = List[String]()
                var dvalues = List[Float64]()
                dnames.append(String("loss"))
                dvalues.append(last_loss)
                append_az_train_diagnostics[ACT, BATCH](
                    pred_node.data, tgt_t.data, dnames, dvalues
                )
                logger.value()[].log_scalars(dnames, dvalues, it + 1)

        # 7. Arena gating: periodically challenge the best with the learner.
        if (
            it >= learning_starts
            and arena_every > 0
            and (it + 1) % arena_every == 0
            and len(replay) >= BATCH
        ):
            var rec = candidate_winrate_mcts[
                ENV,
                NET,
                NET,
                ARENA_GAMES,
                NUM_SIMS,
                MAX_NODES,
                MAX_PLIES,
            ](
                ctx,
                learner,
                net,
                seed=seed + UInt64(it) * 7 + 1,
                open_plies=arena_open_plies,
            )
            var accepted = should_promote(
                rec, promote_threshold, min_decisive=ARENA_GAMES // 2
            )
            if accepted:
                hard_copy["gpu"](learner, net, octx)
                promotions += 1
            if verbose:
                print(
                    "  arena @ move",
                    it + 1,
                    "| learner vs best (MCTS)  W",
                    rec.wins,
                    "D",
                    rec.draws,
                    "L",
                    rec.losses,
                    "→ ACCEPTED" if accepted else "→ rejected",
                    "(promotions",
                    promotions,
                    ")",
                )

        # 8. Periodic report: MCTS-eval the LEARNER (the net actively training,
        #    matching legacy's "eval after train") vs the plugged opponents
        #    (both colors), print a progress line, and flush metrics to logger.
        if rep > 0 and (it + 1) % rep == 0 and it >= learning_starts:
            learner.set_attr["training"](Scalar[DT](0.0))
            var names = List[String]()
            var values = List[Float64]()
            names.append(String("loss"))
            values.append(last_loss)
            names.append(String("games"))
            values.append(Float64(total_games))
            names.append(String("buffer_size"))
            values.append(Float64(len(replay)))
            names.append(String("promotions"))
            values.append(Float64(promotions))

            var line = String("  move ") + String(it + 1)
            line += " | games " + String(total_games)
            line += " | loss " + String(last_loss)
            line += " | replay " + String(len(replay))
            line += " | promo " + String(promotions)

            if do_eval:
                var e1 = _eval_both_colors[
                    ENV, NET, OPP1, EVAL_GAMES, NUM_SIMS, MAX_NODES, MAX_PLIES
                ](
                    ctx,
                    learner,
                    seed=seed + UInt64(it) * 13 + 5,
                    open_plies=eval_open_plies,
                )
                var tot1 = e1.wins + e1.draws + e1.losses
                var wr1 = Float64(e1.wins) / Float64(tot1) if tot1 > 0 else 0.0
                names.append(String("eval1_win"))
                values.append(Float64(e1.wins))
                names.append(String("eval1_draw"))
                values.append(Float64(e1.draws))
                names.append(String("eval1_loss"))
                values.append(Float64(e1.losses))
                names.append(String("eval1_winrate"))
                values.append(wr1)
                line += (
                    " | vs "
                    + OPP1.NAME
                    + " W"
                    + String(e1.wins)
                    + " D"
                    + String(e1.draws)
                    + " L"
                    + String(e1.losses)
                    + " (wr "
                    + String(Int(wr1 * 100.0))
                    + "%)"
                )

            if do_eval2:
                var e2 = _eval_both_colors[
                    ENV, NET, OPP2, EVAL_GAMES, NUM_SIMS, MAX_NODES, MAX_PLIES
                ](
                    ctx,
                    learner,
                    seed=seed + UInt64(it) * 17 + 9,
                    open_plies=eval_open_plies,
                )
                var tot2 = e2.wins + e2.draws + e2.losses
                var wr2 = Float64(e2.wins) / Float64(tot2) if tot2 > 0 else 0.0
                names.append(String("eval2_win"))
                values.append(Float64(e2.wins))
                names.append(String("eval2_draw"))
                values.append(Float64(e2.draws))
                names.append(String("eval2_loss"))
                values.append(Float64(e2.losses))
                names.append(String("eval2_winrate"))
                values.append(wr2)
                line += (
                    " | vs "
                    + OPP2.NAME
                    + " W"
                    + String(e2.wins)
                    + " D"
                    + String(e2.draws)
                    + " L"
                    + String(e2.losses)
                    + " (wr "
                    + String(Int(wr2 * 100.0))
                    + "%)"
                )

            if verbose:
                print(line)
            if logger:
                logger.value()[].log_scalars(names, values, it + 1)

    # Final flush: ensure the returned best is at least as new as the learner if
    # the learner ended clearly ahead (covers runs shorter than arena_every).
    var final_rec = candidate_winrate_mcts[
        ENV,
        NET,
        NET,
        ARENA_GAMES,
        NUM_SIMS,
        MAX_NODES,
        MAX_PLIES,
    ](ctx, learner, net, seed=seed + 9991, open_plies=arena_open_plies)
    if should_promote(
        final_rec, promote_threshold, min_decisive=ARENA_GAMES // 2
    ):
        hard_copy["gpu"](learner, net, octx)
        promotions += 1

    return ArenaRunResult(last_loss=last_loss, promotions=promotions)
