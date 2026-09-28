"""Arena-gated GUMBEL AlphaZero self-play with symmetry augmentation.

The Gumbel-planner sibling of `selfplay_arena.mojo`: SELF-PLAY runs
`GumbelGPUMCTS.search_gpu_alphazero` (Gumbel-Top-k roots + Sequential Halving,
improved-policy targets, serial sims) while the ARENA GATING and the periodic
MCTS-strength EVALS keep the exact PUCT machinery of the original — the same
yardstick measures both drivers, so runs are directly comparable. Original
header follows.

Arena-gated AlphaZero self-play with symmetry augmentation — full AlphaZero.

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
    GumbelGPUMCTS,
    AlphaGoPUCT,
    DirichletNoise,
    SelfPlay,
)

from .selfplay_arena import ArenaRunResult
from .loss_ops import AZLossOp
from .arena import candidate_winrate_mcts, should_promote
from .eval import eval_mcts_vs_opponent, EvalResult
from ..zero.mcts_adapters import AZPredGPU, AZEnvGPU
from ..zero.example_replay import MCTSExampleReplay
from ..zero.symmetries import BoardAugmenter
from ..zero.evaluators import GPUEvaluator, RandomOpponent


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
    var ce_sum: Float64 = 0.0
    var ent_sum: Float64 = 0.0
    var vmse_sum: Float64 = 0.0
    var vmean_sum: Float64 = 0.0
    var tent_sum: Float64 = 0.0
    var tmax_sum: Float64 = 0.0
    var vt_sum: Float64 = 0.0
    var vt_pos = 0
    # Mirror the loss op's logit clamp (±30): a float32 ResNet torso can emit
    # ±inf logits in train mode, which the loss clamps before softmax. Clamping
    # here too keeps the diagnostic curves finite AND faithful to what training
    # actually optimizes (rather than skipping the row or logging zeros).
    var n_fin = BATCH
    for b in range(BATCH):
        var base = b * W

        # Policy: softmax(clamped logits) → CE vs target π, plus pred entropy.
        var maxl = max(min(Float64(pred[base]), 30.0), -30.0)
        for a in range(1, ACT):
            var v = max(min(Float64(pred[base + a]), 30.0), -30.0)
            if v > maxl:
                maxl = v
        var sume: Float64 = 0.0
        for a in range(ACT):
            var la = max(min(Float64(pred[base + a]), 30.0), -30.0)
            sume += exp(la - maxl)
        var ce: Float64 = 0.0
        var ent: Float64 = 0.0
        for a in range(ACT):
            var la = max(min(Float64(pred[base + a]), 30.0), -30.0)
            var prob = exp(la - maxl) / sume
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
        var tv = tanh(max(min(Float64(pred[base + ACT]), 30.0), -30.0))
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

    # All means are over finite rows only (n_fin); if the whole batch was
    # non-finite, fall back to 1 to avoid a divide-by-zero (the sums are 0 →
    # zeros logged, not NaNs).
    var n = Float64(n_fin) if n_fin > 0 else 1.0
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


def run_alphazero_selfplay_arena_gumbel[
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
    MAX_K: Int = 4,  # Gumbel root candidates
    #                                        (power of two, <= ACT).
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
    selfplay_open_plies: Int = 2,  # uniform-random LEGAL action for the first
    #                               N plies of every self-play game (the search
    #                               policy is still recorded as the target).
    #                               Keeps replay diverse when the head goes
    #                               near-one-hot post-promotion — Gumbel logit
    #                               noise (std ~1.3) can't perturb saturated
    #                               logits, and sampling a one-hot improved
    #                               policy is deterministic. C4: 2 plies = 49
    #                               openings vs 1.
    eval_open_plies: Int = 0,  # uniform-random legal plies opening each
    #                               periodic-eval game (both sides). 0 = the
    #                               canonical single-line "perfect-play gate"
    #                               (vs a deterministic opponent the result is
    #                               quantized to 0/EVAL_GAMES per color); ≥2 =
    #                               diversified openings → a real winrate curve.
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
    comptime MCTS = GumbelGPUMCTS[
        N_ENVS,
        ACT,
        OBS,
        1,
        MAX_NODES,
        MAX_K,
        NUM_SIMS,
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

    # Decoupled weight decay (AdamW) — `wd=0` ⇒ plain Adam. The global grad-norm
    # clip is applied separately each step via `clip_grad_norm` (max <= 0 ⇒
    # no-op). Per-param decay flags come from the net's Param decay bits.
    var opt = Adam(lr=lr, wd=Scalar[DT](weight_decay))
    var graph = Graph.make["gpu", Zero](octx)
    var mcts = MCTS(ctx, gamma=1.0, v_min=-1.0, v_max=1.0)
    var replay = MCTSExampleReplay[OBS, W, CAP]()

    # ── Device buffers ──
    var states = ctx.enqueue_create_buffer[DT](N_ENVS * STATE)
    var obs_dev = ctx.enqueue_create_buffer[DT](N_ENVS * OBS)
    var legal_dev = ctx.enqueue_create_buffer[DT](N_ENVS * ACT)
    var rew = ctx.enqueue_create_buffer[DT](N_ENVS)
    var done = ctx.enqueue_create_buffer[DT](N_ENVS)
    var term = ctx.enqueue_create_buffer[DT](N_ENVS)
    var obs_next = ctx.enqueue_create_buffer[DT](N_ENVS * OBS)
    var legal_next = ctx.enqueue_create_buffer[DT](N_ENVS * ACT)
    var ep_steps_dev = ctx.enqueue_create_buffer[DT](
        N_ENVS
    )  # per-env ply count
    var act_dev = ctx.enqueue_create_buffer[DT](N_ENVS)  # sampled actions

    # ── Host staging (MCTS root obs/policy, step results, action selection) ──
    var obs_h = ctx.enqueue_create_host_buffer[DT](N_ENVS * OBS)
    var pol_h = ctx.enqueue_create_host_buffer[DT](N_ENVS * ACT)
    var done_h = ctx.enqueue_create_host_buffer[DT](N_ENVS)
    var rew_h = ctx.enqueue_create_host_buffer[DT](N_ENVS)
    var ep_steps_h = ctx.enqueue_create_host_buffer[DT](N_ENVS)  # per-env ply
    var act_h = ctx.enqueue_create_host_buffer[DT](N_ENVS)  # sampled actions
    var legal_h = ctx.enqueue_create_host_buffer[DT](N_ENVS * ACT)
    var states_h = ctx.enqueue_create_host_buffer[DT](N_ENVS * STATE)  # diag
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

    var az_rng = seed ^ UInt64(0x9E3779B97F4A7C15)
    var last_loss: Float64 = 0.0
    var nan_localized = False  # one-time NaN-source report guard
    var promotions = 0
    var total_games = 0  # cumulative finished self-play games
    # ── Self-play health diagnostics (reset each report) ──
    var games_prev = 0  # total_games at the previous report
    var period_guard_hits = 0  # self-play actions where the legality guard
    #                              had to override an illegal/degenerate pick
    var period_nonfinite_pol = 0  # (env×move) count with a non-finite search
    #                              policy row (eval-mode net emitting NaN/inf)
    var period_skipped_tgt = 0  # finished-game plies DROPPED from replay
    #                              because the recorded search policy had a
    #                              non-finite entry — one NaN target row is
    #                              enough to permanently NaN a policy-head
    #                              weight column via the soft-CE gradient
    #                              (gp[c] = up·(softmax[c] − tgt[c]))

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
            legal_dev,
            k_actual=MAX_K,
            rng_seed=UInt32((seed + UInt64(it)) & 0xFFFFFFFF),
        )

        # 2. Pull root obs + visit-count policy, record into trajectory.
        ctx.enqueue_copy(obs_h, obs_dev)
        ctx.enqueue_copy(pol_h, mcts.policies_view())
        ctx.enqueue_copy(legal_h, legal_dev)
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

        # 2b. Action selection from the IMPROVED policy: sample for the
        #     first TEMP_MOVES plies of each game (opening diversity — Gumbel
        #     root sampling already explores, so this mirrors the PUCT
        #     driver's schedule rather than adding one), argmax after. The
        #     improved policy is legal-masked by the planner.
        for e in range(N_ENVS):
            var ply = Int(Float64(ep_steps_h[e]))
            # DIAG: flag a non-finite search-policy row (eval-mode net NaN/inf).
            for a in range(ACT):
                var pv = Float64(pol_h[e * ACT + a])
                if pv - pv != 0.0:
                    period_nonfinite_pol += 1
                    break
            var a_sel = -1
            if ply < selfplay_open_plies:
                # Opening diversity: uniform over LEGAL actions.
                var n_legal = 0
                for a in range(ACT):
                    if Float64(legal_h[e * ACT + a]) > 0.5:
                        n_legal += 1
                if n_legal > 0:
                    az_rng = az_rng ^ (az_rng << 13)
                    az_rng = az_rng ^ (az_rng >> 7)
                    az_rng = az_rng ^ (az_rng << 17)
                    var pick = Int(az_rng % UInt64(n_legal))
                    var seen = 0
                    for a in range(ACT):
                        if Float64(legal_h[e * ACT + a]) > 0.5:
                            if seen == pick:
                                a_sel = a
                                break
                            seen += 1
            elif ply < TEMP_MOVES:
                az_rng = az_rng ^ (az_rng << 13)
                az_rng = az_rng ^ (az_rng >> 7)
                az_rng = az_rng ^ (az_rng << 17)
                var r = Float64(az_rng % UInt64(1_000_000)) / 1_000_000.0
                var cum = 0.0
                for a in range(ACT):
                    var pv = Float64(pol_h[e * ACT + a])
                    cum += pv
                    if r <= cum and pv > 0.0:
                        a_sel = a
                        break
            if a_sel < 0:  # greedy ply or numeric fallback
                var bv = -1.0
                for a in range(ACT):
                    var pv = Float64(pol_h[e * ACT + a])
                    if pv > bv:
                        bv = pv
                        a_sel = a
            # Final legality guard. The improved policy can place its argmax on
            # an ILLEGAL column when the eval-mode net emits −inf logits for
            # every LEGAL move at this position: illegal moves carry a −1e9
            # sentinel in gz_extract, and −1e9 > −inf, so the illegal column
            # wins the softmax. Playing an illegal column is a NO-OP in the C4
            # step kernel (reward −1, done 0, state + current-player unchanged),
            # so the next iteration re-searches the identical position, picks the
            # same illegal column, and the game HANGS — self-play freezes and
            # training starves of new data (the post-promotion plateau). Force a
            # legal action: keep a_sel if legal, else the highest-policy LEGAL
            # column (first legal if the policy is degenerate). The recorded
            # target is left untouched — only the action PLAYED is corrected.
            if (
                a_sel < 0
                or Float64(legal_h[e * ACT + a_sel]) <= 0.5
            ):
                period_guard_hits += 1  # DIAG: policy wanted an illegal move
                var bestl = -1
                var bvl = -1.0e30
                for a in range(ACT):
                    if Float64(legal_h[e * ACT + a]) > 0.5:
                        var pv = Float64(pol_h[e * ACT + a])
                        if bestl < 0 or pv > bvl:
                            bvl = pv
                            bestl = a
                if bestl >= 0:
                    a_sel = bestl
            act_h[e] = Scalar[DT](a_sel)
        ctx.enqueue_copy(act_dev, act_h)

        # 3. Step every game by its chosen action.
        ENV.step_kernel_gpu[N_ENVS, STATE, OBS](
            ctx,
            states,
            act_dev,
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
                    # Record guard: DROP plies whose search policy carries a
                    # non-finite entry. Recording them poisons training — a
                    # single NaN target column makes that column's soft-CE
                    # gradient NaN, the clip can't catch a NaN norm cheaply
                    # (belt: it now zeroes instead), and Adam's moments would
                    # otherwise NaN that policy-head weight row PERMANENTLY
                    # (the post-promotion collapse). Skip, never substitute:
                    # replacing with uniform feeds fake easy targets (the
                    # reverted sanitizer regression).
                    var pol_ok = True
                    for a in range(ACT):
                        var pv = Float64(traj_pol[pb + a])
                        if pv - pv != 0.0:
                            pol_ok = False
                            break
                    if not pol_ok:
                        period_skipped_tgt += 1
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
                _ = clip_grad_norm["gpu", NET](
                    learner, Scalar[DT](max_grad_norm), octx
                )
                opt.begin_step()
                walk_params["gpu"](learner, opt, octx)
            loss_t.download(ctx)
            # Average over FINITE rows only: a single non-finite row otherwise
            # poisons the whole reported mean (and the diagnostic curves the
            # logger then drops), reading as a dashboard "collapse" even when
            # training is healthy. `x - x == 0` is True iff x is finite (False
            # for NaN and ±inf).
            var ml: Float64 = 0.0
            var n_fin = 0
            var n_bad = 0
            for b in range(BATCH):
                var lv = Float64(loss_t.data[b])
                if lv - lv == 0.0:
                    ml += lv
                    n_fin += 1
                else:
                    n_bad += 1
            if n_fin > 0:
                last_loss = ml / Float64(n_fin)  # else keep prior finite value

            # One-time NaN localizer. A non-finite train loss is structurally
            # impossible from finite inputs (targets are softmax-normalized, obs
            # is one-hot, the loss uses a numerically-stable log-sum-exp), so the
            # first occurrence pulls `pred` once and reports which buffer holds
            # the offender — obs / target / policy-logit / value — plus the
            # finite-logit magnitude range. Pins the source instead of guessing.
            if n_bad > 0 and not nan_localized:
                nan_localized = True
                ref pred_node = graph.node_output["pred"]()
                pred_node.download(ctx)
                var obs_bad = 0
                var tgt_bad = 0
                var logit_bad = 0
                var val_bad = 0
                var lmin: Float64 = 1e30
                var lmax: Float64 = -1e30
                for b in range(BATCH):
                    for j in range(OBS):
                        var ov = Float64(obs_t.data[b * OBS + j])
                        if ov - ov != 0.0:
                            obs_bad += 1
                    for j in range(W):
                        var tvv = Float64(tgt_t.data[b * W + j])
                        if tvv - tvv != 0.0:
                            tgt_bad += 1
                    for a in range(ACT):
                        var pl = Float64(pred_node.data[b * W + a])
                        if pl - pl != 0.0:
                            logit_bad += 1
                        else:
                            if pl < lmin:
                                lmin = pl
                            if pl > lmax:
                                lmax = pl
                    var pv = Float64(pred_node.data[b * W + ACT])
                    if pv - pv != 0.0:
                        val_bad += 1
                print(
                    "  [nan-localizer] move",
                    it + 1,
                    "| bad_loss_rows",
                    n_bad,
                    "of",
                    BATCH,
                    "| obs_bad",
                    obs_bad,
                    "| tgt_bad",
                    tgt_bad,
                    "| logit_bad",
                    logit_bad,
                    "| val_bad",
                    val_bad,
                    "| finite_logit_range [",
                    lmin,
                    ",",
                    lmax,
                    "]",
                )

            # 6b. Dense per-batch training diagnostics (legacy parity), on the
            #     train axis and decoupled from the expensive periodic eval. The
            #     graph's "pred" node still holds the last train batch's net
            #     output; the matching targets are in `tb_tgt_h`. One small D2H.
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

            # ── Self-play health diagnostics ──
            # games_delta: finished self-play games since the last report. If
            #   this hits 0 while the loop keeps running, self-play has STALLED
            #   (no new data → plateau) — the decisive signal.
            # guard_hits: how many played actions the legality guard had to
            #   correct (policy wanted an illegal move). High → the −inf-logit
            #   illegal-argmax pathology is firing.
            # nonfinite_pol: (env×move) with a non-finite search policy
            #   (eval-mode net emitting NaN/inf).
            # state_nan: NaN cells in the persistent self-play state buffer
            #   (corruption that would break done-detection / stepping).
            var games_delta = total_games - games_prev
            ctx.enqueue_copy(states_h, states)
            ctx.synchronize()
            var state_nan = 0
            for s in range(N_ENVS * STATE):
                var sv = Float64(states_h[s])
                if sv - sv != 0.0:
                    state_nan += 1
            names.append(String("games_delta"))
            values.append(Float64(games_delta))
            names.append(String("selfplay_guard_hits"))
            values.append(Float64(period_guard_hits))
            names.append(String("selfplay_nonfinite_pol"))
            values.append(Float64(period_nonfinite_pol))
            names.append(String("selfplay_skipped_tgt"))
            values.append(Float64(period_skipped_tgt))
            names.append(String("selfplay_state_nan"))
            values.append(Float64(state_nan))

            var line = String("  move ") + String(it + 1)
            line += " | games " + String(total_games)
            line += " (+" + String(games_delta) + ")"
            line += " | loss " + String(last_loss)
            line += " | replay " + String(len(replay))
            line += " | promo " + String(promotions)
            line += " | sp[guard " + String(period_guard_hits)
            line += " nanpol " + String(period_nonfinite_pol)
            line += " skiptgt " + String(period_skipped_tgt)
            line += " stnan " + String(state_nan) + "]"

            # Reset the per-period self-play counters.
            games_prev = total_games
            period_guard_hits = 0
            period_nonfinite_pol = 0
            period_skipped_tgt = 0

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
