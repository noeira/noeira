"""Arena-gated AlphaZero self-play — CPU path (single-env, GenericCPUMCTS).

The CPU counterpart to `selfplay_arena.mojo`. Same full-AlphaZero structure —
frozen *best* generates self-play, a *learner* trains, periodic Arena gating
(`candidate_winrate_cpu`, both nets at CPU-MCTS strength) promotes the learner,
symmetry augmentation at record time — but everything runs on the CPU through
`GenericCPUMCTS` + the true-rules adapters and the nn loss graph on
`forward/vjp["cpu"]`. Two pluggable `CPUEvaluator` opponents + a `Logger` drive
the periodic eval/print/flush, mirroring the GPU driver's telemetry.

`net` is the *best* and holds the final weights on return.
"""

from noeira.nn.core.param import walk_params
from noeira.nn.core.ptr import untracked
from std.memory import Pointer

from noeira.nn.constants import DT
from noeira.nn.core.module import Module
from noeira.nn.core.tensor import Tensor
from noeira.nn.optimizer.adam import Adam
from noeira.nn.optimizer.grad_clip import clip_grad_norm
from noeira.nn.core.initializer import Zero
from noeira.nn.core.hard_copy import hard_copy
from noeira.nn.combinators.compute_graph import ComputeGraph
from noeira.nn.combinators.graph_decl import (
    InputSlot, Node, ExternalNode,
)
from noeira.core import TwoPlayerDiscreteEnv, Saveable
from noeira.core.logger import Logger, NoOpLogger
from noeira.planners.tree_search import (
    GenericCPUMCTS, AlphaGoPUCT, DirichletNoise, SelfPlay,
)

from .loss_ops import AZLossOp
from .arena import candidate_winrate_cpu, should_promote
from .eval import eval_mcts_vs_opponent_cpu, EvalResult
from .selfplay_arena import ArenaRunResult, append_az_train_diagnostics
from ..zero.mcts_adapters_cpu import AZRepCPU, AZDynCPU, AZPredCPU
from ..zero.example_replay import MCTSExampleReplay
from ..zero.symmetries import BoardAugmenter
from ..zero.evaluators import CPUEvaluator, RandomOpponent


@always_inline
def _xs(s: UInt64) -> UInt64:
    var x = s
    x ^= x << 13
    x ^= x >> 7
    x ^= x << 17
    return x


def _eval_both_colors_cpu[
    ENV: TwoPlayerDiscreteEnv & Saveable & Defaultable & Deinitable,
    NET: Module,
    OPP: CPUEvaluator,
    N_GAMES: Int,
    NUM_SIMS: Int,
    MAX_NODES: Int,
    MAX_PLIES: Int,
](mut net: NET, seed: UInt64) raises -> EvalResult:
    var p0 = eval_mcts_vs_opponent_cpu[
        ENV, NET, OPP, N_GAMES, NUM_SIMS, MAX_NODES, MAX_PLIES
    ](net, agent_player=0, seed=seed)
    var p1 = eval_mcts_vs_opponent_cpu[
        ENV, NET, OPP, N_GAMES, NUM_SIMS, MAX_NODES, MAX_PLIES
    ](net, agent_player=1, seed=seed + 33333)
    return EvalResult(
        wins=p0.wins + p1.wins,
        draws=p0.draws + p1.draws,
        losses=p0.losses + p1.losses,
    )


def run_alphazero_selfplay_arena_cpu[
    ENV: TwoPlayerDiscreteEnv & Saveable & Defaultable & Deinitable,
    NET: Module,
    AUG: BoardAugmenter,
    NUM_SIMS: Int,
    MAX_NODES: Int,
    BATCH: Int,
    CAP: Int,
    MAX_TRAJ: Int,
    ARENA_GAMES: Int = 32,
    MAX_PLIES: Int = 9,
    OPP1: CPUEvaluator = RandomOpponent,
    OPP2: CPUEvaluator = RandomOpponent,
    L: Logger = NoOpLogger,
    EVAL_GAMES: Int = 64,
    TEMP_MOVES: Int = 4,   # plies sampled ∝ visits before greedy (scale to game)
    BATCH_SIMS: Int = 1,   # MCTS leaves/round (virtual-loss batching); must divide NUM_SIMS
](
    mut net: NET,                 # the BEST net — holds final weights on return
    iterations: Int,
    learning_starts: Int = 0,
    train_per_iter: Int = 1,
    lr: Scalar[DT] = Scalar[DT](0.01),
    seed: UInt64 = 0,
    arena_every: Int = 100,
    arena_open_plies: Int = 2,
    promote_threshold: Float64 = 0.55,
    report_every: Int = 0,
    diag_every: Int = 0,          # cheap per-batch train diagnostics every N
    #                               moves; decoupled from periodic eval; 0 = off
    do_eval: Bool = True,
    do_eval2: Bool = False,
    verbose: Bool = True,
    logger: Optional[Pointer[L, MutAnyOrigin]] = None,
    max_grad_norm: Float64 = 0.0, # global grad-norm clip (0 = off ≡ plain Adam)
    weight_decay: Float64 = 0.0,  # decoupled (AdamW) weight decay (0 = off)
) raises -> ArenaRunResult:
    comptime OBS = NET.IN_DIMS[0]
    comptime ACT = NET.OUT_DIM - 1
    comptime W = NET.OUT_DIM
    comptime LATENT = ENV.SAVE_SIZE
    comptime NSYM = AUG.NUM_SYMMETRIES
    comptime MCTS = GenericCPUMCTS[
        ACT, LATENT, NUM_SIMS, MAX_NODES,
        AlphaGoPUCT[1.0], DirichletNoise[0.25, 0.25], SelfPlay,
        BATCH_SIMS=BATCH_SIMS,
        NORMALIZE_Q=False,  # raw Q∈[-1,1] like legacy (MinMax over-explores)
    ]
    comptime Graph = ComputeGraph[
        InputSlot["obs", OBS],
        ExternalNode["pred", NET, "obs"],
        InputSlot["tgt", W],
        Node["loss", AZLossOp[ACT], "pred", "tgt"],
    ]

    # Learner net (trains) initialised to the best net's weights.
    var learner = NET.make["cpu", Zero]()
    hard_copy["cpu"](net, learner)

    # Decoupled weight decay (AdamW) — `wd=0` ⇒ plain Adam. Grad-norm clip is
    # applied separately each step via `clip_grad_norm` (max <= 0 ⇒ no-op).
    var opt = Adam(lr=lr, wd=Scalar[DT](weight_decay))
    var graph = Graph.make["cpu", Zero]()
    var replay = MCTSExampleReplay[OBS, W, CAP]()

    var env = ENV()

    # ── Host trajectory storage + augmentation scratch — owned Lists. ──
    var traj_obs = List[Scalar[DT]](length=MAX_TRAJ * OBS, fill=0)
    var traj_pol = List[Scalar[DT]](length=MAX_TRAJ * ACT, fill=0)
    var aug_obs = List[Scalar[DT]](length=OBS, fill=0)
    var aug_pol = List[Scalar[DT]](length=ACT, fill=0)
    var tmp_tgt = List[Scalar[DT]](length=W, fill=0)
    var root_save = List[Scalar[DT]](length=LATENT, fill=0)
    var traj_len = 0

    # ── Train-batch storage Tensors (host; the nn surface) ──
    var obs_t = Tensor.alloc(BATCH * OBS)
    var tgt_t = Tensor.alloc(BATCH * W)
    var loss_t = Tensor.alloc(BATCH)
    var grad_t = Tensor.alloc(BATCH)
    for i in range(BATCH):
        grad_t.data[i] = Scalar[DT](1.0) / Scalar[DT](BATCH)

    _ = env.reset()
    var last_loss: Float64 = 0.0
    var promotions = 0
    var total_games = 0
    var rng = seed | 1

    var rep = report_every if report_every > 0 else arena_every
    if verbose:
        print(
            "AlphaZero self-play (CPU):", iterations, "moves,",
            NUM_SIMS, "sims/move | eval(MCTS)1=", OPP1.NAME,
            "eval2=", OPP2.NAME if do_eval2 else String("off"),
            "| report_every=", rep, "moves",
        )

    for it in range(iterations):
        # 1. MCTS search from the live env state, with the BEST net.
        net.set_attr["training"](Scalar[DT](0.0))
        env.save_env_state(root_save)
        var env_ptr = Pointer(to=env)
        var net_ptr = Pointer(to=net)
        var s_rep = AZRepCPU[ENV, OBS](env=untracked(env_ptr))
        var s_dyn = AZDynCPU[ENV, ACT](env=untracked(env_ptr))
        var s_pred = AZPredCPU[ENV, OBS, ACT, NET](
            env=untracked(env_ptr),
            net=untracked(net_ptr),
        )
        var mcts = MCTS(gamma=1.0)
        var legal = env.legal_action_mask()
        var root_obs = List[Float64](length=OBS, fill=Float64(0.0))
        var policy = mcts.search[
            AZRepCPU[ENV, OBS], AZDynCPU[ENV, ACT], AZPredCPU[ENV, OBS, ACT, NET]
        ](s_rep, s_dyn, s_pred, root_obs, add_noise=True, legal_mask=legal)
        env.load_env_state(root_save)

        # 2. Record (canonical obs, visit policy).
        if traj_len < MAX_TRAJ:
            var obs_raw = env.get_obs_list()
            var ob = traj_len * OBS
            for j in range(OBS):
                traj_obs[ob + j] = Scalar[DT](obs_raw[j])
            var pb = traj_len * ACT
            for a in range(ACT):
                traj_pol[pb + a] = Scalar[DT](policy[a])
            traj_len += 1

        # 3. Choose a move via the AlphaZero temperature schedule: τ=1 (sample
        #    ∝ visits) for the first `temp_moves` plies of each game for opening
        #    diversity, then τ→0 (greedy argmax) so the rest of self-play is
        #    high-quality. The legacy driver does exactly this (TEMP_THRESH=4);
        #    sampling EVERY ply leaves late-game play near-random, poisoning the
        #    value targets. `traj_len-1` is the current ply (0-based).
        var temp_moves = TEMP_MOVES
        var chosen = -1

        def _is_legal(a: Int) {imm} -> Bool:
            return a < len(legal) and Bool(legal[a])

        if traj_len - 1 < temp_moves:
            rng = _xs(rng)
            var u = Float64(rng % UInt64(1_000_000)) / 1_000_000.0
            var cum: Float64 = 0.0
            for a in range(ACT):
                if not _is_legal(a):
                    continue
                cum += policy[a]
                if u <= cum and policy[a] > 0.0:
                    chosen = a
                    break
        # Greedy / sampling-fallback: argmax over LEGAL actions only. The MCTS
        # uniform fallback (total visits 0) assigns equal prob to every action
        # incl. illegal ones, so an unmasked argmax would deterministically pick
        # an occupied cell → env.step no-ops → the game never terminates.
        if chosen < 0:
            var bestv = Float64(-1.0)
            for a in range(ACT):
                if _is_legal(a) and policy[a] > bestv:
                    bestv = policy[a]
                    chosen = a
            if chosen < 0:
                for a in range(ACT):
                    if _is_legal(a):
                        chosen = a
                        break
            if chosen < 0:
                chosen = 0
        var step_res = env.step(env.action_from_index(chosen))
        var done = step_res[2]

        # 4. On a finished game: assign z, symmetry-augment, flush, reset.
        if done:
            total_games += 1
            var gr = env.game_result()
            for k in range(traj_len):
                var z: Float64 = 0.0
                if gr == 1:
                    z = 1.0 if (k % 2 == 0) else -1.0
                elif gr == 2:
                    z = 1.0 if (k % 2 == 1) else -1.0
                var ob = k * OBS
                var pb = k * ACT
                for s in range(NSYM):
                    AUG.augment_obs[OBS](traj_obs, ob, s, aug_obs)
                    AUG.augment_policy[ACT](traj_pol, pb, s, aug_pol)
                    for a in range(ACT):
                        tmp_tgt[a] = aug_pol[a]
                    tmp_tgt[ACT] = Scalar[DT](z)
                    replay.record(aug_obs, 0, tmp_tgt, 0)
            traj_len = 0
            _ = env.reset()

        # 5. Train the LEARNER.
        if len(replay) >= BATCH and it >= learning_starts:
            learner.set_attr["training"](Scalar[DT](1.0))
            for _t in range(train_per_iter):
                replay.sample_batch_tensors[BATCH](obs_t, tgt_t)
                learner.zero_grad["cpu"](None)
                graph.set_input["obs", BATCH](obs_t, None)
                graph.set_input["tgt", BATCH](tgt_t, None)
                graph.forward[BATCH, "cpu"](loss_t, None, learner)
                graph.vjp[BATCH, "cpu"](grad_t, None, learner)
                _ = clip_grad_norm["cpu", NET](
                    learner, Scalar[DT](max_grad_norm), None
                )
                opt.begin_step()
                walk_params["cpu"](learner, opt, None)
            var ml: Float64 = 0.0
            for b in range(BATCH):
                ml += Float64(loss_t.data[b])
            last_loss = ml / Float64(BATCH)

            # 5b. Dense per-batch training diagnostics (legacy parity), on the
            #     train axis, decoupled from the expensive periodic eval. The
            #     CPU graph writes host buffers in place, so "pred" is directly
            #     readable; the matching targets are in `tb_tgt`.
            if (
                Bool(logger)
                and diag_every > 0
                and (it + 1) % diag_every == 0
            ):
                ref pred_node = graph.node_output["pred"]()
                var dnames = List[String]()
                var dvalues = List[Float64]()
                dnames.append(String("loss"))
                dvalues.append(last_loss)
                append_az_train_diagnostics[ACT, BATCH](
                    pred_node.data, tgt_t.data, dnames, dvalues
                )
                logger.value()[].log_scalars(dnames, dvalues, it + 1)

        # 6. Arena gating: challenge the best with the learner (CPU MCTS).
        if (
            it >= learning_starts
            and arena_every > 0
            and (it + 1) % arena_every == 0
            and len(replay) >= BATCH
        ):
            var rec = candidate_winrate_cpu[
                ENV, NET, NET, ARENA_GAMES, NUM_SIMS, MAX_NODES, MAX_PLIES
            ](learner, net, seed=seed + UInt64(it) * 7 + 1,
              open_plies=arena_open_plies)
            var accepted = should_promote(
                rec, promote_threshold, min_decisive=ARENA_GAMES // 2
            )
            if accepted:
                hard_copy["cpu"](learner, net)
                promotions += 1
            if verbose:
                print(
                    "  arena @ move", it + 1,
                    "| learner vs best (MCTS)  W", rec.wins, "D", rec.draws,
                    "L", rec.losses,
                    "→ ACCEPTED" if accepted else "→ rejected",
                    "(promotions", promotions, ")",
                )

        # 7. Periodic report: MCTS-eval the LEARNER vs the plugged opponents.
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
            line += " | promo " + String(promotions)

            if do_eval:
                var e1 = _eval_both_colors_cpu[
                    ENV, NET, OPP1, EVAL_GAMES, NUM_SIMS, MAX_NODES, MAX_PLIES
                ](learner, seed=seed + UInt64(it) * 13 + 5)
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
                    " | vs " + OPP1.NAME + " W" + String(e1.wins)
                    + " D" + String(e1.draws) + " L" + String(e1.losses)
                    + " (wr " + String(Int(wr1 * 100.0)) + "%)"
                )

            if do_eval2:
                var e2 = _eval_both_colors_cpu[
                    ENV, NET, OPP2, EVAL_GAMES, NUM_SIMS, MAX_NODES, MAX_PLIES
                ](learner, seed=seed + UInt64(it) * 17 + 9)
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
                    " | vs " + OPP2.NAME + " W" + String(e2.wins)
                    + " D" + String(e2.draws) + " L" + String(e2.losses)
                    + " (wr " + String(Int(wr2 * 100.0)) + "%)"
                )

            if verbose:
                print(line)
            if logger:
                logger.value()[].log_scalars(names, values, it + 1)

    # Final flush: promote the learner if it ends clearly ahead.
    var final_rec = candidate_winrate_cpu[
        ENV, NET, NET, ARENA_GAMES, NUM_SIMS, MAX_NODES, MAX_PLIES
    ](learner, net, seed=seed + 9991, open_plies=arena_open_plies)
    if should_promote(final_rec, promote_threshold, min_decisive=ARENA_GAMES // 2):
        hard_copy["cpu"](learner, net)
        promotions += 1

    return ArenaRunResult(last_loss=last_loss, promotions=promotions)
