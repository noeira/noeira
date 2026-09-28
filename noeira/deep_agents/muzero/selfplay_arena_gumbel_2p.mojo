"""Arena-gated two-player GUMBEL MuZero self-play (GPU) — the board-game driver.

The MuZero counterpart of `alphazero/selfplay_arena_gumbel.mojo`. Where the
AlphaZero arena searches the TRUE game rules with a single net, MuZero plans in
latent space over a *learned model* (three nets h/g/f = representation /
dynamics / prediction) and trains them with a K-step BPTT unroll. This driver
keeps the AlphaZero arena's production structure — best/learner split, Arena
gating, symmetry augmentation, two pluggable eval opponents — but swaps every
algorithmic core to MuZero:

  * **Search.** `GumbelGPUMCTS[..., SelfPlay]` over the on-device h/g/f adapters
    (`MZRepGPU`/`MZDynGPU`/`MZPredGPU`), with the env's legal mask applied at the
    root (`apply_legal=True`) and the two-player (`SelfPlay`) backup that negates
    value each ply. The improved policy is the stored target.

  * **Targets.** Per-step the driver records `(obs, action, π, root_value,
    to_play, legal)` into a device-obs `PrioritizedMCTSSequenceReplay` (the obs
    ring lives on the GPU; PER is toggled by `use_per`). The n-step value targets carry
    the two-player sign flips (`zero/nstep_targets.mojo`). `to_play` is the ply
    parity (ConnectFour's canonical obs is always the mover's frame, P0 first).

  * **Best/learner gating.** A frozen *best* h/g/f trio generates all self-play;
    a *learner* trio trains. Every `arena_every` moves the learner plays the best
    at full Gumbel strength from both colors (`mz_candidate_winrate`); on a clear
    majority of decisive games it is promoted (params-only `hard_copy_params` ×3
    — the MLP h/g/f carry no BatchNorm, so there are no running stats to copy,
    sidestepping the C4-gumbel promotion pitfall seen with BN ResNets).

  * **Augmentation.** Each finished game is replicated under the board's symmetry
    group (`AUG`): obs / policy / legal are permuted and the stored *action*
    index is re-derived through the same permutation (one-hot → augment → argmax),
    so the whole `(obs, a, π)` sequence stays a valid trajectory.

Telemetry mirrors the AlphaZero arena: a periodic full-strength MuZero-MCTS eval
vs `OPP1`/`OPP2` (both colors), per-batch loss split to the logger, and a
progress line. `report_every` / `arena_every` are in self-play *moves* (one loop
pass advances all `N_ENVS` games one move).

Single-file, GPU-only (the Gumbel planner is device-side). Exposed as a free
function (like `run_muzero_selfplay_2p_cpu`) rather than on the single-player
`MuZeroAgent` facade — the example wires the three nets + optimizers directly.
"""

from std.math import exp, log
from std.memory import alloc
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor

from noeira.nn.constants import DT
from noeira.nn.core.module import Module
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.optimizer.adam import Adam
from noeira.nn.optimizer.lr_scheduler import Scheduler, ConstantSchedule
from noeira.nn.core.hard_copy import hard_copy
from noeira.nn.core.checkpoint import save_params_multi
from noeira.nn.core.initializer import Kaiming
from noeira.core.env_traits import GPUTwoPlayerDiscreteEnv
from noeira.core.logger import Logger, NoOpLogger
from noeira.planners.tree_search import (
    GumbelGPUMCTS, SelfPlay,
    RepresentationGPU, DynamicsGPU, PredictionGPU,
)

from noeira.cuda import CUDAGraph, maybe_capture_replay
from .blocks import (
    mz_unroll_train_step_gpu,
    _mz_train_prologue_gpu,
    _mz_train_fwdrev_gpu,
    _mz_arena_opt_step_gpu,
    _mz_train_epilogue_gpu,
    MZScratch,
)
from .selfplay_gpu_device import _mz_emit_train_diag
from ..zero.mcts_adapters_mz import MZRepGPU, MZDynGPU, MZPredGPU
from ..zero.prioritized_sequence_replay_mcts import (
    PrioritizedMCTSSequenceReplay,
)
from ..zero.symmetries import BoardAugmenter, IdentityAugmenter
from ..zero.evaluators import GPUEvaluator, RandomOpponent
from ..zero.temperature import visit_temperature


def _a(n: Int) -> Pointer[Scalar[DT], MutAnyOrigin]:
    """Raw scratch for optional unroll outputs + diag/staging host buffers
    (the unroll's optional-output params are Optional[Pointer])."""
    return alloc[Scalar[DT]]({count = n}).unsafe_leak().as_unsafe_any_origin()


def _ai32(n: Int) -> Pointer[Int32, MutAnyOrigin]:
    return alloc[Int32]({count = n}).unsafe_leak().as_unsafe_any_origin()


def _aint(n: Int) -> Pointer[Int, MutAnyOrigin]:
    return alloc[Int]({count = n}).unsafe_leak().as_unsafe_any_origin()


@always_inline
def _xs(s: UInt64) -> UInt64:
    var x = s
    x = x ^ (x << 13)
    x = x ^ (x >> 7)
    x = x ^ (x << 17)
    return x


@fieldwise_init
struct MZArenaResult(Copyable, Movable):
    """Aggregate match record from the candidate's perspective."""

    var wins: Int
    var draws: Int
    var losses: Int


@fieldwise_init
struct MZArenaRunResult(Copyable, Movable):
    var last_loss: Float64
    var promotions: Int


def _mz_save_trio[
    REP: Module, DYN: Module, PRED: Module,
](
    ctx: DeviceContext,
    mut rep: REP, mut dyn: DYN, mut pred: PRED,
    path: String,
) raises:
    """Write the rep/dyn/pred trio (weights only; optimizers are session-local)
    into a SINGLE checkpoint file via `save_params_multi` (rep→dyn→pred sections
    under one v2 header). The play script's loader (`load_params_multi`) walks the
    trio in the same order."""
    save_params_multi["gpu", REP, DYN, PRED](
        path, Optional(ctx), False, rep, dyn, pred
    )


def _mz_should_promote(
    rec: MZArenaResult, threshold: Float64 = 0.55, min_decisive: Int = 1
) -> Bool:
    """Standard AlphaZero accept rule: among decisive games (draws ignored) the
    candidate must win at least `threshold` of them, and there must be at least
    `min_decisive` decisive games (else a 0-0 tie spuriously promotes)."""
    var decisive = rec.wins + rec.losses
    if decisive < min_decisive:
        return False
    return Float64(rec.wins) >= threshold * Float64(decisive)


# ══════════════════════════════════════════════════════════════════════════
# Batched MuZero-Gumbel search → greedy (argmax-over-legal) actions.
# Shared by the arena (two net trios) and the eval-vs-opponent helper.
# ══════════════════════════════════════════════════════════════════════════


def _mz_search_argmax[
    N: Int,
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
    mut planner: GumbelGPUMCTS[
        N, ACT, LATENT, BINS, MAX_NODES, MAX_K, NUM_SIMS, SelfPlay
    ],
    mut rep_a: RA,
    mut dyn_a: DA,
    mut pred_a: PA,
    obs_dev: DeviceBuffer[DT],
    legal_dev: DeviceBuffer[DT],
    mut h_pol: Pointer[Scalar[DT], MutAnyOrigin],
    mut legal_h: Pointer[Scalar[DT], MutAnyOrigin],
    mut actions_h: Pointer[Scalar[DT], MutAnyOrigin],
    rng_seed: UInt32,
) raises:
    """Run one batched Gumbel search over the learned model on `N` root obs and
    write the greedy (argmax of the improved policy over legal moves) action per
    env into `actions_h`. The planner's gumbel_scale governs determinism (set 0
    at the call site for eval/arena)."""
    ctx.enqueue_copy(planner.legal_mask_view(), legal_dev)
    # `obs_dev` is an immutable param; the planner ABI wants `MutAnyOrigin`.
    # Read-only root observations — cast restores the pre-nightly typing.
    var obs_t = LayoutTensor[DT, Layout.row_major(N, RA.OBS_DIM), MutAnyOrigin](
        obs_dev.unsafe_ptr().as_unsafe_any_origin().unsafe_mut_cast[True]()
    )
    planner.search_gpu[RA, DA, PA](
        ctx, rep_a, dyn_a, pred_a, obs_t,
        apply_legal=True, k_actual=MAX_K, rng_seed=rng_seed,
    )
    ctx.enqueue_copy(h_pol, planner.policies_view())
    ctx.enqueue_copy(legal_h, legal_dev)
    ctx.synchronize()
    for e in range(N):
        var best = -1
        var bestv = Float64(-1e30)
        for a in range(ACT):
            if Float64(legal_h[unsafe_offset=e * ACT + a]) > 0.5:
                var v = Float64(h_pol[unsafe_offset=e * ACT + a])
                if best < 0 or v > bestv:
                    bestv = v
                    best = a
        actions_h[unsafe_offset=e] = Scalar[DT](best if best >= 0 else 0)


# ══════════════════════════════════════════════════════════════════════════
# Arena: learner net trio vs best net trio, both at full Gumbel strength.
# ══════════════════════════════════════════════════════════════════════════


def mz_arena_match[
    ENV: GPUTwoPlayerDiscreteEnv,
    REP: Module,
    DYN: Module,
    PRED: Module,
    N_GAMES: Int,
    OBS: Int,
    ACT: Int,
    LATENT: Int,
    BINS: Int,
    NUM_SIMS: Int,
    MAX_NODES: Int,
    MAX_K: Int,
    MAX_PLIES: Int,
](
    ctx: DeviceContext,
    mut a_rep: REP, mut a_dyn: DYN, mut a_pred: PRED,
    mut b_rep: REP, mut b_dyn: DYN, mut b_pred: PRED,
    gamma: Float64,
    a_player: Int = 0,
    seed: UInt64 = 1,
    open_plies: Int = 2,
) raises -> MZArenaResult:
    """Net trio A vs net trio B, both planning over their own learned model at
    full Gumbel strength (deterministic, gumbel_scale=0). A plays color
    `a_player`. Per-env done tracking + selective reset (early finishers stop
    interfering); attribution by reward + whose turn it was. Returns A's record.
    `open_plies` random opening plies (both sides) diversify the lockstep batch.
    """
    comptime STATE = ENV.STATE_SIZE

    a_rep.set_attr["training"](Scalar[DT](0.0))
    a_dyn.set_attr["training"](Scalar[DT](0.0))
    a_pred.set_attr["training"](Scalar[DT](0.0))
    b_rep.set_attr["training"](Scalar[DT](0.0))
    b_dyn.set_attr["training"](Scalar[DT](0.0))
    b_pred.set_attr["training"](Scalar[DT](0.0))

    var planner = GumbelGPUMCTS[
        N_GAMES, ACT, LATENT, BINS, MAX_NODES, MAX_K, NUM_SIMS, SelfPlay
    ](ctx, gamma=gamma, v_min=-1.0, v_max=1.0, gumbel_scale=0.0)
    var ar = MZRepGPU[OBS, LATENT, REP].make(a_rep)
    var ad = MZDynGPU[LATENT, ACT, BINS, DYN].make(a_dyn)
    var ap = MZPredGPU[LATENT, ACT, BINS, PRED].make(a_pred)
    var br = MZRepGPU[OBS, LATENT, REP].make(b_rep)
    var bd = MZDynGPU[LATENT, ACT, BINS, DYN].make(b_dyn)
    var bp = MZPredGPU[LATENT, ACT, BINS, PRED].make(b_pred)

    var states = ctx.enqueue_create_buffer[DT](N_GAMES * STATE)
    var obs_dev = ctx.enqueue_create_buffer[DT](N_GAMES * OBS)
    var legal_dev = ctx.enqueue_create_buffer[DT](N_GAMES * ACT)
    var rew = ctx.enqueue_create_buffer[DT](N_GAMES)
    var done = ctx.enqueue_create_buffer[DT](N_GAMES)
    var term = ctx.enqueue_create_buffer[DT](N_GAMES)
    var obs_next = ctx.enqueue_create_buffer[DT](N_GAMES * OBS)
    var legal_next = ctx.enqueue_create_buffer[DT](N_GAMES * ACT)
    var act_dev = ctx.enqueue_create_buffer[DT](N_GAMES)

    var h_pol = _a(N_GAMES * ACT)
    var legal_h = _a(N_GAMES * ACT)
    var act_h = _a(N_GAMES)
    var done_h = _a(N_GAMES)
    var rew_h = _a(N_GAMES)
    ctx.synchronize()

    ENV.reset_kernel_gpu[N_GAMES, STATE](ctx, states)
    ENV.extract_obs_kernel_gpu[N_GAMES, STATE, OBS](
        ctx, states, obs_dev, legal_dev
    )
    ctx.synchronize()

    var eval_done = Array[Bool, N_GAMES](fill=False)
    var eval_result = Array[Int, N_GAMES](fill=0)  # 1=A win 2=A loss 3=draw
    var all_done = False
    var move = 0
    var rng = seed | 1

    comptime MAX_MOVES = MAX_PLIES + ACT
    while not all_done and move < MAX_MOVES:
        var a_turn = (move % 2) == a_player
        if move < open_plies:
            rng = _xs(rng)
            RandomOpponent.select_action_gpu[N_GAMES, ACT, STATE](
                ctx, act_dev, legal_dev, states, rng
            )
        elif a_turn:
            _mz_search_argmax[
                N_GAMES, OBS, ACT, LATENT, BINS,
                MAX_NODES, MAX_K, NUM_SIMS,
            ](
                ctx, planner, ar, ad, ap, obs_dev, legal_dev,
                h_pol, legal_h, act_h, rng_seed=UInt32(move + 1),
            )
            ctx.enqueue_copy(act_dev, act_h)
        else:
            _mz_search_argmax[
                N_GAMES, OBS, ACT, LATENT, BINS,
                MAX_NODES, MAX_K, NUM_SIMS,
            ](
                ctx, planner, br, bd, bp, obs_dev, legal_dev,
                h_pol, legal_h, act_h, rng_seed=UInt32(move * 3 + 7),
            )
            ctx.enqueue_copy(act_dev, act_h)
        ctx.synchronize()

        ENV.step_kernel_gpu[N_GAMES, STATE, OBS](
            ctx, states, act_dev, rew, done, term, obs_next, legal_next
        )
        ctx.enqueue_copy(done_h, done)
        ctx.enqueue_copy(rew_h, rew)
        ctx.synchronize()

        all_done = True
        for e in range(N_GAMES):
            if not eval_done[e] and Float64(done_h[unsafe_offset=e]) > 0.5:
                eval_done[e] = True
                var r = Float64(rew_h[unsafe_offset=e])
                if r > 0.5:
                    eval_result[e] = 1 if a_turn else 2
                elif r < -0.5:
                    eval_result[e] = 2 if a_turn else 1
                else:
                    eval_result[e] = 3
            if not eval_done[e]:
                all_done = False

        ENV.selective_reset_kernel_gpu[N_GAMES, STATE](
            ctx, states, done, rng_seed=seed + UInt64(move) * 7 + 3
        )
        ENV.extract_obs_kernel_gpu[N_GAMES, STATE, OBS](
            ctx, states, obs_dev, legal_dev
        )
        ctx.synchronize()
        move += 1

    var wins = 0
    var draws = 0
    var losses = 0
    for e in range(N_GAMES):
        if eval_result[e] == 1:
            wins += 1
        elif eval_result[e] == 2:
            losses += 1
        else:
            draws += 1

    h_pol.unsafe_free(); legal_h.unsafe_free(); act_h.unsafe_free(); done_h.unsafe_free(); rew_h.unsafe_free()
    return MZArenaResult(wins=wins, draws=draws, losses=losses)


def mz_candidate_winrate[
    ENV: GPUTwoPlayerDiscreteEnv,
    REP: Module,
    DYN: Module,
    PRED: Module,
    N_PER_COLOR: Int,
    OBS: Int,
    ACT: Int,
    LATENT: Int,
    BINS: Int,
    NUM_SIMS: Int,
    MAX_NODES: Int,
    MAX_K: Int,
    MAX_PLIES: Int,
](
    ctx: DeviceContext,
    mut c_rep: REP, mut c_dyn: DYN, mut c_pred: PRED,
    mut b_rep: REP, mut b_dyn: DYN, mut b_pred: PRED,
    gamma: Float64,
    seed: UInt64 = 1,
    open_plies: Int = 2,
) raises -> MZArenaResult:
    """Candidate (learner) vs best, both at full Gumbel strength, from BOTH
    colors, aggregated from the candidate's perspective."""
    var p0 = mz_arena_match[
        ENV, REP, DYN, PRED, N_PER_COLOR, OBS, ACT, LATENT, BINS,
        NUM_SIMS, MAX_NODES, MAX_K, MAX_PLIES,
    ](
        ctx, c_rep, c_dyn, c_pred, b_rep, b_dyn, b_pred, gamma,
        a_player=0, seed=seed, open_plies=open_plies,
    )
    var p1 = mz_arena_match[
        ENV, REP, DYN, PRED, N_PER_COLOR, OBS, ACT, LATENT, BINS,
        NUM_SIMS, MAX_NODES, MAX_K, MAX_PLIES,
    ](
        ctx, c_rep, c_dyn, c_pred, b_rep, b_dyn, b_pred, gamma,
        a_player=1, seed=seed + 1, open_plies=open_plies,
    )
    return MZArenaResult(
        wins=p0.wins + p1.wins,
        draws=p0.draws + p1.draws,
        losses=p0.losses + p1.losses,
    )


# ══════════════════════════════════════════════════════════════════════════
# Eval: MuZero-Gumbel agent vs a pluggable GPUEvaluator opponent (both colors).
# ══════════════════════════════════════════════════════════════════════════


def _mz_eval_one_color[
    ENV: GPUTwoPlayerDiscreteEnv,
    REP: Module,
    DYN: Module,
    PRED: Module,
    OPP: GPUEvaluator,
    N_GAMES: Int,
    OBS: Int,
    ACT: Int,
    LATENT: Int,
    BINS: Int,
    NUM_SIMS: Int,
    MAX_NODES: Int,
    MAX_K: Int,
    MAX_PLIES: Int,
](
    ctx: DeviceContext,
    mut rep: REP, mut dyn: DYN, mut pred: PRED,
    gamma: Float64,
    agent_player: Int,
    seed: UInt64,
    open_plies: Int,
) raises -> MZArenaResult:
    comptime STATE = ENV.STATE_SIZE

    rep.set_attr["training"](Scalar[DT](0.0))
    dyn.set_attr["training"](Scalar[DT](0.0))
    pred.set_attr["training"](Scalar[DT](0.0))

    var planner = GumbelGPUMCTS[
        N_GAMES, ACT, LATENT, BINS, MAX_NODES, MAX_K, NUM_SIMS, SelfPlay
    ](ctx, gamma=gamma, v_min=-1.0, v_max=1.0, gumbel_scale=0.0)
    var rep_a = MZRepGPU[OBS, LATENT, REP].make(rep)
    var dyn_a = MZDynGPU[LATENT, ACT, BINS, DYN].make(dyn)
    var pred_a = MZPredGPU[LATENT, ACT, BINS, PRED].make(pred)

    var states = ctx.enqueue_create_buffer[DT](N_GAMES * STATE)
    var obs_dev = ctx.enqueue_create_buffer[DT](N_GAMES * OBS)
    var legal_dev = ctx.enqueue_create_buffer[DT](N_GAMES * ACT)
    var rew = ctx.enqueue_create_buffer[DT](N_GAMES)
    var done = ctx.enqueue_create_buffer[DT](N_GAMES)
    var term = ctx.enqueue_create_buffer[DT](N_GAMES)
    var obs_next = ctx.enqueue_create_buffer[DT](N_GAMES * OBS)
    var legal_next = ctx.enqueue_create_buffer[DT](N_GAMES * ACT)
    var act_dev = ctx.enqueue_create_buffer[DT](N_GAMES)

    var h_pol = _a(N_GAMES * ACT)
    var legal_h = _a(N_GAMES * ACT)
    var act_h = _a(N_GAMES)
    var done_h = _a(N_GAMES)
    var rew_h = _a(N_GAMES)
    ctx.synchronize()

    ENV.reset_kernel_gpu[N_GAMES, STATE](ctx, states)
    ENV.extract_obs_kernel_gpu[N_GAMES, STATE, OBS](
        ctx, states, obs_dev, legal_dev
    )
    ctx.synchronize()

    var eval_done = Array[Bool, N_GAMES](fill=False)
    var eval_result = Array[Int, N_GAMES](fill=0)
    var all_done = False
    var move = 0
    var rng = seed | 1

    comptime MAX_MOVES = MAX_PLIES + ACT
    while not all_done and move < MAX_MOVES:
        var agent_turn = (move % 2) == agent_player
        if agent_turn:
            # The agent ALWAYS plays its best (greedy argmax over search), even
            # in the opening — we never force a random move on it, so it is never
            # handed a lost line. Eval measures pure agent skill.
            _mz_search_argmax[
                N_GAMES, OBS, ACT, LATENT, BINS,
                MAX_NODES, MAX_K, NUM_SIMS,
            ](
                ctx, planner, rep_a, dyn_a, pred_a, obs_dev, legal_dev,
                h_pol, legal_h, act_h, rng_seed=UInt32(move + 1),
            )
            ctx.enqueue_copy(act_dev, act_h)
        elif move < open_plies:
            # Opening variation comes ONLY from the opponent: random for its
            # turns inside the first `open_plies` plies. This diversifies the
            # N_GAMES lockstep batch — essential vs a DETERMINISTIC opponent
            # (Minimax), where otherwise all games would be the same line — while
            # the agent's deterministic replies fan the games out. (vs Random the
            # opponent is already random, so this is a no-op in spirit.)
            rng = _xs(rng)
            RandomOpponent.select_action_gpu[N_GAMES, ACT, STATE](
                ctx, act_dev, legal_dev, states, rng
            )
        else:
            rng = _xs(rng)
            OPP.select_action_gpu[N_GAMES, ACT, STATE](
                ctx, act_dev, legal_dev, states, rng
            )
        ctx.synchronize()

        ENV.step_kernel_gpu[N_GAMES, STATE, OBS](
            ctx, states, act_dev, rew, done, term, obs_next, legal_next
        )
        ctx.enqueue_copy(done_h, done)
        ctx.enqueue_copy(rew_h, rew)
        ctx.synchronize()

        all_done = True
        for e in range(N_GAMES):
            if not eval_done[e] and Float64(done_h[unsafe_offset=e]) > 0.5:
                eval_done[e] = True
                var r = Float64(rew_h[unsafe_offset=e])
                if r > 0.5:
                    eval_result[e] = 1 if agent_turn else 2
                elif r < -0.5:
                    eval_result[e] = 2 if agent_turn else 1
                else:
                    eval_result[e] = 3
            if not eval_done[e]:
                all_done = False

        ENV.selective_reset_kernel_gpu[N_GAMES, STATE](
            ctx, states, done, rng_seed=seed + UInt64(move) * 7 + 3
        )
        ENV.extract_obs_kernel_gpu[N_GAMES, STATE, OBS](
            ctx, states, obs_dev, legal_dev
        )
        ctx.synchronize()
        move += 1

    var wins = 0
    var draws = 0
    var losses = 0
    for e in range(N_GAMES):
        if eval_result[e] == 1:
            wins += 1
        elif eval_result[e] == 2:
            losses += 1
        else:
            draws += 1

    h_pol.unsafe_free(); legal_h.unsafe_free(); act_h.unsafe_free(); done_h.unsafe_free(); rew_h.unsafe_free()
    return MZArenaResult(wins=wins, draws=draws, losses=losses)


def mz_eval_both_colors[
    ENV: GPUTwoPlayerDiscreteEnv,
    REP: Module,
    DYN: Module,
    PRED: Module,
    OPP: GPUEvaluator,
    N_GAMES: Int,
    OBS: Int,
    ACT: Int,
    LATENT: Int,
    BINS: Int,
    NUM_SIMS: Int,
    MAX_NODES: Int,
    MAX_K: Int,
    MAX_PLIES: Int,
](
    ctx: DeviceContext,
    mut rep: REP, mut dyn: DYN, mut pred: PRED,
    gamma: Float64,
    seed: UInt64 = 1,
    open_plies: Int = 0,
) raises -> MZArenaResult:
    """Aggregate the MuZero agent's full-strength (Gumbel) record vs `OPP` over
    `N_GAMES` games as each color, so first-move advantage cancels."""
    var p0 = _mz_eval_one_color[
        ENV, REP, DYN, PRED, OPP, N_GAMES, OBS, ACT, LATENT, BINS,
        NUM_SIMS, MAX_NODES, MAX_K, MAX_PLIES,
    ](ctx, rep, dyn, pred, gamma, agent_player=0, seed=seed, open_plies=open_plies)
    var p1 = _mz_eval_one_color[
        ENV, REP, DYN, PRED, OPP, N_GAMES, OBS, ACT, LATENT, BINS,
        NUM_SIMS, MAX_NODES, MAX_K, MAX_PLIES,
    ](
        ctx, rep, dyn, pred, gamma,
        agent_player=1, seed=seed + 33333, open_plies=open_plies,
    )
    return MZArenaResult(
        wins=p0.wins + p1.wins,
        draws=p0.draws + p1.draws,
        losses=p0.losses + p1.losses,
    )


# ══════════════════════════════════════════════════════════════════════════
# Main driver.
# ══════════════════════════════════════════════════════════════════════════


def run_muzero_selfplay_arena_gumbel_2p[
    ENV: GPUTwoPlayerDiscreteEnv,
    REP: Module,
    DYN: Module,
    PRED: Module,
    AUG: BoardAugmenter,
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
    MAX_PLIES: Int,
    OPP1: GPUEvaluator = RandomOpponent,
    OPP2: GPUEvaluator = RandomOpponent,
    L: Logger = NoOpLogger,
    ARENA_GAMES: Int = 32,
    EVAL_GAMES: Int = 64,
    TEMP_MOVES: Int = 8,
    # LR schedule over OPTIMIZER STEPS (`train_steps`), not moves. Default
    # ConstantSchedule leaves `lr` untouched (bit-identical to before). Use e.g.
    # LinearWarmupSchedule[N] to ramp 0→lr over the first N grad updates — tames
    # the early instability a wider/deeper net shows under the base LR.
    SCHEDULER: Scheduler = ConstantSchedule,
    # CUDA-graph the per-step BPTT train compute (forward+reverse scan + arena
    # optimizer step). Default OFF → the per-param non-captured path, bit-identical
    # to before. ON (NVIDIA): adopts the three Adam optimizers into arena mode,
    # routes the LR schedule through the device buffer (`push_lr_device`), and
    # captures the device-kernel compute into one graph replayed per train step —
    # collapsing the per-kernel launch overhead. Requires uniform replay
    # (`use_per=False`, which board games use); a no-op on Apple (compile-time).
    USE_TRAIN_CUDA_GRAPH: Bool = False,
    # CUDA-graph the Gumbel-MCTS Sequential-Halving sim loop for self-play +
    # reanalyze search. ⚠️ KNOWN-BUGGY — DO NOT ENABLE. On NVIDIA the captured
    # sim-loop REPLAY produces FLAT search targets (target_max_prob ~0.3 vs ~0.55
    # eager), which stalls policy-head learning (policy_entropy pinned at log·ACT).
    # The eager path (refactored `search_gpu`) is correct — the bug is strictly in
    # the sim-loop capture/replay and is NOT yet root-caused. Low priority: MCTS
    # here is compute-bound (CH=64 conv) so the graph adds only ~6%. Kept behind
    # this flag for a future eager-vs-captured parity-test investigation.
    # See docs/MUZERO_CUDA_GRAPH_PLAN.md. (No-op on Apple.)
    USE_MCTS_CUDA_GRAPH: Bool = False,
](
    ctx: DeviceContext,
    mut rep: REP,           # the BEST representation net (final weights on return)
    mut dyn: DYN,           # the BEST dynamics net
    mut pred: PRED,         # the BEST prediction net
    iterations: Int,
    learning_starts: Int = 256,
    train_per_iter: Int = 1,
    lr: Scalar[DT] = Scalar[DT](2e-3),
    gamma: Scalar[DT] = Scalar[DT](1.0),
    value_coef: Scalar[DT] = Scalar[DT](0.25),
    max_grad_norm: Scalar[DT] = Scalar[DT](1.0),
    seed: UInt64 = 0,
    arena_every: Int = 2000,
    arena_open_plies: Int = 2,
    promote_threshold: Float64 = 0.55,
    report_every: Int = 1000,
    diag_every: Int = 0,
    do_eval: Bool = True,
    do_eval2: Bool = False,
    eval_best: Bool = True,          # eval the BEST (deployable) net; False =
    #                                  the learner (training-dynamics view, which
    #                                  drifts away from a frozen best once
    #                                  promotions stall).
    verbose: Bool = True,
    logger: Optional[Pointer[L, MutAnyOrigin]] = None,
    selfplay_open_plies: Int = 2,
    eval_open_plies: Int = 4,
    temperature_decay_steps: Int = 0,
    temp_min: Float64 = 0.0,         # post-TEMP_MOVES sampling temperature; 0 =
    #                                  greedy (legacy), ~0.3 keeps self-play
    #                                  stochastic to preserve replay diversity.
    qnorm_per_node: Bool = True,
    reanalyze_every: Int = 0,        # moves between reanalyze triggers (0 = off)
    reanalyze_batch: Int = N_ENVS,   # stored positions re-targeted per trigger,
    #                                  processed in `reanalyze_batch // N_ENVS`
    #                                  chunks of N_ENVS (the planner's root width);
    #                                  set ≈ B for EfficientZero-style coverage.
    target_sync_interval: Int = 0,   # grad steps between learner→target syncs;
    #                                  0 = refresh target to live just before each
    #                                  trigger (live-learner reanalyze).
    checkpoint_every: Int = 0,       # moves between checkpoint saves (0 = off)
    checkpoint_path: String = String(""),  # rolling save of the BEST trio
    use_per: Bool = True,            # Prioritized Experience Replay toggle. The
    #                                  replay is ALWAYS the device-obs
    #                                  `PrioritizedMCTSSequenceReplay`; this flag
    #                                  only gates the PER behaviour. True: sample
    #                                  ∝ priorityᵅ, weight grads by IS weights,
    #                                  write back root value-error priorities.
    #                                  False: priorities stay constant (uniform
    #                                  sampling) + no IS weighting → bit-identical
    #                                  to uniform replay, still on-device.
    per_alpha: Scalar[DT] = Scalar[DT](1.0),  # priority exponent (EZ Atari: 1.0)
    per_beta: Scalar[DT] = Scalar[DT](1.0),   # IS-weight exponent (EZ Atari: 1.0)
) raises -> MZArenaRunResult:
    """Two-player Gumbel MuZero self-play with Arena gating. `rep`/`dyn`/`pred`
    are the BEST nets and hold the final (best) weights on return. One loop pass
    advances all `N_ENVS` games one move; `iterations`/`report_every`/
    `arena_every` are in moves. v_min/v_max are pinned to -1/+1 (board outcome)."""
    comptime STATE = ENV.STATE_SIZE
    comptime NSYM = AUG.NUM_SYMMETRIES
    comptime VMIN = Scalar[DT](-1.0)
    comptime VMAX = Scalar[DT](1.0)

    # ── Learner net trio (trains), initialised to the best weights ──
    var l_rep = REP.make["gpu", Kaiming](Optional(ctx))
    var l_dyn = DYN.make["gpu", Kaiming](Optional(ctx))
    var l_pred = PRED.make["gpu", Kaiming](Optional(ctx))
    hard_copy["gpu", M=REP](rep, l_rep, Optional(ctx))
    hard_copy["gpu", M=DYN](dyn, l_dyn, Optional(ctx))
    hard_copy["gpu", M=PRED](pred, l_pred, Optional(ctx))

    var orep = Adam(lr=lr)
    var odyn = Adam(lr=lr)
    var opred = Adam(lr=lr)
    # grad clip is applied inside the unroll via clip_grad_norm (max_grad_norm
    # threaded to the mz_unroll_train_step_gpu call below).

    # CUDA-graph capture of the train step (opt-in). Adopt the three optimizers
    # into ARENA mode so the captured step advances β^t / reads the LR from
    # device buffers (the per-param path bakes them host-side → not replay-safe),
    # and own the capture slot (None until first capture). Requires uniform
    # replay; board games use `use_per=False`.
    var train_graph: Optional[CUDAGraph] = None
    # Separate MCTS capture slots — self-play searches the BEST nets, reanalyze
    # the TARGET nets; the captured sim kernels bake the net buffer pointers, so
    # the two roles must NOT share a graph.
    var selfplay_mcts_graph: Optional[CUDAGraph] = None
    var reanalyze_mcts_graph: Optional[CUDAGraph] = None
    # HARD GATE: refuse to run with the MCTS graph on. Its captured sim-loop
    # replay produces flat search targets (policy stops learning) and is not
    # root-caused; this raises so it can't be enabled by mistake. Remove only
    # after an eager-vs-captured parity test proves the replay correct.
    comptime if USE_MCTS_CUDA_GRAPH:
        raise Error(
            "USE_MCTS_CUDA_GRAPH is KNOWN-BUGGY and disabled: the captured"
            " Gumbel-MCTS sim-loop replay produces FLAT search targets"
            " (target_max_prob collapses → policy head stops learning). The"
            " eager search path is correct. See docs/MUZERO_CUDA_GRAPH_PLAN.md."
        )
    comptime if USE_TRAIN_CUDA_GRAPH:
        if use_per:
            raise Error(
                "USE_TRAIN_CUDA_GRAPH requires uniform replay (use_per=False);"
                " PER's host-side priority path is not capture-safe."
            )
        orep.adopt["gpu"](l_rep, Optional(ctx))
        odyn.adopt["gpu"](l_dyn, Optional(ctx))
        opred.adopt["gpu"](l_pred, Optional(ctx))

    # ── Lagging TARGET net trio for reanalyze. Synced from the LEARNER every
    #    `target_sync_interval` grad steps (a delayed target that decouples
    #    target generation from the optimizer step — the standard stabilizer;
    #    matches the single/batched MuZero drivers), or refreshed to live just
    #    before each trigger when 0 (live-learner reanalyze). Reanalyze ALWAYS
    #    searches through these adapters. Params-only copy (the h/g/f carry no
    #    BatchNorm running stats). Only allocated/used when reanalyze is on. ──
    var rep_t = REP.make["gpu", Kaiming](Optional(ctx))
    var dyn_t = DYN.make["gpu", Kaiming](Optional(ctx))
    var pred_t = PRED.make["gpu", Kaiming](Optional(ctx))
    hard_copy["gpu", M=REP](l_rep, rep_t, Optional(ctx))
    hard_copy["gpu", M=DYN](l_dyn, dyn_t, Optional(ctx))
    hard_copy["gpu", M=PRED](l_pred, pred_t, Optional(ctx))
    rep_t.set_attr["training"](Scalar[DT](0.0))
    dyn_t.set_attr["training"](Scalar[DT](0.0))
    pred_t.set_attr["training"](Scalar[DT](0.0))
    var rep_ta = MZRepGPU[OBS, LATENT, REP].make(rep_t)
    var dyn_ta = MZDynGPU[LATENT, ACT, BINS, DYN].make(dyn_t)
    var pred_ta = MZPredGPU[LATENT, ACT, BINS, PRED].make(pred_t)
    var train_steps = 0

    # ── Self-play planner: best nets generate the data (Gumbel exploration on) ──
    var planner = GumbelGPUMCTS[
        N_ENVS, ACT, LATENT, BINS, MAX_NODES, MAX_K, NUM_SIMS, SelfPlay
    ](
        ctx, gamma=Float64(gamma), v_min=-1.0, v_max=1.0,
        gumbel_scale=1.0, qnorm_per_node=qnorm_per_node,
    )
    var rep_a = MZRepGPU[OBS, LATENT, REP].make(rep)
    var dyn_a = MZDynGPU[LATENT, ACT, BINS, DYN].make(dyn)
    var pred_a = MZPredGPU[LATENT, ACT, BINS, PRED].make(pred)

    # Device-obs prioritized replay (serves BOTH modes — see `use_per`). The obs
    # ring lives on the GPU; training samples gather obs device→device (no host
    # obs round-trip), the value-error priorities steer sampling when PER is on.
    var rb = PrioritizedMCTSSequenceReplay[OBS, ACT, CAP](
        ctx, seed=seed ^ UInt64(0xABCDEF), alpha=per_alpha, beta=per_beta
    )
    var scratch = MZScratch[B, K, OBS, ACT, LATENT, BINS].make(ctx)

    # ── diag scratch: re-forward the root prediction on the last train batch,
    #    D2H, and emit the full MuZero head-fit metric set (target entropy /
    #    max-prob / value head) — same set as the other MuZero drivers. ──
    var d_diag_obs = ctx.enqueue_create_buffer[DT](B * OBS)
    var d_diag_z = ctx.enqueue_create_buffer[DT](B * LATENT)
    var d_diag_pred = ctx.enqueue_create_buffer[DT](B * (ACT + BINS))
    var h_diag_pred = _a(B * (ACT + BINS))

    # ── reanalyze scratch: stored obs gathered device→device via the replay's
    #    `gather_obs_for_positions` (no host obs round-trip); the legal mask is
    #    still read host-side (it gates the root search). Own buffers, NOT the
    #    live self-play obs/legal. ──
    var d_reana = ctx.enqueue_create_buffer[DT](N_ENVS * OBS)
    var d_reana_legal = ctx.enqueue_create_buffer[DT](N_ENVS * ACT)
    var d_reana_slots = ctx.enqueue_create_buffer[DType.int32](N_ENVS)
    var h_reana_slots = List[Int32](length=N_ENVS, fill=0)
    var h_reana_legal = _a(N_ENVS * ACT)

    # ── Device buffers ──
    var states = ctx.enqueue_create_buffer[DT](N_ENVS * STATE)
    var obs_dev = ctx.enqueue_create_buffer[DT](N_ENVS * OBS)
    var legal_dev = ctx.enqueue_create_buffer[DT](N_ENVS * ACT)
    var rew = ctx.enqueue_create_buffer[DT](N_ENVS)
    var done = ctx.enqueue_create_buffer[DT](N_ENVS)
    var term = ctx.enqueue_create_buffer[DT](N_ENVS)
    var obs_next = ctx.enqueue_create_buffer[DT](N_ENVS * OBS)
    var legal_next = ctx.enqueue_create_buffer[DT](N_ENVS * ACT)
    var act_dev = ctx.enqueue_create_buffer[DT](N_ENVS)

    # ── Host mirrors ──
    var obs_h = _a(N_ENVS * OBS)
    var pol_h = _a(N_ENVS * ACT)
    var val_h = _a(N_ENVS)
    var legal_h = _a(N_ENVS * ACT)
    var done_h = _a(N_ENVS)
    var rew_h = _a(N_ENVS)
    var act_h = _a(N_ENVS)

    # ── Training slabs (host, time-major) — owned Lists (RAII), passed to the
    # device-PER sample + List-input unroll. ──
    var t_obs0 = List[Scalar[DT]](length=B * OBS, fill=0)  # unused on-device
    var t_act = List[Scalar[DT]](length=K * B, fill=0)
    var t_pol = List[Scalar[DT]](length=(K + 1) * B * ACT, fill=0)
    var t_val = List[Scalar[DT]](length=(K + 1) * B, fill=0)
    var t_rew = List[Scalar[DT]](length=K * B, fill=0)
    var l_parts = _a(3)

    # ── PER / on-device sampling scratch ──
    # The prioritized sampler gathers the FULL [(K+1)*B, OBS] obs window into
    # `d_obs_seq` (MuZero only consumes the root k=0 block → copied into
    # `scratch.d_obs0`). `t_isw`/`t_prio` are the per-sample IS weights / new
    # value-error priorities; `t_slots` the sampled root ring slots (for the
    # priority write-back).
    var d_obs_seq = ctx.enqueue_create_buffer[DT]((K + 1) * B * OBS)
    var d_seq_slots = ctx.enqueue_create_buffer[DType.int32]((K + 1) * B)
    var h_seq_slots = List[Int32](length=(K + 1) * B, fill=0)
    var t_isw = List[Scalar[DT]](length=B, fill=0)
    var t_prio = List[Scalar[DT]](length=B, fill=0)
    var t_slots = List[Int](length=B, fill=0)

    # ── Augmentation scratch (one game's worth, reused) — owned Lists; the
    # storage BoardAugmenter takes List src+offset and writes a List dst. ──
    var aug_obs = List[Scalar[DT]](length=MAX_PLIES * OBS, fill=0)
    var aug_pol = List[Scalar[DT]](length=MAX_PLIES * ACT, fill=0)
    var aug_legal = List[Scalar[DT]](length=MAX_PLIES * ACT, fill=0)
    var aug_act = List[Scalar[DT]](length=MAX_PLIES, fill=0)
    var onehot = List[Scalar[DT]](length=ACT, fill=0)
    var onehot_out = List[Scalar[DT]](length=ACT, fill=0)

    # ── Per-env in-progress episode accumulators ──
    var e_obs = List[List[Scalar[DT]]]()
    var e_act = List[List[Scalar[DT]]]()
    var e_rew = List[List[Scalar[DT]]]()
    var e_pol = List[List[Scalar[DT]]]()
    var e_val = List[List[Scalar[DT]]]()
    var e_tp = List[List[Scalar[DT]]]()
    var e_legal = List[List[Scalar[DT]]]()
    for _ in range(N_ENVS):
        e_obs.append(List[Scalar[DT]]())
        e_act.append(List[Scalar[DT]]())
        e_rew.append(List[Scalar[DT]]())
        e_pol.append(List[Scalar[DT]]())
        e_val.append(List[Scalar[DT]]())
        e_tp.append(List[Scalar[DT]]())
        e_legal.append(List[Scalar[DT]]())
    ctx.synchronize()

    ENV.reset_kernel_gpu[N_ENVS, STATE](ctx, states)
    ENV.extract_obs_kernel_gpu[N_ENVS, STATE, OBS](
        ctx, states, obs_dev, legal_dev
    )
    ctx.synchronize()

    var rng = seed ^ UInt64(0x9E3779B97F4A7C15)
    var mcts_seed = UInt32(seed & UInt64(0xFFFF))
    var last_loss = 0.0
    var promotions = 0
    var total_games = 0
    var games_prev = 0
    var rep_cadence = report_every if report_every > 0 else arena_every

    if verbose:
        print(
            "MuZero(2p Gumbel) self-play:", iterations, "moves,",
            N_ENVS, "envs,", NUM_SIMS, "sims/move | eval1=", OPP1.NAME,
            "eval2=", OPP2.NAME if do_eval2 else String("off"),
            "| report_every=", rep_cadence, "moves",
        )

    for it in range(iterations):
        # ── 1. Self-play search with the BEST nets (eval mode) ──
        rep.set_attr["training"](Scalar[DT](0.0))
        dyn.set_attr["training"](Scalar[DT](0.0))
        pred.set_attr["training"](Scalar[DT](0.0))
        ctx.enqueue_copy(planner.legal_mask_view(), legal_dev)
        var obs_t = LayoutTensor[DT, Layout.row_major(N_ENVS, OBS), MutAnyOrigin](
            obs_dev.unsafe_ptr().as_unsafe_any_origin()
        )
        comptime if USE_MCTS_CUDA_GRAPH:
            planner.search_gpu_captured[
                type_of(rep_a), type_of(dyn_a), type_of(pred_a),
            ](
                ctx, rep_a, dyn_a, pred_a, obs_t, selfplay_mcts_graph,
                apply_legal=True, k_actual=MAX_K, rng_seed=mcts_seed,
            )
        else:
            planner.search_gpu[
                type_of(rep_a), type_of(dyn_a), type_of(pred_a),
            ](
                ctx, rep_a, dyn_a, pred_a, obs_t,
                apply_legal=True, k_actual=MAX_K, rng_seed=mcts_seed,
            )
        mcts_seed += UInt32(1)

        ctx.enqueue_copy(obs_h, obs_dev)
        ctx.enqueue_copy(pol_h, planner.policies_view())
        ctx.enqueue_copy(val_h, planner.root_value_view())
        ctx.enqueue_copy(legal_h, legal_dev)
        ctx.synchronize()

        # ── 2. Per env: record the labelled step, choose the action played ──
        var temp = visit_temperature(it, temperature_decay_steps)
        for e in range(N_ENVS):
            var ply = len(e_act[e])
            var tp = Scalar[DT](ply % 2)
            for j in range(OBS):
                e_obs[e].append(obs_h[unsafe_offset=e * OBS + j])
            for a in range(ACT):
                e_pol[e].append(pol_h[unsafe_offset=e * ACT + a])
                e_legal[e].append(legal_h[unsafe_offset=e * ACT + a])
            e_val[e].append(val_h[unsafe_offset=e])
            e_tp[e].append(tp)

            # Action selection: opening diversity → temp sampling → argmax,
            # with a final legality guard (improved policy is legal-masked, but
            # guard against a degenerate -inf row picking an illegal column).
            var a_sel = -1
            if ply < selfplay_open_plies:
                var n_legal = 0
                for a in range(ACT):
                    if Float64(legal_h[unsafe_offset=e * ACT + a]) > 0.5:
                        n_legal += 1
                if n_legal > 0:
                    rng = _xs(rng)
                    var pick = Int(rng % UInt64(n_legal))
                    var seen = 0
                    for a in range(ACT):
                        if Float64(legal_h[unsafe_offset=e * ACT + a]) > 0.5:
                            if seen == pick:
                                a_sel = a
                                break
                            seen += 1
            else:
                # Effective temperature. Two regimes:
                #  • `temperature_decay_steps > 0` (muzero-general recipe): the
                #    WHOLE game (past the random openings) uses the scheduled
                #    `temp` (1.0 → 0.5 at 50% training → 0.25 at 75%). Early
                #    training stays diverse (T=1.0); the back half sharpens —
                #    incl. the endgame — so the full-MC value target finally
                #    reflects position quality instead of a coin-flip outcome
                #    (the lever for the value_mse / loss_value plateau).
                #  • schedule off (default `temperature_decay_steps = 0`): the
                #    legacy split — scheduled `temp` for the first TEMP_MOVES
                #    plies, then `temp_min` thereafter (temp_min > 0 keeps play
                #    stochastic ∝ visits^(1/temp_min), the AlphaZero.jl
                #    convention; temp_min = 0.0 recovers greedy-after-opening).
                #    Fully-greedy late play collapses self-play diversity. This
                #    branch is bit-identical to the pre-anneal behaviour.
                var eff_temp = (
                    temp
                    if temperature_decay_steps > 0
                    else (temp if ply < TEMP_MOVES else temp_min)
                )
                if eff_temp > 0.0:
                    var wsum = 0.0
                    var w = List[Float64](capacity=ACT)
                    for a in range(ACT):
                        var p = Float64(pol_h[unsafe_offset=e * ACT + a])
                        if eff_temp != 1.0 and p > 0.0:
                            p = exp(log(p) / eff_temp)
                        w.append(p)
                        wsum += p
                    rng = _xs(rng)
                    var r = (
                        Float64(rng % UInt64(1_000_000)) / 1_000_000.0 * wsum
                    )
                    var cum = 0.0
                    for a in range(ACT):
                        cum += w[a]
                        if r <= cum and w[a] > 0.0:
                            a_sel = a
                            break
                # eff_temp <= 0 → greedy: fall through to the argmax below.
            if a_sel < 0:
                var bv = -1.0
                for a in range(ACT):
                    var p = Float64(pol_h[unsafe_offset=e * ACT + a])
                    if p > bv:
                        bv = p
                        a_sel = a
            if a_sel < 0 or Float64(legal_h[unsafe_offset=e * ACT + a_sel]) <= 0.5:
                var bestl = -1
                var bvl = -1.0e30
                for a in range(ACT):
                    if Float64(legal_h[unsafe_offset=e * ACT + a]) > 0.5:
                        var p = Float64(pol_h[unsafe_offset=e * ACT + a])
                        if bestl < 0 or p > bvl:
                            bvl = p
                            bestl = a
                if bestl >= 0:
                    a_sel = bestl
            e_act[e].append(Scalar[DT](a_sel))
            act_h[unsafe_offset=e] = Scalar[DT](a_sel)
        ctx.enqueue_copy(act_dev, act_h)

        # ── 3. Step every game ──
        ENV.step_kernel_gpu[N_ENVS, STATE, OBS](
            ctx, states, act_dev, rew, done, term, obs_next, legal_next
        )
        ctx.enqueue_copy(done_h, done)
        ctx.enqueue_copy(rew_h, rew)
        ctx.synchronize()

        # ── 4. Flush finished games into the replay (reward recorded per step;
        #      symmetry-augmented). Board games always terminate → truncated
        #      = False (every `done` is a real terminal). ──
        for e in range(N_ENVS):
            e_rew[e].append(rew_h[unsafe_offset=e])
            if Float64(done_h[unsafe_offset=e]) > 0.5:
                total_games += 1
                var Lg = len(e_act[e])
                for s in range(NSYM):
                    if s == 0:
                        rb.store_episode(
                            e_obs[e],
                            e_act[e],
                            e_rew[e],
                            e_pol[e],
                            e_val[e],
                            e_tp[e],
                            e_legal[e],
                            Lg,
                            truncated=False,
                        )
                    else:
                        for k in range(Lg):
                            # storage BoardAugmenter writes dst from index 0
                            # (OBS/ACT-sized) → augment into a per-step temp List
                            # then copy into this game's full augmented slab.
                            var tmp_obs = List[Scalar[DT]](length=OBS, fill=0)
                            AUG.augment_obs[OBS](e_obs[e], k * OBS, s, tmp_obs)
                            for j in range(OBS):
                                aug_obs[k * OBS + j] = tmp_obs[j]
                            var tmp_pol = List[Scalar[DT]](length=ACT, fill=0)
                            AUG.augment_policy[ACT](e_pol[e], k * ACT, s, tmp_pol)
                            for a in range(ACT):
                                aug_pol[k * ACT + a] = tmp_pol[a]
                            var tmp_leg = List[Scalar[DT]](length=ACT, fill=0)
                            AUG.augment_policy[ACT](e_legal[e], k * ACT, s, tmp_leg)
                            for a in range(ACT):
                                aug_legal[k * ACT + a] = tmp_leg[a]
                            # Permute the stored action through the same symmetry
                            # (one-hot → augment_policy → argmax).
                            for a in range(ACT):
                                onehot[a] = Scalar[DT](0.0)
                            onehot[Int(e_act[e][k])] = Scalar[DT](1.0)
                            AUG.augment_policy[ACT](onehot, 0, s, onehot_out)
                            var ba = 0
                            for a in range(1, ACT):
                                if Float64(onehot_out[a]) > Float64(onehot_out[ba]):
                                    ba = a
                            aug_act[k] = Scalar[DT](ba)
                        rb.store_episode(
                            aug_obs,
                            aug_act,
                            e_rew[e],
                            aug_pol,
                            e_val[e],
                            e_tp[e],
                            aug_legal,
                            Lg,
                            truncated=False,
                        )
                e_obs[e].clear(); e_act[e].clear(); e_rew[e].clear()
                e_pol[e].clear(); e_val[e].clear(); e_tp[e].clear()
                e_legal[e].clear()

        # ── 5. Reset finished games, refresh obs/legal ──
        ENV.selective_reset_kernel_gpu[N_ENVS, STATE](
            ctx, states, done, rng_seed=seed + UInt64(it)
        )
        ENV.extract_obs_kernel_gpu[N_ENVS, STATE, OBS](
            ctx, states, obs_dev, legal_dev
        )
        ctx.synchronize()

        # ── 6. Train the LEARNER nets (K-step BPTT unroll) ──
        var trained = (
            rb.num_steps() >= learning_starts and rb.num_episodes() > 0
        )
        if trained:
            l_rep.set_attr["training"](Scalar[DT](1.0))
            l_dyn.set_attr["training"](Scalar[DT](1.0))
            l_pred.set_attr["training"](Scalar[DT](1.0))
            for _t in range(train_per_iter):
                # LR schedule over optimizer steps (default constant → no code).
                # Under capture, route the scheduled LR through the device buffer
                # (`push_lr_device`, stable pointer) so the captured step reads the
                # FRESH lr on each replay; the host `orep.lr = …` write would
                # freeze at the capture-time value. Non-capture path is unchanged.
                comptime if not SCHEDULER.IS_CONSTANT:
                    var _scl = Scalar[DT](
                        SCHEDULER.lr_scale_at(
                            train_steps, iterations * train_per_iter
                        )
                    )
                    comptime if USE_TRAIN_CUDA_GRAPH:
                        orep.push_lr_device["gpu"](lr * _scl, Optional(ctx))
                        odyn.push_lr_device["gpu"](lr * _scl, Optional(ctx))
                        opred.push_lr_device["gpu"](lr * _scl, Optional(ctx))
                    else:
                        orep.lr = lr * _scl
                        odyn.lr = lr * _scl
                        opred.lr = lr * _scl
                # Prioritized device sample: gathers the obs window into
                # `d_obs_seq` and the n-step targets into the host slabs. With PER
                # on, also writes the paper priority |ν − z| into `t_prio` (the
                # sample owns this — it has both ν and z); board games leave it
                # None → priorities stay uniform.
                var samp_prio = Optional[
                    Pointer[Scalar[DT], MutAnyOrigin]
                ](None)
                if use_per:
                    samp_prio = Optional[
                        Pointer[Scalar[DT], MutAnyOrigin]
                    ](t_prio.unsafe_ptr().as_unsafe_any_origin())
                rb.sample_training_batch_seq_per_gpu[B, K, N](
                    ctx, gamma, d_obs_seq, d_seq_slots, h_seq_slots,
                    t_act, t_pol, t_val, t_rew, t_isw, t_slots,
                    out_prio=samp_prio,
                )
                # MuZero consumes only the root (k=0) obs block → copy it into
                # the train scratch (`obs_on_device=True` reads `scratch.d_obs0`).
                ctx.enqueue_copy(
                    scratch.d_obs0.dev.value(),
                    d_obs_seq.create_sub_buffer[DT](0, B * OBS),
                )
                # PER gate: IS-weight the grads only when on (priorities are
                # written by the sample above, paper formula |ν − z|). Off →
                # unweighted, priorities untouched (→ uniform sampling).
                var isw_opt = Optional[
                    Pointer[Scalar[DT], MutAnyOrigin]
                ](None)
                if use_per:
                    isw_opt = Optional[
                        Pointer[Scalar[DT], MutAnyOrigin]
                    ](t_isw.unsafe_ptr().as_unsafe_any_origin())
                comptime if USE_TRAIN_CUDA_GRAPH:
                    # Eager prologue (H2D) → captured device compute (forward +
                    # reverse scan + arena optimizer step) → eager epilogue (sync
                    # + D2H + reduce). `use_per` is False here (guarded at setup),
                    # so no IS-weights and no priorities in the captured body.
                    _mz_train_prologue_gpu[
                        B, K, OBS, ACT, LATENT, BINS, obs_on_device=True
                    ](ctx, scratch, t_obs0, t_act, t_pol, t_val, t_rew)

                    def _mz_compute() capturing raises -> None:
                        _mz_train_fwdrev_gpu[
                            REP, DYN, PRED, B, K, OBS, ACT, LATENT, BINS
                        ](
                            ctx, l_rep, l_dyn, l_pred, scratch,
                            VMIN, VMAX, value_coef, False, False,
                        )
                        _mz_arena_opt_step_gpu[REP, DYN, PRED](
                            ctx, l_rep, l_dyn, l_pred, orep, odyn, opred,
                            Float64(max_grad_norm),
                        )

                    maybe_capture_replay[_mz_compute](train_graph, ctx)
                    last_loss = Float64(
                        _mz_train_epilogue_gpu[B, K, OBS, ACT, LATENT, BINS](
                            ctx, scratch, False, loss_parts=l_parts
                        )
                    )
                else:
                    last_loss = Float64(
                        mz_unroll_train_step_gpu[
                            REP, DYN, PRED, B, K, OBS, ACT, LATENT, BINS,
                            obs_on_device=True,
                        ](
                            ctx, l_rep, l_dyn, l_pred, orep, odyn, opred,
                            scratch,
                            t_obs0, t_act, t_pol, t_val, t_rew,
                            VMIN, VMAX, value_coef, Float64(max_grad_norm),
                            loss_parts=l_parts,
                            is_weights=isw_opt,
                        )
                    )
                if use_per:
                    rb.update_priorities(t_slots, t_prio, B)
                train_steps += 1
                if (
                    target_sync_interval > 0
                    and train_steps % target_sync_interval == 0
                ):
                    hard_copy["gpu", M=REP](l_rep, rep_t, Optional(ctx))
                    hard_copy["gpu", M=DYN](l_dyn, dyn_t, Optional(ctx))
                    hard_copy["gpu", M=PRED](l_pred, pred_t, Optional(ctx))

        # ── 6b. Per-batch diagnostics → logger. Re-forwards the LEARNER root
        #      prediction on the last train batch (`t_obs0`) and emits loss +
        #      loss_policy/value/reward split + the head-fit metrics
        #      (policy_ce/entropy, target_entropy/max_prob, value_mse/mean,
        #      value_target_mean) — the same set the other MuZero drivers log. ──
        if (
            Bool(logger)
            and diag_every > 0
            and trained
            and (it + 1) % diag_every == 0
        ):
            # The obs now lives on-device (`scratch.d_obs0`, last train batch);
            # D2H the root block into `t_obs0` so the diag re-forward sees it.
            ctx.enqueue_copy(t_obs0.unsafe_ptr(), scratch.d_obs0.dev.value())
            ctx.synchronize()
            _mz_emit_train_diag[REP, PRED, B, OBS, ACT, BINS, L](
                ctx, l_rep, l_pred, d_diag_obs, d_diag_z, d_diag_pred,
                h_diag_pred, t_obs0, t_pol, t_val, l_parts,
                VMIN, VMAX, last_loss, it + 1, logger.value(),
            )

        # ── 6c. Reanalyze: refresh stale (policy, value) targets on stored
        #      positions with the lagging target net. Each chunk gathers N_ENVS
        #      sampled positions' obs + their stored legal mask host→device, runs
        #      one batched Gumbel search (apply_legal=True — masks the same
        #      illegal moves the original search did), and writes the fresh root
        #      policy + value back in place; the n-step targets pick them up on
        #      the next sample. Lifting `reanalyze_batch` toward B is the
        #      EfficientZero-style coverage lever. ──
        if (
            reanalyze_every > 0
            and trained
            and (it + 1) % reanalyze_every == 0
        ):
            # target_sync_interval == 0 ⇒ live-net reanalyze: refresh the target
            # to the current learner weights now (search through rep_ta == l_rep).
            if target_sync_interval == 0:
                hard_copy["gpu", M=REP](l_rep, rep_t, Optional(ctx))
                hard_copy["gpu", M=DYN](l_dyn, dyn_t, Optional(ctx))
                hard_copy["gpu", M=PRED](l_pred, pred_t, Optional(ctx))
            var n_chunks = reanalyze_batch // N_ENVS
            if n_chunks < 1:
                n_chunks = 1
            for _c in range(n_chunks):
                var rpos_e = List[Int]()
                var rpos_o = List[Int]()
                for e in range(N_ENVS):
                    var rpos = rb.sample_position()
                    rpos_e.append(rpos[0])
                    rpos_o.append(rpos[1])
                    var lm = rb.read_legal(rpos[0], rpos[1])
                    for a in range(ACT):
                        h_reana_legal[unsafe_offset=e * ACT + a] = (
                            Scalar[DT](1.0) if lm[a] else Scalar[DT](0.0)
                        )
                # obs gathered device→device straight into `d_reana`.
                rb.gather_obs_for_positions[N_ENVS](
                    ctx, d_reana, d_reana_slots, h_reana_slots, rpos_e, rpos_o
                )
                ctx.enqueue_copy(d_reana_legal, h_reana_legal)
                ctx.enqueue_copy(planner.legal_mask_view(), d_reana_legal)
                var reana_t = LayoutTensor[
                    DT, Layout.row_major(N_ENVS, OBS), MutAnyOrigin
                ](d_reana.unsafe_ptr().as_unsafe_any_origin())
                comptime if USE_MCTS_CUDA_GRAPH:
                    planner.search_gpu_captured[
                        type_of(rep_ta), type_of(dyn_ta), type_of(pred_ta),
                    ](
                        ctx, rep_ta, dyn_ta, pred_ta, reana_t,
                        reanalyze_mcts_graph,
                        apply_legal=True, k_actual=MAX_K, rng_seed=mcts_seed,
                    )
                else:
                    planner.search_gpu[
                        type_of(rep_ta), type_of(dyn_ta), type_of(pred_ta),
                    ](
                        ctx, rep_ta, dyn_ta, pred_ta, reana_t,
                        apply_legal=True, k_actual=MAX_K, rng_seed=mcts_seed,
                    )
                mcts_seed += UInt32(1)
                ctx.enqueue_copy(pol_h, planner.policies_view())
                ctx.enqueue_copy(val_h, planner.root_value_view())
                ctx.synchronize()
                for e in range(N_ENVS):
                    # update_targets takes a List policy slice; copy this env's
                    # row out of the raw D2H staging mirror pol_h.
                    var pe = List[Scalar[DT]](length=ACT, fill=0)
                    for a in range(ACT):
                        pe[a] = pol_h[unsafe_offset=e * ACT + a]
                    rb.update_targets(rpos_e[e], rpos_o[e], pe, val_h[unsafe_offset=e])

        # ── 7. Arena gating: learner challenges best ──
        if (
            trained
            and arena_every > 0
            and (it + 1) % arena_every == 0
        ):
            var rec = mz_candidate_winrate[
                ENV, REP, DYN, PRED, ARENA_GAMES, OBS, ACT, LATENT, BINS,
                NUM_SIMS, MAX_NODES, MAX_K, MAX_PLIES,
            ](
                ctx, l_rep, l_dyn, l_pred, rep, dyn, pred, Float64(gamma),
                seed=seed + UInt64(it) * 7 + 1, open_plies=arena_open_plies,
            )
            var accepted = _mz_should_promote(
                rec, promote_threshold, min_decisive=ARENA_GAMES // 2
            )
            if accepted:
                hard_copy["gpu", M=REP](l_rep, rep, Optional(ctx))
                hard_copy["gpu", M=DYN](l_dyn, dyn, Optional(ctx))
                hard_copy["gpu", M=PRED](l_pred, pred, Optional(ctx))
                promotions += 1
            if verbose:
                print(
                    "  arena @ move", it + 1,
                    "| learner vs best  W", rec.wins, "D", rec.draws,
                    "L", rec.losses,
                    "→ ACCEPTED" if accepted else "→ rejected",
                    "(promotions", promotions, ")",
                )

        # ── 8. Periodic report: MCTS-eval the deployable BEST net (or the
        #      learner if eval_best=False) vs the opponents ──
        if (
            rep_cadence > 0
            and (it + 1) % rep_cadence == 0
            and trained
        ):
            var games_delta = total_games - games_prev
            games_prev = total_games
            var names = List[String]()
            var values = List[Float64]()
            names.append(String("loss")); values.append(last_loss)
            names.append(String("games")); values.append(Float64(total_games))
            names.append(String("buffer_size"))
            values.append(Float64(rb.num_steps()))
            names.append(String("promotions"))
            values.append(Float64(promotions))

            var line = String("  move ") + String(it + 1)
            line += " | games " + String(total_games)
            line += " (+" + String(games_delta) + ")"
            line += " | loss " + String(last_loss)
            line += " | replay " + String(rb.num_steps())
            line += " | promo " + String(promotions)

            if do_eval:
                var e1 = (
                    mz_eval_both_colors[
                        ENV, REP, DYN, PRED, OPP1, EVAL_GAMES, OBS, ACT, LATENT,
                        BINS, NUM_SIMS, MAX_NODES, MAX_K, MAX_PLIES,
                    ](
                        ctx, rep, dyn, pred, Float64(gamma),
                        seed=seed + UInt64(it) * 13 + 5,
                        open_plies=eval_open_plies,
                    )
                    if eval_best
                    else mz_eval_both_colors[
                        ENV, REP, DYN, PRED, OPP1, EVAL_GAMES, OBS, ACT, LATENT,
                        BINS, NUM_SIMS, MAX_NODES, MAX_K, MAX_PLIES,
                    ](
                        ctx, l_rep, l_dyn, l_pred, Float64(gamma),
                        seed=seed + UInt64(it) * 13 + 5,
                        open_plies=eval_open_plies,
                    )
                )
                var tot1 = e1.wins + e1.draws + e1.losses
                var wr1 = Float64(e1.wins) / Float64(tot1) if tot1 > 0 else 0.0
                names.append(String("eval1_win")); values.append(Float64(e1.wins))
                names.append(String("eval1_draw")); values.append(Float64(e1.draws))
                names.append(String("eval1_loss")); values.append(Float64(e1.losses))
                names.append(String("eval1_winrate")); values.append(wr1)
                line += (
                    " | vs " + OPP1.NAME + " W" + String(e1.wins)
                    + " D" + String(e1.draws) + " L" + String(e1.losses)
                    + " (wr " + String(Int(wr1 * 100.0)) + "%)"
                )

            if do_eval2:
                var e2 = (
                    mz_eval_both_colors[
                        ENV, REP, DYN, PRED, OPP2, EVAL_GAMES, OBS, ACT, LATENT,
                        BINS, NUM_SIMS, MAX_NODES, MAX_K, MAX_PLIES,
                    ](
                        ctx, rep, dyn, pred, Float64(gamma),
                        seed=seed + UInt64(it) * 17 + 9,
                        open_plies=eval_open_plies,
                    )
                    if eval_best
                    else mz_eval_both_colors[
                        ENV, REP, DYN, PRED, OPP2, EVAL_GAMES, OBS, ACT, LATENT,
                        BINS, NUM_SIMS, MAX_NODES, MAX_K, MAX_PLIES,
                    ](
                        ctx, l_rep, l_dyn, l_pred, Float64(gamma),
                        seed=seed + UInt64(it) * 17 + 9,
                        open_plies=eval_open_plies,
                    )
                )
                var tot2 = e2.wins + e2.draws + e2.losses
                var wr2 = Float64(e2.wins) / Float64(tot2) if tot2 > 0 else 0.0
                names.append(String("eval2_win")); values.append(Float64(e2.wins))
                names.append(String("eval2_draw")); values.append(Float64(e2.draws))
                names.append(String("eval2_loss")); values.append(Float64(e2.losses))
                names.append(String("eval2_winrate")); values.append(wr2)
                line += (
                    " | vs " + OPP2.NAME + " W" + String(e2.wins)
                    + " D" + String(e2.draws) + " L" + String(e2.losses)
                    + " (wr " + String(Int(wr2 * 100.0)) + "%)"
                )

            if verbose:
                print(line)
            if logger:
                logger.value()[].log_scalars(names, values, it + 1)

        # ── 9. Periodic checkpoint: rolling save of the BEST trio (the
        #      deployable artifact) so a long run is recoverable and playable
        #      mid-training. Overwrites `checkpoint_path` each time. ──
        if (
            checkpoint_every > 0
            and checkpoint_path.byte_length() > 0
            and (it + 1) % checkpoint_every == 0
        ):
            rep.set_attr["training"](Scalar[DT](0.0))
            dyn.set_attr["training"](Scalar[DT](0.0))
            pred.set_attr["training"](Scalar[DT](0.0))
            _mz_save_trio[REP, DYN, PRED](ctx, rep, dyn, pred, checkpoint_path)
            if verbose:
                print("  checkpoint @ move", it + 1, "→", checkpoint_path)

    # ── Final flush: promote the learner if it ended clearly ahead ──
    var final_rec = mz_candidate_winrate[
        ENV, REP, DYN, PRED, ARENA_GAMES, OBS, ACT, LATENT, BINS,
        NUM_SIMS, MAX_NODES, MAX_K, MAX_PLIES,
    ](
        ctx, l_rep, l_dyn, l_pred, rep, dyn, pred, Float64(gamma),
        seed=seed + 9991, open_plies=arena_open_plies,
    )
    if _mz_should_promote(
        final_rec, promote_threshold, min_decisive=ARENA_GAMES // 2
    ):
        hard_copy["gpu", M=REP](l_rep, rep, Optional(ctx))
        hard_copy["gpu", M=DYN](l_dyn, dyn, Optional(ctx))
        hard_copy["gpu", M=PRED](l_pred, pred, Optional(ctx))
        promotions += 1

    l_parts.unsafe_free(); h_diag_pred.unsafe_free()
    obs_h.unsafe_free(); pol_h.unsafe_free(); val_h.unsafe_free(); legal_h.unsafe_free()
    done_h.unsafe_free(); rew_h.unsafe_free(); act_h.unsafe_free()
    h_reana_legal.unsafe_free()
    # keep the learner + target nets alive past the adapters' borrowed pointers.
    _ = l_rep^
    _ = l_dyn^
    _ = l_pred^
    _ = rep_t^
    _ = dyn_t^
    _ = pred_t^
    return MZArenaRunResult(last_loss=last_loss, promotions=promotions)
