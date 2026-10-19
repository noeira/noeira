"""AlphaZero evaluation — greedy net-policy vs a random opponent (batched GPU).

`eval_policy_vs_random` plays `N_GAMES` complete games in parallel: on the
agent's plies it picks ``argmax`` of the net's policy head over legal moves
(pure policy, no MCTS — tests what the network learned); on the opponent's plies
it picks a uniform random legal move. Games run in lockstep for ``MAX_PLIES``
(no resets, so `current_player` stays synchronized across the batch); finished
games no-op on further steps. Returns the agent's (wins, draws, losses).

In TicTacToe, optimal play as P0 never loses, so a learning agent's loss-rate
vs random should fall toward zero — the convergence signal.

`eval_policy_vs_opponent` is the generalisation: the opponent is any
`GPUEvaluator` (e.g. `RandomOpponent`, `GPUMinimaxTicTacToe`), selecting its
moves in a batched GPU kernel. Against perfect minimax, a correct agent as P0
*never loses* (only draws or, vs a fallible opponent, wins), so loss-rate vs
minimax is the stronger validation signal.

Both helpers put the net in eval mode (`set_attr["training"](0)`) up front so
BatchNorm-bearing torsos (CNN / ResNet) use running stats — a no-op for the MLP.
"""

from noeira.nn.core.ptr import untracked
from std.memory import Pointer
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.nn.constants import DT
from noeira.nn.core.module import Module
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.call import call_forward
from noeira.core import TwoPlayerDiscreteEnv, Saveable
from noeira.core.env_traits import GPUTwoPlayerDiscreteEnv
from noeira.planners.tree_search import (
    GenericGPUMCTS,
    GenericCPUMCTS,
    AlphaGoPUCT,
    NoNoise,
    SelfPlay,
)
from ..zero.evaluators import GPUEvaluator, CPUEvaluator, RandomOpponent
from ..zero.mcts_adapters import AZPredGPU, AZEnvGPU
from ..zero.mcts_adapters_cpu import AZRepCPU, AZDynCPU, AZPredCPU
from noeira.core.tiles import tile


@always_inline
def _xs(s: UInt64) -> UInt64:
    var x = s
    x ^= x << 13
    x ^= x >> 7
    x ^= x << 17
    return x


@fieldwise_init
struct EvalResult(Copyable, Movable):
    var wins: Int
    var draws: Int
    var losses: Int


def eval_policy_vs_random[
    ENV: GPUTwoPlayerDiscreteEnv,
    NET: Module,
    N_GAMES: Int,
    RESULT_IDX: Int,
    MAX_PLIES: Int,
](
    ctx: DeviceContext,
    mut net: NET,
    agent_player: Int = 0,
    seed: UInt64 = 1,
) raises -> EvalResult:
    comptime OBS = NET.IN_DIMS[0]
    comptime ACT = NET.OUT_DIM - 1
    comptime W = NET.OUT_DIM
    comptime STATE = ENV.STATE_SIZE

    net.set_attr["training"](Scalar[DT](0.0))  # BN → eval (no-op for MLP)

    var states = ctx.enqueue_create_buffer[DT](N_GAMES * STATE)
    var obs_dev = ctx.enqueue_create_buffer[DT](N_GAMES * OBS)
    var legal_dev = ctx.enqueue_create_buffer[DT](N_GAMES * ACT)
    var pred_dev = ctx.enqueue_create_buffer[DT](N_GAMES * W)
    var rew = ctx.enqueue_create_buffer[DT](N_GAMES)
    var done = ctx.enqueue_create_buffer[DT](N_GAMES)
    var term = ctx.enqueue_create_buffer[DT](N_GAMES)
    var obs_next = ctx.enqueue_create_buffer[DT](N_GAMES * OBS)
    var legal_next = ctx.enqueue_create_buffer[DT](N_GAMES * ACT)
    var actions_dev = ctx.enqueue_create_buffer[DT](N_GAMES)

    var legal_h = ctx.enqueue_create_host_buffer[DT](N_GAMES * ACT)
    var pred_h = ctx.enqueue_create_host_buffer[DT](N_GAMES * W)
    var actions_h = ctx.enqueue_create_host_buffer[DT](N_GAMES)
    var states_h = ctx.enqueue_create_host_buffer[DT](N_GAMES * STATE)
    ctx.synchronize()

    ENV.reset_kernel_gpu[N_GAMES, STATE](ctx, states)
    ctx.synchronize()

    var pred_ad = AZPredGPU[OBS, ACT, NET].make(net)
    var rng = seed | 1

    for ply in range(MAX_PLIES):
        ENV.extract_obs_kernel_gpu[N_GAMES, STATE, OBS](
            ctx, states, obs_dev, legal_dev
        )
        ctx.enqueue_copy(legal_h, legal_dev)
        ctx.synchronize()

        var agent_turn = (ply % 2) == agent_player
        if agent_turn:
            var obs_lt = tile[N_GAMES, OBS](obs_dev)
            var pred_lt = tile[N_GAMES, W](pred_dev)
            pred_ad.predict_gpu[N_GAMES](ctx, obs_lt, pred_lt)
            ctx.enqueue_copy(pred_h, pred_dev)
            ctx.synchronize()
            for e in range(N_GAMES):
                var best = -1
                var bestv = Float64(-1e30)
                for a in range(ACT):
                    if legal_h[e * ACT + a] > 0.5:
                        var v = Float64(pred_h[e * W + a])
                        if v > bestv:
                            bestv = v
                            best = a
                actions_h[e] = Scalar[DT](best if best >= 0 else 0)
        else:
            for e in range(N_GAMES):
                var cnt = 0
                for a in range(ACT):
                    if legal_h[e * ACT + a] > 0.5:
                        cnt += 1
                if cnt == 0:
                    actions_h[e] = Scalar[DT](0)
                else:
                    rng = _xs(rng)
                    var pick = Int(rng % UInt64(cnt))
                    var chosen = 0
                    var seen = 0
                    for a in range(ACT):
                        if legal_h[e * ACT + a] > 0.5:
                            if seen == pick:
                                chosen = a
                                break
                            seen += 1
                    actions_h[e] = Scalar[DT](chosen)

        ctx.enqueue_copy(actions_dev, actions_h)
        ctx.synchronize()
        ENV.step_kernel_gpu[N_GAMES, STATE, OBS](
            ctx, states, actions_dev, rew, done, term, obs_next, legal_next
        )
        ctx.synchronize()

    ctx.enqueue_copy(states_h, states)
    ctx.synchronize()

    var agent_win = agent_player + 1
    var agent_loss = (1 - agent_player) + 1
    var wins = 0
    var draws = 0
    var losses = 0
    for e in range(N_GAMES):
        var r = Int(states_h[e * STATE + RESULT_IDX])
        if r == agent_win:
            wins += 1
        elif r == agent_loss:
            losses += 1
        else:
            draws += 1
    return EvalResult(wins=wins, draws=draws, losses=losses)


def eval_policy_vs_random_cpu[
    ENV: TwoPlayerDiscreteEnv & Saveable & Defaultable & Deinitable,
    NET: Module,
    N_GAMES: Int,
    MAX_PLIES: Int,
](mut net: NET, agent_player: Int = 0, seed: UInt64 = 1,) raises -> EvalResult:
    """CPU counterpart to `eval_policy_vs_random`: play `N_GAMES` games on a
    single CPU env where the agent picks ``argmax`` of its (CPU) policy head over
    legal moves and the opponent plays a uniform random legal move. Pure-policy
    (no MCTS) — tests what the network learned. Returns the agent's record."""
    comptime IN = NET.IN_DIMS[0]
    comptime ACT = NET.OUT_DIM - 1
    comptime OUT = NET.OUT_DIM

    net.set_attr["training"](Scalar[DT](0.0))

    var env = ENV()
    var obs_t = Tensor.alloc(IN)
    var pred_t = Tensor.alloc(OUT)

    var wins = 0
    var draws = 0
    var losses = 0
    var rng = seed | 1

    for _g in range(N_GAMES):
        _ = env.reset()
        for _ply in range(MAX_PLIES):
            if env.game_result() != 0:
                break
            var legal = env.legal_action_mask()
            var act = 0
            if env.current_player() == agent_player:
                var obs_raw = env.get_obs_list()
                for i in range(IN):
                    obs_t.data[i] = Scalar[DT](obs_raw[i]) if i < len(
                        obs_raw
                    ) else Scalar[DT](0.0)
                call_forward["cpu", 1](
                    net, TensorRefs[NET.ARITY](obs_t), pred_t, None
                )
                var best = -1
                var bestv = Float64(-1e30)
                for a in range(ACT):
                    if a < len(legal) and legal[a]:
                        var v = Float64(pred_t.data[a])
                        if v > bestv:
                            bestv = v
                            best = a
                act = best if best >= 0 else 0
            else:
                var cnt = 0
                for a in range(ACT):
                    if a < len(legal) and legal[a]:
                        cnt += 1
                if cnt > 0:
                    rng = _xs(rng)
                    var pick = Int(rng % UInt64(cnt))
                    var seen = 0
                    for a in range(ACT):
                        if a < len(legal) and legal[a]:
                            if seen == pick:
                                act = a
                                break
                            seen += 1
            _ = env.step(env.action_from_index(act))

        var gr = env.game_result()
        var agent_win = agent_player + 1
        var agent_loss = (1 - agent_player) + 1
        if gr == agent_win:
            wins += 1
        elif gr == agent_loss:
            losses += 1
        else:
            draws += 1

    return EvalResult(wins=wins, draws=draws, losses=losses)


def eval_mcts_vs_opponent_cpu[
    ENV: TwoPlayerDiscreteEnv & Saveable & Defaultable & Deinitable,
    NET: Module,
    OPP: CPUEvaluator,
    N_GAMES: Int,
    NUM_SIMS: Int,
    MAX_NODES: Int,
    MAX_PLIES: Int,
](mut net: NET, agent_player: Int = 0, seed: UInt64 = 1,) raises -> EvalResult:
    """CPU full-strength eval: the agent plays via `GenericCPUMCTS` (temp=0,
    NoNoise, argmax-visit), the opponent via its `CPUEvaluator`. The CPU twin of
    `eval_mcts_vs_opponent`; plays `N_GAMES` games sequentially on one env."""
    comptime OBS = NET.IN_DIMS[0]
    comptime ACT = NET.OUT_DIM - 1
    comptime LATENT = ENV.SAVE_SIZE
    comptime EVAL_MCTS = GenericCPUMCTS[
        ACT,
        LATENT,
        NUM_SIMS,
        MAX_NODES,
        AlphaGoPUCT[1.0],
        NoNoise,
        SelfPlay,
        NORMALIZE_Q=False,  # raw Q∈[-1,1] like legacy
    ]

    net.set_attr["training"](Scalar[DT](0.0))

    var env = ENV()
    var root_save = List[Scalar[DT]](length=LATENT, fill=0)
    var env_ptr = Pointer(to=env)
    var net_ptr = Pointer(to=net)
    var rep = AZRepCPU[ENV, OBS](env=untracked(env_ptr))
    var dyn = AZDynCPU[ENV, ACT](env=untracked(env_ptr))
    var pred = AZPredCPU[ENV, OBS, ACT, NET](
        env=untracked(env_ptr),
        net=untracked(net_ptr),
    )
    var mcts = EVAL_MCTS(gamma=1.0)

    var wins = 0
    var draws = 0
    var losses = 0
    var rng = seed | 1

    for _g in range(N_GAMES):
        _ = env.reset()
        var ply = 0
        while env.game_result() == 0 and ply < MAX_PLIES:
            var act: Int
            if env.current_player() == agent_player:
                env.save_env_state(root_save)
                var legal = env.legal_action_mask()
                var root_obs = List[Float64](length=OBS, fill=Float64(0.0))
                var policy = mcts.search[
                    AZRepCPU[ENV, OBS],
                    AZDynCPU[ENV, ACT],
                    AZPredCPU[ENV, OBS, ACT, NET],
                ](rep, dyn, pred, root_obs, add_noise=False, legal_mask=legal)
                env.load_env_state(root_save)
                var best = -1
                var bestv = Float64(-1.0)
                for a in range(ACT):
                    if a < len(legal) and legal[a] and policy[a] > bestv:
                        bestv = policy[a]
                        best = a
                act = best if best >= 0 else 0
            else:
                rng = _xs(rng)
                act = OPP.select_action_cpu[ENV](env, rng)
            _ = env.step(env.action_from_index(act))
            ply += 1

        var gr = env.game_result()
        var agent_win = agent_player + 1
        var agent_loss = (1 - agent_player) + 1
        if gr == agent_win:
            wins += 1
        elif gr == agent_loss:
            losses += 1
        else:
            draws += 1

    return EvalResult(wins=wins, draws=draws, losses=losses)


def eval_mcts_vs_opponent[
    ENV: GPUTwoPlayerDiscreteEnv,
    NET: Module,
    OPP: GPUEvaluator,
    N_GAMES: Int,
    NUM_SIMS: Int,
    MAX_NODES: Int,
    MAX_PLIES: Int,
](
    ctx: DeviceContext,
    mut net: NET,
    agent_player: Int = 0,
    seed: UInt64 = 1,
    open_plies: Int = 0,
) raises -> EvalResult:
    """Full-strength eval: the agent plays via **MCTS** (temp=0, no Dirichlet
    noise), the opponent via its `GPUEvaluator`. This mirrors the legacy
    `gpu_eval` — the agent's move is the argmax-visit action of a `NUM_SIMS`
    search on top of the net, *not* the raw policy-head argmax. The policy head
    alone cannot draw perfect minimax; MCTS on top can, so this is the eval that
    reflects the deployed agent's true strength.

    Games run in lockstep; as each finishes its result is locked (by reward +
    whose turn it was, color-agnostic) and the env is reset so the batch can run
    on for the slower games. `agent_player` (0/1) picks the agent's color.

    `open_plies = 0` (default): every game plays the canonical line — with a
    deterministic opponent (minimax) all `N_GAMES` collapse to ONE distinct
    game per color ("does optimal-from-start draw perfect play?"), so the
    result is quantized to 0 / N_GAMES and the winrate CURVE swings wildly
    between razor-edge lines. `open_plies > 0`: BOTH sides open with that many
    uniform-random LEGAL plies (per game), diversifying the batch into distinct
    positions so the aggregate is a real winrate (the arena's `open_plies`
    convention). Use 0 as a perfect-play gate, ≥2 for tracking curves."""
    comptime OBS = NET.IN_DIMS[0]
    comptime ACT = NET.OUT_DIM - 1
    comptime STATE = ENV.STATE_SIZE
    comptime EVAL_MCTS = GenericGPUMCTS[
        N_GAMES,
        ACT,
        OBS,
        1,
        MAX_NODES,
        NUM_SIMS,
        1,
        AlphaGoPUCT[1.0],
        NoNoise,
        SelfPlay,
        STATE_SIZE=STATE,
    ]
    # Resets let early-finishing games stop interfering; +ACT slack covers any
    # desync between fast and slow games before all first-games complete.
    comptime MAX_EVAL_MOVES = MAX_PLIES + ACT

    net.set_attr["training"](Scalar[DT](0.0))  # BN → eval (no-op for MLP)

    var mcts = EVAL_MCTS(ctx, gamma=1.0, v_min=-1.0, v_max=1.0)
    var states = ctx.enqueue_create_buffer[DT](N_GAMES * STATE)
    var obs_dev = ctx.enqueue_create_buffer[DT](N_GAMES * OBS)
    var legal_dev = ctx.enqueue_create_buffer[DT](N_GAMES * ACT)
    var rew = ctx.enqueue_create_buffer[DT](N_GAMES)
    var done = ctx.enqueue_create_buffer[DT](N_GAMES)
    var term = ctx.enqueue_create_buffer[DT](N_GAMES)
    var obs_next = ctx.enqueue_create_buffer[DT](N_GAMES * OBS)
    var legal_next = ctx.enqueue_create_buffer[DT](N_GAMES * ACT)
    var actions_dev = ctx.enqueue_create_buffer[DT](N_GAMES)

    var done_h = ctx.enqueue_create_host_buffer[DT](N_GAMES)
    var rew_h = ctx.enqueue_create_host_buffer[DT](N_GAMES)
    ctx.synchronize()

    ENV.reset_kernel_gpu[N_GAMES, STATE](ctx, states)
    ENV.extract_obs_kernel_gpu[N_GAMES, STATE, OBS](
        ctx, states, obs_dev, legal_dev
    )
    ctx.synchronize()

    var eval_done = Array[Bool, N_GAMES](fill=False)
    var eval_result = Array[Int, N_GAMES](fill=0)  # 1=win 2=loss 3=draw
    var all_done = False
    var move = 0

    while not all_done and move < MAX_EVAL_MOVES:
        var agent_turn = (move % 2) == agent_player
        if move < open_plies:
            # Opening diversity: BOTH sides play uniform-random legal moves
            # for the first `open_plies` plies, splitting the lockstep batch
            # into distinct games (see docstring).
            RandomOpponent.select_action_gpu[N_GAMES, ACT, STATE](
                ctx,
                actions_dev,
                legal_dev,
                states,
                seed + UInt64(move) * 131 + 17,
            )
        elif agent_turn:
            var pred = AZPredGPU[OBS, ACT, NET].make(net)
            var env_ad = AZEnvGPU[ENV, STATE, OBS, ACT]()
            var root_obs = tile[N_GAMES, OBS](obs_dev)
            var root_legal = tile[N_GAMES * ACT](legal_dev)
            mcts.search_gpu_alphazero[type_of(pred), type_of(env_ad)](
                ctx,
                pred,
                env_ad,
                root_obs,
                states,
                root_legal,
                rng_seed=seed + UInt64(move),
            )
            ctx.enqueue_copy(actions_dev, mcts.actions_out)
        else:
            OPP.select_action_gpu[N_GAMES, ACT, STATE](
                ctx,
                actions_dev,
                legal_dev,
                states,
                seed + UInt64(move) * 31 + 7,
            )
        ctx.synchronize()

        ENV.step_kernel_gpu[N_GAMES, STATE, OBS](
            ctx, states, actions_dev, rew, done, term, obs_next, legal_next
        )
        ctx.enqueue_copy(done_h, done)
        ctx.enqueue_copy(rew_h, rew)
        ctx.synchronize()

        all_done = True
        for e in range(N_GAMES):
            if not eval_done[e] and done_h[e] > 0.5:
                eval_done[e] = True
                var r = Float64(rew_h[e])
                # Reward accrues to the player who just moved.
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
            draws += 1  # 3 (draw) or 0 (never finished) → draw
    return EvalResult(wins=wins, draws=draws, losses=losses)


def eval_policy_vs_opponent[
    ENV: GPUTwoPlayerDiscreteEnv,
    NET: Module,
    OPP: GPUEvaluator,
    N_GAMES: Int,
    RESULT_IDX: Int,
    MAX_PLIES: Int,
](
    ctx: DeviceContext,
    mut net: NET,
    agent_player: Int = 0,
    seed: UInt64 = 1,
    open_plies: Int = 0,
) raises -> EvalResult:
    """Greedy net-policy vs an arbitrary `GPUEvaluator` opponent (batched GPU).

    Same lockstep structure as `eval_policy_vs_random`, but the opponent's move
    is chosen by `OPP.select_action_gpu` (reading the legal masks + raw state),
    so this works for minimax / perfect-play opponents too. Against perfect
    minimax a correct P0 agent never loses.

    `open_plies` > 0 makes the first `open_plies` plies *uniform random for both
    sides* (seeded distinctly per env), spreading the batch across diverse
    openings before greedy-agent-vs-opponent play resumes — essential when the
    opponent is deterministic (minimax), otherwise every game is the same line.
    Positions lost purely in the random opening penalise fresh and trained nets
    alike, so the *difference* in loss-rate still measures learning."""
    comptime OBS = NET.IN_DIMS[0]
    comptime ACT = NET.OUT_DIM - 1
    comptime W = NET.OUT_DIM
    comptime STATE = ENV.STATE_SIZE

    net.set_attr["training"](Scalar[DT](0.0))  # BN → eval (no-op for MLP)

    var states = ctx.enqueue_create_buffer[DT](N_GAMES * STATE)
    var obs_dev = ctx.enqueue_create_buffer[DT](N_GAMES * OBS)
    var legal_dev = ctx.enqueue_create_buffer[DT](N_GAMES * ACT)
    var pred_dev = ctx.enqueue_create_buffer[DT](N_GAMES * W)
    var rew = ctx.enqueue_create_buffer[DT](N_GAMES)
    var done = ctx.enqueue_create_buffer[DT](N_GAMES)
    var term = ctx.enqueue_create_buffer[DT](N_GAMES)
    var obs_next = ctx.enqueue_create_buffer[DT](N_GAMES * OBS)
    var legal_next = ctx.enqueue_create_buffer[DT](N_GAMES * ACT)
    var actions_dev = ctx.enqueue_create_buffer[DT](N_GAMES)

    var legal_h = ctx.enqueue_create_host_buffer[DT](N_GAMES * ACT)
    var pred_h = ctx.enqueue_create_host_buffer[DT](N_GAMES * W)
    var actions_h = ctx.enqueue_create_host_buffer[DT](N_GAMES)
    var states_h = ctx.enqueue_create_host_buffer[DT](N_GAMES * STATE)
    ctx.synchronize()

    ENV.reset_kernel_gpu[N_GAMES, STATE](ctx, states)
    ctx.synchronize()

    var pred_ad = AZPredGPU[OBS, ACT, NET].make(net)
    var rng = seed | 1

    for ply in range(MAX_PLIES):
        ENV.extract_obs_kernel_gpu[N_GAMES, STATE, OBS](
            ctx, states, obs_dev, legal_dev
        )
        ctx.synchronize()

        var agent_turn = (ply % 2) == agent_player
        if ply < open_plies:
            # Random opening for BOTH sides → diverse positions across the batch.
            rng = _xs(rng)
            RandomOpponent.select_action_gpu[N_GAMES, ACT, STATE](
                ctx, actions_dev, legal_dev, states, rng
            )
            ctx.synchronize()
        elif agent_turn:
            # Greedy net policy: argmax over legal columns (CPU readback).
            ctx.enqueue_copy(legal_h, legal_dev)
            ctx.synchronize()
            var obs_lt = tile[N_GAMES, OBS](obs_dev)
            var pred_lt = tile[N_GAMES, W](pred_dev)
            pred_ad.predict_gpu[N_GAMES](ctx, obs_lt, pred_lt)
            ctx.enqueue_copy(pred_h, pred_dev)
            ctx.synchronize()
            for e in range(N_GAMES):
                var best = -1
                var bestv = Float64(-1e30)
                for a in range(ACT):
                    if Float64(legal_h[e * ACT + a]) > 0.5:
                        var v = Float64(pred_h[e * W + a])
                        if v > bestv:
                            bestv = v
                            best = a
                actions_h[e] = Scalar[DT](best if best >= 0 else 0)
            ctx.enqueue_copy(actions_dev, actions_h)
            ctx.synchronize()
        else:
            # Opponent picks its move on-device from legal masks + raw state.
            rng = _xs(rng)
            OPP.select_action_gpu[N_GAMES, ACT, STATE](
                ctx, actions_dev, legal_dev, states, rng
            )
            ctx.synchronize()

        ENV.step_kernel_gpu[N_GAMES, STATE, OBS](
            ctx, states, actions_dev, rew, done, term, obs_next, legal_next
        )
        ctx.synchronize()

    ctx.enqueue_copy(states_h, states)
    ctx.synchronize()

    var agent_win = agent_player + 1
    var agent_loss = (1 - agent_player) + 1
    var wins = 0
    var draws = 0
    var losses = 0
    for e in range(N_GAMES):
        var r = Int(Float64(states_h[e * STATE + RESULT_IDX]))
        if r == agent_win:
            wins += 1
        elif r == agent_loss:
            losses += 1
        else:
            draws += 1
    return EvalResult(wins=wins, draws=draws, losses=losses)
