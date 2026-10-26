"""GPU opponent evaluators for measuring zero-series agent strength.

A `GPUEvaluator` selects an action for every environment in one batched GPU
kernel launch, given the legal-action masks and the raw env state buffer. This
is the opponent side of the eval harness (`eval_policy_vs_opponent`): the agent
plays its (greedy) net policy, the opponent plays its strategy, and we read off
win/draw/loss from the terminal game result.

Two opponents, mirroring the legacy MuZero eval surface:
  * `RandomOpponent`        — uniform random legal move (weakest baseline).
  * `GPUMinimaxTicTacToe`   — perfect play (full-depth iterative minimax). A
                              correct agent as P0 *never loses* against it.
  * `GPUMinimaxConnectFour` — depth-limited alpha-beta (Connect4 isn't solved
                              at full depth on a per-thread budget).

State-layout contract (matches the board envs): cells at `state[0 : R*C]`
(0=empty, 1=P0, 2=P1), current player at `state[R*C]`. TicTacToe board is
row-major 3×3; Connect4 board is column-major 7×6 (`col*ROWS + row`, row 0 =
bottom), matching `tic_tac_toe.mojo` / `connect_four.mojo`.

Ported from `deep_agents/muzero/evaluators.mojo` onto `nn` (`DT`).
"""

from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.nn.constants import DT
from noeira.core import TwoPlayerDiscreteEnv, Saveable
from noeira.core.tiles import Tile, tile


# ═══════════════════════════════════════════════════════════════════════════
# GPU Evaluator trait
# ═══════════════════════════════════════════════════════════════════════════


trait GPUEvaluator(RegisterPassable):
    """Batched GPU opponent — one kernel selects actions for all envs.

    Has access to the raw state buffer so state-based strategies (minimax)
    can reconstruct the board; stateless strategies (random) ignore it.
    """

    comptime NAME: String

    @staticmethod
    def select_action_gpu[
        N_ENVS: Int, ACT: Int, STATE_SIZE: Int
    ](
        ctx: DeviceContext,
        actions_out: DeviceBuffer[DT],
        legal_masks: DeviceBuffer[DT],
        game_states: DeviceBuffer[DT],
        rng_seed: UInt64,
    ) raises:
        """Write a chosen action into `actions_out[e]` for each env `e`."""
        ...


# ═══════════════════════════════════════════════════════════════════════════
# CPU Evaluator trait
# ═══════════════════════════════════════════════════════════════════════════


trait CPUEvaluator:
    """Single-env CPU opponent — selects one action for the env's current
    player, reading the live env state (board / legal mask) directly. The CPU
    twin of `GPUEvaluator`; the shipped evaluators conform to both so a single
    `train_arena[OPP]` routes on `TARGET`."""

    comptime NAME: String

    @staticmethod
    def select_action_cpu[
        E: TwoPlayerDiscreteEnv & Saveable
    ](mut env: E, rng_seed: UInt64) raises -> Int:
        """Return the chosen action for `env.current_player()` at the env's
        current state. `rng_seed` is used by stochastic strategies (random)."""
        ...


# ═══════════════════════════════════════════════════════════════════════════
# Random opponent
# ═══════════════════════════════════════════════════════════════════════════


def _random_legal_kernel[
    N_ENVS: Int, ACT: Int
](
    actions_out: Tile[DT, N_ENVS],
    legal_masks: Tile[DT, N_ENVS * ACT],
    seed: UInt64,
):
    """Pick a uniform random legal action per env (xorshift on env id+seed)."""
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N_ENVS:
        return
    var off = e * ACT
    var n_legal = 0
    for a in range(ACT):
        if rebind[Scalar[DT]](legal_masks[off + a]) > Scalar[DT](0.5):
            n_legal += 1
    if n_legal == 0:
        actions_out[e] = Scalar[DT](0)
        return
    var x = seed ^ (UInt64(e + 1) * 0x9E3779B97F4A7C15)
    x ^= x << 13
    x ^= x >> 7
    x ^= x << 17
    var pick = Int(x % UInt64(n_legal))
    var seen = 0
    for a in range(ACT):
        if rebind[Scalar[DT]](legal_masks[off + a]) > Scalar[DT](0.5):
            if seen == pick:
                actions_out[e] = Scalar[DT](a)
                return
            seen += 1
    actions_out[e] = Scalar[DT](0)


struct RandomOpponent(GPUEvaluator & CPUEvaluator):
    """Uniform random legal action — the weakest baseline."""

    comptime NAME: String = "Random"

    @staticmethod
    def select_action_cpu[
        E: TwoPlayerDiscreteEnv & Saveable
    ](mut env: E, rng_seed: UInt64) raises -> Int:
        var legal = env.legal_action_mask()
        var n = 0
        for a in range(len(legal)):
            if legal[a]:
                n += 1
        if n == 0:
            return 0
        var x = rng_seed | 1
        x ^= x << 13
        x ^= x >> 7
        x ^= x << 17
        var pick = Int(x % UInt64(n))
        var seen = 0
        for a in range(len(legal)):
            if legal[a]:
                if seen == pick:
                    return a
                seen += 1
        return 0

    @staticmethod
    def select_action_gpu[
        N_ENVS: Int, ACT: Int, STATE_SIZE: Int
    ](
        ctx: DeviceContext,
        actions_out: DeviceBuffer[DT],
        legal_masks: DeviceBuffer[DT],
        game_states: DeviceBuffer[DT],
        rng_seed: UInt64,
    ) raises:
        _ = game_states
        comptime TPB = 256
        comptime ENV_BLOCKS = (N_ENVS + TPB - 1) // TPB
        comptime run = _random_legal_kernel[N_ENVS, ACT]
        var act_t = rebind[
            Tile[DT, N_ENVS]
        ](tile[N_ENVS](actions_out))
        var lm_t = rebind[
            Tile[DT, N_ENVS * ACT]
        ](tile[N_ENVS * ACT](legal_masks))
        ctx.enqueue_function[run](
            act_t,
            lm_t,
            rng_seed,
            grid_dim=(ENV_BLOCKS,),
            block_dim=(TPB,),
        )


# ═══════════════════════════════════════════════════════════════════════════
# TicTacToe minimax (perfect play) — iterative, no recursion (GPU-safe stack)
# ═══════════════════════════════════════════════════════════════════════════


def _ttt_winner(board: Array[Int, 9]) -> Int:
    """0=none, else winning mark (1 or 2)."""
    for r in range(3):
        var i = r * 3
        if (
            board[i] != 0
            and board[i] == board[i + 1]
            and board[i + 1] == board[i + 2]
        ):
            return board[i]
    for c in range(3):
        if (
            board[c] != 0
            and board[c] == board[c + 3]
            and board[c + 3] == board[c + 6]
        ):
            return board[c]
    if board[0] != 0 and board[0] == board[4] and board[4] == board[8]:
        return board[0]
    if board[2] != 0 and board[2] == board[4] and board[4] == board[6]:
        return board[2]
    return 0


def _ttt_minimax_iter(mut board: Array[Int, 9], my_mark: Int) -> Int:
    """Iterative minimax (explicit stack, max depth 9) from a position where
    it is the opponent's turn after our root move. Returns +1/0/-1 from `my`
    perspective."""
    comptime MAX_DEPTH = 10
    var stk_action = Array[Int, MAX_DEPTH](fill=0)
    var stk_best = Array[Int, MAX_DEPTH](fill=0)
    var stk_is_max = Array[Int, MAX_DEPTH](fill=0)
    var stk_mark = Array[Int, MAX_DEPTH](fill=0)
    var stk_placed = Array[Int, MAX_DEPTH](fill=-1)

    var opp_mark = 3 - my_mark
    var depth = 0
    stk_action[0] = 0
    stk_best[0] = 2  # opponent minimizes
    stk_is_max[0] = 0
    stk_mark[0] = opp_mark
    stk_placed[0] = -1

    while depth >= 0:
        var winner = _ttt_winner(board)
        if winner != 0:
            var val = 1 if winner == my_mark else -1
            if depth == 0:
                return val
            board[stk_placed[depth]] = 0
            depth -= 1
            if stk_is_max[depth] != 0:
                if val > stk_best[depth]:
                    stk_best[depth] = val
            else:
                if val < stk_best[depth]:
                    stk_best[depth] = val
            stk_action[depth] += 1
            continue

        var has_empty = False
        for i in range(9):
            if board[i] == 0:
                has_empty = True
                break
        if not has_empty:
            if depth == 0:
                return 0
            board[stk_placed[depth]] = 0
            depth -= 1
            if stk_is_max[depth] != 0:
                if 0 > stk_best[depth]:
                    stk_best[depth] = 0
            else:
                if 0 < stk_best[depth]:
                    stk_best[depth] = 0
            stk_action[depth] += 1
            continue

        var found = False
        while stk_action[depth] < 9:
            var a = stk_action[depth]
            if board[a] == 0:
                board[a] = stk_mark[depth]
                found = True
                depth += 1
                stk_placed[depth] = a
                stk_action[depth] = 0
                if stk_is_max[depth - 1] != 0:
                    stk_best[depth] = 2
                    stk_is_max[depth] = 0
                else:
                    stk_best[depth] = -2
                    stk_is_max[depth] = 1
                stk_mark[depth] = 3 - stk_mark[depth - 1]
                break
            stk_action[depth] += 1

        if not found:
            var val = stk_best[depth]
            if depth == 0:
                return val
            board[stk_placed[depth]] = 0
            depth -= 1
            if stk_is_max[depth] != 0:
                if val > stk_best[depth]:
                    stk_best[depth] = val
            else:
                if val < stk_best[depth]:
                    stk_best[depth] = val
            stk_action[depth] += 1

    return 0


def _ttt_minimax_kernel[
    N_ENVS: Int, STATE_SIZE: Int
](
    actions_out: Tile[DT, N_ENVS],
    game_states: Tile[DT, N_ENVS * STATE_SIZE],
):
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N_ENVS:
        return
    var s_off = e * STATE_SIZE
    var board = Array[Int, 9](fill=0)
    for i in range(9):
        board[i] = Int(rebind[Scalar[DT]](game_states[s_off + i]))
    var player = Int(rebind[Scalar[DT]](game_states[s_off + 9]))
    var my_mark = player + 1

    var best_action = -1
    var best_score = -2
    for a in range(9):
        if board[a] != 0:
            continue
        board[a] = my_mark
        var score = _ttt_minimax_iter(board, my_mark)
        board[a] = 0
        if score > best_score:
            best_score = score
            best_action = a
    if best_action < 0:
        best_action = 0
    actions_out[e] = Scalar[DT](best_action)


struct GPUMinimaxTicTacToe(GPUEvaluator & CPUEvaluator):
    """Full-depth perfect minimax for TicTacToe (one search per env thread)."""

    comptime NAME: String = "Minimax"

    @staticmethod
    def select_action_cpu[
        E: TwoPlayerDiscreteEnv & Saveable
    ](mut env: E, rng_seed: UInt64) raises -> Int:
        _ = rng_seed
        var buf = List[Scalar[DT]](length=E.SAVE_SIZE, fill=0)
        env.save_env_state(buf)
        var board = Array[Int, 9](fill=0)
        for i in range(9):
            board[i] = Int(buf[i])
        var player = Int(buf[9])
        var my_mark = player + 1
        var best_action = -1
        var best_score = -2
        for a in range(9):
            if board[a] != 0:
                continue
            board[a] = my_mark
            var score = _ttt_minimax_iter(board, my_mark)
            board[a] = 0
            if score > best_score:
                best_score = score
                best_action = a
        return best_action if best_action >= 0 else 0

    @staticmethod
    def select_action_gpu[
        N_ENVS: Int, ACT: Int, STATE_SIZE: Int
    ](
        ctx: DeviceContext,
        actions_out: DeviceBuffer[DT],
        legal_masks: DeviceBuffer[DT],
        game_states: DeviceBuffer[DT],
        rng_seed: UInt64,
    ) raises:
        _ = legal_masks
        _ = rng_seed
        comptime TPB = 128
        comptime ENV_BLOCKS = (N_ENVS + TPB - 1) // TPB
        comptime run = _ttt_minimax_kernel[N_ENVS, STATE_SIZE]
        var act_t = rebind[
            Tile[DT, N_ENVS]
        ](tile[N_ENVS](actions_out))
        var gs_t = rebind[
            Tile[DT, N_ENVS * STATE_SIZE]
        ](tile[N_ENVS * STATE_SIZE](game_states))
        ctx.enqueue_function[run](
            act_t,
            gs_t,
            grid_dim=(ENV_BLOCKS,),
            block_dim=(TPB,),
        )


# ═══════════════════════════════════════════════════════════════════════════
# ConnectFour minimax (depth-limited alpha-beta)
# ═══════════════════════════════════════════════════════════════════════════


def _c4_count_dir(
    board: Array[Int, 42], col: Int, row: Int, mark: Int, dc: Int, dr: Int
) -> Int:
    comptime ROWS = 6
    comptime COLS = 7
    var cnt = 0
    var c = col + dc
    var r = row + dr
    while (
        c >= 0
        and c < COLS
        and r >= 0
        and r < ROWS
        and board[c * ROWS + r] == mark
    ):
        cnt += 1
        c += dc
        r += dr
    return cnt


def _c4_check_win(
    board: Array[Int, 42], col: Int, row: Int, mark: Int
) -> Bool:
    return (
        _c4_count_dir(board, col, row, mark, 1, 0)
        + _c4_count_dir(board, col, row, mark, -1, 0)
        >= 3
        or _c4_count_dir(board, col, row, mark, 0, 1)
        + _c4_count_dir(board, col, row, mark, 0, -1)
        >= 3
        or _c4_count_dir(board, col, row, mark, 1, 1)
        + _c4_count_dir(board, col, row, mark, -1, -1)
        >= 3
        or _c4_count_dir(board, col, row, mark, 1, -1)
        + _c4_count_dir(board, col, row, mark, -1, 1)
        >= 3
    )


def _c4_find_row(board: Array[Int, 42], col: Int) -> Int:
    comptime ROWS = 6
    for r in range(ROWS):
        if board[col * ROWS + r] == 0:
            return r
    return -1


def _c4_minimax_ab(
    mut board: Array[Int, 42],
    depth: Int,
    alpha_in: Int,
    beta_in: Int,
    is_max: Int,
    max_mark: Int,
    min_mark: Int,
) -> Int:
    comptime ROWS = 6
    comptime COLS = 7
    var alpha = alpha_in
    var beta = beta_in
    if is_max != 0:
        var best = -200
        for col in range(COLS):
            var row = _c4_find_row(board, col)
            if row < 0:
                continue
            board[col * ROWS + row] = max_mark
            if _c4_check_win(board, col, row, max_mark):
                board[col * ROWS + row] = 0
                return 100 + depth
            if depth <= 1:
                board[col * ROWS + row] = 0
                if 0 > best:
                    best = 0
                continue
            var val = _c4_minimax_ab(
                board, depth - 1, alpha, beta, 0, max_mark, min_mark
            )
            board[col * ROWS + row] = 0
            if val > best:
                best = val
            if val > alpha:
                alpha = val
            if alpha >= beta:
                break
        return best
    else:
        var best = 200
        for col in range(COLS):
            var row = _c4_find_row(board, col)
            if row < 0:
                continue
            board[col * ROWS + row] = min_mark
            if _c4_check_win(board, col, row, min_mark):
                board[col * ROWS + row] = 0
                return -(100 + depth)
            if depth <= 1:
                board[col * ROWS + row] = 0
                if 0 < best:
                    best = 0
                continue
            var val = _c4_minimax_ab(
                board, depth - 1, alpha, beta, 1, max_mark, min_mark
            )
            board[col * ROWS + row] = 0
            if val < best:
                best = val
            if val < beta:
                beta = val
            if alpha >= beta:
                break
        return best


def _c4_minimax_kernel[
    N_ENVS: Int, STATE_SIZE: Int, DEPTH: Int
](
    actions_out: Tile[DT, N_ENVS],
    game_states: Tile[DT, N_ENVS * STATE_SIZE],
):
    comptime ROWS = 6
    comptime COLS = 7
    var e = Int(block_dim.x * block_idx.x + thread_idx.x)
    if e >= N_ENVS:
        return
    var s_off = e * STATE_SIZE
    var board = Array[Int, 42](fill=0)
    for i in range(42):
        board[i] = Int(rebind[Scalar[DT]](game_states[s_off + i]))
    var player = Int(rebind[Scalar[DT]](game_states[s_off + 42]))
    var my_mark = player + 1
    var opp_mark = 2 - player

    var best_score = -300
    var best_action = -1
    for col in range(COLS):
        var row = _c4_find_row(board, col)
        if row < 0:
            continue
        board[col * ROWS + row] = my_mark
        var score: Int
        if _c4_check_win(board, col, row, my_mark):
            score = 200
        else:
            score = _c4_minimax_ab(
                board, DEPTH - 1, -300, 300, 0, my_mark, opp_mark
            )
        board[col * ROWS + row] = 0
        if score > best_score or best_action < 0:
            best_score = score
            best_action = col
    if best_action < 0:
        best_action = 0
    actions_out[e] = Scalar[DT](best_action)


struct GPUMinimaxConnectFour[DEPTH: Int = 5](GPUEvaluator & CPUEvaluator):
    """Depth-limited alpha-beta minimax for ConnectFour (default 5-ply)."""

    comptime NAME: String = "Minimax-D5"

    @staticmethod
    def select_action_cpu[
        E: TwoPlayerDiscreteEnv & Saveable
    ](mut env: E, rng_seed: UInt64) raises -> Int:
        _ = rng_seed
        comptime ROWS = 6
        comptime COLS = 7
        var buf = List[Scalar[DT]](length=E.SAVE_SIZE, fill=0)
        env.save_env_state(buf)
        var board = Array[Int, 42](fill=0)
        for i in range(42):
            board[i] = Int(buf[i])
        var player = Int(buf[42])
        var my_mark = player + 1
        var opp_mark = 2 - player
        var best_score = -300
        var best_action = -1
        for col in range(COLS):
            var row = _c4_find_row(board, col)
            if row < 0:
                continue
            board[col * ROWS + row] = my_mark
            var score: Int
            if _c4_check_win(board, col, row, my_mark):
                score = 200
            else:
                score = _c4_minimax_ab(
                    board, Self.DEPTH - 1, -300, 300, 0, my_mark, opp_mark
                )
            board[col * ROWS + row] = 0
            if score > best_score or best_action < 0:
                best_score = score
                best_action = col
        return best_action if best_action >= 0 else 0

    @staticmethod
    def select_action_gpu[
        N_ENVS: Int, ACT: Int, STATE_SIZE: Int
    ](
        ctx: DeviceContext,
        actions_out: DeviceBuffer[DT],
        legal_masks: DeviceBuffer[DT],
        game_states: DeviceBuffer[DT],
        rng_seed: UInt64,
    ) raises:
        _ = legal_masks
        _ = rng_seed
        comptime TPB = 32
        comptime ENV_BLOCKS = (N_ENVS + TPB - 1) // TPB
        comptime run = _c4_minimax_kernel[N_ENVS, STATE_SIZE, Self.DEPTH]
        var act_t = rebind[
            Tile[DT, N_ENVS]
        ](tile[N_ENVS](actions_out))
        var gs_t = rebind[
            Tile[DT, N_ENVS * STATE_SIZE]
        ](tile[N_ENVS * STATE_SIZE](game_states))
        ctx.enqueue_function[run](
            act_t,
            gs_t,
            grid_dim=(ENV_BLOCKS,),
            block_dim=(TPB,),
        )
