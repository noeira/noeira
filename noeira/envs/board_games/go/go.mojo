"""Go — CPU+GPU environment for two-player self-play RL training.

Parameterized board size (9, 13, 19). Tromp-Taylor rules: area scoring,
simple ko, no suicide. Komi 7.5 for white (player 1).

State layout (STATE_SIZE = SIZE*SIZE + 5):
  [0..N²-1]  board cells (0=empty, 1=black/P0, 2=white/P1)
  [N²]       current_player (0=black, 1=white)
  [N²+1]     ko_point (-1=none, else board index)
  [N²+2]     consecutive_passes (0, 1, or 2)
  [N²+3]     captures_p0 (black's captures)
  [N²+4]     step_count

Canonical obs (OBS_DIM = 4 * SIZE * SIZE):
  Plane 0: my stones
  Plane 1: opponent stones
  Plane 2: legal moves
  Plane 3: all ones (color indicator for canonical symmetry)

Actions: 0..N²-1 = intersection index, N² = pass.
"""

from std.random import random_float64
from std.memory import dealloc, alloc
from noeira.core.tiles import RowMajor, Tile, ImmTile, tile
from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext, DeviceBuffer
from noeira.core import (
    State,
    Action,
    BoxDiscreteActionEnv,
    TwoPlayerDiscreteEnv,
    GPUTwoPlayerDiscreteEnv,
    RenderableEnv,
)
from noeira.render import Renderer2D, SDL_Color
from ..core.board_env import BoardGameState, BoardGameAction, board_dtype

# Game result codes
comptime RESULT_ONGOING: Int = 0
comptime RESULT_P0_WINS: Int = 1
comptime RESULT_P1_WINS: Int = 2
comptime RESULT_DRAW: Int = 3

comptime KOMI: Float64 = 7.5


# ============================================================================
# GoEnv
# ============================================================================


struct GoEnv[SIZE: Int, DTYPE: DType = DType.float64](
    TwoPlayerDiscreteEnv & GPUTwoPlayerDiscreteEnv & RenderableEnv
):
    """Go environment with parameterized board size.

    Parameters:
        SIZE: Board size (9, 13, or 19).
        DTYPE: Data type for CPU operations.
    """

    # Derived constants
    comptime BOARD_SIZE: Int = Self.SIZE * Self.SIZE
    comptime PASS_ACTION: Int = Self.BOARD_SIZE

    # State indices
    comptime S_CURRENT_PLAYER: Int = Self.BOARD_SIZE
    comptime S_KO_POINT: Int = Self.BOARD_SIZE + 1
    comptime S_PASSES: Int = Self.BOARD_SIZE + 2
    comptime S_CAPTURES_P0: Int = Self.BOARD_SIZE + 3
    comptime S_STEP_COUNT: Int = Self.BOARD_SIZE + 4

    # Trait conformance
    comptime dtype = Self.DTYPE
    comptime StateType = BoardGameState
    comptime ActionType = BoardGameAction

    # GPUTwoPlayerDiscreteEnv constants
    comptime STATE_SIZE: Int = Self.BOARD_SIZE + 5
    comptime OBS_DIM: Int = 4 * Self.BOARD_SIZE
    comptime NUM_ACTIONS: Int = Self.BOARD_SIZE + 1  # + pass

    # CPU state
    var state: List[Scalar[Self.dtype]]
    var done: Bool
    # Temp buffer for flood-fill (avoid repeated allocation)
    var _visited: List[Bool]

    # Renderer
    var _renderer: Optional[Pointer[Renderer2D, MutUntrackedOrigin]]
    var _renderer_initialized: Bool

    def __init__(out self):
        self.state = List[Scalar[Self.dtype]](capacity=Self.STATE_SIZE)
        for _ in range(Self.STATE_SIZE):
            self.state.append(Scalar[Self.dtype](0.0))
        self.state[Self.S_KO_POINT] = -1.0
        self.done = False
        self._visited = List[Bool](capacity=Self.BOARD_SIZE)
        for _ in range(Self.BOARD_SIZE):
            self._visited.append(False)
        self._renderer = None
        self._renderer_initialized = False

    # ========================================================================
    # CPU: reset + step
    # ========================================================================

    def reset(mut self) -> BoardGameState:
        for i in range(Self.STATE_SIZE):
            self.state[i] = 0.0
        self.state[Self.S_KO_POINT] = -1.0
        self.done = False
        return BoardGameState(index=0)

    def step(
        mut self, action: BoardGameAction, verbose: Bool = False
    ) -> Tuple[BoardGameState, Scalar[Self.dtype], Bool]:
        var result = self._step_impl(action.value)
        return (
            BoardGameState(index=Int(self.state[Self.S_STEP_COUNT])),
            result[0],
            result[1],
        )

    def _step_impl(mut self, action: Int) -> Tuple[Scalar[Self.dtype], Bool]:
        """Execute one move. Returns (reward, done)."""
        if self.done:
            return (Scalar[Self.dtype](0.0), True)

        var player = Int(self.state[Self.S_CURRENT_PLAYER])
        var mark = Scalar[Self.dtype](player + 1)
        var opp_mark = Scalar[Self.dtype](2 - player)

        # Pass action
        if action == Self.PASS_ACTION:
            self.state[Self.S_PASSES] += 1.0
            self.state[Self.S_KO_POINT] = -1.0
            self.state[Self.S_STEP_COUNT] += 1.0

            # Two consecutive passes → game over
            if Int(self.state[Self.S_PASSES]) >= 2:
                self.done = True
                var score = self._score_area()
                if score > 0.0:
                    self.state[Self.BOARD_SIZE] = Scalar[Self.dtype](RESULT_P0_WINS)  # stored in unused slot
                elif score < 0.0:
                    self.state[Self.BOARD_SIZE] = Scalar[Self.dtype](RESULT_P1_WINS)
                else:
                    self.state[Self.BOARD_SIZE] = Scalar[Self.dtype](RESULT_DRAW)
                # Reward from perspective of player who passed
                if score > 0.0:
                    # Black won — if current player is black, they win
                    if player == 0:
                        return (Scalar[Self.dtype](1.0), True)
                    else:
                        return (Scalar[Self.dtype](-1.0), True)
                elif score < 0.0:
                    if player == 1:
                        return (Scalar[Self.dtype](1.0), True)
                    else:
                        return (Scalar[Self.dtype](-1.0), True)
                else:
                    return (Scalar[Self.dtype](0.0), True)

            # Switch player
            self.state[Self.S_CURRENT_PLAYER] = Scalar[Self.dtype](1 - player)
            return (Scalar[Self.dtype](0.0), False)

        # Validate intersection
        if action < 0 or action >= Self.BOARD_SIZE:
            return (Scalar[Self.dtype](-1.0), False)
        if self.state[action] != 0.0:
            return (Scalar[Self.dtype](-1.0), False)
        if action == Int(self.state[Self.S_KO_POINT]):
            return (Scalar[Self.dtype](-1.0), False)

        # Tentatively place stone
        self.state[action] = mark

        # Capture opponent groups with 0 liberties adjacent to placed stone
        var captured = 0
        var capture_point = -1
        var row = action // Self.SIZE
        var col = action % Self.SIZE
        # Check 4 neighbors
        if row > 0:
            var nb = action - Self.SIZE
            if self.state[nb] == opp_mark:
                var libs = self._count_liberties(nb)
                if libs == 0:
                    var cap = self._remove_group(nb)
                    captured += cap
                    if cap == 1:
                        capture_point = nb
        if row < Self.SIZE - 1:
            var nb = action + Self.SIZE
            if self.state[nb] == opp_mark:
                var libs = self._count_liberties(nb)
                if libs == 0:
                    var cap = self._remove_group(nb)
                    captured += cap
                    if cap == 1:
                        capture_point = nb
        if col > 0:
            var nb = action - 1
            if self.state[nb] == opp_mark:
                var libs = self._count_liberties(nb)
                if libs == 0:
                    var cap = self._remove_group(nb)
                    captured += cap
                    if cap == 1:
                        capture_point = nb
        if col < Self.SIZE - 1:
            var nb = action + 1
            if self.state[nb] == opp_mark:
                var libs = self._count_liberties(nb)
                if libs == 0:
                    var cap = self._remove_group(nb)
                    captured += cap
                    if cap == 1:
                        capture_point = nb

        # Suicide check: if own group has 0 liberties after captures
        var own_libs = self._count_liberties(action)
        if own_libs == 0:
            # Illegal — undo
            self.state[action] = 0.0
            return (Scalar[Self.dtype](-1.0), False)

        # Update captures
        if player == 0:
            self.state[Self.S_CAPTURES_P0] += Scalar[Self.dtype](captured)

        # Ko detection: single stone captured, and the capturing stone
        # would be recaptured if the opponent replayed at capture_point
        if captured == 1:
            self.state[Self.S_KO_POINT] = Scalar[Self.dtype](capture_point)
        else:
            self.state[Self.S_KO_POINT] = -1.0

        # Reset consecutive passes
        self.state[Self.S_PASSES] = 0.0
        self.state[Self.S_STEP_COUNT] += 1.0

        # Switch player
        self.state[Self.S_CURRENT_PLAYER] = Scalar[Self.dtype](1 - player)
        return (Scalar[Self.dtype](0.0), False)

    # ========================================================================
    # Liberty computation (flood-fill)
    # ========================================================================

    def _count_liberties(mut self, start: Int) -> Int:
        """Count liberties of the group containing the stone at `start`."""
        var color = self.state[start]
        if Float64(color) == 0.0:
            return 0

        # Clear visited
        for i in range(Self.BOARD_SIZE):
            self._visited[i] = False

        var liberty_count = 0
        var stack = List[Int](capacity=Self.BOARD_SIZE)
        stack.append(start)
        self._visited[start] = True

        while len(stack) > 0:
            var pos = stack.pop()
            var r = pos // Self.SIZE
            var c = pos % Self.SIZE

            # Check 4 neighbors
            var neighbors = List[Int](capacity=4)
            if r > 0:
                neighbors.append(pos - Self.SIZE)
            if r < Self.SIZE - 1:
                neighbors.append(pos + Self.SIZE)
            if c > 0:
                neighbors.append(pos - 1)
            if c < Self.SIZE - 1:
                neighbors.append(pos + 1)

            for n_idx in range(len(neighbors)):
                var nb = neighbors[n_idx]
                if self._visited[nb]:
                    continue
                if self.state[nb] == color:
                    self._visited[nb] = True
                    stack.append(nb)
                elif Float64(self.state[nb]) == 0.0:
                    self._visited[nb] = True
                    liberty_count += 1

        return liberty_count

    def _remove_group(mut self, start: Int) -> Int:
        """Remove the group containing the stone at `start`. Returns count removed."""
        var color = self.state[start]
        if Float64(color) == 0.0:
            return 0

        for i in range(Self.BOARD_SIZE):
            self._visited[i] = False

        var count = 0
        var stack = List[Int](capacity=Self.BOARD_SIZE)
        stack.append(start)
        self._visited[start] = True

        while len(stack) > 0:
            var pos = stack.pop()
            self.state[pos] = 0.0
            count += 1

            var r = pos // Self.SIZE
            var c = pos % Self.SIZE

            if r > 0:
                var nb = pos - Self.SIZE
                if not self._visited[nb] and self.state[nb] == color:
                    self._visited[nb] = True
                    stack.append(nb)
            if r < Self.SIZE - 1:
                var nb = pos + Self.SIZE
                if not self._visited[nb] and self.state[nb] == color:
                    self._visited[nb] = True
                    stack.append(nb)
            if c > 0:
                var nb = pos - 1
                if not self._visited[nb] and self.state[nb] == color:
                    self._visited[nb] = True
                    stack.append(nb)
            if c < Self.SIZE - 1:
                var nb = pos + 1
                if not self._visited[nb] and self.state[nb] == color:
                    self._visited[nb] = True
                    stack.append(nb)

        return count

    # ========================================================================
    # Scoring (Tromp-Taylor area scoring)
    # ========================================================================

    def _score_area(mut self) -> Float64:
        """Compute area score. Returns black_score - white_score - komi.

        Positive = black wins, negative = white wins.
        """
        var black_area = 0
        var white_area = 0

        # Count stones
        for i in range(Self.BOARD_SIZE):
            if Float64(self.state[i]) == 1.0:
                black_area += 1
            elif Float64(self.state[i]) == 2.0:
                white_area += 1

        # Count empty points enclosed by one color
        for i in range(Self.BOARD_SIZE):
            if Float64(self.state[i]) == 0.0:
                var territory_owner = self._territory_owner(i)
                if territory_owner == 1:
                    black_area += 1
                elif territory_owner == 2:
                    white_area += 1

        return Float64(black_area) - Float64(white_area) - KOMI

    def _territory_owner(mut self, start: Int) -> Int:
        """Determine which color (if any) encloses an empty region.

        Returns 1 (black), 2 (white), or 0 (neutral/both border).
        """
        for i in range(Self.BOARD_SIZE):
            self._visited[i] = False

        var borders_black = False
        var borders_white = False

        var stack = List[Int](capacity=Self.BOARD_SIZE)
        stack.append(start)
        self._visited[start] = True

        while len(stack) > 0:
            var pos = stack.pop()
            var r = pos // Self.SIZE
            var c = pos % Self.SIZE

            var neighbors = List[Int](capacity=4)
            if r > 0:
                neighbors.append(pos - Self.SIZE)
            if r < Self.SIZE - 1:
                neighbors.append(pos + Self.SIZE)
            if c > 0:
                neighbors.append(pos - 1)
            if c < Self.SIZE - 1:
                neighbors.append(pos + 1)

            for n_idx in range(len(neighbors)):
                var nb = neighbors[n_idx]
                if self._visited[nb]:
                    continue
                if Float64(self.state[nb]) == 1.0:
                    borders_black = True
                elif Float64(self.state[nb]) == 2.0:
                    borders_white = True
                else:
                    self._visited[nb] = True
                    stack.append(nb)

        if borders_black and not borders_white:
            return 1
        elif borders_white and not borders_black:
            return 2
        return 0

    # ========================================================================
    # Legal move check
    # ========================================================================

    def _is_legal(mut self, pos: Int, player: Int) -> Bool:
        """Check if placing at `pos` is legal for `player`."""
        if pos < 0 or pos >= Self.BOARD_SIZE:
            return False
        if Float64(self.state[pos]) != 0.0:
            return False
        if pos == Int(self.state[Self.S_KO_POINT]):
            return False

        var mark = Scalar[Self.dtype](player + 1)
        var opp_mark = Scalar[Self.dtype](2 - player)

        # Tentatively place
        self.state[pos] = mark

        # Check if any adjacent opponent group would be captured
        var would_capture = False
        var r = pos // Self.SIZE
        var c = pos % Self.SIZE
        if r > 0 and self.state[pos - Self.SIZE] == opp_mark:
            if self._count_liberties(pos - Self.SIZE) == 0:
                would_capture = True
        if not would_capture and r < Self.SIZE - 1 and self.state[pos + Self.SIZE] == opp_mark:
            if self._count_liberties(pos + Self.SIZE) == 0:
                would_capture = True
        if not would_capture and c > 0 and self.state[pos - 1] == opp_mark:
            if self._count_liberties(pos - 1) == 0:
                would_capture = True
        if not would_capture and c < Self.SIZE - 1 and self.state[pos + 1] == opp_mark:
            if self._count_liberties(pos + 1) == 0:
                would_capture = True

        # If no capture, check own liberties (suicide check)
        if not would_capture:
            var own_libs = self._count_liberties(pos)
            if own_libs == 0:
                self.state[pos] = 0.0  # undo
                return False

        self.state[pos] = 0.0  # undo
        return True

    # ========================================================================
    # Env trait methods
    # ========================================================================

    def get_state(mut self) -> BoardGameState:
        return BoardGameState(index=Int(self.state[Self.S_STEP_COUNT]))

    def close(mut self):
        if self._renderer_initialized:
            self._renderer.value()[].close()
            self._renderer.value().unsafe_free()
            self._renderer_initialized = False

    def action_from_index(self, action_idx: Int) -> BoardGameAction:
        return BoardGameAction(value=action_idx)

    def num_actions(self) -> Int:
        return Self.NUM_ACTIONS

    def obs_dim(self) -> Int:
        return Self.OBS_DIM

    def num_states(self) -> Int:
        return 1

    def state_to_index(self, state: BoardGameState) -> Int:
        return state.index

    # ========================================================================
    # TwoPlayerDiscreteEnv trait methods
    # ========================================================================

    def current_player(self) -> Int:
        return Int(self.state[Self.S_CURRENT_PLAYER])

    def legal_action_mask(self) -> List[Bool]:
        var mask = List[Bool](capacity=Self.NUM_ACTIONS)
        if self.done:
            for _ in range(Self.NUM_ACTIONS):
                mask.append(False)
            return mask^

        var ko = Int(self.state[Self.S_KO_POINT])
        # Simplified legality: empty intersection + not ko point.
        # Suicide is checked and rejected with -1 reward in step().
        for i in range(Self.BOARD_SIZE):
            if Float64(self.state[i]) == 0.0 and i != ko:
                mask.append(True)
            else:
                mask.append(False)
        # Pass is always legal
        mask.append(True)
        return mask^

    def game_result(self) -> Int:
        if not self.done:
            return RESULT_ONGOING
        # Result was stored during scoring
        var score_val = Int(self.state[Self.BOARD_SIZE])
        return score_val

    # ========================================================================
    # ContinuousStateEnv / BoxDiscreteActionEnv (CPU)
    # ========================================================================

    def get_obs_list(self) -> List[Scalar[Self.dtype]]:
        var obs = List[Scalar[Self.dtype]](capacity=Self.OBS_DIM)
        var player = Int(self.state[Self.S_CURRENT_PLAYER])
        var my_mark = Scalar[Self.dtype](player + 1)
        var opp_mark = Scalar[Self.dtype](2 - player)

        # Plane 0: my stones
        for i in range(Self.BOARD_SIZE):
            if self.state[i] == my_mark:
                obs.append(Scalar[Self.dtype](1.0))
            else:
                obs.append(Scalar[Self.dtype](0.0))

        # Plane 1: opponent stones
        for i in range(Self.BOARD_SIZE):
            if self.state[i] == opp_mark:
                obs.append(Scalar[Self.dtype](1.0))
            else:
                obs.append(Scalar[Self.dtype](0.0))

        # Plane 2: legal moves
        var mask = self.legal_action_mask()
        for i in range(Self.BOARD_SIZE):
            if mask[i]:
                obs.append(Scalar[Self.dtype](1.0))
            else:
                obs.append(Scalar[Self.dtype](0.0))

        # Plane 3: all ones (color indicator)
        for _ in range(Self.BOARD_SIZE):
            obs.append(Scalar[Self.dtype](1.0))

        return obs^

    def reset_obs_list(mut self) -> List[Scalar[Self.dtype]]:
        _ = self.reset()
        return self.get_obs_list()

    def step_obs(
        mut self, action: Int
    ) -> Tuple[List[Scalar[Self.dtype]], Scalar[Self.dtype], Bool]:
        """Single-agent step with random opponent."""
        var result = self._step_impl(action)
        var reward = result[0]
        var done = result[1]

        if done:
            return (self.get_obs_list(), reward, done)

        # Random opponent
        var mask = self.legal_action_mask()
        var legal_moves = List[Int](capacity=Self.NUM_ACTIONS)
        for i in range(Self.NUM_ACTIONS):
            if mask[i]:
                legal_moves.append(i)

        if len(legal_moves) > 0:
            var opp_idx = Int(random_float64() * Float64(len(legal_moves)))
            if opp_idx >= len(legal_moves):
                opp_idx = len(legal_moves) - 1
            var opp_result = self._step_impl(legal_moves[opp_idx])
            done = opp_result[1]
            if done:
                var gr = self.game_result()
                if gr == RESULT_P0_WINS:
                    reward = Scalar[Self.dtype](1.0)
                elif gr == RESULT_P1_WINS:
                    reward = Scalar[Self.dtype](-1.0)

        return (self.get_obs_list(), reward, done)

    # ========================================================================
    # RenderableEnv trait methods
    # ========================================================================

    def init_renderer(mut self) raises -> Bool:
        if self._renderer_initialized:
            return True
        var cell_size = 50
        var margin = 30
        var win_w = 2 * margin + (Self.SIZE - 1) * cell_size
        var win_h = 2 * margin + (Self.SIZE - 1) * cell_size + 50
        self._renderer = alloc[Renderer2D]({count = 1}).unsafe_leak()
        self._renderer.value().unsafe_write(
            Renderer2D(width=win_w, height=win_h, fps=30, title="Go " + String(Self.SIZE) + "x" + String(Self.SIZE))
        )
        self._renderer_initialized = True
        return True

    def render_frame(mut self) raises -> None:
        if not self._renderer_initialized:
            return
        self._render(self._renderer.value()[])

    def _render(self, mut renderer: Renderer2D):
        """Render Go board state using SDL3."""
        var board_color = SDL_Color(r=0xDE, g=0xB8, b=0x87, a=0xFF)  # tan/wooden
        if not renderer.begin_frame_with_color(board_color):
            return

        var line_color = SDL_Color(r=0x00, g=0x00, b=0x00, a=0xFF)
        var black_stone = SDL_Color(r=0x10, g=0x10, b=0x10, a=0xFF)
        var white_stone_color = SDL_Color(r=0xF0, g=0xF0, b=0xF0, a=0xFF)
        var black_outline = SDL_Color(r=0xFF, g=0xFF, b=0xFF, a=0xFF)
        var black_outline2 = SDL_Color(r=0x00, g=0x00, b=0x00, a=0xFF)
        var hoshi_color = SDL_Color(r=0x00, g=0x00, b=0x00, a=0xFF)
        var text_color = SDL_Color(r=0xFF, g=0xFF, b=0xFF, a=0xFF)
        var status_bg = SDL_Color(r=0x33, g=0x33, b=0x33, a=0xFF)

        var margin = 30
        var cell_size = 50
        var board_end = Self.SIZE - 1

        # Draw grid lines
        for i in range(Self.SIZE):
            var y = margin + i * cell_size
            renderer.draw_line(
                margin, y, margin + board_end * cell_size, y, line_color, 1
            )
            var x = margin + i * cell_size
            renderer.draw_line(
                x, margin, x, margin + board_end * cell_size, line_color, 1
            )

        # Draw star points (hoshi) for 9x9 board
        comptime
        if Self.SIZE == 9:
            def _hoshi_r(i: Int) -> Int:
                if i == 0:
                    return 2
                if i == 1:
                    return 2
                if i == 2:
                    return 6
                if i == 3:
                    return 6
                return 4

            def _hoshi_c(i: Int) -> Int:
                if i == 0:
                    return 2
                if i == 1:
                    return 6
                if i == 2:
                    return 2
                if i == 3:
                    return 6
                return 4

            for i in range(5):
                var hx = margin + _hoshi_c(i) * cell_size
                var hy = margin + _hoshi_r(i) * cell_size
                renderer.draw_circle(hx, hy, 4, hoshi_color, filled=True)

        # Draw stones
        var stone_r = 20
        for row in range(Self.SIZE):
            for col in range(Self.SIZE):
                var cell_idx = row * Self.SIZE + col
                var cell_val = Int(self.state[cell_idx])
                var cx = margin + col * cell_size
                var cy = margin + row * cell_size

                if cell_val == 1:
                    # Black stone: filled black with white outline
                    renderer.draw_circle(cx, cy, stone_r + 1, black_outline, filled=True)
                    renderer.draw_circle(cx, cy, stone_r, black_stone, filled=True)
                elif cell_val == 2:
                    # White stone: filled white with black outline
                    renderer.draw_circle(cx, cy, stone_r + 1, black_outline2, filled=True)
                    renderer.draw_circle(cx, cy, stone_r, white_stone_color, filled=True)

        # Status bar at bottom
        var win_w = 2 * margin + board_end * cell_size
        var status_y = 2 * margin + board_end * cell_size
        renderer.draw_rect(0, status_y, win_w, 50, status_bg)

        var game_result = self.game_result()
        if game_result == RESULT_ONGOING:
            var player = self.current_player()
            if player == 0:
                renderer.draw_text("Black's turn", win_w // 2 - 40, status_y + 20, text_color)
            else:
                renderer.draw_text("White's turn", win_w // 2 - 40, status_y + 20, text_color)
        elif game_result == RESULT_P0_WINS:
            renderer.draw_text("Black Wins!", win_w // 2 - 40, status_y + 20, text_color)
        elif game_result == RESULT_P1_WINS:
            renderer.draw_text("White Wins!", win_w // 2 - 40, status_y + 20, text_color)
        else:
            renderer.draw_text("Draw!", win_w // 2 - 20, status_y + 20, text_color)

        renderer.flip()

    def close_renderer(mut self) raises -> None:
        if not self._renderer_initialized:
            return
        self._renderer.value()[].close()
        self._renderer.value().unsafe_free()
        self._renderer_initialized = False

    def is_renderer_open(self) -> Bool:
        if not self._renderer_initialized:
            return False
        return not self._renderer.value()[].get_should_quit()

    def check_renderer_quit(mut self) -> Bool:
        if not self._renderer_initialized:
            return False
        return self._renderer.value()[].get_should_quit()

    def renderer_delay(self, ms: Int) -> None:
        if not self._renderer_initialized:
            return
        self._renderer.value()[].renderer_delay(ms)

    def renderer_is_paused(self) -> Bool:
        return False

    def renderer_step_once(self) -> Bool:
        return False

    # ========================================================================
    # GPU: Inline step/reset kernels
    # ========================================================================

    comptime TPB = 256

    @staticmethod
    @always_inline
    def _gpu_count_liberties[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        visited_origin: Origin[mut=True],
    ](
        states: Tile[board_dtype, BATCH_SIZE, STATE_SIZE],
        game: Int,
        start: Int,
        visited: Pointer[Bool, visited_origin],
    ) -> Int:
        """Count liberties of group at `start`. Uses visited buffer on stack.
        Bounded iteration for GPU safety (max BOARD_SIZE iterations)."""
        comptime BS = GoEnv[Self.SIZE].BOARD_SIZE
        var color = states[game, start]

        # Clear visited
        for j in range(BS):
            visited[unsafe_offset=j] = False

        # Simple iterative BFS using fixed-size stack on registers
        # For GPU, we unroll with bounded iteration
        var liberty_count = 0
        var stack_data_alloc = alloc[Int]({count = BS})
        var stack_data = stack_data_alloc.unsafe_ptr()
        var stack_top: Int
        stack_data[unsafe_offset=0] = start
        stack_top = 1
        visited[unsafe_offset=start] = True

        for _ in range(BS):  # bounded iteration
            if stack_top <= 0:
                break
            stack_top -= 1
            var pos = stack_data[unsafe_offset=stack_top]
            var r = pos // Self.SIZE
            var c = pos % Self.SIZE

            # Check 4 neighbors inline
            if r > 0:
                var nb = pos - Self.SIZE
                if not visited[unsafe_offset=nb]:
                    if states[game, nb] == color:
                        visited[unsafe_offset=nb] = True
                        stack_data[unsafe_offset=stack_top] = nb
                        stack_top += 1
                    elif states[game, nb] == 0.0:
                        visited[unsafe_offset=nb] = True
                        liberty_count += 1
            if r < Self.SIZE - 1:
                var nb = pos + Self.SIZE
                if not visited[unsafe_offset=nb]:
                    if states[game, nb] == color:
                        visited[unsafe_offset=nb] = True
                        stack_data[unsafe_offset=stack_top] = nb
                        stack_top += 1
                    elif states[game, nb] == 0.0:
                        visited[unsafe_offset=nb] = True
                        liberty_count += 1
            if c > 0:
                var nb = pos - 1
                if not visited[unsafe_offset=nb]:
                    if states[game, nb] == color:
                        visited[unsafe_offset=nb] = True
                        stack_data[unsafe_offset=stack_top] = nb
                        stack_top += 1
                    elif states[game, nb] == 0.0:
                        visited[unsafe_offset=nb] = True
                        liberty_count += 1
            if c < Self.SIZE - 1:
                var nb = pos + 1
                if not visited[unsafe_offset=nb]:
                    if states[game, nb] == color:
                        visited[unsafe_offset=nb] = True
                        stack_data[unsafe_offset=stack_top] = nb
                        stack_top += 1
                    elif states[game, nb] == 0.0:
                        visited[unsafe_offset=nb] = True
                        liberty_count += 1

        dealloc(stack_data_alloc^)
        return liberty_count

    @staticmethod
    @always_inline
    def _gpu_remove_group[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        visited_origin: Origin[mut=True],
    ](
        states: Tile[board_dtype, BATCH_SIZE, STATE_SIZE],
        game: Int,
        start: Int,
        visited: Pointer[Bool, visited_origin],
    ) -> Int:
        """Remove group at `start`. Returns count removed."""
        comptime BS = GoEnv[Self.SIZE].BOARD_SIZE
        var color = states[game, start]

        for j in range(BS):
            visited[unsafe_offset=j] = False

        var count = 0
        var stack_data_alloc = alloc[Int]({count = BS})
        var stack_data = stack_data_alloc.unsafe_ptr()
        var stack_top: Int
        stack_data[unsafe_offset=0] = start
        stack_top = 1
        visited[unsafe_offset=start] = True

        for _ in range(BS):
            if stack_top <= 0:
                break
            stack_top -= 1
            var pos = stack_data[unsafe_offset=stack_top]
            states[game, pos] = 0.0
            count += 1

            var r = pos // Self.SIZE
            var c = pos % Self.SIZE

            if r > 0:
                var nb = pos - Self.SIZE
                if not visited[unsafe_offset=nb] and states[game, nb] == color:
                    visited[unsafe_offset=nb] = True
                    stack_data[unsafe_offset=stack_top] = nb
                    stack_top += 1
            if r < Self.SIZE - 1:
                var nb = pos + Self.SIZE
                if not visited[unsafe_offset=nb] and states[game, nb] == color:
                    visited[unsafe_offset=nb] = True
                    stack_data[unsafe_offset=stack_top] = nb
                    stack_top += 1
            if c > 0:
                var nb = pos - 1
                if not visited[unsafe_offset=nb] and states[game, nb] == color:
                    visited[unsafe_offset=nb] = True
                    stack_data[unsafe_offset=stack_top] = nb
                    stack_top += 1
            if c < Self.SIZE - 1:
                var nb = pos + 1
                if not visited[unsafe_offset=nb] and states[game, nb] == color:
                    visited[unsafe_offset=nb] = True
                    stack_data[unsafe_offset=stack_top] = nb
                    stack_top += 1

        dealloc(stack_data_alloc^)
        return count

    @staticmethod
    @always_inline
    def step_kernel[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        states: Tile[board_dtype, BATCH_SIZE, STATE_SIZE],
        actions: ImmTile[board_dtype, BATCH_SIZE],
        rewards: Tile[board_dtype, BATCH_SIZE],
        dones: Tile[board_dtype, BATCH_SIZE],
    ):
        """Per-thread Go step kernel."""
        var i = Int(block_dim.x * block_idx.x + thread_idx.x)
        if i >= BATCH_SIZE:
            return

        comptime BS = GoEnv[Self.SIZE].BOARD_SIZE
        comptime S_CP = GoEnv[Self.SIZE].S_CURRENT_PLAYER
        comptime S_KO = GoEnv[Self.SIZE].S_KO_POINT
        comptime S_PA = GoEnv[Self.SIZE].S_PASSES
        comptime S_CAP = GoEnv[Self.SIZE].S_CAPTURES_P0
        comptime S_SC = GoEnv[Self.SIZE].S_STEP_COUNT
        comptime PASS_ACT = GoEnv[Self.SIZE].PASS_ACTION

        var action = Int(actions[i])
        var player = Int(states[i, S_CP])
        var mark = Scalar[board_dtype](player + 1)
        var opp_mark = Scalar[board_dtype](2 - player)

        # Allocate visited buffer
        var visited_alloc = alloc[Bool]({count = BS})
        var visited = visited_alloc.unsafe_ptr()

        # Pass action
        if action == PASS_ACT:
            states[i, S_PA] = states[i, S_PA] + 1.0
            states[i, S_KO] = -1.0
            states[i, S_SC] = states[i, S_SC] + 1.0

            if Int(states[i, S_PA]) >= 2:
                # Game over — simplified scoring for GPU
                # Count stones + territory
                var black_area = 0
                var white_area = 0
                for j in range(BS):
                    if states[i, j] == 1.0:
                        black_area += 1
                    elif states[i, j] == 2.0:
                        white_area += 1
                    # Skip territory for GPU (too expensive for now)

                var score = Scalar[board_dtype](black_area) - Scalar[board_dtype](white_area) - 7.5
                if player == 0:
                    if score > 0:
                        rewards[i] = 1.0
                    elif score < 0:
                        rewards[i] = -1.0
                    else:
                        rewards[i] = 0.0
                else:
                    if score < 0:
                        rewards[i] = 1.0
                    elif score > 0:
                        rewards[i] = -1.0
                    else:
                        rewards[i] = 0.0
                dones[i] = 1.0
            else:
                states[i, S_CP] = Scalar[board_dtype](1 - player)
                rewards[i] = 0.0
                dones[i] = 0.0

            dealloc(visited_alloc^)
            return

        # Validate
        if action < 0 or action >= BS or states[i, action] != 0.0 or action == Int(states[i, S_KO]):
            rewards[i] = -1.0
            dones[i] = 0.0
            dealloc(visited_alloc^)
            return

        # Place stone
        states[i, action] = mark

        # Capture opponent groups
        var captured = 0
        var capture_point = -1
        var r = action // Self.SIZE
        var c = action % Self.SIZE

        if r > 0:
            var nb = action - Self.SIZE
            if states[i, nb] == opp_mark:
                if GoEnv[Self.SIZE]._gpu_count_liberties[BATCH_SIZE, STATE_SIZE](states, i, nb, visited) == 0:
                    var cap = GoEnv[Self.SIZE]._gpu_remove_group[BATCH_SIZE, STATE_SIZE](states, i, nb, visited)
                    captured += cap
                    if cap == 1:
                        capture_point = nb
        if r < Self.SIZE - 1:
            var nb = action + Self.SIZE
            if states[i, nb] == opp_mark:
                if GoEnv[Self.SIZE]._gpu_count_liberties[BATCH_SIZE, STATE_SIZE](states, i, nb, visited) == 0:
                    var cap = GoEnv[Self.SIZE]._gpu_remove_group[BATCH_SIZE, STATE_SIZE](states, i, nb, visited)
                    captured += cap
                    if cap == 1:
                        capture_point = nb
        if c > 0:
            var nb = action - 1
            if states[i, nb] == opp_mark:
                if GoEnv[Self.SIZE]._gpu_count_liberties[BATCH_SIZE, STATE_SIZE](states, i, nb, visited) == 0:
                    var cap = GoEnv[Self.SIZE]._gpu_remove_group[BATCH_SIZE, STATE_SIZE](states, i, nb, visited)
                    captured += cap
                    if cap == 1:
                        capture_point = nb
        if c < Self.SIZE - 1:
            var nb = action + 1
            if states[i, nb] == opp_mark:
                if GoEnv[Self.SIZE]._gpu_count_liberties[BATCH_SIZE, STATE_SIZE](states, i, nb, visited) == 0:
                    var cap = GoEnv[Self.SIZE]._gpu_remove_group[BATCH_SIZE, STATE_SIZE](states, i, nb, visited)
                    captured += cap
                    if cap == 1:
                        capture_point = nb

        # Suicide check
        if GoEnv[Self.SIZE]._gpu_count_liberties[BATCH_SIZE, STATE_SIZE](states, i, action, visited) == 0:
            states[i, action] = 0.0  # undo
            rewards[i] = -1.0
            dones[i] = 0.0
            dealloc(visited_alloc^)
            return

        # Ko point
        if captured == 1:
            states[i, S_KO] = Scalar[board_dtype](capture_point)
        else:
            states[i, S_KO] = -1.0

        states[i, S_PA] = 0.0
        states[i, S_SC] = states[i, S_SC] + 1.0
        states[i, S_CP] = Scalar[board_dtype](1 - player)
        rewards[i] = 0.0
        dones[i] = 0.0

        dealloc(visited_alloc^)

    @staticmethod
    @always_inline
    def reset_kernel[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        states: Tile[board_dtype, BATCH_SIZE, STATE_SIZE],
    ):
        var i = Int(block_dim.x * block_idx.x + thread_idx.x)
        if i >= BATCH_SIZE:
            return
        for c in range(STATE_SIZE):
            states[i, c] = 0.0
        states[i, GoEnv[Self.SIZE].S_KO_POINT] = -1.0

    @staticmethod
    @always_inline
    def selective_reset_kernel[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        states: Tile[board_dtype, BATCH_SIZE, STATE_SIZE],
        dones: Tile[board_dtype, BATCH_SIZE],
    ):
        var i = Int(block_dim.x * block_idx.x + thread_idx.x)
        if i >= BATCH_SIZE:
            return
        if dones[i] > 0.5:
            for c in range(STATE_SIZE):
                states[i, c] = 0.0
            states[i, GoEnv[Self.SIZE].S_KO_POINT] = -1.0
            dones[i] = 0.0

    @staticmethod
    @always_inline
    def extract_obs_and_masks[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        OBS_DIM: Int,
        NUM_ACTIONS: Int,
    ](
        states: ImmTile[board_dtype, BATCH_SIZE, STATE_SIZE],
        obs: Tile[board_dtype, BATCH_SIZE, OBS_DIM],
        legal_masks: Tile[board_dtype, BATCH_SIZE, NUM_ACTIONS],
    ):
        var i = Int(block_dim.x * block_idx.x + thread_idx.x)
        if i >= BATCH_SIZE:
            return

        comptime BS = GoEnv[Self.SIZE].BOARD_SIZE
        comptime S_CP = GoEnv[Self.SIZE].S_CURRENT_PLAYER
        comptime S_GR = GoEnv[Self.SIZE].S_PASSES  # Use passes >= 2 as game over indicator

        var player = Int(states[i, S_CP])
        var my_mark = Scalar[board_dtype](player + 1)
        var opp_mark = Scalar[board_dtype](2 - player)

        for c in range(BS):
            var cell = states[i, c]
            # Plane 0: my stones
            if cell == my_mark:
                obs[i, c] = 1.0
            else:
                obs[i, c] = 0.0
            # Plane 1: opp stones
            if cell == opp_mark:
                obs[i, BS + c] = 1.0
            else:
                obs[i, BS + c] = 0.0
            # Plane 2: legal moves (simplified: empty + not ko)
            var ko = Int(states[i, GoEnv[Self.SIZE].S_KO_POINT])
            if cell == 0.0 and c != ko:
                obs[i, 2 * BS + c] = 1.0
                legal_masks[i, c] = 1.0
            else:
                obs[i, 2 * BS + c] = 0.0
                legal_masks[i, c] = 0.0
            # Plane 3: all ones
            obs[i, 3 * BS + c] = 1.0

        # Pass is always legal
        legal_masks[i, BS] = 1.0

    # ========================================================================
    # GPU Launcher Methods
    # ========================================================================

    @staticmethod
    def step_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        OBS_DIM: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[board_dtype],
        actions_buf: DeviceBuffer[board_dtype],
        mut rewards_buf: DeviceBuffer[board_dtype],
        mut dones_buf: DeviceBuffer[board_dtype],
        mut terminated_buf: DeviceBuffer[board_dtype],
        mut obs_buf: DeviceBuffer[board_dtype],
        mut legal_masks_buf: DeviceBuffer[board_dtype],
        rng_seed: UInt64 = 0,
        rng_counter_ptr: Optional[Pointer[Scalar[DType.uint64], MutAnyOrigin]] = None,
    ) raises:
        var states = tile[BATCH_SIZE, STATE_SIZE](states_buf)
        var actions = tile[BATCH_SIZE](actions_buf)
        var rewards = tile[BATCH_SIZE](rewards_buf)
        var dones = tile[BATCH_SIZE](dones_buf)
        var terminated_out = tile[BATCH_SIZE](terminated_buf)
        var obs = tile[BATCH_SIZE, OBS_DIM](obs_buf)
        var legal_masks = tile[BATCH_SIZE, GoEnv[Self.SIZE].NUM_ACTIONS](legal_masks_buf)

        comptime BLOCKS = (BATCH_SIZE + Self.TPB - 1) // Self.TPB

        @always_inline
        def step_wrapper(
            states: Tile[board_dtype, BATCH_SIZE, STATE_SIZE],
            actions: ImmTile[board_dtype, BATCH_SIZE],
            rewards: Tile[board_dtype, BATCH_SIZE],
            dones: Tile[board_dtype, BATCH_SIZE],
            terminated_out: Tile[board_dtype, BATCH_SIZE],
            obs: Tile[board_dtype, BATCH_SIZE, OBS_DIM],
            legal_masks: Tile[board_dtype, BATCH_SIZE, GoEnv[Self.SIZE].NUM_ACTIONS],
        ):
            GoEnv[Self.SIZE].step_kernel[BATCH_SIZE, STATE_SIZE](states, actions, rewards, dones)
            var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
            if idx < BATCH_SIZE:
                terminated_out[idx] = dones[idx]

            var states_read = tile[BATCH_SIZE, STATE_SIZE](rebind[Pointer[Scalar[board_dtype], ImmutAnyOrigin]](states.ptr))
            GoEnv[Self.SIZE].extract_obs_and_masks[BATCH_SIZE, STATE_SIZE, OBS_DIM, GoEnv[Self.SIZE].NUM_ACTIONS](
                states_read, obs, legal_masks
            )

        ctx.enqueue_function[step_wrapper](
            states, actions, rewards, dones, terminated_out, obs, legal_masks,
            grid_dim=(BLOCKS,), block_dim=(Self.TPB,),
        )

    @staticmethod
    def reset_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[board_dtype],
        rng_seed: UInt64 = 0,
    ) raises:
        var states = tile[BATCH_SIZE, STATE_SIZE](states_buf)
        comptime BLOCKS = (BATCH_SIZE + Self.TPB - 1) // Self.TPB

        @always_inline
        def reset_wrapper(
            states: Tile[board_dtype, BATCH_SIZE, STATE_SIZE],
        ):
            GoEnv[Self.SIZE].reset_kernel[BATCH_SIZE, STATE_SIZE](states)

        ctx.enqueue_function[reset_wrapper](
            states, grid_dim=(BLOCKS,), block_dim=(Self.TPB,),
        )

    @staticmethod
    def selective_reset_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[board_dtype],
        mut dones_buf: DeviceBuffer[board_dtype],
        rng_seed: UInt64,
        rng_counter_ptr: Optional[Pointer[Scalar[DType.uint64], MutAnyOrigin]] = None,
    ) raises:
        var states = tile[BATCH_SIZE, STATE_SIZE](states_buf)
        var dones = tile[BATCH_SIZE](dones_buf)
        comptime BLOCKS = (BATCH_SIZE + Self.TPB - 1) // Self.TPB

        @always_inline
        def sel_reset_wrapper(
            states: Tile[board_dtype, BATCH_SIZE, STATE_SIZE],
            dones: Tile[board_dtype, BATCH_SIZE],
        ):
            GoEnv[Self.SIZE].selective_reset_kernel[BATCH_SIZE, STATE_SIZE](states, dones)

        ctx.enqueue_function[sel_reset_wrapper](
            states, dones, grid_dim=(BLOCKS,), block_dim=(Self.TPB,),
        )

    @staticmethod
    def extract_obs_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        OBS_DIM: Int,
    ](
        ctx: DeviceContext,
        states_buf: DeviceBuffer[board_dtype],
        mut obs_buf: DeviceBuffer[board_dtype],
        mut legal_masks_buf: DeviceBuffer[board_dtype],
    ) raises:
        var states = tile[BATCH_SIZE, STATE_SIZE](states_buf)
        var obs = tile[BATCH_SIZE, OBS_DIM](obs_buf)
        var legal_masks = tile[BATCH_SIZE, GoEnv[Self.SIZE].NUM_ACTIONS](legal_masks_buf)
        comptime BLOCKS = (BATCH_SIZE + Self.TPB - 1) // Self.TPB

        @always_inline
        def extract_wrapper(
            states: ImmTile[board_dtype, BATCH_SIZE, STATE_SIZE],
            obs: Tile[board_dtype, BATCH_SIZE, OBS_DIM],
            legal_masks: Tile[board_dtype, BATCH_SIZE, GoEnv[Self.SIZE].NUM_ACTIONS],
        ):
            GoEnv[Self.SIZE].extract_obs_and_masks[BATCH_SIZE, STATE_SIZE, OBS_DIM, GoEnv[Self.SIZE].NUM_ACTIONS](
                states, obs, legal_masks
            )

        ctx.enqueue_function[extract_wrapper](
            states, obs, legal_masks, grid_dim=(BLOCKS,), block_dim=(Self.TPB,),
        )
