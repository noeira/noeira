"""ConnectFour — CPU+GPU environment for two-player self-play RL training.

7 columns × 6 rows. Pieces drop to lowest empty row. First to connect 4 wins.

State layout (STATE_SIZE = 46):
  [0..41]  board cells (column-major: col*6+row, 0=empty, 1=P0, 2=P1)
           row 0 = bottom, row 5 = top
  [42]     current_player (0 or 1)
  [43]     game_result (0=ongoing, 1=P0 wins, 2=P1 wins, 3=draw)
  [44]     step_count
  [45]     last_col (last column played, for rendering)

Canonical obs (OBS_DIM = 126 = 3 planes × 7×6):
  Plane 0 [0..41]:    my pieces
  Plane 1 [42..83]:   opponent pieces
  Plane 2 [84..125]:  legal moves (1.0 for all cells in non-full columns)

Actions: 0-6 = column index. Full column = illegal (-1.0 reward).
"""

from std.random import random_float64
from std.memory import alloc
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
    Saveable,
)
from noeira.nn.constants import DT as nn_dtype
from noeira.render import Renderer2D, SDL_Color
from ..core.board_env import BoardGameState, BoardGameAction, board_dtype

# Board dimensions
comptime COLS: Int = 7
comptime ROWS: Int = 6
comptime BOARD_SIZE: Int = 42  # 7 * 6

# State slot indices
comptime S_CURRENT_PLAYER: Int = 42
comptime S_GAME_RESULT: Int = 43
comptime S_STEP_COUNT: Int = 44
comptime S_LAST_COL: Int = 45

# Game result codes
comptime RESULT_ONGOING: Int = 0
comptime RESULT_P0_WINS: Int = 1
comptime RESULT_P1_WINS: Int = 2
comptime RESULT_DRAW: Int = 3


@always_inline
def _cell_idx(col: Int, row: Int) -> Int:
    """Column-major index: col * ROWS + row. Row 0 = bottom."""
    return col * ROWS + row


# ============================================================================
# ConnectFourEnv
# ============================================================================


struct ConnectFourEnv[DTYPE: DType = DType.float64](
    TwoPlayerDiscreteEnv
    & GPUTwoPlayerDiscreteEnv
    & RenderableEnv
    & Saveable
    & Defaultable
):
    """ConnectFour environment — CPU+GPU dual path."""

    # Trait conformance
    comptime dtype = Self.DTYPE
    comptime StateType = BoardGameState
    comptime ActionType = BoardGameAction

    # GPUTwoPlayerDiscreteEnv constants
    comptime STATE_SIZE: Int = 46
    comptime OBS_DIM: Int = 126  # 3 planes × 7×6
    comptime NUM_ACTIONS: Int = 7

    # Saveable
    comptime SAVE_SIZE: Int = 47  # 46 state + 1 done flag

    # CPU state
    var state: Array[Scalar[Self.dtype], 46]
    var done: Bool

    # Renderer
    var _renderer: Optional[Pointer[Renderer2D, MutUntrackedOrigin]]
    var _renderer_initialized: Bool

    def __init__(out self):
        self.state = Array[Scalar[Self.dtype], 46](
            fill=Scalar[Self.dtype](0.0)
        )
        self.done = False
        self._renderer = None
        self._renderer_initialized = False

    # ========================================================================
    # CPU: reset + step
    # ========================================================================

    def reset(mut self) -> BoardGameState:
        for i in range(46):
            self.state[i] = 0.0
        self.done = False
        return BoardGameState(index=0)

    def step(
        mut self, action: BoardGameAction, verbose: Bool = False
    ) -> Tuple[BoardGameState, Scalar[Self.dtype], Bool]:
        var result = self._step_impl(action.value)
        return (
            BoardGameState(index=Int(self.state[S_STEP_COUNT])),
            result[0],
            result[1],
        )

    def _find_drop_row(self, col: Int) -> Int:
        """Find the lowest empty row in a column. Returns -1 if full."""
        for row in range(ROWS):
            if self.state[_cell_idx(col, row)] == 0.0:
                return row
        return -1

    def _step_impl(mut self, action: Int) -> Tuple[Scalar[Self.dtype], Bool]:
        """Place piece in column, check win/draw. Returns (reward, done)."""
        if self.done:
            return (Scalar[Self.dtype](0.0), True)

        var col = action
        if col < 0 or col >= COLS:
            return (Scalar[Self.dtype](-1.0), False)

        # Find drop row
        var row = self._find_drop_row(col)
        if row < 0:
            # Column full — illegal move
            return (Scalar[Self.dtype](-1.0), False)

        var player = Int(self.state[S_CURRENT_PLAYER])
        var mark = Scalar[Self.dtype](player + 1)

        # Place piece
        self.state[_cell_idx(col, row)] = mark
        self.state[S_STEP_COUNT] += 1.0
        self.state[S_LAST_COL] = Scalar[Self.dtype](col)

        # Check win
        if self._check_win_cpu(col, row, mark):
            self.state[S_GAME_RESULT] = Scalar[Self.dtype](player + 1)
            self.done = True
            return (Scalar[Self.dtype](1.0), True)

        # Check draw (board full = 42 pieces)
        if Int(self.state[S_STEP_COUNT]) >= BOARD_SIZE:
            self.state[S_GAME_RESULT] = Scalar[Self.dtype](RESULT_DRAW)
            self.done = True
            return (Scalar[Self.dtype](0.0), True)

        # Switch player
        self.state[S_CURRENT_PLAYER] = Scalar[Self.dtype](1 - player)
        return (Scalar[Self.dtype](0.0), False)

    def _check_win_cpu(
        self, col: Int, row: Int, mark: Scalar[Self.dtype]
    ) -> Bool:
        """Check 4-in-a-row from the last placed piece in all 4 directions."""
        # Direction pairs: horizontal, vertical, diagonal /, diagonal \
        # (dc, dr) pairs
        return (
            self._count_dir(col, row, mark, 1, 0)
            + self._count_dir(col, row, mark, -1, 0)
            >= 3
            or self._count_dir(col, row, mark, 0, 1)
            + self._count_dir(col, row, mark, 0, -1)
            >= 3
            or self._count_dir(col, row, mark, 1, 1)
            + self._count_dir(col, row, mark, -1, -1)
            >= 3
            or self._count_dir(col, row, mark, 1, -1)
            + self._count_dir(col, row, mark, -1, 1)
            >= 3
        )

    def _count_dir(
        self, col: Int, row: Int, mark: Scalar[Self.dtype], dc: Int, dr: Int
    ) -> Int:
        """Count consecutive marks in one direction (excluding the starting cell).
        """
        var count = 0
        var c = col + dc
        var r = row + dr
        while c >= 0 and c < COLS and r >= 0 and r < ROWS:
            if self.state[_cell_idx(c, r)] != mark:
                break
            count += 1
            c += dc
            r += dr
        return count

    # ========================================================================
    # Saveable
    # ========================================================================

    def save_env_state(self, mut dst: List[Scalar[nn_dtype]]):
        for i in range(46):
            dst[i] = Scalar[nn_dtype](Float64(self.state[i]))
        dst[46] = Scalar[nn_dtype](1.0) if self.done else Scalar[nn_dtype](0.0)

    def load_env_state(mut self, src: List[Scalar[nn_dtype]]):
        for i in range(46):
            self.state[i] = Scalar[Self.dtype](Float64(src[i]))
        self.done = Float64(src[46]) > 0.5

    # ========================================================================
    # Env trait methods
    # ========================================================================

    def get_state(mut self) -> BoardGameState:
        return BoardGameState(index=Int(self.state[S_STEP_COUNT]))

    def close(mut self):
        if self._renderer_initialized:
            self._renderer.value()[].close()
            self._renderer.value().unsafe_free()
            self._renderer_initialized = False

    def action_from_index(self, action_idx: Int) -> BoardGameAction:
        return BoardGameAction(value=action_idx)

    def num_actions(self) -> Int:
        return 7

    def obs_dim(self) -> Int:
        return 126

    def num_states(self) -> Int:
        return 1

    def state_to_index(self, state: BoardGameState) -> Int:
        return state.index

    # ========================================================================
    # TwoPlayerDiscreteEnv trait methods
    # ========================================================================

    def current_player(self) -> Int:
        return Int(self.state[S_CURRENT_PLAYER])

    def legal_action_mask(self) -> List[Bool]:
        var mask = List[Bool](capacity=7)
        for col in range(COLS):
            # Column is legal if top row (row 5) is empty
            mask.append(
                self.state[_cell_idx(col, ROWS - 1)] == 0.0 and not self.done
            )
        return mask^

    def game_result(self) -> Int:
        return Int(self.state[S_GAME_RESULT])

    # ========================================================================
    # ContinuousStateEnv / BoxDiscreteActionEnv (CPU)
    # ========================================================================

    def get_obs_list(self) -> List[Scalar[Self.dtype]]:
        var obs = List[Scalar[Self.dtype]](capacity=126)
        var player = Int(self.state[S_CURRENT_PLAYER])
        var my_mark = Scalar[Self.dtype](player + 1)
        var opp_mark = Scalar[Self.dtype](2 - player)

        # Plane 0: my pieces (row-major: row * COLS + col for Conv2D)
        for row in range(ROWS):
            for col in range(COLS):
                if self.state[_cell_idx(col, row)] == my_mark:
                    obs.append(Scalar[Self.dtype](1.0))
                else:
                    obs.append(Scalar[Self.dtype](0.0))

        # Plane 1: opponent pieces (row-major)
        for row in range(ROWS):
            for col in range(COLS):
                if self.state[_cell_idx(col, row)] == opp_mark:
                    obs.append(Scalar[Self.dtype](1.0))
                else:
                    obs.append(Scalar[Self.dtype](0.0))

        # Plane 2: legal moves (broadcast: all cells in legal columns = 1.0)
        for _ in range(ROWS):
            for col in range(COLS):
                var col_legal = self.state[_cell_idx(col, ROWS - 1)] == 0.0
                if col_legal:
                    obs.append(Scalar[Self.dtype](1.0))
                else:
                    obs.append(Scalar[Self.dtype](0.0))

        return obs^

    def reset_obs_list(mut self) -> List[Scalar[Self.dtype]]:
        _ = self.reset()
        return self.get_obs_list()

    def step_obs(
        mut self, action: Int
    ) -> Tuple[List[Scalar[Self.dtype]], Scalar[Self.dtype], Bool]:
        """Single-agent step: agent plays, then random opponent plays."""
        var result = self._step_impl(action)
        var reward = result[0]
        var done = result[1]

        if done:
            return (self.get_obs_list(), reward, done)

        # Random opponent
        var legal_cols = List[Int](capacity=7)
        for col in range(COLS):
            if self.state[_cell_idx(col, ROWS - 1)] == 0.0:
                legal_cols.append(col)

        if len(legal_cols) > 0:
            var opp_idx = Int(random_float64() * Float64(len(legal_cols)))
            if opp_idx >= len(legal_cols):
                opp_idx = len(legal_cols) - 1
            var opp_result = self._step_impl(legal_cols[opp_idx])
            done = opp_result[1]
            if done and Int(self.state[S_GAME_RESULT]) != RESULT_DRAW:
                reward = Scalar[Self.dtype](-1.0)

        return (self.get_obs_list(), reward, done)

    # ========================================================================
    # RenderableEnv trait methods
    # ========================================================================

    def init_renderer(mut self) raises -> Bool:
        if self._renderer_initialized:
            return True
        self._renderer = alloc[Renderer2D]({count = 1}).unsafe_leak()
        self._renderer.value().unsafe_write(
            Renderer2D(width=560, height=530, fps=30, title="ConnectFour")
        )
        self._renderer_initialized = True
        return True

    def render_frame(mut self) raises -> None:
        if not self._renderer_initialized:
            return
        self._render(self._renderer.value()[])

    def render_board(self, mut renderer: Renderer2D):
        """Draw the board background, pieces, and grid lines into ``renderer``.

        The reusable board visual — **no** frame begin/flip and **no** status
        bar, so an interactive play script can call ``begin_frame`` itself,
        overlay a column selector + its own status text, then ``flip``. The
        env's own ``_render`` (the `RenderableEnv` path) wraps this with a frame
        + the generic turn/result status bar. Geometry: 7×6 grid of 80px cells,
        board top at y=50 (50px status bar below)."""
        var board_color = SDL_Color(r=0x00, g=0x00, b=0xAA, a=0xFF)
        var empty_color = SDL_Color(r=0x33, g=0x33, b=0x33, a=0xFF)
        var red_color = SDL_Color(r=0xFF, g=0x22, b=0x22, a=0xFF)
        var yellow_color = SDL_Color(r=0xFF, g=0xDD, b=0x00, a=0xFF)

        var cell_size = 80
        var board_cols = 7
        var board_rows = 6
        var board_width = board_cols * cell_size  # 560
        var board_height = board_rows * cell_size  # 480
        var circle_radius = 32

        # Draw board background
        renderer.draw_rect(0, 50, board_width, board_height, board_color)

        # Draw cells
        for col in range(board_cols):
            for row in range(board_rows):
                # Visual: row 0 in env = bottom, so flip vertically
                var visual_row = board_rows - 1 - row
                var cx = col * cell_size + cell_size // 2
                var cy = 50 + visual_row * cell_size + cell_size // 2

                # Get cell value from env state (column-major: col*6+row)
                var cell_idx = col * board_rows + row
                var cell_val = Int(self.state[cell_idx])

                if cell_val == 1:
                    renderer.draw_circle(
                        cx, cy, circle_radius, red_color, filled=True
                    )
                elif cell_val == 2:
                    renderer.draw_circle(
                        cx, cy, circle_radius, yellow_color, filled=True
                    )
                else:
                    renderer.draw_circle(
                        cx, cy, circle_radius, empty_color, filled=True
                    )

        # Draw grid lines over circles for visual separation
        for i in range(1, board_cols):
            renderer.draw_line(
                i * cell_size,
                50,
                i * cell_size,
                50 + board_height,
                board_color,
                1,
            )
        for i in range(1, board_rows):
            renderer.draw_line(
                0,
                50 + i * cell_size,
                board_width,
                50 + i * cell_size,
                board_color,
                1,
            )

    def _render(self, mut renderer: Renderer2D):
        """Render ConnectFour board state (full self-contained frame)."""
        var red_color = SDL_Color(r=0xFF, g=0x22, b=0x22, a=0xFF)
        var yellow_color = SDL_Color(r=0xFF, g=0xDD, b=0x00, a=0xFF)
        var bg_color = SDL_Color(r=0x11, g=0x11, b=0x44, a=0xFF)
        var win_text_color = SDL_Color(r=0xFF, g=0xDD, b=0x00, a=0xFF)
        var status_bg = SDL_Color(r=0x11, g=0x11, b=0x22, a=0xFF)

        var board_height = 6 * 80  # 480
        var board_width = 7 * 80  # 560

        if not renderer.begin_frame_with_color(bg_color):
            return

        self.render_board(renderer)

        # Status bar at bottom (y=530-50..530)
        renderer.draw_rect(0, 50 + board_height, board_width, 50, status_bg)

        var game_result = self.game_result()
        if game_result == 0:
            var player = self.current_player()
            if player == 0:
                renderer.draw_text(
                    "Red's turn", 230, 50 + board_height + 20, red_color
                )
            else:
                renderer.draw_text(
                    "Yellow's turn",
                    218,
                    50 + board_height + 20,
                    yellow_color,
                )
        elif game_result == 1:
            renderer.draw_text(
                "Red Wins!", 220, 50 + board_height + 20, win_text_color
            )
        elif game_result == 2:
            renderer.draw_text(
                "Yellow Wins!",
                208,
                50 + board_height + 20,
                win_text_color,
            )
        else:
            renderer.draw_text(
                "Draw!", 240, 50 + board_height + 20, win_text_color
            )

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
    def _gpu_count_dir[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        states: Tile[board_dtype, BATCH_SIZE, STATE_SIZE],
        i: Int,
        col: Int,
        row: Int,
        mark: Scalar[board_dtype],
        dc: Int,
        dr: Int,
    ) -> Int:
        """Count consecutive marks in one direction on GPU."""
        var count = 0
        var c = col + dc
        var r = row + dr
        while c >= 0 and c < COLS and r >= 0 and r < ROWS:
            if states[i, _cell_idx(c, r)] != mark:
                break
            count += 1
            c += dc
            r += dr
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
        """Per-thread ConnectFour step kernel."""
        var i = Int(block_dim.x * block_idx.x + thread_idx.x)
        if i >= BATCH_SIZE:
            return

        var col = Int(actions[i])

        # Already done?
        if states[i, S_GAME_RESULT] != 0.0:
            rewards[i] = 0.0
            dones[i] = 1.0
            return

        # Validate column
        if col < 0 or col >= COLS:
            rewards[i] = -1.0
            dones[i] = 0.0
            return

        # Find drop row
        var row = -1
        for r in range(ROWS):
            if states[i, _cell_idx(col, r)] == 0.0:
                row = r
                break

        if row < 0:
            # Column full
            rewards[i] = -1.0
            dones[i] = 0.0
            return

        var player = Int(states[i, S_CURRENT_PLAYER])
        var mark = Scalar[board_dtype](player + 1)

        # Place piece
        states[i, _cell_idx(col, row)] = mark
        states[i, S_STEP_COUNT] = states[i, S_STEP_COUNT] + 1.0
        states[i, S_LAST_COL] = Scalar[board_dtype](col)

        # Check win (4 directions)
        var h = Self._gpu_count_dir[BATCH_SIZE, STATE_SIZE](
            states, i, col, row, mark, 1, 0
        ) + Self._gpu_count_dir[BATCH_SIZE, STATE_SIZE](
            states, i, col, row, mark, -1, 0
        )
        var v = Self._gpu_count_dir[BATCH_SIZE, STATE_SIZE](
            states, i, col, row, mark, 0, 1
        ) + Self._gpu_count_dir[BATCH_SIZE, STATE_SIZE](
            states, i, col, row, mark, 0, -1
        )
        var d1 = Self._gpu_count_dir[BATCH_SIZE, STATE_SIZE](
            states, i, col, row, mark, 1, 1
        ) + Self._gpu_count_dir[BATCH_SIZE, STATE_SIZE](
            states, i, col, row, mark, -1, -1
        )
        var d2 = Self._gpu_count_dir[BATCH_SIZE, STATE_SIZE](
            states, i, col, row, mark, 1, -1
        ) + Self._gpu_count_dir[BATCH_SIZE, STATE_SIZE](
            states, i, col, row, mark, -1, 1
        )

        if h >= 3 or v >= 3 or d1 >= 3 or d2 >= 3:
            states[i, S_GAME_RESULT] = Scalar[board_dtype](player + 1)
            rewards[i] = 1.0
            dones[i] = 1.0
            return

        # Check draw
        if Int(states[i, S_STEP_COUNT]) >= BOARD_SIZE:
            states[i, S_GAME_RESULT] = Scalar[board_dtype](RESULT_DRAW)
            rewards[i] = 0.0
            dones[i] = 1.0
            return

        # Switch player
        states[i, S_CURRENT_PLAYER] = Scalar[board_dtype](1 - player)
        rewards[i] = 0.0
        dones[i] = 0.0

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
        for c in range(46):
            states[i, c] = 0.0

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
            for c in range(46):
                states[i, c] = 0.0
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

        var player = Int(states[i, S_CURRENT_PLAYER])
        var my_mark = Scalar[board_dtype](player + 1)
        var opp_mark = Scalar[board_dtype](2 - player)
        var game_over = states[i, S_GAME_RESULT] != 0.0

        # Plane 0: my pieces, Plane 1: opp pieces
        # Conv2D expects row-major spatial layout: row * COLS + col
        # State uses column-major: col * ROWS + row
        for col in range(COLS):
            for row in range(ROWS):
                var state_idx = _cell_idx(col, row)  # col*6 + row
                var obs_idx = row * COLS + col  # row-major for Conv2D
                var cell = states[i, state_idx]
                if cell == my_mark:
                    obs[i, obs_idx] = 1.0
                else:
                    obs[i, obs_idx] = 0.0
                if cell == opp_mark:
                    obs[i, BOARD_SIZE + obs_idx] = 1.0
                else:
                    obs[i, BOARD_SIZE + obs_idx] = 0.0

        # Plane 2: legal moves (broadcast by column) + legal_masks
        for col in range(COLS):
            var top_idx = _cell_idx(col, ROWS - 1)
            var col_legal = states[i, top_idx] == 0.0 and not game_over
            if col_legal:
                legal_masks[i, col] = 1.0
            else:
                legal_masks[i, col] = 0.0
            for row in range(ROWS):
                var obs_idx = 2 * BOARD_SIZE + row * COLS + col  # row-major
                if col_legal:
                    obs[i, obs_idx] = 1.0
                else:
                    obs[i, obs_idx] = 0.0

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
        var legal_masks = tile[BATCH_SIZE, 7](legal_masks_buf)

        comptime BLOCKS = (BATCH_SIZE + Self.TPB - 1) // Self.TPB

        @always_inline
        def step_wrapper(
            states: Tile[board_dtype, BATCH_SIZE, STATE_SIZE],
            actions: ImmTile[board_dtype, BATCH_SIZE],
            rewards: Tile[board_dtype, BATCH_SIZE],
            dones: Tile[board_dtype, BATCH_SIZE],
            terminated_out: Tile[board_dtype, BATCH_SIZE],
            obs: Tile[board_dtype, BATCH_SIZE, OBS_DIM],
            legal_masks: Tile[board_dtype, BATCH_SIZE, 7],
        ):
            ConnectFourEnv.step_kernel[BATCH_SIZE, STATE_SIZE](
                states, actions, rewards, dones
            )
            var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
            if idx < BATCH_SIZE:
                terminated_out[idx] = dones[idx]

            var states_read = tile[BATCH_SIZE, STATE_SIZE](rebind[Pointer[Scalar[board_dtype], ImmutAnyOrigin]]( states.ptr ))
            ConnectFourEnv.extract_obs_and_masks[
                BATCH_SIZE, STATE_SIZE, OBS_DIM, 7
            ](states_read, obs, legal_masks)

        ctx.enqueue_function[step_wrapper](
            states,
            actions,
            rewards,
            dones,
            terminated_out,
            obs,
            legal_masks,
            grid_dim=(BLOCKS,),
            block_dim=(Self.TPB,),
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
            ConnectFourEnv.reset_kernel[BATCH_SIZE, STATE_SIZE](states)

        ctx.enqueue_function[reset_wrapper](
            states,
            grid_dim=(BLOCKS,),
            block_dim=(Self.TPB,),
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
            ConnectFourEnv.selective_reset_kernel[BATCH_SIZE, STATE_SIZE](
                states, dones
            )

        ctx.enqueue_function[sel_reset_wrapper](
            states,
            dones,
            grid_dim=(BLOCKS,),
            block_dim=(Self.TPB,),
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
        var legal_masks = tile[BATCH_SIZE, 7](legal_masks_buf)
        comptime BLOCKS = (BATCH_SIZE + Self.TPB - 1) // Self.TPB

        @always_inline
        def extract_wrapper(
            states: ImmTile[board_dtype, BATCH_SIZE, STATE_SIZE],
            obs: Tile[board_dtype, BATCH_SIZE, OBS_DIM],
            legal_masks: Tile[board_dtype, BATCH_SIZE, 7],
        ):
            ConnectFourEnv.extract_obs_and_masks[
                BATCH_SIZE, STATE_SIZE, OBS_DIM, 7
            ](states, obs, legal_masks)

        ctx.enqueue_function[extract_wrapper](
            states,
            obs,
            legal_masks,
            grid_dim=(BLOCKS,),
            block_dim=(Self.TPB,),
        )
