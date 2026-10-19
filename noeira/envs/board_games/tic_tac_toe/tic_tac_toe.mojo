"""TicTacToe — CPU+GPU environment for two-player self-play RL training.

3×3 board, two players alternate placing marks. First to get 3 in a row wins.

State layout (STATE_SIZE = 12):
  [0..8]  board cells (0=empty, 1=P0, 2=P1)
  [9]     current_player (0 or 1)
  [10]    game_result (0=ongoing, 1=P0 wins, 2=P1 wins, 3=draw)
  [11]    step_count

Canonical obs (OBS_DIM = 27 = 3 planes × 3×3):
  Plane 0 [0..8]:   my pieces (1.0 where current player has a mark)
  Plane 1 [9..17]:  opponent pieces
  Plane 2 [18..26]: legal moves (1.0 where cell is empty)

Actions: 0-8 = cell index (row-major). Illegal move = pass with -1.0 reward.
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

# State slot indices
comptime S_BOARD_START: Int = 0
comptime S_BOARD_END: Int = 9
comptime S_CURRENT_PLAYER: Int = 9
comptime S_GAME_RESULT: Int = 10
comptime S_STEP_COUNT: Int = 11

# Game result codes
comptime RESULT_ONGOING: Int = 0
comptime RESULT_P0_WINS: Int = 1
comptime RESULT_P1_WINS: Int = 2
comptime RESULT_DRAW: Int = 3


# ============================================================================
# TicTacToeEnv
# ============================================================================


struct TicTacToeEnv[DTYPE: DType = DType.float64](
    TwoPlayerDiscreteEnv
    & GPUTwoPlayerDiscreteEnv
    & RenderableEnv
    & Saveable
    & Defaultable
):
    """TicTacToe environment — CPU+GPU dual path.

    CPU: Instance methods for evaluation + single-agent mode.
    GPU: Static inline methods for batched self-play training.
    """

    # Trait conformance
    comptime dtype = Self.DTYPE
    comptime StateType = BoardGameState
    comptime ActionType = BoardGameAction

    # GPUTwoPlayerDiscreteEnv constants
    comptime STATE_SIZE: Int = 12
    comptime OBS_DIM: Int = 27  # 3 planes × 3×3
    comptime NUM_ACTIONS: Int = 9

    # Saveable
    comptime SAVE_SIZE: Int = 13  # 12 state + 1 done flag

    # CPU state
    var state: Array[Scalar[Self.dtype], 12]
    var done: Bool

    # Renderer
    var _renderer: Optional[Pointer[Renderer2D, MutUntrackedOrigin]]
    var _renderer_initialized: Bool

    def __init__(out self):
        self.state = Array[Scalar[Self.dtype], 12](
            fill=Scalar[Self.dtype](0.0)
        )
        self.done = False
        self._renderer = None
        self._renderer_initialized = False

    # ========================================================================
    # CPU: reset + step
    # ========================================================================

    def reset(mut self) -> BoardGameState:
        for i in range(12):
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

    def _step_impl(mut self, action: Int) -> Tuple[Scalar[Self.dtype], Bool]:
        """Internal step: place mark, check win/draw. Returns (reward, done)."""
        if self.done:
            return (Scalar[Self.dtype](0.0), True)

        var player = Int(self.state[S_CURRENT_PLAYER])
        var mark = Scalar[Self.dtype](player + 1)  # 1 or 2

        # Check legality
        if action < 0 or action >= 9 or self.state[action] != 0.0:
            # Illegal move — penalty, no board change
            return (Scalar[Self.dtype](-1.0), False)

        # Place mark
        self.state[action] = mark
        self.state[S_STEP_COUNT] += 1.0

        # Check win
        if Self._check_win_cpu(self.state, mark):
            self.state[S_GAME_RESULT] = Scalar[Self.dtype](player + 1)
            self.done = True
            return (Scalar[Self.dtype](1.0), True)

        # Check draw (board full)
        var empty_count = 0
        for i in range(9):
            if self.state[i] == 0.0:
                empty_count += 1
        if empty_count == 0:
            self.state[S_GAME_RESULT] = Scalar[Self.dtype](RESULT_DRAW)
            self.done = True
            return (Scalar[Self.dtype](0.0), True)

        # Switch player
        self.state[S_CURRENT_PLAYER] = Scalar[Self.dtype](1 - player)
        return (Scalar[Self.dtype](0.0), False)

    @staticmethod
    def _check_win_cpu(
        state: Array[Scalar[Self.DTYPE], 12], mark: Scalar[Self.DTYPE]
    ) -> Bool:
        """Check if the given mark has won (rows, cols, diagonals)."""
        # Rows
        if state[0] == mark and state[1] == mark and state[2] == mark:
            return True
        if state[3] == mark and state[4] == mark and state[5] == mark:
            return True
        if state[6] == mark and state[7] == mark and state[8] == mark:
            return True
        # Columns
        if state[0] == mark and state[3] == mark and state[6] == mark:
            return True
        if state[1] == mark and state[4] == mark and state[7] == mark:
            return True
        if state[2] == mark and state[5] == mark and state[8] == mark:
            return True
        # Diagonals
        if state[0] == mark and state[4] == mark and state[8] == mark:
            return True
        if state[2] == mark and state[4] == mark and state[6] == mark:
            return True
        return False

    # ========================================================================
    # Saveable
    # ========================================================================

    def save_env_state(self, mut dst: List[Scalar[nn_dtype]]):
        for i in range(12):
            dst[i] = Scalar[nn_dtype](Float64(self.state[i]))
        dst[12] = Scalar[nn_dtype](1.0) if self.done else Scalar[nn_dtype](0.0)

    def load_env_state(mut self, src: List[Scalar[nn_dtype]]):
        for i in range(12):
            self.state[i] = Scalar[Self.dtype](Float64(src[i]))
        self.done = Float64(src[12]) > 0.5

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
        return 9

    def obs_dim(self) -> Int:
        return 27

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
        var mask = List[Bool](capacity=9)
        for i in range(9):
            mask.append(self.state[i] == 0.0 and not self.done)
        return mask^

    def game_result(self) -> Int:
        return Int(self.state[S_GAME_RESULT])

    # ========================================================================
    # ContinuousStateEnv / BoxDiscreteActionEnv (CPU)
    # Canonical observation: always from current player's perspective
    # ========================================================================

    def get_obs_list(self) -> List[Scalar[Self.dtype]]:
        var obs = List[Scalar[Self.dtype]](capacity=27)
        var player = Int(self.state[S_CURRENT_PLAYER])
        var my_mark = Scalar[Self.dtype](player + 1)
        var opp_mark = Scalar[Self.dtype](2 - player)

        # Plane 0: my pieces
        for i in range(9):
            if self.state[i] == my_mark:
                obs.append(Scalar[Self.dtype](1.0))
            else:
                obs.append(Scalar[Self.dtype](0.0))

        # Plane 1: opponent pieces
        for i in range(9):
            if self.state[i] == opp_mark:
                obs.append(Scalar[Self.dtype](1.0))
            else:
                obs.append(Scalar[Self.dtype](0.0))

        # Plane 2: legal moves
        for i in range(9):
            if self.state[i] == 0.0:
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
        """Single-agent step: agent plays, then random opponent plays.

        This makes TicTacToe compatible with existing single-agent training
        loops (DQN, PPO, etc.) by embedding a random opponent inside the env.
        """
        # Agent's move
        var result = self._step_impl(action)
        var reward = result[0]
        var done = result[1]

        if done:
            return (self.get_obs_list(), reward, done)

        # Random opponent's move
        var legal_cells = List[Int](capacity=9)
        for i in range(9):
            if self.state[i] == 0.0:
                legal_cells.append(i)

        if len(legal_cells) > 0:
            var opp_idx = Int(random_float64() * Float64(len(legal_cells)))
            if opp_idx >= len(legal_cells):
                opp_idx = len(legal_cells) - 1
            var opp_action = legal_cells[opp_idx]
            var opp_result = self._step_impl(opp_action)
            done = opp_result[1]
            # If opponent won, agent gets -1
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
            Renderer2D(width=400, height=450, fps=30, title="TicTacToe")
        )
        self._renderer_initialized = True
        return True

    def render_frame(mut self) raises -> None:
        if not self._renderer_initialized:
            return
        self._render(self._renderer.value()[])

    def _render(self, mut renderer: Renderer2D):
        """Render TicTacToe board state."""
        var bg_color = SDL_Color(r=0x1A, g=0x5C, b=0x2A, a=0xFF)
        var grid_color = SDL_Color(r=0xFF, g=0xFF, b=0xFF, a=0xFF)
        var x_color = SDL_Color(r=0xFF, g=0x44, b=0x44, a=0xFF)
        var o_color = SDL_Color(r=0x44, g=0x88, b=0xFF, a=0xFF)
        var text_color = SDL_Color(r=0xFF, g=0xFF, b=0xFF, a=0xFF)
        var win_color = SDL_Color(r=0xFF, g=0xDD, b=0x00, a=0xFF)

        var cell_size = 133  # 400 / 3
        var board_size = 400

        if not renderer.begin_frame_with_color(bg_color):
            return

        # Draw grid lines (2 horizontal + 2 vertical)
        for i in range(1, 3):
            renderer.draw_line(
                0, i * cell_size, board_size, i * cell_size, grid_color, 3
            )
            renderer.draw_line(
                i * cell_size, 0, i * cell_size, board_size, grid_color, 3
            )

        # Draw marks
        for row in range(3):
            for col in range(3):
                var cx = col * cell_size + cell_size // 2
                var cy = row * cell_size + cell_size // 2
                var cell_idx = row * 3 + col
                var cell_val = Int(self.state[cell_idx])

                if cell_val == 1:
                    # Draw X: two crossing lines
                    var margin = 25
                    var x0 = col * cell_size + margin
                    var y0 = row * cell_size + margin
                    var x1 = (col + 1) * cell_size - margin
                    var y1 = (row + 1) * cell_size - margin
                    renderer.draw_line(x0, y0, x1, y1, x_color, 4)
                    renderer.draw_line(x1, y0, x0, y1, x_color, 4)
                elif cell_val == 2:
                    # Draw O: circle
                    renderer.draw_circle(
                        cx, cy, cell_size // 2 - 20, o_color, filled=False
                    )

        # Status bar at bottom (y=400..450)
        renderer.draw_rect(
            0,
            board_size,
            400,
            50,
            SDL_Color(r=0x11, g=0x33, b=0x11, a=0xFF),
        )

        var game_result = self.game_result()
        if game_result == 0:
            var player = self.current_player()
            if player == 0:
                renderer.draw_text(
                    "Player X's turn", 140, board_size + 20, text_color
                )
            else:
                renderer.draw_text(
                    "Player O's turn", 140, board_size + 20, text_color
                )
        elif game_result == 1:
            renderer.draw_text("X Wins!", 160, board_size + 20, win_color)
        elif game_result == 2:
            renderer.draw_text("O Wins!", 160, board_size + 20, win_color)
        else:
            renderer.draw_text("Draw!", 170, board_size + 20, win_color)

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
    # GPU: Inline step/reset kernels (called per-thread on GPU)
    # ========================================================================

    comptime TPB = 256

    @staticmethod
    @always_inline
    def _check_win_from_states[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        states: Tile[board_dtype, BATCH_SIZE, STATE_SIZE],
        i: Int,
        mark: Scalar[board_dtype],
    ) -> Bool:
        """Check if the given mark has won, reading directly from states tensor.
        """
        # Rows
        if (
            states[i, 0] == mark
            and states[i, 1] == mark
            and states[i, 2] == mark
        ):
            return True
        if (
            states[i, 3] == mark
            and states[i, 4] == mark
            and states[i, 5] == mark
        ):
            return True
        if (
            states[i, 6] == mark
            and states[i, 7] == mark
            and states[i, 8] == mark
        ):
            return True
        # Columns
        if (
            states[i, 0] == mark
            and states[i, 3] == mark
            and states[i, 6] == mark
        ):
            return True
        if (
            states[i, 1] == mark
            and states[i, 4] == mark
            and states[i, 7] == mark
        ):
            return True
        if (
            states[i, 2] == mark
            and states[i, 5] == mark
            and states[i, 8] == mark
        ):
            return True
        # Diagonals
        if (
            states[i, 0] == mark
            and states[i, 4] == mark
            and states[i, 8] == mark
        ):
            return True
        if (
            states[i, 2] == mark
            and states[i, 4] == mark
            and states[i, 6] == mark
        ):
            return True
        return False

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
        """Per-thread TicTacToe step kernel."""
        var i = Int(block_dim.x * block_idx.x + thread_idx.x)
        if i >= BATCH_SIZE:
            return

        var action = Int(actions[i])

        var player = Int(states[i, S_CURRENT_PLAYER])
        var mark = Scalar[board_dtype](player + 1)

        # Check if game already done
        if states[i, S_GAME_RESULT] != 0.0:
            rewards[i] = 0.0
            dones[i] = 1.0
            return

        # Check legality
        if action < 0 or action >= 9 or states[i, action] != 0.0:
            rewards[i] = -1.0
            dones[i] = 0.0
            return

        # Place mark
        states[i, action] = mark
        states[i, S_STEP_COUNT] = states[i, S_STEP_COUNT] + 1.0

        # Check win — read board directly from states tensor
        var won = Self._check_win_from_states[BATCH_SIZE, STATE_SIZE](
            states, i, mark
        )
        if won:
            states[i, S_GAME_RESULT] = Scalar[board_dtype](player + 1)
            rewards[i] = 1.0
            dones[i] = 1.0
            return

        # Check draw
        var empty = 0
        for c in range(9):
            if states[i, c] == 0.0:
                empty += 1
        if empty == 0:
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
        """Per-thread reset: clear board, player 0 starts."""
        var i = Int(block_dim.x * block_idx.x + thread_idx.x)
        if i >= BATCH_SIZE:
            return

        for c in range(12):
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
        """Per-thread: reset only if done."""
        var i = Int(block_dim.x * block_idx.x + thread_idx.x)
        if i >= BATCH_SIZE:
            return

        if dones[i] > 0.5:
            for c in range(12):
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
        """Per-thread: extract canonical obs + legal masks from state."""
        var i = Int(block_dim.x * block_idx.x + thread_idx.x)
        if i >= BATCH_SIZE:
            return

        var player = Int(states[i, S_CURRENT_PLAYER])
        var my_mark = Scalar[board_dtype](player + 1)
        var opp_mark = Scalar[board_dtype](2 - player)
        var game_over = states[i, S_GAME_RESULT] != 0.0

        for c in range(9):
            var cell = states[i, c]
            # Plane 0: my pieces
            if cell == my_mark:
                obs[i, c] = 1.0
            else:
                obs[i, c] = 0.0
            # Plane 1: opponent pieces
            if cell == opp_mark:
                obs[i, 9 + c] = 1.0
            else:
                obs[i, 9 + c] = 0.0
            # Plane 2: legal moves
            var is_empty = cell == 0.0
            if is_empty and not game_over:
                obs[i, 18 + c] = 1.0
                legal_masks[i, c] = 1.0
            else:
                obs[i, 18 + c] = 0.0
                legal_masks[i, c] = 0.0

    # ========================================================================
    # GPU Launcher Methods (host-side, GPUTwoPlayerDiscreteEnv trait)
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
        var legal_masks = tile[BATCH_SIZE, 9](legal_masks_buf)

        comptime BLOCKS = (BATCH_SIZE + Self.TPB - 1) // Self.TPB

        @always_inline
        def step_wrapper(
            states: Tile[board_dtype, BATCH_SIZE, STATE_SIZE],
            actions: ImmTile[board_dtype, BATCH_SIZE],
            rewards: Tile[board_dtype, BATCH_SIZE],
            dones: Tile[board_dtype, BATCH_SIZE],
            terminated_out: Tile[board_dtype, BATCH_SIZE],
            obs: Tile[board_dtype, BATCH_SIZE, OBS_DIM],
            legal_masks: Tile[board_dtype, BATCH_SIZE, 9],
        ):
            # Step physics
            TicTacToeEnv.step_kernel[BATCH_SIZE, STATE_SIZE](
                states, actions, rewards, dones
            )

            # Copy terminated = dones for board games
            var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
            if idx < BATCH_SIZE:
                terminated_out[idx] = dones[idx]

            # Extract obs + legal masks (need immutable view of states)
            var states_read = tile[BATCH_SIZE, STATE_SIZE](rebind[Pointer[Scalar[board_dtype], ImmutAnyOrigin]]( states.ptr ))
            TicTacToeEnv.extract_obs_and_masks[
                BATCH_SIZE, STATE_SIZE, OBS_DIM, 9
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
            TicTacToeEnv.reset_kernel[BATCH_SIZE, STATE_SIZE](states)

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
            TicTacToeEnv.selective_reset_kernel[BATCH_SIZE, STATE_SIZE](
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
        var legal_masks = tile[BATCH_SIZE, 9](legal_masks_buf)

        comptime BLOCKS = (BATCH_SIZE + Self.TPB - 1) // Self.TPB

        @always_inline
        def extract_wrapper(
            states: ImmTile[board_dtype, BATCH_SIZE, STATE_SIZE],
            obs: Tile[board_dtype, BATCH_SIZE, OBS_DIM],
            legal_masks: Tile[board_dtype, BATCH_SIZE, 9],
        ):
            TicTacToeEnv.extract_obs_and_masks[
                BATCH_SIZE, STATE_SIZE, OBS_DIM, 9
            ](states, obs, legal_masks)

        ctx.enqueue_function[extract_wrapper](
            states,
            obs,
            legal_masks,
            grid_dim=(BLOCKS,),
            block_dim=(Self.TPB,),
        )
