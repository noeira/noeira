"""Native Breakout — CPU+GPU environment for RL training.

Ball + paddle + brick grid. No Atari emulation.
Follows the CartPole/Pong pattern.

State layout (STATE_SIZE = 56):
  [0]  ball_x             \
  [1]  ball_y              |
  [2]  ball_vx             | CLEAN_OBS_DIM = 7
  [3]  ball_vy             |
  [4]  paddle_x            |
  [5]  bricks_remaining    |
  [6]  lives              /
  [7..48]  brick_alive (6 rows × 7 cols = 42 floats, 0.0 or 1.0)
  [49] score
  [50] ball_stuck  (1.0 = ball on paddle before serve)
  [51] step_count
  [52..55] reserved

Actions: 0=NOOP, 1=FIRE (release ball), 2=LEFT, 3=RIGHT
"""

from std.random import random_float64
from std.memory import alloc
from noeira.core import (
    State,
    Action,
    BoxDiscreteActionEnv,
    GPUDiscreteEnv,
    RenderableEnv,
)
from noeira.render import Renderer2D, SDL_Color, Vec2, Camera, black, white
from std.random.philox import Random as PhiloxRandom
from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext, DeviceBuffer

from ..core.gpu_env import ArcadeGameState, ArcadeGameAction, gpu_dtype
from ..core.colors import SCREEN_W, SCREEN_H
from noeira.core.tiles import Tile, ImmTile, tile

# ============================================================================
# Physics Constants
# ============================================================================

comptime PADDLE_WIDTH: Int = 16
comptime PADDLE_HEIGHT: Int = 4
comptime PADDLE_Y: Int = 195  # Near bottom
comptime PADDLE_SPEED: Float64 = 4.0
comptime BALL_SIZE: Int = 2
comptime BALL_SPEED_X: Float64 = 1.5
comptime BALL_SPEED_Y: Float64 = -2.0  # Upward
comptime MAX_BALL_VX: Float64 = 3.0
comptime MAX_BALL_VY: Float64 = 3.0

# Brick grid
comptime BRICK_ROWS: Int = 6
comptime BRICK_COLS: Int = 7
comptime BRICK_WIDTH: Int = 20
comptime BRICK_HEIGHT: Int = 8
comptime BRICK_TOP: Int = 30  # Y start of brick area
comptime BRICK_LEFT: Int = 10  # X start
comptime BRICK_GAP: Int = 2

comptime TOTAL_BRICKS: Int = BRICK_ROWS * BRICK_COLS  # 42
comptime INITIAL_LIVES: Int = 5
comptime BREAKOUT_MAX_STEPS: Int = 10000

# Score per brick row (top rows = more points, like original)
comptime ROW_SCORES: Array[Int, 6] = [7, 7, 4, 4, 1, 1]

# State slot indices
comptime S_BALL_X: Int = 0
comptime S_BALL_Y: Int = 1
comptime S_BALL_VX: Int = 2
comptime S_BALL_VY: Int = 3
comptime S_PADDLE_X: Int = 4
comptime S_BRICKS_REM: Int = 5
comptime S_LIVES: Int = 6
comptime S_BRICKS_START: Int = 7  # 42 brick slots
comptime S_SCORE: Int = 49
comptime S_BALL_STUCK: Int = 50
comptime S_STEP_COUNT: Int = 51

# ============================================================================
# BreakoutEnv
# ============================================================================


struct BreakoutEnv[DTYPE: DType](
    BoxDiscreteActionEnv & GPUDiscreteEnv & RenderableEnv
):
    """Native Breakout environment — CPU+GPU dual path."""

    comptime dtype = Self.DTYPE
    comptime StateType = ArcadeGameState
    comptime ActionType = ArcadeGameAction

    comptime STATE_SIZE: Int = 56
    comptime OBS_DIM: Int = 7
    comptime NUM_ACTIONS: Int = 4  # NOOP, FIRE, LEFT, RIGHT
    comptime STEP_WS_SHARED: Int = 0
    comptime STEP_WS_PER_ENV: Int = 0

    var state: Array[Scalar[Self.dtype], 56]
    var done: Bool
    var _rng_counter: UInt32

    var _renderer: Optional[Pointer[Renderer2D, MutUntrackedOrigin]]
    var _renderer_initialized: Bool

    def __init__(out self):
        self.state = Array[Scalar[Self.dtype], 56](
            fill=Scalar[Self.dtype](0.0)
        )
        self.done = False
        self._rng_counter = 42
        self._renderer = None
        self._renderer_initialized = False

    # ========================================================================
    # CPU reset + step
    # ========================================================================

    def reset(mut self) -> ArcadeGameState:
        self._rng_counter += 1
        # Ball stuck on paddle
        self.state[S_PADDLE_X] = Scalar[Self.dtype](SCREEN_W // 2)
        self.state[S_BALL_X] = Scalar[Self.dtype](SCREEN_W // 2)
        self.state[S_BALL_Y] = Scalar[Self.dtype](PADDLE_Y - BALL_SIZE - 1)
        self.state[S_BALL_VX] = 0.0
        self.state[S_BALL_VY] = 0.0
        self.state[S_BALL_STUCK] = 1.0
        self.state[S_LIVES] = Scalar[Self.dtype](INITIAL_LIVES)
        self.state[S_SCORE] = 0.0
        self.state[S_STEP_COUNT] = 0.0
        self.state[S_BRICKS_REM] = Scalar[Self.dtype](TOTAL_BRICKS)
        # All bricks alive
        for b in range(TOTAL_BRICKS):
            self.state[S_BRICKS_START + b] = 1.0
        self.done = False
        return ArcadeGameState(index=0)

    def step(
        mut self, action: ArcadeGameAction, verbose: Bool = False
    ) -> Tuple[ArcadeGameState, Scalar[Self.dtype], Bool]:
        var result = self._step_impl(action.value)
        return (
            ArcadeGameState(index=Int(self.state[S_STEP_COUNT])),
            result[0],
            result[1],
        )

    def _step_impl(mut self, action: Int) -> Tuple[Scalar[Self.dtype], Bool]:
        var paddle_x = self.state[S_PADDLE_X]

        # Move paddle
        if action == 2:  # LEFT
            paddle_x -= Scalar[Self.dtype](PADDLE_SPEED)
            if paddle_x < Scalar[Self.dtype](PADDLE_WIDTH // 2):
                paddle_x = Scalar[Self.dtype](PADDLE_WIDTH // 2)
        elif action == 3:  # RIGHT
            paddle_x += Scalar[Self.dtype](PADDLE_SPEED)
            if paddle_x > Scalar[Self.dtype](SCREEN_W - PADDLE_WIDTH // 2):
                paddle_x = Scalar[Self.dtype](SCREEN_W - PADDLE_WIDTH // 2)
        self.state[S_PADDLE_X] = paddle_x

        # Fire (release ball)
        if action == 1 and self.state[S_BALL_STUCK] > 0.5:
            self.state[S_BALL_STUCK] = 0.0
            self.state[S_BALL_VX] = Scalar[Self.dtype](
                (random_float64() - 0.5) * 2.0 * BALL_SPEED_X
            )
            self.state[S_BALL_VY] = Scalar[Self.dtype](BALL_SPEED_Y)

        var reward = Scalar[Self.dtype](0.0)

        if self.state[S_BALL_STUCK] > 0.5:
            # Ball follows paddle
            self.state[S_BALL_X] = paddle_x
            self.state[S_BALL_Y] = Scalar[Self.dtype](PADDLE_Y - BALL_SIZE - 1)
            self.state[S_STEP_COUNT] += 1.0
            return (reward, self.done)

        # Move ball
        self.state[S_BALL_X] += self.state[S_BALL_VX]
        self.state[S_BALL_Y] += self.state[S_BALL_VY]

        var bx = self.state[S_BALL_X]
        var by = self.state[S_BALL_Y]

        # Wall bounce (left/right)
        if bx < 0:
            self.state[S_BALL_X] = 0.0
            self.state[S_BALL_VX] = -self.state[S_BALL_VX]
        elif bx > Scalar[Self.dtype](SCREEN_W):
            self.state[S_BALL_X] = Scalar[Self.dtype](SCREEN_W)
            self.state[S_BALL_VX] = -self.state[S_BALL_VX]

        # Top bounce
        if by < 0:
            self.state[S_BALL_Y] = 0.0
            self.state[S_BALL_VY] = -self.state[S_BALL_VY]

        # Bottom: lose life
        if by > Scalar[Self.dtype](SCREEN_H):
            self.state[S_LIVES] -= 1.0
            if self.state[S_LIVES] <= 0:
                self.done = True
                self.state[S_STEP_COUNT] += 1.0
                return (reward, self.done)
            # Reset ball on paddle
            self.state[S_BALL_STUCK] = 1.0
            self.state[S_BALL_X] = paddle_x
            self.state[S_BALL_Y] = Scalar[Self.dtype](PADDLE_Y - BALL_SIZE - 1)
            self.state[S_BALL_VX] = 0.0
            self.state[S_BALL_VY] = 0.0

        # Paddle collision
        bx = self.state[S_BALL_X]
        by = self.state[S_BALL_Y]
        if (
            by >= Scalar[Self.dtype](PADDLE_Y - PADDLE_HEIGHT)
            and by <= Scalar[Self.dtype](PADDLE_Y)
            and bx >= paddle_x - Scalar[Self.dtype](PADDLE_WIDTH // 2)
            and bx <= paddle_x + Scalar[Self.dtype](PADDLE_WIDTH // 2)
            and self.state[S_BALL_VY] > 0
        ):
            self.state[S_BALL_VY] = -self.state[S_BALL_VY]
            var hit_pos = (bx - paddle_x) / Scalar[Self.dtype](
                PADDLE_WIDTH // 2
            )
            self.state[S_BALL_VX] = hit_pos * Scalar[Self.dtype](MAX_BALL_VX)
            self.state[S_BALL_Y] = Scalar[Self.dtype](
                PADDLE_Y - PADDLE_HEIGHT - 1
            )

        # Brick collision
        bx = self.state[S_BALL_X]
        by = self.state[S_BALL_Y]
        for row in range(BRICK_ROWS):
            for col in range(BRICK_COLS):
                var idx = row * BRICK_COLS + col
                if self.state[S_BRICKS_START + idx] < 0.5:
                    continue
                var brick_x = Scalar[Self.dtype](
                    BRICK_LEFT + col * (BRICK_WIDTH + BRICK_GAP)
                )
                var brick_y = Scalar[Self.dtype](
                    BRICK_TOP + row * (BRICK_HEIGHT + BRICK_GAP)
                )
                if (
                    bx >= brick_x
                    and bx <= brick_x + Scalar[Self.dtype](BRICK_WIDTH)
                    and by >= brick_y
                    and by <= brick_y + Scalar[Self.dtype](BRICK_HEIGHT)
                ):
                    self.state[S_BRICKS_START + idx] = 0.0
                    self.state[S_BALL_VY] = -self.state[S_BALL_VY]
                    self.state[S_BRICKS_REM] -= 1.0
                    var scores = materialize[ROW_SCORES]()
                    var points = scores[row]
                    self.state[S_SCORE] += Scalar[Self.dtype](points)
                    reward += Scalar[Self.dtype](points)
                    break  # Only one brick per step

        # Check win
        if self.state[S_BRICKS_REM] <= 0:
            self.done = True

        self.state[S_STEP_COUNT] += 1.0
        if Int(self.state[S_STEP_COUNT]) >= BREAKOUT_MAX_STEPS:
            self.done = True

        return (reward, self.done)

    # ========================================================================
    # Trait methods
    # ========================================================================

    def get_state(mut self) -> ArcadeGameState:
        return ArcadeGameState(index=Int(self.state[S_STEP_COUNT]))

    def close(mut self):
        if self._renderer_initialized:
            self._renderer.value()[].close()
            self._renderer.value().unsafe_free()
            self._renderer_initialized = False

    def action_from_index(self, action_idx: Int) -> ArcadeGameAction:
        return ArcadeGameAction(value=action_idx)

    def num_actions(self) -> Int:
        return 4

    def obs_dim(self) -> Int:
        return 7

    def num_states(self) -> Int:
        return 1

    def state_to_index(self, state: ArcadeGameState) -> Int:
        return state.index

    def get_obs_list(self) -> List[Scalar[Self.dtype]]:
        var obs = List[Scalar[Self.dtype]](capacity=7)
        obs.append(self.state[S_BALL_X] / Scalar[Self.dtype](SCREEN_W))
        obs.append(self.state[S_BALL_Y] / Scalar[Self.dtype](SCREEN_H))
        obs.append(self.state[S_BALL_VX] / Scalar[Self.dtype](MAX_BALL_VX))
        obs.append(self.state[S_BALL_VY] / Scalar[Self.dtype](MAX_BALL_VY))
        obs.append(self.state[S_PADDLE_X] / Scalar[Self.dtype](SCREEN_W))
        obs.append(self.state[S_BRICKS_REM] / Scalar[Self.dtype](TOTAL_BRICKS))
        obs.append(self.state[S_LIVES] / Scalar[Self.dtype](INITIAL_LIVES))
        return obs^

    def reset_obs_list(mut self) -> List[Scalar[Self.dtype]]:
        _ = self.reset()
        return self.get_obs_list()

    def step_obs(
        mut self, action: Int
    ) -> Tuple[List[Scalar[Self.dtype]], Scalar[Self.dtype], Bool]:
        var result = self._step_impl(action)
        return (self.get_obs_list(), result[0], result[1])

    # ========================================================================
    # RenderableEnv
    # ========================================================================

    def init_renderer(mut self) raises -> Bool:
        if self._renderer_initialized:
            return True
        self._renderer = alloc[Renderer2D]({count = 1}).unsafe_leak()
        self._renderer.value().unsafe_write(Renderer2D())
        self._renderer_initialized = True
        return True

    def render_frame(mut self) raises -> None:
        if not self._renderer_initialized:
            return
        self._render(self._renderer.value()[])

    def _render(self, mut renderer: Renderer2D):
        """Render Breakout state — Atari-style dark theme."""
        var bg_color = SDL_Color(20, 20, 40, 255)
        if not renderer.begin_frame_with_color(bg_color):
            return

        var sw = renderer.screen_width
        var sh = renderer.screen_height
        var sx = Float64(sw) / Float64(SCREEN_W)
        var sy = Float64(sh) / Float64(SCREEN_H)

        # Colors
        var score_bar_color = SDL_Color(40, 40, 60, 255)
        var sep_color = SDL_Color(120, 120, 160, 255)
        var paddle_color = SDL_Color(180, 180, 220, 255)
        var ball_color = SDL_Color(240, 240, 255, 255)

        # Brick row colors (vivid, Atari-style)
        var c_red = SDL_Color(220, 50, 50, 255)
        var c_orange = SDL_Color(220, 130, 40, 255)
        var c_yellow = SDL_Color(200, 200, 40, 255)
        var c_green = SDL_Color(40, 200, 40, 255)
        var c_aqua = SDL_Color(40, 200, 200, 255)
        var c_blue = SDL_Color(60, 100, 220, 255)

        # -- Score area at top --
        var score_area_h = Int(24.0 * sy)
        renderer.draw_rect(0, 0, sw, score_area_h, score_bar_color)
        renderer.draw_rect(0, score_area_h, sw, max(1, Int(sy)), sep_color)

        # Score + lives text in score bar
        var score_color = SDL_Color(200, 200, 220, 255)
        var lives_color = SDL_Color(220, 80, 80, 255)
        var score_str = "SCORE: " + String(Int(self.state[S_SCORE]))
        var lives_str = "LIVES: " + String(Int(self.state[S_LIVES]))
        renderer.draw_text(
            score_str, Int(10.0 * sx), score_area_h // 2 - 7, score_color
        )
        renderer.draw_text(
            lives_str, sw - Int(80.0 * sx), score_area_h // 2 - 7, lives_color
        )

        # -- Bottom info bar --
        var info_h = max(1, Int(14.0 * sy))
        var info_y = sh - info_h
        renderer.draw_rect(0, info_y, sw, info_h, score_bar_color)
        renderer.draw_rect(0, info_y, sw, max(1, Int(sy)), sep_color)
        var info_color = SDL_Color(160, 160, 180, 255)
        var info_str = (
            "Bricks: "
            + String(Int(self.state[S_BRICKS_REM]))
            + "      Frame: "
            + String(Int(self.state[S_STEP_COUNT]))
        )
        renderer.draw_text(info_str, 8, info_y + 2, info_color)

        # Play area mapping: game [0..SCREEN_H] → screen [play_top..info_y]
        var play_top = score_area_h + max(1, Int(sy))
        var play_h = info_y - play_top

        # -- Draw bricks --
        for row in range(BRICK_ROWS):
            for col in range(BRICK_COLS):
                var idx = row * BRICK_COLS + col
                if self.state[S_BRICKS_START + idx] < 0.5:
                    continue
                var bx = BRICK_LEFT + col * (BRICK_WIDTH + BRICK_GAP)
                var by = BRICK_TOP + row * (BRICK_HEIGHT + BRICK_GAP)
                var color = c_red
                if row == 1:
                    color = c_orange
                elif row == 2:
                    color = c_yellow
                elif row == 3:
                    color = c_green
                elif row == 4:
                    color = c_aqua
                elif row == 5:
                    color = c_blue
                renderer.draw_rect(
                    Int(Float64(bx) * sx),
                    play_top
                    + Int(Float64(by) / Float64(SCREEN_H) * Float64(play_h)),
                    max(1, Int(Float64(BRICK_WIDTH) * sx)),
                    max(
                        1,
                        Int(
                            Float64(BRICK_HEIGHT)
                            / Float64(SCREEN_H)
                            * Float64(play_h)
                        ),
                    ),
                    color,
                )

        # -- Draw paddle --
        var px = Int(self.state[S_PADDLE_X])
        var pw = max(2, Int(Float64(PADDLE_WIDTH) * sx))
        var p_h = max(
            2, Int(Float64(PADDLE_HEIGHT) / Float64(SCREEN_H) * Float64(play_h))
        )
        renderer.draw_rect(
            Int(Float64(px - PADDLE_WIDTH // 2) * sx),
            play_top
            + Int(
                Float64(PADDLE_Y - PADDLE_HEIGHT)
                / Float64(SCREEN_H)
                * Float64(play_h)
            ),
            pw,
            p_h,
            paddle_color,
        )

        # -- Draw ball --
        var ball_x = Float64(Int(self.state[S_BALL_X]))
        var ball_y = Float64(Int(self.state[S_BALL_Y]))
        var bsz = max(4, Int(Float64(BALL_SIZE) * sx * 2.0))
        renderer.draw_rect(
            Int(ball_x * sx) - bsz // 2,
            play_top
            + Int(ball_y / Float64(SCREEN_H) * Float64(play_h))
            - bsz // 2,
            bsz,
            bsz,
            ball_color,
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

    def start_recording(
        mut self, filename: String, fps: Int = 30, skip: Int = 1
    ) raises:
        if not self._renderer_initialized:
            return
        self._renderer.value()[].start_recording(filename, fps, skip)

    def stop_recording(mut self) raises:
        if not self._renderer_initialized:
            return
        self._renderer.value()[].stop_recording()

    def renderer_delay(self, ms: Int) -> None:
        if not self._renderer_initialized:
            return
        self._renderer.value()[].renderer_delay(ms)

    def renderer_is_paused(self) -> Bool:
        return False

    def renderer_step_once(self) -> Bool:
        return False

    # ========================================================================
    # GPU Inline Kernels
    # ========================================================================

    @staticmethod
    @always_inline
    def step_kernel[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        states: Tile[gpu_dtype, BATCH_SIZE, STATE_SIZE],
        actions: ImmTile[gpu_dtype, BATCH_SIZE],
        rewards: Tile[gpu_dtype, BATCH_SIZE],
        dones: Tile[gpu_dtype, BATCH_SIZE],
        rng_seed: Scalar[DType.uint64],
    ):
        var i = Int(block_dim.x * block_idx.x + thread_idx.x)
        if i >= BATCH_SIZE:
            return

        var action = Int(actions[i])

        # Load state
        var bx = states[i, S_BALL_X]
        var by = states[i, S_BALL_Y]
        var bvx = states[i, S_BALL_VX]
        var bvy = states[i, S_BALL_VY]
        var paddle_x = states[i, S_PADDLE_X]
        var bricks_rem = states[i, S_BRICKS_REM]
        var lives = states[i, S_LIVES]
        var score = states[i, S_SCORE]
        var stuck = states[i, S_BALL_STUCK]
        var steps = states[i, S_STEP_COUNT]
        var reward = Scalar[gpu_dtype](0.0)

        # Move paddle
        if action == 2:  # LEFT
            paddle_x = paddle_x - Scalar[gpu_dtype](PADDLE_SPEED)
            if paddle_x < Scalar[gpu_dtype](PADDLE_WIDTH // 2):
                paddle_x = Scalar[gpu_dtype](PADDLE_WIDTH // 2)
        elif action == 3:  # RIGHT
            paddle_x = paddle_x + Scalar[gpu_dtype](PADDLE_SPEED)
            if paddle_x > Scalar[gpu_dtype](SCREEN_W - PADDLE_WIDTH // 2):
                paddle_x = Scalar[gpu_dtype](SCREEN_W - PADDLE_WIDTH // 2)

        # Fire
        if action == 1 and stuck > Scalar[gpu_dtype](0.5):
            stuck = 0.0
            var rng = PhiloxRandom(
                seed=UInt64(rng_seed) * UInt64(BATCH_SIZE) + UInt64(i), offset=0
            )
            var rand_vals = rng.step_uniform()
            bvx = Scalar[gpu_dtype](-BALL_SPEED_X) + Scalar[gpu_dtype](
                rand_vals[0]
            ) * Scalar[gpu_dtype](2.0 * BALL_SPEED_X)
            bvy = Scalar[gpu_dtype](BALL_SPEED_Y)

        var is_done = False

        if stuck > Scalar[gpu_dtype](0.5):
            bx = paddle_x
            by = Scalar[gpu_dtype](PADDLE_Y - BALL_SIZE - 1)
        else:
            # Move ball
            bx = bx + bvx
            by = by + bvy

            # Wall bounce
            if bx < 0:
                bx = Scalar[gpu_dtype](0)
                bvx = -bvx
            elif bx > Scalar[gpu_dtype](SCREEN_W):
                bx = Scalar[gpu_dtype](SCREEN_W)
                bvx = -bvx

            # Top bounce
            if by < 0:
                by = Scalar[gpu_dtype](0)
                bvy = -bvy

            # Bottom: lose life
            if by > Scalar[gpu_dtype](SCREEN_H):
                lives = lives - 1.0
                if lives <= 0:
                    is_done = True
                else:
                    stuck = 1.0
                    bx = paddle_x
                    by = Scalar[gpu_dtype](PADDLE_Y - BALL_SIZE - 1)
                    bvx = 0.0
                    bvy = 0.0

            # Paddle collision
            if (
                by >= Scalar[gpu_dtype](PADDLE_Y - PADDLE_HEIGHT)
                and by <= Scalar[gpu_dtype](PADDLE_Y)
                and bx >= paddle_x - Scalar[gpu_dtype](PADDLE_WIDTH // 2)
                and bx <= paddle_x + Scalar[gpu_dtype](PADDLE_WIDTH // 2)
                and bvy > 0
            ):
                bvy = -bvy
                var hit_pos = (bx - paddle_x) / Scalar[gpu_dtype](
                    PADDLE_WIDTH // 2
                )
                bvx = hit_pos * Scalar[gpu_dtype](MAX_BALL_VX)
                by = Scalar[gpu_dtype](PADDLE_Y - PADDLE_HEIGHT - 1)

            # Brick collision (check all bricks)
            for row in range(BRICK_ROWS):
                for col in range(BRICK_COLS):
                    var idx = row * BRICK_COLS + col
                    if states[i, S_BRICKS_START + idx] < Scalar[gpu_dtype](0.5):
                        continue
                    var brick_x = Scalar[gpu_dtype](
                        BRICK_LEFT + col * (BRICK_WIDTH + BRICK_GAP)
                    )
                    var brick_y = Scalar[gpu_dtype](
                        BRICK_TOP + row * (BRICK_HEIGHT + BRICK_GAP)
                    )
                    if (
                        bx >= brick_x
                        and bx <= brick_x + Scalar[gpu_dtype](BRICK_WIDTH)
                        and by >= brick_y
                        and by <= brick_y + Scalar[gpu_dtype](BRICK_HEIGHT)
                    ):
                        states[i, S_BRICKS_START + idx] = 0.0
                        bvy = -bvy
                        bricks_rem = bricks_rem - 1.0
                        # Row-based scoring (simplified: top=7, mid=4, bot=1)
                        var pts = Scalar[gpu_dtype](1.0)
                        if row < 2:
                            pts = 7.0
                        elif row < 4:
                            pts = 4.0
                        score = score + pts
                        reward = reward + pts
                        break  # One brick per step

        # Check win
        if bricks_rem <= 0:
            is_done = True

        steps = steps + 1.0
        if Int(steps) >= BREAKOUT_MAX_STEPS:
            is_done = True

        # Write state back
        states[i, S_BALL_X] = bx
        states[i, S_BALL_Y] = by
        states[i, S_BALL_VX] = bvx
        states[i, S_BALL_VY] = bvy
        states[i, S_PADDLE_X] = paddle_x
        states[i, S_BRICKS_REM] = bricks_rem
        states[i, S_LIVES] = lives
        states[i, S_SCORE] = score
        states[i, S_BALL_STUCK] = stuck
        states[i, S_STEP_COUNT] = steps

        rewards[i] = reward
        # `is_done` is a Bool, and `Scalar[float](Bool)` no longer compiles —
        # SIMD's Intable constructor now requires an integral dtype.
        dones[i] = Scalar[gpu_dtype](1.0) if is_done else Scalar[gpu_dtype](
            0.0
        )

    @staticmethod
    @always_inline
    def reset_kernel[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        state: Tile[gpu_dtype, BATCH_SIZE, STATE_SIZE],
    ):
        var i = Int(block_dim.x * block_idx.x + thread_idx.x)
        if i >= BATCH_SIZE:
            return

        state[i, S_BALL_X] = Scalar[gpu_dtype](SCREEN_W // 2)
        state[i, S_BALL_Y] = Scalar[gpu_dtype](PADDLE_Y - BALL_SIZE - 1)
        state[i, S_BALL_VX] = 0.0
        state[i, S_BALL_VY] = 0.0
        state[i, S_PADDLE_X] = Scalar[gpu_dtype](SCREEN_W // 2)
        state[i, S_BRICKS_REM] = Scalar[gpu_dtype](TOTAL_BRICKS)
        state[i, S_LIVES] = Scalar[gpu_dtype](INITIAL_LIVES)
        state[i, S_SCORE] = 0.0
        state[i, S_BALL_STUCK] = 1.0
        state[i, S_STEP_COUNT] = 0.0
        # All bricks alive
        for b in range(TOTAL_BRICKS):
            state[i, S_BRICKS_START + b] = 1.0
        # Zero reserved
        for r in range(52, 56):
            state[i, r] = 0.0

    @staticmethod
    @always_inline
    def selective_reset_kernel[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        state: Tile[gpu_dtype, BATCH_SIZE, STATE_SIZE],
        dones: Tile[gpu_dtype, BATCH_SIZE],
        rng_seed: Scalar[DType.uint32],
    ):
        var i = Int(block_dim.x * block_idx.x + thread_idx.x)
        if i >= BATCH_SIZE:
            return
        if dones[i] < Scalar[gpu_dtype](0.5):
            return

        state[i, S_BALL_X] = Scalar[gpu_dtype](SCREEN_W // 2)
        state[i, S_BALL_Y] = Scalar[gpu_dtype](PADDLE_Y - BALL_SIZE - 1)
        state[i, S_BALL_VX] = 0.0
        state[i, S_BALL_VY] = 0.0
        state[i, S_PADDLE_X] = Scalar[gpu_dtype](SCREEN_W // 2)
        state[i, S_BRICKS_REM] = Scalar[gpu_dtype](TOTAL_BRICKS)
        state[i, S_LIVES] = Scalar[gpu_dtype](INITIAL_LIVES)
        state[i, S_SCORE] = 0.0
        state[i, S_BALL_STUCK] = 1.0
        state[i, S_STEP_COUNT] = 0.0
        for b in range(TOTAL_BRICKS):
            state[i, S_BRICKS_START + b] = 1.0
        for r in range(52, 56):
            state[i, r] = 0.0
        dones[i] = 0.0

    # ========================================================================
    # GPU Launcher Methods
    # ========================================================================

    comptime TPB = 256

    @staticmethod
    def step_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        OBS_DIM: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[gpu_dtype],
        actions_buf: DeviceBuffer[gpu_dtype],
        mut rewards_buf: DeviceBuffer[gpu_dtype],
        mut dones_buf: DeviceBuffer[gpu_dtype],
        mut terminated_buf: DeviceBuffer[gpu_dtype],
        mut obs_buf: DeviceBuffer[gpu_dtype],
        rng_seed: UInt64 = 0,
        workspace_ptr: Optional[
            Pointer[Scalar[gpu_dtype], MutAnyOrigin]
        ] = None,
        rng_counter_ptr: Optional[
            Pointer[Scalar[DType.uint64], MutAnyOrigin]
        ] = None,
    ) raises:
        var states = tile[BATCH_SIZE, STATE_SIZE](states_buf)
        var actions = tile[BATCH_SIZE](actions_buf)
        var rewards = tile[BATCH_SIZE](rewards_buf)
        var dones = tile[BATCH_SIZE](dones_buf)
        var terminated_out = tile[BATCH_SIZE](terminated_buf)
        var obs = tile[BATCH_SIZE, OBS_DIM](obs_buf)

        comptime BLOCKS = (BATCH_SIZE + Self.TPB - 1) // Self.TPB
        var seed = Scalar[DType.uint64](rng_seed)

        @always_inline
        def step_wrapper(
            states: Tile[gpu_dtype, BATCH_SIZE, STATE_SIZE],
            actions: ImmTile[gpu_dtype, BATCH_SIZE],
            rewards: Tile[gpu_dtype, BATCH_SIZE],
            dones: Tile[gpu_dtype, BATCH_SIZE],
            terminated_out: Tile[gpu_dtype, BATCH_SIZE],
            obs: Tile[gpu_dtype, BATCH_SIZE, OBS_DIM],
            rng_seed: Scalar[DType.uint64],
        ):
            Self.step_kernel[BATCH_SIZE, STATE_SIZE](
                states, actions, rewards, dones, rng_seed
            )
            var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
            if idx < BATCH_SIZE:
                terminated_out[idx] = dones[idx]
                obs[idx, 0] = states[idx, S_BALL_X] / Scalar[gpu_dtype](
                    SCREEN_W
                )
                obs[idx, 1] = states[idx, S_BALL_Y] / Scalar[gpu_dtype](
                    SCREEN_H
                )
                obs[idx, 2] = states[idx, S_BALL_VX] / Scalar[gpu_dtype](
                    MAX_BALL_VX
                )
                obs[idx, 3] = states[idx, S_BALL_VY] / Scalar[gpu_dtype](
                    MAX_BALL_VY
                )
                obs[idx, 4] = states[idx, S_PADDLE_X] / Scalar[gpu_dtype](
                    SCREEN_W
                )
                obs[idx, 5] = states[idx, S_BRICKS_REM] / Scalar[gpu_dtype](
                    TOTAL_BRICKS
                )
                obs[idx, 6] = states[idx, S_LIVES] / Scalar[gpu_dtype](
                    INITIAL_LIVES
                )

        ctx.enqueue_function[step_wrapper](
            states,
            actions,
            rewards,
            dones,
            terminated_out,
            obs,
            seed,
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
        states_buf: DeviceBuffer[gpu_dtype],
        mut obs_buf: DeviceBuffer[gpu_dtype],
    ) raises:
        """Seed `obs` from `state` with the SAME normalization the step
        kernel applies. Overrides the trait default (raw state-prefix copy),
        which mismatches the normalized step obs and corrupts the batched-env
        replay (prev_obs raw vs next_obs normalized → uniform collapse). Keep
        in lock-step with the obs block in `step_kernel_gpu`."""
        # `states_buf` read-only here → mut=False view widens into ImmutAnyOrigin.
        var states = tile[BATCH_SIZE, STATE_SIZE](states_buf)
        var obs = tile[BATCH_SIZE, OBS_DIM](obs_buf)
        comptime BLOCKS = (BATCH_SIZE + Self.TPB - 1) // Self.TPB

        @always_inline
        def extract_wrapper(
            states: ImmTile[gpu_dtype, BATCH_SIZE, STATE_SIZE],
            obs: Tile[gpu_dtype, BATCH_SIZE, OBS_DIM],
        ):
            var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
            if idx < BATCH_SIZE:
                obs[idx, 0] = states[idx, S_BALL_X] / Scalar[gpu_dtype](
                    SCREEN_W
                )
                obs[idx, 1] = states[idx, S_BALL_Y] / Scalar[gpu_dtype](
                    SCREEN_H
                )
                obs[idx, 2] = states[idx, S_BALL_VX] / Scalar[gpu_dtype](
                    MAX_BALL_VX
                )
                obs[idx, 3] = states[idx, S_BALL_VY] / Scalar[gpu_dtype](
                    MAX_BALL_VY
                )
                obs[idx, 4] = states[idx, S_PADDLE_X] / Scalar[gpu_dtype](
                    SCREEN_W
                )
                obs[idx, 5] = states[idx, S_BRICKS_REM] / Scalar[gpu_dtype](
                    TOTAL_BRICKS
                )
                obs[idx, 6] = states[idx, S_LIVES] / Scalar[gpu_dtype](
                    INITIAL_LIVES
                )

        ctx.enqueue_function[extract_wrapper](
            states, obs, grid_dim=(BLOCKS,), block_dim=(Self.TPB,),
        )

    @staticmethod
    def reset_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[gpu_dtype],
        rng_seed: UInt64 = 0,
    ) raises:
        var states = tile[BATCH_SIZE, STATE_SIZE](states_buf)
        comptime BLOCKS = (BATCH_SIZE + Self.TPB - 1) // Self.TPB

        @always_inline
        def reset_wrapper(
            states: Tile[gpu_dtype, BATCH_SIZE, STATE_SIZE],
        ):
            Self.reset_kernel[BATCH_SIZE, STATE_SIZE](states)

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
        mut states_buf: DeviceBuffer[gpu_dtype],
        mut dones_buf: DeviceBuffer[gpu_dtype],
        rng_seed: UInt64,
        workspace_ptr: Optional[
            Pointer[Scalar[gpu_dtype], MutAnyOrigin]
        ] = None,
        rng_counter_ptr: Optional[
            Pointer[Scalar[DType.uint64], MutAnyOrigin]
        ] = None,
    ) raises:
        var states = tile[BATCH_SIZE, STATE_SIZE](states_buf)
        var dones = tile[BATCH_SIZE](dones_buf)
        comptime BLOCKS = (BATCH_SIZE + Self.TPB - 1) // Self.TPB

        if Bool(rng_counter_ptr):
            var counter_t = tile[1](rng_counter_ptr.value())

            @always_inline
            def sel_reset_counter_wrapper(
                states: Tile[gpu_dtype, BATCH_SIZE, STATE_SIZE],
                dones: Tile[gpu_dtype, BATCH_SIZE],
                counter: Tile[DType.uint64, 1],
            ):
                Self.selective_reset_kernel[BATCH_SIZE, STATE_SIZE](
                    states,
                    dones,
                    Scalar[DType.uint32](
                        rebind[Scalar[DType.uint64]](counter[0])
                    ),
                )

            ctx.enqueue_function[sel_reset_counter_wrapper](
                states,
                dones,
                counter_t,
                grid_dim=(BLOCKS,),
                block_dim=(Self.TPB,),
            )
        else:
            var seed = Scalar[DType.uint64](rng_seed)

            @always_inline
            def sel_reset_wrapper(
                states: Tile[gpu_dtype, BATCH_SIZE, STATE_SIZE],
                dones: Tile[gpu_dtype, BATCH_SIZE],
                rng_seed: Scalar[DType.uint64],
            ):
                Self.selective_reset_kernel[BATCH_SIZE, STATE_SIZE](
                    states, dones, Scalar[DType.uint32](rng_seed)
                )

            ctx.enqueue_function[sel_reset_wrapper](
                states,
                dones,
                seed,
                grid_dim=(BLOCKS,),
                block_dim=(Self.TPB,),
            )

    @staticmethod
    def init_step_workspace_gpu[
        BATCH_SIZE: Int,
    ](ctx: DeviceContext, mut workspace_buf: DeviceBuffer[gpu_dtype]) raises:
        pass

    @staticmethod
    def update_curriculum_gpu(
        ctx: DeviceContext,
        mut workspace_buf: DeviceBuffer[gpu_dtype],
        curriculum_values: List[Scalar[gpu_dtype]],
    ) raises:
        pass
