"""Native Space Invaders — CPU+GPU environment for RL training.

Ship + alien formation + bullets. No Atari emulation.
Follows the CartPole/Pong pattern.

State layout (STATE_SIZE = 80):
  [0]  ship_x              \
  [1]  bullet_x             |
  [2]  bullet_y             | CLEAN_OBS_DIM = 8
  [3]  bullet_active        |
  [4]  alien_bullet_x       |
  [5]  alien_bullet_y       |
  [6]  alien_bullet_active  |
  [7]  aliens_remaining    /
  [8..62]  alien_alive (5 rows × 11 cols = 55 floats)
  [63] alien_shift_x
  [64] alien_shift_y
  [65] alien_direction  (+1 or -1)
  [66] alien_move_timer
  [67] score
  [68] lives
  [69] step_count
  [70..79] reserved

Actions: 0=NOOP, 1=LEFT, 2=RIGHT, 3=FIRE
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
# Constants
# ============================================================================

comptime SHIP_WIDTH: Int = 12
comptime SHIP_Y: Int = 190
comptime SHIP_SPEED: Float64 = 2.0
comptime BULLET_SPEED: Float64 = -4.0  # Upward
comptime ALIEN_BULLET_SPEED: Float64 = 2.0  # Downward

comptime ALIEN_ROWS: Int = 5
comptime ALIEN_COLS: Int = 11
comptime TOTAL_ALIENS: Int = ALIEN_ROWS * ALIEN_COLS  # 55
comptime ALIEN_WIDTH: Int = 10
comptime ALIEN_HEIGHT: Int = 8
comptime ALIEN_GAP_X: Int = 3
comptime ALIEN_GAP_Y: Int = 4
comptime ALIEN_TOP: Int = 30
comptime ALIEN_LEFT: Int = 10
comptime ALIEN_MOVE_INTERVAL: Int = 8  # Frames between moves
comptime ALIEN_SHIFT_X: Float64 = 2.0
comptime ALIEN_DROP_Y: Float64 = 6.0
comptime ALIEN_FIRE_CHANCE: Float64 = 0.02  # Per step probability

comptime INITIAL_LIVES: Int = 3
comptime SI_MAX_STEPS: Int = 10000

# Row scores: top rows worth more
comptime ROW_SCORES: Array[Int, 5] = [30, 20, 20, 10, 10]

# State indices
comptime S_SHIP_X: Int = 0
comptime S_BULLET_X: Int = 1
comptime S_BULLET_Y: Int = 2
comptime S_BULLET_ACTIVE: Int = 3
comptime S_ABUL_X: Int = 4
comptime S_ABUL_Y: Int = 5
comptime S_ABUL_ACTIVE: Int = 6
comptime S_ALIENS_REM: Int = 7
comptime S_ALIENS_START: Int = 8  # 55 alien slots
comptime S_ALIEN_SX: Int = 63
comptime S_ALIEN_SY: Int = 64
comptime S_ALIEN_DIR: Int = 65
comptime S_ALIEN_TIMER: Int = 66
comptime S_SCORE: Int = 67
comptime S_LIVES: Int = 68
comptime S_STEP_COUNT: Int = 69

# ============================================================================
# SpaceInvadersEnv
# ============================================================================


struct SpaceInvadersEnv[DTYPE: DType](
    BoxDiscreteActionEnv & GPUDiscreteEnv & RenderableEnv
):
    """Native Space Invaders — CPU+GPU dual path."""

    comptime dtype = Self.DTYPE
    comptime StateType = ArcadeGameState
    comptime ActionType = ArcadeGameAction

    comptime STATE_SIZE: Int = 80
    comptime OBS_DIM: Int = 8
    comptime NUM_ACTIONS: Int = 4  # NOOP, LEFT, RIGHT, FIRE
    comptime STEP_WS_SHARED: Int = 0
    comptime STEP_WS_PER_ENV: Int = 0

    var state: Array[Scalar[Self.dtype], 80]
    var done: Bool
    var _rng_counter: UInt32

    var _renderer: Optional[Pointer[Renderer2D, MutUntrackedOrigin]]
    var _renderer_initialized: Bool

    def __init__(out self):
        self.state = Array[Scalar[Self.dtype], 80](
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
        self.state[S_SHIP_X] = Scalar[Self.dtype](SCREEN_W // 2)
        self.state[S_BULLET_ACTIVE] = 0.0
        self.state[S_ABUL_ACTIVE] = 0.0
        self.state[S_ALIENS_REM] = Scalar[Self.dtype](TOTAL_ALIENS)
        for a in range(TOTAL_ALIENS):
            self.state[S_ALIENS_START + a] = 1.0
        self.state[S_ALIEN_SX] = 0.0
        self.state[S_ALIEN_SY] = 0.0
        self.state[S_ALIEN_DIR] = 1.0
        self.state[S_ALIEN_TIMER] = Scalar[Self.dtype](ALIEN_MOVE_INTERVAL)
        self.state[S_SCORE] = 0.0
        self.state[S_LIVES] = Scalar[Self.dtype](INITIAL_LIVES)
        self.state[S_STEP_COUNT] = 0.0
        for r in range(70, 80):
            self.state[r] = 0.0
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
        var reward = Scalar[Self.dtype](0.0)

        # Move ship
        if action == 1:  # LEFT
            self.state[S_SHIP_X] -= Scalar[Self.dtype](SHIP_SPEED)
            if self.state[S_SHIP_X] < Scalar[Self.dtype](SHIP_WIDTH // 2):
                self.state[S_SHIP_X] = Scalar[Self.dtype](SHIP_WIDTH // 2)
        elif action == 2:  # RIGHT
            self.state[S_SHIP_X] += Scalar[Self.dtype](SHIP_SPEED)
            if self.state[S_SHIP_X] > Scalar[Self.dtype](
                SCREEN_W - SHIP_WIDTH // 2
            ):
                self.state[S_SHIP_X] = Scalar[Self.dtype](
                    SCREEN_W - SHIP_WIDTH // 2
                )

        # Fire
        if action == 3 and self.state[S_BULLET_ACTIVE] < 0.5:
            self.state[S_BULLET_ACTIVE] = 1.0
            self.state[S_BULLET_X] = self.state[S_SHIP_X]
            self.state[S_BULLET_Y] = Scalar[Self.dtype](SHIP_Y - 4)

        # Move player bullet
        if self.state[S_BULLET_ACTIVE] > 0.5:
            self.state[S_BULLET_Y] += Scalar[Self.dtype](BULLET_SPEED)
            if self.state[S_BULLET_Y] < 0:
                self.state[S_BULLET_ACTIVE] = 0.0

        # Move alien bullet
        if self.state[S_ABUL_ACTIVE] > 0.5:
            self.state[S_ABUL_Y] += Scalar[Self.dtype](ALIEN_BULLET_SPEED)
            if self.state[S_ABUL_Y] > Scalar[Self.dtype](SCREEN_H):
                self.state[S_ABUL_ACTIVE] = 0.0

        # Alien formation movement
        self.state[S_ALIEN_TIMER] -= 1.0
        if self.state[S_ALIEN_TIMER] <= 0:
            self.state[S_ALIEN_TIMER] = Scalar[Self.dtype](ALIEN_MOVE_INTERVAL)
            self.state[S_ALIEN_SX] += self.state[S_ALIEN_DIR] * Scalar[
                Self.dtype
            ](ALIEN_SHIFT_X)
            # Check bounds: reverse direction + drop
            if self.state[S_ALIEN_SX] > Scalar[Self.dtype](
                SCREEN_W - ALIEN_LEFT - ALIEN_COLS * (ALIEN_WIDTH + ALIEN_GAP_X)
            ) or self.state[S_ALIEN_SX] < Scalar[Self.dtype](-ALIEN_LEFT):
                self.state[S_ALIEN_DIR] = -self.state[S_ALIEN_DIR]
                self.state[S_ALIEN_SY] += Scalar[Self.dtype](ALIEN_DROP_Y)

        # Alien firing (random)
        if (
            self.state[S_ABUL_ACTIVE] < 0.5
            and random_float64() < ALIEN_FIRE_CHANCE
        ):
            # Pick a random alive alien in the bottom row
            for row in range(ALIEN_ROWS - 1, -1, -1):
                for col in range(ALIEN_COLS):
                    var idx = row * ALIEN_COLS + col
                    if self.state[S_ALIENS_START + idx] > 0.5:
                        self.state[S_ABUL_ACTIVE] = 1.0
                        self.state[S_ABUL_X] = (
                            Scalar[Self.dtype](
                                ALIEN_LEFT + col * (ALIEN_WIDTH + ALIEN_GAP_X)
                            )
                            + self.state[S_ALIEN_SX]
                            + Scalar[Self.dtype](ALIEN_WIDTH // 2)
                        )
                        self.state[S_ABUL_Y] = (
                            Scalar[Self.dtype](
                                ALIEN_TOP + row * (ALIEN_HEIGHT + ALIEN_GAP_Y)
                            )
                            + self.state[S_ALIEN_SY]
                            + Scalar[Self.dtype](ALIEN_HEIGHT)
                        )
                        break
                if self.state[S_ABUL_ACTIVE] > 0.5:
                    break

        # Check bullet-alien collision
        if self.state[S_BULLET_ACTIVE] > 0.5:
            var bul_x = self.state[S_BULLET_X]
            var bul_y = self.state[S_BULLET_Y]
            for row in range(ALIEN_ROWS):
                for col in range(ALIEN_COLS):
                    var idx = row * ALIEN_COLS + col
                    if self.state[S_ALIENS_START + idx] < 0.5:
                        continue
                    var ax = (
                        Scalar[Self.dtype](
                            ALIEN_LEFT + col * (ALIEN_WIDTH + ALIEN_GAP_X)
                        )
                        + self.state[S_ALIEN_SX]
                    )
                    var ay = (
                        Scalar[Self.dtype](
                            ALIEN_TOP + row * (ALIEN_HEIGHT + ALIEN_GAP_Y)
                        )
                        + self.state[S_ALIEN_SY]
                    )
                    if (
                        bul_x >= ax
                        and bul_x <= ax + Scalar[Self.dtype](ALIEN_WIDTH)
                        and bul_y >= ay
                        and bul_y <= ay + Scalar[Self.dtype](ALIEN_HEIGHT)
                    ):
                        self.state[S_ALIENS_START + idx] = 0.0
                        self.state[S_BULLET_ACTIVE] = 0.0
                        self.state[S_ALIENS_REM] -= 1.0
                        var scores = materialize[ROW_SCORES]()
                        var pts = scores[row]
                        self.state[S_SCORE] += Scalar[Self.dtype](pts)
                        reward += Scalar[Self.dtype](pts)
                        break
                if self.state[S_BULLET_ACTIVE] < 0.5:
                    break

        # Check alien bullet hits ship
        if self.state[S_ABUL_ACTIVE] > 0.5:
            var abx = self.state[S_ABUL_X]
            var aby = self.state[S_ABUL_Y]
            var sx = self.state[S_SHIP_X]
            if (
                aby >= Scalar[Self.dtype](SHIP_Y - 4)
                and aby <= Scalar[Self.dtype](SHIP_Y + 4)
                and abx >= sx - Scalar[Self.dtype](SHIP_WIDTH // 2)
                and abx <= sx + Scalar[Self.dtype](SHIP_WIDTH // 2)
            ):
                self.state[S_ABUL_ACTIVE] = 0.0
                self.state[S_LIVES] -= 1.0
                if self.state[S_LIVES] <= 0:
                    self.done = True

        # Check aliens reached bottom
        var lowest_alien_y = self.state[S_ALIEN_SY] + Scalar[Self.dtype](
            ALIEN_TOP
            + (ALIEN_ROWS - 1) * (ALIEN_HEIGHT + ALIEN_GAP_Y)
            + ALIEN_HEIGHT
        )
        if lowest_alien_y >= Scalar[Self.dtype](SHIP_Y):
            self.done = True

        # Check win
        if self.state[S_ALIENS_REM] <= 0:
            self.done = True

        self.state[S_STEP_COUNT] += 1.0
        if Int(self.state[S_STEP_COUNT]) >= SI_MAX_STEPS:
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
        return 8

    def num_states(self) -> Int:
        return 1

    def state_to_index(self, state: ArcadeGameState) -> Int:
        return state.index

    def get_obs_list(self) -> List[Scalar[Self.dtype]]:
        var obs = List[Scalar[Self.dtype]](capacity=8)
        obs.append(self.state[S_SHIP_X] / Scalar[Self.dtype](SCREEN_W))
        obs.append(self.state[S_BULLET_X] / Scalar[Self.dtype](SCREEN_W))
        obs.append(self.state[S_BULLET_Y] / Scalar[Self.dtype](SCREEN_H))
        obs.append(self.state[S_BULLET_ACTIVE])
        obs.append(self.state[S_ABUL_X] / Scalar[Self.dtype](SCREEN_W))
        obs.append(self.state[S_ABUL_Y] / Scalar[Self.dtype](SCREEN_H))
        obs.append(self.state[S_ABUL_ACTIVE])
        obs.append(self.state[S_ALIENS_REM] / Scalar[Self.dtype](TOTAL_ALIENS))
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
        """Render Space Invaders — Atari-style dark theme."""
        var bg_color = SDL_Color(10, 10, 30, 255)
        if not renderer.begin_frame_with_color(bg_color):
            return

        var sw = renderer.screen_width
        var sh = renderer.screen_height
        var sx = Float64(sw) / Float64(SCREEN_W)
        var sy = Float64(sh) / Float64(SCREEN_H)

        # Colors
        var score_bar_color = SDL_Color(30, 30, 50, 255)
        var sep_color = SDL_Color(100, 100, 140, 255)
        var ship_color = SDL_Color(60, 220, 60, 255)
        var bullet_color = SDL_Color(220, 220, 255, 255)
        var alien_bullet_color = SDL_Color(255, 60, 60, 255)
        # Alien row colors (top=most dangerous)
        var a_c0 = SDL_Color(255, 60, 60, 255)  # red (30 pts)
        var a_c1 = SDL_Color(255, 160, 60, 255)  # orange (20 pts)
        var a_c2 = SDL_Color(255, 255, 60, 255)  # yellow (20 pts)
        var a_c3 = SDL_Color(60, 255, 120, 255)  # green (10 pts)
        var a_c4 = SDL_Color(60, 200, 255, 255)  # cyan (10 pts)

        # -- Score area at top --
        var score_area_h = Int(24.0 * sy)
        renderer.draw_rect(0, 0, sw, score_area_h, score_bar_color)
        renderer.draw_rect(0, score_area_h, sw, max(1, Int(sy)), sep_color)

        var score_color = SDL_Color(200, 200, 220, 255)
        var lives_color = SDL_Color(60, 220, 60, 255)
        renderer.draw_text(
            "SCORE: " + String(Int(self.state[S_SCORE])),
            Int(10.0 * sx),
            score_area_h // 2 - 7,
            score_color,
        )
        renderer.draw_text(
            "LIVES: " + String(Int(self.state[S_LIVES])),
            sw - Int(80.0 * sx),
            score_area_h // 2 - 7,
            lives_color,
        )

        # -- Bottom info bar --
        var info_h = max(1, Int(14.0 * sy))
        var info_y = sh - info_h
        renderer.draw_rect(0, info_y, sw, info_h, score_bar_color)
        renderer.draw_rect(0, info_y, sw, max(1, Int(sy)), sep_color)
        var info_color = SDL_Color(160, 160, 180, 255)
        renderer.draw_text(
            "Aliens: "
            + String(Int(self.state[S_ALIENS_REM]))
            + "      Frame: "
            + String(Int(self.state[S_STEP_COUNT])),
            8,
            info_y + 2,
            info_color,
        )

        # Play area: game [0..SCREEN_H] → screen [play_top..info_y]
        var play_top = score_area_h + max(1, Int(sy))
        var play_h = info_y - play_top

        # Helper to map game Y to screen Y
        @always_inline
        def gy(game_y: Float64) {imm} -> Int:
            return play_top + Int(game_y / Float64(SCREEN_H) * Float64(play_h))

        # -- Draw aliens --
        for row in range(ALIEN_ROWS):
            var color = a_c0
            if row == 1:
                color = a_c1
            elif row == 2:
                color = a_c2
            elif row == 3:
                color = a_c3
            elif row == 4:
                color = a_c4
            for col in range(ALIEN_COLS):
                var idx = row * ALIEN_COLS + col
                if self.state[S_ALIENS_START + idx] < 0.5:
                    continue
                var ax_f = Float64(
                    ALIEN_LEFT + col * (ALIEN_WIDTH + ALIEN_GAP_X)
                ) + Float64(self.state[S_ALIEN_SX])
                var ay_f = Float64(
                    ALIEN_TOP + row * (ALIEN_HEIGHT + ALIEN_GAP_Y)
                ) + Float64(self.state[S_ALIEN_SY])
                renderer.draw_rect(
                    Int(ax_f * sx),
                    gy(ay_f),
                    max(1, Int(Float64(ALIEN_WIDTH) * sx)),
                    max(
                        1,
                        Int(
                            Float64(ALIEN_HEIGHT)
                            / Float64(SCREEN_H)
                            * Float64(play_h)
                        ),
                    ),
                    color,
                )

        # -- Draw ship --
        var ship_x = Float64(Int(self.state[S_SHIP_X]))
        var ship_w = max(2, Int(Float64(SHIP_WIDTH) * sx))
        var ship_h = max(2, Int(8.0 / Float64(SCREEN_H) * Float64(play_h)))
        renderer.draw_rect(
            Int((ship_x - Float64(SHIP_WIDTH // 2)) * sx),
            gy(Float64(SHIP_Y - 4)),
            ship_w,
            ship_h,
            ship_color,
        )

        # -- Player bullet --
        if self.state[S_BULLET_ACTIVE] > 0.5:
            var bul_w = max(2, Int(2.0 * sx))
            var bul_h = max(2, Int(6.0 / Float64(SCREEN_H) * Float64(play_h)))
            renderer.draw_rect(
                Int(Float64(self.state[S_BULLET_X]) * sx) - bul_w // 2,
                gy(Float64(self.state[S_BULLET_Y])),
                bul_w,
                bul_h,
                bullet_color,
            )

        # -- Alien bullet --
        if self.state[S_ABUL_ACTIVE] > 0.5:
            var abul_w = max(2, Int(2.0 * sx))
            var abul_h = max(2, Int(6.0 / Float64(SCREEN_H) * Float64(play_h)))
            renderer.draw_rect(
                Int(Float64(self.state[S_ABUL_X]) * sx) - abul_w // 2,
                gy(Float64(self.state[S_ABUL_Y])),
                abul_w,
                abul_h,
                alien_bullet_color,
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
        var ship_x = states[i, S_SHIP_X]
        var bul_x = states[i, S_BULLET_X]
        var bul_y = states[i, S_BULLET_Y]
        var bul_active = states[i, S_BULLET_ACTIVE]
        var abul_x = states[i, S_ABUL_X]
        var abul_y = states[i, S_ABUL_Y]
        var abul_active = states[i, S_ABUL_ACTIVE]
        var aliens_rem = states[i, S_ALIENS_REM]
        var alien_sx = states[i, S_ALIEN_SX]
        var alien_sy = states[i, S_ALIEN_SY]
        var alien_dir = states[i, S_ALIEN_DIR]
        var alien_timer = states[i, S_ALIEN_TIMER]
        var score = states[i, S_SCORE]
        var lives = states[i, S_LIVES]
        var steps = states[i, S_STEP_COUNT]
        var reward = Scalar[gpu_dtype](0.0)
        var is_done = False

        # Move ship
        if action == 1:  # LEFT
            ship_x = ship_x - Scalar[gpu_dtype](SHIP_SPEED)
            if ship_x < Scalar[gpu_dtype](SHIP_WIDTH // 2):
                ship_x = Scalar[gpu_dtype](SHIP_WIDTH // 2)
        elif action == 2:  # RIGHT
            ship_x = ship_x + Scalar[gpu_dtype](SHIP_SPEED)
            if ship_x > Scalar[gpu_dtype](SCREEN_W - SHIP_WIDTH // 2):
                ship_x = Scalar[gpu_dtype](SCREEN_W - SHIP_WIDTH // 2)

        # Fire
        if action == 3 and bul_active < Scalar[gpu_dtype](0.5):
            bul_active = 1.0
            bul_x = ship_x
            bul_y = Scalar[gpu_dtype](SHIP_Y - 4)

        # Move player bullet
        if bul_active > Scalar[gpu_dtype](0.5):
            bul_y = bul_y + Scalar[gpu_dtype](BULLET_SPEED)
            if bul_y < 0:
                bul_active = 0.0

        # Move alien bullet
        if abul_active > Scalar[gpu_dtype](0.5):
            abul_y = abul_y + Scalar[gpu_dtype](ALIEN_BULLET_SPEED)
            if abul_y > Scalar[gpu_dtype](SCREEN_H):
                abul_active = 0.0

        # Alien formation movement
        alien_timer = alien_timer - 1.0
        if alien_timer <= 0:
            alien_timer = Scalar[gpu_dtype](ALIEN_MOVE_INTERVAL)
            alien_sx = alien_sx + alien_dir * Scalar[gpu_dtype](ALIEN_SHIFT_X)
            if alien_sx > Scalar[gpu_dtype](
                SCREEN_W - ALIEN_LEFT - ALIEN_COLS * (ALIEN_WIDTH + ALIEN_GAP_X)
            ) or alien_sx < Scalar[gpu_dtype](-ALIEN_LEFT):
                alien_dir = -alien_dir
                alien_sy = alien_sy + Scalar[gpu_dtype](ALIEN_DROP_Y)

        # Alien firing (deterministic based on rng)
        if abul_active < Scalar[gpu_dtype](0.5):
            var rng = PhiloxRandom(
                seed=UInt64(rng_seed) * UInt64(BATCH_SIZE)
                + UInt64(i)
                + UInt64(steps) * UInt64(1000003),
                offset=0,
            )
            var rand_vals = rng.step_uniform()
            if Scalar[gpu_dtype](rand_vals[0]) < Scalar[gpu_dtype](
                ALIEN_FIRE_CHANCE
            ):
                # Find a bottom-row alive alien
                for row in range(ALIEN_ROWS - 1, -1, -1):
                    for col in range(ALIEN_COLS):
                        var idx = row * ALIEN_COLS + col
                        if states[i, S_ALIENS_START + idx] > Scalar[gpu_dtype](
                            0.5
                        ):
                            abul_active = 1.0
                            abul_x = (
                                Scalar[gpu_dtype](
                                    ALIEN_LEFT
                                    + col * (ALIEN_WIDTH + ALIEN_GAP_X)
                                )
                                + alien_sx
                                + Scalar[gpu_dtype](ALIEN_WIDTH // 2)
                            )
                            abul_y = (
                                Scalar[gpu_dtype](
                                    ALIEN_TOP
                                    + row * (ALIEN_HEIGHT + ALIEN_GAP_Y)
                                )
                                + alien_sy
                                + Scalar[gpu_dtype](ALIEN_HEIGHT)
                            )
                            break
                    if abul_active > Scalar[gpu_dtype](0.5):
                        break

        # Bullet-alien collision
        if bul_active > Scalar[gpu_dtype](0.5):
            for row in range(ALIEN_ROWS):
                for col in range(ALIEN_COLS):
                    var idx = row * ALIEN_COLS + col
                    if states[i, S_ALIENS_START + idx] < Scalar[gpu_dtype](0.5):
                        continue
                    var ax = (
                        Scalar[gpu_dtype](
                            ALIEN_LEFT + col * (ALIEN_WIDTH + ALIEN_GAP_X)
                        )
                        + alien_sx
                    )
                    var ay = (
                        Scalar[gpu_dtype](
                            ALIEN_TOP + row * (ALIEN_HEIGHT + ALIEN_GAP_Y)
                        )
                        + alien_sy
                    )
                    if (
                        bul_x >= ax
                        and bul_x <= ax + Scalar[gpu_dtype](ALIEN_WIDTH)
                        and bul_y >= ay
                        and bul_y <= ay + Scalar[gpu_dtype](ALIEN_HEIGHT)
                    ):
                        states[i, S_ALIENS_START + idx] = 0.0
                        bul_active = 0.0
                        aliens_rem = aliens_rem - 1.0
                        # Simplified scoring: top=30, mid=20, bot=10
                        var pts = Scalar[gpu_dtype](10.0)
                        if row == 0:
                            pts = 30.0
                        elif row < 3:
                            pts = 20.0
                        score = score + pts
                        reward = reward + pts
                        break
                if bul_active < Scalar[gpu_dtype](0.5):
                    break

        # Alien bullet hits ship
        if abul_active > Scalar[gpu_dtype](0.5):
            if (
                abul_y >= Scalar[gpu_dtype](SHIP_Y - 4)
                and abul_y <= Scalar[gpu_dtype](SHIP_Y + 4)
                and abul_x >= ship_x - Scalar[gpu_dtype](SHIP_WIDTH // 2)
                and abul_x <= ship_x + Scalar[gpu_dtype](SHIP_WIDTH // 2)
            ):
                abul_active = 0.0
                lives = lives - 1.0
                if lives <= 0:
                    is_done = True

        # Check aliens reached bottom
        if alien_sy + Scalar[gpu_dtype](
            ALIEN_TOP
            + (ALIEN_ROWS - 1) * (ALIEN_HEIGHT + ALIEN_GAP_Y)
            + ALIEN_HEIGHT
        ) >= Scalar[gpu_dtype](SHIP_Y):
            is_done = True

        # Win
        if aliens_rem <= 0:
            is_done = True

        steps = steps + 1.0
        if Int(steps) >= SI_MAX_STEPS:
            is_done = True

        # Write back
        states[i, S_SHIP_X] = ship_x
        states[i, S_BULLET_X] = bul_x
        states[i, S_BULLET_Y] = bul_y
        states[i, S_BULLET_ACTIVE] = bul_active
        states[i, S_ABUL_X] = abul_x
        states[i, S_ABUL_Y] = abul_y
        states[i, S_ABUL_ACTIVE] = abul_active
        states[i, S_ALIENS_REM] = aliens_rem
        states[i, S_ALIEN_SX] = alien_sx
        states[i, S_ALIEN_SY] = alien_sy
        states[i, S_ALIEN_DIR] = alien_dir
        states[i, S_ALIEN_TIMER] = alien_timer
        states[i, S_SCORE] = score
        states[i, S_LIVES] = lives
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

        state[i, S_SHIP_X] = Scalar[gpu_dtype](SCREEN_W // 2)
        state[i, S_BULLET_ACTIVE] = 0.0
        state[i, S_ABUL_ACTIVE] = 0.0
        state[i, S_ALIENS_REM] = Scalar[gpu_dtype](TOTAL_ALIENS)
        for a in range(TOTAL_ALIENS):
            state[i, S_ALIENS_START + a] = 1.0
        state[i, S_ALIEN_SX] = 0.0
        state[i, S_ALIEN_SY] = 0.0
        state[i, S_ALIEN_DIR] = 1.0
        state[i, S_ALIEN_TIMER] = Scalar[gpu_dtype](ALIEN_MOVE_INTERVAL)
        state[i, S_SCORE] = 0.0
        state[i, S_LIVES] = Scalar[gpu_dtype](INITIAL_LIVES)
        state[i, S_STEP_COUNT] = 0.0
        for r in range(70, 80):
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

        state[i, S_SHIP_X] = Scalar[gpu_dtype](SCREEN_W // 2)
        state[i, S_BULLET_ACTIVE] = 0.0
        state[i, S_ABUL_ACTIVE] = 0.0
        state[i, S_ALIENS_REM] = Scalar[gpu_dtype](TOTAL_ALIENS)
        for a in range(TOTAL_ALIENS):
            state[i, S_ALIENS_START + a] = 1.0
        state[i, S_ALIEN_SX] = 0.0
        state[i, S_ALIEN_SY] = 0.0
        state[i, S_ALIEN_DIR] = 1.0
        state[i, S_ALIEN_TIMER] = Scalar[gpu_dtype](ALIEN_MOVE_INTERVAL)
        state[i, S_SCORE] = 0.0
        state[i, S_LIVES] = Scalar[gpu_dtype](INITIAL_LIVES)
        state[i, S_STEP_COUNT] = 0.0
        for r in range(70, 80):
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
                obs[idx, 0] = states[idx, S_SHIP_X] / Scalar[gpu_dtype](
                    SCREEN_W
                )
                obs[idx, 1] = states[idx, S_BULLET_X] / Scalar[gpu_dtype](
                    SCREEN_W
                )
                obs[idx, 2] = states[idx, S_BULLET_Y] / Scalar[gpu_dtype](
                    SCREEN_H
                )
                obs[idx, 3] = states[idx, S_BULLET_ACTIVE]
                obs[idx, 4] = states[idx, S_ABUL_X] / Scalar[gpu_dtype](
                    SCREEN_W
                )
                obs[idx, 5] = states[idx, S_ABUL_Y] / Scalar[gpu_dtype](
                    SCREEN_H
                )
                obs[idx, 6] = states[idx, S_ABUL_ACTIVE]
                obs[idx, 7] = states[idx, S_ALIENS_REM] / Scalar[gpu_dtype](
                    TOTAL_ALIENS
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
                obs[idx, 0] = states[idx, S_SHIP_X] / Scalar[gpu_dtype](
                    SCREEN_W
                )
                obs[idx, 1] = states[idx, S_BULLET_X] / Scalar[gpu_dtype](
                    SCREEN_W
                )
                obs[idx, 2] = states[idx, S_BULLET_Y] / Scalar[gpu_dtype](
                    SCREEN_H
                )
                obs[idx, 3] = states[idx, S_BULLET_ACTIVE]
                obs[idx, 4] = states[idx, S_ABUL_X] / Scalar[gpu_dtype](
                    SCREEN_W
                )
                obs[idx, 5] = states[idx, S_ABUL_Y] / Scalar[gpu_dtype](
                    SCREEN_H
                )
                obs[idx, 6] = states[idx, S_ABUL_ACTIVE]
                obs[idx, 7] = states[idx, S_ALIENS_REM] / Scalar[gpu_dtype](
                    TOTAL_ALIENS
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
