"""Native Pong — CPU+GPU environment for RL training.

Pure physics: ball + 2 paddles. No Atari emulation.
Follows the CartPole pattern: instance methods for CPU, static methods for GPU.

State layout (STATE_SIZE = 12):
  [0] ball_x          (0..159)
  [1] ball_y          (0..209)
  [2] ball_vx         (-3..3)
  [3] ball_vy         (-3..3)
  [4] paddle_y        (agent paddle center, 0..209)
  [5] cpu_paddle_y    (CPU paddle center, 0..209)
  [6] player_score    (0..21)
  [7] cpu_score       (0..21)
  [8] serve_timer     (countdown)
  [9] step_count
  [10] score          (= player_score, for reward delta)
  [11] lives          (unused, 0)

Clean obs (first 6): ball_x/y, ball_vx/vy, paddle_y, cpu_paddle_y (all normalized)
Actions: 0=NOOP, 1=UP, 2=DOWN
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
from ..core.colors import (
    SCREEN_W,
    SCREEN_H,
    COLOR_BLACK,
    COLOR_WHITE,
    COLOR_GRAY,
)
from ..core.gpu_renderer import (
    clear_frame,
    draw_filled_rect,
    draw_dashed_vline,
    draw_number,
)
from noeira.core.tiles import Tile, ImmTile, tile

# ============================================================================
# Physics Constants
# ============================================================================

comptime PADDLE_WIDTH: Int = 4
comptime PADDLE_HEIGHT: Int = 24  # Bigger paddle (was 16) for easier learning
comptime PADDLE_SPEED: Float64 = 3.0
comptime BALL_SIZE: Int = 2
comptime BALL_SPEED: Float64 = 2.0
comptime MAX_BALL_VY: Float64 = 3.0

comptime LEFT_PADDLE_X: Int = 8  # CPU
comptime RIGHT_PADDLE_X: Int = 148  # Agent

# State slot indices
comptime S_BALL_X: Int = 0
comptime S_BALL_Y: Int = 1
comptime S_BALL_VX: Int = 2
comptime S_BALL_VY: Int = 3
comptime S_PADDLE_Y: Int = 4
comptime S_CPU_PADDLE_Y: Int = 5
comptime S_PLAYER_SCORE: Int = 6
comptime S_CPU_SCORE: Int = 7
comptime S_SERVE_TIMER: Int = 8
comptime S_STEP_COUNT: Int = 9
comptime S_SCORE: Int = 10
comptime S_LIVES: Int = 11

comptime CPU_SPEED: Float64 = 1.0  # Slower CPU (was 1.8)
comptime CPU_REACTION_ZONE: Float64 = 8.0  # Larger dead zone (was 2.0)
comptime WIN_SCORE: Int = 21
comptime PONG_MAX_STEPS: Int = 5000


# ============================================================================
# PongEnv
# ============================================================================


struct PongEnv[DTYPE: DType, HIT_REWARD: Float64 = 0.1](
    BoxDiscreteActionEnv & GPUDiscreteEnv & RenderableEnv & Movable
):
    """Native Pong environment — CPU+GPU dual path.

    `HIT_REWARD` is the dense shaping reward granted when the agent's
    paddle returns the ball (default 0.1). Set it to 0.0 for clean sparse
    rewards (±1 on points only) — useful when the dense shaping distorts
    the value scale / C51 support. Back-compatible: existing
    `PongEnv[dtype]` instantiations keep the 0.1 default.

    CPU: Instance methods for evaluation + SDL3 rendering.
    GPU: Static inline methods for batched RL training.

    Usage:
        # CPU
        var env = PongEnv[DType.float64]()
        var obs = env.reset_obs_list()
        var result = env.step_obs(1)  # UP

        # GPU (batched) — see GPUDiscreteEnv trait
        PongEnv.step_kernel_gpu[BATCH, STATE, OBS](ctx, states, ...)
    """

    # Trait conformance
    comptime dtype = Self.DTYPE
    comptime StateType = ArcadeGameState
    comptime ActionType = ArcadeGameAction

    # GPUDiscreteEnv constants
    comptime STATE_SIZE: Int = 12
    comptime OBS_DIM: Int = 6  # Clean obs: ball_xy, ball_vxy, paddle_y, cpu_y
    comptime NUM_ACTIONS: Int = 3  # NOOP, UP, DOWN
    comptime STEP_WS_SHARED: Int = 0
    comptime STEP_WS_PER_ENV: Int = 0

    # CPU state
    var state: Array[Scalar[Self.dtype], 12]
    var done: Bool
    var _rng_counter: UInt32

    # Renderer
    var _renderer: Optional[Pointer[Renderer2D, MutUntrackedOrigin]]
    var _renderer_initialized: Bool

    def __init__(out self):
        self.state = Array[Scalar[Self.dtype], 12](
            fill=Scalar[Self.dtype](0.0)
        )
        self.done = False
        self._rng_counter = 42
        self._renderer = None
        self._renderer_initialized = False

    # ========================================================================
    # CPU: reset + step
    # ========================================================================

    def reset(mut self) -> ArcadeGameState:
        self._rng_counter += 1
        # Ball at center
        self.state[S_BALL_X] = Scalar[Self.dtype](SCREEN_W // 2)
        self.state[S_BALL_Y] = Scalar[Self.dtype](SCREEN_H // 2)
        # Random direction
        var vx = random_float64() - 0.5
        if vx >= 0:
            self.state[S_BALL_VX] = Scalar[Self.dtype](BALL_SPEED)
        else:
            self.state[S_BALL_VX] = Scalar[Self.dtype](-BALL_SPEED)
        self.state[S_BALL_VY] = Scalar[Self.dtype](
            (random_float64() - 0.5) * 3.0
        )
        # Paddles at center
        self.state[S_PADDLE_Y] = Scalar[Self.dtype](SCREEN_H // 2)
        self.state[S_CPU_PADDLE_Y] = Scalar[Self.dtype](SCREEN_H // 2)
        # Scores
        self.state[S_PLAYER_SCORE] = 0.0
        self.state[S_CPU_SCORE] = 0.0
        self.state[S_SERVE_TIMER] = 30.0
        self.state[S_STEP_COUNT] = 0.0
        self.state[S_SCORE] = 0.0
        self.state[S_LIVES] = 0.0
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
        """Internal step: returns (reward, done)."""
        # Move agent paddle
        if action == 1:  # UP
            self.state[S_PADDLE_Y] = max(
                Scalar[Self.dtype](PADDLE_HEIGHT // 2),
                self.state[S_PADDLE_Y] - Scalar[Self.dtype](PADDLE_SPEED),
            )
        elif action == 2:  # DOWN
            self.state[S_PADDLE_Y] = min(
                Scalar[Self.dtype](SCREEN_H - PADDLE_HEIGHT // 2),
                self.state[S_PADDLE_Y] + Scalar[Self.dtype](PADDLE_SPEED),
            )

        # CPU AI — only tracks ball when ball moves toward CPU (vx < 0)
        var cpu_y = self.state[S_CPU_PADDLE_Y]
        if self.state[S_BALL_VX] < 0:
            var ball_y = self.state[S_BALL_Y]
            var diff = ball_y - cpu_y
            if diff > Scalar[Self.dtype](CPU_REACTION_ZONE):
                self.state[S_CPU_PADDLE_Y] = min(
                    Scalar[Self.dtype](SCREEN_H - PADDLE_HEIGHT // 2),
                    cpu_y + Scalar[Self.dtype](CPU_SPEED),
                )
            elif diff < Scalar[Self.dtype](-CPU_REACTION_ZONE):
                self.state[S_CPU_PADDLE_Y] = max(
                    Scalar[Self.dtype](PADDLE_HEIGHT // 2),
                    cpu_y - Scalar[Self.dtype](CPU_SPEED),
                )

        # Serve timer
        if self.state[S_SERVE_TIMER] > 0:
            self.state[S_SERVE_TIMER] -= 1.0
            self.state[S_STEP_COUNT] += 1.0
            return (Scalar[Self.dtype](0.0), False)

        # Move ball
        self.state[S_BALL_X] += self.state[S_BALL_VX]
        self.state[S_BALL_Y] += self.state[S_BALL_VY]

        # Bounce off top/bottom
        if self.state[S_BALL_Y] < Scalar[Self.dtype](BALL_SIZE):
            self.state[S_BALL_Y] = Scalar[Self.dtype](BALL_SIZE)
            self.state[S_BALL_VY] = -self.state[S_BALL_VY]
        elif self.state[S_BALL_Y] > Scalar[Self.dtype](SCREEN_H - BALL_SIZE):
            self.state[S_BALL_Y] = Scalar[Self.dtype](SCREEN_H - BALL_SIZE)
            self.state[S_BALL_VY] = -self.state[S_BALL_VY]

        # Paddle collision (agent = right)
        var bx = self.state[S_BALL_X]
        var by = self.state[S_BALL_Y]
        var pad_y = self.state[S_PADDLE_Y]
        var half_h = Scalar[Self.dtype](PADDLE_HEIGHT // 2)
        var agent_hit = False

        if (
            bx >= Scalar[Self.dtype](RIGHT_PADDLE_X)
            and bx <= Scalar[Self.dtype](RIGHT_PADDLE_X + PADDLE_WIDTH)
            and by >= pad_y - half_h
            and by <= pad_y + half_h
            and self.state[S_BALL_VX] > 0
        ):
            self.state[S_BALL_VX] = -self.state[S_BALL_VX]
            var hit_pos = (by - pad_y) / half_h
            self.state[S_BALL_VY] = hit_pos * Scalar[Self.dtype](MAX_BALL_VY)
            self.state[S_BALL_X] = Scalar[Self.dtype](RIGHT_PADDLE_X - 1)
            agent_hit = True

        # Paddle collision (CPU = left)
        var cpu_pad_y = self.state[S_CPU_PADDLE_Y]
        if (
            bx <= Scalar[Self.dtype](LEFT_PADDLE_X + PADDLE_WIDTH)
            and bx >= Scalar[Self.dtype](LEFT_PADDLE_X)
            and by >= cpu_pad_y - half_h
            and by <= cpu_pad_y + half_h
            and self.state[S_BALL_VX] < 0
        ):
            self.state[S_BALL_VX] = -self.state[S_BALL_VX]
            var hit_pos = (by - cpu_pad_y) / half_h
            self.state[S_BALL_VY] = hit_pos * Scalar[Self.dtype](MAX_BALL_VY)
            self.state[S_BALL_X] = Scalar[Self.dtype](
                LEFT_PADDLE_X + PADDLE_WIDTH + 1
            )

        # Scoring
        var scored_player = False
        var scored_cpu = False

        if self.state[S_BALL_X] > Scalar[Self.dtype](SCREEN_W):
            self.state[S_CPU_SCORE] += 1.0
            scored_cpu = True
        elif self.state[S_BALL_X] < 0:
            self.state[S_PLAYER_SCORE] += 1.0
            scored_player = True

        # Reset ball after score
        if scored_player or scored_cpu:
            self.state[S_BALL_X] = Scalar[Self.dtype](SCREEN_W // 2)
            self.state[S_BALL_Y] = Scalar[Self.dtype](SCREEN_H // 2)
            self.state[S_BALL_VY] = Scalar[Self.dtype](
                (random_float64() - 0.5) * 3.0
            )
            if scored_cpu:
                self.state[S_BALL_VX] = Scalar[Self.dtype](BALL_SPEED)
            else:
                self.state[S_BALL_VX] = Scalar[Self.dtype](-BALL_SPEED)
            self.state[S_SERVE_TIMER] = 30.0

        self.state[S_SCORE] = self.state[S_PLAYER_SCORE]
        self.state[S_STEP_COUNT] += 1.0

        # Terminal
        var player_score = Int(self.state[S_PLAYER_SCORE])
        var cpu_score = Int(self.state[S_CPU_SCORE])
        var step_count = Int(self.state[S_STEP_COUNT])
        var terminated = player_score >= WIN_SCORE or cpu_score >= WIN_SCORE
        var truncated = step_count >= PONG_MAX_STEPS
        self.done = terminated or truncated

        var reward = Scalar[Self.dtype](0.0)
        if scored_player:
            reward = Scalar[Self.dtype](1.0)
        elif scored_cpu:
            reward = Scalar[Self.dtype](-1.0)
        elif agent_hit:
            # Dense shaping reward for returning the ball (0.0 disables it).
            reward = Scalar[Self.dtype](Self.HIT_REWARD)

        return (reward, self.done)

    # ========================================================================
    # Env trait methods
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
        return 3

    def obs_dim(self) -> Int:
        return 6

    def num_states(self) -> Int:
        return 1

    def state_to_index(self, state: ArcadeGameState) -> Int:
        return state.index

    # ========================================================================
    # ContinuousStateEnv / BoxDiscreteActionEnv (CPU)
    # ========================================================================

    def get_obs_list(self) -> List[Scalar[Self.dtype]]:
        var obs = List[Scalar[Self.dtype]](capacity=6)
        # Normalized observations
        obs.append(self.state[S_BALL_X] / Scalar[Self.dtype](SCREEN_W))
        obs.append(self.state[S_BALL_Y] / Scalar[Self.dtype](SCREEN_H))
        obs.append(self.state[S_BALL_VX] / Scalar[Self.dtype](MAX_BALL_VY))
        obs.append(self.state[S_BALL_VY] / Scalar[Self.dtype](MAX_BALL_VY))
        obs.append(self.state[S_PADDLE_Y] / Scalar[Self.dtype](SCREEN_H))
        obs.append(self.state[S_CPU_PADDLE_Y] / Scalar[Self.dtype](SCREEN_H))
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
    # RenderableEnv trait methods
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

    @staticmethod
    def _draw_7seg_digit(
        mut renderer: Renderer2D,
        digit: Int,
        x: Int,
        y: Int,
        w: Int,
        h: Int,
        color: SDL_Color,
    ):
        """Draw a single digit 0-9 in 7-segment style using filled rects.

        Args:
            renderer: The renderer to draw on.
            digit: The digit to draw (0-9).
            x: x-coordinate of top-left corner.
            y: y-coordinate of top-left corner.
            w: total digit width.
            h: total digit height.
            color: The color to draw the digit in.
        """
        var t = max(2, h // 8)  # segment thickness
        var half_h = (h - 3 * t) // 2  # height of each vertical half

        # Segment bits: a=0x01(top), b=0x02(top-right), c=0x04(bot-right),
        #   d=0x08(middle), e=0x10(bot-left), f=0x20(top-left), g=0x40(bottom)
        var segs: Int
        if digit == 0:
            segs = 0x77  # a,b,c,e,f,g (no middle)
        elif digit == 1:
            segs = 0x06  # b,c
        elif digit == 2:
            segs = 0x5B  # a,b,d,e,g
        elif digit == 3:
            segs = 0x4F  # a,b,c,d,g
        elif digit == 4:
            segs = 0x2E  # b,c,d,f
        elif digit == 5:
            segs = 0x6D  # a,c,d,f,g
        elif digit == 6:
            segs = 0x7D  # a,c,d,e,f,g
        elif digit == 7:
            segs = 0x07  # a,b,c
        elif digit == 8:
            segs = 0x7F  # all
        else:
            segs = 0x6F  # a,b,c,d,f,g

        var mid_y = y + t + half_h  # top of middle bar

        # a: top horizontal
        if segs & 0x01:
            renderer.draw_rect(x, y, w, t, color)
        # b: top-right vertical
        if segs & 0x02:
            renderer.draw_rect(x + w - t, y + t, t, half_h, color)
        # c: bottom-right vertical
        if segs & 0x04:
            renderer.draw_rect(x + w - t, mid_y + t, t, half_h, color)
        # d: middle horizontal
        if segs & 0x08:
            renderer.draw_rect(x, mid_y, w, t, color)
        # e: bottom-left vertical
        if segs & 0x10:
            renderer.draw_rect(x, mid_y + t, t, half_h, color)
        # f: top-left vertical
        if segs & 0x20:
            renderer.draw_rect(x, y + t, t, half_h, color)
        # g: bottom horizontal
        if segs & 0x40:
            renderer.draw_rect(x, mid_y + t + half_h, w, t, color)

    @staticmethod
    def _draw_score(
        mut renderer: Renderer2D,
        score: Int,
        center_x: Int,
        y: Int,
        digit_w: Int,
        digit_h: Int,
        color: SDL_Color,
    ):
        """Draw a multi-digit score, centered at center_x."""
        var s = score
        if s < 0:
            s = 0
        var tens = s // 10
        var ones = s % 10
        var gap = digit_w // 3
        if tens > 0:
            var total_w = 2 * digit_w + gap
            var start_x = center_x - total_w // 2
            Self._draw_7seg_digit(
                renderer, tens, start_x, y, digit_w, digit_h, color
            )
            Self._draw_7seg_digit(
                renderer,
                ones,
                start_x + digit_w + gap,
                y,
                digit_w,
                digit_h,
                color,
            )
        else:
            Self._draw_7seg_digit(
                renderer,
                ones,
                center_x - digit_w // 2,
                y,
                digit_w,
                digit_h,
                color,
            )

    def _render(self, mut renderer: Renderer2D):
        """Render Pong state using SDL3 — Atari-style dark theme."""
        var bg_color = SDL_Color(20, 20, 40, 255)
        if not renderer.begin_frame_with_color(bg_color):
            return

        # Scale: Atari 160×210 → full window (1:1 mapping, no score bar offset)
        var sw = renderer.screen_width
        var sh = renderer.screen_height
        var sx = Float64(sw) / Float64(SCREEN_W)
        var sy = Float64(sh) / Float64(SCREEN_H)

        # Colors
        var cpu_color = SDL_Color(80, 170, 170, 255)
        var player_color = SDL_Color(170, 80, 170, 255)
        var ball_color = SDL_Color(230, 230, 245, 255)
        var net_color = SDL_Color(80, 80, 110, 255)
        var sep_color = SDL_Color(120, 120, 160, 255)

        # -- Score area (top ~24 Atari pixels) --
        var score_area_h = Int(24.0 * sy)
        var score_bar_color = SDL_Color(40, 40, 60, 255)
        renderer.draw_rect(0, 0, sw, score_area_h, score_bar_color)
        # Separator line
        renderer.draw_rect(0, score_area_h, sw, max(1, Int(sy)), sep_color)

        # -- Big 7-segment score digits --
        var digit_w = max(6, Int(16.0 * sx))
        var digit_h = max(10, Int(18.0 * sy))
        var score_y = (score_area_h - digit_h) // 2
        # CPU score — left quarter
        Self._draw_score(
            renderer,
            Int(self.state[S_CPU_SCORE]),
            sw // 4,
            score_y,
            digit_w,
            digit_h,
            cpu_color,
        )
        # Player score — right quarter
        Self._draw_score(
            renderer,
            Int(self.state[S_PLAYER_SCORE]),
            sw * 3 // 4,
            score_y,
            digit_w,
            digit_h,
            player_color,
        )

        # -- Bottom info bar (compute first so play area excludes it) --
        var info_h = max(1, Int(14.0 * sy))
        var info_y = sh - info_h

        # Play area: game y [0..SCREEN_H] maps to screen [play_top..info_y]
        var play_top = score_area_h + max(1, Int(sy))
        var play_h = info_y - play_top

        # -- Center net (dashed) --
        var net_x = sw // 2
        var net_w = max(2, Int(2.0 * sx))
        var dash_h = max(2, Int(6.0 * sy))
        var gap_h = max(2, Int(4.0 * sy))
        var y = play_top
        while y < info_y:
            renderer.draw_rect(net_x - net_w // 2, y, net_w, dash_h, net_color)
            y += dash_h + gap_h

        # -- CPU paddle (left, teal) --
        var cpu_y_f = Float64(Int(self.state[S_CPU_PADDLE_Y]))
        var pw = max(3, Int(Float64(PADDLE_WIDTH) * sx))
        var ph = max(
            6, Int(Float64(PADDLE_HEIGHT) / Float64(SCREEN_H) * Float64(play_h))
        )
        renderer.draw_rect(
            Int(Float64(LEFT_PADDLE_X) * sx),
            play_top
            + Int(
                (cpu_y_f - Float64(PADDLE_HEIGHT // 2))
                / Float64(SCREEN_H)
                * Float64(play_h)
            ),
            pw,
            ph,
            cpu_color,
        )

        # -- Agent paddle (right, purple) --
        var pad_y_f = Float64(Int(self.state[S_PADDLE_Y]))
        renderer.draw_rect(
            Int(Float64(RIGHT_PADDLE_X) * sx),
            play_top
            + Int(
                (pad_y_f - Float64(PADDLE_HEIGHT // 2))
                / Float64(SCREEN_H)
                * Float64(play_h)
            ),
            pw,
            ph,
            player_color,
        )

        # -- Ball (bright square) --
        var bx_f = Float64(Int(self.state[S_BALL_X]))
        var by_f = Float64(Int(self.state[S_BALL_Y]))
        var bsz = max(4, Int(Float64(BALL_SIZE) * sx * 2.0))
        renderer.draw_rect(
            Int(bx_f * sx) - bsz // 2,
            play_top
            + Int(by_f / Float64(SCREEN_H) * Float64(play_h))
            - bsz // 2,
            bsz,
            bsz,
            ball_color,
        )

        # -- Draw bottom info bar (on top of everything) --
        renderer.draw_rect(0, info_y, sw, info_h, score_bar_color)
        renderer.draw_rect(0, info_y, sw, max(1, Int(sy)), sep_color)
        var info_color = SDL_Color(160, 160, 180, 255)
        var score_diff = Int(self.state[S_PLAYER_SCORE]) - Int(
            self.state[S_CPU_SCORE]
        )
        var info_str = (
            "Score: "
            + String(score_diff)
            + "      Lives: 0      Frame: "
            + String(Int(self.state[S_STEP_COUNT]))
        )
        renderer.draw_text(info_str, 8, info_y + 2, info_color)

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
    # GPU: Inline step/reset kernels (called per-thread on GPU)
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
        """Per-thread Pong step kernel — inlined game physics."""
        var i = Int(block_dim.x * block_idx.x + thread_idx.x)
        if i >= BATCH_SIZE:
            return

        var action = Int(actions[i])

        # --- Load state into local variables ---
        var bx = states[i, S_BALL_X]
        var by = states[i, S_BALL_Y]
        var bvx = states[i, S_BALL_VX]
        var bvy = states[i, S_BALL_VY]
        var pad_y = states[i, S_PADDLE_Y]
        var cpu_y = states[i, S_CPU_PADDLE_Y]
        var p_score = states[i, S_PLAYER_SCORE]
        var c_score = states[i, S_CPU_SCORE]
        var serve = states[i, S_SERVE_TIMER]
        var steps = states[i, S_STEP_COUNT]

        # --- Move agent paddle ---
        if action == 1:  # UP
            pad_y = pad_y - Scalar[gpu_dtype](PADDLE_SPEED)
            if pad_y < Scalar[gpu_dtype](PADDLE_HEIGHT // 2):
                pad_y = Scalar[gpu_dtype](PADDLE_HEIGHT // 2)
        elif action == 2:  # DOWN
            pad_y = pad_y + Scalar[gpu_dtype](PADDLE_SPEED)
            if pad_y > Scalar[gpu_dtype](SCREEN_H - PADDLE_HEIGHT // 2):
                pad_y = Scalar[gpu_dtype](SCREEN_H - PADDLE_HEIGHT // 2)

        # --- CPU AI — only tracks when ball approaches (vx < 0) ---
        if bvx < 0:
            var diff = by - cpu_y
            if diff > Scalar[gpu_dtype](CPU_REACTION_ZONE):
                cpu_y = cpu_y + Scalar[gpu_dtype](CPU_SPEED)
                if cpu_y > Scalar[gpu_dtype](SCREEN_H - PADDLE_HEIGHT // 2):
                    cpu_y = Scalar[gpu_dtype](SCREEN_H - PADDLE_HEIGHT // 2)
            elif diff < Scalar[gpu_dtype](-CPU_REACTION_ZONE):
                cpu_y = cpu_y - Scalar[gpu_dtype](CPU_SPEED)
                if cpu_y < Scalar[gpu_dtype](PADDLE_HEIGHT // 2):
                    cpu_y = Scalar[gpu_dtype](PADDLE_HEIGHT // 2)

        states[i, S_PADDLE_Y] = pad_y
        states[i, S_CPU_PADDLE_Y] = cpu_y

        # --- Serve timer ---
        if serve > 0:
            states[i, S_SERVE_TIMER] = serve - 1.0
            states[i, S_STEP_COUNT] = steps + 1.0
            rewards[i] = 0.0
            dones[i] = 0.0
            return

        # --- Move ball ---
        bx = bx + bvx
        by = by + bvy

        # --- Top/bottom bounce ---
        if by < Scalar[gpu_dtype](BALL_SIZE):
            by = Scalar[gpu_dtype](BALL_SIZE)
            bvy = -bvy
        elif by > Scalar[gpu_dtype](SCREEN_H - BALL_SIZE):
            by = Scalar[gpu_dtype](SCREEN_H - BALL_SIZE)
            bvy = -bvy

        # --- Agent paddle collision (right) ---
        var agent_hit = False
        if (
            bx >= Scalar[gpu_dtype](RIGHT_PADDLE_X)
            and bx <= Scalar[gpu_dtype](RIGHT_PADDLE_X + PADDLE_WIDTH)
            and by >= pad_y - Scalar[gpu_dtype](PADDLE_HEIGHT // 2)
            and by <= pad_y + Scalar[gpu_dtype](PADDLE_HEIGHT // 2)
            and bvx > 0
        ):
            bvx = -bvx
            var hit_pos = (by - pad_y) / Scalar[gpu_dtype](PADDLE_HEIGHT // 2)
            bvy = hit_pos * Scalar[gpu_dtype](MAX_BALL_VY)
            bx = Scalar[gpu_dtype](RIGHT_PADDLE_X - 1)
            agent_hit = True

        # --- CPU paddle collision (left) ---
        if (
            bx <= Scalar[gpu_dtype](LEFT_PADDLE_X + PADDLE_WIDTH)
            and bx >= Scalar[gpu_dtype](LEFT_PADDLE_X)
            and by >= cpu_y - Scalar[gpu_dtype](PADDLE_HEIGHT // 2)
            and by <= cpu_y + Scalar[gpu_dtype](PADDLE_HEIGHT // 2)
            and bvx < 0
        ):
            bvx = -bvx
            var hit_pos = (by - cpu_y) / Scalar[gpu_dtype](PADDLE_HEIGHT // 2)
            bvy = hit_pos * Scalar[gpu_dtype](MAX_BALL_VY)
            bx = Scalar[gpu_dtype](LEFT_PADDLE_X + PADDLE_WIDTH + 1)

        # --- Scoring ---
        var scored_player = bx < 0
        var scored_cpu = bx > Scalar[gpu_dtype](SCREEN_W)

        if scored_cpu:
            c_score = c_score + 1.0
        if scored_player:
            p_score = p_score + 1.0

        # Reset ball after score
        if scored_player or scored_cpu:
            bx = Scalar[gpu_dtype](SCREEN_W // 2)
            by = Scalar[gpu_dtype](SCREEN_H // 2)
            var rng = PhiloxRandom(
                seed=UInt64(rng_seed) * UInt64(BATCH_SIZE) + UInt64(i), offset=0
            )
            var rand_vals = rng.step_uniform()
            bvy = Scalar[gpu_dtype](-1.5) + Scalar[gpu_dtype](
                rand_vals[0]
            ) * Scalar[gpu_dtype](3.0)
            if scored_cpu:
                bvx = Scalar[gpu_dtype](BALL_SPEED)
            else:
                bvx = Scalar[gpu_dtype](-BALL_SPEED)
            serve = 30.0
        else:
            serve = 0.0

        steps = steps + 1.0

        # --- Write all state back ---
        states[i, S_BALL_X] = bx
        states[i, S_BALL_Y] = by
        states[i, S_BALL_VX] = bvx
        states[i, S_BALL_VY] = bvy
        states[i, S_PLAYER_SCORE] = p_score
        states[i, S_CPU_SCORE] = c_score
        states[i, S_SERVE_TIMER] = serve
        states[i, S_STEP_COUNT] = steps
        states[i, S_SCORE] = p_score

        # --- Reward ---
        if scored_player:
            rewards[i] = 1.0
        elif scored_cpu:
            rewards[i] = -1.0
        elif agent_hit:
            # Dense shaping reward for returning the ball (0.0 disables it).
            rewards[i] = Scalar[gpu_dtype](Self.HIT_REWARD)
        else:
            rewards[i] = 0.0

        # --- Done ---
        var terminated = Int(p_score) >= WIN_SCORE or Int(c_score) >= WIN_SCORE
        var truncated = Int(steps) >= PONG_MAX_STEPS
        # `terminated or truncated` is a Bool, and `Scalar[float](Bool)` no
        # longer compiles — SIMD's Intable constructor now requires an integral
        # dtype.
        dones[i] = Scalar[gpu_dtype](1.0) if (
            terminated or truncated
        ) else Scalar[gpu_dtype](0.0)

    @staticmethod
    @always_inline
    def reset_kernel[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        state: Tile[gpu_dtype, BATCH_SIZE, STATE_SIZE],
    ):
        """Per-thread Pong reset kernel."""
        var i = Int(block_dim.x * block_idx.x + thread_idx.x)
        if i >= BATCH_SIZE:
            return

        var rng = PhiloxRandom(
            seed=UInt64(12345) * UInt64(BATCH_SIZE) + UInt64(i), offset=0
        )

        state[i, S_BALL_X] = Scalar[gpu_dtype](SCREEN_W // 2)
        state[i, S_BALL_Y] = Scalar[gpu_dtype](SCREEN_H // 2)

        var rand_vals = rng.step_uniform()
        if Scalar[gpu_dtype](rand_vals[0]) >= Scalar[gpu_dtype](0.5):
            state[i, S_BALL_VX] = Scalar[gpu_dtype](BALL_SPEED)
        else:
            state[i, S_BALL_VX] = Scalar[gpu_dtype](-BALL_SPEED)

        var rand_vals2 = rng.step_uniform()
        state[i, S_BALL_VY] = Scalar[gpu_dtype](-1.5) + Scalar[gpu_dtype](
            rand_vals2[0]
        ) * Scalar[gpu_dtype](3.0)

        state[i, S_PADDLE_Y] = Scalar[gpu_dtype](SCREEN_H // 2)
        state[i, S_CPU_PADDLE_Y] = Scalar[gpu_dtype](SCREEN_H // 2)
        state[i, S_PLAYER_SCORE] = 0.0
        state[i, S_CPU_SCORE] = 0.0
        state[i, S_SERVE_TIMER] = 30.0
        state[i, S_STEP_COUNT] = 0.0
        state[i, S_SCORE] = 0.0
        state[i, S_LIVES] = 0.0

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
        """Reset only done environments."""
        var i = Int(block_dim.x * block_idx.x + thread_idx.x)
        if i >= BATCH_SIZE:
            return
        if dones[i] < Scalar[gpu_dtype](0.5):
            return

        var rng = PhiloxRandom(
            seed=UInt64(rng_seed) * UInt64(BATCH_SIZE) + UInt64(i), offset=0
        )

        state[i, S_BALL_X] = Scalar[gpu_dtype](SCREEN_W // 2)
        state[i, S_BALL_Y] = Scalar[gpu_dtype](SCREEN_H // 2)

        var rand_vals = rng.step_uniform()
        if Scalar[gpu_dtype](rand_vals[0]) >= Scalar[gpu_dtype](0.5):
            state[i, S_BALL_VX] = Scalar[gpu_dtype](BALL_SPEED)
        else:
            state[i, S_BALL_VX] = Scalar[gpu_dtype](-BALL_SPEED)

        var rand_vals2 = rng.step_uniform()
        state[i, S_BALL_VY] = Scalar[gpu_dtype](-1.5) + Scalar[gpu_dtype](
            rand_vals2[0]
        ) * Scalar[gpu_dtype](3.0)

        state[i, S_PADDLE_Y] = Scalar[gpu_dtype](SCREEN_H // 2)
        state[i, S_CPU_PADDLE_Y] = Scalar[gpu_dtype](SCREEN_H // 2)
        state[i, S_PLAYER_SCORE] = 0.0
        state[i, S_CPU_SCORE] = 0.0
        state[i, S_SERVE_TIMER] = 30.0
        state[i, S_STEP_COUNT] = 0.0
        state[i, S_SCORE] = 0.0
        state[i, S_LIVES] = 0.0

        dones[i] = 0.0

    # ========================================================================
    # GPU Launcher Methods (host-side, GPUDiscreteEnv trait)
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
            # Call step kernel (inlined Pong physics)
            Self.step_kernel[BATCH_SIZE, STATE_SIZE](
                states, actions, rewards, dones, rng_seed
            )

            # Extract observations + copy terminated
            var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
            if idx < BATCH_SIZE:
                terminated_out[idx] = dones[idx]

                # Normalized clean observations
                obs[idx, 0] = states[idx, S_BALL_X] / Scalar[gpu_dtype](
                    SCREEN_W
                )
                obs[idx, 1] = states[idx, S_BALL_Y] / Scalar[gpu_dtype](
                    SCREEN_H
                )
                obs[idx, 2] = states[idx, S_BALL_VX] / Scalar[gpu_dtype](
                    MAX_BALL_VY
                )
                obs[idx, 3] = states[idx, S_BALL_VY] / Scalar[gpu_dtype](
                    MAX_BALL_VY
                )
                obs[idx, 4] = states[idx, S_PADDLE_Y] / Scalar[gpu_dtype](
                    SCREEN_H
                )
                obs[idx, 5] = states[idx, S_CPU_PADDLE_Y] / Scalar[gpu_dtype](
                    SCREEN_H
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
        kernel applies (state / SCREEN_W,H ; vel / MAX_BALL_VY).

        Overrides the trait default, which copies the RAW state prefix
        (`obs[e] = state[e][0:OBS_DIM]`). Pong's obs is normalized, so the
        raw default disagrees with `step_kernel_gpu`'s normalized obs — and
        because the batched-env driver re-seeds obs via this kernel after
        every `selective_reset`, `prev_obs` (snapshotted next iteration) was
        RAW while the stored `next_obs` was NORMALIZED. That ~160× scale
        mismatch on (s, s') silently corrupted every transition and made the
        GPU-batched agent train to a uniform distribution. Keep this in lock-
        step with the obs block in `step_kernel_gpu`."""
        # `states_buf` is read-only here; its mut=False view widens into the
        # ImmutAnyOrigin wrapper param below.
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
                    MAX_BALL_VY
                )
                obs[idx, 3] = states[idx, S_BALL_VY] / Scalar[gpu_dtype](
                    MAX_BALL_VY
                )
                obs[idx, 4] = states[idx, S_PADDLE_Y] / Scalar[gpu_dtype](
                    SCREEN_H
                )
                obs[idx, 5] = states[idx, S_CPU_PADDLE_Y] / Scalar[gpu_dtype](
                    SCREEN_H
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
            def selective_reset_counter_wrapper(
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

            ctx.enqueue_function[selective_reset_counter_wrapper](
                states,
                dones,
                counter_t,
                grid_dim=(BLOCKS,),
                block_dim=(Self.TPB,),
            )
        else:
            var seed = Scalar[DType.uint64](rng_seed)

            @always_inline
            def selective_reset_wrapper(
                states: Tile[gpu_dtype, BATCH_SIZE, STATE_SIZE],
                dones: Tile[gpu_dtype, BATCH_SIZE],
                rng_seed: Scalar[DType.uint64],
            ):
                Self.selective_reset_kernel[BATCH_SIZE, STATE_SIZE](
                    states, dones, Scalar[DType.uint32](rng_seed)
                )

            ctx.enqueue_function[selective_reset_wrapper](
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
