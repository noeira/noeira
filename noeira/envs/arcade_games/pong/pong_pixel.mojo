"""Native Pong — Pixel observation variant for CNN-based DQN training.

Wraps PongEnv physics with GPU-rendered pixel observations.
After each physics step, renders the game to a 160×210 grayscale framebuffer,
resizes to 84×84, and maintains a 4-frame stack for temporal context.

State layout: same as PongEnv (STATE_SIZE = 12)
Pixel obs (OBS_DIM = 28224): 4 × 84 × 84 grayscale frames, normalized to [0, 1]
Actions: 0=NOOP, 1=UP, 2=DOWN (same as PongEnv)

GPU workspace per env (PIXEL_WS_PER_ENV = 36625 float32):
  [0..8399]          160×210 grayscale framebuffer (packed 4 UInt8 per float32)
  [8400..36623]      4 × 84×84 frame stack (float32)
  [36624]            frame_idx (ring buffer write position)

Usage:
    from noeira.envs.arcade_games.pong import PongPixelEnv

    # GPU training with CNN DQN
    var metrics = agent.train_gpu[PongPixelEnv[DType.float32]](ctx, ...)
"""

from std.random import random_float64
from noeira.nn.core.ptr import untracked
from std.memory import alloc, unsafe_memset
from noeira.nn.constants import LAYOUT_NCHW, LAYOUT_NHWC
from noeira.core import (
    State,
    Action,
    BoxDiscreteActionEnv,
    GPUDiscreteEnv,
    RenderableEnv,
)
from noeira.render import Renderer2D, SDL_Color
from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext, DeviceBuffer

from ..core.gpu_env import ArcadeGameState, ArcadeGameAction, gpu_dtype
from ..core.gpu_env import (
    FRAME_BUF_F32_SIZE,
    FRAME_STACK_F32_SIZE,
    PIXEL_WS_PER_ENV,
)
from ..core.colors import (
    SCREEN_W,
    SCREEN_H,
    OBS_W,
    OBS_H,
    FRAME_STACK,
    PIXEL_OBS_DIM,
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

# Import PongEnv for physics kernels
from .pong import (
    PongEnv,
    S_BALL_X,
    S_BALL_Y,
    S_BALL_VX,
    S_BALL_VY,
    S_PADDLE_Y,
    S_CPU_PADDLE_Y,
    S_PLAYER_SCORE,
    S_CPU_SCORE,
    S_STEP_COUNT,
    PADDLE_WIDTH,
    PADDLE_HEIGHT,
    BALL_SIZE,
    LEFT_PADDLE_X,
    RIGHT_PADDLE_X,
    MAX_BALL_VY,
)
from noeira.core.tiles import Tile, ImmTile, tile


# ============================================================================
# Inline helpers for GPU pixel rendering
# ============================================================================


@always_inline
def _render_pong_frame(
    buf: Pointer[UInt8, MutAnyOrigin],
    bx: Int,
    by: Int,
    pad_y: Int,
    cpu_y: Int,
    p_score: Int,
    c_score: Int,
):
    """Render Pong game state to 160×210 grayscale buffer (clear + draw).

    Used by the CPU single-env path. The GPU batched path splits these
    two steps so it can parallelise the (dominant) framebuffer clear
    across pixels — see `_draw_pong_frame` and `step_kernel_gpu`."""
    clear_frame(buf)
    _draw_pong_frame(buf, bx, by, pad_y, cpu_y, p_score, c_score)


@always_inline
def _draw_pong_frame(
    buf: Pointer[UInt8, MutAnyOrigin],
    bx: Int,
    by: Int,
    pad_y: Int,
    cpu_y: Int,
    p_score: Int,
    c_score: Int,
):
    """Draw Pong primitives onto an ALREADY-CLEARED 160×210 buffer.

    Split out of `_render_pong_frame` so the GPU batched path can run the
    framebuffer clear as a separate element-parallel kernel (1 thread per
    pixel) while keeping the cheap primitive drawing per-env."""
    # Center dashed line
    draw_dashed_vline(buf, SCREEN_W // 2, 0, SCREEN_H, 4, 4, COLOR_GRAY)

    # CPU paddle (left)
    draw_filled_rect(
        buf,
        LEFT_PADDLE_X,
        cpu_y - PADDLE_HEIGHT // 2,
        PADDLE_WIDTH,
        PADDLE_HEIGHT,
        COLOR_WHITE,
    )

    # Agent paddle (right)
    draw_filled_rect(
        buf,
        RIGHT_PADDLE_X,
        pad_y - PADDLE_HEIGHT // 2,
        PADDLE_WIDTH,
        PADDLE_HEIGHT,
        COLOR_WHITE,
    )

    # Ball
    draw_filled_rect(
        buf,
        bx - BALL_SIZE,
        by - BALL_SIZE,
        BALL_SIZE * 2,
        BALL_SIZE * 2,
        COLOR_WHITE,
    )

    # Scores at top
    draw_number(buf, 30, 5, c_score, COLOR_WHITE, 2)
    draw_number(buf, 110, 5, p_score, COLOR_WHITE, 2)


@always_inline
def _resize_and_push[
    WS_FRAME_STACK: Int,
    WS_FRAME_IDX: Int,
](
    frame_buf: Pointer[UInt8, MutAnyOrigin],
    workspace: Pointer[Scalar[gpu_dtype], MutAnyOrigin],
    obs_out: Pointer[Scalar[gpu_dtype], MutAnyOrigin],
):
    """Resize 160×210 to 84×84, push to frame stack, output 4-frame obs.

    Combines resize + normalize + frame stack update in one pass.
    """
    comptime FRAME_SIZE = OBS_W * OBS_H  # 7056

    # Get current write slot
    var slot = Int(workspace[unsafe_offset=WS_FRAME_IDX]) % FRAME_STACK
    var slot_base = WS_FRAME_STACK + slot * FRAME_SIZE

    # Resize from 160×210 to 84×84, normalize to [0,1], write to frame stack
    for dy in range(OBS_H):
        var sy0 = dy * SCREEN_H // OBS_H
        var sy1 = (dy + 1) * SCREEN_H // OBS_H
        if sy1 == sy0:
            sy1 = sy0 + 1

        for dx in range(OBS_W):
            var sx0 = dx * SCREEN_W // OBS_W
            var sx1 = (dx + 1) * SCREEN_W // OBS_W
            if sx1 == sx0:
                sx1 = sx0 + 1

            # Box-filter average
            var total: Int = 0
            var count: Int = 0
            for sy in range(sy0, sy1):
                for sx in range(sx0, sx1):
                    total += Int(frame_buf[unsafe_offset=sy * SCREEN_W + sx])
                    count += 1

            workspace[unsafe_offset=slot_base + dy * OBS_W + dx] = (
                Scalar[gpu_dtype](total // count) / 255.0
            )

    # Advance ring buffer index
    workspace[unsafe_offset=WS_FRAME_IDX] = Scalar[gpu_dtype]((slot + 1) % FRAME_STACK)

    # Output chronological frame stack: oldest → newest
    for f in range(FRAME_STACK):
        var read_slot = (slot + 1 + f) % FRAME_STACK
        var read_base = WS_FRAME_STACK + read_slot * FRAME_SIZE
        var out_base = f * FRAME_SIZE
        for i in range(FRAME_SIZE):
            obs_out[unsafe_offset=out_base + i] = workspace[unsafe_offset=read_base + i]


# ============================================================================
# PongPixelEnv
# ============================================================================


struct PongPixelEnv[
    DTYPE: DType,
    HIT_REWARD: Float64 = 0.1,
    FRAME_SKIP: Int = 1,
    # Frame-stack obs memory layout. NCHW (default) = [FRAME, 84, 84] (frame-
    # outer, bit-identical to the original). NHWC = [84, 84, FRAME] (frame-inner)
    # for channels-last conv training — couple this to the agent's conv LAYOUT
    # (RainbowCNNConfig.LAYOUT). The flat OBS_DIM (28224) is identical either way.
    LAYOUT: Int = LAYOUT_NCHW,
](BoxDiscreteActionEnv & GPUDiscreteEnv & RenderableEnv & Movable):
    """Native Pong with pixel observations for CNN-based training.

    Uses the same physics as PongEnv but produces 4×84×84 pixel observations
    instead of 6D clean observations.

    Parameters:
        DTYPE: Floating point type (float32 recommended for GPU).
        HIT_REWARD: Dense ball-return shaping reward (default 0.1, forwarded
            to the inner PongEnv). Set 0.0 for clean sparse ±1 rewards — note
            FRAME_SKIP accumulates this across skipped frames, so shaping
            distorts the value scale even more here than in the clean-obs env.
        FRAME_SKIP: Number of physics steps per action (default 1).
            With FRAME_SKIP=4, each action is repeated 4 times before
            rendering and observing. Rewards are summed across skipped frames.
            If the episode terminates mid-skip, remaining frames are skipped.
        LAYOUT: Observation layout (default LAYOUT_NCHW).

    CPU: Renders to internal grayscale buffer for get_obs_list().
    GPU: Renders in-kernel, maintains per-env frame stacks in workspace.
    """

    comptime dtype = Self.DTYPE
    comptime StateType = ArcadeGameState
    comptime ActionType = ArcadeGameAction

    # GPUDiscreteEnv constants
    comptime STATE_SIZE: Int = 12
    comptime OBS_DIM: Int = PIXEL_OBS_DIM  # 4 × 84 × 84 = 28224
    comptime NUM_ACTIONS: Int = 3  # NOOP, UP, DOWN
    comptime STEP_WS_SHARED: Int = 0
    comptime STEP_WS_PER_ENV: Int = PIXEL_WS_PER_ENV  # 36625

    # Delegate to PongEnv for physics + rendering
    var inner: PongEnv[Self.DTYPE, Self.HIT_REWARD]

    # CPU pixel observation buffers
    var _frame_buf: Pointer[UInt8, MutUntrackedOrigin]  # 160×210 grayscale
    var _frame_stack: Pointer[Scalar[Self.DTYPE], MutUntrackedOrigin]  # 4×84×84
    var _frame_idx: Int

    def __init__(out self):
        self.inner = PongEnv[Self.DTYPE, Self.HIT_REWARD]()
        self._frame_buf = untracked(
            alloc[UInt8]({count = SCREEN_W * SCREEN_H}).unsafe_leak()
        )
        self._frame_stack = untracked(
            alloc[Scalar[Self.DTYPE]]({count = PIXEL_OBS_DIM}).unsafe_leak()
        )
        self._frame_idx = 0
        # Zero frame stack
        for i in range(PIXEL_OBS_DIM):
            self._frame_stack[unsafe_offset=i] = 0

    def __deinit__(deinit self):
        if Int(self._frame_buf) != 0:
            self._frame_buf.unsafe_free()
        if Int(self._frame_stack) != 0:
            self._frame_stack.unsafe_free()

    # ========================================================================
    # CPU: render game to pixel obs
    # ========================================================================

    def _render_to_buf(self):
        """Render current Pong state to internal grayscale buffer."""
        _render_pong_frame(
            self._frame_buf.as_unsafe_any_origin(),
            Int(self.inner.state[S_BALL_X]),
            Int(self.inner.state[S_BALL_Y]),
            Int(self.inner.state[S_PADDLE_Y]),
            Int(self.inner.state[S_CPU_PADDLE_Y]),
            Int(self.inner.state[S_PLAYER_SCORE]),
            Int(self.inner.state[S_CPU_SCORE]),
        )

    def _push_frame(mut self):
        """Resize framebuffer to 84×84 and push to frame stack."""
        comptime FRAME_SIZE = OBS_W * OBS_H  # 7056
        var slot_offset = self._frame_idx * FRAME_SIZE

        for dy in range(OBS_H):
            var sy0 = dy * SCREEN_H // OBS_H
            var sy1 = (dy + 1) * SCREEN_H // OBS_H
            if sy1 == sy0:
                sy1 = sy0 + 1

            for dx in range(OBS_W):
                var sx0 = dx * SCREEN_W // OBS_W
                var sx1 = (dx + 1) * SCREEN_W // OBS_W
                if sx1 == sx0:
                    sx1 = sx0 + 1

                var total: Int = 0
                var count: Int = 0
                for sy in range(sy0, sy1):
                    for sx in range(sx0, sx1):
                        total += Int(self._frame_buf[unsafe_offset=sy * SCREEN_W + sx])
                        count += 1

                self._frame_stack[unsafe_offset=slot_offset + dy * OBS_W + dx] = (
                    Scalar[Self.DTYPE](total // count) / 255.0
                )

        self._frame_idx = (self._frame_idx + 1) % FRAME_STACK

    # ========================================================================
    # Env trait — delegate to inner PongEnv
    # ========================================================================

    def reset(mut self) -> ArcadeGameState:
        var state = self.inner.reset()
        # Render initial frame and fill all 4 slots
        self._render_to_buf()
        self._frame_idx = 0
        for _ in range(FRAME_STACK):
            self._push_frame()
        return state

    def step(
        mut self, action: ArcadeGameAction, verbose: Bool = False
    ) -> Tuple[ArcadeGameState, Scalar[Self.DTYPE], Bool]:
        # FRAME_SKIP: repeat the action, accumulating rewards and stopping
        # on done — matches the GPU `step_batch` kernel. The CPU path used
        # to step ONCE regardless of FRAME_SKIP, so CPU eval of a
        # GPU-trained agent saw different dynamics and reward scale.
        # Render once per action, as on GPU.
        var result = self.inner.step(action, verbose)
        var state = result[0]
        var total_reward = result[1]
        var done = result[2]
        comptime for _skip in range(Self.FRAME_SKIP - 1):
            if not done:
                var r = self.inner.step(action, verbose)
                state = r[0]
                total_reward = total_reward + r[1]
                done = r[2]
        self._render_to_buf()
        self._push_frame()
        return (state, total_reward, done)

    def get_state(mut self) -> ArcadeGameState:
        return self.inner.get_state()

    def close(mut self):
        self.inner.close()

    def action_from_index(self, action_idx: Int) -> ArcadeGameAction:
        return self.inner.action_from_index(action_idx)

    def num_actions(self) -> Int:
        return 3

    def obs_dim(self) -> Int:
        return PIXEL_OBS_DIM

    def num_states(self) -> Int:
        return 1

    def state_to_index(self, state: ArcadeGameState) -> Int:
        return state.index

    # ========================================================================
    # BoxDiscreteActionEnv — pixel observations
    # ========================================================================

    def get_obs_list(self) -> List[Scalar[Self.DTYPE]]:
        """Return 4×84×84 pixel observations as chronological frame stack."""
        comptime FRAME_SIZE = OBS_W * OBS_H
        var obs = List[Scalar[Self.DTYPE]](capacity=PIXEL_OBS_DIM)
        comptime if Self.LAYOUT == LAYOUT_NHWC:
            # [84, 84, FRAME]: pixel-outer, frame-inner (channels-last).
            for i in range(FRAME_SIZE):
                for f in range(FRAME_STACK):
                    var read_slot = (self._frame_idx + f) % FRAME_STACK
                    obs.append(self._frame_stack[unsafe_offset=read_slot * FRAME_SIZE + i])
        else:
            # [FRAME, 84, 84]: frame-outer (channels-first, original).
            for f in range(FRAME_STACK):
                var read_slot = (
                    self._frame_idx + f
                ) % FRAME_STACK  # oldest first
                var read_base = read_slot * FRAME_SIZE
                for i in range(FRAME_SIZE):
                    obs.append(self._frame_stack[unsafe_offset=read_base + i])
        return obs^

    def reset_obs_list(mut self) -> List[Scalar[Self.DTYPE]]:
        _ = self.reset()
        return self.get_obs_list()

    def step_obs(
        mut self, action: Int
    ) -> Tuple[List[Scalar[Self.DTYPE]], Scalar[Self.DTYPE], Bool]:
        # FRAME_SKIP loop mirrors `step` above (and the GPU kernel).
        var result = self.inner._step_impl(action)
        var total_reward = result[0]
        var done = result[1]
        comptime for _skip in range(Self.FRAME_SKIP - 1):
            if not done:
                var r = self.inner._step_impl(action)
                total_reward = total_reward + r[0]
                done = r[1]
        self._render_to_buf()
        self._push_frame()
        return (self.get_obs_list(), total_reward, done)

    # ========================================================================
    # RenderableEnv — delegate to inner PongEnv
    # ========================================================================

    def init_renderer(mut self) raises -> Bool:
        return self.inner.init_renderer()

    def render_frame(mut self) raises -> None:
        self.inner.render_frame()

    def close_renderer(mut self) raises -> None:
        self.inner.close_renderer()

    def is_renderer_open(self) -> Bool:
        return self.inner.is_renderer_open()

    def check_renderer_quit(mut self) -> Bool:
        return self.inner.check_renderer_quit()

    def renderer_delay(self, ms: Int) -> None:
        self.inner.renderer_delay(ms)

    def renderer_is_paused(self) -> Bool:
        return False

    def renderer_step_once(self) -> Bool:
        return False

    # ========================================================================
    # GPU: Pixel observation step/reset kernels
    # ========================================================================

    comptime TPB = 64  # Fewer threads per block — pixel rendering is heavy

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

        comptime BLOCKS = (BATCH_SIZE + Self.TPB - 1) // Self.TPB
        var seed = Scalar[DType.uint64](rng_seed)

        # ── Kernel 1a: Physics (1 thread per env) ──
        # With FRAME_SKIP > 1, physics runs N times per action and rewards
        # accumulate. Rendering is split into the clear (1b) + draw (1c)
        # kernels below so the dominant framebuffer clear runs element-parallel
        # instead of serially inside one thread per env.
        @always_inline
        def physics_wrapper(
            states: Tile[gpu_dtype, BATCH_SIZE, STATE_SIZE],
            actions: ImmTile[gpu_dtype, BATCH_SIZE],
            rewards: Tile[gpu_dtype, BATCH_SIZE],
            dones: Tile[gpu_dtype, BATCH_SIZE],
            terminated_out: Tile[gpu_dtype, BATCH_SIZE],
            rng_seed: Scalar[DType.uint64],
        ):
            # First physics step (action applied normally)
            PongEnv[DType.float32, Self.HIT_REWARD].step_kernel[
                BATCH_SIZE, STATE_SIZE
            ](states, actions, rewards, dones, rng_seed)

            var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
            if idx >= BATCH_SIZE:
                return

            # Frame skip: repeat physics with same action, accumulate rewards
            comptime for _skip in range(Self.FRAME_SKIP - 1):
                # Skip remaining frames if episode already done
                if rebind[Scalar[gpu_dtype]](dones[idx]) < Scalar[gpu_dtype](
                    0.5
                ):
                    var prev_reward = rebind[Scalar[gpu_dtype]](rewards[idx])
                    PongEnv[DType.float32, Self.HIT_REWARD].step_kernel[
                        BATCH_SIZE, STATE_SIZE
                    ](states, actions, rewards, dones, rng_seed)
                    # Accumulate reward
                    rewards[idx] = prev_reward + rebind[Scalar[gpu_dtype]](
                        rewards[idx]
                    )

            terminated_out[idx] = dones[idx]

        ctx.enqueue_function[physics_wrapper](
            states,
            actions,
            rewards,
            dones,
            terminated_out,
            seed,
            grid_dim=(BLOCKS,),
            block_dim=(Self.TPB,),
        )

        # ── Kernel 1b: Clear framebuffers (1 thread per (env × pixel)) ──
        # The dominant render cost was `clear_frame` (160×210 = 33600 byte
        # writes) run serially in one thread per env; here it is fully
        # element-parallel + coalesced. Writes 0 to every byte, matching
        # `clear_frame`.
        comptime FB_SIZE = SCREEN_W * SCREEN_H
        comptime CLEAR_TOTAL = BATCH_SIZE * FB_SIZE
        comptime CLEAR_TPB = 256
        comptime CLEAR_BLOCKS = (CLEAR_TOTAL + CLEAR_TPB - 1) // CLEAR_TPB

        @always_inline
        def clear_wrapper(
            ws_ptr: Pointer[Scalar[gpu_dtype], MutAnyOrigin],
        ):
            var tid = Int(block_dim.x * block_idx.x + thread_idx.x)
            if tid >= CLEAR_TOTAL:
                return
            var env_idx = tid // FB_SIZE
            var pixel = tid % FB_SIZE
            var env_ws = ws_ptr.unsafe_offset(env_idx * PIXEL_WS_PER_ENV)
            var frame_buf = env_ws.unsafe_bitcast[UInt8]()
            frame_buf[unsafe_offset=pixel] = UInt8(0)

        ctx.enqueue_function[clear_wrapper](
            workspace_ptr.value(),
            grid_dim=(CLEAR_BLOCKS,),
            block_dim=(CLEAR_TPB,),
        )

        # ── Kernel 1c: Draw primitives (1 thread per env) ──
        # Paddles + ball + dashed line + scores onto the just-cleared buffer.
        # Tiny vs the clear (a few hundred pixels), so per-env is fine. Must
        # follow both physics (reads new positions) and clear (writes buffer).
        @always_inline
        def draw_wrapper(
            states: Tile[gpu_dtype, BATCH_SIZE, STATE_SIZE],
            ws_ptr: Pointer[Scalar[gpu_dtype], MutAnyOrigin],
        ):
            var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
            if idx >= BATCH_SIZE:
                return
            var env_ws = ws_ptr.unsafe_offset(idx * PIXEL_WS_PER_ENV)
            var frame_buf = env_ws.unsafe_bitcast[UInt8]()
            _draw_pong_frame(
                frame_buf,
                Int(states[idx, S_BALL_X]),
                Int(states[idx, S_BALL_Y]),
                Int(states[idx, S_PADDLE_Y]),
                Int(states[idx, S_CPU_PADDLE_Y]),
                Int(states[idx, S_PLAYER_SCORE]),
                Int(states[idx, S_CPU_SCORE]),
            )

        ctx.enqueue_function[draw_wrapper](
            states,
            workspace_ptr.value(),
            grid_dim=(BLOCKS,),
            block_dim=(Self.TPB,),
        )

        # ── Kernel 2: Resize + Frame Stack (1 thread per output pixel per env) ──
        # 84×84 = 7056 pixels per env × BATCH_SIZE envs
        comptime FRAME_SIZE = OBS_W * OBS_H  # 7056
        comptime RESIZE_TOTAL = BATCH_SIZE * FRAME_SIZE
        comptime RESIZE_TPB = 256
        comptime OBS_LAYOUT = Self.LAYOUT  # frame-stack obs layout (NCHW/NHWC)
        comptime RESIZE_BLOCKS = (RESIZE_TOTAL + RESIZE_TPB - 1) // RESIZE_TPB
        var obs_ptr = obs_buf.unsafe_ptr()

        @always_inline
        def resize_stack_wrapper(
            ws_ptr: Pointer[Scalar[gpu_dtype], MutAnyOrigin],
            obs_ptr: Pointer[Scalar[gpu_dtype], MutAnyOrigin],
        ):
            var tid = Int(block_dim.x * block_idx.x + thread_idx.x)
            if tid >= RESIZE_TOTAL:
                return

            var env_idx = tid // FRAME_SIZE
            var pixel_idx = tid % FRAME_SIZE
            var dy = pixel_idx // OBS_W
            var dx = pixel_idx % OBS_W

            var env_ws = ws_ptr.unsafe_offset(env_idx * PIXEL_WS_PER_ENV)
            var frame_buf = env_ws.unsafe_bitcast[UInt8]()

            # Box-filter resize for this single pixel
            var sy0 = dy * SCREEN_H // OBS_H
            var sy1 = (dy + 1) * SCREEN_H // OBS_H
            if sy1 == sy0:
                sy1 = sy0 + 1
            var sx0 = dx * SCREEN_W // OBS_W
            var sx1 = (dx + 1) * SCREEN_W // OBS_W
            if sx1 == sx0:
                sx1 = sx0 + 1

            var total: Int = 0
            var count: Int = 0
            for sy in range(sy0, sy1):
                for sx in range(sx0, sx1):
                    total += Int(frame_buf[unsafe_offset=sy * SCREEN_W + sx])
                    count += 1

            # Write to frame stack
            comptime WS_FRAME_STACK = FRAME_BUF_F32_SIZE
            comptime WS_FRAME_IDX = FRAME_BUF_F32_SIZE + FRAME_STACK_F32_SIZE
            var slot = Int(env_ws[unsafe_offset=WS_FRAME_IDX]) % FRAME_STACK
            var slot_base = WS_FRAME_STACK + slot * FRAME_SIZE
            env_ws[unsafe_offset=slot_base + pixel_idx] = (
                Scalar[gpu_dtype](total // count) / 255.0
            )

            # Output chronological frame stack for this pixel. NCHW =
            # f*FRAME_SIZE+pixel (frame-outer); NHWC = pixel*FRAME_STACK+f
            # (frame-inner, channels-last).
            var env_obs = obs_ptr.unsafe_offset(env_idx * PIXEL_OBS_DIM)
            for f in range(FRAME_STACK):
                var read_slot = (slot + 1 + f) % FRAME_STACK
                var read_base = WS_FRAME_STACK + read_slot * FRAME_SIZE
                comptime if OBS_LAYOUT == LAYOUT_NHWC:
                    env_obs[unsafe_offset=pixel_idx * FRAME_STACK + f] = env_ws[unsafe_offset=
                        read_base + pixel_idx
                    ]
                else:
                    env_obs[unsafe_offset=f * FRAME_SIZE + pixel_idx] = env_ws[unsafe_offset=
                        read_base + pixel_idx
                    ]

        ctx.enqueue_function[resize_stack_wrapper](
            workspace_ptr.value(),
            obs_ptr,
            grid_dim=(RESIZE_BLOCKS,),
            block_dim=(RESIZE_TPB,),
        )

        # ── Kernel 3: Advance frame index (1 thread per env) ──
        comptime WS_FRAME_IDX_OFF = FRAME_BUF_F32_SIZE + FRAME_STACK_F32_SIZE

        @always_inline
        def advance_frame_idx_wrapper(
            ws_ptr: Pointer[Scalar[gpu_dtype], MutAnyOrigin],
        ):
            var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
            if idx >= BATCH_SIZE:
                return
            var env_ws = ws_ptr.unsafe_offset(idx * PIXEL_WS_PER_ENV)
            var slot = Int(env_ws[unsafe_offset=WS_FRAME_IDX_OFF])
            env_ws[unsafe_offset=WS_FRAME_IDX_OFF] = Scalar[gpu_dtype](
                (slot + 1) % FRAME_STACK
            )

        ctx.enqueue_function[advance_frame_idx_wrapper](
            workspace_ptr.value(),
            grid_dim=(BLOCKS,),
            block_dim=(Self.TPB,),
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
        """Reset all environments. Frame stack is initialized in init_step_workspace_gpu.
        """
        var states = tile[BATCH_SIZE, STATE_SIZE](states_buf)

        comptime BLOCKS = (BATCH_SIZE + Self.TPB - 1) // Self.TPB

        @always_inline
        def reset_wrapper(
            states: Tile[gpu_dtype, BATCH_SIZE, STATE_SIZE],
        ):
            PongEnv[DType.float32].reset_kernel[BATCH_SIZE, STATE_SIZE](states)

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
        """Reset done environments and clear their frame stacks."""
        var states = tile[BATCH_SIZE, STATE_SIZE](states_buf)
        var dones = tile[BATCH_SIZE](dones_buf)

        comptime BLOCKS = (BATCH_SIZE + Self.TPB - 1) // Self.TPB

        if Bool(rng_counter_ptr):
            var counter_t = tile[1](rng_counter_ptr.value())

            @always_inline
            def selective_reset_counter_wrapper(
                states: Tile[gpu_dtype, BATCH_SIZE, STATE_SIZE],
                dones: Tile[gpu_dtype, BATCH_SIZE],
                ws_ptr: Pointer[Scalar[gpu_dtype], MutAnyOrigin],
                counter: Tile[DType.uint64, 1],
            ):
                var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
                if idx >= BATCH_SIZE:
                    return
                if dones[idx] < Scalar[gpu_dtype](0.5):
                    return

                # Reset physics
                PongEnv[DType.float32].selective_reset_kernel[
                    BATCH_SIZE, STATE_SIZE
                ](
                    states,
                    dones,
                    Scalar[DType.uint32](
                        rebind[Scalar[DType.uint64]](counter[0])
                    ),
                )

                # Clear frame stack for this env
                var env_ws = ws_ptr.unsafe_offset(idx * PIXEL_WS_PER_ENV)
                # Zero frame stack region
                for i in range(FRAME_STACK_F32_SIZE):
                    env_ws[unsafe_offset=FRAME_BUF_F32_SIZE + i] = 0.0
                # Reset frame index
                env_ws[unsafe_offset=FRAME_BUF_F32_SIZE + FRAME_STACK_F32_SIZE] = 0.0

            ctx.enqueue_function[selective_reset_counter_wrapper](
                states,
                dones,
                workspace_ptr.value(),
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
                ws_ptr: Pointer[Scalar[gpu_dtype], MutAnyOrigin],
                rng_seed: Scalar[DType.uint64],
            ):
                var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
                if idx >= BATCH_SIZE:
                    return
                if dones[idx] < Scalar[gpu_dtype](0.5):
                    return

                # Reset physics
                PongEnv[DType.float32].selective_reset_kernel[
                    BATCH_SIZE, STATE_SIZE
                ](states, dones, Scalar[DType.uint32](rng_seed))

                # Clear frame stack for this env
                var env_ws = ws_ptr.unsafe_offset(idx * PIXEL_WS_PER_ENV)
                # Zero frame stack region
                for i in range(FRAME_STACK_F32_SIZE):
                    env_ws[unsafe_offset=FRAME_BUF_F32_SIZE + i] = 0.0
                # Reset frame index
                env_ws[unsafe_offset=FRAME_BUF_F32_SIZE + FRAME_STACK_F32_SIZE] = 0.0

            ctx.enqueue_function[selective_reset_wrapper](
                states,
                dones,
                workspace_ptr.value(),
                seed,
                grid_dim=(BLOCKS,),
                block_dim=(Self.TPB,),
            )

    @staticmethod
    def init_step_workspace_gpu[
        BATCH_SIZE: Int,
    ](ctx: DeviceContext, mut workspace_buf: DeviceBuffer[gpu_dtype]) raises:
        """Initialize workspace: zero frame stacks and frame indices."""
        comptime WS_TOTAL = BATCH_SIZE * PIXEL_WS_PER_ENV
        comptime BLOCKS = (WS_TOTAL + 256 - 1) // 256

        var ws_ptr = workspace_buf.unsafe_ptr()

        @always_inline
        def init_ws_wrapper(
            ws: Pointer[Scalar[gpu_dtype], MutAnyOrigin],
        ):
            var i = Int(block_dim.x * block_idx.x + thread_idx.x)
            if i >= WS_TOTAL:
                return
            ws[unsafe_offset=i] = 0.0

        ctx.enqueue_function[init_ws_wrapper](
            ws_ptr,
            grid_dim=(BLOCKS,),
            block_dim=(256,),
        )

    @staticmethod
    def update_curriculum_gpu(
        ctx: DeviceContext,
        mut workspace_buf: DeviceBuffer[gpu_dtype],
        curriculum_values: List[Scalar[gpu_dtype]],
    ) raises:
        pass
