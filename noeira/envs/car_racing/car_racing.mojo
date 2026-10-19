"""CarRacing GPU environment using the physics2d/car/ module.

DEPRECATED — superseded by the CarRacing successors in this package:
  * `CarRacingMB` (car_racing_mb.mojo)   — multi-body slip physics, the env
    the Dreamer/Rainbow CarRacing training stack uses (fixes this env's
    single-body spin-in-place bug off-track).
  * `CarRacingDiscrete` (car_racing_discrete.mojo) — discrete-action variant.
  * `CarRacingPixel` (car_racing_pixel.mojo)       — pixel observations.

Known divergences kept for reference (audited 2026-07): the GPU reward here
is velocity-shaped while the CPU reward is Gymnasium tile-based (and 10x
below Gymnasium's 1000/N tile scale); GPU track generation is a simplified
trace that need not close, unlike the Gymnasium-faithful `track.mojo` used
on CPU. Do not start new work on this env.

This implementation uses the modular car physics components:
- CarRacingLayout for compile-time layout computation
- CarDynamics for slip-based tire friction physics
- CarPhysicsKernel for fused GPU physics step
- TileCollision for friction zone lookup

The flat state layout is compatible with GPU batch operations.
All physics data is packed per-environment for efficient GPU access.

Reference: gymnasium/envs/box2d/car_racing.py
"""

from std.math import sqrt, cos, sin, pi
from layout import Layout
from noeira.core.tiles import Tile, ImmTile, tile
from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceContext, DeviceBuffer
from std.random.philox import Random as PhiloxRandom

from std.memory import alloc
from noeira.core import (
    BoxContinuousActionEnv,
    GPUContinuousEnv,
    Action,
    RenderableEnv,
)
from noeira.render import (
    Renderer2D,
    RotatingCamera,
    Transform2D,
    SDL_Color,
    SDL_Point,
    Vec2 as RenderVec2,
    car_red,
    black,
    white,
    rgb,
)

from .state import CarRacingState
from .action import CarRacingAction
from .constants import CRConstants
from .track import TrackTile, TrackGenerator

from noeira.physics2d import dtype, TPB
from noeira.physics2d.car import (
    CarRacingLayout,
    CarDynamics,
    CarPhysicsKernel,
    TileCollision,
    WheelFriction,
)
from noeira.physics2d.car.constants import (
    HULL_X,
    HULL_Y,
    HULL_ANGLE,
    HULL_VX,
    HULL_VY,
    HULL_OMEGA,
    HULL_STATE_SIZE,
    WHEEL_OMEGA,
    WHEEL_JOINT_ANGLE,
    WHEEL_PHASE,
    WHEEL_STATE_SIZE,
    CTRL_STEERING,
    CTRL_GAS,
    CTRL_BRAKE,
    NUM_WHEELS,
    TILE_DATA_SIZE,
    MAX_TRACK_TILES,
    WHEEL_POS_FL_X,
    WHEEL_POS_FL_Y,
    WHEEL_POS_FR_X,
    WHEEL_POS_FR_Y,
    WHEEL_POS_RL_X,
    WHEEL_POS_RL_Y,
    WHEEL_POS_RR_X,
    WHEEL_POS_RR_Y,
)
from noeira.physics2d.car.layout import (
    META_STEP_COUNT,
    META_TOTAL_REWARD,
    META_DONE,
    META_TRUNCATED,
    META_TILES_VISITED,
    META_NUM_TILES,
)


# =============================================================================
# CarRacing Environment
# =============================================================================


struct CarRacing[DTYPE: DType](
    BoxContinuousActionEnv, Copyable, GPUContinuousEnv, Movable, RenderableEnv
):
    """CarRacing environment with GPU-accelerated physics.

    This environment uses the physics2d/car/ module for slip-based
    tire friction physics with GPU acceleration.

    Features:
    - Top-down racing with procedural track
    - 4-wheel friction model with slip physics
    - Continuous action space: [steering, gas, brake]
    - State-based observation (13D) for RL algorithms

    CPU Mode:
    - Uses CarDynamics.step_single_env() for physics
    - Track stored in List[TrackTile]

    GPU Mode (batch):
    - Uses CarPhysicsKernel.step_gpu() for batched physics
    - Track stored in GPU buffer (shared across envs)
    """

    # Required trait aliases
    comptime dtype = Self.DTYPE  # For Env trait
    comptime Layout = CRConstants.Layout
    comptime STATE_SIZE: Int = CRConstants.STATE_SIZE
    comptime OBS_DIM: Int = CRConstants.OBS_DIM
    comptime ACTION_DIM: Int = CRConstants.ACTION_DIM
    comptime STEP_WS_SHARED: Int = 0
    comptime STEP_WS_PER_ENV: Int = 0
    comptime StateType = CarRacingState[Self.DTYPE]
    comptime ActionType = CarRacingAction[Self.DTYPE]
    comptime NAME: String = "CarRacing"

    # Track generator (Float64 for precision)
    var track: TrackGenerator[DType.float64]

    # Physics state buffer (using physics2d dtype = float32)
    var state_buffer: List[Scalar[dtype]]

    # Track tiles buffer (for TileCollision)
    var tiles_buffer: List[Scalar[dtype]]

    # Episode state
    var step_count: Int
    var total_reward: Float64
    var done: Bool
    var truncated: Bool
    var tiles_visited: Int

    # Configuration
    var max_steps: Int
    var lap_complete_percent: Float64
    var domain_randomize: Bool

    # Cached state
    var cached_state: CarRacingState[Self.dtype]

    # Renderer (RenderableEnv)
    var _renderer: Optional[Pointer[Renderer2D, MutUntrackedOrigin]]
    var _renderer_initialized: Bool

    # =========================================================================
    # Initialization
    # =========================================================================

    def __init__(
        out self,
        max_steps: Int = CRConstants.MAX_STEPS,
        lap_complete_percent: Float64 = CRConstants.LAP_COMPLETE_PERCENT,
        domain_randomize: Bool = False,
    ):
        """Initialize the CarRacing environment.

        Args:
            max_steps: Maximum steps per episode (0 = unlimited).
            lap_complete_percent: Fraction of tiles to complete lap.
            domain_randomize: Randomize colors each reset.
        """
        # Initialize track
        self.track = TrackGenerator[DType.float64]()

        # Initialize state buffer
        self.state_buffer = List[Scalar[dtype]](capacity=Self.STATE_SIZE)
        for _ in range(Self.STATE_SIZE):
            self.state_buffer.append(Scalar[dtype](0.0))

        # Initialize tiles buffer
        var tiles_size = MAX_TRACK_TILES * TILE_DATA_SIZE
        self.tiles_buffer = List[Scalar[dtype]](capacity=tiles_size)
        for _ in range(tiles_size):
            self.tiles_buffer.append(Scalar[dtype](0.0))

        # Episode state
        self.step_count = 0
        self.total_reward = 0.0
        self.done = False
        self.truncated = False
        self.tiles_visited = 0

        # Configuration
        self.max_steps = max_steps
        self.lap_complete_percent = lap_complete_percent
        self.domain_randomize = domain_randomize

        # Cached state
        self.cached_state = CarRacingState[Self.dtype]()

        # Renderer
        self._renderer = None
        self._renderer_initialized = False

    def __init__(out self, *, copy: Self):
        self.track = TrackGenerator[DType.float64]()
        self.track.track = copy.track.track.copy()
        self.track.track_length = copy.track.track_length
        self.state_buffer = copy.state_buffer.copy()
        self.tiles_buffer = copy.tiles_buffer.copy()
        self.step_count = copy.step_count
        self.total_reward = copy.total_reward
        self.done = copy.done
        self.truncated = copy.truncated
        self.tiles_visited = copy.tiles_visited
        self.max_steps = copy.max_steps
        self.lap_complete_percent = copy.lap_complete_percent
        self.domain_randomize = copy.domain_randomize
        self.cached_state = copy.cached_state
        # Do not copy renderer — reset to null
        self._renderer = None
        self._renderer_initialized = False

    def __init__(out self, *, deinit move: Self):
        self.track = move.track^
        self.state_buffer = move.state_buffer^
        self.tiles_buffer = move.tiles_buffer^
        self.step_count = move.step_count
        self.total_reward = move.total_reward
        self.done = move.done
        self.truncated = move.truncated
        self.tiles_visited = move.tiles_visited
        self.max_steps = move.max_steps
        self.lap_complete_percent = move.lap_complete_percent
        self.domain_randomize = move.domain_randomize
        self.cached_state = move.cached_state^
        # Transfer renderer ownership
        self._renderer = move._renderer
        self._renderer_initialized = move._renderer_initialized

    # =========================================================================
    # Env Trait Methods
    # =========================================================================

    def reset(mut self) -> Self.StateType:
        """Reset the environment and return initial state."""
        # Generate track (procedural random track)
        var seed = UInt64(self.step_count + 42)  # Different seed each reset
        self.track.generate_random_track(seed, verbose=False)

        # Copy track to tiles buffer
        self._update_tiles_buffer()

        # Reset episode state
        self.step_count = 0
        self.total_reward = 0.0
        self.done = False
        self.truncated = False
        self.tiles_visited = 0

        # Get start position
        var start = self.track.get_start_position()
        var start_x = Float64(start[0])
        var start_y = Float64(start[1])
        var start_angle = Float64(start[2])

        # Initialize state buffer
        self._init_state_buffer(start_x, start_y, start_angle)

        # Update observation
        self._update_observation()

        # Cache and return state
        self._update_cached_state()
        return self.cached_state

    def step(
        mut self, action: Self.ActionType, verbose: Bool = False
    ) -> Tuple[Self.StateType, Scalar[Self.dtype], Bool]:
        """Take action and return (next_state, reward, done)."""
        # Clamp and remap action inputs
        var steering = clamp(Float64(action.steering), -1.0, 1.0)
        var gas = clamp((Float64(action.gas) + 1.0) * 0.5, 0.0, 1.0)
        var brake = clamp((Float64(action.brake) + 1.0) * 0.5, 0.0, 1.0)

        # Store controls in state buffer
        self._set_control(steering, gas, brake)

        # Step physics
        self._step_physics()

        # Check tile visits and compute reward
        var step_reward = self._check_tiles_and_reward()

        # Increment step count
        self.step_count += 1

        # Check termination conditions
        self._check_termination(step_reward)

        # Update observation
        self._update_observation()

        # Cache and return state
        self._update_cached_state()
        return (self.cached_state, Scalar[Self.dtype](step_reward), self.done)

    def get_state(mut self) -> Self.StateType:
        """Get current state."""
        return self.cached_state

    def render(mut self, mut renderer: Renderer2D):
        """Render the environment with rotating camera following the car."""
        # Get car state from physics buffer
        var state = self._get_state_tensor()
        var hull_x = Float64(
            rebind[Scalar[dtype]](state[0, Self.Layout.HULL_OFFSET + HULL_X])
        )
        var hull_y = Float64(
            rebind[Scalar[dtype]](state[0, Self.Layout.HULL_OFFSET + HULL_Y])
        )
        var hull_angle = Float64(
            rebind[Scalar[dtype]](
                state[0, Self.Layout.HULL_OFFSET + HULL_ANGLE]
            )
        )
        var hull_vx = Float64(
            rebind[Scalar[dtype]](state[0, Self.Layout.HULL_OFFSET + HULL_VX])
        )
        var hull_vy = Float64(
            rebind[Scalar[dtype]](state[0, Self.Layout.HULL_OFFSET + HULL_VY])
        )

        # Background color (grass green)
        var bg_color = SDL_Color(102, 204, 102, 255)
        if not renderer.begin_frame_with_color(bg_color):
            return

        # Create rotating camera - follows car with rotation
        # Camera zoom interpolates during first second
        var t = Float64(self.step_count) * CRConstants.DT
        var zoom = CRConstants.ZOOM * CRConstants.SCALE * min(
            t, 1.0
        ) + 0.1 * CRConstants.SCALE * max(1.0 - t, 0.0)

        # Screen center for camera (car appears in lower portion)
        var screen_center_x = Float64(CRConstants.WINDOW_W) / 2.0
        var screen_center_y = Float64(CRConstants.WINDOW_H) * 3.0 / 4.0

        var camera = renderer.make_rotating_camera_offset(
            hull_x,
            hull_y,
            -hull_angle,  # Negative to rotate view opposite to car
            zoom,
            screen_center_x,
            screen_center_y,
        )

        # Draw grass patches
        self._draw_grass(renderer, camera)

        # Draw track
        self._draw_track(renderer, camera)

        # Draw car
        self._draw_car(renderer, camera, hull_x, hull_y, hull_angle)

        # Draw HUD
        self._draw_info(renderer, hull_vx, hull_vy)

        renderer.flip()

    def _draw_grass(self, mut renderer: Renderer2D, camera: RotatingCamera):
        """Draw grass patches in a checkerboard pattern."""
        var grass_clr = SDL_Color(68, 160, 68, 255)  # Darker grass for pattern

        for gx in range(-10, 10, 2):
            for gy in range(-10, 10, 2):
                var world_x = Float64(gx) * CRConstants.GRASS_DIM
                var world_y = Float64(gy) * CRConstants.GRASS_DIM

                # Skip if too far from camera
                var dist = sqrt(
                    (world_x - camera.x) ** 2 + (world_y - camera.y) ** 2
                )
                if dist > CRConstants.PLAYFIELD:
                    continue

                # Grass quad corners
                var vertices = List[RenderVec2]()
                vertices.append(RenderVec2(world_x, world_y))
                vertices.append(
                    RenderVec2(world_x + CRConstants.GRASS_DIM, world_y)
                )
                vertices.append(
                    RenderVec2(
                        world_x + CRConstants.GRASS_DIM,
                        world_y + CRConstants.GRASS_DIM,
                    )
                )
                vertices.append(
                    RenderVec2(world_x, world_y + CRConstants.GRASS_DIM)
                )

                renderer.draw_polygon_rotating(
                    vertices, camera, grass_clr, filled=True
                )

    def _draw_track(self, mut renderer: Renderer2D, camera: RotatingCamera):
        """Draw track tiles."""
        var road_clr = SDL_Color(102, 102, 102, 255)  # Gray road

        for i in range(self.track.track_length):
            var tile = self.track.track[i]

            # Check if tile is visible (rough culling)
            var dist = sqrt(
                (Float64(tile.center_x) - camera.x) ** 2
                + (Float64(tile.center_y) - camera.y) ** 2
            )
            if dist > 500.0:
                continue

            # Tile color with slight variation
            var c = UInt8(Int(0.01 * Float64(i % 3) * 255.0))
            var tile_color = SDL_Color(
                UInt8(min(Int(road_clr.r) + Int(c), 255)),
                UInt8(min(Int(road_clr.g) + Int(c), 255)),
                UInt8(min(Int(road_clr.b) + Int(c), 255)),
                255,
            )

            # Green tint for visited tiles
            if tile.visited:
                tile_color = SDL_Color(
                    UInt8(max(Int(tile_color.r) - 30, 0)),
                    UInt8(min(Int(tile_color.g) + 30, 255)),
                    UInt8(max(Int(tile_color.b) - 30, 0)),
                    255,
                )

            # Create polygon vertices
            var vertices = List[RenderVec2]()
            vertices.append(RenderVec2(Float64(tile.v0_x), Float64(tile.v0_y)))
            vertices.append(RenderVec2(Float64(tile.v1_x), Float64(tile.v1_y)))
            vertices.append(RenderVec2(Float64(tile.v2_x), Float64(tile.v2_y)))
            vertices.append(RenderVec2(Float64(tile.v3_x), Float64(tile.v3_y)))

            renderer.draw_polygon_rotating(
                vertices, camera, tile_color, filled=True
            )

    def _draw_car(
        self,
        mut renderer: Renderer2D,
        camera: RotatingCamera,
        hull_x: Float64,
        hull_y: Float64,
        hull_angle: Float64,
    ):
        """Draw the car (hull and wheels)."""
        var state = self._get_state_tensor()

        # Hull transform
        var hull_transform = Transform2D(hull_x, hull_y, hull_angle)

        # Hull color
        var hull_color = car_red()
        var car_size = CRConstants.SIZE

        # Hull polygon 1 (front spoiler)
        var hull1 = List[RenderVec2]()
        hull1.append(RenderVec2(-60.0 * car_size, 130.0 * car_size))
        hull1.append(RenderVec2(60.0 * car_size, 130.0 * car_size))
        hull1.append(RenderVec2(60.0 * car_size, 110.0 * car_size))
        hull1.append(RenderVec2(-60.0 * car_size, 110.0 * car_size))
        renderer.draw_transformed_polygon_rotating(
            hull1, hull_transform, camera, hull_color, filled=True
        )

        # Hull polygon 2 (cabin)
        var hull2 = List[RenderVec2]()
        hull2.append(RenderVec2(-15.0 * car_size, 120.0 * car_size))
        hull2.append(RenderVec2(15.0 * car_size, 120.0 * car_size))
        hull2.append(RenderVec2(20.0 * car_size, 20.0 * car_size))
        hull2.append(RenderVec2(-20.0 * car_size, 20.0 * car_size))
        renderer.draw_transformed_polygon_rotating(
            hull2, hull_transform, camera, hull_color, filled=True
        )

        # Hull polygon 3 (body)
        var hull3 = List[RenderVec2]()
        hull3.append(RenderVec2(25.0 * car_size, 20.0 * car_size))
        hull3.append(RenderVec2(50.0 * car_size, -10.0 * car_size))
        hull3.append(RenderVec2(50.0 * car_size, -40.0 * car_size))
        hull3.append(RenderVec2(20.0 * car_size, -90.0 * car_size))
        hull3.append(RenderVec2(-20.0 * car_size, -90.0 * car_size))
        hull3.append(RenderVec2(-50.0 * car_size, -40.0 * car_size))
        hull3.append(RenderVec2(-50.0 * car_size, -10.0 * car_size))
        hull3.append(RenderVec2(-25.0 * car_size, 20.0 * car_size))
        renderer.draw_transformed_polygon_rotating(
            hull3, hull_transform, camera, hull_color, filled=True
        )

        # Hull polygon 4 (rear spoiler)
        var hull4 = List[RenderVec2]()
        hull4.append(RenderVec2(-50.0 * car_size, -120.0 * car_size))
        hull4.append(RenderVec2(50.0 * car_size, -120.0 * car_size))
        hull4.append(RenderVec2(50.0 * car_size, -90.0 * car_size))
        hull4.append(RenderVec2(-50.0 * car_size, -90.0 * car_size))
        renderer.draw_transformed_polygon_rotating(
            hull4, hull_transform, camera, hull_color, filled=True
        )

        # Draw wheels
        var wheel_color = black()
        var hw = 14.0 * car_size  # Wheel width
        var hr = 27.0 * car_size  # Wheel radius

        # Wheel positions (local coords) - FL, FR, RL, RR
        var wheel_pos_x = Array[Float64, 4](
            fill=[
                WHEEL_POS_FL_X,
                WHEEL_POS_FR_X,
                WHEEL_POS_RL_X,
                WHEEL_POS_RR_X,
            ]
        )
        var wheel_pos_y = Array[Float64, 4](
            fill=[
                WHEEL_POS_FL_Y,
                WHEEL_POS_FR_Y,
                WHEEL_POS_RL_Y,
                WHEEL_POS_RR_Y,
            ]
        )

        for i in range(4):
            # Get wheel joint angle from state
            var wheel_off = Self.Layout.WHEELS_OFFSET + i * WHEEL_STATE_SIZE
            var joint_angle = Float64(
                rebind[Scalar[dtype]](state[0, wheel_off + WHEEL_JOINT_ANGLE])
            )

            # World position of wheel
            var local_x = wheel_pos_x[i]
            var local_y = wheel_pos_y[i]
            var c = cos(hull_angle)
            var s = sin(hull_angle)
            var world_x = hull_x + local_x * c - local_y * s
            var world_y = hull_y + local_x * s + local_y * c

            # Wheel angle (hull angle + joint angle for front wheels)
            var wheel_angle = hull_angle
            if i < 2:  # Front wheels have steering
                wheel_angle += joint_angle

            var wheel_transform = Transform2D(world_x, world_y, wheel_angle)

            # Draw wheel
            var wheel_verts = List[RenderVec2]()
            wheel_verts.append(RenderVec2(-hw, hr))
            wheel_verts.append(RenderVec2(hw, hr))
            wheel_verts.append(RenderVec2(hw, -hr))
            wheel_verts.append(RenderVec2(-hw, -hr))
            renderer.draw_transformed_polygon_rotating(
                wheel_verts, wheel_transform, camera, wheel_color, filled=True
            )

    def _draw_info(self, mut renderer: Renderer2D, vx: Float64, vy: Float64):
        """Draw HUD info panel."""
        var W = Float64(CRConstants.WINDOW_W)
        var H = Float64(CRConstants.WINDOW_H)
        var s = W / 40.0  # Scale unit
        var h = H / 40.0  # Height unit

        # Draw black background panel
        var bg_color = SDL_Color(0, 0, 0, 255)
        renderer.draw_rect(0, Int(H - 5 * h), Int(W), Int(5 * h), bg_color, 0)

        # Calculate speed
        var speed = sqrt(vx * vx + vy * vy)

        # Draw speed indicator (white bar)
        if speed > 0.0001:
            var speed_height = 0.02 * speed
            self._draw_vertical_indicator(
                renderer, 5, speed_height, white(), s, h, Int(H)
            )

        # Get wheel omegas from physics buffer
        var state = self._get_state_tensor()
        for i in range(4):
            var wheel_off = Self.Layout.WHEELS_OFFSET + i * WHEEL_STATE_SIZE
            var omega = Float64(
                rebind[Scalar[dtype]](state[0, wheel_off + WHEEL_OMEGA])
            )

            var abs_omega = omega if omega >= 0.0 else -omega
            if abs_omega > 0.0001:
                # Front wheels in blue, rear in purple
                var color = SDL_Color(0, 0, 255, 255) if i < 2 else SDL_Color(
                    128, 0, 255, 255
                )
                self._draw_vertical_indicator(
                    renderer, 7 + i, 0.01 * omega, color, s, h, Int(H)
                )

        # Draw progress indicator (green)
        var progress = Float64(self.tiles_visited) / Float64(
            max(self.track.track_length, 1)
        )
        self._draw_vertical_indicator(
            renderer,
            15,
            progress * 3.0,
            SDL_Color(0, 255, 0, 255),
            s,
            h,
            Int(H),
        )

    def _draw_vertical_indicator(
        self,
        mut renderer: Renderer2D,
        x_pos: Int,
        height: Float64,
        color: SDL_Color,
        s: Float64,
        h: Float64,
        screen_h: Int,
    ):
        """Draw a vertical indicator bar inside the HUD panel."""
        var x = Int(Float64(x_pos) * s)
        var bar_h = min(height * h, 4.0 * h)
        var abs_bar_h = bar_h if bar_h >= 0.0 else -bar_h

        # HUD panel occupies bottom 5*h pixels
        # Baseline is in the middle of the HUD panel
        var baseline = screen_h - Int(2.5 * h)  # Middle of HUD

        if bar_h >= 0.0:
            # Positive: draw upward from baseline
            var y = baseline - Int(abs_bar_h)
            renderer.draw_rect(x, y, Int(s), Int(abs_bar_h) + 1, color, 0)
        else:
            # Negative: draw downward from baseline
            renderer.draw_rect(
                x, baseline, Int(s), Int(abs_bar_h) + 1, color, 0
            )

    def close(mut self):
        """Clean up resources."""
        if self._renderer_initialized:
            self._renderer.value()[].close()
            self._renderer.value().unsafe_free()
            self._renderer_initialized = False

    # =========================================================================
    # RenderableEnv Trait Implementation
    # =========================================================================

    def init_renderer(mut self) raises -> Bool:
        """Initialize the SDL2 renderer."""
        if self._renderer_initialized:
            return True
        self._renderer = alloc[Renderer2D]({count = 1}).unsafe_leak()
        self._renderer.value().unsafe_write(Renderer2D())
        self._renderer_initialized = True
        return True

    def render_frame(mut self) raises -> None:
        """Render the current frame using the internal renderer."""
        if not self._renderer_initialized:
            return
        self.render(self._renderer.value()[])

    def close_renderer(mut self) raises -> None:
        """Close and free the SDL2 renderer."""
        if not self._renderer_initialized:
            return
        self._renderer.value()[].close()
        self._renderer.value().unsafe_free()
        self._renderer_initialized = False

    def is_renderer_open(self) -> Bool:
        """Return True if the renderer window is open."""
        if not self._renderer_initialized:
            return False
        return not self._renderer.value()[].get_should_quit()

    def check_renderer_quit(mut self) -> Bool:
        """Return True if the renderer has received a quit event."""
        if not self._renderer_initialized:
            return False
        return self._renderer.value()[].get_should_quit()

    def renderer_delay(self, ms: Int) -> None:
        """Delay for frame rate control."""
        if not self._renderer_initialized:
            return
        self._renderer.value()[].renderer_delay(ms)

    def renderer_is_paused(self) -> Bool:
        return False

    def renderer_step_once(self) -> Bool:
        return False

    # =========================================================================
    # BoxContinuousActionEnv Trait Methods
    # =========================================================================

    def get_obs_list(self) -> List[Scalar[Self.dtype]]:
        """Return observation as list."""
        return self.cached_state.to_list_typed[Self.dtype]()

    def reset_obs_list(mut self) -> List[Scalar[Self.dtype]]:
        """Reset and return initial observation."""
        var state = self.reset()
        return state.to_list_typed[Self.dtype]()

    def obs_dim(self) -> Int:
        """Observation dimension: 13."""
        return Self.OBS_DIM

    def action_dim(self) -> Int:
        """Action dimension: 3."""
        return Self.ACTION_DIM

    def action_low(self) -> Scalar[Self.dtype]:
        """Action lower bound: -1.0."""
        return -1.0

    def action_high(self) -> Scalar[Self.dtype]:
        """Action upper bound: 1.0."""
        return 1.0

    def step_continuous[
        DTYPE_SC: DType
    ](mut self, action: Scalar[DTYPE_SC]) -> Tuple[
        List[Scalar[DTYPE_SC]], Scalar[DTYPE_SC], Bool
    ]:
        """Step with single action (applies to steering only)."""
        var act = CarRacingAction[Self.dtype](
            Scalar[Self.dtype](action), 0.0, 0.0
        )
        var result = self.step(act)
        var obs_typed = result[0].to_list_typed[Self.dtype]()
        var obs = List[Scalar[DTYPE_SC]](capacity=len(obs_typed))
        for i in range(len(obs_typed)):
            obs.append(Scalar[DTYPE_SC](obs_typed[i]))
        return (
            obs^,
            Scalar[DTYPE_SC](result[1]),
            result[2],
        )

    def step_continuous_vec[
        DTYPE_VEC: DType
    ](
        mut self, action: List[Scalar[DTYPE_VEC]], verbose: Bool = False
    ) -> Tuple[List[Scalar[DTYPE_VEC]], Scalar[DTYPE_VEC], Bool]:
        """Step with action vector (DTYPE convenience method)."""
        var typed_action = List[Scalar[Self.dtype]]()
        for i in range(len(action)):
            typed_action.append(Scalar[Self.dtype](action[i]))
        var act = CarRacingAction[Self.dtype].from_list(typed_action)
        var result = self.step(act)
        return (
            result[0].to_list_typed[DTYPE_VEC](),
            Scalar[DTYPE_VEC](result[1]),
            result[2],
        )

    # =========================================================================
    # Internal State Buffer Access
    # =========================================================================

    def _get_state_tensor(
        self,
    ) -> Tile[dtype, 1, Self.STATE_SIZE]:
        """Get state tensor view (BATCH=1 for CPU mode)."""
        return tile[1, Self.STATE_SIZE](rebind[Pointer[Scalar[dtype], MutAnyOrigin]]( self.state_buffer.unsafe_ptr() ))

    def _get_tiles_tensor(
        self,
    ) -> Tile[dtype, MAX_TRACK_TILES, TILE_DATA_SIZE]:
        """Get tiles tensor view."""
        return tile[MAX_TRACK_TILES, TILE_DATA_SIZE](rebind[Pointer[Scalar[dtype], MutAnyOrigin]]( self.tiles_buffer.unsafe_ptr() ))

    def _init_state_buffer(
        mut self,
        x: Float64,
        y: Float64,
        angle: Float64,
    ):
        """Initialize state buffer with starting position."""
        var state = self._get_state_tensor()

        # Clear buffer
        for i in range(Self.STATE_SIZE):
            state[0, i] = Scalar[dtype](0.0)

        # Set hull state
        state[0, Self.Layout.HULL_OFFSET + HULL_X] = Scalar[dtype](x)
        state[0, Self.Layout.HULL_OFFSET + HULL_Y] = Scalar[dtype](y)
        state[0, Self.Layout.HULL_OFFSET + HULL_ANGLE] = Scalar[dtype](angle)

    def _set_control(
        mut self,
        steering: Float64,
        gas: Float64,
        brake: Float64,
    ):
        """Set control inputs in state buffer."""
        var state = self._get_state_tensor()
        state[0, Self.Layout.CONTROLS_OFFSET + CTRL_STEERING] = Scalar[dtype](
            steering
        )
        state[0, Self.Layout.CONTROLS_OFFSET + CTRL_GAS] = Scalar[dtype](gas)
        state[0, Self.Layout.CONTROLS_OFFSET + CTRL_BRAKE] = Scalar[dtype](
            brake
        )

    def _update_tiles_buffer(mut self):
        """Copy track tiles to tiles buffer."""
        var tiles = self._get_tiles_tensor()
        self.track.to_buffer[MAX_TRACK_TILES](tiles)

    # =========================================================================
    # Physics Step
    # =========================================================================

    def _step_physics(mut self):
        """Step physics using CarDynamics."""
        var state = self._get_state_tensor()
        var tiles = self._get_tiles_tensor()
        var dt = Scalar[dtype](CRConstants.DT)

        CarDynamics.step_with_obs[
            1,  # BATCH = 1 for CPU mode
            Self.STATE_SIZE,
            Self.Layout.OBS_OFFSET,
            Self.Layout.OBS_DIM,
            Self.Layout.HULL_OFFSET,
            Self.Layout.WHEELS_OFFSET,
            Self.Layout.CONTROLS_OFFSET,
            MAX_TRACK_TILES,
        ](0, state, tiles, self.track.track_length, dt)

    # =========================================================================
    # Reward and Termination
    # =========================================================================

    def _check_tiles_and_reward(mut self) -> Float64:
        """Check tile visits and compute reward."""
        var state = self._get_state_tensor()
        var tiles = self._get_tiles_tensor()

        # Get hull position (use rebind to extract Scalar from LayoutTensor)
        var hull_x = rebind[Scalar[dtype]](
            state[0, Self.Layout.HULL_OFFSET + HULL_X]
        )
        var hull_y = rebind[Scalar[dtype]](
            state[0, Self.Layout.HULL_OFFSET + HULL_Y]
        )

        # Check which tile we're on
        var tile_idx = TileCollision.check_tile_visited[MAX_TRACK_TILES](
            hull_x, hull_y, tiles, self.track.track_length
        )

        # Reward for visiting new tiles
        var step_reward: Float64 = -0.1  # Time penalty

        if tile_idx >= 0:
            if self.track.mark_tile_visited(tile_idx):
                # New tile visited!
                self.tiles_visited += 1
                step_reward += 100.0 / Float64(self.track.track_length)

        self.total_reward += step_reward
        return step_reward

    def _check_termination(mut self, mut step_reward: Float64):
        """Check termination conditions."""
        var state = self._get_state_tensor()

        # Get hull position (rebind to Scalar, then cast to Float64)
        var hull_x = Float64(
            rebind[Scalar[dtype]](state[0, Self.Layout.HULL_OFFSET + HULL_X])
        )
        var hull_y = Float64(
            rebind[Scalar[dtype]](state[0, Self.Layout.HULL_OFFSET + HULL_Y])
        )

        var playfield = Float64(CRConstants.PLAYFIELD)

        # Check if off playfield (use explicit comparison)
        var abs_x = hull_x if hull_x >= 0.0 else -hull_x
        var abs_y = hull_y if hull_y >= 0.0 else -hull_y
        if abs_x > playfield or abs_y > playfield:
            self.done = True
            step_reward = -100.0

        # Check lap completion
        var progress = Float64(self.tiles_visited) / Float64(
            max(self.track.track_length, 1)
        )
        if progress >= self.lap_complete_percent:
            self.done = True

        # Check max steps (truncation)
        if self.max_steps > 0 and self.step_count >= self.max_steps:
            self.done = True
            self.truncated = True

    # =========================================================================
    # Observation Update
    # =========================================================================

    def _update_observation(mut self):
        """Update observation in state buffer (called after physics step)."""
        # Copy hull state to observation buffer
        var state = self._get_state_tensor()

        # Hull state (6 values)
        state[0, Self.Layout.OBS_OFFSET + 0] = state[
            0, Self.Layout.HULL_OFFSET + HULL_X
        ]
        state[0, Self.Layout.OBS_OFFSET + 1] = state[
            0, Self.Layout.HULL_OFFSET + HULL_Y
        ]
        state[0, Self.Layout.OBS_OFFSET + 2] = state[
            0, Self.Layout.HULL_OFFSET + HULL_ANGLE
        ]
        state[0, Self.Layout.OBS_OFFSET + 3] = state[
            0, Self.Layout.HULL_OFFSET + HULL_VX
        ]
        state[0, Self.Layout.OBS_OFFSET + 4] = state[
            0, Self.Layout.HULL_OFFSET + HULL_VY
        ]
        state[0, Self.Layout.OBS_OFFSET + 5] = state[
            0, Self.Layout.HULL_OFFSET + HULL_OMEGA
        ]

        # Front wheel angles (2 values)
        var fl_off = Self.Layout.WHEELS_OFFSET + 0 * WHEEL_STATE_SIZE
        var fr_off = Self.Layout.WHEELS_OFFSET + 1 * WHEEL_STATE_SIZE
        state[0, Self.Layout.OBS_OFFSET + 6] = state[
            0, fl_off + WHEEL_JOINT_ANGLE
        ]
        state[0, Self.Layout.OBS_OFFSET + 7] = state[
            0, fr_off + WHEEL_JOINT_ANGLE
        ]

        # Wheel angular velocities (4 values)
        for wheel in range(NUM_WHEELS):
            var wheel_off = Self.Layout.WHEELS_OFFSET + wheel * WHEEL_STATE_SIZE
            state[0, Self.Layout.OBS_OFFSET + 8 + wheel] = state[
                0, wheel_off + WHEEL_OMEGA
            ]

        # Speed indicator (computed from vx, vy)
        var vx = rebind[Scalar[dtype]](
            state[0, Self.Layout.HULL_OFFSET + HULL_VX]
        )
        var vy = rebind[Scalar[dtype]](
            state[0, Self.Layout.HULL_OFFSET + HULL_VY]
        )
        var speed = sqrt(vx * vx + vy * vy)
        state[0, Self.Layout.OBS_OFFSET + 12] = speed

    def _update_cached_state(mut self):
        """Update cached state from buffer."""
        var state = self._get_state_tensor()
        var obs_off = Self.Layout.OBS_OFFSET

        # Extract values using rebind and convert to Float64
        self.cached_state.x = rebind[Scalar[Self.dtype]](state[0, obs_off + 0])
        self.cached_state.y = rebind[Scalar[Self.dtype]](state[0, obs_off + 1])
        self.cached_state.angle = rebind[Scalar[Self.dtype]](
            state[0, obs_off + 2]
        )
        self.cached_state.vx = rebind[Scalar[Self.dtype]](state[0, obs_off + 3])
        self.cached_state.vy = rebind[Scalar[Self.dtype]](state[0, obs_off + 4])
        self.cached_state.angular_velocity = rebind[Scalar[Self.dtype]](
            state[0, obs_off + 5]
        )
        self.cached_state.wheel_angle_fl = rebind[Scalar[Self.dtype]](
            state[0, obs_off + 6]
        )
        self.cached_state.wheel_angle_fr = rebind[Scalar[Self.dtype]](
            state[0, obs_off + 7]
        )
        self.cached_state.wheel_omega_fl = rebind[Scalar[Self.dtype]](
            state[0, obs_off + 8]
        )

        self.cached_state.wheel_omega_fr = rebind[Scalar[Self.dtype]](
            state[0, obs_off + 9]
        )

        self.cached_state.wheel_omega_rl = rebind[Scalar[Self.dtype]](
            state[0, obs_off + 10]
        )

        self.cached_state.wheel_omega_rr = rebind[Scalar[Self.dtype]](
            state[0, obs_off + 11]
        )

        self.cached_state.speed = rebind[Scalar[Self.dtype]](
            state[0, obs_off + 12]
        )

    # =========================================================================
    # GPU Batch Operations (Static Methods) - GPUContinuousEnv Trait
    # =========================================================================

    @staticmethod
    def step_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        OBS_DIM: Int,
        ACTION_DIM: Int,
    ](
        ctx: DeviceContext,
        mut states: DeviceBuffer[dtype],
        actions: DeviceBuffer[dtype],
        mut rewards: DeviceBuffer[dtype],
        mut dones: DeviceBuffer[dtype],
        mut terminated: DeviceBuffer[dtype],
        mut obs: DeviceBuffer[dtype],
        rng_seed: UInt64 = 0,
        curriculum_values: List[Scalar[dtype]] = [],
        workspace_ptr: Optional[
            Pointer[Scalar[dtype], MutAnyOrigin]
        ] = None,
        rng_counter_ptr: Optional[
            Pointer[Scalar[DType.uint64], MutAnyOrigin]
        ] = None,
    ) raises:
        """Perform one environment step with embedded track (GPUContinuousEnv trait).

        Uses track tiles and visited flags embedded in the state buffer at
        TRACK_OFFSET and VISITED_OFFSET respectively. This is the unified
        step kernel that provides full track functionality without separate
        track/visited buffers.

        The embedded layout allows each environment to have its own random
        track generated during reset.

        Args:
            ctx: GPU device context.
            states: State buffer [BATCH_SIZE * STATE_SIZE] with embedded track.
            actions: Continuous actions buffer [BATCH_SIZE * ACTION_DIM].
            rewards: Rewards buffer (output) [BATCH_SIZE].
            dones: Done flags buffer (output) [BATCH_SIZE]. 1.0 if terminated OR truncated.
            terminated: Terminated flags buffer (output) [BATCH_SIZE]. 1.0 only if truly terminated.
            obs: Observations buffer (output) [BATCH_SIZE * OBS_DIM].
            rng_seed: Optional random seed (unused, track already embedded).
            curriculum_values: Optional curriculum values (unused).
            workspace_ptr: Optional workspace pointer (unused).
            rng_counter_ptr: Optional GPU counter pointer for deterministic RNG sequencing.
        """
        var states_tensor = tile[BATCH_SIZE, STATE_SIZE](states)

        var actions_tensor = tile[BATCH_SIZE, ACTION_DIM](actions)

        var rewards_tensor = tile[BATCH_SIZE](rewards)

        var dones_tensor = tile[BATCH_SIZE](dones)

        var terminated_tensor = tile[BATCH_SIZE](terminated)

        var obs_tensor = tile[BATCH_SIZE, OBS_DIM](obs)

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        @always_inline
        def step_embedded_wrapper(
            states: Tile[dtype, BATCH_SIZE, STATE_SIZE],
            actions: ImmTile[dtype, BATCH_SIZE, ACTION_DIM],
            rewards: Tile[dtype, BATCH_SIZE],
            dones: Tile[dtype, BATCH_SIZE],
            terminated_out: Tile[dtype, BATCH_SIZE],
            obs: Tile[dtype, BATCH_SIZE, OBS_DIM],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH_SIZE:
                return
            CarRacing[Self.dtype]._step_env_gpu_embedded[
                BATCH_SIZE, STATE_SIZE, OBS_DIM, ACTION_DIM
            ](env, states, actions, rewards, dones, obs)
            # CarRacing already stores META_TRUNCATED in state metadata
            # terminated = done AND NOT truncated
            var is_done = dones[env]
            var is_truncated = states[
                env, CRConstants.METADATA_OFFSET + META_TRUNCATED
            ]
            terminated_out[env] = is_done * (Scalar[dtype](1.0) - is_truncated)

        ctx.enqueue_function[step_embedded_wrapper](
            states_tensor,
            actions_tensor,
            rewards_tensor,
            dones_tensor,
            terminated_tensor,
            obs_tensor,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    @staticmethod
    def reset_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        ctx: DeviceContext,
        mut states: DeviceBuffer[dtype],
        rng_seed: UInt64 = 0,
    ) raises:
        """Reset all environments with embedded random tracks (GPUContinuousEnv trait).

        Generates a new random track for each environment, embeds track tiles
        in the state buffer at TRACK_OFFSET, clears visited flags at VISITED_OFFSET,
        and positions the car at the track start.

        Args:
            ctx: GPU device context.
            states: State buffer [BATCH_SIZE * STATE_SIZE].
            rng_seed: Random seed for initial state generation. Use different
                     values across calls for varied initial states.
        """
        var states_tensor = tile[BATCH_SIZE, STATE_SIZE](states)

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        @always_inline
        def reset_wrapper(
            states: Tile[dtype, BATCH_SIZE, STATE_SIZE],
            seed: Scalar[dtype],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH_SIZE:
                return
            # Combine seed with env index using prime multiplier for good distribution
            var combined_seed = Int(seed) * 2654435761 + env * 12345
            CarRacing[Self.dtype]._reset_env_gpu[BATCH_SIZE, STATE_SIZE](
                states, env, combined_seed
            )

        ctx.enqueue_function[reset_wrapper](
            states_tensor,
            Scalar[dtype](rng_seed),
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    @staticmethod
    def selective_reset_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        ctx: DeviceContext,
        mut states: DeviceBuffer[dtype],
        mut dones: DeviceBuffer[dtype],
        rng_seed: UInt64,
        workspace_ptr: Optional[
            Pointer[Scalar[dtype], MutAnyOrigin]
        ] = None,
        rng_counter_ptr: Optional[
            Pointer[Scalar[DType.uint64], MutAnyOrigin]
        ] = None,
    ) raises:
        """Reset only done environments with new random tracks (GPUContinuousEnv trait).

        For each done environment, generates a new random track embedded in the
        state buffer and resets the car position and visited flags.

        Args:
            ctx: GPU device context.
            states: State buffer [BATCH_SIZE * STATE_SIZE] with embedded track layout.
            dones: Done flags buffer [BATCH_SIZE].
            rng_seed: Random seed for initialization. Should be different each call
                     (e.g., training step counter) for varied tracks.
            workspace_ptr: Optional workspace pointer (unused).
            rng_counter_ptr: Optional GPU counter pointer. When non-null, reads
                     seed from GPU memory instead of rng_seed parameter.
        """
        var states_tensor = tile[BATCH_SIZE, STATE_SIZE](states)

        var dones_tensor = tile[BATCH_SIZE](dones)

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        if Bool(rng_counter_ptr):
            var counter_t = tile[1](rng_counter_ptr.value())

            @always_inline
            def selective_reset_counter_wrapper(
                states: Tile[dtype, BATCH_SIZE, STATE_SIZE],
                dones: Tile[dtype, BATCH_SIZE],
                counter: Tile[DType.uint64, 1],
            ):
                var env = Int(block_dim.x * block_idx.x + thread_idx.x)
                if env >= BATCH_SIZE:
                    return
                if rebind[Scalar[dtype]](dones[env]) > Scalar[dtype](0.5):
                    var combined_seed = (
                        Int(rebind[Scalar[DType.uint64]](counter[0]))
                        * 2654435761
                        + env * 12345
                    )
                    CarRacing[Self.dtype]._reset_env_gpu[
                        BATCH_SIZE, STATE_SIZE
                    ](states, env, combined_seed)
                    dones[env] = Scalar[dtype](0.0)

            ctx.enqueue_function[selective_reset_counter_wrapper](
                states_tensor,
                dones_tensor,
                counter_t,
                grid_dim=(BLOCKS,),
                block_dim=(TPB,),
            )
        else:

            @always_inline
            def selective_reset_wrapper(
                states: Tile[dtype, BATCH_SIZE, STATE_SIZE],
                dones: Tile[dtype, BATCH_SIZE],
                seed: Scalar[dtype],
            ):
                var env = Int(block_dim.x * block_idx.x + thread_idx.x)
                if env >= BATCH_SIZE:
                    return
                if rebind[Scalar[dtype]](dones[env]) > Scalar[dtype](0.5):
                    var combined_seed = Int(seed) * 2654435761 + env * 12345
                    CarRacing[Self.dtype]._reset_env_gpu[
                        BATCH_SIZE, STATE_SIZE
                    ](states, env, combined_seed)
                    dones[env] = Scalar[dtype](0.0)

            ctx.enqueue_function[selective_reset_wrapper](
                states_tensor,
                dones_tensor,
                Scalar[dtype](rng_seed),
                grid_dim=(BLOCKS,),
                block_dim=(TPB,),
            )

    @staticmethod
    def extract_obs_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE_VAL: Int,
        OBS_DIM_VAL: Int,
    ](
        ctx: DeviceContext,
        states_buf: DeviceBuffer[dtype],
        mut obs_buf: DeviceBuffer[dtype],
    ) raises:
        """Extract observations from state buffer (trivial copy: obs = state[0:OBS_DIM]).
        """
        var states = tile[BATCH_SIZE, STATE_SIZE_VAL](states_buf)
        var obs = tile[BATCH_SIZE, OBS_DIM_VAL](obs_buf)

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        @always_inline
        def extract_obs(
            states: ImmTile[dtype, BATCH_SIZE, STATE_SIZE_VAL],
            obs: Tile[dtype, BATCH_SIZE, OBS_DIM_VAL],
        ):
            var i = Int(block_dim.x * block_idx.x + thread_idx.x)
            if i >= BATCH_SIZE:
                return
            for d in range(OBS_DIM_VAL):
                obs[i, d] = states[i, d]

        ctx.enqueue_function[extract_obs](
            states,
            obs,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    @staticmethod
    def init_step_workspace_gpu[
        BATCH_SIZE: Int,
    ](ctx: DeviceContext, mut workspace_buf: DeviceBuffer[dtype]) raises:
        """No-op: CarRacing doesn't need pre-allocated workspace."""
        pass

    @staticmethod
    def update_curriculum_gpu(
        ctx: DeviceContext,
        mut workspace_buf: DeviceBuffer[dtype],
        curriculum_values: List[Scalar[dtype]],
    ) raises:
        """No-op: CarRacing doesn't use curriculum."""
        pass

    # =========================================================================
    # GPU Reset Helper (Embedded Track)
    # =========================================================================

    @always_inline
    @staticmethod
    def _reset_env_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        states: Tile[dtype, BATCH_SIZE, STATE_SIZE],
        env: Int,
        seed: Int,
    ):
        """Reset a single environment to initial state with embedded random track.

        Generates a random procedural track directly into the state buffer
        at TRACK_OFFSET, initializes visited flags to 0, and positions the
        car at the track start.

        Args:
            states: State buffer with embedded track layout.
            env: Environment index.
            seed: Random seed for track generation.
        """
        # Initialize RNG for this environment
        var rng = PhiloxRandom(seed=UInt64(seed), offset=0)

        # Clear entire state (including track and visited flags)
        for i in range(STATE_SIZE):
            states[env, i] = Scalar[dtype](0.0)

        # Generate random track and get number of tiles
        var num_tiles = CarRacing[Self.dtype]._generate_random_track_gpu[
            BATCH_SIZE, STATE_SIZE
        ](states, env, rng)

        # Store number of tiles in metadata
        states[env, CRConstants.METADATA_OFFSET + META_NUM_TILES] = Scalar[
            dtype
        ](num_tiles)

        # Get start position from first tile center
        var tile0_off = CRConstants.TRACK_OFFSET
        var tile0_v0x = rebind[Scalar[dtype]](states[env, tile0_off + 0])
        var tile0_v0y = rebind[Scalar[dtype]](states[env, tile0_off + 1])
        var tile0_v3x = rebind[Scalar[dtype]](states[env, tile0_off + 6])
        var tile0_v3y = rebind[Scalar[dtype]](states[env, tile0_off + 7])
        var tile0_v1x = rebind[Scalar[dtype]](states[env, tile0_off + 2])
        var tile0_v1y = rebind[Scalar[dtype]](states[env, tile0_off + 3])

        var start_x = (tile0_v0x + tile0_v3x) / Scalar[dtype](2.0)
        var start_y = (tile0_v0y + tile0_v3y) / Scalar[dtype](2.0)

        # Compute angle from tile direction (v0 to v1) using proper atan2 approximation
        var dx = tile0_v1x - tile0_v0x
        var dy = tile0_v1y - tile0_v0y
        var start_angle = Scalar[dtype](0.0)
        var dist_sq = dx * dx + dy * dy
        if dist_sq > Scalar[dtype](0.0001):
            var abs_dx = dx if dx >= Scalar[dtype](0.0) else -dx
            var abs_dy = dy if dy >= Scalar[dtype](0.0) else -dy

            if abs_dx > abs_dy:
                # |dy/dx| <= 1, use atan directly with polynomial approximation
                var t = dy / dx
                var t2 = t * t
                var t3 = t2 * t
                var t5 = t3 * t2
                var atan_val = (
                    t - t3 / Scalar[dtype](3.0) + t5 / Scalar[dtype](5.0)
                )
                # Clamp to valid range for numerical stability
                if atan_val > Scalar[dtype](0.8):
                    atan_val = Scalar[dtype](0.8)
                elif atan_val < Scalar[dtype](-0.8):
                    atan_val = Scalar[dtype](-0.8)
                start_angle = atan_val
                if dx < Scalar[dtype](0.0):
                    if dy >= Scalar[dtype](0.0):
                        start_angle = start_angle + Scalar[dtype](pi)
                    else:
                        start_angle = start_angle - Scalar[dtype](pi)
            else:
                # |dx/dy| < 1, use atan(1/x) = pi/2 - atan(x)
                var t = dx / dy
                var t2 = t * t
                var t3 = t2 * t
                var t5 = t3 * t2
                var atan_val = (
                    t - t3 / Scalar[dtype](3.0) + t5 / Scalar[dtype](5.0)
                )
                # Clamp to valid range
                if atan_val > Scalar[dtype](0.8):
                    atan_val = Scalar[dtype](0.8)
                elif atan_val < Scalar[dtype](-0.8):
                    atan_val = Scalar[dtype](-0.8)
                start_angle = Scalar[dtype](pi / 2.0) - atan_val
                if dy < Scalar[dtype](0.0):
                    start_angle = -start_angle

        # Set hull state
        states[env, CRConstants.HULL_OFFSET + HULL_X] = start_x
        states[env, CRConstants.HULL_OFFSET + HULL_Y] = start_y
        states[env, CRConstants.HULL_OFFSET + HULL_ANGLE] = start_angle
        states[env, CRConstants.HULL_OFFSET + HULL_VX] = Scalar[dtype](0.0)
        states[env, CRConstants.HULL_OFFSET + HULL_VY] = Scalar[dtype](0.0)
        states[env, CRConstants.HULL_OFFSET + HULL_OMEGA] = Scalar[dtype](0.0)

        # Initialize wheel states (all zeros - angles and omegas)
        for wheel in range(NUM_WHEELS):
            var wheel_off = CRConstants.WHEELS_OFFSET + wheel * WHEEL_STATE_SIZE
            states[env, wheel_off + WHEEL_OMEGA] = Scalar[dtype](0.0)
            states[env, wheel_off + WHEEL_JOINT_ANGLE] = Scalar[dtype](0.0)
            states[env, wheel_off + WHEEL_PHASE] = Scalar[dtype](0.0)

        # Initialize controls to zero
        states[env, CRConstants.CONTROLS_OFFSET + CTRL_STEERING] = Scalar[
            dtype
        ](0.0)
        states[env, CRConstants.CONTROLS_OFFSET + CTRL_GAS] = Scalar[dtype](0.0)
        states[env, CRConstants.CONTROLS_OFFSET + CTRL_BRAKE] = Scalar[dtype](
            0.0
        )

        # Initialize remaining metadata
        states[env, CRConstants.METADATA_OFFSET + META_STEP_COUNT] = Scalar[
            dtype
        ](0.0)
        states[env, CRConstants.METADATA_OFFSET + META_TOTAL_REWARD] = Scalar[
            dtype
        ](0.0)
        states[env, CRConstants.METADATA_OFFSET + META_DONE] = Scalar[dtype](
            0.0
        )
        states[env, CRConstants.METADATA_OFFSET + META_TRUNCATED] = Scalar[
            dtype
        ](0.0)
        states[env, CRConstants.METADATA_OFFSET + META_TILES_VISITED] = Scalar[
            dtype
        ](0.0)

        # Initialize observation
        states[env, CRConstants.OBS_OFFSET + 0] = start_x
        states[env, CRConstants.OBS_OFFSET + 1] = start_y
        states[env, CRConstants.OBS_OFFSET + 2] = start_angle
        for i in range(3, CRConstants.OBS_DIM):
            states[env, CRConstants.OBS_OFFSET + i] = Scalar[dtype](0.0)

    @always_inline
    @staticmethod
    def _generate_random_track_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        states: Tile[dtype, BATCH_SIZE, STATE_SIZE],
        env: Int,
        mut rng: PhiloxRandom,
    ) -> Int:
        """Generate random procedural track directly into state buffer.

        Generates a circular track with random checkpoints similar to
        Gymnasium's algorithm, but simplified for GPU compatibility.

        Args:
            states: State buffer.
            env: Environment index.
            rng: Random number generator (modified).

        Returns:
            Number of tiles generated.
        """
        var track_rad = Scalar[dtype](CRConstants.TRACK_RAD)
        var track_width = Scalar[dtype](CRConstants.TRACK_WIDTH)
        var detail_step = Scalar[dtype](CRConstants.TRACK_DETAIL_STEP)
        var turn_rate = Scalar[dtype](CRConstants.TRACK_TURN_RATE)
        var two_pi = Scalar[dtype](2.0 * pi)
        var num_checkpoints = 12

        # Generate random checkpoints around a circle
        var checkpoints_x = Array[Scalar[dtype], 12](
            fill=Scalar[dtype](0)
        )
        var checkpoints_y = Array[Scalar[dtype], 12](
            fill=Scalar[dtype](0)
        )

        for c in range(num_checkpoints):
            var rand_vals = rng.step_uniform()
            var noise = (
                (rand_vals[0] - Scalar[dtype](0.5))
                * two_pi
                / Scalar[dtype](num_checkpoints)
                * Scalar[dtype](0.5)
            )
            var alpha = (
                two_pi * Scalar[dtype](c) / Scalar[dtype](num_checkpoints)
                + noise
            )
            var rad = track_rad * (
                Scalar[dtype](0.5) + rand_vals[1] * Scalar[dtype](0.5)
            )

            # First checkpoint fixed for consistent start
            if c == 0:
                alpha = Scalar[dtype](0.0)
                rad = track_rad

            checkpoints_x[c] = rad * cos(alpha)
            checkpoints_y[c] = rad * sin(alpha)

        # Trace path from checkpoint to checkpoint
        var x = checkpoints_x[0]
        var y = checkpoints_y[0]
        var beta = Scalar[dtype](pi / 2.0)  # Start facing up (tangent)
        var num_tiles = 0
        var dest_i = 1

        var prev_x = x
        var prev_y = y

        for _ in range(500):  # Max iterations
            if num_tiles >= CRConstants.MAX_TRACK_TILES - 1:
                break

            # Get destination checkpoint
            var dest_x = checkpoints_x[dest_i % num_checkpoints]
            var dest_y = checkpoints_y[dest_i % num_checkpoints]

            # Steer towards destination
            var dx = dest_x - x
            var dy = dest_y - y
            var dist = sqrt(dx * dx + dy * dy)

            # Simple steering towards destination
            var angle_to_dest = Scalar[dtype](0.0)
            if dist > Scalar[dtype](0.001):
                # Proper atan2 approximation using polynomial approximation of atan
                # atan(x) ≈ x - x³/3 + x⁵/5 for |x| <= 1 (Taylor series)
                # For |x| > 1, use atan(x) = pi/2 - atan(1/x)
                var abs_dx = dx if dx >= Scalar[dtype](0.0) else -dx
                var abs_dy = dy if dy >= Scalar[dtype](0.0) else -dy

                if abs_dx > abs_dy:
                    # |dy/dx| <= 1, use atan directly
                    var t = dy / dx
                    var t2 = t * t
                    var t3 = t2 * t
                    var t5 = t3 * t2
                    var atan_val = (
                        t - t3 / Scalar[dtype](3.0) + t5 / Scalar[dtype](5.0)
                    )
                    # Clamp to valid range
                    if atan_val > Scalar[dtype](0.8):
                        atan_val = Scalar[dtype](0.8)
                    elif atan_val < Scalar[dtype](-0.8):
                        atan_val = Scalar[dtype](-0.8)
                    angle_to_dest = atan_val
                    if dx < Scalar[dtype](0.0):
                        if dy >= Scalar[dtype](0.0):
                            angle_to_dest = angle_to_dest + Scalar[dtype](pi)
                        else:
                            angle_to_dest = angle_to_dest - Scalar[dtype](pi)
                else:
                    # |dx/dy| < 1, use atan(1/x) = pi/2 - atan(x)
                    var t = dx / dy
                    var t2 = t * t
                    var t3 = t2 * t
                    var t5 = t3 * t2
                    var atan_val = (
                        t - t3 / Scalar[dtype](3.0) + t5 / Scalar[dtype](5.0)
                    )
                    # Clamp to valid range
                    if atan_val > Scalar[dtype](0.8):
                        atan_val = Scalar[dtype](0.8)
                    elif atan_val < Scalar[dtype](-0.8):
                        atan_val = Scalar[dtype](-0.8)
                    angle_to_dest = Scalar[dtype](pi / 2.0) - atan_val
                    if dy < Scalar[dtype](0.0):
                        angle_to_dest = -angle_to_dest

            var angle_diff = angle_to_dest - beta
            # Normalize to [-pi, pi]
            while angle_diff > Scalar[dtype](pi):
                angle_diff = angle_diff - two_pi
            while angle_diff < Scalar[dtype](-pi):
                angle_diff = angle_diff + two_pi

            # Apply steering
            var steer = turn_rate
            if angle_diff < Scalar[dtype](0.0):
                steer = -turn_rate
            if angle_diff > Scalar[dtype](-0.1) and angle_diff < Scalar[dtype](
                0.1
            ):
                steer = angle_diff

            beta = beta + steer
            x = x + detail_step * cos(beta)
            y = y + detail_step * sin(beta)

            # Check if reached checkpoint
            if dist < track_rad * Scalar[dtype](0.3):
                dest_i += 1

            # Create tile from previous to current position
            var perp_x = -sin(beta) * track_width / Scalar[dtype](2.0)
            var perp_y = cos(beta) * track_width / Scalar[dtype](2.0)

            var tile_off = (
                CRConstants.TRACK_OFFSET
                + num_tiles * CRConstants.TILE_DATA_SIZE
            )
            states[env, tile_off + 0] = prev_x - perp_x  # v0x (inner left)
            states[env, tile_off + 1] = prev_y - perp_y  # v0y
            states[env, tile_off + 2] = x - perp_x  # v1x (inner right)
            states[env, tile_off + 3] = y - perp_y  # v1y
            states[env, tile_off + 4] = x + perp_x  # v2x (outer right)
            states[env, tile_off + 5] = y + perp_y  # v2y
            states[env, tile_off + 6] = prev_x + perp_x  # v3x (outer left)
            states[env, tile_off + 7] = prev_y + perp_y  # v3y
            states[env, tile_off + 8] = Scalar[dtype](1.0)  # friction (road)
            num_tiles += 1

            prev_x = x
            prev_y = y

            # Check lap completion
            if dest_i >= num_checkpoints + 1:
                break

        return num_tiles

    @staticmethod
    def _copy_obs_gpu[
        BATCH_SIZE: Int,
        OBS_DIM: Int,
    ](
        ctx: DeviceContext,
        states: DeviceBuffer[dtype],
        mut obs: DeviceBuffer[dtype],
    ) raises:
        """Copy observations from state buffer to output obs buffer."""
        var states_tensor = tile[BATCH_SIZE, CRConstants.STATE_SIZE](states)

        var obs_tensor = tile[BATCH_SIZE, OBS_DIM](obs)

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        @always_inline
        def copy_obs_wrapper(
            states: ImmTile[dtype, BATCH_SIZE, CRConstants.STATE_SIZE],
            obs: Tile[dtype, BATCH_SIZE, OBS_DIM],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH_SIZE:
                return
            for i in range(OBS_DIM):
                obs[env, i] = states[env, CRConstants.OBS_OFFSET + i]

        ctx.enqueue_function[copy_obs_wrapper](
            states_tensor,
            obs_tensor,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    # =========================================================================
    # GPU Step Function (Embedded Track)
    # =========================================================================

    @always_inline
    @staticmethod
    def _step_env_gpu_embedded[
        BATCH: Int,
        STATE_SIZE: Int,
        OBS_DIM: Int,
        ACTION_DIM: Int,
    ](
        env: Int,
        states: Tile[dtype, BATCH, STATE_SIZE],
        actions: ImmTile[dtype, BATCH, ACTION_DIM],
        rewards: Tile[dtype, BATCH],
        dones: Tile[dtype, BATCH],
        obs: Tile[dtype, BATCH, OBS_DIM],
    ):
        """Execute one step using embedded track and visited flags.

        Uses track tiles and visited flags stored in the state buffer
        (at TRACK_OFFSET and VISITED_OFFSET respectively) instead of
        separate buffers.
        """
        # Check if already done
        var was_done = rebind[Scalar[dtype]](
            states[env, CRConstants.METADATA_OFFSET + META_DONE]
        ) > Scalar[dtype](0.5)
        if was_done:
            rewards[env] = Scalar[dtype](0.0)
            dones[env] = Scalar[dtype](1.0)
            return

        # Get number of tiles from metadata
        var num_tiles = Int(
            rebind[Scalar[dtype]](
                states[env, CRConstants.METADATA_OFFSET + META_NUM_TILES]
            )
        )
        if num_tiles <= 0:
            num_tiles = 100  # Fallback

        # Step 1: Copy actions to controls (clamp and remap)
        var steering = rebind[Scalar[dtype]](actions[env, 0])
        var gas_raw = rebind[Scalar[dtype]](actions[env, 1])
        var brake_raw = rebind[Scalar[dtype]](actions[env, 2])

        # Clamp steering to [-1, 1]
        if steering > Scalar[dtype](1.0):
            steering = Scalar[dtype](1.0)
        elif steering < Scalar[dtype](-1.0):
            steering = Scalar[dtype](-1.0)

        # Remap gas and brake from [-1, 1] to [0, 1]
        var gas = (gas_raw + Scalar[dtype](1.0)) * Scalar[dtype](0.5)
        var brake = (brake_raw + Scalar[dtype](1.0)) * Scalar[dtype](0.5)
        if gas > Scalar[dtype](1.0):
            gas = Scalar[dtype](1.0)
        elif gas < Scalar[dtype](0.0):
            gas = Scalar[dtype](0.0)
        if brake > Scalar[dtype](1.0):
            brake = Scalar[dtype](1.0)
        elif brake < Scalar[dtype](0.0):
            brake = Scalar[dtype](0.0)

        states[env, CRConstants.CONTROLS_OFFSET + CTRL_STEERING] = steering
        states[env, CRConstants.CONTROLS_OFFSET + CTRL_GAS] = gas
        states[env, CRConstants.CONTROLS_OFFSET + CTRL_BRAKE] = brake

        # Step 2: Run physics with embedded track
        var dt = Scalar[dtype](CRConstants.DT)
        CarDynamics.step_with_embedded_track[
            BATCH,
            STATE_SIZE,
            CRConstants.OBS_OFFSET,
            OBS_DIM,
            CRConstants.HULL_OFFSET,
            CRConstants.WHEELS_OFFSET,
            CRConstants.CONTROLS_OFFSET,
            CRConstants.TRACK_OFFSET,
            CRConstants.MAX_TRACK_TILES,
        ](env, states, num_tiles, dt)

        # Step 3: Compute reward with embedded visited tracking
        var hull_x = rebind[Scalar[dtype]](
            states[env, CRConstants.HULL_OFFSET + HULL_X]
        )
        var hull_y = rebind[Scalar[dtype]](
            states[env, CRConstants.HULL_OFFSET + HULL_Y]
        )
        var hull_vx = rebind[Scalar[dtype]](
            states[env, CRConstants.HULL_OFFSET + HULL_VX]
        )
        var hull_vy = rebind[Scalar[dtype]](
            states[env, CRConstants.HULL_OFFSET + HULL_VY]
        )
        var hull_angle = rebind[Scalar[dtype]](
            states[env, CRConstants.HULL_OFFSET + HULL_ANGLE]
        )

        # Forward velocity = velocity component in heading direction
        var cos_angle = cos(hull_angle)
        var sin_angle = sin(hull_angle)
        var forward_vel = hull_vx * cos_angle + hull_vy * sin_angle

        # Base reward: forward velocity bonus plus small time penalty
        var step_reward = forward_vel * Scalar[dtype](0.5) - Scalar[dtype](0.01)

        # Check tile visits using embedded track and mark visited
        var visit_result = TileCollision.check_and_mark_visited_embedded[
            BATCH,
            STATE_SIZE,
            CRConstants.TRACK_OFFSET,
            CRConstants.VISITED_OFFSET,
            CRConstants.MAX_TRACK_TILES,
        ](env, hull_x, hull_y, states, num_tiles)

        var tile_idx = visit_result[0]
        var is_newly_visited = visit_result[1]

        # Get current visited count
        var tiles_visited = rebind[Scalar[dtype]](
            states[env, CRConstants.METADATA_OFFSET + META_TILES_VISITED]
        )

        # Reward for newly visited tile
        if is_newly_visited:
            # Guard against division by zero
            var tile_reward = Scalar[dtype](1.0)  # Default to 1.0 reward
            if num_tiles > 0:
                tile_reward = Scalar[dtype](100.0) / Scalar[dtype](num_tiles)
            step_reward = step_reward + tile_reward
            tiles_visited = tiles_visited + Scalar[dtype](1.0)
            states[
                env, CRConstants.METADATA_OFFSET + META_TILES_VISITED
            ] = tiles_visited

        # Step 4: Check termination
        var done = Scalar[dtype](0.0)
        var playfield = Scalar[dtype](CRConstants.PLAYFIELD)

        # Off playfield check
        var abs_x = hull_x if hull_x >= Scalar[dtype](0.0) else -hull_x
        var abs_y = hull_y if hull_y >= Scalar[dtype](0.0) else -hull_y
        if abs_x > playfield or abs_y > playfield:
            done = Scalar[dtype](1.0)
            step_reward = Scalar[dtype](-100.0)

        # Lap completion check
        var num_tiles_f = Scalar[dtype](
            max(num_tiles, 1)
        )  # Guard against division by zero
        var progress = tiles_visited / num_tiles_f
        if progress >= Scalar[dtype](CRConstants.LAP_COMPLETE_PERCENT):
            done = Scalar[dtype](1.0)

        # Max steps check
        var step_count = rebind[Scalar[dtype]](
            states[env, CRConstants.METADATA_OFFSET + META_STEP_COUNT]
        )
        step_count = step_count + Scalar[dtype](1.0)
        states[env, CRConstants.METADATA_OFFSET + META_STEP_COUNT] = step_count

        if step_count >= Scalar[dtype](CRConstants.MAX_STEPS):
            done = Scalar[dtype](1.0)
            states[env, CRConstants.METADATA_OFFSET + META_TRUNCATED] = Scalar[
                dtype
            ](1.0)

        # Update metadata
        var total_reward = rebind[Scalar[dtype]](
            states[env, CRConstants.METADATA_OFFSET + META_TOTAL_REWARD]
        )
        total_reward = total_reward + step_reward
        states[
            env, CRConstants.METADATA_OFFSET + META_TOTAL_REWARD
        ] = total_reward
        states[env, CRConstants.METADATA_OFFSET + META_DONE] = done

        # Step 5: Write outputs
        rewards[env] = step_reward
        dones[env] = done

        # Copy observation to output buffer
        for i in range(OBS_DIM):
            obs[env, i] = states[env, CRConstants.OBS_OFFSET + i]

    @staticmethod
    def generate_track_to_buffer(
        ctx: DeviceContext,
        mut tiles_buf: DeviceBuffer[dtype],
        seed: UInt64 = 42,
    ) raises -> Int:
        """Generate a random track and copy to GPU buffer.

        Args:
            ctx: GPU device context.
            tiles_buf: Track tiles buffer (at least MAX_TRACK_TILES * TILE_DATA_SIZE).
            seed: Random seed for track generation.

        Returns:
            Number of tiles in the generated track.
        """
        # Generate track on CPU
        var track = TrackGenerator[DType.float64]()
        track.generate_random_track(seed, verbose=False)

        # Create host buffer for copying
        var tiles_host = ctx.enqueue_create_host_buffer[dtype](
            MAX_TRACK_TILES * TILE_DATA_SIZE
        )
        ctx.synchronize()

        # Create tensor view for the host buffer
        var tiles_tensor = tile[MAX_TRACK_TILES, TILE_DATA_SIZE](tiles_host)

        # Copy track data to host buffer
        track.to_buffer[MAX_TRACK_TILES](tiles_tensor)

        # Copy to GPU
        ctx.enqueue_copy(tiles_buf, tiles_host)
        ctx.synchronize()

        return track.track_length


# =============================================================================
# Helper Functions
# =============================================================================


def clamp(x: Float64, low: Float64, high: Float64) -> Float64:
    """Clamp value to range."""
    if x < low:
        return low
    if x > high:
        return high
    return x
