"""Native Mojo implementation of CartPole environment with integrated SDL2 rendering.

Physics based on OpenAI Gym / Gymnasium CartPole-v1:
https://gymnasium.farama.org/environments/classic_control/cart_pole/

A pole is attached by an un-actuated joint to a cart, which moves along a
frictionless track. The pendulum is placed upright on the cart and the goal
is to balance the pole by applying forces in the left and right direction
on the cart.

Supports both CPU (instance methods) and GPU (static inline methods) usage:
- CPU: Use reset(), step(), render() for interactive training
- GPU: Use step_kernel(), reset_kernel() in fused GPU kernels

Rendering uses native SDL2 bindings (no Python/pygame dependency).
Requires SDL2 and SDL2_ttf: brew install sdl2 sdl2_ttf
"""

from std.math import cos, sin
from std.random import random_float64
from std.memory import alloc
from noeira.core import (
    State,
    Action,
    DiscreteEnv,
    TileCoding,
    BoxDiscreteActionEnv,
    PolynomialFeatures,
    GPUDiscreteEnv,
    RenderableEnv,
)
from noeira.render import (
    Renderer2D,
    SDL_Color,
    Vec2,
    Camera,
    # Colors
    cart_blue,
    pole_tan,
    axle_purple,
    black,
    white,
)
from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext, DeviceBuffer
from std.random.philox import Random as PhiloxRandom
from noeira.core.tiles import Tile, ImmTile, tile

# =============================================================================
# Physics Constants (shared by CPU and GPU)
# =============================================================================

comptime gpu_dtype = DType.float32

# Physics parameters (same as Gymnasium CartPole-v1)
comptime GRAVITY: Float64 = 9.8
comptime CART_MASS: Float64 = 1.0
comptime POLE_MASS: Float64 = 0.1
comptime TOTAL_MASS: Float64 = CART_MASS + POLE_MASS
comptime POLE_HALF_LENGTH: Float64 = 0.5  # Half the pole's length
comptime POLE_MASS_LENGTH: Float64 = POLE_MASS * POLE_HALF_LENGTH
comptime FORCE_MAG: Float64 = 10.0
comptime TAU: Float64 = 0.02  # Time step (seconds)

# Termination thresholds
comptime X_THRESHOLD: Float64 = 2.4
comptime THETA_THRESHOLD: Float64 = 0.2095  # ~12 degrees

# Initial state randomization range
comptime INIT_RANGE: Float64 = 0.05

# Max episode length
comptime MAX_STEPS: Int = 500


# =============================================================================
# Single-source physics — the ONE copy of the CartPole math.
#
# Called verbatim by the CPU tabular `step`, the CPU `step_raw` and the GPU
# `step_kernel`. These used to be three hand-maintained copies, and the copies
# are exactly where CPU/GPU divergence shipped (the CPU terminal-step reward
# bug). Any physics/termination change goes HERE, once.
# =============================================================================


@always_inline
def cartpole_euler_step[
    DTYPE: DType
](
    x: Scalar[DTYPE],
    x_dot: Scalar[DTYPE],
    theta: Scalar[DTYPE],
    theta_dot: Scalar[DTYPE],
    force: Scalar[DTYPE],
) -> Tuple[Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE]]:
    """One Gymnasium CartPole-v1 Euler step: returns the post-step
    (x, x_dot, theta, theta_dot)."""
    comptime assert DTYPE.is_floating_point(), "DTYPE must be floating point"
    var costheta = cos(theta)
    var sintheta = sin(theta)

    # Equations of motion (derived from Lagrangian mechanics)
    var temp = (
        force + Scalar[DTYPE](POLE_MASS_LENGTH) * theta_dot * theta_dot * sintheta
    ) / Scalar[DTYPE](TOTAL_MASS)

    var thetaacc = (Scalar[DTYPE](GRAVITY) * sintheta - costheta * temp) / (
        Scalar[DTYPE](POLE_HALF_LENGTH)
        * (
            Scalar[DTYPE](4.0 / 3.0)
            - Scalar[DTYPE](POLE_MASS)
            * costheta
            * costheta
            / Scalar[DTYPE](TOTAL_MASS)
        )
    )

    var xacc = temp - Scalar[DTYPE](
        POLE_MASS_LENGTH
    ) * thetaacc * costheta / Scalar[DTYPE](TOTAL_MASS)

    # Euler integration (semi-implicit ordering matches Gymnasium exactly)
    return (
        x + Scalar[DTYPE](TAU) * x_dot,
        x_dot + Scalar[DTYPE](TAU) * xacc,
        theta + Scalar[DTYPE](TAU) * theta_dot,
        theta_dot + Scalar[DTYPE](TAU) * thetaacc,
    )


@always_inline
def cartpole_force[DTYPE: DType](action_idx: Int) -> Scalar[DTYPE]:
    """Action decode: 1 → +FORCE_MAG (push right), else −FORCE_MAG."""
    return Scalar[DTYPE](FORCE_MAG) if action_idx == 1 else Scalar[DTYPE](
        -FORCE_MAG
    )


@always_inline
def cartpole_terminated[
    DTYPE: DType
](x: Scalar[DTYPE], theta: Scalar[DTYPE]) -> Bool:
    """Natural termination: cart or pole out of bounds."""
    return (
        x < Scalar[DTYPE](-X_THRESHOLD)
        or x > Scalar[DTYPE](X_THRESHOLD)
        or theta < Scalar[DTYPE](-THETA_THRESHOLD)
        or theta > Scalar[DTYPE](THETA_THRESHOLD)
    )


# ============================================================================
# CartPole State and Action types for trait conformance
# ============================================================================


@fieldwise_init
struct CartPoleState(Copyable, ImplicitlyCopyable, Movable, State):
    """State for CartPole: discretized state index.

    The continuous observation [x, x_dot, theta, theta_dot] is discretized
    into bins to create a single integer state index for tabular methods.
    """

    var index: Int

    def __init__(out self, *, copy: Self):
        self.index = copy.index

    def __init__(out self, *, deinit move: Self):
        self.index = move.index

    def __eq__(self, other: Self) -> Bool:
        return self.index == other.index


@fieldwise_init
struct CartPoleAction(Action, Copyable, ImplicitlyCopyable, Movable):
    """Action for CartPole: 0 (push left), 1 (push right)."""

    var direction: Int

    def __init__(out self, *, copy: Self):
        self.direction = copy.direction

    def __init__(out self, *, deinit move: Self):
        self.direction = move.direction

    @staticmethod
    def left() -> Self:
        return Self(direction=0)

    @staticmethod
    def right() -> Self:
        return Self(direction=1)


struct CartPoleEnv[DTYPE: DType](
    BoxDiscreteActionEnv & DiscreteEnv & GPUDiscreteEnv & RenderableEnv
):
    """Native Mojo CartPole environment with integrated SDL2 rendering.

    State: [cart_position, cart_velocity, pole_angle, pole_angular_velocity] (4D).
    Actions: 0 (push left), 1 (push right).

    Episode terminates when:
    - Pole angle > ±12° (±0.2095 rad).
    - Cart position > ±2.4.
    - Episode length > 500 steps.

    Implements:
    - DiscreteEnv: for tabular methods
    - BoxDiscreteActionEnv: for function approximation with continuous observations
    - GPUDiscreteEnv: for fused GPU kernels (A2C, PPO, etc.)

    Usage:
        env = CartPoleEnv[DType.float64]()
        obs = env.reset()
        obs, reward, done = env.step(action)
    """

    # Type aliases for CPU trait conformance
    comptime dtype = Self.DTYPE
    comptime StateType = CartPoleState
    comptime ActionType = CartPoleAction

    # GPUDiscreteEnv trait constants
    comptime STATE_SIZE: Int = 5  # [x, x_dot, theta, theta_dot, step_count]
    comptime OBS_DIM: Int = 4  # Same as state for CartPole
    comptime NUM_ACTIONS: Int = 2  # Left (0) or Right (1)
    comptime STEP_WS_SHARED: Int = 0
    comptime STEP_WS_PER_ENV: Int = 0

    # Current state
    var x: Scalar[Self.dtype]  # Cart position
    var x_dot: Scalar[Self.dtype]  # Cart velocity
    var theta: Scalar[Self.dtype]  # Pole angle (radians, 0 = upright)
    var theta_dot: Scalar[Self.dtype]  # Pole angular velocity

    # Episode tracking
    var steps: Int
    var done: Bool
    # Natural-termination flag (pole-fall / out-of-bounds), NOT time-limit
    # truncation. Read by off-policy drivers via `was_terminated()`.
    var _last_terminated: Bool
    var total_reward: Scalar[Self.dtype]

    # Discretization settings (for DiscreteEnv)
    var num_bins: Int

    # Renderer (RenderableEnv)
    var _renderer: Optional[Pointer[Renderer2D, MutUntrackedOrigin]]
    var _renderer_initialized: Bool

    def __init__(out self, num_bins: Int = 10):
        """Initialize CartPole environment."""
        # State
        self.x = 0.0
        self.x_dot = 0.0
        self.theta = 0.0
        self.theta_dot = 0.0

        # Episode
        self.steps = 0
        self.done = False
        self._last_terminated = False
        self.total_reward = 0.0

        # Discretization settings
        self.num_bins = num_bins

        # Renderer
        self._renderer = None
        self._renderer_initialized = False

    # ========================================================================
    # DiscreteEnv trait methods
    # ========================================================================

    def reset(mut self) -> CartPoleState:
        """Reset environment to random initial state.

        Returns CartPoleState with discretized state index.
        """
        # Random initial state in [-0.05, 0.05] for each component
        self.x = Scalar[Self.dtype]((random_float64() - 0.5) * 0.1)
        self.x_dot = Scalar[Self.dtype]((random_float64() - 0.5) * 0.1)
        self.theta = Scalar[Self.dtype]((random_float64() - 0.5) * 0.1)
        self.theta_dot = Scalar[Self.dtype]((random_float64() - 0.5) * 0.1)

        self.steps = 0
        self.done = False
        self.total_reward = 0.0

        return CartPoleState(index=self._discretize_obs())

    def step(
        mut self, action: CartPoleAction, verbose: Bool = False
    ) -> Tuple[CartPoleState, Scalar[Self.dtype], Bool]:
        """Take action and return (state, reward, done).

        Args:
            action: CartPoleAction (direction 0=left, 1=right).
            verbose: Whether to print verbose output (default: False).

        Physics uses Euler integration (same as Gymnasium).
        """
        # Single-source physics (shared with step_raw + the GPU kernel).
        var force = cartpole_force[Self.dtype](action.direction)
        var nxt = cartpole_euler_step[Self.dtype](
            self.x, self.x_dot, self.theta, self.theta_dot, force
        )
        self.x = nxt[0]
        self.x_dot = nxt[1]
        self.theta = nxt[2]
        self.theta_dot = nxt[3]

        self.steps += 1

        var terminated = cartpole_terminated[Self.dtype](self.x, self.theta)
        var truncated = self.steps >= MAX_STEPS

        self.done = terminated or truncated
        self._last_terminated = terminated

        # Reward: +1 for every step taken, INCLUDING the terminating step —
        # Gymnasium gives +1 when the pole "just fell" (0 only for steps
        # after termination, which never happen here since the driver
        # resets); the GPU kernel matches this.
        var reward: Scalar[Self.dtype] = Scalar[Self.dtype](1.0)
        self.total_reward += reward

        return (CartPoleState(index=self._discretize_obs()), reward, self.done)

    def get_state(mut self) -> CartPoleState:
        """Return current discretized state."""
        return CartPoleState(index=self._discretize_obs())

    def state_to_index(self, state: CartPoleState) -> Int:
        """Convert a CartPoleState to an index for tabular methods."""
        return state.index

    def action_from_index(self, action_idx: Int) -> CartPoleAction:
        """Create a CartPoleAction from an index."""
        return CartPoleAction(direction=action_idx)

    # ========================================================================
    # Internal helpers
    # ========================================================================

    @always_inline
    def _get_obs(self) -> SIMD[DType.float64, 4]:
        """Return current continuous observation."""
        var obs = SIMD[DType.float64, 4]()
        obs[0] = Float64(self.x)
        obs[1] = Float64(self.x_dot)
        obs[2] = Float64(self.theta)
        obs[3] = Float64(self.theta_dot)
        return obs

    @always_inline
    def _discretize_obs(self) -> Int:
        """Discretize current continuous observation into a single state index.
        """
        # Inline bin calculation for each dimension
        # Cart position: [-2.4, 2.4]
        var n0 = (Float64(self.x) + 2.4) / 4.8
        if n0 < 0.0:
            n0 = 0.0
        elif n0 > 1.0:
            n0 = 1.0
        var b0 = Int(n0 * Float64(self.num_bins - 1))

        # Cart velocity: [-3.0, 3.0]
        var n1 = (Float64(self.x_dot) + 3.0) / 6.0
        if n1 < 0.0:
            n1 = 0.0
        elif n1 > 1.0:
            n1 = 1.0
        var b1 = Int(n1 * Float64(self.num_bins - 1))

        # Pole angle: [-0.21, 0.21]
        var n2 = (Float64(self.theta) + 0.21) / 0.42
        if n2 < 0.0:
            n2 = 0.0
        elif n2 > 1.0:
            n2 = 1.0
        var b2 = Int(n2 * Float64(self.num_bins - 1))

        # Pole angular velocity: [-3.0, 3.0]
        var n3 = (Float64(self.theta_dot) + 3.0) / 6.0
        if n3 < 0.0:
            n3 = 0.0
        elif n3 > 1.0:
            n3 = 1.0
        var b3 = Int(n3 * Float64(self.num_bins - 1))

        return (
            (b0 * self.num_bins + b1) * self.num_bins + b2
        ) * self.num_bins + b3

    @always_inline
    def get_obs(self) -> SIMD[DType.float64, 4]:
        """Return current continuous observation as SIMD (optimized)."""
        return self._get_obs()

    # ========================================================================
    # ContinuousStateEnv / BoxDiscreteActionEnv trait methods
    # ========================================================================

    def get_obs_list(self) -> List[Scalar[Self.dtype]]:
        """Return current continuous observation as a flexible list (trait method).
        """
        var obs = List[Scalar[Self.dtype]](capacity=4)
        obs.append(self.x)
        obs.append(self.x_dot)
        obs.append(self.theta)
        obs.append(self.theta_dot)
        return obs^

    def reset_obs_list(mut self) -> List[Scalar[Self.dtype]]:
        """Reset environment and return initial observation as list (trait method).
        """
        _ = self.reset()
        return self.get_obs_list()

    def step_obs(
        mut self, action: Int
    ) -> Tuple[List[Scalar[Self.dtype]], Scalar[Self.dtype], Bool]:
        """Take action and return (obs_list, reward, done) - trait method.

        This is the BoxDiscreteActionEnv trait method using List[Scalar].
        For performance-critical code, use step_raw() which returns SIMD.
        """
        var result = self.step_raw(action)
        return (self.get_obs_list(), result[1], result[2])

    def was_terminated(self) -> Bool:
        """True iff the previous step ended via natural termination (pole
        fell / cart out of bounds), NOT the time-limit truncation at
        MAX_STEPS. Used by off-policy drivers to keep the TD bootstrap on
        truncation but drop it on termination."""
        return self._last_terminated

    # ========================================================================
    # SIMD-optimized observation API (for performance)
    # ========================================================================

    def reset_obs(mut self) -> SIMD[DType.float64, 4]:
        """Reset environment and return raw continuous observation.

        Use this for function approximation methods (tile coding, linear FA)
        that need the continuous observation vector.

        Returns:
            Continuous observation [x, x_dot, theta, theta_dot].
        """
        _ = self.reset()  # Reset internal state
        return self._get_obs()

    @always_inline
    def step_raw(
        mut self, action: Int
    ) -> Tuple[SIMD[DType.float64, 4], Scalar[Self.dtype], Bool]:
        """Take action and return raw continuous observation.

        Use this for function approximation methods that need the continuous
        observation vector rather than discretized state.

        Args:
            action: 0 for left force, 1 for right force.

        Returns:
            Tuple of (observation, reward, done).
        """

        comptime assert (
            Self.dtype.is_floating_point()
        ), "DTYPE must be a floating point type"

        # Single-source physics (shared with the tabular step + GPU kernel).
        var force = cartpole_force[Self.dtype](action)
        var nxt = cartpole_euler_step[Self.dtype](
            self.x, self.x_dot, self.theta, self.theta_dot, force
        )
        self.x = nxt[0]
        self.x_dot = nxt[1]
        self.theta = nxt[2]
        self.theta_dot = nxt[3]

        self.steps += 1

        var terminated = cartpole_terminated[Self.dtype](self.x, self.theta)
        var truncated = self.steps >= MAX_STEPS
        self.done = terminated or truncated
        self._last_terminated = terminated

        # +1 including the terminating step (Gymnasium semantics; matches
        # the GPU kernel — see the tabular step above).
        var reward: Scalar[Self.dtype] = Scalar[Self.dtype](1.0)
        self.total_reward += reward

        return (self._get_obs(), reward, self.done)

    def render(mut self, mut renderer: Renderer2D):
        """Render the current state using SDL2.

        Uses Camera for world-to-screen coordinate conversion.
        The renderer should be initialized before calling this method.

        Args:
            renderer: The Renderer2D to use for rendering.
        """
        # Begin frame handles init, events, and clear
        if not renderer.begin_frame():
            return

        # Constants for rendering
        var world_width = 4.8  # x_threshold * 2

        # Create camera centered on screen with Y-flip
        # zoom = pixels per world unit, centered at world origin
        var zoom = Float64(renderer.screen_width) / world_width
        var camera = renderer.make_camera_at(0.0, 0.5, zoom, True)

        # World coordinates: cart moves along X, ground is at Y=0
        # Cast to Float64 for rendering
        var x_f64 = Float64(self.x)
        var theta_f64 = Float64(self.theta)
        var cart_pos = Vec2(x_f64, 0.15)  # Cart center slightly above ground
        var cart_width_world = 0.4
        var cart_height_world = 0.24

        # Colors
        var track_color = black()
        var cart_color_ = cart_blue()
        var pole_color_ = pole_tan()
        var axle_color_ = axle_purple()
        var wheel_color = black()

        # Draw ground/track line at Y=0
        renderer.draw_ground_line(0.0, camera, track_color, 2)

        # Draw cart as rectangle
        renderer.draw_rect_world(
            cart_pos,
            cart_width_world,
            cart_height_world,
            camera,
            cart_color_,
            centered=True,
        )

        # Draw pole - starts at top of cart, rotates around pivot
        var pivot = Vec2(x_f64, cart_pos.y + cart_height_world / 2.0)
        var pole_length = 0.5  # Half the pole length in world units
        # theta=0 means upright (positive Y), positive theta rotates clockwise
        var pole_end = Vec2(
            pivot.x + pole_length * sin(theta_f64),
            pivot.y + pole_length * cos(theta_f64),
        )
        renderer.draw_link(pivot, pole_end, camera, pole_color_, 10)

        # Draw axle (pivot point)
        renderer.draw_joint(pivot, 0.04, camera, axle_color_)

        # Draw wheels
        var wheel_radius = 0.04
        var wheel_y = cart_pos.y - cart_height_world / 2.0
        var wheel_offset = cart_width_world / 2.0 - 0.08
        renderer.draw_circle_world(
            Vec2(x_f64 - wheel_offset, wheel_y),
            wheel_radius,
            camera,
            wheel_color,
            True,
        )
        renderer.draw_circle_world(
            Vec2(x_f64 + wheel_offset, wheel_y),
            wheel_radius,
            camera,
            wheel_color,
            True,
        )

        # Draw info text
        var info_lines = List[String]()
        info_lines.append("Step: " + String(self.steps))
        info_lines.append("Reward: " + String(Int(self.total_reward)))
        renderer.draw_info_box(info_lines)

        # Update display
        renderer.flip()

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
        if self._renderer_initialized:
            return True
        self._renderer = alloc[Renderer2D]({count = 1}).unsafe_leak()
        self._renderer.value().unsafe_write(Renderer2D())
        self._renderer_initialized = True
        return True

    def render_frame(mut self) raises -> None:
        if not self._renderer_initialized:
            return
        self.render(self._renderer.value()[])

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

    @always_inline
    def is_done(self) -> Bool:
        """Check if episode is done."""
        return self.done

    @always_inline
    def num_actions(self) -> Int:
        """Return number of actions (2)."""
        return 2

    @always_inline
    def obs_dim(self) -> Int:
        """Return observation dimension (4)."""
        return 4

    @always_inline
    def num_states(self) -> Int:
        """Return total number of discrete states."""
        return self.num_bins * self.num_bins * self.num_bins * self.num_bins

    # ========================================================================
    # Static methods for discretization
    # ========================================================================

    @staticmethod
    def get_num_states(num_bins: Int = 10) -> Int:
        """Get the number of discrete states for CartPole with given bins."""
        return num_bins * num_bins * num_bins * num_bins

    @staticmethod
    def discretize_obs(obs: SIMD[DType.float64, 4], num_bins: Int = 10) -> Int:
        """Discretize continuous observation into a single state index.

        Args:
            obs: Continuous observation [x, x_dot, theta, theta_dot].
            num_bins: Number of bins per dimension.

        Returns:
            Single integer state index.
        """
        var cart_pos_low: Float64 = -2.4
        var cart_pos_high: Float64 = 2.4
        var cart_vel_low: Float64 = -3.0
        var cart_vel_high: Float64 = 3.0
        var pole_angle_low: Float64 = -0.21
        var pole_angle_high: Float64 = 0.21
        var pole_vel_low: Float64 = -3.0
        var pole_vel_high: Float64 = 3.0

        def bin_value(
            value: Float64, low: Float64, high: Float64, bins: Int
        ) -> Int:
            var normalized = (value - low) / (high - low)
            if normalized < 0.0:
                normalized = 0.0
            elif normalized > 1.0:
                normalized = 1.0
            return Int(normalized * Float64(bins - 1))

        var b0 = bin_value(obs[0], cart_pos_low, cart_pos_high, num_bins)
        var b1 = bin_value(obs[1], cart_vel_low, cart_vel_high, num_bins)
        var b2 = bin_value(obs[2], pole_angle_low, pole_angle_high, num_bins)
        var b3 = bin_value(obs[3], pole_vel_low, pole_vel_high, num_bins)

        return ((b0 * num_bins + b1) * num_bins + b2) * num_bins + b3

    @staticmethod
    def make_tile_coding(
        num_tilings: Int = 8,
        tiles_per_dim: Int = 8,
    ) -> TileCoding[Self.dtype]:
        """Create tile coding configured for CartPole environment.

        CartPole state: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]

        Args:
            num_tilings: Number of tilings (default 8).
            tiles_per_dim: Tiles per dimension (default 8).

        Returns:
            TileCoding configured for CartPole state space.
        """
        var tiles = List[Int]()
        tiles.append(tiles_per_dim)
        tiles.append(tiles_per_dim)
        tiles.append(tiles_per_dim)
        tiles.append(tiles_per_dim)

        # CartPole state bounds (slightly expanded for safety)
        var state_low = List[Scalar[Self.dtype]]()
        state_low.append(-2.5)  # cart position
        state_low.append(-3.5)  # cart velocity
        state_low.append(-0.25)  # pole angle (radians)
        state_low.append(-3.5)  # pole angular velocity

        var state_high = List[Scalar[Self.dtype]]()
        state_high.append(2.5)
        state_high.append(3.5)
        state_high.append(0.25)
        state_high.append(3.5)

        return TileCoding[Self.dtype](
            num_tilings=num_tilings,
            tiles_per_dim=tiles^,
            state_low=state_low^,
            state_high=state_high^,
        )

    @staticmethod
    def make_poly_features(degree: Int = 2) -> PolynomialFeatures[Self.dtype]:
        """Create polynomial features for CartPole (4D state) with normalization.

        CartPole state: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]

        Args:
            degree: Maximum polynomial degree (keep low for 4D to avoid explosion).

        Returns:
            PolynomialFeatures extractor configured for CartPole with normalization.
        """
        var state_low = List[Scalar[Self.dtype]]()
        state_low.append(-2.4)  # cart position
        state_low.append(-3.0)  # cart velocity
        state_low.append(-0.21)  # pole angle (radians)
        state_low.append(-3.0)  # pole angular velocity

        var state_high = List[Scalar[Self.dtype]]()
        state_high.append(2.4)
        state_high.append(3.0)
        state_high.append(0.21)
        state_high.append(3.0)

        return PolynomialFeatures[Self.dtype](
            state_dim=4,
            degree=degree,
            include_bias=True,
            state_low=state_low^,
            state_high=state_high^,
        )

    # ========================================================================
    # GPUDiscreteEnv trait methods (for fused GPU kernels)
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
        # Note: rng_seed is unused in CartPole (no random physics elements)
        # It's included for trait compatibility with GPUDiscreteEnv
        var i = Int(block_dim.x * block_idx.x + thread_idx.x)
        if i >= BATCH_SIZE:
            return

        # Single-source physics (shared with the CPU step/step_raw).
        # Cast to int to handle float actions correctly (0.0 -> 0, 1.0 -> 1)
        var force = cartpole_force[gpu_dtype](Int(actions[i]))
        var nxt = cartpole_euler_step[gpu_dtype](
            rebind[Scalar[gpu_dtype]](states[i, 0]),
            rebind[Scalar[gpu_dtype]](states[i, 1]),
            rebind[Scalar[gpu_dtype]](states[i, 2]),
            rebind[Scalar[gpu_dtype]](states[i, 3]),
            force,
        )
        states[i, 0] = nxt[0]
        states[i, 1] = nxt[1]
        states[i, 2] = nxt[2]
        states[i, 3] = nxt[3]

        # Increment step counter (stored at index 4)
        states[i, 4] += Scalar[gpu_dtype](1.0)

        var terminated = cartpole_terminated[gpu_dtype](nxt[0], nxt[2])

        # Check truncation (max steps reached)
        var truncated = states[i, 4] >= Scalar[gpu_dtype](MAX_STEPS)

        var done = terminated or truncated

        # Reward: +1 for every step taken (including termination step)
        # Matches Gymnasium CartPole-v1 default behavior
        var reward = Scalar[gpu_dtype](1.0)

        rewards[i] = reward
        # `done` is a Bool, and `Scalar[float](Bool)` no longer compiles —
        # SIMD's Intable constructor now requires an integral dtype. Spell the
        # 0/1 encoding out rather than casting through an integer.
        dones[i] = Scalar[gpu_dtype](1.0) if done else Scalar[gpu_dtype](0.0)

    @staticmethod
    @always_inline
    def reset_kernel[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        state: Tile[gpu_dtype, BATCH_SIZE, STATE_SIZE],
    ):
        """Reset state to random initial values using GPU-compatible xorshift RNG.

        Each thread gets a unique seed based on its index, ensuring different
        initial states across the batch while being fully GPU-compatible.
        """
        var i = Int(block_dim.x * block_idx.x + thread_idx.x)
        if i >= BATCH_SIZE:
            return

        var rng = PhiloxRandom(
            seed=UInt64(i) * UInt64(2654435761) + 12345, offset=0
        )
        var rand_vals = rng.step_uniform()

        # Map [0, 1) → [-0.05, 0.05]
        state[i, 0] = Scalar[gpu_dtype](rand_vals[0]) * Scalar[gpu_dtype](
            0.1
        ) - Scalar[gpu_dtype](0.05)
        state[i, 1] = Scalar[gpu_dtype](rand_vals[1]) * Scalar[gpu_dtype](
            0.1
        ) - Scalar[gpu_dtype](0.05)
        state[i, 2] = Scalar[gpu_dtype](rand_vals[2]) * Scalar[gpu_dtype](
            0.1
        ) - Scalar[gpu_dtype](0.05)
        state[i, 3] = Scalar[gpu_dtype](rand_vals[3]) * Scalar[gpu_dtype](
            0.1
        ) - Scalar[gpu_dtype](0.05)
        state[i, 4] = Scalar[gpu_dtype](0.0)  # Reset step counter

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
        """Reset state only for done environments using GPU-compatible xorshift RNG.

        This kernel checks dones[i] and only resets environments where done > 0.5.
        It also clears dones[i] = 0 after reset to prepare for next episode.

        Args:
            state: Environment states [BATCH_SIZE, STATE_SIZE].
            dones: Done flags [BATCH_SIZE]. Will be cleared for reset envs.
            rng_seed: Base seed for random number generation (varies per call).
        """
        var i = Int(block_dim.x * block_idx.x + thread_idx.x)
        if i >= BATCH_SIZE:
            return

        # Only reset done environments
        if dones[i] < Scalar[gpu_dtype](0.5):
            return

        var rng = PhiloxRandom(
            seed=UInt64(i) * UInt64(2654435761) + UInt64(rng_seed), offset=0
        )
        var rand_vals = rng.step_uniform()

        # Map [0, 1) → [-0.05, 0.05]
        state[i, 0] = Scalar[gpu_dtype](rand_vals[0]) * Scalar[gpu_dtype](
            0.1
        ) - Scalar[gpu_dtype](0.05)
        state[i, 1] = Scalar[gpu_dtype](rand_vals[1]) * Scalar[gpu_dtype](
            0.1
        ) - Scalar[gpu_dtype](0.05)
        state[i, 2] = Scalar[gpu_dtype](rand_vals[2]) * Scalar[gpu_dtype](
            0.1
        ) - Scalar[gpu_dtype](0.05)
        state[i, 3] = Scalar[gpu_dtype](rand_vals[3]) * Scalar[gpu_dtype](
            0.1
        ) - Scalar[gpu_dtype](0.05)
        state[i, 4] = Scalar[gpu_dtype](0.0)  # Reset step counter

        # Clear done flag for next episode
        dones[i] = Scalar[gpu_dtype](0.0)

    # ========================================================================
    # GPU Launcher Methods (host-side, call the kernels)
    # ========================================================================

    comptime TPB = 256  # Threads per block

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
        """Launch step kernel on GPU with fused obs extraction.

        Args:
            ctx: GPU device context.
            states_buf: States buffer [BATCH_SIZE * STATE_SIZE].
            actions_buf: Actions buffer [BATCH_SIZE].
            rewards_buf: Rewards buffer [BATCH_SIZE] (written).
            dones_buf: Dones buffer [BATCH_SIZE] (written). 1.0 if terminated OR truncated.
            terminated_buf: Terminated buffer [BATCH_SIZE] (written). 1.0 only if truly terminated.
            obs_buf: Observations buffer [BATCH_SIZE * OBS_DIM] (written).
            rng_seed: Random seed (unused in CartPole, for trait compatibility).
            workspace_ptr: Optional workspace pointer (unused for CartPole).
            rng_counter_ptr: Optional GPU counter pointer (unused for CartPole).
        """
        # Create tensor views from buffers
        var states = tile[BATCH_SIZE, STATE_SIZE](states_buf)
        var actions = tile[BATCH_SIZE](actions_buf)
        var rewards = tile[BATCH_SIZE](rewards_buf)
        var dones = tile[BATCH_SIZE](dones_buf)
        var terminated_out = tile[BATCH_SIZE](terminated_buf)
        var obs = tile[BATCH_SIZE, OBS_DIM](obs_buf)

        # Configure grid
        comptime BLOCKS = (BATCH_SIZE + Self.TPB - 1) // Self.TPB

        # Convert seed (unused in CartPole but needed for trait compatibility)
        var seed = Scalar[DType.uint64](rng_seed)

        # Define kernel wrapper that calls the impl and extracts obs
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
            # CartPole: terminated = physics bounds exceeded (not truncation)
            var i = Int(block_dim.x * block_idx.x + thread_idx.x)
            if i < BATCH_SIZE:
                # Check if terminated by physics (not truncation)
                var is_terminated = (
                    (states[i, 0] < Scalar[gpu_dtype](-X_THRESHOLD))
                    or (states[i, 0] > Scalar[gpu_dtype](X_THRESHOLD))
                    or (states[i, 2] < Scalar[gpu_dtype](-THETA_THRESHOLD))
                    or (states[i, 2] > Scalar[gpu_dtype](THETA_THRESHOLD))
                )
                terminated_out[i] = Scalar[gpu_dtype](
                    1.0
                ) if is_terminated else Scalar[gpu_dtype](0.0)
                # Extract observations (obs == first 4 state elements)
                for d in range(OBS_DIM):
                    obs[i, d] = states[i, d]

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
    def reset_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[gpu_dtype],
        rng_seed: UInt64 = 0,
    ) raises:
        """Launch reset kernel on GPU.

        Args:
            ctx: GPU device context.
            states_buf: States buffer [BATCH_SIZE * STATE_SIZE] (written).
            rng_seed: Random seed for initial state generation (unused for CartPole
                     but included for trait compatibility).
        """
        # Create tensor view from buffer
        var states = tile[BATCH_SIZE, STATE_SIZE](states_buf)

        # Configure grid
        comptime BLOCKS = (BATCH_SIZE + Self.TPB - 1) // Self.TPB

        # Define kernel wrapper
        # Note: MutAnyOrigin allows mutation, no `mut` keyword needed on wrapper params
        # CartPole doesn't have terrain - initial states are deterministic per env index
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
        """Launch selective reset kernel on GPU - only resets done environments.

        Args:
            ctx: GPU device context.
            states_buf: States buffer [BATCH_SIZE * STATE_SIZE] (written for done envs).
            dones_buf: Dones buffer [BATCH_SIZE] (read to check, cleared for done envs).
            rng_seed: Seed for random number generation (should vary between calls).
            workspace_ptr: Optional workspace pointer (unused for CartPole).
            rng_counter_ptr: Optional GPU counter pointer. When non-null, reads
                     seed from GPU memory instead of rng_seed parameter.
        """
        # Create tensor views from buffers
        var states = tile[BATCH_SIZE, STATE_SIZE](states_buf)
        var dones = tile[BATCH_SIZE](dones_buf)

        # Configure grid
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

            # Define kernel wrapper
            @always_inline
            def selective_reset_wrapper(
                states: Tile[gpu_dtype, BATCH_SIZE, STATE_SIZE],
                dones: Tile[gpu_dtype, BATCH_SIZE],
                rng_seed: Scalar[DType.uint64],
            ):
                # Cast to uint32 for the inner kernel (RNG uses 32-bit state)
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
        """No-op: CartPole doesn't need pre-allocated workspace."""
        pass

    @staticmethod
    def update_curriculum_gpu(
        ctx: DeviceContext,
        mut workspace_buf: DeviceBuffer[gpu_dtype],
        curriculum_values: List[Scalar[gpu_dtype]],
    ) raises:
        """No-op: CartPole doesn't use curriculum."""
        pass
