"""Native Mojo implementation of Acrobot environment with integrated SDL2 rendering.

Physics based on OpenAI Gym / Gymnasium Acrobot-v1:
https://gymnasium.farama.org/environments/classic_control/acrobot/

The system consists of two links connected linearly to form a chain, with one end
fixed. The joint between the two links is actuated. The goal is to apply torques
on the actuated joint to swing the free end of the linear chain above a given
height while starting from the initial state of hanging downwards.

Rendering uses native SDL2 bindings (no Python/pygame dependency).
Requires SDL2 and SDL2_ttf: brew install sdl2 sdl2_ttf
"""

from std.math import cos, sin, floor, pi
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
    black,
    rgb,
)
from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext, DeviceBuffer
from std.random.philox import Random as PhiloxRandom
from noeira.core.tiles import Tile, ImmTile, tile


# =============================================================================
# Physics Constants (shared by CPU and GPU kernels)
# =============================================================================

comptime gpu_dtype = DType.float32

# Physics parameters (Gymnasium Acrobot-v1, use_book_dynamics=True)
comptime ACR_GRAVITY: Float64 = 9.8
comptime ACR_LINK_LENGTH_1: Float64 = 1.0
comptime ACR_LINK_LENGTH_2: Float64 = 1.0
comptime ACR_LINK_MASS_1: Float64 = 1.0
comptime ACR_LINK_MASS_2: Float64 = 1.0
comptime ACR_LINK_COM_POS_1: Float64 = 0.5
comptime ACR_LINK_COM_POS_2: Float64 = 0.5
comptime ACR_LINK_MOI: Float64 = 1.0
comptime ACR_MAX_VEL_1: Float64 = 4.0 * pi
comptime ACR_MAX_VEL_2: Float64 = 9.0 * pi
comptime ACR_DT: Float64 = 0.2
comptime ACR_MAX_STEPS: Int = 500


# =============================================================================
# Single-source dynamics — the ONE copy of the Acrobot equations of motion.
#
# Called by both the CPU `_dsdt` (instance link params, runtime
# use_book_dynamics flag) and the GPU `_dsdt_gpu` (comptime ACR_* constants,
# book dynamics). These used to be two hand-maintained copies — the exact
# fork-drift class that produced real CPU/GPU divergences elsewhere.
# =============================================================================


@always_inline
def acrobot_dsdt[
    DTYPE: DType
](
    s: SIMD[DTYPE, 4],
    torque: Scalar[DTYPE],
    m1: Scalar[DTYPE],
    m2: Scalar[DTYPE],
    l1: Scalar[DTYPE],
    lc1: Scalar[DTYPE],
    lc2: Scalar[DTYPE],
    I1: Scalar[DTYPE],
    I2: Scalar[DTYPE],
    g: Scalar[DTYPE],
    use_book_dynamics: Bool,
) -> SIMD[DTYPE, 4]:
    """Equations of motion (Gymnasium Acrobot-v1). `s` is
    [theta1, theta2, dtheta1, dtheta2]; returns their derivatives."""
    comptime assert DTYPE.is_floating_point(), "DTYPE must be floating point"
    var theta1 = s[0]
    var theta2 = s[1]
    var dtheta1 = s[2]
    var dtheta2 = s[3]

    var cos_theta2 = cos(theta2)
    var sin_theta2 = sin(theta2)
    var pi_2 = Scalar[DTYPE](pi / 2.0)
    var cos_t1_t2_pi2 = cos(theta1 + theta2 - pi_2)
    var cos_t1_pi2 = cos(theta1 - pi_2)

    var d1 = (
        m1 * lc1 * lc1
        + m2 * (l1 * l1 + lc2 * lc2 + Scalar[DTYPE](2.0) * l1 * lc2 * cos_theta2)
        + I1
        + I2
    )
    var d2 = m2 * (lc2 * lc2 + l1 * lc2 * cos_theta2) + I2
    var phi2 = m2 * lc2 * g * cos_t1_t2_pi2
    var phi1 = (
        -m2 * l1 * lc2 * dtheta2 * dtheta2 * sin_theta2
        - Scalar[DTYPE](2.0) * m2 * l1 * lc2 * dtheta2 * dtheta1 * sin_theta2
        + (m1 * lc1 + m2 * l1) * g * cos_t1_pi2
        + phi2
    )

    var ddtheta2: Scalar[DTYPE]
    if use_book_dynamics:
        # Book dynamics (includes the extra dtheta1^2 term)
        ddtheta2 = (
            torque
            + d2 / d1 * phi1
            - m2 * l1 * lc2 * dtheta1 * dtheta1 * sin_theta2
            - phi2
        ) / (m2 * lc2 * lc2 + I2 - d2 * d2 / d1)
    else:
        # NIPS paper dynamics
        ddtheta2 = (torque + d2 / d1 * phi1 - phi2) / (
            m2 * lc2 * lc2 + I2 - d2 * d2 / d1
        )

    var ddtheta1 = -(d2 * ddtheta2 + phi1) / d1

    var out = SIMD[DTYPE, 4](0.0)
    out[0] = dtheta1
    out[1] = dtheta2
    out[2] = ddtheta1
    out[3] = ddtheta2
    return out


# ============================================================================
# Acrobot State and Action types for trait conformance
# ============================================================================


@fieldwise_init
struct AcrobotState(Copyable, ImplicitlyCopyable, Movable, State):
    """State for Acrobot: discretized state index.

    The continuous observation [cos(θ1), sin(θ1), cos(θ2), sin(θ2), θ1_dot, θ2_dot]
    is discretized into bins to create a single integer state index for tabular methods.
    """

    var index: Int

    def __init__(out self, *, copy: Self):
        self.index = copy.index

    def __init__(out self, *, deinit move: Self):
        self.index = move.index

    def __eq__(self, other: Self) -> Bool:
        return self.index == other.index


@fieldwise_init
struct AcrobotAction(Action, Copyable, ImplicitlyCopyable, Movable):
    """Action for Acrobot: 0 (-1 torque), 1 (0 torque), 2 (+1 torque)."""

    var torque_idx: Int

    def __init__(out self, *, copy: Self):
        self.torque_idx = copy.torque_idx

    def __init__(out self, *, deinit move: Self):
        self.torque_idx = move.torque_idx

    @staticmethod
    def negative() -> Self:
        """Apply -1 torque."""
        return Self(torque_idx=0)

    @staticmethod
    def zero() -> Self:
        """Apply 0 torque."""
        return Self(torque_idx=1)

    @staticmethod
    def positive() -> Self:
        """Apply +1 torque."""
        return Self(torque_idx=2)


# ============================================================================
# Helper functions for physics
# ============================================================================


def wrap(x: Float64, m: Float64, M: Float64) -> Float64:
    """Wraps x so m <= x <= M using modular arithmetic.

    For example, m = -pi, M = pi, x = 2*pi --> returns 0.

    Args:
        x: A scalar value to wrap.
        m: Minimum possible value in range.
        M: Maximum possible value in range.

    Returns:
        X wrapped to [m, M].
    """
    var diff = M - m
    var result = x
    while result > M:
        result = result - diff
    while result < m:
        result = result + diff
    return result


def bound(x: Float64, m: Float64, M: Float64) -> Float64:
    """Clamps x to be within [m, M].

    Args:
        x: Scalar value to clamp.
        m: Lower bound.
        M: Upper bound.

    Returns:
        X clamped between m and M.
    """
    if x < m:
        return m
    elif x > M:
        return M
    return x


struct AcrobotEnv[DTYPE: DType](
    BoxDiscreteActionEnv
    & DiscreteEnv
    & GPUDiscreteEnv
    & RenderableEnv
    & Movable
):
    """Native Mojo Acrobot environment with integrated SDL2 rendering.

    State: [theta1, theta2, theta1_dot, theta2_dot] (internal).
    Observation: [cos(θ1), sin(θ1), cos(θ2), sin(θ2), θ1_dot, θ2_dot] (6D).
    Actions: 0 (-1 torque), 1 (0 torque), 2 (+1 torque).

    Episode terminates when:
    - Free end reaches target height: -cos(θ1) - cos(θ2 + θ1) > 1.0.
    - Episode length > 500 steps.

    Implements:
    - DiscreteEnv: for tabular methods.
    - BoxDiscreteActionEnv: for function approximation with continuous 6D obs.
    - GPUDiscreteEnv: for fused GPU kernels (A2C, PPO, EZ-V2, etc.). The GPU
      path always uses book dynamics (matches the default CPU `__init__`).
    """

    # Type aliases for trait conformance
    comptime dtype = Self.DTYPE
    comptime StateType = AcrobotState
    comptime ActionType = AcrobotAction

    # GPUDiscreteEnv trait constants
    # State layout: [theta1, theta2, theta1_dot, theta2_dot, step_count]
    comptime STATE_SIZE: Int = 5
    # Obs: [cos(θ1), sin(θ1), cos(θ2), sin(θ2), θ1_dot, θ2_dot]
    comptime OBS_DIM: Int = 6
    comptime NUM_ACTIONS: Int = 3
    comptime STEP_WS_SHARED: Int = 0
    comptime STEP_WS_PER_ENV: Int = 0

    # Physical constants (same as Gymnasium)
    var gravity: Scalar[Self.dtype]
    var link_length_1: Scalar[Self.dtype]  # Length of link 1 [m]
    var link_length_2: Scalar[Self.dtype]  # Length of link 2 [m]
    var link_mass_1: Scalar[Self.dtype]  # Mass of link 1 [kg]
    var link_mass_2: Scalar[Self.dtype]  # Mass of link 2 [kg]
    var link_com_pos_1: Scalar[
        Self.dtype
    ]  # Position of center of mass of link 1 [m]
    var link_com_pos_2: Scalar[
        Self.dtype
    ]  # Position of center of mass of link 2 [m]
    var link_moi: Scalar[Self.dtype]  # Moments of inertia for both links

    var max_vel_1: Scalar[Self.dtype]  # Max angular velocity for joint 1
    var max_vel_2: Scalar[Self.dtype]  # Max angular velocity for joint 2

    var avail_torque: SIMD[Self.dtype, 4]  # Available torques [-1, 0, 1]
    var torque_noise_max: Scalar[Self.dtype]
    var dt: Scalar[Self.dtype]  # Time step

    # Current state (angles and angular velocities)
    var theta1: Scalar[Self.dtype]  # Angle of link 1 (0 = pointing down)
    var theta2: Scalar[Self.dtype]  # Angle of link 2 relative to link 1
    var theta1_dot: Scalar[Self.dtype]  # Angular velocity of link 1
    var theta2_dot: Scalar[Self.dtype]  # Angular velocity of link 2

    # Episode tracking
    var steps: Int
    var max_steps: Int
    var done: Bool
    # Natural-termination flag (free end above target height), NOT time-limit
    # truncation. Read by off-policy/on-policy drivers via `was_terminated()`.
    var _last_terminated: Bool
    var total_reward: Scalar[Self.dtype]

    # Discretization settings (for DiscreteEnv)
    var num_bins: Int

    # Book or NIPS dynamics
    var use_book_dynamics: Bool

    # Renderer (RenderableEnv)
    var _renderer: Optional[Pointer[Renderer2D, MutUntrackedOrigin]]
    var _renderer_initialized: Bool

    def __init__(out self, num_bins: Int = 6, use_book_dynamics: Bool = True):
        """Initialize Acrobot with default physics parameters.

        Args:
            num_bins: Number of bins per dimension for state discretization.
            use_book_dynamics: If True, use book dynamics; if False, use NIPS paper dynamics.
        """
        # Physics constants from Gymnasium
        self.gravity = Scalar[Self.dtype](9.8)
        self.link_length_1 = Scalar[Self.dtype](1.0)
        self.link_length_2 = Scalar[Self.dtype](1.0)
        self.link_mass_1 = Scalar[Self.dtype](1.0)
        self.link_mass_2 = Scalar[Self.dtype](1.0)
        self.link_com_pos_1 = Scalar[Self.dtype](0.5)
        self.link_com_pos_2 = Scalar[Self.dtype](0.5)
        self.link_moi = Scalar[Self.dtype](1.0)

        self.max_vel_1 = Scalar[Self.dtype](4.0 * pi)
        self.max_vel_2 = Scalar[Self.dtype](9.0 * pi)

        # Available torques: [-1, 0, +1] (padded to SIMD width 4)
        self.avail_torque = SIMD[Self.dtype, 4](-1.0, 0.0, 1.0, 0.0)
        self.torque_noise_max = Scalar[Self.dtype](0.0)
        self.dt = Scalar[Self.dtype](0.2)  # Time step

        # State
        self.theta1 = Scalar[Self.dtype](0.0)
        self.theta2 = Scalar[Self.dtype](0.0)
        self.theta1_dot = Scalar[Self.dtype](0.0)
        self.theta2_dot = Scalar[Self.dtype](0.0)

        # Episode
        self.steps = 0
        self.max_steps = 500
        self.done = False
        self._last_terminated = False
        self.total_reward = Scalar[Self.dtype](0.0)

        # Discretization settings
        self.num_bins = num_bins

        # Dynamics mode
        self.use_book_dynamics = use_book_dynamics

        # Renderer
        self._renderer = None
        self._renderer_initialized = False

    def __init__(out self, *, deinit move: Self):
        """Move-init — required for `Movable` conformance, used by
        `Pointer.unsafe_write(...)` in multi-env demos."""
        self.gravity = move.gravity
        self.link_length_1 = move.link_length_1
        self.link_length_2 = move.link_length_2
        self.link_mass_1 = move.link_mass_1
        self.link_mass_2 = move.link_mass_2
        self.link_com_pos_1 = move.link_com_pos_1
        self.link_com_pos_2 = move.link_com_pos_2
        self.link_moi = move.link_moi
        self.max_vel_1 = move.max_vel_1
        self.max_vel_2 = move.max_vel_2
        self.avail_torque = move.avail_torque
        self.torque_noise_max = move.torque_noise_max
        self.dt = move.dt
        self.theta1 = move.theta1
        self.theta2 = move.theta2
        self.theta1_dot = move.theta1_dot
        self.theta2_dot = move.theta2_dot
        self.steps = move.steps
        self.max_steps = move.max_steps
        self.done = move.done
        self._last_terminated = move._last_terminated
        self.total_reward = move.total_reward
        self.num_bins = move.num_bins
        self.use_book_dynamics = move.use_book_dynamics
        self._renderer = move._renderer
        self._renderer_initialized = move._renderer_initialized

    # ========================================================================
    # DiscreteEnv trait methods
    # ========================================================================

    def reset(mut self) -> AcrobotState:
        """Reset environment to random initial state.

        Returns AcrobotState with discretized state index.
        """
        # Random initial state in [-0.1, 0.1] for each component
        self.theta1 = Scalar[Self.dtype]((random_float64() - 0.5) * 0.2)
        self.theta2 = Scalar[Self.dtype]((random_float64() - 0.5) * 0.2)
        self.theta1_dot = Scalar[Self.dtype]((random_float64() - 0.5) * 0.2)
        self.theta2_dot = Scalar[Self.dtype]((random_float64() - 0.5) * 0.2)

        self.steps = 0
        self.done = False
        self._last_terminated = False
        self.total_reward = Scalar[Self.dtype](0.0)

        return AcrobotState(index=self._discretize_obs())

    def step(
        mut self, action: AcrobotAction, verbose: Bool = False
    ) -> Tuple[AcrobotState, Scalar[Self.dtype], Bool]:
        """Take action and return (state, reward, done).

        Args:
            action: AcrobotAction with torque_idx (0=-1, 1=0, 2=+1).
            verbose: Whether to print verbose output (default: False).

        Physics uses 4th-order Runge-Kutta integration (same as Gymnasium).
        """
        # Get torque from action
        var torque = self.avail_torque[action.torque_idx]

        # Add noise to torque if configured
        if self.torque_noise_max > Scalar[Self.dtype](0.0):
            torque += (
                Scalar[Self.dtype]((random_float64() - 0.5) * 2.0)
                * self.torque_noise_max
            )

        # Perform RK4 integration
        var ns = self._rk4_step(torque)

        # Wrap angles to [-pi, pi]
        self.theta1 = self._wrap(
            ns[0], Scalar[Self.dtype](-pi), Scalar[Self.dtype](pi)
        )
        self.theta2 = self._wrap(
            ns[1], Scalar[Self.dtype](-pi), Scalar[Self.dtype](pi)
        )
        # Bound velocities
        self.theta1_dot = self._bound(ns[2], -self.max_vel_1, self.max_vel_1)
        self.theta2_dot = self._bound(ns[3], -self.max_vel_2, self.max_vel_2)

        self.steps += 1

        # Check termination: free end above target height
        var terminated = self._terminal()
        var truncated = self.steps >= self.max_steps

        self.done = terminated or truncated
        self._last_terminated = terminated

        # Reward: -1 for each step, 0 at terminal
        var reward = Scalar[Self.dtype](0.0) if terminated else Scalar[
            Self.dtype
        ](-1.0)
        self.total_reward += reward

        return (AcrobotState(index=self._discretize_obs()), reward, self.done)

    def get_state(mut self) -> AcrobotState:
        """Return current discretized state."""
        return AcrobotState(index=self._discretize_obs())

    def state_to_index(self, state: AcrobotState) -> Int:
        """Convert an AcrobotState to an index for tabular methods."""
        return state.index

    def action_from_index(self, action_idx: Int) -> AcrobotAction:
        """Create an AcrobotAction from an index."""
        return AcrobotAction(torque_idx=action_idx)

    # ========================================================================
    # Internal physics helpers
    # ========================================================================

    def _wrap(
        self,
        x: Scalar[Self.dtype],
        m: Scalar[Self.dtype],
        M: Scalar[Self.dtype],
    ) -> Scalar[Self.dtype]:
        """Wraps x so m <= x <= M using modular arithmetic."""
        var diff = M - m
        var result = x
        while result > M:
            result = result - diff
        while result < m:
            result = result + diff
        return result

    def _bound(
        self,
        x: Scalar[Self.dtype],
        m: Scalar[Self.dtype],
        M: Scalar[Self.dtype],
    ) -> Scalar[Self.dtype]:
        """Clamps x to be within [m, M]."""
        if x < m:
            return m
        elif x > M:
            return M
        return x

    def _dsdt(
        self, s: SIMD[Self.dtype, 4], torque: Scalar[Self.dtype]
    ) -> SIMD[Self.dtype, 4]:
        """Compute derivatives for the equations of motion.

        Args:
            s: State [theta1, theta2, dtheta1, dtheta2]
            torque: Applied torque at the actuated joint

        Returns:
            Derivatives [dtheta1, dtheta2, ddtheta1, ddtheta2]
        """
        # Single-source dynamics (shared with the GPU kernel).
        return acrobot_dsdt[Self.dtype](
            s,
            torque,
            self.link_mass_1,
            self.link_mass_2,
            self.link_length_1,
            self.link_com_pos_1,
            self.link_com_pos_2,
            self.link_moi,
            self.link_moi,
            self.gravity,
            self.use_book_dynamics,
        )

    def _rk4_step(self, torque: Scalar[Self.dtype]) -> SIMD[Self.dtype, 4]:
        """Perform one RK4 integration step.

        Args:
            torque: Applied torque at the actuated joint

        Returns:
            New state [theta1, theta2, dtheta1, dtheta2]
        """
        var y0 = SIMD[Self.dtype, 4](
            self.theta1, self.theta2, self.theta1_dot, self.theta2_dot
        )

        var dt = self.dt
        var dt2 = dt / Scalar[Self.dtype](2.0)

        var k1 = self._dsdt(y0, torque)
        var k2 = self._dsdt(y0 + dt2 * k1, torque)
        var k3 = self._dsdt(y0 + dt2 * k2, torque)
        var k4 = self._dsdt(y0 + dt * k3, torque)

        return y0 + dt / Scalar[Self.dtype](6.0) * (
            k1
            + Scalar[Self.dtype](2.0) * k2
            + Scalar[Self.dtype](2.0) * k3
            + k4
        )

    def _terminal(self) -> Bool:
        """Check if the free end has reached the target height."""
        return (
            -cos(Float64(self.theta1)) - cos(Float64(self.theta2 + self.theta1))
            > 1.0
        )

    # ========================================================================
    # Observation helpers
    # ========================================================================

    @always_inline
    def _get_obs(self) -> SIMD[Self.dtype, 8]:
        """Return current continuous observation.

        Returns [cos(θ1), sin(θ1), cos(θ2), sin(θ2), θ1_dot, θ2_dot, 0, 0]
        (padded to SIMD width 8).
        """
        var obs = SIMD[Self.dtype, 8]()
        obs[0] = Scalar[Self.dtype](cos(Float64(self.theta1)))
        obs[1] = Scalar[Self.dtype](sin(Float64(self.theta1)))
        obs[2] = Scalar[Self.dtype](cos(Float64(self.theta2)))
        obs[3] = Scalar[Self.dtype](sin(Float64(self.theta2)))
        obs[4] = self.theta1_dot
        obs[5] = self.theta2_dot
        obs[6] = Scalar[Self.dtype](0.0)
        obs[7] = Scalar[Self.dtype](0.0)
        return obs

    @always_inline
    def _discretize_obs(self) -> Int:
        """Discretize current continuous observation into a single state index.

        Uses 6 dimensions: [cos(θ1), sin(θ1), cos(θ2), sin(θ2), θ1_dot, θ2_dot]
        """
        var n = self.num_bins

        # cos(theta1): [-1, 1]
        var n0 = (cos(Float64(self.theta1)) + 1.0) / 2.0
        if n0 < 0.0:
            n0 = 0.0
        elif n0 > 1.0:
            n0 = 1.0
        var b0 = Int(n0 * Float64(n - 1))

        # sin(theta1): [-1, 1]
        var n1 = (sin(Float64(self.theta1)) + 1.0) / 2.0
        if n1 < 0.0:
            n1 = 0.0
        elif n1 > 1.0:
            n1 = 1.0
        var b1 = Int(n1 * Float64(n - 1))

        # cos(theta2): [-1, 1]
        var n2 = (cos(Float64(self.theta2)) + 1.0) / 2.0
        if n2 < 0.0:
            n2 = 0.0
        elif n2 > 1.0:
            n2 = 1.0
        var b2 = Int(n2 * Float64(n - 1))

        # sin(theta2): [-1, 1]
        var n3 = (sin(Float64(self.theta2)) + 1.0) / 2.0
        if n3 < 0.0:
            n3 = 0.0
        elif n3 > 1.0:
            n3 = 1.0
        var b3 = Int(n3 * Float64(n - 1))

        # theta1_dot: [-4*pi, 4*pi]
        var n4 = (Float64(self.theta1_dot) + Float64(self.max_vel_1)) / (
            2.0 * Float64(self.max_vel_1)
        )
        if n4 < 0.0:
            n4 = 0.0
        elif n4 > 1.0:
            n4 = 1.0
        var b4 = Int(n4 * Float64(n - 1))

        # theta2_dot: [-9*pi, 9*pi]
        var n5 = (Float64(self.theta2_dot) + Float64(self.max_vel_2)) / (
            2.0 * Float64(self.max_vel_2)
        )
        if n5 < 0.0:
            n5 = 0.0
        elif n5 > 1.0:
            n5 = 1.0
        var b5 = Int(n5 * Float64(n - 1))

        return ((((b0 * n + b1) * n + b2) * n + b3) * n + b4) * n + b5

    @always_inline
    def get_obs(self) -> SIMD[Self.dtype, 8]:
        """Return current continuous observation as SIMD (optimized)."""
        return self._get_obs()

    # ========================================================================
    # ContinuousStateEnv / BoxDiscreteActionEnv trait methods
    # ========================================================================

    def get_obs_list(self) -> List[Scalar[Self.dtype]]:
        """Return current continuous observation as a flexible list (trait method).

        Returns true 6D observation without padding.
        """
        var obs = List[Scalar[Self.dtype]](capacity=6)
        obs.append(Scalar[Self.dtype](cos(Float64(self.theta1))))
        obs.append(Scalar[Self.dtype](sin(Float64(self.theta1))))
        obs.append(Scalar[Self.dtype](cos(Float64(self.theta2))))
        obs.append(Scalar[Self.dtype](sin(Float64(self.theta2))))
        obs.append(self.theta1_dot)
        obs.append(self.theta2_dot)
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

        This is the BoxDiscreteActionEnv trait method using List[Scalar[Self.dtype]].
        For performance-critical code, use step_raw() which returns SIMD.
        """
        var result = self.step_raw(action)
        return (self.get_obs_list(), result[1], result[2])

    def was_terminated(self) -> Bool:
        """True iff the previous step reached the goal height (natural
        termination), NOT the time-limit truncation at max_steps."""
        return self._last_terminated

    # ========================================================================
    # SIMD-optimized observation API (for performance)
    # ========================================================================

    def reset_obs(mut self) -> SIMD[Self.dtype, 8]:
        """Reset environment and return raw continuous observation.

        Use this for function approximation methods (tile coding, linear FA)
        that need the continuous observation vector.

        Returns:
            Continuous observation [cos(θ1), sin(θ1), cos(θ2), sin(θ2), θ1_dot, θ2_dot, 0, 0].
        """
        _ = self.reset()  # Reset internal state
        return self._get_obs()

    @always_inline
    def step_raw(
        mut self, action: Int
    ) -> Tuple[SIMD[Self.dtype, 8], Scalar[Self.dtype], Bool]:
        """Take action and return raw continuous observation.

        Use this for function approximation methods that need the continuous
        observation vector rather than discretized state.

        Args:
            action: 0 for -1 torque, 1 for 0 torque, 2 for +1 torque.

        Returns:
            Tuple of (observation, reward, done).
        """
        # Get torque from action
        var torque = self.avail_torque[action]

        # Add noise to torque if configured
        if self.torque_noise_max > Scalar[Self.dtype](0.0):
            torque += (
                Scalar[Self.dtype]((random_float64() - 0.5) * 2.0)
                * self.torque_noise_max
            )

        # Perform RK4 integration
        var ns = self._rk4_step(torque)

        # Wrap angles to [-pi, pi]
        self.theta1 = self._wrap(
            ns[0], Scalar[Self.dtype](-pi), Scalar[Self.dtype](pi)
        )
        self.theta2 = self._wrap(
            ns[1], Scalar[Self.dtype](-pi), Scalar[Self.dtype](pi)
        )
        # Bound velocities
        self.theta1_dot = self._bound(ns[2], -self.max_vel_1, self.max_vel_1)
        self.theta2_dot = self._bound(ns[3], -self.max_vel_2, self.max_vel_2)

        self.steps += 1

        # Check termination
        var terminated = self._terminal()
        var truncated = self.steps >= self.max_steps
        self.done = terminated or truncated
        self._last_terminated = terminated

        var reward = Scalar[Self.dtype](0.0) if terminated else Scalar[
            Self.dtype
        ](-1.0)
        self.total_reward += reward

        return (self._get_obs(), reward, self.done)

    # ========================================================================
    # Rendering
    # ========================================================================

    def render(mut self, mut renderer: Renderer2D):
        """Render the current state using SDL2.

        Uses Camera for world-to-screen coordinate conversion.

        Args:
            renderer: The Renderer2D to use for rendering.
        """
        # Begin frame handles init, events, and clear
        if not renderer.begin_frame():
            return

        # Convert state variables to Float64 for rendering
        var theta1_f64 = Float64(self.theta1)
        var theta2_f64 = Float64(self.theta2)
        var link_length_1_f64 = Float64(self.link_length_1)
        var link_length_2_f64 = Float64(self.link_length_2)

        # Colors
        var link_color = rgb(0, 204, 204)  # Cyan
        var joint_color = rgb(204, 204, 0)  # Yellow
        var target_color = black()
        var link_width = 10

        # Create camera with appropriate zoom
        # Total reach is link_length_1 + link_length_2, add margin
        var bound_val = link_length_1_f64 + link_length_2_f64 + 0.2
        var zoom = Float64(
            min(renderer.screen_width, renderer.screen_height)
        ) / (bound_val * 2.0)
        var camera = renderer.make_camera(zoom, True)

        # World coordinates (Y points up, 0,0 at center)
        var p0 = Vec2(0.0, 0.0)  # Fixed pivot at origin

        # First link endpoint
        # theta1=0 means pointing straight down (negative Y in world coords)
        var p1 = Vec2(
            p0.x + link_length_1_f64 * sin(theta1_f64),
            p0.y - link_length_1_f64 * cos(theta1_f64),
        )

        # Second link endpoint
        # theta2 is relative to theta1
        var angle2 = theta1_f64 + theta2_f64
        var p2 = Vec2(
            p1.x + link_length_2_f64 * sin(angle2),
            p1.y - link_length_2_f64 * cos(angle2),
        )

        # Draw target line (height = 1.0 above the fixed point)
        renderer.draw_ground_line(1.0, camera, target_color, 2)

        # Draw links using helper methods
        renderer.draw_link(p0, p1, camera, link_color, link_width)
        renderer.draw_link(p1, p2, camera, link_color, link_width)

        # Draw joints
        var joint_radius = 0.05
        renderer.draw_joint(p0, joint_radius, camera, joint_color)
        renderer.draw_joint(p1, joint_radius, camera, joint_color)

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

    @always_inline
    def is_done(self) -> Bool:
        """Check if episode is done."""
        return self.done

    @always_inline
    def num_actions(self) -> Int:
        """Return number of actions (3)."""
        return 3

    @always_inline
    def obs_dim(self) -> Int:
        """Return observation dimension (6)."""
        return 6

    @always_inline
    def num_states(self) -> Int:
        """Return total number of discrete states."""
        var n = self.num_bins
        return n * n * n * n * n * n  # 6 dimensions

    # ========================================================================
    # Static methods for discretization and feature extraction
    # ========================================================================

    @staticmethod
    def get_num_states(num_bins: Int = 6) -> Int:
        """Get the number of discrete states for Acrobot with given bins."""
        return num_bins * num_bins * num_bins * num_bins * num_bins * num_bins

    @staticmethod
    def discretize_obs(obs: SIMD[DType.float64, 8], num_bins: Int = 6) -> Int:
        """Discretize continuous observation into a single state index.

        Args:
            obs: Continuous observation [cos(θ1), sin(θ1), cos(θ2), sin(θ2), θ1_dot, θ2_dot, 0, 0].
            num_bins: Number of bins per dimension.

        Returns:
            Single integer state index.
        """
        var n = num_bins
        var max_vel_1 = 4.0 * pi
        var max_vel_2 = 9.0 * pi

        def bin_value(
            value: Float64, low: Float64, high: Float64, bins: Int
        ) -> Int:
            var normalized = (value - low) / (high - low)
            if normalized < 0.0:
                normalized = 0.0
            elif normalized > 1.0:
                normalized = 1.0
            return Int(normalized * Float64(bins - 1))

        var b0 = bin_value(obs[0], -1.0, 1.0, n)  # cos(theta1)
        var b1 = bin_value(obs[1], -1.0, 1.0, n)  # sin(theta1)
        var b2 = bin_value(obs[2], -1.0, 1.0, n)  # cos(theta2)
        var b3 = bin_value(obs[3], -1.0, 1.0, n)  # sin(theta2)
        var b4 = bin_value(obs[4], -max_vel_1, max_vel_1, n)  # theta1_dot
        var b5 = bin_value(obs[5], -max_vel_2, max_vel_2, n)  # theta2_dot

        return ((((b0 * n + b1) * n + b2) * n + b3) * n + b4) * n + b5

    @staticmethod
    def make_tile_coding(
        num_tilings: Int = 8,
        tiles_per_dim: Int = 8,
    ) -> TileCoding[Self.dtype]:
        """Create tile coding configured for Acrobot environment.

        Acrobot observation: [cos(θ1), sin(θ1), cos(θ2), sin(θ2), θ1_dot, θ2_dot].

        Args:
            num_tilings: Number of tilings (default 8).
            tiles_per_dim: Tiles per dimension (default 8).

        Returns:
            TileCoding[Self.dtype] configured for Acrobot observation space.
        """
        var tiles = List[Int]()
        tiles.append(tiles_per_dim)
        tiles.append(tiles_per_dim)
        tiles.append(tiles_per_dim)
        tiles.append(tiles_per_dim)
        tiles.append(tiles_per_dim)
        tiles.append(tiles_per_dim)

        # Acrobot observation bounds
        var state_low = List[Scalar[Self.dtype]]()
        state_low.append(-1.0)  # cos(theta1)
        state_low.append(-1.0)  # sin(theta1)
        state_low.append(-1.0)  # cos(theta2)
        state_low.append(-1.0)  # sin(theta2)
        state_low.append(Scalar[Self.dtype](-4.0 * pi))  # theta1_dot
        state_low.append(Scalar[Self.dtype](-9.0 * pi))  # theta2_dot

        var state_high = List[Scalar[Self.dtype]]()
        state_high.append(1.0)
        state_high.append(1.0)
        state_high.append(1.0)
        state_high.append(1.0)
        state_high.append(Scalar[Self.dtype](4.0 * pi))
        state_high.append(Scalar[Self.dtype](9.0 * pi))

        return TileCoding[Self.dtype](
            num_tilings=num_tilings,
            tiles_per_dim=tiles^,
            state_low=state_low^,
            state_high=state_high^,
        )

    @staticmethod
    def make_poly_features(degree: Int = 2) -> PolynomialFeatures[Self.dtype]:
        """Create polynomial features for Acrobot (6D observation) with normalization.

        Acrobot observation: [cos(θ1), sin(θ1), cos(θ2), sin(θ2), θ1_dot, θ2_dot].

        Args:
            degree: Maximum polynomial degree (keep low for 6D to avoid explosion).

        Returns:
            PolynomialFeatures[Self.dtype] extractor configured for Acrobot with normalization.
        """
        var state_low = List[Scalar[Self.dtype]]()
        state_low.append(-1.0)  # cos(theta1)
        state_low.append(-1.0)  # sin(theta1)
        state_low.append(-1.0)  # cos(theta2)
        state_low.append(-1.0)  # sin(theta2)
        state_low.append(Scalar[Self.dtype](-4.0 * pi))  # theta1_dot
        state_low.append(Scalar[Self.dtype](-9.0 * pi))  # theta2_dot

        var state_high = List[Scalar[Self.dtype]]()
        state_high.append(1.0)
        state_high.append(1.0)
        state_high.append(1.0)
        state_high.append(1.0)
        state_high.append(Scalar[Self.dtype](4.0 * pi))
        state_high.append(Scalar[Self.dtype](9.0 * pi))

        return PolynomialFeatures[Self.dtype](
            state_dim=6,
            degree=degree,
            include_bias=True,
            state_low=state_low^,
            state_high=state_high^,
        )

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

    # ========================================================================
    # GPUDiscreteEnv trait methods (for fused GPU kernels)
    # ========================================================================

    @staticmethod
    @always_inline
    def _wrap_pi_gpu(theta: Scalar[gpu_dtype]) -> Scalar[gpu_dtype]:
        """Wrap angle to [-pi, pi) using branch-free floor formula."""
        var two_pi = Scalar[gpu_dtype](2.0 * pi)
        var pi_s = Scalar[gpu_dtype](pi)
        return theta - two_pi * floor((theta + pi_s) / two_pi)

    @staticmethod
    @always_inline
    def _bound_gpu(
        x: Scalar[gpu_dtype],
        m: Scalar[gpu_dtype],
        M: Scalar[gpu_dtype],
    ) -> Scalar[gpu_dtype]:
        var y = x
        if y < m:
            y = m
        if y > M:
            y = M
        return y

    @staticmethod
    @always_inline
    def _dsdt_gpu(
        s: SIMD[gpu_dtype, 4], torque: Scalar[gpu_dtype]
    ) -> SIMD[gpu_dtype, 4]:
        """Equations of motion for Acrobot (book dynamics).

        Returns derivative SIMD[dtheta1, dtheta2, ddtheta1, ddtheta2].
        """
        # Single-source dynamics (shared with the CPU `_dsdt`); book
        # dynamics, comptime ACR_* constants.
        return acrobot_dsdt[gpu_dtype](
            s,
            torque,
            Scalar[gpu_dtype](ACR_LINK_MASS_1),
            Scalar[gpu_dtype](ACR_LINK_MASS_2),
            Scalar[gpu_dtype](ACR_LINK_LENGTH_1),
            Scalar[gpu_dtype](ACR_LINK_COM_POS_1),
            Scalar[gpu_dtype](ACR_LINK_COM_POS_2),
            Scalar[gpu_dtype](ACR_LINK_MOI),
            Scalar[gpu_dtype](ACR_LINK_MOI),
            Scalar[gpu_dtype](ACR_GRAVITY),
            True,
        )

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
        # rng_seed unused in Acrobot (no torque noise on the GPU path)
        var i = Int(block_dim.x * block_idx.x + thread_idx.x)
        if i >= BATCH_SIZE:
            return

        # Map action index → torque {-1, 0, +1}
        var a = Int(actions[i])
        var torque = Scalar[gpu_dtype](0.0)
        if a == 0:
            torque = Scalar[gpu_dtype](-1.0)
        elif a == 2:
            torque = Scalar[gpu_dtype](1.0)

        # Load state (rebind LayoutTensor element_type → Scalar[gpu_dtype])
        var y0 = SIMD[gpu_dtype, 4](0.0)
        y0[0] = rebind[Scalar[gpu_dtype]](states[i, 0])
        y0[1] = rebind[Scalar[gpu_dtype]](states[i, 1])
        y0[2] = rebind[Scalar[gpu_dtype]](states[i, 2])
        y0[3] = rebind[Scalar[gpu_dtype]](states[i, 3])

        # RK4 integration with dt = ACR_DT
        var dt = Scalar[gpu_dtype](ACR_DT)
        var dt2 = dt / Scalar[gpu_dtype](2.0)

        var k1 = Self._dsdt_gpu(y0, torque)
        var k2 = Self._dsdt_gpu(y0 + dt2 * k1, torque)
        var k3 = Self._dsdt_gpu(y0 + dt2 * k2, torque)
        var k4 = Self._dsdt_gpu(y0 + dt * k3, torque)

        var ns = y0 + dt / Scalar[gpu_dtype](6.0) * (
            k1 + Scalar[gpu_dtype](2.0) * k2 + Scalar[gpu_dtype](2.0) * k3 + k4
        )

        # Wrap angles, bound velocities
        var theta1 = Self._wrap_pi_gpu(ns[0])
        var theta2 = Self._wrap_pi_gpu(ns[1])
        var theta1_dot = Self._bound_gpu(
            ns[2],
            Scalar[gpu_dtype](-ACR_MAX_VEL_1),
            Scalar[gpu_dtype](ACR_MAX_VEL_1),
        )
        var theta2_dot = Self._bound_gpu(
            ns[3],
            Scalar[gpu_dtype](-ACR_MAX_VEL_2),
            Scalar[gpu_dtype](ACR_MAX_VEL_2),
        )

        # Write back state
        states[i, 0] = theta1
        states[i, 1] = theta2
        states[i, 2] = theta1_dot
        states[i, 3] = theta2_dot
        states[i, 4] += Scalar[gpu_dtype](1.0)  # step counter

        # Termination: free end above target height
        var terminated = -cos(theta1) - cos(theta1 + theta2) > Scalar[
            gpu_dtype
        ](1.0)
        var truncated = states[i, 4] >= Scalar[gpu_dtype](ACR_MAX_STEPS)
        var done = terminated or truncated

        # Reward: 0 on the terminal step, -1 otherwise
        var reward = Scalar[gpu_dtype](0.0) if terminated else Scalar[
            gpu_dtype
        ](-1.0)

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
        """Reset state to random initial values via Philox RNG.

        Matches CPU reset: each component sampled uniform in [-0.1, 0.1].
        """
        var i = Int(block_dim.x * block_idx.x + thread_idx.x)
        if i >= BATCH_SIZE:
            return

        var rng = PhiloxRandom(
            seed=UInt64(i) * UInt64(2654435761) + 12345, offset=0
        )
        var rand_vals = rng.step_uniform()

        # Map [0, 1) → [-0.1, 0.1]
        state[i, 0] = Scalar[gpu_dtype](rand_vals[0]) * Scalar[gpu_dtype](
            0.2
        ) - Scalar[gpu_dtype](0.1)
        state[i, 1] = Scalar[gpu_dtype](rand_vals[1]) * Scalar[gpu_dtype](
            0.2
        ) - Scalar[gpu_dtype](0.1)
        state[i, 2] = Scalar[gpu_dtype](rand_vals[2]) * Scalar[gpu_dtype](
            0.2
        ) - Scalar[gpu_dtype](0.1)
        state[i, 3] = Scalar[gpu_dtype](rand_vals[3]) * Scalar[gpu_dtype](
            0.2
        ) - Scalar[gpu_dtype](0.1)
        state[i, 4] = Scalar[gpu_dtype](0.0)  # step counter

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
        """Reset only env slots where dones[i] > 0.5; clears that done flag."""
        var i = Int(block_dim.x * block_idx.x + thread_idx.x)
        if i >= BATCH_SIZE:
            return

        if dones[i] < Scalar[gpu_dtype](0.5):
            return

        var rng = PhiloxRandom(
            seed=UInt64(i) * UInt64(2654435761) + UInt64(rng_seed),
            offset=0,
        )
        var rand_vals = rng.step_uniform()

        state[i, 0] = Scalar[gpu_dtype](rand_vals[0]) * Scalar[gpu_dtype](
            0.2
        ) - Scalar[gpu_dtype](0.1)
        state[i, 1] = Scalar[gpu_dtype](rand_vals[1]) * Scalar[gpu_dtype](
            0.2
        ) - Scalar[gpu_dtype](0.1)
        state[i, 2] = Scalar[gpu_dtype](rand_vals[2]) * Scalar[gpu_dtype](
            0.2
        ) - Scalar[gpu_dtype](0.1)
        state[i, 3] = Scalar[gpu_dtype](rand_vals[3]) * Scalar[gpu_dtype](
            0.2
        ) - Scalar[gpu_dtype](0.1)
        state[i, 4] = Scalar[gpu_dtype](0.0)

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

        Writes obs[i] = [cos(θ1), sin(θ1), cos(θ2), sin(θ2), θ1_dot, θ2_dot]
        from the post-step state.
        """
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
            var i = Int(block_dim.x * block_idx.x + thread_idx.x)
            if i < BATCH_SIZE:
                # Acrobot: terminated = goal-height crossed (not truncation).
                var theta1 = states[i, 0]
                var theta2 = states[i, 1]
                var is_terminated = -cos(theta1) - cos(
                    theta1 + theta2
                ) > Scalar[gpu_dtype](1.0)
                terminated_out[i] = Scalar[gpu_dtype](
                    1.0
                ) if is_terminated else Scalar[gpu_dtype](0.0)

                # Build observation from state
                obs[i, 0] = cos(theta1)
                obs[i, 1] = sin(theta1)
                obs[i, 2] = cos(theta2)
                obs[i, 3] = sin(theta2)
                obs[i, 4] = states[i, 2]
                obs[i, 5] = states[i, 3]

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
        """Launch reset kernel on GPU."""
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
        """Launch selective reset kernel on GPU - only resets done envs."""
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
        """No-op: Acrobot doesn't need pre-allocated workspace."""
        pass

    @staticmethod
    def update_curriculum_gpu(
        ctx: DeviceContext,
        mut workspace_buf: DeviceBuffer[gpu_dtype],
        curriculum_values: List[Scalar[gpu_dtype]],
    ) raises:
        """No-op: Acrobot doesn't use curriculum."""
        pass

    @staticmethod
    def extract_obs_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        OBS_DIM: Int,
    ](
        ctx: DeviceContext,
        states: DeviceBuffer[gpu_dtype],
        mut obs: DeviceBuffer[gpu_dtype],
    ) raises:
        """Override default: Acrobot's obs is a non-trivial map of state.

        obs[i] = [cos(θ1), sin(θ1), cos(θ2), sin(θ2), θ1_dot, θ2_dot]
        from state[i, 0:4] = [theta1, theta2, theta1_dot, theta2_dot].
        """
        var states_t = tile[BATCH_SIZE, STATE_SIZE](states)
        var obs_t = tile[BATCH_SIZE, OBS_DIM](obs)

        comptime BLOCKS = (BATCH_SIZE + Self.TPB - 1) // Self.TPB

        @always_inline
        def extract_wrapper(
            s: ImmTile[gpu_dtype, BATCH_SIZE, STATE_SIZE],
            o: Tile[gpu_dtype, BATCH_SIZE, OBS_DIM],
        ):
            var i = Int(block_dim.x * block_idx.x + thread_idx.x)
            if i >= BATCH_SIZE:
                return
            var theta1 = s[i, 0]
            var theta2 = s[i, 1]
            o[i, 0] = cos(theta1)
            o[i, 1] = sin(theta1)
            o[i, 2] = cos(theta2)
            o[i, 3] = sin(theta2)
            o[i, 4] = s[i, 2]
            o[i, 5] = s[i, 3]

        ctx.enqueue_function[extract_wrapper](
            states_t,
            obs_t,
            grid_dim=(BLOCKS,),
            block_dim=(Self.TPB,),
        )
