"""LunarLander GPU environment using the physics2d modular architecture.

This implementation uses the new modular physics components:
- LunarLanderLayout for compile-time layout computation
- PhysicsEnvHelpers for environment setup utilities

The flat state layout is compatible with GPUDiscreteEnv trait.
All physics data is packed per-environment for efficient GPU access.
"""

from std.math import sqrt, cos, sin, pi, tanh
from layout import Layout, LayoutTensor
from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceContext, DeviceBuffer
from std.memory import alloc
from std.random.philox import Random as PhiloxRandom

from noeira.core import (
    GPUDiscreteEnv,
    BoxDiscreteActionEnv,
    Action,
    GPUContinuousEnv,
    BoxContinuousActionEnv,
    RenderableEnv,
)

from .state import LunarLanderState
from .particle import Particle
from .action import LunarLanderAction
from .constants import LLConstants
from .helpers import (
    compute_shaping,
    lunar_terminal_and_reward,
    normalize_position,
    normalize_velocity,
    normalize_angular_velocity,
)
from noeira.physics2d.integrators.euler import SemiImplicitEuler
from noeira.physics2d.collision.edge_terrain import EdgeTerrainCollision
from noeira.physics2d.solvers.impulse import ImpulseSolver
from noeira.physics2d.joints.revolute import RevoluteJointSolver
from noeira.physics2d.island import (
    island_sleep_time_single_env,
    island_is_asleep,
)

from noeira.physics2d import (
    dtype,
    TPB,
    BODY_STATE_SIZE,
    SHAPE_MAX_SIZE,
    CONTACT_DATA_SIZE,
    JOINT_DATA_SIZE,
    IDX_X,
    IDX_Y,
    IDX_ANGLE,
    IDX_VX,
    IDX_VY,
    IDX_OMEGA,
    IDX_INV_MASS,
    IDX_INV_INERTIA,
    IDX_SHAPE,
    SHAPE_POLYGON,
    JOINT_REVOLUTE,
    JOINT_TYPE,
    JOINT_BODY_A,
    JOINT_BODY_B,
    JOINT_ANCHOR_AX,
    JOINT_ANCHOR_AY,
    JOINT_ANCHOR_BX,
    JOINT_ANCHOR_BY,
    JOINT_REF_ANGLE,
    JOINT_LOWER_LIMIT,
    JOINT_UPPER_LIMIT,
    JOINT_FLAGS,
    JOINT_FLAG_LIMIT_ENABLED,
    PhysicsState,
    PhysicsStateOwned,
    # Contact data indices for collision-based crash detection
    CONTACT_BODY_A,
    CONTACT_BODY_B,
    CONTACT_DEPTH,
    # New modular architecture
    LunarLanderLayout,
    PhysicsEnvHelpers,
    PhysicsConfig,
)
from noeira.physics2d.integrators.euler import SemiImplicitEuler
from noeira.physics2d.collision.edge_terrain import (
    EdgeTerrainCollision,
    MAX_TERRAIN_EDGES,
)
from noeira.physics2d.solvers.impulse import ImpulseSolver
from noeira.physics2d.joints.revolute import RevoluteJointSolver

# Rendering imports
from noeira.render import (
    Renderer2D,
    SDL_Color,
    Camera,
    Vec2 as RenderVec2,
    Transform2D,
    # Colors
    space_black,
    moon_gray,
    dark_gray,
    white,
    yellow,
    red,
    contact_green,
    inactive_gray,
    rgb,
    darken,
    # Shapes
    make_lander_body,
    make_leg_box,
    scale_vertices,
)

# =============================================================================
# LunarLander Environment
# =============================================================================


struct LunarLander[
    DTYPE: DType,
    ENABLE_WIND: Bool = False,
    WIND_POWER: Float64 = 15.0,
    TURBULENCE_POWER: Float64 = 1.5,
](
    BoxContinuousActionEnv,
    BoxDiscreteActionEnv,
    Copyable,
    GPUContinuousEnv,
    GPUDiscreteEnv,
    Movable,
    RenderableEnv,
):
    """LunarLander environment with full physics using GPU methods.

    This environment uses the existing physics2d architecture:
    - PhysicsState for accessing physics data in flat layout
    - SemiImplicitEuler.integrate_velocities_gpu
    - SemiImplicitEuler.integrate_positions_gpu
    - EdgeTerrainCollision.detect_gpu
    - ImpulseSolver.solve_velocity_gpu / solve_position_gpu
    - RevoluteJointSolver.solve_velocity_gpu / solve_position_gpu

    The structure follows lunar_lander_v2.mojo patterns but adapted for
    the GPUDiscreteEnv trait's flat state layout.
    """

    # Required trait aliases
    comptime STATE_SIZE: Int = LLConstants.STATE_SIZE_VAL
    comptime OBS_DIM: Int = LLConstants.OBS_DIM_VAL
    comptime NUM_ACTIONS: Int = LLConstants.NUM_ACTIONS_VAL
    comptime ACTION_DIM: Int = LLConstants.ACTION_DIM_VAL  # For GPUContinuousEnv
    # Pre-allocated workspace: shapes (shared) + contacts/counts (per-env)
    comptime STEP_WS_SHARED: Int = LLConstants.NUM_SHAPES * SHAPE_MAX_SIZE
    comptime STEP_WS_PER_ENV: Int = (
        LLConstants.MAX_CONTACTS * CONTACT_DATA_SIZE + 3
    )
    comptime dtype = Self.DTYPE
    comptime StateType = LunarLanderState[Self.dtype]
    comptime ActionType = LunarLanderAction
    comptime NAME: String = "LunarLander"

    # Type alias for static method calls (includes all compile-time params)
    comptime SelfType = LunarLander[
        Self.DTYPE, Self.ENABLE_WIND, Self.WIND_POWER, Self.TURBULENCE_POWER
    ]

    # Body index constants for instance methods
    comptime BODY_LANDER: Int = 0
    comptime BODY_LEFT_LEG: Int = 1
    comptime BODY_RIGHT_LEG: Int = 2

    # Particle effects (cosmetic only)
    var particles: List[Particle[Self.dtype]]

    # Physics state for CPU single-env operation
    var physics: PhysicsStateOwned[
        LLConstants.NUM_BODIES,
        LLConstants.NUM_SHAPES,
        LLConstants.MAX_CONTACTS,
        LLConstants.MAX_JOINTS,
        LLConstants.STATE_SIZE_VAL,
        LLConstants.BODIES_OFFSET,
        LLConstants.FORCES_OFFSET,
        LLConstants.JOINTS_OFFSET,
        LLConstants.JOINT_COUNT_OFFSET,
        LLConstants.EDGES_OFFSET,
        LLConstants.EDGE_COUNT_OFFSET,
    ]
    var config: PhysicsConfig

    # Environment state for CPU operation
    var prev_shaping: Scalar[Self.dtype]
    var step_count: Int
    var game_over: Bool
    # Natural-termination flag for the LAST step (crash/OOB/landed), False on
    # time-limit truncation — read by `was_terminated()` so drivers keep the
    # TD bootstrap on truncation and drop it on true terminals.
    var last_terminated: Bool
    var rng_seed: UInt64
    var rng_counter: UInt64

    # Wind state (runtime indices for deterministic wind progression)
    # Wind parameters (enable_wind, wind_power, turbulence_power) are compile-time struct params
    var wind_idx: Int
    var torque_idx: Int

    # Terrain heights (11 chunks for single-env CPU mode)
    var terrain_heights: List[Scalar[Self.dtype]]

    # Edge terrain collision system
    var edge_collision: EdgeTerrainCollision

    # Cached state for immutable get_state() access
    var cached_state: LunarLanderState[Self.dtype]

    # Renderer (RenderableEnv)
    var _renderer: Optional[Pointer[Renderer2D, MutUntrackedOrigin]]
    var _renderer_initialized: Bool

    # =========================================================================
    # Initialization
    # =========================================================================

    def __init__(
        out self,
        seed: UInt64 = 42,
    ):
        """Initialize the environment for CPU single-env operation.

        Wind parameters are now compile-time struct parameters:
        - ENABLE_WIND: Bool = False
        - WIND_POWER: Float64 = 15.0
        - TURBULENCE_POWER: Float64 = 1.5

        Args:
            seed: Random seed for reproducibility.
        """
        # Initialize particle list
        self.particles = List[Particle[Self.dtype]]()

        # Create physics state for single environment
        self.physics = PhysicsStateOwned[
            LLConstants.NUM_BODIES,
            LLConstants.NUM_SHAPES,
            LLConstants.MAX_CONTACTS,
            LLConstants.MAX_JOINTS,
            LLConstants.STATE_SIZE_VAL,
            LLConstants.BODIES_OFFSET,
            LLConstants.FORCES_OFFSET,
            LLConstants.JOINTS_OFFSET,
            LLConstants.JOINT_COUNT_OFFSET,
            LLConstants.EDGES_OFFSET,
            LLConstants.EDGE_COUNT_OFFSET,
        ]()

        # Create physics config
        self.config = PhysicsConfig(
            gravity_x=LLConstants.GRAVITY_X,
            gravity_y=LLConstants.GRAVITY_Y,
            dt=LLConstants.DT,
            friction=LLConstants.FRICTION,
            restitution=LLConstants.RESTITUTION,
            baumgarte=LLConstants.BAUMGARTE,
            slop=LLConstants.SLOP,
            velocity_iterations=LLConstants.VELOCITY_ITERATIONS,
            position_iterations=LLConstants.POSITION_ITERATIONS,
        )

        # Define lander shape as polygon (shape 0)
        var lander_vx = List[Float64]()
        var lander_vy = List[Float64]()
        lander_vx.append(-14.0 / LLConstants.SCALE)
        lander_vy.append(17.0 / LLConstants.SCALE)
        lander_vx.append(-17.0 / LLConstants.SCALE)
        lander_vy.append(0.0 / LLConstants.SCALE)
        lander_vx.append(-17.0 / LLConstants.SCALE)
        lander_vy.append(-10.0 / LLConstants.SCALE)
        lander_vx.append(17.0 / LLConstants.SCALE)
        lander_vy.append(-10.0 / LLConstants.SCALE)
        lander_vx.append(17.0 / LLConstants.SCALE)
        lander_vy.append(0.0 / LLConstants.SCALE)
        lander_vx.append(14.0 / LLConstants.SCALE)
        lander_vy.append(17.0 / LLConstants.SCALE)
        self.physics.define_polygon_shape(0, lander_vx, lander_vy)

        # Define leg shapes (shapes 1 and 2)
        var leg_vx = List[Float64]()
        var leg_vy = List[Float64]()
        leg_vx.append(-LLConstants.LEG_W)
        leg_vy.append(LLConstants.LEG_H)
        leg_vx.append(-LLConstants.LEG_W)
        leg_vy.append(-LLConstants.LEG_H)
        leg_vx.append(LLConstants.LEG_W)
        leg_vy.append(-LLConstants.LEG_H)
        leg_vx.append(LLConstants.LEG_W)
        leg_vy.append(LLConstants.LEG_H)
        self.physics.define_polygon_shape(1, leg_vx, leg_vy)
        self.physics.define_polygon_shape(2, leg_vx, leg_vy)

        # Initialize tracking variables
        self.prev_shaping = Scalar[Self.dtype](0)
        self.step_count = 0
        self.game_over = False
        self.last_terminated = False

        self.rng_seed = seed
        self.rng_counter = 0

        # Wind state indices (parameters are compile-time struct params)
        self.wind_idx = 0
        self.torque_idx = 0

        # Terrain heights
        self.terrain_heights = List[Scalar[Self.dtype]](
            capacity=LLConstants.TERRAIN_CHUNKS
        )
        for _ in range(LLConstants.TERRAIN_CHUNKS):
            self.terrain_heights.append(
                Scalar[Self.dtype](LLConstants.HELIPAD_Y)
            )

        # Edge terrain collision system
        self.edge_collision = EdgeTerrainCollision(1)

        # Initialize cached state
        self.cached_state = LunarLanderState[Self.dtype]()

        # Renderer
        self._renderer = None
        self._renderer_initialized = False

        # Reset to initial state
        self._reset_cpu()

    def __init__(out self, *, copy: Self):
        """Copy constructor — true deep copy: a copied env continues from the
        SAME mid-episode state (physics buffers, terrain, RNG counter). It
        must NOT reset — resetting on copy silently wiped state, regenerated
        terrain and advanced the RNG whenever an env was passed by value."""
        self.particles = List[Particle[Self.dtype]](copy.particles)
        # PhysicsStateOwned deep-copies its state/shapes/contacts buffers, so
        # the copied env's bodies (and the shapes _init_physics_shapes would
        # have rebuilt) come across verbatim.
        self.physics = copy.physics.copy()
        self.config = PhysicsConfig(
            gravity_x=copy.config.gravity_x,
            gravity_y=copy.config.gravity_y,
            dt=copy.config.dt,
            friction=copy.config.friction,
            restitution=copy.config.restitution,
            baumgarte=copy.config.baumgarte,
            slop=copy.config.slop,
            velocity_iterations=copy.config.velocity_iterations,
            position_iterations=copy.config.position_iterations,
        )
        self.prev_shaping = copy.prev_shaping
        self.step_count = copy.step_count
        self.game_over = copy.game_over
        self.last_terminated = copy.last_terminated
        self.rng_seed = copy.rng_seed
        self.rng_counter = copy.rng_counter
        self.wind_idx = copy.wind_idx
        self.torque_idx = copy.torque_idx
        self.terrain_heights = List[Scalar[Self.dtype]](copy.terrain_heights)
        self.edge_collision = copy.edge_collision.copy()
        self.cached_state = copy.cached_state

        # Do not copy renderer — reset to null
        self._renderer = None
        self._renderer_initialized = False

    def __init__(out self, *, deinit move: Self):
        """Move constructor — transfers all state verbatim (buffers move, no
        reset). Resetting on move silently wiped mid-episode state whenever
        an env was returned from a factory or stored into a container."""
        self.particles = move.particles^
        self.physics = move.physics^
        self.config = PhysicsConfig(
            gravity_x=move.config.gravity_x,
            gravity_y=move.config.gravity_y,
            dt=move.config.dt,
            friction=move.config.friction,
            restitution=move.config.restitution,
            baumgarte=move.config.baumgarte,
            slop=move.config.slop,
            velocity_iterations=move.config.velocity_iterations,
            position_iterations=move.config.position_iterations,
        )
        self.prev_shaping = move.prev_shaping
        self.step_count = move.step_count
        self.game_over = move.game_over
        self.last_terminated = move.last_terminated
        self.rng_seed = move.rng_seed
        self.rng_counter = move.rng_counter
        self.wind_idx = move.wind_idx
        self.torque_idx = move.torque_idx
        self.terrain_heights = move.terrain_heights^
        self.edge_collision = move.edge_collision^
        self.cached_state = move.cached_state

        # Transfer renderer ownership
        self._renderer = move._renderer
        self._renderer_initialized = move._renderer_initialized

    # =========================================================================
    # CPU Single-Environment Methods
    # =========================================================================

    def _init_physics_shapes(mut self):
        """Initialize physics shapes. Must be called after creating fresh PhysicsState.
        """
        # Define lander shape as polygon (shape 0)
        var lander_vx = List[Float64]()
        var lander_vy = List[Float64]()
        lander_vx.append(-14.0 / LLConstants.SCALE)
        lander_vy.append(17.0 / LLConstants.SCALE)
        lander_vx.append(-17.0 / LLConstants.SCALE)
        lander_vy.append(0.0 / LLConstants.SCALE)
        lander_vx.append(-17.0 / LLConstants.SCALE)
        lander_vy.append(-10.0 / LLConstants.SCALE)
        lander_vx.append(17.0 / LLConstants.SCALE)
        lander_vy.append(-10.0 / LLConstants.SCALE)
        lander_vx.append(17.0 / LLConstants.SCALE)
        lander_vy.append(0.0 / LLConstants.SCALE)
        lander_vx.append(14.0 / LLConstants.SCALE)
        lander_vy.append(17.0 / LLConstants.SCALE)
        self.physics.define_polygon_shape(0, lander_vx, lander_vy)

        # Define leg shapes (shapes 1 and 2)
        var leg_vx = List[Float64]()
        var leg_vy = List[Float64]()
        leg_vx.append(-LLConstants.LEG_W)
        leg_vy.append(LLConstants.LEG_H)
        leg_vx.append(-LLConstants.LEG_W)
        leg_vy.append(-LLConstants.LEG_H)
        leg_vx.append(LLConstants.LEG_W)
        leg_vy.append(-LLConstants.LEG_H)
        leg_vx.append(LLConstants.LEG_W)
        leg_vy.append(LLConstants.LEG_H)
        self.physics.define_polygon_shape(1, leg_vx, leg_vy)
        self.physics.define_polygon_shape(2, leg_vx, leg_vy)

    def _reset_cpu(mut self):
        """Internal reset for CPU single-env operation."""
        # Generate random values using Philox
        # Use combined_seed formula matching GPU for consistency:
        # combined_seed = seed * 2654435761 + counter * 12345
        # This ensures same seed+counter produces same state on both CPU and GPU
        self.rng_counter += 1
        var combined_seed = (
            UInt64(self.rng_seed) * 2654435761
            + UInt64(self.rng_counter) * 12345
        )
        var rng = PhiloxRandom(seed=combined_seed, offset=0)
        var rand_vals = rng.step_uniform()

        # Generate terrain heights using separate RNG stream (matching GPU)
        var terrain_rng = PhiloxRandom(seed=combined_seed + 1000, offset=0)

        # First pass: generate raw heights
        var raw_heights = Array[Float64, LLConstants.TERRAIN_CHUNKS + 1](
            fill=LLConstants.HELIPAD_Y
        )
        for chunk in range(LLConstants.TERRAIN_CHUNKS + 1):
            var terrain_rand = terrain_rng.step_uniform()
            raw_heights[chunk] = Float64(terrain_rand[0]) * (
                LLConstants.H_UNITS / 2.0
            )

        # Second pass: apply 3-point smoothing
        for chunk in range(LLConstants.TERRAIN_CHUNKS):
            var smooth_height: Float64
            if chunk == 0:
                smooth_height = (
                    raw_heights[0] + raw_heights[0] + raw_heights[1]
                ) / 3.0
            elif chunk == LLConstants.TERRAIN_CHUNKS - 1:
                smooth_height = (
                    raw_heights[chunk - 1]
                    + raw_heights[chunk]
                    + raw_heights[chunk]
                ) / 3.0
            else:
                smooth_height = (
                    raw_heights[chunk - 1]
                    + raw_heights[chunk]
                    + raw_heights[chunk + 1]
                ) / 3.0
            self.terrain_heights[chunk] = Scalar[Self.dtype](smooth_height)

        # Third pass: make helipad area flat
        for chunk in range(
            LLConstants.TERRAIN_CHUNKS // 2 - 2,
            LLConstants.TERRAIN_CHUNKS // 2 + 3,
        ):
            if chunk >= 0 and chunk < LLConstants.TERRAIN_CHUNKS:
                self.terrain_heights[chunk] = Scalar[Self.dtype](
                    LLConstants.HELIPAD_Y
                )

        # Set up edge terrain collision
        var env_heights = List[Scalar[dtype]]()
        for chunk in range(LLConstants.TERRAIN_CHUNKS):
            env_heights.append(
                rebind[Scalar[dtype]](self.terrain_heights[chunk])
            )
        self.edge_collision.set_terrain_from_heights(
            0,
            env_heights,
            x_start=0.0,
            x_spacing=LLConstants.W_UNITS
            / Float64(LLConstants.TERRAIN_CHUNKS - 1),
        )

        # Initial velocity (Gymnasium's ApplyForceToCenter integrated over
        # its in-reset step: v = F dt / m)
        var rand1 = Float64(rand_vals[0])
        var rand2 = Float64(rand_vals[1])
        var init_fx = (rand1 * 2.0 - 1.0) * 1000.0  # INITIAL_RANDOM
        var init_fy = (rand2 * 2.0 - 1.0) * 1000.0
        var init_vx = init_fx * LLConstants.DT / LLConstants.LANDER_MASS
        var init_vy = init_fy * LLConstants.DT / LLConstants.LANDER_MASS

        # Lander, legs and leg joints: the SAME initializer as the GPU reset.
        Self._init_bodies_single_env[1, LLConstants.STATE_SIZE_VAL](
            self.physics.get_state_tensor(),
            0,
            Scalar[dtype](init_vx),
            Scalar[dtype](init_vy),
        )

        # Reset tracking
        self.step_count = 0
        self.game_over = False
        self.last_terminated = False
        self.prev_shaping = self._compute_shaping()

        # Clear particles
        self.particles.clear()

        # Reset wind indices (compile-time eliminated if ENABLE_WIND=False)
        comptime if Self.ENABLE_WIND:
            self.rng_counter += 1
            var wind_rng = PhiloxRandom(
                seed=UInt64(self.rng_seed) + 2000,
                offset=UInt64(self.rng_counter),
            )
            var wind_rand = wind_rng.step_uniform()
            self.wind_idx = Int((Float64(wind_rand[0]) * 2.0 - 1.0) * 9999.0)
            self.torque_idx = Int((Float64(wind_rand[1]) * 2.0 - 1.0) * 9999.0)

        # Update cached state
        self._update_cached_state()

    def _update_cached_state(mut self):
        """Update the cached state from physics state."""
        var x = Scalar[DType.float64](
            self.physics.get_body_x(0, Self.BODY_LANDER)
        )
        var y = Scalar[DType.float64](
            self.physics.get_body_y(0, Self.BODY_LANDER)
        )
        var vx = Scalar[DType.float64](
            self.physics.get_body_vx(0, Self.BODY_LANDER)
        )
        var vy = Scalar[DType.float64](
            self.physics.get_body_vy(0, Self.BODY_LANDER)
        )
        var angle = Float64(self.physics.get_body_angle(0, Self.BODY_LANDER))
        var omega = Scalar[DType.float64](
            self.physics.get_body_omega(0, Self.BODY_LANDER)
        )

        # Normalize using helper functions
        var pos_norm = normalize_position[DType.float64](x, y)
        var vel_norm = normalize_velocity[DType.float64](vx, vy)
        var omega_norm = normalize_angular_velocity[DType.float64](omega)

        var left_leg_y = Float64(self.physics.get_body_y(0, Self.BODY_LEFT_LEG))
        var left_leg_x = Float64(self.physics.get_body_x(0, Self.BODY_LEFT_LEG))
        var left_terrain_y = self._get_terrain_height(left_leg_x)
        var right_leg_y = Float64(
            self.physics.get_body_y(0, Self.BODY_RIGHT_LEG)
        )
        var right_leg_x = Float64(
            self.physics.get_body_x(0, Self.BODY_RIGHT_LEG)
        )
        var right_terrain_y = self._get_terrain_height(right_leg_x)

        self.cached_state.x = Scalar[Self.dtype](pos_norm[0])
        self.cached_state.y = Scalar[Self.dtype](pos_norm[1])
        self.cached_state.vx = Scalar[Self.dtype](vel_norm[0])
        self.cached_state.vy = Scalar[Self.dtype](vel_norm[1])
        self.cached_state.angle = Scalar[Self.dtype](angle)
        self.cached_state.angular_velocity = Scalar[Self.dtype](omega_norm)
        self.cached_state.left_leg_contact = Scalar[Self.dtype](
            1.0
        ) if left_leg_y - LLConstants.LEG_H <= left_terrain_y + 0.01 else Scalar[
            Self.dtype
        ](
            0.0
        )
        self.cached_state.right_leg_contact = Scalar[Self.dtype](
            1.0
        ) if right_leg_y - LLConstants.LEG_H <= right_terrain_y + 0.01 else Scalar[
            Self.dtype
        ](
            0.0
        )

    def _compute_shaping(mut self) -> Scalar[Self.dtype]:
        """Compute the shaping potential for reward calculation."""
        var obs = self.get_observation(0)
        return compute_shaping[Self.dtype](
            obs[0], obs[1], obs[2], obs[3], obs[4], obs[6], obs[7]
        )

    def get_observation(
        mut self, env: Int
    ) -> Array[Scalar[Self.dtype], LLConstants.OBS_DIM_VAL]:
        """Get normalized observation for an environment."""
        # Get main lander body state
        var x = Scalar[DType.float64](
            self.physics.get_body_x(env, Self.BODY_LANDER)
        )
        var y = Scalar[DType.float64](
            self.physics.get_body_y(env, Self.BODY_LANDER)
        )
        var vx = Scalar[DType.float64](
            self.physics.get_body_vx(env, Self.BODY_LANDER)
        )
        var vy = Scalar[DType.float64](
            self.physics.get_body_vy(env, Self.BODY_LANDER)
        )
        var angle = Scalar[DType.float64](
            self.physics.get_body_angle(env, Self.BODY_LANDER)
        )
        var omega = Scalar[DType.float64](
            self.physics.get_body_omega(env, Self.BODY_LANDER)
        )

        # Normalize using helper functions
        var pos_norm = normalize_position[DType.float64](x, y)
        var vel_norm = normalize_velocity[DType.float64](vx, vy)
        var omega_norm = normalize_angular_velocity[DType.float64](omega)

        # Leg contact detection
        var left_contact = Scalar[Self.dtype](0.0)
        var right_contact = Scalar[Self.dtype](0.0)

        # Get leg positions and check contact
        var left_leg_y = Float64(
            self.physics.get_body_y(env, Self.BODY_LEFT_LEG)
        )
        var left_leg_x = Float64(
            self.physics.get_body_x(env, Self.BODY_LEFT_LEG)
        )
        var left_terrain_y = self._get_terrain_height(left_leg_x)

        var right_leg_y = Float64(
            self.physics.get_body_y(env, Self.BODY_RIGHT_LEG)
        )
        var right_leg_x = Float64(
            self.physics.get_body_x(env, Self.BODY_RIGHT_LEG)
        )
        var right_terrain_y = self._get_terrain_height(right_leg_x)

        if left_leg_y - LLConstants.LEG_H <= left_terrain_y + 0.01:
            left_contact = Scalar[Self.dtype](1.0)
        if right_leg_y - LLConstants.LEG_H <= right_terrain_y + 0.01:
            right_contact = Scalar[Self.dtype](1.0)

        var obs = Array[Scalar[Self.dtype], LLConstants.OBS_DIM_VAL](
            fill=Scalar[Self.dtype](0.0)
        )
        obs[0] = Scalar[Self.dtype](pos_norm[0])
        obs[1] = Scalar[Self.dtype](pos_norm[1])
        obs[2] = Scalar[Self.dtype](vel_norm[0])
        obs[3] = Scalar[Self.dtype](vel_norm[1])
        obs[4] = Scalar[Self.dtype](angle)
        obs[5] = Scalar[Self.dtype](omega_norm)
        obs[6] = left_contact
        obs[7] = right_contact
        return obs^

    def _get_terrain_height(self, x: Float64) -> Float64:
        """Get terrain height at given x position."""
        var chunk_width = LLConstants.W_UNITS / Float64(
            LLConstants.TERRAIN_CHUNKS - 1
        )
        var chunk_idx = Int(x / chunk_width)
        if chunk_idx < 0:
            chunk_idx = 0
        if chunk_idx >= LLConstants.TERRAIN_CHUNKS:
            chunk_idx = LLConstants.TERRAIN_CHUNKS - 1
        return Float64(self.terrain_heights[chunk_idx])

    def _update_particles(mut self, dt: Float64):
        """Update particle positions and remove dead particles."""
        var i = 0
        while i < len(self.particles):
            var p = self.particles[i]
            var new_x = Float64(p.x) + Float64(p.vx) * dt
            var new_y = Float64(p.y) + Float64(p.vy) * dt
            var new_vy = Float64(p.vy) + LLConstants.GRAVITY_Y * dt * 0.3
            var new_ttl = Float64(p.ttl) - dt

            if new_ttl <= 0.0:
                _ = self.particles.pop(i)
            else:
                self.particles[i] = Particle[Self.dtype](
                    Scalar[Self.dtype](new_x),
                    Scalar[Self.dtype](new_y),
                    p.vx,
                    Scalar[Self.dtype](new_vy),
                    Scalar[Self.dtype](new_ttl),
                )
                i += 1

    def _spawn_main_engine_particles(
        mut self,
        pos_x: Float64,
        pos_y: Float64,
        tip_x: Float64,
        tip_y: Float64,
        power: Float64,
    ):
        """Spawn flame particles from main engine.

        Args:
            pos_x: X-coordinate of lander center position.
            pos_y: Y-coordinate of lander center position.
            tip_x: X-coordinate of unit vector pointing "up" from lander (sin(angle)).
            tip_y: Y-coordinate of unit vector pointing "up" from lander (cos(angle)).
            power: Engine power (0.0 to 1.0).
        """
        if power <= 0.0:
            return

        # Spawn 2-4 particles per frame when engine is on
        self.rng_counter += 1
        var rng = PhiloxRandom(
            seed=UInt64(self.rng_seed) + 5000, offset=UInt64(self.rng_counter)
        )

        var num_particles = 2 + Int(rng.step_uniform()[0] * 3.0)
        for _ in range(num_particles):
            var rand_vals = rng.step_uniform()

            # Position below the lander (opposite of tip direction)
            var offset_x = (Float64(rand_vals[0]) - 0.5) * 0.3
            var px = pos_x - tip_x * 0.5 + offset_x
            var py = pos_y - tip_y * 0.5  # Below lander

            # Velocity DOWNWARD (opposite of thrust direction = -tip)
            var spread = (Float64(rand_vals[1]) - 0.5) * 2.0
            var vx = -tip_x * 3.0 * power + spread
            var vy = -tip_y * 3.0 * power + (Float64(rand_vals[2]) - 0.5)

            # Short lifetime
            var ttl = 0.1 + Float64(rand_vals[3]) * 0.2

            self.particles.append(
                Particle[Self.dtype](
                    Scalar[Self.dtype](px),
                    Scalar[Self.dtype](py),
                    Scalar[Self.dtype](vx),
                    Scalar[Self.dtype](vy),
                    Scalar[Self.dtype](ttl),
                )
            )

    def _spawn_side_engine_particles(
        mut self,
        pos_x: Float64,
        pos_y: Float64,
        tip_x: Float64,
        tip_y: Float64,
        side_x: Float64,
        side_y: Float64,
        direction: Float64,
        power: Float64,
    ):
        """Spawn flame particles from side engine.

        Args:
            pos_x: X-coordinate of lander center position.
            pos_y: Y-coordinate of lander center position.
            tip_x: X-coordinate of unit vector pointing "up" from lander (sin(angle)).
            tip_y: Y-coordinate of unit vector pointing "up" from lander (cos(angle)).
            side_x: X-coordinate of unit vector pointing "right" from lander (-tip_y).
            side_y: Y-coordinate of unit vector pointing "right" from lander (tip_x).
            direction: -1 for left engine, +1 for right engine.
            power: Engine power (0.0 to 1.0).
        """
        if power <= 0.0:
            return

        # Spawn 1-2 particles per frame when engine is on
        self.rng_counter += 1
        var rng = PhiloxRandom(
            seed=UInt64(self.rng_seed) + 6000, offset=UInt64(self.rng_counter)
        )

        var num_particles = 1 + Int(rng.step_uniform()[0] * 2.0)
        for _ in range(num_particles):
            var rand_vals = rng.step_uniform()

            # Position at side of lander where engine is
            var px = pos_x - side_x * direction * 0.6
            var py = pos_y - side_y * direction * 0.6

            # Velocity: exhaust goes outward from the engine
            var vx = -side_x * direction * 2.0 * power + (
                Float64(rand_vals[0]) - 0.5
            )
            var vy = -side_y * direction * 2.0 * power + (
                Float64(rand_vals[1]) - 0.5
            )

            # Short lifetime
            var ttl = 0.08 + Float64(rand_vals[2]) * 0.15

            self.particles.append(
                Particle[Self.dtype](
                    Scalar[Self.dtype](px),
                    Scalar[Self.dtype](py),
                    Scalar[Self.dtype](vx),
                    Scalar[Self.dtype](vy),
                    Scalar[Self.dtype](ttl),
                )
            )

    # =========================================================================
    # BoxDiscreteActionEnv Trait Methods
    # =========================================================================

    def reset(mut self) -> Self.StateType:
        """Reset the environment and return initial state."""
        self._reset_cpu()
        return self.get_state()

    def step(
        mut self, action: Self.ActionType, verbose: Bool = False
    ) -> Tuple[Self.StateType, Scalar[Self.dtype], Bool]:
        """Take an action and return (next_state, reward, done)."""
        var result = self._step_cpu(action.action_idx)
        return (self.get_state(), result[0], result[1])

    def _step_cpu(mut self, action: Int) -> Tuple[Scalar[Self.dtype], Bool]:
        """Internal CPU step implementation."""
        # Convert action to power values
        var m_power = Float64(0)
        var s_power = Float64(0)
        var direction = Float64(0)

        if action == 2:  # Main engine
            m_power = 1.0
        elif action == 1:  # Left engine
            s_power = 1.0
            direction = -1.0
        elif action == 3:  # Right engine
            s_power = 1.0
            direction = 1.0

        # Apply wind
        self._apply_wind()

        # Apply engine forces
        self._apply_engines(m_power, s_power, direction)

        # Spawn particles for engine flames (cosmetic effect)
        if m_power > 0.0 or s_power > 0.0:
            var pos_x = Float64(self.physics.get_body_x(0, Self.BODY_LANDER))
            var pos_y = Float64(self.physics.get_body_y(0, Self.BODY_LANDER))
            var angle = Float64(
                self.physics.get_body_angle(0, Self.BODY_LANDER)
            )
            var tip_x = sin(angle)
            var tip_y = cos(angle)
            var side_x = -tip_y
            var side_y = tip_x

            if m_power > 0.0:
                self._spawn_main_engine_particles(
                    pos_x, pos_y, tip_x, tip_y, m_power
                )
            if s_power > 0.0:
                self._spawn_side_engine_particles(
                    pos_x,
                    pos_y,
                    tip_x,
                    tip_y,
                    side_x,
                    side_y,
                    direction,
                    s_power,
                )

        # Physics step
        self._step_physics_cpu()

        # Update cached state
        self._update_cached_state()

        # Compute reward and termination
        return self._compute_step_result(m_power, s_power)

    def _apply_wind(mut self):
        """Apply wind and turbulence forces.

        Uses compile-time struct parameters ENABLE_WIND, WIND_POWER, TURBULENCE_POWER.
        When ENABLE_WIND is False, this is a no-op eliminated at compile time.
        """

        comptime if not Self.ENABLE_WIND:
            return

        var obs = self.get_observation(0)
        var left_contact = obs[6] > Scalar[Self.dtype](0.5)
        var right_contact = obs[7] > Scalar[Self.dtype](0.5)
        if left_contact or right_contact:
            return

        var k = 0.01
        var wind_t = Float64(self.wind_idx)
        var wind_mag = (
            tanh(sin(0.02 * wind_t) + sin(pi * k * wind_t)) * Self.WIND_POWER
        )
        self.wind_idx += 1

        var torque_t = Float64(self.torque_idx)
        var torque_mag = (
            tanh(sin(0.02 * torque_t) + sin(pi * k * torque_t))
            * Self.TURBULENCE_POWER
        )
        self.torque_idx += 1

        var vx = Float64(self.physics.get_body_vx(0, Self.BODY_LANDER))
        var vy = Float64(self.physics.get_body_vy(0, Self.BODY_LANDER))
        var omega = Float64(self.physics.get_body_omega(0, Self.BODY_LANDER))

        var dvx = wind_mag * LLConstants.DT / LLConstants.LANDER_MASS
        var domega = torque_mag * LLConstants.DT / LLConstants.LANDER_INERTIA

        self.physics.set_body_velocity(
            0, Self.BODY_LANDER, vx + dvx, vy, omega + domega
        )

    def _apply_engines(
        mut self, m_power: Float64, s_power: Float64, direction: Float64
    ):
        """Apply engine impulses."""
        if m_power == 0.0 and s_power == 0.0:
            return

        var angle = Float64(self.physics.get_body_angle(0, Self.BODY_LANDER))
        var vx = Float64(self.physics.get_body_vx(0, Self.BODY_LANDER))
        var vy = Float64(self.physics.get_body_vy(0, Self.BODY_LANDER))
        var omega = Float64(self.physics.get_body_omega(0, Self.BODY_LANDER))

        var tip_x = sin(angle)
        var tip_y = cos(angle)
        var side_x = -tip_y
        var side_y = tip_x

        # Engine dispersion RNG matches GPU pattern for consistency:
        # GPU uses: seed = env + 12345, offset = step_count
        # CPU (single env = 0) uses: seed = 12345, offset = step_count
        var rng = PhiloxRandom(seed=12345, offset=UInt64(self.step_count))
        var rand_vals = rng.step_uniform()
        var dispersion_x = (
            Float64(rand_vals[0]) * 2.0 - 1.0
        ) / LLConstants.SCALE
        var dispersion_y = (
            Float64(rand_vals[1]) * 2.0 - 1.0
        ) / LLConstants.SCALE

        var dvx = Float64(0)
        var dvy = Float64(0)
        var domega = Float64(0)

        if m_power > 0.0:
            var main_y_offset = 4.0 / LLConstants.SCALE
            var ox = (
                tip_x * (main_y_offset + 2.0 * dispersion_x)
                + side_x * dispersion_y
            )
            var oy = (
                -tip_y * (main_y_offset + 2.0 * dispersion_x)
                - side_y * dispersion_y
            )
            var impulse_x = -ox * LLConstants.MAIN_ENGINE_POWER * m_power
            var impulse_y = -oy * LLConstants.MAIN_ENGINE_POWER * m_power
            dvx += impulse_x / LLConstants.LANDER_MASS
            dvy += impulse_y / LLConstants.LANDER_MASS
            var torque = ox * impulse_y - oy * impulse_x
            domega += torque / LLConstants.LANDER_INERTIA

        if s_power > 0.0:
            var side_away = 12.0 / LLConstants.SCALE
            var ox = tip_x * dispersion_x + side_x * (
                3.0 * dispersion_y + direction * side_away
            )
            var oy = -tip_y * dispersion_x - side_y * (
                3.0 * dispersion_y + direction * side_away
            )
            var impulse_x = -ox * LLConstants.SIDE_ENGINE_POWER * s_power
            var impulse_y = -oy * LLConstants.SIDE_ENGINE_POWER * s_power
            dvx += impulse_x / LLConstants.LANDER_MASS
            dvy += impulse_y / LLConstants.LANDER_MASS
            var side_height = 14.0 / LLConstants.SCALE
            var r_x = ox - tip_x * 17.0 / LLConstants.SCALE
            var r_y = oy + tip_y * side_height
            var torque = r_x * impulse_y - r_y * impulse_x
            domega += torque / LLConstants.LANDER_INERTIA

        self.physics.set_body_velocity(
            0, Self.BODY_LANDER, vx + dvx, vy + dvy, omega + domega
        )

    def _step_physics_cpu(mut self):
        """Execute physics step: detection, then the SAME Box2D island solve
        the GPU kernels run (`_solve_step_single_env`)."""
        var bodies = self.physics.get_bodies_tensor()
        var shapes = self.physics.get_shapes_tensor()
        var contacts = self.physics.get_contacts_tensor()
        var contact_counts = self.physics.get_contact_counts_tensor()
        var states = self.physics.get_state_tensor()

        self.edge_collision.detect[
            1,
            LLConstants.NUM_BODIES,
            LLConstants.NUM_SHAPES,
            LLConstants.MAX_CONTACTS,
        ](bodies, shapes, contacts, contact_counts)

        Self._solve_step_single_env[1, LLConstants.STATE_SIZE_VAL](
            0,
            states,
            contacts,
            Int(contact_counts[0]),
            Int(states[0, LLConstants.JOINT_COUNT_OFFSET]),
            Scalar[dtype](self.config.gravity_x),
            Scalar[dtype](self.config.gravity_y),
            Scalar[dtype](self.config.dt),
        )

        # Clear forces
        var forces = self.physics.get_forces_tensor()
        for body in range(LLConstants.NUM_BODIES):
            forces[0, body, 0] = Scalar[dtype](0)
            forces[0, body, 1] = Scalar[dtype](0)
            forces[0, body, 2] = Scalar[dtype](0)

    def _has_lander_body_contact(mut self) -> Bool:
        """Check if the lander body (not legs) is in contact with terrain.

        Uses the collision detection system results to determine if the main
        lander body (BODY_LANDER = 0) has any contacts. Leg contacts (bodies 1, 2)
        are excluded since those are expected during landing.

        Returns:
            True if lander body is touching terrain (crash condition).
        """
        var contacts = self.physics.get_contacts_tensor()
        var contact_counts = self.physics.get_contact_counts_tensor()
        var n_contacts = Int(contact_counts[0])

        for i in range(n_contacts):
            var body_a = Int(contacts[0, i, CONTACT_BODY_A])
            # BODY_LANDER = 0, legs are 1 and 2
            # Contact with terrain means body_b = 0 (worldbody)
            if body_a == Self.BODY_LANDER:
                return True
        return False

    def _compute_step_result(
        mut self, m_power: Float64, s_power: Float64
    ) -> Tuple[Scalar[Self.dtype], Bool]:
        """Compute reward and termination."""
        self.step_count += 1

        var obs = self.get_observation(0)
        var x_norm = obs[0]

        var new_shaping = self._compute_shaping()
        var reward = new_shaping - self.prev_shaping
        self.prev_shaping = new_shaping

        reward = reward - Scalar[Self.dtype](
            m_power * LLConstants.MAIN_ENGINE_FUEL_COST
        )
        reward = reward - Scalar[Self.dtype](
            s_power * LLConstants.SIDE_ENGINE_FUEL_COST
        )

        # Crash: lander body touches ground (physics-based crash detection).
        var lander_contact = self._has_lander_body_contact()
        if lander_contact:
            self.game_over = True

        # Single-source terminal predicate + terminal reward overrides
        # (shared with BOTH GPU step kernels — see helpers.mojo).
        var asleep = island_is_asleep(
            rebind[Scalar[dtype]](
                self.physics.get_state_tensor()[
                    0, LLConstants.METADATA_OFFSET + LLConstants.META_SLEEP_TIME
                ]
            )
        )
        var tr = lunar_terminal_and_reward[Self.dtype](
            Scalar[Self.dtype](x_norm), lander_contact, asleep, reward
        )
        reward = tr[0]
        var terminated = tr[1]

        # Natural termination (crash/OOB/landed) recorded BEFORE folding in
        # the time-limit cap, so `was_terminated()` distinguishes the two —
        # the GPU kernels already report them separately via terminated_out.
        self.last_terminated = terminated

        if self.step_count >= 1000:
            terminated = True

        return (reward, terminated)

    def get_state(mut self) -> Self.StateType:
        """Return current state representation (from cache)."""
        return self.cached_state

    def was_terminated(self) -> Bool:
        """True iff the previous step ended by natural termination
        (crash/out-of-bounds/landed), False on the 1000-step truncation —
        mirrors the GPU kernels' separate `terminated_out`. Without this
        override the base default (always False) made drivers bootstrap
        through real crashes on the CPU path."""
        return self.last_terminated

    def get_obs_list(self) -> List[Scalar[Self.dtype]]:
        """Return current continuous observation as a list."""
        # Reads the cache directly rather than through `get_state`: the `Env`
        # trait pins that one as `mut self` while `BoxDiscreteActionEnv` pins
        # this one as `self`, so routing through it no longer type-checks.
        # Same value either way — `get_state` only returns `cached_state`.
        return self.cached_state.to_list()

    def reset_obs_list(mut self) -> List[Scalar[Self.dtype]]:
        """Reset environment and return initial continuous observation."""
        var state = self.reset()
        return state.to_list()

    def obs_dim(self) -> Int:
        """Return the dimension of the observation vector."""
        return LLConstants.OBS_DIM_VAL

    def action_from_index(self, action_idx: Int) -> Self.ActionType:
        """Create an action from an integer index."""
        return LunarLanderAction(action_idx=action_idx)

    def num_actions(self) -> Int:
        """Return the number of discrete actions available."""
        return LLConstants.NUM_ACTIONS_VAL

    def step_obs(
        mut self, action: Int
    ) -> Tuple[List[Scalar[Self.dtype]], Scalar[Self.dtype], Bool]:
        """Take discrete action and return (continuous_obs, reward, done)."""
        var result = self._step_cpu(action)
        var obs = self.get_obs_list()
        return (obs^, result[0], result[1])

    # =========================================================================
    # BoxContinuousActionEnv Trait Methods
    # =========================================================================

    def action_dim(self) -> Int:
        """Return action dimension (2 for LunarLander continuous).

        Action space:
        - action[0]: main engine throttle (0.0 to 1.0, 0 = off, 1 = full power)
        - action[1]: side engine control (-1.0 to 1.0, negative = left, positive = right)
        """
        return LLConstants.ACTION_DIM_VAL

    def action_low(self) -> Scalar[Self.dtype]:
        """Return lower bound for action values (-1.0 for side engine)."""
        return Scalar[Self.dtype](-1.0)

    def action_high(self) -> Scalar[Self.dtype]:
        """Return upper bound for action values (1.0)."""
        return Scalar[Self.dtype](1.0)

    def step_continuous[
        DTYPE_SC: DType
    ](mut self, action: Scalar[DTYPE_SC]) -> Tuple[
        List[Scalar[DTYPE_SC]], Scalar[DTYPE_SC], Bool
    ]:
        """Take 1D continuous action (main engine only) and return (obs, reward, done).

        For single-dimensional control, interprets action as main engine throttle.
        Gymnasium: action <= 0 → off, action > 0 → power in [0.5, 1.0].
        """
        var raw = Float64(action)
        var m_power = Float64(0.0)
        if raw > 0.0:
            if raw > 1.0:
                raw = 1.0
            m_power = (raw + 1.0) * 0.5
        var result = self._step_cpu_continuous(m_power, 0.0, 0.0)
        var obs_self = self.get_obs_list()
        var obs = List[Scalar[DTYPE_SC]](capacity=len(obs_self))
        for i in range(len(obs_self)):
            obs.append(Scalar[DTYPE_SC](obs_self[i]))
        return (obs^, Scalar[DTYPE_SC](result[0]), result[1])

    def step_continuous_vec[
        DTYPE_VEC: DType
    ](
        mut self, action: List[Scalar[DTYPE_VEC]], verbose: Bool = False
    ) -> Tuple[List[Scalar[DTYPE_VEC]], Scalar[DTYPE_VEC], Bool]:
        """Take 2D continuous action and return (obs, reward, done).

        Action space (matching Gymnasium):
        - action[0]: main engine. <= 0 → off; (0,1] → power in [0.5, 1.0]
        - action[1]: side engine. |val| <= 0.5 → off; |val| in (0.5,1] → power = |val|
        """
        var m_power = Float64(0.0)
        var s_power = Float64(0.0)
        var direction = Float64(0.0)

        if len(action) > 0:
            var raw = Float64(action[0])
            if raw < -1.0:
                raw = -1.0
            if raw > 1.0:
                raw = 1.0
            if raw > 0.0:
                m_power = (raw + 1.0) * 0.5

        if len(action) > 1:
            var side_action = Float64(action[1])
            if side_action < -1.0:
                side_action = -1.0
            if side_action > 1.0:
                side_action = 1.0

            if side_action < -0.5:
                direction = -1.0
                s_power = -side_action
            elif side_action > 0.5:
                direction = 1.0
                s_power = side_action

        var result = self._step_cpu_continuous(m_power, s_power, direction)

        # Convert observation to requested dtype
        var obs_internal = self.get_obs_list()
        var obs = List[Scalar[DTYPE_VEC]](capacity=LLConstants.OBS_DIM_VAL)
        for i in range(len(obs_internal)):
            obs.append(Scalar[DTYPE_VEC](obs_internal[i]))
        return (obs^, Scalar[DTYPE_VEC](result[0]), result[1])

    def _step_cpu_continuous(
        mut self, m_power: Float64, s_power: Float64, direction: Float64
    ) -> Tuple[Scalar[Self.dtype], Bool]:
        """Internal CPU step implementation for continuous actions.

        Args:
            m_power: Main engine power (0.0 to 1.0).
            s_power: Side engine power (0.0 to 1.0).
            direction: Side engine direction (-1.0 = left, 1.0 = right, 0.0 = off).
        """
        # Apply wind
        self._apply_wind()

        # Apply engine forces with continuous power
        self._apply_engines(m_power, s_power, direction)

        # Spawn particles for engine flames (cosmetic effect)
        if m_power > 0.0 or s_power > 0.0:
            var pos_x = Float64(self.physics.get_body_x(0, Self.BODY_LANDER))
            var pos_y = Float64(self.physics.get_body_y(0, Self.BODY_LANDER))
            var angle = Float64(
                self.physics.get_body_angle(0, Self.BODY_LANDER)
            )
            var tip_x = sin(angle)
            var tip_y = cos(angle)
            var side_x = -tip_y
            var side_y = tip_x

            if m_power > 0.0:
                self._spawn_main_engine_particles(
                    pos_x, pos_y, tip_x, tip_y, m_power
                )
            if s_power > 0.0:
                self._spawn_side_engine_particles(
                    pos_x,
                    pos_y,
                    tip_x,
                    tip_y,
                    side_x,
                    side_y,
                    direction,
                    s_power,
                )

        # Physics step
        self._step_physics_cpu()

        # Update cached state
        self._update_cached_state()

        # Compute reward and termination
        return self._compute_step_result(m_power, s_power)

    def render(mut self, mut renderer: Renderer2D):
        """Render the environment (Env trait method)."""
        # Render env 0 for single-env CPU mode
        self.render(0, renderer)

    def close(mut self):
        """Clean up resources."""
        self.particles.clear()
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

    # =========================================================================
    # GPU Kernels
    # =========================================================================

    @staticmethod
    def step_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        OBS_DIM: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[dtype],
        actions_buf: DeviceBuffer[dtype],
        mut rewards_buf: DeviceBuffer[dtype],
        mut dones_buf: DeviceBuffer[dtype],
        mut terminated_buf: DeviceBuffer[dtype],
        mut obs_buf: DeviceBuffer[dtype],
        rng_seed: UInt64 = 0,
        workspace_ptr: Optional[
            Pointer[Scalar[dtype], MutAnyOrigin]
        ] = None,
        rng_counter_ptr: Optional[
            Pointer[Scalar[DType.uint64], MutAnyOrigin]
        ] = None,
    ) raises:
        """Optimized GPU step kernel with fused obs extraction.

        Uses 2-kernel pipeline and writes observations directly to obs_buf,
        eliminating the need for a separate extract_obs kernel.

        Workspace layout (pre-allocated via init_step_workspace_gpu):
          [0..SHAPES_SIZE): shapes (shared, initialized once)
          [SHAPES_SIZE..+BATCH*CONTACTS_PER_ENV): contacts
          [..+BATCH): contact_counts
          [..+BATCH): edge_counts
          [..+BATCH): joint_counts
        """
        comptime SHAPES_SIZE = LLConstants.NUM_SHAPES * SHAPE_MAX_SIZE
        comptime CONTACTS_PER_ENV = LLConstants.MAX_CONTACTS * CONTACT_DATA_SIZE

        # Carve pre-allocated workspace into sub-buffers (no GPU allocation)
        var ws = workspace_ptr.value()
        var shapes_buf = DeviceBuffer[dtype](ctx, ws, SHAPES_SIZE, owning=False)
        ws = ws.unsafe_offset(SHAPES_SIZE)
        var contacts_buf = DeviceBuffer[dtype](
            ctx, ws, BATCH_SIZE * CONTACTS_PER_ENV, owning=False
        )
        ws = ws.unsafe_offset(BATCH_SIZE * CONTACTS_PER_ENV)
        var contact_counts_buf = DeviceBuffer[dtype](
            ctx, ws, BATCH_SIZE, owning=False
        )
        ws = ws.unsafe_offset(BATCH_SIZE)
        var edge_counts_buf = DeviceBuffer[dtype](
            ctx, ws, BATCH_SIZE, owning=False
        )
        ws = ws.unsafe_offset(BATCH_SIZE)
        var joint_counts_buf = DeviceBuffer[dtype](
            ctx, ws, BATCH_SIZE, owning=False
        )

        # Kernel 1: Fused setup (zero + extract + apply_forces)
        LunarLander[Self.dtype]._setup_fused_gpu[BATCH_SIZE](
            ctx,
            states_buf,
            actions_buf,
            edge_counts_buf,
            joint_counts_buf,
            contact_counts_buf,
        )

        # Kernel 2: Fused physics + finalize + extract_obs
        LunarLander[Self.dtype]._physics_finalize_obs_fused_gpu[
            BATCH_SIZE, OBS_DIM
        ](
            ctx,
            states_buf,
            shapes_buf,
            edge_counts_buf,
            joint_counts_buf,
            contacts_buf,
            contact_counts_buf,
            actions_buf,
            rewards_buf,
            dones_buf,
            terminated_buf,
            obs_buf,
            Scalar[dtype](LLConstants.GRAVITY_X),
            Scalar[dtype](LLConstants.GRAVITY_Y),
            Scalar[dtype](LLConstants.DT),
        )

    @staticmethod
    def step_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        OBS_DIM: Int,
        ACTION_DIM: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[dtype],
        actions_buf: DeviceBuffer[dtype],
        mut rewards_buf: DeviceBuffer[dtype],
        mut dones_buf: DeviceBuffer[dtype],
        mut terminated_buf: DeviceBuffer[dtype],
        mut obs_buf: DeviceBuffer[dtype],
        rng_seed: UInt64 = 0,
        curriculum_values: List[Scalar[dtype]] = [],
        workspace_ptr: Optional[
            Pointer[Scalar[dtype], MutAnyOrigin]
        ] = None,
        rng_counter_ptr: Optional[
            Pointer[Scalar[DType.uint64], MutAnyOrigin]
        ] = None,
    ) raises:
        """GPU step kernel for continuous actions (GPUContinuousEnv trait).

        Actions buffer layout: [BATCH_SIZE, ACTION_DIM] where:
        - action[0]: main engine throttle (policy outputs [-1, 1], remapped to [0, 1])
        - action[1]: side engine control (-1.0 to 1.0)

        Workspace layout: same as discrete overload (see above).
        """
        comptime SHAPES_SIZE = LLConstants.NUM_SHAPES * SHAPE_MAX_SIZE
        comptime CONTACTS_PER_ENV = LLConstants.MAX_CONTACTS * CONTACT_DATA_SIZE

        # Carve pre-allocated workspace into sub-buffers (no GPU allocation)
        var ws = workspace_ptr.value()
        var shapes_buf = DeviceBuffer[dtype](ctx, ws, SHAPES_SIZE, owning=False)
        ws = ws.unsafe_offset(SHAPES_SIZE)
        var contacts_buf = DeviceBuffer[dtype](
            ctx, ws, BATCH_SIZE * CONTACTS_PER_ENV, owning=False
        )
        ws = ws.unsafe_offset(BATCH_SIZE * CONTACTS_PER_ENV)
        var contact_counts_buf = DeviceBuffer[dtype](
            ctx, ws, BATCH_SIZE, owning=False
        )
        ws = ws.unsafe_offset(BATCH_SIZE)
        var edge_counts_buf = DeviceBuffer[dtype](
            ctx, ws, BATCH_SIZE, owning=False
        )
        ws = ws.unsafe_offset(BATCH_SIZE)
        var joint_counts_buf = DeviceBuffer[dtype](
            ctx, ws, BATCH_SIZE, owning=False
        )

        # Kernel 1: Fused setup for continuous actions
        LunarLander[Self.dtype]._setup_fused_gpu_continuous[
            BATCH_SIZE, ACTION_DIM
        ](
            ctx,
            states_buf,
            actions_buf,
            edge_counts_buf,
            joint_counts_buf,
            contact_counts_buf,
        )

        # Kernel 2: Fused physics + finalize + extract_obs (with continuous actions)
        LunarLander[Self.dtype]._physics_finalize_obs_fused_gpu_continuous[
            BATCH_SIZE, OBS_DIM, ACTION_DIM
        ](
            ctx,
            states_buf,
            shapes_buf,
            edge_counts_buf,
            joint_counts_buf,
            contacts_buf,
            contact_counts_buf,
            actions_buf,
            rewards_buf,
            dones_buf,
            terminated_buf,
            obs_buf,
            Scalar[dtype](LLConstants.GRAVITY_X),
            Scalar[dtype](LLConstants.GRAVITY_Y),
            Scalar[dtype](LLConstants.DT),
        )

    @staticmethod
    def reset_kernel_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[dtype],
        rng_seed: UInt64 = 0,
    ) raises:
        """GPU reset kernel.

        Args:
            ctx: Device context for GPU operations.
            states_buf: Buffer to store environment states.
            rng_seed: Random seed for terrain generation. Use different values
                     across calls to get varied terrains. If 0, uses a default
                     seed that still varies per environment.
        """
        # Create 2D LayoutTensor from buffer
        var states = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE)
        ](states_buf)

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        @always_inline
        def reset_wrapper(
            states: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
            ],
            seed: Scalar[dtype],
        ):
            var i = Int(block_dim.x * block_idx.x + thread_idx.x)
            if i >= BATCH_SIZE:
                return
            # Combine seed with env index using prime multiplier for good distribution
            # seed * 2654435761 + (i+1) * 12345 ensures different terrains across:
            # - Different environments in same batch (via (i+1) * 12345)
            # - Different reset calls (via seed * large_prime)
            # Using (i+1) instead of i aligns with CPU's increment-before-use pattern
            # so GPU env 0 matches CPU reset #1, GPU env 1 matches CPU reset #2, etc.
            var combined_seed = Int(seed) * 2654435761 + (i + 1) * 12345
            LunarLander[Self.dtype]._reset_env_gpu[BATCH_SIZE, STATE_SIZE](
                states, i, combined_seed
            )

        ctx.enqueue_function[reset_wrapper](
            states,
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
        mut states_buf: DeviceBuffer[dtype],
        mut dones_buf: DeviceBuffer[dtype],
        rng_seed: UInt64,
        workspace_ptr: Optional[
            Pointer[Scalar[dtype], MutAnyOrigin]
        ] = None,
        rng_counter_ptr: Optional[
            Pointer[Scalar[DType.uint64], MutAnyOrigin]
        ] = None,
    ) raises:
        """GPU selective reset kernel - resets only done environments.

        Args:
            ctx: Device context for GPU operations.
            states_buf: Buffer containing environment states.
            dones_buf: Buffer indicating which environments are done.
            rng_seed: Random seed for terrain generation. Should be different
                     each call (e.g., current training step) to get varied
                     terrains on reset.
            workspace_ptr: Optional workspace pointer (unused for LunarLander).
            rng_counter_ptr: Optional GPU counter pointer. When non-null, reads
                     seed from GPU memory instead of rng_seed parameter.
        """
        var states = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE)
        ](states_buf)

        var dones = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE)
        ](dones_buf)

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        if Bool(rng_counter_ptr):
            var counter_t = LayoutTensor[
                DType.uint64, Layout.row_major(1), MutAnyOrigin
            ](rng_counter_ptr.value())

            @always_inline
            def selective_reset_counter_wrapper(
                states: LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH_SIZE, STATE_SIZE),
                    MutAnyOrigin,
                ],
                dones: LayoutTensor[
                    dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
                ],
                counter: LayoutTensor[
                    DType.uint64, Layout.row_major(1), MutAnyOrigin
                ],
            ):
                var i = Int(block_dim.x * block_idx.x + thread_idx.x)
                if i >= BATCH_SIZE:
                    return
                if rebind[Scalar[dtype]](dones[i]) > Scalar[dtype](0.5):
                    var combined_seed = (
                        Int(rebind[Scalar[DType.uint64]](counter[0]))
                        * 2654435761
                        + (i + 1) * 12345
                    )
                    LunarLander[Self.dtype]._reset_env_gpu[
                        BATCH_SIZE, STATE_SIZE
                    ](states, i, combined_seed)

            ctx.enqueue_function[selective_reset_counter_wrapper](
                states,
                dones,
                counter_t,
                grid_dim=(BLOCKS,),
                block_dim=(TPB,),
            )
        else:

            @always_inline
            def selective_reset_wrapper(
                states: LayoutTensor[
                    dtype,
                    Layout.row_major(BATCH_SIZE, STATE_SIZE),
                    MutAnyOrigin,
                ],
                dones: LayoutTensor[
                    dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
                ],
                seed: Scalar[dtype],
            ):
                var i = Int(block_dim.x * block_idx.x + thread_idx.x)
                if i >= BATCH_SIZE:
                    return
                if rebind[Scalar[dtype]](dones[i]) > Scalar[dtype](0.5):
                    var combined_seed = Int(seed) * 2654435761 + (i + 1) * 12345
                    LunarLander[Self.dtype]._reset_env_gpu[
                        BATCH_SIZE, STATE_SIZE
                    ](states, i, combined_seed)

            ctx.enqueue_function[selective_reset_wrapper](
                states,
                dones,
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
        var states = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE_VAL)
        ](states_buf)
        var obs = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, OBS_DIM_VAL)
        ](obs_buf)

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        @always_inline
        def extract_obs(
            states: LayoutTensor[
                dtype,
                Layout.row_major(BATCH_SIZE, STATE_SIZE_VAL),
                ImmutAnyOrigin,
            ],
            obs: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE, OBS_DIM_VAL), MutAnyOrigin
            ],
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
        """Initialize pre-allocated workspace: shapes buffer at offset 0."""
        comptime SHAPES_SIZE = LLConstants.NUM_SHAPES * SHAPE_MAX_SIZE
        var shapes_buf = DeviceBuffer[dtype](
            ctx,
            workspace_buf.unsafe_ptr(),
            SHAPES_SIZE,
            owning=False,
        )
        LunarLander[Self.dtype]._init_shapes_gpu(ctx, shapes_buf)

    @staticmethod
    def update_curriculum_gpu(
        ctx: DeviceContext,
        mut workspace_buf: DeviceBuffer[dtype],
        curriculum_values: List[Scalar[dtype]],
    ) raises:
        """No-op: LunarLander doesn't use curriculum."""
        pass

    # =========================================================================
    # Helper Functions - GPU
    # =========================================================================

    @always_inline
    # =========================================================================
    # Shared physics (CPU step, both GPU kernels, both resets): ONE copy
    # =========================================================================

    @always_inline
    @staticmethod
    def _init_bodies_single_env[
        BATCH: Int,
        STATE_SIZE: Int,
    ](
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
        init_vx: Scalar[dtype],
        init_vy: Scalar[dtype],
    ):
        """Lander + legs + leg joints as Gymnasium builds them (lunar_lander.py
        reset): the leg i (i = -1 for body 1 = `legs[0]`, +1 for body 2) is
        pinned by a revolute joint from the lander's origin to the point
        (i * LEG_AWAY, LEG_DOWN) of the leg, reference angle 0, limits
        [0.4, 0.9] (i = -1) / [-0.9, -0.4] (i = +1), motor 0.3 * i rad/s at
        most LEG_MOTOR_TORQUE. The legs start pinned at the limit their motor
        pushes against (joint angle -0.4 * i); Gymnasium creates them at
        angle 0.05 * i and lets its first step (inside reset) snap them, to
        ~+-0.49. Velocities: the lander's initial random push as v = F dt / m,
        carried by the legs too. Also clears forces and the sleep clock."""
        var lx = Scalar[dtype](LLConstants.HELIPAD_X)
        var ly = Scalar[dtype](LLConstants.H_UNITS)
        var lander_off = LLConstants.BODIES_OFFSET
        states[env, lander_off + IDX_X] = lx
        states[env, lander_off + IDX_Y] = ly
        states[env, lander_off + IDX_ANGLE] = Scalar[dtype](0)
        states[env, lander_off + IDX_VX] = init_vx
        states[env, lander_off + IDX_VY] = init_vy
        states[env, lander_off + IDX_OMEGA] = Scalar[dtype](0)
        states[env, lander_off + IDX_INV_MASS] = Scalar[dtype](
            1.0 / LLConstants.LANDER_MASS
        )
        states[env, lander_off + IDX_INV_INERTIA] = Scalar[dtype](
            1.0 / LLConstants.LANDER_INERTIA
        )
        states[env, lander_off + IDX_SHAPE] = Scalar[dtype](0)

        states[env, LLConstants.JOINT_COUNT_OFFSET] = Scalar[dtype](2)
        for k in range(2):
            var i = Scalar[dtype](-1.0) if k == 0 else Scalar[dtype](1.0)
            var theta = Scalar[dtype](-0.4) * i
            var bx = i * Scalar[dtype](LLConstants.LEG_AWAY)
            var by = Scalar[dtype](LLConstants.LEG_DOWN)
            var c = cos(theta)
            var sn = sin(theta)
            var leg_off = LLConstants.BODIES_OFFSET + (k + 1) * BODY_STATE_SIZE
            # Leg centre such that R(theta) * (bx, by) lands on the pivot.
            states[env, leg_off + IDX_X] = lx - (c * bx - sn * by)
            states[env, leg_off + IDX_Y] = ly - (sn * bx + c * by)
            states[env, leg_off + IDX_ANGLE] = theta
            states[env, leg_off + IDX_VX] = init_vx
            states[env, leg_off + IDX_VY] = init_vy
            states[env, leg_off + IDX_OMEGA] = Scalar[dtype](0)
            states[env, leg_off + IDX_INV_MASS] = Scalar[dtype](
                1.0 / LLConstants.LEG_MASS
            )
            states[env, leg_off + IDX_INV_INERTIA] = Scalar[dtype](
                1.0 / LLConstants.LEG_INERTIA
            )
            states[env, leg_off + IDX_SHAPE] = Scalar[dtype](k + 1)

            RevoluteJointSolver.write_joint[BATCH, STATE_SIZE](
                states,
                env,
                LLConstants.JOINTS_OFFSET + k * JOINT_DATA_SIZE,
                body_a=LLConstants.BODY_LANDER,
                body_b=k + 1,
                anchor_ax=Scalar[dtype](0),
                anchor_ay=Scalar[dtype](0),
                anchor_bx=bx,
                anchor_by=by,
                reference_angle=Scalar[dtype](0),
                lower_limit=Scalar[dtype](0.4) if k == 0 else Scalar[dtype](
                    -0.9
                ),
                upper_limit=Scalar[dtype](0.9) if k == 0 else Scalar[dtype](
                    -0.4
                ),
                enable_limit=True,
                enable_motor=True,
                motor_speed=Scalar[dtype](LLConstants.LEG_MOTOR_SPEED) * i,
                max_motor_torque=Scalar[dtype](LLConstants.LEG_MOTOR_TORQUE),
            )

        for body in range(LLConstants.NUM_BODIES):
            var force_off = LLConstants.FORCES_OFFSET + body * 3
            states[env, force_off + 0] = Scalar[dtype](0)
            states[env, force_off + 1] = Scalar[dtype](0)
            states[env, force_off + 2] = Scalar[dtype](0)
        states[
            env, LLConstants.METADATA_OFFSET + LLConstants.META_SLEEP_TIME
        ] = Scalar[dtype](0)

    @always_inline
    @staticmethod
    def _solve_step_single_env[
        BATCH: Int,
        STATE_SIZE: Int,
    ](
        env: Int,
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        contacts: LayoutTensor[
            dtype,
            Layout.row_major(BATCH, LLConstants.MAX_CONTACTS, CONTACT_DATA_SIZE),
            MutAnyOrigin,
        ],
        n_contacts: Int,
        n_joints: Int,
        gravity_x: Scalar[dtype],
        gravity_y: Scalar[dtype],
        dt: Scalar[dtype],
    ):
        """b2Island::Solve for this env, after collision detection: integrate
        velocities, init + warm start contacts and joints, velocity
        iterations (joints, then contacts), integrate positions, position
        iterations (contacts, then joints; stop once both are within slop),
        then advance the island's sleep clock."""
        comptime NB = LLConstants.NUM_BODIES
        comptime MC = LLConstants.MAX_CONTACTS
        comptime MJ = LLConstants.MAX_JOINTS
        comptime BO = LLConstants.BODIES_OFFSET
        comptime JO = LLConstants.JOINTS_OFFSET
        SemiImplicitEuler.integrate_velocities_single_env[
            BATCH, NB, STATE_SIZE, BO, LLConstants.FORCES_OFFSET
        ](env, states, gravity_x, gravity_y, dt)
        ImpulseSolver.init_velocity_single_env[BATCH, NB, MC, STATE_SIZE, BO](
            env,
            states,
            contacts,
            n_contacts,
            Scalar[dtype](LLConstants.RESTITUTION),
        )
        RevoluteJointSolver.init_velocity_single_env[
            BATCH, NB, MJ, STATE_SIZE, BO, JO
        ](env, states, n_joints)
        for _ in range(LLConstants.VELOCITY_ITERATIONS):
            RevoluteJointSolver.solve_velocity_single_env[
                BATCH, NB, MJ, STATE_SIZE, BO, JO
            ](env, states, n_joints, dt)
            ImpulseSolver.solve_velocity_single_env[
                BATCH, NB, MC, STATE_SIZE, BO
            ](env, states, contacts, n_contacts, Scalar[dtype](LLConstants.FRICTION))
        SemiImplicitEuler.integrate_positions_single_env[
            BATCH, NB, STATE_SIZE, BO
        ](env, states, dt)
        for _ in range(LLConstants.POSITION_ITERATIONS):
            var contacts_ok = ImpulseSolver.solve_position_single_env[
                BATCH, NB, MC, STATE_SIZE, BO
            ](env, states, contacts, n_contacts)
            var joints_ok = RevoluteJointSolver.solve_position_single_env[
                BATCH, NB, MJ, STATE_SIZE, BO, JO
            ](env, states, n_joints)
            if contacts_ok and joints_ok:
                break
        comptime SLEEP = LLConstants.METADATA_OFFSET + LLConstants.META_SLEEP_TIME
        states[env, SLEEP] = island_sleep_time_single_env[
            BATCH, NB, STATE_SIZE, BO
        ](env, states, rebind[Scalar[dtype]](states[env, SLEEP]), dt)

    @staticmethod
    def _reset_env_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
        seed: Int,
    ):
        """Reset a single environment (GPU version)."""
        # Use Philox RNG (GPU-compatible counter-based RNG)
        # Note: seed is already combined_seed from wrapper (= base_seed * 2654435761 + env * 12345)
        # Use separate RNG streams for velocity and terrain to match CPU exactly
        var velocity_rng = PhiloxRandom(seed=UInt64(seed), offset=0)
        var velocity_rand = velocity_rng.step_uniform()

        # Terrain uses separate RNG stream (seed + 1000) matching CPU
        var terrain_rng = PhiloxRandom(seed=UInt64(seed) + 1000, offset=0)

        # Generate terrain (matching CPU version with smoothing)
        var n_edges = LLConstants.TERRAIN_CHUNKS - 1
        var n_chunks = LLConstants.TERRAIN_CHUNKS
        states[env, LLConstants.EDGE_COUNT_OFFSET] = Scalar[dtype](n_edges)

        var x_spacing: states.element_type = Scalar[dtype](
            LLConstants.W_UNITS
        ) / Scalar[dtype](LLConstants.TERRAIN_CHUNKS - 1)

        # Step 1: Generate raw heights (matching CPU: random * H_UNITS/2)
        # Use a local array to store raw heights before smoothing
        var raw_heights = Array[Scalar[dtype], 12](
            fill=Scalar[dtype](0.0)
        )
        var h_units_half = Scalar[dtype](LLConstants.H_UNITS / 2.0)
        for chunk in range(n_chunks + 1):
            var rand_vals = terrain_rng.step_uniform()
            raw_heights[chunk] = rand_vals[0] * h_units_half

        # Step 2: Apply 3-point smoothing (matching CPU)
        var smoothed_heights = Array[Scalar[dtype], 11](
            fill=Scalar[dtype](LLConstants.HELIPAD_Y)
        )
        for chunk in range(n_chunks):
            var smooth_height: Scalar[dtype]
            if chunk == 0:
                smooth_height = (
                    raw_heights[0] + raw_heights[0] + raw_heights[1]
                ) / Scalar[dtype](3.0)
            elif chunk == n_chunks - 1:
                smooth_height = (
                    raw_heights[chunk - 1]
                    + raw_heights[chunk]
                    + raw_heights[chunk]
                ) / Scalar[dtype](3.0)
            else:
                smooth_height = (
                    raw_heights[chunk - 1]
                    + raw_heights[chunk]
                    + raw_heights[chunk + 1]
                ) / Scalar[dtype](3.0)
            smoothed_heights[chunk] = smooth_height

        # Step 3: Flatten helipad area (matching CPU)
        for chunk in range(
            LLConstants.TERRAIN_CHUNKS // 2 - 2,
            LLConstants.TERRAIN_CHUNKS // 2 + 3,
        ):
            if chunk >= 0 and chunk < n_chunks:
                smoothed_heights[chunk] = Scalar[dtype](LLConstants.HELIPAD_Y)

        # Step 4: Build edges from adjacent chunk heights
        for edge in range(n_edges):
            var x0: states.element_type = Scalar[dtype](edge) * x_spacing
            var x1: states.element_type = Scalar[dtype](edge + 1) * x_spacing
            var y0 = smoothed_heights[edge]
            var y1 = smoothed_heights[edge + 1]

            # Compute edge normal (pointing up)
            var dx = x1 - x0
            var dy = y1 - y0
            var length = sqrt(dx * dx + dy * dy)
            var nx = -dy / length
            var ny = dx / length
            if ny < 0:
                nx = -nx
                ny = -ny

            var edge_off = LLConstants.EDGES_OFFSET + edge * 6
            states[env, edge_off + 0] = x0
            states[env, edge_off + 1] = y0
            states[env, edge_off + 2] = x1
            states[env, edge_off + 3] = y1
            states[env, edge_off + 4] = nx
            states[env, edge_off + 5] = ny

        # Initialize lander
        # Initial velocity matching CPU: (rand * 2 - 1) * INITIAL_RANDOM * DT / LANDER_MASS
        # = (rand * 2 - 1) * INITIAL_RANDOM (1000) * DT / LANDER_MASS
        # Use velocity_rand[0] and [1] to match CPU exactly
        var init_random_scale = Scalar[dtype](
            1000.0 * LLConstants.DT / LLConstants.LANDER_MASS
        )
        var init_vx: states.element_type = (
            velocity_rand[0] * Scalar[dtype](2.0) - Scalar[dtype](1.0)
        ) * init_random_scale
        var init_vy: states.element_type = (
            velocity_rand[1] * Scalar[dtype](2.0) - Scalar[dtype](1.0)
        ) * init_random_scale

        Self._init_bodies_single_env[BATCH_SIZE, STATE_SIZE](
            states, env, rebind[Scalar[dtype]](init_vx), rebind[Scalar[dtype]](init_vy)
        )

        # Initialize observation - use Scalar[dtype] to avoid Float64 issues
        var y_norm: states.element_type = Scalar[dtype](
            (
                LLConstants.H_UNITS
                - (LLConstants.HELIPAD_Y + LLConstants.LEG_DOWN)
            )
            / (LLConstants.H_UNITS / 2.0)
        )
        var vx_norm: states.element_type = (
            init_vx
            * Scalar[dtype](LLConstants.W_UNITS / 2.0)
            / Scalar[dtype](50.0)
        )
        var vy_norm: states.element_type = (
            init_vy
            * Scalar[dtype](LLConstants.H_UNITS / 2.0)
            / Scalar[dtype](50.0)
        )

        states[env, LLConstants.OBS_OFFSET + 0] = Scalar[dtype](0)
        states[env, LLConstants.OBS_OFFSET + 1] = y_norm
        states[env, LLConstants.OBS_OFFSET + 2] = vx_norm
        states[env, LLConstants.OBS_OFFSET + 3] = vy_norm
        states[env, LLConstants.OBS_OFFSET + 4] = Scalar[dtype](0)
        states[env, LLConstants.OBS_OFFSET + 5] = Scalar[dtype](0)
        states[env, LLConstants.OBS_OFFSET + 6] = Scalar[dtype](0)
        states[env, LLConstants.OBS_OFFSET + 7] = Scalar[dtype](0)

        # Compute initial shaping (same formula as _finalize_step_gpu)
        # At reset: x_norm=0, angle=0, left_contact=0, right_contact=0
        var y_norm_abs: states.element_type = y_norm
        if y_norm < 0:
            y_norm_abs = -y_norm
        var dist: states.element_type = (
            y_norm_abs  # sqrt(0² + y_norm²) = |y_norm|
        )
        var speed: states.element_type = sqrt(
            vx_norm * vx_norm + vy_norm * vy_norm
        )
        var init_shaping: states.element_type = (
            Scalar[dtype](-100.0) * dist
            - Scalar[dtype](100.0) * speed
            # angle = 0, left_contact = 0, right_contact = 0
        )

        # Initialize metadata
        states[
            env, LLConstants.METADATA_OFFSET + LLConstants.META_STEP_COUNT
        ] = Scalar[dtype](0)
        states[
            env, LLConstants.METADATA_OFFSET + LLConstants.META_TOTAL_REWARD
        ] = Scalar[dtype](0)
        states[
            env, LLConstants.METADATA_OFFSET + LLConstants.META_PREV_SHAPING
        ] = init_shaping
        states[
            env, LLConstants.METADATA_OFFSET + LLConstants.META_DONE
        ] = Scalar[dtype](0)

    @staticmethod
    def _init_shapes_gpu(
        ctx: DeviceContext,
        mut shapes_buf: DeviceBuffer[dtype],
    ) raises:
        """Initialize shape definitions (shared across all environments)."""
        var shapes = LayoutTensor[
            dtype,
            Layout.row_major(LLConstants.NUM_SHAPES * SHAPE_MAX_SIZE),
        ](shapes_buf)

        @always_inline
        def init_shapes_wrapper(
            shapes: LayoutTensor[
                dtype,
                Layout.row_major(LLConstants.NUM_SHAPES * SHAPE_MAX_SIZE),
                MutAnyOrigin,
            ],
        ):
            var tid = Int(block_dim.x * block_idx.x + thread_idx.x)
            if tid > 0:
                return

            # Lander shape (6-vertex polygon matching Gymnasium)
            shapes[0] = Scalar[dtype](SHAPE_POLYGON)
            shapes[1] = Scalar[dtype](6)
            shapes[2] = Scalar[dtype](-14.0 / LLConstants.SCALE)
            shapes[3] = Scalar[dtype](17.0 / LLConstants.SCALE)
            shapes[4] = Scalar[dtype](-17.0 / LLConstants.SCALE)
            shapes[5] = Scalar[dtype](0.0)
            shapes[6] = Scalar[dtype](-17.0 / LLConstants.SCALE)
            shapes[7] = Scalar[dtype](-10.0 / LLConstants.SCALE)
            shapes[8] = Scalar[dtype](17.0 / LLConstants.SCALE)
            shapes[9] = Scalar[dtype](-10.0 / LLConstants.SCALE)
            shapes[10] = Scalar[dtype](17.0 / LLConstants.SCALE)
            shapes[11] = Scalar[dtype](0.0)
            shapes[12] = Scalar[dtype](14.0 / LLConstants.SCALE)
            shapes[13] = Scalar[dtype](17.0 / LLConstants.SCALE)

            # Leg shapes (rectangles)
            for leg in range(2):
                var base = (leg + 1) * SHAPE_MAX_SIZE
                shapes[base + 0] = Scalar[dtype](SHAPE_POLYGON)
                shapes[base + 1] = Scalar[dtype](4)
                shapes[base + 2] = Scalar[dtype](-LLConstants.LEG_W)
                shapes[base + 3] = Scalar[dtype](LLConstants.LEG_H)
                shapes[base + 4] = Scalar[dtype](-LLConstants.LEG_W)
                shapes[base + 5] = Scalar[dtype](-LLConstants.LEG_H)
                shapes[base + 6] = Scalar[dtype](LLConstants.LEG_W)
                shapes[base + 7] = Scalar[dtype](-LLConstants.LEG_H)
                shapes[base + 8] = Scalar[dtype](LLConstants.LEG_W)
                shapes[base + 9] = Scalar[dtype](LLConstants.LEG_H)

        ctx.enqueue_function[init_shapes_wrapper](
            shapes,
            grid_dim=(1,),
            block_dim=(1,),
        )

    # =========================================================================
    # Fused Kernels for Maximum Performance
    # =========================================================================
    #
    # These kernels fuse multiple operations to reduce kernel launch overhead.
    # Pipeline: Setup (1 kernel) → Physics+Finalize (1 kernel) = 2 kernels total
    #
    # Previously: 5 kernels (zero + extract + forces + physics + finalize)
    # Now:        2 kernels (setup + physics_with_finalize)
    # =========================================================================

    @always_inline
    @staticmethod
    def _apply_wind_gpu[
        BATCH_SIZE: Int,
    ](
        env: Int,
        states: LayoutTensor[
            dtype,
            Layout.row_major(BATCH_SIZE, LLConstants.STATE_SIZE_VAL),
            MutAnyOrigin,
        ],
        step_count: Int,
    ):
        """Apply wind and turbulence forces for GPU computation.

        Uses compile-time parameters ENABLE_WIND, WIND_POWER, TURBULENCE_POWER
        from the struct. When ENABLE_WIND is False, this is a no-op that gets
        eliminated at compile time.
        """

        comptime if not Self.ENABLE_WIND:
            return

        # Check leg contacts from observation - skip wind if grounded
        var left_contact = rebind[Scalar[dtype]](
            states[env, LLConstants.OBS_OFFSET + 6]
        )
        var right_contact = rebind[Scalar[dtype]](
            states[env, LLConstants.OBS_OFFSET + 7]
        )
        if left_contact > Scalar[dtype](0.5) or right_contact > Scalar[dtype](
            0.5
        ):
            return

        # Compute wind magnitude using deterministic wave pattern
        # This matches the CPU implementation's wave formula
        var k = Scalar[dtype](0.01)
        var wind_t = Scalar[dtype](step_count)
        var wind_mag = tanh(
            sin(Scalar[dtype](0.02) * wind_t)
            + sin(Scalar[dtype](pi) * k * wind_t)
        ) * Scalar[dtype](Self.WIND_POWER)
        var torque_mag = tanh(
            sin(Scalar[dtype](0.02) * wind_t)
            + sin(Scalar[dtype](pi) * k * wind_t)
        ) * Scalar[dtype](Self.TURBULENCE_POWER)

        # Apply to lander velocity
        var dt = Scalar[dtype](LLConstants.DT)
        var lander_mass = Scalar[dtype](LLConstants.LANDER_MASS)
        var lander_inertia = Scalar[dtype](LLConstants.LANDER_INERTIA)

        var vx = rebind[Scalar[dtype]](
            states[env, LLConstants.BODIES_OFFSET + IDX_VX]
        )
        var omega = rebind[Scalar[dtype]](
            states[env, LLConstants.BODIES_OFFSET + IDX_OMEGA]
        )

        states[env, LLConstants.BODIES_OFFSET + IDX_VX] = (
            vx + wind_mag * dt / lander_mass
        )
        states[env, LLConstants.BODIES_OFFSET + IDX_OMEGA] = (
            omega + torque_mag * dt / lander_inertia
        )

    @always_inline
    @staticmethod
    def _setup_single_env[
        BATCH_SIZE: Int,
    ](
        env: Int,
        states: LayoutTensor[
            dtype,
            Layout.row_major(BATCH_SIZE, LLConstants.STATE_SIZE_VAL),
            MutAnyOrigin,
        ],
        actions: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), ImmutAnyOrigin
        ],
        edge_counts: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ],
        joint_counts: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ],
        contact_counts: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ],
    ):
        """Fused setup for single environment: zero + extract + apply_forces."""
        # 1. Zero contact count
        contact_counts[env] = Scalar[dtype](0)

        # 2. Extract edge/joint counts from state
        edge_counts[env] = states[env, LLConstants.EDGE_COUNT_OFFSET]
        joint_counts[env] = states[env, LLConstants.JOINT_COUNT_OFFSET]

        # 3. Clear forces
        for body in range(LLConstants.NUM_BODIES):
            var force_off = LLConstants.FORCES_OFFSET + body * 3
            states[env, force_off + 0] = Scalar[dtype](0)
            states[env, force_off + 1] = Scalar[dtype](0)
            states[env, force_off + 2] = Scalar[dtype](0)

        # 4. Check if done - skip force application
        if rebind[Scalar[dtype]](
            states[env, LLConstants.METADATA_OFFSET + LLConstants.META_DONE]
        ) > Scalar[dtype](0.5):
            return

        # 5. Get step count for wind and engine dispersion
        var step_count = Int(
            states[
                env, LLConstants.METADATA_OFFSET + LLConstants.META_STEP_COUNT
            ]
        )

        # 6. Apply wind forces (compile-time eliminated if ENABLE_WIND=False)
        Self.SelfType._apply_wind_gpu[BATCH_SIZE](env, states, step_count)

        # 7. Check action - no-op action skips engine forces
        var action = Int(actions[env])
        if action == 0:
            return

        # 8. Apply engine forces based on action

        var rng = PhiloxRandom(
            seed=UInt64(env) + 12345, offset=UInt64(step_count)
        )
        var rand_vals = rng.step_uniform()

        var dispersion_x = (
            rand_vals[0] * Scalar[dtype](2.0) - Scalar[dtype](1.0)
        ) / Scalar[dtype](LLConstants.SCALE)
        var dispersion_y = (
            rand_vals[1] * Scalar[dtype](2.0) - Scalar[dtype](1.0)
        ) / Scalar[dtype](LLConstants.SCALE)

        var angle = rebind[Scalar[dtype]](
            states[env, LLConstants.BODIES_OFFSET + IDX_ANGLE]
        )
        var tip_x = sin(angle)
        var tip_y = cos(angle)
        var side_x = -tip_y
        var side_y = tip_x

        var vx = rebind[Scalar[dtype]](
            states[env, LLConstants.BODIES_OFFSET + IDX_VX]
        )
        var vy = rebind[Scalar[dtype]](
            states[env, LLConstants.BODIES_OFFSET + IDX_VY]
        )
        var omega = rebind[Scalar[dtype]](
            states[env, LLConstants.BODIES_OFFSET + IDX_OMEGA]
        )

        var dvx = Scalar[dtype](0)
        var dvy = Scalar[dtype](0)
        var domega = Scalar[dtype](0)

        var main_y_offset = Scalar[dtype](LLConstants.MAIN_ENGINE_Y_OFFSET)
        var side_away = Scalar[dtype](LLConstants.SIDE_ENGINE_AWAY)
        var side_height = Scalar[dtype](LLConstants.SIDE_ENGINE_HEIGHT)
        var main_power = Scalar[dtype](LLConstants.MAIN_ENGINE_POWER)
        var side_power = Scalar[dtype](LLConstants.SIDE_ENGINE_POWER)
        var lander_mass = Scalar[dtype](LLConstants.LANDER_MASS)
        var lander_inertia = Scalar[dtype](LLConstants.LANDER_INERTIA)
        var scale = Scalar[dtype](LLConstants.SCALE)

        if action == 2:  # Main engine
            var ox = (
                tip_x * (main_y_offset + Scalar[dtype](2.0) * dispersion_x)
                + side_x * dispersion_y
            )
            var oy = (
                -tip_y * (main_y_offset + Scalar[dtype](2.0) * dispersion_x)
                - side_y * dispersion_y
            )
            var impulse_x = -ox * main_power
            var impulse_y = -oy * main_power
            dvx += impulse_x / lander_mass
            dvy += impulse_y / lander_mass
            var torque = ox * impulse_y - oy * impulse_x
            domega += torque / lander_inertia

        elif action == 1:  # Left engine
            var direction = Scalar[dtype](-1.0)
            var ox = tip_x * dispersion_x + side_x * (
                Scalar[dtype](3.0) * dispersion_y + direction * side_away
            )
            var oy = -tip_y * dispersion_x - side_y * (
                Scalar[dtype](3.0) * dispersion_y + direction * side_away
            )
            var impulse_x = -ox * side_power
            var impulse_y = -oy * side_power
            dvx += impulse_x / lander_mass
            dvy += impulse_y / lander_mass
            var r_x = ox - tip_x * Scalar[dtype](17.0) / scale
            var r_y = oy + tip_y * side_height
            var torque = r_x * impulse_y - r_y * impulse_x
            domega += torque / lander_inertia

        elif action == 3:  # Right engine
            var direction = Scalar[dtype](1.0)
            var ox = tip_x * dispersion_x + side_x * (
                Scalar[dtype](3.0) * dispersion_y + direction * side_away
            )
            var oy = -tip_y * dispersion_x - side_y * (
                Scalar[dtype](3.0) * dispersion_y + direction * side_away
            )
            var impulse_x = -ox * side_power
            var impulse_y = -oy * side_power
            dvx += impulse_x / lander_mass
            dvy += impulse_y / lander_mass
            var r_x = ox - tip_x * Scalar[dtype](17.0) / scale
            var r_y = oy + tip_y * side_height
            var torque = r_x * impulse_y - r_y * impulse_x
            domega += torque / lander_inertia

        states[env, LLConstants.BODIES_OFFSET + IDX_VX] = vx + dvx
        states[env, LLConstants.BODIES_OFFSET + IDX_VY] = vy + dvy
        states[env, LLConstants.BODIES_OFFSET + IDX_OMEGA] = omega + domega

    @always_inline
    @staticmethod
    def _finalize_single_env[
        BATCH_SIZE: Int,
    ](
        env: Int,
        states: LayoutTensor[
            dtype,
            Layout.row_major(BATCH_SIZE, LLConstants.STATE_SIZE_VAL),
            MutAnyOrigin,
        ],
        actions: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), ImmutAnyOrigin
        ],
        contacts: LayoutTensor[
            dtype,
            Layout.row_major(
                BATCH_SIZE, LLConstants.MAX_CONTACTS, CONTACT_DATA_SIZE
            ),
            MutAnyOrigin,
        ],
        contact_counts: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ],
        rewards: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ],
        dones: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
        terminated_out: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ],
    ):
        """Finalize step for single env: obs update + reward + done check."""
        var lander_off = LLConstants.BODIES_OFFSET

        # Check if already done
        if rebind[Scalar[dtype]](
            states[env, LLConstants.METADATA_OFFSET + LLConstants.META_DONE]
        ) > Scalar[dtype](0.5):
            rewards[env] = Scalar[dtype](0)
            dones[env] = Scalar[dtype](1)
            terminated_out[env] = Scalar[dtype](1)
            return

        # Get lander state
        var x = rebind[Scalar[dtype]](states[env, lander_off + IDX_X])
        var y = rebind[Scalar[dtype]](states[env, lander_off + IDX_Y])
        var vx = rebind[Scalar[dtype]](states[env, lander_off + IDX_VX])
        var vy = rebind[Scalar[dtype]](states[env, lander_off + IDX_VY])
        var angle = rebind[Scalar[dtype]](states[env, lander_off + IDX_ANGLE])
        var omega = rebind[Scalar[dtype]](states[env, lander_off + IDX_OMEGA])

        # Normalize observation
        var pos_norm = normalize_position[dtype](x, y)
        var vel_norm = normalize_velocity[dtype](vx, vy)
        var x_norm = pos_norm[0]
        var y_norm = pos_norm[1]
        var vx_norm = vel_norm[0]
        var vy_norm = vel_norm[1]
        var omega_norm = normalize_angular_velocity[dtype](omega)

        # Check leg contacts using actual terrain height at leg positions
        # (matches CPU behavior which checks against actual terrain height)
        var left_contact = Scalar[dtype](0.0)
        var right_contact = Scalar[dtype](0.0)

        # Get leg positions
        var left_x = rebind[Scalar[dtype]](
            states[env, LLConstants.BODIES_OFFSET + BODY_STATE_SIZE + IDX_X]
        )
        var left_y = rebind[Scalar[dtype]](
            states[env, LLConstants.BODIES_OFFSET + BODY_STATE_SIZE + IDX_Y]
        )
        var right_x = rebind[Scalar[dtype]](
            states[env, LLConstants.BODIES_OFFSET + 2 * BODY_STATE_SIZE + IDX_X]
        )
        var right_y = rebind[Scalar[dtype]](
            states[env, LLConstants.BODIES_OFFSET + 2 * BODY_STATE_SIZE + IDX_Y]
        )

        # Compute terrain height at leg positions from edges
        # Edge layout: x0, y0, x1, y1, nx, ny (6 values per edge)
        var n_edges = LLConstants.TERRAIN_CHUNKS - 1
        var x_spacing = Scalar[dtype](
            LLConstants.W_UNITS / Float64(LLConstants.TERRAIN_CHUNKS - 1)
        )

        # Left leg terrain height
        # Use chunk index to get terrain height (matching CPU which has TERRAIN_CHUNKS heights)
        # Edge i stores: x0, y0 (chunk i height), x1, y1 (chunk i+1 height), nx, ny
        # For chunk 0 to n_edges-1, use y0 from edge[chunk]
        # For chunk n_edges (last chunk), use y1 from edge[n_edges-1]
        var left_chunk_idx = Int(left_x / x_spacing)
        if left_chunk_idx < 0:
            left_chunk_idx = 0
        var left_terrain_y: Scalar[dtype]
        if left_chunk_idx >= n_edges:
            # Last chunk: use y1 from the last edge
            var last_edge_off = LLConstants.EDGES_OFFSET + (n_edges - 1) * 6
            left_terrain_y = rebind[Scalar[dtype]](
                states[env, last_edge_off + 3]
            )  # y1
        else:
            var left_edge_off = LLConstants.EDGES_OFFSET + left_chunk_idx * 6
            left_terrain_y = rebind[Scalar[dtype]](
                states[env, left_edge_off + 1]
            )  # y0

        # Right leg terrain height
        var right_chunk_idx = Int(right_x / x_spacing)
        if right_chunk_idx < 0:
            right_chunk_idx = 0
        var right_terrain_y: Scalar[dtype]
        if right_chunk_idx >= n_edges:
            # Last chunk: use y1 from the last edge
            var last_edge_off = LLConstants.EDGES_OFFSET + (n_edges - 1) * 6
            right_terrain_y = rebind[Scalar[dtype]](
                states[env, last_edge_off + 3]
            )  # y1
        else:
            var right_edge_off = LLConstants.EDGES_OFFSET + right_chunk_idx * 6
            right_terrain_y = rebind[Scalar[dtype]](
                states[env, right_edge_off + 1]
            )  # y0

        # Check contact: leg_y - LEG_H <= terrain_y + tolerance (matching CPU)
        var leg_h = Scalar[dtype](LLConstants.LEG_H)
        var contact_tolerance = Scalar[dtype](0.01)
        if left_y - leg_h <= left_terrain_y + contact_tolerance:
            left_contact = Scalar[dtype](1.0)
        if right_y - leg_h <= right_terrain_y + contact_tolerance:
            right_contact = Scalar[dtype](1.0)

        # Update observation
        states[env, LLConstants.OBS_OFFSET + 0] = x_norm
        states[env, LLConstants.OBS_OFFSET + 1] = y_norm
        states[env, LLConstants.OBS_OFFSET + 2] = vx_norm
        states[env, LLConstants.OBS_OFFSET + 3] = vy_norm
        states[env, LLConstants.OBS_OFFSET + 4] = angle
        states[env, LLConstants.OBS_OFFSET + 5] = omega_norm
        states[env, LLConstants.OBS_OFFSET + 6] = left_contact
        states[env, LLConstants.OBS_OFFSET + 7] = right_contact

        # Compute shaping
        var shaping = compute_shaping[dtype](
            x_norm, y_norm, vx_norm, vy_norm, angle, left_contact, right_contact
        )
        var prev_shaping = rebind[Scalar[dtype]](
            states[
                env, LLConstants.METADATA_OFFSET + LLConstants.META_PREV_SHAPING
            ]
        )
        var reward = shaping - prev_shaping
        states[
            env, LLConstants.METADATA_OFFSET + LLConstants.META_PREV_SHAPING
        ] = shaping

        # Fuel costs
        var action = Int(actions[env])
        if action == 2:
            reward = reward - Scalar[dtype](LLConstants.MAIN_ENGINE_FUEL_COST)
        elif action == 1 or action == 3:
            reward = reward - Scalar[dtype](LLConstants.SIDE_ENGINE_FUEL_COST)

        # Check termination
        var done = Scalar[dtype](0.0)
        var is_terminated = Scalar[dtype](0.0)

        # Too high — documented GPU-only deviation (no CPU counterpart);
        # applied BEFORE the shared terminal eval so the crash-override
        # order matches the historical kernel exactly.
        var h_units_max = Scalar[dtype](LLConstants.H_UNITS * 1.5)
        if y > h_units_max:
            done = Scalar[dtype](1.0)
            is_terminated = Scalar[dtype](1.0)
            reward = Scalar[dtype](LLConstants.CRASH_PENALTY)

        # Crash: lander body touches ground
        var n_contacts = Int(contact_counts[env])
        var lander_contact = False
        for c in range(n_contacts):
            var body_a = Int(contacts[env, c, CONTACT_BODY_A])
            if body_a == LLConstants.BODY_LANDER:
                lander_contact = True
                break

        # Landed = the Box2D island fell asleep (clock advanced by the solve).
        var asleep = island_is_asleep(
            rebind[Scalar[dtype]](
                states[
                    env,
                    LLConstants.METADATA_OFFSET + LLConstants.META_SLEEP_TIME,
                ]
            )
        )

        # Single-source terminal predicate + terminal reward overrides
        # (shared with the CPU _compute_step_result — see helpers.mojo).
        var tr = lunar_terminal_and_reward[dtype](
            x_norm, lander_contact, asleep, reward
        )
        reward = tr[0]
        if tr[1]:
            done = Scalar[dtype](1.0)
            is_terminated = Scalar[dtype](1.0)

        # Max steps (truncation only, not termination). `step_count` is the
        # PRE-increment count (incremented below), so the 1000th step reads
        # 999 — matches the CPU/Gymnasium 1000-step limit (the old
        # `> 1000` check truncated at step 1002).
        var step_count = rebind[Scalar[dtype]](
            states[
                env, LLConstants.METADATA_OFFSET + LLConstants.META_STEP_COUNT
            ]
        )
        if step_count >= Scalar[dtype](999.0):
            done = Scalar[dtype](1.0)

        # Update metadata
        states[
            env, LLConstants.METADATA_OFFSET + LLConstants.META_STEP_COUNT
        ] = step_count + Scalar[dtype](1.0)
        states[env, LLConstants.METADATA_OFFSET + LLConstants.META_DONE] = done
        rewards[env] = reward
        dones[env] = done
        terminated_out[env] = is_terminated

    @staticmethod
    def _setup_fused_gpu[
        BATCH_SIZE: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[dtype],
        actions_buf: DeviceBuffer[dtype],
        mut edge_counts_buf: DeviceBuffer[dtype],
        mut joint_counts_buf: DeviceBuffer[dtype],
        mut contact_counts_buf: DeviceBuffer[dtype],
    ) raises:
        """Fused setup kernel: zero + extract + apply_forces in ONE kernel."""
        var states = LayoutTensor[
            dtype,
            Layout.row_major(BATCH_SIZE, LLConstants.STATE_SIZE_VAL),
        ](states_buf)
        var actions = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE)
        ](actions_buf)
        var edge_counts = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE)
        ](edge_counts_buf)
        var joint_counts = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE)
        ](joint_counts_buf)
        var contact_counts = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE)
        ](contact_counts_buf)

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        @always_inline
        def setup_kernel(
            states: LayoutTensor[
                dtype,
                Layout.row_major(BATCH_SIZE, LLConstants.STATE_SIZE_VAL),
                MutAnyOrigin,
            ],
            actions: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE), ImmutAnyOrigin
            ],
            edge_counts: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
            joint_counts: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
            contact_counts: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH_SIZE:
                return
            Self.SelfType._setup_single_env[BATCH_SIZE](
                env, states, actions, edge_counts, joint_counts, contact_counts
            )

        ctx.enqueue_function[setup_kernel](
            states,
            actions,
            edge_counts,
            joint_counts,
            contact_counts,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    @staticmethod
    def _physics_finalize_obs_fused_gpu[
        BATCH_SIZE: Int,
        OBS_DIM: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[dtype],
        shapes_buf: DeviceBuffer[dtype],
        edge_counts_buf: DeviceBuffer[dtype],
        joint_counts_buf: DeviceBuffer[dtype],
        mut contacts_buf: DeviceBuffer[dtype],
        mut contact_counts_buf: DeviceBuffer[dtype],
        actions_buf: DeviceBuffer[dtype],
        mut rewards_buf: DeviceBuffer[dtype],
        mut dones_buf: DeviceBuffer[dtype],
        mut terminated_buf: DeviceBuffer[dtype],
        mut obs_buf: DeviceBuffer[dtype],
        gravity_x: Scalar[dtype],
        gravity_y: Scalar[dtype],
        dt: Scalar[dtype],
    ) raises:
        """Fused physics + finalize + extract_obs kernel.

        Same as _physics_finalize_fused_gpu but also extracts observations
        to obs_buf, eliminating the need for a separate extract_obs kernel.
        """
        var states = LayoutTensor[
            dtype,
            Layout.row_major(BATCH_SIZE, LLConstants.STATE_SIZE_VAL),
        ](states_buf)
        var shapes = LayoutTensor[
            dtype,
            Layout.row_major(LLConstants.NUM_SHAPES, SHAPE_MAX_SIZE),
        ](shapes_buf)
        var edge_counts = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE)
        ](edge_counts_buf)
        var joint_counts = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE)
        ](joint_counts_buf)
        var contacts = LayoutTensor[
            dtype,
            Layout.row_major(
                BATCH_SIZE, LLConstants.MAX_CONTACTS, CONTACT_DATA_SIZE
            ),
        ](contacts_buf)
        var contact_counts = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE)
        ](contact_counts_buf)
        var actions = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE)
        ](actions_buf)
        var rewards = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE)
        ](rewards_buf)
        var dones = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE)
        ](dones_buf)
        var terminated_out = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE)
        ](terminated_buf)
        var obs = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, OBS_DIM)
        ](obs_buf)

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        @always_inline
        def physics_finalize_obs_kernel(
            states: LayoutTensor[
                dtype,
                Layout.row_major(BATCH_SIZE, LLConstants.STATE_SIZE_VAL),
                MutAnyOrigin,
            ],
            shapes: LayoutTensor[
                dtype,
                Layout.row_major(LLConstants.NUM_SHAPES, SHAPE_MAX_SIZE),
                ImmutAnyOrigin,
            ],
            edge_counts: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE), ImmutAnyOrigin
            ],
            joint_counts: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE), ImmutAnyOrigin
            ],
            contacts: LayoutTensor[
                dtype,
                Layout.row_major(
                    BATCH_SIZE, LLConstants.MAX_CONTACTS, CONTACT_DATA_SIZE
                ),
                MutAnyOrigin,
            ],
            contact_counts: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
            actions: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE), ImmutAnyOrigin
            ],
            rewards: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
            dones: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
            terminated_out: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
            obs: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE, OBS_DIM), MutAnyOrigin
            ],
            gravity_x: Scalar[dtype],
            gravity_y: Scalar[dtype],
            dt: Scalar[dtype],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH_SIZE:
                return

            var n_edges = Int(edge_counts[env])
            var n_joints = Int(joint_counts[env])

            # Physics step: detection, then the shared Box2D island solve
            # (the SAME `_solve_step_single_env` the CPU step runs).
            EdgeTerrainCollision.detect_single_env[
                BATCH_SIZE,
                LLConstants.NUM_BODIES,
                LLConstants.NUM_SHAPES,
                LLConstants.MAX_CONTACTS,
                MAX_TERRAIN_EDGES,
                LLConstants.STATE_SIZE_VAL,
                LLConstants.BODIES_OFFSET,
                LLConstants.EDGES_OFFSET,
            ](env, states, shapes, n_edges, contacts, contact_counts)

            var n_contacts = Int(contact_counts[env])
            Self.SelfType._solve_step_single_env[
                BATCH_SIZE, LLConstants.STATE_SIZE_VAL
            ](
                env,
                states,
                contacts,
                n_contacts,
                n_joints,
                gravity_x,
                gravity_y,
                dt,
            )

            # Finalize (writes obs to states at OBS_OFFSET)
            Self.SelfType._finalize_single_env[BATCH_SIZE](
                env,
                states,
                actions,
                contacts,
                contact_counts,
                rewards,
                dones,
                terminated_out,
            )

            # Extract observations to separate obs buffer (OBS_OFFSET = 0)
            for d in range(OBS_DIM):
                obs[env, d] = states[env, d]

        ctx.enqueue_function[physics_finalize_obs_kernel](
            states,
            shapes,
            edge_counts,
            joint_counts,
            contacts,
            contact_counts,
            actions,
            rewards,
            dones,
            terminated_out,
            obs,
            gravity_x,
            gravity_y,
            dt,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    # =========================================================================
    # Continuous Action GPU Kernels (GPUContinuousEnv)
    # =========================================================================

    @staticmethod
    def _setup_fused_gpu_continuous[
        BATCH_SIZE: Int,
        ACTION_DIM: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[dtype],
        actions_buf: DeviceBuffer[dtype],
        mut edge_counts_buf: DeviceBuffer[dtype],
        mut joint_counts_buf: DeviceBuffer[dtype],
        mut contact_counts_buf: DeviceBuffer[dtype],
    ) raises:
        """Fused setup kernel for continuous actions."""
        var states = LayoutTensor[
            dtype,
            Layout.row_major(BATCH_SIZE, LLConstants.STATE_SIZE_VAL),
        ](states_buf)
        var actions = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, ACTION_DIM)
        ](actions_buf)
        var edge_counts = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE)
        ](edge_counts_buf)
        var joint_counts = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE)
        ](joint_counts_buf)
        var contact_counts = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE)
        ](contact_counts_buf)

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        @always_inline
        def setup_kernel_continuous(
            states: LayoutTensor[
                dtype,
                Layout.row_major(BATCH_SIZE, LLConstants.STATE_SIZE_VAL),
                MutAnyOrigin,
            ],
            actions: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE, ACTION_DIM), ImmutAnyOrigin
            ],
            edge_counts: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
            joint_counts: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
            contact_counts: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH_SIZE:
                return
            Self.SelfType._setup_single_env_continuous[BATCH_SIZE, ACTION_DIM](
                env, states, actions, edge_counts, joint_counts, contact_counts
            )

        ctx.enqueue_function[setup_kernel_continuous](
            states,
            actions,
            edge_counts,
            joint_counts,
            contact_counts,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    @always_inline
    @staticmethod
    def _setup_single_env_continuous[
        BATCH_SIZE: Int,
        ACTION_DIM: Int,
    ](
        env: Int,
        states: LayoutTensor[
            dtype,
            Layout.row_major(BATCH_SIZE, LLConstants.STATE_SIZE_VAL),
            MutAnyOrigin,
        ],
        actions: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, ACTION_DIM), ImmutAnyOrigin
        ],
        edge_counts: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ],
        joint_counts: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ],
        contact_counts: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ],
    ):
        """Fused setup for single env with continuous actions."""
        # 1. Zero contact count
        contact_counts[env] = Scalar[dtype](0)

        # 2. Extract edge/joint counts from state
        edge_counts[env] = states[env, LLConstants.EDGE_COUNT_OFFSET]
        joint_counts[env] = states[env, LLConstants.JOINT_COUNT_OFFSET]

        # 3. Clear forces
        for body in range(LLConstants.NUM_BODIES):
            var force_off = LLConstants.FORCES_OFFSET + body * 3
            states[env, force_off + 0] = Scalar[dtype](0)
            states[env, force_off + 1] = Scalar[dtype](0)
            states[env, force_off + 2] = Scalar[dtype](0)

        # 4. Check if done - skip force application
        if rebind[Scalar[dtype]](
            states[env, LLConstants.METADATA_OFFSET + LLConstants.META_DONE]
        ) > Scalar[dtype](0.5):
            return

        # 5. Get step count for wind and engine dispersion
        var step_count = Int(
            states[
                env, LLConstants.METADATA_OFFSET + LLConstants.META_STEP_COUNT
            ]
        )

        # 6. Apply wind forces (compile-time eliminated if ENABLE_WIND=False)
        Self.SelfType._apply_wind_gpu[BATCH_SIZE](env, states, step_count)

        # 7. Extract continuous actions (matching Gymnasium)
        # action[0]: main engine. <= 0 → off; (0,1] → power in [0.5, 1.0]
        # action[1]: side engine. |val| <= 0.5 → off; |val| > 0.5 → power = |val|
        var raw_throttle = rebind[Scalar[dtype]](actions[env, 0])
        var side_control = rebind[Scalar[dtype]](actions[env, 1])

        # Clip inputs to [-1, 1]
        if raw_throttle < Scalar[dtype](-1.0):
            raw_throttle = Scalar[dtype](-1.0)
        if raw_throttle > Scalar[dtype](1.0):
            raw_throttle = Scalar[dtype](1.0)
        if side_control < Scalar[dtype](-1.0):
            side_control = Scalar[dtype](-1.0)
        if side_control > Scalar[dtype](1.0):
            side_control = Scalar[dtype](1.0)

        # Main engine: off when action <= 0, power [0.5, 1.0] when action > 0
        var m_power = Scalar[dtype](0.0)
        if raw_throttle > Scalar[dtype](0.0):
            m_power = (raw_throttle + Scalar[dtype](1.0)) * Scalar[dtype](0.5)

        # Side engine: off when |action| <= 0.5, power = |action| when > 0.5
        var s_power = Scalar[dtype](0.0)
        var direction = Scalar[dtype](0.0)

        if side_control < Scalar[dtype](-0.5):
            direction = Scalar[dtype](-1.0)
            s_power = -side_control
        elif side_control > Scalar[dtype](0.5):
            direction = Scalar[dtype](1.0)
            s_power = side_control

        # Early exit if no thrust
        if m_power <= Scalar[dtype](0.0) and s_power <= Scalar[dtype](0.0):
            return

        # 8. Apply engine forces
        var rng = PhiloxRandom(
            seed=UInt64(env) + 12345, offset=UInt64(step_count)
        )
        var rand_vals = rng.step_uniform()

        var dispersion_x = (
            rand_vals[0] * Scalar[dtype](2.0) - Scalar[dtype](1.0)
        ) / Scalar[dtype](LLConstants.SCALE)
        var dispersion_y = (
            rand_vals[1] * Scalar[dtype](2.0) - Scalar[dtype](1.0)
        ) / Scalar[dtype](LLConstants.SCALE)

        var angle = rebind[Scalar[dtype]](
            states[env, LLConstants.BODIES_OFFSET + IDX_ANGLE]
        )
        var tip_x = sin(angle)
        var tip_y = cos(angle)
        var side_x = -tip_y
        var side_y = tip_x

        var vx = rebind[Scalar[dtype]](
            states[env, LLConstants.BODIES_OFFSET + IDX_VX]
        )
        var vy = rebind[Scalar[dtype]](
            states[env, LLConstants.BODIES_OFFSET + IDX_VY]
        )
        var omega = rebind[Scalar[dtype]](
            states[env, LLConstants.BODIES_OFFSET + IDX_OMEGA]
        )

        var dvx = Scalar[dtype](0)
        var dvy = Scalar[dtype](0)
        var domega = Scalar[dtype](0)

        var main_y_offset = Scalar[dtype](LLConstants.MAIN_ENGINE_Y_OFFSET)
        var side_away = Scalar[dtype](LLConstants.SIDE_ENGINE_AWAY)
        var side_height = Scalar[dtype](LLConstants.SIDE_ENGINE_HEIGHT)
        var main_power_const = Scalar[dtype](LLConstants.MAIN_ENGINE_POWER)
        var side_power_const = Scalar[dtype](LLConstants.SIDE_ENGINE_POWER)
        var lander_mass = Scalar[dtype](LLConstants.LANDER_MASS)
        var lander_inertia = Scalar[dtype](LLConstants.LANDER_INERTIA)
        var scale = Scalar[dtype](LLConstants.SCALE)

        # Apply main engine force (scaled by m_power)
        if m_power > Scalar[dtype](0.0):
            var ox = (
                tip_x * (main_y_offset + Scalar[dtype](2.0) * dispersion_x)
                + side_x * dispersion_y
            )
            var oy = (
                -tip_y * (main_y_offset + Scalar[dtype](2.0) * dispersion_x)
                - side_y * dispersion_y
            )
            var impulse_x = -ox * main_power_const * m_power
            var impulse_y = -oy * main_power_const * m_power
            dvx += impulse_x / lander_mass
            dvy += impulse_y / lander_mass
            var torque = ox * impulse_y - oy * impulse_x
            domega += torque / lander_inertia

        # Apply side engine force (scaled by s_power)
        if s_power > Scalar[dtype](0.0):
            var ox = tip_x * dispersion_x + side_x * (
                Scalar[dtype](3.0) * dispersion_y + direction * side_away
            )
            var oy = -tip_y * dispersion_x - side_y * (
                Scalar[dtype](3.0) * dispersion_y + direction * side_away
            )
            var impulse_x = -ox * side_power_const * s_power
            var impulse_y = -oy * side_power_const * s_power
            dvx += impulse_x / lander_mass
            dvy += impulse_y / lander_mass
            var r_x = ox - tip_x * Scalar[dtype](17.0) / scale
            var r_y = oy + tip_y * side_height
            var torque = r_x * impulse_y - r_y * impulse_x
            domega += torque / lander_inertia

        states[env, LLConstants.BODIES_OFFSET + IDX_VX] = vx + dvx
        states[env, LLConstants.BODIES_OFFSET + IDX_VY] = vy + dvy
        states[env, LLConstants.BODIES_OFFSET + IDX_OMEGA] = omega + domega

    @staticmethod
    def _physics_finalize_obs_fused_gpu_continuous[
        BATCH_SIZE: Int,
        OBS_DIM: Int,
        ACTION_DIM: Int,
    ](
        ctx: DeviceContext,
        mut states_buf: DeviceBuffer[dtype],
        shapes_buf: DeviceBuffer[dtype],
        edge_counts_buf: DeviceBuffer[dtype],
        joint_counts_buf: DeviceBuffer[dtype],
        mut contacts_buf: DeviceBuffer[dtype],
        mut contact_counts_buf: DeviceBuffer[dtype],
        actions_buf: DeviceBuffer[dtype],
        mut rewards_buf: DeviceBuffer[dtype],
        mut dones_buf: DeviceBuffer[dtype],
        mut terminated_buf: DeviceBuffer[dtype],
        mut obs_buf: DeviceBuffer[dtype],
        gravity_x: Scalar[dtype],
        gravity_y: Scalar[dtype],
        dt: Scalar[dtype],
    ) raises:
        """Fused physics + finalize + extract_obs for continuous actions."""
        var states = LayoutTensor[
            dtype,
            Layout.row_major(BATCH_SIZE, LLConstants.STATE_SIZE_VAL),
        ](states_buf)
        var shapes = LayoutTensor[
            dtype,
            Layout.row_major(LLConstants.NUM_SHAPES, SHAPE_MAX_SIZE),
        ](shapes_buf)
        var edge_counts = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE)
        ](edge_counts_buf)
        var joint_counts = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE)
        ](joint_counts_buf)
        var contacts = LayoutTensor[
            dtype,
            Layout.row_major(
                BATCH_SIZE, LLConstants.MAX_CONTACTS, CONTACT_DATA_SIZE
            ),
        ](contacts_buf)
        var contact_counts = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE)
        ](contact_counts_buf)
        var actions = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, ACTION_DIM)
        ](actions_buf)
        var rewards = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE)
        ](rewards_buf)
        var dones = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE)
        ](dones_buf)
        var terminated_out = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE)
        ](terminated_buf)
        var obs = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, OBS_DIM)
        ](obs_buf)

        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

        @always_inline
        def physics_finalize_obs_kernel_continuous(
            states: LayoutTensor[
                dtype,
                Layout.row_major(BATCH_SIZE, LLConstants.STATE_SIZE_VAL),
                MutAnyOrigin,
            ],
            shapes: LayoutTensor[
                dtype,
                Layout.row_major(LLConstants.NUM_SHAPES, SHAPE_MAX_SIZE),
                ImmutAnyOrigin,
            ],
            edge_counts: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE), ImmutAnyOrigin
            ],
            joint_counts: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE), ImmutAnyOrigin
            ],
            contacts: LayoutTensor[
                dtype,
                Layout.row_major(
                    BATCH_SIZE, LLConstants.MAX_CONTACTS, CONTACT_DATA_SIZE
                ),
                MutAnyOrigin,
            ],
            contact_counts: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
            actions: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE, ACTION_DIM), ImmutAnyOrigin
            ],
            rewards: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
            dones: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
            terminated_out: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
            ],
            obs: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE, OBS_DIM), MutAnyOrigin
            ],
            gravity_x: Scalar[dtype],
            gravity_y: Scalar[dtype],
            dt: Scalar[dtype],
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH_SIZE:
                return

            var n_edges = Int(edge_counts[env])
            var n_joints = Int(joint_counts[env])

            # Physics step (same as the discrete kernel): detection, then the
            # shared Box2D island solve.
            EdgeTerrainCollision.detect_single_env[
                BATCH_SIZE,
                LLConstants.NUM_BODIES,
                LLConstants.NUM_SHAPES,
                LLConstants.MAX_CONTACTS,
                MAX_TERRAIN_EDGES,
                LLConstants.STATE_SIZE_VAL,
                LLConstants.BODIES_OFFSET,
                LLConstants.EDGES_OFFSET,
            ](env, states, shapes, n_edges, contacts, contact_counts)

            var n_contacts = Int(contact_counts[env])
            Self.SelfType._solve_step_single_env[
                BATCH_SIZE, LLConstants.STATE_SIZE_VAL
            ](
                env,
                states,
                contacts,
                n_contacts,
                n_joints,
                gravity_x,
                gravity_y,
                dt,
            )

            # Finalize with continuous action fuel costs
            Self.SelfType._finalize_single_env_continuous[
                BATCH_SIZE, ACTION_DIM
            ](
                env,
                states,
                actions,
                contacts,
                contact_counts,
                rewards,
                dones,
                terminated_out,
            )

            # Extract observations to separate obs buffer
            for d in range(OBS_DIM):
                obs[env, d] = states[env, d]

        ctx.enqueue_function[physics_finalize_obs_kernel_continuous](
            states,
            shapes,
            edge_counts,
            joint_counts,
            contacts,
            contact_counts,
            actions,
            rewards,
            dones,
            terminated_out,
            obs,
            gravity_x,
            gravity_y,
            dt,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    @always_inline
    @staticmethod
    def _finalize_single_env_continuous[
        BATCH_SIZE: Int,
        ACTION_DIM: Int,
    ](
        env: Int,
        states: LayoutTensor[
            dtype,
            Layout.row_major(BATCH_SIZE, LLConstants.STATE_SIZE_VAL),
            MutAnyOrigin,
        ],
        actions: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, ACTION_DIM), ImmutAnyOrigin
        ],
        contacts: LayoutTensor[
            dtype,
            Layout.row_major(
                BATCH_SIZE, LLConstants.MAX_CONTACTS, CONTACT_DATA_SIZE
            ),
            MutAnyOrigin,
        ],
        contact_counts: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ],
        rewards: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ],
        dones: LayoutTensor[dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin],
        terminated_out: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE), MutAnyOrigin
        ],
    ):
        """Finalize step for single env with continuous action fuel costs."""
        var lander_off = LLConstants.BODIES_OFFSET

        # Check if already done
        if rebind[Scalar[dtype]](
            states[env, LLConstants.METADATA_OFFSET + LLConstants.META_DONE]
        ) > Scalar[dtype](0.5):
            rewards[env] = Scalar[dtype](0)
            dones[env] = Scalar[dtype](1)
            terminated_out[env] = Scalar[dtype](1)
            return

        # Get lander state
        var x = rebind[Scalar[dtype]](states[env, lander_off + IDX_X])
        var y = rebind[Scalar[dtype]](states[env, lander_off + IDX_Y])
        var vx = rebind[Scalar[dtype]](states[env, lander_off + IDX_VX])
        var vy = rebind[Scalar[dtype]](states[env, lander_off + IDX_VY])
        var angle = rebind[Scalar[dtype]](states[env, lander_off + IDX_ANGLE])
        var omega = rebind[Scalar[dtype]](states[env, lander_off + IDX_OMEGA])

        # Normalize observation
        var pos_norm = normalize_position[dtype](x, y)
        var vel_norm = normalize_velocity[dtype](vx, vy)
        var x_norm = pos_norm[0]
        var y_norm = pos_norm[1]
        var vx_norm = vel_norm[0]
        var vy_norm = vel_norm[1]
        var omega_norm = normalize_angular_velocity[dtype](omega)

        # Check leg contacts using actual terrain height at leg positions
        # (matches CPU behavior which checks against actual terrain height)
        var left_contact = Scalar[dtype](0.0)
        var right_contact = Scalar[dtype](0.0)

        # Get leg positions
        var left_x = rebind[Scalar[dtype]](
            states[env, LLConstants.BODIES_OFFSET + BODY_STATE_SIZE + IDX_X]
        )
        var left_y = rebind[Scalar[dtype]](
            states[env, LLConstants.BODIES_OFFSET + BODY_STATE_SIZE + IDX_Y]
        )
        var right_x = rebind[Scalar[dtype]](
            states[env, LLConstants.BODIES_OFFSET + 2 * BODY_STATE_SIZE + IDX_X]
        )
        var right_y = rebind[Scalar[dtype]](
            states[env, LLConstants.BODIES_OFFSET + 2 * BODY_STATE_SIZE + IDX_Y]
        )

        # Compute terrain height at leg positions from edges
        # Edge layout: x0, y0, x1, y1, nx, ny (6 values per edge)
        var n_edges = LLConstants.TERRAIN_CHUNKS - 1
        var x_spacing = Scalar[dtype](
            LLConstants.W_UNITS / Float64(LLConstants.TERRAIN_CHUNKS - 1)
        )

        # Left leg terrain height
        # Use chunk index to get terrain height (matching CPU which has TERRAIN_CHUNKS heights)
        # Edge i stores: x0, y0 (chunk i height), x1, y1 (chunk i+1 height), nx, ny
        # For chunk 0 to n_edges-1, use y0 from edge[chunk]
        # For chunk n_edges (last chunk), use y1 from edge[n_edges-1]
        var left_chunk_idx = Int(left_x / x_spacing)
        if left_chunk_idx < 0:
            left_chunk_idx = 0
        var left_terrain_y: Scalar[dtype]
        if left_chunk_idx >= n_edges:
            # Last chunk: use y1 from the last edge
            var last_edge_off = LLConstants.EDGES_OFFSET + (n_edges - 1) * 6
            left_terrain_y = rebind[Scalar[dtype]](
                states[env, last_edge_off + 3]
            )  # y1
        else:
            var left_edge_off = LLConstants.EDGES_OFFSET + left_chunk_idx * 6
            left_terrain_y = rebind[Scalar[dtype]](
                states[env, left_edge_off + 1]
            )  # y0

        # Right leg terrain height
        var right_chunk_idx = Int(right_x / x_spacing)
        if right_chunk_idx < 0:
            right_chunk_idx = 0
        var right_terrain_y: Scalar[dtype]
        if right_chunk_idx >= n_edges:
            # Last chunk: use y1 from the last edge
            var last_edge_off = LLConstants.EDGES_OFFSET + (n_edges - 1) * 6
            right_terrain_y = rebind[Scalar[dtype]](
                states[env, last_edge_off + 3]
            )  # y1
        else:
            var right_edge_off = LLConstants.EDGES_OFFSET + right_chunk_idx * 6
            right_terrain_y = rebind[Scalar[dtype]](
                states[env, right_edge_off + 1]
            )  # y0

        # Check contact: leg_y - LEG_H <= terrain_y + tolerance (matching CPU)
        var leg_h = Scalar[dtype](LLConstants.LEG_H)
        var contact_tolerance = Scalar[dtype](0.01)
        if left_y - leg_h <= left_terrain_y + contact_tolerance:
            left_contact = Scalar[dtype](1.0)
        if right_y - leg_h <= right_terrain_y + contact_tolerance:
            right_contact = Scalar[dtype](1.0)

        # Update observation
        states[env, LLConstants.OBS_OFFSET + 0] = x_norm
        states[env, LLConstants.OBS_OFFSET + 1] = y_norm
        states[env, LLConstants.OBS_OFFSET + 2] = vx_norm
        states[env, LLConstants.OBS_OFFSET + 3] = vy_norm
        states[env, LLConstants.OBS_OFFSET + 4] = angle
        states[env, LLConstants.OBS_OFFSET + 5] = omega_norm
        states[env, LLConstants.OBS_OFFSET + 6] = left_contact
        states[env, LLConstants.OBS_OFFSET + 7] = right_contact

        # Compute shaping
        var shaping = compute_shaping[dtype](
            x_norm, y_norm, vx_norm, vy_norm, angle, left_contact, right_contact
        )
        var prev_shaping = rebind[Scalar[dtype]](
            states[
                env, LLConstants.METADATA_OFFSET + LLConstants.META_PREV_SHAPING
            ]
        )
        var reward = shaping - prev_shaping
        states[
            env, LLConstants.METADATA_OFFSET + LLConstants.META_PREV_SHAPING
        ] = shaping

        # Continuous fuel costs — engine gating MUST match the force kernel
        # (and Gymnasium/CPU): main engine OFF (zero fuel) when action[0] <= 0,
        # else power = (a+1)/2 in [0.5, 1]; side engine OFF when
        # |action[1]| <= 0.5, else power = |action[1]| in (0.5, 1]. The old
        # code charged main fuel with the engine off ((raw+1)/2 even at
        # raw <= 0) and used a different side-power curve ((|a|-0.5)*2).
        var raw_throttle = rebind[Scalar[dtype]](actions[env, 0])
        var side_control = rebind[Scalar[dtype]](actions[env, 1])
        if raw_throttle < Scalar[dtype](-1.0):
            raw_throttle = Scalar[dtype](-1.0)
        if raw_throttle > Scalar[dtype](1.0):
            raw_throttle = Scalar[dtype](1.0)
        if side_control < Scalar[dtype](-1.0):
            side_control = Scalar[dtype](-1.0)
        if side_control > Scalar[dtype](1.0):
            side_control = Scalar[dtype](1.0)

        if raw_throttle > Scalar[dtype](0.0):
            var m_power = (raw_throttle + Scalar[dtype](1.0)) * Scalar[dtype](
                0.5
            )
            reward = reward - m_power * Scalar[dtype](
                LLConstants.MAIN_ENGINE_FUEL_COST
            )

        var abs_side = side_control
        if abs_side < Scalar[dtype](0.0):
            abs_side = -abs_side
        if abs_side > Scalar[dtype](0.5):
            reward = reward - abs_side * Scalar[dtype](
                LLConstants.SIDE_ENGINE_FUEL_COST
            )

        # Check termination
        var done = Scalar[dtype](0.0)
        var is_terminated = Scalar[dtype](0.0)

        # Too high — documented GPU-only deviation (no CPU counterpart);
        # applied BEFORE the shared terminal eval so the crash-override
        # order matches the historical kernel exactly.
        var h_units_max = Scalar[dtype](LLConstants.H_UNITS * 1.5)
        if y > h_units_max:
            done = Scalar[dtype](1.0)
            is_terminated = Scalar[dtype](1.0)
            reward = Scalar[dtype](LLConstants.CRASH_PENALTY)

        # Crash: lander body touches ground
        var n_contacts = Int(contact_counts[env])
        var lander_contact = False
        for c in range(n_contacts):
            var body_a = Int(contacts[env, c, CONTACT_BODY_A])
            if body_a == LLConstants.BODY_LANDER:
                lander_contact = True
                break

        # Landed = the Box2D island fell asleep (clock advanced by the solve).
        var asleep = island_is_asleep(
            rebind[Scalar[dtype]](
                states[
                    env,
                    LLConstants.METADATA_OFFSET + LLConstants.META_SLEEP_TIME,
                ]
            )
        )

        # Single-source terminal predicate + terminal reward overrides
        # (shared with the CPU _compute_step_result — see helpers.mojo).
        var tr = lunar_terminal_and_reward[dtype](
            x_norm, lander_contact, asleep, reward
        )
        reward = tr[0]
        if tr[1]:
            done = Scalar[dtype](1.0)
            is_terminated = Scalar[dtype](1.0)

        # Max steps (truncation only, not termination). `step_count` is the
        # PRE-increment count (incremented below), so the 1000th step reads
        # 999 — matches the CPU/Gymnasium 1000-step limit (the old
        # `> 1000` check truncated at step 1002).
        var step_count = rebind[Scalar[dtype]](
            states[
                env, LLConstants.METADATA_OFFSET + LLConstants.META_STEP_COUNT
            ]
        )
        if step_count >= Scalar[dtype](999.0):
            done = Scalar[dtype](1.0)

        # Update metadata
        states[
            env, LLConstants.METADATA_OFFSET + LLConstants.META_STEP_COUNT
        ] = step_count + Scalar[dtype](1.0)
        states[env, LLConstants.METADATA_OFFSET + LLConstants.META_DONE] = done
        rewards[env] = reward
        dones[env] = done
        terminated_out[env] = is_terminated

    # =========================================================================
    # Rendering Methods
    # =========================================================================

    def render(mut self, env: Int, mut renderer: Renderer2D):
        """Render a specific environment using the provided renderer.

        Args:
            env: Environment index to render (0 to BATCH-1).
            renderer: Initialized Renderer2D instance.

        The renderer should be initialized before calling this method.
        Call renderer.init_display() before first use if needed.
        """
        # Begin frame with space background
        if not renderer.begin_frame_with_color(space_black()):
            return

        # Create camera - centered at viewport center, with physics scale
        var W = Float64(LLConstants.VIEWPORT_W) / Float64(LLConstants.SCALE)
        var H = Float64(LLConstants.VIEWPORT_H) / Float64(LLConstants.SCALE)
        var camera = Camera(
            W / 2.0,  # Center X in world units
            H / 2.0,  # Center Y in world units
            Float64(LLConstants.SCALE),  # Zoom = physics scale
            Int(LLConstants.VIEWPORT_W),
            Int(LLConstants.VIEWPORT_H),
            flip_y=True,  # Y increases upward in physics
        )

        # Draw terrain (filled)
        self._draw_terrain(env, camera, renderer)

        # Draw helipad
        self._draw_helipad(env, camera, renderer)

        # Draw helipad flags
        self._draw_flags(env, camera, renderer)

        # Draw legs (before lander so lander draws on top)
        self._draw_legs(env, camera, renderer)

        # Draw lander
        self._draw_lander(env, camera, renderer)

        # Update and draw particles (engine flame effects)
        self._update_particles(LLConstants.TAU)
        self._draw_particles(camera, renderer)

        renderer.flip()

    def _draw_terrain(
        mut self, env: Int, camera: Camera, mut renderer: Renderer2D
    ):
        """Draw terrain as filled polygons using world coordinates."""
        var terrain_color = moon_gray()
        var terrain_dark = dark_gray()

        var W = Float64(LLConstants.VIEWPORT_W) / Float64(LLConstants.SCALE)

        # Draw each terrain segment as a filled quad (from terrain line to bottom)
        for i in range(LLConstants.TERRAIN_CHUNKS - 1):
            # Compute terrain x positions (evenly spaced across viewport)
            var x1 = W / Float64(LLConstants.TERRAIN_CHUNKS - 1) * Float64(i)
            var x2 = (
                W / Float64(LLConstants.TERRAIN_CHUNKS - 1) * Float64(i + 1)
            )

            # Get terrain heights from buffer (single-env CPU mode uses direct index)
            var y1 = Float64(self.terrain_heights[i])
            var y2 = Float64(self.terrain_heights[i + 1])

            # Create polygon vertices in world coordinates
            var vertices = List[RenderVec2]()
            vertices.append(RenderVec2(x1, y1))
            vertices.append(RenderVec2(x2, y2))
            vertices.append(RenderVec2(x2, 0.0))  # Bottom
            vertices.append(RenderVec2(x1, 0.0))

            renderer.draw_polygon_world(
                vertices, camera, terrain_color, filled=True
            )

            # Draw terrain outline for contrast
            renderer.draw_line_world(
                RenderVec2(x1, y1),
                RenderVec2(x2, y2),
                camera,
                terrain_dark,
                2,
            )

    def _draw_helipad(
        mut self, env: Int, camera: Camera, mut renderer: Renderer2D
    ):
        """Draw the helipad landing zone using world coordinates."""
        var helipad_color = darken(moon_gray(), 0.8)

        var W = Float64(LLConstants.VIEWPORT_W) / Float64(LLConstants.SCALE)

        # Compute helipad x positions (centered, spanning a few chunks)
        var helipad_x1 = (
            W
            / Float64(LLConstants.TERRAIN_CHUNKS - 1)
            * Float64(LLConstants.TERRAIN_CHUNKS // 2 - 1)
        )
        var helipad_x2 = (
            W
            / Float64(LLConstants.TERRAIN_CHUNKS - 1)
            * Float64(LLConstants.TERRAIN_CHUNKS // 2 + 1)
        )

        # Helipad is a thick horizontal bar (in world units)
        var bar_height = 4.0 / Float64(
            LLConstants.SCALE
        )  # 4 pixels in world units
        renderer.draw_rect_world(
            RenderVec2(
                (helipad_x1 + helipad_x2) / 2.0,
                Float64(LLConstants.HELIPAD_Y) + bar_height / 2.0,
            ),
            helipad_x2 - helipad_x1,
            bar_height,
            camera,
            helipad_color,
            centered=True,
        )

    def _draw_flags(
        mut self, env: Int, camera: Camera, mut renderer: Renderer2D
    ):
        """Draw helipad flags with poles using world coordinates."""
        var white_color = white()
        var yellow_color = yellow()
        var red_color = red()

        var W = Float64(LLConstants.VIEWPORT_W) / Float64(LLConstants.SCALE)

        # Compute helipad x positions
        var helipad_x1 = (
            W
            / Float64(LLConstants.TERRAIN_CHUNKS - 1)
            * Float64(LLConstants.TERRAIN_CHUNKS // 2 - 1)
        )
        var helipad_x2 = (
            W
            / Float64(LLConstants.TERRAIN_CHUNKS - 1)
            * Float64(LLConstants.TERRAIN_CHUNKS // 2 + 1)
        )

        # Flag dimensions in world units
        var pole_height = 50.0 / Float64(LLConstants.SCALE)
        var flag_width = 25.0 / Float64(LLConstants.SCALE)
        var flag_height = 20.0 / Float64(LLConstants.SCALE)

        for flag_idx in range(2):
            var x_pos = helipad_x1 if flag_idx == 0 else helipad_x2
            var ground_y = Float64(LLConstants.HELIPAD_Y)
            var pole_top_y = ground_y + pole_height

            # Flag pole (white vertical line)
            renderer.draw_line_world(
                RenderVec2(x_pos, ground_y),
                RenderVec2(x_pos, pole_top_y),
                camera,
                white_color,
                2,
            )

            # Flag as a filled triangle
            var flag_color = yellow_color if flag_idx == 0 else red_color
            var flag_verts = List[RenderVec2]()
            flag_verts.append(RenderVec2(x_pos, pole_top_y))
            flag_verts.append(
                RenderVec2(x_pos + flag_width, pole_top_y - flag_height / 2.0)
            )
            flag_verts.append(RenderVec2(x_pos, pole_top_y - flag_height))
            renderer.draw_polygon_world(
                flag_verts, camera, flag_color, filled=True
            )

    def _draw_lander(
        mut self, env: Int, camera: Camera, mut renderer: Renderer2D
    ):
        """Draw lander body as filled polygon using Transform2D."""
        # Get lander position and angle from physics
        var pos_x = Float64(self.physics.get_body_x(env, Self.BODY_LANDER))
        var pos_y = Float64(self.physics.get_body_y(env, Self.BODY_LANDER))
        var angle = Float64(self.physics.get_body_angle(env, Self.BODY_LANDER))

        # Use shape factory for lander body, scale from pixels to world units
        var lander_verts_raw = make_lander_body()
        var lander_verts = scale_vertices(
            lander_verts_raw^, 1.0 / Float64(LLConstants.SCALE)
        )

        # Create transform for lander position and rotation
        var transform = Transform2D(pos_x, pos_y, angle)

        # Draw filled lander body (grayish-white like the original)
        var lander_fill = rgb(230, 230, 230)
        var lander_outline = rgb(100, 100, 100)
        renderer.draw_transformed_polygon(
            lander_verts, transform, camera, lander_fill, filled=True
        )
        renderer.draw_transformed_polygon(
            lander_verts, transform, camera, lander_outline, filled=False
        )

    def _draw_legs(
        mut self, env: Int, camera: Camera, mut renderer: Renderer2D
    ):
        """Draw lander legs as filled polygons using Transform2D."""
        # Get leg contact from observation
        var obs = self.get_observation(env)
        var left_contact = Float64(obs[6]) > 0.5
        var right_contact = Float64(obs[7]) > 0.5

        for leg_idx in range(2):
            var body_idx = (
                Self.BODY_LEFT_LEG if leg_idx == 0 else Self.BODY_RIGHT_LEG
            )

            # Get leg position and angle from physics
            var pos_x = Float64(self.physics.get_body_x(env, body_idx))
            var pos_y = Float64(self.physics.get_body_y(env, body_idx))
            var angle = Float64(self.physics.get_body_angle(env, body_idx))

            # Color changes when leg touches ground (green = contact)
            var is_touching = left_contact if leg_idx == 0 else right_contact
            var leg_fill = contact_green() if is_touching else inactive_gray()
            var leg_outline = darken(leg_fill, 0.6)

            # Leg box vertices using shape factory (in world units)
            var leg_verts = make_leg_box(
                Float64(LLConstants.LEG_W) * 2.0,
                Float64(LLConstants.LEG_H) * 2.0,
            )

            # Create transform for leg position and rotation
            var transform = Transform2D(pos_x, pos_y, angle)

            # Draw filled leg
            renderer.draw_transformed_polygon(
                leg_verts, transform, camera, leg_fill, filled=True
            )
            renderer.draw_transformed_polygon(
                leg_verts, transform, camera, leg_outline, filled=False
            )

    def _draw_particles(mut self, camera: Camera, mut renderer: Renderer2D):
        """Draw engine flame particles."""
        for i in range(len(self.particles)):
            var p = self.particles[i]

            # Compute particle color based on TTL (fade from yellow/orange to red)
            var life_ratio = (
                Float64(p.ttl) / LLConstants.PARTICLE_TTL
            )  # 1.0 = just spawned, 0.0 = about to die

            # Color interpolation: yellow (255, 255, 0) -> orange (255, 128, 0) -> red (255, 0, 0)
            var r = UInt8(255)
            var g = UInt8(
                Int(255 * life_ratio * life_ratio)
            )  # Fade green faster
            var b = UInt8(Int(50 * life_ratio))  # Slight blue for hot particles
            var particle_color = SDL_Color(
                r, g, b, UInt8(Int(255 * life_ratio))
            )

            # Particle size based on TTL (shrink as they age)
            var size = 0.08 + 0.12 * life_ratio  # World units

            # Draw as small filled rectangle
            renderer.draw_rect_world(
                RenderVec2(Float64(p.x), Float64(p.y)),
                size,
                size,
                camera,
                particle_color,
                centered=True,
            )
