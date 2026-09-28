"""BipedalWalker GPU environment using the physics2d modular architecture.

This implementation uses the new modular physics components:
- BipedalWalkerLayout for compile-time layout computation
- PhysicsEnvHelpers for environment setup utilities
- Lidar for terrain sensing

The flat state layout is compatible with GPUContinuousEnv trait.
All physics data is packed per-environment for efficient GPU access.
"""

from std.math import sqrt, cos, sin, pi, tanh
from std.memory import alloc
from layout import Layout, LayoutTensor
from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceContext, DeviceBuffer
from std.random.philox import Random as PhiloxRandom

from noeira.core import (
    GPUContinuousEnv,
    BoxContinuousActionEnv,
    Action,
    RenderableEnv,
)
from noeira.render import (
    Renderer2D,
    SDL_Color,
    Camera,
    Vec2 as RenderVec2,
    Transform2D,
    # Colors
    sky_blue,
    grass_green,
    hull_purple,
    contact_green,
    inactive_gray,
    white,
    # Shapes
    make_rect,
)

from .state import BipedalWalkerState
from .action import BipedalWalkerAction
from .constants import BWConstants

from noeira.physics2d.integrators.euler import SemiImplicitEuler
from noeira.physics2d.collision.edge_terrain import EdgeTerrainCollision
from noeira.physics2d.solvers.impulse import ImpulseSolver
from noeira.physics2d.joints.revolute import RevoluteJointSolver
from noeira.physics2d.lidar import Lidar

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
    JOINT_MAX_MOTOR_TORQUE,
    JOINT_MOTOR_SPEED,
    JOINT_STIFFNESS,
    JOINT_DAMPING,
    JOINT_FLAGS,
    JOINT_FLAG_LIMIT_ENABLED,
    JOINT_FLAG_MOTOR_ENABLED,
    PhysicsState,
    PhysicsStateOwned,
    CONTACT_BODY_A,
    CONTACT_BODY_B,
    CONTACT_DEPTH,
    BipedalWalkerLayout,
    PhysicsEnvHelpers,
    PhysicsConfig,
)


# =============================================================================
# BipedalWalker Environment
# =============================================================================


struct BipedalWalker[
    DTYPE: DType,
](
    BoxContinuousActionEnv,
    Copyable,
    GPUContinuousEnv,
    Movable,
    RenderableEnv,
):
    """BipedalWalker  environment with GPU-compatible physics.

    Uses physics2d architecture for efficient batched simulation:
    - PhysicsState for accessing physics data in flat layout
    - Motor-enabled revolute joints for leg control
    - Edge terrain collision for ground contact
    - Lidar raycast for terrain sensing
    """

    # Required trait aliases
    comptime STATE_SIZE: Int = BWConstants.STATE_SIZE_VAL
    comptime OBS_DIM: Int = BWConstants.OBS_DIM_VAL
    comptime ACTION_DIM: Int = BWConstants.ACTION_DIM_VAL
    # Pre-allocated workspace sizes for step_kernel_gpu
    # Shared: shapes data (same across all envs)
    comptime STEP_WS_SHARED: Int = BWConstants.NUM_SHAPES * SHAPE_MAX_SIZE
    # Per-env: contacts + contact_count + edge_count + joint_count
    comptime STEP_WS_PER_ENV: Int = BWConstants.MAX_CONTACTS * CONTACT_DATA_SIZE + 3
    comptime dtype = Self.DTYPE
    comptime StateType = BipedalWalkerState[Self.dtype]
    comptime ActionType = BipedalWalkerAction[Self.dtype]
    comptime NAME: String = "BipedalWalker"

    # Body index constants
    comptime BODY_HULL: Int = BWConstants.BODY_HULL
    comptime BODY_UPPER_LEG_L: Int = BWConstants.BODY_UPPER_LEG_L
    comptime BODY_LOWER_LEG_L: Int = BWConstants.BODY_LOWER_LEG_L
    comptime BODY_UPPER_LEG_R: Int = BWConstants.BODY_UPPER_LEG_R
    comptime BODY_LOWER_LEG_R: Int = BWConstants.BODY_LOWER_LEG_R

    # Physics state for CPU single-env operation
    var physics: PhysicsStateOwned[
        BWConstants.NUM_BODIES,
        BWConstants.NUM_SHAPES,
        BWConstants.MAX_CONTACTS,
        BWConstants.MAX_JOINTS,
        BWConstants.STATE_SIZE_VAL,
        BWConstants.BODIES_OFFSET,
        BWConstants.FORCES_OFFSET,
        BWConstants.JOINTS_OFFSET,
        BWConstants.JOINT_COUNT_OFFSET,
        BWConstants.EDGES_OFFSET,
        BWConstants.EDGE_COUNT_OFFSET,
    ]
    var config: PhysicsConfig

    # Environment state
    var prev_shaping: Scalar[Self.dtype]
    var step_count: Int
    var game_over: Bool
    # Natural-termination flag for the LAST step (crawl/crash/OOB/success),
    # False on the time-limit truncation — read by `was_terminated()`.
    var last_terminated: Bool
    var scroll: Scalar[Self.dtype]
    var rng_seed: UInt64
    var rng_counter: UInt64

    # Ground contact tracking
    var left_leg_contact: Bool
    var right_leg_contact: Bool

    # Terrain data
    var terrain_x: List[Scalar[Self.dtype]]
    var terrain_y: List[Scalar[Self.dtype]]

    # Edge terrain collision system
    var edge_collision: EdgeTerrainCollision

    # Cached state for immutable get_state() access
    var cached_state: BipedalWalkerState[Self.dtype]

    # Renderer (RenderableEnv)
    var _renderer: Optional[Pointer[Renderer2D, MutUntrackedOrigin]]
    var _renderer_initialized: Bool

    # =========================================================================
    # Initialization
    # =========================================================================

    def __init__(out self, seed: UInt64 = 42):
        """Initialize the environment for CPU single-env operation."""
        # Create physics state for single environment
        self.physics = PhysicsStateOwned[
            BWConstants.NUM_BODIES,
            BWConstants.NUM_SHAPES,
            BWConstants.MAX_CONTACTS,
            BWConstants.MAX_JOINTS,
            BWConstants.STATE_SIZE_VAL,
            BWConstants.BODIES_OFFSET,
            BWConstants.FORCES_OFFSET,
            BWConstants.JOINTS_OFFSET,
            BWConstants.JOINT_COUNT_OFFSET,
            BWConstants.EDGES_OFFSET,
            BWConstants.EDGE_COUNT_OFFSET,
        ]()

        # Create physics config
        self.config = PhysicsConfig(
            gravity_x=BWConstants.GRAVITY_X,
            gravity_y=BWConstants.GRAVITY_Y,
            dt=BWConstants.DT,
            friction=BWConstants.FRICTION,
            restitution=BWConstants.RESTITUTION,
            baumgarte=BWConstants.BAUMGARTE,
            slop=BWConstants.SLOP,
            velocity_iterations=BWConstants.VELOCITY_ITERATIONS,
            position_iterations=BWConstants.POSITION_ITERATIONS,
        )

        # Initialize tracking variables
        self.prev_shaping = Scalar[Self.dtype](0)
        self.step_count = 0
        self.game_over = False
        self.last_terminated = False
        self.scroll = Scalar[Self.dtype](0)
        self.rng_seed = seed
        self.rng_counter = 0

        # Ground contact
        self.left_leg_contact = False
        self.right_leg_contact = False

        # Terrain data
        self.terrain_x = List[Scalar[Self.dtype]]()
        self.terrain_y = List[Scalar[Self.dtype]]()

        # Edge terrain collision system
        self.edge_collision = EdgeTerrainCollision(1)

        # Initialize cached state
        self.cached_state = BipedalWalkerState[Self.dtype]()

        # Renderer
        self._renderer = None
        self._renderer_initialized = False

        # Initialize physics shapes
        self._init_physics_shapes()

        # Reset to initial state
        self._reset_cpu()

    def __init__(out self, *, copy: Self):
        """Copy constructor."""
        self.physics = PhysicsStateOwned[
            BWConstants.NUM_BODIES,
            BWConstants.NUM_SHAPES,
            BWConstants.MAX_CONTACTS,
            BWConstants.MAX_JOINTS,
            BWConstants.STATE_SIZE_VAL,
            BWConstants.BODIES_OFFSET,
            BWConstants.FORCES_OFFSET,
            BWConstants.JOINTS_OFFSET,
            BWConstants.JOINT_COUNT_OFFSET,
            BWConstants.EDGES_OFFSET,
            BWConstants.EDGE_COUNT_OFFSET,
        ]()
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
        self.scroll = copy.scroll
        self.rng_seed = copy.rng_seed
        self.rng_counter = copy.rng_counter
        self.left_leg_contact = copy.left_leg_contact
        self.right_leg_contact = copy.right_leg_contact
        self.terrain_x = List[Scalar[Self.dtype]](copy.terrain_x)
        self.terrain_y = List[Scalar[Self.dtype]](copy.terrain_y)
        self.edge_collision = EdgeTerrainCollision(1)
        self.cached_state = copy.cached_state
        # Do not copy renderer — reset to null
        self._renderer = None
        self._renderer_initialized = False
        self._init_physics_shapes()
        self._reset_cpu()

    def __init__(out self, *, deinit move: Self):
        """Move constructor."""
        self.physics = PhysicsStateOwned[
            BWConstants.NUM_BODIES,
            BWConstants.NUM_SHAPES,
            BWConstants.MAX_CONTACTS,
            BWConstants.MAX_JOINTS,
            BWConstants.STATE_SIZE_VAL,
            BWConstants.BODIES_OFFSET,
            BWConstants.FORCES_OFFSET,
            BWConstants.JOINTS_OFFSET,
            BWConstants.JOINT_COUNT_OFFSET,
            BWConstants.EDGES_OFFSET,
            BWConstants.EDGE_COUNT_OFFSET,
        ]()
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
        self.scroll = move.scroll
        self.rng_seed = move.rng_seed
        self.rng_counter = move.rng_counter
        self.left_leg_contact = move.left_leg_contact
        self.right_leg_contact = move.right_leg_contact
        self.terrain_x = move.terrain_x^
        self.terrain_y = move.terrain_y^
        self.edge_collision = EdgeTerrainCollision(1)
        self.cached_state = move.cached_state
        # Transfer renderer ownership
        self._renderer = move._renderer
        self._renderer_initialized = move._renderer_initialized
        self._init_physics_shapes()
        self._reset_cpu()

    # =========================================================================
    # CPU Single-Environment Methods
    # =========================================================================

    def _init_physics_shapes(mut self):
        """Initialize physics shapes for hull and legs."""
        # Shape 0: Hull (pentagon)
        var hull_vx = List[Float64]()
        var hull_vy = List[Float64]()
        hull_vx.append(-30.0 / BWConstants.SCALE)
        hull_vy.append(0.0)
        hull_vx.append(-6.0 / BWConstants.SCALE)
        hull_vy.append(30.0 / BWConstants.SCALE)
        hull_vx.append(6.0 / BWConstants.SCALE)
        hull_vy.append(30.0 / BWConstants.SCALE)
        hull_vx.append(30.0 / BWConstants.SCALE)
        hull_vy.append(0.0)
        hull_vx.append(0.0)
        hull_vy.append(-12.0 / BWConstants.SCALE)
        self.physics.define_polygon_shape(0, hull_vx, hull_vy)

        # Shape 1: Upper leg left (rectangle)
        var ul_vx = List[Float64]()
        var ul_vy = List[Float64]()
        ul_vx.append(-BWConstants.UPPER_LEG_W / 2)
        ul_vy.append(BWConstants.UPPER_LEG_H / 2)
        ul_vx.append(-BWConstants.UPPER_LEG_W / 2)
        ul_vy.append(-BWConstants.UPPER_LEG_H / 2)
        ul_vx.append(BWConstants.UPPER_LEG_W / 2)
        ul_vy.append(-BWConstants.UPPER_LEG_H / 2)
        ul_vx.append(BWConstants.UPPER_LEG_W / 2)
        ul_vy.append(BWConstants.UPPER_LEG_H / 2)
        self.physics.define_polygon_shape(1, ul_vx, ul_vy)

        # Shape 2: Lower leg left (rectangle)
        var ll_vx = List[Float64]()
        var ll_vy = List[Float64]()
        ll_vx.append(-BWConstants.LOWER_LEG_W / 2)
        ll_vy.append(BWConstants.LOWER_LEG_H / 2)
        ll_vx.append(-BWConstants.LOWER_LEG_W / 2)
        ll_vy.append(-BWConstants.LOWER_LEG_H / 2)
        ll_vx.append(BWConstants.LOWER_LEG_W / 2)
        ll_vy.append(-BWConstants.LOWER_LEG_H / 2)
        ll_vx.append(BWConstants.LOWER_LEG_W / 2)
        ll_vy.append(BWConstants.LOWER_LEG_H / 2)
        self.physics.define_polygon_shape(2, ll_vx, ll_vy)

        # Shapes 3, 4: Same as 1, 2 for right leg
        self.physics.define_polygon_shape(3, ul_vx, ul_vy)
        self.physics.define_polygon_shape(4, ll_vx, ll_vy)

    def _reset_cpu(mut self):
        """Internal reset for CPU single-env operation."""
        self.rng_counter += 1
        var combined_seed = (
            Int(self.rng_seed) * 2654435761 + Int(self.rng_counter) * 12345
        )
        var rng = PhiloxRandom(seed=UInt64(combined_seed), offset=0)

        # Generate terrain
        self._generate_terrain_cpu(Int(combined_seed))

        # Create walker
        var init_x = Float64(BWConstants.TERRAIN_STARTPAD) * Float64(
            BWConstants.TERRAIN_STEP
        )
        var init_y = BWConstants.TERRAIN_HEIGHT + 2.0 * BWConstants.LEG_H
        self._create_walker_cpu(init_x, init_y, rng)

        # Reset state
        self.step_count = 0
        self.game_over = False
        self.last_terminated = False
        self.scroll = Scalar[Self.dtype](0)
        self.left_leg_contact = False
        self.right_leg_contact = False

        # Compute initial shaping
        self._update_cached_state()
        self.prev_shaping = self._compute_shaping()

    def _generate_terrain_cpu(mut self, seed: Int):
        """Generate terrain with smooth random variation."""
        self.terrain_x.clear()
        self.terrain_y.clear()

        var terrain_rng = PhiloxRandom(seed=UInt64(seed) + 1000, offset=0)

        var y = Scalar[Self.dtype](BWConstants.TERRAIN_HEIGHT)
        var velocity = Scalar[Self.dtype](0.0)

        for i in range(BWConstants.TERRAIN_LENGTH):
            var x = Scalar[Self.dtype](i) * Scalar[Self.dtype](
                BWConstants.TERRAIN_STEP
            )
            self.terrain_x.append(x)

            # Smooth random variation
            var terrain_height = Scalar[Self.dtype](BWConstants.TERRAIN_HEIGHT)
            velocity = Scalar[Self.dtype](0.8) * velocity + Scalar[Self.dtype](
                0.01
            ) * (
                Scalar[Self.dtype](1.0) if terrain_height
                > y else Scalar[Self.dtype](-1.0)
            )
            if i > BWConstants.TERRAIN_STARTPAD:
                var rand_vals = terrain_rng.step_uniform()
                velocity = velocity + (
                    Scalar[Self.dtype](rand_vals[0]) * Scalar[Self.dtype](2.0)
                    - Scalar[Self.dtype](1.0)
                ) / Scalar[Self.dtype](BWConstants.SCALE)

            y = y + velocity
            self.terrain_y.append(y)

        # Set up edge terrain
        var n_edges = min(
            BWConstants.TERRAIN_LENGTH - 1, BWConstants.MAX_TERRAIN_EDGES
        )
        var state = self.physics.get_state_tensor()
        state[0, BWConstants.EDGE_COUNT_OFFSET] = Scalar[dtype](n_edges)

        # Update edge collision internal count for CPU collision detection
        self.edge_collision.edge_counts[0] = n_edges

        for i in range(n_edges):
            var edge_off = BWConstants.EDGES_OFFSET + i * 6
            var x0 = rebind[Scalar[dtype]](self.terrain_x[i])
            var y0 = rebind[Scalar[dtype]](self.terrain_y[i])
            var x1 = rebind[Scalar[dtype]](self.terrain_x[i + 1])
            var y1 = rebind[Scalar[dtype]](self.terrain_y[i + 1])

            # Compute normal
            var dx = x1 - x0
            var dy = y1 - y0
            var length = sqrt(dx * dx + dy * dy)
            var nx = -dy / length
            var ny = dx / length
            if ny < Scalar[dtype](0):
                nx = -nx
                ny = -ny

            # Write to state tensor (for GPU)
            state[0, edge_off + 0] = x0
            state[0, edge_off + 1] = y0
            state[0, edge_off + 2] = x1
            state[0, edge_off + 3] = y1
            state[0, edge_off + 4] = nx
            state[0, edge_off + 5] = ny

            # Also update edge collision internal storage (for CPU collision detection)
            var collision_edge_off = i * 6  # env 0, so no env offset needed
            self.edge_collision.edges[collision_edge_off + 0] = x0
            self.edge_collision.edges[collision_edge_off + 1] = y0
            self.edge_collision.edges[collision_edge_off + 2] = x1
            self.edge_collision.edges[collision_edge_off + 3] = y1
            self.edge_collision.edges[collision_edge_off + 4] = nx
            self.edge_collision.edges[collision_edge_off + 5] = ny

    def _create_walker_cpu(
        mut self, init_x: Float64, init_y: Float64, mut rng: PhiloxRandom
    ):
        """Create the bipedal walker bodies and joints."""
        var rand_vals = rng.step_uniform()
        var init_vx = (Float64(rand_vals[0]) * 2.0 - 1.0) * 0.1
        var init_vy = (Float64(rand_vals[1]) * 2.0 - 1.0) * 0.1

        var state = self.physics.get_state_tensor()

        # Hull (body 0)
        var hull_off = BWConstants.BODIES_OFFSET
        state[0, hull_off + IDX_X] = Scalar[dtype](init_x)
        state[0, hull_off + IDX_Y] = Scalar[dtype](init_y)
        state[0, hull_off + IDX_ANGLE] = Scalar[dtype](0)
        state[0, hull_off + IDX_VX] = Scalar[dtype](init_vx)
        state[0, hull_off + IDX_VY] = Scalar[dtype](init_vy)
        state[0, hull_off + IDX_OMEGA] = Scalar[dtype](0)
        state[0, hull_off + IDX_INV_MASS] = Scalar[dtype](
            1.0 / BWConstants.HULL_MASS
        )
        state[0, hull_off + IDX_INV_INERTIA] = Scalar[dtype](
            1.0 / BWConstants.HULL_INERTIA
        )
        state[0, hull_off + IDX_SHAPE] = Scalar[dtype](0)

        # Create legs
        for leg in range(2):
            var sign = Scalar[dtype](-1.0) if leg == 0 else Scalar[dtype](1.0)

            # Upper leg
            var upper_off = (
                BWConstants.BODIES_OFFSET + (leg * 2 + 1) * BODY_STATE_SIZE
            )
            var upper_x = init_x
            var upper_y = (
                init_y + BWConstants.LEG_DOWN - BWConstants.UPPER_LEG_H / 2
            )
            state[0, upper_off + IDX_X] = Scalar[dtype](upper_x)
            state[0, upper_off + IDX_Y] = Scalar[dtype](upper_y)
            state[0, upper_off + IDX_ANGLE] = Scalar[dtype](0)
            state[0, upper_off + IDX_VX] = Scalar[dtype](init_vx)
            state[0, upper_off + IDX_VY] = Scalar[dtype](init_vy)
            state[0, upper_off + IDX_OMEGA] = Scalar[dtype](0)
            state[0, upper_off + IDX_INV_MASS] = Scalar[dtype](
                1.0 / BWConstants.LEG_MASS
            )
            state[0, upper_off + IDX_INV_INERTIA] = Scalar[dtype](
                1.0 / BWConstants.LEG_INERTIA
            )
            state[0, upper_off + IDX_SHAPE] = Scalar[dtype](leg * 2 + 1)

            # Lower leg
            var lower_off = (
                BWConstants.BODIES_OFFSET + (leg * 2 + 2) * BODY_STATE_SIZE
            )
            var lower_x = init_x
            var lower_y = (
                upper_y
                - BWConstants.UPPER_LEG_H / 2
                - BWConstants.LOWER_LEG_H / 2
            )
            state[0, lower_off + IDX_X] = Scalar[dtype](lower_x)
            state[0, lower_off + IDX_Y] = Scalar[dtype](lower_y)
            state[0, lower_off + IDX_ANGLE] = Scalar[dtype](0)
            state[0, lower_off + IDX_VX] = Scalar[dtype](init_vx)
            state[0, lower_off + IDX_VY] = Scalar[dtype](init_vy)
            state[0, lower_off + IDX_OMEGA] = Scalar[dtype](0)
            state[0, lower_off + IDX_INV_MASS] = Scalar[dtype](
                1.0 / BWConstants.LOWER_LEG_MASS
            )
            state[0, lower_off + IDX_INV_INERTIA] = Scalar[dtype](
                1.0 / BWConstants.LOWER_LEG_INERTIA
            )
            state[0, lower_off + IDX_SHAPE] = Scalar[dtype](leg * 2 + 2)

        # Create joints
        state[0, BWConstants.JOINT_COUNT_OFFSET] = Scalar[dtype](4)

        for leg in range(2):
            # Hip joint (hull to upper leg)
            var hip_off = (
                BWConstants.JOINTS_OFFSET + (leg * 2) * JOINT_DATA_SIZE
            )
            state[0, hip_off + JOINT_TYPE] = Scalar[dtype](JOINT_REVOLUTE)
            state[0, hip_off + JOINT_BODY_A] = Scalar[dtype](0)  # Hull
            state[0, hip_off + JOINT_BODY_B] = Scalar[dtype](
                leg * 2 + 1
            )  # Upper leg
            state[0, hip_off + JOINT_ANCHOR_AX] = Scalar[dtype](0)
            state[0, hip_off + JOINT_ANCHOR_AY] = Scalar[dtype](
                BWConstants.LEG_DOWN
            )
            state[0, hip_off + JOINT_ANCHOR_BX] = Scalar[dtype](0)
            state[0, hip_off + JOINT_ANCHOR_BY] = Scalar[dtype](
                BWConstants.UPPER_LEG_H / 2
            )
            state[0, hip_off + JOINT_REF_ANGLE] = Scalar[dtype](0)
            state[0, hip_off + JOINT_LOWER_LIMIT] = Scalar[dtype](
                BWConstants.HIP_LIMIT_LOW
            )
            state[0, hip_off + JOINT_UPPER_LIMIT] = Scalar[dtype](
                BWConstants.HIP_LIMIT_HIGH
            )
            state[0, hip_off + JOINT_MAX_MOTOR_TORQUE] = Scalar[dtype](
                BWConstants.MOTORS_TORQUE
            )
            state[0, hip_off + JOINT_MOTOR_SPEED] = Scalar[dtype](0)
            state[0, hip_off + JOINT_FLAGS] = Scalar[dtype](
                JOINT_FLAG_LIMIT_ENABLED | JOINT_FLAG_MOTOR_ENABLED
            )
            RevoluteJointSolver.clear_warm_start[1, BWConstants.STATE_SIZE_VAL](
                state, 0, hip_off
            )

            # Knee joint (upper leg to lower leg)
            var knee_off = (
                BWConstants.JOINTS_OFFSET + (leg * 2 + 1) * JOINT_DATA_SIZE
            )
            state[0, knee_off + JOINT_TYPE] = Scalar[dtype](JOINT_REVOLUTE)
            state[0, knee_off + JOINT_BODY_A] = Scalar[dtype](
                leg * 2 + 1
            )  # Upper leg
            state[0, knee_off + JOINT_BODY_B] = Scalar[dtype](
                leg * 2 + 2
            )  # Lower leg
            state[0, knee_off + JOINT_ANCHOR_AX] = Scalar[dtype](0)
            state[0, knee_off + JOINT_ANCHOR_AY] = Scalar[dtype](
                -BWConstants.UPPER_LEG_H / 2
            )
            state[0, knee_off + JOINT_ANCHOR_BX] = Scalar[dtype](0)
            state[0, knee_off + JOINT_ANCHOR_BY] = Scalar[dtype](
                BWConstants.LOWER_LEG_H / 2
            )
            state[0, knee_off + JOINT_REF_ANGLE] = Scalar[dtype](0)
            state[0, knee_off + JOINT_LOWER_LIMIT] = Scalar[dtype](
                BWConstants.KNEE_LIMIT_LOW
            )
            state[0, knee_off + JOINT_UPPER_LIMIT] = Scalar[dtype](
                BWConstants.KNEE_LIMIT_HIGH
            )
            state[0, knee_off + JOINT_MAX_MOTOR_TORQUE] = Scalar[dtype](
                BWConstants.MOTORS_TORQUE
            )
            state[0, knee_off + JOINT_MOTOR_SPEED] = Scalar[dtype](0)
            state[0, knee_off + JOINT_FLAGS] = Scalar[dtype](
                JOINT_FLAG_LIMIT_ENABLED | JOINT_FLAG_MOTOR_ENABLED
            )
            RevoluteJointSolver.clear_warm_start[1, BWConstants.STATE_SIZE_VAL](
                state, 0, knee_off
            )

        # Clear forces
        for body in range(BWConstants.NUM_BODIES):
            var force_off = BWConstants.FORCES_OFFSET + body * 3
            state[0, force_off + 0] = Scalar[dtype](0)
            state[0, force_off + 1] = Scalar[dtype](0)
            state[0, force_off + 2] = Scalar[dtype](0)

    def _compute_shaping(mut self) -> Scalar[Self.dtype]:
        """Compute potential-based shaping reward."""
        var state = self.physics.get_state_tensor()
        var hull_x = Float64(
            rebind[Scalar[dtype]](state[0, BWConstants.BODIES_OFFSET + IDX_X])
        )

        # Forward progress is the main shaping
        return Scalar[Self.dtype](130.0 * hull_x / BWConstants.SCALE)

    def _update_cached_state(mut self):
        """Update cached observation state."""
        var state = self.physics.get_state_tensor()

        # Hull state
        var hull_angle = Float64(
            rebind[Scalar[dtype]](
                state[0, BWConstants.BODIES_OFFSET + IDX_ANGLE]
            )
        )
        var hull_omega = Float64(
            rebind[Scalar[dtype]](
                state[0, BWConstants.BODIES_OFFSET + IDX_OMEGA]
            )
        )
        var hull_vx = Float64(
            rebind[Scalar[dtype]](state[0, BWConstants.BODIES_OFFSET + IDX_VX])
        )
        var hull_vy = Float64(
            rebind[Scalar[dtype]](state[0, BWConstants.BODIES_OFFSET + IDX_VY])
        )

        self.cached_state.hull_angle = Scalar[Self.dtype](hull_angle)
        self.cached_state.hull_angular_velocity = Scalar[Self.dtype](
            hull_omega / 5.0
        )  # Normalized
        self.cached_state.vel_x = Scalar[Self.dtype](
            hull_vx
            * (BWConstants.VIEWPORT_W / BWConstants.SCALE)
            / BWConstants.FPS
        )
        self.cached_state.vel_y = Scalar[Self.dtype](
            hull_vy
            * (BWConstants.VIEWPORT_H / BWConstants.SCALE)
            / BWConstants.FPS
        )

        # Leg 1 (left) state
        var hip1_off = BWConstants.JOINTS_OFFSET + 0 * JOINT_DATA_SIZE
        var knee1_off = BWConstants.JOINTS_OFFSET + 1 * JOINT_DATA_SIZE
        var upper1_off = BWConstants.BODIES_OFFSET + 1 * BODY_STATE_SIZE
        var lower1_off = BWConstants.BODIES_OFFSET + 2 * BODY_STATE_SIZE

        var hull_angle_state = Float64(
            rebind[Scalar[dtype]](
                state[0, BWConstants.BODIES_OFFSET + IDX_ANGLE]
            )
        )
        var upper1_angle = Float64(
            rebind[Scalar[dtype]](state[0, upper1_off + IDX_ANGLE])
        )
        var lower1_angle = Float64(
            rebind[Scalar[dtype]](state[0, lower1_off + IDX_ANGLE])
        )
        var upper1_omega = Float64(
            rebind[Scalar[dtype]](state[0, upper1_off + IDX_OMEGA])
        )
        var lower1_omega = Float64(
            rebind[Scalar[dtype]](state[0, lower1_off + IDX_OMEGA])
        )

        self.cached_state.hip1_angle = Scalar[Self.dtype](
            (upper1_angle - hull_angle_state) / 1.0
        )
        self.cached_state.hip1_speed = Scalar[Self.dtype](
            (upper1_omega - hull_omega) / 10.0
        )
        self.cached_state.knee1_angle = Scalar[Self.dtype](
            (lower1_angle - upper1_angle) / 1.0
        )
        self.cached_state.knee1_speed = Scalar[Self.dtype](
            (lower1_omega - upper1_omega) / 10.0
        )
        self.cached_state.leg1_contact = Scalar[Self.dtype](
            1.0
        ) if self.left_leg_contact else Scalar[Self.dtype](0.0)

        # Leg 2 (right) state
        var hip2_off = BWConstants.JOINTS_OFFSET + 2 * JOINT_DATA_SIZE
        var knee2_off = BWConstants.JOINTS_OFFSET + 3 * JOINT_DATA_SIZE
        var upper2_off = BWConstants.BODIES_OFFSET + 3 * BODY_STATE_SIZE
        var lower2_off = BWConstants.BODIES_OFFSET + 4 * BODY_STATE_SIZE

        var upper2_angle = Float64(
            rebind[Scalar[dtype]](state[0, upper2_off + IDX_ANGLE])
        )
        var lower2_angle = Float64(
            rebind[Scalar[dtype]](state[0, lower2_off + IDX_ANGLE])
        )
        var upper2_omega = Float64(
            rebind[Scalar[dtype]](state[0, upper2_off + IDX_OMEGA])
        )
        var lower2_omega = Float64(
            rebind[Scalar[dtype]](state[0, lower2_off + IDX_OMEGA])
        )

        self.cached_state.hip2_angle = Scalar[Self.dtype](
            (upper2_angle - hull_angle_state) / 1.0
        )
        self.cached_state.hip2_speed = Scalar[Self.dtype](
            (upper2_omega - hull_omega) / 10.0
        )
        self.cached_state.knee2_angle = Scalar[Self.dtype](
            (lower2_angle - upper2_angle) / 1.0
        )
        self.cached_state.knee2_speed = Scalar[Self.dtype](
            (lower2_omega - upper2_omega) / 10.0
        )
        self.cached_state.leg2_contact = Scalar[Self.dtype](
            1.0
        ) if self.right_leg_contact else Scalar[Self.dtype](0.0)

        # Lidar raycast
        var hull_x = rebind[Scalar[dtype]](
            state[0, BWConstants.BODIES_OFFSET + IDX_X]
        )
        var hull_y = rebind[Scalar[dtype]](
            state[0, BWConstants.BODIES_OFFSET + IDX_Y]
        )
        var hull_angle_rad = rebind[Scalar[dtype]](
            state[0, BWConstants.BODIES_OFFSET + IDX_ANGLE]
        )
        var n_edges = Int(state[0, BWConstants.EDGE_COUNT_OFFSET])
        var lidar_range = Scalar[dtype](BWConstants.LIDAR_RANGE)

        for i in range(BWConstants.NUM_LIDAR):
            # Angle relative to hull: 0 to 1.5 radians (looking down/forward)
            var local_angle = (
                Scalar[dtype](i)
                * Scalar[dtype](1.5)
                / Scalar[dtype](BWConstants.NUM_LIDAR - 1)
            )
            var world_angle = (
                hull_angle_rad - local_angle - Scalar[dtype](pi / 2.0)
            )

            # Ray direction
            var ray_dx = cos(world_angle) * lidar_range
            var ray_dy = sin(world_angle) * lidar_range

            # Find minimum hit distance
            var min_t = Scalar[dtype](1.0)  # Default to max range (no hit)

            for edge in range(n_edges):
                if edge >= BWConstants.MAX_TERRAIN_EDGES:
                    break

                var edge_off = BWConstants.EDGES_OFFSET + edge * 6
                var edge_x0 = rebind[Scalar[dtype]](state[0, edge_off + 0])
                var edge_y0 = rebind[Scalar[dtype]](state[0, edge_off + 1])
                var edge_x1 = rebind[Scalar[dtype]](state[0, edge_off + 2])
                var edge_y1 = rebind[Scalar[dtype]](state[0, edge_off + 3])

                # Ray-edge intersection (parametric)
                var ex = edge_x1 - edge_x0
                var ey = edge_y1 - edge_y0
                var denom = ray_dx * ey - ray_dy * ex

                if denom > Scalar[dtype](-1e-10) and denom < Scalar[dtype](
                    1e-10
                ):
                    continue  # Parallel

                var dx_ray = hull_x - edge_x0
                var dy_ray = hull_y - edge_y0
                var t = (ex * dy_ray - ey * dx_ray) / denom  # Along ray
                var u = (
                    ray_dx * dy_ray - ray_dy * dx_ray
                ) / denom  # Along edge

                if (
                    t >= Scalar[dtype](0.0)
                    and t <= Scalar[dtype](1.0)
                    and u >= Scalar[dtype](0.0)
                    and u <= Scalar[dtype](1.0)
                ):
                    if t < min_t:
                        min_t = t

            self.cached_state.lidar[i] = Scalar[Self.dtype](min_t)

    def _update_ground_contacts(mut self):
        """Update ground contact flags based on collision detection."""
        var contacts = self.physics.get_contacts_tensor()
        var contact_counts = self.physics.get_contact_counts_tensor()
        var n_contacts = Int(contact_counts[0])

        self.left_leg_contact = False
        self.right_leg_contact = False

        for i in range(n_contacts):
            var body_a = Int(contacts[0, i, CONTACT_BODY_A])
            # Lower legs are bodies 2 (left) and 4 (right)
            if body_a == Self.BODY_LOWER_LEG_L:
                self.left_leg_contact = True
            elif body_a == Self.BODY_LOWER_LEG_R:
                self.right_leg_contact = True

    def _check_hull_contact(mut self):
        """Check if hull is in contact with ground (game over condition)."""
        var contacts = self.physics.get_contacts_tensor()
        var contact_counts = self.physics.get_contact_counts_tensor()
        var n_contacts = Int(contact_counts[0])

        for i in range(n_contacts):
            var body_a = Int(contacts[0, i, CONTACT_BODY_A])
            if body_a == Self.BODY_HULL:
                self.game_over = True
                return

    def _step_physics_cpu(mut self):
        """Execute physics step: detection, then the SAME Box2D island solve
        the GPU kernel runs (`_solve_step_single_env`)."""
        var bodies = self.physics.get_bodies_tensor()
        var shapes = self.physics.get_shapes_tensor()
        var forces = self.physics.get_forces_tensor()
        var contacts = self.physics.get_contacts_tensor()
        var contact_counts = self.physics.get_contact_counts_tensor()
        var states = self.physics.get_state_tensor()

        self.edge_collision.detect[
            1,
            BWConstants.NUM_BODIES,
            BWConstants.NUM_SHAPES,
            BWConstants.MAX_CONTACTS,
        ](bodies, shapes, contacts, contact_counts)

        Self._solve_step_single_env[1, BWConstants.STATE_SIZE_VAL](
            0,
            states,
            contacts,
            Int(contact_counts[0]),
            Int(states[0, BWConstants.JOINT_COUNT_OFFSET]),
            Scalar[dtype](self.config.gravity_x),
            Scalar[dtype](self.config.gravity_y),
            Scalar[dtype](self.config.dt),
        )

        # Clear forces
        for body in range(BWConstants.NUM_BODIES):
            forces[0, body, 0] = Scalar[dtype](0)
            forces[0, body, 1] = Scalar[dtype](0)
            forces[0, body, 2] = Scalar[dtype](0)

    # =========================================================================
    # BoxContinuousActionEnv Trait Methods
    # =========================================================================

    def reset(mut self) -> Self.StateType:
        """Reset the environment and return initial state."""
        self._reset_cpu()
        return self.get_state()

    def step(
        mut self, action: Self.ActionType, verbose: Bool = False
    ) -> Tuple[Self.StateType, Scalar[Self.dtype], Bool]:
        """Take an action and return (next_state, reward, done)."""
        var result = self._step_cpu_continuous(action)
        return (self.get_state(), result[0], result[1])

    def _step_cpu_continuous(
        mut self, action: BipedalWalkerAction[Self.dtype]
    ) -> Tuple[Scalar[Self.dtype], Bool]:
        """Internal CPU step with continuous action."""
        var state = self.physics.get_state_tensor()

        # Apply motor actions to joints
        # Hip 1 (left)
        var hip1_off = BWConstants.JOINTS_OFFSET + 0 * JOINT_DATA_SIZE
        var hip1_action = action.hip1
        if hip1_action > Scalar[Self.dtype](1.0):
            hip1_action = Scalar[Self.dtype](1.0)
        if hip1_action < Scalar[Self.dtype](-1.0):
            hip1_action = Scalar[Self.dtype](-1.0)
        var hip1_sign = Scalar[dtype](1.0) if hip1_action >= Scalar[Self.dtype](
            0
        ) else Scalar[dtype](-1.0)
        state[0, hip1_off + JOINT_MOTOR_SPEED] = hip1_sign * Scalar[dtype](
            BWConstants.SPEED_HIP
        )
        state[0, hip1_off + JOINT_MAX_MOTOR_TORQUE] = Scalar[dtype](
            BWConstants.MOTORS_TORQUE
        ) * Scalar[dtype](abs(hip1_action))

        # Knee 1 (left)
        var knee1_off = BWConstants.JOINTS_OFFSET + 1 * JOINT_DATA_SIZE
        var knee1_action = action.knee1
        if knee1_action > Scalar[Self.dtype](1.0):
            knee1_action = Scalar[Self.dtype](1.0)
        if knee1_action < Scalar[Self.dtype](-1.0):
            knee1_action = Scalar[Self.dtype](-1.0)
        var knee1_sign = Scalar[dtype](1.0) if knee1_action >= Scalar[
            Self.dtype
        ](0) else Scalar[dtype](-1.0)
        state[0, knee1_off + JOINT_MOTOR_SPEED] = knee1_sign * Scalar[dtype](
            BWConstants.SPEED_KNEE
        )
        state[0, knee1_off + JOINT_MAX_MOTOR_TORQUE] = Scalar[dtype](
            BWConstants.MOTORS_TORQUE
        ) * Scalar[dtype](abs(knee1_action))

        # Hip 2 (right)
        var hip2_off = BWConstants.JOINTS_OFFSET + 2 * JOINT_DATA_SIZE
        var hip2_action = action.hip2
        if hip2_action > Scalar[Self.dtype](1.0):
            hip2_action = Scalar[Self.dtype](1.0)
        if hip2_action < Scalar[Self.dtype](-1.0):
            hip2_action = Scalar[Self.dtype](-1.0)
        var hip2_sign = Scalar[dtype](1.0) if hip2_action >= Scalar[Self.dtype](
            0
        ) else Scalar[dtype](-1.0)
        state[0, hip2_off + JOINT_MOTOR_SPEED] = hip2_sign * Scalar[dtype](
            BWConstants.SPEED_HIP
        )
        state[0, hip2_off + JOINT_MAX_MOTOR_TORQUE] = Scalar[dtype](
            BWConstants.MOTORS_TORQUE
        ) * Scalar[dtype](abs(hip2_action))

        # Knee 2 (right)
        var knee2_off = BWConstants.JOINTS_OFFSET + 3 * JOINT_DATA_SIZE
        var knee2_action = action.knee2
        if knee2_action > Scalar[Self.dtype](1.0):
            knee2_action = Scalar[Self.dtype](1.0)
        if knee2_action < Scalar[Self.dtype](-1.0):
            knee2_action = Scalar[Self.dtype](-1.0)
        var knee2_sign = Scalar[dtype](1.0) if knee2_action >= Scalar[
            Self.dtype
        ](0) else Scalar[dtype](-1.0)
        state[0, knee2_off + JOINT_MOTOR_SPEED] = knee2_sign * Scalar[dtype](
            BWConstants.SPEED_KNEE
        )
        state[0, knee2_off + JOINT_MAX_MOTOR_TORQUE] = Scalar[dtype](
            BWConstants.MOTORS_TORQUE
        ) * Scalar[dtype](abs(knee2_action))

        # Physics step
        self._step_physics_cpu()

        # Update contacts
        self._update_ground_contacts()
        self._check_hull_contact()

        # Update scroll for rendering
        var hull_x_scroll = self.physics.get_body_x(0, Self.BODY_HULL)
        self.scroll = Scalar[Self.dtype](hull_x_scroll) - Scalar[Self.dtype](
            BWConstants.VIEWPORT_W / BWConstants.SCALE / 5.0
        )

        # Update cached state
        self._update_cached_state()

        # Compute reward
        return self._compute_step_result(action)

    def _compute_step_result(
        mut self, action: BipedalWalkerAction[Self.dtype]
    ) -> Tuple[Scalar[Self.dtype], Bool]:
        """Compute reward and termination."""
        self.step_count += 1

        var hull_x = Float64(self.physics.get_body_x(0, Self.BODY_HULL))
        var hull_y = Float64(self.physics.get_body_y(0, Self.BODY_HULL))
        var hull_angle = Float64(self.physics.get_body_angle(0, Self.BODY_HULL))

        # Shaping reward (forward progress)
        var new_shaping = self._compute_shaping()
        var reward = new_shaping - self.prev_shaping
        self.prev_shaping = new_shaping

        # Angle penalty
        reward = reward - Scalar[Self.dtype](5.0 * abs(hull_angle))

        # Energy penalty
        var energy = (
            abs(action.hip1)
            + abs(action.knee1)
            + abs(action.hip2)
            + abs(action.knee2)
        )
        reward = (
            reward
            - Scalar[Self.dtype](0.00035 * BWConstants.MOTORS_TORQUE) * energy
        )

        # Height penalty: strongly penalize crawling
        # Standing height is ~TERRAIN_HEIGHT + 2*LEG_H (~5.6)
        var min_height = (
            BWConstants.TERRAIN_HEIGHT + BWConstants.LEG_H * 1.5
        )  # ~5.0, penalty starts
        var critical_height = (
            BWConstants.TERRAIN_HEIGHT + BWConstants.LEG_H * 1.2
        )  # ~4.7, terminate below
        if hull_y < min_height:
            # Strong penalty for being too low (crawling)
            reward = reward - Scalar[Self.dtype](10.0 * (min_height - hull_y))

        # Velocity shaping: reward forward movement, penalize backward
        var vel_x = Float64(self.cached_state.vel_x)
        if vel_x > 0.0:
            # Bonus for moving forward - encourages actual walking
            reward = reward + Scalar[Self.dtype](3.0 * vel_x)
        else:
            # Penalty for going backward
            reward = reward - Scalar[Self.dtype](5.0 * abs(vel_x))

        var terminated = False

        # Terminate if crawling too low (hull almost touching ground)
        if hull_y < critical_height:
            reward = Scalar[Self.dtype](
                -50.0
            )  # Moderate penalty for crawling termination
            terminated = True

        # Game over: hull touched ground
        if self.game_over:
            reward = Scalar[Self.dtype](BWConstants.CRASH_PENALTY)
            terminated = True

        # Out of bounds (fell off left edge)
        if hull_x < 0.0:
            reward = Scalar[Self.dtype](BWConstants.CRASH_PENALTY)
            terminated = True

        # Success: reached end of terrain
        var terrain_end = (
            Float64(BWConstants.TERRAIN_LENGTH - BWConstants.TERRAIN_GRASS)
            * BWConstants.TERRAIN_STEP
        )
        if hull_x > terrain_end:
            terminated = True

        # Natural termination recorded BEFORE folding in the time limit,
        # so `was_terminated()` distinguishes the two — the GPU kernel
        # already reports them separately via its terminated buffer.
        self.last_terminated = terminated

        # Time limit
        if self.step_count >= 2000:
            terminated = True

        return (reward, terminated)

    def get_state(mut self) -> Self.StateType:
        """Return current state representation."""
        return self.cached_state

    def was_terminated(self) -> Bool:
        """True iff the previous step ended by natural termination
        (crawl/crash/out-of-bounds/success), False on the 2000-step
        truncation. Without this override the base default (always False)
        made drivers bootstrap through real crashes on the CPU path."""
        return self.last_terminated

    def get_obs_list(self) -> List[Scalar[Self.dtype]]:
        """Return current observation as a list."""
        return self.cached_state.to_list()

    def reset_obs_list(mut self) -> List[Scalar[Self.dtype]]:
        """Reset and return initial observation."""
        var state = self.reset()
        return state.to_list()

    def obs_dim(self) -> Int:
        """Return observation dimension."""
        return BWConstants.OBS_DIM_VAL

    def action_dim(self) -> Int:
        """Return action dimension."""
        return BWConstants.ACTION_DIM_VAL

    def action_low(self) -> Scalar[Self.dtype]:
        """Return minimum action value."""
        return Scalar[Self.dtype](-1.0)

    def action_high(self) -> Scalar[Self.dtype]:
        """Return maximum action value."""
        return Scalar[Self.dtype](1.0)

    def step_continuous[
        DTYPE_SC: DType
    ](mut self, action: Scalar[DTYPE_SC]) -> Tuple[
        List[Scalar[DTYPE_SC]], Scalar[DTYPE_SC], Bool
    ]:
        """Step with single scalar action (applied to all joints)."""
        var a = Scalar[Self.dtype](action)
        var act = BipedalWalkerAction[Self.dtype](a, a, a, a)
        var result = self._step_cpu_continuous(act)
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
        """Step with vector action."""
        var hip1 = Scalar[Self.dtype](action[0]) if len(action) > 0 else Scalar[
            Self.dtype
        ](0)
        var knee1 = Scalar[Self.dtype](action[1]) if len(
            action
        ) > 1 else Scalar[Self.dtype](0)
        var hip2 = Scalar[Self.dtype](action[2]) if len(action) > 2 else Scalar[
            Self.dtype
        ](0)
        var knee2 = Scalar[Self.dtype](action[3]) if len(
            action
        ) > 3 else Scalar[Self.dtype](0)
        var act = BipedalWalkerAction[Self.dtype](hip1, knee1, hip2, knee2)
        var result = self._step_cpu_continuous(act)

        var obs = self.cached_state.to_list_typed[DTYPE_VEC]()
        return (obs^, Scalar[DTYPE_VEC](result[0]), result[1])

    def render(mut self, mut renderer: Renderer2D):
        """Render the environment.

        Args:
            renderer: External renderer to use for drawing.
        """
        self._render_internal(renderer)

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
    # Rendering Methods
    # =========================================================================

    def _render_internal(mut self, mut renderer: Renderer2D):
        """Render the environment with scrolling Camera.

        Args:
            renderer: External renderer to use for drawing.
        """
        # Begin frame with sky background
        if not renderer.begin_frame_with_color(sky_blue()):
            return

        # Create scrolling camera that follows the walker
        # Camera X = scroll position, Y centered vertically
        var cam_y = (
            Float64(BWConstants.VIEWPORT_H) / Float64(BWConstants.SCALE) / 2.0
        )
        var camera = Camera(
            Float64(self.scroll)
            + Float64(BWConstants.VIEWPORT_W)
            / Float64(BWConstants.SCALE)
            / 2.0,  # Follow scroll
            cam_y,
            Float64(BWConstants.SCALE),
            Int(BWConstants.VIEWPORT_W),
            Int(BWConstants.VIEWPORT_H),
            flip_y=True,
        )

        # Draw terrain
        self._draw_terrain(renderer, camera)

        # Draw walker
        self._draw_hull(renderer, camera)
        self._draw_legs(renderer, camera)

        # Draw info text
        var hull_x = self.physics.get_body_x(0, Self.BODY_HULL)
        var info_text = String("x: ") + String(Int(hull_x * 10) / 10)
        renderer.draw_text(info_text, 10, 10, white())

        renderer.flip()

    def _draw_terrain(mut self, mut renderer: Renderer2D, camera: Camera):
        """Draw terrain polygons using Camera world coordinates."""
        var terrain_color = grass_green()

        for i in range(len(self.terrain_x) - 1):
            var x1 = self.terrain_x[i]
            var x2 = self.terrain_x[i + 1]

            # Skip if off-screen (using camera visibility check)
            if not camera.is_visible(
                RenderVec2(Float64(x1), Float64(self.terrain_y[i])),
                margin=Float64(BWConstants.TERRAIN_STEP) * 2,
            ):
                if not camera.is_visible(
                    RenderVec2(Float64(x2), Float64(self.terrain_y[i + 1])),
                    margin=Float64(BWConstants.TERRAIN_STEP) * 2,
                ):
                    continue

            # Create polygon vertices in world coordinates
            var vertices = List[RenderVec2]()
            vertices.append(RenderVec2(Float64(x1), Float64(self.terrain_y[i])))
            vertices.append(
                RenderVec2(Float64(x2), Float64(self.terrain_y[i + 1]))
            )
            vertices.append(RenderVec2(Float64(x2), 0.0))  # Bottom
            vertices.append(RenderVec2(Float64(x1), 0.0))

            renderer.draw_polygon_world(
                vertices, camera, terrain_color, filled=True
            )

    def _draw_hull(mut self, mut renderer: Renderer2D, camera: Camera):
        """Draw hull polygon using Transform2D and Camera."""
        var pos_x = self.physics.get_body_x(0, Self.BODY_HULL)
        var pos_y = self.physics.get_body_y(0, Self.BODY_HULL)
        var angle = self.physics.get_body_angle(0, Self.BODY_HULL)

        var hull_color = hull_purple()

        # Hull vertices in local coordinates (already in world units)
        var hull_verts = List[RenderVec2]()
        hull_verts.append(
            RenderVec2(
                Float64(-30.0 / BWConstants.SCALE),
                Float64(9.0 / BWConstants.SCALE),
            )
        )
        hull_verts.append(
            RenderVec2(
                Float64(6.0 / BWConstants.SCALE),
                Float64(9.0 / BWConstants.SCALE),
            )
        )
        hull_verts.append(
            RenderVec2(
                Float64(34.0 / BWConstants.SCALE),
                Float64(1.0 / BWConstants.SCALE),
            )
        )
        hull_verts.append(
            RenderVec2(
                Float64(34.0 / BWConstants.SCALE),
                Float64(-8.0 / BWConstants.SCALE),
            )
        )
        hull_verts.append(
            RenderVec2(
                Float64(-30.0 / BWConstants.SCALE),
                Float64(-8.0 / BWConstants.SCALE),
            )
        )

        # Create transform for hull position and rotation
        var transform = Transform2D(
            Float64(pos_x), Float64(pos_y), Float64(angle)
        )

        renderer.draw_transformed_polygon(
            hull_verts, transform, camera, hull_color, filled=True
        )

    def _draw_legs(mut self, mut renderer: Renderer2D, camera: Camera):
        """Draw leg segments using Transform2D and Camera."""
        for side in range(2):
            var upper_body_idx = (
                Self.BODY_UPPER_LEG_L if side == 0 else Self.BODY_UPPER_LEG_R
            )
            var lower_body_idx = (
                Self.BODY_LOWER_LEG_L if side == 0 else Self.BODY_LOWER_LEG_R
            )
            var has_contact = (
                self.left_leg_contact if side == 0 else self.right_leg_contact
            )

            # Color based on ground contact
            var leg_color = contact_green() if has_contact else inactive_gray()

            # Draw upper leg
            self._draw_leg_segment(
                renderer,
                upper_body_idx,
                Float64(BWConstants.LEG_W / 2.0),
                Float64(BWConstants.LEG_H / 2.0),
                leg_color,
                camera,
            )

            # Draw lower leg (narrower)
            self._draw_leg_segment(
                renderer,
                lower_body_idx,
                0.8 * Float64(BWConstants.LEG_W / 2.0),
                Float64(BWConstants.LEG_H / 2.0),
                leg_color,
                camera,
            )

    def _draw_leg_segment(
        mut self,
        mut renderer: Renderer2D,
        body_idx: Int,
        half_w: Float64,
        half_h: Float64,
        color: SDL_Color,
        camera: Camera,
    ):
        """Draw a single leg segment as a rotated box using Transform2D."""
        var pos_x = self.physics.get_body_x(0, body_idx)
        var pos_y = self.physics.get_body_y(0, body_idx)
        var angle = self.physics.get_body_angle(0, body_idx)

        # Use shape factory for leg box
        var leg_verts = make_rect(half_w * 2.0, half_h * 2.0, centered=True)

        # Create transform for leg position and rotation
        var transform = Transform2D(
            Float64(pos_x), Float64(pos_y), Float64(angle)
        )

        renderer.draw_transformed_polygon(
            leg_verts, transform, camera, color, filled=True
        )

    # =========================================================================
    # GPU Batch Methods (GPUContinuousEnv Trait)
    # =========================================================================

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
        """GPU step kernel for batched continuous actions."""
        # Workspace layout (total = SHAPES_SIZE + BATCH * PER_ENV_SIZE):
        #   [0 .. SHAPES_SIZE): shapes (shared across envs)
        #   [SHAPES_SIZE .. +BATCH*CONTACTS_SIZE): contacts (all envs)
        #   [.. +BATCH): contact_counts
        #   [.. +BATCH): edge_counts
        #   [.. +BATCH): joint_counts
        comptime SHAPES_SIZE = BWConstants.NUM_SHAPES * SHAPE_MAX_SIZE
        comptime CONTACTS_SIZE = BWConstants.MAX_CONTACTS * CONTACT_DATA_SIZE
        comptime PER_ENV_SIZE = CONTACTS_SIZE + 3

        var shapes_buf: DeviceBuffer[dtype]
        var contacts_buf: DeviceBuffer[dtype]
        var contact_counts_buf: DeviceBuffer[dtype]
        var edge_counts_buf: DeviceBuffer[dtype]
        var joint_counts_buf: DeviceBuffer[dtype]

        if Bool(workspace_ptr):
            # Reuse pre-allocated workspace
            shapes_buf = DeviceBuffer[dtype](
                ctx,
                workspace_ptr.value(),
                SHAPES_SIZE,
                owning=False,
            )
            var per_env_ptr = workspace_ptr.value().unsafe_offset(SHAPES_SIZE)
            contacts_buf = DeviceBuffer[dtype](
                ctx,
                per_env_ptr,
                BATCH_SIZE * CONTACTS_SIZE,
                owning=False,
            )
            contact_counts_buf = DeviceBuffer[dtype](
                ctx,
                per_env_ptr.unsafe_offset(BATCH_SIZE * CONTACTS_SIZE),
                BATCH_SIZE,
                owning=False,
            )
            edge_counts_buf = DeviceBuffer[dtype](
                ctx,
                per_env_ptr.unsafe_offset(BATCH_SIZE * CONTACTS_SIZE + BATCH_SIZE),
                BATCH_SIZE,
                owning=False,
            )
            joint_counts_buf = DeviceBuffer[dtype](
                ctx,
                per_env_ptr.unsafe_offset(BATCH_SIZE * CONTACTS_SIZE + 2 * BATCH_SIZE),
                BATCH_SIZE,
                owning=False,
            )
        else:
            # Fallback: allocate per-step (backward compatible)
            contacts_buf = ctx.enqueue_create_buffer[dtype](
                BATCH_SIZE * CONTACTS_SIZE
            )
            contact_counts_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE)
            shapes_buf = ctx.enqueue_create_buffer[dtype](SHAPES_SIZE)
            edge_counts_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE)
            joint_counts_buf = ctx.enqueue_create_buffer[dtype](BATCH_SIZE)
            # Initialize shapes (only needed on fresh allocation)
            BipedalWalker[Self.dtype]._init_shapes_gpu(ctx, shapes_buf)

        # Fused step kernel
        BipedalWalker[Self.dtype]._fused_step_gpu[
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
        """GPU reset kernel."""
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
            var combined_seed = Int(seed) * 2654435761 + (i + 1) * 12345
            BipedalWalker[Self.dtype]._reset_env_gpu[BATCH_SIZE, STATE_SIZE](
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
        """GPU selective reset kernel - resets only done environments."""
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
                var done_val = dones[i]
                if done_val > Scalar[dtype](0.5):
                    var combined_seed = (
                        Int(rebind[Scalar[DType.uint64]](counter[0]))
                        * 2654435761
                        + (i + 1) * 12345
                    )
                    BipedalWalker[Self.dtype]._reset_env_gpu[
                        BATCH_SIZE, STATE_SIZE
                    ](states, i, combined_seed)
                    dones[i] = Scalar[dtype](0.0)

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
                var done_val = dones[i]
                if done_val > Scalar[dtype](0.5):
                    var combined_seed = Int(seed) * 2654435761 + (i + 1) * 12345
                    BipedalWalker[Self.dtype]._reset_env_gpu[
                        BATCH_SIZE, STATE_SIZE
                    ](states, i, combined_seed)
                    dones[i] = Scalar[dtype](0.0)

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
        """Initialize shapes in the shared region of the workspace."""
        comptime SHAPES_SIZE = BWConstants.NUM_SHAPES * SHAPE_MAX_SIZE
        var shapes_buf = DeviceBuffer[dtype](
            ctx,
            workspace_buf.unsafe_ptr(),
            SHAPES_SIZE,
            owning=False,
        )
        BipedalWalker[Self.dtype]._init_shapes_gpu(ctx, shapes_buf)

    @staticmethod
    def update_curriculum_gpu(
        ctx: DeviceContext,
        mut workspace_buf: DeviceBuffer[dtype],
        curriculum_values: List[Scalar[dtype]],
    ) raises:
        """No-op: BipedalWalker doesn't use curriculum."""
        pass

    # =========================================================================
    # GPU Helper Functions
    # =========================================================================

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
            Layout.row_major(BATCH, BWConstants.MAX_CONTACTS, CONTACT_DATA_SIZE),
            MutAnyOrigin,
        ],
        n_contacts: Int,
        n_joints: Int,
        gravity_x: Scalar[dtype],
        gravity_y: Scalar[dtype],
        dt: Scalar[dtype],
    ):
        """Box2D island solve after collision detection — the ONE copy the
        CPU step and the GPU kernel run (see LunarLander's twin): integrate
        velocities, init + warm start contacts and joints, velocity
        iterations (joints, then contacts), integrate positions, position
        iterations (contacts, then joints; stop once both are within slop).
        Gymnasium steps BipedalWalker with 180 / 60 iterations; this env
        keeps its own VELOCITY_ITERATIONS / POSITION_ITERATIONS."""
        comptime NB = BWConstants.NUM_BODIES
        comptime MC = BWConstants.MAX_CONTACTS
        comptime MJ = BWConstants.MAX_JOINTS
        comptime BO = BWConstants.BODIES_OFFSET
        comptime JO = BWConstants.JOINTS_OFFSET
        SemiImplicitEuler.integrate_velocities_single_env[
            BATCH, NB, STATE_SIZE, BO, BWConstants.FORCES_OFFSET
        ](env, states, gravity_x, gravity_y, dt)
        ImpulseSolver.init_velocity_single_env[BATCH, NB, MC, STATE_SIZE, BO](
            env,
            states,
            contacts,
            n_contacts,
            Scalar[dtype](BWConstants.RESTITUTION),
        )
        RevoluteJointSolver.init_velocity_single_env[
            BATCH, NB, MJ, STATE_SIZE, BO, JO
        ](env, states, n_joints)
        for _ in range(BWConstants.VELOCITY_ITERATIONS):
            RevoluteJointSolver.solve_velocity_single_env[
                BATCH, NB, MJ, STATE_SIZE, BO, JO
            ](env, states, n_joints, dt)
            ImpulseSolver.solve_velocity_single_env[
                BATCH, NB, MC, STATE_SIZE, BO
            ](env, states, contacts, n_contacts, Scalar[dtype](BWConstants.FRICTION))
        SemiImplicitEuler.integrate_positions_single_env[
            BATCH, NB, STATE_SIZE, BO
        ](env, states, dt)
        for _ in range(BWConstants.POSITION_ITERATIONS):
            var contacts_ok = ImpulseSolver.solve_position_single_env[
                BATCH, NB, MC, STATE_SIZE, BO
            ](env, states, contacts, n_contacts)
            var joints_ok = RevoluteJointSolver.solve_position_single_env[
                BATCH, NB, MJ, STATE_SIZE, BO, JO
            ](env, states, n_joints)
            if contacts_ok and joints_ok:
                break

    @always_inline
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
        var rng = PhiloxRandom(seed=UInt64(seed), offset=0)
        var terrain_rng = PhiloxRandom(seed=UInt64(seed) + 1000, offset=0)

        # Generate terrain
        var n_edges = min(
            BWConstants.TERRAIN_LENGTH - 1, BWConstants.MAX_TERRAIN_EDGES
        )
        states[env, BWConstants.EDGE_COUNT_OFFSET] = Scalar[dtype](n_edges)

        var y: Scalar[dtype] = Scalar[dtype](BWConstants.TERRAIN_HEIGHT)
        var velocity: Scalar[dtype] = Scalar[dtype](0.0)
        var prev_y = y

        for edge in range(n_edges):
            var x0 = Scalar[dtype](edge) * Scalar[dtype](
                BWConstants.TERRAIN_STEP
            )
            var x1 = Scalar[dtype](edge + 1) * Scalar[dtype](
                BWConstants.TERRAIN_STEP
            )

            prev_y = y
            var terrain_height = Scalar[dtype](BWConstants.TERRAIN_HEIGHT)
            velocity = Scalar[dtype](0.8) * velocity + Scalar[dtype](0.01) * (
                Scalar[dtype](1.0) if terrain_height
                > y else Scalar[dtype](-1.0)
            )
            if edge > BWConstants.TERRAIN_STARTPAD:
                var rand_vals = terrain_rng.step_uniform()
                velocity = velocity + (
                    rand_vals[0] * Scalar[dtype](2.0) - Scalar[dtype](1.0)
                ) / Scalar[dtype](BWConstants.SCALE)
            y = y + velocity

            # Compute normal
            var dx = x1 - x0
            var dy = y - prev_y
            var length = sqrt(dx * dx + dy * dy)
            var nx = -dy / length
            var ny = dx / length
            if ny < Scalar[dtype](0):
                nx = -nx
                ny = -ny

            var edge_off = BWConstants.EDGES_OFFSET + edge * 6
            states[env, edge_off + 0] = x0
            states[env, edge_off + 1] = prev_y
            states[env, edge_off + 2] = x1
            states[env, edge_off + 3] = y
            states[env, edge_off + 4] = nx
            states[env, edge_off + 5] = ny

        # Initialize walker
        var rand_vals = rng.step_uniform()
        var init_x = Scalar[dtype](BWConstants.TERRAIN_STARTPAD) * Scalar[
            dtype
        ](BWConstants.TERRAIN_STEP)
        var init_y = Scalar[dtype](
            BWConstants.TERRAIN_HEIGHT + 2.0 * BWConstants.LEG_H
        )
        var init_vx = (
            rand_vals[0] * Scalar[dtype](2.0) - Scalar[dtype](1.0)
        ) * Scalar[dtype](0.1)
        var init_vy = (
            rand_vals[1] * Scalar[dtype](2.0) - Scalar[dtype](1.0)
        ) * Scalar[dtype](0.1)

        # Hull
        var hull_off = BWConstants.BODIES_OFFSET
        states[env, hull_off + IDX_X] = init_x
        states[env, hull_off + IDX_Y] = init_y
        states[env, hull_off + IDX_ANGLE] = Scalar[dtype](0)
        states[env, hull_off + IDX_VX] = init_vx
        states[env, hull_off + IDX_VY] = init_vy
        states[env, hull_off + IDX_OMEGA] = Scalar[dtype](0)
        states[env, hull_off + IDX_INV_MASS] = Scalar[dtype](
            1.0 / BWConstants.HULL_MASS
        )
        states[env, hull_off + IDX_INV_INERTIA] = Scalar[dtype](
            1.0 / BWConstants.HULL_INERTIA
        )
        states[env, hull_off + IDX_SHAPE] = Scalar[dtype](0)

        # Legs
        for leg in range(2):
            var upper_off = (
                BWConstants.BODIES_OFFSET + (leg * 2 + 1) * BODY_STATE_SIZE
            )
            var upper_y = (
                init_y
                + Scalar[dtype](BWConstants.LEG_DOWN)
                - Scalar[dtype](BWConstants.UPPER_LEG_H / 2)
            )
            states[env, upper_off + IDX_X] = init_x
            states[env, upper_off + IDX_Y] = upper_y
            states[env, upper_off + IDX_ANGLE] = Scalar[dtype](0)
            states[env, upper_off + IDX_VX] = init_vx
            states[env, upper_off + IDX_VY] = init_vy
            states[env, upper_off + IDX_OMEGA] = Scalar[dtype](0)
            states[env, upper_off + IDX_INV_MASS] = Scalar[dtype](
                1.0 / BWConstants.LEG_MASS
            )
            states[env, upper_off + IDX_INV_INERTIA] = Scalar[dtype](
                1.0 / BWConstants.LEG_INERTIA
            )
            states[env, upper_off + IDX_SHAPE] = Scalar[dtype](leg * 2 + 1)

            var lower_off = (
                BWConstants.BODIES_OFFSET + (leg * 2 + 2) * BODY_STATE_SIZE
            )
            var lower_y = (
                upper_y
                - Scalar[dtype](BWConstants.UPPER_LEG_H / 2)
                - Scalar[dtype](BWConstants.LOWER_LEG_H / 2)
            )
            states[env, lower_off + IDX_X] = init_x
            states[env, lower_off + IDX_Y] = lower_y
            states[env, lower_off + IDX_ANGLE] = Scalar[dtype](0)
            states[env, lower_off + IDX_VX] = init_vx
            states[env, lower_off + IDX_VY] = init_vy
            states[env, lower_off + IDX_OMEGA] = Scalar[dtype](0)
            states[env, lower_off + IDX_INV_MASS] = Scalar[dtype](
                1.0 / BWConstants.LOWER_LEG_MASS
            )
            states[env, lower_off + IDX_INV_INERTIA] = Scalar[dtype](
                1.0 / BWConstants.LOWER_LEG_INERTIA
            )
            states[env, lower_off + IDX_SHAPE] = Scalar[dtype](leg * 2 + 2)

        # Joints
        states[env, BWConstants.JOINT_COUNT_OFFSET] = Scalar[dtype](4)
        for leg in range(2):
            var hip_off = (
                BWConstants.JOINTS_OFFSET + (leg * 2) * JOINT_DATA_SIZE
            )
            states[env, hip_off + JOINT_TYPE] = Scalar[dtype](JOINT_REVOLUTE)
            states[env, hip_off + JOINT_BODY_A] = Scalar[dtype](0)
            states[env, hip_off + JOINT_BODY_B] = Scalar[dtype](leg * 2 + 1)
            states[env, hip_off + JOINT_ANCHOR_AX] = Scalar[dtype](0)
            states[env, hip_off + JOINT_ANCHOR_AY] = Scalar[dtype](
                BWConstants.LEG_DOWN
            )
            states[env, hip_off + JOINT_ANCHOR_BX] = Scalar[dtype](0)
            states[env, hip_off + JOINT_ANCHOR_BY] = Scalar[dtype](
                BWConstants.UPPER_LEG_H / 2
            )
            states[env, hip_off + JOINT_REF_ANGLE] = Scalar[dtype](0)
            states[env, hip_off + JOINT_LOWER_LIMIT] = Scalar[dtype](
                BWConstants.HIP_LIMIT_LOW
            )
            states[env, hip_off + JOINT_UPPER_LIMIT] = Scalar[dtype](
                BWConstants.HIP_LIMIT_HIGH
            )
            states[env, hip_off + JOINT_MAX_MOTOR_TORQUE] = Scalar[dtype](
                BWConstants.MOTORS_TORQUE
            )
            states[env, hip_off + JOINT_MOTOR_SPEED] = Scalar[dtype](0)
            states[env, hip_off + JOINT_FLAGS] = Scalar[dtype](
                JOINT_FLAG_LIMIT_ENABLED | JOINT_FLAG_MOTOR_ENABLED
            )
            RevoluteJointSolver.clear_warm_start[BATCH_SIZE, STATE_SIZE](
                states, env, hip_off
            )

            var knee_off = (
                BWConstants.JOINTS_OFFSET + (leg * 2 + 1) * JOINT_DATA_SIZE
            )
            states[env, knee_off + JOINT_TYPE] = Scalar[dtype](JOINT_REVOLUTE)
            states[env, knee_off + JOINT_BODY_A] = Scalar[dtype](leg * 2 + 1)
            states[env, knee_off + JOINT_BODY_B] = Scalar[dtype](leg * 2 + 2)
            states[env, knee_off + JOINT_ANCHOR_AX] = Scalar[dtype](0)
            states[env, knee_off + JOINT_ANCHOR_AY] = Scalar[dtype](
                -BWConstants.UPPER_LEG_H / 2
            )
            states[env, knee_off + JOINT_ANCHOR_BX] = Scalar[dtype](0)
            states[env, knee_off + JOINT_ANCHOR_BY] = Scalar[dtype](
                BWConstants.LOWER_LEG_H / 2
            )
            states[env, knee_off + JOINT_REF_ANGLE] = Scalar[dtype](0)
            states[env, knee_off + JOINT_LOWER_LIMIT] = Scalar[dtype](
                BWConstants.KNEE_LIMIT_LOW
            )
            states[env, knee_off + JOINT_UPPER_LIMIT] = Scalar[dtype](
                BWConstants.KNEE_LIMIT_HIGH
            )
            states[env, knee_off + JOINT_MAX_MOTOR_TORQUE] = Scalar[dtype](
                BWConstants.MOTORS_TORQUE
            )
            states[env, knee_off + JOINT_MOTOR_SPEED] = Scalar[dtype](0)
            states[env, knee_off + JOINT_FLAGS] = Scalar[dtype](
                JOINT_FLAG_LIMIT_ENABLED | JOINT_FLAG_MOTOR_ENABLED
            )
            RevoluteJointSolver.clear_warm_start[BATCH_SIZE, STATE_SIZE](
                states, env, knee_off
            )

        # Clear forces
        for body in range(BWConstants.NUM_BODIES):
            var force_off = BWConstants.FORCES_OFFSET + body * 3
            states[env, force_off + 0] = Scalar[dtype](0)
            states[env, force_off + 1] = Scalar[dtype](0)
            states[env, force_off + 2] = Scalar[dtype](0)

        # Compute initial observation (matches CPU _update_cached_state)
        BipedalWalker[Self.dtype]._compute_initial_obs_reset_gpu[
            BATCH_SIZE, STATE_SIZE
        ](states, env)

        # Initialize metadata
        states[
            env, BWConstants.METADATA_OFFSET + BWConstants.META_STEP_COUNT
        ] = Scalar[dtype](0)
        # META_TOTAL_REWARD (index 1) is used to store prev_x for GPU reward computation
        states[
            env, BWConstants.METADATA_OFFSET + BWConstants.META_TOTAL_REWARD
        ] = init_x  # prev_x for shaping reward
        states[
            env, BWConstants.METADATA_OFFSET + BWConstants.META_PREV_SHAPING
        ] = Scalar[dtype](0)
        states[
            env, BWConstants.METADATA_OFFSET + BWConstants.META_DONE
        ] = Scalar[dtype](0)
        states[
            env, BWConstants.METADATA_OFFSET + BWConstants.META_SCROLL
        ] = Scalar[dtype](0)
        states[
            env, BWConstants.METADATA_OFFSET + BWConstants.META_LEFT_CONTACT
        ] = Scalar[dtype](0)
        states[
            env, BWConstants.METADATA_OFFSET + BWConstants.META_RIGHT_CONTACT
        ] = Scalar[dtype](0)
        states[
            env, BWConstants.METADATA_OFFSET + BWConstants.META_GAME_OVER
        ] = Scalar[dtype](0)

    @always_inline
    @staticmethod
    def _compute_initial_obs_reset_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
    ](
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
    ):
        """Compute initial observation after reset (GPU version).

        Matches CPU _update_cached_state() + lidar computation.
        Writes to states[env, OBS_OFFSET:OBS_OFFSET+24].
        """
        var obs_off = BWConstants.OBS_OFFSET

        # Hull state
        var hull_off = BWConstants.BODIES_OFFSET
        var hull_angle = rebind[Scalar[dtype]](
            states[env, hull_off + IDX_ANGLE]
        )
        var hull_omega = rebind[Scalar[dtype]](
            states[env, hull_off + IDX_OMEGA]
        )
        var hull_vx = rebind[Scalar[dtype]](states[env, hull_off + IDX_VX])
        var hull_vy = rebind[Scalar[dtype]](states[env, hull_off + IDX_VY])
        var hull_x = rebind[Scalar[dtype]](states[env, hull_off + IDX_X])
        var hull_y = rebind[Scalar[dtype]](states[env, hull_off + IDX_Y])

        # [0]: hull_angle
        states[env, obs_off + 0] = hull_angle

        # [1]: hull_angular_velocity (normalized by /5.0)
        states[env, obs_off + 1] = hull_omega / Scalar[dtype](5.0)

        # [2]: vel_x = hull_vx * (VIEWPORT_W/SCALE) / FPS
        var vel_scale_x = Scalar[dtype](
            BWConstants.VIEWPORT_W / BWConstants.SCALE / BWConstants.FPS
        )
        states[env, obs_off + 2] = hull_vx * vel_scale_x

        # [3]: vel_y = hull_vy * (VIEWPORT_H/SCALE) / FPS
        var vel_scale_y = Scalar[dtype](
            BWConstants.VIEWPORT_H / BWConstants.SCALE / BWConstants.FPS
        )
        states[env, obs_off + 3] = hull_vy * vel_scale_y

        # Leg 1 (left) - bodies 1 (upper) and 2 (lower)
        var upper1_off = BWConstants.BODIES_OFFSET + 1 * BODY_STATE_SIZE
        var lower1_off = BWConstants.BODIES_OFFSET + 2 * BODY_STATE_SIZE
        var upper1_angle = rebind[Scalar[dtype]](
            states[env, upper1_off + IDX_ANGLE]
        )
        var lower1_angle = rebind[Scalar[dtype]](
            states[env, lower1_off + IDX_ANGLE]
        )
        var upper1_omega = rebind[Scalar[dtype]](
            states[env, upper1_off + IDX_OMEGA]
        )
        var lower1_omega = rebind[Scalar[dtype]](
            states[env, lower1_off + IDX_OMEGA]
        )

        # [4]: hip1_angle = (upper1_angle - hull_angle)
        states[env, obs_off + 4] = upper1_angle - hull_angle

        # [5]: hip1_speed = (upper1_omega - hull_omega) / 10.0
        states[env, obs_off + 5] = (upper1_omega - hull_omega) / Scalar[dtype](
            10.0
        )

        # [6]: knee1_angle = (lower1_angle - upper1_angle)
        states[env, obs_off + 6] = lower1_angle - upper1_angle

        # [7]: knee1_speed = (lower1_omega - upper1_omega) / 10.0
        states[env, obs_off + 7] = (lower1_omega - upper1_omega) / Scalar[
            dtype
        ](10.0)

        # [8]: leg1_contact = 0.0 (no contact at reset)
        states[env, obs_off + 8] = Scalar[dtype](0.0)

        # Leg 2 (right) - bodies 3 (upper) and 4 (lower)
        var upper2_off = BWConstants.BODIES_OFFSET + 3 * BODY_STATE_SIZE
        var lower2_off = BWConstants.BODIES_OFFSET + 4 * BODY_STATE_SIZE
        var upper2_angle = rebind[Scalar[dtype]](
            states[env, upper2_off + IDX_ANGLE]
        )
        var lower2_angle = rebind[Scalar[dtype]](
            states[env, lower2_off + IDX_ANGLE]
        )
        var upper2_omega = rebind[Scalar[dtype]](
            states[env, upper2_off + IDX_OMEGA]
        )
        var lower2_omega = rebind[Scalar[dtype]](
            states[env, lower2_off + IDX_OMEGA]
        )

        # [9]: hip2_angle = (upper2_angle - hull_angle)
        states[env, obs_off + 9] = upper2_angle - hull_angle

        # [10]: hip2_speed = (upper2_omega - hull_omega) / 10.0
        states[env, obs_off + 10] = (upper2_omega - hull_omega) / Scalar[dtype](
            10.0
        )

        # [11]: knee2_angle = (lower2_angle - upper2_angle)
        states[env, obs_off + 11] = lower2_angle - upper2_angle

        # [12]: knee2_speed = (lower2_omega - upper2_omega) / 10.0
        states[env, obs_off + 12] = (lower2_omega - upper2_omega) / Scalar[
            dtype
        ](10.0)

        # [13]: leg2_contact = 0.0 (no contact at reset)
        states[env, obs_off + 13] = Scalar[dtype](0.0)

        # [14-23]: Lidar (10 rays)
        var n_edges = Int(states[env, BWConstants.EDGE_COUNT_OFFSET])
        var lidar_range = Scalar[dtype](BWConstants.LIDAR_RANGE)

        for i in range(BWConstants.NUM_LIDAR):
            # Angle relative to hull: 0 to 1.5 radians (matches CPU Lidar)
            var local_angle = (
                Scalar[dtype](i)
                * Scalar[dtype](1.5)
                / Scalar[dtype](BWConstants.NUM_LIDAR - 1)
            )
            var world_angle = (
                hull_angle - local_angle - Scalar[dtype](1.5707963267948966)
            )  # pi/2

            # Ray direction scaled by range
            var ray_dx = cos(world_angle) * lidar_range
            var ray_dy = sin(world_angle) * lidar_range

            var min_t = Scalar[dtype](1.0)  # Default to max range

            # Check against each terrain edge
            for edge_idx in range(BWConstants.MAX_TERRAIN_EDGES):
                if edge_idx >= n_edges:
                    break

                var edge_off = BWConstants.EDGES_OFFSET + edge_idx * 6
                var edge_x0 = rebind[Scalar[dtype]](states[env, edge_off + 0])
                var edge_y0 = rebind[Scalar[dtype]](states[env, edge_off + 1])
                var edge_x1 = rebind[Scalar[dtype]](states[env, edge_off + 2])
                var edge_y1 = rebind[Scalar[dtype]](states[env, edge_off + 3])

                # Ray-edge intersection
                var ex = edge_x1 - edge_x0
                var ey = edge_y1 - edge_y0
                var denom = ray_dx * ey - ray_dy * ex

                if denom > Scalar[dtype](-1e-10) and denom < Scalar[dtype](
                    1e-10
                ):
                    continue  # Parallel

                var dx = hull_x - edge_x0
                var dy = hull_y - edge_y0
                var t = (ex * dy - ey * dx) / denom
                var u = (ray_dx * dy - ray_dy * dx) / denom

                if (
                    t >= Scalar[dtype](0.0)
                    and t <= Scalar[dtype](1.0)
                    and u >= Scalar[dtype](0.0)
                    and u <= Scalar[dtype](1.0)
                ):
                    if t < min_t:
                        min_t = t

            states[env, obs_off + BWConstants.LIDAR_START_IDX + i] = min_t

    @staticmethod
    def _init_shapes_gpu(
        ctx: DeviceContext,
        mut shapes_buf: DeviceBuffer[dtype],
    ) raises:
        """Initialize shape definitions for GPU."""
        var shapes = LayoutTensor[
            dtype,
            Layout.row_major(BWConstants.NUM_SHAPES * SHAPE_MAX_SIZE),
        ](shapes_buf)

        @always_inline
        def init_shapes_wrapper(
            shapes: LayoutTensor[
                dtype,
                Layout.row_major(BWConstants.NUM_SHAPES * SHAPE_MAX_SIZE),
                MutAnyOrigin,
            ],
        ):
            var tid = Int(block_dim.x * block_idx.x + thread_idx.x)
            if tid > 0:
                return

            # Hull (5-vertex polygon)
            shapes[0] = Scalar[dtype](SHAPE_POLYGON)
            shapes[1] = Scalar[dtype](5)
            shapes[2] = Scalar[dtype](-30.0 / BWConstants.SCALE)
            shapes[3] = Scalar[dtype](0.0)
            shapes[4] = Scalar[dtype](-6.0 / BWConstants.SCALE)
            shapes[5] = Scalar[dtype](30.0 / BWConstants.SCALE)
            shapes[6] = Scalar[dtype](6.0 / BWConstants.SCALE)
            shapes[7] = Scalar[dtype](30.0 / BWConstants.SCALE)
            shapes[8] = Scalar[dtype](30.0 / BWConstants.SCALE)
            shapes[9] = Scalar[dtype](0.0)
            shapes[10] = Scalar[dtype](0.0)
            shapes[11] = Scalar[dtype](-12.0 / BWConstants.SCALE)

            # Upper legs (4-vertex boxes) - shapes 1 and 3
            for s in range(2):
                var base = (s * 2 + 1) * SHAPE_MAX_SIZE
                shapes[base + 0] = Scalar[dtype](SHAPE_POLYGON)
                shapes[base + 1] = Scalar[dtype](4)
                shapes[base + 2] = Scalar[dtype](-BWConstants.UPPER_LEG_W / 2)
                shapes[base + 3] = Scalar[dtype](BWConstants.UPPER_LEG_H / 2)
                shapes[base + 4] = Scalar[dtype](-BWConstants.UPPER_LEG_W / 2)
                shapes[base + 5] = Scalar[dtype](-BWConstants.UPPER_LEG_H / 2)
                shapes[base + 6] = Scalar[dtype](BWConstants.UPPER_LEG_W / 2)
                shapes[base + 7] = Scalar[dtype](-BWConstants.UPPER_LEG_H / 2)
                shapes[base + 8] = Scalar[dtype](BWConstants.UPPER_LEG_W / 2)
                shapes[base + 9] = Scalar[dtype](BWConstants.UPPER_LEG_H / 2)

            # Lower legs (4-vertex boxes) - shapes 2 and 4
            for s in range(2):
                var base = (s * 2 + 2) * SHAPE_MAX_SIZE
                shapes[base + 0] = Scalar[dtype](SHAPE_POLYGON)
                shapes[base + 1] = Scalar[dtype](4)
                shapes[base + 2] = Scalar[dtype](-BWConstants.LOWER_LEG_W / 2)
                shapes[base + 3] = Scalar[dtype](BWConstants.LOWER_LEG_H / 2)
                shapes[base + 4] = Scalar[dtype](-BWConstants.LOWER_LEG_W / 2)
                shapes[base + 5] = Scalar[dtype](-BWConstants.LOWER_LEG_H / 2)
                shapes[base + 6] = Scalar[dtype](BWConstants.LOWER_LEG_W / 2)
                shapes[base + 7] = Scalar[dtype](-BWConstants.LOWER_LEG_H / 2)
                shapes[base + 8] = Scalar[dtype](BWConstants.LOWER_LEG_W / 2)
                shapes[base + 9] = Scalar[dtype](BWConstants.LOWER_LEG_H / 2)

        ctx.enqueue_function[init_shapes_wrapper](
            shapes,
            grid_dim=(1,),
            block_dim=(1,),
        )

    @staticmethod
    def _fused_step_gpu[
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
    ) raises:
        """Fused GPU step kernel - uses proper physics matching CPU."""
        comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB
        comptime STATE_SIZE = BWConstants.STATE_SIZE_VAL

        var states = LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE)
        ](states_buf)
        var shapes = LayoutTensor[
            dtype,
            Layout.row_major(BWConstants.NUM_SHAPES, SHAPE_MAX_SIZE),
        ](shapes_buf)
        var contacts = LayoutTensor[
            dtype,
            Layout.row_major(
                BATCH_SIZE, BWConstants.MAX_CONTACTS, CONTACT_DATA_SIZE
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

        @always_inline
        def step_wrapper(
            states: LayoutTensor[
                dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
            ],
            shapes: LayoutTensor[
                dtype,
                Layout.row_major(BWConstants.NUM_SHAPES, SHAPE_MAX_SIZE),
                ImmutAnyOrigin,
            ],
            contacts: LayoutTensor[
                dtype,
                Layout.row_major(
                    BATCH_SIZE, BWConstants.MAX_CONTACTS, CONTACT_DATA_SIZE
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
        ):
            var env = Int(block_dim.x * block_idx.x + thread_idx.x)
            if env >= BATCH_SIZE:
                return

            # Apply motor actions to joints
            BipedalWalker[Self.dtype]._apply_motor_actions_gpu[
                BATCH_SIZE, STATE_SIZE, ACTION_DIM
            ](env, states, actions)

            # Physics constants
            var dt = Scalar[dtype](BWConstants.DT)
            var gravity_x = Scalar[dtype](BWConstants.GRAVITY_X)
            var gravity_y = Scalar[dtype](BWConstants.GRAVITY_Y)

            # Get joint count
            var joint_count = Int(states[env, BWConstants.JOINT_COUNT_OFFSET])
            var n_edges = Int(states[env, BWConstants.EDGE_COUNT_OFFSET])

            # Step 2: Collision detection against terrain edges
            EdgeTerrainCollision.detect_single_env[
                BATCH_SIZE,
                BWConstants.NUM_BODIES,
                BWConstants.NUM_SHAPES,
                BWConstants.MAX_CONTACTS,
                BWConstants.MAX_TERRAIN_EDGES,
                STATE_SIZE,
                BWConstants.BODIES_OFFSET,
                BWConstants.EDGES_OFFSET,
            ](env, states, shapes, n_edges, contacts, contact_counts)

            var contact_count = Int(contact_counts[env])

            # Steps 3-5: the shared Box2D island solve.
            BipedalWalker[Self.dtype]._solve_step_single_env[
                BATCH_SIZE, STATE_SIZE
            ](
                env,
                states,
                contacts,
                contact_count,
                joint_count,
                gravity_x,
                gravity_y,
                dt,
            )

            # Step 6: Clear forces
            for body in range(BWConstants.NUM_BODIES):
                var force_off = BWConstants.FORCES_OFFSET + body * 3
                states[env, force_off + 0] = Scalar[dtype](0)
                states[env, force_off + 1] = Scalar[dtype](0)
                states[env, force_off + 2] = Scalar[dtype](0)

            # Step 7: Update contact flags and check hull contact
            var hull_contact = Scalar[dtype](0.0)
            var left_leg_contact = Scalar[dtype](0.0)
            var right_leg_contact = Scalar[dtype](0.0)

            for c in range(contact_count):
                var body_a = Int(contacts[env, c, CONTACT_BODY_A])
                if body_a == 0:  # Hull
                    hull_contact = Scalar[dtype](1.0)
                elif body_a == 2:  # Lower leg left
                    left_leg_contact = Scalar[dtype](1.0)
                elif body_a == 4:  # Lower leg right
                    right_leg_contact = Scalar[dtype](1.0)

            # Store contact information in metadata
            states[
                env, BWConstants.METADATA_OFFSET + BWConstants.META_LEFT_CONTACT
            ] = left_leg_contact
            states[
                env,
                BWConstants.METADATA_OFFSET + BWConstants.META_RIGHT_CONTACT,
            ] = right_leg_contact
            states[
                env, BWConstants.METADATA_OFFSET + BWConstants.META_GAME_OVER
            ] = hull_contact

            # Step 8: Compute lidar (10 rays from hull)
            BipedalWalker[Self.dtype]._compute_lidar_gpu[
                BATCH_SIZE, STATE_SIZE, OBS_DIM
            ](env, states, obs)

            # Extract observation (non-lidar parts)
            BipedalWalker[Self.dtype]._extract_obs_gpu[
                BATCH_SIZE, STATE_SIZE, OBS_DIM
            ](env, states, obs)

            # Step 9: Compute reward and termination
            var hull_off = BWConstants.BODIES_OFFSET
            var hull_x = states[env, hull_off + IDX_X]
            var hull_y = states[env, hull_off + IDX_Y]
            var hull_vx = states[env, hull_off + IDX_VX]
            var hull_angle = states[env, hull_off + IDX_ANGLE]
            var step_count = states[
                env, BWConstants.METADATA_OFFSET + BWConstants.META_STEP_COUNT
            ]

            # Increment step count
            states[
                env, BWConstants.METADATA_OFFSET + BWConstants.META_STEP_COUNT
            ] = step_count + Scalar[dtype](1)

            # Shaping reward: forward progress
            var prev_x = states[env, BWConstants.METADATA_OFFSET + 1]
            var forward_progress = hull_x - prev_x
            states[env, BWConstants.METADATA_OFFSET + 1] = hull_x

            # Reward: forward progress minus angle penalty
            var angle_abs = (
                hull_angle if hull_angle >= Scalar[dtype](0) else -hull_angle
            )
            var angle_penalty = Scalar[dtype](5.0) * angle_abs
            var reward = (
                Scalar[dtype](130.0 / BWConstants.SCALE) * forward_progress
                - angle_penalty
            )

            # Energy penalty
            var hip1_action = actions[env, 0]
            var knee1_action = actions[env, 1]
            var hip2_action = actions[env, 2]
            var knee2_action = actions[env, 3]
            var a0 = (
                hip1_action if hip1_action >= Scalar[dtype](0) else -hip1_action
            )
            var a1 = (
                knee1_action if knee1_action
                >= Scalar[dtype](0) else -knee1_action
            )
            var a2 = (
                hip2_action if hip2_action >= Scalar[dtype](0) else -hip2_action
            )
            var a3 = (
                knee2_action if knee2_action
                >= Scalar[dtype](0) else -knee2_action
            )
            var energy = a0 + a1 + a2 + a3
            var energy_penalty = (
                Scalar[dtype](0.00035 * BWConstants.MOTORS_TORQUE) * energy
            )
            reward = reward - energy_penalty

            # Height penalty: strongly penalize crawling
            var min_height = Scalar[dtype](
                BWConstants.TERRAIN_HEIGHT + BWConstants.LEG_H * 1.5
            )  # ~5.0, penalty starts
            var critical_height = Scalar[dtype](
                BWConstants.TERRAIN_HEIGHT + BWConstants.LEG_H * 1.2
            )  # ~4.7, terminate below
            if hull_y < min_height:
                # Strong penalty for being too low (crawling)
                reward = reward - Scalar[dtype](10.0) * (min_height - hull_y)

            # Velocity shaping: reward forward movement, penalize backward
            # Scale hull_vx same as observation vel_x
            var vel_scale_x = Scalar[dtype](
                BWConstants.VIEWPORT_W / BWConstants.SCALE / BWConstants.FPS
            )
            var vel_x = hull_vx * vel_scale_x
            if vel_x > Scalar[dtype](0.0):
                # Bonus for moving forward - encourages actual walking
                reward = reward + Scalar[dtype](3.0) * vel_x
            else:
                var vel_x_abs = -vel_x
                reward = reward - Scalar[dtype](5.0) * vel_x_abs

            # Check termination conditions
            var done = Scalar[dtype](0.0)
            var is_terminated = Scalar[dtype](0.0)
            var max_steps = Scalar[dtype](2000)  # Match CPU max_steps
            var new_step_count = step_count + Scalar[dtype](1)

            # Time limit (truncation only, not termination)
            if new_step_count >= max_steps:
                done = Scalar[dtype](1.0)

            # Hull touched ground (game over)
            if hull_contact > Scalar[dtype](0.5):
                done = Scalar[dtype](1.0)
                is_terminated = Scalar[dtype](1.0)
                reward = Scalar[dtype](BWConstants.CRASH_PENALTY)

            # Crawling too low (hull almost touching ground)
            if hull_y < critical_height:
                done = Scalar[dtype](1.0)
                is_terminated = Scalar[dtype](1.0)
                reward = Scalar[dtype](-50.0)  # Moderate penalty for crawling

            # Out of bounds (fell off left edge)
            if hull_x < Scalar[dtype](0):
                done = Scalar[dtype](1.0)
                is_terminated = Scalar[dtype](1.0)
                reward = Scalar[dtype](BWConstants.CRASH_PENALTY)

            # Success: reached end of terrain
            var terrain_end = Scalar[dtype](
                BWConstants.TERRAIN_LENGTH - BWConstants.TERRAIN_GRASS
            ) * Scalar[dtype](BWConstants.TERRAIN_STEP)
            if hull_x > terrain_end:
                done = Scalar[dtype](1.0)
                is_terminated = Scalar[dtype](1.0)

            rewards[env] = reward
            dones[env] = done
            terminated_out[env] = is_terminated

        ctx.enqueue_function[step_wrapper](
            states,
            shapes,
            contacts,
            contact_counts,
            actions,
            rewards,
            dones,
            terminated_out,
            obs,
            grid_dim=(BLOCKS,),
            block_dim=(TPB,),
        )

    @always_inline
    @staticmethod
    def _compute_lidar_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        OBS_DIM: Int,
    ](
        env: Int,
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ],
        obs: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, OBS_DIM), MutAnyOrigin
        ],
    ):
        """Compute 10 lidar readings from hull position.

        Uses same algorithm as CPU Lidar.raycast_env_gpu:
        - Rays spread from 0 to 1.5 radians relative to hull
        - world_angle = hull_angle - local_angle - pi/2
        - Ray direction scaled by lidar_range
        """
        var hull_off = BWConstants.BODIES_OFFSET
        var hull_x = rebind[Scalar[dtype]](states[env, hull_off + IDX_X])
        var hull_y = rebind[Scalar[dtype]](states[env, hull_off + IDX_Y])
        var hull_angle = rebind[Scalar[dtype]](
            states[env, hull_off + IDX_ANGLE]
        )

        var n_edges = Int(states[env, BWConstants.EDGE_COUNT_OFFSET])
        var lidar_range = Scalar[dtype](BWConstants.LIDAR_RANGE)

        # 10 lidar rays spread from 0 to 1.5 radians relative to hull
        for i in range(BWConstants.NUM_LIDAR):
            # Angle relative to hull: 0 to 1.5 radians (matches CPU Lidar)
            var local_angle = (
                Scalar[dtype](i)
                * Scalar[dtype](1.5)
                / Scalar[dtype](BWConstants.NUM_LIDAR - 1)
            )
            var world_angle = (
                hull_angle - local_angle - Scalar[dtype](1.5707963267948966)
            )  # pi/2

            # Ray direction scaled by range (matches CPU)
            var ray_dx = cos(world_angle) * lidar_range
            var ray_dy = sin(world_angle) * lidar_range

            var min_t = Scalar[dtype](1.0)  # Normalized distance (1.0 = no hit)

            # Check against each terrain edge
            for edge_idx in range(BWConstants.MAX_TERRAIN_EDGES):
                if edge_idx >= n_edges:
                    break

                var edge_off = BWConstants.EDGES_OFFSET + edge_idx * 6
                var edge_x0 = rebind[Scalar[dtype]](states[env, edge_off + 0])
                var edge_y0 = rebind[Scalar[dtype]](states[env, edge_off + 1])
                var edge_x1 = rebind[Scalar[dtype]](states[env, edge_off + 2])
                var edge_y1 = rebind[Scalar[dtype]](states[env, edge_off + 3])

                # Ray-edge intersection (matches Lidar._ray_edge_intersection)
                var ex = edge_x1 - edge_x0
                var ey = edge_y1 - edge_y0

                # Denominator: cross product of directions
                var denom = ray_dx * ey - ray_dy * ex

                # Check if parallel (denom ~= 0)
                if denom > Scalar[dtype](-1e-10) and denom < Scalar[dtype](
                    1e-10
                ):
                    continue  # Skip parallel edges

                # Vector from edge start to ray origin
                var dx = hull_x - edge_x0
                var dy = hull_y - edge_y0

                # Compute intersection parameters
                var t = (ex * dy - ey * dx) / denom  # Parameter along ray [0,1]
                var u = (
                    ray_dx * dy - ray_dy * dx
                ) / denom  # Parameter along edge [0,1]

                # Valid intersection: t in [0, 1] and u in [0, 1]
                if (
                    t >= Scalar[dtype](0.0)
                    and t <= Scalar[dtype](1.0)
                    and u >= Scalar[dtype](0.0)
                    and u <= Scalar[dtype](1.0)
                ):
                    if t < min_t:
                        min_t = t

            obs[env, BWConstants.LIDAR_START_IDX + i] = min_t
            states[
                env, BWConstants.OBS_OFFSET + BWConstants.LIDAR_START_IDX + i
            ] = min_t

    @always_inline
    @staticmethod
    def _apply_motor_actions_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        ACTION_DIM: Int,
    ](
        env: Int,
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ],
        actions: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, ACTION_DIM), ImmutAnyOrigin
        ],
    ):
        """Apply motor actions to joints."""
        # Hip 1
        var hip1_off = BWConstants.JOINTS_OFFSET + 0 * JOINT_DATA_SIZE
        var hip1_action = actions[env, 0]
        if hip1_action > Scalar[dtype](1.0):
            hip1_action = Scalar[dtype](1.0)
        if hip1_action < Scalar[dtype](-1.0):
            hip1_action = Scalar[dtype](-1.0)
        var hip1_sign = Scalar[dtype](1.0) if hip1_action >= Scalar[dtype](
            0
        ) else Scalar[dtype](-1.0)
        states[env, hip1_off + JOINT_MOTOR_SPEED] = hip1_sign * Scalar[dtype](
            BWConstants.SPEED_HIP
        )
        var hip1_abs = (
            hip1_action if hip1_action >= Scalar[dtype](0) else -hip1_action
        )
        states[env, hip1_off + JOINT_MAX_MOTOR_TORQUE] = (
            Scalar[dtype](BWConstants.MOTORS_TORQUE) * hip1_abs
        )

        # Knee 1
        var knee1_off = BWConstants.JOINTS_OFFSET + 1 * JOINT_DATA_SIZE
        var knee1_action = actions[env, 1]
        if knee1_action > Scalar[dtype](1.0):
            knee1_action = Scalar[dtype](1.0)
        if knee1_action < Scalar[dtype](-1.0):
            knee1_action = Scalar[dtype](-1.0)
        var knee1_sign = Scalar[dtype](1.0) if knee1_action >= Scalar[dtype](
            0
        ) else Scalar[dtype](-1.0)
        states[env, knee1_off + JOINT_MOTOR_SPEED] = knee1_sign * Scalar[dtype](
            BWConstants.SPEED_KNEE
        )
        var knee1_abs = (
            knee1_action if knee1_action >= Scalar[dtype](0) else -knee1_action
        )
        states[env, knee1_off + JOINT_MAX_MOTOR_TORQUE] = (
            Scalar[dtype](BWConstants.MOTORS_TORQUE) * knee1_abs
        )

        # Hip 2
        var hip2_off = BWConstants.JOINTS_OFFSET + 2 * JOINT_DATA_SIZE
        var hip2_action = actions[env, 2]
        if hip2_action > Scalar[dtype](1.0):
            hip2_action = Scalar[dtype](1.0)
        if hip2_action < Scalar[dtype](-1.0):
            hip2_action = Scalar[dtype](-1.0)
        var hip2_sign = Scalar[dtype](1.0) if hip2_action >= Scalar[dtype](
            0
        ) else Scalar[dtype](-1.0)
        states[env, hip2_off + JOINT_MOTOR_SPEED] = hip2_sign * Scalar[dtype](
            BWConstants.SPEED_HIP
        )
        var hip2_abs = (
            hip2_action if hip2_action >= Scalar[dtype](0) else -hip2_action
        )
        states[env, hip2_off + JOINT_MAX_MOTOR_TORQUE] = (
            Scalar[dtype](BWConstants.MOTORS_TORQUE) * hip2_abs
        )

        # Knee 2
        var knee2_off = BWConstants.JOINTS_OFFSET + 3 * JOINT_DATA_SIZE
        var knee2_action = actions[env, 3]
        if knee2_action > Scalar[dtype](1.0):
            knee2_action = Scalar[dtype](1.0)
        if knee2_action < Scalar[dtype](-1.0):
            knee2_action = Scalar[dtype](-1.0)
        var knee2_sign = Scalar[dtype](1.0) if knee2_action >= Scalar[dtype](
            0
        ) else Scalar[dtype](-1.0)
        states[env, knee2_off + JOINT_MOTOR_SPEED] = knee2_sign * Scalar[dtype](
            BWConstants.SPEED_KNEE
        )
        var knee2_abs = (
            knee2_action if knee2_action >= Scalar[dtype](0) else -knee2_action
        )
        states[env, knee2_off + JOINT_MAX_MOTOR_TORQUE] = (
            Scalar[dtype](BWConstants.MOTORS_TORQUE) * knee2_abs
        )

    @always_inline
    @staticmethod
    def _extract_obs_gpu[
        BATCH_SIZE: Int,
        STATE_SIZE: Int,
        OBS_DIM: Int,
    ](
        env: Int,
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), MutAnyOrigin
        ],
        obs: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, OBS_DIM), MutAnyOrigin
        ],
    ):
        """Extract 24D observation from state.

        Writes to both obs tensor AND state buffer at OBS_OFFSET for consistency.
        """
        # Hull state
        var hull_off = BWConstants.BODIES_OFFSET
        var hull_angle = states[env, hull_off + IDX_ANGLE]
        var hull_omega = states[env, hull_off + IDX_OMEGA]
        var hull_vx = states[env, hull_off + IDX_VX]
        var hull_vy = states[env, hull_off + IDX_VY]

        var obs0 = hull_angle
        var obs1 = hull_omega / Scalar[dtype](5.0)
        var obs2 = hull_vx * Scalar[dtype](
            BWConstants.VIEWPORT_W / BWConstants.SCALE / BWConstants.FPS
        )
        var obs3 = hull_vy * Scalar[dtype](
            BWConstants.VIEWPORT_H / BWConstants.SCALE / BWConstants.FPS
        )

        obs[env, 0] = obs0
        obs[env, 1] = obs1
        obs[env, 2] = obs2
        obs[env, 3] = obs3
        states[env, BWConstants.OBS_OFFSET + 0] = obs0
        states[env, BWConstants.OBS_OFFSET + 1] = obs1
        states[env, BWConstants.OBS_OFFSET + 2] = obs2
        states[env, BWConstants.OBS_OFFSET + 3] = obs3

        # Leg 1 (left)
        var upper1_off = BWConstants.BODIES_OFFSET + 1 * BODY_STATE_SIZE
        var lower1_off = BWConstants.BODIES_OFFSET + 2 * BODY_STATE_SIZE
        var upper1_angle = states[env, upper1_off + IDX_ANGLE]
        var lower1_angle = states[env, lower1_off + IDX_ANGLE]
        var upper1_omega = states[env, upper1_off + IDX_OMEGA]
        var lower1_omega = states[env, lower1_off + IDX_OMEGA]

        var obs4 = upper1_angle - hull_angle
        var obs5 = (upper1_omega - hull_omega) / Scalar[dtype](10.0)
        var obs6 = lower1_angle - upper1_angle
        var obs7 = (lower1_omega - upper1_omega) / Scalar[dtype](10.0)
        var obs8 = states[
            env, BWConstants.METADATA_OFFSET + BWConstants.META_LEFT_CONTACT
        ]

        obs[env, 4] = obs4
        obs[env, 5] = obs5
        obs[env, 6] = obs6
        obs[env, 7] = obs7
        obs[env, 8] = obs8
        states[env, BWConstants.OBS_OFFSET + 4] = obs4
        states[env, BWConstants.OBS_OFFSET + 5] = obs5
        states[env, BWConstants.OBS_OFFSET + 6] = obs6
        states[env, BWConstants.OBS_OFFSET + 7] = obs7
        states[env, BWConstants.OBS_OFFSET + 8] = obs8

        # Leg 2 (right)
        var upper2_off = BWConstants.BODIES_OFFSET + 3 * BODY_STATE_SIZE
        var lower2_off = BWConstants.BODIES_OFFSET + 4 * BODY_STATE_SIZE
        var upper2_angle = states[env, upper2_off + IDX_ANGLE]
        var lower2_angle = states[env, lower2_off + IDX_ANGLE]
        var upper2_omega = states[env, upper2_off + IDX_OMEGA]
        var lower2_omega = states[env, lower2_off + IDX_OMEGA]

        var obs9 = upper2_angle - hull_angle
        var obs10 = (upper2_omega - hull_omega) / Scalar[dtype](10.0)
        var obs11 = lower2_angle - upper2_angle
        var obs12 = (lower2_omega - upper2_omega) / Scalar[dtype](10.0)
        var obs13 = states[
            env, BWConstants.METADATA_OFFSET + BWConstants.META_RIGHT_CONTACT
        ]

        obs[env, 9] = obs9
        obs[env, 10] = obs10
        obs[env, 11] = obs11
        obs[env, 12] = obs12
        obs[env, 13] = obs13
        states[env, BWConstants.OBS_OFFSET + 9] = obs9
        states[env, BWConstants.OBS_OFFSET + 10] = obs10
        states[env, BWConstants.OBS_OFFSET + 11] = obs11
        states[env, BWConstants.OBS_OFFSET + 12] = obs12
        states[env, BWConstants.OBS_OFFSET + 13] = obs13

        # Lidar values are computed by _compute_lidar_gpu
