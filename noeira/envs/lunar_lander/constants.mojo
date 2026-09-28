# =============================================================================
# Physics Constants - Matched to Gymnasium LunarLander-v3
# =============================================================================


from noeira.physics2d import LunarLanderLayout, PhysicsState


struct LLConstants:
    comptime GRAVITY_X: Float64 = 0.0
    comptime GRAVITY_Y: Float64 = -10.0
    comptime DT: Float64 = 0.02  # 50 FPS
    # Gymnasium: world.Step(1/FPS, 6 * 30, 2 * 30). Box2D stops the position
    # iterations early once every contact and joint is within slop.
    comptime VELOCITY_ITERATIONS: Int = 180
    comptime POSITION_ITERATIONS: Int = 60

    # Lander geometry (matching Gymnasium)
    comptime SCALE: Float64 = 30.0
    comptime LEG_AWAY: Float64 = 20.0 / Self.SCALE
    comptime LEG_DOWN: Float64 = 18.0 / Self.SCALE
    comptime LEG_W: Float64 = 2.0 / Self.SCALE
    comptime LEG_H: Float64 = 8.0 / Self.SCALE
    comptime LANDER_HALF_HEIGHT: Float64 = 17.0 / Self.SCALE
    comptime LANDER_HALF_WIDTH: Float64 = 10.0 / Self.SCALE

    # Lander mass/inertia
    # Box2D mass data of Gymnasium's fixtures (LANDER_POLY at density 5, the
    # legs as 2x8 px boxes at density 1). Inertias are about the centre of
    # mass. The lander's centroid sits 0.101 m above Box2D's body origin;
    # here the body is its centre of mass (a known deviation).
    comptime LANDER_MASS: Float64 = 4.816667
    comptime LANDER_INERTIA: Float64 = 0.783881
    comptime LEG_MASS: Float64 = 0.071111
    comptime LEG_INERTIA: Float64 = 0.001791

    # Leg joint properties
    # Leg joints (Gymnasium revoluteJointDef): motor at +-0.3 rad/s, max
    # torque LEG_SPRING_TORQUE = 40, pushing each leg against its limit.
    comptime LEG_MOTOR_SPEED: Float64 = 0.3
    comptime LEG_MOTOR_TORQUE: Float64 = 40.0

    # Engine power
    comptime MAIN_ENGINE_POWER: Float64 = 13.0
    comptime SIDE_ENGINE_POWER: Float64 = 0.6

    # Engine mount positions (matching lunar_lander_v2.mojo)
    comptime MAIN_ENGINE_Y_OFFSET: Float64 = 4.0 / Self.SCALE
    comptime SIDE_ENGINE_HEIGHT: Float64 = 14.0 / Self.SCALE
    comptime SIDE_ENGINE_AWAY: Float64 = 12.0 / Self.SCALE

    # Reward constants
    comptime CRASH_PENALTY: Float64 = -100.0
    comptime LAND_REWARD: Float64 = 100.0
    comptime MAIN_ENGINE_FUEL_COST: Float64 = 0.30
    comptime SIDE_ENGINE_FUEL_COST: Float64 = 0.03

    # Viewport
    comptime VIEWPORT_W: Float64 = 600.0
    comptime VIEWPORT_H: Float64 = 400.0
    comptime FPS: Float64 = 50.0
    comptime W_UNITS: Float64 = Self.VIEWPORT_W / Self.SCALE
    comptime H_UNITS: Float64 = Self.VIEWPORT_H / Self.SCALE
    comptime HELIPAD_Y: Float64 = Self.H_UNITS / 4.0
    comptime HELIPAD_X: Float64 = Self.W_UNITS / 2.0

    # Physics constants
    # Box2D mixes friction as sqrt(f_a * f_b): legs 0.2 (fixture default) on
    # terrain 0.1. (A lander-body contact ends the episode as a crash.)
    comptime FRICTION: Float64 = 0.14142136
    comptime RESTITUTION: Float64 = 0.0
    comptime BAUMGARTE: Float64 = 0.2
    comptime SLOP: Float64 = 0.005

    # Particle effect constants
    comptime PARTICLE_TTL: Float64 = 0.4  # Particle lifetime in seconds
    comptime TAU: Float64 = Self.DT  # Time step (alias for compatibility)

    # Terrain
    comptime TERRAIN_CHUNKS: Int = 11

    # Body indices (matching lunar_lander_v2.mojo)
    comptime BODY_LANDER: Int = 0
    comptime BODY_LEFT_LEG: Int = 1
    comptime BODY_RIGHT_LEG: Int = 2

    # =============================================================================
    # State Layout for GPUDiscreteEnv (using LunarLanderLayout)
    # =============================================================================

    # Layout type alias for convenience
    comptime LL = LunarLanderLayout

    # Counts (from layout)
    comptime NUM_BODIES: Int = Self.LL.NUM_BODIES
    comptime NUM_SHAPES: Int = Self.LL.NUM_SHAPES
    comptime MAX_CONTACTS: Int = Self.LL.MAX_CONTACTS
    comptime MAX_JOINTS: Int = Self.LL.MAX_JOINTS
    comptime OBS_DIM_VAL: Int = Self.LL.OBS_DIM
    comptime NUM_ACTIONS_VAL: Int = 4  # Discrete: noop, left, main, right
    comptime ACTION_DIM_VAL: Int = 2  # Continuous: [main_throttle, side_throttle]

    # Buffer sizes (from layout)
    comptime BODIES_SIZE: Int = Self.LL.BODIES_SIZE
    comptime FORCES_SIZE: Int = Self.LL.FORCES_SIZE
    comptime JOINTS_SIZE: Int = Self.LL.JOINTS_SIZE
    comptime EDGES_SIZE: Int = Self.LL.EDGES_SIZE
    comptime METADATA_SIZE: Int = Self.LL.METADATA_SIZE

    # Offsets within each environment's state (from layout)
    comptime OBS_OFFSET: Int = Self.LL.OBS_OFFSET
    comptime BODIES_OFFSET: Int = Self.LL.BODIES_OFFSET
    comptime FORCES_OFFSET: Int = Self.LL.FORCES_OFFSET
    comptime JOINTS_OFFSET: Int = Self.LL.JOINTS_OFFSET
    comptime JOINT_COUNT_OFFSET: Int = Self.LL.JOINT_COUNT_OFFSET
    comptime EDGES_OFFSET: Int = Self.LL.EDGES_OFFSET
    comptime EDGE_COUNT_OFFSET: Int = Self.LL.EDGE_COUNT_OFFSET
    comptime METADATA_OFFSET: Int = Self.LL.METADATA_OFFSET

    # Metadata field indices (environment-specific)
    comptime META_STEP_COUNT: Int = 0
    comptime META_TOTAL_REWARD: Int = 1
    comptime META_PREV_SHAPING: Int = 2
    comptime META_DONE: Int = 3
    comptime META_SLEEP_TIME: Int = 4  # Box2D island sleep clock (s)

    # Total state size per environment (from layout)
    comptime STATE_SIZE_VAL: Int = Self.LL.STATE_SIZE

    # Type alias for PhysicsState with our layout (for CPU single-env mode)
    comptime PhysicsHelper = PhysicsState[
        Self.NUM_BODIES,
        Self.STATE_SIZE_VAL,
        Self.BODIES_OFFSET,
        Self.FORCES_OFFSET,
        Self.JOINTS_OFFSET,
        Self.JOINT_COUNT_OFFSET,
        Self.EDGES_OFFSET,
        Self.EDGE_COUNT_OFFSET,
        Self.MAX_JOINTS,
    ]
