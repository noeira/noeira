"""Physics engine constants and type definitions.

This module defines the core constants used throughout the GPU physics engine,
following the pattern established in nn/constants.mojo.
"""

# =============================================================================
# Type Constants
# =============================================================================

# Default data type for physics computations
comptime dtype = DType.float32

# GPU kernel configuration
comptime TILE: Int = 16  # Tile size for tiled operations
comptime TPB: Int = 256  # Threads per block for elementwise ops


@always_inline
def erase_origin[
    dt: DType, o: Origin
](p: Pointer[Scalar[dt], o]) -> Pointer[Scalar[dt], MutAnyOrigin]:
    """Explicitly rebind a device buffer's pointer origin to `MutAnyOrigin`.

    The physics2d GPU kernels build `LayoutTensor[..., MutAnyOrigin]` views over
    `DeviceBuffer` params; `buf.unsafe_ptr()` carries a concrete origin that was
    previously *implicitly* widened to `MutAnyOrigin` (now deprecated). This is
    the physics2d-local analog of `nn.core.module.mptr` — an explicit `rebind`,
    not an implicit conversion, so it preserves behavior while clearing the
    deprecation. (Kept here rather than importing nn to avoid a layer reversal.)
    """
    return rebind[Pointer[Scalar[dt], MutAnyOrigin]](p)

# =============================================================================
# Body State Layout
# =============================================================================
# Bodies are stored as flat arrays with this layout per body:
# [x, y, angle, vx, vy, omega, fx, fy, tau, mass, inv_mass, inv_inertia, shape_idx]

comptime BODY_STATE_SIZE: Int = 13

# Position and orientation
comptime IDX_X: Int = 0
comptime IDX_Y: Int = 1
comptime IDX_ANGLE: Int = 2

# Linear and angular velocity
comptime IDX_VX: Int = 3
comptime IDX_VY: Int = 4
comptime IDX_OMEGA: Int = 5

# Accumulated forces and torque (cleared after each step)
comptime IDX_FX: Int = 6
comptime IDX_FY: Int = 7
comptime IDX_TAU: Int = 8

# Mass properties
comptime IDX_MASS: Int = 9
comptime IDX_INV_MASS: Int = 10
comptime IDX_INV_INERTIA: Int = 11

# Shape reference
comptime IDX_SHAPE: Int = 12

# =============================================================================
# Shape Layout
# =============================================================================
# Shapes are stored with a type discriminator followed by type-specific data

comptime SHAPE_MAX_SIZE: Int = 20  # Max floats per shape (polygon with 8 vertices)

# Shape types
comptime SHAPE_POLYGON: Int = 0
comptime SHAPE_CIRCLE: Int = 1
comptime SHAPE_EDGE: Int = 2
comptime SHAPE_COMPOUND: Int = 3

# Polygon layout: [type, n_verts, v0x, v0y, v1x, v1y, ..., v7x, v7y]
comptime MAX_POLYGON_VERTS: Int = 8

# Circle layout: [type, radius, center_x, center_y]
# Edge layout: [type, v0x, v0y, v1x, v1y, normal_x, normal_y]
# Compound layout: [type, n_subshapes, sub_idx_0, sub_idx_1, ..., sub_idx_7]
# Sub-shapes must be SHAPE_POLYGON (convex). Used to model concave bodies like
# the PushT T-block as a union of convex sub-polygons that share one rigid body.
comptime MAX_COMPOUND_SUBSHAPES: Int = 8

# =============================================================================
# Contact Layout
# =============================================================================
# Contacts store collision information for constraint solving

comptime CONTACT_DATA_SIZE: Int = 14

# Written by collision detection. The normal points from B toward A.
comptime CONTACT_BODY_A: Int = 0
comptime CONTACT_BODY_B: Int = 1  # -1 for static/ground
comptime CONTACT_POINT_X: Int = 2
comptime CONTACT_POINT_Y: Int = 3
comptime CONTACT_NORMAL_X: Int = 4
comptime CONTACT_NORMAL_Y: Int = 5
comptime CONTACT_DEPTH: Int = 6  # penetration at detection (> 0)
# Accumulated impulses of the current step (detection zeroes them).
comptime CONTACT_NORMAL_IMPULSE: Int = 7
comptime CONTACT_TANGENT_IMPULSE: Int = 8
# Written by `ImpulseSolver.init_velocity_single_env` (Box2D's
# InitializeVelocityConstraints): the contact point in each body's frame, so
# the position solve can recompute the separation from the current poses
# (a static B stores the world point), and the restitution velocity bias.
comptime CONTACT_LOCAL_AX: Int = 9
comptime CONTACT_LOCAL_AY: Int = 10
comptime CONTACT_LOCAL_BX: Int = 11
comptime CONTACT_LOCAL_BY: Int = 12
comptime CONTACT_VELOCITY_BIAS: Int = 13

# =============================================================================
# Joint Layout
# =============================================================================
# Joints connect two bodies and constrain their relative motion
# Revolute joint layout (Box2D 2.3 b2RevoluteJoint: motor + limit, no spring):
# [type, body_a, body_b, anchor_ax, anchor_ay, anchor_bx, anchor_by,
#  ref_angle, lower_limit, upper_limit, max_motor_torque, motor_speed,
#  impulse_y, impulse_z, flags, impulse_x, motor_impulse, limit_state]

comptime JOINT_DATA_SIZE: Int = 18

# Joint types
comptime JOINT_REVOLUTE: Int = 0
comptime JOINT_DISTANCE: Int = 1  # For future use
comptime JOINT_PRISMATIC: Int = 2  # For future use

# Joint data indices
comptime JOINT_TYPE: Int = 0
comptime JOINT_BODY_A: Int = 1
comptime JOINT_BODY_B: Int = 2
comptime JOINT_ANCHOR_AX: Int = 3  # Local anchor on body A
comptime JOINT_ANCHOR_AY: Int = 4
comptime JOINT_ANCHOR_BX: Int = 5  # Local anchor on body B
comptime JOINT_ANCHOR_BY: Int = 6
comptime JOINT_REF_ANGLE: Int = 7  # Reference angle (angle_b - angle_a at creation)
comptime JOINT_LOWER_LIMIT: Int = 8  # Lower angle limit
comptime JOINT_UPPER_LIMIT: Int = 9  # Upper angle limit
comptime JOINT_MAX_MOTOR_TORQUE: Int = 10
comptime JOINT_MOTOR_SPEED: Int = 11  # Target motor speed
comptime JOINT_FLAGS: Int = 14  # Bit flags: 1=limit_enabled, 2=motor_enabled
# Accumulated impulses, carried across steps for warm starting (Box2D's
# m_impulse (x, y = point, z = limit) and m_motorImpulse); zero them when the
# joint is (re)created.
comptime JOINT_IMPULSE_X: Int = 15
comptime JOINT_IMPULSE_Y: Int = 12
comptime JOINT_IMPULSE_Z: Int = 13
comptime JOINT_MOTOR_IMPULSE: Int = 16
comptime JOINT_LIMIT_STATE: Int = 17  # one of the JOINT_LIMIT_* states below

# Joint flags
comptime JOINT_FLAG_LIMIT_ENABLED: Int = 1
comptime JOINT_FLAG_MOTOR_ENABLED: Int = 2

# Limit states (Box2D b2LimitState)
comptime JOINT_LIMIT_INACTIVE: Int = 0
comptime JOINT_LIMIT_AT_LOWER: Int = 1
comptime JOINT_LIMIT_AT_UPPER: Int = 2
comptime JOINT_LIMIT_EQUAL: Int = 3

# Maximum joints per environment
comptime MAX_JOINTS_PER_ENV: Int = 8

# =============================================================================
# Default Physics Constants
# =============================================================================

comptime DEFAULT_GRAVITY_X: Float64 = 0.0
comptime DEFAULT_GRAVITY_Y: Float64 = -10.0
comptime DEFAULT_DT: Float64 = 0.02  # 50 FPS

# Solver defaults
comptime DEFAULT_VELOCITY_ITERATIONS: Int = 6
comptime DEFAULT_POSITION_ITERATIONS: Int = 2

# Contact physics defaults
comptime DEFAULT_FRICTION: Float64 = 0.3
comptime DEFAULT_RESTITUTION: Float64 = 0.0
comptime DEFAULT_BAUMGARTE: Float64 = 0.2  # Position correction factor
comptime DEFAULT_SLOP: Float64 = 0.005  # Penetration allowance

# =============================================================================
# Mathematical Constants
# =============================================================================

from std.math import pi

comptime PI: Float64 = pi

# =============================================================================
# Box2D 2.3 solver settings (b2Settings.h, pybox2d 2.3.10 = Gymnasium's Box2D)
# =============================================================================
comptime B2_LINEAR_SLOP: Float64 = 0.005
comptime B2_ANGULAR_SLOP: Float64 = 2.0 / 180.0 * pi
comptime B2_BAUMGARTE: Float64 = 0.2
comptime B2_MAX_LINEAR_CORRECTION: Float64 = 0.2
comptime B2_MAX_ANGULAR_CORRECTION: Float64 = 8.0 / 180.0 * pi
comptime B2_VELOCITY_THRESHOLD: Float64 = 1.0  # restitution applies above it
comptime B2_MAX_TRANSLATION: Float64 = 2.0  # per step
comptime B2_MAX_ROTATION: Float64 = 0.5 * pi  # per step
comptime B2_TIME_TO_SLEEP: Float64 = 0.5
comptime B2_LINEAR_SLEEP_TOLERANCE: Float64 = 0.01
comptime B2_ANGULAR_SLEEP_TOLERANCE: Float64 = 2.0 / 180.0 * pi
comptime TWO_PI: Float64 = 2.0 * pi
