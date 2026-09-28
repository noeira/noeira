"""GPU-accelerated physics engine for Noeira.

This module provides a trait-based, modular physics engine that works
on both CPU and GPU with the same interface. Perfect for RL environments
that need fast, batched physics simulation.

Architecture:
- PhysicsWorld: Central orchestrator for physics simulation
- Integrator trait: Velocity and position integration (SemiImplicitEuler)
- CollisionSystem trait: Collision detection (FlatTerrainCollision)
- ConstraintSolver trait: Contact resolution (ImpulseSolver)


"""

# Core constants and types
from .constants import (
    dtype,
    TPB,
    TILE,
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
    IDX_FX,
    IDX_FY,
    IDX_TAU,
    IDX_MASS,
    IDX_INV_MASS,
    IDX_INV_INERTIA,
    IDX_SHAPE,
    SHAPE_POLYGON,
    SHAPE_CIRCLE,
    SHAPE_EDGE,
    SHAPE_COMPOUND,
    MAX_POLYGON_VERTS,
    MAX_COMPOUND_SUBSHAPES,
    CONTACT_BODY_A,
    CONTACT_BODY_B,
    CONTACT_POINT_X,
    CONTACT_POINT_Y,
    CONTACT_NORMAL_X,
    CONTACT_NORMAL_Y,
    CONTACT_DEPTH,
    CONTACT_NORMAL_IMPULSE,
    CONTACT_TANGENT_IMPULSE,
    # Joint constants
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
    JOINT_IMPULSE_Y,
    JOINT_IMPULSE_Z,
    JOINT_LIMIT_STATE,
    JOINT_FLAGS,
    JOINT_IMPULSE_X,
    JOINT_MOTOR_IMPULSE,
    JOINT_REVOLUTE,
    JOINT_DISTANCE,
    JOINT_PRISMATIC,
    JOINT_FLAG_LIMIT_ENABLED,
    JOINT_FLAG_MOTOR_ENABLED,
    MAX_JOINTS_PER_ENV,
    DEFAULT_GRAVITY_X,
    DEFAULT_GRAVITY_Y,
    DEFAULT_DT,
    DEFAULT_VELOCITY_ITERATIONS,
    DEFAULT_POSITION_ITERATIONS,
    DEFAULT_FRICTION,
    DEFAULT_RESTITUTION,
    DEFAULT_BAUMGARTE,
    DEFAULT_SLOP,
    PI,
    TWO_PI,
)

# Core traits
from .traits import Integrator, CollisionSystem, ConstraintSolver

# Implementations
from .integrators import SemiImplicitEuler
from .collision import (
    FlatTerrainCollision,
    EdgeTerrainCollision,
    MAX_TERRAIN_EDGES,
    CirclePolygonCollision,
    detect_circle_vs_body_pair,
)
from .solvers import ImpulseSolver
from .joints import RevoluteJointSolver


# Strided state helper for GPUDiscreteEnv (flat layout)
from .state import PhysicsState, PhysicsStateOwned

# Strided layout and helpers for modular physics (reduces env boilerplate)
from .layout import (
    PhysicsLayout,
    LunarLanderLayout,
    CartPoleLayout,
    AcrobotLayout,
    BipedalWalkerLayout,
    HalfCheetahLayout,
)
from .config import PhysicsConfig
from .env_helpers import PhysicsEnvHelpers

# Lidar raycast module for BipedalWalker
from .lidar import Lidar

# Car physics module (slip-based tire friction for CarRacing)
from .car import (
    # Layout
    CarRacingLayout,
    # Physics components
    WheelFriction,
    TileCollision,
    CarDynamics,
    # GPU kernel
    CarPhysicsKernel,
    # Constants
    SIZE,
    ENGINE_POWER,
    BRAKE_FORCE,
    WHEEL_MOMENT_OF_INERTIA,
    WHEEL_RADIUS,
    FRICTION_LIMIT,
    FRICTION_COEF,
    ROAD_FRICTION,
    GRASS_FRICTION,
    STEERING_LIMIT,
    STEERING_MOTOR_SPEED,
    CAR_DT,
)
