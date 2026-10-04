"""PushT environment constants and state layout.

Physics + reward match `gym_pusht` (pymunk-based reference) where reasonable:
- 512x512 world (walls at [5, 506] in both axes)
- Kinematic circle agent (radius=15) driven by PD on target position
- Dynamic T-block: 2 rectangles welded as one rigid body
- Goal pose fixed at (256, 256, π/4)
- Reward = clip(coverage / 0.95, 0, 1) where coverage = area(T ∩ goal_T) / area(goal_T)
- Episode terminates when coverage > 0.95
"""

from std.math import pi
from noeira.physics2d.constants import (
    BODY_STATE_SIZE,
    SHAPE_MAX_SIZE,
    CONTACT_DATA_SIZE,
)


struct PConstants:
    """Physics + episode constants for PushT."""

    # World box (pymunk walls live at [5, 506] in both axes).
    comptime WORLD_MIN: Float64 = 5.0
    comptime WORLD_MAX: Float64 = 506.0
    comptime WALL_RADIUS: Float64 = 2.0  # pymunk segment thickness

    # Agent
    comptime AGENT_RADIUS: Float64 = 15.0
    comptime K_P: Float64 = 100.0  # PD proportional gain
    comptime K_V: Float64 = 20.0  # PD velocity gain
    comptime ACTION_LOW: Float64 = 0.0
    comptime ACTION_HIGH: Float64 = 512.0

    # T-block geometry (pymunk reference: scale=30, length=4)
    comptime T_SCALE: Float64 = 30.0
    comptime T_LENGTH: Float64 = 4.0  # ratio so rect1 width = length*scale=120
    comptime T_COG_Y: Float64 = 45.0
    """The T's centre of gravity in its body frame: (0, T_COG_Y). The
    reference sets `body.center_of_gravity` to the mean of the two shapes'
    centroids, (0, 15) and (0, 75). Chipmunk integrates a body ABOUT its cog
    while `body.position` (= the dataset's block x, y) stays the ORIGIN. Our
    physics therefore integrates the T's cog (`t_body_from_origin`) with
    cog-relative collision shapes, and every pose leaving the physics is
    converted back (`t_origin_from_body`); geometry, keypoints, coverage, the
    goal and rendering stay in the origin frame. Rotating about the origin
    instead put the block a median 13 px off pymunk after 25 steps (G4a,
    docs/LEWM_REOPEN_PLAN.md P4)."""
    # rect1 (long horizontal bar): local [-60,0] to [60,30]
    # rect2 (vertical stem):       local [-15,30] to [15,120]
    # In *body local frame*. The body's CoM (set in setup) aligns roughly with
    # the centroid of the union.

    # T-block mass + inertia (pymunk computes moment_for_poly per rect)
    comptime T_MASS: Float64 = 1.0
    # Moments for each rectangle (pymunk's moment_for_poly for the two rects
    # about each rect's own centroid). pymunk uses the same formula and sums:
    # inertia1 = m * (Σ |v_i x v_{i+1}|(|v_i|^2 + |v_i.v_{i+1}| + |v_{i+1}|^2))
    #            / (6 * Σ |v_i x v_{i+1}|).
    # For a 120×30 rectangle around its centroid: I = m*(w² + h²)/12 = m*15300/12.
    # For a 30×90 rectangle around its centroid:  I = m*(w² + h²)/12 = m*9000/12.
    # The pymunk code uses inertia1+inertia2 but with both computed from
    # `vertices=vertices1` — that's a quirk we replicate so dynamics match.
    # (See gym_pusht/envs/pusht.py: `inertia2 = pymunk.moment_for_poly(mass, vertices=vertices1)`)
    # I_total = 2 * moment_for_poly(mass, rect1_verts about (0,0) origin)
    # where the rect1 verts are NOT centered on origin so parallel-axis applies.
    # We just precompute it numerically from the vertex list at body construct.

    # PD substepping
    comptime DT: Float64 = 0.01  # inner physics step
    comptime CONTROL_HZ: Float64 = 10.0
    comptime N_SUBSTEPS: Int = 10  # = int(1 / (DT * CONTROL_HZ))

    # Goal
    comptime GOAL_X: Float64 = 256.0
    comptime GOAL_Y: Float64 = 256.0
    comptime GOAL_ANGLE: Float64 = pi / 4.0
    comptime SUCCESS_THRESHOLD: Float64 = 0.95

    # Reset bounds (matches pymunk reference)
    comptime AGENT_RESET_LOW: Float64 = 50.0
    comptime AGENT_RESET_HIGH: Float64 = 450.0
    comptime BLOCK_RESET_LOW: Float64 = 100.0
    comptime BLOCK_RESET_HIGH: Float64 = 400.0

    # Solver
    comptime VEL_ITERATIONS: Int = 6
    comptime POS_ITERATIONS: Int = 2
    comptime FRICTION: Float64 = 0.0
    """The reference PushT is FRICTIONLESS. It writes `body.friction = 1`, but
    friction is a pymunk SHAPE property (default 0) and `Body` has none, so
    the assignment is a silent no-op: every shape — agent, both T polygons,
    walls — has friction 0.0 (checked on pymunk 7.3 / stable-worldmodel
    0.0.6). This was 1.0, read as "pymunk's default"."""
    comptime RESTITUTION: Float64 = 0.0
    # Per-second velocity retention factor applied to the dynamic T-block
    # before contact resolution. Matches pymunk's `space.damping` semantics
    # (0 = lose 100%/s, 1 = no damping). gym_pusht's default is 0, which
    # makes the T stop almost instantly once the agent loses contact and is
    # what makes the env playable despite the high PD gains.
    comptime BLOCK_DAMPING: Float64 = 0.0

    # Episode
    comptime MAX_STEPS: Int = 300

    # Dimensions
    comptime KEYPOINTS_DIM: Int = 16  # 8 keypoints × 2
    comptime AGENT_POS_DIM: Int = 2
    comptime OBS_DIM: Int = 18  # keypoints (16) + agent_pos (2)
    comptime ACTION_DIM: Int = 2


struct PushTShapeBuf:
    """Compile-time layout of the shape buffer used by physics2d.

    Shapes (5 entries, sized SHAPE_MAX_SIZE = 20 floats each):
        0: agent circle  — [SHAPE_CIRCLE, radius=15, cx=0, cy=0, ...]
        1: T rect-long   — SHAPE_POLYGON, n_verts=4, 8 floats verts (CCW)
        2: T rect-stem   — SHAPE_POLYGON, n_verts=4, 8 floats verts (CCW)
        3: T compound    — SHAPE_COMPOUND, n_sub=2, sub_idx_0=1, sub_idx_1=2
        4: goal T compound (identical structure to 3; kept separate so the
           runtime can pose it at the goal without disturbing the live block)
    """

    comptime NUM_SHAPES: Int = 5
    comptime SHAPE_AGENT: Int = 0
    comptime SHAPE_T_RECT_LONG: Int = 1
    comptime SHAPE_T_RECT_STEM: Int = 2
    comptime SHAPE_T_COMPOUND: Int = 3
    comptime SHAPE_GOAL_COMPOUND: Int = 4


struct PushTLayout:
    """Flat [BATCH, STATE_SIZE] layout for PushT.

    Per-env layout (offsets in float-slots):
        OBS           [0 .. OBS_DIM)               = 18 floats (16 keypoints + 2 agent_pos)
        BODIES        [OBS_DIM .. + 2*13)          = 26 floats (agent + T)
        CONTACTS_WS   [+ MAX_CONTACTS*9)           = 16*9 = 144 floats (workspace)
        CONTACT_COUNT [+ 1)                        = 1 float
        METADATA      [+ METADATA_SIZE)            = step, done, total_reward, coverage
    """

    comptime NUM_BODIES: Int = 2  # agent (0), T-block (1)
    comptime BODY_AGENT: Int = 0
    comptime BODY_T: Int = 1
    comptime MAX_CONTACTS: Int = 16
    comptime METADATA_SIZE: Int = 4

    # Body field offsets (within env state)
    comptime OBS_OFFSET: Int = 0
    comptime OBS_DIM: Int = PConstants.OBS_DIM

    comptime BODIES_OFFSET: Int = Self.OBS_DIM
    comptime BODIES_SIZE: Int = Self.NUM_BODIES * BODY_STATE_SIZE

    comptime CONTACTS_WS_OFFSET: Int = Self.BODIES_OFFSET + Self.BODIES_SIZE
    comptime CONTACTS_WS_SIZE: Int = Self.MAX_CONTACTS * CONTACT_DATA_SIZE

    comptime CONTACT_COUNT_OFFSET: Int = (
        Self.CONTACTS_WS_OFFSET + Self.CONTACTS_WS_SIZE
    )

    comptime METADATA_OFFSET: Int = Self.CONTACT_COUNT_OFFSET + 1
    comptime META_STEP: Int = 0
    comptime META_DONE: Int = 1
    comptime META_TOTAL_REWARD: Int = 2
    comptime META_COVERAGE: Int = 3

    comptime STATE_SIZE: Int = Self.METADATA_OFFSET + Self.METADATA_SIZE

    # Absolute body-state offsets
    comptime BODY_AGENT_OFFSET: Int = (
        Self.BODIES_OFFSET + Self.BODY_AGENT * BODY_STATE_SIZE
    )
    comptime BODY_T_OFFSET: Int = (
        Self.BODIES_OFFSET + Self.BODY_T * BODY_STATE_SIZE
    )
