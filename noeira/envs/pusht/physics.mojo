"""PushT physics step orchestrator.

Pure functions over a flat [BATCH, STATE_SIZE] state buffer, reusable from
both the CPU env (PushTEnv) and the GPU batched env (PushTV2).

A single env step is composed of N_SUBSTEPS iterations of:
    1. PD-control: agent.vel += (k_p * (target - pos) + k_v * (0 - vel)) * dt
    2. Wall clamp: append wall contacts for any vertex penetrating the box
    3. CirclePolygonCollision: append agent ↔ T contacts
    4. solve_velocity_single_env  × VEL_ITERATIONS
    5. SemiImplicitEuler.integrate_positions
    6. solve_position_single_env  × POS_ITERATIONS  (Baumgarte)
"""

from std.math import cos, sin, sqrt, log, exp
from layout import LayoutTensor, Layout

from noeira.physics2d.constants import (
    dtype,
    BODY_STATE_SIZE,
    SHAPE_MAX_SIZE,
    CONTACT_DATA_SIZE,
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
)
from noeira.physics2d.integrators.euler import SemiImplicitEuler
from noeira.physics2d.solvers.impulse import ImpulseSolver
from noeira.physics2d.collision.circle_polygon import (
    detect_circle_vs_body_pair,
)

from .constants import PConstants, PushTLayout


# =============================================================================
# Wall clamp: generate contacts for body vertices penetrating the playing box.
# Walls are axis-aligned: x in [WORLD_MIN, WORLD_MAX], y in [WORLD_MIN, WORLD_MAX].
# BODY_B = -1 ⇒ "static" (the ImpulseSolver treats this as inv_mass=0, infinite mass).
# =============================================================================


@always_inline
def _emit_wall_contact[
    BATCH: Int, MAX_CONTACTS: Int, STATE_SIZE: Int
](
    contacts: LayoutTensor[
        dtype,
        Layout.row_major(BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE),
        MutAnyOrigin,
    ],
    env: Int,
    count_in: Int,
    body_a: Int,
    px: Scalar[dtype],
    py: Scalar[dtype],
    nx: Scalar[dtype],
    ny: Scalar[dtype],
    depth: Scalar[dtype],
) -> Int:
    if count_in >= MAX_CONTACTS or depth <= Scalar[dtype](0.0):
        return count_in
    contacts[env, count_in, CONTACT_BODY_A] = Scalar[dtype](body_a)
    contacts[env, count_in, CONTACT_BODY_B] = Scalar[dtype](-1)
    contacts[env, count_in, CONTACT_POINT_X] = px
    contacts[env, count_in, CONTACT_POINT_Y] = py
    contacts[env, count_in, CONTACT_NORMAL_X] = nx
    contacts[env, count_in, CONTACT_NORMAL_Y] = ny
    contacts[env, count_in, CONTACT_DEPTH] = depth
    contacts[env, count_in, CONTACT_NORMAL_IMPULSE] = Scalar[dtype](0.0)
    contacts[env, count_in, CONTACT_TANGENT_IMPULSE] = Scalar[dtype](0.0)
    return count_in + 1


@always_inline
def detect_wall_contacts_single_env[
    BATCH: Int,
    NUM_SHAPES: Int,
    MAX_CONTACTS: Int,
    STATE_SIZE: Int,
](
    state: LayoutTensor[
        dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
    ],
    shapes: LayoutTensor[
        dtype, Layout.row_major(NUM_SHAPES, SHAPE_MAX_SIZE), MutAnyOrigin
    ],
    contacts: LayoutTensor[
        dtype,
        Layout.row_major(BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE),
        MutAnyOrigin,
    ],
    env: Int,
    body_a: Int,
    body_off: Int,
    world_min: Scalar[dtype],
    world_max: Scalar[dtype],
    contact_count_in: Int,
) -> Int:
    """Emit wall contacts for one body. For polygon/compound bodies, walks
    every vertex of every sub-polygon and emits a contact when it crosses a
    wall plane. For circle bodies, emits up to 4 wall contacts based on
    center ± radius."""
    var count = contact_count_in

    var b_x = rebind[Scalar[dtype]](state[env, body_off + IDX_X])
    var b_y = rebind[Scalar[dtype]](state[env, body_off + IDX_Y])
    var b_a = rebind[Scalar[dtype]](state[env, body_off + IDX_ANGLE])
    var shape_idx = Int(state[env, body_off + IDX_SHAPE])
    var shape_type = Int(shapes[shape_idx, 0])
    var cos_a = cos(b_a)
    var sin_a = sin(b_a)

    if shape_type == 1:
        # SHAPE_CIRCLE
        var r = rebind[Scalar[dtype]](shapes[shape_idx, 1])
        var ox = rebind[Scalar[dtype]](shapes[shape_idx, 2])
        var oy = rebind[Scalar[dtype]](shapes[shape_idx, 3])
        var wx = b_x + ox * cos_a - oy * sin_a
        var wy = b_y + ox * sin_a + oy * cos_a
        # Left wall: x = world_min, normal points +x INTO interior
        var pen = world_min - (wx - r)
        if pen > Scalar[dtype](0.0):
            count = _emit_wall_contact[BATCH, MAX_CONTACTS, STATE_SIZE](
                contacts,
                env,
                count,
                body_a,
                world_min,
                wy,
                Scalar[dtype](1.0),
                Scalar[dtype](0.0),
                pen,
            )
        # Right wall: x = world_max, normal -x
        pen = (wx + r) - world_max
        if pen > Scalar[dtype](0.0):
            count = _emit_wall_contact[BATCH, MAX_CONTACTS, STATE_SIZE](
                contacts,
                env,
                count,
                body_a,
                world_max,
                wy,
                Scalar[dtype](-1.0),
                Scalar[dtype](0.0),
                pen,
            )
        # Bottom (y = world_min), normal +y
        pen = world_min - (wy - r)
        if pen > Scalar[dtype](0.0):
            count = _emit_wall_contact[BATCH, MAX_CONTACTS, STATE_SIZE](
                contacts,
                env,
                count,
                body_a,
                wx,
                world_min,
                Scalar[dtype](0.0),
                Scalar[dtype](1.0),
                pen,
            )
        # Top (y = world_max), normal -y
        pen = (wy + r) - world_max
        if pen > Scalar[dtype](0.0):
            count = _emit_wall_contact[BATCH, MAX_CONTACTS, STATE_SIZE](
                contacts,
                env,
                count,
                body_a,
                wx,
                world_max,
                Scalar[dtype](0.0),
                Scalar[dtype](-1.0),
                pen,
            )
        return count

    # Polygon / compound: collect sub-shape list
    var n_sub: Int = 0
    var sub_indices = Array[Int, MAX_COMPOUND_SUBSHAPES](fill=0)
    if shape_type == SHAPE_POLYGON:
        sub_indices[0] = shape_idx
        n_sub = 1
    elif shape_type == SHAPE_COMPOUND:
        n_sub = Int(shapes[shape_idx, 1])
        if n_sub > MAX_COMPOUND_SUBSHAPES:
            n_sub = MAX_COMPOUND_SUBSHAPES
        for s in range(n_sub):
            sub_indices[s] = Int(shapes[shape_idx, 2 + s])
    else:
        return count

    for s in range(n_sub):
        var sub_idx = sub_indices[s]
        if Int(shapes[sub_idx, 0]) != SHAPE_POLYGON:
            continue
        var n_verts = Int(shapes[sub_idx, 1])
        if n_verts > MAX_POLYGON_VERTS:
            n_verts = MAX_POLYGON_VERTS
        for v in range(n_verts):
            if count >= MAX_CONTACTS:
                return count
            var lx = rebind[Scalar[dtype]](shapes[sub_idx, 2 + v * 2])
            var ly = rebind[Scalar[dtype]](shapes[sub_idx, 3 + v * 2])
            var wx = b_x + lx * cos_a - ly * sin_a
            var wy = b_y + lx * sin_a + ly * cos_a
            # 4 walls (use whichever the vertex penetrates the most; emit only
            # one contact per vertex to keep solver well-conditioned).
            var pen_left = world_min - wx
            var pen_right = wx - world_max
            var pen_bot = world_min - wy
            var pen_top = wy - world_max
            var best = Scalar[dtype](0.0)
            var nx = Scalar[dtype](0.0)
            var ny = Scalar[dtype](0.0)
            var px = wx
            var py = wy
            if pen_left > best:
                best = pen_left
                nx = Scalar[dtype](1.0)
                ny = Scalar[dtype](0.0)
                px = world_min
            if pen_right > best:
                best = pen_right
                nx = Scalar[dtype](-1.0)
                ny = Scalar[dtype](0.0)
                px = world_max
            if pen_bot > best:
                best = pen_bot
                nx = Scalar[dtype](0.0)
                ny = Scalar[dtype](1.0)
                py = world_min
            if pen_top > best:
                best = pen_top
                nx = Scalar[dtype](0.0)
                ny = Scalar[dtype](-1.0)
                py = world_max
            if best > Scalar[dtype](0.0):
                count = _emit_wall_contact[
                    BATCH, MAX_CONTACTS, STATE_SIZE
                ](
                    contacts, env, count, body_a, px, py, nx, ny, best
                )
    return count


# =============================================================================
# Single-env PD control step. Kinematic agent has inv_mass=0, so we override
# its velocity manually and skip the contact-driven update.
# =============================================================================


@always_inline
def pd_update_agent_single_env[
    BATCH: Int, STATE_SIZE: Int
](
    state: LayoutTensor[
        dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
    ],
    env: Int,
    agent_off: Int,
    target_x: Scalar[dtype],
    target_y: Scalar[dtype],
    k_p: Scalar[dtype],
    k_v: Scalar[dtype],
    dt: Scalar[dtype],
):
    """Agent is kinematic: we set its velocity from the PD law and integrate
    position with the new velocity (semi-implicit Euler-style)."""
    var px = rebind[Scalar[dtype]](state[env, agent_off + IDX_X])
    var py = rebind[Scalar[dtype]](state[env, agent_off + IDX_Y])
    var vx = rebind[Scalar[dtype]](state[env, agent_off + IDX_VX])
    var vy = rebind[Scalar[dtype]](state[env, agent_off + IDX_VY])
    var ax = k_p * (target_x - px) + k_v * (Scalar[dtype](0.0) - vx)
    var ay = k_p * (target_y - py) + k_v * (Scalar[dtype](0.0) - vy)
    vx = vx + ax * dt
    vy = vy + ay * dt
    state[env, agent_off + IDX_VX] = vx
    state[env, agent_off + IDX_VY] = vy


# =============================================================================
# Single-env full substep used by both CPU and GPU envs.
# =============================================================================


@always_inline
def pusht_substep_single_env[
    BATCH: Int,
    NUM_SHAPES: Int,
    MAX_CONTACTS: Int,
    STATE_SIZE: Int,
    NUM_BODIES: Int,
    BODIES_OFFSET: Int,
    BODY_AGENT_OFFSET: Int,
    BODY_T_OFFSET: Int,
    BODY_AGENT: Int,
    BODY_T: Int,
    CONTACT_COUNT_OFFSET: Int,
    VEL_ITERATIONS: Int,
    POS_ITERATIONS: Int,
](
    state: LayoutTensor[
        dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
    ],
    shapes: LayoutTensor[
        dtype, Layout.row_major(NUM_SHAPES, SHAPE_MAX_SIZE), MutAnyOrigin
    ],
    contacts: LayoutTensor[
        dtype,
        Layout.row_major(BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE),
        MutAnyOrigin,
    ],
    env: Int,
    target_x: Scalar[dtype],
    target_y: Scalar[dtype],
    k_p: Scalar[dtype],
    k_v: Scalar[dtype],
    dt: Scalar[dtype],
    world_min: Scalar[dtype],
    world_max: Scalar[dtype],
    friction: Scalar[dtype],
    restitution: Scalar[dtype],
    baumgarte: Scalar[dtype],
    slop: Scalar[dtype],
    block_damping: Scalar[dtype] = Scalar[dtype](0.0),
):
    """One physics substep for a single environment. Called N_SUBSTEPS times
    per environment step. Shared by CPU and GPU envs.

    `block_damping` matches pymunk's `space.damping` (the per-second velocity
    retention factor for dynamic bodies — 0 = lose 100%/s, 1 = no damping).
    pymunk applies it as `v *= damping^dt` every substep before solving
    contacts. gym_pusht defaults this to 0, so the T-block stops moving as
    soon as the agent loses contact; this is what makes the env feel
    controllable despite the high PD gains. Override to e.g. 1.0 to disable.
    """
    from noeira.physics2d.integrators.euler import SemiImplicitEuler
    from noeira.physics2d.solvers.impulse import ImpulseSolver
    from noeira.physics2d.collision.circle_polygon import (
        detect_circle_vs_body_pair,
    )

    # 0. Damping (pymunk-compatible): v *= damping^dt before contact resolution.
    # We avoid `pow` (slow on GPU and undefined at base=0) by special-casing
    # the two endpoints and using exp/log for everything else. With dt=0.01,
    # block_damping=0 collapses v to zero each substep, matching pymunk default.
    if block_damping <= Scalar[dtype](0.0):
        state[env, BODY_T_OFFSET + IDX_VX] = Scalar[dtype](0.0)
        state[env, BODY_T_OFFSET + IDX_VY] = Scalar[dtype](0.0)
        state[env, BODY_T_OFFSET + IDX_OMEGA] = Scalar[dtype](0.0)
    elif block_damping < Scalar[dtype](1.0):
        # damp_per_step = damping^dt. Use the identity x^y = exp(y*log(x)).
        # block_damping is in (0, 1) so log is well-defined and negative.
        var damp_per_step = exp(log(block_damping) * dt)
        state[env, BODY_T_OFFSET + IDX_VX] = (
            rebind[Scalar[dtype]](state[env, BODY_T_OFFSET + IDX_VX])
            * damp_per_step
        )
        state[env, BODY_T_OFFSET + IDX_VY] = (
            rebind[Scalar[dtype]](state[env, BODY_T_OFFSET + IDX_VY])
            * damp_per_step
        )
        state[env, BODY_T_OFFSET + IDX_OMEGA] = (
            rebind[Scalar[dtype]](state[env, BODY_T_OFFSET + IDX_OMEGA])
            * damp_per_step
        )
    # block_damping >= 1: no damping (preserves all velocity).

    # 1. Kinematic PD-control on agent velocity
    pd_update_agent_single_env[BATCH, STATE_SIZE](
        state, env, BODY_AGENT_OFFSET, target_x, target_y, k_p, k_v, dt
    )

    # 2. Build contact list (walls on T-block, then circle-vs-T)
    state[env, CONTACT_COUNT_OFFSET] = Scalar[dtype](0.0)
    var c = 0
    c = detect_wall_contacts_single_env[
        BATCH, NUM_SHAPES, MAX_CONTACTS, STATE_SIZE
    ](
        state,
        shapes,
        contacts,
        env,
        BODY_T,
        BODY_T_OFFSET,
        world_min,
        world_max,
        c,
    )
    c = detect_circle_vs_body_pair[
        BATCH, NUM_SHAPES, MAX_CONTACTS, STATE_SIZE
    ](
        state,
        shapes,
        contacts,
        env,
        BODY_AGENT_OFFSET,
        BODY_T_OFFSET,
        BODY_AGENT,
        BODY_T,
        c,
    )
    state[env, CONTACT_COUNT_OFFSET] = Scalar[dtype](c)

    # 3. Velocity constraints (Box2D-style: init + warm start once, then
    # friction-then-normal every iteration)
    ImpulseSolver.init_velocity_single_env[
        BATCH, NUM_BODIES, MAX_CONTACTS, STATE_SIZE, BODIES_OFFSET
    ](env, state, contacts, c, restitution)
    for _ in range(VEL_ITERATIONS):
        ImpulseSolver.solve_velocity_single_env[
            BATCH, NUM_BODIES, MAX_CONTACTS, STATE_SIZE, BODIES_OFFSET
        ](env, state, contacts, c, friction)

    # 4. Position integration (advance agent manually since it's kinematic;
    # then run the standard integrator for the T-block).
    state[env, BODY_AGENT_OFFSET + IDX_X] = (
        rebind[Scalar[dtype]](state[env, BODY_AGENT_OFFSET + IDX_X])
        + rebind[Scalar[dtype]](state[env, BODY_AGENT_OFFSET + IDX_VX]) * dt
    )
    state[env, BODY_AGENT_OFFSET + IDX_Y] = (
        rebind[Scalar[dtype]](state[env, BODY_AGENT_OFFSET + IDX_Y])
        + rebind[Scalar[dtype]](state[env, BODY_AGENT_OFFSET + IDX_VY]) * dt
    )
    SemiImplicitEuler.integrate_positions_single_env[
        BATCH, NUM_BODIES, STATE_SIZE, BODIES_OFFSET
    ](env, state, dt)

    # 5. Baumgarte position correction
    # PushT is in pixels: its own slop / baumgarte, and no per-iteration
    # correction cap (Box2D's 0.2 is in meters).
    for _ in range(POS_ITERATIONS):
        if ImpulseSolver.solve_position_single_env[
            BATCH, NUM_BODIES, MAX_CONTACTS, STATE_SIZE, BODIES_OFFSET
        ](
            env,
            state,
            contacts,
            c,
            linear_slop=slop,
            baumgarte=baumgarte,
            max_linear_correction=Scalar[dtype](1e30),
        ):
            break
