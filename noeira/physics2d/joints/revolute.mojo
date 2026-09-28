"""Revolute joint — a port of Box2D 2.3's b2RevoluteJoint.

Reference: `references/pybox2d-2.3.10/Box2D/Dynamics/Joints/b2RevoluteJoint.cpp`
(the Box2D that Gymnasium's LunarLander / BipedalWalker / CarRacing run).

A revolute joint pins an anchor on body A to an anchor on body B. Optionally:
- an angle limit on `angle_b - angle_a - reference_angle`,
- a motor driving the relative angular velocity toward `motor_speed` with
  at most `max_motor_torque`.

Per world step (see the env steps), for every joint:
  1. `init_velocity_single_env`  — InitVelocityConstraints: limit state, then
     warm start from the impulses accumulated last step.
  2. `solve_velocity_single_env` — SolveVelocityConstraints, once per velocity
     iteration: motor, then limit + point as one 3x3 block (or point alone).
  3. `solve_position_single_env` — SolvePositionConstraints, once per position
     iteration: angular limit, then the point (non-linear Gauss-Seidel).
     Returns whether the joint is within slop, for Box2D's early exit.

Bodies are stored at their centre of mass, so Box2D's `localAnchor -
localCenter` is the stored local anchor. Box2D 2.3 has no joint spring, and
neither does this port.
"""

from std.math import cos, sin, sqrt
from layout import LayoutTensor, Layout

from ..constants import (
    dtype,
    BODY_STATE_SIZE,
    JOINT_DATA_SIZE,
    IDX_X,
    IDX_Y,
    IDX_ANGLE,
    IDX_VX,
    IDX_VY,
    IDX_OMEGA,
    IDX_INV_MASS,
    IDX_INV_INERTIA,
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
    JOINT_FLAGS,
    JOINT_IMPULSE_X,
    JOINT_IMPULSE_Y,
    JOINT_IMPULSE_Z,
    JOINT_MOTOR_IMPULSE,
    JOINT_LIMIT_STATE,
    JOINT_REVOLUTE,
    JOINT_FLAG_LIMIT_ENABLED,
    JOINT_FLAG_MOTOR_ENABLED,
    JOINT_LIMIT_INACTIVE,
    JOINT_LIMIT_AT_LOWER,
    JOINT_LIMIT_AT_UPPER,
    JOINT_LIMIT_EQUAL,
    B2_LINEAR_SLOP,
    B2_ANGULAR_SLOP,
    B2_MAX_ANGULAR_CORRECTION,
)


@always_inline
def _ld[
    BATCH: Int, STATE_SIZE: Int
](
    state: LayoutTensor[dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin],
    env: Int,
    i: Int,
) -> Scalar[dtype]:
    return rebind[Scalar[dtype]](state[env, i])


@always_inline
def _clamp(x: Scalar[dtype], lo: Scalar[dtype], hi: Scalar[dtype]) -> Scalar[dtype]:
    return lo if x < lo else (hi if x > hi else x)


@fieldwise_init
struct _JointFrame(Copyable, Movable):
    """One joint's per-solve quantities (Box2D recomputes them per call too)."""

    var off_a: Int
    var off_b: Int
    var ma: Scalar[dtype]
    var mb: Scalar[dtype]
    var ia: Scalar[dtype]
    var ib: Scalar[dtype]
    var rax: Scalar[dtype]
    var ray: Scalar[dtype]
    var rbx: Scalar[dtype]
    var rby: Scalar[dtype]


@always_inline
def _joint_frame[
    BATCH: Int, STATE_SIZE: Int, BODIES_OFFSET: Int
](
    state: LayoutTensor[dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin],
    env: Int,
    joint_off: Int,
) -> _JointFrame:
    """Body offsets, inverse masses and world-frame anchor arms r_a, r_b."""
    var body_a = Int(_ld(state, env, joint_off + JOINT_BODY_A))
    var body_b = Int(_ld(state, env, joint_off + JOINT_BODY_B))
    var off_a = BODIES_OFFSET + body_a * BODY_STATE_SIZE
    var off_b = BODIES_OFFSET + body_b * BODY_STATE_SIZE
    var aa = _ld(state, env, off_a + IDX_ANGLE)
    var ab = _ld(state, env, off_b + IDX_ANGLE)
    var lax = _ld(state, env, joint_off + JOINT_ANCHOR_AX)
    var lay = _ld(state, env, joint_off + JOINT_ANCHOR_AY)
    var lbx = _ld(state, env, joint_off + JOINT_ANCHOR_BX)
    var lby = _ld(state, env, joint_off + JOINT_ANCHOR_BY)
    var ca = cos(aa)
    var sa = sin(aa)
    var cb = cos(ab)
    var sb = sin(ab)
    return _JointFrame(
        off_a=off_a,
        off_b=off_b,
        ma=_ld(state, env, off_a + IDX_INV_MASS),
        mb=_ld(state, env, off_b + IDX_INV_MASS),
        ia=_ld(state, env, off_a + IDX_INV_INERTIA),
        ib=_ld(state, env, off_b + IDX_INV_INERTIA),
        rax=ca * lax - sa * lay,
        ray=sa * lax + ca * lay,
        rbx=cb * lbx - sb * lby,
        rby=sb * lbx + cb * lby,
    )


struct RevoluteJointSolver:
    """Box2D 2.3 revolute joint: warm-started sequential impulses."""

    # =========================================================================
    # Joint creation / reset
    # =========================================================================

    @always_inline
    @staticmethod
    def write_joint[
        BATCH: Int, STATE_SIZE: Int
    ](
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
        joint_off: Int,
        body_a: Int,
        body_b: Int,
        anchor_ax: Scalar[dtype],
        anchor_ay: Scalar[dtype],
        anchor_bx: Scalar[dtype],
        anchor_by: Scalar[dtype],
        reference_angle: Scalar[dtype],
        lower_limit: Scalar[dtype],
        upper_limit: Scalar[dtype],
        enable_limit: Bool,
        enable_motor: Bool,
        motor_speed: Scalar[dtype],
        max_motor_torque: Scalar[dtype],
    ):
        """Write a revolute joint record at `joint_off` (b2RevoluteJointDef
        fields) with zeroed warm-start state. The ONE writer every env uses."""
        states[env, joint_off + JOINT_TYPE] = Scalar[dtype](JOINT_REVOLUTE)
        states[env, joint_off + JOINT_BODY_A] = Scalar[dtype](body_a)
        states[env, joint_off + JOINT_BODY_B] = Scalar[dtype](body_b)
        states[env, joint_off + JOINT_ANCHOR_AX] = anchor_ax
        states[env, joint_off + JOINT_ANCHOR_AY] = anchor_ay
        states[env, joint_off + JOINT_ANCHOR_BX] = anchor_bx
        states[env, joint_off + JOINT_ANCHOR_BY] = anchor_by
        states[env, joint_off + JOINT_REF_ANGLE] = reference_angle
        states[env, joint_off + JOINT_LOWER_LIMIT] = lower_limit
        states[env, joint_off + JOINT_UPPER_LIMIT] = upper_limit
        states[env, joint_off + JOINT_MOTOR_SPEED] = motor_speed
        states[env, joint_off + JOINT_MAX_MOTOR_TORQUE] = max_motor_torque
        var flags = 0
        if enable_limit:
            flags = flags | JOINT_FLAG_LIMIT_ENABLED
        if enable_motor:
            flags = flags | JOINT_FLAG_MOTOR_ENABLED
        states[env, joint_off + JOINT_FLAGS] = Scalar[dtype](flags)
        RevoluteJointSolver.clear_warm_start[BATCH, STATE_SIZE](
            states, env, joint_off
        )

    @always_inline
    @staticmethod
    def clear_warm_start[
        BATCH: Int, STATE_SIZE: Int
    ](
        states: LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
        joint_off: Int,
    ):
        """Zero one joint's accumulated impulses and limit state — what a
        freshly created b2Joint starts with. Call on every env reset for
        joints whose fields are rewritten in place."""
        states[env, joint_off + JOINT_IMPULSE_X] = Scalar[dtype](0)
        states[env, joint_off + JOINT_IMPULSE_Y] = Scalar[dtype](0)
        states[env, joint_off + JOINT_IMPULSE_Z] = Scalar[dtype](0)
        states[env, joint_off + JOINT_MOTOR_IMPULSE] = Scalar[dtype](0)
        states[env, joint_off + JOINT_LIMIT_STATE] = Scalar[dtype](
            JOINT_LIMIT_INACTIVE
        )

    # =========================================================================
    # InitVelocityConstraints
    # =========================================================================

    @always_inline
    @staticmethod
    def init_velocity_single_env[
        BATCH: Int,
        NUM_BODIES: Int,
        MAX_JOINTS: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
        JOINTS_OFFSET: Int,
    ](
        env: Int,
        state: LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        joint_count: Int,
    ):
        """Update each joint's limit state and apply the warm-start impulses
        (b2RevoluteJoint::InitVelocityConstraints, dtRatio = 1)."""
        for j in range(MAX_JOINTS):
            if j >= joint_count:
                break
            var joff = JOINTS_OFFSET + j * JOINT_DATA_SIZE
            if Int(_ld(state, env, joff + JOINT_TYPE)) != JOINT_REVOLUTE:
                continue
            var f = _joint_frame[BATCH, STATE_SIZE, BODIES_OFFSET](
                state, env, joff
            )
            var flags = Int(_ld(state, env, joff + JOINT_FLAGS))
            var fixed_rotation = (f.ia + f.ib) == Scalar[dtype](0)

            if (flags & JOINT_FLAG_MOTOR_ENABLED) == 0 or fixed_rotation:
                state[env, joff + JOINT_MOTOR_IMPULSE] = Scalar[dtype](0)

            var limit_state = Int(_ld(state, env, joff + JOINT_LIMIT_STATE))
            if (flags & JOINT_FLAG_LIMIT_ENABLED) != 0 and not fixed_rotation:
                var angle = (
                    _ld(state, env, f.off_b + IDX_ANGLE)
                    - _ld(state, env, f.off_a + IDX_ANGLE)
                    - _ld(state, env, joff + JOINT_REF_ANGLE)
                )
                var lower = _ld(state, env, joff + JOINT_LOWER_LIMIT)
                var upper = _ld(state, env, joff + JOINT_UPPER_LIMIT)
                if abs(upper - lower) < Scalar[dtype](2.0 * B2_ANGULAR_SLOP):
                    limit_state = JOINT_LIMIT_EQUAL
                elif angle <= lower:
                    if limit_state != JOINT_LIMIT_AT_LOWER:
                        state[env, joff + JOINT_IMPULSE_Z] = Scalar[dtype](0)
                    limit_state = JOINT_LIMIT_AT_LOWER
                elif angle >= upper:
                    if limit_state != JOINT_LIMIT_AT_UPPER:
                        state[env, joff + JOINT_IMPULSE_Z] = Scalar[dtype](0)
                    limit_state = JOINT_LIMIT_AT_UPPER
                else:
                    limit_state = JOINT_LIMIT_INACTIVE
                    state[env, joff + JOINT_IMPULSE_Z] = Scalar[dtype](0)
            else:
                limit_state = JOINT_LIMIT_INACTIVE
            state[env, joff + JOINT_LIMIT_STATE] = Scalar[dtype](limit_state)

            # Warm start.
            var px = _ld(state, env, joff + JOINT_IMPULSE_X)
            var py = _ld(state, env, joff + JOINT_IMPULSE_Y)
            var pz = _ld(state, env, joff + JOINT_IMPULSE_Z)
            var pm = _ld(state, env, joff + JOINT_MOTOR_IMPULSE)
            var cross_a = f.rax * py - f.ray * px
            var cross_b = f.rbx * py - f.rby * px
            state[env, f.off_a + IDX_VX] = _ld(state, env, f.off_a + IDX_VX) - f.ma * px
            state[env, f.off_a + IDX_VY] = _ld(state, env, f.off_a + IDX_VY) - f.ma * py
            state[env, f.off_a + IDX_OMEGA] = _ld(
                state, env, f.off_a + IDX_OMEGA
            ) - f.ia * (cross_a + pm + pz)
            state[env, f.off_b + IDX_VX] = _ld(state, env, f.off_b + IDX_VX) + f.mb * px
            state[env, f.off_b + IDX_VY] = _ld(state, env, f.off_b + IDX_VY) + f.mb * py
            state[env, f.off_b + IDX_OMEGA] = _ld(
                state, env, f.off_b + IDX_OMEGA
            ) + f.ib * (cross_b + pm + pz)

    # =========================================================================
    # SolveVelocityConstraints
    # =========================================================================

    @always_inline
    @staticmethod
    def solve_velocity_single_env[
        BATCH: Int,
        NUM_BODIES: Int,
        MAX_JOINTS: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
        JOINTS_OFFSET: Int,
    ](
        env: Int,
        state: LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        joint_count: Int,
        dt: Scalar[dtype],
    ):
        """One velocity iteration over all joints
        (b2RevoluteJoint::SolveVelocityConstraints)."""
        for j in range(MAX_JOINTS):
            if j >= joint_count:
                break
            var joff = JOINTS_OFFSET + j * JOINT_DATA_SIZE
            if Int(_ld(state, env, joff + JOINT_TYPE)) != JOINT_REVOLUTE:
                continue
            var f = _joint_frame[BATCH, STATE_SIZE, BODIES_OFFSET](
                state, env, joff
            )
            var ma = f.ma
            var mb = f.mb
            var ia = f.ia
            var ib = f.ib
            var rax = f.rax
            var ray = f.ray
            var rbx = f.rbx
            var rby = f.rby

            var vax = _ld(state, env, f.off_a + IDX_VX)
            var vay = _ld(state, env, f.off_a + IDX_VY)
            var wa = _ld(state, env, f.off_a + IDX_OMEGA)
            var vbx = _ld(state, env, f.off_b + IDX_VX)
            var vby = _ld(state, env, f.off_b + IDX_VY)
            var wb = _ld(state, env, f.off_b + IDX_OMEGA)

            var flags = Int(_ld(state, env, joff + JOINT_FLAGS))
            var limit_state = Int(_ld(state, env, joff + JOINT_LIMIT_STATE))
            var fixed_rotation = (ia + ib) == Scalar[dtype](0)

            # Motor.
            if (
                (flags & JOINT_FLAG_MOTOR_ENABLED) != 0
                and limit_state != JOINT_LIMIT_EQUAL
                and not fixed_rotation
            ):
                var motor_mass = Scalar[dtype](1) / (ia + ib)
                var cdot = wb - wa - _ld(state, env, joff + JOINT_MOTOR_SPEED)
                var impulse = -motor_mass * cdot
                var old_impulse = _ld(state, env, joff + JOINT_MOTOR_IMPULSE)
                var max_impulse = dt * _ld(
                    state, env, joff + JOINT_MAX_MOTOR_TORQUE
                )
                var new_impulse = _clamp(
                    old_impulse + impulse, -max_impulse, max_impulse
                )
                state[env, joff + JOINT_MOTOR_IMPULSE] = new_impulse
                impulse = new_impulse - old_impulse
                wa = wa - ia * impulse
                wb = wb + ib * impulse

            # Effective mass K (Box2D's m_mass, symmetric 3x3).
            var k11 = ma + mb + ray * ray * ia + rby * rby * ib
            var k12 = -ray * rax * ia - rby * rbx * ib
            var k13 = -ray * ia - rby * ib
            var k22 = ma + mb + rax * rax * ia + rbx * rbx * ib
            var k23 = rax * ia + rbx * ib
            var k33 = ia + ib

            var cdot1x = vbx - wb * rby - vax + wa * ray
            var cdot1y = vby + wb * rbx - vay - wa * rax

            if (
                (flags & JOINT_FLAG_LIMIT_ENABLED) != 0
                and limit_state != JOINT_LIMIT_INACTIVE
                and not fixed_rotation
            ):
                var cdot2 = wb - wa
                # impulse = -K^-1 * Cdot  (b2Mat33::Solve33)
                var s3 = _solve33(
                    k11, k12, k13, k22, k23, k33, -cdot1x, -cdot1y, -cdot2
                )
                var ix = s3[0]
                var iy = s3[1]
                var iz = s3[2]
                var acc_x = _ld(state, env, joff + JOINT_IMPULSE_X)
                var acc_y = _ld(state, env, joff + JOINT_IMPULSE_Y)
                var acc_z = _ld(state, env, joff + JOINT_IMPULSE_Z)
                if limit_state == JOINT_LIMIT_EQUAL:
                    acc_x += ix
                    acc_y += iy
                    acc_z += iz
                elif (
                    limit_state == JOINT_LIMIT_AT_LOWER
                    and acc_z + iz < Scalar[dtype](0)
                ) or (
                    limit_state == JOINT_LIMIT_AT_UPPER
                    and acc_z + iz > Scalar[dtype](0)
                ):
                    # The limit would pull: drop it, solve the point alone
                    # with the limit impulse removed (Box2D's reduced 2x2).
                    var rhs_x = -cdot1x + acc_z * k13
                    var rhs_y = -cdot1y + acc_z * k23
                    var s2 = _solve22(k11, k12, k22, rhs_x, rhs_y)
                    ix = s2[0]
                    iy = s2[1]
                    iz = -acc_z
                    acc_x += ix
                    acc_y += iy
                    acc_z = Scalar[dtype](0)
                else:
                    acc_x += ix
                    acc_y += iy
                    acc_z += iz
                state[env, joff + JOINT_IMPULSE_X] = acc_x
                state[env, joff + JOINT_IMPULSE_Y] = acc_y
                state[env, joff + JOINT_IMPULSE_Z] = acc_z
                vax -= ma * ix
                vay -= ma * iy
                wa -= ia * (rax * iy - ray * ix + iz)
                vbx += mb * ix
                vby += mb * iy
                wb += ib * (rbx * iy - rby * ix + iz)
            else:
                # Point-to-point only (b2Mat33::Solve22 on the 2x2 block).
                var s2 = _solve22(k11, k12, k22, -cdot1x, -cdot1y)
                var ix = s2[0]
                var iy = s2[1]
                state[env, joff + JOINT_IMPULSE_X] = (
                    _ld(state, env, joff + JOINT_IMPULSE_X) + ix
                )
                state[env, joff + JOINT_IMPULSE_Y] = (
                    _ld(state, env, joff + JOINT_IMPULSE_Y) + iy
                )
                vax -= ma * ix
                vay -= ma * iy
                wa -= ia * (rax * iy - ray * ix)
                vbx += mb * ix
                vby += mb * iy
                wb += ib * (rbx * iy - rby * ix)

            state[env, f.off_a + IDX_VX] = vax
            state[env, f.off_a + IDX_VY] = vay
            state[env, f.off_a + IDX_OMEGA] = wa
            state[env, f.off_b + IDX_VX] = vbx
            state[env, f.off_b + IDX_VY] = vby
            state[env, f.off_b + IDX_OMEGA] = wb

    # =========================================================================
    # SolvePositionConstraints
    # =========================================================================

    @always_inline
    @staticmethod
    def solve_position_single_env[
        BATCH: Int,
        NUM_BODIES: Int,
        MAX_JOINTS: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
        JOINTS_OFFSET: Int,
    ](
        env: Int,
        state: LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        joint_count: Int,
    ) -> Bool:
        """One position iteration over all joints
        (b2RevoluteJoint::SolvePositionConstraints). Returns True when every
        joint is within `B2_LINEAR_SLOP` / `B2_ANGULAR_SLOP`."""
        var all_ok = True
        for j in range(MAX_JOINTS):
            if j >= joint_count:
                break
            var joff = JOINTS_OFFSET + j * JOINT_DATA_SIZE
            if Int(_ld(state, env, joff + JOINT_TYPE)) != JOINT_REVOLUTE:
                continue
            var body_a = Int(_ld(state, env, joff + JOINT_BODY_A))
            var body_b = Int(_ld(state, env, joff + JOINT_BODY_B))
            var off_a = BODIES_OFFSET + body_a * BODY_STATE_SIZE
            var off_b = BODIES_OFFSET + body_b * BODY_STATE_SIZE
            var ma = _ld(state, env, off_a + IDX_INV_MASS)
            var mb = _ld(state, env, off_b + IDX_INV_MASS)
            var ia = _ld(state, env, off_a + IDX_INV_INERTIA)
            var ib = _ld(state, env, off_b + IDX_INV_INERTIA)
            var cax = _ld(state, env, off_a + IDX_X)
            var cay = _ld(state, env, off_a + IDX_Y)
            var aa = _ld(state, env, off_a + IDX_ANGLE)
            var cbx = _ld(state, env, off_b + IDX_X)
            var cby = _ld(state, env, off_b + IDX_Y)
            var ab = _ld(state, env, off_b + IDX_ANGLE)

            var flags = Int(_ld(state, env, joff + JOINT_FLAGS))
            var limit_state = Int(_ld(state, env, joff + JOINT_LIMIT_STATE))
            var fixed_rotation = (ia + ib) == Scalar[dtype](0)
            var angular_error = Scalar[dtype](0)

            # Angular limit.
            if (
                (flags & JOINT_FLAG_LIMIT_ENABLED) != 0
                and limit_state != JOINT_LIMIT_INACTIVE
                and not fixed_rotation
            ):
                var motor_mass = Scalar[dtype](1) / (ia + ib)
                var angle = ab - aa - _ld(state, env, joff + JOINT_REF_ANGLE)
                var max_corr = Scalar[dtype](B2_MAX_ANGULAR_CORRECTION)
                var slop = Scalar[dtype](B2_ANGULAR_SLOP)
                var limit_impulse = Scalar[dtype](0)
                if limit_state == JOINT_LIMIT_EQUAL:
                    var c = _clamp(
                        angle - _ld(state, env, joff + JOINT_LOWER_LIMIT),
                        -max_corr,
                        max_corr,
                    )
                    limit_impulse = -motor_mass * c
                    angular_error = abs(c)
                elif limit_state == JOINT_LIMIT_AT_LOWER:
                    var c = angle - _ld(state, env, joff + JOINT_LOWER_LIMIT)
                    angular_error = -c
                    c = _clamp(c + slop, -max_corr, Scalar[dtype](0))
                    limit_impulse = -motor_mass * c
                elif limit_state == JOINT_LIMIT_AT_UPPER:
                    var c = angle - _ld(state, env, joff + JOINT_UPPER_LIMIT)
                    angular_error = c
                    c = _clamp(c - slop, Scalar[dtype](0), max_corr)
                    limit_impulse = -motor_mass * c
                aa -= ia * limit_impulse
                ab += ib * limit_impulse

            # Point-to-point.
            var lax = _ld(state, env, joff + JOINT_ANCHOR_AX)
            var lay = _ld(state, env, joff + JOINT_ANCHOR_AY)
            var lbx = _ld(state, env, joff + JOINT_ANCHOR_BX)
            var lby = _ld(state, env, joff + JOINT_ANCHOR_BY)
            var ca = cos(aa)
            var sa = sin(aa)
            var cb = cos(ab)
            var sb = sin(ab)
            var rax = ca * lax - sa * lay
            var ray = sa * lax + ca * lay
            var rbx = cb * lbx - sb * lby
            var rby = sb * lbx + cb * lby
            var cx = cbx + rbx - cax - rax
            var cy = cby + rby - cay - ray
            var position_error = sqrt(cx * cx + cy * cy)
            var k11 = ma + mb + ia * ray * ray + ib * rby * rby
            var k12 = -ia * rax * ray - ib * rbx * rby
            var k22 = ma + mb + ia * rax * rax + ib * rbx * rbx
            var s2 = _solve22(k11, k12, k22, cx, cy)  # b2Mat22::Solve(C)
            var ix = -s2[0]
            var iy = -s2[1]
            cax -= ma * ix
            cay -= ma * iy
            aa -= ia * (rax * iy - ray * ix)
            cbx += mb * ix
            cby += mb * iy
            ab += ib * (rbx * iy - rby * ix)

            state[env, off_a + IDX_X] = cax
            state[env, off_a + IDX_Y] = cay
            state[env, off_a + IDX_ANGLE] = aa
            state[env, off_b + IDX_X] = cbx
            state[env, off_b + IDX_Y] = cby
            state[env, off_b + IDX_ANGLE] = ab

            if not (
                position_error <= Scalar[dtype](B2_LINEAR_SLOP)
                and angular_error <= Scalar[dtype](B2_ANGULAR_SLOP)
            ):
                all_ok = False
        return all_ok


# =============================================================================
# Small dense solves (b2Mat33::Solve33 / Solve22, b2Mat22::Solve)
# =============================================================================


@always_inline
def _solve22(
    k11: Scalar[dtype],
    k12: Scalar[dtype],
    k22: Scalar[dtype],
    bx: Scalar[dtype],
    by: Scalar[dtype],
) -> Tuple[Scalar[dtype], Scalar[dtype]]:
    """Solve [[k11, k12], [k12, k22]] x = b (0 when singular, as Box2D)."""
    var det = k11 * k22 - k12 * k12
    if det != Scalar[dtype](0):
        det = Scalar[dtype](1) / det
    return (det * (k22 * bx - k12 * by), det * (k11 * by - k12 * bx))


@always_inline
def _solve33(
    k11: Scalar[dtype],
    k12: Scalar[dtype],
    k13: Scalar[dtype],
    k22: Scalar[dtype],
    k23: Scalar[dtype],
    k33: Scalar[dtype],
    bx: Scalar[dtype],
    by: Scalar[dtype],
    bz: Scalar[dtype],
) -> Tuple[Scalar[dtype], Scalar[dtype], Scalar[dtype]]:
    """Solve the symmetric 3x3 K x = b by Cramer's rule, as b2Mat33::Solve33
    (columns ex = (k11, k12, k13), ey = (k12, k22, k23), ez = (k13, k23, k33))."""
    # cross(ey, ez)
    var eyez_x = k22 * k33 - k23 * k23
    var eyez_y = k23 * k13 - k12 * k33
    var eyez_z = k12 * k23 - k22 * k13
    var det = k11 * eyez_x + k12 * eyez_y + k13 * eyez_z
    if det != Scalar[dtype](0):
        det = Scalar[dtype](1) / det
    var x = det * (bx * eyez_x + by * eyez_y + bz * eyez_z)
    # dot(ex, cross(b, ez))
    var bez_x = by * k33 - bz * k23
    var bez_y = bz * k13 - bx * k33
    var bez_z = bx * k23 - by * k13
    var y = det * (k11 * bez_x + k12 * bez_y + k13 * bez_z)
    # dot(ex, cross(ey, b))
    var eyb_x = k22 * bz - k23 * by
    var eyb_y = k23 * bx - k12 * bz
    var eyb_z = k12 * by - k22 * bx
    var z = det * (k11 * eyb_x + k12 * eyb_y + k13 * eyb_z)
    return (x, y, z)
