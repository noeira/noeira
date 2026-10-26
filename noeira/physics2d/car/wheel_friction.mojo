"""Slip-based tire friction model for top-down car simulation.

This module implements the core tire physics from Gymnasium's car_dynamics.py.
The model computes friction forces based on slip velocities between the tire
and ground, clamped to a friction envelope (Coulomb friction).

Physics Overview:
1. Compute wheel velocity from hull motion (includes rotation contribution)
2. Decompose velocity into forward and lateral components
3. Compute slip: forward slip = vf - omega * radius, lateral slip = vs
4. Apply friction coefficient to slip to get raw forces
5. Clamp forces to friction envelope (sqrt(f^2 + p^2) <= friction_limit)
6. Update wheel omega from friction reaction

Reference: gymnasium/envs/box2d/car_dynamics.py, lines 171-266
"""

from std.math import sqrt, cos, sin
from noeira.core.tiles import RowMajor, Tile

from .constants import (
    # Physics constants
    ENGINE_POWER,
    BRAKE_FORCE,
    WHEEL_MOMENT_OF_INERTIA,
    WHEEL_RADIUS,
    FRICTION_LIMIT,
    FRICTION_COEF,
    # State indices
    HULL_X,
    HULL_Y,
    HULL_ANGLE,
    HULL_VX,
    HULL_VY,
    HULL_OMEGA,
    WHEEL_OMEGA,
    WHEEL_JOINT_ANGLE,
    WHEEL_PHASE,
    WHEEL_STATE_SIZE,
    CTRL_STEERING,
    CTRL_GAS,
    CTRL_BRAKE,
    # Wheel indices
    WHEEL_FL,
    WHEEL_FR,
    WHEEL_RL,
    WHEEL_RR,
    NUM_WHEELS,
    # Wheel positions
    WHEEL_POS_FL_X,
    WHEEL_POS_FL_Y,
    WHEEL_POS_FR_X,
    WHEEL_POS_FR_Y,
    WHEEL_POS_RL_X,
    WHEEL_POS_RL_Y,
    WHEEL_POS_RR_X,
    WHEEL_POS_RR_Y,
)
from .layout import CarRacingLayout

from ..constants import dtype, TPB


struct WheelFriction:
    """Slip-based tire friction model for top-down car simulation.

    This struct contains static methods for computing wheel friction forces.
    The model is stateless - all state is passed via LayoutTensor.

    Key formulas (from car_dynamics.py):
        vf = forward_dir · wheel_velocity    # Forward speed at wheel
        vs = lateral_dir · wheel_velocity    # Lateral speed at wheel
        vr = omega * wheel_radius            # Rotational surface speed

        f_force = (-vf + vr) * FRICTION_COEF  # Forward friction (slip-based)
        p_force = -vs * FRICTION_COEF         # Lateral friction

        # Coulomb friction envelope
        if sqrt(f² + p²) > friction_limit:
            scale = friction_limit / sqrt(f² + p²)
            f_force *= scale
            p_force *= scale
    """

    # =========================================================================
    # Wheel Position Helpers
    # =========================================================================

    @staticmethod
    @always_inline
    def get_wheel_local_pos(wheel: Int) -> Tuple[Scalar[dtype], Scalar[dtype]]:
        """Get wheel position in hull-local coordinates.

        Args:
            wheel: Wheel index (0=FL, 1=FR, 2=RL, 3=RR).

        Returns:
            (local_x, local_y) position of wheel relative to hull center.
        """
        if wheel == WHEEL_FL:
            return (
                Scalar[dtype](WHEEL_POS_FL_X),
                Scalar[dtype](WHEEL_POS_FL_Y),
            )
        elif wheel == WHEEL_FR:
            return (
                Scalar[dtype](WHEEL_POS_FR_X),
                Scalar[dtype](WHEEL_POS_FR_Y),
            )
        elif wheel == WHEEL_RL:
            return (
                Scalar[dtype](WHEEL_POS_RL_X),
                Scalar[dtype](WHEEL_POS_RL_Y),
            )
        else:  # WHEEL_RR
            return (
                Scalar[dtype](WHEEL_POS_RR_X),
                Scalar[dtype](WHEEL_POS_RR_Y),
            )

    @staticmethod
    @always_inline
    def is_front_wheel(wheel: Int) -> Bool:
        """Check if wheel is a front wheel (has steering)."""
        return wheel == WHEEL_FL or wheel == WHEEL_FR

    @staticmethod
    @always_inline
    def is_rear_wheel(wheel: Int) -> Bool:
        """Check if wheel is a rear wheel (has engine power)."""
        return wheel == WHEEL_RL or wheel == WHEEL_RR

    # =========================================================================
    # Core Friction Computation
    # =========================================================================

    @staticmethod
    @always_inline
    def compute_wheel_forces(
        # Wheel angular velocity
        wheel_omega: Scalar[dtype],
        # Steering angle (0 for rear wheels)
        joint_angle: Scalar[dtype],
        # Hull state
        hull_angle: Scalar[dtype],
        hull_vx: Scalar[dtype],
        hull_vy: Scalar[dtype],
        hull_omega: Scalar[dtype],
        # Wheel offset (local coords)
        wheel_local_x: Scalar[dtype],
        wheel_local_y: Scalar[dtype],
        # Controls
        gas: Scalar[dtype],
        brake: Scalar[dtype],
        is_rear: Bool,
        # Surface friction
        friction_limit: Scalar[dtype],
        dt: Scalar[dtype],
    ) -> Tuple[
        Scalar[dtype],  # force_x (world)
        Scalar[dtype],  # force_y (world)
        Scalar[dtype],  # torque_on_hull
        Scalar[dtype],  # new_omega
    ]:
        """Compute friction forces for a single wheel.

        This implements the slip-based tire friction model from car_dynamics.py.

        Args:
            wheel_omega: Current wheel angular velocity.
            joint_angle: Steering joint angle (radians, 0 for rear wheels).
            hull_angle: Hull orientation angle (radians).
            hull_vx: Hull linear velocity x-component.
            hull_vy: Hull linear velocity y-component.
            hull_omega: Hull angular velocity.
            wheel_local_x: Wheel position x-component in hull-local coords.
            wheel_local_y: Wheel position y-component in hull-local coords.
            gas: Gas pedal input [0, 1].
            brake: Brake pedal input [0, 1].
            is_rear: True for rear wheels (have engine power).
            friction_limit: Surface friction limit (road vs grass).
            dt: Time step.

        Returns:
            (force_x, force_y, torque, new_omega) where forces are in world frame.
        """
        var omega = wheel_omega
        var wheel_rad = Scalar[dtype](WHEEL_RADIUS)
        var moment = Scalar[dtype](WHEEL_MOMENT_OF_INERTIA)
        var engine_power = Scalar[dtype](ENGINE_POWER)
        var brake_force = Scalar[dtype](BRAKE_FORCE)
        var friction_coef = Scalar[dtype](FRICTION_COEF)
        var zero = Scalar[dtype](0.0)
        var one = Scalar[dtype](1.0)

        # =====================================================================
        # Step 1: Update wheel omega from engine/brake
        # =====================================================================

        # Engine power (rear wheels only)
        if is_rear and gas > zero:
            # omega += dt * ENGINE_POWER * gas / (MOMENT * (|omega| + 5))
            var omega_abs = omega if omega >= zero else -omega
            var denom = moment * (omega_abs + Scalar[dtype](5.0))
            omega = omega + (dt * engine_power * gas) / denom

        # Braking
        if brake >= Scalar[dtype](0.9):
            # Full brake = instant stop
            omega = zero
        elif brake > zero:
            # Gradual braking
            var dir = -one if omega > zero else one
            var val = brake_force * brake
            var omega_abs = omega if omega >= zero else -omega
            if val > omega_abs:
                val = omega_abs
            omega = omega + dir * val

        # =====================================================================
        # Step 2: Compute wheel velocity at contact point
        # =====================================================================

        # Transform local wheel position to world
        var cos_hull = cos(hull_angle)
        var sin_hull = sin(hull_angle)

        var wheel_world_x = wheel_local_x * cos_hull - wheel_local_y * sin_hull
        var wheel_world_y = wheel_local_x * sin_hull + wheel_local_y * cos_hull

        # Wheel velocity = hull velocity + hull_omega × r_wheel
        # For 2D: v_wheel = v_hull + omega_hull * (-r_y, r_x)
        var vx_wheel = hull_vx - hull_omega * wheel_world_y
        var vy_wheel = hull_vy + hull_omega * wheel_world_x

        # =====================================================================
        # Step 3: Decompose velocity into forward/lateral
        # =====================================================================

        # Wheel forward direction (hull forward + steering)
        var wheel_angle = hull_angle + joint_angle
        var cos_wheel = cos(wheel_angle)
        var sin_wheel = sin(wheel_angle)

        # Forward direction (local +Y rotated to world)
        var forw_x = -sin_wheel
        var forw_y = cos_wheel

        # Lateral direction (local +X rotated to world)
        var side_x = cos_wheel
        var side_y = sin_wheel

        # Velocity components
        var vf = forw_x * vx_wheel + forw_y * vy_wheel  # Forward velocity
        var vs = side_x * vx_wheel + side_y * vy_wheel  # Lateral velocity

        # =====================================================================
        # Step 4: Compute slip-based friction forces
        # =====================================================================

        # Rotational surface speed
        var vr = omega * wheel_rad

        # Forward friction: tries to eliminate slip (vf - vr)
        var f_force = (-vf + vr) * friction_coef

        # Lateral friction: tries to eliminate lateral slip
        var p_force = -vs * friction_coef

        # =====================================================================
        # Step 5: Clamp to friction envelope
        # =====================================================================

        # Compute force magnitude with protection against floating-point issues
        var force_sq = f_force * f_force + p_force * p_force
        # Clamp to prevent sqrt of negative (shouldn't happen but floating-point)
        if force_sq < zero:
            force_sq = zero
        var force_mag = sqrt(force_sq)

        # Clamp forces to friction envelope
        # Guard against division by zero (only divide when force_mag is significant)
        if force_mag > friction_limit and force_mag > Scalar[dtype](1e-8):
            var scale = friction_limit / force_mag
            f_force = f_force * scale
            p_force = p_force * scale

        # NaN protection: if forces are NaN, reset to zero
        if f_force != f_force:  # NaN check (NaN != NaN is true)
            f_force = zero
        if p_force != p_force:
            p_force = zero

        # =====================================================================
        # Step 6: Update wheel omega from friction reaction
        # =====================================================================

        # Friction on wheel surface creates torque that opposes rotation
        # tau = f_force * wheel_radius (in direction opposing omega)
        # alpha = tau / I
        # Guard: moment should never be zero but add protection
        if moment > Scalar[dtype](1e-8):
            omega = omega - dt * f_force * wheel_rad / moment

        # Clamp wheel omega to reasonable bounds (prevents instability)
        var max_wheel_omega = Scalar[dtype](200.0)  # ~30 rotations/second
        if omega > max_wheel_omega:
            omega = max_wheel_omega
        elif omega < -max_wheel_omega:
            omega = -max_wheel_omega

        # NaN protection for omega
        if omega != omega:  # NaN check
            omega = zero

        # =====================================================================
        # Step 7: Transform forces to world frame
        # =====================================================================

        var fx = p_force * side_x + f_force * forw_x
        var fy = p_force * side_y + f_force * forw_y

        # Torque on hull from this wheel
        # tau = r × F = r_x * F_y - r_y * F_x
        var torque = wheel_world_x * fy - wheel_world_y * fx

        return (fx, fy, torque, omega)

    # =========================================================================
    # Aggregate Forces from All Wheels
    # =========================================================================

    @staticmethod
    @always_inline
    def compute_all_wheels_forces[
        BATCH: Int,
        STATE_SIZE: Int,
        HULL_OFFSET: Int,
        WHEELS_OFFSET: Int,
        CONTROLS_OFFSET: Int,
    ](
        env: Int,
        state: Tile[dtype, BATCH, STATE_SIZE],
        friction_limits: Array[Scalar[dtype], NUM_WHEELS],
        dt: Scalar[dtype],
    ) -> Tuple[Scalar[dtype], Scalar[dtype], Scalar[dtype]]:
        """Compute total force and torque from all 4 wheels.

        Args:
            env: Environment index.
            state: State tensor [BATCH, STATE_SIZE].
            friction_limits: Friction limit for each wheel (from tile lookup).
            dt: Time step.

        Returns:
            (total_fx, total_fy, total_torque) to apply to hull.
        """
        var zero = Scalar[dtype](0.0)
        var total_fx = zero
        var total_fy = zero
        var total_torque = zero

        # Read hull state (use rebind to extract Scalar from LayoutTensor)
        var hull_x = rebind[Scalar[dtype]](state[env, HULL_OFFSET + HULL_X])
        var hull_y = rebind[Scalar[dtype]](state[env, HULL_OFFSET + HULL_Y])
        var hull_angle = rebind[Scalar[dtype]](
            state[env, HULL_OFFSET + HULL_ANGLE]
        )
        var hull_vx = rebind[Scalar[dtype]](state[env, HULL_OFFSET + HULL_VX])
        var hull_vy = rebind[Scalar[dtype]](state[env, HULL_OFFSET + HULL_VY])
        var hull_omega = rebind[Scalar[dtype]](
            state[env, HULL_OFFSET + HULL_OMEGA]
        )

        # Read controls
        var steering = rebind[Scalar[dtype]](
            state[env, CONTROLS_OFFSET + CTRL_STEERING]
        )
        var gas = rebind[Scalar[dtype]](state[env, CONTROLS_OFFSET + CTRL_GAS])
        var brake = rebind[Scalar[dtype]](
            state[env, CONTROLS_OFFSET + CTRL_BRAKE]
        )

        # Process each wheel
        comptime for wheel in range(NUM_WHEELS):
            var wheel_off = WHEELS_OFFSET + wheel * WHEEL_STATE_SIZE
            var wheel_omega = rebind[Scalar[dtype]](
                state[env, wheel_off + WHEEL_OMEGA]
            )
            var joint_angle = rebind[Scalar[dtype]](
                state[env, wheel_off + WHEEL_JOINT_ANGLE]
            )

            # Get wheel local position
            var local_pos = WheelFriction.get_wheel_local_pos(wheel)
            var local_x = local_pos[0]
            var local_y = local_pos[1]

            # Determine wheel properties
            var is_front = WheelFriction.is_front_wheel(wheel)
            var is_rear = WheelFriction.is_rear_wheel(wheel)

            # Front wheels use steering angle, rear wheels fixed
            var steer_angle = steering if is_front else zero

            # Compute forces
            var result = WheelFriction.compute_wheel_forces(
                wheel_omega,
                steer_angle,
                hull_angle,
                hull_vx,
                hull_vy,
                hull_omega,
                local_x,
                local_y,
                gas,
                brake,
                is_rear,
                friction_limits[wheel],
                dt,
            )

            # Accumulate forces
            total_fx = total_fx + result[0]
            total_fy = total_fy + result[1]
            total_torque = total_torque + result[2]

            # Update wheel omega
            state[env, wheel_off + WHEEL_OMEGA] = result[3]

            # Update wheel phase (cumulative rotation for rendering)
            var phase = rebind[Scalar[dtype]](
                state[env, wheel_off + WHEEL_PHASE]
            )
            phase = phase + result[3] * dt
            state[env, wheel_off + WHEEL_PHASE] = phase

        return (total_fx, total_fy, total_torque)

    # =========================================================================
    # CPU Batch Processing
    # =========================================================================

    @staticmethod
    def compute_batch_cpu[
        BATCH: Int,
        STATE_SIZE: Int,
        HULL_OFFSET: Int,
        WHEELS_OFFSET: Int,
        CONTROLS_OFFSET: Int,
    ](
        state: Tile[dtype, BATCH, STATE_SIZE],
        friction_limits: Tile[dtype, BATCH, NUM_WHEELS],
        forces: Tile[dtype, BATCH, 3],
        dt: Scalar[dtype],
    ):
        """Compute wheel forces for all environments (CPU).

        Args:
            state: State tensor [BATCH, STATE_SIZE].
            friction_limits: Friction limits per wheel [BATCH, NUM_WHEELS].
            forces: Output force tensor [BATCH, 3] for (fx, fy, torque).
            dt: Time step.
        """
        for env in range(BATCH):
            # Build friction limits array for this env
            var limits = Array[Scalar[dtype], NUM_WHEELS](
                fill=Scalar[dtype](0)
            )
            limits[0] = rebind[Scalar[dtype]](friction_limits[env, 0])
            limits[1] = rebind[Scalar[dtype]](friction_limits[env, 1])
            limits[2] = rebind[Scalar[dtype]](friction_limits[env, 2])
            limits[3] = rebind[Scalar[dtype]](friction_limits[env, 3])

            var result = WheelFriction.compute_all_wheels_forces[
                BATCH, STATE_SIZE, HULL_OFFSET, WHEELS_OFFSET, CONTROLS_OFFSET
            ](env, state, limits, dt)

            forces[env, 0] = result[0]
            forces[env, 1] = result[1]
            forces[env, 2] = result[2]
