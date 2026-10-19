"""Full car dynamics simulation step.

This module combines all car physics components into a complete simulation step:
1. Update steering joints (front wheels track steering input)
2. Get friction limits from tile collision
3. Compute wheel forces via slip-based friction model
4. Integrate hull dynamics (Euler integration)
5. Update observation vector

The simulation is top-down (no gravity) with slip-based tire friction.
This matches the physics model in Gymnasium's car_dynamics.py.

Reference: gymnasium/envs/box2d/car_dynamics.py
"""

from std.math import cos, sin, sqrt, pi

from .constants import (
    # Physics constants
    HULL_MASS,
    HULL_INV_MASS,
    HULL_INERTIA,
    HULL_INV_INERTIA,
    STEERING_LIMIT,
    STEERING_MOTOR_SPEED,
    STEERING_GAIN,
    FRICTION_LIMIT,
    ROAD_FRICTION,
    CAR_DT,
    CAR_PI,
    CAR_TWO_PI,
    # State indices
    HULL_X,
    HULL_Y,
    HULL_ANGLE,
    HULL_VX,
    HULL_VY,
    HULL_OMEGA,
    HULL_STATE_SIZE,
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
    NUM_WHEELS,
    MAX_TRACK_TILES,
    TILE_DATA_SIZE,
)
from .layout import CarRacingLayout
from .wheel_friction import WheelFriction
from .tile_collision import TileCollision

from ..constants import dtype
from noeira.core.tiles import Tile


struct CarDynamics:
    """Top-down car dynamics: hull + 4 wheels with tire friction.

    This struct provides static methods for simulating car physics.
    The physics is stateless - all state is passed via LayoutTensor.

    Simulation steps per time step:
    1. Update steering (front wheel joint angles)
    2. Compute wheel friction limits from track tiles
    3. Compute wheel forces (updates wheel omegas internally)
    4. Apply forces to hull (integrate dynamics)
    5. Update observation vector for RL
    """

    # =========================================================================
    # Steering Update
    # =========================================================================

    @staticmethod
    @always_inline
    def update_steering[
        BATCH: Int,
        STATE_SIZE: Int,
        WHEELS_OFFSET: Int,
        CONTROLS_OFFSET: Int,
    ](
        env: Int,
        state: Tile[dtype, BATCH, STATE_SIZE],
        dt: Scalar[dtype],
    ):
        """Update front wheel steering angles.

        Front wheels smoothly rotate to match the steering input.
        Steering is limited by STEERING_LIMIT and the rotation rate
        is limited by STEERING_MOTOR_SPEED.

        From car_dynamics.py:
            dir = sign(steer - joint.angle)
            val = abs(steer - joint.angle)
            joint.motorSpeed = dir * min(50 * val, 3.0)

        Args:
            env: Environment index.
            state: State tensor.
            dt: Time step.
        """
        var steering_input = rebind[Scalar[dtype]](
            state[env, CONTROLS_OFFSET + CTRL_STEERING]
        )

        # Clamp steering input to limits
        var steer_limit = Scalar[dtype](STEERING_LIMIT)
        var target_angle = (
            steering_input * steer_limit
        )  # Map [-1,1] to [-limit, limit]

        # Update front-left wheel
        var fl_off = WHEELS_OFFSET + WHEEL_FL * WHEEL_STATE_SIZE
        var fl_angle = rebind[Scalar[dtype]](
            state[env, fl_off + WHEEL_JOINT_ANGLE]
        )
        fl_angle = CarDynamics._update_steering_angle(
            fl_angle, target_angle, dt
        )
        state[env, fl_off + WHEEL_JOINT_ANGLE] = fl_angle

        # Update front-right wheel (same steering angle)
        var fr_off = WHEELS_OFFSET + WHEEL_FR * WHEEL_STATE_SIZE
        var fr_angle = rebind[Scalar[dtype]](
            state[env, fr_off + WHEEL_JOINT_ANGLE]
        )
        fr_angle = CarDynamics._update_steering_angle(
            fr_angle, target_angle, dt
        )
        state[env, fr_off + WHEEL_JOINT_ANGLE] = fr_angle

    @staticmethod
    @always_inline
    def _update_steering_angle(
        current: Scalar[dtype],
        target: Scalar[dtype],
        dt: Scalar[dtype],
    ) -> Scalar[dtype]:
        """Update a single steering angle towards target.

        Args:
            current: Current joint angle.
            target: Target joint angle.
            dt: Time step.

        Returns:
            New joint angle.
        """
        var diff = target - current
        var gain = Scalar[dtype](STEERING_GAIN)
        var max_speed = Scalar[dtype](STEERING_MOTOR_SPEED)

        # Compute motor speed (proportional to error, capped)
        var speed = diff * gain
        if speed > max_speed:
            speed = max_speed
        elif speed < -max_speed:
            speed = -max_speed

        # Integrate angle
        var new_angle = current + speed * dt

        # Clamp to steering limits
        var limit = Scalar[dtype](STEERING_LIMIT)
        if new_angle > limit:
            new_angle = limit
        elif new_angle < -limit:
            new_angle = -limit

        return new_angle

    # =========================================================================
    # Hull Dynamics Integration
    # =========================================================================

    @staticmethod
    @always_inline
    def integrate_hull[
        BATCH: Int,
        STATE_SIZE: Int,
        HULL_OFFSET: Int,
    ](
        env: Int,
        state: Tile[dtype, BATCH, STATE_SIZE],
        fx: Scalar[dtype],
        fy: Scalar[dtype],
        torque: Scalar[dtype],
        dt: Scalar[dtype],
    ):
        """Integrate hull dynamics using semi-implicit Euler.

        Updates hull velocity from forces, then updates position from velocity.

        Args:
            env: Environment index.
            state: State tensor.
            fx: Total force on hull x-coordinate (world frame).
            fy: Total force on hull y-coordinate (world frame).
            torque: Total torque on hull.
            dt: Time step.
        """
        # Read current state
        var x = rebind[Scalar[dtype]](state[env, HULL_OFFSET + HULL_X])
        var y = rebind[Scalar[dtype]](state[env, HULL_OFFSET + HULL_Y])
        var angle = rebind[Scalar[dtype]](state[env, HULL_OFFSET + HULL_ANGLE])
        var vx = rebind[Scalar[dtype]](state[env, HULL_OFFSET + HULL_VX])
        var vy = rebind[Scalar[dtype]](state[env, HULL_OFFSET + HULL_VY])
        var omega = rebind[Scalar[dtype]](state[env, HULL_OFFSET + HULL_OMEGA])

        var inv_mass = Scalar[dtype](HULL_INV_MASS)
        var inv_inertia = Scalar[dtype](HULL_INV_INERTIA)

        # Semi-implicit Euler: update velocity first
        vx = vx + fx * inv_mass * dt
        vy = vy + fy * inv_mass * dt
        omega = omega + torque * inv_inertia * dt

        # Clamp velocities to prevent overflow (max ~500 m/s is reasonable for a car)
        var max_vel = Scalar[dtype](500.0)
        var max_omega = Scalar[dtype](50.0)  # ~8 full rotations per second
        if vx > max_vel:
            vx = max_vel
        elif vx < -max_vel:
            vx = -max_vel
        if vy > max_vel:
            vy = max_vel
        elif vy < -max_vel:
            vy = -max_vel
        if omega > max_omega:
            omega = max_omega
        elif omega < -max_omega:
            omega = -max_omega

        # Check for NaN and reset to safe values if detected
        # NaN != NaN is true, so we use this property
        if vx != vx:  # NaN check
            vx = Scalar[dtype](0.0)
        if vy != vy:
            vy = Scalar[dtype](0.0)
        if omega != omega:
            omega = Scalar[dtype](0.0)

        # Then update position using new velocity
        x = x + vx * dt
        y = y + vy * dt
        angle = angle + omega * dt

        # Check position for NaN
        if x != x:
            x = Scalar[dtype](0.0)
        if y != y:
            y = Scalar[dtype](0.0)
        if angle != angle:
            angle = Scalar[dtype](0.0)

        # Normalize angle to [-pi, pi]
        var pi_val = Scalar[dtype](CAR_PI)
        var two_pi = Scalar[dtype](CAR_TWO_PI)
        if angle > Scalar[dtype](100.0) or angle < Scalar[dtype](-100.0):
            angle = Scalar[dtype](0.0)
        elif angle > pi_val:
            angle = angle - two_pi
            if angle > pi_val:
                angle = angle - two_pi
        elif angle < -pi_val:
            angle = angle + two_pi
            if angle < -pi_val:
                angle = angle + two_pi

        # Write back
        state[env, HULL_OFFSET + HULL_X] = x
        state[env, HULL_OFFSET + HULL_Y] = y
        state[env, HULL_OFFSET + HULL_ANGLE] = angle
        state[env, HULL_OFFSET + HULL_VX] = vx
        state[env, HULL_OFFSET + HULL_VY] = vy
        state[env, HULL_OFFSET + HULL_OMEGA] = omega

    # =========================================================================
    # Single Environment Step
    # =========================================================================

    @staticmethod
    @always_inline
    def step_single_env[
        BATCH: Int,
        STATE_SIZE: Int,
        OBS_OFFSET: Int,
        HULL_OFFSET: Int,
        WHEELS_OFFSET: Int,
        CONTROLS_OFFSET: Int,
        MAX_TILES: Int,
    ](
        env: Int,
        state: Tile[dtype, BATCH, STATE_SIZE],
        tiles: Tile[dtype, MAX_TILES, TILE_DATA_SIZE],
        num_active_tiles: Int,
        dt: Scalar[dtype],
    ):
        """Perform one physics step for a single environment.

        This is the core simulation step that can be called from:
        - CPU batch loop
        - GPU kernel (one thread per environment)

        Args:
            env: Environment index.
            state: State tensor [BATCH, STATE_SIZE].
            tiles: Track tile data [MAX_TILES, TILE_DATA_SIZE].
            num_active_tiles: Number of valid tiles.
            dt: Time step.
        """
        # Step 1: Update steering joints
        CarDynamics.update_steering[
            BATCH, STATE_SIZE, WHEELS_OFFSET, CONTROLS_OFFSET
        ](env, state, dt)

        # Step 2: Get friction limits from track tiles
        var hull_x = rebind[Scalar[dtype]](state[env, HULL_OFFSET + HULL_X])
        var hull_y = rebind[Scalar[dtype]](state[env, HULL_OFFSET + HULL_Y])
        var hull_angle = rebind[Scalar[dtype]](
            state[env, HULL_OFFSET + HULL_ANGLE]
        )

        var friction_limits = TileCollision.get_wheel_friction_limits[
            MAX_TILES
        ](hull_x, hull_y, hull_angle, tiles, num_active_tiles)

        # Step 3: Compute wheel forces (also updates wheel omegas)
        var forces = WheelFriction.compute_all_wheels_forces[
            BATCH, STATE_SIZE, HULL_OFFSET, WHEELS_OFFSET, CONTROLS_OFFSET
        ](env, state, friction_limits, dt)

        var fx = forces[0]
        var fy = forces[1]
        var torque = forces[2]

        # Step 4: Integrate hull dynamics
        CarDynamics.integrate_hull[BATCH, STATE_SIZE, HULL_OFFSET](
            env, state, fx, fy, torque, dt
        )

    # =========================================================================
    # Observation Update
    # =========================================================================

    @staticmethod
    @always_inline
    def update_observation[
        BATCH: Int,
        STATE_SIZE: Int,
        OBS_OFFSET: Int,
        OBS_DIM: Int,
        HULL_OFFSET: Int,
        WHEELS_OFFSET: Int,
    ](
        env: Int,
        state: Tile[dtype, BATCH, STATE_SIZE],
    ):
        """Update observation vector from physics state.

        Observation format (13 values):
        - [0-5]: Hull state (x, y, angle, vx, vy, omega)
        - [6-7]: Front wheel angles (FL, FR)
        - [8-11]: Wheel angular velocities (FL, FR, RL, RR)
        - [12]: (reserved for speed indicator or other)

        Args:
            env: Environment index.
            state: State tensor.
        """
        # Hull state (6 values)
        state[env, OBS_OFFSET + 0] = state[env, HULL_OFFSET + HULL_X]
        state[env, OBS_OFFSET + 1] = state[env, HULL_OFFSET + HULL_Y]
        state[env, OBS_OFFSET + 2] = state[env, HULL_OFFSET + HULL_ANGLE]
        state[env, OBS_OFFSET + 3] = state[env, HULL_OFFSET + HULL_VX]
        state[env, OBS_OFFSET + 4] = state[env, HULL_OFFSET + HULL_VY]
        state[env, OBS_OFFSET + 5] = state[env, HULL_OFFSET + HULL_OMEGA]

        # Front wheel angles (2 values)
        var fl_off = WHEELS_OFFSET + WHEEL_FL * WHEEL_STATE_SIZE
        var fr_off = WHEELS_OFFSET + WHEEL_FR * WHEEL_STATE_SIZE
        state[env, OBS_OFFSET + 6] = state[env, fl_off + WHEEL_JOINT_ANGLE]
        state[env, OBS_OFFSET + 7] = state[env, fr_off + WHEEL_JOINT_ANGLE]

        # Wheel angular velocities (4 values)
        comptime for wheel in range(NUM_WHEELS):
            var wheel_off = WHEELS_OFFSET + wheel * WHEEL_STATE_SIZE
            state[env, OBS_OFFSET + 8 + wheel] = state[
                env, wheel_off + WHEEL_OMEGA
            ]

        # Speed indicator (computed from vx, vy) with NaN protection
        var vx = rebind[Scalar[dtype]](state[env, HULL_OFFSET + HULL_VX])
        var vy = rebind[Scalar[dtype]](state[env, HULL_OFFSET + HULL_VY])
        var speed_sq = vx * vx + vy * vy
        # Clamp to prevent sqrt of negative due to floating-point errors
        if speed_sq < Scalar[dtype](0.0):
            speed_sq = Scalar[dtype](0.0)
        var speed = sqrt(speed_sq)
        # NaN protection
        if speed != speed:  # NaN check
            speed = Scalar[dtype](0.0)
        state[env, OBS_OFFSET + 12] = speed

    # =========================================================================
    # Full Step with Observation Update
    # =========================================================================

    @staticmethod
    @always_inline
    def step_with_obs[
        BATCH: Int,
        STATE_SIZE: Int,
        OBS_OFFSET: Int,
        OBS_DIM: Int,
        HULL_OFFSET: Int,
        WHEELS_OFFSET: Int,
        CONTROLS_OFFSET: Int,
        MAX_TILES: Int,
    ](
        env: Int,
        state: Tile[dtype, BATCH, STATE_SIZE],
        tiles: Tile[dtype, MAX_TILES, TILE_DATA_SIZE],
        num_active_tiles: Int,
        dt: Scalar[dtype],
    ):
        """Perform physics step and update observations.

        This combines step_single_env and update_observation into a single
        method for convenience.

        Args:
            env: Environment index.
            state: State tensor.
            tiles: Track tile data.
            num_active_tiles: Number of valid tiles.
            dt: Time step.
        """
        # Physics step
        CarDynamics.step_single_env[
            BATCH,
            STATE_SIZE,
            OBS_OFFSET,
            HULL_OFFSET,
            WHEELS_OFFSET,
            CONTROLS_OFFSET,
            MAX_TILES,
        ](env, state, tiles, num_active_tiles, dt)

        # Update observations
        CarDynamics.update_observation[
            BATCH, STATE_SIZE, OBS_OFFSET, OBS_DIM, HULL_OFFSET, WHEELS_OFFSET
        ](env, state)

    @staticmethod
    @always_inline
    def step_trackless[
        BATCH: Int,
        STATE_SIZE: Int,
        OBS_OFFSET: Int,
        OBS_DIM: Int,
        HULL_OFFSET: Int,
        WHEELS_OFFSET: Int,
        CONTROLS_OFFSET: Int,
    ](
        env: Int,
        state: Tile[dtype, BATCH, STATE_SIZE],
        dt: Scalar[dtype],
    ):
        """Perform physics step without track (uses default road friction).

        This version assumes all wheels are on road (friction = 1.0).
        Useful for simplified training without track tile collision.

        Args:
            env: Environment index.
            state: State tensor.
            dt: Time step.
        """
        # Step 1: Update steering joints
        CarDynamics.update_steering[
            BATCH, STATE_SIZE, WHEELS_OFFSET, CONTROLS_OFFSET
        ](env, state, dt)

        # Step 2: Use default road friction for all wheels (no tile lookup)
        # friction_limits = [fl, fr, rl, rr] all set to FRICTION_LIMIT * ROAD_FRICTION
        var default_friction = Scalar[dtype](FRICTION_LIMIT * ROAD_FRICTION)
        var friction_limits = Array[Scalar[dtype], NUM_WHEELS](
            fill=[
                default_friction,
                default_friction,
                default_friction,
                default_friction,
            ]
        )

        # Step 3: Compute wheel forces (also updates wheel omegas)
        var forces = WheelFriction.compute_all_wheels_forces[
            BATCH, STATE_SIZE, HULL_OFFSET, WHEELS_OFFSET, CONTROLS_OFFSET
        ](env, state, friction_limits, dt)

        var fx = forces[0]
        var fy = forces[1]
        var torque = forces[2]

        # Step 4: Integrate hull dynamics
        CarDynamics.integrate_hull[BATCH, STATE_SIZE, HULL_OFFSET](
            env, state, fx, fy, torque, dt
        )

        # Step 5: Update observations
        CarDynamics.update_observation[
            BATCH, STATE_SIZE, OBS_OFFSET, OBS_DIM, HULL_OFFSET, WHEELS_OFFSET
        ](env, state)

    @staticmethod
    @always_inline
    def step_with_embedded_track[
        BATCH: Int,
        STATE_SIZE: Int,
        OBS_OFFSET: Int,
        OBS_DIM: Int,
        HULL_OFFSET: Int,
        WHEELS_OFFSET: Int,
        CONTROLS_OFFSET: Int,
        TRACK_OFFSET: Int,
        MAX_TILES: Int,
    ](
        env: Int,
        state: Tile[dtype, BATCH, STATE_SIZE],
        num_active_tiles: Int,
        dt: Scalar[dtype],
    ):
        """Perform physics step with track tiles embedded in state buffer.

        This version reads track data from the per-env state buffer instead
        of a separate tiles buffer. Each environment has its own track tiles
        stored at TRACK_OFFSET within its state.

        Args:
            env: Environment index.
            state: State tensor with embedded track at TRACK_OFFSET.
            num_active_tiles: Number of valid tiles for this env.
            dt: Time step.
        """
        # Step 1: Update steering joints
        CarDynamics.update_steering[
            BATCH, STATE_SIZE, WHEELS_OFFSET, CONTROLS_OFFSET
        ](env, state, dt)

        # Step 2: Get friction limits from embedded track tiles
        var hull_x = rebind[Scalar[dtype]](state[env, HULL_OFFSET + HULL_X])
        var hull_y = rebind[Scalar[dtype]](state[env, HULL_OFFSET + HULL_Y])
        var hull_angle = rebind[Scalar[dtype]](
            state[env, HULL_OFFSET + HULL_ANGLE]
        )

        # Get wheel friction limits using embedded track data
        var friction_limits = CarDynamics._get_wheel_friction_limits_embedded[
            BATCH, STATE_SIZE, HULL_OFFSET, TRACK_OFFSET, MAX_TILES
        ](env, state, hull_x, hull_y, hull_angle, num_active_tiles)

        # Step 3: Compute wheel forces (also updates wheel omegas)
        var forces = WheelFriction.compute_all_wheels_forces[
            BATCH, STATE_SIZE, HULL_OFFSET, WHEELS_OFFSET, CONTROLS_OFFSET
        ](env, state, friction_limits, dt)

        var fx = forces[0]
        var fy = forces[1]
        var torque = forces[2]

        # Step 4: Integrate hull dynamics
        CarDynamics.integrate_hull[BATCH, STATE_SIZE, HULL_OFFSET](
            env, state, fx, fy, torque, dt
        )

        # Step 5: Update observations
        CarDynamics.update_observation[
            BATCH, STATE_SIZE, OBS_OFFSET, OBS_DIM, HULL_OFFSET, WHEELS_OFFSET
        ](env, state)

    @staticmethod
    @always_inline
    def _get_wheel_friction_limits_embedded[
        BATCH: Int,
        STATE_SIZE: Int,
        HULL_OFFSET: Int,
        TRACK_OFFSET: Int,
        MAX_TILES: Int,
    ](
        env: Int,
        state: Tile[dtype, BATCH, STATE_SIZE],
        hull_x: Scalar[dtype],
        hull_y: Scalar[dtype],
        hull_angle: Scalar[dtype],
        num_active_tiles: Int,
    ) -> Array[Scalar[dtype], NUM_WHEELS]:
        """Get friction limits for all 4 wheels using embedded track data.

        Args:
            env: Environment index.
            state: State tensor with embedded track.
            hull_x: Hull center position x-coordinate.
            hull_y: Hull center position y-coordinate.
            hull_angle: Hull orientation.
            num_active_tiles: Number of valid tiles.

        Returns:
            Array of friction limits [FL, FR, RL, RR].
        """
        var cos_a = cos(hull_angle)
        var sin_a = sin(hull_angle)

        # Wheel local positions
        var local_fl = WheelFriction.get_wheel_local_pos(0)  # FL
        var local_fr = WheelFriction.get_wheel_local_pos(1)  # FR
        var local_rl = WheelFriction.get_wheel_local_pos(2)  # RL
        var local_rr = WheelFriction.get_wheel_local_pos(3)  # RR

        # Transform to world coordinates
        var fl_x = hull_x + local_fl[0] * cos_a - local_fl[1] * sin_a
        var fl_y = hull_y + local_fl[0] * sin_a + local_fl[1] * cos_a

        var fr_x = hull_x + local_fr[0] * cos_a - local_fr[1] * sin_a
        var fr_y = hull_y + local_fr[0] * sin_a + local_fr[1] * cos_a

        var rl_x = hull_x + local_rl[0] * cos_a - local_rl[1] * sin_a
        var rl_y = hull_y + local_rl[0] * sin_a + local_rl[1] * cos_a

        var rr_x = hull_x + local_rr[0] * cos_a - local_rr[1] * sin_a
        var rr_y = hull_y + local_rr[0] * sin_a + local_rr[1] * cos_a

        # Get friction limits using embedded track
        var fl_limit = TileCollision.get_friction_limit_at_embedded[
            BATCH, STATE_SIZE, TRACK_OFFSET, MAX_TILES
        ](env, fl_x, fl_y, state, num_active_tiles)
        var fr_limit = TileCollision.get_friction_limit_at_embedded[
            BATCH, STATE_SIZE, TRACK_OFFSET, MAX_TILES
        ](env, fr_x, fr_y, state, num_active_tiles)
        var rl_limit = TileCollision.get_friction_limit_at_embedded[
            BATCH, STATE_SIZE, TRACK_OFFSET, MAX_TILES
        ](env, rl_x, rl_y, state, num_active_tiles)
        var rr_limit = TileCollision.get_friction_limit_at_embedded[
            BATCH, STATE_SIZE, TRACK_OFFSET, MAX_TILES
        ](env, rr_x, rr_y, state, num_active_tiles)

        return Array[Scalar[dtype], NUM_WHEELS](
            fill=[fl_limit, fr_limit, rl_limit, rr_limit]
        )

    # =========================================================================
    # CPU Batch Processing
    # =========================================================================

    @staticmethod
    def step_batch_cpu[
        BATCH: Int,
        STATE_SIZE: Int,
        OBS_OFFSET: Int,
        OBS_DIM: Int,
        HULL_OFFSET: Int,
        WHEELS_OFFSET: Int,
        CONTROLS_OFFSET: Int,
        MAX_TILES: Int,
    ](
        state: Tile[dtype, BATCH, STATE_SIZE],
        tiles: Tile[dtype, MAX_TILES, TILE_DATA_SIZE],
        num_active_tiles: Int,
        dt: Scalar[dtype],
    ):
        """Step all environments (CPU).

        Args:
            state: State tensor [BATCH, STATE_SIZE].
            tiles: Track tile data [MAX_TILES, TILE_DATA_SIZE].
            num_active_tiles: Number of valid tiles.
            dt: Time step.
        """
        for env in range(BATCH):
            CarDynamics.step_with_obs[
                BATCH,
                STATE_SIZE,
                OBS_OFFSET,
                OBS_DIM,
                HULL_OFFSET,
                WHEELS_OFFSET,
                CONTROLS_OFFSET,
                MAX_TILES,
            ](env, state, tiles, num_active_tiles, dt)
