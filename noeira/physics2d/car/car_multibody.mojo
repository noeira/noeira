"""Multi-body (Box2D-faithful) car dynamics.

Models the car as 5 rigid bodies (hull + 4 wheels) connected by 4 revolute
joints, solved with the physics2d sequential-impulse pipeline. This reproduces
Gymnasium's Box2D `car_dynamics.py` far more faithfully than the legacy
single-body `CarDynamics`: the wheels are separate bodies pinned to the hull by
revolute joints, and the iterative constraint solve provides the stability that
a single explicit-Euler force-sum lacks (the latter caused the off-track
spin-in-place limit-cycle).

How it maps to Box2D:
  - Box2D `Car.step(dt)` applies slip-based tire forces to each wheel BODY via
    `ApplyForceToCenter`; here `_tire_force` writes the same force into the
    wheel body's force accumulator (the separate `FORCES_OFFSET` region).
  - Box2D `world.Step(dt, vel_iters, pos_iters)` integrates and solves the
    joints; here that is `SemiImplicitEuler` + `RevoluteJointSolver` over
    `VEL_ITERS`/`POS_ITERS`, optionally sub-stepped `SUBSTEPS` times for
    high-speed stability.
  - The wheel rolling speed (`w.omega`/`w.phase` in Box2D) is NOT a Box2D DOF;
    it is tracked separately here in the `ROLLING_OFFSET` region (one scalar per
    wheel), exactly as Box2D tracks it on the Python side for the tire model.

State sub-block layout (offsets passed as compile-time params, relative to the
env's state row):
  - BODIES_OFFSET : NUM_BODIES * BODY_STATE_SIZE   (hull=0, wheels 1..4)
  - FORCES_OFFSET : NUM_BODIES * 3                 (fx, fy, tau per body)
  - JOINTS_OFFSET : NUM_JOINTS * JOINT_DATA_SIZE   (4 revolute joints)
  - ROLLING_OFFSET: NUM_WHEELS                     (rolling wheel omega)
  - CONTROLS_OFFSET: 3                             (steering, gas, brake)

Wheel body order matches the legacy layout: 1=FL, 2=FR, 3=RL, 4=RR
(front wheels steer; rear wheels drive).

Validated (CPU, grass): full-gas-straight matches Box2D ground truth to ~1e-3
through speed ~65 and stays dead-straight (x~1e-5, omega~1e-4); gas+full-steer
donuts stay bounded (omega ~5.5 vs Box2D ~5.7). See
tests/physics2d/test_car_multibody.mojo.

Reference: gymnasium/envs/box2d/car_dynamics.py + physics2d solver infra.
"""

from std.math import sqrt, cos, sin
from layout import Layout, LayoutTensor

from ..constants import (
    dtype,
    BODY_STATE_SIZE,
    IDX_X,
    IDX_Y,
    IDX_ANGLE,
    IDX_VX,
    IDX_VY,
    IDX_OMEGA,
    IDX_MASS,
    IDX_INV_MASS,
    IDX_INV_INERTIA,
    JOINT_DATA_SIZE,
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
    JOINT_REVOLUTE,
    JOINT_FLAG_LIMIT_ENABLED,
    JOINT_FLAG_MOTOR_ENABLED,
    PI,
    TWO_PI,
)
from ..integrators.euler import SemiImplicitEuler
from ..joints.revolute import RevoluteJointSolver

from .constants import (
    ENGINE_POWER,
    BRAKE_FORCE,
    WHEEL_MOMENT_OF_INERTIA,
    WHEEL_RADIUS,
    FRICTION_COEF,
    SIZE,
    STEERING_LIMIT,
    WHEEL_POS_FL_X,
    WHEEL_POS_FL_Y,
    WHEEL_POS_FR_X,
    WHEEL_POS_FR_Y,
    WHEEL_POS_RL_X,
    WHEEL_POS_RL_Y,
    WHEEL_POS_RR_X,
    WHEEL_POS_RR_Y,
    CTRL_STEERING,
    CTRL_GAS,
    CTRL_BRAKE,
)


struct CarDynamicsMB:
    """Box2D-faithful multi-body car dynamics (hull + 4 wheels + 4 joints).

    Stateless: all state lives in a `[BATCH, STATE_SIZE]` tensor and is accessed
    via compile-time offsets, so the same `step_single_env` runs from a CPU loop
    or a one-thread-per-env GPU kernel.
    """

    # Topology
    comptime NUM_BODIES: Int = 5
    comptime NUM_WHEELS: Int = 4
    comptime NUM_JOINTS: Int = 4
    comptime HULL_BODY: Int = 0  # wheel bodies are 1..4

    # Box2D-measured rigid-body properties (hull.mass/inertia, wheel.mass/inertia)
    comptime HULL_MASS: Float64 = 7.06
    comptime HULL_INERTIA: Float64 = 18.26
    comptime WHEEL_MASS: Float64 = 0.06048
    comptime WHEEL_INERTIA: Float64 = 0.007459

    # Box2D revolute steering joint: maxMotorTorque = 180*900*SIZE*SIZE
    comptime STEER_MAX_MOTOR_TORQUE: Float64 = 180.0 * 900.0 * SIZE * SIZE
    # Steering motor law (Box2D Car.step): motorSpeed = sign(e)*min(50*|e|, 3)
    comptime STEER_GAIN: Float64 = 50.0
    comptime STEER_MAX_SPEED: Float64 = 3.0

    # Solver defaults (validated: substeps are the high-speed-stability lever;
    # iteration count past ~8 made no difference once the joint solve converged).
    comptime DEFAULT_SUBSTEPS: Int = 4
    comptime DEFAULT_VEL_ITERS: Int = 10
    comptime DEFAULT_POS_ITERS: Int = 4

    # =========================================================================
    # Wheel geometry helper
    # =========================================================================

    @always_inline
    @staticmethod
    def _wheel_local(w: Int) -> Tuple[Scalar[dtype], Scalar[dtype]]:
        """Local (hull-frame) position of wheel w in {0=FL,1=FR,2=RL,3=RR}."""
        if w == 0:
            return (Scalar[dtype](WHEEL_POS_FL_X), Scalar[dtype](WHEEL_POS_FL_Y))
        elif w == 1:
            return (Scalar[dtype](WHEEL_POS_FR_X), Scalar[dtype](WHEEL_POS_FR_Y))
        elif w == 2:
            return (Scalar[dtype](WHEEL_POS_RL_X), Scalar[dtype](WHEEL_POS_RL_Y))
        else:
            return (Scalar[dtype](WHEEL_POS_RR_X), Scalar[dtype](WHEEL_POS_RR_Y))

    # =========================================================================
    # Reset: build bodies + joints
    # =========================================================================

    @always_inline
    @staticmethod
    def init_env[
        BATCH: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
        FORCES_OFFSET: Int,
        JOINTS_OFFSET: Int,
        ROLLING_OFFSET: Int,
    ](
        env: Int,
        state: LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        x: Scalar[dtype],
        y: Scalar[dtype],
        angle: Scalar[dtype],
    ):
        """Place the car at (x, y, angle): set up 5 bodies, 4 joints, clear
        rolling speeds and force accumulators."""
        var ca = cos(angle)
        var sa = sin(angle)

        # Hull (body 0)
        var ho = BODIES_OFFSET + Self.HULL_BODY * BODY_STATE_SIZE
        CarDynamicsMB._set_body[BATCH, STATE_SIZE](
            state, env, ho, x, y, angle,
            Scalar[dtype](Self.HULL_MASS), Scalar[dtype](Self.HULL_INERTIA),
        )

        # Wheels (bodies 1..4) + their revolute joints to the hull
        for w in range(Self.NUM_WHEELS):
            var lp = CarDynamicsMB._wheel_local(w)
            var lx = lp[0]
            var ly = lp[1]
            var wx = x + lx * ca - ly * sa
            var wy = y + lx * sa + ly * ca
            var bo = BODIES_OFFSET + (w + 1) * BODY_STATE_SIZE
            CarDynamicsMB._set_body[BATCH, STATE_SIZE](
                state, env, bo, wx, wy, angle,
                Scalar[dtype](Self.WHEEL_MASS),
                Scalar[dtype](Self.WHEEL_INERTIA),
            )

            var jo = JOINTS_OFFSET + w * JOINT_DATA_SIZE
            state[env, jo + JOINT_TYPE] = Scalar[dtype](JOINT_REVOLUTE)
            state[env, jo + JOINT_BODY_A] = Scalar[dtype](Self.HULL_BODY)
            state[env, jo + JOINT_BODY_B] = Scalar[dtype](w + 1)
            state[env, jo + JOINT_ANCHOR_AX] = lx  # hull-local anchor = wheel pos
            state[env, jo + JOINT_ANCHOR_AY] = ly
            state[env, jo + JOINT_ANCHOR_BX] = Scalar[dtype](0.0)  # wheel center
            state[env, jo + JOINT_ANCHOR_BY] = Scalar[dtype](0.0)
            state[env, jo + JOINT_REF_ANGLE] = Scalar[dtype](0.0)
            state[env, jo + JOINT_LOWER_LIMIT] = Scalar[dtype](-STEERING_LIMIT)
            state[env, jo + JOINT_UPPER_LIMIT] = Scalar[dtype](STEERING_LIMIT)
            state[env, jo + JOINT_MAX_MOTOR_TORQUE] = Scalar[dtype](
                Self.STEER_MAX_MOTOR_TORQUE
            )
            state[env, jo + JOINT_MOTOR_SPEED] = Scalar[dtype](0.0)
            state[env, jo + JOINT_FLAGS] = Scalar[dtype](
                JOINT_FLAG_LIMIT_ENABLED | JOINT_FLAG_MOTOR_ENABLED
            )
            RevoluteJointSolver.clear_warm_start[BATCH, STATE_SIZE](
                state, env, jo
            )

            # Rolling wheel speed (tire-model DOF, not a Box2D body DOF)
            state[env, ROLLING_OFFSET + w] = Scalar[dtype](0.0)

        # Clear force accumulators for all bodies
        for b in range(Self.NUM_BODIES):
            state[env, FORCES_OFFSET + b * 3 + 0] = Scalar[dtype](0.0)
            state[env, FORCES_OFFSET + b * 3 + 1] = Scalar[dtype](0.0)
            state[env, FORCES_OFFSET + b * 3 + 2] = Scalar[dtype](0.0)

    @always_inline
    @staticmethod
    def _set_body[
        BATCH: Int, STATE_SIZE: Int
    ](
        state: LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        env: Int,
        o: Int,
        x: Scalar[dtype],
        y: Scalar[dtype],
        angle: Scalar[dtype],
        mass: Scalar[dtype],
        inertia: Scalar[dtype],
    ):
        """Initialize a rigid body at rest with the given pose + mass props."""
        state[env, o + IDX_X] = x
        state[env, o + IDX_Y] = y
        state[env, o + IDX_ANGLE] = angle
        state[env, o + IDX_VX] = Scalar[dtype](0.0)
        state[env, o + IDX_VY] = Scalar[dtype](0.0)
        state[env, o + IDX_OMEGA] = Scalar[dtype](0.0)
        state[env, o + IDX_MASS] = mass
        state[env, o + IDX_INV_MASS] = Scalar[dtype](1.0) / mass
        state[env, o + IDX_INV_INERTIA] = Scalar[dtype](1.0) / inertia

    # =========================================================================
    # Tire force (slip model) applied to one wheel body
    # =========================================================================

    @always_inline
    @staticmethod
    def _tire_force[
        BATCH: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
        FORCES_OFFSET: Int,
        ROLLING_OFFSET: Int,
    ](
        env: Int,
        state: LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        w: Int,
        is_rear: Bool,
        gas: Scalar[dtype],
        brake: Scalar[dtype],
        friction_limit: Scalar[dtype],
        sub_dt: Scalar[dtype],
    ):
        """Box2D slip-based tire model for wheel w: update its rolling speed
        (engine/brake/friction reaction) and write the world-frame contact force
        into the wheel body's force accumulator."""
        var bo = BODIES_OFFSET + (w + 1) * BODY_STATE_SIZE
        var ro = ROLLING_OFFSET + w
        var omega = rebind[Scalar[dtype]](state[env, ro])
        var rad = Scalar[dtype](WHEEL_RADIUS)
        var moment = Scalar[dtype](WHEEL_MOMENT_OF_INERTIA)
        var coef = Scalar[dtype](FRICTION_COEF)
        var zero = Scalar[dtype](0.0)

        # Engine torque (rear wheels): omega += dt*POWER*gas/(MOMENT*(|omega|+5))
        if is_rear and gas > zero:
            var oa = omega if omega >= zero else -omega
            omega = omega + (sub_dt * Scalar[dtype](ENGINE_POWER) * gas) / (
                moment * (oa + Scalar[dtype](5.0))
            )
        # Brake
        if brake >= Scalar[dtype](0.9):
            omega = zero
        elif brake > zero:
            var dir = Scalar[dtype](-1.0) if omega > zero else Scalar[dtype](1.0)
            var val = Scalar[dtype](BRAKE_FORCE) * brake
            var oa = omega if omega >= zero else -omega
            if val > oa:
                val = oa
            omega = omega + dir * val

        # Wheel body world velocity + orientation
        var wangle = rebind[Scalar[dtype]](state[env, bo + IDX_ANGLE])
        var vx = rebind[Scalar[dtype]](state[env, bo + IDX_VX])
        var vy = rebind[Scalar[dtype]](state[env, bo + IDX_VY])
        var cw = cos(wangle)
        var sw = sin(wangle)
        var forw_x = -sw
        var forw_y = cw
        var side_x = cw
        var side_y = sw
        var vf = forw_x * vx + forw_y * vy  # forward speed at wheel
        var vs = side_x * vx + side_y * vy  # lateral speed at wheel
        var vr = omega * rad  # rolling surface speed

        var f_force = (-vf + vr) * coef
        var p_force = -vs * coef
        var fmag = sqrt(f_force * f_force + p_force * p_force)
        if fmag > friction_limit and fmag > Scalar[dtype](1e-8):
            var sc = friction_limit / fmag
            f_force = f_force * sc
            p_force = p_force * sc

        # Friction reaction on rolling speed, then store
        omega = omega - sub_dt * f_force * rad / moment
        state[env, ro] = omega

        # World-frame contact force at the wheel center (no torque on wheel body)
        var fx = p_force * side_x + f_force * forw_x
        var fy = p_force * side_y + f_force * forw_y
        state[env, FORCES_OFFSET + (w + 1) * 3 + 0] = fx
        state[env, FORCES_OFFSET + (w + 1) * 3 + 1] = fy
        state[env, FORCES_OFFSET + (w + 1) * 3 + 2] = zero

    # =========================================================================
    # Full physics step (one environment)
    # =========================================================================

    @always_inline
    @staticmethod
    def _step_core[
        BATCH: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
        FORCES_OFFSET: Int,
        JOINTS_OFFSET: Int,
        ROLLING_OFFSET: Int,
        CONTROLS_OFFSET: Int,
        SUBSTEPS: Int,
        VEL_ITERS: Int,
        POS_ITERS: Int,
    ](
        env: Int,
        state: LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        f_fl: Scalar[dtype],
        f_fr: Scalar[dtype],
        f_rl: Scalar[dtype],
        f_rr: Scalar[dtype],
        dt: Scalar[dtype],
    ):
        """Core step with per-wheel friction limits (FL, FR, RL, RR).

        Reads controls from CONTROLS_OFFSET (steering in [-1,1], gas/brake in
        [0,1]). Mirrors Box2D's `Car.step` + `world.Step`: tire forces onto wheel
        bodies, then integrate + iterative revolute-joint solve, sub-stepped.
        """
        var steering = rebind[Scalar[dtype]](
            state[env, CONTROLS_OFFSET + CTRL_STEERING]
        )
        if steering > Scalar[dtype](1.0):
            steering = Scalar[dtype](1.0)
        elif steering < Scalar[dtype](-1.0):
            steering = Scalar[dtype](-1.0)
        var gas = rebind[Scalar[dtype]](state[env, CONTROLS_OFFSET + CTRL_GAS])
        var brake = rebind[Scalar[dtype]](
            state[env, CONTROLS_OFFSET + CTRL_BRAKE]
        )
        var steer_target = steering * Scalar[dtype](STEERING_LIMIT)

        var sub_dt = dt / Scalar[dtype](SUBSTEPS)
        var gain = Scalar[dtype](Self.STEER_GAIN)
        var max_speed = Scalar[dtype](Self.STEER_MAX_SPEED)
        var ho = BODIES_OFFSET + Self.HULL_BODY * BODY_STATE_SIZE

        for _ in range(SUBSTEPS):
            var hull_angle = rebind[Scalar[dtype]](state[env, ho + IDX_ANGLE])

            # --- steering motors -------------------------------------------
            # Front wheels (joints 0,1 -> bodies 1,2) track the steer target;
            # rear wheels (joints 2,3 -> bodies 3,4) are held at angle 0.
            for w in range(Self.NUM_WHEELS):
                var jo = JOINTS_OFFSET + w * JOINT_DATA_SIZE
                var wbo = BODIES_OFFSET + (w + 1) * BODY_STATE_SIZE
                var rel_angle = (
                    rebind[Scalar[dtype]](state[env, wbo + IDX_ANGLE])
                    - hull_angle
                )
                # The integrator wraps every body angle to [-pi, pi], so this
                # difference jumps by 2*pi whenever the hull spins past +/-pi
                # (it does under sustained full-lock steering). Wrap it back so
                # the motor sees the true small steering angle instead of
                # slamming the wheel a full turn toward a 2*pi-off target.
                if rel_angle > Scalar[dtype](PI):
                    rel_angle -= Scalar[dtype](TWO_PI)
                elif rel_angle < -Scalar[dtype](PI):
                    rel_angle += Scalar[dtype](TWO_PI)
                var ms: Scalar[dtype]
                if w < 2:  # front: toward steer_target
                    var diff = steer_target - rel_angle
                    var d = diff if diff >= Scalar[dtype](0.0) else -diff
                    var dir = (
                        Scalar[dtype](1.0)
                        if diff >= Scalar[dtype](0.0)
                        else Scalar[dtype](-1.0)
                    )
                    var sp = gain * d
                    if sp > max_speed:
                        sp = max_speed
                    ms = dir * sp
                else:  # rear: drive relative angle back to 0
                    ms = -rel_angle * gain
                    if ms > max_speed:
                        ms = max_speed
                    elif ms < -max_speed:
                        ms = -max_speed
                state[env, jo + JOINT_MOTOR_SPEED] = ms

            # --- tire forces -----------------------------------------------
            # Hull carries no direct force (everything via the joints).
            state[env, FORCES_OFFSET + 0] = Scalar[dtype](0.0)
            state[env, FORCES_OFFSET + 1] = Scalar[dtype](0.0)
            state[env, FORCES_OFFSET + 2] = Scalar[dtype](0.0)
            CarDynamicsMB._tire_force[
                BATCH, STATE_SIZE, BODIES_OFFSET, FORCES_OFFSET, ROLLING_OFFSET
            ](env, state, 0, False, gas, brake, f_fl, sub_dt)
            CarDynamicsMB._tire_force[
                BATCH, STATE_SIZE, BODIES_OFFSET, FORCES_OFFSET, ROLLING_OFFSET
            ](env, state, 1, False, gas, brake, f_fr, sub_dt)
            CarDynamicsMB._tire_force[
                BATCH, STATE_SIZE, BODIES_OFFSET, FORCES_OFFSET, ROLLING_OFFSET
            ](env, state, 2, True, gas, brake, f_rl, sub_dt)
            CarDynamicsMB._tire_force[
                BATCH, STATE_SIZE, BODIES_OFFSET, FORCES_OFFSET, ROLLING_OFFSET
            ](env, state, 3, True, gas, brake, f_rr, sub_dt)

            # --- integrate + solve (= Box2D world.Step) --------------------
            SemiImplicitEuler.integrate_velocities_single_env[
                BATCH, Self.NUM_BODIES, STATE_SIZE, BODIES_OFFSET, FORCES_OFFSET
            ](env, state, Scalar[dtype](0.0), Scalar[dtype](0.0), sub_dt)

            RevoluteJointSolver.init_velocity_single_env[
                BATCH,
                Self.NUM_BODIES,
                Self.NUM_JOINTS,
                STATE_SIZE,
                BODIES_OFFSET,
                JOINTS_OFFSET,
            ](env, state, Self.NUM_JOINTS)
            for _ in range(VEL_ITERS):
                RevoluteJointSolver.solve_velocity_single_env[
                    BATCH,
                    Self.NUM_BODIES,
                    Self.NUM_JOINTS,
                    STATE_SIZE,
                    BODIES_OFFSET,
                    JOINTS_OFFSET,
                ](env, state, Self.NUM_JOINTS, sub_dt)

            SemiImplicitEuler.integrate_positions_single_env[
                BATCH, Self.NUM_BODIES, STATE_SIZE, BODIES_OFFSET
            ](env, state, sub_dt)

            for _ in range(POS_ITERS):
                if RevoluteJointSolver.solve_position_single_env[
                    BATCH,
                    Self.NUM_BODIES,
                    Self.NUM_JOINTS,
                    STATE_SIZE,
                    BODIES_OFFSET,
                    JOINTS_OFFSET,
                ](env, state, Self.NUM_JOINTS):
                    break

    @always_inline
    @staticmethod
    def step_single_env[
        BATCH: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
        FORCES_OFFSET: Int,
        JOINTS_OFFSET: Int,
        ROLLING_OFFSET: Int,
        CONTROLS_OFFSET: Int,
        SUBSTEPS: Int = CarDynamicsMB.DEFAULT_SUBSTEPS,
        VEL_ITERS: Int = CarDynamicsMB.DEFAULT_VEL_ITERS,
        POS_ITERS: Int = CarDynamicsMB.DEFAULT_POS_ITERS,
    ](
        env: Int,
        state: LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        friction_limit: Scalar[dtype],
        dt: Scalar[dtype],
    ):
        """Advance one car by `dt` with a UNIFORM surface friction limit
        (FRICTION_LIMIT * road/grass) applied to all four wheels."""
        CarDynamicsMB._step_core[
            BATCH, STATE_SIZE, BODIES_OFFSET, FORCES_OFFSET, JOINTS_OFFSET,
            ROLLING_OFFSET, CONTROLS_OFFSET, SUBSTEPS, VEL_ITERS, POS_ITERS,
        ](env, state, friction_limit, friction_limit, friction_limit, friction_limit, dt)

    @always_inline
    @staticmethod
    def step_single_env_pw[
        BATCH: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
        FORCES_OFFSET: Int,
        JOINTS_OFFSET: Int,
        ROLLING_OFFSET: Int,
        CONTROLS_OFFSET: Int,
        SUBSTEPS: Int = CarDynamicsMB.DEFAULT_SUBSTEPS,
        VEL_ITERS: Int = CarDynamicsMB.DEFAULT_VEL_ITERS,
        POS_ITERS: Int = CarDynamicsMB.DEFAULT_POS_ITERS,
    ](
        env: Int,
        state: LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        f_fl: Scalar[dtype],
        f_fr: Scalar[dtype],
        f_rl: Scalar[dtype],
        f_rr: Scalar[dtype],
        dt: Scalar[dtype],
    ):
        """Advance one car by `dt` with PER-WHEEL friction limits (FL, FR, RL,
        RR). The env looks each wheel body's friction up from the track tiles
        (road vs grass) once per frame — matching Box2D's per-frame `w.tiles`.
        """
        CarDynamicsMB._step_core[
            BATCH, STATE_SIZE, BODIES_OFFSET, FORCES_OFFSET, JOINTS_OFFSET,
            ROLLING_OFFSET, CONTROLS_OFFSET, SUBSTEPS, VEL_ITERS, POS_ITERS,
        ](env, state, f_fl, f_fr, f_rl, f_rr, dt)

    @always_inline
    @staticmethod
    def wheel_world_pos[
        BATCH: Int,
        STATE_SIZE: Int,
        BODIES_OFFSET: Int,
    ](
        env: Int,
        state: LayoutTensor[
            dtype, Layout.row_major(BATCH, STATE_SIZE), MutAnyOrigin
        ],
        w: Int,
    ) -> Tuple[Scalar[dtype], Scalar[dtype]]:
        """World position of wheel body w (0=FL,1=FR,2=RL,3=RR) — used by the
        env to look up per-wheel surface friction from the track tiles."""
        var bo = BODIES_OFFSET + (w + 1) * BODY_STATE_SIZE
        return (
            rebind[Scalar[dtype]](state[env, bo + IDX_X]),
            rebind[Scalar[dtype]](state[env, bo + IDX_Y]),
        )
