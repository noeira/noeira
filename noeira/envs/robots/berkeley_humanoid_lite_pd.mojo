"""Berkeley Humanoid Lite — PD tables, GENERATED, DO NOT EDIT.

Regenerate with:  pixi run python tests/robots/bhl_bake.py
CI checks it with: pixi run python tests/robots/bhl_bake.py --check

Source of truth is HybridRobotics/Berkeley-Humanoid-Lite-Assets fc90fedd008b,
`HUMANOID_LITE_CFG`: stiffness, damping, effort and velocity limits per actuator
group, `init_state` as the default angles, and BHL's action scale of 0.25 rad.
The torque law (computed in the env, every substep):

    target = q_default + BHL_ACTION_SCALE · a          (no clip)
    tau    = kp (target − q) − kd q̇, clipped to ± effort, and set to 0 when
             it would push a joint already faster than its speed limit
"""

comptime BHL_N_DOF: Int = 22
comptime BHL_SIM_TIMESTEP: Float64 = 0.002
comptime BHL_CONTROL_DECIMATION: Int = 10
comptime BHL_INIT_ROOT_Z: Float64 = 0.50334
# The root frame (midway between the hip roll joints) in BHL's own `base` frame, which is its CAD
# origin at floor level: a root position exported for BHL's URDF is p + R · this, here.
comptime BHL_ROOT_IN_CAD_BASE_X: Float64 = -0.0290919
comptime BHL_ROOT_IN_CAD_BASE_Z: Float64 = 0.542631
comptime BHL_ACTION_SCALE: Float64 = 0.25
comptime BHL_SOFT_LIMIT_FACTOR: Float64 = 0.9


@always_inline
def bhl_dof_name(i: Int) -> StaticString:
    """The hinge's name in the MJCF."""
    if i == 0:
        return "arm_left_shoulder_pitch_joint"
    elif i == 1:
        return "arm_left_shoulder_roll_joint"
    elif i == 2:
        return "arm_left_shoulder_yaw_joint"
    elif i == 3:
        return "arm_left_elbow_pitch_joint"
    elif i == 4:
        return "arm_left_elbow_roll_joint"
    elif i == 5:
        return "arm_right_shoulder_pitch_joint"
    elif i == 6:
        return "arm_right_shoulder_roll_joint"
    elif i == 7:
        return "arm_right_shoulder_yaw_joint"
    elif i == 8:
        return "arm_right_elbow_pitch_joint"
    elif i == 9:
        return "arm_right_elbow_roll_joint"
    elif i == 10:
        return "leg_left_hip_roll_joint"
    elif i == 11:
        return "leg_left_hip_yaw_joint"
    elif i == 12:
        return "leg_left_hip_pitch_joint"
    elif i == 13:
        return "leg_left_knee_pitch_joint"
    elif i == 14:
        return "leg_left_ankle_pitch_joint"
    elif i == 15:
        return "leg_left_ankle_roll_joint"
    elif i == 16:
        return "leg_right_hip_roll_joint"
    elif i == 17:
        return "leg_right_hip_yaw_joint"
    elif i == 18:
        return "leg_right_hip_pitch_joint"
    elif i == 19:
        return "leg_right_knee_pitch_joint"
    elif i == 20:
        return "leg_right_ankle_pitch_joint"
    elif i == 21:
        return "leg_right_ankle_roll_joint"
    return ""


@always_inline
def bhl_kp(i: Int) -> Float64:
    """Position gain, N·m/rad."""
    if i == 0:
        return 10.0
    elif i == 1:
        return 10.0
    elif i == 2:
        return 10.0
    elif i == 3:
        return 10.0
    elif i == 4:
        return 10.0
    elif i == 5:
        return 10.0
    elif i == 6:
        return 10.0
    elif i == 7:
        return 10.0
    elif i == 8:
        return 10.0
    elif i == 9:
        return 10.0
    elif i == 10:
        return 20.0
    elif i == 11:
        return 20.0
    elif i == 12:
        return 20.0
    elif i == 13:
        return 20.0
    elif i == 14:
        return 20.0
    elif i == 15:
        return 20.0
    elif i == 16:
        return 20.0
    elif i == 17:
        return 20.0
    elif i == 18:
        return 20.0
    elif i == 19:
        return 20.0
    elif i == 20:
        return 20.0
    elif i == 21:
        return 20.0
    return 0.0


@always_inline
def bhl_kd(i: Int) -> Float64:
    """Velocity gain, N·m·s/rad."""
    if i == 0:
        return 2.0
    elif i == 1:
        return 2.0
    elif i == 2:
        return 2.0
    elif i == 3:
        return 2.0
    elif i == 4:
        return 2.0
    elif i == 5:
        return 2.0
    elif i == 6:
        return 2.0
    elif i == 7:
        return 2.0
    elif i == 8:
        return 2.0
    elif i == 9:
        return 2.0
    elif i == 10:
        return 2.0
    elif i == 11:
        return 2.0
    elif i == 12:
        return 2.0
    elif i == 13:
        return 2.0
    elif i == 14:
        return 2.0
    elif i == 15:
        return 2.0
    elif i == 16:
        return 2.0
    elif i == 17:
        return 2.0
    elif i == 18:
        return 2.0
    elif i == 19:
        return 2.0
    elif i == 20:
        return 2.0
    elif i == 21:
        return 2.0
    return 0.0


@always_inline
def bhl_effort(i: Int) -> Float64:
    """Torque limit BHL trains under, N·m."""
    if i == 0:
        return 4.0
    elif i == 1:
        return 4.0
    elif i == 2:
        return 4.0
    elif i == 3:
        return 4.0
    elif i == 4:
        return 4.0
    elif i == 5:
        return 4.0
    elif i == 6:
        return 4.0
    elif i == 7:
        return 4.0
    elif i == 8:
        return 4.0
    elif i == 9:
        return 4.0
    elif i == 10:
        return 6.0
    elif i == 11:
        return 6.0
    elif i == 12:
        return 6.0
    elif i == 13:
        return 6.0
    elif i == 14:
        return 6.0
    elif i == 15:
        return 6.0
    elif i == 16:
        return 6.0
    elif i == 17:
        return 6.0
    elif i == 18:
        return 6.0
    elif i == 19:
        return 6.0
    elif i == 20:
        return 6.0
    elif i == 21:
        return 6.0
    return 0.0


@always_inline
def bhl_speed(i: Int) -> Float64:
    """Joint speed limit BHL trains under, rad/s."""
    if i == 0:
        return 10.0
    elif i == 1:
        return 10.0
    elif i == 2:
        return 10.0
    elif i == 3:
        return 10.0
    elif i == 4:
        return 10.0
    elif i == 5:
        return 10.0
    elif i == 6:
        return 10.0
    elif i == 7:
        return 10.0
    elif i == 8:
        return 10.0
    elif i == 9:
        return 10.0
    elif i == 10:
        return 10.0
    elif i == 11:
        return 10.0
    elif i == 12:
        return 10.0
    elif i == 13:
        return 10.0
    elif i == 14:
        return 10.0
    elif i == 15:
        return 10.0
    elif i == 16:
        return 10.0
    elif i == 17:
        return 10.0
    elif i == 18:
        return 10.0
    elif i == 19:
        return 10.0
    elif i == 20:
        return 10.0
    elif i == 21:
        return 10.0
    return 0.0


@always_inline
def bhl_default_pos(i: Int) -> Float64:
    """Default angle, rad (`init_state`)."""
    if i == 0:
        return 0.0
    elif i == 1:
        return 0.0
    elif i == 2:
        return 0.0
    elif i == 3:
        return 0.0
    elif i == 4:
        return 0.0
    elif i == 5:
        return 0.0
    elif i == 6:
        return 0.0
    elif i == 7:
        return 0.0
    elif i == 8:
        return 0.0
    elif i == 9:
        return 0.0
    elif i == 10:
        return 0.0
    elif i == 11:
        return 0.0
    elif i == 12:
        return -0.2
    elif i == 13:
        return 0.4
    elif i == 14:
        return -0.3
    elif i == 15:
        return 0.0
    elif i == 16:
        return 0.0
    elif i == 17:
        return 0.0
    elif i == 18:
        return -0.2
    elif i == 19:
        return 0.4
    elif i == 20:
        return -0.3
    elif i == 21:
        return 0.0
    return 0.0
