"""Unitree G1 as mjlab models it — PD tables, GENERATED, DO NOT EDIT.

Regenerate with:  pixi run python tests/robots/g1_mjlab_bake.py
CI checks it with: pixi run python tests/robots/g1_mjlab_bake.py --check

Source of truth is mjlab 8bb2bc9 `asset_zoo/robots/unitree_g1/g1_constants.py`:
kp = armature · (2π · 10 Hz)², kd = 2 · 2 · armature · (2π · 10 Hz), the
effort limit of the joint's motor group, the reflected rotor inertia as the
armature (also written into the XML), and `KNEES_BENT_KEYFRAME` as the default
angle. The torque law (mjlab's `BuiltinPositionActuator`, computed in the env):

    target = q_default + G1M_ACTION_SCALE(i) · a          (no clip)
    tau    = kp (target − q) − kd q̇, clipped to ± effort, EVERY substep
"""

comptime G1M_N_DOF: Int = 29
comptime G1M_SIM_TIMESTEP: Float64 = 0.005
comptime G1M_CONTROL_DECIMATION: Int = 4
comptime G1M_INIT_ROOT_Z: Float64 = 0.76


@always_inline
def g1m_dof_name(i: Int) -> StaticString:
    if i == 0:
        return "left_hip_pitch_joint"
    elif i == 1:
        return "left_hip_roll_joint"
    elif i == 2:
        return "left_hip_yaw_joint"
    elif i == 3:
        return "left_knee_joint"
    elif i == 4:
        return "left_ankle_pitch_joint"
    elif i == 5:
        return "left_ankle_roll_joint"
    elif i == 6:
        return "right_hip_pitch_joint"
    elif i == 7:
        return "right_hip_roll_joint"
    elif i == 8:
        return "right_hip_yaw_joint"
    elif i == 9:
        return "right_knee_joint"
    elif i == 10:
        return "right_ankle_pitch_joint"
    elif i == 11:
        return "right_ankle_roll_joint"
    elif i == 12:
        return "waist_yaw_joint"
    elif i == 13:
        return "waist_roll_joint"
    elif i == 14:
        return "waist_pitch_joint"
    elif i == 15:
        return "left_shoulder_pitch_joint"
    elif i == 16:
        return "left_shoulder_roll_joint"
    elif i == 17:
        return "left_shoulder_yaw_joint"
    elif i == 18:
        return "left_elbow_joint"
    elif i == 19:
        return "left_wrist_roll_joint"
    elif i == 20:
        return "left_wrist_pitch_joint"
    elif i == 21:
        return "left_wrist_yaw_joint"
    elif i == 22:
        return "right_shoulder_pitch_joint"
    elif i == 23:
        return "right_shoulder_roll_joint"
    elif i == 24:
        return "right_shoulder_yaw_joint"
    elif i == 25:
        return "right_elbow_joint"
    elif i == 26:
        return "right_wrist_roll_joint"
    elif i == 27:
        return "right_wrist_pitch_joint"
    elif i == 28:
        return "right_wrist_yaw_joint"
    return ""

@always_inline
def g1m_kp(i: Int) -> Float64:
    """Stiffness, N·m/rad."""
    if i == 0:  # left_hip_pitch_joint
        return 40.17923863450712
    elif i == 1:  # left_hip_roll_joint
        return 99.09842777666111
    elif i == 2:  # left_hip_yaw_joint
        return 40.17923863450712
    elif i == 3:  # left_knee_joint
        return 99.09842777666111
    elif i == 4:  # left_ankle_pitch_joint
        return 28.50124619574858
    elif i == 5:  # left_ankle_roll_joint
        return 28.50124619574858
    elif i == 6:  # right_hip_pitch_joint
        return 40.17923863450712
    elif i == 7:  # right_hip_roll_joint
        return 99.09842777666111
    elif i == 8:  # right_hip_yaw_joint
        return 40.17923863450712
    elif i == 9:  # right_knee_joint
        return 99.09842777666111
    elif i == 10:  # right_ankle_pitch_joint
        return 28.50124619574858
    elif i == 11:  # right_ankle_roll_joint
        return 28.50124619574858
    elif i == 12:  # waist_yaw_joint
        return 40.17923863450712
    elif i == 13:  # waist_roll_joint
        return 28.50124619574858
    elif i == 14:  # waist_pitch_joint
        return 28.50124619574858
    elif i == 15:  # left_shoulder_pitch_joint
        return 14.25062309787429
    elif i == 16:  # left_shoulder_roll_joint
        return 14.25062309787429
    elif i == 17:  # left_shoulder_yaw_joint
        return 14.25062309787429
    elif i == 18:  # left_elbow_joint
        return 14.25062309787429
    elif i == 19:  # left_wrist_roll_joint
        return 14.25062309787429
    elif i == 20:  # left_wrist_pitch_joint
        return 16.77832748089279
    elif i == 21:  # left_wrist_yaw_joint
        return 16.77832748089279
    elif i == 22:  # right_shoulder_pitch_joint
        return 14.25062309787429
    elif i == 23:  # right_shoulder_roll_joint
        return 14.25062309787429
    elif i == 24:  # right_shoulder_yaw_joint
        return 14.25062309787429
    elif i == 25:  # right_elbow_joint
        return 14.25062309787429
    elif i == 26:  # right_wrist_roll_joint
        return 14.25062309787429
    elif i == 27:  # right_wrist_pitch_joint
        return 16.77832748089279
    elif i == 28:  # right_wrist_yaw_joint
        return 16.77832748089279
    return 0.0

@always_inline
def g1m_kd(i: Int) -> Float64:
    """Damping, N·m·s/rad."""
    if i == 0:  # left_hip_pitch_joint
        return 2.557889775413375
    elif i == 1:  # left_hip_roll_joint
        return 6.308801853496639
    elif i == 2:  # left_hip_yaw_joint
        return 2.557889775413375
    elif i == 3:  # left_knee_joint
        return 6.308801853496639
    elif i == 4:  # left_ankle_pitch_joint
        return 1.814445686584846
    elif i == 5:  # left_ankle_roll_joint
        return 1.814445686584846
    elif i == 6:  # right_hip_pitch_joint
        return 2.557889775413375
    elif i == 7:  # right_hip_roll_joint
        return 6.308801853496639
    elif i == 8:  # right_hip_yaw_joint
        return 2.557889775413375
    elif i == 9:  # right_knee_joint
        return 6.308801853496639
    elif i == 10:  # right_ankle_pitch_joint
        return 1.814445686584846
    elif i == 11:  # right_ankle_roll_joint
        return 1.814445686584846
    elif i == 12:  # waist_yaw_joint
        return 2.557889775413375
    elif i == 13:  # waist_roll_joint
        return 1.814445686584846
    elif i == 14:  # waist_pitch_joint
        return 1.814445686584846
    elif i == 15:  # left_shoulder_pitch_joint
        return 0.907222843292423
    elif i == 16:  # left_shoulder_roll_joint
        return 0.907222843292423
    elif i == 17:  # left_shoulder_yaw_joint
        return 0.907222843292423
    elif i == 18:  # left_elbow_joint
        return 0.907222843292423
    elif i == 19:  # left_wrist_roll_joint
        return 0.907222843292423
    elif i == 20:  # left_wrist_pitch_joint
        return 1.06814150219
    elif i == 21:  # left_wrist_yaw_joint
        return 1.06814150219
    elif i == 22:  # right_shoulder_pitch_joint
        return 0.907222843292423
    elif i == 23:  # right_shoulder_roll_joint
        return 0.907222843292423
    elif i == 24:  # right_shoulder_yaw_joint
        return 0.907222843292423
    elif i == 25:  # right_elbow_joint
        return 0.907222843292423
    elif i == 26:  # right_wrist_roll_joint
        return 0.907222843292423
    elif i == 27:  # right_wrist_pitch_joint
        return 1.06814150219
    elif i == 28:  # right_wrist_yaw_joint
        return 1.06814150219
    return 0.0

@always_inline
def g1m_effort(i: Int) -> Float64:
    """Effort limit, N·m (the torque clip and the motor's ctrlrange)."""
    if i == 0:  # left_hip_pitch_joint
        return 88.0
    elif i == 1:  # left_hip_roll_joint
        return 139.0
    elif i == 2:  # left_hip_yaw_joint
        return 88.0
    elif i == 3:  # left_knee_joint
        return 139.0
    elif i == 4:  # left_ankle_pitch_joint
        return 50.0
    elif i == 5:  # left_ankle_roll_joint
        return 50.0
    elif i == 6:  # right_hip_pitch_joint
        return 88.0
    elif i == 7:  # right_hip_roll_joint
        return 139.0
    elif i == 8:  # right_hip_yaw_joint
        return 88.0
    elif i == 9:  # right_knee_joint
        return 139.0
    elif i == 10:  # right_ankle_pitch_joint
        return 50.0
    elif i == 11:  # right_ankle_roll_joint
        return 50.0
    elif i == 12:  # waist_yaw_joint
        return 88.0
    elif i == 13:  # waist_roll_joint
        return 50.0
    elif i == 14:  # waist_pitch_joint
        return 50.0
    elif i == 15:  # left_shoulder_pitch_joint
        return 25.0
    elif i == 16:  # left_shoulder_roll_joint
        return 25.0
    elif i == 17:  # left_shoulder_yaw_joint
        return 25.0
    elif i == 18:  # left_elbow_joint
        return 25.0
    elif i == 19:  # left_wrist_roll_joint
        return 25.0
    elif i == 20:  # left_wrist_pitch_joint
        return 5.0
    elif i == 21:  # left_wrist_yaw_joint
        return 5.0
    elif i == 22:  # right_shoulder_pitch_joint
        return 25.0
    elif i == 23:  # right_shoulder_roll_joint
        return 25.0
    elif i == 24:  # right_shoulder_yaw_joint
        return 25.0
    elif i == 25:  # right_elbow_joint
        return 25.0
    elif i == 26:  # right_wrist_roll_joint
        return 25.0
    elif i == 27:  # right_wrist_pitch_joint
        return 5.0
    elif i == 28:  # right_wrist_yaw_joint
        return 5.0
    return 0.0

@always_inline
def g1m_armature(i: Int) -> Float64:
    """Reflected rotor inertia, kg·m² (as written into the XML)."""
    if i == 0:  # left_hip_pitch_joint
        return 0.01017752004132231
    elif i == 1:  # left_hip_roll_joint
        return 0.025101924999999997
    elif i == 2:  # left_hip_yaw_joint
        return 0.01017752004132231
    elif i == 3:  # left_knee_joint
        return 0.025101924999999997
    elif i == 4:  # left_ankle_pitch_joint
        return 0.00721945
    elif i == 5:  # left_ankle_roll_joint
        return 0.00721945
    elif i == 6:  # right_hip_pitch_joint
        return 0.01017752004132231
    elif i == 7:  # right_hip_roll_joint
        return 0.025101924999999997
    elif i == 8:  # right_hip_yaw_joint
        return 0.01017752004132231
    elif i == 9:  # right_knee_joint
        return 0.025101924999999997
    elif i == 10:  # right_ankle_pitch_joint
        return 0.00721945
    elif i == 11:  # right_ankle_roll_joint
        return 0.00721945
    elif i == 12:  # waist_yaw_joint
        return 0.01017752004132231
    elif i == 13:  # waist_roll_joint
        return 0.00721945
    elif i == 14:  # waist_pitch_joint
        return 0.00721945
    elif i == 15:  # left_shoulder_pitch_joint
        return 0.003609725
    elif i == 16:  # left_shoulder_roll_joint
        return 0.003609725
    elif i == 17:  # left_shoulder_yaw_joint
        return 0.003609725
    elif i == 18:  # left_elbow_joint
        return 0.003609725
    elif i == 19:  # left_wrist_roll_joint
        return 0.003609725
    elif i == 20:  # left_wrist_pitch_joint
        return 0.00425
    elif i == 21:  # left_wrist_yaw_joint
        return 0.00425
    elif i == 22:  # right_shoulder_pitch_joint
        return 0.003609725
    elif i == 23:  # right_shoulder_roll_joint
        return 0.003609725
    elif i == 24:  # right_shoulder_yaw_joint
        return 0.003609725
    elif i == 25:  # right_elbow_joint
        return 0.003609725
    elif i == 26:  # right_wrist_roll_joint
        return 0.003609725
    elif i == 27:  # right_wrist_pitch_joint
        return 0.00425
    elif i == 28:  # right_wrist_yaw_joint
        return 0.00425
    return 0.0

@always_inline
def g1m_default_pos(i: Int) -> Float64:
    """`KNEES_BENT_KEYFRAME`, rad: the action offset and the reset pose."""
    if i == 0:  # left_hip_pitch_joint
        return -0.312
    elif i == 1:  # left_hip_roll_joint
        return 0.0
    elif i == 2:  # left_hip_yaw_joint
        return 0.0
    elif i == 3:  # left_knee_joint
        return 0.669
    elif i == 4:  # left_ankle_pitch_joint
        return -0.363
    elif i == 5:  # left_ankle_roll_joint
        return 0.0
    elif i == 6:  # right_hip_pitch_joint
        return -0.312
    elif i == 7:  # right_hip_roll_joint
        return 0.0
    elif i == 8:  # right_hip_yaw_joint
        return 0.0
    elif i == 9:  # right_knee_joint
        return 0.669
    elif i == 10:  # right_ankle_pitch_joint
        return -0.363
    elif i == 11:  # right_ankle_roll_joint
        return 0.0
    elif i == 12:  # waist_yaw_joint
        return 0.0
    elif i == 13:  # waist_roll_joint
        return 0.0
    elif i == 14:  # waist_pitch_joint
        return 0.0
    elif i == 15:  # left_shoulder_pitch_joint
        return 0.2
    elif i == 16:  # left_shoulder_roll_joint
        return 0.2
    elif i == 17:  # left_shoulder_yaw_joint
        return 0.0
    elif i == 18:  # left_elbow_joint
        return 0.6
    elif i == 19:  # left_wrist_roll_joint
        return 0.0
    elif i == 20:  # left_wrist_pitch_joint
        return 0.0
    elif i == 21:  # left_wrist_yaw_joint
        return 0.0
    elif i == 22:  # right_shoulder_pitch_joint
        return 0.2
    elif i == 23:  # right_shoulder_roll_joint
        return -0.2
    elif i == 24:  # right_shoulder_yaw_joint
        return 0.0
    elif i == 25:  # right_elbow_joint
        return 0.6
    elif i == 26:  # right_wrist_roll_joint
        return 0.0
    elif i == 27:  # right_wrist_pitch_joint
        return 0.0
    elif i == 28:  # right_wrist_yaw_joint
        return 0.0
    return 0.0

@always_inline
def g1m_action_scale(i: Int) -> Float64:
    """`G1_ACTION_SCALE` = 0.25 · effort / kp, rad per action unit."""
    if i == 0:  # left_hip_pitch_joint
        return 0.5475464629911068
    elif i == 1:  # left_hip_roll_joint
        return 0.35066146637882434
    elif i == 2:  # left_hip_yaw_joint
        return 0.5475464629911068
    elif i == 3:  # left_knee_joint
        return 0.35066146637882434
    elif i == 4:  # left_ankle_pitch_joint
        return 0.43857731392336724
    elif i == 5:  # left_ankle_roll_joint
        return 0.43857731392336724
    elif i == 6:  # right_hip_pitch_joint
        return 0.5475464629911068
    elif i == 7:  # right_hip_roll_joint
        return 0.35066146637882434
    elif i == 8:  # right_hip_yaw_joint
        return 0.5475464629911068
    elif i == 9:  # right_knee_joint
        return 0.35066146637882434
    elif i == 10:  # right_ankle_pitch_joint
        return 0.43857731392336724
    elif i == 11:  # right_ankle_roll_joint
        return 0.43857731392336724
    elif i == 12:  # waist_yaw_joint
        return 0.5475464629911068
    elif i == 13:  # waist_roll_joint
        return 0.43857731392336724
    elif i == 14:  # waist_pitch_joint
        return 0.43857731392336724
    elif i == 15:  # left_shoulder_pitch_joint
        return 0.43857731392336724
    elif i == 16:  # left_shoulder_roll_joint
        return 0.43857731392336724
    elif i == 17:  # left_shoulder_yaw_joint
        return 0.43857731392336724
    elif i == 18:  # left_elbow_joint
        return 0.43857731392336724
    elif i == 19:  # left_wrist_roll_joint
        return 0.43857731392336724
    elif i == 20:  # left_wrist_pitch_joint
        return 0.07450087032950714
    elif i == 21:  # left_wrist_yaw_joint
        return 0.07450087032950714
    elif i == 22:  # right_shoulder_pitch_joint
        return 0.43857731392336724
    elif i == 23:  # right_shoulder_roll_joint
        return 0.43857731392336724
    elif i == 24:  # right_shoulder_yaw_joint
        return 0.43857731392336724
    elif i == 25:  # right_elbow_joint
        return 0.43857731392336724
    elif i == 26:  # right_wrist_roll_joint
        return 0.43857731392336724
    elif i == 27:  # right_wrist_pitch_joint
        return 0.07450087032950714
    elif i == 28:  # right_wrist_yaw_joint
        return 0.07450087032950714
    return 0.0
