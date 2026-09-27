"""The SO-101 families' DELTA action — written once, for every place that
turns a policy's output into a joint target.

    target_j = clamp(q_j + a_j * scale_j, lo_j, hi_j)    a in [-1, 1]
    scale_j  = DELTA_ARM (rad per control step) for the five body joints,
               DELTA_GRIPPER for the gripper (the LAST action word)

`q` is the joint's MEASURED position at the step (not the previous target),
`lo`/`hi` the actuator's ctrlrange — the model's joint limits, in model
radians. so101-nexus's `pd_joint_delta_pos` (0.05 / 0.2).

⚠⚠ THREE READERS, ONE RULE. The PPO driver trains with it
(`ppo_family_driver._delta_to_env`), the pixel DAgger student is labelled
and evaluated with it, and the REAL-ARM deploy
(`examples/so101/pixel_student_deploy_real.mojo`) commands the follower with
it. A scale changed in one of them and not the others would send a policy's
learned step size to a real arm at the wrong magnitude — silently, with the
policy looking merely clumsy (`_a_rule_written_inline_twice_drifts`).
"""

comptime DELTA_ARM: Float64 = 0.05
comptime DELTA_GRIPPER: Float64 = 0.2
comptime DELTA_ACT: Int = 6
"""Five body joints then the gripper."""


@always_inline
def delta_scale(j: Int) -> Float64:
    """Radians per control step at a = 1 for action word `j`."""
    return DELTA_GRIPPER if j == DELTA_ACT - 1 else DELTA_ARM


@always_inline
def delta_target(
    q: Float64, a: Float64, j: Int, lo: Float64, hi: Float64
) -> Float64:
    """The joint target (model radians) for action word `a` at measured `q`,
    clamped to [lo, hi]."""
    var t = q + a * delta_scale(j)
    if t < lo:
        return lo
    if t > hi:
        return hi
    return t
