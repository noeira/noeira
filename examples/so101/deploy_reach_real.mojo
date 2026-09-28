"""Run a sim-trained reach policy on the PHYSICAL SO-101 follower.

The last rung of `ROADMAP_2026_08.md` §5.4: *"import -> GPU-batched training ->
policy binary -> the arm moves."* Same checkpoint as
`sac_so_arm101_reach_eval_cpu.mojo`, driving hardware instead of a simulator.

    pixi run build-serial      # ONCE
    pixi run mojo run -I . examples/so101/deploy_reach_real.mojo
    # a training run's own weights instead of the promoted `reach` role:
    pixi run mojo run -I . examples/so101/deploy_reach_real.mojo --ckpt <run_id>

## ⚠⚠ THIS MOVES A REAL ARM. Read the safety notes before the first run.

## The interesting part: the simulator is the observation function

The policy wants 21 numbers — `qpos(6) + qvel(6) + ee(3) + target(3) +
ee_to_target(3)`. The arm can only supply the first twelve:

| term | where it comes from on hardware |
|---|---|
| qpos | servo ticks -> `SimJointMap.to_sim` (gated vs so101-nexus) |
| qvel | `Present_Velocity`, register 58, one more sync-read |
| ee | ⚠ NO SENSOR GIVES THIS — forward kinematics on the measured qpos |
| target | ours to choose; there is no target in the real world |
| ee_to_target | derived |

So this program keeps a `Phyics3dEnv` alive purely as a **kinematics oracle**:
measured `qpos` goes in through `set_state` (which runs
`_sync_mocap_to_fields` then `_fields_fk`), and the end-effector's world
position comes out. It reuses the FK gated against MuJoCo in
`tests/robots/test_so_arm101_vs_mujoco.mojo` — the same model the policy
trained in, used as a sensor.

⚠ Being honest about what this demonstrates: the target is three numbers WE
pick, so the arm is not finding anything. This is RL-learned inverse
kinematics, and plain IK does that job better. Its value is that it exercises
every piece a later task needs — obs construction, action mapping, control
rate, safety — and yields a transfer NUMBER: commanded point vs. where the
end-effector actually ends up.

## Safety

Four guards, on top of the two `SO101Arm` already enforces (clamp to the
calibrated range; clamp the step to `present +/- max_step_ticks`):

1. the follower's goal is parked on its OWN present pose before torque is
   armed, so arming holds instead of snapping to a stale `Goal_Position`;
2. a partial `sync_read` skips the tick rather than commanding a half-updated
   pose;
3. the commanded angle is clamped to the MODEL's `ctrlrange` before it is ever
   mapped back to ticks — the policy trained inside those limits and has no
   reason to be trusted outside them;
4. the run ends through `robot/so101/deploy_shutdown.return_and_release`, from
   a `finally`: the follower ramps back to the pose it started from, and torque
   is released there, on the operator's word — not wherever the policy left
   the arm (the ACT deployment's arm FELL that way). If the ramp does not
   arrive, torque is LEFT ON.

⚠⚠ A `finally` does NOT run on an abort or a signal. If this dies hard, the
follower is left holding its pose — run `pixi run soarm-torque-off`. That is
the recovery, not the `finally`.
"""

from std.ffi import external_call
from std.sys import argv
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.deep_agents.sac import SAC
from noeira.envs.phyics3d_env import Phyics3dEnv
from noeira.envs.robots.so_arm101_xml import SoArm101Model
from noeira.envs.robots.so_arm101 import SoArm101ReachConfig
from noeira.physics3d.fields import actuator_column
from noeira.physics3d.gpu.constants import (
    ACT_IDX_CTRL_MAX,
    ACT_IDX_CTRL_MIN,
    META_IDX_TASK_PARAM_6,
)
from noeira.robot.so101 import SO101Arm, SO101_N, joint_name
from noeira.robot.so101.ports import follower_port, port_refusal
from noeira.robot.so101.sim_map import SimJointMap
from noeira.robot.so101.deploy_shutdown import return_and_release
from noeira.io.fileio import StdinReader, stdin_is_tty
from noeira.utils.fmt import col, fixed, pad_left, pad_right
from noeira.core.policy import describe_policy, resolve_policy
from noeira.core.run import resolve_checkpoint

comptime POLICY_PROJECT = "so101"
comptime POLICY_ROLE = "reach"
comptime CHECKPOINT_FALLBACK = "sac_so_arm101_reach.ckpt"
"""⚠⚠ THE DEPLOY PATH NAMES A ROLE, NOT A RUN — §8, decision 2.

`resolve_policy` returns `projects/so101/policies/reach.ckpt` when a checkpoint
has been promoted into that role, so a better one is a `project-promote` away
and NOTHING HERE IS EDITED, rebuilt or re-flashed.

⚠ It falls back to the old flat constant when no policy exists. A deploy path
is the one place where failing closed is worse than failing open: an arm that
will not start because the project layer is not set up yet is a regression for
someone who only wanted to run what worked yesterday.

⚠ AND THE ARM PRINTS WHICH ONE IT GOT, BEFORE IT MOVES. Silently loading
different weights than the operator expects is precisely the failure this layer
exists to prevent."""

comptime EnvT = Phyics3dEnv[
    SoArm101Model, SoArm101ReachConfig, DT, TERMINATE_ON_UNHEALTHY=False
]
comptime OBS_DIM = EnvT.OBS_DIM  # 27 (incl. the previous action)
comptime ACT_DIM = EnvT.ACTION_DIM  #  6
comptime HIDDEN = 256
comptime BATCH = 256
comptime REPLAY_CAPACITY = 100_000
comptime ACTION_SCALE = Scalar[DT](1.0)  # normalized; MUST match the trainer

comptime HZ = 50
"""⚠ 50, NOT 30 — IT MUST MATCH THE RATE THE POLICY TRAINED AT.
`SoArmReachConfig` is `FRAME_SKIP = 10` over a 0.002 s timestep, i.e. a 50 Hz
control rate, and the action is a POSITION TARGET: at 30 Hz every commanded
pose is held 67% longer than the policy ever saw, which changes the closed
loop, not just the latency. This was 30 because the loop could not go faster —
it spent six serial round trips per tick reading velocity one joint at a time.
`SO101Arm.read_velocities` makes that one `sync_read`, so the tick is now
three transactions (positions, velocities, goals) instead of eight."""
comptime SECONDS = 5
"""Short on purpose. This program had never been executed against hardware, so
the first live run is a BRING-UP of the guards, not a demonstration of the
policy — five seconds is enough to read the achieved rate and see the arm
settle, and short enough that a guard that does not work costs little."""
comptime MAX_STEP_TICKS = 60
"""Per-tick slew bound, in servo ticks. At 50 Hz this is 25 * 50 = 1250
ticks/s ~= 110 deg/s.

⚠⚠ BACK TO 60 (~264 deg/s) BECAUSE 25 WAS THROTTLING THE POLICY, and the
measurement is unambiguous — two live runs differing ONLY in this value:

    --step 25   closest 54.2 mm   inside 20 mm    0/1000   throttle 14.4x
    --step 60   closest  3.9 mm   inside 20 mm  395/1000   throttle  1.3x

At 25 the arm trailed its commanded pose by a mean of 327 ticks and never got
within 5 cm; the CLAMP was holding it out, not the policy. `max_step_ticks`
has no counterpart in sim, so a tight one is a sim2real gap and not merely a
safety margin. Watch the throttle line in the report: above ~2x, this number
is deciding the run.

⚠ The earlier note, kept because it is still true of the OPENING move: It measured the
policy asking for a mean move of **1252 ticks** and a max of 2197 — the
distance from the arm's parked pose to the pose the policy wants, roughly 110
degrees. The clamp is what turns that into a slew rather than a snap, so its
value IS the opening motion's speed: at 60 the arm crosses that gap in 0.4 s,
at 25 in about 1 s. Raise it once the first live run has shown the direction
is right."""

comptime SMOOTH = 0.05
"""Low-pass on the COMMAND: `cmd <- (1-a)*cmd + a*policy`. 1.0 disables it.

⚠⚠ THE RAW COMMAND STREAM IS NOT SOMETHING TO SEND A SERVO. Measured in sim
over 24 episodes (greedy, after the arm has arrived): the commanded position
moves **~500 ticks per joint per control step and reverses direction on 83% of
consecutive steps**. That is a 44-degree-per-step bang-bang command at 50 Hz.
`max_step_ticks` bounds each move but does not stop the reversals, so the arm
would sit at the target buzzing — continuous current reversal through the gear
train, which is the "will this damage the motors" question answered YES-ish.

The reward cannot see it: `tolerance` has a 0.25 m margin against a 0.02 m
success radius, so hovering 40 mm away scores 0.985 per step. Shake is free.

⚠ AND SMOOTHING IS A DEVIATION FROM TRAINING, so it was measured rather than
assumed — same 24 targets, greedy, in sim:

    alpha | reached <=20mm | mean closest | chatter ticks/step | return
     1.00 |        15/24   |     20.4 mm  |            2520    |  480.5
     0.50 |        15/24   |     18.6 mm  |            1029    |  480.2
     0.30 |        15/24   |     17.9 mm  |             665    |  479.1
     0.20 |        16/24   |     16.9 mm  |             470    |  476.9
     0.10 |        17/24   |     18.7 mm  |             223    |  474.7
     0.05 |        12/24   |     26.5 mm  |              72    |  475.5

Anything in 0.1-0.3 costs NOTHING in task terms — the reach rate is flat
inside the noise of 24 episodes — and buys a 4x to 11x calmer command. 0.05
breaks the policy (the filter outruns the task). ⚠⚠ 0.05, NOT THE 0.15 THIS SHIPPED WITH. An EMA's steady-state response to a
signal alternating EVERY step is `a / (2 - a)`: 0.15 leaves **8.1%** of the
chatter standing, 0.05 leaves 2.6%. The sim sweep had already shown 0.05 to be
4x calmer at no measurable cost in reach, and 0.15 was chosen as "the middle
of the free band" — the wrong reading for hardware, where the whole point is
the residual. Confirmed live.

THE REAL FIX IS IN THE REWARD, and it is `REWARD_MARGIN`, not an action-rate
penalty: at the old 0.25 an orbit through the target cost 0.015/step, so the
policy had no reason to stop. This filter is the guard for a policy trained
before that was fixed."""

comptime SUCCESS_MM = 20.0
"""`SoArmReachConfig.TARGET_RADIUS` in millimetres — the radius inside which
the task calls it reached."""

# The target, in the model's base frame. Inside the shell the policy trained
# on (0.15-0.30 m, elevation 0.17-1.22 rad), or it is being asked for a pose
# it never saw.
comptime TARGET_X = 0.22
comptime TARGET_Y = 0.0
comptime TARGET_Z = 0.18


def col_name(i: Int) -> String:
    """Joint name, padded, so the self-check reads as a table."""
    return pad_right(String(joint_name(i)), 12)


def _sleep_until(deadline_ns: Int):
    """Spin. Measured better than `usleep` on this box — see
    `examples/so101/teleop.mojo` for the table."""
    while perf_counter_ns() < deadline_ns:
        pass


def main() raises:
    # ⚠⚠ THE ARM MOVES ONLY WITH AN EXPLICIT `--arm` (`--live`, its old name,
    # is kept as an alias — it always meant "arm"). Everything else — the
    # bus, the observation, the forward kinematics, the policy, the joint
    # mapping, the clamps, the filter and the loop rate — runs identically in
    # DRY RUN, which never arms torque and never writes a goal. A first
    # contact with hardware that exercises every path except the dangerous one
    # is worth more than a cautious live run, because a fault in ANY of those
    # paths shows up as a printed number instead of as a motion.
    # ⚠ RUNTIME, NOT COMPTIME, for `seconds` and `step`. Hardware bring-up is
    # a sweep — the first live run showed the arm rate-limited by the clamp for
    # its whole duration, and answering "how long does it need" or "how much
    # clamp is enough" should not cost a two-minute rebuild each time. `--arm`
    # stays a flag rather than a default for the reason it always was.
    var live = False
    var seconds = SECONDS
    var step_ticks = MAX_STEP_TICKS
    var smooth = SMOOTH
    var ckpt_arg = String("")
    var args = argv()
    for i in range(1, len(args)):
        var a = String(args[i])
        if a == "--arm" or a == "--live":
            live = True
        elif a == "--seconds" and i + 1 < len(args):
            seconds = Int(String(args[i + 1]))
        elif a == "--step" and i + 1 < len(args):
            step_ticks = Int(String(args[i + 1]))
        elif a == "--smooth" and i + 1 < len(args):
            smooth = Float64(String(args[i + 1]))
        elif a == "--ckpt" and i + 1 < len(args):
            ckpt_arg = String(args[i + 1])
    if ckpt_arg.byte_length() > 0:
        # ⚠ RESOLVED BEFORE ANYTHING IS OPENED. A RUN ID becomes its
        # `checkpoints/last.ckpt` (what the reach trainer writes); a file is
        # used as is; anything else raises here, not after the arm is up.
        ckpt_arg = resolve_checkpoint(ckpt_arg, String("last"))
    print("=" * 70)
    if live:
        print("SO-ARM101 reach — SIM-TRAINED POLICY ON THE REAL ARM  [LIVE]")
    else:
        print("SO-ARM101 reach — DRY RUN (no torque, no goals written)")
        print("  pass --arm to actually move the arm")
    print("=" * 70)

    # ── the policy ────────────────────────────────────────────────────────
    var agent = SAC["cpu", OBS_DIM, ACT_DIM, BATCH, REPLAY_CAPACITY, HIDDEN](
        action_scale=ACTION_SCALE,
    )
    var ckpt_path = ckpt_arg
    if ckpt_path.byte_length() == 0:
        ckpt_path = resolve_policy(
            String(POLICY_PROJECT), String(POLICY_ROLE), String(CHECKPOINT_FALLBACK)
        )
    try:
        agent.load(ckpt_path)
    except e:
        print("ERROR loading", ckpt_path, "-", e)
        print("Train first: examples/so101/sac_so_arm101_reach_training_gpu.mojo")
        print("Or promote one: pixi run project-promote <run_id> best --as reach")
        return
    print("  policy          =", ckpt_path)
    # ⚠ WHICH RUN, AND WHAT THE HUMAN SAID ABOUT IT — before the arm moves.
    var provenance = describe_policy(String(POLICY_PROJECT), String(POLICY_ROLE))
    if ckpt_arg.byte_length() > 0:
        print("  promoted from   = (none — named by --ckpt, not the promoted role)")
    elif provenance:
        print("  promoted from   =", provenance)
    else:
        print(
            "  promoted from   = (none — this is the flat fallback"
            " constant, not a promoted policy)"
        )

    # ── the kinematics oracle ─────────────────────────────────────────────
    # A full env, used for FK only. It is never stepped: `set_state` runs the
    # mocap sync and forward kinematics, which is all the observation needs.
    var ctx = DeviceContext()
    var env = EnvT(ctx)
    _ = env.reset()

    var sf = SoArm101Model.make_spec_fields[DType.float64]()
    var lo_col = actuator_column(sf, ACT_IDX_CTRL_MIN, SO101_N)
    var hi_col = actuator_column(sf, ACT_IDX_CTRL_MAX, SO101_N)
    var lo = Array[Float64, SO101_N](fill=0.0)
    var hi = Array[Float64, SO101_N](fill=0.0)
    for i in range(SO101_N):
        lo[i] = Float64(lo_col[i])
        hi[i] = Float64(hi_col[i])
    var jmap = SimJointMap.identity(lo^, hi^)

    # ── the arm ───────────────────────────────────────────────────────────
    var f_port = follower_port()
    var why_f = port_refusal(f_port, String("follower"))
    if why_f.byte_length() > 0:
        raise Error("deploy_reach: " + why_f)
    print("  opening         =", f_port)
    var arm = SO101Arm(f_port, max_step_ticks=step_ticks)
    arm.bus.timeout_ms = 20
    print("  target          = (", TARGET_X, TARGET_Y, TARGET_Z, ")")
    print("=" * 70)

    var raw = Array[Int32, SO101_N](fill=0)
    if arm.read_positions(Span(raw)) != SO101_N:
        raise Error("deploy: follower did not report 6 positions — not arming")
    # The one pose known to be safe: the arm rested here, unpowered, before
    # anything was armed. The shutdown ramps back to it.
    var start_pose = List[Int32](length=SO101_N, fill=0)
    for i in range(SO101_N):
        start_pose[i] = raw[i]
    var stdin = StdinReader()
    var interactive = stdin_is_tty()

    # ── the mapping self-check ────────────────────────────────────────────
    #
    # ⚠⚠ `from_sim` IS THE ONLY LINK IN THIS CHAIN THAT NOTHING HAS EVER
    # EXERCISED. `to_sim` is gated against so101-nexus's reference function and
    # is driven every frame by `teleop_sim.mojo`; its inverse is used HERE and
    # nowhere else. A sign error in it does not produce an error — it produces
    # a mirrored pose, at full slew, on a real arm.
    #
    # So round-trip each joint's MEASURED position through both directions
    # before anything is armed. It is not proof the convention is right (a
    # convention wrong in both directions round-trips perfectly — `to_sim` is
    # what pins that, and it is separately gated), but it is proof the two are
    # consistent, which is the failure this file could introduce on its own.
    # ⚠⚠ THE MAP IS TESTED AT KNOWN-INTERIOR POINTS, NOT AT THE ARM'S POSE.
    # The first version round-tripped each joint's MEASURED position and
    # refused to arm when it did not come back — which conflates two entirely
    # different things, because `to_sim` CLAMPS. A joint parked outside its
    # range cannot round-trip by construction, so the check fired on a healthy
    # arm twice: `shoulder_lift` 72 ticks past the MODEL range, then the
    # gripper 11 ticks past its CALIBRATED range. Neither was a mapping fault.
    #
    # What the check is actually for is a SIGN or OFFSET error in `from_sim`,
    # the one link nothing else exercises, whose failure on hardware is a
    # MIRRORED POSE AT FULL SLEW. That is a property of the MAP and can be
    # tested at points chosen to be interior — where no clamp can fire —
    # independently of where the arm happens to be sitting.
    var worst = 0.0
    for i in range(SO101_N):
        for k in range(3):
            var frac = 0.25 + 0.25 * Float64(k)
            var v = jmap.sim_lo[i] + frac * (jmap.sim_hi[i] - jmap.sim_lo[i])
            var back = jmap.to_sim(arm.cal, i, jmap.from_sim(arm.cal, i, v))
            var e = back - v
            if e < 0.0:
                e = -e
            if e > worst:
                worst = e
    if worst > 0.02:
        raise Error(
            "deploy: to_sim/from_sim do not round-trip at INTERIOR points"
            " (worst " + fixed(worst, 4) + " rad) — NOT arming. A sign or"
            " offset in the joint mapping is inconsistent, and the failure it"
            " produces on hardware is a MIRRORED pose at full slew."
        )
    print(
        "  map round-trips at interior points, worst",
        fixed(worst, 4), "rad (tick quantisation is ~0.0015)",
    )

    # ── where the arm is parked, reported and never fatal ──────────────────
    # Two different "outside"s, and they are not the same fault:
    #   MODEL   — past the `ctrlrange` the policy trained in. Common: three
    #             joints have calibrated travel that exceeds the model's.
    #   CALIB   — past `lerobot-calibrate`'s own recorded endpoints. Means the
    #             arm was moved beyond where it was calibrated, or drifted.
    # Either way the first command clamps back inside, so neither blocks a run.
    var n_model = 0
    var n_calib = 0
    for i in range(SO101_N):
        var over = jmap.clamped_by(arm.cal, i, raw[i])
        var note = String("")
        if over > 0.0:
            n_model += 1
            note += " ⚠ past the MODEL range by " + fixed(over, 3) + " rad"
        if raw[i] < arm.cal.range_min[i]:
            n_calib += 1
            note += " ⚠ " + String(Int(arm.cal.range_min[i] - raw[i])) + (
                " ticks BELOW the calibrated minimum"
            )
        elif raw[i] > arm.cal.range_max[i]:
            n_calib += 1
            note += " ⚠ " + String(Int(raw[i] - arm.cal.range_max[i])) + (
                " ticks ABOVE the calibrated maximum"
            )
        print(
            "  " + col_name(i), "raw", pad_left(String(Int(raw[i])), 6),
            "-> rad", col(jmap.to_sim(arm.cal, i, raw[i]), 7, 3), note,
        )
    if n_model > 0 or n_calib > 0:
        print(
            "  ⚠", n_model, "joint(s) past the model range,", n_calib,
            "past calibration. The policy's first\n     observation is"
            " clamped there; its first command moves them back inside."
        )
    print()

    # Guard 1: park the goal on the present pose BEFORE arming torque.
    # ⚠ SKIPPED ENTIRELY IN DRY RUN — writing `Goal_Position` on an unarmed
    # servo is harmless, but not writing it at all is the only way to be sure
    # a dry run cannot move anything, and "sure" is the point of the mode.
    if live:
        arm.set_position_mode()
        var hold = arm.max_step_ticks
        arm.max_step_ticks = 0
        arm.write_goals(Span(raw))
        arm.max_step_ticks = hold
        arm.set_torque(True)
        print("follower torque ON\n")
    else:
        print("dry run — torque left OFF, the arm is backdrivable\n")

    var obs = List[Scalar[DT]]()
    for _ in range(OBS_DIM):
        obs.append(Scalar[DT](0))
    var action = List[Scalar[DT]]()
    for _ in range(ACT_DIM):
        action.append(Scalar[DT](0))

    var qp = List[Float64]()
    var qv = List[Float64]()
    for _ in range(SO101_N):
        qp.append(0.0)
        qv.append(0.0)

    var vraw = Array[Int32, SO101_N](fill=0)
    var goals = Array[Int32, SO101_N](fill=0)
    # ⚠ SEEDED FROM THE ARM'S OWN POSE, not from zero. A filter starting at
    # zero would spend its first ticks sweeping the arm from wherever it is
    # toward the folded pose — a large unwanted motion, produced by the very
    # thing that is there to make motion gentler.
    var cmd = List[Float64]()
    for i in range(SO101_N):
        cmd.append(jmap.to_sim(arm.cal, i, raw[i]))
    var period_ns = 1_000_000_000 // HZ
    var ticks = HZ * seconds
    var dropped = 0
    var best_err = 1.0e9
    var last_err = 0.0
    var inside = 0
    var max_step = 0.0
    var sum_step = 0.0
    var n_step = 0.0
    var reversals = 0.0
    var last_delta = List[Float64](length=SO101_N, fill=0.0)
    var last_goal = List[Float64](length=SO101_N, fill=0.0)
    for i in range(SO101_N):
        last_goal[i] = Float64(raw[i])

    var loop_t0 = perf_counter_ns()
    try:
        for t in range(ticks):
            var t0 = perf_counter_ns()

            # ── measure ───────────────────────────────────────────────────
            if arm.read_positions(Span(raw)) != SO101_N:
                dropped += 1
                _sleep_until(t0 + period_ns)
                continue
            # ⚠ ONE ROUND TRIP, NOT SIX. `Present_Velocity` is sign-magnitude
            # at bit 15, which `sync_read` decodes through `sign_bit_for` —
            # the same path the positions take. Ticks per second -> rad/s uses
            # the same 4095 the position map does, not 4096.
            var n_vel = arm.read_velocities(Span(vraw))
            if n_vel != SO101_N:
                dropped += 1
                _sleep_until(t0 + period_ns)
                continue
            for i in range(SO101_N):
                qp[i] = jmap.to_sim(arm.cal, i, raw[i])
                qv[i] = Float64(vraw[i]) * 2.0 * 3.141592653589793 / 4095.0

            # ── the simulator AS A SENSOR: qpos -> FK -> end-effector ─────
            env.d.mocap_pos.data[SoArm101ReachConfig.TARGET_BODY * 3 + 0] = (
                Scalar[DT](TARGET_X)
            )
            env.d.mocap_pos.data[SoArm101ReachConfig.TARGET_BODY * 3 + 1] = (
                Scalar[DT](TARGET_Y)
            )
            env.d.mocap_pos.data[SoArm101ReachConfig.TARGET_BODY * 3 + 2] = (
                Scalar[DT](TARGET_Z)
            )
            env.set_state(qp, qv)
            # ⚠⚠ THE PREVIOUS ACTION, WRITTEN BY HAND. The observation's last
            # six entries are it (`SoArmReachConfig.RECORD_PREV_ACTION`), and
            # the ENV fills them at action-application time inside `step` —
            # which this program never calls. It drives the simulator purely
            # as a kinematics oracle through `set_state`, so nothing would
            # write those slots and the policy would read six zeros forever:
            # a policy trained to see its own last command, deployed blind to
            # it, with no error anywhere.
            #
            # `action` still holds the PREVIOUS tick's output at this point —
            # it is overwritten by `select_greedy_action` below — which is
            # exactly the value the env would have recorded. On the first tick
            # it is zero, matching a fresh reset.
            for i in range(SO101_N):
                env.d.meta.data[META_IDX_TASK_PARAM_6 + i] = Scalar[DT](
                    action[i]
                )

            var st = env.get_state()
            for i in range(OBS_DIM):
                obs[i] = Scalar[DT](st.data[i])

            # ── act ──────────────────────────────────────────────────────
            agent.select_greedy_action(obs, action)
            for i in range(SO101_N):
                # ⚠⚠ THE SAME AFFINE MAP THE ENV APPLIES. The policy's action
                # is NORMALIZED [-1, 1] (`SoArmReachConfig.NORMALIZED_ACTIONS`)
                # and the simulator maps it onto each joint's `ctrlrange`
                # inside `apply_actions`. Hardware never runs that code, so
                # this is the one place the same map has to be repeated — and
                # getting it wrong here is a policy that behaves in sim and
                # commands the wrong POSE on the arm, with nothing raising.
                # `jmap.sim_lo/sim_hi` ARE that ctrlrange, read from the model.
                var lo_i = jmap.sim_lo[i]
                var hi_i = jmap.sim_hi[i]
                var a = lo_i + (Float64(action[i]) + 1.0) * 0.5 * (hi_i - lo_i)
                # Guard 3 stays: an action outside [-1, 1] maps outside the
                # range, and the policy is not trusted past its own limits.
                a = min(hi_i, max(lo_i, a))
                # Guard 5: low-pass the command — see `SMOOTH`. AFTER the
                # clamp, so the filter can never carry a state the limits
                # already rejected, and BEFORE `from_sim`, so what is filtered
                # is an ANGLE and not a tick count whose sign convention
                # differs per joint.
                cmd[i] = (1.0 - smooth) * cmd[i] + smooth * a
                goals[i] = jmap.from_sim(arm.cal, i, cmd[i])
                # The move the servo is being asked for THIS tick, in ticks —
                # the quantity `max_step_ticks` bounds and the one that decides
                # whether this is gentle. Tracked in dry run too, where it is
                # the whole output.
                var delta = Float64(goals[i]) - last_goal[i]
                # ⚠ THE REVERSAL RATE IS THE SHAKE, and it is a different
                # quantity from the step SIZE. A goal that advances 25 ticks
                # every tick in one direction is a smooth slew; one that
                # alternates +25/-25 at the same size is a 25 Hz buzz through
                # the gear train. Only this counter tells them apart, and it
                # is the number to watch when tuning `--smooth`.
                if delta * last_delta[i] < 0.0:
                    reversals += 1.0
                last_delta[i] = delta
                last_goal[i] = Float64(goals[i])
                var step = Float64(goals[i] - raw[i])
                if step < 0.0:
                    step = -step
                if step > max_step:
                    max_step = step
                sum_step += step
                n_step += 1.0
            if live:
                arm.write_goals(Span(goals))

            # ── report ───────────────────────────────────────────────────
            var b = 12  # qpos(6) + qvel(6)
            var dx = Float64(obs[b + 6])
            var dy = Float64(obs[b + 7])
            var dz = Float64(obs[b + 8])
            last_err = (dx * dx + dy * dy + dz * dz) ** 0.5
            if last_err < best_err:
                best_err = last_err
            if last_err * 1000.0 <= SUCCESS_MM:
                inside += 1
            if t % HZ == 0:
                print(
                    "t=" + String(t // HZ) + "s  ee->target = "
                    + fixed(last_err * 1000.0, 1) + " mm"
                )
            _sleep_until(t0 + period_ns)
    finally:
        # Home first, then release — on a dry run this is the unconditional
        # one-packet release, the net under every path that could have armed.
        var released = return_and_release(
            arm, start_pose, live, True, stdin, interactive
        )
        if released:
            print("\nfollower torque OFF")
        else:
            print(
                "⚠ the follower is STILL ENERGISED — the return did not"
                " arrive; run `pixi run soarm-torque-off` once it is safe."
            )

    print("=" * 70)
    print("TRANSFER RESULT — sim-trained reach on hardware")
    print("  target            = (", TARGET_X, TARGET_Y, TARGET_Z, ")")
    print("  closest approach  =", fixed(best_err * 1000.0, 1), "mm")
    print("  final error       =", fixed(last_err * 1000.0, 1), "mm")
    # ⚠ CLOSEST APPROACH IS THE FLATTERING NUMBER, and on its own it is the
    # wrong one: in sim this policy's mean CLOSEST approach is ~20 mm while
    # its mean FINAL distance is ~38 mm — it passes through the target and
    # drifts off rather than arriving. "Held" is the question a reach task is
    # actually asking, so it gets its own row.
    print(
        "  steps inside", fixed(SUCCESS_MM, 0), "mm =", inside, "/", ticks,
        "(" + fixed(100.0 * Float64(inside) / Float64(ticks), 0) + "%)",
    )
    print("  dropped ticks     =", dropped, "/", ticks)
    # ⚠ THE RATE THE LOOP ACTUALLY HELD, not the one it asked for. The bus is
    # the budget: three round trips a tick at ~1.3 ms each leaves ~16 ms of
    # the 20 ms period, and a serial hiccup eats it silently. A policy trained
    # at 50 Hz running at 35 would look like a worse policy.
    var achieved = Float64(ticks) * 1e9 / Float64(perf_counter_ns() - loop_t0)
    _ = seconds
    print(
        "  control rate      =", fixed(achieved, 1), "Hz achieved of",
        Int(HZ), "asked (policy trained at 50 Hz)",
    )
    # ⚠ THE ATTENUATION, NOT JUST THE ALPHA. An EMA's steady-state response to
    # a signal that alternates every step is `a / (2 - a)`, so 0.15 leaves 8%
    # of the chatter standing and 0.05 leaves 2.6%. Printing only the alpha
    # invites reading 0.15 as "smoothed" when it is barely filtered — and the
    # policy's command reverses direction on 83-97% of steps, which is exactly
    # the signal this formula is about.
    print(
        "  smoothing         = alpha", fixed(smooth, 3),
        "-> leaves", fixed(100.0 * smooth / (2.0 - smooth), 1),
        "% of an every-step alternation (1.0 = raw policy)",
    )
    print(
        "  commanded step    = mean",
        fixed(sum_step / n_step, 1) if n_step > 0 else String("n/a"),
        "ticks, max", fixed(max_step, 1),
        "(clamp is " + String(step_ticks) + ")",
    )
    # ⚠⚠ THE CLAMP DECIDES THE MOTION WHENEVER THIS RATIO IS LARGE. Measured on
    # the first live run: mean demand 612 ticks against a clamp of 25, i.e. the
    # arm ran at 1/24 of what the policy asked for, for the whole episode, and
    # was still approaching when time ran out. `max_step_ticks` HAS NO
    # COUNTERPART IN SIM — there the commanded pose goes straight to the
    # `<position>` actuator and only its gain limits the response — so a tight
    # clamp is a sim2real gap, not just a safety margin. Read this line before
    # concluding anything about the policy.
    print(
        "  goal reversals    =",
        fixed(100.0 * reversals / n_step, 0) if n_step > 0 else String("n/a"),
        "% of writes changed DIRECTION  <- this is the shake",
    )
    var throttle = (sum_step / n_step) / Float64(step_ticks) if n_step > 0 else 0.0
    if throttle > 2.0:
        print(
            "  ⚠ RATE-LIMITED: the policy asked for",
            fixed(throttle, 1) + "x the clamp on average.",
            "\n     The CLAMP shaped this run, not the policy. Raise --step"
            " or lengthen --seconds",
            "\n     before reading the distance as the policy's fault.",
        )
    if not live:
        print("  ⚠ DRY RUN — nothing was written to the arm. Add --arm.")
    # ⚠ The error is FK-derived, not measured with a ruler: it is where the
    # model says the jaw is given the servo angles. A systematic kinematic
    # error is invisible to it. The sim task's own success radius is 20 mm.
    print("  (FK-derived, not externally measured; task radius is 20 mm)")
    print("=" * 70)
