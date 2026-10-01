# +--------------------------------------------------------------------------+ #
# | Ending a run without dropping the arm — the shutdown every deployment uses
# +--------------------------------------------------------------------------+ #
"""`return_and_release`: ramp the follower home, confirm, hold, then release.

⚠⚠ **THIS MODULE EXISTS BECAUSE THE ARM FELL, AND IT IS SHARED BECAUSE A
SECOND POLICY WOULD OTHERWISE COPY IT.** It was written inside
`act_so101_deploy_real.mojo`; the SmolVLA deployment needs exactly the same
shutdown, and safety code that exists twice is safety code that drifts —
`_a_rule_written_inline_twice_drifts` is the most frequently recurring defect
shape in this tree, and this is the worst possible place for it. Every
deployment that energises the follower calls THIS.

The rule it encodes: releasing torque is not neutral. It is the moment gravity
takes over, and where the arm IS at that instant decides whether that is safe.
The pose the run STARTED from is the one pose known to be safe — the arm was
resting there, unpowered, before anything was armed.

⚠ WHAT HAPPENS ON FAILURE IS THE POINT: if the ramp does not arrive, torque is
LEFT ON and the caller is told. A still-energised arm is recoverable with
`pixi run soarm-torque-off`; a fall is not.
"""

from std.time import perf_counter_ns

from noeira.io.fileio import StdinReader
from noeira.robot.so101 import SO101Arm, SO101_N, joint_name
from noeira.utils.fmt import col


comptime RETURN_STEP_TICKS = 20
"""Per-write slew bound for the RETURN, at 30 Hz: ~52 deg/s.

Deliberately a quarter of `MAX_STEP_TICKS`. The return runs with no inference
in the loop, so it writes ~3.5x more often than the policy did; keeping the
same per-write clamp would make the way home three times faster than anything
the run itself did, which is the wrong direction for a move that happens while
someone is reaching for the arm."""

comptime RETURN_TOLERANCE_TICKS = 25
"""~2 degrees. Close enough to call it home — the servo settles inside its own
deadband and demanding better would spin until the timeout every time."""

comptime RETURN_TIMEOUT_S = 8
"""⚠ AND WHAT HAPPENS AT THE TIMEOUT IS THE POINT: torque is LEFT ON. An arm
that did not reach a pose it is known to rest in is an arm that must not be
released."""


def _spin_until(deadline_ns: Int):
    """Spin. Measured better than `usleep` on this box — see `teleop.mojo`."""
    while perf_counter_ns() < deadline_ns:
        pass


def return_and_release(
    mut arm: SO101Arm,
    ref start: List[Int32],
    armed: Bool,
    do_return: Bool,
    mut stdin: StdinReader,
    interactive: Bool,
    timeout_s: Int = RETURN_TIMEOUT_S,
) -> Bool:
    """Bring the follower home, hold, and only then release. True if released.

    `timeout_s`: how long the way home may take (RETURN_TIMEOUT_S by
    default). A caller whose arm may end with a jaw dragging on the desk —
    the pixel deploy after a dive stop — gives it longer: the friction slows
    the slew-bounded return past 8 s without blocking it.

    ⚠⚠ **THIS EXISTS BECAUSE THE ARM FELL.** The first armed run ended by
    cutting torque wherever the policy happened to leave the arm — extended,
    mid-reach — and it dropped under its own weight. Releasing torque is not a
    neutral act: it is the moment gravity takes over, and where the arm IS at
    that moment decides whether that is safe.

    The pose the run STARTED from is the one pose known to be safe, because the
    arm was already resting there, unpowered, before anything was armed. So the
    shutdown goes: ramp back to it under the step clamp, confirm it arrived,
    hold there, and release only on the operator's word.

    ⚠ IF THE RAMP DOES NOT ARRIVE, TORQUE STAYS ON. That is the whole reason
    the arrival is checked rather than assumed. Torque surviving this process
    is recoverable — `pixi run soarm-torque-off` — and a fall is not.
    """
    if not armed:
        # Nothing was energised. The unconditional release still costs one
        # packet and is the net under every path that could have armed.
        try:
            arm.set_torque(False)
        except:
            pass
        return True

    if do_return:
        print("")
        print(
            "returning to the pose the run started from (<= "
            + String(timeout_s) + " s) ..."
        )
        var hold = arm.max_step_ticks
        arm.max_step_ticks = RETURN_STEP_TICKS
        # ⚠ The return is a deliberate slow move: the tracking phase would
        # otherwise lift its clamp to TRACK_STEP_TICKS the moment it engaged.
        var hold_track = arm.track_step_ticks
        arm.track_step_ticks = 0
        var goals = Array[Int32, SO101_N](fill=0)
        for i in range(SO101_N):
            goals[i] = start[i]
        var present = Array[Int32, SO101_N](fill=0)
        var period = 1_000_000_000 // 30
        var t_end = perf_counter_ns() + timeout_s * 1_000_000_000
        var arrived = False
        var worst = 1 << 30
        while perf_counter_ns() < t_end:
            var t0 = perf_counter_ns()
            try:
                arm.write_goals(Span(goals))
            except:
                # The bus refused. Stop pushing and do NOT release — an arm
                # we can no longer command is the last thing to let go of.
                break
            try:
                if arm.read_positions(Span(present)) == SO101_N:
                    worst = 0
                    for i in range(SO101_N):
                        var d = Int(present[i]) - Int(start[i])
                        if d < 0:
                            d = -d
                        if d > worst:
                            worst = d
                    if worst <= RETURN_TOLERANCE_TICKS:
                        arrived = True
                        break
            except:
                break
            _spin_until(t0 + period)
        arm.max_step_ticks = hold
        arm.track_step_ticks = hold_track
        if not arrived:
            print(
                "⚠⚠ DID NOT REACH THE START POSE (worst joint still "
                + String(worst) + " ticks away)."
            )
            print(
                "   TORQUE IS LEFT ON deliberately — releasing an arm that is"
                " not where it can rest\n   is how it falls. Support the arm,"
                " then run `pixi run soarm-torque-off`."
            )
            return False
        print("   home, worst joint " + String(worst) + " ticks off")

    if interactive:
        print("")
        print(
            "the follower is HOLDING. Take hold of the arm if you want to"
            " move it,\nthen press Enter to release torque."
        )
        stdin.discard_pending()
        try:
            _ = stdin.line()
        except:
            pass
    try:
        arm.set_torque(False)
        print("follower torque OFF")
    except:
        print(
            "⚠ COULD NOT RELEASE FOLLOWER TORQUE — run"
            " `pixi run soarm-torque-off`"
        )
        return False
    return True
