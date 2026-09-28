# +--------------------------------------------------------------------------+ #
# | SO-ARM101 — six STS3215 on one bus
# +--------------------------------------------------------------------------+ #
"""`SO101Arm`: the leader or the follower, driven over `FeetechBus`.

A leader and a follower are the same hardware and the same code; the only
difference is that nobody writes goals to a leader. So there is one struct
here, not two.

**Calibration comes off the servos, not off disk.** `lerobot-calibrate` writes
`Homing_Offset`, `Min_Position_Limit` and `Max_Position_Limit` into each
servo's EEPROM (`feetech.py:268 write_calibration`), so an arm carries its own
calibration and this layer needs no JSON parser — which matters, because the
repo has none and hand-rolling one for a robot driver would be absurd.

⚠ **Units follow lerobot exactly**, because a policy trained in sim has to
speak the same numbers a LeRobot dataset recorded (`so101-nexus`'s
`lerobot_adapter/normalization.py` is the same table):

* body joints — `MotorNormMode.DEGREES`:
  `deg = (raw - mid) * 360 / (4096 - 1)`, `mid = (range_min + range_max) / 2`,
  **unclamped**;
* gripper — `MotorNormMode.RANGE_0_100`:
  `pct = (clamp(raw) - min) / (max - min) * 100`.

⚠⚠ **AND `DEGREES` IS A RECORDING-TIME CHOICE, NOT A FACT.** lerobot's
`so_follower.py` selects `DEGREES if config.use_degrees else RANGE_M100_100`,
with `use_degrees: bool = True` as the default. Both write `*.pos` columns and
neither records which was used, so a dataset is the only witness — and the tell
is CLAMPING, since `RANGE_M100_100` cannot leave [-100, 100]. `range_m100_100`
below exists so that mode is reachable BY NAME rather than by accident.

Note the `4096 - 1`: the inclusive tick range 0..4095 spans one turn. Using
4096 is a systematic ~0.09 degree error, small enough to survive a review and
large enough to sit in every sim-to-real comparison afterwards.
"""

from std.math import pi

from noeira.robot.feetech.bus import FeetechBus
from noeira.robot.feetech.control_table import (
    MODE_POSITION,
    SIZE_1,
    SIZE_2,
    STS_GOAL_POSITION,
    STS_HOMING_OFFSET,
    STS_LOCK,
    STS_MAX_POSITION_LIMIT,
    STS_MIN_POSITION_LIMIT,
    STS_ACCELERATION,
    STS_OPERATING_MODE,
    STS_PRESENT_LOAD,
    STS_PRESENT_POSITION,
    STS_PRESENT_VELOCITY,
    STS_RESOLUTION,
    STS_TORQUE_ENABLE,
    TORQUE_DISABLED,
    TORQUE_ENABLED,
)

comptime SO101_N = 6
"""Joint count: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex,
wrist_roll, gripper —
servo ids 1..6 in that order, as `lerobot-setup-motors` assigns them."""

comptime GRIPPER = 5
"""Index of the gripper, the one joint normalised 0..100 instead of degrees."""

comptime TICKS_PER_TURN = STS_RESOLUTION - 1
"""4095. See the module docstring — NOT 4096."""


def joint_name(i: Int) -> String:
    if i == 0:
        return String("shoulder_pan")
    if i == 1:
        return String("shoulder_lift")
    if i == 2:
        return String("elbow_flex")
    if i == 3:
        return String("wrist_flex")
    if i == 4:
        return String("wrist_roll")
    return String("gripper")


def joint_short(i: Int) -> String:
    """A 4-character label that is UNIQUE per joint.

    ⚠ TRUNCATING `joint_name` TO 4 CHARS DOES NOT WORK: `shoulder_pan` and
    `shoulder_lift` both become `shou`, and `wrist_flex` and `wrist_roll` both
    become `wris`. A live telemetry line then shows two identical labels with
    different numbers, which is worse than no label — it invites reading the
    wrong joint. Seen in a real recording session on 2026-08-31.
    """
    if i == 0:
        return String("pan ")
    if i == 1:
        return String("lift")
    if i == 2:
        return String("elbo")
    if i == 3:
        return String("wfle")
    if i == 4:
        return String("wrol")
    return String("grip")


@fieldwise_init
struct SO101Calibration(Copyable, Movable):
    """What `lerobot-calibrate` left in the servos' EEPROM."""

    var homing_offset: Array[Int32, SO101_N]
    var range_min: Array[Int32, SO101_N]
    var range_max: Array[Int32, SO101_N]

    def mid(self, i: Int) -> Float64:
        return 0.5 * (Float64(self.range_min[i]) + Float64(self.range_max[i]))

    def span(self, i: Int) -> Int:
        return Int(self.range_max[i]) - Int(self.range_min[i])

    def is_unlimited(self, i: Int) -> Bool:
        """True when this joint turns freely and was never swept to end stops.

        ⚠ `range_min == 0 and range_max == 4095` is lerobot's **unlimited
        marker**, not a measurement. `wrist_roll` carries it on both arms here,
        and `tools/soarm/so101_pairing.py` prints "full turn" for exactly that
        reason.

        Reading it as a calibrated span is a category error with a plausible
        answer — it yields "360 degrees of travel", which then looks like the
        joint over-travels its simulated limit by 40 degrees. It does not
        over-travel; it is a continuous joint meeting a bounded model, which
        is a different problem with a different fix.
        """
        return (
            self.range_min[i] == 0 and self.range_max[i] == STS_RESOLUTION - 1
        )

    def degrees(self, i: Int, raw: Int32) -> Float64:
        """Ticks to the units lerobot records — degrees, or 0..100 for the
        gripper."""
        if i == GRIPPER:
            var lo = Int(self.range_min[i])
            var hi = Int(self.range_max[i])
            var v = min(hi, max(lo, Int(raw)))
            return Float64(v - lo) / Float64(hi - lo) * 100.0
        return (Float64(raw) - self.mid(i)) * 360.0 / Float64(TICKS_PER_TURN)

    def radians(self, i: Int, raw: Int32) -> Float64:
        """Body joints in radians; the gripper stays 0..100 (it is an opening
        fraction, not an angle, and pretending otherwise would put a unit
        error into every observation)."""
        if i == GRIPPER:
            return self.degrees(i, raw)
        return self.degrees(i, raw) * pi / 180.0

    def range_m100_100(self, i: Int, raw: Int32) -> Float64:
        """`lerobot`'s OTHER body-joint mode: percent of calibrated travel.

        ⚠⚠ WHICH MODE A DATASET USED IS A RECORDING-TIME FLAG, NOT A PROPERTY
        OF THIS ARM. `so_follower.py` picks
        `MotorNormMode.DEGREES if config.use_degrees else RANGE_M100_100`, and
        `use_degrees` defaults to True — so `degrees()` is the usual answer and
        this exists so the other one cannot be reached by accident, only by
        name.

        ⚠ THE ONLY WITNESS IS THE DATA. Both modes write `*.pos` columns with
        no unit recorded anywhere. They are told apart by CLAMPING: this one is
        exactly [-100, 100] by construction, `degrees()` is unbounded. The
        50-demo store has `shoulder_lift` at -107.16 and `wrist_flex` at
        +102.29, which THIS FUNCTION CANNOT PRODUCE — that is the proof it was
        recorded in degrees. `tools/act/dump_lerobot_units_reference.py
        --check-dataset` re-runs that test on any store.
        """
        var lo = Float64(self.range_min[i])
        var hi = Float64(self.range_max[i])
        var v = min(hi, max(lo, Float64(raw)))
        return ((v - lo) / (hi - lo)) * 200.0 - 100.0

    def raw_from_range_m100_100(self, i: Int, value: Float64) -> Int32:
        var lo = Float64(self.range_min[i])
        var hi = Float64(self.range_max[i])
        var v = min(100.0, max(-100.0, value))
        return Int32(Int(((v + 100.0) / 200.0) * (hi - lo) + lo))

    def raw_from_degrees(self, i: Int, value: Float64) -> Int32:
        if i == GRIPPER:
            var lo = Float64(self.range_min[i])
            var hi = Float64(self.range_max[i])
            var pct = min(100.0, max(0.0, value))
            return Int32(Int(pct / 100.0 * (hi - lo) + lo))
        return Int32(Int(value * Float64(TICKS_PER_TURN) / 360.0 + self.mid(i)))

    def raw_from_radians(self, i: Int, value: Float64) -> Int32:
        if i == GRIPPER:
            return self.raw_from_degrees(i, value)
        return self.raw_from_degrees(i, value * 180.0 / pi)


comptime ALIGN_TICKS = 57
"""~5 degrees. Every joint within this of its goal ends the catch-up phase."""

comptime LEROBOT_ACCELERATION = 254
"""What lerobot writes to `Acceleration` on every connect
(`FeetechMotorsBus.configure_motors`). A RAM register: it reads 0 after a power
cycle, so writing it once at calibration time is not enough."""


def step_limit(tracking: Bool, catch_up_ticks: Int, track_ticks: Int) -> Int:
    """The per-write clamp for the current phase. 0 means no clamp."""
    if tracking and track_ticks > 0:
        return track_ticks
    return catch_up_ticks


def is_aligned(
    ref goals: Array[Int32, SO101_N],
    ref present: Array[Int32, SO101_N],
    align_ticks: Int,
) -> Bool:
    """Every joint within `align_ticks` of its goal."""
    for i in range(SO101_N):
        var d = Int(goals[i]) - Int(present[i])
        if d > align_ticks or d < -align_ticks:
            return False
    return True


struct SO101Arm(Movable):
    var bus: FeetechBus
    var cal: SO101Calibration
    var ids: Array[UInt8, SO101_N]
    var max_step_ticks: Int
    """Largest change from the CURRENT position a single `write_goals` may
    command, per joint.

    ⚠ lerobot's own config for these arms records `max_relative_target: None`
    — no clamp at all — so the first bad goal a policy emits is a full-speed
    slam into the table. 200 ticks is ~17 degrees. Set it to 0 to disable the
    clamp deliberately; do not leave it off by accident.

    ⚠ WITH `track_step_ticks` SET, THIS IS ONLY THE CATCH-UP LIMIT. See below.
    """
    var track_step_ticks: Int
    """The clamp once the follower has CAUGHT UP. 0 (default) = no second phase.

    ⚠⚠ ONE SMALL CLAMP IS A SPEED LIMIT. The servo's speed is proportional to
    how far its goal leads its position, so capping that lead at 80 ticks (7
    deg) capped the follower at ~1.4 deg per 30 Hz tick: recorded demos lagged
    the leader by 300 ms on shoulder_lift, 50.9 deg at worst (2026-09-15,
    trial-01). The small clamp is only needed while the follower closes a
    LARGE gap — just after torque on — so a lunge becomes a ramp. Once every
    joint is within `ALIGN_TICKS`, this larger limit applies until torque is
    next turned on. It still bounds a bad reading or a glitched goal.
    """
    var tracking: Bool
    """True once caught up; reset by `set_torque(True)`."""

    def __init__(
        out self,
        var path: String,
        baud: Int = 1000000,
        max_step_ticks: Int = 200,
        track_step_ticks: Int = 0,
    ) raises:
        self.bus = FeetechBus(path^, baud)
        self.max_step_ticks = max_step_ticks
        self.track_step_ticks = track_step_ticks
        self.tracking = False
        self.ids = Array[UInt8, SO101_N](fill=0)
        for i in range(SO101_N):
            self.ids[i] = UInt8(i + 1)
        self.cal = SO101Calibration(
            Array[Int32, SO101_N](fill=0),
            Array[Int32, SO101_N](fill=0),
            Array[Int32, SO101_N](fill=0),
        )

        # Ping every servo before reading anything: a missing motor otherwise
        # surfaces as a confusing timeout inside calibration.
        for i in range(SO101_N):
            if not self.bus.ping(self.ids[i]):
                raise Error(
                    "so101: no servo answered id "
                    + String(i + 1)
                    + " ("
                    + joint_name(i)
                    + ") — check power and the daisy chain"
                )
        self.read_calibration()

    def read_calibration(mut self) raises:
        """Pull `Homing_Offset` / `Min` / `Max` out of each servo's EEPROM."""
        for i in range(SO101_N):
            var id = self.ids[i]
            self.cal.homing_offset[i] = Int32(
                self.bus.read_register(id, STS_HOMING_OFFSET, SIZE_2)
            )
            self.cal.range_min[i] = Int32(
                self.bus.read_register(id, STS_MIN_POSITION_LIMIT, SIZE_2)
            )
            self.cal.range_max[i] = Int32(
                self.bus.read_register(id, STS_MAX_POSITION_LIMIT, SIZE_2)
            )
            if self.cal.span(i) == 0:
                raise Error(
                    "so101: "
                    + joint_name(i)
                    + " has range_min == range_max ("
                    + String(Int(self.cal.range_min[i]))
                    + "), which is the UNCALIBRATED marker — run"
                    " `lerobot-calibrate` for this arm"
                )

    # ── reading ────────────────────────────────────────────────────────────

    def read_positions[
        o: MutOrigin
    ](mut self, out_raw: Span[Int32, o]) raises -> Int:
        """All six present positions, in ticks, in one round trip.

        Returns how many answered. **Check it** — a partial read means a motor
        dropped off the bus, and treating the untouched entries as current is
        how a teleop loop commands last second's pose.
        """
        return self.bus.sync_read(
            STS_PRESENT_POSITION, SIZE_2, Span(self.ids), out_raw
        )

    def read_velocities[
        o: MutOrigin
    ](mut self, out_raw: Span[Int32, o]) raises -> Int:
        """All six present velocities, in TICKS PER SECOND, in one round trip.

        ⚠ ONE ROUND TRIP IS THE POINT. A control loop that wants qvel was
        calling `read_register` six times — six request/response pairs at
        ~1.3 ms each, which is 8 ms of the loop's budget spent on a quantity a
        single `sync_read` returns. `deploy_reach_real.mojo` could not hold
        the 50 Hz its policy trained at because of exactly that.

        Sign-magnitude at bit 15, decoded by `sync_read` through
        `sign_bit_for` — the same path `read_positions` takes for bit 15 of
        `Present_Position`, so the two share their decoding rather than
        restating it.
        """
        return self.bus.sync_read(
            STS_PRESENT_VELOCITY, SIZE_2, Span(self.ids), out_raw
        )

    def read_loads[
        o: MutOrigin
    ](mut self, out_raw: Span[Int32, o]) raises -> Int:
        """All six present loads in one round trip: signed, in 0.1 % of the
        servo's maximum torque (1000 = full), sign-magnitude at bit 10,
        decoded by `sync_read` like the velocities. The sign is the direction
        the servo pushes.
        """
        return self.bus.sync_read(
            STS_PRESENT_LOAD, SIZE_2, Span(self.ids), out_raw
        )

    # ── writing ────────────────────────────────────────────────────────────

    def set_torque(mut self, on: Bool) raises:
        # ⚠ Every engage starts in CATCH-UP: the leader may be anywhere.
        self.tracking = False
        var v = TORQUE_ENABLED if on else TORQUE_DISABLED
        for i in range(SO101_N):
            self.bus.write_register(self.ids[i], STS_TORQUE_ENABLE, v, SIZE_1)
            if not on:
                # `Lock` guards the EEPROM; lerobot clears it alongside torque
                # (`feetech.py:291 disable_torque`) so a subsequent
                # calibration write is not silently dropped.
                self.bus.write_register(self.ids[i], STS_LOCK, 0, SIZE_1)

    def set_position_mode(mut self) raises:
        """Position mode, with lerobot's on-connect `Acceleration`.

        ⚠ `Acceleration` read 0 on every joint of the 2026-09 follower while
        lerobot's gains (P 16, D 32, I 0) were still in EEPROM: lerobot writes
        254 on EVERY connect because the register does not survive a power
        cycle. Every arming path here calls this first, so they all match.
        """
        for i in range(SO101_N):
            self.bus.write_register(
                self.ids[i], STS_OPERATING_MODE, MODE_POSITION, SIZE_1
            )
            self.bus.write_register(
                self.ids[i], STS_ACCELERATION, LEROBOT_ACCELERATION, SIZE_1
            )

    def write_goals[
        mut: Bool, //, og: Origin[mut=mut]
    ](mut self, goals: Span[Int32, og]) raises:
        """Command all six goal positions in ONE packet, clamped twice.

        Clamped to each joint's calibrated `[range_min, range_max]`, and — if
        `max_step_ticks > 0` — to `present ± max_step_ticks`, which costs one
        extra `sync_read` (~1.3 ms of a ~20 ms tick) and is the difference
        between a bad goal being a jerk and being a slam.
        """
        if len(goals) != SO101_N:
            raise Error(
                "so101: write_goals expects "
                + String(SO101_N)
                + " goals, got "
                + String(len(goals))
            )

        var safe = Array[Int32, SO101_N](fill=0)
        for i in range(SO101_N):
            var lo = Int(self.cal.range_min[i])
            var hi = Int(self.cal.range_max[i])
            safe[i] = Int32(min(hi, max(lo, Int(goals[i]))))

        if self.max_step_ticks > 0 or self.track_step_ticks > 0:
            var present = Array[Int32, SO101_N](fill=0)
            var got = self.read_positions(Span(present))
            if got != SO101_N:
                raise Error(
                    "so101: refusing to write goals — only "
                    + String(got)
                    + " of "
                    + String(SO101_N)
                    + " motors reported a position, so the step clamp cannot"
                    " be applied"
                )
            if not self.tracking and self.track_step_ticks > 0:
                self.tracking = is_aligned(safe, present, ALIGN_TICKS)
            var limit = step_limit(
                self.tracking, self.max_step_ticks, self.track_step_ticks
            )
            if limit > 0:
                for i in range(SO101_N):
                    var p = Int(present[i])
                    var step = Int(safe[i]) - p
                    if step > limit:
                        safe[i] = Int32(p + limit)
                    elif step < -limit:
                        safe[i] = Int32(p - limit)

        self.bus.sync_write(
            STS_GOAL_POSITION, SIZE_2, Span(self.ids), Span(safe)
        )
