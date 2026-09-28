# +--------------------------------------------------------------------------+ #
# | Real SO-101 joints <-> simulated SO-101 joints
# +--------------------------------------------------------------------------+ #
"""The mapping between a servo's calibrated angle and the MJCF model's joint.

**This mapping is the reference implementation's**, not a guess.
`so101-nexus`'s `lerobot_adapter/normalization.py::motor_ticks_to_sim_rad` —
the function a working LeRobot-on-MuJoCo stack uses for these arms — is:

```python
mid  = (range_min + range_max) / 2
sign = -1 if cal.drive_mode else 1
qpos = sign * (ticks - mid) / TICKS_PER_RADIAN          # body joints
frac = (ticks - range_min) / (range_max - range_min)    # gripper
qpos = lower + frac * (upper - lower)
```

which is exactly `to_sim_unclamped` below, with `offset_rad = 0` (the
reference has no offset term at all) and `sign = +1`.

⚠ **`sign = +1` is a property of the platform, not an assumption.** lerobot
HARD-CODES `drive_mode=0` for every SO-101 joint — `robots/so_follower/
so_follower.py:149`, `teleoperators/so_leader/so_leader.py:117` and
`motors/feetech/feetech.py:260` — so there is no inverted joint to discover on
this arm. An earlier version of this file claimed the identity was "almost
certainly wrong for at least one joint"; that was wrong, and reading the
reference rather than guessing is what settled it.

`sign` and `offset_rad` stay as fields because they are the knobs a DIFFERENT
arm would need (a Koch follower does use `drive_mode=1`), and because a
mis-bolted horn is a real, physical thing that no table can predict.

⚠ The gripper is not an angle on either side. The servo reports it 0..100
(`MotorNormMode.RANGE_0_100`) and the model has it as a hinge in radians, so
it maps by FRACTION OF RANGE, and `offset_rad` does not apply to it. The
reference's gripper limits — `SO101_GRIPPER_LIMITS_RAD = (-10 deg, 100 deg)` —
agree with our model's `ctrlrange` to 1e-16, which is a pleasing independent
confirmation that we are pointing at the same joint.
The fraction lands on a LINE (`grip_lo_rad`, `grip_span_rad`): the
reference's is the model's hinge range; the tower follower's is the measured
one (`tower_follower_gripper_span_rad`), because the fraction map squeezes
130.5 deg of real travel into 110.

⚠ **Range still disagrees, and that is measured** — see `range_report`. Our
MJCF ranges are byte-identical to
`references/SO-ARM100-main/Simulation/SO101/so101_new_calib.xml`, so the model
is a faithful port; three body joints simply have more CALIBRATED travel than
it accepts, and `wrist_roll` is a continuous joint meeting a bounded one.

## ⚠⚠ THE TOWER FOLLOWER'S ZERO IS NOT THE MODEL'S (`tower_follower_zero_deg`)

The reference's `offset = 0` assumes the calibrated MIDDLE OF RANGE is the
model's zero. It is not a physical landmark: `lerobot-calibrate` takes the
middle of however far each joint was swept, so it moves with every
recalibration. MEASURED on the so101-tower follower, calibration of
2026-09-14 (the one the 2026-09-15 cube-in-bowl dataset was recorded with),
in the sense `model_rad = deg2rad(lerobot_deg) + zero`:

    joint           extrinsics fit      recorded-frame fit     used
    shoulder_pan    -10.25 (+-0.7)      -11.2                  -10.7
    shoulder_lift    -3.2  (+-1.2)      -3.6 .. -4.6            -3.6
    elbow_flex       -7.3               -7.3                    -7.3
    wrist_flex       +7.6  (+-3.7)      not observable           0
    wrist_roll       +5.0 (see below)   not observable          +5.0

The two fits share no data: (1) 30 marker captures of
`calibrate_camera_extrinsics.mojo` with the camera HELD at the asset's pose
and the joint zeros free (rms 24.3 -> 12.2 mm); (2) the servos' dark pixels
of the sim drawn at the dataset's recorded state against 40 undistorted real
overhead frames (overlap 0.29 -> 0.62). The "used" column on a fresh draw of
40 frames: 0.27 -> 0.59, better than zero on 40/40. The asset camera itself was checked separately on the static
scene, so a camera yaw cannot be what the pan zero is absorbing (a yaw it
could be would move the pan zero by <= 3 deg, not to 0).
`tools/soarm/check_joint_zero.py` re-runs (2) on any dataset.

THE ROLL (2026-09-25). Once a photo pinned the extrinsics marker at the centre
of the wrist camera plate's back face (±1 mm; `bake_so_arm101_tower.py` step 8),
the marker offset stopped being a free parameter, and the roll zero became
observable: the marker sits 74 mm off the roll axis, where 1 deg is 1.3 mm
sideways. With the camera free and the marker fixed, the two independent
captures fit +4.6 deg (2026-09-22, 30 poses) and +5.9 deg (2026-09-25, 32
poses), +5.0 deg together (rms 10.0 -> 9.4 mm). wrist_flex, freed with it,
disagrees between the captures (+5.7 / +3.7) and stays 0. The free-offset fit
had been putting the marker 6-7 mm sideways instead, in both captures.

⚠ Tied to THAT calibration: a `lerobot-calibrate` re-run moves every mid and
voids these numbers. `tower_follower_zero_matches` compares the live mids
with the ones measured against, and a tick-level caller refuses on mismatch;
a caller in LeRobot degrees (a dataset, a rendered store) cannot see the
calibration and must carry the choice itself.
"""

from std.math import pi

from noeira.robot.so101.arm import GRIPPER, SO101Calibration, SO101_N, joint_name
from noeira.utils.fmt import col, fixed, pad_left, pad_right


# Functions, not `comptime` arrays: a comptime `Array` is not
# `ImplicitlyCopyable` and cannot be materialised at runtime.
def tower_follower_zero_deg(i: Int) -> Float64:
    """The so101-tower follower's measured zero, degrees, joint `i` (the
    gripper 0: it is fraction-mapped). See the module docstring."""
    var v: List[Float64] = [-10.7, -3.6, -7.3, 0.0, 5.0, 0.0]
    return v[i]


comptime TOWER_FOLLOWER_GRIPPER_ZERO_RAD: Float64 = -0.16147
"""The tower follower's gripper hinge angle at LeRobot 0 (its calibrated
minimum), model radians — the gripper's own line, see
`tower_follower_gripper_span_rad`."""
comptime TOWER_FOLLOWER_GRIPPER_SPAN_TICKS: Int = 1484
"""The gripper's calibrated span in the calibration the map was measured
against (`follower.json`, 2031..3515)."""


def tower_follower_gripper_span_rad() -> Float64:
    """The angle the gripper turns over its calibrated span: the ticks at
    the arm's own 360/4095 degrees per tick — 130.46 deg, NOT the model's
    110 deg ctrlrange.

    ⚠⚠ THE REFERENCE MAPS THE GRIPPER BY FRACTION OF RANGE onto the model's
    hinge range, which squeezes the real 130.5 deg of travel into 110: every
    percent of real opening became 16% too little model angle. Measured
    2026-09-25 against the recordings' grasp plateaus: the printed 25 mm cube
    at 13.1 (n 70) and the ~31.8 mm Duplo at 17.3 (n 53) put the sim's tip
    boxes 21.3 and 26.3 mm apart through the fraction map — an error that
    GROWS with the width, so a scale, not pad thickness or an offset. The
    physical line with ONE fitted offset (`TOWER_FOLLOWER_GRIPPER_ZERO_RAD`,
    the 25 mm cube at the sim's 25.0 mm gap) puts the Duplo at 30.9 mm.
    Closed on nothing (0.6) the sim still shows 4.5 mm: the tip boxes cannot
    close below 2.4 mm even at the joint limit."""
    return Float64(TOWER_FOLLOWER_GRIPPER_SPAN_TICKS) * 2.0 * pi / 4095.0


def tower_follower_calib_mid(i: Int) -> Float64:
    """The calibrated mid (ticks) of each joint in the calibration the zero was
    measured against — `projects/so101-tower/calibration/follower.json`,
    2026-09-14."""
    var v: List[Float64] = [1987.0, 2064.0, 1896.5, 2079.0, 2047.0, 2773.0]
    return v[i]


def tower_follower_zero_matches(cal: SO101Calibration) -> Bool:
    """True when `cal` is the calibration `tower_follower_zero_deg` was
    measured against (every body joint's mid within half a tick)."""
    for i in range(SO101_N):
        if i == GRIPPER:
            continue
        if abs(cal.mid(i) - tower_follower_calib_mid(i)) > 0.5:
            return False
    return True


@fieldwise_init
struct SimJointMap(Copyable, Movable):
    """Per-joint sign, zero offset and the model's own limits.

    `sim_lo` / `sim_hi` come from the MODEL (a `<position>` servo's
    `ctrlrange` is its joint range), never from a copy of the numbers — a
    second copy of a limit is a second thing to drift.
    """

    var sign: Array[Float64, SO101_N]
    var offset_rad: Array[Float64, SO101_N]
    var sim_lo: Array[Float64, SO101_N]
    var sim_hi: Array[Float64, SO101_N]
    var grip_lo_rad: Float64
    """The gripper's hinge angle at its calibrated minimum (fraction 0)."""
    var grip_span_rad: Float64
    """The angle over its calibrated span. The reference: `sim_lo` and
    `sim_hi - sim_lo` (the fraction map); `tower_follower`: the measured
    line."""

    @staticmethod
    def identity(
        var sim_lo: Array[Float64, SO101_N],
        var sim_hi: Array[Float64, SO101_N],
    ) -> Self:
        """The SO-101 mapping: every sign +1, every offset 0.

        Not a placeholder — this IS `motor_ticks_to_sim_rad` for
        `drive_mode=0`, which lerobot hard-codes for this arm. See the module
        docstring.
        """
        var s = Array[Float64, SO101_N](fill=1.0)
        var o = Array[Float64, SO101_N](fill=0.0)
        var glo = sim_lo[GRIPPER]
        var gspan = sim_hi[GRIPPER] - sim_lo[GRIPPER]
        return Self(s^, o^, sim_lo^, sim_hi^, glo, gspan)

    @staticmethod
    def tower_follower(
        cal: SO101Calibration,
        var sim_lo: Array[Float64, SO101_N],
        var sim_hi: Array[Float64, SO101_N],
    ) raises -> Self:
        """The reference mapping plus the tower follower's measured zero.

        Raises when `cal` is not the calibration the zero was measured
        against: after a recalibration the numbers are someone else's arm."""
        if not tower_follower_zero_matches(cal):
            var got = String("")
            for i in range(SO101_N):
                got += " " + String(cal.mid(i))
            raise Error(
                "SimJointMap.tower_follower: this arm's calibrated mids"
                + got + " are not the ones the zero was measured against"
                " (sim_map.tower_follower_calib_mid) — it was recalibrated;"
                " re-measure the zero (tools/soarm/check_joint_zero.py)"
            )
        if cal.span(GRIPPER) != TOWER_FOLLOWER_GRIPPER_SPAN_TICKS:
            raise Error(
                "SimJointMap.tower_follower: the gripper spans "
                + String(cal.span(GRIPPER)) + " ticks, the map was measured on "
                + String(TOWER_FOLLOWER_GRIPPER_SPAN_TICKS)
            )
        var m = Self.identity(sim_lo^, sim_hi^)
        for i in range(SO101_N):
            if i != GRIPPER:
                m.offset_rad[i] = tower_follower_zero_deg(i) * pi / 180.0
        m.grip_lo_rad = TOWER_FOLLOWER_GRIPPER_ZERO_RAD
        m.grip_span_rad = tower_follower_gripper_span_rad()
        return m^

    def differs_from_lerobot(self) -> Bool:
        """True once someone has moved a sign or an offset off the reference.

        Named for what it means. The previous name, `measured()`, said the
        opposite of the truth: the default is not an unmeasured guess, it is
        the reference implementation's mapping, and a `True` here means we
        have DEPARTED from it — which is the thing worth announcing.
        """
        for i in range(SO101_N):
            if self.sign[i] != 1.0 or self.offset_rad[i] != 0.0:
                return True
        return (
            self.grip_lo_rad != self.sim_lo[GRIPPER]
            or self.grip_span_rad != self.sim_hi[GRIPPER] - self.sim_lo[GRIPPER]
        )

    # ── real -> sim ────────────────────────────────────────────────────────

    def to_sim_unclamped(
        self, cal: SO101Calibration, i: Int, raw: Int32
    ) -> Float64:
        """Servo ticks to model radians, BEFORE the model's limits apply.

        Unclamped on purpose: `clamped_by` needs to see how far outside the
        model's range the real arm actually went, and a function that clamps
        silently cannot answer that.
        """
        if i == GRIPPER:
            # Fraction of the servo's calibrated opening, mapped onto the
            # model's hinge range. `sign` flips the fraction; `offset_rad`
            # does not apply — this is not an angle on the servo side.
            #
            # ⚠⚠ THE FRACTION IS COMPUTED HERE, NOT VIA `cal.degrees`, WHICH
            # CLAMPS. `degrees()` opens with `min(hi, max(lo, raw))` for the
            # gripper, so this function — whose entire contract is to be
            # unclamped, because `clamped_by` cannot report an overshoot it
            # cannot see — was silently clamped for exactly one joint. A
            # gripper parked below its CALIBRATED minimum then reported zero
            # overshoot while `to_sim`/`from_sim` disagreed by the whole
            # shortfall, which `deploy_reach_real.mojo`'s round-trip check
            # reported as a sign error in the mapping. Measured: 11 ticks.
            var lo_t = Float64(cal.range_min[i])
            var hi_t = Float64(cal.range_max[i])
            var span_t = hi_t - lo_t
            var frac = (Float64(raw) - lo_t) / span_t if span_t != 0.0 else 0.0
            if self.sign[i] < 0.0:
                frac = 1.0 - frac
            return self.grip_lo_rad + frac * self.grip_span_rad
        return self.sign[i] * cal.radians(i, raw) + self.offset_rad[i]

    def to_sim(self, cal: SO101Calibration, i: Int, raw: Int32) -> Float64:
        var v = self.to_sim_unclamped(cal, i, raw)
        return min(self.sim_hi[i], max(self.sim_lo[i], v))

    def from_sim(
        self, cal: SO101Calibration, i: Int, value: Float64
    ) -> Int32:
        """Model radians back to servo ticks — the inverse of `to_sim`.

        What a POLICY's action has to go through to reach the hardware: the
        net was trained in the model's joint space, and the bus speaks ticks.
        Exact inverse of `to_sim_unclamped`, gripper fraction included, so a
        round trip through both is the identity up to tick quantisation.
        """
        if i == GRIPPER:
            var span = self.grip_span_rad
            var frac = (value - self.grip_lo_rad) / span if span != 0.0 else 0.0
            if self.sign[i] < 0.0:
                frac = 1.0 - frac
            return cal.raw_from_degrees(i, frac * 100.0)
        return cal.raw_from_radians(i, (value - self.offset_rad[i]) / self.sign[i])

    def clamped_by(
        self, cal: SO101Calibration, i: Int, raw: Int32
    ) -> Float64:
        """Radians of overshoot past the model's limit, 0 when inside.

        A teleop loop reports this rather than hiding it: a joint pinned at
        its simulated limit while the real one keeps moving looks exactly like
        a broken mapping, and this is what tells the two apart.
        """
        var v = self.to_sim_unclamped(cal, i, raw)
        if v > self.sim_hi[i]:
            return v - self.sim_hi[i]
        if v < self.sim_lo[i]:
            return self.sim_lo[i] - v
        return 0.0

    # ── reporting ──────────────────────────────────────────────────────────

    def range_report(self, cal: SO101Calibration) -> String:
        """Calibrated servo travel vs what the model will accept, per joint.

        Measured from the two sources rather than transcribed, so it stays
        true when either side changes. A positive `gap` is real travel the
        simulation cannot represent — the leader can reach a pose the model
        refuses, and the sim joint will sit at its limit.
        """
        var out = String(
            pad_right(String("JOINT"), 15)
            + pad_left(String("real span"), 11)
            + pad_left(String("sim span"), 11)
            + pad_left(String("gap"), 9)
            + "\n"
        )
        out += "-" * 46 + "\n"
        for i in range(SO101_N):
            var sim_span = (
                self.sim_hi[i] - self.sim_lo[i]
            ) * 180.0 / 3.141592653589793

            if cal.is_unlimited(i):
                # ⚠ NOT a gap. `0..4095` is lerobot's unlimited marker, so
                # there is no measured travel to compare against — the joint
                # turns freely and the model bounds it. Reporting a number
                # here would invent one.
                out += (
                    pad_right(joint_name(i), 15)
                    + pad_left(String("free"), 11)
                    + col(sim_span, 11, 1)
                    + pad_left(String("n/a"), 9)
                    + "   <-- CONTINUOUS joint, bounded model\n"
                )
                continue

            var real_span = Float64(cal.span(i)) * 360.0 / 4095.0
            if i == GRIPPER:
                # Both sides are rescaled onto each other by construction, so
                # a "gap" here would be meaningless rather than zero.
                out += (
                    pad_right(joint_name(i), 15)
                    + col(real_span, 11, 1)
                    + col(sim_span, 11, 1)
                    + pad_left(String("n/a"), 9)
                    + "   (fraction-mapped)\n"
                )
                continue

            var gap = real_span - sim_span
            out += (
                pad_right(joint_name(i), 15)
                + col(real_span, 11, 1)
                + col(sim_span, 11, 1)
                + col(gap, 9, 1)
                + ("   <-- real exceeds sim" if gap > 1.0 else "")
                + "\n"
            )
        return out^

    def describe(self) -> String:
        if not self.differs_from_lerobot():
            return String(
                "sim map: lerobot reference (drive_mode=0 => sign +1, no"
                " offset) — matches so101-nexus motor_ticks_to_sim_rad"
            )
        var out = String("sim map:")
        for i in range(SO101_N):
            out += (
                " "
                + String(joint_name(i)[byte=0:4])
                + ("+" if self.sign[i] > 0.0 else "-")
                + fixed(self.offset_rad[i], 2)
            )
        return out^
