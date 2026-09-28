# +--------------------------------------------------------------------------+ #
# | Real SO-101 joints -> simulated SO-101 joints
# +--------------------------------------------------------------------------+ #
"""Gate for `noeira/robot/so101/sim_map.mojo`, with no arm on the desk.

The calibration below is the REAL one, read off this follower's servo EEPROM
on 2026-08-25 (`pixi run soarm-diag`), so the numbers under test are the ones
the hardware actually produces — including the negative sign-magnitude homing
offsets. The simulated limits come from the MODEL (`actuator_column` over
`ACT_IDX_CTRL_MIN/MAX`; a `<position>` servo's ctrlrange IS its joint range),
never from a transcription — so if either side moves, this test moves with it
rather than quietly testing a stale copy.

⚠ **This gates the CONVENTIONS, not the mapping.** Sign and zero are still
unmeasured (`SimJointMap.identity`), and no test can discover them — that
needs a human moving the real arm against
`examples/so101/teleop_sim.mojo`. What is gated here is that the plumbing
around them is right: mid-range maps to zero, the gripper maps by fraction
rather than as an angle, a sign flip actually inverts, and the range
disagreement is REPORTED rather than silently clamped away.

Run: pixi run mojo run -I . tests/robot/test_so101_sim_map.mojo
"""

from std.math import abs, pi
from std.testing import assert_almost_equal, assert_equal, assert_true, assert_false, TestSuite

from noeira.envs.robots.so_arm101_xml import SoArm101Model
from noeira.physics3d.fields import actuator_column
from noeira.physics3d.gpu.constants import ACT_IDX_CTRL_MAX, ACT_IDX_CTRL_MIN
from noeira.robot.so101.arm import GRIPPER, SO101Calibration, SO101_N
from noeira.robot.so101.sim_map import (
    SimJointMap, tower_follower_zero_deg, tower_follower_zero_matches,
    TOWER_FOLLOWER_GRIPPER_ZERO_RAD, tower_follower_gripper_span_rad,
)

# Measured on the follower, 2026-08-25. `ofs` is sign-magnitude-decoded, which
# is why four of the six are negative.
# Functions, not `comptime` arrays: a comptime `Array` is not
# `ImplicitlyCopyable` and cannot be materialised at runtime.
def CAL_OFS(i: Int) -> Int:
    var v: List[Int] = [-430, 563, 382, -51, -353, -485]
    return v[i]


def CAL_LO(i: Int) -> Int:
    var v: List[Int] = [592, 816, 927, 875, 0, 2030]
    return v[i]


def CAL_HI(i: Int) -> Int:
    var v: List[Int] = [3317, 3236, 3130, 3212, 4095, 3513]
    return v[i]


def _cal() raises -> SO101Calibration:
    var ofs = Array[Int32, SO101_N](fill=0)
    var lo = Array[Int32, SO101_N](fill=0)
    var hi = Array[Int32, SO101_N](fill=0)
    for i in range(SO101_N):
        ofs[i] = Int32(CAL_OFS(i))
        lo[i] = Int32(CAL_LO(i))
        hi[i] = Int32(CAL_HI(i))
    return SO101Calibration(ofs^, lo^, hi^)


def _map() raises -> SimJointMap:
    """Model limits straight out of the compiled model."""
    var sf = SoArm101Model.make_spec_fields[DType.float64]()
    var lo_col = actuator_column(sf, ACT_IDX_CTRL_MIN, SO101_N)
    var hi_col = actuator_column(sf, ACT_IDX_CTRL_MAX, SO101_N)
    var lo = Array[Float64, SO101_N](fill=0.0)
    var hi = Array[Float64, SO101_N](fill=0.0)
    for i in range(SO101_N):
        lo[i] = Float64(lo_col[i])
        hi[i] = Float64(hi_col[i])
    return SimJointMap.identity(lo^, hi^)


def _mid(cal: SO101Calibration, i: Int) raises -> Int32:
    return Int32(Int(cal.mid(i)))


def test_identity_maps_the_servo_midpoint_to_zero() raises:
    """A body joint at its calibrated mid-point is 0 rad under the identity.

    This is the definition `SO101Calibration.radians` implements, and it is
    what makes `offset_rad` mean "how far the MODEL's zero sits from the
    servo's mid-point" rather than something ad hoc.
    """
    var cal = _cal()
    var m = _map()
    for i in range(SO101_N):
        if i == GRIPPER:
            continue
        assert_almost_equal(
            m.to_sim_unclamped(cal, i, _mid(cal, i)),
            0.0,
            atol=1e-3,
            msg="joint " + String(i) + " midpoint is not 0 rad",
        )


def test_gripper_maps_by_fraction_not_as_an_angle() raises:
    """The servo reports the gripper 0..100, the model wants radians, and the
    two ranges differ (101.8 deg of servo travel, 110 deg of model hinge). A
    fraction map takes the servo's closed end to the model's low limit and its
    open end to the high one; treating the 0..100 as degrees would land at
    ~1.75 rad for a fully open gripper and clamp."""
    var cal = _cal()
    var m = _map()
    assert_almost_equal(
        m.to_sim_unclamped(cal, GRIPPER, Int32(CAL_LO(GRIPPER))),
        m.sim_lo[GRIPPER],
        atol=1e-9,
        msg="closed gripper -> model low limit",
    )
    assert_almost_equal(
        m.to_sim_unclamped(cal, GRIPPER, Int32(CAL_HI(GRIPPER))),
        m.sim_hi[GRIPPER],
        atol=1e-9,
        msg="open gripper -> model high limit",
    )
    # And it NEVER clamps, at either end — that is what fraction-mapping buys.
    assert_equal(m.clamped_by(cal, GRIPPER, Int32(CAL_LO(GRIPPER))), 0.0)
    assert_equal(m.clamped_by(cal, GRIPPER, Int32(CAL_HI(GRIPPER))), 0.0)


def test_the_range_gap_is_reported_not_hidden() raises:
    """⚠ THE FINDING THIS FILE EXISTS FOR — corrected.

    THREE body joints have more calibrated travel than the model accepts, so
    an arm at its own end stop asks for a pose the simulation refuses.
    `clamped_by` must SAY so: a joint pinned at a limit looks exactly like a
    broken mapping otherwise, which is the confusion `teleop_sim` resolves.

    ⚠⚠ `wrist_roll` is EXCLUDED and is not a fourth. Its calibration is
    `0..4095` — lerobot's UNLIMITED marker, not a sweep — so it has no
    measured travel to exceed. An earlier version of this gate counted it and
    asserted five; that read a category error as a measurement.

    ⚠ The model is not at fault. Our ranges are byte-identical to
    `references/SO-ARM100-main/Simulation/SO101/so101_new_calib.xml`, so this
    is the ARMS' calibration reaching past the official travel (or the
    official travel being conservative) — not a bad port.
    """
    var cal = _cal()
    var m = _map()
    var clamped_joints = 0
    var continuous = 0
    for i in range(SO101_N):
        if i == GRIPPER:
            continue
        if cal.is_unlimited(i):
            continuous += 1
            continue
        var at_lo = m.clamped_by(cal, i, Int32(CAL_LO(i)))
        var at_hi = m.clamped_by(cal, i, Int32(CAL_HI(i)))
        if at_lo > 0.0 or at_hi > 0.0:
            clamped_joints += 1
            var v = m.to_sim(cal, i, Int32(CAL_HI(i)))
            assert_true(
                v <= m.sim_hi[i] + 1e-12 and v >= m.sim_lo[i] - 1e-12,
                "clamped value is inside the model range",
            )
    assert_equal(continuous, 1, "wrist_roll is the one continuous joint")
    assert_equal(
        clamped_joints,
        4,
        (
            "expected 4 LIMITED body joints whose real travel exceeds the"
            " model's — pan, lift, wrist_flex by degrees and elbow_flex by"
            " rounding; see the magnitude split below"
        ),
    )


def test_wrist_roll_is_continuous_and_the_model_is_not() raises:
    """A free-turning joint driven into a bounded model.

    `wrist_roll` reaches +/-pi under the reference mapping while the model
    stops at -2.744 / +2.841 rad, so roughly 20 degrees at each end is
    unreachable in simulation. That is a REAL sim/real difference and it is
    NOT the same kind as the three over-travelling joints: there is no
    calibration to redo and no sweep that was too enthusiastic.
    """
    var cal = _cal()
    var m = _map()
    assert_true(cal.is_unlimited(4), "wrist_roll carries the unlimited marker")
    assert_false(cal.is_unlimited(0), "shoulder_pan does not")
    # Both extremes of a full turn fall outside the model.
    assert_true(m.clamped_by(cal, 4, Int32(0)) > 0.3, "-pi is out of range")
    assert_true(m.clamped_by(cal, 4, Int32(4095)) > 0.2, "+pi is out of range")


def test_elbow_flex_is_marginal_and_the_others_are_not() raises:
    """Non-vacuity for the test above, and a finding of its own.

    A `clamped_by` that returned some positive number unconditionally would
    pass the count test, so the MAGNITUDES have to separate. They do, by three
    orders of magnitude:

    * `elbow_flex` overshoots by **9.2e-05 rad (0.005 deg)** — the MJCF writes
      its range as a rounded `±1.69`, where every other joint carries full
      precision, and that rounding is the entire overshoot. Nothing to fix.
    * pan, lift and wrist_flex overshoot by **> 0.05 rad (3 deg)** — real
      travel the simulation genuinely cannot represent.
    * `wrist_roll` is excluded: it is continuous, not over-travelling.

    ⚠ Which arm's calibration you use matters: this is the FOLLOWER's. The
    leader's elbow span is 11 ticks narrower and does NOT clamp at all, so a
    gate written against the leader would have found four joints, not five.
    """
    var cal = _cal()
    var m = _map()
    var elbow = max(
        m.clamped_by(cal, 2, Int32(CAL_LO(2))),
        m.clamped_by(cal, 2, Int32(CAL_HI(2))),
    )
    assert_true(elbow > 0.0, "elbow does overshoot, if barely")
    assert_true(
        elbow < 1.0e-3,
        "elbow overshoot is rounding, not travel: " + String(elbow),
    )
    for i in [Int(0), Int(1), Int(3)]:
        var worst = max(
            m.clamped_by(cal, i, Int32(CAL_LO(i))),
            m.clamped_by(cal, i, Int32(CAL_HI(i))),
        )
        assert_true(
            worst > 0.05,
            (
                "joint "
                + String(i)
                + " should overshoot by degrees, not rounding: "
                + String(worst)
            ),
        )


def test_sign_flip_actually_inverts() raises:
    """The knob teleop_sim exists to turn. A flipped sign must negate a body
    joint's angle and mirror the gripper's fraction."""
    var cal = _cal()
    var m = _map()
    var raw = Int32(CAL_LO(0) + (CAL_HI(0) - CAL_LO(0)) // 3)
    var pos = m.to_sim_unclamped(cal, 0, raw)
    m.sign[0] = -1.0
    assert_almost_equal(m.to_sim_unclamped(cal, 0, raw), -pos, atol=1e-12)

    var g_raw = Int32(CAL_LO(GRIPPER))
    m.sign[GRIPPER] = -1.0
    assert_almost_equal(
        m.to_sim_unclamped(cal, GRIPPER, g_raw),
        m.sim_hi[GRIPPER],
        atol=1e-9,
        msg="a mirrored gripper takes CLOSED to the model's open end",
    )


def test_offset_shifts_body_joints_and_not_the_gripper() raises:
    """`offset_rad` is a zero correction for angles. The gripper is not an
    angle on the servo side, so applying an offset there would silently move
    one end of the fraction map."""
    var cal = _cal()
    var m = _map()
    var raw = _mid(cal, 1)
    var g_raw = _mid(cal, GRIPPER)
    var before_g = m.to_sim_unclamped(cal, GRIPPER, g_raw)
    m.offset_rad[1] = 0.25
    m.offset_rad[GRIPPER] = 0.25
    assert_almost_equal(m.to_sim_unclamped(cal, 1, raw), 0.25, atol=1e-3)
    assert_almost_equal(
        m.to_sim_unclamped(cal, GRIPPER, g_raw), before_g, atol=1e-12
    )


def test_departure_from_the_reference_is_announced() raises:
    """A caller must be able to say "this is the reference mapping" or "someone
    changed it" — the difference between an honest viewer banner and a fake
    one. Note the polarity: the DEFAULT is the reference, and `True` means we
    have left it."""
    var m = _map()
    assert_false(m.differs_from_lerobot(), "the default IS the reference")
    m.sign[3] = -1.0
    assert_true(m.differs_from_lerobot(), "a flipped sign is a departure")


def test_monotonic_in_ticks() raises:
    """More ticks must never mean a smaller angle under a positive sign. A
    sign or midpoint slip inside `radians` would show up here."""
    var cal = _cal()
    var m = _map()
    for i in range(SO101_N):
        var prev = m.to_sim_unclamped(cal, i, Int32(CAL_LO(i)))
        for k in range(1, 11):
            var raw = Int32(
                CAL_LO(i) + (CAL_HI(i) - CAL_LO(i)) * k // 10
            )
            var v = m.to_sim_unclamped(cal, i, raw)
            assert_true(
                v >= prev - 1e-12,
                "joint " + String(i) + " is not monotonic in ticks",
            )
            prev = v


def test_matches_the_reference_implementation_exactly() raises:
    """⚠ THE GATE THAT MATTERS: our mapping vs so101-nexus's own.

    `lerobot_adapter/normalization.py::motor_ticks_to_sim_rad` is what a
    WORKING LeRobot-on-MuJoCo stack uses for these arms:

        mid  = (range_min + range_max) / 2
        qpos = sign * (ticks - mid) / TICKS_PER_RADIAN        # body
        frac = (ticks - range_min) / (range_max - range_min)  # gripper
        qpos = lower + frac * (upper - lower)

    with `sign = -1 if drive_mode else 1`, and lerobot HARD-CODES
    `drive_mode=0` for every SO-101 joint. The values below were evaluated
    from that formula against this follower's calibration, so this is a
    cross-implementation comparison and not a restatement of our own code.

    Four ticks per joint: both end stops, the mid-point, and an interior
    third — enough that a sign flip, a mid-point slip or a 4095-vs-4096 error
    each move at least one row.

    ⚠ `to_sim_UNCLAMPED`, because the reference does not clamp. Where the two
    would differ is exactly the range disagreement the tests above measure.

    ⚠ The gripper rows carry a looser tolerance, and the reason is upstream's:
    `so_arm101.xml` writes the gripper JOINT range at full precision
    (`-0.17453297762778586`) and its actuator `ctrlrange` ROUNDED to five
    decimals (`-0.17453`). We map onto the ctrlrange, because that is what
    actually clamps `ctrl`. The 2.9e-06 rad that falls out is the model's own
    rounding — byte-identical in `SO-ARM100-main` and Menagerie — not a
    disagreement about the mapping.
    """
    var cal = _cal()
    var m = _map()
    var joint: List[Int] = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2,
                            3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5]
    var ticks: List[Int] = [
        592, 1954, 3317, 1500,
        816, 2026, 3236, 1622,
        927, 2028, 3130, 1661,
        875, 2043, 3212, 1654,
        0, 2047, 4095, 1365,
        2030, 2771, 3513, 2524,
    ]
    var want: List[Float64] = [
        -2.090559213927274, -0.000767177693184, 2.090559213927274, -0.697364523104548,
        -1.856570017506056, 0.000000000000000, 1.856570017506056, -0.619879576092931,
        -1.690092458085058, -0.000767177693184, 1.690092458085058, -0.563875604490476,
        -1.792894268971758, -0.000767177693184, 1.792894268971758, -0.597631422990586,
        -3.141592653589793, -0.000767177693184, 3.141592653589793, -1.047197551196598,
        -0.174532925199433, 0.784750873384908, 1.745329251994330, 0.464989607190128,
    ]
    for k in range(len(joint)):
        var got = m.to_sim_unclamped(cal, joint[k], Int32(ticks[k]))
        # Body joints are exact; the gripper inherits the ctrlrange rounding.
        var tol = 5.0e-06 if joint[k] == GRIPPER else 1.0e-12
        assert_almost_equal(
            got,
            want[k],
            atol=tol,
            msg=(
                "joint "
                + String(joint[k])
                + " at "
                + String(ticks[k])
                + " ticks disagrees with so101-nexus"
            ),
        )


def test_our_gripper_limits_are_the_references() raises:
    """`SO101_GRIPPER_LIMITS_RAD = (radians(-10), radians(100))` in
    so101-nexus. Our model's gripper `ctrlrange` is read straight out of
    `so_arm101.xml`. They agree to 1e-15 — independent confirmation that both
    stacks are pointing at the same joint with the same convention, from two
    sources that never saw each other.

    ⚠ To 3e-06, not to 1e-15, and the shortfall is UPSTREAM'S ROUNDING, not
    ours: `so_arm101.xml` carries the gripper's joint `range` at full
    precision and its actuator `ctrlrange` at five decimals. Both
    `SO-ARM100-main` and Menagerie write the same rounded pair, so the model
    is a faithful port and its actuator simply cannot command the last
    0.00017 degrees of its own joint.
    """
    var m = _map()
    assert_almost_equal(m.sim_lo[GRIPPER], -0.17453292519943295, atol=5e-06)
    assert_almost_equal(m.sim_hi[GRIPPER], 1.7453292519943295, atol=5e-06)
    # And pin the rounding itself, so a future full-precision ctrlrange is a
    # visible change rather than a silent one.
    assert_almost_equal(m.sim_lo[GRIPPER], -0.17453, atol=1e-15)
    assert_almost_equal(m.sim_hi[GRIPPER], 1.74533, atol=1e-15)


# The follower's calibration of 2026-09-14 (`projects/so101-tower/calibration/
# follower.json`) — the one `tower_follower_zero_deg` was measured against.
def TOWER_OFS(i: Int) -> Int:
    var v: List[Int] = [-634, 510, 506, -81, -1216, -506]
    return v[i]


def TOWER_LO(i: Int) -> Int:
    var v: List[Int] = [882, 856, 791, 911, 585, 2031]
    return v[i]


def TOWER_HI(i: Int) -> Int:
    var v: List[Int] = [3092, 3272, 3002, 3247, 3509, 3515]
    return v[i]


def _tower_cal() raises -> SO101Calibration:
    var ofs = Array[Int32, SO101_N](fill=0)
    var lo = Array[Int32, SO101_N](fill=0)
    var hi = Array[Int32, SO101_N](fill=0)
    for i in range(SO101_N):
        ofs[i] = Int32(TOWER_OFS(i))
        lo[i] = Int32(TOWER_LO(i))
        hi[i] = Int32(TOWER_HI(i))
    return SO101Calibration(ofs^, lo^, hi^)


def test_tower_follower_applies_the_measured_zero() raises:
    """At a body joint's calibrated mid the tower follower's map gives the
    measured zero, the gripper is untouched, and the map announces that it
    departs from the reference. The pan and roll zeros are pinned by value:
    -10.7 and +5.0 deg are the MEASUREMENTS (two fits, `sim_map` module docstring), and a change to
    it must be a visible edit here, not a drift."""
    var cal = _tower_cal()
    var ref_map = _map()
    var m = SimJointMap.tower_follower(
        cal, ref_map.sim_lo.copy(), ref_map.sim_hi.copy()
    )
    assert_true(m.differs_from_lerobot(), "a measured zero is a departure")
    assert_almost_equal(tower_follower_zero_deg(0), -10.7, atol=1e-12)
    assert_almost_equal(tower_follower_zero_deg(4), 5.0, atol=1e-12)
    for i in range(SO101_N):
        var raw = Int32(Int(cal.mid(i)))
        var frac = cal.mid(i) - Float64(Int(cal.mid(i)))
        if i == GRIPPER:
            # the measured LINE, not the fraction of the model's range: the
            # mid is half of 130.46 deg past -9.25 deg (the two differ by
            # ~10 deg here — the reference would be 45 deg)
            var want_g = TOWER_FOLLOWER_GRIPPER_ZERO_RAD + (
                (Float64(raw) - Float64(TOWER_LO(i))) / Float64(TOWER_HI(i) - TOWER_LO(i))
            ) * tower_follower_gripper_span_rad()
            assert_almost_equal(m.to_sim_unclamped(cal, i, raw), want_g, atol=1e-12)
            assert_true(
                abs(m.to_sim_unclamped(cal, i, raw) - ref_map.to_sim_unclamped(cal, i, raw)) > 0.1,
                "the tower gripper is the fraction map",
            )
            continue
        # a half-tick mid is read at the tick below it
        var want = (tower_follower_zero_deg(i) - frac * 360.0 / 4095.0) * pi / 180.0
        assert_almost_equal(
            m.to_sim_unclamped(cal, i, raw), want, atol=1e-12,
            msg="joint " + String(i) + " at its mid is not the measured zero",
        )


def test_tower_follower_round_trips() raises:
    """`from_sim(to_sim(t)) == t` with the zero applied — a policy's target
    must reach the servo the joint came from."""
    var cal = _tower_cal()
    var ref_map = _map()
    var m = SimJointMap.tower_follower(
        cal, ref_map.sim_lo.copy(), ref_map.sim_hi.copy()
    )
    for i in range(SO101_N):
        for k in range(1, 10):
            var t = Int32(TOWER_LO(i) + (TOWER_HI(i) - TOWER_LO(i)) * k // 10)
            var back = m.from_sim(cal, i, m.to_sim_unclamped(cal, i, t))
            assert_true(
                abs(Int(back) - Int(t)) <= 1,
                "joint " + String(i) + ": " + String(t) + " -> " + String(back),
            )


def test_tower_follower_gripper_line_is_the_measured_one() raises:
    """The gripper map is pinned by value: 130.46 deg over the 1484-tick
    span (the arm's 360/4095 per tick), and the printed 25 mm cube's real
    grasp reading 13.1 lands at 0.1368 rad — the angle where the model's tip
    boxes are 25.0 mm apart (`sim_map` docstring, measured 2026-09-25)."""
    assert_almost_equal(tower_follower_gripper_span_rad() * 180.0 / pi, 130.46, atol=0.01)
    var cal = _tower_cal()
    var ref_map = _map()
    var m = SimJointMap.tower_follower(
        cal, ref_map.sim_lo.copy(), ref_map.sim_hi.copy()
    )
    var raw = Int32(Int(Float64(TOWER_LO(GRIPPER)) + 0.131 * Float64(TOWER_HI(GRIPPER) - TOWER_LO(GRIPPER)) + 0.5))
    assert_almost_equal(m.to_sim_unclamped(cal, GRIPPER, raw), 0.1368, atol=0.002)


def test_tower_follower_refuses_another_calibration() raises:
    """The zero belongs to ONE calibration. The 2026-08-25 one (this file's
    `_cal`) has other mids, so the map must refuse it rather than apply a
    zero measured on a different sweep."""
    assert_true(tower_follower_zero_matches(_tower_cal()))
    assert_false(tower_follower_zero_matches(_cal()))
    var ref_map = _map()
    var refused = False
    try:
        _ = SimJointMap.tower_follower(
            _cal(), ref_map.sim_lo.copy(), ref_map.sim_hi.copy()
        )
    except e:
        refused = String(e).find("recalibrated") >= 0
    assert_true(refused, "a foreign calibration was accepted")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
