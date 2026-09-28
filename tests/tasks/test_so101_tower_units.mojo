"""`So101TowerUnits` — the store's LeRobot units and the joint-zero choice.

    pixi run mojo run -I . tests/tasks/test_so101_tower_units.mojo

`tower_demo_rerender.mojo` writes a store's `qpos` / `action` with it and
`tower_act_eval.mojo` reads a student's output back with it, so the two must
be exact inverses under either choice:

1. `none` is the lerobot reference: degrees = radians x 180/pi, 0 -> 0;
2. `follower` puts the model's zero at the MEASURED LeRobot reading: model 0
   rad reads `-tower_follower_zero_deg` (pan +10.7 deg), LeRobot 0 deg is the
   model's `zero`;
3. both round-trip (LeRobot -> joint -> LeRobot) to 1e-12 over each joint's
   range, and the env action word round-trips too;
4. the gripper: `none` and `follower-v1` by fraction of the ctrlrange,
   `follower` on its measured line (13.1 -> 0.1368 rad, the 25 mm cube);
5. the choices differ by exactly the zero on the body joints — the ~10
   degree pan gap a store / eval mismatch would silently cost;
6. an unknown choice is refused, and `describe` names the choice (the
   store's provenance line).

CPU only: nothing is rendered.
"""

from std.math import abs, pi
from std.testing import assert_almost_equal, assert_true, assert_false, TestSuite

from noeira.robot.so101.sim_map import tower_follower_zero_deg
from noeira.tasks.so101_tower_rig import (
    So101TowerUnits, RIG_ACT, RIG_GRIPPER, RIG_JOINT_ZERO_NONE,
    RIG_JOINT_ZERO_FOLLOWER, RIG_JOINT_ZERO_FOLLOWER_V1,
)


def test_none_is_the_reference_map() raises:
    var u = So101TowerUnits()
    for k in range(RIG_ACT):
        if k == RIG_GRIPPER:
            continue
        assert_almost_equal(u.joint_to_lerobot(k, 0.0), 0.0, atol=1e-15)
        assert_almost_equal(u.joint_to_lerobot(k, 1.0), 180.0 / pi, atol=1e-12)
        assert_almost_equal(u.lerobot_to_joint(k, 90.0), pi / 2.0, atol=1e-15)


def test_follower_puts_the_zero_at_the_measured_reading() raises:
    var u = So101TowerUnits(RIG_JOINT_ZERO_FOLLOWER)
    assert_almost_equal(u.joint_to_lerobot(0, 0.0), 10.7, atol=1e-12,
                        msg="model pan 0 is where the real arm reads +10.7 deg")
    for k in range(RIG_ACT):
        if k == RIG_GRIPPER:
            continue
        var z = tower_follower_zero_deg(k)
        assert_almost_equal(u.joint_to_lerobot(k, 0.0), -z, atol=1e-12)
        assert_almost_equal(u.lerobot_to_joint(k, 0.0), z * pi / 180.0, atol=1e-15)


def test_both_choices_round_trip() raises:
    var choices: List[String] = [
        RIG_JOINT_ZERO_NONE, RIG_JOINT_ZERO_FOLLOWER, RIG_JOINT_ZERO_FOLLOWER_V1
    ]
    for c in choices:
        var u = So101TowerUnits(c)
        for k in range(RIG_ACT):
            for s in range(11):
                var q = u.lo[k] + (u.hi[k] - u.lo[k]) * Float64(s) / 10.0
                var v = u.joint_to_lerobot(k, q)
                assert_almost_equal(u.lerobot_to_joint(k, v), q, atol=1e-12,
                                    msg=c + " joint " + String(k))
                var a = u.joint_to_action(k, q)
                assert_almost_equal(u.action_to_joint(k, a), q, atol=1e-12)


def test_the_choices_differ_by_the_zero_and_the_gripper_line() raises:
    var n = So101TowerUnits(RIG_JOINT_ZERO_NONE)
    var f = So101TowerUnits(RIG_JOINT_ZERO_FOLLOWER)
    var v = So101TowerUnits(RIG_JOINT_ZERO_FOLLOWER_V1)
    var q = 0.3
    for k in range(RIG_ACT):
        var gap = n.joint_to_lerobot(k, q) - f.joint_to_lerobot(k, q)
        if k == RIG_GRIPPER:
            assert_almost_equal(
                n.joint_to_lerobot(k, q), v.joint_to_lerobot(k, q), atol=1e-15,
                msg="none and follower-v1 share the fraction map",
            )
            # the printed cube's real grasp reading, on the measured line
            assert_almost_equal(f.lerobot_to_joint(k, 13.1), 0.1368, atol=0.002)
            assert_almost_equal(n.lerobot_to_joint(k, 13.1), 0.0770, atol=0.002)
        else:
            assert_almost_equal(gap, tower_follower_zero_deg(k), atol=1e-12)
    # the gap a mismatch costs on the pan: ~10 degrees, not a rounding
    assert_true(abs(tower_follower_zero_deg(0)) > 5.0)


def test_follower_v1_is_follower_with_the_roll_at_zero() raises:
    """The map before 2026-09-25, kept so the `follower` stores and students
    of then (roll zero 0) still read in their own units."""
    var f = So101TowerUnits(RIG_JOINT_ZERO_FOLLOWER)
    var v = So101TowerUnits(RIG_JOINT_ZERO_FOLLOWER_V1)
    for k in range(RIG_ACT):
        if k == 4:
            assert_almost_equal(v.zero_rad[k], 0.0, atol=1e-15)
            assert_almost_equal(f.zero_rad[k], 5.0 * pi / 180.0, atol=1e-15)
        else:
            assert_almost_equal(v.zero_rad[k], f.zero_rad[k], atol=1e-15)
    assert_true(v.describe().find("0.00 deg)") >= 0, v.describe())
    assert_true(f.describe().find("gripper -9.25 + 130.46 deg") >= 0, f.describe())


def test_unknown_choice_is_refused_and_describe_names_it() raises:
    var refused = False
    try:
        _ = So101TowerUnits(String("leader"))
    except:
        refused = True
    assert_true(refused, "'leader' is not a joint zero")
    assert_true(So101TowerUnits().describe() == "joint zero none")
    var d = So101TowerUnits(RIG_JOINT_ZERO_FOLLOWER).describe()
    assert_true(d.startswith("joint zero follower ("), d)
    assert_false(d.find("-10.70 -3.60 -7.30 0.00 5.00") < 0, d)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
