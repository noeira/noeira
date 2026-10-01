"""`delta_action.target_step` — `--action target`, SimToolReal's arm rule.

    pixi run mojo run -I . tests/tasks/test_target_step.mojo

1. the rule: `clamp(prev + a * scale)`, the gripper at its own scale, the
   ctrlrange clamp;
2. the property it exists for — under `ServoLag` (tau 140 ms, delay 2) a
   constant action drives the target `scale` per step whatever the lagging
   joint does, so after N steps it stands at N x scale; the `q`-anchored
   delta under the same lag stands only ~one step ahead of the joint
   (the CONTROL: if both anchors gave the same targets, the test would not
   be measuring the anchor);
3. `lead` bounds the target to `q +- lead` (and still to the ctrlrange);
   0 leaves it unbounded.
"""

from std.testing import assert_almost_equal, assert_true

from noeira.tasks.delta_action import (
    DELTA_ACT, ServoLag, delta_target, target_step,
)


def main() raises:
    # 1. the rule
    assert_almost_equal(
        target_step(0.3, -1.0, 0.5, 0, -2.0, 2.0, 0.04, 0.2), 0.32,
        atol=1e-12, msg="prev + a * arm scale, q ignored",
    )
    assert_almost_equal(
        target_step(0.3, 0.3, 0.5, DELTA_ACT - 1, -2.0, 2.0, 0.04, 0.2), 0.4,
        atol=1e-12, msg="the gripper word at the gripper scale",
    )
    assert_almost_equal(
        target_step(1.99, 1.99, 1.0, 0, -2.0, 2.0, 0.04, 0.2), 2.0,
        atol=1e-12, msg="clamped to the ctrlrange",
    )
    print("  1. clamp(prev + a * scale)")

    # 2. under the servo lag: target anchor vs measured-joint anchor
    comptime N = 10
    var dt = 0.032
    var scale = 0.04
    var lag_t = ServoLag(1, 140.0, 140.0, 2, 2, dt)
    var lag_d = ServoLag(1, 140.0, 140.0, 2, 2, dt)
    var z = List[Float64](length=DELTA_ACT, fill=0.0)
    lag_t.reset_lane(0, z, 0, 0.5, 0.5)
    lag_d.reset_lane(0, z, 0, 0.5, 0.5)
    var prev = 0.0
    var q_t = 0.0  # the joint, idealised as the lagged actuator target
    var q_d = 0.0
    var tgt_d = 0.0
    var q_d_at = 0.0  # the joint the last q-anchored target was taken from
    for _ in range(N):
        q_d_at = q_d
        prev = target_step(prev, q_t, 1.0, 0, -3.0, 3.0, scale)
        tgt_d = delta_target(q_d, 1.0, 0, -3.0, 3.0, scale)
        q_t = lag_t.apply(0, 0, prev)
        q_d = lag_d.apply(0, 0, tgt_d)
        for j in range(1, DELTA_ACT):
            _ = lag_t.apply(0, j, 0.0)
            _ = lag_d.apply(0, j, 0.0)
        lag_t.advance()
        lag_d.advance()
    print("  2. after", N, "steps of a = 1 under the lag: target anchor",
          prev, "(joint", q_t, ") | q anchor", tgt_d, "(joint", q_d, ")")
    assert_almost_equal(prev, Float64(N) * scale, atol=1e-12,
                        msg="the target integrates the actions")
    assert_true(prev > q_t + 0.1, "the target runs ahead of the lagging joint")
    assert_almost_equal(tgt_d - q_d_at, scale, atol=1e-12,
                        msg="CONTROL: the q anchor is one step past the joint")
    assert_true(prev > tgt_d + 0.1, "CONTROL: the two anchors differ under lag")

    # 3. the lead bound
    assert_almost_equal(
        target_step(0.5, 0.1, 1.0, 0, -2.0, 2.0, 0.04, 0.2, 0.2), 0.3,
        atol=1e-12, msg="lead caps the target at q + lead",
    )
    assert_almost_equal(
        target_step(-0.5, 0.1, -1.0, 0, -2.0, 2.0, 0.04, 0.2, 0.2), -0.1,
        atol=1e-12, msg="and at q - lead",
    )
    assert_almost_equal(
        target_step(1.9, 1.95, 1.0, 0, -2.0, 2.0, 0.04, 0.2, 0.2), 1.94,
        atol=1e-12, msg="inside the lead: the plain rule",
    )
    assert_almost_equal(
        target_step(0.5, 0.1, 1.0, 0, -2.0, 2.0, 0.04, 0.2), 0.54,
        atol=1e-12, msg="lead 0: unbounded",
    )
    print("  3. lead bound")
    print("test_target_step: ALL PASS")
