"""`delta_action.ServoLag` — the real servos' delay + first-order lag.

    pixi run mojo run -I . tests/tasks/test_servo_lag.mojo

1. off (no tau, no delay) it is the identity;
2. on, tau 140 ms / dt 32 ms / delay 2 ticks: a step command leaves the
   actuator target untouched for EXACTLY 2 ticks, then closes
   alpha = 1 - exp(-32/140) = 0.2043 of the remaining gap per tick
   (the real follower covered ~20 % of each step per tick on 27 Sep);
3. `reset_lane` settles the lane on its new joints: no target from the last
   episode leaks into the next;
4. lanes are independent (another lane's draw and history untouched).
"""

from std.math import exp
from std.testing import assert_almost_equal, assert_equal, assert_true

from noeira.tasks.delta_action import ServoLag, DELTA_ACT


def main() raises:
    var off = ServoLag(2, 0.0, 0.0, 0, 0, 0.032)
    assert_true(not off.on, "no tau, no delay: off")
    assert_equal(off.apply(0, 0, 0.7), 0.7, "off is the identity")
    print("  1. off: identity")

    var lag = ServoLag(2, 140.0, 140.0, 2, 2, 0.032)
    var q = List[Float64](length=2 * DELTA_ACT, fill=0.0)
    for j in range(DELTA_ACT):
        q[DELTA_ACT + j] = 1.0  # lane 1 rests at 1.0
    lag.reset_lane(0, q, 0, 0.5, 0.5)
    lag.reset_lane(1, q, DELTA_ACT, 0.5, 0.5)
    var alpha = 1.0 - exp(-0.032 / 0.140)
    assert_almost_equal(lag.alpha[0], alpha, atol=1e-12, msg="alpha")
    assert_equal(lag.delay[0], 2, "delay")
    # a step of lane 0 joint 0 from 0 to 0.1; lane 1 held at 1.0
    var ys = List[Float64]()
    for t in range(8):
        var y0 = lag.apply(0, 0, 0.1)
        for j in range(1, DELTA_ACT):
            _ = lag.apply(0, j, 0.0)
        for j in range(DELTA_ACT):
            _ = lag.apply(1, j, 1.0)
        lag.advance()
        ys.append(y0)
    print("  2. step 0 -> 0.1, tau 140 ms, delay 2:", ys[0], ys[1], ys[2], ys[3], ys[4])
    assert_equal(ys[0], 0.0, "tick 0: the command has not arrived")
    assert_equal(ys[1], 0.0, "tick 1: still not")
    var expect = 0.0
    for t in range(2, 8):
        expect = expect + alpha * (0.1 - expect)
        assert_almost_equal(ys[t], expect, atol=1e-12, msg="first-order lag")
    assert_almost_equal(lag.y[DELTA_ACT], 1.0, atol=1e-12, msg="lane 1 untouched")
    # 3. reset settles on the new pose
    var q2 = List[Float64](length=2 * DELTA_ACT, fill=0.0)
    q2[0] = -0.3
    lag.reset_lane(0, q2, 0, 0.5, 0.5)
    var y_first = lag.apply(0, 0, -0.3)
    assert_almost_equal(y_first, -0.3, atol=1e-12, msg="after a reset the lag holds the new pose")
    print("  3. reset settles the lane on its new pose")
    print("SERVO LAG OK")
