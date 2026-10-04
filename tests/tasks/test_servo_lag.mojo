"""`delta_action.ServoLag` — the real servos' delay + first-order lag.

    pixi run mojo run -I . tests/tasks/test_servo_lag.mojo

1. off (no tau, no delay) it is the identity;
2. on, tau 140 ms / dt 32 ms / delay 2 ticks: a step command leaves the
   actuator target untouched for EXACTLY 2 ticks, then closes
   alpha = 1 - exp(-32/140) = 0.2043 of the remaining gap per tick
   (the real follower covered ~20 % of each step per tick on 27 Sep);
3. `reset_lane` settles the lane on its new joints: no target from the last
   episode leaks into the next;
4. lanes are independent (another lane's draw and history untouched);
5. `set_per_joint`: each joint gets its OWN tau / delay / cap from its own
   range (a wrist with delay 3 starts 3 ticks late while the pan, delay 0,
   moves at once; a capped gripper never moves faster than its cap);
6. `set_offset`: each joint settles at its lagged target + its own signed
   offset (lift +0.02, elbow -0.05 here), the others untouched;
7. `set_period`: a longer period moves the lag further per tick
   (alpha = 1 - exp(-period / tau), the cap = vmax x period).
"""

from std.math import exp
from std.random import seed
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

    # 5. per-joint draws: ranges with lo == hi pin each joint's numbers
    seed(7)
    var pj = ServoLag(1, 50.0, 50.0, 1, 1, 0.032)
    pj.set_per_joint(
        "1e-9;100;100;100;100;100",  # pan: no lag (tau 0), others 100 ms
        "0;1;1;3;1;1",               # wrist flex 3 ticks late, pan none
        "0;0;0;0;0;1.0",             # only the gripper capped, 1 rad/s
    )
    assert_true(pj.per_joint, "per-joint mode on")
    var z6 = List[Float64](length=DELTA_ACT, fill=0.0)
    pj.reset_lane(0, z6, 0, 0.5, 0.5)
    assert_equal(pj.delay[3], 3, "wrist flex delay 3")
    assert_equal(pj.delay[0], 0, "pan delay 0")
    assert_almost_equal(pj.alpha[0], 1.0, atol=1e-9, msg="pan: tau ~0, alpha 1")
    assert_almost_equal(pj.alpha[1], 1.0 - exp(-0.032 / 0.1), atol=1e-12, msg="lift alpha")
    var y_pan = List[Float64]()
    var y_wf = List[Float64]()
    var y_gr = List[Float64]()
    for _ in range(5):
        for j in range(DELTA_ACT):
            var v = pj.apply(0, j, 0.5)
            if j == 0:
                y_pan.append(v)
            if j == 3:
                y_wf.append(v)
            if j == DELTA_ACT - 1:
                y_gr.append(v)
        pj.advance()
    assert_almost_equal(y_pan[0], 0.5, atol=1e-9, msg="pan follows at once")
    assert_equal(y_wf[2], 0.0, "wrist flex: nothing for 3 ticks")
    assert_true(y_wf[3] > 0.0, "wrist flex moves on tick 3")
    assert_equal(y_gr[0], 0.0, "gripper delay 1: nothing on tick 0")
    assert_almost_equal(y_gr[1], 0.032, atol=1e-12, msg="gripper capped at 1 rad/s x 32 ms")
    print("  5. per-joint: pan", y_pan[0], "| wrist flex", y_wf[2], "->", y_wf[3],
          "| gripper", y_gr[1], "(cap 0.032)")

    # 6. signed per-joint offsets
    var og = ServoLag(1, 0.0, 0.0, 0, 0, 0.032)
    og.set_offset("0;0.02;-0.05,-0.05;0;0;0")
    assert_true(og.on, "an offset turns the model on")
    og.reset_lane(0, z6, 0, 0.5, 0.5)
    var o1 = 0.0
    var o2 = 0.0
    var o0 = 0.0
    for j in range(DELTA_ACT):
        var v = og.apply(0, j, 0.3)
        if j == 0:
            o0 = v
        if j == 1:
            o1 = v
        if j == 2:
            o2 = v
    assert_almost_equal(o1, 0.32, atol=1e-12, msg="lift settles +0.02 off its target")
    assert_almost_equal(o2, 0.25, atol=1e-12, msg="elbow settles -0.05 off its target")
    assert_almost_equal(o0, 0.3, atol=1e-12, msg="pan has no offset")
    print("  6. offsets: pan", o0, "| lift", o1, "| elbow", o2, "for a 0.3 target")

    # 7. the period: 40 ms against 32 ms, same tau
    var p32 = ServoLag(1, 100.0, 100.0, 0, 0, 0.032)
    var p40 = ServoLag(1, 100.0, 100.0, 0, 0, 0.032)
    p40.set_period("40,40")
    p32.reset_lane(0, z6, 0, 0.5, 0.5)
    p40.reset_lane(0, z6, 0, 0.5, 0.5)
    assert_almost_equal(p32.alpha[0], 1.0 - exp(-0.032 / 0.1), atol=1e-12, msg="32 ms alpha")
    assert_almost_equal(p40.alpha[0], 1.0 - exp(-0.040 / 0.1), atol=1e-12, msg="40 ms alpha")
    print("  7. period: alpha", p32.alpha[0], "(32 ms) ->", p40.alpha[0], "(40 ms)")
    print("SERVO LAG OK")
