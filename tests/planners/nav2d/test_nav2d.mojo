"""Gate `noeira/planners/nav2d` — costmap, scene footprints, MPPI navigation.

    pixi run mojo run -I . tests/planners/nav2d/test_nav2d.mojo

No robot, no physics: the closed-loop arm drives a KINEMATIC lagged unicycle
whose lags differ from the planner's model (so the planner is not checking
its own predictions), around a box that sits on the straight line — and a
straight-line controller on the same plant is run as the control, which must
hit the box, or the obstacle was not in the way and the arm proves nothing.
"""

from std.math import atan2, cos, exp, sin, sqrt, pi
from std.random import seed as _set_seed
from std.testing import assert_true

from noeira.planners.nav2d import (
    Footprint, NavGrid, NAV_UNREACHABLE, footprint_of_geom, ResponseMap,
    nav_params_default, UnicycleNavigator, FP_RECT, FP_CIRCLE,
)


def _close(a: Float64, b: Float64, tol: Float64) -> Bool:
    var d = a - b
    return (d if d > 0.0 else -d) <= tol


def test_footprint_distance() raises:
    var r = Footprint.rect(1.0, 0.0, 0.5, 0.2, pi / 2.0)  # long axis along y
    assert_true(_close(r.distance(1.0, 0.0), 0.0, 1e-12), "centre is inside")
    assert_true(_close(r.distance(1.0, 0.45), 0.0, 1e-12), "inside along the rotated long axis")
    assert_true(_close(r.distance(1.5, 0.0), 0.3, 1e-9), "0.3 beyond the 0.2 short side")
    assert_true(_close(r.distance(1.0, 1.0), 0.5, 1e-9), "0.5 beyond the 0.5 long side")
    var c = Footprint.circle(0.0, 0.0, 0.5)
    assert_true(_close(c.distance(1.0, 0.0), 0.5, 1e-12), "circle distance")
    print("  ok  footprint distances (rotated rect, circle)")


def test_navigation_function_goes_around() raises:
    # a wall across x = 0 from y = -2 to 2, room [-3, 3]^2
    var fp = List[Footprint]()
    fp.append(Footprint.rect(0.0, 0.0, 0.1, 2.0, 0.0))
    var g = NavGrid(fp, -3.0, -3.0, 3.0, 3.0, 0.05, 0.3)
    g.set_goal(2.0, 0.0)
    var straight = 4.0
    # the shortest free path rounds BOTH corners of the inflated wall,
    # (-0.4, 2.3) and (0.4, 2.3): two diagonals plus the wall's inflated width
    var around = 2.0 * sqrt(1.6 ** 2 + 2.3 ** 2) + 0.8
    var c = g.cost_to_go(-2.0, 0.0)
    assert_true(c > straight + 1.0, "behind the wall the cost-to-go must exceed the straight line: " + String(c))
    # an 8-connected grid overestimates Euclidean length by at most 8.24 %
    assert_true(c >= around - 0.1 and c <= around * 1.0824 + 0.1,
                "and lie in [detour, detour * 1.0824], detour " + String(around) + ": " + String(c))
    assert_true(_close(g.cost_to_go(2.5, 0.0), 0.5, 0.1), "near the goal it is the distance")
    assert_true(g.cost_to_go(0.0, 0.0) >= NAV_UNREACHABLE * 0.5, "inside the wall it is unreachable")
    var raised = False
    try:
        g.set_goal(0.0, 1.0)
    except:
        raised = True
    assert_true(raised, "a goal inside an inflated obstacle must raise")
    # descent at the start points AROUND (up or down), not straight through
    var h = g.descent(-2.0, 0.0)
    var ah = h if h > 0.0 else -h
    assert_true(ah > 0.3, "descent heading must turn away from the wall, got " + String(h))
    print("  ok  navigation function: detour", c, "vs straight", straight, "; descent heading", h)


def test_footprint_of_geom() raises:
    # a box yawed 90 deg about z: (x, y, z, w) = (0, 0, sin 45, cos 45)
    var s = sin(pi / 4.0)
    var r = footprint_of_geom(3, 1.0, 2.0, 0.5, 0.0, 0.0, s, s, 0.0, 0.0, 0.4, 0.1, 0.5, 0.02, 2.0)
    assert_true(r[0] and r[1].kind == FP_RECT, "upright box -> rect")
    assert_true(_close(r[1].yaw, pi / 2.0, 1e-9), "yaw carried: " + String(r[1].yaw))
    # the same box tipped 90 deg about x -> bounding circle
    var t = footprint_of_geom(3, 1.0, 2.0, 0.5, s, 0.0, 0.0, s, 0.0, 0.0, 0.4, 0.1, 0.5, 0.02, 2.0)
    assert_true(t[0] and t[1].kind == FP_CIRCLE, "tipped box -> circle")
    # a box lying entirely above z_max is not an obstacle for the ground robot
    var u = footprint_of_geom(3, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.4, 0.4, 0.1, 0.02, 2.0)
    assert_true(not u[0], "above z_max -> skipped")
    # a thin visual floor slab below z_min
    var f = footprint_of_geom(3, 0.0, 0.0, 0.0005, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 4.0, 4.0, 0.0005, 0.02, 2.0)
    assert_true(not f[0], "a 1 mm slab below z_min -> skipped")
    print("  ok  geom -> footprint (yawed box, tipped box, above / below the band)")


struct Plant(Movable):
    """The 'real' robot for the closed-loop arm: lags DIFFERENT from the model."""

    var x: Float64
    var y: Float64
    var yaw: Float64
    var v: Float64
    var w: Float64

    def __init__(out self, x: Float64, y: Float64, yaw: Float64):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = 0.0
        self.w = 0.0

    def step(mut self, vc: Float64, wc: Float64, dt: Float64):
        self.v += (vc - self.v) * (1.0 - exp(-dt / 0.45))
        self.w += (wc - self.w) * (1.0 - exp(-dt / 0.2))
        self.yaw += self.w * dt
        self.x += self.v * cos(self.yaw) * dt
        self.y += self.v * sin(self.yaw) * dt


def test_mppi_goes_around_a_box() raises:
    _set_seed(0x5EED)
    var fp = List[Footprint]()
    fp.append(Footprint.rect(0.0, 0.0, 0.3, 1.0, 0.0))     # ON the straight line
    var inflate = 0.35
    var p = nav_params_default()
    var nav = UnicycleNavigator[](
        NavGrid(fp, -3.0, -3.0, 3.0, 3.0, 0.05, inflate),
        ResponseMap.identity(p.v_min, p.v_max, p.w_max), p,
    )
    nav.set_goal(2.0, 0.0)

    # the CONTROL: aim straight at the goal on the same plant — must hit
    var ctl = Plant(-2.0, 0.0, 0.0)
    var ctl_min_clear = 1.0e9
    for _ in range(80):
        var err = atan2(0.0 - ctl.y, 2.0 - ctl.x) - ctl.yaw
        ctl.step(0.8, 2.0 * err, p.dt)
        var c = nav.clearance(ctl.x, ctl.y)
        if c < ctl_min_clear:
            ctl_min_clear = c
    assert_true(ctl_min_clear < inflate, "control: the straight line must cross the inflated box, min clearance " + String(ctl_min_clear))

    var r = Plant(-2.0, 0.0, 0.0)
    var min_clear = 1.0e9
    var reached = -1
    for k in range(300):
        var cmd = nav.plan(r.x, r.y, r.yaw, r.v, r.w)
        r.step(cmd[0], cmd[1], p.dt)
        var c = nav.clearance(r.x, r.y)
        if c < min_clear:
            min_clear = c
        if reached < 0 and sqrt((r.x - 2.0) ** 2 + r.y ** 2) < 0.3:
            reached = k
    var d_end = sqrt((r.x - 2.0) ** 2 + r.y ** 2)
    print("  arm: reached at step", reached, "| end", r.x, r.y, "dist", d_end, "speed", r.v,
          "| min clearance", min_clear, "(inflate", inflate, ") | control min clearance", ctl_min_clear)
    assert_true(reached >= 0, "MPPI never got within 0.3 m of the goal")
    assert_true(d_end < 0.35, "and did not stay there: " + String(d_end))
    assert_true(min_clear > 0.0, "MPPI touched the box itself")
    assert_true(min_clear > inflate * 0.7, "MPPI cut deep into the inflation: " + String(min_clear))
    print("  ok  MPPI goes around a box the straight line hits")


struct SlowPlant(Movable):
    """A sluggish robot (speed lag 0.9 s) — the case the approach limit is for."""

    var x: Float64
    var y: Float64
    var yaw: Float64
    var v: Float64
    var w: Float64

    def __init__(out self):
        self.x = -1.5
        self.y = 0.0
        self.yaw = 0.0
        self.v = 0.0
        self.w = 0.0

    def step(mut self, vc: Float64, wc: Float64, dt: Float64):
        self.v += (vc - self.v) * (1.0 - exp(-dt / 0.9))
        self.w += (wc - self.w) * (1.0 - exp(-dt / 0.2))
        self.yaw += self.w * dt
        self.x += self.v * cos(self.yaw) * dt
        self.y += self.v * sin(self.yaw) * dt


def _overshoot(slow_radius: Float64) raises -> Float64:
    """Max distance PAST a goal 3 m ahead, on an open floor."""
    _set_seed(0xA11CE)
    var fp = List[Footprint]()
    fp.append(Footprint.circle(10.0, 10.0, 0.1))   # far away: an open floor
    var p = nav_params_default()
    p.tau_v = 0.3                                    # the model is optimistic
    p.slow_radius = slow_radius
    var nav = UnicycleNavigator[](
        NavGrid(fp, -3.0, -3.0, 3.0, 3.0, 0.05, 0.3),
        ResponseMap.identity(p.v_min, p.v_max, p.w_max), p,
    )
    nav.set_goal(1.5, 0.0)
    var r = SlowPlant()
    var past = 0.0
    for _ in range(150):
        var cmd = nav.plan(r.x, r.y, r.yaw, r.v, r.w)
        r.step(cmd[0], cmd[1], p.dt)
        if r.x - 1.5 > past:
            past = r.x - 1.5
    return past


def test_approach_limit_cuts_overshoot() raises:
    var free = _overshoot(0.0)
    var lim = _overshoot(1.5)
    print("  approach limit: overshoot without", free, "m, with slow_radius 1.5:", lim, "m")
    assert_true(lim < free, "the approach limit must reduce the overshoot")
    assert_true(lim < 0.25, "and keep it under 0.25 m on a 0.9 s plant: " + String(lim))
    print("  ok  the approach limit cuts a sluggish robot's overshoot")


def main() raises:
    print("=== planners/nav2d ===")
    test_footprint_distance()
    test_navigation_function_goes_around()
    test_footprint_of_geom()
    test_mppi_goes_around_a_box()
    test_approach_limit_cuts_overshoot()
    print("[PASS] nav2d")
