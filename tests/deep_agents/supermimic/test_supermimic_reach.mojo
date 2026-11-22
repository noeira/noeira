"""The reach interface and its loop (`noeira/deep_agents/supermimic/reach.mojo`).

    pixi run mojo run -I . tests/deep_agents/supermimic/test_supermimic_reach.mojo

Answers known by construction: the interfaces are written by hand, the "robot"
is a line of arithmetic. No simulator, no policy, no data.

WHAT EACH SECTION CAN CATCH
===========================
[1] the file: its numbers come back in their places; a file without the
    header, of another version, with a number missing, or with the cut under
    the loop's threshold is refused.
[2] z: at a pair's own place and one kernel length away, by the closed form;
    two pairs add; a starting pair (counts 0) is in z and NOT in the density.
[3] the grade: sure at the cut, likely at the loop's threshold, refused under
    it, and an interface that was never loaded refuses everything.
[4] the frames: a target in front of a pelvis turned a quarter turn is to its
    left in the world, and back.
[5] the ramp: nothing before the call's z, the min-jerk values at each fifth
    step, the interface's z from step 45 on, no change in between.
[6] the loop on a hand that sits a fixed offset from where it is commanded:
    the correction goes to minus the offset by (1 - gain) a second, in the
    HEADING frame of the moment (a turned pelvis), and stops at its limit.
[7] the domain: a correction that would leave the domain is scaled back to its
    edge, five such updates in a row give up, the correction is dropped and z
    is the interface's at the target from then on.
[8] the refusal starts nothing; the way back ramps to the other z and ends.
"""

from std.math import exp

from noeira.deep_agents.supermimic.reach import (
    SM_REACH_LIKELY, SM_REACH_REFUSED, SM_REACH_SURE, SMReachInterface, SMReachLoop, SMZRamp,
    sm_min_jerk, sm_reach_body, sm_reach_grade_name, sm_reach_world,
)
from noeira.io.fileio import write_file_atomic


struct Tally:
    var checks: Int
    var fails: Int

    def __init__(out self):
        self.checks = 0
        self.fails = 0

    def truth(mut self, ok: Bool, msg: String):
        self.checks += 1
        if ok:
            print("  ok:", msg)
        else:
            self.fails += 1
            print("  FAIL:", msg)


def _near(a: Float64, b: Float64, tol: Float64) -> Bool:
    var d = a - b
    return (d if d >= 0.0 else -d) <= tol


def _write(path: String, text: String) raises:
    var b = List[UInt8]()
    for c in text.as_bytes():
        b.append(c)
    write_file_atomic(path.copy(), b)


def _refused(path: String) -> Bool:
    try:
        _ = SMReachInterface.load(path)
    except:
        return True
    return False


def main() raises:
    var t = Tally()
    var path = String("/tmp/sm_reach_test.txt")
    # three pairs of width 2, kernel length 0.1: two successes 0.1 m apart on x, one starting pair far away
    var body = String("0.5 -1.0\n2.0 4.0\n0.0 0.0 0.0 1 1.0 2.0\n0.1 0.0 0.0 1 -3.0 0.5\n1.0 1.0 1.0 0 7.0 7.0\n")
    _write(path, String("smreach 1 3 2 0.1 0.8 1.5\n") + body)

    print("\n[1] the file")
    var f = SMReachInterface.load(path)
    t.truth(f.loaded() and f.n == 3 and f.width == 2 and _near(f.ell, 0.1, 1e-15) and _near(f.loop_threshold, 0.8, 1e-15) and _near(f.cut, 1.5, 1e-15), "the header")
    t.truth(f.m[0] == 0.5 and f.m[1] == -1.0 and f.s[0] == 2.0 and f.s[1] == 4.0, "m and s")
    t.truth(_near(f.x[3], 0.1, 1e-15) and f.counts[0] == 1.0 and f.counts[2] == 0.0 and f.alpha[2] == -3.0 and f.alpha[5] == 7.0, "a row: place, counts, alpha")
    var bad = String("/tmp/sm_reach_test_bad.txt")
    _write(bad, String("reach 1 3 2 0.1 0.8 1.5\n") + body)
    t.truth(_refused(bad), "no `smreach` header: refused")
    _write(bad, String("smreach 2 3 2 0.1 0.8 1.5\n") + body)
    t.truth(_refused(bad), "another version: refused")
    _write(bad, String("smreach 1 3 2 0.1 0.8 1.5\n0.5 -1.0\n2.0 4.0\n0.0 0.0 0.0 1 1.0 2.0\n0.1 0.0 0.0 1 -3.0 0.5\n1.0 1.0 1.0 0 7.0\n"))
    t.truth(_refused(bad), "a number missing: refused")
    _write(bad, String("smreach 1 3 2 0.1 1.5 0.8\n") + body)
    t.truth(_refused(bad), "the cut under the loop's threshold: refused")

    print("\n[2] z and the density")
    var z = List[Float64](length=2, fill=0.0)
    var e1 = exp(-0.5)                      # one kernel length away
    var at0: List[Float64] = [0.0, 0.0, 0.0]
    f.eval(at0, z)
    # the far starting pair's kernel at the origin: exp(-3 / 0.02) = 0 to the last bit
    t.truth(_near(z[0], (0.5 + 1.0 - 3.0 * e1) * 2.0, 1e-12) and _near(z[1], (-1.0 + 2.0 + 0.5 * e1) * 4.0, 1e-12), "at the first pair's place: its alpha, the second's by exp(-1/2)")
    var at1: List[Float64] = [0.1, 0.0, 0.0]
    f.eval(at1, z)
    t.truth(_near(z[0], (0.5 + e1 - 3.0) * 2.0, 1e-12) and _near(z[1], (-1.0 + 2.0 * e1 + 0.5) * 4.0, 1e-12), "at the second pair's place, the other way round")
    var far: List[Float64] = [1.0, 1.0, 1.0]
    f.eval(far, z)
    t.truth(_near(z[0], (0.5 + 7.0) * 2.0, 1e-12) and _near(z[1], (-1.0 + 7.0) * 4.0, 1e-12), "at the starting pair's place z is its alpha")
    t.truth(_near(f.density(far), 0.0, 1e-12), "and the density there is zero: a starting pair does not count")
    t.truth(_near(f.density(at0), 1.0 + e1, 1e-12), "the density at a success: itself and its neighbour's kernel")

    print("\n[3] the grade")
    # density on the x axis at a from the first pair: exp(-a^2/0.02) + exp(-(a-0.1)^2/0.02); at a = 0.05 it is 2 exp(-1/8) = 1.765
    var mid: List[Float64] = [0.05, 0.0, 0.0]
    var edge: List[Float64] = [-0.05, 0.0, 0.0]         # exp(-1/8) + exp(-9/8) = 1.207: over 0.8, under 1.5
    var out: List[Float64] = [-0.15, 0.0, 0.0]          # exp(-9/8) + exp(-25/8) = 0.369
    t.truth(f.grade(mid) == SM_REACH_SURE and f.grade(edge) == SM_REACH_LIKELY and f.grade(out) == SM_REACH_REFUSED, "sure between the two successes, likely at the side, refused further")
    t.truth(sm_reach_grade_name(SM_REACH_SURE) == "sure" and sm_reach_grade_name(SM_REACH_LIKELY) == "likely" and sm_reach_grade_name(SM_REACH_REFUSED) == "refused", "the grades' names")
    var none = SMReachInterface()
    t.truth(not none.loaded() and none.grade(mid) == SM_REACH_REFUSED, "an interface never loaded refuses")

    print("\n[4] the frames")
    var quarter = 1.5707963267948966
    var pelvis: List[Float64] = [1.0, 2.0, 0.8, quarter]
    var ahead: List[Float64] = [0.3, 0.0, 0.1]
    var w = sm_reach_world(pelvis, ahead)
    t.truth(_near(w[0], 1.0, 1e-12) and _near(w[1], 2.3, 1e-12) and _near(w[2], 0.9, 1e-12), "0.3 m ahead of a pelvis facing +y is 0.3 m along +y")
    var b = sm_reach_body(pelvis, w)
    t.truth(_near(b[0], 0.3, 1e-12) and _near(b[1], 0.0, 1e-12) and _near(b[2], 0.1, 1e-12), "and back")

    print("\n[5] the ramp")
    var still: List[Float64] = [0.0, 0.0, 0.0, 0.0]
    var z0: List[Float64] = [10.0, -10.0]
    var lp = SMReachLoop()
    t.truth(lp.start(f, mid, still, z0) == SM_REACH_SURE and lp.on and lp.z[0] == 10.0 and lp.z[1] == -10.0, "start: the grade, and z is still the call's")
    var zg = List[Float64](length=2, fill=0.0)
    f.eval(mid, zg)
    var ramp_ok = True
    var quiet_ok = True
    var hand_at: List[Float64] = [0.05, 0.0, 0.0]       # the hand exactly on the point: the loop will have nothing to correct
    for s in range(60):
        var ch = lp.step(f, still, hand_at)
        if s < 50 and s % 5 == 0:
            var u = sm_min_jerk(Float64(s + 5) / 50.0)
            ramp_ok = ramp_ok and ch and _near(lp.z[0], 10.0 + u * (zg[0] - 10.0), 1e-12) and _near(lp.z[1], -10.0 + u * (zg[1] + 10.0), 1e-12)
        else:
            quiet_ok = quiet_ok and not ch
    t.truth(ramp_ok, "a value every fifth step on the min-jerk curve, the first at step 0")
    t.truth(quiet_ok, "no change between them, nor from step 50 to the loop's first update")
    t.truth(_near(lp.z[0], zg[0], 1e-12) and _near(lp.z[1], zg[1], 1e-12), "the interface's z from step 45 on")
    t.truth(_near(sm_min_jerk(0.5), 0.5, 1e-15) and sm_min_jerk(0.0) == 0.0 and sm_min_jerk(1.0) == 1.0 and sm_min_jerk(1.7) == 1.0, "min-jerk: 0, a half at a half, 1, and 1 beyond")

    print("\n[6] the loop on an offset hand")
    # a wide interface so that nothing leaves the domain: one success of length 10 m
    var wide_path = String("/tmp/sm_reach_test_wide.txt")
    _write(wide_path, String("smreach 1 1 2 10.0 0.5 0.5\n0.0 0.0\n1.0 1.0\n0.0 0.0 0.0 1 1.0 1.0\n"))
    var fw = SMReachInterface.load(wide_path)
    var tg: List[Float64] = [0.3, -0.2, 0.1]
    var off: List[Float64] = [0.02, -0.03, 0.05]        # the hand sits this far from where it is commanded, in the body
    var turned: List[Float64] = [0.0, 0.0, 0.8, 0.0]
    var l2 = SMReachLoop()
    _ = l2.start(fw, tg, turned, z0)
    var conv_ok = True
    var first_update = -1
    var updates = 0
    for s in range(100 + 50 * 12 + 1):
        # the pelvis turns after the call: the error must be read in the heading frame of the moment
        var pv: List[Float64] = [0.0, 0.0, 0.8, 0.0 if s < 60 else 0.7]
        # the hand's place in the WORLD is the world point plus the offset and the correction turned back to the world, i.e.
        # in the present body frame: body(world point) + offset + correction
        var wb = sm_reach_body(pv, l2.world)
        var hd: List[Float64] = [wb[0] + off[0] + l2.corr[0], wb[1] + off[1] + l2.corr[1], wb[2] + off[2] + l2.corr[2]]
        var ch = l2.step(fw, pv, hd)
        if s >= 100 and (s - 100) % 50 == 0:
            if first_update < 0:
                first_update = s
            updates += 1
            var want = 1.0
            for _ in range(updates):
                want *= 0.7
            for k in range(3):
                conv_ok = conv_ok and ch and _near(l2.corr[k], -off[k] * (1.0 - want), 1e-12)
        elif s >= 50:
            conv_ok = conv_ok and not ch
    t.truth(first_update == 100 and updates == 13, "the first update 2 s after the call, then one a second")
    t.truth(conv_ok, "the correction is minus the offset times 1 - 0.7^k, read in the turned pelvis's frame")
    t.truth(_near(l2.correction(), 0.05 * (1.0 - 0.009688901040700001), 1e-9), "its largest component, reported")
    var l3 = SMReachLoop()
    _ = l3.start(fw, tg, turned, z0)
    var big: List[Float64] = [0.3 - 1.0, -0.2, 0.1 + 1.0]   # the hand a metre short on x, a metre high on z, whatever is commanded
    for _ in range(100 + 50 * 4 + 1):
        _ = l3.step(fw, turned, big)
    t.truth(_near(l3.corr[0], 0.15, 1e-12) and _near(l3.corr[2], -0.15, 1e-12) and _near(l3.corr[1], 0.0, 1e-12), "an error that stays: the correction stops at 0.15 m an axis")
    t.truth(not l3.gave, "and the loop has not given up: nothing left the domain")
    t.truth(_near(l3.error(turned, big), 1.4142135623730951, 1e-12), "the hand's distance to the world point")

    print("\n[7] the domain")
    # `f`'s domain along -x from the first success ends where the density falls to 0.8: the hand is held short on x, so the
    # correction pushes the command toward -x, out of the domain
    var l4 = SMReachLoop()
    t.truth(l4.start(f, edge, still, z0) == SM_REACH_LIKELY, "a target at the side starts, as likely")
    var short: List[Float64] = [-0.05 + 0.4, 0.0, 0.0]   # the hand 0.4 m in front of the point: the error is -0.4 on x
    var inside_ok = True
    var gave_at = -1
    for s in range(100 + 50 * 6 + 1):
        _ = l4.step(f, still, short)
        if s >= 100 and (s - 100) % 50 == 0:
            var q: List[Float64] = [edge[0] + l4.corr[0], edge[1] + l4.corr[1], edge[2] + l4.corr[2]]
            inside_ok = inside_ok and f.density(q) >= f.loop_threshold
            if l4.gave and gave_at < 0:
                gave_at = s
            if not l4.gave:
                inside_ok = inside_ok and l4.corr[0] < 0.0 and l4.corr[0] > -0.12   # pushed toward -x, held at the edge (an unprojected first update is -0.12)
    t.truth(inside_ok, "every update leaves the command inside the domain, at its edge")
    t.truth(gave_at == 100 + 50 * 4, "five projected updates in a row: the loop gives up at the fifth")
    var ze = List[Float64](length=2, fill=0.0)
    f.eval(edge, ze)
    t.truth(l4.gave and l4.corr[0] == 0.0 and l4.z[0] == ze[0] and l4.z[1] == ze[1], "the correction is dropped and z is the interface's at the target")
    var after = l4.z[0]
    var quiet = True
    for _ in range(120):
        quiet = quiet and not l4.step(f, still, short)
    t.truth(quiet and l4.z[0] == after, "and nothing changes any more")
    # a projected update followed by a clean one clears the count
    var l5 = SMReachLoop()
    _ = l5.start(f, edge, still, z0)
    for s in range(100 + 50 * 9 + 1):
        var k = (s - 100) // 50 if s >= 100 else -1
        var hd5 = short.copy()
        if k % 4 == 3:
            hd5[0] = edge[0] + l5.corr[0] - 0.2      # every fourth update the error pulls back inside
        _ = l5.step(f, still, hd5)
    t.truth(not l5.gave, "a clean update between projected ones: the count starts again, no giving up")

    print("\n[8] the refusal and the way back")
    var l6 = SMReachLoop()
    t.truth(l6.start(f, out, still, z0) == SM_REACH_REFUSED and not l6.on and not l6.step(f, still, out), "a refused target starts nothing")
    var l7 = SMReachLoop()
    t.truth(l7.start(none, mid, still, z0) == SM_REACH_REFUSED, "nor does an interface never loaded")
    var back = SMZRamp()
    var za: List[Float64] = [1.0, 2.0]
    var zb: List[Float64] = [3.0, -2.0]
    back.start(za, zb)
    var back_ok = back.on
    var n_ch = 0
    for s in range(50):
        var ch = back.step()
        if ch:
            n_ch += 1
            var u = sm_min_jerk(Float64(s + 5) / 50.0)
            back_ok = back_ok and s % 5 == 0 and _near(back.z[0], 1.0 + 2.0 * u, 1e-12) and _near(back.z[1], 2.0 - 4.0 * u, 1e-12)
    t.truth(back_ok and n_ch == 10, "the way back: ten values on the min-jerk curve")
    t.truth(not back.on and back.z[0] == 3.0 and back.z[1] == -2.0 and not back.step(), "it ends on the other z and stops")
    lp.stop()
    t.truth(not lp.on and not lp.step(f, still, hand_at), "a stopped loop does nothing")

    print("\n===", t.checks - t.fails, "/", t.checks, "passed ===")
    if t.fails > 0:
        raise Error("test_supermimic_reach: " + String(t.fails) + " failed")
