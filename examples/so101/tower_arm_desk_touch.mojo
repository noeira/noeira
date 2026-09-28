"""Where the follower's FK puts the jaw tip when the tip is ON THE DESK — and
which model error explains the difference.

    pixi run -e jetson mojo run -I . examples/so101/tower_arm_desk_touch.mojo \\
        --touches 14
    pixi run mojo run -I . examples/so101/tower_arm_desk_touch.mojo --selftest lift   # or jaw

WHY. Hand-guided touches on the brick (`tower_pose_arm_calib.mojo`, run 2,
2026-09-24) put the fixed jaw's tip 13.4 mm HIGHER in FK than where it
physically was, consistently, with the gripper vertical. Two causes fit that
run equally: a shoulder_lift zero off by ~+3.4 deg, or a real fixed jaw ~13
mm longer than the model's `gripperframe` site (the rig's jaw is the
one-piece wrist-camera mount print, not the stock part). A grasp planned
through FK would put the real jaw ~13 mm lower than planned, so it must be
settled before the expert drives the real arm (OBJECT_POSE_POLICY_PLAN.md,
step 2). No camera is used here: the desk's height is the reference.

How the touches separate the causes:
- a LIFT zero error moves the tip vertically in proportion to the REACH
  (horizontal distance from the shoulder): touch near AND far;
- a JAW LENGTH error moves it by length x cos(tilt): the same at every reach
  with the gripper vertical, less when tilted;
- a WRIST_FLEX zero error moves it sideways-then-down when the gripper is
  TILTED: add tilted touches (noeira-26's suggestion).

## THIS PROGRAM NEVER ENERGISES THE ARM

Torque is released at start and again in a `finally` (if it dies hard:
`pixi run soarm-torque-off`). You move the follower BY HAND.

## The protocol — nothing to press

Put the tip of the FIXED jaw (the jaw that does not move) on the bare desk
and hold still ~1 s; the tool prints `TOUCH n`. Lift the arm, go to the next
pose. A new touch must be >= 3 cm from the previous one or tilted >= 15 deg
differently. Aim for (the tool prints what is still missing):
- gripper VERTICAL (< 15 deg) as NEAR as the arm allows (~20 cm from the
  shoulder) and as FAR (~28-32 cm), on the left, centre and right;
- gripper TILTED 25-50 deg (forward, backward, sideways).
Past 50 deg the jaw's FLANK, not its tip, meets the desk: such touches are
ignored. The suggestions are not requirements: `--touches` of any poses end
the session. The file is rewritten after every touch (Ctrl-C loses nothing;
the fit then runs offline from it). Jaws closed or open does not matter.

## The fit

Residual per touch: FK tip z - `--desk-z` (0.0 = the plate's underside: the
arm stands on its 5 mm base plate on the desk; the base frame is at z 0.005).
Models, each least squares with its LEAVE-ONE-OUT residual:
none | jaw length | lift | lift+elbow | lift+elbow+wflex | length+lift |
length+lift+wflex. A model earns its parameters only if its leave-one-out
residual drops. ⚠ A constant desk-height error looks like a jaw-length error
on vertical touches: the tilted touches separate them, and a RULER on the
printed jaw (wrist_flex horn AXIS -> fixed jaw tip; the model says 165.7 mm
since e9bd63b00 put `gripperframe` at the mesh tip, 159.4 before) settles the
length directly — measured from the axis, not the horn's outer face.

Written: `--out` (default `projects/so101-tower/calibration/desk_touch.txt`), one line
per touch: tip xyz, tilt, reach, the joints (model rad, the 1 s window's
mean), raw ticks.
"""

from std.math import acos, sqrt, pi
from std.sys import argv, exit
from std.time import perf_counter_ns

from noeira.core.concurrent.thread import sleep_us
from noeira.math3d import Mat3 as Mat3Generic, Vec3 as Vec3Generic
from noeira.robot.so101 import SO101Arm, SO101_N
from noeira.robot.so101.ports import follower_port
from noeira.robot.so101.sim_map import SimJointMap
from noeira.tasks.so101_tower_overhead import TowerArmFK
from noeira.utils.fmt import fixed

comptime Vec3d = Vec3Generic[DType.float64]
comptime Mat3d = Mat3Generic[DType.float64]

comptime TIP_STILL = 0.0015
comptime NEAR_DESK = 0.06
"""FK tip below this (world z) counts as a touch candidate."""
comptime REARM = 0.02
"""Lift the tip this far above the last touch before the next one counts."""
comptime NEW_XY = 0.03
comptime NEW_TILT_DEG = 15.0
comptime MAX_TILT_DEG = 50.0
"""Past this the jaw's flank, not its tip, meets the desk (2026-09-24: touches
at 74-102 deg read 26-55 mm above the desk)."""
comptime VERT_DEG = 15.0
comptime NEAR_REACH = 0.22
comptime FAR_REACH = 0.28
"""Reachable with a vertical gripper: not much closer than ~0.20 m to the
shoulder, and not past ~0.30-0.33 m (the expert's own IK note)."""
comptime PAN_AXIS_X = 0.0388353
"""World x of the shoulder-pan axis (`so_arm101_tower.xml` body `shoulder`)."""

comptime P_LEN = -1
"""Parameter code: the jaw extension along the gripper's -z (m). A joint
offset is coded by its joint index 0..4 (rad)."""


struct Touches(Movable):
    var q: List[List[Float64]]

    def __init__(out self):
        self.q = List[List[Float64]]()


def _tip(mut fk: TowerArmFK, site: Int, q: List[Float64], dq: List[Float64], ext: Float64) raises -> Vec3d:
    var qq = List[Float64](length=6, fill=0.0)
    for k in range(6):
        qq[k] = q[k] + (dq[k] if k < 5 else 0.0)
    fk.set_qpos(qq)
    var p = fk.site_pos(site)
    if ext != 0.0:
        var r = fk.site_body_rot(site)
        p = p - r.col(2) * ext
    return p


def _residuals(
    mut fk: TowerArmFK, site: Int, ref T: Touches, codes: List[Int],
    p: List[Float64], desk_z: Float64, skip: Int,
) raises -> List[Float64]:
    var dq = List[Float64](length=5, fill=0.0)
    var ext = 0.0
    for i in range(len(codes)):
        if codes[i] == P_LEN:
            ext = p[i]
        else:
            dq[codes[i]] = p[i]
    var r = List[Float64]()
    for t in range(len(T.q)):
        if t == skip:
            continue
        r.append(Float64(_tip(fk, site, T.q[t], dq, ext).z) - desk_z)
    return r^


def _solve(ref A: List[List[Float64]], ref b: List[Float64]) -> List[Float64]:
    """Gaussian elimination with partial pivoting (n <= 4)."""
    var n = len(b)
    var M = List[List[Float64]]()
    for i in range(n):
        var row = A[i].copy()
        row.append(b[i])
        M.append(row^)
    for c in range(n):
        var piv = c
        for r in range(c + 1, n):
            if abs(M[r][c]) > abs(M[piv][c]):
                piv = r
        if piv != c:
            var tmp = M[c].copy()
            M[c] = M[piv].copy()
            M[piv] = tmp^
        var d = M[c][c]
        if abs(d) < 1e-18:
            continue
        for r in range(n):
            if r == c:
                continue
            var f = M[r][c] / d
            for k in range(c, n + 1):
                M[r][k] -= f * M[c][k]
    var x = List[Float64](length=n, fill=0.0)
    for i in range(n):
        if abs(M[i][i]) > 1e-18:
            x[i] = M[i][n] / M[i][i]
    return x^


def _fit(
    mut fk: TowerArmFK, site: Int, ref T: Touches, codes: List[Int],
    desk_z: Float64, skip: Int = -1,
) raises -> List[Float64]:
    """Gauss-Newton, numeric Jacobian."""
    var n = len(codes)
    var p = List[Float64](length=n, fill=0.0)
    if n == 0:
        return p^
    for _ in range(15):
        var r0 = _residuals(fk, site, T, codes, p, desk_z, skip)
        var J = List[List[Float64]]()
        for j in range(n):
            var h = 1e-5
            var pj = p.copy()
            pj[j] += h
            var rj = _residuals(fk, site, T, codes, pj, desk_z, skip)
            var col = List[Float64]()
            for k in range(len(r0)):
                col.append((rj[k] - r0[k]) / h)
            J.append(col^)
        var A = List[List[Float64]]()
        var g = List[Float64]()
        for a in range(n):
            var row = List[Float64]()
            for b in range(n):
                var s = 0.0
                for k in range(len(r0)):
                    s += J[a][k] * J[b][k]
                row.append(s)
            A.append(row^)
            var s = 0.0
            for k in range(len(r0)):
                s -= J[a][k] * r0[k]
            g.append(s)
        var dp = _solve(A, g)
        var big = 0.0
        for j in range(n):
            p[j] += dp[j]
            big = max(big, abs(dp[j]))
        if big < 1e-9:
            break
    return p^


def _rms(r: List[Float64]) -> Float64:
    var s = 0.0
    for x in r:
        s += x * x
    return sqrt(s / Float64(max(1, len(r))))


def _describe(codes: List[Int], p: List[Float64]) -> String:
    var names: List[String] = ["pan", "lift", "elbow", "wflex", "wroll"]
    var s = String("")
    for i in range(len(codes)):
        if i > 0:
            s += ", "
        if codes[i] == P_LEN:
            s += "jaw +" + fixed(p[i] * 1000.0, 1) + " mm"
        else:
            s += names[codes[i]] + " " + fixed(p[i] * 180.0 / pi, 2) + " deg"
    return s if s != "" else String("(no parameter)")


def _zt(
    mut fk: TowerArmFK, tip: Int, true_dq: List[Float64], true_ext: Float64,
    pitch_sum: Float64, e: Float64, mut q: List[Float64],
) raises -> Float64:
    """The TRUE tip's z with the elbow at `e` and the wrist flex keeping
    lift + elbow + flex = `pitch_sum`."""
    q[2] = e
    q[3] = pitch_sum - q[1] - e
    return Float64(_tip(fk, tip, q, true_dq, true_ext).z)


def _synthetic(
    mut fk: TowerArmFK, tip: Int, true_dq: List[Float64], true_ext: Float64,
    desk_z: Float64, mut T: Touches, mut tilts: List[Float64], mut reaches: List[Float64],
) raises:
    """Touches a TRUE arm (joints q + true_dq, jaw + true_ext) makes on the
    desk: per pose, pan/lift/pitch chosen, the elbow solved (secant) so the
    TRUE tip is at desk_z; what is recorded is the servo's q."""
    comptime C_VERT = 1.55
    """lift + elbow + wrist_flex for a vertical gripper (the rig's run 2)."""
    var pans: List[Float64] = [-0.5, 0.0, 0.5]
    var lifts: List[Float64] = [-0.9, -0.2, 0.5]
    var pitch: List[Float64] = [0.0, 0.0, 0.0, 0.5, -0.5]
    for a in range(len(pans)):
        for b in range(len(lifts)):
            var pv = pitch[(a + b) % len(pitch)] if (a + b) % 3 == 0 else 0.0
            var q = List[Float64](length=6, fill=0.0)
            q[0] = pans[a]
            q[1] = lifts[b]
            q[4] = 1.5
            var e0 = -1.2
            var e1 = 1.2
            var f0 = _zt(fk, tip, true_dq, true_ext, C_VERT + pv, e0, q) - desk_z
            var f1 = _zt(fk, tip, true_dq, true_ext, C_VERT + pv, e1, q) - desk_z
            var ok = False
            for _ in range(60):
                if abs(f1 - f0) < 1e-15:
                    break
                var e2 = e1 - f1 * (e1 - e0) / (f1 - f0)
                e0 = e1
                f0 = f1
                e1 = max(-1.7, min(1.7, e2))
                f1 = _zt(fk, tip, true_dq, true_ext, C_VERT + pv, e1, q) - desk_z
                if abs(f1) < 1e-9:
                    ok = True
                    break
            if not ok:
                continue
            _ = _zt(fk, tip, true_dq, true_ext, C_VERT + pv, e1, q)
            var zero = List[Float64](length=5, fill=0.0)
            var p = _tip(fk, tip, q, zero, 0.0)
            var r = fk.site_body_rot(tip)
            T.q.append(q.copy())
            tilts.append(acos(min(1.0, max(-1.0, Float64(r.col(2).z)))) * 180.0 / pi)
            reaches.append(sqrt((Float64(p.x) - PAN_AXIS_X) ** 2 + Float64(p.y) ** 2))


def _selftest(which: String) raises:
    """`--selftest lift` (truth: lift zero +3.4 deg) or `--selftest jaw`
    (truth: jaw +13 mm): synthetic touches, then the same report."""
    var fk = TowerArmFK()
    var tip = fk.site_index("gripperframe")
    var dq = List[Float64](length=5, fill=0.0)
    var ext = 0.0
    if which == "lift":
        dq[1] = 3.4 * pi / 180.0
    elif which == "jaw":
        ext = 0.013
    else:
        raise Error("--selftest lift|jaw")
    var T = Touches()
    var tilts = List[Float64]()
    var reaches = List[Float64]()
    _synthetic(fk, tip, dq, ext, 0.0, T, tilts, reaches)
    print("SELFTEST truth:", which, "—", len(T.q), "synthetic touches")
    _report(fk, tip, T, tilts, reaches, 0.0)


def _report(
    mut fk: TowerArmFK, tip: Int, ref T: Touches, tilts: List[Float64],
    reaches: List[Float64], desk_z: Float64,
) raises:
    var nt = len(T.q)
    # ── the models ───────────────────────────────────────────────────────
    var models = List[List[Int]]()
    models.append(List[Int]())
    models.append([P_LEN])
    models.append([1])
    models.append([1, 2])
    models.append([1, 2, 3])
    models.append([P_LEN, 1])
    models.append([P_LEN, 1, 3])
    print("\nFK tip z vs the desk (mm), per touch:")
    var r_none = _residuals(fk, tip, T, List[Int](), List[Float64](), desk_z, -1)
    for k in range(nt):
        print(
            "  touch", k + 1, ": reach", fixed(reaches[k] * 100.0, 1), "cm tilt",
            fixed(tilts[k], 0), "deg -> +", fixed(r_none[k] * 1000.0, 1), "mm",
        )
    print("\nmodel                                          fit rms   leave-one-out rms (mm)")
    for mi in range(len(models)):
        var codes = models[mi].copy()
        if len(codes) >= nt - 1:
            continue
        var p = _fit(fk, tip, T, codes, desk_z)
        var r = _residuals(fk, tip, T, codes, p, desk_z, -1)
        var lo_s = 0.0
        for k in range(nt):
            var pk = _fit(fk, tip, T, codes, desk_z, skip=k)
            var dq = List[Float64](length=5, fill=0.0)
            var ext = 0.0
            for j in range(len(codes)):
                if codes[j] == P_LEN:
                    ext = pk[j]
                else:
                    dq[codes[j]] = pk[j]
            var e = Float64(_tip(fk, tip, T.q[k], dq, ext).z) - desk_z
            lo_s += e * e
        print(
            "  ", _describe(codes, p), "  ->", fixed(_rms(r) * 1000.0, 1), "   ",
            fixed(sqrt(lo_s / Float64(nt)) * 1000.0, 1),
        )


def _offline(path: String, desk_z: Float64) raises:
    """`--from FILE`: the report on a saved session, touches past
    MAX_TILT_DEG dropped (their jaw FLANK met the desk)."""
    var fk = TowerArmFK()
    var tip = fk.site_index("gripperframe")
    var T = Touches()
    var tilts = List[Float64]()
    var reaches = List[Float64]()
    var dropped = 0
    with open(path, "r") as fh:
        var text = fh.read()
        for line in text.split("\n"):
            var l = String(line).strip()
            if l.byte_length() == 0 or l.startswith("#"):
                continue
            var parts = l.split("|")
            var tr = parts[1].strip().split(" ")
            var tilt = Float64(String(tr[0]))
            var reach = Float64(String(tr[len(tr) - 1]))
            if tilt > MAX_TILT_DEG:
                dropped += 1
                continue
            var q = List[Float64]()
            for s in parts[2].strip().split(" "):
                var st = String(s).strip()
                if st.byte_length() > 0:
                    q.append(Float64(st))
            T.q.append(q^)
            tilts.append(tilt)
            reaches.append(reach)
    print(path, ":", len(T.q), "touches kept,", dropped, "dropped (tilt >", MAX_TILT_DEG, "deg)")
    if len(T.q) < 4:
        print("too few touches to fit")
        return
    _report(fk, tip, T, tilts, reaches, desk_z)


def main() raises:
    var args = argv()
    var port = String("")
    var n_touch = 14
    var seconds = 900.0
    var desk_z = 0.0
    var out_path = String("projects/so101-tower/calibration/desk_touch.txt")
    var from_path = String("")
    var i = 1
    while i < len(args):
        var a = String(args[i])
        if i + 1 >= len(args):
            raise Error("flag " + a + " needs a value")
        var v = String(args[i + 1])
        if a == "--port":
            port = v
        elif a == "--touches":
            n_touch = Int(v)
        elif a == "--seconds":
            seconds = Float64(v)
        elif a == "--desk-z":
            desk_z = Float64(v)
        elif a == "--out":
            out_path = v
        elif a == "--selftest":
            _selftest(v)
            return
        elif a == "--from":
            from_path = v
        else:
            raise Error("unknown flag " + a)
        i += 2

    if from_path != "":
        _offline(from_path, desk_z)
        return

    var fk = TowerArmFK()
    var tip = fk.site_index("gripperframe")
    var lo = Array[Float64, SO101_N](fill=0.0)
    var hi = Array[Float64, SO101_N](fill=0.0)
    for k in range(SO101_N):
        lo[k] = fk.lo[k]
        hi[k] = fk.hi[k]
    var the_port = follower_port(port)
    print("opening", the_port, "...")
    var arm = SO101Arm(the_port, max_step_ticks=0)
    arm.bus.timeout_ms = 20
    arm.set_torque(False)
    print("  torque RELEASED — move the follower by hand")
    var jmap = SimJointMap.tower_follower(arm.cal, lo^, hi^)
    print("  " + jmap.describe())
    print("  desk z (world) =", desk_z, "m")

    var raw = List[Int32](length=SO101_N, fill=Int32(0))
    var q = List[Float64](length=6, fill=0.0)
    var win_t = List[Float64]()
    var win_p = List[Vec3d]()
    var win_q = List[List[Float64]]()
    var T = Touches()
    var tips = List[Vec3d]()
    var tilts = List[Float64]()
    var reaches = List[Float64]()
    var log = String("# tip_x tip_y tip_z | tilt_deg reach_m | q0..q5 (model rad, 1 s mean) | raw0..raw5\n")
    var t0 = perf_counter_ns()
    var armed = True
    var last_z = -1.0
    print("\nTOUCH 1/", n_touch, ": fixed jaw's tip on the bare desk, hold still ~1 s")
    try:
        while len(T.q) < n_touch:
            var now = Float64(perf_counter_ns() - t0) * 1e-9
            if now > seconds:
                print("time is up")
                break
            if arm.read_positions(Span(raw)) != SO101_N:
                _ = sleep_us(5000)
                continue
            for k in range(6):
                q[k] = jmap.to_sim_unclamped(arm.cal, k, raw[k])
            fk.set_qpos(q)
            var p = fk.site_pos(tip)
            var r = fk.site_body_rot(tip)
            var tilt = acos(min(1.0, max(-1.0, Float64(r.col(2).z)))) * 180.0 / pi
            # re-armed once the tip has been lifted REARM above the last
            # touch (it was "above NEAR_DESK" — 6 cm in FK — which a vertical
            # touch-to-touch move rarely reached: the rig session of
            # 2026-09-24 could not trigger vertical touches)
            if not armed and Float64(p.z) > last_z + REARM:
                armed = True
            if Float64(p.z) > NEAR_DESK:
                win_t.clear()
                win_p.clear()
                win_q.clear()
                _ = sleep_us(10000)
                continue
            win_t.append(now)
            win_p.append(p)
            win_q.append(q.copy())
            while len(win_t) > 0 and win_t[0] < now - 1.0:
                _ = win_t.pop(0)
                _ = win_p.pop(0)
                _ = win_q.pop(0)
            var n = Float64(len(win_p))
            var mx = 0.0
            var my = 0.0
            var mz = 0.0
            for w in win_p:
                mx += Float64(w.x)
                my += Float64(w.y)
                mz += Float64(w.z)
            mx /= n
            my /= n
            mz /= n
            var drift = 0.0
            for w in win_p:
                drift = max(drift, sqrt(
                    (Float64(w.x) - mx) ** 2 + (Float64(w.y) - my) ** 2 + (Float64(w.z) - mz) ** 2
                ))
            if not (armed and now - win_t[0] >= 0.9 and len(win_p) >= 10 and drift <= TIP_STILL):
                _ = sleep_us(10000)
                continue
            # new enough?
            var fresh = True
            if len(tips) > 0:
                var lp = tips[len(tips) - 1]
                var dxy = sqrt((mx - Float64(lp.x)) ** 2 + (my - Float64(lp.y)) ** 2)
                if dxy < NEW_XY and abs(tilt - tilts[len(tilts) - 1]) < NEW_TILT_DEG:
                    fresh = False
            if not fresh:
                continue
            if tilt > MAX_TILT_DEG:
                if armed:
                    print(
                        "  (ignored: gripper tilted", fixed(tilt, 0), "deg — past",
                        MAX_TILT_DEG, "the jaw's FLANK touches, not its tip)",
                    )
                armed = False
                last_z = mz
                continue
            var qm = List[Float64](length=6, fill=0.0)
            for qs in win_q:
                for k in range(6):
                    qm[k] += qs[k] / n
            var reach = sqrt((mx - PAN_AXIS_X) ** 2 + my ** 2)
            T.q.append(qm.copy())
            tips.append(Vec3d(mx, my, mz))
            tilts.append(tilt)
            reaches.append(reach)
            log += fixed(mx, 5) + " " + fixed(my, 5) + " " + fixed(mz, 5) + " | " + fixed(tilt, 1) + " " + fixed(reach, 4) + " |"
            for k in range(6):
                log += " " + fixed(qm[k], 5)
            log += " |"
            for k in range(6):
                log += " " + String(Int(raw[k]))
            log += "\n"
            armed = False
            last_z = mz
            # written after EVERY touch: a Ctrl-C must not lose the session
            with open(out_path, "w") as fh:
                fh.write(log)
            print(
                "  TOUCH", len(T.q), ": FK tip z", fixed((mz - desk_z) * 1000.0, 1),
                "mm above the desk | reach", fixed(reach * 100.0, 1), "cm, tilt",
                fixed(tilt, 0), "deg, at (", fixed(mx * 1000.0, 0), ",", fixed(my * 1000.0, 0), ") mm",
            )
            # what is still missing
            var nv = 0
            var nt = 0
            var nnear = 0
            var nfar = 0
            for k in range(len(tilts)):
                if tilts[k] < VERT_DEG:
                    nv += 1
                    if reaches[k] < NEAR_REACH:
                        nnear += 1
                    if reaches[k] > FAR_REACH:
                        nfar += 1
                elif tilts[k] > 25.0:
                    nt += 1
            var need = String("")
            if nnear < 3:
                need += " vertical<15deg-reach<22cm(" + String(nnear) + "/3)"
            if nfar < 3:
                need += " vertical<15deg-reach>28cm(" + String(nfar) + "/3)"
            if nt < 3:
                need += " tilted-25..50deg(" + String(nt) + "/3)"
            if len(T.q) < n_touch:
                print(
                    "  lift the arm; next pose (" + String(n_touch - len(T.q))
                    + " to go — any pose counts; suggestions:" + (need if need != "" else " none") + ")"
                )
    finally:
        try:
            arm.set_torque(False)
        except:
            print("⚠ COULD NOT RELEASE TORQUE — run `pixi run soarm-torque-off`")

    var nt = len(T.q)
    with open(out_path, "w") as fh:
        fh.write(log)
    print("\nwrote", out_path, "(", nt, "touches )")
    if nt < 4:
        print("too few touches to fit")
        return

    _report(fk, tip, T, tilts, reaches, desk_z)
    print(
        "\nA model earns its parameters only if its LEAVE-ONE-OUT drops. Compare"
        " the jaw length with a ruler: wrist_flex horn AXIS -> fixed jaw tip,"
        " model 165.7 mm (gripperframe at the mesh tip, e9bd63b00)."
    )
