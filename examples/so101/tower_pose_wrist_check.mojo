"""The pose estimator on the WRIST camera of a real LeRobot recording, checked
against the overhead camera — step one of a look-then-move grasp.

    pixi run mojo run -I . examples/so101/tower_pose_wrist_check.mojo \\
        --root projects/so101-tower/datasets/cube-in-bowl-printed [--episodes 0:20] \\
        [--hue-hist] [--snap /tmp/wrist_snaps] [--wrist-hsv H,TOL,SMIN,VMIN,VMAX]

## Why

The overhead camera reads the brick to ~5 mm on the desk, and a brick in
the bowl worse (the wall hides its lower part). From the pre-grasp pose
(~10 cm away) the wrist camera sees the brick at ~10x the resolution, from
the gripper itself — so its error is the brick RELATIVE to the jaws, which is
what the grasp needs. The planned use: stop at the pre-grasp, read the brick
through the wrist camera, re-plan the last approach.

## What it measures, with no ground truth recorded

The wrist camera's world pose at frame t is the arm's FK at that frame's
`observation.state` (the follower's joint zero, `So101TowerUnits`) composed
with the model's `wrist_cam` (the mount measured in 5790c3570) —
`TowerArmFK.camera_pose`. Its lens is `camera_wrist.txt` (fisheye). Until the
grasp nothing moves the brick, so every wrist estimate before the grasp can
be compared with the overhead camera's estimate over the still start:
- the error (x, y) and the yaw difference (mod 90), per frame and per
  episode;
- against the camera's distance to the brick and the arm's speed (a moving
  arm is where a frame/state skew shows);
- how often the wrist estimate is confident at all (the jaws can cover it).
A constant error is the mount (or the overhead camera's own offset); an
error growing with speed is timing; one growing with distance is the lens or
the mount's rotation.

The wrist estimate searches only within `--roi-mm` (80) of the overhead
reference: the executor will know roughly where the brick is, and the wrist
camera also sees the blue tower stand.

`--hue-hist` prints the hue histogram of the pixels whose desk ray lands
within 30 mm of the reference, in the first episode's closest pre-grasp wrist
frame — how to set `--wrist-hsv`. `--snap DIR` writes, per episode, that
frame with the overhead reference's outline (green) and the wrist estimate's
(red).
"""

from std.math import atan2, cos, pi, sin, sqrt, round
from std.os import makedirs
from std.sys import argv

from noeira.data.lerobot import LeRobotInfo, EpisodeIndex, FrameTable, CameraStream
from noeira.io.png import save_png
from noeira.math3d import Mat3 as Mat3Generic, Vec3 as Vec3Generic
from noeira.tasks.so101_tower_overhead import (
    OVERHEAD_CALIB, DESK_Z, tower_overhead_pose, tower_desk_roi, printed_brick_hsv,
    pose_confident, TowerArmFK,
)
from noeira.tasks.so101_tower_rig import So101TowerUnits, RIG_JOINT_ZERO_FOLLOWER
from noeira.utils.fmt import fixed
from noeira.vision.calib_file import read_calib
from noeira.vision.fisheye import FisheyeLens
from noeira.vision.tabletop_pose import (
    RigCamera, ColorClass, PrismModel, DeskROI, PoseEstimate, estimate_prism_pose,
    model_silhouette, rgb_to_hsv, P2,
)

comptime Vec3d = Vec3Generic[DType.float64]
comptime Mat3d = Mat3Generic[DType.float64]
comptime WRIST_CALIB = "projects/so101-tower/cameras/camera_wrist.txt"
comptime N_ARM = 6
comptime GRIPPER = 5
comptime MOVE_EPS = 1.5
comptime GRASP_DROP = 6.0
comptime GRASP_SETTLE = 0.6
comptime MAX_IDLE = 60
comptime SLOW_DEG: Float64 = 1.0
"""`--fit-zeros` uses frames where no arm joint moved more than this since
the previous frame (deg): a moving arm adds frame/state skew."""


def _floats(s: String) raises -> List[Float64]:
    var out = List[Float64]()
    for p in s.split(","):
        out.append(Float64(String(p)))
    return out^


def _sorted(var xs: List[Float64]) -> List[Float64]:
    for i in range(1, len(xs)):
        var p = xs[i]
        var j = i - 1
        while j >= 0 and xs[j] > p:
            xs[j + 1] = xs[j]
            j -= 1
        xs[j + 1] = p
    return xs^


def _pct(xs: List[Float64], q: Float64) -> Float64:
    if len(xs) == 0:
        return 0.0
    var s = _sorted(xs.copy())
    return s[Int(q * Float64(len(s) - 1) + 0.5)]


def _yaw_diff_deg(a: Float64, b: Float64) -> Float64:
    """|a - b| on the cube's 90-degree circle, degrees."""
    var d = (a - b) * 180.0 / pi
    while d > 45.0:
        d -= 90.0
    while d < -45.0:
        d += 90.0
    return abs(d)


def _put(mut img: List[UInt8], w: Int, h: Int, u: Int, v: Int, r: UInt8, g: UInt8, b: UInt8):
    if u < 0 or v < 0 or u >= w or v >= h:
        return
    var k = (v * w + u) * 3
    img[k] = r
    img[k + 1] = g
    img[k + 2] = b


def _outline(
    mut img: List[UInt8], cam: RigCamera, model: PrismModel, x: Float64, y: Float64,
    yaw: Float64, r: UInt8, g: UInt8, b: UInt8,
):
    var poly = model_silhouette(cam, model, DESK_Z, x, y, yaw)
    for i in range(len(poly)):
        var a = poly[i]
        var c = poly[(i + 1) % len(poly)]
        var n = Int(max(abs(c[0] - a[0]), abs(c[1] - a[1]))) + 1
        for s in range(n + 1):
            var t = Float64(s) / Float64(n)
            _put(img, cam.width, cam.height, Int(round(a[0] + t * (c[0] - a[0]))),
                 Int(round(a[1] + t * (c[1] - a[1]))), r, g, b)


def _idle_end(table: FrameTable, g0: Int, length: Int) -> Int:
    var sd = table.state_dim
    for t in range(1, min(length, MAX_IDLE)):
        for k in range(N_ARM):
            if abs(Float64(table.qpos[(g0 + t) * sd + k]) - Float64(table.qpos[g0 * sd + k])) > MOVE_EPS:
                return t
    return min(length, MAX_IDLE)


def _grasp_frame(table: FrameTable, g0: Int, length: Int, start: Int) -> Int:
    """`tower_pose_real_check`'s rule: the gripper dropped GRASP_DROP below
    its running maximum, then settled."""
    var sd = table.state_dim
    var run_max = -1.0e30
    for t in range(start, length - 6):
        var g = Float64(table.qpos[(g0 + t) * sd + GRIPPER])
        run_max = max(run_max, g)
        if g <= run_max - GRASP_DROP:
            for t2 in range(t, length - 6):
                var a = Float64(table.qpos[(g0 + t2) * sd + GRIPPER])
                var b = Float64(table.qpos[(g0 + t2 + 5) * sd + GRIPPER])
                if abs(a - b) < GRASP_SETTLE:
                    return t2
            return -1
    return -1


def _hue_hist(frame: List[UInt8], cam: RigCamera, cx: Float64, cy: Float64):
    var bins = List[Int](length=36, fill=0)
    var n = 0
    for v in range(cam.height):
        for u in range(cam.width):
            var k = (v * cam.width + u) * 3
            var hsv = rgb_to_hsv(frame[k], frame[k + 1], frame[k + 2])
            if hsv[1] < 0.3 or hsv[2] < 0.15:
                continue
            var p = cam.plane_point(Float64(u), Float64(v), DESK_Z)
            if not p[2] or sqrt((p[0] - cx) ** 2 + (p[1] - cy) ** 2) > 0.03:
                continue
            bins[min(35, Int(hsv[0] / 10.0))] += 1
            n += 1
    print("wrist hue histogram,", n, "saturated pixels within 30 mm of the reference:")
    for b in range(36):
        if bins[b] == 0:
            continue
        var bar = String("")
        for _ in range(min(60, bins[b] // 20 + 1)):
            bar += "#"
        print("  ", b * 10, "-", b * 10 + 10, "deg", bins[b], bar)


struct Samples(Movable):
    """Per confident wrist frame: the estimated brick centre's PIXEL (through
    the nominal camera), the camera's parent body pose, the overhead
    reference, the episode."""

    var u: List[Float64]
    var v: List[Float64]
    var bp: List[Vec3d]
    var br: List[Mat3d]
    var rx: List[Float64]
    var ry: List[Float64]
    var ep: List[Int]
    var q: List[List[Float64]]
    """The frame's model joints (the follower zero)."""
    var speed: List[Float64]
    var arm_lift: List[Float64]
    var arm_elbow: List[Float64]
    """The horizontal reach (m) from the upper arm's / the lower arm's origin
    to the gripper at the frame's pose: the gravity load's lever about the
    lift / the elbow (a SAG proxy, noeira-26)."""

    def __init__(out self):
        self.u = List[Float64]()
        self.v = List[Float64]()
        self.bp = List[Vec3d]()
        self.br = List[Mat3d]()
        self.rx = List[Float64]()
        self.ry = List[Float64]()
        self.ep = List[Int]()
        self.q = List[List[Float64]]()
        self.speed = List[Float64]()
        self.arm_lift = List[Float64]()
        self.arm_elbow = List[Float64]()


def _rot_small(a: Float64, b: Float64, c: Float64) -> Mat3d:
    """Rotation by the vector (a, b, c) (rad), Rodrigues."""
    var th = sqrt(a * a + b * b + c * c)
    if th < 1e-12:
        return Mat3d.identity()
    var k = Vec3d(a / th, b / th, c / th)
    var K = Mat3d.from_cols(Vec3d(0.0, k.z, -k.y), Vec3d(-k.z, 0.0, k.x), Vec3d(k.y, -k.x, 0.0))
    return Mat3d.identity() + K * sin(th) + (K @ K) * (1.0 - cos(th))


def _residual(
    ref smp: Samples, i: Int, lens: FisheyeLens, cpos: Vec3d, crot: Mat3d,
    p: List[Float64],
) -> Tuple[Float64, Float64]:
    """Sample i's (x, y) error (m) with the camera-in-body correction `p`
    (rot 0..2, pos 3..5 in the body frame; world shift 6..7 of the
    reference)."""
    var R = _rot_small(p[0], p[1], p[2])
    var pos = smp.bp[i] + smp.br[i] * (cpos + Vec3d(p[3], p[4], p[5]))
    var rot = smp.br[i] @ (R @ crot)
    var cam = RigCamera(lens, pos, rot)
    var pt = cam.plane_point(smp.u[i], smp.v[i], DESK_Z + 0.0125)
    if not pt[2]:
        return (1.0, 1.0)
    return (pt[0] - (smp.rx[i] + p[6]), pt[1] - (smp.ry[i] + p[7]))


def _free(model: Int) -> List[Int]:
    """The free parameters of each model: 0 the world shift only (the
    overhead's own offset); 1 + the camera's rotation; 2 + its position."""
    if model == 0:
        return [6, 7]
    if model == 1:
        return [0, 1, 2, 6, 7]
    return [0, 1, 2, 3, 4, 5, 6, 7]


def _fit(
    ref smp: Samples, lens: FisheyeLens, cpos: Vec3d, crot: Mat3d, skip_ep: Int, model: Int,
) -> List[Float64]:
    """Gauss-Newton on the samples not in episode `skip_ep`, the model's free
    parameters (the rest held at 0)."""
    var free = _free(model)
    var n_par = len(free)
    var p = List[Float64](length=8, fill=0.0)
    for _ in range(12):
        var H = List[Float64](length=64, fill=0.0)
        var g = List[Float64](length=8, fill=0.0)
        for i in range(len(smp.u)):
            if smp.ep[i] == skip_ep:
                continue
            var r0 = _residual(smp, i, lens, cpos, crot, p)
            var J = List[Float64](length=16, fill=0.0)
            for j in range(n_par):
                var pj = p.copy()
                var h = 1e-5
                pj[free[j]] += h
                var r1 = _residual(smp, i, lens, cpos, crot, pj)
                J[j] = (r1[0] - r0[0]) / h
                J[8 + j] = (r1[1] - r0[1]) / h
            for a in range(n_par):
                g[a] += J[a] * r0[0] + J[8 + a] * r0[1]
                for b in range(n_par):
                    H[a * 8 + b] += J[a] * J[b] + J[8 + a] * J[8 + b]
        for a in range(n_par):
            H[a * 8 + a] += 1e-9
        # solve H dp = -g (n_par x n_par, Gauss-Jordan)
        var A = List[Float64](length=n_par * (n_par + 1), fill=0.0)
        for a in range(n_par):
            for b in range(n_par):
                A[a * (n_par + 1) + b] = H[a * 8 + b]
            A[a * (n_par + 1) + n_par] = -g[a]
        for c in range(n_par):
            var piv = c
            for r in range(c + 1, n_par):
                if abs(A[r * (n_par + 1) + c]) > abs(A[piv * (n_par + 1) + c]):
                    piv = r
            for k in range(n_par + 1):
                var t = A[c * (n_par + 1) + k]
                A[c * (n_par + 1) + k] = A[piv * (n_par + 1) + k]
                A[piv * (n_par + 1) + k] = t
            var dd = A[c * (n_par + 1) + c]
            for k in range(n_par + 1):
                A[c * (n_par + 1) + k] /= dd
            for r in range(n_par):
                if r == c:
                    continue
                var f = A[r * (n_par + 1) + c]
                for k in range(n_par + 1):
                    A[r * (n_par + 1) + k] -= f * A[c * (n_par + 1) + k]
        for a in range(n_par):
            p[free[a]] += A[a * (n_par + 1) + n_par]
    return p^


def _rms_mm(
    ref smp: Samples, lens: FisheyeLens, cpos: Vec3d, crot: Mat3d, p: List[Float64], only_ep: Int,
) -> Tuple[Float64, Int]:
    var s = 0.0
    var n = 0
    for i in range(len(smp.u)):
        if only_ep >= 0 and smp.ep[i] != only_ep:
            continue
        var r = _residual(smp, i, lens, cpos, crot, p)
        s += r[0] * r[0] + r[1] * r[1]
        n += 1
    return (sqrt(s / Float64(max(n, 1))) * 1000.0, n)


def _jz_residual(
    ref smp: Samples, i: Int, mut fk: TowerArmFK, wci: Int, lens: FisheyeLens,
    ref p: List[Float64], tilt_axis: Vec3d,
) raises -> Tuple[Float64, Float64]:
    """Sample i's (x, y) error (m) with joint-zero offsets p[0..2] (rad) on
    elbow_flex, wrist_flex, wrist_roll, the camera rolled p[3] (rad) about
    its optical axis, the reference shifted p[4..5] (the overhead's own
    offset), and the camera TILTED p[6] (rad) about `tilt_axis` — the flex
    axis's direction, FIXED in the gripper frame, the camera's position kept:
    a plate-angle error, which moves the camera's view like a flex-zero error
    but not its position (noeira-26's question)."""
    var qq = smp.q[i].copy()
    qq[1] += p[7] + p[8] * smp.arm_lift[i]
    qq[2] += p[0] + p[9] * smp.arm_elbow[i]
    qq[3] += p[1]
    qq[4] += p[2]
    fk.set_qpos(qq)
    var bpose = fk.camera_body_pose(wci)
    # MuJoCo's camera looks down its -z: a roll about the optical axis is a
    # rotation about its local z
    var c = cos(p[3])
    var sn = sin(p[3])
    var rz = Mat3d.from_cols(Vec3d(c, sn, 0.0), Vec3d(-sn, c, 0.0), Vec3d(0.0, 0.0, 1.0))
    var pos = bpose[0] + bpose[1] * fk.cam_pos[wci]
    var rt = _rot_small(tilt_axis.x * p[6], tilt_axis.y * p[6], tilt_axis.z * p[6])
    var rot = bpose[1] @ (rt @ (fk.cam_rot[wci] @ rz))
    var cam = RigCamera(lens, pos, rot)
    var pt = cam.plane_point(smp.u[i], smp.v[i], DESK_Z + 0.0125)
    if not pt[2]:
        return (1.0, 1.0)
    return (pt[0] - (smp.rx[i] + p[4]), pt[1] - (smp.ry[i] + p[5]))


def _droop_mm(ref smp: Samples, mut fk: TowerArmFK, wci: Int, ref p: List[Float64]) raises -> Float64:
    """How much the fitted offsets LOWER the gripper at the median-reach
    sample (mm, negative = droops, what sag must do)."""
    var best = 0
    var ms = 0.0
    var reaches = List[Float64]()
    for i in range(len(smp.u)):
        reaches.append(smp.arm_lift[i])
    var med = _pct(reaches, 0.5)
    var dbest = 1e9
    for i in range(len(smp.u)):
        if abs(smp.arm_lift[i] - med) < dbest:
            dbest = abs(smp.arm_lift[i] - med)
            best = i
    _ = ms
    var q0 = smp.q[best].copy()
    fk.set_qpos(q0)
    var z0 = fk.camera_body_pose(wci)[0].z
    var q1 = q0.copy()
    q1[1] += p[7] + p[8] * smp.arm_lift[best]
    q1[2] += p[0] + p[9] * smp.arm_elbow[best]
    q1[3] += p[1]
    fk.set_qpos(q1)
    var z1 = fk.camera_body_pose(wci)[0].z
    return (Float64(z1) - Float64(z0)) * 1000.0


def _jz_free(model: Int) -> List[Int]:
    """0 shift; 1 + wrist_flex; 2 + wrist_roll; 3 + the camera's roll;
    4 + elbow; 5 shift + the camera's TILT (plate angle) instead of flex;
    6 + wrist_roll; 7 flex AND tilt together; 8 shift + elbow (flex pinned
    at 0: the straight edge measured it); 9 shift + shoulder_lift; 10 shift +
    lift + elbow."""
    if model == 0:
        return [4, 5]
    if model == 8:
        return [0, 4, 5]
    if model == 9:
        return [7, 4, 5]
    if model == 10:
        return [7, 0, 4, 5]
    if model == 11:
        return [9, 4, 5]
    if model == 12:
        return [8, 4, 5]
    if model == 13:
        return [8, 9, 4, 5]
    if model == 5:
        return [6, 4, 5]
    if model == 6:
        return [6, 2, 4, 5]
    if model == 7:
        return [1, 6, 4, 5]
    if model == 1:
        return [1, 4, 5]
    if model == 2:
        return [1, 2, 4, 5]
    if model == 3:
        return [1, 2, 3, 4, 5]
    return [0, 1, 2, 3, 4, 5]


def _jz_fit(
    ref smp: Samples, mut fk: TowerArmFK, wci: Int, lens: FisheyeLens, ref use: List[Bool],
    model: Int, tilt_axis: Vec3d,
) raises -> List[Float64]:
    var free = _jz_free(model)
    var n_par = len(free)
    var p = List[Float64](length=10, fill=0.0)
    for _ in range(10):
        var H = List[Float64](length=36, fill=0.0)
        var g = List[Float64](length=6, fill=0.0)
        for i in range(len(smp.u)):
            if not use[i]:
                continue
            var r0 = _jz_residual(smp, i, fk, wci, lens, p, tilt_axis)
            # a robust weight: residuals past 30 mm are the estimator's
            # misses, not the model's
            var rr = sqrt(r0[0] * r0[0] + r0[1] * r0[1])
            var w = 1.0 if rr < 0.03 else 0.03 / rr
            var J = List[Float64](length=12, fill=0.0)
            for j in range(n_par):
                var pj = p.copy()
                var h = 1e-5
                pj[free[j]] += h
                var r1 = _jz_residual(smp, i, fk, wci, lens, pj, tilt_axis)
                J[j] = (r1[0] - r0[0]) / h
                J[6 + j] = (r1[1] - r0[1]) / h
            for a in range(n_par):
                g[a] += w * (J[a] * r0[0] + J[6 + a] * r0[1])
                for b in range(n_par):
                    H[a * 6 + b] += w * (J[a] * J[b] + J[6 + a] * J[6 + b])
        for a in range(n_par):
            H[a * 6 + a] += 1e-9
        var A = List[Float64](length=n_par * (n_par + 1), fill=0.0)
        for a in range(n_par):
            for b in range(n_par):
                A[a * (n_par + 1) + b] = H[a * 6 + b]
            A[a * (n_par + 1) + n_par] = -g[a]
        for cc in range(n_par):
            var piv = cc
            for r in range(cc + 1, n_par):
                if abs(A[r * (n_par + 1) + cc]) > abs(A[piv * (n_par + 1) + cc]):
                    piv = r
            for k in range(n_par + 1):
                var t = A[cc * (n_par + 1) + k]
                A[cc * (n_par + 1) + k] = A[piv * (n_par + 1) + k]
                A[piv * (n_par + 1) + k] = t
            var dd = A[cc * (n_par + 1) + cc]
            for k in range(n_par + 1):
                A[cc * (n_par + 1) + k] /= dd
            for r in range(n_par):
                if r == cc:
                    continue
                var f = A[r * (n_par + 1) + cc]
                for k in range(n_par + 1):
                    A[r * (n_par + 1) + k] -= f * A[cc * (n_par + 1) + k]
        for a in range(n_par):
            p[free[a]] += A[a * (n_par + 1) + n_par]
    return p^


def _jz_err(
    ref smp: Samples, mut fk: TowerArmFK, wci: Int, lens: FisheyeLens, ref use: List[Bool],
    ref p: List[Float64], tilt_axis: Vec3d,
) raises -> Tuple[Float64, Float64, Int]:
    """(median, rms clipped at 30 mm) of the used samples' errors, mm."""
    var es = List[Float64]()
    var s = 0.0
    for i in range(len(smp.u)):
        if not use[i]:
            continue
        var r = _jz_residual(smp, i, fk, wci, lens, p, tilt_axis)
        var e = min(0.03, sqrt(r[0] * r[0] + r[1] * r[1]))
        es.append(e * 1000.0)
        s += e * e
    return (_pct(es, 0.5), sqrt(s / Float64(max(len(es), 1))) * 1000.0, len(es))


def main() raises:
    var args = argv()
    var root = String("")
    var ep_lo = 0
    var ep_hi = -1
    var stride = 2
    var hue_hist = False
    var snap = String("")
    var roi_mm = 80.0
    var cls_w = printed_brick_hsv()
    var do_fit = False
    var fit_zeros = False
    var rel_max_d = 1.0
    var i = 1
    while i < len(args):
        var a = String(args[i])
        if a == "--hue-hist":
            hue_hist = True
            i += 1
            continue
        if a == "--fit":
            do_fit = True
            i += 1
            continue
        if a == "--fit-zeros":
            fit_zeros = True
            i += 1
            continue
        if i + 1 >= len(args):
            raise Error("flag " + a + " needs a value")
        var v = String(args[i + 1])
        if a == "--root":
            root = v
        elif a == "--episodes":
            var p = v.split(":")
            ep_lo = Int(String(p[0]))
            ep_hi = Int(String(p[1]))
        elif a == "--stride":
            stride = Int(v)
        elif a == "--snap":
            snap = v
        elif a == "--rel-max-mm":
            rel_max_d = Float64(v) / 1000.0
        elif a == "--roi-mm":
            roi_mm = Float64(v)
        elif a == "--wrist-hsv":
            var f = _floats(v)
            cls_w = ColorClass(f[0], f[1], f[2], f[3], f[4])
        else:
            raise Error("unknown flag " + a)
        i += 2
    if root == "":
        raise Error("--root <local LeRobot v3 dataset> is required")
    if snap != "":
        makedirs(snap, exist_ok=True)

    var info = LeRobotInfo(root)
    var oi = -1
    var wi = -1
    for c in range(len(info.cameras)):
        if info.cameras[c] == "observation.images.overhead":
            oi = c
        if info.cameras[c] == "observation.images.wrist":
            wi = c
    if oi < 0 or wi < 0:
        raise Error("the dataset needs observation.images.overhead and observation.images.wrist")
    var index = EpisodeIndex(root, info.cameras)
    var table = FrameTable(root, info.state_dim, info.action_dim)
    if ep_hi < 0 or ep_hi > index.n_episodes():
        ep_hi = index.n_episodes()

    var ocal = read_calib(String(OVERHEAD_CALIB))
    ocal.require_size(640, 480)
    var opose = tower_overhead_pose()
    var ocam = RigCamera(FisheyeLens.from_calib(ocal), opose.pos, opose.rot_mj)
    var wcal = read_calib(String(WRIST_CALIB))
    wcal.require_size(640, 480)
    var wlens = FisheyeLens.from_calib(wcal)
    var roi = tower_desk_roi()
    var brick = PrismModel.tower_brick()
    var fk = TowerArmFK()
    var wcam_i = fk.camera_index("wrist_cam")
    var ub = fk.body_index("robot_upper_arm")
    var lb = fk.body_index("robot_lower_arm")
    var units = So101TowerUnits(String(RIG_JOINT_ZERO_FOLLOWER))
    print("dataset", root, ":", index.n_episodes(), "episodes | checking", ep_lo, "..", ep_hi - 1)
    print("wrist lens", wlens, "| hsv", cls_w, "|", units.describe())

    var ostream = CameraStream(String("observation.images.overhead"), String(root))
    var wstream = CameraStream(String("observation.images.wrist"), String(root))

    var err = List[Float64]()
    var errx = List[Float64]()
    var erry = List[Float64]()
    var yawd = List[Float64]()
    var dist = List[Float64]()
    var speed = List[Float64]()
    var ep_med = List[Float64]()
    var smp = Samples()
    var rel_o_x = List[Float64]()
    var rel_o_y = List[Float64]()
    var rel_o_z = List[Float64]()
    var rel_w_x = List[Float64]()
    var rel_w_y = List[Float64]()
    var rel_w_z = List[Float64]()
    var rel_both = List[Bool]()
    var n_tried = 0
    var n_conf = 0
    var n_ep = 0
    var q = List[Float64](length=6, fill=0.0)
    for e in range(ep_lo, ep_hi):
        var length = index.length[e]
        var g0 = index.from_index[e]
        var idle = _idle_end(table, g0, length)
        var t_c = _grasp_frame(table, g0, length, idle)
        if t_c < 0:
            print("ep", e, ": no grasp found")
            continue
        # the overhead reference over the still start
        ostream.open_at(index.vid_chunk[oi][e], index.vid_file[oi][e], Int(round(index.vid_from_ts[oi][e] * Float64(info.fps))))
        var rx = List[Float64]()
        var ry = List[Float64]()
        var ryaw = List[Float64]()
        for _ in range(idle):
            ostream.next_native()
            var eo = estimate_prism_pose(ostream.raw, ocam, printed_brick_hsv(), brick, roi)
            if pose_confident(eo):
                rx.append(eo.x)
                ry.append(eo.y)
                ryaw.append(eo.yaw)
        if len(rx) == 0:
            print("ep", e, ": no confident overhead brick in the still start")
            continue
        var bx = _pct(rx, 0.5)
        var by = _pct(ry, 0.5)
        var byaw = ryaw[len(ryaw) // 2]
        n_ep += 1
        var wroi = DeskROI(DESK_Z, bx - roi_mm / 1000.0, bx + roi_mm / 1000.0, by - roi_mm / 1000.0, by + roi_mm / 1000.0)
        # the wrist frames from the first move to the grasp
        wstream.open_at(index.vid_chunk[wi][e], index.vid_file[wi][e], Int(round(index.vid_from_ts[wi][e] * Float64(info.fps))))
        var e_err = List[Float64]()
        # the gripper at the grasp frame: where does each sensor say the
        # brick sits in ITS frame? (the operator's own offset is shared)
        var sd0 = table.state_dim
        for k in range(N_ARM):
            q[k] = units.lerobot_to_joint(k, Float64(table.qpos[(g0 + t_c) * sd0 + k]))
        fk.set_qpos(q)
        var gsite = fk.site_index("grasp_center")
        var gp = fk.site_pos(gsite)
        var gR = fk.site_body_rot(gsite)
        var ro = gR.transpose() * (Vec3d(bx, by, DESK_Z + 0.0125) - gp)
        var wrel_x = List[Float64]()
        var wrel_y = List[Float64]()
        var wrel_z = List[Float64]()
        var best_d = 1e9
        var best_frame = List[UInt8]()
        var best_cam = RigCamera(wlens, Vec3d(0.0, 0.0, 1.0), Mat3d.identity())
        var best_est = PoseEstimate.none(0)
        var n_ep_conf = 0
        for t in range(t_c):
            wstream.next_native()
            if t < idle or (t % stride != 0 and t < t_c - 10):
                continue
            var sd = table.state_dim
            var spd = 0.0
            for k in range(N_ARM):
                var vk = Float64(table.qpos[(g0 + t) * sd + k])
                q[k] = units.lerobot_to_joint(k, vk)
                if t > 0 and k != GRIPPER:
                    spd = max(spd, abs(vk - Float64(table.qpos[(g0 + t - 1) * sd + k])))
            fk.set_qpos(q)
            var cp = fk.camera_pose(wcam_i)
            var wcam = RigCamera(wlens, cp[0], cp[1])
            var d = sqrt((cp[0].x - bx) ** 2 + (cp[0].y - by) ** 2 + (cp[0].z - DESK_Z) ** 2)
            # is the reference even in front of the camera?
            var uv = wcam.project(Vec3d(bx, by, DESK_Z + 0.0125))
            if not uv[2] or uv[0] < 0.0 or uv[0] > 639.0 or uv[1] < 0.0 or uv[1] > 479.0:
                continue
            n_tried += 1
            var ew = estimate_prism_pose(wstream.raw, wcam, cls_w, brick, wroi)
            if d < best_d:
                best_d = d
                best_frame = wstream.raw.copy()
                best_cam = wcam.copy()
                best_est = ew.copy()
            if not pose_confident(ew):
                continue
            n_conf += 1
            n_ep_conf += 1
            var puv = wcam.project(Vec3d(ew.x, ew.y, DESK_Z + 0.0125))
            if puv[2]:
                var bpose = fk.camera_body_pose(wcam_i)
                smp.u.append(puv[0])
                smp.v.append(puv[1])
                smp.bp.append(bpose[0])
                smp.br.append(bpose[1])
                smp.rx.append(bx)
                smp.ry.append(by)
                smp.ep.append(e)
                smp.q.append(q.copy())
                smp.speed.append(spd)
                var gpos = bpose[0]
                var ua = fk.body_pos(ub)
                var la = fk.body_pos(lb)
                smp.arm_lift.append(sqrt((gpos.x - ua.x) ** 2 + (gpos.y - ua.y) ** 2))
                smp.arm_elbow.append(sqrt((gpos.x - la.x) ** 2 + (gpos.y - la.y) ** 2))
            # the wrist estimate in the gripper frame AT THE GRASP (the far
            # frames only: near the grasp the jaws may already push it)
            if t <= t_c - 10 and d <= rel_max_d:
                var rw = gR.transpose() * (Vec3d(ew.x, ew.y, DESK_Z + 0.0125) - gp)
                wrel_x.append(Float64(rw.x))
                wrel_y.append(Float64(rw.y))
                wrel_z.append(Float64(rw.z))
            var ex = (ew.x - bx) * 1000.0
            var ey = (ew.y - by) * 1000.0
            var em = sqrt(ex * ex + ey * ey)
            err.append(em)
            errx.append(ex)
            erry.append(ey)
            yawd.append(_yaw_diff_deg(ew.yaw, byaw))
            dist.append(d)
            speed.append(spd)
            e_err.append(em)
        if hue_hist and e == ep_lo and len(best_frame) > 0:
            _hue_hist(best_frame, best_cam, bx, by)
        if snap != "" and len(best_frame) > 0:
            var img = best_frame.copy()
            _outline(img, best_cam, brick, bx, by, byaw, 0, 255, 0)
            if best_est.found:
                _outline(img, best_cam, brick, best_est.x, best_est.y, best_est.yaw, 255, 0, 0)
            save_png(snap + "/ep" + String(e) + "_wrist.png", img, 640, 480, 3)
        var line = String("ep ") + String(e) + ": ref (" + fixed(bx * 1000.0, 1) + ", " + fixed(by * 1000.0, 1) + ") | wrist confident " + String(n_ep_conf)
        rel_o_x.append(Float64(ro.x))
        rel_o_y.append(Float64(ro.y))
        rel_o_z.append(Float64(ro.z))
        if len(wrel_x) > 0:
            rel_w_x.append(_pct(wrel_x, 0.5))
            rel_w_y.append(_pct(wrel_y, 0.5))
            rel_w_z.append(_pct(wrel_z, 0.5))
            rel_both.append(True)
        else:
            rel_w_x.append(0.0)
            rel_w_y.append(0.0)
            rel_w_z.append(0.0)
            rel_both.append(False)
        if len(e_err) > 0:
            ep_med.append(_pct(e_err, 0.5))
            line += " | error median " + fixed(_pct(e_err, 0.5), 1) + " mm | closest camera " + fixed(best_d * 1000.0, 0) + " mm"
        print(line)

    print("-" * 70)
    print("episodes", n_ep, "| wrist frames with the brick in view", n_tried, "| confident", n_conf)
    if len(err) == 0:
        print("no confident wrist estimate — try --hue-hist and --wrist-hsv")
        return
    print(
        "wrist vs overhead, all confident frames: error median", fixed(_pct(err, 0.5), 1),
        "mm, p90", fixed(_pct(err, 0.9), 1), "| mean offset (", fixed(_pct(errx, 0.5), 1), ",",
        fixed(_pct(erry, 0.5), 1), ") mm (medians) | yaw diff median", fixed(_pct(yawd, 0.5), 1), "deg",
    )
    print("per-episode median error: median", fixed(_pct(ep_med, 0.5), 1), "mm, p90", fixed(_pct(ep_med, 0.9), 1))
    # by camera distance and by arm speed
    var dbins: List[Float64] = [0.0, 0.10, 0.15, 0.20, 0.30, 1.0]
    for b in range(len(dbins) - 1):
        var sel = List[Float64]()
        for k in range(len(err)):
            if dist[k] >= dbins[b] and dist[k] < dbins[b + 1]:
                sel.append(err[k])
        if len(sel) > 0:
            print("  camera", fixed(dbins[b] * 1000.0, 0), "-", fixed(dbins[b + 1] * 1000.0, 0),
                  "mm from the brick:", len(sel), "frames, error median", fixed(_pct(sel, 0.5), 1), "mm")
    # ── where each sensor puts the brick in the gripper at the grasp ──
    # across episodes: the operator's grasp varies the same for both, so the
    # sensor with the smaller spread is the better one FOR THE GRASP
    var ox_ = List[Float64]()
    var oy_ = List[Float64]()
    var oz_ = List[Float64]()
    var wx_ = List[Float64]()
    var wy_ = List[Float64]()
    var wz_ = List[Float64]()
    for k in range(len(rel_both)):
        if not rel_both[k]:
            continue
        ox_.append(rel_o_x[k] * 1000.0)
        oy_.append(rel_o_y[k] * 1000.0)
        oz_.append(rel_o_z[k] * 1000.0)
        wx_.append(rel_w_x[k] * 1000.0)
        wy_.append(rel_w_y[k] * 1000.0)
        wz_.append(rel_w_z[k] * 1000.0)
    print(
        "the brick in the gripper frame at the grasp (x pinch axis, z along the finger), over",
        len(ox_), "episodes — median and p10..p90 spread (mm):",
    )
    print(
        "  overhead: x", fixed(_pct(ox_, 0.5), 1), "(", fixed(_pct(ox_, 0.9) - _pct(ox_, 0.1), 1), ")",
        " y", fixed(_pct(oy_, 0.5), 1), "(", fixed(_pct(oy_, 0.9) - _pct(oy_, 0.1), 1), ")",
        " z", fixed(_pct(oz_, 0.5), 1), "(", fixed(_pct(oz_, 0.9) - _pct(oz_, 0.1), 1), ")",
    )
    print(
        "  wrist   : x", fixed(_pct(wx_, 0.5), 1), "(", fixed(_pct(wx_, 0.9) - _pct(wx_, 0.1), 1), ")",
        " y", fixed(_pct(wy_, 0.5), 1), "(", fixed(_pct(wy_, 0.9) - _pct(wy_, 0.1), 1), ")",
        " z", fixed(_pct(wz_, 0.5), 1), "(", fixed(_pct(wz_, 0.9) - _pct(wz_, 0.1), 1), ")",
    )

    # ── joint zeros + the camera's roll, fitted to the overhead reference ──
    if fit_zeros:
        # the slow frames only: a moving arm adds frame/state skew
        var slow = List[Bool]()
        var eps2 = List[Int]()
        for i in range(len(smp.u)):
            slow.append(smp.speed[i] < SLOW_DEG)
            if smp.ep[i] not in eps2:
                eps2.append(smp.ep[i])
        # the flex axis in the gripper frame (at the rest pose, roll 0):
        # the rotation a small flex step makes, seen from the gripper
        var z6 = List[Float64](length=6, fill=0.0)
        fk.set_qpos(z6)
        var R0 = fk.camera_body_pose(wcam_i)[1]
        var z6b = z6.copy()
        z6b[3] = 1e-4
        fk.set_qpos(z6b)
        var W = fk.camera_body_pose(wcam_i)[1] @ R0.transpose()
        var aw = Vec3d(
            (Float64(W.col(1).z) - Float64(W.col(2).y)) / 2e-4,
            (Float64(W.col(2).x) - Float64(W.col(0).z)) / 2e-4,
            (Float64(W.col(0).y) - Float64(W.col(1).x)) / 2e-4,
        )
        var tilt_axis = R0.transpose() * aw
        tilt_axis = tilt_axis * (1.0 / tilt_axis.length())
        print("  the flex axis in the gripper frame:", fixed(tilt_axis.x, 3), fixed(tilt_axis.y, 3), fixed(tilt_axis.z, 3))
        var zero6 = List[Float64](length=10, fill=0.0)
        var e0 = _jz_err(smp, fk, wcam_i, wlens, slow, zero6, tilt_axis)
        print(
            "JOINT-ZERO FIT on", e0[2], "slow frames (<", SLOW_DEG, "deg/frame): before median",
            fixed(e0[0], 1), "mm, rms(clip 30)", fixed(e0[1], 1), "mm",
        )
        var mnames: List[String] = [
            "shift", "+ wrist_flex", "+ wrist_roll", "+ camera roll", "+ elbow",
            "shift + camera TILT (not flex)", "shift + camera TILT + wrist_roll", "shift + flex + camera TILT",
            "shift + ELBOW (flex 0)", "shift + LIFT (flex 0)", "shift + lift + elbow (flex 0)",
            "shift + elbow SAG k*reach", "shift + lift SAG k*reach", "shift + lift SAG + elbow SAG",
        ]
        var models: List[Int] = [0, 8, 11, 12, 13]
        for model in models:
            var pf = _jz_fit(smp, fk, wcam_i, wlens, slow, model, tilt_axis)
            var ef = _jz_err(smp, fk, wcam_i, wlens, slow, pf, tilt_axis)
            # leave one episode out: fit on the others, score the held-out one
            var lo = List[Float64]()
            for ek in eps2:
                var tr_use = List[Bool]()
                var te_use = List[Bool]()
                for i in range(len(smp.u)):
                    tr_use.append(slow[i] and smp.ep[i] != ek)
                    te_use.append(slow[i] and smp.ep[i] == ek)
                var pl = _jz_fit(smp, fk, wcam_i, wlens, tr_use, model, tilt_axis)
                var el = _jz_err(smp, fk, wcam_i, wlens, te_use, pl, tilt_axis)
                if el[2] > 0:
                    lo.append(el[0])
            print(
                "  " + mnames[model] + ": median", fixed(ef[0], 1), "mm, rms", fixed(ef[1], 1),
                "| leave-one-episode-out median of episode medians", fixed(_pct(lo, 0.5), 1),
                "mm | elbow", fixed(pf[0] * 180.0 / pi, 2), "flex", fixed(pf[1] * 180.0 / pi, 2),
                "roll", fixed(pf[2] * 180.0 / pi, 2), "cam roll", fixed(pf[3] * 180.0 / pi, 2),
                "cam tilt", fixed(pf[6] * 180.0 / pi, 2), "lift", fixed(pf[7] * 180.0 / pi, 2),
                "| k_lift", fixed(pf[8] * 180.0 / pi / 10.0, 2), "k_elbow", fixed(pf[9] * 180.0 / pi / 10.0, 2),
                "deg per 10 cm reach | droop at the median pose", fixed(_droop_mm(smp, fk, wcam_i, pf), 1), "mm",
                "deg | shift", fixed(pf[4] * 1000.0, 1), fixed(pf[5] * 1000.0, 1), "mm",
            )

    # ── the camera-in-gripper correction, fitted to the overhead reference ──
    if do_fit:
        var cpos = fk.cam_pos[wcam_i]
        var crot = fk.cam_rot[wcam_i]
        var zero = List[Float64](length=8, fill=0.0)
        var r0 = _rms_mm(smp, wlens, cpos, crot, zero, -1)
        print("FIT on", r0[1], "samples: rms before", fixed(r0[0], 1), "mm")
        var eps = List[Int]()
        for i in range(len(smp.ep)):
            if smp.ep[i] not in eps:
                eps.append(smp.ep[i])
        for model in range(3):
            var names = String("shift") if model == 0 else (String("shift + rot") if model == 1 else String("shift + rot + pos"))
            var pf = _fit(smp, wlens, cpos, crot, -1, model)
            var rf = _rms_mm(smp, wlens, cpos, crot, pf, -1)
            # leave one episode out
            var s = 0.0
            var n = 0
            for ek in eps:
                var pl = _fit(smp, wlens, cpos, crot, ek, model)
                var rl = _rms_mm(smp, wlens, cpos, crot, pl, ek)
                s += rl[0] * rl[0] * Float64(rl[1])
                n += rl[1]
            print(
                "  " + names + ": rms", fixed(rf[0], 1), "mm | leave-one-episode-out",
                fixed(sqrt(s / Float64(max(n, 1))), 1), "mm | rot (deg)",
                fixed(pf[0] * 180.0 / pi, 2), fixed(pf[1] * 180.0 / pi, 2), fixed(pf[2] * 180.0 / pi, 2),
                "| pos (mm)", fixed(pf[3] * 1000.0, 1), fixed(pf[4] * 1000.0, 1), fixed(pf[5] * 1000.0, 1),
                "| shift (mm)", fixed(pf[6] * 1000.0, 1), fixed(pf[7] * 1000.0, 1),
            )

    var sbins: List[Float64] = [0.0, 0.5, 1.5, 3.0, 100.0]
    for b in range(len(sbins) - 1):
        var sel = List[Float64]()
        for k in range(len(err)):
            if speed[k] >= sbins[b] and speed[k] < sbins[b + 1]:
                sel.append(err[k])
        if len(sel) > 0:
            print("  arm speed", fixed(sbins[b], 1), "-", fixed(sbins[b + 1], 1),
                  "deg/frame:", len(sel), "frames, error median", fixed(_pct(sel, 0.5), 1), "mm")
