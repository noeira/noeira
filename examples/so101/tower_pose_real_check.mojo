"""The overhead pose estimator on a REAL LeRobot recording of the tower rig.

    pixi run mojo run -I . examples/so101/tower_pose_real_check.mojo \\
        --root projects/so101-tower/datasets/<name> [--episodes 0:10] \\
        [--snap /tmp/pose_snaps] [--hue-hist]

    # the first recording (lime Duplo, teal cup), with its own colours/shapes:
    ... --root projects/so101-tower/datasets/cube-in-bowl \\
        --brick-hsv 70,20,0.35,0.25,1 --brick-model box:0.0318,0.022 \\
        --bowl-hsv 180,18,0.3,0.15,1 --bowl-model cyl:0.055,0.06

WHAT IT MEASURES, with no ground truth recorded (`OBJECT_POSE_POLICY_PLAN.md`
§3.6). The raw 640x480 fisheye overhead frames are decoded
(`data/lerobot.CameraStream`, the importer's own reader) and fed to
`vision/tabletop_pose.estimate_prism_pose` through the measured lens
(`--calib`) at the ASSET camera pose (`tower_sim_camera`). Per episode:

1. STILL START. Until the arm first moves (state change > 1.5 units, at most
   `--max-idle` frames) nothing touches the objects: the spread of the
   brick's and the bowl's estimates over those frames is the estimator's
   frame-to-frame noise on real pixels.
2. THE ARM AS GROUND TRUTH. The grasp frame is where the gripper, having
   dropped >= 6 units from its running maximum, settles (the jaw has closed
   on the object: 30-41 -> ~18 in `cube-in-bowl`). There the follower's FK of
   `grasp_center` — the tower follower's site where a held brick's centre
   sits — is where the brick is. Its (x, y) against the still-start estimate
   bounds camera + FK + estimator + how off-centre the operator grasped,
   together. FK uses the follower's joint zero (`--joint-zero`, default
   `follower`: `model_rad = deg2rad(lerobot) + zero`, `sim_map`).
3. END. Over the last 30 frames (the arm back at rest) the last brick
   detection (any coverage: the bowl's near wall hides part of a brick inside
   it) against the bowl's confident estimate: a brick left in the bowl is < 45 mm
   from its centre (the task's `Near`). A brick deep in a tall cup is not
   seen at all; the estimator reads everything on the DESK plane, so a
   lifted or stacked object is misplaced — hence only resting frames.

⚠ The arm check is a LOOSE bound: `grasp_center` is where a brick held DEEP
in the jaw sits (3 cm up the jaw from the tip), and an operator pinching
with the tips on a tilted gripper puts it off the object's centre by that
offset's horizontal part. The wrist joints' zeros are also unmeasured (an
attempt to follow the jaw axis down to the object read 25 mm against 8 mm
raw). Measured placements are the real ruler.

"Confident" = coverage in [0.75, 1.3] and residual < 0.35.

`--hue-hist` prints the hue histogram of the saturated pixels inside the
desk ROI on the first episode's first frame — how to choose `--*-hsv` for a
new set of props. `--snap DIR` writes, per episode, the first frame with the
fitted outlines (brick red, bowl magenta) and the grasp frame with the FK
point (green cross) and the brick's still-start outline.

⚠ One ffmpeg per video file, read forward only (the importer's rule):
`--episodes` must be ascending.
⚠ Every `--stride`-th frame is estimated outside the still start and the 20
frames before the grasp (all of which are): ~2 x 30 ms per estimated frame.
"""

from std.math import acos, pi, sqrt, round
from std.os import makedirs
from std.sys import argv, exit

from noeira.data.lerobot import LeRobotInfo, EpisodeIndex, FrameTable, CameraStream
from noeira.io.png import save_png
from noeira.math3d import Mat3 as Mat3Generic, Quat as QuatGeneric, Vec3 as Vec3Generic
from noeira.physics3d.fields import Data, Model, DynDims
from noeira.physics3d.kinematics.forward_kinematics import forward_kinematics
from noeira.physics3d.parser.runtime_load import (
    parse_model_runtime, dims_from_flat, build_model_runtime,
)
from noeira.tasks.so101_tower_rig import (
    So101TowerUnits, RIG_JOINT_ZERO_FOLLOWER, RIG_JOINT_ZERO_NONE,
)
from noeira.tasks.family import scene_path
from noeira.tasks.spec import load_family
from noeira.tasks.so101_tower_overhead import (
    OVERHEAD_CALIB, DESK_Z, tower_overhead_pose, tower_desk_roi, pose_confident,
)
from noeira.tasks.so101_tower_xml import (
    SO101_TOWER_MAX_CONTACTS, SO101_TOWER_NMESH_VERTS,
)
from noeira.utils.fmt import fixed
from noeira.vision.calib_file import read_calib
from noeira.vision.fisheye import FisheyeLens
from noeira.vision.tabletop_pose import (
    RigCamera, ColorClass, PrismModel, DeskROI, PoseEstimate,
    estimate_prism_pose, model_silhouette, rgb_to_hsv, P2,
)

comptime DT = DType.float64
comptime Vec3d = Vec3Generic[DT]
comptime Mat3d = Mat3Generic[DT]
comptime Quatd = QuatGeneric[DT]
comptime FAMILY = "noeira/tasks/families/so101_tower.family"
comptime N_ARM = 6
comptime GRIPPER = 5
comptime NEAR_GOAL = 0.045
"""`so101_tower_cube_in_bowl.task`: `Near(brick, bowl, 0.045)`."""
comptime MOVE_EPS = 1.5
comptime GRASP_DROP = 6.0
comptime GRASP_SETTLE = 0.6
comptime PRE_GRASP = 20
comptime END_FRAMES = 30


# ─── small helpers ─────────────────────────────────────────────────────────


def _floats(s: String) raises -> List[Float64]:
    var out = List[Float64]()
    for p in s.split(","):
        out.append(Float64(String(String(p).strip())))
    return out^


def _hsv_arg(s: String) raises -> ColorClass:
    var v = _floats(s)
    if len(v) != 5:
        raise Error("an HSV class is hue,tol,s_min,v_min,v_max; got " + s)
    return ColorClass(v[0], v[1], v[2], v[3], v[4])


def _model_arg(name: String, s: String) raises -> PrismModel:
    """`cube:side`, `box:side,height`, `cyl:radius,height` (a 16-gon),
    `octagon` (the printed bowl), `brick` (the printed cube)."""
    if s == "octagon":
        return PrismModel.tower_bowl()
    if s == "brick":
        return PrismModel.tower_brick()
    var parts = s.split(":")
    if len(parts) != 2:
        raise Error("a model is cube:s | box:s,h | cyl:r,h | octagon | brick; got " + s)
    var kind = String(parts[0])
    var v = _floats(String(parts[1]))
    if kind == "cube" and len(v) == 1:
        return PrismModel.square(name, v[0], v[0])
    if kind == "box" and len(v) == 2:
        return PrismModel.square(name, v[0], v[1])
    if kind == "cyl" and len(v) == 2:
        return PrismModel.regular(name, 16, v[0], v[1])
    raise Error("cannot read model " + s)


def _median(var xs: List[Float64]) -> Float64:
    var n = len(xs)
    if n == 0:
        return 0.0
    for i in range(1, n):
        var p = xs[i]
        var j = i - 1
        while j >= 0 and xs[j] > p:
            xs[j + 1] = xs[j]
            j -= 1
        xs[j + 1] = p
    if n % 2 == 1:
        return xs[n // 2]
    return 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def _pct(var xs: List[Float64], q: Float64) -> Float64:
    var n = len(xs)
    if n == 0:
        return 0.0
    for i in range(1, n):
        var p = xs[i]
        var j = i - 1
        while j >= 0 and xs[j] > p:
            xs[j + 1] = xs[j]
            j -= 1
        xs[j + 1] = p
    var k = Int(q * Float64(n - 1) + 0.5)
    return xs[k]


def _mm(ax: Float64, ay: Float64, bx: Float64, by: Float64) -> Float64:
    return sqrt((ax - bx) * (ax - bx) + (ay - by) * (ay - by)) * 1000.0


# ─── drawing ───────────────────────────────────────────────────────────────


def _put(mut img: List[UInt8], w: Int, h: Int, u: Int, v: Int, r: UInt8, g: UInt8, b: UInt8):
    if u < 0 or v < 0 or u >= w or v >= h:
        return
    var k = (v * w + u) * 3
    img[k] = r
    img[k + 1] = g
    img[k + 2] = b


def _line(
    mut img: List[UInt8], w: Int, h: Int, a: P2, b: P2, r: UInt8, g: UInt8, bl: UInt8
):
    var n = Int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) + 1
    for i in range(n + 1):
        var t = Float64(i) / Float64(n)
        var u = Int(round(a[0] + t * (b[0] - a[0])))
        var v = Int(round(a[1] + t * (b[1] - a[1])))
        _put(img, w, h, u, v, r, g, bl)


def _outline(
    mut img: List[UInt8], cam: RigCamera, model: PrismModel, x: Float64,
    y: Float64, yaw: Float64, r: UInt8, g: UInt8, b: UInt8,
):
    var poly = model_silhouette(cam, model, DESK_Z, x, y, yaw)
    for i in range(len(poly)):
        _line(img, cam.width, cam.height, poly[i], poly[(i + 1) % len(poly)], r, g, b)


def _cross(mut img: List[UInt8], cam: RigCamera, p: Vec3d, r: UInt8, g: UInt8, b: UInt8):
    var uv = cam.project(p)
    if not uv[2]:
        return
    var u = Int(round(uv[0]))
    var v = Int(round(uv[1]))
    for d in range(-6, 7):
        _put(img, cam.width, cam.height, u + d, v, r, g, b)
        _put(img, cam.width, cam.height, u, v + d, r, g, b)


# ─── the arm ───────────────────────────────────────────────────────────────


struct ArmFK(Movable):
    """The tower scene's FK: LeRobot state -> `grasp_center` world."""

    var m: Model[DT, DynDims]
    var d: Data[DT, DynDims, 1]
    var site: Int
    var body: Int
    """The gripper body — its -z is the jaws' approach axis."""
    var units: So101TowerUnits
    """LeRobot units -> model joints: the one map the rig's tools share
    (the joint zero, and the gripper on its measured line since dbd873e15)."""

    def __init__(out self, joint_zero: String) raises:
        var f = load_family(String(FAMILY))
        var fmd = parse_model_runtime(scene_path(f))
        var dims = dims_from_flat(
            fmd, max_contacts=SO101_TOWER_MAX_CONTACTS,
            nmesh_verts=SO101_TOWER_NMESH_VERTS,
        )
        var m = Model[DT, DynDims](dims)
        build_model_runtime[DT](fmd, dims, m)
        var d = Data[DT, DynDims, 1](dims)
        for k in range(dims.get_nq()):
            d.qpos.data[k] = Scalar[DT](0)
        var adr = 0
        for j in range(len(fmd.joints)):
            if fmd.joints[j].nq == 7:
                d.qpos.data[adr + 3] = Scalar[DT](1)
            adr += fmd.joints[j].nq
        var site = -1
        for s in range(len(fmd.site_names)):
            if String(fmd.site_names[s]).endswith("grasp_center"):
                site = s
        if site < 0:
            raise Error("the tower scene has no grasp_center site")
        if joint_zero != "follower" and joint_zero != "none":
            raise Error("--joint-zero is follower or none, got " + joint_zero)
        self.m = m^
        self.d = d^
        self.site = site
        self.body = fmd.sites[site].body_id
        self.units = So101TowerUnits(
            String(RIG_JOINT_ZERO_FOLLOWER) if joint_zero == "follower" else String(RIG_JOINT_ZERO_NONE)
        )

    def grasp_center(mut self, state: List[Float64]) raises -> Vec3d:
        """`grasp_center` in the world (the state's FK)."""
        return self.grasp(state)[0]

    def grasp(mut self, state: List[Float64]) raises -> Tuple[Vec3d, Vec3d]:
        """(`grasp_center`, the approach axis = the gripper body's -z) in the
        world."""
        for k in range(N_ARM):
            self.d.qpos.data[k] = Scalar[DT](self.units.lerobot_to_joint(k, state[k]))
        forward_kinematics["cpu", DT, DynDims, 1](self.d, self.m)
        var s = self.site
        var b = self.body
        # `Data.xquat` is packed (x, y, z, w); `Quat` takes (w, x, y, z)
        var q = Quatd(
            Float64(self.d.xquat.data[b * 4 + 3]), Float64(self.d.xquat.data[b * 4]),
            Float64(self.d.xquat.data[b * 4 + 1]), Float64(self.d.xquat.data[b * 4 + 2]),
        )
        return (
            Vec3d(
                Float64(self.d.site_xpos.data[s * 3]),
                Float64(self.d.site_xpos.data[s * 3 + 1]),
                Float64(self.d.site_xpos.data[s * 3 + 2]),
            ),
            -Mat3d.from_quat(q).col(2),
        )


# ─── one episode's signals ─────────────────────────────────────────────────


def _idle_end(table: FrameTable, g0: Int, length: Int, max_idle: Int) -> Int:
    var sd = table.state_dim
    for t in range(1, min(length, max_idle)):
        for k in range(min(sd, N_ARM)):
            var dv = abs(Float64(table.qpos[(g0 + t) * sd + k]) - Float64(table.qpos[g0 * sd + k]))
            if dv > MOVE_EPS:
                return t
    return min(length, max_idle)


def _grasp_frame(table: FrameTable, g0: Int, length: Int, start: Int) -> Int:
    """The first frame after `start` where the gripper has dropped
    GRASP_DROP below its running maximum and then settles; -1 if never."""
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


def main() raises:
    var args = argv()
    var root = String("")
    var ep_lo = 0
    var ep_hi = -1
    var stride = 3
    var max_idle = 90
    var snap = String("")
    var hue_hist = False
    var calib_path = String(OVERHEAD_CALIB)
    var extr_path = String("")
    var camera_key = String("observation.images.overhead")
    var joint_zero = String("follower")
    var cls_brick = ColorClass.tower_brick_sim()
    var cls_bowl = ColorClass.tower_bowl_sim()
    var brick = PrismModel.tower_brick()
    var bowl = PrismModel.tower_bowl()
    var roi = tower_desk_roi()
    var i = 1
    while i < len(args):
        var a = String(args[i])
        if a == "--hue-hist":
            hue_hist = True
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
            ep_hi = Int(String(p[1])) if len(p) > 1 else ep_lo + 1
        elif a == "--stride":
            stride = Int(v)
        elif a == "--max-idle":
            max_idle = Int(v)
        elif a == "--snap":
            snap = v
        elif a == "--calib":
            calib_path = v
        elif a == "--extrinsics":
            extr_path = v
        elif a == "--camera":
            camera_key = v
        elif a == "--joint-zero":
            joint_zero = v
        elif a == "--brick-hsv":
            cls_brick = _hsv_arg(v)
        elif a == "--bowl-hsv":
            cls_bowl = _hsv_arg(v)
        elif a == "--brick-model":
            brick = _model_arg("brick", v)
        elif a == "--bowl-model":
            bowl = _model_arg("bowl", v)
        elif a == "--roi":
            var r = _floats(v)
            if len(r) != 4:
                raise Error("--roi is x_min,x_max,y_min,y_max")
            roi = DeskROI(DESK_Z, r[0], r[1], r[2], r[3])
        else:
            raise Error("unknown flag " + a)
        i += 2
    if root == "":
        raise Error("--root <local LeRobot v3 dataset> is required")

    # ── dataset ───────────────────────────────────────────────────────────
    var info = LeRobotInfo(root)
    var cam_i = -1
    for c in range(len(info.cameras)):
        if info.cameras[c] == camera_key:
            cam_i = c
    if cam_i < 0:
        raise Error("the dataset has no camera " + camera_key)
    var index = EpisodeIndex(root, info.cameras)
    var table = FrameTable(root, info.state_dim, info.action_dim)
    if ep_hi < 0 or ep_hi > index.n_episodes():
        ep_hi = index.n_episodes()
    print(
        "dataset", root, ":", index.n_episodes(), "episodes,", info.fps,
        "fps; checking", ep_lo, "..", ep_hi - 1, "stride", stride,
    )

    # ── camera + arm ──────────────────────────────────────────────────────
    var cal = read_calib(calib_path)
    cal.require_size(640, 480)
    var lens = FisheyeLens.from_calib(cal)
    var pose = tower_overhead_pose(extr_path)
    var cam = RigCamera(lens, pose.pos, pose.rot_mj)
    var arm = ArmFK(joint_zero)
    print("lens", lens)
    print("camera pose:", pose.source, "| roi", roi)
    print("brick", brick, "hsv", cls_brick)
    print("bowl ", bowl, "hsv", cls_bowl)
    print("joint zero", joint_zero)
    if snap != "":
        makedirs(snap, exist_ok=True)

    var stream = CameraStream(String(camera_key), String(root))

    # ── per-episode results ───────────────────────────────────────────────
    var still_brick = List[Float64]()
    var still_bowl = List[Float64]()
    var still_yaw = List[Float64]()
    var fk_err = List[Float64]()
    var fk_err_last = List[Float64]()
    var fk_z = List[Float64]()
    var tilt = List[Float64]()
    var end_dist = List[Float64]()
    var n_ep = 0
    var n_brick_found = 0
    var n_bowl_found = 0
    var n_grasp = 0
    var n_in_bowl = 0
    var n_end = 0

    for e in range(ep_lo, ep_hi):
        var length = index.length[e]
        var g0 = index.from_index[e]
        var first = Int(round(index.vid_from_ts[cam_i][e] * Float64(info.fps)))
        stream.open_at(index.vid_chunk[cam_i][e], index.vid_file[cam_i][e], first)
        var idle = _idle_end(table, g0, length, max_idle)
        var t_c = _grasp_frame(table, g0, length, idle)
        n_ep += 1

        var bx = List[Float64]()
        var by = List[Float64]()
        var byaw = List[Float64]()
        var ox = List[Float64]()
        var oy = List[Float64]()
        var last_b = PoseEstimate.none(0)
        var pre_b = PoseEstimate.none(0)
        var last_o = PoseEstimate.none(0)
        var first_frame = List[UInt8]()
        for t in range(length):
            stream.next_native()
            if t == 0 and hue_hist and e == ep_lo:
                _print_hue_hist(stream.raw, cam, roi)
            if t == 0 and snap != "":
                first_frame = stream.raw.copy()
            var in_idle = t < idle
            var pre_grasp = t_c >= 0 and t <= t_c and t >= t_c - PRE_GRASP
            # the arm is back at rest by then: a brick seen now rests on
            # something (a lifted one would be read on the desk plane)
            var near_end = t >= length - END_FRAMES
            if not (in_idle or pre_grasp or t % stride == 0 or t == length - 1):
                continue
            var eb = estimate_prism_pose(stream.raw, cam, cls_brick, brick, roi)
            if pose_confident(eb):
                if in_idle:
                    bx.append(eb.x)
                    by.append(eb.y)
                    byaw.append(eb.yaw)
                if t_c < 0 or t <= t_c:
                    pre_b = eb
            # at the end ANY detection counts: a brick in the bowl is partly
            # hidden by the bowl's near wall (coverage < 0.75), and only its
            # being inside the bowl is asked here
            if near_end and eb.found:
                last_b = eb
            if snap != "" and t == length - 1:
                var img = stream.raw.copy()
                if last_b.found:
                    _outline(img, cam, brick, last_b.x, last_b.y, last_b.yaw, 255, 0, 0)
                if last_o.found:
                    _outline(img, cam, bowl, last_o.x, last_o.y, last_o.yaw, 255, 0, 255)
                save_png(snap + "/ep" + String(e) + "_end.png", img, cam.width, cam.height, 3)
            if in_idle or near_end:
                var eo = estimate_prism_pose(stream.raw, cam, cls_bowl, bowl, roi)
                if pose_confident(eo):
                    if in_idle:
                        ox.append(eo.x)
                        oy.append(eo.y)
                    last_o = eo
            if snap != "" and t == t_c:
                var img = stream.raw.copy()
                if len(bx) > 0:
                    _outline(img, cam, brick, _median(bx.copy()), _median(by.copy()), _median(byaw.copy()), 255, 0, 0)
                var st = List[Float64]()
                for k in range(N_ARM):
                    st.append(Float64(table.qpos[(g0 + t) * table.state_dim + k]))
                var p = arm.grasp_center(st)
                _cross(img, cam, Vec3d(p.x, p.y, DESK_Z + 0.5 * brick.height), 0, 255, 0)
                save_png(snap + "/ep" + String(e) + "_grasp.png", img, cam.width, cam.height, 3)

        # still start
        var line = String("ep ") + String(e) + ": " + String(length) + " fr, still " + String(idle)
        var have_brick = len(bx) > 0
        var mbx = 0.0
        var mby = 0.0
        if have_brick:
            n_brick_found += 1
            mbx = _median(bx.copy())
            mby = _median(by.copy())
            var spread = 0.0
            for k in range(len(bx)):
                spread = max(spread, _mm(bx[k], by[k], mbx, mby))
            var ys = List[Float64]()
            var myaw = _median(byaw.copy())
            for k in range(len(byaw)):
                var dy = abs(byaw[k] - myaw)
                dy = min(dy, brick.period - dy)
                ys.append(dy * 180.0 / pi)
            still_brick.append(spread)
            still_yaw.append(_pct(ys^, 1.0))
            line += (
                " | brick (" + String(Int(mbx * 1000)) + ", " + String(Int(mby * 1000))
                + ") mm, spread " + fixed(spread, 1) + " mm over "
                + String(len(bx))
            )
        else:
            line += " | brick NOT FOUND in the still start"
        if len(ox) > 0:
            n_bowl_found += 1
            var mox = _median(ox.copy())
            var moy = _median(oy.copy())
            var spread = 0.0
            for k in range(len(ox)):
                spread = max(spread, _mm(ox[k], oy[k], mox, moy))
            still_bowl.append(spread)
            line += (
                " | bowl (" + String(Int(mox * 1000)) + ", " + String(Int(moy * 1000))
                + "), spread " + fixed(spread, 1)
            )
        else:
            line += " | bowl NOT FOUND"

        # the arm at the grasp
        if t_c >= 0 and have_brick:
            n_grasp += 1
            var st = List[Float64]()
            for k in range(N_ARM):
                st.append(Float64(table.qpos[(g0 + t_c) * table.state_dim + k]))
            var ga = arm.grasp(st)
            var p = ga[0]
            var err = _mm(Float64(p.x), Float64(p.y), mbx, mby)
            fk_err.append(err)
            fk_z.append(Float64(p.z) * 1000.0)
            var tdeg = acos(min(1.0, max(-1.0, -Float64(ga[1].z)))) * 180.0 / pi
            tilt.append(tdeg)
            line += (
                " | grasp @" + String(t_c) + ": FK (" + String(Int(Float64(p.x) * 1000))
                + ", " + String(Int(Float64(p.y) * 1000)) + ", z " + String(Int(Float64(p.z) * 1000))
                + ") err " + fixed(err, 1) + " mm, jaw tilt " + fixed(tdeg, 0)
            )
            if pre_b.found:
                fk_err_last.append(_mm(Float64(p.x), Float64(p.y), pre_b.x, pre_b.y))
        else:
            line += " | no grasp" if t_c < 0 else ""

        # the end
        if last_b.found and last_o.found:
            n_end += 1
            var dd = _mm(last_b.x, last_b.y, last_o.x, last_o.y)
            end_dist.append(dd)
            if dd < NEAR_GOAL * 1000.0:
                n_in_bowl += 1
            line += (
                " | end brick-bowl " + String(Int(dd)) + " mm (brick at ("
                + String(Int(last_b.x * 1000)) + ", " + String(Int(last_b.y * 1000))
                + "), cov " + fixed(last_b.coverage, 2) + ")"
            )
        print(line)
        if snap != "" and len(first_frame) > 0:
            if have_brick:
                _outline(first_frame, cam, brick, mbx, mby, _median(byaw.copy()), 255, 0, 0)
            if len(ox) > 0:
                _outline(first_frame, cam, bowl, _median(ox.copy()), _median(oy.copy()), 0.0, 255, 0, 255)
            save_png(snap + "/ep" + String(e) + "_start.png", first_frame, cam.width, cam.height, 3)
    stream.close()

    print()
    print("episodes", n_ep, "| brick found in the still start", n_brick_found, "| bowl", n_bowl_found)
    if len(still_brick) > 0:
        print(
            "still-start spread, brick: median", _median(still_brick.copy()), "p90",
            _pct(still_brick.copy(), 0.9), "mm; yaw max-dev median",
            _median(still_yaw.copy()), "deg",
        )
    if len(still_bowl) > 0:
        print(
            "still-start spread, bowl:  median", _median(still_bowl.copy()), "p90",
            _pct(still_bowl.copy(), 0.9), "mm",
        )
    if len(fk_err) > 0:
        print(
            "grasp: FK grasp_center vs still-start brick (", n_grasp, "episodes): median",
            _median(fk_err.copy()), "p90", _pct(fk_err.copy(), 0.9), "max",
            _pct(fk_err.copy(), 1.0), "mm; FK z median", _median(fk_z.copy()),
            "mm; jaw tilt median", _median(tilt.copy()), "deg",
        )
    if len(fk_err_last) > 0:
        print(
            "grasp: vs the last confident brick before the grasp: median",
            _median(fk_err_last.copy()), "p90", _pct(fk_err_last.copy(), 0.9), "mm",
        )
    if n_end > 0:
        print(
            "end: brick within", NEAR_GOAL * 1000.0, "mm of the bowl in", n_in_bowl,
            "of", n_end, "episodes with both seen; median distance",
            _median(end_dist.copy()), "mm",
        )


def _print_hue_hist(frame: List[UInt8], cam: RigCamera, roi: DeskROI):
    """Hue histogram (10 deg bins) of pixels with s >= 0.3, v >= 0.15 whose
    ray meets the desk plane in the ROI — pick `--*-hsv` from its peaks."""
    var bins = List[Int](length=36, fill=0)
    var n = 0
    for v in range(cam.height):
        for u in range(cam.width):
            var k = (v * cam.width + u) * 3
            var hsv = rgb_to_hsv(frame[k], frame[k + 1], frame[k + 2])
            if hsv[1] < 0.3 or hsv[2] < 0.15:
                continue
            var p = cam.plane_point(Float64(u), Float64(v), roi.z)
            if not (p[2] and roi.inside(p[0], p[1])):
                continue
            bins[min(35, Int(hsv[0] / 10.0))] += 1
            n += 1
    print("hue histogram of", n, "saturated ROI pixels (frame 0):")
    for b in range(36):
        if bins[b] == 0:
            continue
        var bar = String("")
        for _ in range(min(60, bins[b] // 10 + 1)):
            bar += "#"
        print("  ", b * 10, "-", b * 10 + 10, "deg", bins[b], bar)
