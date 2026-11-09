"""The depth camera against MuJoCo 3.12 — the P0b gate.

    pixi run mojo run -I . tests/perception/test_depth_vs_mujoco.mojo

TWO REFERENCES, because each can be fooled where the other cannot.

1. `test_depth_vs_mj_ray` — EXACT. Every pixel of every image (ideal camera:
   no noise, no cuts) against `mj_ray` along a pixel ray built IN NUMPY from
   the intrinsics, judged by `mj_reference.RayStats` (the lidar gate's rule).
   ⚠ The reference vector has forward component 1 — `(xo, -yo, -1)` in
   MuJoCo's camera frame, `(1, -xo, -yo)` in a body mount's — so `mj_ray`'s
   `t` IS the planar depth; our normalise-then-scale is not shared. For the
   G1's `head_camera` the pose comes from MuJoCo's own `cam_xpos`/`cam_xmat`
   and its own camera axes, so our camera-to-sensor rotation
   (`depth_mount_from_mjcf_camera`) is checked, not assumed.

2. `test_depth_vs_mujoco_renderer` — CONVENTIONS. Both legs of (1) share the
   OpenCV pixel convention (row 0 at the top, pixel centres at integers,
   `cx = (w-1)/2`), so a half-pixel or flipped-axis mistake in it would pass
   (1). MuJoCo's OpenGL depth image of `head_camera` does not share it. The
   ideal image must match it, AND match it better than the same camera with
   `cx` or `cy` moved by half a pixel or the image flipped — a gate that a
   shifted camera also passes is not checking the convention.

RESULT (2026-11-09, MuJoCo 3.12): (1) 3 509 760 pixels — 12 poses x {D435-class
848x480, ToF-class 640x576, D435-class 424x240} x {head_camera, tilted pelvis,
world rig}: 0 splits, 0 wrong geoms, worst |dz| 1.0e-8 m. (2) 99.88 % of
pixels within 2 mm + 0.2 %, mean |dz| 0.04 mm; cx + 0.5 px: mean 2.4 mm,
cy + 0.5 px: 3.7 mm, flipped: 46 mm.

FALSIFIED: the camera-to-sensor rotation inverted (q_cs conjugated) ->
(1) 47 915 splits + 1 413 473 wrong geoms, (2) 0.5 % of pixels match. Both fail.
"""

from std.math import abs, pi
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite

from noeira.physics3d.fields import DynDims
from noeira.perception import (
    DepthCameraSpec,
    LidarMount,
    load_depth_spec,
    depth_spec_pinhole_fovy,
    depth_directions,
    depth_scan_cpu,
    depth_mount_from_mjcf_camera,
)
from mj_reference import (
    DT, Lcg, Built, RayStats, quat_axis, qmul, set_pose_and_check_fk,
)

comptime SCENE = "tests/perception/assets/lidar_room.xml"
comptime NPOSE = 12


def _random_pose(mut rng: Lcg, mjm: PythonObject, nq: Int) raises -> List[Float64]:
    var q = List[Float64](length=nq, fill=0.0)
    var qpos0 = mjm.qpos0
    for k in range(nq):
        q[k] = Float64(py=qpos0[k])
    q[0] = rng.sym(2.2)
    q[1] = rng.sym(1.8)
    q[2] = 0.79 + rng.sym(0.03)
    var qr = qmul(
        quat_axis(0, 0, 1, rng.sym(pi)),
        qmul(quat_axis(1, 0, 0, rng.sym(0.1)), quat_axis(0, 1, 0, rng.sym(0.1))),
    )
    for k in range(4):
        q[3 + k] = qr[k]
    for k in range(7, nq):
        q[k] = rng.sym(0.6)
    return q^


def _pixel_vecs(np: PythonObject, spec: DepthCameraSpec, camera_frame: Bool) raises -> PythonObject:
    """`[H, W, 3]` pixel rays with FORWARD COMPONENT 1, built in numpy from
    the intrinsics alone. MuJoCo camera frame: (xo, -yo, -1). Sensor frame:
    (1, -xo, -yo)."""
    var u = np.arange(spec.width, dtype=np.float64)
    var v = np.arange(spec.height, dtype=np.float64)
    var xo = (u - spec.cx) / spec.fx
    var yo = (v - spec.cy) / spec.fy
    var g = np.meshgrid(xo, yo)  # [H, W] each, xo along columns
    var X = g[0]
    var Y = g[1]
    var one = np.ones_like(X)
    if camera_frame:
        return np.stack(Python.list(X, -Y, -one), axis=-1)
    return np.stack(Python.list(one, -X, -Y), axis=-1)


def test_depth_vs_mj_ray() raises:
    var mujoco = Python.import_module("mujoco")
    var np = Python.import_module("numpy")
    var mjm = mujoco.MjModel.from_xml_path(SCENE)
    var mjd = mujoco.MjData(mjm)
    var b = Built(SCENE)
    var nq = b.dims.get_nq()
    var body_root = Int(py=mjm.jnt_bodyid[0])
    var ghost = Int(py=mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "ghost"))
    var cam = Int(py=mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_CAMERA, "robot_head_camera"))
    assert_true(cam >= 0, "no robot_head_camera in the scene")

    var specs = List[DepthCameraSpec]()
    specs.append(load_depth_spec("noeira/perception/zoo/d435_class_848x480.depth").ideal())
    specs.append(load_depth_spec("noeira/perception/zoo/tof_class_640x576.depth").ideal())
    specs.append(load_depth_spec("noeira/perception/zoo/d435_class_424x240.depth").ideal())

    # Mount 0: the MJCF head camera. 1: tilted on the pelvis, looking ahead
    # and down. 2: a world rig in a corner looking into the room.
    var mounts = List[LidarMount]()
    mounts.append(depth_mount_from_mjcf_camera(b.m, cam))
    var qt = qmul(quat_axis(0, 0, 1, 0.3), quat_axis(0, 1, 0, 0.5))
    mounts.append(LidarMount(body_root, 0.10, 0.0, 0.05, qt[0], qt[1], qt[2], qt[3]))
    var qw = qmul(quat_axis(0, 0, 1, 0.7), quat_axis(0, 1, 0, 0.25))
    mounts.append(LidarMount(0, -2.6, -2.1, 1.4, qw[0], qw[1], qw[2], qw[3]))

    var rng = Lcg(0xDE97)
    var st = RayStats()
    for pose in range(NPOSE):
        var q = _random_pose(rng, mjm, nq)
        var fk = set_pose_and_check_fk(b, mjm, mjd, q)
        assert_true(fk < 1e-12, "forward kinematics differ by " + String(fk))
        var si = pose % len(specs)
        var mi = (pose // len(specs)) % len(mounts)
        if pose == NPOSE - 1:
            mi = 0
        ref spec = specs[si]
        var mount = mounts[mi]
        var dirs = depth_directions(spec)
        var depth = List[Float64]()
        var gids = List[Int]()
        depth_scan_cpu[DT, DynDims, 1](b.d, b.m, spec, mount, dirs, depth, gids)
        var ours = List[Float64](length=len(depth), fill=-1.0)
        for i in range(len(depth)):
            if depth[i] > 0.0:
                ours[i] = depth[i]

        var origin: PythonObject
        var vecs: PythonObject
        var excl: Int
        if mi == 0:
            # MuJoCo's own camera pose and camera axes — independent of our
            # camera-to-sensor rotation.
            origin = mjd.cam_xpos[cam]
            var cm = mjd.cam_xmat[cam].reshape(3, 3)
            vecs = _pixel_vecs(np, spec, True).reshape(-1, 3).dot(cm.T)
            excl = Int(py=mjm.cam_bodyid[cam])
        else:
            var mq = np.array(Python.list(mount.qw, mount.qx, mount.qy, mount.qz))
            var rm = np.zeros(9)
            _ = mujoco.mju_quat2Mat(rm, mq)
            rm = rm.reshape(3, 3)
            var mp = np.array(Python.list(mount.px, mount.py, mount.pz))
            var rot: PythonObject
            if mount.body > 0:
                var xm = mjd.xmat[mount.body].reshape(3, 3)
                origin = mjd.xpos[mount.body] + xm.dot(mp)
                rot = xm.dot(rm)
                excl = mount.body
            else:
                origin = mp
                rot = rm
                excl = -1
            vecs = _pixel_vecs(np, spec, False).reshape(-1, 3).dot(rot.T)
        vecs = np.ascontiguousarray(vecs).reshape(-1)
        st.add_scan(
            mujoco, np, mjm, mjd, origin, vecs, excl, ours, gids, body_root, ghost,
            "pose " + String(pose) + " " + spec.name + " mount " + String(mi),
        )
    st.report()
    st.assert_clean()


def _gl_vs_ours(
    np: PythonObject, gl: PythonObject, ours: List[Float64], w: Int, h: Int, flip: Bool
) raises -> Tuple[Float64, Float64]:
    """(fraction of pixels within 2 mm + 0.2 % of z, mean |dz| clipped at
    5 cm) over pixels where both have depth."""
    var a = np.zeros(w * h)
    var p = Pointer[Float64, MutAnyOrigin](unsafe_from_address=Int(py=a.ctypes.data))
    for i in range(w * h):
        p.unsafe_offset(i)[] = ours[i]
    var img = a.reshape(h, w)
    if flip:
        img = np.flipud(img)
    var valid = (img > 0) & (gl > 0)
    var dz = np.abs(img - gl)[valid]
    var zz = gl[valid]
    var good = Float64(py=np.mean(dz < 0.002 + 0.002 * zz))
    var mean = Float64(py=np.mean(np.minimum(dz, 0.05)))
    return (good, mean)


def test_depth_vs_mujoco_renderer() raises:
    var mujoco = Python.import_module("mujoco")
    var np = Python.import_module("numpy")
    var mjm = mujoco.MjModel.from_xml_path(SCENE)
    var mjd = mujoco.MjData(mjm)
    var b = Built(SCENE)
    var nq = b.dims.get_nq()
    var cam = Int(py=mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_CAMERA, "robot_head_camera"))
    comptime W = 160
    comptime H = 120
    var fovy = Float64(py=mjm.cam_fovy[cam])
    var base = depth_spec_pinhole_fovy(W, H, fovy, "head_camera")
    var mount = depth_mount_from_mjcf_camera(b.m, cam)
    var rend = mujoco.Renderer(mjm, H, W)
    rend.enable_depth_rendering()
    # Sites are drawn by OpenGL and not seen by rays: hide them.
    var vopt = mujoco.MjvOption()
    for g in range(6):
        vopt.sitegroup[g] = 0

    var variants = List[String]()
    variants.append("as specified")
    variants.append("cx + 0.5 px")
    variants.append("cy + 0.5 px")
    variants.append("rows flipped")
    var goods = List[Float64](length=4, fill=0.0)
    var means = List[Float64](length=4, fill=0.0)
    var rng = Lcg(0x6E7)
    comptime NP = 6
    for pose in range(NP):
        var q = _random_pose(rng, mjm, nq)
        _ = set_pose_and_check_fk(b, mjm, mjd, q)
        rend.update_scene(mjd, camera=cam, scene_option=vopt)
        var gl = rend.render().astype(np.float64)
        for k in range(4):
            var spec = base.copy()
            if k == 1:
                spec.cx += 0.5
            elif k == 2:
                spec.cy += 0.5
            var dirs = depth_directions(spec)
            var depth = List[Float64]()
            var gids = List[Int]()
            depth_scan_cpu[DT, DynDims, 1](b.d, b.m, spec, mount, dirs, depth, gids)
            var r = _gl_vs_ours(np, gl, depth, W, H, k == 3)
            goods[k] += r[0] / Float64(NP)
            means[k] += r[1] / Float64(NP)
    for k in range(4):
        print("  ", variants[k], ": within 2 mm + 0.2 %:", goods[k], "  mean |dz|", means[k])
    assert_true(goods[0] > 0.98, "only " + String(goods[0]) + " of pixels match MuJoCo's depth image")
    for k in range(1, 4):
        assert_true(
            means[0] < 0.5 * means[k],
            "the camera " + variants[k] + " matches MuJoCo almost as well ("
            + String(means[k]) + " vs " + String(means[0]) + "): the gate cannot see the convention",
        )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
