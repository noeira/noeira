"""The so101-tower rig's overhead pose reader: one place for its camera, desk,
colours and confidence rule, and the arm's FK to any site.

    from noeira.tasks.so101_tower_overhead import (
        tower_overhead_camera, tower_desk_roi, printed_brick_hsv,
        printed_bowl_hsv, pose_confident, TowerArmFK,
    )

Three tools read real overhead frames with `vision/tabletop_pose.mojo`
(`tower_pose_real_check`, `tower_pose_live`, `tower_pose_arm_calib`) and the
rig executor will be a fourth; each spelling the ROI, the colours or the
confidence rule on its own is `_a_rule_written_inline_twice_drifts`.

## The camera (`tower_overhead_camera`)

The lens is always the measured one (`--calib`, the fisheye intrinsics file).
The POSE is, in order of preference:
- a calibration file WITH extrinsics written by `tower_pose_arm_calib.mojo`
  (`--extrinsics FILE`): the asset camera corrected against the arm itself;
- the ASSET `overhead_cam` (`tower_sim_camera`), which puts the sim desk on
  the real desk's edges.
⚠ NEVER the extrinsics inside `camera_overhead.txt` itself: those were fitted
on 2026-09-22 under the wrong joint zero (12 mm rms,
`_the_camera_error_was_the_arms_joint_zero`). So the lens file's own
extrinsics are ignored, and an `--extrinsics` file must say it came from the
arm calibration (its `name` ends in `_armcal`).

Calibration extrinsics are the file format's: camera -> BASE rotation in
OpenCV axes and the camera origin in the BASE frame. The base sits at the
family's `base_pos` in the world (z 0.005), which `tower_sim_camera` reports
as `base_off`.

## The desk and the confidence rule

`tower_desk_roi`: world x 0.08..0.57, |y| <= 0.28 — the real desk short of its
edges (its +y edge is at ~0.29 and the floor past it is blue) and of the
tower's foot. `pose_confident`: coverage in [0.75, 1.3], residual < 0.35 —
what the tools score and what the rig may act on.
"""

from std.math import cos, sin

from noeira.math3d import Mat3 as Mat3Generic, Quat as QuatGeneric, Vec3 as Vec3Generic
from noeira.physics3d.fields import Data, Model, DynDims
from noeira.physics3d.kinematics.forward_kinematics import forward_kinematics
from noeira.physics3d.parser.runtime_load import (
    parse_model_runtime, dims_from_flat, build_model_runtime,
)
from noeira.tasks.family import scene_path
from noeira.tasks.spec import load_family
from noeira.tasks.so101_tower_camera_pose import tower_sim_camera, fit_to_mujoco_rot
from noeira.tasks.so101_tower_xml import (
    SO101_TOWER_MAX_CONTACTS, SO101_TOWER_NMESH_VERTS,
)
from noeira.vision.calib_file import CameraCalib, read_calib
from noeira.vision.fisheye import FisheyeLens
from noeira.vision.tabletop_pose import RigCamera, ColorClass, DeskROI, PoseEstimate

comptime Vec3d = Vec3Generic[DType.float64]
comptime Mat3d = Mat3Generic[DType.float64]
comptime Quatd = QuatGeneric[DType.float64]

comptime TOWER_FAMILY = "noeira/tasks/families/so101_tower.family"
comptime OVERHEAD_CALIB = "projects/so101-tower/cameras/camera_overhead.txt"
comptime ARMCAL_SUFFIX = "_armcal"
comptime DESK_Z = 0.002
"""The desk mat's surface, world z (`desk_mat.xml` surface site)."""


def tower_desk_roi() -> DeskROI:
    return DeskROI(DESK_Z, 0.08, 0.57, -0.28, 0.28)


def printed_brick_hsv() -> ColorClass:
    """The printed brick on the rig: hue 195..216 (p5..p95), v 0.40..0.65 on
    `cube-in-bowl-printed`."""
    return ColorClass(205.0, 13.0, 0.3, 0.2, 1.0)


def printed_bowl_hsv() -> ColorClass:
    """The printed bowl on the rig: hue 29..42 (p5..p95), s >= 0.44."""
    return ColorClass(36.0, 10.0, 0.35, 0.3, 1.0)


def pose_confident(e: PoseEstimate) -> Bool:
    return e.found and e.coverage >= 0.75 and e.coverage <= 1.3 and e.residual < 0.35


struct OverheadPose(Copyable, Movable):
    """A camera pose in the WORLD, MuJoCo axes, and where it came from."""

    var pos: Vec3d
    var rot_mj: Mat3d
    var source: String
    var base_off: Vec3d

    def __init__(out self, pos: Vec3d, rot_mj: Mat3d, var source: String, base_off: Vec3d):
        self.pos = pos
        self.rot_mj = rot_mj
        self.source = source^
        self.base_off = base_off


def tower_overhead_pose(extrinsics_path: String = String("")) raises -> OverheadPose:
    """The asset pose, or an arm calibration's (see the module header)."""
    var sim = tower_sim_camera("overhead_cam")
    if not sim.found:
        raise Error("no overhead_cam in the tower scene")
    if extrinsics_path == "":
        return OverheadPose(sim.pos, sim.rot, String("asset overhead_cam"), sim.base_off)
    var c = read_calib(extrinsics_path)
    if not c.has_extrinsics:
        raise Error(extrinsics_path + " has no extrinsics")
    if not c.name.endswith(ARMCAL_SUFFIX):
        raise Error(
            extrinsics_path + " is calibration '" + c.name + "', not an arm"
            " calibration ('*" + ARMCAL_SUFFIX + "', tower_pose_arm_calib.mojo)."
            " The lens file's own extrinsics predate the follower's joint zero"
            " and are never used."
        )
    return OverheadPose(
        c.trans + sim.base_off, fit_to_mujoco_rot(c.rot),
        String("arm calibration ") + extrinsics_path + " (" + String(c.poses)
        + " pairs, " + String(c.rms_mm) + " mm rms)",
        sim.base_off,
    )


def tower_overhead_camera(
    calib_path: String = String(OVERHEAD_CALIB),
    extrinsics_path: String = String(""),
) raises -> RigCamera:
    """The measured fisheye at `tower_overhead_pose(extrinsics_path)`."""
    var cal = read_calib(calib_path)
    cal.require_size(640, 480)
    var lens = FisheyeLens.from_calib(cal)
    var p = tower_overhead_pose(extrinsics_path)
    return RigCamera(lens, p.pos, p.rot_mj)


struct TowerArmFK(Movable):
    """The tower scene's CPU FK: the arm's six model joint values -> a site's
    world position, or a camera's world pose (the props parked, free joints
    at unit quaternions)."""

    var m: Model[DType.float64, DynDims]
    var d: Data[DType.float64, DynDims, 1]
    var site_names: List[String]
    var site_body: List[Int]
    var body_names: List[String]
    var cam_names: List[String]
    var cam_body: List[Int]
    var cam_pos: List[Vec3d]
    """Each camera's position in its body's frame."""
    var cam_rot: List[Mat3d]
    """Each camera's orientation in its body's frame, MuJoCo axes."""
    var lo: List[Float64]
    """The arm's six joint limits (the model's), for `SimJointMap`."""
    var hi: List[Float64]

    def __init__(out self) raises:
        var f = load_family(String(TOWER_FAMILY))
        var fmd = parse_model_runtime(scene_path(f))
        var dims = dims_from_flat(
            fmd, max_contacts=SO101_TOWER_MAX_CONTACTS,
            nmesh_verts=SO101_TOWER_NMESH_VERTS,
        )
        var m = Model[DType.float64, DynDims](dims)
        build_model_runtime[DType.float64](fmd, dims, m)
        var d = Data[DType.float64, DynDims, 1](dims)
        for k in range(dims.get_nq()):
            d.qpos.data[k] = 0.0
        var adr = 0
        for j in range(len(fmd.joints)):
            if fmd.joints[j].nq == 7:
                d.qpos.data[adr + 3] = 1.0
            adr += fmd.joints[j].nq
        var names = List[String]()
        var bodies = List[Int]()
        for s in range(len(fmd.site_names)):
            names.append(String(fmd.site_names[s]))
            bodies.append(fmd.sites[s].body_id)
        var bnames = List[String]()
        for b in range(len(fmd.body_names)):
            bnames.append(String(fmd.body_names[b]))
        var cnames = List[String]()
        var cbody = List[Int]()
        var cpos = List[Vec3d]()
        var crot = List[Mat3d]()
        for c in range(len(fmd.camera_names)):
            ref cc = fmd.cameras[c]
            cnames.append(String(fmd.camera_names[c]))
            cbody.append(cc.body_id)
            cpos.append(Vec3d(cc.pos_x, cc.pos_y, cc.pos_z))
            crot.append(Mat3d.from_quat(Quatd(cc.quat_w, cc.quat_x, cc.quat_y, cc.quat_z)))
        self.m = m^
        self.d = d^
        self.site_names = names^
        self.site_body = bodies^
        self.body_names = bnames^
        self.cam_names = cnames^
        self.cam_body = cbody^
        self.cam_pos = cpos^
        self.cam_rot = crot^
        var lo = List[Float64]()
        var hi = List[Float64]()
        for k in range(6):
            lo.append(fmd.joints[k].range_min)
            hi.append(fmd.joints[k].range_max)
        self.lo = lo^
        self.hi = hi^

    def site_index(self, suffix: String) raises -> Int:
        for s in range(len(self.site_names)):
            if self.site_names[s].endswith(suffix):
                return s
        raise Error("the tower scene has no site '*" + suffix + "'")

    def set_qpos(mut self, q: List[Float64]) raises:
        """The arm's six joints in MODEL radians (unclamped: FK does not care
        about limits, and the real lift goes past the model's)."""
        for k in range(6):
            self.d.qpos.data[k] = q[k]
        forward_kinematics["cpu", DType.float64, DynDims, 1](self.d, self.m)

    def site_pos(self, s: Int) -> Vec3d:
        return Vec3d(
            Float64(self.d.site_xpos.data[s * 3]),
            Float64(self.d.site_xpos.data[s * 3 + 1]),
            Float64(self.d.site_xpos.data[s * 3 + 2]),
        )

    def body_index(self, name: String) raises -> Int:
        for b in range(len(self.body_names)):
            if self.body_names[b] == name:
                return b
        raise Error("the tower scene has no body '" + name + "'")

    def body_pos(self, b: Int) -> Vec3d:
        """Body `b`'s world origin at the last `set_qpos`."""
        return Vec3d(
            Float64(self.d.xpos.data[b * 3]), Float64(self.d.xpos.data[b * 3 + 1]),
            Float64(self.d.xpos.data[b * 3 + 2]),
        )

    def camera_index(self, suffix: String) raises -> Int:
        for c in range(len(self.cam_names)):
            if self.cam_names[c].endswith(suffix):
                return c
        raise Error("the tower scene has no camera '*" + suffix + "'")

    def camera_pose(self, c: Int) -> Tuple[Vec3d, Mat3d]:
        """Camera `c`'s world (position, orientation in MuJoCo axes) at the
        last `set_qpos` — what `RigCamera` takes. `tower_sim_camera`'s
        composition, at any arm pose: the wrist camera rides the gripper."""
        var b = self.cam_body[c]
        var br = Mat3d.from_quat(Quatd(
            Float64(self.d.xquat.data[b * 4 + 3]), Float64(self.d.xquat.data[b * 4]),
            Float64(self.d.xquat.data[b * 4 + 1]), Float64(self.d.xquat.data[b * 4 + 2]),
        ))
        var bp = Vec3d(
            Float64(self.d.xpos.data[b * 3]), Float64(self.d.xpos.data[b * 3 + 1]),
            Float64(self.d.xpos.data[b * 3 + 2]),
        )
        return (bp + br * self.cam_pos[c], br @ self.cam_rot[c])

    def camera_body_pose(self, c: Int) -> Tuple[Vec3d, Mat3d]:
        """Camera `c`'s parent body's world (position, rotation) at the last
        `set_qpos` — to compose a corrected camera-in-body transform."""
        var b = self.cam_body[c]
        var br = Mat3d.from_quat(Quatd(
            Float64(self.d.xquat.data[b * 4 + 3]), Float64(self.d.xquat.data[b * 4]),
            Float64(self.d.xquat.data[b * 4 + 1]), Float64(self.d.xquat.data[b * 4 + 2]),
        ))
        var bp = Vec3d(
            Float64(self.d.xpos.data[b * 3]), Float64(self.d.xpos.data[b * 3 + 1]),
            Float64(self.d.xpos.data[b * 3 + 2]),
        )
        return (bp, br)

    def site_body_rot(self, s: Int) -> Mat3d:
        """The world rotation of the site's BODY (`Data.xquat` is packed
        x, y, z, w; `Quat` takes w, x, y, z)."""
        var b = self.site_body[s]
        return Mat3d.from_quat(Quatd(
            Float64(self.d.xquat.data[b * 4 + 3]), Float64(self.d.xquat.data[b * 4]),
            Float64(self.d.xquat.data[b * 4 + 1]), Float64(self.d.xquat.data[b * 4 + 2]),
        ))
