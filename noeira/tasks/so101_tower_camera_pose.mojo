"""The so101-tower's SIMULATED camera poses, and a measured one against them.

    from noeira.tasks.so101_tower_camera_pose import (
        tower_sim_camera, camera_pose_vs_sim,
    )

`examples/so101/calibrate_camera_extrinsics.mojo` fits a camera -> robot-base
transform from marker/arm correspondences. This module says how far that
measured pose is from the camera the simulator renders (`overhead_cam` on the
tower stand), and prints the corrected `<camera pos=... xyaxes=...>` in the
camera's PARENT BODY frame, ready to paste into the asset.

## The two conventions it bridges, and how it knows it is right

* AXES: the fit is in OpenCV camera axes (x right, y DOWN, looking down
  +z); MuJoCo's camera is x right, y UP, looking down -z. The same frame with
  y and z negated: `R_mj = R_cv @ diag(1, -1, -1)` (`fit_to_mujoco_rot`).
* FRAMES: the fit is in the ROBOT BASE frame (the kinematics oracle is the
  bare SO-101 model at the origin); in the tower scene that base is composed
  at the family's `base_pos`, with no rotation.

`tests/vision/test_extrinsics_fisheye.mojo` feeds the sim camera's OWN pose
through both conversions and requires 0 mm, 0 deg and the asset's own
`<camera>` back, so a flipped axis or a missing offset fails there, not on
the rig.
"""

from std.math import acos

from noeira.math3d import Mat3 as Mat3Generic, Quat as QuatGeneric, Vec3 as Vec3Generic
from noeira.physics3d.fields import Data, Model, DynDims
from noeira.physics3d.kinematics.forward_kinematics import forward_kinematics
from noeira.physics3d.parser.runtime_load import (
    parse_model_runtime, dims_from_flat, build_model_runtime,
)
from noeira.tasks.family import scene_path
from noeira.tasks.spec import load_family
from noeira.tasks.so101_tower_xml import (
    SO101_TOWER_MAX_CONTACTS, SO101_TOWER_NMESH_VERTS,
)
from noeira.utils.fmt import fixed

comptime Vec3d = Vec3Generic[DType.float64]
comptime Mat3d = Mat3Generic[DType.float64]
comptime Quatd = QuatGeneric[DType.float64]


comptime TOWER_FAMILY = "noeira/tasks/families/so101_tower.family"


struct SimCamera(Copyable, Movable):
    """A camera of the `so101_tower` scene: its world pose (MuJoCo axes:
    x right, y UP, looking down -z), its parent body's world pose, and the
    world position of the robot base frame the extrinsics fit is in."""

    var found: Bool
    var name: String
    var pos: Vec3d
    var rot: Mat3d
    var body_pos: Vec3d
    var body_rot: Mat3d
    var base_off: Vec3d

    def __init__(out self):
        self.found = False
        self.name = String("")
        self.pos = Vec3d.zero()
        self.rot = Mat3d.identity()
        self.body_pos = Vec3d.zero()
        self.body_rot = Mat3d.identity()
        self.base_off = Vec3d.zero()


def tower_sim_camera(suffix: String) raises -> SimCamera:
    """FK of the tower scene at rest; the camera whose name ends in `suffix`."""
    var out = SimCamera()
    var fam = load_family(String(TOWER_FAMILY))
    var fmd = parse_model_runtime(scene_path(fam))
    out.base_off = Vec3d(fam.base_x, fam.base_y, fam.base_z)
    var ci = -1
    for i in range(len(fmd.camera_names)):
        if String(fmd.camera_names[i]).endswith(suffix):
            ci = i
    if ci < 0:
        return out^
    var dims = dims_from_flat(
        fmd, max_contacts=SO101_TOWER_MAX_CONTACTS,
        nmesh_verts=SO101_TOWER_NMESH_VERTS,
    )
    var m = Model[DType.float64, DynDims](dims)
    build_model_runtime[DType.float64](fmd, dims, m)
    var d = Data[DType.float64, DynDims, 1](dims)
    # free joints need a unit quaternion even at rest
    var adr = 0
    for j in range(len(fmd.joints)):
        if fmd.joints[j].nq == 7:
            d.qpos.data[adr + 3] = 1.0
        adr += fmd.joints[j].nq
    forward_kinematics["cpu", DType.float64, DynDims, 1](d, m)
    ref c = fmd.cameras[ci]
    var b = c.body_id
    # `Data.xquat` is packed (x, y, z, w); `Quat` takes (w, x, y, z)
    var bq = Quatd(
        Float64(d.xquat.data[b * 4 + 3]), Float64(d.xquat.data[b * 4 + 0]),
        Float64(d.xquat.data[b * 4 + 1]), Float64(d.xquat.data[b * 4 + 2]),
    )
    out.body_rot = Mat3d.from_quat(bq)
    out.body_pos = Vec3d(
        Float64(d.xpos.data[b * 3]), Float64(d.xpos.data[b * 3 + 1]),
        Float64(d.xpos.data[b * 3 + 2]),
    )
    var cq = Quatd(c.quat_w, c.quat_x, c.quat_y, c.quat_z)
    out.pos = out.body_pos + out.body_rot * Vec3d(c.pos_x, c.pos_y, c.pos_z)
    out.rot = out.body_rot @ Mat3d.from_quat(cq)
    out.name = String(fmd.camera_names[ci])
    out.found = True
    return out^


def fit_to_mujoco_rot(r_cv: Mat3d) -> Mat3d:
    """OpenCV camera axes (y down, +z forward) -> MuJoCo's (y up, -z
    forward): negate the y and z columns."""
    return Mat3d.from_cols(r_cv.col(0), -r_cv.col(1), -r_cv.col(2))


def _f3(v: Vec3d, scale: Float64, digits: Int) -> String:
    return (
        fixed(Float64(v.x) * scale, digits) + " " + fixed(Float64(v.y) * scale, digits)
        + " " + fixed(Float64(v.z) * scale, digits)
    )


@fieldwise_init
struct PoseDelta(Copyable, ImplicitlyCopyable, Movable):
    var dpos_mm: Vec3d
    var rot_deg: Float64
    var axis_deg: Float64
    var pos_local: Vec3d
    var x_local: Vec3d
    var y_local: Vec3d


def pose_delta(ref sim: SimCamera, r_cv: Mat3d, t_base: Vec3d) -> PoseDelta:
    """A measured camera (camera -> base, OpenCV axes; origin in the base
    frame) against the sim's: position error, rotation angle, optical-axis
    angle, and the measured pose in the sim camera's parent body frame."""
    var r_mj = fit_to_mujoco_rot(r_cv)
    var p_w = t_base + sim.base_off
    var rel = sim.rot.transpose() @ r_mj
    var c = (Float64(rel.trace()) - 1.0) / 2.0
    c = 1.0 if c > 1.0 else (-1.0 if c < -1.0 else c)
    var ca = Float64((-sim.rot.col(2)).dot(-r_mj.col(2)))
    ca = 1.0 if ca > 1.0 else (-1.0 if ca < -1.0 else ca)
    var r_l = sim.body_rot.transpose() @ r_mj
    return PoseDelta(
        (p_w - sim.pos) * 1000.0,
        acos(c) * 180.0 / 3.141592653589793,
        acos(ca) * 180.0 / 3.141592653589793,
        sim.body_rot.transpose() * (p_w - sim.body_pos),
        r_l.col(0),
        r_l.col(1),
    )


def camera_pose_vs_sim(ref sim: SimCamera, r_cv: Mat3d, t_base: Vec3d) -> String:
    """`pose_delta`, as the lines the extrinsics tool prints."""
    var d = pose_delta(sim, r_cv, t_base)
    return (
        String("vs sim ") + sim.name + ": position off by " + _f3(d.dpos_mm, 1.0, 1)
        + " mm (" + fixed(Float64(d.dpos_mm.length()), 1) + "), rotation off by "
        + fixed(d.rot_deg, 2) + " deg, optical axis off by " + fixed(d.axis_deg, 2)
        + " deg\n  measured, in the parent body's frame:  <camera pos=\""
        + _f3(d.pos_local, 1.0, 5) + "\" xyaxes=\"" + _f3(d.x_local, 1.0, 5) + " "
        + _f3(d.y_local, 1.0, 5) + "\"/>"
    )
