"""The `so101_tower` rig's two cameras — OUR composed world poses vs MuJoCo's.

    pixi run mojo run -I . tests/tasks/test_so101_tower_cameras.mojo

## WHAT THIS GATES

The composed scene `scenes/so101_tower.xml` carries `robot_wrist_cam` (on the
`gripper` body, through the printed mount) and `tower_overhead_cam` (on the
static stand). The batched tracer renders from wherever
`raytrace/camera.camera_world_frame` says they are — `mj_camlight`'s
parent-body rule over OUR parse of the attached assets and OUR forward
kinematics. This pins both against `mjData.cam_xpos` / `cam_xmat` from
`tools/tasks/check_tower_cameras.py`, at TWO arm configurations:

  * `rest`  — every joint at 0; the overhead camera's pose is what it is at
    every configuration (its body is static), so this is ITS check;
  * `moved` — every joint off zero, `wrist_roll` at 2.2 rad: the wrist
    camera's parent body is somewhere else entirely, so a composition that
    dropped the rotation, the translation or the attach frame's 5 mm lift
    cannot pass. The control below asserts the two wrist poses differ by more
    than 5 cm, so the check cannot go vacuous on a model that stops moving.

Two routes to one pose: MuJoCo compiles each attached asset and attaches the
RESULT; we splice text (`parser/expander`), then compose. Numbers are pinned
at 1e-6 m / 1e-6 on the axes — far below the ±3 mm the CAD-derived pupil
carries (`docs/camera-rig.md` §4) — because this gate is about the
COMPOSITION, not the rig: calibration (§7) moves the numbers in the assets,
and both parsers will read the new ones.

⚠ THE ORACLE NUMBERS BELOW ARE COPIED FROM THE PYTHON TOOL'S OUTPUT, and go
stale when the assets change on purpose (a calibrated pose, a moved stand).
Re-run the tool and paste. Both tools read the same XML by design — what is
independent is everything after the text.
"""

from std.math import abs, sqrt
from std.testing import assert_true

from noeira.math3d import Vec3 as Vec3Generic
from noeira.physics3d.fields import Data, Model, DynDims
from noeira.physics3d.fields.rt_layout import DYN1, DYN2, rl1, rl2
from noeira.physics3d.gpu.constants import MAX_GPU_CAMERAS, MODEL_CAM_SIZE
from noeira.physics3d.kinematics.forward_kinematics import forward_kinematics
from noeira.physics3d.parser.runtime_load import (
    parse_model_runtime, dims_from_flat, build_model_runtime,
)
from noeira.physics3d.raytrace.camera import camera_world_frame
from noeira.tasks.family import scene_path
from noeira.tasks.spec import load_family
from noeira.tasks.so101_tower_xml import (
    SO101_TOWER_MAX_CONTACTS, SO101_TOWER_NMESH_VERTS,
)

comptime DT = DType.float64
comptime Vec3 = Vec3Generic[DT]
comptime FAMILY = "noeira/tasks/families/so101_tower.family"
comptime TOL = 1.0e-6

# `tools/tasks/check_tower_cameras.py`, MuJoCo 3.12.0 — pos, then the x, y, z
# columns of cam_xmat, for (pose, camera). Regenerated 2026-09-25 when the
# gripper body moved 6.5 mm out along its roll axis (bake step 4a): the wrist
# camera rides it, so only its two positions changed.
comptime N_POSE = 2
comptime N_CAM = 2


def _pose(p: Int, i: Int) -> Float64:
    if p == 0:
        return 0.0
    if i == 0:
        return 0.35
    if i == 1:
        return -1.10
    if i == 2:
        return 0.90
    if i == 3:
        return 0.60
    if i == 4:
        return 2.20
    return 0.45


def _oracle(p: Int, c: Int, k: Int) -> Vec3:
    """k: 0 pos, 1 xaxis, 2 yaxis, 3 zaxis."""
    if p == 0 and c == 0:
        if k == 0:
            return Vec3(0.312779600, -0.064230037, 0.238752468)
        if k == 1:
            return Vec3(-0.000000000, 0.048660291, -0.998815386)
        if k == 2:
            return Vec3(0.424307973, -0.904445254, -0.044062767)
        return Vec3(-0.905517942, -0.423805332, -0.020646950)
    if c == 1:
        # the overhead camera is on a static body: the same at both poses
        # (the calibrated pose of 2026-09-25, `so101_tower_stand.xml`)
        if k == 0:
            return Vec3(0.041670000, -0.116980000, 0.536660000)
        if k == 1:
            return Vec3(0.005899757, -0.999958793, 0.006899716)
        if k == 2:
            return Vec3(0.904211175, 0.008281183, 0.427005355)
        return Vec3(-0.427044897, 0.003719572, 0.904222772)
    # moved, wrist
    if k == 0:
        return Vec3(0.209851956, -0.023019094, 0.246140070)
    if k == 1:
        return Vec3(-0.086090245, -0.858716797, 0.505167232)
    if k == 2:
        return Vec3(0.814397372, 0.231417602, 0.532168032)
    return Vec3(-0.573886217, 0.457221342, 0.679413905)


def _dist(a: Vec3, b: Vec3) -> Float64:
    var dx = Float64(a.x - b.x)
    var dy = Float64(a.y - b.y)
    var dz = Float64(a.z - b.z)
    return sqrt(dx * dx + dy * dy + dz * dz)


def main() raises:
    print("=== so101_tower cameras: our composition vs MuJoCo ===")
    var f = load_family(String(FAMILY))
    var fmd = parse_model_runtime(scene_path(f))
    assert_true(len(fmd.camera_names) == N_CAM, "the scene has two cameras")
    assert_true(String(fmd.camera_names[0]) == "robot_wrist_cam")
    assert_true(String(fmd.camera_names[1]) == "tower_overhead_cam")
    var dims = dims_from_flat(
        fmd, max_contacts=SO101_TOWER_MAX_CONTACTS,
        nmesh_verts=SO101_TOWER_NMESH_VERTS,
    )
    var m = Model[DT, DynDims](dims)
    build_model_runtime[DT](fmd, dims, m)
    var d = Data[DT, DynDims, 1](dims)
    var nq = dims.get_nq()
    var nb = dims.get_nbody()
    var worst = 0.0
    var wrist_rest = Vec3(0.0, 0.0, 0.0)
    var wrist_moved = Vec3(0.0, 0.0, 0.0)
    for p in range(N_POSE):
        for i in range(nq):
            d.qpos.data[i] = Scalar[DT](0)
        for i in range(6):
            d.qpos.data[i] = Scalar[DT](_pose(p, i))
        # the free slots' quaternions must be unit, as MuJoCo's reset leaves them
        for j in range(len(fmd.joints)):
            if fmd.joints[j].nq == 7:
                var adr = 0
                for k in range(j):
                    adr += fmd.joints[k].nq
                d.qpos.data[adr + 3] = Scalar[DT](1)
        forward_kinematics["cpu", DT, DynDims, 1](d, m)
        var xpos_c = d.xpos.lt_dyn["cpu", DYN2](rl2(1, nb * 3))
        var xquat_c = d.xquat.lt_dyn["cpu", DYN2](rl2(1, nb * 4))
        var com_c = d.subtree_com.lt_dyn["cpu", DYN2](rl2(1, nb * 3))
        var cams_c = m.cameras.lt_dyn["cpu", DYN1](
            rl1(MAX_GPU_CAMERAS * MODEL_CAM_SIZE)
        )
        for c in range(N_CAM):
            var fr = camera_world_frame[DT](
                cams_c, xpos_c, xquat_c, com_c, 0, c
            )
            var e0 = _dist(fr.pos, _oracle(p, c, 0))
            var e1 = _dist(fr.xaxis, _oracle(p, c, 1))
            var e2 = _dist(fr.yaxis, _oracle(p, c, 2))
            var e3 = _dist(fr.zaxis, _oracle(p, c, 3))
            var e = e0
            if e1 > e:
                e = e1
            if e2 > e:
                e = e2
            if e3 > e:
                e = e3
            if e > worst:
                worst = e
            print(
                "  pose", p, String(fmd.camera_names[c]), "pos (",
                fr.pos.x, fr.pos.y, fr.pos.z, ") |dpos|", e0,
                "|daxes|", e1, e2, e3,
            )
            assert_true(
                e < TOL,
                "camera " + String(fmd.camera_names[c]) + " at pose "
                + String(p) + " is off MuJoCo by " + String(e),
            )
            if c == 0:
                if p == 0:
                    wrist_rest = fr.pos
                else:
                    wrist_moved = fr.pos
    # the control: the wrist camera MOVED, so the composition was exercised
    var travel = _dist(wrist_rest, wrist_moved)
    print("  wrist camera travel between poses:", travel, "m (control)")
    assert_true(
        travel > 0.05,
        "the wrist camera barely moved between the poses — the composition"
        " check is vacuous",
    )
    print("  worst disagreement:", worst, "(tol", TOL, ")")
    print("=== PASS ===")
