"""The real-arm preprocessing sees what the sim student was trained on.

    pixi run mojo run -I . tests/tasks/test_pixel_student_crop.mojo

The pixel student (`noeira/tasks/pixel_student.mojo`) was trained on each
rig camera traced as a SQUARE (`RENDER` x `RENDER`, the model's fovy in
both directions) and block-averaged to `OBS_PX`. On the arm, a camera
undistorted to the sim pinhole delivers a 4:3 frame whose vertical field is
that same fovy, and `frame_to_planes` keeps its centred square. So:

1. one posed tower scene is traced at 64x64 (the trainer's picture) and at
   128x96 (a 4:3 frame at the same fovy — what the undistorted real camera
   is, in miniature), both through the rig's renderer on the CPU;
2. SIM path: the 64x64 averaged over 4x4 blocks, -0.5 (the trainer's
   `_pack_camera_kernel`);
3. REAL path: the 128x96 packed to CHW uint8 as a camera frame, then
   `frame_to_planes` — the deploy's own function;
4. they agree to a small mean error, for both cameras;
5. ⚠ THE CONTROL: the 4:3 frame SQUEEZED whole into 16x16 (no crop — the
   mistake the crop exists to avoid) disagrees several times more. Without
   it, a small error would prove nothing about the geometry.
"""

from std.math import abs
from std.testing import assert_true
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.physics3d.fields import Data
from noeira.physics3d.kinematics.forward_kinematics import forward_kinematics
from noeira.physics3d.parser.runtime_load import parse_model_runtime
from noeira.tasks.family import scene_path
from noeira.tasks.family_config import So101TowerConfig
from noeira.tasks.pixel_student import OBS_PX, PLANE, IN_DIM, frame_to_planes
from noeira.tasks.placement.so101_tower import So101TowerPlacement
from noeira.tasks.posed_reset import posed_qpos
from noeira.tasks.so101_tower_rig import (
    RIG_DT, TOWER_MD, make_tower_model, make_tower_renderer, tower_cameras,
    rig_byte,
)
from noeira.tasks.so101_tower_xml import So101TowerModel
from noeira.tasks.spec import load_family

comptime SQ = 64
comptime W43 = 128
comptime H43 = 96


def main() raises:
    var ctx = DeviceContext()
    var f = load_family("noeira/tasks/families/so101_tower.family")
    var fmd = parse_model_runtime(scene_path(f))
    var cams = tower_cameras(fmd)
    var rm = make_tower_model(ctx)
    var rd = Data[RIG_DT, TOWER_MD, 1]()
    var q = posed_qpos[So101TowerPlacement](
        String("so101_tower_cube_in_bowl"), String("so101_tower"),
        So101TowerConfig.SLOT_RADIUS, seed=7,
    )
    for k in range(So101TowerModel.NQ):
        rd.qpos.data[k] = Scalar[RIG_DT](q[k])
    forward_kinematics["cpu", RIG_DT, TOWER_MD, 1](rd, rm)
    var r_sq = make_tower_renderer[1, SQ, SQ, 1](ctx, fmd, rm)
    var r_43 = make_tower_renderer[1, W43, H43, 1](ctx, fmd, rm)
    var names: List[String] = ["overhead", "wrist"]
    for k in range(2):
        r_sq.cam = cams[k]
        r_43.cam = cams[k]
        var rgb_sq = List[Scalar[RIG_DT]]()
        var rgb_43 = List[Scalar[RIG_DT]]()
        var dd = List[Scalar[RIG_DT]]()
        var ss = List[Scalar[RIG_DT]]()
        r_sq.render_cpu(rd, rm, rgb_sq, dd, ss)
        r_43.render_cpu(rd, rm, rgb_43, dd, ss)
        # SIM path: 4x4 blocks of the square
        comptime F = SQ // OBS_PX
        var sim = List[Float64](length=3 * PLANE, fill=0.0)
        for c in range(3):
            for oy in range(OBS_PX):
                for ox in range(OBS_PX):
                    var acc = 0.0
                    for dy in range(F):
                        for dx in range(F):
                            var px = (oy * F + dy) * SQ + ox * F + dx
                            acc += Float64(rgb_sq[px * 3 + c])
                    sim[c * PLANE + oy * OBS_PX + ox] = acc / Float64(F * F) - 0.5
        # REAL path: CHW uint8 frame -> frame_to_planes (camera slot 0)
        var frame = List[UInt8](length=3 * W43 * H43, fill=UInt8(0))
        for c in range(3):
            for p in range(W43 * H43):
                frame[c * W43 * H43 + p] = rig_byte(Float64(rgb_43[p * 3 + c]))
        var x = List[Scalar[DT]](length=IN_DIM, fill=Scalar[DT](0))
        frame_to_planes(frame, W43, H43, 0, x)
        # CONTROL: the whole 4:3 frame squeezed into OBS_PX x OBS_PX
        comptime BX = W43 // OBS_PX
        comptime BY = H43 // OBS_PX
        var err = 0.0
        var err_sq = 0.0
        for c in range(3):
            for oy in range(OBS_PX):
                for ox in range(OBS_PX):
                    var i = c * PLANE + oy * OBS_PX + ox
                    err += abs(Float64(x[i]) - sim[i])
                    var acc = 0.0
                    for dy in range(BY):
                        for dx in range(BX):
                            acc += Float64(frame[c * W43 * H43 + (oy * BY + dy) * W43 + ox * BX + dx])
                    var squeezed = acc / (255.0 * Float64(BX * BY)) - 0.5
                    err_sq += abs(squeezed - sim[i])
        err /= Float64(3 * PLANE)
        err_sq /= Float64(3 * PLANE)
        print("  " + names[k] + ": mean |real path - sim path| =", err,
              "| squeezed 4:3 (the control) =", err_sq)
        assert_true(err < 0.03, names[k] + ": the crop does not reproduce the sim picture")
        assert_true(err_sq > 3.0 * err, names[k] + ": the control does not separate — the gate is blind")
    print("PIXEL STUDENT CROP OK")
