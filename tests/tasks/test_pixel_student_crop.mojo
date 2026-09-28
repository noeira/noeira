"""The real-arm preprocessing sees what the sim student was trained on.

    pixi run mojo run -I . tests/tasks/test_pixel_student_crop.mojo
    pixi run mojo run -I . -D DAGGER_WINDOW -D DAGGER_PX_32 tests/tasks/test_pixel_student_crop.mojo

The pixel student (`noeira/tasks/pixel_student.mojo`) is trained on each rig
camera TRACED at the trainer's resolution (`OVERHEAD_RENDER_W x _H` for the
overhead — the full 4:3 frame with `DAGGER_WINDOW`, the centred square
without; `RENDER` square for the wrist) and averaged over its WINDOW
(`camera_window`, area-weighted). On the arm, a camera undistorted to the sim
pinhole delivers a 4:3 frame of the same field, and `frame_to_planes` takes
the same window of it. So, for one posed tower scene and each camera:

1. SIM path: the trainer's trace, `render_to_planes` (the host twin of the
   trainer's `_pack_camera_kernel`, gated against it by
   `test_pixel_window_kernel.mojo`);
2. REAL path: a 4:3 trace at a higher resolution (the undistorted real
   camera, in miniature), packed to CHW uint8 as a camera frame, then
   `frame_to_planes` — the deploy's own function;
3. they agree to a small mean error;
4. ⚠ THE CONTROL: the whole 4:3 frame SQUEEZED into OBS_PX x OBS_PX (no
   window — the mistake the window exists to avoid) disagrees several times
   more. Without it, a small error would prove nothing about the geometry.
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
from noeira.tasks.pixel_student import (
    OBS_PX, PLANE, IN_DIM, RENDER, OVERHEAD_RENDER_W, OVERHEAD_RENDER_H,
    WINDOWED, frame_to_planes, render_to_planes,
)
from noeira.tasks.placement.so101_tower import So101TowerPlacement
from noeira.tasks.posed_reset import posed_qpos
from noeira.tasks.so101_tower_rig import (
    RIG_DT, TOWER_MD, make_tower_model, make_tower_renderer, tower_cameras,
    rig_byte,
)
from noeira.tasks.so101_tower_xml import So101TowerModel
from noeira.tasks.spec import load_family

comptime W43 = 256
comptime H43 = 192
"""The 'real' frame: 4:3, finer than any training trace."""


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
    var r_w = make_tower_renderer[1, RENDER, RENDER, 1](ctx, fmd, rm)
    var r_o = make_tower_renderer[1, OVERHEAD_RENDER_W, OVERHEAD_RENDER_H, 1](ctx, fmd, rm)
    var r_43 = make_tower_renderer[1, W43, H43, 1](ctx, fmd, rm)
    var names: List[String] = ["overhead", "wrist"]
    print("  build:", OBS_PX, "px,", "workspace window" if WINDOWED else "centre square")
    for k in range(2):
        var dd = List[Scalar[RIG_DT]]()
        var ss = List[Scalar[RIG_DT]]()
        # SIM path: the trainer's trace of this camera, through the host twin
        var rgb_tr = List[Scalar[RIG_DT]]()
        var sim = List[Scalar[DT]](length=IN_DIM, fill=Scalar[DT](0))
        if k == 0:
            r_o.cam = cams[k]
            r_o.render_cpu(rd, rm, rgb_tr, dd, ss)
            render_to_planes(rgb_tr, OVERHEAD_RENDER_W, OVERHEAD_RENDER_H, k, sim)
        else:
            r_w.cam = cams[k]
            r_w.render_cpu(rd, rm, rgb_tr, dd, ss)
            render_to_planes(rgb_tr, RENDER, RENDER, k, sim)
        # REAL path: a finer 4:3 frame, CHW uint8, through the deploy's function
        r_43.cam = cams[k]
        var rgb_43 = List[Scalar[RIG_DT]]()
        r_43.render_cpu(rd, rm, rgb_43, dd, ss)
        var frame = List[UInt8](length=3 * W43 * H43, fill=UInt8(0))
        for c in range(3):
            for p in range(W43 * H43):
                frame[c * W43 * H43 + p] = rig_byte(Float64(rgb_43[p * 3 + c]))
        var real = List[Scalar[DT]](length=IN_DIM, fill=Scalar[DT](0))
        frame_to_planes(frame, W43, H43, k, real)
        # CONTROL: the whole 4:3 frame squeezed into OBS_PX x OBS_PX
        comptime BX = W43 // OBS_PX
        comptime BY = H43 // OBS_PX
        var err = 0.0
        var err_sq = 0.0
        for c in range(3):
            for oy in range(OBS_PX):
                for ox in range(OBS_PX):
                    var i = (3 * k + c) * PLANE + oy * OBS_PX + ox
                    err += abs(Float64(real[i]) - Float64(sim[i]))
                    var acc = 0.0
                    for dy in range(BY):
                        for dx in range(BX):
                            acc += Float64(frame[c * W43 * H43 + (oy * BY + dy) * W43 + ox * BX + dx])
                    var squeezed = acc / (255.0 * Float64(BX * BY)) - 0.5
                    err_sq += abs(squeezed - Float64(sim[i]))
        err /= Float64(3 * PLANE)
        err_sq /= Float64(3 * PLANE)
        print("  " + names[k] + ": mean |real path - sim path| =", err,
              "| squeezed 4:3 (the control) =", err_sq)
        assert_true(err < 0.03, names[k] + ": the window does not reproduce the sim picture")
        assert_true(err_sq > 3.0 * err, names[k] + ": the control does not separate — the gate is blind")
    print("PIXEL STUDENT CROP OK")
