"""Where the props can be in the overhead camera — the window its policy
crop must cover.

    pixi run mojo run -I . tools/tasks/tower_overhead_roi.mojo [TASK] [N]

The pixel student's overhead camera sees the whole room; the brick and the
bowl only ever start inside the placement regions. This traces the overhead
camera (the rig renderer's CPU path, the square picture the student is
trained on) for N placements of TASK's `init=` draws and marks every pixel
that differs from the same scene with both props moved out of view — the
props' footprint, whatever the segmentation channel indexes. It prints the
footprint's bounding box as fractions of the square, and the smallest
square window (centred on the box) that holds it with a margin.
"""

from std.sys import argv
from max.gpu.host import DeviceContext

from noeira.physics3d.fields import Data
from noeira.physics3d.kinematics.forward_kinematics import forward_kinematics
from noeira.physics3d.parser.runtime_load import parse_model_runtime
from noeira.tasks.family import scene_path
from noeira.tasks.family_config import So101TowerConfig
from noeira.tasks.placement.so101_tower import So101TowerPlacement
from noeira.tasks.posed_reset import posed_qpos
from noeira.tasks.so101_tower_rig import (
    RIG_DT, TOWER_MD, make_tower_model, make_tower_renderer, tower_cameras,
)
from noeira.tasks.so101_tower_xml import So101TowerModel
from noeira.tasks.spec import load_family

comptime R = 128
comptime DIFF = 0.04


def main() raises:
    var a = argv()
    var task = String(a[1]) if len(a) > 1 else String("so101_tower_cube_in_bowl")
    var n = Int(String(a[2])) if len(a) > 2 else 300
    var ctx = DeviceContext()
    var f = load_family("noeira/tasks/families/so101_tower.family")
    var fmd = parse_model_runtime(scene_path(f))
    var cams = tower_cameras(fmd)
    var rm = make_tower_model(ctx)
    var rd = Data[RIG_DT, TOWER_MD, 1]()
    var r = make_tower_renderer[1, R, R, 1](ctx, fmd, rm)
    r.cam = cams[0]  # overhead

    # the empty scene: both free props dropped far below the desk
    var q = posed_qpos[So101TowerPlacement](
        task, String("so101_tower"), So101TowerConfig.SLOT_RADIUS, seed=0
    )
    for j in range(So101TowerPlacement.N_FREE):
        q[So101TowerPlacement.free_qadr(j) + 2] -= 20.0
    for k in range(So101TowerModel.NQ):
        rd.qpos.data[k] = Scalar[RIG_DT](q[k])
    forward_kinematics["cpu", RIG_DT, TOWER_MD, 1](rd, rm)
    var base = List[Scalar[RIG_DT]]()
    var dd = List[Scalar[RIG_DT]]()
    var ss = List[Scalar[RIG_DT]]()
    r.render_cpu(rd, rm, base, dd, ss)

    var occ = List[Int](length=R * R, fill=0)
    for s in range(1, n + 1):
        var qs = posed_qpos[So101TowerPlacement](
            task, String("so101_tower"), So101TowerConfig.SLOT_RADIUS,
            seed=UInt64(s),
        )
        for k in range(So101TowerModel.NQ):
            rd.qpos.data[k] = Scalar[RIG_DT](qs[k])
        forward_kinematics["cpu", RIG_DT, TOWER_MD, 1](rd, rm)
        var rgb = List[Scalar[RIG_DT]]()
        r.render_cpu(rd, rm, rgb, dd, ss)
        for p in range(R * R):
            var d = 0.0
            for c in range(3):
                d += abs(Float64(rgb[p * 3 + c]) - Float64(base[p * 3 + c]))
            if d > DIFF:
                occ[p] += 1
    var x0 = R
    var x1 = -1
    var y0 = R
    var y1 = -1
    var total = 0
    for y in range(R):
        for x in range(R):
            if occ[y * R + x] > 0:
                total += 1
                x0 = min(x0, x)
                x1 = max(x1, x)
                y0 = min(y0, y)
                y1 = max(y1, y)
    print("task", task, "|", n, "placements |", R, "x", R, "overhead render")
    print("  prop footprint:", total, "pixels ever covered")
    print("  bbox x", x0, "..", x1, " y", y0, "..", y1, "  (fractions x",
          Float64(x0) / R, "..", Float64(x1 + 1) / R, " y", Float64(y0) / R,
          "..", Float64(y1 + 1) / R, ")")
    var w = max(x1 - x0 + 1, y1 - y0 + 1)
    var cx = Float64(x0 + x1 + 1) / 2.0 / R
    var cy = Float64(y0 + y1 + 1) / 2.0 / R
    print("  square holding it:", w, "px =", Float64(w) / R, "of the side, centred at (",
          cx, ",", cy, ")")
    # the coarse occupancy map, 16 x 16, for the eye
    print("  occupancy (16x16, '#' covered in >= 1 placement):")
    for by in range(16):
        var line = String("    ")
        for bx in range(16):
            var hit = False
            for y in range(by * R // 16, (by + 1) * R // 16):
                for x in range(bx * R // 16, (bx + 1) * R // 16):
                    if occ[y * R + x] > 0:
                        hit = True
            line += "#" if hit else "."
        print(line)
