"""The tower rig's per-camera DR ranges (`so101_tower_rig.scale_tower_camera_dr`).

    pixi run mojo run -I . tests/tasks/test_so101_tower_camera_dr.mojo

Host only: two randomizers over the composed tower scene, the `full` preset
alone and the preset with the rig's camera scales, applied at the same draws:

1. the scaled camera offsets are EXACTLY the preset's times the scale —
   position and fovy per camera, rotation angle per camera — because every
   draw of the stream is taken whatever the scale;
2. nothing else a draw writes (lights, colours, background) moves: a scale
   shifts no other draw of the stream;
3. over 200 draws the overhead camera stays inside +-4 mm / +-0.75 deg and the
   wrist camera inside +-4 mm / +-3.5 deg (the arm's sag), both fovy inside
   +-0.75 deg;
4. a camera that is not the rig's, and a negative scale, are refused.
"""

from std.sys import exit
from std.math import abs, asin, pi, sqrt

from noeira.physics3d.fields import Model, DynDims
from noeira.physics3d.gpu.constants import (
    MODEL_CAM_SIZE, CAM_IDX_POS_X, CAM_IDX_QUAT_X, CAM_IDX_FOVY,
)
from noeira.physics3d.parser.runtime_load import (
    parse_model_runtime, dims_from_flat, build_model_runtime,
)
from noeira.physics3d.raytrace.visual import build_visual_model
from noeira.physics3d.raytrace.randomize import (
    DomainRandConfig, VisualRandomizer, geom_labels, so101_tower_surface_groups,
)
from noeira.tasks.family import scene_path
from noeira.tasks.spec import load_family
from noeira.tasks.so101_tower_rig import (
    RIG_VISUAL_GROUP_MASK, RIG_DR_TARGET, RIG_DR_OVERHEAD_CAM_SCALE,
    RIG_DR_WRIST_CAM_SCALE, rig_background, scale_tower_camera_dr,
    tower_cameras,
)
from noeira.tasks.so101_tower_xml import SO101_TOWER_NMESH_VERTS

comptime DT = DType.float32


def check(mut fails: Int, name: String, ok: Bool, detail: String = ""):
    if ok:
        print("  PASS  " + name + "  " + detail)
    else:
        fails += 1
        print("  FAIL  " + name + "  " + detail)


def _angle_deg(ref a: List[Scalar[DT]], ref b: List[Scalar[DT]], cb: Int) -> Float64:
    """The rotation between two (x, y, z, w) rows, from the VECTOR part of
    conj(b) * a — sin(angle/2) — not acos of the dot, which float32 cannot
    resolve below ~0.05 deg."""
    var ax = Float64(a[cb + CAM_IDX_QUAT_X])
    var ay = Float64(a[cb + CAM_IDX_QUAT_X + 1])
    var az = Float64(a[cb + CAM_IDX_QUAT_X + 2])
    var aw = Float64(a[cb + CAM_IDX_QUAT_X + 3])
    var bx = -Float64(b[cb + CAM_IDX_QUAT_X])
    var by = -Float64(b[cb + CAM_IDX_QUAT_X + 1])
    var bz = -Float64(b[cb + CAM_IDX_QUAT_X + 2])
    var bw = Float64(b[cb + CAM_IDX_QUAT_X + 3])
    var x = bw * ax + bx * aw + by * az - bz * ay
    var y = bw * ay - bx * az + by * aw + bz * ax
    var z = bw * az + bx * ay - by * ax + bz * aw
    return 2.0 * asin(min(1.0, sqrt(x * x + y * y + z * z))) * 180.0 / pi


def main() raises:
    var fails = 0
    print("so101_tower rig: per-camera DR ranges")
    var f = load_family(String("noeira/tasks/families/so101_tower.family"))
    var fmd = parse_model_runtime(scene_path(f))
    var dims = dims_from_flat(fmd, max_contacts=64, nmesh_verts=SO101_TOWER_NMESH_VERTS)
    var m1 = Model[DT, DynDims](dims)
    build_model_runtime[DT](fmd, dims, m1)
    var m2 = Model[DT, DynDims](dims)
    build_model_runtime[DT](fmd, dims, m2)
    var v1 = build_visual_model[DT, DynDims](fmd, m1, group_mask=RIG_VISUAL_GROUP_MASK)
    var v2 = build_visual_model[DT, DynDims](fmd, m2, group_mask=RIG_VISUAL_GROUP_MASK)
    var labels = geom_labels(fmd)
    var cams = tower_cameras(fmd)  # [overhead, wrist]
    var base = m1.cameras.data.copy()
    var cfg = DomainRandConfig.full(1234)
    var d1 = VisualRandomizer[DT](
        cfg, so101_tower_surface_groups(), v1, m1, labels, cams.copy(),
        rig_background(), RIG_DR_TARGET,
    )
    var d2 = VisualRandomizer[DT](
        cfg, so101_tower_surface_groups(), v2, m2, labels, cams.copy(),
        rig_background(), RIG_DR_TARGET,
    )
    scale_tower_camera_dr(d2, fmd)
    var so = RIG_DR_OVERHEAD_CAM_SCALE
    var sw = RIG_DR_WRIST_CAM_SCALE
    var sp: List[Float64] = [so[0], sw[0]]
    var sr: List[Float64] = [so[1], sw[1]]
    var sf: List[Float64] = [so[2], sw[2]]

    var exact = True
    var same_rest = True
    var worst = ""
    var max_pos: List[Float64] = [0.0, 0.0]
    var max_rot: List[Float64] = [0.0, 0.0]
    var max_fov: List[Float64] = [0.0, 0.0]
    for draw in range(200):
        var bg1 = d1.apply(draw, v1, m1)
        var bg2 = d2.apply(draw, v2, m2)
        if bg1.x != bg2.x or bg1.y != bg2.y or bg1.z != bg2.z:
            same_rest = False
        for i in range(len(v1.lights.data)):
            if v1.lights.data[i] != v2.lights.data[i]:
                same_rest = False
        for i in range(len(v1.appearance.data)):
            if v1.appearance.data[i] != v2.appearance.data[i]:
                same_rest = False
        for k in range(2):
            var cb = cams[k] * MODEL_CAM_SIZE
            for a in range(3):
                var o1 = Float64(m1.cameras.data[cb + CAM_IDX_POS_X + a]) - Float64(base[cb + CAM_IDX_POS_X + a])
                var o2 = Float64(m2.cameras.data[cb + CAM_IDX_POS_X + a]) - Float64(base[cb + CAM_IDX_POS_X + a])
                if abs(o2 - sp[k] * o1) > 2e-7:
                    exact = False
                    worst = "cam " + String(k) + " pos " + String(o1) + " -> " + String(o2)
                max_pos[k] = max(max_pos[k], abs(o2))
            var f1 = Float64(m1.cameras.data[cb + CAM_IDX_FOVY]) - Float64(base[cb + CAM_IDX_FOVY])
            var f2 = Float64(m2.cameras.data[cb + CAM_IDX_FOVY]) - Float64(base[cb + CAM_IDX_FOVY])
            if abs(f2 - sf[k] * f1) > 2e-5:
                exact = False
                worst = "cam " + String(k) + " fovy " + String(f1) + " -> " + String(f2)
            max_fov[k] = max(max_fov[k], abs(f2))
            var a1 = _angle_deg(m1.cameras.data, base, cb)
            var a2 = _angle_deg(m2.cameras.data, base, cb)
            if abs(a2 - sr[k] * a1) > 0.005:
                exact = False
                worst = "cam " + String(k) + " rot " + String(a1) + " -> " + String(a2)
            max_rot[k] = max(max_rot[k], a2)

    check(fails, "1 the scaled offsets are the preset's times the scale (pos, rot, fovy; both cameras)", exact, worst)
    check(fails, "2 lights, colours and background are the preset's own draws", same_rest)
    check(fails, "3a overhead within +-4 mm / 0.75 deg / fovy 0.75 over 200 draws",
          max_pos[0] <= 0.004 + 1e-6 and max_rot[0] <= 0.75 + 0.02 and max_fov[0] <= 0.75 + 1e-4,
          String(max_pos[0] * 1000.0) + " mm, " + String(max_rot[0]) + " deg, fovy " + String(max_fov[0]))
    check(fails, "3b wrist within +-4 mm / 3.5 deg / fovy 0.75 over 200 draws",
          max_pos[1] <= 0.004 + 1e-6 and max_rot[1] <= 3.5 + 0.02 and max_fov[1] <= 0.75 + 1e-4,
          String(max_pos[1] * 1000.0) + " mm, " + String(max_rot[1]) + " deg, fovy " + String(max_fov[1]))
    check(fails, "3c the draws reach most of the range (not a zeroed knob)",
          max_pos[0] > 0.003 and max_rot[1] > 3.0 and max_fov[1] > 0.5)

    var refused_neg = False
    try:
        d2.scale_camera(0, -1.0, 1.0, 1.0)
    except:
        refused_neg = True
    var refused_idx = False
    try:
        d2.scale_camera(2, 1.0, 1.0, 1.0)
    except:
        refused_idx = True
    check(fails, "4 a negative scale and an unlisted camera are refused", refused_neg and refused_idx)

    print("")
    if fails == 0:
        print("ALL PASS")
    else:
        print(String(fails) + " FAILED")
        exit(1)
