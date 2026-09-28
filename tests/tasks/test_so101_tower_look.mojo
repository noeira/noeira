"""`so101_tower_rig.apply_tower_look` — the calibrated and the legacy look.

    pixi run mojo run -I . tests/tasks/test_so101_tower_look.mojo

The tracer's tables for the composed tower scene, built on the host:

1. `calibrated` leaves them exactly as built (it IS the composed scene);
2. `legacy` puts back the pre-2026-09-24 look: the headlight .45 ambient /
   .3 diffuse, the floor's sun .7, the desk .93, every calibrated-white arm
   geom .92, the camera mount's grey .92, the props' filament colours — and
   touches no other word
   (geometry, materials, the other geoms' colours);
3. an unknown look is refused;
4. the desk asset's BACKDROP (group 4: floor, desk skin, +y floor strip, -y
   clutter extension, wall), the gripper's ArUco MARKER (group 5: the black
   square + 4 white runs) and the room-grey background belong to the
   calibrated look only.

The end-to-end check is a render: `tower_demo_rerender --look legacy` of a
demo reproduces a store rendered before the calibration byte for byte
(236/236 rows of `p2_human_clean` episode 0, 2026-09-24).
"""

from std.sys import exit
from std.math import abs

from noeira.physics3d.fields import Model, DynDims
from noeira.physics3d.parser.runtime_load import (
    parse_model_runtime, dims_from_flat, build_model_runtime,
)
from noeira.physics3d.raytrace.visual import build_visual_model
from noeira.physics3d.raytrace.visual_records import (
    VIS_GEOM_APPEARANCE, VIS_LIGHT_WORDS, APP_IDX_R, APP_IDX_G, APP_IDX_B,
    LIGHT_IDX_AMBIENT_R, LIGHT_IDX_DIFFUSE_R,
)
from noeira.physics3d.raytrace.randomize import geom_labels
from noeira.tasks.family import scene_path
from noeira.tasks.spec import load_family
from noeira.tasks.so101_tower_rig import (
    apply_tower_look, RIG_LOOK_CALIBRATED, RIG_LOOK_LEGACY,
    RIG_VISUAL_GROUP_MASK, rig_visual_group_mask, rig_background,
)
from noeira.tasks.so101_tower_xml import SO101_TOWER_NMESH_VERTS

comptime DT = DType.float32


def check(mut fails: Int, name: String, ok: Bool, detail: String = ""):
    if ok:
        print("  PASS  " + name + "  " + detail)
    else:
        fails += 1
        print("  FAIL  " + name + "  " + detail)


def near(x: Scalar[DT], y: Float64) -> Bool:
    return abs(Float64(x) - y) < 1e-6


def main() raises:
    var fails = 0
    print("so101_tower rig: the calibrated and the legacy look")
    var f = load_family(String("noeira/tasks/families/so101_tower.family"))
    var fmd = parse_model_runtime(scene_path(f))
    var dims = dims_from_flat(fmd, max_contacts=64, nmesh_verts=SO101_TOWER_NMESH_VERTS)
    var m = Model[DT, DynDims](dims)
    build_model_runtime[DT](fmd, dims, m)
    var labels = geom_labels(fmd)

    # 1. calibrated: untouched
    var vc = build_visual_model[DT, DynDims](fmd, m, group_mask=RIG_VISUAL_GROUP_MASK)
    var app0 = vc.appearance.data.copy()
    var lig0 = vc.lights.data.copy()
    var mat0 = vc.materials.data.copy()
    apply_tower_look(vc, fmd, String(RIG_LOOK_CALIBRATED))
    var same = True
    for i in range(len(app0)):
        if vc.appearance.data[i] != app0[i]:
            same = False
    for i in range(len(lig0)):
        if vc.lights.data[i] != lig0[i]:
            same = False
    check(fails, "1 calibrated is the composed scene, untouched", same)
    check(fails, "1b the composed lights are the calibration's",
          near(lig0[LIGHT_IDX_AMBIENT_R], 0.6) and near(lig0[LIGHT_IDX_DIFFUSE_R], 0.0)
          and near(lig0[VIS_LIGHT_WORDS + LIGHT_IDX_DIFFUSE_R], 0.2),
          String(lig0[LIGHT_IDX_AMBIENT_R]) + " / " + String(lig0[LIGHT_IDX_DIFFUSE_R])
          + " / sun " + String(lig0[VIS_LIGHT_WORDS + LIGHT_IDX_DIFFUSE_R]))

    # 2. legacy
    var vl = build_visual_model[DT, DynDims](fmd, m, group_mask=RIG_VISUAL_GROUP_MASK)
    apply_tower_look(vl, fmd, String(RIG_LOOK_LEGACY))
    check(fails, "2a legacy lights: headlight .45 / .3, sun .7",
          near(vl.lights.data[LIGHT_IDX_AMBIENT_R + 2], 0.45)
          and near(vl.lights.data[LIGHT_IDX_DIFFUSE_R + 1], 0.3)
          and near(vl.lights.data[VIS_LIGHT_WORDS + LIGHT_IDX_DIFFUSE_R + 2], 0.7))
    var desk_ok = True
    var bowl_ok = True
    var brick_ok = True
    var white = 0
    var mount = 0
    var others_same = True
    for k in range(vl.ngeom):
        var lab = labels[vl.src_geom[k]]
        var o = k * VIS_GEOM_APPEARANCE
        var r = vl.appearance.data[o + APP_IDX_R]
        var g = vl.appearance.data[o + APP_IDX_G]
        var b = vl.appearance.data[o + APP_IDX_B]
        if lab.startswith("desk_"):
            desk_ok = desk_ok and near(r, 0.93) and near(b, 0.91)
        elif lab.startswith("bowl_"):
            bowl_ok = bowl_ok and near(r, 0.996) and near(g, 0.776) and near(b, 0.0)
        elif lab.startswith("brick_"):
            brick_ok = brick_ok and near(r, 0.0) and near(g, 0.471) and near(b, 0.749)
        elif lab.startswith("robot_") and near(app0[o + APP_IDX_R], 0.78) and near(app0[o + APP_IDX_B], 0.78):
            if near(r, 0.92) and near(g, 0.92) and near(b, 0.92):
                mount += 1
        elif lab.startswith("robot_") and near(app0[o + APP_IDX_R], 0.78):
            if near(r, 0.92) and near(b, 0.90):
                white += 1
        else:
            for w in range(3):
                if vl.appearance.data[o + w] != app0[o + w]:
                    others_same = False
    check(fails, "2b desk .93, bowl and brick back to the filament", desk_ok and bowl_ok and brick_ok)
    check(fails, "2c every calibrated-white arm geom back to .92", white > 0, String(white) + " geoms")
    check(fails, "2c' the camera mount (calibrated .78 neutral) back to .92 grey", mount == 1, String(mount) + " geoms")
    check(fails, "2d no other geom's colour moves", others_same)
    var mats_same = True
    for i in range(len(mat0)):
        if vl.materials.data[i] != mat0[i]:
            mats_same = False
    check(fails, "2e the material rows are not touched (the shader colours from the appearance row)", mats_same)

    # 4. the backdrop (group 4) and the marker (group 5) are the calibrated look's only
    var vcal = build_visual_model[DT, DynDims](
        fmd, m, group_mask=rig_visual_group_mask(String(RIG_LOOK_CALIBRATED))
    )
    var vleg = build_visual_model[DT, DynDims](
        fmd, m, group_mask=rig_visual_group_mask(String(RIG_LOOK_LEGACY))
    )
    var backdrop = 0
    var marker = 0
    for k in range(vcal.ngeom):
        var lab = labels[vcal.src_geom[k]]
        if lab.find("floor_cover") >= 0 or lab.find("desk_skin") >= 0 or lab.find("floor_strip") >= 0 or lab.find("desk_ext") >= 0 or lab.find("desk_wall") >= 0:
            backdrop += 1
        if lab.find("/robot_aruco_") >= 0:
            marker += 1
    check(fails, "4 the calibrated look draws the 5 backdrop + 5 marker geoms, legacy none",
          backdrop == 5 and marker == 5 and vcal.ngeom == vleg.ngeom + 10 and vleg.ngeom == vl.ngeom,
          String(backdrop) + " + " + String(marker) + "; " + String(vcal.ngeom) + " vs "
          + String(vleg.ngeom) + " visual geoms")
    var bg_c = rig_background(String(RIG_LOOK_CALIBRATED))
    var bg_l = rig_background(String(RIG_LOOK_LEGACY))
    check(fails, "4b the background: calibrated room grey, legacy blue-grey",
          near(bg_c.x, 0.59) and near(bg_l.z, 0.90))

    # 3. unknown look
    var refused = False
    try:
        var vx = build_visual_model[DT, DynDims](fmd, m, group_mask=RIG_VISUAL_GROUP_MASK)
        apply_tower_look(vx, fmd, String("bright"))
    except:
        refused = True
    check(fails, "3 an unknown look is refused", refused)

    print("")
    if fails == 0:
        print("ALL PASS")
    else:
        print(String(fails) + " FAILED")
        exit(1)
