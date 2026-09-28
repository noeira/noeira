"""Render-time domain randomization on the so101_tower rig — plan Phase 2 gates.

  G2a  `off` is a restore: after any draw the tables equal the build's and the
       frame equals the un-randomized render, bit for bit.
  G2b  one fixed state, many draws: with the camera knobs at zero `seg` never
       changes (geometry, groups and the mirror are untouched), while each knob
       family ALONE (lights / colours / camera) moves the frame mean by more
       than a floor. The camera family is the positive control for `seg`: it
       MUST move it, or the seg check is blind.
  G2c  a draw is a function of (seed, index): apply(5), apply(9), apply(5)
       renders the first frame again, bit for bit. Different seeds differ.

Also held on every draw: the reflectance column equals the build's, the light
count stays within MAX_VIS_LIGHTS, and each surface group matched geoms.

Device tracer at 160x120, one lane, one sample — cheap on purpose.
"""

from std.math import sqrt
from std.sys import exit, has_accelerator
from max.gpu.host import DeviceContext

from noeira.envs.phyics3d_env import Phyics3dEnv
from noeira.math3d import Vec3 as Vec3Generic
from noeira.physics3d.fields import Data, Model
from noeira.physics3d.kinematics.forward_kinematics import forward_kinematics
from noeira.physics3d.parser.runtime_load import parse_model_runtime
from noeira.physics3d.raytrace import BatchedCameraRenderer, RGB_CHANNELS
from noeira.physics3d.raytrace.visual import build_visual_model
from noeira.physics3d.raytrace.visual_records import (
    APP_IDX_REFLECT,
    MAX_VIS_LIGHTS,
    VIS_GEOM_APPEARANCE,
)
from noeira.physics3d.raytrace.randomize import (
    DomainRandConfig,
    VisualRandomizer,
    geom_labels,
    so101_tower_surface_groups,
)
from noeira.tasks.family import scene_path
from noeira.tasks.family_config import So101TowerConfig
from noeira.tasks.so101_tower_xml import So101TowerModel
from noeira.tasks.spec import load_family

comptime DT = DType.float32
comptime Vec3 = Vec3Generic[DT]
comptime MD = Phyics3dEnv[So101TowerModel, So101TowerConfig, DT, False].MD
comptime NQ = So101TowerModel.NQ
comptime W = 160
comptime H = 120
comptime NPIX = W * H
comptime R = BatchedCameraRenderer[DT, MD, 1, W, H, False, True, 1]
comptime MASK: Int = (1 << 0) | (1 << 2)
comptime N_DRAWS = 16


def check(mut fails: Int, name: String, ok: Bool, detail: String):
    if ok:
        print("  PASS  " + name + "  " + detail)
    else:
        fails += 1
        print("  FAIL  " + name + "  " + detail)


struct Frames(Movable):
    var rgb: List[Float32]
    var seg: List[Float32]

    def __init__(out self):
        self.rgb = List[Float32]()
        self.seg = List[Float32]()


def render_both(
    ctx: DeviceContext, mut r: R, mut d: Data[DT, MD, 1], mut m: Model[DT, MD],
    cams: List[Int],
) raises -> Frames:
    var f = Frames()
    var h_rgb = ctx.enqueue_create_host_buffer[DT](NPIX * RGB_CHANNELS)
    var h_seg = ctx.enqueue_create_host_buffer[DT](NPIX)
    for c in cams:
        r.render(ctx, d, m, c)
        ctx.enqueue_copy(h_rgb, r.rgb)
        ctx.enqueue_copy(h_seg, r.seg)
        ctx.synchronize()
        for i in range(NPIX * RGB_CHANNELS):
            f.rgb.append(Float32(h_rgb[i]))
        for i in range(NPIX):
            f.seg.append(Float32(h_seg[i]))
    return f^


def maxdiff(ref a: List[Float32], ref b: List[Float32]) -> Float64:
    var m = 0.0
    for i in range(len(a)):
        var dd = abs(Float64(a[i]) - Float64(b[i]))
        if dd > m:
            m = dd
    return m


def mean(ref a: List[Float32]) -> Float64:
    var s = 0.0
    for v in a:
        s += Float64(v)
    return s / Float64(len(a))


def std_of(ref xs: List[Float64]) -> Float64:
    var mu = 0.0
    for x in xs:
        mu += x
    mu /= Float64(len(xs))
    var ss = 0.0
    for x in xs:
        ss += (x - mu) * (x - mu)
    return sqrt(ss / Float64(len(xs)))


def only_lights(seed: UInt64) -> DomainRandConfig:
    var c = DomainRandConfig.full(seed)
    c.strength = 0.0
    c.cam_pos_m = 0.0
    c.cam_rot_deg = 0.0
    c.cam_fovy_deg = 0.0
    c.background = 0.0
    return c


def only_colours(seed: UInt64) -> DomainRandConfig:
    var c = DomainRandConfig.full(seed)
    c.light_tilt_deg = 0.0
    c.light_scale = 0.0
    c.light_tint = 0.0
    c.light_ambient = 0.0
    c.headlight_scale = 0.0
    c.headlight_off_prob = 0.0
    c.max_extra_lights = 0
    c.cam_pos_m = 0.0
    c.cam_rot_deg = 0.0
    c.cam_fovy_deg = 0.0
    c.background = 0.0
    return c


def only_camera(seed: UInt64) -> DomainRandConfig:
    var c = only_colours(seed)
    c.strength = 0.0
    c.cam_pos_m = 0.010
    c.cam_rot_deg = 2.0
    c.cam_fovy_deg = 3.0
    return c


def main() raises:
    comptime if not has_accelerator():
        print("=== SKIPPED (no accelerator; this is not a pass) ===")
        return
    var fails = 0
    var ctx = DeviceContext()
    print("render-time domain randomization gate (so101_tower)")
    print("  device: " + String(ctx.name()))

    var fam = load_family(String("noeira/tasks/families/so101_tower.family"))
    var fmd = parse_model_runtime(scene_path(fam))
    var m = Model[DT, MD]()
    So101TowerModel.init_fields[DT](ctx, m)
    m.upload_all(ctx)
    var d = Data[DT, MD, 1]()

    # A pose with both task objects on the mat, in view of both cameras.
    for k in range(NQ):
        d.qpos.data[k] = Scalar[DT](0)
    var arm = [0.0, -0.6, 0.9, 0.6, 0.0, 0.3]
    for j in range(6):
        d.qpos.data[j] = Scalar[DT](arm[j])
    var adr = 0
    for j in range(len(fmd.joints)):
        var name = String(fmd.joint_names[j])
        if fmd.joints[j].nq == 7 and (name == "bowl_free" or name == "brick_free"):
            var bowl = name == "bowl_free"
            d.qpos.data[adr] = Scalar[DT](0.22 if bowl else 0.28)
            d.qpos.data[adr + 1] = Scalar[DT](0.08 if bowl else -0.08)
            d.qpos.data[adr + 2] = Scalar[DT](0.002 if bowl else 0.0116)
            d.qpos.data[adr + 3] = Scalar[DT](1)
        adr += fmd.joints[j].nq
    forward_kinematics["cpu", DT, MD, 1](d, m)
    d.upload_all(ctx)
    ctx.synchronize()

    var cams = List[Int]()
    for i in range(len(fmd.camera_names)):
        var n = String(fmd.camera_names[i])
        if n.endswith("overhead_cam") or n.endswith("wrist_cam"):
            cams.append(i)
    check(fails, "both rig cameras found", len(cams) == 2, String(len(cams)))

    var r = R(ctx, m, cams[0])
    r.set_visual(ctx, build_visual_model[DT, MD](fmd, m, group_mask=MASK))
    var bg0 = Vec3(0.82, 0.86, 0.90)
    r.background = bg0
    var target = (0.32, 0.0, 0.0)
    var labels = geom_labels(fmd)

    var base_lights = r.vis.lights.data.copy()
    var base_mats = r.vis.materials.data.copy()
    var base_app = r.vis.appearance.data.copy()
    var base_cams = m.cameras.data.copy()
    var base = render_both(ctx, r, d, m, cams)
    var nbg = 0
    for s in base.seg:
        if s < 0.0:
            nbg += 1
    check(fails, "the base frame is not all background",
          nbg < len(base.seg), String(nbg) + " of " + String(len(base.seg)))

    # Every randomizer is built HERE, from the pristine tables: its base is
    # what it restores to, so one built after a draw would restore the draw.
    var full = VisualRandomizer[DT](
        DomainRandConfig.full(1234), so101_tower_surface_groups(), r.vis, m,
        labels, cams.copy(), bg0, target,
    )
    var off = VisualRandomizer[DT](
        DomainRandConfig.off(), so101_tower_surface_groups(), r.vis, m,
        labels, cams.copy(), bg0, target,
    )
    var other = VisualRandomizer[DT](
        DomainRandConfig.full(99), so101_tower_surface_groups(), r.vis, m,
        labels, cams.copy(), bg0, target,
    )
    var fams = List[VisualRandomizer[DT]]()
    fams.append(VisualRandomizer[DT](
        only_lights(77), so101_tower_surface_groups(), r.vis, m,
        labels, cams.copy(), bg0, target))
    fams.append(VisualRandomizer[DT](
        only_colours(77), so101_tower_surface_groups(), r.vis, m,
        labels, cams.copy(), bg0, target))
    fams.append(VisualRandomizer[DT](
        only_camera(77), so101_tower_surface_groups(), r.vis, m,
        labels, cams.copy(), bg0, target))
    var counts = String("")
    for gi in range(len(full.groups)):
        counts += full.groups[gi].name + "=" + String(full.group_count(gi)) + " "
    print("  groups: " + counts)

    # ── G2a: off is a restore ────────────────────────────────────────────
    r.background = full.apply(3, r.vis, m)
    full.upload(ctx, r.vis, m)
    var moved = render_both(ctx, r, d, m, cams)
    r.background = off.apply(3, r.vis, m)
    off.upload(ctx, r.vis, m)
    var restored = render_both(ctx, r, d, m, cams)
    var tables_eq = (
        r.vis.lights.data == base_lights and r.vis.materials.data == base_mats
        and r.vis.appearance.data == base_app and m.cameras.data == base_cams
        and r.vis.nlight == off.base_nlight
    )
    check(fails, "G2a off restores every table byte for byte", tables_eq, "")
    var d_off = maxdiff(restored.rgb, base.rgb)
    check(fails, "G2a off renders the un-randomized frame", d_off == 0.0,
          "maxdiff " + String(d_off))
    var d_moved = maxdiff(moved.rgb, base.rgb)
    check(fails, "a full draw does change the frame (control)", d_moved > 0.05,
          "maxdiff " + String(d_moved))

    # ── G2c: reproducible from (seed, index) ─────────────────────────────
    r.background = full.apply(5, r.vis, m)
    full.upload(ctx, r.vis, m)
    var a5 = render_both(ctx, r, d, m, cams)
    r.background = full.apply(9, r.vis, m)
    full.upload(ctx, r.vis, m)
    var a9 = render_both(ctx, r, d, m, cams)
    r.background = full.apply(5, r.vis, m)
    full.upload(ctx, r.vis, m)
    var a5b = render_both(ctx, r, d, m, cams)
    check(fails, "G2c apply(5) after apply(9) == apply(5)",
          maxdiff(a5.rgb, a5b.rgb) == 0.0, "")
    check(fails, "G2c draws 5 and 9 differ", maxdiff(a5.rgb, a9.rgb) > 0.05, "")
    r.background = other.apply(5, r.vis, m)
    other.upload(ctx, r.vis, m)
    var b5 = render_both(ctx, r, d, m, cams)
    check(fails, "G2c another seed, same index, differs",
          maxdiff(a5.rgb, b5.rgb) > 0.05, "")

    # ── G2b: per knob family, over N_DRAWS draws of one state ───────────
    var names = ["lights", "colours", "camera"]
    for fam_i in range(3):
        ref rz = fams[fam_i]
        var means = List[Float64]()
        var seg_same = True
        var refl_same = True
        var lights_ok = True
        for k in range(N_DRAWS):
            r.background = rz.apply(k, r.vis, m)
            rz.upload(ctx, r.vis, m)
            if r.vis.nlight > MAX_VIS_LIGHTS:
                lights_ok = False
            for g in range(r.vis.ngeom):
                var o = g * VIS_GEOM_APPEARANCE + APP_IDX_REFLECT
                if r.vis.appearance.data[o] != base_app[o]:
                    refl_same = False
            var f = render_both(ctx, r, d, m, cams)
            means.append(mean(f.rgb))
            if maxdiff(f.seg, base.seg) != 0.0:
                seg_same = False
        var sd = std_of(means)
        var nm = String(names[fam_i])
        check(fails, "G2b " + nm + " alone moves the frame mean", sd > 0.005,
              "std over " + String(N_DRAWS) + " draws " + String(sd))
        check(fails, "G2b " + nm + ": reflectance column untouched", refl_same, "")
        check(fails, "G2b " + nm + ": light count <= MAX_VIS_LIGHTS", lights_ok, "")
        if fam_i < 2:
            check(fails, "G2b " + nm + ": seg identical on every draw", seg_same, "")
        else:
            check(fails, "G2b camera MOVES seg (the seg check's control)",
                  not seg_same, "")

    print("")
    if fails == 0:
        print("ALL PASS")
    else:
        print(String(fails) + " FAILED")
        exit(1)
