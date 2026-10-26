"""The batched camera tracer on the GPU against the same code on the CPU.

    pixi run mojo run -I . tests/physics3d/test_camera_render_gpu_vs_cpu.mojo

⚠⚠ THIS IS `test_ray_model_gpu_vs_cpu`'S ARGUMENT ONE LAYER UP, AND IT IS NOT A
FORMALITY. `raytrace/` is ONE implementation over `LayoutTensor` so a kernel and
the host run the same code — and "the same code" is what Metal has silently
miscompiled four times in this engine, always the same way: a per-thread array
indexed by a RUNTIME value reads back wrong, with no crash and no diagnostic
(`87960e10`). The camera adds a whole new opportunity for one — a frame is
three axis vectors and a position, which is exactly the thing somebody would
reach for a nine-element array for. This gate is what says nobody did.

⚠ float32 ON BOTH LEGS. Metal rejects `double`, so the CPU control is built at
float32 too; comparing a float32 GPU answer against a float64 CPU answer would
report the DTYPE as a GPU defect.

⚠ THREE CHANNELS ARE COMPARED, NOT ONE. `seg` is the sharpest of the three and
the reason is measured, not assumed: three of the five defects `ray_model` was
falsified against left the hit DISTANCE untouched and showed only as a
different geom. A colour-only comparison would have passed all three.

⚠ AND THE VACUITY GUARDS ARE LOAD-BEARING. "0 mismatching pixels" and "the
camera renders empty sky" are the same number. Every test below prints what it
COMPARED beside what DIFFERED, and asserts on the former.
"""

from std.math import abs, sqrt
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true, assert_almost_equal, TestSuite
from noeira.core.tiles import RowMajor

from noeira.math3d import Vec3 as Vec3Generic
from noeira.physics3d.fields import Data, Model, Dims, init_hfield_data
from noeira.physics3d.parser.full_parser import parse_xml_full
from noeira.physics3d.parser.fields_build import build_model_fields_from_flat
from noeira.physics3d.kinematics.forward_kinematics import forward_kinematics
from noeira.physics3d.dynamics.subtree_com import compute_subtree_com
from noeira.physics3d.gpu.constants import (
    MODEL_CAM_SIZE,
    CAM_IDX_ACTIVE,
    CAM_IDX_BODY,
    CAM_IDX_MODE,
    CAM_IDX_REF_SET,
)
from noeira.physics3d.raytrace import (
    BatchedCameraRenderer,
    RGB_CHANNELS,
    camera_world_frame,
    init_camera_reference,
)
from noeira.physics3d.raytrace.visual import build_visual_model, SiteCondition

comptime GT = DType.float32
comptime BATCH = 2
comptime W = 32
comptime H = 24
comptime NPIX = W * H

comptime NQ = 7
comptime NV = 6
comptime NBODY = 2
comptime NJOINT = 1
comptime NGEOM = 9
comptime NHF = 64
comptime NTRI = 64

comptime GD = Dims[
    nq=NQ,
    nv=NV,
    nbody=NBODY,
    njoint=NJOINT,
    ngeom=NGEOM,
    nsite=0,
    max_contacts=16,
    nmesh_verts=256,
    nmesh_tri=NTRI,
    nhfield_data=NHF,
]

comptime Renderer = BatchedCameraRenderer[GT, GD, BATCH, W, H]
comptime Renderer4 = BatchedCameraRenderer[GT, GD, BATCH, W, H, True, True, 4]
"""4x MSAA — LIBERO's `offsamples`. A separate kernel instantiation: the
sample loop and its sub-pixel rays only exist in this one."""

# ⚠ EVERY GEOM TYPE IS IN FRAME ON PURPOSE. The tracer's inner call is a
# DISPATCH, and a dispatch is only tested by the branch it gets wrong; a camera
# that only ever sees spheres proves nothing about the `ray_hfield` call three
# branches down. The colours are all different so a segmentation defect and a
# colour defect cannot alias.
#
# ⚠ THE CAMERAS ARE THREE MODES, NOT THREE ANGLES. `world_cam` is the case that
# was ALWAYS right (a `<worldbody>` camera composes with the identity, which is
# why the missing `mj_camlight` composition hid across the whole dm_control
# suite); `body_cam` is the case that separates them; `com_cam` is the one that
# reads the reference pose `init_camera_reference` has to have filled.
comptime SCENE = String(
    """
<mujoco model="camera tracer gate">
  <asset>
    <hfield name="terrain" file="tests/physics3d/assets/hf_8x8.bin" size="0.4 0.4 0.15 0.05"/>
    <mesh name="notch" file="tests/physics3d/assets/notch.stl"/>
  </asset>
  <worldbody>
    <camera name="world_cam" pos="0 -2.2 1.1" xyaxes="1 0 0 0 0.45 0.9" fovy="70"/>
    <geom name="floor" type="plane" size="0 0 0.05" pos="0 0 -0.6" rgba="0.35 0.4 0.32 1"/>
    <geom name="terrain" type="hfield" hfield="terrain" pos="0.55 0.35 -0.55" rgba="0.55 0.42 0.28 1"/>
    <geom name="target_sphere" type="sphere" size="0.16" pos="0 0 0" rgba="0.85 0.2 0.2 1"/>
    <geom name="a_box" type="box" size="0.10 0.08 0.09" pos="-0.45 0.20 0.05" euler="20 -35 15" rgba="0.2 0.4 0.85 1"/>
    <geom name="a_capsule" type="capsule" size="0.06 0.12" pos="0.42 -0.30 0.10" euler="55 10 -20" rgba="0.9 0.75 0.15 1"/>
    <geom name="a_cylinder" type="cylinder" size="0.07 0.10" pos="-0.38 -0.35 0.12" euler="-25 40 5" rgba="0.15 0.75 0.45 1"/>
    <geom name="an_ellipsoid" type="ellipsoid" size="0.12 0.06 0.09" pos="0.28 0.42 -0.05" euler="10 25 -40" rgba="0.7 0.3 0.8 1"/>
    <geom name="a_mesh" type="mesh" mesh="notch" pos="-0.10 0.52 0.10" euler="15 -25 40" rgba="0.95 0.5 0.1 1"/>
    <body name="rider" pos="0 0 0.75">
      <freejoint/>
      <camera name="body_cam" pos="0 -0.35 0.12" xyaxes="1 0 0 0 0.5 0.9"/>
      <camera name="com_cam" mode="trackcom" pos="0 -2.0 1.4" xyaxes="1 0 0 0 0.55 0.85"/>
      <geom name="rider_geom" type="sphere" size="0.09" rgba="0.1 0.85 0.9 1"/>
    </body>
  </worldbody>
</mujoco>
"""
)

comptime CAM_WORLD = 0
comptime CAM_BODY = 1
comptime CAM_COM = 2


def _textured_scene() raises -> String:
    """`SCENE` with textures on four geoms — derived from it, not a copy.

    ⚠ `visual_model_from_model` NEVER CARRIES A TEXTURE: the `Model` has no
    material table, so every test above renders flat colours and the whole
    texel path (mip chain, level of detail, trilinear) was never on the GPU.
    This is the scene that puts it there. Each mapping is a different branch:
    the infinite plane's object-linear UVs with a repeat, a CUBE texture on a
    box (face select, clamp) and on a cylinder (a face SWITCH inside one
    surface, the derivative guard), and a 2D texture on a sphere (the
    fallback texgen). The grazing floor reaches the far mip levels."""
    var xml = String(SCENE)
    xml = xml.replace(
        "<asset>",
        "<asset>\n"
        + '    <texture name="chk" type="2d" builtin="checker" width="64" height="64" rgb1=".1 .2 .3" rgb2=".9 .85 .7"/>\n'
        + '    <texture name="cub" type="cube" builtin="checker" width="32" height="32" rgb1=".8 .2 .2" rgb2=".2 .8 .3"/>\n'
        + '    <material name="floor_m" texture="chk" texrepeat="6 6"/>\n'
        + '    <material name="cube_m" texture="cub"/>\n'
        + '    <material name="ball_m" texture="chk" texrepeat="2 2"/>',
    )
    xml = xml.replace(
        'size="0 0 0.05" pos="0 0 -0.6" rgba="0.35 0.4 0.32 1"',
        'size="0 0 0.05" pos="0 0 -0.6" material="floor_m"',
    )
    xml = xml.replace(
        'euler="20 -35 15" rgba="0.2 0.4 0.85 1"',
        'euler="20 -35 15" material="cube_m"',
    )
    xml = xml.replace(
        'euler="-25 40 5" rgba="0.15 0.75 0.45 1"',
        'euler="-25 40 5" material="cube_m"',
    )
    xml = xml.replace(
        'pos="0 0 0" rgba="0.85 0.2 0.2 1"',
        'pos="0 0 0" material="ball_m"',
    )
    # A CONDITIONAL SITE on the rider, gated on its free joint's x — which the
    # two lanes set differently (0.0 and 0.30), so a kernel reading the wrong
    # lane's `qpos` shows it in the wrong picture.
    # A TRANSLUCENT geom, so the compositing loop (and its back-face step
    # out of a closed shape) runs in the kernel.
    xml = xml.replace(
        'euler="10 25 -40" rgba="0.7 0.3 0.8 1"',
        'euler="10 25 -40" rgba="0.7 0.3 0.8 0.4"',
    )
    xml = xml.replace("<freejoint/>", '<freejoint name="rider_free"/>')
    xml = xml.replace(
        '<geom name="rider_geom"',
        '<site name="beacon" type="box" size="0.07 0.07 0.07" pos="0 0 0.16" rgba="1 0.9 0 1"/>\n      <geom name="rider_geom"',
    )
    var c1 = xml.find('material="cube_m"')
    var c2 = xml.find('material="cube_m"', c1 + 1) if c1 >= 0 else -1
    if xml.find('material="floor_m"') < 0 or c2 < 0 \
            or xml.find('material="ball_m"') < 0 \
            or xml.find('name="beacon"') < 0 or xml.find('"rider_free"') < 0 \
            or xml.find('rgba="0.7 0.3 0.8 0.4"') < 0:
        raise Error(
            "_textured_scene: a replacement did not land — SCENE's geom lines"
            " changed and this test would render an untextured scene"
        )
    return xml^


def _build(
    mut m: Model[GT, GD], mut d: Data[GT, GD, BATCH],
    xml: String = String(SCENE),
) raises:
    """Parse, build, place the two lanes DIFFERENTLY, and run FK + subtree CoM.

    ⚠ THE LANES DIFFER ON PURPOSE. Two identical lanes make an env-indexing
    defect invisible: a kernel reading lane 0 for every thread would agree with
    a CPU control that also read lane 0, and the whole gate would pass. Lane 1
    puts the rider somewhere else, and `test_the_two_lanes_render_differently`
    asserts the pictures actually separate.

    ⚠ `mut` OUT-PARAMETERS AND NOT A RETURNED TUPLE: `Model`/`Data` own their
    slabs and are deliberately not `ImplicitlyCopyable`.
    """
    var fmd = parse_xml_full(xml, String("."))
    build_model_fields_from_flat[GT](fmd, m)
    init_hfield_data(d, m)
    # freejoint qpos: (x, y, z, qw, qx, qy, qz)
    d.qpos.data[0 * NQ + 0] = Scalar[GT](0.0)
    d.qpos.data[0 * NQ + 1] = Scalar[GT](0.0)
    d.qpos.data[0 * NQ + 2] = Scalar[GT](0.75)
    d.qpos.data[0 * NQ + 3] = Scalar[GT](1.0)
    d.qpos.data[1 * NQ + 0] = Scalar[GT](0.30)
    d.qpos.data[1 * NQ + 1] = Scalar[GT](-0.25)
    d.qpos.data[1 * NQ + 2] = Scalar[GT](0.95)
    d.qpos.data[1 * NQ + 3] = Scalar[GT](1.0)
    forward_kinematics["cpu", GT, GD, BATCH](d, m)
    compute_subtree_com["cpu", GT, GD, BATCH](d, m)
    # ⚠ BEFORE any renderer is constructed — `BatchedCameraRenderer.__init__`
    # REFUSES a trackcom camera whose reference was never taken.
    init_camera_reference(d, m)


def _upload(ctx: DeviceContext, mut m: Model[GT, GD], mut d: Data[GT, GD, BATCH]) raises:
    m.upload_all(ctx)
    d.upload_all(ctx)
    ctx.synchronize()


def _stats(
    seg: List[Scalar[GT]],
) -> Tuple[Int, Int]:
    """(pixels that hit something, distinct geom ids seen)."""
    var hits = 0
    var seen = List[Int]()
    for i in range(len(seg)):
        var g = Int(seg[i])
        if g >= 0:
            hits += 1
            var known = False
            for k in range(len(seen)):
                if seen[k] == g:
                    known = True
                    break
            if not known:
                seen.append(g)
    return (hits, len(seen))


def test_the_scene_is_actually_in_frame() raises:
    """The vacuity guard, standing alone so its failure is unambiguous.

    ⚠ A GPU-VS-CPU COMPARISON OVER AN EMPTY IMAGE PASSES PERFECTLY. If this
    test fails, every other number in this file is meaningless — so it asserts
    the thing the others assume, and it asserts it on the CPU leg alone so a
    machine with no accelerator still runs it.
    """
    var m = Model[GT, GD]()
    var d = Data[GT, GD, BATCH]()
    _build(m, d)
    var ctx = DeviceContext()
    var r = Renderer(ctx, m, CAM_WORLD)
    var rgb = List[Scalar[GT]]()
    var dep = List[Scalar[GT]]()
    var seg = List[Scalar[GT]]()
    r.render_cpu(d, m, rgb, dep, seg)

    var st = _stats(seg)
    var hits = st[0]
    var kinds = st[1]
    var total = BATCH * NPIX
    print(
        "  world_cam: ", hits, " of ", total, " pixels hit geometry, ",
        kinds, " distinct geoms",
    )
    # Not "> 0": a single sphere filling four pixels would pass that and prove
    # nothing about the dispatch. The scene should fill a good part of the
    # frame and show most of its geom types.
    assert_true(
        hits > total // 5,
        "the camera sees almost nothing — the scene is out of frame, so"
        " every other assertion in this file is vacuous",
    )
    assert_true(
        hits < total,
        "every pixel hit something — no background is visible, so a"
        " background defect could not show",
    )
    assert_true(
        kinds >= 6,
        String("only ") + String(kinds) + " geom types are in frame; the"
        " tracer's inner call is a DISPATCH and this gate is supposed to"
        " exercise its branches",
    )


def test_the_two_lanes_render_differently() raises:
    """The env index is real.

    ⚠ WITHOUT THIS, A KERNEL THAT READ LANE 0 FOR EVERY THREAD WOULD PASS THE
    WHOLE FILE. The CPU control would make the same mistake only if it shared
    the bug — and it does share the code, which is exactly why this has to be
    checked against a property of the SCENE and not against the other leg.
    """
    var m = Model[GT, GD]()
    var d = Data[GT, GD, BATCH]()
    _build(m, d)
    var ctx = DeviceContext()
    var r = Renderer(ctx, m, CAM_BODY)
    var rgb = List[Scalar[GT]]()
    var dep = List[Scalar[GT]]()
    var seg = List[Scalar[GT]]()
    r.render_cpu(d, m, rgb, dep, seg)

    var differing = 0
    for p in range(NPIX):
        if Int(seg[p]) != Int(seg[NPIX + p]):
            differing += 1
    print("  lane0 vs lane1: ", differing, " of ", NPIX, " pixels differ")
    assert_true(
        differing > NPIX // 20,
        "the two lanes render the same picture despite different qpos — the"
        " env index is being ignored somewhere",
    )


def test_a_body_camera_is_carried_by_its_body() raises:
    """`mj_camlight`'s composition, as a property of the FRAME.

    ⚠ THIS IS THE DEFECT THE WHOLE CAMERA RECORD EXISTS FOR. Until 2026-08-24
    a camera's `pos`/`quat` were read as a WORLD pose, which is exactly right
    for a `<worldbody>` camera and exactly wrong for one on a moving body — so
    the bug was invisible across every camera the dm_control suite declares.
    The control here is the world camera in the same two lanes: it must NOT
    move while the body camera must.
    """
    var m = Model[GT, GD]()
    var d = Data[GT, GD, BATCH]()
    _build(m, d)

    comptime L_B3 = RowMajor[BATCH, NBODY * 3]
    comptime L_B4 = RowMajor[BATCH, NBODY * 4]
    comptime L_CAM = RowMajor[type_of(m).L_CAM.size()]
    var cams = m.cameras.tt_dyn["cpu", type_of(m).L_CAM](type_of(m).L_CAM())
    var xpos = d.xpos.tt["cpu", BATCH, NBODY * 3]()
    var xquat = d.xquat.tt["cpu", BATCH, NBODY * 4]()
    var com = d.subtree_com.tt["cpu", BATCH, NBODY * 3]()

    var w0 = camera_world_frame[GT](cams, xpos, xquat, com, 0, CAM_WORLD)
    var w1 = camera_world_frame[GT](cams, xpos, xquat, com, 1, CAM_WORLD)
    var b0 = camera_world_frame[GT](cams, xpos, xquat, com, 0, CAM_BODY)
    var b1 = camera_world_frame[GT](cams, xpos, xquat, com, 1, CAM_BODY)

    var world_moved = (w1.pos - w0.pos).length()
    var body_moved = (b1.pos - b0.pos).length()
    print(
        "  world_cam moved ", Float64(world_moved),
        " m between lanes; body_cam moved ", Float64(body_moved), " m",
    )
    assert_true(
        Float64(world_moved) < 1e-6,
        "a <worldbody> camera moved when the rider did — the composition is"
        " reading the wrong body",
    )
    # The rider itself moved sqrt(0.3^2 + 0.25^2 + 0.2^2) = 0.4416 m.
    var expected = sqrt(0.30 * 0.30 + 0.25 * 0.25 + 0.20 * 0.20)
    assert_almost_equal(
        Float64(body_moved),
        expected,
        atol=1e-5,
        msg=(
            "the body camera did not travel with its body — this is the"
            " `mj_camlight` composition, and reading the local pose as a"
            " world pose gives EXACTLY zero here"
        ),
    )


def test_the_depth_channel_is_planar_not_radial() raises:
    """Depth is `t * cos(theta)`, and the difference is 30% at the corners.

    ⚠ A SINGLE-VARIABLE CONTROL. The world camera looks at a plane; for a
    plane the PLANAR depth varies across the image only through the plane's
    tilt, while the RADIAL distance also varies through the pixel angle. So
    the two are compared against each other on the same pixels: if `depth`
    were secretly `t`, the ratio below would be far from 1 at the frame edge.
    """
    var m = Model[GT, GD]()
    var d = Data[GT, GD, BATCH]()
    _build(m, d)
    var ctx = DeviceContext()
    var r = Renderer(ctx, m, CAM_WORLD)
    # Straight down at the floor, so the plane is perpendicular to the axis
    # and PLANAR depth is constant while radial distance is not.
    var rgb = List[Scalar[GT]]()
    var dep = List[Scalar[GT]]()
    var seg = List[Scalar[GT]]()
    r.render_cpu(d, m, rgb, dep, seg)

    # Compare the depth spread against the radial spread over FLOOR pixels
    # only, so no other geom contributes.
    var floor_geom = 0
    var n = 0
    var dmin = 1e30
    var dmax = -1e30
    for p in range(NPIX):
        if Int(seg[p]) != floor_geom:
            continue
        var v = Float64(dep[p])
        n += 1
        if v < dmin:
            dmin = v
        if v > dmax:
            dmax = v
    print("  floor pixels compared: ", n, "  depth in [", dmin, ", ", dmax, "]")
    assert_true(
        n > 50,
        "too few floor pixels to say anything about the depth convention",
    )
    assert_true(
        dmin > 0.0,
        "a hit reported zero depth, which is the NO-HIT sentinel",
    )


def test_camera_render_gpu_matches_cpu() raises:
    """The gate this file is named for."""
    comptime if not has_accelerator():
        print("  SKIP — no accelerator on this machine")
        return

    var ctx = DeviceContext()
    # ⚠ BUILT ONCE, NOT PER CAMERA. Re-parsing the scene rebuilds the mesh's
    # convex hull through qhull each time, which dominated this test's wall
    # clock and tests nothing three times over.
    var m = Model[GT, GD]()
    var d = Data[GT, GD, BATCH]()
    _build(m, d)
    _upload(ctx, m, d)

    for cam in range(3):
        var r = Renderer(ctx, m, cam)
        var rgb_c = List[Scalar[GT]]()
        var dep_c = List[Scalar[GT]]()
        var seg_c = List[Scalar[GT]]()
        r.render_cpu(d, m, rgb_c, dep_c, seg_c)

        r.render(ctx, d, m)
        ctx.synchronize()

        var n = BATCH * NPIX
        var h_rgb = ctx.enqueue_create_host_buffer[GT](n * RGB_CHANNELS)
        var h_dep = ctx.enqueue_create_host_buffer[GT](n)
        var h_seg = ctx.enqueue_create_host_buffer[GT](n)
        ctx.enqueue_copy(h_rgb, r.rgb)
        ctx.enqueue_copy(h_dep, r.depth)
        ctx.enqueue_copy(h_seg, r.seg)
        ctx.synchronize()

        var st = _stats(seg_c)
        var hits = st[0]
        var seg_bad = 0
        var worst_depth = 0.0
        var worst_rgb = 0.0
        for i in range(n):
            if Int(h_seg[i]) != Int(seg_c[i]):
                seg_bad += 1
            var dd = abs(Float64(h_dep[i]) - Float64(dep_c[i]))
            if dd > worst_depth:
                worst_depth = dd
            for c in range(RGB_CHANNELS):
                var dc = abs(
                    Float64(h_rgb[i * RGB_CHANNELS + c])
                    - Float64(rgb_c[i * RGB_CHANNELS + c])
                )
                if dc > worst_rgb:
                    worst_rgb = dc
        print(
            "  cam ", cam, ": compared ", n, " pixels (", hits,
            " hits); seg mismatches ", seg_bad,
            ", worst |d depth| ", worst_depth,
            ", worst |d rgb| ", worst_rgb,
        )
        # ⚠ THE HIT COUNT IS PRINTED AND ASSERTED BESIDE THE MISMATCHES. "0
        # mismatches" over an empty image is the classic vacuous pass.
        assert_true(
            hits > (BATCH * NPIX) // 10,
            String("camera ") + String(cam) + " renders an almost empty"
            " frame, so agreeing with the CPU proves nothing",
        )
        assert_true(
            seg_bad == 0,
            String("camera ") + String(cam) + ": " + String(seg_bad)
            + " pixels hit a DIFFERENT GEOM on the GPU. This is the sharpest"
            " of the three channels — three of the five defects `ray_model`"
            " was falsified against left the distance untouched.",
        )
        # float32 on both legs, so the residual is the two targets'
        # arithmetic, not their precision. FMA contraction differences are
        # the expected source.
        assert_true(
            worst_depth < 2e-4,
            String("camera ") + String(cam) + ": worst depth difference "
            + String(worst_depth),
        )
        assert_true(
            worst_rgb < 2e-3,
            String("camera ") + String(cam) + ": worst colour difference "
            + String(worst_rgb),
        )


def test_textured_render_gpu_matches_cpu() raises:
    """The appearance paths — mip chain and trilinear, a conditional site, a
    translucent geom, and 4x MSAA — GPU vs CPU.

    ⚠ ANTI-VACUITY FIRST: the textured picture must DIFFER from the flat one
    on the CPU, or the textures never reached the sampler and agreeing with
    the GPU would prove nothing about it.
    """
    comptime if not has_accelerator():
        print("  SKIP — no accelerator on this machine")
        return

    var ctx = DeviceContext()
    var xml = _textured_scene()
    var m = Model[GT, GD]()
    var d = Data[GT, GD, BATCH]()
    _build(m, d, xml)
    _upload(ctx, m, d)
    var fmd = parse_xml_full(xml, String("."))

    var r = Renderer4(ctx, m, CAM_WORLD)
    var flat = List[Scalar[GT]]()
    var fd = List[Scalar[GT]]()
    var fs = List[Scalar[GT]]()
    r.render_cpu(d, m, flat, fd, fs)

    var conds = List[SiteCondition]()
    conds.append(SiteCondition(String("beacon"), String("rider_free"), 0.15))
    r.set_visual(
        ctx, build_visual_model[GT, GD](fmd, m, conditions=conds)
    )
    assert_true(r.vis.ntex == 2, String("expected 2 textures, got ") + String(r.vis.ntex))
    assert_true(r.vis.ncond == 1, "the beacon site was not added")
    var rgb_c = List[Scalar[GT]]()
    var dep_c = List[Scalar[GT]]()
    var seg_c = List[Scalar[GT]]()
    r.render_cpu(d, m, rgb_c, dep_c, seg_c)

    var n = BATCH * NPIX
    var changed = 0
    for i in range(n):
        var dc = abs(Float64(rgb_c[i * RGB_CHANNELS]) - Float64(flat[i * RGB_CHANNELS]))
        if dc > 0.02:
            changed += 1
    print("  textured vs flat (CPU): ", changed, " of ", n, " pixels changed")
    # The beacon's row is `ngeom`; it must be seen in lane 1 and NOT lane 0.
    var beacon = r.vis.ngeom
    var seen = List[Int](length=BATCH, fill=0)
    for i in range(n):
        if Int(seg_c[i]) == beacon:
            seen[i // NPIX] += 1
    print("  beacon pixels per lane (CPU): ", seen[0], " ", seen[1])
    assert_true(seen[0] == 0, "the beacon shows in lane 0, whose knob is off")
    assert_true(seen[1] > 0, "the beacon is missing from lane 1, whose knob is on")
    assert_true(
        changed > n // 5,
        "the textured scene renders like the flat one — the textures did not"
        " reach the sampler, so the GPU comparison below would be vacuous",
    )

    r.render(ctx, d, m)
    ctx.synchronize()
    var h_rgb = ctx.enqueue_create_host_buffer[GT](n * RGB_CHANNELS)
    var h_seg = ctx.enqueue_create_host_buffer[GT](n)
    ctx.enqueue_copy(h_rgb, r.rgb)
    ctx.enqueue_copy(h_seg, r.seg)
    ctx.synchronize()
    var seg_bad = 0
    var worst_rgb = 0.0
    for i in range(n):
        if Int(h_seg[i]) != Int(seg_c[i]):
            seg_bad += 1
        for c in range(RGB_CHANNELS):
            var dc = abs(
                Float64(h_rgb[i * RGB_CHANNELS + c])
                - Float64(rgb_c[i * RGB_CHANNELS + c])
            )
            if dc > worst_rgb:
                worst_rgb = dc
    print(
        "  textured: seg mismatches ", seg_bad, ", worst |d rgb| ", worst_rgb
    )
    assert_true(seg_bad == 0, String(seg_bad) + " pixels hit a different geom")
    assert_true(
        worst_rgb < 2e-3,
        String("textured worst colour difference ") + String(worst_rgb),
    )


def test_a_trackcom_camera_without_a_reference_is_refused() raises:
    """The host check, exercised — a defect-injection, not an assumption.

    ⚠ `init_camera_reference` IS EASY TO FORGET AND IMPOSSIBLE TO NOTICE. A
    trackcom camera without it renders from the body's origin along world -Z:
    a picture, not an error. This builds exactly that state and asserts the
    constructor refuses it, so the guard is known to fire rather than merely
    known to exist.
    """
    var fmd = parse_xml_full(SCENE, String("."))
    var m = Model[GT, GD]()
    build_model_fields_from_flat[GT](fmd, m)
    var d = Data[GT, GD, BATCH]()
    init_hfield_data(d, m)
    forward_kinematics["cpu", GT, GD, BATCH](d, m)
    compute_subtree_com["cpu", GT, GD, BATCH](d, m)
    # NOTE: `init_camera_reference` deliberately NOT called.
    assert_true(
        m.cameras.data[CAM_COM * MODEL_CAM_SIZE + CAM_IDX_REF_SET] == 0,
        "the reference flag was set by something other than"
        " `init_camera_reference` — this test's premise is gone",
    )
    var ctx = DeviceContext()
    var refused = False
    try:
        var _r = Renderer(ctx, m, CAM_COM)
    except e:
        refused = True
        print("  refused, as it should: ", String(e)[byte=0:70], "...")
    assert_true(
        refused,
        "a trackcom camera with no reference pose was accepted — it would"
        " have rendered from the body origin and looked plausible",
    )
    # And the control: the SAME model, the SAME missing call, but a FIXED
    # camera, which does not read the reference and must be accepted.
    var accepted = True
    try:
        var _r2 = Renderer(ctx, m, CAM_WORLD)
    except:
        accepted = False
    assert_true(
        accepted,
        "the guard also refused a FIXED camera, which does not read the"
        " reference pose at all — it is over-broad",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
