"""`ray/mesh.mojo` vs `mj_rayMesh` — rays at a NON-CONVEX mesh.

    pixi run mojo run -I . tests/physics3d/test_ray_mesh_vs_mujoco.mojo

⚠⚠ THE FIXTURE HAS TO BE NON-CONVEX OR THIS FILE PROVES NOTHING. The triangle
store (`Model.mesh_tris`) exists only because `Model.mesh_verts` is the convex
HULL and a ray aimed into a cutout must find the hole. On a convex mesh the
hull IS the mesh, so a `ray_mesh` that quietly walked hull triangles — or a
`ray_mesh` on a model whose soup was never populated but whose hull was —
would agree with MuJoCo on every ray. Every other STL fixture in
`tests/physics3d/assets/` is convex (a cube, a hex prism, an n-gon prism), so
`notch.stl` was added for this: a box with a slot cut in its top face.

`test_the_fixture_is_not_convex` is the standing guard, and it is not a
formality — it asserts the two rays that separate the surfaces:

    straight down the SLOT  ->  z = -0.010   (the slot floor)
    straight down the LID   ->  z = +0.040   (the top face)

A hull would answer +0.040 for BOTH. If those ever agree, the fixture has
stopped being non-convex and the sweep below is measuring nothing.

⚠ `half_extents` WAS AN OPEN QUESTION THIS FILE ANSWERED, and the answer used
to be "a placeholder". `mj_rayMesh` rejects on `geom_size` before touching a
triangle (engine_ray.c:894, :963); MuJoCo stores
`max(|aamm[k]|, |aamm[k+3]|)` there — [0.0423, 0.05, 0.05] for this notch, not
symmetric, because a mesh is recentred on its centre of mass — and our parser
stored 0.5. That was harmless at 5 cm and wrong on any mesh reaching past half
a metre from its frame origin, where every hit beyond that came back as a
MISS. Fixed 2026-09-13 (AUD-46): `fields_build` derives it from the loaded
hull vertices with the same scan `compute_mesh_rbound_at` uses, and
`test_our_mesh_box_equals_mujocos` now asserts the equality rather than the
inequality that was all the placeholder could support.

WHAT THIS GATE WAS PROVEN ABLE TO FAIL
======================================
    injected defect                            caught by       |dt|
    ----------------------------------------   -------------   ---------
    negative triangle distances accepted        hits 454->103   0.21
    normal not rotated out of the local frame   |dnormal| 1.81  UNCHANGED
    ---
    the bounding-box reject removed             NOTHING         UNCHANGED

⚠ THE LAST ROW IS A CONFIRMED PREDICTION, NOT A HOLE. Removing the reject
cannot change an answer — it only skips a rejection — and since our box
over-approximates, it never rejected a real hit in the first place. Identical
output is what should happen; it is recorded so nobody reads the reject as
untested and tightens it without re-measuring.
"""

from std.math import abs, sqrt
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite

from noeira.math3d import Vec3 as Vec3Generic, Quat as QuatGeneric
from noeira.physics3d.fields import Model, DynDims, DYN1, rl1
from noeira.physics3d.parser.full_parser import parse_xml_full
from noeira.physics3d.parser.runtime_load import (
    dims_from_flat,
    build_model_runtime,
    spec_fields_runtime,
)
from noeira.physics3d.gpu.constants import (
    MODEL_GEOM_SIZE,
    GEOM_IDX_BODY,
    GEOM_IDX_POS_X,
    GEOM_IDX_POS_Y,
    GEOM_IDX_POS_Z,
    GEOM_IDX_QUAT_X,
    GEOM_IDX_QUAT_Y,
    GEOM_IDX_QUAT_Z,
    GEOM_IDX_QUAT_W,
    GEOM_IDX_HALF_X,
    GEOM_IDX_HALF_Y,
    GEOM_IDX_HALF_Z,
    GEOM_IDX_MESH_ID,
    MODEL_MESH_META_SIZE,
    MESH_META_IDX_TRIADR,
    MESH_META_IDX_TRINUM,
)
from noeira.physics3d.ray import ray_mesh

comptime DT = DType.float64
comptime Vec3 = Vec3Generic[DT]
comptime Quat = QuatGeneric[DT]

# Moved and turned, for the reason the hfield gate spells out: at the origin
# with no rotation `ray_map` is the identity and every frame error reads exact.
comptime MESH_XML = String(
    """
<mujoco model="notch ray gate">
  <asset>
    <mesh name="notch" file="tests/physics3d/assets/notch.stl"/>
  </asset>
  <worldbody>
    <geom name="g" type="mesh" mesh="notch" pos="0.09 -0.04 0.03" euler="15 -25 40"/>
  </worldbody>
</mujoco>
"""
)

comptime NCASE = 500

# ⚠⚠ THE FIXTURE THAT MAKES AUD-46 VISIBLE. The notch above is 5 cm, so a 0.5
# placeholder over-covered it and cost nothing. Scaled 20x it reaches ~0.85 m
# from its frame origin, which is PAST 0.5 — and `mj_rayMesh`'s box reject
# runs before any triangle, so with the placeholder every hit on the far
# two-thirds of this mesh came back as a MISS. Same asset, same rays, one
# `scale` attribute: the whole difference between a defect that is inert and
# one that silently loses data.
comptime BIG_SCALE = 20.0
comptime BIG_MESH_XML = String(
    """
<mujoco model="notch ray gate, 20x">
  <asset>
    <mesh name="notch" file="tests/physics3d/assets/notch.stl"
          scale="20 20 20"/>
  </asset>
  <worldbody>
    <geom name="g" type="mesh" mesh="notch" pos="0.09 -0.04 0.03" euler="15 -25 40"/>
  </worldbody>
</mujoco>
"""
)

# Set from the measurement, not inherited: our mesh half-extents agree with
# MuJoCo's to 1.1e-9 on this fixture, and the residual is MuJoCo's float32
# `mesh_vert` (mjmodel.h:643) against our float64, not a difference of rule.
comptime MESH_SIZE_TOL = 1e-7
# The 20x fixture scales the same float32 residual with it: 1.1e-9 * 20 is
# 2.2e-8, so the bound moves with the fixture rather than staying at a number
# that happened to fit the small one.
comptime BIG_MESH_SIZE_TOL = 1e-7 * BIG_SCALE
# Ray distances on a metre-scale mesh, set from the measurement below.
comptime BIG_MESH_T_TOL = 1e-6


struct Lcg(Copyable, Movable):
    var s: UInt64

    def __init__(out self, seed: UInt64):
        self.s = seed

    def u01(mut self) -> Float64:
        self.s = self.s * 1664525 + 1013904223
        return Float64((self.s >> 16) & 0xFFFFFFF) / Float64(0x10000000)

    def sym(mut self, a: Float64) -> Float64:
        return (self.u01() * 2.0 - 1.0) * a


struct Built(Movable):
    var m: Model[DT, DynDims]

    def __init__(out self, xml: String = MESH_XML) raises:
        var fmd = parse_xml_full(xml, String("."))
        # ⚠ `nmesh_tri` is what turns the soup ON. Left at its default 0 the
        # model carries no triangles and every ray reports NO HIT — which is
        # what `test_the_soup_is_actually_carried` exists to catch.
        var dims = dims_from_flat(
            fmd, max_contacts=8, nmesh_verts=256, nmesh_tri=64
        )
        var m = Model[DT, DynDims](dims)
        build_model_runtime[DT](fmd, dims, m)
        _ = spec_fields_runtime[DT](fmd, dims, m)
        self.m = m^


def _geom_pose(b: Built) -> Tuple[Vec3, Quat, Vec3]:
    return (
        Vec3(
            Float64(b.m.geoms.data[GEOM_IDX_POS_X]),
            Float64(b.m.geoms.data[GEOM_IDX_POS_Y]),
            Float64(b.m.geoms.data[GEOM_IDX_POS_Z]),
        ),
        Quat(
            Float64(b.m.geoms.data[GEOM_IDX_QUAT_W]),
            Float64(b.m.geoms.data[GEOM_IDX_QUAT_X]),
            Float64(b.m.geoms.data[GEOM_IDX_QUAT_Y]),
            Float64(b.m.geoms.data[GEOM_IDX_QUAT_Z]),
        ),
        Vec3(
            Float64(b.m.geoms.data[GEOM_IDX_HALF_X]),
            Float64(b.m.geoms.data[GEOM_IDX_HALF_Y]),
            Float64(b.m.geoms.data[GEOM_IDX_HALF_Z]),
        ),
    )


def _tri_window(b: Built) -> Tuple[Int, Int]:
    var mid = Int(Float64(b.m.geoms.data[GEOM_IDX_MESH_ID]))
    var base = mid * MODEL_MESH_META_SIZE
    return (
        Int(Float64(b.m.mesh_meta.data[base + MESH_META_IDX_TRIADR])),
        Int(Float64(b.m.mesh_meta.data[base + MESH_META_IDX_TRINUM])),
    )


def test_the_fixture_is_not_convex() raises:
    """A hull answers the same distance for both rays; the mesh does not."""
    var mujoco = Python.import_module("mujoco")
    var np = Python.import_module("numpy")
    var m = mujoco.MjModel.from_xml_string(
        String(
            """
<mujoco model="axis aligned notch">
  <asset><mesh name="notch" file="tests/physics3d/assets/notch.stl"/></asset>
  <worldbody><geom name="g" type="mesh" mesh="notch" pos="0 0 0"/></worldbody>
</mujoco>
"""
        )
    )
    var d = mujoco.MjData(m)
    _ = mujoco.mj_forward(m, d)
    var down = np.zeros(3)
    down[2] = -1.0
    var nrm = np.zeros(3)

    var slot = np.zeros(3)
    slot[2] = 1.0
    var t_slot = Float64(py=mujoco.mj_rayMesh(m, d, 0, slot, down, nrm))
    var lid = np.zeros(3)
    lid[0] = 0.035
    lid[2] = 1.0
    var t_lid = Float64(py=mujoco.mj_rayMesh(m, d, 0, lid, down, nrm))

    var z_slot = 1.0 - t_slot
    var z_lid = 1.0 - t_lid
    print("  down the SLOT z =", z_slot, "  down the LID z =", z_lid)
    assert_true(
        abs(z_slot + 0.01) < 1e-6,
        "the slot floor is at z=" + String(z_slot) + ", expected -0.01 —"
        " notch.stl is not the shape this gate assumes",
    )
    assert_true(
        abs(z_lid - 0.04) < 1e-6,
        "the lid is at z=" + String(z_lid) + ", expected +0.04",
    )
    assert_true(
        abs(z_slot - z_lid) > 1e-3,
        "THE FIXTURE IS CONVEX: both rays land at the same depth, so this"
        " file cannot tell the triangle store from the hull",
    )


def test_the_soup_is_actually_carried() raises:
    """`nmesh_tri` off is silent, so the count is asserted before any ray."""
    var b = Built()
    var w = _tri_window(b)
    print("  triadr", w[0], " trinum", w[1])
    assert_true(
        w[1] == 28,
        "the mesh carries " + String(w[1]) + " triangles, expected 28 —"
        " a soup of 0 is `nmesh_tri` left at its default, and every ray below"
        " would report NO HIT while looking like a clean pass",
    )


def test_our_mesh_box_equals_mujocos() raises:
    """The box `mj_rayMesh` rejects on — now MuJoCo's own, not a placeholder.

    ⚠⚠ THIS USED TO BE AN INEQUALITY, AND IT SAID WHY. A mesh geom carried a
    0.5 half-extent placeholder (AUD-46):

        MuJoCo   [0.0423, 0.05, 0.05]
        ours     [0.5,    0.5,  0.5 ]

    which was harmless on a 5 cm fixture — a too-LARGE reject box costs
    triangles walked, a too-SMALL one silently loses hits — and wrong on any
    mesh reaching past 0.5 m from its frame origin, where every hit beyond
    that came back as a miss.

    `fields_build` now derives it the way `mjCGeom::Compile` does
    (user_objects.cc:4089-4093):

        size[k] = max(|aamm[k]|, |aamm[k+3]|)

    — the half-extent of the smallest box CENTRED ON THE FRAME ORIGIN that
    contains the mesh. ⚠ NOT the AABB half-size: this notch's box is not
    symmetric about its own frame, so the two differ, and the old docstring
    called it "the AABB" when MuJoCo's rule is the origin-centred fold. The
    comparison below is an equality because it now can be — to 1.1e-9, which
    is not our arithmetic. MuJoCo stores `mesh_vert` as `float`
    (mjmodel.h:643), so its vertices are ROUNDED TO FLOAT32 after the
    recentring shift while ours stay double; 0.0423 * float32-eps is ~5e-9,
    and the observed gap is inside that. The bound below is set from that
    measurement: 1e-7, ninety times the observed difference and five orders
    below the 0.46 that the placeholder was wrong by.
    """
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(String(MESH_XML))
    var b = Built()
    var g = _geom_pose(b)
    var half = g[2]
    var worst = 0.0
    for k in range(3):
        var ours = Float64(half.x if k == 0 else (half.y if k == 1 else half.z))
        var theirs = Float64(py=m.geom_size[0][k])
        print("  axis", k, " ours", ours, " MuJoCo", theirs)
        assert_true(
            abs(ours - theirs) <= MESH_SIZE_TOL,
            "axis " + String(k) + ": our box half-extent " + String(ours)
            + ", MuJoCo's " + String(theirs) + ". 0.5 is the placeholder"
            " AUD-46 removed; anything SMALLER than MuJoCo's would make"
            " `ray_mesh`'s reject drop rays the reference reports as hits.",
        )
        worst = max(worst, abs(ours - theirs))
    print("  ours [", half.x, half.y, half.z, "]  worst |d| ", worst)
    # ⚠ NON-VACUITY: an equality against a value that happened to BE the
    # placeholder would pass without testing anything.
    assert_true(
        abs(Float64(half.x) - 0.5) > 1e-6,
        "our half-extent is still exactly 0.5 — the placeholder is back, and"
        " MuJoCo agreeing with it would be a coincidence of this fixture",
    )


def test_a_mesh_past_the_old_placeholder_still_hits() raises:
    """⚠⚠ THE TEST THAT MAKES AUD-46 A DEFECT RATHER THAN A TIDY-UP.

    Everything else in this file runs on a 5 cm notch, which a 0.5 m reject
    box over-covers: the placeholder cost nothing there, which is exactly why
    it survived. Scaled 20x the same mesh reaches ~0.85 m from its frame
    origin, so `ray_box(geom_size)` — which `mj_rayMesh` runs BEFORE any
    triangle (engine_ray.c:963) — would have cut it at 0.5 and returned a
    MISS for every ray landing beyond that.

    The rays below are aimed at the far end on purpose. The assertion is not
    "our t matches" but "we do not report misses where MuJoCo reports hits",
    because a reject box that is too small fails as SILENCE, and a
    `worst |dt|` computed over the surviving hits would look perfect while
    two thirds of the mesh was gone.
    """
    var mujoco = Python.import_module("mujoco")
    var np = Python.import_module("numpy")
    var m = mujoco.MjModel.from_xml_string(String(BIG_MESH_XML))
    var d = mujoco.MjData(m)
    _ = mujoco.mj_forward(m, d)

    var b = Built(BIG_MESH_XML)
    var g = _geom_pose(b)
    var pos = g[0]
    var quat = g[1]
    var half = g[2]
    var w = _tri_window(b)
    var triadr = w[0]
    var ntri = w[1]
    print("  20x mesh half-extents ours [", half.x, half.y, half.z, "]")
    print("  MuJoCo [", Float64(py=m.geom_size[0][0]),
          Float64(py=m.geom_size[0][1]), Float64(py=m.geom_size[0][2]), "]")
    # ⚠ THE PREMISE, CHECKED: this fixture only tests anything if it actually
    # reaches past 0.5.
    var reach = max(
        Float64(py=m.geom_size[0][0]),
        max(Float64(py=m.geom_size[0][1]), Float64(py=m.geom_size[0][2])),
    )
    assert_true(
        reach > 0.5,
        "the 20x mesh only reaches " + String(reach) + " m, which the old"
        " 0.5 placeholder covered — this test cannot see AUD-46",
    )
    for k in range(3):
        var ours = Float64(half.x if k == 0 else (half.y if k == 1 else half.z))
        assert_true(
            abs(ours - Float64(py=m.geom_size[0][k])) <= BIG_MESH_SIZE_TOL,
            "20x axis " + String(k) + ": ours " + String(ours) + ", MuJoCo "
            + String(Float64(py=m.geom_size[0][k])),
        )

    var a_pnt = np.zeros(3)
    var a_vec = np.zeros(3)
    var a_nrm = np.zeros(3)
    var tri_view = b.m.mesh_tris.tt_dyn["cpu", DYN1](rl1(64 * 9))

    var rng = Lcg(0xB16BE11)
    var hits = 0
    var far_hits = 0
    var missed_a_hit = 0
    var worst_t = 0.0
    for _ in range(NCASE):
        # Aim across the whole solid from well outside it, so a good share of
        # the hits land more than 0.5 m from the geom frame origin.
        var eye = pos + Vec3(rng.sym(3.0), rng.sym(3.0), rng.sym(3.0))
        var aim = pos + Vec3(rng.sym(0.9), rng.sym(0.9), rng.sym(0.9))
        var vec = aim - eye
        var ours = ray_mesh[DT, DYN1](
            pos, quat, half, tri_view, triadr, ntri, eye, vec
        )
        a_pnt[0] = eye.x
        a_pnt[1] = eye.y
        a_pnt[2] = eye.z
        a_vec[0] = vec.x
        a_vec[1] = vec.y
        a_vec[2] = vec.z
        var t_mj = Float64(py=mujoco.mj_rayMesh(m, d, 0, a_pnt, a_vec, a_nrm))
        var t_ours = Float64(ours.t)
        if t_mj < 0.0:
            continue
        hits += 1
        # How far from the geom's frame origin the reference's hit lands —
        # the quantity the placeholder was comparing against 0.5.
        var hp = eye + vec * t_mj - pos
        if hp.length() > 0.5:
            far_hits += 1
        if t_ours < 0.0:
            missed_a_hit += 1
            continue
        worst_t = max(worst_t, abs(t_ours - t_mj))
    print("  MuJoCo hits", hits, " of which beyond 0.5 m:", far_hits)
    print("  hits we reported as MISSES:", missed_a_hit)
    print("  worst |dt| over the shared hits:", worst_t)
    assert_true(
        hits > NCASE // 8,
        "only " + String(hits) + " reference hits — the sweep is vacuous",
    )
    # ⚠ NON-VACUITY, AND IT IS THE WHOLE TEST: without far hits this is the
    # 5 cm fixture again with bigger numbers.
    assert_true(
        far_hits > hits // 4,
        "only " + String(far_hits) + " of " + String(hits) + " reference hits"
        " land beyond 0.5 m from the geom origin, so the old placeholder"
        " would have covered nearly all of them and this test proves little",
    )
    assert_true(
        missed_a_hit == 0,
        String(missed_a_hit) + " of " + String(hits) + " MuJoCo hits came"
        " back as misses: the mesh reject box is smaller than the mesh",
    )
    assert_true(
        worst_t <= BIG_MESH_T_TOL,
        "worst |dt| " + String(worst_t) + " over " + String(hits) + " hits",
    )


def test_ray_mesh_vs_mujoco() raises:
    var mujoco = Python.import_module("mujoco")
    var np = Python.import_module("numpy")
    var m = mujoco.MjModel.from_xml_string(String(MESH_XML))
    var d = mujoco.MjData(m)
    _ = mujoco.mj_forward(m, d)

    var b = Built()
    var g = _geom_pose(b)
    var pos = g[0]
    var quat = g[1]
    var half = g[2]
    var w = _tri_window(b)
    var triadr = w[0]
    var ntri = w[1]

    var a_pnt = np.zeros(3)
    var a_vec = np.zeros(3)
    var a_nrm = np.zeros(3)

    # ⚠ Built ONCE: `lt_dyn` needs `mut` and the soup does not move.
    var tri_view = b.m.mesh_tris.tt_dyn["cpu", DYN1](rl1(64 * 9))

    var rng = Lcg(0x5EED11)
    var hits = 0
    var split = 0
    var worst_t = 0.0
    var worst_n = 0.0

    var lx = quat.rotate_vec(Vec3(1.0, 0.0, 0.0))
    var ly = quat.rotate_vec(Vec3(0.0, 1.0, 0.0))
    var lz = quat.rotate_vec(Vec3(0.0, 0.0, 1.0))

    for _ in range(NCASE):
        var pick = rng.u01()
        var eye: Vec3
        var aim: Vec3
        if pick < 0.35:
            # Down the slot's axis, jittered across it — the rays whose answer
            # the hull would get wrong.
            eye = pos + lz * (0.3 + rng.u01() * 0.5) + lx * rng.sym(0.03) + ly * rng.sym(0.05)
            aim = pos + lx * rng.sym(0.03) + ly * rng.sym(0.05)
        elif pick < 0.6:
            # Origin INSIDE the solid — the family that caught the capsule
            # defect in the `mju_rayGeom` sweep.
            eye = pos + lx * rng.sym(0.04) + ly * rng.sym(0.04) + lz * rng.sym(0.03)
            aim = eye + Vec3(rng.sym(1.0), rng.sym(1.0), rng.sym(1.0))
        elif pick < 0.85:
            # General, aimed at the body.
            eye = pos + Vec3(rng.sym(0.6), rng.sym(0.6), rng.sym(0.6))
            aim = pos + lx * rng.sym(0.06) + ly * rng.sym(0.06) + lz * rng.sym(0.05)
        else:
            # Grazing the silhouette.
            eye = pos + Vec3(rng.sym(0.8), rng.sym(0.8), rng.sym(0.8))
            aim = pos + Vec3(rng.sym(1.0), rng.sym(1.0), rng.sym(1.0)).normalized() * 0.055

        var vec = aim - eye
        var ours = ray_mesh[DT, DYN1](
            pos, quat, half, tri_view, triadr, ntri, eye, vec
        )

        a_pnt[0] = eye.x
        a_pnt[1] = eye.y
        a_pnt[2] = eye.z
        a_vec[0] = vec.x
        a_vec[1] = vec.y
        a_vec[2] = vec.z
        var t_mj = Float64(py=mujoco.mj_rayMesh(m, d, 0, a_pnt, a_vec, a_nrm))

        var t_ours = Float64(ours.t)
        if (t_ours >= 0.0) != (t_mj >= 0.0):
            split += 1
            continue
        if t_mj < 0.0:
            continue

        hits += 1
        worst_t = max(worst_t, abs(t_ours - t_mj))
        var n = ours.normal
        worst_n = max(worst_n, abs(Float64(n.x) - Float64(py=a_nrm[0])))
        worst_n = max(worst_n, abs(Float64(n.y) - Float64(py=a_nrm[1])))
        worst_n = max(worst_n, abs(Float64(n.z) - Float64(py=a_nrm[2])))

    print("  hits", hits, "/", NCASE, " splits", split)
    print("  worst |dt|      ", worst_t)
    print("  worst |dnormal| ", worst_n)
    assert_true(
        hits > NCASE // 4,
        "only " + String(hits) + " hits — the sweep is vacuous",
    )
    assert_true(split == 0, String(split) + " hit/miss disagreements")
    # float32-rounded vertices on both sides, so this is the last-bit fold of
    # the plane intersection and not a tolerance chosen to pass.
    assert_true(worst_t < 1e-9, "worst |dt| " + String(worst_t))
    assert_true(worst_n < 1e-9, "worst |dnormal| " + String(worst_n))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
