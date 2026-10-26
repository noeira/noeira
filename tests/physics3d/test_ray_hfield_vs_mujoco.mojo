"""`ray/hfield.mojo` vs `mj_rayHfield` — rays at a real 8x8 elevation grid.

    pixi run mojo run -I . tests/physics3d/test_ray_hfield_vs_mujoco.mojo

Both sides read the SAME model string, and the grid our side walks is OUR
`Model.hfield_data` — not a copy of MuJoCo's. That the two grids agree is
`test_hfield_vs_mujoco::test_the_elevation_grid_matches_mjmodel`'s job and is
asserted here again as a precondition, because a ray gate that silently walked
a different surface would report a ray defect for a parse one.

⚠⚠ THE HFIELD GEOM IS DELIBERATELY NOT AT THE ORIGIN. It carries a `pos` and
an `euler`, and the reason is the defect this session already paid for once: a
camera composition was invisible for the entire port because every camera sat
on the worldbody, where the transform is the identity. A heightfield at the
origin with no rotation makes `ray_map` the identity too, and every frame error
in `ray_hfield` — the two box offsets along the field's own +z, the normal
rotated back out at the end — would read as exact.
[[feedback_the_identity_commutes_so_the_gate_is_blind]]

WHAT THIS GATE WAS PROVEN ABLE TO FAIL, AND THE ONE THING IT CANNOT SEE
=======================================================================
Defects injected into `ray/hfield.mojo` one at a time:

  injected defect                             caught by         |dt|
  -----------------------------------------   ---------------   --------
  the four side walls (stage 4) removed        7 splits         0.96
  second triangle's winding swapped            |dnormal| 1.88   UNCHANGED
  segment starts at the box entry, not 0       8 splits         1.04
  the base box's hit discarded                33 splits         0.77
  cell window shrunk by one cell each side    15 splits         1.08
  ---
  the reference's `+-1` cell PADDING removed   NOTHING           UNCHANGED

⚠ THE LAST ROW IS A STATED LIMIT, NOT A PASS. Removing the padding changes no
answer here, while shrinking the window by one cell is caught immediately — so
the window is TIGHT (the sweep does reach cells at its boundary) and the extra
padded cell is simply slack on a 1 m field of 8x8 cells, where one cell is
14 cm. The padding is in the reference for a surface that rises to meet the ray
outside the box footprint, which needs a finer grid than this asset to produce.
**Nothing here gates it.** Do not read the green as covering it.

⚠ FOUR RAY FAMILIES, AND THE HORIZONTAL ONE IS THE POINT. `mj_rayHfield` has
four stages and the last — the four vertical SIDES of the top box — only fires
for a ray that arrives at the terrain from the side, passing under no triangle
at all. That is precisely what `quadruped escape`'s rangefinders are: twenty
horizontal rays from a body standing on the terrain. A sweep of downward rays
covers stages 1-3 and leaves the stage this feature exists for untested.
The per-family hit counts are printed so that stays visible.
"""

from std.math import abs, sqrt
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite

from noeira.math3d import Vec3 as Vec3Generic, Quat as QuatGeneric
from noeira.physics3d.fields import (
    Data, Model, DynDims, init_hfield_data, DYN1, rl1,
)
from noeira.physics3d.parser.full_parser import parse_xml_full
from noeira.physics3d.parser.runtime_load import (
    dims_from_flat,
    build_model_runtime,
    spec_fields_runtime,
)
from noeira.physics3d.constants import GEOM_HFIELD
from noeira.physics3d.gpu.constants import (
    MODEL_GEOM_SIZE,
    GEOM_IDX_TYPE,
    GEOM_IDX_BODY,
    GEOM_IDX_POS_X,
    GEOM_IDX_POS_Y,
    GEOM_IDX_POS_Z,
    GEOM_IDX_QUAT_X,
    GEOM_IDX_QUAT_Y,
    GEOM_IDX_QUAT_Z,
    GEOM_IDX_QUAT_W,
    GEOM_IDX_HFIELD_ID,
    MODEL_HFIELD_META_SIZE,
    HFIELD_META_IDX_ADR,
    HFIELD_META_IDX_NROW,
    HFIELD_META_IDX_NCOL,
    HFIELD_META_IDX_SIZE_X,
    HFIELD_META_IDX_SIZE_Y,
    HFIELD_META_IDX_SIZE_Z,
    HFIELD_META_IDX_SIZE_BASE,
)
from noeira.physics3d.ray import ray_hfield

comptime DT = DType.float64
comptime Vec3 = Vec3Generic[DT]
comptime Quat = QuatGeneric[DT]

# The same 8x8 asset `test_hfield_vs_mujoco` uses, but the geom is MOVED and
# TURNED — see the module docstring on why the origin would be vacuous.
#
# ⚠ `euler` IS IN DEGREES HERE. MJCF's `<compiler angle>` defaults to degree
# and this model does not override it. The first draft wrote
# `euler="0.20 -0.15 0.35"` meaning radians, which is a fifth of a degree —
# indistinguishable from no rotation, and `test_the_gate_is_not_axis_aligned`
# caught it (`1-|w|` was 1e-5 against its 1e-3 floor) before any ray was cast.
# That guard is the only reason this file is not silently axis-aligned.
comptime HF_XML = String(
    """
<mujoco model="hfield ray gate">
  <asset>
    <hfield name="terrain" file="tests/physics3d/assets/hf_8x8.bin" size="0.5 0.5 0.2 0.1"/>
  </asset>
  <worldbody>
    <geom name="ground" type="hfield" hfield="terrain" pos="0.13 -0.07 0.05" euler="12 -20 35"/>
  </worldbody>
</mujoco>
"""
)

comptime NCASE = 400


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
    # ⚠ THE GRID LIVES IN `Data` NOW, one per environment — see
    # `fields/model.mojo::hfield_data0`. A `Data` whose `init_hfield_data`
    # never ran holds ZEROS, i.e. a flat terrain, so this file would compare
    # two engines over different surfaces and call it a ray defect.
    var d: Data[DT, DynDims, 1]
    var dims: DynDims

    def __init__(out self) raises:
        var fmd = parse_xml_full(HF_XML, String("."))
        var dims = dims_from_flat(fmd, max_contacts=8, nmesh_verts=8)
        var m = Model[DT, DynDims](dims)
        build_model_runtime[DT](fmd, dims, m)
        _ = spec_fields_runtime[DT](fmd, dims, m)
        var d = Data[DT, DynDims, 1](dims)
        init_hfield_data(d, m)
        self.m = m^
        self.d = d^
        self.dims = dims


def _geom_world(b: Built) raises -> Tuple[Vec3, Quat]:
    """The hfield geom's world pose, from OUR record.

    The geom is on the worldbody (`GEOM_IDX_BODY == 0`), so its stored local
    pose IS its world pose — the branch `_geom_world_pos` takes first. Asserted
    rather than assumed, because on any other body this would need `Data.xpos`
    and silently returning the local pose would be wrong by the body transform.
    """
    var body = Int(Float64(b.m.geoms.data[GEOM_IDX_BODY]))
    assert_true(
        body == 0,
        "the hfield geom moved off the worldbody — this helper returns its"
        " LOCAL pose and would now be wrong by the body transform",
    )
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
    )


def test_the_gate_is_not_axis_aligned() raises:
    """The precondition the whole file rests on: the geom is moved and turned.

    If this ever passes trivially the sweep below stops testing `ray_map`, the
    two box offsets and the normal rotation, and nothing else would say so.
    """
    var b = Built()
    var g = _geom_world(b)
    var pos = g[0]
    var q = g[1]
    var off = sqrt(pos.x * pos.x + pos.y * pos.y + pos.z * pos.z)
    # |w| == 1 is the identity rotation; anything less is a real turn.
    var turn = 1.0 - abs(Float64(q.w))
    assert_true(off > 1e-3, "hfield geom is at the origin — gate is vacuous")
    assert_true(turn > 1e-3, "hfield geom is unrotated — gate is vacuous")
    print("  geom offset", off, " rotation 1-|w|", turn)


def test_our_grid_is_mujocos_grid() raises:
    """Precondition: the surface both sides walk is the same surface."""
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(String(HF_XML))
    var b = Built()

    var hid = Int(Float64(b.m.geoms.data[GEOM_IDX_HFIELD_ID]))
    var base = hid * MODEL_HFIELD_META_SIZE
    var adr = Int(Float64(b.m.hfield_meta.data[base + HFIELD_META_IDX_ADR]))
    var nrow = Int(Float64(b.m.hfield_meta.data[base + HFIELD_META_IDX_NROW]))
    var ncol = Int(Float64(b.m.hfield_meta.data[base + HFIELD_META_IDX_NCOL]))

    assert_true(
        nrow == Int(py=m.hfield_nrow[0]) and ncol == Int(py=m.hfield_ncol[0]),
        "grid dimensions differ: ours " + String(nrow) + "x" + String(ncol),
    )
    var worst = 0.0
    for i in range(nrow * ncol):
        worst = max(
            worst,
            abs(
                Float64(b.d.hfield_data.data[adr + i])
                - Float64(py=m.hfield_data[i])
            ),
        )
    print("  grid", nrow, "x", ncol, " worst |d elevation|", worst)
    # ⚠ MuJoCo stores `hfield_data` as float32; ours is the model's DTYPE.
    # Same source, same rescale, so they agree to float32 and no further.
    assert_true(worst < 1e-7, "elevation grids differ by " + String(worst))


def test_ray_hfield_vs_mujoco() raises:
    var mujoco = Python.import_module("mujoco")
    var np = Python.import_module("numpy")
    var m = mujoco.MjModel.from_xml_string(String(HF_XML))
    var d = mujoco.MjData(m)
    _ = mujoco.mj_forward(m, d)

    var b = Built()
    var g = _geom_world(b)
    var pos = g[0]
    var quat = g[1]

    # Our composed geom pose against MuJoCo's, so a frame error is attributed
    # here rather than showing up as a ray residual.
    var dp = 0.0
    for k in range(3):
        var ours = pos.x if k == 0 else (pos.y if k == 1 else pos.z)
        dp = max(dp, abs(Float64(ours) - Float64(py=d.geom_xpos[0][k])))
    assert_true(dp < 1e-12, "geom_xpos differs by " + String(dp))

    var hid = Int(Float64(b.m.geoms.data[GEOM_IDX_HFIELD_ID]))
    var base = hid * MODEL_HFIELD_META_SIZE
    var adr = Int(Float64(b.m.hfield_meta.data[base + HFIELD_META_IDX_ADR]))
    var nrow = Int(Float64(b.m.hfield_meta.data[base + HFIELD_META_IDX_NROW]))
    var ncol = Int(Float64(b.m.hfield_meta.data[base + HFIELD_META_IDX_NCOL]))
    var sx = Float64(b.m.hfield_meta.data[base + HFIELD_META_IDX_SIZE_X])
    var sy = Float64(b.m.hfield_meta.data[base + HFIELD_META_IDX_SIZE_Y])
    var sz = Float64(b.m.hfield_meta.data[base + HFIELD_META_IDX_SIZE_Z])
    var sb = Float64(b.m.hfield_meta.data[base + HFIELD_META_IDX_SIZE_BASE])

    var a_pnt = np.zeros(3)
    var a_vec = np.zeros(3)
    var a_nrm = np.zeros(3)

    # ⚠ Built ONCE, outside the loop: `lt_dyn` needs `mut`, and the grid does
    # not move between rays.
    var hf_view = b.d.hfield_data.tt_dyn["cpu", DYN1](rl1(nrow * ncol))

    var rng = Lcg(0xC0FFEE)
    var hits = Array[Int, 4](fill=0)
    var cases = Array[Int, 4](fill=0)
    var split = 0
    var worst_t = 0.0
    var worst_n = 0.0

    # Local axes of the field, used to aim the horizontal family along the
    # surface rather than along world x/y — the geom is turned, so those are
    # not the same thing.
    var lx = quat.rotate_vec(Vec3(1.0, 0.0, 0.0))
    var ly = quat.rotate_vec(Vec3(0.0, 1.0, 0.0))
    var lz = quat.rotate_vec(Vec3(0.0, 0.0, 1.0))

    for _ in range(NCASE):
        var pick = rng.u01()
        var fam: Int
        var eye: Vec3
        var aim: Vec3
        if pick < 0.3:
            # (1) from above, looking down — the ordinary terrain query.
            fam = 0
            eye = pos + lz * (0.6 + rng.u01() * 0.8) + lx * rng.sym(0.6) + ly * rng.sym(0.6)
            aim = pos + lx * rng.sym(0.5) + ly * rng.sym(0.5)
        elif pick < 0.65:
            # (2) HORIZONTAL, at terrain height — the stage-4 side walls, and
            # what a rangefinder on a walking robot actually casts.
            fam = 1
            # Start OUTSIDE the field's x/y extent (radius 0.5) and low, so
            # the ray reaches the wall rather than the surface from above.
            eye = (
                pos
                + lx * rng.sym(1.2)
                + ly * rng.sym(1.2)
                + lz * (0.02 + rng.u01() * 0.16)
            )
            aim = pos + lx * rng.sym(0.4) + ly * rng.sym(0.4) + lz * (
                0.02 + rng.u01() * 0.16
            )
        elif pick < 0.85:
            # (3) origin INSIDE the field's volume, pointing anywhere — the
            # family that caught the capsule defect in the `mju_rayGeom` sweep.
            fam = 2
            eye = pos + lx * rng.sym(0.4) + ly * rng.sym(0.4) + lz * rng.sym(0.15)
            aim = eye + Vec3(rng.sym(1.0), rng.sym(1.0), rng.sym(1.0))
        else:
            # (4) anywhere to anywhere — misses, grazes, edges.
            fam = 3
            # ⚠ Aimed at the field's own extent, not at a box around it. The
            # first draft fired into a +-2 m cube and landed 8 hits in 64 —
            # caught by the per-family floor below, which is what that floor
            # is for. Misses are still wanted here (this is the family that
            # exercises the reject paths), just not 88% of them.
            eye = pos + Vec3(rng.sym(2.0), rng.sym(2.0), rng.sym(2.0))
            aim = (
                pos
                + lx * rng.sym(0.8)
                + ly * rng.sym(0.8)
                + lz * rng.sym(0.3)
            )

        var vec = aim - eye
        cases[fam] += 1

        var ours = ray_hfield[DT, DYN1](
            pos, quat, nrow, ncol,
            Scalar[DT](sx), Scalar[DT](sy), Scalar[DT](sz), Scalar[DT](sb),
            hf_view, adr, eye, vec,
        )

        a_pnt[0] = eye.x
        a_pnt[1] = eye.y
        a_pnt[2] = eye.z
        a_vec[0] = vec.x
        a_vec[1] = vec.y
        a_vec[2] = vec.z
        var t_mj = Float64(
            py=mujoco.mj_rayHfield(m, d, 0, a_pnt, a_vec, a_nrm)
        )

        var t_ours = Float64(ours[0])
        var hit_ours = t_ours >= 0.0
        var hit_mj = t_mj >= 0.0
        if hit_ours != hit_mj:
            split += 1
            continue
        if not hit_mj:
            continue

        hits[fam] += 1
        worst_t = max(worst_t, abs(t_ours - t_mj))
        var n = ours[1]
        worst_n = max(worst_n, abs(Float64(n.x) - Float64(py=a_nrm[0])))
        worst_n = max(worst_n, abs(Float64(n.y) - Float64(py=a_nrm[1])))
        worst_n = max(worst_n, abs(Float64(n.z) - Float64(py=a_nrm[2])))

    print("    family              hits/cases")
    print("    from above         ", hits[0], "/", cases[0])
    print("    HORIZONTAL (sides) ", hits[1], "/", cases[1])
    print("    origin inside      ", hits[2], "/", cases[2])
    print("    anywhere           ", hits[3], "/", cases[3])
    print("  worst |dt|      ", worst_t)
    print("  worst |dnormal| ", worst_n)
    print("  splits          ", split)

    for f in range(4):
        assert_true(
            hits[f] > cases[f] // 8,
            "family " + String(f) + " hit only " + String(hits[f]) + " of "
            + String(cases[f]) + " — that row is vacuous, fix the SAMPLER",
        )
    assert_true(
        split == 0,
        String(split) + " rays where one side hit and the other missed",
    )
    # Sized by the elevation grid's float32 provenance, not by taste: an
    # elevation differing in the last float32 bit moves the surface by ~1e-8
    # and the hit distance with it.
    assert_true(worst_t < 1e-6, "worst |dt| " + String(worst_t))
    assert_true(worst_n < 1e-5, "worst |dnormal| " + String(worst_n))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
