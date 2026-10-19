"""float32 GJK must not invent a contact between geoms that are centimetres apart.

    pixi run mojo run -I . tests/physics3d/test_gjk_float32_no_phantom_contacts.mojo

⚠ RUN FROM THE REPO ROOT — the fixture meshes are addressed by repo-root
relative path.

WHAT WENT WRONG. GJK's stopping test is

    |v| - s.v̂  <  GJK_TOLERANCE          (`GJK_TOLERANCE = 1e-10`)

MuJoCo's equivalent threshold is 0 for mesh pairs — `engine_collision_gjk.c`
says "if both geoms are discrete, finite convergence is guaranteed; set
tolerance to 0" — and that guarantee is an EXACT-ARITHMETIC one. At float64 it
effectively holds. At float32 the left-hand side is a difference of two dot
products of magnitude |v|, so its rounding floor is about `1e-7 * |v|` — for
robot-scale geometry, hundreds of times ABOVE 1e-10. The test can then never
fire, GJK runs to `GJK_MAX_ITERATIONS`, and what it returns after the cap is
not a converged distance.

⚠⚠ THE FAILURE IS NOT A SMALL ERROR, IT IS AN INVENTED CONTACT. Measured over
the 256-pose sweep below, float32 returned `-0.0` — reported as touching — for
three poses where float64 puts the hulls 7.2, 16.5 and 16.9 CENTIMETRES apart.
A phantom contact at 17 cm injects a constraint row and a force out of nowhere,
and nothing downstream can tell it from a real one.

⚠ EVERY GJK TEST IN THE TREE RUNS AT FLOAT64, so none of this was visible:
`test_mesh_collision`, `test_gjk_simplex`, `test_epa_optimality_cylinder_mesh`
and the MuJoCo comparisons all use float64, where the criterion behaves. The
engine runs float32 in training and in the viewer.

THE FIX is a scale-relative floor at float32 only — the float64 branch is the
literal old constant, so every float64 gate is bit-identical. See
`GJK_TOLERANCE` in `collision/gjk.mojo`.

⚠ THIS GATES THE PHANTOM CONTACT, NOT AN ERROR BOUND, AND THAT IS DELIBERATE.
Two poses in this sweep (185 and 187) carry float32 distance errors of 1.9e-3
and 8.2e-2 that are PRESENT BEFORE AND AFTER the fix, unchanged to the digit —
a separate, still-open float32 robustness problem in the distance path. Writing
a tight error bound here would either fail on those two or be so loose it
gates nothing. "Never report contact when the reference says centimetres" is
the property that actually matters and the one the fix delivers.
"""

from std.math import sqrt, cos, sin, abs
from noeira.core.tiles import tile
from std.testing import assert_true, TestSuite

from noeira.physics3d.collision.convex_hull import load_mesh_hull
from noeira.physics3d.collision.ccd_workspace import L_CCD_WS1
from noeira.physics3d.collision.ccd_workspace_host import ccd_ws_alloc
from noeira.physics3d.collision.gjk import gjk_epa
from noeira.physics3d.constants import GEOM_MESH
from noeira.physics3d.gpu.constants import mesh_max_edge
from noeira.physics3d.model.mesh_inertia import MeshInertia

comptime NV: Int = 8192
comptime NE: Int = mesh_max_edge(NV)
# Poses sweep separation AND orientation together, so the pair passes through
# near-parallel-face configurations — which is where a greedy walk plus a
# stopping test in the noise goes wrong. A handful of axis poses finds nothing.
comptime NP: Int = 256
# Below this, float64 considers the pair genuinely apart.
comptime APART: Float64 = 1e-3


def _sweep[D: DType]() raises -> List[Float64]:
    """Distance between two real hulls over `NP` poses, at one dtype."""
    var mv = List[Scalar[D]]()
    var mva = List[Int]()
    var mvn = List[Int]()
    var nm = 0
    var mpa = List[Int]()
    var mpn = List[Int]()
    var pv = List[Int]()
    var pva = List[Int]()
    var pvn = List[Int]()
    var pn = List[Scalar[D]]()
    var pm = List[Int]()
    var pma = List[Int]()
    var pmn = List[Int]()
    var ea = List[Int]()
    var el = List[Int]()
    # ⚠ THE TRIANGLE ARENA IS A REQUIRED OUT-PARAMETER, not an option. The mesh
    # BVH (`53f0e830`) publishes its nodes into `mesh_tri` and indexes them
    # with `mesh_triadr`/`mesh_trinum`; `load_mesh_hull` fills all three. This
    # file does not read them — it sweeps GJK over two hulls — but omitting
    # them stopped it COMPILING, and a gate that does not compile is not a
    # gate: this one had been silently absent since the BVH landed.
    var mtri = List[Scalar[D]]()
    var mtriadr = List[Int]()
    var mtrinum = List[Int]()
    var mi = MeshInertia[D]()
    var b = String("noeira/envs/robots/assets/so_arm100/")
    _ = load_mesh_hull[D](
        b + "Fixed_Jaw_Collision_2.stl", mv, mva, mvn, nm, mpa, mpn, pv, pva,
        pvn, pn, pm, pma, pmn, ea, el, mtri, mtriadr, mtrinum, mi,
    )
    _ = load_mesh_hull[D](
        b + "Wrist_Pitch_Roll.stl", mv, mva, mvn, nm, mpa, mpn, pv, pva,
        pvn, pn, pm, pma, pmn, ea, el, mtri, mtriadr, mtrinum, mi,
    )

    # ⚠ HEAP, NOT `Array` — see `test_gjk_hillclimb_support.mojo`. The
    # `_ = buf^` lines below are load-bearing: Mojo destroys at LAST USE.
    var vbuf = List[Scalar[D]](length=NV * 3, fill=Scalar[D](0))
    var verts = tile[NV, 3](vbuf.unsafe_ptr().as_unsafe_any_origin().unsafe_mut_cast[True]())
    for i in range(len(mv) // 3):
        verts[i, 0] = mv[i * 3]
        verts[i, 1] = mv[i * 3 + 1]
        verts[i, 2] = mv[i * 3 + 2]
    var abuf = List[Scalar[D]](length=NV, fill=Scalar[D](-1))
    var eadr = tile[NV](abuf.unsafe_ptr().as_unsafe_any_origin().unsafe_mut_cast[True]())
    for v in range(len(ea)):
        eadr[v] = Scalar[D](ea[v])
    var ebuf = List[Scalar[D]](length=NE, fill=Scalar[D](-1))
    var edges = tile[NE](ebuf.unsafe_ptr().as_unsafe_any_origin().unsafe_mut_cast[True]())
    for k in range(len(el)):
        edges[k] = Scalar[D](el[k])

    var out = List[Float64]()
    var ws = ccd_ws_alloc[D]()
    var one = Scalar[D](1)
    var z = Scalar[D](0)
    for k in range(NP):
        var t = Float64(k) / Float64(NP)
        var gx = Scalar[D](0.02 + 0.30 * t)
        var ang = 6.2831853 * t
        var qw = Scalar[D](cos(ang * 0.5))
        var qz = Scalar[D](sin(ang * 0.5))
        var r = gjk_epa[D](
            GEOM_MESH, z, z, z, z, z, z, one, z, z, z, z, z,
            verts, eadr, edges, mva[0], mvn[0],
            GEOM_MESH, gx, Scalar[D](0.01), Scalar[D](-0.02), z, z, qz, qw,
            z, z, z, z, z,
            mva[1], mvn[1],
            ws.tt_dyn["cpu", L_CCD_WS1](L_CCD_WS1()), 0,
        )
        out.append(Float64(r[0]))
    _ = vbuf^
    _ = abuf^
    _ = ebuf^
    return out^


def test_float32_never_reports_contact_where_float64_sees_a_gap() raises:
    """No pose may read as touching at float32 while float64 sees a real gap.

    ⚠ RED AT THE OLD TOLERANCE: three poses of this sweep returned -0.0 at
    float32 against float64 distances of 0.072, 0.165 and 0.169. If this fires
    again, GJK's stopping test has fallen back below the float32 noise floor
    and the solver is being handed constraint rows for pairs that never touched.
    """
    var d64 = _sweep[DType.float64]()
    var d32 = _sweep[DType.float32]()
    assert_true(
        len(d64) == NP and len(d32) == NP,
        "the sweep must produce one distance per pose in both dtypes",
    )

    # ⚠ WITHOUT THIS THE FILE IS VACUOUS. If every pose were penetrating, the
    # phantom-contact test below would have nothing to check.
    var apart = 0
    for i in range(NP):
        if d64[i] > APART:
            apart += 1
    assert_true(
        apart > NP // 4,
        "only " + String(apart) + " of " + String(NP) + " poses are separated"
        " at float64 — the sweep has drifted into contact and no longer"
        " exercises the case this file exists for",
    )

    var phantom = 0
    var worst_gap = Float64(0)
    var worst_at = -1
    for i in range(NP):
        if d64[i] > APART and d32[i] <= 0.0:
            phantom += 1
            if d64[i] > worst_gap:
                worst_gap = d64[i]
                worst_at = i
    print(
        "   poses", NP, " separated at f64", apart, " phantom contacts at f32",
        phantom,
    )
    assert_true(
        phantom == 0,
        "float32 GJK reported contact for " + String(phantom) + " pose(s)"
        " that float64 places up to " + String(worst_gap) + " m apart (worst"
        " at pose " + String(worst_at) + "). GJK's stopping test cannot fire"
        " at float32 and the routine is returning whatever it holds after"
        " GJK_MAX_ITERATIONS — which reads downstream as a real contact",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
