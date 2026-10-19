"""The hill-climb support point must EQUAL the linear scan's, on real hulls.

    pixi run mojo run -I . tests/physics3d/test_gjk_hillclimb_support.mojo

⚠ RUN FROM THE REPO ROOT — the fixture meshes are addressed by repo-root
relative path.

WHY THIS EXISTS. `_support_mesh` is the hottest function in mesh collision —
GJK and EPA call it 10-30 times per geom pair per step — and it used to scan
every hull vertex. It now walks the hull's edge graph instead, greedily
stepping to whichever neighbour scores higher on `dir`, RESUMING FROM THE
VERTEX THE PREVIOUS CALL LANDED ON. On a CONVEX hull the dot product has
exactly one local maximum, so the walk is EXACT from any start rather than
approximate, and this file is what holds that claim to account:

                          scan     + climb   + warm start
    SO-ARM101 33 076 v   76.03 ms  12.79 ms    4.74 ms   ( 13 -> 211 Hz)
    SO-ARM100  2 551 v   11.95 ms   5.35 ms    2.82 ms   ( 84 -> 354 Hz)

⚠⚠ A WRONG SUPPORT POINT DOES NOT CRASH. It yields a contact in a plausible
but wrong place, or a slightly shallow penetration, and everything downstream
keeps running. Convergence checks cannot see it either — GJK converges happily
against a support function that is consistently wrong. Only comparing against
the exhaustive answer catches it, which is what this does.

HOW THE TWO PATHS ARE SELECTED WITHOUT A FLAG. `_support_mesh` falls back to
the scan when `mesh_vert_edgeadr[vert_adr] < 0`, which is how `fields/model.
mojo` marks a mesh whose graph was never built. Filling that tensor with -1
therefore exercises the scan on the SAME hull and the SAME directions, with no
test-only branch in the shipping code — a flag would have been a second thing
to keep honest.

⚠ THE FIXTURES ARE REAL SCANNED HULLS, NOT A CUBE. Hill climbing is trivially
correct on a cube and on any tidy convex cloud; where it can go wrong is
near-coplanar facets and long thin geometry, which is what CAD exports are made
of. `test_mesh_support.mojo` already covers the cube.

⚠ THE DIRECTIONS ARE DETERMINISTIC AND COVER THE SPHERE. A handful of axis
directions would miss exactly the wedges where a greedy walk could stall; the
sweep below is a Fibonacci lattice, which spreads without needing an RNG.
"""

from std.math import sqrt, cos, sin, pi, acos
from noeira.core.tiles import tile
from std.testing import assert_true, TestSuite

from noeira.physics3d.collision.convex_hull import load_mesh_hull
from noeira.physics3d.collision.gjk import _support_mesh
from noeira.physics3d.gpu.constants import mesh_max_edge
from noeira.physics3d.model.mesh_inertia import MeshInertia

comptime D = DType.float64
# Big enough for the largest fixture below; the tensors are comptime-shaped.
comptime NV: Int = 8192
comptime NE: Int = mesh_max_edge(NV)
comptime NDIR: Int = 512


def _check(name: String, path: String) raises:
    """Build one mesh's hull, then compare both support paths over the sphere."""
    var mesh_vert = List[Scalar[D]]()
    var mesh_vertadr = List[Int]()
    var mesh_vertnum = List[Int]()
    var num_meshes = 0
    var mesh_polyadr = List[Int]()
    var mesh_polynum = List[Int]()
    var poly_vert = List[Int]()
    var poly_vertadr = List[Int]()
    var poly_vertnum = List[Int]()
    var poly_normal = List[Scalar[D]]()
    var polymap = List[Int]()
    var polymap_adr = List[Int]()
    var polymap_num = List[Int]()
    var edge_adr = List[Int]()
    var edge_list = List[Int]()
    # ⚠ THE TRIANGLE ARENA IS A REQUIRED OUT-PARAMETER, not an option. The mesh
    # BVH (`53f0e830`) publishes its nodes into `mesh_tri` and indexes them
    # with `mesh_triadr`/`mesh_trinum`; `load_mesh_hull` fills all three. This
    # file does not read them — it only needs the hull — but omitting them
    # stopped it COMPILING, and a gate that does not compile is not a gate.
    var mesh_tri = List[Scalar[D]]()
    var mesh_triadr = List[Int]()
    var mesh_trinum = List[Int]()
    var mi = MeshInertia[D]()
    _ = load_mesh_hull[D](
        path, mesh_vert, mesh_vertadr, mesh_vertnum, num_meshes,
        mesh_polyadr, mesh_polynum, poly_vert, poly_vertadr, poly_vertnum,
        poly_normal, polymap, polymap_adr, polymap_num, edge_adr, edge_list,
        mesh_tri, mesh_triadr, mesh_trinum,
        mi,
    )
    var nverts = mesh_vertnum[0]
    assert_true(
        nverts <= NV,
        name + ": hull has " + String(nverts) + " vertices, over the "
        + String(NV) + " the fixture tensors are sized for",
    )
    assert_true(
        len(edge_list) <= NE,
        name + ": edge graph has " + String(len(edge_list)) + " slots, over "
        + String(NE),
    )

    # Pack into the tensors `_support_mesh` reads.
    # ⚠ HEAP, NOT `Array`. NV*3 float64 is ~200 KB and the edge buffer
    # more; on the stack that is an overflow, not a slow test.
    # ⚠⚠ THE LISTS MUST OUTLIVE THE TENSORS — Mojo destroys at LAST USE, so the
    # `_ = ...` at the end of this function is load-bearing, not tidy-up.
    var vbuf = List[Scalar[D]](length=NV * 3, fill=Scalar[D](0))
    var verts = tile[NV, 3](vbuf.unsafe_ptr().as_unsafe_any_origin().unsafe_mut_cast[True]())
    for i in range(nverts):
        verts[i, 0] = mesh_vert[i * 3 + 0]
        verts[i, 1] = mesh_vert[i * 3 + 1]
        verts[i, 2] = mesh_vert[i * 3 + 2]

    var abuf = List[Scalar[D]](length=NV, fill=Scalar[D](-1))
    var eadr = tile[NV](abuf.unsafe_ptr().as_unsafe_any_origin().unsafe_mut_cast[True]())
    for v in range(len(edge_adr)):
        eadr[v] = Scalar[D](edge_adr[v])

    var ebuf = List[Scalar[D]](length=NE, fill=Scalar[D](-1))
    var edges = tile[NE](ebuf.unsafe_ptr().as_unsafe_any_origin().unsafe_mut_cast[True]())
    for k in range(len(edge_list)):
        edges[k] = Scalar[D](edge_list[k])

    # ⚠ The scan arm: an all -1 `edgeadr` is exactly how an unbuilt graph is
    # marked, so this takes the fallback WITHOUT a test-only branch.
    var nbuf = List[Scalar[D]](length=NV, fill=Scalar[D](-1))
    var no_graph = tile[NV](nbuf.unsafe_ptr().as_unsafe_any_origin().unsafe_mut_cast[True]())

    # ================================================================
    # THE GRAPH IS A CONVEX POLYTOPE'S 1-SKELETON, OR THE WALK BELOW IS
    # MEANINGLESS.
    # ================================================================
    # A greedy walk provably has no local maximum on the 1-skeleton of a convex
    # polytope, so every direction assertion further down is really an
    # assertion ABOUT THIS GRAPH. Checking the graph directly is what turns a
    # stall from "the walk got unlucky on 13 of 512 directions" into "the hull
    # is not a polytope", which is the statement that names the defect.
    #
    # For a triangulated polytope Euler gives `E = 3V - 6` exactly, and every
    # vertex of a 3-polytope has degree at least 3. ⚠ THESE ARE COMPUTED FROM
    # `edge_adr`/`edge_list`, i.e. from the adjacency the narrow phase actually
    # walks — not from the face list it was derived from, which could satisfy
    # them while the adjacency did not.
    var deg_sum = 0
    var deg_min = 1 << 30
    for v in range(nverts):
        var e0 = edge_adr[v]
        var deg = 0
        while edge_list[e0 + deg] >= 0:
            deg += 1
        deg_sum += deg
        if deg < deg_min:
            deg_min = deg
    var nedge = deg_sum // 2
    assert_true(
        nedge == 3 * nverts - 6,
        name + ": the hull's adjacency has " + String(nedge) + " edges over "
        + String(nverts) + " vertices, and a triangulated convex polytope has"
        " exactly 3V - 6 = " + String(3 * nverts - 6) + ". More means faces"
        " sharing an edge with a third face — a non-manifold stitch; fewer"
        " means a hole. Either way the surface is not a polytope and a greedy"
        " support walk can strand on it",
    )
    assert_true(
        deg_min >= 3,
        name + ": a hull vertex has degree " + String(deg_min) + ". Every"
        " vertex of a 3-polytope meets at least three edges; a lower degree is"
        " a vertex the walk can enter and not leave",
    )
    print("   ", name, " V", nverts, " E", nedge, " 3V-6", 3 * nverts - 6,
          " min degree", deg_min)

    # A non-trivial pose, so the quaternion rotation is exercised too rather
    # than cancelling out of both arms identically.
    var qx = Scalar[D](0.1830127)
    var qy = Scalar[D](0.3535534)
    var qz = Scalar[D](0.1830127)
    var qw = Scalar[D](0.9014649)
    var px = Scalar[D](0.31)
    var py = Scalar[D](-0.17)
    var pz = Scalar[D](0.44)

    var worst = Scalar[D](0)
    for k in range(NDIR):
        # Fibonacci lattice on the sphere — even coverage, no RNG.
        var t = (Scalar[D](k) + Scalar[D](0.5)) / Scalar[D](NDIR)
        var z = Scalar[D](1) - Scalar[D](2) * t
        var r = sqrt(Scalar[D](1) - z * z)
        var phi = Scalar[D](k) * Scalar[D](2.399963229728653)
        var dx = r * cos(phi)
        var dy = r * sin(phi)

        # ⚠ A FRESH SEED PER DIRECTION, so this arm stays the COLD-START
        # comparison it has always been; the warm-started walk is gated
        # separately below, where a carried seed is the point.
        var w_cold = -1
        var w_none = -1
        var hc = _support_mesh[D](
            dx, dy, z, px, py, pz, qx, qy, qz, qw,
            verts, eadr, edges, 0, nverts, w_cold,
        )
        var sc = _support_mesh[D](
            dx, dy, z, px, py, pz, qx, qy, qz, qw,
            verts, no_graph, edges, 0, nverts, w_none,
        )
        var e0 = hc[0] - sc[0]
        var e1 = hc[1] - sc[1]
        var e2 = hc[2] - sc[2]
        var err = sqrt(e0 * e0 + e1 * e1 + e2 * e2)
        if err > worst:
            worst = err
        assert_true(
            err == Scalar[D](0),
            name + ": direction " + String(k) + " (" + String(dx) + ", "
            + String(dy) + ", " + String(z) + ") — hill climb returned ("
            + String(hc[0]) + ", " + String(hc[1]) + ", " + String(hc[2])
            + ") but the exhaustive scan returns (" + String(sc[0]) + ", "
            + String(sc[1]) + ", " + String(sc[2]) + "). The walk stalled at a"
            " vertex that is not the extreme one; on a convex hull that can"
            " only mean the edge graph is not the hull's full adjacency",
        )
    print("  ", name, " hull", nverts, " edges", len(edge_list),
          " directions", NDIR, " worst error", worst)

    # ================================================================
    # WARM START. Everything above starts each walk from scratch. The
    # optimisation that matters is `warm`: the seed carried from the previous
    # call, which is where GJK and EPA get their cheapness because consecutive
    # directions barely move. Three separate things have to hold, and NONE of
    # them is visible to the sweep above.
    # ================================================================

    # (1) A CARRIED SEED STILL FINDS THE EXTREME VERTEX. Same directions, but
    # `warm` now threads through the whole sweep, so every walk starts from a
    # vertex chosen by the PREVIOUS direction rather than from 0.
    # ⚠ COMPARED ON THE SUPPORT VALUE, NOT THE POINT. If two vertices tie
    # exactly on `dir` — a facet perpendicular to it — then both are correct
    # answers and which one the walk stops at legitimately depends on where it
    # started. The extremal VALUE is the property GJK actually relies on;
    # demanding the same vertex would be demanding an accident.
    var warm = -1
    var worst_warm = Scalar[D](0)
    for k in range(NDIR):
        var t = (Scalar[D](k) + Scalar[D](0.5)) / Scalar[D](NDIR)
        var z = Scalar[D](1) - Scalar[D](2) * t
        var r = sqrt(Scalar[D](1) - z * z)
        var phi = Scalar[D](k) * Scalar[D](2.399963229728653)
        var dx = r * cos(phi)
        var dy = r * sin(phi)

        var hw = _support_mesh[D](
            dx, dy, z, px, py, pz, qx, qy, qz, qw,
            verts, eadr, edges, 0, nverts, warm,
        )
        var w_none = -1
        var sc = _support_mesh[D](
            dx, dy, z, px, py, pz, qx, qy, qz, qw,
            verts, no_graph, edges, 0, nverts, w_none,
        )
        var vw = (hw[0] - px) * dx + (hw[1] - py) * dy + (hw[2] - pz) * z
        var vs = (sc[0] - px) * dx + (sc[1] - py) * dy + (sc[2] - pz) * z
        var derr = vw - vs
        if derr < Scalar[D](0):
            derr = -derr
        if derr > worst_warm:
            worst_warm = derr
        assert_true(
            derr <= Scalar[D](1e-12),
            name + ": direction " + String(k) + " — warm-started from vertex "
            + String(warm) + " the walk reaches support value " + String(vw)
            + " but the exhaustive scan reaches " + String(vs) + ". A hill"
            " climb on a convex hull is supposed to converge to the extreme"
            " vertex from ANY starting vertex; if a carried seed can strand it,"
            " the seed is being written back as something that is not a vertex"
            " of THIS mesh, or the graph is not the hull's full adjacency",
        )

    # (2) THE SEED IS ACTUALLY WRITTEN BACK, AND NAMES THE VERTEX RETURNED.
    # ⚠⚠ THIS IS THE ONLY CHECK THAT CAN SEE THE OPTIMISATION AT ALL. Delete
    # `warm = imax` from `_support_mesh` and every other assertion in this file
    # stays green — the walk simply restarts from vertex 0 forever and is
    # correct but slow, which is exactly the failure a correctness gate cannot
    # notice. Under an IDENTITY pose the returned point IS the local vertex, so
    # `verts[warm]` can be compared to it bit-for-bit with no rotation rounding
    # in the way.
    var idq = Scalar[D](0)
    var carried = -1
    for k in range(NDIR):
        var t = (Scalar[D](k) + Scalar[D](0.5)) / Scalar[D](NDIR)
        var z = Scalar[D](1) - Scalar[D](2) * t
        var r = sqrt(Scalar[D](1) - z * z)
        var phi = Scalar[D](k) * Scalar[D](2.399963229728653)
        var dx = r * cos(phi)
        var dy = r * sin(phi)
        var hp = _support_mesh[D](
            dx, dy, z, idq, idq, idq, idq, idq, idq, Scalar[D](1),
            verts, eadr, edges, 0, nverts, carried,
        )
        assert_true(
            carried >= 0 and carried < nverts,
            name + ": after a support call the carried index is "
            + String(carried) + ", outside [0, " + String(nverts) + ")",
        )
        assert_true(
            hp[0] == rebind[Scalar[D]](verts[carried, 0])
            and hp[1] == rebind[Scalar[D]](verts[carried, 1])
            and hp[2] == rebind[Scalar[D]](verts[carried, 2]),
            name + ": direction " + String(k) + " returned ("
            + String(hp[0]) + ", " + String(hp[1]) + ", " + String(hp[2])
            + ") but the carried seed " + String(carried) + " names vertex ("
            + String(verts[carried, 0]) + ", " + String(verts[carried, 1])
            + ", " + String(verts[carried, 2]) + "). The write-back is not the"
            " vertex the walk landed on, so the next call resumes from the"
            " wrong place — correct, but paying full graph diameter every time",
        )

    # (3) ANY SEED, INCLUDING A NONSENSICAL ONE, REACHES THE SAME EXTREME.
    # This is what licenses the guard in `_support_mesh` to clamp instead of
    # trusting the caller, and it is the reason a crossed `warm1`/`warm2` costs
    # only speed. The out-of-range entries are the sharp ones: without the
    # guard they index past this mesh into whatever the model-wide vertex slab
    # holds next (here, zero-filled padding), and the walk terminates
    # immediately on an absent edge list and returns that.
    var seeds = List[Int]()
    seeds.append(-1)
    seeds.append(0)
    seeds.append(1)
    seeds.append(nverts // 4)
    seeds.append(nverts // 2)
    seeds.append(nverts - 1)
    seeds.append(nverts)          # one past the end
    seeds.append(nverts + 1000)   # deep into another mesh's vertices
    seeds.append(-7)              # a stale sentinel that is not -1
    var sdx = Scalar[D](0.37139068)
    var sdy = Scalar[D](-0.55708601)
    var sdz = Scalar[D](0.74278135)
    var w_ref = -1
    var scan_pt = _support_mesh[D](
        sdx, sdy, sdz, idq, idq, idq, idq, idq, idq, Scalar[D](1),
        verts, no_graph, edges, 0, nverts, w_ref,
    )
    var vref = scan_pt[0] * sdx + scan_pt[1] * sdy + scan_pt[2] * sdz
    for i in range(len(seeds)):
        var seed = seeds[i]
        var got = _support_mesh[D](
            sdx, sdy, sdz, idq, idq, idq, idq, idq, idq, Scalar[D](1),
            verts, eadr, edges, 0, nverts, seed,
        )
        var vgot = got[0] * sdx + got[1] * sdy + got[2] * sdz
        assert_true(
            vgot == vref,
            name + ": seeded at " + String(seeds[i]) + " the walk reaches "
            + String(vgot) + ", the exhaustive scan reaches " + String(vref)
            + ". An in-range seed that strands the walk means the graph is"
            " incomplete; an out-of-range one means the seed is being trusted"
            " unguarded and is reading vertices that belong to another mesh",
        )
    print("   ", name, " warm sweep worst |dvalue|", worst_warm,
          " — seeds tried", len(seeds), " all reach", vref)

    # Keep the backing storage alive past the last tensor read — see above.
    _ = vbuf^
    _ = abuf^
    _ = ebuf^
    _ = nbuf^


def test_small_collision_hull() raises:
    """A hand-authored collision mesh — few vertices, many coplanar facets.

    Coplanarity is where a greedy walk can find a plateau, so the SMALL fixture
    is the sharp one here, not the easy one.
    """
    _check(
        "Fixed_Jaw_Collision_2",
        "noeira/envs/robots/assets/so_arm100/Fixed_Jaw_Collision_2.stl",
    )


def test_scanned_hull() raises:
    """A full scanned part — the shape GJK actually walks in the viewer."""
    _check(
        "Wrist_Pitch_Roll  ",
        "noeira/envs/robots/assets/so_arm100/Wrist_Pitch_Roll.stl",
    )


def test_long_thin_hull() raises:
    """SO-ARM101's under-arm: long and thin, so the graph diameter is large and
    the walk takes many steps. If a step budget were ever too tight, this is
    the fixture that would expose it."""
    _check(
        "under_arm_so101_v1",
        "noeira/envs/robots/assets/so_arm101/under_arm_so101_v1.stl",
    )


# =============================================================================
# THE MESHES THAT USED TO STALL
# =============================================================================
# ⚠⚠ EVERY FIXTURE BELOW IS RED ON THE HULL THIS REPLACED, and that is the only
# reason they are here — the four above passed both before and after, so they
# could not have caught the defect. Measured on the shipped build (one shared
# 1e-9 tolerance for mesh degeneracy AND face visibility, an UNDIRECTED
# horizon, and winding repaired at the end from an interior point), over 256
# directions with a cold seed:
#
#   robotiq_2f85 base_mount              630 V   18 non-manifold edges  13 stalls
#   robotiq_2f85_v4 fts300_base         3161 V  826 non-manifold edges  15 stalls
#   ms_human_700 waterbottle             615 V   30 non-manifold edges 124 stalls
#   low_cost_robot_arm elbow_to_wrist    638 V   19 non-manifold edges  71 stalls
#
# They are CAD parts, which is the whole point: a machined boss meshed at 0.1 mm
# on a 100 mm body produces sliver triangles by the thousand, and a sliver has
# no reliable normal. Hand-authored fixtures do not have them.


def test_sliver_heavy_gripper_mount() raises:
    """The robotiq_2f85 base mount — 10 899 input vertices, ~620 on the hull.

    MuJoCo's own qhull graph for this mesh is 620 vertices and 1236 faces,
    which is `2V - 4` exactly; ours came out with two extra edges and a walk
    that lost 28.9 mm of support depth on 13 directions."""
    _check(
        "base_mount        ",
        "references/mujoco_menagerie-main/robotiq_2f85/assets/base_mount.stl",
    )


def test_force_torque_sensor_shell() raises:
    """The worst manifoldness case in Menagerie: 826 malformed edges.

    A thin cylindrical shell meshed at CAD resolution, so nearly every triangle
    on the rim is a sliver."""
    _check(
        "fts300_base       ",
        "references/mujoco_menagerie-main/robotiq_2f85_v4/assets/"
        "robotiq_fts300_base.stl",
    )


def test_revolved_surface_waterbottle() raises:
    """The worst STALL case: 124 of 256 directions lost the support vertex.

    A surface of revolution — every ring of the lathe is a band of near-coplanar
    triangles, which is exactly the configuration a shared visibility tolerance
    splits into disconnected visible sets."""
    _check(
        "waterbottle       ",
        "references/mujoco_menagerie-main/ms_human_700/assets/geometry/"
        "waterbottle/waterbottle.stl",
    )


def test_below_hillclimb_min_uses_the_scan() raises:
    """Under `_HILLCLIMB_MIN = 10` vertices MuJoCo keeps the scan, and so do
    we. Pins that the threshold branch does not change the answer — a mesh in
    that range must give the same point either way."""
    # ⚠ HEAP, NOT `Array`. NV*3 float64 is ~200 KB and the edge buffer
    # more; on the stack that is an overflow, not a slow test.
    # ⚠⚠ THE LISTS MUST OUTLIVE THE TENSORS — Mojo destroys at LAST USE, so the
    # `_ = ...` at the end of this function is load-bearing, not tidy-up.
    var vbuf = List[Scalar[D]](length=NV * 3, fill=Scalar[D](0))
    var verts = tile[NV, 3](vbuf.unsafe_ptr().as_unsafe_any_origin().unsafe_mut_cast[True]())
    var n = 0
    for sx in range(2):
        for sy in range(2):
            for sz in range(2):
                verts[n, 0] = Scalar[D](sx) - Scalar[D](0.5)
                verts[n, 1] = Scalar[D](sy) - Scalar[D](0.5)
                verts[n, 2] = Scalar[D](sz) - Scalar[D](0.5)
                n += 1
    var abuf = List[Scalar[D]](length=NV, fill=Scalar[D](-1))
    var eadr = tile[NV](abuf.unsafe_ptr().as_unsafe_any_origin().unsafe_mut_cast[True]())
    var ebuf = List[Scalar[D]](length=NE, fill=Scalar[D](-1))
    var edges = tile[NE](ebuf.unsafe_ptr().as_unsafe_any_origin().unsafe_mut_cast[True]())
    var w = -1
    var s = _support_mesh[D](
        Scalar[D](0), Scalar[D](0), Scalar[D](1),
        Scalar[D](0), Scalar[D](0), Scalar[D](0),
        Scalar[D](0), Scalar[D](0), Scalar[D](0), Scalar[D](1),
        verts, eadr, edges, 0, 8, w,
    )
    print("   cube (8 verts, below the threshold) +z support z =", s[2])
    assert_true(
        s[2] == Scalar[D](0.5),
        "an 8-vertex cube's +z support should be z = 0.5, got " + String(s[2]),
    )
    _ = vbuf^
    _ = abuf^
    _ = ebuf^


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
