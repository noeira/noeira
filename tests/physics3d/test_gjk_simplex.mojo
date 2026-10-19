"""Unit gate: GJK simplex handling — `subdistance` + a flat-face end-to-end query.

Two defects lived in the routine this file was written against, both silent and
both reachable from ordinary geometry:

  1. FLAT SIMPLEX READ AS AN ENCLOSURE. The outside test was the bare product
     `dot_opp * dot_origin < 0`. For a COPLANAR quadruple every face has
     dot_opp == 0 exactly, so no face is ever flagged, and the routine falls
     through to "origin is inside the tetrahedron" — v = 0, which `gjk_epa`
     reads as penetration. GJK converges onto a planar facet whenever the
     closest feature is one (a hull face parallel to a box or cylinder cap), so
     this is the common case, not a tie: it invented contacts between geoms
     centimetres apart. It cost the SawyerReach mesh gate its meaning — that
     gate's "obj teleported into the gripper hull" pose is 15.1 mm CLEAR of the
     hull, and the contact it asserted was fabricated.

  2. ALIASED REDUCTION. Reducing to the winning face copied slot-by-slot in
     place. Face ADB is (i0,i1,i2) = (0,27,9): slot 1 is overwritten with D and
     then read back as "B", retaining {A,D,D}. A degenerate simplex stalls the
     next iteration.

⚠⚠ THE SUBJECT MOVED AND THE DEFECTS DID NOT. `_closest_point_on_simplex` is
gone; the faithful port of `subdistance` / `S1D` / `S2D` / `S3D` (Montanari et
al, ToG 2017) took its place, and it has a DIFFERENT CONTRACT: it returns
`lambda` alone and never touches the simplex, and the CALLER drops the vertices
whose coordinate is exactly zero, in order. So:

  * defect 1 is still exactly testable — "reported as an enclosure" is now
    "every one of the four `lambda` is non-zero", which is `S3D`'s
    `comp1 && comp2 && comp3 && comp4` arm;
  * defect 2 is STRUCTURALLY IMPOSSIBLE now rather than merely fixed, because
    nothing reduces the simplex inside the routine at all. What this file can
    still pin is the half that decides the reduction — WHICH coordinate comes
    back zero — and it does, because that choice is what `polytope2/3/4` seeds
    EPA from.

⚠ The closest point is `lincomb(lambda, simplex)`, computed here rather than
returned, which is also how `gjk_epa_witness` spells it.

⚠ THE SIMPLEX LIVES IN THE `ccd_ws` ROW, not on the stack — see
`ccd_workspace.mojo`. These fixtures write it through `set_sv` for the same
reason the engine does.

Run: pixi run mojo run -I . tests/physics3d/test_gjk_simplex.mojo
"""

from std.math import abs, sqrt
from layout import TileTensor
from noeira.core.tiles import RowMajor
from noeira.nn.core.tensor import TensorImpl
from noeira.physics3d.collision.ccd_workspace import L_CCD_WS1, CCD_WS_SPX
from noeira.physics3d.collision.ccd_workspace_host import ccd_ws_alloc
from noeira.physics3d.collision.gjk import gjk_epa
from noeira.physics3d.collision.gjk_support import _subdistance
from noeira.physics3d.collision.epa import sv, set_sv
from noeira.physics3d.gpu.constants import mesh_max_edge
from noeira.physics3d.constants import GEOM_CYLINDER, GEOM_MESH

comptime DT = DType.float64
comptime NMV = 8
comptime L_MV = RowMajor[NMV, 3]

# ⚠ NO EDGE GRAPH. `_support_mesh` hill-climbs the hull adjacency when it is
# present and falls back to the exhaustive scan when `edgeadr < 0`, which is
# how `fields/model.mojo` marks an unbuilt graph. These fixtures only need the
# tensors present and marked absent; the two paths are compared against each
# other in `test_gjk_hillclimb_support.mojo`.
comptime L_VEADR_G = RowMajor[NMV]
comptime L_EDGES_G = RowMajor[mesh_max_edge(NMV)]


def _no_graph_G() raises -> TensorImpl[DT]:
    var t = TensorImpl[DT].alloc(NMV)
    for i in range(NMV):
        t.data[i] = -1.0
    return t^


def _no_edges_G() raises -> TensorImpl[DT]:
    var t = TensorImpl[DT].alloc(mesh_max_edge(NMV))
    for i in range(mesh_max_edge(NMV)):
        t.data[i] = -1.0
    return t^



def _put(
    wsv: TileTensor[DT, L_CCD_WS1, MutAnyOrigin],
    slot: Int,
    x: Float64,
    y: Float64,
    z: Float64,
    tag: Float64,
) -> None:
    """Write one simplex vertex; the 6 witness slots carry `tag` so that a
    partial or aliased copy is visible, not just a wrong xyz."""
    set_sv(wsv, 0, CCD_WS_SPX, slot, 0, Scalar[DT](x))
    set_sv(wsv, 0, CCD_WS_SPX, slot, 1, Scalar[DT](y))
    set_sv(wsv, 0, CCD_WS_SPX, slot, 2, Scalar[DT](z))
    for k in range(3, 9):
        set_sv(wsv, 0, CCD_WS_SPX, slot, k, Scalar[DT](tag))


def _xyz(
    wsv: TileTensor[DT, L_CCD_WS1, MutAnyOrigin], slot: Int
) -> Tuple[Float64, Float64, Float64]:
    return (
        Float64(sv(wsv, 0, CCD_WS_SPX, slot, 0)),
        Float64(sv(wsv, 0, CCD_WS_SPX, slot, 1)),
        Float64(sv(wsv, 0, CCD_WS_SPX, slot, 2)),
    )


def _lincomb(
    wsv: TileTensor[DT, L_CCD_WS1, MutAnyOrigin],
    lam: Array[Scalar[DT], 4],
    n: Int,
) -> Tuple[Float64, Float64, Float64]:
    """`lincomb(lambda, n, simplex)` — the point `subdistance` describes."""
    var x = Float64(0)
    var y = Float64(0)
    var z = Float64(0)
    for i in range(n):
        var w = Float64(lam[i])
        x += w * Float64(sv(wsv, 0, CCD_WS_SPX, i, 0))
        y += w * Float64(sv(wsv, 0, CCD_WS_SPX, i, 1))
        z += w * Float64(sv(wsv, 0, CCD_WS_SPX, i, 2))
    return (x, y, z)


def _nnz(lam: Array[Scalar[DT], 4], n: Int) -> Int:
    """How many coordinates survive the caller's `lambda[i] != 0` compaction."""
    var c = 0
    for i in range(n):
        if lam[i] != Scalar[DT](0):
            c += 1
    return c


def test_flat_simplex_is_not_an_enclosure() raises:
    """Four COPLANAR points at z = 0.5 cannot enclose the origin."""
    print("=== flat simplex is not an enclosure ===")
    var ws = ccd_ws_alloc[DT]()
    var wsv = ws.tt_dyn["cpu", L_CCD_WS1](L_CCD_WS1())
    _put(wsv, 0, -1.0, -1.0, 0.5, 100.0)
    _put(wsv, 1, 1.0, -1.0, 0.5, 200.0)
    _put(wsv, 2, 1.0, 1.0, 0.5, 300.0)
    _put(wsv, 3, -1.0, 1.0, 0.5, 400.0)

    var lam = _subdistance[DT](wsv, 0, CCD_WS_SPX, 4)
    var n = _nnz(lam, 4)
    var v = _lincomb(wsv, lam, 4)
    print("  surviving coords =", n, " v =", v[0], v[1], v[2])
    if n == 4:
        raise Error(
            "flat simplex reported as enclosing the origin — every `lambda` is"
            " non-zero, so the caller keeps all four vertices and `gjk_epa`"
            " reads this as penetration"
        )
    # Closest point is the origin's projection onto the plane: (0, 0, 0.5).
    if abs(v[0]) > 1e-12 or abs(v[1]) > 1e-12 or abs(v[2] - 0.5) > 1e-12:
        raise Error("closest point on the z=0.5 plane should be (0,0,0.5)")
    print("  PASS")


def test_true_enclosure_still_detected() raises:
    """A tetrahedron that genuinely contains the origin must still report it —
    the flat-simplex fix must not swing the other way."""
    print("=== true enclosure still detected ===")
    var ws = ccd_ws_alloc[DT]()
    var wsv = ws.tt_dyn["cpu", L_CCD_WS1](L_CCD_WS1())
    # Regular-ish tetra straddling the origin.
    _put(wsv, 0, 1.0, 1.0, 1.0, 100.0)
    _put(wsv, 1, 1.0, -1.0, -1.0, 200.0)
    _put(wsv, 2, -1.0, 1.0, -1.0, 300.0)
    _put(wsv, 3, -1.0, -1.0, 1.0, 400.0)

    var lam = _subdistance[DT](wsv, 0, CCD_WS_SPX, 4)
    var n = _nnz(lam, 4)
    var v = _lincomb(wsv, lam, 4)
    print("  surviving coords =", n, " v =", v[0], v[1], v[2])
    if n != 4:
        raise Error("origin IS inside this tetrahedron — enclosure missed")
    # `S3D`'s all-positive arm is the enclosure, and it puts the origin itself
    # at the linear combination.
    if abs(v[0]) > 1e-12 or abs(v[1]) > 1e-12 or abs(v[2]) > 1e-12:
        raise Error("an enclosing simplex must give lincomb == origin")
    print("  PASS")


def test_face_adb_reduction_keeps_a_d_b() raises:
    """Origin outside face ADB only; the dropped coordinate must be C's.

    A=(1,0,0) D=(0,1,0) B=(0,0,1) span x+y+z=1; the origin is at 0 and the
    opposite vertex C=(1,1,1) at 3, so ADB is the one face the origin is
    outside of. `subdistance` must return `lambda` with C's slot EXACTLY zero
    and the other three non-zero, because the caller's compaction is
    `lambda[i] != 0` and nothing else — which vertex it drops decides what
    `polytope2/3/4` seeds EPA from.

    ⚠ THE OLD ALIASING DEFECT CANNOT RECUR HERE. It lived in an in-place
    slot-by-slot reduction that this routine does not do at all: `subdistance`
    reads the simplex and returns `lambda`. The compaction it feeds copies
    FORWARD with `keep <= i`, so a slot is never read after being written.
    """
    print("=== face-ADB reduction keeps {A, D, B} ===")
    var ws = ccd_ws_alloc[DT]()
    var wsv = ws.tt_dyn["cpu", L_CCD_WS1](L_CCD_WS1())
    _put(wsv, 0, 1.0, 0.0, 0.0, 100.0)  # A
    _put(wsv, 1, 0.0, 0.0, 1.0, 200.0)  # B
    _put(wsv, 2, 1.0, 1.0, 1.0, 300.0)  # C
    _put(wsv, 3, 0.0, 1.0, 0.0, 400.0)  # D

    var lam = _subdistance[DT](wsv, 0, CCD_WS_SPX, 4)
    print("  lambda =", Float64(lam[0]), Float64(lam[1]), Float64(lam[2]),
          Float64(lam[3]))
    if _nnz(lam, 4) != 3:
        raise Error("expected a reduction to the ADB face (three non-zero)")
    if lam[2] != Scalar[DT](0):
        raise Error(
            "the origin is outside face ADB, so C — slot 2 — is the vertex the"
            " caller must drop"
        )
    # v = projection of the origin onto x+y+z=1.
    var v = _lincomb(wsv, lam, 4)
    if (
        abs(v[0] - 1.0 / 3.0) > 1e-12
        or abs(v[1] - 1.0 / 3.0) > 1e-12
        or abs(v[2] - 1.0 / 3.0) > 1e-12
    ):
        raise Error("closest point on ADB should be (1/3, 1/3, 1/3)")

    # The caller's compaction, spelled out — this is `gjk_epa_witness`'s loop.
    var keep = 0
    for i in range(4):
        if lam[i] == Scalar[DT](0):
            continue
        if keep != i:
            for c in range(9):
                set_sv(wsv, 0, CCD_WS_SPX, keep, c,
                       sv(wsv, 0, CCD_WS_SPX, i, c))
        keep += 1
    var v0 = _xyz(wsv, 0)
    var v1 = _xyz(wsv, 1)
    var v2 = _xyz(wsv, 2)
    print("  kept:", v0[0], v0[1], v0[2], "|", v1[0], v1[1], v1[2],
          "|", v2[0], v2[1], v2[2])
    # In order: A (slot 0), B (slot 1), D (slot 3) -> slots 0, 1, 2.
    if v0[0] != 1.0 or v1[2] != 1.0 or v2[1] != 1.0:
        raise Error("compacted simplex is not {A, B, D} in order")
    # Witness tags must travel with their vertex.
    if (
        Float64(sv(wsv, 0, CCD_WS_SPX, 0, 3)) != 100.0
        or Float64(sv(wsv, 0, CCD_WS_SPX, 1, 3)) != 200.0
        or Float64(sv(wsv, 0, CCD_WS_SPX, 2, 3)) != 400.0
    ):
        raise Error("witness points did not follow their vertices")
    print("  PASS")


def _plate(z: Float64) raises -> TensorImpl[DT]:
    """A flat 4-corner plate at height `z` — 4 coplanar mesh verts."""
    var t = TensorImpl[DT].alloc(NMV * 3)
    var xs = [-0.5, 0.5, 0.5, -0.5]
    var ys = [-0.5, -0.5, 0.5, 0.5]
    for i in range(4):
        t.data[i * 3 + 0] = xs[i]
        t.data[i * 3 + 1] = ys[i]
        t.data[i * 3 + 2] = z
    return t^


def test_flat_face_query_reports_exact_gap() raises:
    """End-to-end: a cylinder cap coaxial over a flat plate, gap exactly 0.7.

    Parallel flat features are what drive GJK onto a coplanar simplex, and the
    cap-over-plate arrangement is the synthetic twin of the SawyerReach query
    that exposed this (cylinder obj vs a gripper-hull facet). The gap is
    analytic — cylinder centre 0.72, half-length 0.02, plate at 0 — so a wrong
    simplex shows up as a wrong NUMBER, not just a wrong sign: the old code
    returned 0.69339 here.
    """
    print("=== cylinder cap over plate reports the exact gap ===")
    var mv = _plate(0.0)
    var _ng = _no_graph_G()
    var _ne = _no_edges_G()
    var ws = ccd_ws_alloc[DT]()
    var result = gjk_epa[DT](
        GEOM_CYLINDER,
        0.0, 0.0, 0.72,
        0.0, 0.0, 0.0, 1.0,
        0.2, 0.02,  # radius, half-length
        0.0, 0.0, 0.0,
        mv.tt["cpu", NMV, 3](), _ng.tt["cpu", NMV](), _ne.tt["cpu", mesh_max_edge(NMV)](), 0, 0,
        GEOM_MESH,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
        0.0, 0.0,
        0.0, 0.0, 0.0,
        0, 4,
        ws.tt_dyn["cpu", L_CCD_WS1](L_CCD_WS1()), 0,
    )
    var dist = Float64(result[0])
    print("  dist =", dist, "(expected 0.7)")
    if dist < 0:
        raise Error(
            "separated cylinder/plate reported as penetrating — flat simplex"
            " read as an enclosure"
        )
    if abs(dist - 0.7) > 1e-9:
        raise Error("expected a 0.7 gap, got " + String(dist))
    print("  PASS")


def main() raises:
    test_flat_simplex_is_not_an_enclosure()
    test_true_enclosure_still_detected()
    test_face_adb_reduction_keeps_a_d_b()
    test_flat_face_query_reports_exact_gap()
    print("test_gjk_simplex: ALL PASS")
