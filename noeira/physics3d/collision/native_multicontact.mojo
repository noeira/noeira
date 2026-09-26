"""Native multi-point contact for {BOX, MESH} x MESH — MuJoCo's `multicontact`.

Port of `references/mujoco-3.11.0/src/engine/engine_collision_gjk.c:2111`
(`multicontact`) and the helpers it calls: `polygonClip` (:1620),
`polygonQuad` (:1533), `boxNormals` (:1906), `boxEdgeNormals` (:1961),
`boxFace` (:2000), `meshNormals` (:1769), `meshEdgeNormals` (:1834),
`meshFace` (:2050), `alignedFaces` (:2067), `alignedFaceEdge` (:2083),
`simplexDim` (:2099).

WHAT IT DOES. EPA hands back ONE point plus the face of the Minkowski polytope
it came from. That face's three vertices are each the difference of a support
point on geom1 and one on geom2; how many DISTINCT support points each geom
contributed says whether that geom is touching with a vertex (1), an edge (2)
or a face (3). Given that, the actual face polygon is recovered on each geom
and the two are clipped against each other. A cube resting on a mesh goes from
1 point to 4.

⚠ WHICH PAIRS. `maxContacts` (`engine_collision_convex.c:843`) returns 4 when
BOTH geoms are box-or-mesh, and `mjc_Convex` then takes an EARLY RETURN past
the perturbation loop into this. BOX x BOX never arrives — it dispatches to
`mjc_BoxBox` (`engine_collision_driver.c:53`), which is ported separately and
exactly. MESH x {CYLINDER, CAPSULE} never arrives either — only one of those is
box-or-mesh, so `maxContacts` is 1 and they take the perturbation loop in
`multi_ccd.mojo`. So the set here is BOX x MESH, MESH x BOX and MESH x MESH.

⚠ `maxContacts` RETURNS 1 IF EITHER GEOM HAS A NONZERO MARGIN, before it looks
at types at all. A margin geom therefore keeps its single point in the
reference, and the caller must not route it here.

⚠ THIS REPLACES THE EPA POINT, IT DOES NOT EXTEND IT. `status->nx` is
overwritten by `polygonClip`. When no pair of faces lines up the routine
returns without touching it and the single EPA point stands. So the caller
emits the single point only when this returns 0.

⚠ THE CONTACT NORMAL COMES FROM THE FACE, NOT FROM EPA, and that is a
measurable difference rather than a detail. `mjc_penetration` builds each row's
normal as `normalize(x1 - x2)`, and for these rows `x1 = x2 - approx_dir` with
`approx_dir` the exact FACE normal scaled by the depth — so the normal is the
polygon's, to machine precision. Measured before this landed: our EPA normal
differed from MuJoCo's by up to 1.6e-3 on every mesh group of
`test_mesh_manifold_vs_mujoco`, and that gap is this.
"""

from std.math import sqrt, abs

from layout import Layout, LayoutTensor

from ..constants import GEOM_BOX, GEOM_MESH
from ..kinematics.quat_math import quat_rotate, quat_rotate_inverse
from ..gpu.constants import (
    CONTACT_SIZE,
    CONTACT_IDX_BODY_A,
    CONTACT_IDX_BODY_B,
    CONTACT_IDX_POS_X,
    CONTACT_IDX_POS_Y,
    CONTACT_IDX_POS_Z,
    CONTACT_IDX_NX,
    CONTACT_IDX_NY,
    CONTACT_IDX_NZ,
    CONTACT_IDX_DIST,
    CONTACT_IDX_INCLUDEMARGIN,
    CONTACT_IDX_FRICTION,
    CONTACT_IDX_FRICTION_SPIN,
    CONTACT_IDX_FRICTION_ROLL,
    CONTACT_IDX_CONDIM,
    MODEL_MESH_POLY_SIZE,
    MESH_POLY_IDX_VERTADR,
    MESH_POLY_IDX_VERTNUM,
    MESH_POLY_IDX_NX,
    MESH_POLY_IDX_NY,
    MESH_POLY_IDX_NZ,
)
from ..fields import DimsLike
from .ccd_workspace import (
    MC_MAX_POLYVERT,
    MC_MAX_DEG,
    MC_CLIP_CAP,
    MC_WS_FACE1,
    MC_WS_FACE2,
    MC_WS_OUT,
    MC_WS_POLY,
    MC_WS_CLIPPED,
    MC_WS_PN,
    MC_WS_PD,
    MC_WS_ODIST,
)

# `mjFACE_TOL` / `mjEDGE_TOL` (`engine_collision_gjk.h:41-43`, **3.12**).
# FACE_TOL is a cosine — two face normals count as opposed when their dot is
# below -0.996, i.e. within 5.1 deg of antiparallel — and EDGE_TOL is the
# matching sine for the perpendicularity test.
#
# ⚠⚠ THESE MOVED IN MuJoCo 3.12 (commit 2444defc, "multiccd with arbitrarily
# large meshes"): 3.10 and 3.11 carry 0.99999872 / 0.00159999931 (0.09 deg),
# and this file carried those until 2026-09-08. The runtime pixi ships is
# 3.12, so at the old values a pair of faces 0.1-5 deg off antiparallel got
# a four-point manifold from the reference and the single EPA point from us.
# Found on the Unitree G1: torso_link against right_shoulder_yaw_link during
# an arm swing, MuJoCo 4 contacts / ours 1, 2.5e-3 rad off after one control
# step where the floor contacts on the same body were exact. Nothing on the
# Menagerie board ever put two faces in that band, which is why 85/85 held.
# `docs/PHYSICS3D_CONTACT_FIDELITY_REASSEMBLE5.md` §11.8.10 tightened the OLD
# value as a control and is superseded here.
comptime MC_FACE_TOL: Float64 = 0.996
comptime MC_EDGE_TOL: Float64 = 0.0888

# ⚠ `MC_MAX_POLYVERT` / `MC_MAX_DEG` / `MC_CLIP_CAP` NOW LIVE IN
# `ccd_workspace.mojo`, beside the workspace row they size. MuJoCo carries the
# equivalents (`npolygonmax` / `nmeshdegmax`) as RUNTIME MODEL FIELDS and has
# no cap at all; ours are comptime because the row offsets have to be.
#
# ⚠⚠ EXCEEDING EITHER IS A LOST MANIFOLD, AND IT USED TO BE SILENT. `_mesh_face`
# returns 0 past the cap, which is this routine's own "the features do not line
# up" answer, so the caller emitted the single EPA point — the reference's own
# fallback, reached for a reason the reference does not have. At the previous
# values of 16 and 16 that hit 47 of Menagerie's 59 mesh-bearing scenes on
# polygon width and 39 on vertex degree; the tree's worst are robotiq_2f85's
# 144-vertex face and flexiv_rizon4's degree-47 vertex. kinova_gen3, which
# interpenetrates two 31- and 29-vertex faces at its own keyframe, was
# 4.4e-02 from MuJoCo at step one and 5.7e-12 once they fit.
#
# ⚠ METAL USED TO SET THE CEILING AND NO LONGER DOES. These arrays were
# per-thread stack in the collision kernel, and bisection put the largest
# compiling pair at (56, 48). The `MC_MAX_POLYVERT`-sized ones now live in
# `d.ccd_ws` — MuJoCo's own storage class for exactly this — so the width axis
# is free. The `MC_MAX_DEG`-sized ones (`n1`/`n2`/`idx1`/`idx2`/`endverts`,
# ~4.2 KB) stay on the stack: 48 already covers every vertex degree in the
# tree, so that axis never needed unlocking.
#
# ⚠ THE COST IS THE WORK, NOT THE BUFFERS. A pair that never reaches the
# manifold path never touches them: measured across a 10x raise, barkour
# 106.1 -> 105.7 us/step and spot 115.7 -> 115.5, interleaved, min of 3.
# kinova pays 55.4 -> 70.8 because it now clips a manifold and solves four
# rows.
#
# ⚠ AN EARLIER COMMENT HERE CLAIMED THESE WERE "CHECKED AT MODEL BUILD". They
# were not. `parser/fields_build.mojo` now prints the widest polygon and the
# busiest vertex whenever a model passes either — a print and not a raise,
# because unlike a missing hull this degrades to a working simulation and
# refusing to load a third-party model over it would be worse.
# See `tests/physics3d/test_multicontact_polygon_caps_vs_mujoco.mojo`.

# ⚠⚠ ON since `8b55a978`, and exact. The kill switch stays for ABLATION.
#
# It landed OFF (`aca8c509`) so the code was preserved and kept COMPILING — a
# branch behind `comptime if False` is uncompiled code, which is how this repo
# lost a GPU path once already (`feedback_ungated_generic_is_uncompiled_code`).
# That is why the flag is read in a RUNTIME boolean and not a `comptime if`:
# the whole routine is type-checked and codegen'd either way, and flipping this
# to False for an A/B costs a rebuild, not a repair.
#
# `8b55a978` turned it on. What `test_mesh_manifold_vs_mujoco` measures with it
# ON (160 poses x 5 groups) — point counts EXACT on all five, against the ~40%
# shortfall and the |dn| ~ 2.0 reversals it showed before:
#
#     group                MuJoCo   ours   cnt!=   worst|dn|
#     mesh(cube) x box       321     321     0      5.5e-15
#     box x mesh(cube)       339     339     0      2.8e-13
#     mesh(cube) x mesh      336     336     0      6.3e-13
#     mesh(hex)  x box       333     333     0      7.8e-13
#     mesh(hex)  x mesh      329     329     0      9.8e-13
#
# The three defects behind the old numbers were a flat offset indexing a 2-D
# `mesh_polys` (polygon 3 read row 15, so every normal was garbage), our pair
# order not being `mj_collideGeoms`' `type1 <= type2` (BOX before MESH), and
# the witness pair `(x1, x2)` not swapping with the operands — `dir = x2 - x1`
# is a SIGNED input to `boxNormals2`, not just a magnitude.
#
# ⚠ ALL THREE WERE FOUND BY INSTRUMENTING THE LIVE CALL, and every a-priori
# hypothesis was wrong. If this path regresses, reach for `MC_DEBUG` below
# before the reference — three defects in this arc were "fixed" at inferred
# locations and changed nothing. See
# `feedback_confirm_the_code_under_test_actually_runs`.
comptime MC_ENABLED: Bool = True

# Debug tracing for the arc above. CPU only (a Metal kernel cannot `print`), and
# off in every committed state — it is here so the NEXT person instruments the
# live call instead of re-deriving the branch from the reference, which is what
# produced the wrong numbers in the first place.
comptime MC_DEBUG: Bool = False

# Dumps the full clipped ring before the quad pruner. This is what proved the
# manifold residual is a PRUNER TIE-BREAK and not a clip defect: MuJoCo's four
# emitted points are vertices of the very ring we compute, it just keeps a
# different four of them. Leave it here — the ring is the only thing that
# distinguishes "we clipped differently" from "we pruned differently".
comptime MC_DEBUG_RING: Bool = False


@always_inline
def _wr[
    DTYPE: DType, L_WS: Layout
](
    ws: LayoutTensor[DTYPE, L_WS, MutAnyOrigin], wrow: Int, i: Int
) -> Scalar[DTYPE]:
    """One slot of the CCD workspace row.

    `rebind` because a `LayoutTensor`'s element type is only provably
    `Scalar[DTYPE]` once its layout is concrete, and `L_WS` is a parameter
    here. Reads go through this; writes stay spelled out at the site so the
    slot being written is visible. See `ccd_workspace.mojo`.
    """
    return rebind[Scalar[DTYPE]](ws[wrow, i])


@always_inline
def _wv[
    DTYPE: DType, L_WS: Layout
](
    ws: LayoutTensor[DTYPE, L_WS, MutAnyOrigin],
    wrow: Int,
    off: Int,
    i: Int,
    k: Int,
) -> Scalar[DTYPE]:
    """Component `k` of vertex `i` of the polygon buffer at `off`."""
    return rebind[Scalar[DTYPE]](ws[wrow, off + i * 3 + k])


@always_inline
def _dot3[
    DTYPE: DType
](
    ax: Scalar[DTYPE], ay: Scalar[DTYPE], az: Scalar[DTYPE],
    bx: Scalar[DTYPE], by: Scalar[DTYPE], bz: Scalar[DTYPE],
) -> Scalar[DTYPE]:
    return ax * bx + ay * by + az * bz


@always_inline
def _same_point[
    DTYPE: DType
](
    ax: Scalar[DTYPE], ay: Scalar[DTYPE], az: Scalar[DTYPE],
    bx: Scalar[DTYPE], by: Scalar[DTYPE], bz: Scalar[DTYPE],
    scale: Scalar[DTYPE],
) -> Bool:
    """Do two support points come from the SAME vertex of their geom?

    MuJoCo compares support INDICES (`w->index1 == pt->verts[i].index1`), which
    it can because its polytope vertices carry them. Threading two more ints
    through every simplex and polytope slot here would add runtime-indexed
    per-thread arrays to the GPU path, which is the family that produced defect
    27; comparing POSITIONS is equivalent for the only geom types that reach
    this file. Box corners and mesh vertices are a discrete set, the support
    function returns the stored vertex verbatim, and it is deterministic — so
    the same vertex gives bit-identical coordinates and a different vertex is
    separated by a real edge length. The threshold is relative to the geom, not
    absolute: these are metres, and one epsilon cannot serve a 3 cm gripper and
    a 1 m table.
    """
    var dx = ax - bx
    var dy = ay - by
    var dz = az - bz
    var eps = scale * Scalar[DTYPE](1e-9)
    return dx * dx + dy * dy + dz * dz <= eps * eps


@always_inline
def _to_local[
    DTYPE: DType
](
    px: Scalar[DTYPE], py: Scalar[DTYPE], pz: Scalar[DTYPE],
    gx: Scalar[DTYPE], gy: Scalar[DTYPE], gz: Scalar[DTYPE],
    qx: Scalar[DTYPE], qy: Scalar[DTYPE], qz: Scalar[DTYPE],
    qw: Scalar[DTYPE],
) -> Tuple[Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE]]:
    """World point -> the geom's local frame."""
    var r = quat_rotate_inverse[DTYPE](qx, qy, qz, qw, px - gx, py - gy, pz - gz)
    return (r[0], r[1], r[2])


@always_inline
def _box_corner_index[
    DTYPE: DType
](lx: Scalar[DTYPE], ly: Scalar[DTYPE], lz: Scalar[DTYPE]) -> Int:
    """MuJoCo's box corner numbering, recovered from a local-frame position.

    `boxNormals` reads corner ids as a bitmask — `v1 & 1` is +x, `& 2` is +y,
    `& 4` is +z (`engine_collision_gjk.c:1912`) — and `boxEdgeNormals` builds
    the corner back from the same bits. So the id is exactly the sign pattern.
    """
    var i = 0
    if lx > Scalar[DTYPE](0):
        i += 1
    if ly > Scalar[DTYPE](0):
        i += 2
    if lz > Scalar[DTYPE](0):
        i += 4
    return i


@always_inline
def _mesh_vertex_index[
    DTYPE: DType,
    L_MESH_VERTS: Layout](
    lx: Scalar[DTYPE], ly: Scalar[DTYPE], lz: Scalar[DTYPE],
    mesh_verts: LayoutTensor[
        DTYPE, L_MESH_VERTS, MutAnyOrigin
    ],
    vert_adr: Int, num_verts: Int,
) -> Int:
    """Nearest hull vertex to a local-frame support point, or -1 if none.

    The support point IS one of these vertices, so this is an identity lookup
    written as a search. ⚠ IT IS NOW ONLY A FALLBACK: EPA carries each
    vertex's support index (GJK's simplex included) and `gjk_epa_witness`
    hands the winning face's out as `wfi`, so for a mesh this runs only if an
    index is missing. Measured on so101_tower at 1024 lanes (RTX 5090,
    2026-09-26) one pass of these scans was ~1.08 ms of a 2.23 ms collision
    launch — a serial walk over thousands of hull vertices.
    """
    var best = -1
    var bestd = Scalar[DTYPE](1e30)
    for i in range(num_verts):
        var dx = rebind[Scalar[DTYPE]](mesh_verts[vert_adr + i, 0]) - lx
        var dy = rebind[Scalar[DTYPE]](mesh_verts[vert_adr + i, 1]) - ly
        var dz = rebind[Scalar[DTYPE]](mesh_verts[vert_adr + i, 2]) - lz
        var d = dx * dx + dy * dy + dz * dz
        if d < bestd:
            bestd = d
            best = i
    return best


@always_inline
def _area4[
    DTYPE: DType
](
    ax: Scalar[DTYPE], ay: Scalar[DTYPE], az: Scalar[DTYPE],
    bx: Scalar[DTYPE], by: Scalar[DTYPE], bz: Scalar[DTYPE],
    cx: Scalar[DTYPE], cy: Scalar[DTYPE], cz: Scalar[DTYPE],
    dx: Scalar[DTYPE], dy: Scalar[DTYPE], dz: Scalar[DTYPE],
) -> Scalar[DTYPE]:
    """`area4` — area of the quadrilateral (a, b, c, d) via its diagonals."""
    var adx = dx - ax
    var ady = dy - ay
    var adz = dz - az
    var dbx = bx - dx
    var dby = by - dy
    var dbz = bz - dz
    var bcx = cx - bx
    var bcy = cy - by
    var bcz = cz - bz
    var cax = ax - cx
    var cay = ay - cy
    var caz = az - cz
    var ex = ady * dbz - adz * dby
    var ey = adz * dbx - adx * dbz
    var ez = adx * dby - ady * dbx
    var fx = bcy * caz - bcz * cay
    var fy = bcz * cax - bcx * caz
    var fz = bcx * cay - bcy * cax
    var gx = ex + fx
    var gy = ey + fy
    var gz = ez + fz
    return Scalar[DTYPE](0.5) * sqrt(gx * gx + gy * gy + gz * gz)


@always_inline
def _plane_normal[
    DTYPE: DType
](
    v1x: Scalar[DTYPE], v1y: Scalar[DTYPE], v1z: Scalar[DTYPE],
    v2x: Scalar[DTYPE], v2y: Scalar[DTYPE], v2z: Scalar[DTYPE],
    nx: Scalar[DTYPE], ny: Scalar[DTYPE], nz: Scalar[DTYPE],
) -> Tuple[Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE]]:
    """`planeNormal` — the clipping plane through edge (v1, v2), returns (n, d).

    ⚠⚠ DO NOT NORMALISE `res` HERE, AND CHECK THE TREE BEFORE QUOTING A
    REFERENCE COMMENT. This function carried a `mju_normalize3` and a docstring
    asserting "the reference keeps it to avoid asymmetric rounding later on".
    That comment is real but it belongs to **3.11.0 ONLY** — 3.6.0, 3.3.6 and
    `mujoco-main` all omit the normalise, and the pixi runtime we gate against
    is **3.10.0**, which predates 3.11. I transcribed 3.11's comment into a port
    of the older routine and then cited it as justification.

    The normalise is very nearly a no-op either way, which is why it survived so
    long: `planeIntersect` divides `(pd - dot(pn, a))` by `dot(pn, ab)` and both
    scale linearly with |pn|, so `t` is scale-invariant, and `halfspace` only
    reads the SIGN. It shifts one thing — the fixed `-mjMINVAL` (1e-15) epsilon
    in `halfspace` becomes an effectively different threshold once |pn| changes,
    so a vertex sitting exactly on a clipping plane can fall the other way.

    🎯 WHEN THE RUNTIME MOVES TO 3.11: put the normalise BACK, guarded on the
    runtime version, not on the newest tree lying around. conda-forge topped out
    at 3.10.0 as of 2026-08-12 (3.11.0 is on PyPI); `pixi search mujoco` is the
    check. The 3.11 form is `mju_normalize3(res)` immediately before the
    `dot3(res, v1)` return — nothing else in the routine changed.
    """
    var d1x = v2x - v1x
    var d1y = v2y - v1y
    var d1z = v2z - v1z
    # v3 = v1 + n, so diff2 = n
    var rx = d1y * nz - d1z * ny
    var ry = d1z * nx - d1x * nz
    var rz = d1x * ny - d1y * nx
    return (rx, ry, rz, rx * v1x + ry * v1y + rz * v1z)


@always_inline
def _polygon_quad[
    DTYPE: DType, L_WS: Layout
](
    ws: LayoutTensor[DTYPE, L_WS, MutAnyOrigin],
    wrow: Int,
    nvert: Int,
    mut r0: Int, mut r1: Int, mut r2: Int, mut r3: Int,
):
    """`polygonQuad` — prune a polygon to its maximum-area convex quadrilateral.

    The reference walks four POINTERS around the ring; this walks four INDICES
    and `next` is `(i + 1) % nvert`. The outer `for (; a < end; a += 3)` becomes
    a loop over the starting index, and the trailing `if (b == a)` fixup that
    keeps the four indices distinct is reproduced verbatim — without it a
    degenerate ring collapses b, c and d onto a and every candidate area is 0.
    """
    var a = 0
    var b = 1
    var c = 2
    var d = 3
    r0 = a
    r1 = b
    r2 = c
    r3 = d
    var m = _area4[DTYPE](
        _wv(ws, wrow, MC_WS_POLY, a, 0), _wv(ws, wrow, MC_WS_POLY, a, 1), _wv(ws, wrow, MC_WS_POLY, a, 2),
        _wv(ws, wrow, MC_WS_POLY, b, 0), _wv(ws, wrow, MC_WS_POLY, b, 1), _wv(ws, wrow, MC_WS_POLY, b, 2),
        _wv(ws, wrow, MC_WS_POLY, c, 0), _wv(ws, wrow, MC_WS_POLY, c, 1), _wv(ws, wrow, MC_WS_POLY, c, 2),
        _wv(ws, wrow, MC_WS_POLY, d, 0), _wv(ws, wrow, MC_WS_POLY, d, 1), _wv(ws, wrow, MC_WS_POLY, d, 2),
    )
    for _a in range(nvert):
        while True:
            var dn = (d + 1) % nvert
            var mn = _area4[DTYPE](
                _wv(ws, wrow, MC_WS_POLY, a, 0), _wv(ws, wrow, MC_WS_POLY, a, 1), _wv(ws, wrow, MC_WS_POLY, a, 2),
                _wv(ws, wrow, MC_WS_POLY, b, 0), _wv(ws, wrow, MC_WS_POLY, b, 1), _wv(ws, wrow, MC_WS_POLY, b, 2),
                _wv(ws, wrow, MC_WS_POLY, c, 0), _wv(ws, wrow, MC_WS_POLY, c, 1), _wv(ws, wrow, MC_WS_POLY, c, 2),
                _wv(ws, wrow, MC_WS_POLY, dn, 0), _wv(ws, wrow, MC_WS_POLY, dn, 1), _wv(ws, wrow, MC_WS_POLY, dn, 2),
            )
            if mn <= m:
                break
            m = mn
            d = dn
            r0 = a
            r1 = b
            r2 = c
            r3 = d
            while True:
                var cn = (c + 1) % nvert
                var mc = _area4[DTYPE](
                    _wv(ws, wrow, MC_WS_POLY, a, 0), _wv(ws, wrow, MC_WS_POLY, a, 1), _wv(ws, wrow, MC_WS_POLY, a, 2),
                    _wv(ws, wrow, MC_WS_POLY, b, 0), _wv(ws, wrow, MC_WS_POLY, b, 1), _wv(ws, wrow, MC_WS_POLY, b, 2),
                    _wv(ws, wrow, MC_WS_POLY, cn, 0), _wv(ws, wrow, MC_WS_POLY, cn, 1), _wv(ws, wrow, MC_WS_POLY, cn, 2),
                    _wv(ws, wrow, MC_WS_POLY, d, 0), _wv(ws, wrow, MC_WS_POLY, d, 1), _wv(ws, wrow, MC_WS_POLY, d, 2),
                )
                if mc <= m:
                    break
                m = mc
                c = cn
                r0 = a
                r1 = b
                r2 = c
                r3 = d
            while True:
                var bn = (b + 1) % nvert
                var mb = _area4[DTYPE](
                    _wv(ws, wrow, MC_WS_POLY, a, 0), _wv(ws, wrow, MC_WS_POLY, a, 1), _wv(ws, wrow, MC_WS_POLY, a, 2),
                    _wv(ws, wrow, MC_WS_POLY, bn, 0), _wv(ws, wrow, MC_WS_POLY, bn, 1), _wv(ws, wrow, MC_WS_POLY, bn, 2),
                    _wv(ws, wrow, MC_WS_POLY, c, 0), _wv(ws, wrow, MC_WS_POLY, c, 1), _wv(ws, wrow, MC_WS_POLY, c, 2),
                    _wv(ws, wrow, MC_WS_POLY, d, 0), _wv(ws, wrow, MC_WS_POLY, d, 1), _wv(ws, wrow, MC_WS_POLY, d, 2),
                )
                if mb <= m:
                    break
                m = mb
                b = bn
                r0 = a
                r1 = b
                r2 = c
                r3 = d
        if b == a:
            b = (b + 1) % nvert
            if c == b:
                c = (c + 1) % nvert
                if d == c:
                    d = (d + 1) % nvert
        a += 1
        if a >= nvert:
            break


@always_inline
def _polygon_clip[
    DTYPE: DType, L_WS: Layout
](
    ws: LayoutTensor[DTYPE, L_WS, MutAnyOrigin],
    wrow: Int,
    # ⚠ THE TWO FACES ARRIVE AS OFFSETS, NOT AS FIXED REGIONS, because the
    # caller swaps their roles: the edge/face branches clip face1 against
    # face2 and the face/face branch the other way round. Pinning each to its
    # own region would need a copy at one of the call sites.
    o_face1: Int,
    nface1: Int,
    o_face2: Int,
    nface2: Int,
    nx: Scalar[DTYPE], ny: Scalar[DTYPE], nz: Scalar[DTYPE],
    max_contacts: Int,
) -> Int:
    """`polygonClip` — Sutherland-Hodgman clip of face2 against face1.

    Writes the clipped polygon's vertices (MuJoCo's `status->x2`) into `out`
    and returns how many there are; the caller derives `x1` by subtracting the
    contact direction. Returns 0 when the clip is empty or the clipping face is
    not at least a triangle.

    ⚠ THE EDGE CASE `nface2 == 2` IS A DIFFERENT REDUCTION, not a special case
    of the general one: a clipped EDGE degenerates to collinear points, so the
    reference keeps only the two most distant and returns 2. Falling through to
    the quad pruner instead would emit near-duplicate rows for every edge/face
    contact, which is most of them at a generic pose.
    """
    if nface1 < 3:
        return 0

    # `poly`, `clipped`, `pn` and `pd` are REGIONS OF THE CALLER'S WORKSPACE
    # ROW, not locals — together with `face1`/`face2`/`out` they are 28 floats
    # per `MC_MAX_POLYVERT`, and a per-thread stack that size is what pinned
    # the cap at 56. Every slot is written before it is read within one call,
    # so nothing has to clear them. See `ccd_workspace.mojo`.

    # One clipping plane per edge of face1.
    for i in range(nface1):
        var j = 0 if i == nface1 - 1 else i + 1
        var r = _plane_normal[DTYPE](
            _wv(ws, wrow, o_face1, i, 0), _wv(ws, wrow, o_face1, i, 1), _wv(ws, wrow, o_face1, i, 2),
            _wv(ws, wrow, o_face1, j, 0), _wv(ws, wrow, o_face1, j, 1), _wv(ws, wrow, o_face1, j, 2),
            nx, ny, nz,
        )
        ws[wrow, MC_WS_PN + i * 3 + 0] = r[0]
        ws[wrow, MC_WS_PN + i * 3 + 1] = r[1]
        ws[wrow, MC_WS_PN + i * 3 + 2] = r[2]
        ws[wrow, MC_WS_PD + i] = r[3]

    var npolygon = nface2
    for i in range(nface2):
        ws[wrow, MC_WS_POLY + i * 3 + 0] = _wv(ws, wrow, o_face2, i, 0)
        ws[wrow, MC_WS_POLY + i * 3 + 1] = _wv(ws, wrow, o_face2, i, 1)
        ws[wrow, MC_WS_POLY + i * 3 + 2] = _wv(ws, wrow, o_face2, i, 2)

    for e in range(nface1):
        var nclipped = 0
        for i in range(npolygon):
            var iq = i + 1 if i < npolygon - 1 else 0
            var px = _wv(ws, wrow, MC_WS_POLY, i, 0)
            var py = _wv(ws, wrow, MC_WS_POLY, i, 1)
            var pz = _wv(ws, wrow, MC_WS_POLY, i, 2)
            var qx = _wv(ws, wrow, MC_WS_POLY, iq, 0)
            var qy = _wv(ws, wrow, MC_WS_POLY, iq, 1)
            var qz = _wv(ws, wrow, MC_WS_POLY, iq, 2)

            # `halfspace`: is the point on the inner side of this edge plane?
            var ax = _wv(ws, wrow, o_face1, e, 0)
            var ay = _wv(ws, wrow, o_face1, e, 1)
            var az = _wv(ws, wrow, o_face1, e, 2)
            var enx = _wv(ws, wrow, MC_WS_PN, e, 0)
            var eny = _wv(ws, wrow, MC_WS_PN, e, 1)
            var enz = _wv(ws, wrow, MC_WS_PN, e, 2)
            var in1 = _dot3[DTYPE](
                px - ax, py - ay, pz - az, enx, eny, enz
            ) > Scalar[DTYPE](-1e-15)
            var in2 = _dot3[DTYPE](
                qx - ax, qy - ay, qz - az, enx, eny, enz
            ) > Scalar[DTYPE](-1e-15)

            if not in1 and not in2:
                continue

            if in1 and in2:
                if nclipped < MC_CLIP_CAP:
                    ws[wrow, MC_WS_CLIPPED + nclipped * 3 + 0] = qx
                    ws[wrow, MC_WS_CLIPPED + nclipped * 3 + 1] = qy
                    ws[wrow, MC_WS_CLIPPED + nclipped * 3 + 2] = qz
                    nclipped += 1
                continue

            # `planeIntersect` on PQ, kept only when it lands inside the segment.
            var abx = qx - px
            var aby = qy - py
            var abz = qz - pz
            var den = _dot3[DTYPE](enx, eny, enz, abx, aby, abz)
            if den != Scalar[DTYPE](0):
                var t = (
                    _wr(ws, wrow, MC_WS_PD + e) - _dot3[DTYPE](enx, eny, enz, px, py, pz)
                ) / den
                if t >= Scalar[DTYPE](0) and t <= Scalar[DTYPE](1):
                    if nclipped < MC_CLIP_CAP:
                        ws[wrow, MC_WS_CLIPPED + nclipped * 3 + 0] = px + t * abx
                        ws[wrow, MC_WS_CLIPPED + nclipped * 3 + 1] = py + t * aby
                        ws[wrow, MC_WS_CLIPPED + nclipped * 3 + 2] = pz + t * abz
                        nclipped += 1

            if in2:
                if nclipped < MC_CLIP_CAP:
                    ws[wrow, MC_WS_CLIPPED + nclipped * 3 + 0] = qx
                    ws[wrow, MC_WS_CLIPPED + nclipped * 3 + 1] = qy
                    ws[wrow, MC_WS_CLIPPED + nclipped * 3 + 2] = qz
                    nclipped += 1

        for k in range(nclipped * 3):
            ws[wrow, MC_WS_POLY + k] = _wr(ws, wrow, MC_WS_CLIPPED + k)
        npolygon = nclipped

    if npolygon < 1:
        return 0

    # 3.12: prune the clipped vertices with POSITIVE distance from the
    # clipping face (`polygonClip`, "prune out vertices with positive
    # distance from the face") — a clipped vertex above face1's plane is
    # separated along the normal and is not a witness. 3.10 kept them and
    # gave every point the EPA depth; the reference now drops them and gives
    # each survivor its own signed plane distance (`witnessOnFace`), which
    # `MC_WS_ODIST` carries out alongside `MC_WS_OUT`.
    var f0x = _wv(ws, wrow, o_face1, 0, 0)
    var f0y = _wv(ws, wrow, o_face1, 0, 1)
    var f0z = _wv(ws, wrow, o_face1, 0, 2)
    var nkeep = 0
    for i in range(npolygon):
        var di = _dot3[DTYPE](
            _wv(ws, wrow, MC_WS_POLY, i, 0) - f0x,
            _wv(ws, wrow, MC_WS_POLY, i, 1) - f0y,
            _wv(ws, wrow, MC_WS_POLY, i, 2) - f0z,
            nx, ny, nz,
        )
        if di <= Scalar[DTYPE](0):
            if nkeep != i:
                ws[wrow, MC_WS_POLY + nkeep * 3 + 0] = _wv(ws, wrow, MC_WS_POLY, i, 0)
                ws[wrow, MC_WS_POLY + nkeep * 3 + 1] = _wv(ws, wrow, MC_WS_POLY, i, 1)
                ws[wrow, MC_WS_POLY + nkeep * 3 + 2] = _wv(ws, wrow, MC_WS_POLY, i, 2)
            nkeep += 1
    npolygon = nkeep
    if npolygon < 1:
        return 0

    comptime if MC_DEBUG_RING:
        print("  [ring] npolygon =", npolygon, " nface1 =", nface1,
              " nface2 =", nface2)
        for i in range(npolygon):
            print("    [ring] v", i, "=", _wv(ws, wrow, MC_WS_POLY, i, 0),
                  _wv(ws, wrow, MC_WS_POLY, i, 1), _wv(ws, wrow, MC_WS_POLY, i, 2))

    # More than a quad, and only four rows are wanted: keep the largest quad.
    if max_contacts < 5 and npolygon > 4:
        var r0 = 0
        var r1 = 0
        var r2 = 0
        var r3 = 0
        _polygon_quad[DTYPE](ws, wrow, npolygon, r0, r1, r2, r3)
        _emit_out[DTYPE](ws, wrow, 0, r0, f0x, f0y, f0z, nx, ny, nz)
        _emit_out[DTYPE](ws, wrow, 1, r1, f0x, f0y, f0z, nx, ny, nz)
        _emit_out[DTYPE](ws, wrow, 2, r2, f0x, f0y, f0z, nx, ny, nz)
        _emit_out[DTYPE](ws, wrow, 3, r3, f0x, f0y, f0z, nx, ny, nz)
        return 4

    # A clipped EDGE keeps only its two extremes.
    if nface2 == 2 and npolygon > 2:
        var b1 = 0
        var b2 = 1
        var best = Scalar[DTYPE](0)
        for i in range(npolygon):
            for j in range(i + 1, npolygon):
                var dx = _wv(ws, wrow, MC_WS_POLY, j, 0) - _wv(ws, wrow, MC_WS_POLY, i, 0)
                var dy = _wv(ws, wrow, MC_WS_POLY, j, 1) - _wv(ws, wrow, MC_WS_POLY, i, 1)
                var dz = _wv(ws, wrow, MC_WS_POLY, j, 2) - _wv(ws, wrow, MC_WS_POLY, i, 2)
                var d2 = dx * dx + dy * dy + dz * dz
                if d2 > best:
                    best = d2
                    b1 = i
                    b2 = j
        _emit_out[DTYPE](ws, wrow, 0, b1, f0x, f0y, f0z, nx, ny, nz)
        _emit_out[DTYPE](ws, wrow, 1, b2, f0x, f0y, f0z, nx, ny, nz)
        return 2

    for i in range(npolygon):
        _emit_out[DTYPE](ws, wrow, i, i, f0x, f0y, f0z, nx, ny, nz)
    return npolygon


@always_inline
def _emit_out[
    DTYPE: DType, L_WS: Layout
](
    ws: LayoutTensor[DTYPE, L_WS, MutAnyOrigin],
    wrow: Int,
    k: Int,
    src: Int,
    f0x: Scalar[DTYPE], f0y: Scalar[DTYPE], f0z: Scalar[DTYPE],
    nx: Scalar[DTYPE], ny: Scalar[DTYPE], nz: Scalar[DTYPE],
) :
    """Copy clipped vertex `src` to output slot `k`, with its signed plane
    distance from the clipping face (`witnessOnFace`'s `dist`) beside it in
    `MC_WS_ODIST`. The witness pair itself is reconstructed at emission:
    `w2 = v`, `w1 = v - |dist| * wit_dir`."""
    var vx = _wv(ws, wrow, MC_WS_POLY, src, 0)
    var vy = _wv(ws, wrow, MC_WS_POLY, src, 1)
    var vz = _wv(ws, wrow, MC_WS_POLY, src, 2)
    ws[wrow, MC_WS_OUT + k * 3 + 0] = vx
    ws[wrow, MC_WS_OUT + k * 3 + 1] = vy
    ws[wrow, MC_WS_OUT + k * 3 + 2] = vz
    ws[wrow, MC_WS_ODIST + k] = _dot3[DTYPE](
        vx - f0x, vy - f0y, vz - f0z, nx, ny, nz
    )


@always_inline
def _box_normals2[
    DTYPE: DType
](
    qx: Scalar[DTYPE], qy: Scalar[DTYPE], qz: Scalar[DTYPE], qw: Scalar[DTYPE],
    nx: Scalar[DTYPE], ny: Scalar[DTYPE], nz: Scalar[DTYPE],
    mut n: Array[Scalar[DTYPE], MC_MAX_DEG * 3],
    mut idx: Array[Int, MC_MAX_DEG],
) -> Int:
    """`boxNormals2` — recover the box face closest to the collision direction.

    The fallback when the corner bitmask does not pin down a single face.
    """
    var l = quat_rotate_inverse[DTYPE](qx, qy, qz, qw, nx, ny, nz)
    var lx = l[0]
    var ly = l[1]
    var lz = l[2]
    var ln = sqrt(lx * lx + ly * ly + lz * lz)
    if ln <= Scalar[DTYPE](0):
        return 0
    lx /= ln
    ly /= ln
    lz /= ln

    # The six axis normals, in MuJoCo's order: +x, -x, +y, -y, +z, -z.
    for i in range(6):
        var cx = Scalar[DTYPE](0)
        var cy = Scalar[DTYPE](0)
        var cz = Scalar[DTYPE](0)
        if i == 0:
            cx = Scalar[DTYPE](1)
        elif i == 1:
            cx = Scalar[DTYPE](-1)
        elif i == 2:
            cy = Scalar[DTYPE](1)
        elif i == 3:
            cy = Scalar[DTYPE](-1)
        elif i == 4:
            cz = Scalar[DTYPE](1)
        else:
            cz = Scalar[DTYPE](-1)
        if lx * cx + ly * cy + lz * cz > Scalar[DTYPE](MC_FACE_TOL):
            var w = quat_rotate[DTYPE](qx, qy, qz, qw, cx, cy, cz)
            n[0] = w[0]
            n[1] = w[1]
            n[2] = w[2]
            idx[0] = i
            return 1
    return 0


@always_inline
def _box_normals[
    DTYPE: DType
](
    dim: Int, v1: Int, v2: Int, v3: Int,
    qx: Scalar[DTYPE], qy: Scalar[DTYPE], qz: Scalar[DTYPE], qw: Scalar[DTYPE],
    dx: Scalar[DTYPE], dy: Scalar[DTYPE], dz: Scalar[DTYPE],
    mut n: Array[Scalar[DTYPE], MC_MAX_DEG * 3],
    mut idx: Array[Int, MC_MAX_DEG],
) -> Int:
    """`boxNormals` — candidate face normals from up to three corner ids.

    The bit arithmetic is the reference's: for each axis, +1 when EVERY corner
    is on the positive side, -1 when every corner is on the negative side, and
    0 when they straddle it — so a shared face shows up as the one axis the
    corners agree on.
    """
    if dim == 3:
        var c = 0
        var x = Int((v1 & 1) != 0 and (v2 & 1) != 0 and (v3 & 1) != 0) - Int(
            (v1 & 1) == 0 and (v2 & 1) == 0 and (v3 & 1) == 0
        )
        var y = Int((v1 & 2) != 0 and (v2 & 2) != 0 and (v3 & 2) != 0) - Int(
            (v1 & 2) == 0 and (v2 & 2) == 0 and (v3 & 2) == 0
        )
        var z = Int((v1 & 4) != 0 and (v2 & 4) != 0 and (v3 & 4) != 0) - Int(
            (v1 & 4) == 0 and (v2 & 4) == 0 and (v3 & 4) == 0
        )
        var w = quat_rotate[DTYPE](
            qx, qy, qz, qw,
            Scalar[DTYPE](x), Scalar[DTYPE](y), Scalar[DTYPE](z),
        )
        n[0] = w[0]
        n[1] = w[1]
        n[2] = w[2]
        var sgn = x + y + z
        if x != 0:
            idx[c] = 0
            c += 1
        if y != 0:
            idx[c] = 2
            c += 1
        if z != 0:
            idx[c] = 4
            c += 1
        if sgn == -1:
            idx[0] += 1
        if c == 1:
            return 1
        return _box_normals2[DTYPE](qx, qy, qz, qw, dx, dy, dz, n, idx)

    if dim == 2:
        var c = 0
        var x = Int((v1 & 1) != 0 and (v2 & 1) != 0) - Int(
            (v1 & 1) == 0 and (v2 & 1) == 0
        )
        var y = Int((v1 & 2) != 0 and (v2 & 2) != 0) - Int(
            (v1 & 2) == 0 and (v2 & 2) == 0
        )
        var z = Int((v1 & 4) != 0 and (v2 & 4) != 0) - Int(
            (v1 & 4) == 0 and (v2 & 4) == 0
        )
        if x != 0:
            var w = quat_rotate[DTYPE](
                qx, qy, qz, qw, Scalar[DTYPE](x), Scalar[DTYPE](0),
                Scalar[DTYPE](0),
            )
            n[0] = w[0]
            n[1] = w[1]
            n[2] = w[2]
            idx[c] = 0 if x > 0 else 1
            c += 1
        if y != 0:
            var w = quat_rotate[DTYPE](
                qx, qy, qz, qw, Scalar[DTYPE](0), Scalar[DTYPE](y),
                Scalar[DTYPE](0),
            )
            n[c * 3 + 0] = w[0]
            n[c * 3 + 1] = w[1]
            n[c * 3 + 2] = w[2]
            idx[c] = 2 if y > 0 else 3
            c += 1
        if z != 0:
            # ⚠ THE REFERENCE WRITES `res + 3` HERE, NOT `res + 3*c`
            # (`engine_collision_gjk.c:1938`). Reproduced, not corrected: when
            # both x and z are nonzero this OVERWRITES the second slot, and the
            # normal that survives decides which face the manifold is built on.
            # "Fixing" it would silently diverge from the runtime on exactly
            # the edge cases this routine exists for.
            var w = quat_rotate[DTYPE](
                qx, qy, qz, qw, Scalar[DTYPE](0), Scalar[DTYPE](0),
                Scalar[DTYPE](z),
            )
            n[3] = w[0]
            n[4] = w[1]
            n[5] = w[2]
            idx[c] = 4 if z > 0 else 5
            c += 1
        if c == 2:
            return 2
        return _box_normals2[DTYPE](qx, qy, qz, qw, dx, dy, dz, n, idx)

    if dim == 1:
        var sx = Scalar[DTYPE](1) if (v1 & 1) != 0 else Scalar[DTYPE](-1)
        var sy = Scalar[DTYPE](1) if (v1 & 2) != 0 else Scalar[DTYPE](-1)
        var sz = Scalar[DTYPE](1) if (v1 & 4) != 0 else Scalar[DTYPE](-1)
        var wx = quat_rotate[DTYPE](
            qx, qy, qz, qw, sx, Scalar[DTYPE](0), Scalar[DTYPE](0)
        )
        var wy = quat_rotate[DTYPE](
            qx, qy, qz, qw, Scalar[DTYPE](0), sy, Scalar[DTYPE](0)
        )
        var wz = quat_rotate[DTYPE](
            qx, qy, qz, qw, Scalar[DTYPE](0), Scalar[DTYPE](0), sz
        )
        n[0] = wx[0]
        n[1] = wx[1]
        n[2] = wx[2]
        n[3] = wy[0]
        n[4] = wy[1]
        n[5] = wy[2]
        n[6] = wz[0]
        n[7] = wz[1]
        n[8] = wz[2]
        idx[0] = 0 if sx > Scalar[DTYPE](0) else 1
        idx[1] = 2 if sy > Scalar[DTYPE](0) else 3
        idx[2] = 4 if sz > Scalar[DTYPE](0) else 5
        return 3
    return 0


@always_inline
def _box_edge_normals[
    DTYPE: DType
](
    dim: Int,
    v1x: Scalar[DTYPE], v1y: Scalar[DTYPE], v1z: Scalar[DTYPE],
    v2x: Scalar[DTYPE], v2y: Scalar[DTYPE], v2z: Scalar[DTYPE],
    v1i: Int,
    gx: Scalar[DTYPE], gy: Scalar[DTYPE], gz: Scalar[DTYPE],
    qx: Scalar[DTYPE], qy: Scalar[DTYPE], qz: Scalar[DTYPE], qw: Scalar[DTYPE],
    hx: Scalar[DTYPE], hy: Scalar[DTYPE], hz: Scalar[DTYPE],
    mut n: Array[Scalar[DTYPE], MC_MAX_DEG * 3],
    mut endverts: Array[Scalar[DTYPE], MC_MAX_DEG * 3],
) -> Int:
    """`boxEdgeNormals` — the edge direction(s) leaving the contact feature."""
    if dim == 2:
        endverts[0] = v2x
        endverts[1] = v2y
        endverts[2] = v2z
        var ex = v2x - v1x
        var ey = v2y - v1y
        var ez = v2z - v1z
        var l = sqrt(ex * ex + ey * ey + ez * ez)
        if l > Scalar[DTYPE](0):
            ex /= l
            ey /= l
            ez /= l
        n[0] = ex
        n[1] = ey
        n[2] = ez
        return 1

    if dim == 1:
        var sx = hx if (v1i & 1) != 0 else -hx
        var sy = hy if (v1i & 2) != 0 else -hy
        var sz = hz if (v1i & 4) != 0 else -hz
        for k in range(3):
            var cx = -sx if k == 0 else sx
            var cy = -sy if k == 1 else sy
            var cz = -sz if k == 2 else sz
            var w = quat_rotate[DTYPE](qx, qy, qz, qw, cx, cy, cz)
            var wx = gx + w[0]
            var wy = gy + w[1]
            var wz = gz + w[2]
            endverts[k * 3 + 0] = wx
            endverts[k * 3 + 1] = wy
            endverts[k * 3 + 2] = wz
            var ex = wx - v1x
            var ey = wy - v1y
            var ez = wz - v1z
            var l = sqrt(ex * ex + ey * ey + ez * ez)
            if l > Scalar[DTYPE](0):
                ex /= l
                ey /= l
                ez /= l
            n[k * 3 + 0] = ex
            n[k * 3 + 1] = ey
            n[k * 3 + 2] = ez
        return 3
    return 0


@always_inline
def _box_face[
    DTYPE: DType, L_WS: Layout
](
    ws: LayoutTensor[DTYPE, L_WS, MutAnyOrigin],
    wrow: Int,
    o_face: Int,
    face_id: Int,
    gx: Scalar[DTYPE], gy: Scalar[DTYPE], gz: Scalar[DTYPE],
    qx: Scalar[DTYPE], qy: Scalar[DTYPE], qz: Scalar[DTYPE], qw: Scalar[DTYPE],
    hx: Scalar[DTYPE], hy: Scalar[DTYPE], hz: Scalar[DTYPE],
) -> Int:
    """`boxFace` — the four corners of a box face, in the reference's order.

    ⚠ THE ORDER IS CLOCKWISE relative to the face's OUTWARD normal, which is
    the same convention `_mesh_face` produces by reversing the stored polygon.
    `polygonClip` reads both faces as ordered boundaries, so one of them being
    wound the other way turns every clipping halfspace inside out and the clip
    comes back empty.
    """
    var sx = Array[Scalar[DTYPE], 4](fill=Scalar[DTYPE](0))
    var sy = Array[Scalar[DTYPE], 4](fill=Scalar[DTYPE](0))
    var sz = Array[Scalar[DTYPE], 4](fill=Scalar[DTYPE](0))
    if face_id == 0:  # right (+x)
        sx[0] = hx; sy[0] = hy; sz[0] = hz
        sx[1] = hx; sy[1] = hy; sz[1] = -hz
        sx[2] = hx; sy[2] = -hy; sz[2] = -hz
        sx[3] = hx; sy[3] = -hy; sz[3] = hz
    elif face_id == 1:  # left (-x)
        sx[0] = -hx; sy[0] = hy; sz[0] = -hz
        sx[1] = -hx; sy[1] = hy; sz[1] = hz
        sx[2] = -hx; sy[2] = -hy; sz[2] = hz
        sx[3] = -hx; sy[3] = -hy; sz[3] = -hz
    elif face_id == 2:  # top (+y)
        sx[0] = -hx; sy[0] = hy; sz[0] = -hz
        sx[1] = hx; sy[1] = hy; sz[1] = -hz
        sx[2] = hx; sy[2] = hy; sz[2] = hz
        sx[3] = -hx; sy[3] = hy; sz[3] = hz
    elif face_id == 3:  # bottom (-y)
        sx[0] = -hx; sy[0] = -hy; sz[0] = hz
        sx[1] = hx; sy[1] = -hy; sz[1] = hz
        sx[2] = hx; sy[2] = -hy; sz[2] = -hz
        sx[3] = -hx; sy[3] = -hy; sz[3] = -hz
    elif face_id == 4:  # front (+z)
        sx[0] = -hx; sy[0] = hy; sz[0] = hz
        sx[1] = hx; sy[1] = hy; sz[1] = hz
        sx[2] = hx; sy[2] = -hy; sz[2] = hz
        sx[3] = -hx; sy[3] = -hy; sz[3] = hz
    elif face_id == 5:  # back (-z)
        sx[0] = hx; sy[0] = hy; sz[0] = -hz
        sx[1] = -hx; sy[1] = hy; sz[1] = -hz
        sx[2] = -hx; sy[2] = -hy; sz[2] = -hz
        sx[3] = hx; sy[3] = -hy; sz[3] = -hz
    else:
        return 0
    for k in range(4):
        var w = quat_rotate[DTYPE](qx, qy, qz, qw, sx[k], sy[k], sz[k])
        ws[wrow, o_face + k * 3 + 0] = gx + w[0]
        ws[wrow, o_face + k * 3 + 1] = gy + w[1]
        ws[wrow, o_face + k * 3 + 2] = gz + w[2]
    return 4


@always_inline
def _mesh_normals[
    DTYPE: DType,
    L_MESH_POLYS: Layout,
    L_MESH_POLYMAP: Layout,
    L_MESH_VERT_POLYMAP: Layout](
    dim: Int, v1i: Int, v2i: Int, v3i: Int,
    qx: Scalar[DTYPE], qy: Scalar[DTYPE], qz: Scalar[DTYPE], qw: Scalar[DTYPE],
    vert_adr: Int, poly_adr: Int,
    mesh_polys: LayoutTensor[
        DTYPE, L_MESH_POLYS, MutAnyOrigin
    ],
    mesh_polymap: LayoutTensor[
        DTYPE, L_MESH_POLYMAP, MutAnyOrigin
    ],
    mesh_vert_polymap: LayoutTensor[
        DTYPE, L_MESH_VERT_POLYMAP, MutAnyOrigin
    ],
    mut n: Array[Scalar[DTYPE], MC_MAX_DEG * 3],
    mut idx: Array[Int, MC_MAX_DEG],
) -> Int:
    """`meshNormals` — candidate face normals from up to three hull vertices.

    Three vertices of a convex hull lie on exactly one face; two lie on an
    edge, which two faces share; one lies on all the faces meeting there. The
    lookups go through `polymap`, the vertex -> polygon map built at load.
    """
    var a1 = Int(rebind[Scalar[DTYPE]](mesh_vert_polymap[vert_adr + v1i, 0]))
    var n1 = Int(rebind[Scalar[DTYPE]](mesh_vert_polymap[vert_adr + v1i, 1]))

    if dim == 1:
        var c = 0
        for i in range(n1):
            if c >= MC_MAX_DEG:
                break
            var p = Int(rebind[Scalar[DTYPE]](mesh_polymap[a1 + i]))
            var po = poly_adr + p  # ROW index: mesh_polys is [poly, field]
            var w = quat_rotate[DTYPE](
                qx, qy, qz, qw,
                rebind[Scalar[DTYPE]](mesh_polys[po, MESH_POLY_IDX_NX]),
                rebind[Scalar[DTYPE]](mesh_polys[po, MESH_POLY_IDX_NY]),
                rebind[Scalar[DTYPE]](mesh_polys[po, MESH_POLY_IDX_NZ]),
            )
            n[c * 3 + 0] = w[0]
            n[c * 3 + 1] = w[1]
            n[c * 3 + 2] = w[2]
            idx[c] = p
            c += 1
        return c

    var a2 = Int(rebind[Scalar[DTYPE]](mesh_vert_polymap[vert_adr + v2i, 0]))
    var n2v = Int(rebind[Scalar[DTYPE]](mesh_vert_polymap[vert_adr + v2i, 1]))

    # `intersect` on the two vertices' polygon lists: at most two survive.
    var e0 = -1
    var e1 = -1
    var ne = 0
    for i in range(n1):
        if ne == 2:
            break
        var pi = Int(rebind[Scalar[DTYPE]](mesh_polymap[a1 + i]))
        for j in range(n2v):
            if Int(rebind[Scalar[DTYPE]](mesh_polymap[a2 + j])) == pi:
                if ne == 0:
                    e0 = pi
                else:
                    e1 = pi
                ne += 1
                break
    if ne == 0:
        return 0

    if dim == 2:
        var c = 0
        for i in range(ne):
            var p = e0 if i == 0 else e1
            var po = poly_adr + p  # ROW index: mesh_polys is [poly, field]
            var w = quat_rotate[DTYPE](
                qx, qy, qz, qw,
                rebind[Scalar[DTYPE]](mesh_polys[po, MESH_POLY_IDX_NX]),
                rebind[Scalar[DTYPE]](mesh_polys[po, MESH_POLY_IDX_NY]),
                rebind[Scalar[DTYPE]](mesh_polys[po, MESH_POLY_IDX_NZ]),
            )
            n[c * 3 + 0] = w[0]
            n[c * 3 + 1] = w[1]
            n[c * 3 + 2] = w[2]
            idx[c] = p
            c += 1
        return c

    # dim == 3: intersect the edge's faces with the third vertex's list.
    var a3 = Int(rebind[Scalar[DTYPE]](mesh_vert_polymap[vert_adr + v3i, 0]))
    var n3 = Int(rebind[Scalar[DTYPE]](mesh_vert_polymap[vert_adr + v3i, 1]))
    var face = -1
    for i in range(ne):
        var pi = e0 if i == 0 else e1
        for j in range(n3):
            if Int(rebind[Scalar[DTYPE]](mesh_polymap[a3 + j])) == pi:
                face = pi
                break
        if face >= 0:
            break
    if face < 0:
        return 0
    var po = poly_adr + face  # ROW index, not a flat offset
    var w = quat_rotate[DTYPE](
        qx, qy, qz, qw,
        rebind[Scalar[DTYPE]](mesh_polys[po, MESH_POLY_IDX_NX]),
        rebind[Scalar[DTYPE]](mesh_polys[po, MESH_POLY_IDX_NY]),
        rebind[Scalar[DTYPE]](mesh_polys[po, MESH_POLY_IDX_NZ]),
    )
    n[0] = w[0]
    n[1] = w[1]
    n[2] = w[2]
    idx[0] = face
    return 1


@always_inline
def _mesh_edge_normals[
    DTYPE: DType,
    L_MESH_VERTS: Layout,
    L_MESH_POLYS: Layout,
    L_MESH_POLYVERT: Layout,
    L_MESH_VERT_POLYMAP: Layout](
    dim: Int,
    v1x: Scalar[DTYPE], v1y: Scalar[DTYPE], v1z: Scalar[DTYPE],
    v2x: Scalar[DTYPE], v2y: Scalar[DTYPE], v2z: Scalar[DTYPE],
    v1i: Int,
    gx: Scalar[DTYPE], gy: Scalar[DTYPE], gz: Scalar[DTYPE],
    qx: Scalar[DTYPE], qy: Scalar[DTYPE], qz: Scalar[DTYPE], qw: Scalar[DTYPE],
    vert_adr: Int, poly_adr: Int,
    mesh_verts: LayoutTensor[
        DTYPE, L_MESH_VERTS, MutAnyOrigin
    ],
    mesh_polys: LayoutTensor[
        DTYPE, L_MESH_POLYS, MutAnyOrigin
    ],
    mesh_polyvert: LayoutTensor[
        DTYPE, L_MESH_POLYVERT, MutAnyOrigin
    ],
    mesh_polymap: LayoutTensor[
        DTYPE, L_MESH_POLYVERT, MutAnyOrigin
    ],
    mesh_vert_polymap: LayoutTensor[
        DTYPE, L_MESH_VERT_POLYMAP, MutAnyOrigin
    ],
    mut n: Array[Scalar[DTYPE], MC_MAX_DEG * 3],
    mut endverts: Array[Scalar[DTYPE], MC_MAX_DEG * 3],
) -> Int:
    """`meshEdgeNormals` — edge directions leaving the contact vertex/edge.

    For `dim == 1` the edges are found by walking each incident polygon to the
    vertex BEFORE `v1i` in its cycle, which is the reference's choice and
    depends on the polygon winding being the one `mesh_polygons.mojo` stores.
    """
    if dim == 2:
        endverts[0] = v2x
        endverts[1] = v2y
        endverts[2] = v2z
        var ex = v2x - v1x
        var ey = v2y - v1y
        var ez = v2z - v1z
        var l = sqrt(ex * ex + ey * ey + ez * ez)
        if l > Scalar[DTYPE](0):
            ex /= l
            ey /= l
            ez /= l
        n[0] = ex
        n[1] = ey
        n[2] = ez
        return 1

    if dim == 1:
        var a1 = Int(
            rebind[Scalar[DTYPE]](mesh_vert_polymap[vert_adr + v1i, 0])
        )
        var n1 = Int(
            rebind[Scalar[DTYPE]](mesh_vert_polymap[vert_adr + v1i, 1])
        )
        var c = 0
        for i in range(n1):
            if c >= MC_MAX_DEG:
                break
            var p = Int(rebind[Scalar[DTYPE]](mesh_polymap[a1 + i]))
            var po = poly_adr + p  # ROW index: mesh_polys is [poly, field]
            var adr = Int(
                rebind[Scalar[DTYPE]](mesh_polys[po, MESH_POLY_IDX_VERTADR])
            )
            var num = Int(
                rebind[Scalar[DTYPE]](mesh_polys[po, MESH_POLY_IDX_VERTNUM])
            )
            for j in range(num):
                if Int(rebind[Scalar[DTYPE]](mesh_polyvert[adr + j])) != v1i:
                    continue
                var k = num - 1 if j == 0 else j - 1
                var vk = Int(rebind[Scalar[DTYPE]](mesh_polyvert[adr + k]))
                var w = quat_rotate[DTYPE](
                    qx, qy, qz, qw,
                    rebind[Scalar[DTYPE]](mesh_verts[vert_adr + vk, 0]),
                    rebind[Scalar[DTYPE]](mesh_verts[vert_adr + vk, 1]),
                    rebind[Scalar[DTYPE]](mesh_verts[vert_adr + vk, 2]),
                )
                var wx = gx + w[0]
                var wy = gy + w[1]
                var wz = gz + w[2]
                endverts[c * 3 + 0] = wx
                endverts[c * 3 + 1] = wy
                endverts[c * 3 + 2] = wz
                var ex = wx - v1x
                var ey = wy - v1y
                var ez = wz - v1z
                var l = sqrt(ex * ex + ey * ey + ez * ez)
                if l > Scalar[DTYPE](0):
                    ex /= l
                    ey /= l
                    ez /= l
                n[c * 3 + 0] = ex
                n[c * 3 + 1] = ey
                n[c * 3 + 2] = ez
                break
            c += 1
        return c
    return 0


@always_inline
def _mesh_face[
    DTYPE: DType,
    L_MESH_VERTS: Layout,
    L_MESH_POLYS: Layout,
    L_MESH_POLYVERT: Layout,
    L_WS: Layout](
    ws: LayoutTensor[DTYPE, L_WS, MutAnyOrigin],
    wrow: Int,
    o_face: Int,
    face_id: Int,
    gx: Scalar[DTYPE], gy: Scalar[DTYPE], gz: Scalar[DTYPE],
    qx: Scalar[DTYPE], qy: Scalar[DTYPE], qz: Scalar[DTYPE], qw: Scalar[DTYPE],
    vert_adr: Int, poly_adr: Int,
    mesh_verts: LayoutTensor[
        DTYPE, L_MESH_VERTS, MutAnyOrigin
    ],
    mesh_polys: LayoutTensor[
        DTYPE, L_MESH_POLYS, MutAnyOrigin
    ],
    mesh_polyvert: LayoutTensor[
        DTYPE, L_MESH_POLYVERT, MutAnyOrigin
    ],
) -> Int:
    """`meshFace` — one polygon in world coordinates, REVERSED.

    The reverse (`for i = nvert-1 downto 0`) is the reference's, and it is what
    makes a mesh face agree in winding with `boxFace`'s clockwise order. Our
    stored polygons are CCW as seen from outside, matching `m.mesh_polyvert`.
    """
    var po = poly_adr + face_id  # ROW index, not a flat offset
    var adr = Int(rebind[Scalar[DTYPE]](mesh_polys[po, MESH_POLY_IDX_VERTADR]))
    var num = Int(rebind[Scalar[DTYPE]](mesh_polys[po, MESH_POLY_IDX_VERTNUM]))
    if num > MC_MAX_POLYVERT:
        return 0
    var j = 0
    for i in range(num - 1, -1, -1):
        var vi = Int(rebind[Scalar[DTYPE]](mesh_polyvert[adr + i]))
        var w = quat_rotate[DTYPE](
            qx, qy, qz, qw,
            rebind[Scalar[DTYPE]](mesh_verts[vert_adr + vi, 0]),
            rebind[Scalar[DTYPE]](mesh_verts[vert_adr + vi, 1]),
            rebind[Scalar[DTYPE]](mesh_verts[vert_adr + vi, 2]),
        )
        ws[wrow, o_face + j * 3 + 0] = gx + w[0]
        ws[wrow, o_face + j * 3 + 1] = gy + w[1]
        ws[wrow, o_face + j * 3 + 2] = gz + w[2]
        j += 1
    return num


@always_inline
def _aligned_faces[
    DTYPE: DType
](
    v: Array[Scalar[DTYPE], MC_MAX_DEG * 3], nv: Int,
    w: Array[Scalar[DTYPE], MC_MAX_DEG * 3], nw: Int,
    mut r0: Int, mut r1: Int,
) -> Bool:
    """`alignedFaces` — first pair of normals facing each other within tol."""
    for i in range(nv):
        for j in range(nw):
            if _dot3[DTYPE](
                v[i * 3 + 0], v[i * 3 + 1], v[i * 3 + 2],
                w[j * 3 + 0], w[j * 3 + 1], w[j * 3 + 2],
            ) < Scalar[DTYPE](-MC_FACE_TOL):
                r0 = i
                r1 = j
                return True
    return False


@always_inline
def _aligned_face_edge[
    DTYPE: DType
](
    edge: Array[Scalar[DTYPE], MC_MAX_DEG * 3], nedge: Int,
    face: Array[Scalar[DTYPE], MC_MAX_DEG * 3], nface: Int,
    mut r0: Int, mut r1: Int,
) -> Bool:
    """`alignedFaceEdge` — first edge perpendicular to a face normal.

    ⚠ THE LOOP NEST IS FACE-OUTER, EDGE-INNER and the results come back as
    `r0 = edge index`, `r1 = face index` — the opposite order to the loops.
    Swapping either would pick a different feature on any contact where more
    than one pair qualifies.
    """
    for i in range(nface):
        for j in range(nedge):
            var d = _dot3[DTYPE](
                edge[j * 3 + 0], edge[j * 3 + 1], edge[j * 3 + 2],
                face[i * 3 + 0], face[i * 3 + 1], face[i * 3 + 2],
            )
            if abs(d) < Scalar[DTYPE](MC_EDGE_TOL):
                r0 = j
                r1 = i
                return True
    return False


def native_multicontact_contacts[
    DTYPE: DType,
    D: DimsLike,
    L_MESH_VERTS: Layout,
    L_MESH_POLYS: Layout,
    L_MESH_POLYVERT: Layout,
    L_MESH_VERT_POLYMAP: Layout,
    L_CONTACTS: Layout,
    L_WS: Layout,
](
    env: Int, body_a: Int, body_b: Int,
    gi_type: Int,
    pix: Scalar[DTYPE], piy: Scalar[DTYPE], piz: Scalar[DTYPE],
    qix: Scalar[DTYPE], qiy: Scalar[DTYPE], qiz: Scalar[DTYPE],
    qiw: Scalar[DTYPE],
    hxi: Scalar[DTYPE], hyi: Scalar[DTYPE], hzi: Scalar[DTYPE],
    rbound_i: Scalar[DTYPE], va1: Int, mnv1: Int, pa1: Int, pn1: Int,
    gj_type: Int,
    pjx: Scalar[DTYPE], pjy: Scalar[DTYPE], pjz: Scalar[DTYPE],
    qjx: Scalar[DTYPE], qjy: Scalar[DTYPE], qjz: Scalar[DTYPE],
    qjw: Scalar[DTYPE],
    hxj: Scalar[DTYPE], hyj: Scalar[DTYPE], hzj: Scalar[DTYPE],
    rbound_j: Scalar[DTYPE], va2: Int, mnv2: Int, pa2: Int, pn2: Int,
    dims: D,
    mesh_verts: LayoutTensor[
        DTYPE, L_MESH_VERTS, MutAnyOrigin
    ],
    mesh_polys: LayoutTensor[
        DTYPE, L_MESH_POLYS, MutAnyOrigin
    ],
    mesh_polyvert: LayoutTensor[
        DTYPE, L_MESH_POLYVERT, MutAnyOrigin
    ],
    mesh_polymap: LayoutTensor[
        DTYPE, L_MESH_POLYVERT, MutAnyOrigin
    ],
    mesh_vert_polymap: LayoutTensor[
        DTYPE, L_MESH_VERT_POLYMAP, MutAnyOrigin
    ],
    wf1: Array[Scalar[DTYPE], 9],
    wf2: Array[Scalar[DTYPE], 9],
    wx: Array[Scalar[DTYPE], 6],
    # The witness face's support indices from `gjk_epa_witness` (object 1's
    # three, then object 2's), -1 where EPA had none.
    wfi: Array[Int, 6],
    dist0: Scalar[DTYPE],
    contact_margin: Scalar[DTYPE],
    contact_friction: Scalar[DTYPE],
    contact_friction_spin: Scalar[DTYPE],
    contact_friction_roll: Scalar[DTYPE],
    contact_condim: Int,
    flip_normal: Bool,
    contacts: LayoutTensor[
        DTYPE, L_CONTACTS,
        MutAnyOrigin,
    ],
    # The polygon buffers, one row per env — see `ccd_workspace.mojo`. This
    # routine shares `d.ccd_ws` with EPA and uses a DISJOINT region of it.
    ws: LayoutTensor[DTYPE, L_WS, MutAnyOrigin],
    wrow: Int,
    mut num_contacts: Int,
    # ⚠ THE GAP HALF OF THE PAIR'S MARGIN, DEFAULTED TO 0 SO EVERY
    # EXISTING CALL SITE IS UNCHANGED. `contact_margin` is the narrowphase
    # CUTOFF (`margin + gap`); what a contact STORES as its
    # `includemargin` is `contact_margin - contact_gap`, and the solver excludes
    # `dist >= includemargin`. See `GEOM_IDX_GAP`.
    contact_gap: Scalar[DTYPE] = Scalar[DTYPE](0),
    # The emission cap. -1 = the model's `max_contacts` (every serial
    # caller); the block-per-env collision kernel passes the end of the
    # thread's staging window instead, so a routine cannot write past it.
    max_contacts_in: Int = -1,
) -> Int:
    """`multicontact` — emit the clipped face manifold, or 0 if there is none.

    Returns the number of contact records written. A return of 0 means the two
    contact features do not line up as face/face, edge/face or face/edge, and
    the caller must emit its single EPA point instead — that is exactly what
    the reference does by leaving `status->nx` at 1.

    ⚠ THE OPERANDS MUST ARRIVE IN MuJoCo'S ORDER, LOWER GEOM TYPE FIRST, and
    `flip_normal` is what lets the caller do that without disturbing the
    record's `body_a = gi` invariant. `mj_collideGeoms` sorts the pair so
    `type1 <= type2` before dispatching, and BOX (6) sorts before MESH (7), so
    the reference ALWAYS runs a box/mesh pair with the box as obj1. This
    routine is not symmetric in its operands — the edge/face branch tests
    `nface1 < 3 and nface1 <= nface2`, which prefers geom1 as the edge — so
    running it in geom-index order picks a different feature on exactly the
    poses where the two disagree. Measured: with the operands in our own order,
    `box x mesh` matched MuJoCo exactly (339/339 points, |dn| 2.8e-13) while
    `mesh x box`, the SAME PAIR declared the other way round, had 9 count
    mismatches and fell back to the single EPA point often enough to leave
    |dn| at 1.5e-3.

    When the caller swaps, the manifold normal comes out as `obj2 -> obj1`,
    which is `gi -> gj`; the record wants `body_b -> body_a`, so it is negated.
    """
    var max_contacts = (
        max_contacts_in if max_contacts_in >= 0 else dims.get_max_contacts()
    )
    # A mesh with no polygons (a degenerate hull) is skipped, as MuJoCo skips
    # `!obj->data.mesh.mesh_polynum`.
    if gi_type == GEOM_MESH and pn1 <= 0:
        return 0
    if gj_type == GEOM_MESH and pn2 <= 0:
        return 0

    var scale = rbound_i if rbound_i < rbound_j else rbound_j
    if scale <= Scalar[DTYPE](0):
        scale = Scalar[DTYPE](1)

    # ---- simplexDim on each geom, with the same reordering ------------------
    var a1x = wf1[0]; var a1y = wf1[1]; var a1z = wf1[2]
    var b1x = wf1[3]; var b1y = wf1[4]; var b1z = wf1[5]
    var c1x = wf1[6]; var c1y = wf1[7]; var c1z = wf1[8]
    # The carried support indices move WITH their points through the
    # reordering below, or a collapsed face would name the wrong vertex.
    var k1a = wfi[0]; var k1b = wfi[1]; var k1c = wfi[2]
    var nface1 = 3
    if _same_point[DTYPE](a1x, a1y, a1z, b1x, b1y, b1z, scale):
        if _same_point[DTYPE](a1x, a1y, a1z, c1x, c1y, c1z, scale):
            nface1 = 1
        else:
            b1x = c1x; b1y = c1y; b1z = c1z
            k1b = k1c
            nface1 = 2
    elif _same_point[DTYPE](c1x, c1y, c1z, a1x, a1y, a1z, scale) or _same_point[
        DTYPE
    ](c1x, c1y, c1z, b1x, b1y, b1z, scale):
        nface1 = 2

    var a2x = wf2[0]; var a2y = wf2[1]; var a2z = wf2[2]
    var b2x = wf2[3]; var b2y = wf2[4]; var b2z = wf2[5]
    var c2x = wf2[6]; var c2y = wf2[7]; var c2z = wf2[8]
    var k2a = wfi[3]; var k2b = wfi[4]; var k2c = wfi[5]
    var nface2 = 3
    if _same_point[DTYPE](a2x, a2y, a2z, b2x, b2y, b2z, scale):
        if _same_point[DTYPE](a2x, a2y, a2z, c2x, c2y, c2z, scale):
            nface2 = 1
        else:
            b2x = c2x; b2y = c2y; b2z = c2z
            k2b = k2c
            nface2 = 2
    elif _same_point[DTYPE](c2x, c2y, c2z, a2x, a2y, a2z, scale) or _same_point[
        DTYPE
    ](c2x, c2y, c2z, b2x, b2y, b2z, scale):
        nface2 = 2

    # ---- vertex ids for the (up to three) distinct support points -----------
    var i1a = 0; var i1b = 0; var i1c = 0
    var l1a = _to_local[DTYPE](a1x, a1y, a1z, pix, piy, piz, qix, qiy, qiz, qiw)
    var l1b = _to_local[DTYPE](b1x, b1y, b1z, pix, piy, piz, qix, qiy, qiz, qiw)
    var l1c = _to_local[DTYPE](c1x, c1y, c1z, pix, piy, piz, qix, qiy, qiz, qiw)
    if gi_type == GEOM_BOX:
        i1a = _box_corner_index[DTYPE](l1a[0], l1a[1], l1a[2])
        i1b = _box_corner_index[DTYPE](l1b[0], l1b[1], l1b[2])
        i1c = _box_corner_index[DTYPE](l1c[0], l1c[1], l1c[2])
    else:
        i1a = k1a if k1a >= 0 else _mesh_vertex_index[DTYPE](
            l1a[0], l1a[1], l1a[2], mesh_verts, va1, mnv1
        )
        i1b = k1b if k1b >= 0 else _mesh_vertex_index[DTYPE](
            l1b[0], l1b[1], l1b[2], mesh_verts, va1, mnv1
        )
        i1c = k1c if k1c >= 0 else _mesh_vertex_index[DTYPE](
            l1c[0], l1c[1], l1c[2], mesh_verts, va1, mnv1
        )
        if i1a < 0 or i1b < 0 or i1c < 0:
            return 0

    var i2a = 0; var i2b = 0; var i2c = 0
    var l2a = _to_local[DTYPE](a2x, a2y, a2z, pjx, pjy, pjz, qjx, qjy, qjz, qjw)
    var l2b = _to_local[DTYPE](b2x, b2y, b2z, pjx, pjy, pjz, qjx, qjy, qjz, qjw)
    var l2c = _to_local[DTYPE](c2x, c2y, c2z, pjx, pjy, pjz, qjx, qjy, qjz, qjw)
    if gj_type == GEOM_BOX:
        i2a = _box_corner_index[DTYPE](l2a[0], l2a[1], l2a[2])
        i2b = _box_corner_index[DTYPE](l2b[0], l2b[1], l2b[2])
        i2c = _box_corner_index[DTYPE](l2c[0], l2c[1], l2c[2])
    else:
        i2a = k2a if k2a >= 0 else _mesh_vertex_index[DTYPE](
            l2a[0], l2a[1], l2a[2], mesh_verts, va2, mnv2
        )
        i2b = k2b if k2b >= 0 else _mesh_vertex_index[DTYPE](
            l2b[0], l2b[1], l2b[2], mesh_verts, va2, mnv2
        )
        i2c = k2c if k2c >= 0 else _mesh_vertex_index[DTYPE](
            l2c[0], l2c[1], l2c[2], mesh_verts, va2, mnv2
        )
        if i2a < 0 or i2b < 0 or i2c < 0:
            return 0

    # ---- the contact direction, straight from the two witness points --------
    var dirx = wx[3] - wx[0]
    var diry = wx[4] - wx[1]
    var dirz = wx[5] - wx[2]
    var dirlen = sqrt(dirx * dirx + diry * diry + dirz * dirz)

    var n1 = Array[Scalar[DTYPE], MC_MAX_DEG * 3](fill=Scalar[DTYPE](0))
    var n2 = Array[Scalar[DTYPE], MC_MAX_DEG * 3](fill=Scalar[DTYPE](0))
    var idx1 = Array[Int, MC_MAX_DEG](fill=0)
    var idx2 = Array[Int, MC_MAX_DEG](fill=0)
    var endverts = Array[Scalar[DTYPE], MC_MAX_DEG * 3](
        fill=Scalar[DTYPE](0)
    )

    var nn1 = 0
    if gi_type == GEOM_BOX:
        nn1 = _box_normals[DTYPE](
            nface1, i1a, i1b, i1c, qix, qiy, qiz, qiw,
            -dirx, -diry, -dirz, n1, idx1,
        )
    elif gi_type == GEOM_MESH:
        nn1 = _mesh_normals[DTYPE](
            nface1, i1a, i1b, i1c, qix, qiy, qiz, qiw, va1, pa1,
            mesh_polys, mesh_polymap, mesh_vert_polymap, n1, idx1,
        )
    var nn2 = 0
    if gj_type == GEOM_BOX:
        nn2 = _box_normals[DTYPE](
            nface2, i2a, i2b, i2c, qjx, qjy, qjz, qjw,
            dirx, diry, dirz, n2, idx2,
        )
    elif gj_type == GEOM_MESH:
        nn2 = _mesh_normals[DTYPE](
            nface2, i2a, i2b, i2c, qjx, qjy, qjz, qjw, va2, pa2,
            mesh_polys, mesh_polymap, mesh_vert_polymap, n2, idx2,
        )

    var ri = 0
    var rj = 0
    var edgecon1 = False
    var edgecon2 = False
    comptime if MC_DEBUG:
        print(
            "    [mc] types", gi_type, gj_type,
            " nface", nface1, nface2, " nnorms", nn1, nn2,
            " dirlen", dirlen,
        )
        for t in range(nn1):
            print("      n1[", t, "] =", n1[t * 3 + 0], n1[t * 3 + 1],
                  n1[t * 3 + 2], " idx", idx1[t])
        for t in range(nn2):
            print("      n2[", t, "] =", n2[t * 3 + 0], n2[t * 3 + 1],
                  n2[t * 3 + 2], " idx", idx2[t])
    if not _aligned_faces[DTYPE](n1, nn1, n2, nn2, ri, rj):
        if nface1 < 3 and nface1 <= nface2:
            nn1 = 0
            if gi_type == GEOM_BOX:
                nn1 = _box_edge_normals[DTYPE](
                    nface1, a1x, a1y, a1z, b1x, b1y, b1z, i1a,
                    pix, piy, piz, qix, qiy, qiz, qiw, hxi, hyi, hzi,
                    n1, endverts,
                )
            elif gi_type == GEOM_MESH:
                nn1 = _mesh_edge_normals[
                    DTYPE](
                    nface1, a1x, a1y, a1z, b1x, b1y, b1z, i1a,
                    pix, piy, piz, qix, qiy, qiz, qiw, va1, pa1,
                    mesh_verts, mesh_polys, mesh_polyvert, mesh_polymap,
                    mesh_vert_polymap, n1, endverts,
                )
            if not _aligned_face_edge[DTYPE](n1, nn1, n2, nn2, ri, rj):
                return 0
            edgecon1 = True
        elif nface2 < 3:
            nn2 = 0
            if gj_type == GEOM_BOX:
                nn2 = _box_edge_normals[DTYPE](
                    nface2, a2x, a2y, a2z, b2x, b2y, b2z, i2a,
                    pjx, pjy, pjz, qjx, qjy, qjz, qjw, hxj, hyj, hzj,
                    n2, endverts,
                )
            elif gj_type == GEOM_MESH:
                nn2 = _mesh_edge_normals[
                    DTYPE](
                    nface2, a2x, a2y, a2z, b2x, b2y, b2z, i2a,
                    pjx, pjy, pjz, qjx, qjy, qjz, qjw, va2, pa2,
                    mesh_verts, mesh_polys, mesh_polyvert, mesh_polymap,
                    mesh_vert_polymap, n2, endverts,
                )
            if not _aligned_face_edge[DTYPE](n2, nn2, n1, nn1, ri, rj):
                return 0
            edgecon2 = True
        else:
            return 0

    # ---- recover each geom's matching face (or edge) ------------------------
    # `face1` / `face2` are `MC_WS_FACE1` / `MC_WS_FACE2` in the workspace row.
    var nf1 = 0
    var nf2 = 0

    if edgecon1:
        ws[wrow, MC_WS_FACE1 + 0] = a1x
        ws[wrow, MC_WS_FACE1 + 1] = a1y
        ws[wrow, MC_WS_FACE1 + 2] = a1z
        ws[wrow, MC_WS_FACE1 + 3] = endverts[ri * 3 + 0]
        ws[wrow, MC_WS_FACE1 + 4] = endverts[ri * 3 + 1]
        ws[wrow, MC_WS_FACE1 + 5] = endverts[ri * 3 + 2]
        nf1 = 2
    else:
        var ind = idx1[rj] if edgecon2 else idx1[ri]
        if gi_type == GEOM_BOX:
            nf1 = _box_face[DTYPE](
                ws, wrow, MC_WS_FACE1,
                ind, pix, piy, piz, qix, qiy, qiz, qiw, hxi, hyi, hzi,
            )
        elif gi_type == GEOM_MESH:
            nf1 = _mesh_face[DTYPE](
                ws, wrow, MC_WS_FACE1,
                ind, pix, piy, piz, qix, qiy, qiz, qiw, va1, pa1,
                mesh_verts, mesh_polys, mesh_polyvert,
            )

    if edgecon2:
        ws[wrow, MC_WS_FACE2 + 0] = a2x
        ws[wrow, MC_WS_FACE2 + 1] = a2y
        ws[wrow, MC_WS_FACE2 + 2] = a2z
        ws[wrow, MC_WS_FACE2 + 3] = endverts[ri * 3 + 0]
        ws[wrow, MC_WS_FACE2 + 4] = endverts[ri * 3 + 1]
        ws[wrow, MC_WS_FACE2 + 5] = endverts[ri * 3 + 2]
        nf2 = 2
    else:
        if gj_type == GEOM_BOX:
            nf2 = _box_face[DTYPE](
                ws, wrow, MC_WS_FACE2,
                idx2[rj], pjx, pjy, pjz, qjx, qjy, qjz, qjw,
                hxj, hyj, hzj,
            )
        elif gj_type == GEOM_MESH:
            nf2 = _mesh_face[DTYPE](
                ws, wrow, MC_WS_FACE2,
                idx2[rj], pjx, pjy, pjz, qjx, qjy, qjz, qjw, va2, pa2,
                mesh_verts, mesh_polys, mesh_polyvert,
            )
    if nf1 == 0 or nf2 == 0:
        return 0

    # ---- clip -------------------------------------------------------------
    # The clipped ring lands in `MC_WS_OUT`.
    var nx_out = 0
    var adx: Scalar[DTYPE]
    var ady: Scalar[DTYPE]
    var adz: Scalar[DTYPE]
    var swap = False

    if edgecon1:
        adx = -n2[rj * 3 + 0] * dirlen
        ady = -n2[rj * 3 + 1] * dirlen
        adz = -n2[rj * 3 + 2] * dirlen
        nx_out = _polygon_clip[DTYPE](
            ws, wrow, MC_WS_FACE2, nf2, MC_WS_FACE1, nf1,
            n2[rj * 3 + 0], n2[rj * 3 + 1], n2[rj * 3 + 2], 4,
        )
        swap = True
    elif edgecon2:
        adx = -n1[rj * 3 + 0] * dirlen
        ady = -n1[rj * 3 + 1] * dirlen
        adz = -n1[rj * 3 + 2] * dirlen
        nx_out = _polygon_clip[DTYPE](
            ws, wrow, MC_WS_FACE1, nf1, MC_WS_FACE2, nf2,
            n1[rj * 3 + 0], n1[rj * 3 + 1], n1[rj * 3 + 2], 4,
        )
    else:
        adx = n2[rj * 3 + 0] * dirlen
        ady = n2[rj * 3 + 1] * dirlen
        adz = n2[rj * 3 + 2] * dirlen
        nx_out = _polygon_clip[DTYPE](
            ws, wrow, MC_WS_FACE1, nf1, MC_WS_FACE2, nf2,
            n1[ri * 3 + 0], n1[ri * 3 + 1], n1[ri * 3 + 2], 4,
        )

    comptime if MC_DEBUG:
        print(
            "    [mc] branch edge1", edgecon1, " edge2", edgecon2,
            " ri", ri, " rj", rj, " nf", nf1, nf2, " nx_out", nx_out,
            " ad", adx, ady, adz,
        )
    if nx_out < 1:
        return 0

    # ---- emit ---------------------------------------------------------------
    # `pos` is the midpoint of the two witness points, which is `c - ad/2`
    # either way round; the record's normal is `x2 - x1` normalised, i.e. the
    # `body_b -> body_a` direction the single-point emit also stores.
    var rnx = adx
    var rny = ady
    var rnz = adz
    if swap:
        rnx = -adx
        rny = -ady
        rnz = -adz
    var rl = sqrt(rnx * rnx + rny * rny + rnz * rnz)
    if rl <= Scalar[DTYPE](0):
        return 0
    rnx /= rl
    rny /= rl
    rnz /= rl
    if flip_normal:
        rnx = -rnx
        rny = -rny
        rnz = -rnz

    # 3.12 `witnessOnFace` + `mjc_penetration`: per clipped vertex `v` with
    # its own signed plane distance `d`, the witnesses are `w2 = v` and
    # `w1 = v - |d| * wit_dir`, so `pos = (w1 + w2) / 2 = v - 0.5 |d| wit_dir`
    # and `con.dist = margin + d`. `ad` is `wit_dir` scaled by the EPA depth
    # (3.10's `approx_dir`, kept for the normal), so `wit_dir = ad / |ad|`.
    # 3.10 used the EPA depth for every point; the solver then saw one
    # penetration per manifold instead of one per row.
    #
    # `dist0` (the EPA depth) no longer reaches a contact row; it stays in
    # the signature so the two narrow phases keep one call shape.
    _ = dist0
    var udx = adx / rl
    var udy = ady / rl
    var udz = adz / rl
    var written = 0
    for k in range(nx_out):
        if num_contacts >= max_contacts:
            break
        var off = num_contacts * CONTACT_SIZE
        var dk = _wr(ws, wrow, MC_WS_ODIST + k)
        var half = Scalar[DTYPE](0.5) * (-dk if dk < Scalar[DTYPE](0) else dk)
        contacts[env, off + CONTACT_IDX_BODY_A] = Scalar[DTYPE](body_a)
        contacts[env, off + CONTACT_IDX_BODY_B] = Scalar[DTYPE](body_b)
        contacts[env, off + CONTACT_IDX_POS_X] = (
            _wv(ws, wrow, MC_WS_OUT, k, 0) - half * udx
        )
        contacts[env, off + CONTACT_IDX_POS_Y] = (
            _wv(ws, wrow, MC_WS_OUT, k, 1) - half * udy
        )
        contacts[env, off + CONTACT_IDX_POS_Z] = (
            _wv(ws, wrow, MC_WS_OUT, k, 2) - half * udz
        )
        contacts[env, off + CONTACT_IDX_NX] = rnx
        contacts[env, off + CONTACT_IDX_NY] = rny
        contacts[env, off + CONTACT_IDX_NZ] = rnz
        contacts[env, off + CONTACT_IDX_DIST] = contact_margin + dk
        contacts[
            env, off + CONTACT_IDX_INCLUDEMARGIN
        ] = contact_margin - contact_gap
        contacts[env, off + CONTACT_IDX_FRICTION] = contact_friction
        contacts[env, off + CONTACT_IDX_FRICTION_SPIN] = contact_friction_spin
        contacts[env, off + CONTACT_IDX_FRICTION_ROLL] = contact_friction_roll
        contacts[env, off + CONTACT_IDX_CONDIM] = Scalar[DTYPE](contact_condim)
        num_contacts += 1
        written += 1
    return written
