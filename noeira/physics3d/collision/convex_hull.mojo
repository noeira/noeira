"""Convex hull computation for mesh collision.

Loads STL mesh vertices, deduplicates them, computes the 3D convex hull,
and stores hull vertices for GJK/EPA collision detection.

Algorithm: incremental convex hull — add points one by one, delete the faces
the point can SEE, and cone it to the boundary of that region. O(n*h) where h
is hull size. Runs once at model load time.

⚠⚠ THE OUTPUT IS NOT JUST A SET OF VERTICES, IT IS A GRAPH THE NARROW PHASE
WALKS. `build_hull_edge_graph` turns these faces into the vertex adjacency
`_support_mesh` hill-climbs, and a greedy walk is only guaranteed to reach the
extreme vertex if that adjacency is a CONVEX POLYTOPE'S 1-SKELETON. It is
therefore not enough for the hull to contain the right points: the surface has
to be a closed, consistently wound, convex triangulation, which shows up as
`E == 3V - 6`, `F == 2V - 4`, and every undirected edge shared by exactly two
faces. Three things together are what make that hold, and NONE of them is
optional — each was measured on Menagerie's 944 sweepable STL meshes:

  1. `robust_predicates.orient3d_dd` decides visibility whenever float64
     cannot. A plane test that shares one tolerance with the mesh-degeneracy
     check answered inconsistently on slivers: 93 non-manifold hulls, and on
     19 of them the support walk stalled, worst deficit 28.9 mm.
  2. Only the CONNECTED COMPONENT of the visible set is deleted. Coning two
     disjoint patches to one apex builds a pinched surface the next insertions
     chase; measured, one mesh's face count ran away 701 -> 53 227.
  3. Faces are wound outward from the SEED and every stitch inherits its
     direction. Re-deriving winding from an interior point at the end let a
     sliver's unreliable normal flip a face.

MuJoCo gets the same object from qhull (`mjCMesh::MakeGraph`, `qhull Qt` then
`qh_triangulate`), which is why `mesh_graph[graphadr+1] == 2*numvert - 4` holds
for every mesh in its models.
"""

from std.math import sqrt, abs
from std.memory import bitcast

from .qhull_native import (
    qhull_faces, qhull_max_faces, qhull_shim_available,
)

# The float64 unit roundoff, and the safety factor on the visibility error
# bound. `K` is not tuned: it is a generous constant so the bound stays an
# UPPER bound on the float64 error, and a bound that is too large only sends
# more pairs to `orient3d_dd`, which is always right.
comptime _DBL_EPS: Float64 = 1.1102230246251565e-16
comptime _HULL_ERR_K: Float64 = 16.0

from .hull_cache import (
    HullPayload,
    hull_cache_load,
    hull_cache_path,
    hull_cache_store,
)
from .mesh_polygons import build_mesh_polygons, polygon_normal
from .mesh_seed import append_mesh_seed_table
from .robust_predicates import orient3d_dd
from ..model.mesh_inertia import (
    MeshInertia,
    transform_verts_to_principal_frame,
    apply_mesh_ref_transform,
    mesh_ref_is_identity,
)


def deduplicate_vertices[
    DTYPE: DType,
](
    raw_verts: List[Scalar[DTYPE]],
    num_raw: Int,
    mut out_verts: List[Scalar[DTYPE]],
) -> Int:
    """Deduplicate vertices from a flat [x0,y0,z0, x1,y1,z1, ...] array.

    Appends unique vertices to out_verts, returns num_unique.
    """
    # ⚠⚠ EXACT, NOT A TOLERANCE — MuJoCo's rule, and the hull depends on it.
    # This used to merge anything within `EPS_SQ = 1e-12` (one micron). MuJoCo
    # dedupes on exact equality, confirmed from the other side: an exact-bytes
    # dedup reproduces `mesh_vertnum` on every mesh checked (4131 on
    # shadow_dexee's mid link, 2632 on barkour's head), and the tolerant rule
    # does not. That matters far more than it looks: the deduped array is what
    # `MakeGraph` hulls, so merging a vertex MuJoCo keeps hands qhull a
    # DIFFERENT POINT SET and every facet, polygon and normal downstream is
    # then computed for a different solid. Measured on barkour's head with the
    # tolerant rule: 23 merged polygons beyond `MC_FACE_TOL`, worst normal
    # 81.25 deg out. Exact: 0 bad, worst 0.000000 deg.
    var start = len(out_verts)
    var num_unique = 0

    for i in range(num_raw):
        var vx = raw_verts[i * 3 + 0]
        var vy = raw_verts[i * 3 + 1]
        var vz = raw_verts[i * 3 + 2]

        var found = False
        for j in range(num_unique):
            var off = start + j * 3
            var dx = vx - out_verts[off + 0]
            var dy = vy - out_verts[off + 1]
            var dz = vz - out_verts[off + 2]
            if dx == 0 and dy == 0 and dz == 0:
                found = True
                break

        if not found:
            out_verts.append(vx)
            out_verts.append(vy)
            out_verts.append(vz)
            num_unique += 1

    return num_unique


def compute_mesh_rbound_at[
    DTYPE: DType,
](
    verts: List[Scalar[DTYPE]],
    vert_offset: Int,
    num_verts: Int,
) -> Scalar[
    DTYPE
]:
    """`geom_rbound` for a mesh geom — `mjCGeom::GetRBound`'s mesh case.

    MuJoCo takes the mesh's axis-aligned bounds `aamm`, folds them about the
    FRAME ORIGIN (`haabb[k] = max(|min_k|, |max_k|)`) and returns the AABB
    corner radius `sqrt(sum haabb^2)`. That is the radius of a sphere centred
    on the geom frame enclosing the box enclosing the mesh — deliberately
    LOOSER than the tightest enclosing sphere.

    ⚠⚠ WAS `compute_bounding_radius_at`, WHICH MEASURED FROM THE VERTEX
    CENTROID — renamed as well as rewritten, so that anything still calling
    the old name fails to compile rather than silently changing meaning.
    Three different quantities are easy to confuse here, and only one is
    MuJoCo's:

        max |v - centroid(V)|   <- what this used to return
        max |v|                 <- tightest sphere about the FRAME ORIGIN
        sqrt(sum max(|min|,|max|)^2)  <- MuJoCo, the AABB corner

    The centroid version is not even ordered against the other two: on Jaco's
    mesh 3 it returned 0.1586 against MuJoCo's 0.1364, EXCEEDING a bound that
    max |v| cannot exceed, because it is measured from a different centre.
    Measured on all nine Jaco meshes, ours spanned 0.72x to 1.16x of MuJoCo's.

    This is not cosmetic. `rbound` sets the plane-mesh spread filter
    (`0.3 * rbound` in `_plane_mesh_contacts`), so it decides how many contacts
    a plane-mesh pair emits; it also feeds broadphase culling, where MuJoCo's
    looser value can only widen the candidate set.

    Taken over HULL vertices rather than all vertices, which is the same
    answer: an axis extreme is always a hull vertex.
    """
    if num_verts == 0:
        return Scalar[DTYPE](0)
    var lo_x = verts[vert_offset + 0]
    var lo_y = verts[vert_offset + 1]
    var lo_z = verts[vert_offset + 2]
    var hi_x = lo_x
    var hi_y = lo_y
    var hi_z = lo_z
    for i in range(1, num_verts):
        var vx = verts[vert_offset + i * 3 + 0]
        var vy = verts[vert_offset + i * 3 + 1]
        var vz = verts[vert_offset + i * 3 + 2]
        if vx < lo_x:
            lo_x = vx
        if vy < lo_y:
            lo_y = vy
        if vz < lo_z:
            lo_z = vz
        if vx > hi_x:
            hi_x = vx
        if vy > hi_y:
            hi_y = vy
        if vz > hi_z:
            hi_z = vz
    var hx = max(abs(lo_x), abs(hi_x))
    var hy = max(abs(lo_y), abs(hi_y))
    var hz = max(abs(lo_z), abs(hi_z))
    return sqrt(hx * hx + hy * hy + hz * hz)


def compute_mesh_half_extents_at[
    DTYPE: DType,
](
    verts: List[Scalar[DTYPE]],
    vert_offset: Int,
    num_verts: Int,
) -> Tuple[Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE]]:
    """`geom_size` for a mesh geom — `mjCGeom::Compile`'s mesh case.

        size[k] = max(|aamm[k]|, |aamm[k+3]|)      (user_objects.cc:4089-4093)

    i.e. the half-extent of the smallest box CENTRED ON THE FRAME ORIGIN that
    contains the mesh — not the half-size of its axis-aligned bounding box,
    which is a different number whenever the mesh is off-centre in its own
    frame.

    ⚠⚠ MESH GEOMS CARRIED A 0.5 PLACEHOLDER HERE (AUD-46). `ray_mesh` and
    `ray_mesh_bvh` both run MuJoCo's own `ray_box(geom_size)` reject before
    touching a triangle (`engine_ray.c:894, :963`), so a 5 cm mesh whose
    geom_size said 0.5 was harmless, and a mesh reaching further than 0.5 m
    from its frame origin LOST every hit past that — silently, as a miss. On
    the panda's own mesh geoms MuJoCo's sizes run from 0.002 to 0.039.

    Three numbers are easy to confuse here and only one is MuJoCo's:

        (max_k - min_k) / 2      <- the AABB half-size, centred on the mesh
        max(|min_k|, |max_k|)    <- MuJoCo, centred on the FRAME ORIGIN
        0.5                      <- what was there

    Shares its bounds scan with `compute_mesh_rbound_at` above, which folds
    the same three numbers into the AABB corner radius: two quantities, one
    rule, so they cannot drift apart.
    """
    if num_verts == 0:
        return (Scalar[DTYPE](0), Scalar[DTYPE](0), Scalar[DTYPE](0))
    var lo_x = verts[vert_offset + 0]
    var lo_y = verts[vert_offset + 1]
    var lo_z = verts[vert_offset + 2]
    var hi_x = lo_x
    var hi_y = lo_y
    var hi_z = lo_z
    for i in range(1, num_verts):
        var vx = verts[vert_offset + i * 3 + 0]
        var vy = verts[vert_offset + i * 3 + 1]
        var vz = verts[vert_offset + i * 3 + 2]
        if vx < lo_x:
            lo_x = vx
        if vy < lo_y:
            lo_y = vy
        if vz < lo_z:
            lo_z = vz
        if vx > hi_x:
            hi_x = vx
        if vy > hi_y:
            hi_y = vy
        if vz > hi_z:
            hi_z = vz
    return (
        max(abs(lo_x), abs(hi_x)),
        max(abs(lo_y), abs(hi_y)),
        max(abs(lo_z), abs(hi_z)),
    )


# =============================================================================
# 3D Incremental Convex Hull
# =============================================================================


@always_inline
def _hull_cross[
    DTYPE: DType
](
    ax: Scalar[DTYPE],
    ay: Scalar[DTYPE],
    az: Scalar[DTYPE],
    bx: Scalar[DTYPE],
    by: Scalar[DTYPE],
    bz: Scalar[DTYPE],
) -> Tuple[Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE]]:
    return (ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx)


@always_inline
def _hull_dot[
    DTYPE: DType
](
    ax: Scalar[DTYPE],
    ay: Scalar[DTYPE],
    az: Scalar[DTYPE],
    bx: Scalar[DTYPE],
    by: Scalar[DTYPE],
    bz: Scalar[DTYPE],
) -> Scalar[DTYPE]:
    return ax * bx + ay * by + az * bz


# ⚠ MEASURED, NOT CHOSEN. This is the plane tolerance the facet merge below
# groups by, and it was fitted against qhull's OWN answer rather than picked:
# over 81 Menagerie and in-repo meshes it reproduces qhull's vertex SET exactly
# on 93.8% of them, never drops a vertex qhull keeps, and leaves the worst
# input point 6.168e-15 * scale outside the reduced hull — the SAME figure
# qhull's own reduced hull gives, i.e. roundoff. 1e-9 and above start dropping
# vertices qhull keeps; below 1e-10 nothing changes.
comptime _HULL_REDUCE_TAU: Float64 = 3e-10


def _reduce_hull_f64(
    hv: List[Float64],
    nh: Int,
    faces: List[Int],
) -> List[Int]:
    """qhull's VERTEX REDUCTION, by its effect rather than by its machinery.

    Returns the hull vertices worth keeping, in ascending order.

    ⚠⚠ WHY THIS EXISTS AT ALL. `_convex_hull_f64` builds the EXACT convex hull
    of the point set. qhull, which MuJoCo runs, builds a MERGED one: it fuses
    facets that are coplanar within roundoff and deletes the vertices that stop
    being corners. The two solids agree to roundoff, and their VERTEX SETS do
    not — 1031 against 798 on so_arm101's servo, 766 against 552 on
    hello_robot_stretch_3's base. That difference is invisible to a support
    query and decisive for the CONTACT MANIFOLD, because
    `mesh_polygons.build_mesh_polygons` merges the hull's triangles into the
    faces `native_multicontact` clips, and extra vertices redraw those faces.
    Measured on the board's worst scene: our face contained the whole of the
    other geom's 26-gon and MuJoCo's cut it to a 2-point sliver.

    ⚠ IT IS NOT A PORT OF `merge_r.c`, AND DOES NOT CLAIM TO BE. qhull's
    pre-merge is thousands of lines of merge sets, ridges and vertex
    reduction. This reproduces its OUTPUT: grow maximal planar facets over the
    triangulation, then keep only the corners of each facet's own 2D hull.

    ⚠ THE ONE INVARIANT THAT MAY NOT BREAK is that the solid must not SHRINK —
    a hull missing a genuine extreme point is a smaller shape, and that is an
    error with one sign that no gate downstream can see. Two things guard it:
    a group is only reduced if ALL its vertices really are coplanar within
    `tau` (the check below refuses otherwise), and a group whose 2D hull comes
    back degenerate keeps every vertex it had.
    """
    var nf = len(faces) // 3
    if nf < 4 or nh < 4:
        var all_v = List[Int]()
        for i in range(nh):
            all_v.append(i)
        return all_v^

    var scale = Float64(0)
    for i in range(nh * 3):
        var a = abs(hv[i])
        if a > scale:
            scale = a
    if scale <= Float64(0):
        scale = Float64(1)
    var tau = scale * _HULL_REDUCE_TAU

    # ---- per-face RAW normal (area-weighted by construction) --------------
    var fcx = List[Float64](capacity=nf * 3)
    for f in range(nf):
        var a = faces[f * 3 + 0]
        var b = faces[f * 3 + 1]
        var c = faces[f * 3 + 2]
        var ux = hv[b * 3 + 0] - hv[a * 3 + 0]
        var uy = hv[b * 3 + 1] - hv[a * 3 + 1]
        var uz = hv[b * 3 + 2] - hv[a * 3 + 2]
        var vx = hv[c * 3 + 0] - hv[a * 3 + 0]
        var vy = hv[c * 3 + 1] - hv[a * 3 + 1]
        var vz = hv[c * 3 + 2] - hv[a * 3 + 2]
        fcx.append(uy * vz - uz * vy)
        fcx.append(uz * vx - ux * vz)
        fcx.append(ux * vy - uy * vx)

    # ---- twin table over the 3*nf half-edges ------------------------------
    # ⚠ ONE SORT, NOT AN O(E^2) SCAN. The previous session's horizon walk could
    # afford a quadratic search because it only ever ran over the VISIBLE
    # faces; this runs over every face of a hull that reaches 6 500 vertices,
    # where quadratic is 4e8 comparisons per mesh.
    var nhe = nf * 3
    var codes = List[Int](capacity=nhe)
    for f in range(nf):
        for e in range(3):
            var u = faces[f * 3 + e]
            var v = faces[f * 3 + (e + 1) % 3]
            var lo_v = u if u < v else v
            var hi_v = v if u < v else u
            codes.append((lo_v * nh + hi_v) * nhe + (f * 3 + e))
    sort(codes)
    var twin = List[Int](length=nhe, fill=-1)
    var i = 0
    while i < nhe:
        var k0 = codes[i] // nhe
        var j = i
        while j < nhe and codes[j] // nhe == k0:
            j += 1
        if j - i == 2:
            var ha = codes[i] % nhe
            var hb = codes[i + 1] % nhe
            twin[ha] = hb
            twin[hb] = ha
        i = j

    # ---- grow maximal planar facets ---------------------------------------
    var group = List[Int](length=nf, fill=-1)
    var keep = List[Bool](length=nh, fill=False)
    var stamp = List[Int](length=nh, fill=-1)
    var stack = List[Int]()
    var gverts = List[Int]()
    var gid = 0
    for seed in range(nf):
        if group[seed] >= 0:
            continue
        group[seed] = gid
        var ax = fcx[seed * 3 + 0]
        var ay = fcx[seed * 3 + 1]
        var az = fcx[seed * 3 + 2]
        gverts.clear()
        for e in range(3):
            var v = faces[seed * 3 + e]
            if stamp[v] != gid:
                stamp[v] = gid
                gverts.append(v)
        var seedv = faces[seed * 3 + 0]
        stack.clear()
        stack.append(seed)
        while len(stack) > 0:
            var f = stack.pop()
            for e in range(3):
                var t = twin[f * 3 + e]
                if t < 0:
                    continue
                var g = t // 3
                if group[g] >= 0:
                    continue
                var al = sqrt(ax * ax + ay * ay + az * az)
                if al <= Float64(0):
                    continue
                var nx = ax / al
                var ny = ay / al
                var nz = az / al
                # ⚠ THE TEST IS OVER THE CANDIDATE'S OWN VERTICES PLUS THE
                # SEED, NOT OVER THE WHOLE GROUP. Testing the accumulated set
                # every step is O(group^2) and — measured over 81 meshes —
                # gives a WORSE answer: 91.4% exact against 93.8%, and it is
                # the variant that loses vertices qhull keeps. The final check
                # below is what makes the cheap test safe.
                var mx = hv[seedv * 3 + 0]
                var my = hv[seedv * 3 + 1]
                var mz = hv[seedv * 3 + 2]
                for e2 in range(3):
                    var w = faces[g * 3 + e2]
                    mx += hv[w * 3 + 0]
                    my += hv[w * 3 + 1]
                    mz += hv[w * 3 + 2]
                mx *= Float64(0.25)
                my *= Float64(0.25)
                mz *= Float64(0.25)
                var dmax = abs(
                    (hv[seedv * 3 + 0] - mx) * nx
                    + (hv[seedv * 3 + 1] - my) * ny
                    + (hv[seedv * 3 + 2] - mz) * nz
                )
                for e2 in range(3):
                    var w = faces[g * 3 + e2]
                    var d = abs(
                        (hv[w * 3 + 0] - mx) * nx
                        + (hv[w * 3 + 1] - my) * ny
                        + (hv[w * 3 + 2] - mz) * nz
                    )
                    if d > dmax:
                        dmax = d
                if dmax > tau:
                    continue
                group[g] = gid
                ax += fcx[g * 3 + 0]
                ay += fcx[g * 3 + 1]
                az += fcx[g * 3 + 2]
                for e2 in range(3):
                    var w = faces[g * 3 + e2]
                    if stamp[w] != gid:
                        stamp[w] = gid
                        gverts.append(w)
                stack.append(g)

        _reduce_one_group(hv, gverts, ax, ay, az, tau, keep)
        gid += 1

    var kept = List[Int]()
    for v in range(nh):
        if keep[v]:
            kept.append(v)
    return kept^


def _reduce_one_group(
    hv: List[Float64],
    gverts: List[Int],
    ax: Float64,
    ay: Float64,
    az: Float64,
    tau: Float64,
    mut keep: List[Bool],
):
    """Mark the CORNERS of one planar facet, or the whole of it if it is not
    planar enough to reduce safely."""
    var ng = len(gverts)
    if ng < 3:
        for k in range(ng):
            keep[gverts[k]] = True
        return

    var al = sqrt(ax * ax + ay * ay + az * az)
    if al <= Float64(0):
        for k in range(ng):
            keep[gverts[k]] = True
        return
    var nx = ax / al
    var ny = ay / al
    var nz = az / al

    var cx = Float64(0)
    var cy = Float64(0)
    var cz = Float64(0)
    for k in range(ng):
        var v = gverts[k]
        cx += hv[v * 3 + 0]
        cy += hv[v * 3 + 1]
        cz += hv[v * 3 + 2]
    cx /= Float64(ng)
    cy /= Float64(ng)
    cz /= Float64(ng)

    # ⚠⚠ THE SAFETY CHECK, AND IT IS THE WHOLE REASON THE CHEAP GROWTH TEST IS
    # ALLOWED. Growth only ever compared a candidate against the running
    # normal; nothing so far has verified that the ACCUMULATED group really is
    # planar. If it is not, dropping its interior vertices would cut a corner
    # off the solid — so the group keeps everything and this mesh simply stays
    # at our exact hull, which is never wrong, only finer.
    var worst = Float64(0)
    for k in range(ng):
        var v = gverts[k]
        var d = abs(
            (hv[v * 3 + 0] - cx) * nx
            + (hv[v * 3 + 1] - cy) * ny
            + (hv[v * 3 + 2] - cz) * nz
        )
        if d > worst:
            worst = d
    if worst > tau:
        for k in range(ng):
            keep[gverts[k]] = True
        return

    # in-plane basis
    var t1x = Float64(1)
    var t1y = Float64(0)
    var t1z = Float64(0)
    if abs(nx) > Float64(0.9):
        t1x = Float64(0)
        t1y = Float64(1)
        t1z = Float64(0)
    var dt = t1x * nx + t1y * ny + t1z * nz
    t1x -= nx * dt
    t1y -= ny * dt
    t1z -= nz * dt
    var t1l = sqrt(t1x * t1x + t1y * t1y + t1z * t1z)
    t1x /= t1l
    t1y /= t1l
    t1z /= t1l
    var t2x = ny * t1z - nz * t1y
    var t2y = nz * t1x - nx * t1z
    var t2z = nx * t1y - ny * t1x

    var px = List[Float64](capacity=ng)
    var py = List[Float64](capacity=ng)
    for k in range(ng):
        var v = gverts[k]
        var rx = hv[v * 3 + 0] - cx
        var ry = hv[v * 3 + 1] - cy
        var rz = hv[v * 3 + 2] - cz
        px.append(rx * t1x + ry * t1y + rz * t1z)
        py.append(rx * t2x + ry * t2y + rz * t2z)

    # lexicographic order by (x, y) — insertion sort, because a facet's vertex
    # count is small and this avoids a comparator on a float key.
    var ord = List[Int](capacity=ng)
    for k in range(ng):
        ord.append(k)
    for a2 in range(1, ng):
        var cur = ord[a2]
        var b2 = a2 - 1
        while b2 >= 0 and (
            px[ord[b2]] > px[cur]
            or (px[ord[b2]] == px[cur] and py[ord[b2]] > py[cur])
        ):
            ord[b2 + 1] = ord[b2]
            b2 -= 1
        ord[b2 + 1] = cur

    def _cross(o: Int, a3: Int, b3: Int) {imm} -> Float64:
        return (px[a3] - px[o]) * (py[b3] - py[o]) - (py[a3] - py[o]) * (
            px[b3] - px[o]
        )

    # monotone chain. `<= 0` pops, so COLLINEAR points are dropped — which is
    # the point: they are on the facet's edge, not a corner of it.
    var lower = List[Int]()
    for k in range(ng):
        var idx = ord[k]
        while len(lower) >= 2 and _cross(
            lower[len(lower) - 2], lower[len(lower) - 1], idx
        ) <= Float64(0):
            _ = lower.pop()
        lower.append(idx)
    var upper = List[Int]()
    for k in range(ng - 1, -1, -1):
        var idx = ord[k]
        while len(upper) >= 2 and _cross(
            upper[len(upper) - 2], upper[len(upper) - 1], idx
        ) <= Float64(0):
            _ = upper.pop()
        upper.append(idx)

    var ncorner = len(lower) - 1 + len(upper) - 1
    if ncorner < 3:
        # a degenerate in-plane hull says nothing useful about which vertices
        # are corners; keep them all rather than guess.
        for k in range(ng):
            keep[gverts[k]] = True
        return
    for k in range(len(lower) - 1):
        keep[gverts[lower[k]]] = True
    for k in range(len(upper) - 1):
        keep[gverts[upper[k]]] = True


def _convex_hull_f64(
    verts: List[Float64],
    num_verts: Int,
    mut hull_verts: List[Float64],
    mut hull_faces: List[Int],
) -> Int:
    """The hull itself, ALWAYS float64. See `compute_convex_hull`."""
    if num_verts < 4:
        for i in range(num_verts * 3):
            hull_verts.append(verts[i])
        return num_verts

    # Tolerance scaled to the mesh, not absolute: these are metres and a fixed
    # epsilon would be meaningless across a 3 cm gripper and a 1 m table.
    var scale = Float64(0)
    for i in range(num_verts * 3):
        var av = abs(verts[i])
        if av > scale:
            scale = av
    if scale <= Float64(0):
        scale = Float64(1)
    # ⚠⚠ THIS EPSILON ANSWERS "IS THE MESH DEGENERATE?", AND NOTHING ELSE. It
    # guards the seed-tetrahedron rejections below — all-collinear,
    # all-coplanar — which are questions about the SHAPE and want a coarse
    # threshold. VISIBILITY DOES NOT USE IT: see `_face_err` for the per-face
    # bound that replaced it, and the module header for why one shared number
    # was the bug.
    var eps = scale * 1e-9

    # ---- seed tetrahedron ------------------------------------------------
    var i0 = 0
    var i1 = 0
    var lo = verts[0]
    var hi = verts[0]
    for i in range(num_verts):
        var x = verts[i * 3 + 0]
        if x < lo:
            lo = x
            i0 = i
        if x > hi:
            hi = x
            i1 = i
    if i0 == i1:
        for i in range(num_verts * 3):
            hull_verts.append(verts[i])
        return num_verts

    var dx = verts[i1 * 3 + 0] - verts[i0 * 3 + 0]
    var dy = verts[i1 * 3 + 1] - verts[i0 * 3 + 1]
    var dz = verts[i1 * 3 + 2] - verts[i0 * 3 + 2]

    var i2 = -1
    var best = Float64(0)
    for i in range(num_verts):
        var ax = verts[i * 3 + 0] - verts[i0 * 3 + 0]
        var ay = verts[i * 3 + 1] - verts[i0 * 3 + 1]
        var az = verts[i * 3 + 2] - verts[i0 * 3 + 2]
        var cx = ay * dz - az * dy
        var cy = az * dx - ax * dz
        var cz = ax * dy - ay * dx
        var l = sqrt(cx * cx + cy * cy + cz * cz)
        if l > best:
            best = l
            i2 = i
    if i2 < 0 or best <= eps:
        for i in range(num_verts * 3):
            hull_verts.append(verts[i])
        return num_verts

    var e1x = verts[i1 * 3 + 0] - verts[i0 * 3 + 0]
    var e1y = verts[i1 * 3 + 1] - verts[i0 * 3 + 1]
    var e1z = verts[i1 * 3 + 2] - verts[i0 * 3 + 2]
    var e2x = verts[i2 * 3 + 0] - verts[i0 * 3 + 0]
    var e2y = verts[i2 * 3 + 1] - verts[i0 * 3 + 1]
    var e2z = verts[i2 * 3 + 2] - verts[i0 * 3 + 2]
    var nx = e1y * e2z - e1z * e2y
    var ny = e1z * e2x - e1x * e2z
    var nz = e1x * e2y - e1y * e2x
    var nl = sqrt(nx * nx + ny * ny + nz * nz)
    nx /= nl
    ny /= nl
    nz /= nl

    var i3 = -1
    var bestd = Float64(0)
    for i in range(num_verts):
        var ax = verts[i * 3 + 0] - verts[i0 * 3 + 0]
        var ay = verts[i * 3 + 1] - verts[i0 * 3 + 1]
        var az = verts[i * 3 + 2] - verts[i0 * 3 + 2]
        var d = abs(ax * nx + ay * ny + az * nz)
        if d > bestd:
            bestd = d
            i3 = i
    if i3 < 0 or bestd <= eps:
        for i in range(num_verts * 3):
            hull_verts.append(verts[i])
        return num_verts

    # ⚠⚠ THE SEED IS WOUND OUTWARD HERE AND STAYS WOUND FOR THE WHOLE BUILD.
    # This used to carry an INTERIOR REFERENCE POINT — the seed centroid — and
    # re-derive every face's outward direction from it on every visibility
    # test. That is the one thing a sliver breaks: the direction comes from a
    # normalised cross product, and a triangle whose vertices are within
    # roundoff of a line has no reliable normal, so its orientation could come
    # back FLIPPED and the face would report every point on the wrong side.
    # Orientation is COMBINATORIAL instead: order the seed so `i3` is below
    # `(i0, i1, i2)`, take the four faces that follow, and let every later face
    # inherit its winding from the horizon edge it is built on.
    if orient3d_dd(
        verts[i0 * 3 + 0], verts[i0 * 3 + 1], verts[i0 * 3 + 2],
        verts[i1 * 3 + 0], verts[i1 * 3 + 1], verts[i1 * 3 + 2],
        verts[i2 * 3 + 0], verts[i2 * 3 + 1], verts[i2 * 3 + 2],
        verts[i3 * 3 + 0], verts[i3 * 3 + 1], verts[i3 * 3 + 2],
    ) > Float64(0):
        var t = i1
        i1 = i2
        i2 = t

    var faces = List[Int]()
    # (a, b, c), (a, c, d), (a, d, b), (b, d, c) — the outward winding of a
    # tetrahedron whose fourth vertex `d` lies below face (a, b, c). Each of
    # the six edges appears once in each direction, which is the invariant the
    # horizon walk below relies on.
    var seed: Array[Int, 12] = [
        i0, i1, i2, i0, i2, i3, i0, i3, i1, i1, i3, i2
    ]
    for f in range(4):
        faces.append(seed[f * 3 + 0])
        faces.append(seed[f * 3 + 1])
        faces.append(seed[f * 3 + 2])

    var on_hull = List[Bool]()
    for _ in range(num_verts):
        on_hull.append(False)
    on_hull[i0] = True
    on_hull[i1] = True
    on_hull[i2] = True
    on_hull[i3] = True

    # ---- insertion ORDER: extremes first ----------------------------------
    # ⚠ ORDER IS A PERFORMANCE PROPERTY, NOT A COSMETIC ONE. Inserting points in
    # array order makes the early insertions see most of the polytope, and both
    # the horizon count (quadratic in the visible-face count) and the face-list
    # rebuild are paid at that size, for thousands of points. Measured: sawyer's
    # model build did not finish in TEN MINUTES.
    #
    # Front-loading support points along a spread of directions gets the hull
    # almost to its final shape within the first few dozen insertions, after
    # which nearly every remaining point is interior, sees no face, and costs
    # one visibility scan. Same hull either way — support points are hull
    # vertices by definition, so this changes only the path, never the result.
    var order = List[Int]()
    var queued = List[Bool]()
    for _ in range(num_verts):
        queued.append(False)
    for sx in range(-1, 2):
        for sy in range(-1, 2):
            for sz in range(-1, 2):
                if sx == 0 and sy == 0 and sz == 0:
                    continue
                var ddx = Float64(sx)
                var ddy = Float64(sy)
                var ddz = Float64(sz)
                var bi = 0
                var bd = Float64(-1e30)
                for i in range(num_verts):
                    var dd = (
                        ddx * verts[i * 3 + 0]
                        + ddy * verts[i * 3 + 1]
                        + ddz * verts[i * 3 + 2]
                    )
                    if dd > bd:
                        bd = dd
                        bi = i
                if not queued[bi]:
                    queued[bi] = True
                    order.append(bi)
    for i in range(num_verts):
        if not queued[i]:
            order.append(i)

    # ---- insert the remaining points -------------------------------------
    #
    # ⚠⚠ FACE PLANES ARE CACHED, AND THAT IS A COMPLEXITY FIX, NOT A TIDY-UP.
    # This loop used to recompute every face's normal FROM SCRATCH for every
    # candidate point — a cross product, a `sqrt`, a normalise and an
    # orientation dot, per face, per point. The comment on the insertion order
    # above already says the steady state is "nearly every remaining point is
    # interior, sees no face, and costs one visibility scan"; that scan was the
    # expensive part.
    #
    # MEASURED on SO-ARM101, whose collision meshes are the raw visual ones
    # (136 832 input vertices over 10 meshes, ~33 000 hull vertices): the
    # scan is ~1.6e8 sqrt-bearing normal computations for ONE mesh, and
    # `init_fields` did not finish in 250 s with 100% of the samples in here.
    # Cached, each face costs one dot product.
    #
    # ⚠ THE CACHE NO LONGER NORMALISES. Seven slots per face: the RAW outward
    # normal `(b - a) x (c - a)`, the anchor `a`, and `cf`, the constant of the
    # per-face error bound. Dropping the normalise removes the `sqrt` AND the
    # division that used to amplify a sliver's noise into the visibility test;
    # `det` is now exactly the quantity `orient3d_dd` refines when the bound
    # says the float64 sign cannot be trusted.
    var fcache = List[Float64]()

    def _rebuild_faces() {mut fcache, imm}:
        """(nx, ny, nz, ax, ay, az, cf) per face, wound outward by construction.

        `cf` is the per-face half of the error bound on `det = n . (p - a)`
        evaluated in float64. Two roundings contribute: the cross product,
        whose absolute error goes as `|u| |v|`, and the dot product, whose
        error goes as `|n|`. Both are multiplied by `|p - a|` at the call site,
        so the face-side constant is `K * DBL_EPSILON * (|u|_1 |v|_1 + |n|_1)`.
        1-norms, not 2-norms: they are upper bounds on the 2-norms, which makes
        the bound CONSERVATIVE — the only direction that is safe — and costs no
        `sqrt`.
        """
        fcache.clear()
        var nfc = len(faces) // 3
        for f in range(nfc):
            var a = faces[f * 3 + 0]
            var b = faces[f * 3 + 1]
            var c = faces[f * 3 + 2]
            var ux = verts[b * 3 + 0] - verts[a * 3 + 0]
            var uy = verts[b * 3 + 1] - verts[a * 3 + 1]
            var uz = verts[b * 3 + 2] - verts[a * 3 + 2]
            var vx = verts[c * 3 + 0] - verts[a * 3 + 0]
            var vy = verts[c * 3 + 1] - verts[a * 3 + 1]
            var vz = verts[c * 3 + 2] - verts[a * 3 + 2]
            var fx = uy * vz - uz * vy
            var fy = uz * vx - ux * vz
            var fz = ux * vy - uy * vx
            var u1 = abs(ux) + abs(uy) + abs(uz)
            var v1 = abs(vx) + abs(vy) + abs(vz)
            var n1 = abs(fx) + abs(fy) + abs(fz)
            fcache.append(fx)
            fcache.append(fy)
            fcache.append(fz)
            fcache.append(verts[a * 3 + 0])
            fcache.append(verts[a * 3 + 1])
            fcache.append(verts[a * 3 + 2])
            fcache.append(_HULL_ERR_K * _DBL_EPS * (u1 * v1 + n1))

    _rebuild_faces()

    var vis = List[Bool]()
    var vidx = List[Int]()
    var comp = List[Bool]()
    var stack = List[Int]()
    var heu = List[Int]()
    var hev = List[Int]()
    var twin = List[Int]()
    for oi in range(len(order)):
        var p = order[oi]
        if on_hull[p]:
            continue
        var px = verts[p * 3 + 0]
        var py = verts[p * 3 + 1]
        var pz = verts[p * 3 + 2]

        var nf = len(faces) // 3
        vis.clear()
        vidx.clear()
        var bestf = -1
        var bestd2 = Float64(0)
        for f in range(nf):
            var wx = px - fcache[f * 7 + 3]
            var wy = py - fcache[f * 7 + 4]
            var wz = pz - fcache[f * 7 + 5]
            var det = (
                fcache[f * 7 + 0] * wx
                + fcache[f * 7 + 1] * wy
                + fcache[f * 7 + 2] * wz
            )
            var wm = abs(wx)
            if abs(wy) > wm:
                wm = abs(wy)
            if abs(wz) > wm:
                wm = abs(wz)
            var eb = fcache[f * 7 + 6] * wm
            var seen: Bool
            if det > eb:
                seen = True
            elif det < -eb:
                seen = False
            else:
                # ⚠ THE BAND IS WHERE FLOAT64 CANNOT ANSWER, so nothing here
                # guesses: `orient3d_dd` re-evaluates the same determinant at
                # ~106 bits and returns the sign it really has. Measured on
                # Menagerie, the band is entered for a few tens of thousands of
                # (face, point) pairs per mesh against tens of millions tested,
                # so the cost sits under the noise of the scan itself.
                var a = faces[f * 3 + 0]
                var b = faces[f * 3 + 1]
                var c = faces[f * 3 + 2]
                seen = orient3d_dd(
                    verts[a * 3 + 0], verts[a * 3 + 1], verts[a * 3 + 2],
                    verts[b * 3 + 0], verts[b * 3 + 1], verts[b * 3 + 2],
                    verts[c * 3 + 0], verts[c * 3 + 1], verts[c * 3 + 2],
                    px, py, pz,
                ) > Float64(0)
            vis.append(seen)
            if seen:
                vidx.append(f)
                if bestf < 0 or det > bestd2:
                    bestd2 = det
                    bestf = f
        var nvis = len(vidx)
        if nvis == 0:
            continue

        # ⚠⚠ ONLY THE CONNECTED COMPONENT OF THE VISIBLE SET IS DELETED, and
        # that is not a safety net, it is the algorithm. Deleting a set of
        # faces and coning the point to its boundary is only well defined if
        # that boundary is ONE closed loop. Two disjoint visible patches give
        # two loops, and coning both to a single apex builds a pinched surface
        # the next insertions then chase — MEASURED, the face count of
        # `low_cost_robot_arm/elbow_to_wrist_extension_motor` ran away from 701
        # to 53 227 in four hundred insertions. qhull's `qh_findhorizon` grows
        # the visible set by breadth-first search from the facet the point was
        # assigned to for exactly this reason.
        #
        # With `orient3d_dd` deciding the band the visible set is connected
        # anyway, so on a well-conditioned mesh this walk marks every visible
        # face and changes nothing. It is what keeps a pathological one BOUNDED.
        # ⚠ THE TWIN TABLE IS BUILT ONCE, AND THAT IS A COMPLEXITY FIX. The
        # component walk and the horizon both need "which visible face owns the
        # reverse of this directed edge?"; asking that question inline made each
        # of them O(nvis^2) with a nine-way inner test, i.e. SIX TIMES the work
        # the old undirected horizon did, and it showed — 152 ms to 706 ms on
        # so_arm100's Wrist_Pitch_Roll. Answering it once for all 3*nvis
        # half-edges costs one O(nvis^2) pass, after which both walks are
        # linear.
        var nhe = nvis * 3
        heu.clear()
        hev.clear()
        for i in range(nvis):
            var f = vidx[i]
            for e in range(3):
                heu.append(faces[f * 3 + e])
                hev.append(faces[f * 3 + (e + 1) % 3])
        twin.clear()
        for _ in range(nhe):
            twin.append(-1)
        for k in range(nhe):
            if twin[k] >= 0:
                continue
            for m in range(k + 1, nhe):
                if twin[m] < 0 and heu[m] == hev[k] and hev[m] == heu[k]:
                    twin[k] = m
                    twin[m] = k
                    break

        comp.clear()
        for _ in range(nvis):
            comp.append(False)
        var startpos = 0
        for i in range(nvis):
            if vidx[i] == bestf:
                startpos = i
                break
        comp[startpos] = True
        stack.clear()
        stack.append(startpos)
        while len(stack) > 0:
            var ii = stack.pop()
            for e in range(3):
                var t = twin[ii * 3 + e]
                if t < 0:
                    continue
                var j = t // 3
                if not comp[j]:
                    comp[j] = True
                    stack.append(j)

        # ---- horizon: DIRECTED edges of the component with no twin inside it
        #
        # ⚠ DIRECTED, WHERE THIS USED TO BE UNDIRECTED. The old rule — "an edge
        # exactly one visible face owns" — cannot say which way round the new
        # triangle goes, so it appended `(lo, hi, p)` in whatever order the edge
        # happened to be stored and left the winding to be repaired at the end
        # from an interior point. Taking the edge in the direction the DELETED
        # face traverses it makes the new face `(u, v, p)` agree with the
        # neighbour that kept the reverse direction, and the whole surface stays
        # consistently wound with nothing to repair.
        #
        # `vis` is recycled as the per-face "this one dies" flag: a visible face
        # outside the component is KEPT, so the two are not the same set.
        for i in range(nvis):
            vis[vidx[i]] = comp[i]

        var kept = List[Int]()
        for f in range(nf):
            if vis[f]:
                continue
            kept.append(faces[f * 3 + 0])
            kept.append(faces[f * 3 + 1])
            kept.append(faces[f * 3 + 2])
        var added = 0
        for i in range(nvis):
            if not comp[i]:
                continue
            for e in range(3):
                var t = twin[i * 3 + e]
                if t >= 0 and comp[t // 3]:
                    continue
                kept.append(heu[i * 3 + e])
                kept.append(hev[i * 3 + e])
                kept.append(p)
                added += 1
        # A component with no boundary would be the WHOLE closed surface, which
        # a point outside a convex solid cannot see. Leaving the hull untouched
        # is the only answer that cannot destroy it.
        if added == 0:
            continue
        faces = kept^
        # The face set changed, so the cache must follow it. ⚠ This is the ONLY
        # place `faces` is reassigned; if that stops being true, this call has
        # to move with it or the visibility test reads stale faces — which
        # would not crash, it would quietly build a different hull.
        _rebuild_faces()
        on_hull[p] = True

    # ---- collect the vertices the surviving faces actually use ------------
    for i in range(num_verts):
        on_hull[i] = False
    for i in range(len(faces)):
        on_hull[faces[i]] = True

    # `remap[i]` is vertex i's index in the COMPACTED hull, or -1. The faces
    # are emitted in compacted numbering because that is what every consumer
    # sees: `mesh_vertadr` addresses the compacted block, and MuJoCo's
    # `mesh_polyvert` is likewise relative to `mesh_vertadr`.
    var remap = List[Int](length=num_verts, fill=-1)
    var num_hull = 0
    for i in range(num_verts):
        if on_hull[i]:
            hull_verts.append(verts[i * 3 + 0])
            hull_verts.append(verts[i * 3 + 1])
            hull_verts.append(verts[i * 3 + 2])
            remap[i] = num_hull
            num_hull += 1

    # ---- emit the faces, ALREADY WOUND OUTWARD ---------------------------
    #
    # ⚠⚠ THE WINDING IS NOT COSMETIC, AND IT IS NO LONGER REPAIRED HERE. This
    # loop used to re-derive each face's orientation from the seed centroid,
    # because the construction did not maintain one. It does now — the seed is
    # wound outward and every horizon stitch inherits its direction — so
    # re-deriving would only give a SLIVER the chance to be flipped by its own
    # unreliable normal, which is the failure this whole change removes.
    #
    # Winding matters to `mesh_polygons.build_mesh_polygons`, which merges two
    # triangles by cancelling a shared edge traversed in OPPOSITE directions.
    # With mixed winding the cancellation silently fails, no edges are removed,
    # and every face stays its own polygon — a cube would come back as 12
    # triangles instead of 6 quads, which is exactly the bug that path exists
    # to avoid.
    for f in range(len(faces) // 3):
        hull_faces.append(remap[faces[f * 3 + 0]])
        hull_faces.append(remap[faces[f * 3 + 1]])
        hull_faces.append(remap[faces[f * 3 + 2]])

    return num_hull


# =============================================================================
# Mesh loading pipeline
# =============================================================================




def compute_convex_hull[
    DTYPE: DType,
](
    verts: List[Scalar[DTYPE]],
    num_verts: Int,
    mut hull_verts: List[Scalar[DTYPE]],
    mut hull_faces: List[Int],
) -> Int:
    """EXACT 3D convex hull by incremental insertion.

    Replaces support-point SAMPLING, which could only ever return a SUBSET of
    the hull vertices. That subset is a strictly smaller solid, so GJK/EPA saw
    a shrunken shape and lost shallow contacts — an error with ONE SIGN that no
    gate could catch, because every mesh gate in the suite was frozen from an
    implementation that had it. On eGripperBase the sampler kept 81 of 882 hull
    vertices; raising its direction count was measured and did NOT help
    (`3b1b19db`), because a subset of extreme points is the wrong object no
    matter how it is chosen.

    The hull vertex set is exactly what a support query needs, so this gives
    SHAPE PARITY with MuJoCo, which collides the convex hull of `mesh_vert`.

    `hull_faces` receives the surviving triangles as vertex triples in
    COMPACTED numbering, wound OUTWARD — see the emit loop for why the winding
    has to be established there. A degenerate input (fewer than four points, or
    points that are collinear or coplanar) takes one of the early returns and
    leaves `hull_faces` EMPTY; that is the same state MuJoCo represents with
    `mesh_polynum == 0`, and `multicontact` skips such a mesh rather than
    guessing a face for it.

    Algorithm, matching the module docstring for the first time: seed a
    tetrahedron from four non-degenerate extremes, then insert each remaining
    point — delete the faces it can see, and stitch it to the horizon. The
    horizon is the set of edges belonging to exactly ONE visible face
    (UNDIRECTED, so it does not depend on consistent winding — the same rule
    `gjk.mojo`'s EPA uses, and for the same reason).

    O(n*h). Runs once per mesh at model build.
    """
    # ⚠⚠ THE HULL IS BUILT IN FLOAT64 NO MATTER WHAT `DTYPE` IS, AND THAT IS A
    # CORRECTNESS FIX, NOT A SPEED ONE.
    #
    # This used to run in DTYPE, and float32 built a DIFFERENT HULL: on
    # SO-ARM100's ten collision meshes, 2 636 vertices against float64's 2 551.
    # So a float32 env and a float64 env of the SAME model carried different
    # collision geometry — and every mesh gate we own runs at float64, so the
    # float32 path (the renderer, and the GPU batch) was never gated at all.
    #
    # ⚠ THE SPEED WAS THE SYMPTOM, NOT THE DEFECT. This is plain CPU code and
    # float32 arithmetic is not slower per operation — it was doing MORE WORK,
    # because the construction took a different path. Measured `init_fields`:
    #
    #     SO-ARM100    float64  0.48 s     float32  ~3 s
    #     SO-ARM101    float64  19.2 s     float32  >280 s
    #
    # Building in float64 always makes those identical AND makes the hull
    # dtype-independent, which `test_convex_hull_dtype_invariance` pins.
    #
    # ⚠ This is also what MuJoCo does — its compiler works in double regardless
    # of anything downstream. A convex hull is BUILD-TIME geometry and has no
    # reason to track the runtime dtype.
    #
    # Cost is one conversion pass in and one out against an O(n * faces)
    # construction: unmeasurable.
    var w = List[Float64](capacity=num_verts * 3)
    for i in range(num_verts * 3):
        w.append(Float64(verts[i]))
    var hw = List[Float64]()
    var hf = List[Int]()
    var n = _convex_hull_f64(w, num_verts, hw, hf)

    # ---- qhull's vertex reduction, then rebuild -----------------------------
    #
    # ⚠⚠ TWO HULL BUILDS, AND THE SECOND IS THE CHEAP ONE. `_reduce_hull_f64`
    # says WHICH vertices survive; it does not produce a triangulation, and
    # deleting vertices from an existing one by hand would mean re-stitching
    # the holes — the exact operation whose float-error corner cases the
    # previous commit spent a session removing. Re-running the builder on the
    # surviving points instead gives every invariant back by construction
    # (`E == 3V - 6`, manifold, no support-walk stall) for the price of a hull
    # over a SMALLER set: 766 points instead of 40 324 on
    # hello_robot_stretch_3's base, i.e. the second build is noise beside the
    # first.
    #
    # ⚠ EVERY FALLBACK HERE KEEPS THE EXACT HULL. If the reduction returns
    # nothing to drop, or drops so much that the rebuild degenerates, the
    # unreduced hull is used. It is never WRONG — only finer than MuJoCo's.
    if n >= 4 and len(hf) > 0:
        var kept = _reduce_hull_f64(hw, n, hf)
        if len(kept) >= 4 and len(kept) < n:
            var k = List[Float64](capacity=len(kept) * 3)
            for t in range(len(kept)):
                k.append(hw[kept[t] * 3 + 0])
                k.append(hw[kept[t] * 3 + 1])
                k.append(hw[kept[t] * 3 + 2])
            var hw2 = List[Float64]()
            var hf2 = List[Int]()
            var n2 = _convex_hull_f64(k, len(kept), hw2, hf2)
            if n2 >= 4 and len(hf2) > 0:
                hw = hw2^
                hf = hf2^
                n = n2

    for i in range(len(hf)):
        hull_faces.append(hf[i])
    for i in range(len(hw)):
        hull_verts.append(Scalar[DTYPE](hw[i]))
    return n

def build_hull_edge_graph(
    num_hull: Int,
    hull_faces: List[Int],
    vert_base: Int,
    mut edge_adr: List[Int],
    mut edge_list: List[Int],
):
    """The hull's VERTEX ADJACENCY, in the form `mjc_PlaneConvex` consumes.

    MuJoCo keeps this per mesh in `mesh_graph` and builds it in
    `mjCMesh::MakeGraph`: run qhull, `qh_triangulate`, `qh_vertexneighbors`,
    then for each vertex walk its neighbouring FACETS and collect the other two
    vertices of each, deduplicated, terminated by -1.

    ⚠ THE ADJACENCY IS THE TRIANGULATED HULL'S, NOT THE MERGED-POLYGON HULL'S.
    `qh_triangulate` splits coplanar facets, so the diagonals it introduces
    across a flat face ARE edges here. That is not a detail: those diagonals
    are the LONG edges, and `mjc_PlaneConvex` only accepts a neighbour that
    lies at least `0.3 * rbound` from the first contact. Deriving neighbours
    from `mesh_polygons.mojo`'s MERGED polygons instead would drop exactly the
    edges most likely to pass that filter, and a flat face resting on the
    plane would yield one contact where MuJoCo yields three.

    Confirmed against the 3.6.0 tree and the 3.10.0 runtime: for all nine Jaco
    meshes `mesh_graph[graphadr+1] == 2 * numvert - 4`, which is Euler's
    relation for a fully triangulated polytope — so qhull's stored graph really
    is simplicial, and `compute_convex_hull`'s triangles are the same object.

    ⚠ NEIGHBOUR ORDER IS NOT MUJOCO'S AND CANNOT BE. MuJoCo's per-vertex order
    follows qhull's internal facet set; ours follows `hull_faces`. Order only
    decides WHICH neighbours are taken when more than two pass the filter, so
    contact COUNTS still agree — see `_plane_mesh_contacts` for the measured
    consequence.

    `edge_adr` gets one entry per hull vertex, appended in the packed vertex
    order, so it is indexed by the same GLOBAL vertex index as `mesh_verts`;
    neighbours in `edge_list` are stored as global indices too (`vert_base` is
    this mesh's first vertex). MuJoCo needs its `vert_globalid` indirection
    because `mesh_vert` holds ALL vertices and only some are on the hull; ours
    holds hull vertices only, so local and global numbering coincide.
    """
    var nface = len(hull_faces) // 3

    # Vertex -> incident faces, CSR. The obvious alternative — rescanning
    # every face for every vertex — is O(V*F), which is ~5M comparisons on
    # sawyer's largest hull and grows quadratically with mesh size.
    var inc_count = List[Int](length=num_hull, fill=0)
    for k in range(nface * 3):
        inc_count[hull_faces[k]] += 1
    var inc_adr = List[Int](length=num_hull, fill=0)
    var run = 0
    for v in range(num_hull):
        inc_adr[v] = run
        run += inc_count[v]
    var filled = List[Int](length=num_hull, fill=0)
    var inc = List[Int](length=run, fill=0)
    for f in range(nface):
        for j in range(3):
            var v = hull_faces[f * 3 + j]
            inc[inc_adr[v] + filled[v]] = f
            filled[v] += 1

    for v in range(num_hull):
        edge_adr.append(len(edge_list))
        var start = len(edge_list)
        for t in range(inc_adr[v], inc_adr[v] + filled[v]):
            var f = inc[t]
            for j in range(3):
                var w = hull_faces[f * 3 + j]
                if w == v:
                    continue
                var g = vert_base + w
                var found = False
                for e in range(start, len(edge_list)):
                    if edge_list[e] == g:
                        found = True
                        break
                if not found:
                    edge_list.append(g)
        # The -1 separator is MuJoCo's own terminator, and the consumer walks
        # to it rather than to a stored degree.
        edge_list.append(-1)


@always_inline
def _pos_key3(x: Float64, y: Float64, z: Float64) -> String:
    """An EXACT key for a vertex position — its bits, not its value.

    ⚠ NOT A ROUNDED KEY. The two arrays being matched hold literal copies of
    the same doubles, so equality is exact and a tolerance would only invent
    collisions between genuinely distinct vertices of a fine mesh.
    """
    return (
        String(bitcast[DType.uint64](x)) + ","
        + String(bitcast[DType.uint64](y)) + ","
        + String(bitcast[DType.uint64](z))
    )


def _qhull_hull[
    DTYPE: DType
](
    unique: List[Scalar[DTYPE]],
    num_unique: Int,
    maxhullvert: Int,
    mut hull_verts: List[Scalar[DTYPE]],
    mut hull_faces: List[Int],
) raises -> Int:
    """The hull MuJoCo actually has: qhull's, vertices and face ORDER both.

    ⚠⚠ THIS REPLACES `compute_convex_hull` ON THIS PATH RATHER THAN CORRECTING
    IT, and that is the point. `mjCMesh::MakeGraph` does not implement a hull —
    it calls qhull — so the faithful answer is qhull's whole output, not our
    incremental hull's vertices with qhull's face order laid over them. The
    first version of this did exactly that and it FAILED LOUDLY, which is how
    the disagreement below was found: on barkour's `lower_leg_1to1` our hull
    drops global vertex 3960 of 5223 and qhull keeps it, so no renumbering of
    qhull's faces into our vertex set exists at all.

    ⚠ THE ORDER OF `hull_verts` IS FIRST APPEARANCE ACROSS qhull's FACES, which
    is OURS and not MuJoCo's — `mjModel.mesh_polyvert` indexes the full mesh
    array and we store only the hull. Nothing downstream depends on it: every
    consumer reads `poly_vert` / `polymap` / `edge_list`, which are built from
    these same local ids in the same pass. What must NOT change is the face
    ORDER, and that is qhull's exactly.

    Returns the hull vertex count, or 0 when the shim is absent or qhull
    declines the input (fewer than four points, collinear, coplanar) — the
    caller then falls back to `compute_convex_hull`, which has its own handling
    for those and is what a tree without the dylib uses throughout.
    """
    if not qhull_shim_available():
        return 0
    var cap = qhull_max_faces(num_unique)
    if cap < 1:
        return 0

    # ⚠ DOUBLES OF THE STORED VALUES, which is what `dvert` is: `Process`
    # builds it as `std::vector<double>(vert_.begin(), vert_.end())` from a
    # float32 array. At DTYPE=float32 this widens the same bits; at float64 the
    # values already came through a float32 loader, so both match MuJoCo.
    var dv = List[Float64](capacity=num_unique * 3)
    for i in range(num_unique * 3):
        dv.append(Float64(unique[i]))

    var qf = List[Int32](length=cap * 3, fill=Int32(0))
    var nf = qhull_faces(
        dv.unsafe_ptr(), num_unique, maxhullvert, qf.unsafe_ptr(), cap
    )
    if nf < 1:
        return 0

    # Compact: global id -> local, in first-appearance order.
    var local = List[Int](length=num_unique, fill=-1)
    var nh = 0
    hull_verts.clear()
    hull_faces.clear()
    for i in range(nf * 3):
        var g = Int(qf[i])
        if local[g] < 0:
            local[g] = nh
            hull_verts.append(unique[g * 3 + 0])
            hull_verts.append(unique[g * 3 + 1])
            hull_verts.append(unique[g * 3 + 2])
            nh += 1
        hull_faces.append(local[g])
    _ = dv^
    _ = qf^
    _ = local^
    return nh


def load_mesh_hull[
    DTYPE: DType,
](
    mesh_filename: String,
    mut mesh_vert: List[Scalar[DTYPE]],
    mut mesh_vertadr: List[Int],
    mut mesh_vertnum: List[Int],
    mut num_meshes: Int,
    mut mesh_polyadr: List[Int],
    mut mesh_polynum: List[Int],
    mut poly_vert: List[Int],
    mut poly_vertadr: List[Int],
    mut poly_vertnum: List[Int],
    mut poly_normal: List[Scalar[DTYPE]],
    mut polymap: List[Int],
    mut polymap_adr: List[Int],
    mut polymap_num: List[Int],
    mut edge_adr: List[Int],
    mut edge_list: List[Int],
    mut mesh_tri: List[Scalar[DTYPE]],
    mut mesh_triadr: List[Int],
    mut mesh_trinum: List[Int],
    mi: MeshInertia[DTYPE],
    sx: Float64 = 1.0,
    sy: Float64 = 1.0,
    sz: Float64 = 1.0,
    rpx: Float64 = 0.0,
    rpy: Float64 = 0.0,
    rpz: Float64 = 0.0,
    rqw: Float64 = 1.0,
    rqx: Float64 = 0.0,
    rqy: Float64 = 0.0,
    rqz: Float64 = 0.0,
    maxhullvert: Int = -1,
) raises -> Tuple[Int, Scalar[DTYPE]]:
    """Load STL mesh, deduplicate, compute convex hull, store in model arrays.

    Returns (mesh_id, rbound) for this mesh.

    ⚠ VERTICES ARE STORED IN THE MESH'S PRINCIPAL FRAME, not the STL's frame.
    `mi` carries the centre of mass and principal-axis rotation MuJoCo bakes
    into every mesh (`mesh_pos` / `mesh_quat`), and they are applied here —
    BEFORE the hull, the polygon normals and `rbound`, so all three are built
    in the same frame MuJoCo uses. The caller must compose the same `mi` into
    the geom's `pos`/`quat`, or the mesh will collide in the wrong place: the
    two changes are only equivalent TOGETHER.

    The `poly_*` / `polymap*` lists accumulate the hull's POLYGON topology
    across every mesh, exactly as `mesh_polyvert` and `mesh_polymap` do in
    `mjModel`; `mesh_polyadr` / `mesh_polynum` index into them per mesh. See
    `collision/mesh_polygons.mojo` for what they are for.
    """
    from noeira.render.stl_loader import load_stl

    # ── The cache ────────────────────────────────────────────────────────────
    # Everything between here and `rbound` costs 18.9 s on SO-ARM101, and the
    # viewer re-pays it on every env switch. `hull_cache` stores this mesh's
    # WHOLE output keyed on the STL's contents AND `mi`'s frame; see that
    # module for why the frame belongs in the key and for the rebasing table
    # the append block below implements.
    # ⚠⚠ `refpos`/`refquat` BELONG IN THE CACHE KEY. They change the vertices
    # this function stores, so two models naming the same file with different
    # ones must not share a payload. `mi` is already in the key for the same
    # reason; these ride in as the three scale slots do.
    var cache_path = hull_cache_path[DTYPE](
        mesh_filename, mi,
        sx * (1.0 + 7.0 * rpx + 13.0 * rqx),
        sy * (1.0 + 7.0 * rpy + 13.0 * rqy),
        sz * (1.0 + 7.0 * rpz + 13.0 * rqz + 3.0 * (rqw - 1.0)),
        maxhullvert,
    ) if not mesh_ref_is_identity(
        rpx, rpy, rpz, rqw, rqx, rqy, rqz
    ) else hull_cache_path[DTYPE](
        mesh_filename, mi, sx, sy, sz, maxhullvert
    )
    var p = HullPayload()

    if not hull_cache_load(cache_path, p):
        # ⚠⚠ THE HULL AND THE POLYGON PARTITION ARE BUILT IN THE FILE'S OWN
        # FRAME, NOT IN THE PRINCIPAL ONE. `mjCMesh::Process`
        # (user_mesh.cc:1350) runs `MakeGraph` (:1387) and `MakePolygons`
        # (:1422) on `dvert` while it still holds the raw file values, THEN
        # `ApplyTransformations` (:1444, refpos/refquat/scale), THEN the CoM
        # shift and the principal-axis `Rotate` (:1517-1524), and only then
        # `MakePolygonNormals` (:1538). So the frame changes UNDER the
        # topology: the polygon partition is decided in the file's frame and
        # only the normals are recomputed in the final one.
        #
        # That is not a detail. `MakePolygons` groups triangles by the
        # QUANTISED direction of their normal (`MeshPolygonKey`, 0.01 rad
        # buckets), and a rotation moves every normal across the bucket grid —
        # so the same hull merges into a DIFFERENT set of polygons in a
        # different frame. Measured by transcribing `MakePolygons` into Python
        # and running it on MuJoCo's own hull faces, over Menagerie's 882
        # meshes with a stored graph: the principal frame reproduces
        # `mesh_polyvertnum` on 160 of them and the file's frame on 484.
        #
        # `refpos`/`refquat`/`scale` are on the far side of that line too, so
        # the loader is always asked for UNSCALED vertices now and
        # `apply_mesh_ref_transform` runs below with the rest of the frame.
        var mesh_data = load_stl(mesh_filename, 1.0, 1.0, 1.0)

        # Extract positions from GPUVertex structs into flat array
        var raw = List[Scalar[DTYPE]]()
        var num_raw = len(mesh_data.vertices)
        for i in range(num_raw):
            raw.append(Scalar[DTYPE](mesh_data.vertices[i].px))
            raw.append(Scalar[DTYPE](mesh_data.vertices[i].py))
            raw.append(Scalar[DTYPE](mesh_data.vertices[i].pz))

        # Deduplicate into temp buffer
        var unique = List[Scalar[DTYPE]]()
        var num_unique = deduplicate_vertices[DTYPE](raw, num_raw, unique)

        # ⚠ BUILT INTO FRESH, EMPTY LISTS AT `vert_base = 0`. That is what
        # makes the payload position-independent: every `*_adr` these three
        # calls emit is `len()` of a list that started empty, and every global
        # vertex id in `edge_list` is `0 + w`. A cached mesh is therefore
        # exactly what a from-scratch build of a ONE-MESH model would produce,
        # and the append block below applies the shift `load_mesh_hull` has
        # always applied.
        var lvert = List[Scalar[DTYPE]]()
        var lfaces = List[Int]()
        # ⚠⚠ qhull FIRST, AND ITS WHOLE ANSWER. `MakeGraph` calls qhull; the
        # face ORDER it returns is what decides every merged polygon's normal
        # (`build_mesh_polygons` inserts in this sequence, `Paths` starts each
        # cycle at the first surviving edge of the first face inserted, and
        # `MakePolygonNormals` reads that cycle's first three vertices).
        # `compute_convex_hull` is the fallback for a tree without the dylib.
        var nh = _qhull_hull[DTYPE](
            unique, num_unique, maxhullvert, lvert, lfaces
        )
        var qhull_order = nh > 0
        if not qhull_order:
            # ⚠ THE FALLBACK IGNORES THE BUDGET, and says so. Without the
            # shim there is no qhull to hand `TA` to, and
            # `compute_convex_hull` is an exact incremental hull with no
            # notion of stopping early; a model that asked for 64 gets its
            # full hull. The alternative — trimming afterwards — would be a
            # DIFFERENT hull from MuJoCo's, not a closer one.
            if maxhullvert > -1:
                print(
                    "physics3d: <mesh maxhullvert=" + String(maxhullvert)
                    + "> ignored for " + mesh_filename
                    + " — the qhull shim is unavailable, so the full hull is"
                    " used. Build it with `pixi run build-qhull`."
                )
            nh = compute_convex_hull[DTYPE](
                unique, num_unique, lvert, lfaces
            )

        var lnormal = List[Scalar[DTYPE]]()
        var np_local = build_mesh_polygons[DTYPE](
            lvert,
            0,
            nh,
            lfaces,
            p.poly_vert,
            p.poly_vertadr,
            p.poly_vertnum,
            lnormal,
            p.polymap,
            p.polymap_adr,
            p.polymap_num,
        )
        build_hull_edge_graph(nh, lfaces, 0, p.edge_adr, p.edge_list)

        # ── and NOW the frame ────────────────────────────────────────────────
        # `ApplyTransformations` (refpos, then refquat's inverse, then scale),
        # then the CoM translation and the principal-axis rotation. Only the
        # hull's own vertices need it: `poly_vert`, `polymap` and `edge_list`
        # are indices and the rotation does not touch them.
        apply_mesh_ref_transform[DTYPE](
            lvert, nh, rpx, rpy, rpz, rqw, rqx, rqy, rqz, sx, sy, sz
        )
        transform_verts_to_principal_frame[DTYPE](lvert, nh, mi)

        # `MakePolygonNormals` (user_mesh.cc:2661) — recomputed from the FINAL
        # vertices. The normals `build_mesh_polygons` returned are the file
        # frame's and are discarded.
        #
        # ⚠⚠ THIS LOOP USED TO SPELL THE RULE OUT AGAIN, AND THAT IS WHY THE
        # DEGENERATE CASE SURVIVED A FIX. `mesh_polygons` had the same eight
        # lines; this copy is the one that survives, so correcting the other
        # one changed nothing measurable and looked like the fix had failed.
        # One callee now, and it is the module that owns the rule.
        for pi in range(np_local):
            # ⚠ MuJoCo's FIRST-THREE rule ONLY when the cycle start is
            # MuJoCo's. Without the shim the start is ours, and first-three on
            # the wrong start is measurably worse than the cyclic sum — see
            # `polygon_normal`.
            var wn = polygon_normal[DTYPE](
                lvert, 0, p.poly_vert, p.poly_vertadr[pi], p.poly_vertnum[pi],
                qhull_order,
            )
            lnormal[pi * 3 + 0] = wn[0]
            lnormal[pi * 3 + 1] = wn[1]
            lnormal[pi * 3 + 2] = wn[2]

        # ── THE TRIANGLE SOUP ────────────────────────────────────────────
        # The mesh's ORIGINAL triangles, which the hull is not. `mj_rayMesh`
        # walks `mesh_face`; a ray into a bracket's cutout must find the hole,
        # and the hull has none. Built HERE, inside the cached block, because
        # the frame below is the expensive, cache-worthy part and the soup has
        # to share it exactly: a triangle in the file's frame and a hull in the
        # principal frame describe two different objects.
        #
        # ⚠ `load_stl` returns a SOUP ALREADY — three vertices per triangle in
        # order, undeduplicated — so this is a copy, not a rebuild, and it
        # deliberately does not go through `deduplicate_vertices`. See
        # `HullPayload.tri_vert` for why indices were not worth their map.
        var tri = List[Scalar[DTYPE]]()
        for i in range(num_raw):
            tri.append(Scalar[DTYPE](mesh_data.vertices[i].px))
            tri.append(Scalar[DTYPE](mesh_data.vertices[i].py))
            tri.append(Scalar[DTYPE](mesh_data.vertices[i].pz))
        # ⚠ THE SAME TWO CALLS THE HULL GETS, IN THE SAME ORDER. Getting this
        # wrong is silent: the ray answer stays plausible and is wrong by the
        # mesh's centre-of-mass shift and principal rotation, which is exactly
        # the offset a "roughly right but drifting" rangefinder would show.
        apply_mesh_ref_transform[DTYPE](
            tri, num_raw, rpx, rpy, rpz, rqw, rqx, rqy, rqz, sx, sy, sz
        )
        transform_verts_to_principal_frame[DTYPE](tri, num_raw, mi)
        p.num_tri = num_raw // 3
        for i in range(len(tri)):
            p.tri_vert.append(Float64(tri[i]))

        p.num_hull = nh
        p.npoly = np_local
        p.rbound = Float64(compute_mesh_rbound_at[DTYPE](lvert, 0, nh))
        for i in range(len(lvert)):
            p.hull_vert.append(Float64(lvert[i]))
        for i in range(len(lnormal)):
            p.poly_normal.append(Float64(lnormal[i]))

        hull_cache_store(cache_path, p)

    var mesh_id = num_meshes
    # ⚠ TWO UNITS, ONE NAME. `mesh_vert` is a FLAT scalar list, so
    # `len(mesh_vert)` is an offset in FLOATS — but every collision consumer
    # indexes the packed `mesh_verts` tensor as `[vertex, component]`:
    # `mesh_verts[vert_adr + i, k]` in `gjk._support_mesh`, in
    # `_plane_mesh_contacts` and in the SAP plane-mesh branch. Storing the
    # float offset made `mesh_vertadr` 3x too large for all three.
    #
    # Measured on sawyer: mesh 11 (eGripperBase) had vertadr 1701, which as a
    # VERTEX index points past the 648 vertices actually loaded, so every read
    # returned ZERO — the gripper hull collided as an empty shape. Meshes 1..10
    # were worse than useless rather than empty: their float offsets land
    # inside the populated region, so they collided against SOME OTHER MESH'S
    # vertices. Only mesh 0, at offset 0, was ever right.
    #
    # `mesh_vertadr` is now a VERTEX index, which is also MuJoCo's convention
    # for `mesh_vertadr`. `compute_mesh_rbound_at` walks the flat list and
    # still wants FLOATS, so it is given the float offset explicitly.
    # ── Append into the model-wide arrays ────────────────────────────────────
    # ⚠⚠ THIS IS THE REBASING, AND IT IS WHERE A CACHE BUG WOULD HIDE. `p`
    # holds offsets relative to zero; the shared arrays already contain every
    # earlier mesh. The three `*_adr` arrays are offsets and shift; `poly_vert`
    # and `polymap` hold LOCAL ids and do NOT; `edge_list` holds GLOBAL vertex
    # ids and shifts by the vertex base, with -1 terminators passed through.
    # The table in `hull_cache.mojo` is the reference, and
    # `test_hull_cache.mojo` compares a cold build against a warm one for
    # exactly this block.
    var vert_float_offset = len(mesh_vert)
    var vert_base = vert_float_offset // 3
    mesh_vertadr.append(vert_base)
    # ⚠⚠ ROUNDED TO float32, BECAUSE MUJOCO'S `mjModel.mesh_vert` IS `float*`.
    # Its compiler does every mesh step in double — scale, recentre, hull — and
    # then copies the result into a FLOAT array, which is what every collision
    # routine reads (`mjc_initCCDObj` hands `m->mesh_vert` straight to the
    # support function). Keeping our copy in double leaves the hull a few
    # hundred picometres away from the one the reference collides with.
    #
    # ⚠ THAT IS ENOUGH TO CREATE OR DESTROY A CONTACT. rby1's drive wheels are
    # modelled EXACTLY tangent to the floor: MuJoCo's lowest hull vertex lands
    # at world z = +6.372e-11 and ours at -7.451e-10, so we opened two contacts
    # it does not have and the robot diverged 4.99e-03 in one step. The two
    # differ by ONE float32 ulp at 0.1 — `float32(ours) == MuJoCo's` exactly,
    # on every coordinate.
    #
    # ⚠ AFTER THE HULL, NOT BEFORE. MuJoCo picks hull membership from the
    # double vertices and only the STORED coordinates are float; rounding
    # earlier could change which vertices survive.
    for i in range(len(p.hull_vert)):
        mesh_vert.append(
            Scalar[DTYPE](p.hull_vert[i].cast[DType.float32]())
        )
    var num_hull = p.num_hull
    mesh_vertnum.append(num_hull)

    # The soup, rounded to float32 for the same reason the hull is: MuJoCo's
    # `mesh_vert` is `float*` and `ray_triangle` reads those floats. A double
    # copy would put our surface a few hundred picometres from the one the
    # reference intersects, which on a grazing ray is the difference between a
    # hit and a miss.
    mesh_triadr.append(len(mesh_tri) // 9)
    for i in range(len(p.tri_vert)):
        mesh_tri.append(Scalar[DTYPE](p.tri_vert[i].cast[DType.float32]()))
    mesh_trinum.append(p.num_tri)

    num_meshes += 1

    # Polygons for the native multi-contact path. `polymap_adr` / `polymap_num`
    # are appended one entry per HULL VERTEX, so they stay parallel to the
    # vertex block this mesh just wrote.
    mesh_polyadr.append(len(poly_vertadr))
    var poly_vert_base = len(poly_vert)
    for i in range(p.npoly):
        poly_vertadr.append(p.poly_vertadr[i] + poly_vert_base)
        poly_vertnum.append(p.poly_vertnum[i])
    for i in range(len(p.poly_vert)):
        poly_vert.append(p.poly_vert[i])  # LOCAL vertex id — no shift
    for i in range(len(p.poly_normal)):
        poly_normal.append(Scalar[DTYPE](p.poly_normal[i]))
    mesh_polynum.append(p.npoly)

    var polymap_base = len(polymap)
    for i in range(num_hull):
        polymap_adr.append(p.polymap_adr[i] + polymap_base)
        polymap_num.append(p.polymap_num[i])
    for i in range(len(p.polymap)):
        polymap.append(p.polymap[i])  # LOCAL polygon id — no shift

    # Vertex adjacency for the plane-mesh path. Built from the SAME triangles
    # the polygons were merged from, and indexed by global vertex id, so it
    # stays parallel to the vertex block written above.
    # The hill climb's seed table goes FIRST, so it ends where this mesh's
    # neighbour lists begin — see `mesh_seed.mojo` for the layout.
    append_mesh_seed_table[DTYPE](
        mesh_vert, vert_float_offset, num_hull, edge_list
    )
    var edge_list_base = len(edge_list)
    for i in range(num_hull):
        edge_adr.append(p.edge_adr[i] + edge_list_base)
    for i in range(len(p.edge_list)):
        var e = p.edge_list[i]
        # ⚠ -1 IS THE TERMINATOR, NOT A VERTEX. Shifting it would turn every
        # per-vertex neighbour walk into a run off the end of the list.
        edge_list.append(e if e < 0 else e + vert_base)

    var rbound = Scalar[DTYPE](p.rbound)

    return (mesh_id, rbound)
