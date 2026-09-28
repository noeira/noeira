"""`mj_rayMesh` — ray against a mesh's ORIGINAL triangles, two ways.

`engine_ray.c:952`, which is a bounding-box reject followed by `mju_rayTree`
(`:771`) — a BVH walk whose every leaf runs `ray_triangle` and keeps the
nearest. **The BVH is an acceleration structure and nothing else**: culling a
node removes only triangles the ray provably misses, so a linear sweep over the
same triangles returns the identical distance and normal.

⚠⚠ SO BOTH ARE HERE, AND THAT IS DELIBERATE. `ray_mesh` is the linear sweep and
`ray_mesh_bvh` the tree; they sit in one file because their agreement is the
whole contract and a reader checking it should not have to open two.
`ray_model` dispatches on `MESH_META_IDX_BVHNUM` and
`tests/physics3d/test_ray_bvh_matches_linear.mojo` gates the pair, on the host
AND on the device, by EXACT equality over colour, depth and geom id. The tree was
worth 11.2x at 1 024 lanes on Apple silicon — `bvh_x` in
`benchmarks/camera_tracer_lane_sweep.mojo`, which times both legs in one binary
— and it must stay worth exactly 0 digits.

⚠⚠ THE HULL CANNOT ANSWER THIS QUERY, which is why `Model.mesh_tris` had to
exist. `Model.mesh_verts` is the CONVEX HULL — right for collision, and MuJoCo
collides hulls too — but a ray aimed into a bracket's cutout hits hull where
the real part has a hole. The three consumers of this package (the
`rangefinder` sensor, studio picking, and a camera) all care about the hole.

⚠ THE SOUP IS OFF BY DEFAULT. `nmesh_tri` defaults to 0, so a model built
without a triangle budget carries no soup and `ntri` here is 0 — and this
returns NO HIT for every ray. That is a capacity being read as an absence of
geometry, so `ntri` is an explicit parameter rather than something inferred
from a table: a caller that forgets it gets an obviously empty answer at the
call site rather than a silently invisible mesh.
"""

from layout import Layout, LayoutTensor

from noeira.math3d import Vec3 as Vec3Generic, Quat as QuatGeneric

from ..gpu.constants import (
    MESH_ARENA_RECORD,
    MESH_BVH_IDX_CX,
    MESH_BVH_IDX_CY,
    MESH_BVH_IDX_CZ,
    MESH_BVH_IDX_HX,
    MESH_BVH_IDX_HY,
    MESH_BVH_IDX_HZ,
    MESH_BVH_IDX_ESCAPE,
    MESH_BVH_IDX_TRI,
)
from .geom import RAY_NO_HIT, ray_map, ray_box
from .triangle import ray_basis, ray_triangle


@always_inline
def _v3[
    DTYPE: DType, L_TRI: Layout
](
    tri: LayoutTensor[DTYPE, L_TRI, MutAnyOrigin], o: Int
) -> Vec3Generic[DTYPE]:
    """Three consecutive floats as a vertex."""
    return Vec3Generic[DTYPE](
        rebind[Scalar[DTYPE]](tri[o + 0]),
        rebind[Scalar[DTYPE]](tri[o + 1]),
        rebind[Scalar[DTYPE]](tri[o + 2]),
    )


@fieldwise_init
struct MeshHit[DTYPE: DType](Copyable, ImplicitlyCopyable, Movable):
    """`ray_mesh`'s answer, plus WHICH triangle and where on it.

    `tri` is the winning triangle's ARENA RECORD number — the same index
    `MESH_BVH_IDX_TRI` stores and the same one `MESH_META_IDX_TRIADR` counts
    from — so a parallel per-triangle table (`VisualModel.mesh_uv`) is indexed
    by it directly. -1 on a miss.

    ⚠ THE NORMAL IS WORLD-FRAME AND THE BARYCENTRICS ARE NOT A FRAME AT ALL.
    `bu`/`bv` weight `v0`/`v1` of that triangle; they survive the rotation
    because they are weights, not directions.
    """

    var t: Scalar[Self.DTYPE]
    var normal: Vec3Generic[Self.DTYPE]
    var tri: Int
    var bu: Scalar[Self.DTYPE]
    var bv: Scalar[Self.DTYPE]


def ray_mesh[
    DTYPE: DType, L_TRI: Layout
](
    pos: Vec3Generic[DTYPE],
    quat: QuatGeneric[DTYPE],
    half_extents: Vec3Generic[DTYPE],
    tri: LayoutTensor[DTYPE, L_TRI, MutAnyOrigin],
    triadr: Int,
    ntri: Int,
    pnt: Vec3Generic[DTYPE],
    vec: Vec3Generic[DTYPE],
) -> MeshHit[DTYPE] where DTYPE.is_floating_point():
    """Distance to the mesh surface, its world-frame normal, and the triangle.

    `tri` is `Model.mesh_tris` — nine floats per triangle, in the mesh's
    principal frame, the same frame the geom's pose assumes. `triadr` is this
    mesh's first TRIANGLE (not float, not vertex) and `ntri` its count.

    `half_extents` is the geom's bounding box, `mjModel.geom_size` for a mesh.
    ⚠ IT IS A PARAMETER RATHER THAN A DERIVED VALUE ON PURPOSE. The reference
    rejects on it before touching a triangle, so a box that is too SMALL
    silently loses hits — and whether our `geom_size` for a mesh is MuJoCo's
    AABB is a question for the gate to answer, not for this routine to assume.
    Pass a box you trust; passing one that is too large only costs time.
    """
    var zero = Vec3Generic[DTYPE](0, 0, 0)
    var miss = MeshHit[DTYPE](
        Scalar[DTYPE](RAY_NO_HIT), zero, -1, Scalar[DTYPE](0), Scalar[DTYPE](0)
    )
    if ntri <= 0:
        return miss

    # The reference's bounding-box reject, in the geom's own frame.
    var bb = ray_box[DTYPE](pos, quat, half_extents, pnt, vec)
    if bb[0] < 0:
        return miss

    var m = ray_map[DTYPE](pos, quat, pnt, vec)
    var lpnt = m[0]
    var lvec = m[1]
    var basis = ray_basis[DTYPE](lvec)
    var b0 = basis[0]
    var b1 = basis[1]

    var x = Scalar[DTYPE](RAY_NO_HIT)
    var normal_local = zero
    var best_tri = -1
    var best_bu = Scalar[DTYPE](0)
    var best_bv = Scalar[DTYPE](0)

    for t in range(ntri):
        var o = (triadr + t) * MESH_ARENA_RECORD
        var r = ray_triangle[DTYPE](
            _v3[DTYPE, L_TRI](tri, o + 0),
            _v3[DTYPE, L_TRI](tri, o + 3),
            _v3[DTYPE, L_TRI](tri, o + 6),
            lpnt, lvec, b0, b1,
        )
        # ⚠ `>= 0` AND `< x`, in that order, matching the reference. A
        # triangle behind the ray comes back NEGATIVE from `ray_triangle`
        # (the plane intersection is deliberately unclamped) and would
        # otherwise win every comparison against a positive distance.
        if r.t >= 0 and (x < 0 or r.t < x):
            x = r.t
            normal_local = r.normal
            best_tri = triadr + t
            best_bu = r.bu
            best_bv = r.bv

    if x < 0:
        return miss
    return MeshHit[DTYPE](
        x, quat.rotate_vec(normal_local), best_tri, best_bu, best_bv
    )


# ─── the BVH leg ─────────────────────────────────────────────────────────────


@always_inline
def _inv_safe[
    DTYPE: DType
](d: Scalar[DTYPE]) -> Scalar[DTYPE] where DTYPE.is_floating_point():
    """`1/d`, kept FINITE — and the reason is a lost hit, not an aesthetic.

    The reference's `mju_raySlab` divides by the ray direction and lets a zero
    component produce an infinity. That is fine until the ray origin sits
    exactly on a slab plane, where `(min[d] - src[d]) * invdir[d]` is
    `0 * inf` = **NaN**; every comparison against a NaN is false, `tmin < tmax`
    comes back false, and the node — with the triangle the ray actually hits
    inside it — is CULLED. A linear sweep has no such branch, so that would be
    an acceleration structure changing the answer, which is the one thing this
    file's invariant forbids.

    Clamping the magnitude instead keeps every product finite. It only ever
    makes a slab wider than the reference's would be, so it can add work and
    can never remove a hit.
    """
    comptime BIG = Scalar[DTYPE](1e20)
    comptime TINY = Scalar[DTYPE](1e-20)
    if d > TINY or d < -TINY:
        return Scalar[DTYPE](1) / d
    if d < 0:
        return -BIG
    return BIG


def ray_mesh_bvh[
    DTYPE: DType, L_TRI: Layout
](
    pos: Vec3Generic[DTYPE],
    quat: QuatGeneric[DTYPE],
    half_extents: Vec3Generic[DTYPE],
    tri: LayoutTensor[DTYPE, L_TRI, MutAnyOrigin],
    triadr: Int,
    ntri: Int,
    bvhadr: Int,
    bvhnum: Int,
    pnt: Vec3Generic[DTYPE],
    vec: Vec3Generic[DTYPE],
    tcut: Scalar[DTYPE] = Scalar[DTYPE](RAY_NO_HIT),
) -> MeshHit[DTYPE] where DTYPE.is_floating_point():
    """`mju_rayTree` — the same answer as `ray_mesh`, over a BVH.

    ⚠⚠ THE CONTRACT IS BIT-IDENTITY WITH `ray_mesh`, NOT APPROXIMATION. Culling
    a node removes only triangles the ray provably misses, so the surviving
    `ray_triangle` calls are a SUBSET of the linear sweep's and the winner is
    the same triangle at the same distance. `test_ray_bvh_matches_linear.mojo`
    holds that, and every deviation below — the finite reciprocal, the padded
    boxes, the `tmin` early-out — is argued at its site as one that can only
    add work.

    ⚠⚠ NO STACK, AND THAT IS THE PORT'S ONE REAL DEPARTURE. The reference
    keeps `int stack[mjMAXTREEDEPTH]` and pushes both children; that is a
    per-thread array indexed at a RUNTIME value, which reads back silently
    wrong on Metal and has done four times in this engine. The obvious
    substitute — a `[thread, depth]` scratch `LayoutTensor` — is worse than it
    looks: this kernel runs one thread per (env, pixel), so 1024 lanes of
    84x84 at a 32-deep stack of 4-byte slots is **925 MB** of scratch, for a
    tree whose depth is ~13. So the tree is stored in PRE-ORDER with an ESCAPE
    index per node instead (`MESH_BVH_IDX_ESCAPE`): the left child is always
    the next record, and a miss jumps to the end of the subtree. Traversal is
    then two integers in registers and holds for any depth.

    ⚠⚠ `tcut` IS THE CALLER'S BEST HIT SO FAR (`ray_model` passes the nearest
    distance among the geoms it has already tested; negative = none). A
    triangle at `t >= tcut` cannot win there — `ray_model` keeps the FIRST
    geom on a tie, strictly — so this walk drops it, and a node whose box
    starts at or beyond `tcut` is not entered. MuJoCo Warp does the same
    (`mesh_query_ray(..., max_t)`); the reference walks every mesh to its own
    nearest hit. A pruned call returns NO HIT where the uncut one would have
    returned a farther hit, and `ray_model` would have discarded that hit:
    the model's answer is unchanged, which is what the gate checks.

    ⚠⚠ TIES INSIDE THE MESH GO TO THE LOWER TRIANGLE INDEX, as in the linear
    sweep (which scans in index order and keeps the first strictly nearer).
    Without the rule the winner of an exact tie — a ray through a shared edge
    — depends on the TREE's visiting order, so a different build (SAH vs the
    reference's median split) could return the other face, with its own
    normal and shading. With it, any valid tree returns the sweep's triangle,
    and the node test must admit `tmin == x` (a tied, lower-index triangle can
    sit in a box that starts exactly at `x`).

    `bvhadr`/`bvhnum` are the mesh's node window in the same arena as `tri`,
    in RECORDS. `bvhnum <= 0` means no tree was built — the caller is expected
    to have taken the `ray_mesh` leg instead, and this returns no hit rather
    than guessing.
    """
    var zero = Vec3Generic[DTYPE](0, 0, 0)
    var miss = MeshHit[DTYPE](
        Scalar[DTYPE](RAY_NO_HIT), zero, -1, Scalar[DTYPE](0), Scalar[DTYPE](0)
    )
    if ntri <= 0 or bvhnum <= 0:
        return miss

    # The reference's bounding-box reject, in the geom's own frame. Kept even
    # though the root node is a tighter box: `half_extents` is the geom's
    # `geom_size`, the ONE reject the reference does before the tree, and
    # dropping it here would make the two legs disagree on a mesh whose
    # `geom_size` is smaller than its own triangles (see `ray_mesh`).
    var bb = ray_box[DTYPE](pos, quat, half_extents, pnt, vec)
    if bb[0] < 0:
        return miss

    var m = ray_map[DTYPE](pos, quat, pnt, vec)
    var lpnt = m[0]
    var lvec = m[1]
    var basis = ray_basis[DTYPE](lvec)
    var b0 = basis[0]
    var b1 = basis[1]

    var ivx = _inv_safe[DTYPE](lvec.x)
    var ivy = _inv_safe[DTYPE](lvec.y)
    var ivz = _inv_safe[DTYPE](lvec.z)

    var x = Scalar[DTYPE](RAY_NO_HIT)
    var normal_local = zero
    var best_tri = -1
    var best_bu = Scalar[DTYPE](0)
    var best_bv = Scalar[DTYPE](0)

    comptime FAR = Scalar[DTYPE](1e30)
    var cut = tcut if tcut >= Scalar[DTYPE](0) else FAR
    var node = bvhadr
    var stop = bvhadr + bvhnum

    while node < stop:
        var o = node * MESH_ARENA_RECORD

        # ── mju_raySlab, with the ray already in the mesh frame ───────────
        #
        # ⚠ THE REFERENCE RE-MAPS THE RAY PER NODE. `mju_raySlab` takes
        # `xpos`/`xmat` and calls `ray_map` itself, once for every node it
        # visits; the mapping does not depend on the node, so it is hoisted
        # above the loop here. Same arithmetic, ~13x fewer quaternion
        # rotations on a tree this deep.
        var cx = rebind[Scalar[DTYPE]](tri[o + MESH_BVH_IDX_CX])
        var cy = rebind[Scalar[DTYPE]](tri[o + MESH_BVH_IDX_CY])
        var cz = rebind[Scalar[DTYPE]](tri[o + MESH_BVH_IDX_CZ])
        var hx = rebind[Scalar[DTYPE]](tri[o + MESH_BVH_IDX_HX])
        var hy = rebind[Scalar[DTYPE]](tri[o + MESH_BVH_IDX_HY])
        var hz = rebind[Scalar[DTYPE]](tri[o + MESH_BVH_IDX_HZ])

        var t1 = (cx - hx - lpnt.x) * ivx
        var t2 = (cx + hx - lpnt.x) * ivx
        var tmin = t1 if t1 < t2 else t2
        var tmax = t2 if t1 < t2 else t1

        t1 = (cy - hy - lpnt.y) * ivy
        t2 = (cy + hy - lpnt.y) * ivy
        var lo = t1 if t1 < t2 else t2
        var hi = t2 if t1 < t2 else t1
        tmin = tmin if tmin > lo else lo
        tmax = tmax if tmax < hi else hi

        t1 = (cz - hz - lpnt.z) * ivz
        t2 = (cz + hz - lpnt.z) * ivz
        lo = t1 if t1 < t2 else t2
        hi = t2 if t1 < t2 else t1
        tmin = tmin if tmin > lo else lo
        tmax = tmax if tmax < hi else hi

        # `tmin` starts at 0 in the reference — a box entirely BEHIND the ray
        # is a miss, not a hit at a negative distance.
        if tmin < 0:
            tmin = 0
        if tmax > FAR:
            tmax = FAR

        # ⚠ `tmin <= x` IS AN EARLY-OUT THE REFERENCE DOES NOT HAVE, and it
        # is safe for a reason worth stating: every point of every triangle
        # under this node lies inside the node's box, so nothing under it can
        # be nearer than the box's own entry distance. If we already hold a
        # hit at `x < tmin`, the whole subtree is provably not the winner.
        # `<=`, not `<`: a triangle at exactly `x` with a LOWER index still
        # wins the tie (see the docstring), and it can lie in a box entered at
        # `x`. `tmin < cut` is the caller's best hit, strict for the same
        # reason `ray_model`'s comparison is. (The boxes are also PADDED
        # outwards by the builder, which makes `tmin` an under-estimate — the
        # conservative direction.)
        var visit = tmin < tmax and tmin < cut and (x < 0 or tmin <= x)

        if visit:
            var tid = Int(rebind[Scalar[DTYPE]](tri[o + MESH_BVH_IDX_TRI]))
            if tid >= 0:
                var to = tid * MESH_ARENA_RECORD
                var r = ray_triangle[DTYPE](
                    _v3[DTYPE, L_TRI](tri, to + 0),
                    _v3[DTYPE, L_TRI](tri, to + 3),
                    _v3[DTYPE, L_TRI](tri, to + 6),
                    lpnt, lvec, b0, b1,
                )
                # The linear sweep's two tests, then the index tie-break the
                # sweep gets from its scan order, then the caller's cut.
                if (
                    r.t >= 0
                    and r.t < cut
                    and (x < 0 or r.t < x or (r.t == x and tid < best_tri))
                ):
                    x = r.t
                    normal_local = r.normal
                    best_tri = tid
                    best_bu = r.bu
                    best_bv = r.bv
            # A leaf's escape IS `node + 1`, so one branch serves both.
            node += 1
        else:
            var nxt = Int(rebind[Scalar[DTYPE]](tri[o + MESH_BVH_IDX_ESCAPE]))
            # ⚠ A GUARD AGAINST A HANG, NOT AGAINST A WRONG ANSWER. `escape`
            # is strictly increasing by construction; a zero here would be an
            # arena that was allocated but never built, and a GPU thread
            # spinning forever is a hung queue rather than a bad pixel.
            if nxt <= node:
                break
            node = nxt

    if x < 0:
        return miss
    return MeshHit[DTYPE](
        x, quat.rotate_vec(normal_local), best_tri, best_bu, best_bv
    )
