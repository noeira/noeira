"""Build the per-mesh BVH that `ray/mesh.mojo::ray_mesh_bvh` walks.

`mjCBoundingVolumeHierarchy::MakeBVH` (`user_objects.cc:424`) is a median
split along the node's longest axis with ONE element per leaf. `sah=False`
builds exactly that, with two changes stated where they are made: the nodes
come out in PRE-ORDER carrying an ESCAPE index instead of two child pointers
(so the traversal needs no stack — see `ray_mesh_bvh`), and every box is
padded outwards by a scale-relative epsilon rather than only the flat ones.

⚠⚠ THE DEFAULT IS `sah=True`: the split plane is chosen by a BINNED SURFACE
AREA HEURISTIC (16 bins per axis over the node's centroid bounds, cost
`area(L) * nL + area(R) * nR`), still one triangle per leaf, so the record
format, the `2n - 1` node count and the walk are unchanged. MuJoCo Warp builds
its mesh trees with SAH / cuBQL for the same reason: a median split cuts
through the middle of a part whatever its shape, and on the SO-101 arm's STLs
(320k triangles) the camera rays that end on the DESK were spending ~70 % of
the frame walking mesh trees they never hit
(`noeira-docs/SO101_RENDER_SPEED.md`). A split that fails to separate
anything (every centroid in one bin, or a degenerate cost) falls back to the
median split, so the build always terminates with `2n - 1` nodes.

⚠⚠ HOST CODE. It builds with `List` and recursion, runs once per model load,
and nothing here is ever compiled into a kernel. It lives in `parser/` rather
than next to the traversal for exactly that reason.

⚠⚠ THE TREE IS NOT PART OF THE ANSWER. A BVH culls only triangles the ray
provably misses, so ANY valid tree over the same triangles gives the distance
and normal the linear sweep gives. That is the licence for the departures
above, and it is why the gate
(`tests/physics3d/test_ray_bvh_matches_linear.mojo`) compares against
`ray_mesh` rather than against MuJoCo's node table: matching the reference's
TREE would be a much stronger claim than the one that matters, and the
reference's `nth_element` does not even define its own partition uniquely.
"""

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


def _nth_element[
    DTYPE: DType
](
    mut order: List[Int],
    beg: Int,
    end: Int,
    k: Int,
    cen: List[Scalar[DTYPE]],
    axis: Int,
):
    """Partition `order[beg:end)` so `order[k]` is where a full sort would put
    it — Hoare quickselect, the reference's `std::nth_element`.

    ⚠ A FULL SORT WOULD ALSO BE CORRECT AND IS THE WRONG COST. The split only
    needs the median's POSITION; sorting every level turns an O(n log n) build
    into O(n log^2 n), which on a 100k-triangle mesh is the difference between
    a model that loads and one that appears to hang.

    Ties are safe: with every key equal the inner scans stop immediately, the
    swap still advances both cursors, and the loop terminates on a balanced
    (if spatially arbitrary) split.
    """
    var l = beg
    var r = end - 1
    while l < r:
        var pivot = cen[order[(l + r) // 2] * 3 + axis]
        var i = l
        var j = r
        while i <= j:
            while cen[order[i] * 3 + axis] < pivot:
                i += 1
            while cen[order[j] * 3 + axis] > pivot:
                j -= 1
            if i <= j:
                var tmp = order[i]
                order[i] = order[j]
                order[j] = tmp
                i += 1
                j -= 1
        if k <= j:
            r = j
        elif k >= i:
            l = i
        else:
            return


comptime SAH_BINS: Int = 16


@always_inline
def _area(
    mnx: Float64, mny: Float64, mnz: Float64,
    mxx: Float64, mxy: Float64, mxz: Float64,
) -> Float64:
    """Half the surface area of a box — the SAH only compares, so the factor
    of two is dropped."""
    var ex = mxx - mnx
    var ey = mxy - mny
    var ez = mxz - mnz
    if ex < 0 or ey < 0 or ez < 0:
        return 0.0
    return ex * ey + ey * ez + ez * ex


def _sah_partition[
    DTYPE: DType
](
    mut order: List[Int],
    beg: Int,
    end: Int,
    lo: List[Scalar[DTYPE]],
    hi: List[Scalar[DTYPE]],
    cen: List[Scalar[DTYPE]],
) -> Int:
    """Partition `order[beg:end)` by the best binned-SAH plane and return the
    SIZE of the left part, or -1 when no plane separates the elements (the
    caller then takes the median split).

    Binned on the CENTROIDS' bounds, costed on the triangles' boxes. Every
    candidate plane is a bin boundary, so both sides are non-empty by
    construction whenever the chosen boundary has elements on each side —
    which the cost sweep only considers.
    """
    var n = end - beg
    var best_cost = Float64(1e300)
    var best_axis = -1
    var best_split = -1
    var best_cmin = 0.0
    var best_scale = 0.0

    for axis in range(3):
        var cmin = Float64(cen[order[beg] * 3 + axis])
        var cmax = cmin
        for e in range(beg + 1, end):
            var c = Float64(cen[order[e] * 3 + axis])
            if c < cmin:
                cmin = c
            if c > cmax:
                cmax = c
        if not (cmax > cmin):
            continue
        var scale = Float64(SAH_BINS) / (cmax - cmin)

        var cnt = List[Int](length=SAH_BINS, fill=0)
        var bmn = List[Float64](length=SAH_BINS * 3, fill=1e300)
        var bmx = List[Float64](length=SAH_BINS * 3, fill=-1e300)
        for e in range(beg, end):
            var t = order[e]
            var b = Int((Float64(cen[t * 3 + axis]) - cmin) * scale)
            if b >= SAH_BINS:
                b = SAH_BINS - 1
            if b < 0:
                b = 0
            cnt[b] += 1
            for a in range(3):
                var l = Float64(lo[t * 3 + a])
                var h = Float64(hi[t * 3 + a])
                if l < bmn[b * 3 + a]:
                    bmn[b * 3 + a] = l
                if h > bmx[b * 3 + a]:
                    bmx[b * 3 + a] = h

        # Suffix sweep: the right side's area and count for every plane.
        var r_area = List[Float64](length=SAH_BINS, fill=0.0)
        var r_cnt = List[Int](length=SAH_BINS, fill=0)
        var rmn = List[Float64](length=3, fill=1e300)
        var rmx = List[Float64](length=3, fill=-1e300)
        var rc = 0
        for b in range(SAH_BINS - 1, 0, -1):
            rc += cnt[b]
            if cnt[b] > 0:
                for a in range(3):
                    if bmn[b * 3 + a] < rmn[a]:
                        rmn[a] = bmn[b * 3 + a]
                    if bmx[b * 3 + a] > rmx[a]:
                        rmx[a] = bmx[b * 3 + a]
            r_cnt[b] = rc
            r_area[b] = _area(rmn[0], rmn[1], rmn[2], rmx[0], rmx[1], rmx[2])

        # Prefix sweep: plane `b` puts bins [0, b) left and [b, BINS) right.
        var lmn = List[Float64](length=3, fill=1e300)
        var lmx = List[Float64](length=3, fill=-1e300)
        var lc = 0
        for b in range(1, SAH_BINS):
            lc += cnt[b - 1]
            if cnt[b - 1] > 0:
                for a in range(3):
                    if bmn[(b - 1) * 3 + a] < lmn[a]:
                        lmn[a] = bmn[(b - 1) * 3 + a]
                    if bmx[(b - 1) * 3 + a] > lmx[a]:
                        lmx[a] = bmx[(b - 1) * 3 + a]
            if lc == 0 or r_cnt[b] == 0:
                continue
            var cost = (
                _area(lmn[0], lmn[1], lmn[2], lmx[0], lmx[1], lmx[2])
                * Float64(lc)
                + r_area[b] * Float64(r_cnt[b])
            )
            if cost < best_cost:
                best_cost = cost
                best_axis = axis
                best_split = b
                best_cmin = cmin
                best_scale = scale

    if best_axis < 0:
        return -1

    # In-place partition on the chosen plane, with the SAME bin arithmetic
    # as the counting pass, so the sizes agree exactly.
    var i = beg
    var j = end - 1
    while i <= j:
        var bi = Int((Float64(cen[order[i] * 3 + best_axis]) - best_cmin) * best_scale)
        if bi >= SAH_BINS:
            bi = SAH_BINS - 1
        if bi < best_split:
            i += 1
        else:
            var tmp = order[i]
            order[i] = order[j]
            order[j] = tmp
            j -= 1
    var nl = i - beg
    if nl <= 0 or nl >= n:
        return -1
    return nl


def _build_subtree[
    DTYPE: DType
](
    mut nodes: List[Scalar[DTYPE]],
    mut order: List[Int],
    lo: List[Scalar[DTYPE]],
    hi: List[Scalar[DTYPE]],
    cen: List[Scalar[DTYPE]],
    beg: Int,
    end: Int,
    node_base: Int,
    triadr: Int,
    pad: Scalar[DTYPE],
    sah: Bool,
) raises -> Int:
    """Emit the subtree over `order[beg:end)` and return its ROOT record.

    ⚠ PRE-ORDER IS LOAD-BEARING. The node is appended BEFORE its children, so
    a node's left child is always the record after it and the traversal never
    needs a left pointer. `escape` is then just "the next free record once
    both children are done", which is what the recursion returns to.
    """
    var index = node_base + len(nodes) // MESH_ARENA_RECORD
    for _ in range(MESH_ARENA_RECORD):
        nodes.append(Scalar[DTYPE](0))
    var slot = len(nodes) - MESH_ARENA_RECORD

    var mnx = lo[order[beg] * 3 + 0]
    var mny = lo[order[beg] * 3 + 1]
    var mnz = lo[order[beg] * 3 + 2]
    var mxx = hi[order[beg] * 3 + 0]
    var mxy = hi[order[beg] * 3 + 1]
    var mxz = hi[order[beg] * 3 + 2]
    for e in range(beg + 1, end):
        var t = order[e]
        if lo[t * 3 + 0] < mnx:
            mnx = lo[t * 3 + 0]
        if lo[t * 3 + 1] < mny:
            mny = lo[t * 3 + 1]
        if lo[t * 3 + 2] < mnz:
            mnz = lo[t * 3 + 2]
        if hi[t * 3 + 0] > mxx:
            mxx = hi[t * 3 + 0]
        if hi[t * 3 + 1] > mxy:
            mxy = hi[t * 3 + 1]
        if hi[t * 3 + 2] > mxz:
            mxz = hi[t * 3 + 2]

    var half = Scalar[DTYPE](0.5)
    nodes[slot + MESH_BVH_IDX_CX] = half * (mnx + mxx)
    nodes[slot + MESH_BVH_IDX_CY] = half * (mny + mxy)
    nodes[slot + MESH_BVH_IDX_CZ] = half * (mnz + mxz)
    # ⚠⚠ THE PAD IS NOT COSMETIC AND IT IS NOT MuJoCo's. The reference inflates
    # a box only when an axis is FLAT, by `mjEPS` = 1e-14 — a number below the
    # float32 noise floor at any real coordinate, so in float32 it inflates
    # nothing. A flat box then fails `tmin < tmax` (which is STRICT), and an
    # axis-aligned triangle — a floor tile, a machined face — gets culled from
    # a ray that hits it. Padding every box by a scale-relative amount is the
    # conservative direction: a box that is slightly too LARGE costs a
    # triangle test, a box that is slightly too small loses a hit.
    nodes[slot + MESH_BVH_IDX_HX] = half * (mxx - mnx) + pad
    nodes[slot + MESH_BVH_IDX_HY] = half * (mxy - mny) + pad
    nodes[slot + MESH_BVH_IDX_HZ] = half * (mxz - mnz) + pad

    if end - beg == 1:
        nodes[slot + MESH_BVH_IDX_ESCAPE] = Scalar[DTYPE](index + 1)
        nodes[slot + MESH_BVH_IDX_TRI] = Scalar[DTYPE](triadr + order[beg])
        return index

    var ex = mxx - mnx
    var ey = mxy - mny
    var ez = mxz - mnz
    var axis = 0
    var longest = ex
    if ey >= longest:
        axis = 1
        longest = ey
    if ez >= longest:
        axis = 2

    var m = -1
    if sah:
        m = _sah_partition[DTYPE](order, beg, end, lo, hi, cen)
    if m < 0:
        m = (end - beg) // 2
        _nth_element[DTYPE](order, beg, end, beg + m, cen, axis)
    _ = _build_subtree[DTYPE](
        nodes, order, lo, hi, cen, beg, beg + m, node_base, triadr, pad, sah
    )
    _ = _build_subtree[DTYPE](
        nodes, order, lo, hi, cen, beg + m, end, node_base, triadr, pad, sah
    )

    nodes[slot + MESH_BVH_IDX_ESCAPE] = Scalar[DTYPE](
        node_base + len(nodes) // MESH_ARENA_RECORD
    )
    nodes[slot + MESH_BVH_IDX_TRI] = Scalar[DTYPE](-1)
    return index


def build_mesh_bvh[
    DTYPE: DType
](
    mesh_tri: List[Scalar[DTYPE]],
    mesh_triadr: List[Int],
    mesh_trinum: List[Int],
    mut nodes: List[Scalar[DTYPE]],
    mut bvhadr: List[Int],
    mut bvhnum: List[Int],
    sah: Bool = True,
) raises:
    """One tree per mesh, appended to `nodes` in the arena's record format.

    `sah=False` is the reference's median split (see the module header); both
    give the same ray answers — the gate is
    `tests/physics3d/test_ray_bvh_matches_linear.mojo`.

    `mesh_tri` is the flat triangle soup, nine floats each, exactly as it goes
    into `Model.mesh_tris`; `mesh_triadr`/`mesh_trinum` are its per-mesh
    windows in TRIANGLES. `nodes` comes back holding
    `sum(2 * trinum - 1)` records, and `bvhadr`/`bvhnum` the window each mesh
    got — both ARENA record numbers, i.e. already offset past the triangles.

    ⚠ A MESH WITH NO TRIANGLES GETS `bvhnum = 0`, NOT AN EMPTY TREE. Zero is
    what `ray_model` dispatches on to take the linear leg, and the linear leg
    on zero triangles is the correct "no hit". A one-node tree with no leaf
    would be a shape the traversal has no case for.
    """
    var ntri_total = len(mesh_tri) // MESH_ARENA_RECORD
    var node_base = ntri_total

    bvhadr = List[Int]()
    bvhnum = List[Int]()

    for mi in range(len(mesh_triadr)):
        var adr = mesh_triadr[mi]
        var num = mesh_trinum[mi]
        if num <= 0:
            bvhadr.append(0)
            bvhnum.append(0)
            continue

        # Per-triangle AABB and centroid, once. The centroid is the SUM of the
        # three vertices rather than their mean — the split only compares
        # centroids on one axis, and dropping the divide keeps the comparison
        # exact in the same units.
        var lo = List[Scalar[DTYPE]]()
        var hi = List[Scalar[DTYPE]]()
        var cen = List[Scalar[DTYPE]]()
        var order = List[Int]()
        var third = Scalar[DTYPE](1.0 / 3.0)

        var gmn = Scalar[DTYPE](0)
        var gmx = Scalar[DTYPE](0)

        for t in range(num):
            var o = (adr + t) * MESH_ARENA_RECORD
            for a in range(3):
                var v0 = mesh_tri[o + 0 + a]
                var v1 = mesh_tri[o + 3 + a]
                var v2 = mesh_tri[o + 6 + a]
                var mn = v0
                if v1 < mn:
                    mn = v1
                if v2 < mn:
                    mn = v2
                var mx = v0
                if v1 > mx:
                    mx = v1
                if v2 > mx:
                    mx = v2
                lo.append(mn)
                hi.append(mx)
                cen.append(third * (v0 + v1 + v2))
                if t == 0 and a == 0:
                    gmn = mn
                    gmx = mx
                else:
                    if mn < gmn:
                        gmn = mn
                    if mx > gmx:
                        gmx = mx
            order.append(t)

        # The pad is 1e-5 of the mesh's own extent, so it scales with the
        # model's units and stays far above the float32 spacing at those
        # coordinates. On a 10 cm part that is one micron — invisible to the
        # cull rate, decisive for a flat box.
        var extent = gmx - gmn
        if extent < 0:
            extent = -extent
        var pad = Scalar[DTYPE](1e-5) * extent
        if pad < Scalar[DTYPE](1e-9):
            pad = Scalar[DTYPE](1e-9)

        var start = node_base + len(nodes) // MESH_ARENA_RECORD
        _ = _build_subtree[DTYPE](
            nodes, order, lo, hi, cen, 0, num, node_base, adr, pad, sah
        )
        var count = node_base + len(nodes) // MESH_ARENA_RECORD - start

        # 2n-1 is exact for a one-element-per-leaf binary tree, and a build
        # that produced anything else has a bug the traversal would read as
        # geometry. Cheap to check once per mesh, impossible to see later.
        if count != 2 * num - 1:
            raise Error(
                String("physics3d: mesh BVH for mesh ")
                + String(mi)
                + " emitted "
                + String(count)
                + " nodes for "
                + String(num)
                + " triangles; a one-triangle-per-leaf tree has exactly "
                + String(2 * num - 1)
                + "."
            )

        bvhadr.append(start)
        bvhnum.append(count)
