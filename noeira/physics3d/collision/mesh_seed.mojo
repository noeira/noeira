"""The mesh hill climb's direction seed table.

`hillclimb_support_index` (`gjk.mojo`) walks a hull's vertex graph from a
start vertex to the extreme vertex along a query direction. On a convex hull
the walk is exact from ANY start, so the start only decides how many steps it
takes — and on the dense CAD hulls the arms carry (772-4,125 vertices on
so101_tower) a warm-started walk still took 14.4 neighbourhood scans per call
(CPU `_HILL_PROBE`, 200 control steps), because GJK and EPA query directions
that jump across the hull within one penetrating pair. Each scan is a chain
of dependent global loads on a GPU thread, and on the tower the support
function was ~78% of a penetrating box/mesh pair's GJK/EPA time.

MuJoCo 3.12 seeds the same walk from `mesh_extrema`: the extreme vertex of
each direction of the (-1, 0, 1)^3 grid, taking the better of that seed and
the warm vertex (`mjc_hillclimbSupport`, engine_collision_convex.c). This is
the same rule on a finer grid, `MESH_SEED_Q` points per axis. Replayed on the
tower's walks (the probe's seeded replay, best of warm and seed):

    grid       scans per call    landings that differ from the warm walk
    none            14.4                      —
    3^3 (3.12)       6.2              4 of 18,990, all ties
    5^3              3.6             95 of 18,990, all ties
    9^3              3.0              5 of 18,990, all ties

A different landing is always a TIE — another vertex with the same dot
product, on a face perpendicular to the query — so the support point can move
along that face and nothing else; 3.12 accepted the same.

LAYOUT. The table lives in `mesh_edges`, which every reader indexes through
`mesh_vert_edgeadr` and never scans linearly, so a block placed between two
meshes' neighbour lists is invisible to them:

    mesh_edges: ... | MAGIC | seed[0] .. seed[N-1] | lists of vertex 0, 1, ...
                              ^ edgeadr[v0] - N     ^ edgeadr[v0]

`seed[k]` is a LOCAL vertex index. The reader checks the `MESH_SEED_MAGIC`
slot before trusting the block, so a model built without tables (or a mesh
below `MESH_SEED_MIN_VERTS`) climbs from the warm vertex exactly as before.
"""

from std.math import sqrt

from ..gpu.constants import (
    MESH_SEED_Q, MESH_SEED_N, MESH_SEED_MIN_VERTS, MESH_SEED_MAGIC,
)


@always_inline
def mesh_seed_bin[DTYPE: DType](
    x: Scalar[DTYPE], y: Scalar[DTYPE], z: Scalar[DTYPE]
) -> Int:
    """The seed-table slot for direction (x, y, z), or -1 for a zero vector.

    Each component of the UNIT direction is rounded to the nearest of the
    `MESH_SEED_Q` grid values in [-1, 1]."""
    var n2 = x * x + y * y + z * z
    if not (n2 > Scalar[DTYPE](0)):
        return -1
    var s = Scalar[DTYPE](0.5 * Float64(MESH_SEED_Q - 1)) / sqrt(n2)
    var h = Scalar[DTYPE](0.5 * Float64(MESH_SEED_Q - 1)) + Scalar[DTYPE](0.5)
    var ix = Int(x * s + h)
    var iy = Int(y * s + h)
    var iz = Int(z * s + h)
    ix = 0 if ix < 0 else (MESH_SEED_Q - 1 if ix >= MESH_SEED_Q else ix)
    iy = 0 if iy < 0 else (MESH_SEED_Q - 1 if iy >= MESH_SEED_Q else iy)
    iz = 0 if iz < 0 else (MESH_SEED_Q - 1 if iz >= MESH_SEED_Q else iz)
    return (ix * MESH_SEED_Q + iy) * MESH_SEED_Q + iz


def append_mesh_seed_table[DTYPE: DType](
    verts: List[Scalar[DTYPE]],
    off: Int,
    num_hull: Int,
    mut edge_list: List[Int],
):
    """Append `MAGIC, seed[0..N)` for a hull of `num_hull` vertices whose
    coordinates are `verts[off + 3*i ..]` (the STORED float32-rounded ones the
    walk reads), if it is big enough to carry one.

    The caller appends the mesh's neighbour lists right after, so the table
    ends at the mesh's first `edgeadr`. The grid point of slot (i, j, k) is
    `-1 + 2*(i, j, k)/(Q-1)`; its extreme vertex is the first one of maximal
    dot product."""
    if num_hull < MESH_SEED_MIN_VERTS:
        return
    edge_list.append(MESH_SEED_MAGIC)
    var step = 2.0 / Float64(MESH_SEED_Q - 1)
    for k in range(MESH_SEED_N):
        var cx = -1.0 + step * Float64(k // (MESH_SEED_Q * MESH_SEED_Q))
        var cy = -1.0 + step * Float64((k // MESH_SEED_Q) % MESH_SEED_Q)
        var cz = -1.0 + step * Float64(k % MESH_SEED_Q)
        var best = -1.0e300
        var arg = 0
        for i in range(num_hull):
            var d = (
                Float64(verts[off + 3 * i + 0]) * cx
                + Float64(verts[off + 3 * i + 1]) * cy
                + Float64(verts[off + 3 * i + 2]) * cz
            )
            if d > best:
                best = d
                arg = i
        edge_list.append(arg)
