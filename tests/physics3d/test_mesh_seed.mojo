"""The hill climb's direction seed table is where the reader looks, and right.

    pixi run mojo run -I . tests/physics3d/test_mesh_seed.mojo

⚠ RUN FROM THE REPO ROOT — the fixture meshes are addressed by repo-root
relative path.

`collision/mesh_seed.mojo` stores, for a hull of at least
`MESH_SEED_MIN_VERTS` vertices, one extreme vertex per point of a
`MESH_SEED_Q`^3 direction grid in `mesh_edges`, behind a `MESH_SEED_MAGIC`
slot and directly before the mesh's first neighbour list; `gjk.
hillclimb_support_index` finds it at `edgeadr[v0] - MESH_SEED_N`. A table in
the wrong place is NOT caught by the support gates: any in-range index is a
valid (slower) start for an exact walk, and out-of-range ones are refused, so
a misplaced table degrades to the old speed without a single wrong point.
This pins the placement and the content instead:

- a big hull has the MAGIC slot exactly `MESH_SEED_N + 1` before its first
  neighbour list, and every seed is an extreme vertex of its grid point;
- `mesh_seed_bin` sends each grid point's own direction back to its slot;
- a hull below `MESH_SEED_MIN_VERTS` carries no table;
- a second mesh's table sits before ITS lists, after the first mesh's.

`test_gjk_hillclimb_support.mojo` is the other half: the walk, seeded, still
lands on the extreme vertex on real hulls.
"""

from std.math import sqrt
from std.testing import assert_true, assert_equal, TestSuite

from noeira.physics3d.collision.convex_hull import load_mesh_hull
from noeira.physics3d.collision.mesh_seed import mesh_seed_bin
from noeira.physics3d.gpu.constants import (
    MESH_SEED_Q, MESH_SEED_N, MESH_SEED_MIN_VERTS, MESH_SEED_MAGIC,
)
from noeira.physics3d.model.mesh_inertia import MeshInertia

comptime D = DType.float64

comptime BIG = "noeira/envs/robots/assets/so_arm100/Wrist_Pitch_Roll.stl"
comptime SMALL = "tests/physics3d/assets/mc_hex.stl"


struct _Hulls:
    var mesh_vert: List[Scalar[D]]
    var mesh_vertadr: List[Int]
    var mesh_vertnum: List[Int]
    var edge_adr: List[Int]
    var edge_list: List[Int]

    def __init__(out self, paths: List[String]) raises:
        self.mesh_vert = List[Scalar[D]]()
        self.mesh_vertadr = List[Int]()
        self.mesh_vertnum = List[Int]()
        self.edge_adr = List[Int]()
        self.edge_list = List[Int]()
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
        var mesh_tri = List[Scalar[D]]()
        var mesh_triadr = List[Int]()
        var mesh_trinum = List[Int]()
        for p in paths:
            var mi = MeshInertia[D]()
            _ = load_mesh_hull[D](
                p, self.mesh_vert, self.mesh_vertadr, self.mesh_vertnum,
                num_meshes, mesh_polyadr, mesh_polynum, poly_vert,
                poly_vertadr, poly_vertnum, poly_normal, polymap,
                polymap_adr, polymap_num, self.edge_adr, self.edge_list,
                mesh_tri, mesh_triadr, mesh_trinum, mi,
            )

    def dot(self, m: Int, local: Int, x: Float64, y: Float64, z: Float64) -> Float64:
        var o = (self.mesh_vertadr[m] + local) * 3
        return (
            Float64(self.mesh_vert[o]) * x
            + Float64(self.mesh_vert[o + 1]) * y
            + Float64(self.mesh_vert[o + 2]) * z
        )

    def check_table(self, m: Int, name: String) raises:
        var va = self.mesh_vertadr[m]
        var nv = self.mesh_vertnum[m]
        var head = self.edge_adr[va]
        var sb = head - MESH_SEED_N
        assert_true(sb >= 1, name + ": no room for a table before the lists")
        assert_equal(
            self.edge_list[sb - 1], MESH_SEED_MAGIC,
            name + ": the MAGIC slot is not MESH_SEED_N + 1 before edgeadr[v0]",
        )
        var step = 2.0 / Float64(MESH_SEED_Q - 1)
        for k in range(MESH_SEED_N):
            var cx = -1.0 + step * Float64(k // (MESH_SEED_Q * MESH_SEED_Q))
            var cy = -1.0 + step * Float64((k // MESH_SEED_Q) % MESH_SEED_Q)
            var cz = -1.0 + step * Float64(k % MESH_SEED_Q)
            var s = self.edge_list[sb + k]
            assert_true(
                s >= 0 and s < nv,
                name + ": seed " + String(k) + " = " + String(s)
                + " is not a local vertex index of a " + String(nv)
                + "-vertex hull",
            )
            var best = -1.0e300
            for i in range(nv):
                var d = self.dot(m, i, cx, cy, cz)
                if d > best:
                    best = d
            assert_equal(
                self.dot(m, s, cx, cy, cz), best,
                name + ": seed " + String(k) + " is not extreme along its"
                " grid point",
            )
            # A grid point whose UNIT direction rounds back to it must map to
            # its own slot (the centre point is no direction).
            if (cx != 0.0 or cy != 0.0 or cz != 0.0) and _unit_rounds_home(
                cx, cy, cz
            ):
                assert_equal(
                    mesh_seed_bin[D](
                        Scalar[D](cx), Scalar[D](cy), Scalar[D](cz)
                    ),
                    k,
                    name + ": mesh_seed_bin of grid point " + String(k),
                )


def _unit_rounds_home(cx: Float64, cy: Float64, cz: Float64) -> Bool:
    """Does the grid point's UNIT direction round back to the grid point?
    Not every one does — (1, 1, 0) normalises to (0.71, 0.71, 0), which
    rounds to (0.5, 0.5, 0) at Q=5 — so the rule is checked, not assumed."""
    var n = sqrt(cx * cx + cy * cy + cz * cz)
    var h = 0.5 * Float64(MESH_SEED_Q - 1)
    var ok = True
    for c in [cx, cy, cz]:
        var i = Int((c / n) * h + h + 0.5)
        var back = -1.0 + 2.0 * Float64(i) / Float64(MESH_SEED_Q - 1)
        ok = ok and back == c
    return ok


def test_axis_directions_round_home() raises:
    """The six axis directions, at any length, land on their own slots —
    the check `check_table` skips for grid points that do not round home
    must not be skipping everything."""
    var q = MESH_SEED_Q - 1
    var mid = q // 2
    assert_equal(mesh_seed_bin[D](Scalar[D](3), 0, 0), (q * MESH_SEED_Q + mid) * MESH_SEED_Q + mid)
    assert_equal(mesh_seed_bin[D](Scalar[D](-0.2), 0, 0), (0 * MESH_SEED_Q + mid) * MESH_SEED_Q + mid)
    assert_equal(mesh_seed_bin[D](0, Scalar[D](1), 0), (mid * MESH_SEED_Q + q) * MESH_SEED_Q + mid)
    assert_equal(mesh_seed_bin[D](0, 0, Scalar[D](-5)), (mid * MESH_SEED_Q + mid) * MESH_SEED_Q + 0)
    assert_equal(mesh_seed_bin[D](0, 0, 0), -1)


def test_big_hull_carries_a_correct_table() raises:
    var h = _Hulls([String(BIG)])
    assert_true(
        h.mesh_vertnum[0] >= MESH_SEED_MIN_VERTS,
        "fixture: the big hull must be over the table threshold",
    )
    h.check_table(0, "Wrist_Pitch_Roll")


def test_small_hull_has_none() raises:
    var h = _Hulls([String(SMALL)])
    assert_true(
        h.mesh_vertnum[0] < MESH_SEED_MIN_VERTS,
        "fixture: the small hull must be under the table threshold",
    )
    assert_equal(h.edge_adr[0], 0, "a small hull's lists must start at 0")


def test_second_mesh_table_is_its_own() raises:
    var h = _Hulls([String(SMALL), String(BIG)])
    var head1 = h.edge_adr[h.mesh_vertadr[1]]
    assert_true(
        head1 - MESH_SEED_N - 1 >= h.edge_adr[h.mesh_vertadr[1] - 1],
        "the second mesh's table overlaps the first mesh's lists",
    )
    h.check_table(1, "Wrist_Pitch_Roll after a small hull")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
