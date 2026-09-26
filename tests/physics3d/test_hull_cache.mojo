"""A cached mesh must be BIT-IDENTICAL to a freshly built one.

    pixi run mojo run -I . tests/physics3d/test_hull_cache.mojo

⚠ RUN FROM THE REPO ROOT — the fixture meshes are addressed by repo-root
relative path.

WHAT THIS GATES. `load_mesh_hull` appends into arrays SHARED BY EVERY MESH IN
THE MODEL, and only some of what it writes is positional:

  * `poly_vertadr`, `polymap_adr`, `edge_adr` are OFFSETS into their companion
    arrays and must be shifted by the current length of that companion;
  * `poly_vert` and `polymap` hold LOCAL ids and must NOT be shifted;
  * `edge_list` holds GLOBAL vertex ids and must be shifted by the mesh's
    vertex base — except for the `-1` terminators, which must pass through.

The cache stores everything rebased to zero and re-applies those shifts on
load. ⚠⚠ GET ONE WRONG AND THE INDICES POINT INTO A NEIGHBOURING MESH'S
REGION: a collision shape that is silently, plausibly wrong, with no crash and
no compile error to catch it. That failure mode is the entire reason this file
exists, and it is why the comparison is EXACT rather than approximate — a
rebasing slip is an integer error, so a tolerance would only hide it.

⚠ THREE MESHES, NOT ONE, AND THEY MUST DIFFER IN SIZE. Every shift above is
`+ 0` for the first mesh in a model, so a single-mesh fixture passes with every
rebasing term deleted. Meshes 2 and 3 are what make the offsets non-zero, and
distinct sizes are what stop a wrong-but-uniform shift from cancelling out.

⚠ "WARM EQUALS COLD" IS NOT ENOUGH ON ITS OWN. If the cache never wrote
anything, every build would be cold and every comparison would pass. Each test
therefore also asserts the entry is READABLE afterwards
(`test_entries_are_actually_written`), so a silently-disabled cache fails here
rather than passing quietly and stalling the viewer.
"""

from std.os import getenv, setenv
from std.testing import assert_equal, assert_true, TestSuite

from noeira.physics3d.collision.convex_hull import load_mesh_hull
from noeira.physics3d.collision.hull_cache import (
    HullPayload,
    hull_cache_load,
    hull_cache_path,
)
from noeira.physics3d.model.mesh_inertia import MeshInertia
from noeira.physics3d.gpu.constants import (
    MESH_SEED_N, MESH_SEED_MIN_VERTS, MESH_SEED_MAGIC,
)

comptime CACHE_DIR = ".cache/physics3d_hulls_test"


def _fixtures() -> List[String]:
    """Three SO-ARM100 collision meshes of clearly different sizes.

    Sizes matter: see the header. These are ~200, ~570 and ~120 hull vertices,
    so every offset in the second and third append is distinct and non-zero.
    """
    var out = List[String]()
    out.append(String("noeira/envs/robots/assets/so_arm100/Wrist_Pitch_Roll.stl"))
    out.append(String("noeira/envs/robots/assets/so_arm100/Fixed_Jaw_Collision_2.stl"))
    out.append(String("noeira/envs/robots/assets/so_arm100/Moving_Jaw_Collision_3.stl"))
    return out^


struct Built[DTYPE: DType](Copyable, Movable):
    """Every array `load_mesh_hull` touches, after all three meshes."""

    var mesh_vert: List[Scalar[Self.DTYPE]]
    var mesh_vertadr: List[Int]
    var mesh_vertnum: List[Int]
    var mesh_polyadr: List[Int]
    var mesh_polynum: List[Int]
    var poly_vert: List[Int]
    var poly_vertadr: List[Int]
    var poly_vertnum: List[Int]
    var poly_normal: List[Scalar[Self.DTYPE]]
    var polymap: List[Int]
    var polymap_adr: List[Int]
    var polymap_num: List[Int]
    var edge_adr: List[Int]
    var edge_list: List[Int]
    # The mesh TRIANGLE SOUP (`ray/mesh.mojo`). It rides in the same payload,
    # so a cold build and a warm one must agree on it too — and it is the one
    # array here that is NOT rebased, since a triangle carries coordinates
    # rather than vertex ids.
    var mesh_tri: List[Scalar[Self.DTYPE]]
    var mesh_triadr: List[Int]
    var mesh_trinum: List[Int]
    var rbound: List[Scalar[Self.DTYPE]]

    def __init__(out self):
        self.mesh_vert = List[Scalar[Self.DTYPE]]()
        self.mesh_vertadr = List[Int]()
        self.mesh_vertnum = List[Int]()
        self.mesh_polyadr = List[Int]()
        self.mesh_polynum = List[Int]()
        self.poly_vert = List[Int]()
        self.poly_vertadr = List[Int]()
        self.poly_vertnum = List[Int]()
        self.poly_normal = List[Scalar[Self.DTYPE]]()
        self.polymap = List[Int]()
        self.polymap_adr = List[Int]()
        self.polymap_num = List[Int]()
        self.edge_adr = List[Int]()
        self.edge_list = List[Int]()
        self.mesh_tri = List[Scalar[Self.DTYPE]]()
        self.mesh_triadr = List[Int]()
        self.mesh_trinum = List[Int]()
        self.rbound = List[Scalar[Self.DTYPE]]()


def _build[DTYPE: DType](enabled: Bool) raises -> Built[DTYPE]:
    """Load all three fixtures into one set of shared arrays."""
    _ = setenv("PHYSICS3D_HULL_CACHE", "1" if enabled else "0", True)
    _ = setenv("PHYSICS3D_HULL_CACHE_DIR", CACHE_DIR, True)

    var b = Built[DTYPE]()
    var num_meshes = 0
    var mi = MeshInertia[DTYPE]()
    var paths = _fixtures()
    for i in range(len(paths)):
        var r = load_mesh_hull[DTYPE](
            paths[i],
            b.mesh_vert, b.mesh_vertadr, b.mesh_vertnum, num_meshes,
            b.mesh_polyadr, b.mesh_polynum, b.poly_vert, b.poly_vertadr,
            b.poly_vertnum, b.poly_normal, b.polymap, b.polymap_adr,
            b.polymap_num, b.edge_adr, b.edge_list,
            b.mesh_tri, b.mesh_triadr, b.mesh_trinum, mi,
        )
        b.rbound.append(r[1])
    return b^


def _same_ints(what: String, a: List[Int], b: List[Int]) raises:
    assert_equal(
        len(a), len(b),
        what + ": cold build has " + String(len(a)) + " entries, cached build"
        " has " + String(len(b)),
    )
    for i in range(len(a)):
        assert_true(
            a[i] == b[i],
            what + "[" + String(i) + "]: cold " + String(a[i]) + " vs cached "
            + String(b[i]) + " — an INTEGER difference, so this is a rebasing"
            " error, not rounding; see the table in hull_cache.mojo",
        )


def _same_floats[
    DTYPE: DType
](what: String, a: List[Scalar[DTYPE]], b: List[Scalar[DTYPE]]) raises:
    """⚠ EXACT. The cache stores float64 bits and narrows on load, so a cached
    value that is merely CLOSE means the round-trip is lossy."""
    assert_equal(
        len(a), len(b),
        what + ": cold build has " + String(len(a)) + " values, cached build"
        " has " + String(len(b)),
    )
    for i in range(len(a)):
        assert_true(
            a[i] == b[i],
            what + "[" + String(i) + "]: cold " + String(a[i]) + " vs cached "
            + String(b[i]) + " — the round-trip must be bit-exact",
        )


def _compare[DTYPE: DType](cold: Built[DTYPE], warm: Built[DTYPE]) raises:
    _same_floats[DTYPE]("mesh_vert", cold.mesh_vert, warm.mesh_vert)
    _same_ints("mesh_vertadr", cold.mesh_vertadr, warm.mesh_vertadr)
    _same_ints("mesh_vertnum", cold.mesh_vertnum, warm.mesh_vertnum)
    _same_ints("mesh_polyadr", cold.mesh_polyadr, warm.mesh_polyadr)
    _same_ints("mesh_polynum", cold.mesh_polynum, warm.mesh_polynum)
    _same_ints("poly_vert", cold.poly_vert, warm.poly_vert)
    _same_ints("poly_vertadr", cold.poly_vertadr, warm.poly_vertadr)
    _same_ints("poly_vertnum", cold.poly_vertnum, warm.poly_vertnum)
    _same_floats[DTYPE]("poly_normal", cold.poly_normal, warm.poly_normal)
    _same_ints("polymap", cold.polymap, warm.polymap)
    _same_ints("polymap_adr", cold.polymap_adr, warm.polymap_adr)
    # ⚠ The triangle soup rides the SAME cache payload, so a cold/warm
    # difference here is a serialisation bug in the version-7 format and not a
    # hull one. Compared byte-for-byte like `mesh_vert`: the coordinates are
    # float32-rounded on both paths, so equality is exact, not approximate.
    _same_floats[DTYPE]("mesh_tri", cold.mesh_tri, warm.mesh_tri)
    _same_ints("mesh_triadr", cold.mesh_triadr, warm.mesh_triadr)
    _same_ints("mesh_trinum", cold.mesh_trinum, warm.mesh_trinum)
    _same_ints("polymap_num", cold.polymap_num, warm.polymap_num)
    _same_ints("edge_adr", cold.edge_adr, warm.edge_adr)
    _same_ints("edge_list", cold.edge_list, warm.edge_list)
    _same_floats[DTYPE]("rbound", cold.rbound, warm.rbound)


def test_cached_equals_cold_float64() raises:
    """The main gate: three meshes, cache off vs cache on, every array."""
    var cold = _build[DType.float64](False)
    _ = _build[DType.float64](True)  # populate, in case this is a fresh tree
    var warm = _build[DType.float64](True)
    print(
        "   f64  verts", len(cold.mesh_vert) // 3, " polys", len(cold.poly_vertadr),
        " polymap", len(cold.polymap), " edges", len(cold.edge_list),
    )
    _compare[DType.float64](cold, warm)


def test_cached_equals_cold_float32() raises:
    """Float32 keeps its own entries — the mesh frame is part of the key and
    is rounded to the runtime dtype. Serving a float64 hull here would be the
    dtype bug `test_convex_hull_dtype_invariance.mojo` exists to prevent,
    reintroduced through the cache."""
    var cold = _build[DType.float32](False)
    _ = _build[DType.float32](True)
    var warm = _build[DType.float32](True)
    print(
        "   f32  verts", len(cold.mesh_vert) // 3, " polys", len(cold.poly_vertadr),
        " polymap", len(cold.polymap), " edges", len(cold.edge_list),
    )
    _compare[DType.float32](cold, warm)


def test_entries_are_actually_written() raises:
    """⚠⚠ WITHOUT THIS, EVERY TEST ABOVE PASSES ON A DEAD CACHE. If nothing is
    ever written, cold and warm are the same code path and agree trivially."""
    _ = _build[DType.float64](True)
    _ = setenv("PHYSICS3D_HULL_CACHE", "1", True)
    _ = setenv("PHYSICS3D_HULL_CACHE_DIR", CACHE_DIR, True)
    var mi = MeshInertia[DType.float64]()
    var paths = _fixtures()
    for i in range(len(paths)):
        var p = hull_cache_path[DType.float64](paths[i], mi)
        assert_true(
            p.byte_length() > 0,
            "no cache path for " + paths[i] + " — the cache is switched off"
            " when the tests above assume it is on",
        )
        var payload = HullPayload()
        assert_true(
            hull_cache_load(p, payload),
            "nothing readable at " + p + " after a build — the cache never"
            " wrote, so 'cached equals cold' above compared cold against cold",
        )
        assert_true(
            payload.num_hull > 0 and payload.npoly > 0,
            "empty payload at " + p,
        )


def test_disabled_cache_yields_no_path() raises:
    """`PHYSICS3D_HULL_CACHE=0` must switch the whole thing off — that is what
    the tests above rely on to get a genuinely cold build, and what to reach
    for if a cached hull is ever suspected."""
    _ = setenv("PHYSICS3D_HULL_CACHE", "0", True)
    var mi = MeshInertia[DType.float64]()
    var p = hull_cache_path[DType.float64](_fixtures()[0], mi)
    assert_equal(
        p.byte_length(), 0,
        "PHYSICS3D_HULL_CACHE=0 still produced a cache path (" + p + ")",
    )
    _ = setenv("PHYSICS3D_HULL_CACHE", "1", True)


def test_key_tracks_the_mesh_frame() raises:
    """⚠ THE FRAME IS PART OF THE KEY, NOT JUST THE FILE. `load_mesh_hull`
    bakes `mi`'s centre of mass and principal rotation into the vertices
    BEFORE hulling, so the same STL under a different `mi` is a different hull.
    Keying on file contents alone would serve one geometry for the other."""
    _ = setenv("PHYSICS3D_HULL_CACHE", "1", True)
    _ = setenv("PHYSICS3D_HULL_CACHE_DIR", CACHE_DIR, True)
    var path = _fixtures()[0]
    var a = MeshInertia[DType.float64]()
    var b = MeshInertia[DType.float64]()
    b.com_x = Scalar[DType.float64](0.01)
    var ka = hull_cache_path[DType.float64](path, a)
    var kb = hull_cache_path[DType.float64](path, b)
    print("   frame A ", ka)
    print("   frame B ", kb)
    assert_true(
        ka != kb,
        "the same mesh under two different frames maps to ONE cache entry ("
        + ka + ") — the second model would collide with the first's geometry",
    )


def test_indices_stay_inside_their_own_mesh() raises:
    """⚠⚠ THE REBASING GATE. THE COLD-VS-WARM TESTS ABOVE CANNOT BE IT.

    This was the first thing written here and it was WORTHLESS for its stated
    purpose. `load_mesh_hull` now has ONE append block, taken by the cached and
    the freshly-built path alike, so a rebasing error corrupts both identically
    and they still agree. Verified by deleting `+ poly_vert_base` and watching
    all five tests pass.

    So the rebasing has to be checked against the INVARIANTS it must satisfy,
    not against another run of itself. For every mesh, and with `nv` its hull
    vertex count and `np` its polygon count:

      * `poly_vertadr` / `polymap_adr` / `edge_adr` are CONTIGUOUS across the
        whole model — each entry starts where the previous one ended. A missing
        shift restarts a mesh's block at 0 and breaks this immediately.
      * `poly_vert` holds LOCAL vertex ids, so every value is in `[0, nv)`.
        An unwanted shift pushes them past the end.
      * `polymap` holds LOCAL polygon ids, so every value is in `[0, np)`.
      * `edge_list` holds GLOBAL vertex ids belonging to THIS mesh, so every
        value is in `[vert_base, vert_base + nv)` — a missing shift leaves
        mesh 2's neighbours pointing into mesh 1, which is precisely the
        silent-wrong-geometry failure, and it lands in range for mesh 0 only.
      * `-1` terminates a vertex's neighbour run and must survive as `-1`;
        shifted, it becomes `vert_base - 1`, a plausible vertex id.
      * a mesh of at least `MESH_SEED_MIN_VERTS` vertices has its seed table
        (`mesh_seed.mojo`) right before its first run: `MESH_SEED_MAGIC`, then
        `MESH_SEED_N` LOCAL vertex ids, so every seed is in `[0, nv)`.

    These hold whatever built the arrays, which is what makes them a real
    check rather than a mirror.
    """
    var b = _build[DType.float64](True)
    var nmesh = len(b.mesh_vertnum)
    assert_true(nmesh == 3, "expected 3 meshes, got " + String(nmesh))

    var pv_run = 0
    var pm_run = 0
    var el_run = 0
    for m in range(nmesh):
        var vert_base = b.mesh_vertadr[m]
        var nv = b.mesh_vertnum[m]
        var pa = b.mesh_polyadr[m]
        var np_ = b.mesh_polynum[m]
        var tag = "mesh " + String(m) + " "

        # --- polygons: contiguous offsets, LOCAL vertex ids ------------------
        for k in range(pa, pa + np_):
            assert_true(
                b.poly_vertadr[k] == pv_run,
                tag + "poly_vertadr[" + String(k) + "] is "
                + String(b.poly_vertadr[k]) + " but the previous polygon ended"
                " at " + String(pv_run) + " — the per-mesh block was not"
                " shifted by len(poly_vert)",
            )
            for j in range(b.poly_vertnum[k]):
                var v = b.poly_vert[b.poly_vertadr[k] + j]
                assert_true(
                    v >= 0 and v < nv,
                    tag + "poly_vert holds " + String(v) + ", outside this"
                    " mesh's [0, " + String(nv) + ") — these are LOCAL vertex"
                    " ids and must NOT be shifted",
                )
            pv_run += b.poly_vertnum[k]

        # --- vertex -> polygon map: contiguous, LOCAL polygon ids ------------
        for v in range(vert_base, vert_base + nv):
            assert_true(
                b.polymap_adr[v] == pm_run,
                tag + "polymap_adr[" + String(v) + "] is "
                + String(b.polymap_adr[v]) + ", expected " + String(pm_run)
                + " — not shifted by len(polymap)",
            )
            for j in range(b.polymap_num[v]):
                var p = b.polymap[b.polymap_adr[v] + j]
                assert_true(
                    p >= 0 and p < np_,
                    tag + "polymap holds " + String(p) + ", outside this"
                    " mesh's [0, " + String(np_) + ") — these are LOCAL"
                    " polygon ids and must NOT be shifted",
                )
            pm_run += b.polymap_num[v]

        # --- the hill climb's seed table, ahead of a big mesh's lists --------
        # (`mesh_seed.mojo`): MAGIC, then MESH_SEED_N LOCAL vertex ids.
        if nv >= MESH_SEED_MIN_VERTS:
            assert_true(
                b.edge_list[el_run] == MESH_SEED_MAGIC,
                tag + "no seed-table MAGIC at " + String(el_run)
                + " ahead of a " + String(nv) + "-vertex mesh's lists",
            )
            for k in range(MESH_SEED_N):
                var s = b.edge_list[el_run + 1 + k]
                assert_true(
                    s >= 0 and s < nv,
                    tag + "seed " + String(k) + " holds " + String(s)
                    + ", outside this mesh's [0, " + String(nv) + ") — seeds"
                    " are LOCAL vertex ids and must NOT be shifted",
                )
            el_run += MESH_SEED_N + 1

        # --- edge graph: contiguous, GLOBAL ids, -1 preserved ----------------
        for v in range(vert_base, vert_base + nv):
            assert_true(
                b.edge_adr[v] == el_run,
                tag + "edge_adr[" + String(v) + "] is "
                + String(b.edge_adr[v]) + ", expected " + String(el_run)
                + " — not shifted by len(edge_list)",
            )
            var e = b.edge_adr[v]
            var seen_terminator = False
            while e < len(b.edge_list):
                var w = b.edge_list[e]
                e += 1
                if w == -1:
                    seen_terminator = True
                    break
                assert_true(
                    w >= vert_base and w < vert_base + nv,
                    tag + "edge_list holds " + String(w) + " for vertex "
                    + String(v) + ", outside this mesh's [" + String(vert_base)
                    + ", " + String(vert_base + nv) + ") — neighbours are"
                    " GLOBAL ids and must be shifted by the vertex base; a"
                    " value in another mesh's range is the silent"
                    " wrong-geometry failure this file exists for",
                )
            assert_true(
                seen_terminator,
                tag + "vertex " + String(v) + "'s neighbour run has no -1"
                " terminator — a shifted terminator becomes a plausible vertex"
                " id and the walk runs into the next vertex's neighbours",
            )
            el_run = e

    assert_true(
        pv_run == len(b.poly_vert),
        "poly_vert has " + String(len(b.poly_vert)) + " entries but the"
        " polygons account for " + String(pv_run),
    )
    assert_true(
        pm_run == len(b.polymap),
        "polymap has " + String(len(b.polymap)) + " entries but the vertex map"
        " accounts for " + String(pm_run),
    )
    assert_true(
        el_run == len(b.edge_list),
        "edge_list has " + String(len(b.edge_list)) + " entries but the"
        " neighbour runs account for " + String(el_run),
    )
    print("   rebasing OK  polyvert", pv_run, " polymap", pm_run, " edges", el_run)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
