"""The SAH tree, the median tree and the caller's cut must not move a mesh ray's answer.

    pixi run mojo run -I . tests/physics3d/test_ray_bvh_sah.mojo

`parser/mesh_bvh_build.mojo` builds two trees over the same triangles — the
reference's median split (`sah=False`) and a binned-SAH split (the default) —
and `ray/mesh.mojo::ray_mesh_bvh` now takes `tcut`, the caller's best hit, and
breaks exact ties by triangle index. `test_ray_bvh_matches_linear.mojo` gates
the default tree through the renderer on a small scene; this file gates the
three changes ONE BY ONE, on the ray routine itself, against the linear sweep
`ray_mesh`, on real meshes — including one of the SO-101 arm's STLs (28k
triangles), the part the SAH was written for.

WHAT IS ASSERTED, EXACTLY (no tolerance — both legs run the same
`ray_triangle` on the same triangle, so any difference is a traversal defect):

1. SAH tree == linear sweep and median tree == linear sweep, on `t`, the
   triangle, the normal and the barycentrics, for every ray.
2. With `tcut`: a hit is returned iff the sweep's hit is strictly nearer than
   `tcut`, and it is then the sweep's hit. `tcut` is tried at the hit's own
   distance (must MISS — `ray_model` keeps the earlier geom on a tie), just
   beyond it (must HIT) and at half of it (must MISS).

WHAT KEEPS IT FROM BEING VACUOUS:

- a third of the rays are aimed EXACTLY AT A VERTEX, where several triangles
  meet at one distance: the tie rule must be exercised (`ties > 0`), or the
  "SAH == sweep" claim says nothing about ties;
- the two trees must DIFFER (a builder that ignored `sah` passes item 1);
- a share of the rays must hit, and a third start INSIDE the mesh's box
  (the `tmin` clamp at 0).
"""

from std.testing import assert_true, assert_equal, TestSuite
from std.time import perf_counter_ns
from layout import LayoutTensor

from noeira.math3d import Vec3 as Vec3Generic, Quat as QuatGeneric
from noeira.physics3d.fields import DYN1, rl1
from noeira.physics3d.gpu.constants import MESH_ARENA_RECORD
from noeira.physics3d.parser.mesh_bvh_build import build_mesh_bvh
from noeira.physics3d.ray import ray_mesh, ray_mesh_bvh
from noeira.render.stl_loader import load_stl

comptime DT = DType.float32
comptime Vec3 = Vec3Generic[DT]
comptime Quat = QuatGeneric[DT]
comptime NRAY = 1500
"""Per mesh, split in thirds: aimed into the box, aimed at a vertex, from
inside the box in a random direction."""


struct Lcg(Movable):
    var s: UInt64

    def __init__(out self, seed: UInt64):
        self.s = seed

    def u(mut self) -> Float64:
        self.s = self.s * 6364136223846793005 + 1442695040888963407
        return Float64(self.s >> 11) / Float64(UInt64(1) << 53)

    def sym(mut self, a: Float64) -> Float64:
        return (2.0 * self.u() - 1.0) * a


struct Soup(Movable):
    """Every mesh's triangles, nine floats each, end to end — the arena's
    triangle region — with each mesh's window, box and vertex list."""

    var tri: List[Scalar[DT]]
    var triadr: List[Int]
    var trinum: List[Int]
    var half: List[Vec3]
    var names: List[String]

    def __init__(out self, paths: List[String]) raises:
        self.tri = List[Scalar[DT]]()
        self.triadr = List[Int]()
        self.trinum = List[Int]()
        self.half = List[Vec3]()
        self.names = List[String]()
        for p in paths:
            var md = load_stl(p)
            var adr = len(self.tri) // MESH_ARENA_RECORD
            var n = len(md.indices) // 3
            var hx = Float32(0)
            var hy = Float32(0)
            var hz = Float32(0)
            for t in range(n):
                for c in range(3):
                    var v = md.vertices[Int(md.indices[t * 3 + c])]
                    self.tri.append(v.px)
                    self.tri.append(v.py)
                    self.tri.append(v.pz)
                    hx = max(hx, abs(v.px))
                    hy = max(hy, abs(v.py))
                    hz = max(hz, abs(v.pz))
            self.triadr.append(adr)
            self.trinum.append(n)
            self.half.append(Vec3(hx, hy, hz))
            self.names.append(p)


struct Arena(Movable):
    """The soup plus one tree, as `Model.mesh_tris` holds them."""

    var data: List[Scalar[DT]]
    var bvhadr: List[Int]
    var bvhnum: List[Int]

    def __init__(out self, soup: Soup, sah: Bool) raises:
        var nodes = List[Scalar[DT]]()
        self.bvhadr = List[Int]()
        self.bvhnum = List[Int]()
        build_mesh_bvh[DT](
            soup.tri, soup.triadr, soup.trinum, nodes, self.bvhadr,
            self.bvhnum, sah=sah,
        )
        self.data = soup.tri.copy()
        for i in range(len(nodes)):
            self.data.append(nodes[i])


def _paths() -> List[String]:
    var p = List[String]()
    p.append("tests/physics3d/assets/ngon100_prism.stl")
    p.append("tests/physics3d/assets/notch.stl")
    p.append("noeira/envs/robots/assets/so_arm101/moving_jaw_so101_v1.stl")
    return p^


def _same(
    a_t: Scalar[DT], a_tri: Int, a_n: Vec3, a_u: Scalar[DT], a_v: Scalar[DT],
    b_t: Scalar[DT], b_tri: Int, b_n: Vec3, b_u: Scalar[DT], b_v: Scalar[DT],
) -> Bool:
    if a_t < 0 and b_t < 0:
        return True
    return (
        a_t == b_t and a_tri == b_tri and a_n.x == b_n.x and a_n.y == b_n.y
        and a_n.z == b_n.z and a_u == b_u and a_v == b_v
    )


def test_trees_and_cut_match_the_linear_sweep() raises:
    var soup = Soup(_paths())
    var sah = Arena(soup, True)
    var med = Arena(soup, False)
    var pos = Vec3(0, 0, 0)
    var quat = Quat(1, 0, 0, 0)

    # The trees are well-formed (2n - 1 nodes each) and they are DIFFERENT.
    var differ = False
    for mi in range(len(soup.trinum)):
        assert_equal(sah.bvhnum[mi], 2 * soup.trinum[mi] - 1)
        assert_equal(med.bvhnum[mi], 2 * soup.trinum[mi] - 1)
    for i in range(len(sah.data)):
        if sah.data[i] != med.data[i]:
            differ = True
            break
    assert_true(differ, "the SAH and median trees are identical — the `sah`"
                        " flag does not reach the builder, so item 1 is vacuous")

    var lin_view = LayoutTensor[DT, DYN1, MutAnyOrigin](soup.tri, rl1(len(soup.tri)))
    var sah_view = LayoutTensor[DT, DYN1, MutAnyOrigin](sah.data, rl1(len(sah.data)))
    var med_view = LayoutTensor[DT, DYN1, MutAnyOrigin](med.data, rl1(len(med.data)))

    var rng = Lcg(0x5A4B1D)
    var bad_sah = 0
    var bad_med = 0
    var bad_cut = 0
    var hits = 0
    var ties = 0
    var ns_sah = 0
    var ns_med = 0
    var total = 0
    for mi in range(len(soup.trinum)):
        var adr = soup.triadr[mi]
        var n = soup.trinum[mi]
        var h = soup.half[mi]
        var mesh_hits = 0
        var mesh_ties = 0
        for k in range(NRAY):
            var eye: Vec3
            var vec: Vec3
            var third = k % 3
            if third == 2:
                # From INSIDE the box, a random direction.
                eye = Vec3(
                    Scalar[DT](rng.sym(Float64(h.x))),
                    Scalar[DT](rng.sym(Float64(h.y))),
                    Scalar[DT](rng.sym(Float64(h.z))),
                )
                vec = Vec3(
                    Scalar[DT](rng.sym(1.0)), Scalar[DT](rng.sym(1.0)),
                    Scalar[DT](rng.sym(1.0)),
                )
            else:
                eye = Vec3(
                    Scalar[DT](rng.sym(3.0 * Float64(h.x))),
                    Scalar[DT](rng.sym(3.0 * Float64(h.y))),
                    Scalar[DT](rng.sym(3.0 * Float64(h.z))),
                )
                var aim: Vec3
                if third == 1:
                    # EXACTLY at a vertex: every triangle sharing it is hit at
                    # the same distance — the tie rule's case.
                    var t = Int(rng.u() * Float64(n))
                    if t >= n:
                        t = n - 1
                    var o = (adr + t) * MESH_ARENA_RECORD
                    aim = Vec3(soup.tri[o], soup.tri[o + 1], soup.tri[o + 2])
                else:
                    aim = Vec3(
                        Scalar[DT](rng.sym(Float64(h.x))),
                        Scalar[DT](rng.sym(Float64(h.y))),
                        Scalar[DT](rng.sym(Float64(h.z))),
                    )
                vec = aim - eye
            total += 1

            var lin = ray_mesh[DT, DYN1](pos, quat, h, lin_view, adr, n, eye, vec)
            var t0 = perf_counter_ns()
            var s = ray_mesh_bvh[DT, DYN1](
                pos, quat, h, sah_view, adr, n, sah.bvhadr[mi], sah.bvhnum[mi],
                eye, vec,
            )
            var t1 = perf_counter_ns()
            var m = ray_mesh_bvh[DT, DYN1](
                pos, quat, h, med_view, adr, n, med.bvhadr[mi], med.bvhnum[mi],
                eye, vec,
            )
            var t2 = perf_counter_ns()
            ns_sah += Int(t1 - t0)
            ns_med += Int(t2 - t1)
            if not _same(lin.t, lin.tri, lin.normal, lin.bu, lin.bv,
                         s.t, s.tri, s.normal, s.bu, s.bv):
                bad_sah += 1
            if not _same(lin.t, lin.tri, lin.normal, lin.bu, lin.bv,
                         m.t, m.tri, m.normal, m.bu, m.bv):
                bad_med += 1

            if lin.t >= 0:
                mesh_hits += 1
                # Ties: how many triangles hit at EXACTLY the winning
                # distance (vertex-aimed rays only; the sweep over one-triangle
                # windows runs the same `ray_triangle`).
                if third == 1:
                    var at = 0
                    for t in range(n):
                        var one = ray_mesh[DT, DYN1](
                            pos, quat, h, lin_view, adr + t, 1, eye, vec
                        )
                        if one.t == lin.t:
                            at += 1
                    if at >= 2:
                        mesh_ties += 1
                # The cut: at the hit (miss), just beyond (hit), half (miss).
                var c_at = ray_mesh_bvh[DT, DYN1](
                    pos, quat, h, sah_view, adr, n, sah.bvhadr[mi],
                    sah.bvhnum[mi], eye, vec, lin.t,
                )
                var c_beyond = ray_mesh_bvh[DT, DYN1](
                    pos, quat, h, sah_view, adr, n, sah.bvhadr[mi],
                    sah.bvhnum[mi], eye, vec, lin.t * Scalar[DT](1.0001) + Scalar[DT](1e-12),
                )
                var c_half = ray_mesh_bvh[DT, DYN1](
                    pos, quat, h, sah_view, adr, n, sah.bvhadr[mi],
                    sah.bvhnum[mi], eye, vec, lin.t * Scalar[DT](0.5),
                )
                if c_at.t >= 0 or c_half.t >= 0:
                    bad_cut += 1
                if not _same(lin.t, lin.tri, lin.normal, lin.bu, lin.bv,
                             c_beyond.t, c_beyond.tri, c_beyond.normal,
                             c_beyond.bu, c_beyond.bv):
                    bad_cut += 1
            else:
                var c = ray_mesh_bvh[DT, DYN1](
                    pos, quat, h, sah_view, adr, n, sah.bvhadr[mi],
                    sah.bvhnum[mi], eye, vec, Scalar[DT](1e6),
                )
                if c.t >= 0:
                    bad_cut += 1
        print("  ", soup.names[mi], ":", n, "triangles,", mesh_hits, "of",
              NRAY, "rays hit,", mesh_ties, "exact ties")
        hits += mesh_hits
        ties += mesh_ties

    print("  walk time over", total, "rays: SAH", ns_sah // 1000, "us, median",
          ns_med // 1000, "us —", Float64(Int(Float64(ns_med) / Float64(ns_sah) * 100.0)) / 100.0,
          "x")
    print("  differ from the sweep: SAH", bad_sah, " median", bad_med,
          " cut", bad_cut)
    assert_true(hits > total // 4, "too few rays hit a mesh — the comparison is"
                                   " mostly misses and says little")
    assert_true(ties > 0, "no ray hit an exact tie — the tie rule is untested")
    assert_equal(bad_sah, 0, "the SAH tree disagrees with the linear sweep")
    assert_equal(bad_med, 0, "the median tree disagrees with the linear sweep")
    assert_equal(bad_cut, 0, "`tcut` changed an answer it may only drop")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
