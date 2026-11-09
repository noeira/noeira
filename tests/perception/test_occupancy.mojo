"""`VoxelMap` and `NavGrid.set_obstacle_cells`, against brute force.

    pixi run mojo run -I . tests/perception/test_occupancy.mojo

* the 3D DDA visits exactly the voxels a fine sampler of the segment finds —
  plus only voxels the segment crosses for less than the sampler's step;
* one scan: a hit voxel is occupied, the voxels before it free, each updated
  once however many beams cross it; a voxel hit by one beam and crossed by
  one stays hit (1/2 >= the 0.25 hit fraction), and a voxel hit by ONE beam
  of the hundred that cross it (a noise outlier) is judged free;
* one hit is not an obstacle, two are; the clamp lets the map change its mind
  in a bounded number of scans;
* the distance transform equals the brute-force nearest occupied centre;
* `set_obstacle_cells` clearance = centre distance - res*sqrt(2)/2.
"""

from std.math import abs, sqrt, floor
from std.testing import assert_true, assert_equal, TestSuite

from noeira.perception import VoxelMap
from noeira.planners.nav2d import NavGrid, Footprint, cell_distance_transform
from mj_reference import Lcg


def _seg_len_in_voxel(
    vm: VoxelMap, v: Int,
    ox: Float64, oy: Float64, oz: Float64, px: Float64, py: Float64, pz: Float64,
) -> Float64:
    """Length of the segment inside voxel `v` (slab clip)."""
    var i = v % vm.nx
    var j = (v // vm.nx) % vm.ny
    var k = v // (vm.nx * vm.ny)
    var lo: List[Float64] = [vm.x0 + Float64(i) * vm.res, vm.y0 + Float64(j) * vm.res, vm.z0 + Float64(k) * vm.zres]
    var hi: List[Float64] = [lo[0] + vm.res, lo[1] + vm.res, lo[2] + vm.zres]
    var o: List[Float64] = [ox, oy, oz]
    var d: List[Float64] = [px - ox, py - oy, pz - oz]
    var t0 = 0.0
    var t1 = 1.0
    for a in range(3):
        if abs(d[a]) < 1e-15:
            if o[a] < lo[a] or o[a] > hi[a]:
                return 0.0
            continue
        var ta = (lo[a] - o[a]) / d[a]
        var tb = (hi[a] - o[a]) / d[a]
        if ta > tb:
            var t = ta
            ta = tb
            tb = t
        t0 = max(t0, ta)
        t1 = min(t1, tb)
    if t1 <= t0:
        return 0.0
    return (t1 - t0) * sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)


def test_dda_matches_a_fine_sampler() raises:
    var vm = VoxelMap(-1.0, -1.0, 1.0, 1.0, 0.0, 0.6, res=0.1, zres=0.07)
    var rng = Lcg(0xDDA)
    comptime STEP = 2e-4
    var extra_total = 0
    for ci in range(400):
        # Origins inside and outside the box, ends anywhere (some outside).
        var ox = rng.sym(1.4)
        var oy = rng.sym(1.4)
        var oz = rng.u01() * 0.9 - 0.15
        var px = rng.sym(1.4)
        var py = rng.sym(1.4)
        var pz = rng.u01() * 0.9 - 0.15
        if ci % 5 == 0:
            pz = oz  # axis-parallel in z: the zero-direction branch
        var got = vm.traverse(ox, oy, oz, px, py, pz)
        var seen = List[Bool](length=vm.nx * vm.ny * vm.nz, fill=False)
        for v in got:
            assert_true(not seen[v], "the DDA visited a voxel twice")
            seen[v] = True
        var end = vm.voxel_of(px, py, pz)
        var end_v = -1
        if vm.inside(end[0], end[1], end[2]):
            end_v = vm.index(end[0], end[1], end[2])
        assert_true(end_v < 0 or not seen[end_v], "the end voxel must not be crossed")
        var L = sqrt((px - ox) ** 2 + (py - oy) ** 2 + (pz - oz) ** 2)
        var n = Int(L / STEP) + 1
        var sampled = List[Bool](length=len(seen), fill=False)
        for s in range(n):
            var t = (Float64(s) + 0.5) / Float64(n)
            var c = vm.voxel_of(ox + t * (px - ox), oy + t * (py - oy), oz + t * (pz - oz))
            if vm.inside(c[0], c[1], c[2]):
                var v = vm.index(c[0], c[1], c[2])
                if v != end_v:
                    sampled[v] = True
        for v in range(len(seen)):
            if sampled[v]:
                assert_true(seen[v], "case " + String(ci) + ": the DDA missed a voxel the sampler found")
            elif seen[v]:
                # Visited but not sampled: only a sliver shorter than the step.
                var ln = _seg_len_in_voxel(vm, v, ox, oy, oz, px, py, pz)
                assert_true(ln < 2.0 * STEP, "case " + String(ci) + ": visited a voxel the segment crosses for "
                            + String(ln) + " m")
                extra_total += 1
    print("   400 segments: DDA == sampler, plus", extra_total, "sliver voxels (< 2 x step)")


def test_one_scan_marks_hits_and_clears_once() raises:
    var vm = VoxelMap(0.0, -1.0, 4.0, 1.0, 0.0, 1.0, res=0.1, zres=0.1)
    # A fan of 200 beams from (0.05, 0, 0.55) to a wall at x = 3.05.
    var ends = List[Float64]()
    var hits = List[Bool]()
    for b in range(200):
        var y = -0.9 + 1.8 * Float64(b) / 199.0
        ends.append(3.05)  # inside voxel 30, not on its boundary
        ends.append(y)
        ends.append(0.55)
        hits.append(True)
    vm.integrate_scan(0.05, 0.0, 0.55, ends, hits)
    var k = 5  # z in [0.5, 0.6)
    var wall = vm.lo[vm.index(30, 10, k)]
    var near = vm.lo[vm.index(1, 10, k)]  # crossed by every beam
    print("   after one scan: wall", wall, " voxel next to the sensor", near)
    assert_true(wall == vm.l_hit, "a hit voxel must get exactly one +l_hit")
    assert_true(near == vm.l_miss, "a voxel crossed by 200 beams must get ONE miss, got " + String(near))
    # Hit and crossed in the same scan: a beam ending in a voxel another beam crosses.
    var vm2 = VoxelMap(0.0, -1.0, 4.0, 1.0, 0.0, 1.0, res=0.1, zres=0.1)
    var e2: List[Float64] = [1.55, 0.0, 0.55, 3.0, 0.0, 0.55]
    var h2: List[Bool] = [True, True]
    vm2.integrate_scan(0.05, 0.0, 0.55, e2, h2)
    assert_true(vm2.lo[vm2.index(15, 10, 5)] == vm2.l_hit, "a voxel hit in this scan was cleared by it")
    # A noise outlier: one beam of 100 ends in a voxel the other 99 cross.
    var vm4 = VoxelMap(0.0, -1.0, 4.0, 1.0, 0.0, 1.0, res=0.1, zres=0.1)
    var e4 = List[Float64]()
    var h4 = List[Bool]()
    for b in range(100):
        e4.append(1.55 if b == 0 else 3.05)
        e4.append(0.0)
        e4.append(0.55)
        h4.append(True)
    vm4.integrate_scan(0.05, 0.0, 0.55, e4, h4)
    assert_true(vm4.lo[vm4.index(15, 10, 5)] == vm4.l_miss, "one outlier among 100 beams made an obstacle")
    # A no-return clears its whole length, including the last voxel.
    var vm3 = VoxelMap(0.0, -1.0, 4.0, 1.0, 0.0, 1.0, res=0.1, zres=0.1)
    var e3: List[Float64] = [2.05, 0.0, 0.55]
    var h3: List[Bool] = [False]
    vm3.integrate_scan(0.05, 0.0, 0.55, e3, h3)
    assert_true(vm3.lo[vm3.index(20, 10, 5)] == vm3.l_miss, "a no-return's last voxel is not cleared")
    # One hit is evidence, not yet an obstacle (l_occ = 1.0 > l_hit); a
    # second consistent scan confirms it.
    assert_true(not vm.occupied_2d()[10 * vm.nx + 30], "one hit already made an obstacle")
    vm.integrate_scan(0.05, 0.0, 0.55, ends, hits)
    var occ = vm.occupied_2d()
    assert_true(occ[10 * vm.nx + 30] and not occ[10 * vm.nx + 15], "the 2D projection is wrong")


def test_the_map_changes_its_mind() raises:
    var vm = VoxelMap(0.0, -1.0, 4.0, 1.0, 0.0, 1.0, res=0.1, zres=0.1)
    var hit: List[Float64] = [2.05, 0.0, 0.55]
    var past: List[Float64] = [3.55, 0.0, 0.55]
    var t: List[Bool] = [True]
    for _ in range(20):
        vm.integrate_scan(0.05, 0.0, 0.55, hit, t)
    var v = vm.index(20, 10, 5)
    assert_true(vm.lo[v] == vm.l_max, "repeated hits must clamp at l_max")
    # The object is gone: beams now pass through to x = 3.5.
    var n = 0
    while vm.lo[v] > vm.l_occ:
        vm.integrate_scan(0.05, 0.0, 0.55, past, t)
        n += 1
    var bound = Int(Float64(vm.l_max - vm.l_occ) / Float64(-vm.l_miss)) + 1
    print("   free again after", n, "scans (bound", bound, ")")
    assert_true(n <= bound, "the clamp should bound how long a stale obstacle lives")


def test_distance_transform_is_exact() raises:
    var rng = Lcg(0xED7)
    for _ in range(5):
        var nx = 37
        var ny = 23
        var occ = List[Bool](length=nx * ny, fill=False)
        for c in range(nx * ny):
            occ[c] = rng.u01() < 0.03
        var got = cell_distance_transform(occ, nx, ny)
        var worst = 0.0
        for j in range(ny):
            for i in range(nx):
                var best = 1.0e10
                for jj in range(ny):
                    for ii in range(nx):
                        if occ[jj * nx + ii]:
                            best = min(best, sqrt(Float64((i - ii) ** 2 + (j - jj) ** 2)))
                worst = max(worst, abs(best - got[j * nx + i]) if best < 1e9 else 0.0)
        assert_true(worst < 1e-9, "distance transform off by " + String(worst))


def test_clearance_from_cells() raises:
    var g = NavGrid(List[Footprint](), 0.0, 0.0, 2.0, 2.0, 0.05, 0.3)
    var occ = List[Bool](length=g.nx * g.ny, fill=False)
    occ[20 * g.nx + 20] = True
    g.set_obstacle_cells(occ)
    for k in range(1, 10):
        var c = g.clear[20 * g.nx + 20 + k]
        var want = Float64(k) * 0.05 - 0.05 * 0.7071067811865476
        assert_true(abs(c - want) < 1e-12, "clearance " + String(c) + " vs " + String(want))
    assert_true(g.clear[20 * g.nx + 20] == 0.0, "an occupied cell has clearance 0")
    # With a goal set, a map update re-runs Dijkstra: a wall cutting the grid
    # makes the far side unreachable.
    g.set_goal(0.5, 1.0)
    var before = g.cost_to_go(1.6, 1.0)
    for j in range(g.ny):
        occ[j * g.nx + 25] = True
    g.set_obstacle_cells(occ)
    var after = g.cost_to_go(1.6, 1.0)
    print("   cost-to-go across the grid: before", before, " after a wall", after)
    assert_true(before < 2.0 and after > 1e5, "a map update did not re-run the navigation function")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
