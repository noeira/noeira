"""A map from range data: a log-odds voxel band, projected to a 2D obstacle grid.

    var vm = VoxelMap(xmin, ymin, xmax, ymax, z_lo=0.05, z_hi=1.5, res=0.05, zres=0.05)
    vm.integrate_scan(ox, oy, oz, endpoints, hits)   # world frame, one scan
    var occ = vm.occupied_2d()                        # [ny * nx], column-major in x
    nav_grid.set_obstacle_cells(occ)                  # planners/nav2d

WHY 3D BEFORE 2D. A 2D costmap is a projection, and projecting the RAYS first
is wrong: a beam passing over a table at 1.0 m says nothing about the
tabletop at 0.74, yet a 2D ray-trace would clear the table's cells. Here a
beam clears only the voxels it actually crosses, and a column is an obstacle
when ANY voxel of it inside the height band is occupied. The band is what a
ground robot can hit: from just above the floor (`z_lo`) to its own height
(`z_hi`). The floor itself is below the band, so a floor return clears the
air in front of it and marks nothing.

THE UPDATE: OctoMap's log-odds (Hornung et al. 2013) — per voxel `+l_hit` when
it is judged hit, `+l_miss` (negative) when judged free, clamped to
[l_min, l_max] so the map can change its mind; occupied when > 0. ⚠ ONE
UPDATE PER VOXEL PER SCAN: a voxel near the sensor is crossed by thousands of
beams of one scan, and counting each would let a single scan erase anything.

⚠⚠ AND THE JUDGEMENT IS A FRACTION, NOT "ANY HIT WINS". OctoMap marks a voxel
hit if ANY beam of the scan ends in it. With a dense noisy sensor that makes
noise into walls: a D435-class camera's depth error lifts a few floor returns
per image above the band floor, and "any hit" turned those voxels into
obstacles — measured on the gate scene, 4 804 free cells lost and precision
0.43. Here a voxel is HIT when at least `hit_fraction` (0.25) of the beams that
reached it ended in it: a real surface stops most beams entering its voxel; a
noise outlier is a few percent of beams that otherwise fly through. Beams that
end in a voxel count only as hits there, never as misses.

⚠ AN OBSTACLE NEEDS MORE THAN ONE HIT: occupied means log-odds above `l_occ`
= 1.0, more than one hit's worth (0.847), i.e. two consistent hits. A real
surface is seen scan after scan (10 Hz) and is confirmed one scan later; a
noise outlier rarely lands in the same voxel twice. Measured in the room with
a D435-class head camera: with "one hit is an obstacle", 9 of 60 episodes
stopped because a single far-floor outlier (one hit, 5-10 cm up, 2-3 m away —
where only a few pixels reach a voxel, so the hit fraction cannot help) fell
inside the goal's clearance.

A NO-RETURN BEAM (lidar `-1`) clears up to `max_clear` metres along its
direction — the sensor saw nothing that far — and marks nothing. A depth
camera's 0 pixel is NOT a no-return (too close, shadow, grazing, dropout) and
is skipped entirely; the caller decides by what it passes as endpoints.

⚠ THE TRAVERSAL IS AMANATIDES & WOO's 3D DDA, clipped to the map box: every
voxel the segment crosses, none it does not. `test_occupancy` checks it
against a brute-force sampler.
"""

from std.math import floor, sqrt, abs

from noeira.math3d import Vec3 as Vec3Generic, Quat as QuatGeneric

comptime _F64 = DType.float64


struct VoxelMap(Movable):
    var x0: Float64
    var y0: Float64
    var z0: Float64
    var res: Float64
    var zres: Float64
    var nx: Int
    var ny: Int
    var nz: Int
    var lo: List[Float32]
    var stamp: List[Int32]
    var n_hit: List[Int32]
    var n_miss: List[Int32]
    var touched: List[Int]
    var scan_id: Int32
    var hit_fraction: Float64
    var l_occ: Float32
    var l_hit: Float32
    var l_miss: Float32
    var l_min: Float32
    var l_max: Float32
    var scans: Int

    def __init__(
        out self,
        xmin: Float64, ymin: Float64, xmax: Float64, ymax: Float64,
        z_lo: Float64, z_hi: Float64,
        res: Float64 = 0.05, zres: Float64 = 0.05,
        # OctoMap's defaults: hit p 0.7, miss p 0.4, clamp p [0.12, 0.97].
        l_hit: Float32 = 0.847, l_miss: Float32 = -0.405,
        l_min: Float32 = -1.992, l_max: Float32 = 3.476,
        hit_fraction: Float64 = 0.25,
        l_occ: Float32 = 1.0,
    ) raises:
        if res <= 0.0 or zres <= 0.0 or xmax <= xmin or ymax <= ymin or z_hi <= z_lo:
            raise Error("VoxelMap: empty box or non-positive resolution")
        if not (l_hit > 0 and l_miss < 0 and l_min < 0 and l_max > 0):
            raise Error("VoxelMap: need l_hit > 0 > l_miss and l_min < 0 < l_max")
        self.x0 = xmin
        self.y0 = ymin
        self.z0 = z_lo
        self.res = res
        self.zres = zres
        # Same cell count rule as `NavGrid`, so the two grids line up cell for cell.
        self.nx = Int((xmax - xmin) / res + 0.5)
        self.ny = Int((ymax - ymin) / res + 0.5)
        self.nz = max(1, Int((z_hi - z_lo) / zres + 0.5))
        var n = self.nx * self.ny * self.nz
        self.lo = List[Float32](length=n, fill=0)
        self.stamp = List[Int32](length=n, fill=-1)
        self.n_hit = List[Int32](length=n, fill=0)
        self.n_miss = List[Int32](length=n, fill=0)
        self.touched = List[Int]()
        self.scan_id = 0
        if not (hit_fraction > 0.0 and hit_fraction <= 1.0):
            raise Error("VoxelMap: hit_fraction must be in (0, 1]")
        self.hit_fraction = hit_fraction
        if not (l_occ >= 0 and l_occ < l_max):
            raise Error("VoxelMap: need 0 <= l_occ < l_max")
        self.l_occ = l_occ
        self.l_hit = l_hit
        self.l_miss = l_miss
        self.l_min = l_min
        self.l_max = l_max
        self.scans = 0

    @always_inline
    def index(self, i: Int, j: Int, k: Int) -> Int:
        return (k * self.ny + j) * self.nx + i

    def voxel_of(self, x: Float64, y: Float64, z: Float64) -> Tuple[Int, Int, Int]:
        return (
            Int(floor((x - self.x0) / self.res)),
            Int(floor((y - self.y0) / self.res)),
            Int(floor((z - self.z0) / self.zres)),
        )

    @always_inline
    def inside(self, i: Int, j: Int, k: Int) -> Bool:
        return i >= 0 and j >= 0 and k >= 0 and i < self.nx and j < self.ny and k < self.nz

    def clear(mut self):
        for v in range(len(self.lo)):
            self.lo[v] = 0
        self.scans = 0

    def _touch(mut self, v: Int):
        if self.stamp[v] != self.scan_id:
            self.stamp[v] = self.scan_id
            self.n_hit[v] = 0
            self.n_miss[v] = 0
            self.touched.append(v)

    def _add(mut self, v: Int, d: Float32):
        var x = self.lo[v] + d
        if x < self.l_min:
            x = self.l_min
        if x > self.l_max:
            x = self.l_max
        self.lo[v] = x

    def traverse(
        self,
        ox: Float64, oy: Float64, oz: Float64,
        px: Float64, py: Float64, pz: Float64,
    ) -> List[Int]:
        """Voxel indices the segment o -> p crosses inside the box, in order,
        EXCLUDING the voxel that contains p. (Amanatides & Woo, clipped.)"""
        var out = List[Int]()
        var dx = px - ox
        var dy = py - oy
        var dz = pz - oz
        # Clip the parameter range [0, 1] to the box (slab method).
        var t0 = 0.0
        var t1 = 1.0
        var lo3: List[Float64] = [self.x0, self.y0, self.z0]
        var hi3: List[Float64] = [
            self.x0 + Float64(self.nx) * self.res,
            self.y0 + Float64(self.ny) * self.res,
            self.z0 + Float64(self.nz) * self.zres,
        ]
        var o3: List[Float64] = [ox, oy, oz]
        var d3: List[Float64] = [dx, dy, dz]
        for a in range(3):
            if abs(d3[a]) < 1e-15:
                if o3[a] < lo3[a] or o3[a] >= hi3[a]:
                    return out^
            else:
                var ta = (lo3[a] - o3[a]) / d3[a]
                var tb = (hi3[a] - o3[a]) / d3[a]
                if ta > tb:
                    var t = ta
                    ta = tb
                    tb = t
                t0 = max(t0, ta)
                t1 = min(t1, tb)
        if t0 >= t1:
            return out^
        # Voxel of the entry point (nudged inside) and of the end point.
        var eps = 1e-9
        var ex = ox + (t0 + eps) * dx
        var ey = oy + (t0 + eps) * dy
        var ez = oz + (t0 + eps) * dz
        var c = self.voxel_of(ex, ey, ez)
        var i = min(max(c[0], 0), self.nx - 1)
        var j = min(max(c[1], 0), self.ny - 1)
        var k = min(max(c[2], 0), self.nz - 1)
        var e = self.voxel_of(px, py, pz)
        var size3: List[Float64] = [self.res, self.res, self.zres]
        var step = List[Int](length=3, fill=0)
        var tmax = List[Float64](length=3, fill=1e30)
        var tdelta = List[Float64](length=3, fill=1e30)
        var cur: List[Int] = [i, j, k]
        for a in range(3):
            if d3[a] > 0:
                step[a] = 1
                var edge = lo3[a] + Float64(cur[a] + 1) * size3[a]
                tmax[a] = (edge - o3[a]) / d3[a]
                tdelta[a] = size3[a] / d3[a]
            elif d3[a] < 0:
                step[a] = -1
                var edge = lo3[a] + Float64(cur[a]) * size3[a]
                tmax[a] = (edge - o3[a]) / d3[a]
                tdelta[a] = -size3[a] / d3[a]
        var guard = self.nx + self.ny + self.nz + 3
        for _ in range(guard):
            if cur[0] == e[0] and cur[1] == e[1] and cur[2] == e[2]:
                break
            out.append(self.index(cur[0], cur[1], cur[2]))
            var a = 0
            if tmax[1] < tmax[a]:
                a = 1
            if tmax[2] < tmax[a]:
                a = 2
            if tmax[a] > t1:
                break
            cur[a] += step[a]
            if cur[a] < 0 or (a == 0 and cur[a] >= self.nx) or (a == 1 and cur[a] >= self.ny) or (a == 2 and cur[a] >= self.nz):
                break
            tmax[a] += tdelta[a]
        return out^

    def integrate_scan(
        mut self,
        ox: Float64, oy: Float64, oz: Float64,
        endpoints: List[Float64],
        hits: List[Bool],
    ) raises:
        """One scan from one origin, world frame. `endpoints` is 3 floats per
        beam: the return (hit) or the clearing limit (no return)."""
        var n = len(hits)
        if len(endpoints) != 3 * n:
            raise Error("VoxelMap.integrate_scan: endpoints must hold 3 floats per beam")
        self.scan_id += 1
        self.touched.clear()
        # Count, per voxel, the beams that end in it and the beams that cross it.
        for b in range(n):
            var path = self.traverse(
                ox, oy, oz, endpoints[3 * b], endpoints[3 * b + 1], endpoints[3 * b + 2]
            )
            for v in path:
                self._touch(v)
                self.n_miss[v] += 1
            var c = self.voxel_of(endpoints[3 * b], endpoints[3 * b + 1], endpoints[3 * b + 2])
            if self.inside(c[0], c[1], c[2]):
                var v = self.index(c[0], c[1], c[2])
                self._touch(v)
                if hits[b]:
                    self.n_hit[v] += 1
                else:
                    # A no-return's last voxel is crossed, not hit.
                    self.n_miss[v] += 1
        # One update per voxel: hit if enough of the beams that reached it
        # ended in it.
        for v in self.touched:
            var h = Float64(self.n_hit[v])
            var m = Float64(self.n_miss[v])
            if h > 0 and h >= self.hit_fraction * (h + m):
                self._add(v, self.l_hit)
            else:
                self._add(v, self.l_miss)
        self.scans += 1

    def occupied_2d(self) -> List[Bool]:
        """`[ny * nx]` (row j, column i): any voxel of the column above
        `l_occ`."""
        var out = List[Bool](length=self.nx * self.ny, fill=False)
        for k in range(self.nz):
            for j in range(self.ny):
                for i in range(self.nx):
                    if self.lo[self.index(i, j, k)] > self.l_occ:
                        out[j * self.nx + i] = True
        return out^

    def known_2d(self) -> List[Bool]:
        """`[ny * nx]`: any voxel of the column ever updated."""
        var out = List[Bool](length=self.nx * self.ny, fill=False)
        for k in range(self.nz):
            for j in range(self.ny):
                for i in range(self.nx):
                    if self.lo[self.index(i, j, k)] != 0:
                        out[j * self.nx + i] = True
        return out^


def scan_endpoints_world(
    origin: Vec3Generic[_F64],
    q: QuatGeneric[_F64],
    dirs: List[Float64],
    ranges: List[Float64],
    geom_ids: List[Int],
    self_geom: List[Bool],
    max_clear: Float64,
    mut endpoints: List[Float64],
    mut hits: List[Bool],
):
    """A lidar scan (`ranges`, -1 = no return) to world endpoints for
    `VoxelMap.integrate_scan`.

    ⚠ A BEAM THAT HIT THE ROBOT ITSELF IS DROPPED — neither an obstacle nor a
    clearing (`self_geom[g]` true for the robot's own geoms). On a real robot
    this is the self-filter that removes points inside the robot's own
    kinematic model; here the geom id plays that part. A no-return clears up
    to `max_clear` metres.
    """
    var kept = List[Int]()
    scan_endpoints_world_labelled(
        origin, q, dirs, ranges, geom_ids, self_geom, max_clear, endpoints, hits, kept
    )


def scan_endpoints_world_labelled(
    origin: Vec3Generic[_F64],
    q: QuatGeneric[_F64],
    dirs: List[Float64],
    ranges: List[Float64],
    geom_ids: List[Int],
    self_geom: List[Bool],
    max_clear: Float64,
    mut endpoints: List[Float64],
    mut hits: List[Bool],
    mut kept_geom: List[Int],
):
    """`scan_endpoints_world`, plus the geom id of every KEPT beam (-1 for a
    no-return) — what an oracle detector labels (`places.PlaceMap`)."""
    endpoints = List[Float64]()
    hits = List[Bool]()
    kept_geom = List[Int]()
    var n = len(ranges)
    for b in range(n):
        var g = geom_ids[b]
        if g >= 0 and g < len(self_geom) and self_geom[g]:
            continue
        var d = q.rotate_vec(Vec3Generic[_F64](dirs[3 * b], dirs[3 * b + 1], dirs[3 * b + 2]))
        var r = ranges[b]
        var hit = r >= 0.0
        if not hit:
            r = max_clear
        endpoints.append(origin.x + r * d.x)
        endpoints.append(origin.y + r * d.y)
        endpoints.append(origin.z + r * d.z)
        hits.append(hit)
        kept_geom.append(g if hit else -1)


def depth_endpoints_world(
    origin: Vec3Generic[_F64],
    q: QuatGeneric[_F64],
    dirs: List[Float64],
    depth: List[Float64],
    geom_ids: List[Int],
    self_geom: List[Bool],
    stride: Int,
    mut endpoints: List[Float64],
    mut hits: List[Bool],
):
    """A depth image (planar depth, 0 = none) to world endpoints, every
    `stride`-th pixel. A 0 pixel is SKIPPED (not a no-return; see the module
    docstring), and so is a pixel on the robot itself. `dirs` is
    `depth_directions(spec)` (unit; `x` = the depth-to-length factor)."""
    endpoints = List[Float64]()
    hits = List[Bool]()
    var n = len(depth)
    var s = max(1, stride)
    for p in range(0, n, s):
        var z = depth[p]
        if z <= 0.0:
            continue
        var g = geom_ids[p]
        if g >= 0 and g < len(self_geom) and self_geom[g]:
            continue
        var t = z / dirs[3 * p]
        var d = q.rotate_vec(Vec3Generic[_F64](dirs[3 * p], dirs[3 * p + 1], dirs[3 * p + 2]))
        endpoints.append(origin.x + t * d.x)
        endpoints.append(origin.y + t * d.y)
        endpoints.append(origin.z + t * d.z)
        hits.append(True)
