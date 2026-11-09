"""Places from perception: labelled returns -> object footprints -> approach points.

    var pm = PlaceMap(x0, y0, res, nx, ny, n_labels, z_lo, z_hi)
    pm.add_scan(endpoints, hits, labels)        # world frame (the ESTIMATED pose)
    var box = pm.footprint(label)               # an oriented box, or .n == 0
    var a = choose_approach(box, grid, standoff=1.2, toward_x, toward_y)

WHAT A LABEL IS (P3 step 1): an ORACLE detector — the simulator's geom id of
each return mapped to an object label by the caller (`labels[b]`, -1 = none).
It is the role a segmentation network plays on a real robot (on the camera,
or a semantic lidar), with perfect labels, so that this step measures the
GEOMETRY of finding a place: where the object is from what the sensor saw of
it, and where to stand. Step 2 replaces the oracle with an open-vocabulary
detector; nothing below changes.

FOOTPRINT: every labelled hit inside the height band marks its map cell for
that label; the label's cells give the minimum-area enclosing rectangle
(`oriented_box`). ⚠ ONLY WHAT WAS SEEN:
the back of a fridge against a wall is never hit, so the box's back face is
the farthest SEEN cell — the FRONT face, the one an approach stands off, is
the face that is seen.

APPROACH: four candidates, `standoff` metres out from the centre of each face
along its normal. A candidate counts if the navigation grid calls it free and
reachable; among those, the one whose face looks most toward `(toward_x,
toward_y)` wins — the caller passes the centre of the mapped free space, so
"the front" is the side facing into the room. ⚠ A HEURISTIC: right for what
stands against a wall (one free side), a guess for a free-standing table
whose sides are all free — measured, not hidden, in the gates.
"""

from std.math import sqrt, cos, sin, pi

from noeira.planners.nav2d import NavGrid, NAV_UNREACHABLE


comptime NO_FRAME: Float64 = -100.0
"""`PlaceMap.footprint_in`'s "fit the box" default. ⚠ A SENTINEL NO ANGLE CAN
TAKE: the first version used "negative = no frame", and a room yaw of -0.1 deg
silently turned the room-frame box back into a fitted one (the fridge's 40 deg
boxes again, from the door)."""


@fieldwise_init
struct PlaceBox(Copyable, ImplicitlyCopyable, Movable, Writable):
    """An object's footprint: centre, principal-axis yaw, half extents."""

    var cx: Float64
    var cy: Float64
    var yaw: Float64
    var hu: Float64
    """Half extent along the first axis (cos yaw, sin yaw)."""
    var hv: Float64
    """Half extent along the second axis (-sin yaw, cos yaw)."""
    var n: Int
    """Cells it was fitted to; 0 = the label was never seen."""


@fieldwise_init
struct Approach(Copyable, ImplicitlyCopyable, Movable, Writable):
    var x: Float64
    var y: Float64
    var face: Int
    """0..3: +u, -u, +v, -v; -1 = no candidate was free and reachable."""
    var n_feasible: Int
    """How many of the four faces were free and reachable (1 = unambiguous)."""
    var score: Float64


def oriented_box(xs: List[Float64], ys: List[Float64], cell: Float64) -> PlaceBox:
    """The MINIMUM-AREA rectangle around points (cell centres), searched over
    orientations in [0, 90) deg — every 1 deg, then every 0.05 deg around the
    best; each point covers a `cell`-wide square, so the extents grow by
    `cell / 2`.

    ⚠ NOT PRINCIPAL AXES. They were the first version, and a square object
    has none: its covariance's two eigenvalues are equal and the axes are
    whatever the noise makes them. Measured in the G1 room: the 0.7 x 0.7 m
    fridge came out rotated ~37 deg and 22 cm off-centre, and every one of 15
    episodes approached it diagonally, on the wrong side. The minimum-area
    rectangle of a partly seen object aligns with its seen faces — the face an
    approach stands off.
    """
    var n = len(xs)
    if n == 0:
        return PlaceBox(0.0, 0.0, 0.0, 0.0, 0.0, 0)
    var best = PlaceBox(0.0, 0.0, 0.0, 0.0, 0.0, n)
    var best_area = 1e300
    var lo = 0.0
    var hi = 90.0
    var step = 1.0
    for _pass in range(2):
        var a = lo
        while a < hi - 1e-12:
            var th = a * pi / 180.0
            var c = cos(th)
            var s = sin(th)
            var umin = 1e30
            var umax = -1e30
            var vmin = 1e30
            var vmax = -1e30
            for k in range(n):
                var u = c * xs[k] + s * ys[k]
                var v = -s * xs[k] + c * ys[k]
                umin = min(umin, u)
                umax = max(umax, u)
                vmin = min(vmin, v)
                vmax = max(vmax, v)
            var area = (umax - umin + cell) * (vmax - vmin + cell)
            if area < best_area - 1e-12:
                best_area = area
                var uc = 0.5 * (umin + umax)
                var vc = 0.5 * (vmin + vmax)
                best = PlaceBox(
                    c * uc - s * vc, s * uc + c * vc, th,
                    0.5 * (umax - umin) + 0.5 * cell, 0.5 * (vmax - vmin) + 0.5 * cell, n,
                )
            a += step
        var centre = best.yaw * 180.0 / pi
        lo = centre - 1.0
        hi = centre + 1.0
        step = 0.05
    return best


struct PlaceMap(Movable):
    """Per label, the map cells its returns fell in (height band only)."""

    var x0: Float64
    var y0: Float64
    var res: Float64
    var nx: Int
    var ny: Int
    var n_labels: Int
    var z_lo: Float64
    var z_hi: Float64
    var cells: List[List[Int]]
    var mark: List[Int32]
    """`[n_labels * nx * ny]`: hits per (label, cell)."""
    var min_hits: Int

    def __init__(
        out self, x0: Float64, y0: Float64, res: Float64, nx: Int, ny: Int,
        n_labels: Int, z_lo: Float64, z_hi: Float64, min_hits: Int = 2,
    ):
        """`min_hits`: a cell counts for a label after this many labelled hits —
        the map's own "two consistent hits" rule (`VoxelMap.l_occ`)."""
        self.x0 = x0
        self.y0 = y0
        self.res = res
        self.nx = nx
        self.ny = ny
        self.n_labels = n_labels
        self.z_lo = z_lo
        self.z_hi = z_hi
        self.cells = List[List[Int]]()
        for _ in range(n_labels):
            self.cells.append(List[Int]())
        self.mark = List[Int32](length=n_labels * nx * ny, fill=0)
        self.min_hits = min_hits

    def clear(mut self):
        for l in range(self.n_labels):
            self.cells[l].clear()
        for k in range(len(self.mark)):
            self.mark[k] = 0

    def add_scan(mut self, endpoints: List[Float64], hits: List[Bool], labels: List[Int]):
        for b in range(len(hits)):
            var l = labels[b]
            if not hits[b] or l < 0 or l >= self.n_labels:
                continue
            var z = endpoints[3 * b + 2]
            if z < self.z_lo or z > self.z_hi:
                continue
            var x = endpoints[3 * b]
            var y = endpoints[3 * b + 1]
            if x < self.x0 or y < self.y0:
                continue
            var i = Int((x - self.x0) / self.res)
            var j = Int((y - self.y0) / self.res)
            if i >= self.nx or j >= self.ny:
                continue
            var c = j * self.nx + i
            var k = l * self.nx * self.ny + c
            self.mark[k] += 1
            if Int(self.mark[k]) == self.min_hits:
                self.cells[l].append(c)

    def footprint(self, label: Int) -> PlaceBox:
        return self.footprint_in(label, List[Bool]())

    def footprint_in(self, label: Int, occ: List[Bool], frame_yaw: Float64 = NO_FRAME) -> PlaceBox:
        """The label's box over only the cells the MAP currently calls
        occupied (`occ`, `VoxelMap.occupied_2d`; empty = no mask).

        ⚠ A PLACE IS PART OF THE MAP'S OBSTACLES. The label's cells collect
        every labelled return ever seen — noise, and returns placed while the
        pose estimate was off — and the map clears those cells later; the
        minimum-area box is pulled by every one of them. Measured in the G1
        room: the fridge's box drifted 24 cm as the robot walked, and 9 of 15
        approaches switched to its west side.

        `frame_yaw` given (not `NO_FRAME`): the box's axes are FIXED at that yaw (the room's,
        `dominant_orientation`) instead of fitted — the MANHATTAN assumption
        (furniture aligned with the walls). ⚠ What a partial view needs: from
        the door, the G1 sees the fridge's west face and a sliver of its front;
        no fitted box knows the depth it did not see, and the minimum-area
        rectangle of that L-shape came out rotated, sending the approach
        diagonal. Wrong for furniture that is not aligned (the public room's
        cabinet is turned 17 deg)."""
        var use_mask = len(occ) == self.nx * self.ny
        var xs = List[Float64]()
        var ys = List[Float64]()
        for c in self.cells[label]:
            if use_mask and not occ[c]:
                continue
            xs.append(self.x0 + (Float64(c % self.nx) + 0.5) * self.res)
            ys.append(self.y0 + (Float64(c // self.nx) + 0.5) * self.res)
        if frame_yaw > NO_FRAME + 1.0:
            return box_in_frame(xs, ys, self.res, frame_yaw)
        return oriented_box(xs, ys, self.res)


def choose_approach(
    box: PlaceBox, mut grid: NavGrid, standoff: Float64,
    toward_x: Float64, toward_y: Float64,
    from_x: Float64, from_y: Float64,
) -> Approach:
    """The approach point of `box` (see the module docstring). A candidate is
    free (`grid.is_free`) and reachable from `(from_x, from_y)` — Dijkstra
    from the candidate must reach the start. ⚠ This re-runs `set_goal` on
    `grid` for each candidate: the caller sets its real goal afterwards."""
    if box.n == 0:
        return Approach(0.0, 0.0, -1, 0, -1.0)
    var c = cos(box.yaw)
    var s = sin(box.yaw)
    var nxs: List[Float64] = [c, -c, -s, s]
    var nys: List[Float64] = [s, -s, c, -c]
    var hs: List[Float64] = [box.hu, box.hu, box.hv, box.hv]
    var tx = toward_x - box.cx
    var ty = toward_y - box.cy
    var tn = sqrt(tx * tx + ty * ty)
    if tn > 0:
        tx /= tn
        ty /= tn
    var best = Approach(0.0, 0.0, -1, 0, -1e30)
    var feasible = 0
    for f in range(4):
        var ax = box.cx + nxs[f] * (hs[f] + standoff)
        var ay = box.cy + nys[f] * (hs[f] + standoff)
        if not grid.is_free(ax, ay):
            continue
        try:
            grid.set_goal(ax, ay)
        except:
            continue
        if grid.cost_to_go(from_x, from_y) >= 0.5 * NAV_UNREACHABLE:
            continue
        feasible += 1
        var sc = nxs[f] * tx + nys[f] * ty
        if sc > best.score:
            best = Approach(ax, ay, f, 0, sc)
    best.n_feasible = feasible
    return best


def free_centroid(occ: List[Bool], known: List[Bool], x0: Float64, y0: Float64,
                  res: Float64, nx: Int, ny: Int) -> Tuple[Float64, Float64]:
    """Centre of the mapped FREE space (known and not occupied): "into the room"."""
    var sx = 0.0
    var sy = 0.0
    var n = 0
    for j in range(ny):
        for i in range(nx):
            var c = j * nx + i
            if known[c] and not occ[c]:
                sx += x0 + (Float64(i) + 0.5) * res
                sy += y0 + (Float64(j) + 0.5) * res
                n += 1
    if n == 0:
        return (x0 + 0.5 * Float64(nx) * res, y0 + 0.5 * Float64(ny) * res)
    return (sx / Float64(n), sy / Float64(n))


def box_in_frame(xs: List[Float64], ys: List[Float64], cell: Float64, yaw: Float64) -> PlaceBox:
    """The box of the points with its axes FIXED at `yaw` (typically the
    room's, `dominant_orientation`): extents along (cos yaw, sin yaw) and its
    normal."""
    var n = len(xs)
    if n == 0:
        return PlaceBox(0.0, 0.0, yaw, 0.0, 0.0, 0)
    var c = cos(yaw)
    var s = sin(yaw)
    var umin = 1e30
    var umax = -1e30
    var vmin = 1e30
    var vmax = -1e30
    for k in range(n):
        var u = c * xs[k] + s * ys[k]
        var v = -s * xs[k] + c * ys[k]
        umin = min(umin, u)
        umax = max(umax, u)
        vmin = min(vmin, v)
        vmax = max(vmax, v)
    var uc = 0.5 * (umin + umax)
    var vc = 0.5 * (vmin + vmax)
    return PlaceBox(c * uc - s * vc, s * uc + c * vc, yaw,
                    0.5 * (umax - umin) + 0.5 * cell, 0.5 * (vmax - vmin) + 0.5 * cell, n)


def dominant_orientation(occ: List[Bool], x0: Float64, y0: Float64, res: Float64,
                         nx: Int, ny: Int) -> Float64:
    """The map's dominant axis in [0, 90) deg: the rotation at which the
    occupied cells project onto the FEWEST distinct `res`-wide bins along both
    axes — walls (long straight runs of cells) collapse to a few bins only
    when aligned. 1 deg steps, then 0.1 deg around the best."""
    var xs = List[Float64]()
    var ys = List[Float64]()
    for c in range(nx * ny):
        if occ[c]:
            xs.append(x0 + (Float64(c % nx) + 0.5) * res)
            ys.append(y0 + (Float64(c // nx) + 0.5) * res)
    if len(xs) == 0:
        return 0.0
    var half = 0.5 * sqrt(Float64(nx * nx + ny * ny)) * res + 1.0
    var nb = Int(2.0 * half / res) + 2
    var best = 0.0
    var best_count = 1 << 60
    var lo = 0.0
    var hi = 90.0
    var step = 1.0
    for _pass in range(2):
        var a = lo
        while a < hi - 1e-12:
            var th = a * pi / 180.0
            var c = cos(th)
            var s = sin(th)
            var bu = List[Bool](length=nb, fill=False)
            var bv = List[Bool](length=nb, fill=False)
            var count = 0
            for k in range(len(xs)):
                var iu = Int((c * xs[k] + s * ys[k] + half) / res)
                var iv = Int((-s * xs[k] + c * ys[k] + half) / res)
                if not bu[iu]:
                    bu[iu] = True
                    count += 1
                if not bv[iv]:
                    bv[iv] = True
                    count += 1
            if count < best_count:
                best_count = count
                best = th
            a += step
        var centre = best * 180.0 / pi
        lo = centre - 1.0
        hi = centre + 1.0
        step = 0.1
    # The refinement can step below 0 or past 90 deg: fold back into [0, 90).
    var q = 0.5 * pi
    while best < 0.0:
        best += q
    while best >= q:
        best -= q
    return best
