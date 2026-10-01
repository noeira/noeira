"""A 2D costmap for ground navigation: footprints -> clearance -> navigation function.

    var grid = NavGrid(footprints, xmin, ymin, xmax, ymax, res=0.05, inflate=0.45)
    grid.set_goal(gx, gy)              # Dijkstra over the free cells
    grid.cost_to_go(x, y)              # metres AROUND obstacles to the goal
    grid.clearance(x, y)               # metres to the nearest footprint

Robot-agnostic: it knows footprints (oriented rectangles and circles on the
ground plane), never bodies or robots. `planners/nav2d/scene.mojo` builds the
footprints from a parsed MJCF model; a hand-written list works as well.

⚠ WHY A NAVIGATION FUNCTION AND NOT A DISTANCE-TO-GOAL. A local optimiser
(MPPI, iLQR) minimising straight-line distance to the goal parks in front of
any obstacle that lies between, because every short rollout that goes around
first moves AWAY from the goal. The cost-to-go here is the length of the
shortest FREE path (Dijkstra, 8-connected, on cells whose clearance is at
least `inflate`), which has no local minimum except the goal itself — so a
planner with a short horizon still turns the right way at a table. It is the
"navigation function" of the classical literature, computed on a grid.

⚠ CLEARANCE IS EXACT, NOT A DISTANCE TRANSFORM. Each cell stores the true
distance from its centre to the nearest footprint (min over footprints). A
room has tens of footprints and a few tens of thousands of cells, so the
O(cells x footprints) pass costs milliseconds and avoids a chamfer
approximation's 8 % error exactly where it matters — at the inflation edge.
"""

from std.math import atan2, cos, sin, sqrt


comptime FP_RECT: Int = 0
comptime FP_CIRCLE: Int = 1

comptime NAV_UNREACHABLE: Float64 = 1.0e6
"""`cost_to_go` of a cell no free path reaches (or of a blocked cell)."""


struct Footprint(Copyable, ImplicitlyCopyable, Movable, Writable):
    """One obstacle on the ground plane: an oriented rectangle or a circle."""

    var kind: Int
    var cx: Float64
    var cy: Float64
    var hx: Float64     # rect half-extents along its own axes
    var hy: Float64
    var yaw: Float64    # rect orientation (rad)
    var r: Float64      # circle radius

    def __init__(
        out self, kind: Int, cx: Float64, cy: Float64, hx: Float64,
        hy: Float64, yaw: Float64, r: Float64,
    ):
        self.kind = kind
        self.cx = cx
        self.cy = cy
        self.hx = hx
        self.hy = hy
        self.yaw = yaw
        self.r = r

    @staticmethod
    def rect(cx: Float64, cy: Float64, hx: Float64, hy: Float64, yaw: Float64) -> Footprint:
        return Footprint(FP_RECT, cx, cy, hx, hy, yaw, 0.0)

    @staticmethod
    def circle(cx: Float64, cy: Float64, r: Float64) -> Footprint:
        return Footprint(FP_CIRCLE, cx, cy, 0.0, 0.0, 0.0, r)

    def distance(self, x: Float64, y: Float64) -> Float64:
        """Distance from (x, y) to the footprint; 0 inside it."""
        if self.kind == FP_CIRCLE:
            var d = sqrt((x - self.cx) ** 2 + (y - self.cy) ** 2) - self.r
            return d if d > 0.0 else 0.0
        var c = cos(self.yaw)
        var s = sin(self.yaw)
        var dx = x - self.cx
        var dy = y - self.cy
        var lx = c * dx + s * dy
        var ly = -s * dx + c * dy
        var ox = (lx if lx > 0.0 else -lx) - self.hx
        var oy = (ly if ly > 0.0 else -ly) - self.hy
        if ox < 0.0:
            ox = 0.0
        if oy < 0.0:
            oy = 0.0
        return sqrt(ox * ox + oy * oy)

    def write_to(self, mut w: Some[Writer]):
        if self.kind == FP_CIRCLE:
            w.write("circle(", self.cx, ", ", self.cy, ", r=", self.r, ")")
        else:
            w.write("rect(", self.cx, ", ", self.cy, ", ", self.hx, " x ",
                    self.hy, ", yaw=", self.yaw, ")")


struct _MinHeap(Movable):
    """Binary min-heap of (key, cell) for Dijkstra."""

    var key: List[Float64]
    var cell: List[Int]

    def __init__(out self):
        self.key = List[Float64]()
        self.cell = List[Int]()

    def size(self) -> Int:
        return len(self.key)

    def push(mut self, k: Float64, c: Int):
        self.key.append(k)
        self.cell.append(c)
        var i = len(self.key) - 1
        while i > 0:
            var p = (i - 1) // 2
            if self.key[p] <= self.key[i]:
                break
            self._swap(i, p)
            i = p

    def pop(mut self) -> Tuple[Float64, Int]:
        var k = self.key[0]
        var c = self.cell[0]
        var last = len(self.key) - 1
        self.key[0] = self.key[last]
        self.cell[0] = self.cell[last]
        _ = self.key.pop()
        _ = self.cell.pop()
        var n = len(self.key)
        var i = 0
        while True:
            var l = 2 * i + 1
            var r = l + 1
            var m = i
            if l < n and self.key[l] < self.key[m]:
                m = l
            if r < n and self.key[r] < self.key[m]:
                m = r
            if m == i:
                break
            self._swap(i, m)
            i = m
        return (k, c)

    def _swap(mut self, a: Int, b: Int):
        var tk = self.key[a]
        self.key[a] = self.key[b]
        self.key[b] = tk
        var tc = self.cell[a]
        self.cell[a] = self.cell[b]
        self.cell[b] = tc


struct NavGrid(Movable):
    """Clearance and cost-to-go on a regular grid over [xmin, xmax] x [ymin, ymax].

    Cell (i, j) has its centre at (x0 + (i + 0.5) res, y0 + (j + 0.5) res).
    A cell is FREE when its clearance is at least `inflate` — the robot's
    radius plus whatever margin the caller wants against its own run-out.
    """

    var x0: Float64
    var y0: Float64
    var res: Float64
    var nx: Int
    var ny: Int
    var inflate: Float64
    var clear: List[Float64]
    var c2g: List[Float64]
    var goal_x: Float64
    var goal_y: Float64
    var has_goal: Bool

    def __init__(
        out self, footprints: List[Footprint], xmin: Float64, ymin: Float64,
        xmax: Float64, ymax: Float64, res: Float64, inflate: Float64,
    ) raises:
        if res <= 0.0 or xmax <= xmin or ymax <= ymin:
            raise Error("NavGrid: empty bounds or non-positive resolution")
        self.x0 = xmin
        self.y0 = ymin
        self.res = res
        self.nx = Int((xmax - xmin) / res + 0.5)
        self.ny = Int((ymax - ymin) / res + 0.5)
        self.inflate = inflate
        self.clear = List[Float64](length=self.nx * self.ny, fill=1.0e9)
        self.c2g = List[Float64](length=self.nx * self.ny, fill=NAV_UNREACHABLE)
        self.goal_x = 0.0
        self.goal_y = 0.0
        self.has_goal = False
        for j in range(self.ny):
            var y = self.y0 + (Float64(j) + 0.5) * res
            for i in range(self.nx):
                var x = self.x0 + (Float64(i) + 0.5) * res
                var best = 1.0e9
                for k in range(len(footprints)):
                    var d = footprints[k].distance(x, y)
                    if d < best:
                        best = d
                self.clear[j * self.nx + i] = best

    def cell_of(self, x: Float64, y: Float64) -> Tuple[Int, Int]:
        var i = Int((x - self.x0) / self.res)
        var j = Int((y - self.y0) / self.res)
        if x < self.x0:
            i = -1
        if y < self.y0:
            j = -1
        return (i, j)

    def in_grid(self, i: Int, j: Int) -> Bool:
        return i >= 0 and j >= 0 and i < self.nx and j < self.ny

    def is_free_cell(self, i: Int, j: Int) -> Bool:
        return self.in_grid(i, j) and self.clear[j * self.nx + i] >= self.inflate

    def is_free(self, x: Float64, y: Float64) -> Bool:
        """Clearance at (x, y) itself (not the cell centre) >= inflate."""
        return self.clearance(x, y) >= self.inflate

    def set_goal(mut self, gx: Float64, gy: Float64) raises:
        """Dijkstra from the goal over free cells, 8-connected, in metres.

        ⚠ A goal inside an inflated obstacle is an ERROR, not a silent
        nearest-free substitution: a destination that the robot's own
        footprint cannot occupy means the scene or the inflation is wrong,
        and moving it would hide that.
        """
        var gc = self.cell_of(gx, gy)
        if not self.is_free_cell(gc[0], gc[1]):
            raise Error(
                "NavGrid.set_goal: (" + String(gx) + ", " + String(gy)
                + ") is not free (clearance "
                + String(self.clearance(gx, gy)) + " < inflate "
                + String(self.inflate) + ")"
            )
        for k in range(len(self.c2g)):
            self.c2g[k] = NAV_UNREACHABLE
        # the goal cell's own value: the exact distance from its centre
        var cx = self.x0 + (Float64(gc[0]) + 0.5) * self.res
        var cy = self.y0 + (Float64(gc[1]) + 0.5) * self.res
        var start = gc[1] * self.nx + gc[0]
        self.c2g[start] = sqrt((cx - gx) ** 2 + (cy - gy) ** 2)
        var heap = _MinHeap()
        heap.push(self.c2g[start], start)
        var diag = self.res * 1.4142135623730951
        while heap.size() > 0:
            var popped = heap.pop()
            var dist = popped[0]
            var c = popped[1]
            if dist > self.c2g[c]:
                continue
            var ci = c % self.nx
            var cj = c // self.nx
            for dj in range(-1, 2):
                for di in range(-1, 2):
                    if di == 0 and dj == 0:
                        continue
                    var ni = ci + di
                    var nj = cj + dj
                    if not self.is_free_cell(ni, nj):
                        continue
                    # no corner-cutting between two blocked neighbours
                    if di != 0 and dj != 0:
                        if not self.is_free_cell(ci + di, cj) or not self.is_free_cell(ci, cj + dj):
                            continue
                    var step = diag if (di != 0 and dj != 0) else self.res
                    var nd = dist + step
                    var n = nj * self.nx + ni
                    if nd < self.c2g[n]:
                        self.c2g[n] = nd
                        heap.push(nd, n)
        self.goal_x = gx
        self.goal_y = gy
        self.has_goal = True

    def _bilinear(self, field: List[Float64], x: Float64, y: Float64, outside: Float64) -> Float64:
        var fx = (x - self.x0) / self.res - 0.5
        var fy = (y - self.y0) / self.res - 0.5
        var i0 = Int(fx) if fx >= 0.0 else -1
        var j0 = Int(fy) if fy >= 0.0 else -1
        var tx = fx - Float64(i0)
        var ty = fy - Float64(j0)
        var acc = 0.0
        for dj in range(2):
            for di in range(2):
                var i = i0 + di
                var j = j0 + dj
                var w = (tx if di == 1 else 1.0 - tx) * (ty if dj == 1 else 1.0 - ty)
                var v = outside
                if self.in_grid(i, j):
                    v = field[j * self.nx + i]
                acc += w * v
        return acc

    def clearance(self, x: Float64, y: Float64) -> Float64:
        """Distance to the nearest footprint, interpolated; 0 outside the grid."""
        return self._bilinear(self.clear, x, y, 0.0)

    def cost_to_go(self, x: Float64, y: Float64) -> Float64:
        """Free-path distance to the goal, interpolated. Blocked or
        unreachable neighbours contribute `NAV_UNREACHABLE`, so the value
        climbs steeply toward an obstacle instead of averaging it away."""
        return self._bilinear(self.c2g, x, y, NAV_UNREACHABLE)

    def descent(self, x: Float64, y: Float64) -> Float64:
        """Heading (rad) of steepest descent of the cost-to-go at (x, y) —
        the direction a robot should face to make progress. Central
        differences at one cell."""
        var h = self.res
        var gx = self.cost_to_go(x + h, y) - self.cost_to_go(x - h, y)
        var gy = self.cost_to_go(x, y + h) - self.cost_to_go(x, y - h)
        return atan2(-gy, -gx)
