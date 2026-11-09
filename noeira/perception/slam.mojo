"""Scan-to-map 2D SLAM (and localisation in a frozen map) — one loop.

    var s = ScanToMapSlam(odom_spec, seed, box..., z_lo, z_hi)     # mapping
    var s = ScanToMapSlam.frozen(odom_spec, seed, prior_occ, box...) # localising
    s.start(x, y, yaw)                       # the known start pose
    s.predict(vx, vy, wz, dt)                # every control step (true twist in)
    s.observe(ends, hits, ox, oy, oz, bx, by, byaw)   # every scan (see below)
    s.odo.x, s.odo.y, s.odo.yaw              # the estimate

THE LOOP, per scan: (1) the scan's hits as base-frame points
(`base_frame_points`); (2) `correlative_match` against the CURRENT map's
likelihood field, around the odometry's prediction — accepted when the mean
likelihood exceeds `score_min`; (3) MAPPING ONLY: the scan, moved to the
estimated pose (`retarget_scan`), joins the map if the motion filter lets it.

⚠ THE MOTION FILTER (Cartographer's) IS WHAT KEEPS THE MAP STILL. A scan joins
the map only after the estimate has moved `min_move` (0.2 m) or turned
`min_turn` (10 deg) since the last one that did. Every insertion carries its
match error into the map, and the next match follows the map: inserting every
scan (every 5 cm at 0.5 m/s) walked the map away — measured on the public
gate, RMS 4.6 cm and map precision 0.93 on one seed, against 0.8 cm and 0.998
with the filter.

⚠ NO LOOP CLOSURE. In one room a 360 deg sensor sees the anchoring walls from
everywhere, which is what this relies on; a building would need submaps and
loop closure (Cartographer, Hess et al. 2016) on top.

`observe` takes the simulator's world endpoints AND the true base pose only to
turn them back into the sensor's relative measurement (`base_frame_points`,
`retarget_scan`); the estimate never reads the truth.
"""

from std.math import sqrt, pi, abs

from .occupancy import VoxelMap
from .localise import (
    OdometrySpec, Odometry, LikelihoodField, correlative_match,
    base_frame_points, retarget_scan, wrap_angle,
)


struct ScanToMapSlam(Movable):
    var odo: Odometry
    var vm: VoxelMap
    var lf: LikelihoodField
    var mapping: Bool
    var lf_stale: Bool
    var z_lo: Float64
    var z_hi: Float64
    var min_move: Float64
    var min_turn: Float64
    var score_min: Float64
    var max_points: Int
    var sigma: Float64
    var last_x: Float64
    var last_y: Float64
    var last_yaw: Float64
    var inserted: Int
    var scans: Int
    var matched: Int
    var started: Bool

    def __init__(
        out self, spec: OdometrySpec, seed: Int,
        x0: Float64, y0: Float64, x1: Float64, y1: Float64,
        z_lo: Float64, z_hi: Float64, res: Float64 = 0.05,
        min_move: Float64 = 0.2, min_turn: Float64 = 10.0 * pi / 180.0,
        score_min: Float64 = 0.3, max_points: Int = 1500, sigma: Float64 = 0.05,
    ) raises:
        """MAPPING: an empty map, built from the scans as they come."""
        self.odo = Odometry(spec, seed)
        self.vm = VoxelMap(x0, y0, x1, y1, z_lo, z_hi, res, res)
        self.lf = LikelihoodField(
            List[Bool](length=self.vm.nx * self.vm.ny, fill=False),
            x0, y0, res, self.vm.nx, self.vm.ny, sigma,
        )
        self.mapping = True
        self.lf_stale = False
        self.z_lo = z_lo
        self.z_hi = z_hi
        self.min_move = min_move
        self.min_turn = min_turn
        self.score_min = score_min
        self.max_points = max_points
        self.sigma = sigma
        self.last_x = 0.0
        self.last_y = 0.0
        self.last_yaw = 0.0
        self.inserted = 0
        self.scans = 0
        self.matched = 0
        self.started = False

    @staticmethod
    def frozen(
        spec: OdometrySpec, seed: Int, prior_occ: List[Bool],
        x0: Float64, y0: Float64, x1: Float64, y1: Float64,
        z_lo: Float64, z_hi: Float64, res: Float64 = 0.05,
        score_min: Float64 = 0.3, max_points: Int = 1500, sigma: Float64 = 0.05,
    ) raises -> Self:
        """LOCALISATION ONLY: match against a prior map; nothing is inserted."""
        var s = Self(spec, seed, x0, y0, x1, y1, z_lo, z_hi, res,
                     score_min=score_min, max_points=max_points, sigma=sigma)
        s.mapping = False
        s.lf = LikelihoodField(prior_occ, x0, y0, res, s.vm.nx, s.vm.ny, sigma)
        return s^

    def start(mut self, x: Float64, y: Float64, yaw: Float64):
        """The known start pose. A mapping run's map is cleared."""
        self.odo.set_pose(x, y, yaw)
        # A frozen prior map can be matched from the first scan; a mapping
        # run has nothing to match until its first scans are in.
        self.started = not self.mapping
        self.inserted = 0
        self.scans = 0
        self.matched = 0
        if self.mapping:
            self.vm.clear()
            self.lf_stale = True

    def reseed(mut self, spec: OdometrySpec, seed: Int):
        """Fresh odometry errors (a new episode), keeping the map."""
        self.odo = Odometry(spec, seed)

    def predict(mut self, vx: Float64, vy: Float64, wz: Float64, dt: Float64):
        self.odo.step(vx, vy, wz, dt)

    def observe(
        mut self,
        mut ends: List[Float64], hits: List[Bool],
        ox: Float64, oy: Float64, oz: Float64,
        true_bx: Float64, true_by: Float64, true_byaw: Float64,
    ) raises:
        """One scan: match, then (mapping) maybe insert. `ends` is modified."""
        if self.lf_stale:
            self.lf = LikelihoodField(
                self.vm.occupied_2d(), self.vm.x0, self.vm.y0, self.vm.res,
                self.vm.nx, self.vm.ny, self.sigma,
            )
            self.lf_stale = False
        if self.started:
            var px = List[Float64]()
            var py = List[Float64]()
            base_frame_points(ends, hits, true_bx, true_by, true_byaw, self.z_lo, self.z_hi,
                              self.max_points, px, py)
            var m = correlative_match(self.lf, px, py, self.odo.x, self.odo.y, self.odo.yaw)
            self.scans += 1
            if m.score > self.score_min:
                self.odo.set_pose(m.x, m.y, m.yaw)
                self.matched += 1
        if not self.mapping:
            self.started = True
            return
        var moved = sqrt((self.odo.x - self.last_x) ** 2 + (self.odo.y - self.last_y) ** 2)
        var turned = abs(wrap_angle(self.odo.yaw - self.last_yaw))
        # The first TWO scans always go in: an obstacle needs two hits
        # (`VoxelMap.l_occ`), so one scan would leave nothing to match.
        if self.inserted < 2 or moved >= self.min_move or turned >= self.min_turn:
            var x = ox
            var y = oy
            retarget_scan(ends, x, y, true_bx, true_by, true_byaw, self.odo.x, self.odo.y, self.odo.yaw)
            self.vm.integrate_scan(x, y, oz, ends, hits)
            self.last_x = self.odo.x
            self.last_y = self.odo.y
            self.last_yaw = self.odo.yaw
            self.inserted += 1
            self.lf_stale = True
        self.started = True
