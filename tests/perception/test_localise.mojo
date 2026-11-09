"""P2a gate: dead reckoning vs dead reckoning + scan matching, in a known map.

    pixi run mojo run -I . tests/perception/test_localise.mojo

The map is P1's: the 16-ring at the head, the scripted tour, TRUE poses
(`test_map_vs_truth`'s configuration). The robot is then carried along two
laps of an ellipse through `assets/lidar_room.xml` at 0.5 m/s, heading along
the path, 50 Hz; every 5 steps (10 Hz) the 16-ring scans from the TRUE pose
(noise on) and the localiser matches the scan to the map around its own
estimate. The odometry integrates the true body-frame twist through an
`OdometrySpec` (`zoo/legged_good.odom`, `zoo/legged_poor.odom`).

⚠ THE ROBOT IS CARRIED, NOT WALKED: a standing G1 placed on the path each
step. The legs do not swing (their self-hits are filtered anyway) and the
odometry error is the spec's model, not a gait's. What is gated is the
localiser: map, matcher, the loop.

ASSERTED, for both odometry specs: with matching, RMS position error < 5 cm,
worst < 15 cm, worst heading error < 3 deg; and dead reckoning alone is at
least 3x worse in RMS (or the test is not exercising the correction).
"""

from std.math import abs, cos, sin, sqrt, pi, atan2
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite

from noeira.physics3d.fields import DynDims
from noeira.perception import (
    LidarMount,
    LidarSpec,
    load_lidar_spec,
    lidar_directions,
    lidar_scan_cpu,
    lidar_mount_world,
    VoxelMap,
    scan_endpoints_world,
    OdometrySpec,
    Odometry,
    load_odometry_spec,
    LikelihoodField,
    correlative_match,
    base_frame_points,
    retarget_scan,
    wrap_angle,
    ScanToMapSlam,
)
from mj_reference import DT, Built, quat_axis, set_pose_and_check_fk
from noeira.planners.nav2d import NavGrid, Footprint

comptime SCENE = "tests/perception/assets/lidar_room.xml"
comptime X0 = -3.2
comptime Y0 = -2.7
comptime X1 = 3.2
comptime Y1 = 2.7
comptime RES = 0.05
comptime Z_LO = 0.05
comptime Z_HI = 1.45
comptime DTC = 0.02       # 50 Hz control
comptime SCAN_EVERY = 5   # 10 Hz scans
comptime CX = 0.1
comptime CY = 0.0
comptime RX = 1.5
comptime RY = 0.8
comptime SPEED = 0.5


struct Scanner(Movable):
    var b: Built
    var mjm: PythonObject
    var mjd: PythonObject
    var spec: LidarSpec
    var mount: LidarMount
    var dirs: List[Float64]
    var self_geom: List[Bool]
    var nq: Int
    var frame: Int

    def __init__(out self) raises:
        var mujoco = Python.import_module("mujoco")
        self.mjm = mujoco.MjModel.from_xml_path(SCENE)
        self.mjd = mujoco.MjData(self.mjm)
        self.b = Built(SCENE)
        self.nq = self.b.dims.get_nq()
        var root = Int(py=self.mjm.jnt_bodyid[0])
        var torso = Int(py=mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_BODY, "robot_torso_link"))
        self.spec = load_lidar_spec("noeira/perception/zoo/rings16_100m.lidar")
        self.mount = LidarMount.on_body(torso, 0.05, 0.0, 0.47)
        self.dirs = lidar_directions(self.spec, 0)
        var ng = self.b.dims.get_ngeom()
        self.self_geom = List[Bool](length=ng, fill=False)
        for g in range(ng):
            self.self_geom[g] = Int(py=self.mjm.body_rootid[Int(py=self.mjm.geom_bodyid[g])]) == root
        self.frame = 0

    def place(mut self, x: Float64, y: Float64, yaw: Float64) raises:
        var q = List[Float64](length=self.nq, fill=0.0)
        q[0] = x
        q[1] = y
        q[2] = 0.793
        var qy = quat_axis(0, 0, 1, yaw)
        for k in range(4):
            q[3 + k] = qy[k]
        _ = set_pose_and_check_fk(self.b, self.mjm, self.mjd, q)

    def scan(mut self, mut ends: List[Float64], mut hits: List[Bool]) raises -> Tuple[Float64, Float64, Float64]:
        """One noisy scan from the current pose: world endpoints, and the
        sensor origin."""
        var pose = lidar_mount_world(self.b.d, self.mount)
        var r = List[Float64]()
        var g = List[Int]()
        lidar_scan_cpu[DT, DynDims, 1](self.b.d, self.b.m, self.spec, self.mount, self.dirs, r, g, 0, self.frame, self.frame)
        scan_endpoints_world(pose[0], pose[1], self.dirs, r, g, self.self_geom, 8.0, ends, hits)
        self.frame += 1
        return (pose[0].x, pose[0].y, pose[0].z)


def _prior_map(mut sc: Scanner) raises -> VoxelMap:
    var vm = VoxelMap(X0, Y0, X1, Y1, Z_LO, Z_HI, RES, RES)
    var wx: List[Float64] = [0.0, -1.4, 1.8, 0.3, 0.2, 2.2]
    var wy: List[Float64] = [0.0, 0.2, 0.6, 1.6, -1.5, -1.4]
    for w in range(len(wx)):
        for h in range(4):
            sc.place(wx[w], wy[w], Float64(h) * 0.5 * pi + 0.3 * Float64(w))
            for _ in range(3):
                var ends = List[Float64]()
                var hits = List[Bool]()
                var o = sc.scan(ends, hits)
                vm.integrate_scan(o[0], o[1], o[2], ends, hits)
    return vm^


def _path(t: Float64) -> Tuple[Float64, Float64, Float64, Float64, Float64]:
    """(x, y, yaw, v_forward, yaw_rate) on the ellipse at time t, constant
    speed approximated by a constant parameter rate (the speed varies 0.4-0.75
    m/s around the ellipse — fine for the gate)."""
    var omega = SPEED / sqrt(0.5 * (RX * RX + RY * RY))
    var th = omega * t
    var x = CX + RX * cos(th)
    var y = CY + RY * sin(th)
    var dx = -RX * omega * sin(th)
    var dy = RY * omega * cos(th)
    var ddx = -RX * omega * omega * cos(th)
    var ddy = -RY * omega * omega * sin(th)
    var v = sqrt(dx * dx + dy * dy)
    var yaw = atan2(dy, dx)
    var w = (dx * ddy - dy * ddx) / (v * v)
    return (x, y, yaw, v, w)


@fieldwise_init
struct Run(Copyable, Movable):
    var rms: Float64
    var worst: Float64
    var worst_yaw: Float64
    var matched: Int
    var scans: Int


def _run(mut sc: Scanner, prior: List[Bool], spec: OdometrySpec, seed: Int, mode: Int, mut last_map: List[List[Bool]]) raises -> Run:
    """mode 0: dead reckoning only. 1: localise in the frozen `prior`.
    2: SLAM from nothing. All through `perception.ScanToMapSlam`."""
    var omega = SPEED / sqrt(0.5 * (RX * RX + RY * RY))
    var n_steps = Int(2.0 * 2.0 * pi / omega / DTC)
    var s: ScanToMapSlam
    if mode == 1:
        s = ScanToMapSlam.frozen(spec, seed, prior, X0, Y0, X1, Y1, Z_LO, Z_HI, RES)
    else:
        s = ScanToMapSlam(spec, seed, X0, Y0, X1, Y1, Z_LO, Z_HI, RES)
    var p0 = _path(0.0)
    s.start(p0[0], p0[1], p0[2])
    var se = 0.0
    var worst = 0.0
    var worst_yaw = 0.0
    for k in range(0, n_steps + 1):
        var tp = _path(Float64(k) * DTC)
        if k > 0:
            var p = _path((Float64(k) - 0.5) * DTC)  # the twist over the step
            s.predict(p[3], 0.0, p[4], DTC)
        if mode != 0 and k % SCAN_EVERY == 0 and (mode == 2 or k > 0):
            sc.place(tp[0], tp[1], tp[2])
            var ends = List[Float64]()
            var hits = List[Bool]()
            var o = sc.scan(ends, hits)
            s.observe(ends, hits, o[0], o[1], o[2], tp[0], tp[1], tp[2])
        if k > 0:
            var e = sqrt((s.odo.x - tp[0]) ** 2 + (s.odo.y - tp[1]) ** 2)
            se += e * e
            worst = max(worst, e)
            worst_yaw = max(worst_yaw, abs(wrap_angle(s.odo.yaw - tp[2])))
    var r = Run(sqrt(se / Float64(n_steps)), worst, worst_yaw * 180.0 / pi, s.matched, s.scans)
    if mode == 2:
        last_map[0] = s.vm.occupied_2d()
        print("     (", s.inserted, "scans inserted into the map )")
    return r^


def test_localise_in_a_known_map() raises:
    var sc = Scanner()
    var vm = _prior_map(sc)
    var prior = vm.occupied_2d()
    var keep = List[List[Bool]]()
    keep.append(List[Bool]())
    var specs: List[String] = ["legged_good", "legged_poor"]
    for si in range(len(specs)):
        var spec = load_odometry_spec("noeira/perception/zoo/" + specs[si] + ".odom")
        for seed in range(3):
            var dr = _run(sc, prior, spec, 100 + seed, 0, keep)
            var lo = _run(sc, prior, spec, 100 + seed, 1, keep)
            print("  ", specs[si], "seed", seed, "| dead reckoning: RMS", dr.rms, "m, worst", dr.worst,
                  "m, yaw", dr.worst_yaw, "deg | + map: RMS", lo.rms, "m, worst", lo.worst, "m, yaw",
                  lo.worst_yaw, "deg, matched", lo.matched, "/", lo.scans)
            assert_true(lo.rms < 0.05 and lo.worst < 0.15 and lo.worst_yaw < 3.0,
                        specs[si] + ": localised RMS " + String(lo.rms) + " worst " + String(lo.worst)
                        + " yaw " + String(lo.worst_yaw))
            assert_true(dr.rms > 3.0 * lo.rms, specs[si] + ": dead reckoning is not worse — the gate is vacuous")


def test_slam_without_a_prior_map() raises:
    """P2b. Asserted for both odometry specs: RMS < 5 cm, worst < 15 cm, worst
    heading < 3 deg, better than dead reckoning (3x for `legged_poor`), and the map the run leaves behind has precision >= 0.95
    against the point-in-geom truth (it was built from ESTIMATED poses: a
    drifting estimate smears walls, which precision sees)."""
    var mujoco = Python.import_module("mujoco")
    Python.add_to_path("tests/perception")
    var truth_mod = Python.import_module("occupancy_truth")
    var sc = Scanner()
    var root = Int(py=sc.mjm.jnt_bodyid[0])
    var specs: List[String] = ["legged_good", "legged_poor"]
    for si in range(len(specs)):
        var spec = load_odometry_spec("noeira/perception/zoo/" + specs[si] + ".odom")
        for seed in range(3):
            var keep = List[List[Bool]]()
            keep.append(List[Bool]())
            var none = List[Bool]()
            var dr = _run(sc, none, spec, 100 + seed, 0, keep)
            var lo = _run(sc, none, spec, 100 + seed, 2, keep)
            var occ = keep[0].copy()
            var vm = VoxelMap(X0, Y0, X1, Y1, Z_LO, Z_HI, RES, RES)
            var t = truth_mod.truth_cells(sc.mjm, sc.mjd, X0, Y0, RES, vm.nx, vm.ny, Z_LO, Z_HI, root)
            var mapped = 0
            var right = 0
            for j in range(vm.ny):
                for i in range(vm.nx):
                    if not occ[j * vm.nx + i]:
                        continue
                    var x = X0 + (Float64(i) + 0.5) * RES
                    var y = Y0 + (Float64(j) + 0.5) * RES
                    if abs(x) > 3.05 or abs(y) > 2.55:
                        continue
                    mapped += 1
                    var near = False
                    for dj in range(-1, 2):
                        for di in range(-1, 2):
                            var a = i + di
                            var bb = j + dj
                            if a >= 0 and bb >= 0 and a < vm.nx and bb < vm.ny and Bool(py=t[bb * vm.nx + a]):
                                near = True
                    if near:
                        right += 1
            var prec = Float64(right) / Float64(max(1, mapped))
            print("  SLAM", specs[si], "seed", seed, "| dead reckoning RMS", dr.rms, "| SLAM: RMS", lo.rms,
                  "m, worst", lo.worst, "m, yaw", lo.worst_yaw, "deg, matched", lo.matched, "/", lo.scans,
                  "| map precision", prec, "(", mapped, "cells )")
            assert_true(lo.rms < 0.05 and lo.worst < 0.15 and lo.worst_yaw < 3.0,
                        "SLAM " + specs[si] + ": RMS " + String(lo.rms) + " worst " + String(lo.worst))
            assert_true(prec >= 0.95, "SLAM " + specs[si] + ": map precision " + String(prec))
            # Good odometry barely drifts over 15 m (3.7-7.8 cm RMS), so there
            # SLAM need only not be worse; the poor spec is where the 3x
            # correction must show, or the loop is not doing anything.
            if specs[si] == "legged_poor":
                assert_true(dr.rms > 3.0 * lo.rms, "SLAM: dead reckoning is not 3x worse — vacuous")
            else:
                assert_true(lo.rms < dr.rms, "SLAM is worse than dead reckoning with good odometry")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
