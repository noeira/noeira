"""P1 gate (a): the map a sensor builds on a scripted tour, against the truth.

    pixi run mojo run -I . tests/perception/test_map_vs_truth.mojo

The G1 stands at six waypoints of `assets/lidar_room.xml`, four headings each,
three scans per stop (an obstacle needs two consistent hits, `VoxelMap.l_occ`),
POSE KNOWN (P1: the map is the only thing under test). Each sensor
configuration scans at every stop into a fresh `VoxelMap` (band 0.05-1.45 m,
5 cm voxels), and the 2D projection is compared with `occupancy_truth.py`'s —
point-in-geom on MuJoCo's own poses, nothing of ours.

THREE NUMBERS PER CONFIGURATION:
  recall     truth SURFACE cells (occupied, with a free 4-neighbour: what a
             sensor can see) that the map marks within one cell
  precision  map-occupied cells with a truth cell within one cell
  DANGEROUS  cells the planner would call FREE from the map (clearance >=
             0.45 m, `NavGrid.set_obstacle_cells`) that it calls BLOCKED from
             the truth — the cells a robot could be sent into an obstacle by.
             This is the one that matters for navigation.

The configurations are the start of the purchase bench (plan §3): the same
2D lidar at head height and at hip height, a 16-ring and a MID-360-class 3D
lidar at the head, and the D435-class head camera.

ASSERTED: an ORACLE sensor (dense rings -85..+30 deg, no noise) gives
recall >= 0.90 (the rest occluded, or a 2 cm tabletop rim), precision >= 0.98,
and no cell the map frees by more than one cell that the truth blocks — the
pipeline is right.
The zoo configurations are MEASURED and printed: what a real sensor at a real
mount cannot see is the answer the bench exists to give, not a test failure
(the head 16-ring cannot see a 30 cm ball nearer than (1.3 - 0.3) / tan 15 deg
= 3.7 m; the MID-360 class looks no lower than -7 deg).
"""

from std.math import abs, cos, sin, pi
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite

from noeira.math3d import Vec3 as Vec3Generic, Quat as QuatGeneric
from noeira.physics3d.fields import DynDims
from noeira.planners.nav2d import NavGrid, Footprint
from noeira.perception import (
    LidarSpec,
    LidarMount,
    DepthCameraSpec,
    LIDAR_SPIRAL,
    load_lidar_spec,
    load_depth_spec,
    lidar_directions,
    lidar_scan_cpu,
    lidar_mount_world,
    depth_directions,
    depth_scan_cpu,
    depth_mount_from_mjcf_camera,
    VoxelMap,
    scan_endpoints_world,
    depth_endpoints_world,
)
from mj_reference import DT, Built, quat_axis, set_pose_and_check_fk

comptime SCENE = "tests/perception/assets/lidar_room.xml"
comptime X0 = -3.2
comptime Y0 = -2.7
comptime X1 = 3.2
comptime Y1 = 2.7
comptime RES = 0.05
comptime Z_LO = 0.05
comptime Z_HI = 1.45
comptime INFLATE = 0.45
# The SCORED region: the room inside its walls plus the walls' inner cell row.
# Cells behind a wall are never seen and never driven into; the D435's 6.5 cm
# depth noise at 3 m does put returns there (measured), which is not a map error.
comptime IN_X = 3.05
comptime IN_Y = 2.55


struct Config(Copyable, Movable):
    var name: String
    var is_depth: Bool
    var lidar: LidarSpec
    var depth: DepthCameraSpec
    var mount: LidarMount
    var frames: Int

    def __init__(out self, name: String, var lidar: LidarSpec, mount: LidarMount, frames: Int = 1):
        self.name = name
        self.is_depth = False
        self.lidar = lidar^
        self.depth = DepthCameraSpec()
        self.mount = mount
        self.frames = frames

    def __init__(out self, name: String, var depth: DepthCameraSpec, mount: LidarMount, frames: Int = 1):
        self.name = name
        self.is_depth = True
        self.lidar = LidarSpec()
        self.depth = depth^
        self.mount = mount
        self.frames = frames


@fieldwise_init
struct Score(Copyable, Movable):
    var recall: Float64
    var precision: Float64
    var dangerous: Int
    var dangerous_margin: Int
    var lost: Int
    var known: Float64


def _near(occ: List[Bool], nx: Int, ny: Int, i: Int, j: Int) -> Bool:
    for dj in range(-1, 2):
        for di in range(-1, 2):
            var a = i + di
            var b = j + dj
            if a >= 0 and b >= 0 and a < nx and b < ny and occ[b * nx + a]:
                return True
    return False


def _scored(i: Int, j: Int) -> Bool:
    var x = X0 + (Float64(i) + 0.5) * RES
    var y = Y0 + (Float64(j) + 0.5) * RES
    return x > -IN_X and x < IN_X and y > -IN_Y and y < IN_Y


def _score(occ: List[Bool], known: List[Bool], truth: List[Bool], nx: Int, ny: Int) raises -> Score:
    var surf = 0
    var found = 0
    var mapped = 0
    var right = 0
    var nknown = 0
    for j in range(ny):
        for i in range(nx):
            var c = j * nx + i
            if not _scored(i, j):
                continue
            if known[c]:
                nknown += 1
            if truth[c]:
                var edge = False
                if i > 0 and not truth[c - 1]:
                    edge = True
                if i < nx - 1 and not truth[c + 1]:
                    edge = True
                if j > 0 and not truth[c - nx]:
                    edge = True
                if j < ny - 1 and not truth[c + nx]:
                    edge = True
                if edge:
                    surf += 1
                    if _near(occ, nx, ny, i, j):
                        found += 1
            if occ[c]:
                mapped += 1
                if _near(truth, nx, ny, i, j):
                    right += 1
    var gm = NavGrid(List[Footprint](), X0, Y0, X1, Y1, RES, INFLATE)
    var gt = NavGrid(List[Footprint](), X0, Y0, X1, Y1, RES, INFLATE)
    gm.set_obstacle_cells(occ)
    gt.set_obstacle_cells(truth)
    var dangerous = 0
    var dangerous_margin = 0
    var lost = 0
    var nscored = 0
    for j in range(ny):
        for i in range(nx):
            if not _scored(i, j):
                continue
            nscored += 1
            var fm = gm.is_free_cell(i, j)
            var ft = gt.is_free_cell(i, j)
            if fm and not ft:
                dangerous += 1
                # Free by MORE than one cell beyond the inflation: not a
                # discretisation tie at the edge (the tolerance recall and
                # precision use too).
                if gm.clear[j * nx + i] >= INFLATE + RES:
                    dangerous_margin += 1
            if ft and not fm:
                lost += 1
    return Score(
        Float64(found) / Float64(max(1, surf)),
        Float64(right) / Float64(max(1, mapped)),
        dangerous, dangerous_margin, lost, Float64(nknown) / Float64(max(1, nscored)),
    )


def test_map_vs_truth() raises:
    var mujoco = Python.import_module("mujoco")
    var np = Python.import_module("numpy")
    Python.add_to_path("tests/perception")
    var truth_mod = Python.import_module("occupancy_truth")
    var mjm = mujoco.MjModel.from_xml_path(SCENE)
    var mjd = mujoco.MjData(mjm)
    var b = Built(SCENE)
    var nq = b.dims.get_nq()
    var root = Int(py=mjm.jnt_bodyid[0])
    var torso = Int(py=mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "robot_torso_link"))
    var cam = Int(py=mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_CAMERA, "robot_head_camera"))
    var self_geom = List[Bool](length=b.dims.get_ngeom(), fill=False)
    for g in range(b.dims.get_ngeom()):
        self_geom[g] = Int(py=mjm.body_rootid[Int(py=mjm.geom_bodyid[g])]) == root

    var vm0 = VoxelMap(X0, Y0, X1, Y1, Z_LO, Z_HI, RES, RES)
    var nx = vm0.nx
    var ny = vm0.ny

    var cfgs = List[Config]()
    # THE ORACLE: rings from -85 to +30 deg every ~1 deg, 0.5 deg azimuth,
    # no noise, no dropout. It sees every surface a head can see, so its map
    # tests the PIPELINE (DDA, update rule, projection, clearance), not a
    # sensor's reach. Asserted.
    var oracle = load_lidar_spec("noeira/perception/zoo/rings16_100m.lidar")
    oracle.name = "oracle"
    oracle.h_samples = 720
    oracle.v_min_deg = -85.0
    oracle.v_max_deg = 30.0
    oracle.v_samples = 116
    oracle.range_sigma = 0.0
    oracle.dropout = 0.0
    oracle.range_min = 0.0
    cfgs.append(Config("ORACLE dense rings @ head", oracle^, LidarMount.on_body(torso, 0.05, 0.0, 0.47), 3))
    cfgs.append(Config("2D lidar @ head 1.3 m", load_lidar_spec("noeira/perception/zoo/planar_2d_12m.lidar"),
                       LidarMount.on_body(torso, 0.05, 0.0, 0.47), 3))
    cfgs.append(Config("2D lidar @ hip 0.45 m", load_lidar_spec("noeira/perception/zoo/planar_2d_12m.lidar"),
                       LidarMount.on_body(root, 0.15, 0.0, -0.34), 3))
    cfgs.append(Config("16-ring @ head", load_lidar_spec("noeira/perception/zoo/rings16_100m.lidar"),
                       LidarMount.on_body(torso, 0.05, 0.0, 0.47), 3))
    cfgs.append(Config("MID-360 class @ head", load_lidar_spec("noeira/perception/zoo/spiral_mid360_class.lidar"),
                       LidarMount.on_body(torso, 0.05, 0.0, 0.47), 3))
    cfgs.append(Config("D435 424x240 head_camera", load_depth_spec("noeira/perception/zoo/d435_class_424x240.depth"),
                       depth_mount_from_mjcf_camera(b.m, cam), 3))

    var wx: List[Float64] = [0.0, -1.4, 1.8, 0.3, 0.2, 2.2]
    var wy: List[Float64] = [0.0, 0.2, 0.6, 1.6, -1.5, -1.4]
    var truth = List[Bool]()
    var scores = List[Score]()
    for ci in range(len(cfgs)):
        ref cfg = cfgs[ci]
        var vm = VoxelMap(X0, Y0, X1, Y1, Z_LO, Z_HI, RES, RES)
        var ldirs = lidar_directions(cfg.lidar, 0) if not cfg.is_depth else List[Float64]()
        var ddirs = depth_directions(cfg.depth) if cfg.is_depth else List[Float64]()
        var frame = 0
        for w in range(len(wx)):
            for h in range(4):
                var q = List[Float64](length=nq, fill=0.0)
                q[0] = wx[w]
                q[1] = wy[w]
                q[2] = 0.793
                var qy = quat_axis(0, 0, 1, Float64(h) * 0.5 * pi + 0.3 * Float64(w))
                for k in range(4):
                    q[3 + k] = qy[k]
                var fk = set_pose_and_check_fk(b, mjm, mjd, q)
                assert_true(fk < 1e-12, "forward kinematics differ")
                if len(truth) == 0:
                    # The room is static: one truth for the whole run.
                    var t = truth_mod.truth_cells(mjm, mjd, X0, Y0, RES, nx, ny, Z_LO, Z_HI, root)
                    for c in range(nx * ny):
                        truth.append(Bool(py=t[c]))
                var pose = lidar_mount_world(b.d, cfg.mount)
                var ends = List[Float64]()
                var hits = List[Bool]()
                if cfg.is_depth:
                    for _ in range(cfg.frames):
                        var z = List[Float64]()
                        var g = List[Int]()
                        depth_scan_cpu[DT, DynDims, 1](b.d, b.m, cfg.depth, cfg.mount, ddirs, z, g, 0, frame, frame)
                        depth_endpoints_world(pose[0], pose[1], ddirs, z, g, self_geom, 1, ends, hits)
                        vm.integrate_scan(pose[0].x, pose[0].y, pose[0].z, ends, hits)
                        frame += 1
                else:
                    for _ in range(cfg.frames):
                        var dirs = ldirs.copy()
                        if cfg.lidar.pattern == LIDAR_SPIRAL:
                            dirs = lidar_directions(cfg.lidar, frame)
                        var r = List[Float64]()
                        var g = List[Int]()
                        lidar_scan_cpu[DT, DynDims, 1](b.d, b.m, cfg.lidar, cfg.mount, dirs, r, g, 0, frame, frame)
                        scan_endpoints_world(pose[0], pose[1], dirs, r, g, self_geom,
                                             min(cfg.lidar.range_max, 8.0), ends, hits)
                        vm.integrate_scan(pose[0].x, pose[0].y, pose[0].z, ends, hits)
                        frame += 1
        var sc = _score(vm.occupied_2d(), vm.known_2d(), truth, nx, ny)
        print("  ", cfg.name, ": recall", sc.recall, " precision", sc.precision,
              " DANGEROUS", sc.dangerous, "(", sc.dangerous_margin, "beyond 1 cell )  lost", sc.lost,
              " known", sc.known)
        scores.append(sc.copy())

    var ntruth = 0
    for c in range(len(truth)):
        if truth[c]:
            ntruth += 1
    print("   truth: ", ntruth, "occupied cells of", nx * ny)
    ref so = scores[0]
    # The pipeline's claims are PRECISION and NO CELL FREED BEYOND ONE CELL.
    # Recall is bounded at 0.90 and every miss is accounted for (measured,
    # 2026-11-09): 31 cells seen from no waypoint — the back of the cabinet
    # against the NE corner with the wall cells behind it, the back of the
    # pouf toward the SW corner — and the tabletop's PERIMETER, a 2 cm slab
    # in 5 cm voxels that grazing beams from the other stops cross more than
    # they hit. Its price is at most one cell per side of the table, which is
    # what `dangerous_margin == 0` checks.
    assert_true(so.recall >= 0.90 and so.precision >= 0.98,
                "the ORACLE's map: recall " + String(so.recall) + " precision " + String(so.precision))
    assert_true(so.dangerous_margin == 0, "the ORACLE's map leaves " + String(so.dangerous_margin)
                + " cells free by more than a cell that the truth blocks")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
