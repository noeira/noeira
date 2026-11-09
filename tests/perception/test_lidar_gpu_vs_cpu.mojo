"""`BatchedLidar` (GPU, one thread per lane x ray) against `lidar_scan_cpu`.

    pixi run mojo run -I . tests/perception/test_lidar_gpu_vs_cpu.mojo

The CPU path is gated against MuJoCo (`test_lidar_vs_mujoco`); this closes the
loop for the device. Both legs run the SAME `lidar_cast` / `lidar_return`, so
what this catches is the target: Metal's silent per-thread-array miscompiles
(four in this engine so far), operand packing, lane indexing.

⚠ FLOAT32 ON BOTH LEGS. Metal has no double; a float64 CPU leg would report
the dtype as a GPU defect.

⚠ NOISE ON IN THE SECOND PASS. The draws are counter-based (seed, frame, lane,
ray), so the device must reproduce the host's noise beam for beam — a stream
RNG would pass a noise-free gate and fail here.

Eight lanes in eight different poses, so a lane-indexing slip (reading lane 0's
pose for every lane) reads as a mismatch rather than agreement.
"""

from std.math import abs, cos, sin
from std.sys import has_accelerator
from max.gpu.host import DeviceContext
from std.testing import assert_true, TestSuite

from noeira.physics3d.fields import Data, Model, DynDims, init_hfield_data
from noeira.physics3d.parser.full_parser import parse_xml_full
from noeira.physics3d.parser.runtime_load import read_model_source, build_model_runtime
from noeira.physics3d.kinematics.forward_kinematics import forward_kinematics
from noeira.perception import (
    LidarSpec,
    LidarMount,
    load_lidar_spec,
    lidar_directions,
    lidar_scan_cpu,
    BatchedLidar,
)
from mj_reference import Lcg, mesh_budget_dims

comptime GT = DType.float32
comptime B = 8
comptime SCENE = "tests/perception/assets/lidar_room.xml"
# `robot_torso_link` in this scene (the MuJoCo gate looks it up by name).
comptime TORSO = 24
# The pelvis: the robot's root body.
comptime PELVIS = 1


def _compare(
    name: String,
    gpu_r: List[Float64], gpu_g: List[Int],
    cpu_r: List[Float64], cpu_g: List[Int],
) raises:
    """⚠ THE DISTANCE BOUND SCALES WITH RANGE, AND ITS SIZE IS ARITHMETIC.
    Near a silhouette a quadric's ray root (`ray_capsule`, `ray_sphere`, ...)
    comes from the square root of a nearly-cancelled discriminant, which keeps
    about HALF of float32's digits: the error grows like sqrt(eps32) * r =
    3.45e-4 * r. Two targets contracting different products into FMAs land
    anywhere inside that. Measured worst: 1.67e-4 m at r = 3.59 m, on
    `lamp_arm` (a 3 cm capsule) — 0.13 of the bound. The float64 path is the
    one gated tight, against MuJoCo."""
    var split = 0
    var wrong = 0
    var hits = 0
    var worst = 0.0
    var worst_at = -1
    var over = 0
    for k in range(len(cpu_r)):
        if (gpu_r[k] >= 0) != (cpu_r[k] >= 0):
            split += 1
            continue
        if cpu_r[k] < 0:
            continue
        hits += 1
        if gpu_g[k] != cpu_g[k]:
            wrong += 1
            continue
        var e = abs(gpu_r[k] - cpu_r[k])
        if e > worst:
            worst = e
            worst_at = k
        if e > 3.45e-4 * max(1.0, cpu_r[k]):
            over += 1
    if worst_at >= 0:
        print("     worst beam", worst_at, " r", cpu_r[worst_at], " geom", cpu_g[worst_at])
    print("  ", name, ": beams", len(cpu_r), " hits", hits, " splits", split,
          " wrong geom", wrong, " worst |dr|", worst)
    # A quarter, not a half: with the real `range_min` a PELVIS-mounted lidar
    # loses more than half its beams to the robot's own legs inside 0.3 m
    # (measured: 106 377 of 230 400 kept) — a mounting fact, not a defect.
    assert_true(hits > len(cpu_r) // 4, name + ": too few hits")
    # ⚠ A FEW GRAZING SPLITS ARE float32 ARITHMETIC, NOT A DEFECT: two targets
    # rounding a tangent beam differently. A defect moves thousands (see the
    # MuJoCo gate's falsification table); allow 1 in 10 000.
    assert_true(split + wrong <= len(cpu_r) // 10000,
                name + ": " + String(split) + " splits, " + String(wrong) + " wrong geoms")
    assert_true(over == 0, name + ": " + String(over) + " beams beyond sqrt(eps32) * r")


def test_lidar_gpu_matches_cpu() raises:
    comptime if not has_accelerator():
        print("  SKIP — no accelerator on this machine")
        return
    var ctx = DeviceContext()
    var dims = mesh_budget_dims(SCENE)
    var src = read_model_source(SCENE)
    var fmd = parse_xml_full(src[0], src[1])
    var m = Model[GT, DynDims](dims)
    build_model_runtime[GT](fmd, dims, m)
    var d = Data[GT, DynDims, B](dims)
    init_hfield_data(d, m)

    var nq = dims.get_nq()
    var rng = Lcg(0x6E0)
    for env in range(B):
        var o = env * nq
        for k in range(nq):
            d.qpos.data[o + k] = 0
        d.qpos.data[o + 0] = Scalar[GT](rng.sym(2.0))
        d.qpos.data[o + 1] = Scalar[GT](rng.sym(1.6))
        d.qpos.data[o + 2] = Scalar[GT](0.79)
        var yaw = rng.sym(3.0)
        # A pure yaw (w, 0, 0, z): enough to put each lane facing elsewhere.
        d.qpos.data[o + 3] = Scalar[GT](cos(0.5 * yaw))
        d.qpos.data[o + 6] = Scalar[GT](sin(0.5 * yaw))
        for k in range(7, nq):
            d.qpos.data[o + k] = Scalar[GT](rng.sym(0.6))
    forward_kinematics["cpu", GT, DynDims, B](d, m)
    m.upload_all(ctx)
    d.upload_all(ctx)
    ctx.synchronize()

    var specs = List[LidarSpec]()
    specs.append(load_lidar_spec("noeira/perception/zoo/planar_2d_12m.lidar"))
    specs.append(load_lidar_spec("noeira/perception/zoo/rings16_100m.lidar"))
    specs.append(load_lidar_spec("noeira/perception/zoo/spiral_mid360_class.lidar"))
    var mounts = List[LidarMount]()
    mounts.append(LidarMount.on_body(TORSO, 0.05, 0.0, 0.47))
    mounts.append(LidarMount(PELVIS, 0.12, 0.03, 0.10, 0.968, 0.06, 0.17, 0.17))

    for s in range(len(specs)):
        for noisy in range(2):
            var spec = specs[s].copy()
            if noisy == 0:
                spec.range_sigma = 0.0
                spec.dropout = 0.0
                spec.range_min = 0.0
                spec.range_max = 1.0e6
            else:
                spec.dropout = 0.05  # so the dropout branch runs on every spec
            var mount = mounts[s % len(mounts)]
            var lid = BatchedLidar[GT, DynDims, B](ctx, spec, mount)
            var frame = 3
            var seed = 77
            lid.scan(ctx, d, m, frame, seed)
            var gr = List[Float64]()
            var gg = List[Int]()
            lid.download(ctx, gr, gg)

            var dirs = lidar_directions(spec, frame if spec.pattern == 2 else 0)
            var cr = List[Float64]()
            var cg = List[Int]()
            for env in range(B):
                var r = List[Float64]()
                var g = List[Int]()
                lidar_scan_cpu[GT, DynDims, B](
                    d, m, spec, mount, dirs, r, g, env, frame, seed
                )
                for k in range(len(r)):
                    cr.append(r[k])
                    cg.append(g[k])
            _compare(
                spec.name + (" noisy" if noisy == 1 else " clean"),
                gr, gg, cr, cg,
            )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
