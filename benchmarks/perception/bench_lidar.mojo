"""Lidar scan throughput: host loop vs one GPU launch over many lanes.

    pixi run mojo run -I . benchmarks/perception/bench_lidar.mojo          # CPU only
    pixi run -e apple mojo run -I . benchmarks/perception/bench_lidar.mojo
    pixi run -e nvidia mojo run -I . benchmarks/perception/bench_lidar.mojo

Scene: the gate's (`tests/perception/assets/lidar_room.xml`, the G1 with its
visual meshes + 18 room geoms = 90 geoms), a head mount, every lane in its own
pose. Reports beams/s and what that means as frames/s for one robot, against
each zoo spec's own rate.

⚠ `ray_model` IS A LINEAR SCAN OVER GEOMS (each mesh then walks its BVH), so
the cost per beam grows with the scene. These numbers are for THIS scene and
say nothing about a 1000-geom one.
"""

from std.math import cos, sin
from std.sys import has_accelerator
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from noeira.physics3d.fields import Data, Model, DynDims, init_hfield_data
from noeira.physics3d.parser.full_parser import parse_xml_full
from noeira.physics3d.parser.runtime_load import (
    read_model_source,
    dims_from_flat,
    build_model_runtime,
)
from noeira.physics3d.kinematics.forward_kinematics import forward_kinematics
from noeira.perception import (
    LidarSpec,
    LidarMount,
    load_lidar_spec,
    lidar_directions,
    lidar_scan_cpu,
    BatchedLidar,
)

comptime GT = DType.float32
comptime LANES = 64
comptime SCENE = "tests/perception/assets/lidar_room.xml"
comptime TORSO = 24


def _dims_for(path: String) raises -> DynDims:
    var src = read_model_source(path)
    var fmd = parse_xml_full(src[0], src[1])
    var verts = 65536
    var tris = 1
    for _ in range(30):
        var dims = dims_from_flat(fmd, max_contacts=16, nmesh_verts=verts, nmesh_tri=tris)
        var m = Model[GT, DynDims](dims)
        try:
            build_model_runtime[GT](fmd, dims, m)
            return dims
        except e:
            var msg = String(e)
            var k = msg.find("Pass nmesh_tri=")
            if k >= 0:
                var bytes = msg.as_bytes()
                var j = k + 15
                var v = 0
                while j < len(bytes) and bytes[j] >= 48 and bytes[j] <= 57:
                    v = v * 10 + Int(bytes[j]) - 48
                    j += 1
                tris = v
            elif msg.find("mesh vertex capacity") >= 0:
                verts *= 2
            else:
                raise e^
    raise Error("could not size the mesh budgets")


def main() raises:
    var dims = _dims_for(SCENE)
    var src = read_model_source(SCENE)
    var fmd = parse_xml_full(src[0], src[1])
    var m = Model[GT, DynDims](dims)
    build_model_runtime[GT](fmd, dims, m)
    var d = Data[GT, DynDims, LANES](dims)
    init_hfield_data(d, m)
    var nq = dims.get_nq()
    for env in range(LANES):
        var o = env * nq
        for k in range(nq):
            d.qpos.data[o + k] = 0
        var a = Float64(env) * 0.37
        d.qpos.data[o + 0] = Scalar[GT](1.8 * cos(a))
        d.qpos.data[o + 1] = Scalar[GT](1.5 * sin(a))
        d.qpos.data[o + 2] = Scalar[GT](0.79)
        d.qpos.data[o + 3] = Scalar[GT](cos(0.5 * a))
        d.qpos.data[o + 6] = Scalar[GT](sin(0.5 * a))
    forward_kinematics["cpu", GT, DynDims, LANES](d, m)
    var mount = LidarMount.on_body(TORSO, 0.05, 0.0, 0.47)

    var specs = List[LidarSpec]()
    specs.append(load_lidar_spec("noeira/perception/zoo/planar_2d_12m.lidar"))
    specs.append(load_lidar_spec("noeira/perception/zoo/rings16_100m.lidar"))
    specs.append(load_lidar_spec("noeira/perception/zoo/spiral_mid360_class.lidar"))

    print("scene", SCENE, " ngeom", dims.get_ngeom(), " nmesh_tri", dims.get_nmesh_tri())
    print("\n── CPU, one thread, float32 ──")
    for s in range(len(specs)):
        ref spec = specs[s]
        var dirs = lidar_directions(spec, 0)
        var n = spec.rays_per_frame()
        var r = List[Float64]()
        var g = List[Int]()
        lidar_scan_cpu[GT, DynDims, LANES](d, m, spec, mount, dirs, r, g, 0)
        var reps = max(1, 200000 // n)
        var t0 = perf_counter_ns()
        for k in range(reps):
            lidar_scan_cpu[GT, DynDims, LANES](d, m, spec, mount, dirs, r, g, k % LANES)
        var dt = Float64(perf_counter_ns() - t0) * 1e-9 / Float64(reps)
        print("  ", spec.name, ":", n, "beams/frame ", dt * 1e3, "ms/frame ",
              Float64(n) / dt / 1e6, "M beams/s  =", 1.0 / dt, "frames/s (spec rate",
              spec.rate_hz, "Hz)")

    comptime if has_accelerator():
        var ctx = DeviceContext()
        m.upload_all(ctx)
        d.upload_all(ctx)
        ctx.synchronize()
        print("\n── GPU,", LANES, "lanes per launch, float32 ──")
        for s in range(len(specs)):
            ref spec = specs[s]
            var lid = BatchedLidar[GT, DynDims, LANES](ctx, spec, mount)
            lid.scan(ctx, d, m, 0, 1)
            ctx.synchronize()
            var n = spec.rays_per_frame()
            var reps = 20
            var t0 = perf_counter_ns()
            for k in range(reps):
                lid.scan(ctx, d, m, 0, k)
            ctx.synchronize()
            var dt = Float64(perf_counter_ns() - t0) * 1e-9 / Float64(reps)
            print("  ", spec.name, ":", LANES, "x", n, "beams/launch ", dt * 1e3,
                  "ms/launch ", Float64(LANES * n) / dt / 1e6, "M beams/s  =",
                  Float64(LANES) / dt, "lane-frames/s")
    else:
        print("\n(no accelerator: GPU leg skipped)")
