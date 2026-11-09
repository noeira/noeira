"""`BatchedDepthCamera` (GPU, one thread per lane x pixel) against
`depth_scan_cpu`, with EVERY error on (stereo noise, shadow, grazing cut,
dropout, 1 mm quantisation) and off.

    pixi run -e apple mojo run -I . tests/perception/test_depth_gpu_vs_cpu.mojo

Same contract as the lidar's GPU gate: one shared `depth_cast` /
`depth_return`, float32 on both legs (Metal has no double), eight lanes in
eight poses. The noise is counter-based, so the device must draw the host's
noise pixel for pixel.

TOLERANCES, both arithmetic and both stated:
* depth: `sqrt(eps32) * z` (a quadric root near a silhouette keeps half of
  float32's digits — see the lidar GPU gate) PLUS one quantisation step, since
  a float32 difference can round to the neighbouring millimetre;
* valid/invalid flips: at most 1 in 10 000 pixels — a float32 difference can
  move a pixel across the window edge, the incidence cut or a shadow edge.
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
    DepthCameraSpec,
    LidarMount,
    load_depth_spec,
    depth_directions,
    depth_scan_cpu,
    depth_mount_from_mjcf_camera,
    BatchedDepthCamera,
)
from mj_reference import Lcg, mesh_budget_dims

comptime GT = DType.float32
comptime B = 8
comptime SCENE = "tests/perception/assets/lidar_room.xml"
comptime HEAD_CAMERA = 1  # `robot_head_camera`: the G1's `track` camera is 0
comptime PELVIS = 1


def test_depth_gpu_matches_cpu() raises:
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
    var rng = Lcg(0xD3E)
    for env in range(B):
        var o = env * nq
        for k in range(nq):
            d.qpos.data[o + k] = 0
        d.qpos.data[o + 0] = Scalar[GT](rng.sym(2.0))
        d.qpos.data[o + 1] = Scalar[GT](rng.sym(1.6))
        d.qpos.data[o + 2] = Scalar[GT](0.79)
        var yaw = rng.sym(3.0)
        d.qpos.data[o + 3] = Scalar[GT](cos(0.5 * yaw))
        d.qpos.data[o + 6] = Scalar[GT](sin(0.5 * yaw))
        for k in range(7, nq):
            d.qpos.data[o + k] = Scalar[GT](rng.sym(0.6))
    forward_kinematics["cpu", GT, DynDims, B](d, m)
    m.upload_all(ctx)
    d.upload_all(ctx)
    ctx.synchronize()

    var specs = List[DepthCameraSpec]()
    specs.append(load_depth_spec("noeira/perception/zoo/d435_class_424x240.depth"))
    specs.append(load_depth_spec("noeira/perception/zoo/tof_class_640x576.depth"))
    specs.append(load_depth_spec("noeira/perception/zoo/d435_class_848x480.depth").ideal())
    var mounts = List[LidarMount]()
    mounts.append(depth_mount_from_mjcf_camera(m, HEAD_CAMERA))
    mounts.append(LidarMount(PELVIS, 0.10, 0.0, 0.05, 0.96, 0.0, 0.24, 0.15))

    for s in range(len(specs)):
        for mi in range(len(mounts)):
            ref spec = specs[s]
            var mount = mounts[mi]
            var cam = BatchedDepthCamera[GT, DynDims, B](ctx, spec, mount)
            var frame = 5
            var seed = 31
            cam.scan(ctx, d, m, frame, seed)
            var gz = List[Float64]()
            var gg = List[Int]()
            cam.download(ctx, gz, gg)
            var dirs = depth_directions(spec)
            var n = spec.npix()
            var flips = 0
            var wrong = 0
            var over = 0
            var valid = 0
            var worst = 0.0
            for env in range(B):
                var z = List[Float64]()
                var g = List[Int]()
                depth_scan_cpu[GT, DynDims, B](d, m, spec, mount, dirs, z, g, env, frame, seed)
                for i in range(n):
                    var k = env * n + i
                    if (z[i] > 0.0) != (gz[k] > 0.0):
                        flips += 1
                        continue
                    if z[i] <= 0.0:
                        continue
                    valid += 1
                    if g[i] != gg[k]:
                        wrong += 1
                        continue
                    var e = abs(z[i] - gz[k])
                    worst = max(worst, e)
                    if e > 3.45e-4 * max(1.0, z[i]) + spec.depth_unit:
                        over += 1
            var total = B * n
            print("  ", spec.name, "mount", mi, ": pixels", total, " valid", valid, " flips", flips,
                  " wrong geom", wrong, " worst |dz|", worst, " over bound", over)
            assert_true(valid > total // 4, spec.name + ": too few valid pixels")
            assert_true(flips + wrong <= total // 10000, spec.name + ": " + String(flips)
                        + " valid/invalid flips, " + String(wrong) + " wrong geoms")
            assert_true(over == 0, spec.name + ": " + String(over) + " pixels beyond the bound")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
