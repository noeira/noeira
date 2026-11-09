"""The depth camera's error model, against answers known in closed form.

    pixi run mojo run -I . tests/perception/test_depth_errors.mojo

`test_depth_vs_mujoco` gates the GEOMETRY with every error off. This gates
each error on `assets/depth_bench.xml` (a wall at planar depth 3, a thin
pillar at 1, the floor), where every expected number is a formula:

  stereo noise   sd of (noisy - ideal) on the wall = subpixel/(fx b) * 3^2
  ToF noise      sd = sigma_a + sigma_b * 3
  dropout        the fraction of lost wall pixels = p
  quantisation   every depth a multiple of depth_unit
  grazing cut    a floor pixel has depth iff h / ray_length >= cos(max)
  stereo shadow  a band of fx * b * (1/1 - 1/3) pixels on the pillar's LEFT,
                 none on its right

and the ideal camera's spec file is parsed and refused like the lidar's.
"""

from std.math import abs, sqrt, cos, pi
from std.testing import assert_true, assert_raises, TestSuite

from noeira.physics3d.fields import DynDims
from noeira.perception import (
    DepthCameraSpec,
    LidarMount,
    DEPTH_STEREO,
    DEPTH_TOF,
    load_depth_spec,
    depth_spec_from_text,
    depth_directions,
    depth_to_points,
    depth_scan_cpu,
)
from mj_reference import DT, Built, quat_axis

comptime BENCH = "tests/perception/assets/depth_bench.xml"
comptime H_CAM = 1.0


def _scan(
    mut b: Built, spec: DepthCameraSpec, mount: LidarMount, seed: Int = 0
) raises -> Tuple[List[Float64], List[Int]]:
    var dirs = depth_directions(spec)
    var z = List[Float64]()
    var g = List[Int]()
    depth_scan_cpu[DT, DynDims, 1](b.d, b.m, spec, mount, dirs, z, g, 0, 0, seed)
    return (z^, g^)


def _level() -> LidarMount:
    return LidarMount.on_body(0, 0.0, 0.0, H_CAM)


def _bench_spec(model: Int) raises -> DepthCameraSpec:
    var s = load_depth_spec("noeira/perception/zoo/d435_class_848x480.depth")
    s.model = model
    s.depth_unit = 0.0
    s.dropout = 0.0
    s.max_incidence_deg = 90.0
    s.stereo_shadow = False
    s.range_max = 100.0
    return s^


def _wall_sd(z: List[Float64], ideal: List[Float64], gi: List[Int], wall: Int) raises -> Tuple[Float64, Int]:
    var n = 0
    var s1 = 0.0
    var s2 = 0.0
    for i in range(len(z)):
        if gi[i] != wall or z[i] <= 0.0:
            continue
        var e = z[i] - ideal[i]
        n += 1
        s1 += e
        s2 += e * e
    var mean = s1 / Float64(n)
    return (sqrt(s2 / Float64(n) - mean * mean), n)


def test_stereo_noise_grows_with_z_squared() raises:
    var b = Built(BENCH)
    var spec = _bench_spec(DEPTH_STEREO)
    var ideal = _scan(b, spec.ideal(), _level())
    var noisy = _scan(b, spec, _level(), 11)
    var wall = 1
    var r = _wall_sd(noisy[0], ideal[0], ideal[1], wall)
    var want = spec.subpixel / (spec.fx * spec.baseline) * 9.0
    print("   stereo: wall pixels", r[1], " sd", r[0], " expected", want)
    assert_true(r[1] > 100000, "too few wall pixels")
    assert_true(abs(r[0] - want) < 0.02 * want, "stereo sd " + String(r[0]) + " vs " + String(want))
    # The ideal wall really is at planar depth 3 everywhere — the premise.
    var worst = 0.0
    for i in range(len(ideal[0])):
        if ideal[1][i] == wall:
            worst = max(worst, abs(ideal[0][i] - 3.0))
    assert_true(worst < 1e-9, "the wall's planar depth is not constant: " + String(worst))


def test_tof_noise_is_affine() raises:
    var b = Built(BENCH)
    var spec = _bench_spec(DEPTH_TOF)
    spec.sigma_a = 0.004
    spec.sigma_b = 0.002
    var ideal = _scan(b, spec.ideal(), _level())
    var noisy = _scan(b, spec, _level(), 12)
    var r = _wall_sd(noisy[0], ideal[0], ideal[1], 1)
    var want = 0.004 + 0.002 * 3.0
    print("   tof: sd", r[0], " expected", want)
    assert_true(abs(r[0] - want) < 0.02 * want, "tof sd " + String(r[0]))


def test_dropout_and_quantisation() raises:
    var b = Built(BENCH)
    var spec = _bench_spec(DEPTH_STEREO)
    spec.dropout = 0.07
    spec.depth_unit = 0.001
    var ideal = _scan(b, spec.ideal(), _level())
    var z = _scan(b, spec, _level(), 13)[0].copy()
    var n = 0
    var lost = 0
    var off_grid = 0
    for i in range(len(z)):
        if ideal[1][i] != 1:
            continue
        n += 1
        if z[i] <= 0.0:
            lost += 1
            continue
        var k = z[i] / 0.001
        if abs(k - Float64(Int(k + 0.5))) > 1e-6:
            off_grid += 1
    var frac = Float64(lost) / Float64(n)
    print("   dropout", frac, " (p = 0.07)  off the 1 mm grid:", off_grid)
    assert_true(abs(frac - 0.07) < 0.003, "dropout " + String(frac))
    assert_true(off_grid == 0, String(off_grid) + " depths are not multiples of 1 mm")


def test_grazing_floor_is_cut_exactly() raises:
    """Pitched down 0.6 rad so the floor fills the lower image at every
    incidence: a floor pixel keeps its depth iff cos(incidence) = h / ray
    length >= cos(max). The ray length is z / d.x (planar depth over the
    pixel ray's forward component)."""
    var b = Built(BENCH)
    var spec = _bench_spec(DEPTH_STEREO)
    spec.subpixel = 0.0  # geometry only: noise would move z, not the cut
    spec.max_incidence_deg = 75.0
    var q = quat_axis(0, 1, 0, 0.6)
    var mount = LidarMount(0, 0.0, 0.0, H_CAM, q[0], q[1], q[2], q[3])
    var ideal = _scan(b, spec.ideal(), mount)
    var got = _scan(b, spec, mount)[0].copy()
    var dirs = depth_directions(spec)
    var cmax = cos(75.0 * pi / 180.0)
    var floor_px = 0
    var cut = 0
    var bad = 0
    var near_edge = 0
    for i in range(len(got)):
        if ideal[1][i] != 0:
            continue
        floor_px += 1
        var c = H_CAM * dirs[3 * i] / ideal[0][i]
        if abs(c - cmax) < 1e-9:
            near_edge += 1
            continue
        var keep = c >= cmax
        if not keep:
            cut += 1
        if keep != (got[i] > 0.0):
            bad += 1
    print("   grazing: floor pixels", floor_px, " cut", cut, " wrong", bad)
    assert_true(cut > 1000 and floor_px - cut > 1000, "the cut is not exercised on both sides")
    assert_true(bad == 0, String(bad) + " floor pixels on the wrong side of the incidence cut")


def test_stereo_shadow_band_left_of_the_pillar() raises:
    var b = Built(BENCH)
    var spec = _bench_spec(DEPTH_STEREO)
    spec.subpixel = 0.0
    spec.stereo_shadow = True
    var ideal = _scan(b, spec.ideal(), _level())
    var got = _scan(b, spec, _level())
    var W = spec.width
    var row = Int(spec.cy)  # the horizon row
    # The pillar's columns in that row, then shadowed wall pixels on each side.
    var first = -1
    var last = -1
    for u in range(W):
        if ideal[1][row * W + u] == 2:
            if first < 0:
                first = u
            last = u
    var left = 0
    var right = 0
    for u in range(W):
        var i = row * W + u
        if ideal[1][i] == 1 and got[0][i] <= 0.0:
            if u < first:
                left += 1
            elif u > last:
                right += 1
    var want = spec.fx * spec.baseline * (1.0 / 1.0 - 1.0 / 3.0)
    print("   shadow: pillar columns", first, "..", last, " band left", left, " right", right,
          " expected", want, "px on the left")
    assert_true(first > 0 and last < W - 1, "the pillar is not inside the image")
    assert_true(abs(Float64(left) - want) <= 1.5, "band " + String(left) + " px vs " + String(want))
    assert_true(right == 0, String(right) + " shadow pixels on the pillar's right")
    # Every shadowed wall pixel sits directly next to the pillar (no stray holes).
    for u in range(first - left, first):
        assert_true(got[0][row * W + u] <= 0.0, "the band is not contiguous at column " + String(u))


def test_unprojection_puts_the_wall_at_x_3() raises:
    var b = Built(BENCH)
    var spec = _bench_spec(DEPTH_STEREO).ideal()
    var r = _scan(b, spec, _level())
    var pts = depth_to_points(spec, r[0])
    var k = 0
    var worst = 0.0
    for i in range(len(r[0])):
        if r[0][i] <= 0.0:
            continue
        if r[1][i] == 1:
            worst = max(worst, abs(pts[3 * k] - 3.0))
        k += 1
    assert_true(k * 3 == len(pts), "one point per valid pixel")
    assert_true(worst < 1e-9, "a wall point is off x = 3 by " + String(worst))


def test_a_bad_depth_spec_is_refused() raises:
    with assert_raises(contains="unknown key"):
        _ = depth_spec_from_text("width = 4\nheight = 4\nfx = 1\nfy = 1\nbaselin = 0.05\n")
    with assert_raises(contains="stereo needs baseline"):
        _ = depth_spec_from_text("width = 4\nheight = 4\nfx = 1\nfy = 1\nmodel = stereo\n")
    with assert_raises(contains="stereo_shadow needs model = stereo"):
        _ = depth_spec_from_text("width = 4\nheight = 4\nfx = 1\nfy = 1\nmodel = tof\nbaseline = 0.05\nstereo_shadow = 1\n")
    with assert_raises(contains="model must be"):
        _ = depth_spec_from_text("width = 4\nheight = 4\nfx = 1\nfy = 1\nmodel = lidar\n")
    var names: List[String] = ["d435_class_848x480", "d435_class_424x240", "tof_class_640x576"]
    for nm in names:
        var s = load_depth_spec("noeira/perception/zoo/" + nm + ".depth")
        print("  ", s, " hfov", s.hfov_deg())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
