"""The MuJoCo reference every perception gate compares against — ONE copy.

Imported by the gates beside it (`from mj_reference import ...`). It holds the
judging rule, so a lidar beam and a depth pixel are judged the same way:

* `mj_multiRay` is a FAST FIRST PASS ONLY. MuJoCo 3.12's multi-ray culls a body
  by a bounding sphere at `xipos + bvh_aabb[:3]` with the offset NOT rotated by
  the body's orientation (`mju_singleRay`, engine_ray.c:1476), and misses geoms
  `mj_ray` hits (measured: the G1's rubber hand at 0.24 m). Every ray where
  ours and multi-ray disagree, and a stride of the rest, is re-cast with
  `mj_ray`, which decides.
* A different geom at the same distance is a TIE, not an error, only if
  MuJoCo's own ray against OUR geom lands at the reference distance
  (coincident surfaces: the demo room's door frame boxes overlap).
* Distance bound `1e-6 m + 1e-9 r`: the plan's 1e-6 m at room scale; the
  relative term is for a ray skimming an infinite floor plane hundreds of km
  out (float64 rounding, measured 1.6e-10 relative).

The distances compared are `mj_ray`'s `t` in units of `|vec|` — the caller
chooses `vec` so that `t` is the quantity it reports (unit vectors for a
lidar's range; a vector whose forward component is 1 for a camera's planar
depth).
"""

from std.math import abs, cos, sin
from std.python import Python, PythonObject
from std.testing import assert_true

from noeira.physics3d.fields import Data, Model, DynDims, init_hfield_data
from noeira.physics3d.parser.full_parser import parse_xml_full
from noeira.physics3d.kinematics.forward_kinematics import forward_kinematics
from noeira.perception import ray_ready_dims
from noeira.physics3d.parser.runtime_load import (
    read_model_source,
    dims_from_flat,
    build_model_runtime,
)

comptime DT = DType.float64
comptime MJ_RAY_STRIDE = 41


struct Lcg(Copyable, Movable):
    var s: UInt64

    def __init__(out self, seed: UInt64):
        self.s = seed

    def u01(mut self) -> Float64:
        self.s = self.s * 6364136223846793005 + 1442695040888963407
        return Float64((self.s >> 11) & 0xFFFFFFFFFFFFF) / Float64(1 << 52)

    def sym(mut self, a: Float64) -> Float64:
        return (self.u01() * 2.0 - 1.0) * a


def mesh_budget_dims(path: String) raises -> DynDims:
    """The mesh budgets a scene needs — `perception.ray_ready_dims`."""
    return ray_ready_dims(path)


struct Built(Movable):
    """A float64 runtime model + one-lane data of a scene, heightfields set."""

    var m: Model[DT, DynDims]
    var d: Data[DT, DynDims, 1]
    var dims: DynDims

    def __init__(out self, path: String) raises:
        var dims = mesh_budget_dims(path)
        var src = read_model_source(path)
        var fmd = parse_xml_full(src[0], src[1])
        var m = Model[DT, DynDims](dims)
        build_model_runtime[DT](fmd, dims, m)
        var d = Data[DT, DynDims, 1](dims)
        # ⚠ The heightfield grid lives in `Data` (per lane) and starts at
        # zero: without this a heightfield is flat and sits on its base.
        init_hfield_data(d, m)
        self.m = m^
        self.d = d^
        self.dims = dims


def np_set(arr: PythonObject, vals: List[Float64]) raises:
    """Bulk-copy into a float64 numpy array through its buffer address — an
    element-wise `setitem` costs a Python call per float."""
    var p = Pointer[Float64, MutAnyOrigin](unsafe_from_address=Int(py=arr.ctypes.data))
    for i in range(len(vals)):
        p.unsafe_offset(i)[] = vals[i]


def np_f64(arr: PythonObject, i: Int) raises -> Float64:
    var p = Pointer[Float64, MutAnyOrigin](unsafe_from_address=Int(py=arr.ctypes.data))
    return p.unsafe_offset(i)[]


def np_i32(arr: PythonObject, i: Int) raises -> Int:
    var p = Pointer[Int32, MutAnyOrigin](unsafe_from_address=Int(py=arr.ctypes.data))
    return Int(p.unsafe_offset(i)[])


def quat_axis(ax: Float64, ay: Float64, az: Float64, ang: Float64) -> List[Float64]:
    var s = sin(0.5 * ang)
    return [cos(0.5 * ang), ax * s, ay * s, az * s]


def qmul(a: List[Float64], b: List[Float64]) -> List[Float64]:
    return [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]


def ref_geom_dist(
    mujoco: PythonObject, m: PythonObject, d: PythonObject, g: Int,
    pnt: PythonObject, vec: PythonObject,
) raises -> Float64:
    """MuJoCo's distance along the ray to ONE geom, ignoring every other."""
    var gt = Int(py=m.geom_type[g])
    if gt == 7:  # mjGEOM_MESH
        return Float64(py=mujoco.mj_rayMesh(m, d, g, pnt, vec))
    if gt == 1:  # mjGEOM_HFIELD
        return Float64(py=mujoco.mj_rayHfield(m, d, g, pnt, vec))
    return Float64(
        py=mujoco.mju_rayGeom(d.geom_xpos[g], d.geom_xmat[g], m.geom_size[g], pnt, vec, gt)
    )


def set_pose_and_check_fk(
    mut b: Built, mjm: PythonObject, mjd: PythonObject, q: List[Float64]
) raises -> Float64:
    """Same qpos on both sides, forward kinematics on both, and the largest
    body-position difference (the caller asserts it): rays can only be
    compared on one scene."""
    var mujoco = Python.import_module("mujoco")
    for k in range(len(q)):
        b.d.qpos.data[k] = q[k]
        mjd.qpos[k] = q[k]
    forward_kinematics["cpu", DT, DynDims, 1](b.d, b.m)
    _ = mujoco.mj_kinematics(mjm, mjd)
    # ⚠ `mj_kinematics` does not place cameras or lights: `cam_xpos` /
    # `cam_xmat` stay at whatever they were (zeros on a fresh MjData) until
    # `mj_camlight` runs. A camera reference read without it is a zero matrix.
    _ = mujoco.mj_camlight(mjm, mjd)
    var err = 0.0
    for bb in range(b.dims.get_nbody()):
        for c in range(3):
            err = max(err, abs(Float64(b.d.xpos.data[bb * 3 + c]) - Float64(py=mjd.xpos[bb][c])))
    return err


struct RayStats(Movable):
    var rays: Int
    var hits: Int
    var misses: Int
    var split: Int
    var wrong: Int
    var ties: Int
    var over: Int
    var multi_missed: Int
    var mjray_checked: Int
    var self_hits: Int
    var worst: Float64
    var worst_desc: String

    def __init__(out self):
        self.rays = 0
        self.hits = 0
        self.misses = 0
        self.split = 0
        self.wrong = 0
        self.ties = 0
        self.over = 0
        self.multi_missed = 0
        self.mjray_checked = 0
        self.self_hits = 0
        self.worst = 0.0
        self.worst_desc = String("")

    def add_scan(
        mut self,
        mujoco: PythonObject,
        np: PythonObject,
        mjm: PythonObject,
        mjd: PythonObject,
        origin: PythonObject,
        vecs: PythonObject,
        excl: Int,
        ours_t: List[Float64],
        ours_g: List[Int],
        robot_root: Int,
        ghost: Int,
        tag: String,
    ) raises:
        """`vecs` is a contiguous float64 `[n*3]`; `ours_t[i]` < 0 = no hit."""
        var n = len(ours_t)
        var mgid = np.zeros(n, np.int32)
        var mdist = np.zeros(n)
        _ = mujoco.mj_multiRay(
            mjm, mjd, origin, vecs, None, 1, excl, mgid, mdist, None, n, 1.0e30
        )
        # One address each, not a Python call per element.
        var pd = Pointer[Float64, MutAnyOrigin](unsafe_from_address=Int(py=mdist.ctypes.data))
        var pg = Pointer[Int32, MutAnyOrigin](unsafe_from_address=Int(py=mgid.ctypes.data))
        for i in range(n):
            self.rays += 1
            var tr = pd.unsafe_offset(i)[]
            var gref = Int(pg.unsafe_offset(i)[])
            var to = ours_t[i]
            var agree = (tr >= 0.0) == (to >= 0.0) and (
                tr < 0.0 or (ours_g[i] == gref and abs(to - tr) < 1e-6 + 1e-9 * tr)
            )
            var spot = i % MJ_RAY_STRIDE == 0
            if not agree or spot:
                var g1 = np.zeros(1, np.int32)
                var t1 = Float64(
                    py=mujoco.mj_ray(mjm, mjd, origin, vecs[3 * i : 3 * i + 3], None, 1, excl, g1, None)
                )
                var g1i = Int(py=g1[0])
                if spot:
                    self.mjray_checked += 1
                if abs(t1 - tr) > 1e-9 or g1i != gref:
                    self.multi_missed += 1
                tr = t1
                gref = g1i
            if (tr >= 0.0) != (to >= 0.0):
                self.split += 1
                if self.split <= 5:
                    var gn = String("-")
                    if gref >= 0:
                        gn = String(mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_GEOM, gref))
                    print("    SPLIT", tag, "i", i, " ours", ours_g[i], to, " ref", gref, gn, tr)
                continue
            if tr < 0.0:
                self.misses += 1
                continue
            self.hits += 1
            if ours_g[i] != gref:
                if ours_g[i] >= 0 and abs(
                    ref_geom_dist(mujoco, mjm, mjd, ours_g[i], origin, vecs[3 * i : 3 * i + 3]) - tr
                ) < 1e-9:
                    self.ties += 1
                    continue
                self.wrong += 1
                if self.wrong <= 8:
                    var gn = String(mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_GEOM, gref))
                    print("    WRONG", tag, "i", i, " ours", ours_g[i], to, " ref", gref, gn, tr)
                continue
            var e = abs(to - tr)
            if e > 1e-6 + 1e-9 * tr:
                self.over += 1
            if e > self.worst:
                self.worst = e
                self.worst_desc = (
                    tag + " i " + String(i) + " geom "
                    + String(mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_GEOM, gref)) + " r " + String(tr)
                )
            if excl > 0 and Int(py=mjm.body_rootid[Int(py=mjm.geom_bodyid[gref])]) == robot_root:
                self.self_hits += 1
            assert_true(gref != ghost, "a ray stopped on the invisible box")

    def report(self):
        print("  rays", self.rays, " hits", self.hits, " no-hits", self.misses)
        print("  splits", self.split, " wrong geom", self.wrong, " coincident-surface ties",
              self.ties, " worst |d|", self.worst)
        print("  worst:", self.worst_desc)
        print("  robot self-hits", self.self_hits, " mj_ray stride checks", self.mjray_checked,
              " rays where mj_multiRay != mj_ray:", self.multi_missed)

    def assert_clean(self) raises:
        assert_true(self.hits > self.rays // 2, "only " + String(self.hits) + " hits of " + String(self.rays))
        assert_true(self.misses > self.rays // 50, "only " + String(self.misses) + " no-hits: that branch is barely compared")
        assert_true(self.self_hits > 0, "no ray ever met the robot: bodyexclude is untested")
        assert_true(self.mjray_checked > 1000, "too few mj_ray stride checks")
        assert_true(self.split == 0, String(self.split) + " hit/miss disagreements")
        assert_true(self.wrong == 0, String(self.wrong) + " rays named a different geom")
        assert_true(self.over == 0, String(self.over) + " rays beyond 1e-6 + 1e-9 r; worst " + String(self.worst))
