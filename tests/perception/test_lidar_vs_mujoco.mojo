"""The lidar scan against MuJoCo 3.12's own ray casting — the P0a gate.

    pixi run mojo run -I . tests/perception/test_lidar_vs_mujoco.mojo
    LIDAR_GATE_SCENE=<scene.xml> LIDAR_GATE_BODY=<body name> pixi run mojo run ...

Default scene: `tests/perception/assets/lidar_room.xml` (the public G1 in a
room with every geom type). The two environment variables re-run the same gate
on another scene and head-mount body, by NAME (e.g. the private demo room).

WHAT IS COMPARED, per scan, on EVERY ray of the frame (noise off, window open):
distance (|dr| < 1e-6 m + 1e-9 r), the geom hit, and hit/miss — against MuJoCo's
`mj_ray` from the same origin.

⚠⚠ `mj_multiRay` IS NOT `mj_ray`, AND IT IS ONLY THE FAST FIRST PASS HERE.
MuJoCo 3.12's multi-ray culls each body with a bounding sphere centred at
`xipos + bvh_aabb[:3]` — the AABB offset is NOT rotated by the body's
orientation (`mju_singleRay`, engine_ray.c:1476). On a rotated body the sphere
sits in the wrong place and a beam that hits the geom is culled: measured on the
G1's left rubber hand, 0.241 m from the head — `mj_ray`, `mj_rayMesh` and a
brute-force triangle test all hit it at 0.24123, `mj_multiRay` reported the
wall behind it at 2.07 m, and the sphere test passes once the offset is
rotated by `ximat`. The first version of this gate used `mj_multiRay` alone and
reported 316 320 "wrong geoms" of 1 984 000 beams that were MuJoCo's. So
every ray where ours and `mj_multiRay` disagree is re-cast with `mj_ray`, which
decides, and a stride of the agreements is re-cast too.

⚠⚠ THE REFERENCE COMPOSES THE MOUNT POSE ITSELF, IN NUMPY, FROM MuJoCo's
`xpos`/`xmat` — rotation MATRICES, where `lidar_cast` uses quaternions. A
reference that called our own pose code would share any mistake in it
(`_a_gate_that_shares_its_reference_implementation_is_blind`). The only shared
input is the sensor-frame direction list, whose own properties are gated in
`test_lidar_spec.mojo`.

⚠ THE GEOM ID IS ASSERTED, NOT PRINTED. Skip the wrong geom and a ray reports
whatever is behind it — a plausible distance. The scene carries an INVISIBLE
box in the beams' way so the visibility rule is actually exercised, and the
robot's own meshes (the mount body excluded, the arms not) so `bodyexclude` is.

RESULT (2026-11-09, MuJoCo 3.12, 120 poses x 3 specs x 3 mounts): 1 984 000
beams, 1 716 554 hits + 267 446 no-returns, 0 splits, 0 wrong geoms, worst
|dr| 1.1e-8 m (the heightfield's float32 grid), 309 405 beams on the robot's
own body. ~25 s on the laptop.

WHAT THIS GATE WAS PROVEN ABLE TO FAIL (each injected into `lidar_cast`):

    injected defect                         splits   wrong geom   worst |dr|
    -------------------------------------   ------   ----------   ----------
    no body exclusion (exclude = -1)         16 573      102 773   unchanged
    mount rotation dropped (q = bq)          34 743      250 420   3.54 m
    visual-only meshes not loaded (the         —          8 006   0.114 m
      loader before this gate; see below)

⚠ THE THIRD ROW WAS A REAL DEFECT, IN `physics3d/parser/fields_build`, NOT HERE.
A mesh used only by non-colliding geoms was never loaded: those geoms got mesh
id -1 (so `ray_model` skipped them — the G1's whole visual skin was invisible to
rays) and kept their declared frame (9 of them 1-8 cm off). `mj_ray` sees
visual geoms, so the loader now loads a VISIBLE mesh whenever the model carries
the triangle soup (`nmesh_tri > 0`); a model nobody rays pays nothing.

THREE MOUNTS, so each branch of the pose composition runs: a head mount on
`torso_link` (level), a TILTED mount on the pelvis (pitch + yaw offset), and a
fixed rig on the WORLD body (which must exclude nothing).
"""

from std.math import abs, cos, sin, pi
from std.python import Python, PythonObject
from std.os import getenv
from std.testing import assert_true, TestSuite

from noeira.physics3d.fields import Data, Model, DynDims, init_hfield_data
from noeira.physics3d.parser.full_parser import parse_xml_full
from noeira.physics3d.parser.runtime_load import (
    read_model_source,
    dims_from_flat,
    build_model_runtime,
    spec_fields_runtime,
)
from noeira.physics3d.kinematics.forward_kinematics import forward_kinematics
from noeira.perception import (
    LidarSpec,
    LidarMount,
    LIDAR_SPIRAL,
    load_lidar_spec,
    lidar_directions,
    lidar_scan_cpu,
)

comptime DT = DType.float64
comptime SCENE = "tests/perception/assets/lidar_room.xml"
comptime MOUNT_BODY = "robot_torso_link"
comptime NPOSE = 120
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


struct Built(Movable):
    var m: Model[DT, DynDims]
    var d: Data[DT, DynDims, 1]
    var dims: DynDims

    def __init__(out self, path: String) raises:
        var src = read_model_source(path)
        var fmd = parse_xml_full(src[0], src[1])
        var verts = 65536
        var tris = 1
        var dims = dims_from_flat(
            fmd, max_contacts=16, nmesh_verts=verts, nmesh_tri=tris
        )
        var m = Model[DT, DynDims](dims)
        var tries = 0
        while True:
            try:
                build_model_runtime[DT](fmd, dims, m)
                break
            except e:
                var msg = String(e)
                tries += 1
                if tries > 30:
                    raise e^
                var k = msg.find("Pass nmesh_tri=")
                if k >= 0:
                    var rest = String(msg[byte=k + 15 : msg.byte_length()])
                    var stop = 0
                    var bytes = rest.as_bytes()
                    while stop < len(bytes) and bytes[stop] >= 48 and bytes[stop] <= 57:
                        stop += 1
                    tris = atol(String(rest[byte=0:stop]))
                elif msg.find("mesh vertex capacity") >= 0:
                    verts *= 2
                else:
                    raise e^
                dims = dims_from_flat(
                    fmd, max_contacts=16, nmesh_verts=verts, nmesh_tri=tris
                )
                m = Model[DT, DynDims](dims)
        var d = Data[DT, DynDims, 1](dims)
        # ⚠ The heightfield grid lives in `Data` (per lane) and starts at
        # zero: without this the rug is flat and sits ON the floor plane.
        init_hfield_data(d, m)
        self.m = m^
        self.d = d^
        self.dims = dims


def _np_set(arr: PythonObject, vals: List[Float64]) raises:
    """Bulk-copy into a float64 numpy array through its buffer address — an
    element-wise `setitem` costs a Python call per float."""
    var addr = Int(py=arr.ctypes.data)
    var p = Pointer[Float64, MutAnyOrigin](unsafe_from_address=addr)
    for i in range(len(vals)):
        p.unsafe_offset(i)[] = vals[i]


def _np_f64(arr: PythonObject, i: Int) raises -> Float64:
    var p = Pointer[Float64, MutAnyOrigin](unsafe_from_address=Int(py=arr.ctypes.data))
    return p.unsafe_offset(i)[]


def _np_i32(arr: PythonObject, i: Int) raises -> Int:
    var p = Pointer[Int32, MutAnyOrigin](unsafe_from_address=Int(py=arr.ctypes.data))
    return Int(p.unsafe_offset(i)[])


def _ref_geom_dist(
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
        py=mujoco.mju_rayGeom(
            d.geom_xpos[g], d.geom_xmat[g], m.geom_size[g], pnt, vec, gt
        )
    )


def _quat_axis(ax: Float64, ay: Float64, az: Float64, ang: Float64) -> List[Float64]:
    var s = sin(0.5 * ang)
    return [cos(0.5 * ang), ax * s, ay * s, az * s]


def _qmul(a: List[Float64], b: List[Float64]) -> List[Float64]:
    return [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]


def _gate(scene: String, mount_body_name: String) raises:
    var mujoco = Python.import_module("mujoco")
    var np = Python.import_module("numpy")
    var mjm = mujoco.MjModel.from_xml_path(scene)
    var mjd = mujoco.MjData(mjm)

    var b = Built(scene)
    var nq = b.dims.get_nq()
    assert_true(
        Int(py=mjm.nq) == nq and Int(py=mjm.ngeom) == b.dims.get_ngeom()
        and Int(py=mjm.nbody) == b.dims.get_nbody(),
        "the parse and MuJoCo disagree on nq / ngeom / nbody",
    )
    var body_head = Int(py=mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, mount_body_name))
    assert_true(body_head > 0, "no body named " + mount_body_name)
    # The robot's root body: the body of the first (free) joint.
    var body_root = Int(py=mjm.jnt_bodyid[0])
    assert_true(Int(py=mjm.jnt_type[0]) == 0, "joint 0 is not a free joint")

    var specs = List[LidarSpec]()
    specs.append(load_lidar_spec("noeira/perception/zoo/planar_2d_12m.lidar"))
    specs.append(load_lidar_spec("noeira/perception/zoo/rings16_100m.lidar"))
    specs.append(load_lidar_spec("noeira/perception/zoo/spiral_mid360_class.lidar"))
    # Noise off and the window open: the gate is on the GEOMETRY. The noise
    # and the window are `lidar_return`'s, gated in `test_lidar_spec`.
    for i in range(len(specs)):
        specs[i].range_sigma = 0.0
        specs[i].dropout = 0.0
        specs[i].range_min = 0.0
        specs[i].range_max = 1.0e30  # a beam skimming an infinite plane lands 1000s of km out

    # Mount 1: level head mount. Mount 2: tilted on the root body (pitch
    # down 0.35 rad about y, then a 0.4 rad yaw). Mount 3: a world rig.
    var mounts = List[LidarMount]()
    mounts.append(LidarMount.on_body(body_head, 0.05, 0.0, 0.47))
    var qt = _qmul(_quat_axis(0, 0, 1, 0.4), _quat_axis(0, 1, 0, 0.35))
    mounts.append(LidarMount(body_root, 0.12, 0.03, 0.10, qt[0], qt[1], qt[2], qt[3]))
    mounts.append(LidarMount.on_body(0, -2.5, -2.0, 1.1))

    var rng = Lcg(0x11DA2)
    var rays = 0
    var hits = 0
    var misses = 0
    var split = 0
    var wrong = 0
    var worst = 0.0
    var mjray_checked = 0
    var self_hits = 0
    var fk_worst = 0.0
    var multi_missed = 0
    var ties = 0
    var over = 0
    var worst_desc = String("")
    var ghost_g = Int(py=mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "ghost"))
    var dirs_cache = List[List[Float64]]()
    for s in range(len(specs)):
        dirs_cache.append(lidar_directions(specs[s], 0))

    var qpos0 = mjm.qpos0
    for pose in range(NPOSE):
        # ── one random pose: root in the room, yaw, small tilt, joints ──────
        var q = List[Float64](length=nq, fill=0.0)
        for k in range(nq):
            q[k] = Float64(py=qpos0[k])
        q[0] = rng.sym(2.2)
        q[1] = rng.sym(1.8)
        q[2] = 0.79 + rng.sym(0.03)
        var qr = _qmul(
            _quat_axis(0, 0, 1, rng.sym(pi)),
            _qmul(_quat_axis(1, 0, 0, rng.sym(0.1)), _quat_axis(0, 1, 0, rng.sym(0.1))),
        )
        q[3] = qr[0]
        q[4] = qr[1]
        q[5] = qr[2]
        q[6] = qr[3]
        for k in range(7, nq):
            q[k] = rng.sym(0.6)
        for k in range(nq):
            b.d.qpos.data[k] = Scalar[DT](q[k])
            mjd.qpos[k] = q[k]
        forward_kinematics["cpu", DT, DynDims, 1](b.d, b.m)
        _ = mujoco.mj_kinematics(mjm, mjd)
        # The two forward kinematics must agree before rays can be compared.
        var nbd = b.dims.get_nbody()
        var fk_err = 0.0
        var fk_worst_b = -1
        for bb in range(nbd):
            for c in range(3):
                var e = abs(Float64(b.d.xpos.data[bb * 3 + c]) - Float64(py=mjd.xpos[bb][c]))
                if e > fk_err:
                    fk_err = e
                    fk_worst_b = bb
        fk_worst = max(fk_worst, fk_err)
        assert_true(
            fk_err < 1e-12,
            "pose " + String(pose) + ": forward kinematics differ by "
            + String(fk_err) + " at body " + String(fk_worst_b)
            + " — the rays below would be compared on two different scenes",
        )

        var si = pose % len(specs)
        var mount = mounts[(pose // len(specs)) % len(mounts)]
        var frame = pose if specs[si].pattern == LIDAR_SPIRAL else 0
        var dirs: List[Float64]
        if frame == 0:
            dirs = dirs_cache[si].copy()
        else:
            dirs = lidar_directions(specs[si], frame)
        var n = len(dirs) // 3

        var ranges = List[Float64]()
        var gids = List[Int]()
        lidar_scan_cpu[DT, DynDims, 1](
            b.d, b.m, specs[si], mount, dirs, ranges, gids
        )

        # ── the reference: numpy composes the pose from xpos / xmat ────────
        var sd = np.zeros(3 * n)
        _np_set(sd, dirs)
        sd = sd.reshape(n, 3)
        var mq = np.array(Python.list(mount.qw, mount.qx, mount.qy, mount.qz))
        var rm = np.zeros(9)
        _ = mujoco.mju_quat2Mat(rm, mq)
        rm = rm.reshape(3, 3)
        var mp = np.array(Python.list(mount.px, mount.py, mount.pz))
        var origin: PythonObject
        var rot: PythonObject
        var excl: Int
        if mount.body > 0:
            var xm = mjd.xmat[mount.body].reshape(3, 3)
            origin = mjd.xpos[mount.body] + xm.dot(mp)
            rot = xm.dot(rm)
            excl = mount.body
        else:
            origin = mp
            rot = rm
            excl = -1
        var wd = np.ascontiguousarray(sd.dot(rot.T)).reshape(-1)
        var mgid = np.zeros(n, np.int32)
        var mdist = np.zeros(n)
        _ = mujoco.mj_multiRay(
            mjm, mjd, origin, wd, None, 1, excl, mgid, mdist, None, n, 1.0e9
        )

        for i in range(n):
            rays += 1
            var tr = _np_f64(mdist, i)
            var gref = _np_i32(mgid, i)
            var to = ranges[i]
            var agree = (tr >= 0.0) == (to >= 0.0) and (
                tr < 0.0 or (gids[i] == gref and abs(to - tr) < 1e-6)
            )
            var spot = i % MJ_RAY_STRIDE == 0
            if not agree or spot:
                # ⚠ `mj_ray` IS THE ARBITER. `mj_multiRay` was measured to
                # MISS geoms `mj_ray` hits (module docstring), so it is only
                # the fast first pass: every disagreement, and a stride of the
                # agreements, is re-cast one ray at a time.
                var g1 = np.zeros(1, np.int32)
                var t1 = Float64(
                    py=mujoco.mj_ray(
                        mjm, mjd, origin, wd[3 * i : 3 * i + 3], None, 1, excl, g1, None
                    )
                )
                var g1i = Int(py=g1[0])
                if spot:
                    mjray_checked += 1
                if abs(t1 - tr) > 1e-9 or g1i != gref:
                    multi_missed += 1
                tr = t1
                gref = g1i
            if (tr >= 0.0) != (to >= 0.0):
                split += 1
                if split <= 5:
                    var gn = String("-")
                    if gref >= 0:
                        gn = String(mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_GEOM, gref))
                    print("    SPLIT pose", pose, "spec", si, "mount", mount.body, "ray", i,
                          " ours", gids[i], to, " ref", gref, gn, tr)
                continue
            if tr < 0.0:
                misses += 1
                continue
            hits += 1
            if gids[i] != gref:
                # ⚠ A TIE IS NOT A WRONG GEOM. The G1 carries most meshes
                # TWICE on one body (a visual geom and a collision geom, same
                # mesh, same pose), so a beam meets two coincident surfaces and
                # which one is "nearest" is decided by the last bit. It counts
                # as a tie only if MuJoCo's own ray against OUR geom lands at
                # the same distance.
                if gids[i] >= 0 and abs(
                    _ref_geom_dist(mujoco, mjm, mjd, gids[i], origin, wd[3 * i : 3 * i + 3])
                    - tr
                ) < 1e-9:
                    ties += 1
                    if ties <= 3:
                        print("    TIE pose", pose, "ours", gids[i], String(mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_GEOM, gids[i])),
                              "ref", gref, String(mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_GEOM, gref)), "r", tr)
                    worst = max(worst, abs(to - tr))
                    continue
                wrong += 1
                if wrong <= 12:
                    var nm_ref = String(mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_GEOM, gref))
                    print("    WRONG pose", pose, "spec", si, "mount", mount.body, "ray", i,
                          " ours", gids[i], to, " ref", gref, nm_ref, tr)
                continue
            if abs(to - tr) > 1e-6 + 1e-9 * tr:
                over += 1
            if abs(to - tr) > worst:
                worst = abs(to - tr)
                worst_desc = (
                    "pose " + String(pose) + " spec " + String(si) + " ray " + String(i)
                    + " geom " + String(mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_GEOM, gref))
                    + " r " + String(tr)
                )
            if mount.body > 0 and Int(
                py=mjm.body_rootid[Int(py=mjm.geom_bodyid[gref])]
            ) == body_root:
                # A hit on the robot's own tree: the robot seeing itself.
                self_hits += 1
            assert_true(gref != ghost_g, "a beam stopped on the invisible box")

    print("  scene", scene, " mount body", mount_body_name, "(", body_head, ")")
    print("  poses", NPOSE, " rays", rays, " hits", hits, " no-returns", misses)
    print("  splits", split, " wrong geom", wrong, " coincident-surface ties", ties,
          " worst |dr|", worst)
    print("  worst beam:", worst_desc)
    print("  FK worst |dxpos|", fk_worst, " robot self-hits", self_hits)
    print("  mj_ray stride checks", mjray_checked,
          " rays where mj_multiRay != mj_ray (MuJoCo's cull):", multi_missed)
    assert_true(hits > rays // 2, "only " + String(hits) + " hits of " + String(rays))
    assert_true(misses > rays // 50, "only " + String(misses) + " no-returns: the miss branch is barely compared")
    assert_true(self_hits > 0, "no beam ever met the robot: bodyexclude is untested")
    assert_true(mjray_checked > 1000, "too few mj_ray stride checks")
    assert_true(split == 0, String(split) + " hit/miss disagreements")
    assert_true(wrong == 0, String(wrong) + " beams named a different geom")
    # ⚠ 1e-6 m PLUS 1e-9 OF THE RANGE. At room scale that is the plan's
    # 1e-6 m; the relative term is only for a beam that skims an INFINITE
    # floor plane and lands hundreds of km out (the demo room has no ceiling):
    # measured 1.4e-4 m at r = 840 km, a relative 1.6e-10 — float64 rounding.
    assert_true(over == 0, String(over) + " beams beyond 1e-6 + 1e-9 r; worst |dr| " + String(worst))


def test_lidar_scan_vs_mujoco() raises:
    var scene = getenv("LIDAR_GATE_SCENE", SCENE)
    var body = getenv("LIDAR_GATE_BODY", MOUNT_BODY)
    _gate(scene, body)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
