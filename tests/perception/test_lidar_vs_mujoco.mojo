"""The lidar scan against MuJoCo 3.12's own ray casting — the P0a gate.

    pixi run mojo run -I . tests/perception/test_lidar_vs_mujoco.mojo
    LIDAR_GATE_SCENE=<scene.xml> LIDAR_GATE_BODY=<body name> pixi run mojo run ...

Default scene: `tests/perception/assets/lidar_room.xml` (the public G1 in a
room with every geom type). The two environment variables re-run the same gate
on another scene and head-mount body, by NAME (e.g. the private demo room).

WHAT IS COMPARED, per scan, on EVERY ray of the frame (noise off, window open):
distance (|dr| < 1e-6 m + 1e-9 r), the geom hit, and hit/miss — against MuJoCo's
`mj_ray` from the same origin.

⚠⚠ `mj_multiRay` IS NOT `mj_ray` and is only the fast first pass; the judging
rule (multi-ray, then `mj_ray` as the arbiter, ties, the distance bound) lives
in `mj_reference.mojo`, shared with the depth-camera gate. The first version of
this gate used `mj_multiRay` alone and reported 316 320 "wrong geoms" of
1 984 000 beams that were MuJoCo's own cull.

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

from std.math import pi
from std.python import Python, PythonObject
from std.os import getenv
from std.testing import assert_true, TestSuite

from noeira.perception import (
    LidarSpec,
    LidarMount,
    LIDAR_SPIRAL,
    load_lidar_spec,
    lidar_directions,
    lidar_scan_cpu,
)
from mj_reference import (
    DT, Lcg, Built, RayStats, np_set, quat_axis, qmul, set_pose_and_check_fk,
)
from noeira.physics3d.fields import DynDims

comptime SCENE = "tests/perception/assets/lidar_room.xml"
comptime MOUNT_BODY = "robot_torso_link"
comptime NPOSE = 120


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
    var ghost = Int(py=mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "ghost"))

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
    var qt = qmul(quat_axis(0, 0, 1, 0.4), quat_axis(0, 1, 0, 0.35))
    mounts.append(LidarMount(body_root, 0.12, 0.03, 0.10, qt[0], qt[1], qt[2], qt[3]))
    mounts.append(LidarMount.on_body(0, -2.5, -2.0, 1.1))

    var rng = Lcg(0x11DA2)
    var st = RayStats()
    var fk_worst = 0.0
    var qpos0 = mjm.qpos0
    for pose in range(NPOSE):
        # ── one random pose: root in the room, yaw, small tilt, joints ──────
        var q = List[Float64](length=nq, fill=0.0)
        for k in range(nq):
            q[k] = Float64(py=qpos0[k])
        q[0] = rng.sym(2.2)
        q[1] = rng.sym(1.8)
        q[2] = 0.79 + rng.sym(0.03)
        var qr = qmul(
            quat_axis(0, 0, 1, rng.sym(pi)),
            qmul(quat_axis(1, 0, 0, rng.sym(0.1)), quat_axis(0, 1, 0, rng.sym(0.1))),
        )
        for k in range(4):
            q[3 + k] = qr[k]
        for k in range(7, nq):
            q[k] = rng.sym(0.6)
        var fk_err = set_pose_and_check_fk(b, mjm, mjd, q)
        fk_worst = max(fk_worst, fk_err)
        assert_true(fk_err < 1e-12, "pose " + String(pose) + ": forward kinematics differ by "
                    + String(fk_err) + " — rays would be compared on two scenes")

        var si = pose % len(specs)
        var mount = mounts[(pose // len(specs)) % len(mounts)]
        var frame = pose if specs[si].pattern == LIDAR_SPIRAL else 0
        var dirs = lidar_directions(specs[si], frame)
        var n = len(dirs) // 3
        var ranges = List[Float64]()
        var gids = List[Int]()
        lidar_scan_cpu[DT, DynDims, 1](b.d, b.m, specs[si], mount, dirs, ranges, gids)

        # ── the reference: numpy composes the pose from xpos / xmat ────────
        var sd = np.zeros(3 * n)
        np_set(sd, dirs)
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
        st.add_scan(
            mujoco, np, mjm, mjd, origin, wd, excl, ranges, gids, body_root, ghost,
            "pose " + String(pose) + " spec " + String(si) + " mount " + String(mount.body),
        )

    print("  scene", scene, " mount body", mount_body_name, "(", body_head, ")  poses", NPOSE,
          " FK worst |dxpos|", fk_worst)
    st.report()
    st.assert_clean()


def test_lidar_scan_vs_mujoco() raises:
    var scene = getenv("LIDAR_GATE_SCENE", SCENE)
    var body = getenv("LIDAR_GATE_BODY", MOUNT_BODY)
    _gate(scene, body)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
