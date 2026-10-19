"""OSC_POSE on the composed `libero_goal` scene — L4's unit gate.

    pixi run mojo run -I . tests/libero/test_osc_pose.mojo

The DECISIVE gate is `tools/libero/libero_demo_replay.py` (one recorded
LIBERO demo, our engine + `osc_pose.mojo` against MuJoCo 3.12 running the
same transcription on the same model: 1.5e-5 m over 80 policy steps, and
the same 1.6 cm to the 2022 recording that MuJoCo 3.12 itself shows). It
needs a 426 MB demo file and so is a tool, not a smoke test. This file is
what CI can run without it:

1. the small dense algebra: an inverse times its matrix is the identity,
   `orientation_error` is zero at identity and reads a small rotation
   about z as `(0, 0, theta)`, `axisangle_to_mat` is orthonormal;
2. the gripper ramp: `speed` 0.01 per SUBSTEP — 100 calls (4 policy steps)
   from neutral to an end, 200 (8) end to end — `ctrl = bias + weight *
   current` on both fingers;
3. HOLD: a zero action for 20 policy steps keeps the grip site within
   `HOLD_TOL` of where it started — the controller cancels gravity
   (`qfrc_bias`) and the nullspace term holds the reset joints;
4. TRACK: `+1` on x for 10 policy steps (a 5 cm goal each step) moves the
   grip site in +x by at least `TRACK_MIN`, and `-1` for 10 moves it back
   in -x by as much (the goal is relative to the CURRENT pose each step,
   so the two legs need not land on the same point).

⚠ EVERY CHECK BELOW HAS A NEGATIVE READING. A controller that output zero
torque would fail 3 (the arm falls) and 4 (nothing moves); one that ignored
the goal would pass 3 and fail 4.
"""

from std.os.path import exists
from std.math import sqrt, sin, cos

from noeira.physics3d.fields import Data, Model, DynDims, DynamicsScratch, SpecFields
from noeira.physics3d.parser.runtime_load import (
    parse_model_runtime, dims_from_flat, build_model_runtime,
    spec_fields_runtime,
)
from noeira.physics3d.kinematics.forward_kinematics import forward_kinematics
from noeira.physics3d.dynamics.actuation import apply_actions_fields
from noeira.physics3d.dynamics.osc_pose import (
    OscPose, OscPoseConfig, PandaGripperRamp, ARM_DOF, OSC_ACTION_DIM,
)
from noeira.physics3d.dynamics.osc_pose_gpu import (
    OSC_WORK_WORDS, OSC_W_TMP, OSC_W_TMP2, OSC_W_LAMP,
    _inv_into, _mul_into, axisangle_mat_into,
)
from noeira.nn.core.tensor import TensorImpl
from noeira.physics3d.fields import DYN2, rl2
from noeira.physics3d.studio.stepping import StudioIntegEll
from noeira.tasks.spec import load_family
from noeira.tasks.family import scene_path
from noeira.envs.libero.models.libero_goal_xml import LIBERO_GOAL_MAX_CONTACTS


comptime DT = DType.float64
comptime FAMILY = "noeira/envs/libero/families/libero_goal.family"
comptime PACK = "noeira/envs/libero/assets"
comptime SUBSTEPS = 25
comptime HOLD_STEPS = 20
comptime HOLD_TOL = 0.003      # metres, 1 s of hold
comptime TRACK_STEPS = 10
comptime TRACK_MIN = 0.03      # metres of +x after 10 steps of a 5 cm goal


struct Tally(Copyable, ImplicitlyCopyable, Movable):
    var checks: Int
    var failures: Int

    def __init__(out self):
        self.checks = 0
        self.failures = 0

    def check(mut self, ok: Bool, what: String):
        self.checks += 1
        if ok:
            print("  ok:", what)
        else:
            self.failures += 1
            print("  FAIL:", what)


def _index(names: List[String], want: String) raises -> Int:
    for i in range(len(names)):
        if String(names[i]) == want:
            return i
    raise Error("no '" + want + "' in the composed scene")


def _dist(a: List[Float64], b: List[Float64]) -> Float64:
    var s = 0.0
    for i in range(3):
        s += (a[i] - b[i]) * (a[i] - b[i])
    return sqrt(s)


def policy_step(
    mut osc: OscPose, mut d: Data[DT, DynDims, 1],
    mut m: Model[DT, DynDims], mut scratch: DynamicsScratch[DT, DynDims, 1],
    mut integ: StudioIntegEll, mut act: List[Scalar[DT]],
    sf: SpecFields[DT, DynDims],
    action: List[Float64], nv: Int, timestep: Float64,
) raises:
    """One policy step: 25 substeps of `Robot.control` + `mj_step`."""
    for s in range(SUBSTEPS):
        osc.update(d, m, scratch)
        if s == 0:
            osc.set_goal(action, d, m)
        var ctrl = osc.run(action, d, m, scratch)
        for i in range(nv):
            d.qfrc.data[i] = Scalar[DT](0)
        apply_actions_fields[DT](sf, d, ctrl, act, timestep)
        integ.step["cpu"](d, m)


def main() raises:
    print("=== OSC_POSE — L4 unit gate ===")
    var ta = Tally()

    # ── 1. algebra, THROUGH THE DEVICE HELPERS ───────────────────────────
    # ⚠ THESE ARE THE KERNEL'S OWN FUNCTIONS, on a one-lane work tensor —
    # the same call shape `OscPose` uses. There is no host spelling of a
    # matrix inverse left to test.
    print("--- algebra (osc_pose_gpu's own helpers) ---")
    var wt = TensorImpl[DT].alloc(OSC_WORK_WORDS)
    for i in range(OSC_WORK_WORDS):
        wt.data[i] = Scalar[DT](0)
    var wv = wt.tt_dyn["cpu", DYN2](rl2(1, OSC_WORK_WORDS))
    var n = 6
    for i in range(n):
        for j in range(n):
            wt.data[OSC_W_TMP + i * n + j] = Scalar[DT](
                (4.0 if i == j else 0.0)
                + 0.3 * Float64((i * 7 + j * 3) % 5)
                + 0.3 * Float64((j * 7 + i * 3) % 5)
            )
    var ok_inv = _inv_into[DT](wv, 0, OSC_W_TMP, OSC_W_TMP2, n)
    ta.check(ok_inv, "a well-conditioned 6x6 inverts")
    _mul_into[DT](wv, 0, OSC_W_TMP, OSC_W_TMP2, OSC_W_LAMP, n, n, n)
    var worst = 0.0
    for i in range(n):
        for j in range(n):
            var e = Float64(wt.data[OSC_W_LAMP + i * n + j]) - (1.0 if i == j else 0.0)
            if e < 0.0:
                e = -e
            if e > worst:
                worst = e
    ta.check(worst < 1e-12, "A @ inv(A) == I to " + String(worst))
    for i in range(9):
        wt.data[OSC_W_TMP + i] = Scalar[DT](1.0)
    ta.check(
        not _inv_into[DT](wv, 0, OSC_W_TMP, OSC_W_TMP2, 3),
        "a singular matrix is REFUSED (the flag the kernel raises), not"
        " pseudo-inverted",
    )
    axisangle_mat_into[DT](wv, 0, OSC_W_TMP, Scalar[DT](0), Scalar[DT](0), Scalar[DT](0.1))
    var c = Float64(wt.data[OSC_W_TMP + 0])
    var s01 = Float64(wt.data[OSC_W_TMP + 1])
    ta.check(
        c > 0.995 and c < 0.9951 and s01 < -0.0998 and s01 > -0.0999,
        "axisangle_mat_into(0, 0, 0.1) is Rz(0.1): cos " + String(c)
        + ", -sin " + String(s01),
    )
    _inv_into[DT](wv, 0, OSC_W_TMP, OSC_W_TMP2, 3)
    _mul_into[DT](wv, 0, OSC_W_TMP, OSC_W_TMP2, OSC_W_LAMP, 3, 3, 3)
    var wo = 0.0
    for i in range(3):
        for j in range(3):
            var e = Float64(wt.data[OSC_W_LAMP + i * 3 + j]) - (1.0 if i == j else 0.0)
            if e < 0.0:
                e = -e
            if e > wo:
                wo = e
    ta.check(wo < 1e-12, "and it is orthonormal")

    # ── 2. the gripper ramp ─────────────────────────────────────────────
    print("--- gripper ramp ---")
    var g = PandaGripperRamp(0.0, 0.04, -0.04, 0.0)
    var c0 = g.step(-1.0)
    ta.check(c0[0] == 0.02 + 0.02 * 0.01 and c0[1] == -0.02 - 0.02 * 0.01,
             "one substep moves the ramp by 0.01 (opening): " + String(c0[0]))
    for _ in range(200):
        c0 = g.step(-1.0)
    ta.check(c0[0] == 0.04 and c0[1] == -0.04, "open saturates at the ctrlrange ends")
    for _ in range(200):
        c0 = g.step(1.0)
    ta.check(c0[0] == 0.0 and c0[1] == 0.0, "200 closing substeps (8 policy steps) take it from fully open to fully closed")
    var c1 = g.step(0.0)
    ta.check(c1[0] == c0[0], "a zero action holds (sign 0)")

    if not exists(PACK):
        print("  SKIPPED the scene checks: no LIBERO pack at", PACK)
        print("--- ran", ta.checks, "checks,", ta.failures, "failed ---")
        if ta.failures != 0:
            raise Error("osc_pose: failures")
        print("=== PASS (algebra + ramp; scene SKIPPED — not a full pass) ===")
        return

    # ── 3. hold / 4. track on the composed scene ────────────────────────
    print("--- the composed libero_goal scene ---")
    var f = load_family(FAMILY)
    var fmd = parse_model_runtime(scene_path(f))
    var verts = 32768
    var dims = dims_from_flat(fmd, max_contacts=LIBERO_GOAL_MAX_CONTACTS, nmesh_verts=verts)
    var m = Model[DT, DynDims](dims)
    while True:
        try:
            build_model_runtime[DT](fmd, dims, m)
            break
        except e:
            if String(e).find("mesh vertex capacity") < 0:
                raise e
            verts *= 2
            dims = dims_from_flat(fmd, max_contacts=LIBERO_GOAL_MAX_CONTACTS, nmesh_verts=verts)
            m = Model[DT, DynDims](dims)
    var sf = spec_fields_runtime[DT](fmd, dims, m)
    var d = Data[DT, DynDims, 1](dims)
    var scratch = DynamicsScratch[DT, DynDims, 1](dims)
    var integ = StudioIntegEll(dims)
    var nq = dims.get_nq()
    var nv = dims.get_nv()
    var nact = dims.get_nact()

    var qadr_all = List[Int]()
    var dadr_all = List[Int]()
    var qa = 0
    var da = 0
    for i in range(len(fmd.joints)):
        qadr_all.append(qa)
        dadr_all.append(da)
        qa += fmd.joints[i].nq
        da += fmd.joints[i].nv
    var dof = List[Int]()
    var qadr = List[Int]()
    var jidx = List[Int]()
    var act_idx = List[Int]()
    var tmin = List[Float64]()
    var tmax = List[Float64]()
    for j in range(ARM_DOF):
        var ji = _index(fmd.joint_names, String("robot_joint") + String(j + 1))
        dof.append(dadr_all[ji])
        qadr.append(qadr_all[ji])
        jidx.append(ji)
        var ai = _index(fmd.actuator_names, String("robot_torq_j") + String(j + 1))
        act_idx.append(ai)
        tmin.append(fmd.actuators[ai].ctrl_min)
        tmax.append(fmd.actuators[ai].ctrl_max)
    var site = _index(fmd.site_names, String("robot_grip_site"))
    var site_body = fmd.sites[site].body_id
    var ga1 = _index(fmd.actuator_names, String("robot_gripper_finger_joint1"))
    var ga2 = _index(fmd.actuator_names, String("robot_gripper_finger_joint2"))

    # the family's reset pose; free objects parked far away (qpos 0 = the
    # world origin is inside the table, so lift them well above it)
    for i in range(nq):
        d.qpos.data[i] = Scalar[DT](0)
    for i in range(nv):
        d.qvel.data[i] = Scalar[DT](0)
    for i in range(len(f.base_qpos)):
        d.qpos.data[i] = Scalar[DT](f.base_qpos[i])
    for i in range(len(fmd.joints)):
        if fmd.joints[i].nq == 7:
            d.qpos.data[qadr_all[i]] = Scalar[DT](2.0)
            d.qpos.data[qadr_all[i] + 2] = Scalar[DT](3.0)
            d.qpos.data[qadr_all[i] + 3] = Scalar[DT](1.0)

    var osc = OscPose(
        dof^, qadr^, jidx^, tmin^, tmax^, act_idx.copy(), site, site_body,
        ga1, ga2,
        fmd.actuators[ga1].ctrl_min, fmd.actuators[ga1].ctrl_max,
        fmd.actuators[ga2].ctrl_min, fmd.actuators[ga2].ctrl_max,
        OscPoseConfig(), nact, nq, nv,
    )
    osc.update(d, m, scratch)
    osc.reset(d, m)
    var p0 = osc.ee_pos(d)
    print("  grip site at reset:", p0[0], p0[1], p0[2])
    var act = List[Scalar[DT]](length=nact if nact > 0 else 1, fill=Scalar[DT](0))

    var zero = List[Float64](length=7, fill=0.0)
    zero[6] = -1.0
    var worst_hold = 0.0
    for _ in range(HOLD_STEPS):
        policy_step(osc, d, m, scratch, integ, act, sf, zero, nv, fmd.timestep)
        osc.update(d, m, scratch)
        var dd = _dist(osc.ee_pos(d), p0)
        if dd > worst_hold:
            worst_hold = dd
    print("  hold: worst grip-site drift over", HOLD_STEPS, "zero-action steps:", worst_hold, "m")
    ta.check(worst_hold < HOLD_TOL, "a zero action HOLDS the grip site (gravity cancelled, nullspace held)")
    ta.check(osc.is_ready(), "the controller reports a live goal")

    var px = List[Float64](length=7, fill=0.0)
    px[0] = 1.0
    px[6] = -1.0
    var p1 = osc.ee_pos(d)
    for _ in range(TRACK_STEPS):
        policy_step(osc, d, m, scratch, integ, act, sf, px, nv, fmd.timestep)
    osc.update(d, m, scratch)
    var pnow = osc.ee_pos(d)
    var moved = pnow[0] - p1[0]
    print("  track: +x for", TRACK_STEPS, "steps moved the grip site by", moved, "m in x;",
          "y/z drift", pnow[1] - p1[1], pnow[2] - p1[2])
    ta.check(moved > TRACK_MIN, "a +x action moves the grip site in +x by more than " + String(TRACK_MIN))
    var off = _dist(osc.ee_pos(d), p1) - moved
    ta.check(off < 0.02, "and the other two axes stay put (uncoupled): off-axis " + String(off))
    var mx = List[Float64](length=7, fill=0.0)
    mx[0] = -1.0
    mx[6] = -1.0
    var p2 = osc.ee_pos(d)
    for _ in range(TRACK_STEPS):
        policy_step(osc, d, m, scratch, integ, act, sf, mx, nv, fmd.timestep)
    osc.update(d, m, scratch)
    var pback = osc.ee_pos(d)
    var back = p2[0] - pback[0]
    print("  track: -x for", TRACK_STEPS, "steps moved it back by", back, "m in x;",
          _dist(pback, p1), "m from the start")
    ta.check(back > TRACK_MIN, "a -x action moves the grip site in -x by more than " + String(TRACK_MIN))

    print()
    print("--- ran", ta.checks, "checks,", ta.failures, "failed ---")
    if ta.failures != 0:
        raise Error("osc_pose: " + String(ta.failures) + " of " + String(ta.checks) + " failed")
    print("=== PASS ===")
