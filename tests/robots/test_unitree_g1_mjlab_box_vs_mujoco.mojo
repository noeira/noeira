"""Unitree G1 (mjlab's) plus a free box — our env vs MuJoCo under the same PD.

    pixi run mojo run -I . tests/robots/test_unitree_g1_mjlab_box_vs_mujoco.mojo

Both sides load OUR baked `assets/unitree_g1_mjlab_box.xml`
(`tests/robots/g1_mjlab_box_bake.py`; its `--check` is the bake's own gate, and
the robot inside it is `test_unitree_g1_mjlab_vs_mujoco`'s). The controller is
on both sides, not shared: ours is `unitree_g1_mjlab_config`'s Mojo PD,
MuJoCo's is `g1_mjlab_ref.ReferencePD`.

WHAT THE BOX ADDS: a second free body; box-plane, box-capsule and box-sphere
contacts; and MuJoCo's contact-parameter mixing by PRIORITY — the box geom is
`condim 3`, `priority 1`, so its contacts with the robot's `condim 1` geoms
are frictional with the box's friction, and with the feet (also priority 1)
take the larger friction.

    rest   the key pose, zero action: the box stays where it is
    drop   the box released tilted and spinning above the floor: it lands on a
           corner and tumbles (box-plane, friction)
    catch  the box released above the robot's forearms: it lands on the arms
           and the thighs (box-capsule with the mixed parameters)
    driven catch  the same with the joints driven, so the feet lift and the
           box's contacts land in contact slots the FEET held before — the
           case that showed a stale tangent hint under a box/capsule contact
           (`contact_detection._detect_contacts_env`'s note)

AGAINST THE CONVERGED REFERENCE (tolerance 0, 1000 iterations), over the
first N_GATE control steps, as the robot's own test.
"""

from std.math import abs, cos, sin
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite

from noeira.core.cont_action import ContAction
from noeira.envs.robots import UnitreeG1MjlabBox
from noeira.envs.robots.unitree_g1_mjlab_box_xml import (
    G1M_BOX_BODY, G1M_BOX_GEOM, G1M_BOX_QPOS, G1M_BOX_QVEL, G1M_BOX_REST_X, G1M_BOX_REST_Z, UnitreeG1MjlabBoxModel,
)
from noeira.envs.robots.unitree_g1_mjlab_xml import UnitreeG1MjlabModel
from noeira.envs.robots.unitree_g1_mjlab_pd import G1M_CONTROL_DECIMATION
from noeira.physics3d.gpu.constants import META_IDX_NUM_CONTACTS


comptime NQ = UnitreeG1MjlabBoxModel.NQ
comptime NV = UnitreeG1MjlabBoxModel.NV
comptime NU = 29
comptime N_STEPS = 60
comptime N_GATE = 40
comptime ASSET = "noeira/envs/robots/assets/unitree_g1_mjlab_box.xml"


def _mj(converged: Bool) raises -> PythonObject:
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_path(ASSET)
    if converged:
        m.opt.tolerance = 0.0
        m.opt.iterations = 1000
        m.opt.ls_iterations = 1000
    return m


def _pd() raises -> PythonObject:
    _ = Python.evaluate("__import__('sys').path.insert(0, 'tests/robots')")
    return Python.import_module("g1_mjlab_ref").ReferencePD()


def test_model_counts_and_slots() raises:
    var mujoco = Python.import_module("mujoco")
    var m = _mj(False)
    assert_true(Int(py=m.nbody) == UnitreeG1MjlabBoxModel.NBODY and Int(py=m.nbody) == UnitreeG1MjlabModel.NBODY + 1, "nbody")
    assert_true(Int(py=m.njnt) == UnitreeG1MjlabBoxModel.NJOINT, "njnt")
    assert_true(Int(py=m.nq) == NQ and NQ == UnitreeG1MjlabModel.NQ + 7, "nq")
    assert_true(Int(py=m.nv) == NV and NV == UnitreeG1MjlabModel.NV + 6, "nv")
    assert_true(Int(py=m.ngeom) == UnitreeG1MjlabBoxModel.NGEOM, "ngeom")
    assert_true(String(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, G1M_BOX_BODY)) == "box", "the box is not body " + String(G1M_BOX_BODY))
    assert_true(String(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, G1M_BOX_GEOM)) == "box_collision", "the box geom is not geom " + String(G1M_BOX_GEOM))
    var j = Int(py=m.njnt) - 1
    assert_true(Int(py=m.jnt_qposadr[j]) == G1M_BOX_QPOS and Int(py=m.jnt_dofadr[j]) == G1M_BOX_QVEL, "the box's state slots")
    assert_true(Int(py=m.geom_condim[G1M_BOX_GEOM]) == 3 and Int(py=m.geom_priority[G1M_BOX_GEOM]) == 1, "the box geom's condim / priority")
    print("  counts and slots OK — nbody", Int(py=m.nbody), " nq", NQ, " nv", NV, " box qpos at", G1M_BOX_QPOS)


def _set_box(mujoco: PythonObject, m: PythonObject, d: PythonObject, mut env: UnitreeG1MjlabBox[DType.float64], q: List[Float64], v: List[Float64]) raises:
    """The box's 7 qpos and 6 qvel on both sides, the robot left at the key."""
    var qpos = List[Float64]()
    var qvel = List[Float64]()
    for i in range(NQ):
        qpos.append(Float64(env.d.qpos.data[i]))
    for i in range(NV):
        qvel.append(Float64(env.d.qvel.data[i]))
    for k in range(7):
        qpos[G1M_BOX_QPOS + k] = q[k]
        d.qpos[G1M_BOX_QPOS + k] = q[k]
    for k in range(6):
        qvel[G1M_BOX_QVEL + k] = v[k]
        d.qvel[G1M_BOX_QVEL + k] = v[k]
    env.set_state(qpos, qvel)
    mujoco.mj_forward(m, d)


def _box_robot_contacts(d: PythonObject) raises -> Int:
    """MuJoCo's contacts between the box geom and a geom of the ROBOT (not the
    floor, geom 0)."""
    var n = 0
    for i in range(Int(py=d.ncon)):
        var g1 = Int(py=d.contact[i].geom1)
        var g2 = Int(py=d.contact[i].geom2)
        if (g1 == G1M_BOX_GEOM and g2 != 0) or (g2 == G1M_BOX_GEOM and g1 != 0):
            n += 1
    return n


def _rollout(label: String, q: List[Float64], v: List[Float64], amp: Float64) raises -> Tuple[Float64, Float64, Int, Float64]:
    """(worst |d qpos| over the first N_GATE steps vs the converged reference,
    the same over the BOX's 7 only, the number of the first N_GATE steps that
    end with the box touching the robot in MuJoCo, how far the box moved)."""
    var mujoco = Python.import_module("mujoco")
    var mc = _mj(True)
    var ms = _mj(False)
    var dc = mujoco.MjData(mc)
    var ds = mujoco.MjData(ms)
    var pd = _pd()
    pd.stand_state(mujoco, mc, dc)
    pd.stand_state(mujoco, ms, ds)
    var env = UnitreeG1MjlabBox[DType.float64]()
    _ = env.reset()
    for i in range(NQ):
        assert_true(Float64(env.d.qpos.data[i]) == Float64(py=dc.qpos[i]), "reset qpos[" + String(i) + "] != the keyframe")
    if len(q) == 7:
        _set_box(mujoco, mc, dc, env, q, v)
        _set_box(mujoco, ms, ds, env, q, v)
    var start = List[Float64]()
    for k in range(3):
        start.append(Float64(py=dc.qpos[G1M_BOX_QPOS + k]))
    var worst_gate = 0.0
    var worst_box = 0.0
    var worst_all = 0.0
    var worst_ship = 0.0
    var ncon_max = 0
    var ours_ncon_max = 0
    var touch_steps = 0
    assert_true(String(mujoco.mj_id2name(mc, mujoco.mjtObj.mjOBJ_GEOM, 0)) == "floor", "geom 0 is not the floor")
    for t in range(N_STEPS):
        var a = ContAction[UnitreeG1MjlabBoxModel.ACTION_DIM]()
        var pa = Python.evaluate("[]")
        for j in range(NU):
            var u = amp * sin(Float64(t) * 0.23 + Float64(j) * 0.61)
            a.data[j] = u
            pa.append(u)
        var nc = Int(py=pd.control_step(mujoco, mc, dc, pa, G1M_CONTROL_DECIMATION))
        _ = pd.control_step(mujoco, ms, ds, pa, G1M_CONTROL_DECIMATION)
        ncon_max = max(ncon_max, nc)
        _ = env.step(a)
        ours_ncon_max = max(ours_ncon_max, Int(env.d.meta.data[META_IDX_NUM_CONTACTS]))
        var sw = 0.0
        var sb = 0.0
        var ss = 0.0
        for i in range(NQ):
            var e = abs(Float64(env.d.qpos.data[i]) - Float64(py=dc.qpos[i]))
            sw = max(sw, e)
            if i >= G1M_BOX_QPOS:
                sb = max(sb, e)
            ss = max(ss, abs(Float64(env.d.qpos.data[i]) - Float64(py=ds.qpos[i])))
        worst_all = max(worst_all, sw)
        worst_ship = max(worst_ship, ss)
        if t < N_GATE:
            worst_gate = max(worst_gate, sw)
            worst_box = max(worst_box, sb)
            if _box_robot_contacts(dc) > 0:
                touch_steps += 1
        if t < 2 or t % 10 == 9:
            print("      ", label, "step", t, " |dqpos| vs converged", sw, " box", sb, " vs shipped", ss, " mj ncon", nc,
                  " ours ncon", Int(env.d.meta.data[META_IDX_NUM_CONTACTS]), " box z", Float64(py=dc.qpos[G1M_BOX_QPOS + 2]))
    var moved = 0.0
    for k in range(3):
        moved = max(moved, abs(Float64(py=dc.qpos[G1M_BOX_QPOS + k]) - start[k]))
    print("  ", label, ": worst vs converged first", N_GATE, worst_gate, "(box", worst_box, ") all", worst_all, "| vs shipped all", worst_ship,
          "| max ncon mj", ncon_max, "ours", ours_ncon_max, "| box moved", moved, "| gate steps with the box on the robot", touch_steps)
    return (worst_gate, worst_box, touch_steps, moved)


def test_rest_rollout() raises:
    """The key pose, zero action: the box rests on the floor for the first
    N_GATE steps (the robot sags under its soft servos, as in the robot's own
    test, and does not reach the box before the gate)."""
    var none = List[Float64]()
    var r = _rollout("rest", none, none, 0.0)
    assert_true(r[0] < 1e-6, "rest rollout residual " + String(r[0]))
    assert_true(r[3] < 1e-3, "the resting box moved " + String(r[3]))
    assert_true(r[2] == 0, "the resting box touched the robot")


def test_drop_rollout() raises:
    """Released 0.45 m up, tilted about (1, 1, 0) by 0.5 rad, spinning: lands on
    a corner inside the gate window and tumbles."""
    var h = 0.25
    var c = cos(h)
    var s = sin(h) * 0.7071067811865476
    var q: List[Float64] = [G1M_BOX_REST_X + 0.3, 0.2, 0.45, c, s, s, 0.0]
    var v: List[Float64] = [0.3, -0.2, 0.0, 1.0, -2.0, 0.5]
    var r = _rollout("drop", q, v, 0.0)
    assert_true(r[3] > 0.1, "the box did not fall (moved " + String(r[3]) + ")")
    assert_true(r[0] < 1e-5, "drop rollout residual " + String(r[0]))


def test_catch_rollout() raises:
    """Released above the forearms of the standing robot: lands on the arms
    (capsules, condim 1 against the box's condim 3 / priority 1) and goes down
    with them."""
    var q: List[Float64] = [0.27, 0.0, 1.05, 1.0, 0.0, 0.0, 0.0]
    var v: List[Float64] = [0.0, 0.0, -0.5, 0.0, 0.0, 0.0]
    var r = _rollout("catch", q, v, 0.0)
    assert_true(r[2] >= 3, "the box touched the robot on only " + String(r[2]) + " of the gated steps")
    assert_true(r[0] < 1e-5, "catch rollout residual " + String(r[0]))


def test_driven_catch_rollout() raises:
    """`catch` with the joints driven (amplitude 0.3, the device gate's
    drive). The feet's 28 standing contacts drop to about 13 as the body
    moves, and the box's frictional contacts with the arms then take slots
    13 and up, which held foot capsule/plane contacts a few steps earlier.

    Before 2026-11-12 those contacts read the foot capsule's axis as their
    tangent hint (only the plane/capsule branches write the hint, and nothing
    cleared it): the friction pyramid was turned about the normal and the
    box's velocity was off MuJoCo's by 5e-2 from the first touch. The plain
    `catch` never saw it, because a standing robot keeps slots 0-27."""
    var q: List[Float64] = [0.27, 0.0, 1.05, 1.0, 0.0, 0.0, 0.0]
    var v: List[Float64] = [0.0, 0.0, -0.5, 0.0, 0.0, 0.0]
    var r = _rollout("driven catch", q, v, 0.3)
    assert_true(r[2] >= 3, "the box touched the robot on only " + String(r[2]) + " of the gated steps")
    assert_true(r[0] < 1e-5, "driven catch rollout residual " + String(r[0]))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
