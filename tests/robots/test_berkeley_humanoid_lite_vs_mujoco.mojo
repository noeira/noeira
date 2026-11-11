"""Berkeley Humanoid Lite — parity: our env vs MuJoCo under the same PD.

    pixi run mojo run -I . tests/robots/test_berkeley_humanoid_lite_vs_mujoco.mojo

Both sides load OUR baked `assets/berkeley_humanoid_lite.xml`
(`tests/robots/bhl_bake.py` states how it differs from BHL's file, and
`tests/robots/bhl_ref.py` what the 2 ms step costs).

⚠ THE CONTROLLER IS ON BOTH SIDES, NOT SHARED: our side runs
`berkeley_humanoid_lite_config`'s Mojo PD from the generated
`berkeley_humanoid_lite_pd.mojo` (parsed from BHL's Isaac Lab config);
MuJoCo's side runs `bhl_ref.ReferencePD`, transcribed from BHL's
`policy_humanoid.yaml`.

WHAT THIS MODEL EXERCISES: box and cylinder geoms against a plane, no
self-collision, armature and friction loss on every hinge, a torque cut above
a joint speed. The `fall` rollout drives the body to the floor so the torso
box and the arm cylinders touch too, and joints pass the speed limit.

AGAINST THE CONVERGED REFERENCE (tolerance 0, 1000 iterations); the residual
against MuJoCo as shipped is printed beside it.
"""

from std.math import abs, sin
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite

from noeira.core.cont_action import ContAction
from noeira.envs.robots import BerkeleyHumanoidLite
from noeira.envs.robots.berkeley_humanoid_lite_xml import BerkeleyHumanoidLiteModel
from noeira.envs.robots.berkeley_humanoid_lite_pd import (
    BHL_N_DOF, BHL_CONTROL_DECIMATION, bhl_dof_name, bhl_effort, bhl_speed,
)
from noeira.physics3d.fields import actuator_column
from noeira.physics3d.gpu.constants import ACT_IDX_CTRL_MAX, ACT_IDX_CTRL_MIN, META_IDX_NUM_CONTACTS


comptime NQ = BerkeleyHumanoidLiteModel.NQ
comptime NU = 22
comptime N_STEPS = 100
comptime N_GATE = 40
comptime ASSET = "noeira/envs/robots/assets/berkeley_humanoid_lite.xml"


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
    return Python.import_module("bhl_ref").ReferencePD()


def test_model_counts() raises:
    var m = _mj(False)
    assert_true(Int(py=m.nbody) == BerkeleyHumanoidLiteModel.NBODY, "nbody")
    assert_true(Int(py=m.njnt) == BerkeleyHumanoidLiteModel.NJOINT, "njnt")
    assert_true(Int(py=m.nq) == BerkeleyHumanoidLiteModel.NQ, "nq")
    assert_true(Int(py=m.nv) == BerkeleyHumanoidLiteModel.NV, "nv")
    assert_true(Int(py=m.nu) == NU, "nu")
    assert_true(Int(py=m.ngeom) == BerkeleyHumanoidLiteModel.NGEOM, "ngeom")
    assert_true(Int(py=m.nkey) == 1, "nkey")
    print("  counts OK — nbody", Int(py=m.nbody), " ngeom", Int(py=m.ngeom),)


def test_joint_order_and_ctrlrange() raises:
    var mujoco = Python.import_module("mujoco")
    var m = _mj(False)
    var sf = BerkeleyHumanoidLiteModel.make_spec_fields[DType.float64]()
    var cmin = actuator_column(sf, ACT_IDX_CTRL_MIN, NU)
    var cmax = actuator_column(sf, ACT_IDX_CTRL_MAX, NU)
    for i in range(BHL_N_DOF):
        var jn = String(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i + 1))
        assert_true(jn == String(bhl_dof_name(i)), "joint " + String(i + 1) + " is " + jn)
        assert_true(Int(py=m.actuator_trnid[i][0]) == i + 1, "actuator " + String(i) + " drives the wrong joint")
        var hi = Float64(py=m.actuator_ctrlrange[i][1])
        assert_true(cmax[i] == hi and cmin[i] == -hi, "ctrlrange of actuator " + String(i))
        assert_true(hi == bhl_effort(i), "ctrlrange != effort on actuator " + String(i))
    print("  joint order = the PD table's; ctrlrange = effort on all 22")


def _action(t: Int, j: Int, amp: Float64) -> Float64:
    return amp * sin(Float64(t) * 0.23 + Float64(j) * 0.61)


def _rollout(amp: Float64, label: String) raises -> Tuple[Float64, Float64, Int, Float64]:
    """(worst |d qpos| over the first N_GATE steps vs the converged
    reference, worst over N_STEPS, MuJoCo's max ncon, the fastest joint as
    a multiple of its speed limit); prints the shipped reference's residual."""
    var mujoco = Python.import_module("mujoco")
    var mc = _mj(True)
    var ms = _mj(False)
    var dc = mujoco.MjData(mc)
    var ds = mujoco.MjData(ms)
    var pd = _pd()
    pd.stand_state(mujoco, mc, dc)
    pd.stand_state(mujoco, ms, ds)
    var env = BerkeleyHumanoidLite[DType.float64]()
    _ = env.reset()
    for i in range(NQ):
        assert_true(Float64(env.d.qpos.data[i]) == Float64(py=dc.qpos[i]), "reset qpos[" + String(i) + "] != the keyframe")
    var worst = 0.0
    var worst_gate = 0.0
    var worst_ship = 0.0
    var ncon_max = 0
    var ours_ncon_max = 0
    var fastest = 0.0
    for t in range(N_STEPS):
        var a = ContAction[BerkeleyHumanoidLiteModel.ACTION_DIM]()
        var pa = Python.evaluate("[]")
        for j in range(NU):
            var u = _action(t, j, amp)
            a.data[j] = u
            pa.append(u)
        var nc = Int(py=pd.control_step(mujoco, mc, dc, pa, BHL_CONTROL_DECIMATION))
        _ = pd.control_step(mujoco, ms, ds, pa, BHL_CONTROL_DECIMATION)
        ncon_max = max(ncon_max, nc)
        _ = env.step(a)
        ours_ncon_max = max(ours_ncon_max, Int(env.d.meta.data[META_IDX_NUM_CONTACTS]))
        for j in range(NU):
            fastest = max(fastest, abs(Float64(env.d.qvel.data[6 + j])) / bhl_speed(j))
        var sw = 0.0
        var ss = 0.0
        for i in range(NQ):
            sw = max(sw, abs(Float64(env.d.qpos.data[i]) - Float64(py=dc.qpos[i])))
            ss = max(ss, abs(Float64(env.d.qpos.data[i]) - Float64(py=ds.qpos[i])))
        worst = max(worst, sw)
        worst_ship = max(worst_ship, ss)
        if t < N_GATE:
            worst_gate = max(worst_gate, sw)
        if t < 3 or t % 20 == 19:
            print("      ", label, "step", t, " |dqpos| vs converged", sw, " vs shipped", ss, " mj ncon", nc,
                  " ours ncon", Int(env.d.meta.data[META_IDX_NUM_CONTACTS]), " root z", Float64(py=dc.qpos[2]))
    print("  ", label, ": worst vs converged first", N_GATE, worst_gate, " all", worst, "| vs shipped all", worst_ship,
          "| max ncon mj", ncon_max, "ours", ours_ncon_max, "| fastest joint / its speed limit", fastest)
    return (worst_gate, worst, ncon_max, fastest)


def test_stand_rollout() raises:
    """Zero action from the keyframe: with no policy the body tips over
    within two seconds and ends on the floor (the torso box and the arms
    touch: more than the feet's 8 contact points). Gated on the first N_GATE
    steps.

    Measured 2026-11-11: first 40 steps 2.3e-8 vs converged (4e-14 vs MuJoCo
    as shipped, all 100 steps), max ncon 14."""
    var r = _rollout(0.0, "stand")
    assert_true(r[2] > 8, "the body never reached the floor (max ncon " + String(r[2]) + ")")
    assert_true(r[0] < 1e-6, "stand rollout residual " + String(r[0]))


def test_driven_rollout() raises:
    """Measured 2026-11-11: 5.8e-8 vs converged over all 100 steps."""
    var r = _rollout(0.3, "driven")
    assert_true(r[0] < 1e-6, "driven rollout residual " + String(r[0]))


def test_fall_rollout() raises:
    """Large actions: the body goes down and the torso and arms touch the
    floor; joints exceed their speed limit, so the torque cut is exercised.

    Measured 2026-11-11: first 40 steps 3.8e-14 vs converged, fastest joint
    2.7 times its speed limit."""
    var r = _rollout(4.0, "fall")
    assert_true(r[2] > 0, "no contact formed")
    assert_true(r[3] > 1.0, "no joint ever passed its speed limit: the torque cut was not exercised")
    assert_true(r[0] < 1e-6, "fall rollout residual " + String(r[0]))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
