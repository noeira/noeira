"""Unitree G1 as mjlab models it — layer-2 parity: our env vs MuJoCo under the same PD.

    pixi run mojo run -I . tests/robots/test_unitree_g1_mjlab_vs_mujoco.mojo

⚠ LAYER 2 DOES NOT REPLACE LAYER 1. Both sides load OUR baked
`assets/unitree_g1_mjlab.xml`; `tests/robots/g1_mjlab_ref.py` proves that
file is mjlab's runtime model. Run it first.

⚠ THE CONTROLLER IS ON BOTH SIDES, NOT SHARED: our side runs
`unitree_g1_mjlab_config`'s Mojo PD from the generated
`unitree_g1_mjlab_pd.mojo`; MuJoCo's side runs `g1_mjlab_ref.ReferencePD`, a
separate transcription of mjlab's `g1_constants.py`.

WHAT THIS MODEL ADDS OVER `test_unitree_g1_vs_mujoco`: capsule and sphere
collision geoms only (no mesh), condim 1 on everything but the feet, foot
priority 1 / friction 0.6, self-collisions (contype = conaffinity = 1 on
every collision geom, 4 excluded body pairs), per-joint armature, and the
reproduction's 10 Newton iterations. The `fall` rollout drives the body to
the floor so knees, hands, pelvis and self-contacts all engage.

AGAINST THE CONVERGED REFERENCE (tolerance 0, 1000 iterations) as the BFM G1
test does — and the residual against MuJoCo as shipped (10 iterations, the
reproduction's) is printed beside it.
"""

from std.math import abs, sin
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite

from noeira.core.cont_action import ContAction
from noeira.envs.robots import UnitreeG1Mjlab
from noeira.envs.robots.unitree_g1_mjlab_xml import UnitreeG1MjlabModel
from noeira.envs.robots.unitree_g1_mjlab_pd import G1M_N_DOF, G1M_CONTROL_DECIMATION, g1m_dof_name, g1m_effort
from noeira.physics3d.fields import actuator_column
from noeira.physics3d.gpu.constants import ACT_IDX_CTRL_MAX, ACT_IDX_CTRL_MIN, META_IDX_NUM_CONTACTS


comptime NQ = UnitreeG1MjlabModel.NQ
comptime NU = 29
comptime N_STEPS = 100
comptime N_GATE = 40
comptime ASSET = "noeira/envs/robots/assets/unitree_g1_mjlab.xml"


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


def test_model_counts() raises:
    var m = _mj(False)
    assert_true(Int(py=m.nbody) == UnitreeG1MjlabModel.NBODY, "nbody")
    assert_true(Int(py=m.njnt) == UnitreeG1MjlabModel.NJOINT, "njnt")
    assert_true(Int(py=m.nq) == UnitreeG1MjlabModel.NQ, "nq")
    assert_true(Int(py=m.nv) == UnitreeG1MjlabModel.NV, "nv")
    assert_true(Int(py=m.nu) == NU, "nu")
    assert_true(Int(py=m.ngeom) == UnitreeG1MjlabModel.NGEOM, "ngeom")
    assert_true(Int(py=m.nexclude) == UnitreeG1MjlabModel.NEXCLUDE, "nexclude")
    assert_true(Int(py=m.nkey) == 1, "nkey")
    assert_true(Int(py=m.opt.iterations) == 10 and Int(py=m.opt.ls_iterations) == 20, "solver iterations")
    print("  counts OK — nbody", Int(py=m.nbody), " ngeom", Int(py=m.ngeom), " nexclude", Int(py=m.nexclude))


def test_joint_order_and_ctrlrange() raises:
    var mujoco = Python.import_module("mujoco")
    var m = _mj(False)
    var sf = UnitreeG1MjlabModel.make_spec_fields[DType.float64]()
    var cmin = actuator_column(sf, ACT_IDX_CTRL_MIN, NU)
    var cmax = actuator_column(sf, ACT_IDX_CTRL_MAX, NU)
    for i in range(G1M_N_DOF):
        var jn = String(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i + 1))
        assert_true(jn == String(g1m_dof_name(i)), "joint " + String(i + 1) + " is " + jn)
        assert_true(Int(py=m.actuator_trnid[i][0]) == i + 1, "actuator " + String(i) + " drives the wrong joint")
        var hi = Float64(py=m.actuator_ctrlrange[i][1])
        assert_true(cmax[i] == hi and cmin[i] == -hi, "ctrlrange of actuator " + String(i))
        assert_true(hi == g1m_effort(i), "ctrlrange != effort on actuator " + String(i))
    print("  joint order = the PD table's; ctrlrange = effort on all 29")


def _action(t: Int, j: Int, amp: Float64) -> Float64:
    return amp * sin(Float64(t) * 0.23 + Float64(j) * 0.61)


def _rollout(amp: Float64, label: String) raises -> Tuple[Float64, Float64, Int]:
    """(worst |d qpos| over the first N_GATE steps vs the converged
    reference, worst over N_STEPS, MuJoCo's max ncon); prints the shipped
    reference's residual too."""
    var mujoco = Python.import_module("mujoco")
    var mc = _mj(True)
    var ms = _mj(False)
    var dc = mujoco.MjData(mc)
    var ds = mujoco.MjData(ms)
    var pd = _pd()
    pd.stand_state(mujoco, mc, dc)
    pd.stand_state(mujoco, ms, ds)
    var env = UnitreeG1Mjlab[DType.float64]()
    _ = env.reset()
    for i in range(NQ):
        assert_true(Float64(env.d.qpos.data[i]) == Float64(py=dc.qpos[i]), "reset qpos[" + String(i) + "] != the keyframe")
    var worst = 0.0
    var worst_gate = 0.0
    var worst_ship = 0.0
    var ncon_max = 0
    var ours_ncon_max = 0
    for t in range(N_STEPS):
        var a = ContAction[UnitreeG1MjlabModel.ACTION_DIM]()
        var pa = Python.evaluate("[]")
        for j in range(NU):
            var u = _action(t, j, amp)
            a.data[j] = u
            pa.append(u)
        var nc = Int(py=pd.control_step(mujoco, mc, dc, pa, G1M_CONTROL_DECIMATION))
        _ = pd.control_step(mujoco, ms, ds, pa, G1M_CONTROL_DECIMATION)
        ncon_max = max(ncon_max, nc)
        _ = env.step(a)
        ours_ncon_max = max(ours_ncon_max, Int(env.d.meta.data[META_IDX_NUM_CONTACTS]))
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
    print("  ", label, ": worst vs converged first", N_GATE, worst_gate, " all", worst, "| vs shipped (10 its) all", worst_ship,
          "| max ncon mj", ncon_max, "ours", ours_ncon_max)
    return (worst_gate, worst, ncon_max)


def test_stand_rollout() raises:
    """Zero action from the keyframe. ⚠ mjlab's servos are soft (hip pitch kp
    40): with no policy the body sags and falls by ~1.2 s (root z 0.76 ->
    0.13 at step 79, on both sides), after which the residual grows
    chaotically. Gated on the first N_GATE steps like the others.

    Measured 2026-11-04: first 40 steps 3.5e-8 vs converged (2e-15 vs
    MuJoCo as shipped), MuJoCo max ncon 28; all 100: 4.2e-5."""
    var r = _rollout(0.0, "stand")
    assert_true(r[2] > 0, "no contact formed")
    assert_true(r[0] < 1e-6, "stand rollout residual " + String(r[0]))


def test_driven_rollout() raises:
    """Measured 2026-11-04: first 40 steps 6.5e-8 vs converged (9e-16 vs
    shipped), max ncon 28; all 100 (after the fall): 6.1e-3."""
    var r = _rollout(0.3, "driven")
    assert_true(r[0] < 1e-6, "driven rollout residual " + String(r[0]))


def test_fall_rollout() raises:
    """Large actions: the body goes down, knees / hands / pelvis and
    self-contacts engage. Gated on the first N_GATE steps.

    Measured 2026-11-04: first 40 steps 2.5e-14 vs converged, max ncon 18;
    the body is down (root z 0.32) by step 39."""
    var r = _rollout(2.0, "fall")
    assert_true(r[2] > 8, "the fall never brought more than the feet down (max ncon " + String(r[2]) + ")")
    assert_true(r[0] < 1e-6, "fall rollout residual " + String(r[0]))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
