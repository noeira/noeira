"""`ctrlrange` is only applied when the actuator is `ctrllimited`.

`apply_actions` clamped every actuator's `ctrl` to `motor_ctrl_min/max`
unconditionally, and those FALL BACK TO (-1, 1) when no level of the model
supplies a `ctrlrange`. So an actuator MuJoCo leaves unclamped had its command
silently squeezed into +-1.

⚠ `ctrllimited` DEFAULTS TO "auto", NOT TO FALSE. The absent attribute means
"limited iff a range was defined", which is why the missing check is invisible
on any model that declares ranges — and every model in this tree does.
Measured against the 3.10.0 runtime with `qfrc_actuator` at `ctrl = 5.0`:

    <motor/>                                      limited 0  [0, 0]   +5.0
    <motor ctrlrange="-1 1"/>                     limited 1  [-1, 1]  +1.0
    <motor ctrlrange="-2 3"/>                     limited 1  [-2, 3]  +3.0
    <motor ctrlrange="0 0"/>                      limited 0  [0, 0]   +5.0
    <motor ctrlrange="-1 1" ctrllimited="false"/> limited 0  [-1, 1]  +5.0
    <motor ctrllimited="false"/>                  limited 0  [0, 0]   +5.0

(`ctrllimited="true"` with no range is a COMPILE ERROR in MuJoCo — "invalid
control range for actuator" — so limited-with-zero-range is unrepresentable
and this file does not test it. Same as `forcelimited`.)

⚠⚠ WHY NOTHING HERE CAUGHT IT, AND WHY IT STILL MATTERS. Measured across the
31 dm_control + Gymnasium reference models: **0 of 254 actuators are
unlimited** — default classes supply a ctrlrange everywhere, so the guard was
dead code on every model this engine has ever run. It is NOT dead on the
models being ported now: **423 of Menagerie's 2244 actuators are unlimited**,
and ToddlerBot is **30 of 30** on every variant. Its actuators are
`<position>`, whose `ctrl` is a target ANGLE, over joints ranging to 18.5 rad
— so that robot could not have been commanded past 1 radian, on any joint.

⚠ A GREP CANNOT MEASURE THIS. Scanning actuator tags for a missing
`ctrlrange` reports dog and quadruped as unlimited; both inherit ranges from a
`class="..."` default and are fully limited. Every count above comes from
loading the model and reading `m.actuator_ctrllimited`.

TWO CLAMP SITES, both gated here: `apply_actions` (CPU) and
`apply_actions_kernel_gpu`. Both were unconditional, and fixing one alone
would have left the two targets computing different forces from the same
action — the shape of defect #54.

Run with:
    pixi run -e apple mojo run -I . tests/physics3d/test_ctrllimited_vs_mujoco.mojo
"""

from std.math import abs
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite
from noeira.core.tiles import RowMajor
from max.gpu.host import DeviceContext

from noeira.nn.core.tensor import TensorImpl
from noeira.physics3d.parser import parse_xml, ModelDefFromXML
from noeira.physics3d.gpu.constants import (
    MODEL_ACTUATOR_SIZE,
    MODEL_ACT_TENDON_SIZE,
    JLIM_SIZE,
    METADATA_SIZE,
)
from noeira.physics3d.fields import Data, SpecFields, Dims
from noeira.physics3d.model.model_dims import ModelDims

comptime DTYPE = DType.float64

# SIX actuators, one per spelling, on six independent hinges. One model so the
# whole matrix costs one compile — and so a mistake that keys off actuator
# INDEX rather than actuator attributes cannot pass by accident.
#
# ⚠ THREE DIFFERENT REASONS TO BE UNLIMITED are present (no range at all, an
# explicit `ctrllimited="false"` over a real range, and the degenerate
# `"0 0"` range), because they take different branches in the resolver. A
# fixture with only the first would leave the other two untested.
comptime XML = String(
    """<mujoco model="ctrllimited_matrix">
  <option timestep="0.001" gravity="0 0 0"/>
  <worldbody>
    <body name="b0" pos="0 0 0"><joint name="j0" type="hinge" axis="0 0 1"/>
      <geom type="capsule" fromto="0 0 0 .2 0 0" size=".02" mass="1"/></body>
    <body name="b1" pos="0 1 0"><joint name="j1" type="hinge" axis="0 0 1"/>
      <geom type="capsule" fromto="0 0 0 .2 0 0" size=".02" mass="1"/></body>
    <body name="b2" pos="0 2 0"><joint name="j2" type="hinge" axis="0 0 1"/>
      <geom type="capsule" fromto="0 0 0 .2 0 0" size=".02" mass="1"/></body>
    <body name="b3" pos="0 3 0"><joint name="j3" type="hinge" axis="0 0 1"/>
      <geom type="capsule" fromto="0 0 0 .2 0 0" size=".02" mass="1"/></body>
    <body name="b4" pos="0 4 0"><joint name="j4" type="hinge" axis="0 0 1"/>
      <geom type="capsule" fromto="0 0 0 .2 0 0" size=".02" mass="1"/></body>
    <body name="b5" pos="0 5 0"><joint name="j5" type="hinge" axis="0 0 1"/>
      <geom type="capsule" fromto="0 0 0 .2 0 0" size=".02" mass="1"/></body>
  </worldbody>
  <actuator>
    <motor name="a0" joint="j0"/>
    <motor name="a1" joint="j1" ctrlrange="-1 1"/>
    <motor name="a2" joint="j2" ctrlrange="-2 3"/>
    <motor name="a3" joint="j3" ctrlrange="-1 1" ctrllimited="false"/>
    <motor name="a4" joint="j4" ctrllimited="false"/>
    <motor name="a5" joint="j5" ctrlrange="0 0"/>
  </actuator>
</mujoco>"""
)

comptime pm = parse_xml(XML)
comptime M = ModelDefFromXML[
    xml=XML,
    nbody=pm.NBODY, njoint=pm.NJOINT, nq=pm.NQ, nv=pm.NV,
    ngeom=pm.NGEOM, nact=pm.NACT, ntex=pm.NTEX, nmat=pm.NMAT,
    nlight=pm.NLIGHT, ncam=pm.NCAM, nsite=pm.NSITE, neq=pm.NEQ,
    nexclude=pm.NEXCLUDE, npair=pm.NPAIR, max_tendon=pm.NTENDON,
    max_condim=pm.MAX_CONDIM, max_contacts=8,
    obs_dim_override=1, obs_qpos_skip=0,
    timestep=pm.TIMESTEP,
    noslip_iter=pm.NOSLIP_ITER,
]
comptime MD = ModelDims[M]

# `ctrl` well outside every range in the fixture, so the clamp — if it fires —
# is unmistakable. The pre-fix answers were [1, 1, 3, 1, 1, 0]: four of six
# wrong, and the last one driven to ZERO by the degenerate range.
comptime PROBE_CTRL: Float64 = 5.0
comptime TOL: Float64 = 1e-12
# The GPU leg runs float32 (Metal rejects `double`), so it gets its own bound.
comptime TOL32: Float64 = 1e-6


@always_inline
def _max1(n: Int) -> Int:
    """A zero-extent tensor operand segfaults rather than being empty."""
    return n if n > 0 else 1


def _mj_qfrc(mujoco: PythonObject, ctrl: Float64) raises -> List[Float64]:
    """MuJoCo's `qfrc_actuator` with every `ctrl` set to `ctrl`."""
    var m = mujoco.MjModel.from_xml_string(XML)
    var d = mujoco.MjData(m)
    var nu = Int(py=m.nu)
    for i in range(nu):
        d.ctrl[i] = ctrl
    mujoco.mj_forward(m, d)
    var out = List[Float64]()
    for i in range(Int(py=m.nv)):
        out.append(Float64(py=d.qfrc_actuator[i]))
    return out^


def test_ctrllimited_flag_matches_mujoco() raises:
    """Our resolved `ctrllimited` against `m.actuator_ctrllimited`, per
    actuator. This is the parse; the two legs below are the consequence."""
    var sf = M.make_spec_fields[DTYPE]()
    print("=== ctrllimited: the resolved flag vs MuJoCo ===")
    var warnings = Python.import_module("warnings")
    _ = warnings.filterwarnings("ignore")
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(XML)
    var nu = Int(py=m.nu)

    assert_true(
        nu == 6 and M.nact == 6,
        "the fixture must expose all six spellings (MuJoCo nu=" + String(nu)
        + ", ours nact=" + String(M.nact) + ")",
    )

    var n_lim = 0
    var mismatches = 0
    for i in range(nu):
        var mj_lim = Int(py=m.actuator_ctrllimited[i]) != 0
        var our_lim = M.ctrl_limited_at[DTYPE](sf, i)
        if mj_lim:
            n_lim += 1
        if mj_lim != our_lim:
            mismatches += 1
        print("   act", i, " MuJoCo limited", mj_lim, " ours", our_lim,
              "  our range [", M.ctrl_min_at[DTYPE](sf, i), ",", M.ctrl_max_at[DTYPE](sf, i), "]")

    # Vacuity: the fixture must contain BOTH kinds, or "all limited" and "all
    # unlimited" would both pass.
    assert_true(
        n_lim == 2 and nu - n_lim == 4,
        "the fixture stopped straddling the boundary (" + String(n_lim)
        + " limited, " + String(nu - n_lim) + " unlimited) — expected 2 and 4,"
        " so it no longer gates the resolver",
    )
    assert_true(
        mismatches == 0,
        String(mismatches) + " of " + String(nu) + " actuators disagree with"
        " MuJoCo on `ctrllimited`. ⚠ the absent attribute means AUTO ('limited"
        " iff a range was defined'), NOT false, and `ctrlrange=\"0 0\"` is the"
        " undefined marker — check which spelling before adjusting anything",
    )
    print("  PASS")


def test_apply_actions_cpu_matches_mujoco() raises:
    """`apply_actions` (CPU) against `qfrc_actuator` at an out-of-range ctrl.

    ⚠ THE OBSERVABLE IS THE FORCE, not the flag. A resolver that got the flag
    right while the clamp stayed unconditional would pass the leg above and
    fail here, which is the split this file exists to close."""
    var sf = M.make_spec_fields[DTYPE]()
    print("=== ctrllimited: apply_actions (CPU) vs MuJoCo ===")
    var warnings = Python.import_module("warnings")
    _ = warnings.filterwarnings("ignore")
    var mujoco = Python.import_module("mujoco")
    var mj = _mj_qfrc(mujoco, PROBE_CTRL)

    var d = Data[DTYPE, MD, 1]()
    M.reset_data[DTYPE](sf, d)
    var actions = List[Float64]()
    for _ in range(M.nact):
        actions.append(PROBE_CTRL)
    var act = List[Scalar[DTYPE]]()
    for _ in range(M.NA_F):
        act.append(Scalar[DTYPE](0))
    M.apply_actions[DTYPE](sf, d, actions, act)

    var worst = Float64(0)
    var n_unclamped = 0
    for i in range(M.NV):
        var ours = Float64(d.qfrc.data[i])
        var e = abs(ours - mj[i])
        if e > worst:
            worst = e
        if abs(mj[i] - PROBE_CTRL) < TOL:
            n_unclamped += 1
        print("   dof", i, " ours", ours, " MuJoCo", mj[i], "  |d|", e)

    # Vacuity: at least one actuator must actually pass the probe through
    # unclamped, or this is a clamp test with nothing unclamped in it.
    assert_true(
        n_unclamped >= 3,
        "only " + String(n_unclamped) + " actuators pass ctrl=5.0 through"
        " unclamped in MuJoCo — the fixture no longer exercises the unlimited"
        " path",
    )
    assert_true(
        worst <= TOL,
        "qfrc_actuator differs from MuJoCo by " + String(worst)
        + " at ctrl=" + String(PROBE_CTRL) + ". ⚠ the historical failure was"
        " an UNCONDITIONAL clamp giving [1, 1, 3, 1, 1, 0] where MuJoCo gives"
        " [5, 1, 3, 5, 5, 5] — check WHICH dofs before assuming a gear or"
        " transmission problem",
    )
    print("  PASS  (worst |d(qfrc)| =", worst, ")")


def test_apply_actions_gpu_matches_cpu() raises:
    """The GPU mirror must clamp identically — it is the second clamp site.

    Gating only the CPU one would leave `Phyics3dBatchedEnv` applying a
    different force from `Phyics3dEnv` for the same action, silently. Same
    shape as defect #54, where a solver pass ran on one branch of a
    `[target]`-dispatched function and not the other.

    ⚠ FLOAT32 ON BOTH SIDES, not float64 like the legs above. Metal rejects a
    `double` outright ("instruction ... uses unsupported type 'double' /
    Failed to verify LLVM IR for Metal"), so the GPU actuator kernel can only
    be instantiated at float32 — and the CPU arm has to match it, or this
    would be a precision comparison wearing a clamp comparison's clothes.
    """
    print("=== ctrllimited: apply_actions_kernel_gpu vs CPU (float32) ===")
    comptime GT = DType.float32
    var sf = M.make_spec_fields[GT]()
    var ctx = DeviceContext()
    comptime B = 2
    comptime AD = M.nact

    var d32 = Data[GT, MD, 1]()
    M.reset_data[GT](sf, d32)
    var actions_c = List[Float64]()
    for _ in range(M.nact):
        actions_c.append(PROBE_CTRL)
    var act_c = List[Scalar[GT]]()
    for _ in range(M.NA_F):
        act_c.append(Scalar[GT](0))
    M.apply_actions[GT](sf, d32, actions_c, act_c)

    comptime L_QF = RowMajor[B, M.NV]
    comptime L_AC = RowMajor[B, AD]
    comptime L_QP = RowMajor[B, M.NQ]
    comptime L_QV = RowMajor[B, M.NV]
    comptime L_AT = RowMajor[B, M.NA_F]

    var t_qfrc = TensorImpl[GT].alloc_gpu(ctx, B * M.NV)
    var t_act = TensorImpl[GT].alloc_gpu(ctx, B * AD)
    var t_qpos = TensorImpl[GT].alloc_gpu(ctx, B * M.NQ)
    var t_qvel = TensorImpl[GT].alloc_gpu(ctx, B * M.NV)
    var t_a = TensorImpl[GT].alloc_gpu(ctx, B * _max1(M.NA_F))

    t_act.data = List[Scalar[GT]](length=B * AD, fill=Scalar[GT](0))
    for e in range(B):
        for i in range(AD):
            t_act.data[e * AD + i] = Scalar[GT](PROBE_CTRL)
    t_act.n = B * AD
    t_act.upload(ctx)

    var sfg = SpecFields[GT, MD]()
    M.init_spec_fields[GT](ctx, sfg)
    var t_actd = TensorImpl[GT]()
    t_actd.data = List[Scalar[GT]](length=B * M.NV, fill=Scalar[GT](0))
    t_actd.n = B * M.NV
    t_actd.upload(ctx)
    var t_aact = TensorImpl[GT]()
    t_aact.data = List[Scalar[GT]](
        length=B * M.NACT_F, fill=Scalar[GT](0)
    )
    t_aact.n = B * M.NACT_F
    t_aact.upload(ctx)
    var t_meta = TensorImpl[GT]()
    t_meta.data = List[Scalar[GT]](
        length=B * METADATA_SIZE, fill=Scalar[GT](0)
    )
    t_meta.n = B * METADATA_SIZE
    t_meta.upload(ctx)
    M.apply_actions_kernel_gpu[GT, B, AD](
        ctx,
        t_qfrc.tt["gpu", B, M.NV](),
        t_act.tt["gpu", B, AD](),
        t_qpos.tt["gpu", B, M.NQ](),
        t_qvel.tt["gpu", B, M.NV](),
        t_a.tt["gpu", B, M.NA_F](),
        sfg.actuators.tt["gpu", M.NACT_F * MODEL_ACTUATOR_SIZE](),
        sfg.act_tendons.tt["gpu", M.NTEN_F * MODEL_ACT_TENDON_SIZE](),
        # `jnt_actfrcrange` — MuJoCo's per-JOINT clamp on the accumulated
        # `qfrc_actuator`. This fixture declares none, so it is inert here;
        # it is passed because the kernel now requires it, and the model
        # that DOES exercise it is g1 in
        # `test_jnt_actfrcrange_vs_mujoco`.
        sfg.joint_limits.tt["gpu", M.NJOINT * JLIM_SIZE](),
        # `dof_actdamp` + `actdamp_act` + `meta` — this step's actuator
        # damping, diagonal and per-actuator gate. Inert for this fixture (no
        # forcerange saturation); passed because the kernel requires them and
        # the CPU/GPU pair must not diverge.
        t_actd.tt["gpu", B, M.NV](),
        t_aact.tt["gpu", B, M.NACT_F](),
        t_meta.tt["gpu", B, METADATA_SIZE](),
    )
    ctx.synchronize()
    t_qfrc.download(ctx)

    var worst = Float64(0)
    var moved = 0
    for e in range(B):
        for i in range(M.NV):
            var g = Float64(t_qfrc.data[e * M.NV + i])
            var c = Float64(d32.qfrc.data[i])
            if e == 0:
                print("   dof", i, " GPU", g, " CPU", c)
                if abs(c) > 1.5:
                    moved += 1
            var err = abs(g - c) / (1.0 + abs(c))
            if err > worst:
                worst = err
    # Vacuity: if every force were clamped to +-1 this would compare two
    # identically-wrong arrays and pass. At least the unlimited actuators must
    # be carrying the full 5.0.
    assert_true(
        moved >= 3,
        "only " + String(moved) + " dofs carry a force above 1.5 — the GPU"
        " arm is not exercising the unlimited path, so agreeing with the CPU"
        " arm proves nothing",
    )
    print("  worst rel |GPU - CPU| qfrc:", worst)
    assert_true(
        worst <= TOL32,
        "the GPU actuator path differs from the CPU one by " + String(worst)
        + " — the two clamp sites have diverged",
    )
    print("  PASS")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
