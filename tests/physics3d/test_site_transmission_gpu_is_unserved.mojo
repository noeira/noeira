"""AUD-41 — a pose transmission cannot reach the GPU leg by accident.

    pixi run -e apple mojo run -I . tests/physics3d/test_site_transmission_gpu_is_unserved.mojo

`apply_actions_kernel_gpu` walks a transmission stored as `(qadr, dadr, coef)`
triples. A `<motor site=>` has none — its moment is `jac^T · (R_site · gear)`,
a function of the POSE — and neither does a `<motor tendon=>` naming a
SPATIAL tendon. Both leave `trn_n = 0`, so the kernel has nothing to walk and
the actuator would keep its slot in `nact`, consume its control and apply
ZERO FORCE. On the CPU leg `pose_transmission.apply_pose_transmission`
supplies it.

⚠⚠ THE AUDIT RECORDED THIS AS "BATCHED ENVS GET ZERO FORCE FROM THOSE
ACTUATORS", AND THAT IS ONLY HALF TRUE. `ModelDefFromXML` — the comptime
model the batched GPU env is built from — REFUSES `trn_n == 0` at build
(`model_def_from_xml.mojo:1278`). A pose transmission therefore cannot reach
the GPU path silently at all: it raises, by name, unless the caller passes
`allow_unsupported_actuators=True`, whose own contract is "this env's CONFIG
drives the DOF itself". The silent-zero reading of AUD-41 requires someone to
open that hatch and then not drive the dof. Test one below is that guard;
tests two and three are what is behind it.

WHO IS EXPOSED. Across `noeira/envs` and `noeira/tasks` — the only trees
that reach the batched GPU path — there is exactly one `<spatial>` tendon
(`ball_in_cup.xml`) and it is a LIMIT, not a transmission; no actuator has a
`site=`; none is `<adhesion>`. quadruped's eight tendon actuators drive
`<fixed>` tendons, which ARE walked as triples. Every model that needs this
one (tetheria's hands, skydio_x2, bitcraze_crazyflie_2, fruitfly) is a
Menagerie scene driven by the CPU studio or the validation sweep, where the
transmission is served in full.

⚠⚠ THE CONTROL ARM IS WHAT MAKES THE ZERO EVIDENCE. "The GPU qfrc is all
zero" is also exactly what a kernel that never launched looks like. The
fixture therefore carries a SECOND actuator, an ordinary `<motor joint=>` on
a child hinge, whose dof the kernel must drive correctly in the same call.
Dofs 0..5 are the free joint, reachable only through the site wrench; dof 6
is the hinge, reachable only through the joint motor. One launch, two
verdicts.

⚠ `gravity="0 0 0"`. `qfrc_actuator` is then the actuation alone, so a
nonzero reading cannot be weight leaking in.
"""

from std.math import abs
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext

from noeira.nn.core.tensor import TensorImpl
from noeira.physics3d.parser import parse_xml, ModelDefFromXML
from noeira.physics3d.types import ConeType
from noeira.physics3d.gpu.constants import (
    MODEL_ACTUATOR_SIZE,
    MODEL_ACT_TENDON_SIZE,
    METADATA_SIZE,
    JLIM_SIZE,
)
from noeira.physics3d.fields import Data, Model, SpecFields, Dims
from noeira.physics3d.fields.dynamics_scratch import DynamicsScratch
from noeira.physics3d.dynamics.pose_transmission import (
    apply_pose_transmission,
)

# ⚠ float32: this is a Metal kernel, and Metal has no `double`. The MuJoCo
# comparison below is loosened to match, and it is a CORROBORATION here — the
# exact CPU-vs-MuJoCo gate for this transmission is
# `test_site_transmission_vs_mujoco`, on skydio and crazyflie at float64.
comptime DTYPE = DType.float32

# Thrust off-axis and off-centre on purpose: a `gear` along a body axis
# through the centre of mass would produce zero torque, and then three of the
# six free dofs could not tell a served transmission from an unserved one.
comptime XML = """
<mujoco model="site transmission gpu">
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <body name="drone" pos="0 0 1">
      <freejoint name="root"/>
      <geom name="gb" type="box" size="0.1 0.1 0.02" density="500"/>
      <site name="thrust" pos="0.08 0.06 0.01" euler="0 12 25" size="0.01"/>
      <body name="arm" pos="0.1 0 0">
        <joint name="el" type="hinge" axis="0 1 0"/>
        <geom name="ga" type="capsule" fromto="0 0 0 0.15 0 0" size="0.02"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="thr" site="thrust" gear="0 0 1 0 0 -0.0201"/>
    <motor name="mel" joint="el" gear="2"/>
  </actuator>
</mujoco>
"""

comptime p = parse_xml(XML)


def _mk_strict() -> ModelDefFromXML[
    xml=XML, nbody=p.NBODY, njoint=p.NJOINT, nq=p.NQ, nv=p.NV,
    ngeom=p.NGEOM, nact=p.NACT, ntex=p.NTEX, nmat=p.NMAT, nlight=p.NLIGHT,
    ncam=p.NCAM, nsite=p.NSITE, max_tendon=p.NTENDON,
    cone_type=ConeType.PYRAMIDAL, max_contacts=8, max_condim=p.MAX_CONDIM,
    nexclude=p.NEXCLUDE, npair=p.NPAIR, obs_dim_override=1, obs_qpos_skip=0,
    timestep=p.TIMESTEP,
]:
    """The default. `allow_unsupported_actuators` is False, so this one
    refuses the site actuator — see test one."""
    return {}


def _mk_open() -> ModelDefFromXML[
    xml=XML, nbody=p.NBODY, njoint=p.NJOINT, nq=p.NQ, nv=p.NV,
    ngeom=p.NGEOM, nact=p.NACT, ntex=p.NTEX, nmat=p.NMAT, nlight=p.NLIGHT,
    ncam=p.NCAM, nsite=p.NSITE, max_tendon=p.NTENDON,
    cone_type=ConeType.PYRAMIDAL, max_contacts=8, max_condim=p.MAX_CONDIM,
    nexclude=p.NEXCLUDE, npair=p.NPAIR, obs_dim_override=1, obs_qpos_skip=0,
    timestep=p.TIMESTEP, allow_unsupported_actuators=True,
]:
    """The hatch, open. ⚠ ONLY DIFFERENCE FROM `_mk_strict` IS THAT FLAG, and
    the two are kept adjacent so a drift between them is visible."""
    return {}


comptime MS = _mk_strict()
comptime M = _mk_open()
comptime MD = Dims[
    nq=M.NQ, nv=M.NV, nbody=M.NBODY, njoint=M.NJOINT, ngeom=M.NGEOM,
    nsite=M.NSITE, max_contacts=M.MAX_CONTACTS, nequality=M.MAX_EQUALITY,
    ntendon=M.MAX_TENDON, nexclude=M.NEXCLUDE, nmesh_verts=0, npair=M.NPAIR,
    nact=M.NACT, nten=M.NTEN_F, nkey=M.NKEY,
]

comptime BATCH = 1
comptime NV = M.NV
comptime NQ = M.NQ
comptime NACT = M.nact
comptime NA_F = M.NA_F

# The two controls. Nonzero and different, so a swapped actuator index shows.
comptime C_THRUST: Float64 = 3.4
comptime C_ELBOW: Float64 = 0.7

# dofs 0..5 are the free joint (site wrench only); dof 6 is the hinge (joint
# motor only). The split is the whole design of the fixture.
comptime N_FREE_DOF = 6
comptime DOF_ELBOW = 6


def _cpu_qfrc() raises -> List[Float64]:
    """`apply_actions` + `apply_pose_transmission`, i.e. the served path."""
    var ctx = DeviceContext()
    var sf = M.make_spec_fields[DTYPE]()
    var mf = Model[DTYPE, MD]()
    M.init_fields[DTYPE](ctx, mf)
    var d = Data[DTYPE, MD, 1]()
    M.reset_data(sf, d)
    var sc = DynamicsScratch[DTYPE, MD, 1]()

    var actions = List[Float64]()
    actions.append(C_THRUST)
    actions.append(C_ELBOW)
    var act = List[Scalar[DTYPE]]()
    M.apply_actions[DTYPE](sf, d, actions, act)
    apply_pose_transmission[DTYPE](
        sf, mf, d, sc, actions, act, M.TIMESTEP
    )

    var out = List[Float64]()
    for i in range(NV):
        out.append(Float64(d.qfrc.data[i]))
    return out^


def _gpu_qfrc() raises -> List[Float64]:
    """`apply_actions_kernel_gpu` alone — the batched path's whole actuation.

    Standalone `[BATCH, ...]` tensors rather than a batched env: the kernel's
    ABI is exactly these LayoutTensors, so this exercises it directly. The
    `make["cpu"]` + `upload` idiom is the one `test_velocity_actuator_gpu_
    parity` documents — a `make["gpu"]` tensor has an EMPTY host list.
    """
    var ctx = DeviceContext()

    var t_qfrc = TensorImpl[DTYPE].make["cpu"](BATCH * NV)
    var t_act_in = TensorImpl[DTYPE].make["cpu"](BATCH * NACT)
    var t_qpos = TensorImpl[DTYPE].make["cpu"](BATCH * NQ)
    var t_qvel = TensorImpl[DTYPE].make["cpu"](BATCH * NV)
    var t_actv = TensorImpl[DTYPE].make["cpu"](BATCH * NA_F)

    # ⚠ THE SAME POSE THE CPU LEG IS AT. `reset_data` puts the free joint at
    # `pos="0 0 1"` with the identity quaternion; a GPU leg started from a
    # zeroed `qpos` would have quat (0,0,0,0) and the comparison would be
    # about the pose rather than about the transmission.
    t_qpos.data[2] = Scalar[DTYPE](1.0)
    t_qpos.data[3] = Scalar[DTYPE](1.0)
    t_act_in.data[0] = Scalar[DTYPE](C_THRUST)
    t_act_in.data[1] = Scalar[DTYPE](C_ELBOW)

    t_qfrc.upload(ctx)
    t_act_in.upload(ctx)
    t_qpos.upload(ctx)
    t_qvel.upload(ctx)
    t_actv.upload(ctx)

    var t_actd = TensorImpl[DTYPE]()
    t_actd.data = List[Scalar[DTYPE]](length=BATCH * NV, fill=Scalar[DTYPE](0))
    t_actd.n = BATCH * NV
    t_actd.upload(ctx)
    var t_aact = TensorImpl[DTYPE]()
    t_aact.data = List[Scalar[DTYPE]](
        length=BATCH * M.NACT_F, fill=Scalar[DTYPE](0)
    )
    t_aact.n = BATCH * M.NACT_F
    t_aact.upload(ctx)
    var t_meta = TensorImpl[DTYPE]()
    t_meta.data = List[Scalar[DTYPE]](
        length=BATCH * METADATA_SIZE, fill=Scalar[DTYPE](0)
    )
    t_meta.n = BATCH * METADATA_SIZE
    t_meta.upload(ctx)

    var sfg = SpecFields[DTYPE, MD]()
    M.init_spec_fields[DTYPE](ctx, sfg)
    M.apply_actions_kernel_gpu[DTYPE, BATCH, NACT](
        ctx,
        t_qfrc.tt["gpu", BATCH, NV](),
        t_act_in.tt["gpu", BATCH, NACT](),
        t_qpos.tt["gpu", BATCH, NQ](),
        t_qvel.tt["gpu", BATCH, NV](),
        t_actv.tt["gpu", BATCH, NA_F](),
        sfg.actuators.tt["gpu", M.NACT_F * MODEL_ACTUATOR_SIZE](),
        sfg.act_tendons.tt["gpu", M.NTEN_F * MODEL_ACT_TENDON_SIZE](),
        sfg.joint_limits.tt["gpu", M.NJOINT * JLIM_SIZE](),
        t_actd.tt["gpu", BATCH, NV](),
        t_aact.tt["gpu", BATCH, M.NACT_F](),
        t_meta.tt["gpu", BATCH, METADATA_SIZE](),
    )
    ctx.synchronize()
    t_qfrc.download(ctx)

    var out = List[Float64]()
    for i in range(NV):
        out.append(Float64(t_qfrc.data[i]))
    return out^


def _mj_qfrc() raises -> List[Float64]:
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(PythonObject(String(XML)))
    var dat = mujoco.MjData(m)
    dat.ctrl[0] = C_THRUST
    dat.ctrl[1] = C_ELBOW
    mujoco.mj_forward(m, dat)
    var out = List[Float64]()
    for i in range(NV):
        out.append(Float64(py=dat.qfrc_actuator[i]))
    return out^


def test_a_pose_transmission_refuses_to_build_a_comptime_model() raises:
    """⚠ RUN FIRST — THIS IS THE GUARD, and everything after it is behind it.

    `ModelDefFromXML` rejects `trn_n == 0` by name rather than building an
    actuator against a garbage index. So the batched GPU path cannot acquire
    a silent dead actuator by loading a model; it acquires one only if a
    caller opens `allow_unsupported_actuators`, whose documented meaning is
    "this env's CONFIG drives the DOF itself".

    Both arms are asserted: the strict model must raise, and the open one
    must NOT. Without the second arm this would pass on a build that refused
    every model.
    """
    print("=== a site transmission refuses the default comptime model ===")
    # ⚠ `init_fields`, NOT `make_spec_fields`. The refusal lives in the
    # field builder (`model_def_from_xml.mojo:1278`), which is where the
    # resolved transmission list exists; `make_spec_fields` returns happily
    # and was the first thing this test called, so it read "no refusal".
    var ctx = DeviceContext()
    var raised = False
    var msg = String("")
    try:
        var mf_strict = Model[DTYPE, MD]()
        MS.init_fields[DTYPE](ctx, mf_strict)
    except e:
        raised = True
        msg = String(e)
    print("  strict model raised:", raised)
    if raised:
        print("  message:", msg)
    assert_true(
        raised,
        "the default `ModelDefFromXML` built a model whose actuator 0 has no"
        " resolvable transmission. If site transmissions are modelled on this"
        " path now, this whole file is stale — see its header",
    )
    assert_true(
        msg.find("no resolvable transmission") >= 0,
        "it raised, but not for the transmission: " + msg,
    )

    # The other arm: the hatch actually opens, on the SAME call.
    var mf_open = Model[DTYPE, MD]()
    M.init_fields[DTYPE](ctx, mf_open)
    var sf_open = M.make_spec_fields[DTYPE]()
    print("  open model built,", sf_open.actuators.n, "actuator words")
    assert_true(
        sf_open.actuators.n > 0,
        "`allow_unsupported_actuators=True` produced an empty actuator"
        " table — the rest of this file would be testing nothing",
    )


def test_the_cpu_leg_serves_the_site_wrench() raises:
    """The served leg, against MuJoCo, so the numbers below are MuJoCo's.

    Without this the file would only be asserting that two of our own paths
    disagree, which does not say which one is right.
    """
    print("=== the CPU leg's site wrench is MuJoCo's ===")
    var mujoco = Python.import_module("mujoco")
    print("  mujoco", String(mujoco.__version__))
    var ours = _cpu_qfrc()
    var theirs = _mj_qfrc()
    var moving = 0
    var worst = 0.0
    for i in range(NV):
        var scale = abs(theirs[i])
        if scale < 1.0:
            scale = 1.0
        var rel = abs(ours[i] - theirs[i]) / scale
        if rel > worst:
            worst = rel
        if abs(theirs[i]) > 1e-6:
            moving += 1
        print("  dof", i, " ours =", ours[i], " MuJoCo =", theirs[i])
        assert_true(
            rel <= 2e-6,
            "dof " + String(i) + ": ours " + String(ours[i]) + " vs MuJoCo "
            + String(theirs[i]) + " (float32 leg, relative " + String(rel)
            + ")",
        )
    print("  worst relative |d| =", worst, " dofs MuJoCo drives:", moving)
    assert_true(
        moving >= 4,
        "MuJoCo drives only " + String(moving) + " of " + String(NV)
        + " dofs — the fixture has gone quiet and cannot separate a served"
        " transmission from an unserved one",
    )


def test_the_gpu_kernel_did_launch_and_drove_the_joint_actuator() raises:
    """⚠ RUN BEFORE THE ZERO TEST. An unlaunched kernel leaves `qfrc` at the
    zeros it was uploaded with, which is indistinguishable from AUD-41's
    symptom on every dof. The hinge dof is the discriminator: the joint motor
    IS served on the GPU leg, so dof 6 must carry `gear * ctrl`.
    """
    print("=== the kernel ran: the joint actuator's dof is driven ===")
    var gpu = _gpu_qfrc()
    var cpu = _cpu_qfrc()
    var want = 2.0 * C_ELBOW  # gear="2"
    print(
        "  dof", DOF_ELBOW, " gpu =", gpu[DOF_ELBOW], " cpu =",
        cpu[DOF_ELBOW], " gear*ctrl =", want,
    )
    assert_true(
        abs(gpu[DOF_ELBOW] - want) <= 1e-5,
        "dof " + String(DOF_ELBOW) + " reads " + String(gpu[DOF_ELBOW])
        + " where the joint motor alone gives " + String(want)
        + " — the kernel did not run, and every zero below is meaningless",
    )
    assert_true(
        abs(gpu[DOF_ELBOW] - cpu[DOF_ELBOW]) <= 1e-5,
        "the two legs disagree on the JOINT dof (" + String(gpu[DOF_ELBOW])
        + " vs " + String(cpu[DOF_ELBOW]) + "), which is not AUD-41 — that"
        " transmission is served on both",
    )


def test_the_gpu_kernel_leaves_the_site_actuator_at_zero() raises:
    """AUD-41, pinned. ⚠ THIS TEST FLIPS WHEN THE FIX LANDS.

    Every free dof is zero on the GPU leg and carries MuJoCo's wrench on the
    CPU one. When `apply_actions_kernel_gpu` grows the site transmission,
    this file should assert equality on all seven dofs instead — the numbers
    it needs are printed here.
    """
    print("=== AUD-41: the site wrench is absent on the GPU leg ===")
    var gpu = _gpu_qfrc()
    var cpu = _cpu_qfrc()
    var lost = 0.0
    for i in range(N_FREE_DOF):
        print(
            "  dof", i, " gpu =", gpu[i], "  cpu (served) =", cpu[i],
        )
        assert_true(
            gpu[i] == 0.0,
            "dof " + String(i) + " reads " + String(gpu[i]) + " on the GPU"
            " leg. If the site transmission now IS served there, this file's"
            " claim is stale: assert equality with the CPU leg on all "
            + String(NV) + " dofs and strike AUD-41 from the audit",
        )
        if abs(cpu[i]) > lost:
            lost = abs(cpu[i])
    assert_true(
        lost > 1e-3,
        "the CPU leg's largest free-dof force is only " + String(lost)
        + " — too small for the zero above to mean anything",
    )
    print(
        "  the batched GPU path loses up to", lost,
        "of generalized force on this fixture",
    )


def main() raises:
    var suite = TestSuite()
    suite.test[test_a_pose_transmission_refuses_to_build_a_comptime_model]()
    suite.test[test_the_cpu_leg_serves_the_site_wrench]()
    suite.test[test_the_gpu_kernel_did_launch_and_drove_the_joint_actuator]()
    suite.test[test_the_gpu_kernel_leaves_the_site_actuator_at_zero]()
    suite^.run()
