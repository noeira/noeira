"""`<velocity>` actuator: `apply_actions_kernel_gpu` against the CPU twin.

WHY A SEPARATE FILE FROM `test_velocity_actuator_vs_mujoco.mojo`: that one is
float64, and float64 is not runnable on Metal — a float64 "GPU test" does not
fail, it fails to BUILD A KERNEL. So the MuJoCo gate stays float64 and this one
is float32.

WHY IT EXISTS AT ALL: `apply_actions_kernel_gpu` is a comptime-unrolled MIRROR
of the CPU `apply_actions`, not a shared implementation — the POSITION/VELOCITY
branch had to be taught the same thing twice. An untested GPU branch is
uncompiled code, and Metal codegens lazily at pipeline-state creation, so a
green build says nothing about whether the kernel runs.

⚠ The GPU kernel guards its `qpos` load with `_kind == ACT_KIND_POSITION` so a
velocity actuator emits no dead read. That guard is the one place the two paths
are deliberately NOT term-for-term, which makes it the one worth gating: get it
inverted and the velocity law silently picks up a position term.

Fixtures mirror the MuJoCo gate: a velocity-only model, the `<general>`
spelling, and a mixed one that also exercises the POSITION branch beside it.

Run: pixi run mojo run -I . tests/physics3d/test_velocity_actuator_gpu_parity.mojo
"""

from std.math import abs
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext

from noeira.nn.core.tensor import TensorImpl
from noeira.physics3d.parser import parse_xml, ModelDefFromXML
from noeira.physics3d.types import ConeType
from noeira.physics3d.gpu.constants import (
    MODEL_ACTUATOR_SIZE,
    MODEL_ACT_TENDON_SIZE,
    JLIM_SIZE,
    METADATA_SIZE,
)
from noeira.physics3d.fields import Data, Model, SpecFields, Dims

comptime DTYPE = DType.float32

comptime _BODY = """
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <body name="link1" pos="0 0 0">
      <joint name="j1" type="hinge" axis="0 0 1" damping="0.1"/>
      <geom name="g1" type="capsule" fromto="0 0 0 0.4 0 0" size="0.04"/>
      <body name="link2" pos="0.4 0 0">
        <joint name="j2" type="hinge" axis="0 1 0" damping="0.1"/>
        <geom name="g2" type="capsule" fromto="0 0 0 0.3 0 0" size="0.03"/>
      </body>
    </body>
  </worldbody>
"""

comptime XML_VEL = (
    "<mujoco>"
    + _BODY
    + '<actuator>'
    + '<velocity ctrllimited="true" ctrlrange="-10 10" name="a1"'
    + ' joint="j1" kv="9.0"/>'
    + '<velocity ctrllimited="true" ctrlrange="-10 10" name="a2"'
    + ' joint="j2" kv="2.5" gear="1.7"/>'
    + '</actuator></mujoco>'
)

comptime XML_GEN = (
    "<mujoco>"
    + _BODY
    + '<actuator>'
    + '<general ctrllimited="true" ctrlrange="-10 10" name="a1" joint="j1"'
    + ' gaintype="fixed" biastype="affine" gainprm="6.0 0 0"'
    + ' biasprm="0 0 -9.0"/>'
    + '<general ctrllimited="true" ctrlrange="-10 10" name="a2" joint="j2"'
    + ' gaintype="fixed" biastype="affine" gainprm="2.0 0 0"'
    + ' biasprm="0 0 -0.5"/>'
    + '</actuator></mujoco>'
)

# VELOCITY beside POSITION: the comptime `_kind` guard is per-actuator, so a
# model with both is the one that catches a guard hoisted to the wrong scope.
comptime XML_MIX = (
    "<mujoco>"
    + _BODY
    + '<actuator>'
    + '<velocity ctrllimited="true" ctrlrange="-10 10" name="a1"'
    + ' joint="j1" kv="5.0"/>'
    + '<position ctrllimited="true" ctrlrange="-10 10" name="a2"'
    + ' joint="j2" kp="8.0" kv="0.3"/>'
    + '</actuator></mujoco>'
)

# ⚠ Jaco's shape: kv=500 against forcerange 30.5. The GPU kernel applies the
# clamp with a `comptime if motor_force_limited[act_i]`, which is a SEPARATE
# spelling from the CPU path's runtime `if` — the two can disagree silently.
# a1 saturates, a2 does not, so one fixture covers both sides of the branch.
comptime XML_FR = (
    "<mujoco>"
    + _BODY
    + '<actuator>'
    + '<velocity ctrllimited="true" ctrlrange="-5 5" forcelimited="true"'
    + ' forcerange="-30.5 30.5" name="a1" joint="j1" kv="500"/>'
    + '<velocity ctrllimited="true" ctrlrange="-5 5" name="a2" joint="j2"'
    + ' kv="2.5"/>'
    + '</actuator></mujoco>'
)

comptime p_vel = parse_xml(XML_VEL)
comptime p_fr = parse_xml(XML_FR)
comptime p_gen = parse_xml(XML_GEN)
comptime p_mix = parse_xml(XML_MIX)


def _m_vel() -> ModelDefFromXML[
    xml=XML_VEL, nbody=p_vel.NBODY, njoint=p_vel.NJOINT, nq=p_vel.NQ,
    nv=p_vel.NV, ngeom=p_vel.NGEOM, nact=p_vel.NACT, ntex=p_vel.NTEX,
    nmat=p_vel.NMAT, nlight=p_vel.NLIGHT, ncam=p_vel.NCAM, nsite=p_vel.NSITE,
    max_tendon=p_vel.NTENDON, cone_type=ConeType.PYRAMIDAL, max_contacts=8,
    max_condim=p_vel.MAX_CONDIM, nexclude=p_vel.NEXCLUDE, npair=p_vel.NPAIR,
    obs_dim_override=1, obs_qpos_skip=0, timestep=p_vel.TIMESTEP,
]:
    return {}


def _m_gen() -> ModelDefFromXML[
    xml=XML_GEN, nbody=p_gen.NBODY, njoint=p_gen.NJOINT, nq=p_gen.NQ,
    nv=p_gen.NV, ngeom=p_gen.NGEOM, nact=p_gen.NACT, ntex=p_gen.NTEX,
    nmat=p_gen.NMAT, nlight=p_gen.NLIGHT, ncam=p_gen.NCAM, nsite=p_gen.NSITE,
    max_tendon=p_gen.NTENDON, cone_type=ConeType.PYRAMIDAL, max_contacts=8,
    max_condim=p_gen.MAX_CONDIM, nexclude=p_gen.NEXCLUDE, npair=p_gen.NPAIR,
    obs_dim_override=1, obs_qpos_skip=0, timestep=p_gen.TIMESTEP,
]:
    return {}


def _m_mix() -> ModelDefFromXML[
    xml=XML_MIX, nbody=p_mix.NBODY, njoint=p_mix.NJOINT, nq=p_mix.NQ,
    nv=p_mix.NV, ngeom=p_mix.NGEOM, nact=p_mix.NACT, ntex=p_mix.NTEX,
    nmat=p_mix.NMAT, nlight=p_mix.NLIGHT, ncam=p_mix.NCAM, nsite=p_mix.NSITE,
    max_tendon=p_mix.NTENDON, cone_type=ConeType.PYRAMIDAL, max_contacts=8,
    max_condim=p_mix.MAX_CONDIM, nexclude=p_mix.NEXCLUDE, npair=p_mix.NPAIR,
    obs_dim_override=1, obs_qpos_skip=0, timestep=p_mix.TIMESTEP,
]:
    return {}


def _m_fr() -> ModelDefFromXML[
    xml=XML_FR, nbody=p_fr.NBODY, njoint=p_fr.NJOINT, nq=p_fr.NQ,
    nv=p_fr.NV, ngeom=p_fr.NGEOM, nact=p_fr.NACT, ntex=p_fr.NTEX,
    nmat=p_fr.NMAT, nlight=p_fr.NLIGHT, ncam=p_fr.NCAM, nsite=p_fr.NSITE,
    max_tendon=p_fr.NTENDON, cone_type=ConeType.PYRAMIDAL, max_contacts=8,
    max_condim=p_fr.MAX_CONDIM, nexclude=p_fr.NEXCLUDE, npair=p_fr.NPAIR,
    obs_dim_override=1, obs_qpos_skip=0, timestep=p_fr.TIMESTEP,
]:
    return {}


comptime M_VEL = _m_vel()
comptime M_FR = _m_fr()
comptime M_GEN = _m_gen()
comptime M_MIX = _m_mix()


def _gate[
    M: ModelDefFromXML
](
    label: String,
    q0: Float64,
    q1: Float64,
    v0: Float64,
    v1: Float64,
    c0: Float64,
    c1: Float64,
    mut failures: Int,
) raises:
    """Run both `apply_actions` paths on the same state and diff `qfrc`."""
    comptime MD = Dims[
        nq=M.NQ,
        nv=M.NV,
        nbody=M.NBODY,
        njoint=M.NJOINT,
        ngeom=M.NGEOM,
        nsite=M.NSITE,
        max_contacts=M.MAX_CONTACTS,
        nequality=M.MAX_EQUALITY,
        ntendon=M.MAX_TENDON,
        nexclude=M.NEXCLUDE,
        nmesh_verts=0,
        npair=M.NPAIR,
        nact=M.NACT,
        nten=M.NTEN_F,
        nkey=M.NKEY,
    ]
    var sf = M.make_spec_fields[DTYPE]()
    comptime BATCH = 1
    comptime NV = M.NV
    comptime NQ = M.NQ
    comptime NACT = M.nact
    comptime NA_F = M.NA_F

    var ctx = DeviceContext()

    # ---- CPU reference -----------------------------------------------------
    comptime Dat = Data[DTYPE, MD, 1]
    comptime Mod = Model[DTYPE, MD]
    var mf = Mod()
    M.init_fields[DTYPE](ctx, mf)
    var d = Dat()
    M.reset_data(sf, d)
    d.qpos.data[0] = Scalar[DTYPE](q0)
    d.qpos.data[1] = Scalar[DTYPE](q1)
    d.qvel.data[0] = Scalar[DTYPE](v0)
    d.qvel.data[1] = Scalar[DTYPE](v1)

    var actions = List[Float64]()
    actions.append(c0)
    actions.append(c1)
    var act_cpu = List[Scalar[DTYPE]]()
    M.apply_actions[DTYPE](sf, d, actions, act_cpu)

    # ---- GPU ---------------------------------------------------------------
    # Standalone [BATCH, ...] tensors rather than a batched env: the kernel's
    # ABI is exactly these five LayoutTensors, so this exercises it directly.
    # ⚠ `make["cpu"]`, not `make["gpu"]`: the GPU allocator creates the DEVICE
    # buffer only and leaves the host `data` list EMPTY, so filling it would
    # index out of bounds. `upload` below creates the device buffer from the
    # host side, which is the direction this test needs.
    var t_qfrc = TensorImpl[DTYPE].make["cpu"](BATCH * NV)
    var t_act_in = TensorImpl[DTYPE].make["cpu"](BATCH * NACT)
    var t_qpos = TensorImpl[DTYPE].make["cpu"](BATCH * NQ)
    var t_qvel = TensorImpl[DTYPE].make["cpu"](BATCH * NV)
    var t_actv = TensorImpl[DTYPE].make["cpu"](BATCH * NA_F)

    t_qpos.data[0] = Scalar[DTYPE](q0)
    t_qpos.data[1] = Scalar[DTYPE](q1)
    t_qvel.data[0] = Scalar[DTYPE](v0)
    t_qvel.data[1] = Scalar[DTYPE](v1)
    t_act_in.data[0] = Scalar[DTYPE](c0)
    t_act_in.data[1] = Scalar[DTYPE](c1)

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
        # `jnt_actfrcrange` — inert on this fixture, which declares none.
        # `test_jnt_actfrcrange_vs_mujoco` is where it is exercised.
        sfg.joint_limits.tt["gpu", M.NJOINT * JLIM_SIZE](),
        # Inert here — this fixture's `forcerange` case is exercised for the
        # FORCE, not the derivative. `test_saturated_actuator_deriv_vs_mujoco`
        # is where the derivative is gated.
        t_actd.tt["gpu", BATCH, NV](),
        t_aact.tt["gpu", BATCH, M.NACT_F](),
        t_meta.tt["gpu", BATCH, METADATA_SIZE](),
    )
    ctx.synchronize()
    t_qfrc.download(ctx)

    var worst = Float64(0)
    for i in range(NV):
        var diff = abs(
            Float64(d.qfrc.data[i]) - Float64(t_qfrc.data[i])
        )
        if diff > worst:
            worst = diff

    # ⚠ NOT bit-identical, and that is not a defect: the CPU path accumulates
    # the whole actuator law in Float64 (`motor_kp` and the qpos/qvel reads are
    # all widened) and rounds ONCE at the store, while the GPU kernel works in
    # `Scalar[DTYPE]` throughout. Measured residual here is 6e-08 / 2.4e-07 on
    # values of order 1-10, i.e. float32 ULPs. A bit-exact assert would be a
    # gate that fails for the wrong reason.
    var ok = worst < 1e-5
    print(
        "  ",
        label,
        " cpu qfrc[0]=",
        Float64(d.qfrc.data[0]),
        " gpu qfrc[0]=",
        Float64(t_qfrc.data[0]),
        " worst|d|=",
        worst,
        " -> ",
        "PASS" if ok else "FAIL",
    )
    if not ok:
        failures += 1


def test_velocity_actuator_gpu_parity() raises:
    print("=== <velocity> apply_actions: CPU vs GPU (float32) ===")
    var failures = 0
    # q != 0 so a stray length term would show up.
    _gate[M_VEL]("velocity_only", 0.7, -0.4, 0.3, -0.9, 1.5, -0.6, failures)
    _gate[M_GEN]("general_form ", 0.7, -0.4, 0.3, -0.9, 1.5, -0.6, failures)
    _gate[M_MIX]("mixed_pos_vel", 0.7, -0.4, 0.3, -0.9, 1.5, -0.6, failures)
    # a1 saturates (500*1.2 >> 30.5), a2 is unlimited — both sides of the
    # comptime clamp branch in one model.
    _gate[M_FR]("forcerange   ", 0.7, -0.4, 0.3, -0.9, 1.5, -0.6, failures)
    assert_true(
        failures == 0, String(failures) + " GPU-parity case(s) failed"
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
