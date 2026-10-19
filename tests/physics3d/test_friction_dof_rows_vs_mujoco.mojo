"""Dry-friction dof rows (`mjCNSTR_FRICTION_DOF`) vs MuJoCo, on BOTH pyramidal
solver paths.

WHY THIS FILE EXISTS. `frictionloss` became a constraint ROW in 04a7c508 (an
explicit Coulomb force cannot arrest motion — it overshoots zero and settles
into a period-2 limit cycle). Those rows were added to the per-env pyramidal
path and to the elliptic core, and on 2026-07-31 to the cooperative BLOCKED
kernel. But NO SHIPPED MODEL COULD EXERCISE THEM ON A PYRAMIDAL CONE: the only
model in the repo that sets `frictionloss` is dm_control's finger, and finger
is ELLIPTIC. So the pyramidal friction rows — in both solvers — were written
and never gated.

That is the exact shape of every silent bug this engine has produced (20, 21,
25, 26): code whose defect the test set cannot express. Hence a PURPOSE-BUILT
model rather than waiting for a domain that happens to need one.

THE MODEL is the smallest thing that expresses the coupling: a SPHERE on two
slides (x with `frictionloss`, z free) dropped onto a plane. Started AIRBORNE
so the run contains a friction-only phase as well as the coupled one.

⚠ SPHERE, not box, and that is load-bearing. The first version used a box and
the coupled bucket came out at 0.0234 — which looks exactly like a
friction/contact coupling bug. It is not: re-running with `frictionloss="0"`,
so that NO friction row exists at all, gives 0.02337063116882 against
0.02337063116880 with friction. Identical to 13 digits. The whole residual is
a pre-existing BOX-vs-PLANE contact gap and has nothing to do with friction.
Sphere-vs-plane is exact (8.9e-16), so a sphere isolates the row under test.
The box gap is logged separately; do not "fix" it here. Characterised: for a
box resting on a plane MuJoCo emits FOUR contacts (the bottom corners) while
our narrow phase emitted ONE, so the pressure distribution cannot match. See
[box-vs-plane contact count] in physics3d/GPU_PYRAMIDAL_TODO.md.

⚠ A friction row is UNCONDITIONAL for any dof with `frictionloss > 0` —
`mj_instantiateFriction` does not test for sliding — so there is no
"contact only" or "neither" bucket to be had here. The split that matters, and
the one asserted below, is:

    friction only     box in flight   — isolates the row
    friction+contact  box on ground   — the coupled regime, shared dofs

Part A  per-env pyramidal path vs MuJoCo, resynced every control step so the
        error is one step rather than accumulated drift.
Part B  the BLOCKED cooperative kernel: GPU vs CPU, and blocked vs per-env.
        float32, because Metal rejects `double` in kernels.

Run with:
    pixi run mojo run -I . tests/physics3d/test_friction_dof_rows_vs_mujoco.mojo
"""

from std.random import random_float64
from std.python import Python, PythonObject
from std.testing import assert_true, TestSuite
from std.math import abs, sin
from std.sys import default_accelerator
from max.gpu.host import DeviceContext
from layout import Layout

from noeira.nn.core.tensor import TensorImpl
from noeira.physics3d.fields import (
    AsStatic,
    Data,
    Model,
    DynamicsScratch,
    ContactScratch,
    Dims,
 DimsLike,)
from noeira.physics3d.types import ConeType
from noeira.physics3d.solver.je_budget import je_ws_size
from noeira.physics3d.joint_types import JNT_HINGE, JNT_SLIDE
from noeira.physics3d.integrator.euler import (
    _armature_kernel,
    _fnet_passive_kernel,
    _qacc_writeback_kernel,
    _armature_env,
    _fnet_passive_env,
    _qacc_writeback_env,
)
from noeira.physics3d.kinematics.forward_kinematics import (
    forward_kinematics,
    compute_body_velocities,
)
from noeira.physics3d.dynamics.subtree_com import (
    compute_subtree_com,
)
from noeira.physics3d.dynamics.cdof import compute_cdof
from noeira.physics3d.dynamics.mass_matrix import (
    compute_mass_matrix,
)
from noeira.physics3d.dynamics.ldl import (
    ldl_factor,
    ldl_solve,
    compute_m_inv,
)
from noeira.physics3d.dynamics.rne import (
    compute_bias_forces_rne,
)
from noeira.physics3d.collision.contact_detection import (
    detect_contacts,
)
from noeira.physics3d.solver.newton_solve import (
    solve_newton,
    solve_newton_blocked,
)
from noeira.physics3d.gpu.constants import (
    META_IDX_NUM_CONTACTS,
    METADATA_SIZE,
    CONTACT_SIZE,
    MODEL_BODY_SIZE,
    MODEL_JOINT_SIZE,
    JOINT_IDX_FRICTIONLOSS,
    JOINT_IDX_TYPE,
    JOINT_IDX_QPOS_ADR,
    JOINT_IDX_RANGE_MIN,
    JOINT_IDX_RANGE_MAX,
)


from noeira.physics3d.parser import parse_xml, ModelDefFromXML
from noeira.envs.phyics3d_env import Phyics3dEnv
from noeira.envs.phyics3d_env_config import Phyics3dEnvConfig
from noeira.core.cont_action import ContAction
from noeira.physics3d.model.model_dims import ModelDims


comptime FRIC_XML = """
<mujoco model="fric slider">
  <option cone="pyramidal" timestep="0.002"/>
  <worldbody>
    <geom name="ground" type="plane" pos="0 0 0" size="2 2 1"/>
    <body name="box" pos="0 0 .05">
      <joint name="sx" type="slide" axis="1 0 0" frictionloss="0.5" damping="0.01"/>
      <joint name="sz" type="slide" axis="0 0 1"/>
      <geom name="gb" type="sphere" size=".05"/>
    </body>
  </worldbody>
  <actuator><motor name="mx" joint="sx" gear="3" ctrllimited="true" ctrlrange="-1 1"/></actuator>
</mujoco>
"""

comptime fp = parse_xml(FRIC_XML)

comptime FricModel = ModelDefFromXML[
    xml=FRIC_XML,
    nbody=fp.NBODY, njoint=fp.NJOINT, nq=fp.NQ, nv=fp.NV,
    ngeom=fp.NGEOM, nact=fp.NACT, ntex=fp.NTEX, nmat=fp.NMAT,
    nlight=fp.NLIGHT, ncam=fp.NCAM, nsite=fp.NSITE,
    cone_type=ConeType.PYRAMIDAL,
    max_contacts=8,
    obs_dim_override=4,
    obs_qpos_skip=0,
    timestep=fp.TIMESTEP,
]


struct FricCfg(Phyics3dEnvConfig):
    comptime FRAME_SKIP: Int = 1
    comptime MAX_STEPS: Int = 1000
    comptime INTEGRATOR_WS_EXTRA: Int = 0
    comptime SYNC_FK_AFTER_STEP: Bool = True
    comptime INTEGRATOR: StaticString = "euler"

    @staticmethod
    def custom_extract_obs_cpu[DTYPE: DType, D: DimsLike](
        d: Data[DTYPE, D, 1],
        m_bodies: List[Scalar[DTYPE]], m_joints: List[Scalar[DTYPE]],
        m_geoms: List[Scalar[DTYPE]], m_sites: List[Scalar[DTYPE]],
        mut obs: List[Scalar[DTYPE]],
    ) -> Bool:
        for i in range(D.NQ):
            obs.append(d.qpos.data[i])
        for i in range(D.NV):
            obs.append(d.qvel.data[i])
        return True

    @staticmethod
    def compute_reward_and_done_cpu[DTYPE: DType, D: DimsLike](
        d: Data[DTYPE, D, 1],
        m_bodies: List[Scalar[DTYPE]], m_joints: List[Scalar[DTYPE]],
        m_geoms: List[Scalar[DTYPE]], m_sites: List[Scalar[DTYPE]],
        prev_x: Scalar[DTYPE], actions: List[Float64], step_count: Int,
        frame_skip: Int,
    ) -> Tuple[Scalar[DTYPE], Bool]:
        return (Scalar[DTYPE](0), False)

    @staticmethod
    def get_timestep() -> Float64:
        return Float64(FricModel.TIMESTEP)


comptime FricEnv = Phyics3dEnv[FricModel, FricCfg, DType.float64, False]

comptime N_STEPS: Int = 600
comptime START_Z: Float64 = 0.40
# Both buckets measure at machine precision; 1e-13 is a gate with headroom.
comptime TOL: Float64 = 1e-13


def test_friction_rows_vs_mujoco() raises:
    """Part A — per-env pyramidal path vs MuJoCo, split by regime."""
    var sys_m = Python.import_module("sys")
    var mujoco = Python.import_module("mujoco")
    var model = mujoco.MjModel.from_xml_string(String(FRIC_XML))
    var data = mujoco.MjData(model)

    # The row must exist on MuJoCo's side, and our parser must have kept the
    # attribute — otherwise this whole file is vacuous.
    assert_true(
        Float64(py=model.dof_frictionloss[0]) > 0.0,
        "MuJoCo has no dof_frictionloss — model lost the attribute",
    )
    assert_true(Int(py=model.opt.cone) == 0, "model is not PYRAMIDAL")

    var env = FricEnv()
    _ = env.reset()

    mujoco.mj_resetData(model, data)
    data.qpos[0] = 0.0
    data.qpos[1] = START_Z
    mujoco.mj_forward(model, data)

    var n_fric = 0
    var n_both = 0
    var w_fric = 0.0
    var w_both = 0.0

    for step in range(N_STEPS):
        # Resync: measure ONE step, not accumulated drift.
        var qs = List[Float64]()
        var vs = List[Float64]()
        for i in range(2):
            qs.append(Float64(py=data.qpos[i]))
        for i in range(2):
            vs.append(Float64(py=data.qvel[i]))
        env.set_state(qs, vs)

        var a = ContAction[FricModel.ACTION_DIM]()
        var c = 0.9 * sin(0.02 * Float64(step))
        a.data[0] = c
        data.ctrl[0] = c

        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        _ = env.step(a)

        var nefc = Int(py=data.nefc)
        var has_fric = False
        for e in range(nefc):
            if Int(py=data.efc_type[e]) == 1:  # mjCNSTR_FRICTION_DOF
                has_fric = True
                break
        var has_con = Int(py=data.ncon) > 0

        var worst = 0.0
        for i in range(2):
            var dq = abs(Float64(py=data.qpos[i]) - Float64(env.d.qpos.data[i]))
            if dq > worst:
                worst = dq
            var dv = abs(Float64(py=data.qvel[i]) - Float64(env.d.qvel.data[i]))
            if dv > worst:
                worst = dv

        if has_fric and has_con:
            n_both += 1
            if worst > w_both:
                w_both = worst
        elif has_fric:
            n_fric += 1
            if worst > w_fric:
                w_fric = worst

    print("friction-dof rows vs MuJoCo (resynced, per-env pyramidal path):")
    print("  friction only   :", n_fric, " worst |d(state)| =", w_fric)
    print("  friction+contact:", n_both, " worst |d(state)| =", w_both)

    assert_true(
        n_fric > 0,
        "no friction-only substep — the row is never isolated, so a coupling"
        " bug and a row bug would be indistinguishable",
    )
    assert_true(
        n_both > 0,
        "no substep had a friction row and a contact live at once — this"
        " rollout no longer exercises the coupling it was built for",
    )
    assert_true(w_fric <= TOL, "friction-only substeps diverge from MuJoCo")
    assert_true(
        w_both <= TOL, "coupled friction+contact substeps diverge from MuJoCo"
    )


# ---------------------------------------------------------------------------
# Part B — the BLOCKED cooperative kernel carries the same rows.
# float32: Metal rejects `double` in kernels, and keeping this runnable on
# Apple is the only local coverage of the cooperative path with friction rows.
# ---------------------------------------------------------------------------

comptime DTYPE = DType.float32
comptime NQ = FricModel.NQ
comptime NV = FricModel.NV
comptime NBODY = FricModel.NBODY
comptime NJOINT = FricModel.NJOINT
comptime NGEOM = FricModel.NGEOM
comptime MC = FricModel.MAX_CONTACTS
comptime NEQ = FricModel.MAX_EQUALITY
comptime NTD = FricModel.MAX_TENDON
comptime NSITE = FricModel.NSITE
comptime NEXCL = FricModel.NEXCLUDE
comptime MD = ModelDims[FricModel]
comptime BATCH = 2
comptime REL_TOL: Float64 = 1e-4

def _fields_prep[
    target: StaticString
](
    mut d: Data[DTYPE, MD, BATCH],
    mut mf: Model[DTYPE, MD],
    mut scratch: DynamicsScratch[DTYPE, MD, BATCH],
    ctx: Optional[DeviceContext],
) raises:
    """Smooth-dynamics prep + detection, mirroring EulerIntegrator.step
    up to the constraint seam (order verbatim)."""
    forward_kinematics[target, DTYPE, BATCH=BATCH](d, mf, ctx)
    compute_body_velocities[target, DTYPE, BATCH=BATCH](d, mf, ctx)
    compute_subtree_com[target, DTYPE, BATCH=BATCH](d, mf, ctx)
    compute_cdof[target, DTYPE, BATCH=BATCH](d, mf, scratch, ctx)
    compute_mass_matrix[target, DTYPE, BATCH=BATCH](d, mf, scratch, ctx)

    comptime L_JOINT = Layout.row_major(NJOINT, MODEL_JOINT_SIZE)
    comptime L_M = Layout.row_major(BATCH, NV * NV)
    comptime L_NV = Layout.row_major(BATCH, NV)
    comptime L_QPOS = Layout.row_major(BATCH, NQ)

    comptime if target == "cpu":
        var joints_v = mf.joints.lt["cpu", L_JOINT]()
        var M_v = scratch.M.lt["cpu", L_M]()
        for e in range(BATCH):
            _armature_env[DTYPE](e, AsStatic[MD](), joints_v, M_v)
        ldl_factor[target, DTYPE, BATCH=BATCH](mf, scratch, ctx)
        compute_m_inv[target, DTYPE, BATCH=BATCH](mf, scratch, ctx)
        compute_bias_forces_rne[target, DTYPE, BATCH=BATCH](d, mf, scratch, ctx)
        var qpos_v = d.qpos.lt["cpu", L_QPOS]()
        var qvel_v = d.qvel.lt["cpu", L_NV]()
        var qfrc_v = d.qfrc.lt["cpu", L_NV]()
        var bias_v = scratch.bias.lt["cpu", L_NV]()
        var fnet_v = scratch.fnet.lt["cpu", L_NV]()
        for e in range(BATCH):
            _fnet_passive_env[DTYPE](
                e, AsStatic[MD](), qpos_v, qvel_v, qfrc_v, joints_v, bias_v, fnet_v
            )
        ldl_solve[target, DTYPE, BATCH=BATCH](mf, scratch, ctx)
        var qacc_ws_v = scratch.qacc_ws.lt["cpu", L_NV]()
        var qacc_v = d.qacc.lt["cpu", L_NV]()
        var qacc_c_v = scratch.qacc_constrained.lt["cpu", L_NV]()
        for e in range(BATCH):
            _qacc_writeback_env[DTYPE](
                e, AsStatic[MD](), qacc_ws_v, qacc_v, qacc_c_v
            )
    else:
        ctx.value().enqueue_function[
            _armature_kernel[DTYPE, NV, NJOINT, BATCH]
        ](
            mf.joints.lt["gpu", L_JOINT](),
            scratch.M.lt["gpu", L_M](),
            grid_dim=(BATCH,),
            block_dim=(1,),
        )
        ldl_factor[target, DTYPE, BATCH=BATCH](mf, scratch, ctx)
        compute_m_inv[target, DTYPE, BATCH=BATCH](mf, scratch, ctx)
        compute_bias_forces_rne[target, DTYPE, BATCH=BATCH](d, mf, scratch, ctx)
        ctx.value().enqueue_function[
            _fnet_passive_kernel[DTYPE, NQ, NV, NJOINT, BATCH]
        ](
            d.qpos.lt["gpu", L_QPOS](),
            d.qvel.lt["gpu", L_NV](),
            d.qfrc.lt["gpu", L_NV](),
            mf.joints.lt["gpu", L_JOINT](),
            scratch.bias.lt["gpu", L_NV](),
            scratch.fnet.lt["gpu", L_NV](),
            grid_dim=(BATCH,),
            block_dim=(1,),
        )
        ldl_solve[target, DTYPE, BATCH=BATCH](mf, scratch, ctx)
        ctx.value().enqueue_function[
            _qacc_writeback_kernel[DTYPE, NV, BATCH]
        ](
            scratch.qacc_ws.lt["gpu", L_NV](),
            d.qacc.lt["gpu", L_NV](),
            scratch.qacc_constrained.lt["gpu", L_NV](),
            grid_dim=(BATCH,),
            block_dim=(1,),
        )

    detect_contacts[target, DTYPE, BATCH=BATCH](d, mf, ctx)



def _seed_fric(mut d: Data[DTYPE, MD, BATCH]):
    """Box resting ON the ground with a lateral velocity, so the friction row
    and the contact are BOTH live — the regime the blocked kernel never saw."""
    for e in range(BATCH):
        d.qpos.data[e * NQ + 0] = Scalar[DTYPE](0.1 * Float64(e))
        d.qpos.data[e * NQ + 1] = Scalar[DTYPE](-0.004)  # slight penetration
        d.qvel.data[e * NV + 0] = Scalar[DTYPE](0.6 + 0.2 * Float64(e))
        d.qvel.data[e * NV + 1] = Scalar[DTYPE](-0.1)
        d.qfrc.data[e * NV + 0] = Scalar[DTYPE](0.3)
        d.qfrc.data[e * NV + 1] = Scalar[DTYPE](0.0)


def test_blocked_friction_rows() raises:
    """Blocked GPU vs blocked CPU vs per-env, all carrying friction rows."""
    var ctx = DeviceContext()
    var mf = Model[DTYPE, MD]()
    FricModel.init_fields[DTYPE](ctx, mf)

    # Non-vacuity: our model must actually carry the frictionloss.
    var floss_seen = Float64(0)
    for j in range(NJOINT):
        var f = Float64(
            mf.joints.data[j * MODEL_JOINT_SIZE + JOINT_IDX_FRICTIONLOSS]
        )
        if f > floss_seen:
            floss_seen = f
    print("  parsed frictionloss:", floss_seen)
    if floss_seen <= 0.0:
        raise Error(
            "our model has NO frictionloss — the friction rows are vacuous"
        )

    var dg = Data[DTYPE, MD, BATCH]()
    var dc = Data[DTYPE, MD, BATCH]()
    var dp = Data[DTYPE, MD, BATCH]()
    _seed_fric(dg)
    _seed_fric(dc)
    _seed_fric(dp)
    dg.upload_all(ctx)

    var sg = DynamicsScratch[DTYPE, MD, BATCH]()
    var sc = DynamicsScratch[DTYPE, MD, BATCH]()
    var sp = DynamicsScratch[DTYPE, MD, BATCH]()
    comptime JE_WS = je_ws_size[DTYPE, MD.NV, MD.NJOINT, MD.NTENDON, MD.NEQUALITY, MD.MAX_CONTACTS, 3]()
    var cg = ContactScratch[DTYPE, MD, BATCH, JE_WS]()
    var cc = ContactScratch[DTYPE, MD, BATCH, JE_WS]()
    var cp = ContactScratch[DTYPE, MD, BATCH, JE_WS]()
    sg.upload_all(ctx)
    cg.upload_all(ctx)

    _fields_prep["gpu"](dg, mf, sg, ctx)
    _fields_prep["cpu"](dc, mf, sc, None)
    _fields_prep["cpu"](dp, mf, sp, None)

    var ncon = 0
    for e in range(BATCH):
        ncon += Int(dc.meta.data[e * METADATA_SIZE + META_IDX_NUM_CONTACTS])
    print("  contacts (cpu prep):", ncon)
    if ncon == 0:
        raise Error(
            "no contacts — the COUPLED friction+contact regime is untested"
        )

    solve_newton_blocked["gpu", DTYPE, CONE_TYPE=ConeType.PYRAMIDAL, BATCH=BATCH, JE_WS=JE_WS](dg, mf, sg, cg, ctx)
    solve_newton_blocked["cpu", DTYPE, CONE_TYPE=ConeType.PYRAMIDAL, BATCH=BATCH, JE_WS=JE_WS](dc, mf, sc, cc, None)
    solve_newton["cpu", DTYPE, CONE_TYPE=ConeType.PYRAMIDAL, BATCH=BATCH, JE_WS=JE_WS](dp, mf, sp, cp, None)

    sg.qacc_constrained.download(ctx)

    var w_gpu = Float64(0)
    var w_env = Float64(0)
    for k in range(BATCH * NV):
        var a = Float64(sg.qacc_constrained.data[k])
        var b = Float64(sc.qacc_constrained.data[k])
        var c = Float64(sp.qacc_constrained.data[k])
        var den = abs(b)
        if den < 1e-6:
            den = 1e-6
        var r1 = abs(a - b) / den
        var r2 = abs(b - c) / den
        if r1 > w_gpu:
            w_gpu = r1
        if r2 > w_env:
            w_env = r2
    print("  blocked GPU vs blocked CPU worst rel err:", w_gpu)
    print("  blocked      vs per-env     worst rel err:", w_env)
    if w_gpu > REL_TOL:
        raise Error(
            "blocked GPU disagrees with blocked CPU on friction rows — the"
            " cooperative publication of kind/R/floss/state is wrong"
        )
    if w_env > REL_TOL:
        raise Error(
            "the blocked and per-env pyramidal solvers DRIFTED on friction"
            " rows — they are meant to build the same rows"
        )
    print("  PASS: both pyramidal solvers carry the dry-friction rows")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
