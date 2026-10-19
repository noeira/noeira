"""Connect/weld equality as ROWS on the PYRAMIDAL and BLOCKED Newton paths.

Defect 29a: solving weld/connect as a post-pass after the contact solve
rewrites the dofs the contacts had just balanced, so the contact force is
computed as if the coupling were absent. On sawyer that left the welded
object 77.6 mm from where MuJoCo rests it; building the rows INSIDE the
elliptic Newton system (`d22144ee`) brought it to 0.087 mm.

That conversion was ELLIPTIC-ONLY. The pyramidal per-env path and the
cooperative blocked kernel kept the post-pass until 2026-08-12, which is what
this file gates.

THE FIXTURE IS DEFECT 29a'S SHAPE, MINIMISED. `arm` hangs in the air held
only by a soft weld to the world; `obj` (10 kg, against the arm's 0.5 kg)
rests on its top face. The obj's weight reaches the weld THROUGH the contact,
so contact rows and weld rows act on the same dofs and have to be solved
together. Measured from MuJoCo: removing `obj` raises the arm by 9.79e-2 m, so
~98 mm of the arm's position is load carried across that coupling — the same
scale as the original sawyer defect, and the part a post-pass gets wrong.

⚠ A CONTACT MUST BE LIVE OR THIS FILE GATES NOTHING. With no contact the weld
is the only constraint, a post-pass converges to the same answer as a row
(that is exactly why the defect hid: "a quadruped in free flight matched
MuJoCo to 1e-7"), and every assertion below would pass with the bug present.
`test_the_fixture_is_not_vacuous` pins this from MuJoCo's side.

⚠ ALSO CHECK THE ARM CLEARS THE FLOOR. At heavier loads it lands on the plane,
ncon goes 4 -> 8, and the arm's position stops being set by the weld/contact
balance at all.

COVERAGE LIMIT, stated rather than implied: the blocked leg here compares
blocked-CPU against per-env-CPU, so it gates the ROW BUILDING on that kernel.
The cooperative GPU publication of `kind_e_sh` across threads is covered by
`test_newton_blocked_tendon_fields` (same mechanism, tendon rows) and not
re-tested here.

Run with:
    pixi run mojo run -I . tests/physics3d/test_weld_rows_pyramidal_vs_mujoco.mojo
"""

from std.math import abs
from std.python import Python
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext

from noeira.physics3d.parser import parse_xml, ModelDefFromXML
from noeira.physics3d.parser.xml_parser import merge_mjcf
from noeira.physics3d.fields import (
    AsStatic,
    Model,
    Data,
    DynamicsScratch,
    ContactScratch,
    Dims,
    Scratch,
    cap,
)
from noeira.physics3d.kinematics.forward_kinematics import (
    forward_kinematics,
    compute_body_velocities,
)
from noeira.physics3d.integrator.euler import (
    EulerIntegrator,
    _armature_env,
    _fnet_passive_env,
    _qacc_writeback_env,
)
from noeira.physics3d.dynamics.subtree_com import compute_subtree_com
from noeira.physics3d.dynamics.cdof import compute_cdof
from noeira.physics3d.dynamics.mass_matrix import compute_mass_matrix
from noeira.physics3d.dynamics.ldl import (
    ldl_factor,
    ldl_solve,
    compute_m_inv,
)
from noeira.physics3d.dynamics.rne import compute_bias_forces_rne
from noeira.physics3d.collision.contact_detection import detect_contacts
from noeira.physics3d.solver.newton_solve import (
    solve_newton,
    solve_newton_blocked,
)
from noeira.physics3d.constraints.equality_tendon import (
    build_weld_equality_rows,
)
from noeira.physics3d.types import _max_one
from noeira.physics3d.dynamics.ldl import compute_m_inv as _compute_m_inv
from noeira.physics3d.gpu.constants import (
    MODEL_JOINT_SIZE,
    META_IDX_NUM_CONTACTS,
    METADATA_SIZE,
    MODEL_EQ_SIZE,
    MODEL_BODY_SIZE,
    MODEL_META_SIZE,
    EQ_IDX_ANCHOR_AX,
    EQ_IDX_ANCHOR_AY,
    EQ_IDX_ANCHOR_AZ,
    EQ_IDX_RELPOSE_X,
    EQ_IDX_RELPOSE_Y,
    EQ_IDX_RELPOSE_Z,
    EQ_IDX_RELPOSE_W,
)
from noeira.physics3d.types import ConeType
from noeira.physics3d.solver.je_budget import je_ws_size
from noeira.core.tiles import RowMajor
from noeira.physics3d.model.model_dims import ModelDims

comptime DTYPE = DType.float64
comptime NSTEPS = 600

comptime _RAW = """
<mujoco model="weld_contact">
  <option timestep="0.002" gravity="0 0 -9.81" solver="Newton" cone="pyramidal"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 .1"/>
    <body name="arm" pos="0 0 0.3">
      <freejoint/>
      <geom name="g_arm" type="box" size="0.15 0.15 0.05" mass="0.5"/>
    </body>
    <body name="obj" pos="0 0 0.42">
      <freejoint/>
      <geom name="g_obj" type="box" size="0.05 0.05 0.05" mass="10"/>
    </body>
  </worldbody>
  <equality>
    <weld name="hold" body1="arm" solref="0.1 1"/>
  </equality>
</mujoco>
"""

comptime XML = merge_mjcf(_RAW)
comptime pm = parse_xml(XML)
comptime M = ModelDefFromXML[
    xml=XML,
    nbody = pm.NBODY,
    njoint = pm.NJOINT,
    nq = pm.NQ,
    nv = pm.NV,
    ngeom = pm.NGEOM,
    nact = pm.NACT,
    ntex = pm.NTEX,
    nmat = pm.NMAT,
    nlight = pm.NLIGHT,
    ncam = pm.NCAM,
    nsite = pm.NSITE,
    max_tendon = pm.NTENDON,
    cone_type = ConeType.PYRAMIDAL,
    max_contacts=16,
    max_condim = pm.MAX_CONDIM,
    neq = pm.NEQ,
    max_equality = pm.NEQ,
    nexclude = pm.NEXCLUDE,
    timestep = pm.TIMESTEP,
]
comptime MD = ModelDims[M]

# The SAME model with `relpose` written out by hand. MuJoCo derives
# `(0, 0, -0.3, 1, 0, 0, 0)` for the model above, so these two are physically
# identical — which is the assertion: deriving must reproduce writing it out.
# ⚠ Written out in full rather than `_RAW.replace(...)`: `_RAW` is a
# StringLiteral, which has no `replace`, and comptime String slicing is a known
# trap in this codebase.
comptime _RAW_EXPLICIT = """
<mujoco model="weld_contact">
  <option timestep="0.002" gravity="0 0 -9.81" solver="Newton" cone="pyramidal"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 .1"/>
    <body name="arm" pos="0 0 0.3">
      <freejoint/>
      <geom name="g_arm" type="box" size="0.15 0.15 0.05" mass="0.5"/>
    </body>
    <body name="obj" pos="0 0 0.42">
      <freejoint/>
      <geom name="g_obj" type="box" size="0.05 0.05 0.05" mass="10"/>
    </body>
  </worldbody>
  <equality>
    <weld name="hold" body1="arm" relpose="0 0 -0.3 1 0 0 0" solref="0.1 1"/>
  </equality>
</mujoco>
"""
comptime XML_EXPLICIT = merge_mjcf(_RAW_EXPLICIT)
comptime pmx = parse_xml(XML_EXPLICIT)
comptime MX = ModelDefFromXML[
    xml=XML_EXPLICIT,
    nbody = pmx.NBODY, njoint = pmx.NJOINT, nq = pmx.NQ, nv = pmx.NV,
    ngeom = pmx.NGEOM, nact = pmx.NACT, ntex = pmx.NTEX, nmat = pmx.NMAT,
    nlight = pmx.NLIGHT, ncam = pmx.NCAM, nsite = pmx.NSITE,
    max_tendon = pmx.NTENDON, cone_type = ConeType.PYRAMIDAL,
    max_contacts=16, max_condim = pmx.MAX_CONDIM,
    neq = pmx.NEQ, max_equality = pmx.NEQ, nexclude = pmx.NEXCLUDE,
    timestep = pmx.TIMESTEP,
]
comptime MD_2 = ModelDims[MX]


# ── torque specimen: an OFFSET MASS on a welded body, NO CONTACTS ────────────
#
# Gravity acts on a centre of mass 0.25 m to the side of the weld, so it applies
# a pure MOMENT and the orientation rows carry the whole load. `ncon = 0` in
# both engines — deliberately.
#
# ⚠ THE FIRST VERSION OF THIS FIXTURE PUT THE TORQUE ON THROUGH A CONTACT (an
# off-centre object resting on the arm) AND MEASURED THE WRONG THING. Stepping
# both engines in lockstep showed |d| = 0.0 for the first 20 steps — while the
# object was still falling — and divergence only once contacts formed, with
# MuJoCo reporting ncon = 2 where we report 4. That is the box-box multicontact
# gap (section 20, deliberately unported), not the weld. A weld gate must not
# route its load through a contact manifold.
comptime _RAW_TQ1 = """
<mujoco model="weld_torque">
  <option timestep="0.002" gravity="0 0 -9.81" solver="Newton" cone="pyramidal"/>
  <worldbody>
    <body name="arm" pos="0 0 0.5">
      <freejoint/>
      <geom name="g_arm" type="box" size="0.05 0.05 0.05" pos="0.25 0 0" mass="2"/>
    </body>
  </worldbody>
  <equality>
    <weld name="hold" body1="arm" torquescale="1" relpose="0 0 -0.5 1 0 0 0" solref="0.1 1"/>
  </equality>
</mujoco>
"""

comptime _RAW_TQ5 = """
<mujoco model="weld_torque">
  <option timestep="0.002" gravity="0 0 -9.81" solver="Newton" cone="pyramidal"/>
  <worldbody>
    <body name="arm" pos="0 0 0.5">
      <freejoint/>
      <geom name="g_arm" type="box" size="0.05 0.05 0.05" pos="0.25 0 0" mass="2"/>
    </body>
  </worldbody>
  <equality>
    <weld name="hold" body1="arm" torquescale="5" relpose="0 0 -0.5 1 0 0 0" solref="0.1 1"/>
  </equality>
</mujoco>
"""

comptime XML_TQ1 = merge_mjcf(_RAW_TQ1)
comptime XML_TQ5 = merge_mjcf(_RAW_TQ5)
comptime pq1 = parse_xml(XML_TQ1)
comptime pq5 = parse_xml(XML_TQ5)
comptime MTQ1 = ModelDefFromXML[
    xml=XML_TQ1,
    nbody = pq1.NBODY, njoint = pq1.NJOINT, nq = pq1.NQ, nv = pq1.NV,
    ngeom = pq1.NGEOM, nact = pq1.NACT, ntex = pq1.NTEX, nmat = pq1.NMAT,
    nlight = pq1.NLIGHT, ncam = pq1.NCAM, nsite = pq1.NSITE,
    max_tendon = pq1.NTENDON, cone_type = ConeType.PYRAMIDAL,
    max_contacts=4, max_condim = pq1.MAX_CONDIM,
    neq = pq1.NEQ, max_equality = pq1.NEQ, nexclude = pq1.NEXCLUDE,
    timestep = pq1.TIMESTEP,
]
comptime MD_3 = ModelDims[MTQ1]
comptime MTQ5 = ModelDefFromXML[
    xml=XML_TQ5,
    nbody = pq5.NBODY, njoint = pq5.NJOINT, nq = pq5.NQ, nv = pq5.NV,
    ngeom = pq5.NGEOM, nact = pq5.NACT, ntex = pq5.NTEX, nmat = pq5.NMAT,
    nlight = pq5.NLIGHT, ncam = pq5.NCAM, nsite = pq5.NSITE,
    max_tendon = pq5.NTENDON, cone_type = ConeType.PYRAMIDAL,
    max_contacts=4, max_condim = pq5.MAX_CONDIM,
    neq = pq5.NEQ, max_equality = pq5.NEQ, nexclude = pq5.NEXCLUDE,
    timestep = pq5.TIMESTEP,
]
comptime MD_4 = ModelDims[MTQ5]


# qpos = [arm(7), obj(7)]; z is index 2 of each free joint.
comptime ARM_Z = 2
comptime OBJ_Z = 9


def _mujoco_roll(drop_obj: Bool) raises -> Tuple[Float64, Float64, Int]:
    """Roll MuJoCo `NSTEPS` and return (arm_z, obj_z, ncon).

    `drop_obj` removes the obj body entirely — the control that says how much
    of the arm's position is load carried through the contact. ⚠ NOT "move the
    obj far away": at z = 3 it falls straight back onto the arm's footprint
    with more energy, which reads as a control and is not one.
    """
    var mujoco = Python.import_module("mujoco")
    var src = materialize[XML]()
    if drop_obj:
        var body_start = src.find('<body name="obj"')
        var body_end = src.find("</body>", body_start)
        src = String(src[byte=0:body_start]) + String(
            src[byte = body_end + 7 : src.byte_length()]
        )
    var m = mujoco.MjModel.from_xml_string(src)
    var dat = mujoco.MjData(m)
    for _ in range(NSTEPS):
        mujoco.mj_step(m, dat)
    var oz = Float64(0)
    if not drop_obj:
        oz = Float64(py=dat.qpos[OBJ_Z])
    return (Float64(py=dat.qpos[ARM_Z]), oz, Int(py=dat.ncon))


def test_the_fixture_is_not_vacuous() raises:
    """A weld with no live contact converges the same either way."""
    print("--- weld rows: the fixture couples a contact to the weld ---")
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(materialize[XML]())
    var dat = mujoco.MjData(m)
    for _ in range(NSTEPS):
        mujoco.mj_step(m, dat)

    var ncon = Int(py=dat.ncon)
    print("  MuJoCo ncon =", ncon, " nefc =", Int(py=dat.nefc))
    assert_true(
        ncon == 4,
        "expected the obj's 4-point face contact on the arm and nothing else;"
        " got ncon = "
        + String(ncon)
        + ". At 0 the weld is unopposed and a post-pass would pass this file;"
        " at 8 the arm has landed on the floor and its position is no longer"
        " set by the weld/contact balance.",
    )
    var n_eq = 0
    for i in range(Int(py=dat.nefc)):
        if Int(py=dat.efc_type[i]) == Int(
            py=mujoco.mjtConstraint.mjCNSTR_EQUALITY
        ):
            n_eq += 1
    print("  MuJoCo equality rows =", n_eq)
    assert_true(n_eq == 6, "the weld is not contributing 6 rows")

    var held = _mujoco_roll(False)
    var unloaded = _mujoco_roll(True)
    var load_sag = unloaded[0] - held[0]
    print("  arm_z loaded  =", held[0])
    print("  arm_z unloaded=", unloaded[0])
    print("  load carried across the coupling =", load_sag, "m")
    assert_true(
        load_sag > 0.01,
        "less than 10 mm of the arm's position comes from the obj's load —"
        " the coupling this file exists to measure has gone weak, and a"
        " post-passed weld would be within tolerance",
    )
    # The arm must be clear of the plane, or the floor is holding it up.
    assert_true(
        held[0] - 0.05 > 0.02,
        "the arm's underside is within 20 mm of the floor — check ncon above;"
        " the fixture is drifting toward the regime where the plane, not the"
        " weld, sets the answer",
    )


def test_pyramidal_weld_matches_mujoco() raises:
    """Per-env PYRAMIDAL Newton, full rollout, against MuJoCo."""
    var sf = M.make_spec_fields[DTYPE]()
    print("--- weld rows: per-env pyramidal vs MuJoCo ---")
    var ctx = DeviceContext()
    var mf = Model[DTYPE, MD]()
    M.init_fields[DTYPE](ctx, mf)

    var d = Data[DTYPE, MD, 1]()
    M.reset_data[DTYPE](sf, d)
    forward_kinematics["cpu"](d, mf)

    var integ = EulerIntegrator[DTYPE, MD, M.CONE_TYPE, 1, SOLVER="newton", MAX_CONDIM = M.MAX_CONDIM, NOSLIP_ITER = M.NOSLIP_ITER]()
    for _ in range(NSTEPS):
        integ.step["cpu", CONTACTS=True](d, mf)

    var mj = _mujoco_roll(False)
    var arm = Float64(d.qpos.data[ARM_Z])
    var obj = Float64(d.qpos.data[OBJ_Z])
    print("  arm_z ours", arm, " MuJoCo", mj[0], " |d|", abs(arm - mj[0]))
    print("  obj_z ours", obj, " MuJoCo", mj[1], " |d|", abs(obj - mj[1]))
    assert_true(
        abs(arm - mj[0]) < 1e-6 and abs(obj - mj[1]) < 1e-6,
        "the pyramidal path disagrees with MuJoCo on a weld carrying a"
        " contact. As a POST-PASS this is the defect-29a failure: the contact"
        " force is computed as if the weld were absent.",
    )


def _prep(
    mut d: Data[DTYPE, MD, 1],
    mut mf: Model[DTYPE, MD],
    mut sc: DynamicsScratch[DTYPE, MD, 1],
) raises:
    """Smooth dynamics + detection, up to the constraint seam.

    Mirrors `EulerIntegrator.step`'s order so the two solvers below are handed
    the state the integrator would hand them — the same helper shape
    `test_newton_blocked_tendon_fields` uses, CPU-only.
    """
    forward_kinematics["cpu"](d, mf, None)
    compute_body_velocities["cpu"](d, mf, None)
    compute_subtree_com["cpu"](d, mf, None)
    compute_cdof["cpu"](d, mf, sc, None)
    compute_mass_matrix["cpu"](d, mf, sc, None)

    comptime L_JOINT = RowMajor[M.NJOINT, MODEL_JOINT_SIZE]
    comptime L_M = RowMajor[1, M.NV * M.NV]
    comptime L_NV = RowMajor[1, M.NV]
    comptime L_QPOS = RowMajor[1, M.NQ]

    var joints_v = mf.joints.tt["cpu", M.NJOINT, MODEL_JOINT_SIZE]()
    _armature_env[DTYPE](
        0, AsStatic[MD](), joints_v, sc.M.tt["cpu", 1, M.NV * M.NV]()
    )
    ldl_factor["cpu", DTYPE, BATCH=1](mf, sc, None)
    compute_m_inv["cpu", DTYPE, BATCH=1](mf, sc, None)
    compute_bias_forces_rne["cpu"](d, mf, sc, None)
    _fnet_passive_env[DTYPE](
        0,
        AsStatic[MD](),
        d.qpos.tt["cpu", 1, M.NQ](),
        d.qvel.tt["cpu", 1, M.NV](),
        d.qfrc.tt["cpu", 1, M.NV](),
        joints_v,
        sc.bias.tt["cpu", 1, M.NV](),
        sc.fnet.tt["cpu", 1, M.NV](),
    )
    ldl_solve["cpu", DTYPE, BATCH=1](mf, sc, None)
    _qacc_writeback_env[DTYPE](
        0,
        AsStatic[MD](),
        sc.qacc_ws.tt["cpu", 1, M.NV](),
        d.qacc.tt["cpu", 1, M.NV](),
        sc.qacc_constrained.tt["cpu", 1, M.NV](),
    )
    detect_contacts["cpu"](d, mf, None)


def test_blocked_kernel_builds_the_same_weld_rows() raises:
    """Blocked cooperative kernel vs the per-env pyramidal solver, same state.

    Both build their rows from `build_weld_equality_rows`; if the blocked
    kernel had kept the post-pass — or sized `ME` without `6*NEQUALITY`, so
    the rows fell off the end of the edge list — these would diverge.
    """
    var sf = M.make_spec_fields[DTYPE]()
    print("--- weld rows: blocked kernel vs per-env ---")
    var ctx = DeviceContext()
    var mf = Model[DTYPE, MD]()
    M.init_fields[DTYPE](ctx, mf)

    # Settle into the COUPLED state first. Comparing two solvers at the reset
    # pose would compare them with no contact live, which is the regime where
    # a post-pass and a row agree.
    var d = Data[DTYPE, MD, 1]()
    M.reset_data[DTYPE](sf, d)
    forward_kinematics["cpu"](d, mf)
    var integ = EulerIntegrator[DTYPE, MD, M.CONE_TYPE, 1, SOLVER="newton", MAX_CONDIM = M.MAX_CONDIM, NOSLIP_ITER = M.NOSLIP_ITER]()
    for _ in range(NSTEPS):
        integ.step["cpu", CONTACTS=True](d, mf)

    var db = Data[DTYPE, MD, 1]()
    var dp = Data[DTYPE, MD, 1]()
    M.reset_data[DTYPE](sf, db)
    M.reset_data[DTYPE](sf, dp)
    for i in range(M.NQ):
        db.qpos.data[i] = d.qpos.data[i]
        dp.qpos.data[i] = d.qpos.data[i]
    for i in range(M.NV):
        db.qvel.data[i] = d.qvel.data[i]
        dp.qvel.data[i] = d.qvel.data[i]

    var sb = DynamicsScratch[DTYPE, MD, 1]()
    var sp = DynamicsScratch[DTYPE, MD, 1]()
    comptime JE_WS = je_ws_size[DTYPE, MD.NV, MD.NJOINT, MD.NTENDON, MD.NEQUALITY, MD.MAX_CONTACTS, 3]()
    var cb = ContactScratch[DTYPE, MD, 1, JE_WS]()
    var cp = ContactScratch[DTYPE, MD, 1, JE_WS]()
    _prep(db, mf, sb)
    _prep(dp, mf, sp)

    var ncon = Int(db.meta.data[META_IDX_NUM_CONTACTS])
    print("  contacts in the prepared state:", ncon)
    assert_true(
        ncon > 0,
        "no contacts in the prepared state — the two solvers are being"
        " compared in the regime where a post-passed weld and a weld row"
        " agree, so this comparison would pass with the bug present",
    )

    solve_newton_blocked["cpu", DTYPE, CONE_TYPE=ConeType.PYRAMIDAL, BATCH=1, JE_WS=JE_WS](db, mf, sb, cb, None)
    solve_newton["cpu", DTYPE, CONE_TYPE=ConeType.PYRAMIDAL, BATCH=1, JE_WS=JE_WS](dp, mf, sp, cp, None)

    var worst = Float64(0)
    for i in range(M.NV):
        var e = abs(
            Float64(sb.qacc_constrained.data[i])
            - Float64(sp.qacc_constrained.data[i])
        )
        if e > worst:
            worst = e
    print("  worst |d(qacc)| blocked vs per-env =", worst)
    assert_true(
        worst < 1e-9,
        "the blocked kernel and the per-env pyramidal solver disagree on the"
        " weld rows (|d| " + String(worst) + ")",
    )


def test_relpose_default_is_derived_from_qpos0() raises:
    """MJCF's default `relpose` means "derive from qpos0", NOT "identity".

    We defaulted to identity until 2026-08-12, which welds the two bodies
    COINCIDENT rather than holding the offset they start with — here it dragged
    the arm to the world origin until the floor stopped it. Invisible because
    sawyer, the only model in the tree with a weld, has mocap and hand at the
    same pose at qpos0, so identity was accidentally right.

    Gated two ways, because a rollout alone would not say WHICH value is wrong:
    the stored record against MuJoCo's `eq_data`, and the derived model against
    a twin with `relpose` written out by hand.
    """
    print("--- weld rows: relpose default derives from qpos0 ---")
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(materialize[XML]())

    var ctx = DeviceContext()
    var mf = Model[DTYPE, MD]()
    M.init_fields[DTYPE](ctx, mf)

    # MuJoCo eq_data for a weld: [anchor(3), relpose_pos(3), relpose_quat(4),
    # torquescale]. `mj_equalityAnchors` anchors body1 at relpose_pos, which is
    # our `anchor_a`; the quaternion is (w,x,y,z) against our (x,y,z,w).
    var want_px = Float64(py=m.eq_data[0][3])
    var want_py = Float64(py=m.eq_data[0][4])
    var want_pz = Float64(py=m.eq_data[0][5])
    var want_qw = Float64(py=m.eq_data[0][6])
    var want_qx = Float64(py=m.eq_data[0][7])
    var want_qy = Float64(py=m.eq_data[0][8])
    var want_qz = Float64(py=m.eq_data[0][9])
    print("  MuJoCo relpose pos", want_px, want_py, want_pz,
          " quat(wxyz)", want_qw, want_qx, want_qy, want_qz)

    assert_true(
        abs(want_pz) > 1e-6,
        "MuJoCo derived a ZERO relpose position — the fixture no longer"
        " distinguishes 'derive from qpos0' from 'identity' and this test"
        " would pass with the old default",
    )

    var got_px = Float64(mf.equality.data[EQ_IDX_ANCHOR_AX])
    var got_py = Float64(mf.equality.data[EQ_IDX_ANCHOR_AY])
    var got_pz = Float64(mf.equality.data[EQ_IDX_ANCHOR_AZ])
    var got_qx = Float64(mf.equality.data[EQ_IDX_RELPOSE_X])
    var got_qy = Float64(mf.equality.data[EQ_IDX_RELPOSE_Y])
    var got_qz = Float64(mf.equality.data[EQ_IDX_RELPOSE_Z])
    var got_qw = Float64(mf.equality.data[EQ_IDX_RELPOSE_W])
    print("  ours   relpose pos", got_px, got_py, got_pz,
          " quat(wxyz)", got_qw, got_qx, got_qy, got_qz)

    var worst = abs(got_px - want_px)
    if abs(got_py - want_py) > worst:
        worst = abs(got_py - want_py)
    if abs(got_pz - want_pz) > worst:
        worst = abs(got_pz - want_pz)
    if abs(got_qw - want_qw) > worst:
        worst = abs(got_qw - want_qw)
    if abs(got_qx - want_qx) > worst:
        worst = abs(got_qx - want_qx)
    if abs(got_qy - want_qy) > worst:
        worst = abs(got_qy - want_qy)
    if abs(got_qz - want_qz) > worst:
        worst = abs(got_qz - want_qz)
    print("  worst |d| =", worst)
    assert_true(
        worst < 1e-12,
        "the derived relpose disagrees with MuJoCo's by " + String(worst),
    )


def test_explicit_relpose_is_still_honoured() raises:
    """Writing `relpose` out by hand must give the same model as deriving it.

    MuJoCo derives exactly `0 0 -0.3 1 0 0 0` for the fixture, so the twin is
    physically identical — and an EXPLICIT identity quaternion has to stay
    identity rather than being re-derived, which is why the "unset" test is on
    the quaternion rather than on whether the attribute was written.
    """
    var sf = MX.make_spec_fields[DTYPE]()
    print("--- weld rows: explicit relpose ---")
    var ctx = DeviceContext()
    var mf = Model[DTYPE, MD_2]()
    MX.init_fields[DTYPE](ctx, mf)

    var d = Data[DTYPE, MD_2, 1]()
    MX.reset_data[DTYPE](sf, d)
    forward_kinematics["cpu"](d, mf)
    var integ = EulerIntegrator[DTYPE, MD_2, MX.CONE_TYPE, 1, SOLVER="newton", MAX_CONDIM = MX.MAX_CONDIM, NOSLIP_ITER = MX.NOSLIP_ITER]()
    for _ in range(NSTEPS):
        integ.step["cpu", CONTACTS=True](d, mf)

    var mj = _mujoco_roll(False)
    var arm = Float64(d.qpos.data[ARM_Z])
    print("  arm_z ours", arm, " MuJoCo", mj[0], " |d|", abs(arm - mj[0]))
    assert_true(
        abs(arm - mj[0]) < 1e-6,
        "an EXPLICIT relpose no longer reproduces MuJoCo — check that the"
        " all-zero-quaternion 'unset' test is not swallowing written values",
    )


def _mujoco_arm_qpos(xml: String) raises -> List[Float64]:
    """Roll MuJoCo on `xml` and return the arm free joint's 7 qpos."""
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(xml)
    var dat = mujoco.MjData(m)
    for _ in range(NSTEPS):
        mujoco.mj_step(m, dat)
    var out = List[Float64]()
    for i in range(7):
        out.append(Float64(py=dat.qpos[i]))
    return out^



def test_weld_orientation_rows_match_mujoco() raises:
    """Weld rows vs MuJoCo's `efc_*` at a TILTED pose, torquescale = 1.

    ROWS, not a rollout. This fixture is a heavy offset mass on a soft weld —
    an unstable swinging system — and stepping it 600 steps amplifies a
    ~1e-10 solver-tolerance difference into 0.39 by the end while agreeing to
    1.0e-19 at step ONE. A trajectory comparison there measures Lyapunov
    growth, not the constraint. `efc_J` / `efc_aref` / `efc_D` are exact and
    pose-local, so they say what the rows are.
    """
    var sf = MTQ1.make_spec_fields[DTYPE]()
    print("--- weld rows vs efc, torquescale = 1 ---")
    var mujoco = Python.import_module("mujoco")
    var np = Python.import_module("numpy")
    var m = mujoco.MjModel.from_xml_string(materialize[XML_TQ1]())
    var dat = mujoco.MjData(m)
    var th = 0.4
    dat.qpos[0] = 0.07
    dat.qpos[1] = -0.02
    dat.qpos[2] = 0.33
    dat.qpos[3] = np.cos(th / 2)
    dat.qpos[4] = 0.0
    dat.qpos[5] = np.sin(th / 2)
    dat.qpos[6] = 0.0
    mujoco.mj_forward(m, dat)
    var nefc = Int(py=dat.nefc)
    assert_true(
        nefc == 6,
        "expected the weld's 6 rows and nothing else; got " + String(nefc),
    )

    var ctx = DeviceContext()
    var mf = Model[DTYPE, MD_3]()
    MTQ1.init_fields[DTYPE](ctx, mf)
    var d = Data[DTYPE, MD_3, 1]()
    MTQ1.reset_data[DTYPE](sf, d)
    d.qpos.data[0] = 0.07
    d.qpos.data[1] = -0.02
    d.qpos.data[2] = 0.33
    d.qpos.data[3] = Float64(py=np.cos(th / 2))
    d.qpos.data[4] = 0.0
    d.qpos.data[5] = Float64(py=np.sin(th / 2))
    d.qpos.data[6] = 0.0
    for i in range(MTQ1.NV):
        d.qvel.data[i] = 0

    var sc = DynamicsScratch[DTYPE, MD_3, 1]()
    forward_kinematics["cpu"](d, mf, None)
    compute_body_velocities["cpu"](d, mf, None)
    compute_subtree_com["cpu"](d, mf, None)
    compute_cdof["cpu"](d, mf, sc, None)
    compute_mass_matrix["cpu"](d, mf, sc, None)
    ldl_factor["cpu", DTYPE, BATCH=1](mf, sc, None)
    _compute_m_inv["cpu", DTYPE, BATCH=1](mf, sc, None)

    comptime WR = 6 * cap[MTQ1.MAX_EQUALITY]()
    comptime WJ = 6 * cap[MTQ1.MAX_EQUALITY]() * cap[MTQ1.NV]()
    var w_K = Scratch[Scalar[DTYPE], WR](6 * MTQ1.MAX_EQUALITY, Scalar[DTYPE](1))
    var w_bias = Scratch[Scalar[DTYPE], WR](6 * MTQ1.MAX_EQUALITY, Scalar[DTYPE](0))
    var w_D = Scratch[Scalar[DTYPE], WR](6 * MTQ1.MAX_EQUALITY, Scalar[DTYPE](0))
    var w_J = Scratch[Scalar[DTYPE], WJ](6 * MTQ1.MAX_EQUALITY * MTQ1.NV, Scalar[DTYPE](0))
    var w_MinvJ = Scratch[Scalar[DTYPE], WJ](6 * MTQ1.MAX_EQUALITY * MTQ1.NV, Scalar[DTYPE](0))

    comptime L_B3 = RowMajor[1, MTQ1.NBODY * 3]
    comptime L_B4 = RowMajor[1, MTQ1.NBODY * 4]
    comptime L_NV = RowMajor[1, MTQ1.NV]
    comptime L_NQ = RowMajor[1, MTQ1.NQ]
    comptime L_DW = RowMajor[MTQ1.NV]
    comptime L_JT = RowMajor[MTQ1.NJOINT, MODEL_JOINT_SIZE]
    comptime L_BD = RowMajor[MTQ1.NBODY, MODEL_BODY_SIZE]
    comptime L_MT = RowMajor[MODEL_META_SIZE]
    comptime L_EQ = RowMajor[MTQ1.MAX_EQUALITY, MODEL_EQ_SIZE]
    comptime L_IW = RowMajor[MTQ1.NBODY, 2]
    comptime L_CD = RowMajor[1, MTQ1.NV * 6]
    comptime L_MI = RowMajor[1, MTQ1.NV * MTQ1.NV]

    var n = build_weld_equality_rows[DTYPE, MTQ1.NV](
        0,
        AsStatic[MD_3](),
        d.qpos.tt["cpu", 1, MTQ1.NQ](),
        d.qvel.tt["cpu", 1, MTQ1.NV](),
        d.xpos.tt["cpu", 1, MTQ1.NBODY * 3](),
        d.xquat.tt["cpu", 1, MTQ1.NBODY * 4](),
        d.subtree_com.tt["cpu", 1, MTQ1.NBODY * 3](),
        mf.joints.tt["cpu", MTQ1.NJOINT, MODEL_JOINT_SIZE](),
        mf.bodies.tt["cpu", MTQ1.NBODY, MODEL_BODY_SIZE](),
        mf.meta.tt["cpu", MODEL_META_SIZE](),
        mf.equality.tt["cpu", MTQ1.MAX_EQUALITY, MODEL_EQ_SIZE](),
        mf.body_invweight0.tt["cpu", MTQ1.NBODY, 2](),
        mf.dof_invweight0.tt["cpu", MTQ1.NV](),
        sc.cdof.tt["cpu", 1, MTQ1.NV * 6](),
        sc.m_inv.tt["cpu", 1, MTQ1.NV * MTQ1.NV](),
        w_K, w_bias, w_D, w_J, w_MinvJ,
    )
    assert_true(n == 6, "expected 6 weld rows, built " + String(n))

    var efc = dat.efc_J.reshape(nefc, Int(py=m.nv))
    var wJ = Float64(0)
    var off_diag = Float64(0)
    for r in range(6):
        for j in range(MTQ1.NV):
            var ours = Float64(w_J[r * MTQ1.NV + j])
            var theirs = Float64(py=efc[r][j])
            var e = abs(ours - theirs)
            if e > wJ:
                wJ = e
            # rotational block, off-diagonal: zero iff the world-axis
            # approximation would have been good enough here
            if r >= 3 and j >= 3 and (r - 3) != (j - 3):
                if abs(theirs) > off_diag:
                    off_diag = abs(theirs)
    print("  worst |d(J)| =", wJ, "  rotational off-diagonal =", off_diag)
    assert_true(
        off_diag > 0.01,
        "MuJoCo's rotational Jacobian is diagonal at this pose, so the"
        " quaternion correction is invisible and three world-axis rows would"
        " pass — tilt the fixture",
    )
    assert_true(wJ < 1e-12, "weld Jacobian disagrees by " + String(wJ))

    # `bias` is MuJoCo's -aref (qvel = 0 here, so it is K*imp*pos).
    var wB = Float64(0)
    for r in range(6):
        var ours = Float64(w_bias[r])
        var theirs = -Float64(py=dat.efc_aref[r])
        var e = abs(ours - theirs)
        if e > wB:
            wB = e
    print("  worst |d(bias vs -aref)| =", wB)
    assert_true(wB < 1e-9, "weld bias disagrees by " + String(wB))

    # D: the builder returns the PGS step size 1/(k+R); recover R the way the
    # Newton paths do and compare 1/R against MuJoCo's efc_D.
    var wD = Float64(0)
    for r in range(6):
        var R = 1.0 / Float64(w_D[r]) - Float64(w_K[r])
        var ours = 1.0 / R
        var theirs = Float64(py=dat.efc_D[r])
        var rel = abs(ours - theirs) / (abs(theirs) if abs(theirs) > 1e-12 else 1.0)
        if rel > wD:
            wD = rel
    print("  worst rel |d(D)| =", wD)
    assert_true(wD < 1e-9, "weld efc_D disagrees by rel " + String(wD))


def test_weld_torquescale_matches_mujoco() raises:
    """Weld rows vs MuJoCo's `efc_*` at a TILTED pose, torquescale = 5.

    ROWS, not a rollout. This fixture is a heavy offset mass on a soft weld —
    an unstable swinging system — and stepping it 600 steps amplifies a
    ~1e-10 solver-tolerance difference into 0.39 by the end while agreeing to
    1.0e-19 at step ONE. A trajectory comparison there measures Lyapunov
    growth, not the constraint. `efc_J` / `efc_aref` / `efc_D` are exact and
    pose-local, so they say what the rows are.
    """
    var sf = MTQ5.make_spec_fields[DTYPE]()
    print("--- weld rows vs efc, torquescale = 5 ---")
    var mujoco = Python.import_module("mujoco")
    var np = Python.import_module("numpy")
    var m = mujoco.MjModel.from_xml_string(materialize[XML_TQ5]())
    var dat = mujoco.MjData(m)
    var th = 0.4
    dat.qpos[0] = 0.07
    dat.qpos[1] = -0.02
    dat.qpos[2] = 0.33
    dat.qpos[3] = np.cos(th / 2)
    dat.qpos[4] = 0.0
    dat.qpos[5] = np.sin(th / 2)
    dat.qpos[6] = 0.0
    mujoco.mj_forward(m, dat)
    var nefc = Int(py=dat.nefc)
    assert_true(
        nefc == 6,
        "expected the weld's 6 rows and nothing else; got " + String(nefc),
    )

    var ctx = DeviceContext()
    var mf = Model[DTYPE, MD_4]()
    MTQ5.init_fields[DTYPE](ctx, mf)
    var d = Data[DTYPE, MD_4, 1]()
    MTQ5.reset_data[DTYPE](sf, d)
    d.qpos.data[0] = 0.07
    d.qpos.data[1] = -0.02
    d.qpos.data[2] = 0.33
    d.qpos.data[3] = Float64(py=np.cos(th / 2))
    d.qpos.data[4] = 0.0
    d.qpos.data[5] = Float64(py=np.sin(th / 2))
    d.qpos.data[6] = 0.0
    for i in range(MTQ5.NV):
        d.qvel.data[i] = 0

    var sc = DynamicsScratch[DTYPE, MD_4, 1]()
    forward_kinematics["cpu"](d, mf, None)
    compute_body_velocities["cpu"](d, mf, None)
    compute_subtree_com["cpu"](d, mf, None)
    compute_cdof["cpu"](d, mf, sc, None)
    compute_mass_matrix["cpu"](d, mf, sc, None)
    ldl_factor["cpu", DTYPE, BATCH=1](mf, sc, None)
    _compute_m_inv["cpu", DTYPE, BATCH=1](mf, sc, None)

    comptime WR = 6 * cap[MTQ5.MAX_EQUALITY]()
    comptime WJ = 6 * cap[MTQ5.MAX_EQUALITY]() * cap[MTQ5.NV]()
    var w_K = Scratch[Scalar[DTYPE], WR](6 * MTQ5.MAX_EQUALITY, Scalar[DTYPE](1))
    var w_bias = Scratch[Scalar[DTYPE], WR](6 * MTQ5.MAX_EQUALITY, Scalar[DTYPE](0))
    var w_D = Scratch[Scalar[DTYPE], WR](6 * MTQ5.MAX_EQUALITY, Scalar[DTYPE](0))
    var w_J = Scratch[Scalar[DTYPE], WJ](6 * MTQ5.MAX_EQUALITY * MTQ5.NV, Scalar[DTYPE](0))
    var w_MinvJ = Scratch[Scalar[DTYPE], WJ](6 * MTQ5.MAX_EQUALITY * MTQ5.NV, Scalar[DTYPE](0))

    comptime L_B3 = RowMajor[1, MTQ5.NBODY * 3]
    comptime L_B4 = RowMajor[1, MTQ5.NBODY * 4]
    comptime L_NV = RowMajor[1, MTQ5.NV]
    comptime L_NQ = RowMajor[1, MTQ5.NQ]
    comptime L_DW = RowMajor[MTQ5.NV]
    comptime L_JT = RowMajor[MTQ5.NJOINT, MODEL_JOINT_SIZE]
    comptime L_BD = RowMajor[MTQ5.NBODY, MODEL_BODY_SIZE]
    comptime L_MT = RowMajor[MODEL_META_SIZE]
    comptime L_EQ = RowMajor[MTQ5.MAX_EQUALITY, MODEL_EQ_SIZE]
    comptime L_IW = RowMajor[MTQ5.NBODY, 2]
    comptime L_CD = RowMajor[1, MTQ5.NV * 6]
    comptime L_MI = RowMajor[1, MTQ5.NV * MTQ5.NV]

    var n = build_weld_equality_rows[DTYPE, MTQ5.NV](
        0,
        AsStatic[MD_4](),
        d.qpos.tt["cpu", 1, MTQ5.NQ](),
        d.qvel.tt["cpu", 1, MTQ5.NV](),
        d.xpos.tt["cpu", 1, MTQ5.NBODY * 3](),
        d.xquat.tt["cpu", 1, MTQ5.NBODY * 4](),
        d.subtree_com.tt["cpu", 1, MTQ5.NBODY * 3](),
        mf.joints.tt["cpu", MTQ5.NJOINT, MODEL_JOINT_SIZE](),
        mf.bodies.tt["cpu", MTQ5.NBODY, MODEL_BODY_SIZE](),
        mf.meta.tt["cpu", MODEL_META_SIZE](),
        mf.equality.tt["cpu", MTQ5.MAX_EQUALITY, MODEL_EQ_SIZE](),
        mf.body_invweight0.tt["cpu", MTQ5.NBODY, 2](),
        mf.dof_invweight0.tt["cpu", MTQ5.NV](),
        sc.cdof.tt["cpu", 1, MTQ5.NV * 6](),
        sc.m_inv.tt["cpu", 1, MTQ5.NV * MTQ5.NV](),
        w_K, w_bias, w_D, w_J, w_MinvJ,
    )
    assert_true(n == 6, "expected 6 weld rows, built " + String(n))

    var efc = dat.efc_J.reshape(nefc, Int(py=m.nv))
    var wJ = Float64(0)
    var off_diag = Float64(0)
    for r in range(6):
        for j in range(MTQ5.NV):
            var ours = Float64(w_J[r * MTQ5.NV + j])
            var theirs = Float64(py=efc[r][j])
            var e = abs(ours - theirs)
            if e > wJ:
                wJ = e
            # rotational block, off-diagonal: zero iff the world-axis
            # approximation would have been good enough here
            if r >= 3 and j >= 3 and (r - 3) != (j - 3):
                if abs(theirs) > off_diag:
                    off_diag = abs(theirs)
    print("  worst |d(J)| =", wJ, "  rotational off-diagonal =", off_diag)
    assert_true(
        off_diag > 0.01,
        "MuJoCo's rotational Jacobian is diagonal at this pose, so the"
        " quaternion correction is invisible and three world-axis rows would"
        " pass — tilt the fixture",
    )
    assert_true(wJ < 1e-12, "weld Jacobian disagrees by " + String(wJ))

    # `bias` is MuJoCo's -aref (qvel = 0 here, so it is K*imp*pos).
    var wB = Float64(0)
    for r in range(6):
        var ours = Float64(w_bias[r])
        var theirs = -Float64(py=dat.efc_aref[r])
        var e = abs(ours - theirs)
        if e > wB:
            wB = e
    print("  worst |d(bias vs -aref)| =", wB)
    assert_true(wB < 1e-9, "weld bias disagrees by " + String(wB))

    # D: the builder returns the PGS step size 1/(k+R); recover R the way the
    # Newton paths do and compare 1/R against MuJoCo's efc_D.
    var wD = Float64(0)
    for r in range(6):
        var R = 1.0 / Float64(w_D[r]) - Float64(w_K[r])
        var ours = 1.0 / R
        var theirs = Float64(py=dat.efc_D[r])
        var rel = abs(ours - theirs) / (abs(theirs) if abs(theirs) > 1e-12 else 1.0)
        if rel > wD:
            wD = rel
    print("  worst rel |d(D)| =", wD)
    assert_true(wD < 1e-9, "weld efc_D disagrees by rel " + String(wD))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
