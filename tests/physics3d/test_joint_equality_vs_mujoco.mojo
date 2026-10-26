"""`<equality><joint>` — mjEQ_JOINT, the polycoef coupling.

THE GAP THIS CLOSES. `mjEQ_JOINT` was absent entirely: `_fill_equality`
scanned only `<weld` and `<connect`, so an `<equality><joint>` was skipped
without a word and the two joints ran uncoupled. ToddlerBot has NINE of them —
every `_drive`/`_driven` pair in its neck, shoulders, elbows, wrists and hips.

THE ROW (engine_core_constraint.c:556), with `dif = q2 - q2_ref`:

    cpos  = q1 - q1_ref - p0 - (p1*dif + p2*dif^2 + p3*dif^3 + p4*dif^4)
    deriv = p1 + 2*p2*dif + 3*p3*dif^2 + 4*p4*dif^3
    J     = e_dof1 - deriv * e_dof2

One row, bilateral. With `joint2` absent the polynomial drops out entirely and
the row pins `q1` to `q1_ref + p0`.

⚠ THE FIXTURE USES A CUBIC, NOT ToddlerBot's LINEAR COUPLING. Every one of
ToddlerBot's nine is `polycoef="0 c 0 0 0"` — purely linear, where `deriv` is
the constant `c` and the whole quartic/derivative structure is invisible. A
fixture built from the model we care about would pass with `p2`, `p3`, `p4`
ignored and with `deriv` hard-coded to `p1`. `test_linear_polycoef_matches_
mujoco` covers ToddlerBot's actual shape; the cubic fixture is what makes the
rest of the implementation non-optional.

⚠ AND IT USES A NONZERO `ref` ON BOTH JOINTS. MuJoCo subtracts `qpos0` from
q1 AND from q2. With `ref="0"` everywhere — the obvious fixture — both
subtractions vanish and an implementation that forgot them is exact.

Run with:
    pixi run mojo run -I . tests/physics3d/test_joint_equality_vs_mujoco.mojo
"""

from std.math import abs
from std.python import Python
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext
from noeira.core.tiles import RowMajor

from noeira.physics3d.parser import parse_xml, ModelDefFromXML
from noeira.physics3d.parser.xml_parser import merge_mjcf
from noeira.physics3d.fields import Model, Data, DynamicsScratch, Dims, DimsLike, AsStatic, Scratch, cap
from noeira.physics3d.kinematics.forward_kinematics import (
    forward_kinematics,
    compute_body_velocities,
)
from noeira.physics3d.integrator.euler import EulerIntegrator
from noeira.physics3d.dynamics.subtree_com import compute_subtree_com
from noeira.physics3d.dynamics.cdof import compute_cdof
from noeira.physics3d.dynamics.mass_matrix import compute_mass_matrix
from noeira.physics3d.dynamics.ldl import ldl_factor
from noeira.physics3d.dynamics.ldl import compute_m_inv as _compute_m_inv
from noeira.physics3d.constraints.equality_tendon import (
    build_weld_equality_rows,
)
from noeira.physics3d.types import _max_one, ConeType
from noeira.physics3d.gpu.constants import (
    MODEL_JOINT_SIZE,
    MODEL_EQ_SIZE,
    MODEL_BODY_SIZE,
    MODEL_META_SIZE,
    EQ_IDX_TYPE,
)

comptime DTYPE = DType.float64
comptime NSTEPS = 40

# Two independent hinges on separate bodies, each with a NONZERO `ref`, no
# floor and no contacts — the equality is the only thing coupling them.
comptime _BODIES = """
  <option timestep="0.002" gravity="0 0 -9.81" solver="Newton"/>
  <worldbody>
    <body name="b1" pos="0 0 1">
      <joint name="j1" type="hinge" axis="0 1 0" ref="0.13"/>
      <geom name="g1" type="capsule" fromto="0 0 0 0.25 0 0" size="0.03"
            mass="1" contype="0" conaffinity="0"/>
    </body>
    <body name="b2" pos="0.6 0 1">
      <joint name="j2" type="hinge" axis="0 1 0" ref="-0.21"/>
      <geom name="g2" type="capsule" fromto="0 0 0 0.25 0 0" size="0.03"
            mass="1" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
"""

# CUBIC: p2 and p3 nonzero, so `deriv` genuinely varies with q2 and the
# quadratic/cubic residual terms are live.
comptime _RAW_CUBIC = (
    '<mujoco model="eqjoint_cubic">'
    + _BODIES
    + """
  <equality>
    <joint joint1="j1" joint2="j2" polycoef="0.05 -0.8 0.3 0.15 0"/>
  </equality>
</mujoco>
"""
)

# LINEAR: ToddlerBot's actual shape, `polycoef="0 c 0 0 0"`.
comptime _RAW_LINEAR = (
    '<mujoco model="eqjoint_linear">'
    + _BODIES
    + """
  <equality>
    <joint joint1="j1" joint2="j2" polycoef="0 -0.9090909091 0 0 0"/>
  </equality>
</mujoco>
"""
)

# SINGLE-JOINT form: joint2 omitted. MuJoCo drops the polynomial and pins
# q1 to q1_ref + p0.
comptime _RAW_SINGLE = (
    '<mujoco model="eqjoint_single">'
    + _BODIES
    + """
  <equality>
    <joint joint1="j1" polycoef="0.07 0 0 0 0"/>
  </equality>
</mujoco>
"""
)

comptime XML_C = merge_mjcf(_RAW_CUBIC)
comptime XML_L = merge_mjcf(_RAW_LINEAR)
comptime XML_S = merge_mjcf(_RAW_SINGLE)
comptime pc = parse_xml(XML_C)
comptime pl = parse_xml(XML_L)
comptime ps = parse_xml(XML_S)


def _md_c() -> ModelDefFromXML[
    xml=XML_C, nbody = pc.NBODY, njoint = pc.NJOINT, nq = pc.NQ, nv = pc.NV,
    ngeom = pc.NGEOM, nact = pc.NACT, ntex = pc.NTEX, nmat = pc.NMAT,
    nlight = pc.NLIGHT, ncam = pc.NCAM, nsite = pc.NSITE,
    max_tendon = pc.NTENDON, cone_type = ConeType.PYRAMIDAL,
    max_contacts=4, max_condim = pc.MAX_CONDIM,
    neq = pc.NEQ, max_equality = pc.NEQ,
    nexclude = pc.NEXCLUDE, timestep = pc.TIMESTEP,
]:
    return {}


def _md_l() -> ModelDefFromXML[
    xml=XML_L, nbody = pl.NBODY, njoint = pl.NJOINT, nq = pl.NQ, nv = pl.NV,
    ngeom = pl.NGEOM, nact = pl.NACT, ntex = pl.NTEX, nmat = pl.NMAT,
    nlight = pl.NLIGHT, ncam = pl.NCAM, nsite = pl.NSITE,
    max_tendon = pl.NTENDON, cone_type = ConeType.PYRAMIDAL,
    max_contacts=4, max_condim = pl.MAX_CONDIM,
    neq = pl.NEQ, max_equality = pl.NEQ,
    nexclude = pl.NEXCLUDE, timestep = pl.TIMESTEP,
]:
    return {}


def _md_s() -> ModelDefFromXML[
    xml=XML_S, nbody = ps.NBODY, njoint = ps.NJOINT, nq = ps.NQ, nv = ps.NV,
    ngeom = ps.NGEOM, nact = ps.NACT, ntex = ps.NTEX, nmat = ps.NMAT,
    nlight = ps.NLIGHT, ncam = ps.NCAM, nsite = ps.NSITE,
    max_tendon = ps.NTENDON, cone_type = ConeType.PYRAMIDAL,
    max_contacts=4, max_condim = ps.MAX_CONDIM,
    neq = ps.NEQ, max_equality = ps.NEQ,
    nexclude = ps.NEXCLUDE, timestep = ps.TIMESTEP,
]:
    return {}


comptime MC = _md_c()
comptime ML = _md_l()
comptime MS = _md_s()


def _mj_roll(
    xml: String, disable_equality: Bool, nsteps: Int
) raises -> Tuple[Float64, Float64]:
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(xml)
    if disable_equality:
        m.opt.disableflags = (
            Int(py=m.opt.disableflags)
            | Int(py=mujoco.mjtDisableBit.mjDSBL_EQUALITY)
        )
    var dat = mujoco.MjData(m)
    for _ in range(nsteps):
        mujoco.mj_step(m, dat)
    return (Float64(py=dat.qpos[0]), Float64(py=dat.qpos[1]))


def test_the_fixtures_are_not_vacuous() raises:
    """MuJoCo must build ONE equality row per fixture, and it must bite."""
    print("--- eq joint: the fixtures discriminate ---")
    var mujoco = Python.import_module("mujoco")
    for tag_xml in [
        ("cubic ", materialize[XML_C]()),
        ("linear", materialize[XML_L]()),
        ("single", materialize[XML_S]()),
    ]:
        var m = mujoco.MjModel.from_xml_string(tag_xml[1])
        var dat = mujoco.MjData(m)
        mujoco.mj_forward(m, dat)
        assert_true(
            Int(py=m.neq) == 1,
            tag_xml[0] + ": expected exactly one equality",
        )
        # mjEQ_JOINT == 2
        assert_true(
            Int(py=m.eq_type[0]) == 2,
            tag_xml[0] + ": MuJoCo did not compile this as mjEQ_JOINT",
        )
        assert_true(
            Int(py=dat.nefc) == 1 and Int(py=dat.ncon) == 0,
            tag_xml[0] + ": expected exactly the equality's single row and no"
            " contacts; got nefc " + String(Int(py=dat.nefc)),
        )
        var on = _mj_roll(tag_xml[1], False, NSTEPS)
        var off = _mj_roll(tag_xml[1], True, NSTEPS)
        print(
            "  ", tag_xml[0], " q = (", on[0], ",", on[1], ")  uncoupled = (",
            off[0], ",", off[1], ")",
        )
        assert_true(
            abs(on[0] - off[0]) > 1e-3,
            tag_xml[0] + ": the equality moves j1 by less than 1e-3 — an"
            " engine that ignored mjEQ_JOINT entirely would pass",
        )


def _check_rows[M: ModelDefFromXML](
    xml: String, label: String, q1: Float64, q2: Float64
) raises:
    """Build our single row at a perturbed pose and diff against `efc_*`."""
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
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(xml)
    var dat = mujoco.MjData(m)
    dat.qpos[0] = q1
    dat.qpos[1] = q2
    mujoco.mj_forward(m, dat)
    assert_true(Int(py=dat.nefc) == 1, label + ": expected 1 efc row")

    var ctx = DeviceContext()
    var mf = Model[DTYPE, MD]()
    M.init_fields[DTYPE](ctx, mf)
    assert_true(
        M.MAX_EQUALITY == 1,
        label + ": MAX_EQUALITY is not 1 — the equality slab is unsized and"
        " every comparison here would read zeros",
    )
    # EQ_JOINT == 2, and it must equal MuJoCo's own eq_type.
    assert_true(
        Int(mf.equality.data[EQ_IDX_TYPE]) == Int(py=m.eq_type[0]),
        label + ": our EQ_IDX_TYPE does not match MuJoCo's eq_type",
    )

    var d = Data[DTYPE, MD, 1]()
    M.reset_data[DTYPE](sf, d)
    d.qpos.data[0] = q1
    d.qpos.data[1] = q2
    for i in range(M.NV):
        d.qvel.data[i] = 0

    var sc = DynamicsScratch[DTYPE, MD, 1]()
    forward_kinematics["cpu"](d, mf, None)
    compute_body_velocities["cpu"](d, mf, None)
    compute_subtree_com["cpu"](d, mf, None)
    compute_cdof["cpu"](d, mf, sc, None)
    compute_mass_matrix["cpu"](d, mf, sc, None)
    ldl_factor["cpu", DTYPE, BATCH=1](mf, sc, None)
    _compute_m_inv["cpu", DTYPE, BATCH=1](mf, sc, None)

    comptime WR = 6 * cap[M.MAX_EQUALITY]()
    comptime WJ = 6 * cap[M.MAX_EQUALITY]() * cap[M.NV]()
    var w_K = Scratch[Scalar[DTYPE], WR](6 * M.MAX_EQUALITY, Scalar[DTYPE](1))
    var w_bias = Scratch[Scalar[DTYPE], WR](6 * M.MAX_EQUALITY, Scalar[DTYPE](0))
    var w_D = Scratch[Scalar[DTYPE], WR](6 * M.MAX_EQUALITY, Scalar[DTYPE](0))
    var w_J = Scratch[Scalar[DTYPE], WJ](6 * M.MAX_EQUALITY * M.NV, Scalar[DTYPE](0))
    var w_MinvJ = Scratch[Scalar[DTYPE], WJ](6 * M.MAX_EQUALITY * M.NV, Scalar[DTYPE](0))

    comptime L_B3 = RowMajor[1, M.NBODY * 3]
    comptime L_B4 = RowMajor[1, M.NBODY * 4]
    comptime L_NV = RowMajor[1, M.NV]
    comptime L_NQ = RowMajor[1, M.NQ]
    comptime L_DW = RowMajor[M.NV]
    comptime L_JT = RowMajor[M.NJOINT, MODEL_JOINT_SIZE]
    comptime L_BD = RowMajor[M.NBODY, MODEL_BODY_SIZE]
    comptime L_MT = RowMajor[MODEL_META_SIZE]
    comptime L_EQ = RowMajor[M.MAX_EQUALITY, MODEL_EQ_SIZE]
    comptime L_IW = RowMajor[M.NBODY, 2]
    comptime L_CD = RowMajor[1, M.NV * 6]
    comptime L_MI = RowMajor[1, M.NV * M.NV]

    var n = build_weld_equality_rows[DTYPE, M.NV](
        0,
        AsStatic[MD](),
        d.qpos.tt["cpu", 1, M.NQ](),
        d.qvel.tt["cpu", 1, M.NV](),
        d.xpos.tt["cpu", 1, M.NBODY * 3](),
        d.xquat.tt["cpu", 1, M.NBODY * 4](),
        d.subtree_com.tt["cpu", 1, M.NBODY * 3](),
        mf.joints.tt["cpu", M.NJOINT, MODEL_JOINT_SIZE](),
        mf.bodies.tt["cpu", M.NBODY, MODEL_BODY_SIZE](),
        mf.meta.tt["cpu", MODEL_META_SIZE](),
        mf.equality.tt["cpu", M.MAX_EQUALITY, MODEL_EQ_SIZE](),
        mf.body_invweight0.tt["cpu", M.NBODY, 2](),
        mf.dof_invweight0.tt["cpu", M.NV](),
        sc.cdof.tt["cpu", 1, M.NV * 6](),
        sc.m_inv.tt["cpu", 1, M.NV * M.NV](),
        w_K, w_bias, w_D, w_J, w_MinvJ,
    )
    assert_true(n == 1, label + ": expected 1 joint row, built " + String(n))

    var efc = dat.efc_J.reshape(1, Int(py=m.nv))
    var wJ = Float64(0)
    var joff = Float64(0)
    for j in range(M.NV):
        var ours = Float64(w_J[j])
        var theirs = Float64(py=efc[0][j])
        if abs(ours - theirs) > wJ:
            wJ = abs(ours - theirs)
        if j == 1 and abs(theirs) > joff:
            joff = abs(theirs)
    print("  ", label, "worst |d(J)| =", wJ, "  |J[dof2]| =", joff)
    assert_true(wJ < 1e-12, label + ": joint J disagrees by " + String(wJ))

    var theirs_b = -Float64(py=dat.efc_aref[0])
    var wB = abs(Float64(w_bias[0]) - theirs_b)
    print("  ", label, "|d(bias vs -aref)| =", wB, "  -aref =", theirs_b)
    assert_true(
        abs(theirs_b) > 1e-3,
        label + ": the residual is ~zero at this pose, so a row can match J"
        " and aref while its impedance is wrong — perturb further",
    )
    assert_true(wB < 1e-9, label + ": joint bias disagrees by " + String(wB))

    var R = 1.0 / Float64(w_D[0]) - Float64(w_K[0])
    var ourD = 1.0 / R
    var theirD = Float64(py=dat.efc_D[0])
    var relD = abs(ourD - theirD) / (abs(theirD) if abs(theirD) > 1e-12 else 1.0)
    print("  ", label, "rel |d(D)| =", relD)
    assert_true(relD < 1e-9, label + ": joint efc_D disagrees by rel " + String(relD))


def test_cubic_polycoef_matches_mujoco() raises:
    """The full quartic: p2 and p3 nonzero, so `deriv` varies with q2."""
    print("--- eq joint rows vs efc, CUBIC polycoef ---")
    _check_rows[MC](materialize[XML_C](), "cubic", 0.37, -0.44)


def test_linear_polycoef_matches_mujoco() raises:
    """ToddlerBot's own shape — `polycoef="0 c 0 0 0"`."""
    print("--- eq joint rows vs efc, LINEAR polycoef ---")
    _check_rows[ML](materialize[XML_L](), "linear", 0.37, -0.44)


def test_single_joint_form_matches_mujoco() raises:
    """`joint2` omitted: MuJoCo drops the polynomial, J is e_dof1 alone."""
    print("--- eq joint rows vs efc, SINGLE joint ---")
    _check_rows[MS](materialize[XML_S](), "single", 0.37, -0.44)


def _our_roll[M: ModelDefFromXML]() raises -> Tuple[Float64, Float64]:
    comptime MD_2 = Dims[
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
    var ctx = DeviceContext()
    var mf = Model[DTYPE, MD_2]()
    M.init_fields[DTYPE](ctx, mf)
    var d = Data[DTYPE, MD_2, 1]()
    M.reset_data[DTYPE](sf, d)
    forward_kinematics["cpu"](d, mf)
    var integ = EulerIntegrator[DTYPE, MD_2, M.CONE_TYPE, 1, SOLVER="newton", MAX_CONDIM = M.MAX_CONDIM, NOSLIP_ITER = M.NOSLIP_ITER]()
    # ⚠ CONTACTS=True is load-bearing: the constraint seam only runs on that
    # branch, so with CONTACTS=False this returns the uncoupled answer no
    # matter what the solvers do.
    for _ in range(NSTEPS):
        integ.step["cpu", CONTACTS=True](d, mf)
    return (Float64(d.qpos.data[0]), Float64(d.qpos.data[1]))


def test_cubic_rollout_matches_mujoco() raises:
    print("--- eq joint rollout, CUBIC ---")
    var ours = _our_roll[MC]()
    var theirs = _mj_roll(materialize[XML_C](), False, NSTEPS)
    var off = _mj_roll(materialize[XML_C](), True, NSTEPS)
    print("  ours =", ours[0], ours[1])
    print("  MuJoCo =", theirs[0], theirs[1], "  uncoupled =", off[0], off[1])
    var e0 = abs(ours[0] - theirs[0])
    var e1 = abs(ours[1] - theirs[1])
    assert_true(
        e0 < 1e-6 and e1 < 1e-6,
        "cubic rollout disagrees by " + String(e0) + " / " + String(e1),
    )
    assert_true(
        abs(ours[0] - off[0]) > 10.0 * e0,
        "our rollout is no closer to MuJoCo's COUPLED answer than to its"
        " uncoupled one — the joint equality is not being applied",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
