"""`<equality><connect>` — mjEQ_CONNECT, BODY and SITE semantics.

THE GAP THIS CLOSES. Until 2026-08-12 `_fill_equality` RAISED on `<connect>`
rather than parsing it, because `anchor_b` — MuJoCo's `eq_data[3:6]`, the
shared anchor expressed in body2's frame — was never populated, so body2 would
have been anchored at its own origin. That raise was the right call (loud beats
latent) but it left every model with a connect unloadable, ToddlerBot included.

TWO HALVES, BOTH REQUIRED. MJCF spells a connect two mutually exclusive ways
and `mj_instantiateEquality` (engine_core_constraint.c:448) branches on
`eq_objtype`:

  BODY:  pos[j] = xpos[body_j] + xmat[body_j] * eq_data[3j : 3j+3]
         `eq_data[0:3]` is the MJCF `anchor` (body1's frame); `eq_data[3:6]`
         is DERIVED at qpos0 by `mj_setConst`.
  SITE:  pos[j] = site_xpos[site_j], bodies from `site_bodyid`, and
         `eq_data` is zeroed and never read.

We store the site form REDUCED to the body form — `(site_bodyid, site local
pos)` — because FK defines `site_xpos = xpos[b] + xmat[b] * site_pos`, which
is the same expression. That leaves the row builder and every solver path
untouched, and is why this file gates the two spellings against each other as
well as against MuJoCo.

⚠ THE DERIVATION IS SKIPPED ON THE SITE FORM and would otherwise overwrite
site2's offset with an anchor MuJoCo never computes.
`test_site_connect_leaves_eq_data_alone` is what catches that: it pins both
anchor slots to the two sites' local `pos`, which a leaked derivation moves.

WHY THIS FIXTURE. The bob's only support is the connect — no floor, both geoms
have `contype=0 conaffinity=0`, no limits, no springs. Nothing else in the
engine can hold it up, so "held" and "free fall" are metres apart rather than a
tolerance argument. `arm` carries a 45-degree `quat` so that `R_b^T` in the
anchor derivation is NOT the identity: with an axis-aligned body2 the whole
derivation degenerates to a subtraction and a transposed rotation that is never
applied would pass.

⚠ ROWS, NOT A LONG ROLLOUT — and the horizon below is short DELIBERATELY.
Measured ours-vs-MuJoCo |d(qpos)|_inf on the BODY fixture:

    step    1     1.6e-18      <- bit-exact
    step    2     9.1e-09      <- the iterative solve's own tolerance
    step    5     4.0e-07
    step   10     5.4e-06
    step   20     6.0e-05
    step   60     2.7e-03
    step  240     3.6e-01

Step 1 is exact and step 2 lands on MuJoCo's default solver tolerance (1e-8);
everything after is that seed amplified geometrically by a swinging pendulum.
Lengthening `NSTEPS_ROLL` therefore measures Lyapunov growth, not the
constraint, and no tolerance that "fixes" a longer horizon means anything.
`efc_J` / `efc_aref` / `efc_D` are pose-local and exact — they are what says
the rows are right. The rollout only proves the constraint is wired into the
integrator at all.

Run with:
    pixi run mojo run -I . tests/physics3d/test_connect_equality_vs_mujoco.mojo
"""

from std.math import abs
from std.python import Python
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext
from noeira.core.tiles import RowMajor

from noeira.physics3d.parser import parse_xml, ModelDefFromXML
from noeira.physics3d.parser.xml_parser import merge_mjcf
from noeira.physics3d.fields import Model, Data, DynamicsScratch, Dims, DimsLike, AsStatic
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
from noeira.physics3d.fields import Scratch, cap
from noeira.physics3d.types import _max_one, ConeType
from noeira.physics3d.gpu.constants import (
    EQ_IDX_SOLREF_0,
    EQ_IDX_SOLREF_1,
    EQ_IDX_SOLIMP_0,
    EQ_IDX_SOLIMP_1,
    EQ_IDX_SOLIMP_2,
    MODEL_JOINT_SIZE,
    MODEL_EQ_SIZE,
    MODEL_BODY_SIZE,
    MODEL_META_SIZE,
    EQ_IDX_OBJTYPE,
    EQ_IDX_ANCHOR_AX,
    EQ_IDX_ANCHOR_AY,
    EQ_IDX_ANCHOR_AZ,
    EQ_IDX_ANCHOR_BX,
    EQ_IDX_ANCHOR_BY,
    EQ_IDX_ANCHOR_BZ,
)

comptime DTYPE = DType.float64
# Ours-vs-MuJoCo rollout horizon. SHORT ON PURPOSE — see the growth table in
# the module docstring. `NSTEPS_DISC` is MuJoCo-against-MuJoCo only, so it can
# be long without any chaos concern.
comptime NSTEPS_ROLL = 5
comptime NSTEPS_DISC = 240

# 45 degrees about +y, so body2's frame is genuinely rotated.
comptime _ARM_QUAT = "0.9238795325112867 0 0.3826834323650898 0"

comptime _BODIES = (
    """
  <option timestep="0.002" gravity="0 0 -9.81" solver="Newton"/>
  <worldbody>
    <body name="arm" pos="0 0 1" quat=\""""
    + _ARM_QUAT
    + """\">
      <joint name="arm_hinge" type="hinge" axis="0 1 0"/>
      <geom name="g_arm" type="capsule" fromto="0 0 0 0.3 0 0" size="0.03"
            mass="1" contype="0" conaffinity="0"/>
      <site name="s_arm" pos="0.4828326112068523 0.057 0.15656349186104051"
            size="0.01"/>
    </body>
    <body name="bob" pos="0.4 0 0.7">
      <joint name="bob_free" type="free"/>
      <geom name="g_bob" type="sphere" size="0.05" mass="1"
            contype="0" conaffinity="0"/>
      <site name="s_bob" pos="0.05 0.06 0.07" size="0.01"/>
    </body>
  </worldbody>
"""
)

# BODY spelling: anchor lives in bob's (body1's) frame.
comptime _RAW_BODY = (
    '<mujoco model="connect_body">'
    + _BODIES
    + """
  <equality>
    <connect body1="bob" body2="arm" anchor="0.05 0.06 0.07"/>
  </equality>
</mujoco>
"""
)

# SITE spelling. `s_bob` sits at the same body-local offset as the BODY
# fixture's `anchor`, and `s_arm` sits at the point that offset maps to on the
# arm at qpos0 — PLUS a deliberate few-millimetre offset of
# (+0.002, -0.003, +0.001).
#
# ⚠ THAT OFFSET IS LOAD-BEARING, IN BOTH DIRECTIONS. Without it the fixture
# would be exactly consistent at qpos0, and then the anchor a leaked qpos0
# derivation would write is EXACTLY `s_arm`'s local pos — so
# `test_site_connect_leaves_eq_data_alone` would pass with the leak present.
# Made much larger (it was 0.26 m at first) the constraint snaps at t = 0 and
# the rollout is pure chaos. A few mm discriminates by twelve orders against a
# 1e-15 assert while leaving the dynamics mild.
comptime _RAW_SITE = (
    '<mujoco model="connect_site">'
    + _BODIES
    + """
  <equality>
    <connect site1="s_bob" site2="s_arm" solref="0.004 1"
      solimp="0.9999 0.9999 0.001 0.5 2"/>
  </equality>
</mujoco>
"""
)

comptime XML_BODY = merge_mjcf(_RAW_BODY)
comptime XML_SITE = merge_mjcf(_RAW_SITE)
comptime pmb = parse_xml(XML_BODY)
comptime pms = parse_xml(XML_SITE)


def _model_body() -> ModelDefFromXML[
    xml=XML_BODY,
    nbody = pmb.NBODY,
    njoint = pmb.NJOINT,
    nq = pmb.NQ,
    nv = pmb.NV,
    ngeom = pmb.NGEOM,
    nact = pmb.NACT,
    ntex = pmb.NTEX,
    nmat = pmb.NMAT,
    nlight = pmb.NLIGHT,
    ncam = pmb.NCAM,
    nsite = pmb.NSITE,
    max_tendon = pmb.NTENDON,
    cone_type = ConeType.PYRAMIDAL,
    max_contacts=4,
    max_condim = pmb.MAX_CONDIM,
    neq = pmb.NEQ,
    # ⚠ `neq` ALONE SIZES NOTHING. `max_equality` is what allocates the
    # equality slab; passing only `neq` leaves MAX_EQUALITY at 0, the connect
    # vanishes, and every ours-vs-ours comparison below reads 0.0 and passes.
    max_equality = pmb.NEQ,
    nexclude = pmb.NEXCLUDE,
    timestep = pmb.TIMESTEP,
]:
    return {}


def _model_site() -> ModelDefFromXML[
    xml=XML_SITE,
    nbody = pms.NBODY,
    njoint = pms.NJOINT,
    nq = pms.NQ,
    nv = pms.NV,
    ngeom = pms.NGEOM,
    nact = pms.NACT,
    ntex = pms.NTEX,
    nmat = pms.NMAT,
    nlight = pms.NLIGHT,
    ncam = pms.NCAM,
    nsite = pms.NSITE,
    max_tendon = pms.NTENDON,
    cone_type = ConeType.PYRAMIDAL,
    max_contacts=4,
    max_condim = pms.MAX_CONDIM,
    neq = pms.NEQ,
    max_equality = pms.NEQ,
    nexclude = pms.NEXCLUDE,
    timestep = pms.TIMESTEP,
]:
    return {}


comptime MB = _model_body()
comptime MD = Dims[
    nq=MB.NQ,
    nv=MB.NV,
    nbody=MB.NBODY,
    njoint=MB.NJOINT,
    ngeom=MB.NGEOM,
    nsite=MB.NSITE,
    max_contacts=MB.MAX_CONTACTS,
    nequality=MB.MAX_EQUALITY,
    ntendon=MB.MAX_TENDON,
    nexclude=MB.NEXCLUDE,
    nmesh_verts=0,
    npair=MB.NPAIR,
    nact=MB.NACT,
    nten=MB.NTEN_F,
    nkey=MB.NKEY,
]
comptime MS = _model_site()
comptime MD_2 = Dims[
    nq=MS.NQ,
    nv=MS.NV,
    nbody=MS.NBODY,
    njoint=MS.NJOINT,
    ngeom=MS.NGEOM,
    nsite=MS.NSITE,
    max_contacts=MS.MAX_CONTACTS,
    nequality=MS.MAX_EQUALITY,
    ntendon=MS.MAX_TENDON,
    nexclude=MS.NEXCLUDE,
    nmesh_verts=0,
    npair=MS.NPAIR,
    nact=MS.NACT,
    nten=MS.NTEN_F,
    nkey=MS.NKEY,
]


# =============================================================================
# The fixture must discriminate
# =============================================================================


def test_the_fixture_is_not_vacuous() raises:
    """MuJoCo must build exactly the connect's 3 rows and must hold the bob.

    Without this the file could pass while the constraint does nothing — the
    state the engine was in for every other unimplemented equality in this
    arc.
    """
    print("--- connect: the fixture discriminates ---")
    var mujoco = Python.import_module("mujoco")

    for tag_xml in [
        ("BODY", materialize[XML_BODY]()),
        ("SITE", materialize[XML_SITE]()),
    ]:
        var m = mujoco.MjModel.from_xml_string(tag_xml[1])
        var dat = mujoco.MjData(m)
        mujoco.mj_forward(m, dat)
        var nefc = Int(py=dat.nefc)
        var ncon = Int(py=dat.ncon)
        print("  ", tag_xml[0], " nefc =", nefc, " ncon =", ncon)
        assert_true(
            ncon == 0,
            "the fixture generated contacts — contype/conaffinity no longer"
            " disable them and the row attribution below is not clean",
        )
        assert_true(
            nefc == 3,
            "expected exactly the connect's THREE rows; got " + String(nefc),
        )
        var want_eq = Int(py=mujoco.mjtConstraint.mjCNSTR_EQUALITY)
        for r in range(3):
            assert_true(
                Int(py=dat.efc_type[r]) == want_eq,
                "row is not mjCNSTR_EQUALITY — the fixture stopped expressing"
                " a connect",
            )

        var held = _mujoco_roll(tag_xml[1], False, NSTEPS_DISC)
        var free = _mujoco_roll(tag_xml[1], True, NSTEPS_DISC)
        print("    MuJoCo held z =", held, "  free fall z =", free)
        assert_true(
            abs(held - free) > 0.5,
            "held and free fall are within 50 cm at the long horizon — the"
            " fixture has stopped discriminating and an engine that ignores"
            " the connect entirely would pass",
        )

        # The SHORT horizon the rollout tests actually use must discriminate
        # too, by a comfortable multiple of their 1e-6 tolerance — otherwise
        # those tests would pass on free fall.
        var held_s = _mujoco_roll(tag_xml[1], False, NSTEPS_ROLL)
        var free_s = _mujoco_roll(tag_xml[1], True, NSTEPS_ROLL)
        print(
            "    at NSTEPS_ROLL: held =", held_s, " free =", free_s,
            " gap =", abs(held_s - free_s),
        )
        # Measured: body 3.2e-05, site 1.7e-04, against a 1e-5 rollout
        # tolerance. Thin in absolute terms because the horizon is 5 steps —
        # the decisive discriminator is the RATIO assert in each rollout test
        # (`|ours - free| > 10 * |ours - held|`), which measures 2300x on the
        # body fixture and 212x on the site one and adapts to the horizon.
        # This assert only catches the fixture degenerating into free fall.
        assert_true(
            abs(held_s - free_s) > 1e-5,
            "at NSTEPS_ROLL the constraint moves the bob by less than the"
            " rollout tests' own 1e-5 tolerance — those tests would pass on"
            " free fall",
        )


def _mujoco_roll(
    xml: String, disable_equality: Bool, nsteps: Int
) raises -> Float64:
    """Step MuJoCo `nsteps` and return the bob's world z."""
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
    # qpos layout: [arm_hinge, bob free (x y z qw qx qy qz)]
    return Float64(py=dat.qpos[3])


# =============================================================================
# eq_data: the derived body2-side anchor
# =============================================================================


def test_connect_anchor_b_matches_mujoco() raises:
    """Our derived `anchor_b` vs MuJoCo's `eq_data[3:6]`, BODY semantics.

    This is the value whose absence was the reason `<connect>` raised. MuJoCo
    derives it in `mj_setConst` ("compute missing eq_data for body
    constraints") as `xmat[b2]^T * (world_anchor - xpos[b2])`, and the code is
    byte-identical in 3.3.6, 3.6.0, 3.11.0 and `mujoco-main` — but the runtime
    is 3.10.0, which matches none of those trees, so the number below comes
    from the runtime and not from a tree.
    """
    print("--- connect: derived anchor_b vs eq_data[3:6] ---")
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(materialize[XML_BODY]())
    assert_true(Int(py=m.neq) == 1, "expected exactly one equality")

    var ctx = DeviceContext()
    var mf = Model[DTYPE, MD]()
    MB.init_fields[DTYPE](ctx, mf)

    assert_true(
        MB.MAX_EQUALITY == 1,
        "MAX_EQUALITY is "
        + String(MB.MAX_EQUALITY)
        + ", not 1 — the equality slab is unsized and every comparison in this"
        " file would read zeros",
    )

    var ours_x = Float64(mf.equality.data[EQ_IDX_ANCHOR_BX])
    var ours_y = Float64(mf.equality.data[EQ_IDX_ANCHOR_BY])
    var ours_z = Float64(mf.equality.data[EQ_IDX_ANCHOR_BZ])
    var th_x = Float64(py=m.eq_data[0][3])
    var th_y = Float64(py=m.eq_data[0][4])
    var th_z = Float64(py=m.eq_data[0][5])
    print("  ours   anchor_b =", ours_x, ours_y, ours_z)
    print("  MuJoCo anchor_b =", th_x, th_y, th_z)

    # A derivation that forgot the transpose, or applied no rotation at all,
    # lands somewhere else entirely — pin that the rotation is doing work.
    var plain_x = Float64(py=m.eq_data[0][0])
    assert_true(
        abs(th_x - plain_x) > 0.1,
        "MuJoCo's derived anchor_b is within 0.1 of the RAW anchor at this"
        " pose, so a derivation that copied `anchor` across would pass — tilt"
        " or move the fixture",
    )

    var w = abs(ours_x - th_x)
    if abs(ours_y - th_y) > w:
        w = abs(ours_y - th_y)
    if abs(ours_z - th_z) > w:
        w = abs(ours_z - th_z)
    print("  worst |d| =", w)
    assert_true(w < 1e-12, "connect anchor_b disagrees by " + String(w))

    assert_true(
        Int(mf.equality.data[EQ_IDX_OBJTYPE]) == 0,
        "BODY-spelled connect did not record EQ_OBJ_BODY",
    )


def test_site_connect_leaves_eq_data_alone() raises:
    """SITE semantics: anchors are the site offsets, NOT a derived value.

    MuJoCo zeroes `eq_data` for a site-based connect and reads `site_xpos`; we
    store the site offsets in the anchor slots instead. The qpos0 derivation
    must therefore SKIP this row — if it ran, it would overwrite site2's
    offset with `xmat[b2]^T (world_anchor - xpos[b2])`, which for this fixture
    is a different point.
    """
    print("--- connect: site semantics keeps the site offsets ---")
    var ctx = DeviceContext()
    var mf = Model[DTYPE, MD_2]()
    MS.init_fields[DTYPE](ctx, mf)

    assert_true(
        Int(mf.equality.data[EQ_IDX_OBJTYPE]) == 1,
        "SITE-spelled connect did not record EQ_OBJ_SITE — the qpos0"
        " derivation will overwrite site2's offset",
    )

    # s_bob pos="0.05 0.06 0.07"; s_arm pos is the mapped anchor plus the
    # deliberate few-mm offset (see the fixture comment).
    var ax = Float64(mf.equality.data[EQ_IDX_ANCHOR_AX])
    var ay = Float64(mf.equality.data[EQ_IDX_ANCHOR_AY])
    var az = Float64(mf.equality.data[EQ_IDX_ANCHOR_AZ])
    var bx = Float64(mf.equality.data[EQ_IDX_ANCHOR_BX])
    var by = Float64(mf.equality.data[EQ_IDX_ANCHOR_BY])
    var bz = Float64(mf.equality.data[EQ_IDX_ANCHOR_BZ])
    print("  anchor_a =", ax, ay, az, "  anchor_b =", bx, by, bz)
    assert_true(
        abs(ax - 0.05) < 1e-15
        and abs(ay - 0.06) < 1e-15
        and abs(az - 0.07) < 1e-15,
        "anchor_a is not s_bob's local pos",
    )
    assert_true(
        abs(bx - 0.4828326112068523) < 1e-15
        and abs(by - 0.057) < 1e-15
        and abs(bz - 0.15656349186104051) < 1e-15,
        "anchor_b is not s_arm's local pos — the qpos0 derivation leaked into"
        " the site path and overwrote it with the mapped anchor",
    )

    # The site fixture's `<connect>` is written the way ToddlerBot writes its
    # four — attributes split ACROSS LINES, with explicit solref/solimp. A tag
    # scanner that stopped at the newline instead of the closing `>` would
    # drop them and silently fall back to the MJCF defaults (0.02 / 1 and
    # 0.9 / 0.95), which is a much softer constraint than 0.004 / 0.9999.
    var mujoco = Python.import_module("mujoco")
    var m = mujoco.MjModel.from_xml_string(materialize[XML_SITE]())
    var sr0 = Float64(mf.equality.data[EQ_IDX_SOLREF_0])
    var sr1 = Float64(mf.equality.data[EQ_IDX_SOLREF_1])
    var si0 = Float64(mf.equality.data[EQ_IDX_SOLIMP_0])
    var si1 = Float64(mf.equality.data[EQ_IDX_SOLIMP_1])
    var si2 = Float64(mf.equality.data[EQ_IDX_SOLIMP_2])
    print("  solref =", sr0, sr1, "  solimp[0:3] =", si0, si1, si2)
    assert_true(
        abs(sr0 - Float64(py=m.eq_solref[0][0])) < 1e-15
        and abs(sr1 - Float64(py=m.eq_solref[0][1])) < 1e-15,
        "solref did not survive the multi-line tag — got "
        + String(sr0) + " " + String(sr1),
    )
    assert_true(
        abs(si0 - Float64(py=m.eq_solimp[0][0])) < 1e-15
        and abs(si1 - Float64(py=m.eq_solimp[0][1])) < 1e-15
        and abs(si2 - Float64(py=m.eq_solimp[0][2])) < 1e-15,
        "solimp did not survive the multi-line tag — got "
        + String(si0) + " " + String(si1) + " " + String(si2),
    )


# =============================================================================
# The rows themselves
# =============================================================================


def _check_rows[M: ModelDefFromXML](xml: String, label: String) raises:
    """Build our 3 connect rows at a perturbed pose and diff vs `efc_*`."""
    comptime MD_3 = Dims[
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
    var np = Python.import_module("numpy")
    var m = mujoco.MjModel.from_xml_string(xml)
    var dat = mujoco.MjData(m)

    # Perturb off qpos0 so the residual is nonzero and the rows are not
    # trivially satisfied. qpos: [hinge, bob x y z qw qx qy qz]
    var th = 0.3
    dat.qpos[0] = 0.21
    dat.qpos[1] = 0.44
    dat.qpos[2] = 0.03
    dat.qpos[3] = 0.66
    dat.qpos[4] = Float64(py=np.cos(th / 2))
    dat.qpos[5] = 0.0
    dat.qpos[6] = Float64(py=np.sin(th / 2))
    dat.qpos[7] = 0.0
    mujoco.mj_forward(m, dat)
    var nefc = Int(py=dat.nefc)
    assert_true(
        nefc == 3, "expected the connect's 3 rows; got " + String(nefc)
    )

    var ctx = DeviceContext()
    var mf = Model[DTYPE, MD_3]()
    M.init_fields[DTYPE](ctx, mf)
    var d = Data[DTYPE, MD_3, 1]()
    M.reset_data[DTYPE](sf, d)
    d.qpos.data[0] = 0.21
    d.qpos.data[1] = 0.44
    d.qpos.data[2] = 0.03
    d.qpos.data[3] = 0.66
    d.qpos.data[4] = Float64(py=np.cos(th / 2))
    d.qpos.data[5] = 0.0
    d.qpos.data[6] = Float64(py=np.sin(th / 2))
    d.qpos.data[7] = 0.0
    for i in range(M.NV):
        d.qvel.data[i] = 0

    var sc = DynamicsScratch[DTYPE, MD_3, 1]()
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
    assert_true(
        n == 3,
        label + ": expected 3 connect rows, built " + String(n),
    )

    var efc = dat.efc_J.reshape(nefc, Int(py=m.nv))
    var wJ = Float64(0)
    var jmag = Float64(0)
    for r in range(3):
        for j in range(M.NV):
            var ours = Float64(w_J[r * M.NV + j])
            var theirs = Float64(py=efc[r][j])
            if abs(ours - theirs) > wJ:
                wJ = abs(ours - theirs)
            if abs(theirs) > jmag:
                jmag = abs(theirs)
    print("  ", label, "worst |d(J)| =", wJ, " max|J| =", jmag)
    assert_true(
        jmag > 0.1,
        label + ": MuJoCo's Jacobian is ~zero, so an all-zero J would pass",
    )
    assert_true(wJ < 1e-12, label + ": connect J disagrees by " + String(wJ))

    # `bias` is MuJoCo's -aref (qvel = 0 here, so it is K*imp*pos).
    var wB = Float64(0)
    var bmag = Float64(0)
    for r in range(3):
        var theirs = -Float64(py=dat.efc_aref[r])
        if abs(Float64(w_bias[r]) - theirs) > wB:
            wB = abs(Float64(w_bias[r]) - theirs)
        if abs(theirs) > bmag:
            bmag = abs(theirs)
    print("  ", label, "worst |d(bias vs -aref)| =", wB, " max|aref| =", bmag)
    assert_true(
        bmag > 1e-3,
        label + ": the residual is ~zero at this pose — a row can match J and"
        " aref and still be wrong when the residual vanishes, so perturb the"
        " fixture further",
    )
    assert_true(wB < 1e-9, label + ": connect bias disagrees by " + String(wB))

    # D: the builder returns the PGS step size 1/(k+R); recover R the way the
    # Newton paths do and compare 1/R against MuJoCo's efc_D. This is the one
    # that exposes a wrong impedance — J and aref can both match with it wrong.
    var wD = Float64(0)
    for r in range(3):
        var R = 1.0 / Float64(w_D[r]) - Float64(w_K[r])
        var ours = 1.0 / R
        var theirs = Float64(py=dat.efc_D[r])
        var den = abs(theirs) if abs(theirs) > 1e-12 else 1.0
        if abs(ours - theirs) / den > wD:
            wD = abs(ours - theirs) / den
    print("  ", label, "worst rel |d(D)| =", wD)
    assert_true(wD < 1e-9, label + ": connect efc_D disagrees by rel " + String(wD))


def test_body_connect_rows_match_mujoco() raises:
    print("--- connect rows vs efc, BODY semantics ---")
    _check_rows[MB](materialize[XML_BODY](), "body")


def test_site_connect_rows_match_mujoco() raises:
    print("--- connect rows vs efc, SITE semantics ---")
    _check_rows[MS](materialize[XML_SITE](), "site")


# =============================================================================
# Rollout
# =============================================================================


def _our_roll[M: ModelDefFromXML]() raises -> Float64:
    """Step our engine `NSTEPS` and return the bob's world z (qpos[3]).

    ⚠ `CONTACTS=True` is load-bearing: the constraint seam only runs on that
    branch, so with `CONTACTS=False` this returns free fall no matter what the
    solvers do.
    """
    comptime MD_4 = Dims[
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
    var mf = Model[DTYPE, MD_4]()
    M.init_fields[DTYPE](ctx, mf)
    var d = Data[DTYPE, MD_4, 1]()
    M.reset_data[DTYPE](sf, d)
    forward_kinematics["cpu"](d, mf)

    var integ = EulerIntegrator[DTYPE, MD_4, M.CONE_TYPE, 1, SOLVER="newton", MAX_CONDIM = M.MAX_CONDIM, NOSLIP_ITER = M.NOSLIP_ITER]()
    for _ in range(NSTEPS_ROLL):
        integ.step["cpu", CONTACTS=True](d, mf)
    return Float64(d.qpos.data[3])


def test_body_connect_rollout_matches_mujoco() raises:
    print("--- connect rollout, BODY semantics ---")
    var ours = _our_roll[MB]()
    var theirs = _mujoco_roll(materialize[XML_BODY](), False, NSTEPS_ROLL)
    var free = _mujoco_roll(materialize[XML_BODY](), True, NSTEPS_ROLL)
    print("  ours =", ours, "  MuJoCo =", theirs, "  free fall =", free)
    # 1e-5, not 1e-6: the stiffer site fixture lands at ~8e-7 and a test that
    # passes at 0.8x its own tolerance flakes. The RATIO assert below is the
    # real discriminator — it adapts to the horizon instead of being tuned to
    # it. Measured here: body 1.4e-08, site 8.0e-07.
    assert_true(
        abs(ours - theirs) < 1e-5,
        "body connect rollout disagrees by " + String(abs(ours - theirs)),
    )
    assert_true(
        abs(ours - free) > 10.0 * abs(ours - theirs),
        "our rollout is no closer to MuJoCo's HELD answer than to its FREE"
        " FALL answer — the connect is not being applied in our integrator",
    )


def test_site_connect_rollout_matches_mujoco() raises:
    print("--- connect rollout, SITE semantics ---")
    var ours = _our_roll[MS]()
    var theirs = _mujoco_roll(materialize[XML_SITE](), False, NSTEPS_ROLL)
    var free = _mujoco_roll(materialize[XML_SITE](), True, NSTEPS_ROLL)
    print("  ours =", ours, "  MuJoCo =", theirs, "  free fall =", free)
    assert_true(
        abs(ours - theirs) < 1e-5,
        "site connect rollout disagrees by " + String(abs(ours - theirs)),
    )
    assert_true(
        abs(ours - free) > 10.0 * abs(ours - theirs),
        "our rollout is no closer to MuJoCo's HELD answer than to its FREE"
        " FALL answer — the connect is not being applied in our integrator",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
