"""`<option ccd_tolerance= ccd_iterations=>` — parsed, stored, AND consumed.

EPA's stopping rule used to be hardcoded at 1e-8 / 64 iterations. MuJoCo's are
`mjOption.ccd_tolerance` = 1e-6 and `.ccd_iterations` = 35, and a model that
set either was ignored outright. This gates the whole path:

  1. `parse_xml` (COMPTIME counter) reads both, and defaults to MuJoCo's when
     the attributes are absent;
  2. `<option>` values survive into model META, which is where the narrow
     phase reads them from — the RUNTIME parser, which is a different parser
     (`feedback_physics3d_two_parser_paths`);
  3. changing `ccd_tolerance` CHANGES A CONTACT. Without this the first two
     are satisfied by dead data: a value can be parsed and stored and read by
     nothing, which is exactly what `<geom solref>` did for months.

⚠⚠ 3 IS THE ONLY ONE THAT CANNOT BE FAKED, and it is deliberately asserted as
an INEQUALITY on a quantity, not as a frozen number. A golden here would pin
whatever EPA happens to produce at two tolerances and would go on passing if
the tolerance stopped reaching the solver — the assertion has to be "these two
differ", because that is the claim.

⚠ THE FIXTURE MUST BE A SMOOTH PAIR. `discreteGeoms` sends mesh/box/hfield
pairs (at zero margin, in double precision) to `mjMINVAL` instead, so
`ccd_tolerance` provably does nothing there — a mesh-vs-box fixture would
assert "changing the tolerance changes the answer" against code that ignores
it, and fail for the right reason but the wrong lesson. A CYLINDER against a box
is smooth on one side, which is enough, and needs no mesh asset.

Run with:
    pixi run mojo run -I . tests/physics3d/test_ccd_options_are_parsed_and_used.mojo
"""

from std.math import abs, sqrt
from noeira.core.tiles import RowMajor
from max.gpu.host import DeviceContext
from std.testing import assert_true, TestSuite

from noeira.physics3d.parser import parse_xml, ModelDefFromXML
from noeira.physics3d.fields import Model, Dims
from noeira.nn.core.tensor import TensorImpl
from noeira.physics3d.collision.ccd_workspace import L_CCD_WS1
from noeira.physics3d.collision.ccd_workspace_host import ccd_ws_alloc
from noeira.physics3d.collision.gjk import gjk_epa
from noeira.physics3d.constants import GEOM_BOX, GEOM_CYLINDER
from noeira.physics3d.model.model_dims import ModelDims
from noeira.physics3d.gpu.constants import (
    mesh_max_edge,
    MODEL_META_IDX_CCD_TOLERANCE,
    MODEL_META_IDX_CCD_ITERATIONS,
    MJ_CCD_TOLERANCE,
    MJ_CCD_ITERATIONS,
    CONTACT_SIZE,
    CONTACT_IDX_NX,
    CONTACT_IDX_DIST,
    META_IDX_NUM_CONTACTS,
)

comptime DTYPE = DType.float64
comptime NMV: Int = 4096

# Scratch mesh operand for the direct EPA call — unused (both geoms are
# primitives) but `gjk_epa` binds it unconditionally.
comptime NMV_EPA: Int = 8
comptime L_MV_EPA = RowMajor[NMV_EPA, 3]
# ⚠ Both geoms here are PRIMITIVES, so the mesh operand is never read; these
# just have to bind. All -1 is how `fields/model.mojo` marks "no hull graph",
# which is also the state `_support_mesh` falls back to its scan on.
comptime L_VEADR_EPA = RowMajor[NMV_EPA]
comptime L_EDGES_EPA = RowMajor[mesh_max_edge(NMV_EPA)]


def _no_graph_epa() raises -> TensorImpl[DTYPE]:
    var t = TensorImpl[DTYPE].alloc(NMV_EPA)
    for i in range(NMV_EPA):
        t.data[i] = -1.0
    return t^


def _no_edges_epa() raises -> TensorImpl[DTYPE]:
    var t = TensorImpl[DTYPE].alloc(mesh_max_edge(NMV_EPA))
    for i in range(mesh_max_edge(NMV_EPA)):
        t.data[i] = -1.0
    return t^

# MuJoCo 3.10.0 on the same cylinder/box pose, `mj_forward`: 3 contacts, all at
# this depth. Frozen so the "does the tolerance change anything" check cannot
# be satisfied by two equally wrong answers.
#
# Reproduce it with:
#
#     <geom name="g_box" type="box" size="0.05 0.04 0.03"/>
#     <body pos="0.030 0.020 0.038" quat="0.83 0.29 0.41 0.24">
#       <freejoint/><geom type="cylinder" size="0.03 0.02" mass="1"/>
#     </body>
#
# ⚠⚠ AND NOTE THE TWO THINGS MuJoCo DOES TO THAT FILE BEFORE EPA SEES IT,
# BECAUSE THIS TEST HAND-BUILDS THE CALL AND SO HAS TO DO THEM ITSELF:
#
#   1. IT NORMALIZES THE QUATERNION. `0.83 0.29 0.41 0.24` has norm 0.9993498,
#      and `mjCModel` stores the unit version in `geom_quat`. A non-unit
#      quaternion is not a rotation — it builds a SCALED matrix — so passing
#      the literal poses a different cylinder than the one this number
#      describes. In production `gjk_epa` never sees one: the world quaternion
#      it is given comes out of `forward_kinematics`, which normalizes (see
#      `gpu_quat_normalize`), which is exactly why the routine does not
#      normalize defensively on a per-call hot path.
#
#   2. IT ORDERS THE PAIR (CYLINDER, BOX), NOT (BOX, CYLINDER). `mj_forward`
#      reports `geom1 = 1` (the cylinder, `mjGEOM_CYLINDER` = 5) and
#      `geom2 = 0` (the box, `mjGEOM_BOX` = 6): `pushPairArena` sorts by
#      `mjtGeom` id, which is what `mj_geom_type_rank` mirrors. Our EPA is not
#      symmetric under the swap either — see the measurements below.
#
# ⚠⚠ THIS TEST PASSED FOR TWELVE DAYS WITH BOTH OF THEM WRONG, ON A
# COINCIDENCE. Measured against this constant, `ccd_tolerance` 1e-6:
#
#     tree         order     quat          |d - mj|
#     353696b4     BOX,CYL   raw           7.610e-06   <- passed, budget 1e-5
#     353696b4     BOX,CYL   normalized    1.363e-07
#     HEAD         BOX,CYL   raw           4.673e-05   <- the red
#     HEAD         BOX,CYL   normalized    2.544e-08
#     HEAD         CYL,BOX   normalized    0.0         <- what we assert now
#
# (The last row is 0.0 computing the unit quaternion at run time; from the
# rounded decimal literals below it lands ONE ULP away, 3.5e-18.)
#
# ⚠ THE NORMAL CAME ALONG WITH THE ORDER, WHICH IS THE CORROBORATION. `gjk_epa`
# returns geom1 -> geom2, so asking (CYLINDER, BOX) asks MuJoCo's question:
# ours is now (-2.711e-06, -3.378e-06, -0.99999999999) against MuJoCo's
# `contact.frame[0:3]` = (-0.000003, -0.000003, -1.000000), SIGN INCLUDED,
# where the (BOX, CYLINDER) call returned it pointing the other way. That is
# about the argument order at THIS call, not about
# `_our_contact_normal_is_the_negation_of_mjcontacts`, which is a separate open
# item about how the `Contact` RECORD is built (`body_b -> body_a`).
#
# Read the middle column: with the RIGHT inputs the engine went 1.363e-07 ->
# 0.0 over that span, i.e. the EPA work (`fee37034`, `d67673df`) moved it ONTO
# the reference. What it also did was move our answer for the SKEWED pose to
# that pose's own correct answer, which is 4.7e-05 from the answer for a
# different pose — so the fixture's error stopped being masked. The failure
# was the gate asking one question and comparing against another's answer, and
# a budget loose enough to hide the difference.
comptime MJ_CYLBOX_DIST: Float64 = -0.027966478744914335

# `0.83 0.29 0.41 0.24` (w,x,y,z) normalized, spelled (x,y,z,w) — the order
# every geom argument in this tree uses. Written out rather than computed so
# the call site shows the values EPA actually receives.
comptime QX: Float64 = 0.29018868398682984
comptime QY: Float64 = 0.4102667601193112
comptime QZ: Float64 = 0.24015615226496265
comptime QW: Float64 = 0.8305400265829957

# ⚠ 0.0 IS THE MEASURED ANSWER AND THIS IS STILL A BUDGET. Both engines run
# float64 over the same arithmetic, so the agreement is exact today on this
# machine; 1e-12 leaves room for a compiler reassociating the support function
# without leaving room for a real defect — it is seven orders tighter than the
# 1e-5 that let the fixture's own error through.
comptime TOL_VS_MJ: Float64 = 1e-12


# A cylinder driven obliquely into a BOX corner.
#
# ⚠ CYLINDER-vs-BOX IS A SMOOTH PAIR AND THAT IS THE WHOLE POINT.
# `discreteGeoms` needs mesh/box/hfield on BOTH sides; the cylinder fails it,
# so `_epa_tolerance` returns `ccd_tolerance` and this fixture can see it. A
# box-vs-box or mesh-vs-box fixture would route to `mjMINVAL` and assert
# "changing the tolerance changes the answer" against code that provably
# ignores it.
#
# ⚠ THE OBLIQUE QUATERNION IS LOad-BEARING. Coaxial or axis-aligned poses have
# an exact supporting plane that every tolerance finds, so the two arms would
# agree and the gate would read as "the option does nothing".
#
# ⚠ AN INLINE `<mesh vertex=...>` WAS TRIED FIRST AND SILENTLY GAVE ZERO
# CONTACTS: `load_mesh_hull` loads STL FILES, so an inline vertex list builds a
# mesh with no vertices and every query misses. The vacuity guard below is what
# caught it, which is why that guard is an assert and not a comment.
def _xml(ccd: String) -> String:
    return String(
        "<mujoco><option gravity='0 0 0'"
    ) + ccd + String(
        """/>
  <worldbody>
    <geom name='g_box' type='box' size='0.05 0.04 0.03'/>
    <body name='b1' pos='0.030 0.020 0.038' quat='0.83 0.29 0.41 0.24'>
      <freejoint/>
      <geom name='g_cyl' type='sphere' size='0.03' mass='1'/>
    </body>
  </worldbody>
</mujoco>"""
    )


comptime XML_DEFAULT = _xml("")
comptime XML_LOOSE = _xml(" ccd_tolerance='2e-2' ccd_iterations='4'")

comptime PM_D = parse_xml(XML_DEFAULT)
comptime PM_L = parse_xml(XML_LOOSE)

comptime MD_D = ModelDefFromXML[
    xml=XML_DEFAULT, nbody=PM_D.NBODY, njoint=PM_D.NJOINT, nq=PM_D.NQ,
    nv=PM_D.NV, ngeom=PM_D.NGEOM, nact=PM_D.NACT, ntex=PM_D.NTEX,
    nmat=PM_D.NMAT, nlight=PM_D.NLIGHT, ncam=PM_D.NCAM, nsite=PM_D.NSITE,
    neq=PM_D.NEQ, nexclude=PM_D.NEXCLUDE, npair=PM_D.NPAIR,
    max_tendon=PM_D.NTENDON, max_condim=PM_D.MAX_CONDIM,
    max_equality=1, max_contacts=32, timestep=PM_D.TIMESTEP,
]
comptime MD = ModelDims[MD_D, 4096]
comptime MD_L = ModelDefFromXML[
    xml=XML_LOOSE, nbody=PM_L.NBODY, njoint=PM_L.NJOINT, nq=PM_L.NQ,
    nv=PM_L.NV, ngeom=PM_L.NGEOM, nact=PM_L.NACT, ntex=PM_L.NTEX,
    nmat=PM_L.NMAT, nlight=PM_L.NLIGHT, ncam=PM_L.NCAM, nsite=PM_L.NSITE,
    neq=PM_L.NEQ, nexclude=PM_L.NEXCLUDE, npair=PM_L.NPAIR,
    max_tendon=PM_L.NTENDON, max_condim=PM_L.MAX_CONDIM,
    max_equality=1, max_contacts=32, timestep=PM_L.TIMESTEP,
]
comptime MD_2 = ModelDims[MD_L, 4096]


def test_comptime_parser_reads_the_option() raises:
    """`parse_xml` — the COMPTIME counter, not the runtime parser."""
    print("=== parse_xml reads ccd_tolerance / ccd_iterations ===")
    print("  absent  ->", PM_D.CCD_TOL, PM_D.CCD_ITER)
    print("  present ->", PM_L.CCD_TOL, PM_L.CCD_ITER)
    assert_true(
        PM_D.CCD_TOL == MJ_CCD_TOLERANCE and PM_D.CCD_ITER == MJ_CCD_ITERATIONS,
        "a model with no ccd attributes must get MuJoCo's defaults (1e-6, 35),"
        " got " + String(PM_D.CCD_TOL) + " / " + String(PM_D.CCD_ITER),
    )
    assert_true(
        PM_L.CCD_TOL == 2e-2 and PM_L.CCD_ITER == 4,
        "`<option ccd_tolerance='2e-2' ccd_iterations='4'>` was not read: got "
        + String(PM_L.CCD_TOL) + " / " + String(PM_L.CCD_ITER),
    )
    print("  PASS")


def test_option_reaches_model_meta() raises:
    """The RUNTIME parser stores both into model META.

    ⚠ THIS IS A DIFFERENT PARSER FROM THE ONE THE FIRST TEST CHECKS.
    `parse_xml` only counts; `init_fields` goes through `parse_xml_full` +
    `build_model_fields_from_flat`, and a fix in one has repeatedly not been a
    fix in the other (`feedback_physics3d_two_parser_paths`).

    ⚠ THE TWO MODELS ARE DIFFERENT TYPES, so this cannot be a generic helper —
    `ModelDefFromXML` specialises on the XML string and the whole point is that
    the two XMLs differ. The build is written out twice.
    """
    print("=== ccd options reach model META ===")
    var ctx = DeviceContext()

    var mf_a = Model[DTYPE, MD]()
    MD_D.init_fields[DTYPE](ctx, mf_a)
    var tol_a = Float64(mf_a.meta.data[MODEL_META_IDX_CCD_TOLERANCE])
    var itr_a = Float64(mf_a.meta.data[MODEL_META_IDX_CCD_ITERATIONS])

    var mf_b = Model[DTYPE, MD_2]()
    MD_L.init_fields[DTYPE](ctx, mf_b)
    var tol_b = Float64(mf_b.meta.data[MODEL_META_IDX_CCD_TOLERANCE])
    var itr_b = Float64(mf_b.meta.data[MODEL_META_IDX_CCD_ITERATIONS])

    print("  absent  -> META tol", tol_a, "iter", itr_a)
    print("  present -> META tol", tol_b, "iter", itr_b)
    assert_true(
        tol_a == MJ_CCD_TOLERANCE and Int(itr_a) == MJ_CCD_ITERATIONS,
        "META did not get MuJoCo's defaults for a model that sets neither:"
        " got " + String(tol_a) + " / " + String(itr_a),
    )
    assert_true(
        tol_b == 2e-2 and Int(itr_b) == 4,
        "the RUNTIME parser dropped `<option ccd_tolerance ccd_iterations>` —"
        " META holds " + String(tol_b) + " / " + String(itr_b),
    )
    print("  PASS")


def test_epa_actually_consumes_the_tolerance() raises:
    """Feeding EPA the two tolerances gives two different contacts.

    ⚠ WITHOUT THIS THE TWO CHECKS ABOVE ARE SATISFIED BY DEAD DATA. A value can
    be parsed, stored in META and read by nothing — `<geom solref>` was exactly
    that for months, written into the geom record and consumed by no one.

    ⚠ ASSERTED AS AN INEQUALITY, NOT A GOLDEN. A frozen pair of numbers here
    would keep passing if the argument stopped reaching EPA, because both arms
    would then produce the same frozen value and only a hand-checked constant
    would notice. "These two differ" IS the claim.

    ⚠ WHAT THIS FILE DOES **NOT** COVER: the hop from META to the `gjk_epa`
    call site in `_detect_contacts_env` / `_detect_contacts_sap_env`. Building
    a synthetic two-body XML for that was tried and abandoned — a free sphere
    plainly overlapping a static box produced ZERO contacts through
    `detect_contacts`, while MuJoCo reports 3 for the cylinder version of the
    same file, so the fixture was testing the dispatch rather than the option.
    That anomaly is filed separately; it is not this file's subject and it must
    not be used as evidence that `ccd_tolerance` works.

    The hop IS evidenced, just not here — three measured behaviour changes when
    the hardcoded 1e-8 became `opt.ccd_tolerance`: Jaco reach pose 38's
    cylinder-mesh normal moved 9.43e-3 -> 7.55e-3 closer to MuJoCo's,
    `test_mesh_detection_fields` env0 went 1 -> 0 contacts, and
    `test_mesh_manifold_vs_mujoco` depth moved 1e-16 -> 4.98e-7 until
    `discreteGeoms` was added.
    """
    print("=== EPA consumes ccd_tolerance ===")
    # ⚠⚠ THE PRECONDITION, ASSERTED RATHER THAN COMMENTED. The literal
    # `0.29 0.41 0.24 0.83` sat in this call for twelve days and cost 4.7e-05
    # — more than four times the budget it was checked against — because a
    # non-unit quaternion poses a SCALED cylinder and nothing here said so.
    # `forward_kinematics` supplies a unit quaternion in production; a
    # hand-built call has to supply one too.
    var qn = sqrt(QX * QX + QY * QY + QZ * QZ + QW * QW)
    print("  |q| =", qn)
    assert_true(
        abs(qn - 1.0) <= 1e-15,
        "the fixture quaternion is not UNIT (|q| = " + String(qn) + "), so it"
        " is a scaled matrix rather than a rotation and the pose it builds is"
        " not the one MJ_CYLBOX_DIST was measured on",
    )
    var mv = TensorImpl[DTYPE].alloc(NMV_EPA * 3)
    var _ng = _no_graph_epa()
    var _ne = _no_edges_epa()
    var ws = ccd_ws_alloc[DTYPE]()
    var out = Array[Float64, 8](fill=0.0)
    for i in range(2):
        var tol = 1e-6 if i == 0 else 2e-2
        # ⚠⚠ CYLINDER FIRST, AND THE MESH OPERAND MOVES WITH IT. `gjk_epa`
        # takes the mesh arrays between the FIRST geom's dimensions and the
        # second geom's type, so swapping the pair is not swapping two blocks
        # — the `mv` / graph / edges triple stays where it is and now sits
        # after the cylinder. Both geoms are primitives here so nothing reads
        # it, but a future mesh arm has to bind it to geom 1.
        var r = gjk_epa[DTYPE](
            GEOM_CYLINDER,
            Scalar[DTYPE](0.030), Scalar[DTYPE](0.020), Scalar[DTYPE](0.038),
            Scalar[DTYPE](QX), Scalar[DTYPE](QY), Scalar[DTYPE](QZ),
            Scalar[DTYPE](QW),
            Scalar[DTYPE](0.03), Scalar[DTYPE](0.02),
            Scalar[DTYPE](0), Scalar[DTYPE](0), Scalar[DTYPE](0),
            mv.tt["cpu", NMV_EPA, 3](), _ng.tt["cpu", NMV_EPA](), _ne.tt["cpu", mesh_max_edge(NMV_EPA)](), 0, 0,
            GEOM_BOX,
            Scalar[DTYPE](0), Scalar[DTYPE](0), Scalar[DTYPE](0),
            Scalar[DTYPE](0), Scalar[DTYPE](0), Scalar[DTYPE](0),
            Scalar[DTYPE](1),
            Scalar[DTYPE](0), Scalar[DTYPE](0),
            Scalar[DTYPE](0.05), Scalar[DTYPE](0.04), Scalar[DTYPE](0.03),
            0, 0,
            ws.tt_dyn["cpu", L_CCD_WS1](L_CCD_WS1()), 0,
            Scalar[DTYPE](tol), MJ_CCD_ITERATIONS,
        )
        out[i * 4 + 0] = Float64(r[0])
        out[i * 4 + 1] = Float64(r[4])
        out[i * 4 + 2] = Float64(r[5])
        out[i * 4 + 3] = Float64(r[6])
        print("  ccd_tolerance", tol, " dist", out[i * 4 + 0],
              " n (", out[i * 4 + 1], out[i * 4 + 2], out[i * 4 + 3], ")")

    # The fixture must penetrate, or the comparison below is vacuous.
    assert_true(
        out[0] < 0.0 and out[4] < 0.0,
        "the cylinder/box fixture is not penetrating (dist " + String(out[0])
        + " / " + String(out[4]) + "); MuJoCo reports -0.027966 here, so a"
        " non-negative dist means EPA lost the pair and nothing below is a"
        " test of the tolerance",
    )
    # And it must agree with MuJoCo at MuJoCo's own tolerance — otherwise this
    # would happily gate two flavours of wrong.
    assert_true(
        abs(out[0] - MJ_CYLBOX_DIST) <= TOL_VS_MJ,
        "at MuJoCo's own ccd_tolerance our depth is " + String(out[0])
        + " against MuJoCo's " + String(MJ_CYLBOX_DIST) + " (budget "
        + String(TOL_VS_MJ) + "). BEFORE HUNTING IN EPA, CHECK THE FIXTURE:"
        " this call hand-builds what the model pipeline normally supplies, and"
        " it has been wrong here before — the quaternion must be UNIT (see QX"
        " above; a raw `0.29 0.41 0.24 0.83` costs 4.7e-05) and the pair must"
        " be ordered (CYLINDER, BOX) as `mj_geom_type_rank` orders it (the"
        " swap costs 2.5e-08). With both right this has been EXACT.",
    )

    var d_dist = abs(out[0] - out[4])
    var d_n = abs(out[1] - out[5]) + abs(out[2] - out[6]) + abs(out[3] - out[7])
    print("  |d(dist)| =", d_dist, "  |d(normal)|_1 =", d_n)
    assert_true(
        d_dist > 1e-9 or d_n > 1e-9,
        "ccd_tolerance 1e-6 and 2e-2 produced the SAME contact (d_dist "
        + String(d_dist) + ", d_normal " + String(d_n) + "). EPA is ignoring"
        " its tolerance argument",
    )
    print("  PASS")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
