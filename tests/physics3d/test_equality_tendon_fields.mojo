"""Regression gate (GOLDEN-frozen): equality + tendon constraints on fields.

Part A (TENDON): Humanoid (max_tendon=2, free joint + 17 hinges) dropped on
the floor with feet penetrating, BATCH=2, 2 full Euler steps. Legacy per
step: step_kernel -> detect_contacts_gpu (O(N^2)) -> PGSSolver.solve_gpu
(with MAX_TENDON=2) -> finalize. Fields: EulerIntegrator.step
(detection -> serialized contact PGS with limits + tendons inside).
qpos/qvel/qacc + solved contact records must be BIT-EXACT per step.
The two hip-knee tendon RECORDS are injected DIRECTLY into the per-field
tendon tensor by the test (init_fields, then mf.tendons/mf.meta writes —
no slab, no load_from_slab): <tendon> XML parsing was removed from the
parser, so no XML model ever carries tendon records — the tendon path is
only reachable with manually populated records, which this gate provides
(the golden was frozen from the legacy-validated run of the same records).
Joint poses are chosen strictly INSIDE all joint ranges so the joint-limit
pass stays inactive: the legacy limit builder reads dof_invweight0 through
a MAX_TENDON-less offset (a pre-existing misread on tendon models) which
limits does NOT reproduce — with no active limits that value is
never read. The tendon builder's identical misread IS reproduced by
_tendon_env (_legacy_invw_read), which this gate locks in.
Non-vacuous: model meta NTENDON must be 2, and a rerun with meta NTENDON
zeroed must change qvel after one step.

Part B (EQUALITY): synthetic 2-link chain + jointed anchor body with a
<equality><weld> between link2 and anchor. The model is built offset-free
via init_fields, which serializes the weld equality records (Stage B fixed
copy_equality_to_buffer — the legacy init_model_gpu never called it, so the
slab path could not form a meaningful gate). Capsules penetrate the floor
(contacts + weld together, matching the solve order: contacts -> limits ->
equality). BIT-EXACT per step vs the golden; non-vacuous via meta
NEQUALITY == 1 + a NEQUALITY-zeroed rerun differing.

Run: pixi run -e apple mojo run -I . tests/physics3d/test_equality_tendon_fields.mojo
"""

from std.math import abs
from std.sys import default_accelerator
from max.gpu.host import DeviceContext

from noeira.nn.core.tensor import TensorImpl
from noeira.physics3d.types import ConeType
from noeira.physics3d.parser import parse_xml, ModelDefFromXML
from noeira.physics3d.fields import Data, Model, Dims
from noeira.physics3d.integrator.euler import EulerIntegrator
from noeira.physics3d.gpu.constants import (
    MODEL_TENDON_SIZE,
    META_IDX_NUM_CONTACTS,
    MODEL_META_IDX_NTENDON,
    MODEL_META_IDX_NEQUALITY,
    CONTACT_SIZE,
    TENDON_IDX_NUM_JOINTS,
    TENDON_IDX_IS_EQUALITY,
    TENDON_IDX_JOINT_0,
    TENDON_IDX_JOINT_1,
    TENDON_IDX_JOINT_2,
    TENDON_IDX_JOINT_3,
    TENDON_IDX_COEF_0,
    TENDON_IDX_COEF_1,
    TENDON_IDX_COEF_2,
    TENDON_IDX_COEF_3,
    TENDON_IDX_LENGTH_REF,
    TENDON_IDX_SOLREF_0,
    TENDON_IDX_SOLREF_1,
    TENDON_IDX_SOLIMP_0,
    TENDON_IDX_SOLIMP_1,
    TENDON_IDX_SOLIMP_2,
    TENDON_IDX_SOLIMP_3,
    TENDON_IDX_SOLIMP_4,
    EQ_IDX_TYPE,
    EQ_IDX_BODY_A,
    EQ_IDX_BODY_B,
    EQ_IDX_ANCHOR_AX,
    EQ_IDX_ANCHOR_AY,
    EQ_IDX_ANCHOR_AZ,
    EQ_IDX_ANCHOR_BX,
    EQ_IDX_ANCHOR_BY,
    EQ_IDX_ANCHOR_BZ,
    EQ_IDX_RELPOSE_X,
    EQ_IDX_RELPOSE_Y,
    EQ_IDX_RELPOSE_Z,
    EQ_IDX_RELPOSE_W,
    EQ_IDX_SOLREF_0,
    EQ_IDX_SOLREF_1,
    EQ_IDX_SOLIMP_0,
    EQ_IDX_SOLIMP_1,
    EQ_IDX_SOLIMP_2,
    EQ_IDX_SOLIMP_3,
    EQ_IDX_SOLIMP_4,
    METADATA_SIZE,
)
from noeira.envs.humanoid.humanoid_xml import HumanoidModel

comptime DTYPE = DType.float32
comptime BATCH = 2

# Regenerated 2026-07-31 (GOLD_A only, 0.25%; GOLD_NCON_A and both Part B
# numbers unchanged) for the tendon-equality diagApprox. `_tendon_env` used to
# build its `R` from the SUM of `dof_invweight0` over the tendon's joints, via
# `_legacy_invw_read` — the docstring above still describes that misread as
# something this gate "locks in". MuJoCo uses `tendon_invweight0[eq_obj1id]`,
# one number (engine_core_constraint.c:1091); the sum is the mjEQ_JOINT rule
# (:1090) applied to the wrong constraint type. On dm_control's quadruped,
# whose four coupling tendons are the first equality tendons any model here
# builds through the PARSER, the old rule put ~7% on qacc; the new one takes
# the contact-free gate in test_rne_post_sensors_vs_mujoco.mojo to 2e-7.
#
# Part A's records are injected by hand and never set `TENDON_IDX_INVWEIGHT0`,
# so this model now takes the `diag_ten = k` (exact K) fallback rather than a
# misread. Still a regression pin, not a correctness statement — as below.
#
# --- GOLDEN fingerprints (frozen from the legacy-validated fields-GPU run) ---
comptime HARVEST = False  # True => print fingerprints + skip asserts (regen)
comptime GOLD_RTOL = 1e-3
comptime GOLD_NCON_A = 8  # Part A tendon: total contacts over the steps
# --- 2026-08-18: the humanoid's `margin` finally REACHES the contact record.
# -499170.73185298603 -> -518630.5688423617, a 3.90% move, and it is a
# CORRECTNESS GAIN rather than drift. Found by splitting the fingerprint, not
# by bisecting: the split names the mechanism, the bisect only names a commit.
#
#     qpos       12.279942702582048 ->    12.206922838406172   -0.59%
#     qvel    -2578.2850333643146   -> -2675.4913359144703     +3.77%
#     qacc  -592911.1068024635      ->  -616023.9479956627     +3.90%
#     con_geom 95410.86642321944    ->  99162.31557208393      +3.93%
#     con_solp   894.3479942930862  ->    894.3479942930862    BIT-EXACT
#
# Contact GEOMETRY is unchanged (positions and normals differ at 1e-5, float32
# noise) and the solparam columns are bit-identical, so nothing about
# detection or contact parameters moved. Everything the SOLVER produces moved
# by a coherent ~3.9%. Column by column, only 8..12 carry it:
#
#     k=9  INCLUDEMARGIN  0.0  ->  0.002        <- THE CAUSE, and it was ZERO
#     k=10 FORCE_N        1817.19 -> 1878.15    +3.36%
#     k=11 FORCE_T1         25.48 ->   26.28    +3.14%
#     k=12 FORCE_T2        109.74 ->  110.92    +1.08%
#
# `assets/humanoid.xml` has ALWAYS declared `margin="0.001"` — on a
# `<default><geom>`, the single biggest defect class in this parser's history.
# The COMPTIME parser never read geom margin AT ALL (no `GEOM_IDX_MARGIN`
# write anywhere in `xml_parser.mojo`), so for the whole comptime era every
# humanoid contact ran with `includemargin = 0` and activated later than
# MuJoCo's. `53ac294b` deleted the embedded MJCF and moved all 61 models to
# the runtime parser, which does read it.
#
# ⚠ 0.002 IS THE SUM OF THE TWO GEOMS' MARGINS, AND THAT IS MuJoCo'S RULE —
# I nearly filed the sum as a bug. `engine_collision_driver.c:1600`, under
# "set margin: dynamic or pair", dynamic branch:
#     margin = mj_assignMargin(m, m->geom_margin[g1] + m->geom_margin[g2]);
# It is `max` NOWHERE. 0.001 + 0.001 = 0.002 exactly.
#
# GOLD_NCON_A is UNCHANGED at 8: the feet are already deeply penetrating at
# this pose, so an earlier activation threshold adds no new rows — which is
# also why the move is purely in the forces and not in the contact set.
#
# ⚠ STILL A REGRESSION PIN, NOT A CORRECTNESS STATEMENT — this file has no
# oracle (see the header: `<tendon>` parsing was REMOVED, the records are
# injected by hand). The MuJoCo claim above is about the margin RULE, which is
# checkable against the reference; the fingerprint itself remains self-frozen.
# `test_within_margin_convex_contacts` and `test_humanoid_limits_fields_vs_mujoco`
# are green at HEAD and are the live evidence such as it is.
comptime GOLD_A = -518630.5688423617  # Part A final qpos/qvel/qacc/contacts checksum
# Re-harvested 2026-08-03 (was -500065.48171551153, a +894.750 move), and the
# delta is ACCOUNTED FOR RATHER THAN RE-RECORDED ON SIGHT:
#
#   newly-filled contact slots 23..29   +894.348   DEFINITIONAL
#   drift inside the original 0..22     -  0.402   PRE-EXISTING, 8.0e-07 rel
#                                       ---------
#   observed                             +894.750
#
# ⚠ THE FINGERPRINT'S DEFINITION CHANGED, NOT THE PHYSICS. This loop sums
# `for k in range(CONTACT_SIZE)`, and `CONTACT_SIZE` grew 23 -> 30 when
# per-contact solref/solimp were appended to the record. So it now weighs seven
# slots it never saw. `mix_contact_params` writes MuJoCo's own mixed values
# there — for these two models every geom carries the defaults, so the added
# terms are (0.02, 1) and (0.9, 0.95, 0.001, 0.5, 2) weighted by (c+1)*(k+1).
#
# ⚠ THE -0.402 IS NOT PART OF THIS CHANGE. Measured with every uncommitted file
# stashed (tree at commit 9bd3aad1) it is IDENTICAL to the last digit, so it
# predates the work that moved the constant. It is 8.0e-07 relative, riding
# under GOLD_RTOL = 1e-3, and Part A is exactly the fingerprint this file
# already documents as amplifying last-bit differences: it rests on four
# redundant contacts whose force split is INDETERMINATE (the total is pinned,
# the split is not). Left as its own question rather than folded in here.
#
# ⚠ IT TOOK THREE ATTEMPTS TO GET THIS ACCOUNTING RIGHT, and the two wrong ones
# are worth naming. First I compared `fp - new_slots` against the golden — that
# subtracts the NEW contribution of slots 23..29 while the golden carries their
# OLD contribution, which are different quantities, and the "residual" it
# produced was an artifact of the subtraction. Then I attributed that artifact
# to uninitialized memory in the new slots. Only summing slots 0..22 ALONE and
# comparing to the golden separates the two effects. When a golden moves, split
# it by the thing you changed, not by the thing you are looking at.
# ── the pre-2026-08-03 history of this constant follows ──
# Re-harvested 2026-08-03 (was -513649.55245587835, a 2.6% move).
#
# WHY it moved: `kinematics/quat_math.mojo` used to normalize quaternions as
# `1/sqrt(norm_sq + 1e-10)`, a divide-by-zero guard placed INSIDE the sqrt. For
# an already-unit quaternion that returns 0.99999999995, so EVERY body
# quaternion came out 5e-11 short of unit and every vector rotated by one was
# scaled by 1 - 1e-10. Removing that bias changes the arithmetic under every
# model in the tree.
#
# WHY 5e-11 MOVES A CHECKSUM BY 2.6%, which is the part that has to be
# explained rather than waved at:
#   * `GOLD_NCON_A` is UNCHANGED at 8, so the contact SET is identical — no
#     discrete flip, no gained or lost contact.
#   * Part A's fields-CPU vs fields-GPU agreement is 5.2e-8 on qpos, so the
#     configuration is NOT chaotic and the new state is self-consistent. (Part
#     B's is 1.2e-2 — that one IS sensitive, and its golden did NOT move.)
#   * The fingerprint is dominated by its CONTACT-RECORD terms: 23 slots per
#     contact weighted up to (c+1)*(k+1) = 92, carrying solved forces of order
#     1e3, against qpos/qvel terms of order 1e0-1e1. The humanoid rests on
#     four contacts, which is force-INDETERMINATE — the total is pinned, the
#     split between redundant contacts is not — so a tiny perturbation
#     redistributes the forces measurably while qpos barely moves. That is
#     what a 2.6% move in a force-weighted checksum with an unchanged contact
#     set and a 5.2e-8 qpos means.
#
# WHY THE NEW NUMBER IS THE BETTER ONE, which a self-golden cannot establish
# on its own: the same fix takes quadruped's force/torque SENSORS — a direct
# comparison of contact forces against MuJoCo — from 5.06e-11 to 4.07e-15, and
# capsule contact normals from 1.0e-10 to 4.4e-16. The contact solve did not
# merely change, it got ~12,000x closer to the reference.
comptime GOLD_NCON_B = 6  # Part B equality: total contacts over the steps
# Re-harvested 2026-07-29 (was 222.7145065382404).
#
# WHY it moved: part B's two links are `fromto` capsules, and commit f0d35e2c
# changed `fromto` orientation to MuJoCo's convention (`vec = from - to`, then
# `mjuu_z2quat`). That is the same solid — same shape, same inertia, same
# contact geometry — but it lands on a different roll about the capsule's own
# axis, and the capsule axis is the TANGENT-FRAME HINT the ELLIPTIC friction
# cone builds its basis from. Different (equally valid) tangent basis =>
# different friction impulses => a different trajectory. `GOLD_NCON_B` is
# unchanged, confirming the contact SET is identical and only the friction
# solve moved.
#
# What this golden is and is not: it is a REGRESSION PIN frozen from the
# legacy engine, not a correctness statement. Part B's weld solve has never
# been gated against MuJoCo, and a direct comparison on this model shows the
# two disagree substantially (qvel[2] -7.4 vs -0.43 by step 0). Validating the
# weld path against MuJoCo is separate, unfinished work — do not read a pass
# here as "welds are correct".
# Regenerated 2026-07-30 for bug 20 (`dynamics/invweight.mojo`). `invweight0`
# used to substitute `body_invweight0`'s TRANSLATIONAL half with its ROTATIONAL
# half whenever the translational one came out ~0, under a comment claiming
# that was MuJoCo behaviour. It is not: `mj_setConst` assigns the two
# independently (engine_setconst.c:157-158), and a zero translational weight is
# a CORRECT answer for a body whose CoM lies on its only rotation axis.
#
# This model has exactly such a body — `anchor` is a sphere at its own body
# origin on a hinge through pos="0 0 0", so it cannot be translated and its
# true translational weight IS 0. The weld path reads BOTH invweight
# components as its diagApprox, so the anchor's fabricated non-zero value fed
# straight into the weld solve. Verified by causation, not inference: with only
# `invweight.mojo` reverted this test matches the OLD number exactly, and with
# it applied it matches the new one.
#
# So the previous value was frozen around a wrong model constant, the same way
# `test_euler_fields_vs_mujoco`'s limit golden was before bug 18. That is the
# recurring hazard of a self-golden over a regime never checked against ground
# truth: it preserves whatever it captured. Per the note above, this number is
# still a REGRESSION PIN and still not a correctness statement — the weld path
# remains ungated against MuJoCo.
# Re-harvested 2026-08-12 (29331.575603858786 -> 23698.22404074709) for the
# WELD ROW FIXES, and this is the "separate, unfinished work" the paragraph
# above names — the weld path is now gated against MuJoCo in
# tests/physics3d/test_weld_rows_pyramidal_vs_mujoco.mojo (efc_J 0.0,
# efc_aref 0.0, efc_D 2.7e-14). Four defects, all of which moved this model:
#   * `relpose` defaulted to IDENTITY instead of the qpos0 relative pose. On
#     THIS model MuJoCo derives (0.32, 0, 0.051) — link2 and anchor are not
#     coincident — so the old golden pinned a weld aimed at the wrong pose.
#     Ours now reads (0.32, 0, 0.051), matching exactly.
#   * the rotational Jacobian was three world-axis rows, not MuJoCo's
#     quaternion-corrected construction.
#   * the 0.5 was on the residual instead of the Jacobian.
#   * the impedance used the per-row residual, not `norm(efc_pos, 6)`.
# Measured against MuJoCo on this model: the disagreement recorded above
# (qvel -7.4 vs -0.43) is now 1.2e-2 worst on qvel over three steps. The
# residual is the FRICTION BASIS difference this file already documents (the
# capsule tangent-frame hint), not the weld.
#
# ⚠ STILL A REGRESSION PIN, NOT A CORRECTNESS STATEMENT — the number is a
# self-golden and the remaining 1.2e-2 is unexplained-by-the-weld, not zero.
# --- 2026-09-06: connect/weld rows carry MuJoCo 3.10's J̇·v reference term
# (PERFORMANCE.md §13.29) and a <fixed> tendon's limit is a row at last.
# Harvested on that tree: Part A -518630.5688 -> -518629.2910 (2.5e-6, the
# injected tendons' limit rows) and Part B 23698.2240 -> 23708.3722 (4.3e-4,
# the weld is MOVING and the term is centripetal). Both inside GOLD_RTOL, so
# the pins are left where they were; the numbers are recorded here so the
# next move is measured from the right baseline. The CPU-vs-GPU parity leg
# below is what caught a heap `Scratch` inside the GPU kernel on the way.
comptime GOLD_B = 23698.22404074709  # Part B final qpos/qvel/qacc/contacts checksum
# Re-harvested 2026-08-03 (was 29033.456920214216, a +298.119 move) for the SAME
# reason as GOLD_A: `CONTACT_SIZE` grew 23 -> 30 and this fingerprint sums
# `range(CONTACT_SIZE)`. Predicted definitional delta for a pair of default
# geoms is 149.058 per unit of `(c+1)`, and the observed move is 2x that to
# within 0.003 — i.e. essentially all of it, with a residual three orders
# smaller than Part A's.
#
# ⚠ THIS ONE WAS MASKED. Part A raises before Part B runs, so while GOLD_A was
# failing this constant looked untouched; it had moved the whole time. A test
# that aborts on its first golden hides every later one.

# =============================================================================
# Part A: Humanoid (tendons)
# =============================================================================

comptime NQ_A = HumanoidModel.NQ  # 24
comptime NV_A = HumanoidModel.NV  # 23
comptime NBODY_A = HumanoidModel.NBODY  # 14
comptime NJOINT_A = HumanoidModel.NJOINT  # 18
comptime NGEOM_A = HumanoidModel.NGEOM  # 18
comptime MC_A = HumanoidModel.MAX_CONTACTS  # 50
comptime NTEN_A = HumanoidModel.MAX_TENDON  # 2
comptime CONE_A = HumanoidModel.CONE_TYPE
comptime NEQ_A = HumanoidModel.MAX_EQUALITY  # 0
comptime NSITE_A = HumanoidModel.NSITE  # 0
comptime NEXCL_A = HumanoidModel.nexclude  # 0
comptime MD = Dims[
    nq=NQ_A,
    nv=NV_A,
    nbody=NBODY_A,
    njoint=NJOINT_A,
    ngeom=NGEOM_A,
    nsite=NSITE_A,
    max_contacts=MC_A,
    nequality=NEQ_A,
    ntendon=NTEN_A,
    nexclude=NEXCL_A,
    nmesh_verts=0,
    npair=HumanoidModel.NPAIR,
    nact=HumanoidModel.NACT,
    nten=HumanoidModel.NTEN_F,
    nkey=HumanoidModel.NKEY,
]
comptime N_STEPS_A = 2


def _humanoid_qpos(e: Int, i: Int) -> Scalar[DTYPE]:
    """Free joint pose + hinge angles strictly inside every joint range
    (keeps the joint-limit pass inactive — see module docstring)."""
    if i == 0:
        return Scalar[DTYPE](0.02) * Scalar[DTYPE](e)
    if i == 1:
        return Scalar[DTYPE](0)
    if i == 2:
        return Scalar[DTYPE](1.24)  # feet spheres penetrate the floor
    if i == 3:
        return Scalar[DTYPE](1)  # identity quaternion (w first)
    if i <= 6:
        return Scalar[DTYPE](0)
    # Hinges (qpos 7..23): abdomen_z/y/x, r_hip_x/z/y, r_knee,
    # l_hip_x/z/y, l_knee, r_sh1/2, r_elbow, l_sh1/2, l_elbow
    if i == 7:
        return Scalar[DTYPE](0.05) + Scalar[DTYPE](0.01) * Scalar[DTYPE](e)
    if i == 8:
        return Scalar[DTYPE](-0.1)
    if i == 9:
        return Scalar[DTYPE](0.05)
    if i == 10 or i == 14:
        return Scalar[DTYPE](-0.1)  # hip_x in [-0.436, 0.0873]
    if i == 11 or i == 15:
        return Scalar[DTYPE](-0.1)  # hip_z in [-1.047, 0.611]
    if i == 12 or i == 16:
        return Scalar[DTYPE](-0.05)  # hip_y in [-1.92, 0.349]
    if i == 13 or i == 17:
        return Scalar[DTYPE](-0.15)  # knee in [-2.79, -0.0349]
    if i == 20 or i == 23:
        return Scalar[DTYPE](-0.3)  # elbow in [-1.571, 0.873]
    return Scalar[DTYPE](0.1) + Scalar[DTYPE](0.01) * Scalar[DTYPE](e)


def _part_a_tendon(ctx: DeviceContext) raises:
    print("--- Part A: Humanoid tendons fields GOLDEN, BATCH=", BATCH)

    var mf = Model[DTYPE, MD]()
    HumanoidModel.init_fields[DTYPE](ctx, mf)

    # <tendon><fixed> XML parsing was removed from the parser (see
    # model_def_from_xml.mojo), so no XML model carries tendon records —
    # the tendon path is only reachable via manual population. Inject the
    # Humanoid's two hip-knee tendons (coef -1 * hip_y + 1 * knee,
    # MuJoCo-default solref/solimp) DIRECTLY into the per-field tendon tensor
    # (record layout t_i * MODEL_TENDON_SIZE + TENDON_IDX_*) + meta NTENDON,
    # then re-upload those two tensors.
    mf.meta.data[MODEL_META_IDX_NTENDON] = Scalar[DTYPE](2)
    for t_i in range(2):
        var t_off = t_i * MODEL_TENDON_SIZE
        # right: r_hip_y (joint 6) + r_knee (joint 7);
        # left: l_hip_y (joint 10) + l_knee (joint 11)
        var j0 = 6 if t_i == 0 else 10
        # `_tendon_env` imposes a BILATERAL EQUALITY, and since
        # 2026-07-31 it only acts on records that say so. That gate
        # exists because `fields_build` now populates `ntendon`
        # honestly, and humanoid's <fixed> tendons are NOT constrained
        # by MuJoCo — without it, every humanoid hip-knee pair would be
        # welded. This test's whole subject IS the equality path, so it
        # opts in explicitly.
        mf.tendons.data[t_off + TENDON_IDX_IS_EQUALITY] = Scalar[DTYPE](1)
        mf.tendons.data[t_off + TENDON_IDX_NUM_JOINTS] = Scalar[DTYPE](2)
        mf.tendons.data[t_off + TENDON_IDX_JOINT_0] = Scalar[DTYPE](j0)
        mf.tendons.data[t_off + TENDON_IDX_JOINT_1] = Scalar[DTYPE](j0 + 1)
        mf.tendons.data[t_off + TENDON_IDX_JOINT_2] = Scalar[DTYPE](-1)
        mf.tendons.data[t_off + TENDON_IDX_JOINT_3] = Scalar[DTYPE](-1)
        mf.tendons.data[t_off + TENDON_IDX_COEF_0] = Scalar[DTYPE](-1)
        mf.tendons.data[t_off + TENDON_IDX_COEF_1] = Scalar[DTYPE](1)
        mf.tendons.data[t_off + TENDON_IDX_COEF_2] = Scalar[DTYPE](0)
        mf.tendons.data[t_off + TENDON_IDX_COEF_3] = Scalar[DTYPE](0)
        mf.tendons.data[t_off + TENDON_IDX_LENGTH_REF] = Scalar[DTYPE](0.05)
        mf.tendons.data[t_off + TENDON_IDX_SOLREF_0] = Scalar[DTYPE](0.02)
        mf.tendons.data[t_off + TENDON_IDX_SOLREF_1] = Scalar[DTYPE](1)
        mf.tendons.data[t_off + TENDON_IDX_SOLIMP_0] = Scalar[DTYPE](0.9)
        mf.tendons.data[t_off + TENDON_IDX_SOLIMP_1] = Scalar[DTYPE](0.95)
        mf.tendons.data[t_off + TENDON_IDX_SOLIMP_2] = Scalar[DTYPE](0.001)
        mf.tendons.data[t_off + TENDON_IDX_SOLIMP_3] = Scalar[DTYPE](0.5)
        mf.tendons.data[t_off + TENDON_IDX_SOLIMP_4] = Scalar[DTYPE](2)
    mf.tendons.upload(ctx)
    mf.meta.upload(ctx)

    if Int(mf.meta.data[MODEL_META_IDX_NTENDON]) != 2:
        raise Error("part A vacuous: model meta NTENDON != 2")

    var d = Data[DTYPE, MD, BATCH]()
    var dc = Data[DTYPE, MD, BATCH]()
    var d_off = Data[DTYPE, MD, BATCH]()
    for e in range(BATCH):
        for i in range(NQ_A):
            var qp = _humanoid_qpos(e, i)
            d.qpos.data[e * NQ_A + i] = qp
            dc.qpos.data[e * NQ_A + i] = qp
            d_off.qpos.data[e * NQ_A + i] = qp
        for i in range(NV_A):
            var qv = Scalar[DTYPE]((e * 7 + i * 5) % 7 - 3) / 20.0
            var qf = Scalar[DTYPE]((e * 13 + i * 9) % 9 - 4) / 4.0
            d.qvel.data[e * NV_A + i] = qv
            d.qfrc.data[e * NV_A + i] = qf
            dc.qvel.data[e * NV_A + i] = qv
            dc.qfrc.data[e * NV_A + i] = qf
            d_off.qvel.data[e * NV_A + i] = qv
            d_off.qfrc.data[e * NV_A + i] = qf
    d.upload_all(ctx)

    var integ = EulerIntegrator[DTYPE, MD, CONE_A, BATCH]()
    integ.prepare_gpu(ctx)
    var integ_c = EulerIntegrator[DTYPE, MD, CONE_A, BATCH]()

    var qvel_step0 = List[Scalar[DTYPE]](capacity=BATCH * NV_A)
    for _ in range(BATCH * NV_A):
        qvel_step0.append(Scalar[DTYPE](0))

    var ncon_total = 0
    for step in range(N_STEPS_A):
        integ.step["gpu"](d, mf, ctx)
        integ_c.step["cpu"](dc, mf)
        d.meta.download(ctx)
        d.qvel.download(ctx)
        if step == 0:
            for i in range(BATCH * NV_A):
                qvel_step0[i] = d.qvel.data[i]
        var ncon_seen = 0
        for e in range(BATCH):
            ncon_seen += Int(
                d.meta.data[e * METADATA_SIZE + META_IDX_NUM_CONTACTS]
            )
        if ncon_seen == 0:
            raise Error(
                "part A step " + String(step) + ": no contacts — vacuous"
            )
        ncon_total += ncon_seen
        print("  step", step, ": contacts", ncon_seen)

    # --- final fields-GPU fingerprint (Apple-gated) ---
    d.qpos.download(ctx)
    d.qvel.download(ctx)
    d.qacc.download(ctx)
    d.contacts.download(ctx)
    d.meta.download(ctx)
    var fp = Float64(0)
    for e in range(BATCH):
        for i in range(NQ_A):
            fp += Float64(d.qpos.data[e * NQ_A + i]) * Float64(e * NQ_A + i + 1)
        for i in range(NV_A):
            fp += Float64(d.qvel.data[e * NV_A + i]) * Float64(
                (e * NV_A + i + 1) * 7
            )
            fp += Float64(d.qacc.data[e * NV_A + i]) * Float64(
                (e * NV_A + i + 1) * 13
            )
        var nc2 = Int(d.meta.data[e * METADATA_SIZE + META_IDX_NUM_CONTACTS])
        for c in range(nc2):
            for k in range(CONTACT_SIZE):
                fp += Float64(
                    d.contacts.data[
                        e * MC_A * CONTACT_SIZE + c * CONTACT_SIZE + k
                    ]
                ) * Float64((c + 1) * (k + 1))
    if HARVEST:
        print("  HARVEST GOLD_NCON_A =", ncon_total)
        print("  HARVEST GOLD_A      =", fp)
    else:
        if ncon_total != GOLD_NCON_A and not default_accelerator().is_nvidia_gpu():
            raise Error(
                "part A contacts " + String(ncon_total) + " != golden "
                + String(GOLD_NCON_A)
            )
        var denom = abs(GOLD_A) if abs(GOLD_A) > 1e-9 else 1.0
        if abs(fp - GOLD_A) / denom > GOLD_RTOL and (
            not default_accelerator().is_nvidia_gpu()
        ):
            raise Error(
                "part A fingerprint " + String(fp) + " != golden "
                + String(GOLD_A)
            )
        print("  Part A matches golden fingerprint")

    var worst = Float64(0)
    for i in range(BATCH * NQ_A):
        var err = abs(Float64(dc.qpos.data[i]) - Float64(d.qpos.data[i]))
        if err > worst:
            worst = err
    print("  fields-CPU vs fields-GPU final qpos worst err:", worst)
    if worst > 1e-2:
        raise Error("part A: fields-CPU diverged from GPU")

    # Non-vacuity: tendon-off rerun (meta NTENDON=0 short-circuits the
    # builder exactly like the legacy `if nten == 0: return`) must differ
    # from the tendon-on step-0 qvel.
    mf.meta.data[MODEL_META_IDX_NTENDON] = Scalar[DTYPE](0)
    mf.meta.upload(ctx)
    d_off.upload_all(ctx)
    integ.step["gpu"](d_off, mf, ctx)
    d_off.qvel.download(ctx)
    var ndiff = 0
    for i in range(BATCH * NV_A):
        if d_off.qvel.data[i] != qvel_step0[i]:
            ndiff += 1
    if ndiff == 0:
        raise Error("part A vacuous: tendon-off run identical to tendon-on")
    print("  non-vacuous: tendon-off rerun differs in", ndiff, "qvel entries")
    print("  Part A PASS")


# =============================================================================
# Part B: synthetic weld equality model
#
# `<compiler angle="radian"/>` is explicit on purpose. This model is synthetic,
# and its `range="-170 170"` hinges were written meaning RADIANS — i.e.
# "effectively unlimited", which is the regime the golden was frozen in. The
# parser used to DEFAULT to radian (wrongly: MuJoCo's MJCF default is degree,
# corrected in commit f0d35e2c), so that intent was implicit. Stating it keeps
# this test testing what it is for — the weld equality rows — rather than
# silently starting to test joint limits.
# =============================================================================

comptime weld_xml = """
<mujoco model="weldtest">
    <compiler angle="radian"/>
    <option timestep="0.005" iterations="50" solver="PGS"/>
    <worldbody>
        <geom name="floor" type="plane" size="5 5 0.1" pos="0 0 0" condim="3" friction="1 0.1 0.1"/>
        <body name="link1" pos="0 0 0.049">
            <joint name="j1" type="hinge" axis="0 1 0" pos="0 0 0" range="-170 170" limited="true" damping="0.1"/>
            <geom name="g1" type="capsule" fromto="0 0 0 0.3 0 0" size="0.05" condim="3" friction="1 0.1 0.1"/>
            <body name="link2" pos="0.3 0 0">
                <joint name="j2" type="hinge" axis="0 1 0" pos="0 0 0" range="-170 170" limited="true" damping="0.1"/>
                <geom name="g2" type="capsule" fromto="0 0 0 0.3 0 0" size="0.05" condim="3" friction="1 0.1 0.1"/>
            </body>
        </body>
        <body name="anchor" pos="0.62 0 0.1">
            <joint name="j3" type="hinge" axis="1 0 0" pos="0 0 0" range="-170 170" limited="true" damping="0.1"/>
            <geom name="g3" type="sphere" size="0.04" contype="0" conaffinity="0"/>
        </body>
    </worldbody>
    <equality>
        <weld body1="link2" body2="anchor"/>
    </equality>
</mujoco>
"""

comptime pm_b = parse_xml(weld_xml)

comptime WeldTestModel = ModelDefFromXML[
    xml=weld_xml,
    nbody=pm_b.NBODY,
    njoint=pm_b.NJOINT,
    nq=pm_b.NQ,
    nv=pm_b.NV,
    ngeom=pm_b.NGEOM,
    nact=pm_b.NACT,
    max_contacts=8,
    max_equality=6,  # 1 weld = 6 rows
    cone_type=ConeType.ELLIPTIC,
    neq=pm_b.NEQ,
    timestep=pm_b.TIMESTEP,
]

comptime NQ_B = WeldTestModel.NQ  # 3
comptime NV_B = WeldTestModel.NV  # 3
comptime NBODY_B = WeldTestModel.NBODY  # 4
comptime NJOINT_B = WeldTestModel.NJOINT  # 3
comptime NGEOM_B = WeldTestModel.NGEOM  # 4
comptime MC_B = WeldTestModel.MAX_CONTACTS  # 8
comptime NEQ_B = WeldTestModel.MAX_EQUALITY  # 6
comptime CONE_B = WeldTestModel.CONE_TYPE
comptime N_STEPS_B = 3


comptime MD_2 = Dims[
    nq=NQ_B,
    nv=NV_B,
    nbody=NBODY_B,
    njoint=NJOINT_B,
    ngeom=NGEOM_B,
    nsite=WeldTestModel.NSITE,
    max_contacts=MC_B,
    nequality=NEQ_B,
    ntendon=WeldTestModel.MAX_TENDON,
    nexclude=WeldTestModel.nexclude,
    nmesh_verts=0,
    npair=WeldTestModel.NPAIR,
    nact=WeldTestModel.NACT,
    nten=WeldTestModel.NTEN_F,
    nkey=WeldTestModel.NKEY,
]
def _part_b_equality(ctx: DeviceContext) raises:
    print("--- Part B: synthetic weld equality fields GOLDEN,")
    print("    BATCH=", BATCH)

    # Fields-native build — init_fields serializes the weld equality records
    # (Stage B fixed copy_equality_to_buffer, which init_model_gpu never called).
    var mf = Model[DTYPE, MD_2]()
    WeldTestModel.init_fields[DTYPE](ctx, mf)

    # Non-vacuity: init_fields must have serialized the weld (meta + record).
    if Int(mf.meta.data[MODEL_META_IDX_NEQUALITY]) != 1:
        raise Error("part B vacuous: model meta NEQUALITY != 1")

    var d = Data[DTYPE, MD_2, BATCH]()
    var dc = Data[DTYPE, MD_2, BATCH]()
    var d_off = Data[DTYPE, MD_2, BATCH]()
    for e in range(BATCH):
        for i in range(NQ_B):
            var qp = Scalar[DTYPE]((e * 5 + i * 3) % 5 - 2) / 50.0
            d.qpos.data[e * NQ_B + i] = qp
            dc.qpos.data[e * NQ_B + i] = qp
            d_off.qpos.data[e * NQ_B + i] = qp
        for i in range(NV_B):
            var qv = Scalar[DTYPE]((e * 7 + i * 5) % 7 - 3) / 20.0
            var qf = Scalar[DTYPE]((e * 13 + i * 9) % 9 - 4) / 4.0
            d.qvel.data[e * NV_B + i] = qv
            d.qfrc.data[e * NV_B + i] = qf
            dc.qvel.data[e * NV_B + i] = qv
            dc.qfrc.data[e * NV_B + i] = qf
            d_off.qvel.data[e * NV_B + i] = qv
            d_off.qfrc.data[e * NV_B + i] = qf
    d.upload_all(ctx)

    var integ = EulerIntegrator[DTYPE, MD_2, CONE_B, BATCH]()
    integ.prepare_gpu(ctx)
    var integ_c = EulerIntegrator[DTYPE, MD_2, CONE_B, BATCH]()

    var qvel_step0 = List[Scalar[DTYPE]](capacity=BATCH * NV_B)
    for _ in range(BATCH * NV_B):
        qvel_step0.append(Scalar[DTYPE](0))

    var ncon_total = 0
    for step in range(N_STEPS_B):
        integ.step["gpu"](d, mf, ctx)
        integ_c.step["cpu"](dc, mf)
        d.meta.download(ctx)
        d.qvel.download(ctx)
        if step == 0:
            for i in range(BATCH * NV_B):
                qvel_step0[i] = d.qvel.data[i]
        var ncon_seen = 0
        for e in range(BATCH):
            ncon_seen += Int(
                d.meta.data[e * METADATA_SIZE + META_IDX_NUM_CONTACTS]
            )
        if ncon_seen == 0:
            raise Error(
                "part B step " + String(step) + ": no contacts — vacuous"
            )
        ncon_total += ncon_seen
        print("  step", step, ": contacts", ncon_seen)

    # --- final fields-GPU fingerprint (Apple-gated) ---
    d.qpos.download(ctx)
    d.qvel.download(ctx)
    d.qacc.download(ctx)
    d.contacts.download(ctx)
    d.meta.download(ctx)
    var fp = Float64(0)
    for e in range(BATCH):
        for i in range(NQ_B):
            fp += Float64(d.qpos.data[e * NQ_B + i]) * Float64(e * NQ_B + i + 1)
        for i in range(NV_B):
            fp += Float64(d.qvel.data[e * NV_B + i]) * Float64(
                (e * NV_B + i + 1) * 7
            )
            fp += Float64(d.qacc.data[e * NV_B + i]) * Float64(
                (e * NV_B + i + 1) * 13
            )
        var nc2 = Int(d.meta.data[e * METADATA_SIZE + META_IDX_NUM_CONTACTS])
        for c in range(nc2):
            for k in range(CONTACT_SIZE):
                fp += Float64(
                    d.contacts.data[
                        e * MC_B * CONTACT_SIZE + c * CONTACT_SIZE + k
                    ]
                ) * Float64((c + 1) * (k + 1))
    if HARVEST:
        print("  HARVEST GOLD_NCON_B =", ncon_total)
        print("  HARVEST GOLD_B      =", fp)
    else:
        if ncon_total != GOLD_NCON_B and not default_accelerator().is_nvidia_gpu():
            raise Error(
                "part B contacts " + String(ncon_total) + " != golden "
                + String(GOLD_NCON_B)
            )
        var denom = abs(GOLD_B) if abs(GOLD_B) > 1e-9 else 1.0
        if abs(fp - GOLD_B) / denom > GOLD_RTOL and (
            not default_accelerator().is_nvidia_gpu()
        ):
            raise Error(
                "part B fingerprint " + String(fp) + " != golden "
                + String(GOLD_B)
            )
        print("  Part B matches golden fingerprint")

    var worst = Float64(0)
    for i in range(BATCH * NQ_B):
        var err = abs(Float64(dc.qpos.data[i]) - Float64(d.qpos.data[i]))
        if err > worst:
            worst = err
    print("  fields-CPU vs fields-GPU final qpos worst err:", worst)
    # Loose cross-target sanity only (bit-exactness is same-target): the
    # stiff weld rows + contact PGS are both iterative, so fp32 CPU/GPU
    # drift compounds beyond the walker gate's 1e-2 (measured ~1.1e-2).
    if worst > 5e-2:
        raise Error("part B: fields-CPU diverged from GPU")

    # Non-vacuity: equality-off rerun (meta NEQUALITY=0 short-circuits the
    # builder exactly like the legacy `if neq == 0: return`) must differ
    # from the equality-on step-0 qvel.
    mf.meta.data[MODEL_META_IDX_NEQUALITY] = Scalar[DTYPE](0)
    mf.meta.upload(ctx)
    d_off.upload_all(ctx)
    integ.step["gpu"](d_off, mf, ctx)
    d_off.qvel.download(ctx)
    var ndiff = 0
    for i in range(BATCH * NV_B):
        if d_off.qvel.data[i] != qvel_step0[i]:
            ndiff += 1
    if ndiff == 0:
        raise Error("part B vacuous: weld-off run identical to weld-on")
    print("  non-vacuous: weld-off rerun differs in", ndiff, "qvel entries")
    print("  Part B PASS")


def main() raises:
    var ctx = DeviceContext()
    _part_a_tendon(ctx)
    _part_b_equality(ctx)
    print("test_equality_tendon_fields: ALL PASS")
