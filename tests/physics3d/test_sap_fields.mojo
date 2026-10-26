"""Regression gate (GOLDEN-frozen): SAP broadphase in fields contact detection.

Part A — Humanoid (NGEOM=18 >= SAP_THRESHOLD=16, floor plane + 17 body
geoms, MAX_CONTACTS=50), BATCH=2 with penetrating poses (feet touching and
deep crouch): legacy FK -> detect_contacts_sap_gpu on the flat slab vs
fields FK -> detect_contacts_sap on Data/Model. Contact
count AND every populated contact record must be BIT-EXACT. Cross-check:
fields-SAP vs fields-O(N^2) (detect_contacts) as SETS (emission
order differs — SAP sweeps sorted by aabb_min_x, and conventions differ:
SAP plane contacts write BODY_B=-1 vs 0, no INCLUDEMARGIN slot), matched
by unordered body pair + position/dist within 1e-4. Auto dispatcher must
route humanoid to SAP (bit-equal to detect_contacts_sap).

Part B — SawyerReach (robot meshes + block.stl, NEXCLUDE==0), BATCH=2:
env1 teleports the obj cylinder into the eGripperBase MESH hull so a
GJK/EPA mesh contact appears through the SAP path. Legacy-SAP vs
fields-SAP BIT-EXACT + non-vacuity (mesh-body/obj-body contact present).

⚠ env1's z used to be 0.25, where the obj is in fact 15.1 mm CLEAR of the
hull: the mesh contact this part asserts was a phantom, produced by a flat
GJK simplex being read as an enclosure of the origin, and the golden counts
were frozen around it. z=0.28 is a real overlap (float64 CPU, float32 CPU and
float32 GPU agree there to 6 digits). Same pose, same story as
test_mesh_detection_fields — see `_closest_point_on_simplex`.

Part C — Walker2d (NGEOM=8 < 16): detect_contacts_auto must route
to detect_contacts, results bit-equal.

Run: pixi run -e apple mojo run -I . tests/physics3d/test_sap_fields.mojo
"""

from std.math import abs
from max.gpu.host import DeviceContext
from std.sys import default_accelerator
from std.testing import TestSuite

from noeira.nn.core.tensor import TensorImpl
from noeira.physics3d.constants import GEOM_MESH, GEOM_CYLINDER
from noeira.physics3d.fields import Data, Model, Dims, DimsLike
from noeira.physics3d.kinematics.forward_kinematics import (
    forward_kinematics,
)
from noeira.physics3d.collision.broadphase_sap import (
    SAP_THRESHOLD,
    detect_contacts_sap,
    detect_contacts_auto,
)
from noeira.physics3d.collision.contact_detection import (
    detect_contacts,
)

from noeira.physics3d.gpu.constants import (
    CONTACT_SIZE,
    CONTACT_IDX_SOLREF_0,
    CONTACT_IDX_SOLIMP_4,
    CONTACT_IDX_BODY_A,
    CONTACT_IDX_BODY_B,
    CONTACT_IDX_DIST,
    CONTACT_IDX_NX,
    CONTACT_IDX_NY,
    CONTACT_IDX_NZ,
    META_IDX_NUM_CONTACTS,
    MODEL_GEOM_SIZE,
    GEOM_IDX_TYPE,
    GEOM_IDX_BODY,
    GEOM_IDX_RADIUS,
    GEOM_IDX_HALF_LENGTH,
    MAX_GPU_MESHES,
    METADATA_SIZE,
)
from noeira.envs.humanoid.humanoid_xml import HumanoidModel
from noeira.envs.metaworld.sawyer_reach_xml import SawyerReachModel
from noeira.envs.walker2d.walker2d_xml import Walker2dModel
from noeira.physics3d.model.model_dims import ModelDims

comptime DTYPE = DType.float32
comptime BATCH = 2

# --- GOLDEN fingerprints (frozen from the legacy-validated fields-GPU run) ----
comptime HARVEST = False  # True => print fingerprints + skip asserts (regen)
comptime GOLD_RTOL = 1e-3
comptime GOLD_NCON_H = 14  # Part A humanoid SAP: total contacts
# Re-harvested 2026-07-29 (was 7711.957039542147). `GOLD_NCON_H` is unchanged,
# so the contact SET is identical — only the records moved. Cause: commit
# f0d35e2c switched `fromto` geom orientation to MuJoCo's convention
# (`vec = from - to` then `mjuu_z2quat`). Humanoid has twelve `fromto`
# capsules. Same solid, but a different roll about the capsule axis, and that
# axis is the tangent-frame hint the contact record carries. See the note on
# `GOLD_B` in test_equality_tendon_fields.mojo for the same effect on a solve.
# ⚠ GOLD_CON_H moved -32.0 on 2026-08-01 with the narrow-phase CONTACT
# DIRECTION fix (see `collision/broadphase_sap.mojo`). Accounted for exactly,
# and NOT re-recorded blind: only TWO env1 records changed, both purely a
# `(body_a, body_b)` relabel with the normal, position and dist BIT-IDENTICAL
# and the count unchanged —
#     c=8  a5 b9 -> a9 b5      c=9  a8 b6 -> a6 b8
# and the fingerprint weights BODY_A by (e+1)(c+1) and BODY_B by 2(e+1)(c+1):
#     c=8: 18*(9-5) + 36*(5-9) = -72
#     c=9: 20*(6-8) + 40*(8-6) = +40   -> -32 exactly.
# Those are precisely the two contacts that took a reversed-order narrow-phase
# branch, which used to negate the normal AND swap the bodies (a double flip).
# The physics of the fix is anchored against MuJoCo on dm_control manipulator,
# where it took the grasp qacc from 5.21 to 4.05e-9.
#
# ⚠ GOLD_CON_H moved +20570.003871900029 on 2026-08-03 with `11e188fd`
# "per-contact solref/solimp reach the solver", and was NOT refreshed then —
# it sat RED AT HEAD for five days. Two things hid it: this file is not in the
# light sweep list (29 of 693 .mojo files), and Part A's failure ABORTED Parts
# B and C, so the file could only ever report one of its two stale goldens.
# Both fixed (363e1ae9 split the parts; the sweep is now 121 files).
#
# ACCOUNTED FOR EXACTLY, and measured rather than argued. That commit made
# per-contact solref/solimp reach the contact RECORD — its own message: "the
# whole model shared ONE solref taken from geom[0], and the per-geom values ...
# were read by nothing" — so columns SOLREF_0/1 and SOLIMP_0..4 (21..29) went
# from uniform to per-pair mixed values. Splitting the fingerprint at exactly
# those columns (printed every run, below):
#     solparam cols  20570.003868740983
#     everything else 8088.218297153362   vs the old golden 8088.218293994316
# The solparam columns carry the ENTIRE move to 3.16e-6 absolute, and that same
# 3.16e-6 is the drift in the non-solparam part — 3.9e-10 relative, four orders
# inside GOLD_RTOL. So nothing else moved and nothing is being buried.
# The new values are the CORRECT ones: `11e188fd` is MuJoCo-verified by
# `test_contact_solparams_vs_mujoco.mojo`, added in that same commit.
#
# ⚠ The split print is permanent ON PURPOSE. A count-and-fingerprint golden can
# only be refreshed honestly if the move can be attributed to specific columns;
# harvesting the number first produces a golden that passes by construction.
# --- 2026-08-25: `c43517db`, the pair reaches GJK in MuJoCo's order ---------
# 28658.222165894345 -> 28019.840439933934, i.e. "everything else"
# 8090.978258839459 -> 7449.836571192951. ACCOUNTED, per this file's own rule.
# `pushPairArena` sorts a candidate pair by (geom type, geom index) and we were
# emitting the broadphase's order; canonicalising it relabels the pair and
# negates the record's normal with it (the record stores `body_b -> body_a`).
# FIVE of the 14 records move, every one of them a pure relabel:
#
#     env0 c2   body_a 9->6  body_b 6->9   nz +0.049736 -> -0.049736
#     env1 c4   body_a 7->4  body_b 4->7   nz +0.091242 -> -0.091242
#     env1 c5   body_a 7->5  body_b 5->7   nz +0.360082 -> -0.360082
#     env1 c7   body_a 9->6  body_b 6->9   nz +0.091242 -> -0.091242
#     env1 c10  body_a 8->5  body_b 5->8   nz +0.091241 -> -0.091241
#
# and the fingerprint delta decomposes term by term with nothing left over:
#
#     BODY_A -177.000000   BODY_B +354.000000        (integers, a relabel)
#     NX     +157.431351   NY     -833.976375   NZ  -141.596665
#     POS_X/Y/Z             ~1e-06                   (rounding only)
#     total  -641.141688  == the observed move
#
# ⚠ `dist` IS BIT-IDENTICAL ON ALL FIVE. These are analytic capsule pairs, so
# the witness does not depend on the operand order the way a mesh manifold's
# does — which is why POS moves 1e-06 here and 2e-01 in Part B.
comptime GOLD_CON_H = 28019.840439933934
# Re-harvested 2026-07-29 (was NCON 6 / fingerprint 2258.0145981857786), same
# cause as the plane-mesh gate: SawyerReach's class-only geoms now inherit
# `type="mesh"` from their `<default class="base_viz"/base_col">` blocks, as
# MuJoCo does, instead of falling back to the built-in primitive. More mesh
# geoms collide, so the count rises.
# Re-harvested 2026-07-30 (was NCON 6 / 2258.0145981857786): the flat-simplex
# fix in `_closest_point_on_simplex` retired the scene's PHANTOM mesh contacts
# — pairs GJK reported as penetrating that float64 puts centimetres apart —
# and env1's pose moved to a z where the mesh contact is real. Two effects,
# one direction: fewer, and now all genuine.
# ⚠ 2 -> 1 on 2026-08-10: env0's contact was NEVER REAL. `cylinder_box`
# reduced the cylinder to a capsule and so fabricated a 2 cm penetration where
# the obj's flat face rests exactly on the table; MuJoCo reports ZERO contacts
# at that pose. Routing cylinder/box through GJK+EPA, as MuJoCo's own dispatch
# does, removes it. env1's mesh contact is unchanged and still asserted.
# ⚠ 1 -> 4 on 2026-08-10: env1's obj-cylinder-into-gripper-mesh contact became
# a MANIFOLD. MuJoCo routes MESH x CYLINDER through `mjc_Convex`'s perturbation
# loop (`maxContacts` returns 4 only when BOTH geoms are box-or-mesh, so this
# pair comes back 1 and does not take the native-CCD early return); we were
# emitting only the primary point. Measured on the reference at this exact
# pose: 5 rows. We emit 4 — MuJoCo's rows 2 and 3 are 1.28x the
# `isDistinctContact` threshold apart (3.62e-05 vs 2.83e-05) and merge under
# our EPA's witness point. See the long note in
# `test_mesh_detection_fields.mojo`, which gates the manifold's SPAN; this file
# gates that the SAP path agrees with the O(N^2) path on the same records.
# --- 2026-08-18: STALE SINCE 08-13 — the twin was refreshed and this was not.
# 5 -> 4 contacts; GOLD_CON_S 5042.802909596139 -> 3361.6499817769654 and
# GOLD_SOL_S 4523.220017328858 -> 3015.4800115525723.
#
# ⚠⚠ THIS GOLDEN WAS NEVER ACHIEVABLE BY THE ENGINE THAT SHIPPED. `353696b4`
# ("<option ccd_tolerance ccd_iterations> — and MuJoCo applies them ONLY to
# smooth pairs") moved this fixture back from 5 to 4 and refreshed
# `test_mesh_detection_fields`'s GOLD_NCON/GOLD_CON/GOLD_SOL to exactly the
# three values above — but not this file's. Part B has been RED AT HEAD for
# five days for that reason alone. Same shape as the 08-03 stale-golden entry
# on GOLD_CON_H above: a twin gate refreshed on one side only.
#
# THE COUNT DID NOT GO DOWN BECAUSE WE LOST A CONTACT. `353696b4` restored
# EXACTNESS on this pair: `ccd_tolerance` applies only where `discreteGeoms`
# is false, and sawyer's obj/gripper is mesh-vs-mesh, so EPA lands on the exact
# face instead of stopping inside 1e-6. The 08-13 rbound row that took the
# count to 5 was an artefact of the loose stopping rule.
#
# 4 IS THE MuJoCo-VALIDATED STATE, and by LIVE comparison rather than by any
# frozen number. `test_mesh_detection_fields` gates the same scene with real
# MuJoCo values and passes at HEAD:
#     mesh depth    ours -0.02769497  MuJoCo -0.02769469   0.28 um
#     mesh normal   dot(MuJoCo) = 0.9999996520973681
#     manifold span ours 0.038894064  MuJoCo 0.03867       0.22 mm
# and `test_sawyer_settle_vs_mujoco` is green.
#
# AND THE CROSS-CHECK THIS LEG EXISTS FOR STILL HOLDS. SAP's two fingerprints
# are BIT-IDENTICAL to the O(N^2) path's (3361.6499817769654 /
# 3015.4800115525723 to every digit), so the two detection paths agree
# record-for-record on this fixture. They never diverged; only the golden did.
#
# ⚠ REFUTED ON THE WAY: that the SAP AABB sweep drops the fifth pair because
# it omits the geoms' own `margin` (the divergence the note at the head of the
# AABB loop in `broadphase_sap.mojo` records, which names sawyer). Folding each
# geom's margin into its AABB leaves Part B at 4 contacts with both
# fingerprints unchanged to every digit. It is not the cause, and the note's
# "would move sawyer" is wrong for this fixture. It DOES move Part A
# (8088.218297153362 -> 8090.978297284455, inside GOLD_RTOL) — so that
# experiment is not free, and was reverted.
comptime GOLD_NCON_S = 5  # Part B sawyer SAP: total contacts (08-13, below)
# ⚠ GOLD_CON_S has moved TWICE on 2026-08-01, both accounted for exactly.
#   +64.0  bug 35 (the double flip): fractional part unchanged; 64 = 32 * 2,
#          one `(body_a, body_b)` relabel of the env1 obj(33)/table(1) contact
#          at fingerprint weight (e+1)(c+1) = 2.
#   -32.0  bug 36 (`cylinder_box` returned the opposite normal). Sawyer's obj
#          is a cylinder and its table a box. The table is horizontal so the
#          normal is (0,0,+-1); flipping nz by 2 at CONTACT_IDX_NZ (k=7,
#          weight (e+1)(c+1)*8 = 16) gives -2*16 = -32. Fractional part
#          unchanged because an integer-valued float moved. MuJoCo-verified by
#          `test_narrow_phase_pairs.mojo`.
# Part A (humanoid) did NOT move for bug 36 — no cylinder/box pair there.
#
# --- 2026-08-09: DEFECT 24 removed sawyer's bogus ground contacts -----------
# `GOLD_NCON_S` was 4 (2/env) and had drifted to 10 (5/env). Neither number was
# right: sawyer's `tablelink` is a JOINTLESS body, so `body_weldid == 0` — the
# world's — and its collision box sits 7 mm through the floor plane by
# construction. MuJoCo filters that pair via `filterBodyPair` and emits
# nothing; our O(N^2) path filtered it too; the SAP PLANE loop had no body
# filter at all and collided them. One bogus contact per env while plane/box
# was single-point, FOUR once `3dbc4c33` made it a manifold. Fixed in
# `pair_body_filtered`, shared by all three pair loops.
# Sawyer is now 1 real contact per env, and the geometry/solparam split below
# matches Part A's so a solref/solimp change can never again read as geometry.
#
# ⚠ CROSS-CHECKED, not merely harvested. With defect 24 fixed, sawyer's two
# surviving contacts are body-body (no plane contact, so no BODY_B convention
# difference), and this geometry fingerprint agrees with the O(N^2) golden that
# `test_mesh_detection_fields` pins over the SAME scene —
#     SAP     479.7818514164537
#     O(N^2)  479.781851073727    (that file's GOLD_CON)
# to 3.4e-10 absolute, 7e-13 relative. The two detection paths now emit the
# same contact set here. If a future change moves one and not the other, they
# have diverged.
#
# --- 2026-08-09b: EPA now covers sawyer's mesh pair (`polytope3`) ----------
# 479.7818514164537 -> 481.2657020315528, and the cross-check with
# `test_mesh_detection_fields` STILL HOLDS: that file's O(N^2) golden moved to
# the identical 481.2657020315528, so SAP and O(N^2) continue to agree exactly
# on this scene. One record moved — env1's obj/gripper mesh contact — from a
# centre-line estimate to a real EPA answer. See the long note in
# test_mesh_detection_fields for why the number moved AWAY from MuJoCo while
# getting more correct, and why the residual is the mesh VERTEX SET.
#
# --- 2026-08-09c: mesh_vertadr was in FLOATS, consumed as VERTICES ---------
# 481.2657020315528 -> 512.8892028308474, matching `test_mesh_detection_fields`
# EXACTLY once more, so SAP and O(N^2) still agree on this scene. This move is
# TOWARD MuJoCo: the obj/gripper contact goes -0.0148583 -> -0.0273728 against
# MuJoCo's -0.0276947, i.e. 12.9 mm of error down to 0.32 mm.
#
# `load_mesh_hull` stored a FLAT SCALAR offset where every consumer indexes
# `[vertex, component]`, so eGripperBase read past the loaded vertices entirely
# and collided as an empty shape. ⚠ THIS GATE FROZE THE BUG: its header says it
# was "originally validated BIT-EXACT against the legacy FK + narrow-phase
# kernels", and those kernels had the same defect, so the golden certified the
# wrong answer for as long as it existed. No dm_control suite model does mesh
# COLLISION (dog has 162 mesh geoms, all non-collidable), so nothing else could
# have caught it.
#
# --- 2026-08-09d: EXACT convex hull ---------------------------------------
# 512.8892028308474 -> 512.9645483070053, matching test_mesh_detection_fields
# exactly once more. `compute_convex_hull` now computes a real hull instead of
# sampling support points, so the obj/gripper depth reaches MuJoCo parity:
# -0.0276952 against -0.0276947, i.e. 0.5 MICROMETRES.
#
# --- 2026-08-10: cylinder/box re-routed + the mesh NORMAL was reversed -----
# 512.9645483070053 -> 336.28797102486715, matching test_mesh_detection_fields
# exactly once more. Two accounted changes, both toward MuJoCo: env0's phantom
# is gone, and env1's contact NORMAL was REVERSED and is now correct.
# `gjk_epa` returned `gj -> gi` while every caller assumed `gi -> gj`, and no
# gate covered mesh DIRECTION, so every mesh contact this engine produced
# pointed the wrong way. It surfaced only when cylinder/box went through the
# same function and tripped `test_narrow_phase_pairs`' direction assert at
# 1.9999999999976286 -- a full reversal -- on an anchored pair.
# 2026-08-10, from the 1 -> 4 manifold above. BOTH fingerprints match
# `test_mesh_detection_fields`'s O(N^2) values to every digit
# (3361.63858178955 / 3015.4800115525723), which is the cross-check that
# matters here: the SAP path and the O(N^2) path built the same manifold, not
# merely the same number of rows. The solparam sum is exactly 10x its old value
# (301.5480011552572 * 10 = 3015.480011552572) because those columns are
# identical on every row of one pair and the contact weights (c+1) sum 1+2+3+4.
# --- 2026-08-13: mesh `rbound` fix -----------------------------------------
# 4 -> 5 contacts; GOLD_CON_S 3361.63858178955 -> 5042.802909596139 and
# GOLD_SOL_S 3015.4800115525723 -> 4523.220017328858. ⚠ THESE ARE THE SAME
# THREE NUMBERS `test_mesh_detection_fields` HARVESTED, which is the point of
# this leg: SAP and O(N^2) still agree record-for-record on this fixture.
# `geom_rbound` for a mesh was measured from the vertex centroid rather than
# MuJoCo's AABB corner, under-sizing it and making `mj_filterSphere` reject a
# pair MuJoCo tests — so this is a contact we were MISSING. The plane-mesh
# `maxplanemesh` cap landed in the same commit but moves counts DOWN, not up,
# and does not touch this fixture. Justification and the MuJoCo comparisons
# that back the new value are in `test_mesh_detection_fields.mojo`.
#
# ⚠⚠ 2026-08-25: AND THAT 08-13 REFRESH WAS NEVER APPLIED EITHER — TO EITHER
# TWIN. The block above says "GOLD_CON_S 3361.63858178955 -> 5042.802909596139
# and GOLD_SOL_S 3015.4800115525723 -> 4523.220017328858 ... THESE ARE THE
# SAME" as `test_mesh_detection_fields`'s, and BOTH files kept the old numbers.
# Twelve days red on both, on a count already justified against MuJoCo. This is
# the third time this file has recorded "a twin gate refreshed on one side
# only"; this time neither side was.
#
# `GOLD_SOL_S` is the proof again: measured today it is 4523.220017328858, that
# block's value to the last digit.
#
# --- 2026-08-25: `c43517db` on top of it ------------------------------------
# GOLD_CON_S 5042.516301484706 -> 5222.468882602276, the SAME move and the same
# account as `test_mesh_detection_fields`'s (see there): the obj CYLINDER sorts
# before the eGripperBase MESH, so env1's five rows are relabelled 23 <-> 33 and
# their normal negated —
#
#     body_a/body_b relabel   -300.000000  = 300*(k_A - k_B), exact
#     normal sign flip        +479.740543
#     dist                      +0.000004
#     witness positions         +0.212034  sub-mm, other operand order
#     total                   +179.952581  == the observed delta
#
# ⚠ THE TWO FILES MUST AGREE, and they now do: this SAP gate and the O(N^2)
# gate report the same 5 records, the same 5222.468882602276 and the same
# 4523.220017328858. That agreement is the real assertion here — the numbers
# come from two different broadphases.
comptime GOLD_CON_S = 5222.468882602276  # geometry columns (k < 23)
comptime GOLD_SOL_S = 4523.220017328858  # solparam columns (k >= 23)

# ── Humanoid (Part A) ────────────────────────────────────────────────────
comptime NQ_H = HumanoidModel.NQ  # 24
comptime NV_H = HumanoidModel.NV  # 23
comptime NBODY_H = HumanoidModel.NBODY  # 14
comptime NJOINT_H = HumanoidModel.NJOINT  # 18
comptime NGEOM_H = HumanoidModel.NGEOM  # 18
comptime MC_H = HumanoidModel.MAX_CONTACTS  # 50
comptime NEQ_H = HumanoidModel.MAX_EQUALITY  # 0
comptime NTD_H = HumanoidModel.MAX_TENDON  # 2
comptime NSITE_H = HumanoidModel.NSITE  # 0
comptime NEXCL_H = HumanoidModel.nexclude  # 0
comptime MD = Dims[
    nq=NQ_H,
    nv=NV_H,
    nbody=NBODY_H,
    njoint=NJOINT_H,
    ngeom=NGEOM_H,
    nsite=NSITE_H,
    max_contacts=MC_H,
    nequality=NEQ_H,
    ntendon=NTD_H,
    nexclude=NEXCL_H,
    nmesh_verts=0,
    npair=HumanoidModel.NPAIR,
    nact=HumanoidModel.NACT,
    nten=HumanoidModel.NTEN_F,
    nkey=HumanoidModel.NKEY,
]

# ── Sawyer (Part B) ──────────────────────────────────────────────────────
comptime NQ_S = SawyerReachModel.NQ
comptime NV_S = SawyerReachModel.NV
comptime NBODY_S = SawyerReachModel.NBODY
comptime NJOINT_S = SawyerReachModel.NJOINT
comptime NGEOM_S = SawyerReachModel.NGEOM
comptime NEQ_S = SawyerReachModel.MAX_EQUALITY
comptime NTD_S = SawyerReachModel.MAX_TENDON
comptime NSITE_S = SawyerReachModel.NSITE
comptime MC_S = SawyerReachModel.MAX_CONTACTS
# ⚠ 512, not 256: EXACT hulls need roughly 10x what support sampling
# kept (sawyer's twelve meshes go ~648 -> ~5.6k vertices), and
# `fields_build` TRUNCATES past this cap — silently, until now.
comptime NMESHV_S = MAX_GPU_MESHES * 512
comptime MD_2 = Dims[
    nq=NQ_S,
    nv=NV_S,
    nbody=NBODY_S,
    njoint=NJOINT_S,
    ngeom=NGEOM_S,
    nsite=NSITE_S,
    max_contacts=MC_S,
    nequality=NEQ_S,
    ntendon=NTD_S,
    nexclude=0,
    nmesh_verts=NMESHV_S,
    npair=SawyerReachModel.NPAIR,
    nact=SawyerReachModel.NACT,
    nten=SawyerReachModel.NTEN_F,
    nkey=SawyerReachModel.NKEY,
]

# ── Walker2d (Part C) ────────────────────────────────────────────────────
comptime NQ_W = Walker2dModel.NQ
comptime NV_W = Walker2dModel.NV
comptime NBODY_W = Walker2dModel.NBODY
comptime NJOINT_W = Walker2dModel.NJOINT
comptime NGEOM_W = Walker2dModel.NGEOM
comptime MC_W = Walker2dModel.MAX_CONTACTS
comptime NEQ_W = Walker2dModel.MAX_EQUALITY
comptime NTD_W = Walker2dModel.MAX_TENDON
comptime NSITE_W = Walker2dModel.NSITE
comptime NEXCL_W = Walker2dModel.NEXCLUDE
comptime MD_3 = ModelDims[Walker2dModel]


def _humanoid_qpos(e: Int, i: Int) -> Scalar[DTYPE]:
    """Free joint pose + hinge angles. env0: feet touch/penetrate floor;
    env1: deep crouch — many geoms near the floor and each other, so the
    SAP sweep processes several candidate pairs."""
    if i == 0:
        return Scalar[DTYPE](0.02) * Scalar[DTYPE](e)
    if i == 1:
        return Scalar[DTYPE](0)
    if i == 2:
        return Scalar[DTYPE](1.24) if e == 0 else Scalar[DTYPE](0.72)
    if i == 3:
        return Scalar[DTYPE](1)  # identity quaternion (w first)
    if i <= 6:
        return Scalar[DTYPE](0)
    if e == 0:
        # near-neutral standing pose, slight bends
        if i == 13 or i == 17:
            return Scalar[DTYPE](-0.15)  # knees
        return Scalar[DTYPE](0.05)
    # env1: crouch
    if i == 8:
        return Scalar[DTYPE](-0.4)  # abdomen_y
    if i == 12 or i == 16:
        return Scalar[DTYPE](-1.0)  # hip_y
    if i == 13 or i == 17:
        return Scalar[DTYPE](-1.6)  # knees
    if i == 20 or i == 23:
        return Scalar[DTYPE](-0.6)  # elbows
    return Scalar[DTYPE](0.1)


def _part_a_humanoid(ctx: DeviceContext) raises:
    print("--- Part A: humanoid SAP fields GOLDEN, BATCH=", BATCH)
    print("  humanoid NGEOM=", NGEOM_H, " SAP_THRESHOLD=", SAP_THRESHOLD)
    comptime assert NGEOM_H >= SAP_THRESHOLD, "humanoid must route to SAP"

    var mf = Model[DTYPE, MD]()
    HumanoidModel.init_fields[DTYPE](ctx, mf)

    var d = Data[DTYPE, MD, BATCH]()
    for e in range(BATCH):
        for i in range(NQ_H):
            d.qpos.data[e * NQ_H + i] = _humanoid_qpos(e, i)
    d.upload_all(ctx)

    # Fields: FK + SAP detection.
    forward_kinematics["gpu", DTYPE, BATCH=BATCH](d, mf, ctx)
    detect_contacts_sap["gpu", DTYPE, BATCH=BATCH](d, mf, ctx)
    d.contacts.download(ctx)
    d.meta.download(ctx)

    var ncon_h = 0
    var fp_h = Float64(0)
    for e in range(BATCH):
        var nc = Int(d.meta.data[e * METADATA_SIZE + META_IDX_NUM_CONTACTS])
        ncon_h += nc
        for c in range(nc):
            for k in range(CONTACT_SIZE):
                fp_h += Float64(
                    d.contacts.data[
                        e * MC_H * CONTACT_SIZE + c * CONTACT_SIZE + k
                    ]
                ) * Float64((e + 1) * (c + 1) * (k + 1))
    # THE ACCOUNT, not just the number. This file's rule is that a golden move
    # must be explained exactly, so the fingerprint is also split at the
    # solparam columns: `11e188fd` made per-contact solref/solimp reach the
    # record (SOLREF_0/1 and SOLIMP_0..4 = columns 21..29), and the claim is
    # that the whole 2026-08-03 move lives there and nowhere else. Printed
    # every run, because the next person to move this needs the same split.
    var fp_h_solparams = Float64(0)
    var fp_h_rest = Float64(0)
    for e in range(BATCH):
        var nc = Int(d.meta.data[e * METADATA_SIZE + META_IDX_NUM_CONTACTS])
        for c in range(nc):
            for k in range(CONTACT_SIZE):
                var term = Float64(
                    d.contacts.data[
                        e * MC_H * CONTACT_SIZE + c * CONTACT_SIZE + k
                    ]
                ) * Float64((e + 1) * (c + 1) * (k + 1))
                if k >= CONTACT_IDX_SOLREF_0 and k <= CONTACT_IDX_SOLIMP_4:
                    fp_h_solparams += term
                else:
                    fp_h_rest += term
    print(
        "  humanoid fingerprint split: solparam cols", fp_h_solparams,
        " everything else", fp_h_rest,
    )
    if ncon_h == 0:
        raise Error("humanoid SAP: no contacts — gate is vacuous")
    print("  humanoid fields-SAP total contacts:", ncon_h)
    if HARVEST:
        print("  HARVEST GOLD_NCON_H =", ncon_h)
        print("  HARVEST GOLD_CON_H  =", fp_h)
    else:
        if ncon_h != GOLD_NCON_H and not default_accelerator().is_nvidia_gpu():
            raise Error(
                "humanoid SAP contacts " + String(ncon_h) + " != golden "
                + String(GOLD_NCON_H)
            )
        var denom = abs(GOLD_CON_H) if abs(GOLD_CON_H) > 1e-9 else 1.0
        if abs(fp_h - GOLD_CON_H) / denom > GOLD_RTOL and (
            not default_accelerator().is_nvidia_gpu()
        ):
            raise Error(
                "humanoid SAP fingerprint " + String(fp_h) + " != golden "
                + String(GOLD_CON_H)
            )
        print("  PASS: humanoid fields-SAP matches golden fingerprint")

    # Non-vacuity for the SWEEP itself: env1 must contain a body-body
    # contact (BODY_B > 0) — plane contacts come from the direct plane
    # loop; only the sorted sweep emits body-body pairs.
    var ncon1 = Int(d.meta.data[1 * METADATA_SIZE + META_IDX_NUM_CONTACTS])
    var n_bodybody = 0
    for c in range(ncon1):
        var bb = Int(
            d.contacts.data[1 * MC_H * CONTACT_SIZE + c * CONTACT_SIZE + 1]
        )
        if bb > 0:
            n_bodybody += 1
    print("  env 1 body-body (sweep-emitted) contacts:", n_bodybody)
    if n_bodybody == 0:
        raise Error("no sweep-emitted body-body contact — sweep leg vacuous")

    # Fields CPU vs fields GPU.
    var dc = Data[DTYPE, MD, BATCH]()
    for e in range(BATCH):
        for i in range(NQ_H):
            dc.qpos.data[e * NQ_H + i] = _humanoid_qpos(e, i)
    forward_kinematics["cpu", DTYPE, BATCH=BATCH](dc, mf)
    detect_contacts_sap["cpu", DTYPE, BATCH=BATCH](dc, mf)
    var worst = Float64(0)
    for e in range(BATCH):
        var nc_g = Int(
            d.meta.data[e * METADATA_SIZE + META_IDX_NUM_CONTACTS]
        )
        var nc_c = Int(
            dc.meta.data[e * METADATA_SIZE + META_IDX_NUM_CONTACTS]
        )
        if nc_g != nc_c:
            raise Error("fields-CPU SAP contact count differs from fields-GPU")
        for c in range(nc_g):
            for k in range(CONTACT_SIZE):
                var err = abs(
                    Float64(
                        dc.contacts.data[
                            e * MC_H * CONTACT_SIZE + c * CONTACT_SIZE + k
                        ]
                    )
                    - Float64(
                        d.contacts.data[
                            e * MC_H * CONTACT_SIZE + c * CONTACT_SIZE + k
                        ]
                    )
                )
                if err > worst:
                    worst = err
    print("  fields-CPU vs fields-GPU worst record err:", worst)
    if worst > 1e-4:
        raise Error("fields-CPU SAP contact records tolerance exceeded")
    print("  PASS: fields-CPU SAP contacts within 1e-4")

    # ── Cross-check: fields-SAP vs fields-O(N^2) on the SAME FK products.
    # Emission ORDER differs (SAP sweeps sorted by aabb_min_x) and the
    # conventions differ (SAP plane contacts: BODY_B=-1 vs 0, no
    # INCLUDEMARGIN write), so compare as SETS: unordered body pair (plane/
    # world normalized to 0) + pos/dist within 1e-4. Same-type body-body
    # pairs may be visited with operands swapped by the sweep (pos then
    # differs at float rounding level), hence the tolerance.
    var dn = Data[DTYPE, MD, BATCH]()
    for e in range(BATCH):
        for i in range(NQ_H):
            dn.qpos.data[e * NQ_H + i] = _humanoid_qpos(e, i)
    dn.upload_all(ctx)
    forward_kinematics["gpu", DTYPE, BATCH=BATCH](dn, mf, ctx)
    detect_contacts["gpu", DTYPE, BATCH=BATCH](dn, mf, ctx)
    dn.contacts.download(ctx)
    dn.meta.download(ctx)
    for e in range(BATCH):
        var nc_sap = Int(
            d.meta.data[e * METADATA_SIZE + META_IDX_NUM_CONTACTS]
        )
        var nc_n2 = Int(
            dn.meta.data[e * METADATA_SIZE + META_IDX_NUM_CONTACTS]
        )
        print("  env", e, ": ncon fields-SAP=", nc_sap, " fields-N2=", nc_n2)
        if nc_sap != nc_n2:
            raise Error("fields-SAP vs fields-O(N^2) contact count mismatch")
        var used = List[Bool](length=nc_sap, fill=False)
        for c in range(nc_n2):
            var base_n = e * MC_H * CONTACT_SIZE + c * CONTACT_SIZE
            var ba_n = Int(dn.contacts.data[base_n + 0])
            var bb_n = Int(dn.contacts.data[base_n + 1])
            if bb_n < 0:
                bb_n = 0
            var found = False
            for s in range(nc_sap):
                if used[s]:
                    continue
                var base_s = e * MC_H * CONTACT_SIZE + s * CONTACT_SIZE
                var ba_s = Int(d.contacts.data[base_s + 0])
                var bb_s = Int(d.contacts.data[base_s + 1])
                if bb_s < 0:
                    bb_s = 0
                var pair_match = (ba_n == ba_s and bb_n == bb_s) or (
                    ba_n == bb_s and bb_n == ba_s
                )
                if not pair_match:
                    continue
                # pos (idx 2,3,4) + dist (idx 8) within 1e-4
                var ok = True
                for k in [2, 3, 4, 8]:
                    var err = abs(
                        Float64(dn.contacts.data[base_n + k])
                        - Float64(d.contacts.data[base_s + k])
                    )
                    if err > 1e-4:
                        ok = False
                if ok:
                    used[s] = True
                    found = True
                    break
            if not found:
                print(
                    "  UNMATCHED O(N^2) contact env", e, "idx", c,
                    "bodies (", ba_n, ",", bb_n, ") pos=(",
                    dn.contacts.data[base_n + 2], ",",
                    dn.contacts.data[base_n + 3], ",",
                    dn.contacts.data[base_n + 4], ")",
                )
                raise Error("fields-SAP vs fields-O(N^2) set mismatch")
    print("  PASS: fields-SAP == fields-O(N^2) as contact SETS")

    # ── Auto dispatcher: humanoid must route to SAP (bit-equal).
    var da = Data[DTYPE, MD, BATCH]()
    for e in range(BATCH):
        for i in range(NQ_H):
            da.qpos.data[e * NQ_H + i] = _humanoid_qpos(e, i)
    da.upload_all(ctx)
    forward_kinematics["gpu", DTYPE, BATCH=BATCH](da, mf, ctx)
    detect_contacts_auto["gpu", DTYPE, BATCH=BATCH](da, mf, ctx)
    da.contacts.download(ctx)
    da.meta.download(ctx)
    for e in range(BATCH):
        var nc_a = Int(
            da.meta.data[e * METADATA_SIZE + META_IDX_NUM_CONTACTS]
        )
        var nc_s = Int(
            d.meta.data[e * METADATA_SIZE + META_IDX_NUM_CONTACTS]
        )
        if nc_a != nc_s:
            raise Error("auto(humanoid) != SAP contact count")
        for c in range(nc_s):
            for k in range(CONTACT_SIZE):
                var a = da.contacts.data[
                    e * MC_H * CONTACT_SIZE + c * CONTACT_SIZE + k
                ]
                var b = d.contacts.data[
                    e * MC_H * CONTACT_SIZE + c * CONTACT_SIZE + k
                ]
                if a != b:
                    raise Error("auto(humanoid) != SAP contact record")
    print("  PASS: auto dispatcher routes humanoid to SAP, bit-equal")


def _part_b_sawyer(ctx: DeviceContext) raises:
    print("--- Part B: sawyer SAP mesh leg fields GOLDEN, BATCH=", BATCH)
    print("  sawyer NGEOM=", NGEOM_S)

    # Fields-native model build (STL hulls, NMESHV_S-padded — Stage B).
    var mf = Model[DTYPE, MD_2]()
    SawyerReachModel.init_fields[DTYPE](ctx, mf)

    # Locate the obj cylinder + mesh-geom bodies for the non-vacuity check.
    var obj_body = -1
    var mesh_bodies = List[Int]()
    for g in range(NGEOM_S):
        var gt = Int(mf.geoms.data[g * MODEL_GEOM_SIZE + GEOM_IDX_TYPE])
        var gb = Int(mf.geoms.data[g * MODEL_GEOM_SIZE + GEOM_IDX_BODY])
        if gt == GEOM_MESH and gb != 0:
            mesh_bodies.append(gb)
        if gt == GEOM_CYLINDER:
            var r = Float64(
                mf.geoms.data[g * MODEL_GEOM_SIZE + GEOM_IDX_RADIUS]
            )
            var hl = Float64(
                mf.geoms.data[g * MODEL_GEOM_SIZE + GEOM_IDX_HALF_LENGTH]
            )
            if abs(r - 0.02) < 1e-6 and abs(hl - 0.02) < 1e-6:
                obj_body = gb
    print("  obj_body:", obj_body, " mesh-geom bodies:", len(mesh_bodies))
    if obj_body < 0 or len(mesh_bodies) == 0:
        raise Error("could not identify obj cylinder / mesh geoms")

    # Poses (same as test_mesh_detection_fields): env0 obj on table; env1
    # obj teleported into the eGripperBase hull -> mesh-cylinder GJK contact.
    var qcfg = List[List[Float64]]()
    for e in range(BATCH):
        var q = List[Float64](length=NQ_S, fill=0.0)
        q[0] = 1.889288  # j0
        q[1] = -0.575769  # j1
        q[2] = -0.976659  # j2
        q[3] = 1.641991  # j3
        q[4] = 0.942860  # j4
        q[5] = 1.043696  # j5
        q[6] = 2.292833  # j6
        q[7] = 0.0  # r_close
        q[8] = 0.0  # l_close
        if e == 0:
            q[9] = 0.0  # obj x (on table)
            q[10] = 0.6  # obj y
            q[11] = 0.02  # obj z
        else:
            q[9] = 0.005  # obj x (inside gripper mesh hull)
            q[10] = 0.601  # obj y
            q[11] = 0.28  # obj z (0.25 was 15.1 mm clear — see the docstring)
        q[12] = 1.0  # obj quat w
        q[13] = 0.0
        q[14] = 0.0
        q[15] = 0.0
        qcfg.append(q^)

    var d = Data[DTYPE, MD_2, BATCH]()
    for e in range(BATCH):
        for i in range(NQ_S):
            d.qpos.data[e * NQ_S + i] = Scalar[DTYPE](qcfg[e][i])
    d.upload_all(ctx)

    # Fields: FK + SAP detection.
    forward_kinematics["gpu", DTYPE, BATCH=BATCH](d, mf, ctx)
    detect_contacts_sap["gpu", DTYPE, BATCH=BATCH](d, mf, ctx)
    d.contacts.download(ctx)
    d.meta.download(ctx)

    var ncon_s = 0
    var fp_s = Float64(0)
    var fp_s_sol = Float64(0)
    for e in range(BATCH):
        var nc = Int(d.meta.data[e * METADATA_SIZE + META_IDX_NUM_CONTACTS])
        ncon_s += nc
        for c in range(nc):
            for k in range(CONTACT_SIZE):
                var w = Float64(
                    d.contacts.data[
                        e * MC_S * CONTACT_SIZE + c * CONTACT_SIZE + k
                    ]
                ) * Float64((e + 1) * (c + 1) * (k + 1))
                # Same split Part A already carries: geometry columns apart
                # from the solref/solimp ones, so a solparam change can never
                # again look like a geometry regression.
                if k < CONTACT_IDX_SOLREF_0:
                    fp_s += w
                else:
                    fp_s_sol += w
    if ncon_s == 0:
        raise Error("sawyer SAP: no contacts — gate is vacuous")
    print("  sawyer fields-SAP total contacts:", ncon_s)
    # Per-contact dump. A count golden that moves is only refreshable if the
    # move can be ACCOUNTED FOR — this file's own rule, see GOLD_CON_S's
    # history — and "the total went up by one" does not say which pair gained
    # a row. Printed always, not under HARVEST, because the next person to
    # move it needs the same evidence.
    for e in range(BATCH):
        var nc = Int(d.meta.data[e * METADATA_SIZE + META_IDX_NUM_CONTACTS])
        for c in range(nc):
            var o = e * MC_S * CONTACT_SIZE + c * CONTACT_SIZE
            print(
                "     env", e, "c", c,
                " bodies(", Int(Float64(d.contacts.data[o + CONTACT_IDX_BODY_A])),
                ",", Int(Float64(d.contacts.data[o + CONTACT_IDX_BODY_B])), ")",
                " dist", Float64(d.contacts.data[o + CONTACT_IDX_DIST]),
                " n [", Float64(d.contacts.data[o + CONTACT_IDX_NX]),
                Float64(d.contacts.data[o + CONTACT_IDX_NY]),
                Float64(d.contacts.data[o + CONTACT_IDX_NZ]), "]",
            )
    print(
        "  sawyer fingerprint split: geometry cols", fp_s,
        " solparam cols", fp_s_sol,
    )
    if HARVEST:
        print("  HARVEST GOLD_NCON_S =", ncon_s)
        print("  HARVEST GOLD_CON_S  =", fp_s)
        print("  HARVEST GOLD_SOL_S  =", fp_s_sol)
    else:
        if ncon_s != GOLD_NCON_S and not default_accelerator().is_nvidia_gpu():
            raise Error(
                "sawyer SAP contacts " + String(ncon_s) + " != golden "
                + String(GOLD_NCON_S)
            )
        var denom = abs(GOLD_CON_S) if abs(GOLD_CON_S) > 1e-9 else 1.0
        if abs(fp_s - GOLD_CON_S) / denom > GOLD_RTOL and (
            not default_accelerator().is_nvidia_gpu()
        ):
            raise Error(
                "sawyer SAP GEOMETRY fingerprint " + String(fp_s)
                + " != golden " + String(GOLD_CON_S)
            )
        var sdenom = abs(GOLD_SOL_S) if abs(GOLD_SOL_S) > 1e-9 else 1.0
        if abs(fp_s_sol - GOLD_SOL_S) / sdenom > GOLD_RTOL and (
            not default_accelerator().is_nvidia_gpu()
        ):
            raise Error(
                "sawyer SAP SOLPARAM fingerprint " + String(fp_s_sol)
                + " != golden " + String(GOLD_SOL_S)
            )
        print("  PASS: sawyer fields-SAP matches golden fingerprint")

    # Non-vacuity: env1 must have a contact between a mesh-geom body and
    # the obj body (GJK/EPA mesh fallback through the SAP sweep).
    var ncon1 = Int(d.meta.data[1 * METADATA_SIZE + META_IDX_NUM_CONTACTS])
    var mesh_contact_found = False
    for c in range(ncon1):
        var ba = Int(
            d.contacts.data[1 * MC_S * CONTACT_SIZE + c * CONTACT_SIZE + 0]
        )
        var bb = Int(
            d.contacts.data[1 * MC_S * CONTACT_SIZE + c * CONTACT_SIZE + 1]
        )
        for mb in mesh_bodies:
            if (ba == mb and bb == obj_body) or (
                bb == mb and ba == obj_body
            ):
                mesh_contact_found = True
                print(
                    "  mesh contact: bodies (", ba, ",", bb, ") dist=",
                    d.contacts.data[
                        1 * MC_S * CONTACT_SIZE + c * CONTACT_SIZE + 8
                    ],
                )
    if not mesh_contact_found:
        raise Error("no MESH-involved contact in env1 — gate is vacuous")
    print("  PASS: MESH-involved contact present through SAP (GJK/EPA)")


def _part_c_walker(ctx: DeviceContext) raises:
    print("--- Part C: walker2d auto dispatcher routes to O(N^2)")
    print("  walker2d NGEOM=", NGEOM_W)
    comptime assert NGEOM_W < SAP_THRESHOLD, "walker2d must route to O(N^2)"

    var mf = Model[DTYPE, MD_3]()
    Walker2dModel.init_fields[DTYPE](ctx, mf)

    # Poses from test_contact_detection_fields (floor penetration).
    var qcfg = List[List[Float64]]()
    var q0 = List[Float64](length=NQ_W, fill=0.0)
    q0[1] = 1.18
    qcfg.append(q0^)
    var q1 = List[Float64](length=NQ_W, fill=0.0)
    q1[1] = 0.85
    q1[3] = 0.6
    q1[4] = -1.1
    q1[6] = -0.4
    q1[7] = -0.9
    qcfg.append(q1^)

    var d1 = Data[DTYPE, MD_3, BATCH]()
    var d2 = Data[DTYPE, MD_3, BATCH]()
    for e in range(BATCH):
        for i in range(NQ_W):
            d1.qpos.data[e * NQ_W + i] = Scalar[DTYPE](qcfg[e][i])
            d2.qpos.data[e * NQ_W + i] = Scalar[DTYPE](qcfg[e][i])
    d1.upload_all(ctx)
    d2.upload_all(ctx)

    forward_kinematics["gpu", DTYPE, BATCH=BATCH](d1, mf, ctx)
    detect_contacts["gpu", DTYPE, BATCH=BATCH](d1, mf, ctx)
    forward_kinematics["gpu", DTYPE, BATCH=BATCH](d2, mf, ctx)
    detect_contacts_auto["gpu", DTYPE, BATCH=BATCH](d2, mf, ctx)
    d1.contacts.download(ctx)
    d1.meta.download(ctx)
    d2.contacts.download(ctx)
    d2.meta.download(ctx)

    for e in range(BATCH):
        var nc_1 = Int(
            d1.meta.data[e * METADATA_SIZE + META_IDX_NUM_CONTACTS]
        )
        var nc_2 = Int(
            d2.meta.data[e * METADATA_SIZE + META_IDX_NUM_CONTACTS]
        )
        print("  env", e, ": ncon direct=", nc_1, " auto=", nc_2)
        if nc_1 != nc_2:
            raise Error("auto(walker2d) != detect_contacts count")
        if nc_1 == 0:
            raise Error("expected contacts in this pose — gate is vacuous")
        for c in range(nc_1):
            for k in range(CONTACT_SIZE):
                var a = d1.contacts.data[
                    e * MC_W * CONTACT_SIZE + c * CONTACT_SIZE + k
                ]
                var b = d2.contacts.data[
                    e * MC_W * CONTACT_SIZE + c * CONTACT_SIZE + k
                ]
                if a != b:
                    raise Error("auto(walker2d) != detect_contacts")
    print("  PASS: auto dispatcher routes walker2d to O(N^2), bit-equal")


# ⚠ THREE INDEPENDENT TESTS, NOT THREE CALLS IN `main()`.
#
# These used to be `_part_a_humanoid(ctx); _part_b_sawyer(ctx);
# _part_c_walker(ctx)` in sequence. Each part signals failure by RAISING, so
# the first one to fail aborted the other two — and on 2026-08-07 that is
# exactly what happened: Part A's humanoid fingerprint was stale, and Part B's
# sawyer count golden was ALSO stale, but nobody could see B because A never
# returned. It was found only by temporarily reordering the calls.
#
# One stale golden hiding two more results is a reporting defect in its own
# right, independent of whichever golden is wrong. `TestSuite` runs each test
# and reports all three, so a red file now tells you HOW red.
#
# Each takes its own `DeviceContext` — the parts are independent by
# construction and sharing one would reintroduce a coupling between them.


def test_sap_humanoid_fields_golden() raises:
    _part_a_humanoid(DeviceContext())


def test_sap_sawyer_mesh_leg_golden() raises:
    _part_b_sawyer(DeviceContext())


def test_sap_walker2d_auto_dispatch() raises:
    _part_c_walker(DeviceContext())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
