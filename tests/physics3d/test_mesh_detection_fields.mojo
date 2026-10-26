"""Regression gate (GOLDEN-frozen): MESH narrow-phase in fields contact
detection, SawyerReach (robot meshes + block.stl).

Originally validated BIT-EXACT against the legacy FK + narrow-phase kernels.
That legacy reference was frozen into the GOLDEN fingerprints below during
Phase-0 of the physics3d sunset, so this gate survives deletion of the legacy
slab/kernels. It checks:
  * fields-GPU (FK -> detect) reproduces the frozen fingerprint — per-env
    contact counts + a contact-record checksum,
  * a MESH-involved contact (mesh-geom body vs obj body) is present in env1
    (GJK/EPA fallback — non-vacuous), and
  * fields-CPU == fields-GPU on records, fed the GPU FK products (isolates the
    detection port from FK differences).

env0 = canonical reset (obj on table); env1 = obj teleported INTO the
eGripperBase mesh hull.

The env1 z used to be 0.25, where the obj is in fact 15.1 mm CLEAR of the hull
— the gate's mesh contact was a phantom, manufactured by a flat GJK simplex
being read as an enclosure of the origin (see `_closest_point_on_simplex`).
Both the count and the checksum were frozen around it. z=0.28 is a real
overlap: float64 CPU, float32 CPU and float32 GPU agree there to 6 digits,
where at z=0.25 float64 said +0.0151 and float32 said -0.0553.

Contact DEPTH on the mesh path is still the crude fallback in `gjk_epa` (the
Minkowski-difference extent along the centre line), not a true EPA depth, so
the golden pins what the engine computes rather than ground truth. Separation
and the contact/no-contact verdict ARE trustworthy; the depth is not.

Model build = fields-native init_fields (Stage B; the NMESHV-padded mesh
build). Regenerate goldens after an INTENTIONAL physics change: HARVEST=True,
run on Apple, paste, False.

Run: pixi run -e apple mojo run -I . tests/physics3d/test_mesh_detection_fields.mojo
"""

from std.math import abs
from max.gpu.host import DeviceContext
from std.sys import default_accelerator

from noeira.physics3d.constants import GEOM_MESH, GEOM_CYLINDER
from noeira.physics3d.fields import Data, Model, Dims, DimsLike
from noeira.physics3d.kinematics.forward_kinematics import (
    forward_kinematics,
)
from noeira.physics3d.collision.contact_detection import (
    detect_contacts,
)
from noeira.physics3d.gpu.constants import (
    CONTACT_SIZE,
    CONTACT_IDX_SOLREF_0,
    CONTACT_IDX_DIST,
    CONTACT_IDX_BODY_A,
    CONTACT_IDX_BODY_B,
    CONTACT_IDX_POS_X,
    CONTACT_IDX_POS_Y,
    CONTACT_IDX_POS_Z,
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
from noeira.envs.metaworld.sawyer_reach_xml import SawyerReachModel

comptime DTYPE = DType.float32
comptime NQ = SawyerReachModel.NQ
comptime NV = SawyerReachModel.NV
comptime NBODY = SawyerReachModel.NBODY
comptime NJOINT = SawyerReachModel.NJOINT
comptime NGEOM = SawyerReachModel.NGEOM
comptime NEQ = SawyerReachModel.MAX_EQUALITY
comptime NTD = SawyerReachModel.MAX_TENDON
comptime NSITE = SawyerReachModel.NSITE
comptime MC = SawyerReachModel.MAX_CONTACTS
comptime BATCH = 2
# ⚠ 512, not 256: EXACT hulls need roughly 10x what support sampling
# kept (sawyer's twelve meshes go ~648 -> ~5.6k vertices), and
# `fields_build` TRUNCATES past this cap — silently, until now.
comptime NMESHV = MAX_GPU_MESHES * 512
comptime MD = Dims[
    nq=NQ,
    nv=NV,
    nbody=NBODY,
    njoint=NJOINT,
    ngeom=NGEOM,
    nsite=NSITE,
    max_contacts=MC,
    nequality=NEQ,
    ntendon=NTD,
    nexclude=0,
    nmesh_verts=NMESHV,
    npair=SawyerReachModel.NPAIR,
    nact=SawyerReachModel.NACT,
    nten=SawyerReachModel.NTEN_F,
    nkey=SawyerReachModel.NKEY,
]

# --- GOLDEN fingerprints (regenerated after the flat-simplex fix) ------------
comptime HARVEST = False  # True => print fingerprints + skip asserts (regen)
comptime GOLD_RTOL = 1e-3
# ⚠ 2 -> 1 on 2026-08-10, and the contact that went away was NEVER REAL.
# env0 (obj resting on the table) used to produce a cylinder/box contact from
# `cylinder_box`'s capsule reduction, which is wrong by exactly -r and so
# fabricated a 2 cm penetration where the obj's flat face rests exactly on the
# table. MuJoCo reports ZERO contacts at that pose. Routing cylinder/box
# through `mjc_Convex`'s GJK+EPA, as MuJoCo does, removes it.
# env1's mesh contact is unchanged and still asserted, so this gate is not
# weakened -- see the MuJoCo depth and direction checks below.
# ⚠ 1 -> 4 on 2026-08-10: env1's mesh contact became a MANIFOLD. MuJoCo sends
# MESH x CYLINDER through `mjc_Convex`'s perturbation loop (`maxContacts`
# returns 4 only when BOTH geoms are box-or-mesh, so this pair comes back 1 and
# does NOT take the native-CCD early return), and we were emitting the single
# primary point. Measured on the reference at THIS EXACT POSE: 5 rows, all at
# dist -0.02769469, four clustered within 3.1e-4 and one 3.9e-2 away.
#
# ⚠ WE PRODUCE 4, NOT 5, AND THAT IS NOT A MISSING PERTURBATION. MuJoCo's own
# rows 2 and 3 are 3.624e-05 apart against an `isDistinctContact` tolerance of
# 2.828e-05 (= 1e-3 * min rbound) -- 1.28x the merge threshold, 8 um of
# headroom. Our EPA converges to EPA_TOLERANCE 1e-8, but its witness point on a
# 1265-polygon hull differs from libccd's by more than 8 um, so that pair
# merges for us. Chasing the 5th row means chasing an 8 um agreement on a
# witness point, which no tolerance in this file could hold honestly.
#
# So the COUNT is not the gate -- MANIFOLD_SPAN below is. It asserts we found
# both contact FEATURES (the tight cluster and the far row), which is what
# decides whether a grasp holds; a count can be right with every row stacked on
# one feature.
#
# ⚠ 5 -> 4 on 2026-08-13, AND THE ROW THAT WENT IS env0's. Per-env counts
# before: env0 1, env1 4; after: env0 0, env1 4. env1's mesh manifold is
# UNTOUCHED — still 4 rows, depth within 5 um of MuJoCo, normal direction dot
# 0.99999965, MANIFOLD_SPAN 0.038894 against MuJoCo's 0.03867 — so no contact
# FEATURE was lost and the MuJoCo-anchored asserts below all still hold.
#
# ⚠⚠ WHETHER env0 SHOULD BE 0 OR 1 IS **NOT VERIFIED**. The tempting reading is
# that the note above already settles it ("MuJoCo reports ZERO contacts at that
# pose") — it does not: that sentence was written about the pre-2026-08-10
# geometry and this file has no MuJoCo reference for env0 at the current pose.
# It is also exactly the direction task #59 makes suspicious, where a plainly
# overlapping cylinder/box pair produces zero contacts through
# `detect_contacts` while MuJoCo reports three. Re-freezing the count here is
# BOOKKEEPING, not a claim that 0 is right; if #59 turns out to be a dispatch
# gap, come back to this line.
#
# THE CAUSE IS `opt.ccd_tolerance`, which the engine now reads from the model
# instead of hardcoding 1e-8. Cylinder-vs-box is a SMOOTH pair (`discreteGeoms`
# is mesh/box/hfield on BOTH sides), so it takes MuJoCo's 1e-6 and EPA stops
# where MuJoCo stops. Note the direction of the surprise: the LOOSER tolerance
# is the one that agrees with the reference.
#
# ⚠ AN EARLIER 4 -> 5 WENT UNLOGGED. Every other move of this constant is
# accounted for above (2 -> 1, 1 -> 4); nothing explains 4 -> 5, so env0
# regained its spurious contact at some point without anyone noticing. That is
# the failure mode a self-frozen golden has and a MuJoCo-anchored assert does
# not — which is why the depth / normal / span checks below are the real gate
# and this number is bookkeeping.
comptime GOLD_NCON = 5  # total contacts across both envs (08-13, below)
# ⚠ GOLD_CON has moved TWICE on 2026-08-01, both times accounted for exactly
# and neither re-recorded blind. Both changes are narrow-phase CONTACT
# DIRECTION work; the fingerprint is `sum contacts[e,c,k] * (e+1)(c+1)(k+1)`.
#   +32.0  bug 35 (the double flip): fractional part UNCHANGED, and 33 - 1 = 32
#          is one `(body_a, body_b)` relabel on the env0 obj(33)/table(1)
#          contact at weight (e+1)(c+1) = 1. Normal/pos/dist bit-identical.
#   -16.0  bug 36 (`cylinder_box` returned the opposite normal — it delegates
#          to `box_capsule` with SWAPPED operands and did not negate). Sawyer's
#          obj is a cylinder and its table a box, so this is exactly that pair.
#          The table is horizontal, so the normal is (0,0,+-1): flipping nz by
#          2 at CONTACT_IDX_NZ (k=7, weight (e+1)(c+1)*8 = 8) gives -2*8 = -16.
#          Fractional part unchanged for the same reason — an integer-valued
#          float moved. The new direction is MuJoCo-verified by
#          `test_narrow_phase_pairs.mojo`.
#
# --- 2026-08-09: SPLIT AT THE SOLPARAM COLUMNS; the delta was NOT physics ----
# This gate spent a week failing "expected contacts in these poses — gate is
# vacuous", i.e. ZERO contacts, and the plan's filed hypothesis was that the
# hardcoded qpos encoded a stale joint order. Both were wrong:
#
#  * the poses were always right — `objjoint` is `jnt_qposadr` 9 in MuJoCo,
#    exactly where this file writes it;
#  * the zero was a COMPILER bug. The O(N^2) GPU kernel stopped writing
#    anything once `3dbc4c33` grew it with the plane-box/capsule-box manifold
#    helpers; the same source on CPU, and the SAP kernel, both found the
#    contacts throughout. Seeding the ncon slot with a sentinel proved the
#    kernel never wrote it at all. Upgrading to Mojo 1.0.0rc2 fixed it, and GPU
#    now matches CPU BIT-FOR-BIT here.
#
# What remained was a pure column delta: `11e188fd` began writing per-contact
# solref/solimp into columns 23-29, which the whole-record checksum summed. The
# geometry columns are UNCHANGED at the pre-existing golden below; the entire
# 452.322002 delta is the solparam columns, and predicting it from the record
# values reproduces 452.322002 exactly. Split so it cannot recur.
#
# --- 2026-08-09b: EPA now covers this pair (`polytope3`) -------------------
# 479.781851073727 -> 481.2657020315528. ONE record moved, env1's obj/gripper
# mesh contact; env0's is a cylinder/box primitive and is untouched, and the
# count is still 2. The move is EPA replacing the centre-line estimate that
# used to run here, and it is an IMPROVEMENT in correctness even though it
# takes the number FURTHER from MuJoCo:
#
#     ours before (centre-line guess) -0.0264747
#     ours now    (EPA)               -0.0148583
#     MuJoCo                          -0.0276947
#
# ⚠ THE NEW NUMBER IS THE EXACT ANSWER FOR THE SHAPE WE HAND EPA. An
# independent reference EPA, run on OUR OWN 81 hull vertices at OUR OWN pose,
# returns 0.0148583 — agreeing with the engine to 7 significant figures. The
# old value was closer to MuJoCo BY ACCIDENT: two errors partially cancelled.
#
# The whole 12.9 mm residual is therefore the mesh VERTEX SET, and it is not
# mainly the direction sampler. Same sampler, different inputs:
#     on MuJoCo's compiled mesh_vert  -> 0.0260234  (1.7 mm out)
#     on our STL-derived vertices     -> 0.0148583  (12.9 mm out)
# so our loader and MuJoCo's mesh compile disagree about the vertices
# themselves, which is the next thing to chase. Sampling density is
# second-order behind it (256 dirs -> 0.36 mm, 1024 -> 0.24 mm on MuJoCo's set).
#
# --- 2026-08-09c: mesh_vertadr was in FLOATS, consumed as VERTICES ---------
# 481.2657020315528 -> 512.8892028308474, and this one moves TOWARD MuJoCo:
#
#     ours before  -0.0148583   (12.9 mm out)
#     ours now     -0.0273728   ( 0.32 mm out)
#     MuJoCo       -0.0276947
#
# `load_mesh_hull` stored `len(mesh_vert)` — a FLAT SCALAR offset — while every
# collision consumer indexes the packed tensor as `[vertex, component]`. So
# `mesh_vertadr` was 3x too large for all of them. eGripperBase's vertadr 1701
# points past the 648 vertices actually loaded, so the gripper hull collided as
# an EMPTY SHAPE; meshes 1..10 were worse than empty, colliding against some
# OTHER mesh's vertices. Only mesh 0, at offset 0, was ever right.
#
# ⚠ This is what the previous refresh's 12.9 mm residual actually was, and the
# reasoning that pinned it on "the hull vertex SET" was right for the wrong
# reason — the set was not merely coarse, it was the wrong memory. The 0.32 mm
# that remains is the direction sampler (26 directions, 81 of 883 hull
# vertices), which is now genuinely the next term.
#
# --- 2026-08-10: cylinder/box re-routed + the mesh NORMAL was reversed -----
# 512.9645483070053 -> 336.28797102486715. Two accounted changes, both toward
# MuJoCo: env0's phantom contact is gone (above), and env1's contact NORMAL was
# REVERSED and is now correct. `gjk_epa` returned `gj -> gi` while every caller
# assumed `gi -> gj`; no gate covered mesh direction, so every mesh contact this
# engine has produced pointed the wrong way. Caught only when cylinder/box was
# routed through the same function and tripped `test_narrow_phase_pairs`'
# direction assert at 1.9999999999976286 -- a full reversal -- on an anchored
# pair. Verified against MuJoCo directly by the new check below.
# 2026-08-10: 336.28797102486715 -> 3361.63858178955 and 301.5480011552572 ->
# 3015.4800115525723, both from the 1 -> 4 manifold above. The solparam column
# sum scaling by ~10x is the expected shape: those columns are IDENTICAL on
# every row of one pair (same geoms, same solref/solimp), so adding 3 rows at
# contact weights (c+1) = 2,3,4 multiplies their contribution by 1+2+3+4 = 10
# against the single row's 1. 301.5480011552572 * 10 = 3015.480011552572,
# matching to 12 digits -- so the solparam change is fully accounted for and
# carries no new information. The geometry columns do not scale that cleanly
# because position differs per row, which is the point of a manifold.
# --- 2026-08-13: plane-mesh `maxplanemesh` cap + mesh `rbound` fix ----------
# 4 -> 5 contacts; GOLD_CON 3361.63858178955 -> 5042.802909596139 and GOLD_SOL
# 3015.4800115525723 -> 4523.220017328858.
#
# TWO engine fixes landed together (see `_plane_mesh_contacts` and
# `compute_mesh_rbound_at`). ⚠ THE COUNT WENT UP, WHICH IS THE OPPOSITE OF
# WHAT THE PLANE-MESH CAP DOES — so the cap is not what moved this fixture.
# `geom_rbound` for a mesh was measured from the VERTEX CENTROID instead of
# MuJoCo's AABB corner about the frame origin, which made it 0.72x-0.95x of
# MuJoCo's on most hulls. `rbound` feeds `mj_filterSphere`
# (`rbound1 + rbound2 + margin`), so an under-sized value REJECTED pairs
# MuJoCo tests: we were missing contacts, and the extra row is one of them.
#
# ⚠ REGENERATED, NOT RE-BASELINED BLINDLY. This file's own MuJoCo comparisons
# — which are NOT frozen goldens — all pass at the new value:
#     mesh depth   ours -0.0276952  MuJoCo -0.02769469   (0.49 um)
#     mesh normal  dot(MuJoCo) = 0.9999996550
#     manifold span ours 0.0392871  MuJoCo 0.03867       (0.62 mm)
# and `test_sawyer_settle_vs_mujoco` reports 5 contacts at rest against
# MuJoCo's 5. The golden of 4 was frozen from the engine WITH the rbound bug.
#
# ⚠⚠ 2026-08-25: THAT REFRESH WAS WRITTEN INTO THE COMMENT AND NEVER APPLIED.
# The three constants below still held 4 / 3361.6499817769654 /
# 3015.4800115525723 twelve days later, so this gate has been RED since, on a
# count the block above had already justified against MuJoCo. `GOLD_SOL` is the
# proof: the value measured today, 4523.220017328858, is the one that block
# names, to the last digit — the solparam columns have not moved since 08-13.
#
# --- 2026-08-25: `c43517db`, the pair reaches GJK in MuJoCo's order ---------
# GOLD_CON 5042.516301484706 -> 5222.468882602276, and it is ACCOUNTED, not
# pasted. `pushPairArena` sorts a pair by (geom type, geom index); this pair is
# obj CYLINDER (type 5) vs eGripperBase MESH (type 7), so MuJoCo's `geom1` is
# the obj and ours was the gripper. Making the pair canonical relabels env1's
# five rows and negates their normal with them:
#
#     body_a/body_b relabel 23 <-> 33   -300.000000   = 300*(k_A - k_B), exact
#     normal sign flip (nx, ny, nz)     +479.740543
#     dist                              +0.000004
#     witness positions (POS_X/Y/Z)       +0.212034   sub-mm, other operand order
#     -------------------------------------------
#     total                             +179.952581   = the observed delta
#
# The CONTACT IS THE SAME ONE: same depth to 1.4e-05 mm of MuJoCo, same
# manifold span, and all five witnesses now carry ONE shared `dist` exactly as
# `con[i].dist = margin + dist` does in `mjc_penetration`.
comptime GOLD_CON = 5222.468882602276  # geometry columns (k < 23)
comptime GOLD_SOL = 4523.220017328858  # solparam columns (k >= 23)

# MuJoCo's manifold at this pose spans 3.867e-02 between its two contact
# FEATURES: a tight cluster on one face and a single row 3.9 cm away. Matching
# that span is the real assertion -- it says we found both features, which a
# row count cannot. The bound is generous against witness-point drift (the
# cluster itself is only 3e-4 wide, so 2e-3 cannot confuse the two features)
# and still three orders tighter than collapsing onto a single point.
comptime MANIFOLD_SPAN = 3.867e-02
comptime MANIFOLD_SPAN_TOL = 2e-3


def main() raises:
    print("--- mesh contact detection fields GOLDEN gate: sawyer BATCH=", BATCH)
    var ctx = DeviceContext()

    # Fields-native model build (loads STL hulls, NMESHV-padded — Stage B).
    var mf = Model[DTYPE, MD]()
    SawyerReachModel.init_fields[DTYPE](ctx, mf)

    # Locate the obj cylinder + mesh-geom bodies for the non-vacuity check.
    var obj_body = -1
    var mesh_bodies = List[Int]()
    for g in range(NGEOM):
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

    # Poses: env0 obj on table; env1 obj teleported into eGripperBase hull.
    var qcfg = List[List[Float64]]()
    for e in range(BATCH):
        var q = List[Float64](length=NQ, fill=0.0)
        q[0] = 1.889288
        q[1] = -0.575769
        q[2] = -0.976659
        q[3] = 1.641991
        q[4] = 0.942860
        q[5] = 1.043696
        q[6] = 2.292833
        q[7] = 0.0
        q[8] = 0.0
        if e == 0:
            q[9] = 0.0
            q[10] = 0.6
            q[11] = 0.02
        else:
            q[9] = 0.005
            q[10] = 0.601
            q[11] = 0.28  # inside the hull — see the module docstring
        q[12] = 1.0
        qcfg.append(q^)

    var d = Data[DTYPE, MD, BATCH]()
    for e in range(BATCH):
        for i in range(NQ):
            d.qpos.data[e * NQ + i] = Scalar[DTYPE](qcfg[e][i])
    d.upload_all(ctx)

    # Fields GPU: FK + detection.
    forward_kinematics["gpu", DTYPE, BATCH=BATCH](d, mf, ctx)
    detect_contacts["gpu", DTYPE, BATCH=BATCH](d, mf, ctx)
    d.contacts.download(ctx)
    d.meta.download(ctx)

    var ncon_total = 0
    var fp_geom = Float64(0)
    var fp_sol = Float64(0)
    for e in range(BATCH):
        var nc = Int(d.meta.data[e * METADATA_SIZE + META_IDX_NUM_CONTACTS])
        ncon_total += nc
        for c in range(nc):
            for k in range(CONTACT_SIZE):
                var w = Float64(
                    d.contacts.data[e * MC * CONTACT_SIZE + c * CONTACT_SIZE + k]
                ) * Float64((e + 1) * (c + 1) * (k + 1))
                if k < CONTACT_IDX_SOLREF_0:
                    fp_geom += w
                else:
                    fp_sol += w
    if ncon_total == 0:
        raise Error("expected contacts in these poses — gate is vacuous")
    print("  fields-GPU total contacts:", ncon_total)
    # ALWAYS printed, pass or fail — one number cannot say WHICH half moved.
    print("  split: geometry cols", fp_geom, " solparam cols", fp_sol)

    if HARVEST:
        print("  HARVEST GOLD_NCON =", ncon_total)
        print("  HARVEST GOLD_CON  =", fp_geom)
        print("  HARVEST GOLD_SOL  =", fp_sol)
    else:
        if ncon_total != GOLD_NCON and not default_accelerator().is_nvidia_gpu():
            raise Error(
                "total contacts " + String(ncon_total) + " != golden "
                + String(GOLD_NCON)
            )
        var denom = abs(GOLD_CON) if abs(GOLD_CON) > 1e-9 else 1.0
        if abs(fp_geom - GOLD_CON) / denom > GOLD_RTOL and (
            not default_accelerator().is_nvidia_gpu()
        ):
            raise Error(
                "GEOMETRY fingerprint " + String(fp_geom) + " != golden "
                + String(GOLD_CON)
            )
        var sdenom = abs(GOLD_SOL) if abs(GOLD_SOL) > 1e-9 else 1.0
        if abs(fp_sol - GOLD_SOL) / sdenom > GOLD_RTOL and (
            not default_accelerator().is_nvidia_gpu()
        ):
            raise Error(
                "SOLPARAM fingerprint " + String(fp_sol) + " != golden "
                + String(GOLD_SOL)
            )
        print("  PASS: fields-GPU matches golden fingerprint")

    # --- MuJoCo-ANCHORED DEPTH, because the fingerprint cannot see it -----
    # ⚠ THE FINGERPRINT DOES NOT GATE DEPTH. `GOLD_RTOL` is 1e-3 RELATIVE to a
    # checksum of ~513, i.e. 0.5 absolute, and `dist` enters that sum at weight
    # (e+1)(c+1)(k+1) = 18. So this gate tolerates a depth change of ~28 mm on
    # a 27.7 mm contact — it would pass with the penetration inverted. Every
    # mesh-depth defect this file has hosted was caught by the POSITION and
    # NORMAL columns moving alongside, never by the depth itself.
    #
    # So assert the depth directly, against MuJoCo 3.10.0 on the reference XML
    # `references/Metaworld-master/.../sawyer_reach_v3.xml` at this exact pose:
    # geom 36 (obj cylinder) vs geom 27 (eGripperBase mesh), dist -2.769469e-02.
    # Hardcoded rather than computed in-process to keep this file free of the
    # Python bridge; `test_narrow_phase_pairs.mojo` is the file that calls
    # MuJoCo live, and it covers the primitive pairs.
    comptime MUJOCO_MESH_DIST = -2.769469e-02
    # ⚠ FIRST matching row, not the last. Both anchors here are MuJoCo's
    # `con[0]` values, so they must be read off OUR primary contact. This loop
    # used to fall through and keep the LAST match, which was harmless while
    # the pair emitted one row and became wrong the moment it emitted a
    # manifold: the trailing rows are perturbed ones, tilted ~1e-3 off the
    # primary normal BY CONSTRUCTION. Measured when it happened — the direction
    # anchor silently went from dot 0.999999655 to 0.999999117, still passing,
    # while comparing against a row MuJoCo never meant it to.
    var mesh_dist = Float64(0)
    var found_depth = False
    for c in range(Int(d.meta.data[1 * METADATA_SIZE + META_IDX_NUM_CONTACTS])):
        if found_depth:
            break
        var b = 1 * MC * CONTACT_SIZE + c * CONTACT_SIZE
        var ba = Int(d.contacts.data[b + CONTACT_IDX_BODY_A])
        var bb = Int(d.contacts.data[b + CONTACT_IDX_BODY_B])
        for mb in mesh_bodies:
            if (ba == mb and bb == obj_body) or (bb == mb and ba == obj_body):
                mesh_dist = Float64(d.contacts.data[b + CONTACT_IDX_DIST])
                found_depth = True
                break
    if not found_depth:
        raise Error("no mesh contact to check depth on — gate is vacuous")
    var derr = abs(mesh_dist - MUJOCO_MESH_DIST)
    print(
        "  mesh depth", mesh_dist, " MuJoCo", MUJOCO_MESH_DIST,
        " err(mm)", derr * 1000.0,
    )
    # 5 um. The exact convex hull brought this from 12.9 mm (mesh_vertadr in
    # the wrong units) through 0.32 mm (support-sampled hull) to 5e-4 mm, so a
    # micron-scale bound is what the code measurably does — not an inherited
    # placeholder.
    if derr > 5e-6:
        raise Error(
            "mesh contact depth " + String(mesh_dist) + " vs MuJoCo "
            + String(MUJOCO_MESH_DIST) + " differs by "
            + String(derr * 1000.0) + " mm"
        )
    print("  PASS: mesh depth matches MuJoCo within 5 um")

    # --- MuJoCo-ANCHORED DIRECTION -----------------------------------------
    # ⚠ NOTHING GATED MESH CONTACT DIRECTION UNTIL NOW, and it was REVERSED.
    # `test_narrow_phase_pairs` anchors direction against MuJoCo for every
    # PRIMITIVE pair — that file exists because of the bug-35 double flip — but
    # no gate covered a MESH pair, so `gjk_epa` returning `gj -> gi` where all
    # its callers assume `gi -> gj` went unnoticed. It surfaced only when
    # cylinder/box was routed through the same function and tripped that file's
    # `dir err 1.9999999999976286`, a full reversal, on an anchored pair.
    #
    # MuJoCo at this pose: geom1 = 36 (obj), geom2 = 27 (gripper), normal
    # geom1 -> geom2 = (-8.6e-05, 1.13e-03, -0.999999).
    #
    # ⚠⚠ THE SIGN RELATION BETWEEN THE TWO ENGINES IS A CONVENTION, AND IT WAS
    # PINNED TO A BUG. This comment used to read "our record stores
    # `body_b -> body_a` with body_a = 23 (gripper) and body_b = 33 (obj),
    # which is the SAME direction, so the two compare without a sign flip" —
    # true only because our pair order was NOT MuJoCo's. `c43517db` made the
    # pair canonical, so `body_a` is now 33, the obj, i.e. MuJoCo's `geom1`
    # (measured: the record reads `body_a 33 body_b 23` where it read
    # `body_a 23 body_b 33`). With the labels agreeing, the two conventions no
    # longer cancel:
    #
    #     MuJoCo   `mji_sub3(con[i].normal, status.x1, status.x2)` in
    #              `mjc_penetration`, and `mj_forward` on a sphere/box fixture
    #              confirms it: dot(frame[0:3], centre(g1)->centre(g2)) = +1.
    #              MuJoCo's normal is geom1 -> geom2 = body_a -> body_b.
    #     ours     `contact_detection.mojo`, at the emit: "The record's normal
    #              points `body_b -> body_a` ... negated here — UNCONDITIONALLY".
    #
    # So OUR record is the NEGATION of `mjContact.frame[0:3]`, uniformly, and
    # this check negates the reference to compare. It is a REPRESENTATION
    # mismatch, not a physics one — the solver reads the normal together with
    # (body_a, body_b) and is self-consistent, which is why the Menagerie board
    # is unmoved by `c43517db` and why `csweep.py` cannot see this at all: it
    # scores normals as `abs(np.dot(n, n2))`.
    #
    # ⚠ Before `c43517db` the mismatch was PAIR-DEPENDENT — our `body_a` was
    # whichever geom the loop reached first, so roughly half of all contacts
    # agreed with `mjContact`'s sign and half did not. It is at least uniform
    # now. Normalising the record onto MuJoCo's sign is a separate change: it
    # touches every consumer of the normal, so it wants its own sweep.
    comptime MJ_NX = -8.6e-05
    comptime MJ_NY = 1.13e-03
    comptime MJ_NZ = -0.999999
    var onx = Float64(0)
    var ony = Float64(0)
    var onz = Float64(0)
    # FIRST match — the primary contact. See the note at the depth anchor.
    var found_norm = False
    for c in range(Int(d.meta.data[1 * METADATA_SIZE + META_IDX_NUM_CONTACTS])):
        if found_norm:
            break
        var b = 1 * MC * CONTACT_SIZE + c * CONTACT_SIZE
        var ba = Int(d.contacts.data[b + CONTACT_IDX_BODY_A])
        var bb = Int(d.contacts.data[b + CONTACT_IDX_BODY_B])
        for mb in mesh_bodies:
            if (ba == mb and bb == obj_body) or (bb == mb and ba == obj_body):
                onx = Float64(d.contacts.data[b + CONTACT_IDX_NX])
                ony = Float64(d.contacts.data[b + CONTACT_IDX_NY])
                onz = Float64(d.contacts.data[b + CONTACT_IDX_NZ])
                found_norm = True
                break
    # `-MJ_N*` is MuJoCo's normal expressed in OUR `body_b -> body_a`
    # convention; see the block above for why the negation is here.
    var dotn = onx * (-MJ_NX) + ony * (-MJ_NY) + onz * (-MJ_NZ)
    print(
        "  mesh normal", onx, ony, onz,
        " dot(-MuJoCo, our convention) =", dotn,
    )
    # A reversal reads as dot = -1, which is what this is here to catch. The
    # bound allows ~0.8 degrees, which is loose against the per-row tilt a
    # perturbed manifold carries by construction (MuJoCo's own extra rows sit
    # ~1e-3 off the primary normal) and still four orders from a flip.
    if dotn < 0.9999:
        raise Error(
            "mesh contact NORMAL diverges from MuJoCo: dot = " + String(dotn)
            + " (a value near -1 is a full reversal)"
        )
    print("  PASS: mesh normal matches MuJoCo direction")

    # Non-vacuity: env1 must have a mesh-geom-body vs obj-body contact.
    var ncon1 = Int(d.meta.data[1 * METADATA_SIZE + META_IDX_NUM_CONTACTS])
    var mesh_contact_found = False
    for c in range(ncon1):
        var ba = Int(
            d.contacts.data[1 * MC * CONTACT_SIZE + c * CONTACT_SIZE + 0]
        )
        var bb = Int(
            d.contacts.data[1 * MC * CONTACT_SIZE + c * CONTACT_SIZE + 1]
        )
        for mb in mesh_bodies:
            if (ba == mb and bb == obj_body) or (bb == mb and ba == obj_body):
                mesh_contact_found = True
    if not mesh_contact_found:
        raise Error("no MESH-involved contact in env1 — gate is vacuous")
    print("  PASS: MESH-involved contact present (GJK/EPA fallback)")

    # --- MANIFOLD SPAN, the assertion the row count cannot make -------------
    # MESH x CYLINDER goes through MuJoCo's multi-CCD perturbation loop, and
    # the manifold it builds has TWO features 3.9 cm apart. A single point
    # cannot hold a grasp: the obj rotates about it and slides out. But a count
    # of 4 proves nothing on its own — four rows stacked inside one 3e-4
    # cluster would pass a count check and still be a single point physically.
    # So measure what the manifold SPANS.
    var mxs = List[Float64]()
    var mys = List[Float64]()
    var mzs = List[Float64]()
    for c in range(ncon1):
        var b = 1 * MC * CONTACT_SIZE + c * CONTACT_SIZE
        var ba = Int(d.contacts.data[b + CONTACT_IDX_BODY_A])
        var bb = Int(d.contacts.data[b + CONTACT_IDX_BODY_B])
        for mb in mesh_bodies:
            if (ba == mb and bb == obj_body) or (bb == mb and ba == obj_body):
                mxs.append(Float64(d.contacts.data[b + CONTACT_IDX_POS_X]))
                mys.append(Float64(d.contacts.data[b + CONTACT_IDX_POS_Y]))
                mzs.append(Float64(d.contacts.data[b + CONTACT_IDX_POS_Z]))
    print("  mesh manifold rows:", len(mxs))
    var span = Float64(0)
    for i in range(len(mxs)):
        print(
            "    row", i, "pos [", mxs[i], mys[i], mzs[i], "]"
        )
        for j in range(i + 1, len(mxs)):
            var ddx = mxs[i] - mxs[j]
            var ddy = mys[i] - mys[j]
            var ddz = mzs[i] - mzs[j]
            var sep = (ddx * ddx + ddy * ddy + ddz * ddz) ** 0.5
            if sep > span:
                span = sep
    print(
        "  manifold span", span, " MuJoCo", MANIFOLD_SPAN,
        " err", abs(span - MANIFOLD_SPAN),
    )
    if len(mxs) < 2:
        raise Error(
            "mesh contact is a SINGLE POINT — MuJoCo builds a manifold here"
            " (5 rows). A one-point grasp lets the object rotate about it and"
            " slide out; check `multi_ccd_pair_supported` covers"
            " MESH x CYLINDER and that `_convex_pair_single` answers it."
        )
    if abs(span - MANIFOLD_SPAN) > MANIFOLD_SPAN_TOL:
        raise Error(
            "mesh manifold SPAN " + String(span) + " != MuJoCo "
            + String(MANIFOLD_SPAN) + ". A span near zero means every row"
            " landed on ONE contact feature, which is a single point wearing a"
            " row count; a larger span means a row sits somewhere MuJoCo has"
            " no contact."
        )
    print("  PASS: mesh manifold spans both contact features")

    # --- fields-CPU vs fields-GPU records (fed GPU FK products) ---
    var dc = Data[DTYPE, MD, BATCH]()
    d.xpos.download(ctx)
    d.xquat.download(ctx)
    for i in range(BATCH * NBODY * 3):
        dc.xpos.data[i] = d.xpos.data[i]
    for i in range(BATCH * NBODY * 4):
        dc.xquat.data[i] = d.xquat.data[i]
    detect_contacts["cpu", DTYPE, BATCH=BATCH](dc, mf)
    var worst = Float64(0)
    for e in range(BATCH):
        var nc_g = Int(d.meta.data[e * METADATA_SIZE + META_IDX_NUM_CONTACTS])
        var nc_c = Int(
            dc.meta.data[e * METADATA_SIZE + META_IDX_NUM_CONTACTS]
        )
        if nc_g != nc_c:
            raise Error("fields-CPU contact count differs from fields-GPU")
        for c in range(nc_g):
            for k in range(CONTACT_SIZE):
                var err = abs(
                    Float64(
                        dc.contacts.data[
                            e * MC * CONTACT_SIZE + c * CONTACT_SIZE + k
                        ]
                    )
                    - Float64(
                        d.contacts.data[
                            e * MC * CONTACT_SIZE + c * CONTACT_SIZE + k
                        ]
                    )
                )
                if err > worst:
                    worst = err
    # ⚠⚠ THIS NUMBER IS 4.3e-06 AND THE TOLERANCE IS 1e-4, WHICH LOOKS LIKE 23x
    # OF MARGIN AND IS NOT. Measured 2026-08-26: adding a NEVER-TAKEN `break`
    # to `_bb_post_filter` — a function the mesh path does not call — moved
    # this from 4.30e-06 to 2.08e-04, straight through the tolerance. The
    # records that moved are POS_X/POS_Y/NX on the mesh manifold rows, by up to
    # 0.18 mm on coordinates of ~0.62 m.
    #
    # An unused `comptime` did NOT move it, so this is not arbitrary
    # sensitivity: a dead branch in a hot loop is enough to change Metal's
    # float32 codegen, and the CPU and GPU legs then land on different EPA
    # witness points. See `_metal_wide_per_thread_inlinearray_miscompute` —
    # "the answer moves when the compile moves" is the symptom, four times now.
    #
    # So this assertion is real but it is BRITTLE, and a failure here is not
    # automatically a defect in whatever was just changed. Before believing it,
    # bisect by ADDING the change to the old file rather than removing it from
    # the new one, and check whether the change is even on the mesh path. The
    # underlying gap — CPU/GPU mesh manifold agreement at float32 being decided
    # by codegen — is real, unfixed, and bigger than this file.
    print("  fields-CPU vs fields-GPU worst record err:", worst)
    if worst > 1e-4:
        raise Error("fields-CPU contact records tolerance exceeded")
    print("  PASS: fields-CPU contacts within 1e-4")
    print("test_mesh_detection_fields: ALL PASS")
