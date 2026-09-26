"""EPA's polytope, in a tensor row instead of on the per-thread stack.

⚠ THIS IS THE REFERENCE'S OWN STORAGE CLASS, not a Mojo workaround.
`mjc_penetration` (`engine_collision_convex.c:98`) hands EPA a
`config->buffer` that is either the thread-local `ccd_buffer` or
`mj_stackAllocByte(d, mjc_ccdSize(...))` — i.e. a slab carved out of
**mjData's arena**, sized by `mjc_ccdSize` (`engine_collision_gjk.c:2283`)
and re-used by every collision in the step. MuJoCo has never put the
polytope on the C stack. Ours did, and that is what pinned heightfields to
the CPU: the prism is a sixth shape for GJK, so `HFIELD_ENABLED` compiles a
SECOND instantiation of `gjk_epa_witness` into the collision kernel, and
two ~7 KB frames overflow the Metal per-thread stack ("Compute function
exceeds available stack space" — the ceiling `MC_MAX_POLYVERT` sits on too).

⚠ ONE ROW PER ENV, AND THAT IS WHAT MAKES IT RACE-FREE. The collision
kernels run one THREAD PER ENV (`_detect_contacts_fields_kernel`,
`_sap_narrow_kernel`), so `ws[env, ...]` is private to a thread by exactly
the same argument that makes `d.contacts[env, ...]` private — MuJoCo's
`ccd_buffer` is `mjTHREADLOCAL` for the same reason. A single shared row
WOULD be a race; the note in `_support` about a scratch tensor being racy
was written against that shape and does not apply here.

The prism itself stays on the stack: `mjc_ConvexHField` rebuilds those six
vertices for every grid cell it walks, so they are per-CALL data, eighteen
floats wide, not model or per-env data.

⚠ THE CAPS ARE UNCHANGED BY THE MOVE. They still bound the polytope and
overflow is still REPORTED rather than truncated; only the storage moved.
Raising them is a separate, measurable change — `EPA_ITER_HARD_CAP` and the
`nev`/`nef` guards are what a model's `ccd_iterations` is actually clamped
by, and MuJoCo's own allocation (`5 + iterations` verts, `6*iterations`
faces) is far larger.
"""

from std.sys import has_nvidia_gpu_accelerator
from layout import Layout


# ---- EPA's polytope caps ---------------------------------------------------
# ⚠⚠ THESE ARE MuJoCo'S OWN ALLOCATION FORMULA, NOT NUMBERS WE CHOSE.
# `mjc_ccd` carves the polytope out of `config->buffer` as
#
#     pt.verts = Vertex[5 + N];   pt.faces = Face[6 * N];   pt.map = Face*[6 * N]
#     pt.maxfaces = 6 * N
#
# with `N = m->opt.ccd_iterations` (`engine_collision_gjk.c`, `mjc_ccdSize`).
# `5` is the largest seed (`polytope2` and `polytope3` both leave five
# vertices) and EPA adds exactly one vertex per iteration, so `5 + N` is tight
# rather than generous. Deriving both caps from one iteration bound keeps that
# relationship visible: raising the bound raises both, and the ROW GROWS, which
# is the cost that has to be paid deliberately.
#
# ⚠ FACES ARE NEVER REUSED. `attachFace` takes `pt->faces[pt->nfaces++]` and
# `deleteFace` only marks `index = -2` — the slot stays allocated, because
# `Face::adj` refers to faces by INDEX and compacting the array would rewrite
# every adjacency. So the cap has to cover every face ever created, not the
# live set, which is why it is `6 * N` and not something near the live peak.
comptime EPA_ITER_CAP: Int = 64
comptime EPA_V_CAP: Int = 5 + EPA_ITER_CAP
comptime EPA_F_CAP: Int = 6 * EPA_ITER_CAP

# ---- the multi-contact caps ------------------------------------------------
# MuJoCo's `npolygonmax` / `nmeshdegmax`, which are RUNTIME model fields there
# — sized per model, so the reference has no cap at all. Ours are comptime
# because the offsets above have to be.
#
# `MC_MAX_POLYVERT` is the largest number of vertices in one face polygon;
# `MC_MAX_DEG` the most polygons meeting at one vertex. They live HERE, beside
# the row they size, rather than in `native_multicontact` — a constant and the
# buffer it dimensions drifting apart is exactly how the old "checked at model
# build" comment came to be false.
# ⚠ BOTH ARE MEASURED WORSTS, not round numbers, and they are measured over
# MENAGERIE **AND** THIS REPO'S OWN MODELS. robotiq_2f85's base_mount carries a
# 144-vertex face and 21 scenes have one wider than the 56 the width used to
# be; the degree used to be 48, which covers Menagerie (flexiv_rizon4's 47) and
# does NOT cover `envs/robots/assets/so_arm101.xml`, whose STS3215 servo hulls
# (`sts3215_03a_v1`, `sts3215_03a_no_horn_v1`) each carry a vertex with **50**
# incident polygons and its mirror with 49.
#
# ⚠ SO THE CENSUS HAS TO INCLUDE THE MODELS WE SHIP, NOT ONLY THE REFERENCE
# TREE. The old 48 was Menagerie's worst plus one, and it read as a bound on
# "every model" because nothing had ever measured the other half of the corpus.
# Per-scene, collision meshes only: 96 Menagerie scenes give 144 / 47, and all
# 57 in-repo models give 82 / 50 — worst overall 144 / 50.
#
# ⚠⚠ A WHOLE-DIRECTORY CENSUS OVER MESH **FILES** ANSWERS A DIFFERENT QUESTION.
# Sweeping all 2 149 `.stl`/`.obj` in the tree gives 395 / 187, because most of
# them are visual-only geometry that never reaches a collision routine. The
# number these have to cover is the per-SCENE one.
#
# ⚠ WHAT THE RAISE BUYS IS A BOUND, NOT A MEASURED CONTACT. Pressed on that
# exact corner from 128 distinct orientations, shrinking the cap to **8** —
# dropping 42 of the 50 candidates, not 2 — moves NOTHING: every contact still
# matches MuJoCo to 3.6e-15 in position and 8.5e-07 degrees in normal. The
# reason is structural and worth knowing before anyone tries to gate this: a
# vertex has high degree because it is finely tessellated, so its incident
# normals arrive in near-duplicate clusters (indices 46 and 48 here are 0.1
# degrees apart), `_aligned_faces` takes the FIRST match within `MC_FACE_TOL`
# (0.092 degrees), and an early member of the cluster wins before a late one is
# ever reached. Running a shipped model with a knowingly truncated candidate
# list is still not a thing to leave standing.
comptime MC_MAX_POLYVERT: Int = 144
comptime MC_MAX_DEG: Int = 50
comptime MC_CLIP_CAP: Int = 2 * MC_MAX_POLYVERT

# ---- row layout ------------------------------------------------------------
# The `Polytope` struct of `engine_collision_gjk.c`, one region per field.
# Everything is stored as `DTYPE` — including the indices, exactly as the rest
# of the fields path stores them (`mesh_polyvert`, `mesh_edges`) — and every
# index is bounded by `EPA_F_CAP`, so all of them are exact in float32 too.
#
# `ev` — polytope vertices, `Vertex`. 11 floats: the Minkowski point (0..2),
# the two witness points (3..5, 6..8) and the two SUPPORT INDICES (9, 10).
# EPA carries the witnesses through expansion, and the indices are MuJoCo's
# `index1`/`index2` — the box corner code or the mesh hull vertex, which is
# what the discrete repeated-support-point break compares.
comptime CCD_WS_EV: Int = 0
comptime EPA_V_STRIDE: Int = 11
# `ef` — `Face::verts`, three vertex indices per face.
comptime CCD_WS_EF: Int = CCD_WS_EV + EPA_V_CAP * EPA_V_STRIDE
# `eadj` — `Face::adj`, the face across each edge: [v1,v2], [v2,v3], [v3,v1].
comptime CCD_WS_EADJ: Int = CCD_WS_EF + EPA_F_CAP * 3
# `efv` — `Face::v`, the origin projected onto the face's plane. Doubles as the
# face normal, unnormalised, with |v| the distance to the origin.
comptime CCD_WS_EFV: Int = CCD_WS_EADJ + EPA_F_CAP * 3
# `efd` — `Face::dist2`, the squared norm of `v`.
comptime CCD_WS_EFD: Int = CCD_WS_EFV + EPA_F_CAP * 3
# `efi` — `Face::index`: >= 0 the slot in `map`, -1 not in map, -2 deleted.
comptime CCD_WS_EFI: Int = CCD_WS_EFD + EPA_F_CAP
# `map` — the CANDIDATE face list. A face joins it only when its distance lies
# between the current lower and upper bounds, so this is a strict subset of the
# polytope and NOT the same thing as "every face".
comptime CCD_WS_MAP: Int = CCD_WS_EFI + EPA_F_CAP
# `hor` — the horizon, (face index, edge index) per entry. ⚠ MuJoCo sizes
# `horizon.indices` at 24 and never checks it; ours is `EPA_F_CAP` because the
# horizon cannot exceed the faces it is built from.
comptime CCD_WS_HOR: Int = CCD_WS_MAP + EPA_F_CAP
# `hstk` — the explicit stack for `horizonRec`, (face, edge, state) per frame.
# ⚠ MuJoCo RECURSES. A GPU kernel cannot, so the recursion is unrolled into
# this stack; the traversal ORDER is preserved exactly, because the order the
# horizon edges are added in decides which edge seeds the new face fan.
comptime CCD_WS_HSTK: Int = CCD_WS_HOR + EPA_F_CAP * 2
# `center` — `Polytope::center`, the seed's centroid. `attachFace` orients each
# face's projection away from it.
comptime CCD_WS_CTR: Int = CCD_WS_HSTK + EPA_F_CAP * 3

# `spx` — GJK'S OWN SIMPLEX, four vertices of eleven floats: the Minkowski
# point (0..2), the two witness points (3..5, 6..8) and the two SUPPORT INDICES
# (9, 10) — the SAME layout as an EPA vertex (`ev`), so seeding the polytope
# from the simplex is a straight copy and a seed keeps its hull vertex. It was
# nine floats (no indices) until 2026-09-26; multicontact then had to recover
# every seed's vertex by scanning the whole hull, ~0.75 ms of a 1.5 ms
# collision launch on so101_tower at 1024 lanes.
#
# ⚠⚠ IT IS HERE FOR A DIFFERENT REASON THAN THE POLYTOPE ABOVE, AND THE REASON
# IS A METAL MISCOMPILE, NOT SIZE. Thirty-six floats is nothing; what matters
# is that GJK indexes this array BY A RUNTIME VALUE in eleven places
# (`simplex[i * 9 + c]`, the `lambda`-compaction that overwrites
# `simplex[keep * 9 + c]`, `gjkIntersect`'s permutation, `polytope3`'s
# rotation). A per-thread `Array` indexed by a runtime value is the
# defect recorded in `feedback_metal_wide_per_thread_inlinearray_miscompute` —
# it reads back the WRONG VALUE with no crash, and it has now cost this engine
# three separate hunts. THREE elements was enough the second time; this one is
# thirty-six.
#
# ⚠ MEASURED, NOT ASSUMED. On `test_hfield_vs_mujoco`'s GPU leg the sphere —
# whose `mjc_pointSupport` returns a CONSTANT, so GJK converges before the
# simplex ever grows — was bit-identical to the CPU across all six of its
# contacts, while the box and the capsule, which make GJK iterate, diverged
# from the first vertex count onwards. That split is what named this array.
comptime CCD_WS_SPX: Int = CCD_WS_CTR + 3
comptime SPX_STRIDE: Int = 11
# `spx2` — `gjkIntersect`'s scratch copy. The reference builds the permuted
# tetrahedron in a local `Vertex simplex[4]` and copies it back over the
# caller's; ours cannot alias the same region while it does that.
comptime CCD_WS_SPX2: Int = CCD_WS_SPX + 4 * SPX_STRIDE
comptime EPA_WS_SIZE: Int = CCD_WS_SPX2 + 4 * SPX_STRIDE


# ---- the multi-contact region ----------------------------------------------
# `native_multicontact`'s polygon buffers, for the same reason and by the same
# mechanism. MuJoCo sizes its equivalents from `npolygonmax` / `nmeshdegmax`,
# which are RUNTIME MODEL FIELDS — it has no cap at all.
#
# ⚠ ONLY THE `MC_MAX_POLYVERT`-SIZED ARRAYS MOVE. The `MC_MAX_DEG` ones
# (`n1`/`n2`/`idx1`/`idx2`/`endverts`, ~4.4 KB together at 50) stay on the
# stack: that axis is small and grows slowly — 48 -> 50 is 240 more bytes per
# frame — so it was never the one that needed unlocking. The width axis is: the
# tree's worst face is robotiq_2f85's 144 vertices, and 21 scenes carry a
# polygon wider than 56. ⚠ The degree arrays being on the STACK is why raising
# `MC_MAX_DEG` still has to re-run the Metal canary
# (`tests/physics3d/test_plane_mesh_fields.mojo`) even though `CCD_WS_SIZE`
# does not move: the ceiling it would hit is the per-thread stack, not the row.
#
# ⚠ THE CAP DEGRADES SILENTLY WHEN IT BITES. `_mesh_face` returns 0 past it,
# which is the routine's own "the features do not line up" answer, so the
# caller emits the single EPA point — the reference's own fallback, reached
# for a reason the reference does not have. It is a LOST MANIFOLD.
comptime MC_WS_FACE1: Int = EPA_WS_SIZE
comptime MC_WS_FACE2: Int = MC_WS_FACE1 + MC_MAX_POLYVERT * 3
# The clipped ring. A clip can reach the sum of the two input sizes, hence
# `MC_CLIP_CAP = 2 * MC_MAX_POLYVERT` rather than `MC_MAX_POLYVERT`.
comptime MC_WS_OUT: Int = MC_WS_FACE2 + MC_MAX_POLYVERT * 3
# `_polygon_clip`'s two working rings and its per-edge plane cache.
comptime MC_WS_POLY: Int = MC_WS_OUT + MC_CLIP_CAP * 3
comptime MC_WS_CLIPPED: Int = MC_WS_POLY + MC_CLIP_CAP * 3
comptime MC_WS_PN: Int = MC_WS_CLIPPED + MC_CLIP_CAP * 3
comptime MC_WS_PD: Int = MC_WS_PN + MC_MAX_POLYVERT * 3

# ⚠ ONE TENSOR, TWO REGIONS, AND THE WHOLE ROW IS ALWAYS ALLOCATED. EPA's
# polytope and the multi-contact polygons are live at DIFFERENT times within
# one collision — the manifold routine runs after `gjk_epa_witness` returns —
# so they could have overlapped. They do not, deliberately: an aliasing bug
# between two regions that are "obviously" disjoint in time is invisible in a
# diff and fires only on the pose where the assumption breaks.
#
# MuJoCo 3.12's per-witness-point distance (`status->dist[i]`,
# `witnessOnFace`): one signed plane distance per emitted manifold point,
# aligned with `MC_WS_OUT`. 3.10 gave every point the EPA depth; 3.12 gives
# each clipped vertex its own, and the solver sees a different penetration
# per row. Sized like the OUT ring it annotates.
comptime MC_WS_ODIST: Int = MC_WS_PD + MC_MAX_POLYVERT
# EPA's 964 floats plus 28 * MC_MAX_POLYVERT, plus the 2 * MC_MAX_POLYVERT
# per-point distances.
comptime HW_WS_OFF: Int = MC_WS_ODIST + MC_CLIP_CAP

# ── The cross-step warm start of the mesh hill climb (PERFORMANCE.md §13.48,
# 2026-09-08). On the park scene every mesh support call was COLD: a
# candidate's GJK proves the pair apart on its first support point and
# exits, so the run-scoped warm vertex (`warm1`/`warm2`, MuJoCo's
# `meshindex`) never fired and each walk started from vertex 0, 13.8
# neighbourhood scans from the answer — each scan a dependent chain of
# global loads on one GPU thread. The vertex the SAME candidate landed on in
# the previous step is the answer in 4,491 of 4,491 replays (1.00 scans).
# So the row keeps, after the polygon regions, `HILL_WARM_SLOTS` pairs of
# vertex indices keyed by a hash of the geom pair (`_sap_pair_narrow`):
# `gjk_epa_witness` seeds both walks from the slot and writes the landings
# back. The row is per env (per CCD lane in the block kernel), so the state
# persists across steps exactly where the pair recurs.
#
# ⚠ ANY VALUE IN A SLOT IS SAFE. A stale, crossed or garbage index costs
# steps, never a point: `hillclimb_support_index` clamps an out-of-range
# seed to 0, and on a convex hull the walk converges from any vertex. A hash
# collision between two pairs hands one the other's vertex — steps again.
# What a seed CAN change is a tie: on a face perpendicular to the query
# direction two vertices share the maximum and the walk stops at whichever
# it reaches first, so the support POINT can move along that face. MuJoCo
# 3.12 accepted the same (its `mesh_extrema` seed); the goldens are the gate.
comptime HILL_WARM_ACROSS_STEPS: Bool = True
comptime HILL_WARM_SLOTS: Int = 128
comptime CCD_WS_SIZE: Int = HW_WS_OFF + 2 * HILL_WARM_SLOTS

# ── The block-per-env collision kernel (`broadphase_sap.mojo`,
# `_detect_contacts_sap_block_kernel`), 2026-09-07 ────────────────────────
#
# One block per env, `COLL_TPB` threads over the candidate pairs. A pair that
# reaches GJK/EPA or the multicontact clipper needs a CCD workspace ROW, and a
# row is `CCD_WS_SIZE` = 11,394 scalars (45 KB in float32). EVERY THREAD OF
# THE BLOCK HAS ONE (2026-09-11): `Data.ccd_ws` is
# `[BATCH * COLL_CCD_LANES, CCD_WS_SIZE]` with `COLL_CCD_LANES == COLL_TPB`,
# 1.49 GB at 1024 envs x 32 lanes — the price of running 32 GJK candidates
# of one env at once instead of four. (Until 2026-09-11 four lanes carried
# the CCD candidates and 28 threads the cheap ones, 187 MB; on a sprawled
# G1 nearly every candidate is a mesh pair, so that put ~40 GJKs through
# four lanes ten rounds deep, PERFORMANCE.md §13.51.) The serial kernels keep
# using row `env` (< BATCH, so never another env's lane).
#
# Contacts are emitted into a per-env STAGING region of `COLL_STAGE_SLOTS`
# records, `COLL_STAGE_MAXC` per candidate, at offsets thread 0 assigns in
# the serial emission order; thread 0 then compacts them in that order into
# `Data.contacts`, so the array is the serial kernel's bit for bit. A
# candidate whose routine fills its whole window (8 is `mjMAXCONPAIR`'s
# box-box ceiling; a mesh manifold can exceed it), or a candidate list past
# `COLL_NCAND_CAP`, sends the env to the serial per-env function on thread 0
# — slow and exact, never wrong.
# 32, not 64: E2 on Metal (block ledger §6) put the thread count at a 1.4x
# term between 8 and 64 threads, and a block's register footprint scales
# with it, which is what bounds blocks per SM on CUDA (255 regs x 32 = 8K of
# 64K).
comptime COLL_TPB: Int = 32
# ⚠ ON SINCE 2026-09-11, BY A MEASUREMENT ON THE TARGET GPU. The first
# block kernel (2026-09-07) was measured NEUTRAL on the RTX 5090 at the k=0
# park scene (269.7 vs 269.0 µs; 381 vs 430 at k=13), with the bisect
# putting 206 of its 270 µs in FOUR GJK candidates on four lanes of one
# warp next to 28 lanes of cheap candidates — lanes are parallel only on
# the SAME instructions, and that layout diverged on the code itself.
# PERFORMANCE.md §13.51 then found the serial kernel at 47% of a G1 training
# run (1024 sprawled humanoids, 131 ms a launch, 16 blocks on 170 SMs), and
# §13.52 rebuilt this kernel for that point: candidates run in KIND order
# (a warp's lanes on the same routine), every thread with its own CCD row,
# the plane candidates gated before they are listed, no per-thread pose
# copies. RTX 5090, 1024 sprawled G1 lanes: 107.7 -> 10.4 ms a launch
# (10.4x), bit-identical to the serial kernel on every snapshot with the
# warm start off, no env sent to the serial fallback; Apple M1 Pro 298 ->
# 142 (2.10x); every collision gate green both ways.
# `benchmarks/physics3d_gpu/bench_g1_collision.mojo` is the A/B, its CPU
# column with `diag_lanes` the correctness witness, and its `csum` line the
# bit-identity gate — STATELESS (`HILL_WARM_ACROSS_STEPS=False` both sides)
# and on NVIDIA only: on Apple the SERIAL kernel is the side off the CPU
# (§13.52). False = the one-thread-per-env kernel, kept as the reference.
comptime COLL_BLOCK_KERNEL: Bool = True
comptime COLL_CCD_LANES: Int = COLL_TPB if COLL_BLOCK_KERNEL else 1
# ⚠ 256, NOT 128: a sprawled G1 lists ~60-120 sweep candidates plus the
# plane's survivors, and an env past the cap goes to the SERIAL fallback —
# the whole win evaporates one env at a time. The staging slab is
# `BATCH * 256 * 8 * CONTACT_SIZE` floats (251 MB at 1024 lanes).
comptime COLL_NCAND_CAP: Int = 256
comptime COLL_STAGE_MAXC: Int = 8
comptime COLL_STAGE_SLOTS: Int = (
    COLL_NCAND_CAP * COLL_STAGE_MAXC if COLL_BLOCK_KERNEL else 1
)
# ⚠ A TIMING INSTRUMENT, NOT A MODE. True skips the serial fallback launch
# for the envs the block kernel marked (`ncon = -1`), so a benchmark can
# COUNT them (they keep the mark) and time the block kernel alone. The
# contact set is WRONG for those envs while this is on.
comptime COLL_NO_FALLBACK: Bool = False

# ⚠ A MEASUREMENT KNOB, THE CONTACT SET UNCHANGED (PERFORMANCE.md §13.55).
# True makes thread 0 of the block kernel write a per-env REPORT of its
# candidate list into the TAIL of that env's `coll_stage` row, after phase 3
# has compacted the staging windows into `contacts` — the slab is dead by
# then, and nothing reads it until the next launch's phase 2 writes it
# again, so `contacts` and `ncon` are bit-identical with the knob on. What
# it answers, per env: how many candidates the narrow phase ran (hence how
# many ROUNDS of `COLL_TPB`), which KINDS they were in round order, how
# long thread 0's sweep was (AABB tests, insertion-sort shifts), and why an
# env went to the serial fallback. Layout, from
# `broadphase_sap.COLL_REPORT_BASE`:
#
#     [0] ncand      [1] overflow (list past COLL_NCAND_CAP)
#     [2] fallback   (overflow OR a staging window filled; ncon = -1)
#     [3] ncon out   [4] sap_n (geoms in the sweep)
#     [5] sweep AABB tests   [6] insertion-sort shifts   [7] COLL_TPB
#     [8] sweep AABB passes, UNCAPPED (the list stops at COLL_NCAND_CAP; this
#         does not)   [9] of those, the pairs that survive the narrow phase's
#         first rejects — the predefined-pair lookup, `pair_body_filtered`
#         and the contype/conaffinity mask, the SAME functions it calls
#     [10] plane candidates listed
#     [11 ..] the candidates' kind keys in phase-2 ORDER, -1 past ncand
#
# ⚠ NOT FOR TIMING: the report is serial work on thread 0 at the end of
# every launch, and the two counters sit inside the sweep's inner loop.
# Read times from a build with this off. `benchmarks/physics3d_gpu/
# bench_libero_collision.mojo` decodes it.
comptime COLL_CAND_REPORT: Bool = False

# ⚠⚠ ON FOR NVIDIA, OFF FOR METAL — PRODUCTION SINCE 2026-09-15 (§13.55).
# True makes the block kernel's thread-0 sweep drop a pair BEFORE listing it
# when the narrow phase would reject it at once — the predefined-pair lookup,
# then `_sap_pair_filter_rejects` (body filter + mask), through
# `broadphase_sap._sap_pair_listable`, which decides with the same function
# `_sap_pair_narrow` does. Exact by construction: such a pair emits no
# contact and touches no warm slot, and the survivors keep their emission
# order, so the compacted `contacts` are the same list.
#
# WHY: on LIBERO (240 geoms, fixtures built from many boxes on one body) the
# sweep passed 377-550 pairs per lane against the 256-candidate cap, so EVERY
# lane overflowed and the serial per-env kernel did its collision. Only
# 26-198 survive these rejects. RTX 5090, 256 lanes, recorded demo poses:
# the launch 10.1 / 26.3 / 34.4 ms -> 0.97 / 1.86 / 1.80 (10-19x), CPU check
# identical to the unfiltered build on every window.
#
# WHY NOT METAL — the REASON IS FIXED, the MEASUREMENT IS NOT DONE (§13.56,
# 2026-09-16). It was NVIDIA-only because once the block kernel actually ran
# LIBERO's box/box pairs, Metal returned NO contacts for them: a per-thread
# `Array` in `box_box_manifold` read at a RUNTIME index, so the separating-axis
# test picked an edge axis for a resting box. `collision_primitives.mojo` now
# has no runtime index there and
# `tests/physics3d/test_box_box_sap_gpu_parity.mojo` PASSES on Metal (4 = 4,
# 28 = 28, 4 = 4). So flipping this True on Metal is now a measurement, not a
# risk — it has simply not been measured there. The arm script's `nopre` arm
# sets it False to A/B the old production.
comptime COLL_PREFILTER: Bool = has_nvidia_gpu_accelerator()

# ── The FLAT narrow phase (`broadphase_sap.detect_contacts_sap`), 2026-09-26 ─
#
# ⚠ OFF UNTIL MEASURED ON THE TARGET GPU. The block kernel runs an env's
# candidates on the 32 lanes of ONE warp, and a warp's lanes on different
# long pairs SERIALIZE: on so101_tower (1024 lanes, RTX 5090) the narrow
# phase of an env measured 0.55x the SUM of its candidates' single-thread
# times (correlation 0.978 over 1024 envs), because its ~2.5 penetrating
# box/mesh pairs (43 us each after the hill-climb seed, up to 250) diverge
# from each other. The kernel waits for the slowest env: 540 us of narrow
# phase where the slowest single pair is 250.
#
# True replaces phase 2 with four launches:
#   1. the block kernel with `FLAT_LIST`: poses, AABBs, the sweep, then the
#      candidate list into `Data.coll_flat` — each candidate HOT (its pair's
#      last measured narrow phase >= `COLL_FLAT_HOT_NS`, bucketed by cost) or
#      cold — and return;
#   2. `_sap_flat_prefix_kernel`, one block: the envs' lists into two flat
#      queues, the hot one in DESCENDING cost bucket;
#   3. `_sap_narrow_flat_kernel`, one warp per env slot: warp `w` runs hot
#      tasks `w, w + W, ...` on ONE lane each (round robin over a descending
#      order: a longest-first schedule), then cold tasks 32 to a warp;
#      every task's time is written back as its pair's cost;
#   4. `_sap_flat_output_kernel`: the block kernel's phase 3 on the list.
# Simulated on the measured per-pair times: the narrow phase 540 -> ~250 us
# with a longest-first order (~420 in arbitrary order). Contacts are the
# block kernel's bit for bit: same narrow phase per candidate, same staging
# windows, same compaction. Only WHICH warp and CCD row runs a pair changes.
#
# ⚠ The costs are timed with `perf_counter_ns` (the global timer) on NVIDIA
# only; elsewhere nothing is ever hot and the path degrades to cold warps.
comptime COLL_FLAT_NARROW: Bool = False
comptime COLL_FLAT_HOT_NS: Int = 16384
# hot cost buckets: [1, 2), [2, 4), [4, 8), [8, inf) x COLL_FLAT_HOT_NS
comptime COLL_FLAT_NB: Int = 4
# `Data.coll_flat`: one row per env, then a global block.
#   row: [0] ncand [1] overflow [2] full (a window filled) [4..8) hot count per
#   bucket [8] cold count | a, b, t per candidate | records written per
#   candidate | the env's task list — hot bucket 3, 2, 1, 0, then cold in kind
#   order | the per-pair cost (ns), keyed like the hill climb's warm slots
comptime CF_NCAND: Int = 0
comptime CF_OVERFLOW: Int = 1
comptime CF_FULL: Int = 2
comptime CF_NHOT: Int = 4
comptime CF_NCOLD: Int = 8
comptime CF_CAND: Int = 16
comptime CF_CNT: Int = CF_CAND + 3 * COLL_NCAND_CAP
comptime CF_LIST: Int = CF_CNT + COLL_NCAND_CAP
comptime CF_COST: Int = CF_LIST + COLL_NCAND_CAP
comptime CF_ROW: Int = CF_COST + HILL_WARM_SLOTS
#   global, at `batch * CF_ROW`: [0] hot tasks [1] cold tasks, then the hot
#   and the cold queue, `batch * COLL_NCAND_CAP` each; a task is
#   `env * COLL_NCAND_CAP + candidate`.
comptime CF_G_HDR: Int = 16


def coll_flat_words(batch: Int) -> Int:
    """`Data.coll_flat`'s length for `batch` envs — allocated whether the
    path is on or not (~8 KB an env), so a gate can run both paths on one
    `Data` and compare them."""
    return batch * CF_ROW + CF_G_HDR + 2 * batch * COLL_NCAND_CAP


comptime COLL_REPORT_HDR: Int = 11
comptime COLL_REPORT_WORDS: Int = COLL_REPORT_HDR + COLL_NCAND_CAP

# The single-row spelling, for host callers that collide one pair at a time
# (every gate and probe in `tests/physics3d`). The engine binds
# `[BATCH, CCD_WS_SIZE]` instead and passes the env index as `wrow`.
comptime L_CCD_WS1 = Layout.row_major(1, CCD_WS_SIZE)
