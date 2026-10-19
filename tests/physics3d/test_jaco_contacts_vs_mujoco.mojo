"""Jaco's CONTACT SET against MuJoCo — the prerequisite for collision rejection.

The last piece of the dm_control Phase 7 reset path is
`ToolCenterPointInitializer`'s rejection sampler, which accepts an IK solution
only if `_has_relevant_collisions` says no. That predicate reads the contact
set directly — geom pairs and `contact.dist` — so wiring it on top of an
unmeasured contact set would put an accept/reject decision on numbers nobody
has checked.

⚠ JACO'S COLLISION PATH HAS NEVER RUN. The mesh-inertia gate that landed
earlier in Phase 7 is MODEL-ONLY; it never called collision detection, so the
9 mesh hulls have never been through narrow phase on this model. This file
exists to find out where we actually stand before anything depends on it.

WHAT JACO CONTAINS, measured on the runtime over 400 random in-range poses:
1 plane, 6 cylinders, 14 meshes; 25% of poses collide, mean ncon 6.4, max 48.
The type-pair histogram is plane-mesh 1868, plane-cylinder 503, mesh-mesh 106,
cylinder-mesh 86, cylinder-cylinder 8. So plane-mesh dominates and mesh-mesh
is real but rarer.

⚠ THE GROUND COUNTS. `_has_relevant_collisions` treats any external body
WITHOUT a free joint as relevant, and the arena's plane is exactly that — so
arm-versus-ground contacts decide rejections just as much as self-collisions
do. A gate that only exercised self-collision would miss the dominant case.

COMPARED AT BODY LEVEL, NOT GEOM LEVEL. Our contact record stores
`body_a`/`body_b`, not geom ids (`CONTACT_IDX_BODY_A`). That is enough for the
predicate, which classifies a geom by which entity owns it — a body property —
and it avoids having to establish a geom index mapping on top of everything
else.

THIS TEST FOUND TWO DEFECTS, BOTH NOW FIXED. What it measured first:

    total contacts     ours 3311   MuJoCo 433     (~7.6x over-generation)
    worst single pose  ours 256    MuJoCo 24      (256 IS OUR BUFFER CAP)
    penetrating body-pairs   both 53, ours-only 0, MUJOCO-ONLY 78

1. `_plane_mesh_contacts` emitted ONE CONTACT PER HULL VERTEX below the plane,
   unbounded. `mjc_PlaneConvex` emits AT MOST THREE (`maxplanemesh = 3`): the
   support point in -normal, plus up to two vertices from ITS HULL-EDGE
   NEIGHBOURHOOD passing `addplanemesh`'s spread filter, which rejects
   `dist3(pnt, first) < tolplanemesh * rbound`, `tolplanemesh = 0.3`. Both
   constants are identical in the 3.3.6, 3.6.0 and 3.11.0 trees.
2. `geom_rbound` for a mesh measured from the VERTEX CENTROID, where MuJoCo
   uses the AABB corner radius about the frame origin (`mjCGeom::GetRBound`).
   Ours spanned 0.72x to 1.16x of MuJoCo's across Jaco's nine meshes. Only the
   first fix made this observable: `rbound` scales the spread filter, so it
   decides how many contacts a plane-mesh pair emits.

⚠⚠ THE OVER-GENERATION WAS CAUSING THE MISSING PAIRS, AS SUSPECTED. This was
recorded here as unproven — "not independent evidence of a second bug until
the count is fixed and this is re-measured" — and the re-measurement settled
it: with the cap in place `mujoco-only` went 78 -> 0 with no other change. One
plane-mesh pair had been flooding the buffer before other pairs were reached,
so a COUNT bug presented as MISSING COLLISIONS.

AFTER (same 60 poses): totals ours 436 vs MuJoCo 433, per-pose counts equal on
58/60, body-pairs both 131 with ours-only 0 AND mujoco-only 0.

⚠ THE RESIDUAL IS NOT ZERO, AND IS NOT EXPECTED TO BE. MuJoCo picks its two
extra contacts in qhull's internal facet order, which nothing here reproduces,
and our hull differs from qhull's on one of the nine meshes (199 vertices
against 198). Counts survive that — they depend only on the candidate SET —
but the two poses that still differ are t=43 (+4, plane-mesh dominated: 15 of
MuJoCo's 23 contacts there) and t=20 (-1, only 2 plane-mesh contacts of 23, so
it lives in the cylinder-mesh / mesh-mesh paths instead). Both have IDENTICAL
body-pair sets, which is what the rejection sampler reads.

WHAT IS ASSERTED: body FK (a hard precondition), that we neither invent nor
miss a penetrating body-pair, the ANY-penetration predicate the sampler
reduces to, that no pose comes near the contact-buffer cap, and contact counts
against MuJoCo's — the last three added once the defects above were fixed.

⚠ THE COUNT BOUNDS CARRY HEADROOM RATHER THAN TODAY'S EXACT NUMBERS. Freezing
58/60 and 436 would make this fail on any hull-triangulation change, including
a correct one, while telling nobody what actually moved. The bounds are set to
catch a return of the 7.6x over-generation, not to pin the residual.

Run with:
    pixi run mojo run -I . tests/physics3d/test_jaco_contacts_vs_mujoco.mojo
"""

from std.math import abs
from std.python import Python
from std.testing import assert_true, TestSuite
from std.collections import Array
from max.gpu.host import DeviceContext

from noeira.physics3d.fields import Model, Data, Dims, DimsLike
from noeira.physics3d.parser.full_parser import parse_xml_full
from noeira.physics3d.parser.fields_build import build_model_fields_from_flat
from noeira.physics3d.kinematics.forward_kinematics import forward_kinematics
from noeira.physics3d.collision.contact_detection import detect_contacts
from noeira.physics3d.gpu.constants import (
    CONTACT_SIZE,
    CONTACT_IDX_BODY_A,
    CONTACT_IDX_BODY_B,
    CONTACT_IDX_DIST,
    META_IDX_NUM_CONTACTS,
)

comptime DTYPE = DType.float64

comptime NBODY = 17
comptime NQ = 9
comptime NV = 9
comptime NJOINT = 9
comptime NGEOM = 21
comptime NSITE = 12
comptime NEXCLUDE = 4
comptime NMESH_VERTS = 60000
# MuJoCo peaked at 48 over 400 random poses; 256 leaves room for our own
# multicontact to emit more without a capacity truncation being mistaken for
# a detection difference.
comptime MAXC = 256
comptime MD = Dims[
    nq=NQ,
    nv=NV,
    nbody=NBODY,
    njoint=NJOINT,
    ngeom=NGEOM,
    nsite=NSITE,
    max_contacts=MAXC,
    nequality=0,
    ntendon=0,
    nexclude=NEXCLUDE,
    nmesh_verts=NMESH_VERTS,
    npair=0,
]

comptime N_POSES: Int = 60
comptime FK_TOL: Float64 = 1e-9


def _read(path: String) raises -> String:
    var builtins = Python.import_module("builtins")
    var f = builtins.open(path, "r")
    var txt = String(f.read())
    _ = f.close()
    return txt


def test_jaco_contacts_vs_mujoco() raises:
    print("=== Jaco reach_site_features: contact set vs MuJoCo ===")
    var sys = Python.import_module("sys")
    _ = sys.path.insert(0, "tests/dm_control")
    var warnings = Python.import_module("warnings")
    _ = warnings.filterwarnings("ignore")
    var tempfile = Python.import_module("tempfile")
    var os = Python.import_module("os")
    var mujoco = Python.import_module("mujoco")
    var np = Python.import_module("numpy")
    var refmod = Python.import_module("manipulation_ref")

    var tmp = String(tempfile.mkdtemp(prefix="jaco_con_"))
    var xml_path = String(refmod.bake("reach_site_features", tmp))
    var cwd = String(os.getcwd())
    _ = os.chdir(tmp)
    var mm = mujoco.MjModel.from_xml_path(xml_path)
    var dat = mujoco.MjData(mm)
    var fmd = parse_xml_full(_read(xml_path))

    var ctx = DeviceContext()
    var mf = Model[DTYPE, MD]()
    build_model_fields_from_flat[DTYPE](fmd, mf)
    _ = os.chdir(cwd)
    var d = Data[DTYPE, MD, 1]()

    assert_true(Int(py=mm.ngeom) == NGEOM, "ngeom mismatch")
    assert_true(Int(py=mm.nbody) == NBODY, "nbody mismatch")

    # ── precondition: Jaco's body FK, which nothing has gated before ─────
    for i in range(NQ):
        var qv = 0.11 * Float64(i + 1) - 0.4
        dat.qpos[i] = qv
        d.qpos.data[i] = Scalar[DTYPE](qv)
    mujoco.mj_forward(mm, dat)
    forward_kinematics["cpu"](d, mf)
    var worst_fk = 0.0
    for b in range(NBODY):
        for k in range(3):
            var e = abs(
                Float64(d.xpos.data[b * 3 + k]) - Float64(py=dat.xpos[b][k])
            )
            if e > worst_fk:
                worst_fk = e
    print("  body FK: worst |d(xpos)| over", NBODY, "bodies:", worst_fk)
    assert_true(
        worst_fk <= FK_TOL,
        "Jaco's body forward kinematics disagrees with MuJoCo — every contact"
        " below would be computed at a different pose, so nothing here would"
        " mean what it says",
    )

    # ── sweep ────────────────────────────────────────────────────────────
    var lo = refmod.arm_joint_bounds()[0]
    var hi = refmod.arm_joint_bounds()[1]
    var rng = np.random.default_rng(4)

    var n_contact_poses = 0
    var n_ncon_equal = 0
    var worst_dncon = 0
    var sum_ours = 0
    var sum_mj = 0
    var pair_both = 0
    var pair_ours_only = 0
    var pair_mj_only = 0
    var worst_dist = 0.0
    var n_pred_agree = 0
    var max_our_ncon = 0

    for t in range(N_POSES):
        for i in range(NQ):
            var v: Float64
            if i < 6:
                v = Float64(
                    py=rng.uniform(
                        Python.evaluate("float")(lo[i]),
                        Python.evaluate("float")(hi[i]),
                    )
                )
            else:
                v = Float64(py=rng.uniform(0.15, 1.35))
            dat.qpos[i] = v
            d.qpos.data[i] = Scalar[DTYPE](v)
        mujoco.mj_forward(mm, dat)
        forward_kinematics["cpu"](d, mf)
        detect_contacts["cpu"](d, mf)

        var mj_n = Int(py=dat.ncon)
        var our_n = Int(Float64(d.meta.data[META_IDX_NUM_CONTACTS]))
        sum_ours += our_n
        sum_mj += mj_n
        if mj_n > 0 or our_n > 0:
            n_contact_poses += 1
        if our_n > max_our_ncon:
            max_our_ncon = our_n
        if our_n == mj_n:
            n_ncon_equal += 1
        var dn = our_n - mj_n
        if dn < 0:
            dn = -dn
        if dn > worst_dncon:
            worst_dncon = dn

        # Body-pair sets, unordered, PENETRATING contacts only (dist <= 0) —
        # that is the subset `_has_relevant_collisions` acts on.
        var ours_a = List[Int]()
        var ours_b = List[Int]()
        for c in range(our_n):
            var o = c * CONTACT_SIZE
            if Float64(d.contacts.data[o + CONTACT_IDX_DIST]) > 0.0:
                continue
            var ba = Int(Float64(d.contacts.data[o + CONTACT_IDX_BODY_A]))
            var bb = Int(Float64(d.contacts.data[o + CONTACT_IDX_BODY_B]))
            var x = ba if ba < bb else bb
            var y = bb if ba < bb else ba
            var seen = False
            for k in range(len(ours_a)):
                if ours_a[k] == x and ours_b[k] == y:
                    seen = True
            if not seen:
                ours_a.append(x)
                ours_b.append(y)

        var mj_a = List[Int]()
        var mj_b = List[Int]()
        for c in range(mj_n):
            if Float64(py=dat.contact[c].dist) > 0.0:
                continue
            var g1 = Int(py=dat.contact[c].geom1)
            var g2 = Int(py=dat.contact[c].geom2)
            var ba = Int(py=mm.geom_bodyid[g1])
            var bb = Int(py=mm.geom_bodyid[g2])
            var x = ba if ba < bb else bb
            var y = bb if ba < bb else ba
            var seen = False
            for k in range(len(mj_a)):
                if mj_a[k] == x and mj_b[k] == y:
                    seen = True
            if not seen:
                mj_a.append(x)
                mj_b.append(y)

        for k in range(len(ours_a)):
            var found = False
            for j in range(len(mj_a)):
                if mj_a[j] == ours_a[k] and mj_b[j] == ours_b[k]:
                    found = True
            if found:
                pair_both += 1
            else:
                pair_ours_only += 1
        for j in range(len(mj_a)):
            var found = False
            for k in range(len(ours_a)):
                if mj_a[j] == ours_a[k] and mj_b[j] == ours_b[k]:
                    found = True
            if not found:
                pair_mj_only += 1

        # The predicate the reset path actually needs: "is anything
        # penetrating at all".
        var ours_any = len(ours_a) > 0
        var mj_any = len(mj_a) > 0
        if ours_any == mj_any:
            n_pred_agree += 1

        if t < 10 or our_n != mj_n:
            print(
                "   t", t, " ncon ours", our_n, " mj", mj_n,
                "  penetrating body-pairs ours", len(ours_a),
                " mj", len(mj_a),
                "  <-- NCON DIFFERS" if our_n != mj_n else "",
            )

    print("  poses:", N_POSES, " with any contact:", n_contact_poses)
    print("  ncon equal on", n_ncon_equal, "/", N_POSES,
          "  worst |d ncon|", worst_dncon,
          "  totals ours", sum_ours, " mj", sum_mj)
    print("  penetrating body-pairs: both", pair_both,
          " ours-only", pair_ours_only, " mujoco-only", pair_mj_only)
    print("  ANY-penetration predicate agrees on", n_pred_agree, "/", N_POSES)

    print("  worst single-pose ncon (ours):", max_our_ncon, " buffer cap", MAXC)

    assert_true(
        n_contact_poses >= 8,
        "too few poses produced any contact on either side — the comparison"
        " would be dominated by empty sets and prove nothing",
    )
    assert_true(
        pair_ours_only == 0,
        "we reported a penetrating body-pair MuJoCo does not have — a FALSE"
        " collision, which would make the rejection sampler throw away valid"
        " arm poses",
    )
    assert_true(
        pair_mj_only == 0,
        "MuJoCo found a penetrating body-pair we did not — a MISSED"
        " collision, which would let the rejection sampler accept a pose that"
        " is actually in collision. This was 78 before the plane-mesh cap"
        " landed, and it was NOT a detection gap: one plane-mesh pair filled"
        " the contact buffer before the other pairs were reached",
    )
    # The saturation guard, and the reason the missed pairs above existed.
    # A pose that reaches MAXC has silently stopped emitting, so every set
    # comparison in this test would be measuring the buffer, not the engine.
    assert_true(
        max_our_ncon < MAXC // 2,
        "a single pose produced enough contacts to approach the buffer cap"
        " — with the cap reached, contacts are dropped in emission order and"
        " the body-pair sets above stop meaning anything",
    )
    # Contact COUNT parity. The candidate set for plane-mesh extras is the
    # support vertex's hull neighbourhood, so counts agree even though WHICH
    # extras are chosen is not bit-comparable with qhull's ordering.
    # Measured 2026-08-13: 58/60 exact, totals 436 vs 433, worst |d| 4.
    # The two differing poses have IDENTICAL body-pair sets, and only one of
    # them is plane-mesh dominated (t=43: 15 of MuJoCo's 23 contacts there are
    # plane-mesh; t=20 has just 2, so its -1 lives in the cylinder-mesh /
    # mesh-mesh paths, which this fix does not touch). Bounds are set with
    # headroom rather than frozen at today's numbers.
    assert_true(
        n_ncon_equal >= 54,
        "per-pose contact counts drifted from MuJoCo's — measured 58/60"
        " exact, so falling below 54 means a real change, not the known"
        " hull-triangulation residual",
    )
    assert_true(
        sum_ours <= sum_mj + sum_mj // 10,
        "total contact count exceeded MuJoCo's by more than 10% — the"
        " over-generation this test exists to catch. It was 3311 against 433"
        " (7.6x) before `maxplanemesh`",
    )
    assert_true(
        n_pred_agree == N_POSES,
        "the ANY-penetration predicate disagreed with MuJoCo — that is the"
        " single value `_has_relevant_collisions` reduces to, so the"
        " rejection sampler cannot be built until it holds",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
