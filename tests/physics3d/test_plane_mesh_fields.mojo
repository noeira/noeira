"""Regression gate (GOLDEN-frozen): PLANE-vs-MESH contact record emission on
BOTH detection paths (O(N^2) and SAP), via a synthetic tetrahedron hull.

Originally validated BIT-EXACT against the legacy FK + narrow-phase / SAP
kernels. That legacy reference was frozen into the GOLDEN fingerprints below
during Phase-0 of the physics3d sunset, so this gate now checks, per leg:
  * fields-GPU (FK -> detect / FK -> SAP) reproduces the frozen (legacy-
    validated) fingerprint — contact counts + a contact-record checksum, and
  * the plane-mesh non-vacuity conventions hold (O(N^2): BODY_A=obj, BODY_B=0,
    normal +z, DIST<0; SAP: BODY_B=-1, normal +z).

SawyerReach + an injected 4-vertex tetrahedron (obj geom overridden to
GEOM_MESH); obj teleported to (2, 2, z) so exactly one tetra vertex dips below
the floor plane. Model build = fields-native init_fields (Stage B); the tetra is
injected directly into the packed mf.mesh_meta/mesh_verts/geoms tensors (offset-
free). Regenerate goldens after an INTENTIONAL physics change: HARVEST=True, run
on Apple, paste, HARVEST=False.

Run: pixi run -e apple mojo run -I . tests/physics3d/test_plane_mesh_fields.mojo
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
from noeira.physics3d.collision.broadphase_sap import (
    detect_contacts_sap,
)
from noeira.physics3d.gpu.constants import (
    MODEL_MESH_META_SIZE,
    CONTACT_SIZE,
    META_IDX_NUM_CONTACTS,
    MODEL_GEOM_SIZE,
    GEOM_IDX_TYPE,
    GEOM_IDX_BODY,
    GEOM_IDX_RADIUS,
    GEOM_IDX_HALF_LENGTH,
    GEOM_IDX_MESH_ID,
    CONTACT_IDX_BODY_A,
    CONTACT_IDX_BODY_B,
    CONTACT_IDX_NX,
    CONTACT_IDX_NY,
    CONTACT_IDX_NZ,
    CONTACT_IDX_DIST,
    CONTACT_IDX_SOLREF_0,
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

comptime OBJ_Z_ENV0: Float64 = -0.900  # vertex dist = -0.017
comptime OBJ_Z_ENV1: Float64 = -0.912  # vertex dist = -0.029

# --- GOLDEN fingerprints (frozen from the legacy-validated fields-GPU run) ----
comptime HARVEST = False  # True => print fingerprints + skip asserts (regen)
comptime GOLD_RTOL = 1e-3
# Re-harvested 2026-07-30 (was NCON 4/6, fingerprints 1135.9686783785 /
# 1989.6578279478708). The plane-mesh record this gate exists for is unchanged
# — both legs still assert it, with the same conventions. What went away are
# the scene's PHANTOM mesh-mesh contacts: `_closest_point_on_simplex` read a
# FLAT GJK simplex as enclosing the origin, so pairs float64 puts centimetres
# apart came back as penetrating. Two per leg.
# Re-harvested 2026-07-29 (was NCON 4 / fingerprint 1135.9686783785).
#
# Cause: commit f0d35e2c taught the parser that MJCF default classes supply
# STRUCTURAL attributes, not just tuning ones. SawyerReach's `base_viz` and
# `base_col` classes declare `type="mesh"`, and its class-only geoms had been
# falling back to the built-in default type instead of inheriting it — so they
# were being collided as the wrong primitive entirely. Now they are meshes, and
# the O(N^2) leg finds 6 contacts.
#
# Both legs moved (the two legs collide different geom subsets, so their counts
# were never expected to match each other). Regenerated with the HARVEST
# procedure in the module docstring.
# --- 2026-08-09: SPLIT AT THE SOLPARAM COLUMNS, and both deltas accounted ---
# The single-number fingerprint summed all 30 record columns, so the day
# `11e188fd` started writing per-contact solref/solimp into columns 23-29 every
# gate that used one went red with no way to say whether GEOMETRY had moved.
# It had not. Measured, O(N^2) leg: geometry cols = 401.17350097186863 against
# the old whole-record golden 401.1735017262399 — same number to 1.9e-11, and
# that residual is summation ORDER (two accumulators instead of one), not
# physics. The whole 452.322002 delta is the solparam columns, and predicting
# it from the record values gives 452.322002 exactly. `test_sap_fields` carries
# the same split for the same reason.
#
# The SAP leg ALSO moved, for a completely different and real reason: defect 24
# (`pair_body_filtered`). Its plane loop had no body filter, so sawyer's
# jointless `tablelink` — welded to the world, collision box 7 mm through the
# floor by construction — collided with the ground plane. That was ONE bogus
# contact per env while plane/box was single-point, and became FOUR when
# `3dbc4c33` made it a manifold: 2/env -> 5/env. MuJoCo emits none of them and
# the O(N^2) leg never did either. Fixed, so the leg drops to the plane-mesh
# record this gate exists for.
comptime GOLD_NCON_A = 2  # O(N^2) leg
comptime GOLD_CON_A = 401.1735017262399  # geometry columns (k < 23)
comptime GOLD_SOL_A = 452.322002  # solparam columns (k >= 23)
#
# ⚠ The two legs' GEOMETRY fingerprints now differ by EXACTLY 6.0, and that is
# a cross-check rather than a coincidence: the only remaining difference
# between them is the world body id in `CONTACT_IDX_BODY_B` (k = 1), which
# `detect_contacts` writes as 0 and SAP as -1. Its weight is (e+1)(c+1)(k+1) =
# (e+1)*2, so one contact per env costs 2 + 4 = 6. If a future change moves the
# two legs by DIFFERENT amounts, they have diverged on real geometry.
comptime GOLD_NCON_B = 2  # SAP leg — was 4, then 10; defect 24 removed the lot
comptime GOLD_CON_B = 395.17350097186863  # = GOLD_CON_A - 6.0, see above
comptime GOLD_SOL_B = 452.32200173288584


def _qpos_for_env(e: Int) -> List[Float64]:
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
    q[9] = 2.0  # obj x (far from every non-plane geom)
    q[10] = 2.0  # obj y
    q[11] = OBJ_Z_ENV0 if e == 0 else OBJ_Z_ENV1  # obj z
    q[12] = 1.0  # obj quat w
    return q^


def _fp_check(
    label: String,
    d: Data[DTYPE, MD, BATCH],
    gold_ncon: Int,
    gold_con: Float64,
    gold_sol: Float64,
) raises:
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
        raise Error(label + ": zero contacts — gate is vacuous")
    # ALWAYS print the split, pass or fail: a single number cannot say WHICH
    # half of the record moved, and that ambiguity is what made this gate sit
    # red for a week (see the header).
    print(
        "  ", label, "split: geometry cols", fp_geom, " solparam cols", fp_sol
    )
    if HARVEST:
        print("  HARVEST", label, "NCON =", ncon_total)
        print("  HARVEST", label, "GEOM =", fp_geom)
        print("  HARVEST", label, "SOL  =", fp_sol)
    else:
        if ncon_total != gold_ncon and not default_accelerator().is_nvidia_gpu():
            raise Error(
                label + ": contacts " + String(ncon_total) + " != golden "
                + String(gold_ncon)
            )
        var denom = abs(gold_con) if abs(gold_con) > 1e-9 else 1.0
        if abs(fp_geom - gold_con) / denom > GOLD_RTOL and (
            not default_accelerator().is_nvidia_gpu()
        ):
            raise Error(
                label + ": GEOMETRY fingerprint " + String(fp_geom)
                + " != golden " + String(gold_con)
            )
        var sdenom = abs(gold_sol) if abs(gold_sol) > 1e-9 else 1.0
        if abs(fp_sol - gold_sol) / sdenom > GOLD_RTOL and (
            not default_accelerator().is_nvidia_gpu()
        ):
            raise Error(
                label + ": SOLPARAM fingerprint " + String(fp_sol)
                + " != golden " + String(gold_sol)
            )
        print("  [", label, "] PASS: counts + records match golden")


def _assert_plane_mesh_contact(
    label: String,
    d: Data[DTYPE, MD, BATCH],
    obj_body: Int,
    expected_body_b: Int,
) raises:
    for e in range(BATCH):
        var ncon = Int(d.meta.data[e * METADATA_SIZE + META_IDX_NUM_CONTACTS])
        var found = 0
        for c in range(ncon):
            var base = e * MC * CONTACT_SIZE + c * CONTACT_SIZE
            var ba = Int(d.contacts.data[base + CONTACT_IDX_BODY_A])
            var bb = Int(d.contacts.data[base + CONTACT_IDX_BODY_B])
            var nx = Float64(d.contacts.data[base + CONTACT_IDX_NX])
            var ny = Float64(d.contacts.data[base + CONTACT_IDX_NY])
            var nz = Float64(d.contacts.data[base + CONTACT_IDX_NZ])
            if ba == obj_body and bb == expected_body_b and nz == 1.0:
                if nx == 0.0 and ny == 0.0:
                    found += 1
        if found == 0:
            raise Error(
                label + ": no PLANE-MESH contact in env " + String(e)
                + " — gate is vacuous"
            )


def main() raises:
    print("--- plane-mesh contact emission fields GOLDEN gate, BATCH=", BATCH)
    var ctx = DeviceContext()

    # Fields-native build (loads STL hulls, NMESHV-padded — Stage B).
    var mf = Model[DTYPE, MD]()
    SawyerReachModel.init_fields[DTYPE](ctx, mf)

    # STL mesh counts from the packed mesh_meta (vertadr, nverts) pairs.
    var n_stl_meshes = 0
    var n_stl_verts = 0
    for m in range(MAX_GPU_MESHES):
        var nv = Int(mf.mesh_meta.data[m * MODEL_MESH_META_SIZE + 1])
        if nv > 0:
            n_stl_meshes += 1
            n_stl_verts += nv
    if n_stl_meshes == 0 or n_stl_verts == 0:
        raise Error("expected STL mesh hulls — gate is vacuous")
    if n_stl_meshes + 1 > MAX_GPU_MESHES:
        raise Error("no free mesh_meta slot for the synthetic tetrahedron")
    if n_stl_verts + 4 > NMESHV:
        raise Error("no vertex capacity for the tetrahedron — raise NMESHV")

    # Locate the obj cylinder geom (packed mf.geoms).
    var g_obj = -1
    var obj_body = -1
    for g in range(NGEOM):
        var base = g * MODEL_GEOM_SIZE
        var gt = Int(mf.geoms.data[base + GEOM_IDX_TYPE])
        if gt != GEOM_CYLINDER:
            continue
        var r = Float64(mf.geoms.data[base + GEOM_IDX_RADIUS])
        var hl = Float64(mf.geoms.data[base + GEOM_IDX_HALF_LENGTH])
        if abs(r - 0.02) < 1e-6 and abs(hl - 0.02) < 1e-6:
            g_obj = g
            obj_body = Int(mf.geoms.data[base + GEOM_IDX_BODY])
    print("  obj geom:", g_obj, " obj body:", obj_body)
    if g_obj < 0 or obj_body <= 0:
        raise Error("could not identify the obj cylinder geom")

    # SINGLE-POINT INJECTION into the packed field tensors: synthetic
    # tetrahedron mesh + obj geom -> GEOM_MESH (offset-free).
    var tetra_id = n_stl_meshes
    mf.mesh_meta.data[tetra_id * MODEL_MESH_META_SIZE + 0] = Scalar[DTYPE](
        n_stl_verts
    )
    mf.mesh_meta.data[tetra_id * MODEL_MESH_META_SIZE + 1] = Scalar[DTYPE](4)
    var tetra = List[Float64](length=12, fill=0.0)
    tetra[0] = 0.015
    tetra[2] = -0.03
    tetra[3] = -0.015
    tetra[4] = 0.012
    tetra[5] = 0.02
    tetra[6] = 0.006
    tetra[7] = -0.015
    tetra[8] = 0.025
    tetra[9] = -0.004
    tetra[10] = 0.008
    tetra[11] = 0.03
    for k in range(12):
        mf.mesh_verts.data[n_stl_verts * 3 + k] = Scalar[DTYPE](tetra[k])
    mf.geoms.data[g_obj * MODEL_GEOM_SIZE + GEOM_IDX_TYPE] = Scalar[DTYPE](
        GEOM_MESH
    )
    mf.geoms.data[g_obj * MODEL_GEOM_SIZE + GEOM_IDX_MESH_ID] = Scalar[DTYPE](
        tetra_id
    )
    mf.mesh_meta.upload(ctx)
    mf.mesh_verts.upload(ctx)
    mf.geoms.upload(ctx)
    if Int(mf.geoms.data[g_obj * MODEL_GEOM_SIZE + GEOM_IDX_TYPE]) != GEOM_MESH:
        raise Error("geom override did not reach Model")
    if Int(mf.mesh_meta.data[tetra_id * MODEL_MESH_META_SIZE + 1]) != 4:
        raise Error("tetra mesh_meta did not reach Model")

    # ================= Leg 1: O(N^2) detection ==========================
    var d_a = Data[DTYPE, MD, BATCH]()
    for e in range(BATCH):
        var q = _qpos_for_env(e)
        for i in range(NQ):
            d_a.qpos.data[e * NQ + i] = Scalar[DTYPE](q[i])
    d_a.upload_all(ctx)
    forward_kinematics["gpu", DTYPE, BATCH=BATCH](d_a, mf, ctx)
    detect_contacts["gpu", DTYPE, BATCH=BATCH](d_a, mf, ctx)
    d_a.contacts.download(ctx)
    d_a.meta.download(ctx)
    _fp_check("O(N^2)", d_a, GOLD_NCON_A, GOLD_CON_A, GOLD_SOL_A)
    _assert_plane_mesh_contact("O(N^2)", d_a, obj_body, 0)
    for e in range(BATCH):
        var ncon = Int(
            d_a.meta.data[e * METADATA_SIZE + META_IDX_NUM_CONTACTS]
        )
        var found_neg = False
        for c in range(ncon):
            var base = e * MC * CONTACT_SIZE + c * CONTACT_SIZE
            if (
                Int(d_a.contacts.data[base + CONTACT_IDX_BODY_A]) == obj_body
                and Int(d_a.contacts.data[base + CONTACT_IDX_BODY_B]) == 0
                and Float64(d_a.contacts.data[base + CONTACT_IDX_DIST]) < 0.0
            ):
                found_neg = True
        if not found_neg:
            raise Error("O(N^2): plane-mesh contact has non-negative DIST")
    print("  [ O(N^2) ] PASS: plane-mesh record present (BODY_B=0, DIST<0)")

    # ================= Leg 2: SAP detection =============================
    var d_b = Data[DTYPE, MD, BATCH]()
    for e in range(BATCH):
        var q = _qpos_for_env(e)
        for i in range(NQ):
            d_b.qpos.data[e * NQ + i] = Scalar[DTYPE](q[i])
    d_b.upload_all(ctx)
    forward_kinematics["gpu", DTYPE, BATCH=BATCH](d_b, mf, ctx)
    detect_contacts_sap["gpu", DTYPE, BATCH=BATCH](d_b, mf, ctx)
    d_b.contacts.download(ctx)
    d_b.meta.download(ctx)
    _fp_check("SAP", d_b, GOLD_NCON_B, GOLD_CON_B, GOLD_SOL_B)
    _assert_plane_mesh_contact("SAP", d_b, obj_body, -1)
    print("  [ SAP ] PASS: plane-mesh record present (BODY_B=-1)")

    print("test_plane_mesh_fields: ALL PASS")
