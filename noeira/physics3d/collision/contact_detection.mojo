"""Contact detection over per-field tensors (migration P2, single-source).

Per-field port of `_geom_world_pos_gpu` + `detect_contacts_gpu`
(collision/contact_detection.mojo) — arithmetic verbatim. Reads FK products
(`d.xpos`, `d.xquat`) + geom/body records + model meta + exclude pairs;
writes packed contact records into `d.contacts` and the contact count into
`d.meta` (META_IDX_NUM_CONTACTS).

Operands (10): xpos, xquat (data) + geoms, bodies, mmeta, excludes,
mesh_meta, mesh_verts (model) + contacts, smeta (data outputs). Mesh
collision (plane-mesh vertex scans + GJK/EPA fallback via gjk) is
compiled in only when NMESH_VERTS > 0; zero-mesh models keep today's
branch structure (mesh branches degrade to `continue`)."""

from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from std.math import sqrt

from ..kinematics.quat_math import (
    gpu_quat_rotate,
    gpu_quat_mul,
    quat_rotate_inverse,
)
from ..constants import (
    GEOM_SPHERE,
    GEOM_CAPSULE,
    GEOM_BOX,
    GEOM_PLANE,
    GEOM_CYLINDER,
    GEOM_MESH,
    GEOM_ELLIPSOID,
    GEOM_HFIELD,
    mj_geom_type_rank,
)
from ..fields.scratch import Scratch, cap
from ..fields import (
    Data,
    Model,
    Dims,
    DimsLike,
    AsStatic,
    may_exist,
    DYN1,
    DYN2,
    rl1,
    rl2,
)
from ..gpu.constants import (
    MODEL_BODY_SIZE,
    MODEL_GEOM_SIZE,
    MODEL_META_SIZE,
    METADATA_SIZE,
    MODEL_META_IDX_NEXCLUDE,
    MODEL_META_IDX_NPAIR,
    MODEL_META_IDX_CCD_TOLERANCE,
    MODEL_META_IDX_CCD_ITERATIONS,
    MODEL_META_IDX_MULTICCD_DISABLED,
    MJ_CCD_TOLERANCE,
    MJ_CCD_ITERATIONS,
    MODEL_PAIR_SIZE,
    PAIR_IDX_GEOM1,
    PAIR_IDX_GEOM2,
    PAIR_IDX_CONDIM,
    PAIR_IDX_FRICTION,
    PAIR_IDX_FRICTION_SPIN,
    PAIR_IDX_FRICTION_ROLL,
    PAIR_IDX_SOLREF_0,
    PAIR_IDX_SOLREF_1,
    PAIR_IDX_SOLIMP_0,
    PAIR_IDX_SOLIMP_1,
    PAIR_IDX_SOLIMP_2,
    PAIR_IDX_SOLIMP_3,
    PAIR_IDX_SOLIMP_4,
    PAIR_IDX_MARGIN,
    PAIR_IDX_GAP,
    BODY_IDX_PARENT,
    BODY_IDX_WELDID,
    BODY_IDX_DOFNUM,
    META_IDX_NUM_CONTACTS,
    CONTACT_SIZE,
    CONTACT_IDX_BODY_A,
    CONTACT_IDX_BODY_B,
    CONTACT_IDX_POS_X,
    CONTACT_IDX_POS_Y,
    CONTACT_IDX_POS_Z,
    CONTACT_IDX_NX,
    CONTACT_IDX_NY,
    CONTACT_IDX_NZ,
    CONTACT_IDX_DIST,
    CONTACT_IDX_INCLUDEMARGIN,
    CONTACT_IDX_FRICTION,
    CONTACT_IDX_FRICTION_SPIN,
    CONTACT_IDX_FRICTION_ROLL,
    CONTACT_IDX_CONDIM,
    CONTACT_IDX_SOLREF_0,
    CONTACT_IDX_SOLREF_1,
    CONTACT_IDX_SOLIMP_0,
    CONTACT_IDX_SOLIMP_1,
    CONTACT_IDX_SOLIMP_2,
    CONTACT_IDX_SOLIMP_3,
    CONTACT_IDX_SOLIMP_4,
    CONTACT_IDX_FRAME_T1_X,
    CONTACT_IDX_FRAME_T1_Y,
    CONTACT_IDX_FRAME_T1_Z,
    GEOM_IDX_TYPE,
    GEOM_IDX_BODY,
    GEOM_IDX_POS_X,
    GEOM_IDX_POS_Y,
    GEOM_IDX_POS_Z,
    GEOM_IDX_QUAT_X,
    GEOM_IDX_QUAT_Y,
    GEOM_IDX_QUAT_Z,
    GEOM_IDX_QUAT_W,
    GEOM_IDX_RADIUS,
    GEOM_IDX_HALF_LENGTH,
    GEOM_IDX_HALF_X,
    GEOM_IDX_HALF_Y,
    GEOM_IDX_HALF_Z,
    GEOM_IDX_FRICTION,
    GEOM_IDX_CONTYPE,
    GEOM_IDX_CONAFFINITY,
    GEOM_IDX_CONDIM,
    GEOM_IDX_PRIORITY,
    GEOM_IDX_SOLREF_0,
    GEOM_IDX_SOLREF_1,
    GEOM_IDX_SOLIMP_0,
    GEOM_IDX_SOLIMP_1,
    GEOM_IDX_SOLIMP_2,
    GEOM_IDX_SOLIMP_3,
    GEOM_IDX_SOLIMP_4,
    GEOM_IDX_FRICTION_SPIN,
    GEOM_IDX_FRICTION_ROLL,
    GEOM_IDX_RBOUND,
    GEOM_IDX_MARGIN,
    GEOM_IDX_GAP,
    GEOM_IDX_MESH_ID,
    GEOM_IDX_HFIELD_ID,
    MAX_GPU_MESHES,
    MAX_GPU_HFIELDS,
    MODEL_HFIELD_META_SIZE,
    MODEL_MESH_META_SIZE,
    MODEL_MESH_POLY_SIZE,
    MESH_META_IDX_POLYADR,
    MESH_META_IDX_POLYNUM,
    mesh_max_poly,
    mesh_max_polyvert,
    mesh_max_edge,
    MODEL_META_IDX_FILTERPARENT_DISABLED,
)
from .contact_order import sort_contacts_mujoco_order
from .plane_frame import (
    plane_world_normal,
    to_plane_frame,
    from_plane_frame,
    quat_to_plane_frame,
)
from .collision_primitives import (
    sphere_sphere,
    capsule_sphere,
    capsule_capsule_manifold,
    CC_MAX_POINTS,
    box_sphere,
    box_capsule_manifold,
    CB_MAX_POINTS,
    box_box,
    box_box_manifold,
    BB_MAX_POINTS,
    box_plane,
    ellipsoid_plane,
    cylinder_plane,
    cylinder_sphere,
    cylinder_capsule,
    cylinder_cylinder,
    cylinder_box,
)
@always_inline
def _hf_len(n: Int) -> Int:
    """`Model.hfield_data` is allocated with `_at_least_one`, so a model with
    no heightfield still has ONE element. A `Layout.row_major(0)` over it is a
    zero-size view the runtime rejects; every other tensor here is sized by a
    dimension that is never legitimately zero."""
    return n if n > 0 else 1


from .hfield_convex import hfield_convex_contacts
from .ccd_workspace import CCD_WS_SIZE, COLL_CCD_LANES
from .gjk import gjk_epa, gjk_epa_witness, hillclimb_support_index
from .gjk import _HILL_PROBE, hill_probe
from .native_multicontact import (
    native_multicontact_contacts,
    MC_ENABLED,
)
from .multi_ccd import multi_ccd_pair_supported, multi_ccd_extra_contacts

comptime CD_TPB: Int = 64


@always_inline
def _geom_world_pos[
    DTYPE: DType,
    L_GEOMS: Layout,
    L_XPOS: Layout,
    L_XQUAT: Layout,
](
    env: Int,
    g: Int,
    geoms: LayoutTensor[
        DTYPE, L_GEOMS, MutAnyOrigin
    ],
    xpos: LayoutTensor[
        DTYPE, L_XPOS, MutAnyOrigin
    ],
    xquat: LayoutTensor[
        DTYPE, L_XQUAT, MutAnyOrigin
    ],
    mut out_px: Scalar[DTYPE],
    mut out_py: Scalar[DTYPE],
    mut out_pz: Scalar[DTYPE],
    mut out_qx: Scalar[DTYPE],
    mut out_qy: Scalar[DTYPE],
    mut out_qz: Scalar[DTYPE],
    mut out_qw: Scalar[DTYPE],
):
    """Compute geom world pos/quat (verbatim from _geom_world_pos_gpu)."""
    var body_idx = Int(rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_BODY]))
    var lx = rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_POS_X])
    var ly = rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_POS_Y])
    var lz = rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_POS_Z])
    var lqx = rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_QUAT_X])
    var lqy = rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_QUAT_Y])
    var lqz = rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_QUAT_Z])
    var lqw = rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_QUAT_W])
    if body_idx == 0:
        out_px = lx
        out_py = ly
        out_pz = lz
        out_qx = lqx
        out_qy = lqy
        out_qz = lqz
        out_qw = lqw
        return
    var bpx = rebind[Scalar[DTYPE]](xpos[env, body_idx * 3 + 0])
    var bpy = rebind[Scalar[DTYPE]](xpos[env, body_idx * 3 + 1])
    var bpz = rebind[Scalar[DTYPE]](xpos[env, body_idx * 3 + 2])
    var bqx = rebind[Scalar[DTYPE]](xquat[env, body_idx * 4 + 0])
    var bqy = rebind[Scalar[DTYPE]](xquat[env, body_idx * 4 + 1])
    var bqz = rebind[Scalar[DTYPE]](xquat[env, body_idx * 4 + 2])
    var bqw = rebind[Scalar[DTYPE]](xquat[env, body_idx * 4 + 3])
    if (
        lx == Scalar[DTYPE](0)
        and ly == Scalar[DTYPE](0)
        and lz == Scalar[DTYPE](0)
        and lqx == Scalar[DTYPE](0)
        and lqy == Scalar[DTYPE](0)
        and lqz == Scalar[DTYPE](0)
        and lqw == Scalar[DTYPE](1)
    ):
        out_px = bpx
        out_py = bpy
        out_pz = bpz
        out_qx = bqx
        out_qy = bqy
        out_qz = bqz
        out_qw = bqw
        return
    var rotated = gpu_quat_rotate(bqx, bqy, bqz, bqw, lx, ly, lz)
    out_px = bpx + rotated[0]
    out_py = bpy + rotated[1]
    out_pz = bpz + rotated[2]
    var wq = gpu_quat_mul(bqx, bqy, bqz, bqw, lqx, lqy, lqz, lqw)
    out_qx = wq[0]
    out_qy = wq[1]
    out_qz = wq[2]
    out_qw = wq[3]


# `mjc_PlaneConvex`'s two constants, `engine_collision_convex.c`. ⚠ VERIFIED
# IDENTICAL in the 3.3.6, 3.6.0 and 3.11.0 trees, so the 3.10.0 runtime shares
# them — one of the few places in this engine with no version drift to resolve.
comptime MAXPLANEMESH: Int = 3
comptime TOLPLANEMESH: Float64 = 0.3


@always_inline
def _plane_mesh_contacts[
    DTYPE: DType,
    BODY_B: Int,
    MARGIN_IN_DIST: Bool,
    WRITE_INCLUDEMARGIN: Bool,
    D: DimsLike,
    L_GEOMS: Layout,
    L_MESH_META: Layout,
    L_MESH_VERTS: Layout,
    L_MESH_VERT_EDGEADR: Layout,
    L_MESH_EDGES: Layout,
    L_CONTACTS: Layout,
](
    env: Int,
    g: Int,
    g_body: Int,
    p_x: Scalar[DTYPE],
    p_y: Scalar[DTYPE],
    p_z: Scalar[DTYPE],
    q_x: Scalar[DTYPE],
    q_y: Scalar[DTYPE],
    q_z: Scalar[DTYPE],
    q_w: Scalar[DTYPE],
    ground_z: Scalar[DTYPE],
    plp_x: Scalar[DTYPE],
    plp_y: Scalar[DTYPE],
    plp_z: Scalar[DTYPE],
    plq_x: Scalar[DTYPE],
    plq_y: Scalar[DTYPE],
    plq_z: Scalar[DTYPE],
    plq_w: Scalar[DTYPE],
    contact_margin: Scalar[DTYPE],
    contact_friction: Scalar[DTYPE],
    contact_friction_spin: Scalar[DTYPE],
    contact_friction_roll: Scalar[DTYPE],
    contact_condim: Int,
    dims: D,
    geoms: LayoutTensor[
        DTYPE, L_GEOMS, MutAnyOrigin
    ],
    mesh_meta: LayoutTensor[
        DTYPE,
        L_MESH_META,
        MutAnyOrigin,
    ],
    mesh_verts: LayoutTensor[
        DTYPE, L_MESH_VERTS, MutAnyOrigin
    ],
    mesh_vert_edgeadr: LayoutTensor[
        DTYPE, L_MESH_VERT_EDGEADR, MutAnyOrigin
    ],
    mesh_edges: LayoutTensor[
        DTYPE, L_MESH_EDGES, MutAnyOrigin
    ],
    contacts: LayoutTensor[
        DTYPE, L_CONTACTS,
        MutAnyOrigin,
    ],
    mut num_contacts: Int,
    # The mesh hill climb's warm vertex, in and out (`hillclimb_support_index`):
    # -1 = cold. The SAP plane phase hands in the pair's cross-step slot
    # (`ccd_workspace.mojo`); the O(N^2) detector stays cold.
    mut warm: Int,
    # ⚠ THE GAP HALF OF THE PAIR'S MARGIN, DEFAULTED TO 0 SO EVERY
    # EXISTING CALL SITE IS UNCHANGED. `contact_margin` is the narrowphase
    # CUTOFF (`margin + gap`); what a contact STORES as its
    # `includemargin` is `contact_margin - contact_gap`, and the solver excludes
    # `dist >= includemargin`. See `GEOM_IDX_GAP`.
    contact_gap: Scalar[DTYPE] = Scalar[DTYPE](0),
    # The emission cap. -1 = the model's `max_contacts` (every serial
    # caller); the block-per-env collision kernel passes the end of the
    # thread's staging window instead, so a routine cannot write past it.
    max_contacts_in: Int = -1,
):
    """Plane-mesh, after `mjc_PlaneConvex` (`engine_collision_convex.c`).

    `p_*` / `q_*` are the mesh geom's pose IN THE PLANE'S FRAME and `ground_z`
    is 0 there, so the vertex heights below are heights above the plane
    whatever way the plane faces. `plp_*` / `plq_*` are the plane's own world
    pose, used only to put the contact point and normal back into world —
    see `collision/plane_frame.mojo`.

    ⚠⚠ THIS USED TO EMIT ONE CONTACT PER HULL VERTEX BELOW THE PLANE, WITH NO
    CAP. MuJoCo emits at most `maxplanemesh = 3`. Measured on Jaco
    `reach_site_features` over 60 in-range poses before the fix: 3311 contacts
    against MuJoCo's 433, and one pose produced 256 — the contact buffer's
    capacity, reached by a SINGLE plane-mesh pair. The failure did not look
    like over-generation: that one pair filled the buffer before other pairs
    were reached, so we found FEWER distinct penetrating body-pairs than
    MuJoCo (78 missing, 0 spurious). A count bug presenting as missing
    collisions. See `tests/physics3d/test_jaco_contacts_vs_mujoco.mojo`.

    The three contacts are:

      * contact 0, ALWAYS the support point in the `-normal` direction — the
        lowest hull vertex. MuJoCo hill-climbs the edge graph to find it; we
        take the global argmin, which is the same vertex whenever the minimum
        is unique. ⚠ They can differ on an exact PLATEAU (a face lying flat on
        the plane, every vertex at one height): hill-climbing stops at
        whichever tied vertex it reached, the argmin below takes the lowest
        index. Both are legitimate support points, and the contact SET that
        follows is the same size either way.
      * up to two more, each below `margin` and at least
        `tolplanemesh * rbound` from contact 0. Candidates come from that
        vertex's hull-edge NEIGHBOURS when the mesh has a graph, and from a
        scan of every vertex when it does not — MuJoCo's two branches on
        `mesh_graphadr < 0`, which we signal with a -1 edge address. The
        neighbour restriction is not an optimisation: scanning all vertices
        for a mesh that HAS a graph would find candidates all over the shape
        and return 3 contacts almost always, where MuJoCo returns 1 or 2 for
        89 of 644 pairs (measured). Every Jaco mesh has a graph.

    ⚠ THE SPREAD FILTER IS ASYMMETRIC, AND DELIBERATELY SO: it measures from
    the RAW candidate vertex to contact 0's POS, which has already been pushed
    half its penetration depth along the normal. It also compares only against
    contact 0, never against the other extra — so contacts 1 and 2 may be
    arbitrarily close to each other. Both are `addplanemesh`'s behaviour.

    ⚠ WHICH extras are chosen is NOT bit-comparable with MuJoCo when more than
    two neighbours pass: MuJoCo takes them in qhull's internal facet order,
    which nothing here reproduces. The COUNT matches (it depends only on the
    candidate set), and so does contact 0. See `build_hull_edge_graph`.

    ⚠⚠ AND THE CANDIDATE SET ITSELF DIFFERS, SO THIS IS NOT PURELY AN ORDERING
    QUESTION. `build_hull_edge_graph` says order is the only difference; that
    is what was believed, and it is measurably incomplete. Against MuJoCo's
    `mesh_graph` for all nine Jaco meshes:

        mesh  hull verts (ours/MuJoCo)  vertices whose neighbour SET differs
          0        207 / 207              47 of 207   (82 extra, 82 missing)
          1        199 / 198              67 of 198   (98 extra, 96 missing)
          2        130 / 130              10 of 130    (8 extra,  8 missing)
          3        136 / 136              31 of 136   (56 extra, 56 missing)
          4        230 / 230              14 of 230    (8 extra,  8 missing)
          5        151 / 151              39 of 151   (72 extra, 72 missing)
          6         35 /  35               0 of  35
          7        156 / 156              46 of 156   (28 extra, 28 missing)
          8        213 / 213              10 of 213    (6 extra,  6 missing)

    EXTRA AND MISSING ARE EQUAL ON EVERY MESH and the mean degree matches to
    two decimals — the signature of a different TRIANGULATION of the SAME
    polytope (edge count fixed by Euler, diagonals across coplanar facets
    chosen differently), not of a different polytope.

    ⚠ THE HULL ITSELF IS NOT IN QUESTION, which is what makes this a tie-break
    rather than a defect. Vertex sets are in BIJECTION on eight of the nine
    meshes (worst position mismatch 7.5e-9 = float32 quantisation of
    `mesh_vert`), and every one of our vertices lies within MuJoCo's own
    facet-plane slack of its hull: per mesh, our max overshoot equals MuJoCo's
    OWN to three significant figures (mesh 0: 3.929e-3 for both, qhull's merge
    tolerance). Mesh 1 is the exception — we keep one extra vertex, and it
    sits INSIDE MuJoCo's hull by 8.5e-10, i.e. on the surface: a coplanar
    point qhull merged away and `compute_convex_hull` does not.

    MEASURED CONSEQUENCE, Jaco `reach_site_features`, 40 in-range poses,
    comparing `qvel` after a full `mj_step` on both sides:

        worst |d(pos)| on a PENETRATING contact     |d(qvel)|
        ~1e-9                                       1e-9 .. 4e-6
        ~2e-2                                       2e-1 .. 1.9

    Contact 0 is exact wherever the pair is plane-mesh (our deepest contact
    agrees with MuJoCo's to 1e-10 or better on every such pose), so the whole
    residual is carried by the up-to-two extras landing ~2e-2 m away with a
    different penetration depth, hence a different `aref`. The counts agree on
    38 of the 40 poses.

    ⚠ ANY MEASUREMENT HERE MUST FILTER ON `dist < 0`. A contact inside the
    margin but not touching carries no force, so including it measures this
    routine's bookkeeping rather than anything the solve can see: pose 17 of
    that sweep has |d(pos)| = 1.8e-2 against |d(qvel)| = 3.6e-7.

    CLOSING THIS MEANS RUNNING QHULL, not tightening a tolerance — the choice
    lives in `qh_triangulate`'s diagonals and in the facet order
    `vertex->neighbors` happens to hold.
    """
    var max_contacts = (
        max_contacts_in if max_contacts_in >= 0 else dims.get_max_contacts()
    )
    var pn = plane_world_normal[DTYPE](plq_x, plq_y, plq_z, plq_w)
    var m_id = Int(rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_MESH_ID]))
    if m_id < 0:
        return
    var pm_vadr = Int(rebind[Scalar[DTYPE]](mesh_meta[m_id, 0]))
    var pm_vnum = Int(rebind[Scalar[DTYPE]](mesh_meta[m_id, 1]))
    if pm_vnum <= 0:
        return

    # ── contact 0: the support point in -normal, i.e. the lowest vertex ──
    #
    # ⚠⚠ THIS WAS A FULL ARGMIN OVER EVERY HULL VERTEX, AND IT WAS THE SINGLE
    # LARGEST LINE ITEM IN THE STEP. Per-stage timers over 20 000 physics
    # steps put the SAP plane loop at 8.1 µs/step on SO-ARM100 and 11.1 µs on
    # SO-ARM101 — 23-25% of the entire physics step — against 0.23 µs for
    # world poses and 0.11 µs for AABBs. On SO-101 that was ONE call per step
    # scanning one ~4 000-vertex hull. The neighbour walk below already used
    # the edge graph; only this search did not.
    #
    # Minimising height above the plane IS a support query. Height is
    # `p_z + (R v)_z = p_z + dot(v, R^T ez)`, so the lowest vertex maximises
    # `dot(v, R^T(0,0,-1))` — the same hill climb GJK runs, in the direction
    # the plane's own frame calls straight down. This is also what MuJoCo does
    # (`mjc_PlaneConvex` calls `mjccd_support`), so it moves us TOWARD the
    # reference rather than away from it.
    #
    # ⚠ THE FALLBACK IS NOT DEAD CODE. `hillclimb_support_index` returns -1
    # for a mesh below `_HILLCLIMB_MIN` vertices or without adjacency, and
    # MuJoCo has the same two branches (`mesh_graphadr < 0`). Small collision
    # primitives baked as meshes take it every step.
    #
    # ⚠ TIE-BREAK, on an exact PLATEAU (a facet lying flat on the plane): the
    # climb stops at the first local maximum it reaches, the old argmin took
    # the lowest index. Both are legitimate support points and `best_h` is
    # identical, but the EXTRAS are drawn from this vertex's neighbours, so
    # which extras appear can differ on such a pose. See the note above on the
    # candidate set already differing from MuJoCo's by triangulation.
    var best = -1
    var best_h = Scalar[DTYPE](0)
    var best_x = Scalar[DTYPE](0)
    var best_y = Scalar[DTYPE](0)

    # Straight down in the PLANE's frame, expressed in the mesh's own frame.
    var ld = quat_rotate_inverse[DTYPE](
        q_x, q_y, q_z, q_w,
        Scalar[DTYPE](0), Scalar[DTYPE](0), Scalar[DTYPE](-1),
    )
    comptime if _HILL_PROBE:
        hill_probe()[].plane_calls += 1
    var hc = hillclimb_support_index[DTYPE](
        ld[0], ld[1], ld[2],
        mesh_verts, mesh_vert_edgeadr, mesh_edges,
        pm_vadr, pm_vnum, warm,
    )
    if hc >= 0:
        warm = hc
        var vx = rebind[Scalar[DTYPE]](mesh_verts[pm_vadr + hc, 0])
        var vy = rebind[Scalar[DTYPE]](mesh_verts[pm_vadr + hc, 1])
        var vz = rebind[Scalar[DTYPE]](mesh_verts[pm_vadr + hc, 2])
        var lp = gpu_quat_rotate(q_x, q_y, q_z, q_w, vx, vy, vz)
        best = hc
        best_h = (p_z + lp[2]) - ground_z
        best_x = p_x + lp[0]
        best_y = p_y + lp[1]
    else:
        for vi in range(pm_vnum):
            var vx = rebind[Scalar[DTYPE]](mesh_verts[pm_vadr + vi, 0])
            var vy = rebind[Scalar[DTYPE]](mesh_verts[pm_vadr + vi, 1])
            var vz = rebind[Scalar[DTYPE]](mesh_verts[pm_vadr + vi, 2])
            var lp = gpu_quat_rotate(q_x, q_y, q_z, q_w, vx, vy, vz)
            var h = (p_z + lp[2]) - ground_z
            if best < 0 or h < best_h:
                best = vi
                best_h = h
                best_x = p_x + lp[0]
                best_y = p_y + lp[1]

    # `if (dist > margin) return 0` — NON-strict acceptance, unlike the
    # extras below, which use a strict `vdot > threshold`. Kept distinct
    # because they are distinct in the reference.
    if best_h > contact_margin:
        return

    # Contact 0's POS, which the spread filter measures against.
    var first_z = ground_z + best_h * Scalar[DTYPE](0.5)

    # ── up to two extra contacts around the support vertex ───────────────
    var sel_x = Array[Scalar[DTYPE], MAXPLANEMESH](fill=Scalar[DTYPE](0))
    var sel_y = Array[Scalar[DTYPE], MAXPLANEMESH](fill=Scalar[DTYPE](0))
    var sel_h = Array[Scalar[DTYPE], MAXPLANEMESH](fill=Scalar[DTYPE](0))
    sel_x[0] = best_x
    sel_y[0] = best_y
    sel_h[0] = best_h
    var nsel = 1

    var rbound = rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_RBOUND])
    var min_spread = rbound * Scalar[DTYPE](TOLPLANEMESH)
    # MuJoCo has TWO candidate sources and picks on `mesh_graphadr < 0`; ours
    # signals the same thing with a -1 edge address. The graph branch walks the
    # support vertex's neighbours; the no-graph branch scans every vertex in
    # index order, skipping the support vertex (`i != obj.meshindex`). Both
    # then apply the SAME margin and spread tests, which is why they share the
    # body below rather than being written out twice.
    var e = Int(rebind[Scalar[DTYPE]](mesh_vert_edgeadr[pm_vadr + best]))
    var use_graph = e >= 0
    var scan = 0
    while nsel < MAXPLANEMESH:
        var nb: Int
        if use_graph:
            nb = Int(rebind[Scalar[DTYPE]](mesh_edges[e]))
            if nb < 0:
                break
            e += 1
        else:
            if scan >= pm_vnum:
                break
            # `nb` is assigned before the skip so that it is definitely
            # initialised on every path out of this branch.
            nb = pm_vadr + scan
            var vi = scan
            scan += 1
            if vi == best:
                continue
        var vx = rebind[Scalar[DTYPE]](mesh_verts[nb, 0])
        var vy = rebind[Scalar[DTYPE]](mesh_verts[nb, 1])
        var vz = rebind[Scalar[DTYPE]](mesh_verts[nb, 2])
        var lp = gpu_quat_rotate(q_x, q_y, q_z, q_w, vx, vy, vz)
        var wx = p_x + lp[0]
        var wy = p_y + lp[1]
        var wz = p_z + lp[2]
        var h = wz - ground_z
        if h >= contact_margin:
            continue
        var dx = wx - best_x
        var dy = wy - best_y
        var dz = wz - first_z
        if sqrt(dx * dx + dy * dy + dz * dz) < min_spread:
            continue
        sel_x[nsel] = wx
        sel_y[nsel] = wy
        sel_h[nsel] = h
        nsel += 1

    # ── emit ─────────────────────────────────────────────────────────────
    for k in range(nsel):
        if num_contacts >= max_contacts:
            break
        var dist_v = sel_h[k]
        var c_off = num_contacts * CONTACT_SIZE
        contacts[env, c_off + CONTACT_IDX_BODY_A] = Scalar[DTYPE](g_body)
        contacts[env, c_off + CONTACT_IDX_BODY_B] = Scalar[DTYPE](BODY_B)
        var cw = from_plane_frame[DTYPE](
            plp_x, plp_y, plp_z, plq_x, plq_y, plq_z, plq_w,
            sel_x[k], sel_y[k], ground_z + dist_v * Scalar[DTYPE](0.5),
        )
        contacts[env, c_off + CONTACT_IDX_POS_X] = cw[0]
        contacts[env, c_off + CONTACT_IDX_POS_Y] = cw[1]
        contacts[env, c_off + CONTACT_IDX_POS_Z] = cw[2]
        contacts[env, c_off + CONTACT_IDX_NX] = pn[0]
        contacts[env, c_off + CONTACT_IDX_NY] = pn[1]
        contacts[env, c_off + CONTACT_IDX_NZ] = pn[2]
        # ⚠⚠ TWO CONVENTIONS, ONE MEANING. `MARGIN_IN_DIST` folds the margin
        # into the stored distance and leaves `INCLUDEMARGIN` untouched;
        # `WRITE_INCLUDEMARGIN` stores the raw distance and the margin beside
        # it. The solver reads `dist - includemargin` either way — so BOTH must
        # subtract the INCLUDEMARGIN (`margin`), never the cutoff
        # (`margin + gap`), or a gap-band contact reads as penetrating.
        comptime if MARGIN_IN_DIST:
            contacts[env, c_off + CONTACT_IDX_DIST] = dist_v - (
                contact_margin - contact_gap
            )
        else:
            contacts[env, c_off + CONTACT_IDX_DIST] = dist_v
        comptime if WRITE_INCLUDEMARGIN:
            contacts[
                env, c_off + CONTACT_IDX_INCLUDEMARGIN
            ] = contact_margin - contact_gap
        contacts[env, c_off + CONTACT_IDX_FRICTION] = contact_friction
        contacts[
            env, c_off + CONTACT_IDX_FRICTION_SPIN
        ] = contact_friction_spin
        contacts[
            env, c_off + CONTACT_IDX_FRICTION_ROLL
        ] = contact_friction_roll
        contacts[env, c_off + CONTACT_IDX_CONDIM] = Scalar[DTYPE](
            contact_condim
        )
        num_contacts += 1


@always_inline
def exclude_signatures[
    DTYPE: DType,
    EX_CAP: Int,
    L_MMETA: Layout,
    L_EXCLUDES: Layout,
](
    nbody: Int,
    nexclude_cap: Int,
    mmeta: LayoutTensor[DTYPE, L_MMETA, MutAnyOrigin],
    excludes: LayoutTensor[DTYPE, L_EXCLUDES, MutAnyOrigin],
    mut ex_sig: Scratch[Int, EX_CAP],
) -> Int:
    """The `<contact><exclude>` table as SORTED integer signatures
    `min(b1, b2) * nbody + max(b1, b2)`, MuJoCo's `exclude_signature`. Returns
    the count. Built once per detection call so `pair_body_filtered` can
    binary-search it instead of re-reading and re-converting every row of the
    float table per candidate pair — dog has 30 excludes and ~400 candidate
    pairs a step, 24k reads (PERFORMANCE.md §13.26). Insertion sort: the
    table is small, and the count is bounded by the model's budget."""
    var n_ex = Int(rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_NEXCLUDE]))
    if n_ex > nexclude_cap:
        n_ex = nexclude_cap
    for ex in range(n_ex):
        var b1 = Int(rebind[Scalar[DTYPE]](excludes[ex, 0]))
        var b2 = Int(rebind[Scalar[DTYPE]](excludes[ex, 1]))
        var lo = b1 if b1 <= b2 else b2
        var hi = b2 if b1 <= b2 else b1
        var sig = lo * nbody + hi
        var k = ex
        while k > 0 and ex_sig[k - 1] > sig:
            ex_sig[k] = ex_sig[k - 1]
            k -= 1
        ex_sig[k] = sig
    return n_ex


@always_inline
def pair_body_filtered[
    DTYPE: DType,
    L_BODIES: Layout,
    L_MMETA: Layout,
    L_EXCLUDES: Layout,
    EX_CAP: Int = 1,
](
    gi_body: Int,
    gj_body: Int,
    bodies: LayoutTensor[
        DTYPE, L_BODIES, MutAnyOrigin
    ],
    mmeta: LayoutTensor[
        DTYPE, L_MMETA, MutAnyOrigin
    ],
    excludes: LayoutTensor[
        DTYPE, L_EXCLUDES, MutAnyOrigin
    ],
    # The sorted signature table from `exclude_signatures`, optional: with
    # `n_sig >= 0` the exclude test is a binary search on it; otherwise the
    # float table is scanned as before. Same answer either way.
    ex_sig: Scratch[Int, EX_CAP] = Scratch[Int, EX_CAP](1, fill=0),
    n_sig: Int = -1,
    nbody_sig: Int = 0,
) -> Bool:
    """MuJoCo's body-pair filter. True = DISCARD the pair.

    Port of `filterBodyPair` (`engine_collision_driver.c:160`) plus the
    `exclude_signature` scan at `:403`. Shared by all three pair loops — the
    O(N^2) loop, the SAP sweep loop and the SAP PLANE loop — because keeping
    three copies is exactly how they drifted apart:

    ⚠ **The SAP plane loop had NO body filter at all** (defect 24). It applied
    only `gj_body == 0` and the contype/conaffinity mask, so every geom on a
    world-WELDED body collided with the ground. Measured on sawyer: `tablelink`
    is jointless, so `body_weldid == 0` like the world's, and its collision box
    sits 7 mm through the floor by construction — MuJoCo emits nothing, the
    O(N^2) path emitted nothing, SAP emitted FOUR corner contacts per env.
    MuJoCo filters the plane path the same way everything else is filtered, via
    `filterBodyPair(0, 0, 1, weld2, parent_weld2, ...)` at `:1277`.

    ⚠ **The exclude scan used to be NESTED inside the weld-parent branch**
    (defect 25) on both of the loops that had one, i.e. `if weld_i != 0 and
    weld_j != 0:`. MuJoCo applies exclusion per body pair UNCONDITIONALLY, so
    any `<contact><exclude>` naming a world-welded (static) body was silently
    ignored. Latent — `nexclude` is 0 in every model shipped today — but
    hand-written Menagerie/ToddlerBot XML is exactly where it would bite.

    `asleep` is not ported: we have no sleeping bodies, so MuJoCo's two asleep
    clauses are unreachable and deliberately omitted rather than stubbed.
    """
    var weld_i = Int(rebind[Scalar[DTYPE]](bodies[gi_body, BODY_IDX_WELDID]))
    var weld_j = Int(rebind[Scalar[DTYPE]](bodies[gj_body, BODY_IDX_WELDID]))

    # Same weld body — this is the clause the plane loop was missing.
    if weld_i == weld_j:
        return True

    # Both dof-less (3.12, engine_collision_driver.c:296-300): no force can
    # act between two static/mocap weld roots, so the pair is skipped. This
    # landed with the mocap weld-root rule (AUD-35): a mocap body is now its
    # own weld root, and without this clause it would collide with static
    # geometry it used to share weld id 0 with.
    var dof_i = Int(rebind[Scalar[DTYPE]](bodies[weld_i, BODY_IDX_DOFNUM]))
    var dof_j = Int(rebind[Scalar[DTYPE]](bodies[weld_j, BODY_IDX_DOFNUM]))
    if dof_i == 0 and dof_j == 0:
        return True

    # Weld-parent. Guarded on both being non-world, as MuJoCo guards it: a
    # non-world weld body always has a parent >= 0, so `bodies[wp, ...]` is in
    # range only under this test.
    #
    # ⚠ AND ON `<flag filterparent="disable">` (AUD-34), WHICH GATES THIS
    # CLAUSE AND ONLY THIS ONE. MuJoCo's whole test is
    # `(!dsbl_filterparent && weldbody1 != 0 && weldbody2 != 0) && (…)`
    # (engine_collision_driver.c:311-314): same-weldbody, both-dof-less and
    # the `<exclude>` scan below stay unconditional. A model that disables it
    # WANTS a body to collide with the one it hangs from — a gripper pad
    # against its own hand — and reading the flag as "no filtering at all"
    # would instead let every geom collide with its own body's other geoms,
    # which MuJoCo never does.
    var filterparent_off = rebind[Scalar[DTYPE]](
        mmeta[MODEL_META_IDX_FILTERPARENT_DISABLED]
    ) != Scalar[DTYPE](0)
    if not filterparent_off and weld_i != 0 and weld_j != 0:
        var wp_i = Int(rebind[Scalar[DTYPE]](bodies[weld_i, BODY_IDX_PARENT]))
        var wp_j = Int(rebind[Scalar[DTYPE]](bodies[weld_j, BODY_IDX_PARENT]))
        var weld_parent_i = Int(
            rebind[Scalar[DTYPE]](bodies[wp_i, BODY_IDX_WELDID])
        )
        var weld_parent_j = Int(
            rebind[Scalar[DTYPE]](bodies[wp_j, BODY_IDX_WELDID])
        )
        if weld_i == weld_parent_j or weld_j == weld_parent_i:
            return True

    # ⚠ DECODING THESE PER PAIR IS NOT WORTH HOISTING, WHICH IS A MEASUREMENT
    # AND NOT AN ASSUMPTION. Every field above is static model data and the SAP
    # sweep re-reads it for ~465 pairs per step against ~33 geoms, so hoisting
    # the decode to once per geom looks like free money. Built and measured
    # interleaved over 5 rounds on SO-ARM100: 15.60 s -> 15.72 s, i.e. nothing,
    # and reverted. The reason is in the ablation: stubbing the geom-geom
    # narrow phase leaves the ENTIRE sweep — 487 iterations, 466 AABB tests, 65
    # filter and mix evaluations — at 0.91 µs/step. There is no time here to
    # win. See `physics3d/PERFORMANCE.md` §5.

    # Body-pair exclusion, at the SAME level as the weld tests — not nested.
    if n_sig >= 0:
        if n_sig == 0:
            return False
        var ba = gi_body if gi_body <= gj_body else gj_body
        var bb = gj_body if gi_body <= gj_body else gi_body
        var sig = ba * nbody_sig + bb
        var lo = 0
        var hi = n_sig - 1
        while lo <= hi:
            var mid = (lo + hi) // 2
            var v = ex_sig[mid]
            if v == sig:
                return True
            elif v < sig:
                lo = mid + 1
            else:
                hi = mid - 1
        return False

    var n_ex = Int(rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_NEXCLUDE]))
    if n_ex > 0:
        var ba = gi_body if gi_body <= gj_body else gj_body
        var bb = gj_body if gi_body <= gj_body else gi_body
        for ex in range(n_ex):
            var eb1 = Int(rebind[Scalar[DTYPE]](excludes[ex, 0]))
            var eb2 = Int(rebind[Scalar[DTYPE]](excludes[ex, 1]))
            if eb1 == ba and eb2 == bb:
                return True

    return False


@always_inline
def find_predefined_pair[
    DTYPE: DType,
    D: DimsLike,
    L_PAIRS: Layout,
    L_MMETA: Layout](
    gi: Int,
    gj: Int,
    dims: D,
    pairs: LayoutTensor[
        DTYPE, L_PAIRS, MutAnyOrigin
    ],
    mmeta: LayoutTensor[
        DTYPE, L_MMETA, MutAnyOrigin
    ],
) -> Int:
    """Index of the `<contact><pair>` covering geoms (gi, gj), or -1.

    MuJoCo's `ipair`. Every filter in every detection loop is conditioned on
    this being negative, and every contact parameter is taken from the pair
    record when it is not — see `mj_collideGeoms`, where `ipair >= 0` skips the
    contact-mask test outright and reads condim/friction/solref/solimp/margin
    from `m->pair_*` instead of mixing the two geoms'.

    Compares BOTH orders. `_fill_pairs` stores the record sorted (geom1 <
    geom2, as MuJoCo's compiler sorts it) and the O(N^2) loop iterates
    `gi < gj`, so a single-order test would work there — but the SAP sweep
    visits geoms in AABB order, and the SAP plane loop always passes the plane
    as `gi` whatever its index. A pair naming the plane second would then be
    found by one path and missed by the other, which is precisely the
    two-path drift this file has been bitten by before.

    ⚠ THE SCAN IS OVER ALL PAIRS, not just those merged for one body pair as
    in MuJoCo's driver. Same outcome — a predefined pair's geoms determine its
    body signature, so any pair matching (gi, gj) is necessarily in the range
    MuJoCo would have merged — and it drops the signature bookkeeping the
    interleaved merge loop needs. `npair` is a handful of records.
    """
    var npair = dims.get_npair()
    # Clamped to the COMPTIME capacity, not trusted from the meta slot.
    # `build_model_fields_from_flat` raises when the two disagree, so this
    # cannot silently truncate — it is here so that a Model built by some
    # other path can never walk off the end of a `[npair, ...]` tensor.
    var n_pair = Int(rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_NPAIR]))
    if n_pair > npair:
        n_pair = npair
    for p in range(n_pair):
        var p1 = Int(rebind[Scalar[DTYPE]](pairs[p, PAIR_IDX_GEOM1]))
        var p2 = Int(rebind[Scalar[DTYPE]](pairs[p, PAIR_IDX_GEOM2]))
        if (p1 == gi and p2 == gj) or (p1 == gj and p2 == gi):
            return p
    return -1


@always_inline
def pair_params[
    DTYPE: DType,
    L_PAIRS: Layout](
    ipair: Int,
    pairs: LayoutTensor[
        DTYPE, L_PAIRS, MutAnyOrigin
    ],
) -> Array[Scalar[DTYPE], 12]:
    """A predefined pair's parameters, in `mix_contact_params`' layout.

    Returned in the same 12-slot shape the mixing helper produces so the two
    are interchangeable at every call site: `[condim, friction, friction_spin,
    friction_roll, solref0, solref1, solimp0..solimp4]`.

    These are the pair's OWN values, never mixed with the geoms'. An omitted
    XML attribute has already been defaulted by `_fill_pairs` to MuJoCo's
    global default — NOT to anything derived from geom1/geom2. See
    `MODEL_PAIR_SIZE` for the measurement showing the derivation in
    `mjCPair::Compile` is unreachable from XML.
    """
    var out = Array[Scalar[DTYPE], 12](fill=Scalar[DTYPE](0))
    out[0] = rebind[Scalar[DTYPE]](pairs[ipair, PAIR_IDX_CONDIM])
    out[1] = rebind[Scalar[DTYPE]](pairs[ipair, PAIR_IDX_FRICTION])
    out[2] = rebind[Scalar[DTYPE]](pairs[ipair, PAIR_IDX_FRICTION_SPIN])
    out[3] = rebind[Scalar[DTYPE]](pairs[ipair, PAIR_IDX_FRICTION_ROLL])
    out[4] = rebind[Scalar[DTYPE]](pairs[ipair, PAIR_IDX_SOLREF_0])
    out[5] = rebind[Scalar[DTYPE]](pairs[ipair, PAIR_IDX_SOLREF_1])
    out[6] = rebind[Scalar[DTYPE]](pairs[ipair, PAIR_IDX_SOLIMP_0])
    out[7] = rebind[Scalar[DTYPE]](pairs[ipair, PAIR_IDX_SOLIMP_1])
    out[8] = rebind[Scalar[DTYPE]](pairs[ipair, PAIR_IDX_SOLIMP_2])
    out[9] = rebind[Scalar[DTYPE]](pairs[ipair, PAIR_IDX_SOLIMP_3])
    out[10] = rebind[Scalar[DTYPE]](pairs[ipair, PAIR_IDX_SOLIMP_4])
    return out^


@always_inline
def mix_contact_params[
    DTYPE: DType
](
    prio_i: Int,
    condim_i: Int,
    fri_i: Scalar[DTYPE],
    fsp_i: Scalar[DTYPE],
    frl_i: Scalar[DTYPE],
    sr0_i: Scalar[DTYPE],
    sr1_i: Scalar[DTYPE],
    si0_i: Scalar[DTYPE],
    si1_i: Scalar[DTYPE],
    si2_i: Scalar[DTYPE],
    si3_i: Scalar[DTYPE],
    si4_i: Scalar[DTYPE],
    prio_j: Int,
    condim_j: Int,
    fri_j: Scalar[DTYPE],
    fsp_j: Scalar[DTYPE],
    frl_j: Scalar[DTYPE],
    sr0_j: Scalar[DTYPE],
    sr1_j: Scalar[DTYPE],
    si0_j: Scalar[DTYPE],
    si1_j: Scalar[DTYPE],
    si2_j: Scalar[DTYPE],
    si3_j: Scalar[DTYPE],
    si4_j: Scalar[DTYPE],
) -> Array[Scalar[DTYPE], 12]:
    """MuJoCo's contact-parameter mixing. Port of
    `engine_collision_driver.c:1426-1480`.

    Returns `[condim, friction, friction_spin, friction_roll, solref0, solref1,
    solimp0..solimp4]` — index 0 holds condim as a float.

    THE RULE, and every branch of it matters for dm_control's quadruped:

      * **Priorities DIFFER** -> the higher-priority geom supplies condim,
        solref, solimp AND friction, wholesale. No mixing of any kind. This is
        how quadruped's ball (`priority="1"`) forces its own `condim="6"` and
        `solref="-10000 -30"` onto every contact it takes part in, including
        against a floor whose parameters are entirely different.
      * **Priorities EQUAL**:
          - condim   -> max
          - friction -> elementwise max
          - solref   -> elementwise MEAN if BOTH `solref[0] > 0`, otherwise
            elementwise MIN. That second branch is why a DIRECT (negative)
            solref wins over a standard one even at equal priority: it is not
            averaged, it is taken.
          - solimp   -> elementwise mean, with no direct branch at all.

    ⚠ THE SOLREF TEST IS ON COMPONENT [0] OF **BOTH** GEOMS
    (`solref1[0] > 0 && solref2[0] > 0`), not "either is negative". Same
    outcome for the sign combinations MuJoCo's compiler permits, but the
    condition is the source's.

    ⚠ THE MEAN IS A SPECIAL CASE OF A `solmix` WEIGHTING —
    `mix = solmix1/(solmix1 + solmix2)` — which is 0.5 only because every geom
    defaults to `solmix = 1`. `full_parser` REJECTS a non-default `solmix`
    rather than letting it silently degrade to this mean. No suite model sets
    one; a five-point probe against MuJoCo could not have revealed this, and
    did not — the source did.

    ⚠ Until 2026-08-03 the narrow phase applied the equal-priority max rule to
    friction and condim UNCONDITIONALLY and never looked at solref/solimp at
    all, so `priority` was ignored and per-geom solparams were dead data.
    """
    var out = Array[Scalar[DTYPE], 12](fill=Scalar[DTYPE](0))

    if prio_i != prio_j:
        var hi_i = prio_i > prio_j
        out[0] = Scalar[DTYPE](condim_i if hi_i else condim_j)
        out[1] = fri_i if hi_i else fri_j
        out[2] = fsp_i if hi_i else fsp_j
        out[3] = frl_i if hi_i else frl_j
        out[4] = sr0_i if hi_i else sr0_j
        out[5] = sr1_i if hi_i else sr1_j
        out[6] = si0_i if hi_i else si0_j
        out[7] = si1_i if hi_i else si1_j
        out[8] = si2_i if hi_i else si2_j
        out[9] = si3_i if hi_i else si3_j
        out[10] = si4_i if hi_i else si4_j
        return out^

    out[0] = Scalar[DTYPE](condim_i if condim_i > condim_j else condim_j)
    out[1] = fri_i if fri_i > fri_j else fri_j
    out[2] = fsp_i if fsp_i > fsp_j else fsp_j
    out[3] = frl_i if frl_i > frl_j else frl_j

    comptime HALF = Scalar[DTYPE](0.5)
    if sr0_i > Scalar[DTYPE](0) and sr0_j > Scalar[DTYPE](0):
        out[4] = HALF * (sr0_i + sr0_j)
        out[5] = HALF * (sr1_i + sr1_j)
    else:
        out[4] = sr0_i if sr0_i < sr0_j else sr0_j
        out[5] = sr1_i if sr1_i < sr1_j else sr1_j

    out[6] = HALF * (si0_i + si0_j)
    out[7] = HALF * (si1_i + si1_j)
    out[8] = HALF * (si2_i + si2_j)
    out[9] = HALF * (si3_i + si3_j)
    out[10] = HALF * (si4_i + si4_j)
    return out^


def _plane_cylinder_contacts[
    DTYPE: DType,
    BATCH: Int,
    D: DimsLike,
    L_CONTACTS: Layout,
](
    env: Int,
    g_body: Int,
    p_x: Scalar[DTYPE],
    p_y: Scalar[DTYPE],
    p_z: Scalar[DTYPE],
    q_x: Scalar[DTYPE],
    q_y: Scalar[DTYPE],
    q_z: Scalar[DTYPE],
    q_w: Scalar[DTYPE],
    radius: Scalar[DTYPE],
    half_length: Scalar[DTYPE],
    ground_z: Scalar[DTYPE],
    plp_x: Scalar[DTYPE],
    plp_y: Scalar[DTYPE],
    plp_z: Scalar[DTYPE],
    plq_x: Scalar[DTYPE],
    plq_y: Scalar[DTYPE],
    plq_z: Scalar[DTYPE],
    plq_w: Scalar[DTYPE],
    contact_margin: Scalar[DTYPE],
    contact_friction: Scalar[DTYPE],
    contact_friction_spin: Scalar[DTYPE],
    contact_friction_roll: Scalar[DTYPE],
    contact_condim: Int,
    world_body: Int,
    dims: D,
    contacts: LayoutTensor[
        DTYPE, L_CONTACTS,
        MutAnyOrigin,
    ],
    mut num_contacts: Int,
    # ⚠ THREADED THROUGH, NOT RECOMPUTED. This function does not see the
    # geom pair; it forwards the cutoff/includemargin split to the emit
    # helper below. Defaulted to 0 so every existing call site is
    # unchanged. See `GEOM_IDX_GAP`.
    contact_gap: Scalar[DTYPE] = Scalar[DTYPE](0),
    # The emission cap. -1 = the model's `max_contacts` (every serial
    # caller); the block-per-env collision kernel passes the end of the
    # thread's staging window instead, so a routine cannot write past it.
    max_contacts_in: Int = -1,
):
    """Plane-cylinder: up to FOUR points — two rim, two triangle.

    Port of `mjc_PlaneCylinder` (`engine_collision_primitive.c`). We emitted ONE
    point per pair until 2026-08-03, so a cylinder standing on its flat face had
    no restoring torque and tipped — the same defect class as bug 39 (box/plane
    and box/box), and it survived for the same reason: no test compared a
    plane's contact SET against MuJoCo.

    ⚠ THE ROUTINE IS BEHAVIOURALLY IDENTICAL IN 3.6.0 AND 3.3.6 — checked,
    because `mjc_BoxBox` is NOT (3.3.6 halves the face-path depth), and a
    faithful transcription from the wrong tree there would have been a silent
    factor of two. Here the two differ only in `mji_*` vs `mju_*` inlining and
    whitespace.

    THE POINTS, in MuJoCo's order, which is part of the answer:
      1. the deepest rim point on the NEAR cap — `+vec +axis`. If this one is
         above `margin` the routine returns ZERO contacts outright, without
         testing the others.
      2. the same rim direction on the FAR cap — `+vec -axis`.
      3-4. two triangle points at `±vec1` on the near cap, where
         `vec1 = normalize(cross(vec, axis)) * radius*sqrt(3)/2`, offset by
         `-vec*0.5`. Together with point 1 they are the inscribed triangle that
         gives a flat-resting cylinder its support polygon.

    `p_*` / `q_*` are the cylinder's pose IN THE PLANE'S FRAME, where the plane
    is z = `ground_z` facing +z — so the world normal is (0,0,1) here and
    `dist0` is just `p_z - ground_z`. `plp_*` / `plq_*` put the point and the
    normal back into world (`collision/plane_frame.mojo`).

    ⚠ `axis` IS ALREADY SCALED BY `half_length` WHERE THE CROSS PRODUCT USES
    IT. That is MuJoCo's order (`mju_scl3(axis, axis, size2[1])` runs before the
    first point is built), and it is harmless only because `vec1` is normalized
    straight after — but transcribing it in the other order would change
    nothing until someone removed the normalize.

    Verified before this was written: the routine was transcribed to Python and
    swept against the MuJoCo runtime over 400 random poses — 272 contacting,
    557 points, 0 count mismatches, dist 2.1e-17, position 5.6e-17.
    """
    var max_contacts = (
        max_contacts_in if max_contacts_in >= 0 else dims.get_max_contacts()
    )
    var pn = plane_world_normal[DTYPE](plq_x, plq_y, plq_z, plq_w)
    comptime MINVAL = Scalar[DTYPE](1e-15)

    # Cylinder axis, flipped so it points TOWARDS the plane.
    var ax0 = gpu_quat_rotate(
        q_x, q_y, q_z, q_w,
        Scalar[DTYPE](0), Scalar[DTYPE](0), Scalar[DTYPE](1),
    )
    var ax = ax0[0]
    var ay = ax0[1]
    var az = ax0[2]
    var prjaxis = az  # dot((0,0,1), axis) in the plane frame
    if prjaxis > Scalar[DTYPE](0):
        ax = -ax
        ay = -ay
        az = -az
        prjaxis = -prjaxis

    var dist0 = p_z - ground_z

    # vec = axis*prjaxis - normal, then rescaled to the cylinder radius. This
    # is the radial direction most steeply into the plane.
    var vx = ax * prjaxis
    var vy = ay * prjaxis
    var vz = az * prjaxis - Scalar[DTYPE](1)
    var len_sqr = vx * vx + vy * vy + vz * vz
    if len_sqr >= MINVAL * MINVAL:
        var scl = radius / sqrt(len_sqr)
        vx = vx * scl
        vy = vy * scl
        vz = vz * scl
    else:
        # Disk parallel to the plane: the radial direction is undefined, so
        # MuJoCo picks the cylinder's own x-axis. This is the branch a
        # flat-resting cylinder takes, i.e. the common case, not a corner one.
        var xa = gpu_quat_rotate(
            q_x, q_y, q_z, q_w,
            Scalar[DTYPE](1), Scalar[DTYPE](0), Scalar[DTYPE](0),
        )
        vx = xa[0] * radius
        vy = xa[1] * radius
        vz = xa[2] * radius

    var prjvec = vz
    ax = ax * half_length
    ay = ay * half_length
    az = az * half_length
    prjaxis = prjaxis * half_length

    # Point 1 — near-cap rim. Its rejection ends the routine.
    var d1 = dist0 + prjaxis + prjvec
    if d1 > contact_margin or num_contacts >= max_contacts:
        return
    _emit_plane_contact[DTYPE](
        env, g_body, p_x + vx + ax, p_y + vy + ay,
        p_z + vz + az - d1 * Scalar[DTYPE](0.5), d1,
        plp_x, plp_y, plp_z, plq_x, plq_y, plq_z, plq_w, pn,
        contact_margin, contact_friction, contact_friction_spin,
        contact_friction_roll, contact_condim, world_body,
        contacts, num_contacts,
        contact_gap,
    )

    # Point 2 — far-cap rim, same radial direction.
    var d2 = dist0 - prjaxis + prjvec
    if d2 <= contact_margin and num_contacts < max_contacts:
        _emit_plane_contact[DTYPE](
            env, g_body, p_x + vx - ax, p_y + vy - ay,
            p_z + vz - az - d2 * Scalar[DTYPE](0.5), d2,
            plp_x, plp_y, plp_z, plq_x, plq_y, plq_z, plq_w, pn,
            contact_margin, contact_friction, contact_friction_spin,
            contact_friction_roll, contact_condim, world_body,
            contacts, num_contacts,
            contact_gap,
        )

    # Points 3 and 4 — the triangle on the near cap.
    var prjvec1 = -prjvec * Scalar[DTYPE](0.5)
    var d3 = dist0 + prjaxis + prjvec1
    if d3 <= contact_margin:
        var w1x = vy * az - vz * ay
        var w1y = vz * ax - vx * az
        var w1z = vx * ay - vy * ax
        var wl = sqrt(w1x * w1x + w1y * w1y + w1z * w1z)
        if wl > MINVAL:
            w1x = w1x / wl
            w1y = w1y / wl
            w1z = w1z / wl
        else:
            # `mju_normalize3` rewrites a zero vector as (1,0,0); unreachable
            # here because `vec` is perpendicular to `axis` in both branches
            # above, but transcribed so the degenerate case cannot silently
            # become a zero-length offset.
            w1x = Scalar[DTYPE](1)
            w1y = Scalar[DTYPE](0)
            w1z = Scalar[DTYPE](0)
        var s3 = radius * sqrt(Scalar[DTYPE](3.0)) / Scalar[DTYPE](2)
        w1x = w1x * s3
        w1y = w1y * s3
        w1z = w1z * s3

        var bx = ax - vx * Scalar[DTYPE](0.5)
        var by = ay - vy * Scalar[DTYPE](0.5)
        var bz = az - vz * Scalar[DTYPE](0.5) - d3 * Scalar[DTYPE](0.5)
        if num_contacts < max_contacts:
            _emit_plane_contact[DTYPE](
                env, g_body, p_x + w1x + bx, p_y + w1y + by, p_z + w1z + bz,
                d3, plp_x, plp_y, plp_z, plq_x, plq_y, plq_z, plq_w, pn,
                contact_margin, contact_friction, contact_friction_spin,
                contact_friction_roll, contact_condim, world_body,
                contacts, num_contacts,
                contact_gap,
            )
        if num_contacts < max_contacts:
            _emit_plane_contact[DTYPE](
                env, g_body, p_x - w1x + bx, p_y - w1y + by, p_z - w1z + bz,
                d3, plp_x, plp_y, plp_z, plq_x, plq_y, plq_z, plq_w, pn,
                contact_margin, contact_friction, contact_friction_spin,
                contact_friction_roll, contact_condim, world_body,
                contacts, num_contacts,
                contact_gap,
            )


def _emit_plane_contact[
    DTYPE: DType,
    L_CONTACTS: Layout,
](
    env: Int,
    g_body: Int,
    lx: Scalar[DTYPE],
    ly: Scalar[DTYPE],
    lz: Scalar[DTYPE],
    dist: Scalar[DTYPE],
    plp_x: Scalar[DTYPE],
    plp_y: Scalar[DTYPE],
    plp_z: Scalar[DTYPE],
    plq_x: Scalar[DTYPE],
    plq_y: Scalar[DTYPE],
    plq_z: Scalar[DTYPE],
    plq_w: Scalar[DTYPE],
    pn: Array[Scalar[DTYPE], 3],
    contact_margin: Scalar[DTYPE],
    contact_friction: Scalar[DTYPE],
    contact_friction_spin: Scalar[DTYPE],
    contact_friction_roll: Scalar[DTYPE],
    contact_condim: Int,
    world_body: Int,
    contacts: LayoutTensor[
        DTYPE, L_CONTACTS,
        MutAnyOrigin,
    ],
    mut num_contacts: Int,
    # ⚠ THE GAP HALF OF THE PAIR'S MARGIN, DEFAULTED TO 0 SO EVERY
    # EXISTING CALL SITE IS UNCHANGED. `contact_margin` is the narrowphase
    # CUTOFF (`margin + gap`); what a contact STORES as its
    # `includemargin` is `contact_margin - contact_gap`, and the solver excludes
    # `dist >= includemargin`. See `GEOM_IDX_GAP`.
    contact_gap: Scalar[DTYPE] = Scalar[DTYPE](0),
):
    """Write one plane contact whose point is given IN THE PLANE FRAME.

    Factored out of `_plane_cylinder_contacts`, which emits four points that
    differ only in position — repeating the twelve-field write four times is how
    a slot gets missed in one copy. ⚠ `world_body` is passed rather than
    hardcoded for the same reason as in `_plane_box_contacts`: the naive path
    writes 0 and the SAP path -1.
    """
    var cw = from_plane_frame[DTYPE](
        plp_x, plp_y, plp_z, plq_x, plq_y, plq_z, plq_w, lx, ly, lz
    )
    var c_off = num_contacts * CONTACT_SIZE
    contacts[env, c_off + CONTACT_IDX_BODY_A] = Scalar[DTYPE](g_body)
    contacts[env, c_off + CONTACT_IDX_BODY_B] = Scalar[DTYPE](world_body)
    contacts[env, c_off + CONTACT_IDX_POS_X] = cw[0]
    contacts[env, c_off + CONTACT_IDX_POS_Y] = cw[1]
    contacts[env, c_off + CONTACT_IDX_POS_Z] = cw[2]
    contacts[env, c_off + CONTACT_IDX_NX] = pn[0]
    contacts[env, c_off + CONTACT_IDX_NY] = pn[1]
    contacts[env, c_off + CONTACT_IDX_NZ] = pn[2]
    contacts[env, c_off + CONTACT_IDX_DIST] = dist
    contacts[
            env, c_off + CONTACT_IDX_INCLUDEMARGIN
        ] = contact_margin - contact_gap
    contacts[env, c_off + CONTACT_IDX_FRICTION] = contact_friction
    contacts[env, c_off + CONTACT_IDX_FRICTION_SPIN] = contact_friction_spin
    contacts[env, c_off + CONTACT_IDX_FRICTION_ROLL] = contact_friction_roll
    contacts[env, c_off + CONTACT_IDX_CONDIM] = Scalar[DTYPE](contact_condim)
    num_contacts += 1


def _plane_box_contacts[
    DTYPE: DType,
    D: DimsLike,
    L_CONTACTS: Layout,
](
    env: Int,
    g_body: Int,
    p_x: Scalar[DTYPE],
    p_y: Scalar[DTYPE],
    p_z: Scalar[DTYPE],
    q_x: Scalar[DTYPE],
    q_y: Scalar[DTYPE],
    q_z: Scalar[DTYPE],
    q_w: Scalar[DTYPE],
    hx: Scalar[DTYPE],
    hy: Scalar[DTYPE],
    hz: Scalar[DTYPE],
    ground_z: Scalar[DTYPE],
    plp_x: Scalar[DTYPE],
    plp_y: Scalar[DTYPE],
    plp_z: Scalar[DTYPE],
    plq_x: Scalar[DTYPE],
    plq_y: Scalar[DTYPE],
    plq_z: Scalar[DTYPE],
    plq_w: Scalar[DTYPE],
    contact_margin: Scalar[DTYPE],
    contact_friction: Scalar[DTYPE],
    contact_friction_spin: Scalar[DTYPE],
    contact_friction_roll: Scalar[DTYPE],
    contact_condim: Int,
    world_body: Int,
    dims: D,
    contacts: LayoutTensor[
        DTYPE, L_CONTACTS,
        MutAnyOrigin,
    ],
    mut num_contacts: Int,
    # ⚠ THE GAP HALF OF THE PAIR'S MARGIN, DEFAULTED TO 0 SO EVERY
    # EXISTING CALL SITE IS UNCHANGED. `contact_margin` is the narrowphase
    # CUTOFF (`margin + gap`); what a contact STORES as its
    # `includemargin` is `contact_margin - contact_gap`, and the solver excludes
    # `dist >= includemargin`. See `GEOM_IDX_GAP`.
    contact_gap: Scalar[DTYPE] = Scalar[DTYPE](0),
    # The emission cap. -1 = the model's `max_contacts` (every serial
    # caller); the block-per-env collision kernel passes the end of the
    # thread's staging window instead, so a routine cannot write past it.
    max_contacts_in: Int = -1,
):
    """Plane-box: one contact per box CORNER below the plane, up to four.

    Port of `mjc_PlaneBox` (`engine_collision_primitive.c`). A box resting flat
    on a plane touches it over a whole face, and a single point cannot express
    that: a rigid body supported at ONE point has no restoring torque about it,
    so a cube on a floor pivots instead of resting. That is task #42, and this
    is its box/plane half.

    `p_*` / `q_*` are the box's pose IN THE PLANE'S FRAME, where the plane is
    z = `ground_z` facing +z, so a corner's height above the plane is just its
    z. `plp_*` / `plq_*` are the plane's world pose, used only to put the
    contact point and normal back into world — see `collision/plane_frame.mojo`.

    ⚠ TWO FILTERS, BOTH MuJoCo's, AND THE SECOND IS NOT REDUNDANT. A corner is
    skipped when it is further than `margin` above the plane (obviously) AND
    when its offset from the box CENTRE points along +normal (`ldist > 0`),
    which drops the box's upper four corners even when a deeply sunk box has all
    eight below the plane. Without it a fully submerged box would emit four
    contacts on its TOP face pushing the wrong way.

    ⚠ THE ITERATION ORDER IS PART OF THE ANSWER when more than four corners
    qualify: MuJoCo keeps the first four in `i = 0..7` with x = i&1, y = i&2,
    z = i&4, so this loop matches that order rather than sorting by depth.
    """
    var max_contacts = (
        max_contacts_in if max_contacts_in >= 0 else dims.get_max_contacts()
    )
    var pn = plane_world_normal[DTYPE](plq_x, plq_y, plq_z, plq_w)
    var cnt = 0
    for i in range(8):
        if num_contacts >= max_contacts or cnt >= 4:
            break
        var vx = hx if (i & 1) != 0 else -hx
        var vy = hy if (i & 2) != 0 else -hy
        var vz = hz if (i & 4) != 0 else -hz
        var rel = gpu_quat_rotate(q_x, q_y, q_z, q_w, vx, vy, vz)
        # `ldist` is the corner offset along the plane normal, which in this
        # frame is simply its z component.
        var ldist = rel[2]
        var cdist = (p_z + rel[2]) - ground_z
        if cdist > contact_margin or ldist > Scalar[DTYPE](0):
            continue

        var cw = from_plane_frame[DTYPE](
            plp_x, plp_y, plp_z, plq_x, plq_y, plq_z, plq_w,
            p_x + rel[0],
            p_y + rel[1],
            (p_z + rel[2]) - cdist * Scalar[DTYPE](0.5),
        )
        var c_off = num_contacts * CONTACT_SIZE
        contacts[env, c_off + CONTACT_IDX_BODY_A] = Scalar[DTYPE](g_body)
        # ⚠ THE TWO PATHS DISAGREE ON THE WORLD BODY ID — `detect_contacts`
        # writes 0 and the SAP broadphase writes -1 — so it is passed in rather
        # than hardcoded here, which would silently change one of them.
        contacts[env, c_off + CONTACT_IDX_BODY_B] = Scalar[DTYPE](world_body)
        contacts[env, c_off + CONTACT_IDX_POS_X] = cw[0]
        contacts[env, c_off + CONTACT_IDX_POS_Y] = cw[1]
        contacts[env, c_off + CONTACT_IDX_POS_Z] = cw[2]
        contacts[env, c_off + CONTACT_IDX_NX] = pn[0]
        contacts[env, c_off + CONTACT_IDX_NY] = pn[1]
        contacts[env, c_off + CONTACT_IDX_NZ] = pn[2]
        contacts[env, c_off + CONTACT_IDX_DIST] = cdist
        contacts[
            env, c_off + CONTACT_IDX_INCLUDEMARGIN
        ] = contact_margin - contact_gap
        contacts[env, c_off + CONTACT_IDX_FRICTION] = contact_friction
        contacts[
            env, c_off + CONTACT_IDX_FRICTION_SPIN
        ] = contact_friction_spin
        contacts[
            env, c_off + CONTACT_IDX_FRICTION_ROLL
        ] = contact_friction_roll
        contacts[env, c_off + CONTACT_IDX_CONDIM] = Scalar[DTYPE](
            contact_condim
        )
        num_contacts += 1
        cnt += 1


@always_inline
def _capsule_capsule_contacts[
    DTYPE: DType,
    D: DimsLike,
    L_CONTACTS: Layout,
](
    env: Int,
    body_a: Int,
    body_b: Int,
    ai_x: Scalar[DTYPE], ai_y: Scalar[DTYPE], ai_z: Scalar[DTYPE],
    ai_qx: Scalar[DTYPE], ai_qy: Scalar[DTYPE], ai_qz: Scalar[DTYPE],
    ai_qw: Scalar[DTYPE],
    ai_hl: Scalar[DTYPE], ai_r: Scalar[DTYPE],
    bj_x: Scalar[DTYPE], bj_y: Scalar[DTYPE], bj_z: Scalar[DTYPE],
    bj_qx: Scalar[DTYPE], bj_qy: Scalar[DTYPE], bj_qz: Scalar[DTYPE],
    bj_qw: Scalar[DTYPE],
    bj_hl: Scalar[DTYPE], bj_r: Scalar[DTYPE],
    contact_margin: Scalar[DTYPE],
    contact_friction: Scalar[DTYPE],
    contact_friction_spin: Scalar[DTYPE],
    contact_friction_roll: Scalar[DTYPE],
    contact_condim: Int,
    dims: D,
    contacts: LayoutTensor[
        DTYPE, L_CONTACTS,
        MutAnyOrigin,
    ],
    mut num_contacts: Int,
    # ⚠ THE GAP HALF OF THE PAIR'S MARGIN, DEFAULTED TO 0 SO EVERY
    # EXISTING CALL SITE IS UNCHANGED. `contact_margin` is the narrowphase
    # CUTOFF (`margin + gap`); what a contact STORES as its
    # `includemargin` is `contact_margin - contact_gap`, and the solver excludes
    # `dist >= includemargin`. See `GEOM_IDX_GAP`.
    contact_gap: Scalar[DTYPE] = Scalar[DTYPE](0),
    # The emission cap. -1 = the model's `max_contacts` (every serial
    # caller); the block-per-env collision kernel passes the end of the
    # thread's staging window instead, so a routine cannot write past it.
    max_contacts_in: Int = -1,
) -> Int:
    """Capsule/capsule: up to TWO contacts, MuJoCo's manifold.

    Two PARALLEL capsules touch along a segment and `mjraw_CapsuleCapsule`
    emits a point at each end; a single closest-point query emits one, which
    leaves the pair free to pivot about it. See `capsule_capsule_manifold`.

    ⚠ NORMAL SIGN. The manifold's normal is capsule A -> capsule B, i.e.
    `gi -> gj`, and the record's convention is `body_b -> body_a`. So it is
    NEGATED here, which is what the shared single-point emit does
    unconditionally at the bottom of this loop.

    Returns the number of records written.
    """
    var max_contacts = (
        max_contacts_in if max_contacts_in >= 0 else dims.get_max_contacts()
    )
    var cc_dist = Array[Scalar[DTYPE], CC_MAX_POINTS](
        fill=Scalar[DTYPE](0)
    )
    var cc_pos = Array[Scalar[DTYPE], 3 * CC_MAX_POINTS](
        fill=Scalar[DTYPE](0)
    )
    var cc_n = Array[Scalar[DTYPE], 3 * CC_MAX_POINTS](
        fill=Scalar[DTYPE](0)
    )
    var n_cc = capsule_capsule_manifold[DTYPE](
        ai_x, ai_y, ai_z, ai_qx, ai_qy, ai_qz, ai_qw, ai_hl, ai_r,
        bj_x, bj_y, bj_z, bj_qx, bj_qy, bj_qz, bj_qw, bj_hl, bj_r,
        contact_margin,
        cc_dist,
        cc_pos,
        cc_n,
    )

    var written = 0
    for c in range(n_cc):
        if num_contacts >= max_contacts:
            break
        if cc_dist[c] >= contact_margin:
            continue
        var c_off = num_contacts * CONTACT_SIZE
        contacts[env, c_off + CONTACT_IDX_BODY_A] = Scalar[DTYPE](body_a)
        contacts[env, c_off + CONTACT_IDX_BODY_B] = Scalar[DTYPE](body_b)
        contacts[env, c_off + CONTACT_IDX_POS_X] = cc_pos[3 * c + 0]
        contacts[env, c_off + CONTACT_IDX_POS_Y] = cc_pos[3 * c + 1]
        contacts[env, c_off + CONTACT_IDX_POS_Z] = cc_pos[3 * c + 2]
        contacts[env, c_off + CONTACT_IDX_NX] = -cc_n[3 * c + 0]
        contacts[env, c_off + CONTACT_IDX_NY] = -cc_n[3 * c + 1]
        contacts[env, c_off + CONTACT_IDX_NZ] = -cc_n[3 * c + 2]
        contacts[env, c_off + CONTACT_IDX_DIST] = cc_dist[c]
        contacts[
            env, c_off + CONTACT_IDX_INCLUDEMARGIN
        ] = contact_margin - contact_gap
        contacts[env, c_off + CONTACT_IDX_FRICTION] = contact_friction
        contacts[
            env, c_off + CONTACT_IDX_FRICTION_SPIN
        ] = contact_friction_spin
        contacts[
            env, c_off + CONTACT_IDX_FRICTION_ROLL
        ] = contact_friction_roll
        contacts[env, c_off + CONTACT_IDX_CONDIM] = Scalar[DTYPE](
            contact_condim
        )
        num_contacts += 1
        written += 1
    return written


def _hfield_contacts[
    DTYPE: DType,
    D: DimsLike,
    L_HF_META: Layout,
    L_HF_DATA: Layout,
    L_MESH_VERTS: Layout,
    L_MESH_VERT_EDGEADR: Layout,
    L_MESH_EDGES: Layout,
    L_CONTACTS: Layout,
    L_WS: Layout,
](
    env: Int,
    body_a: Int,
    body_b: Int,
    hfield_id: Int,
    hf_x: Scalar[DTYPE], hf_y: Scalar[DTYPE], hf_z: Scalar[DTYPE],
    hf_qx: Scalar[DTYPE], hf_qy: Scalar[DTYPE], hf_qz: Scalar[DTYPE],
    hf_qw: Scalar[DTYPE],
    gj_type: Int,
    pj_x: Scalar[DTYPE], pj_y: Scalar[DTYPE], pj_z: Scalar[DTYPE],
    qj_x: Scalar[DTYPE], qj_y: Scalar[DTYPE], qj_z: Scalar[DTYPE],
    qj_w: Scalar[DTYPE],
    rj: Scalar[DTYPE], hlj: Scalar[DTYPE],
    hxj: Scalar[DTYPE], hyj: Scalar[DTYPE], hzj: Scalar[DTYPE],
    rboundj: Scalar[DTYPE],
    va2: Int, mnv2: Int,
    contact_margin: Scalar[DTYPE],
    contact_friction: Scalar[DTYPE],
    contact_friction_spin: Scalar[DTYPE],
    contact_friction_roll: Scalar[DTYPE],
    contact_condim: Int,
    # `-1` when the heightfield is geom `gi`, `+1` when it is `gj`.
    nsign: Scalar[DTYPE],
    hfield_meta: LayoutTensor[DTYPE, L_HF_META, MutAnyOrigin],
    hfield_data: LayoutTensor[DTYPE, L_HF_DATA, MutAnyOrigin],
    hf_stride: Int,
    mesh_verts: LayoutTensor[DTYPE, L_MESH_VERTS, MutAnyOrigin],
    mesh_vert_edgeadr: LayoutTensor[
        DTYPE, L_MESH_VERT_EDGEADR, MutAnyOrigin
    ],
    mesh_edges: LayoutTensor[DTYPE, L_MESH_EDGES, MutAnyOrigin],
    dims: D,
    contacts: LayoutTensor[DTYPE, L_CONTACTS, MutAnyOrigin],
    ws: LayoutTensor[DTYPE, L_WS, MutAnyOrigin],
    mut num_contacts: Int,
    # ⚠ THREADED THROUGH, NOT RECOMPUTED. This function does not see the
    # geom pair; it forwards the cutoff/includemargin split to the emit
    # helper below. Defaulted to 0 so every existing call site is
    # unchanged. See `GEOM_IDX_GAP`.
    contact_gap: Scalar[DTYPE] = Scalar[DTYPE](0),
) -> Int:
    """HEIGHTFIELD x convex — one record per prism `mjc_ConvexHField` reports.

    ⚠⚠ IT WRITES THE RECORDS ITSELF rather than returning a manifold for this
    function to copy, which is the opposite of `_capsule_capsule_contacts` and
    `_capsule_box_contacts` beside it. Those buffer at most two and four
    points; a heightfield's ceiling is `mjMAXCONPAIR` = 50, and three
    `Array`s of that size are 350 float64 of PER-THREAD stack. The Metal
    collision kernel does not have it — the first version of this file was
    written the buffering way and `test_plane_mesh_fields` failed to build with
    "Compute function exceeds available stack space".

    ⚠ NORMAL SIGN, AND IT DEPENDS ON WHICH SIDE THE HEIGHTFIELD IS. The query
    always returns `hfield -> convex`; the record always stores
    `body_b -> body_a` with `body_a = gi_body`. So when the field is `gi` the
    query is `gi -> gj` and must be NEGATED, and when it is `gj` it is already
    `gj -> gi` and must not be. `nsign` carries that, and the bodies are NOT
    swapped to compensate — that would land on the right normal and the wrong
    `body_a`, the double flip the dispatch's own comment warns about.

    ⚠ Only the first case occurs in the tree today (barkour declares its
    heightfield as geom 0), which is why the second is spelled out here.
    """
    return hfield_convex_contacts[DTYPE](
        hfield_id,
        hf_x, hf_y, hf_z, hf_qx, hf_qy, hf_qz, hf_qw,
        gj_type,
        pj_x, pj_y, pj_z, qj_x, qj_y, qj_z, qj_w,
        rj, hlj, hxj, hyj, hzj, rboundj, va2, mnv2,
        contact_margin,
        nsign,
        body_a, body_b,
        contact_friction, contact_friction_spin, contact_friction_roll,
        contact_condim,
        hfield_meta, hfield_data, hf_stride,
        mesh_verts, mesh_vert_edgeadr, mesh_edges,
        contacts, ws, num_contacts, dims.get_max_contacts(), env,
        contact_gap,
    )


def _capsule_box_contacts[
    DTYPE: DType,
    D: DimsLike,
    L_CONTACTS: Layout,
](
    env: Int,
    body_a: Int,
    body_b: Int,
    box_x: Scalar[DTYPE],
    box_y: Scalar[DTYPE],
    box_z: Scalar[DTYPE],
    box_qx: Scalar[DTYPE],
    box_qy: Scalar[DTYPE],
    box_qz: Scalar[DTYPE],
    box_qw: Scalar[DTYPE],
    box_hx: Scalar[DTYPE],
    box_hy: Scalar[DTYPE],
    box_hz: Scalar[DTYPE],
    cap_x: Scalar[DTYPE],
    cap_y: Scalar[DTYPE],
    cap_z: Scalar[DTYPE],
    cap_qx: Scalar[DTYPE],
    cap_qy: Scalar[DTYPE],
    cap_qz: Scalar[DTYPE],
    cap_qw: Scalar[DTYPE],
    cap_hl: Scalar[DTYPE],
    cap_r: Scalar[DTYPE],
    # -1 when the BOX is geom i, +1 when the CAPSULE is. See below.
    nsgn: Scalar[DTYPE],
    contact_margin: Scalar[DTYPE],
    contact_friction: Scalar[DTYPE],
    contact_friction_spin: Scalar[DTYPE],
    contact_friction_roll: Scalar[DTYPE],
    contact_condim: Int,
    dims: D,
    contacts: LayoutTensor[
        DTYPE, L_CONTACTS,
        MutAnyOrigin,
    ],
    mut num_contacts: Int,
    # ⚠ THE GAP HALF OF THE PAIR'S MARGIN, DEFAULTED TO 0 SO EVERY
    # EXISTING CALL SITE IS UNCHANGED. `contact_margin` is the narrowphase
    # CUTOFF (`margin + gap`); what a contact STORES as its
    # `includemargin` is `contact_margin - contact_gap`, and the solver excludes
    # `dist >= includemargin`. See `GEOM_IDX_GAP`.
    contact_gap: Scalar[DTYPE] = Scalar[DTYPE](0),
    # The emission cap. -1 = the model's `max_contacts` (every serial
    # caller); the block-per-env collision kernel passes the end of the
    # thread's staging window instead, so a routine cannot write past it.
    max_contacts_in: Int = -1,
) -> Int:
    """Capsule/box: up to TWO contacts, MuJoCo's manifold.

    A capsule lying along a box face touches over a segment; one point leaves
    it free to pivot, the same defect a box on one contact point has. See
    `box_capsule_manifold`.

    Returns the number of records written, so the caller knows whether to fall
    back (it never has to — 0 means no contact).

    ⚠ NORMAL SIGN. `box_capsule_manifold` returns box -> capsule. The record's
    convention is `body_b -> body_a` = `gj -> gi`, so when the BOX is geom i
    that is the negation of the manifold normal (`nsgn = -1`) and when the
    CAPSULE is geom i it is the manifold normal itself (`nsgn = +1`). The two
    single-point branches this replaces encoded the same thing as `nx = r[4]`
    versus `nx = -r[4]` followed by the shared emit's unconditional negation.
    """
    var max_contacts = (
        max_contacts_in if max_contacts_in >= 0 else dims.get_max_contacts()
    )
    var cb_dist = Array[Scalar[DTYPE], CB_MAX_POINTS](
        fill=Scalar[DTYPE](0)
    )
    var cb_pos = Array[Scalar[DTYPE], 3 * CB_MAX_POINTS](
        fill=Scalar[DTYPE](0)
    )
    var cb_n = Array[Scalar[DTYPE], 3 * CB_MAX_POINTS](
        fill=Scalar[DTYPE](0)
    )
    var n_cb = box_capsule_manifold[DTYPE](
        box_x, box_y, box_z, box_qx, box_qy, box_qz, box_qw,
        box_hx, box_hy, box_hz,
        cap_x, cap_y, cap_z, cap_qx, cap_qy, cap_qz, cap_qw,
        cap_hl, cap_r,
        contact_margin,
        cb_dist,
        cb_pos,
        cb_n,
    )

    var written = 0
    for c in range(n_cb):
        if num_contacts >= max_contacts:
            break
        if cb_dist[c] >= contact_margin:
            continue
        var c_off = num_contacts * CONTACT_SIZE
        contacts[env, c_off + CONTACT_IDX_BODY_A] = Scalar[DTYPE](body_a)
        contacts[env, c_off + CONTACT_IDX_BODY_B] = Scalar[DTYPE](body_b)
        contacts[env, c_off + CONTACT_IDX_POS_X] = cb_pos[3 * c + 0]
        contacts[env, c_off + CONTACT_IDX_POS_Y] = cb_pos[3 * c + 1]
        contacts[env, c_off + CONTACT_IDX_POS_Z] = cb_pos[3 * c + 2]
        contacts[env, c_off + CONTACT_IDX_NX] = nsgn * cb_n[3 * c + 0]
        contacts[env, c_off + CONTACT_IDX_NY] = nsgn * cb_n[3 * c + 1]
        contacts[env, c_off + CONTACT_IDX_NZ] = nsgn * cb_n[3 * c + 2]
        contacts[env, c_off + CONTACT_IDX_DIST] = cb_dist[c]
        contacts[
            env, c_off + CONTACT_IDX_INCLUDEMARGIN
        ] = contact_margin - contact_gap
        contacts[env, c_off + CONTACT_IDX_FRICTION] = contact_friction
        contacts[
            env, c_off + CONTACT_IDX_FRICTION_SPIN
        ] = contact_friction_spin
        contacts[
            env, c_off + CONTACT_IDX_FRICTION_ROLL
        ] = contact_friction_roll
        contacts[env, c_off + CONTACT_IDX_CONDIM] = Scalar[DTYPE](
            contact_condim
        )
        num_contacts += 1
        written += 1
    return written


@always_inline
def _box_box_contacts[
    DTYPE: DType,
    D: DimsLike,
    L_CONTACTS: Layout,
](
    env: Int,
    body_a: Int,
    body_b: Int,
    ai_x: Scalar[DTYPE],
    ai_y: Scalar[DTYPE],
    ai_z: Scalar[DTYPE],
    ai_qx: Scalar[DTYPE],
    ai_qy: Scalar[DTYPE],
    ai_qz: Scalar[DTYPE],
    ai_qw: Scalar[DTYPE],
    ai_hx: Scalar[DTYPE],
    ai_hy: Scalar[DTYPE],
    ai_hz: Scalar[DTYPE],
    bj_x: Scalar[DTYPE],
    bj_y: Scalar[DTYPE],
    bj_z: Scalar[DTYPE],
    bj_qx: Scalar[DTYPE],
    bj_qy: Scalar[DTYPE],
    bj_qz: Scalar[DTYPE],
    bj_qw: Scalar[DTYPE],
    bj_hx: Scalar[DTYPE],
    bj_hy: Scalar[DTYPE],
    bj_hz: Scalar[DTYPE],
    contact_margin: Scalar[DTYPE],
    contact_friction: Scalar[DTYPE],
    contact_friction_spin: Scalar[DTYPE],
    contact_friction_roll: Scalar[DTYPE],
    contact_condim: Int,
    dims: D,
    contacts: LayoutTensor[
        DTYPE, L_CONTACTS,
        MutAnyOrigin,
    ],
    mut num_contacts: Int,
    # ⚠ THE GAP HALF OF THE PAIR'S MARGIN, DEFAULTED TO 0 SO EVERY
    # EXISTING CALL SITE IS UNCHANGED. `contact_margin` is the narrowphase
    # CUTOFF (`margin + gap`); what a contact STORES as its
    # `includemargin` is `contact_margin - contact_gap`, and the solver excludes
    # `dist >= includemargin`. See `GEOM_IDX_GAP`.
    contact_gap: Scalar[DTYPE] = Scalar[DTYPE](0),
    # The emission cap. -1 = the model's `max_contacts` (every serial
    # caller); the block-per-env collision kernel passes the end of the
    # thread's staging window instead, so a routine cannot write past it.
    max_contacts_in: Int = -1,
) -> Int:
    """Box/box: the whole manifold, on both the FACE and EDGE-EDGE axes.

    Returns MuJoCo's `code` so the caller can fall back for the one case this
    does not write: `-1`, separated. On any `code >= 0` the records are written
    here and the caller must NOT emit again.

    A box resting on another touches over a whole face, and one point cannot
    express that — the same reason `_plane_box_contacts` exists. See
    `box_box_manifold` — MuJoCo 3.12's `mjc_BoxBox` since 2026-09-12 (AUD-31).
    """
    var max_contacts = (
        max_contacts_in if max_contacts_in >= 0 else dims.get_max_contacts()
    )
    var n_bb = 0
    var bb_dist = Array[Scalar[DTYPE], BB_MAX_POINTS](
        fill=Scalar[DTYPE](0)
    )
    var bb_pos = Array[Scalar[DTYPE], 3 * BB_MAX_POINTS](
        fill=Scalar[DTYPE](0)
    )
    var bb_n = Array[Scalar[DTYPE], 3](fill=Scalar[DTYPE](0))
    var code = box_box_manifold[DTYPE](
        ai_x, ai_y, ai_z, ai_qx, ai_qy, ai_qz, ai_qw, ai_hx, ai_hy, ai_hz,
        bj_x, bj_y, bj_z, bj_qx, bj_qy, bj_qz, bj_qw, bj_hx, bj_hy, bj_hz,
        contact_margin,
        n_bb,
        bb_dist,
        bb_pos,
        bb_n,
    )
    if code < 0:
        return code

    for c in range(n_bb):
        if num_contacts >= max_contacts:
            break
        if bb_dist[c] >= contact_margin:
            continue
        var c_off = num_contacts * CONTACT_SIZE
        contacts[env, c_off + CONTACT_IDX_BODY_A] = Scalar[DTYPE](body_a)
        contacts[env, c_off + CONTACT_IDX_BODY_B] = Scalar[DTYPE](body_b)
        contacts[env, c_off + CONTACT_IDX_POS_X] = bb_pos[3 * c + 0]
        contacts[env, c_off + CONTACT_IDX_POS_Y] = bb_pos[3 * c + 1]
        contacts[env, c_off + CONTACT_IDX_POS_Z] = bb_pos[3 * c + 2]
        # `box_box_manifold` returns the normal pointing A -> B; the record's
        # convention is `body_b -> body_a`, which is what the shared emit below
        # gets by negating. Same negation here.
        contacts[env, c_off + CONTACT_IDX_NX] = -bb_n[0]
        contacts[env, c_off + CONTACT_IDX_NY] = -bb_n[1]
        contacts[env, c_off + CONTACT_IDX_NZ] = -bb_n[2]
        contacts[env, c_off + CONTACT_IDX_DIST] = bb_dist[c]
        contacts[
            env, c_off + CONTACT_IDX_INCLUDEMARGIN
        ] = contact_margin - contact_gap
        contacts[env, c_off + CONTACT_IDX_FRICTION] = contact_friction
        contacts[
            env, c_off + CONTACT_IDX_FRICTION_SPIN
        ] = contact_friction_spin
        contacts[
            env, c_off + CONTACT_IDX_FRICTION_ROLL
        ] = contact_friction_roll
        contacts[env, c_off + CONTACT_IDX_CONDIM] = Scalar[DTYPE](
            contact_condim
        )
        num_contacts += 1
    return code


@always_inline
@always_inline
def _fill_pair_solparams[
    DTYPE: DType,
    L_CONTACTS: Layout,
](
    env: Int,
    n0: Int,
    n1: Int,
    mx: Array[Scalar[DTYPE], 12],
    contacts: LayoutTensor[
        DTYPE, L_CONTACTS,
        MutAnyOrigin,
    ],
):
    """Stamp a geom pair's mixed solref/solimp onto every contact it emitted.

    The mixed values are constant across all points of one pair, so they are
    written once per pair rather than at each of the nineteen emit sites — a
    narrow-phase branch added later then inherits them instead of silently
    shipping zeros.

    ⚠ CALL THIS AT EVERY EXIT OF THE PAIR LOOP BODY, NOT JUST THE BOTTOM. The
    first version ran only at the bottom, and the two PLANE branches end with
    `continue` — so every plane contact in the engine got solref (0, 0), which
    `solref_spring_damper` then read as the DIRECT form with zero stiffness and
    zero damping. It showed up as `test_contacts_vs_mujoco` failing on hopper
    within one build, but only because that gate exists; a post-pass at the
    bottom of a loop body is safe only when the body has a single exit.
    """
    for c in range(n0, n1):
        var o = c * CONTACT_SIZE
        contacts[env, o + CONTACT_IDX_SOLREF_0] = mx[4]
        contacts[env, o + CONTACT_IDX_SOLREF_1] = mx[5]
        contacts[env, o + CONTACT_IDX_SOLIMP_0] = mx[6]
        contacts[env, o + CONTACT_IDX_SOLIMP_1] = mx[7]
        contacts[env, o + CONTACT_IDX_SOLIMP_2] = mx[8]
        contacts[env, o + CONTACT_IDX_SOLIMP_3] = mx[9]
        contacts[env, o + CONTACT_IDX_SOLIMP_4] = mx[10]


def _detect_contacts_env[
    DTYPE: DType,
    BATCH: Int,
    D: DimsLike,
    L_XPOS: Layout,
    L_XQUAT: Layout,
    L_GEOMS: Layout,
    L_BODIES: Layout,
    L_MMETA: Layout,
    L_EXCLUDES: Layout,
    L_PAIRS: Layout,
    L_MESH_META: Layout,
    L_MESH_VERTS: Layout,
    L_MESH_POLYS: Layout,
    L_MESH_POLYVERT: Layout,
    L_MESH_VERT_POLYMAP: Layout,
    L_MESH_VERT_EDGEADR: Layout,
    L_MESH_EDGES: Layout,
    L_HF_META: Layout,
    L_HF_DATA: Layout,
    L_CONTACTS: Layout,
    L_SMETA: Layout,
    L_WS: Layout,
    # Heightfield collision compiles on BOTH targets. It used to be CPU-only:
    # `hfield_convex_contacts` needs its own instantiation of the whole
    # GJK/EPA stack (the prism is a sixth shape type), and a second copy of
    # EPA's polytope on the per-thread stack pushed the Metal kernel past
    # "Compute function exceeds available stack space". Moving the polytope
    # into `d.ccd_ws` — which is where MuJoCo has always kept it, see
    # `ccd_workspace` — took ~7.5 KB per instantiation off that stack and the
    # parameter with it. It stays a parameter only so a caller with no
    # heightfields pays nothing for the branch.
    HFIELD_ENABLED: Bool = True,
](
    env: Int,
    dims: D,
    xpos: LayoutTensor[
        DTYPE, L_XPOS, MutAnyOrigin
    ],
    xquat: LayoutTensor[
        DTYPE, L_XQUAT, MutAnyOrigin
    ],
    geoms: LayoutTensor[
        DTYPE, L_GEOMS, MutAnyOrigin
    ],
    bodies: LayoutTensor[
        DTYPE, L_BODIES, MutAnyOrigin
    ],
    mmeta: LayoutTensor[
        DTYPE, L_MMETA, MutAnyOrigin
    ],
    excludes: LayoutTensor[
        DTYPE, L_EXCLUDES, MutAnyOrigin
    ],
    pairs: LayoutTensor[
        DTYPE, L_PAIRS, MutAnyOrigin
    ],
    mesh_meta: LayoutTensor[
        DTYPE,
        L_MESH_META,
        MutAnyOrigin,
    ],
    mesh_verts: LayoutTensor[
        DTYPE, L_MESH_VERTS, MutAnyOrigin
    ],
    mesh_polys: LayoutTensor[
        DTYPE,
        L_MESH_POLYS,
        MutAnyOrigin,
    ],
    mesh_polyvert: LayoutTensor[
        DTYPE, L_MESH_POLYVERT, MutAnyOrigin
    ],
    mesh_polymap: LayoutTensor[
        DTYPE, L_MESH_POLYVERT, MutAnyOrigin
    ],
    mesh_vert_polymap: LayoutTensor[
        DTYPE, L_MESH_VERT_POLYMAP, MutAnyOrigin
    ],
    mesh_vert_edgeadr: LayoutTensor[
        DTYPE, L_MESH_VERT_EDGEADR, MutAnyOrigin
    ],
    mesh_edges: LayoutTensor[
        DTYPE, L_MESH_EDGES, MutAnyOrigin
    ],
    hfield_meta: LayoutTensor[
        DTYPE, L_HF_META, MutAnyOrigin
    ],
    hfield_data: LayoutTensor[
        DTYPE, L_HF_DATA, MutAnyOrigin
    ],
    contacts: LayoutTensor[
        DTYPE, L_CONTACTS,
        MutAnyOrigin,
    ],
    smeta: LayoutTensor[
        DTYPE, L_SMETA, MutAnyOrigin
    ],
    # EPA's polytope, one row per env — MuJoCo's `config->buffer`. See
    # `ccd_workspace`.
    ws: LayoutTensor[
        DTYPE, L_WS, MutAnyOrigin
    ],
):
    """Unified contact detection for one env (verbatim from
    detect_contacts_gpu; mesh branches compiled in iff nmesh_verts > 0)."""
    var nq = dims.get_nq()
    var nv = dims.get_nv()
    var nbody = dims.get_nbody()
    # `<exclude>` signatures, sorted once per call (see `exclude_signatures`).
    # ⚠ A model with NO excludes gets a one-slot static array, not the heap
    # leg: `cap` is 0 for a static zero as well as for a dynamic dim, and a
    # heap slab per call was +3% on the RK4 gym rows (four calls a step).
    comptime EX_CAP = cap[D.NEXCLUDE]() if may_exist[D.NEXCLUDE]() else 1
    var ex_sig = Scratch[Int, EX_CAP](
        dims.get_nexclude() if dims.get_nexclude() > 0 else 1, fill=0
    )
    var n_sig = exclude_signatures[DTYPE, EX_CAP](
        nbody, dims.get_nexclude(), mmeta, excludes, ex_sig
    )
    var njoint = dims.get_njoint()
    var max_contacts = dims.get_max_contacts()
    var ngeom = dims.get_ngeom()
    var nexclude = dims.get_nexclude()
    var nmesh_verts = dims.get_nmesh_verts()
    var npair = dims.get_npair()
    # `mjModel.opt.ccd_tolerance` / `.ccd_iterations` — EPA's stopping rule,
    # read from model META rather than hardcoded in `gjk.mojo`. Seeded to
    # MuJoCo's defaults by `Model.__init__` and overwritten from `<option>` by
    # `fields_build`, so a non-positive value here means the slot was clobbered
    # by a builder predating it: fall back rather than iterate zero times.
    var ccd_tol = rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_CCD_TOLERANCE])
    if ccd_tol <= 0:
        ccd_tol = Scalar[DTYPE](MJ_CCD_TOLERANCE)
    var ccd_iter = Int(
        rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_CCD_ITERATIONS])
    )
    if ccd_iter < 1:
        ccd_iter = MJ_CCD_ITERATIONS
    # `<option><flag multiccd="disable"/></option>` — `mjDSBL_MULTICCD`. When
    # set, `mjc_Convex` never reaches its perturbation loop and every convex
    # pair that is not BOX x BOX keeps the single point `mjc_penetration`
    # found. 0 means the flag is absent, i.e. MuJoCo's default of ON.
    var multiccd_off = (
        rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_MULTICCD_DISABLED]) != 0
    )
    var num_contacts = 0

    # ⚠⚠ `sa`/`sb` ARE THE LOOP'S ORDER AND `gi`/`gj` ARE MuJoCo'S. The pair
    # this loop names is `sa < sb`, which is the order MuJoCo's own geom loops
    # produce — but the reference then SORTS IT BY TYPE in `pushPairArena`
    # (`engine_collision_driver.c:489`) before anything touches it:
    #
    #     if (m->geom_type[g1] > m->geom_type[g2]) { swap(g1, g2); }
    #
    # and the narrow phase is not symmetric in its operands. This loop's
    # analytic branches each spell both orders out and so were never affected;
    # the GJK/EPA branches were, on every pair whose lower-indexed geom has the
    # higher type — cylinder before ellipsoid, mesh before box, and so on. See
    # the same canonicalisation in `broadphase_sap`, where the sweep's own
    # order made it much worse.
    # Which geoms a `<pair>` names, so the mask skip below cannot drop them.
    var pair_geom = Scratch[Int, cap[D.NGEOM]()](ngeom, fill=0)
    var n_pair_live = Int(rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_NPAIR]))
    if n_pair_live > npair:
        n_pair_live = npair
    for p in range(n_pair_live):
        for side in range(2):
            var pg = Int(
                rebind[Scalar[DTYPE]](
                    pairs[p, PAIR_IDX_GEOM1 if side == 0 else PAIR_IDX_GEOM2]
                )
            )
            if pg >= 0 and pg < ngeom:
                pair_geom[pg] = 1
    for sa in range(ngeom):
        var sa_type = Int(
            rebind[Scalar[DTYPE]](geoms[sa, GEOM_IDX_TYPE])
        )
        # A geom with contype and conaffinity both zero collides with nothing
        # (`filterBitmask`, engine_collision_driver.c:535): skip it before the
        # pair loop, as the SAP sweep does (`broadphase_sap.mojo`, same rule).
        # ⚠ UNLESS A `<pair>` NAMES IT — predefined pairs bypass the mask
        # (engine_collision_driver.c:611, and the SAP twin's note); the
        # `pair_geom` flags are built above the loop.
        if (
            Int(rebind[Scalar[DTYPE]](geoms[sa, GEOM_IDX_CONTYPE])) == 0
            and Int(rebind[Scalar[DTYPE]](geoms[sa, GEOM_IDX_CONAFFINITY])) == 0
            and pair_geom[sa] == 0
        ):
            continue
        for sb in range(sa + 1, ngeom):
            if (
                Int(rebind[Scalar[DTYPE]](geoms[sb, GEOM_IDX_CONTYPE])) == 0
                and Int(rebind[Scalar[DTYPE]](geoms[sb, GEOM_IDX_CONAFFINITY])) == 0
                and pair_geom[sb] == 0
            ):
                continue
            if num_contacts >= max_contacts:
                # ⚠ ORDERED BEFORE THE EARLY EXIT TOO — see the SAP twin.
                sort_contacts_mujoco_order[DTYPE](
                    env, contacts, num_contacts
                )
                smeta[env, META_IDX_NUM_CONTACTS] = Scalar[DTYPE](
                    num_contacts
                )
                return
            var sb_type = Int(
                rebind[Scalar[DTYPE]](geoms[sb, GEOM_IDX_TYPE])
            )
            var gi = sa
            var gj = sb
            # ⚠ RANK, NOT THE RAW ID — see `mj_geom_type_rank`. This enum is
            # not `mjtGeom`, and `pushPairArena` sorts by `mjtGeom`.
            if mj_geom_type_rank(sa_type) > mj_geom_type_rank(sb_type):
                gi = sb
                gj = sa

            var gi_type = Int(
                rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_TYPE])
            )
            var gi_body = Int(
                rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_BODY])
            )
            var gi_contype = Int(
                rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_CONTYPE])
            )
            var gi_conaffinity = Int(
                rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_CONAFFINITY])
            )
            var gj_type = Int(
                rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_TYPE])
            )
            var gj_body = Int(
                rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_BODY])
            )
            # `<contact><pair>`: EVERY filter below is skipped when this geom
            # pair is predefined. MuJoCo runs the predefined-pair merge loop
            # BEFORE `canCollide2` and the exclude scan
            # (`engine_collision_driver.c:390-412`), and `mj_collideGeoms`
            # skips the contact mask whenever `ipair >= 0` — so a pair
            # collides through cleared masks, through `<exclude>`, and through
            # the weld tests, all four confirmed against the 3.10.0 runtime.
            # ⚠ ONE bypass is not enough. Implementing only the mask skip
            # would still lose every pair between geoms on the same body or on
            # a welded parent/child, which is a normal thing to declare.
            var ipair = find_predefined_pair[DTYPE](
                gi, gj, dims, pairs, mmeta
            )
            if ipair < 0:
                if gi_type == GEOM_PLANE and gj_body == 0:
                    continue
                if gj_type == GEOM_PLANE and gi_body == 0:
                    continue
                # MuJoCo's body-pair filter — weld, weld-parent and exclude.
                # See `pair_body_filtered`; shared with BOTH SAP loops.
                if pair_body_filtered[DTYPE, EX_CAP=EX_CAP](
                    gi_body, gj_body, bodies, mmeta, excludes,
                    ex_sig, n_sig, nbody,
                ):
                    continue
                var gj_contype = Int(
                    rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_CONTYPE])
                )
                var gj_conaffinity = Int(
                    rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_CONAFFINITY])
                )
                if (gi_contype & gj_conaffinity) == 0 and (
                    gj_contype & gi_conaffinity
                ) == 0:
                    continue

            # Margin: the PAIR's own value when predefined, otherwise the sum
            # of the two geoms' — `mj_collideGeoms` picks one or the other and
            # never combines them. Hoisted above the bounding-sphere test
            # because MuJoCo's `mj_filterSphere` is called WITH it.
            var contact_includemargin = rebind[Scalar[DTYPE]](
                geoms[gi, GEOM_IDX_MARGIN]
            ) + rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_MARGIN])
            var contact_gap = rebind[Scalar[DTYPE]](
                geoms[gi, GEOM_IDX_GAP]
            ) + rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_GAP])
            if ipair >= 0:
                contact_includemargin = rebind[Scalar[DTYPE]](
                    pairs[ipair, PAIR_IDX_MARGIN]
                )
                contact_gap = rebind[Scalar[DTYPE]](
                    pairs[ipair, PAIR_IDX_GAP]
                )
            # ⚠⚠ TWO VALUES, NOT ONE, and the CUTOFF is the wider of them.
            # 3.10.0 hands the collision function `margin + gap` and
            # `mj_setContact` `margin` alone, so a contact in
            # [margin, margin+gap) is DETECTED and then excluded from the
            # solver by `con->exclude = dist >= includemargin`. Every test
            # below reads `contact_margin` and every `includemargin` WRITE
            # reads `contact_margin - contact_gap`; with no `<geom gap>` they
            # are the same number and nothing moves.
            # ⚠⚠ EVERY `INCLUDEMARGIN` WRITE IN THIS FUNCTION READS
            # `contact_includemargin`, NEVER `contact_margin`. There are NINE
            # of them inline in the branches below plus six more inside the
            # emit helpers, and the first pass at this feature caught only the
            # helpers — a sphere held in the gap band because its record said
            # the CUTOFF was its includemargin, which is the same rule written
            # fifteen times. If you add a branch here, the value to store is
            # this one.
            var contact_margin = contact_includemargin + contact_gap

            var pi_x: Scalar[DTYPE] = 0
            var pi_y: Scalar[DTYPE] = 0
            var pi_z: Scalar[DTYPE] = 0
            var qi_x: Scalar[DTYPE] = 0
            var qi_y: Scalar[DTYPE] = 0
            var qi_z: Scalar[DTYPE] = 0
            var qi_w: Scalar[DTYPE] = 1
            _geom_world_pos[DTYPE](
                env,
                gi,
                geoms,
                xpos,
                xquat,
                pi_x,
                pi_y,
                pi_z,
                qi_x,
                qi_y,
                qi_z,
                qi_w,
            )
            var pj_x: Scalar[DTYPE] = 0
            var pj_y: Scalar[DTYPE] = 0
            var pj_z: Scalar[DTYPE] = 0
            var qj_x: Scalar[DTYPE] = 0
            var qj_y: Scalar[DTYPE] = 0
            var qj_z: Scalar[DTYPE] = 0
            var qj_w: Scalar[DTYPE] = 1
            _geom_world_pos[DTYPE](
                env,
                gj,
                geoms,
                xpos,
                xquat,
                pj_x,
                pj_y,
                pj_z,
                qj_x,
                qj_y,
                qj_z,
                qj_w,
            )

            # Broadphase bounding sphere check (skip for plane geoms — they're infinite)
            if gi_type != GEOM_PLANE and gj_type != GEOM_PLANE:
                var dx = pi_x - pj_x
                var dy = pi_y - pj_y
                var dz = pi_z - pj_z
                var dist_sq = dx * dx + dy * dy + dz * dz
                var ri_bound = rebind[Scalar[DTYPE]](
                    geoms[gi, GEOM_IDX_RBOUND]
                )
                var rj_bound = rebind[Scalar[DTYPE]](
                    geoms[gj, GEOM_IDX_RBOUND]
                )
                # ⚠ `+ contact_margin` — MuJoCo's `mj_filterSphere` is called
                # with the margin (`rbound1 + rbound2 + margin`) and this test
                # omitted it, so a pair separated by less than its margin but
                # more than the two radii was discarded before the narrow
                # phase ever ran. Latent while every margin was small relative
                # to the geometry; NOT latent for `<pair margin=>`, which is
                # frequently the whole point of declaring the pair.
                var bound = ri_bound + rj_bound + contact_margin
                if dist_sq > bound * bound:
                    continue

            var ri = rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_RADIUS])
            var rj = rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_RADIUS])
            # Bounding radii, re-read here rather than reused from the
            # broadphase check above: that one is scoped to its `if` and skips
            # plane pairs entirely. Multi-CCD scales its distinctness tolerance
            # by the smaller of the two (`mjc_Convex`).
            var rbound_i = rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_RBOUND])
            var rbound_j = rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_RBOUND])
            var hli = rebind[Scalar[DTYPE]](
                geoms[gi, GEOM_IDX_HALF_LENGTH]
            )
            var hlj = rebind[Scalar[DTYPE]](
                geoms[gj, GEOM_IDX_HALF_LENGTH]
            )
            var hxi = rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_HALF_X])
            var hyi = rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_HALF_Y])
            var hzi = rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_HALF_Z])
            var hxj = rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_HALF_X])
            var hyj = rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_HALF_Y])
            var hzj = rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_HALF_Z])
            # Contact parameters: MuJoCo's full rule, PRIORITY FIRST. This
            # used to be an unconditional elementwise max on friction and
            # condim, with solref/solimp not consulted at all — so
            # `<geom priority>` was ignored and the per-geom solparams already
            # in the geom record were dead data. See `mix_contact_params`.
            #
            # A PREDEFINED PAIR IS NOT MIXED AT ALL — it supplies its own
            # condim/friction/solref/solimp wholesale, and `priority` never
            # enters. `mj_collideGeoms` branches on `ipair` for exactly this.
            var _mx = pair_params[DTYPE](
                ipair, pairs
            ) if ipair >= 0 else mix_contact_params[DTYPE](
                Int(rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_PRIORITY])),
                Int(rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_CONDIM])),
                rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_FRICTION]),
                rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_FRICTION_SPIN]),
                rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_FRICTION_ROLL]),
                rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_SOLREF_0]),
                rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_SOLREF_1]),
                rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_SOLIMP_0]),
                rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_SOLIMP_1]),
                rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_SOLIMP_2]),
                rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_SOLIMP_3]),
                rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_SOLIMP_4]),
                Int(rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_PRIORITY])),
                Int(rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_CONDIM])),
                rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_FRICTION]),
                rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_FRICTION_SPIN]),
                rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_FRICTION_ROLL]),
                rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_SOLREF_0]),
                rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_SOLREF_1]),
                rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_SOLIMP_0]),
                rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_SOLIMP_1]),
                rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_SOLIMP_2]),
                rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_SOLIMP_3]),
                rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_SOLIMP_4]),
            )
            var contact_condim = Int(_mx[0])
            var contact_friction = _mx[1]
            var contact_friction_spin = _mx[2]
            var contact_friction_roll = _mx[3]
            # First contact slot this PAIR will fill. The mixed solver
            # parameters are constant across every point the pair emits, so
            # they are written once at the bottom of this loop body rather
            # than at each of the nineteen emit sites — which also means a
            # narrow-phase branch added later gets them for free instead of
            # silently shipping zeros.
            var _n0 = num_contacts

            # `contact_margin` is computed above, before the bounding-sphere
            # test that has to see it. (It used to be recomputed here from the
            # two geoms, under a comment claiming "max of both geoms"
            # immediately above a line summing them — the sum is right, the
            # comment never was.)

            # --- Plane handling ---
            if gi_type == GEOM_PLANE:
                # The plane's FULL pose. This branch used to keep only `pi_z` as a
                # `ground_z` and hardcode the contact normal to (0,0,1), i.e. it modelled
                # every plane as a horizontal floor at the height of its origin. See
                # `collision/plane_frame.mojo`. The other geom is rebased into the PLANE'S
                # FRAME (`fp_*` / `fq_*`), where the plane really is z=0 with normal +z —
                # exactly what every `*_plane` primitive assumes — so the arithmetic below
                # is unchanged and only the contact point and normal map back to world.
                var plp_x = pi_x
                var plp_y = pi_y
                var plp_z = pi_z
                var plq_x = qi_x
                var plq_y = qi_y
                var plq_z = qi_z
                var plq_w = qi_w
                var pn = plane_world_normal[DTYPE](plq_x, plq_y, plq_z, plq_w)
                var lfp = to_plane_frame[DTYPE](
                    plp_x, plp_y, plp_z, plq_x, plq_y, plq_z, plq_w,
                    pj_x, pj_y, pj_z,
                )
                var lfq = quat_to_plane_frame[DTYPE](
                    plq_x, plq_y, plq_z, plq_w, qj_x, qj_y, qj_z, qj_w
                )
                var fp_x = lfp[0]
                var fp_y = lfp[1]
                var fp_z = lfp[2]
                var fq_x = lfq[0]
                var fq_y = lfq[1]
                var fq_z = lfq[2]
                var fq_w = lfq[3]
                var ground_z = Scalar[DTYPE](0)
                # PLANE-SIDE BOUNDING-SPHERE REJECT — MuJoCo's second
                # `mj_filterSphere` arm. In the plane's frame `fp_z` IS
                # `planeGeomDist`. Kept in step with the SAP path, which is
                # where it actually pays: `_plane_mesh_contacts` scans EVERY
                # hull vertex with no early out, so a floor a mesh never
                # touches costs its full vertex count every step.
                # ⚠ `+ contact_margin` — a geom hovering within its margin of
                # the floor is a contact MuJoCo reports; without the term it
                # vanishes silently. Gated by `plane_margin/{naive,sap}` in
                # `test_contact_pair_vs_mujoco.mojo`.
                # ⚠ Skipping the `_fill_pair_solparams` tail below is safe ONLY
                # because nothing has been emitted yet — it stamps the range
                # [_n0, num_contacts), which is empty here.
                var rb_pl_gj = rebind[Scalar[DTYPE]](
                    geoms[gj, GEOM_IDX_RBOUND]
                )
                if rb_pl_gj > Scalar[DTYPE](0) and fp_z > contact_margin + rb_pl_gj:
                    continue
                if gj_type == GEOM_CAPSULE:
                    # MuJoCo mjc_PlaneCapsule: test BOTH endpoints, up to 2 contacts
                    var axis_w = gpu_quat_rotate(
                        fq_x,
                        fq_y,
                        fq_z,
                        fq_w,
                        Scalar[DTYPE](0),
                        Scalar[DTYPE](0),
                        Scalar[DTYPE](1),
                    )
                    # Endpoint 1: center + half_length * axis
                    # `axis_w` is in the PLANE'S frame (fq_* is the rebased
                    # orientation), which is what the endpoint arithmetic
                    # needs. The FRAME_T1 hint is read in WORLD space, so it
                    # goes back — see collision/contact_frame.mojo for what
                    # that slot is and is not.
                    var axis_wd = gpu_quat_rotate(
                        plq_x, plq_y, plq_z, plq_w,
                        axis_w[0], axis_w[1], axis_w[2],
                    )
                    var e1_x = fp_x + hlj * axis_w[0]
                    var e1_y = fp_y + hlj * axis_w[1]
                    var e1_z = fp_z + hlj * axis_w[2]
                    var dist1 = e1_z - rj - ground_z
                    if dist1 < contact_margin and num_contacts < max_contacts:
                        var c_off = num_contacts * CONTACT_SIZE
                        contacts[env, c_off + CONTACT_IDX_BODY_A] = Scalar[
                            DTYPE
                        ](gj_body)
                        contacts[env, c_off + CONTACT_IDX_BODY_B] = Scalar[
                            DTYPE
                        ](0)
                        var cw = from_plane_frame[DTYPE](
                            plp_x, plp_y, plp_z, plq_x, plq_y, plq_z, plq_w,
                            e1_x, e1_y,
                            ground_z + dist1 * Scalar[DTYPE](0.5),
                        )
                        contacts[env, c_off + CONTACT_IDX_POS_X] = cw[0]
                        contacts[env, c_off + CONTACT_IDX_POS_Y] = cw[1]
                        contacts[env, c_off + CONTACT_IDX_POS_Z] = cw[2]
                        contacts[env, c_off + CONTACT_IDX_NX] = pn[0]
                        contacts[env, c_off + CONTACT_IDX_NY] = pn[1]
                        contacts[env, c_off + CONTACT_IDX_NZ] = pn[2]
                        contacts[env, c_off + CONTACT_IDX_DIST] = dist1
                        contacts[
                            env, c_off + CONTACT_IDX_INCLUDEMARGIN
                        ] = contact_includemargin
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION
                        ] = contact_friction
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION_SPIN
                        ] = contact_friction_spin
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION_ROLL
                        ] = contact_friction_roll
                        contacts[env, c_off + CONTACT_IDX_CONDIM] = Scalar[
                            DTYPE
                        ](contact_condim)
                        contacts[env, c_off + CONTACT_IDX_FRAME_T1_X] = axis_wd[
                            0
                        ]
                        contacts[env, c_off + CONTACT_IDX_FRAME_T1_Y] = axis_wd[
                            1
                        ]
                        contacts[env, c_off + CONTACT_IDX_FRAME_T1_Z] = axis_wd[
                            2
                        ]
                        num_contacts += 1
                    # Endpoint 2: center - half_length * axis
                    var e2_x = fp_x - hlj * axis_w[0]
                    var e2_y = fp_y - hlj * axis_w[1]
                    var e2_z = fp_z - hlj * axis_w[2]
                    var dist2 = e2_z - rj - ground_z
                    if dist2 < contact_margin and num_contacts < max_contacts:
                        var c_off = num_contacts * CONTACT_SIZE
                        contacts[env, c_off + CONTACT_IDX_BODY_A] = Scalar[
                            DTYPE
                        ](gj_body)
                        contacts[env, c_off + CONTACT_IDX_BODY_B] = Scalar[
                            DTYPE
                        ](0)
                        var cw = from_plane_frame[DTYPE](
                            plp_x, plp_y, plp_z, plq_x, plq_y, plq_z, plq_w,
                            e2_x, e2_y,
                            ground_z + dist2 * Scalar[DTYPE](0.5),
                        )
                        contacts[env, c_off + CONTACT_IDX_POS_X] = cw[0]
                        contacts[env, c_off + CONTACT_IDX_POS_Y] = cw[1]
                        contacts[env, c_off + CONTACT_IDX_POS_Z] = cw[2]
                        contacts[env, c_off + CONTACT_IDX_NX] = pn[0]
                        contacts[env, c_off + CONTACT_IDX_NY] = pn[1]
                        contacts[env, c_off + CONTACT_IDX_NZ] = pn[2]
                        contacts[env, c_off + CONTACT_IDX_DIST] = dist2
                        contacts[
                            env, c_off + CONTACT_IDX_INCLUDEMARGIN
                        ] = contact_includemargin
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION
                        ] = contact_friction
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION_SPIN
                        ] = contact_friction_spin
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION_ROLL
                        ] = contact_friction_roll
                        contacts[env, c_off + CONTACT_IDX_CONDIM] = Scalar[
                            DTYPE
                        ](contact_condim)
                        contacts[env, c_off + CONTACT_IDX_FRAME_T1_X] = axis_wd[
                            0
                        ]
                        contacts[env, c_off + CONTACT_IDX_FRAME_T1_Y] = axis_wd[
                            1
                        ]
                        contacts[env, c_off + CONTACT_IDX_FRAME_T1_Z] = axis_wd[
                            2
                        ]
                        num_contacts += 1
                elif gj_type == GEOM_CYLINDER:
                    # Up to FOUR points — two rim, two triangle — not
                    # one. See `_plane_cylinder_contacts`; a cylinder on
                    # its flat face needs a support polygon or it tips.
                    _plane_cylinder_contacts[DTYPE, BATCH](
                        env,
                        gj_body,
                        fp_x, fp_y, fp_z,
                        fq_x, fq_y, fq_z, fq_w,
                        rj,
                        hlj,
                        ground_z,
                        plp_x, plp_y, plp_z,
                        plq_x, plq_y, plq_z, plq_w,
                        contact_margin,
                        contact_friction,
                        contact_friction_spin,
                        contact_friction_roll,
                        contact_condim,
                        0,
                        dims,
                        contacts,
                        num_contacts,
                        contact_gap,
                    )
                elif gj_type == GEOM_SPHERE:
                    var dist = fp_z - rj - ground_z
                    if dist < contact_margin and num_contacts < max_contacts:
                        var c_off = num_contacts * CONTACT_SIZE
                        contacts[env, c_off + CONTACT_IDX_BODY_A] = Scalar[
                            DTYPE
                        ](gj_body)
                        contacts[env, c_off + CONTACT_IDX_BODY_B] = Scalar[
                            DTYPE
                        ](0)
                        var cw = from_plane_frame[DTYPE](
                            plp_x, plp_y, plp_z, plq_x, plq_y, plq_z, plq_w,
                            fp_x, fp_y,
                            ground_z + dist * Scalar[DTYPE](0.5),
                        )
                        contacts[env, c_off + CONTACT_IDX_POS_X] = cw[0]
                        contacts[env, c_off + CONTACT_IDX_POS_Y] = cw[1]
                        contacts[env, c_off + CONTACT_IDX_POS_Z] = cw[2]
                        contacts[env, c_off + CONTACT_IDX_NX] = pn[0]
                        contacts[env, c_off + CONTACT_IDX_NY] = pn[1]
                        contacts[env, c_off + CONTACT_IDX_NZ] = pn[2]
                        contacts[env, c_off + CONTACT_IDX_DIST] = dist
                        contacts[
                            env, c_off + CONTACT_IDX_INCLUDEMARGIN
                        ] = contact_includemargin
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION
                        ] = contact_friction
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION_SPIN
                        ] = contact_friction_spin
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION_ROLL
                        ] = contact_friction_roll
                        contacts[env, c_off + CONTACT_IDX_CONDIM] = Scalar[
                            DTYPE
                        ](contact_condim)
                        num_contacts += 1
                elif gj_type == GEOM_ELLIPSOID:
                    # MuJoCo routes plane x ellipsoid through mjc_PlaneConvex,
                    # which reports the single deepest support point. A smooth
                    # strictly-convex surface touches a plane at one point, and
                    # MuJoCo was measured to emit exactly 1 contact over 500
                    # random poses — so, unlike box_plane, there is no second
                    # contact to look for.
                    var ep = ellipsoid_plane[DTYPE](
                        fp_x, fp_y, fp_z,
                        fq_x, fq_y, fq_z, fq_w,
                        hxj, hyj, hzj,
                        ground_z,
                    )
                    var dist = ep[0]
                    if dist < contact_margin and num_contacts < max_contacts:
                        var c_off = num_contacts * CONTACT_SIZE
                        contacts[env, c_off + CONTACT_IDX_BODY_A] = Scalar[
                            DTYPE
                        ](gj_body)
                        contacts[env, c_off + CONTACT_IDX_BODY_B] = Scalar[
                            DTYPE
                        ](0)
                        var cw = from_plane_frame[DTYPE](
                            plp_x, plp_y, plp_z, plq_x, plq_y, plq_z, plq_w,
                            ep[1], ep[2], ep[3],
                        )
                        contacts[env, c_off + CONTACT_IDX_POS_X] = cw[0]
                        contacts[env, c_off + CONTACT_IDX_POS_Y] = cw[1]
                        contacts[env, c_off + CONTACT_IDX_POS_Z] = cw[2]
                        contacts[env, c_off + CONTACT_IDX_NX] = pn[0]
                        contacts[env, c_off + CONTACT_IDX_NY] = pn[1]
                        contacts[env, c_off + CONTACT_IDX_NZ] = pn[2]
                        contacts[env, c_off + CONTACT_IDX_DIST] = dist
                        contacts[
                            env, c_off + CONTACT_IDX_INCLUDEMARGIN
                        ] = contact_includemargin
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION
                        ] = contact_friction
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION_SPIN
                        ] = contact_friction_spin
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION_ROLL
                        ] = contact_friction_roll
                        contacts[env, c_off + CONTACT_IDX_CONDIM] = Scalar[
                            DTYPE
                        ](contact_condim)
                        num_contacts += 1
                elif gj_type == GEOM_BOX:
                    # Up to FOUR corners, not one — see
                    # `_plane_box_contacts` and task #42.
                    _plane_box_contacts[DTYPE](
                        env,
                        gj_body,
                        fp_x, fp_y, fp_z,
                        fq_x, fq_y, fq_z, fq_w,
                        hxj, hyj, hzj,
                        ground_z,
                        plp_x, plp_y, plp_z,
                        plq_x, plq_y, plq_z, plq_w,
                        contact_margin,
                        contact_friction,
                        contact_friction_spin,
                        contact_friction_roll,
                        contact_condim,
                        0,
                        dims,
                        contacts,
                        num_contacts,
                        contact_gap,
                    )
                elif gj_type == GEOM_MESH:
                    # Plane-mesh: scan hull vertices below plane
                    comptime if may_exist[D.NMESH_VERTS]():
                        var pmw = -1
                        _plane_mesh_contacts[
                            DTYPE,
                            0, False, True](
                            env,
                            gj,
                            gj_body,
                            fp_x,
                            fp_y,
                            fp_z,
                            fq_x,
                            fq_y,
                            fq_z,
                            fq_w,
                            ground_z,
                            plp_x, plp_y, plp_z,
                            plq_x, plq_y, plq_z, plq_w,
                            contact_margin,
                            contact_friction,
                            contact_friction_spin,
                            contact_friction_roll,
                            contact_condim,
                            dims,
                            geoms,
                            mesh_meta,
                            mesh_verts,
                            mesh_vert_edgeadr,
                            mesh_edges,
                            contacts,
                            num_contacts,
                            pmw,
                            contact_gap,
                        )
                _fill_pair_solparams[DTYPE](
                    env, _n0, num_contacts, _mx, contacts
                )
                continue

            if gj_type == GEOM_PLANE:
                # The plane's FULL pose. This branch used to keep only `pj_z` as a
                # `ground_z` and hardcode the contact normal to (0,0,1), i.e. it modelled
                # every plane as a horizontal floor at the height of its origin. See
                # `collision/plane_frame.mojo`. The other geom is rebased into the PLANE'S
                # FRAME (`fp_*` / `fq_*`), where the plane really is z=0 with normal +z —
                # exactly what every `*_plane` primitive assumes — so the arithmetic below
                # is unchanged and only the contact point and normal map back to world.
                var plp_x = pj_x
                var plp_y = pj_y
                var plp_z = pj_z
                var plq_x = qj_x
                var plq_y = qj_y
                var plq_z = qj_z
                var plq_w = qj_w
                var pn = plane_world_normal[DTYPE](plq_x, plq_y, plq_z, plq_w)
                var lfp = to_plane_frame[DTYPE](
                    plp_x, plp_y, plp_z, plq_x, plq_y, plq_z, plq_w,
                    pi_x, pi_y, pi_z,
                )
                var lfq = quat_to_plane_frame[DTYPE](
                    plq_x, plq_y, plq_z, plq_w, qi_x, qi_y, qi_z, qi_w
                )
                var fp_x = lfp[0]
                var fp_y = lfp[1]
                var fp_z = lfp[2]
                var fq_x = lfq[0]
                var fq_y = lfq[1]
                var fq_z = lfq[2]
                var fq_w = lfq[3]
                var ground_z = Scalar[DTYPE](0)
                # PLANE-SIDE BOUNDING-SPHERE REJECT — MuJoCo's second
                # `mj_filterSphere` arm. In the plane's frame `fp_z` IS
                # `planeGeomDist`. Kept in step with the SAP path, which is
                # where it actually pays: `_plane_mesh_contacts` scans EVERY
                # hull vertex with no early out, so a floor a mesh never
                # touches costs its full vertex count every step.
                # ⚠ `+ contact_margin` — a geom hovering within its margin of
                # the floor is a contact MuJoCo reports; without the term it
                # vanishes silently. Gated by `plane_margin/{naive,sap}` in
                # `test_contact_pair_vs_mujoco.mojo`.
                # ⚠ Skipping the `_fill_pair_solparams` tail below is safe ONLY
                # because nothing has been emitted yet — it stamps the range
                # [_n0, num_contacts), which is empty here.
                var rb_pl_gi = rebind[Scalar[DTYPE]](
                    geoms[gi, GEOM_IDX_RBOUND]
                )
                if rb_pl_gi > Scalar[DTYPE](0) and fp_z > contact_margin + rb_pl_gi:
                    continue
                if gi_type == GEOM_CAPSULE:
                    # MuJoCo mjc_PlaneCapsule: test BOTH endpoints, up to 2 contacts
                    var axis_w = gpu_quat_rotate(
                        fq_x,
                        fq_y,
                        fq_z,
                        fq_w,
                        Scalar[DTYPE](0),
                        Scalar[DTYPE](0),
                        Scalar[DTYPE](1),
                    )
                    # `axis_w` is in the PLANE'S frame; the FRAME_T1 hint is
                    # read in WORLD space, so it goes back.
                    var axis_wd = gpu_quat_rotate(
                        plq_x, plq_y, plq_z, plq_w,
                        axis_w[0], axis_w[1], axis_w[2],
                    )
                    # Endpoint 1: center + half_length * axis
                    var e1_x = fp_x + hli * axis_w[0]
                    var e1_y = fp_y + hli * axis_w[1]
                    var e1_z = fp_z + hli * axis_w[2]
                    var dist1 = e1_z - ri - ground_z
                    if dist1 < contact_margin and num_contacts < max_contacts:
                        var c_off = num_contacts * CONTACT_SIZE
                        contacts[env, c_off + CONTACT_IDX_BODY_A] = Scalar[
                            DTYPE
                        ](gi_body)
                        contacts[env, c_off + CONTACT_IDX_BODY_B] = Scalar[
                            DTYPE
                        ](0)
                        var cw = from_plane_frame[DTYPE](
                            plp_x, plp_y, plp_z, plq_x, plq_y, plq_z, plq_w,
                            e1_x, e1_y,
                            ground_z + dist1 * Scalar[DTYPE](0.5),
                        )
                        contacts[env, c_off + CONTACT_IDX_POS_X] = cw[0]
                        contacts[env, c_off + CONTACT_IDX_POS_Y] = cw[1]
                        contacts[env, c_off + CONTACT_IDX_POS_Z] = cw[2]
                        contacts[env, c_off + CONTACT_IDX_NX] = pn[0]
                        contacts[env, c_off + CONTACT_IDX_NY] = pn[1]
                        contacts[env, c_off + CONTACT_IDX_NZ] = pn[2]
                        contacts[env, c_off + CONTACT_IDX_DIST] = dist1
                        contacts[
                            env, c_off + CONTACT_IDX_INCLUDEMARGIN
                        ] = contact_includemargin
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION
                        ] = contact_friction
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION_SPIN
                        ] = contact_friction_spin
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION_ROLL
                        ] = contact_friction_roll
                        contacts[env, c_off + CONTACT_IDX_CONDIM] = Scalar[
                            DTYPE
                        ](contact_condim)
                        contacts[env, c_off + CONTACT_IDX_FRAME_T1_X] = axis_wd[
                            0
                        ]
                        contacts[env, c_off + CONTACT_IDX_FRAME_T1_Y] = axis_wd[
                            1
                        ]
                        contacts[env, c_off + CONTACT_IDX_FRAME_T1_Z] = axis_wd[
                            2
                        ]
                        num_contacts += 1
                    # Endpoint 2: center - half_length * axis
                    var e2_x = fp_x - hli * axis_w[0]
                    var e2_y = fp_y - hli * axis_w[1]
                    var e2_z = fp_z - hli * axis_w[2]
                    var dist2 = e2_z - ri - ground_z
                    if dist2 < contact_margin and num_contacts < max_contacts:
                        var c_off = num_contacts * CONTACT_SIZE
                        contacts[env, c_off + CONTACT_IDX_BODY_A] = Scalar[
                            DTYPE
                        ](gi_body)
                        contacts[env, c_off + CONTACT_IDX_BODY_B] = Scalar[
                            DTYPE
                        ](0)
                        var cw = from_plane_frame[DTYPE](
                            plp_x, plp_y, plp_z, plq_x, plq_y, plq_z, plq_w,
                            e2_x, e2_y,
                            ground_z + dist2 * Scalar[DTYPE](0.5),
                        )
                        contacts[env, c_off + CONTACT_IDX_POS_X] = cw[0]
                        contacts[env, c_off + CONTACT_IDX_POS_Y] = cw[1]
                        contacts[env, c_off + CONTACT_IDX_POS_Z] = cw[2]
                        contacts[env, c_off + CONTACT_IDX_NX] = pn[0]
                        contacts[env, c_off + CONTACT_IDX_NY] = pn[1]
                        contacts[env, c_off + CONTACT_IDX_NZ] = pn[2]
                        contacts[env, c_off + CONTACT_IDX_DIST] = dist2
                        contacts[
                            env, c_off + CONTACT_IDX_INCLUDEMARGIN
                        ] = contact_includemargin
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION
                        ] = contact_friction
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION_SPIN
                        ] = contact_friction_spin
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION_ROLL
                        ] = contact_friction_roll
                        contacts[env, c_off + CONTACT_IDX_CONDIM] = Scalar[
                            DTYPE
                        ](contact_condim)
                        contacts[env, c_off + CONTACT_IDX_FRAME_T1_X] = axis_wd[
                            0
                        ]
                        contacts[env, c_off + CONTACT_IDX_FRAME_T1_Y] = axis_wd[
                            1
                        ]
                        contacts[env, c_off + CONTACT_IDX_FRAME_T1_Z] = axis_wd[
                            2
                        ]
                        num_contacts += 1
                elif gi_type == GEOM_CYLINDER:
                    # Up to FOUR points — two rim, two triangle — not
                    # one. See `_plane_cylinder_contacts`; a cylinder on
                    # its flat face needs a support polygon or it tips.
                    _plane_cylinder_contacts[DTYPE, BATCH](
                        env,
                        gi_body,
                        fp_x, fp_y, fp_z,
                        fq_x, fq_y, fq_z, fq_w,
                        ri,
                        hli,
                        ground_z,
                        plp_x, plp_y, plp_z,
                        plq_x, plq_y, plq_z, plq_w,
                        contact_margin,
                        contact_friction,
                        contact_friction_spin,
                        contact_friction_roll,
                        contact_condim,
                        0,
                        dims,
                        contacts,
                        num_contacts,
                        contact_gap,
                    )
                elif gi_type == GEOM_SPHERE:
                    var dist = fp_z - ri - ground_z
                    if dist < contact_margin and num_contacts < max_contacts:
                        var c_off = num_contacts * CONTACT_SIZE
                        contacts[env, c_off + CONTACT_IDX_BODY_A] = Scalar[
                            DTYPE
                        ](gi_body)
                        contacts[env, c_off + CONTACT_IDX_BODY_B] = Scalar[
                            DTYPE
                        ](0)
                        var cw = from_plane_frame[DTYPE](
                            plp_x, plp_y, plp_z, plq_x, plq_y, plq_z, plq_w,
                            fp_x, fp_y,
                            ground_z + dist * Scalar[DTYPE](0.5),
                        )
                        contacts[env, c_off + CONTACT_IDX_POS_X] = cw[0]
                        contacts[env, c_off + CONTACT_IDX_POS_Y] = cw[1]
                        contacts[env, c_off + CONTACT_IDX_POS_Z] = cw[2]
                        contacts[env, c_off + CONTACT_IDX_NX] = pn[0]
                        contacts[env, c_off + CONTACT_IDX_NY] = pn[1]
                        contacts[env, c_off + CONTACT_IDX_NZ] = pn[2]
                        contacts[env, c_off + CONTACT_IDX_DIST] = dist
                        contacts[
                            env, c_off + CONTACT_IDX_INCLUDEMARGIN
                        ] = contact_includemargin
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION
                        ] = contact_friction
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION_SPIN
                        ] = contact_friction_spin
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION_ROLL
                        ] = contact_friction_roll
                        contacts[env, c_off + CONTACT_IDX_CONDIM] = Scalar[
                            DTYPE
                        ](contact_condim)
                        num_contacts += 1
                elif gi_type == GEOM_ELLIPSOID:
                    # Mirror of the gj branch above; see it for why one
                    # contact is the whole story.
                    var ep = ellipsoid_plane[DTYPE](
                        fp_x, fp_y, fp_z,
                        fq_x, fq_y, fq_z, fq_w,
                        hxi, hyi, hzi,
                        ground_z,
                    )
                    var dist = ep[0]
                    if dist < contact_margin and num_contacts < max_contacts:
                        var c_off = num_contacts * CONTACT_SIZE
                        contacts[env, c_off + CONTACT_IDX_BODY_A] = Scalar[
                            DTYPE
                        ](gi_body)
                        contacts[env, c_off + CONTACT_IDX_BODY_B] = Scalar[
                            DTYPE
                        ](0)
                        var cw = from_plane_frame[DTYPE](
                            plp_x, plp_y, plp_z, plq_x, plq_y, plq_z, plq_w,
                            ep[1], ep[2], ep[3],
                        )
                        contacts[env, c_off + CONTACT_IDX_POS_X] = cw[0]
                        contacts[env, c_off + CONTACT_IDX_POS_Y] = cw[1]
                        contacts[env, c_off + CONTACT_IDX_POS_Z] = cw[2]
                        contacts[env, c_off + CONTACT_IDX_NX] = pn[0]
                        contacts[env, c_off + CONTACT_IDX_NY] = pn[1]
                        contacts[env, c_off + CONTACT_IDX_NZ] = pn[2]
                        contacts[env, c_off + CONTACT_IDX_DIST] = dist
                        contacts[
                            env, c_off + CONTACT_IDX_INCLUDEMARGIN
                        ] = contact_includemargin
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION
                        ] = contact_friction
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION_SPIN
                        ] = contact_friction_spin
                        contacts[
                            env, c_off + CONTACT_IDX_FRICTION_ROLL
                        ] = contact_friction_roll
                        contacts[env, c_off + CONTACT_IDX_CONDIM] = Scalar[
                            DTYPE
                        ](contact_condim)
                        num_contacts += 1
                elif gi_type == GEOM_BOX:
                    # Up to FOUR corners, not one — see
                    # `_plane_box_contacts` and task #42.
                    _plane_box_contacts[DTYPE](
                        env,
                        gi_body,
                        fp_x, fp_y, fp_z,
                        fq_x, fq_y, fq_z, fq_w,
                        hxi, hyi, hzi,
                        ground_z,
                        plp_x, plp_y, plp_z,
                        plq_x, plq_y, plq_z, plq_w,
                        contact_margin,
                        contact_friction,
                        contact_friction_spin,
                        contact_friction_roll,
                        contact_condim,
                        0,
                        dims,
                        contacts,
                        num_contacts,
                        contact_gap,
                    )
                elif gi_type == GEOM_MESH:
                    comptime if may_exist[D.NMESH_VERTS]():
                        var pmw = -1
                        _plane_mesh_contacts[
                            DTYPE,
                            0, False, True](
                            env,
                            gi,
                            gi_body,
                            fp_x,
                            fp_y,
                            fp_z,
                            fq_x,
                            fq_y,
                            fq_z,
                            fq_w,
                            ground_z,
                            plp_x, plp_y, plp_z,
                            plq_x, plq_y, plq_z, plq_w,
                            contact_margin,
                            contact_friction,
                            contact_friction_spin,
                            contact_friction_roll,
                            contact_condim,
                            dims,
                            geoms,
                            mesh_meta,
                            mesh_verts,
                            mesh_vert_edgeadr,
                            mesh_edges,
                            contacts,
                            num_contacts,
                            pmw,
                            contact_gap,
                        )
                _fill_pair_solparams[DTYPE](
                    env, _n0, num_contacts, _mx, contacts
                )
                continue

            # --- Non-plane geom pair ---
            var dist: Scalar[DTYPE] = 1.0
            var cx: Scalar[DTYPE] = 0
            var cy: Scalar[DTYPE] = 0
            var cz: Scalar[DTYPE] = 0
            var nx: Scalar[DTYPE] = 0
            var ny: Scalar[DTYPE] = 0
            var nz: Scalar[DTYPE] = 1
            # CONTACT DIRECTION INVARIANT — see the same note in
            # `broadphase_sap.mojo`. Every branch below emits
            # `normal = gi -> gj` with `body_a = gi_body, body_b = gj_body`.
            # The reversed-order branches negate the primitive's normal to get
            # there; they must NOT also swap the bodies, because the double
            # flip lands back on `body_b -> body_a` and desynchronises
            # `jar = aref + J*qacc` (aref is built from the penetration depth
            # and does not flip with the normal).
            var body_a = gi_body
            var body_b = gj_body
            # Mesh vertex ranges, hoisted out of the mesh branch so multi-CCD
            # can re-run the SAME convex query at its perturbed poses. Zero for
            # every non-mesh pair, which is what `gjk_epa` wants there.
            var va1 = 0
            var mnv1 = 0
            var va2 = 0
            var mnv2 = 0

            # ── HEIGHTFIELD, before every primitive pair ──────────────────
            #
            # `mjCOLLISIONFUNC`'s HFIELD row is `mjc_ConvexHField` against
            # every type but PLANE and HFIELD (`engine_collision_driver.c:48`)
            # — the two it leaves at 0 are the two that cannot bound a volume.
            # It writes its own records, one per prism, so it exits the loop
            # the way the capsule manifold does.
            if HFIELD_ENABLED and (
                gi_type == GEOM_HFIELD or gj_type == GEOM_HFIELD
            ):
                # PLANE x HFIELD and HFIELD x HFIELD are 0 in the table.
                if (
                    gi_type == GEOM_PLANE
                    or gj_type == GEOM_PLANE
                    or (gi_type == GEOM_HFIELD and gj_type == GEOM_HFIELD)
                ):
                    continue
                var hf_is_i = gi_type == GEOM_HFIELD
                var hf_g = gi if hf_is_i else gj
                var cx_g = gj if hf_is_i else gi
                var hid = Int(
                    rebind[Scalar[DTYPE]](geoms[hf_g, GEOM_IDX_HFIELD_ID])
                )
                if hid < 0:
                    continue
                # The convex geom's mesh range, if it has one.
                var cvm = Int(
                    rebind[Scalar[DTYPE]](geoms[cx_g, GEOM_IDX_MESH_ID])
                )
                var cva = 0
                var cmnv = 0
                if cvm >= 0:
                    cva = Int(rebind[Scalar[DTYPE]](mesh_meta[cvm, 0]))
                    cmnv = Int(rebind[Scalar[DTYPE]](mesh_meta[cvm, 1]))
                # ⚠ THE BODIES ARE NEVER SWAPPED — `body_a` is `gi_body`
                # whichever side the field is on, exactly as every other
                # branch in this loop. The normal's sign carries the
                # difference instead; see `_hfield_contacts`.
                var nsg = Scalar[DTYPE](-1) if hf_is_i else Scalar[DTYPE](1)
                _ = _hfield_contacts[DTYPE](
                    env, gi_body, gj_body, hid,
                    pi_x if hf_is_i else pj_x,
                    pi_y if hf_is_i else pj_y,
                    pi_z if hf_is_i else pj_z,
                    qi_x if hf_is_i else qj_x,
                    qi_y if hf_is_i else qj_y,
                    qi_z if hf_is_i else qj_z,
                    qi_w if hf_is_i else qj_w,
                    gj_type if hf_is_i else gi_type,
                    pj_x if hf_is_i else pi_x,
                    pj_y if hf_is_i else pi_y,
                    pj_z if hf_is_i else pi_z,
                    qj_x if hf_is_i else qi_x,
                    qj_y if hf_is_i else qi_y,
                    qj_z if hf_is_i else qi_z,
                    qj_w if hf_is_i else qi_w,
                    rj if hf_is_i else ri,
                    hlj if hf_is_i else hli,
                    hxj if hf_is_i else hxi,
                    hyj if hf_is_i else hyi,
                    hzj if hf_is_i else hzi,
                    rebind[Scalar[DTYPE]](geoms[cx_g, GEOM_IDX_RBOUND]),
                    cva, cmnv,
                    contact_margin,
                    contact_friction,
                    contact_friction_spin,
                    contact_friction_roll,
                    contact_condim,
                    nsg,
                    hfield_meta, hfield_data, dims.get_nhfield_data(),
                    mesh_verts, mesh_vert_edgeadr, mesh_edges,
                    dims, contacts, ws, num_contacts,
                    contact_gap,
                )
                _fill_pair_solparams[DTYPE](
                    env, _n0, num_contacts, _mx, contacts
                )
                continue

            if gi_type == GEOM_SPHERE and gj_type == GEOM_SPHERE:
                var r = sphere_sphere[DTYPE](
                    pi_x, pi_y, pi_z, ri, pj_x, pj_y, pj_z, rj
                )
                dist = r[0]
                cx = r[1]
                cy = r[2]
                cz = r[3]
                nx = r[4]
                ny = r[5]
                nz = r[6]
            elif gi_type == GEOM_CAPSULE and gj_type == GEOM_SPHERE:
                var r = capsule_sphere[DTYPE](
                    pi_x,
                    pi_y,
                    pi_z,
                    qi_x,
                    qi_y,
                    qi_z,
                    qi_w,
                    hli,
                    ri,
                    pj_x,
                    pj_y,
                    pj_z,
                    rj,
                )
                dist = r[0]
                cx = r[1]
                cy = r[2]
                cz = r[3]
                nx = r[4]
                ny = r[5]
                nz = r[6]
            elif gi_type == GEOM_SPHERE and gj_type == GEOM_CAPSULE:
                var r = capsule_sphere[DTYPE](
                    pj_x,
                    pj_y,
                    pj_z,
                    qj_x,
                    qj_y,
                    qj_z,
                    qj_w,
                    hlj,
                    rj,
                    pi_x,
                    pi_y,
                    pi_z,
                    ri,
                )
                dist = r[0]
                cx = r[1]
                cy = r[2]
                cz = r[3]
                nx = -r[4]
                ny = -r[5]
                nz = -r[6]
            elif gi_type == GEOM_CAPSULE and gj_type == GEOM_CAPSULE:
                # Two PARALLEL capsules touch over a segment, and MuJoCo emits
                # a point at each end — see `_capsule_capsule_contacts`, which
                # writes its own records.
                _ = _capsule_capsule_contacts[DTYPE](
                    env, gi_body, gj_body,
                    pi_x, pi_y, pi_z, qi_x, qi_y, qi_z, qi_w, hli, ri,
                    pj_x, pj_y, pj_z, qj_x, qj_y, qj_z, qj_w, hlj, rj,
                    contact_margin,
                    contact_friction,
                    contact_friction_spin,
                    contact_friction_roll,
                    contact_condim,
                    dims, contacts, num_contacts,
                    contact_gap,
                )
                _fill_pair_solparams[DTYPE](
                    env, _n0, num_contacts, _mx, contacts
                )
                continue
            elif gi_type == GEOM_BOX and gj_type == GEOM_SPHERE:
                var r = box_sphere[DTYPE](
                    pi_x,
                    pi_y,
                    pi_z,
                    qi_x,
                    qi_y,
                    qi_z,
                    qi_w,
                    hxi,
                    hyi,
                    hzi,
                    pj_x,
                    pj_y,
                    pj_z,
                    rj,
                )
                dist = r[0]
                cx = r[1]
                cy = r[2]
                cz = r[3]
                nx = r[4]
                ny = r[5]
                nz = r[6]
            elif gi_type == GEOM_SPHERE and gj_type == GEOM_BOX:
                var r = box_sphere[DTYPE](
                    pj_x,
                    pj_y,
                    pj_z,
                    qj_x,
                    qj_y,
                    qj_z,
                    qj_w,
                    hxj,
                    hyj,
                    hzj,
                    pi_x,
                    pi_y,
                    pi_z,
                    ri,
                )
                dist = r[0]
                cx = r[1]
                cy = r[2]
                cz = r[3]
                nx = -r[4]
                ny = -r[5]
                nz = -r[6]
            elif gi_type == GEOM_BOX and gj_type == GEOM_CAPSULE:
                # A capsule along a box face is a two-point manifold — see
                # `_capsule_box_contacts`, which writes its own records.
                _ = _capsule_box_contacts[DTYPE](
                    env, gi_body, gj_body,
                    pi_x, pi_y, pi_z, qi_x, qi_y, qi_z, qi_w, hxi, hyi, hzi,
                    pj_x, pj_y, pj_z, qj_x, qj_y, qj_z, qj_w, hlj, rj,
                    Scalar[DTYPE](-1),
                    contact_margin,
                    contact_friction,
                    contact_friction_spin,
                    contact_friction_roll,
                    contact_condim,
                    dims, contacts, num_contacts,
                    contact_gap,
                )
                _fill_pair_solparams[DTYPE](
                    env, _n0, num_contacts, _mx, contacts
                )
                continue
            elif gi_type == GEOM_CAPSULE and gj_type == GEOM_BOX:
                _ = _capsule_box_contacts[DTYPE](
                    env, gi_body, gj_body,
                    pj_x, pj_y, pj_z, qj_x, qj_y, qj_z, qj_w, hxj, hyj, hzj,
                    pi_x, pi_y, pi_z, qi_x, qi_y, qi_z, qi_w, hli, ri,
                    Scalar[DTYPE](1),
                    contact_margin,
                    contact_friction,
                    contact_friction_spin,
                    contact_friction_roll,
                    contact_condim,
                    dims, contacts, num_contacts,
                    contact_gap,
                )
                _fill_pair_solparams[DTYPE](
                    env, _n0, num_contacts, _mx, contacts
                )
                continue
            elif gi_type == GEOM_BOX and gj_type == GEOM_BOX:
                # A box/box contact is a whole manifold, not a point — see
                # `_box_box_contacts`. It writes its own records and this
                # branch is done; only a SEPARATED pair (code -1) falls through
                # to `box_box`, which then rejects it too.
                var code = _box_box_contacts[DTYPE](
                    env,
                    gi_body,
                    gj_body,
                    pi_x, pi_y, pi_z, qi_x, qi_y, qi_z, qi_w, hxi, hyi, hzi,
                    pj_x, pj_y, pj_z, qj_x, qj_y, qj_z, qj_w, hxj, hyj, hzj,
                    contact_margin,
                    contact_friction,
                    contact_friction_spin,
                    contact_friction_roll,
                    contact_condim,
                    dims,
                    contacts,
                    num_contacts,
                    contact_gap,
                )
                if code >= 0:
                    _fill_pair_solparams[DTYPE](
                        env, _n0, num_contacts, _mx, contacts
                    )
                    continue
                var r = box_box[DTYPE](
                    pi_x,
                    pi_y,
                    pi_z,
                    qi_x,
                    qi_y,
                    qi_z,
                    qi_w,
                    hxi,
                    hyi,
                    hzi,
                    pj_x,
                    pj_y,
                    pj_z,
                    qj_x,
                    qj_y,
                    qj_z,
                    qj_w,
                    hxj,
                    hyj,
                    hzj,
                )
                dist = r[0]
                cx = r[1]
                cy = r[2]
                cz = r[3]
                nx = r[4]
                ny = r[5]
                nz = r[6]
            elif gi_type == GEOM_CYLINDER and gj_type == GEOM_SPHERE:
                var r = cylinder_sphere[DTYPE](
                    pi_x,
                    pi_y,
                    pi_z,
                    qi_x,
                    qi_y,
                    qi_z,
                    qi_w,
                    hli,
                    ri,
                    pj_x,
                    pj_y,
                    pj_z,
                    rj,
                )
                dist = r[0]
                cx = r[1]
                cy = r[2]
                cz = r[3]
                nx = r[4]
                ny = r[5]
                nz = r[6]
            elif gi_type == GEOM_SPHERE and gj_type == GEOM_CYLINDER:
                var r = cylinder_sphere[DTYPE](
                    pj_x,
                    pj_y,
                    pj_z,
                    qj_x,
                    qj_y,
                    qj_z,
                    qj_w,
                    hlj,
                    rj,
                    pi_x,
                    pi_y,
                    pi_z,
                    ri,
                )
                dist = r[0]
                cx = r[1]
                cy = r[2]
                cz = r[3]
                nx = -r[4]
                ny = -r[5]
                nz = -r[6]
            elif (
                (gi_type == GEOM_CYLINDER and gj_type == GEOM_BOX)
                or (gi_type == GEOM_BOX and gj_type == GEOM_CYLINDER)
                or (gi_type == GEOM_CYLINDER and gj_type == GEOM_CAPSULE)
                or (gi_type == GEOM_CAPSULE and gj_type == GEOM_CYLINDER)
                or (gi_type == GEOM_CYLINDER and gj_type == GEOM_CYLINDER)
                # ⚠ EVERY ELLIPSOID PAIR EXCEPT PLANE. Row ELLIPSOID of
                # `mjCOLLISIONFUNC` is `mjc_Convex` against ELLIPSOID,
                # CYLINDER, BOX and MESH, and column ELLIPSOID is `mjc_Convex`
                # from SPHERE and CAPSULE down — only `mjc_PlaneConvex` is a
                # separate path, and it has its own loop above. Before this
                # branch existed those pairs fell through to nothing at all,
                # because `_support` returns a geom's CENTRE for a type it
                # does not know: an ellipsoid collided as a zero-radius dot.
                # flybody's two labrum ellipsoids are the case in Menagerie —
                # MuJoCo has them in contact at the model's own keyframe.
                # (ELLIPSOID x MESH is caught by the mesh branch below, which
                # also goes through the same support function.)
                or (gi_type == GEOM_ELLIPSOID and gj_type != GEOM_MESH)
                or (gj_type == GEOM_ELLIPSOID and gi_type != GEOM_MESH)
            ):
                # ⚠⚠ EVERY CYLINDER PAIR EXCEPT SPHERE AND PLANE COMES HERE,
                # and MuJoCo's own table is why: row CYLINDER of
                # `mjCOLLISIONFUNC` (`engine_collision_driver.c:52`) is
                # `mjc_Convex` against CYLINDER, BOX and MESH, and column
                # CYLINDER is `mjc_Convex` from CAPSULE and ELLIPSOID down.
                # Only `mjc_SphereCylinder` and `mjc_PlaneCylinder` are real
                # primitives. CAPSULE x CYLINDER and CYLINDER x CYLINDER were
                # still going to `cylinder_capsule` / `cylinder_cylinder`,
                # which compute `dist = axis_axis_distance - r1 - r2` — the
                # CAPSULE-capsule formula. That rounds the cylinder's flat end
                # caps into hemispheres, so the surface bulges a full radius
                # past where it is.
                #
                # CYLINDER x BOX came here first, for the same reason: it used
                # `cylinder_box`, the same capsule reduction. Measured against
                # the analytic depth that is an error of exactly -r in EVERY
                # configuration, separated or penetrating:
                # at 1 cm of CLEARANCE it still reported a 4 cm penetration. On
                # sawyer (obj r = 0.02) it manufactured a 2 cm contact at the
                # canonical reset pose, where MuJoCo has none and where all 13
                # Phase 7 manipulation tasks begin.
                #
                # ⚠ THIS RE-ROUTE WAS ATTEMPTED ONCE BEFORE AND REVERTED. It
                # dropped contacts at SHALLOW penetration in the RIM
                # configuration, because GJK handed EPA a 2-simplex that did
                # not enclose the origin. `gjkIntersect` (`4b773bdf`) is what
                # made it viable; without that commit this branch is wrong.
                #
                # One branch for both orderings: `cylinder_box` needed two
                # because the primitive is asymmetric in its operands, but the
                # convex query is symmetric and returns `gi -> gj` either way.
                var r = gjk_epa[DTYPE](
                    gi_type,
                    pi_x, pi_y, pi_z, qi_x, qi_y, qi_z, qi_w,
                    ri, hli, hxi, hyi, hzi,
                    mesh_verts, mesh_vert_edgeadr, mesh_edges, 0, 0,
                    gj_type,
                    pj_x, pj_y, pj_z, qj_x, qj_y, qj_z, qj_w,
                    rj, hlj, hxj, hyj, hzj,
                    0, 0,
                    ws, env,
                    ccd_tol, ccd_iter, contact_margin,
                    dist_cutoff=contact_margin,
                )
                dist = r[0]
                cx = r[1]
                cy = r[2]
                cz = r[3]
                nx = r[4]
                ny = r[5]
                nz = r[6]

            # GJK/EPA fallback for any pair involving a mesh geom
            elif gi_type == GEOM_MESH or gj_type == GEOM_MESH:
                comptime if may_exist[D.NMESH_VERTS]():
                    # Read mesh IDs from geom data
                    var mi_id = Int(
                        rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_MESH_ID])
                    )
                    var mj_id = Int(
                        rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_MESH_ID])
                    )
                    # Resolve mesh vertex ranges from mesh_meta records
                    if mi_id >= 0:
                        va1 = Int(rebind[Scalar[DTYPE]](mesh_meta[mi_id, 0]))
                        mnv1 = Int(rebind[Scalar[DTYPE]](mesh_meta[mi_id, 1]))
                    if mj_id >= 0:
                        va2 = Int(rebind[Scalar[DTYPE]](mesh_meta[mj_id, 0]))
                        mnv2 = Int(rebind[Scalar[DTYPE]](mesh_meta[mj_id, 1]))

                    # NATIVE MULTI-CONTACT. `maxContacts` returns 4 when BOTH
                    # geoms are box-or-mesh and neither carries a margin, and
                    # `mjc_Convex` then clips the two contacting face polygons
                    # instead of returning one point. See
                    # `collision/native_multicontact.mojo`; BOX x BOX is not
                    # here because MuJoCo sends it to `mjc_BoxBox`.
                    # `MC_ENABLED` sits LAST because it is a comptime
                    # `True`: on the left it folds and the compiler flags the
                    # rest of the chain unreachable. Every other operand is a
                    # pure comparison, so the order is not observable.
                    var mc_pair = (
                        (gi_type == GEOM_MESH or gi_type == GEOM_BOX)
                        and (gj_type == GEOM_MESH or gj_type == GEOM_BOX)
                        and contact_margin <= Scalar[DTYPE](0)
                        and MC_ENABLED
                    )
                    var wf1 = Array[Scalar[DTYPE], 9](
                        fill=Scalar[DTYPE](0)
                    )
                    var wf2 = Array[Scalar[DTYPE], 9](
                        fill=Scalar[DTYPE](0)
                    )
                    var wxx = Array[Scalar[DTYPE], 6](
                        fill=Scalar[DTYPE](0)
                    )
                    var wf_ok = 0
                    var wfi = Array[Int, 6](fill=-1)
                    var result = gjk_epa_witness[DTYPE](
                        gi_type,
                        pi_x, pi_y, pi_z, qi_x, qi_y, qi_z, qi_w,
                        ri, hli, hxi, hyi, hzi,
                        mesh_verts, mesh_vert_edgeadr, mesh_edges, va1, mnv1,
                        gj_type,
                        pj_x, pj_y, pj_z, qj_x, qj_y, qj_z, qj_w,
                        rj, hlj, hxj, hyj, hzj,
                        va2, mnv2,
                        wf1, wf2, wxx, wf_ok, wfi,
                        ws, env,
                        ccd_tol, ccd_iter, contact_margin,
                        # Same opt-in as the SAP path — the two narrow phases
                        # must move together (`feedback_sap_path_missing_a_whole_geom_type`).
                        contact_margin,
                    )
                    dist = result[0]
                    cx = result[1]
                    cy = result[2]
                    cz = result[3]
                    nx = result[4]
                    ny = result[5]
                    nz = result[6]
                    body_a = gi_body
                    body_b = gj_body

                    if (
                        mc_pair
                        and wf_ok == 1
                        and dist < contact_margin
                        and num_contacts < max_contacts
                    ):
                        var pa1 = 0
                        var pn1 = 0
                        var pa2 = 0
                        var pn2 = 0
                        if mi_id >= 0:
                            pa1 = Int(rebind[Scalar[DTYPE]](
                                mesh_meta[mi_id, MESH_META_IDX_POLYADR]
                            ))
                            pn1 = Int(rebind[Scalar[DTYPE]](
                                mesh_meta[mi_id, MESH_META_IDX_POLYNUM]
                            ))
                        if mj_id >= 0:
                            pa2 = Int(rebind[Scalar[DTYPE]](
                                mesh_meta[mj_id, MESH_META_IDX_POLYADR]
                            ))
                            pn2 = Int(rebind[Scalar[DTYPE]](
                                mesh_meta[mj_id, MESH_META_IDX_POLYNUM]
                            ))
                        # ⚠ THE OPERANDS ARE ALREADY MuJoCo'S — `(gi, gj)` is
                        # `pushPairArena`'s pair, sorted by (type, geom index)
                        # at the top of this loop — so the manifold clips from
                        # the SAME query GJK just ran, which is the reference's
                        # own structure: `mjc_Convex` hands `multicontact` the
                        # `status` of its own `mjc_ccd`.
                        #
                        # ⚠⚠ THERE USED TO BE A SECOND, LOCAL SWAP HERE AND IT
                        # WAS HALF A FIX. It ordered the MANIFOLD and left
                        # GJK on whatever order the loop emitted, so
                        # `wf1`/`wf2`/`wx` — the witness the manifold clips
                        # from — came out of a query in the OTHER order and had
                        # to be re-swapped to compensate. Canonicalising where
                        # the pair is named makes that predicate always false.
                        var mcn = native_multicontact_contacts[
                            DTYPE](
                            env, body_a, body_b,
                            gi_type,
                            pi_x, pi_y, pi_z, qi_x, qi_y, qi_z, qi_w,
                            hxi, hyi, hzi, rbound_i, va1, mnv1, pa1, pn1,
                            gj_type,
                            pj_x, pj_y, pj_z, qj_x, qj_y, qj_z, qj_w,
                            hxj, hyj, hzj, rbound_j, va2, mnv2, pa2, pn2,
                            dims,
                            mesh_verts, mesh_polys, mesh_polyvert,
                            mesh_polymap, mesh_vert_polymap,
                            wf1, wf2, wxx, wfi,
                            dist,
                            contact_margin,
                            contact_friction,
                            contact_friction_spin,
                            contact_friction_roll,
                            contact_condim,
                            False,
                            contacts, ws, env, num_contacts,
                            contact_gap,
                        )
                        # ⚠ THE MANIFOLD REPLACES THE POINT, it does not extend
                        # it — the reference overwrites `status->nx`. Falling
                        # through to the single-point emit as well would leave a
                        # fifth row on top of a four-row face.
                        if mcn > 0:
                            _fill_pair_solparams[
                                DTYPE](env, _n0, num_contacts, _mx, contacts)
                            continue
                else:
                    _fill_pair_solparams[DTYPE](
                        env, _n0, num_contacts, _mx, contacts
                    )
                    continue

            if dist < contact_margin and num_contacts < max_contacts:
                # The `gi -> gj` normal, captured BEFORE the emit negates it in
                # place — `multi_ccd_extra_contacts` re-runs the same query and
                # so works in the same convention the branches above produced.
                var mccd_nx = nx
                var mccd_ny = ny
                var mccd_nz = nz
                var mccd_first = num_contacts
                var c_off = num_contacts * CONTACT_SIZE
                contacts[env, c_off + CONTACT_IDX_BODY_A] = Scalar[DTYPE](
                    body_a
                )
                contacts[env, c_off + CONTACT_IDX_BODY_B] = Scalar[DTYPE](
                    body_b
                )
                contacts[env, c_off + CONTACT_IDX_POS_X] = cx
                contacts[env, c_off + CONTACT_IDX_POS_Y] = cy
                contacts[env, c_off + CONTACT_IDX_POS_Z] = cz
                # The record's normal points `body_b -> body_a`. Every branch
                # above computed `gi -> gj` with `body_a = gi`, so it is
                # negated here — UNCONDITIONALLY.
                #
                # ⚠ This used to be `if body_b > 0:`, which skipped the negation
                # whenever the second geom sat on the WORLD body and left those
                # contacts as `a -> b` while every other contact was `b -> a`.
                # Two conventions in one record, selected by a body id. Planes
                # are not affected either way — they have their own loop and
                # never reach this emit — so `body_b == 0` here means a
                # NON-PLANE world geom, which no shipped model currently has.
                # Latent, but it made body labels and normal direction
                # interdependent, and it nearly derailed the bug 35 fix.
                # Measured by `tests/physics3d/test_narrow_phase_pairs.mojo`'s
                # WORLD groups: a full 2.0 reversal on a unit vector.
                nx = -nx
                ny = -ny
                nz = -nz
                contacts[env, c_off + CONTACT_IDX_NX] = nx
                contacts[env, c_off + CONTACT_IDX_NY] = ny
                contacts[env, c_off + CONTACT_IDX_NZ] = nz
                contacts[env, c_off + CONTACT_IDX_DIST] = dist
                contacts[
                    env, c_off + CONTACT_IDX_INCLUDEMARGIN
                ] = contact_includemargin
                contacts[env, c_off + CONTACT_IDX_FRICTION] = contact_friction
                contacts[
                    env, c_off + CONTACT_IDX_FRICTION_SPIN
                ] = contact_friction_spin
                contacts[
                    env, c_off + CONTACT_IDX_FRICTION_ROLL
                ] = contact_friction_roll
                contacts[env, c_off + CONTACT_IDX_CONDIM] = Scalar[DTYPE](
                    contact_condim
                )
                num_contacts += 1

                # MULTI-POINT CONVEX CONTACT — defect 21. A single point cannot
                # hold a flat contact: the body rotates about it and sinks.
                # MuJoCo re-queries at four tilted poses and keeps the distinct
                # hits, which is what takes a cylinder resting on a box from 1
                # row to 5. `multi_ccd_pair_supported` is the guard; everything
                # it rejects — spheres, ellipsoids, meshes, the plane pairs
                # (which never reach this emit) — keeps the single point it
                # had, deliberately. See collision/multi_ccd.mojo.
                #
                # ⚠ `multiccd_off` IS THE MODEL'S OWN SWITCH, not a tuning
                # knob: `<flag multiccd="disable"/>` is `mjDSBL_MULTICCD`, and
                # a model that sets it gets single-point convex contacts from
                # MuJoCo. Ignoring it cost `reassemble5` 437 contacts against
                # the reference's 111. The SAP narrow phase carries the same
                # guard — see the note at its copy of this hook.
                if not multiccd_off and multi_ccd_pair_supported(
                    gi_type, gj_type, contact_margin > Scalar[DTYPE](0)
                ):
                    _ = multi_ccd_extra_contacts[
                        DTYPE](
                        env, body_a, body_b, mccd_first,
                        gi_type,
                        pi_x, pi_y, pi_z, qi_x, qi_y, qi_z, qi_w,
                        ri, hli, hxi, hyi, hzi, rbound_i, va1, mnv1,
                        gj_type,
                        pj_x, pj_y, pj_z, qj_x, qj_y, qj_z, qj_w,
                        rj, hlj, hxj, hyj, hzj, rbound_j, va2, mnv2,
                        dims,
                        mesh_verts,
                        mesh_vert_edgeadr,
                        mesh_edges,
                        cx, cy, cz,
                        mccd_nx, mccd_ny, mccd_nz,
                        dist,
                        contact_margin,
                        contact_friction,
                        contact_friction_spin,
                        contact_friction_roll,
                        contact_condim,
                        contacts, num_contacts,
                        ws, env,
                        ccd_tol, ccd_iter, contact_margin,
                        contact_gap,
                    )

            _fill_pair_solparams[DTYPE](
                env, _n0, num_contacts, _mx, contacts
            )

    # ── MuJoCo's contact ORDER — see `collision/contact_order.mojo` ────────
    # ⚠ APPLIED HERE TOO, and that is the point. `detect_contacts_auto`'s
    # docstring records that the two broadphases emit in DIFFERENT orders and
    # warns against swapping one for the other in a bit-exact pipeline. Both
    # now end in the same canonical order, so that split is closed rather than
    # documented.
    sort_contacts_mujoco_order[DTYPE](env, contacts, num_contacts)

    smeta[env, META_IDX_NUM_CONTACTS] = Scalar[DTYPE](num_contacts)


def _detect_contacts_fields_kernel[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    NGEOM: Int,
    NEXCLUDE: Int,
    NMESH_VERTS: Int,
    BATCH: Int,
    # Appended rather than grouped with NEXCLUDE — see `fields.Model`.
    NPAIR: Int,
    NHFIELD_DATA: Int,
](
    xpos: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 3), MutAnyOrigin
    ],
    xquat: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, NBODY * 4), MutAnyOrigin
    ],
    geoms: LayoutTensor[
        DTYPE, Layout.row_major(NGEOM, MODEL_GEOM_SIZE), MutAnyOrigin
    ],
    bodies: LayoutTensor[
        DTYPE, Layout.row_major(NBODY, MODEL_BODY_SIZE), MutAnyOrigin
    ],
    mmeta: LayoutTensor[
        DTYPE, Layout.row_major(MODEL_META_SIZE), MutAnyOrigin
    ],
    excludes: LayoutTensor[
        DTYPE, Layout.row_major(NEXCLUDE, 2), MutAnyOrigin
    ],
    pairs: LayoutTensor[
        DTYPE, Layout.row_major(NPAIR, MODEL_PAIR_SIZE), MutAnyOrigin
    ],
    mesh_meta: LayoutTensor[
        DTYPE,
        Layout.row_major(MAX_GPU_MESHES, MODEL_MESH_META_SIZE),
        MutAnyOrigin,
    ],
    mesh_verts: LayoutTensor[
        DTYPE, Layout.row_major(NMESH_VERTS, 3), MutAnyOrigin
    ],
    mesh_polys: LayoutTensor[
        DTYPE,
        Layout.row_major(mesh_max_poly(NMESH_VERTS), MODEL_MESH_POLY_SIZE),
        MutAnyOrigin,
    ],
    mesh_polyvert: LayoutTensor[
        DTYPE, Layout.row_major(mesh_max_polyvert(NMESH_VERTS)), MutAnyOrigin
    ],
    mesh_polymap: LayoutTensor[
        DTYPE, Layout.row_major(mesh_max_polyvert(NMESH_VERTS)), MutAnyOrigin
    ],
    mesh_vert_polymap: LayoutTensor[
        DTYPE, Layout.row_major(NMESH_VERTS, 2), MutAnyOrigin
    ],
    mesh_vert_edgeadr: LayoutTensor[
        DTYPE, Layout.row_major(NMESH_VERTS), MutAnyOrigin
    ],
    mesh_edges: LayoutTensor[
        DTYPE, Layout.row_major(mesh_max_edge(NMESH_VERTS)), MutAnyOrigin
    ],
    hfield_meta: LayoutTensor[
        DTYPE,
        Layout.row_major(MAX_GPU_HFIELDS * MODEL_HFIELD_META_SIZE),
        MutAnyOrigin,
    ],
    hfield_data: LayoutTensor[
        DTYPE, Layout.row_major(BATCH * NHFIELD_DATA), MutAnyOrigin
    ],
    contacts: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, MAX_CONTACTS * CONTACT_SIZE),
        MutAnyOrigin,
    ],
    smeta: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, METADATA_SIZE), MutAnyOrigin
    ],
    ccd_ws: LayoutTensor[
        DTYPE, Layout.row_major(BATCH * COLL_CCD_LANES, CCD_WS_SIZE), MutAnyOrigin
    ],
):
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    _detect_contacts_env[DTYPE, BATCH](
        env, Dims[nq=NQ, nv=NV, nbody=NBODY, njoint=NJOINT, max_contacts=MAX_CONTACTS, ngeom=NGEOM, nexclude=NEXCLUDE, nmesh_verts=NMESH_VERTS, npair=NPAIR](), xpos, xquat, geoms, bodies, mmeta, excludes, pairs, mesh_meta,
        mesh_verts, mesh_polys, mesh_polyvert, mesh_polymap,
        mesh_vert_polymap, mesh_vert_edgeadr, mesh_edges,
        hfield_meta, hfield_data, contacts, smeta, ccd_ws,
    )


def detect_contacts[target: StaticString, DTYPE: DType, D: DimsLike, BATCH: Int = 1](
    mut d: Data[DTYPE, D, BATCH],
    mut m: Model[DTYPE, D],
    ctx: Optional[DeviceContext] = None,
) raises:
    """Unified geom contact detection from FK products, both targets, one
    body. Reads `d.xpos`/`d.xquat` + geom/body/meta/exclude/mesh records;
    writes `d.contacts` + the ncon slot of `d.meta`."""
    comptime L_B3 = Layout.row_major(BATCH, D.NBODY * 3)
    comptime L_B4 = Layout.row_major(BATCH, D.NBODY * 4)
    comptime L_GEOM = Layout.row_major(D.NGEOM, MODEL_GEOM_SIZE)
    comptime L_BODY = Layout.row_major(D.NBODY, MODEL_BODY_SIZE)
    comptime L_MMETA = Layout.row_major(MODEL_META_SIZE)
    comptime L_EXCLUDE = Layout.row_major(D.NEXCLUDE, 2)
    comptime L_PAIR = Layout.row_major(D.NPAIR, MODEL_PAIR_SIZE)
    comptime L_MESH_META = Layout.row_major(
        MAX_GPU_MESHES, MODEL_MESH_META_SIZE
    )
    comptime L_MESH_VERT = Layout.row_major(D.NMESH_VERTS, 3)
    comptime L_MESH_POLY = Layout.row_major(
        mesh_max_poly(D.NMESH_VERTS), MODEL_MESH_POLY_SIZE
    )
    comptime L_MESH_POLYVERT = Layout.row_major(
        mesh_max_polyvert(D.NMESH_VERTS)
    )
    comptime L_MESH_VPMAP = Layout.row_major(D.NMESH_VERTS, 2)
    comptime L_MESH_VEADR = Layout.row_major(D.NMESH_VERTS)
    comptime L_MESH_EDGE = Layout.row_major(mesh_max_edge(D.NMESH_VERTS))
    comptime L_HF_META = Layout.row_major(
        MAX_GPU_HFIELDS * MODEL_HFIELD_META_SIZE
    )
    comptime L_HF_DATA = Layout.row_major(BATCH * _hf_len(D.NHFIELD_DATA))
    comptime L_CONTACTS = Layout.row_major(BATCH, D.MAX_CONTACTS * CONTACT_SIZE)
    comptime L_SMETA = Layout.row_major(BATCH, METADATA_SIZE)
    comptime L_CCD_WS = Layout.row_major(BATCH * COLL_CCD_LANES, CCD_WS_SIZE)

    comptime if target == "cpu":
        var dm = d.dims
        var rl_B3 = rl2(BATCH, dm.get_nbody() * 3)
        var rl_B4 = rl2(BATCH, dm.get_nbody() * 4)
        var rl_GEOM = rl2(dm.get_ngeom(), MODEL_GEOM_SIZE)
        var rl_BODY = rl2(dm.get_nbody(), MODEL_BODY_SIZE)
        var rl_MMETA = rl1(MODEL_META_SIZE)
        var rl_EXCLUDE = rl2(dm.get_nexclude(), 2)
        var rl_PAIR = rl2(dm.get_npair(), MODEL_PAIR_SIZE)
        var rl_MESH_META = rl2(MAX_GPU_MESHES, MODEL_MESH_META_SIZE)
        var rl_MESH_VERT = rl2(dm.get_nmesh_verts(), 3)
        var rl_MESH_POLY = rl2(mesh_max_poly(dm.get_nmesh_verts()), MODEL_MESH_POLY_SIZE)
        var rl_MESH_POLYVERT = rl1(mesh_max_polyvert(dm.get_nmesh_verts()))
        var rl_MESH_VPMAP = rl2(dm.get_nmesh_verts(), 2)
        var rl_MESH_VEADR = rl1(dm.get_nmesh_verts())
        var rl_MESH_EDGE = rl1(mesh_max_edge(dm.get_nmesh_verts()))
        var rl_HF_META = rl1(MAX_GPU_HFIELDS * MODEL_HFIELD_META_SIZE)
        var rl_HF_DATA = rl1(BATCH * _hf_len(dm.get_nhfield_data()))
        var rl_CONTACTS = rl2(BATCH, dm.get_max_contacts() * CONTACT_SIZE)
        var rl_SMETA = rl2(BATCH, METADATA_SIZE)
        var rl_CCD_WS = rl2(BATCH * COLL_CCD_LANES, CCD_WS_SIZE)
        var xpos_v = d.xpos.lt_dyn["cpu", DYN2](rl_B3)
        var xquat_v = d.xquat.lt_dyn["cpu", DYN2](rl_B4)
        var geoms_v = m.geoms.lt_dyn["cpu", DYN2](rl_GEOM)
        var bodies_v = m.bodies.lt_dyn["cpu", DYN2](rl_BODY)
        var mmeta_v = m.meta.lt_dyn["cpu", DYN1](rl_MMETA)
        var excludes_v = m.excludes.lt_dyn["cpu", DYN2](rl_EXCLUDE)
        var pairs_v = m.pairs.lt_dyn["cpu", DYN2](rl_PAIR)
        var mesh_meta_v = m.mesh_meta.lt_dyn["cpu", DYN2](rl_MESH_META)
        var mesh_verts_v = m.mesh_verts.lt_dyn["cpu", DYN2](rl_MESH_VERT)
        var mesh_polys_v = m.mesh_polys.lt_dyn["cpu", DYN2](rl_MESH_POLY)
        var mesh_polyvert_v = m.mesh_polyvert.lt_dyn["cpu", DYN1](rl_MESH_POLYVERT)
        var mesh_polymap_v = m.mesh_polymap.lt_dyn["cpu", DYN1](rl_MESH_POLYVERT)
        var mesh_vert_polymap_v = m.mesh_vert_polymap.lt_dyn[
            "cpu", DYN2
        ](rl_MESH_VPMAP)
        var mesh_vert_edgeadr_v = m.mesh_vert_edgeadr.lt_dyn[
            "cpu", DYN1
        ](rl_MESH_VEADR)
        var mesh_edges_v = m.mesh_edges.lt_dyn["cpu", DYN1](rl_MESH_EDGE)
        var hfield_meta_v = m.hfield_meta.lt_dyn["cpu", DYN1](rl_HF_META)
        var hfield_data_v = d.hfield_data.lt_dyn["cpu", DYN1](rl_HF_DATA)
        var contacts_v = d.contacts.lt_dyn["cpu", DYN2](rl_CONTACTS)
        var smeta_v = d.meta.lt_dyn["cpu", DYN2](rl_SMETA)
        var ccd_ws_v = d.ccd_ws.lt_dyn["cpu", DYN2](rl_CCD_WS)
        for e in range(BATCH):
            _detect_contacts_env[DTYPE, BATCH](
                e, dm, xpos_v, xquat_v, geoms_v, bodies_v, mmeta_v,
                excludes_v, pairs_v, mesh_meta_v, mesh_verts_v, mesh_polys_v,
                mesh_polyvert_v, mesh_polymap_v, mesh_vert_polymap_v,
                mesh_vert_edgeadr_v, mesh_edges_v,
                hfield_meta_v, hfield_data_v,
                contacts_v, smeta_v, ccd_ws_v,
            )
    else:
        var c = ctx.value()
        comptime BLOCKS = (BATCH + CD_TPB - 1) // CD_TPB
        c.enqueue_function[
            _detect_contacts_fields_kernel[
                DTYPE, D.NQ, D.NV, D.NBODY, D.NJOINT, D.MAX_CONTACTS, D.NGEOM,
                D.NEXCLUDE, D.NMESH_VERTS, BATCH, D.NPAIR,
                _hf_len(D.NHFIELD_DATA),
            ]
        ](
            d.xpos.lt["gpu", L_B3](),
            d.xquat.lt["gpu", L_B4](),
            m.geoms.lt["gpu", L_GEOM](),
            m.bodies.lt["gpu", L_BODY](),
            m.meta.lt["gpu", L_MMETA](),
            m.excludes.lt["gpu", L_EXCLUDE](),
            m.pairs.lt["gpu", L_PAIR](),
            m.mesh_meta.lt["gpu", L_MESH_META](),
            m.mesh_verts.lt["gpu", L_MESH_VERT](),
            m.mesh_polys.lt["gpu", L_MESH_POLY](),
            m.mesh_polyvert.lt["gpu", L_MESH_POLYVERT](),
            m.mesh_polymap.lt["gpu", L_MESH_POLYVERT](),
            m.mesh_vert_polymap.lt["gpu", L_MESH_VPMAP](),
            m.mesh_vert_edgeadr.lt["gpu", L_MESH_VEADR](),
            m.mesh_edges.lt["gpu", L_MESH_EDGE](),
            m.hfield_meta.lt["gpu", L_HF_META](),
            d.hfield_data.lt["gpu", L_HF_DATA](),
            d.contacts.lt["gpu", L_CONTACTS](),
            d.meta.lt["gpu", L_SMETA](),
            d.ccd_ws.lt["gpu", L_CCD_WS](),
            grid_dim=(BLOCKS,),
            block_dim=(CD_TPB,),
        )
