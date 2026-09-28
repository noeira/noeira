"""SAP broadphase contact detection over per-field tensors (migration P4).

Per-field port of `detect_contacts_sap_gpu` (collision/broadphase_sap.mojo)
— arithmetic, iteration order, insertion-sort order and branch structure
verbatim. Reads FK products (`d.xpos`, `d.xquat`) + geom/body records +
model meta + exclude pairs + mesh hulls; writes packed contact records into
`d.contacts` and the contact count into `d.meta` (META_IDX_NUM_CONTACTS).

Operands (10): xpos, xquat (data) + geoms, bodies, mmeta, excludes,
mesh_meta, mesh_verts (model) + contacts, smeta (data outputs). Mesh
collision (plane-mesh vertex scan + GJK/EPA fallback via gjk) is
compiled in only when NMESH_VERTS > 0; zero-mesh models keep the legacy
branch structure (mesh branches degrade to no-emission / `continue`).

NOTE: the legacy SAP kernel's contact conventions differ from
detect_contacts_gpu (plane contacts write BODY_B=-1 instead of 0, no
INCLUDEMARGIN slot, plane-mesh DIST is `dist_v - cm`); this port preserves
the SAP conventions verbatim — bit-exactness is gated against legacy SAP.

`detect_contacts_auto` mirrors `detect_contacts_auto_gpu`:
NGEOM >= SAP_THRESHOLD dispatches to SAP, else to `detect_contacts`.
The fields integrators are NOT rewired to auto here (SAP emission ORDER
differs from O(N^2), which would shift existing bit-exact gates)."""

from std.time import perf_counter_ns
from std.math import sqrt, abs
from std.bit import count_trailing_zeros
from std.sys.info import is_nvidia_gpu
from std.atomic import Atomic
from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

from ..kinematics.quat_math import gpu_quat_rotate
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
from ..fields import (
    Data,
    Model,
    Dims,
    DimsLike,
    AsStatic,
    may_exist,
    DIM_POISON,
    Scratch,
    cap,
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
    PAIR_IDX_MARGIN,
    PAIR_IDX_GAP,
    BODY_IDX_PARENT,
    BODY_IDX_WELDID,
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
    CONTACT_IDX_FRAME_T1_X,
    CONTACT_IDX_FRAME_T1_Y,
    CONTACT_IDX_FRAME_T1_Z,
    GEOM_IDX_TYPE,
    GEOM_IDX_BODY,
    GEOM_IDX_RADIUS,
    GEOM_IDX_RBOUND,
    GEOM_IDX_HALF_LENGTH,
    GEOM_IDX_HALF_X,
    GEOM_IDX_HALF_Y,
    GEOM_IDX_HALF_Z,
    GEOM_IDX_FRICTION,
    GEOM_IDX_CONTYPE,
    GEOM_IDX_CONAFFINITY,
    GEOM_IDX_CONDIM,
    GEOM_IDX_FRICTION_SPIN,
    GEOM_IDX_FRICTION_ROLL,
    GEOM_IDX_MARGIN,
    GEOM_IDX_GAP,
    GEOM_IDX_PRIORITY,
    GEOM_IDX_SOLREF_0,
    GEOM_IDX_SOLREF_1,
    GEOM_IDX_SOLIMP_0,
    GEOM_IDX_SOLIMP_1,
    GEOM_IDX_SOLIMP_2,
    GEOM_IDX_SOLIMP_3,
    GEOM_IDX_SOLIMP_4,
    CONTACT_IDX_SOLREF_0,
    CONTACT_IDX_SOLREF_1,
    CONTACT_IDX_SOLIMP_0,
    CONTACT_IDX_SOLIMP_1,
    CONTACT_IDX_SOLIMP_2,
    CONTACT_IDX_SOLIMP_3,
    CONTACT_IDX_SOLIMP_4,
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
)
from .collision_primitives import (
    sphere_sphere,
    capsule_sphere,
    box_sphere,
    box_box,
    box_plane,
    cylinder_plane,
    cylinder_sphere,
    cylinder_capsule,
    cylinder_cylinder,
    cylinder_box,
    ellipsoid_plane,
)
from .plane_frame import (
    plane_world_normal,
    to_plane_frame,
    from_plane_frame,
    quat_to_plane_frame,
)
@always_inline
def _hf_len(n: Int) -> Int:
    """`Model.hfield_data` is allocated with `_at_least_one`, so a model with
    no heightfield still has ONE element. A `Layout.row_major(0)` over it is a
    zero-size view the runtime rejects; every other tensor here is sized by a
    dimension that is never legitimately zero."""
    return n if n > 0 else 1


from .ccd_workspace import (
    COLL_FLAT_NARROW, COLL_FLAT_HOT_NS, coll_flat_words,
    CF_NCAND, CF_OVERFLOW, CF_FULL, CF_NHOT, CF_NCOLD, CF_CAND, CF_CNT,
    CF_COST, CF_ROW, CF_G_HDR,
    CCD_WS_SIZE, COLL_TPB, COLL_CCD_LANES, COLL_NCAND_CAP, COLL_STAGE_MAXC,
    HILL_WARM_ACROSS_STEPS, HILL_WARM_SLOTS, HW_WS_OFF,
    COLL_STAGE_SLOTS, COLL_BLOCK_KERNEL, COLL_NO_FALLBACK,
    COLL_CAND_REPORT, COLL_REPORT_HDR, COLL_REPORT_WORDS, COLL_PREFILTER,
)
from max.gpu.sync import barrier
from max.gpu.memory import AddressSpace
from .gjk import gjk_epa, gjk_epa_witness
from .multi_ccd import multi_ccd_pair_supported, multi_ccd_extra_contacts
from .native_multicontact import (
    native_multicontact_contacts,
    MC_ENABLED,
)
from .contact_order import sort_contacts_mujoco_order
from .contact_detection import (
    _plane_mesh_contacts,
    mix_contact_params,
    pair_body_filtered,
    exclude_signatures,
    find_predefined_pair,
    pair_params,
    _fill_pair_solparams,
    _plane_box_contacts,
    _plane_cylinder_contacts,
    _box_box_contacts,
    _capsule_box_contacts,
    _capsule_capsule_contacts,
    _hfield_contacts,
    _geom_world_pos,
    detect_contacts,
)

# SAP broadphase activation threshold + AABB helper (relocated here at the P6
# legacy sunset; formerly imported from the deleted legacy `broadphase_sap`).
comptime SAP_THRESHOLD: Int = 16

comptime SAP_TPB: Int = 64

# ⚠ A STAGE PROBE FOR THE CPU NARROW PHASE, off and free by default (the twin
# of `newton_solve._CPU_PROBE` / `euler._EULER_PROBE`). On, every call of
# `_detect_contacts_sap_env` prints one `[cprobe]` line: nanoseconds before
# the pair loop (`broad`), the pair loop's own overhead (`other`: filters,
# AABB and bounding-sphere rejects), then `<kind> ns calls` per narrow-phase
# routine. Every timer block sits at the indentation of the statement it
# wraps and holds only timer lines (PERFORMANCE.md §13.16).
comptime _COLL_PROBE: Bool = False

# ⚠ A PRICING KNOB FOR THE GPU COLLISION KERNEL, BIT-IDENTICAL AT EVERY VALUE.
# `_COLL_PROBE` is CPU-only (`perf_counter_ns`), so the share of the per-env
# GPU chain that GJK/EPA takes cannot be read from it; on the k=13 park scene
# the CPU phase is 10.3 us with GJK at 45% (4 calls), while the GPU kernel
# spends ~430 us per env — a serial thread walking hull adjacency one
# dependent global load at a time. This repeats every GJK/EPA call this many
# times with THROWAWAY outputs (the real call runs last and overwrites the
# workspace row it shares), so `t(R) - t(1)` over `R - 1` is one GJK's cost
# on the device — the `NEWTON_SERIAL_PROBE` pattern. 1 = production.
comptime _COLL_REPEAT_GJK: Int = 1


def _aabb_half_extents[
    DTYPE: DType
](
    geom_type: Int,
    qx: Scalar[DTYPE],
    qy: Scalar[DTYPE],
    qz: Scalar[DTYPE],
    qw: Scalar[DTYPE],
    radius: Scalar[DTYPE],
    half_length: Scalar[DTYPE],
    half_x: Scalar[DTYPE],
    half_y: Scalar[DTYPE],
    half_z: Scalar[DTYPE],
    rbound: Scalar[DTYPE],
) -> Tuple[Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE]]:
    """Return (ex, ey, ez) — the AABB half-extents for one geom in world space.

    The world-space AABB is [center - e, center + e] on each axis.
    Planes are not handled here (they use infinite bounds, handled separately).

    ⚠⚠ AN UNDER-BOUNDED AABB IS A MISSING CONTACT, SILENTLY. The pair never
    reaches the narrow phase, so every downstream check agrees that there is
    nothing there. `rbound` is therefore the FALLBACK for any type without a
    tight formula here — it is the geom's own bounding-sphere radius, which is
    correct for every type by construction, where `radius` is `size[0]` and
    means something different for each of them. The ellipsoid case below is
    exactly that bug: `size[0]` is the x semi-axis, and flybody's labrum
    ellipsoids are `0.0035 0.00875 0.0131`, so their AABB came out 3.7x too
    small on z and the pair was dropped before `mjc_Convex` ever ran. ⚠ The
    naive path has no AABB stage at all, which is why the same model collided
    correctly under 16 geoms and not over it.
    """
    if geom_type == GEOM_SPHERE:
        return (radius, radius, radius)

    if geom_type == GEOM_CAPSULE or geom_type == GEOM_CYLINDER:
        # World-space capsule/cylinder axis = rotate local Z (0,0,1) by quat.
        # Derivation: v' = (2(qx*qz+qy*qw), 2(qy*qz-qx*qw), 1-2(qx²+qy²))
        var two = Scalar[DTYPE](2)
        var ax = two * (qx * qz + qy * qw)
        var ay = two * (qy * qz - qx * qw)
        var az = Scalar[DTYPE](1) - two * (qx * qx + qy * qy)
        return (
            abs(ax) * half_length + radius,
            abs(ay) * half_length + radius,
            abs(az) * half_length + radius,
        )

    if geom_type == GEOM_BOX:
        # Tight AABB via rotation matrix: half_extent[k] = Σ |R[k][j]| * half[j]
        var two = Scalar[DTYPE](2)
        var r00 = Scalar[DTYPE](1) - two * (qy * qy + qz * qz)
        var r01 = two * (qx * qy - qz * qw)
        var r02 = two * (qx * qz + qy * qw)
        var r10 = two * (qx * qy + qz * qw)
        var r11 = Scalar[DTYPE](1) - two * (qx * qx + qz * qz)
        var r12 = two * (qy * qz - qx * qw)
        var r20 = two * (qx * qz - qy * qw)
        var r21 = two * (qy * qz + qx * qw)
        var r22 = Scalar[DTYPE](1) - two * (qx * qx + qy * qy)
        var ex = abs(r00) * half_x + abs(r01) * half_y + abs(r02) * half_z
        var ey = abs(r10) * half_x + abs(r11) * half_y + abs(r12) * half_z
        var ez = abs(r20) * half_x + abs(r21) * half_y + abs(r22) * half_z
        return (ex, ey, ez)

    if geom_type == GEOM_ELLIPSOID:
        # ⚠ NOT the box formula. The support of an ellipsoid along a world
        # axis is the 2-NORM of that row of `R * diag(a, b, c)`, not its
        # 1-norm; using the box's sum would still bound it, but loosely.
        var two = Scalar[DTYPE](2)
        var r00 = Scalar[DTYPE](1) - two * (qy * qy + qz * qz)
        var r01 = two * (qx * qy - qz * qw)
        var r02 = two * (qx * qz + qy * qw)
        var r10 = two * (qx * qy + qz * qw)
        var r11 = Scalar[DTYPE](1) - two * (qx * qx + qz * qz)
        var r12 = two * (qy * qz - qx * qw)
        var r20 = two * (qx * qz - qy * qw)
        var r21 = two * (qy * qz + qx * qw)
        var r22 = Scalar[DTYPE](1) - two * (qx * qx + qy * qy)
        var ax = r00 * half_x
        var ay = r01 * half_y
        var az = r02 * half_z
        var bx = r10 * half_x
        var by = r11 * half_y
        var bz = r12 * half_z
        var cx = r20 * half_x
        var cy = r21 * half_y
        var cz = r22 * half_z
        return (
            sqrt(ax * ax + ay * ay + az * az),
            sqrt(bx * bx + by * by + bz * bz),
            sqrt(cx * cx + cy * cy + cz * cz),
        )

    if geom_type == GEOM_HFIELD:
        # A HEIGHTFIELD is a box, and its bounding sphere is uselessly large:
        # barkour's field is 20 x 20 x 0.15 m, so `rbound` is 14.1 and the
        # sphere's z half-extent alone would pair the ground with every geom
        # in the model.
        #
        # ⚠ THE Z EXTENT IS RECOVERED FROM `rbound`, WHICH IS NOT AS OBSCURE AS
        # IT LOOKS. `mjCGeom::GetRBound` is
        # `sqrt(rx^2 + ry^2 + max(elev, base)^2)` and this function already has
        # `rx`/`ry` in `half_x`/`half_y`, so the remaining term is exactly the
        # z half-extent MuJoCo's own `geom_aabb` uses. Storing it in a geom
        # slot of its own would be a cleaner spelling and costs a slot.
        #
        # ⚠ IT IS DELIBERATELY SYMMETRIC AND MuJoCo'S IS NOT: its AABB runs
        # from `-base` to `+elevation`. Taking the LARGER of the two on both
        # sides is a strict SUPERSET, so no pair is ever missed — this is a
        # broadphase bound, and being loose costs an early-out in the narrow
        # phase while being tight in the wrong direction costs a contact.
        var t2 = rbound * rbound - half_x * half_x - half_y * half_y
        var hz = sqrt(t2) if t2 > Scalar[DTYPE](0) else Scalar[DTYPE](0)
        var two = Scalar[DTYPE](2)
        var h00 = Scalar[DTYPE](1) - two * (qy * qy + qz * qz)
        var h01 = two * (qx * qy - qz * qw)
        var h02 = two * (qx * qz + qy * qw)
        var h10 = two * (qx * qy + qz * qw)
        var h11 = Scalar[DTYPE](1) - two * (qx * qx + qz * qz)
        var h12 = two * (qy * qz - qx * qw)
        var h20 = two * (qx * qz - qy * qw)
        var h21 = two * (qy * qz + qx * qw)
        var h22 = Scalar[DTYPE](1) - two * (qx * qx + qy * qy)
        return (
            abs(h00) * half_x + abs(h01) * half_y + abs(h02) * hz,
            abs(h10) * half_x + abs(h11) * half_y + abs(h12) * hz,
            abs(h20) * half_x + abs(h21) * half_y + abs(h22) * hz,
        )

    # Any other type — MESH above all — gets its BOUNDING SPHERE, which is
    # what `rbound_of` and the mesh hull loader already compute and store.
    # This used to be `radius`, i.e. `size[0]`.
    return (rbound, rbound, rbound)



struct _SapProbe(Copyable, Movable):
    """`_COLL_PROBE`'s accumulators, one struct so the per-pair narrow phase
    (`_sap_pair_narrow`) can carry them as ONE `mut` argument. Every field is
    an `Int` and every read/write sits under `comptime if _COLL_PROBE`, so at
    the production value the struct is dead weight the compiler drops."""

    var _c_t0: Int
    var _c_start: Int
    var _c_loop0: Int
    var _n_gjkhit: Int
    var _c_cas: Int
    var _n_cas: Int
    var _c_bs: Int
    var _n_bs: Int
    var _c_bbf: Int
    var _n_bbf: Int
    var _c_cys: Int
    var _n_cys: Int
    var _c_gjkp: Int
    var _n_gjkp: Int
    var _c_pbf: Int
    var _n_pbf: Int
    var _c_ppair: Int
    var _n_ppair: Int
    var _c_pparm: Int
    var _n_pparm: Int
    var _n_pairs: Int
    var _n_aabb: Int
    var _c_pcyl: Int
    var _n_pcyl: Int
    var _c_pbox: Int
    var _n_pbox: Int
    var _c_pmesh: Int
    var _n_pmesh: Int
    var _c_hf: Int
    var _n_hf: Int
    var _c_ss: Int
    var _n_ss: Int
    var _c_cc: Int
    var _n_cc: Int
    var _c_cb: Int
    var _n_cb: Int
    var _c_bb: Int
    var _n_bb: Int
    var _c_gjk: Int
    var _n_gjk: Int
    var _c_mcn: Int
    var _n_mcn: Int
    var _c_mccd: Int
    var _n_mccd: Int

    def __init__(out self):
        self._c_t0 = 0
        self._c_start = 0
        self._c_loop0 = 0
        self._n_gjkhit = 0
        self._c_cas = 0
        self._n_cas = 0
        self._c_bs = 0
        self._n_bs = 0
        self._c_bbf = 0
        self._n_bbf = 0
        self._c_cys = 0
        self._n_cys = 0
        self._c_gjkp = 0
        self._n_gjkp = 0
        self._c_pbf = 0
        self._n_pbf = 0
        self._c_ppair = 0
        self._n_ppair = 0
        self._c_pparm = 0
        self._n_pparm = 0
        self._n_pairs = 0
        self._n_aabb = 0
        self._c_pcyl = 0
        self._n_pcyl = 0
        self._c_pbox = 0
        self._n_pbox = 0
        self._c_pmesh = 0
        self._n_pmesh = 0
        self._c_hf = 0
        self._n_hf = 0
        self._c_ss = 0
        self._n_ss = 0
        self._c_cc = 0
        self._n_cc = 0
        self._c_cb = 0
        self._n_cb = 0
        self._c_bb = 0
        self._n_bb = 0
        self._c_gjk = 0
        self._n_gjk = 0
        self._c_mcn = 0
        self._n_mcn = 0
        self._c_mccd = 0
        self._n_mccd = 0



struct _PlaneGate[DTYPE: DType](Copyable, Movable):
    """What `_sap_plane_gate` decided and computed on the way: `ok` = the
    candidate survives every filter and the bounding reject, plus the
    values the narrow phase needs next (the pair index, the margins, the
    geom's pose in the plane's frame)."""
    var ok: Bool
    var gj_type: Int
    var gj_body: Int
    var ipair: Int
    var cim: Scalar[Self.DTYPE]
    var cgp: Scalar[Self.DTYPE]
    var cm: Scalar[Self.DTYPE]
    var pj_x: Scalar[Self.DTYPE]
    var pj_y: Scalar[Self.DTYPE]
    var pj_z: Scalar[Self.DTYPE]
    var qj_x: Scalar[Self.DTYPE]
    var qj_y: Scalar[Self.DTYPE]
    var qj_z: Scalar[Self.DTYPE]
    var qj_w: Scalar[Self.DTYPE]

    def __init__(out self):
        """A rejected candidate."""
        self.ok = False
        self.gj_type = -1
        self.gj_body = -1
        self.ipair = -1
        self.cim = Scalar[Self.DTYPE](0)
        self.cgp = Scalar[Self.DTYPE](0)
        self.cm = Scalar[Self.DTYPE](0)
        self.pj_x = Scalar[Self.DTYPE](0)
        self.pj_y = Scalar[Self.DTYPE](0)
        self.pj_z = Scalar[Self.DTYPE](0)
        self.qj_x = Scalar[Self.DTYPE](0)
        self.qj_y = Scalar[Self.DTYPE](0)
        self.qj_z = Scalar[Self.DTYPE](0)
        self.qj_w = Scalar[Self.DTYPE](1)


@always_inline
def _sap_plane_gate[
    DTYPE: DType,
    BATCH: Int,
    D: DimsLike,
    EX_CAP: Int,
    L_GEOMS: Layout,
    L_BODIES: Layout,
    L_MMETA: Layout,
    L_EXCLUDES: Layout,
    L_PAIRS: Layout,
](
    gi: Int,
    gj: Int,
    gi_body: Int,
    gi_contype: Int,
    gi_conaffinity: Int,
    plp_x: Scalar[DTYPE],
    plp_y: Scalar[DTYPE],
    plp_z: Scalar[DTYPE],
    plq_x: Scalar[DTYPE],
    plq_y: Scalar[DTYPE],
    plq_z: Scalar[DTYPE],
    plq_w: Scalar[DTYPE],
    gj_px: Scalar[DTYPE],
    gj_py: Scalar[DTYPE],
    gj_pz: Scalar[DTYPE],
    gj_qx: Scalar[DTYPE],
    gj_qy: Scalar[DTYPE],
    gj_qz: Scalar[DTYPE],
    gj_qw: Scalar[DTYPE],
    dims: D,
    nbody: Int,
    ex_sig: Scratch[Int, EX_CAP],
    n_sig: Int,
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
) -> _PlaneGate[DTYPE]:
    """Everything of the plane phase that can REJECT a (plane, geom)
    candidate before any contact is computed, in the order the serial loop
    always ran it: the geom is itself a plane; no `<pair>` and (the world
    body, `pair_body_filtered`, the contype/conaffinity masks); then the
    bounding-sphere reject in the plane's frame with the pair's cutoff.
    Pure in the model and the two poses, so the block kernel's candidate
    list and the serial narrow phase take the SAME decision from the same
    inputs — a candidate the gate drops would have emitted nothing.

    ⚠ ONE RULE. `_sap_plane_narrow` used to hold these tests inline; the
    block kernel needs them BEFORE it hands a candidate to a thread (the
    plane phase lists every geom in the model), and a second copy of a
    filter is how `<geom gap>` went wrong fifteen times
    (`_a_rule_written_inline_twice_drifts`). Extracted 2026-09-11."""
    var gj_type = Int(
        rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_TYPE])
    )
    if gj_type == GEOM_PLANE:
        return _PlaneGate[DTYPE]()
    var gj_body = Int(
        rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_BODY])
    )
    # `<contact><pair>` bypasses every filter below — see the same
    # gate in `_detect_contacts_env`. A plane/geom pair is a normal
    # thing to declare (it is the ONLY form ToddlerBot's scene files
    # use), and the world plane's body is 0, so without this the
    # `gj_body == 0` skip and the weld test would drop it.
    var ipair = find_predefined_pair[DTYPE](
        gi, gj, dims, pairs, mmeta
    )
    if ipair < 0:
        if gj_body == 0:
            return _PlaneGate[DTYPE]()
        # DEFECT 24 — this loop had NO body filter. MuJoCo runs the
        # plane path through `filterBodyPair` like every other pair
        # (`engine_collision_driver.c:1277`), which discards on
        # `weldbody1 == weldbody2`; a jointless body welds to the
        # world, so every static geom was colliding with the ground
        # here while the O(N^2) path correctly emitted nothing. See
        # `pair_body_filtered`.
        if pair_body_filtered[DTYPE, EX_CAP=EX_CAP](
            gi_body, gj_body, bodies, mmeta, excludes,
            ex_sig, n_sig, nbody,
        ):
            return _PlaneGate[DTYPE]()
        var gj_contype = Int(
            rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_CONTYPE])
        )
        var gj_conaffinity = Int(
            rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_CONAFFINITY])
        )
        if (gi_contype & gj_conaffinity) == 0 and (
            gj_contype & gi_conaffinity
        ) == 0:
            return _PlaneGate[DTYPE]()

    # MuJoCo's full contact-parameter rule, PRIORITY FIRST — shared
    # with `detect_contacts` so the two paths cannot drift, which is
    # exactly how the SAP ellipsoid branch went missing. A predefined
    # pair supplies its own parameters instead, unmixed.
    var mgi = rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_MARGIN])
    var mgj = rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_MARGIN])
    # Sum of the two geoms' margins, or the PAIR's own — never both.
    var cim = mgi + mgj  # MuJoCo 3.5+: sum of margins
    var cgp = (
        rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_GAP])
        + rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_GAP])
    )
    if ipair >= 0:
        cim = rebind[Scalar[DTYPE]](pairs[ipair, PAIR_IDX_MARGIN])
        cgp = rebind[Scalar[DTYPE]](pairs[ipair, PAIR_IDX_GAP])
    # ⚠⚠ TWO VALUES, NOT ONE. `cm` is the narrowphase CUTOFF and `cim`
    # is what the contact stores as its `includemargin`; 3.10.0 passes
    # `margin + gap` to the collision function and `margin` alone to
    # `mj_setContact`, so a contact in [margin, margin+gap) is DETECTED
    # and then EXCLUDED from the solver by
    # `con->exclude = dist >= includemargin`. With no `<geom gap>` the
    # two are equal and every line below is what it always was.
    var cm = cim + cgp

    # Pose IN THE PLANE'S FRAME, so `ground_z` below is 0 and the
    # branch arithmetic is the same as it always was.
    var lpj = to_plane_frame[DTYPE](
        plp_x, plp_y, plp_z, plq_x, plq_y, plq_z, plq_w,
        gj_px, gj_py, gj_pz,
    )
    var lqj = quat_to_plane_frame[DTYPE](
        plq_x, plq_y, plq_z, plq_w,
        gj_qx, gj_qy, gj_qz, gj_qw,
    )
    var pj_x = lpj[0]
    var pj_y = lpj[1]
    var pj_z = lpj[2]
    var qj_x = lqj[0]
    var qj_y = lqj[1]
    var qj_z = lqj[2]
    var qj_w = lqj[3]

    # ── PLANE-SIDE BOUNDING-SPHERE REJECT — MuJoCo's second
    # `mj_filterSphere` arm. In the plane's own frame `pj_z` IS
    # `planeGeomDist`: the signed distance from the plane to the geom
    # centre. If the geom's bounding sphere cannot reach the plane,
    # nothing downstream can produce a contact.
    #
    # ⚠⚠ WITHOUT THIS, A PLANE PAIRED WITH A MESH SCANS EVERY HULL
    # VERTEX, EVERY STEP, FOREVER. `_plane_mesh_contacts` has no early
    # out — it transforms all `pm_vnum` vertices looking for the
    # deepest. SO-ARM101 carries 30 mesh geoms totalling 33 076 hull
    # vertices and a floor its arm never touches, and that scan was
    # 72% of its entire physics step. It is also why the arm-to-arm
    # cost ratio tracked HULL SIZE rather than anything physical.
    #
    #     SO-ARM101   1.86 -> 0.65 ms/env step   ( 539 -> 1544 Hz)
    #     SO-ARM100   1.11 -> 1.04 ms/env step   ( 901 ->  959 Hz)
    #
    # ⚠ THE TWO ARMS SEPARATE HERE, AND THAT IS THE POINT. SO-ARM100
    # barely moves: 2 551 hull vertices is a scan it could afford.
    # SO-ARM101's 33 076 is not, and removing it INVERTS the pair —
    # the arm with 13x the geometry is now the FASTER of the two,
    # because what remains is no longer proportional to hull size.
    # SO-ARM100's residual is elsewhere (its Newton solve is ~25% of
    # its step, against ~0.5% of SO-ARM101's).
    #
    # ⚠ `+ cm` AGAIN, for the same silent reason as the geom-geom arm
    # above: a geom hovering within its margin of the floor is a
    # contact MuJoCo reports.
    var rbound_j_pl = rebind[Scalar[DTYPE]](
        geoms[gj, GEOM_IDX_RBOUND]
    )
    if rbound_j_pl > Scalar[DTYPE](0) and pj_z > cm + rbound_j_pl:
        return _PlaneGate[DTYPE]()
    var out = _PlaneGate[DTYPE]()
    out.ok = True
    out.gj_type = gj_type
    out.gj_body = gj_body
    out.ipair = ipair
    out.cim = cim
    out.cgp = cgp
    out.cm = cm
    out.pj_x = pj_x
    out.pj_y = pj_y
    out.pj_z = pj_z
    out.qj_x = qj_x
    out.qj_y = qj_y
    out.qj_z = qj_z
    out.qj_w = qj_w
    return out^


@always_inline
def _sap_plane_narrow[
    DTYPE: DType,
    BATCH: Int,
    D: DimsLike,
    EX_CAP: Int,
    L_GEOMS: Layout,
    L_BODIES: Layout,
    L_MMETA: Layout,
    L_EXCLUDES: Layout,
    L_PAIRS: Layout,
    L_MESH_META: Layout,
    L_MESH_VERTS: Layout,
    L_MESH_VERT_EDGEADR: Layout,
    L_MESH_EDGES: Layout,
    L_CONTACTS: Layout,
    L_WS: Layout,
    HFIELD_ENABLED: Bool,
](
    env: Int,
    # The contact slab row and the CCD workspace row. The serial path
    # passes `env` for both; the block kernel gives each thread its own
    # staging window and CCD lane. `env` itself stays the per-env DATA
    # row (heightfield samples), which is why it is three parameters.
    crow: Int,
    wrow: Int,
    dims: D,
    gi: Int,
    gj: Int,
    gi_body: Int,
    gi_contype: Int,
    gi_conaffinity: Int,
    plp_x: Scalar[DTYPE],
    plp_y: Scalar[DTYPE],
    plp_z: Scalar[DTYPE],
    plq_x: Scalar[DTYPE],
    plq_y: Scalar[DTYPE],
    plq_z: Scalar[DTYPE],
    plq_w: Scalar[DTYPE],
    pn: Array[Scalar[DTYPE], 3],
    nbody: Int,
    max_contacts: Int,
    ex_sig: Scratch[Int, EX_CAP],
    n_sig: Int,
    mut pr: _SapProbe,
    mut num_contacts: Int,
    # `gj`'s world pose. The two narrow-phase routines used to take the
    # seven per-geom pose arrays and index them; they only ever read the
    # candidate's own geoms, and the block kernel paid a private copy of
    # all seven arrays PER THREAD to call them (7 x NGEOM floats of local
    # memory — the 255-register signature of PERFORMANCE.md §13.51).
    gj_px: Scalar[DTYPE],
    gj_py: Scalar[DTYPE],
    gj_pz: Scalar[DTYPE],
    gj_qx: Scalar[DTYPE],
    gj_qy: Scalar[DTYPE],
    gj_qz: Scalar[DTYPE],
    gj_qw: Scalar[DTYPE],
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
    ws: LayoutTensor[DTYPE, L_WS, MutAnyOrigin],
    # The warm-slot row (`gjk_epa_witness`); -1 = `wrow`.
    hw_row: Int = -1,
):
    """ONE (plane, non-plane geom) candidate of the plane phase — filters,
    contact parameters, the plane narrow phase and its emission — moved out
    of `_detect_contacts_sap_env`'s `gj` loop verbatim (2026-09-07), after
    the loop's `max_contacts` guard, which stays with the loop. Each
    `continue` is a `return`; nothing followed the body inside the loop.

    ⚠ THE CPU PATH CALLS THIS TOO — one plane narrow phase, both targets.

    ⚠ THE FILTERS AND THE BOUNDING REJECT LIVE IN `_sap_plane_gate`, which
    the block-per-env kernel also runs BEFORE it lists a plane candidate —
    the plane phase used to hand every geom of the model to a thread (74 on
    the G1, of which the gate keeps the dozen near the floor), and that
    alone overflowed `COLL_NCAND_CAP` and sent the env to the serial path.
    One rule, two callers; this routine keeps only what follows a pass."""
    var g8 = _sap_plane_gate[DTYPE, BATCH, D, EX_CAP](
        gi, gj, gi_body, gi_contype, gi_conaffinity,
        plp_x, plp_y, plp_z, plq_x, plq_y, plq_z, plq_w,
        gj_px, gj_py, gj_pz, gj_qx, gj_qy, gj_qz, gj_qw,
        dims, nbody, ex_sig, n_sig, geoms, bodies, mmeta, excludes, pairs,
    )
    if not g8.ok:
        return
    var gj_type = g8.gj_type
    var gj_body = g8.gj_body
    var ipair = g8.ipair
    var _n0 = num_contacts
    var cim = g8.cim
    var cgp = g8.cgp
    var cm = g8.cm
    var pj_x = g8.pj_x
    var pj_y = g8.pj_y
    var pj_z = g8.pj_z
    var qj_x = g8.qj_x
    var qj_y = g8.qj_y
    var qj_z = g8.qj_z
    var qj_w = g8.qj_w
    var ground_z = Scalar[DTYPE](0)
    # ⚠ MIXED AFTER THE REJECT, as the SAP pair loop below already
    # does (its note above `mix_contact_params`): the mix is ~30
    # tensor reads plus MuJoCo's priority/solref/solimp rules, for
    # every geom against every plane — 391 a step on dog, of which
    # the bounding test keeps ~15. Nothing above the test reads it.
    # Same values for every survivor: bit-exact (PERFORMANCE.md §13.26).
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
    var cdim = Int(_mx[0])
    var cf = _mx[1]
    var cfs = _mx[2]
    var cfr = _mx[3]

    var rj = rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_RADIUS])
    var hlj = rebind[Scalar[DTYPE]](
        geoms[gj, GEOM_IDX_HALF_LENGTH]
    )

    if gj_type == GEOM_SPHERE:
        var dist = pj_z - rj - ground_z
        if dist < cm and num_contacts < max_contacts:
            var c_off = num_contacts * CONTACT_SIZE
            contacts[crow, c_off + CONTACT_IDX_BODY_A] = Scalar[DTYPE](
                gj_body
            )
            contacts[crow, c_off + CONTACT_IDX_BODY_B] = Scalar[DTYPE](
                -1
            )
            var cw = from_plane_frame[DTYPE](
                plp_x, plp_y, plp_z, plq_x, plq_y, plq_z, plq_w,
                pj_x, pj_y,
                ground_z + dist * Scalar[DTYPE](0.5),
            )
            contacts[crow, c_off + CONTACT_IDX_POS_X] = cw[0]
            contacts[crow, c_off + CONTACT_IDX_POS_Y] = cw[1]
            contacts[crow, c_off + CONTACT_IDX_POS_Z] = cw[2]
            contacts[crow, c_off + CONTACT_IDX_NX] = pn[0]
            contacts[crow, c_off + CONTACT_IDX_NY] = pn[1]
            contacts[crow, c_off + CONTACT_IDX_NZ] = pn[2]
            contacts[crow, c_off + CONTACT_IDX_DIST] = dist
            contacts[crow, c_off + CONTACT_IDX_INCLUDEMARGIN] = cim
            contacts[crow, c_off + CONTACT_IDX_FRICTION] = cf
            contacts[crow, c_off + CONTACT_IDX_FRICTION_SPIN] = cfs
            contacts[crow, c_off + CONTACT_IDX_FRICTION_ROLL] = cfr
            contacts[crow, c_off + CONTACT_IDX_CONDIM] = Scalar[DTYPE](
                cdim
            )
            num_contacts += 1

    elif gj_type == GEOM_CAPSULE:
        var axis_w = gpu_quat_rotate(
            qj_x,
            qj_y,
            qj_z,
            qj_w,
            Scalar[DTYPE](0),
            Scalar[DTYPE](0),
            Scalar[DTYPE](1),
        )
        # `axis_w` is in the PLANE'S frame (qj_* were rebased above),
        # which is what the endpoint arithmetic below needs. The
        # FRAME_T1 hint written into the record is read in WORLD space,
        # so it goes back — see collision/contact_frame.mojo for what
        # that slot is and is not.
        var axis_wd = gpu_quat_rotate(
            plq_x, plq_y, plq_z, plq_w,
            axis_w[0], axis_w[1], axis_w[2],
        )
        var e1_x = pj_x + hlj * axis_w[0]
        var e1_y = pj_y + hlj * axis_w[1]
        var e1_z = pj_z + hlj * axis_w[2]
        var dist1 = e1_z - rj - ground_z
        if dist1 < cm and num_contacts < max_contacts:
            var c_off = num_contacts * CONTACT_SIZE
            contacts[crow, c_off + CONTACT_IDX_BODY_A] = Scalar[DTYPE](
                gj_body
            )
            contacts[crow, c_off + CONTACT_IDX_BODY_B] = Scalar[DTYPE](
                -1
            )
            var cw = from_plane_frame[DTYPE](
                plp_x, plp_y, plp_z, plq_x, plq_y, plq_z, plq_w,
                e1_x, e1_y,
                ground_z + dist1 * Scalar[DTYPE](0.5),
            )
            contacts[crow, c_off + CONTACT_IDX_POS_X] = cw[0]
            contacts[crow, c_off + CONTACT_IDX_POS_Y] = cw[1]
            contacts[crow, c_off + CONTACT_IDX_POS_Z] = cw[2]
            contacts[crow, c_off + CONTACT_IDX_NX] = pn[0]
            contacts[crow, c_off + CONTACT_IDX_NY] = pn[1]
            contacts[crow, c_off + CONTACT_IDX_NZ] = pn[2]
            contacts[crow, c_off + CONTACT_IDX_DIST] = dist1
            contacts[crow, c_off + CONTACT_IDX_INCLUDEMARGIN] = cim
            contacts[crow, c_off + CONTACT_IDX_FRICTION] = cf
            contacts[crow, c_off + CONTACT_IDX_FRICTION_SPIN] = cfs
            contacts[crow, c_off + CONTACT_IDX_FRICTION_ROLL] = cfr
            contacts[crow, c_off + CONTACT_IDX_CONDIM] = Scalar[DTYPE](
                cdim
            )
            contacts[crow, c_off + CONTACT_IDX_FRAME_T1_X] = axis_wd[0]
            contacts[crow, c_off + CONTACT_IDX_FRAME_T1_Y] = axis_wd[1]
            contacts[crow, c_off + CONTACT_IDX_FRAME_T1_Z] = axis_wd[2]
            num_contacts += 1
        var e2_x = pj_x - hlj * axis_w[0]
        var e2_y = pj_y - hlj * axis_w[1]
        var e2_z = pj_z - hlj * axis_w[2]
        var dist2 = e2_z - rj - ground_z
        if dist2 < cm and num_contacts < max_contacts:
            var c_off = num_contacts * CONTACT_SIZE
            contacts[crow, c_off + CONTACT_IDX_BODY_A] = Scalar[DTYPE](
                gj_body
            )
            contacts[crow, c_off + CONTACT_IDX_BODY_B] = Scalar[DTYPE](
                -1
            )
            var cw = from_plane_frame[DTYPE](
                plp_x, plp_y, plp_z, plq_x, plq_y, plq_z, plq_w,
                e2_x, e2_y,
                ground_z + dist2 * Scalar[DTYPE](0.5),
            )
            contacts[crow, c_off + CONTACT_IDX_POS_X] = cw[0]
            contacts[crow, c_off + CONTACT_IDX_POS_Y] = cw[1]
            contacts[crow, c_off + CONTACT_IDX_POS_Z] = cw[2]
            contacts[crow, c_off + CONTACT_IDX_NX] = pn[0]
            contacts[crow, c_off + CONTACT_IDX_NY] = pn[1]
            contacts[crow, c_off + CONTACT_IDX_NZ] = pn[2]
            contacts[crow, c_off + CONTACT_IDX_DIST] = dist2
            contacts[crow, c_off + CONTACT_IDX_INCLUDEMARGIN] = cim
            contacts[crow, c_off + CONTACT_IDX_FRICTION] = cf
            contacts[crow, c_off + CONTACT_IDX_FRICTION_SPIN] = cfs
            contacts[crow, c_off + CONTACT_IDX_FRICTION_ROLL] = cfr
            contacts[crow, c_off + CONTACT_IDX_CONDIM] = Scalar[DTYPE](
                cdim
            )
            contacts[crow, c_off + CONTACT_IDX_FRAME_T1_X] = axis_wd[0]
            contacts[crow, c_off + CONTACT_IDX_FRAME_T1_Y] = axis_wd[1]
            contacts[crow, c_off + CONTACT_IDX_FRAME_T1_Z] = axis_wd[2]
            num_contacts += 1

    elif gj_type == GEOM_CYLINDER:
        # Up to FOUR points — two rim, two triangle — not one.
        # See `_plane_cylinder_contacts` in contact_detection.mojo;
        # shared with the naive path so the two cannot drift, which
        # is exactly how the ellipsoid branch below went missing.
        comptime if _COLL_PROBE:
            pr._c_t0 = Int(perf_counter_ns())
        _plane_cylinder_contacts[DTYPE, BATCH](
            crow,
            gj_body,
            pj_x, pj_y, pj_z,
            qj_x, qj_y, qj_z, qj_w,
            rj,
            hlj,
            ground_z,
            plp_x, plp_y, plp_z,
            plq_x, plq_y, plq_z, plq_w,
            cm,
            cf,
            cfs,
            cfr,
            cdim,
            -1,
            dims,
            contacts,
            num_contacts,
            cgp,
            max_contacts_in=max_contacts,
        )
        comptime if _COLL_PROBE:
            pr._c_pcyl += Int(perf_counter_ns()) - pr._c_t0
            pr._n_pcyl += 1

    elif gj_type == GEOM_ELLIPSOID:
        # ⚠ ADDED 2026-08-03. This branch did not exist, and
        # `broadphase_sap.mojo` contained no mention of ELLIPSOID at
        # all, so every ellipsoid geom was INVISIBLE TO COLLISION in
        # any model that takes the SAP path — `detect_contacts_auto`
        # switches to SAP at ngeom >= 16, and nothing warns.
        #
        # Shipped and silently wrong at the time of the fix:
        #   quadruped     26 geoms, SAP, ellipsoid = `torso`
        #   humanoid_CMU  50 geoms, SAP, ellipsoids = `lhand`, `rhand`
        # i.e. the quadruped's TORSO never collided with the floor.
        # fish (12 geoms, 7 ellipsoids) and swimmer (7, 1) sit under
        # the threshold and take the naive path, which is why the
        # ellipsoid narrow phase looked exercised.
        #
        # It hid because no test compared a plane's contact SET
        # against MuJoCo — every plane in the suite is an axis-aligned
        # floor and every gate read qacc at poses where the ellipsoid
        # was not touching it. `test_oriented_plane_vs_mujoco` is what
        # found it, and it found it on the AXIS-ALIGNED control rather
        # than the tilted case it was written for.
        #
        # Body id is -1 here where `detect_contacts` writes 0 — the
        # documented split between the two emit paths.
        var hxje = rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_HALF_X])
        var hyje = rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_HALF_Y])
        var hzje = rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_HALF_Z])
        # MuJoCo routes plane x ellipsoid through `mjc_PlaneConvex`,
        # which reports the single deepest support point; a smooth
        # strictly-convex surface meets a plane at one point, so unlike
        # the box there is no second contact to look for.
        var epe = ellipsoid_plane[DTYPE](
            pj_x, pj_y, pj_z,
            qj_x, qj_y, qj_z, qj_w,
            hxje, hyje, hzje,
            ground_z,
        )
        var diste = epe[0]
        if diste < cm and num_contacts < max_contacts:
            var c_off = num_contacts * CONTACT_SIZE
            contacts[crow, c_off + CONTACT_IDX_BODY_A] = Scalar[DTYPE](
                gj_body
            )
            contacts[crow, c_off + CONTACT_IDX_BODY_B] = Scalar[DTYPE](
                -1
            )
            # `ellipsoid_plane` already returns the contact point in
            # the PLANE frame, including the half-depth offset, so
            # unlike the sphere branch there is nothing to add here.
            var cwe = from_plane_frame[DTYPE](
                plp_x, plp_y, plp_z, plq_x, plq_y, plq_z, plq_w,
                epe[1], epe[2], epe[3],
            )
            contacts[crow, c_off + CONTACT_IDX_POS_X] = cwe[0]
            contacts[crow, c_off + CONTACT_IDX_POS_Y] = cwe[1]
            contacts[crow, c_off + CONTACT_IDX_POS_Z] = cwe[2]
            contacts[crow, c_off + CONTACT_IDX_NX] = pn[0]
            contacts[crow, c_off + CONTACT_IDX_NY] = pn[1]
            contacts[crow, c_off + CONTACT_IDX_NZ] = pn[2]
            contacts[crow, c_off + CONTACT_IDX_DIST] = diste
            contacts[crow, c_off + CONTACT_IDX_INCLUDEMARGIN] = cim
            contacts[crow, c_off + CONTACT_IDX_FRICTION] = cf
            contacts[crow, c_off + CONTACT_IDX_FRICTION_SPIN] = cfs
            contacts[crow, c_off + CONTACT_IDX_FRICTION_ROLL] = cfr
            contacts[crow, c_off + CONTACT_IDX_CONDIM] = Scalar[DTYPE](
                cdim
            )
            num_contacts += 1

    elif gj_type == GEOM_BOX:
        var hxj = rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_HALF_X])
        var hyj = rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_HALF_Y])
        var hzj = rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_HALF_Z])
        # Up to FOUR corners, not one — see `_plane_box_contacts` and
        # task #42. ⚠ This path writes -1 for the world body where
        # `detect_contacts` writes 0, hence the explicit argument.
        comptime if _COLL_PROBE:
            pr._c_t0 = Int(perf_counter_ns())
        _plane_box_contacts[DTYPE](
            crow,
            gj_body,
            pj_x, pj_y, pj_z,
            qj_x, qj_y, qj_z, qj_w,
            hxj, hyj, hzj,
            ground_z,
            plp_x, plp_y, plp_z,
            plq_x, plq_y, plq_z, plq_w,
            cm,
            cf,
            cfs,
            cfr,
            cdim,
            -1,
            dims,
            contacts,
            num_contacts,
            cgp,
            max_contacts_in=max_contacts,
        )
        comptime if _COLL_PROBE:
            pr._c_pbox += Int(perf_counter_ns()) - pr._c_t0
            pr._n_pbox += 1

    elif gj_type == GEOM_MESH:
        # Plane-mesh. Was a verbatim copy of the O(N^2) path's vertex
        # scan, and carried the same defect: one contact per hull
        # vertex, uncapped. Both now go through the single
        # `_plane_mesh_contacts`, so the fix cannot land on one path
        # and miss the other — the duplication is what let a
        # `maxplanemesh` cap be absent from BOTH for as long as it was.
        #
        # ⚠ SAP'S RECORD CONVENTIONS ARE PRESERVED, NOT UNIFIED: this
        # path writes BODY_B = -1, stores `dist - margin` in DIST and
        # has no INCLUDEMARGIN slot (see the module docstring). Those
        # are gated bit-exactly elsewhere, so they are passed as
        # parameters rather than quietly aligned with the other path.
        comptime if may_exist[D.NMESH_VERTS]():
            comptime if _COLL_PROBE:
                pr._c_t0 = Int(perf_counter_ns())
            # The pair's cross-step warm slot (ccd_workspace.mojo), as for
            # the GJK pairs: the plane's lowest vertex last step is this
            # step's answer.
            var pm_slot: Int
            comptime if HILL_WARM_ACROSS_STEPS:
                pm_slot = (gi * 131 + gj) % HILL_WARM_SLOTS
            else:
                pm_slot = -1
            var pmw = -1
            var hrow = hw_row if hw_row >= 0 else wrow
            if pm_slot >= 0:
                var pf = rebind[Scalar[DTYPE]](ws[hrow, HW_WS_OFF + 2 * pm_slot])
                if pf >= Scalar[DTYPE](0) and pf < Scalar[DTYPE](1e8):
                    pmw = Int(pf)
            _plane_mesh_contacts[
                DTYPE,
                -1, True, False](
                crow,
                gj,
                gj_body,
                pj_x, pj_y, pj_z,
                qj_x, qj_y, qj_z, qj_w,
                ground_z,
                plp_x, plp_y, plp_z,
                plq_x, plq_y, plq_z, plq_w,
                cm,
                cf,
                cfs,
                cfr,
                cdim,
                dims,
                geoms,
                mesh_meta,
                mesh_verts,
                mesh_vert_edgeadr,
                mesh_edges,
                contacts,
                num_contacts,
                pmw,
                cgp,
                max_contacts_in=max_contacts,
            )
            if pm_slot >= 0:
                ws[hrow, HW_WS_OFF + 2 * pm_slot] = Scalar[DTYPE](pmw)
            comptime if _COLL_PROBE:
                pr._c_pmesh += Int(perf_counter_ns()) - pr._c_t0
                pr._n_pmesh += 1

    _fill_pair_solparams[DTYPE](
        crow, _n0, num_contacts, _mx, contacts
    )


@always_inline
def _obb_separated[
    DTYPE: DType
](
    pix: Scalar[DTYPE], piy: Scalar[DTYPE], piz: Scalar[DTYPE],
    qix: Scalar[DTYPE], qiy: Scalar[DTYPE], qiz: Scalar[DTYPE],
    qiw: Scalar[DTYPE],
    a0: Scalar[DTYPE], a1: Scalar[DTYPE], a2: Scalar[DTYPE],
    pjx: Scalar[DTYPE], pjy: Scalar[DTYPE], pjz: Scalar[DTYPE],
    qjx: Scalar[DTYPE], qjy: Scalar[DTYPE], qjz: Scalar[DTYPE],
    qjw: Scalar[DTYPE],
    b0: Scalar[DTYPE], b1: Scalar[DTYPE], b2: Scalar[DTYPE],
) -> Bool:
    """True when two oriented boxes are provably apart — MuJoCo's
    `mj_collideOBB` midphase test, the 15-axis separating-axis theorem in
    Ericson's form (Real-Time Collision Detection 4.4.1). Box i has
    half-sizes `a*` and box j `b*`, each centred on its geom frame; the
    caller inflates `a*` by the pair's cutoff, which is conservative (an
    inflated box contains the margin-inflated shape).

    Written out with scalars: a per-thread array indexed at runtime is the
    Metal miscompile of `feedback_metal_wide_per_thread_inlinearray_miscompute`.
    """
    comptime ONE = Scalar[DTYPE](1)
    comptime TWO = Scalar[DTYPE](2)
    # Column k of each rotation is that box's k-th axis in the world.
    var A00 = ONE - TWO * (qiy * qiy + qiz * qiz)
    var A10 = TWO * (qix * qiy + qiz * qiw)
    var A20 = TWO * (qix * qiz - qiy * qiw)
    var A01 = TWO * (qix * qiy - qiz * qiw)
    var A11 = ONE - TWO * (qix * qix + qiz * qiz)
    var A21 = TWO * (qiy * qiz + qix * qiw)
    var A02 = TWO * (qix * qiz + qiy * qiw)
    var A12 = TWO * (qiy * qiz - qix * qiw)
    var A22 = ONE - TWO * (qix * qix + qiy * qiy)
    var B00 = ONE - TWO * (qjy * qjy + qjz * qjz)
    var B10 = TWO * (qjx * qjy + qjz * qjw)
    var B20 = TWO * (qjx * qjz - qjy * qjw)
    var B01 = TWO * (qjx * qjy - qjz * qjw)
    var B11 = ONE - TWO * (qjx * qjx + qjz * qjz)
    var B21 = TWO * (qjy * qjz + qjx * qjw)
    var B02 = TWO * (qjx * qjz + qjy * qjw)
    var B12 = TWO * (qjy * qjz - qjx * qjw)
    var B22 = ONE - TWO * (qjx * qjx + qjy * qjy)
    # R[r][c] = A_r . B_c ; t = the centre offset in A's frame.
    var R00 = A00 * B00 + A10 * B10 + A20 * B20
    var R01 = A00 * B01 + A10 * B11 + A20 * B21
    var R02 = A00 * B02 + A10 * B12 + A20 * B22
    var R10 = A01 * B00 + A11 * B10 + A21 * B20
    var R11 = A01 * B01 + A11 * B11 + A21 * B21
    var R12 = A01 * B02 + A11 * B12 + A21 * B22
    var R20 = A02 * B00 + A12 * B10 + A22 * B20
    var R21 = A02 * B01 + A12 * B11 + A22 * B21
    var R22 = A02 * B02 + A12 * B12 + A22 * B22
    var dx = pjx - pix
    var dy = pjy - piy
    var dz = pjz - piz
    var t0 = A00 * dx + A10 * dy + A20 * dz
    var t1 = A01 * dx + A11 * dy + A21 * dz
    var t2 = A02 * dx + A12 * dy + A22 * dz
    # ⚠ THE EPSILON ONLY MAKES A REJECT HARDER. Near-parallel edges give a
    # cross-product axis of ~zero length, where rounding alone can fake a
    # gap; RTCD adds it to |R| for exactly that.
    comptime EPS = Scalar[DTYPE](1e-6)
    var E00 = abs(R00) + EPS
    var E01 = abs(R01) + EPS
    var E02 = abs(R02) + EPS
    var E10 = abs(R10) + EPS
    var E11 = abs(R11) + EPS
    var E12 = abs(R12) + EPS
    var E20 = abs(R20) + EPS
    var E21 = abs(R21) + EPS
    var E22 = abs(R22) + EPS

    @always_inline
    def apart(x: Scalar[DTYPE], r: Scalar[DTYPE]) -> Bool:
        # A relative slack on every test: rejecting a touching pair would drop
        # a contact, keeping an apart one only costs the narrow phase.
        return abs(x) > r * Scalar[DTYPE](1.0001) + Scalar[DTYPE](1e-6)

    # A's face axes.
    if apart(t0, a0 + b0 * E00 + b1 * E01 + b2 * E02):
        return True
    if apart(t1, a1 + b0 * E10 + b1 * E11 + b2 * E12):
        return True
    if apart(t2, a2 + b0 * E20 + b1 * E21 + b2 * E22):
        return True
    # B's face axes.
    if apart(t0 * R00 + t1 * R10 + t2 * R20, a0 * E00 + a1 * E10 + a2 * E20 + b0):
        return True
    if apart(t0 * R01 + t1 * R11 + t2 * R21, a0 * E01 + a1 * E11 + a2 * E21 + b1):
        return True
    if apart(t0 * R02 + t1 * R12 + t2 * R22, a0 * E02 + a1 * E12 + a2 * E22 + b2):
        return True
    # The nine edge-edge axes A_i x B_j.
    if apart(t2 * R10 - t1 * R20, a1 * E20 + a2 * E10 + b1 * E02 + b2 * E01):
        return True
    if apart(t2 * R11 - t1 * R21, a1 * E21 + a2 * E11 + b0 * E02 + b2 * E00):
        return True
    if apart(t2 * R12 - t1 * R22, a1 * E22 + a2 * E12 + b0 * E01 + b1 * E00):
        return True
    if apart(t0 * R20 - t2 * R00, a0 * E20 + a2 * E00 + b1 * E12 + b2 * E11):
        return True
    if apart(t0 * R21 - t2 * R01, a0 * E21 + a2 * E01 + b0 * E12 + b2 * E10):
        return True
    if apart(t0 * R22 - t2 * R02, a0 * E22 + a2 * E02 + b0 * E11 + b1 * E10):
        return True
    if apart(t1 * R00 - t0 * R10, a0 * E10 + a1 * E00 + b1 * E22 + b2 * E21):
        return True
    if apart(t1 * R01 - t0 * R11, a0 * E11 + a1 * E01 + b0 * E22 + b2 * E20):
        return True
    if apart(t1 * R02 - t0 * R12, a0 * E12 + a1 * E02 + b0 * E21 + b1 * E20):
        return True
    return False


@always_inline
def _sap_pair_filter_rejects[
    DTYPE: DType,
    EX_CAP: Int,
    L_BODIES: Layout,
    L_MMETA: Layout,
    L_EXCLUDES: Layout,
](
    gi_body: Int,
    gj_body: Int,
    gi_contype: Int,
    gi_conaffinity: Int,
    gj_contype: Int,
    gj_conaffinity: Int,
    bodies: LayoutTensor[DTYPE, L_BODIES, MutAnyOrigin],
    mmeta: LayoutTensor[DTYPE, L_MMETA, MutAnyOrigin],
    excludes: LayoutTensor[DTYPE, L_EXCLUDES, MutAnyOrigin],
    ex_sig: Scratch[Int, EX_CAP],
    n_sig: Int,
    nbody: Int,
) -> Bool:
    """True = the SAP pair path discards this NON-predefined pair before
    any geometry: `pair_body_filtered` (weld, weld-parent, exclude), then
    the contype/conaffinity mask. Pass the pair CANONICALISED as
    `_sap_pair_narrow` does. The narrow phase and the block kernel's listing
    (`_sap_pair_listable`) both decide through this — see the note at the
    narrow phase's call."""
    if pair_body_filtered[DTYPE, EX_CAP=EX_CAP](
        gi_body, gj_body, bodies, mmeta, excludes, ex_sig, n_sig, nbody,
    ):
        return True
    return (gi_contype & gj_conaffinity) == 0 and (
        gj_contype & gi_conaffinity
    ) == 0


@always_inline
def _sap_pair_narrow[
    DTYPE: DType,
    BATCH: Int,
    D: DimsLike,
    EX_CAP: Int,
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
    L_WS: Layout,
    HFIELD_ENABLED: Bool,
](
    env: Int,
    # The contact slab row and the CCD workspace row. The serial path
    # passes `env` for both; the block kernel gives each thread its own
    # staging window and CCD lane. `env` itself stays the per-env DATA
    # row (heightfield samples), which is why it is three parameters.
    crow: Int,
    wrow: Int,
    dims: D,
    si: Int,
    sj: Int,
    si_type: Int,
    nbody: Int,
    max_contacts: Int,
    ex_sig: Scratch[Int, EX_CAP],
    n_sig: Int,
    mut pr: _SapProbe,
    mut num_contacts: Int,
    # The two geoms' world poses, `si`'s then `sj`'s — in the SWEEP's
    # order; the canonical (gi, gj) below picks from them. See the same
    # note on `_sap_plane_narrow`: the routine reads only its own pair.
    si_px: Scalar[DTYPE],
    si_py: Scalar[DTYPE],
    si_pz: Scalar[DTYPE],
    si_qx: Scalar[DTYPE],
    si_qy: Scalar[DTYPE],
    si_qz: Scalar[DTYPE],
    si_qw: Scalar[DTYPE],
    sj_px: Scalar[DTYPE],
    sj_py: Scalar[DTYPE],
    sj_pz: Scalar[DTYPE],
    sj_qx: Scalar[DTYPE],
    sj_qy: Scalar[DTYPE],
    sj_qz: Scalar[DTYPE],
    sj_qw: Scalar[DTYPE],
    ccd_tol: Scalar[DTYPE],
    ccd_iter: Int,
    multiccd_off: Bool,
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
    ws: LayoutTensor[
        DTYPE, L_WS, MutAnyOrigin
    ],
    # The warm-slot row (`gjk_epa_witness`); -1 = `wrow`.
    hw_row: Int = -1,
):
    """ONE candidate geom pair of the SAP sweep — canonicalisation, filters,
    contact parameters, the narrow-phase dispatch and its emission — moved
    out of `_detect_contacts_sap_env`'s `j` loop verbatim (2026-09-07) so a
    block-per-env kernel can run candidates on separate threads. Every
    `continue` of the loop body is a `return` here; nothing followed the
    body inside the loop, so the two are the same control flow.

    ⚠ THE CPU PATH CALLS THIS TOO. It is the single narrow-phase dispatch
    for the SAP sweep on both targets; a rule written here is written once."""
    var sj_type = Int(
        rebind[Scalar[DTYPE]](geoms[sj, GEOM_IDX_TYPE])
    )
    var lo = si if si < sj else sj
    var hi = sj if si < sj else si
    var lo_type = si_type if si < sj else sj_type
    var hi_type = sj_type if si < sj else si_type
    var gi = lo
    var gj = hi
    # ⚠ RANK, NOT THE RAW ID. `pushPairArena` sorts by `mjtGeom`, and
    # this enum is not `mjtGeom` — comparing raw ids orders 10 of the
    # 28 type pairs the OPPOSITE way. See `mj_geom_type_rank`.
    if mj_geom_type_rank(lo_type) > mj_geom_type_rank(hi_type):
        gi = hi
        gj = lo

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
    var gi_is_si = gi == si
    var pi_x = si_px if gi_is_si else sj_px
    var pi_y = si_py if gi_is_si else sj_py
    var pi_z = si_pz if gi_is_si else sj_pz
    var qi_x = si_qx if gi_is_si else sj_qx
    var qi_y = si_qy if gi_is_si else sj_qy
    var qi_z = si_qz if gi_is_si else sj_qz
    var qi_w = si_qw if gi_is_si else sj_qw
    var ri = rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_RADIUS])
    var hli = rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_HALF_LENGTH])
    var hxi = rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_HALF_X])
    var hyi = rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_HALF_Y])
    var hzi = rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_HALF_Z])
    # Multi-CCD scales its distinctness tolerance by the smaller
    # bounding radius (`mjc_Convex`).
    var rbound_i = rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_RBOUND])

    var gj_type = Int(
        rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_TYPE])
    )
    var gj_body = Int(
        rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_BODY])
    )
    # `<contact><pair>` bypasses every filter below — see the same
    # gate in `_detect_contacts_env`. The AABB tests above still
    # apply, which is why the AABBs are inflated by the pair margin
    # where they are built: MuJoCo collides predefined pairs outside
    # the broadphase entirely, so a pair must not be prunable by a
    # bound that ignores its margin.
    comptime if _COLL_PROBE:
        pr._c_t0 = Int(perf_counter_ns())
    var ipair = find_predefined_pair[DTYPE](
        gi, gj, dims, pairs, mmeta
    )
    comptime if _COLL_PROBE:
        pr._c_ppair += Int(perf_counter_ns()) - pr._c_t0
        pr._n_ppair += 1
    if ipair < 0:
        # MuJoCo's body-pair filter — weld, weld-parent and exclude — then
        # the contype/conaffinity mask, through `_sap_pair_filter_rejects`.
        # ⚠ ONE FUNCTION FOR TWO READERS: this narrow phase, and the block
        # kernel's listing (`_sap_pair_listable`, `COLL_PREFILTER`), which
        # drops a pair BEFORE it is a candidate on the promise that this
        # call would reject it. A reject added here without going through
        # that function would make the listing keep a pair this rejects
        # (harmless); a reject RELAXED here without it would make the
        # listing drop a pair this collides — a silently missing contact.
        # (`_c_pbf` / `_n_pbf` time both tests since 2026-09-15.)
        comptime if _COLL_PROBE:
            pr._c_t0 = Int(perf_counter_ns())
        var _rej = _sap_pair_filter_rejects[DTYPE, EX_CAP=EX_CAP](
            gi_body, gj_body, gi_contype, gi_conaffinity,
            Int(rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_CONTYPE])),
            Int(rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_CONAFFINITY])),
            bodies, mmeta, excludes, ex_sig, n_sig, nbody,
        )
        comptime if _COLL_PROBE:
            pr._c_pbf += Int(perf_counter_ns()) - pr._c_t0
            pr._n_pbf += 1
        if _rej:
            return

    var mgi = rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_MARGIN])
    var mgj = rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_MARGIN])
    # Sum of the two geoms' margins, or the PAIR's own — never both.
    var cim = mgi + mgj  # MuJoCo 3.5+: sum of margins
    var cgp = (
        rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_GAP])
        + rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_GAP])
    )
    if ipair >= 0:
        cim = rebind[Scalar[DTYPE]](pairs[ipair, PAIR_IDX_MARGIN])
        cgp = rebind[Scalar[DTYPE]](pairs[ipair, PAIR_IDX_GAP])
    # ⚠⚠ TWO VALUES, NOT ONE. `cm` is the narrowphase CUTOFF and `cim`
    # is what the contact stores as its `includemargin`; 3.10.0 passes
    # `margin + gap` to the collision function and `margin` alone to
    # `mj_setContact`, so a contact in [margin, margin+gap) is DETECTED
    # and then EXCLUDED from the solver by
    # `con->exclude = dist >= includemargin`. With no `<geom gap>` the
    # two are equal and every line below is what it always was.
    var cm = cim + cgp

    var pj_x = sj_px if gi_is_si else si_px
    var pj_y = sj_py if gi_is_si else si_py
    var pj_z = sj_pz if gi_is_si else si_pz
    var qj_x = sj_qx if gi_is_si else si_qx
    var qj_y = sj_qy if gi_is_si else si_qy
    var qj_z = sj_qz if gi_is_si else si_qz
    var qj_w = sj_qw if gi_is_si else si_qw
    var rj = rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_RADIUS])
    var hlj = rebind[Scalar[DTYPE]](
        geoms[gj, GEOM_IDX_HALF_LENGTH]
    )
    var hxj = rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_HALF_X])
    var hyj = rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_HALF_Y])
    var hzj = rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_HALF_Z])
    var rbound_j = rebind[Scalar[DTYPE]](geoms[gj, GEOM_IDX_RBOUND])

    # ── BOUNDING-SPHERE REJECT — MuJoCo's `mj_filterSphere` ────────
    # ⚠⚠ THIS PATH RAN WITHOUT IT AND THE O(N^2) PATH DID NOT. MuJoCo
    # applies the test inside `mj_collideGeoms`, which sits DOWNSTREAM
    # of whichever broadphase produced the pair, so it covers every
    # candidate. Ours lived only in `contact_detection.mojo`, so every
    # model big enough to take the SAP branch (`ngeom >= 16` — which is
    # every interesting one) sent pairs into GJK that MuJoCo rejects
    # with three subtractions. The AABB tests above do NOT subsume it:
    # a sweep overlap on inflated world AABBs is far weaker than the
    # two bounding spheres actually touching.
    #
    # Measured, ms per env step (`FRAME_SKIP=10`), MIN of two
    # interleaved rounds against a pristine worktree of the parent:
    #
    #     SO-ARM100   2.87 -> 1.09   (349 -> 918 Hz)
    #     SO-ARM101   4.77 -> 1.84   (210 -> 544 Hz)
    #
    # MuJoCo steps the same two XMLs at 0.078 and 0.121 ms, so the
    # remaining gap is 14x and 15x, down from 37x and 39x.
    #
    # ⚠ `+ cm` IS LOAD-BEARING, and its absence is silent. A pair
    # separated by more than the two radii but LESS than its margin is
    # a contact MuJoCo reports; drop the term and it vanishes with no
    # error anywhere. This is the same trap the O(N^2) copy documents,
    # which is where the term was missing once before.
    #
    # ⚠ PLANES ARE EXCLUDED BY `rbound > 0`, which is how MuJoCo
    # detects them here too (a plane's `rbound` is 0 because it is
    # unbounded). MuJoCo additionally has a plane-specific arm using
    # `planeGeomDist`; that is NOT implemented here or in the O(N^2)
    # path, so plane pairs fall through to narrow phase exactly as
    # they did before this change.
    if rbound_i > Scalar[DTYPE](0) and rbound_j > Scalar[DTYPE](0):
        var sfx = pi_x - pj_x
        var sfy = pi_y - pj_y
        var sfz = pi_z - pj_z
        var sfb = rbound_i + rbound_j + cm
        if sfx * sfx + sfy * sfy + sfz * sfz > sfb * sfb:
            return
    # ── ORIENTED-BOX REJECT — MuJoCo's midphase `mj_collideOBB` ─────────
    # MuJoCo tests every geom pair of two multi-geom bodies box-against-box
    # (`mj_collideTree`, engine_collision_driver.c:1079) before the narrow
    # phase; we ran GJK on them. Measured on so101_tower with MuJoCo's own
    # trajectories: of the box/mesh pairs that pass the sphere test above,
    # 27.3 per env per step, the oriented boxes reject all but 3.1 — and none
    # of the 0.10 that are in contact.
    #
    # ⚠ THE BOX IS `geom_size`, CENTRED ON THE GEOM FRAME: exact for a box,
    # and for a mesh the smallest origin-centred box holding the hull
    # (`compute_mesh_half_extents_at`), looser than MuJoCo's off-centre
    # `geom_aabb` but still a bound. Box i is inflated by the cutoff `cm`, so
    # a pair within its margin is never rejected. Only box/mesh pairs with
    # nonzero sizes: a record whose sizes were never filled must not reject.
    if (
        (gi_type == GEOM_BOX or gi_type == GEOM_MESH)
        and (gj_type == GEOM_BOX or gj_type == GEOM_MESH)
        and hxi > Scalar[DTYPE](0) and hyi > Scalar[DTYPE](0)
        and hzi > Scalar[DTYPE](0) and hxj > Scalar[DTYPE](0)
        and hyj > Scalar[DTYPE](0) and hzj > Scalar[DTYPE](0)
    ):
        if _obb_separated[DTYPE](
            pi_x, pi_y, pi_z, qi_x, qi_y, qi_z, qi_w,
            hxi + cm, hyi + cm, hzi + cm,
            pj_x, pj_y, pj_z, qj_x, qj_y, qj_z, qj_w,
            hxj, hyj, hzj,
        ):
            return

    # ⚠⚠ THE CONTACT-PARAMETER MIX RUNS **AFTER** THE SPHERE
    # REJECT, NOT BEFORE, AND THE ORDER IS THE POINT.
    # `mix_contact_params` is ~30 tensor reads plus MuJoCo's
    # priority/max/min rules, and it used to run on every pair that
    # survived the body/contype filters — 65 per step on SO-ARM100,
    # of which the bounding-sphere test then rejects all but 2.
    # Nothing above needs it: the reject reads only the two rbounds
    # and `cm`, and `cm` comes from the geoms' own margins (or the
    # pair's), never from the mix. `_n0` moves with it because it is
    # a snapshot of `num_contacts`, which the reject cannot change.
    #
    # ⚠ THIS IS NOT THE HOIST §5.1 MEASURED AT ZERO. That one tried
    # to compute the per-GEOM decode once per geom; the mix is
    # per-PAIR and hoisting cannot remove it. Deferring past the
    # reject removes 97% of the CALLS.
    # MuJoCo's full contact-parameter rule, PRIORITY FIRST — shared
    # with `detect_contacts` so the two paths cannot drift, which is
    # exactly how the SAP ellipsoid branch went missing. A predefined
    # pair supplies its own parameters instead, unmixed.
    comptime if _COLL_PROBE:
        pr._c_t0 = Int(perf_counter_ns())
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
    comptime if _COLL_PROBE:
        pr._c_pparm += Int(perf_counter_ns()) - pr._c_t0
        pr._n_pparm += 1
    var cdim = Int(_mx[0])
    var cf = _mx[1]
    var cfs = _mx[2]
    var cfr = _mx[3]
    var _n0 = num_contacts

    var dist: Scalar[DTYPE] = 1.0
    var cx: Scalar[DTYPE] = 0
    var cy: Scalar[DTYPE] = 0
    var cz: Scalar[DTYPE] = 0
    var nx: Scalar[DTYPE] = 0
    var ny: Scalar[DTYPE] = 0
    var nz: Scalar[DTYPE] = 1
    # CONTACT DIRECTION INVARIANT — every branch below emits
    # `normal = gi -> gj` with `body_a = gi_body, body_b = gj_body`.
    #
    # The REVERSED-ORDER branches call a primitive written for the
    # other operand order, so they negate the returned normal to get
    # back to gi->gj. They used to ALSO swap body_a/body_b, and that
    # double flip left them emitting `normal = body_b -> body_a` while
    # the ten canonical-order branches emitted `body_a -> body_b`.
    # Either operation alone is correct; both is not.
    #
    # Silent until dm_control manipulator, which is the first model
    # where one physical pair type reaches BOTH orderings — a sphere
    # (the ball) contacting capsules (the fingers), under the SAP
    # broadphase where (gi, gj) comes from the sweep rather than the
    # geom index. `aref` is built from the penetration DEPTH and so
    # does not flip with the normal, so a flipped normal desynchronises
    # `jar = aref + J*qacc`: one contact was self-consistent and the
    # other was not, giving contact forces 9% and 20% below MuJoCo's
    # while every row constant matched to 15 digits.
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
            return
        var hf_is_i = gi_type == GEOM_HFIELD
        var hf_g = gi if hf_is_i else gj
        var cx_g = gj if hf_is_i else gi
        var hid = Int(
            rebind[Scalar[DTYPE]](geoms[hf_g, GEOM_IDX_HFIELD_ID])
        )
        if hid < 0:
            return
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
        comptime if _COLL_PROBE:
            pr._c_t0 = Int(perf_counter_ns())
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
            cm,
            cf,
            cfs,
            cfr,
            cdim,
            nsg,
            hfield_meta, hfield_data, dims.get_nhfield_data(),
            mesh_verts, mesh_vert_edgeadr, mesh_edges,
            dims, contacts, ws, num_contacts,
            cgp,
        )
        comptime if _COLL_PROBE:
            pr._c_hf += Int(perf_counter_ns()) - pr._c_t0
            pr._n_hf += 1
        _fill_pair_solparams[DTYPE](
            crow, _n0, num_contacts, _mx, contacts
        )
        return

    if gi_type == GEOM_SPHERE and gj_type == GEOM_SPHERE:
        comptime if _COLL_PROBE:
            pr._c_t0 = Int(perf_counter_ns())
        var r = sphere_sphere[DTYPE](
            pi_x, pi_y, pi_z, ri, pj_x, pj_y, pj_z, rj
        )
        comptime if _COLL_PROBE:
            pr._c_ss += Int(perf_counter_ns()) - pr._c_t0
            pr._n_ss += 1
        dist = r[0]
        cx = r[1]
        cy = r[2]
        cz = r[3]
        nx = r[4]
        ny = r[5]
        nz = r[6]
    elif gi_type == GEOM_CAPSULE and gj_type == GEOM_SPHERE:
        comptime if _COLL_PROBE:
            pr._c_t0 = Int(perf_counter_ns())
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
        comptime if _COLL_PROBE:
            pr._c_cas += Int(perf_counter_ns()) - pr._c_t0
            pr._n_cas += 1
        dist = r[0]
        cx = r[1]
        cy = r[2]
        cz = r[3]
        nx = r[4]
        ny = r[5]
        nz = r[6]
    elif gi_type == GEOM_SPHERE and gj_type == GEOM_CAPSULE:
        comptime if _COLL_PROBE:
            pr._c_t0 = Int(perf_counter_ns())
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
        comptime if _COLL_PROBE:
            pr._c_cas += Int(perf_counter_ns()) - pr._c_t0
            pr._n_cas += 1
        dist = r[0]
        cx = r[1]
        cy = r[2]
        cz = r[3]
        nx = -r[4]
        ny = -r[5]
        nz = -r[6]
    elif gi_type == GEOM_CAPSULE and gj_type == GEOM_CAPSULE:
        # ⚠ THE TWO NARROW PHASES MUST MOVE TOGETHER
        # (`feedback_sap_path_missing_a_whole_geom_type`). Parallel
        # capsules are a two-point manifold; see
        # `_capsule_capsule_contacts`, which writes its own records.
        comptime if _COLL_PROBE:
            pr._c_t0 = Int(perf_counter_ns())
        _ = _capsule_capsule_contacts[DTYPE](
            crow, gi_body, gj_body,
            pi_x, pi_y, pi_z, qi_x, qi_y, qi_z, qi_w, hli, ri,
            pj_x, pj_y, pj_z, qj_x, qj_y, qj_z, qj_w, hlj, rj,
            cm, cf, cfs, cfr, cdim,
            dims, contacts, num_contacts,
            cgp,
            max_contacts_in=max_contacts,
        )
        comptime if _COLL_PROBE:
            pr._c_cc += Int(perf_counter_ns()) - pr._c_t0
            pr._n_cc += 1
        _fill_pair_solparams[DTYPE](
            crow, _n0, num_contacts, _mx, contacts
        )
        return
    elif gi_type == GEOM_BOX and gj_type == GEOM_SPHERE:
        comptime if _COLL_PROBE:
            pr._c_t0 = Int(perf_counter_ns())
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
        comptime if _COLL_PROBE:
            pr._c_bs += Int(perf_counter_ns()) - pr._c_t0
            pr._n_bs += 1
        dist = r[0]
        cx = r[1]
        cy = r[2]
        cz = r[3]
        nx = r[4]
        ny = r[5]
        nz = r[6]
    elif gi_type == GEOM_SPHERE and gj_type == GEOM_BOX:
        comptime if _COLL_PROBE:
            pr._c_t0 = Int(perf_counter_ns())
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
        comptime if _COLL_PROBE:
            pr._c_bs += Int(perf_counter_ns()) - pr._c_t0
            pr._n_bs += 1
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
        comptime if _COLL_PROBE:
            pr._c_t0 = Int(perf_counter_ns())
        _ = _capsule_box_contacts[DTYPE](
            crow, gi_body, gj_body,
            pi_x, pi_y, pi_z, qi_x, qi_y, qi_z, qi_w, hxi, hyi, hzi,
            pj_x, pj_y, pj_z, qj_x, qj_y, qj_z, qj_w, hlj, rj,
            Scalar[DTYPE](-1),
            cm, cf, cfs, cfr, cdim,
            dims, contacts, num_contacts,
            cgp,
            max_contacts_in=max_contacts,
        )
        comptime if _COLL_PROBE:
            pr._c_cb += Int(perf_counter_ns()) - pr._c_t0
            pr._n_cb += 1
        _fill_pair_solparams[DTYPE](
            crow, _n0, num_contacts, _mx, contacts
        )
        return
    elif gi_type == GEOM_CAPSULE and gj_type == GEOM_BOX:
        comptime if _COLL_PROBE:
            pr._c_t0 = Int(perf_counter_ns())
        _ = _capsule_box_contacts[DTYPE](
            crow, gi_body, gj_body,
            pj_x, pj_y, pj_z, qj_x, qj_y, qj_z, qj_w, hxj, hyj, hzj,
            pi_x, pi_y, pi_z, qi_x, qi_y, qi_z, qi_w, hli, ri,
            Scalar[DTYPE](1),
            cm, cf, cfs, cfr, cdim,
            dims, contacts, num_contacts,
            cgp,
            max_contacts_in=max_contacts,
        )
        comptime if _COLL_PROBE:
            pr._c_cb += Int(perf_counter_ns()) - pr._c_t0
            pr._n_cb += 1
        _fill_pair_solparams[DTYPE](
            crow, _n0, num_contacts, _mx, contacts
        )
        return
    elif gi_type == GEOM_BOX and gj_type == GEOM_BOX:
        # A box/box contact is a whole manifold, not a point — see
        # `_box_box_contacts`. It writes its own records and this
        # branch is done; only a SEPARATED pair (code -1) falls through
        # to `box_box`, which then rejects it too.
        comptime if _COLL_PROBE:
            pr._c_t0 = Int(perf_counter_ns())
        var code = _box_box_contacts[DTYPE](
            crow,
            gi_body,
            gj_body,
            pi_x, pi_y, pi_z, qi_x, qi_y, qi_z, qi_w, hxi, hyi, hzi,
            pj_x, pj_y, pj_z, qj_x, qj_y, qj_z, qj_w, hxj, hyj, hzj,
            cm,
            cf,
            cfs,
            cfr,
            cdim,
            dims,
            contacts,
            num_contacts,
            cgp,
            max_contacts_in=max_contacts,
        )
        comptime if _COLL_PROBE:
            pr._c_bb += Int(perf_counter_ns()) - pr._c_t0
            pr._n_bb += 1
        if code >= 0:
            _fill_pair_solparams[DTYPE](
                crow, _n0, num_contacts, _mx, contacts
            )
            return
        comptime if _COLL_PROBE:
            pr._c_t0 = Int(perf_counter_ns())
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
        comptime if _COLL_PROBE:
            pr._c_bbf += Int(perf_counter_ns()) - pr._c_t0
            pr._n_bbf += 1
        dist = r[0]
        cx = r[1]
        cy = r[2]
        cz = r[3]
        nx = r[4]
        ny = r[5]
        nz = r[6]
    elif gi_type == GEOM_CYLINDER and gj_type == GEOM_SPHERE:
        comptime if _COLL_PROBE:
            pr._c_t0 = Int(perf_counter_ns())
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
        comptime if _COLL_PROBE:
            pr._c_cys += Int(perf_counter_ns()) - pr._c_t0
            pr._n_cys += 1
        dist = r[0]
        cx = r[1]
        cy = r[2]
        cz = r[3]
        nx = r[4]
        ny = r[5]
        nz = r[6]
    elif gi_type == GEOM_SPHERE and gj_type == GEOM_CYLINDER:
        comptime if _COLL_PROBE:
            pr._c_t0 = Int(perf_counter_ns())
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
        comptime if _COLL_PROBE:
            pr._c_cys += Int(perf_counter_ns()) - pr._c_t0
            pr._n_cys += 1
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
        # ⚠⚠ THE SAME MERGE AS `contact_detection.mojo` — see the
        # long note there. MuJoCo's `mjCOLLISIONFUNC` sends every
        # cylinder pair except SPHERE and PLANE to `mjc_Convex`;
        # `cylinder_capsule` / `cylinder_cylinder` use the
        # CAPSULE-capsule formula, which rounds the cylinder's flat
        # ends into hemispheres and bulges its surface a full radius.
        #
        # ⚠ THIS FILE IS A SECOND DISPATCH COPY of the same table, and
        # the CYLINDER x BOX re-route below landed in BOTH. The two
        # must move together or a model collides differently depending
        # on which path ran it.
        # MuJoCo routes CYLINDER x BOX to `mjc_Convex` — GJK plus EPA
        # (`engine_collision_driver.c:41`), not to a primitive. Ours
        # used `cylinder_box`, which REDUCES THE CYLINDER TO A CAPSULE,
        # so the hemispherical cap dips a full radius below the flat
        # face. Measured against the analytic depth that is an error of
        # exactly -r in EVERY configuration, separated or penetrating:
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
        comptime if _COLL_PROBE:
            pr._c_t0 = Int(perf_counter_ns())
        comptime if _COLL_REPEAT_GJK > 1:
            for _rep in range(_COLL_REPEAT_GJK - 1):
                var rq = gjk_epa[DTYPE](
                    gi_type,
                    pi_x, pi_y, pi_z, qi_x, qi_y, qi_z, qi_w,
                    ri, hli, hxi, hyi, hzi,
                    mesh_verts, mesh_vert_edgeadr, mesh_edges, 0, 0,
                    gj_type,
                    pj_x, pj_y, pj_z, qj_x, qj_y, qj_z, qj_w,
                    rj, hlj, hxj, hyj, hzj,
                    0, 0,
                    ws, wrow,
                    ccd_tol, ccd_iter, cm,
                    dist_cutoff=cm,
                )
                # Consumed against a value it cannot produce.
                if rq[0] == Scalar[DTYPE](-1.0e30):
                    dist = rq[0]
        var r = gjk_epa[DTYPE](
            gi_type,
            pi_x, pi_y, pi_z, qi_x, qi_y, qi_z, qi_w,
            ri, hli, hxi, hyi, hzi,
            mesh_verts, mesh_vert_edgeadr, mesh_edges, 0, 0,
            gj_type,
            pj_x, pj_y, pj_z, qj_x, qj_y, qj_z, qj_w,
            rj, hlj, hxj, hyj, hzj,
            0, 0,
            ws, wrow,
            ccd_tol, ccd_iter, cm,
            dist_cutoff=cm,
        )
        comptime if _COLL_PROBE:
            pr._c_gjkp += Int(perf_counter_ns()) - pr._c_t0
            pr._n_gjkp += 1
            if r[0] < cm:
                pr._n_gjkhit += 1
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

            # NATIVE MULTI-CONTACT — the SAME dispatch as
            # `contact_detection.mojo`. ⚠ THIS FILE IS A SECOND COPY OF
            # THE NARROW PHASE, and when the manifold path landed there
            # first the two producers disagreed: an env on the SAP path
            # got ONE point for a mesh pair where the O(N^2) path gave
            # four. Same model, different contacts, decided by which
            # broadphase the config happened to select. See
            # `feedback_one_field_two_producers`.
            # `MC_ENABLED` sits LAST because it is a comptime
            # `True`: on the left it folds and the compiler flags the
            # rest of the chain unreachable. Every other operand is a
            # pure comparison, so the order is not observable.
            var mc_pair = (
                (gi_type == GEOM_MESH or gi_type == GEOM_BOX)
                and (gj_type == GEOM_MESH or gj_type == GEOM_BOX)
                and cm <= Scalar[DTYPE](0)
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
            # The pair's warm slot for the mesh hill climb (ccd_workspace.mojo):
            # a hash of the sorted geom pair, so the state follows the PAIR
            # across steps whatever the candidate set does around it.
            var hw_slot: Int
            comptime if HILL_WARM_ACROSS_STEPS:
                hw_slot = (si * 131 + sj) % HILL_WARM_SLOTS
            else:
                hw_slot = -1
            comptime if _COLL_PROBE:
                pr._c_t0 = Int(perf_counter_ns())
            comptime if _COLL_REPEAT_GJK > 1:
                for _rep in range(_COLL_REPEAT_GJK - 1):
                    var qf1 = Array[Scalar[DTYPE], 9](
                        fill=Scalar[DTYPE](0)
                    )
                    var qf2 = Array[Scalar[DTYPE], 9](
                        fill=Scalar[DTYPE](0)
                    )
                    var qxx = Array[Scalar[DTYPE], 6](
                        fill=Scalar[DTYPE](0)
                    )
                    var qf_ok = 0
                    var qfi = Array[Int, 6](fill=-1)
                    var rq = gjk_epa_witness[DTYPE](
                        gi_type,
                        pi_x, pi_y, pi_z, qi_x, qi_y, qi_z, qi_w,
                        ri, hli, hxi, hyi, hzi,
                        mesh_verts, mesh_vert_edgeadr, mesh_edges, va1, mnv1,
                        gj_type,
                        pj_x, pj_y, pj_z, qj_x, qj_y, qj_z, qj_w,
                        rj, hlj, hxj, hyj, hzj,
                        va2, mnv2,
                        qf1, qf2, qxx, qf_ok, qfi,
                        ws, wrow,
                        ccd_tol, ccd_iter, cm,
                        cm,
                        warm_slot=hw_slot,
                        hw_row=hw_row,
                    )
                    if rq[0] == Scalar[DTYPE](-1.0e30):
                        dist = rq[0]
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
                ws, wrow,
                ccd_tol, ccd_iter, cm,
                # Opt in to the cutoff exit: `dist` below is read ONLY
                # by `if dist < cm`, and everything that consumes the
                # witness sits inside that branch.
                cm,
                warm_slot=hw_slot,
                hw_row=hw_row,
            )
            comptime if _COLL_PROBE:
                pr._c_gjk += Int(perf_counter_ns()) - pr._c_t0
                pr._n_gjk += 1
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
                and dist < cm
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
                # ⚠ THE OPERANDS ARE ALREADY MuJoCo'S. `(gi, gj)` is
                # `pushPairArena`'s pair — sorted by (type, geom
                # index) where the sweep names it — so the manifold
                # runs on the SAME order GJK just ran on, which is the
                # reference's structure: `mjc_Convex` hands
                # `multicontact` the `status` of its own `mjc_ccd`.
                #
                # ⚠⚠ THERE USED TO BE A SECOND, LOCAL SWAP HERE, and
                # it was half a fix. It ordered the MANIFOLD correctly
                # and left GJK running on whatever the broadphase
                # emitted, so `wf1`/`wf2`/`wx` — the witness the
                # manifold clips from — came out of a query in the
                # OTHER order and had to be re-swapped to match. With
                # the pair canonicalised where it is named, that
                # predicate is always false and the re-swap is gone.
                comptime if _COLL_PROBE:
                    pr._c_t0 = Int(perf_counter_ns())
                var mcn = native_multicontact_contacts[
                    DTYPE](
                    crow, body_a, body_b,
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
                    dist, cm, cf, cfs, cfr, cdim,
                    False,
                    contacts, ws, wrow, num_contacts,
                    cgp,
                    max_contacts_in=max_contacts,
                )
                comptime if _COLL_PROBE:
                    pr._c_mcn += Int(perf_counter_ns()) - pr._c_t0
                    pr._n_mcn += 1
                # The manifold REPLACES the single point.
                if mcn > 0:
                    _fill_pair_solparams[
                        DTYPE](crow, _n0, num_contacts, _mx, contacts)
                    return
        else:
            _fill_pair_solparams[DTYPE](
                crow, _n0, num_contacts, _mx, contacts
            )
            return

    if dist < cm and num_contacts < max_contacts:
        # The `gi -> gj` normal, captured BEFORE the emit negates it in
        # place — see the identical capture in `contact_detection.mojo`.
        var mccd_nx = nx
        var mccd_ny = ny
        var mccd_nz = nz
        var mccd_first = num_contacts
        var c_off = num_contacts * CONTACT_SIZE
        contacts[crow, c_off + CONTACT_IDX_BODY_A] = Scalar[DTYPE](
            body_a
        )
        contacts[crow, c_off + CONTACT_IDX_BODY_B] = Scalar[DTYPE](
            body_b
        )
        contacts[crow, c_off + CONTACT_IDX_POS_X] = cx
        contacts[crow, c_off + CONTACT_IDX_POS_Y] = cy
        contacts[crow, c_off + CONTACT_IDX_POS_Z] = cz
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
        contacts[crow, c_off + CONTACT_IDX_NX] = nx
        contacts[crow, c_off + CONTACT_IDX_NY] = ny
        contacts[crow, c_off + CONTACT_IDX_NZ] = nz
        contacts[crow, c_off + CONTACT_IDX_DIST] = dist
        contacts[crow, c_off + CONTACT_IDX_INCLUDEMARGIN] = cim
        contacts[crow, c_off + CONTACT_IDX_FRICTION] = cf
        contacts[crow, c_off + CONTACT_IDX_FRICTION_SPIN] = cfs
        contacts[crow, c_off + CONTACT_IDX_FRICTION_ROLL] = cfr
        contacts[crow, c_off + CONTACT_IDX_CONDIM] = Scalar[DTYPE](
            cdim
        )
        num_contacts += 1

        # MULTI-POINT CONVEX CONTACT — defect 21.
        #
        # ⚠⚠ THIS FILE IS THE SECOND NARROW PHASE. `contact_detection`
        # carries the same dispatch and the same emit, and SAP takes
        # over at ngeom >= SAP_THRESHOLD — so patching only the other
        # one would have left every LARGE model (dog, quadruped: the
        # exact models this was found on) with single-point cylinder
        # contacts while the small-model gate went green. That is the
        # shape of `feedback_sap_path_missing_a_whole_geom_type`, and
        # it is why this hook is duplicated rather than "left for
        # later". The two must move together.
        #
        # ⚠ AND THEY DID, for `mjDSBL_MULTICCD`. `<flag
        # multiccd="disable"/>` is the model asking for single-point
        # convex contacts; honouring it in only one narrow phase would
        # have left every model at or above `SAP_THRESHOLD` — which is
        # every dm_control manipulation model, at 185-431 geoms — with
        # the 4-point manifold the flag exists to switch off.
        if not multiccd_off and multi_ccd_pair_supported(
            gi_type, gj_type, cm > Scalar[DTYPE](0)
        ):
            comptime if _COLL_PROBE:
                pr._c_t0 = Int(perf_counter_ns())
            _ = multi_ccd_extra_contacts[
                DTYPE](
                crow, body_a, body_b, mccd_first,
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
                cm, cf, cfs, cfr, cdim,
                contacts, num_contacts,
                ws, wrow,
                ccd_tol, ccd_iter, cm,
                cgp,
                max_contacts_in=max_contacts,
            )
            comptime if _COLL_PROBE:
                pr._c_mccd += Int(perf_counter_ns()) - pr._c_t0
                pr._n_mccd += 1

    _fill_pair_solparams[DTYPE](
        crow, _n0, num_contacts, _mx, contacts
    )


@always_inline
def _detect_contacts_sap_env[
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
    # Compiled on both targets — see the twin note in `contact_detection`.
    # EPA's polytope lives in `d.ccd_ws`, not on the per-thread stack, which
    # is what let the second GJK/EPA instantiation into the Metal kernel.
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
    # CCD workspace row; -1 = `env` (every serial caller).
    wrow_in: Int = -1,
):
    """AABB/SAP broadphase contact detection for one env (verbatim from
    detect_contacts_sap_gpu; mesh branches compiled in iff nmesh_verts > 0).
    """
    var wrow = wrow_in if wrow_in >= 0 else env
    var nq = dims.get_nq()
    var nv = dims.get_nv()
    var nbody = dims.get_nbody()
    var njoint = dims.get_njoint()
    var max_contacts = dims.get_max_contacts()
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
    # `_COLL_PROBE` accumulators (compiled out when the flag is False).
    var pr = _SapProbe()
    comptime if _COLL_PROBE:
        pr._c_start = Int(perf_counter_ns())
    var ngeom = dims.get_ngeom()
    var nexclude = dims.get_nexclude()
    var nmesh_verts = dims.get_nmesh_verts()
    var npair = dims.get_npair()
    var num_contacts = 0

    # ------------------------------------------------------------------
    # 1. Precompute world positions for all ngeom geoms.
    # ------------------------------------------------------------------
    var wpx = Scratch[Scalar[DTYPE], cap[D.NGEOM]()](ngeom, uninitialized=0)
    var wpy = Scratch[Scalar[DTYPE], cap[D.NGEOM]()](ngeom, uninitialized=0)
    var wpz = Scratch[Scalar[DTYPE], cap[D.NGEOM]()](ngeom, uninitialized=0)
    var wqx = Scratch[Scalar[DTYPE], cap[D.NGEOM]()](ngeom, uninitialized=0)
    var wqy = Scratch[Scalar[DTYPE], cap[D.NGEOM]()](ngeom, uninitialized=0)
    var wqz = Scratch[Scalar[DTYPE], cap[D.NGEOM]()](ngeom, uninitialized=0)
    var wqw = Scratch[Scalar[DTYPE], cap[D.NGEOM]()](ngeom, uninitialized=0)

    for g in range(ngeom):
        var px: Scalar[DTYPE] = 0
        var py: Scalar[DTYPE] = 0
        var pz: Scalar[DTYPE] = 0
        var qx: Scalar[DTYPE] = 0
        var qy: Scalar[DTYPE] = 0
        var qz: Scalar[DTYPE] = 0
        var qw: Scalar[DTYPE] = 1
        _geom_world_pos[DTYPE](
            env, g, geoms, xpos, xquat, px, py, pz, qx, qy, qz, qw
        )
        wpx[g] = px
        wpy[g] = py
        wpz[g] = pz
        wqx[g] = qx
        wqy[g] = qy
        wqz[g] = qz
        wqw[g] = qw

    # ------------------------------------------------------------------
    # 2. Compute AABBs for non-plane geoms.
    # ------------------------------------------------------------------
    var aabb_min_x = Scratch[Scalar[DTYPE], cap[D.NGEOM]()](ngeom, uninitialized=0)
    var aabb_max_x = Scratch[Scalar[DTYPE], cap[D.NGEOM]()](ngeom, uninitialized=0)
    var aabb_min_y = Scratch[Scalar[DTYPE], cap[D.NGEOM]()](ngeom, uninitialized=0)
    var aabb_max_y = Scratch[Scalar[DTYPE], cap[D.NGEOM]()](ngeom, uninitialized=0)
    var aabb_min_z = Scratch[Scalar[DTYPE], cap[D.NGEOM]()](ngeom, uninitialized=0)
    var aabb_max_z = Scratch[Scalar[DTYPE], cap[D.NGEOM]()](ngeom, uninitialized=0)

    for g in range(ngeom):
        var gt = Int(rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_TYPE]))
        if gt == GEOM_PLANE:
            continue
        var r = rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_RADIUS])
        var hl = rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_HALF_LENGTH])
        var hx = rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_HALF_X])
        var hy = rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_HALF_Y])
        var hz = rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_HALF_Z])
        var rb = rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_RBOUND])
        var he = _aabb_half_extents[DTYPE](
            gt, wqx[g], wqy[g], wqz[g], wqw[g], r, hl, hx, hy, hz, rb
        )
        # ⚠⚠ THE GEOM'S OWN MARGIN, WHICH THIS SWEEP USED TO OMIT. MuJoCo's
        # `filterBox` and `mj_filterSphere` are both called WITH the pair's
        # margin, and the pair's margin is `geom_margin[g1] + geom_margin[g2]`
        # — a SUM — so widening each geom by its own covers it exactly. Without
        # it a pair separated by less than its margin but more than its extents
        # never reaches the narrow phase, and the contact simply does not
        # happen: flybody's two labrum ellipsoids are `dist = +5.106e-05` with
        # `margin = 0.001`, and MuJoCo has them ACTIVE (`exclude 0`) while this
        # engine had nothing. Only the PAIR margin was folded in, below.
        # ⚠ Conservative by construction — a wider AABB offers the narrow
        # phase more candidates, it never invents a contact.
        # ⚠ `+ gap` TOO. The narrowphase cutoff is `margin + gap`, so an AABB
        # inflated by `margin` alone drops every contact in the gap band before
        # it is ever tested — which is the whole band `<adhesion>` reaches.
        var gm = (
            rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_MARGIN])
            + rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_GAP])
        )
        if gm < Scalar[DTYPE](0):
            gm = Scalar[DTYPE](0)
        aabb_min_x[g] = wpx[g] - he[0] - gm
        aabb_max_x[g] = wpx[g] + he[0] + gm
        aabb_min_y[g] = wpy[g] - he[1] - gm
        aabb_max_y[g] = wpy[g] + he[1] + gm
        aabb_min_z[g] = wpz[g] - he[2] - gm
        aabb_max_z[g] = wpz[g] + he[2] + gm

    # Inflate by any predefined pair's margin. MuJoCo never subjects a
    # `<contact><pair>` to the broadphase at all — the merge loop collides it
    # whatever the AABBs say — so a pair whose two geoms sit further apart
    # than their extents but closer than its margin has to survive the sweep
    # below. Conservative by construction: a wider AABB only offers the narrow
    # phase more candidates, it never changes a contact.
    #
    # ⚠ The geoms' OWN margin is folded in above, at the AABB itself; this
    # loop is only about a `<contact><pair margin=>`, which belongs to the
    # pair and not to either geom.
    var n_pair_aabb = Int(rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_NPAIR]))
    # EPA's stopping rule, from model META — see `_detect_contacts_env` for
    # why it is read rather than hardcoded, and why a non-positive value falls
    # back instead of meaning "zero iterations".
    var ccd_tol = rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_CCD_TOLERANCE])
    if ccd_tol <= 0:
        ccd_tol = Scalar[DTYPE](MJ_CCD_TOLERANCE)
    var ccd_iter = Int(
        rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_CCD_ITERATIONS])
    )
    if ccd_iter < 1:
        ccd_iter = MJ_CCD_ITERATIONS
    # `mjDSBL_MULTICCD` — read here for the same reason `ccd_tol` is, and it
    # must stay in lockstep with `_detect_contacts_env`'s copy.
    var multiccd_off = (
        rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_MULTICCD_DISABLED]) != 0
    )
    if n_pair_aabb > npair:
        n_pair_aabb = npair
    for p in range(n_pair_aabb):
        # ⚠ `margin + gap`, THE SAME SUM THE GEOM AABB ABOVE USES (AUD-36).
        # `filterCollisionPair` runs `mj_filterSphere(m, d, g1, g2,
        # margin + gap)` for a predefined pair exactly as it does for a
        # dynamic one (engine_collision_driver.c), so widening by `margin`
        # alone dropped every in-gap pair contact before the narrow phase
        # could see it — on the SAP path only, which is what made it invisible
        # to the O(N^2) gates. And the GUARD has to be the sum too: a pair
        # with `margin="0" gap="0.01"` skipped this loop entirely.
        var pm = (
            rebind[Scalar[DTYPE]](pairs[p, PAIR_IDX_MARGIN])
            + rebind[Scalar[DTYPE]](pairs[p, PAIR_IDX_GAP])
        )
        if pm <= Scalar[DTYPE](0):
            continue
        for side in range(2):
            var g = Int(
                rebind[Scalar[DTYPE]](
                    pairs[p, PAIR_IDX_GEOM1 if side == 0 else PAIR_IDX_GEOM2]
                )
            )
            if Int(rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_TYPE])) == (
                GEOM_PLANE
            ):
                continue  # planes have no AABB here
            aabb_min_x[g] -= pm
            aabb_max_x[g] += pm
            aabb_min_y[g] -= pm
            aabb_max_y[g] += pm
            aabb_min_z[g] -= pm
            aabb_max_z[g] += pm

    # ------------------------------------------------------------------
    # 3. Plane vs non-plane pairs.
    # ------------------------------------------------------------------
    for gi in range(ngeom):
        var gi_type = Int(
            rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_TYPE])
        )
        if gi_type != GEOM_PLANE:
            continue
        var gi_body = Int(
            rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_BODY])
        )
        var gi_contype = Int(
            rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_CONTYPE])
        )
        var gi_conaffinity = Int(
            rebind[Scalar[DTYPE]](geoms[gi, GEOM_IDX_CONAFFINITY])
        )
        # The plane's full pose. This loop used to keep only `wpz[gi]` as a
        # `ground_z` and hardcode the normal to (0,0,1), i.e. it modelled every
        # plane as a horizontal floor at the height of its origin. See
        # `collision/plane_frame.mojo`. Everything below now works in the
        # PLANE'S FRAME — where the plane really is z=0 with normal +z, which
        # is what all the `*_plane` primitives assume — and maps the contact
        # point and normal back to world at the write.
        var plp_x = wpx[gi]
        var plp_y = wpy[gi]
        var plp_z = wpz[gi]
        var plq_x = wqx[gi]
        var plq_y = wqy[gi]
        var plq_z = wqz[gi]
        var plq_w = wqw[gi]
        var pn = plane_world_normal[DTYPE](plq_x, plq_y, plq_z, plq_w)

        for gj in range(ngeom):
            if num_contacts >= max_contacts:
                # ⚠ ORDERED BEFORE THE EARLY EXIT TOO. This return is the
                # `max_contacts` overflow guard, and a truncated contact set is
                # still handed to the solver — leaving it in sweep order would
                # make the ordering fix silently conditional on not overflowing.
                sort_contacts_mujoco_order[DTYPE](
                    env, contacts, num_contacts
                )
                smeta[env, META_IDX_NUM_CONTACTS] = Scalar[DTYPE](
                    num_contacts
                )
                return
            _sap_plane_narrow[
                DTYPE, BATCH, D, EX_CAP, HFIELD_ENABLED=HFIELD_ENABLED
            ](
                env, env, wrow, dims, gi, gj, gi_body, gi_contype, gi_conaffinity,
                plp_x, plp_y, plp_z, plq_x, plq_y, plq_z, plq_w,
                pn, nbody, max_contacts, ex_sig, n_sig, pr, num_contacts,
                wpx[gj], wpy[gj], wpz[gj], wqx[gj], wqy[gj], wqz[gj], wqw[gj],
                geoms, bodies, mmeta, excludes, pairs, mesh_meta, mesh_verts, mesh_vert_edgeadr, mesh_edges, contacts, ws,
            )

    # ------------------------------------------------------------------
    # 4. SAP sweep for non-plane pairs.
    # ------------------------------------------------------------------

    # 4a. Build SAP index list.
    # Which geoms a `<pair>` names — see the note on the mask filter below.
    var pair_geom = Scratch[Int, cap[D.NGEOM]()](ngeom, fill=0)
    for p in range(n_pair_aabb):
        for side in range(2):
            var pg = Int(
                rebind[Scalar[DTYPE]](
                    pairs[p, PAIR_IDX_GEOM1 if side == 0 else PAIR_IDX_GEOM2]
                )
            )
            if pg >= 0 and pg < ngeom:
                pair_geom[pg] = 1
    var sap_idx = Scratch[Int, cap[D.NGEOM]()](ngeom, uninitialized=0)
    var sap_n = 0
    for g in range(ngeom):
        var gt = Int(rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_TYPE]))
        if gt == GEOM_PLANE:
            continue
        # ⚠ A GEOM THAT CANNOT COLLIDE NEVER ENTERS THE SWEEP. MuJoCo's
        # broadphase walks only bodies that `canCollide`
        # (engine_collision_driver.c:320) and `filterBitmask` (:535) rejects a
        # pair unless `contype_1 & conaffinity_2 || contype_2 & conaffinity_1`;
        # a geom with both words zero fails against EVERY partner. We used to
        # sweep all of them and reject each pair after the AABB test, the
        # predefined-pair lookup and the body filter: on reassemble3 (267
        # geoms, 120 collidable) that was 10,600 sweep iterations and 2,100
        # AABB-passing pairs per step for 79 real candidates, 180 µs of a
        # 200 µs collision phase (PERFORMANCE.md §13.18). Exact: no contact
        # can come from such a geom, so the contact set and its order are
        # untouched.
        #
        # ⚠⚠ UNLESS A `<pair>` NAMES IT. MuJoCo's predefined pairs never meet
        # `filterBitmask`: `mj_collision` merges them into the broadphase's
        # body pairs by signature and collides them as they are
        # (engine_collision_driver.c:611-615, :779-780), which is the whole
        # point of `<pair>` — ToddlerBot's torso-to-arm contacts are 65 such
        # pairs between geoms whose class sets `contype="0" conaffinity="0"`.
        # From 3b97ce19 to this fix none of those geoms entered the sweep, so
        # the arm went through the chest in the studio while every board row
        # stayed green (no keyframe puts an arm in a torso). The rule that
        # the comment above states is MuJoCo's rule for the MASK; the pair
        # table is the other door, and `pair_geom` keeps it open.
        var g_ct = Int(rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_CONTYPE]))
        var g_ca = Int(rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_CONAFFINITY]))
        if g_ct == 0 and g_ca == 0 and pair_geom[g] == 0:
            continue
        sap_idx[sap_n] = g
        sap_n += 1

    # 4b. Insertion sort by aabb_min_x.
    for i in range(1, sap_n):
        var key = sap_idx[i]
        var key_val = aabb_min_x[key]
        var j = i - 1
        while j >= 0 and aabb_min_x[sap_idx[j]] > key_val:
            sap_idx[j + 1] = sap_idx[j]
            j -= 1
        sap_idx[j + 1] = key

    # 4c. Sweep.
    #
    # ⚠⚠ `si`/`sj` ARE THE SWEEP'S ORDER AND `gi`/`gj` ARE MuJoCo'S. They are
    # not the same pair order and the difference is a real defect, not a
    # cosmetic one — see the canonicalisation inside the `j` loop. Only the
    # AABB tests and the `break` may read `si`/`sj`; everything downstream of
    # them reads `gi`/`gj`.
    comptime if _COLL_PROBE:
        pr._c_loop0 = Int(perf_counter_ns())
    for i in range(sap_n):
        var si = sap_idx[i]
        var si_max_x = aabb_max_x[si]
        var si_type = Int(
            rebind[Scalar[DTYPE]](geoms[si, GEOM_IDX_TYPE])
        )

        for j in range(i + 1, sap_n):
            if num_contacts >= max_contacts:
                # ⚠ ORDERED BEFORE THE EARLY EXIT TOO. This return is the
                # `max_contacts` overflow guard, and a truncated contact set is
                # still handed to the solver — leaving it in sweep order would
                # make the ordering fix silently conditional on not overflowing.
                sort_contacts_mujoco_order[DTYPE](
                    env, contacts, num_contacts
                )
                smeta[env, META_IDX_NUM_CONTACTS] = Scalar[DTYPE](
                    num_contacts
                )
                return
            comptime if _COLL_PROBE:
                pr._n_pairs += 1
            var sj = sap_idx[j]

            if aabb_min_x[sj] > si_max_x:
                break

            if (
                aabb_min_y[sj] > aabb_max_y[si]
                or aabb_min_y[si] > aabb_max_y[sj]
            ):
                continue
            if (
                aabb_min_z[sj] > aabb_max_z[si]
                or aabb_min_z[si] > aabb_max_z[sj]
            ):
                continue
            comptime if _COLL_PROBE:
                pr._n_aabb += 1

            # ── THE PAIR IN MuJoCo'S ORDER — `pushPairArena` ──────────────
            #
            # ⚠⚠ THE NARROW PHASE IS NOT SYMMETRIC IN ITS OPERANDS, so which
            # geom is `obj1` is part of the answer, not a convention. MuJoCo
            # fixes it in `pushPairArena` (`engine_collision_driver.c:489`):
            #
            #     if (m->geom_type[g1] > m->geom_type[g2]) { swap(g1, g2); }
            #
            # on a pair that arrives in ASCENDING GEOM INDEX — `add_pair`
            # stores the bodyflex pair as `(min<<16) + max`, and the geom loops
            # under it run `for g1 in body bf1: for g2 in body bf2` with
            # `bf1 < bf2`, so `g1 < g2` always. (A `<contact><pair>` is swapped
            # by `mjCPair::ResolveReferences` on BODY id, which gives the same
            # thing for the cross-body pairs that can actually collide.) Net
            # rule: **sort by (type, geom index)** — which is exactly the
            # predicate `native_multicontact`'s caller already spelled out for
            # the manifold routine, and which nothing had applied to GJK/EPA.
            #
            # ⚠ THIS SWEEP EMITS PAIRS IN AABB-SORT ORDER, so roughly half of
            # them reached GJK the wrong way round on every model at or above
            # `SAP_THRESHOLD` — which is every dm_control and Menagerie scene.
            # Measured: `test_ellipsoid_convex_vs_mujoco`'s pair reproduces
            # MuJoCo BIT-FOR-BIT in index order (-0.005577960335) and lands
            # 2.0e-07 away in the other, and `test_mesh_manifold_gpu_parity`
            # had SAP and the O(N^2) loop reporting DIFFERENT CONTACT COUNTS
            # on the same pose.
            #
            # ⚠ WHY IT IS DONE HERE AND NOT AT THE CALL SITES. Every `gi_*`
            # local below is read by ~800 lines of narrow-phase dispatch; doing
            # it once, where the pair is named, keeps all of that untouched and
            # makes the invariant checkable in one place. The cost is that the
            # `gi_*` reads move from once-per-sweep-column to once-per-pair —
            # the `gj_*` ones already were, and both sit AFTER the AABB tests
            # that reject most candidates.
            _sap_pair_narrow[
                DTYPE, BATCH, D, EX_CAP, HFIELD_ENABLED=HFIELD_ENABLED
            ](
                env, env, wrow, dims, si, sj, si_type, nbody, max_contacts,
                ex_sig, n_sig, pr, num_contacts,
                wpx[si], wpy[si], wpz[si], wqx[si], wqy[si], wqz[si], wqw[si],
                wpx[sj], wpy[sj], wpz[sj], wqx[sj], wqy[sj], wqz[sj], wqw[sj],
                ccd_tol, ccd_iter, multiccd_off,
                geoms, bodies, mmeta, excludes, pairs, mesh_meta, mesh_verts, mesh_polys, mesh_polyvert, mesh_polymap, mesh_vert_polymap, mesh_vert_edgeadr, mesh_edges, hfield_meta, hfield_data, contacts, ws,
            )

    comptime if _COLL_PROBE:
        var _c_end = Int(perf_counter_ns())
        var _c_sum = pr._c_pcyl + pr._c_pbox + pr._c_pmesh + pr._c_hf + pr._c_ss + pr._c_cc + pr._c_cb + pr._c_bb + pr._c_gjk + pr._c_mcn + pr._c_mccd + pr._c_cas + pr._c_bs + pr._c_bbf + pr._c_cys + pr._c_gjkp
        print("[cprobe]", "broad", pr._c_loop0 - pr._c_start, 0, "other", _c_end - pr._c_loop0 - _c_sum, 0, "pairs", 0, pr._n_pairs, "aabb", 0, pr._n_aabb, "ppair", pr._c_ppair, pr._n_ppair, "pparm", pr._c_pparm, pr._n_pparm, "pbf", pr._c_pbf, pr._n_pbf, "pcyl", pr._c_pcyl, pr._n_pcyl, "pbox", pr._c_pbox, pr._n_pbox, "pmesh", pr._c_pmesh, pr._n_pmesh, "hf", pr._c_hf, pr._n_hf, "ss", pr._c_ss, pr._n_ss, "cc", pr._c_cc, pr._n_cc, "cb", pr._c_cb, pr._n_cb, "bb", pr._c_bb, pr._n_bb, "gjk", pr._c_gjk, pr._n_gjk, "mcn", pr._c_mcn, pr._n_mcn, "mccd", pr._c_mccd, pr._n_mccd, "cas", pr._c_cas, pr._n_cas, "bs", pr._c_bs, pr._n_bs, "bbf", pr._c_bbf, pr._n_bbf, "cys", pr._c_cys, pr._n_cys, "gjkp", pr._c_gjkp, pr._n_gjkp, "gjkhit", 0, pr._n_gjkhit)
    # ── MuJoCo's contact ORDER (`bfsort`, engine_collision_driver.c:1683) ──
    # The sweep above emits in AABB order and runs PLANES in a separate phase
    # before it; MuJoCo runs body pair by body pair in SORTED signature order.
    # See `collision/contact_order.mojo` — noslip and PGS are Gauss-Seidel, so
    # this is part of the answer, not a presentation choice.
    sort_contacts_mujoco_order[DTYPE](env, contacts, num_contacts)

    smeta[env, META_IDX_NUM_CONTACTS] = Scalar[DTYPE](num_contacts)


def _detect_contacts_sap_fields_kernel[
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
    # True = the block kernel's fallback launch: run only for envs it marked
    # with `ncon = -1`, leave every other env's contacts untouched.
    ONLY_FLAGGED: Bool = False,
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
    comptime if ONLY_FLAGGED:
        if rebind[Scalar[DTYPE]](smeta[env, META_IDX_NUM_CONTACTS]) >= Scalar[DTYPE](0):
            return
    _detect_contacts_sap_env[DTYPE, BATCH](
        env, Dims[nq=NQ, nv=NV, nbody=NBODY, njoint=NJOINT, max_contacts=MAX_CONTACTS, ngeom=NGEOM, nexclude=NEXCLUDE, nmesh_verts=NMESH_VERTS, npair=NPAIR](), xpos, xquat, geoms, bodies, mmeta, excludes, pairs, mesh_meta,
        mesh_verts, mesh_polys, mesh_polyvert, mesh_polymap,
        mesh_vert_polymap, mesh_vert_edgeadr, mesh_edges,
        hfield_meta, hfield_data, contacts, smeta, ccd_ws,
    )



# The block-kernel switch lives in `ccd_workspace.mojo` (`COLL_BLOCK_KERNEL`),
# next to the sizes that depend on it.

# ⚠ A BISECT KNOB, TIMING INSTRUMENT ONLY. Returns from the block kernel after
# phase N, writing `ncon = 0` so the step stays bounded (no contacts: the
# solver sees no rows — the WORKLOAD changes, so read only the collision
# kernel's own per-launch time, never the step). 1 = after the world-pose /
# AABB phase, 2 = after thread 0's candidate generation, 3 = after the
# per-thread narrow phase, 0 = production. Same pattern as
# `newton_solve.NEWTON_STOP_AFTER`, for the same reason: on the RTX 5090 the
# per-thread narrow phase (phase 2) read 240 of the 270 us at the k=0 park
# scene (2026-09-07; the 21/22/23 sub-stops of that bisect split the old
# cheap-thread / CCD-lane assignment and went with it, 2026-09-11).
comptime COLL_STOP_AFTER: Int = 0
# ⚠ A TIMING INSTRUMENT, OFF IN PRODUCTION. True makes the block kernel read
# the device clock (`perf_counter_ns`, the global timer) at its phase
# boundaries on thread 0, and per lane around the narrow phase, and write them
# into the `COLL_REPORT_BASE` tail of the env's `stage` row — the region
# `COLL_CAND_REPORT` uses, so the two are exclusive. Layout from that base:
#     [0..3] pose/AABB, sweep, narrow phase, output (ns)   [4] total
#     [8 + 4*l ..] lane l: narrow-phase ns, candidates run, slowest
#                  candidate's ns, its kind key
# Results are unchanged.
#
# ⚠⚠ TRUST THE PHASE SPLIT, NOT THE PER-LANE ATTRIBUTION. The four phases are
# block-wide (thread 0, between barriers) and sound. The per-lane numbers are
# not what they look like: the 32 lanes of a warp run their candidates in
# lockstep, so while lanes take different branches (box/box on some, GJK on
# others) EVERY lane's timer spans the serialized total. On so101_tower
# (2026-09-26) that made "the slowest candidate on the critical lane" read
# box/box in all 1024 envs — an artifact of the kind-sorted candidate order
# (lane 0 holds the first kind), refuted by skipping kinds: the GJK kinds
# were ~83% of the narrow phase, box/box ~8%. To price a kind, SKIP it
# (timing-only build) and compare launches; do not read it off a lane.
comptime COLL_TIMING: Bool = False

# Candidate KIND keys for the block kernel's phase-2 order: a geom pair is
# `rank_lo * 8 + rank_hi` (`mj_geom_type_rank`, 0..7), a plane candidate is
# `_KIND_PLANE_BASE + rank(geom)`. Lanes of a warp run the same narrow-phase
# routine when their candidates share a key — see the kernel's docstring.
comptime _KIND_PLANE_BASE: Int = 64
comptime _KIND_MAX: Int = 80

# Where `COLL_CAND_REPORT` writes in an env's `coll_stage` row: the last
# `COLL_REPORT_WORDS` scalars (layout in `ccd_workspace.mojo`). ONE spelling
# for the kernel and the benchmark that decodes it.
comptime COLL_REPORT_BASE: Int = (
    COLL_STAGE_SLOTS * CONTACT_SIZE - COLL_REPORT_WORDS
)


@always_inline
def _sap_pair_listable[
    DTYPE: DType,
    D: DimsLike,
    EX_CAP: Int,
    L_PAIRS: Layout,
    L_MMETA: Layout,
    L_BODIES: Layout,
    L_EXCLUDES: Layout,
](
    si: Int,
    sj: Int,
    si_type: Int,
    sj_type: Int,
    si_body: Int,
    sj_body: Int,
    si_contype: Int,
    si_conaffinity: Int,
    sj_contype: Int,
    sj_conaffinity: Int,
    dims: D,
    pairs: LayoutTensor[DTYPE, L_PAIRS, MutAnyOrigin],
    mmeta: LayoutTensor[DTYPE, L_MMETA, MutAnyOrigin],
    bodies: LayoutTensor[DTYPE, L_BODIES, MutAnyOrigin],
    excludes: LayoutTensor[DTYPE, L_EXCLUDES, MutAnyOrigin],
    ex_sig: Scratch[Int, EX_CAP],
    n_sig: Int,
    nbody: Int,
) -> Bool:
    """Would `_sap_pair_narrow` get past its first rejects for this AABB pair?

    ⚠ A TRANSCRIPTION OF THAT FUNCTION'S HEAD, IN ITS ORDER, THROUGH THE
    FUNCTIONS IT CALLS: the pair canonicalised by (`mj_geom_type_rank`,
    geom index); a predefined `<pair>` skips the filters; otherwise
    `_sap_pair_filter_rejects` — the SAME function `_sap_pair_narrow`
    decides with. False means the narrow phase returns before any geometry,
    emitting nothing and touching no warm slot — so dropping the pair from
    the list is exact. Read by `COLL_CAND_REPORT` (the survivor count) and
    `COLL_PREFILTER` (the listing itself)."""
    var lo = si if si < sj else sj
    var hi = sj if si < sj else si
    var lo_type = si_type if si < sj else sj_type
    var hi_type = sj_type if si < sj else si_type
    var gi = lo
    var gj = hi
    if mj_geom_type_rank(lo_type) > mj_geom_type_rank(hi_type):
        gi = hi
        gj = lo
    if find_predefined_pair[DTYPE](gi, gj, dims, pairs, mmeta) >= 0:
        return True
    var gi_is_si = gi == si
    return not _sap_pair_filter_rejects[DTYPE, EX_CAP=EX_CAP](
        si_body if gi_is_si else sj_body,
        sj_body if gi_is_si else si_body,
        si_contype if gi_is_si else sj_contype,
        si_conaffinity if gi_is_si else sj_conaffinity,
        sj_contype if gi_is_si else si_contype,
        sj_conaffinity if gi_is_si else si_conaffinity,
        bodies, mmeta, excludes, ex_sig, n_sig, nbody,
    )


@always_inline
def _sap_block_candidate[
    DTYPE: DType,
    BATCH: Int,
    D: DimsLike,
    EX_CAP: Int,
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
    L_STAGE: Layout,
    L_WS: Layout,
](
    env: Int,
    wrow: Int,
    hw_row: Int,
    dims: D,
    a: Int,
    b: Int,
    t: Int,
    start: Int,
    a_px: Scalar[DTYPE], a_py: Scalar[DTYPE], a_pz: Scalar[DTYPE],
    a_qx: Scalar[DTYPE], a_qy: Scalar[DTYPE], a_qz: Scalar[DTYPE],
    a_qw: Scalar[DTYPE],
    b_px: Scalar[DTYPE], b_py: Scalar[DTYPE], b_pz: Scalar[DTYPE],
    b_qx: Scalar[DTYPE], b_qy: Scalar[DTYPE], b_qz: Scalar[DTYPE],
    b_qw: Scalar[DTYPE],
    # geom `a`'s body / contype / conaffinity — read by a PLANE candidate only
    a_body: Int,
    a_contype: Int,
    a_conaffinity: Int,
    nbody: Int,
    ex_sig: Scratch[Int, EX_CAP],
    n_sig: Int,
    mut pr: _SapProbe,
    ccd_tol: Scalar[DTYPE],
    ccd_iter: Int,
    multiccd_off: Bool,
    geoms: LayoutTensor[DTYPE, L_GEOMS, MutAnyOrigin],
    bodies: LayoutTensor[DTYPE, L_BODIES, MutAnyOrigin],
    mmeta: LayoutTensor[DTYPE, L_MMETA, MutAnyOrigin],
    excludes: LayoutTensor[DTYPE, L_EXCLUDES, MutAnyOrigin],
    pairs: LayoutTensor[DTYPE, L_PAIRS, MutAnyOrigin],
    mesh_meta: LayoutTensor[DTYPE, L_MESH_META, MutAnyOrigin],
    mesh_verts: LayoutTensor[DTYPE, L_MESH_VERTS, MutAnyOrigin],
    mesh_polys: LayoutTensor[DTYPE, L_MESH_POLYS, MutAnyOrigin],
    mesh_polyvert: LayoutTensor[DTYPE, L_MESH_POLYVERT, MutAnyOrigin],
    mesh_polymap: LayoutTensor[DTYPE, L_MESH_POLYVERT, MutAnyOrigin],
    mesh_vert_polymap: LayoutTensor[DTYPE, L_MESH_VERT_POLYMAP, MutAnyOrigin],
    mesh_vert_edgeadr: LayoutTensor[DTYPE, L_MESH_VERT_EDGEADR, MutAnyOrigin],
    mesh_edges: LayoutTensor[DTYPE, L_MESH_EDGES, MutAnyOrigin],
    hfield_meta: LayoutTensor[DTYPE, L_HF_META, MutAnyOrigin],
    hfield_data: LayoutTensor[DTYPE, L_HF_DATA, MutAnyOrigin],
    stage: LayoutTensor[DTYPE, L_STAGE, MutAnyOrigin],
    ccd_ws: LayoutTensor[DTYPE, L_WS, MutAnyOrigin],
) -> Int:
    """One listed candidate's narrow phase into its staging window
    `stage[env, start ..]`; returns the records it wrote.

    ⚠ ONE DISPATCH FOR BOTH GPU LAYOUTS. The block kernel's phase 2 runs a
    candidate on the thread of its lane, the flat narrow phase
    (`COLL_FLAT_NARROW`) on a lane of whatever warp its queue hands it to;
    both call this, so the plane/pair split and its arguments exist once.
    `t < 0` is a plane candidate (`a` the plane), as the listing writes it."""
    var num_contacts = start
    var win_end = start + COLL_STAGE_MAXC
    if t < 0:
        # the plane's own data, as the serial loop head computes it once per
        # plane
        var pn = plane_world_normal[DTYPE](a_qx, a_qy, a_qz, a_qw)
        _sap_plane_narrow[
            DTYPE, BATCH, D, EX_CAP, HFIELD_ENABLED=False
        ](
            env, env, wrow, dims, a, b, a_body, a_contype, a_conaffinity,
            a_px, a_py, a_pz, a_qx, a_qy, a_qz, a_qw,
            pn, nbody, win_end, ex_sig, n_sig, pr, num_contacts,
            b_px, b_py, b_pz, b_qx, b_qy, b_qz, b_qw,
            geoms, bodies, mmeta, excludes, pairs, mesh_meta, mesh_verts, mesh_vert_edgeadr, mesh_edges, stage, ccd_ws,
            hw_row=hw_row,
        )
    else:
        _sap_pair_narrow[
            DTYPE, BATCH, D, EX_CAP, HFIELD_ENABLED=False
        ](
            env, env, wrow, dims, a, b, t, nbody, win_end,
            ex_sig, n_sig, pr, num_contacts,
            a_px, a_py, a_pz, a_qx, a_qy, a_qz, a_qw,
            b_px, b_py, b_pz, b_qx, b_qy, b_qz, b_qw,
            ccd_tol, ccd_iter, multiccd_off,
            geoms, bodies, mmeta, excludes, pairs, mesh_meta, mesh_verts, mesh_polys, mesh_polyvert, mesh_polymap, mesh_vert_polymap, mesh_vert_edgeadr, mesh_edges, hfield_meta, hfield_data, stage, ccd_ws,
            hw_row=hw_row,
        )
    return num_contacts - start


@always_inline
def _sap_block_output[
    DTYPE: DType,
    NC: Int,
    L_STAGE: Layout,
    L_CONTACTS: Layout,
    L_SMETA: Layout,
](
    env: Int,
    tid: Int,
    ncand: Int,
    max_contacts: Int,
    nbody: Int,
    cand_sh: LayoutTensor[
        DTYPE, Layout.row_major(6 * NC), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ],
    sk_sh: LayoutTensor[
        DTYPE, Layout.row_major(NC), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ],
    ctrl_sh: LayoutTensor[
        DTYPE, Layout.row_major(8), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ],
    stage: LayoutTensor[DTYPE, L_STAGE, MutAnyOrigin],
    contacts: LayoutTensor[DTYPE, L_CONTACTS, MutAnyOrigin],
    smeta: LayoutTensor[DTYPE, L_SMETA, MutAnyOrigin],
):
    """The block kernel's phase 3 — offsets (thread 0), the MuJoCo-order sort
    as ranks, the copy, `ncon` — on EVERY thread of the block (it barriers).

    Reads, per candidate `c < ncand`: its staging start (records) at
    `cand_sh[3*NC + c]` and its record count at `cand_sh[5*NC + c]`;
    `ctrl_sh[1] != 0` sends the env to the serial fallback. The fused block
    kernel calls it after its own phase 2, the flat narrow phase's compaction
    kernel (`COLL_FLAT_NARROW`) after loading the same slots from
    `Data.coll_flat` — one output rule for both."""
    if tid == 0:
        if Int(rebind[Scalar[DTYPE]](ctrl_sh[1])) != 0:
            # ⚠ THE FALLBACK IS A SECOND LAUNCH, NOT A CALL. Calling the
            # serial per-env function from here put a SECOND copy of the
            # narrow phase in this kernel, and on Metal that corrupted the
            # FIRST: the plane-mesh fixture came back with three contacts
            # instead of one, the third one different between two runs —
            # the per-thread miscompute `feedback_metal_wide_per_thread_
            # inlinearray_miscompute` records, with no crash. With one copy
            # per kernel the block path is bit-exact. So this env is MARKED
            # (`ncon = -1`) and `detect_contacts_sap` launches the serial
            # kernel with `ONLY_FLAGGED=True` right after, which runs it for
            # marked envs only and overwrites the mark with the real count.
            smeta[env, META_IDX_NUM_CONTACTS] = Scalar[DTYPE](-1)
            ctrl_sh[3] = Scalar[DTYPE](0)
        else:
            # Destination offset per candidate: a prefix over the counts,
            # capped at `max_contacts` contact by contact (the serial guard's
            # semantics), parked in the candidate list's thread slot, which
            # is done with.
            var n = 0
            for c in range(ncand):
                var cnt = Int(rebind[Scalar[DTYPE]](cand_sh[5 * NC + c]))
                if n + cnt > max_contacts:
                    cnt = max_contacts - n
                    cand_sh[5 * NC + c] = Scalar[DTYPE](cnt)
                cand_sh[4 * NC + c] = Scalar[DTYPE](n)
                n += cnt
            ctrl_sh[3] = Scalar[DTYPE](n)
    barrier()
    # ⚠ THE MuJoCo-ORDER SORT IS A RANK, NOT AN INSERTION SORT ON THREAD 0
    # (2026-09-26). `sort_contacts_mujoco_order` orders the compacted array
    # STABLY by body pair; every contact of a candidate carries the same pair
    # (one geom pair), so that order is the candidates' own, stably by pair,
    # each keeping its records in narrow-phase order. So each candidate's
    # destination is the count of contacts of the candidates ahead of it —
    # smaller pair, or the same pair listed earlier — and the copy lands the
    # records sorted. Thread 0 moved whole records in global memory, O(n^2)
    # on a crowded env (152 us at the tower's worst). The key is read from
    # the candidate's first staged record, the field the sort itself reads.
    var fb = Int(rebind[Scalar[DTYPE]](ctrl_sh[1])) != 0
    if not fb:
        for c in range(tid, ncand, COLL_TPB):
            var key = Scalar[DTYPE](-1)
            if Int(rebind[Scalar[DTYPE]](cand_sh[5 * NC + c])) > 0:
                var r0 = Int(rebind[Scalar[DTYPE]](cand_sh[3 * NC + c])) * CONTACT_SIZE
                var ba = Int(rebind[Scalar[DTYPE]](stage[env, r0 + CONTACT_IDX_BODY_A]))
                var bb = Int(rebind[Scalar[DTYPE]](stage[env, r0 + CONTACT_IDX_BODY_B]))
                if ba < 0:
                    ba = 0
                if bb < 0:
                    bb = 0
                var lo = ba if ba < bb else bb
                var hi = bb if ba < bb else ba
                key = Scalar[DTYPE](lo * (nbody + 1) + hi)
            sk_sh[c] = key
    barrier()
    if not fb:
        for c in range(tid, ncand, COLL_TPB):
            var k_c = rebind[Scalar[DTYPE]](sk_sh[c])
            var dst = 0
            for c2 in range(ncand):
                var k2 = rebind[Scalar[DTYPE]](sk_sh[c2])
                if k2 < k_c or (k2 == k_c and c2 < c):
                    dst += Int(rebind[Scalar[DTYPE]](cand_sh[5 * NC + c2]))
            cand_sh[4 * NC + c] = Scalar[DTYPE](dst)
    barrier()
    # Every thread copies records; the order is fixed by the offsets.
    var n_out = Int(rebind[Scalar[DTYPE]](ctrl_sh[3]))
    if not fb:
        for c in range(ncand):
            var start = Int(rebind[Scalar[DTYPE]](cand_sh[3 * NC + c]))
            var cnt = Int(rebind[Scalar[DTYPE]](cand_sh[5 * NC + c]))
            var dst0 = Int(rebind[Scalar[DTYPE]](cand_sh[4 * NC + c]))
            for q in range(tid, cnt * CONTACT_SIZE, COLL_TPB):
                var k = q // CONTACT_SIZE
                var f = q - k * CONTACT_SIZE
                contacts[env, (dst0 + k) * CONTACT_SIZE + f] = rebind[
                    Scalar[DTYPE]
                ](stage[env, (start + k) * CONTACT_SIZE + f])
    barrier()
    if tid == 0 and not fb:
        smeta[env, META_IDX_NUM_CONTACTS] = Scalar[DTYPE](n_out)


def _detect_contacts_sap_block_kernel[
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
    # List the candidates into `coll_flat` and return (`COLL_FLAT_NARROW`).
    FLAT_LIST: Bool = False,
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
    stage: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, COLL_STAGE_SLOTS * CONTACT_SIZE),
        MutAnyOrigin,
    ],
    # `Data.coll_flat` — read and written only with `FLAT_LIST`
    coll_flat: LayoutTensor[
        DTYPE, Layout.row_major(coll_flat_words(BATCH)), MutAnyOrigin
    ],
):
    """ONE BLOCK PER ENV, `COLL_TPB` threads over the candidate pairs.

    The per-env serial kernel spends its wall time on one thread walking the
    geom table, the sweep and every GJK hill climb one dependent global load
    at a time — 430 µs per env at the k=13 park scene against 10 µs on a CPU
    core (block ledger §6), and at 1024 lanes of a sprawled G1 a warp of 32
    such threads takes the time of its slowest env (PERFORMANCE.md §13.51).
    Here: phase 0 computes world poses and AABBs one geom per thread into
    threadgroup memory; phase 1 lists the CANDIDATES in the serial EMISSION
    ORDER — the plane phase gated by `_sap_plane_gate` on all threads
    (the serial loop hands every geom to the plane; the gate keeps the few
    near it), then thread 0's pair-margin inflation, sweep list, sort and
    sweep (the AABB tests and the `break` only) — each with a staging
    window at its emission offset; then sorts the list by KIND (the geom
    type pair, `_KIND_*`) with a counting sort. Phase 2 runs the candidates
    in that kind order, `COLL_TPB` at a time: candidate `p` of the sorted
    list on thread `p % COLL_TPB` in round `p // COLL_TPB`, so the lanes of
    a warp are on the SAME narrow-phase routine (mesh-mesh next to
    mesh-mesh) and diverge only on their trip counts, not on their code —
    the 2026-09-07 layout put a GJK lane beside 28 cheap lanes and read
    2x the serial time for four candidates. Every thread has its own CCD
    row (`COLL_CCD_LANES == COLL_TPB`); the hill climb's cross-step warm
    slots stay on lane 0's row of the env (`hw_row`), so a pair finds its
    previous vertex whatever lane it lands on. Phase 3 (thread 0) compacts
    the windows in candidate order into `contacts[env]`, which reproduces
    the serial array bit for bit, then the MuJoCo-order sort and `ncon`.

    ⚠ EXACT OR SERIAL, NEVER APPROXIMATE. A candidate list past
    `COLL_NCAND_CAP` or a routine that filled its whole window (it may have
    been truncated) sends the env through `_detect_contacts_sap_env` on
    thread 0 — the serial kernel, on lane 0's CCD row.

    ⚠ NO HEIGHTFIELDS: `_hfield_contacts` reads per-env samples through the
    same index it writes contacts with; the dispatch keeps such models on the
    serial kernel and this kernel compiles the heightfield branch out."""
    var env = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var _tt0: Int = 0
    var _tt1: Int = 0
    var _tt2: Int = 0
    var _tt3: Int = 0
    comptime if COLL_TIMING:
        comptime assert not COLL_CAND_REPORT, (
            "COLL_TIMING and COLL_CAND_REPORT write the same stage tail"
        )
        _tt0 = perf_counter_ns()
    if env >= BATCH:
        return
    comptime NG = NGEOM if NGEOM > 0 else 1
    comptime NC = COLL_NCAND_CAP
    comptime EX_CAP = cap[NEXCLUDE]() if may_exist[NEXCLUDE]() else 1
    var dims = Dims[nq=NQ, nv=NV, nbody=NBODY, njoint=NJOINT, max_contacts=MAX_CONTACTS, ngeom=NGEOM, nexclude=NEXCLUDE, nmesh_verts=NMESH_VERTS, npair=NPAIR]()
    var ngeom = NGEOM
    var nbody = NBODY
    var max_contacts = MAX_CONTACTS
    var npair = NPAIR

    # ── threadgroup memory ───────────────────────────────────────────────
    var wp_sh = LayoutTensor[
        DTYPE, Layout.row_major(7 * NG), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var ab_sh = LayoutTensor[
        DTYPE, Layout.row_major(6 * NG), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var idx_sh = LayoutTensor[
        DTYPE, Layout.row_major(2 * NG), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    # geom type / body / contype / conaffinity, so thread 0's candidate
    # generation reads its per-pair fields from threadgroup memory instead
    # of one dependent global load per field per pair.
    var gf_sh = LayoutTensor[
        DTYPE, Layout.row_major(4 * NG), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    # candidate list: a, b, si_type (-1 = plane candidate), staging offset,
    # kind key (phase 3 reuses the slot for the destination offset),
    # emitted count
    var cand_sh = LayoutTensor[
        DTYPE, Layout.row_major(6 * NC), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    # the candidates in kind order (indices into the list above)
    var ord_sh = LayoutTensor[
        DTYPE, Layout.row_major(NC), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var kcnt_sh = LayoutTensor[
        DTYPE, Layout.row_major(_KIND_MAX), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    # the plane gate's verdict per geom, for the plane being listed; then
    # the sweep's per-row survivor count and first candidate slot
    var pf_sh = LayoutTensor[
        DTYPE, Layout.row_major(NG), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    # [0] ncand [1] overflow [2] n_sig [3] n_out; the sweep's hand-over:
    # [4] sap_n [5] candidates listed before it [6] their overflow
    # [7] candidates it lists, uncapped
    var ctrl_sh = LayoutTensor[
        DTYPE, Layout.row_major(8), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    # the sweep's survivors, one bit per sorted (i, j), row-major
    comptime MW = (NG + 31) // 32
    var mask_sh = LayoutTensor[
        DType.uint32, Layout.row_major(NG * MW), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    # phase 3's sort key per candidate: its body pair
    var sk_sh = LayoutTensor[
        DTYPE, Layout.row_major(NC), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    # `COLL_CAND_REPORT`'s per-thread sweep counters, summed by thread 0
    var rep_sh = LayoutTensor[
        DTYPE, Layout.row_major(4 * COLL_TPB), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    # `<exclude>` signatures, sorted once per block by thread 0 (they are a
    # per-model table; every thread used to sort its own copy).
    var ex_sh = LayoutTensor[
        DTYPE, Layout.row_major(EX_CAP), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()

    # ── phase 0: world pose and AABB, one geom per thread ────────────────
    for g in range(tid, ngeom, COLL_TPB):
        var px: Scalar[DTYPE] = 0
        var py: Scalar[DTYPE] = 0
        var pz: Scalar[DTYPE] = 0
        var qx: Scalar[DTYPE] = 0
        var qy: Scalar[DTYPE] = 0
        var qz: Scalar[DTYPE] = 0
        var qw: Scalar[DTYPE] = 1
        _geom_world_pos[DTYPE](
            env, g, geoms, xpos, xquat, px, py, pz, qx, qy, qz, qw
        )
        wp_sh[0 * NG + g] = px
        wp_sh[1 * NG + g] = py
        wp_sh[2 * NG + g] = pz
        wp_sh[3 * NG + g] = qx
        wp_sh[4 * NG + g] = qy
        wp_sh[5 * NG + g] = qz
        wp_sh[6 * NG + g] = qw
        var gt = Int(rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_TYPE]))
        gf_sh[0 * NG + g] = Scalar[DTYPE](gt)
        gf_sh[1 * NG + g] = geoms[g, GEOM_IDX_BODY]
        gf_sh[2 * NG + g] = geoms[g, GEOM_IDX_CONTYPE]
        gf_sh[3 * NG + g] = geoms[g, GEOM_IDX_CONAFFINITY]
        if gt == GEOM_PLANE:
            continue
        var r = rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_RADIUS])
        var hl = rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_HALF_LENGTH])
        var hx = rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_HALF_X])
        var hy = rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_HALF_Y])
        var hz = rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_HALF_Z])
        var rb = rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_RBOUND])
        var he = _aabb_half_extents[DTYPE](
            gt, qx, qy, qz, qw, r, hl, hx, hy, hz, rb
        )
        var gm = (
            rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_MARGIN])
            + rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_GAP])
        )
        if gm < Scalar[DTYPE](0):
            gm = Scalar[DTYPE](0)
        ab_sh[0 * NG + g] = px - he[0] - gm
        ab_sh[1 * NG + g] = px + he[0] + gm
        ab_sh[2 * NG + g] = py - he[1] - gm
        ab_sh[3 * NG + g] = py + he[1] + gm
        ab_sh[4 * NG + g] = pz - he[2] - gm
        ab_sh[5 * NG + g] = pz + he[2] + gm
    barrier()
    comptime if COLL_TIMING:
        _tt1 = perf_counter_ns()
    comptime if COLL_STOP_AFTER == 1:
        if tid == 0:
            smeta[env, META_IDX_NUM_CONTACTS] = Scalar[DTYPE](0)
        return

    # ── phase 1a: the model's `<exclude>` signatures, once per block ─────
    if tid == 0:
        var ex0 = Scratch[Int, EX_CAP](
            NEXCLUDE if NEXCLUDE > 0 else 1, fill=0
        )
        var n_sig0 = exclude_signatures[DTYPE, EX_CAP](
            nbody, NEXCLUDE, mmeta, excludes, ex0
        )
        for k in range(EX_CAP):
            ex_sh[k] = Scalar[DTYPE](ex0[k])
        ctrl_sh[2] = Scalar[DTYPE](n_sig0)
    barrier()
    # Every thread's private copy (the helpers take a `Scratch`; it is one
    # slot on a model with no `<exclude>`).
    var ex_sig = Scratch[Int, EX_CAP](
        NEXCLUDE if NEXCLUDE > 0 else 1, fill=0
    )
    for k in range(EX_CAP):
        ex_sig[k] = Int(rebind[Scalar[DTYPE]](ex_sh[k]))
    var n_sig = Int(rebind[Scalar[DTYPE]](ctrl_sh[2]))

    # ── phase 1b: candidates, in the serial emission order ───────────────
    # The push state is thread 0's. Every thread declares the variables so
    # the closure can be defined at block scope: the plane phase alternates
    # all-thread gating and thread-0 pushes across barriers.
    var ncand = 0
    var overflow = 0
    var off = 0
    # `COLL_CAND_REPORT`'s thread-0 counters. Declared for every build and
    # consumed below either way, so the production build compiles them out.
    var rep_tests = 0
    var rep_shifts = 0
    var rep_sap_n = 0
    var rep_aabb_all = 0
    var rep_survive = 0
    var rep_planes = 0

    @always_inline
    def _push(a: Int, b: Int, t: Int, key: Int) {mut ncand, mut off, mut overflow, imm}:
        if ncand >= NC:
            overflow = 1
            return
        cand_sh[0 * NC + ncand] = Scalar[DTYPE](a)
        cand_sh[1 * NC + ncand] = Scalar[DTYPE](b)
        cand_sh[2 * NC + ncand] = Scalar[DTYPE](t)
        cand_sh[3 * NC + ncand] = Scalar[DTYPE](off)
        cand_sh[4 * NC + ncand] = Scalar[DTYPE](key)
        cand_sh[5 * NC + ncand] = Scalar[DTYPE](0)
        off += COLL_STAGE_MAXC
        ncand += 1

    # 3. plane vs non-plane, the serial loop's order. All threads gate
    # their share of the geoms against the plane; thread 0 lists the
    # survivors in geom order. (`continue` is uniform: the type is shared.)
    for gi in range(ngeom):
        if Int(rebind[Scalar[DTYPE]](gf_sh[0 * NG + gi])) != GEOM_PLANE:
            continue
        var gi_body = Int(rebind[Scalar[DTYPE]](gf_sh[1 * NG + gi]))
        var gi_contype = Int(rebind[Scalar[DTYPE]](gf_sh[2 * NG + gi]))
        var gi_conaffinity = Int(rebind[Scalar[DTYPE]](gf_sh[3 * NG + gi]))
        for gj in range(tid, ngeom, COLL_TPB):
            var g8 = _sap_plane_gate[DTYPE, BATCH, type_of(dims), EX_CAP](
                gi, gj, gi_body, gi_contype, gi_conaffinity,
                rebind[Scalar[DTYPE]](wp_sh[0 * NG + gi]),
                rebind[Scalar[DTYPE]](wp_sh[1 * NG + gi]),
                rebind[Scalar[DTYPE]](wp_sh[2 * NG + gi]),
                rebind[Scalar[DTYPE]](wp_sh[3 * NG + gi]),
                rebind[Scalar[DTYPE]](wp_sh[4 * NG + gi]),
                rebind[Scalar[DTYPE]](wp_sh[5 * NG + gi]),
                rebind[Scalar[DTYPE]](wp_sh[6 * NG + gi]),
                rebind[Scalar[DTYPE]](wp_sh[0 * NG + gj]),
                rebind[Scalar[DTYPE]](wp_sh[1 * NG + gj]),
                rebind[Scalar[DTYPE]](wp_sh[2 * NG + gj]),
                rebind[Scalar[DTYPE]](wp_sh[3 * NG + gj]),
                rebind[Scalar[DTYPE]](wp_sh[4 * NG + gj]),
                rebind[Scalar[DTYPE]](wp_sh[5 * NG + gj]),
                rebind[Scalar[DTYPE]](wp_sh[6 * NG + gj]),
                dims, nbody, ex_sig, n_sig,
                geoms, bodies, mmeta, excludes, pairs,
            )
            pf_sh[gj] = Scalar[DTYPE](1) if g8.ok else Scalar[DTYPE](0)
        barrier()
        if tid == 0:
            for gj in range(ngeom):
                if rebind[Scalar[DTYPE]](pf_sh[gj]) != Scalar[DTYPE](0):
                    var gj_type = Int(rebind[Scalar[DTYPE]](gf_sh[0 * NG + gj]))
                    _push(gi, gj, -1, _KIND_PLANE_BASE + mj_geom_type_rank(gj_type))
                    comptime if COLL_CAND_REPORT:
                        rep_planes += 1
        barrier()

    if tid == 0:
        var n_pair_aabb = Int(rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_NPAIR]))
        if n_pair_aabb > npair:
            n_pair_aabb = npair
        for p in range(n_pair_aabb):
            # `margin + gap`, as in the per-env builder above (AUD-36).
            var pm = (
                rebind[Scalar[DTYPE]](pairs[p, PAIR_IDX_MARGIN])
                + rebind[Scalar[DTYPE]](pairs[p, PAIR_IDX_GAP])
            )
            if pm <= Scalar[DTYPE](0):
                continue
            for side in range(2):
                var g = Int(
                    rebind[Scalar[DTYPE]](
                        pairs[p, PAIR_IDX_GEOM1 if side == 0 else PAIR_IDX_GEOM2]
                    )
                )
                if Int(rebind[Scalar[DTYPE]](gf_sh[0 * NG + g])) == GEOM_PLANE:
                    continue
                ab_sh[0 * NG + g] = rebind[Scalar[DTYPE]](ab_sh[0 * NG + g]) - pm
                ab_sh[1 * NG + g] = rebind[Scalar[DTYPE]](ab_sh[1 * NG + g]) + pm
                ab_sh[2 * NG + g] = rebind[Scalar[DTYPE]](ab_sh[2 * NG + g]) - pm
                ab_sh[3 * NG + g] = rebind[Scalar[DTYPE]](ab_sh[3 * NG + g]) + pm
                ab_sh[4 * NG + g] = rebind[Scalar[DTYPE]](ab_sh[4 * NG + g]) - pm
                ab_sh[5 * NG + g] = rebind[Scalar[DTYPE]](ab_sh[5 * NG + g]) + pm
        # 4a. the sweep list — `pair_geom` parked in `idx_sh[NG + g]`
        for g in range(ngeom):
            idx_sh[NG + g] = Scalar[DTYPE](0)
        for p in range(n_pair_aabb):
            for side in range(2):
                var pg = Int(
                    rebind[Scalar[DTYPE]](
                        pairs[p, PAIR_IDX_GEOM1 if side == 0 else PAIR_IDX_GEOM2]
                    )
                )
                if pg >= 0 and pg < ngeom:
                    idx_sh[NG + pg] = Scalar[DTYPE](1)
        var sap_n = 0
        for g in range(ngeom):
            var gt = Int(rebind[Scalar[DTYPE]](gf_sh[0 * NG + g]))
            if gt == GEOM_PLANE:
                continue
            var g_ct = Int(rebind[Scalar[DTYPE]](gf_sh[2 * NG + g]))
            var g_ca = Int(rebind[Scalar[DTYPE]](gf_sh[3 * NG + g]))
            if g_ct == 0 and g_ca == 0 and Int(rebind[Scalar[DTYPE]](idx_sh[NG + g])) == 0:
                continue
            idx_sh[sap_n] = Scalar[DTYPE](g)
            sap_n += 1
        comptime if COLL_CAND_REPORT:
            rep_sap_n = sap_n
        ctrl_sh[4] = Scalar[DTYPE](sap_n)
        ctrl_sh[5] = Scalar[DTYPE](ncand)
        ctrl_sh[6] = Scalar[DTYPE](overflow)
    barrier()
    # ⚠ 4b-4c RUN ON EVERY THREAD, AND LIST WHAT THREAD 0's SERIAL SWEEP
    # LISTED, IN ITS ORDER (2026-09-26). The serial sweep was ~106 us of the
    # tower's ~220 us mean env (`COLL_TIMING`), most of it the listing filter's
    # global loads, one pair after another on thread 0.
    var sap_n = Int(rebind[Scalar[DTYPE]](ctrl_sh[4]))
    # 4b. the sweep order by aabb_min_x: a RANK sort, into `idx_sh[NG ..]`
    # (the list build above is done with it). A geom's rank counts the
    # smaller keys and the EQUAL keys listed before it — the order the
    # insertion sort leaves, which shifts only past a strictly greater key.
    for i in range(tid, sap_n, COLL_TPB):
        var g_i = Int(rebind[Scalar[DTYPE]](idx_sh[i]))
        var v_i = rebind[Scalar[DTYPE]](ab_sh[0 * NG + g_i])
        var r = 0
        for k in range(sap_n):
            var v_k = rebind[Scalar[DTYPE]](
                ab_sh[0 * NG + Int(rebind[Scalar[DTYPE]](idx_sh[k]))]
            )
            if v_k < v_i or (v_k == v_i and k < i):
                r += 1
            comptime if COLL_CAND_REPORT:
                # the insertion sort's shifts are the inversions
                if k < i and v_k > v_i:
                    rep_shifts += 1
        idx_sh[NG + r] = Scalar[DTYPE](g_i)
    barrier()
    # 4c. the sweep, one sorted row `i` per thread: the AABB tests, the break
    # and the listing filter. The survivors are marked in row i's bit mask
    # and counted; a prefix over the rows (thread 0) gives each row its
    # first candidate slot, and each row lists its marks in `j` order — so
    # the list is the serial `_push` order, (i, j) lexicographic, after the
    # plane candidates.
    for i in range(tid, sap_n, COLL_TPB):
        for w in range(MW):
            mask_sh[i * MW + w] = 0
        var si = Int(rebind[Scalar[DTYPE]](idx_sh[NG + i]))
        var si_max_x = rebind[Scalar[DTYPE]](ab_sh[1 * NG + si])
        var si_type = Int(rebind[Scalar[DTYPE]](gf_sh[0 * NG + si]))
        var cnt_i = 0
        for j in range(i + 1, sap_n):
            comptime if COLL_CAND_REPORT:
                rep_tests += 1
            var sj = Int(rebind[Scalar[DTYPE]](idx_sh[NG + j]))
            if rebind[Scalar[DTYPE]](ab_sh[0 * NG + sj]) > si_max_x:
                break
            if (
                rebind[Scalar[DTYPE]](ab_sh[2 * NG + sj]) > rebind[Scalar[DTYPE]](ab_sh[3 * NG + si])
                or rebind[Scalar[DTYPE]](ab_sh[2 * NG + si]) > rebind[Scalar[DTYPE]](ab_sh[3 * NG + sj])
            ):
                continue
            if (
                rebind[Scalar[DTYPE]](ab_sh[4 * NG + sj]) > rebind[Scalar[DTYPE]](ab_sh[5 * NG + si])
                or rebind[Scalar[DTYPE]](ab_sh[4 * NG + si]) > rebind[Scalar[DTYPE]](ab_sh[5 * NG + sj])
            ):
                continue
            comptime if COLL_CAND_REPORT or COLL_PREFILTER:
                var keep = _sap_pair_listable[DTYPE, EX_CAP=EX_CAP](
                    si, sj, si_type,
                    Int(rebind[Scalar[DTYPE]](gf_sh[0 * NG + sj])),
                    Int(rebind[Scalar[DTYPE]](gf_sh[1 * NG + si])),
                    Int(rebind[Scalar[DTYPE]](gf_sh[1 * NG + sj])),
                    Int(rebind[Scalar[DTYPE]](gf_sh[2 * NG + si])),
                    Int(rebind[Scalar[DTYPE]](gf_sh[3 * NG + si])),
                    Int(rebind[Scalar[DTYPE]](gf_sh[2 * NG + sj])),
                    Int(rebind[Scalar[DTYPE]](gf_sh[3 * NG + sj])),
                    dims, pairs, mmeta, bodies, excludes, ex_sig, n_sig,
                    nbody,
                )
                comptime if COLL_CAND_REPORT:
                    rep_aabb_all += 1
                    if keep:
                        rep_survive += 1
                comptime if COLL_PREFILTER:
                    if not keep:
                        continue
            mask_sh[i * MW + j // 32] = mask_sh[i * MW + j // 32] | (
                UInt32(1) << UInt32(j % 32)
            )
            cnt_i += 1
        pf_sh[i] = Scalar[DTYPE](cnt_i)
    barrier()
    if tid == 0:
        var run = Int(rebind[Scalar[DTYPE]](ctrl_sh[5]))
        for i in range(sap_n):
            var c_i = Int(rebind[Scalar[DTYPE]](pf_sh[i]))
            pf_sh[i] = Scalar[DTYPE](run)
            run += c_i
        ctrl_sh[7] = Scalar[DTYPE](run)
    barrier()
    for i in range(tid, sap_n, COLL_TPB):
        var si = Int(rebind[Scalar[DTYPE]](idx_sh[NG + i]))
        var si_type = Int(rebind[Scalar[DTYPE]](gf_sh[0 * NG + si]))
        var si_rank = mj_geom_type_rank(si_type)
        var pos = Int(rebind[Scalar[DTYPE]](pf_sh[i]))
        for w in range(MW):
            var bits = rebind[UInt32](mask_sh[i * MW + w])
            while bits != 0:
                var j = w * 32 + Int(count_trailing_zeros(bits))
                bits = bits & (bits - 1)
                # Past the cap the serial `_push` drops the pair and flags
                # the env for the serial fallback — `total` below.
                if pos < NC:
                    var sj = Int(rebind[Scalar[DTYPE]](idx_sh[NG + j]))
                    var sj_rank = mj_geom_type_rank(
                        Int(rebind[Scalar[DTYPE]](gf_sh[0 * NG + sj]))
                    )
                    var key = (
                        si_rank * 8 + sj_rank if si_rank <= sj_rank
                        else sj_rank * 8 + si_rank
                    )
                    cand_sh[0 * NC + pos] = Scalar[DTYPE](si)
                    cand_sh[1 * NC + pos] = Scalar[DTYPE](sj)
                    cand_sh[2 * NC + pos] = Scalar[DTYPE](si_type)
                    cand_sh[3 * NC + pos] = Scalar[DTYPE](pos * COLL_STAGE_MAXC)
                    cand_sh[4 * NC + pos] = Scalar[DTYPE](key)
                    cand_sh[5 * NC + pos] = Scalar[DTYPE](0)
                pos += 1
    comptime if COLL_CAND_REPORT:
        rep_sh[4 * tid + 0] = Scalar[DTYPE](rep_tests)
        rep_sh[4 * tid + 1] = Scalar[DTYPE](rep_shifts)
        rep_sh[4 * tid + 2] = Scalar[DTYPE](rep_aabb_all)
        rep_sh[4 * tid + 3] = Scalar[DTYPE](rep_survive)
    barrier()
    if tid == 0:
        var total = Int(rebind[Scalar[DTYPE]](ctrl_sh[7]))
        ncand = total if total < NC else NC
        overflow = Int(rebind[Scalar[DTYPE]](ctrl_sh[6]))
        if total > NC:
            overflow = 1
        comptime if COLL_CAND_REPORT:
            rep_tests = 0
            rep_shifts = 0
            rep_aabb_all = 0
            rep_survive = 0
            for t in range(COLL_TPB):
                rep_tests += Int(rebind[Scalar[DTYPE]](rep_sh[4 * t + 0]))
                rep_shifts += Int(rebind[Scalar[DTYPE]](rep_sh[4 * t + 1]))
                rep_aabb_all += Int(rebind[Scalar[DTYPE]](rep_sh[4 * t + 2]))
                rep_survive += Int(rebind[Scalar[DTYPE]](rep_sh[4 * t + 3]))
        # 5. the kind order: a counting sort on the keys, stable, so two
        # candidates of one kind keep their emission order.
        for k in range(_KIND_MAX):
            kcnt_sh[k] = Scalar[DTYPE](0)
        for c in range(ncand):
            var key = Int(rebind[Scalar[DTYPE]](cand_sh[4 * NC + c]))
            kcnt_sh[key] = rebind[Scalar[DTYPE]](kcnt_sh[key]) + Scalar[DTYPE](1)
        var run = 0
        for k in range(_KIND_MAX):
            var n = Int(rebind[Scalar[DTYPE]](kcnt_sh[k]))
            kcnt_sh[k] = Scalar[DTYPE](run)
            run += n
        for c in range(ncand):
            var key = Int(rebind[Scalar[DTYPE]](cand_sh[4 * NC + c]))
            var pos = Int(rebind[Scalar[DTYPE]](kcnt_sh[key]))
            ord_sh[pos] = Scalar[DTYPE](c)
            kcnt_sh[key] = Scalar[DTYPE](pos + 1)
        ctrl_sh[0] = Scalar[DTYPE](ncand)
        ctrl_sh[1] = Scalar[DTYPE](overflow)
    barrier()
    # From here every thread reads the block's count, not its own copy.
    ncand = Int(rebind[Scalar[DTYPE]](ctrl_sh[0]))
    overflow = Int(rebind[Scalar[DTYPE]](ctrl_sh[1]))
    comptime if COLL_TIMING:
        _tt2 = perf_counter_ns()
    comptime if COLL_STOP_AFTER == 2:
        if tid == 0:
            smeta[env, META_IDX_NUM_CONTACTS] = Scalar[DTYPE](0)
        return

    comptime if FLAT_LIST:
        comptime assert not (
            COLL_TIMING or COLL_CAND_REPORT or COLL_STOP_AFTER != 0
        ), (
            "the flat narrow phase has no timing / report / stop instrument:"
            " time its kernels with nsys"
        )
        _sap_flat_list_env[DTYPE, NC, BATCH](
            env, tid, ncand, overflow, cand_sh, ord_sh, coll_flat
        )
        _ = rep_tests
        _ = rep_shifts
        _ = rep_sap_n
        _ = rep_aabb_all
        _ = rep_survive
        _ = rep_planes
    else:
        # ── phase 2: the candidates in kind order, COLL_TPB per round ────────
        if overflow == 0:
            var ccd_tol = rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_CCD_TOLERANCE])
            if ccd_tol <= 0:
                ccd_tol = Scalar[DTYPE](MJ_CCD_TOLERANCE)
            var ccd_iter = Int(
                rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_CCD_ITERATIONS])
            )
            if ccd_iter < 1:
                ccd_iter = MJ_CCD_ITERATIONS
            var multiccd_off = (
                rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_MULTICCD_DISABLED]) != 0
            )
            var pr = _SapProbe()
            # Every thread its own CCD row; the warm slots on the env's first.
            comptime assert COLL_CCD_LANES == COLL_TPB, (
                "the block collision kernel gives every thread a CCD row:"
                " COLL_CCD_LANES must equal COLL_TPB (ccd_workspace.mojo)"
            )
            var wrow = env * COLL_CCD_LANES + tid
            var hw_row = env * COLL_CCD_LANES
            var full = 0
            var _lane_t0: Int = 0
            var _lane_n = 0
            var _lane_max: Int = 0
            var _lane_max_key = -1
            comptime if COLL_TIMING:
                _lane_t0 = perf_counter_ns()
            for p in range(tid, ncand, COLL_TPB):
                var c = Int(rebind[Scalar[DTYPE]](ord_sh[p]))
                var _c_t0: Int = 0
                comptime if COLL_TIMING:
                    _c_t0 = perf_counter_ns()
                var a = Int(rebind[Scalar[DTYPE]](cand_sh[0 * NC + c]))
                var b = Int(rebind[Scalar[DTYPE]](cand_sh[1 * NC + c]))
                var t = Int(rebind[Scalar[DTYPE]](cand_sh[2 * NC + c]))
                var start = Int(rebind[Scalar[DTYPE]](cand_sh[3 * NC + c]))
                var cnt = _sap_block_candidate[DTYPE, BATCH, EX_CAP=EX_CAP](
                    env, wrow, hw_row, dims, a, b, t, start,
                    rebind[Scalar[DTYPE]](wp_sh[0 * NG + a]),
                    rebind[Scalar[DTYPE]](wp_sh[1 * NG + a]),
                    rebind[Scalar[DTYPE]](wp_sh[2 * NG + a]),
                    rebind[Scalar[DTYPE]](wp_sh[3 * NG + a]),
                    rebind[Scalar[DTYPE]](wp_sh[4 * NG + a]),
                    rebind[Scalar[DTYPE]](wp_sh[5 * NG + a]),
                    rebind[Scalar[DTYPE]](wp_sh[6 * NG + a]),
                    rebind[Scalar[DTYPE]](wp_sh[0 * NG + b]),
                    rebind[Scalar[DTYPE]](wp_sh[1 * NG + b]),
                    rebind[Scalar[DTYPE]](wp_sh[2 * NG + b]),
                    rebind[Scalar[DTYPE]](wp_sh[3 * NG + b]),
                    rebind[Scalar[DTYPE]](wp_sh[4 * NG + b]),
                    rebind[Scalar[DTYPE]](wp_sh[5 * NG + b]),
                    rebind[Scalar[DTYPE]](wp_sh[6 * NG + b]),
                    Int(rebind[Scalar[DTYPE]](gf_sh[1 * NG + a])),
                    Int(rebind[Scalar[DTYPE]](gf_sh[2 * NG + a])),
                    Int(rebind[Scalar[DTYPE]](gf_sh[3 * NG + a])),
                    nbody, ex_sig, n_sig, pr, ccd_tol, ccd_iter, multiccd_off,
                    geoms, bodies, mmeta, excludes, pairs, mesh_meta, mesh_verts,
                    mesh_polys, mesh_polyvert, mesh_polymap, mesh_vert_polymap,
                    mesh_vert_edgeadr, mesh_edges, hfield_meta, hfield_data,
                    stage, ccd_ws,
                )
                if cnt >= COLL_STAGE_MAXC:
                    full = 1
                cand_sh[5 * NC + c] = Scalar[DTYPE](cnt)
                comptime if COLL_TIMING:
                    var _c_dt = perf_counter_ns() - _c_t0
                    _lane_n += 1
                    if _c_dt > _lane_max:
                        _lane_max = _c_dt
                        _lane_max_key = Int(rebind[Scalar[DTYPE]](cand_sh[4 * NC + c]))
            comptime if COLL_TIMING:
                stage[env, COLL_REPORT_BASE + 8 + 4 * tid + 0] = Scalar[DTYPE](Int(perf_counter_ns() - _lane_t0))
                stage[env, COLL_REPORT_BASE + 8 + 4 * tid + 1] = Scalar[DTYPE](_lane_n)
                stage[env, COLL_REPORT_BASE + 8 + 4 * tid + 2] = Scalar[DTYPE](Int(_lane_max))
                stage[env, COLL_REPORT_BASE + 8 + 4 * tid + 3] = Scalar[DTYPE](_lane_max_key)
            if full == 1:
                ctrl_sh[1] = Scalar[DTYPE](1)
        barrier()
        comptime if COLL_TIMING:
            _tt3 = perf_counter_ns()
        comptime if COLL_STOP_AFTER == 3:
            if tid == 0:
                smeta[env, META_IDX_NUM_CONTACTS] = Scalar[DTYPE](0)
            return

        # ── phase 3: offsets (thread 0), the sort as ranks, the copy, ncon ────
        _sap_block_output[DTYPE, NC](
            env, tid, ncand, max_contacts, nbody, cand_sh, sk_sh, ctrl_sh,
            stage, contacts, smeta,
        )
        comptime if COLL_TIMING:
            if tid == 0:
                var _tt4 = perf_counter_ns()
                stage[env, COLL_REPORT_BASE + 0] = Scalar[DTYPE](Int(_tt1 - _tt0))
                stage[env, COLL_REPORT_BASE + 1] = Scalar[DTYPE](Int(_tt2 - _tt1))
                stage[env, COLL_REPORT_BASE + 2] = Scalar[DTYPE](Int(_tt3 - _tt2))
                stage[env, COLL_REPORT_BASE + 3] = Scalar[DTYPE](Int(_tt4 - _tt3))
                stage[env, COLL_REPORT_BASE + 4] = Scalar[DTYPE](Int(_tt4 - _tt0))

        # ── `COLL_CAND_REPORT`: the candidate list, into the dead staging tail ──
        # After the compaction above nothing reads `stage` until the next
        # launch's phase 2 writes it again, so `contacts` / `ncon` are unchanged.
        # The kind key is RECOMPUTED from the candidate's geom types, because
        # phase 3 reused the key slot for the destination offset; `ord_sh`
        # (the phase-2 order) is untouched by phase 3.
        comptime if COLL_CAND_REPORT:
            comptime assert COLL_BLOCK_KERNEL, (
                "COLL_CAND_REPORT reports the BLOCK kernel's candidate list"
            )
            if tid == 0:
                comptime RB = COLL_REPORT_BASE
                stage[env, RB + 0] = Scalar[DTYPE](ncand)
                stage[env, RB + 1] = Scalar[DTYPE](overflow)
                stage[env, RB + 2] = Scalar[DTYPE](
                    Int(rebind[Scalar[DTYPE]](ctrl_sh[1]))
                )
                stage[env, RB + 3] = smeta[env, META_IDX_NUM_CONTACTS]
                stage[env, RB + 4] = Scalar[DTYPE](rep_sap_n)
                stage[env, RB + 5] = Scalar[DTYPE](rep_tests)
                stage[env, RB + 6] = Scalar[DTYPE](rep_shifts)
                stage[env, RB + 7] = Scalar[DTYPE](COLL_TPB)
                stage[env, RB + 8] = Scalar[DTYPE](rep_aabb_all)
                stage[env, RB + 9] = Scalar[DTYPE](rep_survive)
                stage[env, RB + 10] = Scalar[DTYPE](rep_planes)
                for p in range(COLL_NCAND_CAP):
                    var key = -1
                    if p < ncand:
                        var c = Int(rebind[Scalar[DTYPE]](ord_sh[p]))
                        var a = Int(rebind[Scalar[DTYPE]](cand_sh[0 * NC + c]))
                        var b = Int(rebind[Scalar[DTYPE]](cand_sh[1 * NC + c]))
                        var t = Int(rebind[Scalar[DTYPE]](cand_sh[2 * NC + c]))
                        var rb_ = mj_geom_type_rank(
                            Int(rebind[Scalar[DTYPE]](gf_sh[0 * NG + b]))
                        )
                        if t < 0:
                            key = _KIND_PLANE_BASE + rb_
                        else:
                            var ra = mj_geom_type_rank(
                                Int(rebind[Scalar[DTYPE]](gf_sh[0 * NG + a]))
                            )
                            key = ra * 8 + rb_ if ra <= rb_ else rb_ * 8 + ra
                    stage[env, RB + COLL_REPORT_HDR + p] = Scalar[DTYPE](key)
        else:
            _ = rep_tests
            _ = rep_shifts
            _ = rep_sap_n
            _ = rep_aabb_all
            _ = rep_survive
            _ = rep_planes


@always_inline
def _flat_bucket[DTYPE: DType](cost: Scalar[DTYPE]) -> Int:
    """A candidate's hot bucket from its pair's last narrow-phase time (ns):
    -1 = cold, else 0..3 for [1, 2), [2, 4), [4, 8), [8, inf) x
    `COLL_FLAT_HOT_NS`. Written as comparisons: NaN or garbage is cold."""
    comptime H = Float64(COLL_FLAT_HOT_NS)
    if not (cost >= Scalar[DTYPE](H)):
        return -1
    if cost >= Scalar[DTYPE](8 * H):
        return 3
    if cost >= Scalar[DTYPE](4 * H):
        return 2
    if cost >= Scalar[DTYPE](2 * H):
        return 1
    return 0


@always_inline
def _flat_cost_slot(a: Int, b: Int) -> Int:
    """The cost slot of a listed candidate `(a, b)` — the warm-slot hash."""
    return (a * 131 + b) % HILL_WARM_SLOTS


@always_inline
def _flat_counter(
    ptr: Pointer[Scalar[DType.float32], MutAnyOrigin], k: Int
) -> Pointer[Scalar[DType.int32], MutAnyOrigin]:
    """Queue counter `k` (0..3 hot buckets, 4 cold) of `coll_flat`, whose
    global block starts at `ptr`, as an int32 cell for `std.atomic`."""
    return ptr.unsafe_offset(k).unsafe_bitcast[Scalar[DType.int32]]()


@always_inline
def _sap_flat_list_env[
    DTYPE: DType,
    NC: Int,
    BATCH: Int,
](
    env: Int,
    tid: Int,
    ncand: Int,
    overflow: Int,
    cand_sh: LayoutTensor[
        DTYPE, Layout.row_major(6 * NC), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ],
    ord_sh: LayoutTensor[
        DTYPE, Layout.row_major(NC), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ],
    coll_flat: LayoutTensor[
        DTYPE, Layout.row_major(coll_flat_words(BATCH)), MutAnyOrigin
    ],
):
    """`FLAT_LIST`'s ending (EVERY thread — it barriers): the env's listed
    candidates into its `coll_flat` row (`a, b, t`, the counts), each one
    classified by its pair's cost in parallel, then thread 0 reserves a range
    of each bucket's global queue with ONE atomic per bucket and appends the
    env's tasks in the phase-2 kind order (`ord_sh`). An env past the cap
    lists no task; the output kernel sends it to the serial fallback."""
    comptime assert DTYPE == DType.float32, (
        "the queue counters are int32 cells in a float32 `coll_flat`"
    )
    comptime G = BATCH * CF_ROW
    comptime QS = BATCH * NC
    var base = env * CF_ROW
    for c in range(tid, ncand, COLL_TPB):
        var a = Int(rebind[Scalar[DTYPE]](cand_sh[0 * NC + c]))
        var b = Int(rebind[Scalar[DTYPE]](cand_sh[1 * NC + c]))
        coll_flat[base + CF_CAND + c] = cand_sh[0 * NC + c]
        coll_flat[base + CF_CAND + NC + c] = cand_sh[1 * NC + c]
        coll_flat[base + CF_CAND + 2 * NC + c] = cand_sh[2 * NC + c]
        # the bucket, parked in the (still unused) record-count slot
        cand_sh[5 * NC + c] = Scalar[DTYPE](
            _flat_bucket[DTYPE](
                rebind[Scalar[DTYPE]](
                    coll_flat[base + CF_COST + _flat_cost_slot(a, b)]
                )
            )
        )
    barrier()
    if tid != 0:
        return
    coll_flat[base + CF_NCAND] = Scalar[DTYPE](ncand)
    coll_flat[base + CF_OVERFLOW] = Scalar[DTYPE](overflow)
    coll_flat[base + CF_FULL] = Scalar[DTYPE](0)
    # Scalars, not a runtime-indexed array: see
    # `feedback_metal_wide_per_thread_inlinearray_miscompute`.
    var n0 = 0
    var n1 = 0
    var n2 = 0
    var n3 = 0
    var nc = 0
    if overflow == 0:
        for p in range(ncand):
            var bk = Int(rebind[Scalar[DTYPE]](cand_sh[5 * NC + Int(rebind[Scalar[DTYPE]](ord_sh[p]))]))
            if bk == 3:
                n3 += 1
            elif bk == 2:
                n2 += 1
            elif bk == 1:
                n1 += 1
            elif bk == 0:
                n0 += 1
            else:
                nc += 1
    coll_flat[base + CF_NHOT + 0] = Scalar[DTYPE](n0)
    coll_flat[base + CF_NHOT + 1] = Scalar[DTYPE](n1)
    coll_flat[base + CF_NHOT + 2] = Scalar[DTYPE](n2)
    coll_flat[base + CF_NHOT + 3] = Scalar[DTYPE](n3)
    coll_flat[base + CF_NCOLD] = Scalar[DTYPE](nc)
    if overflow != 0 or ncand == 0:
        return
    var gp = rebind[Pointer[Scalar[DType.float32], MutAnyOrigin]](
        coll_flat.ptr.unsafe_offset(G)
    )
    # each list's first slot in its queue, then its queue's offset
    var s0 = 0 * QS + Int(Atomic[Int32].fetch_add(_flat_counter(gp, 0), Int32(n0)))
    var s1 = 1 * QS + Int(Atomic[Int32].fetch_add(_flat_counter(gp, 1), Int32(n1)))
    var s2 = 2 * QS + Int(Atomic[Int32].fetch_add(_flat_counter(gp, 2), Int32(n2)))
    var s3 = 3 * QS + Int(Atomic[Int32].fetch_add(_flat_counter(gp, 3), Int32(n3)))
    var sc = 4 * QS + Int(Atomic[Int32].fetch_add(_flat_counter(gp, 4), Int32(nc)))
    comptime Q0 = G + CF_G_HDR
    for p in range(ncand):
        var c = Int(rebind[Scalar[DTYPE]](ord_sh[p]))
        var bk = Int(rebind[Scalar[DTYPE]](cand_sh[5 * NC + c]))
        var pos: Int
        if bk == 3:
            pos = s3
            s3 += 1
        elif bk == 2:
            pos = s2
            s2 += 1
        elif bk == 1:
            pos = s1
            s1 += 1
        elif bk == 0:
            pos = s0
            s0 += 1
        else:
            pos = sc
            sc += 1
        coll_flat[Q0 + pos] = Scalar[DTYPE](env * NC + c)


def _sap_narrow_flat_kernel[
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
    stage: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, COLL_STAGE_SLOTS * CONTACT_SIZE),
        MutAnyOrigin,
    ],
    # `Data.coll_flat` — the queues, lists and costs
    coll_flat: LayoutTensor[
        DTYPE, Layout.row_major(coll_flat_words(BATCH)), MutAnyOrigin
    ],
):
    """The flat narrow phase: one WARP (block) per env slot, over the queues
    the listing appended to. Warp `w` runs hot tasks `w, w + W, ...` on its
    lane 0 — counted over the buckets in descending cost, so this is a round
    robin over a longest-first order — then cold chunks of 32, one per lane,
    from the far end of the warps. Each task is one listed candidate
    through `_sap_block_candidate` into its staging window, exactly as the
    block kernel's phase 2 runs it; the CCD row is the thread's own
    (`w * COLL_TPB + lane`, the same rows the block kernel uses) and the hill
    climb's warm slots stay on the ENV's first row. The record count goes to
    the list, and on NVIDIA the task's time to its pair's cost slot, which
    schedules the next step."""
    comptime assert COLL_CCD_LANES == COLL_TPB, (
        "the flat narrow phase gives every thread a CCD row"
    )
    var w = Int(block_idx.x)
    var lane = Int(thread_idx.x)
    comptime W = BATCH
    comptime NC = COLL_NCAND_CAP
    comptime G = BATCH * CF_ROW
    comptime QS = BATCH * NC
    comptime Q0 = G + CF_G_HDR
    comptime EX_CAP = cap[NEXCLUDE]() if may_exist[NEXCLUDE]() else 1
    var dims = Dims[nq=NQ, nv=NV, nbody=NBODY, njoint=NJOINT, max_contacts=MAX_CONTACTS, ngeom=NGEOM, nexclude=NEXCLUDE, nmesh_verts=NMESH_VERTS, npair=NPAIR]()
    var nbody = NBODY
    var ex_sh = LayoutTensor[
        DTYPE, Layout.row_major(EX_CAP), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var nsig_sh = LayoutTensor[
        DTYPE, Layout.row_major(1), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    # The model's `<exclude>` signatures, as the block kernel's phase 1a.
    if lane == 0:
        var ex0 = Scratch[Int, EX_CAP](
            NEXCLUDE if NEXCLUDE > 0 else 1, fill=0
        )
        var n_sig0 = exclude_signatures[DTYPE, EX_CAP](
            nbody, NEXCLUDE, mmeta, excludes, ex0
        )
        for k in range(EX_CAP):
            ex_sh[k] = Scalar[DTYPE](ex0[k])
        nsig_sh[0] = Scalar[DTYPE](n_sig0)
    barrier()
    var ex_sig = Scratch[Int, EX_CAP](
        NEXCLUDE if NEXCLUDE > 0 else 1, fill=0
    )
    for k in range(EX_CAP):
        ex_sig[k] = Int(rebind[Scalar[DTYPE]](ex_sh[k]))
    var n_sig = Int(rebind[Scalar[DTYPE]](nsig_sh[0]))

    var ccd_tol = rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_CCD_TOLERANCE])
    if ccd_tol <= 0:
        ccd_tol = Scalar[DTYPE](MJ_CCD_TOLERANCE)
    var ccd_iter = Int(
        rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_CCD_ITERATIONS])
    )
    if ccd_iter < 1:
        ccd_iter = MJ_CCD_ITERATIONS
    var multiccd_off = (
        rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_MULTICCD_DISABLED]) != 0
    )
    var pr = _SapProbe()
    var wrow = w * COLL_TPB + lane

    comptime assert DTYPE == DType.float32, (
        "the queue counters are int32 cells in a float32 `coll_flat`"
    )
    var gp = rebind[Pointer[Scalar[DType.float32], MutAnyOrigin]](
        coll_flat.ptr.unsafe_offset(G)
    )
    var c0 = Int(_flat_counter(gp, 0)[])
    var c1 = Int(_flat_counter(gp, 1)[])
    var c2 = Int(_flat_counter(gp, 2)[])
    var c3 = Int(_flat_counter(gp, 3)[])
    var nh = c0 + c1 + c2 + c3
    var ncold = Int(_flat_counter(gp, 4)[])
    # This warp's tasks, hot then cold, as ONE loop with the task body once
    # (a capturing closure here crashed the compiler, 2026-09-27): hot tasks
    # `w, w + W, ...` on lane 0, then cold chunks `W-1-w, 2W-1-w, ...` of
    # `COLL_TPB`, a lane each.
    var n_hot = (nh - w + W - 1) // W if w < nh else 0
    var nchunk = (ncold + COLL_TPB - 1) // COLL_TPB
    var k0 = W - 1 - w
    var n_cold = (nchunk - k0 + W - 1) // W if k0 < nchunk else 0
    for it in range(n_hot + n_cold):
        var item = -1
        if it < n_hot:
            if lane == 0:
                # hot index h over buckets 3, 2, 1, 0
                var h = w + it * W
                var q: Int
                if h < c3:
                    q = 3 * QS + h
                elif h < c3 + c2:
                    q = 2 * QS + h - c3
                elif h < c3 + c2 + c1:
                    q = 1 * QS + h - c3 - c2
                else:
                    q = h - c3 - c2 - c1
                item = Int(rebind[Scalar[DTYPE]](coll_flat[Q0 + q]))
        else:
            var i = (k0 + (it - n_hot) * W) * COLL_TPB + lane
            if i < ncold:
                item = Int(rebind[Scalar[DTYPE]](coll_flat[Q0 + 4 * QS + i]))
        if item >= 0:
            var env = item // NC
            var c = item - env * NC
            var base = env * CF_ROW
            var a = Int(rebind[Scalar[DTYPE]](coll_flat[base + CF_CAND + c]))
            var b = Int(rebind[Scalar[DTYPE]](coll_flat[base + CF_CAND + NC + c]))
            var t = Int(rebind[Scalar[DTYPE]](coll_flat[base + CF_CAND + 2 * NC + c]))
            # the block kernel's phase-0 poses, recomputed by the same function
            var a_px: Scalar[DTYPE] = 0
            var a_py: Scalar[DTYPE] = 0
            var a_pz: Scalar[DTYPE] = 0
            var a_qx: Scalar[DTYPE] = 0
            var a_qy: Scalar[DTYPE] = 0
            var a_qz: Scalar[DTYPE] = 0
            var a_qw: Scalar[DTYPE] = 1
            _geom_world_pos[DTYPE](
                env, a, geoms, xpos, xquat, a_px, a_py, a_pz, a_qx, a_qy, a_qz, a_qw
            )
            var b_px: Scalar[DTYPE] = 0
            var b_py: Scalar[DTYPE] = 0
            var b_pz: Scalar[DTYPE] = 0
            var b_qx: Scalar[DTYPE] = 0
            var b_qy: Scalar[DTYPE] = 0
            var b_qz: Scalar[DTYPE] = 0
            var b_qw: Scalar[DTYPE] = 1
            _geom_world_pos[DTYPE](
                env, b, geoms, xpos, xquat, b_px, b_py, b_pz, b_qx, b_qy, b_qz, b_qw
            )
            var t0: Int = 0
            comptime if is_nvidia_gpu():
                t0 = perf_counter_ns()
            var cnt = _sap_block_candidate[DTYPE, BATCH, EX_CAP=EX_CAP](
                env, wrow, env * COLL_CCD_LANES, dims, a, b, t,
                c * COLL_STAGE_MAXC,
                a_px, a_py, a_pz, a_qx, a_qy, a_qz, a_qw,
                b_px, b_py, b_pz, b_qx, b_qy, b_qz, b_qw,
                Int(rebind[Scalar[DTYPE]](geoms[a, GEOM_IDX_BODY])),
                Int(rebind[Scalar[DTYPE]](geoms[a, GEOM_IDX_CONTYPE])),
                Int(rebind[Scalar[DTYPE]](geoms[a, GEOM_IDX_CONAFFINITY])),
                nbody, ex_sig, n_sig, pr, ccd_tol, ccd_iter, multiccd_off,
                geoms, bodies, mmeta, excludes, pairs, mesh_meta, mesh_verts,
                mesh_polys, mesh_polyvert, mesh_polymap, mesh_vert_polymap,
                mesh_vert_edgeadr, mesh_edges, hfield_meta, hfield_data,
                stage, ccd_ws,
            )
            coll_flat[base + CF_CNT + c] = Scalar[DTYPE](cnt)
            if cnt >= COLL_STAGE_MAXC:
                coll_flat[base + CF_FULL] = Scalar[DTYPE](1)
            # ⚠ ONLY A HOT TASK'S CLOCK IS ITS OWN. A cold task shares its warp
            # with 31 others, and while their branches diverge every lane's
            # timer spans them all (the same trap as `COLL_TIMING`'s per-lane
            # numbers): timed there, nearly every mesh pair read >= 16 us and
            # the next step ran 21k "hot" tasks for ~2.5k slow ones. So a
            # cold task is costed by what makes a pair slow — a MESH pair
            # that emitted a contact penetrated, so it ran EPA — at the
            # threshold (bucket 0), and its first hot run measures the rest.
            # A penetrating mesh pair stays at least at the threshold, hot or
            # cold — else one that runs under it alone would flip every step.
            comptime if is_nvidia_gpu():
                var cost = Scalar[DTYPE](0)
                if it < n_hot:
                    cost = Scalar[DTYPE](Int(perf_counter_ns() - t0))
                if cnt > 0 and t >= 0 and (
                    t == GEOM_MESH
                    or Int(rebind[Scalar[DTYPE]](geoms[b, GEOM_IDX_TYPE])) == GEOM_MESH
                ):
                    if cost < Scalar[DTYPE](COLL_FLAT_HOT_NS):
                        cost = Scalar[DTYPE](COLL_FLAT_HOT_NS)
                coll_flat[base + CF_COST + _flat_cost_slot(a, b)] = cost


def _sap_flat_output_kernel[
    DTYPE: DType,
    MAX_CONTACTS: Int,
    NBODY: Int,
    BATCH: Int,
](
    contacts: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, MAX_CONTACTS * CONTACT_SIZE),
        MutAnyOrigin,
    ],
    smeta: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, METADATA_SIZE), MutAnyOrigin
    ],
    stage: LayoutTensor[
        DTYPE, Layout.row_major(BATCH, COLL_STAGE_SLOTS * CONTACT_SIZE),
        MutAnyOrigin,
    ],
    coll_flat: LayoutTensor[
        DTYPE, Layout.row_major(coll_flat_words(BATCH)), MutAnyOrigin
    ],
):
    """One block per env: the flat narrow phase's staging windows through
    the block kernel's phase 3 (`_sap_block_output`). A list past the cap or
    a filled window marks the env for the serial fallback, as there."""
    comptime NC = COLL_NCAND_CAP
    var env = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    if env >= BATCH:
        return
    var cand_sh = LayoutTensor[
        DTYPE, Layout.row_major(6 * NC), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var sk_sh = LayoutTensor[
        DTYPE, Layout.row_major(NC), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var ctrl_sh = LayoutTensor[
        DTYPE, Layout.row_major(8), MutAnyOrigin,
        address_space=AddressSpace.SHARED,
    ].stack_allocation()
    var base = env * CF_ROW
    if env == 0 and tid == 0:
        # The queues are consumed (the narrow kernel has run): zero their
        # counters for the next listing.
        comptime assert DTYPE == DType.float32, (
            "the queue counters are int32 cells in a float32 `coll_flat`"
        )
        var gp = rebind[Pointer[Scalar[DType.float32], MutAnyOrigin]](
            coll_flat.ptr.unsafe_offset(BATCH * CF_ROW)
        )
        for k in range(5):
            _flat_counter(gp, k)[] = Int32(0)
    if tid == 0:
        ctrl_sh[0] = coll_flat[base + CF_NCAND]
        var fb = (
            rebind[Scalar[DTYPE]](coll_flat[base + CF_OVERFLOW]) != 0
            or rebind[Scalar[DTYPE]](coll_flat[base + CF_FULL]) != 0
        )
        ctrl_sh[1] = Scalar[DTYPE](1) if fb else Scalar[DTYPE](0)
    barrier()
    var ncand = Int(rebind[Scalar[DTYPE]](ctrl_sh[0]))
    for c in range(tid, ncand, COLL_TPB):
        cand_sh[3 * NC + c] = Scalar[DTYPE](c * COLL_STAGE_MAXC)
        cand_sh[5 * NC + c] = coll_flat[base + CF_CNT + c]
    barrier()
    _sap_block_output[DTYPE, NC](
        env, tid, ncand, MAX_CONTACTS, NBODY, cand_sh, sk_sh, ctrl_sh,
        stage, contacts, smeta,
    )


def detect_contacts_sap[
    target: StaticString,
    DTYPE: DType,
    D: DimsLike,
    BATCH: Int = 1,
    # The GPU narrow phase's layout — see `ccd_workspace.COLL_FLAT_NARROW`.
    # A parameter so a gate can run both on one `Data`.
    FLAT: Bool = COLL_FLAT_NARROW,
](
    mut d: Data[DTYPE, D, BATCH],
    mut m: Model[DTYPE, D],
    ctx: Optional[DeviceContext] = None,
) raises:
    """AABB/SAP broadphase geom contact detection from FK products, both
    targets, one body. Reads `d.xpos`/`d.xquat` + geom/body/meta/exclude/mesh
    records; writes `d.contacts` + the ncon slot of `d.meta`."""
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
    comptime L_MESH_POLYVERT = Layout.row_major(mesh_max_polyvert(D.NMESH_VERTS))
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
    comptime L_COLL_STAGE = Layout.row_major(BATCH, COLL_STAGE_SLOTS * CONTACT_SIZE)
    comptime L_COLL_FLAT = Layout.row_major(coll_flat_words(BATCH))

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
        var mesh_vert_polymap_v = m.mesh_vert_polymap.lt_dyn["cpu", DYN2](rl_MESH_VPMAP)
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
            _detect_contacts_sap_env[DTYPE, BATCH](
                e, dm, xpos_v, xquat_v, geoms_v, bodies_v, mmeta_v,
                excludes_v, pairs_v, mesh_meta_v, mesh_verts_v, mesh_polys_v,
                mesh_polyvert_v, mesh_polymap_v, mesh_vert_polymap_v,
                mesh_vert_edgeadr_v, mesh_edges_v,
                hfield_meta_v, hfield_data_v,
                contacts_v, smeta_v, ccd_ws_v,
            )
    else:
        var c = ctx.value()
        comptime BLOCKS = (BATCH + SAP_TPB - 1) // SAP_TPB
        comptime USE_BLOCK = COLL_BLOCK_KERNEL and D.NHFIELD_DATA == 0
        comptime if USE_BLOCK and FLAT:
            # The flat narrow phase: list (and queue), narrow, output — see
            # `ccd_workspace.COLL_FLAT_NARROW`.
            c.enqueue_function[
                _detect_contacts_sap_block_kernel[
                    DTYPE, D.NQ, D.NV, D.NBODY, D.NJOINT, D.MAX_CONTACTS, D.NGEOM,
                    D.NEXCLUDE, D.NMESH_VERTS, BATCH, D.NPAIR,
                    _hf_len(D.NHFIELD_DATA),
                    FLAT_LIST=True,
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
                d.coll_stage.lt["gpu", L_COLL_STAGE](),
                d.coll_flat.lt["gpu", L_COLL_FLAT](),
                grid_dim=(BATCH,),
                block_dim=(COLL_TPB,),
            )
            c.enqueue_function[
                _sap_narrow_flat_kernel[
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
                d.coll_stage.lt["gpu", L_COLL_STAGE](),
                d.coll_flat.lt["gpu", L_COLL_FLAT](),
                grid_dim=(BATCH,),
                block_dim=(COLL_TPB,),
            )
            c.enqueue_function[
                _sap_flat_output_kernel[
                    DTYPE, D.MAX_CONTACTS, D.NBODY, BATCH
                ]
            ](
                d.contacts.lt["gpu", L_CONTACTS](),
                d.meta.lt["gpu", L_SMETA](),
                d.coll_stage.lt["gpu", L_COLL_STAGE](),
                d.coll_flat.lt["gpu", L_COLL_FLAT](),
                grid_dim=(BATCH,),
                block_dim=(COLL_TPB,),
            )
        elif USE_BLOCK:
            c.enqueue_function[
                _detect_contacts_sap_block_kernel[
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
                d.coll_stage.lt["gpu", L_COLL_STAGE](),
                d.coll_flat.lt["gpu", L_COLL_FLAT](),
                grid_dim=(BATCH,),
                block_dim=(COLL_TPB,),
            )
        # The serial kernel: every env when the block kernel is off, only the
        # envs it marked otherwise (see the mark in phase 3).
        comptime if USE_BLOCK and COLL_NO_FALLBACK:
            return
        c.enqueue_function[
            _detect_contacts_sap_fields_kernel[
                DTYPE, D.NQ, D.NV, D.NBODY, D.NJOINT, D.MAX_CONTACTS, D.NGEOM,
                D.NEXCLUDE, D.NMESH_VERTS, BATCH, D.NPAIR,
                _hf_len(D.NHFIELD_DATA), ONLY_FLAGGED=USE_BLOCK,
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
            block_dim=(SAP_TPB,),
        )


def detect_contacts_auto[
    target: StaticString,
    DTYPE: DType,
    D: DimsLike,
    BATCH: Int = 1,
](
    mut d: Data[DTYPE, D, BATCH],
    mut m: Model[DTYPE, D],
    ctx: Optional[DeviceContext] = None,
) raises:
    """Contact detection with automatic broadphase selection (fields).

    Uses detect_contacts_sap when NGEOM >= SAP_THRESHOLD (default
    16), otherwise falls back to detect_contacts. The branch is
    resolved at compile time. NOTE: SAP contact emission ORDER differs from
    the O(N^2) path — do not swap this into a bit-exact-gated pipeline
    without re-baselining."""

    # ⚠⚠ THE SELECTION IS COMPTIME ON A STATIC PROVIDER AND RUNTIME ON A
    # DYNAMIC ONE, and the split is deliberate. Unlike the `CAP_*` gates this
    # is not a capacity test — it SELECTS AN ALGORITHM, and both branches
    # compute the same contacts. A blanket runtime `if` would therefore
    # compile both bodies into every binary for no behavioural gain, which is
    # why it was left alone in 3c-b.
    #
    # ⚠⚠ WHAT 3c-b GOT WRONG WAS CALLING THE CONSEQUENCE BENIGN. `D.NGEOM` is
    # `DIM_POISON`, so `-1 >= 16` was false and a runtime-loaded model ALWAYS
    # took the O(N^2) path — and the two paths DO NOT AGREE TO THE BIT. Their
    # contact ORDER differs (the docstring above says so), and SAP's record
    # conventions differ too: BODY_B = -1, `dist - margin` in DIST, no
    # INCLUDEMARGIN slot. So for any model at or above the threshold the two
    # legs were solving DIFFERENT contact sets in a different order, which is
    # most of what `test_runtime_step_both_legs` was measuring as "the caps
    # disable constraint families" on the humanoid (ngeom 18 >= 16). It is a
    # correctness split, not a performance note.
    #
    # ⚠ THE COMPTIME LEG'S INSTANTIATION SET IS UNCHANGED. Only the dynamic
    # provider — the one whose `NGEOM` is poison, i.e. exactly the one that
    # could not answer at compile time — pays for both bodies. The studio is
    # also where it matters: a composed scene is precisely the NGEOM regime
    # SAP exists for, and SO-ARM100 alone is 33 geoms.
    comptime if D.NGEOM == DIM_POISON:
        if d.dims.get_ngeom() >= SAP_THRESHOLD:
            detect_contacts_sap[target, DTYPE, BATCH=BATCH](d, m, ctx)
        else:
            detect_contacts[target, DTYPE, BATCH=BATCH](d, m, ctx)
    elif D.NGEOM >= SAP_THRESHOLD:
        detect_contacts_sap[target, DTYPE, BATCH=BATCH](d, m, ctx)
    else:
        detect_contacts[target, DTYPE, BATCH=BATCH](d, m, ctx)
