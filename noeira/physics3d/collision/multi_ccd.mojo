"""Multi-point convex contact — MuJoCo's `mjc_MultiCCD` (defect 21).

A convex-convex narrow phase that returns ONE point cannot hold a flat contact:
the body rotates about that point and sinks. MuJoCo's answer is to find the
first contact, then re-run the SAME query four more times with the two geoms
TILTED slightly about the axes perpendicular to the contact normal, keeping
whichever new points are distinct. A cylinder resting on a box goes from 1 row
to 5.

Ported from `references/mujoco-3.11.0/src/engine/engine_collision_convex.c`
:881-935 (`mjc_Convex`'s tail). 3.11.0 is the closest tree to the 3.10.0
runtime; the older trees differ on this in a way that matters — see the flag
note below.

    relative_tolerance = 1e-3;  perturbation_angle = 1e-3;
    frame = makeFrame(con[0].normal);
    tolerance = relative_tolerance * min(geom_rbound[g1], geom_rbound[g2]);
    axes = {frame+3, frame+6};  angles = {-1e-3, +1e-3};
    for axis, angle:
        rot = axisAngle2Mat(axis, angle);
        rotate g1 about con[0].pos by rot;
        rotate g2 about con[0].pos by rot^T;      // INVERSELY
        n = penetration(g1, g2, 1 contact);
        if n and isDistinctContact(con, ncon+1, tolerance):
            con[ncon].dist = con[0].dist;         // COPIED, not measured
            ncon += 1;
        restore g1, g2;

⚠ `con[ncon].dist = con[0].dist` is not an optimisation. The perturbed query
measures a penetration at a TILTED pose, which is not the penetration at the
real pose; MuJoCo overwrites it with the first point's. Keeping the measured
value makes every extra row slightly wrong and the manifold asymmetric. It is
also directly observable: all five rows of a cylinder/capsule contact report
dist exactly -0.005000 at a 5 mm fixture.

⚠⚠ THE ENABLE FLAG INVERTED BETWEEN VERSIONS, and reading the wrong tree gets
this backwards. MuJoCo 3.6.0 has `mjENBL_MULTICCD` (1<<4) — OPT-IN, off by
default. The 3.10.0 runtime and the 3.11.0 tree have `mjDSBL_MULTICCD` (1<<19)
— a DISABLE bit, so the feature is ON by default. We implement the default-on
behaviour, unconditionally, because that is what the runtime we gate against
does. Measured: with `opt.disableflags = 0` the reference emits 31 contacts on
`tests/physics3d/narrow_phase_pairs_ref.py`; with `mjDSBL_MULTICCD` it emits 15,
which is exactly what this engine emitted before this file existed.

⚠ SPHERE AND ELLIPSOID ARE EXCLUDED by MuJoCo's own guard, and that is not an
oversight to "fix": a sphere or ellipsoid touches a convex body at ONE point,
so a tilted requery finds the same point again and the manifold is a single
row by geometry. Dog's 44 collidable ellipsoids are unaffected, as are fish,
swimmer and humanoid_CMU whose only convex geoms are ellipsoids. Measured:
mesh/sphere is 1 row in the reference too.

SCOPE: the pairs `mjc_Convex` sends through this loop, which is decided by a
SECOND guard sitting in front of it (`engine_collision_convex.c:875`):

    int max_contacts = maxContacts(m, &obj1, &obj2);
    int ncon = mjc_penetration(..., max_contacts, margin);
    if (!mjDISABLED(mjDSBL_NATIVECCD) && max_contacts > 1) return ncon;

`maxContacts` (`:843`) gives 8 for BOX x BOX, 4 when BOTH geoms are box-or-mesh,
and 1 for everything else; `mjDSBL_NATIVECCD` is a DISABLE bit that is off by
default. So on the 3.10.0 runtime:

  * BOX x BOX, BOX x MESH, MESH x MESH -> take the early return. Their manifold
    comes from `multicontact()` (`engine_collision_gjk.c:2111`), which clips the
    two contacting polygons and prunes to a max-area quad, and which needs mesh
    polygon topology (`mesh_polynum`) that our `mesh_meta` does not carry. NOT
    PORTED, deliberately, and NOT to be faked here.
  * everything else, INCLUDING MESH x CYLINDER and MESH x CAPSULE -> reaches
    this loop, because only one of the two geoms is box-or-mesh.

⚠⚠ THAT SPLIT WAS GOT WRONG TWICE IN THIS FILE'S HISTORY, IN BOTH DIRECTIONS.
First a design note claimed MuJoCo's only exclusions were sphere and ellipsoid,
which would have routed box/mesh here; then the correction to it over-swung and
claimed mesh pairs never reach this loop at all, which would have left
mesh/cylinder at a single point forever. Measured against the runtime at ~5 mm
of overlap — MuJoCo's row count vs ours:

    block.stl / cylinder   5 vs 1      block.stl / box    4 vs 1  (native)
    block.stl / capsule    5 vs 1      block.stl / mesh   4 vs 1  (native)
    gripper   / cylinder   4 vs 1      gripper   / box    2 vs 1  (native)
    gripper   / capsule    2 vs 1      gripper   / mesh   1 vs 1
    either    / sphere     1 vs 1

⚠ THE OBVIOUS DISCRIMINATOR IS CONFOUNDED AND GAVE THE WRONG ANSWER FIRST.
Toggling `mjDSBL_MULTICCD` collapses EVERY pair above to a single row, which
reads as "they all come from this loop" — but `maxContacts` itself branches on
that bit (`return mjDISABLED(mjDSBL_MULTICCD) ? 1 : 4`), so disabling it also
switches the native path off. The path is decided by geom TYPE, not by that
flag.
"""

from std.math import sqrt

from layout import TileTensor, TensorLayout

from ..constants import (
    GEOM_SPHERE,
    GEOM_CAPSULE,
    GEOM_ELLIPSOID,
    GEOM_CYLINDER,
    GEOM_BOX,
    GEOM_MESH,
)
from ..kinematics.quat_math import (
    gpu_quat_mul,
    gpu_quat_rotate,
    gpu_axis_angle_to_quat,
)
from ..gpu.constants import mesh_max_edge
from ..gpu.constants import (
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
)
from .collision_primitives import (
    cylinder_capsule,
    cylinder_cylinder,
)
from .gjk import gjk_epa
from ..gpu.constants import MJ_CCD_TOLERANCE, MJ_CCD_ITERATIONS

# `mjc_Convex`'s two constants, named as MuJoCo names them.
from ..fields import DimsLike
comptime MULTICCD_RELATIVE_TOLERANCE: Float64 = 1e-3
comptime MULTICCD_PERTURBATION_ANGLE: Float64 = 1e-3


@always_inline
def multi_ccd_pair_supported(
    gi_type: Int, gj_type: Int, margin_positive: Bool = False
) -> Bool:
    """Does this geom pair get a perturbed manifold?

    ⚠ `margin_positive` (AUD-33): `maxContacts` returns 1 — not 4 — for ANY
    pair when either object carries a margin (engine_collision_convex.c:
    856-859), so a {box, mesh} x {box, mesh} pair with `margin + gap > 0`
    skips the polygon clip and takes the perturbation loop: up to 5 points
    where this used to leave 1. BOX x BOX never reaches `mjc_Convex`.

    MuJoCo's guard is "`mjc_Convex` handled it, `maxContacts` came back as 1,
    AND neither geom is a sphere or an ellipsoid". `maxContacts` is the part
    that is easy to miss: it returns >1 for BOX x BOX and for {BOX, MESH}
    pairs, and those take an early return before the perturbation loop. So the
    set that reaches this loop in the reference IS the cylinder family, and
    this function is not narrower than MuJoCo — it matches it. Mesh pairs are
    excluded for the reference's own reason; see the module docstring.

    ⚠ Anything added here MUST also be answered by `_convex_pair_single`, or
    the perturbed query silently returns "no contact" and the manifold quietly
    stays a single point — a gap that looks exactly like success.
    """
    if gi_type == GEOM_SPHERE or gj_type == GEOM_SPHERE:
        return False
    if gi_type == GEOM_ELLIPSOID or gj_type == GEOM_ELLIPSOID:
        return False
    if margin_positive:
        var bm_i = gi_type == GEOM_BOX or gi_type == GEOM_MESH
        var bm_j = gj_type == GEOM_BOX or gj_type == GEOM_MESH
        if bm_i and bm_j and not (gi_type == GEOM_BOX and gj_type == GEOM_BOX):
            return True
    if gi_type == GEOM_CYLINDER and gj_type == GEOM_CAPSULE:
        return True
    if gi_type == GEOM_CAPSULE and gj_type == GEOM_CYLINDER:
        return True
    if gi_type == GEOM_CYLINDER and gj_type == GEOM_CYLINDER:
        return True
    if gi_type == GEOM_CYLINDER and gj_type == GEOM_BOX:
        return True
    if gi_type == GEOM_BOX and gj_type == GEOM_CYLINDER:
        return True
    # MESH x {CYLINDER, CAPSULE} — `maxContacts` returns 4 only when BOTH geoms
    # are box-or-mesh, so these come back 1 and DO reach the perturbation loop.
    # Measured against the 3.10.0 runtime at ~5 mm of overlap:
    #
    #   block.stl  / cylinder   MuJoCo 5   ours 1
    #   block.stl  / capsule    MuJoCo 5   ours 1
    #   gripper    / cylinder   MuJoCo 4   ours 1
    #   gripper    / capsule    MuJoCo 2   ours 1
    #
    # `gripper` is sawyer's eGripperBase, and obj-cylinder-into-gripper-mesh is
    # the pair `test_sap_fields` Part B collides — i.e. the pair Phase 7 grasps
    # with. MESH x {BOX, MESH} is NOT here: those return 4 from `maxContacts`
    # and take the early return into native polygon clipping instead.
    if gi_type == GEOM_MESH and (
        gj_type == GEOM_CYLINDER or gj_type == GEOM_CAPSULE
    ):
        return True
    if gj_type == GEOM_MESH and (
        gi_type == GEOM_CYLINDER or gi_type == GEOM_CAPSULE
    ):
        return True
    return False


@always_inline
def _convex_pair_single[
    DTYPE: DType,
    L_MESH_VERTS: TensorLayout,
    L_MESH_VERT_EDGEADR: TensorLayout,
    L_MESH_EDGES: TensorLayout,
    L_WS: TensorLayout](
    gi_type: Int,
    pi_x: Scalar[DTYPE], pi_y: Scalar[DTYPE], pi_z: Scalar[DTYPE],
    qi_x: Scalar[DTYPE], qi_y: Scalar[DTYPE], qi_z: Scalar[DTYPE],
    qi_w: Scalar[DTYPE],
    ri: Scalar[DTYPE], hli: Scalar[DTYPE],
    hxi: Scalar[DTYPE], hyi: Scalar[DTYPE], hzi: Scalar[DTYPE],
    mesh_verts: TileTensor[DTYPE, L_MESH_VERTS, MutAnyOrigin],
    mesh_vert_edgeadr: TileTensor[DTYPE, L_MESH_VERT_EDGEADR, MutAnyOrigin],
    mesh_edges: TileTensor[DTYPE, L_MESH_EDGES, MutAnyOrigin],
    va1: Int, mnv1: Int,
    gj_type: Int,
    pj_x: Scalar[DTYPE], pj_y: Scalar[DTYPE], pj_z: Scalar[DTYPE],
    qj_x: Scalar[DTYPE], qj_y: Scalar[DTYPE], qj_z: Scalar[DTYPE],
    qj_w: Scalar[DTYPE],
    rj: Scalar[DTYPE], hlj: Scalar[DTYPE],
    hxj: Scalar[DTYPE], hyj: Scalar[DTYPE], hzj: Scalar[DTYPE],
    va2: Int, mnv2: Int,
    ws: TileTensor[DTYPE, L_WS, MutAnyOrigin],
    wrow: Int,
    ccd_tol: Scalar[DTYPE] = Scalar[DTYPE](MJ_CCD_TOLERANCE),
    ccd_iter: Int = MJ_CCD_ITERATIONS,
    ccd_margin: Scalar[DTYPE] = Scalar[DTYPE](0),
) -> Array[Scalar[DTYPE], 7]:
    """One contact for a `multi_ccd_pair_supported` pair — `(dist, pos, normal)`.

    ⚠ THE NORMAL IS `gi -> gj`, matching the inline dispatch in
    `contact_detection.mojo` exactly, INCLUDING its negation when the primitive
    is invoked with swapped operands. This function must stay a mirror of those
    branches: it is re-running the same query at a tilted pose, so any
    divergence between the two would make the extra manifold points disagree
    with the point they are extending. `feedback_contact_direction_conventions`
    is the record of what that class of mistake costs.

    ⚠ THAT MIRROR WAS BROKEN ONCE, SILENTLY. When the dispatch moved CYLINDER x
    BOX from `cylinder_box` to `gjk_epa` (`d93b0a29`) this function kept calling
    the primitive, so the first manifold point came from the convex query and
    its four perturbed extensions came from the CAPSULE REDUCTION — which is
    wrong by exactly `-r`. The gate did not catch it because MuJoCo copies
    `con[0].dist` onto every extra row, so the depth error was overwritten and
    only the perturbed POSITIONS carried it. Passing within tolerance is not the
    same as consistent: when a branch here changes, this file changes with it.
    """
    var out = Array[Scalar[DTYPE], 7](uninitialized=True)
    for k in range(7):
        out[k] = Scalar[DTYPE](0)
    out[0] = Scalar[DTYPE](1e30)  # no contact

    if (
        (gi_type == GEOM_CYLINDER and gj_type == GEOM_BOX)
        or (gi_type == GEOM_BOX and gj_type == GEOM_CYLINDER)
        or (gi_type == GEOM_CYLINDER and gj_type == GEOM_CAPSULE)
        or (gi_type == GEOM_CAPSULE and gj_type == GEOM_CYLINDER)
        or (gi_type == GEOM_CYLINDER and gj_type == GEOM_CYLINDER)
    ):
        # ⚠⚠ THE THIRD COPY OF THE SAME TABLE, moved with the other two. This
        # file's own docstring says it must be: its four perturbed extensions
        # used to come from the capsule reduction, wrong by exactly -r, and no
        # gate could see it because MuJoCo copies `con[0].dist` onto every
        # extra row. CAPSULE x CYLINDER and CYLINDER x CYLINDER are
        # `mjc_Convex` in MuJoCo's table, exactly as CYLINDER x BOX is.
        # ONE branch for both orderings, exactly as the dispatch writes it:
        # `cylinder_box` needed two because the primitive is asymmetric in its
        # operands, but the convex query is symmetric and returns `gi -> gj`
        # either way.
        var r = gjk_epa[DTYPE](
            gi_type,
            pi_x, pi_y, pi_z, qi_x, qi_y, qi_z, qi_w,
            ri, hli, hxi, hyi, hzi,
            mesh_verts, mesh_vert_edgeadr, mesh_edges, va1, mnv1,
            gj_type,
            pj_x, pj_y, pj_z, qj_x, qj_y, qj_z, qj_w,
            rj, hlj, hxj, hyj, hzj,
            va2, mnv2,
            ws, wrow,
            ccd_tol, ccd_iter, ccd_margin,
        )
        out[0] = r[0]
        out[1] = r[1]
        out[2] = r[2]
        out[3] = r[3]
        out[4] = r[4]
        out[5] = r[5]
        out[6] = r[6]
    elif gi_type == GEOM_MESH or gj_type == GEOM_MESH:
        # Mirrors the dispatch's mesh branch, which sends ANY mesh pair to
        # `gjk_epa` with the geoms in place. Only MESH x {CYLINDER, CAPSULE}
        # reaches here — `multi_ccd_pair_supported` is the gate — but the
        # condition is written as the dispatch writes it so the two read the
        # same, and `gjk_epa` is symmetric and returns `gi -> gj` regardless of
        # ordering, so there is no negation and no second branch.
        var r = gjk_epa[DTYPE](
            gi_type,
            pi_x, pi_y, pi_z, qi_x, qi_y, qi_z, qi_w,
            ri, hli, hxi, hyi, hzi,
            mesh_verts, mesh_vert_edgeadr, mesh_edges, va1, mnv1,
            gj_type,
            pj_x, pj_y, pj_z, qj_x, qj_y, qj_z, qj_w,
            rj, hlj, hxj, hyj, hzj,
            va2, mnv2,
            ws, wrow,
            ccd_tol, ccd_iter, ccd_margin,
        )
        out[0] = r[0]
        out[1] = r[1]
        out[2] = r[2]
        out[3] = r[3]
        out[4] = r[4]
        out[5] = r[5]
        out[6] = r[6]

    return out^


@always_inline
def _make_frame_axes[
    DTYPE: DType
](
    nx: Scalar[DTYPE], ny: Scalar[DTYPE], nz: Scalar[DTYPE]
) -> Array[Scalar[DTYPE], 6]:
    """`mju_makeFrame`'s y and z axes, for a normal already used as x.

    Port of `engine_util_spatial.c:508`. The seed axis is (0,1,0) unless the
    normal is too close to it, in which case (0,0,1) — the same `|n_y| < 0.5`
    test, kept because a different seed rotates the perturbation axes and so
    changes WHICH extra points are found.

    The sign of `n` does not matter: the two axes span the plane perpendicular
    to it either way, and the perturbation angles are applied symmetrically as
    `{-a, +a}`. So callers may pass the record's stored normal or its negation.
    """
    var ax = nx
    var ay = ny
    var az = nz
    var n = sqrt(ax * ax + ay * ay + az * az)
    if n > Scalar[DTYPE](0):
        ax = ax / n
        ay = ay / n
        az = az / n

    # Seed y, exactly as MuJoCo chooses it.
    var yx = Scalar[DTYPE](0)
    var yy = Scalar[DTYPE](0)
    var yz = Scalar[DTYPE](0)
    if ay < Scalar[DTYPE](0.5) and ay > Scalar[DTYPE](-0.5):
        yy = Scalar[DTYPE](1)
    else:
        yz = Scalar[DTYPE](1)

    # Orthogonalise against x, then normalise.
    var d = ax * yx + ay * yy + az * yz
    yx = yx - ax * d
    yy = yy - ay * d
    yz = yz - az * d
    var yn = sqrt(yx * yx + yy * yy + yz * yz)
    if yn > Scalar[DTYPE](0):
        yx = yx / yn
        yy = yy / yn
        yz = yz / yn

    var out = Array[Scalar[DTYPE], 6](uninitialized=True)
    out[0] = yx
    out[1] = yy
    out[2] = yz
    # z = cross(x, y)
    out[3] = ay * yz - az * yy
    out[4] = az * yx - ax * yz
    out[5] = ax * yy - ay * yx
    return out^


@always_inline
def _rotate_pose_about[
    DTYPE: DType
](
    ox: Scalar[DTYPE], oy: Scalar[DTYPE], oz: Scalar[DTYPE],
    rqx: Scalar[DTYPE], rqy: Scalar[DTYPE], rqz: Scalar[DTYPE],
    rqw: Scalar[DTYPE],
    px: Scalar[DTYPE], py: Scalar[DTYPE], pz: Scalar[DTYPE],
    gqx: Scalar[DTYPE], gqy: Scalar[DTYPE], gqz: Scalar[DTYPE],
    gqw: Scalar[DTYPE],
) -> Array[Scalar[DTYPE], 7]:
    """`mju_rotateFrame` — rotate a geom pose about `o`, returning pos + quat.

        xmat = rot * xmat
        rel  = origin - xpos
        vec  = rot*rel - rel
        xpos = xpos - vec

    The orientation is LEFT-multiplied (a world-frame rotation), and the
    position correction is written as MuJoCo writes it rather than the
    algebraically equivalent `o + rot*(p - o)`; they differ in the last bits
    and this is re-running a query whose answer decides whether a contact
    counts as distinct.
    """
    var q = gpu_quat_mul[DTYPE](rqx, rqy, rqz, rqw, gqx, gqy, gqz, gqw)

    var relx = ox - px
    var rely = oy - py
    var relz = oz - pz
    var rr = gpu_quat_rotate[DTYPE](rqx, rqy, rqz, rqw, relx, rely, relz)
    var vecx = rr[0] - relx
    var vecy = rr[1] - rely
    var vecz = rr[2] - relz

    var out = Array[Scalar[DTYPE], 7](uninitialized=True)
    out[0] = px - vecx
    out[1] = py - vecy
    out[2] = pz - vecz
    out[3] = q[0]
    out[4] = q[1]
    out[5] = q[2]
    out[6] = q[3]
    return out^


@always_inline
def multi_ccd_extra_contacts[
    DTYPE: DType,
    D: DimsLike,
    L_MESH_VERTS: TensorLayout,
    L_MESH_VERT_EDGEADR: TensorLayout,
    L_MESH_EDGES: TensorLayout,
    L_CONTACTS: TensorLayout,
    L_WS: TensorLayout](
    env: Int,
    body_a: Int,
    body_b: Int,
    first_idx: Int,
    gi_type: Int,
    pi_x: Scalar[DTYPE], pi_y: Scalar[DTYPE], pi_z: Scalar[DTYPE],
    qi_x: Scalar[DTYPE], qi_y: Scalar[DTYPE], qi_z: Scalar[DTYPE],
    qi_w: Scalar[DTYPE],
    ri: Scalar[DTYPE], hli: Scalar[DTYPE],
    hxi: Scalar[DTYPE], hyi: Scalar[DTYPE], hzi: Scalar[DTYPE],
    rbound_i: Scalar[DTYPE],
    va1: Int, mnv1: Int,
    gj_type: Int,
    pj_x: Scalar[DTYPE], pj_y: Scalar[DTYPE], pj_z: Scalar[DTYPE],
    qj_x: Scalar[DTYPE], qj_y: Scalar[DTYPE], qj_z: Scalar[DTYPE],
    qj_w: Scalar[DTYPE],
    rj: Scalar[DTYPE], hlj: Scalar[DTYPE],
    hxj: Scalar[DTYPE], hyj: Scalar[DTYPE], hzj: Scalar[DTYPE],
    rbound_j: Scalar[DTYPE],
    va2: Int, mnv2: Int,
    dims: D,
    mesh_verts: TileTensor[DTYPE, L_MESH_VERTS, MutAnyOrigin],
    mesh_vert_edgeadr: TileTensor[DTYPE, L_MESH_VERT_EDGEADR, MutAnyOrigin],
    mesh_edges: TileTensor[DTYPE, L_MESH_EDGES, MutAnyOrigin],
    c0x: Scalar[DTYPE], c0y: Scalar[DTYPE], c0z: Scalar[DTYPE],
    n0x: Scalar[DTYPE], n0y: Scalar[DTYPE], n0z: Scalar[DTYPE],
    dist0: Scalar[DTYPE],
    contact_margin: Scalar[DTYPE],
    contact_friction: Scalar[DTYPE],
    contact_friction_spin: Scalar[DTYPE],
    contact_friction_roll: Scalar[DTYPE],
    contact_condim: Int,
    contacts: TileTensor[DTYPE, L_CONTACTS, MutAnyOrigin],
    mut num_contacts: Int,
    ws: TileTensor[DTYPE, L_WS, MutAnyOrigin],
    wrow: Int,
    ccd_tol: Scalar[DTYPE] = Scalar[DTYPE](MJ_CCD_TOLERANCE),
    ccd_iter: Int = MJ_CCD_ITERATIONS,
    ccd_margin: Scalar[DTYPE] = Scalar[DTYPE](0),
    # ⚠ THE GAP HALF OF THE PAIR'S MARGIN, DEFAULTED TO 0 SO EVERY EXISTING
    # CALL SITE IS UNCHANGED. `contact_margin` is the narrowphase CUTOFF
    # (`margin + gap`); what a contact STORES as its `includemargin` is
    # `contact_margin - contact_gap`, and the solver excludes
    # `dist >= includemargin`. See `GEOM_IDX_GAP`.
    contact_gap: Scalar[DTYPE] = Scalar[DTYPE](0),
    # The emission cap. -1 = the model's `max_contacts` (every serial
    # caller); the block-per-env collision kernel passes the end of the
    # thread's staging window instead, so a routine cannot write past it.
    max_contacts_in: Int = -1,
) -> Int:
    """Append the perturbed manifold points for one already-emitted contact.

    `first_idx` is the record index of that contact; `c0`/`n0`/`dist0` are its
    position, its `gi -> gj` normal and its distance. Returns how many extra
    records were written.

    ⚠ CALL THIS ONLY AFTER THE FIRST CONTACT IS EMITTED, and only when the pair
    passed `multi_ccd_pair_supported`. MuJoCo's guard is `ncon == 1`: with no
    contact there is nothing to perturb about, and with a manifold already in
    hand the extra points are redundant.
    """
    var max_contacts = (
        max_contacts_in if max_contacts_in >= 0 else dims.get_max_contacts()
    )
    var written = 0

    # `mjc_Convex`: tolerance scales with the SMALLER bounding radius, so a
    # small geom against a large one does not have its distinct points merged
    # by the large one's scale.
    var tol = Scalar[DTYPE](MULTICCD_RELATIVE_TOLERANCE) * (
        rbound_i if rbound_i < rbound_j else rbound_j
    )

    var axes = _make_frame_axes[DTYPE](n0x, n0y, n0z)
    var ang = Scalar[DTYPE](MULTICCD_PERTURBATION_ANGLE)

    for axis_id in range(2):
        var axx = axes[0] if axis_id == 0 else axes[3]
        var axy = axes[1] if axis_id == 0 else axes[4]
        var axz = axes[2] if axis_id == 0 else axes[5]

        for angle_id in range(2):
            if num_contacts >= max_contacts:
                return written
            var angle = -ang if angle_id == 0 else ang

            # rot, and its inverse for the second geom.
            #
            # ⚠ `gpu_axis_angle_to_quat` rather than a hand-rolled half-angle:
            # a bare `sin`/`cos` on `Scalar[DTYPE]` fails to compile at an
            # unconstrained trait `DTYPE` ("lacking evidence to prove
            # correctness"), and that helper already carries the
            # floating-point evidence AND the zero-axis branch whose epsilon
            # was itself a fixed bug. See
            # `feedback_where_clause_cannot_cross_trait_boundary`.
            var rq = gpu_axis_angle_to_quat[DTYPE](axx, axy, axz, angle)
            var rqx = rq[0]
            var rqy = rq[1]
            var rqz = rq[2]
            var rqw = rq[3]

            var pi = _rotate_pose_about[DTYPE](
                c0x, c0y, c0z, rqx, rqy, rqz, rqw,
                pi_x, pi_y, pi_z, qi_x, qi_y, qi_z, qi_w,
            )
            # ⚠ THE SECOND GEOM ROTATES BY THE INVERSE. Rotating both the same
            # way is a rigid motion of the pair and finds the same point again;
            # the point of the perturbation is to change their RELATIVE
            # orientation, which is what exposes a second contact on a flat
            # face. Quaternion inverse of a unit quat is its conjugate.
            var pj = _rotate_pose_about[DTYPE](
                c0x, c0y, c0z, -rqx, -rqy, -rqz, rqw,
                pj_x, pj_y, pj_z, qj_x, qj_y, qj_z, qj_w,
            )

            var r = _convex_pair_single[DTYPE](
                gi_type,
                pi[0], pi[1], pi[2], pi[3], pi[4], pi[5], pi[6],
                ri, hli, hxi, hyi, hzi,
                mesh_verts, mesh_vert_edgeadr, mesh_edges, va1, mnv1,
                gj_type,
                pj[0], pj[1], pj[2], pj[3], pj[4], pj[5], pj[6],
                rj, hlj, hxj, hyj, hzj,
                va2, mnv2,
                ws, wrow,
                ccd_tol, ccd_iter, ccd_margin,
            )
            if not (r[0] < contact_margin):
                continue

            # `mjc_isDistinctContact`: the new point must be farther than
            # `tol` from EVERY point already in this pair's manifold.
            var distinct = True
            for k in range(first_idx, num_contacts):
                var o = k * CONTACT_SIZE
                var dx = r[1] - rebind[Scalar[DTYPE]](
                    contacts[env, o + CONTACT_IDX_POS_X]
                )
                var dy = r[2] - rebind[Scalar[DTYPE]](
                    contacts[env, o + CONTACT_IDX_POS_Y]
                )
                var dz = r[3] - rebind[Scalar[DTYPE]](
                    contacts[env, o + CONTACT_IDX_POS_Z]
                )
                if sqrt(dx * dx + dy * dy + dz * dz) <= tol:
                    distinct = False
                    break
            if not distinct:
                continue

            var off = num_contacts * CONTACT_SIZE
            contacts[env, off + CONTACT_IDX_BODY_A] = Scalar[DTYPE](body_a)
            contacts[env, off + CONTACT_IDX_BODY_B] = Scalar[DTYPE](body_b)
            contacts[env, off + CONTACT_IDX_POS_X] = r[1]
            contacts[env, off + CONTACT_IDX_POS_Y] = r[2]
            contacts[env, off + CONTACT_IDX_POS_Z] = r[3]
            # Same `body_b -> body_a` flip the single-point emit applies.
            contacts[env, off + CONTACT_IDX_NX] = -r[4]
            contacts[env, off + CONTACT_IDX_NY] = -r[5]
            contacts[env, off + CONTACT_IDX_NZ] = -r[6]
            # ⚠ THE FIRST POINT'S DISTANCE, not the perturbed query's — see
            # the module docstring.
            contacts[env, off + CONTACT_IDX_DIST] = dist0
            contacts[
                env, off + CONTACT_IDX_INCLUDEMARGIN
            ] = contact_margin - contact_gap
            contacts[env, off + CONTACT_IDX_FRICTION] = contact_friction
            contacts[
                env, off + CONTACT_IDX_FRICTION_SPIN
            ] = contact_friction_spin
            contacts[
                env, off + CONTACT_IDX_FRICTION_ROLL
            ] = contact_friction_roll
            contacts[env, off + CONTACT_IDX_CONDIM] = Scalar[DTYPE](
                contact_condim
            )
            num_contacts += 1
            written += 1

    return written