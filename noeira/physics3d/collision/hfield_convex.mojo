"""HEIGHTFIELD x convex — MuJoCo's `mjc_ConvexHField`.

Port of `references/mujoco-3.11.0/src/engine/engine_collision_convex.c:1125`
and the two helpers above it, `addVert` (:1090) and `addPrismVert` (:1105).

WHAT IT DOES. A heightfield is never collided as a heightfield. The other geom
is transformed into the field's local frame, its AABB there is measured with
six support queries, that box is clipped to a sub-rectangle of the grid, and
then ONE TRIANGULAR PRISM PER HALF-CELL is collided against it with the
ordinary convex query. Each prism has six vertices — three on the base at
`z = -size[3]`, three on the sampled surface — and they are pushed through a
SLIDING WINDOW, so consecutive calls share two vertices and the two triangles
of a quad cell cost three pushes rather than six.

⚠⚠ THE WINDOW IS THE ALGORITHM, NOT AN OPTIMISATION. `addPrismVert` shifts
`prism[0]<-[1]`, `[1]<-[2]`, `[3]<-[4]`, `[4]<-[5]` and writes the new vertex
into `[2]` and `[5]`. Emitting the six vertices of each triangle directly
gives the same SET for the first triangle of a row and a different one for
every triangle after it, because the shift is what makes triangle `i` reuse
triangle `i-1`'s last two columns. The row is primed with two pushes before
the test loop starts for exactly that reason.

⚠ `i` ALTERNATES THE ROW, NOT THE COLUMN. `int dr = 1 - i`, so `i == 0` takes
row `r+1` and `i == 1` takes row `r`. Reading it the other way round mirrors
every prism about the cell diagonal.

⚠ THE MARGIN GOES INTO THE PRISM'S TOP *AND* INTO THE GEOM'S SUPPORT.
`addPrismVert` adds `margin` to `prism[5][2]`, `obj2.margin = margin` makes
the geom's support grow by `margin/2`, and `mjc_penetration` is then
called with a margin of ZERO — so the reported distance is inflated by
1.5 x margin (AUD-32; this docstring used to claim the geom was not
inflated, and the code believed it: every hfield contact with margin > 0
read `margin/2` too shallow).

⚠ THE PRISM HEIGHT TEST IS A CHEAP REJECT WITH A REAL EFFECT ON THE CONTACT
SET: a prism whose three top vertices are all below the other geom's lowest
support point is skipped BEFORE the convex query, so it can never produce a
contact even a grazing one. Dropping the test changes which cells report.

⚠ CONTACTS COME BACK IN THE FIELD'S LOCAL FRAME and are rotated into the world
by the caller's `mat1` — `normal` by rotation alone, `pos` by rotation and
translation.
"""

from std.math import sqrt, floor, ceil

from layout import TileTensor, TensorLayout

from ..constants import GEOM_HFIELD
from ..kinematics.quat_math import quat_rotate, quat_rotate_inverse, quat_mul
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
    MODEL_HFIELD_META_SIZE,
    HFIELD_META_IDX_ADR,
    HFIELD_META_IDX_NROW,
    HFIELD_META_IDX_NCOL,
    HFIELD_META_IDX_SIZE_X,
    HFIELD_META_IDX_SIZE_Y,
    HFIELD_META_IDX_SIZE_Z,
    HFIELD_META_IDX_SIZE_BASE,
    MJ_CCD_TOLERANCE,
    MJ_CCD_ITERATIONS,
)
from .gjk import gjk_epa, _support, EPA_DBG

# `mjMAXCONPAIR` (`mjmodel.h:29`) — MuJoCo's own per-pair ceiling, and the loop
# in `mjc_ConvexHField` breaks out of all three loops when it is reached.
#
# ⚠⚠ IT IS NOT A BUFFER SIZE HERE, AND IT MUST NOT BECOME ONE. This routine
# writes each contact into the model's record tensor AS IT FINDS IT rather
# than collecting a manifold first, the way `_capsule_capsule_contacts` and
# `_capsule_box_contacts` do. Buffering fifty contacts as three `Array`s
# is 350 float64 of PER-THREAD STACK, and the Metal collision kernel does not
# have it: `test_plane_mesh_fields` died with "Compute function exceeds
# available stack space" the moment this file was written that way. Every
# prism produces at most ONE contact, so there is nothing to collect.
comptime HF_MAX_POINTS: Int = 50

# Debug tracing for the sub-grid walk. CPU only (a Metal kernel cannot
# `print`), off in every committed state — it exists so the next person
# instruments the LIVE call rather than re-deriving the loop from the
# reference. Prints one line per prism that reports a contact: the cell, the
# six prism vertices and the query's answer.
comptime HF_DEBUG: Bool = False


@always_inline
def _hf_push[
    DTYPE: DType,
    L_HF_DATA: TensorLayout,
](
    mut prism: Array[Scalar[DTYPE], 18],
    r: Int,
    c: Int,
    i: Int,
    dx: Scalar[DTYPE],
    dy: Scalar[DTYPE],
    sx: Scalar[DTYPE],
    sy: Scalar[DTYPE],
    sz: Scalar[DTYPE],
    margin: Scalar[DTYPE],
    hfield_data: TileTensor[DTYPE, L_HF_DATA, MutAnyOrigin],
    adr: Int,
    ncol: Int,
):
    """`addPrismVert` — shift the window, then write the new vertex."""
    for k in range(3):
        prism[0 * 3 + k] = prism[1 * 3 + k]
        prism[1 * 3 + k] = prism[2 * 3 + k]
        prism[3 * 3 + k] = prism[4 * 3 + k]
        prism[4 * 3 + k] = prism[5 * 3 + k]
    var dr = 1 - i
    var vx = dx * Scalar[DTYPE](c) - sx
    var vy = dy * Scalar[DTYPE](r + dr) - sy
    prism[2 * 3 + 0] = vx
    prism[5 * 3 + 0] = vx
    prism[2 * 3 + 1] = vy
    prism[5 * 3 + 1] = vy
    var h = rebind[Scalar[DTYPE]](hfield_data[adr + (r + dr) * ncol + c])
    prism[5 * 3 + 2] = h * sz + margin


def hfield_convex_contacts[
    DTYPE: DType,
    L_HF_META: TensorLayout,
    L_HF_DATA: TensorLayout,
    L_MESH_VERTS: TensorLayout,
    L_MESH_VERT_EDGEADR: TensorLayout,
    L_MESH_EDGES: TensorLayout,
    L_CONTACTS: TensorLayout,
    L_WS: TensorLayout,
](
    hfield_id: Int,
    p1x: Scalar[DTYPE], p1y: Scalar[DTYPE], p1z: Scalar[DTYPE],
    q1x: Scalar[DTYPE], q1y: Scalar[DTYPE], q1z: Scalar[DTYPE],
    q1w: Scalar[DTYPE],
    gj_type: Int,
    p2x: Scalar[DTYPE], p2y: Scalar[DTYPE], p2z: Scalar[DTYPE],
    q2x: Scalar[DTYPE], q2y: Scalar[DTYPE], q2z: Scalar[DTYPE],
    q2w: Scalar[DTYPE],
    r2: Scalar[DTYPE], hl2: Scalar[DTYPE],
    hx2: Scalar[DTYPE], hy2: Scalar[DTYPE], hz2: Scalar[DTYPE],
    rbound2: Scalar[DTYPE],
    va2: Int, mnv2: Int,
    margin: Scalar[DTYPE],
    nsign: Scalar[DTYPE],
    body_a: Int,
    body_b: Int,
    contact_friction: Scalar[DTYPE],
    contact_friction_spin: Scalar[DTYPE],
    contact_friction_roll: Scalar[DTYPE],
    contact_condim: Int,
    hfield_meta: TileTensor[DTYPE, L_HF_META, MutAnyOrigin],
    hfield_data: TileTensor[DTYPE, L_HF_DATA, MutAnyOrigin],
    # ⚠ THE PER-ENVIRONMENT STRIDE OF `hfield_data`, i.e. `NHFIELD_DATA`.
    # The grid moved from `Model` to `Data` when `quadruped escape` needed a
    # terrain per episode, so the tensor is now `[BATCH * NHFIELD_DATA]` and
    # `hfield_adr` addresses ONE environment's block. Without this every lane
    # would read environment 0's terrain — which on a batch that resets at
    # different times looks like correlated policy behaviour, not a bug.
    hf_stride: Int,
    mesh_verts: TileTensor[DTYPE, L_MESH_VERTS, MutAnyOrigin],
    mesh_vert_edgeadr: TileTensor[DTYPE, L_MESH_VERT_EDGEADR, MutAnyOrigin],
    mesh_edges: TileTensor[DTYPE, L_MESH_EDGES, MutAnyOrigin],
    contacts: TileTensor[DTYPE, L_CONTACTS, MutAnyOrigin],
    ws: TileTensor[DTYPE, L_WS, MutAnyOrigin],
    mut num_contacts: Int,
    max_contacts: Int,
    env: Int,
    # ⚠ THE GAP HALF OF THE PAIR'S MARGIN, DEFAULTED TO 0 SO EVERY
    # EXISTING CALL SITE IS UNCHANGED. `margin` is the narrowphase
    # CUTOFF (`margin + gap`); what a contact STORES as its
    # `includemargin` is `margin - gap`, and the solver excludes
    # `dist >= includemargin`. See `GEOM_IDX_GAP`.
    gap: Scalar[DTYPE] = Scalar[DTYPE](0),
) -> Int:
    """Contacts between heightfield geom 1 and convex geom 2, written to the
    record tensor as they are found.

    The query's normal points `geom1 -> geom2`; `nsign` turns it into the
    record's `body_b -> body_a` (see `_hfield_contacts`' caller).
    """
    var mo = hfield_id * MODEL_HFIELD_META_SIZE
    var adr = env * hf_stride + Int(
        rebind[Scalar[DTYPE]](hfield_meta[mo + HFIELD_META_IDX_ADR])
    )
    var nrow = Int(
        rebind[Scalar[DTYPE]](hfield_meta[mo + HFIELD_META_IDX_NROW])
    )
    var ncol = Int(
        rebind[Scalar[DTYPE]](hfield_meta[mo + HFIELD_META_IDX_NCOL])
    )
    var sx = rebind[Scalar[DTYPE]](hfield_meta[mo + HFIELD_META_IDX_SIZE_X])
    var sy = rebind[Scalar[DTYPE]](hfield_meta[mo + HFIELD_META_IDX_SIZE_Y])
    var sz = rebind[Scalar[DTYPE]](hfield_meta[mo + HFIELD_META_IDX_SIZE_Z])
    var sb = rebind[Scalar[DTYPE]](
        hfield_meta[mo + HFIELD_META_IDX_SIZE_BASE]
    )
    if nrow < 2 or ncol < 2:
        return 0

    # ── geom 2 in the field's local frame ────────────────────────────────
    var lp = quat_rotate_inverse[DTYPE](
        q1x, q1y, q1z, q1w, p2x - p1x, p2y - p1y, p2z - p1z
    )
    # `mji_mulMatTMat3(mat, mat1, mat2)` as a quaternion: conj(q1) * q2.
    var lq = quat_mul[DTYPE](-q1x, -q1y, -q1z, q1w, q2x, q2y, q2z, q2w)

    # ── early return 1: box vs the other geom's bounding SPHERE ──────────
    var radius = rbound2 + margin
    if (
        (sx < lp[0] - radius)
        or (-sx > lp[0] + radius)
        or (sy < lp[1] - radius)
        or (-sy > lp[1] + radius)
        or (sz < lp[2] - radius)
        or (-sb > lp[2] + radius)
    ):
        return 0

    # ── early return 2: box vs the other geom's true AABB ────────────────
    #
    # Six support queries, which is what MuJoCo does rather than inflating a
    # sphere: a long capsule lying flat has an rbound many times its real
    # extent in z, and the sub-grid derived from it would be most of the map.
    var prism = Array[Scalar[DTYPE], 18](fill=Scalar[DTYPE](0))
    var ext = Array[Scalar[DTYPE], 6](fill=Scalar[DTYPE](0))
    for axis in range(3):
        for side in range(2):
            var dx0 = Scalar[DTYPE](0)
            var dy0 = Scalar[DTYPE](0)
            var dz0 = Scalar[DTYPE](0)
            var sgn = Scalar[DTYPE](1) if side == 0 else Scalar[DTYPE](-1)
            if axis == 0:
                dx0 = sgn
            elif axis == 1:
                dy0 = sgn
            else:
                dz0 = sgn
            var warm = 0
            var s = _support[DTYPE, NPRISM=18](
                gj_type, lp[0], lp[1], lp[2],
                lq[0], lq[1], lq[2], lq[3],
                r2, hl2, hx2, hy2, hz2,
                mesh_verts, mesh_vert_edgeadr, mesh_edges, va2, mnv2,
                dx0, dy0, dz0, warm, prism,
            )
            ext[axis * 2 + side] = s[axis]
    var xmax = ext[0]
    var xmin = ext[1]
    var ymax = ext[2]
    var ymin = ext[3]
    var zmax = ext[4]
    var zmin = ext[5]
    if (
        (xmin - margin > sx)
        or (xmax + margin < -sx)
        or (ymin - margin > sy)
        or (ymax + margin < -sy)
        or (zmin - margin > sz)
        or (zmax + margin < -sb)
    ):
        return 0

    # ── the sub-grid ─────────────────────────────────────────────────────
    var fncol = Scalar[DTYPE](ncol - 1)
    var fnrow = Scalar[DTYPE](nrow - 1)
    var cmin = Int(floor((xmin + sx) / (Scalar[DTYPE](2) * sx) * fncol))
    var cmax = Int(ceil((xmax + sx) / (Scalar[DTYPE](2) * sx) * fncol))
    var rmin = Int(floor((ymin + sy) / (Scalar[DTYPE](2) * sy) * fnrow))
    var rmax = Int(ceil((ymax + sy) / (Scalar[DTYPE](2) * sy) * fnrow))
    if cmin < 0:
        cmin = 0
    if cmax > ncol - 1:
        cmax = ncol - 1
    if rmin < 0:
        rmin = 0
    if rmax > nrow - 1:
        rmax = nrow - 1

    var dx = (Scalar[DTYPE](2) * sx) / fncol
    var dy = (Scalar[DTYPE](2) * sy) / fnrow

    # The three base vertices sit at `-size[3]` for every prism and are never
    # rewritten by the window; only x and y move.
    prism[0 * 3 + 2] = -sb
    prism[1 * 3 + 2] = -sb
    prism[2 * 3 + 2] = -sb

    var ncon = 0
    var cap = HF_MAX_POINTS
    for r in range(rmin, rmax):
        _hf_push[DTYPE](
            prism, r, cmin, 0, dx, dy, sx, sy, sz, margin,
            hfield_data, adr, ncol,
        )
        _hf_push[DTYPE](
            prism, r, cmin, 1, dx, dy, sx, sy, sz, margin,
            hfield_data, adr, ncol,
        )
        for c in range(cmin + 1, cmax + 1):
            for i in range(2):
                _hf_push[DTYPE](
                    prism, r, c, i, dx, dy, sx, sy, sz, margin,
                    hfield_data, adr, ncol,
                )
                # prism height test — see the header.
                if (
                    prism[3 * 3 + 2] < zmin
                    and prism[4 * 3 + 2] < zmin
                    and prism[5 * 3 + 2] < zmin
                ):
                    continue
                # ⚠ THE PRISM'S "POSITION" IS ITS CENTROID, NOT THE GEOM'S.
                # `mjc_center` (`engine_collision_convex.c:143`) special-cases
                # mjGEOM_HFIELD to the mean of the six prism vertices; every
                # other type returns `geom_xpos`. That value seeds GJK's first
                # search direction, so passing the heightfield's own origin
                # instead starts the search somewhere else and converges to a
                # different witness on a grazing pair.
                var pcx = Scalar[DTYPE](0)
                var pcy = Scalar[DTYPE](0)
                var pcz = Scalar[DTYPE](0)
                for _v in range(6):
                    pcx += prism[_v * 3 + 0]
                    pcy += prism[_v * 3 + 1]
                    pcz += prism[_v * 3 + 2]
                pcx /= Scalar[DTYPE](6)
                pcy /= Scalar[DTYPE](6)
                pcz /= Scalar[DTYPE](6)
                var res = gjk_epa[DTYPE, NPRISM=18](
                    GEOM_HFIELD,
                    pcx, pcy, pcz,
                    Scalar[DTYPE](0), Scalar[DTYPE](0), Scalar[DTYPE](0),
                    Scalar[DTYPE](1),
                    Scalar[DTYPE](0), Scalar[DTYPE](0),
                    Scalar[DTYPE](0), Scalar[DTYPE](0), Scalar[DTYPE](0),
                    mesh_verts, mesh_vert_edgeadr, mesh_edges, 0, 0,
                    gj_type,
                    lp[0], lp[1], lp[2], lq[0], lq[1], lq[2], lq[3],
                    r2, hl2, hx2, hy2, hz2,
                    va2, mnv2,
                    ws, env,
                    Scalar[DTYPE](MJ_CCD_TOLERANCE),
                    MJ_CCD_ITERATIONS,
                    # ⚠ AUD-32: the GEOM is inflated by margin/2 (MuJoCo sets
                    # `obj2.margin = margin`, convex.c:1246) while the prism —
                    # already raised by `margin` at its top — is not
                    # (`inflate1=False`). `gjk_epa` adds `margin` back to
                    # its result; MuJoCo's `mjc_penetration(..., 0.0)` adds
                    # nothing, so the record stores `res[0] - margin`: the
                    # doubly-inflated distance the reference reports (true
                    # -0.001 at margin 0.04 reads -0.061 on both sides).
                    margin,
                    prism,
                    inflate1=False,
                )
                comptime if HF_DEBUG:
                    if res[0] < Scalar[DTYPE](0):
                        var ds = String("  [hf] r=") + String(r) + " c=" + String(c) + " i=" + String(i) + "  dist=" + String(Float64(res[0])) + "  p=(" + String(Float64(res[1])) + "," + String(Float64(res[2])) + "," + String(Float64(res[3])) + ")"
                        for _v in range(6):
                            ds += " (" + String(Float64(prism[_v*3+0])) + "," + String(Float64(prism[_v*3+1])) + "," + String(Float64(prism[_v*3+2])) + ")"
                        print(ds)
                if res[0] < margin:
                    if num_contacts >= max_contacts:
                        return ncon
                    # local -> world
                    var wn = quat_rotate[DTYPE](
                        q1x, q1y, q1z, q1w, res[4], res[5], res[6]
                    )
                    var wp = quat_rotate[DTYPE](
                        q1x, q1y, q1z, q1w, res[1], res[2], res[3]
                    )
                    var o = num_contacts * CONTACT_SIZE
                    contacts[env, o + CONTACT_IDX_BODY_A] = Scalar[DTYPE](
                        body_a
                    )
                    contacts[env, o + CONTACT_IDX_BODY_B] = Scalar[DTYPE](
                        body_b
                    )
                    contacts[env, o + CONTACT_IDX_POS_X] = wp[0] + p1x
                    contacts[env, o + CONTACT_IDX_POS_Y] = wp[1] + p1y
                    contacts[env, o + CONTACT_IDX_POS_Z] = wp[2] + p1z
                    contacts[env, o + CONTACT_IDX_NX] = nsign * wn[0]
                    contacts[env, o + CONTACT_IDX_NY] = nsign * wn[1]
                    contacts[env, o + CONTACT_IDX_NZ] = nsign * wn[2]
                    contacts[env, o + CONTACT_IDX_DIST] = res[0] - margin
                    contacts[env, o + CONTACT_IDX_INCLUDEMARGIN] = margin - gap
                    contacts[env, o + CONTACT_IDX_FRICTION] = contact_friction
                    contacts[
                        env, o + CONTACT_IDX_FRICTION_SPIN
                    ] = contact_friction_spin
                    contacts[
                        env, o + CONTACT_IDX_FRICTION_ROLL
                    ] = contact_friction_roll
                    contacts[env, o + CONTACT_IDX_CONDIM] = Scalar[DTYPE](
                        contact_condim
                    )
                    # ⚠ THE SUB-GRID WALK, SMUGGLED OUT THROUGH THE FORCE
                    # SLOTS. See `EPA_DBG` in `gjk.mojo`: the columns below are
                    # written by the solver later and are free at detection
                    # time, so a GPU run can be diffed against a CPU one cell
                    # by cell. This is what proved the AABB and the sub-grid
                    # IDENTICAL on both targets while the contacts were not,
                    # which is what moved the hunt off this file and into GJK.
                    comptime if EPA_DBG:
                        contacts[env, o + 10] = Scalar[DTYPE](r)
                        contacts[env, o + 11] = Scalar[DTYPE](c)
                        contacts[env, o + 12] = Scalar[DTYPE](i)
                        contacts[env, o + 17] = Scalar[DTYPE](cmin)
                        contacts[env, o + 18] = Scalar[DTYPE](cmax)
                        contacts[env, o + 19] = Scalar[DTYPE](rmin)
                        contacts[env, o + 20] = Scalar[DTYPE](rmax)
                        contacts[env, o + 21] = xmin
                        contacts[env, o + 22] = xmax
                        contacts[env, o + 23] = ymin
                        contacts[env, o + 24] = ymax
                        contacts[env, o + 25] = zmin
                        contacts[env, o + 26] = zmax
                    num_contacts += 1
                    ncon += 1
                    if ncon >= cap:
                        return ncon
    return ncon
