"""`mj_ray` — the whole model against one ray, and `ray_eliminate`'s filter.

`engine_ray.c:1308` and `:68`. This is the piece that turns the per-geom
routines into a query: loop every geom, skip the ones the filter excludes,
dispatch on type, keep the nearest, and report which geom it was.

⚠⚠ THIS MODULE IS THE ONE THAT TOUCHES `Model` AND `Data`, and it is separate
from the rest of `ray/` for exactly that reason. `geom.mojo`, `triangle.mojo`,
`hfield.mojo` and `mesh.mojo` take scalars and know nothing about record
layouts, which is what lets a GPU kernel call them with its own storage. Only
this file binds the CPU-side tables, so a batched tracer replaces THIS and
reuses everything underneath.

WHAT `ray_eliminate` ACTUALLY DECIDES, because three of its four rules are
easy to get subtly wrong:

  1. **body exclusion** — one body id, not a subtree. A rangefinder on a
     quadruped's torso excludes the TORSO and still sees its own legs. That
     is MuJoCo's behaviour and it is not obviously the intended one; it is
     reproduced rather than improved.
  2. **invisible** — precomputed into `GEOM_IDX_RAY_VISIBLE` at build time,
     because both of MuJoCo's spellings (no material with `rgba[3] == 0`, or a
     material with `rgba[3] == 0`) are fixed once the model exists.
  3. **static** — `flg_static == 0` drops every geom welded to the world.
     ⚠ `body_weldid == 0`, NOT `body == 0`: a body rigidly welded to the world
     through a chain of weld constraints is static too, and testing the body
     id alone would let the floor through on a model that welds to it.
  4. **group mask** — only when a mask is given. The index is CLAMPED into
     `[0, mjNGROUP-1]` rather than rejected, so a `<geom group="9">` reads
     the last mask slot instead of falling off the end.

⚠ A RANGEFINDER USES NONE OF 3 AND 4. `engine_sensor.c:616` calls
`mj_ray(..., geomgroup=NULL, flg_static=1, bodyexclude=site_bodyid, ...)`, so
statics are INCLUDED and no group is filtered. Rules 1 and 2 are the whole
filter for the sensor this was built for — which is why 2 is not optional and
why it is precomputed rather than skipped.
"""

from layout import Layout, LayoutTensor

from noeira.math3d import Vec3 as Vec3Generic, Quat as QuatGeneric

from ..constants import (
    GEOM_PLANE,
    GEOM_SPHERE,
    GEOM_CAPSULE,
    GEOM_CYLINDER,
    GEOM_MESH,
    GEOM_HFIELD,
)
from ..gpu.constants import (
    MODEL_GEOM_SIZE,
    GEOM_IDX_TYPE,
    GEOM_IDX_BODY,
    GEOM_IDX_POS_X,
    GEOM_IDX_QUAT_X,
    GEOM_IDX_QUAT_W,
    GEOM_IDX_QUAT_Z,
    GEOM_IDX_QUAT_Y,
    GEOM_IDX_POS_Z,
    GEOM_IDX_POS_Y,
    GEOM_IDX_RADIUS,
    GEOM_IDX_HALF_LENGTH,
    GEOM_IDX_HALF_X,
    GEOM_IDX_HALF_Y,
    GEOM_IDX_HALF_Z,
    GEOM_IDX_MESH_ID,
    GEOM_IDX_HFIELD_ID,
    GEOM_IDX_RAY_VISIBLE,
    GEOM_IDX_GROUP,
    MODEL_BODY_SIZE,
    BODY_IDX_WELDID,
    MODEL_MESH_META_SIZE,
    MESH_META_IDX_BVHADR,
    MESH_META_IDX_BVHNUM,
    MESH_META_IDX_TRIADR,
    MESH_META_IDX_TRINUM,
    MODEL_HFIELD_META_SIZE,
    HFIELD_META_IDX_ADR,
    HFIELD_META_IDX_NROW,
    HFIELD_META_IDX_NCOL,
    HFIELD_META_IDX_SIZE_X,
    HFIELD_META_IDX_SIZE_Y,
    HFIELD_META_IDX_SIZE_Z,
    HFIELD_META_IDX_SIZE_BASE,
)
from .geom import RAY_NO_HIT, ray_geom
from .hfield import ray_hfield
from .mesh import ray_mesh, ray_mesh_bvh

comptime RAY_NGROUP: Int = 6
"""`mjNGROUP`. The group mask is this wide and the geom's group is clamped
into it, never rejected."""


@fieldwise_init
struct RayHit[DTYPE: DType](Copyable, ImplicitlyCopyable, Movable):
    """`mj_ray`'s three outputs, plus where on a mesh it landed.

    `geom` is -1 exactly when `t` is negative.

    ⚠ `tri` / `bu` / `bv` ARE FOR TEXTURING AND FOR NOTHING ELSE. They are the
    winning triangle's arena record and its barycentric weights when the hit
    was on a MESH, and `tri = -1` otherwise — a primitive geom has no triangle
    and MuJoCo generates its texture coordinates from the surface point
    instead (`settexture`'s object-linear texgen). Every consumer that
    predates them ignores them; `mj_ray` itself does not report them, so a
    gate against MuJoCo can only check the distance and the geom, which is
    what `ray/`'s four gates already do.
    """

    var t: Scalar[Self.DTYPE]
    var geom: Int
    var normal: Vec3Generic[Self.DTYPE]
    var tri: Int
    var bu: Scalar[Self.DTYPE]
    var bv: Scalar[Self.DTYPE]


def ray_model[
    DTYPE: DType,
    L_GEOMS: Layout,
    L_BODIES: Layout,
    L_XPOS: Layout,
    L_XQUAT: Layout,
    L_MESH_META: Layout,
    L_TRI: Layout,
    L_HF_META: Layout,
    L_HF: Layout,
](
    geoms: LayoutTensor[DTYPE, L_GEOMS, MutAnyOrigin],
    ngeom: Int,
    bodies: LayoutTensor[DTYPE, L_BODIES, MutAnyOrigin],
    xpos: LayoutTensor[DTYPE, L_XPOS, MutAnyOrigin],
    xquat: LayoutTensor[DTYPE, L_XQUAT, MutAnyOrigin],
    env: Int,
    mesh_meta: LayoutTensor[DTYPE, L_MESH_META, MutAnyOrigin],
    mesh_tris: LayoutTensor[DTYPE, L_TRI, MutAnyOrigin],
    hfield_meta: LayoutTensor[DTYPE, L_HF_META, MutAnyOrigin],
    hfield_data: LayoutTensor[DTYPE, L_HF, MutAnyOrigin],
    hf_stride: Int,
    pnt: Vec3Generic[DTYPE],
    vec: Vec3Generic[DTYPE],
    bodyexclude: Int = -1,
    flg_static: Bool = True,
    use_group: Bool = False,
    # ⚠ A BITMASK, NOT AN ARRAY. `geomgroup` is six booleans and the geom's
    # group indexes them at RUNTIME — which is the per-thread-array read that
    # is silently wrong on Metal (`87960e10`, the fourth instance in this
    # engine). Six bits in an `Int` is the same information with no thread
    # storage: bit `g` set means group `g` is visible.
    group_mask: Int = 0x3F,
    # Rows `[g_start, ngeom)` are tested. The camera tracer's conditional
    # sites sit past its ordinary geoms and are tested one row at a time
    # with this, through the SAME dispatch — a second copy of the per-type
    # size spelling below is how two ray paths drift apart.
    g_start: Int = 0,
    tcut: Scalar[DTYPE] = Scalar[DTYPE](RAY_NO_HIT),
) -> RayHit[DTYPE] where DTYPE.is_floating_point():
    """Nearest intersection of `pnt + x*vec` with the model.

    `geom_xpos`/`geom_xquat` are the geoms' WORLD poses, composed by the
    caller — this module does not run forward kinematics, so a caller stepping
    a batch passes the env it means.

    ⚠ `vec` NEED NOT BE NORMALISED and `t` is in units of `|vec|`, the same
    contract every routine underneath keeps. The reference RAISES on a
    zero-length `vec`; here that case simply hits nothing, because a ray query
    inside a sensor loop is not a place to abort a simulation from.

    `tcut` (negative = none) is a hit the CALLER already holds: only a hit
    strictly nearer is returned, and a mesh walk is cut there. The renderer
    calls this one geom at a time (`render.trace_visual`, so it can skip
    culled geoms) and passes its running best, which keeps the answer the one
    a single call over the same geoms would give.
    """
    var best = Scalar[DTYPE](RAY_NO_HIT)
    var best_geom = -1
    var best_normal = Vec3Generic[DTYPE](0, 0, 0)
    var best_tri = -1
    var best_bu = Scalar[DTYPE](0)
    var best_bv = Scalar[DTYPE](0)

    for g in range(g_start, ngeom):

        # ── ray_eliminate ────────────────────────────────────────────────
        var body = Int(rebind[Scalar[DTYPE]]((geoms[g, GEOM_IDX_BODY])))
        if body == bodyexclude:
            continue
        if rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_RAY_VISIBLE]) == 0:
            continue
        if not flg_static:
            # ⚠ weldid, not the body id — see the module docstring.
            var wb = body
            if body >= 0:
                wb = Int(
                    rebind[Scalar[DTYPE]](bodies[body, BODY_IDX_WELDID])
                )
            if wb == 0:
                continue
        if use_group:
            var gid = Int(rebind[Scalar[DTYPE]]((geoms[g, GEOM_IDX_GROUP])))
            if gid < 0:
                gid = 0
            if gid > RAY_NGROUP - 1:
                gid = RAY_NGROUP - 1
            if (group_mask >> gid) & 1 == 0:
                continue

        var gtype = Int(rebind[Scalar[DTYPE]]((geoms[g, GEOM_IDX_TYPE])))
        # ⚠ COMPOSED HERE RATHER THAN HOISTED. An earlier form took
        # precomputed pose arrays, which a kernel cannot hold: a thread owns
        # one RAY, not a scene. Recomputing costs one quaternion rotate and
        # one multiply per (ray, geom) — ~30 flops against a narrow-phase
        # query — and it is what makes ONE implementation serve both targets.
        var lp = Vec3Generic[DTYPE](
            rebind[Scalar[DTYPE]]((geoms[g, GEOM_IDX_POS_X])),
            rebind[Scalar[DTYPE]]((geoms[g, GEOM_IDX_POS_Y])),
            rebind[Scalar[DTYPE]]((geoms[g, GEOM_IDX_POS_Z])),
        )
        var lq = QuatGeneric[DTYPE](
            rebind[Scalar[DTYPE]]((geoms[g, GEOM_IDX_QUAT_W])),
            rebind[Scalar[DTYPE]]((geoms[g, GEOM_IDX_QUAT_X])),
            rebind[Scalar[DTYPE]]((geoms[g, GEOM_IDX_QUAT_Y])),
            rebind[Scalar[DTYPE]]((geoms[g, GEOM_IDX_QUAT_Z])),
        )
        var pos = lp
        var quat = lq
        if body > 0:
            # `Data.xquat` is packed (x, y, z, w); `Quat` takes (w, x, y, z).
            var bq = QuatGeneric[DTYPE](
                rebind[Scalar[DTYPE]](xquat[env, body * 4 + 3]),
                rebind[Scalar[DTYPE]](xquat[env, body * 4 + 0]),
                rebind[Scalar[DTYPE]](xquat[env, body * 4 + 1]),
                rebind[Scalar[DTYPE]](xquat[env, body * 4 + 2]),
            )
            var bp = Vec3Generic[DTYPE](
                rebind[Scalar[DTYPE]](xpos[env, body * 3 + 0]),
                rebind[Scalar[DTYPE]](xpos[env, body * 3 + 1]),
                rebind[Scalar[DTYPE]](xpos[env, body * 3 + 2]),
            )
            pos = bp + bq.rotate_vec(lp)
            quat = bq * lq
        var t = Scalar[DTYPE](RAY_NO_HIT)
        var n = Vec3Generic[DTYPE](0, 0, 0)
        var gtri = -1
        var gbu = Scalar[DTYPE](0)
        var gbv = Scalar[DTYPE](0)

        if gtype == GEOM_MESH:
            var mid = Int(
                rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_MESH_ID])
            )
            if mid >= 0:
                var mb = mid * MODEL_MESH_META_SIZE
                var half = Vec3Generic[DTYPE](
                    rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_HALF_X]),
                    rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_HALF_Y]),
                    rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_HALF_Z]),
                )
                var triadr = Int(
                    rebind[Scalar[DTYPE]](mesh_meta[mb + MESH_META_IDX_TRIADR])
                )
                var trinum = Int(
                    rebind[Scalar[DTYPE]](mesh_meta[mb + MESH_META_IDX_TRINUM])
                )
                var bvhnum = Int(
                    rebind[Scalar[DTYPE]](mesh_meta[mb + MESH_META_IDX_BVHNUM])
                )
                # ⚠⚠ TWO LEGS THAT MUST RETURN THE SAME THING, AND ONE OF THEM
                # IS ~35x SLOWER. `BVHNUM == 0` means the model carries a soup
                # with no tree — every model the parser builds now carries
                # one, so this is a fallback for a hand-assembled `Model` and
                # NOT a mode anything selects. It is here rather than an error
                # because a `Model` built field-by-field in a test is a real
                # caller (`_a_hand_built_call_must_do_what_the_loader_does`),
                # and the linear sweep is the correct answer, just slowly.
                if bvhnum > 0:
                    var rb = ray_mesh_bvh[DTYPE, L_TRI](
                        pos,
                        quat,
                        half,
                        mesh_tris,
                        triadr,
                        trinum,
                        Int(
                            rebind[Scalar[DTYPE]](
                                mesh_meta[mb + MESH_META_IDX_BVHADR]
                            )
                        ),
                        bvhnum,
                        pnt,
                        vec,
                        # The best hit so far (or the caller's): a mesh
                        # BEHIND it is not walked (a tie keeps the earlier
                        # geom, as the comparison below does, so the answer
                        # cannot move).
                        best if best >= 0 else tcut,
                    )
                    t = rb.t
                    n = rb.normal
                    gtri = rb.tri
                    gbu = rb.bu
                    gbv = rb.bv
                else:
                    var r = ray_mesh[DTYPE, L_TRI](
                        pos,
                        quat,
                        half,
                        mesh_tris,
                        triadr,
                        trinum,
                        pnt,
                        vec,
                    )
                    t = r.t
                    n = r.normal
                    gtri = r.tri
                    gbu = r.bu
                    gbv = r.bv
        elif gtype == GEOM_HFIELD:
            var hid = Int(
                rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_HFIELD_ID])
            )
            if hid >= 0:
                var hb = hid * MODEL_HFIELD_META_SIZE
                var r = ray_hfield[DTYPE, L_HF](
                    pos,
                    quat,
                    Int(
                        rebind[Scalar[DTYPE]](hfield_meta[hb + HFIELD_META_IDX_NROW])
                    ),
                    Int(
                        rebind[Scalar[DTYPE]](hfield_meta[hb + HFIELD_META_IDX_NCOL])
                    ),
                    rebind[Scalar[DTYPE]](hfield_meta[hb + HFIELD_META_IDX_SIZE_X]),
                    rebind[Scalar[DTYPE]](hfield_meta[hb + HFIELD_META_IDX_SIZE_Y]),
                    rebind[Scalar[DTYPE]](hfield_meta[hb + HFIELD_META_IDX_SIZE_Z]),
                    rebind[Scalar[DTYPE]](hfield_meta[hb + HFIELD_META_IDX_SIZE_BASE]),
                    hfield_data,
                    env * hf_stride
                    + Int(
                        rebind[Scalar[DTYPE]](
                            rebind[Scalar[DTYPE]](hfield_meta[hb + HFIELD_META_IDX_ADR])
                        )
                    ),
                    pnt,
                    vec,
                )
                t = r[0]
                n = r[1]
        else:
            # ⚠ THE PURE GEOMS TAKE `size` IN THEIR OWN SPELLING, and the
            # record keeps each type's numbers in its own slots rather than a
            # shared `size[3]`. A sphere's radius is `RADIUS`, a capsule's
            # half-length is `HALF_LENGTH`, a box's extents are `HALF_*` —
            # feeding `HALF_*` to a capsule would silently make it a different
            # capsule, which is why this is spelled per type and not hoisted.
            var hx = rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_HALF_X])
            var hy = rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_HALF_Y])
            var hz = rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_HALF_Z])
            var rad = rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_RADIUS])
            var hl = rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_HALF_LENGTH])
            # Read from `full_parser`'s per-type assignment, not guessed:
            #   SPHERE     radius = s0, and half_* = s0 as well
            #   CAPSULE    radius = s0, half_length = s1
            #   CYLINDER   radius = s0, half_length = s1
            #   BOX        half_* = s0,s1,s2   (radius is the DIAGONAL)
            #   ELLIPSOID  half_* = s0,s1,s2   (radius keeps s0 for the AABB)
            #   PLANE      half_x/y = s0,s1;  half_z is the RENDER GRID SPACING
            # ⚠ `radius` is NOT `size[0]` for a box or an ellipsoid, so a
            # single `(radius, half_length, half_z)` for every type would give
            # `ray_ellipsoid` a sphere of the x semi-axis — the same mistake
            # `_aabb_half_extents` made on flybody's labrum.
            # ⚠ A PLANE'S `half_z` MUST NOT REACH `ray_plane`: it is the grid
            # spacing, and `ray_plane` only reads x and y, so it is passed as
            # zero to keep a future reader from finding a meaningful-looking
            # number there.
            var size = Vec3Generic[DTYPE](hx, hy, hz)
            if gtype == GEOM_SPHERE:
                size = Vec3Generic[DTYPE](rad, rad, rad)
            elif gtype == GEOM_CAPSULE or gtype == GEOM_CYLINDER:
                size = Vec3Generic[DTYPE](rad, hl, Scalar[DTYPE](0))
            elif gtype == GEOM_PLANE:
                size = Vec3Generic[DTYPE](hx, hy, Scalar[DTYPE](0))
            var r = ray_geom[DTYPE](pos, quat, size, pnt, vec, gtype)
            t = r[0]
            n = r[1]

        if t >= 0 and (best < 0 or t < best) and (tcut < 0 or t < tcut):
            best = t
            best_geom = g
            best_normal = n
            best_tri = gtri
            best_bu = gbu
            best_bv = gbv

    return RayHit[DTYPE](
        best, best_geom, best_normal, best_tri, best_bu, best_bv
    )

