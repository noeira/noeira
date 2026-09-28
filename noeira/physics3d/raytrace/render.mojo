"""One pixel of a batched camera observation — `render.py`'s megakernel body.

The whole of the tracer's per-pixel work, as a plain function over
`LayoutTensor`s and an `env` index, so the host leg and the device leg are the
SAME code. `physics3d/ray` is built on that discipline and
`tests/physics3d/test_ray_model_gpu_vs_cpu.mojo` exists to enforce it; this
module inherits both, and `test_camera_render_gpu_vs_cpu.mojo` is its half of
the bargain.

⚠⚠ NO PER-THREAD ARRAY APPEARS HERE, AND THAT IS A REQUIREMENT, NOT A HABIT.
An `Array` indexed by a runtime value reads back silently wrong on Metal
and has done so four times in this engine (`87960e10` is the most recent). A
pixel's three colour channels are three named scalars in a `Vec3`, not a
three-element array, for exactly that reason.

WHY A TRACER AND NOT `Renderer3D`
=================================
`Renderer3D` is a rasteriser over CPU-built draw commands: N environments cost
N sequential scenes, a window, a swapchain and a shadow pass with its own
depth-map resolution and bias. This is one kernel over (env, pixel) that reads
the batched `Data` in place. The assessment
(`docs/DM_CONTROL_AND_CAMERA_ASSESSMENT_2026_08_24.md` §6) is blunt about which
one belongs where: **the SDL pipeline stays the VIEWER, the tracer is the
OBSERVATION path**, and MuJoCo itself keeps the same split. Per frame the
tracer is the more expensive of the two — it is chosen because it batches and
because it reuses collision geometry that is already on the device, not because
it is faster at one image.
"""

from layout import Layout, LayoutTensor

from noeira.math3d import Vec3 as Vec3Generic

from noeira.math3d import Quat as QuatGeneric

from ..constants import GEOM_BOX
from ..gpu.constants import (
    GEOM_IDX_HALF_Z,
    GEOM_IDX_TYPE,
    GEOM_IDX_POS_X,
    GEOM_IDX_POS_Y,
    GEOM_IDX_POS_Z,
    GEOM_IDX_QUAT_X,
    GEOM_IDX_QUAT_Y,
    GEOM_IDX_QUAT_Z,
    GEOM_IDX_QUAT_W,
    GEOM_IDX_BODY,
)
from ..ray.model import RayHit, ray_model
from .camera import CameraFrame, camera_pixel_ray, camera_sample_ray
from .appearance import (
    _clamp01,
    geom_uv,
    sample_texture_lod,
    texture_lod,
    shade_lights,
    Texel,
    UV,
)
from ..parser.flat_model import TEX_CUBE
from ..gpu.constants import MESH_ARENA_RECORD
from .visual_records import *


@fieldwise_init
struct PixelHit[DTYPE: DType](Copyable, Movable):
    """What one pixel produced: colour, planar depth, and what was hit."""

    var rgb: Vec3Generic[Self.DTYPE]
    """Linear colour in [0, 1]. The background colour on a miss."""

    var depth: Scalar[Self.DTYPE]
    """PLANAR depth — the hit distance projected onto the optical axis, which
    is what a depth camera reports and what `render.py` writes.

    ⚠ NOT THE RAY PARAMETER. `dist * (-ray_dir_local.z)` differs from `dist`
    by `cos(theta)` off-axis; at a 90 deg fovy the corner pixels differ by
    ~30%. A policy trained on one and deployed against the other sees a
    barrel-distorted world.

    ⚠ **0 MEANS NO HIT**, not "zero metres away". Nothing can be at zero
    distance from a camera that is not inside a geom, so the sentinel is
    unambiguous — but it is a SENTINEL, and a consumer normalising depth must
    handle it before dividing. Same contract, same reason, as
    `rangefinder`'s -1."""

    var geom: Int
    """The geom the primary ray hit, or -1 for background.

    ⚠ THIS IS THE SEGMENTATION CHANNEL AND IT IS FREE. `render.py` writes a
    whole `seg_data` buffer for it; here it falls out of the same `RayHit`.
    It is also what makes a colour-blind gate possible: three of the five
    defects `ray_model` was falsified against left the distance untouched and
    showed only as a different geom."""

    var refl_geom: Int
    """What the MIRROR shows at this pixel, or -1 — no mirror here, or the
    reflected ray escaped.

    ⚠ THE COLOUR-BLIND CHECK FOR THE REFLECTION PASS, and it is not free
    decoration: our first reflection filled 100% of the mirror where the
    reference fills 76% of it, and the colour alone could not say whether
    that was a shading gain or a geometry error. `seg` answers that question
    for the primary ray; this answers it for the second one.
    """


@always_inline
def _geom_world_pose[
    DTYPE: DType, L_GEOMS: Layout, L_XPOS: Layout, L_XQUAT: Layout
](
    geoms: LayoutTensor[DTYPE, L_GEOMS, MutAnyOrigin],
    xpos: LayoutTensor[DTYPE, L_XPOS, MutAnyOrigin],
    xquat: LayoutTensor[DTYPE, L_XQUAT, MutAnyOrigin],
    env: Int,
    g: Int,
) -> Tuple[Vec3Generic[DTYPE], QuatGeneric[DTYPE]] where (
    DTYPE.is_floating_point()
):
    """The geom's world pose, composed exactly as `ray_model` composes it.

    ⚠ RECOMPUTED RATHER THAN RETURNED BY THE HIT. A `RayHit` carrying a frame
    would carry it on every miss too, and a miss is the common case in a
    per-pixel loop over a whole scene.
    """
    var lp = Vec3Generic[DTYPE](
        rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_POS_X]),
        rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_POS_Y]),
        rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_POS_Z]),
    )
    var lq = QuatGeneric[DTYPE](
        rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_QUAT_W]),
        rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_QUAT_X]),
        rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_QUAT_Y]),
        rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_QUAT_Z]),
    )
    var body = Int(rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_BODY]))
    if body <= 0:
        return (lp, lq)
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
    return (bp + bq.rotate_vec(lp), bq * lq)


@always_inline
def _tri_bary[
    DTYPE: DType, L_TRI: Layout
](
    mesh_tris: LayoutTensor[DTYPE, L_TRI, MutAnyOrigin],
    tri: Int,
    p: Vec3Generic[DTYPE],
) -> UV[DTYPE] where DTYPE.is_floating_point():
    """Barycentric weights `(bu, bv)` of `v0`, `v1` for a point in the
    triangle's PLANE, `ray_triangle`'s convention (`v2` takes the rest).

    ⚠ NOT CLAMPED TO THE TRIANGLE, ON PURPOSE. It is evaluated at a
    neighbouring pixel's footprint, which is usually outside the hit triangle;
    extrapolating that triangle's UV map is what keeps the derivative
    continuous across a UV seam, where the neighbouring triangle's own UVs
    would jump to the other side of the atlas."""
    var o = tri * MESH_ARENA_RECORD
    var v2 = Vec3Generic[DTYPE](
        rebind[Scalar[DTYPE]](mesh_tris[o + 6]),
        rebind[Scalar[DTYPE]](mesh_tris[o + 7]),
        rebind[Scalar[DTYPE]](mesh_tris[o + 8]),
    )
    var e0 = Vec3Generic[DTYPE](
        rebind[Scalar[DTYPE]](mesh_tris[o + 0]),
        rebind[Scalar[DTYPE]](mesh_tris[o + 1]),
        rebind[Scalar[DTYPE]](mesh_tris[o + 2]),
    ) - v2
    var e1 = Vec3Generic[DTYPE](
        rebind[Scalar[DTYPE]](mesh_tris[o + 3]),
        rebind[Scalar[DTYPE]](mesh_tris[o + 4]),
        rebind[Scalar[DTYPE]](mesh_tris[o + 5]),
    ) - v2
    var d = p - v2
    var a00 = e0.dot(e0)
    var a01 = e0.dot(e1)
    var a11 = e1.dot(e1)
    var r0 = d.dot(e0)
    var r1 = d.dot(e1)
    var det = a00 * a11 - a01 * a01
    if det == Scalar[DTYPE](0):
        return UV[DTYPE](Scalar[DTYPE](0), Scalar[DTYPE](0))
    return UV[DTYPE](
        (r0 * a11 - r1 * a01) / det, (r1 * a00 - r0 * a01) / det
    )


def shade_hit[
    DTYPE: DType,
    SHADOWS: Bool,
    L_GEOMS: Layout,
    L_APP: Layout,
    L_BODIES: Layout,
    L_XPOS: Layout,
    L_XQUAT: Layout,
    L_MESH_META: Layout,
    L_TRI: Layout,
    L_UV: Layout,
    L_HF_META: Layout,
    L_HF: Layout,
    L_MAT: Layout,
    L_TEX: Layout,
    L_TEXELS: Layout,
    L_LIGHTS: Layout,
](
    geoms: LayoutTensor[DTYPE, L_GEOMS, MutAnyOrigin],
    ngeom: Int,
    appearance: LayoutTensor[DTYPE, L_APP, MutAnyOrigin],
    bodies: LayoutTensor[DTYPE, L_BODIES, MutAnyOrigin],
    xpos: LayoutTensor[DTYPE, L_XPOS, MutAnyOrigin],
    xquat: LayoutTensor[DTYPE, L_XQUAT, MutAnyOrigin],
    env: Int,
    mesh_meta: LayoutTensor[DTYPE, L_MESH_META, MutAnyOrigin],
    mesh_tris: LayoutTensor[DTYPE, L_TRI, MutAnyOrigin],
    mesh_uv: LayoutTensor[DTYPE, L_UV, MutAnyOrigin],
    hfield_meta: LayoutTensor[DTYPE, L_HF_META, MutAnyOrigin],
    hfield_data: LayoutTensor[DTYPE, L_HF, MutAnyOrigin],
    hf_stride: Int,
    materials: LayoutTensor[DTYPE, L_MAT, MutAnyOrigin],
    textures: LayoutTensor[DTYPE, L_TEX, MutAnyOrigin],
    texels: LayoutTensor[DType.uint8, L_TEXELS, MutAnyOrigin],
    lights: LayoutTensor[DTYPE, L_LIGHTS, MutAnyOrigin],
    nlight: Int,
    hit: RayHit[DTYPE],
    hitpoint: Vec3Generic[DTYPE],
    eye: Vec3Generic[DTYPE],
    gaze: Vec3Generic[DTYPE],
    base_scale: Scalar[DTYPE],
    ray_org: Vec3Generic[DTYPE],
    ray_dx: Vec3Generic[DTYPE],
    ray_dy: Vec3Generic[DTYPE],
) -> Vec3Generic[DTYPE] where DTYPE.is_floating_point():
    """The colour of one surface point: material, texel, then the lights.

    `ray_org` with `ray_dx` / `ray_dy` are the rays of the pixels one to the
    RIGHT and one BELOW, from the origin this hit's ray left — the texture's
    level of detail is measured with them (see the texel block below). For a
    reflection they are the mirrored eye and the mirrored neighbours.

    `base_scale` multiplies the geom's rgba BEFORE the shading, which is
    `renderGeomReflection`'s whole body:

        for (k) { old[k] = rgba[k]; rgba[k] *= reflectance; }
        renderGeom(...); restore

    ⚠ THAT IS NOT THE SAME AS SCALING THE RESULT. The ambient and diffuse
    terms are linear in the material colour and the SPECULAR term does not
    contain it at all (`GL_SPECULAR` is the material's own `specular`, not
    its colour), so a reflection at 0.5 keeps its highlights at full
    strength. Scaling the returned colour instead would dim them, which on a
    polished floor is most of what a reflection shows.
    """
    var g = hit.geom
    var ab = g * VIS_GEOM_APPEARANCE
    var base = Vec3Generic[DTYPE](
        rebind[Scalar[DTYPE]](appearance[ab + APP_IDX_R]) * base_scale,
        rebind[Scalar[DTYPE]](appearance[ab + APP_IDX_G]) * base_scale,
        rebind[Scalar[DTYPE]](appearance[ab + APP_IDX_B]) * base_scale,
    )
    var matid = Int(rebind[Scalar[DTYPE]](appearance[ab + APP_IDX_MATID]))
    var specular = Scalar[DTYPE](0.5)
    var shininess = Scalar[DTYPE](0.5)
    var texid = -1
    var repeat_u = Scalar[DTYPE](1)
    var repeat_v = Scalar[DTYPE](1)
    var texuniform = False
    var ttype = -1
    if matid >= 0 and matid < MAX_VIS_MATERIALS:
        var mb = matid * VIS_MAT_WORDS
        if rebind[Scalar[DTYPE]](materials[mb + MAT_IDX_ACTIVE]) != 0:
            specular = rebind[Scalar[DTYPE]](materials[mb + MAT_IDX_SPECULAR])
            shininess = rebind[Scalar[DTYPE]](
                materials[mb + MAT_IDX_SHININESS]
            )
            repeat_u = rebind[Scalar[DTYPE]](
                materials[mb + MAT_IDX_TEXREPEAT_U]
            )
            repeat_v = rebind[Scalar[DTYPE]](
                materials[mb + MAT_IDX_TEXREPEAT_V]
            )
            texuniform = (
                rebind[Scalar[DTYPE]](materials[mb + MAT_IDX_TEXUNIFORM]) != 0
            )
            texid = Int(rebind[Scalar[DTYPE]](materials[mb + MAT_IDX_TEXID]))
            if texid >= 0:
                var tb = texid * VIS_TEX_WORDS
                ttype = Int(
                    rebind[Scalar[DTYPE]](textures[tb + TEX_IDX_TYPE])
                )

    var tx = Texel[DTYPE](
        Scalar[DTYPE](1), Scalar[DTYPE](1), Scalar[DTYPE](1), False
    )
    if texid >= 0:
        # ⚠ THE HIT POINT AND THE NORMAL GO BACK INTO THE GEOM'S FRAME, which
        # is where every one of `settexture`'s texgen planes lives.
        var pose = _geom_world_pose[DTYPE](geoms, xpos, xquat, env, g)
        var inv = pose[1].conjugate()
        var lp = inv.rotate_vec(hitpoint - pose[0])
        var ln = inv.rotate_vec(hit.normal)
        # ⚠⚠ THE LEVEL OF DETAIL IS A RASTERISER'S, MEASURED WITH RAYS.
        # OpenGL takes lambda from how far the texture coordinates move
        # between neighbouring pixels. Here the neighbouring pixels' rays are
        # intersected with this hit's TANGENT PLANE — not with the scene, so a
        # silhouette cannot hand a pixel a derivative from a geom behind it —
        # and `geom_uv` is evaluated there with this hit's own normal (so a
        # box keeps its face) and this triangle's own UV map (so a mesh keeps
        # its chart). Pass 0 is the hit itself, passes 1 and 2 the right and
        # lower neighbours: ONE `geom_uv` call site (see `trace_visual`).
        var uv = UV[DTYPE](Scalar[DTYPE](0), Scalar[DTYPE](0))
        var dux = Scalar[DTYPE](0)
        var dvx = Scalar[DTYPE](0)
        var duy = Scalar[DTYPE](0)
        var dvy = Scalar[DTYPE](0)
        var okx = False
        var oky = False
        var nrm = hit.normal
        var plane_d = (hitpoint - ray_org).dot(nrm)
        for pass_ in range(3):
            var lq = lp
            var qbu = hit.bu
            var qbv = hit.bv
            if pass_ > 0:
                var dd = ray_dx if pass_ == 1 else ray_dy
                var den = dd.dot(nrm)
                if abs(den) <= Scalar[DTYPE](1e-12):
                    continue
                var tt = plane_d / den
                if tt <= Scalar[DTYPE](0):
                    continue
                lq = inv.rotate_vec(ray_org + dd * tt - pose[0])
                if hit.tri >= 0:
                    var bq = _tri_bary[DTYPE](mesh_tris, hit.tri, lq)
                    qbu = bq.u
                    qbv = bq.v
            var uq = geom_uv[DTYPE](
                geoms, mesh_uv, g, hit.tri, qbu, qbv,
                lq, ln, ttype, repeat_u, repeat_v, texuniform,
            )
            if pass_ == 0:
                uv = uq
                continue
            var du = uq.u - uv.u
            var dv = uq.v - uv.v
            # ⚠ A CUBE FACE CAN SWITCH BETWEEN TWO PIXELS, and the jump is a
            # face change, not a footprint. Its (s, t) live in [0, 1] per
            # face, so a step over half a face is discarded rather than read
            # as a 12-level minification. Every other mapping here is
            # continuous across the tangent plane and keeps what it gets.
            if ttype == TEX_CUBE and (
                abs(du) > Scalar[DTYPE](0.5) or abs(dv) > Scalar[DTYPE](0.5)
            ):
                continue
            if pass_ == 1:
                dux = du
                dvx = dv
                okx = True
            else:
                duy = du
                dvy = dv
                oky = True
        # A missing axis (grazing, or a discarded face switch) borrows the
        # other one: an isotropic footprint is the better guess than none.
        if okx and not oky:
            duy = dux
            dvy = dvx
        elif oky and not okx:
            dux = duy
            dvx = dvy
        var lod = texture_lod[DTYPE](textures, texid, dux, dvx, duy, dvy)
        tx = sample_texture_lod[DTYPE](
            textures, texels, texid, uv.u, uv.v, lod
        )

    var lit = shade_lights[DTYPE, SHADOWS](
        lights, nlight, geoms, ngeom, bodies, xpos, xquat, env,
        mesh_meta, mesh_tris, hfield_meta, hfield_data, hf_stride,
        hitpoint, hit.normal, eye, gaze, base, specular, shininess,
        Scalar[DTYPE](0),
    )
    # ⚠⚠ `GL_MODULATE` MULTIPLIES THE *LIT* COLOUR, NOT THE MATERIAL COLOUR.
    # `render_gl3.c:699` sets it with the default single-colour specular, so
    # OpenGL lights the vertex colour (highlight included), clamps the sum to
    # [0, 1], and only then multiplies by the texel. Modulating `base` before
    # the lights instead added every textured surface's highlight on top of
    # the texture at full strength: LIBERO's wood table came out ~7% bright
    # and the metal stove base (shininess 1, specular .5) near white.
    if tx.hit:
        lit = Vec3Generic[DTYPE](
            _clamp01[DTYPE](lit.x) * tx.r,
            _clamp01[DTYPE](lit.y) * tx.g,
            _clamp01[DTYPE](lit.z) * tx.b,
        )
    return lit


def trace_visual[
    DTYPE: DType,
    L_GEOMS: Layout,
    L_APP: Layout,
    L_BODIES: Layout,
    L_XPOS: Layout,
    L_XQUAT: Layout,
    L_MESH_META: Layout,
    L_TRI: Layout,
    L_HF_META: Layout,
    L_HF: Layout,
    L_QPOS: Layout,
    L_CULL: Layout,
](
    geoms: LayoutTensor[DTYPE, L_GEOMS, MutAnyOrigin],
    ngeom: Int,
    ncond: Int,
    appearance: LayoutTensor[DTYPE, L_APP, MutAnyOrigin],
    qpos: LayoutTensor[DTYPE, L_QPOS, MutAnyOrigin],
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
    cull: LayoutTensor[DTYPE, L_CULL, MutAnyOrigin],
    cull_base: Int,
    px: Int,
    py: Int,
) -> RayHit[DTYPE] where DTYPE.is_floating_point():
    """`ray_model` over the ordinary geoms, then each CONDITIONAL SITE row
    whose lane-local condition holds (`APP_IDX_COND_QADR`). The nearest wins,
    and a site's hit reports its row index, `>= ngeom`, as the geom.

    ⚠ `cull_base >= 0` SKIPS GEOMS BY SCREEN RECTANGLE: `cull[cull_base +
    4 g ..]` is geom `g`'s `(x0, x1, y0, y1)` for this lane and camera
    (`cull.geom_screen_rect`), and a ray through pixel `(px, py)` — any of its
    samples, all inside the unit square `[px, px+1) x [py, py+1)` — cannot
    reach a geom whose rectangle misses that square. ONLY a camera's primary
    rays may pass it: a reflected or continued ray does not start at the
    camera, and passes `-1` (with any 1-D tensor, never read). The geoms are
    still visited in index order, so a tie is won by the same geom."""
    # ⚠ ONE `ray_model` CALL SITE, IN A LOOP: pass 0 is the ordinary geoms
    # `[0, ngeom)`, pass `k` the one conditional row `ngeom + k - 1`. Mojo
    # inlines every call site of a generic kernel function, and two sites of
    # `ray_model` (mesh BVH, hfield and primitives each) measurably multiplied
    # the Metal compile of the camera kernel.
    var hit = RayHit[DTYPE](
        Scalar[DTYPE](-1), -1, Vec3Generic[DTYPE](0, 0, 0), -1,
        Scalar[DTYPE](0), Scalar[DTYPE](0),
    )
    # ⚠ ONE ROW PER CALL NOW (it was one call over `[0, ngeom)`), so a culled
    # geom can be skipped; the running best goes in as `tcut`, which makes
    # the sequence of calls answer exactly what the single call did.
    var fx = Scalar[DTYPE](px)
    var fy = Scalar[DTYPE](py)
    var one = Scalar[DTYPE](1)
    for k in range(ngeom + ncond):
        var g0 = k
        var g1 = k + 1
        if k < ngeom:
            if cull_base >= 0:
                var o = cull_base + k * 4
                if (
                    fx + one < rebind[Scalar[DTYPE]](cull[o + 0])
                    or fx > rebind[Scalar[DTYPE]](cull[o + 1])
                    or fy + one < rebind[Scalar[DTYPE]](cull[o + 2])
                    or fy > rebind[Scalar[DTYPE]](cull[o + 3])
                ):
                    continue
        else:
            var ab = g0 * VIS_GEOM_APPEARANCE
            var qadr = Int(rebind[Scalar[DTYPE]](appearance[ab + APP_IDX_COND_QADR]))
            if qadr >= 0:
                var qv = rebind[Scalar[DTYPE]](qpos[env, qadr])
                if not (qv >= rebind[Scalar[DTYPE]](appearance[ab + APP_IDX_COND_MIN])):
                    continue
        var h = ray_model[DTYPE](
            geoms, g1, bodies, xpos, xquat, env,
            mesh_meta, mesh_tris, hfield_meta, hfield_data, hf_stride,
            pnt, vec, -1, True, False, 0x3F, g0, hit.t,
        )
        if h.geom >= 0 and (hit.geom < 0 or h.t < hit.t):
            hit = h
    return hit


def render_pixel[
    DTYPE: DType,
    SHADOWS: Bool,
    REFLECT: Bool,
    L_GEOMS: Layout,
    L_APP: Layout,
    L_BODIES: Layout,
    L_XPOS: Layout,
    L_XQUAT: Layout,
    L_MESH_META: Layout,
    L_TRI: Layout,
    L_UV: Layout,
    L_HF_META: Layout,
    L_HF: Layout,
    L_MAT: Layout,
    L_TEX: Layout,
    L_TEXELS: Layout,
    L_LIGHTS: Layout,
    L_QPOS: Layout,
    L_CULL: Layout,
    SAMPLES: Int = 1,
](
    geoms: LayoutTensor[DTYPE, L_GEOMS, MutAnyOrigin],
    ngeom: Int,
    appearance: LayoutTensor[DTYPE, L_APP, MutAnyOrigin],
    bodies: LayoutTensor[DTYPE, L_BODIES, MutAnyOrigin],
    xpos: LayoutTensor[DTYPE, L_XPOS, MutAnyOrigin],
    xquat: LayoutTensor[DTYPE, L_XQUAT, MutAnyOrigin],
    env: Int,
    mesh_meta: LayoutTensor[DTYPE, L_MESH_META, MutAnyOrigin],
    mesh_tris: LayoutTensor[DTYPE, L_TRI, MutAnyOrigin],
    mesh_uv: LayoutTensor[DTYPE, L_UV, MutAnyOrigin],
    hfield_meta: LayoutTensor[DTYPE, L_HF_META, MutAnyOrigin],
    hfield_data: LayoutTensor[DTYPE, L_HF, MutAnyOrigin],
    hf_stride: Int,
    materials: LayoutTensor[DTYPE, L_MAT, MutAnyOrigin],
    textures: LayoutTensor[DTYPE, L_TEX, MutAnyOrigin],
    texels: LayoutTensor[DType.uint8, L_TEXELS, MutAnyOrigin],
    lights: LayoutTensor[DTYPE, L_LIGHTS, MutAnyOrigin],
    nlight: Int,
    qpos: LayoutTensor[DTYPE, L_QPOS, MutAnyOrigin],
    ncond: Int,
    frame: CameraFrame[DTYPE],
    width: Int,
    height: Int,
    px: Int,
    py: Int,
    background: Vec3Generic[DTYPE],
    cull: LayoutTensor[DTYPE, L_CULL, MutAnyOrigin],
    cull_base: Int,
) -> PixelHit[DTYPE] where DTYPE.is_floating_point():
    """One pixel: the centre ray, and with `SAMPLES = 4` OpenGL's 4x MSAA.

    `cull` / `cull_base`: this lane's screen rectangles for this camera, or
    `-1` (any 1-D tensor, never read) — see `trace_visual`.

    ⚠⚠ LIBERO'S PICTURES ARE MULTISAMPLED, AND THAT IS MOST OF THE GAP A
    SINGLE RAY LEAVES. `mjr_makeContext` allocates the offscreen buffer with
    `vis.quality.offsamples` samples — 4 by default, and 4 in every LIBERO
    `model_file` — and robosuite records through it. Rendering LIBERO's own
    model in MuJoCo 3.12 and scoring against the recorded frames: 43.13 dB at
    4 samples, 28.85 dB at none (measured on all 10 `libero_goal` demos). One
    ray per pixel is the second picture, whatever the shading.

    What MSAA does, and what this does: COVERAGE is tested at 4 sub-pixel
    positions, SHADING runs once per primitive at the pixel centre, and the 4
    samples are averaged. So the centre ray is shaded as before (and alone
    supplies `depth`, `geom` and `refl_geom`); each sample ray is only TRACED,
    and reuses the centre colour when it lands on the centre's geom — the
    interior of a surface costs 4 extra traces and no extra shading, and only
    a sample across an edge is shaded itself. Positions are the standard 4x
    rotated grid, (±1/8, ±3/8) and (±3/8, ∓1/8) of a pixel, with `y` flipped
    because row 0 here is the TOP while OpenGL's is the bottom.

    ⚠ ARRAY-FREE ON PURPOSE: the offsets are an if-chain on the sample index,
    not a table — a per-thread array read by a runtime index is the storage
    class Metal has silently miscomputed in this engine.
    """
    # The neighbours' rays, for the texture level of detail only.
    var dir_dx = camera_pixel_ray[DTYPE](frame, width, height, px + 1, py)
    var dir_dy = camera_pixel_ray[DTYPE](frame, width, height, px, py + 1)
    # ⚠ ONE TRACE AND ONE SHADE CALL SITE for the centre and every sample:
    # pass 0 is the centre, passes 1-4 the samples (see `trace_visual` for
    # why a second call site is a compile-time cost, not a style point).
    comptime NS: Int = 0 if SAMPLES <= 1 else 4
    comptime assert SAMPLES == 1 or SAMPLES == 4, (
        "render_pixel: SAMPLES must be 1 or 4"
    )
    var centre_rgb = background
    var centre_depth = Scalar[DTYPE](0)
    var centre_geom = -1
    var centre_refl = -1
    var acc = Vec3Generic[DTYPE](0, 0, 0)
    for k in range(1 + NS):
        var ox = Scalar[DTYPE](0)
        var oy = Scalar[DTYPE](0)
        if k == 1:
            ox = Scalar[DTYPE](-0.125)
            oy = Scalar[DTYPE](0.375)
        elif k == 2:
            ox = Scalar[DTYPE](0.375)
            oy = Scalar[DTYPE](0.125)
        elif k == 3:
            ox = Scalar[DTYPE](-0.375)
            oy = Scalar[DTYPE](-0.125)
        elif k == 4:
            ox = Scalar[DTYPE](0.125)
            oy = Scalar[DTYPE](-0.375)
        var sdir = camera_sample_ray[DTYPE](
            frame, width, height, px, py, ox, oy
        )
        var sh = trace_visual[DTYPE](
            geoms, ngeom, ncond, appearance, qpos, bodies, xpos, xquat, env,
            mesh_meta, mesh_tris, hfield_meta, hfield_data, hf_stride,
            frame.pos, sdir, cull, cull_base, px, py,
        )
        if k > 0 and sh.geom == centre_geom:
            acc = acc + centre_rgb
            continue
        var ph = _shade_ray[DTYPE, SHADOWS, REFLECT](
            geoms, ngeom, appearance, bodies, xpos, xquat, env,
            mesh_meta, mesh_tris, mesh_uv, hfield_meta, hfield_data,
            hf_stride, materials, textures, texels, lights, nlight, qpos,
            ncond, frame, sdir, dir_dx, dir_dy, sh, background,
        )
        if k == 0:
            centre_rgb = ph.rgb
            centre_depth = ph.depth
            centre_geom = ph.geom
            centre_refl = ph.refl_geom
        else:
            acc = acc + ph.rgb
    comptime if NS == 0:
        return PixelHit[DTYPE](centre_rgb, centre_depth, centre_geom, centre_refl)
    else:
        return PixelHit[DTYPE](
            acc * Scalar[DTYPE](0.25), centre_depth, centre_geom, centre_refl
        )


def _shade_surface[
    DTYPE: DType,
    SHADOWS: Bool,
    REFLECT: Bool,
    L_GEOMS: Layout,
    L_APP: Layout,
    L_BODIES: Layout,
    L_XPOS: Layout,
    L_XQUAT: Layout,
    L_MESH_META: Layout,
    L_TRI: Layout,
    L_UV: Layout,
    L_HF_META: Layout,
    L_HF: Layout,
    L_MAT: Layout,
    L_TEX: Layout,
    L_TEXELS: Layout,
    L_LIGHTS: Layout,
    L_QPOS: Layout,
](
    geoms: LayoutTensor[DTYPE, L_GEOMS, MutAnyOrigin],
    ngeom: Int,
    appearance: LayoutTensor[DTYPE, L_APP, MutAnyOrigin],
    bodies: LayoutTensor[DTYPE, L_BODIES, MutAnyOrigin],
    xpos: LayoutTensor[DTYPE, L_XPOS, MutAnyOrigin],
    xquat: LayoutTensor[DTYPE, L_XQUAT, MutAnyOrigin],
    env: Int,
    mesh_meta: LayoutTensor[DTYPE, L_MESH_META, MutAnyOrigin],
    mesh_tris: LayoutTensor[DTYPE, L_TRI, MutAnyOrigin],
    mesh_uv: LayoutTensor[DTYPE, L_UV, MutAnyOrigin],
    hfield_meta: LayoutTensor[DTYPE, L_HF_META, MutAnyOrigin],
    hfield_data: LayoutTensor[DTYPE, L_HF, MutAnyOrigin],
    hf_stride: Int,
    materials: LayoutTensor[DTYPE, L_MAT, MutAnyOrigin],
    textures: LayoutTensor[DTYPE, L_TEX, MutAnyOrigin],
    texels: LayoutTensor[DType.uint8, L_TEXELS, MutAnyOrigin],
    lights: LayoutTensor[DTYPE, L_LIGHTS, MutAnyOrigin],
    nlight: Int,
    qpos: LayoutTensor[DTYPE, L_QPOS, MutAnyOrigin],
    ncond: Int,
    frame: CameraFrame[DTYPE],
    dir: Vec3Generic[DTYPE],
    dir_dx: Vec3Generic[DTYPE],
    dir_dy: Vec3Generic[DTYPE],
    hit: RayHit[DTYPE],
    background: Vec3Generic[DTYPE],
) -> PixelHit[DTYPE] where DTYPE.is_floating_point():
    """One camera ray's colour, given its hit: material, texel, lights, and
    MuJoCo's reflection pass. `dir_dx` / `dir_dy` are the neighbouring
    pixels' rays, for the texture level of detail.

    ⚠ EVERY GEOM IN `geoms` IS DRAWN. The group filter and the alpha-zero
    filter both ran when `VisualModel` was built, so there is no mask here and
    no `flg_static`: this table already IS the set of things a camera sees.
    That is also why the loop is 36 geoms on `libero_goal` rather than 240.

    ⚠ NO `bodyexclude`. A wrist camera SHOULD see the gripper it is mounted
    on — that is most of what it is for. The opposite default from
    `rangefinder_site`, which excludes its own body because MuJoCo's sensor
    does.

    ⚠⚠ THE REFLECTION IS **ADDED**, NOT BLENDED, AND THAT IS THE REFERENCE.
    `mjr_render` draws the mirrored scene first (each geom's rgb scaled by
    `reflectance`), then draws the reflective geom over it with
    `glBlendFunc(GL_ONE, GL_ONE)`. So the mirror's own colour is NOT
    attenuated by `1 - reflectance`; it is added at full strength on top of
    the reflection, and the sum is clamped on write. A lerp — the reflex a
    ray tracer reaches for — is DIMMER than the reference everywhere and is
    not what the recorded pixels show: on the stove it is the difference
    between 75 and 156 out of 255.

    ⚠ AND THE MIRROR SHOWS BLACK WHERE THE REFLECTED RAY MISSES.
    `glClearColor(0, 0, 0, 0)` is what the reflection pass draws over, and the
    skybox pass runs LATER and is depth-tested away inside the mirror. So a
    miss contributes nothing at all — not the background this function
    returns for a primary miss.
    """
    if hit.geom < 0:
        return PixelHit[DTYPE](background, Scalar[DTYPE](0), -1, -1)

    # PLANAR depth. `dir` is normalised and world-space, so the cosine to the
    # optical axis is `dot(dir, -zaxis)` — the same quantity the reference
    # spells `-ray_dir_local_cam[2]` in the camera's own frame.
    var cos_axis = -dir.dot(frame.zaxis)
    var depth = hit.t * cos_axis
    var hitpoint = frame.pos + dir * hit.t
    var gaze = frame.zaxis * Scalar[DTYPE](-1)

    var rgb = shade_hit[DTYPE, SHADOWS](
        geoms, ngeom, appearance, bodies, xpos, xquat, env,
        mesh_meta, mesh_tris, mesh_uv, hfield_meta, hfield_data, hf_stride,
        materials, textures, texels, lights, nlight,
        hit, hitpoint, frame.pos, gaze, Scalar[DTYPE](1),
        frame.pos, dir_dx, dir_dy,
    )

    var seen = -1
    comptime if REFLECT:
        var refl = rebind[Scalar[DTYPE]](
            appearance[hit.geom * VIS_GEOM_APPEARANCE + APP_IDX_REFLECT]
        )
        if refl > Scalar[DTYPE](0):
            var pose = _geom_world_pose[DTYPE](
                geoms, xpos, xquat, env, hit.geom
            )
            # ⚠⚠ THE MIRROR IS THE GEOM'S **+Z FACE**, NOT ITS SURFACE. For a
            # box the reference builds a temporary PLANE and pushes it out to
            # the +Z side (`pos += size[2] * mat[:,2]`); for a plane the geom
            # already is that plane. Reflecting about the surface the ray
            # actually hit would mirror the scene about a side face and put
            # the reflection somewhere else entirely.
            var n = pose[1].rotate_vec(Vec3Generic[DTYPE](0, 0, 1))
            var ppos = pose[0]
            var gtype = Int(
                rebind[Scalar[DTYPE]](geoms[hit.geom, GEOM_IDX_TYPE])
            )
            if gtype == GEOM_BOX:
                ppos = ppos + n * rebind[Scalar[DTYPE]](
                    geoms[hit.geom, GEOM_IDX_HALF_Z]
                )
            # `isBehind` — the camera on the far side of the plane sees the
            # back of the mirror and there is nothing to show.
            var side = (frame.pos - ppos).dot(n)
            var along = dir.dot(n)
            # ⚠⚠ ONLY WHERE THE +Z FACE ITSELF IS VISIBLE. The reference's
            # stencil is that temporary PLANE rendered into the stencil
            # buffer, so it is the +Z face's silhouette and NOT the box's: a
            # box seen from above shows its top face and one or two SIDES, and
            # the sides are outside the stencil and get no reflection at all.
            # Reflecting wherever the primary ray hits the mirror geom fills
            # 100% of the stove where the reference fills 76%, and those extra
            # pixels are its brightest — they are side faces catching a
            # reflection the reference does not draw there.
            var face = hit.normal.dot(n)
            if (
                side >= Scalar[DTYPE](0)
                and along < Scalar[DTYPE](-1e-9)
                and face > Scalar[DTYPE](0.5)
            ):
                # Where the PRIMARY ray crosses the mirror plane. For a hit on
                # the +Z face this is the hit point; for one on a side face it
                # is where the reflected image would have come from, which is
                # what the stencilled mirrored render draws there.
                var tp = side / (-along)
                var org = frame.pos + dir * tp
                var rdir = dir - n * (Scalar[DTYPE](2) * along)
                var rhit = trace_visual[DTYPE](
                    geoms, ngeom, ncond, appearance, qpos, bodies, xpos,
                    xquat, env, mesh_meta, mesh_tris, hfield_meta,
                    hfield_data, hf_stride,
                    org + n * Scalar[DTYPE](1.0e-6), rdir,
                    mesh_meta, -1, 0, 0,
                )
                # `i != j` in the reference's loop: the mirror does not
                # reflect itself.
                if rhit.geom >= 0 and rhit.geom != hit.geom:
                    var rp = org + rdir * rhit.t
                    # ⚠⚠ THE EYE IS THE **MIRRORED** ONE, AND ONLY THE EYE.
                    # `mjr_render` pushes the reflection matrix and THEN calls
                    # `adjustLight`, so in its pass the geometry and every
                    # light are mirrored and the eye is not. Mapping that
                    # whole configuration back through the (isometric)
                    # reflection — which leaves every dot product alone —
                    # gives real geometry, real lights, the camera's own gaze
                    # for the headlight, and the eye reflected. Getting this
                    # wrong moves every specular highlight inside the mirror
                    # and nothing else, which reads as a shading bug rather
                    # than a frame bug.
                    #
                    # The mirrored eye is `tp` back along the reflected ray:
                    # the reflected ray leaves the plane at `org`, and the
                    # plane is equidistant from the eye and its image.
                    seen = rhit.geom
                    var meye = org - rdir * tp
                    # ⚠ AND NO SHADOWS IN THE REFLECTION. The reference's
                    # reflection pass enables the lights plainly and never
                    # binds the shadow map, so a mirrored scene is unshadowed
                    # even when the direct one is not.
                    rgb = rgb + shade_hit[DTYPE, False](
                        geoms, ngeom, appearance, bodies, xpos, xquat, env,
                        mesh_meta, mesh_tris, mesh_uv, hfield_meta,
                        hfield_data, hf_stride, materials, textures, texels,
                        lights, nlight, rhit, rp, meye, gaze, refl,
                        # The mirrored neighbours leave the mirrored eye.
                        meye,
                        dir_dx - n * (Scalar[DTYPE](2) * dir_dx.dot(n)),
                        dir_dy - n * (Scalar[DTYPE](2) * dir_dy.dot(n)),
                    )

    return PixelHit[DTYPE](
        Vec3Generic[DTYPE](
            _clamp01[DTYPE](rgb.x),
            _clamp01[DTYPE](rgb.y),
            _clamp01[DTYPE](rgb.z),
        ),
        depth,
        hit.geom,
        seen,
    )


def _shade_ray[
    DTYPE: DType,
    SHADOWS: Bool,
    REFLECT: Bool,
    L_GEOMS: Layout,
    L_APP: Layout,
    L_BODIES: Layout,
    L_XPOS: Layout,
    L_XQUAT: Layout,
    L_MESH_META: Layout,
    L_TRI: Layout,
    L_UV: Layout,
    L_HF_META: Layout,
    L_HF: Layout,
    L_MAT: Layout,
    L_TEX: Layout,
    L_TEXELS: Layout,
    L_LIGHTS: Layout,
    L_QPOS: Layout,
](
    geoms: LayoutTensor[DTYPE, L_GEOMS, MutAnyOrigin],
    ngeom: Int,
    appearance: LayoutTensor[DTYPE, L_APP, MutAnyOrigin],
    bodies: LayoutTensor[DTYPE, L_BODIES, MutAnyOrigin],
    xpos: LayoutTensor[DTYPE, L_XPOS, MutAnyOrigin],
    xquat: LayoutTensor[DTYPE, L_XQUAT, MutAnyOrigin],
    env: Int,
    mesh_meta: LayoutTensor[DTYPE, L_MESH_META, MutAnyOrigin],
    mesh_tris: LayoutTensor[DTYPE, L_TRI, MutAnyOrigin],
    mesh_uv: LayoutTensor[DTYPE, L_UV, MutAnyOrigin],
    hfield_meta: LayoutTensor[DTYPE, L_HF_META, MutAnyOrigin],
    hfield_data: LayoutTensor[DTYPE, L_HF, MutAnyOrigin],
    hf_stride: Int,
    materials: LayoutTensor[DTYPE, L_MAT, MutAnyOrigin],
    textures: LayoutTensor[DTYPE, L_TEX, MutAnyOrigin],
    texels: LayoutTensor[DType.uint8, L_TEXELS, MutAnyOrigin],
    lights: LayoutTensor[DTYPE, L_LIGHTS, MutAnyOrigin],
    nlight: Int,
    qpos: LayoutTensor[DTYPE, L_QPOS, MutAnyOrigin],
    ncond: Int,
    frame: CameraFrame[DTYPE],
    dir: Vec3Generic[DTYPE],
    dir_dx: Vec3Generic[DTYPE],
    dir_dy: Vec3Generic[DTYPE],
    hit: RayHit[DTYPE],
    background: Vec3Generic[DTYPE],
) -> PixelHit[DTYPE] where DTYPE.is_floating_point():
    """`_shade_surface` for the first hit, composited over what is behind it
    when that surface is TRANSPARENT.

    ⚠⚠ MuJoCo DRAWS EVERY TRANSPARENT GEOM TWICE. `mjr_render` marks a geom
    transparent at `rgba[3] < 0.995`, then renders the transparent list
    front-to-back AND back-to-front with `glBlendFunc(GL_SRC_ALPHA,
    GL_ONE_MINUS_SRC_ALPHA)`, depth writes off, back faces culled
    (`mjRND_CULL_FACE`, on by default). A surface therefore lands with
    opacity `a * (2 - a)`, not `a`: LIBERO's wine-rack stoppers
    (`rgba="0 .345 .545 .1"`) cover 19% of what is behind them, not 10%, and
    drawn opaque they were a bright blue line across the rack.

    Up to three surfaces are composited front to back; the third is taken as
    opaque. Depth, segmentation and the mirror geom stay the FIRST surface's,
    which is what the existing outputs have always meant.
    """
    # ⚠ ONE `_shade_surface` CALL SITE: pass 0 shades the first surface and,
    # when it is opaque, returns — the common case costs what it always did.
    var one = Scalar[DTYPE](1)
    var acc = Vec3Generic[DTYPE](0, 0, 0)
    var keep = one
    var cur = hit
    var first_depth = Scalar[DTYPE](0)
    var first_geom = -1
    var first_refl = -1
    for layer in range(3):
        var ph = _shade_surface[DTYPE, SHADOWS, REFLECT](
            geoms, ngeom, appearance, bodies, xpos, xquat, env,
            mesh_meta, mesh_tris, mesh_uv, hfield_meta, hfield_data,
            hf_stride, materials, textures, texels, lights, nlight, qpos,
            ncond, frame, dir, dir_dx, dir_dy, cur, background,
        )
        if layer == 0:
            first_depth = ph.depth
            first_geom = ph.geom
            first_refl = ph.refl_geom
            if cur.geom < 0:
                return ph^
        var cur_a = rebind[Scalar[DTYPE]](
            appearance[cur.geom * VIS_GEOM_APPEARANCE + APP_IDX_A]
        )
        var opaque = layer == 2 or not (cur_a < Scalar[DTYPE](0.995))
        if opaque:
            if layer == 0:
                return ph^
            acc = acc + ph.rgb * keep
            break
        var eff = cur_a * (Scalar[DTYPE](2) - cur_a)
        if eff < Scalar[DTYPE](0):
            eff = Scalar[DTYPE](0)
        acc = acc + ph.rgb * (keep * eff)
        keep = keep * (one - eff)
        # Behind this surface: the same ray, just past it. ⚠ PAST ITS BACK
        # FACES TOO: the continuation starts INSIDE a closed geom, so the
        # next thing it meets is that geom's own far side, which MuJoCo
        # never draws (`mjRND_CULL_FACE`). A back face is one whose normal
        # points along the ray; a few are stepped over before giving up.
        var tpos = cur.t
        var nxt = RayHit[DTYPE](
            Scalar[DTYPE](-1), -1, Vec3Generic[DTYPE](0, 0, 0), -1,
            Scalar[DTYPE](0), Scalar[DTYPE](0),
        )
        for _ in range(4):
            tpos = tpos + Scalar[DTYPE](1.0e-5)
            nxt = trace_visual[DTYPE](
                geoms, ngeom, ncond, appearance, qpos, bodies, xpos, xquat,
                env, mesh_meta, mesh_tris, hfield_meta, hfield_data,
                hf_stride, frame.pos + dir * tpos, dir,
                mesh_meta, -1, 0, 0,
            )
            if nxt.geom < 0:
                break
            tpos = tpos + nxt.t
            if not (nxt.normal.dot(dir) > Scalar[DTYPE](0)):
                break
            nxt = RayHit[DTYPE](
                Scalar[DTYPE](-1), -1, Vec3Generic[DTYPE](0, 0, 0), -1,
                Scalar[DTYPE](0), Scalar[DTYPE](0),
            )
        if nxt.geom < 0:
            acc = acc + background * keep
            break
        # `t` is measured from the camera from here on.
        cur = RayHit[DTYPE](
            tpos, nxt.geom, nxt.normal, nxt.tri, nxt.bu, nxt.bv,
        )
    return PixelHit[DTYPE](acc, first_depth, first_geom, first_refl)
