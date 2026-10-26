"""Colour at a hit point — MuJoCo's classic renderer, transcribed.

`shade.mojo` was deliberately small and said so: "the COLOUR is not [gateable]"
and "nothing that would only be justified by a pixel comparison that cannot be
run". L5 runs that comparison
(`tools/libero/libero_camera_gate.py`), so the model here is no longer a
plausible ambient-plus-Lambert — it is `render_gl3.c` and the OpenGL fixed
pipeline it drives, term by term, because every term is now measurable.

THE LIGHTING EQUATION, and where each piece comes from
======================================================
`mjr_render` sets `GL_LIGHT_MODEL_LOCAL_VIEWER = 1`, `GL_LIGHT_MODEL_TWO_SIDE
= 0`, a global ambient of 0.3 grey ONLY when no light is supported, and per
light the `ambient` / `diffuse` / `specular` from `mjvLight`. `mjr_makeGeom`
sets `GL_EMISSION = emission * rgba`, `GL_SHININESS = shininess * 128`, and
`GL_SPECULAR = (specular, specular, specular)`; the ambient and diffuse
material colours come from `glColor4fv(rgba)` through `GL_COLOR_MATERIAL`. So,
per light:

    att   = 1 / (a0 + a1*d + a2*d^2)          directional: 1
    spot  = 0 outside the cutoff cone, else (-L . dir)^exponent
    ndotl = max(N . L, 0)
    H     = normalize(L + V)                  V = towards the eye
    spec  = ndotl > 0 ? max(N . H, 0)^(shininess*128) : 0
    c    += att * spot * (ambient*C + ndotl*diffuse*C + spec*specular*s)

and `C` is the geom's rgba modulated by its texel (`GL_MODULATE`).

⚠⚠ THE HEADLIGHT IS A LIGHT AND IT IS NOT IN THE MODEL'S LIGHT LIST.
`mjv_makeLights` puts it FIRST, as a DIRECTIONAL light along the camera's
gaze, with `mjModel.vis.headlight`'s colours — defaults ambient 0.1, diffuse
0.4, specular 0.5. Every MuJoCo picture of a scene that never declares
`<visual><headlight>` is lit mostly by it. Leaving it out is not a subtle
shading difference; it is most of the light in the room.

⚠ `directional="false"` IS A **SPOT**, NOT A POINT LIGHT. MuJoCo's classic
renderer supports `mjLIGHT_DIRECTIONAL` and `mjLIGHT_SPOT` and silently
IGNORES `mjLIGHT_POINT` (`isSupportedLight`). LIBERO's arena lights are
`directional="false"` with the default `cutoff="45"` and `exponent="10"`, so
they are 45-degree cones with a steep angular falloff — reading them as point
lights over-lights the far corners of the room.

⚠ NO SHADOW IS CAST IN A LIBERO FRAME, and that is the model's own doing:
both arena lights carry `castshadow="false"` and a headlight never casts. The
shadow ray is still here, still `ray_model` against the same geometry, and
still a comptime parameter — it is simply not reached on this scene.

TEXTURE COORDINATES, and the four cases `settexture` actually has
==================================================================
1. **A mesh with texcoords** — the UVs interpolate over the triangle with the
   barycentrics `RayHit` now carries, then scale by `texrepeat` (times the
   geom's size when `texuniform`).
2. **A builtin shape with a 2D texture** — `isBuiltinWithUV` is true for
   plane, sphere, ellipsoid, box, cylinder, capsule, so these use the UVs
   baked into MuJoCo's display lists rather than texgen. Those UVs are
   reproduced here in closed form: a plane is `(x+sx)/2sx, 1-(y+sy)/2sy`; a
   box is the [0,1] square of whichever face the normal picks, `v` flipped.
3. **A cube texture** — object-linear texgen `(s,t,r) = k*(unit shape
   coords)`, then the standard cube-map face select. ⚠ THE OBJECT COORDINATES
   ARE THE UNIT SHAPE'S, NOT METRES: `mjr_makeGeom` puts the size in the
   MODELVIEW (`glScalef(size)`) and texgen reads what `glVertex` was handed,
   which for every builtin is the ±1 shape. So the local hit point is divided
   by the geom's own size first. A mesh is the exception — it is drawn at its
   real coordinates and is not scaled — which is exactly why the 2D branch
   divides `scl` by the size when `dataid >= 0`.
4. **Anything else with a 2D texture** — the object-linear fallback
   `s = 0.5*scl0*x - 0.5`, `t = -0.5*scl1*y - 0.5`.

⚠⚠ TRILINEAR, WITH MUJOCO'S MIP CHAIN. `render_context.c` calls
`glGenerateMipmap` on every texture and samples `GL_LINEAR_MIPMAP_LINEAR`, so
at 128x128 a 4096-texel plate map is averaged over the hundreds of texels a
pixel covers. Sampling level 0 only drew the plate's rim pattern and the
bowl's glaze sharp and sparkling where the reference is smooth — most of the
camera gate's remaining error in 2026-09. The level is OpenGL's own lambda:
`render.shade_hit` gets the derivatives by intersecting the NEIGHBOURING
pixels' rays with the hit's tangent plane and evaluating `geom_uv` there,
which is what a rasteriser's 2x2 finite difference measures
(`texture_lod`, `sample_texture_lod`).

⚠ WRAP IS REPEAT, which is MuJoCo's default for a 2D texture
(`GL_REPEAT`)... except that `mjr_uploadTexture` sets `GL_CLAMP_TO_EDGE` for
a CUBE face. Both are here and dispatched on the texture's type.
"""

from std.math import floor, sqrt, exp, log, cos

from layout import TileTensor, TensorLayout

from noeira.math3d import Vec3 as Vec3Generic, Quat as QuatGeneric

from ..constants import (
    GEOM_PLANE,
    GEOM_SPHERE,
    GEOM_CAPSULE,
    GEOM_BOX,
    GEOM_CYLINDER,
    GEOM_ELLIPSOID,
    GEOM_MESH,
)
from ..gpu.constants import (
    MODEL_GEOM_SIZE,
    GEOM_IDX_TYPE,
    GEOM_IDX_RADIUS,
    GEOM_IDX_HALF_LENGTH,
    GEOM_IDX_HALF_X,
    GEOM_IDX_HALF_Y,
    GEOM_IDX_HALF_Z,
)
from ..parser.flat_model import TEX_2D, TEX_CUBE
from ..ray.model import ray_model
from .visual_records import *


@always_inline
def _clamp01[
    DTYPE: DType
](x: Scalar[DTYPE]) -> Scalar[DTYPE] where DTYPE.is_floating_point():
    if x < Scalar[DTYPE](0):
        return Scalar[DTYPE](0)
    if x > Scalar[DTYPE](1):
        return Scalar[DTYPE](1)
    return x


@always_inline
def _wrap01[
    DTYPE: DType
](x: Scalar[DTYPE]) -> Scalar[DTYPE] where DTYPE.is_floating_point():
    """`GL_REPEAT`. `x - floor(x)`, and the guard is not decoration: at
    `x = -1e-9` the subtraction can round to exactly 1.0, which indexes one
    texel past the last column."""
    var f = x - floor(x)
    if f < Scalar[DTYPE](0):
        f = Scalar[DTYPE](0)
    if f >= Scalar[DTYPE](1):
        f = Scalar[DTYPE](0)
    return f


@always_inline
def _powi[
    DTYPE: DType
](x: Scalar[DTYPE], e: Scalar[DTYPE]) -> Scalar[DTYPE] where (
    DTYPE.is_floating_point()
):
    """`x^e` for `x >= 0`, as `exp(e*log(x))`.

    ⚠ `pow` ON A GENERIC `Scalar[DTYPE]` IS NOT A DEVICE SYMBOL in this tree
    the way `exp` and `log` are — the same wall `atan2` hit
    (`_the_stdlibs_atan2_is_a_libm_symbol_on_the_device`). Both operands here
    are bounded: `x` is a clamped cosine and `e` is `shininess*128` or a spot
    exponent, so the identity is safe everywhere it is reached.
    """
    if x <= Scalar[DTYPE](0):
        return Scalar[DTYPE](0)
    if e == Scalar[DTYPE](0):
        return Scalar[DTYPE](1)
    return exp(e * log(x))


@fieldwise_init
struct Texel[DTYPE: DType](Copyable, ImplicitlyCopyable, Movable):
    var r: Scalar[Self.DTYPE]
    var g: Scalar[Self.DTYPE]
    var b: Scalar[Self.DTYPE]
    var hit: Bool


@always_inline
def _bilinear_level[
    DTYPE: DType, L_TEXELS: TensorLayout
](
    texels: TileTensor[DType.uint8, L_TEXELS, MutAnyOrigin],
    adr: Int,
    w: Int,
    h: Int,
    ttype: Int,
    uu: Scalar[DTYPE],
    vv: Scalar[DTYPE],
) -> Texel[DTYPE] where DTYPE.is_floating_point():
    """`GL_LINEAR` on ONE level of `w` x `h` texels starting at `adr`. `uu`,
    `vv` are already wrapped or clamped to [0, 1]."""
    # Half-texel offsets: a sample at u = 0 sits at the CENTRE of texel 0, not
    # on its left edge. Getting this wrong shifts the whole image by half a
    # texel, which is invisible on a 4096 map and obvious on a 1x1 builtin.
    var fx = uu * Scalar[DTYPE](w) - Scalar[DTYPE](0.5)
    var fy = vv * Scalar[DTYPE](h) - Scalar[DTYPE](0.5)
    var x0 = Int(floor(fx))
    var y0 = Int(floor(fy))
    var ax = fx - Scalar[DTYPE](x0)
    var ay = fy - Scalar[DTYPE](y0)

    var r = Scalar[DTYPE](0)
    var g = Scalar[DTYPE](0)
    var b = Scalar[DTYPE](0)
    var inv = Scalar[DTYPE](1.0 / 255.0)
    for j in range(2):
        for i in range(2):
            var xi = x0 + i
            var yi = y0 + j
            if ttype == TEX_CUBE:
                xi = 0 if xi < 0 else (w - 1 if xi > w - 1 else xi)
                yi = 0 if yi < 0 else (h - 1 if yi > h - 1 else yi)
            else:
                xi = xi % w
                if xi < 0:
                    xi += w
                yi = yi % h
                if yi < 0:
                    yi += h
            var wgt = (ax if i == 1 else Scalar[DTYPE](1) - ax) * (
                ay if j == 1 else Scalar[DTYPE](1) - ay
            )
            var o = adr + (yi * w + xi) * 3
            r += wgt * Scalar[DTYPE](Int(texels[o + 0])) * inv
            g += wgt * Scalar[DTYPE](Int(texels[o + 1])) * inv
            b += wgt * Scalar[DTYPE](Int(texels[o + 2])) * inv
    return Texel[DTYPE](r, g, b, True)


@always_inline
def mip_level_adr(adr: Int, w: Int, h: Int, level: Int) -> Int:
    """Atlas offset of mip `level` — see `TEX_IDX_NLEVELS` for the layout.
    A loop of at most `log2(max(w, h))` steps, integer only, kernel-safe."""
    var o = adr
    var lw = w
    var lh = h
    for _ in range(level):
        o += lw * lh * 3
        lw = lw >> 1 if lw > 1 else 1
        lh = lh >> 1 if lh > 1 else 1
    return o


def sample_texture_lod[
    DTYPE: DType, L_TEX: TensorLayout, L_TEXELS: TensorLayout
](
    textures: TileTensor[DTYPE, L_TEX, MutAnyOrigin],
    texels: TileTensor[DType.uint8, L_TEXELS, MutAnyOrigin],
    texid: Int,
    u: Scalar[DTYPE],
    v: Scalar[DTYPE],
    lod: Scalar[DTYPE],
) -> Texel[DTYPE] where DTYPE.is_floating_point():
    """`GL_LINEAR_MIPMAP_LINEAR` RGB from the atlas, `[0, 1]`, at level of
    detail `lod` (OpenGL's lambda, `texture_lod`).

    The GL 2.1 rules (§3.8.8-3.8.9), with MuJoCo's filters: the magnification
    filter is `GL_LINEAR` and the minification one `GL_LINEAR_MIPMAP_LINEAR`,
    so the switch-over constant `c` is 0 — `lod <= 0` is plain bilinear on
    level 0, and above it the two levels `floor(lod)` and `floor(lod) + 1`
    (clamped to the last one) are each sampled bilinearly and blended by the
    fraction.

    ⚠ ROW 0 IS `v = 0`. The PNG decodes top row first and the atlas keeps that
    order, and MuJoCo's own UVs are written with `v` already flipped
    (`1 - (y+sy)/2sy` for a plane) — so no flip belongs here. A flip in this
    function would be a second one, and the two would cancel on a plane and
    NOT on a mesh.
    """
    var miss = Texel[DTYPE](
        Scalar[DTYPE](1), Scalar[DTYPE](1), Scalar[DTYPE](1), False
    )
    if texid < 0 or texid >= MAX_VIS_TEXTURES:
        return miss
    var tb = texid * VIS_TEX_WORDS
    if rebind[Scalar[DTYPE]](textures[tb + TEX_IDX_ACTIVE]) == 0:
        return miss
    var w = Int(rebind[Scalar[DTYPE]](textures[tb + TEX_IDX_WIDTH]))
    var h = Int(rebind[Scalar[DTYPE]](textures[tb + TEX_IDX_HEIGHT]))
    var adr = Int(rebind[Scalar[DTYPE]](textures[tb + TEX_IDX_ADR]))
    var ttype = Int(rebind[Scalar[DTYPE]](textures[tb + TEX_IDX_TYPE]))
    var nlevels = Int(rebind[Scalar[DTYPE]](textures[tb + TEX_IDX_NLEVELS]))
    if w <= 0 or h <= 0:
        return miss
    if nlevels < 1:
        nlevels = 1

    var uu = u
    var vv = v
    if ttype == TEX_CUBE:
        # `mjr_uploadTexture` clamps a cube face; a repeat would show the
        # opposite edge of the wood grain across the table's silhouette.
        uu = _clamp01[DTYPE](uu)
        vv = _clamp01[DTYPE](vv)
    else:
        uu = _wrap01[DTYPE](uu)
        vv = _wrap01[DTYPE](vv)

    var q = nlevels - 1
    if not (lod > Scalar[DTYPE](0)) or q == 0:
        return _bilinear_level[DTYPE](texels, adr, w, h, ttype, uu, vv)
    var d1 = Int(floor(lod))
    var frac = lod - Scalar[DTYPE](d1)
    if d1 >= q:
        d1 = q
        frac = Scalar[DTYPE](0)
    var w1 = w >> d1 if (w >> d1) > 0 else 1
    var h1 = h >> d1 if (h >> d1) > 0 else 1
    var a1 = mip_level_adr(adr, w, h, d1)
    var t1 = _bilinear_level[DTYPE](texels, a1, w1, h1, ttype, uu, vv)
    if frac <= Scalar[DTYPE](0):
        return t1
    var d2 = d1 + 1
    var w2 = w >> d2 if (w >> d2) > 0 else 1
    var h2 = h >> d2 if (h >> d2) > 0 else 1
    var a2 = a1 + w1 * h1 * 3
    var t2 = _bilinear_level[DTYPE](texels, a2, w2, h2, ttype, uu, vv)
    var k = Scalar[DTYPE](1) - frac
    return Texel[DTYPE](
        t1.r * k + t2.r * frac,
        t1.g * k + t2.g * frac,
        t1.b * k + t2.b * frac,
        True,
    )


def sample_texture[
    DTYPE: DType, L_TEX: TensorLayout, L_TEXELS: TensorLayout
](
    textures: TileTensor[DTYPE, L_TEX, MutAnyOrigin],
    texels: TileTensor[DType.uint8, L_TEXELS, MutAnyOrigin],
    texid: Int,
    u: Scalar[DTYPE],
    v: Scalar[DTYPE],
) -> Texel[DTYPE] where DTYPE.is_floating_point():
    """Bilinear on level 0 — `sample_texture_lod` at `lod = 0`."""
    return sample_texture_lod[DTYPE](
        textures, texels, texid, u, v, Scalar[DTYPE](0)
    )


def texture_lod[
    DTYPE: DType, L_TEX: TensorLayout
](
    textures: TileTensor[DTYPE, L_TEX, MutAnyOrigin],
    texid: Int,
    dudx: Scalar[DTYPE],
    dvdx: Scalar[DTYPE],
    dudy: Scalar[DTYPE],
    dvdy: Scalar[DTYPE],
) -> Scalar[DTYPE] where DTYPE.is_floating_point():
    """OpenGL's lambda from the texture-coordinate derivatives across one
    pixel, in [0, 1] units: `log2(max(|d(uv)/dx|, |d(uv)/dy|))` with `u`
    scaled by the width and `v` by the height (GL 2.1 §3.8.8, the isotropic
    `rho` every driver implements). Returns 0 for a texture that is not there.
    """
    if texid < 0 or texid >= MAX_VIS_TEXTURES:
        return Scalar[DTYPE](0)
    var tb = texid * VIS_TEX_WORDS
    var w = rebind[Scalar[DTYPE]](textures[tb + TEX_IDX_WIDTH])
    var h = rebind[Scalar[DTYPE]](textures[tb + TEX_IDX_HEIGHT])
    var ax = dudx * w
    var bx = dvdx * h
    var ay = dudy * w
    var by = dvdy * h
    var rx = ax * ax + bx * bx
    var ry = ay * ay + by * by
    var rho2 = rx if rx > ry else ry
    if not (rho2 > Scalar[DTYPE](1)):
        return Scalar[DTYPE](0)
    # log2(sqrt(rho2)) = 0.5 * ln(rho2) / ln(2). `log` lowers on the device
    # (`_powi` relies on the same); `log2` is not assumed to.
    return Scalar[DTYPE](0.5) * log(rho2) / Scalar[DTYPE](0.6931471805599453)


@fieldwise_init
struct UV[DTYPE: DType](Copyable, ImplicitlyCopyable, Movable):
    var u: Scalar[Self.DTYPE]
    var v: Scalar[Self.DTYPE]


def geom_uv[
    DTYPE: DType, L_GEOMS: TensorLayout, L_UV: TensorLayout
](
    geoms: TileTensor[DTYPE, L_GEOMS, MutAnyOrigin],
    mesh_uv: TileTensor[DTYPE, L_UV, MutAnyOrigin],
    g: Int,
    tri: Int,
    bu: Scalar[DTYPE],
    bv: Scalar[DTYPE],
    lp: Vec3Generic[DTYPE],
    ln: Vec3Generic[DTYPE],
    ttype: Int,
    repeat_u: Scalar[DTYPE],
    repeat_v: Scalar[DTYPE],
    texuniform: Bool,
) -> UV[DTYPE] where DTYPE.is_floating_point():
    """`settexture`'s four cases, at a hit point in the geom's LOCAL frame.

    `lp` is the hit point and `ln` the surface normal, both local. `tri` is
    the arena record of the triangle that was hit, or -1.
    """
    var gtype = Int(rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_TYPE]))
    var hx = rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_HALF_X])
    var hy = rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_HALF_Y])
    var hz = rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_HALF_Z])
    var rad = rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_RADIUS])
    var hl = rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_HALF_LENGTH])

    # The three "sizes" `glScalef` would have applied, per type. A mesh gets
    # ones because MuJoCo does not scale a mesh display list.
    var sx = hx
    var sy = hy
    var sz = hz
    if gtype == GEOM_SPHERE:
        sx = rad
        sy = rad
        sz = rad
    elif gtype == GEOM_CAPSULE or gtype == GEOM_CYLINDER:
        sx = rad
        sy = rad
        sz = hl
    elif gtype == GEOM_MESH:
        sx = Scalar[DTYPE](1)
        sy = Scalar[DTYPE](1)
        sz = Scalar[DTYPE](1)
    var one = Scalar[DTYPE](1)
    var tiny = Scalar[DTYPE](1e-12)
    var ux = one / (sx if sx > tiny else one)
    var uy = one / (sy if sy > tiny else one)
    var uz = one / (sz if sz > tiny else one)

    # ── the cube branch: object-linear texgen + the cube-map face select ──
    if ttype == TEX_CUBE:
        var kx = one
        var ky = one
        var kz = one
        if texuniform:
            kx = sx
            ky = sy
            kz = sz
        var s = kx * lp.x * ux
        var t = ky * lp.y * uy
        var r = kz * lp.z * uz
        var asx = abs(s)
        var asy = abs(t)
        var asz = abs(r)
        var sc: Scalar[DTYPE]
        var tc: Scalar[DTYPE]
        var ma: Scalar[DTYPE]
        if asx >= asy and asx >= asz:
            sc = -r if s > 0 else r
            tc = -t
            ma = asx
        elif asy >= asx and asy >= asz:
            sc = s
            tc = r if t > 0 else -r
            ma = asy
        else:
            sc = s if r > 0 else -s
            tc = -t
            ma = asz
        if ma < tiny:
            ma = tiny
        return UV[DTYPE](
            Scalar[DTYPE](0.5) * (sc / ma + one),
            Scalar[DTYPE](0.5) * (tc / ma + one),
        )

    # ── the scale the texture matrix applies to an explicit UV ────────────
    var scl_u = repeat_u if repeat_u > 0 else one
    var scl_v = repeat_v if repeat_v > 0 else one
    if texuniform:
        if sx > 0:
            scl_u = scl_u * sx
        if sy > 0:
            scl_v = scl_v * sy

    # 1. a mesh triangle with texcoords
    if gtype == GEOM_MESH and tri >= 0:
        var o = tri * VIS_UV_WORDS
        var w2 = one - bu - bv
        var u = (
            bu * rebind[Scalar[DTYPE]](mesh_uv[o + 0])
            + bv * rebind[Scalar[DTYPE]](mesh_uv[o + 2])
            + w2 * rebind[Scalar[DTYPE]](mesh_uv[o + 4])
        )
        var v = (
            bu * rebind[Scalar[DTYPE]](mesh_uv[o + 1])
            + bv * rebind[Scalar[DTYPE]](mesh_uv[o + 3])
            + w2 * rebind[Scalar[DTYPE]](mesh_uv[o + 5])
        )
        return UV[DTYPE](u * scl_u, v * scl_v)

    # 2a. the plane's own UVs — `makePlane`, for a plane with a finite size
    if gtype == GEOM_PLANE:
        var u = Scalar[DTYPE](0.5) * lp.x
        var v = -Scalar[DTYPE](0.5) * lp.y
        if hx > tiny:
            u = (lp.x + hx) / (Scalar[DTYPE](2) * hx)
        if hy > tiny:
            v = one - (lp.y + hy) / (Scalar[DTYPE](2) * hy)
        return UV[DTYPE](u * scl_u, v * scl_v)

    # 2b. the box's own UVs — `makeBuiltin`'s `mjrBOX`, per face, `v` flipped
    if gtype == GEOM_BOX:
        var nx = abs(ln.x)
        var ny = abs(ln.y)
        var nz = abs(ln.z)
        # ⚠ THE FACE IS PICKED BY THE NORMAL, NOT BY THE POSITION. On an edge
        # pixel the largest |coordinate| and the largest |normal component|
        # disagree, and the position would wrap the neighbouring face's
        # texture around the corner.
        var a: Scalar[DTYPE]
        var b: Scalar[DTYPE]
        if nz >= nx and nz >= ny:
            a = lp.x * ux
            b = lp.y * uy
        elif nx >= ny:
            a = lp.y * uy
            b = lp.z * uz
        else:
            a = lp.x * ux
            b = lp.z * uz
        return UV[DTYPE](
            Scalar[DTYPE](0.5) * (a + one) * scl_u,
            (one - Scalar[DTYPE](0.5) * (b + one)) * scl_v,
        )

    # 4. everything else: the object-linear fallback. ⚠ A SPHERE, AN
    # ELLIPSOID, A CAPSULE AND A CYLINDER LAND HERE AND MUJOCO WOULD USE
    # THEIR BUILT-IN UVs. No LIBERO surface is one of those with a 2D
    # texture; the day one is, the shape's display-list UVs go above this
    # line and this stays what it says it is — a fallback, not the rule.
    return UV[DTYPE](
        Scalar[DTYPE](0.5) * scl_u * lp.x * ux - Scalar[DTYPE](0.5),
        -Scalar[DTYPE](0.5) * scl_v * lp.y * uy - Scalar[DTYPE](0.5),
    )


def shade_lights[
    DTYPE: DType,
    SHADOWS: Bool,
    L_LIGHTS: TensorLayout,
    L_GEOMS: TensorLayout,
    L_BODIES: TensorLayout,
    L_XPOS: TensorLayout,
    L_XQUAT: TensorLayout,
    L_MESH_META: TensorLayout,
    L_TRI: TensorLayout,
    L_HF_META: TensorLayout,
    L_HF: TensorLayout,
](
    lights: TileTensor[DTYPE, L_LIGHTS, MutAnyOrigin],
    nlight: Int,
    geoms: TileTensor[DTYPE, L_GEOMS, MutAnyOrigin],
    ngeom: Int,
    bodies: TileTensor[DTYPE, L_BODIES, MutAnyOrigin],
    xpos: TileTensor[DTYPE, L_XPOS, MutAnyOrigin],
    xquat: TileTensor[DTYPE, L_XQUAT, MutAnyOrigin],
    env: Int,
    mesh_meta: TileTensor[DTYPE, L_MESH_META, MutAnyOrigin],
    mesh_tris: TileTensor[DTYPE, L_TRI, MutAnyOrigin],
    hfield_meta: TileTensor[DTYPE, L_HF_META, MutAnyOrigin],
    hfield_data: TileTensor[DTYPE, L_HF, MutAnyOrigin],
    hf_stride: Int,
    point: Vec3Generic[DTYPE],
    normal: Vec3Generic[DTYPE],
    eye: Vec3Generic[DTYPE],
    gaze: Vec3Generic[DTYPE],
    base: Vec3Generic[DTYPE],
    specular: Scalar[DTYPE],
    shininess: Scalar[DTYPE],
    emission: Scalar[DTYPE],
) -> Vec3Generic[DTYPE] where DTYPE.is_floating_point():
    """The OpenGL fixed-pipeline sum over the headlight and the model's lights.

    `base` is the material colour AFTER the texel modulates it — the `C` of
    the module header. `gaze` is the camera's forward direction, which is the
    headlight's `dir`.

    ⚠⚠ THE HEADLIGHT IS A ROW OF THE LIGHT TABLE, NOT AN ARGUMENT, and that
    is not tidiness — it is Metal's argument table. Six more scalars for its
    three colours and its flag would have put this kernel at 33 operands
    against a ceiling where 29 is a SILENT metallib failure and 27 ships
    (`_metals_limit_is_the_argument_table_not_the_stack`). It is also what
    `mjv_makeLights` does: the headlight is `scn->lights[0]`, a directional
    light like any other. `LIGHT_IDX_BODY == LIGHT_BODY_HEADLIGHT` marks it,
    and its direction is taken from `gaze` rather than from its own row —
    a camera-relative light cannot be baked into a model table.

    ⚠ THE GLOBAL AMBIENT IS 0 WHENEVER ANY LIGHT IS SUPPORTED, and 0.3 grey
    when none is (`mjr_render`: `float global = nsupported ? 0 : 0.3f`). With
    the headlight active that first case is always the one taken, which is why
    a scene with `<light>` elements and one without do not differ by a
    constant lift.
    """
    var zero = Vec3Generic[DTYPE](0, 0, 0)
    var n = normal
    var nl = n.length()
    if nl > Scalar[DTYPE](0):
        n = n / nl
    var view = eye - point
    var vl = view.length()
    if vl > Scalar[DTYPE](0):
        view = view / vl
    else:
        view = -gaze

    var acc = base * emission
    # `mjr_render`: "create some ambient light if no supported lights are
    # present" — `float global = nsupported ? 0 : 0.3f`. ⚠ COUNTED BY
    # `ACTIVE`, not by rows: the headlight's row is always written (index 0),
    # so `<headlight active="0">` on a model with no `<light>` has one row
    # and no light, and MuJoCo gives it the 0.3.
    var nactive = 0
    for li in range(nlight):
        if rebind[Scalar[DTYPE]](lights[li * VIS_LIGHT_WORDS + LIGHT_IDX_ACTIVE]) != 0:
            nactive += 1
    if nactive == 0:
        acc = acc + base * Scalar[DTYPE](0.3)

    var shine = shininess * Scalar[DTYPE](128)
    for li in range(nlight):
        var lb = li * VIS_LIGHT_WORDS
        if rebind[Scalar[DTYPE]](lights[lb + LIGHT_IDX_ACTIVE]) == 0:
            continue
        var lamb = Vec3Generic[DTYPE](
            rebind[Scalar[DTYPE]](lights[lb + LIGHT_IDX_AMBIENT_R]),
            rebind[Scalar[DTYPE]](lights[lb + LIGHT_IDX_AMBIENT_G]),
            rebind[Scalar[DTYPE]](lights[lb + LIGHT_IDX_AMBIENT_B]),
        )
        var ldif = Vec3Generic[DTYPE](
            rebind[Scalar[DTYPE]](lights[lb + LIGHT_IDX_DIFFUSE_R]),
            rebind[Scalar[DTYPE]](lights[lb + LIGHT_IDX_DIFFUSE_G]),
            rebind[Scalar[DTYPE]](lights[lb + LIGHT_IDX_DIFFUSE_B]),
        )
        var lspe = Vec3Generic[DTYPE](
            rebind[Scalar[DTYPE]](lights[lb + LIGHT_IDX_SPECULAR_R]),
            rebind[Scalar[DTYPE]](lights[lb + LIGHT_IDX_SPECULAR_G]),
            rebind[Scalar[DTYPE]](lights[lb + LIGHT_IDX_SPECULAR_B]),
        )
        var casts = (
            rebind[Scalar[DTYPE]](lights[lb + LIGHT_IDX_CASTSHADOW]) != 0
        )
        var is_head = (
            rebind[Scalar[DTYPE]](lights[lb + LIGHT_IDX_BODY])
            == Scalar[DTYPE](LIGHT_BODY_HEADLIGHT)
        )
        var directional = (
            is_head
            or rebind[Scalar[DTYPE]](lights[lb + LIGHT_IDX_DIRECTIONAL]) != 0
        )

        var sd = Vec3Generic[DTYPE](
            rebind[Scalar[DTYPE]](lights[lb + LIGHT_IDX_DIR_X]),
            rebind[Scalar[DTYPE]](lights[lb + LIGHT_IDX_DIR_Y]),
            rebind[Scalar[DTYPE]](lights[lb + LIGHT_IDX_DIR_Z]),
        )
        if is_head:
            sd = gaze
        var sdl = sd.length()
        if sdl <= Scalar[DTYPE](0):
            continue
        sd = sd / sdl

        var ldir = zero
        var att = Scalar[DTYPE](1)
        var spot = Scalar[DTYPE](1)
        if directional:
            # `adjustLight`: a directional light's GL position is `-dir`, so
            # the vector TOWARDS it is the negated travel direction.
            ldir = -sd
        else:
            var lp = Vec3Generic[DTYPE](
                rebind[Scalar[DTYPE]](lights[lb + LIGHT_IDX_POS_X]),
                rebind[Scalar[DTYPE]](lights[lb + LIGHT_IDX_POS_Y]),
                rebind[Scalar[DTYPE]](lights[lb + LIGHT_IDX_POS_Z]),
            )
            var to = lp - point
            var dist = to.length()
            if dist <= Scalar[DTYPE](0):
                continue
            ldir = to / dist
            # ⚠ ATTENUATION IS (1, 0, 0) ON EVERY LIGHT IN THIS CORPUS —
            # MuJoCo's own default — so `att` stays 1 and no `<light
            # attenuation>` is carried into the table. The day one is, it
            # arrives as three more words in the row and is read here.
            var c = -ldir.dot(sd)
            var cutoff = rebind[Scalar[DTYPE]](lights[lb + LIGHT_IDX_CUTOFF])
            # `GL_SPOT_CUTOFF` is in DEGREES; 180 means "no cone at all".
            if cutoff >= Scalar[DTYPE](180):
                spot = Scalar[DTYPE](1)
            else:
                var ccut = cos(
                    cutoff * Scalar[DTYPE](3.14159265358979323846 / 180.0)
                )
                if c < ccut or c <= Scalar[DTYPE](0):
                    continue
                # `GL_SPOT_EXPONENT` — `light_exponent`, default 10. A
                # constant rather than a column for the same reason
                # attenuation is: nothing in the corpus varies it, and a
                # column nobody writes is a column that goes stale.
                spot = _powi[DTYPE](c, Scalar[DTYPE](10))

        var ndotl = n.dot(ldir)
        if ndotl < Scalar[DTYPE](0):
            ndotl = Scalar[DTYPE](0)

        var vis = Scalar[DTYPE](1)

        comptime if SHADOWS:
            if casts and ndotl > Scalar[DTYPE](0):
                var origin = point + n * Scalar[DTYPE](1.0e-4)
                var hit = ray_model[DTYPE](
                    geoms, ngeom, bodies, xpos, xquat, env,
                    mesh_meta, mesh_tris, hfield_meta, hfield_data,
                    hf_stride, origin, ldir,
                )
                if hit.geom >= 0:
                    # The reference's shadow pass darkens rather than blacks
                    # out; 0.3 is the factor the first `shade.mojo` carried
                    # and it is kept.
                    vis = Scalar[DTYPE](0.3)

        var k = att * spot
        acc = acc + Vec3Generic[DTYPE](
            k * lamb.x * base.x, k * lamb.y * base.y, k * lamb.z * base.z
        )
        if ndotl > Scalar[DTYPE](0):
            var kd = k * ndotl * vis
            acc = acc + Vec3Generic[DTYPE](
                kd * ldif.x * base.x,
                kd * ldif.y * base.y,
                kd * ldif.z * base.z,
            )
            # ⚠ THE SPECULAR TERM IS GATED ON `N . L > 0`, which is the GL
            # spec's own `f` factor and not an optimisation: without it a
            # surface facing AWAY from a light still catches its highlight.
            if specular > Scalar[DTYPE](0):
                var hv = ldir + view
                var hl2 = hv.length()
                if hl2 > Scalar[DTYPE](0):
                    hv = hv / hl2
                    var ndoth = n.dot(hv)
                    if ndoth > Scalar[DTYPE](0):
                        var sf = (
                            k * vis * specular * _powi[DTYPE](ndoth, shine)
                        )
                        acc = acc + Vec3Generic[DTYPE](
                            sf * lspe.x, sf * lspe.y, sf * lspe.z
                        )
    return acc
