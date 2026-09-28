"""The tracer's appearance path, on cases whose answer is arithmetic.

`examples/libero/libero_camera_gate.mojo` is L5's real gate and it needs ~6 GB
of demonstrations. This file needs nothing: it drives `sample_texture`,
`geom_uv` and `shade_lights` on hand-built tables whose answer is a closed
form, so a clone with no assets still catches a broken sampler.

⚠⚠ IT IS NOT A SUBSTITUTE FOR THE GATE AND MUST NOT BE READ AS ONE. Every
expected value here is one I derived from the same reading of
`render_gl3.c`/`render_context.c` that the code was written from, so the two
agree by construction — the blind shape this tree keeps paying for
(`_a_gate_that_shares_its_reference_implementation_is_blind`). What it defends
is the ARITHMETIC: a half-texel offset, a wrap that lands one column over, a
box face picked by position instead of by normal, a specular term that fires
on a surface facing away from the light. The camera gate is what says the
MODEL is right.

⚠ ANTI-VACUITY IS EXPLICIT. Several checks below would pass on a function
that returned a constant, so each is paired with a second call that must
differ — the "0 mismatches means nothing was tested" failure this tree treats
as the default one.
"""

from std.math import sqrt

from layout import Layout, LayoutTensor

from noeira.math3d import Vec3
from noeira.nn.core.tensor import TensorImpl
from noeira.physics3d.constants import GEOM_BOX, GEOM_PLANE, GEOM_MESH
from noeira.physics3d.fields.rt_layout import DYN1, DYN2, rl1, rl2
from noeira.physics3d.gpu.constants import (
    MODEL_GEOM_SIZE,
    GEOM_IDX_TYPE,
    GEOM_IDX_HALF_X,
    GEOM_IDX_HALF_Y,
    GEOM_IDX_HALF_Z,
    GEOM_IDX_RADIUS,
    GEOM_IDX_HALF_LENGTH,
)
from noeira.physics3d.parser.flat_model import TEX_2D, TEX_CUBE
from noeira.physics3d.gpu.constants import (
    MODEL_BODY_SIZE,
    GEOM_IDX_BODY,
    GEOM_IDX_POS_X,
    GEOM_IDX_QUAT_W,
    GEOM_IDX_RAY_VISIBLE,
)
from noeira.physics3d.raytrace.appearance import (
    geom_uv,
    sample_texture,
    sample_texture_lod,
    texture_lod,
    mip_level_adr,
)
from noeira.physics3d.raytrace.visual import append_mip_chain
from noeira.physics3d.raytrace.camera import CameraFrame
from noeira.physics3d.raytrace.render import render_pixel
from noeira.physics3d.raytrace.visual_records import (
    APP_IDX_A,
    APP_IDX_B,
    APP_IDX_G,
    APP_IDX_MATID,
    APP_IDX_R,
    APP_IDX_REFLECT,
    APP_IDX_UVADR,
    LIGHT_BODY_HEADLIGHT,
    LIGHT_IDX_ACTIVE,
    LIGHT_IDX_AMBIENT_R,
    LIGHT_IDX_CUTOFF,
    LIGHT_IDX_DIFFUSE_R,
    LIGHT_IDX_DIRECTIONAL,
    LIGHT_IDX_DIR_Z,
    MAX_VIS_LIGHTS,
    MAX_VIS_MATERIALS,
    VIS_GEOM_APPEARANCE,
    VIS_LIGHT_WORDS,
    VIS_MAT_WORDS,
    MAX_VIS_TEXTURES,
    TEX_IDX_ACTIVE,
    TEX_IDX_ADR,
    TEX_IDX_HEIGHT,
    TEX_IDX_NCHAN,
    TEX_IDX_NLEVELS,
    TEX_IDX_TYPE,
    TEX_IDX_WIDTH,
    VIS_TEX_WORDS,
    VIS_UV_WORDS,
)

comptime DT = DType.float64


struct Tally(Copyable, Movable):
    var n: Int
    var bad: Int

    def __init__(out self):
        self.n = 0
        self.bad = 0

    def check(mut self, ok: Bool, what: String) raises:
        self.n += 1
        if not ok:
            self.bad += 1
            print("  FAIL:", what)

    def near(
        mut self, got: Float64, want: Float64, tol: Float64, what: String
    ) raises:
        var d = got - want
        if d < 0:
            d = -d
        self.n += 1
        if not (d <= tol):
            self.bad += 1
            print(
                "  FAIL:", what, "got", got, "want", want, "tol", tol
            )


def _tex_tables(
    mut meta: TensorImpl[DT],
    mut px: TensorImpl[DType.uint8],
    w: Int,
    h: Int,
    ttype: Int,
) raises:
    """One texture at slot 0, `w x h`, red ramping with the COLUMN and green
    with the ROW, so a swapped axis or a flipped row is a different number and
    not a different shade of the same one."""
    meta = TensorImpl[DT].alloc(MAX_VIS_TEXTURES * VIS_TEX_WORDS)
    meta.data[TEX_IDX_ADR] = 0
    meta.data[TEX_IDX_WIDTH] = Scalar[DT](w)
    meta.data[TEX_IDX_HEIGHT] = Scalar[DT](h)
    meta.data[TEX_IDX_TYPE] = Scalar[DT](ttype)
    meta.data[TEX_IDX_ACTIVE] = 1
    meta.data[TEX_IDX_NCHAN] = 3
    px = TensorImpl[DType.uint8].alloc(w * h * 3)
    for y in range(h):
        for x in range(w):
            var o = (y * w + x) * 3
            px.data[o + 0] = UInt8(255 * x // max(1, w - 1))
            px.data[o + 1] = UInt8(255 * y // max(1, h - 1))
            px.data[o + 2] = UInt8(0)


def _look(fx: Float64, fy: Float64, fz: Float64) -> CameraFrame[DT]:
    """A 1x1 camera at (0, -0.8, 0.8) whose only ray is `(fx, fy, fz)`.

    At width = height = 1 the single pixel's ray is exactly `-zaxis`
    (`camera_pixel_ray`'s `u = v = 0.5` puts both frustum offsets at 0), so
    the other two axes reach the shading only through `gaze` — and the one
    light in this scene is not the headlight, so they cannot change the
    answer.
    """
    var f = Vec3[DT](fx, fy, fz)
    var n = f / f.length()
    var xa = Vec3[DT](1, 0, 0)
    var ya = (-n).cross(xa)
    return CameraFrame[DT](
        Vec3[DT](0, -0.8, 0.8), xa, ya, -n, 45.0, 0.41421356237309503
    )


def _shoot[
    REFLECT: Bool, SAMPLES: Int = 1
](
    geoms: LayoutTensor[DT, DYN2, MutAnyOrigin],
    app: LayoutTensor[DT, DYN1, MutAnyOrigin],
    bodies: LayoutTensor[DT, DYN2, MutAnyOrigin],
    xpos: LayoutTensor[DT, DYN2, MutAnyOrigin],
    xquat: LayoutTensor[DT, DYN2, MutAnyOrigin],
    empty: LayoutTensor[DT, DYN1, MutAnyOrigin],
    texels: LayoutTensor[DType.uint8, DYN1, MutAnyOrigin],
    mats: LayoutTensor[DT, DYN1, MutAnyOrigin],
    texs: LayoutTensor[DT, DYN1, MutAnyOrigin],
    lights: LayoutTensor[DT, DYN1, MutAnyOrigin],
    frame: CameraFrame[DT],
    bg: Vec3[DT],
) raises -> Float64:
    """The red channel of the one pixel this camera has.

    ⚠ TOP LEVEL, NOT NESTED. A nested `def` closing over these views fails
    with "Could not infer capture convention of the captured value" — the
    same footgun the LIBERO viewer hit; the fix is the same one.
    """
    var h = render_pixel[DT, False, REFLECT, SAMPLES=SAMPLES](
        geoms, 2, app, bodies, xpos, xquat, 0,
        empty, empty, empty, empty, empty, 1,
        # `bodies` stands in for `qpos`: with 0 conditional sites it is never read.
        mats, texs, texels, lights, 1, bodies, 0, frame, 1, 1, 0, 0, bg,
        empty, -1,
    )
    return Float64(h.rgb.x)


def main() raises:
    var t = Tally()
    print("=" * 74)
    print("the tracer's appearance path — sampler, texgen, lighting")
    print("=" * 74)

    # ── the sampler ───────────────────────────────────────────────────────
    var meta = TensorImpl[DT]()
    var px = TensorImpl[DType.uint8]()
    _tex_tables(meta, px, 4, 4, TEX_2D)
    var mv = meta.lt_dyn["cpu", DYN1](rl1(meta.n))
    var pv = px.lt_dyn["cpu", DYN1](rl1(px.n))

    # A texel CENTRE is at (i + 0.5)/w, and there the bilinear weights are
    # (1, 0): the sample must be that texel exactly. Getting the half-texel
    # offset wrong shifts every texture by half a texel — invisible on a 4096
    # map and the whole image on a 4x4 one.
    for i in range(4):
        var u = (Float64(i) + 0.5) / 4.0
        var s = sample_texture[DT](mv, pv, 0, Scalar[DT](u), Scalar[DT](0.125))
        t.near(
            Float64(s.r), Float64(255 * i // 3) / 255.0, 1e-9,
            "texel centre column " + String(i),
        )
        t.near(Float64(s.g), 0.0, 1e-9, "texel centre row 0")

    # Halfway between two texel CENTRES is their mean, exactly. On a 4-wide
    # texture the centres are 0.125, 0.375, 0.625, 0.875, so 0.5 is the
    # midpoint of columns 1 and 2 — not 0.375, which is a centre.
    var mid = sample_texture[DT](
        mv, pv, 0, Scalar[DT](0.5), Scalar[DT](0.125)
    )
    t.near(
        Float64(mid.r),
        0.5 * (Float64(85) + Float64(170)) / 255.0,
        1e-9,
        "bilinear midpoint between columns 1 and 2",
    )

    # ⚠ ANTI-VACUITY: the two samples above must DIFFER. A sampler returning a
    # constant would satisfy neither check on its own if the constant were
    # right, and both if it were the mean.
    t.check(
        Float64(mid.r) != Float64(
            sample_texture[DT](mv, pv, 0, Scalar[DT](0.125), Scalar[DT](0.125)).r
        ),
        "the sampler returns different colours for different u",
    )

    # `GL_REPEAT` on a 2D texture: u and u+1 are the same texel.
    var a0 = sample_texture[DT](mv, pv, 0, Scalar[DT](0.375), Scalar[DT](0.625))
    var a1 = sample_texture[DT](mv, pv, 0, Scalar[DT](3.375), Scalar[DT](0.625))
    t.near(Float64(a0.r), Float64(a1.r), 1e-12, "2D wrap: u and u+3 agree")
    t.near(Float64(a0.g), Float64(a1.g), 1e-12, "2D wrap: v unchanged")

    # A CUBE face CLAMPS instead — `mjr_uploadTexture` sets GL_CLAMP_TO_EDGE,
    # and a repeat there shows the opposite edge across a silhouette.
    var cmeta = TensorImpl[DT]()
    var cpx = TensorImpl[DType.uint8]()
    _tex_tables(cmeta, cpx, 4, 4, TEX_CUBE)
    var cmv = cmeta.lt_dyn["cpu", DYN1](rl1(cmeta.n))
    var cpv = cpx.lt_dyn["cpu", DYN1](rl1(cpx.n))
    var c_out = sample_texture[DT](cmv, cpv, 0, Scalar[DT](1.9), Scalar[DT](0.5))
    var c_edge = sample_texture[DT](cmv, cpv, 0, Scalar[DT](1.0), Scalar[DT](0.5))
    t.near(
        Float64(c_out.r), Float64(c_edge.r), 1e-12,
        "cube clamp: u past 1 reads the last column",
    )
    t.check(
        Float64(c_out.r) != Float64(a0.r),
        "cube clamp and 2D wrap are not the same answer",
    )

    # An inactive slot is a MISS and returns white, so an unset material
    # multiplies its colour by 1 rather than by black.
    var miss = sample_texture[DT](mv, pv, 7, Scalar[DT](0.5), Scalar[DT](0.5))
    t.check(not miss.hit, "an inactive texture slot reports a miss")
    t.near(Float64(miss.r), 1.0, 1e-12, "a miss is white, not black")

    # ── the mip chain and the trilinear sampler ───────────────────────────
    #
    # A 4x4 with a known 2x2 box per quadrant, so every level is a closed
    # form: level 1 is the four quadrant means, level 2 the image mean.
    var mip = List[UInt8]()
    for y in range(4):
        for x in range(4):
            var q = (0 if x < 2 else 1) + (0 if y < 2 else 2)
            # quadrant q holds 4 values averaging to 40 + 60*q, spread +-10
            var jitter = ((x % 2) + 2 * (y % 2)) * 20 - 30
            for c in range(3):
                mip.append(UInt8(40 + 60 * q + jitter + c))
    var nlev = append_mip_chain(mip, 0, 4, 4)
    t.check(nlev == 3, "a 4x4 texture has 3 levels (4, 2, 1)")
    t.check(len(mip) == (16 + 4 + 1) * 3, "the chain appends 4 + 1 texels")
    t.check(mip_level_adr(0, 4, 4, 1) == 48, "level 1 starts after level 0")
    t.check(mip_level_adr(0, 4, 4, 2) == 60, "level 2 starts after level 1")
    for q in range(4):
        t.check(
            Int(mip[48 + q * 3]) == 40 + 60 * q,
            "level 1 texel " + String(q) + " is its quadrant's mean",
        )
    # levels 1's mean is 130; level 2 is that, rounded
    t.check(Int(mip[60]) == 130, "level 2 is the whole image's mean")
    # ⚠ ROUNDED, not truncated: (1 + 2 + 2 + 2 + 2) // 4 would be 1.
    var rnd = List[UInt8]()
    rnd.append(1); rnd.append(0); rnd.append(0)
    rnd.append(2); rnd.append(0); rnd.append(0)
    rnd.append(2); rnd.append(0); rnd.append(0)
    rnd.append(2); rnd.append(0); rnd.append(0)
    _ = append_mip_chain(rnd, 0, 2, 2)
    t.check(Int(rnd[12]) == 2, "a mip texel ROUNDS (7/4 -> 2, not 1)")
    # An ODD size clamps: a 3x2's level 1 is 1x1 and averages columns 0-1.
    var odd = List[UInt8]()
    for y in range(2):
        for x in range(3):
            for _ in range(3):
                odd.append(UInt8(10 if x < 2 else 250))
    t.check(append_mip_chain(odd, 0, 3, 2) == 2, "a 3x2 has levels 3x2, 1x1")
    t.check(Int(odd[18]) == 10, "an odd width drops its last column (GL floor)")

    var mmeta = TensorImpl[DT].alloc(MAX_VIS_TEXTURES * VIS_TEX_WORDS)
    mmeta.data[TEX_IDX_ADR] = 0
    mmeta.data[TEX_IDX_WIDTH] = 4
    mmeta.data[TEX_IDX_HEIGHT] = 4
    mmeta.data[TEX_IDX_TYPE] = Scalar[DT](TEX_2D)
    mmeta.data[TEX_IDX_ACTIVE] = 1
    mmeta.data[TEX_IDX_NCHAN] = 3
    mmeta.data[TEX_IDX_NLEVELS] = Scalar[DT](nlev)
    var mpx = TensorImpl[DType.uint8].alloc(len(mip))
    for i in range(len(mip)):
        mpx.data[i] = mip[i]
    var mmv = mmeta.lt_dyn["cpu", DYN1](rl1(mmeta.n))
    var mpv = mpx.lt_dyn["cpu", DYN1](rl1(mpx.n))
    # At a level-1 texel centre (0.25, 0.25) the sample IS that texel.
    var l1 = sample_texture_lod[DT](
        mmv, mpv, 0, Scalar[DT](0.25), Scalar[DT](0.25), Scalar[DT](1)
    )
    t.near(Float64(l1.r), 40.0 / 255.0, 1e-9, "lod 1 reads level 1")
    # ⚠ AT A LEVEL-0 TEXEL CENTRE, NOT AT (0.25, 0.25): there level 0's
    # bilinear weights average exactly the quadrant's four texels, which IS
    # level 1's texel, and the two levels cannot be told apart.
    var c0 = sample_texture_lod[DT](
        mmv, mpv, 0, Scalar[DT](0.125), Scalar[DT](0.125), Scalar[DT](0)
    )
    var c1 = sample_texture_lod[DT](
        mmv, mpv, 0, Scalar[DT](0.125), Scalar[DT](0.125), Scalar[DT](1)
    )
    var ch = sample_texture_lod[DT](
        mmv, mpv, 0, Scalar[DT](0.125), Scalar[DT](0.125), Scalar[DT](0.5)
    )
    t.near(Float64(c0.r), 10.0 / 255.0, 1e-9, "lod 0 at a texel centre is that texel")
    t.near(
        Float64(ch.r), 0.5 * (Float64(c0.r) + Float64(c1.r)), 1e-9,
        "lod 0.5 is the mean of the two levels (GL_LINEAR_MIPMAP_LINEAR)",
    )
    t.check(Float64(c0.r) != Float64(c1.r), "levels 0 and 1 differ (anti-vacuity)")
    var l0 = c0
    var lbig = sample_texture_lod[DT](
        mmv, mpv, 0, Scalar[DT](0.9), Scalar[DT](0.1), Scalar[DT](40)
    )
    t.near(Float64(lbig.r), 130.0 / 255.0, 1e-9, "lod past the last level clamps to it")
    var lneg = sample_texture_lod[DT](
        mmv, mpv, 0, Scalar[DT](0.125), Scalar[DT](0.125), Scalar[DT](-3)
    )
    t.near(Float64(lneg.r), Float64(l0.r), 1e-12, "lod <= 0 is level 0 (c = 0)")

    # OpenGL's lambda: one texel per pixel is 0, four is 2, and the LARGER
    # axis wins.
    t.near(Float64(texture_lod[DT](mmv, 0, 0.25, 0.0, 0.0, 0.25)), 0.0, 1e-9,
           "1 texel per pixel -> lod 0")
    t.near(Float64(texture_lod[DT](mmv, 0, 1.0, 0.0, 0.0, 1.0)), 2.0, 1e-9,
           "4 texels per pixel -> lod 2")
    t.near(Float64(texture_lod[DT](mmv, 0, 0.25, 0.0, 0.0, 2.0)), 3.0, 1e-9,
           "anisotropic: the larger derivative decides")
    t.near(Float64(texture_lod[DT](mmv, 0, 0.0, 0.5, 0.0, 0.0)), 1.0, 1e-9,
           "v is scaled by the height")

    # ── texgen ────────────────────────────────────────────────────────────
    var geoms = TensorImpl[DT].alloc(2 * MODEL_GEOM_SIZE)
    geoms.data[GEOM_IDX_TYPE] = Scalar[DT](GEOM_BOX)
    geoms.data[GEOM_IDX_HALF_X] = 0.5
    geoms.data[GEOM_IDX_HALF_Y] = 0.25
    geoms.data[GEOM_IDX_HALF_Z] = 0.1
    geoms.data[MODEL_GEOM_SIZE + GEOM_IDX_TYPE] = Scalar[DT](GEOM_PLANE)
    geoms.data[MODEL_GEOM_SIZE + GEOM_IDX_HALF_X] = 3.0
    geoms.data[MODEL_GEOM_SIZE + GEOM_IDX_HALF_Y] = 2.0
    var gv = geoms.lt_dyn["cpu", DYN2](rl2(2, MODEL_GEOM_SIZE))
    var uvs = TensorImpl[DT].alloc(VIS_UV_WORDS)
    uvs.data[0] = 0.0
    uvs.data[1] = 0.0
    uvs.data[2] = 1.0
    uvs.data[3] = 0.0
    uvs.data[4] = 0.0
    uvs.data[5] = 1.0
    var uvv = uvs.lt_dyn["cpu", DYN1](rl1(uvs.n))

    var one = Scalar[DT](1)
    # A box's +Z face: `u = (x/sx + 1)/2`, `v = 1 - (y/sy + 1)/2`. The centre
    # of the face is the centre of the texture.
    var bc = geom_uv[DT](
        gv, uvv, 0, -1, 0, 0,
        Vec3[DT](0, 0, 0.1), Vec3[DT](0, 0, 1),
        TEX_2D, one, one, False,
    )
    t.near(Float64(bc.u), 0.5, 1e-12, "box +Z centre u")
    t.near(Float64(bc.v), 0.5, 1e-12, "box +Z centre v")
    var bx = geom_uv[DT](
        gv, uvv, 0, -1, 0, 0,
        Vec3[DT](0.5, -0.25, 0.1), Vec3[DT](0, 0, 1),
        TEX_2D, one, one, False,
    )
    t.near(Float64(bx.u), 1.0, 1e-12, "box +Z corner u")
    t.near(Float64(bx.v), 1.0, 1e-12, "box +Z corner v is FLIPPED")

    # ⚠ THE FACE COMES FROM THE NORMAL. The same point with a +X normal reads
    # the (y, z) pair instead, and must NOT give the +Z answer.
    var bxx = geom_uv[DT](
        gv, uvv, 0, -1, 0, 0,
        Vec3[DT](0.5, -0.25, 0.1), Vec3[DT](1, 0, 0),
        TEX_2D, one, one, False,
    )
    t.check(
        Float64(bxx.u) != Float64(bx.u),
        "the box face is chosen by the NORMAL, not by the position",
    )
    t.near(Float64(bxx.u), 0.0, 1e-12, "box +X face u is (y/sy + 1)/2")

    # `texrepeat` multiplies, and `texuniform` multiplies again by the size —
    # `settexture`'s `scl[k] * geom->size[k]`.
    var br = geom_uv[DT](
        gv, uvv, 0, -1, 0, 0,
        Vec3[DT](0.5, -0.25, 0.1), Vec3[DT](0, 0, 1),
        TEX_2D, Scalar[DT](3), Scalar[DT](3), False,
    )
    t.near(Float64(br.u), 3.0, 1e-12, "texrepeat scales the box UV")
    var bu = geom_uv[DT](
        gv, uvv, 0, -1, 0, 0,
        Vec3[DT](0.5, -0.25, 0.1), Vec3[DT](0, 0, 1),
        TEX_2D, Scalar[DT](3), Scalar[DT](3), True,
    )
    t.near(
        Float64(bu.u), 3.0 * 0.5, 1e-12,
        "texuniform multiplies the repeat by the geom's half-size",
    )

    # A finite plane: `u = (x + sx)/(2 sx)`, `v = 1 - (y + sy)/(2 sy)`.
    var pc = geom_uv[DT](
        gv, uvv, 1, -1, 0, 0,
        Vec3[DT](1.5, 1.0, 0), Vec3[DT](0, 0, 1),
        TEX_2D, one, one, False,
    )
    t.near(Float64(pc.u), 0.75, 1e-12, "plane u")
    t.near(Float64(pc.v), 0.25, 1e-12, "plane v")

    # A mesh triangle: the barycentrics interpolate the corner UVs. `bu`
    # weights corner 0, `bv` corner 1, and `1 - bu - bv` corner 2 — the same
    # order `ray_triangle` solves for.
    geoms.data[GEOM_IDX_TYPE] = Scalar[DT](GEOM_MESH)
    var gv2 = geoms.lt_dyn["cpu", DYN2](rl2(2, MODEL_GEOM_SIZE))
    var mu = geom_uv[DT](
        gv2, uvv, 0, 0, Scalar[DT](0.25), Scalar[DT](0.5),
        Vec3[DT](0, 0, 0), Vec3[DT](0, 0, 1),
        TEX_2D, one, one, False,
    )
    t.near(Float64(mu.u), 0.5, 1e-12, "mesh u = bv * u1")
    t.near(Float64(mu.v), 0.25, 1e-12, "mesh v = (1 - bu - bv) * v2")

    # ── the reflection pass ───────────────────────────────────────────────
    #
    # Two boxes and one light, so the answer is a product of three numbers.
    #   geom 0  the MIRROR: half (0.2, 0.2, 0.1) at the origin, rgba 0.4,
    #           reflectance 0.5
    #   geom 1  a white lid, half (4, 4, 0.1) at z = 2 — what the mirror sees
    #   light   DIRECTIONAL travelling +z (so it shines from BELOW), ambient
    #           0.2, diffuse 1, specular 0
    #
    # The lid's UNDERSIDE faces the light, so `ndotl` is 1 there; the mirror's
    # top face is turned away from it and gets ambient only. That separates
    # the mirror's own colour from what it reflects with no overlap.
    var rg = TensorImpl[DT].alloc(2 * MODEL_GEOM_SIZE)
    for gi in range(2):
        var o = gi * MODEL_GEOM_SIZE
        rg.data[o + GEOM_IDX_TYPE] = Scalar[DT](GEOM_BOX)
        rg.data[o + GEOM_IDX_BODY] = 0
        rg.data[o + GEOM_IDX_QUAT_W] = 1
        rg.data[o + GEOM_IDX_RAY_VISIBLE] = 1
    rg.data[GEOM_IDX_HALF_X] = 0.2
    rg.data[GEOM_IDX_HALF_Y] = 0.2
    rg.data[GEOM_IDX_HALF_Z] = 0.1
    var o1 = MODEL_GEOM_SIZE
    rg.data[o1 + GEOM_IDX_POS_X + 2] = 2.0
    rg.data[o1 + GEOM_IDX_HALF_X] = 4.0
    rg.data[o1 + GEOM_IDX_HALF_Y] = 4.0
    rg.data[o1 + GEOM_IDX_HALF_Z] = 0.1

    var ra = TensorImpl[DT].alloc(2 * VIS_GEOM_APPEARANCE)
    for gi in range(2):
        var o = gi * VIS_GEOM_APPEARANCE
        var c = Scalar[DT](0.4) if gi == 0 else Scalar[DT](1.0)
        ra.data[o + APP_IDX_R] = c
        ra.data[o + APP_IDX_G] = c
        ra.data[o + APP_IDX_B] = c
        ra.data[o + APP_IDX_A] = 1
        ra.data[o + APP_IDX_MATID] = -1
        ra.data[o + APP_IDX_UVADR] = -1
    ra.data[APP_IDX_REFLECT] = 0.5

    var rl = TensorImpl[DT].alloc(MAX_VIS_LIGHTS * VIS_LIGHT_WORDS)
    rl.data[LIGHT_IDX_DIRECTIONAL] = 1
    rl.data[LIGHT_IDX_CUTOFF] = 180
    rl.data[LIGHT_IDX_DIR_Z] = 1.0
    rl.data[LIGHT_IDX_AMBIENT_R + 0] = 0.2
    rl.data[LIGHT_IDX_AMBIENT_R + 1] = 0.2
    rl.data[LIGHT_IDX_AMBIENT_R + 2] = 0.2
    rl.data[LIGHT_IDX_DIFFUSE_R + 0] = 1.0
    rl.data[LIGHT_IDX_DIFFUSE_R + 1] = 1.0
    rl.data[LIGHT_IDX_DIFFUSE_R + 2] = 1.0
    rl.data[LIGHT_IDX_ACTIVE] = 1

    var rb = TensorImpl[DT].alloc(MODEL_BODY_SIZE)
    var rx = TensorImpl[DT].alloc(3)
    var rq = TensorImpl[DT].alloc(4)
    rq.data[3] = 1
    var one_f = TensorImpl[DT].alloc(1)
    var one_u8 = TensorImpl[DType.uint8].alloc(1)
    var rmat = TensorImpl[DT].alloc(MAX_VIS_MATERIALS * VIS_MAT_WORDS)
    var rtex = TensorImpl[DT].alloc(MAX_VIS_TEXTURES * VIS_TEX_WORDS)

    var gv3 = rg.lt_dyn["cpu", DYN2](rl2(2, MODEL_GEOM_SIZE))
    var av = ra.lt_dyn["cpu", DYN1](rl1(ra.n))
    var bv = rb.lt_dyn["cpu", DYN2](rl2(1, MODEL_BODY_SIZE))
    var xv = rx.lt_dyn["cpu", DYN2](rl2(1, 3))
    var qv = rq.lt_dyn["cpu", DYN2](rl2(1, 4))
    var e1 = one_f.lt_dyn["cpu", DYN1](rl1(1))
    var eu = one_u8.lt_dyn["cpu", DYN1](rl1(1))
    var mtv = rmat.lt_dyn["cpu", DYN1](rl1(rmat.n))
    var ttv = rtex.lt_dyn["cpu", DYN1](rl1(rtex.n))
    var ltv = rl.lt_dyn["cpu", DYN1](rl1(rl.n))
    var bg = Vec3[DT](0, 0, 0)

    # The top face: ambient on the mirror (0.2 * 0.4) plus HALF the lid's own
    # colour (ambient 0.2 + diffuse 1.0, times its white 1.0, times 0.5).
    var top = _look(0.0, 0.8, -0.75)
    t.near(
        _shoot[True](gv3, av, bv, xv, qv, e1, eu, mtv, ttv, ltv, top, bg), 0.2 * 0.4 + 0.5 * (0.2 + 1.0) * 1.0, 1e-9,
        "the reflection is ADDED to the mirror's own colour",
    )
    # ⚠ AND NOT BLENDED. `mjr_render` draws the mirror with
    # `glBlendFunc(GL_ONE, GL_ONE)` over the mirrored scene, so a lerp — the
    # reflex a ray tracer reaches for — is a DIFFERENT and dimmer answer.
    t.check(
        abs(_shoot[True](gv3, av, bv, xv, qv, e1, eu, mtv, ttv, ltv, top, bg)
            - (0.5 * 0.2 * 0.4 + 0.5 * (0.2 + 1.0))) > 1e-6,
        "the reflection is not a lerp between the two",
    )
    t.near(
        _shoot[False](gv3, av, bv, xv, qv, e1, eu, mtv, ttv, ltv, top, bg), 0.2 * 0.4, 1e-9,
        "REFLECT=False leaves the mirror's own colour alone",
    )

    # A SIDE face of the same box gets no reflection at all: the reference's
    # stencil is the +Z face's silhouette, not the box's.
    var side = _look(0.0, 0.6, -0.85)
    t.near(
        _shoot[True](gv3, av, bv, xv, qv, e1, eu, mtv, ttv, ltv, side, bg), 0.2 * 0.4, 1e-9,
        "a side face of the mirror geom reflects NOTHING",
    )
    t.check(
        abs(_shoot[True](gv3, av, bv, xv, qv, e1, eu, mtv, ttv, ltv, side, bg) - _shoot[True](gv3, av, bv, xv, qv, e1, eu, mtv, ttv, ltv, top, bg)) > 0.1,
        "the two rays land on different faces (anti-vacuity)",
    )

    # ── multisampling ────────────────────────────────────────────────────
    # A pixel wholly inside one surface: every sample lands on the centre's
    # geom, so 4x MSAA reuses the centre's shading and the answer is the
    # single-sample one EXACTLY (shading runs once per primitive).
    # ⚠ AIMED AT THE 8 m LID, NOT THE MIRROR: `_shoot` is a 1x1 image at 45
    # degrees, so a 3/8-pixel sample offset swings the ray ~16 degrees and
    # would miss the 0.4 m mirror entirely.
    var up = _look(0.0, 0.3, 1.0)
    var one_s = _shoot[False, 1](gv3, av, bv, xv, qv, e1, eu, mtv, ttv, ltv, up, bg)
    t.check(one_s > 0.1, "the up-looking ray hits the lid (anti-vacuity)")
    t.near(
        _shoot[False, 4](gv3, av, bv, xv, qv, e1, eu, mtv, ttv, ltv, up, bg),
        one_s, 1e-12, "4x MSAA inside a surface equals one sample",
    )
    # And at the mirror, where every sample misses: 4x MSAA is the COVERAGE
    # average, so a pixel whose samples all miss is the background, even
    # though its centre ray hit.
    t.near(
        _shoot[True, 4](gv3, av, bv, xv, qv, e1, eu, mtv, ttv, ltv, top, bg),
        0.0, 1e-12, "4x MSAA: a centre hit with no covered sample is background",
    )

    # ── transparency ─────────────────────────────────────────────────────
    # The mirror at alpha 0.5, against a WHITE background, reflection off.
    # Its top face is ambient-only (0.2 * 0.4 = 0.08). MuJoCo blends a
    # transparent geom TWICE (front-to-back and back-to-front passes), so
    # it lands at 0.5 * (2 - 0.5) = 0.75. Below the top face the ray is
    # INSIDE the box: its bottom face is a BACK face, culled, and must not be
    # composited — the lid is above, so what shows through is the white.
    av[APP_IDX_A] = Scalar[DT](0.5)
    var white = Vec3[DT](1, 1, 1)
    var tr = _shoot[False](gv3, av, bv, xv, qv, e1, eu, mtv, ttv, ltv, top, white)
    t.near(tr, 0.75 * 0.08 + 0.25 * 1.0, 1e-9,
           "a transparent surface: opacity a(2-a), over what is behind")
    # Drawn with the box's lit bottom face composited, it would read
    # 0.75*0.08 + 0.25*(0.75*0.48 + 0.25) — the back-face guard is the difference.
    t.check(abs(tr - (0.06 + 0.25 * (0.75 * 0.48 + 0.25))) > 1e-3,
            "the transparent box's own back face is not composited")
    t.check(abs(tr - (0.5 * 0.08 + 0.5)) > 1e-3,
            "the opacity is a(2-a), not a (anti-vacuity)")
    av[APP_IDX_A] = Scalar[DT](1)

    print()
    print(t.n, "checks,", t.bad, "failed")
    if t.bad != 0:
        raise Error("the appearance path is wrong in " + String(t.bad)
                    + " of " + String(t.n) + " checks")
    print("=== PASS ===")
