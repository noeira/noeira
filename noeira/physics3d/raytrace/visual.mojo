"""`VisualModel` — the tracer's own scene, beside the solver's.

    var vis = build_visual_model[DT](fmd, m, group_mask=1 << 1)
    vis.upload(ctx)

WHY A SECOND GEOM TABLE AND NOT A FLAG ON THE FIRST
===================================================
Until L5 the tracer read `Model.geoms` directly, which made three things
impossible at once and a fourth one wrong:

* **The visual meshes are not in `Model` at all.** `fields_build` skips a mesh
  geom that is neither collidable nor a source of body inertia — dog carries
  162 of them — and sets its `GEOM_IDX_MESH_ID` to -1 so the stale asset index
  cannot be dereferenced. Every LIBERO object's appearance is exactly such a
  geom, so a tracer over `Model` draws the collision boxes and nothing else.
* **A group mask in the kernel costs a branch per (pixel, geom).** robosuite
  renders with `render_collision_mesh=False`; the picture is the group-1
  geoms. Filtering at BUILD time is the same answer with 36 geoms in the loop
  instead of 240 — and the count, not the branch, is what a per-pixel loop
  pays.
* **Materials and textures reach nothing.** They are parsed
  (`RenderFields`) and then dropped at the `Model` boundary.
* And the fourth: `Model` bakes the mesh's PRINCIPAL frame into a mesh geom's
  pose, because collision wants the hull in that frame. This table does not,
  and keeps the soup in the file's own frame instead — the pair is
  self-consistent either way, and choosing the file's frame is what lets a
  VISUAL mesh skip `mesh_inertia_from_file` entirely.

WHAT IT DOES NOT DO
===================
⚠ IT IS BUILT ONCE AND IS NOT PER-LANE. Every field here is a MODEL constant;
the only per-lane input the tracer reads is `Data.xpos` / `xquat` /
`subtree_com`, in place, exactly as before. A task that randomises an
appearance per lane would need a different structure and does not exist.

⚠ NO SKYBOX AND NO REFLECTION. The background stays a flat colour. Both are
listed in `shade.mojo` as absent; the skybox is now PARSED (it is a texture
like any other) and simply not sampled, because nothing in the LIBERO frame
reaches it — the arena's walls enclose the camera.

⚠ `mark` / `random` ON A BUILTIN TEXTURE ARE IGNORED. `TextureData` carries
them and `<texture mark="edge">` would draw a border; no model in this tree
sets one, and a builtin nobody samples is the wrong place to spend the
transcription.
"""

from std.math import sqrt

from noeira.nn.core.tensor import TensorImpl
from max.gpu.host import DeviceContext

from ..gpu.constants import (
    MODEL_GEOM_SIZE,
    GEOM_IDX_TYPE,
    GEOM_IDX_POS_X,
    GEOM_IDX_POS_Y,
    GEOM_IDX_POS_Z,
    GEOM_IDX_QUAT_X,
    GEOM_IDX_QUAT_Y,
    GEOM_IDX_QUAT_Z,
    GEOM_IDX_QUAT_W,
    GEOM_IDX_HALF_X,
    GEOM_IDX_HALF_Y,
    GEOM_IDX_HALF_Z,
    GEOM_IDX_MESH_ID,
    GEOM_IDX_GROUP,
    GEOM_IDX_RAY_VISIBLE,
    GEOM_IDX_BODY,
    GEOM_IDX_RADIUS,
    GEOM_IDX_HALF_LENGTH,
    MODEL_MESH_META_SIZE,
    MESH_META_IDX_TRIADR,
    MESH_META_IDX_TRINUM,
    MESH_META_IDX_BVHADR,
    MESH_META_IDX_BVHNUM,
    MESH_ARENA_RECORD,
)
from ..constants import GEOM_BOX, GEOM_MESH, GEOM_PLANE
from ..fields import Model
from ..fields.dims import DimsLike
from ..parser.flat_model import (
    FlatModelDef,
    TEX_2D,
    TEX_CUBE,
    TEX_SKYBOX,
    TEX_BUILTIN_NONE,
    TEX_BUILTIN_FLAT,
    TEX_BUILTIN_CHECKER,
    TEX_BUILTIN_GRADIENT,
)
from ..parser.mesh_bvh_build import build_mesh_bvh
from ..model.mesh_inertia import apply_mesh_ref_transform
from .visual_records import *


@always_inline
def _pos1(n: Int) -> Int:
    return n if n > 0 else 1


struct VisualModel[DTYPE: DType](Movable):
    """Everything a camera reads that the solver does not.

    ⚠ `ngeom` IS THIS TABLE'S COUNT, NOT THE MODEL'S. Pass it, not
    `D.NGEOM`, to `ray_model`.
    """

    var ngeom: Int
    var ncond: Int
    """Conditional-site rows after the `ngeom` ordinary ones — see
    `APP_IDX_COND_QADR`. `geoms` and `appearance` hold `ngeom + ncond` rows."""
    var nmesh: Int
    var ntri: Int
    var nmat: Int
    var ntex: Int
    var nlight: Int
    var ntexel_bytes: Int
    var mirror: Int
    """The one geom `mjr_render` lets reflect, or -1. See `APP_IDX_REFLECT`."""
    var src_geom: List[Int]
    """`[ngeom]` — the MODEL geom each visual row was built from (the build's
    `keep` list), so a caller can name a visual row by its geom's name. The
    domain randomizer (`randomize.mojo`) groups surfaces this way."""

    var geoms: TensorImpl[Self.DTYPE]
    """`[ngeom, MODEL_GEOM_SIZE]` — the solver's own record layout, so
    `ray_model` reads it unchanged. Poses are the XML's; see the header."""

    var appearance: TensorImpl[Self.DTYPE]
    """`[ngeom, VIS_GEOM_APPEARANCE]` — rgba, material id, mesh UV address."""

    var mesh_meta: TensorImpl[Self.DTYPE]
    var mesh_tris: TensorImpl[Self.DTYPE]
    var mesh_uv: TensorImpl[Self.DTYPE]
    var materials: TensorImpl[Self.DTYPE]
    var textures: TensorImpl[Self.DTYPE]
    var lights: TensorImpl[Self.DTYPE]

    var texels: TensorImpl[DType.uint8]
    """The texture atlas, three bytes a texel, every texture end to end.

    ⚠⚠ `uint8`, AND THAT IS A SIZE DECISION WITH A NUMBER ON IT. libero_goal
    carries 143 MB of texels — the bowl's and the plate's diffuse maps are
    4096x4096 each. As `DTYPE` floats that is 572 MB at float32 and 1.1 GB at
    float64, for values that were eight bits when they left the PNG. The
    sampler converts one texel at a time, in registers.
    """

    def __init__(out self):
        self.ngeom = 0
        self.ncond = 0
        self.nmesh = 0
        self.ntri = 0
        self.nmat = 0
        self.ntex = 0
        self.nlight = 0
        self.ntexel_bytes = 0
        self.mirror = -1
        self.src_geom = List[Int]()
        self.geoms = TensorImpl[Self.DTYPE]()
        self.appearance = TensorImpl[Self.DTYPE]()
        self.mesh_meta = TensorImpl[Self.DTYPE]()
        self.mesh_tris = TensorImpl[Self.DTYPE]()
        self.mesh_uv = TensorImpl[Self.DTYPE]()
        self.materials = TensorImpl[Self.DTYPE]()
        self.textures = TensorImpl[Self.DTYPE]()
        self.lights = TensorImpl[Self.DTYPE]()
        self.texels = TensorImpl[DType.uint8]()

    def upload(mut self, ctx: DeviceContext) raises:
        """Every table to the device. Once, after the build."""
        self.geoms.upload(ctx)
        self.appearance.upload(ctx)
        self.mesh_meta.upload(ctx)
        self.mesh_tris.upload(ctx)
        self.mesh_uv.upload(ctx)
        self.materials.upload(ctx)
        self.textures.upload(ctx)
        self.lights.upload(ctx)
        self.texels.upload(ctx)

    def describe(self) -> String:
        return (
            String("visual: ") + String(self.ngeom) + " geoms, "
            + String(self.nmesh) + " meshes / " + String(self.ntri)
            + " tris, " + String(self.nmat) + " materials, "
            + String(self.ntex) + " textures ("
            + String(self.ntexel_bytes // 1000000) + " MB), "
            + String(self.nlight) + " lights, mirror "
            + (String(self.mirror) if self.mirror >= 0 else String("none"))
        )


# ─── the build ───────────────────────────────────────────────────────────────


@fieldwise_init
struct SiteCondition(Copyable, Movable):
    """A site the camera draws, when a joint says so.

    `site` and `joint` are the COMPILED names (prefixes included). The site
    is visible in a lane when that lane's `qpos` at the joint's address is at
    least `min_qpos`; an empty `joint` makes it always visible. What decides
    which sites qualify is the benchmark's business, not the tracer's —
    LIBERO's rule lives in `envs/libero/visual.mojo`."""

    var site: String
    var joint: String
    var min_qpos: Float64


def append_mip_chain(mut texels: List[UInt8], adr: Int, w: Int, h: Int) -> Int:
    """Append levels 1.. of the RGB image at `texels[adr:]` and return how many
    levels the texture now has, level 0 included.

    `glGenerateMipmap`, which `render_context.c` calls on every texture it
    uploads. OpenGL leaves the filter to the implementation; this is the 2x2
    box every desktop driver uses for a power-of-two level. ⚠ AN ODD SIZE
    DROPS ITS LAST ROW OR COLUMN INTO THE CLAMP: level `k+1` is
    `max(1, dim >> 1)` (OpenGL's `floor(dim / 2)`), and a 1-wide level
    averages its single column with itself. Averaging in the stored space is
    correct because the atlas is LINEAR — an sRGB PNG was decoded before this
    ran, and `GL_SRGB8` mipmaps are filtered in linear too.

    ⚠ ROUNDED, NOT TRUNCATED: `(a + b + c + d + 2) // 4`. Truncating darkens
    every level by half a step, and twelve levels of it is six grey levels on
    the far end of a floor.
    """
    var levels = 1
    var pw = w
    var ph = h
    var padr = adr
    while pw > 1 or ph > 1:
        var nw = pw >> 1 if pw > 1 else 1
        var nh = ph >> 1 if ph > 1 else 1
        var nadr = len(texels)
        texels.reserve(nadr + nw * nh * 3)
        for y in range(nh):
            var y0 = 2 * y if 2 * y < ph else ph - 1
            var y1 = 2 * y + 1 if 2 * y + 1 < ph else ph - 1
            for x in range(nw):
                var x0 = 2 * x if 2 * x < pw else pw - 1
                var x1 = 2 * x + 1 if 2 * x + 1 < pw else pw - 1
                for c in range(3):
                    var sum = (
                        Int(texels[padr + (y0 * pw + x0) * 3 + c])
                        + Int(texels[padr + (y0 * pw + x1) * 3 + c])
                        + Int(texels[padr + (y1 * pw + x0) * 3 + c])
                        + Int(texels[padr + (y1 * pw + x1) * 3 + c])
                    )
                    texels.append(UInt8((sum + 2) // 4))
        padr = nadr
        pw = nw
        ph = nh
        levels += 1
    return levels


@always_inline
def _clampg(g: Int) -> Int:
    """`mj_ray`'s own clamp: a group outside `[0, 5]` reads the last mask slot
    rather than falling off the end."""
    if g < 0:
        return 0
    return 5 if g > 5 else g


def _builtin_texels(
    builtin: Int,
    mark: Int,
    w: Int,
    h: Int,
    r1: Float64, g1: Float64, b1: Float64,
    r2: Float64, g2: Float64, b2: Float64,
    mut out: List[UInt8],
):
    """`mjCTexture::Builtin` for the three patterns a model here declares.

    ⚠ THE GRADIENT RUNS DOWN THE IMAGE, not across, and it is `rgb1` at the
    TOP. MuJoCo builds a skybox as six faces stacked vertically and blends
    `rgb1 -> rgb2` over the whole strip; a 2D gradient is the same blend over
    its rows. Nothing samples one today (the skybox is not traced and no geom
    carries a gradient), so this is here to keep the atlas complete rather
    than because a pixel depends on it.
    """
    var n = w * h
    for i in range(n):
        var row = i // w
        var col = i - row * w
        var rr = r1
        var gg = g1
        var bb = b1
        if builtin == TEX_BUILTIN_GRADIENT:
            var f = Float64(row) / Float64(h - 1) if h > 1 else 0.0
            rr = r1 * (1.0 - f) + r2 * f
            gg = g1 * (1.0 - f) + g2 * f
            bb = b1 * (1.0 - f) + b2 * f
        elif builtin == TEX_BUILTIN_CHECKER:
            # MuJoCo's checker is 2x2 over the image, `rgb1` at the origin.
            var qx = 1 if col * 2 >= w else 0
            var qy = 1 if row * 2 >= h else 0
            if (qx + qy) % 2 == 1:
                rr = r2
                gg = g2
                bb = b2
        var o = i * 3
        out[o + 0] = UInt8(max(0.0, min(255.0, rr * 255.0 + 0.5)))
        out[o + 1] = UInt8(max(0.0, min(255.0, gg * 255.0 + 0.5)))
        out[o + 2] = UInt8(max(0.0, min(255.0, bb * 255.0 + 0.5)))


def build_visual_model[
    DTYPE: DType, D: DimsLike
](
    fmd: FlatModelDef,
    mut m: Model[DTYPE, D],
    group_mask: Int = 0b111,
    verbose: Bool = False,
    conditions: List[SiteCondition] = List[SiteCondition](),
    bvh_sah: Bool = True,
) raises -> VisualModel[DTYPE]:
    """Everything a camera needs, from the parse and the built `Model`.

    `group_mask` is six bits, one per `<geom group>`, and the default is
    MuJoCo's own visualiser default (0, 1, 2 on). **LIBERO wants `1 << 1`** —
    robosuite renders with `render_collision_mesh=False` and every LIBERO
    object is a group-1 visual mesh sitting on top of a pile of group-0
    collision boxes. Rendering MuJoCo's default set instead scores 10.8 dB
    against the recording where the visual set scores 43.9 (measured,
    `tools/libero/libero_camera_gate.py`): the collision primitives are opaque
    and they are in front.

    ⚠ `GEOM_IDX_RAY_VISIBLE` STILL APPLIES ON TOP OF THE MASK. It is MuJoCo's
    "alpha == 0" rule, precomputed by the parser, and a geom the model itself
    calls invisible should not appear because its group was asked for.
    """
    from noeira.render.stl_loader import load_stl
    from noeira.io.png import load_png_file

    var vis = VisualModel[DTYPE]()
    var ngeom_model = len(fmd.geoms)

    # ── 1. which geoms ───────────────────────────────────────────────────
    var keep = List[Int]()
    for g in range(ngeom_model):
        var gd = fmd.geoms[g]
        if (group_mask >> _clampg(gd.group)) & 1 == 0:
            continue
        if m.geoms.data[g * MODEL_GEOM_SIZE + GEOM_IDX_RAY_VISIBLE] == 0:
            continue
        keep.append(g)
    vis.ngeom = len(keep)
    vis.src_geom = keep.copy()
    if vis.ngeom == 0:
        raise Error(
            "build_visual_model: group mask " + String(group_mask)
            + " keeps NO geom of " + String(ngeom_model) + ". A camera over an"
            " empty scene renders the background and looks like a broken"
            " shader; say which groups the model draws in."
        )

    # ── 2. the meshes those geoms name ───────────────────────────────────
    #
    # Keyed on the file AND the frame, like `hull_cache_path` is: one asset at
    # two scales (or two `refquat`s) is two different surfaces, and sharing a
    # soup between them would draw one of them wrong.
    var mesh_key = List[String]()
    var geom_mesh = List[Int](length=vis.ngeom, fill=-1)
    var tri = List[Scalar[DTYPE]]()
    var uv = List[Scalar[DTYPE]]()
    var triadr = List[Int]()
    var trinum = List[Int]()
    var mesh_half = List[Scalar[DTYPE]]()

    for k in range(vis.ngeom):
        var gd = fmd.geoms[keep[k]]
        if gd.geom_type != GEOM_MESH or gd.mesh_filename.byte_length() == 0:
            continue
        var key = (
            gd.mesh_filename + "|" + String(gd.mesh_scale_x) + ","
            + String(gd.mesh_scale_y) + "," + String(gd.mesh_scale_z) + "|"
            + String(gd.mesh_ref_pos_x) + "," + String(gd.mesh_ref_pos_y) + ","
            + String(gd.mesh_ref_pos_z) + "," + String(gd.mesh_ref_quat_w)
            + "," + String(gd.mesh_ref_quat_x) + ","
            + String(gd.mesh_ref_quat_y) + "," + String(gd.mesh_ref_quat_z)
        )
        var found = -1
        for i in range(len(mesh_key)):
            if mesh_key[i] == key:
                found = i
                break
        if found >= 0:
            geom_mesh[k] = found
            continue

        # ⚠ SCALE 1 INTO THE LOADER, THEN `apply_mesh_ref_transform` WITH THE
        # SCALE — the order MuJoCo's `ApplyTransformations` uses (refpos,
        # then the INVERSE refquat, then scale). Handing `load_stl` the scale
        # and the transform a scale of 1 puts a non-identity `refquat` on the
        # wrong side of a non-uniform scale.
        var md = load_stl(gd.mesh_filename, 1.0, 1.0, 1.0)
        var nvtx = len(md.vertices)
        var flat = List[Scalar[DTYPE]](length=nvtx * 3, fill=Scalar[DTYPE](0))
        for i in range(nvtx):
            flat[i * 3 + 0] = Scalar[DTYPE](Float64(md.vertices[i].px))
            flat[i * 3 + 1] = Scalar[DTYPE](Float64(md.vertices[i].py))
            flat[i * 3 + 2] = Scalar[DTYPE](Float64(md.vertices[i].pz))
        apply_mesh_ref_transform[DTYPE](
            flat, nvtx,
            gd.mesh_ref_pos_x, gd.mesh_ref_pos_y, gd.mesh_ref_pos_z,
            gd.mesh_ref_quat_w, gd.mesh_ref_quat_x,
            gd.mesh_ref_quat_y, gd.mesh_ref_quat_z,
            gd.mesh_scale_x, gd.mesh_scale_y, gd.mesh_scale_z,
        )

        var adr = len(tri) // MESH_ARENA_RECORD
        var ntri_here = len(md.indices) // 3
        var hx = Scalar[DTYPE](0)
        var hy = Scalar[DTYPE](0)
        var hz = Scalar[DTYPE](0)
        for t in range(ntri_here):
            for c in range(3):
                var vi = Int(md.indices[t * 3 + c])
                var x = flat[vi * 3 + 0]
                var y = flat[vi * 3 + 1]
                var z = flat[vi * 3 + 2]
                tri.append(x)
                tri.append(y)
                tri.append(z)
                uv.append(Scalar[DTYPE](Float64(md.vertices[vi].u)))
                uv.append(Scalar[DTYPE](Float64(md.vertices[vi].v)))
                hx = hx if hx > abs(x) else abs(x)
                hy = hy if hy > abs(y) else abs(y)
                hz = hz if hz > abs(z) else abs(z)
        mesh_key.append(key)
        triadr.append(adr)
        trinum.append(ntri_here)
        # ⚠ THE HALF-EXTENTS ARE THE RAY'S FIRST REJECT (`ray_mesh` runs
        # `ray_box` before any triangle), so an under-sized box loses hits
        # SILENTLY. Taken from the soup itself, which is the surface being
        # tested, rather than from `Model`'s hull-derived `geom_size`.
        mesh_half.append(hx)
        mesh_half.append(hy)
        mesh_half.append(hz)
        geom_mesh[k] = len(mesh_key) - 1

    vis.nmesh = len(mesh_key)
    vis.ntri = len(tri) // MESH_ARENA_RECORD

    # ── 3. the BVH, into the same arena ──────────────────────────────────
    var bvh = List[Scalar[DTYPE]]()
    var bvhadr = List[Int]()
    var bvhnum = List[Int]()
    if vis.ntri > 0:
        # `bvh_sah=False` is the reference's median-split tree — the same
        # pictures, only slower; kept so a benchmark can time both.
        build_mesh_bvh[DTYPE](
            tri, triadr, trinum, bvh, bvhadr, bvhnum, sah=bvh_sah
        )
    else:
        for _ in range(vis.nmesh):
            bvhadr.append(0)
            bvhnum.append(0)

    # ── 4. the textures those geoms' materials name ──────────────────────
    #
    # Only the ones a KEPT geom can reach. libero_goal declares 18 and every
    # one is reachable; a model whose collision half carried its own material
    # would otherwise pay for texels no ray can sample.
    var tex_want = List[Bool](length=len(fmd.textures), fill=False)
    for k in range(vis.ngeom):
        var mid = fmd.geoms[keep[k]].material_id
        if mid < 0 or mid >= len(fmd.materials):
            continue
        var tid = fmd.materials[mid].tex_id
        if tid >= 0 and tid < len(fmd.textures):
            tex_want[tid] = True

    var tex_slot = List[Int](length=len(fmd.textures), fill=-1)
    var texels = List[UInt8]()
    # ⚠⚠ sRGB TEXELS ARE DECODED TO LINEAR HERE, ONCE, BEFORE ANY FILTERING.
    # A PNG with an `sRGB` chunk compiles to `mjCOLORSPACE_SRGB` under
    # `colorspace="auto"` (`mjCTexture::Load2D`), and `render_context.c`
    # uploads it as `GL_SRGB8`: the GPU turns each texel linear BEFORE the
    # bilinear filter, and the framebuffer is not sRGB, so nothing encodes it
    # back. LIBERO's floor tile is such a PNG. Decoding into 8-bit linear
    # quantises only below sRGB ~12, where the linear value is under 1/255 —
    # the same resolution the 8-bit output has, so no picture loses anything.
    var srgb_lut = List[UInt8](length=256, fill=UInt8(0))
    for i in range(256):
        var c = Float64(i) / 255.0
        var lin = c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
        srgb_lut[i] = UInt8(Int(lin * 255.0 + 0.5))
    var tex_rows = List[Scalar[DTYPE]]()
    for t in range(len(fmd.textures)):
        if not tex_want[t]:
            continue
        var td = fmd.textures[t]
        var w = td.width
        var h = td.height
        var adr = len(texels)
        if td.file.byte_length() > 0:
            var img = load_png_file(td.file)
            w = img.width
            h = img.height
            texels.reserve(len(texels) + w * h * 3)
            var nc = img.channels
            # `<texture colorspace>`: 1 linear, 2 sRGB, 0 auto = the file's
            # own `sRGB` chunk. The same rule as `render.png_loader`.
            var srgb = img.srgb
            if td.colorspace == 1:
                srgb = False
            elif td.colorspace == 2:
                srgb = True
            for i in range(w * h):
                var r = img.pixels[i * nc + 0]
                var gch = img.pixels[i * nc + 1 if nc >= 3 else i * nc]
                var b = img.pixels[i * nc + 2 if nc >= 3 else i * nc]
                if srgb:
                    r = srgb_lut[Int(r)]
                    gch = srgb_lut[Int(gch)]
                    b = srgb_lut[Int(b)]
                texels.append(r)
                texels.append(gch)
                texels.append(b)
        elif td.builtin != TEX_BUILTIN_NONE:
            var blk = List[UInt8](length=w * h * 3, fill=UInt8(0))
            _builtin_texels(
                td.builtin, td.mark, w, h,
                td.rgb1_r, td.rgb1_g, td.rgb1_b,
                td.rgb2_r, td.rgb2_g, td.rgb2_b,
                blk,
            )
            for i in range(len(blk)):
                texels.append(blk[i])
        else:
            # A `<texture>` with neither a file nor a builtin. MuJoCo would
            # have refused at compile time; here it becomes a flat mid grey
            # rather than a zero-size row the sampler would divide by.
            w = 1
            h = 1
            texels.append(UInt8(128))
            texels.append(UInt8(128))
            texels.append(UInt8(128))
        if len(tex_rows) // VIS_TEX_WORDS >= MAX_VIS_TEXTURES:
            raise Error(
                "build_visual_model: more than " + String(MAX_VIS_TEXTURES)
                + " textures reachable from the visible geoms. Raise"
                " MAX_VIS_TEXTURES in raytrace/visual_records.mojo — it sizes"
                " one small table."
            )
        var nlevels = append_mip_chain(texels, adr, w, h)
        tex_slot[t] = len(tex_rows) // VIS_TEX_WORDS
        var row = List[Scalar[DTYPE]](
            length=VIS_TEX_WORDS, fill=Scalar[DTYPE](0)
        )
        row[TEX_IDX_ADR] = Scalar[DTYPE](adr)
        row[TEX_IDX_WIDTH] = Scalar[DTYPE](w)
        row[TEX_IDX_HEIGHT] = Scalar[DTYPE](h)
        row[TEX_IDX_TYPE] = Scalar[DTYPE](td.tex_type)
        row[TEX_IDX_ACTIVE] = Scalar[DTYPE](1)
        row[TEX_IDX_NCHAN] = Scalar[DTYPE](3)
        row[TEX_IDX_NLEVELS] = Scalar[DTYPE](nlevels)
        for i in range(VIS_TEX_WORDS):
            tex_rows.append(row[i])
    vis.ntex = len(tex_rows) // VIS_TEX_WORDS
    vis.ntexel_bytes = len(texels)

    # ── 5. materials ─────────────────────────────────────────────────────
    var nmat = len(fmd.materials)
    if nmat > MAX_VIS_MATERIALS:
        raise Error(
            "build_visual_model: " + String(nmat) + " materials and"
            " MAX_VIS_MATERIALS is " + String(MAX_VIS_MATERIALS)
            + ". A geom pointing past the table would render black."
        )
    var mat_rows = List[Scalar[DTYPE]](
        length=_pos1(nmat) * VIS_MAT_WORDS, fill=Scalar[DTYPE](0)
    )
    for i in range(nmat):
        var md2 = fmd.materials[i]
        var o = i * VIS_MAT_WORDS
        mat_rows[o + MAT_IDX_R] = Scalar[DTYPE](md2.rgba_r)
        mat_rows[o + MAT_IDX_G] = Scalar[DTYPE](md2.rgba_g)
        mat_rows[o + MAT_IDX_B] = Scalar[DTYPE](md2.rgba_b)
        mat_rows[o + MAT_IDX_A] = Scalar[DTYPE](md2.rgba_a)
        var tid = md2.tex_id
        mat_rows[o + MAT_IDX_TEXID] = Scalar[DTYPE](
            tex_slot[tid] if tid >= 0 and tid < len(tex_slot) else -1
        )
        mat_rows[o + MAT_IDX_TEXREPEAT_U] = Scalar[DTYPE](md2.texrepeat_u)
        mat_rows[o + MAT_IDX_TEXREPEAT_V] = Scalar[DTYPE](md2.texrepeat_v)
        mat_rows[o + MAT_IDX_TEXUNIFORM] = Scalar[DTYPE](
            1 if md2.texuniform else 0
        )
        mat_rows[o + MAT_IDX_SPECULAR] = Scalar[DTYPE](md2.specular)
        mat_rows[o + MAT_IDX_SHININESS] = Scalar[DTYPE](md2.shininess)
        mat_rows[o + MAT_IDX_REFLECTANCE] = Scalar[DTYPE](md2.reflectance)
        mat_rows[o + MAT_IDX_ACTIVE] = Scalar[DTYPE](1)
    vis.nmat = nmat

    # ── 6. lights, the headlight first ───────────────────────────────────
    #
    # ⚠⚠ `mjv_makeLights` PUTS THE HEADLIGHT AT INDEX 0 AND SO DOES THIS.
    # It is directional, along the camera's gaze, and it carries
    # `mjModel.vis.headlight`'s colours, which `FlatModelDef` carries in
    # full (ambient, diffuse, specular, active), each at MuJoCo's default when
    # the model declares no `<headlight>`. The row is still WRITTEN when the
    # headlight is off, with `LIGHT_IDX_ACTIVE` 0, so index 0 keeps meaning
    # "the headlight" for every reader.
    var nlight = len(fmd.lights) + 1
    if nlight > MAX_VIS_LIGHTS:
        raise Error(
            "build_visual_model: " + String(nlight)
            + " lights (the headlight included) and MAX_VIS_LIGHTS is "
            + String(MAX_VIS_LIGHTS) + "."
        )
    var light_rows = List[Scalar[DTYPE]](
        length=MAX_VIS_LIGHTS * VIS_LIGHT_WORDS, fill=Scalar[DTYPE](0)
    )
    light_rows[LIGHT_IDX_BODY] = Scalar[DTYPE](LIGHT_BODY_HEADLIGHT)
    light_rows[LIGHT_IDX_DIRECTIONAL] = Scalar[DTYPE](1)
    light_rows[LIGHT_IDX_CASTSHADOW] = Scalar[DTYPE](0)
    light_rows[LIGHT_IDX_CUTOFF] = Scalar[DTYPE](180)
    light_rows[LIGHT_IDX_AMBIENT_R] = Scalar[DTYPE](fmd.vis_headlight_ambient_r)
    light_rows[LIGHT_IDX_AMBIENT_G] = Scalar[DTYPE](fmd.vis_headlight_ambient_g)
    light_rows[LIGHT_IDX_AMBIENT_B] = Scalar[DTYPE](fmd.vis_headlight_ambient_b)
    light_rows[LIGHT_IDX_DIFFUSE_R] = Scalar[DTYPE](fmd.vis_headlight_diffuse_r)
    light_rows[LIGHT_IDX_DIFFUSE_G] = Scalar[DTYPE](fmd.vis_headlight_diffuse_g)
    light_rows[LIGHT_IDX_DIFFUSE_B] = Scalar[DTYPE](fmd.vis_headlight_diffuse_b)
    light_rows[LIGHT_IDX_SPECULAR_R] = Scalar[DTYPE](fmd.vis_headlight_specular_r)
    light_rows[LIGHT_IDX_SPECULAR_G] = Scalar[DTYPE](fmd.vis_headlight_specular_g)
    light_rows[LIGHT_IDX_SPECULAR_B] = Scalar[DTYPE](fmd.vis_headlight_specular_b)
    light_rows[LIGHT_IDX_ACTIVE] = Scalar[DTYPE](
        1 if fmd.vis_headlight_active else 0
    )
    for i in range(len(fmd.lights)):
        var ld = fmd.lights[i]
        var o = (i + 1) * VIS_LIGHT_WORDS
        light_rows[o + LIGHT_IDX_BODY] = Scalar[DTYPE](ld.body_id)
        light_rows[o + LIGHT_IDX_POS_X] = Scalar[DTYPE](ld.pos_x)
        light_rows[o + LIGHT_IDX_POS_Y] = Scalar[DTYPE](ld.pos_y)
        light_rows[o + LIGHT_IDX_POS_Z] = Scalar[DTYPE](ld.pos_z)
        light_rows[o + LIGHT_IDX_DIR_X] = Scalar[DTYPE](ld.dir_x)
        light_rows[o + LIGHT_IDX_DIR_Y] = Scalar[DTYPE](ld.dir_y)
        light_rows[o + LIGHT_IDX_DIR_Z] = Scalar[DTYPE](ld.dir_z)
        light_rows[o + LIGHT_IDX_DIFFUSE_R] = Scalar[DTYPE](ld.diffuse_r)
        light_rows[o + LIGHT_IDX_DIFFUSE_G] = Scalar[DTYPE](ld.diffuse_g)
        light_rows[o + LIGHT_IDX_DIFFUSE_B] = Scalar[DTYPE](ld.diffuse_b)
        light_rows[o + LIGHT_IDX_SPECULAR_R] = Scalar[DTYPE](ld.specular_r)
        light_rows[o + LIGHT_IDX_SPECULAR_G] = Scalar[DTYPE](ld.specular_g)
        light_rows[o + LIGHT_IDX_SPECULAR_B] = Scalar[DTYPE](ld.specular_b)
        light_rows[o + LIGHT_IDX_AMBIENT_R] = Scalar[DTYPE](ld.ambient_r)
        light_rows[o + LIGHT_IDX_AMBIENT_G] = Scalar[DTYPE](ld.ambient_g)
        light_rows[o + LIGHT_IDX_AMBIENT_B] = Scalar[DTYPE](ld.ambient_b)
        light_rows[o + LIGHT_IDX_DIRECTIONAL] = Scalar[DTYPE](
            1 if ld.directional else 0
        )
        light_rows[o + LIGHT_IDX_CASTSHADOW] = Scalar[DTYPE](
            1 if ld.castshadow else 0
        )
        light_rows[o + LIGHT_IDX_CUTOFF] = Scalar[DTYPE](ld.cutoff)
        light_rows[o + LIGHT_IDX_ACTIVE] = Scalar[DTYPE](1)
    vis.nlight = nlight

    # ── 6b. WHICH geom reflects — at most one ────────────────────────────
    #
    # `mjr_render`'s own precedence, applied once here rather than re-derived
    # per pixel: the FIRST geom that is a plane or a box, opaque, and carries
    # a positive `mat_reflectance` is the mirror, and every later candidate
    # has its reflectance zeroed. See `APP_IDX_REFLECT`.
    var refl_of = List[Float64](length=vis.ngeom, fill=0.0)
    var mirror = -1
    for k in range(vis.ngeom):
        var gd = fmd.geoms[keep[k]]
        if gd.geom_type != GEOM_PLANE and gd.geom_type != GEOM_BOX:
            continue
        var mid = gd.material_id
        if mid < 0 or mid >= len(fmd.materials):
            continue
        var md3 = fmd.materials[mid]
        if md3.reflectance <= 0:
            continue
        # `!geom->transparent` — MuJoCo marks a geom transparent from its
        # alpha, and a see-through mirror is not one.
        if Float64(m.geom_rgba.data[keep[k] * 4 + 3]) < 1.0:
            continue
        if mirror < 0:
            mirror = k
            refl_of[k] = md3.reflectance
        # every later candidate keeps 0

    # ── 7. the geom records ──────────────────────────────────────────────
    var grows = List[Scalar[DTYPE]](
        length=vis.ngeom * MODEL_GEOM_SIZE, fill=Scalar[DTYPE](0)
    )
    var arows = List[Scalar[DTYPE]](
        length=vis.ngeom * VIS_GEOM_APPEARANCE, fill=Scalar[DTYPE](0)
    )
    for k in range(vis.ngeom):
        var g = keep[k]
        var gd = fmd.geoms[g]
        var src = g * MODEL_GEOM_SIZE
        var dst = k * MODEL_GEOM_SIZE
        for i in range(MODEL_GEOM_SIZE):
            grows[dst + i] = m.geoms.data[src + i]
        # ⚠⚠ THE POSE GOES BACK TO THE XML'S. `fields_build` composes a mesh
        # geom's pose with the mesh's PRINCIPAL frame because the hull it
        # stores lives there; our soup lives in the file's frame, so the two
        # would be off by the mesh's centre of mass and its principal
        # rotation. Writing `gd.pos`/`gd.quat` is a no-op for every other geom
        # type — `fields_build` writes exactly these and then only the mesh
        # branch overwrites them.
        grows[dst + GEOM_IDX_POS_X] = Scalar[DTYPE](gd.pos_x)
        grows[dst + GEOM_IDX_POS_Y] = Scalar[DTYPE](gd.pos_y)
        grows[dst + GEOM_IDX_POS_Z] = Scalar[DTYPE](gd.pos_z)
        grows[dst + GEOM_IDX_QUAT_X] = Scalar[DTYPE](gd.quat_x)
        grows[dst + GEOM_IDX_QUAT_Y] = Scalar[DTYPE](gd.quat_y)
        grows[dst + GEOM_IDX_QUAT_Z] = Scalar[DTYPE](gd.quat_z)
        grows[dst + GEOM_IDX_QUAT_W] = Scalar[DTYPE](gd.quat_w)
        grows[dst + GEOM_IDX_RAY_VISIBLE] = Scalar[DTYPE](1)
        var vm = geom_mesh[k]
        grows[dst + GEOM_IDX_MESH_ID] = Scalar[DTYPE](vm)
        if vm >= 0:
            grows[dst + GEOM_IDX_HALF_X] = mesh_half[vm * 3 + 0]
            grows[dst + GEOM_IDX_HALF_Y] = mesh_half[vm * 3 + 1]
            grows[dst + GEOM_IDX_HALF_Z] = mesh_half[vm * 3 + 2]

        var ao = k * VIS_GEOM_APPEARANCE
        arows[ao + APP_IDX_R] = m.geom_rgba.data[g * 4 + 0]
        arows[ao + APP_IDX_G] = m.geom_rgba.data[g * 4 + 1]
        arows[ao + APP_IDX_B] = m.geom_rgba.data[g * 4 + 2]
        arows[ao + APP_IDX_A] = m.geom_rgba.data[g * 4 + 3]
        arows[ao + APP_IDX_MATID] = Scalar[DTYPE](gd.material_id)
        arows[ao + APP_IDX_UVADR] = Scalar[DTYPE](
            triadr[vm] if vm >= 0 else -1
        )
        arows[ao + APP_IDX_REFLECT] = Scalar[DTYPE](refl_of[k])

    # ── 8. the mesh table ────────────────────────────────────────────────
    var mrows = List[Scalar[DTYPE]](
        length=_pos1(vis.nmesh) * MODEL_MESH_META_SIZE, fill=Scalar[DTYPE](0)
    )
    for i in range(vis.nmesh):
        var o = i * MODEL_MESH_META_SIZE
        mrows[o + MESH_META_IDX_TRIADR] = Scalar[DTYPE](triadr[i])
        mrows[o + MESH_META_IDX_TRINUM] = Scalar[DTYPE](trinum[i])
        mrows[o + MESH_META_IDX_BVHADR] = Scalar[DTYPE](bvhadr[i])
        mrows[o + MESH_META_IDX_BVHNUM] = Scalar[DTYPE](bvhnum[i])

    # ── 8b. conditional sites, after every ordinary geom ─────────────────
    #
    # A site becomes a geom record of its own type, size and pose, and an
    # appearance row with its rgba and no material — MuJoCo draws a site with
    # the `mjvGeom` defaults, specular and shininess 0.5, which is exactly
    # what `shade_hit` uses for a geom with no material.
    var qadr_of = List[Int]()
    var qa = 0
    for j in range(len(fmd.joints)):
        qadr_of.append(qa)
        qa += fmd.joints[j].nq
    for c in range(len(conditions)):
        var si = -1
        for k in range(len(fmd.site_names)):
            if fmd.site_names[k] == conditions[c].site:
                si = k
                break
        if si < 0:
            raise Error(
                "build_visual_model: no site '" + conditions[c].site
                + "' for a SiteCondition — the camera would never show it"
            )
        var qadr = -1
        if conditions[c].joint.byte_length() > 0:
            for j in range(len(fmd.joint_names)):
                if fmd.joint_names[j] == conditions[c].joint:
                    qadr = qadr_of[j]
                    break
            if qadr < 0:
                raise Error(
                    "build_visual_model: no joint '" + conditions[c].joint
                    + "' for the condition on site '" + conditions[c].site + "'"
                )
        var sd = fmd.sites[si]
        var row = List[Scalar[DTYPE]](length=MODEL_GEOM_SIZE, fill=Scalar[DTYPE](0))
        row[GEOM_IDX_BODY] = Scalar[DTYPE](sd.body_id)
        row[GEOM_IDX_TYPE] = Scalar[DTYPE](sd.site_type)
        row[GEOM_IDX_POS_X] = Scalar[DTYPE](sd.pos_x)
        row[GEOM_IDX_POS_Y] = Scalar[DTYPE](sd.pos_y)
        row[GEOM_IDX_POS_Z] = Scalar[DTYPE](sd.pos_z)
        row[GEOM_IDX_QUAT_X] = Scalar[DTYPE](sd.quat_x)
        row[GEOM_IDX_QUAT_Y] = Scalar[DTYPE](sd.quat_y)
        row[GEOM_IDX_QUAT_Z] = Scalar[DTYPE](sd.quat_z)
        row[GEOM_IDX_QUAT_W] = Scalar[DTYPE](sd.quat_w)
        # The size in `ray_model`'s per-type spelling (see its dispatch):
        # sphere radius, capsule/cylinder radius + half-length, box and
        # ellipsoid half-extents.
        row[GEOM_IDX_RADIUS] = Scalar[DTYPE](sd.size_0)
        row[GEOM_IDX_HALF_LENGTH] = Scalar[DTYPE](sd.size_1)
        row[GEOM_IDX_HALF_X] = Scalar[DTYPE](sd.size_0)
        row[GEOM_IDX_HALF_Y] = Scalar[DTYPE](sd.size_1 if sd.size_1 > 0 else sd.size_0)
        row[GEOM_IDX_HALF_Z] = Scalar[DTYPE](sd.size_2 if sd.size_2 > 0 else sd.size_0)
        row[GEOM_IDX_RAY_VISIBLE] = Scalar[DTYPE](1)
        row[GEOM_IDX_MESH_ID] = Scalar[DTYPE](-1)
        for i in range(MODEL_GEOM_SIZE):
            grows.append(row[i])
        var ap = List[Scalar[DTYPE]](length=VIS_GEOM_APPEARANCE, fill=Scalar[DTYPE](0))
        ap[APP_IDX_R] = Scalar[DTYPE](sd.rgba_r)
        ap[APP_IDX_G] = Scalar[DTYPE](sd.rgba_g)
        ap[APP_IDX_B] = Scalar[DTYPE](sd.rgba_b)
        ap[APP_IDX_A] = Scalar[DTYPE](sd.rgba_a)
        ap[APP_IDX_MATID] = Scalar[DTYPE](-1)
        ap[APP_IDX_UVADR] = Scalar[DTYPE](-1)
        ap[APP_IDX_COND_QADR] = Scalar[DTYPE](qadr)
        ap[APP_IDX_COND_MIN] = Scalar[DTYPE](conditions[c].min_qpos)
        for i in range(VIS_GEOM_APPEARANCE):
            arows.append(ap[i])
    vis.ncond = len(conditions)

    # ── 9. into the tensors ──────────────────────────────────────────────
    for i in range(len(bvh)):
        tri.append(bvh[i])

    vis.geoms = _tensor_from[DTYPE](grows)
    vis.appearance = _tensor_from[DTYPE](arows)
    vis.mesh_meta = _tensor_from[DTYPE](mrows)
    vis.mesh_tris = _tensor_from[DTYPE](tri, _pos1(len(tri)))
    vis.mesh_uv = _tensor_from[DTYPE](uv, _pos1(len(uv)))
    vis.materials = _tensor_from[DTYPE](mat_rows)
    vis.textures = _tensor_from[DTYPE](
        tex_rows, _pos1(MAX_VIS_TEXTURES * VIS_TEX_WORDS)
    )
    vis.lights = _tensor_from[DTYPE](light_rows)
    var tt = TensorImpl[DType.uint8].alloc(_pos1(len(texels)))
    for i in range(len(texels)):
        tt.data[i] = texels[i]
    vis.texels = tt^

    vis.mirror = mirror
    if verbose:
        print("  " + vis.describe())
    return vis^


def _tensor_from[
    DTYPE: DType
](src: List[Scalar[DTYPE]], pad_to: Int = 0) raises -> TensorImpl[DTYPE]:
    """A `TensorImpl` holding `src`, at least `pad_to` long.

    ⚠ THE PAD IS `_at_least_one`'S RULE AND A LITTLE MORE. A zero-length
    device buffer is not allocatable, and a table the kernel indexes by a
    RUNTIME id (the texture table) must be its full comptime height even when
    the model filled two rows — otherwise a lane reading row 5 of a two-row
    buffer walks off the end.
    """
    var n = len(src)
    var t = TensorImpl[DTYPE].alloc(n if n > pad_to else pad_to)
    for i in range(n):
        t.data[i] = src[i]
    return t^


def visual_model_from_model[
    DTYPE: DType, D: DimsLike
](
    mut m: Model[DTYPE, D],
    light_dir: List[Float64],
) raises -> VisualModel[DTYPE]:
    """A `VisualModel` for a caller that has only a built `Model`.

    Every geom the model calls visible, its own `geom_rgba`, the meshes
    `Model` already carries, no materials, no textures, and ONE directional
    light plus the headlight.

    ⚠⚠ THIS IS THE PRE-L5 PICTURE, KEPT WORKING — NOT THE FAITHFUL ONE. A
    `Model` has no `<material>`, no `<texture>` and no `<light>`: they are
    parsed into `FlatModelDef` and dropped at that boundary. A caller that
    wants LIBERO's pixels must build from the parse
    (`build_visual_model`); a caller that just wants a picture of a
    half-cheetah gets this, and it is what every camera in this tree got
    before L5 — one light, flat colours — now expressed in the reference's
    lighting equation instead of a hemispheric stand-in.

    ⚠ THE MESH TABLES ARE `Model`'S OWN, SHARED BY VALUE. A `Model` mesh soup
    is the COLLIDABLE meshes only, in the mesh's principal frame, and
    `Model.geoms` poses match that frame — so this path copies the geom
    records verbatim rather than putting the XML's pose back, which would be
    the wrong half of the pair.
    """
    var vis = VisualModel[DTYPE]()
    var ng = m.dims.get_ngeom()
    var keep = List[Int]()
    for g in range(ng):
        if m.geoms.data[g * MODEL_GEOM_SIZE + GEOM_IDX_RAY_VISIBLE] == 0:
            continue
        keep.append(g)
    vis.ngeom = len(keep)
    vis.src_geom = keep.copy()

    var grows = List[Scalar[DTYPE]](
        length=_pos1(vis.ngeom * MODEL_GEOM_SIZE), fill=Scalar[DTYPE](0)
    )
    var arows = List[Scalar[DTYPE]](
        length=_pos1(vis.ngeom * VIS_GEOM_APPEARANCE), fill=Scalar[DTYPE](0)
    )
    for k in range(vis.ngeom):
        var g = keep[k]
        for i in range(MODEL_GEOM_SIZE):
            grows[k * MODEL_GEOM_SIZE + i] = m.geoms.data[
                g * MODEL_GEOM_SIZE + i
            ]
        var ao = k * VIS_GEOM_APPEARANCE
        arows[ao + APP_IDX_R] = m.geom_rgba.data[g * 4 + 0]
        arows[ao + APP_IDX_G] = m.geom_rgba.data[g * 4 + 1]
        arows[ao + APP_IDX_B] = m.geom_rgba.data[g * 4 + 2]
        arows[ao + APP_IDX_A] = m.geom_rgba.data[g * 4 + 3]
        arows[ao + APP_IDX_MATID] = Scalar[DTYPE](-1)
        arows[ao + APP_IDX_UVADR] = Scalar[DTYPE](-1)

    var nmesh_meta = len(m.mesh_meta.data)
    var mrows = List[Scalar[DTYPE]](
        length=_pos1(nmesh_meta), fill=Scalar[DTYPE](0)
    )
    for i in range(nmesh_meta):
        mrows[i] = m.mesh_meta.data[i]
    var ntri_arena = len(m.mesh_tris.data)
    var trows = List[Scalar[DTYPE]](
        length=_pos1(ntri_arena), fill=Scalar[DTYPE](0)
    )
    for i in range(ntri_arena):
        trows[i] = m.mesh_tris.data[i]
    vis.nmesh = nmesh_meta // MODEL_MESH_META_SIZE
    vis.ntri = ntri_arena // MESH_ARENA_RECORD

    var light_rows = List[Scalar[DTYPE]](
        length=MAX_VIS_LIGHTS * VIS_LIGHT_WORDS, fill=Scalar[DTYPE](0)
    )
    light_rows[LIGHT_IDX_BODY] = Scalar[DTYPE](LIGHT_BODY_HEADLIGHT)
    light_rows[LIGHT_IDX_DIRECTIONAL] = Scalar[DTYPE](1)
    light_rows[LIGHT_IDX_CUTOFF] = Scalar[DTYPE](180)
    light_rows[LIGHT_IDX_AMBIENT_R] = Scalar[DTYPE](0.1)
    light_rows[LIGHT_IDX_AMBIENT_G] = Scalar[DTYPE](0.1)
    light_rows[LIGHT_IDX_AMBIENT_B] = Scalar[DTYPE](0.1)
    # No parse here, so MuJoCo's default headlight (`mjv_defaultVisual`).
    light_rows[LIGHT_IDX_DIFFUSE_R] = Scalar[DTYPE](0.4)
    light_rows[LIGHT_IDX_DIFFUSE_G] = Scalar[DTYPE](0.4)
    light_rows[LIGHT_IDX_DIFFUSE_B] = Scalar[DTYPE](0.4)
    light_rows[LIGHT_IDX_SPECULAR_R] = Scalar[DTYPE](0.5)
    light_rows[LIGHT_IDX_SPECULAR_G] = Scalar[DTYPE](0.5)
    light_rows[LIGHT_IDX_SPECULAR_B] = Scalar[DTYPE](0.5)
    light_rows[LIGHT_IDX_ACTIVE] = Scalar[DTYPE](1)
    var o = VIS_LIGHT_WORDS
    light_rows[o + LIGHT_IDX_DIRECTIONAL] = Scalar[DTYPE](1)
    light_rows[o + LIGHT_IDX_CASTSHADOW] = Scalar[DTYPE](1)
    light_rows[o + LIGHT_IDX_CUTOFF] = Scalar[DTYPE](180)
    light_rows[o + LIGHT_IDX_DIR_X] = Scalar[DTYPE](light_dir[0])
    light_rows[o + LIGHT_IDX_DIR_Y] = Scalar[DTYPE](light_dir[1])
    light_rows[o + LIGHT_IDX_DIR_Z] = Scalar[DTYPE](light_dir[2])
    light_rows[o + LIGHT_IDX_DIFFUSE_R] = Scalar[DTYPE](0.8)
    light_rows[o + LIGHT_IDX_DIFFUSE_G] = Scalar[DTYPE](0.8)
    light_rows[o + LIGHT_IDX_DIFFUSE_B] = Scalar[DTYPE](0.8)
    light_rows[o + LIGHT_IDX_SPECULAR_R] = Scalar[DTYPE](0.3)
    light_rows[o + LIGHT_IDX_SPECULAR_G] = Scalar[DTYPE](0.3)
    light_rows[o + LIGHT_IDX_SPECULAR_B] = Scalar[DTYPE](0.3)
    light_rows[o + LIGHT_IDX_ACTIVE] = Scalar[DTYPE](1)
    vis.nlight = 2

    vis.geoms = _tensor_from[DTYPE](grows)
    vis.appearance = _tensor_from[DTYPE](arows)
    vis.mesh_meta = _tensor_from[DTYPE](mrows)
    vis.mesh_tris = _tensor_from[DTYPE](trows)
    vis.mesh_uv = _tensor_from[DTYPE](List[Scalar[DTYPE]](), 1)
    vis.materials = _tensor_from[DTYPE](
        List[Scalar[DTYPE]](), MAX_VIS_MATERIALS * VIS_MAT_WORDS
    )
    vis.textures = _tensor_from[DTYPE](
        List[Scalar[DTYPE]](), MAX_VIS_TEXTURES * VIS_TEX_WORDS
    )
    vis.lights = _tensor_from[DTYPE](light_rows)
    vis.texels = TensorImpl[DType.uint8].alloc(1)
    return vis^
