"""The kernel and the host that owns its buffers.

One thread per `(env, pixel)`. No shared memory, no per-thread array, no
window, no swapchain, no draw command — the whole scene is the batched `Data`
the physics already wrote, read in place.

⚠ `SHADOWS` AND `REFLECT` ARE COMPTIME PARAMETERS BECAUSE EACH IS A SECOND
FULL `ray_model` PER PIXEL. `REFLECT` defaults to TRUE because
`mjRND_REFLECTION` does; the runtime cost on a scene with no reflective geom
is one compare against a zero in the appearance row, but the CODE is in the
kernel, so a caller that knows it has no mirror can compile it out.

⚠ `SHADOWS` IS A COMPTIME PARAMETER BECAUSE THE SHADOW RAY IS A SECOND FULL
`ray_model` PER PIXEL. A runtime flag would leave that code in the kernel and
still pay for it on every miss; as a parameter, `BatchedCameraRenderer[...,
SHADOWS=False]` compiles a kernel that does not contain it. See
`benchmarks/camera_tracer_lift_brick.mojo` for what it costs on a mesh-heavy
scene.

⚠ AN EARLIER VERSION OF THIS NOTE CLAIMED A ~550 s METAL COMPILE FOR THIS
KERNEL. THAT WAS WRONG, AND WRONG TWICE: the number came from a test harness
line that reports MILLISECONDS, and it was the test's RUNTIME rather than a
compile. Measured properly — edit this file, run the gate, wall-clock the
command — a full recompile of the camera kernel is about **8 s**. The minutes
that produced the false reading were a cold cache over the whole package graph,
which any first build pays. `SHADOWS` earns its keep on RUN time, not compile
time.

⚠ RGB IS FLOAT, NOT PACKED `uint32`. The reference packs ABGR into a `uint32`
and ships an unpack kernel (`render_util.unpack_rgb_kernel`) because Warp's
render context stores every camera in one buffer and memory is the binding
constraint. Here the consumer is an `nn` module that wants `[0, 1]` floats, so
packing would be a kernel to compress followed by a kernel to undo it. The
price is real and worth stating: at 1024 envs and 84x84 an RGB frame is 87 MB
at float32 against 22 MB packed. If a batch ever needs the packed form, it is
an output-format parameter on this struct and nothing else changes — the pixel
function returns a `Vec3` either way.
"""

from std.math import ceildiv
from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor

from noeira.math3d import Vec3 as Vec3Generic

from ..fields import Data, Model
from ..fields.dims import DimsLike
from ..gpu.constants import (
    TPB,
    MODEL_GEOM_SIZE,
    MODEL_BODY_SIZE,
    MODEL_GEOM_RGBA_SIZE,
    MODEL_MESH_META_SIZE,
    MAX_GPU_MESHES,
    MESH_ARENA_FLOATS_PER_TRI,
    MODEL_HFIELD_META_SIZE,
    MAX_GPU_HFIELDS,
    MAX_GPU_CAMERAS,
    MODEL_CAM_SIZE,
    CAM_IDX_ACTIVE,
    CAM_IDX_MODE,
    CAM_IDX_FOVY,
    CAM_IDX_REF_SET,
)
from .camera import (
    camera_world_frame,
    RT_CAM_MODE_TRACK,
    RT_CAM_MODE_TRACKCOM,
)
from .render import render_pixel
from .cull import CULL_WORDS, geom_screen_rect
from ..fields.rt_layout import DYN1, DYN2, rl1, rl2
from .visual import VisualModel, visual_model_from_model
from .visual_records import (
    MAX_VIS_LIGHTS,
    MAX_VIS_MATERIALS,
    MAX_VIS_TEXTURES,
    VIS_GEOM_APPEARANCE,
    VIS_LIGHT_WORDS,
    VIS_MAT_WORDS,
    VIS_TEX_WORDS,
    VIS_UV_WORDS,
)

comptime RGB_CHANNELS: Int = 3


@always_inline
def _to_byte(x: Float64) -> Int:
    """[0, 1] -> [0, 255], clamped. `render_pixel` already clamps, so this is
    belt and braces against a caller that scaled the buffer."""
    var v = Int(x * 255.0 + 0.5)
    if v < 0:
        return 0
    if v > 255:
        return 255
    return v


@always_inline
def _pos(n: Int) -> Int:
    """`_at_least_one` — every tensor allocates one element even when unused."""
    return n if n > 0 else 1


def check_camera[DTYPE: DType, D: DimsLike](
    m: Model[DTYPE, D], cam: Int
) raises:
    """REFUSE a camera index that would render a wrong picture, not an error.

    An inactive row has `fovy = 0` (a degenerate frustum, one point), a
    tracking camera with no reference sits at its body's origin looking along
    world -Z, and an out-of-range index reads whatever is in the next row. A
    kernel cannot raise, so this host check is the only place any of them can
    be named. Run by the constructor on its camera and by `render` on any
    other camera it is asked for.
    """
    if cam < 0 or cam >= MAX_GPU_CAMERAS:
        raise Error(
            String("BatchedCameraRenderer: camera index ") + String(cam)
            + " is outside [0, " + String(MAX_GPU_CAMERAS) + ")."
        )
    var cb = cam * MODEL_CAM_SIZE
    if m.cameras.data[cb + CAM_IDX_ACTIVE] == 0:
        raise Error(
            String("BatchedCameraRenderer: camera ") + String(cam)
            + " is not a camera this model declares — the row is padding."
            " Its `fovy` is 0, which renders a single point rather than"
            " failing. Does the MJCF have a <camera> at this index?"
        )
    var fovy = m.cameras.data[cb + CAM_IDX_FOVY]
    if not (fovy > 0):
        raise Error(
            String("BatchedCameraRenderer: camera ") + String(cam)
            + " has fovy = " + String(Float64(fovy))
            + ", a degenerate frustum."
        )
    var mode = Int(m.cameras.data[cb + CAM_IDX_MODE])
    if mode == RT_CAM_MODE_TRACK or mode == RT_CAM_MODE_TRACKCOM:
        if m.cameras.data[cb + CAM_IDX_REF_SET] == 0:
            raise Error(
                String("BatchedCameraRenderer: camera ") + String(cam)
                + ' is mode="track"/"trackcom", which reads the reference'
                " pose MuJoCo's compiler bakes at qpos0, and that pose has"
                " not been taken. Call"
                " `raytrace.init_camera_reference(d, m)` after a reset's"
                " forward kinematics and subtree pass. Without it the"
                " camera renders from the body origin along world -Z,"
                " which is a picture and not an error."
            )


struct BatchedCameraRenderer[
    DTYPE: DType,
    D: DimsLike,
    BATCH: Int,
    WIDTH: Int,
    HEIGHT: Int,
    SHADOWS: Bool = True,
    REFLECT: Bool = True,
    SAMPLES: Int = 1,
](Movable):
    """RGB + depth + segmentation for one camera, over every lane.

    ⚠ ONE CAMERA PER RENDERER, BY CHOICE. The reference packs N cameras of
    different resolutions into one buffer and resolves `(camid, local pixel)`
    from a cumulative-size scan at the top of every thread. That scan is a loop
    over cameras in the innermost hot path, and it buys a case — cameras of
    DIFFERENT resolutions in one launch — that nothing here has. Two cameras
    are two renderers and two launches.

    ⚠ `SAMPLES = 4` IS WHAT A LIBERO (OR ANY DEFAULT MuJoCo) PICTURE IS: the
    offscreen buffer is 4x multisampled (`vis.quality.offsamples`), and one
    ray per pixel scores 28.9 dB against LIBERO's recording where MuJoCo's own
    4-sample render scores 43.1. It costs up to five traces a pixel, so it is
    a choice and not the default — see `render.render_pixel`.
    """

    comptime NPIX: Int = Self.WIDTH * Self.HEIGHT
    comptime NGEOM_F: Int = _pos(Self.D.NGEOM)
    comptime NMESH_TRI_F: Int = _pos(Self.D.NMESH_TRI)
    comptime NHF_F: Int = _pos(Self.D.NHFIELD_DATA)

    comptime L_GEOMS = Layout.row_major(Self.NGEOM_F, MODEL_GEOM_SIZE)
    comptime L_RGBA = Layout.row_major(Self.NGEOM_F * MODEL_GEOM_RGBA_SIZE)
    comptime L_BODIES = Layout.row_major(Self.D.NBODY, MODEL_BODY_SIZE)
    comptime L_B3 = Layout.row_major(Self.BATCH, Self.D.NBODY * 3)
    comptime L_B4 = Layout.row_major(Self.BATCH, Self.D.NBODY * 4)
    comptime L_QPOS = Layout.row_major(Self.BATCH, _pos(Self.D.NQ))
    comptime L_CAM = Layout.row_major(MAX_GPU_CAMERAS * MODEL_CAM_SIZE)
    comptime L_MESH_META = Layout.row_major(
        MAX_GPU_MESHES * MODEL_MESH_META_SIZE
    )
    comptime L_TRI = Layout.row_major(
        Self.NMESH_TRI_F * MESH_ARENA_FLOATS_PER_TRI
    )
    comptime L_HF_META = Layout.row_major(
        MAX_GPU_HFIELDS * MODEL_HFIELD_META_SIZE
    )
    comptime L_HF = Layout.row_major(Self.BATCH * Self.NHF_F)
    comptime L_RGB = Layout.row_major(Self.BATCH, Self.NPIX * RGB_CHANNELS)
    comptime L_SCALARPIX = Layout.row_major(Self.BATCH, Self.NPIX)
    # ── the appearance tables ────────────────────────────────────────────
    #
    # ⚠⚠ THESE BIND THROUGH `DYN1`/`DYN2` WHILE EVERY LAYOUT ABOVE IS
    # COMPTIME, and the reason is that a `VisualModel`'s sizes are not known
    # until the meshes and the PNGs have been read — they are not in `Dims`
    # and they cannot be, because `Dims` is the SOLVER's shape. A comptime
    # layout would have to be a parameter on this struct, i.e. a second whole
    # kernel compile per scene, and it would still be unspellable on the
    # dynamic dims provider the task layer runs on. `rt_layout.mojo` measured
    # runtime layouts at 0.80-0.93x of comptime ones on the Newton solve, so
    # the cost here is at worst nothing.

    var rgb: DeviceBuffer[Self.DTYPE]
    """`[BATCH, HEIGHT*WIDTH*3]`, row-major, channels interleaved, in [0, 1].

    ⚠ ROW 0 IS THE TOP OF THE IMAGE — see `camera_pixel_ray`. A consumer
    writing this straight to a PNG needs no flip; one feeding a convolution
    needs to know which end is the sky."""

    var depth: DeviceBuffer[Self.DTYPE]
    """`[BATCH, HEIGHT*WIDTH]` PLANAR depth in metres; **0 means no hit**.
    See `PixelHit.depth` — the sentinel is deliberate and a normaliser must
    handle it."""

    var seg: DeviceBuffer[Self.DTYPE]
    """`[BATCH, HEIGHT*WIDTH]` geom id, or -1 for background.

    ⚠ STORED AS `DTYPE`, NOT AN INTEGER TYPE, so the whole renderer stays one
    dtype and one kernel. Geom counts here are in the hundreds, far inside
    float32's exact-integer range; a model with more than 2^24 geoms would
    have other problems first."""

    var vis: VisualModel[Self.DTYPE]
    """The scene AS DRAWN — see `raytrace/visual.mojo`.

    Built from the `Model` alone by `__init__` (every visible geom, its own
    rgba, one directional light and the headlight) and REPLACED by
    `set_visual` when the caller has a parse to build the faithful one from.
    """

    var cam: Int
    var cull: DeviceBuffer[Self.DTYPE]
    """`[BATCH, ngeom, CULL_WORDS]` — each visual geom's screen rectangle for
    the camera of the current launch (`cull.geom_screen_rect`), written by a
    pre-pass in `render` and read by the primary rays."""
    var cull_cap: Int
    var cull_enabled: Bool
    """True by default. False renders every pixel against every geom — the
    same pictures (the cull only skips geoms a pixel cannot reach), slower;
    kept so a gate and a benchmark can compare the two in one binary."""
    var light_dir: Vec3Generic[Self.DTYPE]
    """The direction the light TRAVELS, as `mjModel.light_dir` is. The default
    points down and slightly forward, matching the key light `Renderer3D`
    uses, so a scene looks the same way up in both pipelines."""
    var background: Vec3Generic[Self.DTYPE]

    def __init__(
        out self,
        ctx: DeviceContext,
        mut m: Model[Self.DTYPE, Self.D],
        cam: Int = 0,
    ) raises:
        """Allocate the buffers and REFUSE a camera that would render wrong.

        ⚠⚠ THE CHECKS ARE THE POINT OF HAVING A HOST ENTRY POINT AT ALL. Every
        failure this guards against renders a picture rather than an error:
        an inactive row has `fovy = 0` (a degenerate frustum, one point), a
        tracking camera with no reference sits at its body's origin looking
        along world -Z, and an out-of-range index reads whatever is in the
        next row. A kernel cannot raise, so this is the only place any of
        them can be named.
        """
        comptime assert Self.WIDTH > 0 and Self.HEIGHT > 0, (
            "BatchedCameraRenderer: WIDTH and HEIGHT must be positive."
        )
        check_camera(m, cam)

        self.cam = cam
        self.cull = ctx.enqueue_create_buffer[Self.DTYPE](CULL_WORDS)
        self.cull_cap = CULL_WORDS
        self.cull_enabled = True
        self.rgb = ctx.enqueue_create_buffer[Self.DTYPE](
            Self.BATCH * Self.NPIX * RGB_CHANNELS
        )
        self.depth = ctx.enqueue_create_buffer[Self.DTYPE](
            Self.BATCH * Self.NPIX
        )
        self.seg = ctx.enqueue_create_buffer[Self.DTYPE](
            Self.BATCH * Self.NPIX
        )
        ctx.enqueue_memset(self.rgb, 0)
        ctx.enqueue_memset(self.depth, 0)
        ctx.enqueue_memset(self.seg, 0)
        # Down and slightly forward — `Renderer3D`'s key light direction, so a
        # scene is lit from the same side in the viewer and in the tracer.
        self.light_dir = Vec3Generic[Self.DTYPE](
            Scalar[Self.DTYPE](-0.35),
            Scalar[Self.DTYPE](-0.25),
            Scalar[Self.DTYPE](-0.90),
        )
        self.background = Vec3Generic[Self.DTYPE](
            Scalar[Self.DTYPE](0.60),
            Scalar[Self.DTYPE](0.72),
            Scalar[Self.DTYPE](0.90),
        )
        var ld = List[Float64]()
        ld.append(-0.35)
        ld.append(-0.25)
        ld.append(-0.90)
        self.vis = visual_model_from_model[Self.DTYPE, Self.D](m, ld)
        self.vis.upload(ctx)

    def set_visual(
        mut self, ctx: DeviceContext, var v: VisualModel[Self.DTYPE]
    ) raises:
        """Install a `VisualModel` built from the parse and upload it.

        ⚠ THE CALLER OWNS THE CHOICE OF GROUP MASK, and it is the one thing
        here a picture cannot survive getting wrong — see
        `build_visual_model`.
        """
        self.vis = v^
        self.vis.upload(ctx)

    def render(
        mut self,
        ctx: DeviceContext,
        mut d: Data[Self.DTYPE, Self.D, Self.BATCH],
        mut m: Model[Self.DTYPE, Self.D],
        cam: Int = -1,
    ) raises:
        """One launch: every lane, every pixel.

        `cam` names the camera for THIS launch; `-1` (the default) is the
        constructor's. ⚠ ONE RENDERER, SEVERAL CAMERAS OF ONE RESOLUTION — the
        camera index is already a scalar operand of the kernel, so switching
        it per launch costs nothing and shares the visual upload (the LIBERO
        soup, atlas and BVH are tens of MB) between `agentview` and
        `eye_in_hand`. Two renderers were two copies of all of it. The lanes'
        pictures of the previous camera are OVERWRITTEN: copy `rgb` out
        between the launches.

        ⚠ THE CALLER OWNS FRESHNESS. This reads `xpos`/`xquat`/`subtree_com`
        as the device holds them; it does not run forward kinematics. Render
        after the step's FK, or the image is one frame stale — which for a
        camera OBSERVATION is an off-by-one in the MDP and not a visual
        artefact anyone would notice.

        ⚠⚠ TWENTY-ONE BUFFERS AND SIX SCALARS — 27 OPERANDS. Metal's argument
        table fails SILENTLY at 29 and ships at 27
        (`_metals_limit_is_the_argument_table_not_the_stack`), so there is NO
        headroom left: the cull table (2026-09-26) took the last slot, and its
        on/off flag rides in `nl`'s bit 20 rather than taking a scalar. It is
        also why the headlight is a row of the light table rather than six
        scalars. `qpos` was the twentieth buffer (the
        conditional sites read it per lane), and the conditional-site COUNT
        rides in `ng`'s high bits rather than taking a scalar of its own.
        Adding an operand to this kernel is a decision, not a detail.
        """
        var cam_used = self.cam if cam < 0 else cam
        if cam_used != self.cam:
            check_camera(m, cam_used)
        var nvg = self.vis.ngeom
        var nlight = self.vis.nlight
        var ncond = self.vis.ncond
        if nlight >= (1 << 20):
            raise Error(
                "BatchedCameraRenderer: " + String(nlight) + " lights do not"
                " fit `nl`'s 20 bits (bit 20 is the cull flag)"
            )
        if nvg >= 65536 or ncond >= 32768:
            raise Error(
                "BatchedCameraRenderer: " + String(nvg) + " geoms and "
                + String(ncond) + " conditional sites do not fit `ng`'s"
                " 16/15-bit packing"
            )

        @always_inline
        def cam_kernel(
            geoms: LayoutTensor[Self.DTYPE, DYN2, MutAnyOrigin],
            appearance: LayoutTensor[Self.DTYPE, DYN1, MutAnyOrigin],
            bodies: LayoutTensor[Self.DTYPE, Self.L_BODIES, MutAnyOrigin],
            xpos: LayoutTensor[Self.DTYPE, Self.L_B3, MutAnyOrigin],
            xquat: LayoutTensor[Self.DTYPE, Self.L_B4, MutAnyOrigin],
            subtree_com: LayoutTensor[Self.DTYPE, Self.L_B3, MutAnyOrigin],
            cameras: LayoutTensor[Self.DTYPE, Self.L_CAM, MutAnyOrigin],
            mesh_meta: LayoutTensor[Self.DTYPE, DYN1, MutAnyOrigin],
            mesh_tris: LayoutTensor[Self.DTYPE, DYN1, MutAnyOrigin],
            mesh_uv: LayoutTensor[Self.DTYPE, DYN1, MutAnyOrigin],
            hfield_meta: LayoutTensor[
                Self.DTYPE, Self.L_HF_META, MutAnyOrigin
            ],
            hfield_data: LayoutTensor[Self.DTYPE, Self.L_HF, MutAnyOrigin],
            materials: LayoutTensor[Self.DTYPE, DYN1, MutAnyOrigin],
            textures: LayoutTensor[Self.DTYPE, DYN1, MutAnyOrigin],
            texels: LayoutTensor[DType.uint8, DYN1, MutAnyOrigin],
            lights: LayoutTensor[Self.DTYPE, DYN1, MutAnyOrigin],
            qpos: LayoutTensor[Self.DTYPE, Self.L_QPOS, MutAnyOrigin],
            cull_t: LayoutTensor[Self.DTYPE, DYN1, MutAnyOrigin],
            rgb_out: LayoutTensor[Self.DTYPE, Self.L_RGB, MutAnyOrigin],
            depth_out: LayoutTensor[
                Self.DTYPE, Self.L_SCALARPIX, MutAnyOrigin
            ],
            seg_out: LayoutTensor[
                Self.DTYPE, Self.L_SCALARPIX, MutAnyOrigin
            ],
            # ⚠ `Int32`, NOT `Int`. Neither `Int` nor `UInt` conforms to
            # `DevicePassable` — "use a fixed-width type" — so a plain `Int`
            # kernel operand does not compile.
            cam: Int32,
            ng: Int32,
            nl: Int32,
            br: Scalar[Self.DTYPE],
            bg: Scalar[Self.DTYPE],
            bb: Scalar[Self.DTYPE],
        ):
            var i = Int(block_dim.x * block_idx.x + thread_idx.x)
            if i >= Self.BATCH * Self.NPIX:
                return
            # ⚠⚠ THE POSITIVE BRANCH CARRIES THE EVIDENCE, AND AN EARLY
            # `comptime if not ...: return` DOES NOT. `camera_world_frame` and
            # `render_pixel` are both constrained to a floating-point DTYPE,
            # and the compiler accepts the calls only inside a branch that
            # PROVES the constraint — the guard-clause spelling reads
            # identically and fails with "lacking evidence to prove
            # correctness". Same shape, same lesson, as the ray hook in
            # `quadruped_escape_config`.
            comptime if Self.DTYPE.is_floating_point():
                var env = i // Self.NPIX
                var pix = i - env * Self.NPIX
                var py = pix // Self.WIDTH
                var pxx = pix - py * Self.WIDTH

                # ⚠ THE FRAME IS RECOMPUTED PER PIXEL, not hoisted. Same trade
                # `ray_model` makes for geom poses and for the same reason: a
                # thread owns one pixel and cannot hold a scene, and the shared
                # alternative is a per-thread array — the storage class Metal has
                # silently miscomputed four times here.
                var frame = camera_world_frame[Self.DTYPE](
                    cameras, xpos, xquat, subtree_com, env, Int(cam)
                )
                # `ng` = geoms in the low 16 bits, conditional sites above.
                var ngi = Int(ng)
                var n_geom = ngi & 0xFFFF
                var n_cond = ngi >> 16
                var hit = render_pixel[
                    Self.DTYPE, Self.SHADOWS, Self.REFLECT,
                    SAMPLES=Self.SAMPLES,
                ](
                    geoms,
                    n_geom,
                    appearance,
                    bodies,
                    xpos,
                    xquat,
                    env,
                    mesh_meta,
                    mesh_tris,
                    mesh_uv,
                    hfield_meta,
                    hfield_data,
                    Self.D.NHFIELD_DATA,
                    materials,
                    textures,
                    texels,
                    lights,
                    Int(nl) & 0xFFFFF,
                    qpos,
                    n_cond,
                    frame,
                    Self.WIDTH,
                    Self.HEIGHT,
                    pxx,
                    py,
                    Vec3Generic[Self.DTYPE](br, bg, bb),
                    cull_t,
                    # `nl` bit 20 = the cull table is valid for this launch.
                    env * n_geom * CULL_WORDS if ((Int(nl) >> 20) & 1) != 0 else -1,
                )
                rgb_out[env, pix * RGB_CHANNELS + 0] = hit.rgb.x
                rgb_out[env, pix * RGB_CHANNELS + 1] = hit.rgb.y
                rgb_out[env, pix * RGB_CHANNELS + 2] = hit.rgb.z
                depth_out[env, pix] = hit.depth
                seg_out[env, pix] = Scalar[Self.DTYPE](hit.geom)

        # ── the cull pre-pass: one thread per (lane, visual geom) ─────────
        var cull_on = self.cull_enabled and nvg > 0
        if cull_on:
            var need = Self.BATCH * nvg * CULL_WORDS
            if need > self.cull_cap:
                self.cull = ctx.enqueue_create_buffer[Self.DTYPE](need)
                self.cull_cap = need

            @always_inline
            def cull_kernel(
                geoms: LayoutTensor[Self.DTYPE, DYN2, MutAnyOrigin],
                xpos: LayoutTensor[Self.DTYPE, Self.L_B3, MutAnyOrigin],
                xquat: LayoutTensor[Self.DTYPE, Self.L_B4, MutAnyOrigin],
                subtree_com: LayoutTensor[Self.DTYPE, Self.L_B3, MutAnyOrigin],
                cameras: LayoutTensor[Self.DTYPE, Self.L_CAM, MutAnyOrigin],
                cull_out: LayoutTensor[Self.DTYPE, DYN1, MutAnyOrigin],
                cam: Int32,
                ng: Int32,
            ):
                var i = Int(block_dim.x * block_idx.x + thread_idx.x)
                var ngv = Int(ng)
                if i >= Self.BATCH * ngv:
                    return
                comptime if Self.DTYPE.is_floating_point():
                    var env = i // ngv
                    var g = i - env * ngv
                    var frame = camera_world_frame[Self.DTYPE](
                        cameras, xpos, xquat, subtree_com, env, Int(cam)
                    )
                    var r = geom_screen_rect[Self.DTYPE](
                        geoms, xpos, xquat, env, g, frame,
                        Self.WIDTH, Self.HEIGHT,
                    )
                    var o = i * CULL_WORDS
                    cull_out[o + 0] = r[0]
                    cull_out[o + 1] = r[1]
                    cull_out[o + 2] = r[2]
                    cull_out[o + 3] = r[3]

            var ncull = Self.BATCH * nvg
            ctx.enqueue_function[cull_kernel](
                self.vis.geoms.lt_dyn["gpu", DYN2](
                    rl2(nvg + ncond, MODEL_GEOM_SIZE)
                ),
                d.xpos.lt["gpu", Self.L_B3](),
                d.xquat.lt["gpu", Self.L_B4](),
                d.subtree_com.lt["gpu", Self.L_B3](),
                m.cameras.lt["gpu", Self.L_CAM](),
                LayoutTensor[Self.DTYPE, DYN1, MutAnyOrigin](
                    self.cull, rl1(self.cull_cap)
                ),
                Int32(cam_used),
                Int32(nvg),
                grid_dim=(ceildiv(ncull, TPB),),
                block_dim=(TPB,),
            )

        var total = Self.BATCH * Self.NPIX
        ctx.enqueue_function[cam_kernel](
            self.vis.geoms.lt_dyn["gpu", DYN2](
                rl2(nvg + ncond, MODEL_GEOM_SIZE)
            ),
            self.vis.appearance.lt_dyn["gpu", DYN1](
                rl1(self.vis.appearance.n)
            ),
            m.bodies.lt["gpu", Self.L_BODIES](),
            d.xpos.lt["gpu", Self.L_B3](),
            d.xquat.lt["gpu", Self.L_B4](),
            d.subtree_com.lt["gpu", Self.L_B3](),
            m.cameras.lt["gpu", Self.L_CAM](),
            self.vis.mesh_meta.lt_dyn["gpu", DYN1](rl1(self.vis.mesh_meta.n)),
            self.vis.mesh_tris.lt_dyn["gpu", DYN1](rl1(self.vis.mesh_tris.n)),
            self.vis.mesh_uv.lt_dyn["gpu", DYN1](rl1(self.vis.mesh_uv.n)),
            m.hfield_meta.lt["gpu", Self.L_HF_META](),
            d.hfield_data.lt["gpu", Self.L_HF](),
            self.vis.materials.lt_dyn["gpu", DYN1](rl1(self.vis.materials.n)),
            self.vis.textures.lt_dyn["gpu", DYN1](rl1(self.vis.textures.n)),
            self.vis.texels.lt_dyn["gpu", DYN1](rl1(self.vis.texels.n)),
            self.vis.lights.lt_dyn["gpu", DYN1](rl1(self.vis.lights.n)),
            d.qpos.lt["gpu", Self.L_QPOS](),
            LayoutTensor[Self.DTYPE, DYN1, MutAnyOrigin](
                self.cull, rl1(self.cull_cap)
            ),
            LayoutTensor[Self.DTYPE, Self.L_RGB](self.rgb),
            LayoutTensor[Self.DTYPE, Self.L_SCALARPIX](self.depth),
            LayoutTensor[Self.DTYPE, Self.L_SCALARPIX](self.seg),
            Int32(cam_used),
            Int32(nvg + (ncond << 16)),
            Int32(nlight + ((1 if cull_on else 0) << 20)),
            self.background.x,
            self.background.y,
            self.background.z,
            grid_dim=(ceildiv(total, TPB),),
            block_dim=(TPB,),
        )

    def frame_bgra(
        mut self,
        ctx: DeviceContext,
        env: Int,
        mut out: List[UInt8],
        scale: Int = 1,
    ) raises:
        """One lane's RGB as **BGRA uint8**, ready for `VideoRecorder`.

        `scale` replicates each pixel `scale` times in both axes — nearest
        neighbour, on the host. An 84x84 observation is what the AGENT sees and
        is the honest thing to record, but it is also a postage stamp in a video
        player; upscaling here shows the real pixels bigger rather than
        rendering a different, prettier image the agent never gets.

        ⚠⚠ BGRA, NOT RGBA, AND THE ORDER IS NOT NEGOTIABLE.
        `VideoRecorder.add_frame_bgra` takes the address of a B8G8R8A8 buffer
        because that is the Metal/SDL3 swapchain format it was written for, and
        it re-orders to RGB with `np.take(arr, [2, 1, 0], axis=2)`. Handing it
        RGBA produces a picture that is correct in every respect except that
        red and blue are swapped — which looks like a plausible colour-grading
        choice and not like a bug.

        ⚠ THIS IS THE "uint8 UNPACK" the assessment asked for, arriving where
        it is actually needed rather than as a second kernel. The reference
        packs ABGR into a `uint32` on the DEVICE because its render context
        stores every camera in one buffer; here the float buffer is already the
        RL observation, and the byte conversion is only for the video sink. So
        it runs on the HOST, once per recorded frame, over one lane.

        `out` is resized, so a caller may reuse it across frames.
        """
        var n = Self.NPIX
        var s = scale if scale >= 1 else 1
        var ow = Self.WIDTH * s
        var oh = Self.HEIGHT * s
        out = List[UInt8](length=ow * oh * 4, fill=UInt8(0))
        var host = ctx.enqueue_create_host_buffer[Self.DTYPE](
            Self.BATCH * n * RGB_CHANNELS
        )
        ctx.enqueue_copy(host, self.rgb)
        ctx.synchronize()
        var base = env * n * RGB_CHANNELS
        for py in range(Self.HEIGHT):
            for px in range(Self.WIDTH):
                var p = py * Self.WIDTH + px
                var b = _to_byte(Float64(host[base + p * RGB_CHANNELS + 2]))
                var g = _to_byte(Float64(host[base + p * RGB_CHANNELS + 1]))
                var r = _to_byte(Float64(host[base + p * RGB_CHANNELS + 0]))
                for dy in range(s):
                    for dx in range(s):
                        var o = ((py * s + dy) * ow + (px * s + dx)) * 4
                        out[o + 0] = UInt8(b)
                        out[o + 1] = UInt8(g)
                        out[o + 2] = UInt8(r)
                        out[o + 3] = UInt8(255)

    def frame_bgra_from_cpu(
        self,
        rgb: List[Scalar[Self.DTYPE]],
        env: Int,
        mut out: List[UInt8],
    ):
        """`frame_bgra` for a picture that came off `render_cpu`.

        Exists so a video can be produced on a machine with no accelerator,
        and so the CPU and GPU legs can be written to two files and LOOKED AT
        side by side — which is a comparison the pixel residual cannot make.
        """
        var n = Self.NPIX
        out = List[UInt8](length=n * 4, fill=UInt8(0))
        var base = env * n * RGB_CHANNELS
        for p in range(n):
            out[p * 4 + 0] = UInt8(
                _to_byte(Float64(rgb[base + p * RGB_CHANNELS + 2]))
            )
            out[p * 4 + 1] = UInt8(
                _to_byte(Float64(rgb[base + p * RGB_CHANNELS + 1]))
            )
            out[p * 4 + 2] = UInt8(
                _to_byte(Float64(rgb[base + p * RGB_CHANNELS + 0]))
            )
            out[p * 4 + 3] = UInt8(255)

    def render_cpu(
        mut self,
        mut d: Data[Self.DTYPE, Self.D, Self.BATCH],
        mut m: Model[Self.DTYPE, Self.D],
        mut rgb: List[Scalar[Self.DTYPE]],
        mut depth: List[Scalar[Self.DTYPE]],
        mut seg: List[Scalar[Self.DTYPE]],
    ) raises:
        """The same pixels, on the host, into plain `List`s.

        ⚠⚠ THIS IS THE CONTROL LEG OF THE GATE, AND IT IS THE SAME CODE. It
        calls `camera_world_frame` and `render_pixel` exactly as the kernel
        does — the only difference is `lt["cpu", ...]` and a serial loop. That
        is the property `tests/physics3d/test_ray_model_gpu_vs_cpu.mojo`
        exists to defend one layer down, and the reason a GPU miscompute shows
        up here as a DIFFERENCE rather than as two matching wrong answers.

        The lists are resized, not appended to, so a caller may reuse them.
        """
        var n = Self.BATCH * Self.NPIX
        rgb = List[Scalar[Self.DTYPE]](
            length=n * RGB_CHANNELS, fill=Scalar[Self.DTYPE](0)
        )
        depth = List[Scalar[Self.DTYPE]](length=n, fill=Scalar[Self.DTYPE](0))
        seg = List[Scalar[Self.DTYPE]](length=n, fill=Scalar[Self.DTYPE](0))

        # ⚠ THE POSITIVE BRANCH, for the same reason as the kernel above.
        comptime if Self.DTYPE.is_floating_point():
            var nvg = self.vis.ngeom
            var ncond = self.vis.ncond
            var geoms_c = self.vis.geoms.lt_dyn["cpu", DYN2](
                rl2(nvg + ncond, MODEL_GEOM_SIZE)
            )
            var qpos_c = d.qpos.lt["cpu", Self.L_QPOS]()
            var app_c = self.vis.appearance.lt_dyn["cpu", DYN1](
                rl1(self.vis.appearance.n)
            )
            var bodies_c = m.bodies.lt["cpu", Self.L_BODIES]()
            var xpos_c = d.xpos.lt["cpu", Self.L_B3]()
            var xquat_c = d.xquat.lt["cpu", Self.L_B4]()
            var com_c = d.subtree_com.lt["cpu", Self.L_B3]()
            var cams_c = m.cameras.lt["cpu", Self.L_CAM]()
            var mm_c = self.vis.mesh_meta.lt_dyn["cpu", DYN1](
                rl1(self.vis.mesh_meta.n)
            )
            var mt_c = self.vis.mesh_tris.lt_dyn["cpu", DYN1](
                rl1(self.vis.mesh_tris.n)
            )
            var uv_c = self.vis.mesh_uv.lt_dyn["cpu", DYN1](
                rl1(self.vis.mesh_uv.n)
            )
            var hm_c = m.hfield_meta.lt["cpu", Self.L_HF_META]()
            var hd_c = d.hfield_data.lt["cpu", Self.L_HF]()
            var mat_c = self.vis.materials.lt_dyn["cpu", DYN1](
                rl1(self.vis.materials.n)
            )
            var tex_c = self.vis.textures.lt_dyn["cpu", DYN1](
                rl1(self.vis.textures.n)
            )
            var txl_c = self.vis.texels.lt_dyn["cpu", DYN1](
                rl1(self.vis.texels.n)
            )
            var lit_c = self.vis.lights.lt_dyn["cpu", DYN1](
                rl1(self.vis.lights.n)
            )
            var nlight = self.vis.nlight

            for env in range(Self.BATCH):
                var frame = camera_world_frame[Self.DTYPE](
                    cams_c, xpos_c, xquat_c, com_c, env, self.cam
                )
                for py in range(Self.HEIGHT):
                    for pxx in range(Self.WIDTH):
                        var pix = py * Self.WIDTH + pxx
                        var hit = render_pixel[
                            Self.DTYPE, Self.SHADOWS, Self.REFLECT,
                            SAMPLES=Self.SAMPLES,
                        ](
                            geoms_c,
                            nvg,
                            app_c,
                            bodies_c,
                            xpos_c,
                            xquat_c,
                            env,
                            mm_c,
                            mt_c,
                            uv_c,
                            hm_c,
                            hd_c,
                            Self.D.NHFIELD_DATA,
                            mat_c,
                            tex_c,
                            txl_c,
                            lit_c,
                            nlight,
                            qpos_c,
                            ncond,
                            frame,
                            Self.WIDTH,
                            Self.HEIGHT,
                            pxx,
                            py,
                            self.background,
                            # The host leg is UNCULLED: the gate compares the
                            # culled device render against it.
                            mm_c,
                            -1,
                        )
                        var b = env * Self.NPIX + pix
                        rgb[b * RGB_CHANNELS + 0] = hit.rgb.x
                        rgb[b * RGB_CHANNELS + 1] = hit.rgb.y
                        rgb[b * RGB_CHANNELS + 2] = hit.rgb.z
                        depth[b] = hit.depth
                        seg[b] = Scalar[Self.DTYPE](hit.geom)
