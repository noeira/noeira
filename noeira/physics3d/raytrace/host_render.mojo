"""One lane, on the host, over a RUNTIME-dimensioned model.

`BatchedCameraRenderer` cannot serve the task layer: its layouts are spelt
`Layout.row_major(Self.D.NBODY, ...)` and `D.NBODY` is `DIM_POISON` on the
dynamic dims provider that every `.family` scene is loaded through. That is
`rt_layout.mojo`'s whole subject one layer up, and the answer is the same —
`DYN1`/`DYN2` views with the extents supplied at run time.

⚠⚠ IT CALLS THE SAME `render_pixel`. This is a DISPATCHER, not a second
tracer: every pixel here goes through the function the kernel calls, with the
same `VisualModel`. If the two ever produce different colours it is a layout
or a launch bug, and `test_camera_render_gpu_vs_cpu.mojo` is the leg that says
so. A copy of the shading here would make that gate blind, which is the
recurring defect this tree keeps paying for
(`_a_gate_that_shares_its_reference_implementation_is_blind`).

⚠ HOST ONLY, ONE LANE, AND SLOW ON PURPOSE. 128x128 over a LIBERO scene is
16 384 primary rays against 36 geoms and a quarter of a million triangles in
BVHs; it takes seconds, and it is a GATE rather than an observation path. A
batched observation over a runtime-dimensioned model needs the kernel, and
that needs the layouts above to reach it.
"""

from layout import Layout, LayoutTensor

from noeira.math3d import Vec3 as Vec3Generic

from ..fields import Data, Model
from ..fields.dims import DimsLike
from ..fields.rt_layout import DYN1, DYN2, rl1, rl2
from ..gpu.constants import (
    MODEL_GEOM_SIZE,
    MODEL_BODY_SIZE,
    MAX_GPU_CAMERAS,
    MODEL_CAM_SIZE,
    MODEL_HFIELD_META_SIZE,
    MAX_GPU_HFIELDS,
)
from .camera import camera_world_frame
from .render import render_pixel
from .visual import VisualModel

comptime RGB_CHANNELS_HOST: Int = 3


def render_lane_cpu[
    DTYPE: DType, D: DimsLike, BATCH: Int, SHADOWS: Bool = False,
    REFLECT: Bool = True, SAMPLES: Int = 1
](
    mut d: Data[DTYPE, D, BATCH],
    mut m: Model[DTYPE, D],
    mut vis: VisualModel[DTYPE],
    cam: Int,
    env: Int,
    width: Int,
    height: Int,
    background: Vec3Generic[DTYPE],
    mut rgb: List[Scalar[DTYPE]],
    mut depth: List[Scalar[DTYPE]],
    mut seg: List[Scalar[DTYPE]],
    mut refl: List[Scalar[DTYPE]],
) raises:
    """`[height*width*3]` RGB in [0, 1], plus planar depth, the geom id and
    what the mirror shows (`PixelHit.refl_geom`).

    ⚠ ROW 0 IS THE TOP OF THE IMAGE, as `camera_pixel_ray` writes it.
    """
    var npix = width * height
    rgb = List[Scalar[DTYPE]](
        length=npix * RGB_CHANNELS_HOST, fill=Scalar[DTYPE](0)
    )
    depth = List[Scalar[DTYPE]](length=npix, fill=Scalar[DTYPE](0))
    seg = List[Scalar[DTYPE]](length=npix, fill=Scalar[DTYPE](0))
    refl = List[Scalar[DTYPE]](length=npix, fill=Scalar[DTYPE](0))

    comptime if DTYPE.is_floating_point():
        var nb = m.dims.get_nbody()
        var nvg = vis.ngeom
        # ⚠ `ngeom + ncond` ROWS: the conditional sites follow the geoms.
        var geoms_c = vis.geoms.lt_dyn["cpu", DYN2](
            rl2(nvg + vis.ncond, MODEL_GEOM_SIZE)
        )
        var qpos_c = d.qpos.lt_dyn["cpu", DYN2](rl2(BATCH, m.dims.get_nq()))
        var app_c = vis.appearance.lt_dyn["cpu", DYN1](rl1(vis.appearance.n))
        var bodies_c = m.bodies.lt_dyn["cpu", DYN2](rl2(nb, MODEL_BODY_SIZE))
        var xpos_c = d.xpos.lt_dyn["cpu", DYN2](rl2(BATCH, nb * 3))
        var xquat_c = d.xquat.lt_dyn["cpu", DYN2](rl2(BATCH, nb * 4))
        var com_c = d.subtree_com.lt_dyn["cpu", DYN2](rl2(BATCH, nb * 3))
        var cams_c = m.cameras.lt_dyn["cpu", DYN1](
            rl1(MAX_GPU_CAMERAS * MODEL_CAM_SIZE)
        )
        var mm_c = vis.mesh_meta.lt_dyn["cpu", DYN1](rl1(vis.mesh_meta.n))
        var mt_c = vis.mesh_tris.lt_dyn["cpu", DYN1](rl1(vis.mesh_tris.n))
        var uv_c = vis.mesh_uv.lt_dyn["cpu", DYN1](rl1(vis.mesh_uv.n))
        var hm_c = m.hfield_meta.lt_dyn["cpu", DYN1](
            rl1(MAX_GPU_HFIELDS * MODEL_HFIELD_META_SIZE)
        )
        var hd_c = d.hfield_data.lt_dyn["cpu", DYN1](
            rl1(len(d.hfield_data.data))
        )
        var mat_c = vis.materials.lt_dyn["cpu", DYN1](rl1(vis.materials.n))
        var tex_c = vis.textures.lt_dyn["cpu", DYN1](rl1(vis.textures.n))
        var txl_c = vis.texels.lt_dyn["cpu", DYN1](rl1(vis.texels.n))
        var lit_c = vis.lights.lt_dyn["cpu", DYN1](rl1(vis.lights.n))
        var hf_stride = len(d.hfield_data.data) // BATCH

        var frame = camera_world_frame[DTYPE](
            cams_c, xpos_c, xquat_c, com_c, env, cam
        )
        for py in range(height):
            for px in range(width):
                var hit = render_pixel[DTYPE, SHADOWS, REFLECT, SAMPLES=SAMPLES](
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
                    hf_stride,
                    mat_c,
                    tex_c,
                    txl_c,
                    lit_c,
                    vis.nlight,
                    qpos_c,
                    vis.ncond,
                    frame,
                    width,
                    height,
                    px,
                    py,
                    background,
                    mm_c,
                    -1,
                )
                var b = py * width + px
                rgb[b * RGB_CHANNELS_HOST + 0] = hit.rgb.x
                rgb[b * RGB_CHANNELS_HOST + 1] = hit.rgb.y
                rgb[b * RGB_CHANNELS_HOST + 2] = hit.rgb.z
                depth[b] = hit.depth
                seg[b] = Scalar[DTYPE](hit.geom)
                refl[b] = Scalar[DTYPE](hit.refl_geom)
