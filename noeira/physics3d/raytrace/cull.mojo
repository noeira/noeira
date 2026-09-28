"""Per-camera screen rectangles — `mj_multiRay`'s bounding-angle cull, in pixels.

`mju_multiRayPrepare` (`engine_ray.c:1355`) computes, once per ray SOURCE, each
geom's bounding box as an azimuth / elevation window seen from that point, and
`mju_singleRay` then skips a geom whose window does not contain the ray. For a
pinhole camera the same bound is sharper in PIXELS: the rays through a pixel
can only reach a geom whose projected bounding box covers that pixel. So once
per (lane, camera, frame) each visual geom's box is projected to a rectangle
in continuous pixel coordinates, and `render.trace_visual` skips every geom
whose rectangle misses the pixel — four compares instead of a pose
composition, an OBB test and, for a mesh, a tree walk
(`noeira-docs/SO101_RENDER_SPEED.md`, lever 3).

⚠⚠ IT MAY ONLY SKIP WHAT CANNOT BE HIT, SO EVERY APPROXIMATION IS OUTWARD.

- The box is the one the ray test itself rejects on first: a mesh's
  `HALF_*` (the soup's own extent, `ray_mesh`'s `ray_box`), a box's and an
  ellipsoid's semi-axes, a sphere's radius, a capsule's radius plus
  half-length along z, a cylinder's radius and half-length. Every hit point
  lies inside it. It is padded by 1e-4 of its size.
- A convex box projects inside the convex hull of its 8 projected corners, so
  their pixel bounds contain every ray that meets the box — IF every corner is
  in front of the camera. A box STRADDLING the camera plane gets the WHOLE
  screen (its projection is unbounded). A box with EVERY corner at or behind
  the plane gets an EMPTY rectangle: it lies in the closed half-space behind
  the camera, and a camera ray (`local z = -1`, `t > 0`) never enters it —
  on the wrist camera, most of the scene is behind it in most poses.
- The rectangle is widened by one pixel on every side, and the test in
  `trace_visual` treats the pixel as its whole unit square `[px, px+1)`, so the
  4x MSAA sample rays (±0.375 px) and float32 disagreement between this
  projection and `camera_sample_ray` stay inside.
- Planes, heightfields and any type not listed get the whole screen.

The geoms are still tested in index order, so which geom wins a tie cannot
change: `tests/tasks/test_so101_tower_render_variants.mojo` holds the culled
pictures to the unculled ones byte for byte, and
`tests/physics3d/test_camera_render_gpu_vs_cpu.mojo` compares the (culled)
device render against the (unculled) host one.
"""

from layout import Layout, LayoutTensor

from noeira.math3d import Vec3 as Vec3Generic

from ..constants import (
    GEOM_SPHERE,
    GEOM_CAPSULE,
    GEOM_CYLINDER,
    GEOM_BOX,
    GEOM_MESH,
    GEOM_ELLIPSOID,
)
from ..gpu.constants import (
    GEOM_IDX_TYPE,
    GEOM_IDX_RADIUS,
    GEOM_IDX_HALF_LENGTH,
    GEOM_IDX_HALF_X,
    GEOM_IDX_HALF_Y,
    GEOM_IDX_HALF_Z,
)
from .camera import CameraFrame
from .render import _geom_world_pose

comptime CULL_WORDS: Int = 4
"""`x0, x1, y0, y1` per geom, continuous pixel coordinates (x right, y DOWN
from the top row, as `camera_sample_ray` counts)."""

comptime CULL_FULL: Float64 = 1.0e9
"""A rectangle of `±CULL_FULL` covers any screen: never skipped."""


def geom_screen_rect[
    DTYPE: DType, L_GEOMS: Layout, L_XPOS: Layout, L_XQUAT: Layout
](
    geoms: LayoutTensor[DTYPE, L_GEOMS, MutAnyOrigin],
    xpos: LayoutTensor[DTYPE, L_XPOS, MutAnyOrigin],
    xquat: LayoutTensor[DTYPE, L_XQUAT, MutAnyOrigin],
    env: Int,
    g: Int,
    frame: CameraFrame[DTYPE],
    width: Int,
    height: Int,
) -> SIMD[DTYPE, 4] where DTYPE.is_floating_point():
    """`(x0, x1, y0, y1)`: every ray from `frame.pos` that meets geom `g` of
    lane `env` crosses the image plane inside this rectangle."""
    var full = SIMD[DTYPE, 4](
        Scalar[DTYPE](-CULL_FULL), Scalar[DTYPE](CULL_FULL),
        Scalar[DTYPE](-CULL_FULL), Scalar[DTYPE](CULL_FULL),
    )
    var gtype = Int(rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_TYPE]))
    var hx = rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_HALF_X])
    var hy = rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_HALF_Y])
    var hz = rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_HALF_Z])
    var rad = rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_RADIUS])
    var hl = rebind[Scalar[DTYPE]](geoms[g, GEOM_IDX_HALF_LENGTH])
    if gtype == GEOM_SPHERE:
        hx = rad
        hy = rad
        hz = rad
    elif gtype == GEOM_CAPSULE:
        hx = rad
        hy = rad
        hz = hl + rad
    elif gtype == GEOM_CYLINDER:
        hx = rad
        hy = rad
        hz = hl
    elif gtype == GEOM_MESH or gtype == GEOM_BOX or gtype == GEOM_ELLIPSOID:
        pass
    else:
        return full
    if not (hx >= 0 and hy >= 0 and hz >= 0):
        return full
    var big = hx if hx > hy else hy
    big = big if big > hz else hz
    var pad = big * Scalar[DTYPE](1.0e-4) + Scalar[DTYPE](1.0e-6)
    hx = hx + pad
    hy = hy + pad
    hz = hz + pad

    var pose = _geom_world_pose[DTYPE](geoms, xpos, xquat, env, g)
    var half_h = frame.tan_half_fovy
    var half_w = half_h * (Scalar[DTYPE](width) / Scalar[DTYPE](height))
    var fw = Scalar[DTYPE](width)
    var fh = Scalar[DTYPE](height)
    var x0 = Scalar[DTYPE](CULL_FULL)
    var x1 = Scalar[DTYPE](-CULL_FULL)
    var y0 = Scalar[DTYPE](CULL_FULL)
    var y1 = Scalar[DTYPE](-CULL_FULL)
    var behind = 0
    # Eight corners from the bits of `c` — no per-thread array (the storage
    # class Metal has miscomputed in this engine).
    for c in range(8):
        var sx = hx if (c & 1) != 0 else -hx
        var sy = hy if (c & 2) != 0 else -hy
        var sz = hz if (c & 4) != 0 else -hz
        var corner = pose[0] + pose[1].rotate_vec(Vec3Generic[DTYPE](sx, sy, sz))
        var rel = corner - frame.pos
        var depth = -rel.dot(frame.zaxis)
        if not (depth > Scalar[DTYPE](1.0e-6)):
            behind += 1
            continue
        var lx = rel.dot(frame.xaxis) / depth
        var ly = rel.dot(frame.yaxis) / depth
        # The inverse of `camera_sample_ray`: lx = -half_w + 2 half_w (x / W),
        # ly = half_h - 2 half_h (y / H).
        var xp = (lx + half_w) / (Scalar[DTYPE](2) * half_w) * fw
        var yp = (half_h - ly) / (Scalar[DTYPE](2) * half_h) * fh
        x0 = x0 if x0 < xp else xp
        x1 = x1 if x1 > xp else xp
        y0 = y0 if y0 < yp else yp
        y1 = y1 if y1 > yp else yp
    if behind == 8:
        # Entirely behind the camera: an EMPTY rectangle (x0 > x1), which no
        # pixel square overlaps.
        return SIMD[DTYPE, 4](
            Scalar[DTYPE](CULL_FULL), Scalar[DTYPE](-CULL_FULL),
            Scalar[DTYPE](CULL_FULL), Scalar[DTYPE](-CULL_FULL),
        )
    if behind > 0:
        return full
    var one = Scalar[DTYPE](1)
    return SIMD[DTYPE, 4](x0 - one, x1 + one, y0 - one, y1 + one)
