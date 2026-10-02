"""PushT 96x96 RGB pixel rasterizer (CPU + GPU).

Produces an RGB image for the `pixels` observation type, mirroring the
pymunk reference's z-order:
    1. White background
    2. LightGreen goal-T (drawn first so anything else on top occludes it)
    3. LightGray walls (4 segments at the world box edges)
    4. LightSlateGray block-T
    5. RoyalBlue agent circle

All world→pixel conversion is a simple scale (96/512); we do NOT flip Y so
keypoints in obs and pixel obs share a consistent frame.

Pixel layout: `[BATCH, H, W, 3]` as Float32 in [0, 255].
"""

from std.math import cos, sin
from layout import Layout, LayoutTensor
from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.physics2d import dtype, TPB
from noeira.physics2d.constants import IDX_X, IDX_Y, IDX_ANGLE
from .constants import PConstants, PushTLayout
from .geometry import (
    t_rect_long_vertex,
    t_rect_stem_vertex,
    t_origin_from_body,
)


comptime IMG_H: Int = 96
comptime IMG_W: Int = 96
comptime IMG_C: Int = 3
comptime IMG_SIZE: Int = IMG_H * IMG_W * IMG_C
comptime PIXEL_SCALE: Float64 = 512.0 / 96.0  # world per pixel


# RGB tuples
@always_inline
def _white() -> Tuple[Scalar[dtype], Scalar[dtype], Scalar[dtype]]:
    return (Scalar[dtype](255.0), Scalar[dtype](255.0), Scalar[dtype](255.0))


@always_inline
def _light_green() -> Tuple[
    Scalar[dtype], Scalar[dtype], Scalar[dtype]
]:
    return (Scalar[dtype](144.0), Scalar[dtype](238.0), Scalar[dtype](144.0))


@always_inline
def _light_gray() -> Tuple[
    Scalar[dtype], Scalar[dtype], Scalar[dtype]
]:
    return (Scalar[dtype](211.0), Scalar[dtype](211.0), Scalar[dtype](211.0))


@always_inline
def _light_slate_gray() -> Tuple[
    Scalar[dtype], Scalar[dtype], Scalar[dtype]
]:
    return (Scalar[dtype](119.0), Scalar[dtype](136.0), Scalar[dtype](153.0))


@always_inline
def _royal_blue() -> Tuple[
    Scalar[dtype], Scalar[dtype], Scalar[dtype]
]:
    return (Scalar[dtype](65.0), Scalar[dtype](105.0), Scalar[dtype](225.0))


# =============================================================================
# Point-in-shape tests in world coords.
# =============================================================================


@always_inline
def _point_in_rot_rect(
    px: Scalar[dtype],
    py: Scalar[dtype],
    cx: Scalar[dtype],
    cy: Scalar[dtype],
    cos_a: Scalar[dtype],
    sin_a: Scalar[dtype],
    rect_idx: Int,  # 0 = long bar, 1 = stem
) -> Bool:
    """True if world point (px, py) is inside the body's rotated rect at
    rect_idx. We test by rotating (px, py) into the body's local frame and
    checking against the axis-aligned bounds of the local-frame rectangle."""
    var dx = px - cx
    var dy = py - cy
    # Inverse rotation: local = R(-angle) * world_offset
    var lx = dx * cos_a + dy * sin_a
    var ly = -dx * sin_a + dy * cos_a
    # Local bounds of each rect (CCW vertex 0 and 2 form min and max corners
    # in local frame since the rect is axis-aligned in local frame).
    if rect_idx == 0:
        # Long bar: x in [-60, 60], y in [0, 30]
        var s = Scalar[dtype](PConstants.T_SCALE)
        var hw = Scalar[dtype](PConstants.T_LENGTH * PConstants.T_SCALE / 2.0)
        return (
            lx >= -hw
            and lx <= hw
            and ly >= Scalar[dtype](0.0)
            and ly <= s
        )
    else:
        # Stem: x in [-15, 15], y in [30, 120]
        var hs = Scalar[dtype](PConstants.T_SCALE / 2.0)
        var s = Scalar[dtype](PConstants.T_SCALE)
        var ly_max = Scalar[dtype](
            PConstants.T_LENGTH * PConstants.T_SCALE
        )
        return lx >= -hs and lx <= hs and ly >= s and ly <= ly_max


@always_inline
def _point_in_t(
    px: Scalar[dtype],
    py: Scalar[dtype],
    cx: Scalar[dtype],
    cy: Scalar[dtype],
    angle: Scalar[dtype],
) -> Bool:
    var ca = cos(angle)
    var sa = sin(angle)
    return _point_in_rot_rect(px, py, cx, cy, ca, sa, 0) or _point_in_rot_rect(
        px, py, cx, cy, ca, sa, 1
    )


@always_inline
def _point_in_circle(
    px: Scalar[dtype],
    py: Scalar[dtype],
    cx: Scalar[dtype],
    cy: Scalar[dtype],
    r: Scalar[dtype],
) -> Bool:
    var dx = px - cx
    var dy = py - cy
    return dx * dx + dy * dy <= r * r


@always_inline
def _point_on_wall(
    px: Scalar[dtype], py: Scalar[dtype]
) -> Bool:
    """True if a world point falls on one of the 4 box-perimeter strips."""
    var lo = Scalar[dtype](PConstants.WORLD_MIN)
    var hi = Scalar[dtype](PConstants.WORLD_MAX)
    var w = Scalar[dtype](PConstants.WALL_RADIUS)
    var on_left = px >= lo - w and px <= lo + w and py >= lo - w and py <= hi + w
    var on_right = (
        px >= hi - w and px <= hi + w and py >= lo - w and py <= hi + w
    )
    var on_bot = py >= lo - w and py <= lo + w and px >= lo - w and px <= hi + w
    var on_top = py >= hi - w and py <= hi + w and px >= lo - w and px <= hi + w
    return on_left or on_right or on_bot or on_top


# =============================================================================
# Per-pixel renderer
# =============================================================================


@always_inline
def _render_pixel(
    px_world: Scalar[dtype],
    py_world: Scalar[dtype],
    block_cx: Scalar[dtype],
    block_cy: Scalar[dtype],
    block_angle: Scalar[dtype],
    agent_cx: Scalar[dtype],
    agent_cy: Scalar[dtype],
    goal_cx: Scalar[dtype],
    goal_cy: Scalar[dtype],
    goal_angle: Scalar[dtype],
) -> Tuple[Scalar[dtype], Scalar[dtype], Scalar[dtype]]:
    # Top of z-order first: agent → block-T → walls → goal-T → white
    if _point_in_circle(
        px_world, py_world, agent_cx, agent_cy, Scalar[dtype](PConstants.AGENT_RADIUS)
    ):
        return _royal_blue()
    if _point_in_t(px_world, py_world, block_cx, block_cy, block_angle):
        return _light_slate_gray()
    if _point_on_wall(px_world, py_world):
        return _light_gray()
    if _point_in_t(px_world, py_world, goal_cx, goal_cy, goal_angle):
        return _light_green()
    return _white()


# =============================================================================
# CPU: render a single env to a flat [IMG_H * IMG_W * 3] buffer.
# =============================================================================


@always_inline
def render_pixel_obs_single(
    block_cx: Scalar[dtype],
    block_cy: Scalar[dtype],
    block_angle: Scalar[dtype],
    agent_cx: Scalar[dtype],
    agent_cy: Scalar[dtype],
    pixels: LayoutTensor[
        dtype, Layout.row_major(IMG_H, IMG_W, IMG_C), MutAnyOrigin
    ],
):
    """CPU rasterizer for a single env. `pixels` is the H × W × 3 output."""
    var scale = Scalar[dtype](PIXEL_SCALE)
    var goal_cx = Scalar[dtype](PConstants.GOAL_X)
    var goal_cy = Scalar[dtype](PConstants.GOAL_Y)
    var goal_angle = Scalar[dtype](PConstants.GOAL_ANGLE)
    for r in range(IMG_H):
        for c in range(IMG_W):
            var py = (Scalar[dtype](r) + Scalar[dtype](0.5)) * scale
            var px = (Scalar[dtype](c) + Scalar[dtype](0.5)) * scale
            var rgb = _render_pixel(
                px,
                py,
                block_cx,
                block_cy,
                block_angle,
                agent_cx,
                agent_cy,
                goal_cx,
                goal_cy,
                goal_angle,
            )
            pixels[r, c, 0] = rgb[0]
            pixels[r, c, 1] = rgb[1]
            pixels[r, c, 2] = rgb[2]


# =============================================================================
# CPU: render a single env at an ARBITRARY square resolution (OUT × OUT).
# Same world→pixel math + z-order as `render_pixel_obs_single`, but the
# output side length is a comptime parameter — used to feed the LeWM world
# model (trained at 224²) from the 96²-native sim for the sim-domain probe.
# =============================================================================


@always_inline
def render_pusht_rgb_at[
    OUT: Int
](
    block_cx: Scalar[dtype],
    block_cy: Scalar[dtype],
    block_angle: Scalar[dtype],
    agent_cx: Scalar[dtype],
    agent_cy: Scalar[dtype],
    pixels: LayoutTensor[dtype, Layout.row_major(OUT, OUT, IMG_C), MutAnyOrigin],
):
    """CPU rasterizer for a single env at OUT × OUT × 3 (HWC, [0, 255])."""
    var scale = Scalar[dtype](512.0 / Float64(OUT))
    var goal_cx = Scalar[dtype](PConstants.GOAL_X)
    var goal_cy = Scalar[dtype](PConstants.GOAL_Y)
    var goal_angle = Scalar[dtype](PConstants.GOAL_ANGLE)
    for r in range(OUT):
        for c in range(OUT):
            var py = (Scalar[dtype](r) + Scalar[dtype](0.5)) * scale
            var px = (Scalar[dtype](c) + Scalar[dtype](0.5)) * scale
            var rgb = _render_pixel(
                px, py, block_cx, block_cy, block_angle,
                agent_cx, agent_cy, goal_cx, goal_cy, goal_angle,
            )
            pixels[r, c, 0] = rgb[0]
            pixels[r, c, 1] = rgb[1]
            pixels[r, c, 2] = rgb[2]


# =============================================================================
# GPU: kernel rendering all envs in a batch.
# =============================================================================


def render_pixel_obs_kernel_gpu[
    BATCH_SIZE: Int, STATE_SIZE: Int,
](
    ctx: DeviceContext,
    states: DeviceBuffer[dtype],
    mut pixels: DeviceBuffer[dtype],
) raises:
    """Launch one thread per (env, pixel) and write `[BATCH, H, W, 3]`.
    `pixels` must have `BATCH * H * W * 3` floats allocated.
    """
    var state_t = LayoutTensor[
        dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE)
    ](states)
    var pix_t = LayoutTensor[
        dtype,
        Layout.row_major(BATCH_SIZE, IMG_H, IMG_W, IMG_C),
    ](pixels)
    # Grid: (BATCH, H), block: (W,). Each thread handles one pixel.
    comptime BLOCKS_X = (IMG_W + TPB - 1) // TPB

    @always_inline
    def render_wrapper(
        st: LayoutTensor[
            dtype, Layout.row_major(BATCH_SIZE, STATE_SIZE), ImmutAnyOrigin
        ],
        px: LayoutTensor[
            dtype,
            Layout.row_major(BATCH_SIZE, IMG_H, IMG_W, IMG_C),
            MutAnyOrigin,
        ],
    ):
        var env = Int(block_idx.z)
        var r = Int(block_idx.y)
        var c = Int(block_dim.x * block_idx.x + thread_idx.x)
        if env >= BATCH_SIZE or r >= IMG_H or c >= IMG_W:
            return
        var ao = PushTLayout.BODY_AGENT_OFFSET
        var to_ = PushTLayout.BODY_T_OFFSET
        var ax = rebind[Scalar[dtype]](st[env, ao + IDX_X])
        var ay = rebind[Scalar[dtype]](st[env, ao + IDX_Y])
        var bx = rebind[Scalar[dtype]](st[env, to_ + IDX_X])
        var by = rebind[Scalar[dtype]](st[env, to_ + IDX_Y])
        var ba = rebind[Scalar[dtype]](st[env, to_ + IDX_ANGLE])
        # the physics integrates the T about its cog: back to the origin pose
        var _org = t_origin_from_body(bx, by, ba)
        bx = _org[0]
        by = _org[1]
        var scale = Scalar[dtype](PIXEL_SCALE)
        var pyw = (Scalar[dtype](r) + Scalar[dtype](0.5)) * scale
        var pxw = (Scalar[dtype](c) + Scalar[dtype](0.5)) * scale
        var rgb = _render_pixel(
            pxw,
            pyw,
            bx,
            by,
            ba,
            ax,
            ay,
            Scalar[dtype](PConstants.GOAL_X),
            Scalar[dtype](PConstants.GOAL_Y),
            Scalar[dtype](PConstants.GOAL_ANGLE),
        )
        px[env, r, c, 0] = rgb[0]
        px[env, r, c, 1] = rgb[1]
        px[env, r, c, 2] = rgb[2]

    ctx.enqueue_function[render_wrapper](
        state_t,
        pix_t,
        grid_dim=(BLOCKS_X, IMG_H, BATCH_SIZE),
        block_dim=(TPB,),
    )
