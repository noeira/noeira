"""PushT frames drawn the way stable-worldmodel 0.0.6 draws them — the
renderer behind the LeWM dataset's frames (`envs/pusht/env.py:_render_frame`
+ `envs/utils.py:DrawOptions`).

`render.mojo` is this env's own renderer (flat fills, sampled straight at the
output size); other consumers were trained on it, so it stays. This one is for
models trained on the swm frames (LeWM, docs/LEWM_REOPEN_PLAN.md P4, G4b):

1. a 512 x 512 white canvas, pymunk coordinates used as pixels (y down,
   `positive_y_is_up = False`), every vertex rounded to an integer;
2. the GOAL T: both polygons filled LightGreen (144, 238, 144), no outline;
3. `space.debug_draw`, shapes in the order they were added:
   * the walls: fat segments, radius 2, LightGray (211, 211, 211);
   * the agent: a disc of radius 15 in RoyalBlue (65, 105, 225), then a disc
     of radius 11 in its `light_color` = min(1.2 c, 255) = (78, 126, 255);
   * the T: each polygon filled with `light_color(LightSlateGray)` =
     (142, 163, 183), then every edge as a fat segment (radius 2) in
     LightSlateGray (119, 136, 153) — the darker outline;
4. `cv2.resize(img, (OUT, OUT))`: bilinear (INTER_LINEAR), source sample
   at (x + 0.5) * 512 / OUT - 0.5, no area averaging.

Not drawn: `debug_draw`'s collision-point markers (thin lines at contacts,
only while the agent touches the T).

Poses are ORIGIN poses (the dataset's block x, y — see PConstants.T_COG_Y).
Pixel-exactness with pygame's rasteriser is not claimed: a pixel is inside a
polygon / within a fat segment when its integer coordinates are (boundary
inclusive). G4b measures what is left.
"""

from std.math import cos, sin, sqrt, floor
from layout import Layout, LayoutTensor

from noeira.physics2d import dtype
from .constants import PConstants
from .geometry import t_rect_long_vertex, t_rect_stem_vertex


comptime CANVAS = 512


@always_inline
def _put(
    mut canvas: List[UInt8], x: Int, y: Int, r: UInt8, g: UInt8, b: UInt8
):
    if x < 0 or y < 0 or x >= CANVAS or y >= CANVAS:
        return
    var i = (y * CANVAS + x) * 3
    canvas[i] = r
    canvas[i + 1] = g
    canvas[i + 2] = b


def _fill_polygon(
    mut canvas: List[UInt8], xs: List[Int], ys: List[Int], r: UInt8, g: UInt8, b: UInt8
):
    """Integer-vertex polygon, boundary inclusive (pygame.draw.polygon)."""
    var n = len(xs)
    var x0 = xs[0]
    var x1 = xs[0]
    var y0 = ys[0]
    var y1 = ys[0]
    for i in range(1, n):
        x0 = min(x0, xs[i])
        x1 = max(x1, xs[i])
        y0 = min(y0, ys[i])
        y1 = max(y1, ys[i])
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            # boundary-inclusive crossing test + on-edge check
            var inside = False
            var on_edge = False
            var j = n - 1
            for i in range(n):
                var xi = Float64(xs[i])
                var yi = Float64(ys[i])
                var xj = Float64(xs[j])
                var yj = Float64(ys[j])
                var px = Float64(x)
                var py = Float64(y)
                var cross = (xj - xi) * (py - yi) - (yj - yi) * (px - xi)
                if (
                    abs(cross) < 1e-9
                    and min(xi, xj) <= px <= max(xi, xj)
                    and min(yi, yj) <= py <= max(yi, yj)
                ):
                    on_edge = True
                if (yi > py) != (yj > py):
                    var xc = xi + (py - yi) * (xj - xi) / (yj - yi)
                    if px < xc:
                        inside = not inside
                j = i
            if inside or on_edge:
                _put(canvas, x, y, r, g, b)


def _fill_disc(
    mut canvas: List[UInt8], cx: Int, cy: Int, rad: Int, r: UInt8, g: UInt8, b: UInt8
):
    for y in range(cy - rad, cy + rad + 1):
        for x in range(cx - rad, cx + rad + 1):
            var dx = x - cx
            var dy = y - cy
            if dx * dx + dy * dy <= rad * rad:
                _put(canvas, x, y, r, g, b)


def _fat_segment(
    mut canvas: List[UInt8], ax: Int, ay: Int, bx: Int, by: Int, radius: Float64,
    r: UInt8, g: UInt8, b: UInt8,
):
    """`DrawOptions.draw_fat_segment`: every pixel within `radius` of the
    segment (a line of width 2r with round caps)."""
    var x0 = min(ax, bx) - Int(radius) - 1
    var x1 = max(ax, bx) + Int(radius) + 1
    var y0 = min(ay, by) - Int(radius) - 1
    var y1 = max(ay, by) + Int(radius) + 1
    var dx = Float64(bx - ax)
    var dy = Float64(by - ay)
    var l2 = dx * dx + dy * dy
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            var t = 0.0
            if l2 > 0.0:
                t = ((Float64(x - ax)) * dx + (Float64(y - ay)) * dy) / l2
                t = max(0.0, min(1.0, t))
            var qx = Float64(ax) + t * dx - Float64(x)
            var qy = Float64(ay) + t * dy - Float64(y)
            if qx * qx + qy * qy <= radius * radius:
                _put(canvas, x, y, r, g, b)


def _t_polys_world(
    ox: Float64, oy: Float64, angle: Float64
) -> List[List[Int]]:
    """The T's two polygons in pixels: [xs0, ys0, xs1, ys1], vertices rounded
    (`to_pygame`). Body-local vertices are origin-relative (pymunk's
    `get_vertices`), placed by the ORIGIN pose."""
    var c = cos(angle)
    var s = sin(angle)
    var out = List[List[Int]]()
    for rect in range(2):
        var xs = List[Int]()
        var ys = List[Int]()
        for v in range(4):
            var p = t_rect_long_vertex(v) if rect == 0 else t_rect_stem_vertex(v)
            var lx = Float64(p[0])
            var ly = Float64(p[1])
            xs.append(Int(round(ox + c * lx - s * ly)))
            ys.append(Int(round(oy + s * lx + c * ly)))
        out.append(xs^)
        out.append(ys^)
    return out^


def render_pusht_swm_canvas(
    block_x: Float64, block_y: Float64, block_angle: Float64,
    agent_x: Float64, agent_y: Float64,
) -> List[UInt8]:
    """The 512 x 512 x 3 uint8 canvas, before the resize."""
    var canvas = List[UInt8](length=CANVAS * CANVAS * 3, fill=UInt8(255))
    # goal (filled, no outline)
    var goal = _t_polys_world(PConstants.GOAL_X, PConstants.GOAL_Y, PConstants.GOAL_ANGLE)
    for k in range(2):
        _fill_polygon(canvas, goal[2 * k], goal[2 * k + 1], 144, 238, 144)
    # walls: the static segments, radius 2, LightGray
    _fat_segment(canvas, 5, 506, 5, 5, 2.0, 211, 211, 211)
    _fat_segment(canvas, 5, 5, 506, 5, 2.0, 211, 211, 211)
    _fat_segment(canvas, 506, 5, 506, 506, 2.0, 211, 211, 211)
    _fat_segment(canvas, 5, 506, 506, 506, 2.0, 211, 211, 211)
    # agent: RoyalBlue disc, light inner disc
    var ax = Int(round(agent_x))
    var ay = Int(round(agent_y))
    _fill_disc(canvas, ax, ay, 15, 65, 105, 225)
    _fill_disc(canvas, ax, ay, 11, 78, 126, 255)
    # the T: light fill, then LightSlateGray edges
    var t = _t_polys_world(block_x, block_y, block_angle)
    for k in range(2):
        _fill_polygon(canvas, t[2 * k], t[2 * k + 1], 142, 163, 183)
        for i in range(4):
            var j = (i + 1) % 4
            _fat_segment(
                canvas, t[2 * k][i], t[2 * k + 1][i], t[2 * k][j], t[2 * k + 1][j],
                2.0, 119, 136, 153,
            )
    return canvas^


def render_pusht_swm_at[OUT: Int](
    block_x: Scalar[dtype], block_y: Scalar[dtype], block_angle: Scalar[dtype],
    agent_x: Scalar[dtype], agent_y: Scalar[dtype],
    pix: LayoutTensor[dtype, Layout.row_major(OUT, OUT, 3), MutAnyOrigin],
):
    """swm 0.0.6's frame at OUT x OUT, HWC in [0, 255] — the same contract as
    `render.render_pusht_rgb_at`. The resize is `cv2.resize`'s INTER_LINEAR."""
    var canvas = render_pusht_swm_canvas(
        Float64(block_x), Float64(block_y), Float64(block_angle),
        Float64(agent_x), Float64(agent_y),
    )
    var scale = Float64(CANVAS) / Float64(OUT)
    for y in range(OUT):
        var fy = (Float64(y) + 0.5) * scale - 0.5
        var y0 = Int(floor(fy))
        var wy = fy - Float64(y0)
        var ya = max(0, min(CANVAS - 1, y0))
        var yb = max(0, min(CANVAS - 1, y0 + 1))
        for x in range(OUT):
            var fx = (Float64(x) + 0.5) * scale - 0.5
            var x0 = Int(floor(fx))
            var wx = fx - Float64(x0)
            var xa = max(0, min(CANVAS - 1, x0))
            var xb = max(0, min(CANVAS - 1, x0 + 1))
            for ch in range(3):
                var v00 = Float64(canvas[(ya * CANVAS + xa) * 3 + ch])
                var v01 = Float64(canvas[(ya * CANVAS + xb) * 3 + ch])
                var v10 = Float64(canvas[(yb * CANVAS + xa) * 3 + ch])
                var v11 = Float64(canvas[(yb * CANVAS + xb) * 3 + ch])
                var v = (1.0 - wy) * ((1.0 - wx) * v00 + wx * v01) + wy * (
                    (1.0 - wx) * v10 + wx * v11
                )
                pix[y, x, ch] = Scalar[dtype](round(v))
