"""PushT T-block geometry + Sutherland-Hodgman polygon-clip coverage.

Provides:
- Local-frame T vertices (2 convex rectangles, CCW)
- Body→world transform helpers
- 8-keypoint extraction (matches pymunk reference's `get_keypoints`)
- Sutherland-Hodgman convex-vs-convex polygon clipping
- Coverage fraction: area(T ∩ goal_T) / area(goal_T) via 4 rect-rect clips
- Shape-buffer initialization helpers

All routines are written in a CPU/GPU-friendly style (no heap allocation,
small fixed-size Array scratchpads, no recursion).
"""

from std.math import cos, sin, sqrt
from noeira.physics2d.constants import (
    dtype,
    SHAPE_MAX_SIZE,
    SHAPE_CIRCLE,
    SHAPE_POLYGON,
    SHAPE_COMPOUND,
)
from .constants import PConstants, PushTShapeBuf


# Max vertices we ever clip a polygon down to. Each Sutherland-Hodgman step
# against one half-plane can increase vertex count by at most 1; clipping a
# 4-gon against another 4-gon yields at most 8 vertices in the worst case.
comptime MAX_CLIP_VERTS: Int = 16


# =============================================================================
# T-block local vertices (CCW, matches pymunk reference vertices1/vertices2
# but reordered to CCW so our circle-polygon SAT works correctly).
#
# pymunk gym_pusht uses scale=30, length=4. Vertices in pymunk:
#   rect1 (long bar):  [(-60,30), (60,30), (60,0), (-60,0)]
#   rect2 (stem):      [(-15,30), (-15,120), (15,120), (15,30)]
#
# In screen coords pymunk uses +y down; pymunk's Poly.get_vertices returns the
# original list. We treat +y as the same convention as pymunk: angle wraps
# in 2π and goal pose uses np.pi/4. Reward depends only on relative geometry.
# We use Pymunk's ordering directly here (already a valid winding).
# =============================================================================


@always_inline
def t_rect_long_vertex(
    i: Int,
) -> Tuple[Scalar[dtype], Scalar[dtype]]:
    """Long bar of the T, CCW (math coords: +y up). pymunk's reference
    listing is CW; we reorder to CCW so the half-plane test in
    `clip_convex_polygon` and the outward edge normal in
    `_circle_vs_convex_polygon` produce correct results.
    """
    var s = Scalar[dtype](PConstants.T_SCALE)  # 30
    var half_w = Scalar[dtype](
        PConstants.T_LENGTH * PConstants.T_SCALE / 2.0
    )  # 60
    if i == 0:
        return (-half_w, Scalar[dtype](0.0))
    elif i == 1:
        return (half_w, Scalar[dtype](0.0))
    elif i == 2:
        return (half_w, s)
    else:
        return (-half_w, s)


@always_inline
def t_rect_stem_vertex(
    i: Int,
) -> Tuple[Scalar[dtype], Scalar[dtype]]:
    """Stem of the T, CCW."""
    var s = Scalar[dtype](PConstants.T_SCALE)  # 30
    var long_y = Scalar[dtype](
        PConstants.T_LENGTH * PConstants.T_SCALE
    )  # 120
    var half_s = Scalar[dtype](PConstants.T_SCALE / 2.0)  # 15
    if i == 0:
        return (-half_s, s)
    elif i == 1:
        return (half_s, s)
    elif i == 2:
        return (half_s, long_y)
    else:
        return (-half_s, long_y)


# =============================================================================
# Initialize the shape buffer at run time. Done once during env construction;
# the buffer is then read-only on GPU.
# =============================================================================


def init_pusht_shape_buffer[
    NUM_SHAPES: Int
](
    mut shapes: Array[
        Array[Scalar[dtype], SHAPE_MAX_SIZE], NUM_SHAPES
    ],
):
    """Populate the 5 shape slots used by PushT (see PushTShapeBuf)."""
    # Slot 0: agent circle
    for j in range(SHAPE_MAX_SIZE):
        shapes[PushTShapeBuf.SHAPE_AGENT][j] = Scalar[dtype](0.0)
    shapes[PushTShapeBuf.SHAPE_AGENT][0] = Scalar[dtype](SHAPE_CIRCLE)
    shapes[PushTShapeBuf.SHAPE_AGENT][1] = Scalar[dtype](PConstants.AGENT_RADIUS)
    shapes[PushTShapeBuf.SHAPE_AGENT][2] = Scalar[dtype](0.0)  # cx
    shapes[PushTShapeBuf.SHAPE_AGENT][3] = Scalar[dtype](0.0)  # cy

    # Slot 1: T long-bar polygon (4 verts)
    for j in range(SHAPE_MAX_SIZE):
        shapes[PushTShapeBuf.SHAPE_T_RECT_LONG][j] = Scalar[dtype](0.0)
    shapes[PushTShapeBuf.SHAPE_T_RECT_LONG][0] = Scalar[dtype](SHAPE_POLYGON)
    shapes[PushTShapeBuf.SHAPE_T_RECT_LONG][1] = Scalar[dtype](4)
    for v in range(4):
        var p = t_rect_long_vertex(v)
        # collision shapes are relative to the body's integration point, the
        # cog (0, T_COG_Y) — see PConstants.T_COG_Y
        shapes[PushTShapeBuf.SHAPE_T_RECT_LONG][2 + v * 2] = p[0]
        shapes[PushTShapeBuf.SHAPE_T_RECT_LONG][3 + v * 2] = p[1] - Scalar[dtype](
            PConstants.T_COG_Y
        )

    # Slot 2: T stem polygon
    for j in range(SHAPE_MAX_SIZE):
        shapes[PushTShapeBuf.SHAPE_T_RECT_STEM][j] = Scalar[dtype](0.0)
    shapes[PushTShapeBuf.SHAPE_T_RECT_STEM][0] = Scalar[dtype](SHAPE_POLYGON)
    shapes[PushTShapeBuf.SHAPE_T_RECT_STEM][1] = Scalar[dtype](4)
    for v in range(4):
        var p = t_rect_stem_vertex(v)
        # collision shapes are relative to the body's integration point, the
        # cog (0, T_COG_Y) — see PConstants.T_COG_Y
        shapes[PushTShapeBuf.SHAPE_T_RECT_STEM][2 + v * 2] = p[0]
        shapes[PushTShapeBuf.SHAPE_T_RECT_STEM][3 + v * 2] = p[1] - Scalar[dtype](
            PConstants.T_COG_Y
        )

    # Slot 3: T compound — references slots 1 and 2
    for j in range(SHAPE_MAX_SIZE):
        shapes[PushTShapeBuf.SHAPE_T_COMPOUND][j] = Scalar[dtype](0.0)
    shapes[PushTShapeBuf.SHAPE_T_COMPOUND][0] = Scalar[dtype](SHAPE_COMPOUND)
    shapes[PushTShapeBuf.SHAPE_T_COMPOUND][1] = Scalar[dtype](2)
    shapes[PushTShapeBuf.SHAPE_T_COMPOUND][2] = Scalar[dtype](
        PushTShapeBuf.SHAPE_T_RECT_LONG
    )
    shapes[PushTShapeBuf.SHAPE_T_COMPOUND][3] = Scalar[dtype](
        PushTShapeBuf.SHAPE_T_RECT_STEM
    )

    # Slot 4: goal compound (same body geometry, used only for reward
    # computation; no rigid body backs it).
    for j in range(SHAPE_MAX_SIZE):
        shapes[PushTShapeBuf.SHAPE_GOAL_COMPOUND][j] = Scalar[dtype](0.0)
    shapes[PushTShapeBuf.SHAPE_GOAL_COMPOUND][0] = Scalar[dtype](SHAPE_COMPOUND)
    shapes[PushTShapeBuf.SHAPE_GOAL_COMPOUND][1] = Scalar[dtype](2)
    shapes[PushTShapeBuf.SHAPE_GOAL_COMPOUND][2] = Scalar[dtype](
        PushTShapeBuf.SHAPE_T_RECT_LONG
    )
    shapes[PushTShapeBuf.SHAPE_GOAL_COMPOUND][3] = Scalar[dtype](
        PushTShapeBuf.SHAPE_T_RECT_STEM
    )


# =============================================================================
# Body-transform: map a local-frame point to world frame.
# =============================================================================


@always_inline
def transform_point(
    lx: Scalar[dtype],
    ly: Scalar[dtype],
    cx: Scalar[dtype],
    cy: Scalar[dtype],
    cos_a: Scalar[dtype],
    sin_a: Scalar[dtype],
) -> Tuple[Scalar[dtype], Scalar[dtype]]:
    return (cx + lx * cos_a - ly * sin_a, cy + lx * sin_a + ly * cos_a)


# =============================================================================
# Get the 8 T-keypoints in world coordinates given a pose. Matches pymunk's
# `get_keypoints` which iterates the two shapes and emits each vertex
# transformed by body angle + position.
# =============================================================================


@always_inline
def t_body_from_origin(
    ox: Scalar[dtype], oy: Scalar[dtype], angle: Scalar[dtype]
) -> Tuple[Scalar[dtype], Scalar[dtype]]:
    """The T's physics position (its cog) from its ORIGIN pose:
    origin + R(angle) (0, T_COG_Y). See PConstants.T_COG_Y."""
    var cg = Scalar[dtype](PConstants.T_COG_Y)
    return (ox - cg * sin(angle), oy + cg * cos(angle))


@always_inline
def t_origin_from_body(
    bx: Scalar[dtype], by: Scalar[dtype], angle: Scalar[dtype]
) -> Tuple[Scalar[dtype], Scalar[dtype]]:
    """Inverse of `t_body_from_origin`: the ORIGIN pose every geometry
    consumer (keypoints, coverage, rendering, the dataset's block x, y)
    uses."""
    var cg = Scalar[dtype](PConstants.T_COG_Y)
    return (bx + cg * sin(angle), by - cg * cos(angle))


@always_inline
def get_t_keypoints_world(
    cx: Scalar[dtype],
    cy: Scalar[dtype],
    angle: Scalar[dtype],
    mut out: Array[Scalar[dtype], PConstants.KEYPOINTS_DIM],
):
    """Write 8 (x,y) keypoints into `out` (16 floats total)."""
    var cos_a = cos(angle)
    var sin_a = sin(angle)
    for v in range(4):
        var p = t_rect_long_vertex(v)
        var w = transform_point(p[0], p[1], cx, cy, cos_a, sin_a)
        out[v * 2] = w[0]
        out[v * 2 + 1] = w[1]
    for v in range(4):
        var p = t_rect_stem_vertex(v)
        var w = transform_point(p[0], p[1], cx, cy, cos_a, sin_a)
        out[(4 + v) * 2] = w[0]
        out[(4 + v) * 2 + 1] = w[1]


# =============================================================================
# Sutherland-Hodgman polygon clipping (CONVEX subject ∩ CONVEX clipper).
# Output vertex count is bounded by 2 * max(n_subj, n_clip).
# =============================================================================


@always_inline
def _inside_half_plane(
    px: Scalar[dtype],
    py: Scalar[dtype],
    ax: Scalar[dtype],
    ay: Scalar[dtype],
    bx: Scalar[dtype],
    by: Scalar[dtype],
) -> Bool:
    """True if point P is on the *left* side of directed edge A→B (inside the
    half-plane of a CCW polygon)."""
    var cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
    return cross >= Scalar[dtype](0.0)


@always_inline
def _line_intersect(
    p1x: Scalar[dtype],
    p1y: Scalar[dtype],
    p2x: Scalar[dtype],
    p2y: Scalar[dtype],
    ax: Scalar[dtype],
    ay: Scalar[dtype],
    bx: Scalar[dtype],
    by: Scalar[dtype],
) -> Tuple[Scalar[dtype], Scalar[dtype]]:
    """Intersection of segment P1→P2 with the supporting line of edge A→B.
    Assumes the lines are not parallel (Sutherland-Hodgman invariant).
    """
    var dx1 = p2x - p1x
    var dy1 = p2y - p1y
    var dx2 = bx - ax
    var dy2 = by - ay
    var denom = dx1 * dy2 - dy1 * dx2
    if denom == Scalar[dtype](0.0):
        return (p1x, p1y)
    var t = ((ax - p1x) * dy2 - (ay - p1y) * dx2) / denom
    return (p1x + t * dx1, p1y + t * dy1)


@always_inline
def clip_convex_polygon(
    subj_x: Array[Scalar[dtype], MAX_CLIP_VERTS],
    subj_y: Array[Scalar[dtype], MAX_CLIP_VERTS],
    n_subj: Int,
    clip_x: Array[Scalar[dtype], MAX_CLIP_VERTS],
    clip_y: Array[Scalar[dtype], MAX_CLIP_VERTS],
    n_clip: Int,
    mut out_x: Array[Scalar[dtype], MAX_CLIP_VERTS],
    mut out_y: Array[Scalar[dtype], MAX_CLIP_VERTS],
) -> Int:
    """Clip convex subject polygon against convex (CCW) clipper polygon.
    Returns the number of vertices written into out_x/out_y."""
    # Work buffers (we ping-pong between two scratchpads).
    var cur_x = Array[Scalar[dtype], MAX_CLIP_VERTS](fill=Scalar[dtype](0.0))
    var cur_y = Array[Scalar[dtype], MAX_CLIP_VERTS](fill=Scalar[dtype](0.0))
    var nxt_x = Array[Scalar[dtype], MAX_CLIP_VERTS](fill=Scalar[dtype](0.0))
    var nxt_y = Array[Scalar[dtype], MAX_CLIP_VERTS](fill=Scalar[dtype](0.0))
    for i in range(n_subj):
        cur_x[i] = subj_x[i]
        cur_y[i] = subj_y[i]
    var n_cur = n_subj

    for ce in range(n_clip):
        var ax = clip_x[ce]
        var ay = clip_y[ce]
        var bx_ = clip_x[(ce + 1) % n_clip]
        var by_ = clip_y[(ce + 1) % n_clip]
        var n_nxt = 0
        if n_cur == 0:
            return 0
        for i in range(n_cur):
            var p1x = cur_x[i]
            var p1y = cur_y[i]
            var p2x = cur_x[(i + 1) % n_cur]
            var p2y = cur_y[(i + 1) % n_cur]
            var in_p1 = _inside_half_plane(p1x, p1y, ax, ay, bx_, by_)
            var in_p2 = _inside_half_plane(p2x, p2y, ax, ay, bx_, by_)
            if in_p1:
                if n_nxt < MAX_CLIP_VERTS:
                    nxt_x[n_nxt] = p1x
                    nxt_y[n_nxt] = p1y
                    n_nxt += 1
                if not in_p2:
                    var ip = _line_intersect(
                        p1x, p1y, p2x, p2y, ax, ay, bx_, by_
                    )
                    if n_nxt < MAX_CLIP_VERTS:
                        nxt_x[n_nxt] = ip[0]
                        nxt_y[n_nxt] = ip[1]
                        n_nxt += 1
            else:
                if in_p2:
                    var ip = _line_intersect(
                        p1x, p1y, p2x, p2y, ax, ay, bx_, by_
                    )
                    if n_nxt < MAX_CLIP_VERTS:
                        nxt_x[n_nxt] = ip[0]
                        nxt_y[n_nxt] = ip[1]
                        n_nxt += 1
        # Swap cur <- nxt
        for i in range(n_nxt):
            cur_x[i] = nxt_x[i]
            cur_y[i] = nxt_y[i]
        n_cur = n_nxt

    for i in range(n_cur):
        out_x[i] = cur_x[i]
        out_y[i] = cur_y[i]
    return n_cur


@always_inline
def polygon_area(
    xs: Array[Scalar[dtype], MAX_CLIP_VERTS],
    ys: Array[Scalar[dtype], MAX_CLIP_VERTS],
    n: Int,
) -> Scalar[dtype]:
    """Shoelace area of CCW polygon (absolute value taken at end)."""
    if n < 3:
        return Scalar[dtype](0.0)
    var a = Scalar[dtype](0.0)
    for i in range(n):
        var j = (i + 1) % n
        a = a + xs[i] * ys[j] - xs[j] * ys[i]
    if a < Scalar[dtype](0.0):
        a = -a
    return a * Scalar[dtype](0.5)


# =============================================================================
# Coverage: area(block T ∩ goal T) / area(goal T)
# Each T splits into 2 convex rects → 4 rect-vs-rect clips → sum + divide.
# =============================================================================


@always_inline
def _fill_t_rect_world(
    rect_idx: Int,  # 0 = long bar, 1 = stem
    cx: Scalar[dtype],
    cy: Scalar[dtype],
    angle: Scalar[dtype],
    mut out_x: Array[Scalar[dtype], MAX_CLIP_VERTS],
    mut out_y: Array[Scalar[dtype], MAX_CLIP_VERTS],
) -> Int:
    var cos_a = cos(angle)
    var sin_a = sin(angle)
    for v in range(4):
        var p = t_rect_long_vertex(v) if rect_idx == 0 else t_rect_stem_vertex(v)
        var w = transform_point(p[0], p[1], cx, cy, cos_a, sin_a)
        out_x[v] = w[0]
        out_y[v] = w[1]
    return 4


@always_inline
def compute_coverage(
    block_cx: Scalar[dtype],
    block_cy: Scalar[dtype],
    block_angle: Scalar[dtype],
    goal_cx: Scalar[dtype],
    goal_cy: Scalar[dtype],
    goal_angle: Scalar[dtype],
) -> Scalar[dtype]:
    """Coverage fraction = clip(area(T ∩ goal_T) / area(goal_T), 0, 1)."""
    var inter_area = Scalar[dtype](0.0)
    var goal_area = Scalar[dtype](0.0)

    var b_x = Array[Scalar[dtype], MAX_CLIP_VERTS](fill=Scalar[dtype](0.0))
    var b_y = Array[Scalar[dtype], MAX_CLIP_VERTS](fill=Scalar[dtype](0.0))
    var g_x = Array[Scalar[dtype], MAX_CLIP_VERTS](fill=Scalar[dtype](0.0))
    var g_y = Array[Scalar[dtype], MAX_CLIP_VERTS](fill=Scalar[dtype](0.0))
    var c_x = Array[Scalar[dtype], MAX_CLIP_VERTS](fill=Scalar[dtype](0.0))
    var c_y = Array[Scalar[dtype], MAX_CLIP_VERTS](fill=Scalar[dtype](0.0))

    for bi in range(2):
        var nb = _fill_t_rect_world(bi, block_cx, block_cy, block_angle, b_x, b_y)
        for gi in range(2):
            var ng = _fill_t_rect_world(
                gi, goal_cx, goal_cy, goal_angle, g_x, g_y
            )
            var nc = clip_convex_polygon(b_x, b_y, nb, g_x, g_y, ng, c_x, c_y)
            inter_area = inter_area + polygon_area(c_x, c_y, nc)

    # Goal area: sum of two rect areas (no rotation dependence)
    for gi in range(2):
        var ng = _fill_t_rect_world(gi, goal_cx, goal_cy, goal_angle, g_x, g_y)
        goal_area = goal_area + polygon_area(g_x, g_y, ng)

    if goal_area <= Scalar[dtype](0.0):
        return Scalar[dtype](0.0)
    var cov = inter_area / goal_area
    if cov < Scalar[dtype](0.0):
        cov = Scalar[dtype](0.0)
    elif cov > Scalar[dtype](1.0):
        cov = Scalar[dtype](1.0)
    return cov
