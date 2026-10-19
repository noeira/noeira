"""Circle-vs-polygon narrow-phase collision detection.

Generates contact manifolds for circle bodies colliding with polygon (or
compound polygon) bodies. The detected contacts are written into the
standard CONTACT_DATA format so they can be resolved by the existing
ImpulseSolver / UnifiedConstraintSolver.

Algorithm (per pair of circle body C and polygon body P):
    1. Transform circle center into world coordinates.
    2. For each convex sub-polygon of P (1 sub-poly if SHAPE_POLYGON,
       n_subshapes sub-polys if SHAPE_COMPOUND), do:
         a. Walk every edge (v_i, v_{i+1}).
         b. Compute the closest point on that edge segment to the circle
            center.
         c. Track the minimum distance found over all edges.
         d. Simultaneously track whether the circle center is on the
            interior half-plane of every edge (=> center is inside the
            convex sub-poly).
       If the center is inside: generate a contact pushing the circle out
       along the edge with smallest signed penetration. Otherwise, if the
       min distance is <= radius, generate a contact at the closest point
       with normal pointing FROM polygon TO circle.

Contact convention written into the buffer:
    BODY_A = circle body index, BODY_B = polygon body index.
    NORMAL points from B (polygon) toward A (circle), so a positive normal
    impulse pushes A out of B — matching the convention used by ImpulseSolver.
"""

from std.math import cos, sin, sqrt
from max.gpu import thread_idx, block_idx, block_dim

from ..constants import (
    dtype,
    TPB,
    BODY_STATE_SIZE,
    SHAPE_MAX_SIZE,
    CONTACT_DATA_SIZE,
    IDX_X,
    IDX_Y,
    IDX_ANGLE,
    IDX_SHAPE,
    SHAPE_POLYGON,
    SHAPE_CIRCLE,
    SHAPE_COMPOUND,
    MAX_POLYGON_VERTS,
    MAX_COMPOUND_SUBSHAPES,
    CONTACT_BODY_A,
    CONTACT_BODY_B,
    CONTACT_POINT_X,
    CONTACT_POINT_Y,
    CONTACT_NORMAL_X,
    CONTACT_NORMAL_Y,
    CONTACT_DEPTH,
    CONTACT_NORMAL_IMPULSE,
    CONTACT_TANGENT_IMPULSE,
)
from noeira.core.tiles import Tile


# =============================================================================
# Inline geometric helpers
# =============================================================================


@always_inline
def _transform_local_to_world(
    local_x: Scalar[dtype],
    local_y: Scalar[dtype],
    body_x: Scalar[dtype],
    body_y: Scalar[dtype],
    cos_a: Scalar[dtype],
    sin_a: Scalar[dtype],
) -> Tuple[Scalar[dtype], Scalar[dtype]]:
    """Apply body transform (rotation + translation) to a local-frame point."""
    var wx = body_x + local_x * cos_a - local_y * sin_a
    var wy = body_y + local_x * sin_a + local_y * cos_a
    return (wx, wy)


@always_inline
def _circle_vs_convex_polygon(
    circle_world_x: Scalar[dtype],
    circle_world_y: Scalar[dtype],
    radius: Scalar[dtype],
    poly_world_verts_x: Array[Scalar[dtype], MAX_POLYGON_VERTS],
    poly_world_verts_y: Array[Scalar[dtype], MAX_POLYGON_VERTS],
    n_verts: Int,
) -> Tuple[
    Bool,  # is_contact
    Scalar[dtype],  # contact_point_x (on polygon surface)
    Scalar[dtype],  # contact_point_y
    Scalar[dtype],  # normal_x  (from polygon to circle)
    Scalar[dtype],  # normal_y
    Scalar[dtype],  # penetration depth
]:
    """Detect contact between a circle and a convex polygon (CCW vertices in
    world frame). Returns (hit, px, py, nx, ny, depth).

    For convex polygons assumed CCW (counter-clockwise), the outward-pointing
    edge normal of edge (v_i, v_{i+1}) is `(dy, -dx) / len` where
    (dx, dy) = v_{i+1} - v_i. A point P is on the interior half-plane of
    that edge iff dot(P - v_i, normal_out) <= 0.
    """
    var inside = True
    var min_dist_sq = Scalar[dtype](1.0e30)
    var best_px = Scalar[dtype](0.0)
    var best_py = Scalar[dtype](0.0)
    var min_outward_pen = Scalar[dtype](1.0e30)
    var best_inside_nx = Scalar[dtype](0.0)
    var best_inside_ny = Scalar[dtype](0.0)
    var best_inside_px = Scalar[dtype](0.0)
    var best_inside_py = Scalar[dtype](0.0)

    for i in range(n_verts):
        var j = i + 1
        if j == n_verts:
            j = 0
        var v0x = poly_world_verts_x[i]
        var v0y = poly_world_verts_y[i]
        var v1x = poly_world_verts_x[j]
        var v1y = poly_world_verts_y[j]
        var ex = v1x - v0x
        var ey = v1y - v0y
        var elen_sq = ex * ex + ey * ey
        # Outward normal (CCW polygon)
        var nx_e = ey
        var ny_e = -ex
        var nlen = sqrt(nx_e * nx_e + ny_e * ny_e)
        if nlen > Scalar[dtype](1.0e-12):
            nx_e = nx_e / nlen
            ny_e = ny_e / nlen
        # Signed distance from circle center to this edge's supporting line
        # (positive => on outside half-plane)
        var sd = (circle_world_x - v0x) * nx_e + (circle_world_y - v0y) * ny_e
        if sd > Scalar[dtype](0.0):
            inside = False
        else:
            # sd <= 0: candidate for "least-penetrating" outward direction
            # when fully inside (we then push out along this edge's normal).
            var pen = -sd
            if pen < min_outward_pen:
                min_outward_pen = pen
                best_inside_nx = nx_e
                best_inside_ny = ny_e
                # Project center onto this edge's supporting line as the
                # contact point on the polygon surface.
                best_inside_px = circle_world_x - sd * nx_e
                best_inside_py = circle_world_y - sd * ny_e
        # Closest point on this edge segment
        var t = Scalar[dtype](0.0)
        if elen_sq > Scalar[dtype](1.0e-12):
            t = (
                (circle_world_x - v0x) * ex + (circle_world_y - v0y) * ey
            ) / elen_sq
            if t < Scalar[dtype](0.0):
                t = Scalar[dtype](0.0)
            elif t > Scalar[dtype](1.0):
                t = Scalar[dtype](1.0)
        var cpx = v0x + t * ex
        var cpy = v0y + t * ey
        var dx = circle_world_x - cpx
        var dy = circle_world_y - cpy
        var dsq = dx * dx + dy * dy
        if dsq < min_dist_sq:
            min_dist_sq = dsq
            best_px = cpx
            best_py = cpy

    if inside:
        # Circle center inside polygon — push out along best (least-penetrating)
        # outward edge normal. Depth includes the radius.
        var depth = min_outward_pen + radius
        return (
            True,
            best_inside_px,
            best_inside_py,
            best_inside_nx,
            best_inside_ny,
            depth,
        )
    else:
        var dist = sqrt(min_dist_sq)
        if dist > radius:
            return (
                False,
                Scalar[dtype](0.0),
                Scalar[dtype](0.0),
                Scalar[dtype](0.0),
                Scalar[dtype](0.0),
                Scalar[dtype](0.0),
            )
        # Contact: normal from polygon-surface point toward circle center
        var nx: Scalar[dtype]
        var ny: Scalar[dtype]
        if dist > Scalar[dtype](1.0e-9):
            nx = (circle_world_x - best_px) / dist
            ny = (circle_world_y - best_py) / dist
        else:
            # Degenerate: circle center coincides with closest point on edge.
            # Fall back to the candidate inside-edge normal.
            nx = best_inside_nx
            ny = best_inside_ny
        var depth = radius - dist
        return (True, best_px, best_py, nx, ny, depth)


@always_inline
def _write_contact[
    BATCH: Int, MAX_CONTACTS: Int
](
    contacts: Tile[dtype, BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE],
    env: Int,
    c: Int,
    body_a: Int,
    body_b: Int,
    px: Scalar[dtype],
    py: Scalar[dtype],
    nx: Scalar[dtype],
    ny: Scalar[dtype],
    depth: Scalar[dtype],
):
    contacts[env, c, CONTACT_BODY_A] = Scalar[dtype](body_a)
    contacts[env, c, CONTACT_BODY_B] = Scalar[dtype](body_b)
    contacts[env, c, CONTACT_POINT_X] = px
    contacts[env, c, CONTACT_POINT_Y] = py
    contacts[env, c, CONTACT_NORMAL_X] = nx
    contacts[env, c, CONTACT_NORMAL_Y] = ny
    contacts[env, c, CONTACT_DEPTH] = depth
    contacts[env, c, CONTACT_NORMAL_IMPULSE] = Scalar[dtype](0.0)
    contacts[env, c, CONTACT_TANGENT_IMPULSE] = Scalar[dtype](0.0)


# =============================================================================
# Per-pair detector (CPU + GPU compatible, inline)
# =============================================================================


@always_inline
def detect_circle_vs_body_pair[
    BATCH: Int,
    NUM_SHAPES: Int,
    MAX_CONTACTS: Int,
    STATE_SIZE: Int,
](
    state: Tile[dtype, BATCH, STATE_SIZE],
    shapes: Tile[dtype, NUM_SHAPES, SHAPE_MAX_SIZE],
    contacts: Tile[dtype, BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE],
    env: Int,
    circle_body_off: Int,  # absolute offset (within env row) to the circle body's state slot
    poly_body_off: Int,  # absolute offset to the polygon/compound body's state slot
    circle_body_idx: Int,  # value to store in CONTACT_BODY_A
    poly_body_idx: Int,  # value to store in CONTACT_BODY_B
    contact_count_in: Int,
) -> Int:
    """Run circle-vs-polygon narrow phase for one (circle, polygon-or-compound)
    body pair within environment `env`. Appends contacts starting at index
    `contact_count_in` and returns the new contact count.
    """
    var count = contact_count_in

    # Read circle body state
    var cb_x = rebind[Scalar[dtype]](state[env, circle_body_off + IDX_X])
    var cb_y = rebind[Scalar[dtype]](state[env, circle_body_off + IDX_Y])
    var cb_a = rebind[Scalar[dtype]](state[env, circle_body_off + IDX_ANGLE])
    var cb_shape_idx = Int(state[env, circle_body_off + IDX_SHAPE])
    var cb_shape_type = Int(shapes[cb_shape_idx, 0])
    if cb_shape_type != SHAPE_CIRCLE:
        return count

    var radius = rebind[Scalar[dtype]](shapes[cb_shape_idx, 1])
    var c_offset_x = rebind[Scalar[dtype]](shapes[cb_shape_idx, 2])
    var c_offset_y = rebind[Scalar[dtype]](shapes[cb_shape_idx, 3])
    var cb_cos = cos(cb_a)
    var cb_sin = sin(cb_a)
    var circle_world = _transform_local_to_world(
        c_offset_x, c_offset_y, cb_x, cb_y, cb_cos, cb_sin
    )
    var circle_world_x = circle_world[0]
    var circle_world_y = circle_world[1]

    # Read polygon body transform
    var pb_x = rebind[Scalar[dtype]](state[env, poly_body_off + IDX_X])
    var pb_y = rebind[Scalar[dtype]](state[env, poly_body_off + IDX_Y])
    var pb_a = rebind[Scalar[dtype]](state[env, poly_body_off + IDX_ANGLE])
    var pb_shape_idx = Int(state[env, poly_body_off + IDX_SHAPE])
    var pb_shape_type = Int(shapes[pb_shape_idx, 0])
    var pb_cos = cos(pb_a)
    var pb_sin = sin(pb_a)

    # Build list of convex sub-polygon shape indices to test
    var n_sub: Int = 0
    var sub_indices = Array[Int, MAX_COMPOUND_SUBSHAPES](fill=0)
    if pb_shape_type == SHAPE_POLYGON:
        sub_indices[0] = pb_shape_idx
        n_sub = 1
    elif pb_shape_type == SHAPE_COMPOUND:
        n_sub = Int(shapes[pb_shape_idx, 1])
        if n_sub > MAX_COMPOUND_SUBSHAPES:
            n_sub = MAX_COMPOUND_SUBSHAPES
        for s in range(n_sub):
            sub_indices[s] = Int(shapes[pb_shape_idx, 2 + s])
    else:
        return count

    for s in range(n_sub):
        if count >= MAX_CONTACTS:
            break
        var sub_idx = sub_indices[s]
        if Int(shapes[sub_idx, 0]) != SHAPE_POLYGON:
            continue
        var n_verts = Int(shapes[sub_idx, 1])
        if n_verts < 3:
            continue
        if n_verts > MAX_POLYGON_VERTS:
            n_verts = MAX_POLYGON_VERTS

        # Transform sub-polygon vertices to world frame
        var wx = Array[Scalar[dtype], MAX_POLYGON_VERTS](
            fill=Scalar[dtype](0.0)
        )
        var wy = Array[Scalar[dtype], MAX_POLYGON_VERTS](
            fill=Scalar[dtype](0.0)
        )
        for v in range(n_verts):
            var lx = rebind[Scalar[dtype]](shapes[sub_idx, 2 + v * 2])
            var ly = rebind[Scalar[dtype]](shapes[sub_idx, 3 + v * 2])
            var w = _transform_local_to_world(
                lx, ly, pb_x, pb_y, pb_cos, pb_sin
            )
            wx[v] = w[0]
            wy[v] = w[1]

        var result = _circle_vs_convex_polygon(
            circle_world_x, circle_world_y, radius, wx, wy, n_verts
        )
        if result[0]:
            _write_contact[BATCH, MAX_CONTACTS](
                contacts,
                env,
                count,
                circle_body_idx,
                poly_body_idx,
                result[1],
                result[2],
                result[3],
                result[4],
                result[5],
            )
            count += 1

    return count


# =============================================================================
# CirclePolygonCollision: high-level detector for a single (circle, polygon)
# body pair across a strided 2D [BATCH, STATE_SIZE] buffer.
# =============================================================================


struct CirclePolygonCollision(ImplicitlyCopyable):
    """Circle-vs-polygon (or circle-vs-compound) narrow phase for one specific
    (circle, polygon) body pair per environment.

    Unlike broader collision systems that test all body pairs, this one is
    parameterized by the two body offsets and runs only that pair — fitting
    envs like PushT where the topology is known and fixed (agent circle vs
    one polygon block). Removes the need for broad phase.

    Contacts are appended to the provided contact buffer; existing entries
    (e.g., wall contacts) are preserved via the starting `contact_counts`.
    """

    var circle_body_off: Int
    var poly_body_off: Int
    var circle_body_idx: Int
    var poly_body_idx: Int

    def __init__(
        out self,
        circle_body_off: Int,
        poly_body_off: Int,
        circle_body_idx: Int,
        poly_body_idx: Int,
    ):
        """Args:
        circle_body_off: Absolute offset (within an env row) to the
            circle body's state slot (i.e., BODIES_OFFSET + ci*BODY_STATE_SIZE).
        poly_body_off: Absolute offset to the polygon body's state slot.
        circle_body_idx: Body index value to record in CONTACT_BODY_A.
        poly_body_idx: Body index value to record in CONTACT_BODY_B.
        """
        self.circle_body_off = circle_body_off
        self.poly_body_off = poly_body_off
        self.circle_body_idx = circle_body_idx
        self.poly_body_idx = poly_body_idx

    def __init__(out self, *, copy: Self):
        self.circle_body_off = copy.circle_body_off
        self.poly_body_off = copy.poly_body_off
        self.circle_body_idx = copy.circle_body_idx
        self.poly_body_idx = copy.poly_body_idx

    def __init__(out self, *, deinit move: Self):
        self.circle_body_off = move.circle_body_off
        self.poly_body_off = move.poly_body_off
        self.circle_body_idx = move.circle_body_idx
        self.poly_body_idx = move.poly_body_idx

    # CPU detection — appends contacts for each env (does not reset counts).
    def detect[
        BATCH: Int,
        NUM_SHAPES: Int,
        MAX_CONTACTS: Int,
        STATE_SIZE: Int,
    ](
        self,
        state: Tile[dtype, BATCH, STATE_SIZE],
        shapes: Tile[dtype, NUM_SHAPES, SHAPE_MAX_SIZE],
        contacts: Tile[dtype, BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE],
        contact_counts: Tile[dtype, BATCH],
    ):
        for env in range(BATCH):
            var c = Int(contact_counts[env])
            c = detect_circle_vs_body_pair[
                BATCH, NUM_SHAPES, MAX_CONTACTS, STATE_SIZE
            ](
                state,
                shapes,
                contacts,
                env,
                self.circle_body_off,
                self.poly_body_off,
                self.circle_body_idx,
                self.poly_body_idx,
                c,
            )
            contact_counts[env] = Scalar[dtype](c)

    # GPU kernel — one thread per environment. Appends to existing contacts.
    @always_inline
    @staticmethod
    def detect_kernel[
        BATCH: Int,
        NUM_SHAPES: Int,
        MAX_CONTACTS: Int,
        STATE_SIZE: Int,
    ](
        state: Tile[dtype, BATCH, STATE_SIZE],
        shapes: Tile[dtype, NUM_SHAPES, SHAPE_MAX_SIZE],
        contacts: Tile[dtype, BATCH, MAX_CONTACTS, CONTACT_DATA_SIZE],
        contact_counts: Tile[dtype, BATCH],
        circle_body_off: Int,
        poly_body_off: Int,
        circle_body_idx: Int,
        poly_body_idx: Int,
    ):
        var env = Int(block_dim.x * block_idx.x + thread_idx.x)
        if env >= BATCH:
            return
        var c = Int(contact_counts[env])
        c = detect_circle_vs_body_pair[
            BATCH, NUM_SHAPES, MAX_CONTACTS, STATE_SIZE
        ](
            state,
            shapes,
            contacts,
            env,
            circle_body_off,
            poly_body_off,
            circle_body_idx,
            poly_body_idx,
            c,
        )
        contact_counts[env] = Scalar[dtype](c)
