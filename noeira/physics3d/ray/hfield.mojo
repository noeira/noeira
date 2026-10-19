"""`mj_rayHfield` — ray against a heightfield geom.

`engine_ray.c:562`. A heightfield is not a pure geom: it needs the elevation
grid, so it cannot live behind `mju_rayGeom`'s signature and the reference does
not put it there either.

THE SHAPE OF THE ROUTINE, which is not obvious from the name:

  1. **Two boxes, not one.** A `<hfield>` is a raised surface sitting on a
     solid BASE that extends `size[3]` BELOW z=0. The base is an ordinary box
     and is intersected as one; the elevation grid only ever lives inside the
     TOP box, `size[2]` tall. The answer starts as the base hit and is
     improved by the grid.
  2. **The top box clips the ray to a segment**, and the segment's horizontal
     footprint gives a row/column window. Only cells in that window are
     visited — this is the acceleration structure, and it is why the routine
     is cheap on a 256x256 field.
  3. **Each cell is two triangles**, tested with the shared `ray_triangle`.
  4. **Then the four vertical SIDES of the top box**, because a ray entering
     from the side passes under no triangle at all: it strikes the wall of the
     field. That is tested against the elevation profile interpolated along
     the wall.

⚠⚠ STEP 4 IS THE ONE THAT LOOKS OPTIONAL AND IS NOT. Without it a ray arriving
horizontally at the edge of the terrain reports the BASE box — i.e. the ground
under the hill rather than the hill — and `quadruped escape`'s rangefinders are
exactly horizontal rays at terrain. It is also the only part that reads `all[]`
from `ray_box`, which is why `ray_box_all` exists.

⚠ `hfield_data` IS NORMALISED TO [0, 1] and scaled by `size[2]` at read time,
in MuJoCo and here alike. A grid holding metres would be wrong by the elevation
scale everywhere and correct at exactly one value of it.

⚠ MuJoCo's grid is `float*` (float32) and ours is the model's `DTYPE`. The
elevations came from the same PNG or binary and were rescaled the same way, so
they agree to float32 and no further — the gate's tolerance is sized for that,
not for float64 exactness. [[feedback_match_the_references_storage_type]]
"""

from std.math import floor, ceil

from layout import TileTensor, TensorLayout

from noeira.math3d import Vec3 as Vec3Generic, Quat as QuatGeneric

from .geom import RAY_NO_HIT, ray_map, ray_box, ray_box_all
from .triangle import ray_basis, ray_triangle


@always_inline
def _elev[
    DTYPE: DType, L_HF: TensorLayout
](
    data: TileTensor[DTYPE, L_HF, MutAnyOrigin],
    adr: Int,
    ncol: Int,
    r: Int,
    c: Int,
) -> Scalar[DTYPE]:
    return rebind[Scalar[DTYPE]](data[adr + r * ncol + c])


def ray_hfield[
    DTYPE: DType, L_HF: TensorLayout
](
    xpos: Vec3Generic[DTYPE],
    xquat: QuatGeneric[DTYPE],
    nrow: Int,
    ncol: Int,
    size_x: Scalar[DTYPE],
    size_y: Scalar[DTYPE],
    size_z: Scalar[DTYPE],
    size_base: Scalar[DTYPE],
    data: TileTensor[DTYPE, L_HF, MutAnyOrigin],
    adr: Int,
    pnt: Vec3Generic[DTYPE],
    vec: Vec3Generic[DTYPE],
) -> Tuple[Scalar[DTYPE], Vec3Generic[DTYPE]] where DTYPE.is_floating_point():
    """Distance to the heightfield surface and its world-frame normal.

    `xpos`/`xquat` are the GEOM's world pose (`geom_xpos`/`geom_xmat`), not the
    body's. `size_*` are `mjModel.hfield_size` — x and y radii, the elevation
    scale, and the base depth.
    """
    var zero = Vec3Generic[DTYPE](0, 0, 0)
    var half = Scalar[DTYPE](0.5)

    # Local +z in world, used to place the two boxes along the field's normal.
    # The reference reads `xmat[2], xmat[5], xmat[8]` — the third COLUMN.
    var zax = xquat.rotate_vec(Vec3Generic[DTYPE](0, 0, 1))

    # ── the base box ────────────────────────────────────────────────────
    var base_size = Vec3Generic[DTYPE](size_x, size_y, size_base * half)
    var base_pos = xpos - zax * (size_base * half)
    var base = ray_box[DTYPE](base_pos, xquat, base_size, pnt, vec)
    var x = base[0]
    var normal_base = base[1]

    # ── the top box ─────────────────────────────────────────────────────
    var top_size = Vec3Generic[DTYPE](size_x, size_y, size_z * half)
    var top_pos = xpos + zax * (size_z * half)
    var top = ray_box_all[DTYPE](top_pos, xquat, top_size, pnt, vec)
    var top_intersect = top.t
    if top_intersect < 0:
        # The grid is unreachable; whatever the base said stands.
        return (x, normal_base if x >= 0 else zero)

    var m = ray_map[DTYPE](xpos, xquat, pnt, vec)
    var lpnt = m[0]
    var lvec = m[1]
    var basis = ray_basis[DTYPE](lvec)
    var b0 = basis[0]
    var b1 = basis[1]

    # ── the segment of the ray inside the top box ───────────────────────
    # ⚠ `seg[0]` starts at 0, not at `top_intersect`: a ray ORIGINATING inside
    # the box has no entry face, and clipping to the entry would skip every
    # cell between the origin and the far wall.
    var seg0 = Scalar[DTYPE](0)
    var seg1 = top_intersect
    for i in range(6):
        var ai6 = top.face(i)
        if ai6 > seg1:
            seg0 = top_intersect
            seg1 = ai6

    var dx = (size_x + size_x) / Scalar[DTYPE](ncol - 1)
    var dy = (size_y + size_y) / Scalar[DTYPE](nrow - 1)
    var sx0 = (lpnt.x + seg0 * lvec.x + size_x) / dx
    var sx1 = (lpnt.x + seg1 * lvec.x + size_x) / dx
    var sy0 = (lpnt.y + seg0 * lvec.y + size_y) / dy
    var sy1 = (lpnt.y + seg1 * lvec.y + size_y) / dy

    # ⚠ The +-1 padding is the reference's and is load-bearing: the segment's
    # endpoints are where the ray meets the BOX, and the triangle it actually
    # hits can be one cell outside that footprint when the surface rises to
    # meet it.
    var cmin = Int(floor(min(sx0, sx1))) - 1
    var cmax = Int(ceil(max(sx0, sx1))) + 1
    var rmin = Int(floor(min(sy0, sy1))) - 1
    var rmax = Int(ceil(max(sy0, sy1))) + 1
    if cmin < 0:
        cmin = 0
    if rmin < 0:
        rmin = 0
    if cmax > ncol - 1:
        cmax = ncol - 1
    if rmax > nrow - 1:
        rmax = nrow - 1

    # The running normal lives in the LOCAL frame and is seeded from the base
    # box's, taken back out of world. One rotation at the end covers all cases.
    var normal_local = zero
    if x >= 0:
        normal_local = xquat.rotate_vec_inverse(normal_base)

    # ── the grid ────────────────────────────────────────────────────────
    for r in range(rmin, rmax):
        for c in range(cmin, cmax):
            var x0 = dx * Scalar[DTYPE](c) - size_x
            var x1 = dx * Scalar[DTYPE](c + 1) - size_x
            var y0 = dy * Scalar[DTYPE](r) - size_y
            var y1 = dy * Scalar[DTYPE](r + 1) - size_y
            var z00 = _elev[DTYPE, L_HF](data, adr, ncol, r, c) * size_z
            var z01 = _elev[DTYPE, L_HF](data, adr, ncol, r, c + 1) * size_z
            var z11 = _elev[DTYPE, L_HF](data, adr, ncol, r + 1, c + 1) * size_z
            var z10 = _elev[DTYPE, L_HF](data, adr, ncol, r + 1, c) * size_z

            # ⚠ Vertex ORDER is the reference's, and it is not the obvious one
            # — its own comment says "swap v1 and v2 for consistent CCW
            # winding (normals point up)". `ray_triangle` does not flip the
            # normal toward the ray, so getting this wrong inverts the surface
            # normal on half the cells while every DISTANCE stays correct.
            var a = ray_triangle[DTYPE](
                Vec3Generic[DTYPE](x0, y0, z00),
                Vec3Generic[DTYPE](x1, y0, z01),
                Vec3Generic[DTYPE](x1, y1, z11),
                lpnt, lvec, b0, b1,
            )
            if a.t >= 0 and (x < 0 or a.t < x):
                x = a.t
                normal_local = a.normal

            var b = ray_triangle[DTYPE](
                Vec3Generic[DTYPE](x0, y0, z00),
                Vec3Generic[DTYPE](x1, y1, z11),
                Vec3Generic[DTYPE](x0, y1, z10),
                lpnt, lvec, b0, b1,
            )
            if b.t >= 0 and (x < 0 or b.t < x):
                x = b.t
                normal_local = b.normal

    # ── the four vertical sides of the top box ──────────────────────────
    # Faces 0..3 are -x, +x, -y, +y in `all`'s packing (`2*axis + side`).
    for i in range(4):
        var ai = top.face(i)
        if ai < 0 or not (ai < x or x < 0):
            continue
        # Height of the crossing, normalised the way the grid is stored.
        var z = (lpnt.z + ai * lvec.z) / size_z
        var y: Scalar[DTYPE]
        var y0f: Scalar[DTYPE]
        var e0: Scalar[DTYPE]
        var e1: Scalar[DTYPE]
        if i < 2:
            # Wall normal to x: walk the grid in ROWS along the far column.
            y = (lpnt.y + ai * lvec.y + size_y) / dy
            y0f = floor(y)
            if y0f < 0:
                y0f = 0
            if y0f > Scalar[DTYPE](nrow - 2):
                y0f = Scalar[DTYPE](nrow - 2)
            var col = ncol - 1 if i == 1 else 0
            e0 = _elev[DTYPE, L_HF](data, adr, ncol, Int(round(y0f)), col)
            e1 = _elev[DTYPE, L_HF](data, adr, ncol, Int(round(y0f)) + 1, col)
        else:
            # Wall normal to y: walk in COLUMNS along the far row.
            y = (lpnt.x + ai * lvec.x + size_x) / dx
            y0f = floor(y)
            if y0f < 0:
                y0f = 0
            if y0f > Scalar[DTYPE](ncol - 2):
                y0f = Scalar[DTYPE](ncol - 2)
            var row = nrow - 1 if i == 3 else 0
            e0 = _elev[DTYPE, L_HF](data, adr, ncol, row, Int(round(y0f)))
            e1 = _elev[DTYPE, L_HF](data, adr, ncol, row, Int(round(y0f)) + 1)

        # Below the linearly interpolated profile means the ray goes INTO the
        # solid wall here rather than over the top of it.
        if z < e0 * (y0f + 1 - y) + e1 * (y - y0f):
            x = ai
            normal_local = Vec3Generic[DTYPE](
                Scalar[DTYPE](-1) if i == 0 else (
                    Scalar[DTYPE](1) if i == 1 else Scalar[DTYPE](0)
                ),
                Scalar[DTYPE](-1) if i == 2 else (
                    Scalar[DTYPE](1) if i == 3 else Scalar[DTYPE](0)
                ),
                Scalar[DTYPE](0),
            )

    if x < 0:
        return (Scalar[DTYPE](RAY_NO_HIT), zero)
    return (x, xquat.rotate_vec(normal_local))
