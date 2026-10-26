"""EPA's polytope — the parts of `engine_collision_gjk.c` that need no support
function.

WHAT THIS IS. `attachFace`, `deleteFace`, `getEdge`, `addEdge`, `horizon` /
`horizonRec`, `projectOriginPlane`, `triAffineCoord`, `triPointIntersect`,
`sameSide` / `testTetra`, `rotmat`, `rayTriangle` and `epaWitness`, ported one
for one. `polytope2` / `polytope3` / `polytope4` and the `epa` loop itself stay
in `gjk.mojo`, because those are the only ones that call the support function
and its argument list is forty parameters wide.

⚠⚠ WHY THE POLYTOPE IS A STRUCT AND NOT A FACE LIST. The EPA this replaced
kept `nef` triangles, rescanned all of them for the closest one each iteration,
marked EVERY globally visible face and COMPACTED the array afterwards. MuJoCo
does none of those four things:

  1. it selects the closest face from `pt->map`, a CANDIDATE LIST a face joins
     only when `lower2 <= dist2 <= upper2` — a face closer to the origin than
     the current lower bound is DISCARDED, because EPA's lower bound must not
     move backwards;
  2. it builds the horizon by walking `Face::adj` recursively OUT FROM the
     closest face, so only CONNECTED visible faces are deleted;
  3. it keeps `upper` as a running MINIMUM across iterations;
  4. it never reuses a face slot, so `adj` indices stay valid for the whole
     run.

On a polytope that is nearly flat — which is exactly what a shallow contact
between two smooth geoms produces — (1) and (2) select a different face from
a full scan, and the contact NORMAL is that face's. See
`feedback_the_reference_can_be_the_unconverged_one` for the case that forced
this: MuJoCo lands 2.00 degrees off the true penetration direction on one
perturbed cylinder-mesh query, and the whole 3-versus-2 contact count of
`hello_robot_stretch_3`'s wheel pair is that one face.

⚠ THE INDICES LIVE IN THE ROW, AS `DTYPE`. Every one of them is bounded by
`EPA_F_CAP`, so they are exact at float32 as well; this is the same choice the
rest of the fields path makes for `mesh_polyvert` and `mesh_edges`.
"""

from std.math import sqrt, abs
from layout import TileTensor, TensorLayout

from .ccd_workspace import (
    EPA_V_CAP,
    EPA_F_CAP,
    EPA_V_STRIDE,
    CCD_WS_EV,
    CCD_WS_EF,
    CCD_WS_EADJ,
    CCD_WS_EFV,
    CCD_WS_EFD,
    CCD_WS_EFI,
    CCD_WS_MAP,
    CCD_WS_HOR,
    CCD_WS_HSTK,
    CCD_WS_CTR,
    CCD_WS_SPX,
    CCD_WS_SPX2,
    SPX_STRIDE,
)

# `mjMINVAL` (`mjtnum.h`) and its square.
comptime EPA_MINVAL: Float64 = 1e-15


# ---------------------------------------------------------------------------
# row accessors — `Polytope`'s fields
# ---------------------------------------------------------------------------


@always_inline
def epa_minval[DTYPE: DType]() -> Scalar[DTYPE]:
    """`mjMINVAL`, at the working precision."""
    return Scalar[DTYPE](EPA_MINVAL)


@always_inline
def epa_mindist[DTYPE: DType]() -> Scalar[DTYPE]:
    """`mjMINDIST2` / `mjMINDIST3` / `mjMINDIST4`.

    ⚠ THE REFERENCE SPLITS THESE BY PRECISION AND WE MUST TOO. At double they
    are all `mjMINVAL2` (1e-30); the single-precision build uses 1e-10 for
    `polytope2` / `polytope3` and 1e-17 for `polytope4`
    (`engine_collision_gjk.c`, the `mjUSESINGLE` block). Only `polytope4`'s
    differs between the two, and at float32 1e-30 is a denormal — the test
    would never fire and a degenerate seed would be kept.
    """
    comptime if DTYPE == DType.float64:
        return Scalar[DTYPE](EPA_MINVAL * EPA_MINVAL)
    else:
        return Scalar[DTYPE](1e-10)


@always_inline
def epa_mindist4[DTYPE: DType]() -> Scalar[DTYPE]:
    """`mjMINDIST4` — `polytope4`'s, which is the one that differs."""
    comptime if DTYPE == DType.float64:
        return Scalar[DTYPE](EPA_MINVAL * EPA_MINVAL)
    else:
        return Scalar[DTYPE](1e-17)


@always_inline
def ev[DTYPE: DType, L_WS: TensorLayout](
    ws: TileTensor[DTYPE, L_WS, MutAnyOrigin], wrow: Int, i: Int, k: Int
) -> Scalar[DTYPE]:
    """`pt->verts[i]` component `k` — 0..2 Minkowski, 3..5 / 6..8 witnesses,
    9 / 10 the support indices."""
    return rebind[Scalar[DTYPE]](ws[wrow, CCD_WS_EV + i * EPA_V_STRIDE + k])


@always_inline
def set_ev[DTYPE: DType, L_WS: TensorLayout](
    ws: TileTensor[DTYPE, L_WS, MutAnyOrigin],
    wrow: Int,
    i: Int,
    k: Int,
    v: Scalar[DTYPE],
):
    ws[wrow, CCD_WS_EV + i * EPA_V_STRIDE + k] = v


@always_inline
def sv[DTYPE: DType, L_WS: TensorLayout](
    ws: TileTensor[DTYPE, L_WS, MutAnyOrigin],
    wrow: Int,
    base: Int,
    i: Int,
    k: Int,
) -> Scalar[DTYPE]:
    """GJK's simplex vertex `i`, component `k` — 0..2 the Minkowski point,
    3..5 and 6..8 the two witness points.

    `base` is `CCD_WS_SPX` or `CCD_WS_SPX2`; see the note in
    `ccd_workspace.mojo` for why the simplex is in the row at all.
    """
    return rebind[Scalar[DTYPE]](ws[wrow, base + i * SPX_STRIDE + k])


@always_inline
def set_sv[DTYPE: DType, L_WS: TensorLayout](
    ws: TileTensor[DTYPE, L_WS, MutAnyOrigin],
    wrow: Int,
    base: Int,
    i: Int,
    k: Int,
    v: Scalar[DTYPE],
):
    ws[wrow, base + i * SPX_STRIDE + k] = v


@always_inline
def ef[DTYPE: DType, L_WS: TensorLayout](
    ws: TileTensor[DTYPE, L_WS, MutAnyOrigin], wrow: Int, f: Int, c: Int
) -> Int:
    """`Face::verts[c]`."""
    return Int(rebind[Scalar[DTYPE]](ws[wrow, CCD_WS_EF + f * 3 + c]))


@always_inline
def eadj[DTYPE: DType, L_WS: TensorLayout](
    ws: TileTensor[DTYPE, L_WS, MutAnyOrigin], wrow: Int, f: Int, c: Int
) -> Int:
    """`Face::adj[c]` — the face across edge `c`."""
    return Int(rebind[Scalar[DTYPE]](ws[wrow, CCD_WS_EADJ + f * 3 + c]))


@always_inline
def set_eadj[DTYPE: DType, L_WS: TensorLayout](
    ws: TileTensor[DTYPE, L_WS, MutAnyOrigin],
    wrow: Int,
    f: Int,
    c: Int,
    v: Int,
):
    ws[wrow, CCD_WS_EADJ + f * 3 + c] = Scalar[DTYPE](v)


@always_inline
def efv[DTYPE: DType, L_WS: TensorLayout](
    ws: TileTensor[DTYPE, L_WS, MutAnyOrigin], wrow: Int, f: Int, c: Int
) -> Scalar[DTYPE]:
    """`Face::v[c]` — the origin projected onto the face's plane."""
    return rebind[Scalar[DTYPE]](ws[wrow, CCD_WS_EFV + f * 3 + c])


@always_inline
def efd[DTYPE: DType, L_WS: TensorLayout](
    ws: TileTensor[DTYPE, L_WS, MutAnyOrigin], wrow: Int, f: Int
) -> Scalar[DTYPE]:
    """`Face::dist2`."""
    return rebind[Scalar[DTYPE]](ws[wrow, CCD_WS_EFD + f])


@always_inline
def efi[DTYPE: DType, L_WS: TensorLayout](
    ws: TileTensor[DTYPE, L_WS, MutAnyOrigin], wrow: Int, f: Int
) -> Int:
    """`Face::index` — >= 0 slot in `map`, -1 not in map, -2 deleted."""
    return Int(rebind[Scalar[DTYPE]](ws[wrow, CCD_WS_EFI + f]))


@always_inline
def set_efi[DTYPE: DType, L_WS: TensorLayout](
    ws: TileTensor[DTYPE, L_WS, MutAnyOrigin], wrow: Int, f: Int, v: Int
):
    ws[wrow, CCD_WS_EFI + f] = Scalar[DTYPE](v)


@always_inline
def emap[DTYPE: DType, L_WS: TensorLayout](
    ws: TileTensor[DTYPE, L_WS, MutAnyOrigin], wrow: Int, i: Int
) -> Int:
    """`pt->map[i]` — the face index in candidate slot `i`."""
    return Int(rebind[Scalar[DTYPE]](ws[wrow, CCD_WS_MAP + i]))


@always_inline
def set_emap[DTYPE: DType, L_WS: TensorLayout](
    ws: TileTensor[DTYPE, L_WS, MutAnyOrigin], wrow: Int, i: Int, f: Int
):
    ws[wrow, CCD_WS_MAP + i] = Scalar[DTYPE](f)


@always_inline
def ehor[DTYPE: DType, L_WS: TensorLayout](
    ws: TileTensor[DTYPE, L_WS, MutAnyOrigin], wrow: Int, h: Int, c: Int
) -> Int:
    """`horizon.indices[h]` (c == 0) and `horizon.edges[h]` (c == 1)."""
    return Int(rebind[Scalar[DTYPE]](ws[wrow, CCD_WS_HOR + h * 2 + c]))


@always_inline
def set_center[DTYPE: DType, L_WS: TensorLayout](
    ws: TileTensor[DTYPE, L_WS, MutAnyOrigin],
    wrow: Int,
    cx: Scalar[DTYPE],
    cy: Scalar[DTYPE],
    cz: Scalar[DTYPE],
):
    ws[wrow, CCD_WS_CTR + 0] = cx
    ws[wrow, CCD_WS_CTR + 1] = cy
    ws[wrow, CCD_WS_CTR + 2] = cz


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


@always_inline
def project_origin_plane[DTYPE: DType](
    v1x: Scalar[DTYPE], v1y: Scalar[DTYPE], v1z: Scalar[DTYPE],
    v2x: Scalar[DTYPE], v2y: Scalar[DTYPE], v2z: Scalar[DTYPE],
    v3x: Scalar[DTYPE], v3y: Scalar[DTYPE], v3z: Scalar[DTYPE],
) -> Array[Scalar[DTYPE], 4]:
    """`projectOriginPlane` — the origin projected onto the plane v1 v2 v3.

    Returns `(ok, x, y, z)` with `ok == 0` on the reference's failure return.

    ⚠ THE THREE ATTEMPTS DIFFER ONLY IN WHICH VERTEX SUPPLIES `nv`. The three
    cross products are the same vector — `(b-a) x (c-a)` is invariant under
    cycling the triangle — so `nn` is identical in all three and the reference
    is choosing between three floating-point spellings of `n . v`, which are
    equal in exact arithmetic and are not equal here. The last one divides by
    `nn` unguarded; that is safe for the same reason, because the first branch
    already returned if `nn` was zero.
    """
    var d21x = v2x - v1x
    var d21y = v2y - v1y
    var d21z = v2z - v1z
    var d31x = v3x - v1x
    var d31y = v3y - v1y
    var d31z = v3z - v1z
    var d32x = v3x - v2x
    var d32y = v3y - v2y
    var d32z = v3z - v2z

    var out = Array[Scalar[DTYPE], 4](fill=Scalar[DTYPE](0))

    # n = (v3 - v2) x (v2 - v1)
    var nx = d32y * d21z - d32z * d21y
    var ny = d32z * d21x - d32x * d21z
    var nz = d32x * d21y - d32y * d21x
    var nv = nx * v2x + ny * v2y + nz * v2z
    var nn = nx * nx + ny * ny + nz * nz
    if nn == Scalar[DTYPE](0):
        return out^
    if nv != Scalar[DTYPE](0) and nn > Scalar[DTYPE](EPA_MINVAL):
        var s = nv / nn
        out[0] = Scalar[DTYPE](1)
        out[1] = nx * s
        out[2] = ny * s
        out[3] = nz * s
        return out^

    # n = (v2 - v1) x (v3 - v1)
    nx = d21y * d31z - d21z * d31y
    ny = d21z * d31x - d21x * d31z
    nz = d21x * d31y - d21y * d31x
    nv = nx * v1x + ny * v1y + nz * v1z
    nn = nx * nx + ny * ny + nz * nz
    if nn == Scalar[DTYPE](0):
        return out^
    if nv != Scalar[DTYPE](0) and nn > Scalar[DTYPE](EPA_MINVAL):
        var s = nv / nn
        out[0] = Scalar[DTYPE](1)
        out[1] = nx * s
        out[2] = ny * s
        out[3] = nz * s
        return out^

    # n = (v3 - v1) x (v3 - v2)
    nx = d31y * d32z - d31z * d32y
    ny = d31z * d32x - d31x * d32z
    nz = d31x * d32y - d31y * d32x
    nv = nx * v3x + ny * v3y + nz * v3z
    nn = nx * nx + ny * ny + nz * nz
    var s2 = nv / nn
    out[0] = Scalar[DTYPE](1)
    out[1] = nx * s2
    out[2] = ny * s2
    out[3] = nz * s2
    return out^


@always_inline
def tri_affine_coord[DTYPE: DType](
    v1x: Scalar[DTYPE], v1y: Scalar[DTYPE], v1z: Scalar[DTYPE],
    v2x: Scalar[DTYPE], v2y: Scalar[DTYPE], v2z: Scalar[DTYPE],
    v3x: Scalar[DTYPE], v3y: Scalar[DTYPE], v3z: Scalar[DTYPE],
    px: Scalar[DTYPE], py: Scalar[DTYPE], pz: Scalar[DTYPE],
) -> Array[Scalar[DTYPE], 3]:
    """`triAffineCoord` — barycentric coordinates of `p` on triangle v1 v2 v3.

    ⚠ NOT A GRAM SOLVE. The reference drops the axis with the SMALLEST minor
    and computes three signed areas in the remaining 2D plane; that choice of
    projection plane is what makes it stable on a sliver, and it is a different
    rounding path from the `d00 d11 - d01 d01` normal-equations form this
    replaced.
    """
    var m14 = (
        v2y * v3z - v2z * v3y - v1y * v3z + v1z * v3y + v1y * v2z - v1z * v2y
    )
    var m24 = (
        v2x * v3z - v2z * v3x - v1x * v3z + v1z * v3x + v1x * v2z - v1z * v2x
    )
    var m34 = (
        v2x * v3y - v2y * v3x - v1x * v3y + v1y * v3x + v1x * v2y - v1y * v2x
    )

    var mu1 = abs(m14)
    var mu2 = abs(m24)
    var mu3 = abs(m34)

    var m_max = m14
    # (x, y) are the two surviving axes; 0 = x, 1 = y, 2 = z.
    var v1a = v1y
    var v1b = v1z
    var v2a = v2y
    var v2b = v2z
    var v3a = v3y
    var v3b = v3z
    var pa = py
    var pb = pz
    if mu1 >= mu2 and mu1 >= mu3:
        pass
    elif mu2 >= mu3:
        m_max = m24
        v1a = v1x
        v1b = v1z
        v2a = v2x
        v2b = v2z
        v3a = v3x
        v3b = v3z
        pa = px
        pb = pz
    else:
        m_max = m34
        v1a = v1x
        v1b = v1y
        v2a = v2x
        v2b = v2y
        v3a = v3x
        v3b = v3y
        pa = px
        pb = py

    var c31 = pa * v2b + pb * v3a + v2a * v3b - pa * v3b - pb * v2a - v3a * v2b
    var c32 = pa * v3b + pb * v1a + v3a * v1b - pa * v1b - pb * v3a - v1a * v3b
    var c33 = pa * v1b + pb * v2a + v1a * v2b - pa * v2b - pb * v1a - v2a * v1b

    var out = Array[Scalar[DTYPE], 3](uninitialized=True)
    out[0] = c31 / m_max
    out[1] = c32 / m_max
    out[2] = c33 / m_max
    return out^


@always_inline
def tri_point_intersect[DTYPE: DType](
    v1x: Scalar[DTYPE], v1y: Scalar[DTYPE], v1z: Scalar[DTYPE],
    v2x: Scalar[DTYPE], v2y: Scalar[DTYPE], v2z: Scalar[DTYPE],
    v3x: Scalar[DTYPE], v3y: Scalar[DTYPE], v3z: Scalar[DTYPE],
    px: Scalar[DTYPE], py: Scalar[DTYPE], pz: Scalar[DTYPE],
) -> Bool:
    """`triPointIntersect` — is `p` a point of the triangle v1 v2 v3."""
    var l = tri_affine_coord[DTYPE](
        v1x, v1y, v1z, v2x, v2y, v2z, v3x, v3y, v3z, px, py, pz
    )
    if l[0] < Scalar[DTYPE](0) or l[1] < Scalar[DTYPE](0) or l[2] < Scalar[
        DTYPE
    ](0):
        return False
    var prx = v1x * l[0] + v2x * l[1] + v3x * l[2]
    var pry = v1y * l[0] + v2y * l[1] + v3y * l[2]
    var prz = v1z * l[0] + v2z * l[1] + v3z * l[2]
    var dx = prx - px
    var dy = pry - py
    var dz = prz - pz
    return sqrt(dx * dx + dy * dy + dz * dz) < Scalar[DTYPE](EPA_MINVAL)


@always_inline
def same_side[DTYPE: DType](
    p0x: Scalar[DTYPE], p0y: Scalar[DTYPE], p0z: Scalar[DTYPE],
    p1x: Scalar[DTYPE], p1y: Scalar[DTYPE], p1z: Scalar[DTYPE],
    p2x: Scalar[DTYPE], p2y: Scalar[DTYPE], p2z: Scalar[DTYPE],
    p3x: Scalar[DTYPE], p3y: Scalar[DTYPE], p3z: Scalar[DTYPE],
) -> Bool:
    """`sameSide` — are the origin and p3 on one side of the plane p0 p1 p2."""
    var d1x = p1x - p0x
    var d1y = p1y - p0y
    var d1z = p1z - p0z
    var d2x = p2x - p0x
    var d2y = p2y - p0y
    var d2z = p2z - p0z
    var nx = d1y * d2z - d1z * d2y
    var ny = d1z * d2x - d1x * d2z
    var nz = d1x * d2y - d1y * d2x
    var dot1 = nx * (p3x - p0x) + ny * (p3y - p0y) + nz * (p3z - p0z)
    var dot2 = nx * (-p0x) + ny * (-p0y) + nz * (-p0z)
    if dot1 > Scalar[DTYPE](0) and dot2 > Scalar[DTYPE](0):
        return True
    if dot1 < Scalar[DTYPE](0) and dot2 < Scalar[DTYPE](0):
        return True
    return False


@always_inline
def test_tetra[DTYPE: DType](
    p0x: Scalar[DTYPE], p0y: Scalar[DTYPE], p0z: Scalar[DTYPE],
    p1x: Scalar[DTYPE], p1y: Scalar[DTYPE], p1z: Scalar[DTYPE],
    p2x: Scalar[DTYPE], p2y: Scalar[DTYPE], p2z: Scalar[DTYPE],
    p3x: Scalar[DTYPE], p3y: Scalar[DTYPE], p3z: Scalar[DTYPE],
) -> Bool:
    """`testTetra` — is the origin inside the tetrahedron p0 p1 p2 p3."""
    return (
        same_side[DTYPE](p0x, p0y, p0z, p1x, p1y, p1z, p2x, p2y, p2z, p3x, p3y, p3z)
        and same_side[DTYPE](p1x, p1y, p1z, p2x, p2y, p2z, p3x, p3y, p3z, p0x, p0y, p0z)
        and same_side[DTYPE](p2x, p2y, p2z, p3x, p3y, p3z, p0x, p0y, p0z, p1x, p1y, p1z)
        and same_side[DTYPE](p3x, p3y, p3z, p0x, p0y, p0z, p1x, p1y, p1z, p2x, p2y, p2z)
    )


@always_inline
def rotmat120[DTYPE: DType](
    ax: Scalar[DTYPE], ay: Scalar[DTYPE], az: Scalar[DTYPE]
) -> Array[Scalar[DTYPE], 9]:
    """`rotmat` — 120 degrees about `axis`.

    ⚠ TRANSCRIBED INCLUDING ITS TYPO. `R[6]` in the reference is
    `u1*u3*(1-cos) - u2*sin`, which is `R[2]`'s formula with a flipped sign
    rather than the `u3*u1*(1-cos) - u2*sin` a correct Rodrigues matrix has —
    the same value, since `u1*u3 == u3*u1`, so it is only a typo in the naming.
    Kept as written because `polytope2` uses this matrix to place three support
    directions and any change moves them.
    """
    var n = sqrt(ax * ax + ay * ay + az * az)
    var u1 = ax / n
    var u2 = ay / n
    var u3 = az / n
    var s = Scalar[DTYPE](0.86602540378)
    var c = Scalar[DTYPE](-0.5)
    var omc = Scalar[DTYPE](1) - c
    var out = Array[Scalar[DTYPE], 9](uninitialized=True)
    out[0] = c + u1 * u1 * omc
    out[1] = u1 * u2 * omc - u3 * s
    out[2] = u1 * u3 * omc + u2 * s
    out[3] = u2 * u1 * omc + u3 * s
    out[4] = c + u2 * u2 * omc
    out[5] = u2 * u3 * omc - u1 * s
    out[6] = u1 * u3 * omc - u2 * s
    out[7] = u2 * u3 * omc + u1 * s
    out[8] = c + u3 * u3 * omc
    return out^


@always_inline
def ray_triangle[DTYPE: DType](
    v1x: Scalar[DTYPE], v1y: Scalar[DTYPE], v1z: Scalar[DTYPE],
    v2x: Scalar[DTYPE], v2y: Scalar[DTYPE], v2z: Scalar[DTYPE],
    v3x: Scalar[DTYPE], v3y: Scalar[DTYPE], v3z: Scalar[DTYPE],
    v4x: Scalar[DTYPE], v4y: Scalar[DTYPE], v4z: Scalar[DTYPE],
    v5x: Scalar[DTYPE], v5y: Scalar[DTYPE], v5z: Scalar[DTYPE],
) -> Int:
    """`rayTriangle` — does the ray v1v2 meet the triangle v3v4v5 (+-1 / 0)."""
    var a1 = v2x - v1x
    var a2 = v2y - v1y
    var a3 = v2z - v1z
    var b1 = v3x - v1x
    var b2 = v3y - v1y
    var b3 = v3z - v1z
    var c1 = v4x - v1x
    var c2 = v4y - v1y
    var c3 = v4z - v1z
    var d1 = v5x - v1x
    var d2 = v5y - v1y
    var d3 = v5z - v1z

    # det3(u, v, w) = u . (v x w)
    var vol1 = (
        b1 * (c2 * a3 - c3 * a2)
        + b2 * (c3 * a1 - c1 * a3)
        + b3 * (c1 * a2 - c2 * a1)
    )
    var vol2 = (
        c1 * (d2 * a3 - d3 * a2)
        + c2 * (d3 * a1 - d1 * a3)
        + c3 * (d1 * a2 - d2 * a1)
    )
    var vol3 = (
        d1 * (b2 * a3 - b3 * a2)
        + d2 * (b3 * a1 - b1 * a3)
        + d3 * (b1 * a2 - b2 * a1)
    )
    var z = Scalar[DTYPE](0)
    if vol1 >= z and vol2 >= z and vol3 >= z:
        return 1
    if vol1 <= z and vol2 <= z and vol3 <= z:
        return -1
    return 0


# ---------------------------------------------------------------------------
# the polytope
# ---------------------------------------------------------------------------


@always_inline
def attach_face[DTYPE: DType, L_WS: TensorLayout](
    ws: TileTensor[DTYPE, L_WS, MutAnyOrigin],
    wrow: Int,
    f: Int,
    v1: Int,
    v2: Int,
    v3: Int,
    adj1: Int,
    adj2: Int,
    adj3: Int,
) -> Scalar[DTYPE]:
    """`attachFace` — write face `f` and return its squared distance.

    ⚠ THE PROJECTION IS CALLED WITH THE VERTICES REVERSED (`v3, v2, v1`), and
    that is not cosmetic: `projectOriginPlane` picks the first of three
    floating-point spellings of `n . v` that is non-zero, so the argument
    order decides which one wins.

    ⚠ THE ORIENTATION TEST IS AGAINST THE POLYTOPE'S CENTRE, not against
    `n . a >= 0`. Those agree only while the origin is strictly inside; the
    reference uses the centre because on a flat polytope the origin can sit on
    the wrong side of a face's plane and `n . a >= 0` then flips a normal that
    is already correct.

    ⚠ THE SLOT IS CONSUMED EVEN ON FAILURE, exactly as the reference consumes
    it (`&pt->faces[pt->nfaces++]` runs before the projection). The caller must
    advance `nfaces` whatever this returns. On failure the reference leaves
    `dist2` and `index` uninitialised and every caller bails immediately; we
    write `-2` / `0` instead, so a stray read is a deleted face rather than
    whatever the row last held.
    """
    ws[wrow, CCD_WS_EF + f * 3 + 0] = Scalar[DTYPE](v1)
    ws[wrow, CCD_WS_EF + f * 3 + 1] = Scalar[DTYPE](v2)
    ws[wrow, CCD_WS_EF + f * 3 + 2] = Scalar[DTYPE](v3)
    set_eadj[DTYPE, L_WS](ws, wrow, f, 0, adj1)
    set_eadj[DTYPE, L_WS](ws, wrow, f, 1, adj2)
    set_eadj[DTYPE, L_WS](ws, wrow, f, 2, adj3)

    var p = project_origin_plane[DTYPE](
        ev[DTYPE, L_WS](ws, wrow, v3, 0),
        ev[DTYPE, L_WS](ws, wrow, v3, 1),
        ev[DTYPE, L_WS](ws, wrow, v3, 2),
        ev[DTYPE, L_WS](ws, wrow, v2, 0),
        ev[DTYPE, L_WS](ws, wrow, v2, 1),
        ev[DTYPE, L_WS](ws, wrow, v2, 2),
        ev[DTYPE, L_WS](ws, wrow, v1, 0),
        ev[DTYPE, L_WS](ws, wrow, v1, 1),
        ev[DTYPE, L_WS](ws, wrow, v1, 2),
    )
    if p[0] == Scalar[DTYPE](0):
        ws[wrow, CCD_WS_EFD + f] = Scalar[DTYPE](0)
        set_efi[DTYPE, L_WS](ws, wrow, f, -2)
        return Scalar[DTYPE](0)

    var vx = p[1]
    var vy = p[2]
    var vz = p[3]

    var ox = ev[DTYPE, L_WS](ws, wrow, v1, 0) - rebind[Scalar[DTYPE]](
        ws[wrow, CCD_WS_CTR + 0]
    )
    var oy = ev[DTYPE, L_WS](ws, wrow, v1, 1) - rebind[Scalar[DTYPE]](
        ws[wrow, CCD_WS_CTR + 1]
    )
    var oz = ev[DTYPE, L_WS](ws, wrow, v1, 2) - rebind[Scalar[DTYPE]](
        ws[wrow, CCD_WS_CTR + 2]
    )
    if vx * ox + vy * oy + vz * oz < Scalar[DTYPE](0):
        vx = -vx
        vy = -vy
        vz = -vz

    ws[wrow, CCD_WS_EFV + f * 3 + 0] = vx
    ws[wrow, CCD_WS_EFV + f * 3 + 1] = vy
    ws[wrow, CCD_WS_EFV + f * 3 + 2] = vz
    var d2 = vx * vx + vy * vy + vz * vz
    ws[wrow, CCD_WS_EFD + f] = d2
    set_efi[DTYPE, L_WS](ws, wrow, f, -1)
    return d2


@always_inline
def delete_face[DTYPE: DType, L_WS: TensorLayout](
    ws: TileTensor[DTYPE, L_WS, MutAnyOrigin],
    wrow: Int,
    f: Int,
    mut nmap: Int,
):
    """`deleteFace` — swap-remove from `map`, then mark the face deleted."""
    var idx = efi[DTYPE, L_WS](ws, wrow, f)
    if idx >= 0:
        nmap -= 1
        var last = emap[DTYPE, L_WS](ws, wrow, nmap)
        set_emap[DTYPE, L_WS](ws, wrow, idx, last)
        set_efi[DTYPE, L_WS](ws, wrow, last, idx)
    set_efi[DTYPE, L_WS](ws, wrow, f, -2)


@always_inline
def get_edge[DTYPE: DType, L_WS: TensorLayout](
    ws: TileTensor[DTYPE, L_WS, MutAnyOrigin], wrow: Int, f: Int, vertex: Int
) -> Int:
    """`getEdge` — which edge of `f` starts at `vertex`."""
    if ef[DTYPE, L_WS](ws, wrow, f, 0) == vertex:
        return 0
    if ef[DTYPE, L_WS](ws, wrow, f, 1) == vertex:
        return 1
    return 2


@always_inline
def _add_edge[DTYPE: DType, L_WS: TensorLayout](
    ws: TileTensor[DTYPE, L_WS, MutAnyOrigin],
    wrow: Int,
    mut nhor: Int,
    face: Int,
    edge: Int,
):
    """`addEdge`."""
    if nhor >= EPA_F_CAP:
        return
    ws[wrow, CCD_WS_HOR + nhor * 2 + 0] = Scalar[DTYPE](face)
    ws[wrow, CCD_WS_HOR + nhor * 2 + 1] = Scalar[DTYPE](edge)
    nhor += 1


@always_inline
def _visible[DTYPE: DType, L_WS: TensorLayout](
    ws: TileTensor[DTYPE, L_WS, MutAnyOrigin],
    wrow: Int,
    f: Int,
    wx: Scalar[DTYPE],
    wy: Scalar[DTYPE],
    wz: Scalar[DTYPE],
) -> Bool:
    """`dot3(face->v, w) - face->dist2 > mjMINVAL`.

    ⚠ THE THRESHOLD IS ON `v . w - |v|^2`, WHICH HAS UNITS OF METRES SQUARED.
    The EPA this replaced tested `n_hat . (w - a) > 1e-14` — the same predicate
    divided by `|v|`, so its epsilon scaled with the penetration depth and the
    reference's does not.
    """
    var d = (
        efv[DTYPE, L_WS](ws, wrow, f, 0) * wx
        + efv[DTYPE, L_WS](ws, wrow, f, 1) * wy
        + efv[DTYPE, L_WS](ws, wrow, f, 2) * wz
    )
    return d - efd[DTYPE, L_WS](ws, wrow, f) > Scalar[DTYPE](EPA_MINVAL)


@always_inline
def _horizon_rec[DTYPE: DType, L_WS: TensorLayout](
    ws: TileTensor[DTYPE, L_WS, MutAnyOrigin],
    wrow: Int,
    f0: Int,
    e0: Int,
    wx: Scalar[DTYPE],
    wy: Scalar[DTYPE],
    wz: Scalar[DTYPE],
    mut nmap: Int,
    mut nhor: Int,
) -> Int:
    """`horizonRec`, with the recursion unrolled onto an explicit stack.

    Each frame is `(face, edge, state)`; the state machine walks the two edges
    the reference walks (`k = 1, 2`, i.e. `(e+1)%3` then `(e+2)%3`), descends
    into an adjacent face that is still alive, and adds the edge when the
    descent reports the adjacent face was NOT visible. `ret` carries the
    reference's return value back up.

    ⚠ THE ORDER IS THE POINT. `horizon.indices[0]` seeds the fan of new faces
    attached after the expansion, and every subsequent adjacency is written
    relative to it, so a traversal that visits the same faces in a different
    order builds a different polytope.
    """
    var sp = 0
    ws[wrow, CCD_WS_HSTK + 0] = Scalar[DTYPE](f0)
    ws[wrow, CCD_WS_HSTK + 1] = Scalar[DTYPE](e0)
    ws[wrow, CCD_WS_HSTK + 2] = Scalar[DTYPE](0)
    sp = 1
    var ret = 0

    # ⚠ BOUNDED. Every frame either pops or descends into a face that is
    # deleted on entry, so the traversal cannot exceed four visits per face;
    # spelling that as a trip count rather than `while sp > 0` keeps the loop
    # reducible for back ends that need one.
    for _step in range(4 * EPA_F_CAP):
        if sp <= 0:
            break
        var base = CCD_WS_HSTK + (sp - 1) * 3
        var f = Int(rebind[Scalar[DTYPE]](ws[wrow, base + 0]))
        var e = Int(rebind[Scalar[DTYPE]](ws[wrow, base + 1]))
        var st = Int(rebind[Scalar[DTYPE]](ws[wrow, base + 2]))

        if st == 0:
            if not _visible[DTYPE, L_WS](ws, wrow, f, wx, wy, wz):
                sp -= 1
                ret = 0
                continue
            delete_face[DTYPE, L_WS](ws, wrow, f, nmap)
            ws[wrow, base + 2] = Scalar[DTYPE](1)
            continue

        if st == 1 or st == 3:
            var k = 1 if st == 1 else 2
            var i = (e + k) % 3
            var a = eadj[DTYPE, L_WS](ws, wrow, f, i)
            ws[wrow, base + 2] = Scalar[DTYPE](st + 1)
            if efi[DTYPE, L_WS](ws, wrow, a) > -2:
                var vtx = ef[DTYPE, L_WS](ws, wrow, f, (i + 1) % 3)
                var ae = get_edge[DTYPE, L_WS](ws, wrow, a, vtx)
                if sp >= EPA_F_CAP:
                    # Out of stack: treat the descent as "not visible", which
                    # is the conservative answer — it adds the edge and stops
                    # deleting, leaving a valid horizon.
                    ret = 0
                    continue
                var nb = CCD_WS_HSTK + sp * 3
                ws[wrow, nb + 0] = Scalar[DTYPE](a)
                ws[wrow, nb + 1] = Scalar[DTYPE](ae)
                ws[wrow, nb + 2] = Scalar[DTYPE](0)
                sp += 1
            else:
                # The reference's `if (adjFace->index > -2)` guards BOTH the
                # recursion and the `addEdge`, so a dead neighbour contributes
                # no horizon edge at all.
                ws[wrow, base + 2] = Scalar[DTYPE](st + 2)
            continue

        if st == 2 or st == 4:
            var k = 1 if st == 2 else 2
            var i = (e + k) % 3
            if ret == 0:
                var a = eadj[DTYPE, L_WS](ws, wrow, f, i)
                var vtx = ef[DTYPE, L_WS](ws, wrow, f, (i + 1) % 3)
                var ae = get_edge[DTYPE, L_WS](ws, wrow, a, vtx)
                _add_edge[DTYPE, L_WS](ws, wrow, nhor, a, ae)
            ws[wrow, base + 2] = Scalar[DTYPE](st + 1)
            continue

        # st == 5: both edges done
        sp -= 1
        ret = 1

    return ret


@always_inline
def horizon[DTYPE: DType, L_WS: TensorLayout](
    ws: TileTensor[DTYPE, L_WS, MutAnyOrigin],
    wrow: Int,
    f0: Int,
    wx: Scalar[DTYPE],
    wy: Scalar[DTYPE],
    wz: Scalar[DTYPE],
    mut nmap: Int,
    mut nhor: Int,
):
    """`horizon` — delete the closest face and walk out from its three edges.

    ⚠ THE FIRST EDGE HAS NO `index > -2` GUARD in the reference and the other
    two do. Nothing has been deleted yet except `f0` itself when the first is
    taken, so the guard would be vacuous there — but writing it in anyway would
    not be, because `f0`'s own deletion is what the guard would see.
    """
    var v0 = ef[DTYPE, L_WS](ws, wrow, f0, 0)
    var v1 = ef[DTYPE, L_WS](ws, wrow, f0, 1)
    var v2 = ef[DTYPE, L_WS](ws, wrow, f0, 2)
    var a0 = eadj[DTYPE, L_WS](ws, wrow, f0, 0)
    var a1 = eadj[DTYPE, L_WS](ws, wrow, f0, 1)
    var a2 = eadj[DTYPE, L_WS](ws, wrow, f0, 2)

    delete_face[DTYPE, L_WS](ws, wrow, f0, nmap)

    var e0 = get_edge[DTYPE, L_WS](ws, wrow, a0, v1)
    if _horizon_rec[DTYPE, L_WS](
        ws, wrow, a0, e0, wx, wy, wz, nmap, nhor
    ) == 0:
        _add_edge[DTYPE, L_WS](ws, wrow, nhor, a0, e0)

    var e1 = get_edge[DTYPE, L_WS](ws, wrow, a1, v2)
    if efi[DTYPE, L_WS](ws, wrow, a1) > -2:
        if _horizon_rec[DTYPE, L_WS](
            ws, wrow, a1, e1, wx, wy, wz, nmap, nhor
        ) == 0:
            _add_edge[DTYPE, L_WS](ws, wrow, nhor, a1, e1)

    var e2 = get_edge[DTYPE, L_WS](ws, wrow, a2, v0)
    if efi[DTYPE, L_WS](ws, wrow, a2) > -2:
        if _horizon_rec[DTYPE, L_WS](
            ws, wrow, a2, e2, wx, wy, wz, nmap, nhor
        ) == 0:
            _add_edge[DTYPE, L_WS](ws, wrow, nhor, a2, e2)


@always_inline
def epa_witness[DTYPE: DType, L_WS: TensorLayout](
    ws: TileTensor[DTYPE, L_WS, MutAnyOrigin], wrow: Int, f: Int
) -> Array[Scalar[DTYPE], 6]:
    """`epaWitness` — the witness points on each geom for face `f`.

    ⚠ THE CONTACT NORMAL THE REFERENCE STORES IS `normalize(x1 - x2)`, from
    these two points, NOT the unit face normal. They are the same vector in
    exact arithmetic — `x1 - x2` is the affine combination of the face's
    Minkowski vertices, which IS `face->v` — and they are not the same bits.
    """
    var i0 = ef[DTYPE, L_WS](ws, wrow, f, 0)
    var i1 = ef[DTYPE, L_WS](ws, wrow, f, 1)
    var i2 = ef[DTYPE, L_WS](ws, wrow, f, 2)
    var l = tri_affine_coord[DTYPE](
        ev[DTYPE, L_WS](ws, wrow, i0, 0),
        ev[DTYPE, L_WS](ws, wrow, i0, 1),
        ev[DTYPE, L_WS](ws, wrow, i0, 2),
        ev[DTYPE, L_WS](ws, wrow, i1, 0),
        ev[DTYPE, L_WS](ws, wrow, i1, 1),
        ev[DTYPE, L_WS](ws, wrow, i1, 2),
        ev[DTYPE, L_WS](ws, wrow, i2, 0),
        ev[DTYPE, L_WS](ws, wrow, i2, 1),
        ev[DTYPE, L_WS](ws, wrow, i2, 2),
        efv[DTYPE, L_WS](ws, wrow, f, 0),
        efv[DTYPE, L_WS](ws, wrow, f, 1),
        efv[DTYPE, L_WS](ws, wrow, f, 2),
    )
    var out = Array[Scalar[DTYPE], 6](uninitialized=True)
    for c in range(3):
        out[c] = (
            l[0] * ev[DTYPE, L_WS](ws, wrow, i0, 3 + c)
            + l[1] * ev[DTYPE, L_WS](ws, wrow, i1, 3 + c)
            + l[2] * ev[DTYPE, L_WS](ws, wrow, i2, 3 + c)
        )
        out[3 + c] = (
            l[0] * ev[DTYPE, L_WS](ws, wrow, i0, 6 + c)
            + l[1] * ev[DTYPE, L_WS](ws, wrow, i1, 6 + c)
            + l[2] * ev[DTYPE, L_WS](ws, wrow, i2, 6 + c)
        )
    return out^
