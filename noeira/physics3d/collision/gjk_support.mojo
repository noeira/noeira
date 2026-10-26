"""GJK support functions for all geometry types.

A support function returns the point on a convex shape's surface that is
furthest along a given direction. These are the building blocks of GJK/EPA.

Reference: MuJoCo engine_collision_convex.c lines 162-398
"""

from std.math import sqrt, abs

from layout import TileTensor, TensorLayout
from ..constants import (
    GEOM_SPHERE,
    GEOM_CAPSULE,
    GEOM_BOX,
    GEOM_CYLINDER,
    GEOM_ELLIPSOID,
    GEOM_MESH,
)
from ..kinematics.quat_math import quat_rotate, quat_rotate_inverse
from .epa import project_origin_plane, sv


# ---------------------------------------------------------------------------
# the geom frame
# ---------------------------------------------------------------------------


@always_inline
def quat2mat[
    DTYPE: DType
](
    qx: Scalar[DTYPE], qy: Scalar[DTYPE], qz: Scalar[DTYPE],
    qw: Scalar[DTYPE],
) -> Tuple[
    Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE],
    Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE],
    Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE],
]:
    """`mju_quat2Mat` (`engine_util_spatial.c`), row-major.

    ⚠⚠ EVERY SUPPORT FUNCTION IN THE REFERENCE WORKS IN THE GEOM'S LOCAL FRAME
    AND GETS THERE THROUGH THIS MATRIX. `mjc_boxSupport` and the rest open with
    `mulMatTVec3(local_dir, obj->mat, dir)` and close with
    `localToGlobal(res, obj->mat, local_supp, obj->pos)`, where `obj->mat` is
    `d->geom_xmat` — which `mj_kinematics` built from the geom's quaternion with
    exactly the formula below. This engine rotated BASIS VECTORS by the
    quaternion and did the arithmetic in WORLD coordinates instead: the same
    mathematics and a different rounding path.

    ⚠⚠ IT RETURNS A TUPLE, NOT AN `Array`, AND THAT IS A GPU
    REQUIREMENT. This is `@always_inline`d into every branch of `_support`, so
    a nine-element per-thread array here is nine more stack slots per call site
    in a Metal kernel that is already at its ceiling — the heightfield GPU leg
    went from 17 contacts to 11 when this returned an array. A tuple of scalars
    stays in registers. See
    `feedback_metal_wide_per_thread_inlinearray_miscompute`.

    ⚠ THE MATRIX IS REBUILT PER CALL rather than carried in `Data`. That is
    repeated work — ten multiplies — and NOT a different answer: the reference
    derives it from the same quaternion with the same formula, so the bits
    agree.

    ⚠ THE IDENTITY SHORT-CIRCUIT IS TRANSCRIBED, not an optimisation. It makes
    an axis-aligned geom's matrix EXACTLY the identity instead of
    `1 - 2*(y^2 + z^2)`, and those differ by an ulp when the quaternion is not
    bit-exactly normalised.

    ⚠ ORDER: MuJoCo stores `(w, x, y, z)` and this engine stores `(x, y, z, w)`.
    """
    if (
        qw == Scalar[DTYPE](1)
        and qx == Scalar[DTYPE](0)
        and qy == Scalar[DTYPE](0)
        and qz == Scalar[DTYPE](0)
    ):
        return (
            Scalar[DTYPE](1), Scalar[DTYPE](0), Scalar[DTYPE](0),
            Scalar[DTYPE](0), Scalar[DTYPE](1), Scalar[DTYPE](0),
            Scalar[DTYPE](0), Scalar[DTYPE](0), Scalar[DTYPE](1),
        )
    var q00 = qw * qw
    var q01 = qw * qx
    var q02 = qw * qy
    var q03 = qw * qz
    var q11 = qx * qx
    var q12 = qx * qy
    var q13 = qx * qz
    var q22 = qy * qy
    var q23 = qy * qz
    var q33 = qz * qz
    return (
        q00 + q11 - q22 - q33,
        Scalar[DTYPE](2) * (q12 - q03),
        Scalar[DTYPE](2) * (q13 + q02),
        Scalar[DTYPE](2) * (q12 + q03),
        q00 - q11 + q22 - q33,
        Scalar[DTYPE](2) * (q23 - q01),
        Scalar[DTYPE](2) * (q13 - q02),
        Scalar[DTYPE](2) * (q23 + q01),
        q00 - q11 - q22 + q33,
    )


@always_inline
def mat_t_vec3[
    DTYPE: DType
](
    m0: Scalar[DTYPE], m1: Scalar[DTYPE], m2: Scalar[DTYPE],
    m3: Scalar[DTYPE], m4: Scalar[DTYPE], m5: Scalar[DTYPE],
    m6: Scalar[DTYPE], m7: Scalar[DTYPE], m8: Scalar[DTYPE],
    dx: Scalar[DTYPE], dy: Scalar[DTYPE], dz: Scalar[DTYPE],
) -> Tuple[Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE]]:
    """`mulMatTVec3` — world direction into the geom's local frame."""
    return (
        m0 * dx + m3 * dy + m6 * dz,
        m1 * dx + m4 * dy + m7 * dz,
        m2 * dx + m5 * dy + m8 * dz,
    )


@always_inline
def local_to_global[
    DTYPE: DType
](
    m0: Scalar[DTYPE], m1: Scalar[DTYPE], m2: Scalar[DTYPE],
    m3: Scalar[DTYPE], m4: Scalar[DTYPE], m5: Scalar[DTYPE],
    m6: Scalar[DTYPE], m7: Scalar[DTYPE], m8: Scalar[DTYPE],
    lx: Scalar[DTYPE], ly: Scalar[DTYPE], lz: Scalar[DTYPE],
    px: Scalar[DTYPE], py: Scalar[DTYPE], pz: Scalar[DTYPE],
) -> Array[Scalar[DTYPE], 3]:
    """`localToGlobal` — `mat * local + pos`, with the add written LAST exactly
    as the reference writes it.

    ⚠ RETURNS THE `Array` THE SUPPORTS ALREADY RETURN, so this adds no
    slot of its own — it is the existing return value, filled in place.
    """
    var r = Array[Scalar[DTYPE], 3](uninitialized=True)
    r[0] = m0 * lx + m1 * ly + m2 * lz
    r[1] = m3 * lx + m4 * ly + m5 * lz
    r[2] = m6 * lx + m7 * ly + m8 * lz
    r[0] += px
    r[1] += py
    r[2] += pz
    return r^


@always_inline
def support_sphere[
    DTYPE: DType,
](
    dir_x: Scalar[DTYPE],
    dir_y: Scalar[DTYPE],
    dir_z: Scalar[DTYPE],
    pos_x: Scalar[DTYPE],
    pos_y: Scalar[DTYPE],
    pos_z: Scalar[DTYPE],
    radius: Scalar[DTYPE],
) -> Array[Scalar[DTYPE], 3]:
    """Support point on sphere: center + radius * dir."""
    var result = Array[Scalar[DTYPE], 3](fill=Scalar[DTYPE](0))
    result[0] = pos_x + radius * dir_x
    result[1] = pos_y + radius * dir_y
    result[2] = pos_z + radius * dir_z
    return result^


@always_inline
def support_capsule[
    DTYPE: DType,
](
    dir_x: Scalar[DTYPE],
    dir_y: Scalar[DTYPE],
    dir_z: Scalar[DTYPE],
    pos_x: Scalar[DTYPE],
    pos_y: Scalar[DTYPE],
    pos_z: Scalar[DTYPE],
    qx: Scalar[DTYPE],
    qy: Scalar[DTYPE],
    qz: Scalar[DTYPE],
    qw: Scalar[DTYPE],
    radius: Scalar[DTYPE],
    half_length: Scalar[DTYPE],
) -> Array[Scalar[DTYPE], 3]:
    """`mjc_capsuleSupport` (`engine_collision_convex.c`), in the LOCAL frame.

        mulMatTVec3(local_dir, mat, dir)
        local_supp = local_dir * radius
        local_supp[2] += (local_dir[2] >= 0 ? length : -length)
        localToGlobal(res, mat, local_supp, pos)

    ⚠ THE SPHERE PART IS `local_dir * radius`, NOT the world `dir * radius`
    added to a rotated endpoint. The two are equal in exact arithmetic — a
    rotation preserves the norm — and are not equal in floating point, and this
    engine took the second route.
    """
    var m = quat2mat[DTYPE](qx, qy, qz, qw)
    var ld = mat_t_vec3[DTYPE](
        m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8],
        dir_x, dir_y, dir_z,
    )
    var l0 = ld[0] * radius
    var l1 = ld[1] * radius
    var l2 = ld[2] * radius
    l2 += half_length if ld[2] >= Scalar[DTYPE](0) else -half_length
    return local_to_global[DTYPE](
        m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8],
        l0, l1, l2, pos_x, pos_y, pos_z,
    )


@always_inline
def support_box[
    DTYPE: DType,
](
    dir_x: Scalar[DTYPE],
    dir_y: Scalar[DTYPE],
    dir_z: Scalar[DTYPE],
    pos_x: Scalar[DTYPE],
    pos_y: Scalar[DTYPE],
    pos_z: Scalar[DTYPE],
    qx: Scalar[DTYPE],
    qy: Scalar[DTYPE],
    qz: Scalar[DTYPE],
    qw: Scalar[DTYPE],
    half_x: Scalar[DTYPE],
    half_y: Scalar[DTYPE],
    half_z: Scalar[DTYPE],
    mut corner: Int,
) -> Array[Scalar[DTYPE], 3]:
    """`mjc_boxSupport` — the sign of each LOCAL component, in the local frame.

    ⚠ `warm` comes back as MuJoCo's `obj->vertindex`, which for a box is the
    CORNER CODE `(x>0) | (y>0)<<1 | (z>0)<<2` and not a mesh vertex. The
    reference stores both in the same field, and EPA's discrete
    repeated-support break compares it.
    """
    var m = quat2mat[DTYPE](qx, qy, qz, qw)
    var ld = mat_t_vec3[DTYPE](
        m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8],
        dir_x, dir_y, dir_z,
    )
    var l0 = half_x if ld[0] >= Scalar[DTYPE](0) else -half_x
    var l1 = half_y if ld[1] >= Scalar[DTYPE](0) else -half_y
    var l2 = half_z if ld[2] >= Scalar[DTYPE](0) else -half_z
    var code = 0
    if l0 > Scalar[DTYPE](0):
        code |= 1
    if l1 > Scalar[DTYPE](0):
        code |= 2
    if l2 > Scalar[DTYPE](0):
        code |= 4
    corner = code
    return local_to_global[DTYPE](
        m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8],
        l0, l1, l2, pos_x, pos_y, pos_z,
    )


@always_inline
def support_ellipsoid[
    DTYPE: DType,
](
    dir_x: Scalar[DTYPE],
    dir_y: Scalar[DTYPE],
    dir_z: Scalar[DTYPE],
    pos_x: Scalar[DTYPE],
    pos_y: Scalar[DTYPE],
    pos_z: Scalar[DTYPE],
    qx: Scalar[DTYPE],
    qy: Scalar[DTYPE],
    qz: Scalar[DTYPE],
    qw: Scalar[DTYPE],
    half_x: Scalar[DTYPE],
    half_y: Scalar[DTYPE],
    half_z: Scalar[DTYPE],
) -> Array[Scalar[DTYPE], 3]:
    """`mjc_ellipsoidSupport` (`engine_collision_convex.c:562`), verbatim.

        res = local_dir * size          (elementwise)
        normalize(res)
        res = res * size                (elementwise)

    i.e. `(a^2 dx, b^2 dy, c^2 dz) / |(a dx, b dy, c dz)|` in the geom frame.

    ⚠⚠ WITHOUT THIS AN ELLIPSOID IS A POINT. `_support`'s fallback returns the
    geom's CENTRE for a type it does not know, silently, so every ellipsoid
    pair that is not against a plane collided as if the ellipsoid were a
    zero-radius dot at its origin — no error, no warning, just no contact.
    MuJoCo sends EVERY ellipsoid pair except plane to `mjc_Convex`
    (`mjCOLLISIONFUNC` row ELLIPSOID), including sphere x ellipsoid.
    """
    var m = quat2mat[DTYPE](qx, qy, qz, qw)
    var ld = mat_t_vec3[DTYPE](
        m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8],
        dir_x, dir_y, dir_z,
    )
    var l0 = ld[0] * half_x
    var l1 = ld[1] * half_y
    var l2 = ld[2] * half_z
    var norm2 = l0 * l0 + l1 * l1 + l2 * l2
    # ⚠ THE DEGENERATE BRANCH IS THE FIRST COLUMN OF THE MATRIX, not a
    # normalise-to-(1,0,0)-then-rotate. `mjc_ellipsoidSupport` writes
    # `res[i] = mat[3*i] * size[0] + pos[i]` outright, and the threshold is
    # `norm2 < mjMINVAL^2` on the SQUARED norm rather than 1e-15 on the norm.
    if norm2 < Scalar[DTYPE](1e-30):
        var deg = Array[Scalar[DTYPE], 3](uninitialized=True)
        deg[0] = m[0] * half_x + pos_x
        deg[1] = m[3] * half_x + pos_y
        deg[2] = m[6] * half_x + pos_z
        return deg^
    var inv = Scalar[DTYPE](1) / sqrt(norm2)
    l0 = l0 * inv * half_x
    l1 = l1 * inv * half_y
    l2 = l2 * inv * half_z
    return local_to_global[DTYPE](
        m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8],
        l0, l1, l2, pos_x, pos_y, pos_z,
    )


@always_inline
def support_cylinder[
    DTYPE: DType,
](
    dir_x: Scalar[DTYPE],
    dir_y: Scalar[DTYPE],
    dir_z: Scalar[DTYPE],
    pos_x: Scalar[DTYPE],
    pos_y: Scalar[DTYPE],
    pos_z: Scalar[DTYPE],
    qx: Scalar[DTYPE],
    qy: Scalar[DTYPE],
    qz: Scalar[DTYPE],
    qw: Scalar[DTYPE],
    radius: Scalar[DTYPE],
    half_length: Scalar[DTYPE],
) -> Array[Scalar[DTYPE], 3]:
    """`mjc_cylinderSupport` (`engine_collision_convex.c`), in the LOCAL frame.

        mulMatTVec3(local_dir, mat, dir)
        n2  = local_dir[0]^2 + local_dir[1]^2
        scl = n2 >= mjMINVAL^2 ? size[0] / sqrt(n2) : 0
        local_supp = (scl*local_dir[0], scl*local_dir[1],
                      local_dir[2] >= 0 ? size[1] : -size[1])
        localToGlobal(res, mat, local_supp, pos)

    ⚠⚠ THE DEGENERATE BRANCH RETURNS THE CAP CENTRE, `scl = 0`. This engine
    returned an ARBITRARY RIM POINT (the geom's local x axis scaled by the
    radius) when `dir` was parallel to the axis — a completely different
    support point — and it took that branch on a threshold of `1e-10` against
    the norm where the reference uses `mjMINVAL^2` against the SQUARED norm,
    so it fired a hundred thousand times more often.
    """
    var m = quat2mat[DTYPE](qx, qy, qz, qw)
    var ld = mat_t_vec3[DTYPE](
        m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8],
        dir_x, dir_y, dir_z,
    )
    var n2 = ld[0] * ld[0] + ld[1] * ld[1]
    var scl = Scalar[DTYPE](0)
    if n2 >= Scalar[DTYPE](1e-30):
        scl = radius / sqrt(n2)
    var l0 = scl * ld[0]
    var l1 = scl * ld[1]
    var l2 = half_length if ld[2] >= Scalar[DTYPE](0) else -half_length
    return local_to_global[DTYPE](
        m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8],
        l0, l1, l2, pos_x, pos_y, pos_z,
    )


@always_inline
def support_mesh[
    DTYPE: DType,
](
    dir_x: Scalar[DTYPE],
    dir_y: Scalar[DTYPE],
    dir_z: Scalar[DTYPE],
    pos_x: Scalar[DTYPE],
    pos_y: Scalar[DTYPE],
    pos_z: Scalar[DTYPE],
    qx: Scalar[DTYPE],
    qy: Scalar[DTYPE],
    qz: Scalar[DTYPE],
    qw: Scalar[DTYPE],
    verts: List[Scalar[DTYPE]],
    vert_offset: Int,
    num_verts: Int,
) -> Array[Scalar[DTYPE], 3]:
    """Support point on mesh: exhaustive scan of hull vertices.

    Vertices are stored in local frame. We rotate dir to local frame,
    find max-dot vertex, then transform to world frame. O(n) per call,
    but n is typically 50-200 for robot convex hulls.
    """
    # Rotate direction to local frame
    var local_dir = quat_rotate_inverse[DTYPE](
        qx, qy, qz, qw, dir_x, dir_y, dir_z
    )
    var ld_x = local_dir[0]
    var ld_y = local_dir[1]
    var ld_z = local_dir[2]

    # Exhaustive scan for max dot product
    var best_dot: Scalar[DTYPE] = -1e30
    var best_idx = 0
    for i in range(num_verts):
        var off = vert_offset + i * 3
        var d = (
            ld_x * verts[off] + ld_y * verts[off + 1] + ld_z * verts[off + 2]
        )
        if d > best_dot:
            best_dot = d
            best_idx = i

    # Transform best vertex to world frame
    var off = vert_offset + best_idx * 3
    var local_pt = quat_rotate[DTYPE](
        qx, qy, qz, qw, verts[off], verts[off + 1], verts[off + 2]
    )

    var result = Array[Scalar[DTYPE], 3](fill=Scalar[DTYPE](0))
    result[0] = pos_x + local_pt[0]
    result[1] = pos_y + local_pt[1]
    result[2] = pos_z + local_pt[2]
    return result^


@always_inline
def _dot3[
    DTYPE: DType
](
    ax: Scalar[DTYPE],
    ay: Scalar[DTYPE],
    az: Scalar[DTYPE],
    bx: Scalar[DTYPE],
    by: Scalar[DTYPE],
    bz: Scalar[DTYPE],
) -> Scalar[DTYPE]:
    return ax * bx + ay * by + az * bz


@always_inline
def _cross3[
    DTYPE: DType
](
    ax: Scalar[DTYPE],
    ay: Scalar[DTYPE],
    az: Scalar[DTYPE],
    bx: Scalar[DTYPE],
    by: Scalar[DTYPE],
    bz: Scalar[DTYPE],
) -> Tuple[Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE]]:
    return (ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx)


def _project_origin_line[
    DTYPE: DType
](
    v1x: Scalar[DTYPE], v1y: Scalar[DTYPE], v1z: Scalar[DTYPE],
    v2x: Scalar[DTYPE], v2y: Scalar[DTYPE], v2z: Scalar[DTYPE],
) -> Array[Scalar[DTYPE], 3]:
    """`projectOriginLine` — the origin projected onto the line v1 v2.

        res = v2 - <v2, v2 - v1> / <v2 - v1, v2 - v1> * (v2 - v1)
    """
    var dx = v2x - v1x
    var dy = v2y - v1y
    var dz = v2z - v1z
    var scl = -(
        (v2x * dx + v2y * dy + v2z * dz) / (dx * dx + dy * dy + dz * dz)
    )
    var out = Array[Scalar[DTYPE], 3](uninitialized=True)
    out[0] = v2x + scl * dx
    out[1] = v2y + scl * dy
    out[2] = v2z + scl * dz
    return out^


@always_inline
def _same_sign2[DTYPE: DType](a: Scalar[DTYPE], b: Scalar[DTYPE]) -> Int:
    """`sameSign2` — 1 if both positive, -1 if both negative, 0 otherwise."""
    if a > Scalar[DTYPE](0) and b > Scalar[DTYPE](0):
        return 1
    if a < Scalar[DTYPE](0) and b < Scalar[DTYPE](0):
        return -1
    return 0


@always_inline
def _s1d[
    DTYPE: DType
](
    s1x: Scalar[DTYPE], s1y: Scalar[DTYPE], s1z: Scalar[DTYPE],
    s2x: Scalar[DTYPE], s2y: Scalar[DTYPE], s2z: Scalar[DTYPE],
) -> Array[Scalar[DTYPE], 2]:
    """`S1D` — barycentric coordinates of the origin's closest point on a segment.

    ⚠ THE FALLBACK IS `(0, 1)`, NOT A CLAMP. When the projection falls outside
    the segment the reference returns the SECOND vertex outright — the newest
    support point, which is the one GJK just added. Clamping to the nearer end
    would keep the older vertex and stall the descent.
    """
    var p = _project_origin_line[DTYPE](s1x, s1y, s1z, s2x, s2y, s2z)

    # the axis with the largest projection "shadow" of the simplex
    # ⚠ `>=` ON BOTH COMPARISONS, so a tie goes to the LATER axis — the
    # opposite of `triAffineCoord`'s tie rule, and transcribed as written.
    var mu_max = s1x - s2x
    var index = 0
    var mu = s1y - s2y
    if abs(mu) >= abs(mu_max):
        mu_max = mu
        index = 1
    mu = s1z - s2z
    if abs(mu) >= abs(mu_max):
        mu_max = mu
        index = 2

    var pi = p[0]
    var a1 = s1x
    var a2 = s2x
    if index == 1:
        pi = p[1]
        a1 = s1y
        a2 = s2y
    elif index == 2:
        pi = p[2]
        a1 = s1z
        a2 = s2z

    var c1 = pi - a2
    var c2 = a1 - pi
    var same = (
        _same_sign2[DTYPE](mu_max, c1) != 0
        and _same_sign2[DTYPE](mu_max, c2) != 0
    )
    var out = Array[Scalar[DTYPE], 2](uninitialized=True)
    if same:
        out[0] = c1 / mu_max
        out[1] = c2 / mu_max
    else:
        out[0] = Scalar[DTYPE](0)
        out[1] = Scalar[DTYPE](1)
    return out^


@always_inline
def _s2d[
    DTYPE: DType
](
    s1x: Scalar[DTYPE], s1y: Scalar[DTYPE], s1z: Scalar[DTYPE],
    s2x: Scalar[DTYPE], s2y: Scalar[DTYPE], s2z: Scalar[DTYPE],
    s3x: Scalar[DTYPE], s3y: Scalar[DTYPE], s3z: Scalar[DTYPE],
) -> Array[Scalar[DTYPE], 3]:
    """`S2D` — barycentric coordinates of the origin's closest point on a triangle."""
    var pr = project_origin_plane[DTYPE](
        s1x, s1y, s1z, s2x, s2y, s2z, s3x, s3y, s3z
    )
    var out = Array[Scalar[DTYPE], 3](fill=Scalar[DTYPE](0))
    if pr[0] == Scalar[DTYPE](0):
        # degenerate plane: drop to the segment s1 s2
        var l1 = _s1d[DTYPE](s1x, s1y, s1z, s2x, s2y, s2z)
        out[0] = l1[0]
        out[1] = l1[1]
        out[2] = Scalar[DTYPE](0)
        return out^
    var pox = pr[1]
    var poy = pr[2]
    var poz = pr[3]

    var m14 = (
        s2y * s3z - s2z * s3y - s1y * s3z + s1z * s3y + s1y * s2z - s1z * s2y
    )
    var m24 = (
        s2x * s3z - s2z * s3x - s1x * s3z + s1z * s3x + s1x * s2z - s1z * s2x
    )
    var m34 = (
        s2x * s3y - s2y * s3x - s1x * s3y + s1y * s3x + s1x * s2y - s1y * s2x
    )

    var mu1 = abs(m14)
    var mu2 = abs(m24)
    var mu3 = abs(m34)
    var m_max = m14
    var a1 = s1y
    var b1 = s1z
    var a2 = s2y
    var b2 = s2z
    var a3 = s3y
    var b3 = s3z
    var pa = poy
    var pb = poz
    if mu1 >= mu2 and mu1 >= mu3:
        pass
    elif mu2 >= mu3:
        m_max = m24
        a1 = s1x
        b1 = s1z
        a2 = s2x
        b2 = s2z
        a3 = s3x
        b3 = s3z
        pa = pox
        pb = poz
    else:
        m_max = m34
        a1 = s1x
        b1 = s1y
        a2 = s2x
        b2 = s2y
        a3 = s3x
        b3 = s3y
        pa = pox
        pb = poy

    var c31 = pa * b2 + pb * a3 + a2 * b3 - pa * b3 - pb * a2 - a3 * b2
    var c32 = pa * b3 + pb * a1 + a3 * b1 - pa * b1 - pb * a3 - a1 * b3
    var c33 = pa * b1 + pb * a2 + a1 * b2 - pa * b2 - pb * a1 - a2 * b1

    var comp1 = _same_sign2[DTYPE](m_max, c31) != 0
    var comp2 = _same_sign2[DTYPE](m_max, c32) != 0
    var comp3 = _same_sign2[DTYPE](m_max, c33) != 0

    if comp1 and comp2 and comp3:
        out[0] = c31 / m_max
        out[1] = c32 / m_max
        out[2] = c33 / m_max
        return out^

    var dmin = Scalar[DTYPE](1e30)
    if not comp1:
        var l = _s1d[DTYPE](s2x, s2y, s2z, s3x, s3y, s3z)
        var xx = l[0] * s2x + l[1] * s3x
        var xy = l[0] * s2y + l[1] * s3y
        var xz = l[0] * s2z + l[1] * s3z
        dmin = xx * xx + xy * xy + xz * xz
        out[0] = Scalar[DTYPE](0)
        out[1] = l[0]
        out[2] = l[1]
    if not comp2:
        var l = _s1d[DTYPE](s1x, s1y, s1z, s3x, s3y, s3z)
        var xx = l[0] * s1x + l[1] * s3x
        var xy = l[0] * s1y + l[1] * s3y
        var xz = l[0] * s1z + l[1] * s3z
        var d = xx * xx + xy * xy + xz * xz
        if d < dmin:
            dmin = d
            out[0] = l[0]
            out[1] = Scalar[DTYPE](0)
            out[2] = l[1]
    if not comp3:
        var l = _s1d[DTYPE](s1x, s1y, s1z, s2x, s2y, s2z)
        var xx = l[0] * s1x + l[1] * s2x
        var xy = l[0] * s1y + l[1] * s2y
        var xz = l[0] * s1z + l[1] * s2z
        var d = xx * xx + xy * xy + xz * xz
        if d < dmin:
            out[0] = l[0]
            out[1] = l[1]
            out[2] = Scalar[DTYPE](0)
    return out^


@always_inline
def _det3[
    DTYPE: DType
](
    ax: Scalar[DTYPE], ay: Scalar[DTYPE], az: Scalar[DTYPE],
    bx: Scalar[DTYPE], by: Scalar[DTYPE], bz: Scalar[DTYPE],
    cx: Scalar[DTYPE], cy: Scalar[DTYPE], cz: Scalar[DTYPE],
) -> Scalar[DTYPE]:
    """`det3` — the determinant with columns a, b, c, written as `a . (b x c)`."""
    return (
        ax * (by * cz - bz * cy)
        + ay * (bz * cx - bx * cz)
        + az * (bx * cy - by * cx)
    )


def _subdistance[
    DTYPE: DType, L_WS: TensorLayout
](
    ws: TileTensor[DTYPE, L_WS, MutAnyOrigin],
    wrow: Int,
    base: Int,
    n: Int,
) -> Array[Scalar[DTYPE], 4]:
    """`subdistance` / `S3D` — the barycentric coordinates of the point in the
    simplex closest to the origin. Montanari et al, ToG 2017.

    ⚠⚠ IT DOES NOT TOUCH THE SIMPLEX. The reference returns only `lambda`, and
    the CALLER drops the vertices whose coordinate is EXACTLY zero, in order.
    The routine this replaced reduced and re-packed the simplex itself and
    returned the closest POINT, which threw `lambda` away — so the witness
    points at the end of GJK were a UNIFORM average of the surviving vertices
    rather than `lincomb(lambda, ...)`, and the caller had no way to spell the
    reference's `equal3(x_next, x_k)` convergence break.

    ⚠ WHICH VERTEX IS DROPPED IS THE WHOLE POINT. The retained set is what
    `polytope2/3/4` seeds EPA from, so a different tie-break here is a
    different seed, a different final face and a different contact normal.
    """
    var lam = Array[Scalar[DTYPE], 4](fill=Scalar[DTYPE](0))
    if n < 2:
        lam[0] = Scalar[DTYPE](1)
        return lam^
    if n == 2:
        var l = _s1d[DTYPE](
            sv(ws, wrow, base, 0, 0), sv(ws, wrow, base, 0, 1), sv(ws, wrow, base, 0, 2),
            sv(ws, wrow, base, 1, 0), sv(ws, wrow, base, 1, 1), sv(ws, wrow, base, 1, 2),
        )
        lam[0] = l[0]
        lam[1] = l[1]
        return lam^
    if n == 3:
        var l = _s2d[DTYPE](
            sv(ws, wrow, base, 0, 0), sv(ws, wrow, base, 0, 1), sv(ws, wrow, base, 0, 2),
            sv(ws, wrow, base, 1, 0), sv(ws, wrow, base, 1, 1), sv(ws, wrow, base, 1, 2),
            sv(ws, wrow, base, 2, 0), sv(ws, wrow, base, 2, 1), sv(ws, wrow, base, 2, 2),
        )
        lam[0] = l[0]
        lam[1] = l[1]
        lam[2] = l[2]
        return lam^

    # S3D
    var s1x = sv(ws, wrow, base, 0, 0)
    var s1y = sv(ws, wrow, base, 0, 1)
    var s1z = sv(ws, wrow, base, 0, 2)
    var s2x = sv(ws, wrow, base, 1, 0)
    var s2y = sv(ws, wrow, base, 1, 1)
    var s2z = sv(ws, wrow, base, 1, 2)
    var s3x = sv(ws, wrow, base, 2, 0)
    var s3y = sv(ws, wrow, base, 2, 1)
    var s3z = sv(ws, wrow, base, 2, 2)
    var s4x = sv(ws, wrow, base, 3, 0)
    var s4y = sv(ws, wrow, base, 3, 1)
    var s4z = sv(ws, wrow, base, 3, 2)

    var c41 = -_det3[DTYPE](s2x, s2y, s2z, s3x, s3y, s3z, s4x, s4y, s4z)
    var c42 = _det3[DTYPE](s1x, s1y, s1z, s3x, s3y, s3z, s4x, s4y, s4z)
    var c43 = -_det3[DTYPE](s1x, s1y, s1z, s2x, s2y, s2z, s4x, s4y, s4z)
    var c44 = _det3[DTYPE](s1x, s1y, s1z, s2x, s2y, s2z, s3x, s3y, s3z)
    var m_det = c41 + c42 + c43 + c44

    var comp1 = _same_sign2[DTYPE](m_det, c41) != 0
    var comp2 = _same_sign2[DTYPE](m_det, c42) != 0
    var comp3 = _same_sign2[DTYPE](m_det, c43) != 0
    var comp4 = _same_sign2[DTYPE](m_det, c44) != 0

    if comp1 and comp2 and comp3 and comp4:
        lam[0] = c41 / m_det
        lam[1] = c42 / m_det
        lam[2] = c43 / m_det
        lam[3] = c44 / m_det
        return lam^

    var dmin = Scalar[DTYPE](1e30)
    if not comp1:
        var l = _s2d[DTYPE](
            s2x, s2y, s2z, s3x, s3y, s3z, s4x, s4y, s4z
        )
        var xx = l[0] * s2x + l[1] * s3x + l[2] * s4x
        var xy = l[0] * s2y + l[1] * s3y + l[2] * s4y
        var xz = l[0] * s2z + l[1] * s3z + l[2] * s4z
        dmin = xx * xx + xy * xy + xz * xz
        lam[0] = Scalar[DTYPE](0)
        lam[1] = l[0]
        lam[2] = l[1]
        lam[3] = l[2]
    if not comp2:
        var l = _s2d[DTYPE](
            s1x, s1y, s1z, s3x, s3y, s3z, s4x, s4y, s4z
        )
        var xx = l[0] * s1x + l[1] * s3x + l[2] * s4x
        var xy = l[0] * s1y + l[1] * s3y + l[2] * s4y
        var xz = l[0] * s1z + l[1] * s3z + l[2] * s4z
        var d = xx * xx + xy * xy + xz * xz
        if d < dmin:
            dmin = d
            lam[0] = l[0]
            lam[1] = Scalar[DTYPE](0)
            lam[2] = l[1]
            lam[3] = l[2]
    if not comp3:
        var l = _s2d[DTYPE](
            s1x, s1y, s1z, s2x, s2y, s2z, s4x, s4y, s4z
        )
        var xx = l[0] * s1x + l[1] * s2x + l[2] * s4x
        var xy = l[0] * s1y + l[1] * s2y + l[2] * s4y
        var xz = l[0] * s1z + l[1] * s2z + l[2] * s4z
        var d = xx * xx + xy * xy + xz * xz
        if d < dmin:
            dmin = d
            lam[0] = l[0]
            lam[1] = l[1]
            lam[2] = Scalar[DTYPE](0)
            lam[3] = l[2]
    if not comp4:
        var l = _s2d[DTYPE](
            s1x, s1y, s1z, s2x, s2y, s2z, s3x, s3y, s3z
        )
        var xx = l[0] * s1x + l[1] * s2x + l[2] * s3x
        var xy = l[0] * s1y + l[1] * s2y + l[2] * s3y
        var xz = l[0] * s1z + l[1] * s2z + l[2] * s3z
        var d = xx * xx + xy * xy + xz * xz
        if d < dmin:
            lam[0] = l[0]
            lam[1] = l[1]
            lam[2] = l[2]
            lam[3] = Scalar[DTYPE](0)
    return lam^


@always_inline
def support_prism[
    DTYPE: DType
](
    dir_x: Scalar[DTYPE],
    dir_y: Scalar[DTYPE],
    dir_z: Scalar[DTYPE],
    prism: Array[Scalar[DTYPE], 18],
) -> Array[Scalar[DTYPE], 3]:
    """Support point of a HEIGHTFIELD PRISM — six explicit vertices.

    `mjc_ConvexHField` (`engine_collision_convex.c:1125`) does not collide the
    heightfield; it walks the sub-grid under the other geom's AABB and collides
    one triangular PRISM per cell. `obj1.data.hfield.prism` holds six points —
    three on the base at `z = -size[3]`, three on the sampled surface — and
    `mjc_prismSupport` is `argmax_i dot(prism[i], dir)`.

    ⚠ NO POSE. The prism is built directly in the heightfield's local frame and
    the OTHER geom is transformed into that frame by the caller
    (`obj2.pos = local_pos`, `obj2.mat = mat1^T mat2`), so there is no
    translation or rotation to apply here. `_support`'s `pos`/`quat` arguments
    are ignored for this type, which is why the caller passes the identity.

    ⚠⚠ IT SEARCHES THREE VERTICES, NOT SIX, AND THAT IS THE REFERENCE'S OWN
    SHORTCUT:

        istart = dir[2] < 0 ? 0 : 3;

    — a downward direction can only be extremal on the BASE triangle and an
    upward one only on the TOP. A true six-vertex support is a DIFFERENT
    function wherever the two disagree, which is exactly the near-horizontal
    directions GJK spends most of its iterations on. Measured on an 8x8
    fixture: searching all six invented a contact between a sphere and a prism
    whose nearest point is 1.9 cm away, at a depth of -4.0e-03.

    ⚠ NOTE WHERE THE BOUNDARY FALLS. `dir[2] < 0` is strict, so a perfectly
    horizontal direction takes the TOP triangle.

    ⚠ TIES GO TO THE FIRST. MuJoCo's loop is `if (dot > best)`, strictly
    greater, so a direction perpendicular to an edge returns the
    lowest-indexed vertex of it. A `>=` here would pick a different witness on
    every flat contact and move the EPA polytope's seed.
    """
    var istart = 0 if dir_z < Scalar[DTYPE](0) else 3
    var best = istart
    var bestdot = (
        prism[istart * 3 + 0] * dir_x
        + prism[istart * 3 + 1] * dir_y
        + prism[istart * 3 + 2] * dir_z
    )
    for k in range(1, 3):
        var i = istart + k
        var d = (
            prism[i * 3 + 0] * dir_x
            + prism[i * 3 + 1] * dir_y
            + prism[i * 3 + 2] * dir_z
        )
        if d > bestdot:
            bestdot = d
            best = i
    var out = Array[Scalar[DTYPE], 3](uninitialized=True)
    out[0] = prism[best * 3 + 0]
    out[1] = prism[best * 3 + 1]
    out[2] = prism[best * 3 + 2]
    return out^
