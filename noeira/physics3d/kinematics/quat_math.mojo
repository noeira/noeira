"""Quaternion mathematics for forward kinematics.

Quaternion convention: [x, y, z, w] (scalar last, same as MuJoCo).
Identity quaternion: [0, 0, 0, 1]

Functions:
- quat_mul: Quaternion multiplication
- quat_conjugate: Quaternion conjugate (inverse for unit quaternions)
- quat_rotate: Rotate a vector by a quaternion
- quat_normalize: Normalize a quaternion to unit length
- axis_angle_to_quat: Convert axis-angle to quaternion
"""

from std.math import sqrt, sin, cos, acos, atan2, abs
from std.sys import is_nvidia_gpu

# MuJoCo's `mjMINVAL` (`mjmodel.h`). `mju_normalize4` skips the division when
# `mju_abs(norm - 1) <= mjMINVAL`; the compiler's `mjuu_normvec` uses `mjEPS`
# = 1e-14, ten times looser, on the same test. The RUNTIME value is the one
# these two functions implement.
comptime _MJMINVAL: Float64 = 1e-15


# =============================================================================
# Quaternion Multiplication
# =============================================================================


def quat_mul[
    DTYPE: DType
](
    ax: Scalar[DTYPE],
    ay: Scalar[DTYPE],
    az: Scalar[DTYPE],
    aw: Scalar[DTYPE],
    bx: Scalar[DTYPE],
    by: Scalar[DTYPE],
    bz: Scalar[DTYPE],
    bw: Scalar[DTYPE],
) -> Tuple[Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE]]:
    """Multiply two quaternions: result = a * b.

    Args:
        ax: First quaternion x.
        ay: First quaternion y.
        az: First quaternion z.
        aw: First quaternion w.
        bx: Second quaternion x.
        by: Second quaternion y.
        bz: Second quaternion z.
        bw: Second quaternion w.

    Returns:
        Product quaternion [x, y, z, w].
    """
    # Hamilton product for quaternions in [x, y, z, w] format
    var rx = aw * bx + ax * bw + ay * bz - az * by
    var ry = aw * by - ax * bz + ay * bw + az * bx
    var rz = aw * bz + ax * by - ay * bx + az * bw
    var rw = aw * bw - ax * bx - ay * by - az * bz
    return (rx, ry, rz, rw)


# =============================================================================
# Quaternion Conjugate
# =============================================================================


def quat_conjugate[
    DTYPE: DType
](
    qx: Scalar[DTYPE],
    qy: Scalar[DTYPE],
    qz: Scalar[DTYPE],
    qw: Scalar[DTYPE],
) -> Tuple[Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE]]:
    """Compute quaternion conjugate (inverse for unit quaternions).

    Args:
        qx: Quaternion x.
        qy: Quaternion y.
        qz: Quaternion z.
        qw: Quaternion w.

    Returns:
        Conjugate quaternion [-x, -y, -z, w].
    """
    return (-qx, -qy, -qz, qw)


# =============================================================================
# Quaternion Rotation of Vector
# =============================================================================


def quat_rotate[
    DTYPE: DType
](
    qx: Scalar[DTYPE],
    qy: Scalar[DTYPE],
    qz: Scalar[DTYPE],
    qw: Scalar[DTYPE],
    vx: Scalar[DTYPE],
    vy: Scalar[DTYPE],
    vz: Scalar[DTYPE],
) -> Tuple[Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE]]:
    """Rotate a vector by a quaternion.

    Computes: q * v * q^(-1) using the efficient formula:
    v' = v + 2*w*(w x v) + 2*(w x (w x v))
    where q = [qx, qy, qz, qw]

    Args:
        qx: Unit quaternion x.
        qy: Unit quaternion y.
        qz: Unit quaternion z.
        qw: Unit quaternion w.
        vx: Vector to rotate x.
        vy: Vector to rotate y.
        vz: Vector to rotate z.

    Returns:
        Rotated vector (rx, ry, rz).
    """
    # Compute 2 * (q_xyz x v)
    var tx = Scalar[DTYPE](2) * (qy * vz - qz * vy)
    var ty = Scalar[DTYPE](2) * (qz * vx - qx * vz)
    var tz = Scalar[DTYPE](2) * (qx * vy - qy * vx)

    # Result = v + w*t + (q_xyz x t)
    var rx = vx + qw * tx + (qy * tz - qz * ty)
    var ry = vy + qw * ty + (qz * tx - qx * tz)
    var rz = vz + qw * tz + (qx * ty - qy * tx)

    return (rx, ry, rz)


def quat_rotate_inverse[
    DTYPE: DType
](
    qx: Scalar[DTYPE],
    qy: Scalar[DTYPE],
    qz: Scalar[DTYPE],
    qw: Scalar[DTYPE],
    vx: Scalar[DTYPE],
    vy: Scalar[DTYPE],
    vz: Scalar[DTYPE],
) -> Tuple[Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE]]:
    """Inverse-rotate a vector by a quaternion: q^(-1) * v * q.

    Equivalent to quat_rotate with conjugate quaternion (-qx, -qy, -qz, qw).
    """
    return quat_rotate[DTYPE](-qx, -qy, -qz, qw, vx, vy, vz)


# =============================================================================
# Quaternion Normalization
# =============================================================================


def quat_normalize[
    DTYPE: DType
](
    qx: Scalar[DTYPE],
    qy: Scalar[DTYPE],
    qz: Scalar[DTYPE],
    qw: Scalar[DTYPE],
) -> Tuple[Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE]]:
    """Normalize a quaternion to unit length.

    ⚠ THE DEGENERATE GUARD IS A BRANCH, NOT AN EPSILON UNDER THE SQRT — the
    same fix `gpu_quat_normalize` got, one epsilon smaller. This was
    `1.0 / sqrt(length_sq + 1e-12)`, which returns 0.9999999999995 for an
    already-unit quaternion, so everything this touched came out 5.00e-13
    short of unit.

    IT MATTERED MORE THAN THE MAGNITUDE SUGGESTS because of where it sits:
    all three integrators renormalize the free/ball-joint quaternion in `qpos`
    through here after every step (`integrator/euler.mojo:278`,
    `implicit.mojo:190`, `rk4.mojo:133`), and `quat_integrate` below calls it
    again on the way in. So every free-jointed model carried a root quaternion
    permanently below unit — and repeated renormalisation does NOT decay to
    zero, it converges to the fixed point of `q -> q/sqrt(q^2 + 1e-12)`, which
    is |q| = 0.9999999999995. Every vector rotated by the root quaternion was
    scaled by 1 - 1e-12 forever.

    Args:
        qx: Quaternion x.
        qy: Quaternion y.
        qz: Quaternion z.
        qw: Quaternion w.

    Returns:
        Normalized unit quaternion.
    """
    var length_sq = qx * qx + qy * qy + qz * qz + qw * qw
    # ⚠⚠ THE NEAR-UNIT GUARD IS `mju_normalize4`'s, and the reasoning is
    # written out in full on `gpu_quat_normalize` below — read it there. Short
    # version: MuJoCo returns an already-unit quaternion UNTOUCHED
    # (`engine_util_blas.c:258`), and moving one by an ulp made
    # `_bb_post_filter`'s `==` duplicate removal inert.
    #
    # ⚠ THE TWO NORMALISERS MUST CARRY THE SAME RULE. This one runs on the CPU
    # leg and `gpu_quat_normalize` on the other; a guard added to one only
    # would make the two legs disagree about every body pose in the model,
    # which is a worse failure than the one being fixed.
    var length = sqrt(length_sq)
    if abs(length - Scalar[DTYPE](1)) <= Scalar[DTYPE](_MJMINVAL):
        return (qx, qy, qz, qw)

    # Degenerate input keeps the OLD formula bit-for-bit, so this is a pure
    # precision change. As with the GPU pair, that arm is believed unreachable
    # — every caller passes a quaternion that is already unit to rounding.
    var inv_length: Scalar[DTYPE]
    if length_sq < Scalar[DTYPE](1e-6):
        inv_length = Scalar[DTYPE](1.0) / sqrt(
            length_sq + Scalar[DTYPE](1e-12)
        )
    else:
        inv_length = Scalar[DTYPE](1.0) / length
    return (qx * inv_length, qy * inv_length, qz * inv_length, qw * inv_length)


# =============================================================================
# Axis-Angle to Quaternion
# =============================================================================


def axis_angle_to_quat[
    DTYPE: DType
](
    ax: Scalar[DTYPE],
    ay: Scalar[DTYPE],
    az: Scalar[DTYPE],
    angle: Scalar[DTYPE],
) -> Tuple[Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE]]:
    """Convert axis-angle representation to quaternion.

    Args:
        ax: Rotation x-component of axis (should be normalized).
        ay: Rotation y-component of axis (should be normalized).
        az: Rotation z-component of axis (should be normalized).
        angle: Rotation angle in radians.

    Returns:
        Quaternion [x, y, z, w] = [sin(θ/2)*axis, cos(θ/2)].
    """
    comptime assert (
        DTYPE.is_floating_point()
    ), "DTYPE must be a floating point type"
    var half = angle.cast[DTYPE]() * 0.5
    var sc = sin_cos_device(half)
    var s = sc[0]
    var c = sc[1]

    return (ax * s, ay * s, az * s, c)


# =============================================================================
# Quaternion to Axis-Angle (for debugging/visualization)
# =============================================================================


def quat_to_axis_angle[
    DTYPE: DType
](
    qx: Scalar[DTYPE],
    qy: Scalar[DTYPE],
    qz: Scalar[DTYPE],
    qw: Scalar[DTYPE],
) -> Tuple[Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE]]:
    """Convert quaternion to axis-angle representation.

    Args:
        qx: Quaternion x.
        qy: Quaternion y.
        qz: Quaternion z.
        qw: Quaternion w.

    Returns:
        (axis_x, axis_y, axis_z, angle) where angle is in radians.
    """

    # Ensure qw is in valid range for acos
    var w_f64 = Float64(qw).clamp(-1.0, 1.0)

    # Compute angle
    var angle = Scalar[DTYPE](2.0 * acos(w_f64))

    # Compute axis (handle near-zero angle case)
    var sin_half = sqrt(Float64(qx * qx + qy * qy + qz * qz))
    if sin_half < 1e-10:
        # Near-identity rotation, axis is arbitrary
        return (Scalar[DTYPE](0), Scalar[DTYPE](0), Scalar[DTYPE](1), angle)

    var inv_sin = Scalar[DTYPE](1.0 / sin_half)
    return (qx * inv_sin, qy * inv_sin, qz * inv_sin, angle)


# =============================================================================
# Quaternion Integration (for integrating angular velocity)
# =============================================================================


def quat_integrate[
    DTYPE: DType
](
    qx: Scalar[DTYPE],
    qy: Scalar[DTYPE],
    qz: Scalar[DTYPE],
    qw: Scalar[DTYPE],
    wx: Scalar[DTYPE],
    wy: Scalar[DTYPE],
    wz: Scalar[DTYPE],
    dt: Scalar[DTYPE],
) -> Tuple[Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE]]:
    """Integrate quaternion with angular velocity, EXACTLY as MuJoCo does.

    `q(t+dt) = normalize(q(t)) * axisAngle(omega/|omega|, dt*|omega|)` —
    a RIGHT multiplication, i.e. omega is in the BODY-LOCAL frame, matching
    MuJoCo's free/ball-joint qvel convention.

    Verbatim `mju_quatIntegrate` (engine_util_spatial.c:241): normalize the
    axis, scale the angle by dt, build the rotation quaternion, normalize the
    current quat, right-multiply. The rotation applied is the exact
    exponential map, NOT a truncated series.

    ⚠ This was a FIRST-ORDER approximation (`q + 0.5*dt*q*omega`, then
    normalize) until 2026-07-30, which is a different integrator from
    MuJoCo's for any free- or ball-jointed model. Cost: the root quaternion
    was the single largest state discrepancy vs MuJoCo at EVERY step of a
    dm_control humanoid rollout (1.1e-8 after one step, where the 21 hinge
    DOFs sat at 4e-10), and it compounds. Only free-rooted models were
    affected, and their gates were loose enough to miss it — the Ant Euler
    gate budgets 1e-4 on qpos.

    Args:
        qx: Current orientation quaternion x.
        qy: Current orientation quaternion y.
        qz: Current orientation quaternion z.
        qw: Current orientation quaternion w.
        wx: Angular velocity x in body-local frame.
        wy: Angular velocity y in body-local frame.
        wz: Angular velocity z in body-local frame.
        dt: Time step.

    Returns:
        Updated (normalized) quaternion.
    """
    comptime assert (
        DTYPE.is_floating_point()
    ), "DTYPE must be a floating point type"
    # mju_normalize3 on the angular velocity; below mjMINVAL MuJoCo leaves the
    # angle at 0, which makes the rotation the identity.
    var wn = sqrt(wx * wx + wy * wy + wz * wz)
    var q = quat_normalize(qx, qy, qz, qw)
    if wn < Scalar[DTYPE](1e-15):
        return q

    var angle = dt * wn
    var half = Scalar[DTYPE](0.5) * angle
    var sc = sin_cos_device(half)
    var s = sc[0] / wn  # fold the axis normalization in
    var c = sc[1]

    var rx = wx * s
    var ry = wy * s
    var rz = wz * s

    # q * qrot — right multiplication (local frame).
    return quat_mul(q[0], q[1], q[2], q[3], rx, ry, rz, c)


@always_inline
def sin_cos_device[
    DTYPE: DType
](x: Scalar[DTYPE]) -> Tuple[Scalar[DTYPE], Scalar[DTYPE]]:
    """`(sin(x), cos(x))`, accurate to ~1 ulp on NVIDIA as well.

    ⚠ CUDA LOWERS float32 `std.math.sin` / `cos` TO `sin.approx.f32` /
    `cos.approx.f32` (the MUFU unit): measured on an RTX 5090, 3.3e-7 absolute
    error on [-1.6, 1.6] (Metal 1.2e-7, host 6e-8) and 1e-5 RELATIVE at
    small angles: sin(0.025) = 0.024997149 for 0.024997396. Every hinge
    quaternion in FK carried that, and it alone moved
    `test_euler_finalize_gpu_parity` from 1.5e-6 to 2.24e-5; with this
    function the 5090 reads 1.52e-6 again.

    On NVIDIA float32: cephes `sinf` / `cosf`. j = the octant of |x| rounded
    to even, a three-part Cody-Waite reduction of |x| by j * pi/4 into
    [-pi/4, pi/4], then the minimax polynomials, chosen and signed by the
    quadrant. Accurate to ~1 ulp up to |x| ~ 8192 (the reduction's limit).
    Everywhere else (the CPU, Metal, float64) it is `std.math` unchanged, so
    those results do not move. Gated in `tests/physics3d/test_sin_cos_device.mojo`.

    ⚠ IT COSTS ~8x THE MUFU PAIR IN ISOLATION, ~1% OF A STEP. 5090,
    `benchmarks/bench_sin_cos_device.mojo`: 0.078 vs 0.63 ms for 268M pairs.
    `bench_half_cheetah_batch`, A/B interleaved twice: +0.4 to +1.3% per env
    step (1 to 4096 envs, eager and graph). This is the price CUDA's own
    non-fast-math `sinf` charges; an if-chain or selects for the quadrant
    measured the same.
    """
    comptime assert (
        DTYPE.is_floating_point()
    ), "DTYPE must be a floating point type"
    comptime if is_nvidia_gpu() and DTYPE == DType.float32:
        comptime FOUR_OVER_PI = Scalar[DTYPE](1.2732395447351628)
        comptime DP1 = Scalar[DTYPE](0.78515625)
        comptime DP2 = Scalar[DTYPE](2.4187564849853515625e-4)
        comptime DP3 = Scalar[DTYPE](3.77489497744594108e-8)
        var ax = abs(x)
        var j = Int(ax * FOUR_OVER_PI)
        if (j & 1) == 1:
            j += 1
        var y = Scalar[DTYPE](j)
        var q = j & 7
        var z = ((ax - y * DP1) - y * DP2) - y * DP3
        var zz = z * z
        var ps = z + z * zz * (
            (Scalar[DTYPE](-1.9515295891e-4) * zz + Scalar[DTYPE](8.3321608736e-3))
            * zz
            - Scalar[DTYPE](1.6666654611e-1)
        )
        var pc = (
            Scalar[DTYPE](1.0)
            - Scalar[DTYPE](0.5) * zz
            + zz
            * zz
            * (
                (Scalar[DTYPE](2.443315711809948e-5) * zz - Scalar[DTYPE](1.388731625493765e-3))
                * zz
                + Scalar[DTYPE](4.166664568298827e-2)
            )
        )
        # q in {0, 2, 4, 6}: the quarter turn |x| was reduced by.
        #   q   sin   cos
        #   0   +ps   +pc
        #   2   +pc   -ps
        #   4   -ps   -pc
        #   6   -pc   +ps
        var swap = (q & 2) != 0
        var s = pc if swap else ps
        var c = ps if swap else pc
        var s_neg = ((q & 4) != 0) != (x < Scalar[DTYPE](0))
        var c_neg = ((q + 2) & 4) != 0
        return (-s if s_neg else s, -c if c_neg else c)
    else:
        return (sin(x), cos(x))


@always_inline
def atan2_device[
    DTYPE: DType
](y: Scalar[DTYPE], x: Scalar[DTYPE]) -> Scalar[DTYPE]:
    """`atan2(y, x)` from `sqrt` and arithmetic only — for code that runs in a
    GPU kernel. `std.math.atan2` is a libm symbol that does not lower on the
    device (ptxas: unresolved extern 'atan2f'; see
    `_the_stdlibs_atan2_is_a_libm_symbol_on_the_device`), and the passive
    ball-joint spring (AUD-43) needs the rotation angle inside
    `_fnet_passive_env`, which is one body for the CPU path and the kernel.

    Method: t = |y/x| reduced into [0, 1] (atan(t) = pi/2 - atan(1/t)), then
    three halvings atan(t) = 2 atan(t / (1 + sqrt(1 + t^2))) bring it below
    0.13, where a 10-term Taylor series is exact to double precision
    (0.13^21 / 21 ~ 1e-19). Quadrants are restored by sign. Gated against
    libm on the CPU in `test_ball_spring_and_tendon_damping_vs_mujoco`.
    """
    comptime PI = Scalar[DTYPE](3.141592653589793)
    comptime HALF_PI = Scalar[DTYPE](1.5707963267948966)
    var ax = abs(x)
    var ay = abs(y)
    if ax == Scalar[DTYPE](0) and ay == Scalar[DTYPE](0):
        return Scalar[DTYPE](0)
    var swap = ay > ax
    var t = (ax / ay) if swap else (ay / ax)  # in [0, 1]
    # three halvings
    for _ in range(3):
        t = t / (Scalar[DTYPE](1) + sqrt(Scalar[DTYPE](1) + t * t))
    var t2 = t * t
    var term = t
    var acc = t
    for k in range(1, 11):
        term = term * t2
        var den = Scalar[DTYPE](2 * k + 1)
        if (k & 1) == 1:
            acc = acc - term / den
        else:
            acc = acc + term / den
    var a = acc * Scalar[DTYPE](8)  # undo the three halvings
    if swap:
        a = HALF_PI - a
    if x < Scalar[DTYPE](0):
        a = PI - a
    if y < Scalar[DTYPE](0):
        a = -a
    return a


def quat2vel[
    DTYPE: DType
](
    qx: Scalar[DTYPE],
    qy: Scalar[DTYPE],
    qz: Scalar[DTYPE],
    qw: Scalar[DTYPE],
    dt: Scalar[DTYPE],
) -> Tuple[Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE]]:
    """`mju_quat2Vel` — a rotation quaternion as an angular velocity.

    Verbatim `engine_util_spatial.c:mju_quat2Vel`. The rotation is expressed as
    `axis * angle / dt`, with the axis normalised out of the vector part and
    the angle from `2*atan2(|v|, w)`.

    ⚠ DO NOT "SIMPLIFY" THIS INTO `quat_to_axis_angle` ABOVE. That function
    looks like the same thing and is not: it takes the angle from `acos(w)`
    rather than `atan2(|v|, w)`, and it has NO `angle > pi` wrap. Both
    differences change results, and the second changes them by a whole turn.

    ⚠ THE `speed > pi` WRAP IS WHAT MAKES THIS SIGN-INVARIANT, and that
    property is load-bearing for the IK caller. `q` and `-q` are the same
    rotation, but `2*atan2(|v|, -w) = 2*pi - 2*atan2(|v|, w)`, so without the
    wrap a negated quaternion would come back as the rotation the LONG WAY
    ROUND — same axis flipped, angle `2*pi - a`. With the wrap the two agree
    exactly. Measured against the runtime rather than argued: over 20000
    random unit quaternions `max|quat2Vel(q) - quat2Vel(-q)|` is 8.9e-16, and
    it is exactly 0 at the identity.

    That invariance is why the IK error chain can feed this the site's world
    quaternion composed directly (`xquat[body] * site_quat`) instead of
    porting `mju_mat2Quat` to round-trip through `site_xmat` — the round trip
    can only differ by an overall sign, which this function cannot see.

    Args:
        qx: Rotation quaternion x.
        qy: Rotation quaternion y.
        qz: Rotation quaternion z.
        qw: Rotation quaternion w.
        dt: Time to spread the rotation over; the IK caller passes 1.

    Returns:
        Angular velocity (vx, vy, vz).
    """
    comptime assert (
        DTYPE.is_floating_point()
    ), "DTYPE must be a floating point type"

    # `mju_normalize3` on the vector part. ⚠ Below mjMINVAL it substitutes the
    # axis (1,0,0) but STILL RETURNS THE ORIGINAL NORM, so `speed` is computed
    # from the tiny norm rather than from zero. Reproduced exactly; at the
    # identity both give 0 anyway, but only because the norm is what it is.
    var norm = sqrt(qx * qx + qy * qy + qz * qz)
    var ax = Scalar[DTYPE](1)
    var ay = Scalar[DTYPE](0)
    var az = Scalar[DTYPE](0)
    if norm >= Scalar[DTYPE](1e-15):  # mjMINVAL
        var inv = Scalar[DTYPE](1) / norm
        ax = qx * inv
        ay = qy * inv
        az = qz * inv

    var speed = Scalar[DTYPE](2) * Scalar[DTYPE](
        atan2(Float64(norm), Float64(qw))
    )
    if speed > Scalar[DTYPE](3.14159265358979323846):
        speed -= Scalar[DTYPE](2.0 * 3.14159265358979323846)
    speed /= dt

    return (ax * speed, ay * speed, az * speed)


# =============================================================================
# GPU Quaternion Operations (Array return for GPU compatibility)
# =============================================================================


@always_inline
def gpu_quat_mul[
    DTYPE: DType
](
    ax: Scalar[DTYPE],
    ay: Scalar[DTYPE],
    az: Scalar[DTYPE],
    aw: Scalar[DTYPE],
    bx: Scalar[DTYPE],
    by: Scalar[DTYPE],
    bz: Scalar[DTYPE],
    bw: Scalar[DTYPE],
) -> Array[Scalar[DTYPE], 4]:
    """Quaternion multiplication a * b (GPU version with Array return)."""
    var result = Array[Scalar[DTYPE], 4](uninitialized=True)
    result[0] = aw * bx + ax * bw + ay * bz - az * by
    result[1] = aw * by - ax * bz + ay * bw + az * bx
    result[2] = aw * bz + ax * by - ay * bx + az * bw
    result[3] = aw * bw - ax * bx - ay * by - az * bz
    return result^


@always_inline
def gpu_quat_rotate[
    DTYPE: DType
](
    qx: Scalar[DTYPE],
    qy: Scalar[DTYPE],
    qz: Scalar[DTYPE],
    qw: Scalar[DTYPE],
    vx: Scalar[DTYPE],
    vy: Scalar[DTYPE],
    vz: Scalar[DTYPE],
) -> Array[Scalar[DTYPE], 3]:
    """Rotate vector v by quaternion q: q * v * q^-1 (GPU version)."""
    var t_x = Scalar[DTYPE](2) * (qy * vz - qz * vy)
    var t_y = Scalar[DTYPE](2) * (qz * vx - qx * vz)
    var t_z = Scalar[DTYPE](2) * (qx * vy - qy * vx)

    var result = Array[Scalar[DTYPE], 3](uninitialized=True)
    result[0] = vx + qw * t_x + (qy * t_z - qz * t_y)
    result[1] = vy + qw * t_y + (qz * t_x - qx * t_z)
    result[2] = vz + qw * t_z + (qx * t_y - qy * t_x)
    return result^


@always_inline
def gpu_axis_angle_to_quat[
    DTYPE: DType
](
    axis_x: Scalar[DTYPE],
    axis_y: Scalar[DTYPE],
    axis_z: Scalar[DTYPE],
    angle: Scalar[DTYPE],
) -> Array[Scalar[DTYPE], 4]:
    """Convert axis-angle to quaternion (GPU version)."""
    comptime assert (
        DTYPE.is_floating_point()
    ), "DTYPE must be a floating point type"
    var half = angle.cast[DTYPE]() * 0.5
    var sc = sin_cos_device(half)
    var s = sc[0]
    var c = sc[1]

    # ⚠ THE ZERO-AXIS GUARD IS A BRANCH, NOT AN EPSILON UNDER THE SQRT. This
    # was `1.0 / sqrt(len_sq + 1e-10)`, which for the unit axis every caller
    # actually passes returns `1 - 5e-11` instead of `1` — see
    # `gpu_quat_normalize` for the full story and the measurement.
    var len_sq = axis_x * axis_x + axis_y * axis_y + axis_z * axis_z
    var result = Array[Scalar[DTYPE], 4](uninitialized=True)
    # Degenerate axis keeps the old formula, for the reason in
    # `gpu_quat_normalize` — this change is about the epsilon, not about what
    # a zero axis should mean. Like that one, this branch is unreached by every
    # model in the suite: all three callers pass a joint AXIS, which is unit.
    var inv_len: Scalar[DTYPE]
    if len_sq < Scalar[DTYPE](1e-6):
        inv_len = Scalar[DTYPE](1.0) / sqrt(len_sq + Scalar[DTYPE](1e-10))
    else:
        inv_len = Scalar[DTYPE](1.0) / sqrt(len_sq)

    result[0] = axis_x * inv_len * s
    result[1] = axis_y * inv_len * s
    result[2] = axis_z * inv_len * s
    result[3] = c
    return result^


@always_inline
def gpu_quat_normalize[
    DTYPE: DType
](
    qx: Scalar[DTYPE],
    qy: Scalar[DTYPE],
    qz: Scalar[DTYPE],
    qw: Scalar[DTYPE],
) -> Array[Scalar[DTYPE], 4]:
    """Normalize quaternion (GPU version).

    ⚠ THE DEGENERATE GUARD IS A BRANCH, NOT AN EPSILON UNDER THE SQRT. This
    was `1.0 / sqrt(norm_sq + 1e-10)`, which POISONS THE COMMON CASE to
    protect a case that never happens: for an already-unit quaternion it
    returns `1/sqrt(1 + 1e-10)` = 0.99999999995, so every quaternion this
    function touched came out 5e-11 SHORT of unit, and every vector rotated by
    one was scaled by `|q|^2` = 1 - 1e-10.

    That is not rounding — it is a deterministic one-directional bias, and it
    was the cause of BOTH of the ~1e-10 residuals that had been filed as
    separate mysteries:

      * task #48 — quadruped's forward kinematics reproducing MuJoCo to only
        ~1e-10 where other models manage 1e-15, "showing up identically in
        cvel, qfrc_bias and tendon_invweight0". Every body quaternion in the
        chain is normalized here.
      * task #49 — capsule contact normals ~1e-10 off MuJoCo's while every
        sphere pair was exact. A capsule's world AXIS is its local quaternion
        rotated by the body's; spheres have no axis to corrupt, which is
        exactly why the error looked type-specific.

    The observed body quaternion was 0.99999999995 to every digit of
    `1/sqrt(1+1e-10)`. Branch on the degenerate case instead, so the
    arithmetic is exact for the input every caller actually passes.
    """
    var norm_sq = qx * qx + qy * qy + qz * qz + qw * qw
    var result = Array[Scalar[DTYPE], 4](uninitialized=True)
    # THE DEGENERATE BRANCH KEEPS THE OLD FORMULA ON PURPOSE, so that this
    # change is a pure precision fix and nothing else. Below 1e-6 the result is
    # bit-identical to what it always was; above it, exact.
    #
    # ⚠ CORRECTION (task #50, 2026-08-03): an earlier version of this comment
    # claimed an A/B "proved the branch IS reached" because returning identity
    # there moved `test_equality_tendon_fields`'s fingerprint by 0.83. THAT WAS
    # WRONG — the A/B also changed the function's SHAPE (an early `return`
    # inside the `if` versus a fallthrough assignment), which changes FP codegen
    # in the NON-degenerate path, and this file's humanoid rests on four
    # contacts whose force split is indeterminate and amplifies last-bit
    # differences. The real test injects an absurd `inv_norm = -12345` into both
    # degenerate branches and checks the fingerprint: it comes out UNCHANGED, so
    # neither branch is reached at all, on the CPU or the GPU path. Instrumented
    # CPU runs printed nothing either.
    #
    # So this branch is dead code for every model in the test suite, and the
    # old `(0,0,0,0)`-for-zero-norm behaviour it preserves is not observable.
    # Keep it anyway: it costs one comparison, and "unreachable in this suite"
    # is not "unreachable". If you ever make it reachable, note that returning
    # a ZERO quaternion is not a rotation and identity is the sane answer.
    # ⚠⚠ AN ALREADY-UNIT QUATERNION IS RETURNED UNTOUCHED, AND THAT IS
    # `mju_normalize4` (`engine_util_blas.c:258`), NOT an optimisation:
    #
    #     } else if (mju_abs(norm - 1) > mjMINVAL) { ... vec[i] *= normInv; }
    #
    # `mjuu_normvec` (`user_util.cc:162`) carries the same rule at compile
    # time, with the comment "don't normalize if nrm is within mjEPS of 1".
    # A quaternion written to full double precision is generally NOT exactly
    # unit — `(0.8158341149610219, 0, 0, 0.5782859991956973)`, the yaw every
    # Duplo brick in `reassemble_5` carries, has `norm^2 = 0.9999999999999999`
    # — and renormalising it MOVES it by an ulp for nothing.
    #
    # ⚠ AN ULP IS NOT COSMETIC WHEN SOMETHING DOWNSTREAM COMPARES WITH `==`.
    # `_bb_post_filter` (`collision/collision_primitives.mojo`) removes
    # duplicate box/box manifold points with `pos[i] == pos[j]`, exactly as
    # `engine_collision_box.c:1394` does, and that filter is only correct
    # because MuJoCo's coincident points come out BIT-IDENTICAL. With this
    # ulp in the body quaternion they came out one ulp apart, the filter went
    # inert, and two stacked bricks handed the solver a DUPLICATED CONSTRAINT
    # ROW — an exactly rank-deficient Hessian. Gated by
    # `tests/physics3d/test_box_box_degenerate_stack.mojo`.
    var norm = sqrt(norm_sq)
    if abs(norm - Scalar[DTYPE](1)) <= Scalar[DTYPE](_MJMINVAL):
        result[0] = qx
        result[1] = qy
        result[2] = qz
        result[3] = qw
        return result^

    var inv_norm: Scalar[DTYPE]
    if norm_sq < Scalar[DTYPE](1e-6):
        inv_norm = Scalar[DTYPE](1.0) / sqrt(norm_sq + Scalar[DTYPE](1e-10))
    else:
        inv_norm = Scalar[DTYPE](1.0) / norm

    result[0] = qx * inv_norm
    result[1] = qy * inv_norm
    result[2] = qz * inv_norm
    result[3] = qw * inv_norm
    return result^
