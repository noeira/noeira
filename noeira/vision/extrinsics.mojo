# +--------------------------------------------------------------------------+ #
# | Camera -> robot base, solved from correspondences the arm already knows
# +--------------------------------------------------------------------------+ #
"""The rigid fit that turns a marker pose into a number the policy can use.

`docs/VISION_ASSESSMENT_2026_09_09.md` §2 item 1, and the reason `svd_3x3` was
added to the OpenCV shim's group F in the first place — it had **one caller,
its own test**, until this file.

## What this is for

A detected ArUco marker gives a pose in the CAMERA frame. Every policy in this
tree reasons in the ROBOT BASE frame. Nothing converted between the two, which
is why `deploy_reach_real.mojo` still picks its target as "three numbers WE
pick": there was no way to read one off the world.

The conversion is a rigid transform, and the arm can supply the correspondences
to solve for it without any extra instrument:

  1. put a marker on the gripper at a MEASURED offset;
  2. move the arm to N poses (torque off, by hand, is fine and safer);
  3. at each pose, `p_base` = forward kinematics on the measured `qpos`
     (gated in `tests/robots/test_so_arm101_vs_mujoco.mojo`), and `p_cam` =
     `solve_pnp` on the detected marker;
  4. `fit_rigid` returns `R`, `t` with `p_base = R p_cam + t`.

⚠ **THIS IS NOT HAND-EYE CALIBRATION, AND IT CANNOT BE.**
`cv2.calibrateHandEye` is **not in this OpenCV 5.0 build** (checked, and
recorded in `docs/OPENCV_SHIM_SCOPE.md`). This solves the easier problem that
a FIXED camera admits: 3D-3D correspondences by Kabsch, no AX = XB. Move the
camera after the fit and the fit is void — `DM_CONTROL_AND_CAMERA_ASSESSMENT`
§10.1 says the telescopic arms are the risk, not the solve.

## ⚠⚠ EVERYTHING HERE IS METRES IN, MILLIMETRES OUT

Inputs are metres because both producers are: `physics3d` FK is metres and
`solve_pnp` returns metres when its object points are metres. Outputs quote
millimetres because the residual is the number that decides whether
reach-to-a-marker works, and nobody judges 0.0031 as easily as 3.1.

⚠ A UNITS MIX DOES NOT FAIL, IT SCALES. Feeding millimetres to one side and
metres to the other produces a perfectly convergent fit around a rotation that
is nonsense, with a residual 1000x too large — large enough to notice, which is
the only reason this is a warning and not a guard.

## ⚠⚠ THE DEFECT THIS FILE IS BUILT AROUND: A REFLECTION FITS TOO

`R = V Uᵀ` is the textbook one-liner and it is WRONG, occasionally and
silently. The maximiser of `tr(R H)` over all ORTHOGONAL matrices includes
reflections, so on noisy or near-degenerate data the naive form can return a
matrix with `det = -1`: a mirror world, which fits the points it was given
*better than any rotation can* and describes a robot that does not exist. The
fix is one sign — `R = V diag(1, 1, det(V Uᵀ)) Uᵀ` — and
`tests/vision/test_extrinsics_kabsch.mojo` falsifies it by handing the fit a
deliberately mirrored point set: the corrected form must return `det = +1` AND
a large residual, where the naive form returns the reflection and ~zero.

## ⚠⚠ AND THE RESIDUAL IS NOT THE WHOLE STORY, FOR THE SAME REASON AS `rms`

`camera_studio.mojo` learned this on the intrinsics side and it transfers
exactly: a residual only says the model explains the points it was given. Poses
clustered in one small volume fit beautifully and leave the transform
under-constrained everywhere else. So the fit also reports **spread** — the RMS
extent of the sampled points along their three principal axes, in mm — and
REFUSES a collinear set outright, because three points on a line leave the
rotation about that line completely free while fitting to zero.
"""

from std.math import abs, sqrt

from noeira.math3d import Mat3 as Mat3Generic, Vec3 as Vec3Generic

from .opencv import svd_3x3

comptime Vec3d = Vec3Generic[DType.float64]
comptime Mat3d = Mat3Generic[DType.float64]

comptime MIN_POINTS = 3
"""Three non-collinear correspondences determine a rigid transform exactly.
⚠ EXACTLY, WITH ZERO RESIDUAL, WHICH IS WHY THREE IS ALSO USELESS AS EVIDENCE:
the fit cannot disagree with the data, so `rms_mm` is 0 whatever the data says.
Use enough poses that the residual is a measurement — a dozen, spread."""

comptime COLLINEAR_MM = 1.0
"""Below this second principal extent the points are a LINE, and the rotation
about that line is unconstrained. Refuse rather than return a confident fit
whose free parameter nobody will think to check."""


@fieldwise_init
struct RigidFit(Copyable, Movable, Writable):
    """`p_base = rot * p_cam + trans`, with the numbers that judge it."""

    var rot: Mat3d
    """Camera -> base rotation. `det = +1`, checked, never a reflection."""

    var trans: Vec3d
    """Camera origin expressed in the base frame, metres."""

    var rms_mm: Float64
    """RMS of `|R p_cam + t - p_base|` over the correspondences.

    ⚠ THIS IS THE NUMBER THE WHOLE VISION TRACK IS JUDGED BY — it is in
    MILLIMETRES IN THE ROBOT FRAME, which is the frame the gripper has to
    arrive in. It also absorbs a principal-point bias fully at constant depth
    and partially across a working volume, which is why
    `docs/OPENCV_SHIM_SCOPE.md` says to let it be the judge rather than
    over-optimising the intrinsics in isolation."""

    var max_mm: Float64
    """The worst single correspondence. ⚠ READ IT BESIDE `rms_mm`: one
    misdetected marker at 40 mm hides inside an RMS of 6 mm over twelve
    poses, and it is the outlier that says a pose was mis-taken, not the
    mean."""

    var worst: Int
    """Index of that correspondence, so it can be dropped and re-fitted."""

    var n: Int
    """Correspondences used."""

    var spread_mm: Array[Float64, 3]
    """RMS extent of the sampled points along their three principal axes, mm,
    descending.

    ⚠⚠ THE THIRD NUMBER IS THE ONE TO READ. A set of poses swept across a
    table is two-dimensional; its third extent is near zero, and the fit is
    then interpolating inside a plane and extrapolating out of it. That is not
    an error and the residual will not mention it — but a brick picked 15 cm
    above the calibration plane is being localised by an unmeasured direction.
    """

    def apply(self, p_cam: Vec3d) -> Vec3d:
        """A point in the camera frame, in the base frame. Metres both ways."""
        return self.rot * p_cam + self.trans

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "camera -> base fit, ",
            self.n,
            " poses\n",
            "  rms     ",
            self.rms_mm,
            " mm\n",
            "  worst   ",
            self.max_mm,
            " mm at pose ",
            self.worst,
            "\n",
            "  spread  ",
            self.spread_mm[0],
            " / ",
            self.spread_mm[1],
            " / ",
            self.spread_mm[2],
            " mm\n",
            "  origin  ",
            self.trans.x,
            " ",
            self.trans.y,
            " ",
            self.trans.z,
            " m\n",
        )


def _principal_spread_mm(
    centred: List[Float64], n: Int
) raises -> Array[Float64, 3]:
    """RMS extent along the principal axes of an ALREADY CENTRED point set.

    The scatter matrix is symmetric positive semi-definite, so its singular
    values ARE its eigenvalues and the SVD already linked serves as the
    eigensolver `math3d` does not have.
    """
    var s = List[Float64](length=9, fill=0.0)
    for k in range(n):
        for i in range(3):
            for j in range(3):
                s[i * 3 + j] += centred[k * 3 + i] * centred[k * 3 + j]
    for i in range(9):
        s[i] /= Float64(n)
    var u9 = List[Float64]()
    var s3 = List[Float64]()
    var vt9 = List[Float64]()
    svd_3x3(s, u9, s3, vt9)
    var out = Array[Float64, 3](fill=0.0)
    for i in range(3):
        # ⚠ CLAMP BEFORE THE ROOT. A PSD matrix can produce a singular value
        # of -1e-18 in float64, and `sqrt` of that is a NaN that then travels
        # into a printed report as a silent "-nan" nobody attributes to here.
        var e = s3[i] if s3[i] > 0.0 else 0.0
        out[i] = sqrt(e) * 1000.0
    return out^


def fit_rigid(
    cam_xyz: List[Float64], base_xyz: List[Float64]
) raises -> RigidFit:
    """Kabsch: the rigid `R`, `t` with `base ~= R * cam + t`, least squares.

    Both lists are FLAT, three METRES per point, same count and same order —
    flat because both producers hand back flat lists (`solve_pnp`'s `tvec` and
    the FK readout), and pairing them is the caller's job.

    ⚠ ORDER IS THE CORRESPONDENCE AND NOTHING CHECKS IT. A shuffled pair of
    lists is a valid input describing a different world; the only symptom is a
    residual in centimetres. Append to both in the same place, always.

    Raises:
        If the counts disagree, fewer than `MIN_POINTS` are given, or the
        points are collinear.
    """
    if len(cam_xyz) != len(base_xyz):
        raise (
            String("fit_rigid: ")
            + String(len(cam_xyz) // 3)
            + " camera points against "
            + String(len(base_xyz) // 3)
            + " base points"
        )
    if len(cam_xyz) % 3 != 0:
        raise String("fit_rigid: point lists must be a multiple of 3")
    var n = len(cam_xyz) // 3
    if n < MIN_POINTS:
        raise (
            String("fit_rigid: ")
            + String(n)
            + " correspondences, need at least "
            + String(MIN_POINTS)
        )

    # ── centroids, then centre both sets ────────────────────────────────────
    var cx = Array[Float64, 3](fill=0.0)
    var cy = Array[Float64, 3](fill=0.0)
    for k in range(n):
        for i in range(3):
            cx[i] += cam_xyz[k * 3 + i]
            cy[i] += base_xyz[k * 3 + i]
    for i in range(3):
        cx[i] /= Float64(n)
        cy[i] /= Float64(n)

    var xc = List[Float64](length=n * 3, fill=0.0)
    var yc = List[Float64](length=n * 3, fill=0.0)
    for k in range(n):
        for i in range(3):
            xc[k * 3 + i] = cam_xyz[k * 3 + i] - cx[i]
            yc[k * 3 + i] = base_xyz[k * 3 + i] - cy[i]

    # ── refuse a line before fitting a confident nonsense to it ─────────────
    var spread = _principal_spread_mm(yc, n)
    if spread[1] < COLLINEAR_MM:
        raise (
            String("fit_rigid: the poses are COLLINEAR (second extent ")
            + String(spread[1])
            + " mm). The rotation about that line is unconstrained and the fit"
            " would be confidently wrong — move the arm off the line."
        )

    # ── H = sum x_c y_cᵀ, then R = V diag(1, 1, d) Uᵀ ───────────────────────
    var h = List[Float64](length=9, fill=0.0)
    for k in range(n):
        for i in range(3):
            for j in range(3):
                h[i * 3 + j] += xc[k * 3 + i] * yc[k * 3 + j]
    var u9 = List[Float64]()
    var s3 = List[Float64]()
    var vt9 = List[Float64]()
    svd_3x3(h, u9, s3, vt9)

    # `vt9` is Vᵀ row-major, so V[i][k] = vt9[k*3+i]; Uᵀ[k][j] = u9[j*3+k].
    var w = Array[Float64, 3](fill=1.0)
    var r = List[Float64](length=9, fill=0.0)
    for _pass in range(2):
        for i in range(3):
            for j in range(3):
                var acc = 0.0
                for k in range(3):
                    acc += vt9[k * 3 + i] * w[k] * u9[j * 3 + k]
                r[i * 3 + j] = acc
        # ⚠⚠ THE SIGN, AND IT IS THE WHOLE POINT OF THIS LOOP. `tr(R H)` is
        # maximised over ORTHOGONAL matrices, which includes reflections — so
        # the first pass can produce `det = -1`, a mirror world that fits the
        # data BETTER than any rotation and describes a robot that does not
        # exist. Flipping the last column of V (the direction the smallest
        # singular value already says is least determined) is the standard fix
        # and the second pass rebuilds with it.
        var rot_try = Mat3d(
            r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8]
        )
        if rot_try.determinant() > 0.0:
            break
        w[2] = -1.0

    var rot = Mat3d(r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8])
    var det = rot.determinant()
    if abs(det - 1.0) > 1.0e-6:
        # Unreachable by construction; kept because "unreachable" is a claim
        # about the code above and this is a claim about the ANSWER.
        raise (
            String("fit_rigid: the fitted matrix is not a rotation, det = ")
            + String(det)
        )

    var ccam = Vec3d(cx[0], cx[1], cx[2])
    var cbase = Vec3d(cy[0], cy[1], cy[2])
    var trans = cbase - rot * ccam

    # ── residuals, in the frame and the unit that decide anything ───────────
    var sq_sum = 0.0
    var max_mm = 0.0
    var worst = 0
    for k in range(n):
        var p = rot * Vec3d(
            cam_xyz[k * 3], cam_xyz[k * 3 + 1], cam_xyz[k * 3 + 2]
        ) + trans
        var dx = p.x - base_xyz[k * 3]
        var dy = p.y - base_xyz[k * 3 + 1]
        var dz = p.z - base_xyz[k * 3 + 2]
        var d2 = dx * dx + dy * dy + dz * dz
        sq_sum += d2
        var mm = sqrt(d2) * 1000.0
        if mm > max_mm:
            max_mm = mm
            worst = k
    var rms_mm = sqrt(sq_sum / Float64(n)) * 1000.0

    return RigidFit(rot, trans, rms_mm, max_mm, worst, n, spread^)


# ── the marker's offset, solved WITH the camera ─────────────────────────────


comptime MIN_ROT_SPREAD_DEG = 10.0
"""Below this rotational spread of the gripper over the captured poses the
marker offset is not determined — see `fit_rigid_with_offset`."""


@fieldwise_init
struct OffsetFit(Copyable, Movable, Writable):
    """`fit_rigid` with the marker offset estimated rather than measured."""

    var fit: RigidFit
    """The camera -> base fit at the solved offset."""
    var offset: Vec3d
    """The marker centre in the GRIPPER body's frame, metres."""
    var rot_spread_deg: Float64
    """How much the gripper's orientation varied over the poses: the RMS
    angle about its LEAST-varied axis, degrees (see the function)."""
    var iterations: Int

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            self.fit, "  offset  ", self.offset.x, " ", self.offset.y, " ",
            self.offset.z, " m (solved, gripper frame)\n  wrist   ",
            self.rot_spread_deg, " deg rotational spread (least-varied axis), ",
            self.iterations, " iterations\n",
        )


def fit_rigid_with_offset(
    cam_xyz: List[Float64],
    grip_pos: List[Float64],
    grip_rot: List[Float64],
    max_iter: Int = 500,
    tol_m: Float64 = 1e-9,
) raises -> OffsetFit:
    """Solve the camera -> base transform AND the marker's offset on the
    gripper, from `n` poses: `cam_xyz` the marker in the camera (flat, 3 per
    pose), `grip_pos` / `grip_rot` the gripper body's FK position (flat, 3)
    and rotation (flat, 9, row-major) in the base frame.

    ⚠⚠ WHY: `--offset` is a hand measurement of the marker centre in the
    gripper body's frame — a frame with no visible origin — and a 10 mm error
    there is ~10 mm of error in the result that no number of poses averages
    away (the tool's header). But the offset is DETERMINED by the data once
    the wrist turns: the marker sits at `t_i + R_i * off` in the base frame,
    and a different `off` moves that point differently for each `R_i`.

    Block coordinate descent on the one least-squares objective
    `sum |R c_i + T - t_i - R_i off|^2`, each block exact:
      * `off` fixed: Kabsch (`fit_rigid`) gives R, T;
      * R, T fixed: `R_i` is orthogonal, so `off = mean(R_i^T (R c_i + T - t_i))`.
    Monotone, from `off = 0`.

    ⚠ REFUSED WITHOUT ROTATION. If every `R_i` is the same, `R_i off` is a
    constant that `T` absorbs and `off` is anything. The spread is the
    smallest eigenvalue of `sum (R_i - Rbar)^T (R_i - Rbar)`, per pose, as an
    angle; below `MIN_ROT_SPREAD_DEG` the call raises — turn the wrist
    (roll AND pitch) between captures. Three more unknowns also mean the
    residual reads LOWER than with a measured offset; it needs more poses to
    mean the same thing (the tool asks for 10)."""
    var n = len(cam_xyz) // 3
    if len(grip_pos) != n * 3 or len(grip_rot) != n * 9:
        raise String("fit_rigid_with_offset: pose lists do not match")
    if n < MIN_POINTS + 2:
        raise String("fit_rigid_with_offset: need at least ") + String(MIN_POINTS + 2) + " poses"

    # ── is the offset determined at all? ────────────────────────────────
    var rbar = List[Float64](length=9, fill=0.0)
    for k in range(n):
        for e in range(9):
            rbar[e] += grip_rot[k * 9 + e] / Float64(n)
    var dmat = List[Float64](length=9, fill=0.0)
    for k in range(n):
        for i in range(3):
            for j in range(3):
                var acc = 0.0
                for r in range(3):
                    acc += (grip_rot[k * 9 + r * 3 + i] - rbar[r * 3 + i]) * (
                        grip_rot[k * 9 + r * 3 + j] - rbar[r * 3 + j]
                    )
                dmat[i * 3 + j] += acc
    var u9 = List[Float64]()
    var s3 = List[Float64]()
    var vt9 = List[Float64]()
    svd_3x3(dmat, u9, s3, vt9)
    var lmin = min(s3[0], min(s3[1], s3[2]))
    # for small rotations |R_i - Rbar|^2 about an axis ~ 2 theta^2 per
    # off-axis direction; sqrt(lmin / n) is that theta in radians
    var spread_deg = sqrt(max(lmin, 0.0) / Float64(n)) * 180.0 / 3.141592653589793
    if spread_deg < MIN_ROT_SPREAD_DEG:
        raise (
            String("fit_rigid_with_offset: the gripper barely rotated (")
            + String(spread_deg) + " deg about its least-varied axis, need "
            + String(MIN_ROT_SPREAD_DEG)
            + ") — the marker offset is undetermined. Turn the wrist (roll AND"
            " pitch) between captures, or pass a measured --offset."
        )

    # ── alternate ───────────────────────────────────────────────────────
    var off = Vec3d.zero()
    var base = List[Float64](length=n * 3, fill=0.0)
    var it = 0
    while it < max_iter:
        it += 1
        for k in range(n):
            var rk = Mat3d(
                grip_rot[k * 9], grip_rot[k * 9 + 1], grip_rot[k * 9 + 2],
                grip_rot[k * 9 + 3], grip_rot[k * 9 + 4], grip_rot[k * 9 + 5],
                grip_rot[k * 9 + 6], grip_rot[k * 9 + 7], grip_rot[k * 9 + 8],
            )
            var b = Vec3d(grip_pos[k * 3], grip_pos[k * 3 + 1], grip_pos[k * 3 + 2]) + rk * off
            base[k * 3] = b.x
            base[k * 3 + 1] = b.y
            base[k * 3 + 2] = b.z
        var fit_i = fit_rigid(cam_xyz, base)
        var acc = Vec3d.zero()
        for k in range(n):
            var rk = Mat3d(
                grip_rot[k * 9], grip_rot[k * 9 + 1], grip_rot[k * 9 + 2],
                grip_rot[k * 9 + 3], grip_rot[k * 9 + 4], grip_rot[k * 9 + 5],
                grip_rot[k * 9 + 6], grip_rot[k * 9 + 7], grip_rot[k * 9 + 8],
            )
            var y = fit_i.apply(Vec3d(cam_xyz[k * 3], cam_xyz[k * 3 + 1], cam_xyz[k * 3 + 2])) - Vec3d(
                grip_pos[k * 3], grip_pos[k * 3 + 1], grip_pos[k * 3 + 2]
            )
            acc = acc + rk.transpose() * y
        var new_off = acc / Float64(n)
        var step = Float64((new_off - off).length())
        off = new_off
        if step < tol_m:
            break
    # the final fit AT the converged offset
    for k in range(n):
        var rk = Mat3d(
            grip_rot[k * 9], grip_rot[k * 9 + 1], grip_rot[k * 9 + 2],
            grip_rot[k * 9 + 3], grip_rot[k * 9 + 4], grip_rot[k * 9 + 5],
            grip_rot[k * 9 + 6], grip_rot[k * 9 + 7], grip_rot[k * 9 + 8],
        )
        var b = Vec3d(grip_pos[k * 3], grip_pos[k * 3 + 1], grip_pos[k * 3 + 2]) + rk * off
        base[k * 3] = b.x
        base[k * 3 + 1] = b.y
        base[k * 3 + 2] = b.z
    var fit = fit_rigid(cam_xyz, base)
    return OffsetFit(fit^, off, spread_deg, it)
