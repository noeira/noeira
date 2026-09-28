# +--------------------------------------------------------------------------+ #
# | A fisheye calibration that survives a 170-degree lens
# +--------------------------------------------------------------------------+ #
"""`cv::fisheye::calibrate`, wrapped so it converges on the rig's lens.

    var views = CalibViews()
    views.add(obj_xyz, img_xy)          # one per captured board pose
    var fit = calibrate_fisheye(views, 640, 480)
    fit.lens                            # a `vision.fisheye.FisheyeLens`

## ⚠⚠ WHY NOT ONE CALL

`cv::fisheye::calibrate` starts from `f = max(w, h) / pi` and zero
distortion, and its per-view initialisation (`InitExtrinsics`) first
undistorts the view's corners under THAT guess. A corner 80+ degrees off axis
in a 170-degree lens maps to a near-infinite normalised coordinate, the
view's homography degenerates, and the WHOLE fit aborts on
`fabs(norm_u1) > 0` — measured on the synthetic 170-degree lens of
`tests/vision/test_fisheye.mojo` with 40 ordinary views. Real captures put
corners at the edge on purpose (that is where the terms are determined), so
this is the normal case, not a corner case.

So, two stages:

  1. CENTRE: only each view's corners within `centre_frac` of the image's
     half-height from its centre (and only views keeping >= 8 of them), with
     k3 and k4 FIXED at 0 (then k2 too if that still aborts). The default
     guess is good there, and the fit gets `f`, the principal point and the
     low-order terms. ⚠ Freeing k3/k4 on centre corners does NOT work: they
     are unobservable there, run off, and the next iteration's per-view
     re-initialisation (RECOMPUTE_EXTRINSIC) aborts on the same assertion.
  2. ALL corners, `CALIB_USE_INTRINSIC_GUESS` from stage 1. With a real `f`
     and real terms the edge corners undistort to sane rays.

If stage 2 still aborts, the view reaching furthest off centre is dropped
and it is retried, up to a quarter of the views — and every dropped view is
REPORTED. Then views whose own reprojection rms exceeds 3x the median (and
1 px) are dropped once and the fit repeated: a blurred or mis-detected
capture otherwise pulls the whole lens toward itself.
"""

from std.math import sqrt

from .fisheye import FisheyeLens, Pinhole, UndistortMap
from .opencv import (
    fisheye_calibrate, fisheye_project, FISHEYE_CALIB_RECOMPUTE_EXTRINSIC,
    FISHEYE_CALIB_FIX_SKEW, FISHEYE_CALIB_USE_INTRINSIC_GUESS,
    FISHEYE_CALIB_FIX_K2, FISHEYE_CALIB_FIX_K3, FISHEYE_CALIB_FIX_K4,
)


struct CalibViews(Copyable, Movable):
    """Board poses: per view, `n` object points (metres, board frame) and the
    `n` pixels they were detected at."""

    var obj: List[List[Float64]]
    var img: List[List[Float64]]

    def __init__(out self):
        self.obj = List[List[Float64]]()
        self.img = List[List[Float64]]()

    def add(mut self, var obj_xyz: List[Float64], var img_xy: List[Float64]) raises:
        if len(obj_xyz) // 3 != len(img_xy) // 2:
            raise Error("CalibViews.add: object and image point counts differ")
        self.obj.append(obj_xyz^)
        self.img.append(img_xy^)

    def count(self) -> Int:
        return len(self.obj)


struct FisheyeFit(Movable):
    var lens: FisheyeLens
    var rms: Float64
    var used: List[Int]
    """Indices (into the input views) the final fit used."""
    var dropped: List[Int]
    var why: List[String]
    """One reason per dropped view."""
    var view_rms: List[Float64]
    """Reprojection rms per USED view, parallel to `used`."""

    def __init__(out self, lens: FisheyeLens):
        self.lens = lens
        self.rms = 0.0
        self.used = List[Int]()
        self.dropped = List[Int]()
        self.why = List[String]()
        self.view_rms = List[Float64]()


def _radius(x: Float64, y: Float64, w: Int, h: Int) -> Float64:
    var dx = x - Float64(w) / 2.0
    var dy = y - Float64(h) / 2.0
    return sqrt(dx * dx + dy * dy)


def _fit(
    ref views: CalibViews, ref idx: List[Int], w: Int, h: Int,
    centre_r: Float64, guess: List[Float64], guess_d: List[Float64],
    mut k: List[Float64], mut d: List[Float64],
    mut rv: List[Float64], mut tv: List[Float64], extra_flags: Int = 0,
) raises -> Float64:
    """One `fisheye_calibrate` over the views `idx`, keeping only corners
    within `centre_r` pixels of the centre when `centre_r > 0`."""
    var obj = List[Float64]()
    var img = List[Float64]()
    var counts = List[Int32]()
    for v in idx:
        var n = 0
        for p in range(len(views.img[v]) // 2):
            var x = views.img[v][p * 2]
            var y = views.img[v][p * 2 + 1]
            if centre_r > 0.0 and _radius(x, y, w, h) > centre_r:
                continue
            obj.append(views.obj[v][p * 3])
            obj.append(views.obj[v][p * 3 + 1])
            obj.append(views.obj[v][p * 3 + 2])
            img.append(x)
            img.append(y)
            n += 1
        counts.append(Int32(n))
    var flags = FISHEYE_CALIB_RECOMPUTE_EXTRINSIC | FISHEYE_CALIB_FIX_SKEW | extra_flags
    k = guess.copy()
    d = guess_d.copy()
    if len(guess) == 9:
        flags |= FISHEYE_CALIB_USE_INTRINSIC_GUESS
    else:
        k = List[Float64](length=9, fill=0.0)
        d = List[Float64](length=4, fill=0.0)
    return fisheye_calibrate(obj, img, counts, w, h, k, d, rv, tv, flags)


def _max_radius(ref views: CalibViews, v: Int, w: Int, h: Int) -> Float64:
    var m = 0.0
    for p in range(len(views.img[v]) // 2):
        m = max(m, _radius(views.img[v][p * 2], views.img[v][p * 2 + 1], w, h))
    return m


def calibrate_fisheye(
    ref views: CalibViews, w: Int, h: Int, centre_frac: Float64 = 0.55,
    min_centre_points: Int = 8,
) raises -> FisheyeFit:
    """See the module header. Raises when fewer than 6 views survive."""
    var centre_r = centre_frac * Float64(min(w, h)) / 2.0
    var all_idx = List[Int]()
    for v in range(views.count()):
        all_idx.append(v)

    # ── stage 1: the centre ─────────────────────────────────────────────
    var c_idx = List[Int]()
    for v in all_idx:
        var n = 0
        for p in range(len(views.img[v]) // 2):
            if _radius(views.img[v][p * 2], views.img[v][p * 2 + 1], w, h) <= centre_r:
                n += 1
        if n >= min_centre_points:
            c_idx.append(v)
    if len(c_idx) < 4:
        raise Error(
            "calibrate_fisheye: only " + String(len(c_idx)) + " views put "
            + String(min_centre_points) + "+ corners near the image centre —"
            " capture some with the board in the MIDDLE of the frame too"
        )
    var k = List[Float64]()
    var d = List[Float64]()
    var rv = List[Float64]()
    var tv = List[Float64]()
    try:
        _ = _fit(
            views, c_idx, w, h, centre_r, List[Float64](), List[Float64](),
            k, d, rv, tv, FISHEYE_CALIB_FIX_K3 | FISHEYE_CALIB_FIX_K4,
        )
    except:
        _ = _fit(
            views, c_idx, w, h, centre_r, List[Float64](), List[Float64](),
            k, d, rv, tv,
            FISHEYE_CALIB_FIX_K2 | FISHEYE_CALIB_FIX_K3 | FISHEYE_CALIB_FIX_K4,
        )
    var k1 = k.copy()
    var d1 = d.copy()

    # ── stage 2: everything, seeded; drop the furthest view on abort ────
    var idx = all_idx.copy()
    var dropped = List[Int]()
    var why = List[String]()
    var rms = -1.0
    var tries = 0
    while rms < 0.0:
        try:
            rms = _fit(views, idx, w, h, 0.0, k1, d1, k, d, rv, tv)
        except e:
            tries += 1
            if tries > max(1, len(all_idx) // 4) or len(idx) <= 6:
                raise Error(
                    "calibrate_fisheye: the full fit aborts even after dropping "
                    + String(len(dropped)) + " views: " + String(e)
                )
            var worst = 0
            for j in range(len(idx)):
                if _max_radius(views, idx[j], w, h) > _max_radius(views, idx[worst], w, h):
                    worst = j
            dropped.append(idx[worst])
            why.append(String("aborted the full fit (furthest off centre)"))
            _ = idx.pop(worst)

    # ── per-view rms, one round of outlier rejection ────────────────────
    for round in range(2):
        var vr = List[Float64]()
        for j in range(len(idx)):
            var v = idx[j]
            var proj = List[Float64]()
            fisheye_project(
                views.obj[v], [rv[j * 3], rv[j * 3 + 1], rv[j * 3 + 2]],
                [tv[j * 3], tv[j * 3 + 1], tv[j * 3 + 2]], k, d, proj,
            )
            var se = 0.0
            var n = len(views.img[v]) // 2
            for p in range(n):
                var dx = proj[p * 2] - views.img[v][p * 2]
                var dy = proj[p * 2 + 1] - views.img[v][p * 2 + 1]
                se += dx * dx + dy * dy
            vr.append(sqrt(se / Float64(n)))
        var sorted_r = vr.copy()
        sort(sorted_r)
        var med = sorted_r[len(sorted_r) // 2]
        var keep = List[Int]()
        var n_out = 0
        for j in range(len(idx)):
            if round == 0 and vr[j] > 3.0 * med and vr[j] > 1.0:
                dropped.append(idx[j])
                why.append("reprojection rms " + String(vr[j]) + " px > 3x median " + String(med))
                n_out += 1
            else:
                keep.append(idx[j])
        if round == 1 or n_out == 0:
            var fit = FisheyeFit(
                FisheyeLens(k[0], k[4], k[2], k[5], d[0], d[1], d[2], d[3], w, h)
            )
            fit.rms = rms
            fit.used = idx.copy()
            fit.dropped = dropped.copy()
            fit.why = why.copy()
            fit.view_rms = vr.copy()
            return fit^
        idx = keep^
        if len(idx) < 6:
            raise Error("calibrate_fisheye: fewer than 6 views survive the outlier pass")
        var kk = k.copy()
        var dd = d.copy()
        rms = _fit(views, idx, w, h, 0.0, kk, dd, k, d, rv, tv)
    raise Error("calibrate_fisheye: unreachable")


def undistort_spread(
    ref views: CalibViews, ref fit: FisheyeFit, pin: Pinhole, n_boot: Int = 16,
    seed: UInt64 = 0x5EED,
) raises -> Float64:
    """How far the undistortion map MOVES when the calibration is redone on
    the used views resampled with replacement — the rms, over the pinhole
    field, of each bootstrap map against `fit`'s, averaged over `n_boot`.

    ⚠ THIS IS THE CALIBRATION'S ERROR BAR, AND `rms` IS NOT. `rms` is the
    residual of the model against the corners it was fitted to; a small board
    seen from few directions fits beautifully with a focal length and four
    terms that trade off against each other. On the synthetic 170-degree lens
    (`tests/vision/test_fisheye.mojo`) 40 views gave rms 0.27 px and a map
    0.67 px rms off the truth; this number is what shows that before a store
    is imported through it."""
    var base = UndistortMap(fit.lens, pin)
    var s = seed
    var acc = 0.0
    for _ in range(n_boot):
        var idx = List[Int]()
        for _ in range(len(fit.used)):
            s += UInt64(0x9E3779B97F4A7C15)
            var z = s
            z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
            z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
            z = z ^ (z >> 31)
            idx.append(fit.used[Int(z % UInt64(len(fit.used)))])
        var k = List[Float64]()
        var d = List[Float64]()
        var rv = List[Float64]()
        var tv = List[Float64]()
        _ = _fit(
            views, idx, fit.lens.width, fit.lens.height, 0.0,
            fit.lens.k_matrix(), fit.lens.d_vector(), k, d, rv, tv,
        )
        var um = UndistortMap(
            FisheyeLens(k[0], k[4], k[2], k[5], d[0], d[1], d[2], d[3],
                        fit.lens.width, fit.lens.height),
            pin,
        )
        var se = 0.0
        for i in range(len(um.map_x)):
            var ex = Float64(um.map_x[i]) - Float64(base.map_x[i])
            var ey = Float64(um.map_y[i]) - Float64(base.map_y[i])
            se += ex * ex + ey * ey
        acc += sqrt(se / Float64(len(um.map_x)))
    return acc / Float64(n_boot)


@fieldwise_init
struct SpreadParts(Copyable, ImplicitlyCopyable, Movable, Writable):
    """`undistort_spread`, decomposed into what each part MEANS for a camera.

    Measured on the rig's overhead camera (60 views, rigid board): a total
    spread of 1.76 px was ~1 px of global SHIFT (the principal point) plus a
    radial SCALE (the focal length, +-0.6%), with the lens's shape itself
    well determined. A shift of the undistorted image is a small camera
    ROTATION and a scale is a FIELD-OF-VIEW change, and the render-time
    randomization trains the student under both (`randomize.mojo` full:
    +-2 deg rotation, +-3 deg fovy), while the extrinsics calibration
    absorbs the shift anyway. So each part is judged against those ranges,
    not the total against one pixel threshold."""

    var total_px: Float64
    """`undistort_spread`: rms map movement over the pinhole field."""
    var shift_deg: Float64
    """rms of the per-resample mean shift, as a camera rotation."""
    var scale_pct: Float64
    """rms focal-length change, percent — a field-of-view change."""
    var shape_px: Float64
    """What is left after removing each resample's shift and scale."""

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "total ", self.total_px, " px = shift ", self.shift_deg,
            " deg + focal ", self.scale_pct, " % + shape ", self.shape_px,
            " px",
        )


def undistort_spread_parts(
    ref views: CalibViews, ref fit: FisheyeFit, pin: Pinhole, n_boot: Int = 16,
    seed: UInt64 = 0x5EED,
) raises -> SpreadParts:
    """`undistort_spread`, with each resample's map change split into a mean
    shift, a radial scale about the pinhole's centre (least squares), and the
    remaining shape. See `SpreadParts`."""
    var base = UndistortMap(fit.lens, pin)
    var n = len(base.map_x)
    var s = seed
    var tot = 0.0
    var sh = 0.0
    var sc = 0.0
    var shp = 0.0
    for _ in range(n_boot):
        var idx = List[Int]()
        for _ in range(len(fit.used)):
            s += UInt64(0x9E3779B97F4A7C15)
            var z = s
            z = (z ^ (z >> 30)) * UInt64(0xBF58476D1CE4E5B9)
            z = (z ^ (z >> 27)) * UInt64(0x94D049BB133111EB)
            z = z ^ (z >> 31)
            idx.append(fit.used[Int(z % UInt64(len(fit.used)))])
        var k = List[Float64]()
        var d = List[Float64]()
        var rv = List[Float64]()
        var tv = List[Float64]()
        _ = _fit(
            views, idx, fit.lens.width, fit.lens.height, 0.0,
            fit.lens.k_matrix(), fit.lens.d_vector(), k, d, rv, tv,
        )
        var um = UndistortMap(
            FisheyeLens(k[0], k[4], k[2], k[5], d[0], d[1], d[2], d[3],
                        fit.lens.width, fit.lens.height),
            pin,
        )
        var mx = 0.0
        var my = 0.0
        var se = 0.0
        for i in range(n):
            var ex = Float64(um.map_x[i]) - Float64(base.map_x[i])
            var ey = Float64(um.map_y[i]) - Float64(base.map_y[i])
            mx += ex
            my += ey
            se += ex * ex + ey * ey
        mx /= Float64(n)
        my /= Float64(n)
        tot += se / Float64(n)
        sh += mx * mx + my * my
        # radial scale about the base map's own centre: e ~ a * (p - centre)
        var cx = 0.0
        var cy = 0.0
        for i in range(n):
            cx += Float64(base.map_x[i])
            cy += Float64(base.map_y[i])
        cx /= Float64(n)
        cy /= Float64(n)
        var num = 0.0
        var den = 0.0
        for i in range(n):
            var rx = Float64(base.map_x[i]) - cx
            var ry = Float64(base.map_y[i]) - cy
            num += (Float64(um.map_x[i]) - Float64(base.map_x[i]) - mx) * rx
            num += (Float64(um.map_y[i]) - Float64(base.map_y[i]) - my) * ry
            den += rx * rx + ry * ry
        var a = num / den if den > 0.0 else 0.0
        sc += (k[0] / fit.lens.fx - 1.0) ** 2
        var rs = 0.0
        for i in range(n):
            var rx = Float64(base.map_x[i]) - cx
            var ry = Float64(base.map_y[i]) - cy
            var ex = Float64(um.map_x[i]) - Float64(base.map_x[i]) - mx - a * rx
            var ey = Float64(um.map_y[i]) - Float64(base.map_y[i]) - my - a * ry
            rs += ex * ex + ey * ey
        shp += rs / Float64(n)
    var nb = Float64(n_boot)
    # a shift of `px` pixels in the SOURCE fisheye near its centre is a
    # rotation of px / f radians
    var shift_rad = sqrt(sh / nb) / fit.lens.fx
    return SpreadParts(
        sqrt(tot / nb), shift_rad * 180.0 / 3.141592653589793,
        100.0 * sqrt(sc / nb), sqrt(shp / nb),
    )
