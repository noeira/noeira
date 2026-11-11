"""Camera frame -> the exact tensor ACT was trained on.

⚠⚠ **THIS FILE EXISTS BECAUSE NO OpenCV RESIZE MATCHES THE TRAINING PIPELINE.**
`tools/act/lerobot_v3_to_store.py` downscales every 640x480 frame to 320x240
with **PIL BILINEAR**, and PIL scales its filter support with the reduction
factor — so at 2x it is a 4-tap triangle, not the 2-tap of `INTER_LINEAR` nor
the 2x2 box of `INTER_AREA`. Measured on five real frames from the 50-demo
dataset, against PIL:

    INTER_AREA      mean |d| 1.195   max 33   12.9% of pixels off by >2
    INTER_LINEAR    mean |d| 1.217   max 34   13.1%
    INTER_CUBIC     mean |d| 2.206   max 61   19.5%
    INTER_LANCZOS4  mean |d| 2.376   max 64   20.6%

~1.2/255 in the mean is small. It is also an AVOIDABLE train/deploy gap in a
project that keeps being bitten by silent representation differences, and the
alternative — re-converting the dataset with a different filter — costs a
retrain. So PIL's algorithm is reproduced here bit for bit instead, and the
gate asserts EXACT equality rather than a tolerance.

WHAT "PIL'S ALGORITHM" MEANS, PRECISELY (libImaging/Resample.c):

  * `support = 1.0 * max(scale, 1)`, `ksize = ceil(support) * 2 + 1` — the
    window GROWS when downscaling, which is the whole difference.
  * coefficients in double, normalised to sum 1, then converted to INT32 fixed
    point with 22 fractional bits.
  * the accumulator starts at `1 << 21` (round-to-nearest), sums
    `uint8 * int32`, then `>> 22` and clamps to 0..255.
  * ⚠ TWO PASSES WITH AN **8-BIT INTERMEDIATE**: horizontal into a temporary
    image, then vertical over that. Keeping the intermediate in higher
    precision is more accurate and produces DIFFERENT numbers, which is the
    wrong goal here — matching is.
"""

from std.math import ceil
from max.algorithm import parallelize

comptime _RESIZE_PAR_MIN_TAPS: Int = 250_000
"""Below this many multiply-accumulates the resize stays on one core.

`parallelize`'s fixed cost is ~200 us (measured in `nn/primitives/conv2d.mojo`,
where a blanket call made five ResNet18 shapes slower and one 20x slower), and
this function is called on thumbnails as well as camera frames."""

comptime _PRECISION_BITS: Int = 22
"""`32 - 8 - 2`, as PIL defines it."""


@always_inline
def _triangle(x: Float64) -> Float64:
    var a = -x if x < 0.0 else x
    if a < 1.0:
        return 1.0 - a
    return 0.0


struct _Coeffs(Movable):
    """One axis's resampling plan: window bounds and fixed-point weights."""

    var ksize: Int
    var bounds: List[Int]
    """`2 * out_size`: (xmin, count) per output pixel."""
    var kk: List[Int64]
    """`out_size * ksize` fixed-point weights, `precision` fractional bits."""
    var precision: Int
    """22 (PIL), or torch's int16 rule (`torch_int16`)."""

    def __init__(out self, in_size: Int, out_size: Int, torch_int16: Bool = False):
        var scale = Float64(in_size) / Float64(out_size)
        var filterscale = scale
        if filterscale < 1.0:
            filterscale = 1.0
        var support = filterscale  # bilinear support is 1.0
        self.ksize = Int(ceil(support)) * 2 + 1
        self.bounds = List[Int](unsafe_uninit_length=2 * out_size)
        self.kk = List[Int64](unsafe_uninit_length=out_size * self.ksize)
        self.precision = _PRECISION_BITS
        var all_w = List[Float64](length=out_size * self.ksize, fill=0.0)

        var w = List[Float64](unsafe_uninit_length=self.ksize)
        for xx in range(out_size):
            var center = (Float64(xx) + 0.5) * scale
            var ss = 1.0 / filterscale
            # ⚠ C TRUNCATION TOWARD ZERO, then clamp — the order matters only
            # for the negative case, which the clamp absorbs.
            var xmin = Int(center - support + 0.5)
            if xmin < 0:
                xmin = 0
            var xmax = Int(center + support + 0.5)
            if xmax > in_size:
                xmax = in_size
            xmax -= xmin

            var ww = 0.0
            for x in range(xmax):
                var v = _triangle((Float64(x + xmin) - center + 0.5) * ss)
                w[x] = v
                ww += v
            for x in range(xmax):
                if ww != 0.0:
                    w[x] = w[x] / ww
            for x in range(xmax, self.ksize):
                w[x] = 0.0

            for x in range(self.ksize):
                all_w[xx * self.ksize + x] = w[x]
            self.bounds[xx * 2] = xmin
            self.bounds[xx * 2 + 1] = xmax
        if torch_int16:
            # torch's uint8 antialias kernel (`_compute_index_ranges_int16_weights`):
            # the most fractional bits that keep the axis's largest weight below 2^15
            var w_max = 0.0
            for v in all_w:
                w_max = max(w_max, v)
            var prec = 0
            while prec < 15:
                if Int(0.5 + w_max * Float64(1 << (prec + 1))) >= (1 << 15):
                    break
                prec += 1
            self.precision = prec
        # Fixed point, PIL's `normalize_coeffs_8bpc` (torch's rounding is the same)
        for i in range(out_size * self.ksize):
            var f = all_w[i] * Float64(1 << self.precision)
            if all_w[i] < 0.0:
                self.kk[i] = Int64(-0.5 + f)
            else:
                self.kk[i] = Int64(0.5 + f)

    def __init__(out self, *, deinit move: Self):
        self.ksize = move.ksize
        self.bounds = move.bounds^
        self.kk = move.kk^
        self.precision = move.precision


@always_inline
def _clip8(v: Int64, precision: Int = _PRECISION_BITS) -> UInt8:
    var s = v >> Int64(precision)
    if s < 0:
        return 0
    if s > 255:
        return 255
    return UInt8(s)


def torch_bilinear_aa_u8(
    src: List[UInt8], src_w: Int, src_h: Int, channels: Int, mut dst: List[UInt8], dst_w: Int, dst_h: Int,
) raises:
    """torch's uint8 BILINEAR with antialias (`F.interpolate(..., antialias=True)`
    on a uint8 tensor: torchvision's `resize`, so transformers' fast image
    processors — SAM 3's): PIL's coefficients and two 8-bit passes, but int16
    weights whose fractional bits are chosen per axis (the most that keep the
    largest weight below 2^15) where PIL uses 22. Measured on SAM 3's 640x480 ->
    1008x1008: PIL's arithmetic was 19 727 values off (by up to 2); this, 0."""
    _resample_u8(src, src_w, src_h, channels, dst, dst_w, dst_h, True)


def pil_bilinear_u8(
    src: List[UInt8],
    src_w: Int,
    src_h: Int,
    channels: Int,
    mut dst: List[UInt8],
    dst_w: Int,
    dst_h: Int,
) raises:
    """Resize interleaved uint8 HWC, bit-identical to `PIL.Image.BILINEAR`.

    ⚠ NOT A GENERIC RESIZE. It reproduces one specific library's arithmetic
    because a dataset was built with it; see the module docstring. If the
    training pipeline ever changes filter, this must change with it or the
    gate will say so.
    """
    _resample_u8(src, src_w, src_h, channels, dst, dst_w, dst_h, False)


def _resample_u8(
    src: List[UInt8], src_w: Int, src_h: Int, channels: Int, mut dst: List[UInt8], dst_w: Int, dst_h: Int,
    torch_int16: Bool,
) raises:
    if src_w <= 0 or src_h <= 0 or dst_w <= 0 or dst_h <= 0:
        raise String("pil_bilinear_u8: degenerate size")
    if len(src) < src_w * src_h * channels:
        raise String("pil_bilinear_u8: source holds ") + String(
            len(src)
        ) + " bytes, need " + String(src_w * src_h * channels)
    if len(dst) < dst_w * dst_h * channels:
        dst.resize(dst_w * dst_h * channels, 0)

    var hc = _Coeffs(src_w, dst_w, torch_int16)
    var vc = _Coeffs(src_h, dst_h, torch_int16)

    # ⚠ 8-BIT INTERMEDIATE, matching PIL. See the module docstring.
    var tmp = List[UInt8](unsafe_uninit_length=dst_w * src_h * channels)

    var hp = hc.precision
    var vp = vc.precision
    var hround = Int64(1) << Int64(hp - 1)
    var vround = Int64(1) << Int64(vp - 1)

    # ⚠⚠ RAW POINTERS AND TWO ROW-PARALLEL PASSES. This was 8.7 ms of a 69 ms
    # ACT deployment query — the largest item outside the network, and the only
    # one nothing had ever looked at. It is ~2.8M scalar fixed-point
    # multiply-accumulates per camera: 480x320x3x4 horizontally, 240x320x3x4
    # vertically. Every output ROW is independent within a pass, so each `yy`
    # owns a disjoint span and there is nothing to synchronise.
    #
    # ⚠ THE TWO PASSES CANNOT BE FUSED INTO ONE `parallelize`. The vertical
    # pass reads rows of `tmp` that the horizontal pass writes — up to `ksize`
    # of them per output row — so the barrier between them is the algorithm,
    # not a missed optimisation. Two launches, and their cost is why the gate
    # below exists.
    var sp = src.unsafe_ptr()
    var tp = tmp.unsafe_ptr()
    var dp = dst.unsafe_ptr()
    var hb = hc.bounds.unsafe_ptr()
    var hk = hc.kk.unsafe_ptr()
    var vb = vc.bounds.unsafe_ptr()
    var vk = vc.kk.unsafe_ptr()
    var hks = hc.ksize
    var vks = vc.ksize

    # horizontal: (src_h, src_w) -> (src_h, dst_w)
    def _hrow(yy: Int) {imm}:
        var srow = yy * src_w * channels
        var trow = yy * dst_w * channels
        for xx in range(dst_w):
            var xmin = hb[unsafe_offset = xx * 2]
            var xmax = hb[unsafe_offset = xx * 2 + 1]
            var kbase = xx * hks
            for c in range(channels):
                var acc = hround
                for x in range(xmax):
                    acc += (
                        Int64(sp[unsafe_offset = srow + (xmin + x) * channels + c])
                        * hk[unsafe_offset = kbase + x]
                    )
                tp[unsafe_offset = trow + xx * channels + c] = _clip8(acc, hp)

    # vertical: (src_h, dst_w) -> (dst_h, dst_w)
    def _vrow(yy: Int) {imm}:
        var ymin = vb[unsafe_offset = yy * 2]
        var ymax = vb[unsafe_offset = yy * 2 + 1]
        var kbase = yy * vks
        var drow = yy * dst_w * channels
        for xx in range(dst_w):
            for c in range(channels):
                var acc = vround
                for y in range(ymax):
                    acc += (
                        Int64(
                            tp[
                                unsafe_offset = (ymin + y) * dst_w * channels
                                + xx * channels
                                + c
                            ]
                        )
                        * vk[unsafe_offset = kbase + y]
                    )
                dp[unsafe_offset = drow + xx * channels + c] = _clip8(acc, vp)

    # ⚠ A FLOOR, NOT A TUNED CROSSOVER — same shape of rule as the conv
    # kernels, and for the same measured reason: `parallelize` costs a fixed
    # ~200 us here, so a small thumbnail resize must not touch the thread pool.
    # The taps are counted, not the pixels, because the tap count is what the
    # inner loop actually executes. A 640x480 -> 320x240 camera frame is 2.3M
    # taps, clearing this by 10x; a 64x64 icon is 50k and stays serial.
    var taps = src_h * dst_w * channels * hks + dst_h * dst_w * channels * vks
    if taps >= _RESIZE_PAR_MIN_TAPS:
        parallelize(_hrow, src_h)
        parallelize(_vrow, dst_h)
    else:
        for yy in range(src_h):
            _hrow(yy)
        for yy in range(dst_h):
            _vrow(yy)


def camera_frame_to_chw_rgb(
    bgr: List[UInt8],
    src_w: Int,
    src_h: Int,
    mut chw: List[UInt8],
    dst_w: Int,
    dst_h: Int,
) raises:
    """One camera frame to exactly what the ACT store holds: CHW, RGB, uint8.

    The converter's pipeline is decode(RGB) -> PIL BILINEAR -> HWC-to-CHW, and
    this is the same three steps with the capture's BGR swapped on the way out.

    ⚠ SWAPPING AFTER THE RESIZE IS EXACT, NOT AN APPROXIMATION: the filter is
    applied per channel independently, so channel order commutes with it. The
    swap is done during the transpose because that pass already touches every
    byte once.

    ⚠⚠ THE OUTPUT IS uint8, DELIBERATELY. `/255` and the ImageNet normalisation
    belong to `deep_agents/act/data.mojo`, which already does them for training;
    doing them here as well would be a second implementation of a step that has
    to agree exactly. This function's contract is "produce the bytes the store
    would have held", and that is what the gate checks.
    """
    var n = dst_w * dst_h
    if len(chw) < n * 3:
        chw.resize(n * 3, 0)
    var hwc = List[UInt8]()
    pil_bilinear_u8(bgr, src_w, src_h, 3, hwc, dst_w, dst_h)
    for i in range(n):
        chw[i] = hwc[i * 3 + 2]  # R plane <- B slot of a BGR frame
        chw[n + i] = hwc[i * 3 + 1]
        chw[2 * n + i] = hwc[i * 3]
