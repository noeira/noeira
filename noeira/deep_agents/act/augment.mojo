# +--------------------------------------------------------------------------+ #
# | ACT image augmentation — the L1 level of domain randomization
# +--------------------------------------------------------------------------+ #
"""Per-sample, per-camera photometric + geometric jitter of the uint8 frames.

`docs/DOMAIN_RANDOMIZATION_PLAN.md` §3.4 / Phase 1. The cheapest DR level:
no renderer involved, applied at TRAINING time to whatever store is loaded —
the sim-rendered one, or the real LeRobot import (where it is plain
regularisation against exposure and white-balance drift).

## The pipeline, per output pixel of one camera slot

    v  = src[clamp(y - dy), clamp(x - dx)] / 255        shift, edge clamp
    v *= gain[ch]                                       white balance
    v  = (v - 0.5) * contrast + 0.5 + brightness
    v  = clamp(v, 0, 1) ** gamma
    v += noise_sigma * sqrt(3) * u,  u ~ U[-1, 1)       sensor noise
    v  = cutout_fill        if (y, x) is in the cutout rectangle
    out_u8 = round(clamp(v, 0, 1) * 255)

The result is **uint8**, and it then goes through the unchanged
`/255 -> ImageNet` rule (`inference.normalize_camera_chw` on the host, the
same arithmetic in `_act_gather_images_aug_kernel`). That is the point of
re-quantising: the normalisation stays the ONE rule the real-robot deploy path
shares, and augmentation is strictly "a different uint8 frame". A real camera
produces uint8 too, so the quantisation is not a loss.

Contrast pivots on 0.5, not on the image mean (torchvision's `ColorJitter`
uses the grey mean): the gather is one thread per element and a per-image
reduction would be a second pass. Brightness + contrast about a fixed pivot
spans the same affine family; only the parametrisation differs.

Not done here (plan §2.4 lists them): blur and JPEG-like softening. Both need
neighbourhood reads; add them when a gate says the gap is there.

## Parameters are DRAWN once per (slot, camera), then APPLIED per pixel

`AUG_WORDS` floats per (b, cam), written by `_act_aug_draw_kernel` from
Philox and read by the gather. Splitting draw from apply is what makes the
host twin testable: `augment_camera_u8` takes the SAME parameter record, so a
gate can hand both legs identical parameters and compare bytes. The draw
itself is `aug_params_from_uniforms` — one function, called by the device
kernel and by the host sampler, so the RANGES are also written once.

Per-pixel noise is a counter hash of `(noise_key, pixel)`, not Philox: it is
the same integer function on both legs, so it is bit-reproducible, and it
costs one hash per element instead of a Philox round.
"""

from std.math import exp, log

from noeira.nn.constants import DT


# ── the parameter record ────────────────────────────────────────────────

comptime AUG_WORDS = 16
comptime AUG_UNIFORMS = 16
"""Four Philox `step_uniform` calls of 4 lanes each."""

comptime W_BRIGHT = 0
comptime W_CONTRAST = 1
comptime W_GAMMA = 2
comptime W_GAIN_R = 3
comptime W_GAIN_G = 4
comptime W_GAIN_B = 5
comptime W_NOISE = 6
comptime W_DX = 7
comptime W_DY = 8
comptime W_CUT_X0 = 9
comptime W_CUT_Y0 = 10
comptime W_CUT_X1 = 11
comptime W_CUT_Y1 = 12
comptime W_CUT_FILL = 13
comptime W_NOISE_KEY = 14
"""An integer < 2^24 stored as a float — exact in float32."""
comptime W_ENABLED = 15
"""1.0 = apply the record, 0.0 = identity (the un-augmented bytes)."""

comptime AUG_SALT: UInt64 = 0xA06D_1A7E_5EED_0001
"""XORed into the dataset seed so the augmentation streams never coincide with
the row-draw streams `_act_draw_kernel` keys on the same seed."""


# ── the config ──────────────────────────────────────────────────────────


@fieldwise_init
struct ImageAugConfig(Copyable, ImplicitlyCopyable, Movable, Writable):
    """Ranges of every knob. Each is drawn uniformly in `[-r, +r]` (or the
    stated interval) per sample and per camera.

    ⚠ `enabled = False` is not "all ranges zero": it selects the ORIGINAL
    gather kernel, so the un-augmented batch is bit-identical to what the tree
    produced before augmentation existed (gate G1a)."""

    var enabled: Bool
    var brightness: Float32
    """Additive, in [0,1] units: `v += U(-b, b)`."""
    var contrast: Float32
    """Multiplicative: `c ~ U(1-r, 1+r)` about 0.5."""
    var gamma: Float32
    """`g = exp(U(-r, r))` — symmetric in log so darkening and brightening are
    equally likely."""
    var gain: Float32
    """Per-channel multiplicative: `U(1-r, 1+r)` independently on R, G, B."""
    var noise_sigma: Float32
    """The noise std is drawn in `[0, noise_sigma]` per sample."""
    var max_shift: Int
    """Integer pixel shift in `[-s, s]` on each axis, edge-clamped."""
    var cutout_prob: Float32
    var cutout_max_frac: Float32
    """Cutout side, as a fraction of the image side, drawn in `[0.1, f] * side`."""

    @staticmethod
    def off() -> Self:
        return Self(False, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0)

    @staticmethod
    def light() -> Self:
        """Photometric only — no geometry, no occlusion."""
        return Self(True, 0.08, 0.15, 0.15, 0.05, 0.01, 0, 0.0, 0.0)

    @staticmethod
    def default() -> Self:
        """The plan's §2.4 sensor set at moderate strength."""
        return Self(True, 0.12, 0.25, 0.25, 0.08, 0.02, 8, 0.25, 0.25)

    @staticmethod
    def parse(name: String) raises -> Self:
        """`off` | `light` | `default` — the `ACT_AUGMENT` values."""
        if name == "" or name == "off":
            return Self.off()
        if name == "light":
            return Self.light()
        if name == "default":
            return Self.default()
        raise Error(
            "ImageAugConfig: unknown preset '" + name
            + "' (expected off | light | default)"
        )

    def write_to(self, mut writer: Some[Writer]):
        if not self.enabled:
            writer.write("off")
            return
        writer.write(
            "bright ±", self.brightness, ", contrast ±", self.contrast,
            ", gamma e^±", self.gamma, ", gain ±", self.gain,
            ", noise σ<=", self.noise_sigma, ", shift ±", self.max_shift,
            "px, cutout p=", self.cutout_prob, " side<=",
            self.cutout_max_frac,
        )


# ── draw: uniforms -> parameter record (shared by host and device) ──────


@always_inline
def _sym(u: Float32, r: Float32) -> Float32:
    """`U(0,1)` -> `U(-r, r)`."""
    return (Float32(2.0) * u - Float32(1.0)) * r


@always_inline
def _int_sym(u: Float32, s: Int) -> Float32:
    """`U(0,1)` -> an integer in `[-s, s]`, as a float."""
    var k = Int(u * Float32(2 * s + 1))
    if k > 2 * s:
        k = 2 * s
    return Float32(k - s)


@always_inline
def aug_param(
    cfg: ImageAugConfig, u: SIMD[DType.float32, AUG_UNIFORMS], H: Int, W: Int
) -> SIMD[DType.float32, AUG_WORDS]:
    """The record for one (sample, camera) from 16 uniforms in [0, 1).

    ⚠ The ONE place the ranges become numbers. The device draw kernel and
    the host sampler both call it; a second copy is how the two legs would
    come to disagree on what "default" means."""
    var p = SIMD[DType.float32, AUG_WORDS](0.0)
    if not cfg.enabled:
        p[W_CONTRAST] = 1.0
        p[W_GAMMA] = 1.0
        p[W_GAIN_R] = 1.0
        p[W_GAIN_G] = 1.0
        p[W_GAIN_B] = 1.0
        p[W_CUT_X0] = -1.0
        p[W_CUT_X1] = -1.0
        return p
    p[W_BRIGHT] = _sym(u[0], cfg.brightness)
    p[W_CONTRAST] = Float32(1.0) + _sym(u[1], cfg.contrast)
    p[W_GAMMA] = exp(_sym(u[2], cfg.gamma))
    p[W_GAIN_R] = Float32(1.0) + _sym(u[3], cfg.gain)
    p[W_GAIN_G] = Float32(1.0) + _sym(u[4], cfg.gain)
    p[W_GAIN_B] = Float32(1.0) + _sym(u[5], cfg.gain)
    p[W_NOISE] = u[6] * cfg.noise_sigma
    p[W_DX] = _int_sym(u[7], cfg.max_shift)
    p[W_DY] = _int_sym(u[8], cfg.max_shift)
    # Cutout: an empty rectangle (x0 = x1 = -1) unless the gate fires.
    p[W_CUT_X0] = -1.0
    p[W_CUT_X1] = -1.0
    p[W_CUT_Y0] = -1.0
    p[W_CUT_Y1] = -1.0
    if u[9] < cfg.cutout_prob and cfg.cutout_max_frac > 0.0:
        var lo = Float32(0.1)
        var hi = cfg.cutout_max_frac if cfg.cutout_max_frac > lo else lo
        var cw = Int((lo + u[12] * (hi - lo)) * Float32(W))
        var ch = Int((lo + u[13] * (hi - lo)) * Float32(H))
        var cx = Int(u[10] * Float32(W))
        var cy = Int(u[11] * Float32(H))
        p[W_CUT_X0] = Float32(cx - cw // 2)
        p[W_CUT_X1] = Float32(cx - cw // 2 + cw)
        p[W_CUT_Y0] = Float32(cy - ch // 2)
        p[W_CUT_Y1] = Float32(cy - ch // 2 + ch)
    p[W_CUT_FILL] = u[14]
    p[W_NOISE_KEY] = Float32(Int(u[15] * Float32(1 << 24)) & ((1 << 24) - 1))
    p[W_ENABLED] = 1.0
    return p


# ── apply: one output pixel (shared by host and device) ─────────────────


@always_inline
def _hash32(x: UInt32) -> UInt32:
    """lowbias32 (Wellons). A counter hash: same integer function on the host
    and on every device, so the noise is bit-reproducible across legs."""
    var h = x
    h ^= h >> 16
    h *= UInt32(0x7FEB352D)
    h ^= h >> 15
    h *= UInt32(0x846CA68B)
    h ^= h >> 16
    return h


@always_inline
def aug_src_index(
    p: SIMD[DType.float32, AUG_WORDS], y: Int, x: Int, H: Int, W: Int
) -> Int:
    """Pixel index `sy * W + sx` inside the channel plane to read for output
    `(y, x)` — the shift, edge-clamped."""
    var sy = y - Int(p[W_DY])
    var sx = x - Int(p[W_DX])
    if sy < 0:
        sy = 0
    if sy > H - 1:
        sy = H - 1
    if sx < 0:
        sx = 0
    if sx > W - 1:
        sx = W - 1
    return sy * W + sx


@always_inline
def aug_value_u8(
    p: SIMD[DType.float32, AUG_WORDS],
    src_u8: UInt8,
    ch: Int,
    y: Int,
    x: Int,
    pix: Int,
) -> UInt8:
    """The augmented byte for channel `ch` at output `(y, x)`, given the byte
    `aug_src_index` pointed at. `pix = ch * H*W + y*W + x` keys the noise."""
    if p[W_ENABLED] == 0.0:
        return src_u8
    var v = Float32(Int(src_u8)) / Float32(255.0)
    var gain = p[W_GAIN_R] if ch == 0 else (
        p[W_GAIN_G] if ch == 1 else p[W_GAIN_B]
    )
    v = v * gain
    v = (v - Float32(0.5)) * p[W_CONTRAST] + Float32(0.5) + p[W_BRIGHT]
    if v < 0.0:
        v = 0.0
    if v > 1.0:
        v = 1.0
    # `exp(g * log v)`, not `v ** g`: `powf` is a libm symbol that does not
    # lower on every device (the `atan2f` ptxas failure). Floor keeps log finite.
    v = exp(p[W_GAMMA] * log(max(v, Float32(1e-6))))
    if p[W_NOISE] > 0.0:
        var key = UInt32(Int(p[W_NOISE_KEY]))
        var h = _hash32(key * UInt32(0x9E3779B1) ^ UInt32(pix))
        var u = Float32(Int(h >> 8)) * Float32(1.0 / 16777216.0)
        # U[-1,1) scaled by sqrt(3) has unit std.
        v += p[W_NOISE] * Float32(1.7320508) * (Float32(2.0) * u - Float32(1.0))
    if (
        Float32(x) >= p[W_CUT_X0]
        and Float32(x) < p[W_CUT_X1]
        and Float32(y) >= p[W_CUT_Y0]
        and Float32(y) < p[W_CUT_Y1]
    ):
        v = p[W_CUT_FILL]
    if v < 0.0:
        v = 0.0
    if v > 1.0:
        v = 1.0
    return UInt8(Int(v * Float32(255.0) + Float32(0.5)))


# ── host twin ────────────────────────────────────────────────────────────


def augment_camera_u8[
    IMG_H: Int, IMG_W: Int
](
    ref src: List[Scalar[DType.uint8]],
    src_off: Int,
    p: SIMD[DType.float32, AUG_WORDS],
    mut dst: List[Scalar[DType.uint8]],
    dst_off: Int,
) raises:
    """One camera's `[3, H, W]` uint8 slot -> augmented uint8 slot.

    The host leg of the device gather: feed the result to
    `normalize_camera_chw` and it must match `_act_gather_images_aug_kernel`
    given the same record (to one quantisation step — float contraction can
    differ between legs by an ULP, and an ULP at a .5 boundary is a byte)."""
    comptime HW = IMG_H * IMG_W
    if len(src) < src_off + 3 * HW:
        raise Error("augment_camera_u8: source slot is short")
    if len(dst) < dst_off + 3 * HW:
        raise Error("augment_camera_u8: destination slot is short")
    for ch in range(3):
        var base = ch * HW
        for y in range(IMG_H):
            for x in range(IMG_W):
                var s = src[
                    src_off + base + aug_src_index(p, y, x, IMG_H, IMG_W)
                ]
                var pix = base + y * IMG_W + x
                dst[dst_off + pix] = aug_value_u8(p, s, ch, y, x, pix)
