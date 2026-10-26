"""Per-epoch GPU input augmentation for the storage Trainer.

`Augmenter` is a compile-time hook: `Trainer.train_gpu` takes an `AUGMENTER`
parameter (default `IdentityAugmenter`) and, when it isn't a no-op, allocates an
augmentation buffer and calls `AUGMENTER.augment(...)` once per epoch to
(re)fill it from the raw training set before the mini-batch loop. A non-no-op
augmenter must FULLY write its `aug` output from `raw` each call (the trainer
does not pre-copy).

`IS_NOOP` lets the trainer skip the extra buffer + per-epoch call when no
augmentation is requested. Ported verbatim from the legacy
`nn/training/augmenter.mojo` (framework-agnostic GPU code; operates on raw
`LayoutTensor` device views, the storage `Trainer` passes `tensor.lt[...]`).
"""

from max.gpu import thread_idx, block_idx
from max.gpu.host import DeviceContext
from std.random.philox import Random as PhiloxRandom

from noeira.nn.constants import TPB, LAYOUT_NCHW, LAYOUT_NHWC
from noeira.core.tiles import Tile, tile


trait Augmenter(ImplicitlyCopyable):
    """Per-epoch GPU input augmentation hook."""

    comptime IS_NOOP: Bool

    @staticmethod
    def augment[N: Int, IN_DIM: Int, dtype: DType](
        ctx: DeviceContext,
        aug: Tile[dtype, N, IN_DIM],
        raw: Tile[dtype, N, IN_DIM],
        epoch: Int,
        seed: UInt64,
    ) raises:
        """Write augmented samples into `aug` from `raw` for this epoch.

        Must fully populate `aug` from `raw` (the trainer does not pre-copy).
        Combine `seed` with `epoch` for per-epoch variation.
        """
        ...


struct IdentityAugmenter(Augmenter):
    """No-op augmenter (default). Never invoked by the trainer."""

    comptime IS_NOOP: Bool = True

    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    def __init__(out self, *, deinit move: Self):
        pass

    @staticmethod
    def augment[N: Int, IN_DIM: Int, dtype: DType](
        ctx: DeviceContext,
        aug: Tile[dtype, N, IN_DIM],
        raw: Tile[dtype, N, IN_DIM],
        epoch: Int,
        seed: UInt64,
    ) raises:
        pass


def _cifar_augment_kernel[
    N: Int, dtype: DType, LAYOUT: Int = LAYOUT_NCHW,
](
    aug: Tile[dtype, N, 3 * 32 * 32],
    raw: Tile[dtype, N, 3 * 32 * 32],
    epoch_seed: Scalar[DType.uint64],
):
    """One block per sample; threads parallelize the 3072 output pixels. All
    threads in a block derive the same dx/dy/flip from
    PhiloxRandom(epoch_seed, b); out-of-bounds pixels get 0. LAYOUT selects how
    the flat index decodes to (c, y, x): NCHW = c*HW + y*W + x (channel-outer),
    NHWC = (y*W + x)*C + c (channel-inner). Both `aug` and `raw` are in LAYOUT, so
    the augmented batch matches the net's expected channels-last/first input."""
    var b = Int(block_idx.x)
    if b >= N:
        return
    var tid = Int(thread_idx.x)

    comptime C = 3
    comptime H = 32
    comptime W = 32
    comptime CHAN = H * W
    comptime IMG_SIZE = C * CHAN

    var rng = PhiloxRandom(seed=UInt64(epoch_seed), offset=UInt64(b))
    var r = rng.step_uniform()
    var dx = Int(Scalar[DType.float32](r[0]) * 9.0) - 4  # [-4, 4]
    var dy = Int(Scalar[DType.float32](r[1]) * 9.0) - 4  # [-4, 4]
    var flip = Scalar[DType.float32](r[2]) > 0.5

    var idx = tid
    while idx < IMG_SIZE:
        # Decode flat idx → (c, oy, ox). LAYOUT is comptime so each ternary folds
        # to one branch (no uninitialized locals — Metal rejects those).
        var c = (idx % C) if LAYOUT == LAYOUT_NHWC else (idx // CHAN)
        var sp = (idx // C) if LAYOUT == LAYOUT_NHWC else (idx % CHAN)
        var oy = sp // W
        var ox = sp % W
        var src_y = oy + dy
        var vx = ox + dx
        var val = Scalar[dtype](0.0)
        if src_y >= 0 and src_y < H and vx >= 0 and vx < W:
            var src_x = (W - 1 - vx) if flip else vx
            var ridx = (
                (src_y * W + src_x) * C + c
            ) if LAYOUT == LAYOUT_NHWC else (c * CHAN + src_y * W + src_x)
            val = rebind[Scalar[dtype]](raw[b, ridx])
        aug[b, idx] = val
        idx += TPB


struct CIFAR10CropFlipAugmenter(Augmenter):
    """CIFAR-10 random pad-4 crop + horizontal flip, per sample, per epoch (the
    standard CIFAR-10 ResNet recipe). Hardcoded to 3×32×32 = 3072."""

    comptime IS_NOOP: Bool = False

    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    def __init__(out self, *, deinit move: Self):
        pass

    @staticmethod
    def augment[N: Int, IN_DIM: Int, dtype: DType](
        ctx: DeviceContext,
        aug: Tile[dtype, N, IN_DIM],
        raw: Tile[dtype, N, IN_DIM],
        epoch: Int,
        seed: UInt64,
    ) raises:
        comptime assert (
            IN_DIM == 3 * 32 * 32
        ), "CIFAR10CropFlipAugmenter requires IN_DIM == 3*32*32"
        var aug_fixed = tile[N, 3 * 32 * 32](aug.ptr)
        var raw_fixed = tile[N, 3 * 32 * 32](raw.ptr)
        comptime aug_k = _cifar_augment_kernel[N, dtype]
        ctx.enqueue_function[aug_k](
            aug_fixed,
            raw_fixed,
            Scalar[DType.uint64](seed + UInt64(epoch)),
            grid_dim=(N,),
            block_dim=(TPB,),
        )


struct CIFAR10CropFlipAugmenterNHWC(Augmenter):
    """Channels-last twin of `CIFAR10CropFlipAugmenter` — identical pad-4 crop +
    horizontal flip recipe, but indexes both `raw` and `aug` in NHWC (y*W+x)*C+c
    order so the augmented batch feeds a channels-last net. Same seeds/dx/dy/flip
    ⇒ pixel-for-pixel the same augmentation as the NCHW version, just transposed.
    Used by the NHWC ResNet-20 CIFAR-10 convergence A/B."""

    comptime IS_NOOP: Bool = False

    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    def __init__(out self, *, deinit move: Self):
        pass

    @staticmethod
    def augment[N: Int, IN_DIM: Int, dtype: DType](
        ctx: DeviceContext,
        aug: Tile[dtype, N, IN_DIM],
        raw: Tile[dtype, N, IN_DIM],
        epoch: Int,
        seed: UInt64,
    ) raises:
        comptime assert (
            IN_DIM == 3 * 32 * 32
        ), "CIFAR10CropFlipAugmenterNHWC requires IN_DIM == 3*32*32"
        var aug_fixed = tile[N, 3 * 32 * 32](aug.ptr)
        var raw_fixed = tile[N, 3 * 32 * 32](raw.ptr)
        comptime aug_k = _cifar_augment_kernel[N, dtype, LAYOUT_NHWC]
        ctx.enqueue_function[aug_k](
            aug_fixed,
            raw_fixed,
            Scalar[DType.uint64](seed + UInt64(epoch)),
            grid_dim=(N,),
            block_dim=(TPB,),
        )
