"""Uint8 HWC → fp32 CHW/÷255 pixel conversion (LeWM real-data path).

Environments hand frames as `uint8` in HWC (height, width, channel)
layout; the encoder's `PatchEmbed` expects `fp32` CHW normalised to
[0,1]. This is the one bridge between the two. Stored offline data is
4× smaller as uint8, so the real NVIDIA driver uploads uint8 and converts
on-device right before the encoder — `u8_hwc_to_chw_norm[target=...]`
covers both CPU (reference / offline ingest) and GPU (the batched path).

  chw[(c·H + h)·W + w] = hwc[(h·W + w)·C + c] / 255.0           per sample
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from noeira.core.tiles import RowMajor, Tile, tile

from ...nn.constants import DT, TPB


def _u8_hwc_to_chw_kernel[C: Int, H: Int, W: Int, BATCH: Int](
    src: Tile[DType.uint8, BATCH * H * W * C],
    dst: Tile[DT, BATCH * C * H * W],
):
    comptime HWC = H * W * C
    var idx = Int(global_idx.x)
    if idx >= BATCH * HWC:
        return
    # decode flat dst index → (b, c, h, w)
    var b = idx // HWC
    var rem = idx % HWC
    var c = rem // (H * W)
    var hw = rem % (H * W)
    var h = hw // W
    var w = hw % W
    var src_idx = b * HWC + (h * W + w) * C + c
    dst[idx] = src[src_idx].cast[DT]() / Scalar[DT](255.0)


def _u8_norm_kernel[N: Int](
    src: Tile[DType.uint8, N],
    dst: Tile[DT, N],
):
    var idx = Int(global_idx.x)
    if idx < N:
        dst[idx] = src[idx].cast[DT]() / Scalar[DT](255.0)


def u8_to_fp32_norm[
    target: StaticString,
    N: Int,
    so: MutOrigin = MutAnyOrigin,
    do: MutOrigin = MutAnyOrigin,
](
    src: Pointer[Scalar[DType.uint8], so],
    dst: Pointer[Scalar[DT], do],
    ctx: Optional[DeviceContext] = None,
) raises:
    """Layout-preserving uint8 → fp32 ÷255 over `N` elements. Use when the
    source is already in the encoder's channel-major (CHW) layout — e.g.
    Pong frames (`PongOfflineBuffer.INPUT_LAYOUT_HWC == False`), so no
    permute is needed, only the normalise. `ctx` required for target='gpu'."""
    comptime assert target == "cpu" or target == "gpu", (
        "u8_to_fp32_norm: target must be 'cpu' or 'gpu'"
    )
    comptime if target == "cpu":
        for i in range(N):
            dst[unsafe_offset=i] = src[unsafe_offset=i].cast[DT]() / Scalar[DT](255.0)
    else:
        if not ctx:
            raise Error("u8_to_fp32_norm[target='gpu']: ctx required")
        var c = ctx.value()
        var src_lt = tile[N](src)
        var dst_lt = tile[N](dst)
        comptime n_blocks = (N + TPB - 1) // TPB
        comptime kernel = _u8_norm_kernel[N]
        c.enqueue_function[kernel](
            src_lt, dst_lt, grid_dim=n_blocks, block_dim=TPB,
        )


def u8_hwc_to_chw_norm[
    target: StaticString,
    C: Int, H: Int, W: Int, BATCH: Int,
    so: MutOrigin = MutAnyOrigin,
    do: MutOrigin = MutAnyOrigin,
](
    src: Pointer[Scalar[DType.uint8], so],
    dst: Pointer[Scalar[DT], do],
    ctx: Optional[DeviceContext] = None,
) raises:
    """Convert `src` (BATCH·H·W·C uint8, HWC) → `dst` (BATCH·C·H·W fp32,
    CHW, ÷255). `ctx` required for target='gpu'."""
    comptime assert target == "cpu" or target == "gpu", (
        "u8_hwc_to_chw_norm: target must be 'cpu' or 'gpu'"
    )
    comptime HWC = H * W * C
    comptime if target == "cpu":
        for b in range(BATCH):
            for c in range(C):
                for h in range(H):
                    for w in range(W):
                        var d = b * HWC + (c * H + h) * W + w
                        var sidx = b * HWC + (h * W + w) * C + c
                        dst[unsafe_offset=d] = src[unsafe_offset=sidx].cast[DT]() / Scalar[DT](255.0)
    else:
        if not ctx:
            raise Error("u8_hwc_to_chw_norm[target='gpu']: ctx required")
        var c = ctx.value()
        var src_lt = tile[BATCH * HWC](src)
        var dst_lt = tile[BATCH * HWC](dst)
        comptime n = BATCH * HWC
        comptime n_blocks = (n + TPB - 1) // TPB
        comptime kernel = _u8_hwc_to_chw_kernel[C, H, W, BATCH]
        c.enqueue_function[kernel](
            src_lt, dst_lt, grid_dim=n_blocks, block_dim=TPB,
        )
