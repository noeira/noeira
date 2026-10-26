# +--------------------------------------------------------------------------+ #
# | Three slab ops the backward driver needs and no `nn` leaf provides
# +--------------------------------------------------------------------------+ #
"""Accumulate, copy, and take the suffix tail of a `[B, FULL*KVW]` slab.

These are not layers and have no `vjp` of their own — they are the plumbing
between leaf `vjp` calls, which the `nn` contract does not cover:

  * **`accum_into`** exists because `Module.vjp` ASSIGNS its `grad_inputs`
    rather than accumulating into them. When one activation feeds several
    consumers — `H` feeds q, k and v; `X2` feeds both the residual and the
    second norm — each `vjp` must write to its own slab and the driver sums
    them. Letting two `vjp`s share a destination would silently keep only the
    last.

  * **`suffix_tail`** undoes `SmolVLAKVCache.build_scratch`. The forward
    splices `[prefix ; suffix]` per batch element; the backward wants the
    suffix rows back, because only those carry a gradient into the expert's
    own K/V. The prefix rows carry the gradient into the VLM, which the
    `train_state_proj = False` regime discards.

  * **`prefix_head`** is the other half of the same slab — the rows that
    carry the gradient into the VLM's cached K/V, and so into `state_proj`.
    Discarded under `train_state_proj = False`, which is why it arrived later
    than its twin.

⚠ `suffix_tail` and `prefix_head` mirror `build_scratch`'s BATCH-MAJOR layout — batch b's
prefix followed by batch b's own suffix. If one of the two ever changes the
other must, and at B == 1 neither would notice.
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT, TPB
from noeira.nn.core.tensor import Tensor
from noeira.core.tiles import Tile


def _accum_kernel[N: Int](
    dst: Tile[DT, N],
    src: Tile[DT, N],
):
    var i = Int(global_idx.x)
    if i < N:
        dst.ptr[unsafe_offset=i] = rebind[Scalar[DT]](
            dst.ptr[unsafe_offset=i]
        ) + rebind[Scalar[DT]](src.ptr[unsafe_offset=i])


def _copy_kernel[N: Int](
    dst: Tile[DT, N],
    src: Tile[DT, N],
):
    var i = Int(global_idx.x)
    if i < N:
        dst.ptr[unsafe_offset=i] = rebind[Scalar[DT]](
            src.ptr[unsafe_offset=i]
        )


def _suffix_kernel[B: Int, FULL_B: Int, PRE_B: Int, SUF_B: Int](
    src: Tile[DT, B, FULL_B],
    dst: Tile[DT, B, SUF_B],
):
    var i = Int(global_idx.x)
    if i >= B * SUF_B:
        return
    var b = i // SUF_B
    var j = i % SUF_B
    dst.ptr[unsafe_offset=i] = rebind[Scalar[DT]](
        src.ptr[unsafe_offset = b * FULL_B + PRE_B + j]
    )


def accum_into[
    target: StaticString, N: Int
](mut dst: Tensor, mut src: Tensor, ctx: Optional[DeviceContext] = None
) raises:
    """`dst += src` over N elements. `dst` must already hold a gradient."""
    comptime if target == "cpu":
        for i in range(N):
            dst.data[i] = dst.data[i] + src.data[i]
    else:
        var c = ctx.value()
        c.enqueue_function[_accum_kernel[N]](
            dst.tt["gpu", N](),
            src.tt["gpu", N](),
            grid_dim=(N + TPB - 1) // TPB,
            block_dim=TPB,
        )


def copy_into[
    target: StaticString, N: Int
](mut dst: Tensor, mut src: Tensor, ctx: Optional[DeviceContext] = None
) raises:
    """`dst = src` over N elements, sizing `dst` first."""
    comptime if target == "cpu":
        dst.ensure(N)
        for i in range(N):
            dst.data[i] = src.data[i]
    else:
        var c = ctx.value()
        dst.ensure_gpu(c, N)
        c.enqueue_function[_copy_kernel[N]](
            dst.tt["gpu", N](),
            src.tt["gpu", N](),
            grid_dim=(N + TPB - 1) // TPB,
            block_dim=TPB,
        )


def suffix_tail[
    target: StaticString, B: Int, FULL_B: Int, PRE_B: Int, SUF_B: Int
](mut src: Tensor, mut dst: Tensor, ctx: Optional[DeviceContext] = None
) raises:
    """The suffix rows of a batch-major `[B, FULL_B]` scratch slab."""
    comptime if target == "cpu":
        dst.ensure(B * SUF_B)
        for b in range(B):
            for j in range(SUF_B):
                dst.data[b * SUF_B + j] = src.data[b * FULL_B + PRE_B + j]
    else:
        var c = ctx.value()
        dst.ensure_gpu(c, B * SUF_B)
        c.enqueue_function[_suffix_kernel[B, FULL_B, PRE_B, SUF_B]](
            src.tt["gpu", B, FULL_B](),
            dst.tt["gpu", B, SUF_B](),
            grid_dim=(B * SUF_B + TPB - 1) // TPB,
            block_dim=TPB,
        )


def _prefix_kernel[B: Int, FULL_B: Int, PRE_B: Int](
    src: Tile[DT, B, FULL_B],
    dst: Tile[DT, B, PRE_B],
):
    var i = Int(global_idx.x)
    if i >= B * PRE_B:
        return
    var b = i // PRE_B
    var j = i % PRE_B
    dst.ptr[unsafe_offset=i] = rebind[Scalar[DT]](
        src.ptr[unsafe_offset = b * FULL_B + j]
    )


def prefix_head[
    target: StaticString, B: Int, FULL_B: Int, PRE_B: Int
](
    mut src: Tensor, mut dst: Tensor, ctx: Optional[DeviceContext] = None
) raises:
    """The PREFIX rows of a batch-major `[B, FULL_B]` scratch slab.

    ⚠ The complement of `suffix_tail`, and the two must stay complementary:
    together they are the whole slab, and a gradient that lands in neither is
    silently dropped while a gradient counted in both is doubled.
    """
    comptime if target == "cpu":
        dst.ensure(B * PRE_B)
        for b in range(B):
            for j in range(PRE_B):
                dst.data[b * PRE_B + j] = src.data[b * FULL_B + j]
    else:
        var c = ctx.value()
        dst.ensure_gpu(c, B * PRE_B)
        c.enqueue_function[_prefix_kernel[B, FULL_B, PRE_B]](
            src.tt["gpu", B, FULL_B](),
            dst.tt["gpu", B, PRE_B](),
            grid_dim=(B * PRE_B + TPB - 1) // TPB,
            block_dim=TPB,
        )


def _zero_kernel[N: Int](
    dst: Tile[DT, N],
):
    var i = Int(global_idx.x)
    if i < N:
        dst.ptr[unsafe_offset=i] = Scalar[DT](0)


def zero_into[
    target: StaticString, N: Int
](mut dst: Tensor, ctx: Optional[DeviceContext] = None) raises:
    """`dst[0:N] = 0`, sizing it first.

    Used to seed a backward whose starting gradient is genuinely zero — the
    prefill's, whose final hidden state the loss never reads.
    """
    comptime if target == "cpu":
        dst.ensure(N)
        for i in range(N):
            dst.data[i] = Scalar[DT](0)
    else:
        var c = ctx.value()
        dst.ensure_gpu(c, N)
        c.enqueue_function[_zero_kernel[N]](
            dst.tt["gpu", N](),
            grid_dim=(N + TPB - 1) // TPB,
            block_dim=TPB,
        )
