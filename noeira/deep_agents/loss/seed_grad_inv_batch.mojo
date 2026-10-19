"""Seed gradient tile of shape [BATCH, 1] with 1/BATCH.

This is the standard "seed the backward pass for a mean-over-batch
loss" helper: when the forward output is `loss_per_b` of shape
[BATCH, 1] and the trainer wants `d(mean_b(loss_per_b))/d(loss_per_b)`,
that gradient is the constant `1/BATCH` in every slot.

Replaces the inline `_fill_constant_kernel` previously open-coded in
`sac_actor_loss.mojo`. Both CPU and GPU paths.

Phase 3 — Lives in `loss/` because mean-batch backward is the loss
contract. Other primitives that need a constant-fill should call this
helper rather than re-rolling.
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from noeira.core.tiles import Tile

from noeira.nn.constants import DT, TPB


def seed_inv_batch_kernel[
    N: Int
](
    grad_out: Tile[DT, N, 1],
    value: Scalar[DT],
):
    var idx = Int(global_idx.x)
    if idx < N:
        grad_out[idx, 0] = value


def seed_grad_inv_batch[
    target: StaticString,
    BATCH: Int,
](
    grad_out: Tile[DT, BATCH, 1],
    ctx: Optional[DeviceContext] = None,
) raises:
    """Fill `grad_out_ptr[0:BATCH]` with `1/BATCH`.

    Caller-provided `ctx` is required for `target='gpu'`; ignored for
    `target='cpu'`.
    """
    var inv_b: Scalar[DT] = Scalar[DT](1.0) / Scalar[DT](BATCH)
    comptime if target == "cpu":
        for b in range(BATCH):
            grad_out[b, 0] = inv_b
    else:
        var c = ctx.value()

        comptime n_blocks = (BATCH + TPB - 1) // TPB
        comptime kernel = seed_inv_batch_kernel[BATCH]
        c.enqueue_function[kernel](
            grad_out,
            inv_b,
            grid_dim=n_blocks,
            block_dim=TPB,
        )
