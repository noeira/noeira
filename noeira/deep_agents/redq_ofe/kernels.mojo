"""REDQ-OFE kernels — auxiliary MSE loss + its gradient (STORAGE surface).

The grad function is the load-bearing one — it seeds the backward chain
through `OFEPredictorHead → OFEActionBranch → OFEStateBranch`. The loss
function is a diagnostics-only host reduction.

STORAGE migration (Stage 5): the public CPU/GPU functions take owning
storage `Tensor`s (CPU `.data` host loop / GPU `.lt` device views) instead
of raw `Pointer`s — mirrors `redq/kernels.mojo`. The `rebind` /
raw-ptr usage that survives is confined to inside the GPU kernel (the GPU
ABI).

Math:
    loss = (1 / (BATCH * OBS)) * Σ_b Σ_d (pred[b,d] - target[b,d])^2
    d loss / d pred[b,d] = 2 * (pred[b,d] - target[b,d]) / (BATCH * OBS)
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT, TPB
from noeira.nn.core.tensor import Tensor
from noeira.core.tiles import RowMajor, Tile


def aux_mse_grad_cpu[
    BATCH: Int, OBS: Int,
](
    mut pred: Tensor,        # [BATCH, OBS]
    mut target: Tensor,      # [BATCH, OBS]
    mut grad_pred: Tensor,   # [BATCH, OBS]
):
    """Writes `grad_pred[b,d] = 2 * (pred[b,d] - target[b,d]) / (BATCH * OBS)`
    over the full `BATCH * OBS` slab. Overwrites grad_pred."""
    var scale = Scalar[DT](2.0) / Scalar[DT](BATCH * OBS)
    for b in range(BATCH):
        for d in range(OBS):
            var off = b * OBS + d
            grad_pred.data[off] = scale * (pred.data[off] - target.data[off])


def aux_mse_loss_cpu[
    BATCH: Int, OBS: Int,
](
    mut pred: Tensor,        # [BATCH, OBS]
    mut target: Tensor,      # [BATCH, OBS]
) -> Scalar[DT]:
    """Returns the scalar `(1 / (BATCH * OBS)) * Σ (pred - target)^2`.
    Pure host reduction — diagnostics only."""
    var s: Scalar[DT] = Scalar[DT](0.0)
    for b in range(BATCH):
        for d in range(OBS):
            var off = b * OBS + d
            var diff = pred.data[off] - target.data[off]
            s += diff * diff
    return s / Scalar[DT](BATCH * OBS)


# ────────────────────────────────────────────────────────────────────
# GPU MSE-grad kernel.
# ────────────────────────────────────────────────────────────────────


def _aux_mse_grad_kernel[
    BATCH: Int, OBS: Int,
](
    pred: Tile[DT, BATCH, OBS],
    target: Tile[DT, BATCH, OBS],
    grad_pred: Tile[DT, BATCH, OBS],
    scale: Scalar[DT],
):
    var idx = Int(global_idx.x)
    var total = BATCH * OBS
    if idx >= total:
        return
    var b = idx // OBS
    var d = idx % OBS
    grad_pred[b, d] = scale * (
        rebind[Scalar[DT]](pred[b, d]) - rebind[Scalar[DT]](target[b, d])
    )


def aux_mse_grad_gpu[
    BATCH: Int, OBS: Int,
](
    ctx: DeviceContext,
    mut pred: Tensor,        # [BATCH, OBS]
    mut target: Tensor,      # [BATCH, OBS]
    mut grad_pred: Tensor,   # [BATCH, OBS]
) raises:
    """Device variant of `aux_mse_grad_cpu`. Writes
    `grad_pred[b,d] = 2·(pred[b,d] - target[b,d]) / (BATCH·OBS)`."""
    var scale = Scalar[DT](2.0) / Scalar[DT](BATCH * OBS)
    comptime lyt = RowMajor[BATCH, OBS]
    comptime total = BATCH * OBS
    comptime n_blocks = (total + TPB - 1) // TPB
    comptime kernel = _aux_mse_grad_kernel[BATCH, OBS]
    ctx.enqueue_function[kernel](
        pred.tt["gpu", BATCH, OBS](),
        target.tt["gpu", BATCH, OBS](),
        grad_pred.tt["gpu", BATCH, OBS](),
        scale,
        grid_dim=n_blocks, block_dim=TPB,
    )
