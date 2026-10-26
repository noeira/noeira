"""Masked reconstruction loss (model.py:recon_loss_from_mae).

Dreamer 4 trains the tokenizer with masked autoencoding: the MSE is computed
ONLY over the patches that MAE dropped (the model must reconstruct them).

    masked = (keep == 0)          # keep is MAEReplacer's per-patch flag
    denom  = (#masked patches) * DP
    loss   = Σ_{masked, d} (pred - target)² / denom
    grad_pred = 2·(pred - target)·[masked] / denom

Operates at nn-BATCH = B·T with per-frame patch tokens (NP × DP). `keep` is
`[BATCH*NP]` (1.0 kept / 0.0 dropped) from `encoder.mae_mask_ptr()`. Pure CPU
arithmetic; returns the scalar loss and fills `grad_pred`.
"""

from std.math import max, log10
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from max.gpu.memory import AddressSpace

from noeira.nn.constants import DT, TPB
from noeira.core.tiles import Tile, tile


def masked_recon_loss[
    NP: Int, DP: Int, BATCH: Int
](
    pred: Pointer[Scalar[DT], MutAnyOrigin],
    target: Pointer[Scalar[DT], MutAnyOrigin],
    keep: Pointer[Scalar[DT], MutAnyOrigin],
    grad_pred: Pointer[Scalar[DT], MutAnyOrigin],
) raises -> Float64:
    # Count masked patches (keep == 0) across the whole B·T batch.
    var n_masked = 0
    for j in range(BATCH * NP):
        if keep[unsafe_offset=j] == Scalar[DT](0.0):
            n_masked += 1
    var denom = Float64(max(n_masked, 1) * DP)

    var loss: Float64 = 0.0
    for bt in range(BATCH):
        for i in range(NP):
            var masked = keep[unsafe_offset=bt * NP + i] == Scalar[DT](0.0)
            for d in range(DP):
                var idx = bt * NP * DP + i * DP + d
                if masked:
                    var diff = Float64(pred[unsafe_offset=idx]) - Float64(target[unsafe_offset=idx])
                    loss += diff * diff
                    grad_pred[unsafe_offset=idx] = Scalar[DT](2.0 * diff / denom)
                else:
                    grad_pred[unsafe_offset=idx] = Scalar[DT](0.0)
    return loss / denom


def full_recon_psnr[
    NP: Int, DP: Int, BATCH: Int
](
    pred: Pointer[Scalar[DT], MutAnyOrigin],
    target: Pointer[Scalar[DT], MutAnyOrigin],
) raises -> Float64:
    """Full-frame reconstruction PSNR over ALL patches (eval; run with MAE
    p=0). Assumes pixels in [0,1] (peak = 1): PSNR = -10·log10(MSE)."""
    from std.math import log10
    comptime n = BATCH * NP * DP
    var sse: Float64 = 0.0
    for i in range(n):
        var d = Float64(pred[unsafe_offset=i]) - Float64(target[unsafe_offset=i])
        sse += d * d
    var mse = sse / Float64(n)
    if mse <= 1e-12:
        return 120.0
    return -10.0 * log10(mse)


# ── GPU training gradient ───────────────────────────────────────────────
# grad_pred[idx] = 2·(pred - target)·[masked] / DENOM, masked = (keep==0).
# DENOM is the fixed total element count (NOT #masked·DP) — the gradient
# direction is identical to the reference masked-MSE; the constant scale is
# absorbed by Adam. Avoids a per-step keep download just to count masked
# patches. PSNR eval (p=0) is the real quality metric and is unaffected.
def _masked_grad_kernel[
    N: Int, DP: Int
](
    pred: Tile[DT, N],
    target: Tile[DT, N],
    keep: Tile[DT, N // DP],
    grad_pred: Tile[DT, N],
    denom: Scalar[DT],
):
    var idx = Int(global_idx.x)
    if idx >= N:
        return
    var patch = idx // DP
    if rebind[Scalar[DT]](keep.ptr[unsafe_offset=patch]) == Scalar[DT](0.0):
        grad_pred.ptr[unsafe_offset=idx] = (
            Scalar[DT](2.0)
            * (rebind[Scalar[DT]](pred.ptr[unsafe_offset=idx]) - rebind[Scalar[DT]](target.ptr[unsafe_offset=idx]))
            / denom
        )
    else:
        grad_pred.ptr[unsafe_offset=idx] = Scalar[DT](0.0)


def masked_recon_grad_gpu[
    NP: Int, DP: Int, BATCH: Int
](
    pred: Pointer[Scalar[DT], MutAnyOrigin],
    target: Pointer[Scalar[DT], MutAnyOrigin],
    keep: Pointer[Scalar[DT], MutAnyOrigin],
    grad_pred: Pointer[Scalar[DT], MutAnyOrigin],
    ctx: DeviceContext,
) raises:
    comptime N = BATCH * NP * DP
    var denom = Scalar[DT](Float64(N))
    var p_lt = tile[N](pred)
    var t_lt = tile[N](target)
    var k_lt = tile[N // DP](keep)
    var g_lt = tile[N](grad_pred)
    comptime nb = (N + TPB - 1) // TPB
    ctx.enqueue_function[_masked_grad_kernel[N, DP]](
        p_lt, t_lt, k_lt, g_lt, denom, grid_dim=nb, block_dim=TPB
    )
