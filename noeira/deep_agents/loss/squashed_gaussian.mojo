"""Canonical squashed-Gaussian forward/backward (Phase E + Block D).

Single source of truth for the SAC reparameterized squashed-Gaussian
sampling + log-prob math. Today this math is duplicated in three places
(audit P1 finding):

  1. `nn/primitives/rsample.mojo` — as `RSample[ACT]` Module.
  2. `nn/loss/sac_actor_loss.mojo` — as free functions (basically the
     same as here).
  3. `nn/loss/sac_actor_loss.mojo` — composes the Module form.

Phase E added this file as the canonical CPU implementation. Block D
(2026-05-21) adds the GPU variants `*_gpu`. Same math, one thread per
batch row in forward (per-row log-prob reduction), one thread per
[b, j] in backward.

# Forward

    action[b, j]   = action_scale · tanh(mu_j + exp(clamp(log_std_j)) · z_j)
    log_prob[b]    = Σ_j ( -0.5·z_j²  - log_std_j  - 0.5·log(2π)
                            - log(action_scale·(1-y²) + ε) )

# Backward (per-element chain factors)

    y               = tanh(mu + std·z)
    da/dmu          = c · (1 - y²)             c = action_scale
    da/dlog_std     = c · (1 - y²) · z · std
    dlp/dmu         = 2·y·c·(1-y²) / corr      corr = c·(1-y²) + ε
    dlp/dlog_std    = -1 + 2·y·c·(1-y²)·z·std / corr

    grad_mu[b, j]      = grad_action[b, j]  · da/dmu
                        + grad_log_prob[b]  · dlp/dmu
    grad_log_std[b, j] = grad_action[b, j]  · da/dlog_std
                        + grad_log_prob[b]  · dlp/dlog_std

Log-std clamp mask: when raw log_std is outside [LOG_STD_MIN,
LOG_STD_MAX], the un-clamped value didn't affect downstream — its
grad is zeroed.
"""

from std.math import exp, log, tanh as ftanh
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from max.gpu.memory import AddressSpace
from layout import TileTensor
from noeira.core.tiles import Tile, tile

from noeira.nn.constants import DT, TPB, TPB_REDUCE


comptime LOG_STD_MIN: Scalar[DT] = -5.0
comptime LOG_STD_MAX: Scalar[DT] =  2.0
comptime EPS_TANH_CORR: Scalar[DT] = 1e-6
comptime LOG_2PI: Scalar[DT] = 1.8378770664093453


def _clamp_log_std(ls: Scalar[DT]) -> Scalar[DT]:
    if ls < LOG_STD_MIN:
        return LOG_STD_MIN
    elif ls > LOG_STD_MAX:
        return LOG_STD_MAX
    return ls


def squashed_gaussian_forward[ACT: Int, BATCH: Int](
    actor_output: TileTensor[
        dtype=DT, address_space=AddressSpace.GENERIC, ...
    ],
    z: TileTensor[
        dtype=DT, address_space=AddressSpace.GENERIC, ...
    ],
    action_scale: Scalar[DT],
    mut action: TileTensor[
        mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
        ...,
    ],
    mut log_prob: TileTensor[
        mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
        ...,
    ],
) raises:
    """Compute (action, log_prob) from (mu, log_std, z).

    actor_output [BATCH, 2*ACT]   packed [mu | log_std]
    z            [BATCH, ACT]     reparam noise (caller draws)
    action       [BATCH, ACT]     squashed sample
    log_prob     [BATCH]          per-sample log-prob (sum over j)
    """
    comptime assert actor_output.flat_rank == 2, "actor_output rank-2 [BATCH, 2*ACT]"
    comptime assert z.flat_rank == 2,            "z rank-2 [BATCH, ACT]"
    comptime assert action.flat_rank == 2,       "action rank-2 [BATCH, ACT]"
    comptime assert log_prob.flat_rank == 1,     "log_prob rank-1 [BATCH]"
    comptime assert ACT >= 1, "ACT >= 1"

    for b in range(BATCH):
        var lp: Scalar[DT] = 0.0
        for j in range(ACT):
            var mu = rebind[Scalar[DT]](actor_output[b, j])
            var ls = _clamp_log_std(
                rebind[Scalar[DT]](actor_output[b, ACT + j])
            )
            var std = exp(ls)
            var zj = rebind[Scalar[DT]](z[b, j])
            var pre = mu + std * zj
            var y = ftanh(pre)
            action[b, j] = action_scale * y
            var corr = action_scale * (Scalar[DT](1.0) - y * y) + EPS_TANH_CORR
            lp += (
                Scalar[DT](-0.5) * zj * zj
                - ls
                - Scalar[DT](0.5) * LOG_2PI
                - log(corr)
            )
        log_prob[b] = lp


def squashed_gaussian_backward[ACT: Int, BATCH: Int](
    actor_output: TileTensor[
        dtype=DT, address_space=AddressSpace.GENERIC, ...
    ],
    z: TileTensor[
        dtype=DT, address_space=AddressSpace.GENERIC, ...
    ],
    grad_action: TileTensor[
        dtype=DT, address_space=AddressSpace.GENERIC, ...
    ],
    grad_log_prob: TileTensor[
        dtype=DT, address_space=AddressSpace.GENERIC, ...
    ],
    action_scale: Scalar[DT],
    mut grad_actor_output: TileTensor[
        mut=True, dtype=DT, address_space=AddressSpace.GENERIC,
        ...,
    ],
) raises:
    """Chain (grad_action, grad_log_prob) back to grad_[mu | log_std].

    grad_action      [BATCH, ACT]
    grad_log_prob    [BATCH]
    grad_actor_output [BATCH, 2*ACT]

    The caller pre-bakes any BATCH-normalization / α-scaling into
    `grad_action` / `grad_log_prob`. This function is dimensionally
    pure — it only knows the squashed-Gaussian Jacobian.
    """
    comptime assert actor_output.flat_rank == 2,      "actor_output rank-2"
    comptime assert z.flat_rank == 2,                 "z rank-2"
    comptime assert grad_action.flat_rank == 2,       "grad_action rank-2"
    comptime assert grad_log_prob.flat_rank == 1,     "grad_log_prob rank-1"
    comptime assert grad_actor_output.flat_rank == 2, "grad_actor_output rank-2"
    comptime assert ACT >= 1, "ACT >= 1"

    for b in range(BATCH):
        var glp = rebind[Scalar[DT]](grad_log_prob[b])
        for j in range(ACT):
            var mu = rebind[Scalar[DT]](actor_output[b, j])
            var ls_raw = rebind[Scalar[DT]](actor_output[b, ACT + j])
            var ls = _clamp_log_std(ls_raw)
            var ls_clamped = (ls_raw < LOG_STD_MIN) or (ls_raw > LOG_STD_MAX)

            var std = exp(ls)
            var zj = rebind[Scalar[DT]](z[b, j])
            var pre = mu + std * zj
            var y = ftanh(pre)
            var one_minus_y2 = Scalar[DT](1.0) - y * y
            var c_om = action_scale * one_minus_y2
            var corr = c_om + EPS_TANH_CORR

            var da_dmu = c_om
            var da_dls = c_om * zj * std
            var dlp_dmu = (Scalar[DT](2.0) * y * c_om) / corr
            var dlp_dls = (
                Scalar[DT](-1.0) + (Scalar[DT](2.0) * y * c_om * zj * std) / corr
            )

            var ga = rebind[Scalar[DT]](grad_action[b, j])
            var gmu = ga * da_dmu + glp * dlp_dmu
            var gls = ga * da_dls + glp * dlp_dls

            grad_actor_output[b, j] = gmu
            if ls_clamped:
                grad_actor_output[b, ACT + j] = Scalar[DT](0.0)
            else:
                grad_actor_output[b, ACT + j] = gls


# ──────────────────────────────────────────────────────────────────────
# GPU kernels — block D. One thread per batch row in forward (does the
# inner-j reduction sequentially since ACT is small for SAC envs); one
# thread per [b, j] in backward (no reduction needed there).
# ──────────────────────────────────────────────────────────────────────


def _sg_forward_kernel[
    ACT: Int, BATCH: Int,
](
    actor_output: Tile[DT, BATCH, 2 * ACT],
    z: Tile[DT, BATCH, ACT],
    action_scale: Scalar[DT],
    action: Tile[DT, BATCH, ACT],
    log_prob: Tile[DT, BATCH],
):
    var b = Int(global_idx.x)
    if b >= BATCH:
        return
    var lp: Scalar[DT] = 0.0
    for j in range(ACT):
        var mu = rebind[Scalar[DT]](actor_output[b, j])
        var ls_raw = rebind[Scalar[DT]](actor_output[b, ACT + j])
        var ls = ls_raw
        if ls < LOG_STD_MIN:
            ls = LOG_STD_MIN
        elif ls > LOG_STD_MAX:
            ls = LOG_STD_MAX
        var std = exp(ls)
        var zj = rebind[Scalar[DT]](z[b, j])
        var pre = mu + std * zj
        var y = ftanh(pre)
        action[b, j] = action_scale * y
        var corr = action_scale * (Scalar[DT](1.0) - y * y) + EPS_TANH_CORR
        lp += (
            Scalar[DT](-0.5) * zj * zj
            - ls
            - Scalar[DT](0.5) * LOG_2PI
            - log(corr)
        )
    log_prob[b] = lp


def _sg_backward_kernel[
    ACT: Int, BATCH: Int,
](
    actor_output: Tile[DT, BATCH, 2 * ACT],
    z: Tile[DT, BATCH, ACT],
    grad_action: Tile[DT, BATCH, ACT],
    grad_log_prob: Tile[DT, BATCH],
    action_scale: Scalar[DT],
    grad_actor_output: Tile[DT, BATCH, 2 * ACT],
):
    var idx = Int(global_idx.x)
    var total = BATCH * ACT
    if idx >= total:
        return
    var b = idx // ACT
    var j = idx % ACT

    var mu = rebind[Scalar[DT]](actor_output[b, j])
    var ls_raw = rebind[Scalar[DT]](actor_output[b, ACT + j])
    var ls = ls_raw
    var ls_clamped: Bool = False
    if ls < LOG_STD_MIN:
        ls = LOG_STD_MIN
        ls_clamped = True
    elif ls > LOG_STD_MAX:
        ls = LOG_STD_MAX
        ls_clamped = True

    var std = exp(ls)
    var zj = rebind[Scalar[DT]](z[b, j])
    var pre = mu + std * zj
    var y = ftanh(pre)
    var one_minus_y2 = Scalar[DT](1.0) - y * y
    var c_om = action_scale * one_minus_y2
    var corr = c_om + EPS_TANH_CORR

    var da_dmu = c_om
    var da_dls = c_om * zj * std
    var dlp_dmu = (Scalar[DT](2.0) * y * c_om) / corr
    var dlp_dls = (
        Scalar[DT](-1.0) + (Scalar[DT](2.0) * y * c_om * zj * std) / corr
    )

    var ga = rebind[Scalar[DT]](grad_action[b, j])
    var glp = rebind[Scalar[DT]](grad_log_prob[b])
    var gmu = ga * da_dmu + glp * dlp_dmu
    var gls = ga * da_dls + glp * dlp_dls

    grad_actor_output[b, j] = gmu
    if ls_clamped:
        grad_actor_output[b, ACT + j] = Scalar[DT](0.0)
    else:
        grad_actor_output[b, ACT + j] = gls


def squashed_gaussian_forward_gpu[ACT: Int, BATCH: Int](
    ctx: DeviceContext,
    actor_output_ptr: Pointer[Scalar[DT], MutAnyOrigin],
    z_ptr: Pointer[Scalar[DT], MutAnyOrigin],
    action_scale: Scalar[DT],
    action_ptr: Pointer[Scalar[DT], MutAnyOrigin],
    log_prob_ptr: Pointer[Scalar[DT], MutAnyOrigin],
) raises:
    """GPU forward — one thread per batch row, inner-j sequential.

    Pointer-based API (not TileTensor) because callers thread raw
    DeviceBuffer pointers through the SAC trainer scratch."""
    var ao_lt = tile[BATCH, 2 * ACT](actor_output_ptr)
    var z_lt = tile[BATCH, ACT](z_ptr)
    var a_lt = tile[BATCH, ACT](action_ptr)
    var lp_lt = tile[BATCH](log_prob_ptr)
    comptime TPB = TPB_REDUCE
    comptime n_blocks = (BATCH + TPB - 1) // TPB
    comptime kernel = _sg_forward_kernel[ACT, BATCH]
    ctx.enqueue_function[kernel](
        ao_lt, z_lt, action_scale, a_lt, lp_lt,
        grid_dim=n_blocks, block_dim=TPB,
    )


def squashed_gaussian_backward_gpu[ACT: Int, BATCH: Int](
    ctx: DeviceContext,
    actor_output_ptr: Pointer[Scalar[DT], MutAnyOrigin],
    z_ptr: Pointer[Scalar[DT], MutAnyOrigin],
    grad_action_ptr: Pointer[Scalar[DT], MutAnyOrigin],
    grad_log_prob_ptr: Pointer[Scalar[DT], MutAnyOrigin],
    action_scale: Scalar[DT],
    grad_actor_output_ptr: Pointer[Scalar[DT], MutAnyOrigin],
) raises:
    """GPU backward — one thread per [b, j], no reduction."""
    var ao_lt = tile[BATCH, 2 * ACT](actor_output_ptr)
    var z_lt = tile[BATCH, ACT](z_ptr)
    var ga_lt = tile[BATCH, ACT](grad_action_ptr)
    var glp_lt = tile[BATCH](grad_log_prob_ptr)
    var gao_lt = tile[BATCH, 2 * ACT](grad_actor_output_ptr)
    comptime total = BATCH * ACT
    comptime n_blocks = (total + TPB - 1) // TPB
    comptime kernel = _sg_backward_kernel[ACT, BATCH]
    ctx.enqueue_function[kernel](
        ao_lt, z_lt, ga_lt, glp_lt, action_scale, gao_lt,
        grid_dim=n_blocks, block_dim=TPB,
    )
