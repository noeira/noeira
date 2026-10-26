"""REDQ ensemble-target kernels (CPU + GPU) — STORAGE surface.

Computes the REDQ TD target from a stacked block of N target-Q
evaluations:

    y[b] = r[b] + (1 - term[b]) * γ * (combined_Q(s', a')[b] - α * log_π(a'|s')[b])

`combined_Q` depends on `MODE`:
  MODE = REDQ_TARGET_MIN — min over `q_next[subset_idxs[m], b]` for m in 0..N_MIN
  MODE = REDQ_TARGET_AVE — mean over `q_next[n, b]` for n in 0..N_ENSEMBLE
  MODE = REDQ_TARGET_REM — random convex mixture (Philox draws per (b, n));
                          deferred (legacy GPU-only path; not on this surface)

STORAGE migration (Stage 5): the public CPU/GPU functions take owning storage
`Tensor`s (CPU `.data` host loop / GPU `.lt` device views) instead of raw
`Pointer`s. The `rebind` / raw-ptr usage that survives is confined to
inside the GPU kernel (the GPU ABI), matching SAC/DDPG/TD3.

CleanRL natural-termination semantics: bootstrap is DROPPED on `term=1` (real
termination) and KEPT on `term=0` (time-limit truncation). For truncation-only
envs (`term ≡ 0`) the mask reduces to `r + γ·(combined − α·lp)`.
"""

from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT, TPB
from noeira.nn.core.tensor import Tensor
from noeira.core.tiles import RowMajor, Tile


# =============================================================================
# MODE selectors — runtime ints carried as comptime params at the call site.
# Match the legacy `deep_agents/redq/config.mojo` constants exactly.
# =============================================================================


comptime REDQ_TARGET_MIN: Int = 0
comptime REDQ_TARGET_AVE: Int = 1
comptime REDQ_TARGET_REM: Int = 2


def redq_ensemble_target_cpu[
    N_ENSEMBLE: Int,
    N_MIN: Int,
    MODE: Int,
    BATCH: Int,
](
    mut rewards: Tensor,        # [BATCH]
    mut q_next: Tensor,         # [N_ENSEMBLE, BATCH] row-major
    mut terms: Tensor,          # [BATCH]
    mut log_probs: Tensor,      # [BATCH]
    subset_idxs: List[Int],     # [N_MIN]
    gamma: Scalar[DT],
    alpha: Scalar[DT],
    mut out_y: Tensor,          # [BATCH]
):
    """`out_y[b] = r[b] + (1 - term[b]) * γ * (combined_Q[b] - α * log_prob[b])`
    in-place, one host iteration per batch sample.

    `q_next` is the row-major [N_ENSEMBLE, BATCH] stack of target-Q
    evaluations; `q_next[n, b] = Q_target^n(s'[b], a'[b])`.

    `subset_idxs` (length N_MIN) names which target critics participate in
    MIN-mode. Set by the caller via host-side Fisher-Yates each step. Ignored
    when MODE != MIN."""
    comptime assert (
        MODE == REDQ_TARGET_MIN or MODE == REDQ_TARGET_AVE
    ), (
        "redq_ensemble_target_cpu: only MIN/AVE supported on CPU "
        "(REM is GPU-only; see kernels.mojo docstring)"
    )
    comptime assert N_MIN >= 1, "N_MIN must be >= 1"
    comptime assert N_MIN <= N_ENSEMBLE, "N_MIN must be <= N_ENSEMBLE"

    for b in range(BATCH):
        var combined: Scalar[DT]
        comptime if MODE == REDQ_TARGET_MIN:
            var first_idx = subset_idxs[0]
            combined = q_next.data[first_idx * BATCH + b]
            for m in range(1, N_MIN):
                var v = q_next.data[subset_idxs[m] * BATCH + b]
                if v < combined:
                    combined = v
        else:
            # MODE == REDQ_TARGET_AVE
            var acc: Scalar[DT] = Scalar[DT](0.0)
            for n in range(N_ENSEMBLE):
                acc += q_next.data[n * BATCH + b]
            combined = acc / Scalar[DT](N_ENSEMBLE)

        var soft_v = combined - alpha * log_probs.data[b]
        var bootstrap = gamma * soft_v
        var nonterm = Scalar[DT](1.0) - terms.data[b]
        out_y.data[b] = rewards.data[b] + nonterm * bootstrap


# =============================================================================
# GPU kernel — one thread per batch lane. Mirrors the CPU formula.
# =============================================================================


def _redq_ensemble_target_kernel[
    N_ENSEMBLE: Int,
    N_MIN: Int,
    MODE: Int,
    BATCH: Int,
](
    out_y: Tile[DT, BATCH],
    rewards: Tile[DT, BATCH],
    q_next: Tile[DT, N_ENSEMBLE, BATCH],
    terms: Tile[DT, BATCH],
    log_probs: Tile[DT, BATCH],
    subset_idxs: Tile[DType.uint32, N_MIN],
    gamma: Scalar[DT],
    alpha: Scalar[DT],
):
    """Compute the REDQ TD target on-device. NaN-guards both the per-critic Q
    read and the final target (match legacy semantics)."""
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= BATCH:
        return

    var combined: Scalar[DT] = Scalar[DT](0.0)
    comptime if MODE == REDQ_TARGET_MIN:
        var first_idx = Int(subset_idxs.ptr[unsafe_offset=0])
        var v0 = q_next.ptr[unsafe_offset=first_idx * BATCH + b]
        if v0 != v0:
            v0 = Scalar[DT](0.0)
        combined = v0
        comptime for m in range(1, N_MIN):
            var idx = Int(subset_idxs.ptr[unsafe_offset=m])
            var v = q_next.ptr[unsafe_offset=idx * BATCH + b]
            if v != v:
                v = Scalar[DT](0.0)
            if v < combined:
                combined = v
    comptime if MODE == REDQ_TARGET_AVE:
        var acc: Scalar[DT] = Scalar[DT](0.0)
        comptime for n in range(N_ENSEMBLE):
            var v = q_next.ptr[unsafe_offset=n * BATCH + b]
            if v != v:
                v = Scalar[DT](0.0)
            acc = acc + v
        combined = acc / Scalar[DT](N_ENSEMBLE)

    var lp = log_probs.ptr[unsafe_offset=b]
    var nonterm = Scalar[DT](1.0) - terms.ptr[unsafe_offset=b]
    var tgt = rewards.ptr[unsafe_offset=b] + nonterm * gamma * (combined - alpha * lp)
    if tgt != tgt:
        tgt = Scalar[DT](0.0)
    out_y.ptr[unsafe_offset=b] = tgt


# =============================================================================
# Device-alpha kernel — CUDA-graph-capturable variant. Reads alpha from
# `alpha_ptr[0]` (REDQ's on-device temperature buffer, ScalarAdam.alpha_dev_ptr)
# instead of a baked scalar arg, so the combine stays valid across graph replays
# while the device ScalarAdam refreshes alpha. Template:
# `sac/target_y_block.mojo::_target_y_dev_kernel`.
# =============================================================================


def _redq_ensemble_target_dev_kernel[
    N_ENSEMBLE: Int,
    N_MIN: Int,
    MODE: Int,
    BATCH: Int,
](
    out_y: Tile[DT, BATCH],
    rewards: Tile[DT, BATCH],
    q_next: Tile[DT, N_ENSEMBLE, BATCH],
    terms: Tile[DT, BATCH],
    log_probs: Tile[DT, BATCH],
    subset_idxs: Tile[DType.uint32, N_MIN],
    gamma: Scalar[DT],
    alpha_ptr: Pointer[Scalar[DT], MutAnyOrigin],
):
    """As `_redq_ensemble_target_kernel` but α read from `alpha_ptr[0]`."""
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= BATCH:
        return

    var combined: Scalar[DT] = Scalar[DT](0.0)
    comptime if MODE == REDQ_TARGET_MIN:
        var first_idx = Int(subset_idxs.ptr[unsafe_offset=0])
        var v0 = q_next.ptr[unsafe_offset=first_idx * BATCH + b]
        if v0 != v0:
            v0 = Scalar[DT](0.0)
        combined = v0
        comptime for m in range(1, N_MIN):
            var idx = Int(subset_idxs.ptr[unsafe_offset=m])
            var v = q_next.ptr[unsafe_offset=idx * BATCH + b]
            if v != v:
                v = Scalar[DT](0.0)
            if v < combined:
                combined = v
    comptime if MODE == REDQ_TARGET_AVE:
        var acc: Scalar[DT] = Scalar[DT](0.0)
        comptime for n in range(N_ENSEMBLE):
            var v = q_next.ptr[unsafe_offset=n * BATCH + b]
            if v != v:
                v = Scalar[DT](0.0)
            acc = acc + v
        combined = acc / Scalar[DT](N_ENSEMBLE)

    var lp = log_probs.ptr[unsafe_offset=b]
    var nonterm = Scalar[DT](1.0) - terms.ptr[unsafe_offset=b]
    var tgt = rewards.ptr[unsafe_offset=b] + nonterm * gamma * (combined - alpha_ptr[unsafe_offset=0] * lp)
    if tgt != tgt:
        tgt = Scalar[DT](0.0)
    out_y.ptr[unsafe_offset=b] = tgt


def redq_ensemble_target_gpu_dev[
    N_ENSEMBLE: Int,
    N_MIN: Int,
    MODE: Int,
    BATCH: Int,
](
    ctx: DeviceContext,
    mut out_y: Tensor,          # [BATCH]
    mut rewards: Tensor,        # [BATCH]
    mut q_next: Tensor,         # [N_ENSEMBLE, BATCH]
    mut terms: Tensor,          # [BATCH]
    mut log_probs: Tensor,      # [BATCH]
    subset_dev: Tile[DType.uint32, N_MIN],
    gamma: Scalar[DT],
    alpha_ptr: Pointer[Scalar[DT], MutAnyOrigin],
) raises:
    """Device-alpha variant of `redq_ensemble_target_gpu` — alpha read from
    `alpha_ptr[0]` (REDQ's on-device temperature). CUDA-graph capturable."""
    comptime assert (
        MODE == REDQ_TARGET_MIN or MODE == REDQ_TARGET_AVE
    ), "redq_ensemble_target_gpu_dev: only MIN/AVE supported"
    comptime assert N_MIN >= 1, "N_MIN must be >= 1"
    comptime assert N_MIN <= N_ENSEMBLE, "N_MIN must be <= N_ENSEMBLE"

    comptime lb = RowMajor[BATCH]
    comptime lq = RowMajor[N_ENSEMBLE, BATCH]
    comptime n_blocks = (BATCH + TPB - 1) // TPB
    comptime kernel = _redq_ensemble_target_dev_kernel[
        N_ENSEMBLE, N_MIN, MODE, BATCH,
    ]
    ctx.enqueue_function[kernel](
        out_y.tt["gpu", BATCH](),
        rewards.tt["gpu", BATCH](),
        q_next.tt["gpu", N_ENSEMBLE, BATCH](),
        terms.tt["gpu", BATCH](),
        log_probs.tt["gpu", BATCH](),
        subset_dev,
        gamma,
        alpha_ptr,
        grid_dim=n_blocks,
        block_dim=TPB,
    )


def redq_ensemble_target_gpu[
    N_ENSEMBLE: Int,
    N_MIN: Int,
    MODE: Int,
    BATCH: Int,
](
    ctx: DeviceContext,
    mut out_y: Tensor,          # [BATCH]
    mut rewards: Tensor,        # [BATCH]
    mut q_next: Tensor,         # [N_ENSEMBLE, BATCH]
    mut terms: Tensor,          # [BATCH]
    mut log_probs: Tensor,      # [BATCH]
    subset_dev: Tile[DType.uint32, N_MIN],
    gamma: Scalar[DT],
    alpha: Scalar[DT],
) raises:
    """Host launcher for the REDQ ensemble-target kernel over storage `Tensor`
    device buffers. `subset_dev` is a length-N_MIN uint32 device view (uploaded
    by the target_y block once per train_step)."""
    comptime assert (
        MODE == REDQ_TARGET_MIN or MODE == REDQ_TARGET_AVE
    ), (
        "redq_ensemble_target_gpu: only MIN/AVE supported (REM not"
        " ported; see kernels.mojo docstring)"
    )
    comptime assert N_MIN >= 1, "N_MIN must be >= 1"
    comptime assert N_MIN <= N_ENSEMBLE, "N_MIN must be <= N_ENSEMBLE"

    comptime lb = RowMajor[BATCH]
    comptime lq = RowMajor[N_ENSEMBLE, BATCH]
    comptime n_blocks = (BATCH + TPB - 1) // TPB
    comptime kernel = _redq_ensemble_target_kernel[
        N_ENSEMBLE, N_MIN, MODE, BATCH,
    ]
    ctx.enqueue_function[kernel](
        out_y.tt["gpu", BATCH](),
        rewards.tt["gpu", BATCH](),
        q_next.tt["gpu", N_ENSEMBLE, BATCH](),
        terms.tt["gpu", BATCH](),
        log_probs.tt["gpu", BATCH](),
        subset_dev,
        gamma,
        alpha,
        grid_dim=n_blocks,
        block_dim=TPB,
    )
