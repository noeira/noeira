"""Running observation normalization (env-side, opt-in).

Mirrors CleanRL's `VecNormalize` semantics:
  - Running mean / variance / count tracker updated from every env step.
  - Normalization: o ← (o − μ) / √(σ² + ε)
  - Updates can be frozen (e.g., for eval); apply still uses the frozen stats.

Stats live in GPU device buffers during training; the host mirror is
synced lazily for CPU-side select_action paths.

Opt-in: env GPU kernels apply normalization only when the driver passes
an `ObsNormStats` handle. Default behavior is unchanged across all
existing training scripts.

The `count_prior` matches the EfficientZero V2 reference
(`references/EfficientZeroV2-main/ez/agents/ez_dmc_state.py:112`):
initialize the running count to 1e3 so the first batches don't move the
running stats by orders of magnitude.
"""

from std.collections import Array
from std.math import sqrt
from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.nn.constants import DT as gpu_dtype
from noeira.core.tiles import Tile, ImmTile, tile


# =============================================================================
# GPU kernels
# =============================================================================


def update_obs_norm_kernel[
    BATCH: Int,
    OBS_DIM: Int,
](
    obs: Tile[gpu_dtype, BATCH, OBS_DIM],
    mean: Tile[gpu_dtype, OBS_DIM],
    var_: Tile[gpu_dtype, OBS_DIM],
    count: Tile[gpu_dtype, 1],
):
    """One thread per obs dim; merges this step's batch into running stats.

    Uses the Chan et al. parallel algorithm so accumulation stays numerically
    stable across millions of steps. `count` is global and shared across
    dims — only thread 0 writes it back. Other threads only read it for the
    per-dim mean/var merge, so the read is consistent pre-write.

    All math is in `gpu_dtype` (Float32) — Metal can't compile Float64. This
    matches the precision regime of CleanRL VecNormalize and the EZ-V2
    reference (`ez_dmc_state.py` RunningMeanStd) which both use Float32.
    """
    var d = Int(block_dim.x * block_idx.x + thread_idx.x)
    if d >= OBS_DIM:
        return

    # Pass 1: batch mean over BATCH envs.
    var batch_mean = Scalar[gpu_dtype](0.0)
    for e in range(BATCH):
        batch_mean += rebind[Scalar[gpu_dtype]](obs[e, d])
    batch_mean /= Scalar[gpu_dtype](BATCH)

    # Pass 2: batch M2 (sum of squared deviations from batch mean).
    var batch_m2 = Scalar[gpu_dtype](0.0)
    for e in range(BATCH):
        var diff = rebind[Scalar[gpu_dtype]](obs[e, d]) - batch_mean
        batch_m2 += diff * diff

    # Merge with running stats.
    var old_mean = rebind[Scalar[gpu_dtype]](mean[d])
    var old_var = rebind[Scalar[gpu_dtype]](var_[d])
    var old_count = rebind[Scalar[gpu_dtype]](count[0])
    var n = Scalar[gpu_dtype](BATCH)
    var new_count = old_count + n

    var delta = batch_mean - old_mean
    var new_mean = old_mean + delta * n / new_count

    var m2_old = old_var * old_count
    var m2_new = m2_old + batch_m2 + delta * delta * old_count * n / new_count
    var new_var = m2_new / new_count

    mean[d] = new_mean
    var_[d] = new_var
    if d == 0:
        count[0] = new_count


def apply_obs_norm_kernel[
    BATCH: Int,
    OBS_DIM: Int,
](
    obs: Tile[gpu_dtype, BATCH, OBS_DIM],
    mean: ImmTile[gpu_dtype, OBS_DIM],
    var_: ImmTile[gpu_dtype, OBS_DIM],
    eps: Scalar[gpu_dtype],
):
    """One thread per (env, dim); writes obs ← (o − μ) / √(σ² + ε) in place.

    `mean`/`var_` are read-only here (declared `ImmutAnyOrigin`) so the
    apply path can build their views from a non-`mut` `self` without the
    deprecated Pointer->MutAnyOrigin laundering.
    """
    var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
    if idx >= BATCH * OBS_DIM:
        return
    var e = idx // OBS_DIM
    var d = idx % OBS_DIM

    var m = rebind[Scalar[gpu_dtype]](mean[d])
    var v = rebind[Scalar[gpu_dtype]](var_[d])
    var o = rebind[Scalar[gpu_dtype]](obs[e, d])
    var inv_std = Scalar[gpu_dtype](1.0) / sqrt(v + eps)
    obs[e, d] = (o - m) * inv_std


# =============================================================================
# ObsNormStats — runtime container
# =============================================================================


comptime _TPB_UPDATE = 32
comptime _TPB_APPLY = 64


struct ObsNormStats[OBS_DIM: Int](Movable):
    """Per-dim running mean/var + scalar count tracker, GPU-resident.

    Created once per training run by the driver, passed by reference to
    the env's GPU step kernels. CPU mirror is refreshed on demand (eval,
    CPU select_action). Initial state: mean=0, var=1, count=`count_prior`.

    Parameters:
        OBS_DIM: Observation dimension (compile-time).
    """

    var mean_buf: DeviceBuffer[gpu_dtype]
    var var_buf: DeviceBuffer[gpu_dtype]
    var count_buf: DeviceBuffer[gpu_dtype]

    var host_mean: Array[Float64, Self.OBS_DIM]
    var host_var: Array[Float64, Self.OBS_DIM]
    var host_count: Float64

    var frozen: Bool
    var eps: Float64

    def __init__(
        out self,
        ctx: DeviceContext,
        eps: Float64 = 1e-8,
        count_prior: Float64 = 1e3,
    ) raises:
        self.mean_buf = ctx.enqueue_create_buffer[gpu_dtype](Self.OBS_DIM)
        self.var_buf = ctx.enqueue_create_buffer[gpu_dtype](Self.OBS_DIM)
        self.count_buf = ctx.enqueue_create_buffer[gpu_dtype](1)

        # Mean → 0 via unsafe_memset; var → 1 and count → count_prior require a
        # host-side init (unsafe_memset can't set a non-zero float pattern).
        ctx.enqueue_memset(self.mean_buf, 0)

        var var_host = ctx.enqueue_create_host_buffer[gpu_dtype](Self.OBS_DIM)
        for d in range(Self.OBS_DIM):
            var_host[d] = Scalar[gpu_dtype](1.0)
        ctx.enqueue_copy(self.var_buf, var_host)

        var count_host = ctx.enqueue_create_host_buffer[gpu_dtype](1)
        count_host[0] = Scalar[gpu_dtype](count_prior)
        ctx.enqueue_copy(self.count_buf, count_host)
        ctx.synchronize()

        self.host_mean = Array[Float64, Self.OBS_DIM](fill=0.0)
        self.host_var = Array[Float64, Self.OBS_DIM](fill=1.0)
        self.host_count = count_prior

        self.frozen = False
        self.eps = eps

    def update_and_apply[
        BATCH: Int
    ](
        mut self, ctx: DeviceContext, mut obs_buf: DeviceBuffer[gpu_dtype]
    ) raises:
        """Update running stats from `obs_buf` (BATCH × OBS_DIM), then
        normalize `obs_buf` in place. Skips the update if frozen.
        """
        if not self.frozen:
            self._update[BATCH](ctx, obs_buf)
        self._apply[BATCH](ctx, obs_buf)

    def apply_only[
        BATCH: Int
    ](self, ctx: DeviceContext, mut obs_buf: DeviceBuffer[gpu_dtype]) raises:
        """Apply normalization in place without touching running stats.

        Use after `reset_kernel_gpu` / `selective_reset_kernel_gpu` when
        you don't want post-reset obs to perturb stats — `update_and_apply`
        in the regular step covers ongoing learning.
        """
        self._apply[BATCH](ctx, obs_buf)

    def _update[
        BATCH: Int
    ](
        mut self, ctx: DeviceContext, mut obs_buf: DeviceBuffer[gpu_dtype]
    ) raises:
        var obs = tile[BATCH, Self.OBS_DIM](obs_buf)
        var mean_t = tile[Self.OBS_DIM](self.mean_buf)
        var var_t = tile[Self.OBS_DIM](self.var_buf)
        var count_t = tile[1](self.count_buf)

        comptime kernel = update_obs_norm_kernel[BATCH, Self.OBS_DIM]
        comptime BLOCKS = (Self.OBS_DIM + _TPB_UPDATE - 1) // _TPB_UPDATE
        ctx.enqueue_function[kernel](
            obs,
            mean_t,
            var_t,
            count_t,
            grid_dim=(BLOCKS,),
            block_dim=(_TPB_UPDATE,),
        )

    def _apply[
        BATCH: Int
    ](self, ctx: DeviceContext, mut obs_buf: DeviceBuffer[gpu_dtype]) raises:
        # `obs_buf` is `mut` (apply writes in place) -> obs view is mut=True.
        # `self` is non-`mut`, so mean/var views are mut=False; the apply
        # kernel reads them (ImmutAnyOrigin params), so this matches without
        # the deprecated immutable->mutable laundering.
        var obs = tile[BATCH, Self.OBS_DIM](obs_buf)
        var mean_t = tile[Self.OBS_DIM](self.mean_buf)
        var var_t = tile[Self.OBS_DIM](self.var_buf)

        comptime kernel = apply_obs_norm_kernel[BATCH, Self.OBS_DIM]
        comptime TOTAL = BATCH * Self.OBS_DIM
        comptime BLOCKS = (TOTAL + _TPB_APPLY - 1) // _TPB_APPLY
        ctx.enqueue_function[kernel](
            obs,
            mean_t,
            var_t,
            Scalar[gpu_dtype](self.eps),
            grid_dim=(BLOCKS,),
            block_dim=(_TPB_APPLY,),
        )

    def freeze(mut self):
        self.frozen = True

    def unfreeze(mut self):
        self.frozen = False

    def sync_host(mut self, ctx: DeviceContext) raises:
        """Pull device stats back into the host mirror.

        Call before saving a checkpoint or before using `apply_cpu` for a
        CPU-side rollout that needs the latest stats.
        """
        var mean_h = ctx.enqueue_create_host_buffer[gpu_dtype](Self.OBS_DIM)
        var var_h = ctx.enqueue_create_host_buffer[gpu_dtype](Self.OBS_DIM)
        var count_h = ctx.enqueue_create_host_buffer[gpu_dtype](1)
        ctx.enqueue_copy(mean_h, self.mean_buf)
        ctx.enqueue_copy(var_h, self.var_buf)
        ctx.enqueue_copy(count_h, self.count_buf)
        ctx.synchronize()
        for d in range(Self.OBS_DIM):
            self.host_mean[d] = Float64(mean_h[d])
            self.host_var[d] = Float64(var_h[d])
        self.host_count = Float64(count_h[0])

    def apply_cpu(
        self, raw: List[Scalar[gpu_dtype]]
    ) -> List[Scalar[gpu_dtype]]:
        """Normalize a single host-side obs vector using the host mirror.

        Caller is responsible for `sync_host` first if device stats have
        moved since the last sync.
        """
        var out = List[Scalar[gpu_dtype]](capacity=Self.OBS_DIM)
        for d in range(Self.OBS_DIM):
            var diff = Float64(raw[d]) - self.host_mean[d]
            var inv_std = 1.0 / sqrt(self.host_var[d] + self.eps)
            out.append(Scalar[gpu_dtype](diff * inv_std))
        return out^
