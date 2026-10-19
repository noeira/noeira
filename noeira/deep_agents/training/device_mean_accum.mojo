"""DeviceMeanAccum — device-resident running mean of a `[N]` buffer.

The off-policy trainers (SAC, MBPO) emit per-batch diagnostic means like
`mean_q` and `mean_reward`. On CPU those are summed straight off the host
`mb_*` scratches in the trainer's diag block. On GPU the same scratches live
in device memory, so summing them on the host would force a per-update D2H of
every `[BATCH]` buffer — exactly the overhead the Slice-3 critic-loss
accumulator was built to avoid.

This mirrors the `MSELoss` device-accumulator pattern (see
`noeira/nn/loss/mse.mojo`): `accumulate_gpu[N]` reduces the `[N]` buffer on
device and adds `(sum/N, +1)` into a `[2]` accumulator (`[sum_of_means,
count]`) with NO per-step D2H — so it is CUDA-graph capturable. The host reads
the running mean once per `diag_every` flush via `read`, then `reset`s the
window. The reduction order differs from the CPU left-to-right sweep (~1e-5 in
fp32), but these scalars feed only metrics — never a gradient — so training
stays bit-identical.

CPU is supported as a host-scalar mirror (`accumulate_cpu`) so the type can be
used uniformly, but the existing trainers keep their own CPU diag path and use
this only on GPU.
"""

from max.gpu import thread_idx
from max.gpu.primitives import block
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.nn.constants import DT, TPB_REDUCE


# ──────────────────────────────────────────────────────────────────────────
# GPU kernel — single-block reduce of `data[0..N]`, then thread 0 adds
# `total/N` into `acc[0]` and `+1` into `acc[1]`. Launch grid_dim=1,
# block_dim=TPB_REDUCE. Mirrors `_mse_reduce_add_kernel`.
# ──────────────────────────────────────────────────────────────────────────
from noeira.nn.core.target_storage import require_ctx
from noeira.core.tiles import Tile, tile


def _mean_reduce_add_kernel[N: Int](
    data: Pointer[Scalar[DT], MutAnyOrigin],
    acc: Pointer[Scalar[DT], MutAnyOrigin],
):
    var t = Int(thread_idx.x)
    var my_sum: Scalar[DT] = 0.0
    var k = t
    while k < N:
        my_sum += data[unsafe_offset=k]
        k += TPB_REDUCE
    var total = block.sum[block_size=TPB_REDUCE, broadcast=False](val=my_sum)
    if t == 0:
        acc[unsafe_offset=0] = acc[unsafe_offset=0] + total[0] / Scalar[DT](N)
        acc[unsafe_offset=1] = acc[unsafe_offset=1] + Scalar[DT](1.0)


def _mean_abs_reduce_add_kernel[N: Int](
    data: Pointer[Scalar[DT], MutAnyOrigin],
    acc: Pointer[Scalar[DT], MutAnyOrigin],
):
    """Same as `_mean_reduce_add_kernel` but reduces `mean(|data[k]|)`. Used
    for `mean_abs_action` (sum of absolute action components / N)."""
    var t = Int(thread_idx.x)
    var my_sum: Scalar[DT] = 0.0
    var k = t
    while k < N:
        var v = data[unsafe_offset=k]
        my_sum += v if v >= Scalar[DT](0.0) else -v
        k += TPB_REDUCE
    var total = block.sum[block_size=TPB_REDUCE, broadcast=False](val=my_sum)
    if t == 0:
        acc[unsafe_offset=0] = acc[unsafe_offset=0] + total[0] / Scalar[DT](N)
        acc[unsafe_offset=1] = acc[unsafe_offset=1] + Scalar[DT](1.0)


def _mean_abs_diff_reduce_add_kernel[N: Int](
    a: Pointer[Scalar[DT], MutAnyOrigin],
    b: Pointer[Scalar[DT], MutAnyOrigin],
    acc: Pointer[Scalar[DT], MutAnyOrigin],
):
    """Reduces `mean(|a[k] - b[k]|)` over `[N]`. Used for `mean_td_error`
    (the Bellman residual magnitude |Q − y|)."""
    var t = Int(thread_idx.x)
    var my_sum: Scalar[DT] = 0.0
    var k = t
    while k < N:
        var d = a[unsafe_offset=k] - b[unsafe_offset=k]
        my_sum += d if d >= Scalar[DT](0.0) else -d
        k += TPB_REDUCE
    var total = block.sum[block_size=TPB_REDUCE, broadcast=False](val=my_sum)
    if t == 0:
        acc[unsafe_offset=0] = acc[unsafe_offset=0] + total[0] / Scalar[DT](N)
        acc[unsafe_offset=1] = acc[unsafe_offset=1] + Scalar[DT](1.0)


# ──────────────────────────────────────────────────────────────────────────
# Storage-native variants: take `LayoutTensor` views (built from a storage
# `Tensor`'s device buffer via `lt` / direct buffer access) instead of raw
# `Pointer`s — so the storage SAC path never touches `unsafe_ptr`. Same
# reduction as the raw-ptr kernels above.
# ──────────────────────────────────────────────────────────────────────────


def _mean_reduce_add_kernel_lt[N: Int](
    data: Tile[DT, N],
    acc: Tile[DT, 2],
):
    var t = Int(thread_idx.x)
    var my_sum: Scalar[DT] = 0.0
    var k = t
    while k < N:
        my_sum += rebind[Scalar[DT]](data[k])
        k += TPB_REDUCE
    var total = block.sum[block_size=TPB_REDUCE, broadcast=False](val=my_sum)
    if t == 0:
        acc[0] = acc[0] + total[0] / Scalar[DT](N)
        acc[1] = acc[1] + Scalar[DT](1.0)


def _mean_abs_reduce_add_kernel_lt[N: Int](
    data: Tile[DT, N],
    acc: Tile[DT, 2],
):
    """LayoutTensor twin of `_mean_abs_reduce_add_kernel` (mean of |data[k]|)."""
    var t = Int(thread_idx.x)
    var my_sum: Scalar[DT] = 0.0
    var k = t
    while k < N:
        var v = rebind[Scalar[DT]](data[k])
        my_sum += v if v >= Scalar[DT](0.0) else -v
        k += TPB_REDUCE
    var total = block.sum[block_size=TPB_REDUCE, broadcast=False](val=my_sum)
    if t == 0:
        acc[0] = acc[0] + total[0] / Scalar[DT](N)
        acc[1] = acc[1] + Scalar[DT](1.0)


def _mean_abs_diff_reduce_add_kernel_lt[N: Int](
    a: Tile[DT, N],
    b: Tile[DT, N],
    acc: Tile[DT, 2],
):
    """LayoutTensor twin of `_mean_abs_diff_reduce_add_kernel` (mean of
    |a[k] − b[k]|) — for the DQN `mean_td_error` diag without `unsafe_ptr`."""
    var t = Int(thread_idx.x)
    var my_sum: Scalar[DT] = 0.0
    var k = t
    while k < N:
        var d = rebind[Scalar[DT]](a[k]) - rebind[Scalar[DT]](b[k])
        my_sum += d if d >= Scalar[DT](0.0) else -d
        k += TPB_REDUCE
    var total = block.sum[block_size=TPB_REDUCE, broadcast=False](val=my_sum)
    if t == 0:
        acc[0] = acc[0] + total[0] / Scalar[DT](N)
        acc[1] = acc[1] + Scalar[DT](1.0)


struct DeviceMeanAccum(Copyable, Movable, Deinitable):
    """Running mean of a `[N]` buffer over a flush window.

    GPU: a `[2]` device buffer `[sum_of_batch_means, count]`. CPU mirror:
    `_acc_sum` / `_acc_n` host scalars. Default-constructed instances hold no
    device buffer (used on CPU trainers, or before `make`)."""

    var acc_dev: Optional[DeviceBuffer[DT]]
    var _acc_sum: Scalar[DT]
    var _acc_n: Int
    var ctx: Optional[DeviceContext]

    def __init__(out self):
        self.acc_dev = None
        self._acc_sum = Scalar[DT](0.0)
        self._acc_n = 0
        self.ctx = None

    @staticmethod
    def make[target: StaticString](
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        """Unified CPU/GPU factory. `ctx=None` on CPU; required on GPU."""
        comptime assert target == "cpu" or target == "gpu", (
            "DeviceMeanAccum: target must be 'cpu' or 'gpu'"
        )
        var a = Self()
        comptime if target == "gpu":
            var c = require_ctx["DeviceMeanAccum.make[target='gpu']"](ctx)
            var b = c.enqueue_create_buffer[DT](2)
            b.enqueue_fill(0.0)
            a.acc_dev = b^
            a.ctx = ctx
        return a^

    def accumulate_gpu[N: Int](
        mut self,
        data_ptr: Pointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        """Reduce `data_ptr[0..N]` on device and fold `(mean, +1)` into the
        accumulator. No D2H — capture-safe. `data_ptr` must be a device
        pointer to an `[N]` buffer."""
        var ctx = self.ctx.value()
        comptime red_k = _mean_reduce_add_kernel[N]
        ctx.enqueue_function[red_k](
            data_ptr,
            self.acc_dev.value().unsafe_ptr(),
            grid_dim=1,
            block_dim=TPB_REDUCE,
        )

    def accumulate_gpu_abs[N: Int](
        mut self,
        data_ptr: Pointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        """Like `accumulate_gpu` but folds `mean(|data[k]|)` (for
        `mean_abs_action`)."""
        var ctx = self.ctx.value()
        comptime red_k = _mean_abs_reduce_add_kernel[N]
        ctx.enqueue_function[red_k](
            data_ptr,
            self.acc_dev.value().unsafe_ptr(),
            grid_dim=1,
            block_dim=TPB_REDUCE,
        )

    def set_ctx(mut self, ctx: DeviceContext):
        """Point this accumulator at `ctx` for subsequent `accumulate_gpu_*`.

        ⚠⚠ THIS EXISTS FOR DEVICE-GRAPH RECORDING AND ITS ABSENCE IS SILENT.
        `accumulate_gpu_lt` and friends take no context — they reach for
        `self.ctx`, the one stored at `make` time. Under
        `maybe_record_replay`'s recording pass that enqueues EAGERLY on the
        real stream instead of recording, and the damage is two-sided:

          * the build pass, which is supposed to execute NOTHING, folds one
            spurious batch in;
          * every REPLAY skips the fold entirely, so the diagnostics quietly
            stop advancing while training stays correct.

        Nothing raises either way. So a trainer that records its step must
        point EVERY accumulator it owns at the recording context first — see
        `sac/trainer.mojo::train_device_kernels_on` — and the fix is only
        correct if it is COMPLETE, which is what makes this a method worth
        grepping for rather than a bare field assignment.
        """
        self.ctx = ctx

    def accumulate_gpu_lt[N: Int](
        mut self,
        data: Tile[DT, N],
    ) raises:
        """Storage-native `accumulate_gpu`: reduce the `[N]` device view on
        device and fold `(mean, +1)` in. The caller builds `data` via the
        storage tensor's `lt["gpu", Layout.row_major(N)]()` — no `unsafe_ptr`.
        """
        var ctx = self.ctx.value()
        var acc = tile[2](self.acc_dev.value())
        ctx.enqueue_function[_mean_reduce_add_kernel_lt[N]](
            data, acc, grid_dim=1, block_dim=TPB_REDUCE
        )

    def accumulate_gpu_abs_lt[N: Int](
        mut self,
        data: Tile[DT, N],
    ) raises:
        """Storage-native `accumulate_gpu_abs` (mean of |data[k]|)."""
        var ctx = self.ctx.value()
        var acc = tile[2](self.acc_dev.value())
        ctx.enqueue_function[_mean_abs_reduce_add_kernel_lt[N]](
            data, acc, grid_dim=1, block_dim=TPB_REDUCE
        )

    def accumulate_gpu_abs_diff[N: Int](
        mut self,
        a_ptr: Pointer[Scalar[DT], MutAnyOrigin],
        b_ptr: Pointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        """Folds `mean(|a[k] - b[k]|)` (for `mean_td_error`)."""
        var ctx = self.ctx.value()
        comptime red_k = _mean_abs_diff_reduce_add_kernel[N]
        ctx.enqueue_function[red_k](
            a_ptr,
            b_ptr,
            self.acc_dev.value().unsafe_ptr(),
            grid_dim=1,
            block_dim=TPB_REDUCE,
        )

    def accumulate_gpu_abs_diff_lt[N: Int](
        mut self,
        a: Tile[DT, N],
        b: Tile[DT, N],
    ) raises:
        """Storage-native `accumulate_gpu_abs_diff` (mean of |a[k] − b[k]|) —
        the DQN `mean_td_error` fold built from storage `lt` views."""
        var ctx = self.ctx.value()
        var acc = tile[2](self.acc_dev.value())
        ctx.enqueue_function[_mean_abs_diff_reduce_add_kernel_lt[N]](
            a, b, acc, grid_dim=1, block_dim=TPB_REDUCE
        )

    def accumulate_cpu[N: Int](
        mut self,
        data_ptr: Pointer[Scalar[DT], MutAnyOrigin],
    ):
        """Host-scalar mirror of `accumulate_gpu` (sum/N into the running
        host accumulators)."""
        var s: Scalar[DT] = 0.0
        for i in range(N):
            s += data_ptr[unsafe_offset=i]
        self._acc_sum += s / Scalar[DT](N)
        self._acc_n += 1

    def read_into[
        target: StaticString
    ](mut self, mut mean: Scalar[DT], mut count: Int) raises:
        """`read`, plus how many folds the window holds — ONE D2H for both.

        The count matters under CUDA-graph replay: the accumulate kernel is
        inside the captured region, so it advances on every replay, while any
        host-side counter the caller keeps does NOT. A window whose `n` comes
        from the host therefore under-reports the moment a graph is replaying,
        and it under-reports SILENTLY — the mean stays right (the divisor is
        the device count either way), only the reported step count is a lie.
        Read both from the device and there is nothing to keep in sync."""
        comptime if target == "cpu":
            count = self._acc_n
            mean = (
                Scalar[DT](0.0) if self._acc_n == 0
                else self._acc_sum / Scalar[DT](self._acc_n)
            )
        else:
            var ctx = self.ctx.value()
            var h = ctx.enqueue_create_host_buffer[DT](2)
            ctx.enqueue_copy(h, self.acc_dev.value())
            ctx.synchronize()
            var s = h.unsafe_ptr()[unsafe_offset=0]
            var n = h.unsafe_ptr()[unsafe_offset=1]
            count = Int(n)
            mean = Scalar[DT](0.0) if n == Scalar[DT](0.0) else s / n

    def read[target: StaticString](mut self) raises -> Scalar[DT]:
        """Mean of the accumulated per-batch means over the window
        (`sum / count`); 0 if no updates. GPU path D2Hs the `[2]` buffer once
        — flush cadence only, NOT in the per-step hot loop."""
        comptime if target == "cpu":
            if self._acc_n == 0:
                return Scalar[DT](0.0)
            return self._acc_sum / Scalar[DT](self._acc_n)
        else:
            var ctx = self.ctx.value()
            var h = ctx.enqueue_create_host_buffer[DT](2)
            ctx.enqueue_copy(h, self.acc_dev.value())
            ctx.synchronize()
            var s = h.unsafe_ptr()[unsafe_offset=0]
            var n = h.unsafe_ptr()[unsafe_offset=1]
            if n == Scalar[DT](0.0):
                return Scalar[DT](0.0)
            return s / n

    def reset[target: StaticString](mut self) raises:
        """Zero the accumulator. Call once per `diag_every` window after
        `read`, outside any capture region."""
        comptime if target == "cpu":
            self._acc_sum = Scalar[DT](0.0)
            self._acc_n = 0
        else:
            self.acc_dev.value().enqueue_fill(0.0)
