"""`fill_dev` — fill a device buffer with ONE KERNEL LAUNCH, not `enqueue_fill`.

⚠⚠ WHY THIS EXISTS. On Apple Metal, `DeviceBuffer.enqueue_fill` is not an
enqueue: MAX's runtime commits the fill and WAITS for the queue
(`-[_MTLCommandBuffer waitUntilCompleted]` under `libAsyncRTMojoBindings`,
seen in a `sample` of the SAC trainer). Measured on an M1 Pro by
`benchmarks/device_host_ops_bench.mojo` (19 Sep 2026):

    enqueue_function (trivial kernel, 64k elems)     18.7 us/call
    enqueue_fill     (64k elems)                    172.3 us/call
    synchronize      (idle queue)                   176.0 us/call

A fill costs a synchronize. One SAC update zeroes ~100 buffers that way (the
compute graph's grad slots, the parameter arenas, the loss accumulators), and
`benchmarks/sac_update_stages_bench.mojo` put the update at 17.5 ms of HOST
time — `train_step` returning, before any wait — which is those fills.

This is the same fill through `enqueue_function`: a launch, ordered on the
stream like every other kernel, no wait. The value written is the same
scalar, so the result is bit-identical to `enqueue_fill`.

⚠ USE IT ON ANY PATH THAT RUNS PER STEP. A fill at `make` time is paid once
and may stay `enqueue_fill`; a fill in a `forward`, `vjp`, `step`, `reset`
or replay write is paid per step and belongs here. Grep `enqueue_fill`
before assuming a hot path is clean.
"""

from max.gpu import global_idx
from max.gpu.host import DeviceContext, DeviceBuffer


comptime FILL_TPB: Int = 256


def _fill_kernel[dt: DType](
    dst: Pointer[Scalar[dt], MutAnyOrigin],
    n_arg: Int64,
    value: Scalar[dt],
):
    var i = Int(global_idx.x)
    var n = Int(n_arg)
    if i < n:
        dst[unsafe_offset=i] = value


@always_inline
def fill_dev[dt: DType](
    buf: DeviceBuffer[dt], n: Int, value: Scalar[dt], ctx: DeviceContext
) raises:
    """`buf[0:n] = value`, as one kernel launch on `ctx`'s stream."""
    if n <= 0:
        return
    ctx.enqueue_function[_fill_kernel[dt]](
        buf, Int64(n), value,
        grid_dim=(n + FILL_TPB - 1) // FILL_TPB,
        block_dim=FILL_TPB,
    )
