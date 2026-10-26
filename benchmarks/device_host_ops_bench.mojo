# +--------------------------------------------------------------------------+ #
# | What each host-side device operation costs to ENQUEUE on this device
# +--------------------------------------------------------------------------+ #
"""Per-call host cost of the `DeviceContext` operations a training step uses.

    pixi run -e apple  mojo run -I . benchmarks/device_host_ops_bench.mojo
    pixi run -e nvidia mojo run -I . benchmarks/device_host_ops_bench.mojo

⚠⚠ WHY. `benchmarks/sac_update_stages_bench.mojo` (19 Sep, M1 Pro): one SAC
update is 17.5 ms of HOST time — `train_step` takes that long to return,
before any wait on the GPU — and a `sample` of the process puts the main
thread in `-[_MTLCommandBuffer waitUntilCompleted]` under MAX's runtime
bindings, called from inside the update. Some host operation is waiting for
the GPU on every call. This times each candidate alone, N calls between two
syncs, so the one that waits stands out by an order of magnitude.

Each row is `N_CALLS` back-to-back calls; the number is per call. The
"launch" row is the floor every other row is read against.
"""

from std.sys import has_accelerator
from std.time import perf_counter_ns

from max.gpu import global_idx
from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from noeira.core.tiles import RowMajor, Tile, tile

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.fill import fill_dev
from std.memory import Pointer


comptime N = 256 * 256
comptime N_CALLS = 200
comptime WARMUP = 2
comptime REPS = 5


def _touch_kernel(
    x: Tile[DT, N],
):
    var i = Int(global_idx.x)
    if i < N:
        x[i] = x[i] * Scalar[DT](1.0001) + Scalar[DT](0.5)


def _six_arg_kernel(
    x: Pointer[Scalar[DT], MutAnyOrigin],
    y: Pointer[Scalar[DT], MutAnyOrigin],
    z: Pointer[Scalar[DT], MutAnyOrigin],
    n_arg: Int64,
    a_arg: Int64,
    b_arg: Int64,
):
    var i = Int(global_idx.x)
    if i < Int(n_arg):
        y[i] = x[i] * Scalar[DT](a_arg) + z[i] * Scalar[DT](b_arg)


def _fmt(x: Float64, digits: Int) -> String:
    var scale = 1.0
    for _ in range(digits):
        scale *= 10.0
    var r = Float64(Int(x * scale + (0.5 if x >= 0.0 else -0.5))) / scale
    return String(r)


def _pad(s: String, w: Int) -> String:
    var out = s
    while out.byte_length() < w:
        out += " "
    return out


def _row(name: String, best_us: Float64):
    print("   " + _pad(name, 52) + _pad(_fmt(best_us, 1), 10) + "us/call")


def main() raises:
    comptime assert has_accelerator(), "this benchmark times device host ops"
    var ctx = DeviceContext()
    print("=" * 76)
    print("Host cost per device operation — " + String(ctx.name()))
    print("=" * 76)
    print(
        "   " + _pad(String("operation (x" + String(N_CALLS) + " between syncs)"), 52)
        + "best"
    )

    var a = Tensor.alloc(N)
    for i in range(N):
        a.data[i] = Scalar[DT](i % 7)
    a.upload(ctx)
    var b = Tensor()
    b.ensure_gpu(ctx, N)
    var small_host = ctx.enqueue_create_host_buffer[DT](4)
    var small_dev = ctx.enqueue_create_buffer[DT](4)
    ctx.synchronize()
    comptime lay = RowMajor[N]

    # 0. kernel launch — the floor
    var best = 1.0e30
    for rep in range(WARMUP + REPS):
        ctx.synchronize()
        var t0 = perf_counter_ns()
        for _ in range(N_CALLS):
            ctx.enqueue_function[_touch_kernel](
                a.tt["gpu", N](), grid_dim=(N + 255) // 256, block_dim=256
            )
        var t_enq = Float64(perf_counter_ns() - t0) / 1e3 / N_CALLS
        ctx.synchronize()
        if rep >= WARMUP and t_enq < best:
            best = t_enq
    _row(String("enqueue_function (trivial kernel, 64k elems)"), best)

    # 1. enqueue_fill
    best = 1.0e30
    for rep in range(WARMUP + REPS):
        ctx.synchronize()
        var t0 = perf_counter_ns()
        for _ in range(N_CALLS):
            b.dev.value().enqueue_fill(Scalar[DT](0))
        var t_enq = Float64(perf_counter_ns() - t0) / 1e3 / N_CALLS
        ctx.synchronize()
        if rep >= WARMUP and t_enq < best:
            best = t_enq
    _row(String("enqueue_fill (64k elems)"), best)

    # 2. device -> device copy
    best = 1.0e30
    for rep in range(WARMUP + REPS):
        ctx.synchronize()
        var t0 = perf_counter_ns()
        for _ in range(N_CALLS):
            ctx.enqueue_copy(b.dev.value(), a.dev.value())
        var t_enq = Float64(perf_counter_ns() - t0) / 1e3 / N_CALLS
        ctx.synchronize()
        if rep >= WARMUP and t_enq < best:
            best = t_enq
    _row(String("enqueue_copy device->device (64k elems)"), best)

    # 3. create + free a device buffer
    best = 1.0e30
    for rep in range(WARMUP + REPS):
        ctx.synchronize()
        var t0 = perf_counter_ns()
        for _ in range(N_CALLS):
            var tmp = ctx.enqueue_create_buffer[DT](N)
            _ = tmp
        var t_enq = Float64(perf_counter_ns() - t0) / 1e3 / N_CALLS
        ctx.synchronize()
        if rep >= WARMUP and t_enq < best:
            best = t_enq
    _row(String("enqueue_create_buffer + free (64k elems, idle)"), best)

    # 4. create, USE in a kernel, free — the buffer is in flight when freed
    best = 1.0e30
    for rep in range(WARMUP + REPS):
        ctx.synchronize()
        var t0 = perf_counter_ns()
        for _ in range(N_CALLS):
            var tmp = ctx.enqueue_create_buffer[DT](N)
            var tv = tile[N](tmp)
            ctx.enqueue_function[_touch_kernel](
                tv, grid_dim=(N + 255) // 256, block_dim=256
            )
            _ = tmp
        var t_enq = Float64(perf_counter_ns() - t0) / 1e3 / N_CALLS
        ctx.synchronize()
        if rep >= WARMUP and t_enq < best:
            best = t_enq
    _row(String("create + launch on it + free (in flight)"), best)

    # 5. Tensor.ensure_gpu on an already-sized tensor
    best = 1.0e30
    for rep in range(WARMUP + REPS):
        ctx.synchronize()
        var t0 = perf_counter_ns()
        for _ in range(N_CALLS):
            b.ensure_gpu(ctx, N)
        var t_enq = Float64(perf_counter_ns() - t0) / 1e3 / N_CALLS
        ctx.synchronize()
        if rep >= WARMUP and t_enq < best:
            best = t_enq
    _row(String("Tensor.ensure_gpu (already sized)"), best)

    # 6. small D2H copy (4 elems) into a host buffer
    best = 1.0e30
    for rep in range(WARMUP + REPS):
        ctx.synchronize()
        var t0 = perf_counter_ns()
        for _ in range(N_CALLS):
            ctx.enqueue_copy(small_host, small_dev)
        var t_enq = Float64(perf_counter_ns() - t0) / 1e3 / N_CALLS
        ctx.synchronize()
        if rep >= WARMUP and t_enq < best:
            best = t_enq
    _row(String("enqueue_copy device->host (4 elems)"), best)

    # 7. small H2D copy
    best = 1.0e30
    for rep in range(WARMUP + REPS):
        ctx.synchronize()
        var t0 = perf_counter_ns()
        for _ in range(N_CALLS):
            ctx.enqueue_copy(small_dev, small_host)
        var t_enq = Float64(perf_counter_ns() - t0) / 1e3 / N_CALLS
        ctx.synchronize()
        if rep >= WARMUP and t_enq < best:
            best = t_enq
    _row(String("enqueue_copy host->device (4 elems)"), best)

    # 8. Tensor.download (D2H of 64k elems into the tensor's host slab)
    best = 1.0e30
    for rep in range(WARMUP + REPS):
        ctx.synchronize()
        var t0 = perf_counter_ns()
        for _ in range(N_CALLS // 10):
            a.download(ctx)
        var t_enq = Float64(perf_counter_ns() - t0) / 1e3 / (N_CALLS // 10)
        ctx.synchronize()
        if rep >= WARMUP and t_enq < best:
            best = t_enq
    _row(String("Tensor.download (64k elems)"), best)

    # 8b. Tensor.upload (host List -> device, 64k elems) and a 4-elem one
    var a4 = Tensor.alloc(4)
    a4.upload(ctx)
    ctx.synchronize()
    best = 1.0e30
    for rep in range(WARMUP + REPS):
        ctx.synchronize()
        var t0 = perf_counter_ns()
        for _ in range(N_CALLS // 10):
            a.upload(ctx)
        var t_enq = Float64(perf_counter_ns() - t0) / 1e3 / (N_CALLS // 10)
        ctx.synchronize()
        if rep >= WARMUP and t_enq < best:
            best = t_enq
    _row(String("Tensor.upload (64k elems, host List -> device)"), best)
    best = 1.0e30
    for rep in range(WARMUP + REPS):
        ctx.synchronize()
        var t0 = perf_counter_ns()
        for _ in range(N_CALLS):
            a4.upload(ctx)
        var t_enq = Float64(perf_counter_ns() - t0) / 1e3 / N_CALLS
        ctx.synchronize()
        if rep >= WARMUP and t_enq < best:
            best = t_enq
    _row(String("Tensor.upload (4 elems)"), best)

    # 8c. enqueue_copy from an UNPINNED host pointer (what upload does)
    var host_list = List[Scalar[DT]](length=4, fill=Scalar[DT](1))
    best = 1.0e30
    for rep in range(WARMUP + REPS):
        ctx.synchronize()
        var t0 = perf_counter_ns()
        for _ in range(N_CALLS):
            ctx.enqueue_copy(small_dev, host_list.unsafe_ptr())
        var t_enq = Float64(perf_counter_ns() - t0) / 1e3 / N_CALLS
        ctx.synchronize()
        if rep >= WARMUP and t_enq < best:
            best = t_enq
    _row(String("enqueue_copy host UnsafePointer -> device (4 elems)"), best)

    # 8d. create_sub_buffer + device copy through it
    best = 1.0e30
    for rep in range(WARMUP + REPS):
        ctx.synchronize()
        var t0 = perf_counter_ns()
        for _ in range(N_CALLS):
            var sa = a.dev.value().create_sub_buffer[DT](0, 1024)
            var sb = b.dev.value().create_sub_buffer[DT](0, 1024)
            ctx.enqueue_copy(sb, sa)
        var t_enq = Float64(perf_counter_ns() - t0) / 1e3 / N_CALLS
        ctx.synchronize()
        if rep >= WARMUP and t_enq < best:
            best = t_enq
    _row(String("create_sub_buffer x2 + copy (1k elems)"), best)

    # 8e. host buffer create + free; event create + wait
    best = 1.0e30
    for rep in range(WARMUP + REPS):
        ctx.synchronize()
        var t0 = perf_counter_ns()
        for _ in range(N_CALLS):
            var hb = ctx.enqueue_create_host_buffer[DT](1024)
            _ = hb
        var t_enq = Float64(perf_counter_ns() - t0) / 1e3 / N_CALLS
        ctx.synchronize()
        if rep >= WARMUP and t_enq < best:
            best = t_enq
    _row(String("enqueue_create_host_buffer + free (1k elems)"), best)
    best = 1.0e30
    for rep in range(WARMUP + REPS):
        ctx.synchronize()
        var t0 = perf_counter_ns()
        for _ in range(N_CALLS):
            var ev = ctx.create_event()
            ev.synchronize()
            _ = ev
        var t_enq = Float64(perf_counter_ns() - t0) / 1e3 / N_CALLS
        ctx.synchronize()
        if rep >= WARMUP and t_enq < best:
            best = t_enq
    _row(String("create_event + event.synchronize (nothing in flight)"), best)
    # a launch followed by an event wait — the pattern `upload_resident` uses
    best = 1.0e30
    for rep in range(WARMUP + REPS):
        ctx.synchronize()
        var t0 = perf_counter_ns()
        for _ in range(N_CALLS // 10):
            ctx.enqueue_function[_touch_kernel](
                a.tt["gpu", N](), grid_dim=(N + 255) // 256, block_dim=256
            )
            var ev = ctx.create_event()
            ev.synchronize()
            _ = ev
        var t_enq = Float64(perf_counter_ns() - t0) / 1e3 / (N_CALLS // 10)
        ctx.synchronize()
        if rep >= WARMUP and t_enq < best:
            best = t_enq
    _row(String("launch + create_event + event.synchronize"), best)

    # 8f. our kernel fill, and a launch with SIX pointer/scalar args
    best = 1.0e30
    for rep in range(WARMUP + REPS):
        ctx.synchronize()
        var t0 = perf_counter_ns()
        for _ in range(N_CALLS):
            fill_dev(b.dev.value(), N, Scalar[DT](0), ctx)
        var t_enq = Float64(perf_counter_ns() - t0) / 1e3 / N_CALLS
        ctx.synchronize()
        if rep >= WARMUP and t_enq < best:
            best = t_enq
    _row(String("fill_dev (kernel fill, 64k elems)"), best)
    best = 1.0e30
    for rep in range(WARMUP + REPS):
        ctx.synchronize()
        var t0 = perf_counter_ns()
        for _ in range(N_CALLS):
            ctx.enqueue_function[_six_arg_kernel](
                a.dev.value(), b.dev.value(), a.dev.value(),
                Int64(N), Int64(1), Int64(2),
                grid_dim=(N + 255) // 256, block_dim=256,
            )
        var t_enq = Float64(perf_counter_ns() - t0) / 1e3 / N_CALLS
        ctx.synchronize()
        if rep >= WARMUP and t_enq < best:
            best = t_enq
    _row(String("enqueue_function (3 buffers + 3 Int64 args)"), best)

    # 9. synchronize alone
    best = 1.0e30
    for rep in range(WARMUP + REPS):
        ctx.synchronize()
        var t0 = perf_counter_ns()
        for _ in range(N_CALLS // 10):
            ctx.synchronize()
        var t_enq = Float64(perf_counter_ns() - t0) / 1e3 / (N_CALLS // 10)
        if rep >= WARMUP and t_enq < best:
            best = t_enq
    _row(String("synchronize (idle queue)"), best)
