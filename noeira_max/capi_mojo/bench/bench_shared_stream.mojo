"""The stream experiment: a "Mojo kernel -> MAX model -> Mojo kernel"
loop on one CUDA stream, against the same loop on two streams with the host
synchronisations a correct hand-off needs today.

MAX's C API does not expose its stream, so this is a deliberate hack, to
measure what a supported way to share it would buy:
- noeira's CUDA interposer (`noeira/cuda/cuda_intercept.c`, which `pixi run`
  preloads on Linux) records the stream of every kernel launch. Read right
  after a MAX execute, it is MAX's stream;
- `DeviceContext.create_external_stream` wraps that handle, and the Mojo
  kernels are enqueued on it.

The model is the actor MLP of `make_mlp_mefs.py` (actor-b1, 17 -> 256 -> 256
-> 6), captured once and replayed. Iteration i: the producer writes the MLP's
input from i into a Mojo buffer lent to MAX, MAX replays, and the consumer
copies MAX's output into row i of a history. Four variants:
- A, two streams: producer, host sync, replay, host sync, consumer, host sync
  (today's correct protocol: three synchronisations per iteration);
- B, one stream: producer, replay, consumer, one host sync;
- C, one stream: no synchronisation until the end (pipelined);
- D, two streams, no synchronisation: the race the syncs prevent.
B, C and D's histories are compared with A's, element for element.

    noeira_max/capi_mojo/bench/run.sh --stream
"""

from std.ffi import OwnedDLHandle
from std.os import getenv
from std.sys import argv
from std.time import perf_counter_ns

from max.gpu import global_idx
from max.gpu.host import DeviceBuffer, DeviceContext, DeviceStream, HostBuffer
from max.gpu.host._nvidia_cuda import CUDA

from maxrt import Runtime, Tensor
from maxrt.tensor import typed

comptime IN = 17
comptime OUT = 6
comptime WARMUP = 100
comptime CALLS = 1000
comptime KEY: UInt64 = 1


def produce(x: MutPointer[Float32, MutAnyOrigin], step: Int32, n: Int32):
    """The MLP's input for iteration `step`: x[j] = (step + 1) * 0.001 * (j + 1)."""
    var j = Int(global_idx.x)
    if j < Int(n):
        x[unsafe_offset=j] = Float32(Int(step) + 1) * Float32(0.001) * Float32(j + 1)


def consume(
    y: ImmPointer[Float32, ImmutAnyOrigin],
    hist: MutPointer[Float32, MutAnyOrigin],
    step: Int32,
    n: Int32,
):
    """Row `step` of the history: MAX's output, as this iteration saw it."""
    var k = Int(global_idx.x)
    if k < Int(n):
        hist[unsafe_offset=Int(step) * Int(n) + k] = y[unsafe_offset=k]


def interposer() raises -> OwnedDLHandle:
    var path = getenv("LD_PRELOAD")
    if "libcuda_intercept.so" not in path:
        raise Error(
            "noeira's CUDA interposer is not preloaded (LD_PRELOAD='" + path
            + "'): run under `pixi run`"
        )
    return OwnedDLHandle(path)


def recorded_stream(lib: OwnedDLHandle) raises -> Int:
    """The stream of the last kernel launch in this process, as the
    interposer saw it (0 once that stream is destroyed)."""
    return lib.get_function[Int]("intercept_get_mojo_stream")()


def median_us(mut samples: List[Int]) -> Float64:
    sort(samples)
    return Float64(samples[len(samples) // 2]) / 1000.0


def read_history(
    ctx: DeviceContext, hist: DeviceBuffer[DType.float32], host: HostBuffer[DType.float32]
) raises -> List[Float32]:
    ctx.enqueue_copy(host, hist)
    ctx.synchronize()
    var out = List[Float32](capacity=CALLS * OUT)
    for i in range(CALLS * OUT):
        out.append(host[i])
    return out^


def mismatches(got: List[Float32], want: List[Float32]) -> Int:
    var n = 0
    for i in range(len(want)):
        if got[i] != want[i]:
            n += 1
    return n


def main() raises:
    var dir = String(argv()[1])
    var lib = interposer()
    var rt = Runtime(accelerator=True)
    var ctx = DeviceContext()
    var model = rt.load(dir + "/actor-b1.mef")
    var prod = ctx.compile_function[produce]()
    var cons = ctx.compile_function[consume]()
    var x_buf = ctx.enqueue_create_buffer[DType.float32](IN)
    var hist = ctx.enqueue_create_buffer[DType.float32](CALLS * OUT)
    var host_hist = ctx.enqueue_create_host_buffer[DType.float32](CALLS * OUT)
    ctx.synchronize()
    var inputs = rt.tensor_map()
    inputs.borrow_address("input0", Int(x_buf.unsafe_ptr()), DType.float32, [1, IN], on_device=True)

    # 1. Whose stream is whose. A Mojo launch, then a MAX execute, each read
    # back through the interposer; then whether MAX keeps its stream across
    # synchronisations and executes.
    var mojo_stream = ctx.stream()
    var mojo_native = Int(CUDA(mojo_stream).value())
    mojo_stream.enqueue_function(prod, x_buf, Int32(0), Int32(IN), grid_dim=1, block_dim=32)
    ctx.synchronize()
    var after_mojo = recorded_stream(lib)
    _ = model.execute(inputs)
    rt.synchronize()
    var max_native = recorded_stream(lib)
    print("Mojo's stream (DeviceContext):", hex(mojo_native))
    print("  interposer, after a Mojo launch:", hex(after_mojo))
    print("  interposer, after a MAX execute:", hex(max_native))
    if after_mojo != mojo_native:
        raise Error("the interposer did not see Mojo's launch on Mojo's stream")
    if max_native == 0 or max_native == mojo_native:
        raise Error("no separate MAX stream seen")
    var changed = 0
    for _ in range(20):
        ctx.synchronize()
        rt.synchronize()
        _ = model.execute(inputs)
        rt.synchronize()
        if recorded_stream(lib) != max_native:
            changed += 1
    print("  MAX's stream after 20 more executes and synchronisations: changed", changed, "times")
    var max_stream = ctx.create_external_stream(
        Optional(Pointer[NoneType, UntrackedOrigin[mut=True]](unsafe_from_address=max_native))
    )

    # 2. Capture the model once; its output buffer is then fixed.
    var tensors = List[Tensor]()
    tensors.append(inputs.tensor("input0"))
    var captured = model.capture(KEY, tensors)
    var y_buf = DeviceBuffer[DType.float32](ctx, typed[Float32](captured[0].address()), OUT, owning=False)

    # 3. MAX alone in this process (under the interposer), for reference.
    var samples = List[Int]()
    for i in range(WARMUP + CALLS):
        var t = perf_counter_ns()
        model.replay(KEY, tensors)
        rt.synchronize()
        if i >= WARMUP:
            samples.append(Int(perf_counter_ns() - t))
    var replay_us = median_us(samples)

    # A: two streams, a host synchronisation at each hand-off.
    samples.clear()
    for i in range(WARMUP + CALLS):
        var step = Int32(i - WARMUP if i >= WARMUP else i)
        var t = perf_counter_ns()
        mojo_stream.enqueue_function(prod, x_buf, step, Int32(IN), grid_dim=1, block_dim=32)
        ctx.synchronize()
        model.replay(KEY, tensors)
        rt.synchronize()
        mojo_stream.enqueue_function(cons, y_buf, hist, step, Int32(OUT), grid_dim=1, block_dim=32)
        ctx.synchronize()
        if i >= WARMUP:
            samples.append(Int(perf_counter_ns() - t))
    var a_us = median_us(samples)
    var want = read_history(ctx, hist, host_hist)
    var distinct = 0
    for i in range(1, CALLS):
        if want[i * OUT] != want[(i - 1) * OUT]:
            distinct += 1

    # B: one stream, one host synchronisation per iteration.
    ctx.enqueue_memset(hist, Float32(0))
    ctx.synchronize()
    samples.clear()
    for i in range(WARMUP + CALLS):
        var step = Int32(i - WARMUP if i >= WARMUP else i)
        var t = perf_counter_ns()
        max_stream.enqueue_function(prod, x_buf, step, Int32(IN), grid_dim=1, block_dim=32)
        model.replay(KEY, tensors)
        max_stream.enqueue_function(cons, y_buf, hist, step, Int32(OUT), grid_dim=1, block_dim=32)
        max_stream.synchronize()
        if i >= WARMUP:
            samples.append(Int(perf_counter_ns() - t))
    var b_us = median_us(samples)
    var b_bad = mismatches(read_history(ctx, hist, host_hist), want)

    # C: one stream, no synchronisation until the end.
    ctx.enqueue_memset(hist, Float32(0))
    ctx.synchronize()
    var start = perf_counter_ns()
    for i in range(CALLS):
        max_stream.enqueue_function(prod, x_buf, Int32(i), Int32(IN), grid_dim=1, block_dim=32)
        model.replay(KEY, tensors)
        max_stream.enqueue_function(cons, y_buf, hist, Int32(i), Int32(OUT), grid_dim=1, block_dim=32)
    max_stream.synchronize()
    var c_us = Float64(Int(perf_counter_ns() - start)) / 1000.0 / Float64(CALLS)
    var c_bad = mismatches(read_history(ctx, hist, host_hist), want)

    # D: two streams, no synchronisation: the race.
    ctx.enqueue_memset(hist, Float32(0))
    ctx.synchronize()
    start = perf_counter_ns()
    for i in range(CALLS):
        mojo_stream.enqueue_function(prod, x_buf, Int32(i), Int32(IN), grid_dim=1, block_dim=32)
        model.replay(KEY, tensors)
        mojo_stream.enqueue_function(cons, y_buf, hist, Int32(i), Int32(OUT), grid_dim=1, block_dim=32)
    ctx.synchronize()
    rt.synchronize()
    var d_us = Float64(Int(perf_counter_ns() - start)) / 1000.0 / Float64(CALLS)
    var d_bad = mismatches(read_history(ctx, hist, host_hist), want)

    _ = y_buf^  # views MAX's output, which `captured` owns
    _ = captured^
    _ = x_buf^  # lent to MAX by address until here
    print(
        "RESULT {\"max_replay_synced_us\": " + String(replay_us)
        + ", \"a_two_streams_3_syncs_us\": " + String(a_us)
        + ", \"b_one_stream_1_sync_us\": " + String(b_us)
        + ", \"c_one_stream_pipelined_us\": " + String(c_us)
        + ", \"d_two_streams_no_sync_us\": " + String(d_us)
        + ", \"history_rows_that_differ_from_the_previous\": " + String(distinct)
        + ", \"b_mismatches\": " + String(b_bad) + ", \"c_mismatches\": " + String(c_bad)
        + ", \"d_mismatches\": " + String(d_bad) + ", \"of\": " + String(CALLS * OUT)
        + ", \"max_stream_changes\": " + String(changed) + "}"
    )
