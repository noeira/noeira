"""Columns (c)-(e) of the MLP table in `noeira_max/README.md`: the MEFs of
`make_mlp_mefs.py`, run on the accelerator through `maxrt`.

- (c) host input: lend it, `M_copyTensorToDevice`, execute, copy the output
  back to the host; per call, so every call is synchronous.
- (d) the input in a Mojo `DeviceBuffer`, lent to MAX by address once; per
  call `ctx.synchronize()` (the producer is done), execute,
  `M_synchronizeDevice` (the output is ready for a consumer).
- (e) (d) with the execution captured once and replayed.

(d) and (e) are also timed pipelined: `CALLS` calls back to back and one
synchronisation, like the Python-side "MAX raw compute" (a) and the nn
baseline (f). Synchronised columns are the median of `CALLS` timed calls.

Every column's output is checked against the one MAX computed in Python. For
(d), the input is written and the output read back by Mojo's own
`DeviceContext`: the round trip that shows the two share the CUDA context.

    noeira_max/capi_mojo/bench/run.sh
"""

from max.gpu.host import DeviceBuffer, DeviceContext
from std.sys import argv
from std.time import perf_counter_ns

from maxrt import HostBuffer, Runtime, Tensor
from maxrt.tensor import typed

comptime WARMUP = 100
comptime CALLS = 1000
comptime KEY: UInt64 = 1


def read_floats(path: String) raises -> HostBuffer:
    var bytes: List[UInt8]
    with open(path, "r") as f:
        bytes = f.read_bytes()
    var out = HostBuffer(copy_from=Int(bytes.unsafe_ptr()), nbytes=len(bytes))
    _ = bytes^  # alive through the copy
    return out^


def check(got: Pointer[Float32, _], expected: HostBuffer, n: Int, what: String) raises:
    """Compares `n` floats at `got` with `expected`, exactly: the same MEF on
    the same GPU computes the same bits.

    `got` comes from its owner's `data()`, so the owner lives through the
    comparison. Read through a raw `address()`, an owner past its last use is
    already freed, and the allocator's free-list pointers show up as the
    first outputs (`maxrt.tensor`, "Lifetimes")."""
    var e = expected.data[Float32]()
    for i in range(n):
        if got[unsafe_offset=i] != e[unsafe_offset=i]:
            raise Error(
                what + ": output " + String(i) + " is " + String(got[unsafe_offset=i])
                + ", MAX from Python gave " + String(e[unsafe_offset=i])
            )


def median_us(mut samples: List[Int]) -> Float64:
    sort(samples)
    return Float64(samples[len(samples) // 2]) / 1000.0


def bench_shape(
    rt: Runtime, ctx: DeviceContext, dir: String, name: String,
    in_dim: Int, out_dim: Int, batch: Int,
) raises:
    var model = rt.load(dir + "/" + name + ".mef")
    var x = read_floats(dir + "/" + name + ".in")
    var expected = read_floats(dir + "/" + name + ".out")
    var n_in = batch * in_dim
    var n_out = batch * out_dim
    var shape: List[Int] = [batch, in_dim]

    # (c) Host input, copied to the device and back on every call.
    var samples = List[Int]()
    for i in range(WARMUP + CALLS):
        var start = perf_counter_ns()
        var host_side = rt.tensor_map()
        host_side.borrow_address("input0", x.address(), DType.float32, shape, on_device=False)
        var x_dev = host_side.tensor("input0").to_device(rt.device())
        var on_device = rt.tensor_map()
        on_device.borrow_address("input0", x_dev.address(), DType.float32, shape, on_device=True)
        var y = model.execute(on_device).tensor("output0").to_host()
        var elapsed = Int(perf_counter_ns() - start)
        _ = x_dev^  # lent by address: alive through the execute
        if i >= WARMUP:
            samples.append(elapsed)
        if i == WARMUP + CALLS - 1:
            check(y.data[Float32](), expected, n_out, name + " (c)")
    var c = median_us(samples)

    # (d) A Mojo device buffer, written by Mojo, lent to MAX once.
    var x_buf = ctx.enqueue_create_buffer[DType.float32](n_in)
    ctx.enqueue_copy(x_buf, x.data[Float32]())
    ctx.synchronize()
    _ = x^  # the copy's source, until it is done
    var inputs = rt.tensor_map()
    inputs.borrow_address("input0", Int(x_buf.unsafe_ptr()), DType.float32, shape, on_device=True)
    samples.clear()
    for i in range(WARMUP + CALLS):
        var start = perf_counter_ns()
        ctx.synchronize()
        var outputs = model.execute(inputs)
        rt.synchronize()
        var elapsed = Int(perf_counter_ns() - start)
        if i >= WARMUP:
            samples.append(elapsed)
        if i == WARMUP + CALLS - 1:
            # Mojo reads MAX's output through its own context.
            var y = outputs.tensor("output0")
            var y_buf = DeviceBuffer[DType.float32](
                ctx, typed[Float32](y.address()), n_out, owning=False
            )
            var host_y = HostBuffer(4 * n_out)
            ctx.enqueue_copy(host_y.data[Float32](), y_buf)
            ctx.synchronize()
            _ = y_buf^  # the copy's source, until it is done
            _ = y^  # the output viewed by y_buf
            check(host_y.data[Float32](), expected, n_out, name + " (d)")
    var d = median_us(samples)
    var start = perf_counter_ns()
    for _ in range(CALLS):
        _ = model.execute(inputs)
    rt.synchronize()
    var d_pipelined = Float64(perf_counter_ns() - start) / 1000.0 / Float64(CALLS)

    # (e) (d), captured once and replayed.
    var tensors = List[Tensor]()
    tensors.append(inputs.tensor("input0"))
    var captured = model.capture(KEY, tensors)
    samples.clear()
    for i in range(WARMUP + CALLS):
        var t = perf_counter_ns()
        ctx.synchronize()
        model.replay(KEY, tensors)
        rt.synchronize()
        if i >= WARMUP:
            samples.append(Int(perf_counter_ns() - t))
    var e = median_us(samples)
    start = perf_counter_ns()
    for _ in range(CALLS):
        model.replay(KEY, tensors)
    rt.synchronize()
    var e_pipelined = Float64(perf_counter_ns() - start) / 1000.0 / Float64(CALLS)
    var replayed = captured[0].to_host()
    check(replayed.data[Float32](), expected, n_out, name + " (e)")
    _ = x_buf^  # lent by address until here

    print(
        "RESULT {\"shape\": \"" + name + "\", \"c_us\": " + String(c)
        + ", \"d_us\": " + String(d) + ", \"d_pipelined_us\": " + String(d_pipelined)
        + ", \"e_us\": " + String(e) + ", \"e_pipelined_us\": " + String(e_pipelined) + "}"
    )


def main() raises:
    var dir = String(argv()[1])
    var rt = Runtime(accelerator=True)
    var ctx = DeviceContext()
    print("MAX on", rt.device_label(), "| Mojo DeviceContext on", ctx.name())
    bench_shape(rt, ctx, dir, "actor-b1", 17, 6, 1)
    bench_shape(rt, ctx, dir, "actor-b64", 17, 6, 64)
    bench_shape(rt, ctx, dir, "actor-b1024", 17, 6, 1024)
    bench_shape(rt, ctx, dir, "wide-b1", 256, 64, 1)
    bench_shape(rt, ctx, dir, "wide-b1024", 256, 64, 1024)
