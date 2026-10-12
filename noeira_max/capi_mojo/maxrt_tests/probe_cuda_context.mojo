"""Does a Mojo `DeviceContext` share the CUDA context with the MAX runtime,
so that device memory from one is valid in the other? (CUDA only)

Asks the driver which context is current and which one each pointer belongs
to, then moves data both ways: MAX reads a buffer Mojo allocated and wrote,
and Mojo reads the output MAX computed from it. Raises if any value read back
is wrong. (Pool memory reports a null context: the addresses, one allocator's
neighbours, say more.)

    built and run by maxrt_tests/run.sh on a machine with an NVIDIA GPU
"""

from std.ffi import external_call
from std.sys import argv

from max.gpu.host import DeviceBuffer, DeviceContext
from maxrt import HostBuffer, Runtime
from maxrt.tensor import typed

comptime CU_POINTER_ATTRIBUTE_CONTEXT: Int32 = 1
comptime N = 8


def pointer_context(address: Int) raises -> Int:
    """The CUcontext the driver says owns `address`."""
    var cell = HostBuffer(8)
    var rc = external_call["cuPointerGetAttribute", Int32](
        cell.address(), CU_POINTER_ATTRIBUTE_CONTEXT, UInt64(address)
    )
    if rc != 0:
        raise Error("cuPointerGetAttribute: CUresult " + String(rc))
    return cell.data[Int]()[unsafe_offset=0]


def current_context() raises -> Int:
    var cell = HostBuffer(8)
    var rc = external_call["cuCtxGetCurrent", Int32](cell.address())
    if rc != 0:
        raise Error("cuCtxGetCurrent: CUresult " + String(rc))
    return cell.data[Int]()[unsafe_offset=0]


def values(p: Pointer[Float32, _]) -> String:
    """`p` comes from its owner's `data()`: read through a raw address, an
    owner past its last use is already freed (`maxrt.tensor`, "Lifetimes")."""
    var out = String("")
    for i in range(N):
        out += String(p[unsafe_offset=i]) + " "
    return out


def expect_multiples(p: Pointer[Float32, _], k: Int, what: String) raises:
    for i in range(N):
        if p[unsafe_offset=i] != Float32(k * i):
            raise Error(what + ": element " + String(i) + " is " + String(p[unsafe_offset=i])
                        + ", not " + String(k * i))


def main() raises:
    var dir = String(argv()[1])
    var rt = Runtime(accelerator=True)
    print("current context after the MAX runtime:", current_context())
    var ctx = DeviceContext()
    print("current context after a Mojo DeviceContext:", current_context())

    var host = HostBuffer(4 * N)
    for i in range(N):
        host.data[Float32]()[unsafe_offset=i] = Float32(i)
    var x_buf = ctx.enqueue_create_buffer[DType.float32](N)
    ctx.enqueue_copy(x_buf, host.data[Float32]())
    ctx.synchronize()
    var x = Int(x_buf.unsafe_ptr())
    print("Mojo buffer", x, "in context", pointer_context(x))

    var host_side = rt.tensor_map()
    host_side.borrow("x", HostBuffer(copy_from=host.address(), nbytes=4 * N), DType.float32, [N])
    var max_copy = host_side.tensor("x").to_device(rt.device())
    print("MAX device copy", max_copy.address(), "in context", pointer_context(max_copy.address()))

    # 1. MAX reads the Mojo buffer: (x + x) * 2.
    var model = rt.load(dir + "/add_gpu.mef")
    var inputs = rt.tensor_map()
    inputs.borrow_address("input0", x, DType.float32, [N], on_device=True)
    inputs.borrow_address("input1", x, DType.float32, [N], on_device=True)
    var seen = inputs.tensor("input0").to_host()
    print("the Mojo buffer, copied to the host by MAX:", values(seen.data[Float32]()))
    expect_multiples(seen.data[Float32](), 1, "MAX reading the Mojo buffer")
    var y = model.execute(inputs).tensor("output0")
    rt.synchronize()
    print("MAX output", y.address(), "in context", pointer_context(y.address()))
    var y_host = y.to_host()
    print("  copied to the host by MAX: ", values(y_host.data[Float32]()))
    expect_multiples(y_host.data[Float32](), 4, "MAX's output, read by MAX")

    # 2. Mojo reads MAX's output.
    var y_buf = DeviceBuffer[DType.float32](ctx, typed[Float32](y.address()), N, owning=False)
    var back = HostBuffer(4 * N)
    ctx.enqueue_copy(back.data[Float32](), y_buf)
    ctx.synchronize()
    _ = y_buf^  # the copy's source, until it is done
    print("  copied to the host by Mojo:", values(back.data[Float32]()))
    expect_multiples(back.data[Float32](), 4, "MAX's output, read by Mojo")
    print("PASS the Mojo <-> MAX device round trip")
    _ = y^
    _ = max_copy^
    _ = x_buf^
