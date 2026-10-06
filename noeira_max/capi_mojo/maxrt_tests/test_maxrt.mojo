"""Tests maxrt against the MEFs `make_mefs.py` writes. Build and run with
`maxrt_tests/run.sh`; prints PASS / FAIL per test and raises if any fails."""

from std.os.path import exists
from std.sys import CompilationTarget, argv

from maxrt import HostBuffer, Runtime, Tensor, accelerator_count, version


def fill(mut buffer: HostBuffer, n: Int, scale: Float32, offset: Float32):
    var p = buffer.data[Float32]()
    for i in range(n):
        p[unsafe_offset=i] = Float32(i) * scale + offset


def expect(cond: Bool, what: String) raises:
    if not cond:
        raise Error(what)


def test_version() raises:
    var v = version()
    expect(v.byte_length() > 0, "empty version")
    print("    MAX C API version", v)


def test_add_symbolic(dir: String) raises:
    """One MEF, three sizes of the symbolic dim, inputs owned by the map."""
    var rt = Runtime()
    var model = rt.load(dir + "/add.mef")
    for n in [1, 5, 1000]:
        var inputs = rt.tensor_map()
        var a = HostBuffer(4 * n)
        var b = HostBuffer(4 * n)
        fill(a, n, 1.0, 0.0)
        fill(b, n, 2.0, 1.0)
        inputs.borrow("input0", a^, DType.float32, [n])
        inputs.borrow("input1", b^, DType.float32, [n])
        var outputs = model.execute(inputs)
        var y = outputs.tensor("output0")
        expect(y.num_elements() == n, "output size")
        var p = y.data[Float32]()
        for i in range(n):
            # (i + (2i + 1)) * 2
            expect(p[unsafe_offset=i] == Float32(6 * i + 2), "add value at " + String(i))


def test_in_place_across_calls(dir: String) raises:
    """A buffer the model stores into is updated in the map's own memory."""
    var rt = Runtime()
    var model = rt.load(dir + "/inplace.mef")
    var inputs = rt.tensor_map()
    var x = HostBuffer(32)
    fill(x, 8, 0.0, 0.5)
    inputs.borrow("input0", HostBuffer(32), DType.float32, [8])
    inputs.borrow("input1", x^, DType.float32, [8])
    for _ in range(3):
        _ = model.execute(inputs)
    var outputs = model.execute(inputs)  # a fourth call
    var b = inputs.owned_data[Float32]("input0")
    for i in range(8):
        expect(b[unsafe_offset=i] == 2.0, "buffer after 4 calls")
    expect(outputs.tensor("output0").item[DType.float32]() == 16.0, "sum output")
    expect(outputs.tensor("output1").data[Float32]()[unsafe_offset=7] == 4.0, "second output")


def test_copy_is_complete(dir: String) raises:
    """`to_host()` returns a finished copy. `M_copyTensorToDevice` itself
    returns before the copy is done: read at once, up to 20% of the elements
    were stale (host, MAX 26.6), so `Tensor.to_device` synchronises."""
    var rt = Runtime()
    var model = rt.load(dir + "/add.mef")
    var n = 65536
    for trial in range(5):
        var inputs = rt.tensor_map()
        var a = HostBuffer(4 * n)
        var b = HostBuffer(4 * n)
        fill(a, n, 1.0, Float32(trial))
        fill(b, n, 0.5, 0.0)
        inputs.borrow("input0", a^, DType.float32, [n])
        inputs.borrow("input1", b^, DType.float32, [n])
        var out = model.execute(inputs).tensor("output0")
        var copy = out.to_host()
        var direct = out.data[Float32]()
        var copied = copy.data[Float32]()
        for i in range(n):
            expect(copied[unsafe_offset=i] == direct[unsafe_offset=i], "copy element " + String(i))


def test_errors(dir: String) raises:
    var rt = Runtime()
    var raised = False
    try:
        _ = rt.load(dir + "/missing.mef")
    except e:
        raised = True
        print("    load error:", e)
    expect(raised, "loading a missing MEF did not raise")

    var model = rt.load(dir + "/add.mef")
    var inputs = rt.tensor_map()
    inputs.borrow("input0", HostBuffer(16), DType.float32, [4])  # input1 missing
    raised = False
    try:
        _ = model.execute(inputs)
    except e:
        raised = True
        print("    execute error:", e)
    expect(raised, "executing without input1 did not raise")


def test_lifetimes(dir: String) raises:
    """100 runtimes made and dropped; objects outlive the runtime variable."""
    for _ in range(100):
        var model = Runtime().load(dir + "/add.mef")  # the Runtime value dies here
        var rt = Runtime()
        var inputs = rt.tensor_map()
        inputs.borrow("input0", HostBuffer(8), DType.float32, [2])
        inputs.borrow("input1", HostBuffer(8), DType.float32, [2])
        # A model from one runtime, a map from another: MAX allows it on the host.
        _ = model.execute(inputs)


def test_capture_on_cpu(dir: String) raises:
    var rt = Runtime()
    var model = rt.load(dir + "/inplace.mef")
    var inputs = rt.tensor_map()
    inputs.borrow("input0", HostBuffer(32), DType.float32, [8])
    inputs.borrow("input1", HostBuffer(32), DType.float32, [8])
    var tensors = List[Tensor]()
    tensors.append(inputs.tensor("input0"))
    tensors.append(inputs.tensor("input1"))
    var raised = False
    try:
        _ = model.capture(1, tensors)
    except e:
        raised = True
        print("    capture on the host:", e)
    expect(raised, "capture on the host did not raise")


def test_accelerator(dir: String) raises:
    """Host memory lent under the accelerator is staged, so in-place writes
    never come back; device memory (`to_device`) is borrowed in place."""
    var rt = Runtime(accelerator=True)
    print("    device:", rt.device_label())
    var add = rt.load(dir + "/add_gpu.mef")
    var inputs = rt.tensor_map()
    var a = HostBuffer(16)
    var b = HostBuffer(16)
    fill(a, 4, 1.0, 0.0)
    fill(b, 4, 2.0, 1.0)
    inputs.borrow_address("input0", a.address(), DType.float32, [4], on_device=True)
    inputs.borrow_address("input1", b.address(), DType.float32, [4], on_device=True)
    var y = add.execute(inputs).tensor("output0").to_host()
    _ = a^  # lent by address: alive until the execute is done
    _ = b^
    expect(y.data[Float32]()[unsafe_offset=3] == 20.0, "add on the accelerator")

    var model = rt.load(dir + "/inplace_gpu.mef")
    # 1. Host memory under the accelerator spec: staged, writes lost.
    var staged = rt.tensor_map()
    var hb = HostBuffer(32)
    var hx = HostBuffer(32)
    fill(hx, 8, 0.0, 0.5)
    staged.borrow_address("input0", hb.address(), DType.float32, [8], on_device=True)
    staged.borrow_address("input1", hx.address(), DType.float32, [8], on_device=True)
    for _ in range(4):
        _ = model.execute(staged)
    _ = hx^
    var host_view = hb.data[Float32]()[unsafe_offset=0]
    print("    host buffer lent under the accelerator, after 4 calls:", host_view)
    # 2. Device memory: copy once, borrow the device address, writes stay.
    var host_side = rt.tensor_map()
    var x = HostBuffer(32)
    fill(x, 8, 0.0, 0.5)
    host_side.borrow("b", HostBuffer(32), DType.float32, [8])
    host_side.borrow("x", x^, DType.float32, [8])
    var b_dev = host_side.tensor("b").to_device(rt.device())
    var x_dev = host_side.tensor("x").to_device(rt.device())
    rt.synchronize()
    print("    x copied to the device, read back:", x_dev.to_host().item[DType.float32](),
          "(device address", x_dev.address(), "on_host", x_dev.on_host(), ")")
    var on_device = rt.tensor_map()
    on_device.borrow_address("input0", b_dev.address(), DType.float32, [8], on_device=True)
    on_device.borrow_address("input1", x_dev.address(), DType.float32, [8], on_device=True)
    var sums = String("")
    for _ in range(4):
        var outputs = model.execute(on_device)
        sums += String(outputs.tensor("output0").to_host().item[DType.float32]()) + " "
    rt.synchronize()
    print("    sum(b) per call, device buffer:", sums)
    var back = b_dev.to_host().data[Float32]()[unsafe_offset=0]
    print("    device buffer, after 4 calls:", back)
    expect(host_view == 0.0, "staged host memory unexpectedly updated")
    _ = hb^  # lent until here
    # Device-graph capture on this accelerator (CUDA / HIP only, per MAX).
    var tensors = List[Tensor]()
    tensors.append(on_device.tensor("input0"))
    tensors.append(on_device.tensor("input1"))
    try:
        var captured = model.capture(7, tensors)
        model.replay(7, tensors)
        print("    capture on", rt.device_label() + ": ok,", len(captured), "outputs")
    except e:
        print("    capture on", rt.device_label() + ":", e)
    comptime if CompilationTarget.is_macos():
        # Metal, MAX 26.6: a device tensor's address lent back under the
        # accelerator spec is read as zeros and not written: the in-place
        # route of `tensor.h` does not hold. Reported, not asserted.
        print("    Metal: device memory borrowed by address is",
              "updated in place" if back == 2.0 else "NOT updated (the model saw zeros)")
    else:
        expect(back == 2.0, "device buffer not updated in place")


def main() raises:
    var args = argv()
    var dir = String(args[1])
    var failed = 0
    var names: List[String] = [
        "version", "add_symbolic", "in_place_across_calls", "copy_is_complete", "errors",
        "lifetimes", "capture_on_cpu", "accelerator",
    ]
    for name in names:
        try:
            if name == "version":
                test_version()
            elif name == "add_symbolic":
                test_add_symbolic(dir)
            elif name == "in_place_across_calls":
                test_in_place_across_calls(dir)
            elif name == "copy_is_complete":
                test_copy_is_complete(dir)
            elif name == "errors":
                test_errors(dir)
            elif name == "lifetimes":
                test_lifetimes(dir)
            elif name == "capture_on_cpu":
                test_capture_on_cpu(dir)
            elif name == "accelerator":
                if accelerator_count() == 0 or not exists(dir + "/add_gpu.mef"):
                    print("SKIP accelerator (none, or no GPU MEFs)")
                    continue
                test_accelerator(dir)
            print("PASS", name)
        except e:
            print("FAIL", name + ":", e)
            failed += 1
    if failed > 0:
        raise Error(String(failed) + " maxrt test(s) failed")
