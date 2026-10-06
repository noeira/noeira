from std.ffi import c_char, external_call
from std.time import perf_counter_ns

comptime M_HOST: Int32 = 0
comptime M_FLOAT32: Int32 = 17 | (1 << 6)


def check(status: Int, what: String) raises:
    if external_call["M_isError", Int32](status) != 0:
        var msg = external_call["M_getError", Pointer[c_char, MutUntrackedOrigin]](status)
        raise Error(what + " failed: " + String(unsafe_from_utf8_ptr=msg))


def main() raises:
    var status = external_call["M_newStatus", Int]()
    var rcfg = external_call["M_newRuntimeConfig", Int]()
    var host = external_call["M_newDevice", Int](M_HOST, Int32(0), status)
    check(status, "M_newDevice")
    external_call["M_runtimeConfigAddDevice", NoneType](rcfg, host)
    var ctx = external_call["M_newRuntimeContext", Int](rcfg, status)
    check(status, "M_newRuntimeContext")

    var cfg = external_call["M_newCompileConfig", Int]()
    var path = String("graph.mef")
    external_call["M_setModelPath", NoneType](cfg, path.as_c_string_span().ptr())
    var compiled = external_call["M_compileModelSync", Int](ctx, Pointer(to=cfg), status)
    check(status, "M_compileModelSync")
    var model = external_call["M_initModel", Int](ctx, compiled, Int(0), status)
    check(status, "M_initModel")

    var a = List[Float32](capacity=8)
    var b = List[Float32](capacity=8)
    for i in range(8):
        a.append(Float32(i + 1))
        b.append(Float32(8 - i))
    var shape = List[Int64]()
    shape.append(8)
    var n0 = String("input0")
    var n1 = String("input1")
    var s0 = external_call["M_newTensorSpec", Int](shape.unsafe_ptr(), Int64(1), M_FLOAT32, n0.as_c_string_span().ptr(), host)
    var s1 = external_call["M_newTensorSpec", Int](shape.unsafe_ptr(), Int64(1), M_FLOAT32, n1.as_c_string_span().ptr(), host)
    var inputs = external_call["M_newAsyncTensorMap", Int](ctx)
    external_call["M_borrowTensorInto", NoneType](inputs, a.unsafe_ptr(), s0, status)
    check(status, "borrow input0")
    external_call["M_borrowTensorInto", NoneType](inputs, b.unsafe_ptr(), s1, status)
    check(status, "borrow input1")

    var outputs = external_call["M_executeModelSync", Int](ctx, model, inputs, status)
    check(status, "M_executeModelSync")
    var N = 2000
    var t0 = perf_counter_ns()
    for _ in range(N):
        var o2 = external_call["M_executeModelSync", Int](ctx, model, inputs, status)
        external_call["M_freeAsyncTensorMap", NoneType](o2)
    var t1 = perf_counter_ns()
    check(status, "loop")
    print("per-call execute (CPU, n=8):", Float64(t1 - t0) / Float64(N) / 1000.0, "us")
    var on = String("output0")
    var out = external_call["M_getTensorByNameFrom", Int](outputs, on.as_c_string_span().ptr(), status)
    check(status, "get output0")
    var data = external_call["M_getTensorData", Pointer[Float32, ImmutAnyOrigin]](out)
    var n = external_call["M_getTensorNumElements", Int](out)
    var line = String("Mojo -> MAX C API, CPU, n=") + String(n) + ": ["
    for i in range(n):
        line += String(data[unsafe_offset=i]) + (", " if i < n - 1 else "]")
    print(line)
    # `a` and `b` are lent to the runtime by address, and Mojo frees a value
    # at its last use: without these, they could be freed before the first
    # execute reads them.
    _ = a^
    _ = b^
