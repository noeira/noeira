"""Can Mojo drive Metal itself? Compile a Mojo kernel to a metallib with the
stdlib's `_compile_code`, then load and launch it through the Objective-C
runtime (`objc_msgSend`) from Mojo, with Apple's public Metal API only.

Times K dependent 1-thread launches three ways:
  - one command buffer per launch (what a per-launch runtime does);
  - B launches per command buffer, one serial encoder;
  - all K in one command buffer.
"""

from std.ffi import OwnedDLHandle
from std.time import perf_counter_ns
from max.gpu import global_idx
from max.gpu.host.compile import _compile_code

comptime K = 2000


def bump(x: Pointer[Float32, MutAnyOrigin]):
    if Int(global_idx.x) == 0:
        x[unsafe_offset=0] = x[unsafe_offset=0] + 1.0


struct ObjC(Movable):
    var objc: OwnedDLHandle
    var metal: OwnedDLHandle
    var foundation: OwnedDLHandle
    var system: OwnedDLHandle

    def __init__(out self) raises:
        self.objc = OwnedDLHandle("/usr/lib/libobjc.A.dylib")
        self.metal = OwnedDLHandle("/System/Library/Frameworks/Metal.framework/Metal")
        self.foundation = OwnedDLHandle(
            "/System/Library/Frameworks/Foundation.framework/Foundation"
        )
        self.system = OwnedDLHandle("/usr/lib/libSystem.B.dylib")

    def sel(self, name: StaticString) raises -> Int:
        return self.objc.get_function[Int]("sel_registerName")(Int(name.unsafe_ptr()))

    def cls(self, name: StaticString) raises -> Int:
        return self.objc.get_function[Int]("objc_getClass")(Int(name.unsafe_ptr()))

    # objc_msgSend must be called with the method's exact (non-variadic)
    # prototype; each arity gets its own typed call.
    def send0(self, obj: Int, sel: Int) raises -> Int:
        return self.objc.get_function[Int]("objc_msgSend")(obj, sel)

    def send1(self, obj: Int, sel: Int, a: Int) raises -> Int:
        return self.objc.get_function[Int]("objc_msgSend")(obj, sel, a)

    def send2(self, obj: Int, sel: Int, a: Int, b: Int) raises -> Int:
        return self.objc.get_function[Int]("objc_msgSend")(obj, sel, a, b)

    def send3(self, obj: Int, sel: Int, a: Int, b: Int, c: Int) raises -> Int:
        return self.objc.get_function[Int]("objc_msgSend")(obj, sel, a, b, c)

    def pool_push(self) raises -> Int:
        return self.objc.get_function[Int]("objc_autoreleasePoolPush")()

    def pool_pop(self, p: Int) raises:
        _ = self.objc.get_function[NoneType]("objc_autoreleasePoolPop")(p)


def nsstring_utf8(o: ObjC, s: Int) raises -> String:
    var p = o.send0(s, o.sel("UTF8String"))
    var out = String()
    var i = 0
    while True:
        var c = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=p + i)[unsafe_offset=0]
        if c == 0:
            break
        out += chr(Int(c))
        i += 1
    return out^


def check_err(o: ObjC, err: Int, what: StaticString) raises:
    if err != 0:
        var d = o.send0(err, o.sel("localizedDescription"))
        raise Error(String(what, ": ", nsstring_utf8(o, d)))


def main() raises:
    var o = ObjC()
    var dev = o.metal.get_function[Int]("MTLCreateSystemDefaultDevice")()
    print("device:", nsstring_utf8(o, o.send0(dev, o.sel("name"))))

    # 1. Mojo kernel -> metallib bytes (stdlib, Apache 2.0)
    var info = _compile_code[bump, emission_kind="object"]()
    var lib_bytes = info.asm
    var fname: String = String(info.function_name)
    print("kernel:", fname, "| metallib bytes:", lib_bytes.byte_length())

    # 2. metallib -> MTLLibrary -> MTLFunction -> pipeline
    var data = o.system.get_function[Int]("dispatch_data_create")(
        Int(lib_bytes.unsafe_ptr()), lib_bytes.byte_length(), 0, 0
    )
    var err_box = List[Int](length=1, fill=0)
    var errp = Int(err_box.unsafe_ptr())
    var mlib = o.send2(dev, o.sel("newLibraryWithData:error:"), data, errp)
    check_err(o, err_box[0], "newLibraryWithData")
    var cname = fname + "\0"
    var nsname = o.send1(
        o.cls("NSString"), o.sel("stringWithUTF8String:"), Int(cname.unsafe_ptr())
    )
    var mfn = o.send1(mlib, o.sel("newFunctionWithName:"), nsname)
    if mfn == 0:
        raise Error("function not found in metallib: " + fname)
    var pso = o.send2(dev, o.sel("newComputePipelineStateWithFunction:error:"), mfn, errp)
    check_err(o, err_box[0], "newComputePipelineState")

    # 3. a shared 4-byte buffer; Mojo pointer args are passed as their
    #    8-byte GPU address via setBytes
    var buf = o.send2(dev, o.sel("newBufferWithLength:options:"), 4, 0)
    var host = Pointer[Float32, MutAnyOrigin](
        unsafe_from_address=o.send0(buf, o.sel("contents"))
    )
    var addr = List[Int](length=1, fill=o.send0(buf, o.sel("gpuAddress")))
    var queue = o.send0(dev, o.sel("newCommandQueue"))

    # MTLSize is 24 bytes: arm64 passes it by reference to a copy, so a
    # pointer to three NSUIntegers IS the by-value ABI.
    var one = List[Int](length=3, fill=1)

    var s_cb = o.sel("commandBuffer")
    var s_enc = o.sel("computeCommandEncoder")
    var s_pso = o.sel("setComputePipelineState:")
    var s_bytes = o.sel("setBytes:length:atIndex:")
    var s_use = o.sel("useResource:usage:")
    var s_disp = o.sel("dispatchThreadgroups:threadsPerThreadgroup:")
    var s_end = o.sel("endEncoding")
    var s_commit = o.sel("commit")
    var s_wait = o.sel("waitUntilCompleted")

    print("launches per command buffer | us/launch | x after (expect", K, ")")
    for per_cb in [1, 4, 16, 64, K]:
        var best = Float64(1e30)
        for r in range(4):
            host[unsafe_offset=0] = 0.0
            var t0 = perf_counter_ns()
            var pool = o.pool_push()
            var done = 0
            var last = 0
            while done < K:
                var cb = o.send0(queue, s_cb)
                var enc = o.send0(cb, s_enc)
                _ = o.send1(enc, s_pso, pso)
                _ = o.send2(enc, s_use, buf, 3)  # read|write
                var n = min(per_cb, K - done)
                for _ in range(n):
                    _ = o.send3(enc, s_bytes, Int(addr.unsafe_ptr()), 8, 0)
                    _ = o.send2(enc, s_disp, Int(one.unsafe_ptr()), Int(one.unsafe_ptr()))
                _ = o.send0(enc, s_end)
                _ = o.send0(cb, s_commit)
                last = cb
                done += n
            _ = o.send0(last, s_wait)
            o.pool_pop(pool)
            var us = Float64(perf_counter_ns() - t0) / 1e3 / Float64(K)
            if r > 0 and us < best:
                best = us
            if r == 3:
                print(per_cb, "|", best, "|", host[unsafe_offset=0])
