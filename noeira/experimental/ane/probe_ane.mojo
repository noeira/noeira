"""Step 0: can Mojo run a program on the Apple Neural Engine by itself?

Pure Mojo through the Objective-C runtime (`noeira.experimental.metal_rt.objc`)
and Apple's PRIVATE `AppleNeuralEngine.framework`: compile a MIL program in
memory, load it, run it on IOSurface-backed inputs/outputs, check the result
against the CPU, and time it.

The program is a matmul whose weights arrive WITH the input (no recompile when
they change), following maderix/ANE (MIT, github.com/maderix/ANE,
training/test_dynamic_matmul.m):

    input  [1, IC, 1, SEQ+OC] fp16   x[d, t] at (d, t), W[d, c] at (d, SEQ+c)
    output [1, OC, 1, SEQ]    fp16   y[c, t] = sum_d x[d, t] * W[d, c]

All fp16: the M1's ANE (h13) refuses a program with fp32 inputs or outputs
(`ANECCompile() FAILED`, see `--bisect`); maderix's M4 casts fp32 in and out.

    pixi run -e apple mojo run -I . noeira/experimental/ane/probe_ane.mojo
    ... probe_ane.mojo --bisect   # which small programs compile (fp16 vs fp32 I/O, opsets)
"""

from std.ffi import OwnedDLHandle
from std.time import perf_counter_ns
from std.sys import argv
from noeira.experimental.metal_rt.objc import ObjC, _sym, _as_fn, CFn1

comptime Send7 = def(Int, Int, Int, Int, Int, Int, Int, Int, Int) thin abi("C") -> Int
comptime CFn0 = def() thin abi("C") -> Int
comptime CFn3 = def(Int, Int, Int) thin abi("C") -> Int

comptime QOS = 21  # the QoS value maderix/ANE passes everywhere


def mil_dynamic_matmul(ic: Int, oc: Int, seq: Int) -> String:
    var sp = seq + oc
    var m = String(
        "program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\","
        " \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"},"
        " {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\","
        " \"9.0\"}})]\n{\n"
    )
    m += String("    func main<ios18>(tensor<fp16, [1, ", ic, ", 1, ", sp, "]> xh) {\n")
    m += "        tensor<int32, [4]> ba = const()[name = string(\"ba\"), val = tensor<int32, [4]>([0,0,0,0])];\n"
    m += String("        tensor<int32, [4]> sa = const()[name = string(\"sa\"), val = tensor<int32, [4]>([1,", ic, ",1,", seq, "])];\n")
    m += String("        tensor<fp16, [1,", ic, ",1,", seq, "]> act = slice_by_size(x=xh,begin=ba,size=sa)[name=string(\"act\")];\n")
    m += String("        tensor<int32, [4]> bw = const()[name = string(\"bw\"), val = tensor<int32, [4]>([0,0,0,", seq, "])];\n")
    m += String("        tensor<int32, [4]> sw = const()[name = string(\"sw\"), val = tensor<int32, [4]>([1,", ic, ",1,", oc, "])];\n")
    m += String("        tensor<fp16, [1,", ic, ",1,", oc, "]> wt = slice_by_size(x=xh,begin=bw,size=sw)[name=string(\"wt\")];\n")
    m += String("        tensor<int32, [4]> ra = const()[name = string(\"ra\"), val = tensor<int32, [4]>([1,1,", ic, ",", seq, "])];\n")
    m += String("        tensor<fp16, [1,1,", ic, ",", seq, "]> a2 = reshape(shape=ra,x=act)[name=string(\"a2\")];\n")
    m += "        tensor<int32, [4]> pm = const()[name = string(\"pm\"), val = tensor<int32, [4]>([0,1,3,2])];\n"
    m += String("        tensor<fp16, [1,1,", seq, ",", ic, "]> a3 = transpose(perm=pm,x=a2)[name=string(\"a3\")];\n")
    m += String("        tensor<int32, [4]> rw = const()[name = string(\"rw\"), val = tensor<int32, [4]>([1,1,", ic, ",", oc, "])];\n")
    m += String("        tensor<fp16, [1,1,", ic, ",", oc, "]> W = reshape(shape=rw,x=wt)[name=string(\"W\")];\n")
    m += "        bool bF = const()[name = string(\"bF\"), val = bool(false)];\n"
    m += String("        tensor<fp16, [1,1,", seq, ",", oc, "]> yh = matmul(transpose_x=bF,transpose_y=bF,x=a3,y=W)[name=string(\"mm\")];\n")
    m += String("        tensor<fp16, [1,1,", oc, ",", seq, "]> yt = transpose(perm=pm,x=yh)[name=string(\"yt\")];\n")
    m += String("        tensor<int32, [4]> ro = const()[name = string(\"ro\"), val = tensor<int32, [4]>([1,", oc, ",1,", seq, "])];\n")
    m += String("        tensor<fp16, [1,", oc, ",1,", seq, "]> y = reshape(shape=ro,x=yt)[name=string(\"yr\")];\n")
    m += "    } -> (y);\n}\n"
    return m^


struct ANE(Movable):
    var o: ObjC
    var ane: OwnedDLHandle
    var iosurface: OwnedDLHandle
    var _s7: Send7
    var _surf_create: CFn1
    var _surf_lock: CFn3
    var _surf_unlock: CFn3
    var _surf_base: CFn1
    var _cf_release: CFn1
    var _tmpdir: CFn0
    var k_width: Int
    var k_height: Int
    var k_bpe: Int
    var k_bpr: Int
    var k_alloc: Int
    var k_pixfmt: Int

    def __init__(out self) raises:
        self.o = ObjC()
        self.ane = OwnedDLHandle(
            "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine"
        )
        self.iosurface = OwnedDLHandle(
            "/System/Library/Frameworks/IOSurface.framework/IOSurface"
        )
        ref sy = self.o.system
        self._s7 = _as_fn[Send7](self.o.msg_send)
        self._surf_create = _as_fn[CFn1](_sym(sy, "IOSurfaceCreate"))
        self._surf_lock = _as_fn[CFn3](_sym(sy, "IOSurfaceLock"))
        self._surf_unlock = _as_fn[CFn3](_sym(sy, "IOSurfaceUnlock"))
        self._surf_base = _as_fn[CFn1](_sym(sy, "IOSurfaceGetBaseAddress"))
        self._cf_release = _as_fn[CFn1](_sym(sy, "CFRelease"))
        self._tmpdir = _as_fn[CFn0](_sym(sy, "NSTemporaryDirectory"))
        # the kIOSurface* keys are CFStringRef GLOBALS: dlsym gives the variable
        self.k_width = _deref(_sym(sy, "kIOSurfaceWidth"))
        self.k_height = _deref(_sym(sy, "kIOSurfaceHeight"))
        self.k_bpe = _deref(_sym(sy, "kIOSurfaceBytesPerElement"))
        self.k_bpr = _deref(_sym(sy, "kIOSurfaceBytesPerRow"))
        self.k_alloc = _deref(_sym(sy, "kIOSurfaceAllocSize"))
        self.k_pixfmt = _deref(_sym(sy, "kIOSurfacePixelFormat"))

    def num(self, v: Int) raises -> Int:
        return self.o.send1(self.o.cls("NSNumber"), self.o.sel("numberWithLongLong:"), v)

    def surface(self, nbytes: Int) raises -> Int:
        var d = self.o.send0(self.o.cls("NSMutableDictionary"), self.o.sel("dictionary"))
        var put = self.o.sel("setObject:forKey:")
        _ = self.o.send2(d, put, self.num(nbytes), self.k_width)
        _ = self.o.send2(d, put, self.num(1), self.k_height)
        _ = self.o.send2(d, put, self.num(1), self.k_bpe)
        _ = self.o.send2(d, put, self.num(nbytes), self.k_bpr)
        _ = self.o.send2(d, put, self.num(nbytes), self.k_alloc)
        _ = self.o.send2(d, put, self.num(0), self.k_pixfmt)
        var s = self._surf_create(d)
        if s == 0:
            raise Error("IOSurfaceCreate failed")
        return s

    def lock(self, s: Int, read_only: Bool):
        _ = self._surf_lock(s, 1 if read_only else 0, 0)

    def unlock(self, s: Int, read_only: Bool):
        _ = self._surf_unlock(s, 1 if read_only else 0, 0)

    def base(self, s: Int) -> Pointer[Float16, MutAnyOrigin]:
        return Pointer[Float16, MutAnyOrigin](unsafe_from_address=self._surf_base(s))


def _deref(addr: Int) -> Int:
    return Pointer[Int, MutAnyOrigin](unsafe_from_address=addr)[unsafe_offset=0]


struct Kernel(Movable):
    var model: Int
    var request: Int
    var s_in: Int
    var s_out: Int
    var tmpdir: Int
    var compile_ms: Float64

    def __init__(
        out self, a: ANE, mil: String, in_bytes: Int, out_bytes: Int
    ) raises:
        ref o = a.o
        var err = List[Int](length=1, fill=0)
        var errp = Int(err.unsafe_ptr())
        var empty = o.send0(o.cls("NSDictionary"), o.sel("dictionary"))
        var md = o.send2(
            o.cls("NSData"), o.sel("dataWithBytes:length:"),
            Int(mil.unsafe_ptr()), mil.byte_length(),
        )
        var desc = o.send3(
            o.cls("_ANEInMemoryModelDescriptor"),
            o.sel("modelWithMILText:weights:optionsPlist:"), md, empty, 0,
        )
        if desc == 0:
            raise Error("modelWithMILText returned nil")
        var model = o.send1(
            o.cls("_ANEInMemoryModel"), o.sel("inMemoryModelWithDescriptor:"), desc
        )
        if model == 0:
            raise Error("inMemoryModelWithDescriptor returned nil")
        _ = o.send0(model, o.sel("retain"))
        # the "in-memory" compile still reads <tmp>/<hex id>/model.mil
        var hx = o.send0(model, o.sel("hexStringIdentifier"))
        var td = o.send1(a._tmpdir(), o.sel("stringByAppendingPathComponent:"), hx)
        _ = o.send0(td, o.sel("retain"))
        var wdir = o.send1(td, o.sel("stringByAppendingPathComponent:"), o.nsstring("weights"))
        var fm = o.send0(o.cls("NSFileManager"), o.sel("defaultManager"))
        _ = o.send4(
            fm, o.sel("createDirectoryAtPath:withIntermediateDirectories:attributes:error:"),
            wdir, 1, 0, errp,
        )
        o.check_error(err[0], "createDirectory")
        var mpath = o.send1(td, o.sel("stringByAppendingPathComponent:"), o.nsstring("model.mil"))
        if (o.send2(md, o.sel("writeToFile:atomically:"), mpath, 1) & 0xFF) == 0:
            raise Error("writing model.mil failed")

        var t0 = perf_counter_ns()
        var ok = o.send3(model, o.sel("compileWithQoS:options:error:"), QOS, empty, errp) & 0xFF
        o.check_error(err[0], "compileWithQoS")
        if ok == 0:
            raise Error("compileWithQoS returned NO")
        ok = o.send3(model, o.sel("loadWithQoS:options:error:"), QOS, empty, errp) & 0xFF
        o.check_error(err[0], "loadWithQoS")
        if ok == 0:
            raise Error("loadWithQoS returned NO")
        self.compile_ms = Float64(perf_counter_ns() - t0) / 1e6

        self.s_in = a.surface(in_bytes)
        self.s_out = a.surface(out_bytes)
        var aio = o.cls("_ANEIOSurfaceObject")
        var w_in = o.send1(aio, o.sel("objectWithIOSurface:"), self.s_in)
        var w_out = o.send1(aio, o.sel("objectWithIOSurface:"), self.s_out)
        var arr = o.cls("NSArray")
        var one = o.sel("arrayWithObject:")
        var zero = o.send1(arr, one, a.num(0))
        var req = a._s7(
            o.cls("_ANERequest"),
            o.sel("requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:"),
            o.send1(arr, one, w_in), zero, o.send1(arr, one, w_out), zero, 0, 0, a.num(0),
        )
        if req == 0:
            raise Error("_ANERequest returned nil")
        _ = o.send0(req, o.sel("retain"))
        self.model = model
        self.request = req
        self.tmpdir = td
        _ = mil

    def run(self, a: ANE) raises:
        ref o = a.o
        var err = List[Int](length=1, fill=0)
        var ok = o.send4(
            self.model, o.sel("evaluateWithQoS:options:request:error:"),
            QOS, o.send0(o.cls("NSDictionary"), o.sel("dictionary")),
            self.request, Int(err.unsafe_ptr()),
        ) & 0xFF
        o.check_error(err[0], "evaluateWithQoS")
        if ok == 0:
            raise Error("evaluateWithQoS returned NO")

    def free(self, a: ANE) raises:
        ref o = a.o
        var err = List[Int](length=1, fill=0)
        _ = o.send2(self.model, o.sel("unloadWithQoS:error:"), QOS, Int(err.unsafe_ptr()))
        var fm = o.send0(o.cls("NSFileManager"), o.sel("defaultManager"))
        _ = o.send2(fm, o.sel("removeItemAtPath:error:"), self.tmpdir, Int(err.unsafe_ptr()))
        _ = a._cf_release(self.s_in)
        _ = a._cf_release(self.s_out)
        o.release(self.request)
        o.release(self.model)
        o.release(self.tmpdir)
        # the NSError** out-slot must outlive the calls that write it: Mojo
        # frees `err` at its last use, which would be the `unsafe_ptr()`
        # above, and the callee then writes into freed heap (tcmalloc crash)
        _ = err^


def _f16(x: Float32) -> Float64:
    return Float64(x.cast[DType.float16]().cast[DType.float32]())


def check_and_time(a: ANE, ic: Int, oc: Int, seq: Int, reps: Int) raises -> Bool:
    var pool = a.o.pool_push()
    var sp = seq + oc
    var k = Kernel(a, mil_dynamic_matmul(ic, oc, seq), ic * sp * 2, oc * seq * 2)

    # deterministic inputs in [-1, 1) and W ~ 1/sqrt(ic)
    var x = List[Float32](length=ic * seq, fill=0)
    var w = List[Float32](length=ic * oc, fill=0)
    var state = UInt64(0x9E3779B97F4A7C15)
    for i in range(ic * seq):
        state = state * 6364136223846793005 + 1442695040888963407
        x[i] = Float32(Int(state >> 40) % 2000 - 1000) / 1000.0
    var sc = 1.0 / Float32(ic) ** 0.5
    for i in range(ic * oc):
        state = state * 6364136223846793005 + 1442695040888963407
        w[i] = Float32(Int(state >> 40) % 2000 - 1000) / 1000.0 * sc

    var xh = List[Float64](length=ic * seq, fill=0)
    for i in range(ic * seq):
        xh[i] = _f16(x[i])
    var t_step = 1 if ic * oc <= 512 * 512 else 8
    var worst = Float64(0)  # max |err| / sum_d |x w|
    var worst_abs = Float64(0)
    # two weight sets through the SAME compiled program: the second proves
    # a weight change needs no recompile
    for pass_ in range(2):
        var wmul = Float32(1.0) if pass_ == 0 else Float32(-0.5)
        a.lock(k.s_in, False)
        var inp = a.base(k.s_in)
        for d in range(ic):
            for t in range(seq):
                inp[unsafe_offset = d * sp + t] = x[d * seq + t].cast[DType.float16]()
            for c in range(oc):
                inp[unsafe_offset = d * sp + seq + c] = (w[d * oc + c] * wmul).cast[DType.float16]()
        a.unlock(k.s_in, False)
        k.run(a)
        a.lock(k.s_out, True)
        var out = a.base(k.s_out)
        var wh = List[Float64](length=ic * oc, fill=0)
        for i in range(ic * oc):
            wh[i] = _f16(w[i] * wmul)
        for c in range(oc):
            for t in range(0, seq, t_step):
                var r = Float64(0)
                var mag = Float64(0)
                for d in range(ic):
                    var p = xh[d * seq + t] * wh[d * oc + c]
                    r += p
                    mag += abs(p)
                var e = abs(Float64(out[unsafe_offset = c * seq + t].cast[DType.float32]()) - r)
                worst_abs = max(worst_abs, e)
                worst = max(worst, e / max(mag, 1e-30))
        a.unlock(k.s_out, True)

    # timing: input already written; just evaluate
    for _ in range(5):
        k.run(a)
    var best = Float64(1e30)
    for _ in range(3):
        var t0 = perf_counter_ns()
        for _ in range(reps):
            k.run(a)
        best = min(best, Float64(perf_counter_ns() - t0) / 1e3 / Float64(reps))
    var gflop = 2.0 * Float64(ic) * Float64(oc) * Float64(seq) / 1e9
    # fp16 inputs and output: allow 4 fp16 ulps of the magnitude
    var ok = worst < 4.0 * 2.0 ** -11
    print(
        "  IC", ic, "OC", oc, "SEQ", seq,
        "| compile+load", k.compile_ms, "ms",
        "| eval", best, "us",
        "|", gflop / (best * 1e-6) / 1e3, "TFLOP/s",
        "| max err/sum|xw|", worst, "abs", worst_abs,
        "OK" if ok else "FAIL",
    )
    k.free(a)
    a.o.pool_pop(pool)
    return ok


def _hdr() -> String:
    return String(
        "program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\","
        " \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"},"
        " {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\","
        " \"9.0\"}})]\n{\n"
    )


def mil_add(dt: String, c: Int, s: Int, opset: String) -> String:
    var t = String("tensor<", dt, ", [1, ", c, ", 1, ", s, "]>")
    var m = _hdr()
    m += String("    func main<", opset, ">(", t, " x) {\n")
    m += String("        ", t, " y = add(x = x, y = x)[name = string(\"y\")];\n")
    m += "    } -> (y);\n}\n"
    return m^


def mil_cast_add(c: Int, s: Int, opset: String) -> String:
    var t32 = String("tensor<fp32, [1, ", c, ", 1, ", s, "]>")
    var t16 = String("tensor<fp16, [1, ", c, ", 1, ", s, "]>")
    var m = _hdr()
    m += String("    func main<", opset, ">(", t32, " x) {\n")
    m += "        string to16 = const()[name = string(\"to16\"), val = string(\"fp16\")];\n"
    m += String("        ", t16, " xh = cast(dtype = to16, x = x)[name = string(\"cin\")];\n")
    m += String("        ", t16, " yh = add(x = xh, y = xh)[name = string(\"a\")];\n")
    m += "        string to32 = const()[name = string(\"to32\"), val = string(\"fp32\")];\n"
    m += String("        ", t32, " y = cast(dtype = to32, x = yh)[name = string(\"cout\")];\n")
    m += "    } -> (y);\n}\n"
    return m^


def try_compile(a: ANE, name: String, mil: String, in_b: Int, out_b: Int):
    var pool = a.o.pool_push()
    try:
        var k = Kernel(a, mil, in_b, out_b)
        k.run(a)
        print("  ", name, ": compile+load", k.compile_ms, "ms, eval OK")
        k.free(a)
    except e:
        print("  ", name, ": FAIL", e)
    a.o.pool_pop(pool)


def bisect(a: ANE):
    print("bisect: which programs does this ANE compile?")
    for opset in ["ios16", "ios17", "ios18"]:
        try_compile(a, String("add fp16 ", opset), mil_add("fp16", 64, 64, opset), 64 * 64 * 2, 64 * 64 * 2)
        try_compile(a, String("add fp32 ", opset), mil_add("fp32", 64, 64, opset), 64 * 64 * 4, 64 * 64 * 4)
        try_compile(a, String("cast fp32->fp16->fp32 ", opset), mil_cast_add(64, 64, opset), 64 * 64 * 4, 64 * 64 * 4)


def main() raises:
    var a = ANE()
    var args = argv()
    if len(args) > 1 and args[1] == "--bisect":
        bisect(a)
        return
    var info = a.o.cls("_ANEDeviceInfo")
    print(
        "ANE:", a.o.send0(info, a.o.sel("numANECores")), "cores, arch",
        a.o.utf8(a.o.send0(info, a.o.sel("aneArchitectureType"))),
    )
    var all_ok = True
    all_ok = check_and_time(a, 64, 64, 64, 200) and all_ok
    all_ok = check_and_time(a, 256, 256, 256, 200) and all_ok
    all_ok = check_and_time(a, 512, 512, 512, 100) and all_ok
    all_ok = check_and_time(a, 1024, 1024, 512, 50) and all_ok
    all_ok = check_and_time(a, 2048, 2048, 512, 20) and all_ok
    print("ALL OK" if all_ok else "FAILED")
    if not all_ok:
        raise Error("ANE probe failed")
