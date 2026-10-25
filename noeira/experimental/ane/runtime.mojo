"""Running MIL programs on the Apple Neural Engine, from pure Mojo.

Through the Objective-C runtime (`noeira.experimental.metal_rt.objc`) and
Apple's PRIVATE `AppleNeuralEngine.framework` (it may break on any macOS
update). The call sequence follows maderix/ANE (MIT,
github.com/maderix/ANE, `inmem_basic.m`, `training/stories_io.h`):

    _ANEInMemoryModelDescriptor modelWithMILText:weights:optionsPlist:
    _ANEInMemoryModel inMemoryModelWithDescriptor:
    write <NSTemporaryDirectory()>/<hexStringIdentifier>/model.mil
    compileWithQoS:options:error:, loadWithQoS:options:error:
    _ANERequest requestWithInputs:...  over _ANEIOSurfaceObject's
    evaluateWithQoS:options:request:error:   (one synchronous run)

Every input and output is an fp16 IOSurface holding a `[1, C, 1, S]`
tensor, channel-major (`(c, s)` at `c * S + s`): the M1's ANE (h13)
refuses fp32 inputs and outputs (`ANECCompile() FAILED`).
"""

from std.ffi import OwnedDLHandle
from std.time import perf_counter_ns
from noeira.experimental.metal_rt.objc import ObjC, _sym, _as_fn, CFn1

comptime Send7 = def(Int, Int, Int, Int, Int, Int, Int, Int, Int) thin abi("C") -> Int
comptime CFn0 = def() thin abi("C") -> Int
comptime CFn3 = def(Int, Int, Int) thin abi("C") -> Int

comptime QOS = 21  # the QoS value maderix/ANE passes everywhere


def mil_header() -> String:
    return String(
        "program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\","
        " \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"},"
        " {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\","
        " \"9.0\"}})]\n{\n"
    )


def _deref(addr: Int) -> Int:
    return Pointer[Int, MutAnyOrigin](unsafe_from_address=addr)[unsafe_offset=0]


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

    def array(self, items: List[Int]) raises -> Int:
        """An autoreleased NSArray of `items` (ids)."""
        var r = self.o.send2(
            self.o.cls("NSArray"), self.o.sel("arrayWithObjects:count:"),
            Int(items.unsafe_ptr()), len(items),
        )
        _ = items
        return r

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


struct Layout(Copyable):
    """Where element `(c, h, w)` of a `[1, C, H, W]` fp16 input or output
    lives on its surface: at `c * plane + h * row + w` (in fp16 elements).
    Read from the loaded model (`modelAttributes`): rows are padded to 64
    BYTES, so a surface is not dense unless `W` is a multiple of 32."""

    var name: String
    var channels: Int
    var height: Int
    var width: Int
    var row: Int
    var plane: Int
    var nbytes: Int

    def __init__(out self, var name: String, channels: Int, height: Int, width: Int,
                 row: Int, plane: Int, nbytes: Int):
        self.name = name^
        self.channels = channels
        self.height = height
        self.width = width
        self.row = row
        self.plane = plane
        self.nbytes = nbytes


def _obj(a: ANE, d: Int, key: StaticString) raises -> Int:
    var r = a.o.send1(d, a.o.sel("objectForKey:"), a.o.nsstring(key))
    if r == 0:
        raise Error(String("model attributes: no key ", key))
    return r


def _int(a: ANE, d: Int, key: StaticString) raises -> Int:
    return a.o.send0(_obj(a, d, key), a.o.sel("longLongValue"))


def _layouts(a: ANE, model: Int, key: StaticString) raises -> List[Layout]:
    """`modelAttributes.NetworkStatusList[0].<key>`, one `Layout` per entry
    (sorted by symbol name, which is the order requests index them by)."""
    ref o = a.o
    var attrs = o.send0(model, o.sel("modelAttributes"))
    var nets = _obj(a, attrs, "NetworkStatusList")
    var lst = _obj(a, o.send1(nets, o.sel("objectAtIndex:"), 0), key)
    var out = List[Layout]()
    for i in range(o.send0(lst, o.sel("count"))):
        var d = o.send1(lst, o.sel("objectAtIndex:"), i)
        if _int(a, d, "Batches") != 1 or _int(a, d, "Depth") != 1:
            raise Error("model attributes: batched/depth I/O not supported")
        out.append(Layout(
            o.utf8(_obj(a, d, "Name")), _int(a, d, "Channels"), _int(a, d, "Height"),
            _int(a, d, "Width"), _int(a, d, "RowStride") // 2,
            _int(a, d, "PlaneStride") // 2, _int(a, d, "BatchStride"),
        ))
    return out^


struct Program(Movable):
    """One compiled and loaded MIL program, its fp16 input and output
    surfaces and the request that binds them.

    Surfaces are in SYMBOL-NAME order (what the model's attributes list and
    a request's indices refer to), not argument order: name the arguments
    and outputs so that both orders agree (the emitter uses i000, i001, …
    and o000, o001, …)."""

    var model: Int
    var request: Int
    var ins: List[Int]
    var outs: List[Int]
    var in_layout: List[Layout]
    var out_layout: List[Layout]
    var tmpdir: Int
    var compile_ms: Float64

    def __init__(out self, a: ANE, mil: String) raises:
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

        self.in_layout = _layouts(a, model, "LiveInputList")
        self.out_layout = _layouts(a, model, "LiveOutputList")
        self.ins = List[Int]()
        self.outs = List[Int]()
        var aio = o.cls("_ANEIOSurfaceObject")
        var wrap = o.sel("objectWithIOSurface:")
        var w_in = List[Int]()
        var w_out = List[Int]()
        var i_idx = List[Int]()
        var o_idx = List[Int]()
        for i in range(len(self.in_layout)):
            self.ins.append(a.surface(self.in_layout[i].nbytes))
            w_in.append(o.send1(aio, wrap, self.ins[i]))
            i_idx.append(a.num(i))
        for i in range(len(self.out_layout)):
            self.outs.append(a.surface(self.out_layout[i].nbytes))
            w_out.append(o.send1(aio, wrap, self.outs[i]))
            o_idx.append(a.num(i))
        var req = a._s7(
            o.cls("_ANERequest"),
            o.sel("requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:"),
            a.array(w_in), a.array(i_idx), a.array(w_out), a.array(o_idx), 0, 0, a.num(0),
        )
        if req == 0:
            raise Error("_ANERequest returned nil")
        _ = o.send0(req, o.sel("retain"))
        self.model = model
        self.request = req
        self.tmpdir = td
        _ = mil
        _ = err^

    def write(self, a: ANE, k: Int, src: Pointer[Float32, _]) raises:
        """Input `k` from `src`: the tensor's C*H rows of W floats, row-major
        and dense, converted to fp16 into the surface's padded rows."""
        ref L = self.in_layout[k]
        a.lock(self.ins[k], False)
        var b = a.base(self.ins[k])
        var j = 0
        for c in range(L.channels):
            for h in range(L.height):
                var off = c * L.plane + h * L.row
                for w in range(L.width):
                    b[unsafe_offset = off + w] = src[unsafe_offset=j].cast[DType.float16]()
                    j += 1
        a.unlock(self.ins[k], False)

    def read(self, a: ANE, k: Int, dst: Pointer[Float32, MutAnyOrigin], scale: Float32 = 1.0) raises:
        """Output `k` into `dst` (dense, row-major), each element times `scale`."""
        ref L = self.out_layout[k]
        a.lock(self.outs[k], True)
        var b = a.base(self.outs[k])
        var j = 0
        for c in range(L.channels):
            for h in range(L.height):
                var off = c * L.plane + h * L.row
                for w in range(L.width):
                    dst[unsafe_offset=j] = b[unsafe_offset = off + w].cast[DType.float32]() * scale
                    j += 1
        a.unlock(self.outs[k], True)

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
        _ = err^

    def free(self, a: ANE) raises:
        ref o = a.o
        var err = List[Int](length=1, fill=0)
        _ = o.send2(self.model, o.sel("unloadWithQoS:error:"), QOS, Int(err.unsafe_ptr()))
        var fm = o.send0(o.cls("NSFileManager"), o.sel("defaultManager"))
        _ = o.send2(fm, o.sel("removeItemAtPath:error:"), self.tmpdir, Int(err.unsafe_ptr()))
        for s in self.ins:
            _ = a._cf_release(s)
        for s in self.outs:
            _ = a._cf_release(s)
        o.release(self.request)
        o.release(self.model)
        o.release(self.tmpdir)
        # the NSError** out-slot must outlive the calls that write it: Mojo
        # frees `err` at its last use, which would be the `unsafe_ptr()`
        # above, and the callee then writes into freed heap (tcmalloc crash)
        _ = err^


def describe(a: ANE, obj: Int, selname: StaticString) raises -> String:
    """`[[obj selname] description]`, or a note if `obj` does not respond."""
    ref o = a.o
    var s = o.sel(selname)
    if (o.send1(obj, o.sel("respondsToSelector:"), s) & 0xFF) == 0:
        return String("<does not respond to ", selname, ">")
    var r = o.send0(obj, s)
    return o.utf8(o.send0(r, o.sel("description")))
