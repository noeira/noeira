"""A minimal Objective-C bridge: `objc_msgSend` with a typed prototype per
arity, resolved ONCE.

Written from Apple's public objc/runtime headers only. On arm64,
`objc_msgSend` is not variadic at the call site: it must be called through a
pointer cast to the method's exact C prototype (self, _cmd, args...). Every
Metal method this package uses takes and returns integer-class values
(pointers, NSUInteger, uint64_t, BOOL), so one `Int`-only prototype per arity
covers them all. Struct arguments larger than 16 bytes (`MTLSize`, 24 bytes)
are passed by reference on arm64: pass the address of three NSUIntegers.

`OwnedDLHandle.get_function` does a `dlsym` on every call (~500 ns on an M1
Pro); a cached thin function pointer costs ~1.6 ns. A launch makes ~4-6 sends,
so the cache is worth ~2-3 us per launch on its own.

C strings: a `String` passed as `Int(s.unsafe_ptr())` is FREED at its last
use, which is the `unsafe_ptr()` call, before the callee runs. Every helper
here ends the string's life explicitly after the call (`_ = s^`).

BOOL returns: arm64 BOOL is a C `bool` in w0; only the low byte is defined.
Mask with `& 0xFF` (see `send_bool`).
"""

from std.ffi import OwnedDLHandle

comptime Send0 = def(Int, Int) thin abi("C") -> Int
comptime Send1 = def(Int, Int, Int) thin abi("C") -> Int
comptime Send2 = def(Int, Int, Int, Int) thin abi("C") -> Int
comptime Send3 = def(Int, Int, Int, Int, Int) thin abi("C") -> Int
comptime Send4 = def(Int, Int, Int, Int, Int, Int) thin abi("C") -> Int
comptime CFn0 = def() thin abi("C") -> Int
comptime CFn1 = def(Int) thin abi("C") -> Int


def _sym(system: OwnedDLHandle, name: StaticString) raises -> Int:
    """`dlsym(RTLD_DEFAULT, name)`; RTLD_DEFAULT is ((void *)-2) on Darwin."""
    var n = String(name) + "\0"
    var a = system.get_function[Int]("dlsym")(-2, Int(n.unsafe_ptr()))
    _ = n^  # keep the C string alive past the call (Mojo frees at last use)
    if a == 0:
        raise Error(String("dlsym failed: ", name))
    return a


def _as_fn[T: TrivialRegisterPassable](addr: Int) -> T:
    """Reinterpret a code address as a thin C function pointer of type `T`."""
    var slot = List[Int](length=1, fill=addr)
    return Pointer[Int, MutAnyOrigin](
        unsafe_from_address=Int(slot.unsafe_ptr())
    ).unsafe_bitcast[T]()[]


struct ObjC(Movable):
    """The libraries Metal needs, plus `objc_msgSend` and the autorelease-pool
    entry points as cached function pointers."""

    var objc: OwnedDLHandle
    var metal: OwnedDLHandle
    var foundation: OwnedDLHandle
    var system: OwnedDLHandle
    var msg_send: Int  # address of objc_msgSend
    var _s0: Send0
    var _s1: Send1
    var _s2: Send2
    var _s3: Send3
    var _s4: Send4
    var _pool_push: CFn0
    var _pool_pop: CFn1
    var _sel_register: CFn1

    def __init__(out self) raises:
        self.objc = OwnedDLHandle("/usr/lib/libobjc.A.dylib")
        self.metal = OwnedDLHandle(
            "/System/Library/Frameworks/Metal.framework/Metal"
        )
        self.foundation = OwnedDLHandle(
            "/System/Library/Frameworks/Foundation.framework/Foundation"
        )
        self.system = OwnedDLHandle("/usr/lib/libSystem.B.dylib")
        self.msg_send = _sym(self.system, "objc_msgSend")
        self._s0 = _as_fn[Send0](self.msg_send)
        self._s1 = _as_fn[Send1](self.msg_send)
        self._s2 = _as_fn[Send2](self.msg_send)
        self._s3 = _as_fn[Send3](self.msg_send)
        self._s4 = _as_fn[Send4](self.msg_send)
        self._pool_push = _as_fn[CFn0](_sym(self.system, "objc_autoreleasePoolPush"))
        self._pool_pop = _as_fn[CFn1](_sym(self.system, "objc_autoreleasePoolPop"))
        self._sel_register = _as_fn[CFn1](_sym(self.system, "sel_registerName"))

    def sel(self, name: StaticString) -> Int:
        var n = String(name) + "\0"
        var r = self._sel_register(Int(n.unsafe_ptr()))
        _ = n^
        return r

    def cls(self, name: StaticString) raises -> Int:
        var n = String(name) + "\0"
        var c = self.objc.get_function[Int]("objc_getClass")(Int(n.unsafe_ptr()))
        _ = n^
        if c == 0:
            raise Error(String("no ObjC class ", name))
        return c

    @always_inline
    def send0(self, obj: Int, sel: Int) -> Int:
        return self._s0(obj, sel)

    @always_inline
    def send1(self, obj: Int, sel: Int, a: Int) -> Int:
        return self._s1(obj, sel, a)

    @always_inline
    def send2(self, obj: Int, sel: Int, a: Int, b: Int) -> Int:
        return self._s2(obj, sel, a, b)

    @always_inline
    def send3(self, obj: Int, sel: Int, a: Int, b: Int, c: Int) -> Int:
        return self._s3(obj, sel, a, b, c)

    @always_inline
    def send4(self, obj: Int, sel: Int, a: Int, b: Int, c: Int, d: Int) -> Int:
        return self._s4(obj, sel, a, b, c, d)

    @always_inline
    def send_bool(self, obj: Int, sel: Int, a: Int) -> Bool:
        return (self._s1(obj, sel, a) & 0xFF) != 0

    @always_inline
    def pool_push(self) -> Int:
        return self._pool_push()

    @always_inline
    def pool_pop(self, p: Int):
        _ = self._pool_pop(p)

    def release(self, obj: Int):
        if obj != 0:
            _ = self.send0(obj, self.sel("release"))

    def nsstring(self, s: String) raises -> Int:
        """An autoreleased NSString (call inside a pool, or accept the leak)."""
        var c = s + "\0"
        var r = self.send1(
            self.cls("NSString"),
            self.sel("stringWithUTF8String:"),
            Int(c.unsafe_ptr()),
        )
        _ = c^
        return r

    def utf8(self, nsstr: Int) -> String:
        if nsstr == 0:
            return String("<nil>")
        var p = self.send0(nsstr, self.sel("UTF8String"))
        var out = String()
        var i = 0
        while True:
            var c = Pointer[UInt8, MutAnyOrigin](unsafe_from_address=p + i)[
                unsafe_offset=0
            ]
            if c == 0:
                break
            out += chr(Int(c))
            i += 1
        return out^

    def check_error(self, err: Int, what: String) raises:
        if err != 0:
            var d = self.send0(err, self.sel("localizedDescription"))
            raise Error(what + ": " + self.utf8(d))
