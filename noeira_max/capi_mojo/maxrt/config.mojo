"""Finding the MAX runtime's own libraries without an activated environment.

The C API loads its compiler runtime from paths listed in
`$MODULAR_HOME/modular.cfg`. Inside `pixi run` or an activated conda env,
activation sets `MODULAR_HOME`; elsewhere `M_initModel` fails with `unable to
locate compiler_rt /lib/libKGENCompilerRTShared.so`. When `MODULAR_HOME` is
unset, `ensure_modular_home` locates the `libmax` this process linked
(`dlsym` + `dladdr`), writes a minimal `modular.cfg` next to the environment
it came from, and sets `MODULAR_HOME` to it. A workaround for
RFC_MAX_FROM_MOJO item 1a: the runtime should find itself.
"""

from std.ffi import external_call
from std.os import getenv, makedirs, setenv
from std.sys import CompilationTarget

from . import capi


@fieldwise_init
struct _DlInfo(Copyable, Movable):
    """`Dl_info`: four pointers, the first the shared object's path."""

    var fname: Int
    var fbase: Int
    var sname: Int
    var saddr: Int


def libmax_path() raises -> String:
    """The path of the `libmax` that defines `M_newStatus` in this process."""
    var rtld_default: Int
    comptime if CompilationTarget.is_macos():
        rtld_default = -2  # ((void *) -2) on Darwin
    else:
        rtld_default = 0  # glibc
    var symbol = String("M_newStatus")
    var address = external_call["dlsym", Int](
        rtld_default, symbol.as_c_string_span().ptr()
    )
    if address == 0:
        raise Error("maxrt: M_newStatus not found; is libmax linked (-lmax)?")
    var info = _DlInfo(0, 0, 0, 0)
    if external_call["dladdr", Int32](address, Pointer(to=info)) == 0:
        raise Error("maxrt: dladdr failed on M_newStatus")
    var copy = external_call["strdup", capi.CString](info.fname)
    var path = capi.to_string(copy)
    external_call["free", NoneType](Int(copy))
    return path


def _parent(path: String) -> String:
    var parts = path.split("/")
    var out = String("")
    for i in range(len(parts) - 1):
        if i > 0:
            out += "/"
        out += String(parts[i])
    return out


def ensure_modular_home() raises:
    """Sets `MODULAR_HOME` to a generated `modular.cfg` if it is unset."""
    if getenv("MODULAR_HOME").byte_length() > 0:
        return
    var lib_dir = _parent(libmax_path())
    var prefix = _parent(lib_dir)
    var ext: String
    comptime if CompilationTarget.is_macos():
        ext = ".dylib"
    else:
        ext = ".so"
    var home = getenv("TMPDIR", "/tmp") + "/maxrt-modular-home"
    makedirs(home, exist_ok=True)
    var cfg = String("[max]\n")
    cfg += "package_root = " + prefix + "\n"
    cfg += "cache_dir = " + prefix + "/share/max/.max_cache\n"
    cfg += "\n[mojo-max]\n"
    cfg += "package_root = " + prefix + "\n"
    cfg += "compilerrt_path = " + lib_dir + "/libKGENCompilerRTShared" + ext + "\n"
    cfg += "mgprt_path = " + lib_dir + "/libMGPRT" + ext + "\n"
    with open(home + "/modular.cfg", "w") as f:
        f.write(cfg)
    _ = setenv("MODULAR_HOME", home, True)
