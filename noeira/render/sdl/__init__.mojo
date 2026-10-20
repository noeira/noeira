# +--------------------------------------------------------------------------+ #
# | SDL3 Bindings in Mojo
# +--------------------------------------------------------------------------+ #

"""SDL3 Bindings in Mojo."""
from std.memory import Pointer
from .sdl_audio import *
from .sdl_blendmode import *
from .sdl_camera import *
from .sdl_clipboard import *
from .sdl_error import *
from .sdl_events import *
from .sdl_filesystem import *
from .sdl_gamepad import *
from .sdl_gpu import *
from .sdl_guid import *
from .sdl_haptic import *
from .sdl_hints import *
from .sdl_init import *
from .sdl_iostream import *
from .sdl_joystick import *
from .sdl_keyboard import *
from .sdl_keycode import *
from .sdl_log import *
from .sdl_mouse import *
from .sdl_pen import *
from .sdl_pixels import *
from .sdl_power import *
from .sdl_properties import *
from .sdl_rect import *
from .sdl_render import *
from .sdl_scancode import *
from .sdl_sensor import *
from .sdl_storage import *
from .sdl_surface import *
from .sdl_time import *
from .sdl_timer import *
from .sdl_touch import *
from .sdl_version import *
from .sdl_video import *


comptime Ptr = Pointer


@always_inline
def untracked[
    T: AnyType, o: Origin
](p: Pointer[T, o]) -> Pointer[T, MutUntrackedOrigin]:
    """Re-key a pointer's origin to `MutUntrackedOrigin` for storage in an FFI
    struct field (AnyOrigin is banned in fields as of Mojo 1.0; SDL owns the
    pointee, lifetime is managed explicitly across the C ABI)."""
    return rebind[Pointer[T, MutUntrackedOrigin]](p)


from std.os import abort, getenv
from std.sys import CompilationTarget, Endian
from std.ffi import (
    _Global,
    OwnedDLHandle,
    _get_dylib_function,
    c_char,
    c_uchar,
    c_int,
    c_uint,
    c_short,
    c_ushort,
    c_long,
    c_long_long,
    c_size_t,
    c_ssize_t,
    c_float,
    c_double,
)

comptime lib = _Global["SDL", _init_sdl_handle]()


def _sdl_lib_name() -> String:
    comptime if CompilationTarget.is_macos():
        return String("libSDL3.dylib")
    elif CompilationTarget.is_linux():
        return String("libSDL3.so")
    else:
        comptime assert False, "OS is not supported"


def _init_sdl_handle() -> OwnedDLHandle:
    """Locate + dlopen libSDL3, trying (in order):

      1. `SDL3_LIB` env var — explicit full path override.
      2. `$CONDA_PREFIX/lib/` — set by `pixi run` for WHICHEVER env is
         active (default/apple/nvidia), absolute so it works from any CWD.
      3. `.pixi/envs/default/lib/` relative to CWD — legacy fallback for
         running the binary from the repo root without `pixi run`.
      4. The bare library name — system dlopen search
         (DYLD_LIBRARY_PATH / LD_LIBRARY_PATH / system paths).

    On total failure this ABORTS with the list of attempted paths. It used
    to `print` one line and return an UNINITIALIZED OwnedDLHandle (garbage
    memory) — the failure then surfaced as a segfault at the first SDL
    call, far from the cause."""
    var name = _sdl_lib_name()
    var candidates = List[String]()
    var override = getenv("SDL3_LIB")
    if override.byte_length() > 0:
        candidates.append(override)
    var prefix = getenv("CONDA_PREFIX")
    if prefix.byte_length() > 0:
        candidates.append(prefix + "/lib/" + name)
    candidates.append(".pixi/envs/default/lib/" + name)
    candidates.append(name)

    for i in range(len(candidates)):
        try:
            return OwnedDLHandle(candidates[i])
        except:
            pass

    var tried = String("")
    for i in range(len(candidates)):
        tried += "\n  - " + candidates[i]
    abort(
        "libSDL3 not found. Tried:"
        + tried
        + "\nInstall it via `pixi install`, run through `pixi run`, or set"
        + " SDL3_LIB=/path/to/"
        + name
    )


@always_inline
def _uninit[T: AnyType](out value: T):
    """Returns uninitialized data."""
    __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(value))


@always_inline
def _null_ptr[T: AnyType, O: Origin]() -> Pointer[T, O]:
    """Construct a NULL Pointer for C-ABI FFI calls.

    Mojo nightly made the literal-zero `unsafe_from_address=0` constructor
    a comptime constraint failure, so it cannot be used directly. The
    runtime-`Int` overload of the same constructor still accepts 0 and
    yields the desired zero-address pointer — that's what we need at C
    FFI boundaries (SDL3 calls that document NULL as "use default").
    """
    var addr: Int = 0
    return Pointer[T, O](unsafe_from_address=addr)


comptime ArrayHelper[
    type: ImplicitlyCopyable, size: Int, origin: Origin
] = Ptr[Array[type, size], origin]
