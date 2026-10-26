# +--------------------------------------------------------------------------+ #
# | libhdf5 bindings for Mojo
# +--------------------------------------------------------------------------+ #
"""Minimal Mojo FFI over the libhdf5 C library.

Modeled on the SDL3 bindings layout under ``noeira/render/sdl/``: the
shared library handle is loaded lazily via ``_Global``, each H5 sub-API
lives in its own ``h5*.mojo`` file, and ``reader.mojo`` provides the
high-level ``H5File`` / ``H5Dataset`` structs that the dataset loaders
actually consume.

Read surface (``reader.mojo``): file/dataset open+close, dataspace
selection, native-type introspection, bulk + hyperslab + strided reads.

Write surface (``writer.mojo``): file create, chunked datasets with an
UNLIMITED leading axis, append via extent+hyperslab, optional
shuffle+deflate. ``h5native.mojo`` resolves the predefined
``H5T_NATIVE_*`` ids the write path needs — read its docstring before
touching it, the globals are ``-1`` until ``H5open()`` runs.

Compression filters distributed by the ``hdf5plugin`` Python package
(Blosc/LZ4/ZSTD/...) are registered with libhdf5 at first
``H5File.__init__`` via ``H5PLprepend``, so ``HDF5_PLUGIN_PATH`` does
not need to be set as an environment variable.
"""

from std.memory import Pointer
from std.ffi import (
    _Global,
    OwnedDLHandle,
    _get_dylib_function,
    c_char,
    c_uchar,
    c_int,
    c_uint,
    c_long,
    c_size_t,
)
from std.os import getenv
from std.sys import CompilationTarget

from .h5_types import *
from .h5f import *
from .h5d import *
from .h5s import *
from .h5t import *
from .h5e import *
from .h5l import *
from .h5pl import *
from .h5native import *
from .h5p import *
from .reader import *
from .writer import *


comptime Ptr = Pointer

comptime lib = _Global["NOEIRA_HDF5_LIB", _init_hdf5_handle]()


def _hdf5_lib_name() -> String:
    comptime if CompilationTarget.is_macos():
        return String("libhdf5.dylib")
    elif CompilationTarget.is_linux():
        return String("libhdf5.so")
    else:
        comptime assert False, "OS is not supported"


def _init_hdf5_handle() -> OwnedDLHandle:
    """Load the libhdf5 shared library from the pixi env.

    Tries `$CONDA_PREFIX/lib/` first — set by `pixi run` for WHICHEVER env is
    active (the CI tier runs `-e cpu`, whose prefix is `.pixi/envs/cpu`) —
    then `.pixi/envs/default/lib/` relative to CWD, as the SDL loader does.

    Returns an uninitialized handle if the platform is unsupported or
    the library is not found — downstream FFI calls will then raise
    when the handle is dereferenced.
    """
    var name = _hdf5_lib_name()
    var candidates = List[String]()
    var prefix = getenv("CONDA_PREFIX")
    if prefix.byte_length() > 0:
        candidates.append(prefix + "/lib/" + name)
    candidates.append(".pixi/envs/default/lib/" + name)
    for i in range(len(candidates)):
        try:
            return OwnedDLHandle(candidates[i])
        except:
            pass
    var tried = String("")
    for i in range(len(candidates)):
        tried += "\n  - " + candidates[i]
    print("libhdf5 not found. Tried:" + tried)
    return _uninit[OwnedDLHandle]()


@always_inline
def _uninit[T: AnyType](out value: T):
    """Returns uninitialized data."""
    __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(value))


comptime HDF5_PLUGIN_PATH = (
    ".pixi/envs/default/lib/python3.13/site-packages/hdf5plugin/plugins"
)
"""Compile-time path to the hdf5plugin filter directory installed by pixi.

Pinned to Python 3.13 to match ``pixi.toml``. If the python minor version
in pixi.toml is bumped, update this string too.
"""

comptime HDF5_PLUGIN_SUBDIR = "/lib/python3.13/site-packages/hdf5plugin/plugins"
"""The same directory under the active env's ``$CONDA_PREFIX`` (``-e cpu``,
``-e apple``, ...), which ``H5File.__init__`` registers in front of the
default env's."""
