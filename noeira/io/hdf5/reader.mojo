# +--------------------------------------------------------------------------+ #
# | libhdf5 — high-level read API
# +--------------------------------------------------------------------------+ #
"""High-level HDF5 reader: ``H5File`` and ``H5Dataset``.

These structs own their underlying ``hid_t`` handles and release them
on destruction. Movable but not Copyable — moves transfer the handles,
copies are forbidden.

Read flow:
    var f = H5File("data.h5")
    var ds = f.open_dataset("pixels")
    print(ds.dims, ds.dtype_class, ds.elem_size)
    var buf = alloc[Scalar[DType.uint8]](ds.n_elements())
    ds.read_all(buf)
    ...
    buf.free()
"""

from std.memory import alloc, Pointer
from std.os import getenv

from . import c_int, HDF5_PLUGIN_PATH, HDF5_PLUGIN_SUBDIR
from .h5_types import (
    H5F_ACC_RDONLY,
    H5P_DEFAULT,
    H5S_ALL,
    H5S_SELECT_SET,
    H5T_DIR_DEFAULT,
    H5T_INTEGER,
    H5T_SGN_NONE,
    hid_t,
    hsize_t,
)
from .h5d import h5d_close, h5d_get_space, h5d_get_type, h5d_open2, h5d_read
from .h5f import h5f_close, h5f_open
from .h5l import list_link_names
from .h5pl import h5pl_prepend
from .h5s import (
    h5s_close,
    h5s_create_simple,
    h5s_get_simple_extent_dims,
    h5s_get_simple_extent_ndims,
    h5s_select_hyperslab,
)
from .h5t import (
    h5t_close,
    h5t_get_class,
    h5t_get_native_type,
    h5t_get_sign,
    h5t_get_size,
)


@always_inline
def _null_ptr[T: AnyType, O: Origin]() -> Pointer[T, O]:
    """NULL Pointer for HDF5 FFI "optional output" args.

    Mojo nightly's comptime `unsafe_from_address=0` literal is rejected;
    the runtime-Int overload still accepts 0 to produce a real NULL.
    """
    var addr: Int = 0
    return Pointer[T, O](unsafe_from_address=addr)


struct H5File(Movable):
    """Owning handle to an open HDF5 file (read-only)."""

    var file_id: hid_t

    def __init__(out self, var path: String) raises:
        # Register hdf5plugin's filter directory so compressed datasets
        # decode without HDF5_PLUGIN_PATH being set in the shell env.
        # H5PLprepend is idempotent enough — duplicate paths are harmless.
        try:
            _ = h5pl_prepend(String(HDF5_PLUGIN_PATH))
            var prefix = getenv("CONDA_PREFIX")
            if prefix.byte_length() > 0:
                _ = h5pl_prepend(prefix + HDF5_PLUGIN_SUBDIR)
        except:
            pass

        self.file_id = h5f_open(path^, H5F_ACC_RDONLY, H5P_DEFAULT)
        if self.file_id < 0:
            raise Error(
                "H5Fopen failed: hid_t=" + String(Int(self.file_id))
            )

    def __deinit__(deinit self):
        if self.file_id > 0:
            try:
                _ = h5f_close(self.file_id)
            except:
                pass

    def dataset_names(self) raises -> List[String]:
        """Top-level link names in the file, in name order.

        The entry point for ingesting a file we did not write: enumerate, then
        `open_dataset` each to recover its shape and dtype.
        """
        return list_link_names(self.file_id)

    def has_dataset(self, name: String) raises -> Bool:
        """Does `name` open as a dataset? PATHS INCLUDED.

        ⚠⚠ IT OPENS AND CLOSES, IT DOES NOT SEARCH `dataset_names()`. That is
        the top-level link list, so the earlier version answered False for
        every NESTED path — `"data/demo_0/actions"` in a LIBERO recording —
        while `open_dataset` on the same string succeeded. A predicate that
        contradicts the accessor it guards is worse than no predicate: the
        LIBERO demo importer probed `data/demo_<i>/actions` to count episodes
        and concluded a 50-demo file was empty.

        `H5Dopen2` returns a negative id for a missing path and the library's
        auto error printing is off (`h5native._init_h5_library`), so this is
        silent and allocates nothing on the miss.
        """
        var dset_id = h5d_open2(self.file_id, String(name), H5P_DEFAULT)
        if dset_id < 0:
            return False
        _ = h5d_close(dset_id)
        return True

    def open_dataset(self, var name: String) raises -> H5Dataset:
        """Open a dataset by path within the file (e.g. ``"pixels"``)."""
        var dset_id = h5d_open2(self.file_id, name^, H5P_DEFAULT)
        if dset_id < 0:
            raise Error(
                "H5Dopen2 failed: hid_t=" + String(Int(dset_id))
            )
        return H5Dataset(dset_id)


struct H5Dataset(Movable):
    """Owning handle to an open HDF5 dataset.

    On construction this reads the dataset's shape and computes the native
    type for in-memory reads. ``dims`` reports the full extent; ``elem_size``
    is bytes per element; ``dtype_class``/``signedness`` describe the dtype.
    """

    var dset_id: hid_t
    var native_type_id: hid_t
    var dims: List[hsize_t]
    var dtype_class: c_int
    """One of ``H5T_INTEGER`` / ``H5T_FLOAT`` / etc."""

    var elem_size: Int
    """Bytes per element (1 = u8/i8, 4 = i32/f32, 8 = i64/f64)."""

    var signedness: c_int
    """``H5T_SGN_NONE`` (unsigned) or ``H5T_SGN_2`` (signed). Only meaningful
    when ``dtype_class == H5T_INTEGER``."""

    def __init__(out self, dset_id: hid_t) raises:
        if dset_id < 0:
            raise Error("H5Dataset: invalid dset_id")
        self.dset_id = dset_id

        # ── shape ──────────────────────────────────────────────────────
        var space_id = h5d_get_space(dset_id)
        if space_id < 0:
            raise Error("H5Dget_space failed")
        var ndims_c = h5s_get_simple_extent_ndims(space_id)
        if ndims_c < 0:
            _ = h5s_close(space_id)
            raise Error("H5Sget_simple_extent_ndims failed")
        var ndims = Int(ndims_c)

        # `List` rather than `alloc`: every `h5*` call raises, so the manual
        # frees leaked these on each error path — and there were four exit
        # points to keep in sync. The lists free themselves on every exit.
        var dims_buf = List[hsize_t](length=ndims, fill=hsize_t(0))
        # HDF5 maxdims arg can be NULL ("don't return maxdims").
        _ = h5s_get_simple_extent_dims(
            space_id,
            dims_buf.unsafe_ptr(),
            _null_ptr[hsize_t, MutUntrackedOrigin](),
        )
        _ = h5s_close(space_id)

        var dims = List[hsize_t](capacity=ndims)
        for i in range(ndims):
            dims.append(dims_buf[i])
        self.dims = dims^

        # ── dtype (file → native) ──────────────────────────────────────
        var file_type_id = h5d_get_type(dset_id)
        if file_type_id < 0:
            raise Error("H5Dget_type failed")
        var native_type_id = h5t_get_native_type(
            file_type_id, H5T_DIR_DEFAULT
        )
        _ = h5t_close(file_type_id)
        if native_type_id < 0:
            raise Error("H5Tget_native_type failed")

        self.native_type_id = native_type_id
        self.dtype_class = h5t_get_class(native_type_id)
        self.elem_size = Int(h5t_get_size(native_type_id))
        if self.dtype_class == H5T_INTEGER:
            self.signedness = h5t_get_sign(native_type_id)
        else:
            self.signedness = H5T_SGN_NONE

    def __deinit__(deinit self):
        if self.dset_id > 0:
            try:
                _ = h5d_close(self.dset_id)
            except:
                pass
        if self.native_type_id > 0:
            try:
                _ = h5t_close(self.native_type_id)
            except:
                pass

    def ndim(self) -> Int:
        """Rank of the dataset."""
        return len(self.dims)

    def n_elements(self) -> Int:
        """Total element count = product of all dims."""
        var n = 1
        for i in range(len(self.dims)):
            n *= Int(self.dims[i])
        return n

    def n_bytes(self) -> Int:
        """Total in-memory size in bytes = ``n_elements() * elem_size``."""
        return self.n_elements() * self.elem_size

    def read_all[
        dtype: DType
    ](self, buf: Pointer[Scalar[dtype], MutAnyOrigin]) raises:
        """Read the entire dataset into ``buf``.

        ``buf`` must point to at least ``n_bytes()`` bytes. ``dtype`` must
        match the dataset's element size and class (caller's responsibility —
        check via ``dtype_class`` / ``elem_size`` / ``signedness``).
        """
        var ret = h5d_read(
            self.dset_id,
            self.native_type_id,
            H5S_ALL,
            H5S_ALL,
            H5P_DEFAULT,
            buf.unsafe_bitcast[NoneType](),
        )
        if ret < 0:
            raise Error("H5Dread failed: ret=" + String(Int(ret)))

    def read_range[
        dtype: DType
    ](
        self,
        start: Int,
        end: Int,
        buf: Pointer[Scalar[dtype], MutAnyOrigin],
    ) raises:
        """Read rows ``[start:end, :, :, ...]`` into ``buf``.

        For a 2D dataset of shape ``[N, F]``, reads a contiguous block
        of ``(end-start) * F`` elements. ``buf`` must be large enough.
        """
        var ndims = len(self.dims)
        if ndims == 0:
            raise Error("read_range: dataset is scalar (rank 0)")
        if start < 0 or end > Int(self.dims[0]) or end < start:
            raise Error(
                "read_range: out of bounds (start=" + String(start)
                + " end=" + String(end)
                + " dim0=" + String(Int(self.dims[0])) + ")"
            )

        # start_arr = [start, 0, 0, ...]; count_arr = [end-start, dim1, dim2, ...]
        # `List` rather than `alloc`: every `h5*` call raises, so the manual
        # frees leaked these on each error path — and there were four exit
        # points to keep in sync. The lists free themselves on every exit.
        var start_arr = List[hsize_t](length=ndims, fill=hsize_t(0))
        var count_arr = List[hsize_t](length=ndims, fill=hsize_t(0))
        start_arr[0] = hsize_t(start)
        count_arr[0] = hsize_t(end - start)
        for i in range(1, ndims):
            start_arr[i] = 0
            count_arr[i] = self.dims[i]

        var file_space = h5d_get_space(self.dset_id)
        if file_space < 0:
            raise Error("H5Dget_space failed")

        var stride_unit = List[hsize_t](length=ndims, fill=hsize_t(1))
        var block_unit = List[hsize_t](length=ndims, fill=hsize_t(1))
        var sel_ret = h5s_select_hyperslab(
            file_space,
            H5S_SELECT_SET,
            start_arr.unsafe_ptr(),
            stride_unit.unsafe_ptr(),
            count_arr.unsafe_ptr(),
            block_unit.unsafe_ptr(),
        )
        if sel_ret < 0:
            _ = h5s_close(file_space)
            raise Error("H5Sselect_hyperslab failed")

        var mem_space = h5s_create_simple(
            c_int(ndims),
            count_arr.unsafe_ptr(),
            count_arr.unsafe_ptr(),  # maxdims = dims (same as dims)
        )
        if mem_space < 0:
            _ = h5s_close(file_space)
            raise Error("H5Screate_simple failed")

        var read_ret = h5d_read(
            self.dset_id,
            self.native_type_id,
            mem_space,
            file_space,
            H5P_DEFAULT,
            buf.unsafe_bitcast[NoneType](),
        )
        _ = h5s_close(mem_space)
        _ = h5s_close(file_space)

        if read_ret < 0:
            raise Error("H5Dread failed: ret=" + String(Int(read_ret)))

    def read_strided[
        dtype: DType
    ](
        self,
        start: Int,
        count: Int,
        stride: Int,
        buf: Pointer[Scalar[dtype], MutAnyOrigin],
    ) raises:
        """Strided dim-0 read: select rows ``[start, start+stride, …,
        start+(count-1)*stride]`` and write them contiguously into ``buf``.

        Equivalent to ``dataset[start:start + count*stride : stride, …]``.
        For ``stride == 1`` use ``read_range`` (one hyperslab op vs two).

        ``buf`` must hold ``count * prod(dims[1:])`` elements of ``dtype``.
        """
        var ndims = len(self.dims)
        if ndims == 0:
            raise Error("read_strided: dataset is scalar")
        if stride <= 0:
            raise Error("read_strided: stride must be positive")
        if count <= 0:
            raise Error("read_strided: count must be positive")
        if start < 0 or start + (count - 1) * stride >= Int(self.dims[0]):
            raise Error(
                "read_strided: out of bounds (start=" + String(start)
                + " count=" + String(count)
                + " stride=" + String(stride)
                + " dim0=" + String(Int(self.dims[0])) + ")"
            )

        # Hyperslab: start=[start, 0, 0, ...], stride=[stride, 1, 1, ...],
        #            count=[count, 1, 1, ...], block=[1, dim1, dim2, ...]
        # `List` rather than `alloc`: every `h5*` call raises, so the manual
        # frees leaked these on each error path — and there were four exit
        # points to keep in sync. The lists free themselves on every exit.
        var start_arr = List[hsize_t](length=ndims, fill=hsize_t(0))
        var stride_arr = List[hsize_t](length=ndims, fill=hsize_t(0))
        var count_arr = List[hsize_t](length=ndims, fill=hsize_t(0))
        var block_arr = List[hsize_t](length=ndims, fill=hsize_t(0))
        start_arr[0] = hsize_t(start)
        stride_arr[0] = hsize_t(stride)
        count_arr[0] = hsize_t(count)
        block_arr[0] = 1
        # Memory-space shape: (count, dim1, dim2, ...) — the contiguous
        # destination layout after deinterleaving.
        var mem_dims = List[hsize_t](length=ndims, fill=hsize_t(0))
        mem_dims[0] = hsize_t(count)
        for i in range(1, ndims):
            start_arr[i] = 0
            stride_arr[i] = 1
            count_arr[i] = 1
            block_arr[i] = self.dims[i]
            mem_dims[i] = self.dims[i]

        var file_space = h5d_get_space(self.dset_id)
        if file_space < 0:
            raise Error("H5Dget_space failed")

        var sel_ret = h5s_select_hyperslab(
            file_space,
            H5S_SELECT_SET,
            start_arr.unsafe_ptr(),
            stride_arr.unsafe_ptr(),
            count_arr.unsafe_ptr(),
            block_arr.unsafe_ptr(),
        )
        if sel_ret < 0:
            _ = h5s_close(file_space)
            raise Error("H5Sselect_hyperslab (strided) failed")

        var mem_space = h5s_create_simple(
            c_int(ndims),
            mem_dims.unsafe_ptr(),
            mem_dims.unsafe_ptr(),  # maxdims = dims (same as dims)
        )
        if mem_space < 0:
            _ = h5s_close(file_space)
            raise Error("H5Screate_simple failed")

        var read_ret = h5d_read(
            self.dset_id,
            self.native_type_id,
            mem_space,
            file_space,
            H5P_DEFAULT,
            buf.unsafe_bitcast[NoneType](),
        )
        _ = h5s_close(mem_space)
        _ = h5s_close(file_space)

        if read_ret < 0:
            raise Error("H5Dread (strided) failed: ret=" + String(Int(read_ret)))
