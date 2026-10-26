"""Row-major TileTensor types and constructors: what replaces LayoutTensor.

MAX 26.7 deprecates `LayoutTensor`, and modular `a65e584` removes it (with
`to_layout_tensor()` and `lt_to_tt`). These aliases give the native TileTensor
spelling of the form that covers ~75 % of our LayoutTensor sites:

    LayoutTensor[DT, Layout.row_major(B, N), MutAnyOrigin]   # before
    Tile[DT, B, N]                                           # after

`Tile[DT, B, N]` IS `TileTensor[DT, type_of(row_major[B, N]()), MutAnyOrigin]`,
the exact type a native `row_major[B, N]()` view has, so a kernel taking it
accepts every tensor built from that layout. Do not reach for `LTToTTLayout`:
`LTToTTLayout[Layout.row_major(B, N)]` is a DIFFERENT type, and a kernel
declared with it rejects a natively built tensor (checked on 1.2rc2).

Construction (`tile[B, N](x)`, dtype inferred) follows
docs/UNSAFE_ANY_ORIGIN_MIGRATION_PLAN.md: host memory (a `List` / `Span`, a
pointer) keeps its concrete origin and mutability, and a MUTABLE narrow view
converts implicitly to a `Tile` kernel argument at `enqueue_function`. Device
and pinned host BUFFERS are handles, so their views are `MutAnyOrigin`, the
kernel ABI, exactly as LayoutTensor's buffer constructor made them.

⚠ TileTensor traps (both silent):
- a SINGLE int index on a multi-dim tile splits `k` column-major: on
  `row_major[3, 4]`, `t[5]` is `t[2, 1]` (flat 9), not flat element 5.
  LayoutTensor rejected this at compile time; TileTensor accepts it. Flat
  reads go through `.ptr[k]`.
- `row_major[...]` assumes contiguous strides; a view over a section of a
  buffer with per-batch gaps reads into the next batch.
"""

from std.memory import stack_allocation as std_stack_allocation
from std.sys import align_of
from layout import Coord, TileTensor, row_major
from max.gpu.host import DeviceBuffer, HostBuffer
from max.gpu.memory import AddressSpace


comptime RowMajor[*dims: Int] = type_of(row_major[*dims]())
"""The layout type of `row_major[*dims]()`: fully static extents and strides.
`RowMajor[B, N]()` is its value."""

comptime DYN1 = type_of(row_major(Coord(Int(0))))
"""A 1-D layout whose extent is a RUNTIME value; `rl1(n)` is one. Replaces
`Layout.row_major[1]()` (UNKNOWN extent) + `RuntimeLayout[DYN1]`."""

comptime DYN2 = type_of(row_major(Coord(Int(0), Int(0))))
"""A 2-D row-major layout with RUNTIME extents; `rl2(rows, cols)` is one."""


@always_inline
def rl1(n: Int) -> DYN1:
    """A `DYN1` layout of live length `n`."""
    return row_major(Coord(n))


@always_inline
def rl2(rows: Int, cols: Int) -> DYN2:
    """A `DYN2` layout, row-major `[rows, cols]`. ⚠ `cols` is the row stride:
    a wrong width is a legal view over the wrong memory, not an error."""
    return row_major(Coord(rows, cols))


comptime Tile[dt: DType, *dims: Int] = TileTensor[
    dt, RowMajor[*dims], MutAnyOrigin
]
"""Mutable row-major view: the kernel-argument type replacing
`LayoutTensor[dt, Layout.row_major(*dims), MutAnyOrigin]`."""

comptime ImmTile[dt: DType, *dims: Int] = TileTensor[
    dt, RowMajor[*dims], ImmutAnyOrigin
]
"""Read-only row-major view (was `LayoutTensor[..., ImmutAnyOrigin]`)."""

comptime SharedTile[dt: DType, *dims: Int] = TileTensor[
    dt, RowMajor[*dims], MutAnyOrigin, address_space=AddressSpace.SHARED
]
"""A block's shared-memory tile (was `LayoutTensor[..., MutAnyOrigin,
address_space=AddressSpace.SHARED]`); allocate with `shared_tile`."""


@always_inline
def tile[
    dt: DType, //, *dims: Int
](buf: DeviceBuffer[dt]) -> Tile[dt, *dims]:
    """Row-major view of a device buffer: `tile[B, N](buf)` (was
    `LayoutTensor[dt, Layout.row_major(B, N), MutAnyOrigin](buf)`).

    The ONE origin erasure in this module, and the same one LayoutTensor's
    buffer constructor made: a `DeviceBuffer` is a HANDLE to device memory,
    so a read-only handle (every GPU env trait method takes
    `states_buf: DeviceBuffer[dtype]`) still names writable memory, and the
    kernel ABI is `MutAnyOrigin` anyway (see `Tensor.lt`). The buffer must
    outlive the kernels that use the view."""
    return Tile[dt, *dims](
        ptr=rebind[Pointer[Scalar[dt], MutAnyOrigin]](buf.unsafe_ptr()),
        layout=RowMajor[*dims](),
    )


@always_inline
def tile[
    dt: DType, //, *dims: Int
](buf: HostBuffer[dt]) -> Tile[dt, *dims]:
    """Row-major view of a pinned host buffer (a handle, like `DeviceBuffer`)."""
    return Tile[dt, *dims](
        ptr=rebind[Pointer[Scalar[dt], MutAnyOrigin]](buf.unsafe_ptr()),
        layout=RowMajor[*dims](),
    )


@always_inline
def tile[
    dt: DType, mut: Bool, origin: Origin[mut=mut], //, *dims: Int
](span: Span[Scalar[dt], origin]) -> TileTensor[dt, RowMajor[*dims], origin]:
    """Row-major view of host memory, e.g. a `List` (the CPU leg). Keeps the
    list's origin AND mutability: an immutable list gives a read-only view,
    which a `Tile` argument correctly rejects."""
    return TileTensor(span, RowMajor[*dims]())


@always_inline
def tile[
    dt: DType, mut: Bool, origin: Origin[mut=mut], //, *dims: Int
](ptr: Pointer[Scalar[dt], origin]) -> TileTensor[dt, RowMajor[*dims], origin]:
    """Row-major view at a pointer (was `LayoutTensor[dt, ...,
    MutAnyOrigin](ptr)`). The view keeps the pointer's origin: a
    `MutAnyOrigin` slab pointer gives a `Tile`."""
    return TileTensor(ptr=ptr, layout=RowMajor[*dims]())


@always_inline
def shared_tile[
    dt: DType, *dims: Int, alignment: Int = align_of[Scalar[dt]]()
]() -> SharedTile[dt, *dims]:
    """Allocate a shared-memory tile (was `LayoutTensor[dt,
    Layout.row_major(*dims), MutAnyOrigin,
    address_space=AddressSpace.SHARED].stack_allocation()`). Call it in a
    kernel body, like the stack allocation it replaces. `alignment` (bytes)
    is the allocation's, e.g. 16 for SIMD-width shared loads.

    Goes through `std.memory.stack_allocation` + a pointer view, not
    `layout.stack_allocation`: that one carries a `where
    LayoutType.all_dims_known` the compiler cannot prove through the variadic
    `dims` (1.2rc2), though every `RowMajor` is fully static."""
    comptime n = RowMajor[*dims]().size()
    var p = std_stack_allocation[
        n, Scalar[dt], alignment=alignment, address_space=AddressSpace.SHARED
    ]()
    # Stack memory lives for the kernel (shared: the block); the legacy tiles
    # were MutAnyOrigin too, so this is the same erasure, made explicit.
    return SharedTile[dt, *dims](
        ptr=p.as_unsafe_any_origin(), layout=RowMajor[*dims]()
    )


@always_inline
def local_tile[dt: DType, *dims: Int]() -> Tile[dt, *dims]:
    """Allocate a per-thread stack tile (was `LayoutTensor[dt,
    Layout.row_major(*dims), MutAnyOrigin].stack_allocation()`). Same route
    as `shared_tile`."""
    comptime n = RowMajor[*dims]().size()
    var p = std_stack_allocation[n, Scalar[dt]]()
    return Tile[dt, *dims](
        ptr=p.as_unsafe_any_origin(), layout=RowMajor[*dims]()
    )
