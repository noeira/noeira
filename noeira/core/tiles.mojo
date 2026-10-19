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

Construction keeps a concrete origin (docs/UNSAFE_ANY_ORIGIN_MIGRATION_PLAN.md):
`tile[B, N](buf)` (dtype inferred) takes the buffer `mut` and returns a view at
`origin_of(buf)`. A MUTABLE narrow-origin view converts implicitly to a
`Tile` kernel argument at `enqueue_function`, so no `as_unsafe_any_origin()`
is needed at the launch. An immutable buffer gives an immutable view, which a
`MutAnyOrigin` kernel argument correctly rejects.

⚠ TileTensor traps (both silent):
- a SINGLE int index on a multi-dim tile returns `[k, 0, ...]`, not flat
  element `k`. Flat reads go through `.ptr[k]`.
- `row_major[...]` assumes contiguous strides; a view over a section of a
  buffer with per-batch gaps reads into the next batch.
"""

from std.memory import stack_allocation as std_stack_allocation
from layout import TileTensor, row_major
from max.gpu.host import DeviceBuffer, HostBuffer
from max.gpu.memory import AddressSpace


comptime RowMajor[*dims: Int] = type_of(row_major[*dims]())
"""The layout type of `row_major[*dims]()`: fully static extents and strides.
`RowMajor[B, N]()` is its value."""

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
](mut buf: DeviceBuffer[dt]) -> TileTensor[
    dt, RowMajor[*dims], origin_of(buf)
]:
    """Row-major view of a device buffer: `tile[B, N](buf)` (was
    `LayoutTensor[dt, Layout.row_major(B, N), MutAnyOrigin](buf)`)."""
    return TileTensor(buf, RowMajor[*dims]())


@always_inline
def tile[
    dt: DType, //, *dims: Int
](mut buf: HostBuffer[dt]) -> TileTensor[dt, RowMajor[*dims], origin_of(buf)]:
    """Row-major view of a pinned host buffer."""
    return TileTensor(buf, RowMajor[*dims]())


@always_inline
def tile[
    dt: DType, //, *dims: Int
](mut list: List[Scalar[dt]]) -> TileTensor[
    dt, RowMajor[*dims], origin_of(list)
]:
    """Row-major view of a host `List` (the CPU leg)."""
    return TileTensor(list, RowMajor[*dims]())


@always_inline
def tile[
    dt: DType, mut: Bool, origin: Origin[mut=mut], //, *dims: Int
](ptr: Pointer[Scalar[dt], origin]) -> TileTensor[dt, RowMajor[*dims], origin]:
    """Row-major view at a pointer (was `LayoutTensor[dt, ...,
    MutAnyOrigin](ptr)`). The view keeps the pointer's origin: a
    `MutAnyOrigin` slab pointer gives a `Tile`, a buffer's `unsafe_ptr()` a
    view tied to that buffer."""
    return TileTensor(ptr=ptr, layout=RowMajor[*dims]())


@always_inline
def shared_tile[dt: DType, *dims: Int]() -> SharedTile[dt, *dims]:
    """Allocate a shared-memory tile (was `LayoutTensor[dt,
    Layout.row_major(*dims), MutAnyOrigin,
    address_space=AddressSpace.SHARED].stack_allocation()`). Call it in a
    kernel body, like the stack allocation it replaces.

    Goes through `std.memory.stack_allocation` + a pointer view, not
    `layout.stack_allocation`: that one carries a `where
    LayoutType.all_dims_known` the compiler cannot prove through the variadic
    `dims` (1.2rc2), though every `RowMajor` is fully static."""
    comptime n = RowMajor[*dims]().size()
    var p = std_stack_allocation[
        n, Scalar[dt], address_space=AddressSpace.SHARED
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
