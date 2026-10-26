# +--------------------------------------------------------------------------+ #
# | Residency + gather — the sampling path
# +--------------------------------------------------------------------------+ #
"""A store column held in memory, and gather-by-index over it.

**Why residency is not an optimisation.** Measured cost of assembling one
4096-row minibatch (`docs/DATA_PLATFORM_PLAN.md` §3b):

    scattered rows, one read per row, from disk    272 ms
    scattered rows in 64-row groups, from disk    2.16 ms
    contiguous random slab, from disk            0.072 ms
    scattered rows from a resident host buffer   0.060 ms

Row-by-row gather from HDF5 is ~4500x worse than from RAM, so sampling
*requires* the column to be resident. `TrajectoryStore.read_range` stays the
streaming path; this is the sampling path.

**Whole-store residency first** (regime A). Every dataset we generate fits —
walker at 10 M transitions is 992 MiB against 16 GiB of host RAM and 24 GiB on
a 4090. Windowed streaming for the datasets that do NOT fit (PushT pixels at
44 GB) is deferred; building the general mechanism first would be paying for a
case the near-term consumers never hit.

**Runtime dims.** `n_rows` and `row_dim` are store metadata, not compile-time
constants, so the device path uses the runtime-shaped `LayoutTensor` idiom —
`Layout.row_major[N]()` with UNKNOWN extents plus a `RuntimeLayout` carrying
the real ones. That spelling is verified on Metal in
`docs/PHYSICS3D_RUNTIME_DIMS_ASSESSMENT.md` §2.2 and is the direction
physics3d's CPU leg is already committed to; inventing a raw-pointer kernel
ABI here would have been a second convention for no gain.

⚠ **Metal has a Float64 wall.** A `float64` column gathers fine on the host
and will fail to compile/launch on Apple GPU — the same wall documented in
`feedback_metal_nested_generics`, not a limitation of this code. Store state
columns as `float32`.
"""

from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext, DeviceBuffer
from std.memory import alloc
from std.utils import IndexList
from layout import TileTensor

from noeira.nn.constants import TPB
from noeira.nn.core.ptr import mptr
from .column import ColumnSpec, dtype_bytes
from .store import TrajectoryStore


from noeira.core.tiles import DYN1, DYN2, rl1, rl2

comptime IDX_DT = DType.int32
"""Row indices are int32: a store beyond 2^31 rows would be ~200 GB of state
columns, far past the residency regime this path serves."""


@always_inline
def _dyn2[
    dtype: DType
](
    buf: DeviceBuffer[dtype], rows: Int, cols: Int
) -> TileTensor[dtype, DYN2, MutAnyOrigin]:
    return TileTensor[dtype, DYN2, MutAnyOrigin](
        ptr=mptr(buf.unsafe_ptr()), layout=rl2(rows, cols)
    )


@always_inline
def _dyn1[
    dtype: DType
](buf: DeviceBuffer[dtype], n: Int) -> TileTensor[dtype, DYN1, MutAnyOrigin]:
    return TileTensor[dtype, DYN1, MutAnyOrigin](
        ptr=mptr(buf.unsafe_ptr()), layout=rl1(n)
    )


# ── kernel ────────────────────────────────────────────────────────────────

def _gather_rows_kernel[
    dtype: DType
](
    src: TileTensor[dtype, DYN2, MutAnyOrigin],
    idx: TileTensor[IDX_DT, DYN1, MutAnyOrigin],
    dst: TileTensor[dtype, DYN2, MutAnyOrigin],
):
    """One thread per (lane, element): `dst[i, d] = src[idx[i], d]`.

    Element-parallel rather than one-thread-per-lane, for the reason recorded
    in `gpu_replay.mojo`'s gather: a per-lane kernel serialises the row copy
    inside each thread and launches only BATCH threads, which cost ~73% of GPU
    time on wide (pixel) rows.

    Shapes ride in the `RuntimeLayout`, so `row_dim` is read off the tensor
    rather than baked in as a comptime parameter.
    """
    var batch = Int(dst.dim[0]())
    var row_dim = Int(dst.dim[1]())
    var t = Int(block_dim.x * block_idx.x + thread_idx.x)
    if t >= batch * row_dim:
        return
    var i = t // row_dim
    var d = t % row_dim
    var r = Int(idx[i])
    dst[i, d] = src[r, d]


# ── index batch ───────────────────────────────────────────────────────────

struct IndexBatch(Movable & Deinitable):
    """A batch of row indices, host-side with an optional device mirror.

    Stage 3's samplers (uniform / PER / n-step / sequence-window / FB-dual)
    produce these; the store knows nothing about how they were chosen. That
    split is what stops every {policy x backend x column set} combination from
    needing its own buffer type.
    """

    var host: List[Scalar[IDX_DT]]
    var dev: Optional[DeviceBuffer[IDX_DT]]
    var dev_len: Int
    """Elements currently staged on device. `DeviceBuffer` exposes no length,
    so capacity is tracked here — the same thing `nn.core.Tensor` does."""

    def __init__(out self):
        self.host = List[Scalar[IDX_DT]]()
        self.dev = None
        self.dev_len = 0

    def __init__(out self, var host: List[Scalar[IDX_DT]]):
        self.host = host^
        self.dev = None
        self.dev_len = 0

    def __init__(out self, *, deinit move: Self):
        self.host = move.host^
        self.dev = move.dev^
        self.dev_len = move.dev_len

    def size(self) -> Int:
        return len(self.host)

    def to_device(mut self, ctx: DeviceContext) raises:
        """Stage the indices on device, allocating once and reusing after."""
        var n = len(self.host)
        if n == 0:
            raise Error("IndexBatch.to_device: empty batch")
        if not self.dev or self.dev_len != n:
            self.dev = ctx.enqueue_create_buffer[IDX_DT](n)
            self.dev_len = n
        ctx.enqueue_copy(self.dev.value(), self.host.unsafe_ptr())


# ── resident column ───────────────────────────────────────────────────────

struct ResidentColumn[dtype: DType](Movable & Deinitable):
    """One whole column of a store, held in host memory and optionally
    mirrored on device.

    Typed on `dtype` because Mojo needs a comptime dtype to type the buffer;
    `load` checks it against the store's registered spec, so a mismatch raises
    rather than reinterpreting bytes.
    """

    var host: List[Scalar[Self.dtype]]
    var dev: Optional[DeviceBuffer[Self.dtype]]
    var dev_len: Int
    var n_rows: Int
    var row_dim: Int
    var name: String

    def __init__(
        out self,
        var host: List[Scalar[Self.dtype]],
        n_rows: Int,
        row_dim: Int,
        var name: String,
    ):
        self.host = host^
        self.dev = None
        self.dev_len = 0
        self.n_rows = n_rows
        self.row_dim = row_dim
        self.name = name^

    def __init__(out self, *, deinit move: Self):
        self.host = move.host^
        self.dev = move.dev^
        self.dev_len = move.dev_len
        self.n_rows = move.n_rows
        self.row_dim = move.row_dim
        self.name = move.name^

    @staticmethod
    def load(
        store: TrajectoryStore,
        name: String,
        max_bytes: Int = 8 * 1024 * 1024 * 1024,
    ) raises -> Self:
        """Load a whole column into host memory."""
        var spec = store.column(name)
        var host = store.load_column[Self.dtype](name, max_bytes)
        return Self(host^, store.n_rows(), spec.row_dim(), String(name))

    def n_elements(self) -> Int:
        return self.n_rows * self.row_dim

    def to_device(mut self, ctx: DeviceContext) raises:
        """Mirror the column onto the device."""
        var n = self.n_elements()
        if not self.dev or self.dev_len != n:
            self.dev = ctx.enqueue_create_buffer[Self.dtype](n)
            self.dev_len = n
        ctx.enqueue_copy(self.dev.value(), self.host.unsafe_ptr())

    def _check_indices(self, ref idx: IndexBatch) raises:
        """Bounds-check every index once, up front.

        The kernel deliberately does NOT range-check: an out-of-range row
        would read arbitrary device memory and produce plausible garbage
        rather than a fault. Catching it here makes it an error with a name.
        """
        var n = len(idx.host)
        if n == 0:
            raise Error("gather: empty index batch")
        for i in range(n):
            var r = Int(idx.host[i])
            if r < 0 or r >= self.n_rows:
                raise Error(
                    "gather: index[" + String(i) + "] = " + String(r)
                    + " out of range for column '" + self.name + "' with "
                    + String(self.n_rows) + " rows"
                )

    def gather_host(
        self, ref idx: IndexBatch, mut out: List[Scalar[Self.dtype]]
    ) raises:
        """`out[i, :] = host[idx[i], :]`. Resizes `out` to `batch * row_dim`."""
        self._check_indices(idx)
        var batch = len(idx.host)
        var need = batch * self.row_dim
        if len(out) != need:
            out = List[Scalar[Self.dtype]](unsafe_uninit_length=need)
        for i in range(batch):
            var r = Int(idx.host[i])
            var src = r * self.row_dim
            var dst = i * self.row_dim
            for d in range(self.row_dim):
                out[dst + d] = self.host[src + d]

    def gather_device(
        mut self,
        ctx: DeviceContext,
        mut idx: IndexBatch,
        dst: DeviceBuffer[Self.dtype],
        dst_elems: Int,
    ) raises:
        """`dst[i, :] = dev[idx[i], :]` on device.

        Mirrors the column and the indices first if they are not already
        staged. `dst` must hold `batch * row_dim` elements. (Named `dst`, not
        `out`: `out` is an argument-convention keyword in Mojo.)
        """
        self._check_indices(idx)
        var batch = len(idx.host)
        var need = batch * self.row_dim
        if dst_elems < need:
            raise Error(
                "gather_device: dst buffer holds " + String(dst_elems)
                + " elements, needs " + String(need)
            )
        if not self.dev:
            self.to_device(ctx)
        if not idx.dev or idx.dev_len != batch:
            idx.to_device(ctx)

        var src_lt = _dyn2[Self.dtype](self.dev.value(), self.n_rows, self.row_dim)
        var idx_lt = _dyn1[IDX_DT](idx.dev.value(), batch)
        var out_lt = _dyn2[Self.dtype](dst, batch, self.row_dim)

        var n_threads = need
        var n_blocks = (n_threads + TPB - 1) // TPB
        comptime kernel = _gather_rows_kernel[Self.dtype]
        ctx.enqueue_function[kernel](
            src_lt, idx_lt, out_lt, grid_dim=n_blocks, block_dim=TPB
        )

    def gather_device_to_host(
        mut self,
        ctx: DeviceContext,
        mut idx: IndexBatch,
        mut out: List[Scalar[Self.dtype]],
    ) raises:
        """Device gather with the result read back — for gating the device
        path against the host path, not for the training loop."""
        var batch = len(idx.host)
        var need = batch * self.row_dim
        var dev_out = ctx.enqueue_create_buffer[Self.dtype](need)
        self.gather_device(ctx, idx, dev_out, need)
        if len(out) != need:
            out = List[Scalar[Self.dtype]](unsafe_uninit_length=need)
        ctx.enqueue_copy(out.unsafe_ptr(), dev_out)
        ctx.synchronize()
