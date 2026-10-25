"""`Dev` — what a GPU tape launches through: MAX's `DeviceContext`, or noeira's
own Metal launcher (`experimental/metal_rt`).

Every GPU function of the tape (`gpu.mojo`, `nd.mojo`, `extra.mojo`) takes a
`Dev` where it used to take a `DeviceContext`, and calls the same methods on
it — `enqueue_function[k](args…, grid_dim=, block_dim=)`, `enqueue_copy`,
`synchronize` — so the kernel call sites did not change. `Dev` forwards each
call to MAX, or, on a metal_rt tape, to `MetalRT`:

  - a `DeviceBuffer` argument becomes its device address (bound with
    `setBytes`), every other argument its own bytes;
  - storage is MetalRT's shared MTLBuffers, held in the tape's `Tensor`s as
    NON-OWNING `DeviceBuffer`s over their GPU address
    (`DeviceBuffer(ctx, ptr, n, owning=False)`), so the tape's storage type,
    `create_sub_buffer` views and `unsafe_ptr()` all work unchanged;
  - host -> device copies go through MetalRT's staging ring (ordered like a
    launch); device -> host copies sync, then read shared memory.

THE RULE: on a metal_rt tape no MAX launch, copy or fill may touch a tape
buffer — MAX does not know our MTLBuffers (it cannot make them resident in
its command buffers), and our command buffers do not know MAX's. So MAX's
matmul, nn's `Adam` and module ops are refused or replaced there:

  - matmul: the tape's own kernels (`gpu.mojo`, `own` path);
  - optimizer: `TapeAdam` (`optim.mojo`), nn's per-parameter Adam kernel
    launched through `Dev`;
  - modules as ops: unsupported (raise).

`own=True` on a MAX tape runs the same replacements on MAX's launch path:
the bit-identity gate compares that against the metal_rt tape.
"""

from std.builtin.device_passable import DevicePassable
from std.sys import size_of
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.fill import _fill_kernel, FILL_TPB
from noeira.nn.primitives.linear import (
    _lin_gb_partial_kernel,
    _lin_gb_final_kernel,
    _GB_CHUNK,
    _GB_COLS,
    _GB_ROWS,
)
from noeira.experimental.metal_rt.runtime import MetalRT


struct Dev(ImplicitlyCopyable):
    var ctx: DeviceContext
    var rt: Int
    """Address of the tape's `MetalRT` (owned by `TapeData`), 0 on MAX."""
    var own: Bool
    """Run the tape's own matmul instead of MAX's (always on a metal_rt
    tape; optional on MAX for the bit-identity gate)."""

    def __init__(out self, ctx: DeviceContext, rt: Int, own: Bool):
        self.ctx = ctx
        self.rt = rt
        self.own = own or rt != 0

    @always_inline
    def is_rt(self) -> Bool:
        return self.rt != 0

    @always_inline
    def _r(self) -> Pointer[MetalRT, MutAnyOrigin]:
        return Pointer[MetalRT, MutAnyOrigin](unsafe_from_address=self.rt)

    def _bind[*Ts: DevicePassable](self, *args: *Ts):
        var r = self._r()
        comptime for i in range(args.__len__()):
            comptime if conforms_to(Ts[i], TrivialRegisterPassable):
                r[].set_arg(i, Int(Pointer(to=args[i])), size_of[Ts[i]]())
            else:
                # The only non-trivial argument the tape passes: a buffer.
                var a = Int(rebind[DeviceBuffer[DT]](args[i]).unsafe_ptr())
                r[].set_arg(i, Int(Pointer(to=a)), 8)

    def enqueue_function[
        decl: TypeList[Trait=AnyType, ...], //,
        func: def(*args: *decl) thin -> None,
        *Ts: DevicePassable,
    ](self, *args: *Ts, grid_dim: Int, block_dim: Int) raises:
        if self.rt == 0:
            self.ctx.enqueue_function[func](
                *args, grid_dim=grid_dim, block_dim=block_dim
            )
            return
        var r = self._r()
        r[].begin_dispatch[func]()
        self._bind(*args)
        r[].dispatch(grid_dim, block_dim)

    def enqueue_function[
        decl: TypeList[Trait=AnyType, ...], //,
        func: def(*args: *decl) thin -> None,
        *Ts: DevicePassable,
    ](
        self, *args: *Ts, grid_dim: Tuple[Int, Int], block_dim: Tuple[Int, Int]
    ) raises:
        if self.rt == 0:
            self.ctx.enqueue_function[func](
                *args, grid_dim=grid_dim, block_dim=block_dim
            )
            return
        var r = self._r()
        r[].begin_dispatch[func]()
        self._bind(*args)
        r[].dispatch(grid_dim[0], block_dim[0], grid_dim[1], block_dim[1])

    def enqueue_copy(
        self, dst: DeviceBuffer[DT], src: Pointer[Scalar[DT], _]
    ) raises:
        """Host -> device, `len(dst)` elements."""
        if self.rt == 0:
            self.ctx.enqueue_copy(
                dst, Pointer[Scalar[DT], MutAnyOrigin](unsafe_from_address=Int(src))
            )
            return
        self._r()[].upload(
            Int(dst.unsafe_ptr()), Int(src), len(dst) * size_of[Scalar[DT]]()
        )

    def enqueue_copy(
        self, dst: Pointer[Scalar[DT], _], src: DeviceBuffer[DT]
    ) raises:
        """Device -> host, `len(src)` elements. On metal_rt this SYNCS (the
        tape always synchronises right after one anyway)."""
        if self.rt == 0:
            self.ctx.enqueue_copy(
                Pointer[Scalar[DT], MutAnyOrigin](unsafe_from_address=Int(dst)), src
            )
            return
        self._r()[].download(
            Int(dst), Int(src.unsafe_ptr()), len(src) * size_of[Scalar[DT]]()
        )

    def synchronize(self) raises:
        if self.rt == 0:
            self.ctx.synchronize()
        else:
            self._r()[].sync()

    def flush(self):
        """metal_rt: commit what is encoded without waiting (call before
        host work that does not need the GPU's result). No-op on MAX."""
        if self.rt != 0:
            self._r()[].flush()

    def create_buffer(self, n: Int) raises -> DeviceBuffer[DT]:
        if self.rt == 0:
            return self.ctx.enqueue_create_buffer[DT](n)
        var b = self._r()[].alloc[DT](n)
        return DeviceBuffer[DT](self.ctx, b.dev[DT](), n, owning=False)


def ensure_gpu(mut t: Tensor, d: Dev, n: Int) raises:
    """`Tensor.ensure_gpu` through `d`: grows only, not zero-filled."""
    if d.rt == 0:
        t.ensure_gpu(d.ctx, n)
        return
    if not t.dev or t.n < n:
        t.dev = d.create_buffer(n)
        t.n = n


def fill_dev(buf: DeviceBuffer[DT], n: Int, value: Scalar[DT], d: Dev) raises:
    """`nn.core.fill.fill_dev` — the same kernel — through `d`."""
    if n <= 0:
        return
    d.enqueue_function[_fill_kernel[DT]](
        buf, Int64(n), value,
        grid_dim=(n + FILL_TPB - 1) // FILL_TPB,
        block_dim=FILL_TPB,
    )


def enqueue_bias_grad[ADT: DType](
    c: Dev,
    go: DeviceBuffer[ADT],
    gb: DeviceBuffer[DT],
    B: Int,
    OUT: Int,
    mut part: Tensor,
) raises:
    """`nn.primitives.linear.enqueue_bias_grad` — the same two kernels, the
    same grids — through `c`: `gb += colsum(go)`, deterministic."""
    var chunks = (B + _GB_CHUNK - 1) // _GB_CHUNK
    ensure_gpu(part, c, chunks * OUT)
    c.enqueue_function[_lin_gb_partial_kernel[ADT]](
        go,
        part.dev.value(),
        Int64(B),
        Int64(OUT),
        grid_dim=((OUT + _GB_COLS - 1) // _GB_COLS, chunks),
        block_dim=(_GB_COLS, _GB_ROWS),
    )
    c.enqueue_function[_lin_gb_final_kernel](
        part.dev.value(),
        gb,
        Int64(chunks),
        Int64(OUT),
        grid_dim=(OUT + 255) // 256,
        block_dim=256,
    )
