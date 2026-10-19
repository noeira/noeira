"""`Tensor.tt` / `tt_at` / `tt_dyn`: the TileTensor views address the owning storage.

The LayoutTensor migration moved every kernel from `t.lt["gpu", L]()` to
`t.tt["gpu", dims...]()`. This gates that the views point at the tensor's own
storage, in row-major order, on both legs:

  [1] cpu: `tt` points at `data`, and `tt[i, j]` reads element i*N+j;
      `tt_dyn` with `rl2(B, N)` reads the same elements
  [2] gpu: `tt` points at the device buffer; a kernel taking `Tile[DT, B, N]`
      writes through it, and the download sees row-major values
  [3] gpu: `tt_at(offset)` points `offset` elements into the device buffer,
      and a kernel writing through it lands at the offset and nowhere else
"""

from std.sys import size_of
from max.gpu import thread_idx
from max.gpu.host import DeviceContext
from noeira.core.tiles import Tile, DYN2, rl2
from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor

comptime B = 3
comptime N = 5


def _write_rows(t: Tile[DT, B, N]):
    var i = Int(thread_idx.x)
    for j in range(N):
        t[i, j] = Scalar[DT](100 * i + j)


def _write_row(t: Tile[DT, 1, N]):
    var j = Int(thread_idx.x)
    t[0, j] = Scalar[DT](-1 - j)


def main() raises:
    var fails = 0

    # [1] cpu
    var x = Tensor.alloc(B * N)
    for k in range(B * N):
        x.data[k] = Scalar[DT](k)
    var xt = x.tt["cpu", B, N]()
    var xd = x.tt_dyn["cpu", DYN2](rl2(B, N))
    if Int(xt.ptr) != Int(x.data.unsafe_ptr()):
        print("  FAIL [1] cpu tt does not point at data")
        fails += 1
    for i in range(B):
        for j in range(N):
            if xt[i, j] != Scalar[DT](i * N + j):
                print("  FAIL [1] cpu tt[", i, ",", j, "] =", xt[i, j])
                fails += 1
            if xd[i, j] != Scalar[DT](i * N + j):
                print("  FAIL [1] cpu tt_dyn[", i, ",", j, "] =", xd[i, j])
                fails += 1
    print("[1] cpu view: tt / tt_dyn on data, row-major reads")

    # [2] gpu
    var ctx = DeviceContext()
    var y = Tensor.alloc(B * N)
    y.ensure_gpu(ctx, B * N)
    y.upload(ctx)
    var yt = y.tt["gpu", B, N]()
    if Int(yt.ptr) != Int(y.dev.value().unsafe_ptr()):
        print("  FAIL [2] gpu tt does not point at the device buffer")
        fails += 1
    ctx.enqueue_function[_write_rows](yt, grid_dim=1, block_dim=B)
    y.download(ctx)
    for i in range(B):
        for j in range(N):
            if y.data[i * N + j] != Scalar[DT](100 * i + j):
                print("  FAIL [2] gpu y[", i * N + j, "] =", y.data[i * N + j])
                fails += 1
    print("[2] gpu view: kernel through tt writes row-major")

    # [3] gpu offset view
    comptime OFF = N  # the second row
    var z = Tensor.alloc(B * N)
    z.ensure_gpu(ctx, B * N)
    z.upload(ctx)
    var zt = z.tt_at["gpu", 1, N](OFF)
    var want_ptr = Int(z.dev.value().unsafe_ptr()) + OFF * size_of[Scalar[DT]]()
    if Int(zt.ptr) != want_ptr:
        print("  FAIL [3] gpu tt_at is not OFF elements into the buffer")
        fails += 1
    ctx.enqueue_function[_write_row](zt, grid_dim=1, block_dim=N)
    z.download(ctx)
    for k in range(B * N):
        var want = Scalar[DT](-1 - (k - OFF)) if OFF <= k < OFF + N else 0
        if z.data[k] != want:
            print("  FAIL [3] z[", k, "] =", z.data[k], "want", want)
            fails += 1
    print("[3] gpu offset view: tt_at writes at the offset only")

    if fails > 0:
        raise Error(String(fails, " failure(s)"))
    print("test_tensor_tt_accessor: ALL PASS")
