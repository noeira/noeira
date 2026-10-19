"""`Tensor.tt` / `tt_at` (TileTensor) against `Tensor.lt` / `lt_at` (LayoutTensor).

The LayoutTensor migration moves every kernel from `t.lt["gpu", L]()` to
`t.tt["gpu", dims...]()`. This gates that the new views address the SAME
storage, in the same row-major order, on both legs:

  [1] cpu: `tt` and `lt` share a pointer, and `tt[i, j]` reads element i*N+j
  [2] gpu: `tt` and `lt` share a device pointer; a kernel taking `Tile[DT, B, N]`
      writes through `tt`, and the download sees row-major values
  [3] gpu: `tt_at(offset)` and `lt_at(offset)` share a pointer, and a kernel
      writing through `tt_at` lands at `offset` and nowhere else
"""

from layout import Layout
from max.gpu import thread_idx
from max.gpu.host import DeviceContext
from noeira.core.tiles import Tile
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
    var xl = x.lt["cpu", Layout.row_major(B, N)]()
    if Int(xt.ptr) != Int(xl.ptr):
        print("  FAIL [1] cpu tt / lt pointers differ")
        fails += 1
    for i in range(B):
        for j in range(N):
            if xt[i, j] != Scalar[DT](i * N + j):
                print("  FAIL [1] cpu tt[", i, ",", j, "] =", xt[i, j])
                fails += 1
    print("[1] cpu view: tt == lt storage, row-major reads")

    # [2] gpu
    var ctx = DeviceContext()
    var y = Tensor.alloc(B * N)
    y.ensure_gpu(ctx, B * N)
    y.upload(ctx)
    var yt = y.tt["gpu", B, N]()
    var yl = y.lt["gpu", Layout.row_major(B, N)]()
    if Int(yt.ptr) != Int(yl.ptr):
        print("  FAIL [2] gpu tt / lt pointers differ")
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
    var zl = z.lt_at["gpu", Layout.row_major(1, N)](OFF)
    if Int(zt.ptr) != Int(zl.ptr):
        print("  FAIL [3] gpu tt_at / lt_at pointers differ")
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
