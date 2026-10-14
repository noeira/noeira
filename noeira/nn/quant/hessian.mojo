"""The calibration Hessian GPTQ needs: H = Σ x xᵀ over every row x a linear
reads during calibration, accumulated in fp64 on the host.

`HessianAccumulator.add` takes the linear's input slab as the forward saw it
([rows, n] row-major, on the host or the device) and adds Xᵀ X with
`dense.gemm_acc`. On a GPU the slab is downloaded first: a synchronization
per call, which calibration can afford (it runs once, off the hot path) and
which keeps one host implementation for every target. In fp64 because the
sums run over tens of thousands of rows whose magnitudes differ by orders
(outlier channels), and the Cholesky that follows is fp64 anyway.

The scale of H does not matter to GPTQ (its damping is relative), so this is
the plain sum, not the mean; `rows` counts what went in.
"""

from max.gpu.host import DeviceContext

from noeira.nn.constants import DT
from noeira.nn.core.tensor import TensorImpl

from .dense import gemm_acc


struct HessianAccumulator(Copyable, Movable):
    var n: Int
    var rows: Int
    var h: List[Float64]

    def __init__(out self):
        self.n = 0
        self.rows = 0
        self.h = List[Float64]()

    def __init__(out self, n: Int):
        self.n = n
        self.rows = 0
        self.h = List[Float64](length=n * n, fill=0)

    def add[
        target: StaticString
    ](
        mut self,
        mut x: TensorImpl[DT],
        rows: Int,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """H += Xᵀ X for the first `rows` rows of `x` ([rows, n] row-major).
        On a GPU target `x` is downloaded (synchronizes)."""
        comptime if target != "cpu":
            x.download(ctx.value())
        var n = self.n
        if len(x.data) < rows * n:
            raise Error(
                "HessianAccumulator.add: the slab holds " + String(len(x.data))
                + " values, [" + String(rows) + ", " + String(n) + "] need "
                + String(rows * n)
            )
        var xr = List[Float64](length=rows * n, fill=0)
        var xt = List[Float64](length=n * rows, fill=0)
        for r in range(rows):
            for c in range(n):
                var v = Float64(x.data[r * n + c])
                xr[r * n + c] = v
                xt[c * rows + r] = v
        gemm_acc[DType.float64](
            n,
            n,
            rows,
            1.0,
            rebind[Pointer[Float64, MutAnyOrigin]](xt.unsafe_ptr()),
            rows,
            rebind[Pointer[Float64, MutAnyOrigin]](xr.unsafe_ptr()),
            n,
            rebind[Pointer[Float64, MutAnyOrigin]](self.h.unsafe_ptr()),
            n,
        )
        self.rows += rows
