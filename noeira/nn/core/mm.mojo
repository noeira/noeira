"""`mm` / `bmm` — MAX's matmul behind ONE policy for how the operand views
are shaped, instead of that policy being spelled at ~60 call sites.

Two facts decide the view layout (docs/COMPILE_TIME_PROFILING.md §3.8):

  * A static `row_major[R, C]()` view instantiates every candidate of MAX's
    dispatch for that shape — ~10 Metal modules per matmul shape. Runtime
    extents, `row_major(R, C)`, share one set across shapes: on the ACT
    trainer 1101 -> 456 Metal modules, and on Apple the runtime form is
    also faster at every training batch (296 -> 163 us at [64x512]@
    [512x512]).
  * MAX's NVIDIA path is `_multistage_gemm_gpu`, which tiles the M axis
    at compile time and REFUSES a dynamic dimension ("Shouldn't split
    dynamic dimension"). The runtime views broke every GPU driver on the
    5090 (2026-09-11). And the Apple gemv specialisation is lost at one
    row (55 -> 90 us), which is the acting path.

So: static views on NVIDIA and at a single row, runtime views otherwise.
`MM_FORCE_STATIC` lets a gate compile the static branch on a box that
would take the runtime one, so both branches are built here.
"""
from std.sys import has_nvidia_gpu_accelerator
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import TileTensor, row_major
from linalg.matmul import matmul as max_matmul
from linalg.utils import elementwise_epilogue_type
from std.utils.index import IndexList
from linalg.bmm import batched_matmul


comptime MM_FORCE_STATIC: Bool = False
"""Set by a test to compile the static (NVIDIA) branch on an Apple box."""


@always_inline
def mm_static_views[rows: Int]() -> Bool:
    return has_nvidia_gpu_accelerator() or MM_FORCE_STATIC or rows == 1


@always_inline
def mm[
    transpose_b: Bool = False, *,
    A0: Int, A1: Int, B0: Int, B1: Int, O0: Int, O1: Int,
    dta: DType, dtb: DType, dto: DType,
](
    mut o: DeviceBuffer[dto],
    a: DeviceBuffer[dta],
    b: DeviceBuffer[dtb],
    c: DeviceContext,
) raises:
    """`o[O0, O1] = a[A0, A1] @ b[B0, B1]` (or `@ b^T`), row-major, on `c`.
    The six dims are the views' extents exactly as the call site had them."""
    comptime if mm_static_views[O0]():
        var av = TileTensor(a, row_major[A0, A1]())
        var bv = TileTensor(b, row_major[B0, B1]())
        var ov = TileTensor(o, row_major[O0, O1]())
        max_matmul[transpose_b=transpose_b, target="gpu"](ov, av, bv, c)
    else:
        var av = TileTensor(a, row_major(A0, A1))
        var bv = TileTensor(b, row_major(B0, B1))
        var ov = TileTensor(o, row_major(O0, O1))
        max_matmul[transpose_b=transpose_b, target="gpu"](ov, av, bv, c)


@always_inline
def mm_bias[
    *, A0: Int, A1: Int, B0: Int, B1: Int, O0: Int, O1: Int, dt: DType,
](
    mut o: DeviceBuffer[dt],
    a: DeviceBuffer[dt],
    b: DeviceBuffer[dt],
    bias: DeviceBuffer[dt],
    c: DeviceContext,
) raises:
    """`o[O0, O1] = a[A0, A1] @ b[B0, B1] + bias[O1]` in ONE launch: the bias
    is added in MAX's GEMM epilogue (`elementwise_lambda_fn`, run on each
    output tile before its store), so the separate bias pass — a full
    read-modify-write of the output — is gone.

    NVIDIA only. On Metal a buffer reached only through the epilogue closure
    is not made resident and reads as zeros with no error
    (`benchmarks/bench_matmul_epilogue_fusion.mojo`, `residency_repro`).
    The stored value is the GEMM's fp32 result plus the bias, the same
    operation the separate pass did."""
    comptime assert has_nvidia_gpu_accelerator(), (
        "mm_bias: the epilogue's captured bias is not resident on Metal"
    )
    var av = TileTensor(a, row_major[A0, A1]())
    var bv = TileTensor(b, row_major[B0, B1]())
    var ov = TileTensor(o, row_major[O0, O1]())
    var biasv = TileTensor(bias, row_major[O1]())

    @__parameter
    @always_inline
    @__copy_capture(ov, biasv)
    def _add_bias[
        dtype: DType, width: SIMDLength, *, alignment: Int = 1
    ](coords: IndexList[2], val: SIMD[dtype, width]) capturing -> None:
        var out = val.cast[dt]()
        comptime for i in range(width):
            out[i] += rebind[Scalar[dt]](biasv[coords[1] + i])
        ov.store_linear[alignment=alignment](coords, out)

    max_matmul[
        target="gpu",
        elementwise_lambda_fn=Optional[elementwise_epilogue_type](_add_bias),
    ](ov, av, bv, c)


@always_inline
def bmm[
    transpose_b: Bool = False, *,
    A0: Int, A1: Int, A2: Int, B0: Int, B1: Int, B2: Int,
    O0: Int, O1: Int, O2: Int,
    dta: DType, dtb: DType, dto: DType,
](
    mut o: DeviceBuffer[dto],
    a: DeviceBuffer[dta],
    b: DeviceBuffer[dtb],
    c: DeviceContext,
) raises:
    """Batched: `o[O0, O1, O2] = a[A0, A1, A2] @ b[B0, B1, B2]` (or `@ b^T`)
    per leading index."""
    comptime if mm_static_views[O1]():
        var av = TileTensor(a, row_major[A0, A1, A2]())
        var bv = TileTensor(b, row_major[B0, B1, B2]())
        var ov = TileTensor(o, row_major[O0, O1, O2]())
        batched_matmul[transpose_b=transpose_b, target="gpu"](
            ov, av, bv, context=c
        )
    else:
        var av = TileTensor(a, row_major(A0, A1, A2))
        var bv = TileTensor(b, row_major(B0, B1, B2))
        var ov = TileTensor(o, row_major(O0, O1, O2))
        batched_matmul[transpose_b=transpose_b, target="gpu"](
            ov, av, bv, context=c
        )
