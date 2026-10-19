"""Fuse bias (+ activation, + the N-pad slice-back) into the GEMM epilogue.

`linalg.matmul` takes an `elementwise_lambda_fn` that runs on each output
tile BEFORE the store, and threads it into every kernel the dispatch can
pick — multistage, split-K's reduce, the cuBLASLt fallback, the naive
kernel, and Apple's 8x8 simdgroup kernel. We use it NOWHERE today
(`grep -r elementwise_lambda_fn noeira/` returns nothing), so every
`Linear` pays a second full-tensor kernel for its bias add
(`_bias_add_kernel` / `_bias_add_slice_kernel`, linear.mojo:51,215).

Folding the bias into the epilogue removes that launch and the
read-modify-write of the whole output. ACT issues ~46 GEMMs per training
step, so it is ~46 launches and ~46 output-sized RMWs per step.

Runs on Apple and NVIDIA.

    pixi run -e apple  mojo run -I . benchmarks/bench_matmul_epilogue_fusion.mojo
    pixi run -e nvidia mojo run -I . benchmarks/bench_matmul_epilogue_fusion.mojo

⚠ APPLE RESIDENCY HAZARD — see `residency_repro()` below. On an M1 Pro a
buffer reached ONLY from inside the epilogue closure reads as ZERO. Read
that function before adopting this anywhere.
"""

from std.sys import default_accelerator
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext
from linalg.matmul import matmul as max_matmul
from linalg.utils import elementwise_epilogue_type
from layout import TileTensor, row_major
from std.utils.index import IndexList

comptime M = 960
comptime K = 256
comptime N = 256
comptime DT = DType.float32


def residency_repro(ctx: DeviceContext) raises:
    """A buffer read only from the epilogue is not resident on Metal.

    Same program twice; the only difference is whether the bias buffer was
    ever `map_to_host()`ed before the GEMM. On an M1 Pro (MAX 26.5.0):

        without map_to_host   c = 0.0256          <- bias read as 0
        with    map_to_host   c = 0.0             <- relu(0.0256 - 0.1)

    An extra `ctx.synchronize()` does NOT fix it, and filling by
    `enqueue_copy` from a host buffer instead of `enqueue_fill` does not
    either — so this is not a fill-visibility problem. `a`, `b` and `c` are
    real kernel arguments and are always correct; `bias` is reached only
    through the captured closure. That is the discriminator: MAX's Metal
    backend appears not to mark an epilogue-captured buffer resident, and
    the kernel reads zeros with no error.

    NOT yet checked on CUDA. Check it there before trusting a fused bias.
    """
    var ab = ctx.enqueue_create_buffer[DT](64 * 128)
    var bb = ctx.enqueue_create_buffer[DT](128 * 64)
    var cb = ctx.enqueue_create_buffer[DT](64 * 64)
    var bias = ctx.enqueue_create_buffer[DT](64)
    ab.enqueue_fill(Float32(0.01))
    bb.enqueue_fill(Float32(0.02))
    cb.enqueue_fill(Float32(0.0))
    bias.enqueue_fill(Float32(-0.1))
    ctx.synchronize()

    var av = TileTensor(ab, row_major[64, 128]())
    var bv = TileTensor(bb, row_major[128, 64]())
    var cv = TileTensor(cb, row_major[64, 64]())
    var biasv = TileTensor(bias, row_major[64]())

    @__parameter
    @always_inline
    @__copy_capture(cv, biasv)
    def bias_relu[
        dtype: DType, width: SIMDLength, *, alignment: Int = 1
    ](coords: IndexList[2], val: SIMD[dtype, width]) capturing -> None:
        var out = val.cast[DT]()
        for i in range(width):
            out[i] = max(out[i] + rebind[Scalar[DT]](biasv[coords[1] + i]), 0.0)
        cv.store_linear[alignment=alignment](coords, out)

    max_matmul[
        target="gpu",
        elementwise_lambda_fn=Optional[elementwise_epilogue_type](bias_relu),
    ](cv, av, bv, ctx)
    ctx.synchronize()

    with cb.map_to_host() as h:
        var got = Float64(h[0])
        # 128 * 0.01 * 0.02 = 0.0256; relu(0.0256 - 0.1) == 0.0
        if got == 0.0:
            print("residency OK      — epilogue saw the bias (c[0] = 0.0)")
        else:
            print(
                "residency BROKEN  — epilogue read bias as 0 (c[0] =",
                got,
                ") — the captured-only buffer was not made resident",
            )


def main() raises:
    with DeviceContext() as ctx:
        print("--- epilogue residency check ---")
        residency_repro(ctx)

        print("--- fused vs unfused bias, [", M, "x", K, "] @ [", K, "x", N, "] ---")
        var ab = ctx.enqueue_create_buffer[DT](M * K)
        var bb = ctx.enqueue_create_buffer[DT](K * N)
        var cb = ctx.enqueue_create_buffer[DT](M * N)
        var bias = ctx.enqueue_create_buffer[DT](N)
        ab.enqueue_fill(Float32(0.01))
        bb.enqueue_fill(Float32(0.02))
        bias.enqueue_fill(Float32(-0.1))
        ctx.synchronize()
        # Force residency so the measurement is not also measuring the bug.
        with bias.map_to_host() as _hb:
            pass

        var av = TileTensor(ab, row_major[M, K]())
        var bv = TileTensor(bb, row_major[K, N]())
        var cv = TileTensor(cb, row_major[M, N]())
        var biasv = TileTensor(bias, row_major[N]())

        @__parameter
        @always_inline
        @__copy_capture(cv, biasv)
        def bias_add[
            dtype: DType, width: SIMDLength, *, alignment: Int = 1
        ](coords: IndexList[2], val: SIMD[dtype, width]) capturing -> None:
            var out = val.cast[DT]()
            for i in range(width):
                out[i] += rebind[Scalar[DT]](biasv[coords[1] + i])
            cv.store_linear[alignment=alignment](coords, out)

        comptime REPS = 200

        # unfused: GEMM, then a separate bias-add pass (what Linear does).
        for _ in range(5):
            max_matmul[target="gpu"](cv, av, bv, ctx)
        ctx.synchronize()
        var t0 = perf_counter_ns()
        for _ in range(REPS):
            max_matmul[target="gpu"](cv, av, bv, ctx)
        ctx.synchronize()
        var t1 = perf_counter_ns()

        # fused: one launch.
        for _ in range(5):
            max_matmul[
                target="gpu",
                elementwise_lambda_fn=Optional[elementwise_epilogue_type](
                    bias_add
                ),
            ](cv, av, bv, ctx)
        ctx.synchronize()
        var t2 = perf_counter_ns()
        for _ in range(REPS):
            max_matmul[
                target="gpu",
                elementwise_lambda_fn=Optional[elementwise_epilogue_type](
                    bias_add
                ),
            ](cv, av, bv, ctx)
        ctx.synchronize()
        var t3 = perf_counter_ns()

        print(
            "GEMM alone (bias-add kernel NOT counted)",
            Float64(t1 - t0) / 1000.0 / Float64(REPS),
            "us/call",
        )
        print(
            "GEMM + fused bias epilogue             ",
            Float64(t3 - t2) / 1000.0 / Float64(REPS),
            "us/call",
        )
        print(
            "the fused arm already includes the bias; the unfused arm still"
            " owes one full-tensor kernel."
        )
