"""Verify SIMD gains hold when using `LayoutTensor` (production access pattern).

The pointer-only bench (`benchmark_vectorize_cpu.mojo`) showed 3-12× speedups
from explicit SIMD. But noeira/nn/ uses `LayoutTensor[dtype, Layout.row_major(BATCH, DIM)]`
for indexing, not raw pointers. Two questions:

  Q1. Does the scalar `output[b, i] = ...` pattern in `ReLUOp.eval` already
      auto-vectorize? (If yes, no rollout needed for elementwise activations.)
  Q2. If not, does extracting `output.ptr` and doing manual SIMD restore the
      pointer-bench speedup?

We bench three implementations of ReLU and Mish forward (the cheap and the
worst-case-transcendental from the previous bench):

  A. scalar via LayoutTensor[B, D] indexing       (current production)
  B. scalar via .ptr offset                       (does autovec engage?)
  C. SIMD via .ptr load/store + tail              (what we'd ship)

Run:
    pixi run mojo run -I . benchmarks/benchmark_vectorize_layouttensor.mojo
"""

from std.math import tanh, exp, log
from std.memory import alloc
from std.random import seed, random_float64
from std.sys import simd_width_of
from std.time import perf_counter_ns
from noeira.core.tiles import Tile, tile

from noeira.nn.constants import DT as dtype


comptime SIMD_WIDTH = simd_width_of[dtype]()
comptime BATCH = 256
comptime DIM = 256


# =============================================================================
# A. LayoutTensor scalar — mirror of noeira/nn/autodiff/primitives/activations.mojo
# =============================================================================


def relu_lt_scalar(
    inp: Tile[dtype, BATCH, DIM],
    res: Tile[dtype, BATCH, DIM],
):
    for b in range(BATCH):
        for i in range(DIM):
            var v = inp[b, i]
            res[b, i] = v if v > 0 else 0


def mish_lt_scalar(
    inp: Tile[dtype, BATCH, DIM],
    res: Tile[dtype, BATCH, DIM],
):
    for b in range(BATCH):
        for i in range(DIM):
            var x = inp[b, i]
            res[b, i] = x * tanh(log(Scalar[dtype](1) + exp(x)))


# =============================================================================
# B. .ptr scalar — does autovec engage now?
# =============================================================================


def relu_ptr_scalar(
    inp: Pointer[Scalar[dtype], MutAnyOrigin],
    res: Pointer[Scalar[dtype], MutAnyOrigin],
    n: Int,
):
    for i in range(n):
        var v = inp[i]
        res[i] = v if v > 0 else 0


def mish_ptr_scalar(
    inp: Pointer[Scalar[dtype], MutAnyOrigin],
    res: Pointer[Scalar[dtype], MutAnyOrigin],
    n: Int,
):
    for i in range(n):
        var x = inp[i]
        res[i] = x * tanh(log(Scalar[dtype](1) + exp(x)))


# =============================================================================
# C. Manual SIMD on .ptr
# =============================================================================


def relu_ptr_simd(
    inp: Pointer[Scalar[dtype], MutAnyOrigin],
    res: Pointer[Scalar[dtype], MutAnyOrigin],
    n: Int,
):
    var i = 0
    var zero_v = SIMD[dtype, SIMD_WIDTH](0)
    while i + SIMD_WIDTH <= n:
        var v = inp.load[width=SIMD_WIDTH](i)
        var mask = v.gt(zero_v)
        res.store(i, mask.select(v, zero_v))
        i += SIMD_WIDTH
    while i < n:
        var v = inp[i]
        res[i] = v if v > 0 else 0
        i += 1


def mish_ptr_simd(
    inp: Pointer[Scalar[dtype], MutAnyOrigin],
    res: Pointer[Scalar[dtype], MutAnyOrigin],
    n: Int,
):
    var i = 0
    var one_v = SIMD[dtype, SIMD_WIDTH](1)
    while i + SIMD_WIDTH <= n:
        var x = inp.load[width=SIMD_WIDTH](i)
        res.store(i, x * tanh(log(one_v + exp(x))))
        i += SIMD_WIDTH
    var one = Scalar[dtype](1)
    while i < n:
        var x = inp[i]
        res[i] = x * tanh(log(one + exp(x)))
        i += 1


def main() raises:
    print("=" * 90)
    print("LayoutTensor vs .ptr-scalar vs .ptr-SIMD")
    print(
        "  BATCH=",
        BATCH,
        "  DIM=",
        DIM,
        "  N=",
        BATCH * DIM,
        "  SIMD_WIDTH=",
        SIMD_WIDTH,
    )
    print("=" * 90)

    var inp_buf = alloc[Scalar[dtype]](BATCH * DIM)
    var res_a = alloc[Scalar[dtype]](BATCH * DIM)
    var res_b = alloc[Scalar[dtype]](BATCH * DIM)
    var res_c = alloc[Scalar[dtype]](BATCH * DIM)
    seed(42)
    for i in range(BATCH * DIM):
        inp_buf[i] = Scalar[dtype](random_float64(-1.0, 1.0))

    var inp_lt = tile[BATCH, DIM](inp_buf)
    var res_a_lt = tile[BATCH, DIM](res_a)

    var iters = 1000

    # ----- ReLU -----
    print("\n--- ReLU (BATCH=", BATCH, "x DIM=", DIM, ") ---")

    relu_lt_scalar(inp_lt, res_a_lt)
    relu_ptr_scalar(inp_buf, res_b, BATCH * DIM)
    relu_ptr_simd(inp_buf, res_c, BATCH * DIM)

    var t0 = perf_counter_ns()
    for _ in range(iters):
        relu_lt_scalar(inp_lt, res_a_lt)
    var t1 = perf_counter_ns()
    var t_a = Float64(t1 - t0) / Float64(iters)

    t0 = perf_counter_ns()
    for _ in range(iters):
        relu_ptr_scalar(inp_buf, res_b, BATCH * DIM)
    t1 = perf_counter_ns()
    var t_b = Float64(t1 - t0) / Float64(iters)

    t0 = perf_counter_ns()
    for _ in range(iters):
        relu_ptr_simd(inp_buf, res_c, BATCH * DIM)
    t1 = perf_counter_ns()
    var t_c = Float64(t1 - t0) / Float64(iters)

    print(
        "LayoutTensor scalar (production) =",
        Int(t_a / 1000.0),
        "us  [baseline 1.00x]",
    )
    print(
        ".ptr scalar (autovec?)           =",
        Int(t_b / 1000.0),
        "us  [",
        Float64(Int(100 * t_a / t_b)) / 100.0,
        "x]",
    )
    print(
        ".ptr SIMD (manual)               =",
        Int(t_c / 1000.0),
        "us  [",
        Float64(Int(100 * t_a / t_c)) / 100.0,
        "x]",
    )

    # ----- Mish -----
    print("\n--- Mish (transcendental, BATCH=", BATCH, "x DIM=", DIM, ") ---")

    mish_lt_scalar(inp_lt, res_a_lt)
    mish_ptr_scalar(inp_buf, res_b, BATCH * DIM)
    mish_ptr_simd(inp_buf, res_c, BATCH * DIM)

    t0 = perf_counter_ns()
    for _ in range(iters):
        mish_lt_scalar(inp_lt, res_a_lt)
    t1 = perf_counter_ns()
    t_a = Float64(t1 - t0) / Float64(iters)

    t0 = perf_counter_ns()
    for _ in range(iters):
        mish_ptr_scalar(inp_buf, res_b, BATCH * DIM)
    t1 = perf_counter_ns()
    t_b = Float64(t1 - t0) / Float64(iters)

    t0 = perf_counter_ns()
    for _ in range(iters):
        mish_ptr_simd(inp_buf, res_c, BATCH * DIM)
    t1 = perf_counter_ns()
    t_c = Float64(t1 - t0) / Float64(iters)

    print(
        "LayoutTensor scalar (production) =",
        Int(t_a / 1000.0),
        "us  [baseline 1.00x]",
    )
    print(
        ".ptr scalar (autovec?)           =",
        Int(t_b / 1000.0),
        "us  [",
        Float64(Int(100 * t_a / t_b)) / 100.0,
        "x]",
    )
    print(
        ".ptr SIMD (manual)               =",
        Int(t_c / 1000.0),
        "us  [",
        Float64(Int(100 * t_a / t_c)) / 100.0,
        "x]",
    )

    print("\n" + "=" * 90)
    print("Done.")
    print("=" * 90)
