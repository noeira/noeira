# +--------------------------------------------------------------------------+ #
# | The two attention matmuls — MAX's batched_matmul vs tiled kernels
# +--------------------------------------------------------------------------+ #
"""What the Q.Kt and A.V products cost, and whether a tiled kernel beats them.

    pixi run -e jetson attn-matmul-bench-jetson
    pixi run -e apple mojo run -I . benchmarks/attention_matmul_bench.mojo

⚠⚠ WHY. After the softmax port (`cross_attention.mojo`, 18 Sep) the two
matmuls ARE the SigLIP attention layer on the Orin: 8.67 + 8.69 ms of ~22,
about 420 ms of a ~980 ms SmolVLA query across 24 tower layers. Each is
1.6 GFLOP in 8.7 ms = 186 GFLOPS, roughly 10% of the board's fp32 peak —
the same "it works, but at a tenth of the machine" signature the softmax had.

⚠ THE TWO PRODUCTS HAVE OPPOSITE SHAPES, which is why both are measured:

    scores = Q . Kt    (BH, M=QL, K=HD,  N=KL)   K is TINY (64), N wide
    out    = A . V     (BH, M=QL, K=KL,  N=HD)   K is WIDE (1024), N tiny

A tile size that suits one may be wrong for the other, and `batched_matmul`'s
own dispatch is shape-dependent: MAX's multistage GEMM wants `n % 128 == 0`
and `k >= 128`, so BOTH of these fall to its vendor path.

    M0  bmm — what ships today
    M1  shared-memory tiles, 16x16, one output per thread
    M2  shared-memory tiles, 64x64 per block, 4x4 outputs per thread

⚠ CORRECTNESS, AND WHY THE REFERENCE IS SAMPLED. A float64 reference for every
output would be 1.6 G MACs per shape in scalar float64 — minutes of CPU per
run, for a benchmark. Instead: a float64 reference over SAMPLED output
elements (`N_SAMPLE` of them, drawn across the whole output), plus a FULL
comparison against M0 in std units and in bits. The sample catches a variant
that computes the wrong thing; the full M0 comparison catches a variant that
computes it right in the sampled corner and wrong elsewhere. Destinations are
pre-poisoned, so a kernel that writes nothing fails instead of inheriting M0's
answer from reused memory.
"""

from std.math import sqrt
from std.sys import has_accelerator
from std.time import perf_counter_ns

from max.gpu import barrier, block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext
from max.gpu.memory import AddressSpace
from noeira.core.tiles import RowMajor, Tile, shared_tile

from noeira.nn.constants import DT
from noeira.nn.core.mm import bmm
from noeira.nn.core.tensor import Tensor


comptime WARMUP = 3
comptime REPS = 10
comptime GATE_STD_UNITS = 1.0e-4
comptime POISON = 12345.0
comptime N_SAMPLE = 2048

comptime T1 = 16
"""M1: tile edge, one output per thread (256 threads per block)."""
comptime T2_M = 64
comptime T2_N = 64
comptime T2_K = 16
comptime T2_TM = 4
comptime T2_TN = 4
"""M2: a 64x64 output tile per block, 16x16 threads, 4x4 outputs each."""


# ═══════════════════════════════════════════════════════════════════════════
# candidates
# ═══════════════════════════════════════════════════════════════════════════


def _mm_tiled_kernel[BH: Int, M: Int, N: Int, K: Int](
    a: Tile[DT, BH * M * K],
    b: Tile[DT, BH * K * N],
    o: Tile[DT, BH * M * N],
):
    """M1: one output per thread, both operands staged through shared memory."""
    var sa = shared_tile[DT, T1, T1]()
    var sb = shared_tile[DT, T1, T1]()

    var bh = Int(block_idx.z)
    var ty = Int(thread_idx.y)
    var tx = Int(thread_idx.x)
    var row = Int(block_idx.y) * T1 + ty
    var col = Int(block_idx.x) * T1 + tx
    var a_base = bh * M * K
    var b_base = bh * K * N
    var acc = Scalar[DT](0)

    for kt in range(0, K, T1):
        var ak = kt + tx
        sa[ty, tx] = (
            rebind[Scalar[DT]](a.ptr[unsafe_offset=a_base + row * K + ak])
            if (row < M and ak < K) else Scalar[DT](0)
        )
        var bk = kt + ty
        sb[ty, tx] = (
            rebind[Scalar[DT]](b.ptr[unsafe_offset=b_base + bk * N + col])
            if (bk < K and col < N) else Scalar[DT](0)
        )
        barrier()
        for kk in range(T1):
            acc += rebind[Scalar[DT]](sa[ty, kk]) * rebind[Scalar[DT]](
                sb[kk, tx]
            )
        barrier()

    if row < M and col < N:
        o.ptr[unsafe_offset=bh * M * N + row * N + col] = acc


def _mm_tiled4_kernel[BH: Int, M: Int, N: Int, K: Int](
    a: Tile[DT, BH * M * K],
    b: Tile[DT, BH * K * N],
    o: Tile[DT, BH * M * N],
):
    """M2: 4x4 outputs per thread — each shared-memory element is reused four
    times from registers instead of once, which is what lifts a tiled matmul
    off the shared-memory bandwidth bound."""
    var sa = shared_tile[DT, T2_M, T2_K]()
    var sb = shared_tile[DT, T2_K, T2_N]()

    var bh = Int(block_idx.z)
    var ty = Int(thread_idx.y)
    var tx = Int(thread_idx.x)
    var tid = ty * 16 + tx
    var m0 = Int(block_idx.y) * T2_M
    var n0 = Int(block_idx.x) * T2_N
    var a_base = bh * M * K
    var b_base = bh * K * N

    var acc = Array[Scalar[DT], T2_TM * T2_TN](fill=Scalar[DT](0))

    for kt in range(0, K, T2_K):
        # A tile: T2_M x T2_K = 1024 elements, 256 threads, 4 each.
        comptime for l in range(T2_M * T2_K // 256):
            var idx = tid + l * 256
            var r = idx // T2_K
            var cc = idx % T2_K
            sa[r, cc] = (
                rebind[Scalar[DT]](
                    a.ptr[unsafe_offset=a_base + (m0 + r) * K + kt + cc]
                )
                if (m0 + r < M and kt + cc < K) else Scalar[DT](0)
            )
        # B tile: T2_K x T2_N = 1024 elements.
        comptime for l in range(T2_K * T2_N // 256):
            var idx = tid + l * 256
            var r = idx // T2_N
            var cc = idx % T2_N
            sb[r, cc] = (
                rebind[Scalar[DT]](
                    b.ptr[unsafe_offset=b_base + (kt + r) * N + n0 + cc]
                )
                if (kt + r < K and n0 + cc < N) else Scalar[DT](0)
            )
        barrier()
        for kk in range(T2_K):
            comptime for i in range(T2_TM):
                var av = rebind[Scalar[DT]](sa[ty * T2_TM + i, kk])
                comptime for j in range(T2_TN):
                    acc[i * T2_TN + j] += av * rebind[Scalar[DT]](
                        sb[kk, tx * T2_TN + j]
                    )
        barrier()

    comptime for i in range(T2_TM):
        var r = m0 + ty * T2_TM + i
        comptime for j in range(T2_TN):
            var cc = n0 + tx * T2_TN + j
            if r < M and cc < N:
                o.ptr[unsafe_offset=bh * M * N + r * N + cc] = acc[
                    i * T2_TN + j
                ]


# ═══════════════════════════════════════════════════════════════════════════
# host
# ═══════════════════════════════════════════════════════════════════════════


struct Lcg(Movable):
    var s: UInt64

    def __init__(out self, seed: Int):
        self.s = UInt64(seed * 2 + 1)

    def __init__(out self, *, deinit move: Self):
        self.s = move.s

    def unit(mut self) -> Float32:
        self.s = self.s * UInt64(6364136223846793005) + UInt64(
            1442695040888963407
        )
        return Float32(
            Float64((self.s >> 40) & UInt64(0xFFFFFF)) / 16777216.0
        )

    def below(mut self, n: Int) -> Int:
        return Int(Float64(self.unit()) * Float64(n)) % n


def _fmt(x: Float64, digits: Int) -> String:
    var p = 1.0
    for _ in range(digits):
        p *= 10.0
    return String(Float64(Int(x * p + 0.5)) / p)


def _pad(s: String, w: Int) -> String:
    var out = s
    while out.byte_length() < w:
        out += " "
    return out^


def run_gemm[
    BH: Int, M: Int, N: Int, K: Int
](name: String, mut ctx: DeviceContext) raises -> Bool:
    comptime AN = BH * M * K
    comptime BN = BH * K * N
    comptime ON = BH * M * N
    comptime FLOP = 2.0 * Float64(BH) * Float64(M) * Float64(N) * Float64(K)

    comptime lay_a = RowMajor[AN]
    comptime lay_b = RowMajor[BN]
    comptime lay_o = RowMajor[ON]

    var rng = Lcg(BH * 7919 + M * 131 + N * 17 + K)
    var a = Tensor.alloc(AN)
    var b = Tensor.alloc(BN)
    for n in range(AN):
        a.data[n] = Scalar[DT](rng.unit() * 2.0 - 1.0)
    for n in range(BN):
        b.data[n] = Scalar[DT](rng.unit() * 2.0 - 1.0)
    a.upload(ctx)
    b.upload(ctx)

    # float64 reference over sampled outputs (see the header)
    var s_idx = List[Int](length=N_SAMPLE, fill=0)
    var s_ref = List[Float64](length=N_SAMPLE, fill=0.0)
    for t in range(N_SAMPLE):
        var bh = rng.below(BH)
        var i = rng.below(M)
        var j = rng.below(N)
        var acc = 0.0
        for k in range(K):
            acc += Float64(a.data[bh * M * K + i * K + k]) * Float64(
                b.data[bh * K * N + k * N + j]
            )
        s_idx[t] = bh * M * N + i * N + j
        s_ref[t] = acc

    var names: List[String] = [
        String("M0  bmm (MAX batched_matmul)"),
        String("M1  tiled 16x16, 1 out/thread"),
        String("M2  tiled 64x64, 4x4 out/thread"),
    ]
    print("")
    print(
        "── " + name + ": BH " + String(BH) + ", M " + String(M) + " x N "
        + String(N) + " x K " + String(K) + "  ("
        + _fmt(FLOP / 1.0e9, 2) + " GFLOP) ──"
    )
    print(
        "   " + _pad(String("variant"), 34) + _pad(String("best ms"), 10)
        + _pad(String("vs M0"), 8) + _pad(String("GFLOP/s"), 10)
        + _pad(String("sampled (std u.)"), 26) + "bits != M0"
    )

    var ok = True
    var best_m0 = 0.0
    var m0_out = List[Scalar[DT]]()
    for variant in range(3):
        var o = Tensor()
        o.ensure_gpu(ctx, ON)
        o.dev.value().enqueue_fill(Scalar[DT](POISON))
        var best = 1.0e30
        for rep in range(WARMUP + REPS):
            ctx.synchronize()
            var t0 = perf_counter_ns()
            if variant == 0:
                bmm[
                    A0=BH, A1=M, A2=K, B0=BH, B1=K, B2=N, O0=BH, O1=M, O2=N
                ](o.dev.value(), a.dev.value(), b.dev.value(), ctx)
            elif variant == 1:
                ctx.enqueue_function[_mm_tiled_kernel[BH, M, N, K]](
                    a.tt["gpu", AN](), b.tt["gpu", BN](),
                    o.tt["gpu", ON](),
                    grid_dim=((N + T1 - 1) // T1, (M + T1 - 1) // T1, BH),
                    block_dim=(T1, T1),
                )
            else:
                ctx.enqueue_function[_mm_tiled4_kernel[BH, M, N, K]](
                    a.tt["gpu", AN](), b.tt["gpu", BN](),
                    o.tt["gpu", ON](),
                    grid_dim=(
                        (N + T2_N - 1) // T2_N, (M + T2_M - 1) // T2_M, BH
                    ),
                    block_dim=(16, 16),
                )
            ctx.synchronize()
            var dt = Float64(perf_counter_ns() - t0) / 1e6
            if rep >= WARMUP and dt < best:
                best = dt
        o.download(ctx)

        # sampled float64 agreement, in std units of the sampled reference
        var ss = 0.0
        for t in range(N_SAMPLE):
            ss += s_ref[t] * s_ref[t]
        var rms = sqrt(ss / Float64(N_SAMPLE))
        if rms < 1.0e-30:
            rms = 1.0
        var worst = 0.0
        for t in range(N_SAMPLE):
            worst = max(
                worst, abs(Float64(o.data[s_idx[t]]) - s_ref[t])
            )
        var sample_err = worst / rms

        var bits = 0
        if variant == 0:
            best_m0 = best
            for n in range(ON):
                m0_out.append(o.data[n])
        else:
            for n in range(ON):
                if o.data[n] != m0_out[n]:
                    bits += 1
        var flag = String("")
        if sample_err > GATE_STD_UNITS:
            flag = "   ⚠⚠ DISAGREES"
            ok = False
        print(
            "   " + _pad(names[variant], 34) + _pad(_fmt(best, 3), 10)
            + _pad(_fmt(best_m0 / best, 2) + "x", 8)
            + _pad(_fmt(FLOP / (best / 1000.0) / 1.0e9, 1), 10)
            + _pad(String(sample_err), 26)
            + String(bits) + "/" + String(ON) + flag
        )
    return ok


def main() raises:
    comptime assert has_accelerator(), "this benchmark times GPU kernels"
    var ctx = DeviceContext()
    print("=" * 104)
    print("Attention matmuls — " + String(ctx.name()))
    print("=" * 104)
    var ok = True
    # SigLIP, B=1: BH = 12 heads. Scores then context.
    if not run_gemm[12, 1024, 1024, 64](String("SigLIP  Q.Kt"), ctx):
        ok = False
    if not run_gemm[12, 1024, 64, 1024](String("SigLIP  A.V "), ctx):
        ok = False
    # ACT encoder self, training batch 16: BH = 16*8.
    if not run_gemm[128, 162, 162, 32](String("ACT enc Q.Kt"), ctx):
        ok = False
    if not run_gemm[128, 162, 32, 162](String("ACT enc A.V "), ctx):
        ok = False
    # ACT decoder cross, deploy batch 1: BH = 8.
    if not run_gemm[8, 60, 162, 32](String("ACT dec Q.Kt"), ctx):
        ok = False
    if not run_gemm[8, 60, 32, 162](String("ACT dec A.V "), ctx):
        ok = False
    print("")
    if not ok:
        raise Error("a variant disagrees with the float64 reference")
    print("  every variant within 1e-4 std units of float64 on the sample")
