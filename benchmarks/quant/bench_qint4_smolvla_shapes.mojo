# +--------------------------------------------------------------------------+ #
# | MAX's int4 GEMM at SmolVLA's shapes: which compile, which are right
# +--------------------------------------------------------------------------+ #
"""Q0 of the LoRA / int4 prototype: the Q4_0 padding policy for SmolVLA.

`multistage_gemm_q` (MAX's Q4_0 tensor-core GEMM, NVIDIA only) after
`gpu_qint4_repack_Q4_0`, against a host dequantize-and-multiply of the same
bytes, on EVERY output element, for the padding policy SmolVLA's text tower
needs.

## What MAX 26.6 accepts (compile checks, sm_120, run on the Mac)

`gpu_qint4_repack_Q4_0` instantiates only when **N % 128 == 0 and
K % 1024 == 0** (its 128 x 1024 tiles otherwise go down a masked path whose
`store` requires a depth-1 layout, and the repacked layout is nested:
`layout_tensor.mojo:3074`). Kev's shapes (4096, 12288) satisfy both; none of
SmolVLA's text shapes do (N in {960, 320, 2560}, K in {960, 2560}).
`multistage_gemm_q` itself compiles at every SmolVLA shape, N = 720 included,
so a repack of our own would remove the K padding (a Q4 option). The
refused shapes are kept, compile-only, behind `-D UNPADDED=1`, so the check
can be repeated on a newer MAX:

    pixi run -e apple mojo build -I . --target-accelerator sm_120 --emit asm \\
        -D UNPADDED=1 -D REPACK=0 -D ONLY=0 benchmarks/quant/bench_qint4_smolvla_shapes.mojo -o /tmp/q4.s

## What this file runs: the padded policy

N padded to a multiple of 128 and K to a multiple of 1024, with zero groups
in the real `QLinear`. K % 1024 == 0 also makes every warp-K split exact
(partition p starts at tile (K / 32 // P) * p; with K = 960 and P = 4 the
partitions would overlap at tiles 7, 14, 21 and skip tile 29).

    shape     from        rows padded   K padded     extra bytes
    1024x1024 q, o        960 -> 1024   960 -> 1024  +13.8 %
    384x1024  k, v        320 -> 384    960 -> 1024  +28.0 %
    2560x1024 gate, up    -             960 -> 1024  +6.7 %
    1024x3072 down        960 -> 1024   2560 -> 3072 +28.0 %
    per text layer                                   +14.7 %

The prediction for every row is OK. Three configs per shape: noeira's two
(`noeira_max/kev/kernels/q4_0_matmul.mojo`, warp-K 4 below M = 64 and 1
above) and the small-M one with warp-K 2. M in {48, 140, 1120}: small, the
prefill's 2 x 64 + 11 + 1 tokens, and a batch of 8 of them.

    pixi run -e nvidia mojo build -I . benchmarks/quant/bench_qint4_smolvla_shapes.mojo -o build/q4shapes
    for s in 0 1 2 3; do ./build/q4shapes $s; done     # one process per shape
    ./build/q4shapes all

The bytes are random (codes 0..15, scales in [1e-3, 2e-3] rounded to bf16 so
the fp16 scale MAX reads, which it truncates to bf16, is exact). What is
under test is the kernel at these shapes, not a quantizer.
"""

from std.memory import bitcast
from std.random import rand, random_float64, seed
from std.sys import argv, get_defined_int
from std.time import perf_counter_ns
from std.utils.index import Index

from max.gpu.host import DeviceContext, DeviceBuffer
from layout import Coord, Idx, TileTensor, row_major
from linalg.utils_gpu import MatmulConfig

from quantization.qmatmul_gpu import (
    gpu_qint4_repack_Q4_0, multistage_gemm_q, q_smem_usage,
)


comptime GS = 32
comptime GB = 2 + GS // 2   # bytes per Q4_0 group: fp16 scale + 16 bytes of nibbles
comptime ITERS = 20
comptime ONLY = get_defined_int["ONLY", -1]()
"""Compile only this shape index (-1 = all). For the Mac compile check."""
comptime UNPADDED = get_defined_int["UNPADDED", 0]() != 0
"""1 = compile SmolVLA's unpadded shapes instead (compile check only)."""
comptime REPACK = get_defined_int["REPACK", 1]() != 0
"""0 = skip MAX's repack and feed the GEMM the logical bytes. The numbers are
then meaningless; it exists to compile-check `multistage_gemm_q` alone at
shapes MAX's repack refuses (K % 1024 != 0), i.e. to ask whether a repack of
our own would be enough."""
comptime TOL = 2.0e-2
"""max|C - ref| / max|ref|. bf16 output alone is ~4e-3 per element; a lost
K tile or a corrupted scale is an O(1) relative error somewhere."""

comptime BF = DType.bfloat16
comptime Cfg = MatmulConfig[BF, DType.uint8, BF, True]

# noeira_max/kev/kernels/q4_0_matmul.mojo's two configs, plus the small-M one
# with warp-K 2: the fix the source reading suggests for K = 960.
comptime SMALL4 = Cfg(
    block_tile_shape=Index(64, 128, 32), warp_tile_shape=Index(64, 64, 32),
    num_pipeline_stages=4, num_k_partitions=1, num_warp_k_partitions=4,
)
comptime SMALL2 = Cfg(
    block_tile_shape=Index(64, 128, 32), warp_tile_shape=Index(64, 64, 32),
    num_pipeline_stages=4, num_k_partitions=1, num_warp_k_partitions=2,
)
comptime SMALL4_S3 = Cfg(
    block_tile_shape=Index(64, 128, 32), warp_tile_shape=Index(64, 64, 32),
    num_pipeline_stages=3, num_k_partitions=1, num_warp_k_partitions=4,
)
"""SMALL4 with 3 stages: what MAX's own shared-memory fallback would pick if
its budget were the per-block limit. On an RTX 5090 SMALL4 needs 102,400 bytes
of dynamic shared memory; MAX's table gives consumer Blackwell 100 KB "per
multiprocessor", so `multistage_gemm_q` does not fall back, and the driver
refuses the kernel (per-block opt-in max on sm_120 is 101,376 bytes: CUDA
reserves 1 KB per block). The Orin's 163 KB never met it."""
comptime LARGE = Cfg(
    block_tile_shape=Index(128, 128, 32), warp_tile_shape=Index(64, 64, 32),
    num_pipeline_stages=5, num_k_partitions=1, num_warp_k_partitions=1,
)



def q4_0_value(blk: UnsafePointer[UInt8, _], j: Int) -> Float32:
    var u = UInt16(blk[0]) | (UInt16(blk[1]) << 8)
    var d = bitcast[DType.float16, 1](SIMD[DType.uint16, 1](u)).cast[DType.float32]()
    var byte = blk[2 + (j % 16)]
    var nib = (byte & 0xF) if j < 16 else (byte >> 4)
    return (Float32(Int(nib)) - 8.0) * d[0]


struct Verdict(Copyable, Movable):
    var rel: Float64
    var nans: Int
    var us: Float64
    var error: String
    """Non-empty when the kernel did not load or launch."""

    def __init__(out self, rel: Float64, nans: Int, us: Float64):
        self.rel = rel
        self.nans = nans
        self.us = us
        self.error = String("")

    def __init__(out self, *, error: String):
        self.rel = -1.0
        self.nans = 0
        self.us = 0.0
        self.error = error

    def ok(self) -> Bool:
        return self.error == "" and self.nans == 0 and self.rel < TOL


def try_cfg[N: Int, K: Int, CFG: Cfg](
    ctx: DeviceContext, bpk_d: DeviceBuffer[DType.uint8],
    M: Int, ref a_h: List[BFloat16], ref c_ref: List[Float32],
) -> Verdict:
    """`run_cfg`, with a load or launch failure reported instead of fatal."""
    try:
        return run_cfg[N, K, CFG](ctx, bpk_d, M, a_h, c_ref)
    except e:
        return Verdict(error=String(e))


def run_cfg[N: Int, K: Int, CFG: Cfg](
    ctx: DeviceContext, bpk_d: DeviceBuffer[DType.uint8],
    M: Int, ref a_h: List[BFloat16], ref c_ref: List[Float32],
) raises -> Verdict:
    """One config at one M: run, compare every element, then time."""
    comptime KB = (K // GS) * GB
    var bpk_tt = TileTensor(bpk_d, row_major(Coord(Idx[N], Idx[KB])))
    var a_hb = ctx.enqueue_create_host_buffer[BF](M * K)
    var c_hb = ctx.enqueue_create_host_buffer[BF](M * N)
    ctx.synchronize()
    for i in range(M * K):
        a_hb.unsafe_ptr()[i] = a_h[i]
    var a_d = ctx.enqueue_create_buffer[BF](M * K)
    var c_d = ctx.enqueue_create_buffer[BF](M * N)
    ctx.enqueue_copy(a_d, a_hb)
    c_d.enqueue_fill(BFloat16(0))
    var a_tt = TileTensor(a_d, row_major(Coord(M, Idx[K])))
    var c_tt = TileTensor(c_d, row_major(Coord(M, Idx[N])))
    multistage_gemm_q[group_size=GS, pack_factor=8, config=CFG](
        c_tt.to_layout_tensor(), a_tt.as_immut().to_layout_tensor(),
        bpk_tt.as_immut().to_layout_tensor(), CFG, ctx,
    )
    ctx.enqueue_copy(c_hb, c_d)
    ctx.synchronize()
    var err = Float64(0)
    var mag = Float64(0)
    var nans = 0
    for i in range(M * N):
        var got = Float64(c_hb.unsafe_ptr()[i].cast[DType.float32]())
        var want = Float64(c_ref[i])
        if got != got:
            nans += 1
            continue
        err = max(err, abs(got - want))
        mag = max(mag, abs(want))
    for _ in range(3):
        multistage_gemm_q[group_size=GS, pack_factor=8, config=CFG](
            c_tt.to_layout_tensor(), a_tt.as_immut().to_layout_tensor(),
            bpk_tt.as_immut().to_layout_tensor(), CFG, ctx,
        )
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(ITERS):
        multistage_gemm_q[group_size=GS, pack_factor=8, config=CFG](
            c_tt.to_layout_tensor(), a_tt.as_immut().to_layout_tensor(),
            bpk_tt.as_immut().to_layout_tensor(), CFG, ctx,
        )
    ctx.synchronize()
    var us = Float64(perf_counter_ns() - t0) / 1e3 / ITERS
    return Verdict(err / mag if mag > 0.0 else 1.0, nans, us)


def _cell(v: Verdict, expect_ok: Bool, mut refuted: Int) -> String:
    if v.error != "":
        return "LAUNCH FAILED: " + v.error
    var got = "OK" if v.ok() else "WRONG"
    var want = "OK" if expect_ok else "WRONG"
    if v.ok() != expect_ok:
        refuted += 1
    var s = String(got) + " (pred " + want + ") rel " + String(v.rel)
    if v.nans > 0:
        s += " nan " + String(v.nans)
    return s + "  " + String(Int(v.us)) + " us"


def run_shape[N: Int, K: Int](
    ctx: DeviceContext, idx: Int, what: String,
    p4: Bool, p2: Bool, pl: Bool, mut refuted: Int,
) raises:
    comptime KB = (K // GS) * GB
    print("[", idx, "] N=", N, " K=", K, "  ", what, "   N%128=", N % 128,
          " N%64=", N % 64, " K/32=", K // GS, sep="")
    var b_h = ctx.enqueue_create_host_buffer[DType.uint8](N * KB)
    ctx.synchronize()
    var bp = b_h.unsafe_ptr()
    rand[DType.uint8](bp, N * KB, min=0, max=255)
    for blk in range(N * (K // GS)):
        var d32 = Float32(0.001 + 0.001 * random_float64())
        var d_bf = d32.cast[BF]().cast[DType.float32]()
        var u = bitcast[DType.uint16, 1](SIMD[DType.float16, 1](d_bf.cast[DType.float16]()))
        bp[blk * GB] = UInt8(u & 0xFF)
        bp[blk * GB + 1] = UInt8(u >> 8)
    # The dequantized weight [N, K], once, for every reference below.
    var w = List[Float32](capacity=N * K)
    for n in range(N):
        for g in range(K // GS):
            var blk = bp + (n * (K // GS) + g) * GB
            for j in range(GS):
                w.append(q4_0_value(blk, j))

    var b_d = ctx.enqueue_create_buffer[DType.uint8](N * KB)
    var bpk_d = ctx.enqueue_create_buffer[DType.uint8](N * KB)
    ctx.enqueue_copy(b_d, b_h)
    var b_tt = TileTensor(b_d, row_major(Coord(Idx[N], Idx[KB])))
    var bpk_tt = TileTensor(bpk_d, row_major(Coord(Idx[N], Idx[KB])))
    comptime if REPACK:
        gpu_qint4_repack_Q4_0["gpu"](b_tt.as_immut(), bpk_tt, ctx)
    else:
        ctx.enqueue_copy(bpk_d, b_d)
    ctx.synchronize()

    # Small M, the prefill's 2 x 64 + 11 + 1 tokens, and a batch of 8 of them.
    var ms: List[Int] = [48, 140, 1120]
    for mi in range(len(ms)):
        var M = ms[mi]
        var a_h = List[BFloat16](capacity=M * K)
        for _ in range(M * K):
            a_h.append(Float32(random_float64() * 2.0 - 1.0).cast[BF]())
        var c_ref = List[Float32](capacity=M * N)
        var ap = a_h.unsafe_ptr()
        var wp = w.unsafe_ptr()
        for m in range(M):
            for n in range(N):
                var acc = Float64(0)
                for k in range(K):
                    acc += Float64(ap[m * K + k].cast[DType.float32]()) * Float64(wp[n * K + k])
                c_ref.append(Float32(acc))
        # The configs that load first; noeira's warp-K 4 one last.
        var v2 = try_cfg[N, K, SMALL2](ctx, bpk_d, M, a_h, c_ref)
        var vl = try_cfg[N, K, LARGE](ctx, bpk_d, M, a_h, c_ref)
        var v3 = try_cfg[N, K, SMALL4_S3](ctx, bpk_d, M, a_h, c_ref)
        var v4 = try_cfg[N, K, SMALL4](ctx, bpk_d, M, a_h, c_ref)
        print("    M=", M, sep="")
        print("      small-M warp-K 2, 4 stages   ", _cell(v2, p2, refuted))
        print("      large-M warp-K 1 (noeira's)  ", _cell(vl, pl, refuted))
        print("      small-M warp-K 4, 3 stages   ", _cell(v3, p4, refuted))
        print("      small-M warp-K 4 (noeira's)  ", _cell(v4, p4, refuted))
    _ = b_h^  # the reference read through `bp`: keep the host buffer alive


def main() raises:
    seed(0)
    var args = argv()
    var sel = -1
    if len(args) > 1 and String(args[1]) != "all":
        sel = atol(String(args[1]))
    var ctx = DeviceContext()
    print("MAX multistage_gemm_q (Q4_0, g32) at SmolVLA's text shapes, on",
          ctx.name())
    print("dynamic shared memory per block (bytes): small-M warp-K 4",
          q_smem_usage[SMALL4, GS](), "| 3 stages", q_smem_usage[SMALL4_S3, GS](),
          "| warp-K 2", q_smem_usage[SMALL2, GS](), "| large-M",
          q_smem_usage[LARGE, GS](), "| MAX's budget",
          ctx.default_device_info.shared_memory_per_multiprocessor)
    var refuted = 0
    # predictions: (small-M warp-K 4, small-M warp-K 2, large-M) right?
    comptime if not UNPADDED:
        comptime if ONLY < 0 or ONLY == 0:
            if sel < 0 or sel == 0:
                run_shape[1024, 1024](ctx, 0, "q / o, from 960 x 960", True, True, True, refuted)
        comptime if ONLY < 0 or ONLY == 1:
            if sel < 0 or sel == 1:
                run_shape[384, 1024](ctx, 1, "k / v, from 320 x 960", True, True, True, refuted)
        comptime if ONLY < 0 or ONLY == 2:
            if sel < 0 or sel == 2:
                run_shape[2560, 1024](ctx, 2, "gate / up, from 2560 x 960", True, True, True, refuted)
        comptime if ONLY < 0 or ONLY == 3:
            if sel < 0 or sel == 3:
                run_shape[1024, 3072](ctx, 3, "down, from 960 x 2560", True, True, True, refuted)
    else:
        # Compile-only: MAX's repack refuses these (build with -D REPACK=0).
        # `sel` is never -2, so nothing runs; the calls exist to instantiate.
        if sel == -2:
            comptime if ONLY < 0 or ONLY == 0:
                run_shape[960, 960](ctx, 0, "q / o", False, False, False, refuted)
            comptime if ONLY < 0 or ONLY == 1:
                run_shape[320, 960](ctx, 1, "k / v", False, False, False, refuted)
            comptime if ONLY < 0 or ONLY == 2:
                run_shape[2560, 960](ctx, 2, "gate / up", False, False, False, refuted)
            comptime if ONLY < 0 or ONLY == 3:
                run_shape[960, 2560](ctx, 3, "down", False, False, False, refuted)
            comptime if ONLY < 0 or ONLY == 4:
                run_shape[720, 960](ctx, 4, "expert o_proj", False, False, False, refuted)
        print("compile-check build (UNPADDED=1): nothing to run")
        return
    print("predictions refuted:", refuted)
