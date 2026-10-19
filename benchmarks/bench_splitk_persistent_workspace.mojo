"""Split-K GEMM with a CALLER-OWNED workspace — the alloc-free, capturable variant.

Context: `docs/MODULAR_MATMUL_ALLOC_REPORT.md` Measurements 4 and 5.
`linalg.matmul.gpu.multistage_gemm` allocates its split-K reduction
workspace on every call (matmul/gpu/__init__.mojo:1845) and frees it at
the end of the call. That costs ~250 us of cuMemAlloc/cuMemFree per GEMM
and makes the GEMM impossible to put inside a CUDA-graph capture.

The report asked Modular to reuse that workspace. This file shows we do
not have to wait for them: every piece `multistage_gemm` uses on that
branch is exported from the SHIPPED `linalg` package, so we can inline
the same ~40 lines against a buffer we own.

  from linalg.matmul.gpu import multistage_gemm_split_k_kernel, split_k_reduce
  from linalg.utils_gpu   import MatmulConfig, MatmulKernels, select_config

Verified importable against MAX 26.5.0 / Mojo 1.0.0 (ed45d567).

  arm A   linalg.matmul                      allocates (baseline)
  arm B   splitk_gemm + SplitKWorkspace      no allocation
  arm C   arm B inside a CUDA-graph capture  the thing Measurement 5 blocked

Run (NVIDIA only — the multistage kernel does not build for Metal):

    pixi run -e nvidia mojo run -I . benchmarks/bench_splitk_persistent_workspace.mojo
    pixi run -e nvidia mojo run -D LOGGING_LEVEL=INFO -I . \
        benchmarks/bench_splitk_persistent_workspace.mojo

`-D LOGGING_LEVEL=INFO` makes MAX name the path it took and print
"K partitions: N" for each call. It is a COMPILE-TIME define; the
same-named environment variable does nothing (verified).
"""

from std.math import ceildiv, align_up
from std.sys import default_accelerator
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext, DeviceBuffer, FuncAttribute
from layout import Coord, Idx, TileTensor, row_major
from std.utils.index import Index

from linalg.matmul import matmul as max_matmul
from linalg.utils_gpu import MatmulConfig, MatmulKernels, select_config
from linalg.matmul.gpu import multistage_gemm_split_k_kernel, split_k_reduce

from noeira.cuda import CUDAGraph


struct SplitKWorkspace[dtype: DType](Movable):
    """A device buffer reused across split-K GEMMs.

    Sized once for the largest `num_k_partitions * M * N` any call will
    ask for, then handed to every launch. Because it is an ordinary
    long-lived buffer, a capture region containing the GEMM never sees an
    allocation — which is the whole point.
    """

    var buf: DeviceBuffer[Self.dtype]
    var capacity: Int

    def __init__(out self, ctx: DeviceContext, capacity: Int) raises:
        self.buf = ctx.enqueue_create_buffer[Self.dtype](capacity)
        self.capacity = capacity
        ctx.synchronize()


def splitk_gemm[
    c_type: DType,
    a_type: DType,
    b_type: DType, //,
    *,
    transpose_b: Bool,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
](
    c: TileTensor[mut=True, c_type, ...],
    a: TileTensor[mut=False, a_type, ...],
    b: TileTensor[mut=False, b_type, ...],
    num_partitions: Int,
    mut ws: SplitKWorkspace[config.split_k_reduction_type],
    ctx: DeviceContext,
) raises:
    """`multistage_gemm`'s split-K branch, verbatim, but on `ws.buf`.

    Mirrors matmul/gpu/__init__.mojo:1840-1915 with the
    `ctx.enqueue_create_buffer` / `_ = work_space_data^` pair removed.

    On NVIDIA the comptime `config` supplies the TILE SHAPE only: the
    partition count is a runtime kernel argument (the kernel slices the
    workspace by `block_idx.z * M * N`), which is why MAX itself passes
    the static `kernels.ampere_*` config alongside a runtime config whose
    `num_k_partitions` differs. `MatmulConfig.__eq__` compares only
    `block_tile_shape` and `num_pipeline_stages`, confirming that split.
    """
    var M = Int(c.dim[0]())
    var N = Int(c.dim[1]())

    if num_partitions * M * N > ws.capacity:
        raise Error("SplitKWorkspace too small for this GEMM")

    # ⚠ PARTITION COUNT SAFETY. `multistage_gemm_split_k_kernel`'s NVIDIA path
    # splits K with `LayoutTensor.split[axis, split_alignment=BK]`, which is
    # (layout_tensor.mojo:3870):
    #
    #     part = align_up(K // P, BK)                     <- FLOOR div, then align
    #     size_of_partition_i = min(part, K - i * part)
    #     ptr_i               = base + i * part * stride
    #
    # so when `(P-1) * part >= K` the last block's pointer starts PAST the end
    # of A and B and its size goes NEGATIVE. Measured on the 5090 at
    # K=2592, BK=16: P<=12 is fine (11*224 = 2464 < 2592) and P=16 faults
    # (15*176 = 2640 > 2592, last size -48) with CUDA_ERROR_ILLEGAL_ADDRESS.
    #
    # There are TWO ways to get this wrong, and only one of them is loud:
    #
    #   OVERRUN       (P-1) * part >= K   the last block starts past the end of
    #                                     A and B with a NEGATIVE size
    #                                     -> CUDA_ERROR_ILLEGAL_ADDRESS
    #   UNDERCOVERAGE  P * part < K       `part` is floor-then-align, so the P
    #                                     partitions can fail to span K. The
    #                                     tail is never accumulated and the
    #                                     GEMM SILENTLY RETURNS A WRONG ANSWER
    #
    # Both measured on the 5090 at K=2592, BK=16. P=16 overruns
    # (15*176 = 2640 > 2592). P=40 undercovers: part = align_up(64,16) = 64 and
    # 40*64 = 2560, so 32 of 2592 contraction elements are dropped -- and the
    # observed relative error was 0.012344, against 32/2592 = 0.012346. The
    # error IS the fraction of K dropped, to five digits.
    #
    # ⚠ MAX'S OWN GUARD COVERS NEITHER. `select_config` breaks on `K < P * bk`
    # (2592 < 256 is false at P=16, so it passes) and is saved only by the
    # SEPARATE `min_k_partition = 1024` test capping P at 2 here for unrelated
    # reasons. Any caller that chooses P itself -- us, or MAX's own
    # `TUNE_NUM_K_PARTITIONS` autotune define -- can reach both.
    var K_dim = Int(a.dim[1]())
    comptime BK = config.block_tile_shape[2]
    var part = align_up(K_dim // num_partitions, BK) if num_partitions > 0 else 0
    if num_partitions < 1:
        raise Error("num_k_partitions must be >= 1")
    if (num_partitions - 1) * part >= K_dim:
        raise Error(
            "num_k_partitions OVERRUNS K: the first P-1 partitions of"
            " align_up(K//P, BK) already cover K, so the last block would read"
            " past the operands (this one crashes loudly)"
        )
    if num_partitions * part < K_dim:
        raise Error(
            "num_k_partitions UNDERCOVERS K: P*align_up(K//P, BK) < K, so the"
            " tail of the contraction is never accumulated and the GEMM"
            " silently returns a wrong answer"
        )


    comptime ws_type = config.split_k_reduction_type
    # MAX 26.7: the kernel takes TileTensors, with a LayoutType and a
    # linear_idx_type parameter per operand (same as nn/core/splitk_gemm.mojo).
    comptime static_N = c.static_shape[1]
    var ws_tt = TileTensor(ws.buf, row_major(Coord(num_partitions, M, Idx[static_N])))

    comptime kern = multistage_gemm_split_k_kernel[
        c_type, type_of(c).LayoutType,
        a_type, type_of(a).LayoutType,
        b_type, type_of(b).LayoutType,
        ws_type, type_of(ws_tt).LayoutType,
        transpose_b,
        type_of(c).linear_idx_type,
        type_of(a).linear_idx_type,
        type_of(b).linear_idx_type,
        type_of(ws_tt).linear_idx_type,
        config,
    ]

    ctx.enqueue_function[kern](
        c.as_unsafe_any_origin(),
        a.as_unsafe_any_origin(),
        b.as_unsafe_any_origin(),
        ws_tt.as_unsafe_any_origin(),
        Int32(num_partitions),
        grid_dim=(
            ceildiv(N, config.block_tile_shape[1]),
            ceildiv(M, config.block_tile_shape[0]),
            num_partitions,
        ),
        block_dim=config.block_dim(),
        shared_mem_bytes=config.shared_mem_usage(),
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(config.shared_mem_usage())
        ),
    )

    split_k_reduce(c, ws_tt, ctx)


def main() raises:
    comptime if not default_accelerator().is_nvidia_gpu():
        print(
            "NVIDIA only — the multistage kernel has no Metal codegen."
            " Build with `pixi run -e nvidia`."
        )
    else:
        # An ACT dW shape: grad_w = x^T @ go, K = batch x tokens.
        # `[256 x 2592] @ [2592 x 256]` measured 90.98 us + an alloc/free
        # pair in MODULAR_MATMUL_ALLOC_REPORT.md Measurement 4.
        comptime M = 256
        comptime K = 2592
        comptime N = 256
        comptime DT = DType.float32
        comptime REPS = 101

        with DeviceContext() as ctx:
            var ab = ctx.enqueue_create_buffer[DT](M * K)
            var bb = ctx.enqueue_create_buffer[DT](K * N)
            var c_ref = ctx.enqueue_create_buffer[DT](M * N)
            var c_ours = ctx.enqueue_create_buffer[DT](M * N)
            ab.enqueue_fill(Float32(0.01))
            bb.enqueue_fill(Float32(0.02))
            c_ref.enqueue_fill(Float32(0.0))
            c_ours.enqueue_fill(Float32(0.0))
            ctx.synchronize()

            var av = TileTensor(ab, row_major[M, K]())
            var bv = TileTensor(bb, row_major[K, N]())
            var cref = TileTensor(c_ref, row_major[M, N]())
            var cours = TileTensor(c_ours, row_major[M, N]())

            var picked = select_config[DT, DT, DT, False](M, N, K, ctx)
            print(
                "select_config: block",
                picked.block_tile_shape,
                " stages",
                picked.num_pipeline_stages,
                " k_partitions",
                picked.num_k_partitions,
            )
            if picked.num_k_partitions <= 1:
                print(
                    "!! this shape does NOT split K here — pick another shape,"
                    " the A/B below is vacuous."
                )

            comptime kernels = MatmulKernels[DT, DT, DT, False]()
            comptime cfg = kernels.ampere_128x128_4
            # `cfg` supplies the grid and the shared-memory request, so it has
            # to be the tile `select_config` actually chose. `__eq__` compares
            # block_tile_shape + num_pipeline_stages only, which is exactly the
            # part that must agree (the partition count is a runtime arg).
            if not (picked == cfg):
                raise Error(
                    "select_config picked a different tile than `cfg`; launch"
                    " geometry would not match the kernel"
                )
            var ws = SplitKWorkspace[cfg.split_k_reduction_type](
                ctx, 8 * M * N
            )

            # ---- arm A: MAX's matmul (allocates per call) ----------------
            for _ in range(5):
                max_matmul[target="gpu"](cref, av, bv, ctx)
            ctx.synchronize()
            var t0 = perf_counter_ns()
            for _ in range(REPS):
                max_matmul[target="gpu"](cref, av, bv, ctx)
            ctx.synchronize()
            var t1 = perf_counter_ns()

            # ---- arm B: ours (no allocation) ------------------------------
            for _ in range(5):
                splitk_gemm[transpose_b=False, config=cfg](
                    cours, av, bv, picked.num_k_partitions, ws, ctx
                )
            ctx.synchronize()
            var t2 = perf_counter_ns()
            for _ in range(REPS):
                splitk_gemm[transpose_b=False, config=cfg](
                    cours, av, bv, picked.num_k_partitions, ws, ctx
                )
            ctx.synchronize()
            var t3 = perf_counter_ns()

            print(
                "A linalg.matmul     ",
                Float64(t1 - t0) / 1000.0 / Float64(REPS),
                "us/call",
            )
            print(
                "B persistent split-K",
                Float64(t3 - t2) / 1000.0 / Float64(REPS),
                "us/call",
            )

            # ---- correctness: B must equal A ------------------------------
            # Keep A's result on the host: arm C overwrites c_ours, and we
            # want to check the REPLAYED value against the same reference.
            var ref_host = List[Float64](capacity=M * N)
            var worst = Float64(0.0)
            with c_ref.map_to_host() as hr:
                with c_ours.map_to_host() as ho:
                    for i in range(M * N):
                        ref_host.append(Float64(hr[i]))
                        var d = abs(Float64(hr[i]) - Float64(ho[i]))
                        if d > worst:
                            worst = d
            print("max |A - B| =", worst)

            # ---- arm C: the same GEMM inside a capture --------------------
            # Measurement 5 aborted the PROCESS here, because MAX's split-K
            # allocated its workspace inside the capture region. With the
            # workspace owned by the caller there is nothing left to allocate.
            print("--- capture attempt (arm C) ---")
            c_ours.enqueue_fill(Float32(0.0))
            ctx.synchronize()

            var g = CUDAGraph(ctx)
            if g.is_disabled():
                print("CUDAGraph disabled — no usable stream; skipping arm C.")
            else:
                g.begin_capture()
                splitk_gemm[transpose_b=False, config=cfg](
                    cours, av, bv, picked.num_k_partitions, ws, ctx
                )
                g.end_capture()
                print("captured", g.num_nodes(), "nodes")

                g.replay_on_mojo_stream()
                ctx.synchronize()

                var worst_replay = Float64(0.0)
                with c_ours.map_to_host() as ho:
                    for i in range(M * N):
                        var d = abs(ref_host[i] - Float64(ho[i]))
                        if d > worst_replay:
                            worst_replay = d
                print("replay ok — max |A - replayed| =", worst_replay)
                _ = g^

            # ⚠⚠ KEEP-ALIVE. LOAD-BEARING, NOT TIDINESS.
            #
            # Mojo destroys a value at its LAST USE, not at end of scope. The
            # first version of this file ended at the `splitk_gemm` call inside
            # the capture block, which was the last textual mention of `ws`,
            # `ab`, `bb` and `c_ours`. All four DeviceBuffers were therefore
            # freed BETWEEN `end_capture()` and `replay_on_mojo_stream()`, and
            # the replay wrote into freed device memory:
            #
            #     captured 2 nodes
            #     CUDA call failed: CUDA_ERROR_ILLEGAL_ADDRESS
            #
            # Capture had already succeeded; only the replay faulted, which is
            # exactly the signature of operands dying after capture. This is
            # the same rule that forces `noeira/cuda/graph.mojo` to store its
            # own `DeviceContext` in a field.
            #
            # A CUDA graph holds RAW POINTERS to every operand it was captured
            # with. Anything a captured region touches must outlive the LAST
            # REPLAY, so a real caller should own the workspace and the operand
            # buffers in the same struct as the graph, not in a scope that ends
            # earlier.
            _ = ab^
            _ = bb^
            _ = c_ref^
            _ = c_ours^
            _ = ws^
