"""CUDA Graph Capture Diagnostic.

Tests which stream Mojo's ctx.enqueue_function actually dispatches on,
and whether CUDA graph capture can intercept those launches.

Tries 3 capture strategies:
  1. CU_STREAM_PER_THREAD (handle=2)
  2. NULL stream (handle=0) — the legacy default stream
  3. A freshly created stream

Run with:
    pixi run -e nvidia mojo run -I . benchmarks/_test_cuda_ffi.mojo
"""

from max.gpu.host import DeviceContext, DeviceBuffer
from max.gpu import thread_idx, block_idx, block_dim
from std.ffi import OwnedDLHandle, c_int
from std.memory import alloc
from std.time import perf_counter_ns
from noeira.core.tiles import Tile, ImmTile, tile
from noeira.nn.constants import DT as dtype


comptime CUptr = Pointer[NoneType, MutAnyOrigin]


def add_kernel[N: Int](
    output: Tile[dtype, N],
    a: ImmTile[dtype, N],
    b: ImmTile[dtype, N],
):
    var tid = Int(thread_idx.x) + Int(block_idx.x) * Int(block_dim.x)
    if tid < N:
        output[tid] = a[tid] + b[tid]


def scale_kernel[N: Int](
    output: Tile[dtype, N],
    input: ImmTile[dtype, N],
):
    var tid = Int(thread_idx.x) + Int(block_idx.x) * Int(block_dim.x)
    if tid < N:
        output[tid] = input[tid] * Scalar[dtype](2.0)


def relu_kernel[N: Int](
    data: Tile[dtype, N],
):
    var tid = Int(thread_idx.x) + Int(block_idx.x) * Int(block_dim.x)
    if tid < N:
        var v = data.ptr[tid]
        data.ptr[tid] = v if v > Scalar[dtype](0.0) else Scalar[dtype](0.0)


def main() raises:
    print("=== CUDA Graph Capture — Diagnostic ===\n")

    var ctx = DeviceContext()
    ctx.synchronize()

    var cuda = OwnedDLHandle("libcuda.so")

    # --- Load all CUDA driver functions we need ---
    var cuStreamCreate = cuda.get_function[c_int]("cuStreamCreate")
    var cuStreamBeginCapture = cuda.get_function[c_int]("cuStreamBeginCapture")
    var cuStreamEndCapture = cuda.get_function[c_int]("cuStreamEndCapture")
    var cuGraphInstantiate = cuda.get_function[c_int]("cuGraphInstantiate")
    var cuGraphLaunch = cuda.get_function[c_int]("cuGraphLaunch")
    var cuStreamSynchronize = cuda.get_function[c_int]("cuStreamSynchronize")
    var cuGraphDestroy = cuda.get_function[c_int]("cuGraphDestroy")
    var cuGraphExecDestroy = cuda.get_function[c_int]("cuGraphExecDestroy")

    # Diagnostic: count nodes in captured graph
    # cuGraphGetNodes(graph, nodes_ptr, numNodes_ptr) -> CUresult
    # When nodes_ptr is null, just fills numNodes with the count
    var cuGraphGetNodes = cuda.get_function[c_int]("cuGraphGetNodes")

    # Diagnostic: query capture status of a stream
    # cuStreamIsCapturing(stream, captureStatus_ptr) -> CUresult
    var cuStreamIsCapturing = cuda.get_function[c_int]("cuStreamIsCapturing")

    # Diagnostic: get capture info (CUDA 10.1+)
    # cuStreamGetCaptureInfo(stream, captureStatus_ptr, id_ptr) -> CUresult
    var cuStreamGetCaptureInfo = cuda.get_function[c_int]("cuStreamGetCaptureInfo")

    # --- Buffers ---
    comptime N = 8192
    comptime TPB = 256
    comptime grid = ((N + TPB - 1) // TPB,)
    comptime block = (TPB,)

    var a_buf = ctx.enqueue_create_buffer[dtype](N)
    var b_buf = ctx.enqueue_create_buffer[dtype](N)
    var c_buf = ctx.enqueue_create_buffer[dtype](N)
    var d_buf = ctx.enqueue_create_buffer[dtype](N)
    a_buf.enqueue_fill(Scalar[dtype](1.0))
    b_buf.enqueue_fill(Scalar[dtype](2.0))
    c_buf.enqueue_fill(Scalar[dtype](0.0))
    d_buf.enqueue_fill(Scalar[dtype](0.0))
    ctx.synchronize()

    var a_t = tile[N](a_buf)
    var b_t = tile[N](b_buf)
    var c_t = tile[N](c_buf)
    var c_i = tile[N](c_buf)
    var d_t = tile[N](d_buf)
    var d_i = tile[N](d_buf)

    comptime k_add = add_kernel[N]
    comptime k_scale = scale_kernel[N]
    comptime k_relu = relu_kernel[N]

    def run_chain() raises:
        ctx.enqueue_function[k_add, k_add](c_t, a_t, b_t, grid_dim=grid, block_dim=block)
        ctx.enqueue_function[k_scale, k_scale](d_t, c_i, grid_dim=grid, block_dim=block)
        ctx.enqueue_function[k_relu, k_relu](d_t, grid_dim=grid, block_dim=block)
        ctx.enqueue_function[k_scale, k_scale](c_t, d_i, grid_dim=grid, block_dim=block)
        ctx.enqueue_function[k_relu, k_relu](c_t, grid_dim=grid, block_dim=block)

    # Warmup + verify direct dispatch works
    run_chain()
    ctx.synchronize()
    with c_buf.map_to_host() as h:
        print("Direct dispatch result: c[0] =", h[0], "(expected 12.0)")

    # --- Helper: try capture on a given stream ---
    var replay_buf = alloc[CUptr](1)
    replay_buf[] = CUptr()
    _ = cuStreamCreate(replay_buf, UInt32(0))
    var replay_stream = replay_buf[]

    var graph_buf = alloc[CUptr](1)
    var exec_buf = alloc[CUptr](1)
    var num_nodes_buf = alloc[UInt64](1)
    var cap_status_buf = alloc[c_int](1)
    var cap_id_buf = alloc[UInt64](1)

    # ============================================================
    # Strategy 1: CU_STREAM_PER_THREAD (handle = 2)
    # ============================================================
    print("\n" + "=" * 60)
    print("Strategy 1: CU_STREAM_PER_THREAD (handle=2)")
    print("=" * 60)

    var per_thread = CUptr(unsafe_from_address=2)

    c_buf.enqueue_fill(Scalar[dtype](0.0))
    d_buf.enqueue_fill(Scalar[dtype](0.0))
    ctx.synchronize()

    var r1 = cuStreamBeginCapture(per_thread, c_int(0))
    print("  BeginCapture:", r1)

    # Check capture status mid-capture
    cap_status_buf[] = c_int(0)
    var r1_status = cuStreamIsCapturing(per_thread, cap_status_buf)
    print("  IsCapturing (PER_THREAD):", r1_status, "status:", cap_status_buf[],
          "(1=capturing, 0=not)")

    run_chain()

    # Check capture status after kernels
    cap_status_buf[] = c_int(0)
    r1_status = cuStreamIsCapturing(per_thread, cap_status_buf)
    print("  IsCapturing after kernels:", r1_status, "status:", cap_status_buf[])

    graph_buf[] = CUptr()
    var r1_end = cuStreamEndCapture(per_thread, graph_buf)
    var graph1 = graph_buf[]
    print("  EndCapture:", r1_end, "Graph:", Int(graph1))

    if Int(graph1) != 0:
        num_nodes_buf[] = UInt64(0)
        var r1_nodes = cuGraphGetNodes(graph1, CUptr(), num_nodes_buf)
        print("  >>> Graph has", num_nodes_buf[], "nodes (expected 5) <<<")

        exec_buf[] = CUptr()
        var r1_inst = cuGraphInstantiate(exec_buf, graph1, UInt64(0))
        print("  Instantiate:", r1_inst)

        if r1_inst == 0:
            c_buf.enqueue_fill(Scalar[dtype](0.0))
            d_buf.enqueue_fill(Scalar[dtype](0.0))
            ctx.synchronize()
            _ = cuGraphLaunch(exec_buf[], replay_stream)
            _ = cuStreamSynchronize(replay_stream)
            with c_buf.map_to_host() as h:
                print("  Replay result: c[0] =", h[0], "(expected 12.0)")
            _ = cuGraphExecDestroy(exec_buf[])

        _ = cuGraphDestroy(graph1)
    else:
        print("  >>> No graph returned <<<")

    # ============================================================
    # Strategy 2: NULL stream (handle = 0) — legacy default
    # ============================================================
    print("\n" + "=" * 60)
    print("Strategy 2: NULL stream (handle=0)")
    print("=" * 60)

    var null_stream = CUptr()

    c_buf.enqueue_fill(Scalar[dtype](0.0))
    d_buf.enqueue_fill(Scalar[dtype](0.0))
    ctx.synchronize()

    var r2 = cuStreamBeginCapture(null_stream, c_int(0))
    print("  BeginCapture:", r2)

    if r2 == 0:
        cap_status_buf[] = c_int(0)
        _ = cuStreamIsCapturing(null_stream, cap_status_buf)
        print("  IsCapturing (NULL):", cap_status_buf[])

        run_chain()

        graph_buf[] = CUptr()
        var r2_end = cuStreamEndCapture(null_stream, graph_buf)
        var graph2 = graph_buf[]
        print("  EndCapture:", r2_end, "Graph:", Int(graph2))

        if Int(graph2) != 0:
            num_nodes_buf[] = UInt64(0)
            _ = cuGraphGetNodes(graph2, CUptr(), num_nodes_buf)
            print("  >>> Graph has", num_nodes_buf[], "nodes <<<")

            exec_buf[] = CUptr()
            var r2_inst = cuGraphInstantiate(exec_buf, graph2, UInt64(0))
            print("  Instantiate:", r2_inst)

            if r2_inst == 0:
                c_buf.enqueue_fill(Scalar[dtype](0.0))
                d_buf.enqueue_fill(Scalar[dtype](0.0))
                ctx.synchronize()
                _ = cuGraphLaunch(exec_buf[], replay_stream)
                _ = cuStreamSynchronize(replay_stream)
                with c_buf.map_to_host() as h:
                    print("  Replay result: c[0] =", h[0], "(expected 12.0)")
                _ = cuGraphExecDestroy(exec_buf[])

            _ = cuGraphDestroy(graph2)
        else:
            print("  >>> No graph returned <<<")
    else:
        print("  BeginCapture FAILED (error", r2, ")")
        print("  (NULL stream may not support capture without CUDA_API_PER_THREAD_DEFAULT_STREAM)")

    # ============================================================
    # Strategy 3: Fresh created stream
    # ============================================================
    print("\n" + "=" * 60)
    print("Strategy 3: Fresh created stream")
    print("=" * 60)

    var fresh_buf = alloc[CUptr](1)
    fresh_buf[] = CUptr()
    _ = cuStreamCreate(fresh_buf, UInt32(0))
    var fresh_stream = fresh_buf[]
    print("  Created stream:", Int(fresh_stream))

    c_buf.enqueue_fill(Scalar[dtype](0.0))
    d_buf.enqueue_fill(Scalar[dtype](0.0))
    ctx.synchronize()

    var r3 = cuStreamBeginCapture(fresh_stream, c_int(0))
    print("  BeginCapture:", r3)

    if r3 == 0:
        cap_status_buf[] = c_int(0)
        _ = cuStreamIsCapturing(fresh_stream, cap_status_buf)
        print("  IsCapturing (fresh):", cap_status_buf[])

        run_chain()

        graph_buf[] = CUptr()
        var r3_end = cuStreamEndCapture(fresh_stream, graph_buf)
        var graph3 = graph_buf[]
        print("  EndCapture:", r3_end, "Graph:", Int(graph3))

        if Int(graph3) != 0:
            num_nodes_buf[] = UInt64(0)
            _ = cuGraphGetNodes(graph3, CUptr(), num_nodes_buf)
            print("  >>> Graph has", num_nodes_buf[], "nodes <<<")

            exec_buf[] = CUptr()
            var r3_inst = cuGraphInstantiate(exec_buf, graph3, UInt64(0))
            print("  Instantiate:", r3_inst)

            if r3_inst == 0:
                c_buf.enqueue_fill(Scalar[dtype](0.0))
                d_buf.enqueue_fill(Scalar[dtype](0.0))
                ctx.synchronize()
                _ = cuGraphLaunch(exec_buf[], replay_stream)
                _ = cuStreamSynchronize(replay_stream)
                with c_buf.map_to_host() as h:
                    print("  Replay result: c[0] =", h[0], "(expected 12.0)")
                _ = cuGraphExecDestroy(exec_buf[])

            _ = cuGraphDestroy(graph3)
        else:
            print("  >>> No graph returned <<<")
    else:
        print("  BeginCapture FAILED (error", r3, ")")

    fresh_buf.free()

    # ============================================================
    # Strategy 4: Capture on Mojo's ACTUAL internal stream
    #   Requires LD_PRELOAD with libcuda_intercept.so
    # ============================================================
    print("\n" + "=" * 60)
    print("Strategy 4: Mojo's actual AsyncRT stream (via interceptor)")
    print("=" * 60)

    # Load interceptor — use the same dlopen pattern as libcuda.so
    var intercept = OwnedDLHandle("./benchmarks/libcuda_intercept.so")

    var get_mojo_stream = intercept.get_function[CUptr]("intercept_get_mojo_stream")
    var set_logging = intercept.get_function[NoneType]("intercept_set_logging")

    var mojo_stream = get_mojo_stream()
    print("  Mojo's internal stream:", Int(mojo_stream))

    if Int(mojo_stream) == 0:
        print("  No stream captured yet — need at least one warmup kernel")
    else:
        # Disable interceptor logging for cleaner output
        set_logging(c_int(0))

        c_buf.enqueue_fill(Scalar[dtype](0.0))
        d_buf.enqueue_fill(Scalar[dtype](0.0))
        ctx.synchronize()

        var r4 = cuStreamBeginCapture(mojo_stream, c_int(0))
        print("  BeginCapture on Mojo stream:", r4)

        if r4 == 0:
            var cap_buf2 = alloc[c_int](1)
            cap_buf2[] = c_int(0)
            _ = cuStreamIsCapturing(mojo_stream, cap_buf2)
            print("  IsCapturing:", cap_buf2[])
            cap_buf2.free()

            run_chain()

            var gbuf2 = alloc[CUptr](1)
            gbuf2[] = CUptr()
            var r4_end = cuStreamEndCapture(mojo_stream, gbuf2)
            var graph4 = gbuf2[]
            print("  EndCapture:", r4_end, "Graph:", Int(graph4))

            if Int(graph4) != 0:
                var nnbuf = alloc[UInt64](1)
                nnbuf[] = UInt64(0)
                _ = cuGraphGetNodes(graph4, CUptr(), nnbuf)
                print("  >>> Graph has", nnbuf[], "nodes (expected 5) <<<")

                var ebuf = alloc[CUptr](1)
                ebuf[] = CUptr()
                var r4_inst = cuGraphInstantiate(ebuf, graph4, UInt64(0))
                print("  Instantiate:", r4_inst)

                if r4_inst == 0:
                    c_buf.enqueue_fill(Scalar[dtype](0.0))
                    d_buf.enqueue_fill(Scalar[dtype](0.0))
                    ctx.synchronize()
                    _ = cuGraphLaunch(ebuf[], replay_stream)
                    _ = cuStreamSynchronize(replay_stream)
                    with c_buf.map_to_host() as h:
                        print("  Replay result: c[0] =", h[0], "(expected 12.0)")
                        if h[0] == Scalar[dtype](12.0):
                            print("\n  *** SUCCESS — CUDA Graph captured Mojo kernels! ***")

                            # Quick benchmark
                            print("\n  --- Quick benchmark: 1000 iterations ---")
                            var warmup_n = 100
                            var iters_n = 1000

                            for _ in range(warmup_n):
                                run_chain()
                                ctx.synchronize()

                            var total_direct: UInt = 0
                            for _ in range(iters_n):
                                var start = perf_counter_ns()
                                run_chain()
                                ctx.synchronize()
                                total_direct += perf_counter_ns() - start

                            for _ in range(warmup_n):
                                _ = cuGraphLaunch(ebuf[], replay_stream)
                                _ = cuStreamSynchronize(replay_stream)

                            var total_graph: UInt = 0
                            for _ in range(iters_n):
                                var start = perf_counter_ns()
                                _ = cuGraphLaunch(ebuf[], replay_stream)
                                _ = cuStreamSynchronize(replay_stream)
                                total_graph += perf_counter_ns() - start

                            var avg_direct = Float64(total_direct // UInt(iters_n)) / 1000.0
                            var avg_graph = Float64(total_graph // UInt(iters_n)) / 1000.0
                            print("  Direct dispatch: ", avg_direct, " us")
                            print("  Graph replay:    ", avg_graph, " us")
                            if avg_graph > 0.0:
                                print("  Speedup:         ", avg_direct / avg_graph, "x")

                    _ = cuGraphExecDestroy(ebuf[])
                _ = cuGraphDestroy(graph4)
                nnbuf.free()
                ebuf.free()
            else:
                print("  >>> No graph returned <<<")

            gbuf2.free()
        else:
            print("  BeginCapture FAILED (error", r4, ")")

        # Re-enable logging
        set_logging(c_int(1))

    replay_buf.free()

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
