"""Launch-overhead micro-benchmarks for `MetalRT` vs MAX's `enqueue_function`.

    pixi run -e apple mojo run -I . noeira/experimental/metal_rt/bench_launch.mojo

1. objc_msgSend through `get_function` (dlsym per call) vs a cached pointer.
2. Host-side encode cost per launch (one huge command buffer, no waiting).
3. K dependent 1-thread launches, N per command buffer (the probe's table).
4. Live buffers: 0 / 256 / 1024 extra live buffers, our three residency modes
   vs MAX. Times are us/launch, best of 3 repeats after a warm-up.
"""

from std.time import perf_counter_ns
from max.gpu import global_idx
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.experimental.metal_rt.objc import ObjC
from noeira.experimental.metal_rt.metal4 import Metal4RT
from noeira.experimental.metal_rt.runtime import (
    MetalRT,
    Buffer,
    COMMIT_EACH,
    COMMIT_EVERY_N,
    COMMIT_ADAPTIVE,
    RESIDENCY_SET,
    RESIDENCY_PER_CB,
    RESIDENCY_PER_LAUNCH,
    ORDER_FENCE,
    ORDER_EVENT,
    ORDER_NONE,
    ORDER_TRACKED,
    DONE_STATUS,
    DONE_EVENT,
)

comptime K = 2000
comptime P = Pointer[Float32, MutAnyOrigin]


def bump(x: P):
    if global_idx.x == 0:
        x[unsafe_offset=0] = x[unsafe_offset=0] + 1.0


def stamp(x: P, v: Float32):
    if global_idx.x == 0:
        x[unsafe_offset=0] = v


def roundtrip_ours(mut rt: MetalRT, c: Buffer, spin_on_value: Bool) raises -> Float64:
    """Median us for ONE 1-thread launch + wait, 400 times. With
    `spin_on_value`, the host spins on the value the kernel writes (shared
    memory) instead of waiting for command-buffer completion."""
    var ts = List[Int]()
    var hp = c.host_ptr[DType.float32]()
    for i in range(400):
        var v = Float32(i + 1)
        var t0 = Int(perf_counter_ns())
        rt.launch[stamp](c.dev[DType.float32](), v, grid=1, block=1)
        if spin_on_value:
            rt.flush()
            while hp.unsafe_load[volatile=True]() != v:
                pass
        else:
            rt.sync()
        ts.append(Int(perf_counter_ns()) - t0)
        if hp[unsafe_offset=0] != v:
            raise Error("round trip read a stale value")
    rt.sync()
    sort(ts)
    return Float64(ts[len(ts) // 2]) / 1e3


def m4_dependent(mut rt: Metal4RT, c: Buffer) raises -> Float64:
    var best = Float64(1e30)
    for r in range(4):
        c.host_ptr[DType.float32]()[unsafe_offset=0] = 0.0
        var t0 = perf_counter_ns()
        for _ in range(K):
            rt.launch[bump](c.dev[DType.float32](), grid=1, block=1)
        rt.sync()
        var t = us(t0, perf_counter_ns(), K)
        if r > 0 and t < best:
            best = t
    var got = c.host_ptr[DType.float32]()[unsafe_offset=0]
    if got != Float32(K):
        print("  !! METAL4 LOST UPDATES: counter", got, "of", K, "(batch", rt.batch_n, "barriers", rt.barriers, ")")
        return -1.0
    return best


def m4_roundtrip(mut rt: Metal4RT, c: Buffer) raises -> Float64:
    var ts = List[Int]()
    var hp = c.host_ptr[DType.float32]()
    for i in range(400):
        var v = Float32(i + 1)
        var t0 = Int(perf_counter_ns())
        rt.launch[stamp](c.dev[DType.float32](), v, grid=1, block=1)
        rt.sync()
        ts.append(Int(perf_counter_ns()) - t0)
        if hp[unsafe_offset=0] != v:
            raise Error("metal4 round trip read a stale value")
    sort(ts)
    return Float64(ts[len(ts) // 2]) / 1e3


def roundtrip_max(ctx: DeviceContext, c: DeviceBuffer[DType.float32]) raises -> Float64:
    var ts = List[Int]()
    for i in range(400):
        var t0 = Int(perf_counter_ns())
        ctx.enqueue_function[stamp](c, Float32(i + 1), grid_dim=1, block_dim=1)
        ctx.synchronize()
        ts.append(Int(perf_counter_ns()) - t0)
    sort(ts)
    return Float64(ts[len(ts) // 2]) / 1e3


def us(t0: Int, t1: Int, n: Int) -> Float64:
    return Float64(t1 - t0) / 1e3 / Float64(n)


def ours_dependent(mut rt: MetalRT, c: Buffer) raises -> Float64:
    """Best-of-3 us/launch for K dependent launches, then sync."""
    var best = Float64(1e30)
    for r in range(4):
        c.host_ptr[DType.float32]()[unsafe_offset=0] = 0.0
        var t0 = perf_counter_ns()
        for _ in range(K):
            rt.launch[bump](c.dev[DType.float32](), grid=1, block=1)
        rt.sync()
        var t = us(t0, perf_counter_ns(), K)
        if r > 0 and t < best:
            best = t
    var got = c.host_ptr[DType.float32]()[unsafe_offset=0]
    if got != Float32(K):
        print("  !! LOST UPDATES: counter", got, "of", K, "(batch", rt.batch_n,
              "order", rt.order, "done", rt.done, "residency", rt.residency, ")")
        return -1.0
    return best


def max_dependent(ctx: DeviceContext, c: DeviceBuffer[DType.float32]) raises -> Float64:
    var best = Float64(1e30)
    for r in range(4):
        ctx.enqueue_memset(c, 0.0)
        ctx.synchronize()
        var t0 = perf_counter_ns()
        for _ in range(K):
            ctx.enqueue_function[bump](c, grid_dim=1, block_dim=1)
        ctx.synchronize()
        var t = us(t0, perf_counter_ns(), K)
        if r > 0 and t < best:
            best = t
    with c.map_to_host() as h:
        if h[0] != Float32(K):
            raise Error("MAX dependent launches lost an update")
    return best


def main() raises:
    # ---- 1. msgSend dispatch cost -------------------------------------
    var o = ObjC()
    var cls = o.cls("NSObject")
    var sel = o.sel("hash")
    comptime M = 200000
    var acc = 0
    var t0 = perf_counter_ns()
    for _ in range(M):
        acc += o.send0(cls, sel)
    var t1 = perf_counter_ns()
    for _ in range(M):
        acc += o.objc.get_function[Int]("objc_msgSend")(cls, sel)
    var t2 = perf_counter_ns()
    for _ in range(M):
        acc += o.send0(cls, o.sel("hash"))
    var t3 = perf_counter_ns()
    print("## 1. objc_msgSend, ns/call  (", acc % 7, ")")
    print("| cached fn ptr | get_function (dlsym) per call | cached + sel_registerName per call |")
    print("|---|---|---|")
    print("|", Float64(t1 - t0) / Float64(M), "|", Float64(t2 - t1) / Float64(M), "|",
          Float64(t3 - t2) / Float64(M), "|")

    # ---- 2. host-side encode cost --------------------------------------
    var rt = MetalRT(policy=COMMIT_EVERY_N, batch_n=1 << 30)
    var c = rt.alloc[DType.float32](1)
    print("\nGPU:", rt.name())
    for _ in range(100):
        rt.launch[bump](c.dev[DType.float32](), grid=1, block=1)
    rt.sync()
    var best_enc = Float64(1e30)
    for _ in range(3):
        var e0 = perf_counter_ns()
        for _ in range(K):
            rt.launch[bump](c.dev[DType.float32](), grid=1, block=1)
        var e1 = perf_counter_ns()
        rt.sync()
        best_enc = min(best_enc, us(e0, e1, K))
    print("\n## 2. host encode cost (setPSO-cached + setBytes + dispatch), us/launch:", best_enc)

    # ---- 3. per-command-buffer batching --------------------------------
    print("\n## 3.", K, "dependent 1-thread launches, us/launch (-1 = LOST UPDATES)")
    print("| launches / cb | fence + status (default) | fence + status, blocking sync | event wait chain + event | tracked + useResources/cb | fence + event signal | no ordering (unsafe) |")
    print("|---|---|---|---|---|---|---|")
    for n in [1, 4, 16, 64, K]:
        var row = String("| ") + String(n)
        for cfg in range(6):
            var order = ORDER_FENCE
            var done = DONE_STATUS
            var spin = True
            if cfg == 1:
                spin = False
            elif cfg == 2:
                order = ORDER_EVENT
                done = DONE_EVENT
            elif cfg == 3:
                order = ORDER_TRACKED
            elif cfg == 4:
                done = DONE_EVENT
            elif cfg == 5:
                order = ORDER_NONE
            var r = MetalRT(policy=COMMIT_EVERY_N, batch_n=n, order=order, done=done, spin_sync=spin)
            var cc = r.alloc[DType.float32](1)
            row += " | " + String(ours_dependent(r, cc))
        print(row + " |")
    var rad = MetalRT(policy=COMMIT_ADAPTIVE, batch_n=16)
    var cad = rad.alloc[DType.float32](1)
    print("| adaptive (N=16) |", ours_dependent(rad, cad), "| | | | | |")
    var rad2 = MetalRT(policy=COMMIT_ADAPTIVE, batch_n=64)
    var cad2 = rad2.alloc[DType.float32](1)
    print("| adaptive (N=64) |", ours_dependent(rad2, cad2), "| | | | | |")

    # ---- 3b. synced round trip ---------------------------------------------
    print("\n## 3b. one 1-thread launch + wait for it, median us (400 samples)")
    print("| MAX enqueue_function + synchronize | ours, spin on status | ours, waitUntilCompleted | ours, spin on event (sync_event, default) | ours, spin on the value the kernel wrote |")
    print("|---|---|---|---|---|")
    var ctx0 = DeviceContext()
    var mc0 = ctx0.enqueue_create_buffer[DType.float32](1)
    var rr = String("| ") + String(roundtrip_max(ctx0, mc0))
    for cfg in range(4):
        var r = MetalRT(policy=COMMIT_EVERY_N, batch_n=1 << 30, spin_sync=cfg != 1, sync_event=cfg == 2)
        var cc = r.alloc[DType.float32](1)
        rr += " | " + String(roundtrip_ours(r, cc, cfg == 3))
    print(rr + " |")

    # ---- 3c. Metal 4 ----------------------------------------------------
    print("\n## 3c. Metal 4 path,", K, "dependent 1-thread launches, us/launch (-1 = LOST UPDATES)")
    print("| launches / cb | Metal 4 (barriers) | Metal 4, no barriers (unsafe) |")
    print("|---|---|---|")
    for n in [1, 4, 16, 64, K]:
        var ra = Metal4RT(policy=COMMIT_EVERY_N, batch_n=n)
        var ca = ra.alloc[DType.float32](1)
        var rb = Metal4RT(policy=COMMIT_EVERY_N, batch_n=n, barriers=False)
        var cb = rb.alloc[DType.float32](1)
        print("|", n, "|", m4_dependent(ra, ca), "|", m4_dependent(rb, cb), "|")
    var r4 = Metal4RT(policy=COMMIT_EVERY_N, batch_n=1 << 30)
    var c4 = r4.alloc[DType.float32](1)
    var e0 = perf_counter_ns()
    for _ in range(K):
        r4.launch[bump](c4.dev[DType.float32](), grid=1, block=1)
    var e1 = perf_counter_ns()
    r4.sync()
    print("Metal 4 host encode cost, us/launch:", us(e0, e1, K))
    var r4b = Metal4RT(policy=COMMIT_EVERY_N, batch_n=1 << 30)
    var c4b = r4b.alloc[DType.float32](1)
    print("Metal 4 one launch + sync (event spin), median us:", m4_roundtrip(r4b, c4b))

    # ---- 4. live buffers -----------------------------------------------
    print("\n## 4. live buffers,", K, "dependent launches, us/launch")
    print("| live extra buffers | MAX enqueue_function | ours set, 1/cb | ours set, 16/cb | ours per-cb useResources, 1/cb | ours per-cb, 16/cb | ours per-launch useResources, 16/cb |")
    print("|---|---|---|---|---|---|---|")
    var ctx = DeviceContext()
    for nlive in [0, 256, 1024]:
        var keep = List[DeviceBuffer[DType.float32]]()
        for _ in range(nlive):
            keep.append(ctx.enqueue_create_buffer[DType.float32](16))
        var mc = ctx.enqueue_create_buffer[DType.float32](1)
        var t_max = max_dependent(ctx, mc)
        _ = keep^
        var row = String("| ") + String(nlive) + " | " + String(t_max)
        for cfg in range(5):
            var res = RESIDENCY_SET
            var n = 1
            if cfg == 1:
                n = 16
            elif cfg == 2:
                res = RESIDENCY_PER_CB
            elif cfg == 3:
                res = RESIDENCY_PER_CB
                n = 16
            elif cfg == 4:
                res = RESIDENCY_PER_LAUNCH
                n = 16
            var r = MetalRT(policy=COMMIT_EVERY_N, batch_n=n, residency=res)
            var extra = List[Buffer]()
            for _ in range(nlive):
                extra.append(r.alloc[DType.float32](16))
            var cc = r.alloc[DType.float32](1)
            row += " | " + String(ours_dependent(r, cc))
        print(row + " |")
