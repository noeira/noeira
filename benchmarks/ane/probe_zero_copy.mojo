"""Can the GPU (through metal_rt) and the Neural Engine share memory, with no
copy? The ANE's fp16 IOSurfaces are wrapped as Metal buffers
(`MetalRT.wrap_external`: `newBufferWithBytesNoCopy`, page-aligned surface
memory, nil deallocator); a GPU kernel writes the ANE's inputs in place and
another reads its output in place. Exits 1 on failure.

    pixi run -e apple mojo run -I . benchmarks/ane/probe_zero_copy.mojo

a. GPU -> ANE: GPU kernels write x, w, b of `y = x @ w + b` (fp16, at the
   padded row stride the model reports); the CPU reads y (Program.read) and
   checks it against a float64 reference.
b. ANE -> GPU: a GPU kernel reads y from the output surface (padded) into a
   plain fp32 metal_rt buffer; the CPU checks that.
c. Coherency: 200 iterations of y = a + b with values that change every
   iteration, every element checked every time (exact), with no
   IOSurfaceLock anywhere; then again with lock/unlock around each handoff.
d. Shapes with padding: B=17 I=5 O=6, and B=256 I=64 O=64.
e. Timing: median us per iteration of the handoff loop (GPU write, sync,
   ANE run, GPU read, sync) vs the CPU path (generate, Program.write,
   ANE run, Program.read), at B=4096 I=O=512 and at the small shapes; and
   the training step's shape: ONE GPU command buffer per iteration (read the
   output, write the next inputs), one sync, ANE run.
f. Mutant: the write kernel uses the DENSE row stride; the check must fail.
"""

from std.sys import exit
from std.time import perf_counter_ns
from max.gpu import block_idx, thread_idx

from noeira.experimental.metal_rt.objc import _sym, _as_fn, CFn1
from noeira.experimental.metal_rt.runtime import MetalRT, Buffer, COMMIT_EVERY_N
from noeira.experimental.ane.runtime import ANE, Program, mil_header

comptime P16 = Pointer[Float16, MutAnyOrigin]
comptime P32 = Pointer[Float32, MutAnyOrigin]


@always_inline
def val(r: Int, c: Int, it: Int, k: Int) -> Float32:
    """Deterministic, exact in fp16 (multiples of 1/8 in [-1.75, 1.75])."""
    return Float32((r * 7 + c * 13 + it * 3 + k * 5) % 29 - 14) / 8.0


@always_inline
def val32(r: Int32, c: Int32, it: Int32, k: Int32) -> Float32:
    """`val` in 32-bit integer math. With `Int` (64-bit) indices and a 1-D
    grid divided into (row, col), the write kernel took 2-5 ms at B=4096,
    I=512: the GPU emulates 64-bit division."""
    return Float32((r * 7 + c * 13 + it * 3 + k * 5) % 29 - 14) / 8.0


def gen_kernel(dst: P16, rows_: Int64, cols_: Int64, stride_: Int64, it_: Int64, k_: Int64):
    """dst[r * stride + c] = val(r, c, it, k) in fp16: a [rows, cols] tensor
    at row stride `stride` (the surface's padded one, or the mutant's).
    Grid: x over columns (256 a group), y over rows: no index division."""
    var c = Int32(block_idx.x) * 256 + Int32(thread_idx.x)
    var r = Int32(block_idx.y)
    if c >= Int32(cols_) or r >= Int32(rows_):
        return
    dst[unsafe_offset=Int(r * Int32(stride_) + c)] = val32(r, c, Int32(it_), Int32(k_)).cast[DType.float16]()


def read_kernel(dst: P32, src: P16, rows_: Int64, cols_: Int64, stride_: Int64):
    """dst (dense fp32) = the [rows, cols] fp16 tensor at row stride `stride`."""
    var c = Int32(block_idx.x) * 256 + Int32(thread_idx.x)
    var r = Int32(block_idx.y)
    var cols = Int32(cols_)
    if c >= cols or r >= Int32(rows_):
        return
    dst[unsafe_offset=Int(r * cols + c)] = src[unsafe_offset=Int(r * Int32(stride_) + c)].cast[DType.float32]()


def t4(d0: Int, d1: Int, d2: Int, d3: Int) -> String:
    return String("tensor<fp16, [", d0, ", ", d1, ", ", d2, ", ", d3, "]>")


def linear_mil(B: Int, I: Int, O: Int) -> String:
    var m = mil_header()
    m += String("    func main<ios18>(", t4(1, 1, B, I), " i0, ", t4(1, 1, I, O), " i1, ", t4(1, 1, 1, O), " i2) {\n")
    m += "        bool bF = const()[name = string(\"bF\"), val = bool(false)];\n"
    m += String("        ", t4(1, 1, B, O), " z = matmul(transpose_x = bF, transpose_y = bF, x = i0, y = i1)[name = string(\"z\")];\n")
    m += String("        ", t4(1, 1, B, O), " y = add(x = z, y = i2)[name = string(\"y\")];\n")
    m += "    } -> (y);\n}\n"
    return m


def add_mil(R: Int, K: Int) -> String:
    var m = mil_header()
    m += String("    func main<ios18>(", t4(1, 1, R, K), " i0, ", t4(1, 1, R, K), " i1) {\n")
    m += String("        ", t4(1, 1, R, K), " y = add(x = i0, y = i1)[name = string(\"y\")];\n")
    m += "    } -> (y);\n}\n"
    return m


struct Wrapped(Movable):
    """The program's surfaces as metal_rt buffers."""

    var ins: List[Buffer]
    var outs: List[Buffer]

    def __init__(out self, mut rt: MetalRT, a: ANE, p: Program) raises:
        var alloc_size = _as_fn[CFn1](_sym(a.o.system, "IOSurfaceGetAllocSize"))
        self.ins = List[Buffer]()
        self.outs = List[Buffer]()
        for s in p.ins:
            self.ins.append(rt.wrap_external(a.base_addr(s), alloc_size(s)))
        for s in p.outs:
            self.outs.append(rt.wrap_external(a.base_addr(s), alloc_size(s)))


def g(n: Int) -> Int:
    return (n + 255) // 256


def gpu_write(mut rt: MetalRT, dst: Buffer, rows: Int, cols: Int, stride: Int, it: Int, k: Int) raises:
    rt.launch2d[gen_kernel](
        dst.dev[DType.float16](), Int64(rows), Int64(cols), Int64(stride), Int64(it), Int64(k),
        grid_x=g(cols), grid_y=rows, block_x=256,
    )


def gpu_read(mut rt: MetalRT, dst: Buffer, src: Buffer, rows: Int, cols: Int, stride: Int) raises:
    rt.launch2d[read_kernel](
        dst.dev[DType.float32](), src.dev[DType.float16](), Int64(rows), Int64(cols), Int64(stride),
        grid_x=g(cols), grid_y=rows, block_x=256,
    )


def ref_linear(B: Int, I: Int, O: Int, it: Int, r: Int, c: Int) -> Float64:
    var s = Float64(val(0, c, it, 2))
    for d in range(I):
        s += Float64(val(r, d, it, 0)) * Float64(val(d, c, it, 1))
    return s


def tol(ref_: Float64) -> Float64:
    """fp16 output: 1 ulp at max(|ref|, 1) (2^-10 relative), plus the
    ANE's accumulation; a stale or misplaced element is off by ~0.1 or more."""
    return max(abs(ref_), 1.0) * 2.0 ** -9


def lock_all(a: ANE, p: Program, read_only_out: Bool):
    for s in p.ins:
        a.lock(s, False)
    for s in p.outs:
        a.lock(s, read_only_out)


def unlock_all(a: ANE, p: Program, read_only_out: Bool):
    for s in p.ins:
        a.unlock(s, False)
    for s in p.outs:
        a.unlock(s, read_only_out)


def check_linear(
    mut rt: MetalRT, a: ANE, p: Program, w: Wrapped, B: Int, I: Int, O: Int,
    iters: Int, mutant: Bool, locks: Bool,
) raises -> Tuple[Int, Int]:
    """Bad elements in (a) the CPU's read of y, (b) the GPU's read of y."""
    var y32 = rt.alloc[DType.float32](B * O)
    var ycpu = List[Float32](length=B * O, fill=0)
    var bad_a = 0
    var bad_b = 0
    var L0 = p.in_layout[0].row
    var L1 = p.in_layout[1].row
    var L2 = p.in_layout[2].row
    var Ly = p.out_layout[0].row
    for it in range(iters):
        if locks:
            lock_all(a, p, False)
        gpu_write(rt, w.ins[0], B, I, I if mutant else L0, it, 0)
        gpu_write(rt, w.ins[1], I, O, O if mutant else L1, it, 1)
        gpu_write(rt, w.ins[2], 1, O, O if mutant else L2, it, 2)
        rt.sync()
        if locks:
            unlock_all(a, p, False)
        p.run(a)
        if locks:
            lock_all(a, p, True)
        gpu_read(rt, y32, w.outs[0], B, O, Ly)
        rt.sync()
        if locks:
            unlock_all(a, p, True)
        p.read(a, 0, P32(unsafe_from_address=Int(ycpu.unsafe_ptr())))
        var yg = y32.host_ptr[DType.float32]()
        for r in range(B):
            for c in range(O):
                var want = ref_linear(B, I, O, it, r, c)
                if abs(Float64(ycpu[r * O + c]) - want) > tol(want):
                    bad_a += 1
                if abs(Float64(yg[unsafe_offset=r * O + c]) - want) > tol(want):
                    bad_b += 1
    rt.free(y32)
    return (bad_a, bad_b)


def coherency(mut rt: MetalRT, a: ANE, R: Int, K: Int, iters: Int, locks: Bool) raises -> Int:
    var p = Program(a, add_mil(R, K))
    var w = Wrapped(rt, a, p)
    var y32 = rt.alloc[DType.float32](R * K)
    var bad = 0
    for it in range(iters):
        if locks:
            lock_all(a, p, False)
        gpu_write(rt, w.ins[0], R, K, p.in_layout[0].row, it, 0)
        gpu_write(rt, w.ins[1], R, K, p.in_layout[1].row, it, 1)
        rt.sync()
        if locks:
            unlock_all(a, p, False)
        p.run(a)
        if locks:
            lock_all(a, p, True)
        gpu_read(rt, y32, w.outs[0], R, K, p.out_layout[0].row)
        rt.sync()
        if locks:
            unlock_all(a, p, True)
        var yg = y32.host_ptr[DType.float32]()
        for r in range(R):
            for c in range(K):
                if yg[unsafe_offset=r * K + c] != val(r, c, it, 0) + val(r, c, it, 1):
                    bad += 1
    rt.free(y32)
    for b in w.ins:
        rt.free(b)
    for b in w.outs:
        rt.free(b)
    p.free(a)
    return bad


def median(var v: List[Float64]) -> Float64:
    sort(v)
    return v[len(v) // 2]


struct Times(Movable):
    """Median us per iteration: GPU path (write+sync, ANE, read+sync, total)
    and CPU path (generate, write, ANE, read, total)."""
    var g: List[Float64]
    var c: List[Float64]

    def __init__(out self):
        self.g = List[Float64]()
        self.c = List[Float64]()


def timing(mut rt: MetalRT, a: ANE, B: Int, I: Int, O: Int, iters: Int) raises -> Times:
    var p = Program(a, linear_mil(B, I, O))
    var w = Wrapped(rt, a, p)
    var y32 = rt.alloc[DType.float32](B * O)
    var tg = List[List[Float64]]()
    var tc = List[List[Float64]]()
    for _ in range(5):
        tg.append(List[Float64]())
    for _ in range(5):
        tc.append(List[Float64]())
    var hx = List[Float32](length=B * I, fill=0)
    var hw = List[Float32](length=I * O, fill=0)
    var hb = List[Float32](length=O, fill=0)
    var hy = List[Float32](length=B * O, fill=0)
    for it in range(iters + 3):
        var t0 = perf_counter_ns()
        gpu_write(rt, w.ins[0], B, I, p.in_layout[0].row, it, 0)
        gpu_write(rt, w.ins[1], I, O, p.in_layout[1].row, it, 1)
        gpu_write(rt, w.ins[2], 1, O, p.in_layout[2].row, it, 2)
        rt.sync()
        var t1 = perf_counter_ns()
        p.run(a)
        var t2 = perf_counter_ns()
        gpu_read(rt, y32, w.outs[0], B, O, p.out_layout[0].row)
        rt.sync()
        var t3 = perf_counter_ns()
        for r in range(B):
            for c in range(I):
                hx[r * I + c] = val(r, c, it, 0)
        for r in range(I):
            for c in range(O):
                hw[r * O + c] = val(r, c, it, 1)
        for c in range(O):
            hb[c] = val(0, c, it, 2)
        var t4_ = perf_counter_ns()
        p.write(a, 0, Pointer(to=hx[0]))
        p.write(a, 1, Pointer(to=hw[0]))
        p.write(a, 2, Pointer(to=hb[0]))
        var t5 = perf_counter_ns()
        p.run(a)
        var t6 = perf_counter_ns()
        p.read(a, 0, P32(unsafe_from_address=Int(hy.unsafe_ptr())))
        var t7 = perf_counter_ns()
        # the step's shape: ONE GPU command buffer per handoff (read the
        # output, write the next inputs), one sync, then the ANE
        gpu_read(rt, y32, w.outs[0], B, O, p.out_layout[0].row)
        gpu_write(rt, w.ins[0], B, I, p.in_layout[0].row, it, 0)
        gpu_write(rt, w.ins[1], I, O, p.in_layout[1].row, it, 1)
        gpu_write(rt, w.ins[2], 1, O, p.in_layout[2].row, it, 2)
        rt.sync()
        p.run(a)
        var t8 = perf_counter_ns()
        if it >= 3:
            tg[4].append(Float64(t8 - t7) / 1e3)
            tg[0].append(Float64(t1 - t0) / 1e3)
            tg[1].append(Float64(t2 - t1) / 1e3)
            tg[2].append(Float64(t3 - t2) / 1e3)
            tg[3].append(Float64(t3 - t0) / 1e3)
            tc[0].append(Float64(t4_ - t3) / 1e3)
            tc[1].append(Float64(t5 - t4_) / 1e3)
            tc[2].append(Float64(t6 - t5) / 1e3)
            tc[3].append(Float64(t7 - t6) / 1e3)
            tc[4].append(Float64(t7 - t3) / 1e3)
    # both paths must agree on the last iteration
    var yg = y32.host_ptr[DType.float32]()
    for i in range(B * O):
        if yg[unsafe_offset=i] != hy[i]:
            raise Error("timing: GPU and CPU paths read different outputs")
    rt.free(y32)
    for b in w.ins:
        rt.free(b)
    for b in w.outs:
        rt.free(b)
    p.free(a)
    var r = Times()
    for k in range(5):
        r.g.append(median(tg[k].copy()))
    for k in range(5):
        r.c.append(median(tc[k].copy()))
    return r^


def main() raises:
    var a = ANE()
    var rt = MetalRT(policy=COMMIT_EVERY_N, batch_n=16)
    print("GPU:", rt.name())
    var fails = 0

    print("(a, b, d) GPU writes the ANE's inputs in place, ANE runs, GPU and CPU read y; 20 iterations each")
    for sh in [(17, 5, 6), (256, 64, 64), (64, 32, 48), (4096, 512, 512)]:
        var B = sh[0]
        var I = sh[1]
        var O = sh[2]
        var p = Program(a, linear_mil(B, I, O))
        var w = Wrapped(rt, a, p)
        var dense = String("")
        for k in range(len(p.in_layout)):
            dense += String(" in", k, " row ", p.in_layout[k].row, " (width ", p.in_layout[k].width, ")")
        dense += String(" out row ", p.out_layout[0].row)
        var r = check_linear(rt, a, p, w, B, I, O, 20 if B < 4096 else 3, False, False)
        var ok = r[0] == 0 and r[1] == 0
        if not ok:
            fails += 1
        print("   B", B, "I", I, "O", O, "|", dense, "| (a) CPU read: bad", r[0], "| (b) GPU read: bad", r[1], "OK" if ok else "FAIL")
        for b in w.ins:
            rt.free(b)
        for b in w.outs:
            rt.free(b)
        p.free(a)

    print("(f) mutant: the GPU writes the inputs at the DENSE row stride")
    for sh in [(17, 5, 6), (256, 64, 64)]:
        var B = sh[0]
        var I = sh[1]
        var O = sh[2]
        var p = Program(a, linear_mil(B, I, O))
        var w = Wrapped(rt, a, p)
        var r = check_linear(rt, a, p, w, B, I, O, 3, True, False)
        var padded = p.in_layout[0].row != I or p.in_layout[1].row != O or p.in_layout[2].row != O
        var caught = r[0] > 0 and r[1] > 0
        print("   B", B, "I", I, "O", O, "| rows padded:", padded, "| bad", r[0], "/", r[1],
              "| mutant", "CAUGHT" if caught else ("not caught" + (" (FAIL)" if padded else " (no padding: nothing to catch)")))
        if padded and not caught:
            fails += 1
        for b in w.ins:
            rt.free(b)
        for b in w.outs:
            rt.free(b)
        p.free(a)

    print("(c) coherency: 200 iterations of y = a + b, every element checked")
    for sh in [(64, 40), (17, 6), (512, 512)]:
        var bad_nolock = coherency(rt, a, sh[0], sh[1], 200, False)
        var bad_lock = coherency(rt, a, sh[0], sh[1], 200, True)
        print("   [", sh[0], "x", sh[1], "] no locks: bad", bad_nolock, "| lock/unlock per handoff: bad", bad_lock)
        if bad_nolock != 0 or bad_lock != 0:
            fails += 1

    print("(e) timing, median us per iteration (GPU write -> sync -> ANE run -> GPU read -> sync) vs CPU path")
    print("| shape | zero-copy total | = GPU write+sync / ANE run / GPU read+sync | zero-copy, one sync per iteration | CPU path total | = generate / Program.write / ANE run / Program.read |")
    print("|---|---|---|---|---|---|")
    for sh in [(17, 5, 6), (256, 64, 64), (4096, 512, 512)]:
        var t = timing(rt, a, sh[0], sh[1], sh[2], 40 if sh[0] < 4096 else 20)
        print("| B", sh[0], "I", sh[1], "O", sh[2], "|", t.g[3], "|", t.g[0], "/", t.g[1], "/", t.g[2], "|",
              t.g[4], "|", t.c[4], "|", t.c[0], "/", t.c[1], "/", t.c[2], "/", t.c[3], "|")

    if fails != 0:
        print("FAILED:", fails)
        exit(1)
    print("ALL OK")
