"""Probe: which kernels make MAX's compiled MLP step slower than the eager
tape on Metal? Single-op MO graphs at the step's shapes, each timed against
the eager tape's kernel for the same op, both on the GPU.

    mm        x[B,H] @ w[H,H]                      (forward)
    mm_tA     transpose(x) @ g                     (weight gradient; MAX may fuse the transpose)
    mm_tA_in  xt @ g                               (xt passed already transposed)
    mm_tA_st  transpose(x) stored to a buffer, loaded back, @ g
    mm_tB     g @ transpose(w)                     (input gradient)
    tr        transpose(x)
    sum0      reduce.add over axis 0 of g          (bias gradient)

MAX: one `execute` + device synchronize per call, the fastest of REPS, minus
a 1x1 add graph (the round trip). Eager: the same op synced per call, and
amortised over 20 launches per sync (what a step without syncs pays).

Build and run: `noeira_max/tape_to_mo/run_probe.sh`.
"""

from std.random import seed, random_float64
from std.sys import argv
from std.time import perf_counter_ns

from maxrt import HostBuffer, Runtime, Tensor
from max.gpu.host import DeviceContext, DeviceBuffer
from noeira.nn.constants import DT
from noeira.experimental.functional.gpu import (
    gpu_matmul,
    gpu_matmul_bwd_a,
    gpu_matmul_bwd_b,
    gpu_transpose,
    gpu_sum_rows,
)


comptime REPS = 30
comptime AMORT = 20


def ty(r: Int, c: Int) -> String:
    return String("!mo.tensor<[", r, ", ", c, "], f32, gpu:0>")


def bty(r: Int, c: Int) -> String:
    return String("!mo.buffer<[", r, ", ", c, "], f32, gpu:0>")


comptime PERM = "  %p = mo.constant <#M.dense_array<1, 0> : tensor<2xsi64>> : !mo.tensor<[2], si64>\n"
comptime PTY = "!mo.tensor<[2], si64>"


def graph(name: String, args: List[String], tys: List[String], body: String, out_ty: String) -> String:
    var s = String("mo.graph @", name, "(")
    for i in range(len(tys)):
        if i > 0:
            s += ", "
        s += String("%arg", i, ": ", tys[i])
    s += String(") -> ", out_ty, " attributes {_kernel_library_paths = [], argument_names = [")
    for i in range(len(args)):
        if i > 0:
            s += ", "
        s += "\"" + args[i] + "\""
    s += "], result_names = [\"o\"]} {\n  %chain0 = mo.chain.create()\n" + body
    return s + "  mo.output %o : " + out_ty + "\n}\n"


def rand(n: Int) -> List[Scalar[DT]]:
    var v = List[Scalar[DT]](capacity=n)
    for _ in range(n):
        v.append(Scalar[DT](random_float64(-1.0, 1.0)))
    return v^


struct MaxResult(Movable):
    var us: Float64
    var compile_s: Float64
    var vals: List[Float32]

    def __init__(out self):
        self.us = 1e30
        self.compile_s = 0
        self.vals = List[Float32]()


def run_max(
    rt: Runtime, path: String, text: String, args: List[String],
    names: List[String], shapes: List[List[Int]], dev: List[Tensor],
) raises -> MaxResult:
    var r = MaxResult()
    with open(path, "w") as f:
        f.write(text)
    var t0 = perf_counter_ns()
    var model = rt.load(path)
    r.compile_s = Float64(perf_counter_ns() - t0) / 1e9
    var inputs = rt.tensor_map()
    for a in args:
        for i in range(len(names)):
            if names[i] == a:
                inputs.borrow_address(a, dev[i].address(), DType.float32, shapes[i].copy(), on_device=True)
    for k in range(REPS):
        var t1 = perf_counter_ns()
        var out = model.execute(inputs)
        rt.synchronize()
        r.us = min(r.us, Float64(perf_counter_ns() - t1) / 1e3)
        if k == REPS - 1:
            var h = out.tensor("o").to_host()
            var p = h.data[Float32]()
            for i in range(h.num_elements()):
                r.vals.append(p[unsafe_offset=i])
    return r^


struct Eager(Movable):
    var x: DeviceBuffer[DT]
    var w: DeviceBuffer[DT]
    var g: DeviceBuffer[DT]
    var xt: DeviceBuffer[DT]
    var o_bh: DeviceBuffer[DT]
    var o_hh: DeviceBuffer[DT]
    var o_hb: DeviceBuffer[DT]
    var o_1h: DeviceBuffer[DT]
    var tmp_t: DeviceBuffer[DT]
    var tmp_o: DeviceBuffer[DT]

    def __init__(
        out self, ctx: DeviceContext, B: Int, H: Int, x: List[Scalar[DT]],
        w: List[Scalar[DT]], g: List[Scalar[DT]], xt: List[Scalar[DT]],
    ) raises:
        self.x = up(ctx, x)
        self.w = up(ctx, w)
        self.g = up(ctx, g)
        self.xt = up(ctx, xt)
        self.o_bh = ctx.enqueue_create_buffer[DT](B * H)
        self.o_hh = ctx.enqueue_create_buffer[DT](H * H)
        self.o_hb = ctx.enqueue_create_buffer[DT](H * B)
        self.o_1h = ctx.enqueue_create_buffer[DT](H)
        self.tmp_t = ctx.enqueue_create_buffer[DT](B * H)
        self.tmp_o = ctx.enqueue_create_buffer[DT](B * H)


def up(ctx: DeviceContext, v: List[Scalar[DT]]) raises -> DeviceBuffer[DT]:
    var b = ctx.enqueue_create_buffer[DT](len(v))
    with b.map_to_host() as h:
        for i in range(len(v)):
            h[i] = v[i]
    return b^


def eager_op(ctx: DeviceContext, e: Eager, op: Int, B: Int, H: Int) raises:
    if op == 0:
        gpu_matmul(ctx, e.o_bh, e.x, e.w, B, H, H)
    elif op == 1:
        gpu_matmul_bwd_b[False](ctx, e.o_hh, e.x, e.g, B, H, H, e.tmp_t, e.tmp_o)
    elif op == 2:
        gpu_matmul(ctx, e.o_hh, e.xt, e.g, H, H, B)
    elif op == 3:
        gpu_matmul_bwd_a[False](ctx, e.o_bh, e.g, e.w, B, H, H, e.tmp_o)
    elif op == 4:
        gpu_transpose(ctx, e.o_hb, e.x, B, H)
    else:
        gpu_sum_rows(ctx, e.o_1h, e.g, B, H)


def eager_out(e: Eager, op: Int) raises -> List[Float32]:
    var v = List[Float32]()
    if op == 0 or op == 3:
        with e.o_bh.map_to_host() as h:
            for i in range(len(e.o_bh)):
                v.append(Float32(h[i]))
    elif op == 1 or op == 2:
        with e.o_hh.map_to_host() as h:
            for i in range(len(e.o_hh)):
                v.append(Float32(h[i]))
    elif op == 4:
        with e.o_hb.map_to_host() as h:
            for i in range(len(e.o_hb)):
                v.append(Float32(h[i]))
    else:
        with e.o_1h.map_to_host() as h:
            for i in range(len(e.o_1h)):
                v.append(Float32(h[i]))
    return v^


def rel_diff(a: List[Float32], b: List[Float32]) -> Float64:
    if len(a) != len(b):
        return 1e30
    var worst: Float64 = 0
    var scale: Float64 = 1e-30
    for i in range(len(a)):
        worst = max(worst, abs(Float64(a[i]) - Float64(b[i])))
        scale = max(scale, abs(Float64(b[i])))
    return worst / scale


def probe(rt: Runtime, ctx: DeviceContext, B: Int, H: Int, dir: String) raises:
    seed(7)
    var x = rand(B * H)
    var w = rand(H * H)
    var g = rand(B * H)
    var xt = List[Scalar[DT]](length=B * H, fill=0)
    for i in range(B):
        for j in range(H):
            xt[j * B + i] = x[i * H + j]
    var one: List[Scalar[DT]] = [1.0]

    var staging = rt.tensor_map()
    var dev = List[Tensor]()
    var names = List[String]()
    var shapes = List[List[Int]]()

    def put(name: String, vals: List[Scalar[DT]], shape: List[Int]) raises {mut staging, mut dev, mut names, mut shapes, rt}:
        staging.borrow(name, HostBuffer(copy_from=Int(vals.unsafe_ptr()), nbytes=4 * len(vals)),
                       DType.float32, shape.copy())
        dev.append(staging.tensor(name).to_device(rt.device()))
        names.append(name)
        shapes.append(shape.copy())

    put("x", x, [B, H])
    put("w", w, [H, H])
    put("g", g, [B, H])
    put("xt", xt, [H, B])
    put("buf", xt, [H, B])
    put("a11", one, [1, 1])

    var e = Eager(ctx, B, H, x, w, g, xt)
    ctx.synchronize()

    var tag = String(B, "x", H)
    var base = run_max(rt, String(dir, "/probe_base.mlir"),
                       graph("base", ["a11"], [ty(1, 1)],
                             String("  %o = rmo.add(%arg0, %arg0) : (", ty(1, 1), ", ", ty(1, 1), ") -> ", ty(1, 1), "\n"),
                             ty(1, 1)),
                       ["a11"], names, shapes, dev)
    print(String("\n[B=", B, ", H=", H, "]  MAX round trip (1x1 add graph): ", base.us, " us"))
    print("  op        | MAX us (- trip) | eager synced us | eager amortised us | MAX GF/s | eager GF/s | rel diff | compile s")

    var ops: List[String] = ["mm", "mm_tA", "mm_tA_in", "mm_tA_st", "mm_tB", "tr", "sum0"]
    for k in range(len(ops)):
        var op = ops[k]
        var text: String
        var args: List[String]
        var eop: Int
        if op == "mm":
            args = ["x", "w"]
            text = graph(op, args, [ty(B, H), ty(H, H)],
                         String("  %o = rmo.matmul(%arg0, %arg1) : (", ty(B, H), ", ", ty(H, H), ") -> ", ty(B, H), "\n"),
                         ty(B, H))
            eop = 0
        elif op == "mm_tA":
            args = ["x", "g"]
            text = graph(op, args, [ty(B, H), ty(B, H)],
                         String(PERM,
                                "  %t = rmo.mo.transpose(%arg0, %p) : (", ty(B, H), ", ", PTY, ") -> ", ty(H, B), "\n",
                                "  %o = rmo.matmul(%t, %arg1) : (", ty(H, B), ", ", ty(B, H), ") -> ", ty(H, H), "\n"),
                         ty(H, H))
            eop = 1
        elif op == "mm_tA_in":
            args = ["xt", "g"]
            text = graph(op, args, [ty(H, B), ty(B, H)],
                         String("  %o = rmo.matmul(%arg0, %arg1) : (", ty(H, B), ", ", ty(B, H), ") -> ", ty(H, H), "\n"),
                         ty(H, H))
            eop = 2
        elif op == "mm_tA_st":
            args = ["buf", "x", "g"]
            text = graph(op, args, [bty(H, B), ty(B, H), ty(B, H)],
                         String(PERM,
                                "  %t = rmo.mo.transpose(%arg1, %p) : (", ty(B, H), ", ", PTY, ") -> ", ty(H, B), "\n",
                                "  %c1 = rmo.mo.mutable.store(%arg0, %t, %chain0) : (", bty(H, B), ", ", ty(H, B),
                                ", !mo.chain) -> !mo.chain\n",
                                "  %l, %c2 = rmo.mo.mutable.load(%arg0, %c1) : (", bty(H, B),
                                ", !mo.chain) -> (", ty(H, B), ", !mo.chain)\n",
                                "  %o = rmo.matmul(%l, %arg2) : (", ty(H, B), ", ", ty(B, H), ") -> ", ty(H, H), "\n"),
                         ty(H, H))
            eop = 1
        elif op == "mm_tB":
            args = ["g", "w"]
            text = graph(op, args, [ty(B, H), ty(H, H)],
                         String(PERM,
                                "  %t = rmo.mo.transpose(%arg1, %p) : (", ty(H, H), ", ", PTY, ") -> ", ty(H, H), "\n",
                                "  %o = rmo.matmul(%arg0, %t) : (", ty(B, H), ", ", ty(H, H), ") -> ", ty(B, H), "\n"),
                         ty(B, H))
            eop = 3
        elif op == "tr":
            args = ["x"]
            text = graph(op, args, [ty(B, H)],
                         String(PERM, "  %o = rmo.mo.transpose(%arg0, %p) : (", ty(B, H), ", ", PTY, ") -> ", ty(H, B), "\n"),
                         ty(H, B))
            eop = 4
        else:
            args = ["g"]
            text = graph(op, args, [ty(B, H)],
                         String("  %o = rmo.mo.reduce.add(%arg0) [axis = 0] : (", ty(B, H), ") -> ", ty(1, H), "\n"),
                         ty(1, H))
            eop = 5
        var m = run_max(rt, String(dir, "/probe_", tag, "_", op, ".mlir"), text, args, names, shapes, dev)

        var synced: Float64 = 1e30
        for _ in range(REPS):
            var t0 = perf_counter_ns()
            eager_op(ctx, e, eop, B, H)
            ctx.synchronize()
            synced = min(synced, Float64(perf_counter_ns() - t0) / 1e3)
        var amort: Float64 = 1e30
        for _ in range(5):
            var t0 = perf_counter_ns()
            for _ in range(AMORT):
                eager_op(ctx, e, eop, B, H)
            ctx.synchronize()
            amort = min(amort, Float64(perf_counter_ns() - t0) / 1e3 / Float64(AMORT))
        var d = rel_diff(m.vals, eager_out(e, eop))

        var net = m.us - base.us
        var gf_m = String("-")
        var gf_e = String("-")
        if eop <= 3:
            var flop = 2.0 * Float64(B) * Float64(H) * Float64(H)
            gf_m = String(Int(flop / net / 1e3))
            gf_e = String(Int(flop / amort / 1e3))
        print(String("  ", op, " | ", Int(net), " (", Int(m.us), ") | ", Int(synced), " | ", Int(amort),
                     " | ", gf_m, " | ", gf_e, " | ", d, " | ", m.compile_s))
    _ = staging^
    _ = dev^
    _ = x^
    _ = w^
    _ = g^
    _ = xt^
    _ = one^


def probe_inplace(rt: Runtime, dir: String) raises:
    """What an in-place update (Adam's shape) costs: `v = l + l` with `l`
    loaded from a buffer and `v` stored back, against the same graph on a
    plain tensor argument. Both output `reduce.add(v, axis 0)` so `v` is live."""
    print("\n[in place: load + store of a [R, 512] buffer vs a tensor argument]")
    print("  R | MB | tensor us | buffer us | load+store us | GB/s moved")
    var rows: List[Int] = [128, 512, 1024, 2048]
    for R in rows:
        var C = 512
        var vals = rand(R * C)
        var staging = rt.tensor_map()
        var dev = List[Tensor]()
        var names: List[String] = ["x", "buf"]
        var shapes = List[List[Int]]()
        for n in names:
            staging.borrow(n, HostBuffer(copy_from=Int(vals.unsafe_ptr()), nbytes=4 * R * C), DType.float32, [R, C])
            dev.append(staging.tensor(n).to_device(rt.device()))
            shapes.append([R, C])
        var t = ty(R, C)
        var t1 = ty(1, C)
        var red = String("  %o = rmo.mo.reduce.add(%v) [axis = 0] : (", t, ") -> ", t1, "\n")
        var plain = graph("plain", ["x"], [t],
                          String("  %v = rmo.add(%arg0, %arg0) : (", t, ", ", t, ") -> ", t, "\n", red), t1)
        var inplace = graph("inplace", ["buf"], [bty(R, C)],
                            String("  %l, %c1 = rmo.mo.mutable.load(%arg0, %chain0) : (", bty(R, C),
                                   ", !mo.chain) -> (", t, ", !mo.chain)\n",
                                   "  %v = rmo.add(%l, %l) : (", t, ", ", t, ") -> ", t, "\n",
                                   "  %c2 = rmo.mo.mutable.store(%arg0, %v, %c1) : (", bty(R, C), ", ", t,
                                   ", !mo.chain) -> !mo.chain\n", red), t1)
        var a = run_max(rt, String(dir, "/probe_plain_", R, ".mlir"), plain, ["x"], names, shapes, dev)
        var b = run_max(rt, String(dir, "/probe_inplace_", R, ".mlir"), inplace, ["buf"], names, shapes, dev)
        var mb = Float64(4 * R * C) / 1e6
        var extra = b.us - a.us
        print(String("  ", R, " | ", mb, " | ", Int(a.us), " | ", Int(b.us), " | ", Int(extra), " | ",
                     2.0 * mb / extra * 1e3 if extra > 0 else 0.0))
        _ = staging^
        _ = dev^
        _ = vals^


def probe_fusion(rt: Runtime, dir: String) raises:
    """Does a `mutable.store` stop MAX fusing the elementwise chain that
    feeds it? A chain of K ops (`v = v * 0.999 + 0.001`, 2 ops per link)
    on a [256, 256] tensor: OUTPUT it (from a tensor argument), or STORE it
    back into the buffer it was loaded from. If the store blocks fusion,
    its time grows with K by about one launch per op; the output's does not.
    Min of 30 synced executes; interleaved over 3 rounds."""
    print("\n[fusion under a store: K elementwise ops on [256, 256], us]")
    print("  K | output | store | store - output")
    var R = 256
    var C = 256
    var vals = rand(R * C)
    var one: List[Scalar[DT]] = [1.0]
    var staging = rt.tensor_map()
    var dev = List[Tensor]()
    var names: List[String] = ["x", "buf", "s"]
    var shapes: List[List[Int]] = [[R, C], [R, C], [1, 1]]
    for i in range(3):
        if i < 2:
            staging.borrow(names[i], HostBuffer(copy_from=Int(vals.unsafe_ptr()), nbytes=4 * R * C),
                           DType.float32, shapes[i].copy())
        else:
            staging.borrow(names[i], HostBuffer(copy_from=Int(one.unsafe_ptr()), nbytes=4),
                           DType.float32, shapes[i].copy())
        dev.append(staging.tensor(names[i]).to_device(rt.device()))
    var t = ty(R, C)
    var t11 = ty(1, 1)
    comptime ST = "!mo.tensor<[], f32, gpu:0>"
    var ks: List[Int] = [1, 4, 16, 32]
    for rd in range(3):
        for K in ks:
            var chain = String("  %ka = mo.constant <#M.dense_array<0.999> : tensor<f32>> : ", ST, "\n",
                               "  %kb = mo.constant <#M.dense_array<0.001> : tensor<f32>> : ", ST, "\n")
            for j in range(K):
                chain += String("  %m", j, " = rmo.mul(%v", j, ", %ka) : (", t, ", ", ST, ") -> ", t, "\n")
                chain += String("  %v", j + 1, " = rmo.add(%m", j, ", %kb) : (", t, ", ", ST, ") -> ", t, "\n")
            var st_g = graph("fuse_store", ["buf", "s"], [bty(R, C), t11],
                             String("  %v0, %c1 = rmo.mo.mutable.load(%arg0, %chain0) : (", bty(R, C),
                                    ", !mo.chain) -> (", t, ", !mo.chain)\n", chain,
                                    "  %c2 = rmo.mo.mutable.store(%arg0, %v", K, ", %c1) : (", bty(R, C), ", ", t,
                                    ", !mo.chain) -> !mo.chain\n",
                                    "  %o = rmo.add(%arg1, %kb) : (", t11, ", ", ST, ") -> ", t11, "\n"), t11)
            # the output graph: the same chain on a tensor argument
            var out2 = graph("fuse_out", ["x"], [t],
                             String("  %v0 = rmo.add(%arg0, %arg0) : (", t, ", ", t, ") -> ", t, "\n", chain,
                                    "  %o = rmo.add(%v", K, ", %kb) : (", t, ", ", ST, ") -> ", t, "\n"), t)
            var a = run_max(rt, String(dir, "/fuse_out_", K, ".mlir"), out2, ["x"], names, shapes, dev)
            var b = run_max(rt, String(dir, "/fuse_store_", K, ".mlir"), st_g, ["buf", "s"], names, shapes, dev)
            print(String("  ", K, " | ", Int(a.us), " | ", Int(b.us), " | ", Int(b.us - a.us)))
    _ = staging^
    _ = dev^
    _ = vals^
    _ = one^


def probe_side_store(rt: Runtime, dir: String) raises:
    """Does ONE unrelated store slow EVERY kernel of a graph? A chain of K
    matmuls on [64, 64] (K kernels MAX cannot fuse), output alone, or with a
    side store `s = s + 1` into an unrelated [1, 1] buffer. If a store makes
    MAX run the graph's kernels one at a time, the store arm grows by a
    fixed cost per kernel (the ~20 us Metal command-buffer retirement).
    Min of 30 synced executes, 3 interleaved rounds."""
    print("\n[one side store vs none: K chained [64, 64] matmuls, us]")
    print("  K | no store | + side store | difference | per kernel")
    var N = 64
    var vals = rand(N * N)
    for i in range(N * N):
        vals[i] = vals[i] * Scalar[DT](0.125)   # keeps the chain bounded
    var one: List[Scalar[DT]] = [1.0]
    var staging = rt.tensor_map()
    var dev = List[Tensor]()
    var names: List[String] = ["x", "w", "s"]
    var shapes: List[List[Int]] = [[N, N], [N, N], [1, 1]]
    for i in range(3):
        if i < 2:
            staging.borrow(names[i], HostBuffer(copy_from=Int(vals.unsafe_ptr()), nbytes=4 * N * N),
                           DType.float32, shapes[i].copy())
        else:
            staging.borrow(names[i], HostBuffer(copy_from=Int(one.unsafe_ptr()), nbytes=4),
                           DType.float32, shapes[i].copy())
        dev.append(staging.tensor(names[i]).to_device(rt.device()))
    var t = ty(N, N)
    var t11 = ty(1, 1)
    comptime ST = "!mo.tensor<[], f32, gpu:0>"
    var ks: List[Int] = [4, 16, 32]
    for rd in range(3):
        for K in ks:
            var chain = String("  %v0 = rmo.matmul(%arg0, %arg1) : (", t, ", ", t, ") -> ", t, "\n")
            for j in range(1, K):
                chain += String("  %v", j, " = rmo.matmul(%v", j - 1, ", %arg1) : (", t, ", ", t, ") -> ", t, "\n")
            chain += String("  %o = rmo.mo.tanh(%v", K - 1, ") : (", t, ") -> ", t, "\n")
            var plain = graph("mm_chain", ["x", "w"], [t, t], chain, t)
            var side = graph("mm_chain_store", ["x", "w", "s"], [t, t, bty(1, 1)],
                             String(chain,
                                    "  %ks = mo.constant <#M.dense_array<1.0> : tensor<f32>> : ", ST, "\n",
                                    "  %sl, %c1 = rmo.mo.mutable.load(%arg2, %chain0) : (", bty(1, 1),
                                    ", !mo.chain) -> (", t11, ", !mo.chain)\n",
                                    "  %sn = rmo.add(%sl, %ks) : (", t11, ", ", ST, ") -> ", t11, "\n",
                                    "  %c2 = rmo.mo.mutable.store(%arg2, %sn, %c1) : (", bty(1, 1), ", ", t11,
                                    ", !mo.chain) -> !mo.chain\n"), t)
            var a = run_max(rt, String(dir, "/mmc_", K, ".mlir"), plain, ["x", "w"], names, shapes, dev)
            var b = run_max(rt, String(dir, "/mmc_store_", K, ".mlir"), side, ["x", "w", "s"], names, shapes, dev)
            var d = b.us - a.us
            print(String("  ", K, " | ", Int(a.us), " | ", Int(b.us), " | ", Int(d), " | ", Int(d / Float64(K))))
    _ = staging^
    _ = dev^
    _ = vals^
    _ = one^


def probe_shared_store(rt: Runtime, dir: String) raises:
    """Adam's shape: a value that is BOTH stored and read by something else
    (m_t is stored and feeds the parameter update). NB = 20 buffers
    [64, 64]: v_i = l_i * 0.999 + 0.001, and the output sums every v_i.
    Arms: (out) tensor arguments, every v_i also an output; (store) v_i
    stored back AND summed; (store-only) v_i stored, the output a trivial
    add. If a store whose value has a second consumer is not fused (a
    materialise + copy kernel), (store) costs ~NB kernels more than (out)."""
    print("\n[stored values with a second consumer: 20 buffers [64, 64], us]")
    print("  out | store + sum | store only | (store + sum) - out")
    comptime NB = 20
    var N = 64
    var vals = rand(N * N)
    var one: List[Scalar[DT]] = [1.0]
    var staging = rt.tensor_map()
    var dev = List[Tensor]()
    var names = List[String]()
    var shapes = List[List[Int]]()
    for i in range(NB):
        names.append(String("b", i))
        shapes.append([N, N])
        staging.borrow(names[i], HostBuffer(copy_from=Int(vals.unsafe_ptr()), nbytes=4 * N * N),
                       DType.float32, [N, N])
        dev.append(staging.tensor(names[i]).to_device(rt.device()))
    names.append("s")
    shapes.append([1, 1])
    staging.borrow("s", HostBuffer(copy_from=Int(one.unsafe_ptr()), nbytes=4), DType.float32, [1, 1])
    dev.append(staging.tensor("s").to_device(rt.device()))
    var t = ty(N, N)
    var t1 = ty(1, N)
    var t11 = ty(1, 1)
    comptime ST = "!mo.tensor<[], f32, gpu:0>"
    var consts = String("  %ka = mo.constant <#M.dense_array<0.999> : tensor<f32>> : ", ST, "\n",
                        "  %kb = mo.constant <#M.dense_array<0.001> : tensor<f32>> : ", ST, "\n")
    def body(store: Bool, sum: Bool, t: String, t1: String, t11: String, consts: String) -> String:
        var b = consts
        var chain = String("%chain0")
        if store:   # every load first, then the math, then every store (the step's order)
            for i in range(NB):
                b += String("  %l", i, ", %cl", i, " = rmo.mo.mutable.load(%arg", i, ", ", chain, ") : (",
                            bty(64, 64), ", !mo.chain) -> (", t, ", !mo.chain)\n")
                chain = String("%cl", i)
        for i in range(NB):
            var src = String("%l", i) if store else String("%arg", i)
            b += String("  %m", i, " = rmo.mul(", src, ", %ka) : (", t, ", ", ST, ") -> ", t, "\n")
            b += String("  %v", i, " = rmo.add(%m", i, ", %kb) : (", t, ", ", ST, ") -> ", t, "\n")
        if store:
            for i in range(NB):
                b += String("  %cs", i, " = rmo.mo.mutable.store(%arg", i, ", %v", i, ", ", chain, ") : (",
                            bty(64, 64), ", ", t, ", !mo.chain) -> !mo.chain\n")
                chain = String("%cs", i)
        if sum:
            b += String("  %a0 = rmo.add(%v0, %v1) : (", t, ", ", t, ") -> ", t, "\n")
            for i in range(2, NB):
                b += String("  %a", i - 1, " = rmo.add(%a", i - 2, ", %v", i, ") : (", t, ", ", t, ") -> ", t, "\n")
            b += String("  %o = rmo.mo.reduce.add(%a", NB - 2, ") [axis = 0] : (", t, ") -> ", t1, "\n")
        else:
            b += String("  %o = rmo.add(%arg", NB, ", %kb) : (", t11, ", ", ST, ") -> ", t11, "\n")
        return b
    var argn = List[String]()
    var tys_t = List[String]()
    var tys_b = List[String]()
    for i in range(NB):
        argn.append(names[i])
        tys_t.append(t)
        tys_b.append(bty(N, N))
    # (out): every v_i is an output too, so each is materialised once
    var outb = body(False, True, t, t1, t11, consts)
    var res = String("\"o\"")
    var oids = String("%o")
    var otys = t1
    for i in range(NB):
        res += String(", \"v", i, "\"")
        oids += String(", %v", i)
        otys += String(", ", t)
    var out_g = graph("shared_out", argn, tys_t, outb, t1)
    out_g = out_g.replace("result_names = [\"o\"]", String("result_names = [", res, "]"))
    out_g = out_g.replace(String("mo.output %o : ", t1), String("mo.output ", oids, " : ", otys))
    out_g = out_g.replace(String(") -> ", t1, " attributes"), String(") -> (", otys, ") attributes"))
    var st_g = graph("shared_store", argn, tys_b, body(True, True, t, t1, t11, consts), t1)
    var argn2 = argn.copy()
    argn2.append("s")
    var tys_b2 = tys_b.copy()
    tys_b2.append(t11)
    var so_g = graph("store_only", argn2, tys_b2, body(True, False, t, t1, t11, consts), t11)
    for _ in range(3):
        var a = run_max(rt, dir + "/shared_out.mlir", out_g, argn, names, shapes, dev)
        var b = run_max(rt, dir + "/shared_store.mlir", st_g, argn, names, shapes, dev)
        var c = run_max(rt, dir + "/store_only.mlir", so_g, argn2, names, shapes, dev)
        print(String("  ", Int(a.us), " | ", Int(b.us), " | ", Int(c.us), " | ", Int(b.us - a.us)))
    _ = staging^
    _ = dev^
    _ = vals^
    _ = one^


def main() raises:
    var dir = String(argv()[1])
    var rt = Runtime(accelerator=True)
    var ctx = DeviceContext()
    if len(argv()) > 2 and String(argv()[2]) == "--shared-store":
        probe_shared_store(rt, dir)
        return
    if len(argv()) > 2 and String(argv()[2]) == "--side-store":
        probe_side_store(rt, dir)
        return
    if len(argv()) > 2 and String(argv()[2]) == "--fusion":
        probe_fusion(rt, dir)
        return
    probe(rt, ctx, 1024, 512, dir)
    probe(rt, ctx, 256, 256, dir)
    probe_inplace(rt, dir)
