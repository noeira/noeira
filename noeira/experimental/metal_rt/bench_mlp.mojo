"""2-layer MLP training step: MAX's launch path vs `MetalRT`, with a
bit-identity gate.

    pixi run -e apple mojo run -I . noeira/experimental/metal_rt/bench_mlp.mojo [--quick]

The SAME kernel functions (`mlp_kernels.mojo`, 14 launches per step) run
through `DeviceContext.enqueue_function` and through `MetalRT` under each
commit policy. Every run starts from the same seeded init and does
WARMUP + STEPS steps; the final W1, b1, W2, b2 and the loss of every step must
be bit-identical to the MAX run of the same size, or the program exits 1.

Loop shapes: loss read back every 20 steps, and every step (host waits each
step, the RL-style loop). Times are us/step over the STEPS timed steps,
including the final sync.

A third section puts host CPU work between each step's launches and its loss
read-back (CPU work that does not need the GPU result, e.g. an env step), to
show the batching policies' tail problem and the `flush()` fix.
"""

from std.math import sqrt
from std.memory import bitcast
from std.sys import argv, exit
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer

from noeira.experimental.metal_rt.metal4 import Metal4RT
from noeira.experimental.metal_rt.runtime import (
    MetalRT,
    Launcher,
    Buffer,
    COMMIT_EACH,
    COMMIT_EVERY_N,
    COMMIT_ADAPTIVE,
    ORDER_FENCE,
    ORDER_NONE,
)
from noeira.experimental.metal_rt.mlp_kernels import (
    TPB,
    STEP_LAUNCHES,
    fc_fwd,
    relu_fwd,
    mse_grad,
    mse_loss,
    grad_w,
    grad_b,
    grad_in_relu,
    adam,
)

# buffer slots
comptime X = 0
comptime Y = 1
comptime W1 = 2
comptime B1 = 3
comptime W2 = 4
comptime B2 = 5
comptime Z1 = 6
comptime A1 = 7
comptime Z2 = 8
comptime DZ2 = 9
comptime DW2 = 10
comptime DB2 = 11
comptime DZ1 = 12
comptime DW1 = 13
comptime DB1 = 14
comptime M0 = 15  # M0..M0+3: Adam first moments of W1, b1, W2, b2
comptime V0 = 19  # V0..V0+3: second moments
comptime LOSS = 23
comptime NBUF = 24

comptime LR: Float32 = 1e-3
comptime BETA1: Float32 = 0.9
comptime BETA2: Float32 = 0.999
comptime EPS: Float32 = 1e-8


@fieldwise_init
struct Cfg(ImplicitlyCopyable, Movable):
    var B: Int
    var I: Int
    var H: Int
    var O: Int
    var read_every: Int
    var warmup: Int
    var steps: Int
    var cpu_us: Int  # host busy-work after each step's launches
    var flush_each_step: Bool

    def total(self) -> Int:
        return self.warmup + self.steps

    def nelem(self, k: Int) -> Int:
        if k == X:
            return self.B * self.I
        if k == Y or k == Z2 or k == DZ2:
            return self.B * self.O
        if k == W1 or k == DW1 or k == M0 or k == V0:
            return self.I * self.H
        if k == B1 or k == DB1 or k == M0 + 1 or k == V0 + 1:
            return self.H
        if k == W2 or k == DW2 or k == M0 + 2 or k == V0 + 2:
            return self.H * self.O
        if k == B2 or k == DB2 or k == M0 + 3 or k == V0 + 3:
            return self.O
        if k == LOSS:
            return self.total()
        return self.B * self.H  # Z1, A1, DZ1


struct Result(Movable):
    var us_step: Float64
    var params: List[Float32]
    var losses: List[Float32]
    var commits: Int
    var idle_commits: Int

    def __init__(out self):
        self.us_step = 0.0
        self.params = List[Float32]()
        self.losses = List[Float32]()
        self.commits = 0
        self.idle_commits = 0


def init_host(c: Cfg) -> List[List[Float32]]:
    """Deterministic init (LCG, fixed seed) for every buffer slot."""
    var out = List[List[Float32]]()
    var s: UInt64 = 0x9E3779B97F4A7C15

    for k in range(NBUF):
        var n = c.nelem(k)
        var v = List[Float32](length=n, fill=0.0)
        var scale: Float32 = 0.0
        if k == X:
            scale = 1.0
        elif k == W1 or k == B1:
            scale = 1.0 / Float32(sqrt(Float64(c.I)))
        elif k == W2 or k == B2:
            scale = 1.0 / Float32(sqrt(Float64(c.H)))
        if scale != 0.0:
            for i in range(n):
                s = s * 6364136223846793005 + 1442695040888963407
                var u = Float32(Float64(s >> 40) / Float64(1 << 24))  # [0,1)
                v[i] = (2.0 * u - 1.0) * scale
        out.append(v^)
    # Y[b, o] = sin(sum_i X[b, i] * ((i + o) % 3 - 1))
    for b in range(c.B):
        for o in range(c.O):
            var acc: Float32 = 0.0
            for i in range(c.I):
                acc += out[X][b * c.I + i] * Float32((i + o) % 3 - 1)
            out[Y][b * c.O + o] = Float32(sqrt(Float64(acc * acc) + 1.0)) - 1.0
    return out^


def bias_corr(t: Int) -> Tuple[Float32, Float32]:
    var b1 = 1.0 / (1.0 - Float64(BETA1) ** Float64(t + 1))
    var b2 = 1.0 / (1.0 - Float64(BETA2) ** Float64(t + 1))
    return (Float32(b1), Float32(b2))


@always_inline
def pslot(j: Int) -> Int:
    """Adam's j-th parameter slot: W1, b1, W2, b2."""
    return W1 if j == 0 else (B1 if j == 1 else (W2 if j == 2 else B2))


@always_inline
def gslot(j: Int) -> Int:
    return DW1 if j == 0 else (DB1 if j == 1 else (DW2 if j == 2 else DB2))


@always_inline
def g(n: Int64) -> Int:
    return (Int(n) + TPB - 1) // TPB


def spin_us(us: Int):
    if us <= 0:
        return
    var t0 = perf_counter_ns()
    while Int(perf_counter_ns() - t0) < us * 1000:
        pass


# ------------------------------------------------------------------ MAX path


def max_step(ctx: DeviceContext, b: List[DeviceBuffer[DType.float32]], c: Cfg, t: Int) raises:
    var B = Int64(c.B)
    var I = Int64(c.I)
    var H = Int64(c.H)
    var O = Int64(c.O)
    ctx.enqueue_function[fc_fwd](b[Z1], b[X], b[W1], b[B1], B, I, H, grid_dim=g(B * H), block_dim=TPB)
    ctx.enqueue_function[relu_fwd](b[A1], b[Z1], B * H, grid_dim=g(B * H), block_dim=TPB)
    ctx.enqueue_function[fc_fwd](b[Z2], b[A1], b[W2], b[B2], B, H, O, grid_dim=g(B * O), block_dim=TPB)
    ctx.enqueue_function[mse_loss](b[LOSS], b[Z2], b[Y], B * O, Int64(t), grid_dim=1, block_dim=TPB)
    ctx.enqueue_function[mse_grad](b[DZ2], b[Z2], b[Y], B * O, Float32(2.0) / Float32(B * O), grid_dim=g(B * O), block_dim=TPB)
    ctx.enqueue_function[grad_w](b[DW2], b[A1], b[DZ2], B, H, O, grid_dim=g(H * O), block_dim=TPB)
    ctx.enqueue_function[grad_b](b[DB2], b[DZ2], B, O, grid_dim=g(O), block_dim=TPB)
    ctx.enqueue_function[grad_in_relu](b[DZ1], b[DZ2], b[W2], b[Z1], B, H, O, grid_dim=g(B * H), block_dim=TPB)
    ctx.enqueue_function[grad_w](b[DW1], b[X], b[DZ1], B, I, H, grid_dim=g(I * H), block_dim=TPB)
    ctx.enqueue_function[grad_b](b[DB1], b[DZ1], B, H, grid_dim=g(H), block_dim=TPB)
    var bc = bias_corr(t)
    for j in range(4):
        var n = Int64(c.nelem(pslot(j)))
        ctx.enqueue_function[adam](b[pslot(j)], b[gslot(j)], b[M0 + j], b[V0 + j], n, LR, BETA1, BETA2, EPS, bc[0], bc[1], grid_dim=g(n), block_dim=TPB)


def run_max(ctx: DeviceContext, c: Cfg, init: List[List[Float32]]) raises -> Result:
    var b = List[DeviceBuffer[DType.float32]]()
    for k in range(NBUF):
        var buf = ctx.enqueue_create_buffer[DType.float32](c.nelem(k))
        ctx.enqueue_copy(buf, init[k].unsafe_ptr())
        b.append(buf^)
    var hloss = ctx.enqueue_create_host_buffer[DType.float32](c.total())
    ctx.synchronize()
    var sink: Float32 = 0.0
    var t0 = 0
    for t in range(c.total()):
        if t == c.warmup:
            ctx.synchronize()
            t0 = Int(perf_counter_ns())
        max_step(ctx, b, c, t)
        spin_us(c.cpu_us)
        if t % c.read_every == c.read_every - 1:
            ctx.enqueue_copy(hloss, b[LOSS])
            ctx.synchronize()
            sink += hloss[t]
    ctx.synchronize()
    var r = Result()
    r.us_step = Float64(Int(perf_counter_ns()) - t0) / 1e3 / Float64(c.steps)
    for k in [W1, B1, W2, B2]:
        var h = List[Float32](length=c.nelem(k), fill=0.0)
        ctx.enqueue_copy(h.unsafe_ptr(), b[k])
        ctx.synchronize()
        r.params.extend(h^)
    ctx.enqueue_copy(hloss, b[LOSS])
    ctx.synchronize()
    for t in range(c.total()):
        r.losses.append(hloss[t])
    _ = sink
    return r^


# ------------------------------------------------------------------ our path


@always_inline
def d(b: List[Buffer], k: Int) -> Pointer[Float32, MutAnyOrigin]:
    return b[k].dev[DType.float32]()


def ours_step[L: Launcher](mut rt: L, b: List[Buffer], c: Cfg, t: Int) raises:
    var B = Int64(c.B)
    var I = Int64(c.I)
    var H = Int64(c.H)
    var O = Int64(c.O)
    rt.launch[fc_fwd](d(b, Z1), d(b, X), d(b, W1), d(b, B1), B, I, H, grid=g(B * H), block=TPB)
    rt.launch[relu_fwd](d(b, A1), d(b, Z1), B * H, grid=g(B * H), block=TPB)
    rt.launch[fc_fwd](d(b, Z2), d(b, A1), d(b, W2), d(b, B2), B, H, O, grid=g(B * O), block=TPB)
    rt.launch[mse_loss](d(b, LOSS), d(b, Z2), d(b, Y), B * O, Int64(t), grid=1, block=TPB)
    rt.launch[mse_grad](d(b, DZ2), d(b, Z2), d(b, Y), B * O, Float32(2.0) / Float32(B * O), grid=g(B * O), block=TPB)
    rt.launch[grad_w](d(b, DW2), d(b, A1), d(b, DZ2), B, H, O, grid=g(H * O), block=TPB)
    rt.launch[grad_b](d(b, DB2), d(b, DZ2), B, O, grid=g(O), block=TPB)
    rt.launch[grad_in_relu](d(b, DZ1), d(b, DZ2), d(b, W2), d(b, Z1), B, H, O, grid=g(B * H), block=TPB)
    rt.launch[grad_w](d(b, DW1), d(b, X), d(b, DZ1), B, I, H, grid=g(I * H), block=TPB)
    rt.launch[grad_b](d(b, DB1), d(b, DZ1), B, H, grid=g(H), block=TPB)
    var bc = bias_corr(t)
    for j in range(4):
        var n = Int64(c.nelem(pslot(j)))
        rt.launch[adam](d(b, pslot(j)), d(b, gslot(j)), d(b, M0 + j), d(b, V0 + j), n, LR, BETA1, BETA2, EPS, bc[0], bc[1], grid=g(n), block=TPB)


def run_ours(
    c: Cfg,
    init: List[List[Float32]],
    policy: Int,
    batch_n: Int,
    idle_age_us: Int = 0,
    order: Int = ORDER_FENCE,
    metal4: Bool = False,
) raises -> Result:
    if metal4:
        return drive(Metal4RT(policy=policy, batch_n=batch_n, idle_age_us=idle_age_us), c, init)
    return drive(
        MetalRT(policy=policy, batch_n=batch_n, idle_age_us=idle_age_us, order=order),
        c,
        init,
    )


def drive[L: Launcher](var rt: L, c: Cfg, init: List[List[Float32]]) raises -> Result:
    var b = List[Buffer]()
    for k in range(NBUF):
        var buf = rt.alloc[DType.float32](c.nelem(k))
        var hp = buf.host_ptr[DType.float32]()
        for i in range(c.nelem(k)):
            hp[unsafe_offset=i] = init[k][i]
        b.append(buf)
    var sink: Float32 = 0.0
    var t0 = 0
    var c0 = 0
    var i0 = 0
    var lp = b[LOSS].host_ptr[DType.float32]()
    for t in range(c.total()):
        if t == c.warmup:
            rt.sync()
            t0 = Int(perf_counter_ns())
            c0 = rt.commits()
            i0 = rt.idle_commits()
        ours_step(rt, b, c, t)
        if c.flush_each_step:
            rt.flush()
        spin_us(c.cpu_us)
        if t % c.read_every == c.read_every - 1:
            rt.sync()
            sink += lp[unsafe_offset=t]
    rt.sync()
    var r = Result()
    r.us_step = Float64(Int(perf_counter_ns()) - t0) / 1e3 / Float64(c.steps)
    r.commits = rt.commits() - c0
    r.idle_commits = rt.idle_commits() - i0
    for k in [W1, B1, W2, B2]:
        var hp = b[k].host_ptr[DType.float32]()
        for i in range(c.nelem(k)):
            r.params.append(hp[unsafe_offset=i])
    for t in range(c.total()):
        r.losses.append(lp[unsafe_offset=t])
    _ = sink
    return r^


# ------------------------------------------------------------------ gate


def same_bits(a: List[Float32], b: List[Float32]) -> Int:
    """Index of the first bitwise difference, or -1."""
    if len(a) != len(b):
        return 0
    for i in range(len(a)):
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            return i
    return -1


def check(ref_: Result, got: Result, what: String) -> Bool:
    var ip = same_bits(ref_.params, got.params)
    var il = same_bits(ref_.losses, got.losses)
    if ip >= 0 or il >= 0:
        print("  !! GATE FAIL", what, "param idx", ip, "loss idx", il)
        if il >= 0 and il < len(got.losses):
            print("     loss ref", ref_.losses[il], "got", got.losses[il])
        return False
    return True


def fmt(x: Float64) -> String:
    var r = Int(x * 10.0 + 0.5)
    return String(r // 10) + "." + String(r % 10)


def median(var v: List[Float64]) -> Float64:
    for i in range(len(v)):
        for j in range(i + 1, len(v)):
            if v[j] < v[i]:
                var t = v[i]
                v[i] = v[j]
                v[j] = t
    return v[len(v) // 2]


@fieldwise_init
struct Pol(ImplicitlyCopyable, Movable):
    var name: String
    var policy: Int
    var batch_n: Int
    var idle_age_us: Int
    var flush_each_step: Bool
    var metal4: Bool


def main() raises:
    var quick = False
    for a in argv():
        if a == "--quick":
            quick = True
    var steps = 100 if quick else 500
    var warmup = 20 if quick else 50
    var reps = 1 if quick else 5
    var ctx = DeviceContext()
    var fails = 0
    var gates = 0

    var pols = List[Pol]()
    pols.append(Pol("each", COMMIT_EACH, 1, 0, False, False))
    pols.append(Pol("every-16", COMMIT_EVERY_N, 16, 0, False, False))
    pols.append(Pol("every-64", COMMIT_EVERY_N, 64, 0, False, False))
    pols.append(Pol("adaptive-16", COMMIT_ADAPTIVE, 16, 0, False, False))
    pols.append(Pol("adaptive-16, age 20us", COMMIT_ADAPTIVE, 16, 20, False, False))
    pols.append(Pol("Metal 4 every-16", COMMIT_EVERY_N, 16, 0, False, True))

    print("MLP step: I=16, O=4,", STEP_LAUNCHES, "launches/step,", steps,
          "timed steps after", warmup, "warm-up; us/step, median of", reps, "repeats")
    var hs = [64, 256]
    var bs = [1, 64, 256, 1024]
    for re in [20, 1]:
        print("\n### loss read back every", re, "step(s)\n")
        var hdr = String("| H | B | loss step 0 -> last | MAX")
        var sep = String("|---|---|---|---")
        for p in range(len(pols)):
            hdr += " | " + pols[p].name
            sep += "|---"
        print(hdr + " | best ours vs MAX |")
        print(sep + "|---|")
        var stats = String()
        for h in hs:
            for bb in bs:
                var c = Cfg(bb, 16, h, 4, re, warmup, steps, 0, False)
                var init = init_host(c)
                var tm = List[Float64]()
                var tp = List[List[Float64]]()
                for _ in range(len(pols)):
                    tp.append(List[Float64]())
                var ref_ = run_max(ctx, c, init)
                for rep in range(reps):
                    var rm = run_max(ctx, c, init)
                    gates += 1
                    if not check(ref_, rm, String("MAX vs MAX H=", h, " B=", bb)):
                        fails += 1
                    tm.append(rm.us_step)
                    for p in range(len(pols)):
                        var r = run_ours(c, init, pols[p].policy, pols[p].batch_n, pols[p].idle_age_us, metal4=pols[p].metal4)
                        gates += 1
                        if not check(ref_, r, String("H=", h, " B=", bb, " read_every=", re, " ", pols[p].name)):
                            fails += 1
                        tp[p].append(r.us_step)
                        if rep == 0 and pols[p].policy == COMMIT_ADAPTIVE:
                            stats += String("| ", h, " | ", bb, " | ", pols[p].name, " | ",
                                fmt(Float64(r.commits) / Float64(steps)), " | ",
                                fmt(100.0 * Float64(r.idle_commits) / Float64(max(r.commits, 1))), "% |\n")
                var mmax = median(tm^)
                var row = String("| ") + String(h) + " | " + String(bb) + " | " + String(ref_.losses[0]) + " -> " + String(ref_.losses[len(ref_.losses) - 1]) + " | " + fmt(mmax)
                var best = Float64(1e30)
                for p in range(len(pols)):
                    var m = median(tp[p].copy())
                    row += " | " + fmt(m)
                    best = min(best, m)
                print(row + " | " + fmt(mmax / best) + "x |")
        print("\nadaptive policies, read back every", re, ": command buffers per step, and the share committed because the GPU was idle")
        print("| H | B | policy | cb/step | idle commits |\n|---|---|---|---|---|")
        print(stats)

    # ---- the tail problem: host CPU work after each step's launches -------
    print("\n### host CPU work between a step's launches and its loss read-back, H=64 B=64, loss every step; us/step, median of", reps, "\n")
    var tpols = List[Pol]()
    tpols.append(Pol("each", COMMIT_EACH, 1, 0, False, False))
    tpols.append(Pol("every-16", COMMIT_EVERY_N, 16, 0, False, False))
    tpols.append(Pol("adaptive-16", COMMIT_ADAPTIVE, 16, 0, False, False))
    tpols.append(Pol("adaptive-16, age 20us", COMMIT_ADAPTIVE, 16, 20, False, False))
    tpols.append(Pol("every-16 + flush()/step", COMMIT_EVERY_N, 16, 0, True, False))
    var hdr = String("| cpu us/step | MAX")
    var sep = String("|---|---")
    for p in range(len(tpols)):
        hdr += " | " + tpols[p].name
        sep += "|---"
    print(hdr + " |\n" + sep + "|")
    for cpu in [0, 100, 300, 600]:
        var c = Cfg(64, 16, 64, 4, 1, warmup, steps, cpu, False)
        var init = init_host(c)
        var ref_ = run_max(ctx, c, init)
        var tm = List[Float64]()
        var tp = List[List[Float64]]()
        for _ in range(len(tpols)):
            tp.append(List[Float64]())
        for _ in range(reps):
            tm.append(run_max(ctx, c, init).us_step)
            for p in range(len(tpols)):
                var cc = c
                cc.flush_each_step = tpols[p].flush_each_step
                var r = run_ours(cc, init, tpols[p].policy, tpols[p].batch_n, tpols[p].idle_age_us)
                gates += 1
                if not check(ref_, r, String("cpu=", cpu, " ", tpols[p].name)):
                    fails += 1
                tp[p].append(r.us_step)
        var row = String("| ") + String(cpu) + " | " + fmt(median(tm^))
        for p in range(len(tpols)):
            row += " | " + fmt(median(tp[p].copy()))
        print(row + " |")

    # ---- negative control: the gate must see a missing ordering ----------
    var cn = Cfg(64, 16, 64, 4, 20, warmup, steps, 0, False)
    var initn = init_host(cn)
    var refn = run_max(ctx, cn, initn)
    var rn = run_ours(cn, initn, COMMIT_EACH, 1, 0, ORDER_NONE)
    print("\nnegative control (ORDER_NONE, commit each, H=64 B=64): gate",
          "DETECTS the race" if same_bits(refn.params, rn.params) >= 0 or same_bits(refn.losses, rn.losses) >= 0
          else "did NOT see a difference this run")

    print("\nGATE:", gates - fails, "/", gates, "runs bit-identical to MAX (params + every step's loss)")
    if fails != 0:
        print("GATE FAILED")
        exit(1)
    print("GATE OK")
