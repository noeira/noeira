"""The ANE training step with Adam on the GPU (`AneAdamGpu`) against the same
step with nn's Adam on the host (`AneGrad.run` + `Adam.step["cpu"]`).

Network: test_ane_step's sine fit, 1 -> 64 tanh -> 64 tanh -> 1, batch 256.
Its w3 [64, 1] has padded rows on the ANE (one fp16 per 64-byte row), so a
weight written at the wrong stride shows.

1. FIVE STEPS from the same weights and batches, both arms at the same loss
   scale. After every step:
   - the fp32 masters agree within BOUND_ULPS (8) ulps of max(|p|, lr), per
     element. The two arms see the same fp16 weights and so (bit for bit)
     the same gradients; only Adam's own fp32 arithmetic can differ: the
     GPU's division and sqrt are not correctly rounded and it flushes fp32
     denormals, and fma contraction may differ. Each step's update (size
     <= ~lr) differs by a few ulps, and m, v carry the difference into the
     next step: measured 0, 2, 5, 6.5, 6.75 ulps after steps 1..5;
   - the fp16 weights in the ANE input surfaces are BIT-equal to the CPU's
     fp16 cast of the downloaded masters, read at the padded row stride.
2. SKIP: the largest fp16 seed (65504) overflows the gradients. Both arms skip;
   the GPU's masters, m and v stay bit-identical and its step count does not
   move; the next good step agrees with the host arm as in (1).
3. TRAINING: 2000 steps of Adam 3e-3 with the dynamic loss scaler (halve
   and skip on NaN, double after 500 good steps, seed cap 32768), GPU Adam
   vs host Adam, same init and batches. The two runs diverge from ulp-level
   differences (one fp16 rounding flip moves a whole gradient), so only the
   outcome is comparable, not the curve. The outcome is the MEDIAN loss of
   the last 100 steps (the late curve has spikes: one spike moves a mean
   several-fold). One run says little: from inits 1..11 fp32 ulps away the
   host arm alone lands anywhere in 2e-5 .. 2.5e-4. So each arm runs an
   ENSEMBLE of 12 (the baseline and those 11 inits), and the bound is: the
   GPU ensemble's median within a factor 2 of the host ensemble's. (With 5
   runs per arm the two medians differed 3.4x by chance; with 12, 1.3x.)
4. MUTANTS must FAIL (1): (a) fp16 weights written at the dense row stride,
   (b) no bias correction.

    pixi run -e apple mojo run -I . tests/experimental/ane/test_ane_gpu_adam.mojo
"""

from std.math import pi, sin
from std.memory import bitcast
from std.random import seed, random_float64
from std.sys import exit
from std.time import perf_counter_ns

from noeira.nn.constants import DT
from noeira.nn.core.initializer import Xavier
from noeira.nn.optimizer.adam import Adam
from noeira.experimental.functional import Tape, FTensor
import noeira.experimental.functional.F as F
from noeira.experimental.ane.runtime import ANE
from noeira.experimental.ane.step import AneGrad
from noeira.experimental.ane.gpu_adam import AneAdamGpu

comptime B = 256
comptime H = 64
comptime LR: Float32 = 3e-3
comptime BOUND_ULPS: Float64 = 8.0


def params(tape: Tape) raises -> List[FTensor]:
    """test_ane_step's `Net.params` for the sine fit (1-64-64-1)."""
    seed(7)
    var p = List[FTensor]()
    p.append(tape.param[Xavier](1, H, name="w1"))
    p.append(tape.zeros(1, H, name="b1"))
    p.append(tape.param[Xavier](H, H, name="w2"))
    p.append(tape.zeros(1, H, name="b2"))
    p.append(tape.param[Xavier](H, 1, name="w3"))
    p.append(tape.zeros(1, 1, name="b3"))
    for k in [1, 3, 5]:
        for i in range(p[k].size()):
            p[k].set_value(i, Scalar[DT](0.1 * random_float64() - 0.05))
    return p^


def loss_of(p: List[FTensor], x: FTensor, y: FTensor) raises -> FTensor:
    var h = F.linear(x, p[0], p[1], "tanh")
    h = F.linear(h, p[2], p[3], "tanh")
    return F.mse(F.linear(h, p[4], p[5]), y)


def batch(step: Int) -> Tuple[List[Scalar[DT]], List[Scalar[DT]]]:
    seed(1000 + step)
    var x = List[Scalar[DT]]()
    var y = List[Scalar[DT]]()
    for _ in range(B):
        var v = (random_float64() * 2.0 - 1.0) * pi
        x.append(Scalar[DT](v))
        y.append(Scalar[DT](sin(v)))
    return (x^, y^)


def compile(a: ANE, mut tape: Tape, p: List[FTensor]) raises -> AneGrad:
    var bt = batch(0)
    var xb = tape.input(B, 1, bt[0].copy())
    var yb = tape.input(B, 1, bt[1].copy())
    var g = AneGrad(a, tape, loss_of(p, xb, yb), [xb, yb])
    tape.reset()
    return g^


def ulp(x: Float64) -> Float64:
    """The fp32 ulp at |x|."""
    var b = bitcast[DType.uint32](Float32(abs(x)))
    var e = Int((b >> 23) & 0xFF)
    return 2.0 ** Float64(max(e, 1) - 150)


struct Cmp(Movable):
    var worst_ulps: Float64
    var surf_bad: Int

    def __init__(out self):
        self.worst_ulps = 0
        self.surf_bad = 0


def compare(a: ANE, mut ga: AneAdamGpu, mut tg: Tape, th: Tape) raises -> Cmp:
    """Masters (GPU, downloaded into `tg`) vs the host arm's values (`th`), in
    ulps of max(|p|, lr); and the input surfaces vs fp16(masters)."""
    ga.download(tg)
    var r = Cmp()
    for q in range(len(ga.params)):
        var id = ga.params[q].id
        var rows = ga.params[q].rows
        var cols = ga.params[q].cols
        var pg = tg.data[].val_ptr(id)
        var ph = th.data[].val_ptr(id)
        var row = ga.g.prog.in_layout[ga.params[q].input].row
        var surf = a.base(ga.g.prog.ins[ga.params[q].input])
        for i in range(rows):
            for j in range(cols):
                var x = Float64(pg[unsafe_offset=i * cols + j])
                var y = Float64(ph[unsafe_offset=i * cols + j])
                var u = ulp(max(max(abs(x), abs(y)), Float64(LR)))
                r.worst_ulps = max(r.worst_ulps, abs(x - y) / u)
                var want = bitcast[DType.uint16](pg[unsafe_offset=i * cols + j].cast[DType.float16]())
                if bitcast[DType.uint16](surf[unsafe_offset=i * row + j]) != want:
                    r.surf_bad += 1
    return r^


def gate1(a: ANE, steps: Int, mutant_dense: Bool, mutant_no_bc: Bool, do_skip: Bool) raises -> Bool:
    var th = Tape()
    var ph = params(th)
    var gh = compile(a, th, ph)
    var opt = Adam(lr=Scalar[DT](LR))
    var tg = Tape()
    var pg = params(tg)
    var gg = compile(a, tg, pg)
    var ga = AneAdamGpu(a, gg^, tg, lr=LR, mutant_dense=mutant_dense, mutant_no_bc=mutant_no_bc)
    # the largest power-of-two scale (from start_scale down) whose gradients
    # are finite on every batch the gate uses (a NaN would make both arms
    # skip and compare nothing)
    var scale = gh.start_scale()
    var finite = False
    while not finite:
        finite = True
        for s in range(steps + 2):
            var bt = batch(s)
            gh.set_data(a, 0, Pointer(to=bt[0][0]))
            gh.set_data(a, 1, Pointer(to=bt[1][0]))
            var l = gh.run(a, th, scale)
            if l != l:
                finite = False
        if not finite:
            scale *= 0.5
    print("     loss scale (seed)", scale)
    var ok = True
    var worst: Float64 = 0
    var surf_bad = 0
    for s in range(steps):
        var bt = batch(s)
        gh.set_data(a, 0, Pointer(to=bt[0][0]))
        gh.set_data(a, 1, Pointer(to=bt[1][0]))
        var lh = gh.run(a, th, scale)
        var t = th
        opt.step["cpu"](t, None)
        ga.set_data(a, 0, Pointer(to=bt[0][0]))
        ga.set_data(a, 1, Pointer(to=bt[1][0]))
        var lg = ga.step(a, scale)
        var c = compare(a, ga, tg, th)
        worst = max(worst, c.worst_ulps)
        surf_bad += c.surf_bad
        print("     step", s, "| loss host", lh, "gpu", lg, "| masters: worst", c.worst_ulps,
              "ulps | fp16 surface elements != fp16(master):", c.surf_bad)
        if lh != lh or lg != lg or c.worst_ulps > BOUND_ULPS or c.surf_bad != 0:
            ok = False
    if do_skip:
        # snapshot, then a NaN batch: both arms must skip
        var snap = List[List[Float32]]()
        for q in range(len(ga.params)):
            ref p = ga.params[q]
            for buf in [p.w, p.m, p.v]:
                var v = List[Float32]()
                for i in range(p.rows * p.cols):
                    v.append(buf.host_ptr[DType.float32]()[unsafe_offset=i])
                snap.append(v^)
        var t_before = ga.t
        # fp16's largest seed: the summed output-layer gradient (256 rows of
        # ~seed each) overflows to Inf. (A NaN in the DATA does not do it:
        # the ANE's tanh returns a finite value for a NaN input.)
        var bt = batch(steps)
        var big = Float32(65504.0)
        gh.set_data(a, 0, Pointer(to=bt[0][0]))
        gh.set_data(a, 1, Pointer(to=bt[1][0]))
        var lh = gh.run(a, th, big)
        ga.set_data(a, 0, Pointer(to=bt[0][0]))
        ga.set_data(a, 1, Pointer(to=bt[1][0]))
        var lg = ga.step(a, big)
        var same = 0
        var total = 0
        var k = 0
        for q in range(len(ga.params)):
            ref p = ga.params[q]
            for buf in [p.w, p.m, p.v]:
                for i in range(p.rows * p.cols):
                    total += 1
                    if bitcast[DType.uint32](buf.host_ptr[DType.float32]()[unsafe_offset=i]) == bitcast[DType.uint32](snap[k][i]):
                        same += 1
                k += 1
        var skip_ok = lh != lh and lg != lg and same == total and ga.t == t_before
        print("   2. skip: host loss", lh, "| gpu loss", lg, "| masters/m/v bit-identical:", same, "/", total,
              "| step count", t_before, "->", ga.t, "OK" if skip_ok else "FAIL")
        ok = ok and skip_ok
        # the next good step, host arm skipped too (no opt.step)
        bt = batch(steps + 1)
        gh.set_data(a, 0, Pointer(to=bt[0][0]))
        gh.set_data(a, 1, Pointer(to=bt[1][0]))
        lh = gh.run(a, th, scale)
        var t = th
        opt.step["cpu"](t, None)
        ga.set_data(a, 0, Pointer(to=bt[0][0]))
        ga.set_data(a, 1, Pointer(to=bt[1][0]))
        lg = ga.step(a, scale)
        var c = compare(a, ga, tg, th)
        var next_ok = c.worst_ulps <= BOUND_ULPS and c.surf_bad == 0 and lg == lg
        print("      next good step | loss host", lh, "gpu", lg, "| masters: worst", c.worst_ulps,
              "ulps | surface mismatches", c.surf_bad, "OK" if next_ok else "FAIL")
        ok = ok and next_ok
    print("     worst over the steps:", worst, "ulps (bound", BOUND_ULPS, ") | surface mismatches", surf_bad)
    ga.free(a)
    gh.free(a)
    return ok


def train(a: ANE, steps: Int, gpu: Bool, mut curve: List[Float64], perturb: Int = 0) raises -> Float64:
    """test_ane_step's ANE training loop; Adam on the host or on the GPU.
    `perturb` = k > 0: every initial weight moved by k fp32 ulps (one member
    of the ensemble that measures the outcome's own spread). Returns the mean
    us per step after the first 20."""
    var tape = Tape()
    var p = params(tape)
    if perturb > 0:
        for k in range(len(p)):
            var vals = p[k].values()
            for i in range(p[k].size()):
                p[k].set_value(i, bitcast[DType.float32](bitcast[DType.uint32](vals[i]) + UInt32(perturb)))
    var g = compile(a, tape, p)
    var scale = g.start_scale()
    var opt = Adam(lr=Scalar[DT](LR))
    var ga = Optional[AneAdamGpu]()
    var gh = Optional[AneGrad]()
    if gpu:
        ga = AneAdamGpu(a, g^, tape, lr=LR)
    else:
        gh = g^
    var good = 0
    var skipped = 0
    var t_sum = Float64(0)
    for s in range(steps):
        var bt = batch(s)
        var t0 = perf_counter_ns()
        var l: Float64
        if gpu:
            ga.value().set_data(a, 0, Pointer(to=bt[0][0]))
            ga.value().set_data(a, 1, Pointer(to=bt[1][0]))
            l = ga.value().step(a, scale)
        else:
            gh.value().set_data(a, 0, Pointer(to=bt[0][0]))
            gh.value().set_data(a, 1, Pointer(to=bt[1][0]))
            l = gh.value().run(a, tape, scale)
        if l != l:
            scale *= 0.5
            skipped += 1
            good = 0
            continue
        good += 1
        if good == 500 and scale < 32768.0:
            scale *= 2.0
            good = 0
        if not gpu:
            var t = tape
            opt.step["cpu"](t, None)
        if s >= 20:
            t_sum += Float64(perf_counter_ns() - t0) / 1e3
        curve.append(l)
    print("     ", "GPU" if gpu else "host", "Adam: loss scale at the end", scale, ", skipped steps", skipped)
    if gpu:
        print("      GPU arm: mean ANE eval", Float64(ga.value().t_eval) / Float64(ga.value().steps) / 1e3,
              "us, GPU command buffer incl. sync", Float64(ga.value().t_gpu) / Float64(ga.value().steps) / 1e3, "us")
        ga.value().free(a)
    else:
        gh.value().free(a)
    return t_sum / Float64(steps - 20)


def tail_median(c: List[Float64], n: Int) -> Float64:
    var v = List[Float64]()
    for i in range(len(c) - n, len(c)):
        v.append(c[i])
    sort(v)
    return v[n // 2]


def tail_mean(c: List[Float64], n: Int) -> Float64:
    var s = Float64(0)
    for i in range(len(c) - n, len(c)):
        s += c[i]
    return s / Float64(n)


def main() raises:
    var a = ANE()
    var ok = True
    print("1. five steps, GPU Adam vs host Adam (sine fit 1-64-64-1, B=256)")
    var g1 = gate1(a, 5, False, False, True)
    print("   ", "OK" if g1 else "FAIL")
    ok = ok and g1
    print("4a. MUTANT: fp16 weights at the dense row stride (must FAIL gate 1)")
    var m1 = gate1(a, 5, True, False, False)
    print("   ", "caught (OK)" if not m1 else "NOT CAUGHT (FAIL)")
    ok = ok and not m1
    print("4b. MUTANT: no bias correction (must FAIL gate 1)")
    var m2 = gate1(a, 5, False, True, False)
    print("   ", "caught (OK)" if not m2 else "NOT CAUGHT (FAIL)")
    ok = ok and not m2

    comptime STEPS = 2000
    comptime ENSEMBLE = 11
    print("3. fit sin(x), 2000 steps, Adam 3e-3, dynamic loss scale: host Adam vs GPU Adam")
    var c_h = List[Float64]()
    var c_g = List[Float64]()
    var us_h = train(a, STEPS, False, c_h)
    var us_g = train(a, STEPS, True, c_g)
    var line = String("     step:")
    for s in [0, 100, 500, 1000, 1500, len(c_g) - 1]:
        if s < len(c_h) and s < len(c_g):
            line += String(" ", s, ": ", c_h[s], " / ", c_g[s], ";")
    print(line, "(host / GPU)")
    # the outcome is chaotic in the init's last bit: compare ENSEMBLES, the
    # baseline plus inits 1..ENSEMBLE ulps away, one per arm
    var fh = List[Float64]()
    var fg = List[Float64]()
    fh.append(tail_median(c_h, 100))
    fg.append(tail_median(c_g, 100))
    for k in range(1, ENSEMBLE + 1):
        var c1 = List[Float64]()
        _ = train(a, STEPS, False, c1, perturb=k)
        fh.append(tail_median(c1, 100))
        var c2 = List[Float64]()
        _ = train(a, STEPS, True, c2, perturb=k)
        fg.append(tail_median(c2, 100))
    print("     last 100 steps of the baseline run (mean | median): host Adam", tail_mean(c_h, 100), "|", fh[0],
          "; GPU Adam", tail_mean(c_g, 100), "|", fg[0], "; CPU tape (test_ane_step) mean 1.7e-4")
    var lh = String()
    var lg = String()
    for k in range(len(fh)):
        lh += String(" ", fh[k])
        lg += String(" ", fg[k])
    print("     medians of the last 100, inits +0..+", ENSEMBLE, "ulps: host", lh)
    print("                                                     GPU ", lg)
    var mh = fh.copy()
    var mg = fg.copy()
    sort(mh)
    sort(mg)
    var eh = mh[len(mh) // 2]
    var eg = mg[len(mg) // 2]
    var comparable = eg >= eh / 2.0 and eg <= eh * 2.0
    print("     ensemble median: host", eh, "GPU", eg, "ratio", eg / eh, "(bound: within a factor 2)",
          "OK" if comparable else "FAIL")
    print("     us per step (not a benchmark): ANE + host Adam", us_h, "| ANE + GPU Adam", us_g)
    ok = ok and comparable
    if not ok:
        print("FAILED")
        exit(1)
    print("ALL OK")
