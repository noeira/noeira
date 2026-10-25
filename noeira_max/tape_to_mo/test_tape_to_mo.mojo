"""Spike gate: the functional tape compiled by MAX's graph compiler, from Mojo
with no Python. One eager forward is recorded, `emit_train_step` turns it
into an MO graph (forward + backward + Adam in place), MAX's C API compiles
it, and K steps run on host buffers. The reference is the eager tape with
nn's Adam from the same weights on the same batches.

1. MLP (F.linear relu, tanh, linear) + F.mse: losses and final parameters.
2. A graph of every other supported op (+ - * / with broadcasting, scalar
   ops, matmul, transpose, sigmoid/exp/log/sqrt, sum, sum_rows, mean).
3. A mutant (tanh's backward wrong) must FAIL gate 1.
4. Time per step: eager tape vs the compiled step; compile time.

Build and run: `noeira_max/tape_to_mo/run.sh` (CPU; `TAPE_MO_ENV=apple`).
"""

from std.math import sqrt as fsqrt
from std.random import seed, random_float64
from std.sys import argv
from std.time import perf_counter_ns

from maxrt import HostBuffer, Runtime, TensorMap, Model, Tensor
from max.gpu.host import DeviceContext
from noeira.nn.constants import DT
from noeira.nn.core.initializer import Xavier
from noeira.nn.optimizer.adam import Adam
from noeira.experimental.functional import Tape, FTensor
import noeira.experimental.functional.F as F

from tape_to_mo.emit import emit_train_step, Step


comptime LR: Float32 = 3e-3


struct Net(Copyable):
    var b: Int
    var din: Int
    var h: Int
    var dout: Int
    var kind: Int
    """0 = MLP + MSE; 1 = the every-op graph."""

    def __init__(out self, b: Int, din: Int, h: Int, dout: Int, kind: Int):
        self.b = b
        self.din = din
        self.h = h
        self.dout = dout
        self.kind = kind

    def params(self, tape: Tape) raises -> List[FTensor]:
        seed(7)
        var p = List[FTensor]()
        p.append(tape.param[Xavier](self.din, self.h, name="w1"))
        p.append(tape.zeros(1, self.h, name="b1"))
        p.append(tape.param[Xavier](self.h, self.h, name="w2"))
        p.append(tape.zeros(1, self.h, name="b2"))
        p.append(tape.param[Xavier](self.h, self.dout, name="w3"))
        p.append(tape.zeros(1, self.dout, name="b3"))
        return p^

    def loss(self, p: List[FTensor], x: FTensor, y: FTensor) raises -> FTensor:
        if self.kind == 0:
            var h = F.linear(x, p[0], p[1], "relu")
            h = F.linear(h, p[2], p[3], "tanh")
            return F.mse(F.linear(h, p[4], p[5]), y)
        # every other op, rank 2
        var z = x @ p[0] + p[1]                                  # matmul, add (broadcast)
        var a = z.sigmoid() * Scalar[DT](2.0) + Scalar[DT](0.5)  # scale, shift
        var c = (a.exp() + Scalar[DT](1.0)).log()                # exp, log
        var d = (c * c + Scalar[DT](1.0)).sqrt() / (a + Scalar[DT](1.0))  # mul, sqrt, div
        var e = (d @ p[2] - p[3]).tanh()                         # sub (broadcast)
        var f = (Scalar[DT](1.0) - e) / Scalar[DT](4.0)          # rsubc, divc
        var pred = (f @ p[4]) * p[5]                             # mul (broadcast)
        var r = (pred - y).relu()
        var t = ((x.T() @ x) / Scalar[DT](100.0)).sum_rows()      # transpose, sum_rows
        return r.mean() + (t * t).sum() * Scalar[DT](1e-4) + (pred * pred).mean()


def batch(net: Net, step: Int) -> Tuple[List[Scalar[DT]], List[Scalar[DT]]]:
    seed(1000 + step)
    var x = List[Scalar[DT]]()
    var y = List[Scalar[DT]]()
    for _ in range(net.b * net.din):
        x.append(Scalar[DT](random_float64() * 2.0 - 1.0))
    for _ in range(net.b * net.dout):
        y.append(Scalar[DT](random_float64() * 2.0 - 1.0))
    return (x^, y^)


def flat(ps: List[FTensor]) raises -> List[Scalar[DT]]:
    var out = List[Scalar[DT]]()
    for i in range(len(ps)):
        out.extend(ps[i].values())
    return out^


struct Best(Movable):
    """Mean time of 40-step chunks, keeping the fastest chunk: another
    process's burst then costs a chunk, not the measurement."""
    var acc: Float64
    var n: Int
    var best: Float64

    def __init__(out self):
        self.acc = 0
        self.n = 0
        self.best = 1e30

    def add(mut self, us: Float64):
        self.acc += us
        self.n += 1
        if self.n == 40:
            self.best = min(self.best, self.acc / 40.0)
            self.acc = 0
            self.n = 0


struct Run(Movable):
    var losses: List[Float64]
    var params: List[Scalar[DT]]
    var step_us: Float64
    var compile_ms: Float64
    var relend_us: Float64

    def __init__(out self):
        self.losses = List[Float64]()
        self.params = List[Scalar[DT]]()
        self.step_us = 0
        self.compile_ms = 0
        self.relend_us = 0


def eager(net: Net, steps: Int, timed: Int, fixed: Bool = False) raises -> Run:
    """CPU tape; `fixed`: batch 0 at every step (what the GPU runs use)."""
    var r = Run()
    var tape = Tape()
    var p = net.params(tape)
    var opt = Adam(lr=Scalar[DT](LR))
    var best = Best()
    for s in range(steps + timed):
        var bt = batch(net, 0 if fixed else s % steps)
        var t0 = perf_counter_ns()
        tape.reset()
        tape.zero_grad()
        var loss = net.loss(p, tape.input(net.b, net.din, bt[0].copy()),
                            tape.input(net.b, net.dout, bt[1].copy()))
        loss.backward()
        var t = tape
        opt.step["cpu"](t, None)
        var l = Float64(loss.item())
        if s < steps:
            r.losses.append(l)
            if s == steps - 1:
                r.params = flat(p)
        else:
            best.add(Float64(perf_counter_ns() - t0) / 1e3)
    r.step_us = best.best
    return r^


def eager_gpu(ctx: DeviceContext, net: Net, timed: Int, stage: Int = 2) raises -> Float64:
    """Mean step time of the eager tape on the GPU: batch 0 on the device
    (`input_device`), adopted Adam, the loss read back every step.
    `stage`: 0 = forward only, 1 = + backward, 2 = + Adam."""
    var tape = Tape(ctx)
    var p = net.params(tape)
    var opt = Adam(lr=Scalar[DT](LR))
    opt.adopt["gpu"](tape, ctx)
    var bt = batch(net, 0)
    var xd = ctx.enqueue_create_buffer[DT](len(bt[0]))
    var yd = ctx.enqueue_create_buffer[DT](len(bt[1]))
    with xd.map_to_host() as hx, yd.map_to_host() as hy:
        for i in range(len(bt[0])):
            hx[i] = bt[0][i]
        for i in range(len(bt[1])):
            hy[i] = bt[1][i]
    var best = Best()
    for s in range(20 + timed):
        var t0 = perf_counter_ns()
        tape.reset()
        if stage >= 1:
            tape.zero_grad()
        var loss = net.loss(p, tape.input_device(net.b, net.din, xd), tape.input_device(net.b, net.dout, yd))
        if stage >= 1:
            loss.backward()
        if stage >= 2:
            var t = tape
            opt.step["gpu"](t, ctx)
        _ = loss.item()
        if s >= 20:
            best.add(Float64(perf_counter_ns() - t0) / 1e3)
    _ = xd^
    _ = yd^
    return best.best


def tape_gpu(ctx: DeviceContext, net: Net, steps: Int, timed: Int, capture: Bool,
             read_loss: Bool = True) raises -> Run:
    """The eager tape on the GPU as `eager_gpu` runs it, gated like
    `compiled_gpu`. `capture`: step 0 runs eagerly (allocations, cuBLASLt
    plans), step 1 is captured into a CUDA graph and every later step
    replays it (NVIDIA only: raises where nothing is captured)."""
    var r = Run()
    var tape = Tape(ctx)
    var p = net.params(tape)
    var opt = Adam(lr=Scalar[DT](LR))
    opt.adopt["gpu"](tape, ctx)
    var bt = batch(net, 0)
    var xd = ctx.enqueue_create_buffer[DT](len(bt[0]))
    var yd = ctx.enqueue_create_buffer[DT](len(bt[1]))
    with xd.map_to_host() as hx, yd.map_to_host() as hy:
        for i in range(len(bt[0])):
            hx[i] = bt[0][i]
        for i in range(len(bt[1])):
            hy[i] = bt[1][i]

    def one(mut opt: Adam) raises {mut tape, p, net, xd, yd, ctx} -> FTensor:
        tape.reset()
        tape.zero_grad()
        var loss = net.loss(p, tape.input_device(net.b, net.din, xd), tape.input_device(net.b, net.dout, yd))
        loss.backward()
        opt.step["gpu"](tape, ctx)
        return loss

    var loss = one(opt)
    if steps > 0:
        r.losses.append(Float64(loss.item()))
    if capture:
        tape.capture_begin()
        loss = one(opt)
        tape.capture_end()
        if not tape.captured():
            raise Error("tape_gpu: nothing captured (capture is NVIDIA-only)")
    else:
        loss = one(opt)
    if steps > 1:
        r.losses.append(Float64(loss.item()))
    var n = max(steps, 2)
    var best = Best()
    for s in range(2, n + timed):
        var t0 = perf_counter_ns()
        if capture:
            tape.replay()
        else:
            loss = one(opt)
        var l: Float64 = 0
        if read_loss:
            l = Float64(loss.item())
        else:   # timing only: a device sync, no loss read
            ctx.synchronize()
        if s < steps:
            r.losses.append(l)
        elif s >= n:
            best.add(Float64(perf_counter_ns() - t0) / 1e3)
    if timed == 0:
        r.params = flat(p)
    r.step_us = best.best
    _ = xd^
    _ = yd^
    return r^


def compiled_gpu(rt: Runtime, net: Net, steps: Int, timed: Int, path: String, stage: Int = 2,
                 chain_per_buffer: Bool = False, packed: Bool = False, capture: Bool = False,
                 read_loss: Bool = True) raises -> Run:
    """The compiled step on the accelerator: every argument copied to the
    device and lent by address; batch 0 at every step. `stage` as in
    `eager_gpu` (1 outputs the gradients instead of storing an update).
    `capture`: step 0 executes, step 1 is recorded as MAX's device graph
    (`M_captureModelSync`, CUDA / HIP only) and replayed, and every later
    step replays it on the same buffers."""
    var r = Run()
    var tape = Tape()
    var p = net.params(tape)
    var bt0 = batch(net, 0)
    var xb = tape.input(net.b, net.din, bt0[0].copy())
    var yb = tape.input(net.b, net.dout, bt0[1].copy())
    var step = emit_train_step(tape, net.loss(p, xb, yb), LR, device="gpu:0",
                               update=stage == 2, grads_out=stage == 1, adam_out=stage == 3,
                               chain_per_buffer=chain_per_buffer, packed=packed)
    with open(path, "w") as f:
        f.write(step.text)
    var t0 = perf_counter_ns()
    var model = rt.load(path)
    r.compile_ms = Float64(perf_counter_ns() - t0) / 1e6

    var staging = rt.tensor_map()
    var inputs = rt.tensor_map()
    var on_dev = List[Tensor]()

    def lend(name: String, var buf: HostBuffer, shape: List[Int]) raises {mut staging, mut inputs, mut on_dev, rt}:
        staging.borrow(name, buf^, DType.float32, shape.copy())
        var dev = staging.tensor(name).to_device(rt.device())
        inputs.borrow_address(name, dev.address(), DType.float32, shape.copy(), on_device=True)
        on_dev.append(dev^)

    var total = 0
    if packed:   # [every p, every m, every v, pow1, pow2], flat
        var st = flat(p)
        total = len(st)
        for _ in range(2 * total):
            st.append(0.0)
        st.append(1.0)
        st.append(1.0)
        lend("state", HostBuffer(copy_from=Int(st.unsafe_ptr()), nbytes=4 * len(st)), [len(st)])
        _ = st^
    for k in range(len(step.params) if not packed else 0):
        var vals = p[k].values()
        var shp = step.shapes[step.params[k]].copy()
        var nb = 4 * len(vals)
        lend(String("p", k), HostBuffer(copy_from=Int(vals.unsafe_ptr()), nbytes=nb), shp)
        lend(String("m", k), HostBuffer(nb), shp)
        lend(String("v", k), HostBuffer(nb), shp)
        _ = vals^
    var one: List[Scalar[DT]] = [1.0]
    if not packed:
        lend("pow1", HostBuffer(copy_from=Int(one.unsafe_ptr()), nbytes=4), [1, 1])
        lend("pow2", HostBuffer(copy_from=Int(one.unsafe_ptr()), nbytes=4), [1, 1])
    _ = one^
    # `bt0[k]` is a temporary copy: its pointer dangles once the expression
    # ends, so bind the lists first and keep them alive past the copies.
    var x0 = bt0[0].copy()
    var y0 = bt0[1].copy()
    lend(String("x", xb.id), HostBuffer(copy_from=Int(x0.unsafe_ptr()), nbytes=4 * len(x0)), [net.b, net.din])
    lend(String("x", yb.id), HostBuffer(copy_from=Int(y0.unsafe_ptr()), nbytes=4 * len(y0)), [net.b, net.dout])
    _ = x0^
    _ = y0^
    # capture takes the arguments positionally: the header's order
    var order = List[String]()
    for pre in ["p", "m", "v"]:
        for k in range(len(step.params) if not packed else 0):
            order.append(String(pre, k))
    if packed:
        order.append("state")
    else:
        order.append("pow1")
        order.append("pow2")
    order.append(String("x", xb.id))
    order.append(String("x", yb.id))
    var want = String("argument_names = [")
    for i in range(len(order)):
        want += String("\"", order[i], "\"", ", " if i + 1 < len(order) else "]")
    if capture and step.text.find(want) < 0:
        raise Error("compiled_gpu: argument order is not " + want)
    var args = List[Tensor]()
    for i in range(len(order) if capture else 0):
        args.append(inputs.tensor(order[i]))
    var outs = List[Tensor]()
    var best = Best()
    for s in range(steps + timed):
        var t1 = perf_counter_ns()
        var l: Float64
        if capture and s >= 1:
            if s == 1:
                # capture RECORDS without executing (measured on a 5090,
                # MAX 26.7: this step's loss read 0.0 and every later one
                # lagged a step): replay once so step 1 happens
                outs = model.capture(1, args)
            model.replay(1, args)
            # timing only (`read_loss=False`): the replay is synchronous, no loss read
            l = Float64(outs[0].to_host().item[DType.float32]()) if read_loss else 0.0
        else:
            l = Float64(model.execute(inputs).tensor("loss").to_host().item[DType.float32]())
        if s < steps:
            r.losses.append(l)
            if s == steps - 1 and packed:
                var h = inputs.tensor("state").to_host()
                var pk = h.data[Float32]()
                for i in range(total):
                    r.params.append(pk[unsafe_offset=i])
            elif s == steps - 1:
                for k in range(len(step.params)):
                    # read through the map MAX executes on: the Tensor
                    # handle `to_device` returned shows its INITIAL bytes on
                    # Metal (MAX 26.7) once the model has stored into them
                    var h = inputs.tensor(String("p", k)).to_host()
                    var pk = h.data[Float32]()
                    for i in range(h.num_elements()):
                        r.params.append(pk[unsafe_offset=i])
        else:
            best.add(Float64(perf_counter_ns() - t1) / 1e3)
    r.step_us = best.best
    _ = staging^
    return r^


def compiled_gpu_pingpong(rt: Runtime, net: Net, steps: Int, timed: Int, path: String,
                          refeed: Bool = True) raises -> Run:
    """The compiled step with NO buffers (`state_io`): parameters, moments
    and powers are tensor arguments, and each step's outputs P/M/V/pow*n
    are lent, by device address, as the next step's inputs. The previous
    step's output map is kept alive until the step that reads it is done."""
    var r = Run()
    var tape = Tape()
    var p = net.params(tape)
    var bt0 = batch(net, 0)
    var xb = tape.input(net.b, net.din, bt0[0].copy())
    var yb = tape.input(net.b, net.dout, bt0[1].copy())
    var step = emit_train_step(tape, net.loss(p, xb, yb), LR, device="gpu:0", state_io=True)
    with open(path, "w") as f:
        f.write(step.text)
    var t0 = perf_counter_ns()
    var model = rt.load(path)
    r.compile_ms = Float64(perf_counter_ns() - t0) / 1e6

    var staging = rt.tensor_map()
    var feed = rt.tensor_map()
    var on_dev = List[Tensor]()

    def lend(name: String, var buf: HostBuffer, shape: List[Int]) raises {mut staging, mut feed, mut on_dev, rt}:
        staging.borrow(name, buf^, DType.float32, shape.copy())
        var dev = staging.tensor(name).to_device(rt.device())
        feed.borrow_address(name, dev.address(), DType.float32, shape.copy(), on_device=True)
        on_dev.append(dev^)

    var np = len(step.params)
    var pshapes = List[List[Int]]()
    for k in range(np):
        var vals = p[k].values()
        var shp = step.shapes[step.params[k]].copy()
        var nb = 4 * len(vals)
        lend(String("p", k), HostBuffer(copy_from=Int(vals.unsafe_ptr()), nbytes=nb), shp)
        lend(String("m", k), HostBuffer(nb), shp)
        lend(String("v", k), HostBuffer(nb), shp)
        pshapes.append(shp^)
        _ = vals^
    var one: List[Scalar[DT]] = [1.0]
    lend("pow1", HostBuffer(copy_from=Int(one.unsafe_ptr()), nbytes=4), [1, 1])
    lend("pow2", HostBuffer(copy_from=Int(one.unsafe_ptr()), nbytes=4), [1, 1])
    _ = one^
    var x0 = bt0[0].copy()
    var y0 = bt0[1].copy()
    var xname = String("x", xb.id)
    var yname = String("x", yb.id)
    lend(xname, HostBuffer(copy_from=Int(x0.unsafe_ptr()), nbytes=4 * len(x0)), [net.b, net.din])
    lend(yname, HostBuffer(copy_from=Int(y0.unsafe_ptr()), nbytes=4 * len(y0)), [net.b, net.dout])
    _ = x0^
    _ = y0^
    var xaddr = on_dev[len(on_dev) - 2].address()
    var yaddr = on_dev[len(on_dev) - 1].address()
    var prev = rt.tensor_map()        # the outputs `feed` points into
    var held = List[Tensor]()
    var best = Best()
    for s in range(steps + timed):
        var t1 = perf_counter_ns()
        var out = model.execute(feed)
        var l = Float64(out.tensor("loss").to_host().item[DType.float32]())
        if not refeed:
            if s >= steps:
                best.add(Float64(perf_counter_ns() - t1) / 1e3)
            continue
        var t2 = perf_counter_ns()
        var nxt = rt.tensor_map()
        var hold = List[Tensor]()
        def relend(src: String, dst: String, shape: List[Int]) raises {mut nxt, mut hold, out}:
            var t = out.tensor(src)
            nxt.borrow_address(dst, t.address(), DType.float32, shape.copy(), on_device=True)
            hold.append(t^)
        for k in range(np):
            relend(String("P", k), String("p", k), pshapes[k])
            relend(String("M", k), String("m", k), pshapes[k])
            relend(String("V", k), String("v", k), pshapes[k])
        relend("pow1n", "pow1", [1, 1])
        relend("pow2n", "pow2", [1, 1])
        nxt.borrow_address(xname, xaddr, DType.float32, [net.b, net.din], on_device=True)
        nxt.borrow_address(yname, yaddr, DType.float32, [net.b, net.dout], on_device=True)
        feed = nxt^
        held = hold^
        prev = out^
        r.relend_us += Float64(perf_counter_ns() - t2) / 1e3
        if s < steps:
            r.losses.append(l)
            if s == steps - 1:
                for k in range(np):
                    var h = prev.tensor(String("P", k)).to_host()
                    var pk = h.data[Float32]()
                    for i in range(h.num_elements()):
                        r.params.append(pk[unsafe_offset=i])
        else:
            best.add(Float64(perf_counter_ns() - t1) / 1e3)
    r.step_us = best.best
    r.relend_us /= Float64(steps + timed)
    _ = held^
    _ = prev^
    _ = staging^
    _ = on_dev^
    return r^


def levers(dir: String) raises:
    """The two ways to drop the in-place stores' cost on the GPU: one chain
    per buffer, and no buffers (outputs fed back as inputs). Gated against
    the eager CPU tape, then timed against the eager GPU tape and the
    one-chain stores, interleaved over 3 rounds."""
    var grt = Runtime(accelerator=True)
    var ctx = DeviceContext()
    comptime STEPS = 20
    var mlp = Net(64, 17, 64, 6, 0)
    var e = eager(mlp, STEPS, 0, fixed=True)
    var ok = True
    var c1 = compiled_gpu(grt, mlp, STEPS, 0, dir + "/lever_gate_chains.mlir", chain_per_buffer=True)
    ok = compare("one chain per buffer vs eager CPU", c1, e, 1e-4, 2.5e-3) and ok
    var c2 = compiled_gpu_pingpong(grt, mlp, STEPS, 0, dir + "/lever_gate_pingpong.mlir")
    ok = compare("ping-pong vs eager CPU", c2, e, 1e-4, 2.5e-3) and ok
    var shapes: List[List[Int]] = [[64, 17, 256, 6], [1024, 128, 512, 16]]
    for i in range(len(shapes)):
        var nt = Net(shapes[i][0], shapes[i][1], shapes[i][2], shapes[i][3], 0)
        print("B ", nt.b, ", ", nt.din, " -> ", nt.h, " -> ", nt.h, " -> ", nt.dout,
              " (us per step; eager / stores, one chain / one chain per buffer / ping-pong)", sep="")
        for rd in range(3):
            var eg = eager_gpu(ctx, nt, 200)
            var cs = compiled_gpu(grt, nt, 0, 200, String(dir, "/lever_", i, "_stores.mlir"))
            var cc = compiled_gpu(grt, nt, 0, 200, String(dir, "/lever_", i, "_chains.mlir"),
                                  chain_per_buffer=True)
            var cp = compiled_gpu_pingpong(grt, nt, 0, 200, String(dir, "/lever_", i, "_pingpong.mlir"))
            print("  round ", rd, ": ", Int(eg), " / ", Int(cs.step_us), " / ", Int(cc.step_us), " / ",
                  Int(cp.step_us), sep="")
    print("LEVERS GATES OK" if ok else "LEVERS GATES FAILED")
    if not ok:
        raise Error("lever gate failed")


def compiled_gpu_double(rt: Runtime, net: Net, steps: Int, timed: Int, path: String) raises -> Run:
    """The compiled step with DOUBLE-BUFFERED state (`double_buffer`): two
    device copies of every parameter, moment and power, lent to ONE map as
    set 0 and set 1. Two compiled directions alternate (0 -> 1, then
    1 -> 0), so a step does no host work beyond the execute."""
    var r = Run()
    var tape = Tape()
    var p = net.params(tape)
    var bt0 = batch(net, 0)
    var xb = tape.input(net.b, net.din, bt0[0].copy())
    var yb = tape.input(net.b, net.dout, bt0[1].copy())
    var loss = net.loss(p, xb, yb)
    var fwd = emit_train_step(tape, loss, LR, device="gpu:0", double_buffer=True)
    var bwd = emit_train_step(tape, loss, LR, device="gpu:0", double_buffer=True, flip=True)
    with open(path + ".a.mlir", "w") as f:
        f.write(fwd.text)
    with open(path + ".b.mlir", "w") as f:
        f.write(bwd.text)
    var t0 = perf_counter_ns()
    var model_a = rt.load(path + ".a.mlir")
    var model_b = rt.load(path + ".b.mlir")
    r.compile_ms = Float64(perf_counter_ns() - t0) / 1e6

    var staging = rt.tensor_map()
    var inputs = rt.tensor_map()
    var on_dev = List[Tensor]()

    def lend(name: String, var buf: HostBuffer, shape: List[Int]) raises {mut staging, mut inputs, mut on_dev, rt}:
        staging.borrow(name, buf^, DType.float32, shape.copy())
        var dev = staging.tensor(name).to_device(rt.device())
        inputs.borrow_address(name, dev.address(), DType.float32, shape.copy(), on_device=True)
        on_dev.append(dev^)

    for k in range(len(fwd.params)):
        var vals = p[k].values()
        var shp = fwd.shapes[fwd.params[k]].copy()
        var nb = 4 * len(vals)
        for sfx in ["_0", "_1"]:
            lend(String("p", k, sfx), HostBuffer(copy_from=Int(vals.unsafe_ptr()), nbytes=nb), shp)
            lend(String("m", k, sfx), HostBuffer(nb), shp)
            lend(String("v", k, sfx), HostBuffer(nb), shp)
        _ = vals^
    var one: List[Scalar[DT]] = [1.0]
    for sfx in ["_0", "_1"]:
        lend("pow1" + sfx, HostBuffer(copy_from=Int(one.unsafe_ptr()), nbytes=4), [1, 1])
        lend("pow2" + sfx, HostBuffer(copy_from=Int(one.unsafe_ptr()), nbytes=4), [1, 1])
    _ = one^
    var x0 = bt0[0].copy()
    var y0 = bt0[1].copy()
    lend(String("x", xb.id), HostBuffer(copy_from=Int(x0.unsafe_ptr()), nbytes=4 * len(x0)), [net.b, net.din])
    lend(String("x", yb.id), HostBuffer(copy_from=Int(y0.unsafe_ptr()), nbytes=4 * len(y0)), [net.b, net.dout])
    _ = x0^
    _ = y0^
    var best = Best()
    for s in range(steps + timed):
        var t1 = perf_counter_ns()
        var l: Float64
        if s % 2 == 0:
            l = Float64(model_a.execute(inputs).tensor("loss").to_host().item[DType.float32]())
        else:
            l = Float64(model_b.execute(inputs).tensor("loss").to_host().item[DType.float32]())
        if s < steps:
            r.losses.append(l)
            if s == steps - 1:
                # step s stored into set 1 if s is even, set 0 if odd
                var sfx = "_1" if s % 2 == 0 else "_0"
                for k in range(len(fwd.params)):
                    var h = inputs.tensor(String("p", k, sfx)).to_host()
                    var pk = h.data[Float32]()
                    for i in range(h.num_elements()):
                        r.params.append(pk[unsafe_offset=i])
        else:
            best.add(Float64(perf_counter_ns() - t1) / 1e3)
    r.step_us = best.best
    _ = staging^
    _ = on_dev^
    return r^


def pingpong_probe(dir: String) raises:
    """Why ping-pong is not faster, and what slows MAX round after round:
    the same B64 graph run back to back, alternating ping-pong (outputs fed
    back), the same inputs re-fed, and the stores, 4 times each."""
    var grt = Runtime(accelerator=True)
    var ctx = DeviceContext()
    var nt = Net(64, 17, 256, 6, 0)
    for rd in range(4):
        var eg = eager_gpu(ctx, nt, 200)
        var cs = compiled_gpu(grt, nt, 0, 200, dir + "/lever_0_stores.mlir")
        var c3 = compiled_gpu(grt, nt, 0, 200, dir + "/stage_0_3.mlir", 3)
        var cf = compiled_gpu_pingpong(grt, nt, 0, 200, dir + "/lever_0_pingpong.mlir", refeed=False)
        var cp = compiled_gpu_pingpong(grt, nt, 0, 200, dir + "/lever_0_pingpong.mlir")
        print("  round ", rd, ": eager ", Int(eg), " | stores ", Int(cs.step_us), " | outputs+buffers ",
              Int(c3.step_us), " | state io, same inputs ", Int(cf.step_us), " | ping-pong ",
              Int(cp.step_us), " (re-lend ", Int(cp.relend_us), ")", sep="")


def packed_probe(dir: String) raises:
    """One store instead of 20: every parameter and moment in ONE flat
    buffer (`packed`). Gated against the eager CPU tape, then timed
    against eager, the 20 stores and the outputs-only graph, interleaved."""
    var grt = Runtime(accelerator=True)
    var ctx = DeviceContext()
    var mlp = Net(64, 17, 64, 6, 0)
    var e = eager(mlp, 20, 0, fixed=True)
    var cg = compiled_gpu(grt, mlp, 20, 0, dir + "/packed_gate.mlir", packed=True)
    var ok = compare("one packed store vs eager CPU", cg, e, 1e-4, 2.5e-3)
    var cd = compiled_gpu_double(grt, mlp, 20, 0, dir + "/double_gate")
    ok = compare("double-buffered stores vs eager CPU", cd, e, 1e-4, 2.5e-3) and ok
    var shapes: List[List[Int]] = [[64, 17, 256, 6], [1024, 128, 512, 16]]
    for i in range(len(shapes)):
        var nt = Net(shapes[i][0], shapes[i][1], shapes[i][2], shapes[i][3], 0)
        print("B ", nt.b, ", ", nt.din, " -> ", nt.h, " -> ", nt.h, " -> ", nt.dout, " (us per step)", sep="")
        for rd in range(3):
            var eg = eager_gpu(ctx, nt, 200)
            var cs = compiled_gpu(grt, nt, 0, 200, String(dir, "/lever_", i, "_stores.mlir"))
            var cp = compiled_gpu(grt, nt, 0, 200, String(dir, "/packed_", i, ".mlir"), packed=True)
            var c3 = compiled_gpu(grt, nt, 0, 200, String(dir, "/stage_", i, "_3.mlir"), 3)
            var cd = compiled_gpu_double(grt, nt, 0, 200, String(dir, "/double_", i))
            print("  round ", rd, ": eager ", Int(eg), " | 20 stores ", Int(cs.step_us), " | 1 packed store ",
                  Int(cp.step_us), " | outputs, no store ", Int(c3.step_us), " | double-buffered ",
                  Int(cd.step_us), sep="")
    print("PACKED/DOUBLE GATES OK" if ok else "PACKED/DOUBLE GATES FAILED")
    if not ok:
        raise Error("packed gate failed")


def nvidia(dir: String) raises:
    """On NVIDIA the baseline to beat is the CAPTURED tape. Gates against
    the eager CPU tape: the eager and captured tape on the GPU, MAX with
    in-place stores, ping-pong and one packed store. Then the step time at
    four sizes, interleaved over 3 rounds: eager tape | captured tape |
    MAX 20 stores | MAX 1 packed store | MAX ping-pong | MAX outputs, no
    store (a lower bound, not a training step)."""
    var grt = Runtime(accelerator=True)
    var ctx = DeviceContext()
    print("GPU:", ctx.name())
    var mlp = Net(64, 17, 64, 6, 0)
    var e = eager(mlp, 20, 0, fixed=True)
    var ok = True
    var tg = tape_gpu(ctx, mlp, 20, 0, False)
    ok = compare("eager tape on the GPU vs eager CPU", tg, e, 1e-4, 2.5e-3) and ok
    var tc = tape_gpu(ctx, mlp, 20, 0, True)
    ok = compare("captured tape vs eager CPU", tc, e, 1e-4, 2.5e-3) and ok
    var cs = compiled_gpu(grt, mlp, 20, 0, dir + "/nv_gate_stores.mlir")
    print("    compile:", cs.compile_ms, "ms")
    ok = compare("MAX stores vs eager CPU", cs, e, 1e-4, 2.5e-3) and ok
    var cp = compiled_gpu_pingpong(grt, mlp, 20, 0, dir + "/nv_gate_pingpong.mlir")
    ok = compare("MAX ping-pong vs eager CPU", cp, e, 1e-4, 2.5e-3) and ok
    var cpk = compiled_gpu(grt, mlp, 20, 0, dir + "/nv_gate_packed.mlir", packed=True)
    ok = compare("MAX packed vs eager CPU", cpk, e, 1e-4, 2.5e-3) and ok
    var ccap = compiled_gpu(grt, mlp, 20, 0, dir + "/nv_gate_stores.mlir", capture=True)
    ok = compare("MAX captured vs eager CPU", ccap, e, 1e-4, 2.5e-3) and ok
    ok = compare("MAX captured vs MAX executed (same TF32 math)", ccap, cs, 1e-6, 1e-6) and ok
    print("NVIDIA GATES OK" if ok else "NVIDIA GATES FAILED")
    if len(argv()) > 3 and String(argv()[3]) == "--captured-only":
        var shapes1: List[List[Int]] = [[1, 17, 256, 6], [1024, 128, 512, 16]]
        for i in range(len(shapes1)):
            var nt = Net(shapes1[i][0], shapes1[i][1], shapes1[i][2], shapes1[i][3], 0)
            print("B ", nt.b, ", ", nt.din, " -> ", nt.h, " -> ", nt.h, " -> ", nt.dout, " (us per step)", sep="")
            for rd in range(3):
                var ec = tape_gpu(ctx, nt, 0, 200, True)
                var ms = compiled_gpu(grt, nt, 0, 200, String(dir, "/nv_", i, "_stores.mlir"))
                var mc = compiled_gpu(grt, nt, 0, 200, String(dir, "/nv_", i, "_stores.mlir"), capture=True)
                var en = tape_gpu(ctx, nt, 0, 200, True, read_loss=False)
                var mn = compiled_gpu(grt, nt, 0, 200, String(dir, "/nv_", i, "_stores.mlir"), capture=True,
                                      read_loss=False)
                print("  round ", rd, ": captured tape ", Int(ec.step_us), " | MAX executed ", Int(ms.step_us),
                      " | MAX captured ", Int(mc.step_us), " || no loss read: captured tape ", Int(en.step_us),
                      " | MAX captured ", Int(mn.step_us), "  [load ms: ", Int(mc.compile_ms), "]", sep="")
        if not ok:
            raise Error("nvidia gate failed")
        return
    var shapes: List[List[Int]] = [[1, 17, 256, 6], [64, 17, 256, 6], [256, 64, 256, 8], [1024, 128, 512, 16]]
    for i in range(len(shapes)):
        var nt = Net(shapes[i][0], shapes[i][1], shapes[i][2], shapes[i][3], 0)
        print("B ", nt.b, ", ", nt.din, " -> ", nt.h, " -> ", nt.h, " -> ", nt.dout, " (us per step)", sep="")
        for rd in range(3):
            var eg = tape_gpu(ctx, nt, 0, 200, False)
            var ec = tape_gpu(ctx, nt, 0, 200, True)
            var ms = compiled_gpu(grt, nt, 0, 200, String(dir, "/nv_", i, "_stores.mlir"))
            var mk = compiled_gpu(grt, nt, 0, 200, String(dir, "/nv_", i, "_packed.mlir"), packed=True)
            var mp = compiled_gpu_pingpong(grt, nt, 0, 200, String(dir, "/nv_", i, "_pingpong.mlir"))
            var mo = compiled_gpu(grt, nt, 0, 200, String(dir, "/nv_", i, "_outputs.mlir"), 3)
            print("  round ", rd, ": eager ", Int(eg.step_us), " | captured ", Int(ec.step_us),
                  " | MAX 20 stores ", Int(ms.step_us), " | 1 packed ", Int(mk.step_us),
                  " | ping-pong ", Int(mp.step_us), " (relend ", Int(mp.relend_us), ")",
                  " | outputs, no store ", Int(mo.step_us),
                  "  [compile ms: ", Int(ms.compile_ms), "]", sep="")
    if not ok:
        raise Error("nvidia gate failed")


def debug_gpu(rt: Runtime, net: Net, path: String, update: Bool) raises:
    """Which MLP node drifts on the GPU: each layer's value from ONE compiled
    execution vs the CPU tape's forward on the same weights and batch."""
    var tape = Tape()
    var p = net.params(tape)
    var bt = batch(net, 0)
    var x = tape.input(net.b, net.din, bt[0].copy())
    var y = tape.input(net.b, net.dout, bt[1].copy())
    var h1 = F.linear(x, p[0], p[1], "relu")
    var h2 = F.linear(h1, p[2], p[3], "tanh")
    var pred = F.linear(h2, p[4], p[5])
    var dd = pred - y
    var loss = (dd * dd).mean()
    var ids: List[Int] = [h1.id, h2.id, pred.id, y.id, dd.id]
    var step = emit_train_step(tape, loss, LR, device="gpu:0", extra=ids.copy(), update=update)
    with open(path, "w") as f:
        f.write(step.text)
    var model = rt.load(path)
    var staging = rt.tensor_map()
    var inputs = rt.tensor_map()
    var on_dev = List[Tensor]()
    def lend(name: String, var buf: HostBuffer, shape: List[Int]) raises {mut staging, mut inputs, mut on_dev, rt}:
        staging.borrow(name, buf^, DType.float32, shape.copy())
        var dev = staging.tensor(name).to_device(rt.device())
        inputs.borrow_address(name, dev.address(), DType.float32, shape.copy(), on_device=True)
        on_dev.append(dev^)
    for k in range(len(step.params)):
        var vals = p[k].values()
        var shp = step.shapes[step.params[k]].copy()
        lend(String("p", k), HostBuffer(copy_from=Int(vals.unsafe_ptr()), nbytes=4 * len(vals)), shp)
        lend(String("m", k), HostBuffer(4 * len(vals)), shp)
        lend(String("v", k), HostBuffer(4 * len(vals)), shp)
        _ = vals^
    var one: List[Scalar[DT]] = [1.0]
    lend("pow1", HostBuffer(copy_from=Int(one.unsafe_ptr()), nbytes=4), [1, 1])
    lend("pow2", HostBuffer(copy_from=Int(one.unsafe_ptr()), nbytes=4), [1, 1])
    var x0 = bt[0].copy()
    var y0 = bt[1].copy()
    lend(String("x", x.id), HostBuffer(copy_from=Int(x0.unsafe_ptr()), nbytes=4 * len(x0)), [net.b, net.din])
    lend(String("x", y.id), HostBuffer(copy_from=Int(y0.unsafe_ptr()), nbytes=4 * len(y0)), [net.b, net.dout])
    _ = x0^
    _ = y0^
    var out = model.execute(inputs)
    var names: List[String] = ["h1 = relu(x w1 + b1)", "h2 = tanh(h1 w2 + b2)", "pred = h2 w3 + b3", "y (input)", "pred - y"]
    var nodes: List[FTensor] = [h1, h2, pred, y, dd]
    for j in range(5):
        var want = nodes[j].values()
        var t = out.tensor(String("n", ids[j])).to_host()
        var d = t.data[Float32]()
        var worst: Float64 = 0
        var at = 0
        for i in range(len(want)):
            var e = abs(Float64(d[unsafe_offset=i]) - Float64(want[i]))
            if e > worst:
                worst = e
                at = i
        print("    ", names[j], ": worst abs ", worst, " at ", at, " (MAX ", d[unsafe_offset=at],
              ", eager ", want[at], ")", sep="")
    print("    loss: MAX", out.tensor("loss").to_host().item[DType.float32](), " eager", loss.item())
    _ = staging^
    _ = one^


def compiled(rt: Runtime, net: Net, steps: Int, timed: Int, path: String, mutant: Bool) raises -> Run:
    var r = Run()
    var tape = Tape()
    var p = net.params(tape)
    var bt0 = batch(net, 0)
    var xb = tape.input(net.b, net.din, bt0[0].copy())
    var yb = tape.input(net.b, net.dout, bt0[1].copy())
    var step = emit_train_step(tape, net.loss(p, xb, yb), LR, mutant=mutant)
    with open(path, "w") as f:
        f.write(step.text)
    var t0 = perf_counter_ns()
    var model = rt.load(path)
    r.compile_ms = Float64(perf_counter_ns() - t0) / 1e6

    var inputs = rt.tensor_map()
    for k in range(len(step.params)):
        var vals = p[k].values()
        var shp = step.shapes[step.params[k]].copy()
        var nb = 4 * len(vals)
        inputs.borrow(String("p", k), HostBuffer(copy_from=Int(vals.unsafe_ptr()), nbytes=nb),
                      DType.float32, shp.copy())
        inputs.borrow(String("m", k), HostBuffer(nb), DType.float32, shp.copy())
        inputs.borrow(String("v", k), HostBuffer(nb), DType.float32, shp.copy())
        _ = vals^
    var one: List[Scalar[DT]] = [1.0]
    inputs.borrow("pow1", HostBuffer(copy_from=Int(one.unsafe_ptr()), nbytes=4), DType.float32, [1, 1])
    inputs.borrow("pow2", HostBuffer(copy_from=Int(one.unsafe_ptr()), nbytes=4), DType.float32, [1, 1])
    _ = one^
    var xname = String("x", xb.id)
    var yname = String("x", yb.id)
    inputs.borrow(xname, HostBuffer(4 * net.b * net.din), DType.float32, [net.b, net.din])
    inputs.borrow(yname, HostBuffer(4 * net.b * net.dout), DType.float32, [net.b, net.dout])

    var best = Best()
    for s in range(steps + timed):
        var bt = batch(net, s % steps)
        var t1 = perf_counter_ns()
        var px = inputs.owned_data[Float32](xname)
        var py = inputs.owned_data[Float32](yname)
        for i in range(len(bt[0])):
            px[unsafe_offset=i] = bt[0][i]
        for i in range(len(bt[1])):
            py[unsafe_offset=i] = bt[1][i]
        var l = Float64(model.execute(inputs).tensor("loss").item[DType.float32]())
        if s < steps:
            r.losses.append(l)
            if s == steps - 1:
                for k in range(len(step.params)):
                    var pk = inputs.owned_data[Float32](String("p", k))
                    var n = step.shapes[step.params[k]][0] * step.shapes[step.params[k]][1]
                    for i in range(n):
                        r.params.append(pk[unsafe_offset=i])
        else:
            best.add(Float64(perf_counter_ns() - t1) / 1e3)
    r.step_us = best.best
    return r^


def compare(name: String, got: Run, want: Run, ltol: Float64, ptol: Float64) -> Bool:
    var lw: Float64 = 0
    for i in range(len(want.losses)):
        lw = max(lw, abs(got.losses[i] - want.losses[i]) / max(1e-6, abs(want.losses[i])))
    var pw: Float64 = 0
    var pmoved = 0
    for i in range(len(want.params)):
        var d = abs(Float64(got.params[i]) - Float64(want.params[i]))
        pw = max(pw, d)
        if d > 1e-5:
            pmoved += 1
    var head = String("      first losses eager/MAX:")
    for i in range(min(4, len(want.losses))):
        head += String(" ", want.losses[i], "/", got.losses[i])
    print(head)
    var ok = len(got.losses) == len(want.losses) and len(got.params) == len(want.params) \
        and lw <= ltol and pw <= ptol
    print("    ", name, ": loss ", want.losses[0], " -> ", want.losses[len(want.losses) - 1],
          " (eager) vs ", got.losses[len(got.losses) - 1], " (MAX); worst loss rel ", lw,
          ", worst param abs ", pw, " (", pmoved, " of ", len(want.params), " > 1e-5)  ",
          "OK" if ok else "FAIL", sep="")
    return ok


def stages(dir: String) raises:
    """The GPU step in stages (forward / + backward / + Adam), eager tape
    vs MAX, interleaved over 3 rounds (synced timings on Metal drift with
    the GPU clock). Fastest 40-step chunk of 200 per cell."""
    var grt = Runtime(accelerator=True)
    var ctx = DeviceContext()
    var names: List[String] = ["forward", "+ backward", "+ Adam", "+ Adam, outputs not stores"]
    var shapes: List[List[Int]] = [[64, 17, 256, 6], [1024, 128, 512, 16]]
    for i in range(len(shapes)):
        var nt = Net(shapes[i][0], shapes[i][1], shapes[i][2], shapes[i][3], 0)
        print("B ", nt.b, ", ", nt.din, " -> ", nt.h, " -> ", nt.h, " -> ", nt.dout,
              " (us; eager / MAX per round)", sep="")
        for st in range(4):
            var line = String("  ", names[st], ":")
            for _ in range(3):
                var eg = eager_gpu(ctx, nt, 200, min(st, 2))
                var cg = compiled_gpu(grt, nt, 0, 200, String(dir, "/stage_", i, "_", st, ".mlir"), st)
                line += String("  ", Int(eg), " / ", Int(cg.step_us))
            print(line)


def main() raises:
    var dir = String(argv()[1]) if len(argv()) > 1 else String("/tmp")
    var rt = Runtime()
    var ok = True
    comptime STEPS = 20
    if len(argv()) > 2 and String(argv()[2]) == "--stages":
        stages(dir)
        return
    if len(argv()) > 2 and String(argv()[2]) == "--nvidia":
        nvidia(dir)
        return
    if len(argv()) > 2 and String(argv()[2]) == "--packed":
        packed_probe(dir)
        return
    if len(argv()) > 2 and String(argv()[2]) == "--pingpong-probe":
        pingpong_probe(dir)
        return
    if len(argv()) > 2 and String(argv()[2]) == "--levers":
        levers(dir)
        return
    if len(argv()) > 2 and String(argv()[2]) == "--gpu-debug":
        var drt = Runtime(accelerator=True)
        var dnet = Net(64, 17, 64, 6, 0)
        print("  with the in-place stores:")
        debug_gpu(drt, dnet, dir + "/mlp_gpu_debug.mlir", True)
        print("  WITHOUT the stores (the step updates nothing):")
        debug_gpu(drt, dnet, dir + "/mlp_gpu_debug_nostore.mlir", False)
        return

    print("1. MLP 17 -> 64 relu -> 64 tanh -> 6, MSE, Adam, batch 64,", STEPS, "steps")
    var mlp = Net(64, 17, 64, 6, 0)
    var e1 = eager(mlp, STEPS, 0)
    var c1 = compiled(rt, mlp, STEPS, 0, dir + "/mlp.mlir", False)
    print("    compile (first load in this process):", c1.compile_ms, "ms")
    ok = compare("MAX vs eager", c1, e1, 1e-4, 2.5e-3) and ok

    print("2. every other rank-2 op")
    var wide = Net(32, 9, 16, 5, 1)
    var e2 = eager(wide, STEPS, 0)
    var c2 = compiled(rt, wide, STEPS, 0, dir + "/wide.mlir", False)
    print("    compile:", c2.compile_ms, "ms")
    ok = compare("MAX vs eager", c2, e2, 1e-4, 2.5e-3) and ok

    print("3. mutant (tanh backward 1 + y^2) must fail gate 1")
    var cm = compiled(rt, mlp, STEPS, 0, dir + "/mlp_mutant.mlir", True)
    var caught = not compare("mutant vs eager", cm, e1, 1e-4, 2.5e-3)
    print("    mutant caught:", caught)
    ok = caught and ok

    print("4. time per train step (CPU, host buffers; fastest 40-step chunk of 200)")
    var shapes: List[List[Int]] = [[1, 17, 256, 6], [64, 17, 256, 6], [256, 64, 256, 8], [1024, 128, 512, 16]]
    for i in range(len(shapes)):
        var nt = Net(shapes[i][0], shapes[i][1], shapes[i][2], shapes[i][3], 0)
        var et = eager(nt, STEPS, 200)
        var ct = compiled(rt, nt, STEPS, 200, String(dir, "/time_", i, ".mlir"), False)
        print("    B ", nt.b, ", ", nt.din, " -> ", nt.h, " -> ", nt.h, " -> ", nt.dout, ": eager ",
              et.step_us, " us, MAX ", ct.step_us, " us (x", et.step_us / ct.step_us,
              "), compile ", ct.compile_ms, " ms", sep="")
    if len(argv()) > 2 and String(argv()[2]) == "--gpu-debug":
        var drt = Runtime(accelerator=True)
        print("  with the in-place stores:")
        debug_gpu(drt, mlp, dir + "/mlp_gpu_debug.mlir", True)
        print("  WITHOUT the stores (the step updates nothing):")
        debug_gpu(drt, mlp, dir + "/mlp_gpu_debug_nostore.mlir", False)
        return
    var want_gpu = len(argv()) > 2 and String(argv()[2]) == "--gpu"
    if want_gpu:
        var grt = Runtime(accelerator=True)
        var ctx = DeviceContext()
        print("5. GPU (", ctx.name(), "): the compiled step on the device, batch 0 every step", sep="")
        var e5 = eager(mlp, STEPS, 0, fixed=True)
        var c5 = compiled_gpu(grt, mlp, STEPS, 0, dir + "/mlp_gpu.mlir")
        print("    compile:", c5.compile_ms, "ms")
        ok = compare("MAX on the GPU vs eager CPU", c5, e5, 1e-4, 2.5e-3) and ok
        for i in range(len(shapes)):
            var nt = Net(shapes[i][0], shapes[i][1], shapes[i][2], shapes[i][3], 0)
            var eg = eager_gpu(ctx, nt, 200)
            var cg = compiled_gpu(grt, nt, STEPS, 200, String(dir, "/gtime_", i, ".mlir"))
            print("    B ", nt.b, ", ", nt.din, " -> ", nt.h, " -> ", nt.h, " -> ", nt.dout,
                  ": eager tape on the GPU ", eg, " us, MAX ", cg.step_us, " us (x", eg / cg.step_us,
                  "), compile ", cg.compile_ms, " ms", sep="")
    print("ALL OK" if ok else "SOME FAILED")
    if not ok:
        raise Error("tape_to_mo gate failed")
