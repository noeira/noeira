"""`AneAdamGpu`: the ANE training step with Adam on the GPU, and no CPU loop
over parameters or gradients.

    var g = AneGrad(a, tape, loss, [x, y])    # emit + compile, as before
    var opt = AneAdamGpu(a, g^, tape, lr=3e-3) # fp32 masters from the tape
    opt.set_data(a, 0, x_ptr); opt.set_data(a, 1, y_ptr)   # the CPU, as before
    var loss = opt.step(a, scale)              # NaN: not finite, nothing updated
    opt.download(tape)                         # masters -> the tape's values

One step:
1. the CPU writes the loss-scale seed when it changed (one scalar);
2. the ANE evaluates (synchronous);
3. ONE metal_rt command buffer:
   - per gradient, a check kernel reads the fp16 gradient straight from its
     ANE output surface (wrapped with `MetalRT.wrap_external`, padded rows)
     and raises a flag on any non-finite value;
   - per parameter, an Adam kernel returns at once if the flag is up. Else
     it reads the gradient again, multiplies by 1/(scale · grad_div),
     updates the fp32 master w, m, v, and writes fp16(w) straight into the
     ANE input surface (padded rows) for the next evaluation;
4. one sync;
5. the CPU reads the flag and the loss (output o000, one fp16).

Adam is nn's (`nn.optimizer.adam`, the CPU per-parameter path): the same host
step state (running products β₁ᵗ, β₂ᵗ in fp32, `bc = 1 - βᵗ`), the same
expression `p - lr * (m / bc1) / (sqrt(v / bc2) + eps)`, decoupled decay on
matrices only. A skipped step (non-finite gradient) changes nothing: not
the masters, not the moments, not the step count.

Kernels use 32-bit index math on 2-D grids (columns in x, rows in y): the
GPU emulates 64-bit division (`probe_zero_copy.mojo`).
"""

from std.math import sqrt
from std.memory import bitcast
from std.time import perf_counter_ns
from max.gpu import block_idx, thread_idx

from noeira.nn.constants import DT
from noeira.experimental.functional import Tape
from noeira.experimental.metal_rt.objc import _sym, _as_fn, CFn1
from noeira.experimental.metal_rt.runtime import MetalRT, Buffer, COMMIT_EVERY_N
from .runtime import ANE, Program
from .step import AneGrad

comptime P16 = Pointer[Float16, MutAnyOrigin]
comptime P32 = Pointer[Float32, MutAnyOrigin]
comptime PU32 = Pointer[UInt32, MutAnyOrigin]


def check_kernel(flag: PU32, g: P16, rows_: Int64, cols_: Int64, gstride_: Int64, bx_: Int64):
    """`flag[0] = 1` if any element of the [rows, cols] fp16 gradient (row
    stride `gstride`) is Inf or NaN: its exponent bits are all ones."""
    var c = Int32(block_idx.x) * Int32(bx_) + Int32(thread_idx.x)
    var r = Int32(block_idx.y)
    if c >= Int32(cols_) or r >= Int32(rows_):
        return
    var bits = bitcast[DType.uint16](g[unsafe_offset=Int(r * Int32(gstride_) + c)])
    if (bits & 0x7C00) == 0x7C00:
        flag[unsafe_offset=0] = 1


def adam_kernel(
    w: P32, m: P32, v: P32, g: P16, out16: P16, flag: PU32,
    rows_: Int64, cols_: Int64, gstride_: Int64, wstride_: Int64, bx_: Int64,
    gscale: Float32, lr: Float32, beta1: Float32, beta2: Float32, eps: Float32,
    bc1: Float32, bc2: Float32, wd: Float32, decay: Int64,
):
    """`nn` Adam's CPU per-parameter update on one element, then fp16(w) into
    the ANE input surface at row stride `wstride`."""
    if flag[unsafe_offset=0] != 0:
        return
    var c = Int32(block_idx.x) * Int32(bx_) + Int32(thread_idx.x)
    var r = Int32(block_idx.y)
    var cols = Int32(cols_)
    if c >= cols or r >= Int32(rows_):
        return
    var i = Int(r * cols + c)
    var one = Float32(1.0)
    var p = w[unsafe_offset=i]
    if decay != 0:
        p -= lr * wd * p
    var gt = g[unsafe_offset=Int(r * Int32(gstride_) + c)].cast[DType.float32]() * gscale
    var m_new = beta1 * m[unsafe_offset=i] + (one - beta1) * gt
    var v_new = beta2 * v[unsafe_offset=i] + (one - beta2) * gt * gt
    m[unsafe_offset=i] = m_new
    v[unsafe_offset=i] = v_new
    p = p - lr * (m_new / bc1) / (sqrt(v_new / bc2) + eps)
    w[unsafe_offset=i] = p
    out16[unsafe_offset=Int(r * Int32(wstride_) + c)] = p.cast[DType.float16]()


def cast_kernel(out16: P16, w: P32, rows_: Int64, cols_: Int64, wstride_: Int64, bx_: Int64):
    """`fp16(w)` into the surface at row stride `wstride`."""
    var c = Int32(block_idx.x) * Int32(bx_) + Int32(thread_idx.x)
    var r = Int32(block_idx.y)
    var cols = Int32(cols_)
    if c >= cols or r >= Int32(rows_):
        return
    out16[unsafe_offset=Int(r * Int32(wstride_) + c)] = w[unsafe_offset=Int(r * cols + c)].cast[DType.float16]()


struct _P(Copyable):
    """One parameter: its node, shape, program input / output, buffers."""

    var id: Int
    var rows: Int
    var cols: Int
    var input: Int
    var output: Int  # -1: no gradient (frozen)
    var w: Buffer
    var m: Buffer
    var v: Buffer
    var bx: Int  # threadgroup width: cols rounded up to 32, at most 256

    def __init__(out self, id: Int, rows: Int, cols: Int, input: Int, output: Int,
                 w: Buffer, m: Buffer, v: Buffer):
        self.id = id
        self.rows = rows
        self.cols = cols
        self.input = input
        self.output = output
        self.w = w
        self.m = m
        self.v = v
        self.bx = min(256, (cols + 31) // 32 * 32)


struct AneAdamGpu(Movable):
    var g: AneGrad
    var rt: MetalRT
    var params: List[_P]
    var ins: List[Buffer]  # wrapped ANE input surfaces (by program input)
    var outs: List[Buffer]  # wrapped ANE output surfaces
    var flag: Buffer
    var lr: Float32
    var beta1: Float32
    var beta2: Float32
    var eps: Float32
    var wd: Float32
    var t: Int
    var b1_pow: Float32
    var b2_pow: Float32
    var seed_now: Float32
    var t_eval: Int
    var t_gpu: Int
    var steps: Int
    var skipped: Int
    var mutant_dense: Bool
    """TEST ONLY: write fp16 weights at the dense row stride."""
    var mutant_no_bc: Bool
    """TEST ONLY: drop Adam's bias correction."""

    def __init__(
        out self, a: ANE, var g: AneGrad, mut tape: Tape,
        lr: Float32, beta1: Float32 = 0.9, beta2: Float32 = 0.999,
        eps: Float32 = 1e-8, wd: Float32 = 0.0,
        mutant_dense: Bool = False, mutant_no_bc: Bool = False,
    ) raises:
        self.g = g^
        self.rt = MetalRT(policy=COMMIT_EVERY_N, batch_n=1 << 30)
        self.params = List[_P]()
        self.ins = List[Buffer]()
        self.outs = List[Buffer]()
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.wd = wd
        self.t = 0
        self.b1_pow = 1.0
        self.b2_pow = 1.0
        self.seed_now = -1.0
        self.t_eval = 0
        self.t_gpu = 0
        self.steps = 0
        self.skipped = 0
        self.mutant_dense = mutant_dense
        self.mutant_no_bc = mutant_no_bc
        self.flag = self.rt.alloc[DType.uint32](4)
        var alloc_size = _as_fn[CFn1](_sym(a.o.system, "IOSurfaceGetAllocSize"))
        ref prog = self.g.prog
        for s in prog.ins:
            self.ins.append(self.rt.wrap_external(a.base_addr(s), alloc_size(s)))
        for s in prog.outs:
            self.outs.append(self.rt.wrap_external(a.base_addr(s), alloc_size(s)))
        ref td = tape.data[]
        ref st = self.g.step
        for k in range(len(st.args)):
            var id = st.args[k]
            if id >= td.n_params:
                continue
            var rows = st.shapes[id][0]
            var cols = st.shapes[id][1]
            var out = -1
            for j in range(len(st.grads)):
                if st.grads[j] == id:
                    out = j + 1
            var w = self.rt.alloc[DType.float32](rows * cols)
            var src = td.val_ptr(id)
            var dst = w.host_ptr[DType.float32]()
            for i in range(rows * cols):
                dst[unsafe_offset=i] = src[unsafe_offset=i]
            var m = self.rt.alloc[DType.float32](rows * cols)
            var v = self.rt.alloc[DType.float32](rows * cols)
            self.params.append(_P(id, rows, cols, k, out, w, m, v))
        # the first fp16 weights, straight into the surfaces
        for q in range(len(self.params)):
            ref p = self.params[q]
            self.rt.launch2d[cast_kernel](
                self.ins[p.input].dev[DType.float16](), p.w.dev[DType.float32](),
                Int64(p.rows), Int64(p.cols), Int64(self._wstride(p)), Int64(p.bx),
                grid_x=(p.cols + p.bx - 1) // p.bx, grid_y=p.rows, block_x=p.bx,
            )
        self.rt.sync()

    def _wstride(self, p: _P) -> Int:
        return p.cols if self.mutant_dense else self.g.prog.in_layout[p.input].row

    def set_data(self, a: ANE, k: Int, src: Pointer[Float32, _]) raises:
        self.g.set_data(a, k, src)

    def step(mut self, a: ANE, scale: Float32, evaluate: Bool = True) raises -> Float64:
        """One training step at loss scale `scale` (the seed). Returns the
        loss, or NaN when a gradient was not finite (then nothing changed).
        `evaluate=False` (measurement only) skips the ANE evaluation and
        reruns the GPU part on the previous gradients."""
        ref prog = self.g.prog
        if scale != self.seed_now:
            var s: List[Float32] = [scale]
            prog.write(a, len(self.g.step.args), Pointer(to=s[0]))
            self.seed_now = scale
        var t0 = perf_counter_ns()
        if evaluate:
            prog.run(a)
        var t1 = perf_counter_ns()
        # the step state of step t + 1, kept only if the step is taken
        var b1p = self.b1_pow * self.beta1
        var b2p = self.b2_pow * self.beta2
        var bc1 = Float32(1.0) - b1p
        var bc2 = Float32(1.0) - b2p
        if self.mutant_no_bc:
            bc1 = 1.0
            bc2 = 1.0
        var gscale = Float32(1.0 / (Float64(scale) * self.g.step.grad_div))
        self.flag.host_ptr[DType.uint32]()[unsafe_offset=0] = 0
        for q in range(len(self.params)):
            ref p = self.params[q]
            if p.output < 0:
                continue
            self.rt.launch2d[check_kernel](
                self.flag.dev[DType.uint32](), self.outs[p.output].dev[DType.float16](),
                Int64(p.rows), Int64(p.cols), Int64(prog.out_layout[p.output].row), Int64(p.bx),
                grid_x=(p.cols + p.bx - 1) // p.bx, grid_y=p.rows, block_x=p.bx,
            )
        for q in range(len(self.params)):
            ref p = self.params[q]
            if p.output < 0:
                continue
            self.rt.launch2d[adam_kernel](
                p.w.dev[DType.float32](), p.m.dev[DType.float32](), p.v.dev[DType.float32](),
                self.outs[p.output].dev[DType.float16](), self.ins[p.input].dev[DType.float16](),
                self.flag.dev[DType.uint32](),
                Int64(p.rows), Int64(p.cols), Int64(prog.out_layout[p.output].row),
                Int64(self._wstride(p)), Int64(p.bx),
                gscale, self.lr, self.beta1, self.beta2, self.eps, bc1, bc2, self.wd,
                Int64(1 if p.rows > 1 else 0),
                grid_x=(p.cols + p.bx - 1) // p.bx, grid_y=p.rows, block_x=p.bx,
            )
        self.rt.sync()
        var t2 = perf_counter_ns()
        self.t_eval += Int(t1 - t0)
        self.t_gpu += Int(t2 - t1)
        self.steps += 1
        if self.flag.host_ptr[DType.uint32]()[unsafe_offset=0] != 0:
            self.skipped += 1
            var z = Float64(0)
            return z / z
        self.t += 1
        self.b1_pow = b1p
        self.b2_pow = b2p
        return Float64(a.base(prog.outs[0])[unsafe_offset=0].cast[DType.float32]())

    def download(mut self, mut tape: Tape) raises:
        """The fp32 masters into the tape's parameter values."""
        self.rt.sync()
        ref td = tape.data[]
        for q in range(len(self.params)):
            ref p = self.params[q]
            var src = p.w.host_ptr[DType.float32]()
            var dst = td.val_ptr(p.id)
            for i in range(p.rows * p.cols):
                dst[unsafe_offset=i] = src[unsafe_offset=i]

    def master(self, q: Int) -> Buffer:
        return self.params[q].w

    def free(mut self, a: ANE) raises:
        for b in self.ins:
            self.rt.free(b)
        for b in self.outs:
            self.rt.free(b)
        self.g.free(a)
