"""Muon — orthogonalised momentum for hidden weight matrices.

`torch.optim.Muon` (Jordan et al. 2024; Liu et al. 2025, "Muon is Scalable for
LLM Training") with `adjust_lr_fn="match_rms_adamw"`, for the matrices a model
REGISTERS with it; every other parameter stays with the model's Adam:

    buf  ← buf + (1 − μ)(g − buf)                     momentum, as an EMA
    U    = g + μ(buf − g)                             Nesterov (or buf)
    X    = U / max(‖U‖_F, eps)
    5 ×  X ← a X + (b G + c G²) X,  G = X Xᵀ          Newton–Schulz, (a, b, c) =
                                                      (3.4445, −4.7750, 2.0315)
    W    ← W (1 − lr·wd) − lr · 0.2 √max(IN, OUT) · X

The iteration pushes every singular value of U towards 1 (it lands them in
about [0.7, 1.2], by design of the coefficients), so the step has the same
size in every direction of the weight matrix. The 0.2 √max(IN, OUT) factor
gives it Adam's update RMS, so Adam's learning rate is reused unchanged.

HOW IT SHARES A MODEL WITH ADAM. `step` runs AFTER the gradient clip and
BEFORE the model's Adam step. It walks the model's params, updates the
registered matrices from their (clipped) gradients and then ZEROES those
gradients. Adam's step over the whole model then leaves them alone: with a
gradient that is always zero its first moment is zero, and so is its update.
Adam must have no weight decay of its own on these matrices.

Weights are `[IN, OUT]` row-major (torch's transposed). Orthogonalisation
commutes with the transpose, so the Gram matrix is taken on the smaller side:
X Xᵀ when IN ≤ OUT, Xᵀ X otherwise.

TARGETS. CPU: `gemm_cpu` (Accelerate on macOS). NVIDIA: `cublas_gemm` in full
fp32, every buffer kept on the device, no host read. Other GPUs (Metal): the
step runs on the host from a download of the matrix and its gradient; that
path is for gates, not for speed.

Differences from torch: the Newton–Schulz products are fp32 here and bfloat16
there, so the two agree to bfloat16's noise, not to the bit
(`tests/nn/test_muon.mojo` gates against torch's algorithm run in float64, and
reports the distance to torch's own bfloat16 result).
"""

from std.math import sqrt
from std.sys import default_accelerator

from max.gpu import global_idx, thread_idx, block_idx
from max.gpu.primitives import block
from max.gpu.host import DeviceBuffer, DeviceContext

from noeira.nn.constants import DT, TPB
from ..core.cpu_gemm import gemm_cpu
from ..core.cublas_gemm import cublas_gemm
from ..core.param import ParamVisitor, ParamWalkable
from ..core.tensor import Tensor


comptime MUON_NS_A: Float64 = 3.4445
comptime MUON_NS_B: Float64 = -4.7750
comptime MUON_NS_C: Float64 = 2.0315
comptime MUON_NS_STEPS: Int = 5
comptime MUON_EPS: Float64 = 1e-7
comptime MUON_RED_TPB: Int = 128          # the single-block reduction's width


# ── device kernels (NVIDIA path) ─────────────────────────────────────────────
def _muon_momentum_kernel(
    g: Pointer[Scalar[DT], MutAnyOrigin],
    buf: Pointer[Scalar[DT], MutAnyOrigin],
    x: Pointer[Scalar[DT], MutAnyOrigin],
    n_arg: Int64,
    mu: Scalar[DT],
    nesterov: Int64,
    partials: Pointer[Scalar[DT], MutAnyOrigin],
):
    """buf ← buf + (1 − μ)(g − buf); x = the update U; partials[block] = Σ U²."""
    var n = Int(n_arg)
    var i = Int(global_idx.x)
    var sq: Scalar[DT] = 0.0
    if i < n:
        var gi = g[unsafe_offset=i]
        var b = buf[unsafe_offset=i]
        b = b + (Scalar[DT](1.0) - mu) * (gi - b)
        buf[unsafe_offset=i] = b
        var u = gi + mu * (b - gi) if nesterov != 0 else b
        x[unsafe_offset=i] = u
        sq = u * u
    var tot = block.sum[block_size=TPB, broadcast=False](val=sq)
    if Int(thread_idx.x) == 0:
        partials[unsafe_offset=Int(block_idx.x)] = tot[0]


def _muon_inv_norm_kernel(
    partials: Pointer[Scalar[DT], MutAnyOrigin],
    n_blocks_arg: Int64,
    inv: Pointer[Scalar[DT], MutAnyOrigin],
    eps: Scalar[DT],
):
    """inv[0] = 1 / max(√Σ partials, eps), one block."""
    var n_blocks = Int(n_blocks_arg)
    var t = Int(thread_idx.x)
    var my_sum: Scalar[DT] = 0.0
    var k = t
    while k < n_blocks:
        my_sum += partials[unsafe_offset=k]
        k += MUON_RED_TPB
    var s = block.sum[block_size=MUON_RED_TPB, broadcast=False](val=my_sum)
    if t == 0:
        var norm = sqrt(s[0])
        inv[unsafe_offset=0] = Scalar[DT](1.0) / (norm if norm > eps else eps)


def _muon_scale_kernel(x: Pointer[Scalar[DT], MutAnyOrigin], n_arg: Int64, inv: Pointer[Scalar[DT], MutAnyOrigin]):
    var i = Int(global_idx.x)
    if i < Int(n_arg):
        x[unsafe_offset=i] = x[unsafe_offset=i] * inv[unsafe_offset=0]


def _muon_apply_kernel(
    w: Pointer[Scalar[DT], MutAnyOrigin],
    x: Pointer[Scalar[DT], MutAnyOrigin],
    g: Pointer[Scalar[DT], MutAnyOrigin],
    n_arg: Int64,
    keep: Scalar[DT],
    step: Scalar[DT],
):
    """w ← w·keep − step·x, and the gradient is spent: g ← 0."""
    var i = Int(global_idx.x)
    if i < Int(n_arg):
        w[unsafe_offset=i] = w[unsafe_offset=i] * keep - step * x[unsafe_offset=i]
        g[unsafe_offset=i] = 0


# ── host Newton–Schulz ───────────────────────────────────────────────────────
def _ns_step_host(x: Tensor, mut out: Tensor, mut g: Tensor, mut g2: Tensor, r: Int, c: Int) raises:
    """One Newton–Schulz step of `x` [r, c] into `out`, with the Gram matrix
    on the smaller side (`g`, `g2`: [k, k], k = min(r, c))."""
    var k = min(r, c)
    comptime A = Float32(MUON_NS_A)
    comptime B = Float32(MUON_NS_B)
    comptime C = Float32(MUON_NS_C)
    if r <= c:
        gemm_cpu[False, True](r, r, c, 1.0, x, c, x, c, 0.0, g, r)            # G = X Xᵀ
    else:
        gemm_cpu[True, False](c, c, r, 1.0, x, c, x, c, 0.0, g, c)            # G = Xᵀ X
    for i in range(k * k):
        g2.data[i] = g.data[i]
    gemm_cpu[False, False](k, k, k, C, g, k, g, k, B, g2, k)                  # G2 = b G + c G G
    for i in range(r * c):
        out.data[i] = x.data[i]
    if r <= c:
        gemm_cpu[False, False](r, c, r, 1.0, g2, r, x, c, A, out, c)          # a X + G2 X
    else:
        gemm_cpu[False, False](r, c, c, 1.0, x, c, g2, c, A, out, c)          # a X + X G2


def muon_orthogonalize_host(mut x: Tensor, r: Int, c: Int, steps: Int = MUON_NS_STEPS) raises:
    """`x` [r, c] (already of Frobenius norm ≤ 1) ← its Newton–Schulz
    orthogonalisation, in place."""
    var k = min(r, c)
    var y = Tensor.alloc(r * c)
    var g = Tensor.alloc(k * k)
    var g2 = Tensor.alloc(k * k)
    var in_x = True
    for _ in range(steps):
        if in_x:
            _ns_step_host(x, y, g, g2, r, c)
        else:
            _ns_step_host(y, x, g, g2, r, c)
        in_x = not in_x
    if not in_x:
        for i in range(r * c):
            x.data[i] = y.data[i]


struct Muon(Movable, ParamVisitor):
    """Muon over the matrices added with `add`; see the module docstring."""
    var lr: Scalar[DT]            # Adam's lr for the same model (set it before every step)
    var momentum: Float64
    var nesterov: Bool
    var wd: Float64               # decoupled, with the UNADJUSTED lr: W ← W (1 − lr·wd); torch's default is 0.1
    var ns_steps: Int
    var names: List[String]       # the registered params' dotted names
    var rows: List[Int]           # IN of each [IN, OUT] row-major weight
    var cols: List[Int]           # OUT
    var bufs: List[Tensor]        # momentum, on the target the first step runs on
    # NVIDIA scratch, allocated at a matrix's first step and kept
    var xs: List[Tensor]
    var x2s: List[Tensor]
    var gs: List[Tensor]
    var g2s: List[Tensor]
    var parts: List[Tensor]
    var invs: List[Tensor]
    var hits: Int                 # registered params met in the current step's walk
    var steps: Int

    def __init__(out self, lr: Scalar[DT], momentum: Float64 = 0.95, nesterov: Bool = True, wd: Float64 = 0.0):
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.wd = wd
        self.ns_steps = MUON_NS_STEPS
        self.names = List[String]()
        self.rows = List[Int]()
        self.cols = List[Int]()
        self.bufs = List[Tensor]()
        self.xs = List[Tensor]()
        self.x2s = List[Tensor]()
        self.gs = List[Tensor]()
        self.g2s = List[Tensor]()
        self.parts = List[Tensor]()
        self.invs = List[Tensor]()
        self.hits = 0
        self.steps = 0

    def __init__(out self, *, deinit move: Self):
        self.lr = move.lr
        self.momentum = move.momentum
        self.nesterov = move.nesterov
        self.wd = move.wd
        self.ns_steps = move.ns_steps
        self.names = move.names^
        self.rows = move.rows^
        self.cols = move.cols^
        self.bufs = move.bufs^
        self.xs = move.xs^
        self.x2s = move.x2s^
        self.gs = move.gs^
        self.g2s = move.g2s^
        self.parts = move.parts^
        self.invs = move.invs^
        self.hits = move.hits
        self.steps = move.steps

    def add(mut self, name: String, rows: Int, cols: Int):
        """Register the weight `name` (its dotted name in the model's param
        walk), a `[rows, cols]` = `[IN, OUT]` row-major matrix."""
        self.names.append(name.copy())
        self.rows.append(rows)
        self.cols.append(cols)
        self.bufs.append(Tensor())
        self.xs.append(Tensor())
        self.x2s.append(Tensor())
        self.gs.append(Tensor())
        self.g2s.append(Tensor())
        self.parts.append(Tensor())
        self.invs.append(Tensor())

    def step[target: StaticString, M: ParamWalkable](mut self, mut model: M, ctx: Optional[DeviceContext] = None) raises:
        """One Muon step on every registered matrix of `model`, from its
        current gradients, which it then zeroes. Raises when a registered
        name is not among the model's params (a typo would otherwise leave
        that matrix to Adam, silently)."""
        self.hits = 0
        model.for_each_param[target](self, ctx)
        if self.hits != len(self.names):
            raise Error(
                "Muon.step: " + String(len(self.names)) + " matrices registered, " + String(self.hits)
                + " found in the model's params (check the names given to `add`)"
            )
        self.steps += 1

    def _slot(self, name: String) -> Int:
        for i in range(len(self.names)):
            if self.names[i] == name:
                return i
        return -1

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool, ctx: Optional[DeviceContext],
    ) raises:
        var s = self._slot(name)
        if s < 0:
            return
        var r = self.rows[s]
        var c = self.cols[s]
        if r * c != N:
            raise Error("Muon: " + name + " was registered as [" + String(r) + ", " + String(c) + "], the param holds " + String(N))
        self.hits += 1
        var keep = Scalar[DT](1.0 - Float64(self.lr) * self.wd)
        var step = Scalar[DT](Float64(self.lr) * 0.2 * sqrt(Float64(max(r, c))))
        comptime if target == "gpu":
            comptime if default_accelerator().is_nvidia_gpu():
                self._step_cuda(s, r, c, param, grad, keep, step, ctx.value())
            else:
                # no BLAS-shaped device GEMM here: the step runs on the host
                param.download(ctx.value())
                grad.download(ctx.value())
                self._step_host(s, r, c, param, grad, keep, step)
                param.upload_resident(ctx.value())
                grad.upload_resident(ctx.value())
        else:
            self._step_host(s, r, c, param, grad, keep, step)
        param.version += 1          # the layers' derived weight caches rebuild

    def _step_host(mut self, s: Int, r: Int, c: Int, mut param: Tensor, mut grad: Tensor, keep: Scalar[DT], step: Scalar[DT]) raises:
        var n = r * c
        self.bufs[s].ensure(n)
        var x = Tensor.alloc(n)
        var mu = Scalar[DT](self.momentum)
        var sq = 0.0
        for i in range(n):
            var gi = grad.data[i]
            var b = self.bufs[s].data[i]
            b = b + (Scalar[DT](1.0) - mu) * (gi - b)
            self.bufs[s].data[i] = b
            var u = gi + mu * (b - gi) if self.nesterov else b
            x.data[i] = u
            sq += Float64(u) * Float64(u)
        var inv = Scalar[DT](1.0 / max(sqrt(sq), MUON_EPS))
        for i in range(n):
            x.data[i] *= inv
        muon_orthogonalize_host(x, r, c, self.ns_steps)
        for i in range(n):
            param.data[i] = param.data[i] * keep - step * x.data[i]
            grad.data[i] = 0

    def _step_cuda(
        mut self, s: Int, r: Int, c: Int, mut param: Tensor, mut grad: Tensor, keep: Scalar[DT], step: Scalar[DT], ctx: DeviceContext,
    ) raises:
        comptime if default_accelerator().is_nvidia_gpu():
            var n = r * c
            var k = min(r, c)
            var nblk = (n + TPB - 1) // TPB
            if not self.bufs[s].dev:
                self.bufs[s] = Tensor.alloc_gpu(ctx, n)          # zero-filled
                self.xs[s] = Tensor.alloc_gpu(ctx, n)
                self.x2s[s] = Tensor.alloc_gpu(ctx, n)
                self.gs[s] = Tensor.alloc_gpu(ctx, k * k)
                self.g2s[s] = Tensor.alloc_gpu(ctx, k * k)
                self.parts[s] = Tensor.alloc_gpu(ctx, nblk)
                self.invs[s] = Tensor.alloc_gpu(ctx, 1)
            var cur = self.xs[s].dev.value()
            var nxt = self.x2s[s].dev.value()
            var g = self.gs[s].dev.value()
            var g2 = self.g2s[s].dev.value()
            ctx.enqueue_function[_muon_momentum_kernel](
                grad.dev.value(), self.bufs[s].dev.value(), cur, Int64(n), Scalar[DT](self.momentum),
                Int64(1 if self.nesterov else 0), self.parts[s].dev.value(),
                grid_dim=nblk, block_dim=TPB,
            )
            ctx.enqueue_function[_muon_inv_norm_kernel](
                self.parts[s].dev.value(), Int64(nblk), self.invs[s].dev.value(), Scalar[DT](MUON_EPS),
                grid_dim=1, block_dim=MUON_RED_TPB,
            )
            ctx.enqueue_function[_muon_scale_kernel](cur, Int64(n), self.invs[s].dev.value(), grid_dim=nblk, block_dim=TPB)
            comptime A = Float32(MUON_NS_A)
            comptime B = Float32(MUON_NS_B)
            comptime C = Float32(MUON_NS_C)
            for _ in range(self.ns_steps):
                if r <= c:
                    cublas_gemm[False, True, False](ctx, g, cur, cur, r, r, c, 0.0)          # G = X Xᵀ
                else:
                    cublas_gemm[True, False, False](ctx, g, cur, cur, c, c, r, 0.0)          # G = Xᵀ X
                ctx.enqueue_copy(g2, g)
                cublas_gemm[False, False, False](ctx, g2, g, g, k, k, k, B, C)               # G2 = b G + c G G
                ctx.enqueue_copy(nxt, cur)
                if r <= c:
                    cublas_gemm[False, False, False](ctx, nxt, g2, cur, r, c, r, A)          # a X + G2 X
                else:
                    cublas_gemm[False, False, False](ctx, nxt, cur, g2, r, c, c, A)          # a X + X G2
                var tmp = cur
                cur = nxt
                nxt = tmp
            ctx.enqueue_function[_muon_apply_kernel](
                param.dev.value(), cur, grad.dev.value(), Int64(n), keep, step, grid_dim=nblk, block_dim=TPB,
            )
        else:
            raise Error("Muon._step_cuda: not an NVIDIA build")
