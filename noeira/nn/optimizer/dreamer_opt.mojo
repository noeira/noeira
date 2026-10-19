"""DreamerOpt — the DreamerV3 reference optimizer chain, storage-native.

Reproduces, bit-for-bit, the chain in `references/dreamerv3-main/embodied/jax/opt.py`:

    optax.chain(
        clip_by_agc(agc),                   # agc=0.3, pmin=1e-3
        scale_by_rms(beta2, eps),           # beta2=0.999, eps=1e-20
        scale_by_momentum(beta1, nesterov), # beta1=0.9, nesterov=False
        scale_by_learning_rate(sched),      # const + linear warmup
    )

NOT AdamW — the moment buffers are wired differently. The RMS transform runs
FIRST (normalising the AGC-clipped grad by its own running RMS); the momentum
buffer then accumulates the RMS-NORMALISED grad. Per element (after AGC has
scaled the whole-leaf gradient by `agc_scale`):

    g      = grad · agc_scale
    nu     = beta2·nu + (1-beta2)·g²          # rms 2nd moment
    nu_hat = nu / (1 - beta2^t)
    g_rms  = g / (sqrt(nu_hat) + eps)
    mu     = beta1·mu + (1-beta1)·g_rms        # momentum on the normalised g
    mu_hat = mu / (1 - beta1^t)
    p      = p - lr·mu_hat

AGC (adaptive gradient clipping) is per-Param (= per JAX pytree leaf):

    gnorm = ‖grad‖₂ ; pnorm = ‖param‖₂
    upper = agc_clip · max(agc_pmin, pnorm)
    agc_scale = min(1, upper/gnorm)   (== 1/max(1, gnorm/upper))

Storage port (vs `nn/optimizer/dreamer_opt.mojo`): DRAMATICALLY simpler. Storage's
`Param` already owns its two optimizer-moment Tensors (`m`, `v`), so the two
Dreamer moments map directly — **`mu` → `m` (momentum), `nu` → `v` (rms)** — and
the entire legacy `nu_flat`/`mu_flat`/`offsets`/`TargetStorage` flat-side-table
machinery is GONE. This is a per-param `ParamVisitor` (one CPU loop / GPU
kernel-pair per Param), mirroring storage `Adam`'s shape: `step` bumps the bias
correction (`begin_step`) then walks `for_each_param`; AGC's per-leaf reduction
fits inside the SAME `visit` (no separate pass), reusing one [1] device scale
slot across the sequential per-param walk.

For graph-owned params (DreamerV3's WM/AC loss graphs own params as ComputeGraph
nodes — a ComputeGraph is NOT a Module), drive it as
`opt.begin_step(); walk_params[target](graph, prefix, opt, ctx)`.

Drive the warmup schedule from the trainer via `opt.lr = sched.lr_at(step)`
before each step (see `schedules.LinearWarmupSchedule`). No CUDA-graph / arena
mode in this port — the legacy per-param path was the default too.
"""

from std.math import sqrt
from max.gpu import global_idx, thread_idx, block_idx
from max.gpu.primitives import block
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT, TPB
from ..core.tensor import Tensor
from ..core.param import ParamVisitor, walk_params, ParamVisitorRT
from ..core.param import ParamWalkable
from .optimizer import Optimizer
from noeira.core.tiles import RowMajor, Tile


comptime AGC_TPB: Int = 128   # per-block reduction width
comptime AGC_MAXB: Int = 256  # blocks for the multi-block grad/param-norm reduce


# ── GPU kernels ─────────────────────────────────────────────────────────
# AGC's per-leaf ‖grad‖²/‖param‖² used to be a SINGLE 128-thread block over the
# whole param → latency-bound (no memory-level parallelism) and catastrophic for
# big leaves (a ~14M-element Linear took ~16 ms). It is now a two-pass reduction:
# `_agc_partials_kernel` (AGC_MAXB blocks across all SMs → AGC_MAXB partial sums)
# then `_agc_finalize_kernel` (1 block reduces the partials → agc_scale).
def _agc_partials_kernel_rt(
    param: Pointer[Scalar[DT], MutAnyOrigin],
    grad: Pointer[Scalar[DT], MutAnyOrigin],
    n_arg: Int64,
    partials: Tile[DT, 2 * AGC_MAXB],
):
    """Pass A with the length at RUNTIME (one kernel for every Param; reached
    through `ParamVisitorRef`): AGC_MAXB blocks each strided-sum a chunk of
    ‖grad‖²/‖param‖² → partials[block] (g²) and partials[AGC_MAXB+block] (p²)."""
    var n = Int(n_arg)
    var t = Int(thread_idx.x)
    var b = Int(block_idx.x)
    comptime STRIDE = AGC_MAXB * AGC_TPB
    var g_sum: Scalar[DT] = 0.0
    var p_sum: Scalar[DT] = 0.0
    var k = b * AGC_TPB + t
    while k < n:
        var g = grad[unsafe_offset=k]
        var pv = param[unsafe_offset=k]
        g_sum += g * g
        p_sum += pv * pv
        k += STRIDE
    var gt = block.sum[block_size=AGC_TPB, broadcast=False](val=g_sum)
    var pt = block.sum[block_size=AGC_TPB, broadcast=False](val=p_sum)
    if t == 0:
        partials[b] = gt[0]
        partials[AGC_MAXB + b] = pt[0]


def _agc_finalize_kernel(
    partials: Tile[DT, 2 * AGC_MAXB],
    scale_buf: Tile[DT, 1],
    agc_clip: Scalar[DT],
    agc_pmin: Scalar[DT],
):
    """Pass B: reduce the AGC_MAXB partials → ‖grad‖/‖param‖ → agc_scale (1
    block, AGC_MAXB threads — one per partial)."""
    var t = Int(thread_idx.x)
    var g_total = block.sum[block_size=AGC_MAXB, broadcast=False](
        val=rebind[Scalar[DT]](partials[t])
    )
    var p_total = block.sum[block_size=AGC_MAXB, broadcast=False](
        val=rebind[Scalar[DT]](partials[AGC_MAXB + t])
    )
    if t == 0:
        var scale: Scalar[DT] = 1.0
        if agc_clip > Scalar[DT](0.0):
            var gnorm = sqrt(g_total[0])
            var pnorm = sqrt(p_total[0])
            var pclamp = pnorm if pnorm > agc_pmin else agc_pmin
            var upper = agc_clip * pclamp
            if upper > Scalar[DT](0.0):
                var ratio = gnorm / upper
                if ratio > Scalar[DT](1.0):
                    scale = Scalar[DT](1.0) / ratio
        scale_buf[0] = scale


def _dreamer_update_kernel_rt(
    param: Pointer[Scalar[DT], MutAnyOrigin],
    grad: Pointer[Scalar[DT], MutAnyOrigin],
    m: Pointer[Scalar[DT], MutAnyOrigin],  # mu (momentum)
    v: Pointer[Scalar[DT], MutAnyOrigin],  # nu (rms)
    n_arg: Int64,
    scale_buf: Tile[DT, 1],
    lr: Scalar[DT],
    beta1: Scalar[DT],
    beta2: Scalar[DT],
    eps: Scalar[DT],
    powbuf: Tile[DT, 2],
):
    """rms → momentum → lr with the length at RUNTIME; the AGC scale is read
    from `scale_buf[0]` and `bc1/bc2 = 1 − β^t` from the device `powbuf` so
    the bias correction advances under CUDA-graph replay."""
    var i = Int(global_idx.x)
    if i >= Int(n_arg):
        return
    var one: Scalar[DT] = 1.0
    var bc1 = one - rebind[Scalar[DT]](powbuf[0])
    var bc2 = one - rebind[Scalar[DT]](powbuf[1])
    var sc = rebind[Scalar[DT]](scale_buf[0])
    var g = grad[unsafe_offset=i] * sc
    var nu_new = beta2 * v[unsafe_offset=i] + (one - beta2) * g * g
    v[unsafe_offset=i] = nu_new
    var nu_hat = nu_new / bc2
    var g_rms = g / (sqrt(nu_hat) + eps)
    var mu_new = beta1 * m[unsafe_offset=i] + (one - beta1) * g_rms
    m[unsafe_offset=i] = mu_new
    var mu_hat = mu_new / bc1
    param[unsafe_offset=i] = param[unsafe_offset=i] - lr * mu_hat


def _dreamer_advance_pow_kernel(
    powbuf: Tile[DT, 2],
    beta1: Scalar[DT],
    beta2: Scalar[DT],
):
    """Advance the device bias-correction powers `[β₁ᵗ, β₂ᵗ]` (1 thread, once
    per step before the per-param walk). β^t lives on-device so it advances on
    every CUDA-graph replay. Mirrors `_adam_advance_pow_kernel`."""
    if Int(global_idx.x) != 0:
        return
    powbuf[0] = rebind[Scalar[DT]](powbuf[0]) * beta1
    powbuf[1] = rebind[Scalar[DT]](powbuf[1]) * beta2


struct DreamerOpt(Movable, ParamVisitor, ParamVisitorRT, Optimizer):
    var lr: Scalar[DT]
    var beta1: Scalar[DT]  # momentum
    var beta2: Scalar[DT]  # rms
    var eps: Scalar[DT]
    var agc_clip: Scalar[DT]
    var agc_pmin: Scalar[DT]
    var t: Int
    var _b1_pow: Scalar[DT]
    var _b2_pow: Scalar[DT]
    var bc1: Scalar[DT]
    var bc2: Scalar[DT]
    # One [1] device scale slot reused across the sequential per-param GPU walk
    # (AGC kernel writes it, the update kernel reads it; ordered on the stream).
    # Allocated lazily on the first GPU visit. Empty on CPU.
    var _scale: Tensor
    # Device-resident bias-correction powers `[β₁ᵗ, β₂ᵗ]` (GPU path only).
    # Advanced on-device once per step by `begin_step_gpu` so the correction
    # advances under CUDA-graph replay; the update kernel reads `bc = 1 − β^t`
    # from it. CPU path uses the host `bc1/bc2`. Empty until the first GPU step.
    var _pow_dev: Tensor
    # [2*AGC_MAXB] device scratch for the multi-block AGC norm reduction
    # (partial ‖grad‖²/‖param‖² sums). Reused across the sequential per-param
    # walk (each leaf overwrites it, stream-ordered). Allocated lazily; empty on
    # CPU.
    var _agc_partials: Tensor

    def __init__(out self):
        """No-arg default (satisfies Defaultable for the generic Trainer)."""
        self = Self(lr=Scalar[DT](4e-5))

    def __init__(
        out self,
        lr: Scalar[DT],  # required → disambiguates from the no-arg ctor
        beta1: Scalar[DT] = 0.9,
        beta2: Scalar[DT] = 0.999,
        eps: Scalar[DT] = 1e-20,
        agc_clip: Scalar[DT] = 0.3,
        agc_pmin: Scalar[DT] = 1e-3,
    ):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.agc_clip = agc_clip
        self.agc_pmin = agc_pmin
        self.t = 0
        self._b1_pow = Scalar[DT](1.0)
        self._b2_pow = Scalar[DT](1.0)
        self.bc1 = Scalar[DT](1.0)
        self.bc2 = Scalar[DT](1.0)
        self._scale = Tensor()
        self._pow_dev = Tensor()
        self._agc_partials = Tensor()

    def begin_step(mut self):
        """Bump the step counter + refresh bias corrections. Once per step,
        BEFORE the param walk (the graph-driving trainer calls this explicitly)."""
        self.t += 1
        self._b1_pow = self._b1_pow * self.beta1
        self._b2_pow = self._b2_pow * self.beta2
        self.bc1 = Scalar[DT](1.0) - self._b1_pow
        self.bc2 = Scalar[DT](1.0) - self._b2_pow

    def begin_step_gpu(mut self, ctx: DeviceContext) raises:
        """GPU twin of `begin_step`: advance the device powers `[β₁ᵗ, β₂ᵗ]`
        on-device (capture-safe) once per step, BEFORE the per-param walk. Lazily
        allocates `_pow_dev=[1,1]` on the first call (β⁰). The host `t` is still
        bumped for diagnostics; host `bc1/bc2` are unused on GPU."""
        self.t += 1
        if not self._pow_dev.dev:
            self._pow_dev = Tensor.alloc_gpu(ctx, 2)
            self._pow_dev.dev.value().enqueue_fill(Scalar[DT](1.0))
        ctx.enqueue_function[_dreamer_advance_pow_kernel](
            self._pow_dev.tt["gpu", 2](),
            self.beta1, self.beta2, grid_dim=1, block_dim=1,
        )

    def step[
        target: StaticString, M: ParamWalkable
    ](mut self, mut model: M, ctx: Optional[DeviceContext] = None) raises:
        """Bump the step then update every Param (per-param walk, both targets).
        GPU advances the device `[β₁ᵗ, β₂ᵗ]` (capture-safe); CPU bumps host bc."""
        comptime if target == "gpu":
            self.begin_step_gpu(ctx.value())
        else:
            self.begin_step()
        walk_params[target](model, self, ctx)

    def set_lr(mut self, lr: Scalar[DT]):
        self.lr = lr

    def get_lr(self) -> Scalar[DT]:
        return self.lr

    def visit_rt[target: StaticString](
        mut self,
        name: String,
        mut param: Tensor,
        mut grad: Tensor,
        mut m: Tensor,
        mut v: Tensor,
        n: Int,
        apply_decay: Bool,  # DreamerV3 config wd=0 → ignored (no decay term)
        ctx: Optional[DeviceContext],
    ) raises:
        if grad.n == 0:  # frozen param: no gradient, nothing to do
            return
        comptime if target == "cpu":
            m.ensure(n)  # mu (momentum) — lazy zero-alloc on first step
            v.ensure(n)  # nu (rms)
            var one = Scalar[DT](1.0)

            # ── AGC: per-leaf ‖grad‖₂ and ‖param‖₂ → agc_scale ──
            var g_sumsq: Scalar[DT] = 0.0
            var p_sumsq: Scalar[DT] = 0.0
            for i in range(n):
                var gs = grad.data[i]
                var ps = param.data[i]
                g_sumsq += gs * gs
                p_sumsq += ps * ps
            var agc_scale: Scalar[DT] = 1.0
            if self.agc_clip > Scalar[DT](0.0):
                var gnorm = sqrt(g_sumsq)
                var pnorm = sqrt(p_sumsq)
                var pclamp = pnorm if pnorm > self.agc_pmin else self.agc_pmin
                var upper = self.agc_clip * pclamp
                if upper > Scalar[DT](0.0):
                    var ratio = gnorm / upper
                    if ratio > Scalar[DT](1.0):
                        agc_scale = one / ratio

            # ── rms → momentum → lr ──
            for i in range(n):
                var g = grad.data[i] * agc_scale
                var nu_new = self.beta2 * v.data[i] + (one - self.beta2) * g * g
                v.data[i] = nu_new
                var nu_hat = nu_new / self.bc2
                var g_rms = g / (sqrt(nu_hat) + self.eps)
                var mu_new = self.beta1 * m.data[i] + (one - self.beta1) * g_rms
                m.data[i] = mu_new
                var mu_hat = mu_new / self.bc1
                param.data[i] = param.data[i] - self.lr * mu_hat
        else:
            var c = ctx.value()
            if not m.dev:  # first step: allocate + zero the moments
                m.ensure_gpu(c, n)
                m.dev.value().enqueue_fill(Scalar[DT](0))
                v.ensure_gpu(c, n)
                v.dev.value().enqueue_fill(Scalar[DT](0))
            if not self._scale.dev:
                self._scale = Tensor.alloc_gpu(c, 1)
            if not self._agc_partials.dev:
                self._agc_partials = Tensor.alloc_gpu(c, 2 * AGC_MAXB)
            comptime players = RowMajor[2 * AGC_MAXB]
            # Pass A: multi-block partial ‖grad‖²/‖param‖² → _agc_partials.
            c.enqueue_function[_agc_partials_kernel_rt](
                param.dev.value(),
                grad.dev.value(),
                Int64(n),
                self._agc_partials.tt["gpu", 2 * AGC_MAXB](),
                grid_dim=AGC_MAXB,
                block_dim=AGC_TPB,
            )
            # Pass A2: reduce partials → per-leaf AGC scale → _scale[0].
            c.enqueue_function[_agc_finalize_kernel](
                self._agc_partials.tt["gpu", 2 * AGC_MAXB](),
                self._scale.tt["gpu", 1](),
                self.agc_clip,
                self.agc_pmin,
                grid_dim=1,
                block_dim=AGC_MAXB,
            )
            # Pass B: rms → momentum → lr (grid over elements). Same stream →
            # ordered after Pass A, so _scale[0] is ready.
            var nblk = (n + TPB - 1) // TPB
            c.enqueue_function[_dreamer_update_kernel_rt](
                param.dev.value(),
                grad.dev.value(),
                m.dev.value(),
                v.dev.value(),
                Int64(n),
                self._scale.tt["gpu", 1](),
                self.lr,
                self.beta1,
                self.beta2,
                self.eps,
                self._pow_dev.tt["gpu", 2](),
                grid_dim=nblk,
                block_dim=TPB,
            )

    def visit[target: StaticString, N: Int](
        mut self,
        name: String,
        mut param: Tensor,
        mut grad: Tensor,
        mut m: Tensor,
        mut v: Tensor,
        apply_decay: Bool,
        ctx: Optional[DeviceContext],
    ) raises:
        self.visit_rt[target](name, param, grad, m, v, N, apply_decay, ctx)
