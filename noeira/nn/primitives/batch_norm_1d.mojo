"""BatchNorm1D[DIM, MOMENTUM, EPSILON] — per-feature batch norm (storage surface).

Transformed from legacy `nn.primitives.BatchNorm1D` (surface-only change; the
math, GPU kernels, finite-check, train/eval split, and cache discipline are
carried over verbatim). This is the first leaf with **running State**:

  - `gamma` / `beta` are `Param` (decay=False) — walked by `for_each_param`,
    optimized like any weight.
  - `running_mean` / `running_var` are plain owned `Tensor`s (the storage
    equivalent of legacy's decay-exempt `State`): they evolve ONLY via the
    forward EMA, are NOT visited by `for_each_param` (the optimizer never
    touches them), and will ride the checkpoint via a State walker in Stage 4.
  - `training: Bool` is a runtime field (default True), flipped via
    `set_training(...)`. Train caches x̂ + inv_std for backward; eval uses the
    running stats and leaves the cache untouched. `cache_is_training` guards
    against vjp-after-eval (a stale cache).

Backward (training cache):
    dx̂ = dy·γ ; m1 = mean_b(dx̂) ; m2 = mean_b(dx̂·x̂)
    dx = inv_std·(dx̂ - m1 - x̂·m2) ; dγ += Σ_b dy·x̂ ; dβ += Σ_b dy
"""

from std.math import sqrt
from max.gpu import thread_idx, block_idx, global_idx
from max.gpu.primitives import block
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from noeira.core.tiles import RowMajor, Tile

from noeira.nn.constants import DT
from ..core.polyak import polyak_tensor
from ..core.tensor import Tensor, TensorImpl
from .linear import _cast_f2b_kernel, _cast_b2f_kernel, enqueue_dot_bias_grad
from ..core.fill import fill_dev
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import Param, ParamVisitor
from ..core.state import State
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP


comptime BN_DEFAULT_EPS: Float64 = 1e-5
comptime BN_DEFAULT_MOM: Float64 = 0.1
comptime BN_TPB: Int = 128
comptime BN_TWO_PASS_MIN_ELEMS = 1 << 18
"""Smallest B·DIM whose train-mode backward takes the two-pass column sums
(a fill, `enqueue_dot_bias_grad`'s two passes and a coalesced dx pass: four
launches against the fused kernel's one). On an M1 Pro: 512 x 2048 4.1x
faster, 256 x 1024 (2^18) 1.4x, 256 x 512 0.83x
(`bench_param_grad_reductions.mojo`)."""


# ── GPU kernels (verbatim from legacy; args MutAnyOrigin = GPU ABI) ─────
def _bn1d_forward_train_kernel[
    BATCH: Int,
    DIM: Int,
    EPSILON: Float64,
](
    input: Tile[DT, BATCH, DIM],
    output: Tile[DT, BATCH, DIM],
    gamma: Tile[DT, DIM],
    beta: Tile[DT, DIM],
    cache_mean: Tile[DT, DIM],
    cache_var: Tile[DT, DIM],
    cache_xhat: Tile[DT, BATCH, DIM],
    cache_inv_std: Tile[DT, DIM],
):
    var f = Int(block_idx.x)
    var t = Int(thread_idx.x)
    if f >= DIM:
        return
    var inv_n: Scalar[DT] = 1.0 / Scalar[DT](Float32(BATCH))
    var eps = Scalar[DT](EPSILON)
    var my_sum: input.ElementType = 0.0
    var b = t
    while b < BATCH:
        my_sum += input[b, f]
        b += BN_TPB
    var mean = block.sum[block_size=BN_TPB, broadcast=True](val=my_sum) * inv_n
    var my_var: input.ElementType = 0.0
    b = t
    while b < BATCH:
        var d = input[b, f] - mean
        my_var += d * d
        b += BN_TPB
    var var_ = block.sum[block_size=BN_TPB, broadcast=True](val=my_var) * inv_n
    var inv_std: input.ElementType = 1.0 / sqrt(var_ + eps)
    # Caches only — NO running-stat write here. The EMA runs in its own kernel
    # (_bn1d_update_running_kernel); folding it in dropped the stores on NVIDIA
    # at large BATCH (the BatchNorm2D B>64 store-drop, same cause).
    if t == 0:
        cache_mean[f] = mean
        cache_var[f] = var_
        cache_inv_std[f] = inv_std
    var g = gamma[f]
    var bt = beta[f]
    b = t
    while b < BATCH:
        var x = input[b, f]
        var xh = (x - mean) * inv_std
        cache_xhat[b, f] = xh
        output[b, f] = g * xh + bt
        b += BN_TPB


def _bn1d_update_running_kernel[
    DIM: Int,
    MOMENTUM: Float64,
    VAR_CORR: Float64 = 1.0,
](
    cache_mean: Tile[DT, DIM],
    cache_var: Tile[DT, DIM],
    running_mean: Tile[DT, DIM],
    running_var: Tile[DT, DIM],
):
    """Running-stat EMA in its OWN kernel (one thread / feature). Mirrors the
    BatchNorm2D split: keeping the running_mean/running_var read-modify-write out
    of the reduction kernel keeps the NVIDIA backend from dropping the stores at
    large BATCH. Unconditional (matches the CPU path; no NaN guard)."""
    var f = Int(block_idx.x)
    if f >= DIM:
        return
    var mom = Scalar[DT](MOMENTUM)
    var one_m = Scalar[DT](1.0) - mom
    running_mean[f] = one_m * running_mean[f] + mom * cache_mean[f]
    running_var[f] = one_m * running_var[f] + mom * (
        Scalar[DT](VAR_CORR) * cache_var[f]
    )


def _bn1d_forward_eval_kernel[
    BATCH: Int,
    DIM: Int,
    EPSILON: Float64,
](
    input: Tile[DT, BATCH, DIM],
    output: Tile[DT, BATCH, DIM],
    gamma: Tile[DT, DIM],
    beta: Tile[DT, DIM],
    running_mean: Tile[DT, DIM],
    running_var: Tile[DT, DIM],
):
    var f = Int(block_idx.x)
    var t = Int(thread_idx.x)
    if f >= DIM:
        return
    var eps = Scalar[DT](EPSILON)
    var rm = running_mean[f]
    var rv = running_var[f]
    var inv_std: input.ElementType = 1.0 / sqrt(rv + eps)
    var g = gamma[f]
    var bt = beta[f]
    var b = t
    while b < BATCH:
        var x = input[b, f]
        output[b, f] = g * (x - rm) * inv_std + bt
        b += BN_TPB


def _bn1d_backward_eval_kernel[
    BATCH: Int,
    DIM: Int,
    EPSILON: Float64,
](
    grad_output: Tile[DT, BATCH, DIM],
    input: Tile[DT, BATCH, DIM],
    gamma: Tile[DT, DIM],
    running_mean: Tile[DT, DIM],
    running_var: Tile[DT, DIM],
    grad_input: Tile[DT, BATCH, DIM],
    grad_gamma: Tile[DT, DIM],
    grad_beta: Tile[DT, DIM],
):
    """Eval mode: y = γ·(x − μ_run)·inv + β is affine in x, so
    dx = dy·γ·inv, dγ += Σ dy·x̂, dβ += Σ dy (x̂ from the running stats)."""
    var f = Int(block_idx.x)
    var t = Int(thread_idx.x)
    if f >= DIM:
        return
    var inv_std = rebind[Scalar[DT]](1.0 / sqrt(running_var[f] + Scalar[DT](EPSILON)))
    var g = rebind[Scalar[DT]](gamma[f])
    var rm = rebind[Scalar[DT]](running_mean[f])
    var my_dgamma: Scalar[DT] = 0.0
    var my_dbeta: Scalar[DT] = 0.0
    var b = t
    while b < BATCH:
        var dy = rebind[Scalar[DT]](grad_output[b, f])
        var xh = (rebind[Scalar[DT]](input[b, f]) - rm) * inv_std
        my_dgamma += dy * xh
        my_dbeta += dy
        grad_input[b, f] = dy * g * inv_std
        b += BN_TPB
    var d_gamma_tot = block.sum[block_size=BN_TPB, broadcast=False](val=my_dgamma)
    var d_beta_tot = block.sum[block_size=BN_TPB, broadcast=False](val=my_dbeta)
    if t == 0:
        grad_gamma[f] = grad_gamma[f] + d_gamma_tot[0]
        grad_beta[f] = grad_beta[f] + d_beta_tot[0]


def _bn1d_backward_kernel[
    BATCH: Int,
    DIM: Int,
](
    grad_output: Tile[DT, BATCH, DIM],
    gamma: Tile[DT, DIM],
    cache_xhat: Tile[DT, BATCH, DIM],
    cache_inv_std: Tile[DT, DIM],
    grad_input: Tile[DT, BATCH, DIM],
    grad_gamma: Tile[DT, DIM],
    grad_beta: Tile[DT, DIM],
):
    var f = Int(block_idx.x)
    var t = Int(thread_idx.x)
    if f >= DIM:
        return
    var inv_n: Scalar[DT] = 1.0 / Scalar[DT](Float32(BATCH))
    var g = gamma[f]
    var inv_std = cache_inv_std[f]
    var my_sum_dxhat: grad_output.ElementType = 0.0
    var my_sum_dxhat_xhat: grad_output.ElementType = 0.0
    var my_dgamma: grad_output.ElementType = 0.0
    var my_dbeta: grad_output.ElementType = 0.0
    var b = t
    while b < BATCH:
        var dy = grad_output[b, f]
        var xh = cache_xhat[b, f]
        var dxhat = dy * g
        my_sum_dxhat += dxhat
        my_sum_dxhat_xhat += dxhat * xh
        my_dgamma += dy * xh
        my_dbeta += dy
        b += BN_TPB
    var sum_dxhat = block.sum[block_size=BN_TPB, broadcast=True](
        val=my_sum_dxhat
    )
    var sum_dxhat_xhat = block.sum[block_size=BN_TPB, broadcast=True](
        val=my_sum_dxhat_xhat
    )
    var d_gamma_tot = block.sum[block_size=BN_TPB, broadcast=False](
        val=my_dgamma
    )
    var d_beta_tot = block.sum[block_size=BN_TPB, broadcast=False](val=my_dbeta)
    if t == 0:
        grad_gamma[f] = grad_gamma[f] + d_gamma_tot[0]
        grad_beta[f] = grad_beta[f] + d_beta_tot[0]
    var m1 = sum_dxhat * inv_n
    var m2 = sum_dxhat_xhat * inv_n
    b = t
    while b < BATCH:
        var dy = grad_output[b, f]
        var xh = cache_xhat[b, f]
        var dxhat = dy * g
        grad_input[b, f] = inv_std * (dxhat - m1 - xh * m2)
        b += BN_TPB


def _bn1d_backward_dx_kernel[
    BATCH: Int,
    DIM: Int,
](
    grad_output: Tile[DT, BATCH, DIM],
    gamma: Tile[DT, DIM],
    cache_xhat: Tile[DT, BATCH, DIM],
    cache_inv_std: Tile[DT, DIM],
    sums: Tile[DT, 2 * DIM],
    grad_input: Tile[DT, BATCH, DIM],
    grad_gamma: Tile[DT, DIM],
    grad_beta: Tile[DT, DIM],
):
    """`_bn1d_backward_kernel`'s dx, one thread per element (coalesced), from
    the column sums `sums[f] = Σ dy·x̂` and `sums[DIM + f] = Σ dy` the
    two-pass helper wrote. Row 0 adds them to the γ / β grads. Σ dxhat and
    Σ dxhat·x̂ are γ times those sums."""
    var i = Int(global_idx.x)
    if i >= BATCH * DIM:
        return
    var b = i // DIM
    var f = i % DIM
    var inv_n: Scalar[DT] = 1.0 / Scalar[DT](Float32(BATCH))
    var g = rebind[Scalar[DT]](gamma[f])
    var s_gx = rebind[Scalar[DT]](sums[f])
    var s_g = rebind[Scalar[DT]](sums[DIM + f])
    var m1 = g * s_g * inv_n
    var m2 = g * s_gx * inv_n
    var dy = rebind[Scalar[DT]](grad_output[b, f])
    var xh = rebind[Scalar[DT]](cache_xhat[b, f])
    grad_input[b, f] = rebind[Scalar[DT]](cache_inv_std[f]) * (
        dy * g - m1 - xh * m2
    )
    if b == 0:
        grad_gamma[f] = rebind[Scalar[DT]](grad_gamma[f]) + s_gx
        grad_beta[f] = rebind[Scalar[DT]](grad_beta[f]) + s_g


struct BatchNorm1D[
    DIM_: Int,
    MOMENTUM: Float64 = BN_DEFAULT_MOM,
    EPSILON: Float64 = BN_DEFAULT_EPS,
    ADT: DType = DT,
    UNBIASED_RUNNING: Bool = False,
](Module):
    """`UNBIASED_RUNNING`: torch's `BatchNorm1d` normalises with the BIASED
    batch variance but feeds the UNBIASED one (x B/(B-1)) to `running_var`.
    False (the default, unchanged) keeps the biased update; LeWM sets True to
    match its torch reference. ⚠ EfficientZero v2 ports torch too and still
    runs the biased update — invisible today only because its drivers never
    put BN in eval mode (the reference does, for MCTS / reanalyze)."""
    comptime ARITY = 1
    comptime IN_DIMS = Array[Int, 1](fill=Self.DIM_)
    comptime OUT_DIM = Self.DIM_
    # Activation-flow dtype (AMP §3 fp32-INTERNAL): BN accepts/emits ACT_DT but
    # computes stats/normalize in fp32 internally. ACT_DT == DT (default) →
    # the cast wrappers collapse and the fp32 path is byte-identical.
    comptime ACT_DT = Self.ADT

    var gamma: Param["gamma", False, Self.DIM_]
    var beta: Param["beta", False, Self.DIM_]
    # Running stats (State): EMA-updated in forward, never optimized.
    var running_mean: State["running_mean", Self.DIM_]
    var running_var: State["running_var", Self.DIM_]
    # Training cache (owned storage — sound; not a back-pointer).
    var cache_xhat: Tensor  # [BATCH, DIM]
    var cache_inv_std: Tensor  # [DIM]
    var cache_mean: Tensor  # [DIM] (GPU: batch mean, fed to the running-stat EMA)
    var cache_var: Tensor  # [DIM] (GPU: batch var, fed to the running-stat EMA)
    var cache_is_training: Bool
    var training: Bool
    var bwd_sums: Tensor
    """[2·DIM] Σ dy·x̂ then Σ dy: the two-pass backward's column sums."""
    var bwd_part: Tensor
    """Chunk partials of those sums (`enqueue_dot_bias_grad`)."""

    @staticmethod
    def _var_corr[B: Int]() -> Float64:
        """Factor applied to the batch variance in the running update."""
        comptime if Self.UNBIASED_RUNNING:
            return Float64(B) / Float64(B - 1)
        return 1.0

    def __init__(out self):
        self.gamma = Param["gamma", False, Self.DIM_]()
        self.beta = Param["beta", False, Self.DIM_]()
        self.running_mean = State["running_mean", Self.DIM_]()
        self.running_var = State["running_var", Self.DIM_]()
        self.cache_xhat = Tensor()
        self.cache_inv_std = Tensor()
        self.cache_mean = Tensor()
        self.cache_var = Tensor()
        self.cache_is_training = False
        self.training = True
        self.bwd_sums = Tensor()
        self.bwd_part = Tensor()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        var bn = Self()
        bn.gamma = Param["gamma", False, Self.DIM_].make[target](ctx)
        bn.beta = Param["beta", False, Self.DIM_].make[target](ctx)
        for k in range(Self.DIM_):
            bn.gamma.val.data[k] = Scalar[DT](1.0)  # γ←1, β←0
        bn.running_mean = State["running_mean", Self.DIM_].make[target](ctx)  # ←0
        bn.running_var = State["running_var", Self.DIM_].make[target](ctx)
        for k in range(Self.DIM_):
            bn.running_var.t.data[k] = Scalar[DT](1.0)  # σ²_run←1
        comptime if target != "cpu":
            var dctx = ctx.value()
            bn.gamma.val.upload(dctx)
            bn.beta.val.upload(dctx)  # zeros → device
            bn.running_mean.t.upload(dctx)
            bn.running_var.t.upload(dctx)
        return bn^

    def set_training(mut self, v: Bool):
        self.training = v

    def set_attr[ATTR: StaticString](mut self, value: Scalar[DT]):
        """Named-attr hook so a parent combinator can toggle BN train/eval via
        `net.set_attr["training"](1.0/0.0)`. `value != 0` → training."""
        comptime if ATTR == "training":
            self.training = value != Scalar[DT](0.0)

    def polyak_from[
        target: StaticString
    ](
        mut self,
        mut src: Self,
        tau: Scalar[DT],
        ctx: Optional[DeviceContext],
    ) raises:
        """Soft-update toward `src` (target <- online). The `Module`
        default is a NO-OP, so without this override a target net containing
        this layer would keep its init values forever while the online copy
        trains (the `LayerNorm` / `Conv2D` bug class; audit 2026-10-07)."""
        polyak_tensor[target, Self.DIM_](self.gamma.val, src.gamma.val, tau, ctx)
        polyak_tensor[target, Self.DIM_](self.beta.val, src.beta.val, tau, ctx)
        # The running statistics follow too: with tau = 1 this is the hard sync
        # `hard_copy` does (params AND State).
        polyak_tensor[target, Self.DIM_](self.running_mean.t, src.running_mean.t, tau, ctx)
        polyak_tensor[target, Self.DIM_](self.running_var.t, src.running_var.t, tau, ctx)


    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[1, o, Self.ACT_DT],
        mut out: TensorImpl[Self.ACT_DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        # AMP §3 fp32-internal: ACT_DT==DT → bit-identical fp32 path; else cast
        # the bf16 activation in→fp32, run the fp32 BN, cast out→bf16.
        comptime if Self.ACT_DT == DT:
            ref in0d = rebind[Tensor](inputs[0])
            ref outd = rebind[Tensor](out)
            self._forward_f32[target, B](in0d, outd, ctx)
        else:
            comptime N = B * Self.DIM_
            # LOCAL fp32 scratch (not self-fields → no mut-self aliasing).
            var in_f32 = Tensor()
            in_f32.ensure[target](N, ctx)
            var out_f32 = Tensor()
            out_f32.ensure[target](N, ctx)
            out.ensure[target](N, ctx)
            ref in0 = inputs[0]
            comptime if target == "cpu":
                for i in range(N):
                    in_f32.data[i] = in0.data[i].cast[DT]()
            else:
                var c = ctx.value()
                c.enqueue_function[_cast_b2f_kernel](
                    in0.dev.value(),
                    in_f32.dev.value(),
                    Int64(N),
                    grid_dim=(N + 255) // 256,
                    block_dim=256,
                )
            self._forward_f32[target, B](in_f32, out_f32, ctx)
            comptime if target == "cpu":
                for i in range(N):
                    out.data[i] = out_f32.data[i].cast[Self.ACT_DT]()
            else:
                var c = ctx.value()
                c.enqueue_function[_cast_f2b_kernel](
                    out_f32.dev.value(),
                    out.dev.value(),
                    Int64(N),
                    grid_dim=(N + 255) // 256,
                    block_dim=256,
                )

    def _forward_f32[target: StaticString, B: Int](
        mut self,
        mut in0: Tensor,
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        var eps = Scalar[DT](Self.EPSILON)
        comptime if target == "cpu":
            out.ensure(B * Self.DIM_)
            var input = TileTensor(in0.data, row_major[B, Self.DIM_]())
            var output_v = TileTensor(out.data, row_major[B, Self.DIM_]())
            var gamma_v = TileTensor(
                self.gamma.val.data, row_major[Self.DIM_]()
            )
            var beta_v = TileTensor(self.beta.val.data, row_major[Self.DIM_]())
            var rm_v = TileTensor(
                self.running_mean.t.data, row_major[Self.DIM_]()
            )
            var rv_v = TileTensor(self.running_var.t.data, row_major[Self.DIM_]())
            if self.training:
                self.cache_xhat.ensure(B * Self.DIM_)
                self.cache_inv_std.ensure(Self.DIM_)
                var xhat_v = TileTensor(
                    self.cache_xhat.data, row_major[B, Self.DIM_]()
                )
                var inv_v = TileTensor(
                    self.cache_inv_std.data, row_major[Self.DIM_]()
                )
                var inv_n = Scalar[DT](1.0) / Scalar[DT](Float64(B))
                var mom = Scalar[DT](Self.MOMENTUM)
                var one_m = Scalar[DT](1.0) - mom
                for f in range(Self.DIM_):
                    var mean: Scalar[DT] = 0.0
                    for b in range(B):
                        mean += input[b, f]
                    mean *= inv_n
                    var var_: Scalar[DT] = 0.0
                    for b in range(B):
                        var diff = input[b, f] - mean
                        var_ += diff * diff
                    var_ *= inv_n
                    var inv_std = Scalar[DT](1.0) / sqrt(var_ + eps)
                    inv_v[f] = inv_std
                    var g = gamma_v[f]
                    var bt = beta_v[f]
                    for b in range(B):
                        var xh = (input[b, f] - mean) * inv_std
                        xhat_v[b, f] = xh
                        output_v[b, f] = g * xh + bt
                    rm_v[f] = one_m * rm_v[f] + mom * mean
                    rv_v[f] = one_m * rv_v[f] + mom * (
                        Scalar[DT](Self._var_corr[B]()) * var_
                    )
                self.cache_is_training = True
            else:
                for f in range(Self.DIM_):
                    var inv_std = Scalar[DT](1.0) / sqrt(rv_v[f] + eps)
                    var g = gamma_v[f]
                    var bt = beta_v[f]
                    var rm = rm_v[f]
                    for b in range(B):
                        output_v[b, f] = g * (input[b, f] - rm) * inv_std + bt
                self.cache_is_training = False
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B * Self.DIM_)
            comptime l2d = RowMajor[B, Self.DIM_]
            comptime ld = RowMajor[Self.DIM_]
            if self.training:
                self.cache_xhat.ensure_gpu(c, B * Self.DIM_)
                self.cache_inv_std.ensure_gpu(c, Self.DIM_)
                self.cache_mean.ensure_gpu(c, Self.DIM_)
                self.cache_var.ensure_gpu(c, Self.DIM_)
                c.enqueue_function[
                    _bn1d_forward_train_kernel[
                        B,
                        Self.DIM_,
                        Self.EPSILON,
                    ]
                ](
                    in0.tt["gpu", B, Self.DIM_](),
                    out.tt["gpu", B, Self.DIM_](),
                    self.gamma.val.tt["gpu", Self.DIM_](),
                    self.beta.val.tt["gpu", Self.DIM_](),
                    self.cache_mean.tt["gpu", Self.DIM_](),
                    self.cache_var.tt["gpu", Self.DIM_](),
                    self.cache_xhat.tt["gpu", B, Self.DIM_](),
                    self.cache_inv_std.tt["gpu", Self.DIM_](),
                    grid_dim=Self.DIM_,
                    block_dim=BN_TPB,
                )
                # Running-stat EMA in a dedicated kernel (see kernel docstring).
                c.enqueue_function[
                    _bn1d_update_running_kernel[
                        Self.DIM_, Self.MOMENTUM, Self._var_corr[B]()
                    ]
                ](
                    self.cache_mean.tt["gpu", Self.DIM_](),
                    self.cache_var.tt["gpu", Self.DIM_](),
                    self.running_mean.t.tt["gpu", Self.DIM_](),
                    self.running_var.t.tt["gpu", Self.DIM_](),
                    grid_dim=Self.DIM_,
                    block_dim=1,
                )
                self.cache_is_training = True
            else:
                c.enqueue_function[
                    _bn1d_forward_eval_kernel[
                        B,
                        Self.DIM_,
                        Self.EPSILON,
                    ]
                ](
                    in0.tt["gpu", B, Self.DIM_](),
                    out.tt["gpu", B, Self.DIM_](),
                    self.gamma.val.tt["gpu", Self.DIM_](),
                    self.beta.val.tt["gpu", Self.DIM_](),
                    self.running_mean.t.tt["gpu", Self.DIM_](),
                    self.running_var.t.tt["gpu", Self.DIM_](),
                    grid_dim=Self.DIM_,
                    block_dim=BN_TPB,
                )
                self.cache_is_training = False

    def vjp[
        target: StaticString,
        B: Int,
        ofi: MutOrigin,
        ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[1, ofi, Self.ACT_DT],
        mut grad_output: TensorImpl[Self.ACT_DT],
        grad_inputs: TensorRefs[1, ogi, Self.ACT_DT],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        # The vjp follows the LAST forward's mode. ⚠ The flag used to be set
        # by a training forward and never cleared, so an eval forward + vjp
        # silently reused that training forward's caches; an eval forward now
        # clears it and the eval vjp below runs instead (it used to raise
        # only before ANY training forward).
        if not self.cache_is_training:
            comptime if Self.ACT_DT == DT:
                ref xin = rebind[Tensor](forward_input[0])
                ref god = rebind[Tensor](grad_output)
                ref gind = rebind[Tensor](grad_inputs[0])
                self._vjp_eval[target, B](xin, god, gind, ctx)
                return
            else:
                raise Error(
                    "BatchNorm1D.vjp: eval-mode backward is fp32-only (the"
                    " last forward ran in eval mode)"
                )
        # AMP §3 fp32-internal (forward_input unused, as in the fp32 body).
        comptime if Self.ACT_DT == DT:
            ref god = rebind[Tensor](grad_output)
            ref gind = rebind[Tensor](grad_inputs[0])
            self._vjp_f32[target, B](god, gind, ctx)
        else:
            comptime N = B * Self.DIM_
            # LOCAL fp32 scratch (not self-fields → no mut-self aliasing).
            var go_f32 = Tensor()
            go_f32.ensure[target](N, ctx)
            var gin_f32 = Tensor()
            gin_f32.ensure[target](N, ctx)
            ref gin = grad_inputs[0]
            gin.ensure[target](N, ctx)
            comptime if target == "cpu":
                for i in range(N):
                    go_f32.data[i] = grad_output.data[i].cast[DT]()
            else:
                var c = ctx.value()
                c.enqueue_function[_cast_b2f_kernel](
                    grad_output.dev.value(),
                    go_f32.dev.value(),
                    Int64(N),
                    grid_dim=(N + 255) // 256,
                    block_dim=256,
                )
            self._vjp_f32[target, B](go_f32, gin_f32, ctx)
            comptime if target == "cpu":
                for i in range(N):
                    gin.data[i] = gin_f32.data[i].cast[Self.ACT_DT]()
            else:
                var c = ctx.value()
                c.enqueue_function[_cast_f2b_kernel](
                    gin_f32.dev.value(),
                    gin.dev.value(),
                    Int64(N),
                    grid_dim=(N + 255) // 256,
                    block_dim=256,
                )

    def _vjp_f32[target: StaticString, B: Int](
        mut self,
        mut grad_output: Tensor,
        mut gin: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime if target == "cpu":
            gin.ensure(B * Self.DIM_)
            var go_v = TileTensor(grad_output.data, row_major[B, Self.DIM_]())
            var gi_v = TileTensor(gin.data, row_major[B, Self.DIM_]())
            var gamma_v = TileTensor(
                self.gamma.val.data, row_major[Self.DIM_]()
            )
            var dgamma_v = TileTensor(
                self.gamma.grd.data, row_major[Self.DIM_]()
            )
            var dbeta_v = TileTensor(self.beta.grd.data, row_major[Self.DIM_]())
            var xhat_v = TileTensor(
                self.cache_xhat.data, row_major[B, Self.DIM_]()
            )
            var inv_v = TileTensor(
                self.cache_inv_std.data, row_major[Self.DIM_]()
            )
            var inv_n = Scalar[DT](1.0) / Scalar[DT](Float64(B))
            for f in range(Self.DIM_):
                var g = gamma_v[f]
                var inv_std = inv_v[f]
                var sum_dxhat: Scalar[DT] = 0.0
                var sum_dxhat_xhat: Scalar[DT] = 0.0
                var d_gamma: Scalar[DT] = 0.0
                var d_beta: Scalar[DT] = 0.0
                for b in range(B):
                    var dy = go_v[b, f]
                    var xh = xhat_v[b, f]
                    var dxhat = dy * g
                    sum_dxhat += dxhat
                    sum_dxhat_xhat += dxhat * xh
                    d_gamma += dy * xh
                    d_beta += dy
                var m1 = sum_dxhat * inv_n
                var m2 = sum_dxhat_xhat * inv_n
                for b in range(B):
                    var dy = go_v[b, f]
                    var xh = xhat_v[b, f]
                    var dxhat = dy * g
                    gi_v[b, f] = inv_std * (dxhat - m1 - xh * m2)
                dgamma_v[f] += d_gamma
                dbeta_v[f] += d_beta
        else:
            var c = ctx.value()
            gin.ensure_gpu(c, B * Self.DIM_)
            comptime l2d = RowMajor[B, Self.DIM_]
            comptime ld = RowMajor[Self.DIM_]
            comptime if B * Self.DIM_ >= BN_TWO_PASS_MIN_ELEMS:
                # One block per feature walks the rows twice, 32 different
                # rows per warp load: 574 us at LeWM's projector (512 x
                # 2048) on an M1 Pro, 141 us as column sums + a dx pass.
                comptime D = Self.DIM_
                self.bwd_sums.ensure_gpu(c, 2 * D)
                fill_dev(self.bwd_sums.dev.value(), 2 * D, Scalar[DT](0), c)
                var s_dy = self.bwd_sums.dev.value().create_sub_buffer[DT](D, D)
                enqueue_dot_bias_grad[DT](
                    c, grad_output.dev.value(), self.cache_xhat.dev.value(),
                    self.bwd_sums.dev.value(), s_dy, B, D, self.bwd_part,
                )
                c.enqueue_function[_bn1d_backward_dx_kernel[B, D]](
                    grad_output.tt["gpu", B, D](),
                    self.gamma.val.tt["gpu", D](),
                    self.cache_xhat.tt["gpu", B, D](),
                    self.cache_inv_std.tt["gpu", D](),
                    self.bwd_sums.tt["gpu", 2 * D](),
                    gin.tt["gpu", B, D](),
                    self.gamma.grd.tt["gpu", D](),
                    self.beta.grd.tt["gpu", D](),
                    grid_dim=(B * D + 255) // 256,
                    block_dim=256,
                )
            else:
                c.enqueue_function[_bn1d_backward_kernel[B, Self.DIM_]](
                    grad_output.tt["gpu", B, Self.DIM_](),
                    self.gamma.val.tt["gpu", Self.DIM_](),
                    self.cache_xhat.tt["gpu", B, Self.DIM_](),
                    self.cache_inv_std.tt["gpu", Self.DIM_](),
                    gin.tt["gpu", B, Self.DIM_](),
                    self.gamma.grd.tt["gpu", Self.DIM_](),
                    self.beta.grd.tt["gpu", Self.DIM_](),
                    grid_dim=Self.DIM_,
                    block_dim=BN_TPB,
                )

    def _vjp_eval[target: StaticString, B: Int](
        mut self,
        mut x: Tensor,
        mut grad_output: Tensor,
        mut gin: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        """Eval-mode backward (the forward was the affine map with the
        running statistics): dx = dy·γ·inv, dγ += Σ dy·x̂, dβ += Σ dy. Used to
        adapt a model at test time with its BatchNorms frozen in eval mode."""
        var eps = Scalar[DT](Self.EPSILON)
        comptime if target == "cpu":
            gin.ensure(B * Self.DIM_)
            for f in range(Self.DIM_):
                var inv_std = Scalar[DT](1.0) / sqrt(self.running_var.t.data[f] + eps)
                var g = self.gamma.val.data[f]
                var rm = self.running_mean.t.data[f]
                var dg: Scalar[DT] = 0.0
                var db: Scalar[DT] = 0.0
                for b in range(B):
                    var dy = grad_output.data[b * Self.DIM_ + f]
                    dg += dy * (x.data[b * Self.DIM_ + f] - rm) * inv_std
                    db += dy
                    gin.data[b * Self.DIM_ + f] = dy * g * inv_std
                self.gamma.grd.data[f] += dg
                self.beta.grd.data[f] += db
        else:
            var c = ctx.value()
            gin.ensure_gpu(c, B * Self.DIM_)
            comptime l2d = RowMajor[B, Self.DIM_]
            comptime ld = RowMajor[Self.DIM_]
            c.enqueue_function[_bn1d_backward_eval_kernel[B, Self.DIM_, Self.EPSILON]](
                grad_output.tt["gpu", B, Self.DIM_](),
                x.tt["gpu", B, Self.DIM_](),
                self.gamma.val.tt["gpu", Self.DIM_](),
                self.running_mean.t.tt["gpu", Self.DIM_](),
                self.running_var.t.tt["gpu", Self.DIM_](),
                gin.tt["gpu", B, Self.DIM_](),
                self.gamma.grd.tt["gpu", Self.DIM_](),
                self.beta.grd.tt["gpu", Self.DIM_](),
                grid_dim=Self.DIM_,
                block_dim=BN_TPB,
            )

    # for_each_param / zero_grad inherit the Module reflection defaults
    # (core/walkers.mojo auto-discovers the Param fields).
