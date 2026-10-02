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
from max.gpu import thread_idx, block_idx
from max.gpu.primitives import block
from max.gpu.host import DeviceContext
from layout import Layout, LayoutTensor, TileTensor, row_major

from noeira.nn.constants import DT
from ..core.tensor import Tensor, TensorImpl
from .linear import _cast_f2b_kernel, _cast_b2f_kernel
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import Param, ParamVisitor
from ..core.state import State
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP


comptime BN_DEFAULT_EPS: Float64 = 1e-5
comptime BN_DEFAULT_MOM: Float64 = 0.1
comptime BN_TPB: Int = 128


# ── GPU kernels (verbatim from legacy; args MutAnyOrigin = GPU ABI) ─────
def _bn1d_forward_train_kernel[
    BATCH: Int,
    DIM: Int,
    EPSILON: Float64,
](
    input: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    gamma: LayoutTensor[DT, Layout.row_major(DIM), MutAnyOrigin],
    beta: LayoutTensor[DT, Layout.row_major(DIM), MutAnyOrigin],
    cache_mean: LayoutTensor[DT, Layout.row_major(DIM), MutAnyOrigin],
    cache_var: LayoutTensor[DT, Layout.row_major(DIM), MutAnyOrigin],
    cache_xhat: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    cache_inv_std: LayoutTensor[DT, Layout.row_major(DIM), MutAnyOrigin],
):
    var f = Int(block_idx.x)
    var t = Int(thread_idx.x)
    if f >= DIM:
        return
    var inv_n: Scalar[DT] = 1.0 / Scalar[DT](Float32(BATCH))
    var eps = Scalar[DT](EPSILON)
    var my_sum: input.element_type = 0.0
    var b = t
    while b < BATCH:
        my_sum += input[b, f]
        b += BN_TPB
    var mean = block.sum[block_size=BN_TPB, broadcast=True](val=my_sum) * inv_n
    var my_var: input.element_type = 0.0
    b = t
    while b < BATCH:
        var d = input[b, f] - mean
        my_var += d * d
        b += BN_TPB
    var var_ = block.sum[block_size=BN_TPB, broadcast=True](val=my_var) * inv_n
    var inv_std: input.element_type = 1.0 / sqrt(var_ + eps)
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
    cache_mean: LayoutTensor[DT, Layout.row_major(DIM), MutAnyOrigin],
    cache_var: LayoutTensor[DT, Layout.row_major(DIM), MutAnyOrigin],
    running_mean: LayoutTensor[DT, Layout.row_major(DIM), MutAnyOrigin],
    running_var: LayoutTensor[DT, Layout.row_major(DIM), MutAnyOrigin],
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
    input: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    output: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    gamma: LayoutTensor[DT, Layout.row_major(DIM), MutAnyOrigin],
    beta: LayoutTensor[DT, Layout.row_major(DIM), MutAnyOrigin],
    running_mean: LayoutTensor[DT, Layout.row_major(DIM), MutAnyOrigin],
    running_var: LayoutTensor[DT, Layout.row_major(DIM), MutAnyOrigin],
):
    var f = Int(block_idx.x)
    var t = Int(thread_idx.x)
    if f >= DIM:
        return
    var eps = Scalar[DT](EPSILON)
    var rm = running_mean[f]
    var rv = running_var[f]
    var inv_std: input.element_type = 1.0 / sqrt(rv + eps)
    var g = gamma[f]
    var bt = beta[f]
    var b = t
    while b < BATCH:
        var x = input[b, f]
        output[b, f] = g * (x - rm) * inv_std + bt
        b += BN_TPB


def _bn1d_backward_kernel[
    BATCH: Int,
    DIM: Int,
](
    grad_output: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    gamma: LayoutTensor[DT, Layout.row_major(DIM), MutAnyOrigin],
    cache_xhat: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    cache_inv_std: LayoutTensor[DT, Layout.row_major(DIM), MutAnyOrigin],
    grad_input: LayoutTensor[DT, Layout.row_major(BATCH, DIM), MutAnyOrigin],
    grad_gamma: LayoutTensor[DT, Layout.row_major(DIM), MutAnyOrigin],
    grad_beta: LayoutTensor[DT, Layout.row_major(DIM), MutAnyOrigin],
):
    var f = Int(block_idx.x)
    var t = Int(thread_idx.x)
    if f >= DIM:
        return
    var inv_n: Scalar[DT] = 1.0 / Scalar[DT](Float32(BATCH))
    var g = gamma[f]
    var inv_std = cache_inv_std[f]
    var my_sum_dxhat: grad_output.element_type = 0.0
    var my_sum_dxhat_xhat: grad_output.element_type = 0.0
    var my_dgamma: grad_output.element_type = 0.0
    var my_dbeta: grad_output.element_type = 0.0
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
    match its torch reference. EfficientZero v2 and REDQ-OFE port torch
    references too and still run the biased update."""
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
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B * Self.DIM_)
            comptime l2d = Layout.row_major(B, Self.DIM_)
            comptime ld = Layout.row_major(Self.DIM_)
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
                    in0.lt["gpu", l2d](),
                    out.lt["gpu", l2d](),
                    self.gamma.val.lt["gpu", ld](),
                    self.beta.val.lt["gpu", ld](),
                    self.cache_mean.lt["gpu", ld](),
                    self.cache_var.lt["gpu", ld](),
                    self.cache_xhat.lt["gpu", l2d](),
                    self.cache_inv_std.lt["gpu", ld](),
                    grid_dim=Self.DIM_,
                    block_dim=BN_TPB,
                )
                # Running-stat EMA in a dedicated kernel (see kernel docstring).
                c.enqueue_function[
                    _bn1d_update_running_kernel[
                        Self.DIM_, Self.MOMENTUM, Self._var_corr[B]()
                    ]
                ](
                    self.cache_mean.lt["gpu", ld](),
                    self.cache_var.lt["gpu", ld](),
                    self.running_mean.t.lt["gpu", ld](),
                    self.running_var.t.lt["gpu", ld](),
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
                    in0.lt["gpu", l2d](),
                    out.lt["gpu", l2d](),
                    self.gamma.val.lt["gpu", ld](),
                    self.beta.val.lt["gpu", ld](),
                    self.running_mean.t.lt["gpu", ld](),
                    self.running_var.t.lt["gpu", ld](),
                    grid_dim=Self.DIM_,
                    block_dim=BN_TPB,
                )

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
        if not self.cache_is_training:
            raise Error(
                "BatchNorm1D.vjp: training-mode cache not populated. Call"
                " forward with training=True before vjp."
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
            comptime l2d = Layout.row_major(B, Self.DIM_)
            comptime ld = Layout.row_major(Self.DIM_)
            c.enqueue_function[_bn1d_backward_kernel[B, Self.DIM_]](
                grad_output.lt["gpu", l2d](),
                self.gamma.val.lt["gpu", ld](),
                self.cache_xhat.lt["gpu", l2d](),
                self.cache_inv_std.lt["gpu", ld](),
                gin.lt["gpu", l2d](),
                self.gamma.grd.lt["gpu", ld](),
                self.beta.grd.lt["gpu", ld](),
                grid_dim=Self.DIM_,
                block_dim=BN_TPB,
            )

    # for_each_param / zero_grad inherit the Module reflection defaults
    # (core/walkers.mojo auto-discovers the Param fields).
