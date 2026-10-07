"""NoisyLinear[IN, OUT] — Factorized Gaussian Noisy Linear (Fortunato 2018).

Transformed from legacy `nn.primitives.NoisyLinear` (surface-only change).
4 Params (µ_W, σ_W decay=True; µ_b, σ_b decay=False); the factorized noise
ε_in [IN] / ε_out [OUT] and the materialized W_eff / b_eff are leaf-owned
`Tensor` scratch (sampled fresh each forward). `noise_scale` (1.0 train / 0.0
eval-greedy) scales ε_out. The CPU host Box-Muller sampler + the GPU Philox path
are carried over verbatim — and CLEANLY: the GPU path calls the shared
`_box_muller_kernel_dev` (a LayoutTensor kernel) directly via `lt_gpu` views, so
there is NO raw `dev_ptr`/`unsafe_ptr`/`rebind` (the box-muller raw-ptr wrapper
is bypassed). The device Philox offset is a `TensorImpl[uint64]` (its `lt_gpu`
yields the offset `LayoutTensor` the kernel wants).

Backward: grad_µ_b = Σ_b go ; grad_σ_b = grad_µ_b · ε_out ; dW = xᵀ @ go ;
    grad_µ_w += dW ; grad_σ_w += dW · ε_in · ε_out ; grad_x = go @ W_effᵀ.
"""

from noeira.nn.core.mm import mm, bmm
from std.math import sqrt as fsqrt, log as flog, cos as fcos, pi
from std.random import random_float64
from max.gpu import global_idx, thread_idx, block_idx
from max.gpu.host import DeviceContext
from max.gpu.primitives import block
from layout import Layout, LayoutTensor, TileTensor, row_major
from linalg.matmul import matmul as max_matmul
from noeira.nn.core.splitk_gemm import (
    splitk_path_applies,
    decide_partitions,
    dispatch_splitk_gemm,
)

from noeira.nn.constants import DT, TPB
from noeira.nn.random.box_muller import (
    _box_muller_kernel_dev,
    advance_rng_offset_kernel,
)
from ..core.tensor import Tensor, TensorImpl
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import Param, ParamVisitor
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP
from ..core.polyak import polyak_tensor
from .linear import _transpose_tiled_kernel, _accum_kernel, _T_TILE, _T_BR


# ── CPU host sampler: f(z) = sign(z)·√|z|, z ~ N(0,1) via Box-Muller ────
def _fnoise(x: Scalar[DT]) -> Scalar[DT]:
    if x >= Scalar[DT](0.0):
        return fsqrt(x)
    return -fsqrt(-x)


def _sample_factorized_noise(mut buf: List[Scalar[DT]], n: Int):
    """Fill buf[0:n] with f(z), z ~ N(0,1) (Box-Muller, both branches)."""
    var k = 0
    while k + 1 < n:
        var u1 = random_float64()
        var u2 = random_float64()
        if u1 < 1e-12:
            u1 = 1e-12
        var r = fsqrt(Float64(-2.0) * flog(u1))
        var theta = Float64(2.0) * pi * u2
        var z0 = Scalar[DT](r * fcos(theta))
        var z1 = Scalar[DT](r * fcos(theta + 0.5 * pi))
        buf[k] = _fnoise(z0)
        buf[k + 1] = _fnoise(z1)
        k += 2
    if k < n:
        var u1 = random_float64()
        var u2 = random_float64()
        if u1 < 1e-12:
            u1 = 1e-12
        var r = fsqrt(Float64(-2.0) * flog(u1))
        var theta = Float64(2.0) * pi * u2
        var z0 = Scalar[DT](r * fcos(theta))
        buf[k] = _fnoise(z0)


# ── GPU kernels (verbatim from legacy; args MutAnyOrigin = GPU ABI) ─────
def _apply_f_noise_kernel[
    N: Int
](noise: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin]):
    var idx = Int(global_idx.x)
    if idx < N:
        var x = rebind[Scalar[DT]](noise[idx])
        if x >= Scalar[DT](0.0):
            noise[idx] = fsqrt(x)
        else:
            noise[idx] = -fsqrt(-x)


def _scale_inplace_kernel[
    N: Int
](buf: LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin], s: Scalar[DT]):
    var idx = Int(global_idx.x)
    if idx < N:
        buf[idx] = rebind[Scalar[DT]](buf[idx]) * s


def _materialize_w_eff_kernel[
    IN: Int, OUT: Int
](
    mu_w: LayoutTensor[DT, Layout.row_major(IN, OUT), MutAnyOrigin],
    sigma_w: LayoutTensor[DT, Layout.row_major(IN, OUT), MutAnyOrigin],
    n_in: LayoutTensor[DT, Layout.row_major(IN), MutAnyOrigin],
    n_out: LayoutTensor[DT, Layout.row_major(OUT), MutAnyOrigin],
    w_eff: LayoutTensor[DT, Layout.row_major(IN, OUT), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx < IN * OUT:
        var i = idx // OUT
        var j = idx % OUT
        w_eff[i, j] = rebind[Scalar[DT]](mu_w[i, j]) + rebind[Scalar[DT]](
            sigma_w[i, j]
        ) * rebind[Scalar[DT]](n_in[i]) * rebind[Scalar[DT]](n_out[j])


def _materialize_b_eff_kernel[
    OUT: Int
](
    mu_b: LayoutTensor[DT, Layout.row_major(OUT), MutAnyOrigin],
    sigma_b: LayoutTensor[DT, Layout.row_major(OUT), MutAnyOrigin],
    n_out: LayoutTensor[DT, Layout.row_major(OUT), MutAnyOrigin],
    b_eff: LayoutTensor[DT, Layout.row_major(OUT), MutAnyOrigin],
):
    var j = Int(global_idx.x)
    if j < OUT:
        b_eff[j] = rebind[Scalar[DT]](mu_b[j]) + rebind[Scalar[DT]](
            sigma_b[j]
        ) * rebind[Scalar[DT]](n_out[j])


def _noisy_bias_add_kernel[
    BATCH: Int, OUT: Int
](
    output: LayoutTensor[DT, Layout.row_major(BATCH, OUT), MutAnyOrigin],
    b_eff: LayoutTensor[DT, Layout.row_major(OUT), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx < BATCH * OUT:
        var b = idx // OUT
        var j = idx % OUT
        output[b, j] = rebind[Scalar[DT]](output[b, j]) + rebind[Scalar[DT]](
            b_eff[j]
        )


def _grad_b_pair_reduce_kernel[
    BATCH: Int, OUT: Int
](
    grad_output: LayoutTensor[DT, Layout.row_major(BATCH, OUT), MutAnyOrigin],
    n_out: LayoutTensor[DT, Layout.row_major(OUT), MutAnyOrigin],
    grad_mu_b: LayoutTensor[DT, Layout.row_major(OUT), MutAnyOrigin],
    grad_sigma_b: LayoutTensor[DT, Layout.row_major(OUT), MutAnyOrigin],
):
    var col = Int(block_idx.x)
    var t = Int(thread_idx.x)
    if col >= OUT:
        return
    var my_s: Scalar[DT] = 0.0
    var bi = t
    while bi < BATCH:
        my_s += rebind[Scalar[DT]](grad_output[bi, col])
        bi += TPB
    var total = block.sum[block_size=TPB, broadcast=False](val=my_s)
    if t == 0:
        var s = total[0]
        grad_mu_b[col] = rebind[Scalar[DT]](grad_mu_b[col]) + s
        grad_sigma_b[col] = rebind[Scalar[DT]](grad_sigma_b[col]) + s * rebind[
            Scalar[DT]
        ](n_out[col])


def _scaled_accum_factorized_kernel[
    IN: Int, OUT: Int
](
    dst: LayoutTensor[DT, Layout.row_major(IN, OUT), MutAnyOrigin],
    src: LayoutTensor[DT, Layout.row_major(IN, OUT), MutAnyOrigin],
    n_in: LayoutTensor[DT, Layout.row_major(IN), MutAnyOrigin],
    n_out: LayoutTensor[DT, Layout.row_major(OUT), MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx < IN * OUT:
        var i = idx // OUT
        var j = idx % OUT
        dst[i, j] = rebind[Scalar[DT]](dst[i, j]) + rebind[Scalar[DT]](
            src[i, j]
        ) * rebind[Scalar[DT]](n_in[i]) * rebind[Scalar[DT]](n_out[j])


struct NoisyLinear[IN_: Int, OUT_: Int](Module):
    comptime ARITY = 1
    comptime IN_DIMS = Array[Int, 1](fill=Self.IN_)
    comptime OUT_DIM = Self.OUT_
    comptime W_SIZE = Self.IN_ * Self.OUT_
    comptime B_SIZE = Self.OUT_
    comptime SIGMA0 = Scalar[DT](0.5)

    var mu_w: Param["mu_w", True, Self.W_SIZE]
    var sigma_w: Param["sigma_w", True, Self.W_SIZE]
    var mu_b: Param["mu_b", False, Self.B_SIZE]
    var sigma_b: Param["sigma_b", False, Self.B_SIZE]
    var noise_in: Tensor  # [IN]  (f-transformed)
    var noise_out: Tensor  # [OUT] (f-transformed, scaled)
    var w_eff: Tensor  # [IN*OUT]
    var b_eff: Tensor  # [OUT]
    var noise_scale: Scalar[DT]
    var noise_seed: UInt64
    var noise_offset: TensorImpl[DType.uint64]  # GPU Philox offset (1 elem)
    var cacheT: Tensor  # GPU grad_w: xᵀ [IN, BATCH] (lazy)
    var dW_tmp: Tensor  # GPU grad_w: dW [IN, OUT]
    # Split-K reduction workspace for the dW GEMM, `[P, IN_, OUT_]`, plus the
    # cached partition count (-1 = undecided, 1 = do not split). Owned here so
    # the GEMM does not hit `linalg.matmul`'s per-call cuMemAlloc/cuMemFree —
    # which is also what makes a step containing it capturable. See
    # `Linear.sk_ws` for the full rationale.
    #
    # ⚠ Expect this to be inert in the DQN / C51 configs that use NoisyLinear:
    # K is the minibatch and `select_config` needs K >= 2048 before it splits
    # at all, so `decide_partitions` returns 1 at the usual 32-512. It is here
    # so a large-batch run gets the same treatment as every other 2-D dW,
    # rather than silently falling back onto the allocating path.
    var sk_ws: Tensor
    var _sk_p: Int

    def __init__(out self):
        self.mu_w = Param["mu_w", True, Self.W_SIZE]()
        self.sigma_w = Param["sigma_w", True, Self.W_SIZE]()
        self.mu_b = Param["mu_b", False, Self.B_SIZE]()
        self.sigma_b = Param["sigma_b", False, Self.B_SIZE]()
        self.noise_in = Tensor()
        self.noise_out = Tensor()
        self.w_eff = Tensor()
        self.b_eff = Tensor()
        self.noise_scale = Scalar[DT](1.0)
        self.noise_seed = UInt64(1)
        self.noise_offset = TensorImpl[DType.uint64]()
        self.cacheT = Tensor()
        self.dW_tmp = Tensor()
        self.sk_ws = Tensor()
        self._sk_p = -1

    @staticmethod
    def _init_sigma(mut self):
        # σ init is the NoisyNet factorized prescription σ0/√IN — algorithm-
        # specified (NOT a general initializer), so it stays hardcoded. The µ
        # (mean) weights/biases honor INIT in `make`; the old `(k%7-3)*0.1`
        # placeholder that IGNORED INIT was the bug (`Deterministic` reproduces
        # that exact pattern, so the parity gate is unchanged).
        var sigma_init = Self.SIGMA0 / Scalar[DT](fsqrt(Float64(Self.IN_)))
        for k in range(Self.W_SIZE):
            self.sigma_w.val.data[k] = sigma_init
        for k in range(Self.B_SIZE):
            self.sigma_b.val.data[k] = sigma_init

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        var nl = Self()
        nl.mu_w = Param["mu_w", True, Self.W_SIZE].make[target](ctx)
        nl.sigma_w = Param["sigma_w", True, Self.W_SIZE].make[target](ctx)
        nl.mu_b = Param["mu_b", False, Self.B_SIZE].make[target](ctx)
        nl.sigma_b = Param["sigma_b", False, Self.B_SIZE].make[target](ctx)
        # µ (mean) params honor INIT; σ stays the NoisyNet prescription.
        INIT.init_weight[target](
            nl.mu_w.val, Self.W_SIZE, Self.IN_, Self.OUT_, ctx
        )
        INIT.init_bias[target](nl.mu_b.val, Self.B_SIZE, ctx)
        Self._init_sigma(nl)
        comptime if target == "cpu":
            nl.noise_in = Tensor.alloc(Self.IN_)
            nl.noise_out = Tensor.alloc(Self.OUT_)
            nl.w_eff = Tensor.alloc(Self.W_SIZE)
            nl.b_eff = Tensor.alloc(Self.B_SIZE)
        else:
            var dctx = ctx.value()
            # mu_w/mu_b uploaded by init_weight/init_bias; σ set host-side here.
            nl.sigma_w.val.upload(dctx)
            nl.sigma_b.val.upload(dctx)
            nl.noise_in.ensure_gpu(dctx, Self.IN_)
            nl.noise_out.ensure_gpu(dctx, Self.OUT_)
            nl.w_eff.ensure_gpu(dctx, Self.W_SIZE)
            nl.b_eff.ensure_gpu(dctx, Self.B_SIZE)
            nl.dW_tmp.ensure_gpu(dctx, Self.W_SIZE)
            nl.noise_offset.ensure_gpu(dctx, 1)
            nl.noise_offset.dev.value().enqueue_fill(UInt64(0))
        return nl^

    def set_noise_seed(mut self, seed: UInt64) raises:
        self.noise_seed = seed
        if self.noise_offset.dev:
            self.noise_offset.dev.value().enqueue_fill(UInt64(0))

    def set_noise_scale(mut self, v: Scalar[DT]):
        self.noise_scale = v

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[1, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref in0 = inputs[0]
        comptime if target == "cpu":
            out.ensure(B * Self.OUT_)
            # 1. sample factorized noise; scale ε_out.
            _sample_factorized_noise(self.noise_in.data, Self.IN_)
            _sample_factorized_noise(self.noise_out.data, Self.OUT_)
            for j in range(Self.OUT_):
                self.noise_out.data[j] = (
                    self.noise_out.data[j] * self.noise_scale
                )
            # 2. materialize W_eff, b_eff.
            for i in range(Self.IN_):
                var ni = self.noise_in.data[i]
                for j in range(Self.OUT_):
                    var idx = i * Self.OUT_ + j
                    self.w_eff.data[idx] = (
                        self.mu_w.val.data[idx]
                        + self.sigma_w.val.data[idx]
                        * ni
                        * self.noise_out.data[j]
                    )
            for j in range(Self.OUT_):
                self.b_eff.data[j] = (
                    self.mu_b.val.data[j]
                    + self.sigma_b.val.data[j] * self.noise_out.data[j]
                )
            # 3. out = x @ W_eff + b_eff.
            var x_v = TileTensor(in0.data, row_major[B, Self.IN_]())
            var w_v = TileTensor(
                self.w_eff.data, row_major[Self.IN_, Self.OUT_]()
            )
            var out_v = TileTensor(out.data, row_major[B, Self.OUT_]())
            max_matmul[target="cpu"](out_v, x_v, w_v, None)
            for b in range(B):
                for j in range(Self.OUT_):
                    out.data[b * Self.OUT_ + j] += self.b_eff.data[j]
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B * Self.OUT_)
            comptime lin = Layout.row_major(Self.IN_)
            comptime lout = Layout.row_major(Self.OUT_)
            comptime lw = Layout.row_major(Self.IN_, Self.OUT_)
            comptime loff = Layout.row_major(1)
            # 1. Philox box-muller into ε_in / ε_out via the shared LT kernel
            #    (no raw ptr — lt_gpu views + the uint64 offset Tensor's lt_gpu).
            comptime nb_in = (Self.IN_ + TPB - 1) // TPB
            comptime nb_out = (Self.OUT_ + TPB - 1) // TPB
            c.enqueue_function[_box_muller_kernel_dev[Self.IN_]](
                self.noise_in.lt["gpu", lin](),
                self.noise_seed,
                self.noise_offset.lt["gpu", loff](),
                grid_dim=nb_in,
                block_dim=TPB,
            )
            c.enqueue_function[
                advance_rng_offset_kernel[((Self.IN_ + 1) // 2) * 2]
            ](self.noise_offset.lt["gpu", loff](), grid_dim=1, block_dim=1)
            c.enqueue_function[_box_muller_kernel_dev[Self.OUT_]](
                self.noise_out.lt["gpu", lout](),
                self.noise_seed,
                self.noise_offset.lt["gpu", loff](),
                grid_dim=nb_out,
                block_dim=TPB,
            )
            c.enqueue_function[
                advance_rng_offset_kernel[((Self.OUT_ + 1) // 2) * 2]
            ](self.noise_offset.lt["gpu", loff](), grid_dim=1, block_dim=1)
            c.enqueue_function[_apply_f_noise_kernel[Self.IN_]](
                self.noise_in.lt["gpu", lin](), grid_dim=nb_in, block_dim=TPB
            )
            c.enqueue_function[_apply_f_noise_kernel[Self.OUT_]](
                self.noise_out.lt["gpu", lout](), grid_dim=nb_out, block_dim=TPB
            )
            c.enqueue_function[_scale_inplace_kernel[Self.OUT_]](
                self.noise_out.lt["gpu", lout](),
                self.noise_scale,
                grid_dim=nb_out,
                block_dim=TPB,
            )
            # 2. materialize W_eff, b_eff.
            comptime nb_w = (Self.W_SIZE + TPB - 1) // TPB
            c.enqueue_function[_materialize_w_eff_kernel[Self.IN_, Self.OUT_]](
                self.mu_w.val.lt["gpu", lw](),
                self.sigma_w.val.lt["gpu", lw](),
                self.noise_in.lt["gpu", lin](),
                self.noise_out.lt["gpu", lout](),
                self.w_eff.lt["gpu", lw](),
                grid_dim=nb_w,
                block_dim=TPB,
            )
            c.enqueue_function[_materialize_b_eff_kernel[Self.OUT_]](
                self.mu_b.val.lt["gpu", lout](),
                self.sigma_b.val.lt["gpu", lout](),
                self.noise_out.lt["gpu", lout](),
                self.b_eff.lt["gpu", lout](),
                grid_dim=nb_out,
                block_dim=TPB,
            )
            # 3. out = x @ W_eff + b_eff.
            mm[A0=B, A1=Self.IN_, B0=Self.IN_, B1=Self.OUT_, O0=B, O1=Self.OUT_](
                out.dev.value(), in0.dev.value(), self.w_eff.dev.value(), c
            )
            comptime nb_ba = (B * Self.OUT_ + TPB - 1) // TPB
            c.enqueue_function[_noisy_bias_add_kernel[B, Self.OUT_]](
                out.lt["gpu", Layout.row_major(B, Self.OUT_)](),
                self.b_eff.lt["gpu", lout](),
                grid_dim=nb_ba,
                block_dim=TPB,
            )

    def _decide_sk_p(mut self, B: Int, ctx: DeviceContext) raises:
        """Decide the dW GEMM's partition count, once, and cache it.

        `[IN_, B] @ [B, OUT_]`, so K is the row count. Decided on the first
        backward — an EAGER step, before any capture — and never again: P sets
        `grid_dim`, which is baked into a captured graph.
        """
        self._sk_p = decide_partitions(Self.IN_, Self.OUT_, B, ctx)
        if self._sk_p > 1:
            self.sk_ws.ensure_gpu(ctx, self._sk_p * Self.W_SIZE)

    def vjp[
        target: StaticString,
        B: Int,
        ofi: MutOrigin,
        ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[1, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[1, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref fin = forward_input[0]
        ref gin = grad_inputs[0]
        comptime if target == "cpu":
            gin.ensure(B * Self.IN_)
            # grad_b
            for j in range(Self.OUT_):
                var sb: Scalar[DT] = 0.0
                for b in range(B):
                    sb += grad_output.data[b * Self.OUT_ + j]
                self.mu_b.grd.data[j] += sb
                self.sigma_b.grd.data[j] += sb * self.noise_out.data[j]
            # grad_w: dW = xᵀ @ go ; grad_µ_w += dW ; grad_σ_w += dW·ε_in·ε_out
            var cT = List[Scalar[DT]](length=Self.IN_ * B, fill=Scalar[DT](0))
            var dW = List[Scalar[DT]](length=Self.W_SIZE, fill=Scalar[DT](0))
            for b in range(B):
                for i in range(Self.IN_):
                    cT[i * B + b] = fin.data[b * Self.IN_ + i]
            var cT_tt = TileTensor(cT, row_major[Self.IN_, B]())
            var go_tt = TileTensor(grad_output.data, row_major[B, Self.OUT_]())
            var dW_tt = TileTensor(dW, row_major[Self.IN_, Self.OUT_]())
            max_matmul[target="cpu"](dW_tt, cT_tt, go_tt, None)
            for i in range(Self.IN_):
                var ni = self.noise_in.data[i]
                for j in range(Self.OUT_):
                    var idx = i * Self.OUT_ + j
                    var dw = dW[idx]
                    self.mu_w.grd.data[idx] += dw
                    self.sigma_w.grd.data[idx] += (
                        dw * ni * self.noise_out.data[j]
                    )
            # grad_x = go @ W_effᵀ
            var gi_v = TileTensor(gin.data, row_major[B, Self.IN_]())
            var go_v = TileTensor(grad_output.data, row_major[B, Self.OUT_]())
            var w_v = TileTensor(
                self.w_eff.data, row_major[Self.IN_, Self.OUT_]()
            )
            max_matmul[transpose_b=True, target="cpu"](gi_v, go_v, w_v, None)
        else:
            var c = ctx.value()
            gin.ensure_gpu(c, B * Self.IN_)
            self.cacheT.ensure_gpu(c, B * Self.IN_)
            comptime lin = Layout.row_major(Self.IN_)
            comptime lout = Layout.row_major(Self.OUT_)
            comptime lw = Layout.row_major(Self.W_SIZE)
            comptime lw2 = Layout.row_major(Self.IN_, Self.OUT_)
            # grad_b pair.
            c.enqueue_function[_grad_b_pair_reduce_kernel[B, Self.OUT_]](
                grad_output.lt["gpu", Layout.row_major(B, Self.OUT_)](),
                self.noise_out.lt["gpu", lout](),
                self.mu_b.grd.lt["gpu", lout](),
                self.sigma_b.grd.lt["gpu", lout](),
                grid_dim=Self.OUT_,
                block_dim=TPB,
            )
            # grad_w: transpose x → cacheᵀ (B1' tiled), dW = cacheᵀ @ go.
            c.enqueue_function[_transpose_tiled_kernel[DT]](
                fin.dev.value(),
                self.cacheT.dev.value(),
                Int64(B),
                Int64(Self.IN_),
                grid_dim=(
                    (Self.IN_ + _T_TILE - 1) // _T_TILE,
                    (B + _T_TILE - 1) // _T_TILE,
                ),
                block_dim=(_T_TILE, _T_BR),
            )
            var cT_tt = TileTensor(
                self.cacheT.dev.value(), row_major[Self.IN_, B]()
            )
            var go_tt = TileTensor(
                grad_output.dev.value(), row_major[B, Self.OUT_]()
            )
            var dW_tt = TileTensor(
                self.dW_tmp.dev.value(), row_major[Self.IN_, Self.OUT_]()
            )
            comptime if splitk_path_applies[c.default_device_info]():
                if self._sk_p < 0:
                    self._decide_sk_p(B, c)
                if self._sk_p > 1:
                    dispatch_splitk_gemm(
                        dW_tt, cT_tt, go_tt,
                        Self.IN_, Self.OUT_, B,
                        self._sk_p, self.sk_ws, c,
                    )
                else:
                    mm[A0=Self.IN_, A1=B, B0=B, B1=Self.OUT_, O0=Self.IN_, O1=Self.OUT_](
                        self.dW_tmp.dev.value(), self.cacheT.dev.value(), grad_output.dev.value(), c
                    )
            else:
                mm[A0=Self.IN_, A1=B, B0=B, B1=Self.OUT_, O0=Self.IN_, O1=Self.OUT_](
                    self.dW_tmp.dev.value(), self.cacheT.dev.value(), grad_output.dev.value(), c
                )
            comptime nb_w = (Self.W_SIZE + TPB - 1) // TPB
            c.enqueue_function[_accum_kernel](
                self.mu_w.grd.dev.value(),
                self.dW_tmp.dev.value(),
                Int64(Self.W_SIZE),
                grid_dim=nb_w,
                block_dim=TPB,
            )
            c.enqueue_function[
                _scaled_accum_factorized_kernel[Self.IN_, Self.OUT_]
            ](
                self.sigma_w.grd.lt["gpu", lw2](),
                self.dW_tmp.lt["gpu", lw2](),
                self.noise_in.lt["gpu", lin](),
                self.noise_out.lt["gpu", lout](),
                grid_dim=nb_w,
                block_dim=TPB,
            )
            # grad_x = go @ W_effᵀ
            mm[transpose_b=True, A0=B, A1=Self.OUT_, B0=Self.IN_, B1=Self.OUT_, O0=B, O1=Self.IN_](
                gin.dev.value(), grad_output.dev.value(), self.w_eff.dev.value(), c
            )

    def polyak_from[
        target: StaticString
    ](
        mut self,
        mut src: Self,
        tau: Scalar[DT],
        ctx: Optional[DeviceContext],
    ) raises:
        """Soft-update all four noise params toward `src` (target ← online).
        Required for use as a target net (Rainbow/Noisy-DQN target nets are
        NoisyLinear); the Module default is a no-op which would silently
        freeze the target (see LinearReLU polyak bug, Stage-5)."""
        polyak_tensor[target, Self.W_SIZE](self.mu_w.val, src.mu_w.val, tau, ctx)
        polyak_tensor[target, Self.W_SIZE](
            self.sigma_w.val, src.sigma_w.val, tau, ctx
        )
        polyak_tensor[target, Self.B_SIZE](self.mu_b.val, src.mu_b.val, tau, ctx)
        polyak_tensor[target, Self.B_SIZE](
            self.sigma_b.val, src.sigma_b.val, tau, ctx
        )

    # for_each_param / zero_grad inherit the Module reflection defaults
    # (core/walkers.mojo auto-discovers the Param fields).
