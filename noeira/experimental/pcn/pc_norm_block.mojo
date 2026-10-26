"""NormPCBlock — an RMSNorm PCN level (Bogacz canonical, bottom-up).

A normalization level: predicts the normalized version of the below latent,

    μ = γ ⊙ RMSNorm(x_below),   RMSNorm(x)_i = x_i / r,  r = sqrt(mean_i(x_i²) + ε)

with IN_DIM == OUT_DIM == DIM and a learnable per-feature scale γ (DIM params).
Inserted between conv levels it yields the standard Conv → Norm → ReLU → Conv
ordering (the ReLU is the next ConvPCBlock's activation), bounding latent
magnitude so the deep conv stack trains stably at higher LR.

RMSNorm's Jacobian depends on the INPUT, but `pull_back` only sees `params` and
`act_derivative_mul` only sees the input. The split (the framework calls
`pull_back` then `act_derivative_mul(x_below, ·)`):

    pull_back:           z = ε ⊙ γ                            (has ε, γ)
    act_derivative_mul:  z ← (1/r)[z − n·(Σ_k z_k n_k)/D]     (has x_below → n, r)

This composes to exactly J^T ε for μ = γ⊙(x/r). Per-feature reduction is over
the FULL flattened feature vector (global RMSNorm), per sample.

CPU naive + GPU (per-row reduction kernels). Conforms to PCBlockTrait.
"""

from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceContext
from std.math import sqrt

from .pc_constants import TPB
from .pc_initializer import PCInitializer

from .predictive_model import PCBlockTrait
from noeira.core.tiles import Tile, tile

comptime _RMS_EPS: Float64 = 1e-6


struct NormPCBlock[dim: Int](PCBlockTrait):
    comptime IN_DIM: Int = Self.dim
    comptime OUT_DIM: Int = Self.dim
    comptime PARAM_SIZE: Int = Self.dim  # γ (per-feature scale)

    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    def __init__(out self, *, deinit move: Self):
        pass

    @staticmethod
    def pc_init_params[
        INIT: PCInitializer, dtype: DType = DType.float32
    ](
        params: Tile[dtype, Self.PARAM_SIZE],
    ) raises:
        """Init: γ = 1 (INIT unused — RMSNorm scale, not a weight)."""
        for i in range(Self.dim):
            params.ptr[unsafe_offset=i] = Scalar[dtype](1)

    # =========================================================================
    # predict:  μ = γ ⊙ (x_below / r);  a_below cached = x_below (raw)
    # =========================================================================

    @staticmethod
    def predict[
        BATCH: Int, dtype: DType = DType.float32
    ](
        x_below: Tile[dtype, BATCH, Self.IN_DIM],
        params: Tile[dtype, Self.PARAM_SIZE],
        mu: Tile[dtype, BATCH, Self.OUT_DIM],
        a_below: Tile[dtype, BATCH, Self.IN_DIM],
    ):
        # a_below cached = x_below (raw); direct LayoutTensor element indexing.
        for b in range(BATCH):
            for i in range(Self.dim):
                a_below[b, i] = x_below[b, i]
        for b in range(BATCH):
            var ss: Float64 = 0.0
            for i in range(Self.dim):
                var v = Float64(rebind[Scalar[dtype]](x_below[b, i]))
                ss += v * v
            var r = sqrt(ss / Float64(Self.dim) + _RMS_EPS)
            var inv_r = Scalar[dtype](1.0 / r)
            for i in range(Self.dim):
                mu[b, i] = params[i] * x_below[b, i] * inv_r

    # =========================================================================
    # eps_compute:  ε = x_above − μ
    # =========================================================================

    @staticmethod
    def eps_compute[
        BATCH: Int, dtype: DType = DType.float32
    ](
        x_above: Tile[dtype, BATCH, Self.OUT_DIM],
        mu: Tile[dtype, BATCH, Self.OUT_DIM],
        eps: Tile[dtype, BATCH, Self.OUT_DIM],
    ):
        # ε = x_above − μ (elementwise; scalar LayoutTensor indexing).
        for b in range(BATCH):
            for i in range(Self.dim):
                eps[b, i] = x_above[b, i] - mu[b, i]

    # =========================================================================
    # pull_back:  z = ε ⊙ γ   (per-feature scale; RMSNorm Jacobian deferred to
    #             act_derivative_mul, which sees x_below)
    # =========================================================================

    @staticmethod
    def pull_back[
        BATCH: Int, dtype: DType = DType.float32
    ](
        eps_above: Tile[dtype, BATCH, Self.OUT_DIM],
        params: Tile[dtype, Self.PARAM_SIZE],
        z_below: Tile[dtype, BATCH, Self.IN_DIM],
    ):
        # z = ε ⊙ γ (per-feature scale; direct element indexing).
        for b in range(BATCH):
            for i in range(Self.dim):
                z_below[b, i] = eps_above[b, i] * params[i]

    # =========================================================================
    # act_derivative_mul:  apply the RMSNorm Jacobian to z (=ε⊙γ) using x_below:
    #   z_out_i = (1/r)[ z_in_i − n_i · (Σ_k z_in_k n_k) / D ],  n = x_below/r
    # =========================================================================

    @staticmethod
    def act_derivative_mul[
        BATCH: Int, dtype: DType = DType.float32
    ](
        x_below: Tile[dtype, BATCH, Self.IN_DIM],
        z_in: Tile[dtype, BATCH, Self.IN_DIM],
        z_out: Tile[dtype, BATCH, Self.IN_DIM],
    ):
        # Apply the RMSNorm Jacobian to z using x_below (direct element indexing;
        # Float64 reductions preserved bit-for-bit).
        for b in range(BATCH):
            var ss: Float64 = 0.0
            for i in range(Self.dim):
                var v = Float64(rebind[Scalar[dtype]](x_below[b, i]))
                ss += v * v
            var r = sqrt(ss / Float64(Self.dim) + _RMS_EPS)
            var inv_r = 1.0 / r
            # s = Σ_k z_in_k · n_k  with n_k = x_k / r
            var s: Float64 = 0.0
            for i in range(Self.dim):
                s += (
                    Float64(rebind[Scalar[dtype]](z_in[b, i]))
                    * Float64(rebind[Scalar[dtype]](x_below[b, i]))
                    * inv_r
                )
            var s_over_d = s / Float64(Self.dim)
            for i in range(Self.dim):
                var n_i = Float64(rebind[Scalar[dtype]](x_below[b, i])) * inv_r
                z_out[b, i] = Scalar[dtype](
                    inv_r
                    * (
                        Float64(rebind[Scalar[dtype]](z_in[b, i]))
                        - n_i * s_over_d
                    )
                )

    # =========================================================================
    # weight_grad:  dγ_i = −Σ_b ε[b,i] · n[b,i]   (n = a_below/r, −sign baked)
    # =========================================================================

    @staticmethod
    def weight_grad[
        BATCH: Int, dtype: DType = DType.float32
    ](
        eps_above: Tile[dtype, BATCH, Self.OUT_DIM],
        a_below: Tile[dtype, BATCH, Self.IN_DIM],
        grads: Tile[dtype, Self.PARAM_SIZE],
    ):
        # dγ_i = −Σ_b ε[b,i]·n[b,i] (direct element indexing; same accumulation
        # order as before → bit-identical; −sign baked in).
        for i in range(Self.dim):
            grads[i] = Scalar[dtype](0)
        for b in range(BATCH):
            var ss: Float64 = 0.0
            for i in range(Self.dim):
                var v = Float64(rebind[Scalar[dtype]](a_below[b, i]))
                ss += v * v
            var inv_r = 1.0 / sqrt(ss / Float64(Self.dim) + _RMS_EPS)
            for i in range(Self.dim):
                var n_i = Float64(rebind[Scalar[dtype]](a_below[b, i])) * inv_r
                grads[i] = grads[i] - Scalar[dtype](
                    Float64(rebind[Scalar[dtype]](eps_above[b, i])) * n_i
                )

    # =========================================================================
    # GPU kernels (per-row reductions; one thread per sample row)
    # =========================================================================

    @staticmethod
    def _predict_kernel[
        BATCH: Int, dtype: DType
    ](
        x_below: Tile[dtype, BATCH, Self.dim],
        params: Tile[dtype, Self.dim],
        mu: Tile[dtype, BATCH, Self.dim],
        a_below: Tile[dtype, BATCH, Self.dim],
    ):
        var b = Int(block_dim.x * block_idx.x + thread_idx.x)
        if b >= BATCH:
            return
        var off = b * Self.dim
        var ss: Float64 = 0.0
        for i in range(Self.dim):
            var v = Float64(x_below.ptr[unsafe_offset=off + i])
            ss += v * v
            a_below.ptr[unsafe_offset=off + i] = x_below.ptr[unsafe_offset=off + i]
        var inv_r = Scalar[dtype](1.0 / sqrt(ss / Float64(Self.dim) + _RMS_EPS))
        for i in range(Self.dim):
            mu.ptr[unsafe_offset=off + i] = params.ptr[unsafe_offset=i] * x_below.ptr[unsafe_offset=off + i] * inv_r

    @staticmethod
    def _eps_kernel[
        BATCH: Int, dtype: DType
    ](
        x_above: Tile[dtype, BATCH, Self.dim],
        mu: Tile[dtype, BATCH, Self.dim],
        eps: Tile[dtype, BATCH, Self.dim],
    ):
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * Self.dim:
            return
        eps.ptr[unsafe_offset=idx] = x_above.ptr[unsafe_offset=idx] - mu.ptr[unsafe_offset=idx]

    @staticmethod
    def _pull_back_kernel[
        BATCH: Int, dtype: DType
    ](
        eps_above: Tile[dtype, BATCH, Self.dim],
        params: Tile[dtype, Self.dim],
        z_below: Tile[dtype, BATCH, Self.dim],
    ):
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * Self.dim:
            return
        var col = idx % Self.dim
        z_below.ptr[unsafe_offset=idx] = eps_above.ptr[unsafe_offset=idx] * params.ptr[unsafe_offset=col]

    @staticmethod
    def _act_deriv_kernel[
        BATCH: Int, dtype: DType
    ](
        x_below: Tile[dtype, BATCH, Self.dim],
        z_in: Tile[dtype, BATCH, Self.dim],
        z_out: Tile[dtype, BATCH, Self.dim],
    ):
        var b = Int(block_dim.x * block_idx.x + thread_idx.x)
        if b >= BATCH:
            return
        var off = b * Self.dim
        var ss: Float64 = 0.0
        for i in range(Self.dim):
            var v = Float64(x_below.ptr[unsafe_offset=off + i])
            ss += v * v
        var inv_r = 1.0 / sqrt(ss / Float64(Self.dim) + _RMS_EPS)
        var s: Float64 = 0.0
        for i in range(Self.dim):
            s += (
                Float64(z_in.ptr[unsafe_offset=off + i])
                * Float64(x_below.ptr[unsafe_offset=off + i])
                * inv_r
            )
        var s_over_d = s / Float64(Self.dim)
        for i in range(Self.dim):
            var n_i = Float64(x_below.ptr[unsafe_offset=off + i]) * inv_r
            z_out.ptr[unsafe_offset=off + i] = Scalar[dtype](
                inv_r * (Float64(z_in.ptr[unsafe_offset=off + i]) - n_i * s_over_d)
            )

    @staticmethod
    def _rms_per_row_kernel[
        BATCH: Int, dtype: DType
    ](
        a_below: Tile[dtype, BATCH, Self.dim],
        inv_r_buf: Tile[dtype, BATCH],
    ):
        var b = Int(block_dim.x * block_idx.x + thread_idx.x)
        if b >= BATCH:
            return
        var off = b * Self.dim
        var ss: Float64 = 0.0
        for i in range(Self.dim):
            var v = Float64(a_below.ptr[unsafe_offset=off + i])
            ss += v * v
        inv_r_buf.ptr[unsafe_offset=b] = Scalar[dtype](
            1.0 / sqrt(ss / Float64(Self.dim) + _RMS_EPS)
        )

    @staticmethod
    def _weight_grad_kernel[
        BATCH: Int, dtype: DType
    ](
        eps_above: Tile[dtype, BATCH, Self.dim],
        a_below: Tile[dtype, BATCH, Self.dim],
        inv_r_buf: Tile[dtype, BATCH],
        grads: Tile[dtype, Self.dim],
    ):
        var i = Int(block_dim.x * block_idx.x + thread_idx.x)
        if i >= Self.dim:
            return
        var acc: Float64 = 0.0
        for b in range(BATCH):
            var off = b * Self.dim
            var n_i = Float64(a_below.ptr[unsafe_offset=off + i]) * Float64(inv_r_buf.ptr[unsafe_offset=b])
            acc += Float64(eps_above.ptr[unsafe_offset=off + i]) * n_i
        grads.ptr[unsafe_offset=i] = Scalar[dtype](-acc)

    # ── GPU dispatchers ──────────────────────────────────────────────────────

    @staticmethod
    def predict_gpu[
        BATCH: Int, dtype: DType = DType.float32
    ](
        ctx: DeviceContext,
        x_below: Tile[dtype, BATCH, Self.IN_DIM],
        params: Tile[dtype, Self.PARAM_SIZE],
        mu: Tile[dtype, BATCH, Self.OUT_DIM],
        a_below: Tile[dtype, BATCH, Self.IN_DIM],
    ) raises:
        comptime k = Self._predict_kernel[BATCH, dtype]
        var blocks = (BATCH + TPB - 1) // TPB
        ctx.enqueue_function[k](
            x_below, params, mu, a_below, grid_dim=(blocks,), block_dim=(TPB,)
        )

    @staticmethod
    def eps_compute_gpu[
        BATCH: Int, dtype: DType = DType.float32
    ](
        ctx: DeviceContext,
        x_above: Tile[dtype, BATCH, Self.OUT_DIM],
        mu: Tile[dtype, BATCH, Self.OUT_DIM],
        eps: Tile[dtype, BATCH, Self.OUT_DIM],
    ) raises:
        comptime k = Self._eps_kernel[BATCH, dtype]
        var threads = BATCH * Self.dim
        var blocks = (threads + TPB - 1) // TPB
        ctx.enqueue_function[k](
            x_above, mu, eps, grid_dim=(blocks,), block_dim=(TPB,)
        )

    @staticmethod
    def pull_back_gpu[
        BATCH: Int, dtype: DType = DType.float32
    ](
        ctx: DeviceContext,
        eps_above: Tile[dtype, BATCH, Self.OUT_DIM],
        params: Tile[dtype, Self.PARAM_SIZE],
        z_below: Tile[dtype, BATCH, Self.IN_DIM],
    ) raises:
        comptime k = Self._pull_back_kernel[BATCH, dtype]
        var threads = BATCH * Self.dim
        var blocks = (threads + TPB - 1) // TPB
        ctx.enqueue_function[k](
            eps_above, params, z_below, grid_dim=(blocks,), block_dim=(TPB,)
        )

    @staticmethod
    def act_derivative_mul_gpu[
        BATCH: Int, dtype: DType = DType.float32
    ](
        ctx: DeviceContext,
        x_below: Tile[dtype, BATCH, Self.IN_DIM],
        z_in: Tile[dtype, BATCH, Self.IN_DIM],
        z_out: Tile[dtype, BATCH, Self.IN_DIM],
    ) raises:
        comptime k = Self._act_deriv_kernel[BATCH, dtype]
        var blocks = (BATCH + TPB - 1) // TPB
        ctx.enqueue_function[k](
            x_below, z_in, z_out, grid_dim=(blocks,), block_dim=(TPB,)
        )

    @staticmethod
    def weight_grad_gpu[
        BATCH: Int, dtype: DType = DType.float32
    ](
        ctx: DeviceContext,
        eps_above: Tile[dtype, BATCH, Self.OUT_DIM],
        a_below: Tile[dtype, BATCH, Self.IN_DIM],
        grads: Tile[dtype, Self.PARAM_SIZE],
    ) raises:
        var inv_r_buf = ctx.enqueue_create_buffer[dtype](BATCH)
        var inv_r = tile[BATCH](inv_r_buf)
        comptime kr = Self._rms_per_row_kernel[BATCH, dtype]
        var rblocks = (BATCH + TPB - 1) // TPB
        ctx.enqueue_function[kr](
            a_below, inv_r, grid_dim=(rblocks,), block_dim=(TPB,)
        )
        comptime kg = Self._weight_grad_kernel[BATCH, dtype]
        var gblocks = (Self.dim + TPB - 1) // TPB
        ctx.enqueue_function[kg](
            eps_above,
            a_below,
            inv_r,
            grads,
            grid_dim=(gblocks,),
            block_dim=(TPB,),
        )
        _ = inv_r_buf
