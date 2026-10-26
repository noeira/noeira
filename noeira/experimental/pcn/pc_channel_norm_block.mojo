"""ChannelNormPCBlock — per-channel RMSNorm PCN level (Bogacz canonical).

The PC-native answer to "what BatchNorm gives a conv net" — but per-SAMPLE
(no batch coupling) and per-CHANNEL (preserves inter-channel scale, unlike the
global `NormPCBlock`). For a feature map [C, H·W] flattened, each channel is
RMS-normalized independently over its H·W positions, with a per-channel scale γ:

    r_c   = sqrt( mean_{s}( x[c,s]² ) + ε )          # per channel, per sample
    μ[c,s] = γ_c · x[c,s] / r_c

IN_DIM == OUT_DIM == channels·spatial.  γ is per-channel (PARAM_SIZE = channels).
Layout matches the conv blocks: index = c·spatial + s.

Same Jacobian split as NormPCBlock, scoped per channel (the framework calls
`pull_back` then `act_derivative_mul(x_below, ·)`):

    pull_back:           z[c,s] = ε[c,s] · γ_c
    act_derivative_mul:  z[c,s] ← (1/r_c)[ z[c,s] − n[c,s]·(Σ_s' z[c,s']·n[c,s'])/HW ]

CPU naive + GPU (one thread per (sample, channel) reduction). Conforms to
PCBlockTrait.
"""

from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceContext
from std.math import sqrt

from .pc_constants import TPB
from .pc_initializer import PCInitializer

from .predictive_model import PCBlockTrait
from noeira.core.tiles import Tile, tile

comptime _RMS_EPS: Float64 = 1e-6


struct ChannelNormPCBlock[channels: Int, spatial: Int](PCBlockTrait):
    comptime DIM: Int = Self.channels * Self.spatial
    comptime IN_DIM: Int = Self.DIM
    comptime OUT_DIM: Int = Self.DIM
    comptime PARAM_SIZE: Int = Self.channels  # γ per channel

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
        """NN init: per-channel γ = 1 (INIT unused — normalization scale)."""
        for c in range(Self.channels):
            params[c] = Scalar[dtype](1)

    # =========================================================================
    # predict:  μ[c,s] = γ_c · x[c,s] / r_c ;  a_below cached = x (raw)
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
        for b in range(BATCH):
            for col in range(Self.DIM):
                a_below[b, col] = x_below[b, col]
        for b in range(BATCH):
            for c in range(Self.channels):
                var base = c * Self.spatial
                var ss: Float64 = 0.0
                for s in range(Self.spatial):
                    var v = Float64(rebind[Scalar[dtype]](x_below[b, base + s]))
                    ss += v * v
                var inv_r = Scalar[dtype](
                    1.0 / sqrt(ss / Float64(Self.spatial) + _RMS_EPS)
                )
                var g = rebind[Scalar[dtype]](params[c])
                for s in range(Self.spatial):
                    mu[b, base + s] = (
                        g * rebind[Scalar[dtype]](x_below[b, base + s]) * inv_r
                    )

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
        for b in range(BATCH):
            for col in range(Self.DIM):
                eps[b, col] = x_above[b, col] - mu[b, col]

    # =========================================================================
    # pull_back:  z[c,s] = ε[c,s] · γ_c
    # =========================================================================

    @staticmethod
    def pull_back[
        BATCH: Int, dtype: DType = DType.float32
    ](
        eps_above: Tile[dtype, BATCH, Self.OUT_DIM],
        params: Tile[dtype, Self.PARAM_SIZE],
        z_below: Tile[dtype, BATCH, Self.IN_DIM],
    ):
        for b in range(BATCH):
            for c in range(Self.channels):
                var base = c * Self.spatial
                var g = rebind[Scalar[dtype]](params[c])
                for s in range(Self.spatial):
                    z_below[b, base + s] = (
                        rebind[Scalar[dtype]](eps_above[b, base + s]) * g
                    )

    # =========================================================================
    # act_derivative_mul:  per-channel RMSNorm Jacobian applied to z, using x:
    #   z_out[c,s] = (1/r_c)[ z_in[c,s] − n[c,s]·(Σ_s' z_in[c,s']·n[c,s'])/HW ]
    # =========================================================================

    @staticmethod
    def act_derivative_mul[
        BATCH: Int, dtype: DType = DType.float32
    ](
        x_below: Tile[dtype, BATCH, Self.IN_DIM],
        z_in: Tile[dtype, BATCH, Self.IN_DIM],
        z_out: Tile[dtype, BATCH, Self.IN_DIM],
    ):
        for b in range(BATCH):
            for c in range(Self.channels):
                var base = c * Self.spatial
                var ss: Float64 = 0.0
                for s in range(Self.spatial):
                    var v = Float64(rebind[Scalar[dtype]](x_below[b, base + s]))
                    ss += v * v
                var inv_r = 1.0 / sqrt(ss / Float64(Self.spatial) + _RMS_EPS)
                var dot: Float64 = 0.0
                for s in range(Self.spatial):
                    dot += (
                        Float64(rebind[Scalar[dtype]](z_in[b, base + s]))
                        * Float64(rebind[Scalar[dtype]](x_below[b, base + s]))
                        * inv_r
                    )
                var dot_over = dot / Float64(Self.spatial)
                for s in range(Self.spatial):
                    var n_s = Float64(rebind[Scalar[dtype]](x_below[b, base + s])) * inv_r
                    z_out[b, base + s] = Scalar[dtype](
                        inv_r * (Float64(rebind[Scalar[dtype]](z_in[b, base + s])) - n_s * dot_over)
                    )

    # =========================================================================
    # weight_grad:  dγ_c = −Σ_b Σ_s ε[b,c,s] · n[b,c,s]   (−sign baked)
    # =========================================================================

    @staticmethod
    def weight_grad[
        BATCH: Int, dtype: DType = DType.float32
    ](
        eps_above: Tile[dtype, BATCH, Self.OUT_DIM],
        a_below: Tile[dtype, BATCH, Self.IN_DIM],
        grads: Tile[dtype, Self.PARAM_SIZE],
    ):
        for c in range(Self.channels):
            grads[c] = Scalar[dtype](0)
        for b in range(BATCH):
            for c in range(Self.channels):
                var base = c * Self.spatial
                var ss: Float64 = 0.0
                for s in range(Self.spatial):
                    var v = Float64(rebind[Scalar[dtype]](a_below[b, base + s]))
                    ss += v * v
                var inv_r = 1.0 / sqrt(ss / Float64(Self.spatial) + _RMS_EPS)
                var acc: Float64 = 0.0
                for s in range(Self.spatial):
                    acc += (
                        Float64(rebind[Scalar[dtype]](eps_above[b, base + s]))
                        * Float64(rebind[Scalar[dtype]](a_below[b, base + s]))
                        * inv_r
                    )
                grads[c] = rebind[Scalar[dtype]](grads[c]) - Scalar[dtype](acc)

    # =========================================================================
    # GPU kernels (one thread per (sample, channel) for reductions)
    # =========================================================================

    @staticmethod
    def _predict_kernel[
        BATCH: Int, dtype: DType
    ](
        x_below: Tile[dtype, BATCH, Self.DIM],
        params: Tile[dtype, Self.channels],
        mu: Tile[dtype, BATCH, Self.DIM],
        a_below: Tile[dtype, BATCH, Self.DIM],
    ):
        var bc = Int(block_dim.x * block_idx.x + thread_idx.x)
        if bc >= BATCH * Self.channels:
            return
        var b = bc // Self.channels
        var c = bc % Self.channels
        var off = b * Self.DIM + c * Self.spatial
        var ss: Float64 = 0.0
        for s in range(Self.spatial):
            var v = Float64(x_below.ptr[unsafe_offset=off + s])
            ss += v * v
            a_below.ptr[unsafe_offset=off + s] = x_below.ptr[unsafe_offset=off + s]
        var inv_r = Scalar[dtype](
            1.0 / sqrt(ss / Float64(Self.spatial) + _RMS_EPS)
        )
        var g = params.ptr[unsafe_offset=c]
        for s in range(Self.spatial):
            mu.ptr[unsafe_offset=off + s] = g * x_below.ptr[unsafe_offset=off + s] * inv_r

    @staticmethod
    def _eps_kernel[
        BATCH: Int, dtype: DType
    ](
        x_above: Tile[dtype, BATCH, Self.DIM],
        mu: Tile[dtype, BATCH, Self.DIM],
        eps: Tile[dtype, BATCH, Self.DIM],
    ):
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * Self.DIM:
            return
        eps.ptr[unsafe_offset=idx] = x_above.ptr[unsafe_offset=idx] - mu.ptr[unsafe_offset=idx]

    @staticmethod
    def _pull_back_kernel[
        BATCH: Int, dtype: DType
    ](
        eps_above: Tile[dtype, BATCH, Self.DIM],
        params: Tile[dtype, Self.channels],
        z_below: Tile[dtype, BATCH, Self.DIM],
    ):
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * Self.DIM:
            return
        var c = (idx % Self.DIM) // Self.spatial
        z_below.ptr[unsafe_offset=idx] = eps_above.ptr[unsafe_offset=idx] * params.ptr[unsafe_offset=c]

    @staticmethod
    def _act_deriv_kernel[
        BATCH: Int, dtype: DType
    ](
        x_below: Tile[dtype, BATCH, Self.DIM],
        z_in: Tile[dtype, BATCH, Self.DIM],
        z_out: Tile[dtype, BATCH, Self.DIM],
    ):
        var bc = Int(block_dim.x * block_idx.x + thread_idx.x)
        if bc >= BATCH * Self.channels:
            return
        var b = bc // Self.channels
        var c = bc % Self.channels
        var off = b * Self.DIM + c * Self.spatial
        var ss: Float64 = 0.0
        for s in range(Self.spatial):
            var v = Float64(x_below.ptr[unsafe_offset=off + s])
            ss += v * v
        var inv_r = 1.0 / sqrt(ss / Float64(Self.spatial) + _RMS_EPS)
        var dot: Float64 = 0.0
        for s in range(Self.spatial):
            dot += (
                Float64(z_in.ptr[unsafe_offset=off + s])
                * Float64(x_below.ptr[unsafe_offset=off + s])
                * inv_r
            )
        var dot_over = dot / Float64(Self.spatial)
        for s in range(Self.spatial):
            var n_s = Float64(x_below.ptr[unsafe_offset=off + s]) * inv_r
            z_out.ptr[unsafe_offset=off + s] = Scalar[dtype](
                inv_r * (Float64(z_in.ptr[unsafe_offset=off + s]) - n_s * dot_over)
            )

    @staticmethod
    def _inv_r_kernel[
        BATCH: Int, dtype: DType
    ](
        a_below: Tile[dtype, BATCH, Self.DIM],
        inv_r_buf: Tile[dtype, BATCH * Self.channels],
    ):
        var bc = Int(block_dim.x * block_idx.x + thread_idx.x)
        if bc >= BATCH * Self.channels:
            return
        var b = bc // Self.channels
        var c = bc % Self.channels
        var off = b * Self.DIM + c * Self.spatial
        var ss: Float64 = 0.0
        for s in range(Self.spatial):
            var v = Float64(a_below.ptr[unsafe_offset=off + s])
            ss += v * v
        inv_r_buf.ptr[unsafe_offset=bc] = Scalar[dtype](
            1.0 / sqrt(ss / Float64(Self.spatial) + _RMS_EPS)
        )

    @staticmethod
    def _weight_grad_kernel[
        BATCH: Int, dtype: DType
    ](
        eps_above: Tile[dtype, BATCH, Self.DIM],
        a_below: Tile[dtype, BATCH, Self.DIM],
        inv_r_buf: Tile[dtype, BATCH * Self.channels],
        grads: Tile[dtype, Self.channels],
    ):
        var c = Int(block_dim.x * block_idx.x + thread_idx.x)
        if c >= Self.channels:
            return
        var acc: Float64 = 0.0
        for b in range(BATCH):
            var off = b * Self.DIM + c * Self.spatial
            var inv_r = Float64(inv_r_buf.ptr[unsafe_offset=b * Self.channels + c])
            for s in range(Self.spatial):
                acc += (
                    Float64(eps_above.ptr[unsafe_offset=off + s])
                    * Float64(a_below.ptr[unsafe_offset=off + s])
                    * inv_r
                )
        grads.ptr[unsafe_offset=c] = Scalar[dtype](-acc)

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
        var threads = BATCH * Self.channels
        var blocks = (threads + TPB - 1) // TPB
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
        var threads = BATCH * Self.DIM
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
        var threads = BATCH * Self.DIM
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
        var threads = BATCH * Self.channels
        var blocks = (threads + TPB - 1) // TPB
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
        var inv_r_buf = ctx.enqueue_create_buffer[dtype](BATCH * Self.channels)
        var inv_r = tile[BATCH * Self.channels](inv_r_buf)
        comptime kr = Self._inv_r_kernel[BATCH, dtype]
        var rthreads = BATCH * Self.channels
        var rblocks = (rthreads + TPB - 1) // TPB
        ctx.enqueue_function[kr](
            a_below, inv_r, grid_dim=(rblocks,), block_dim=(TPB,)
        )
        comptime kg = Self._weight_grad_kernel[BATCH, dtype]
        var gblocks = (Self.channels + TPB - 1) // TPB
        ctx.enqueue_function[kg](
            eps_above,
            a_below,
            inv_r,
            grads,
            grid_dim=(gblocks,),
            block_dim=(TPB,),
        )
        _ = inv_r_buf
