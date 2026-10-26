"""PCBlock — one PCN level (Bogacz canonical, bottom-up).

W stored as `[in_dim, out_dim]`, bias as `[out_dim]`. Bottom-up prediction:

    a_below = ACT(x_below)                                 # cached
    μ_above = a_below @ W + b                              # [B, OUT_DIM]
    ε_above = x_above − μ_above                            # local error
    z_below = ε_above @ W^T                                # pull-back to below
    f'-mod  = z_below ⊙ act'(x_below)                      # gain-modulated
    dE/dW   = −a_below^T @ ε_above                         # weight gradient
    dE/db   = −Σ_b ε_above[b, :]                           # bias gradient

Sign convention: ε = x − μ (latent above minus prediction), uniform across
interior and readout. The block does NOT add the "−" sign for ε itself —
that's handled by the caller. But weight_grad / bias_grad bake in the −sign
so that callers can pass the result directly to an Optimizer.step().

For the readout, ε_L = target − output (target plays the role of x_above).
"""

from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from max.gpu.memory import AddressSpace
from std.sys import CompilationTarget
from linalg.matmul import matmul as max_matmul
from linalg.matmul.cpu.apple_accelerate import (
    get_cblas_f32_function,
    _CBLASOrder,
    _CBLASTranspose,
)
from noeira.core.tiles import Tile, tile

from .pc_constants import TPB
from .pc_initializer import PCInitializer

from .predictive_model import PCActivation, PCReLU, PCBlockTrait


struct PCBlock[
    in_dim: Int,
    out_dim: Int,
    ACT: PCActivation = PCReLU,
](PCBlockTrait):
    """One PCN level: W [in_dim, out_dim] + b [out_dim] + bundled `ACT`.

    Naming matches `nn.Linear[in_dim, out_dim]`:
      - in_dim  is the below side (predicting block reads x_below of this dim)
      - out_dim is the above side (latent x_above of this dim is predicted)

    For the readout, pass `ACT=PCIdentity` and target plays role of x_above.

    GPU matmuls (predict / pull_back / weight_grad) go through
    `linalg.matmul` (`max_matmul`) on BOTH Apple + NVIDIA — nn convention,
    no custom MMA. (The old 2×2 register-tiled fallback was removed in the
    nn re-architecture once `max_matmul` was proven parity-equal on Apple.)
    """

    comptime IN_DIM: Int = Self.in_dim
    comptime OUT_DIM: Int = Self.out_dim
    comptime PARAM_SIZE: Int = Self.in_dim * Self.out_dim + Self.out_dim

    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    def __init__(out self, *, deinit move: Self):
        pass

    # =========================================================================
    # Initialization
    # =========================================================================

    @staticmethod
    def pc_init_params[
        INIT: PCInitializer, dtype: DType = DType.float32
    ](
        params: Tile[dtype, Self.PARAM_SIZE],
    ) raises:
        """Init: W via INIT.fill(fan_in=in_dim, fan_out=out_dim); zero b."""
        var W_view = tile[Self.in_dim * Self.out_dim](params.ptr)
        INIT.fill[Self.in_dim * Self.out_dim, Self.in_dim, Self.out_dim, dtype](
            W_view
        )
        for j in range(Self.out_dim):
            params.ptr[unsafe_offset=Self.in_dim * Self.out_dim + j] = Scalar[dtype](0)

    # =========================================================================
    # Bottom-up prediction:  a_below = ACT(x_below);  μ = a_below @ W + b
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
        """μ[b, j] = sum_i ACT(x_below[b, i]) * W[i, j] + b[j].

        a_below is cached for use in weight_grad later in the same training
        step; caller owns the buffer.
        """
        # 1. a_below = ACT(x_below)
        Self.ACT.apply[BATCH, Self.in_dim, dtype](x_below, a_below)

        # 2. μ = a_below @ W + b. W/b are sub-views of the param slab; build
        #    them like the GPU path (LayoutTensor over the slab ptr, no rebind),
        #    and feed the matmul the TileTensor views directly.
        var W = tile[Self.in_dim, Self.out_dim](params.ptr)
        var b_view = tile[Self.out_dim](params.ptr.unsafe_offset(Self.in_dim * Self.out_dim))
        try:
            max_matmul[target="cpu"](
                mu, a_below, W, None
            )
        except:
            pass
        # bias add (scalar, matching the nn.storage CPU bias-add)
        for sb in range(BATCH):
            for j in range(Self.out_dim):
                mu[sb, j] = mu[sb, j] + b_view[j]

    # =========================================================================
    # Local prediction error:  ε = x_above − μ
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
        for sb in range(BATCH):
            for j in range(Self.out_dim):
                eps[sb, j] = x_above[sb, j] - mu[sb, j]

    # =========================================================================
    # Pull-back to latent below:  z_below = ε_above @ W^T
    # =========================================================================

    @staticmethod
    def pull_back[
        BATCH: Int, dtype: DType = DType.float32
    ](
        eps_above: Tile[dtype, BATCH, Self.OUT_DIM],
        params: Tile[dtype, Self.PARAM_SIZE],
        z_below: Tile[dtype, BATCH, Self.IN_DIM],
    ):
        """Formula = z_below[b, i] = sum_j ε_above[b, j] * W[i, j]."""
        # z_below = eps_above @ W^T  → matmul with transpose_b. W is the [in,out]
        # head of the param slab; build it like the GPU path (no rebind) and run
        # the GEMM on the TileTensor views.
        var W = tile[Self.in_dim, Self.out_dim](params.ptr)
        try:
            max_matmul[transpose_b=True, target="cpu"](
                z_below, eps_above, W, None
            )
        except:
            pass

    # =========================================================================
    # Activation derivative gating:  z_out = z_in ⊙ act'(x_below)
    # =========================================================================

    @staticmethod
    def act_derivative_mul[
        BATCH: Int, dtype: DType = DType.float32
    ](
        x_below: Tile[dtype, BATCH, Self.IN_DIM],
        z_in: Tile[dtype, BATCH, Self.IN_DIM],
        z_out: Tile[dtype, BATCH, Self.IN_DIM],
    ):
        Self.ACT.apply_derivative_mul[BATCH, Self.in_dim, dtype](
            x_below, z_in, z_out
        )

    # =========================================================================
    # Weight + bias gradients (signed for direct Optimizer.step consumption)
    #   dE/dW[i, j] = −Σ_b ε_above[b, j] * a_below[b, i]
    #   dE/db[j]    = −Σ_b ε_above[b, j]
    # =========================================================================

    @staticmethod
    def weight_grad[
        BATCH: Int, dtype: DType = DType.float32
    ](
        eps_above: Tile[dtype, BATCH, Self.OUT_DIM],
        a_below: Tile[dtype, BATCH, Self.IN_DIM],
        grads: Tile[dtype, Self.PARAM_SIZE],
    ):
        """Writes gradients (W, b) into `grads` in [W flat | b] layout.

        Sign baked in so that `params -= lr * grads` performs gradient descent
        on the local energy (or output loss for the readout block).
        """
        # Grad sub-views over the slab (like the GPU path; no rebind).
        var W_grad = tile[Self.in_dim, Self.out_dim](grads.ptr)
        var b_grad = tile[Self.out_dim](grads.ptr.unsafe_offset(Self.in_dim * Self.out_dim))

        # dE/dW = −(a_below^T @ eps_above), written into W_grad then negated.
        comptime if CompilationTarget.is_macos() and dtype == DType.float32:
            # Apple Accelerate: dW = a_belowᵀ @ eps in one cblas_sgemm call
            # (transpose_a, beta=0). The rebinds here are the cblas FFI boundary
            # (Scalar[dtype]→Float32 + origin), exactly as nn.storage's Linear.
            try:
                var cblas_gemm = get_cblas_f32_function()
                cblas_gemm(
                    _CBLASOrder.ROW_MAJOR,
                    _CBLASTranspose.TRANSPOSE,
                    _CBLASTranspose.NO_TRANSPOSE,
                    Int32(Self.in_dim),
                    Int32(Self.out_dim),
                    Int32(BATCH),
                    Float32(1.0),
                    rebind[Pointer[Float32, ImmutAnyOrigin]](a_below.ptr),
                    Int32(Self.in_dim),
                    rebind[Pointer[Float32, ImmutAnyOrigin]](
                        eps_above.ptr
                    ),
                    Int32(Self.out_dim),
                    Float32(0.0),
                    rebind[Pointer[Float32, MutAnyOrigin]](grads.ptr),
                    Int32(Self.out_dim),
                )
            except:
                pass
        else:
            # Portable: materialize a_belowᵀ into an owned List (no raw alloc),
            # then GEMM on the TileTensor views (max_matmul rejects transpose_a).
            var cT = List[Scalar[dtype]](
                length=Self.in_dim * BATCH, fill=Scalar[dtype](0)
            )
            for bi in range(BATCH):
                for i in range(Self.in_dim):
                    cT[i * BATCH + bi] = rebind[Scalar[dtype]](a_below[bi, i])
            var cT_view = tile[Self.in_dim, BATCH](cT)
            try:
                max_matmul[target="cpu"](
                    W_grad,
                    cT_view,
                    eps_above,
                    None,
                )
            except:
                pass
            _ = cT^

        # Bake in the −sign expected by Optimizer.step (params -= lr·grads).
        for i in range(Self.in_dim):
            for j in range(Self.out_dim):
                W_grad[i, j] = -W_grad[i, j]

        # dE/db[j] = −Σ_b ε_above[b, j] (column-sum + negate; stays in the
        # LayoutTensor element type so no Scalar conversion is needed).
        for j in range(Self.out_dim):
            var s = eps_above[0, j]
            for sb in range(1, BATCH):
                s = s + eps_above[sb, j]
            b_grad[j] = -s

    # =========================================================================
    # GPU kernels (elementwise: ε computation, bias gradient)
    # =========================================================================

    @staticmethod
    def _eps_kernel[
        BATCH: Int,
        OUT: Int,
        dtype: DType,
    ](
        x_above: Tile[dtype, BATCH, OUT],
        mu: Tile[dtype, BATCH, OUT],
        eps: Tile[dtype, BATCH, OUT],
    ):
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * OUT:
            return
        var b = idx // OUT
        var j = idx % OUT
        eps[b, j] = rebind[Scalar[dtype]](x_above[b, j]) - rebind[
            Scalar[dtype]
        ](mu[b, j])

    @staticmethod
    def _bias_grad_kernel[
        BATCH: Int,
        OUT: Int,
        dtype: DType,
    ](
        eps_above: Tile[dtype, BATCH, OUT],
        b_grad: Tile[dtype, OUT],
    ):
        var j = Int(block_dim.x * block_idx.x + thread_idx.x)
        if j >= OUT:
            return
        var s: Scalar[dtype] = 0
        for sb in range(BATCH):
            s += rebind[Scalar[dtype]](eps_above[sb, j])
        b_grad[j] = -s

    # ── Helpers used by the max_matmul fast path ─────────────────────────────

    @staticmethod
    def _bias_add_kernel[
        BATCH: Int,
        OUT: Int,
        dtype: DType,
    ](
        mu: Tile[dtype, BATCH, OUT],
        b: Tile[dtype, OUT],
    ):
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * OUT:
            return
        var col = idx % OUT
        mu.ptr[unsafe_offset=idx] = mu.ptr[unsafe_offset=idx] + rebind[Scalar[dtype]](b[col])

    @staticmethod
    def _negate_kernel[
        N: Int,
        dtype: DType,
    ](buf: Tile[dtype, N],):
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= N:
            return
        buf.ptr[unsafe_offset=idx] = -buf.ptr[unsafe_offset=idx]

    @staticmethod
    def _transpose_2d_kernel[
        ROWS: Int,
        COLS: Int,
        dtype: DType,
    ](
        dst: Tile[dtype, COLS, ROWS],
        src: Tile[dtype, ROWS, COLS],
    ):
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= ROWS * COLS:
            return
        var row = idx // COLS
        var col = idx % COLS
        dst[col, row] = src[row, col]

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
        # 1. a_below = ACT(x_below)
        Self.ACT.apply_gpu[BATCH, Self.in_dim, dtype](ctx, x_below, a_below)

        # 2. μ = a_below @ W + b
        var W = tile[Self.in_dim, Self.out_dim](params.ptr)
        var b_view = tile[Self.out_dim](params.ptr.unsafe_offset(Self.in_dim * Self.out_dim))

        # max_matmul writes mu = a_below @ W (no bias), then a separate
        # bias-add kernel folds in b. Apple + NVIDIA (nn convention).
        max_matmul[target="gpu"](
            mu,
            a_below,
            W,
            ctx,
        )

        comptime kb = Self._bias_add_kernel[BATCH, Self.out_dim, dtype]
        var ba_threads = BATCH * Self.out_dim
        var ba_blocks = (ba_threads + TPB - 1) // TPB
        ctx.enqueue_function[kb](
            mu,
            b_view,
            grid_dim=(ba_blocks,),
            block_dim=(TPB,),
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
        comptime k = Self._eps_kernel[BATCH, Self.out_dim, dtype]
        var threads = BATCH * Self.out_dim
        var blocks = (threads + TPB - 1) // TPB
        ctx.enqueue_function[k](
            x_above,
            mu,
            eps,
            grid_dim=(blocks,),
            block_dim=(TPB,),
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
        var W = tile[Self.in_dim, Self.out_dim](params.ptr)
        # z_below[B, IN] = eps_above[B, OUT] @ W^T[OUT, IN]
        # max_matmul with transpose_b=True treats W (stored [IN, OUT]) as if
        # it were W^T, giving exactly this contraction. Apple + NVIDIA.
        max_matmul[transpose_b=True, target="gpu"](
            z_below,
            eps_above,
            W,
            ctx,
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
        Self.ACT.apply_derivative_mul_gpu[BATCH, Self.in_dim, dtype](
            ctx, x_below, z_in, z_out
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
        var W_grad = tile[Self.in_dim, Self.out_dim](grads.ptr)
        var b_grad = tile[Self.out_dim](grads.ptr.unsafe_offset(Self.in_dim * Self.out_dim))

        # dW[IN, OUT] = -a_below^T[IN, BATCH] @ eps_above[BATCH, OUT]
        # max_matmul has no transpose_a, so materialize a_below^T into a
        # scratch buffer first, then compute the GEMM and negate dW.
        var a_T_buf = ctx.enqueue_create_buffer[dtype](BATCH * Self.in_dim)
        var a_T = tile[Self.in_dim, BATCH](a_T_buf)

        comptime kt = Self._transpose_2d_kernel[BATCH, Self.in_dim, dtype]
        comptime t_blocks = (BATCH * Self.in_dim + TPB - 1) // TPB
        ctx.enqueue_function[kt](
            a_T,
            a_below,
            grid_dim=(t_blocks,),
            block_dim=(TPB,),
        )

        max_matmul[target="gpu"](
            W_grad,
            a_T,
            eps_above,
            ctx,
        )

        # Bake in the −sign expected by Optimizer.step (params -= lr·grads).
        var W_grad_flat = tile[Self.in_dim * Self.out_dim](grads.ptr)
        comptime kn = Self._negate_kernel[Self.in_dim * Self.out_dim, dtype]
        comptime n_blocks = (Self.in_dim * Self.out_dim + TPB - 1) // TPB
        ctx.enqueue_function[kn](
            W_grad_flat,
            grid_dim=(n_blocks,),
            block_dim=(TPB,),
        )

        comptime kb = Self._bias_grad_kernel[BATCH, Self.out_dim, dtype]
        var b_threads = Self.out_dim
        var b_blocks = (b_threads + TPB - 1) // TPB
        ctx.enqueue_function[kb](
            eps_above,
            b_grad,
            grid_dim=(b_blocks,),
            block_dim=(TPB,),
        )
