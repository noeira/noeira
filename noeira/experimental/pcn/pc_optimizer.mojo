# =============================================================================
# PCN-local slab optimizers (PCAdam, PCAdamW)
# =============================================================================
#
# These are vendored verbatim from the legacy `noeira.nn.optimizer` package
# (adam.mojo + adamw.mojo) during the nn re-architecture, so that
# `noeira.experimental.pcn` carries no dependency on legacy `nn` (which can
# then be deleted). The ONLY changes from the originals are:
#   - `dtype` now sourced from `noeira.nn.constants` (`DT`) and `TPB` from
#     the PCN-local `pc_constants` module.
#   - structs renamed `Adam` -> `PCAdam`, `AdamW` -> `PCAdamW` and no longer
#     conform to the legacy `Optimizer` trait (PCN tests call `.step[...]`
#     directly).
# All algorithm/logic (CPU SIMD step, GPU kernels, launchers) is unchanged.

from noeira.nn.constants import DT as dtype
from .pc_constants import TPB
from std.math import sqrt, exp, log
from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceContext, DeviceBuffer
from std.sys import simd_width_of
from noeira.core.tiles import Tile, tile


comptime _CPU_SIMD_W = simd_width_of[dtype]()


struct PCAdam[
    LR: Float64 = 0.001,
    BETA1: Float64 = 0.9,
    BETA2: Float64 = 0.999,
    EPS: Float64 = 1e-8,
    WEIGHT_DECAY: Float64 = 0.0,
]:
    """Adam optimizer with adaptive learning rates.

    Matches PyTorch's `torch.optim.Adam(weight_decay=W)` semantics: when
    `WEIGHT_DECAY > 0`, the L2 regularization term `W * param` is added to
    the gradient BEFORE the m/v update. This is the "L2-in-gradient" form
    used by muzero-general and most reference RL/CV codebases (different
    from AdamW's decoupled weight decay, which is `param *= (1 - LR*W)`
    AFTER the Adam update).

    Update rule:
        g = grad + WEIGHT_DECAY * param   (only if WEIGHT_DECAY > 0)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g^2
        m_hat = m / (1 - beta1^step)
        v_hat = v / (1 - beta2^step)
        param = param - lr * m_hat / (sqrt(v_hat) + eps)

    The L2-in-gradient form has an important late-training property the
    decoupled (AdamW) form lacks: when grad → 0, v_hat → (W·param)², so
    the per-step update magnitude caps at LR·sign(param) regardless of W.
    AdamW's decoupled decay continues at rate LR·W·param indefinitely,
    which over-decays small late-training gradients and bleeds the network
    to zero weights — exactly what we observed for MuZero CartPole prior
    to this commit (see docs/MUZERO_AUDIT.md).

    STATE_PER_PARAM = 2:
        - state[i, 0] = m (first moment)
        - state[i, 1] = v (second moment)

    GLOBAL_STATE_SIZE = 2:
        - slot 0: UInt32 step counter (bit-patterned). A preamble kernel
          bumps it on-device inside `step_gpu` so bias correction survives
          CUDA-graph replay (Phase 4 of docs/STATE_SIZE_DESIGN.md).
        - slot 1: Scalar[dtype] lr_scale, written by GPUNetworkState.set_lr_scale
          and read by the kernel each step so LR schedules survive
          CUDA-graph replay.
    The CPU `step()` path keeps using the host `step_num` parameter directly.

    Hyperparameters are compile-time struct parameters.
    """

    comptime STATE_PER_PARAM: Int = 2
    comptime GLOBAL_STATE_SIZE: Int = 2

    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    def __init__(out self, *, deinit move: Self):
        pass

    @staticmethod
    def step[
        PARAM_SIZE: Int, dtype: DType = DType.float32
    ](
        params: Tile[dtype, PARAM_SIZE],
        grads: Tile[dtype, PARAM_SIZE],
        state: Tile[dtype, PARAM_SIZE, Self.STATE_PER_PARAM],
        opt_global_state: Tile[dtype, Self.GLOBAL_STATE_SIZE],
        step_num: Int,
        lr_scale: Float64 = 1.0,
    ):
        """Adam update step.

        Args:
            params: Parameters to update.
            grads: Gradients.
            state: Optimizer state layout `(PARAM_SIZE, 2)`: m at col 0, v at col 1.
            opt_global_state: Unused on the CPU path (the device counter is
                only consulted by `step_gpu`). Required by the trait signature.
            step_num: Current step (1-based), used for bias correction.
            lr_scale: Multiplicative LR scale (default 1.0). Set < 1.0 for LR annealing.
        """
        var bias_correction1 = Scalar[dtype](1.0 - (Self.BETA1**step_num))
        var bias_correction2 = Scalar[dtype](1.0 - (Self.BETA2**step_num))
        var one_minus_beta1 = Scalar[dtype](1.0 - Self.BETA1)
        var one_minus_beta2 = Scalar[dtype](1.0 - Self.BETA2)
        var beta1 = Scalar[dtype](Self.BETA1)
        var beta2 = Scalar[dtype](Self.BETA2)
        var lr = Scalar[dtype](Self.LR * lr_scale)
        var eps = Scalar[dtype](Self.EPS)
        var wd = Scalar[dtype](Self.WEIGHT_DECAY)

        # SIMD path: state layout is AoS [m0, v0, m1, v1, ...] (row_major
        # (PARAM_SIZE, 2)). Load 2W consecutive scalars per chunk, deinterleave
        # into (m_W, v_W) lanes, compute, interleave back.
        comptime W = _CPU_SIMD_W
        var p_p = params.ptr
        var g_p = grads.ptr
        var s_p = state.ptr
        var bc1_v = SIMD[dtype, W](bias_correction1)
        var bc2_v = SIMD[dtype, W](bias_correction2)
        var omb1_v = SIMD[dtype, W](one_minus_beta1)
        var omb2_v = SIMD[dtype, W](one_minus_beta2)
        var b1_v = SIMD[dtype, W](beta1)
        var b2_v = SIMD[dtype, W](beta2)
        var lr_v = SIMD[dtype, W](lr)
        var eps_v = SIMD[dtype, W](eps)
        var wd_v = SIMD[dtype, W](wd)
        var i = 0
        while i + W <= PARAM_SIZE:
            var p = p_p.unsafe_load[width=W](i)
            var g = g_p.unsafe_load[width=W](i) + wd_v * p
            var mv = s_p.unsafe_load[width=2 * W](2 * i).deinterleave()
            var m = rebind[SIMD[dtype, W]](mv[0])
            var v = rebind[SIMD[dtype, W]](mv[1])
            var m_new = b1_v * m + omb1_v * g
            var v_new = b2_v * v + omb2_v * g * g
            s_p.unsafe_store(2 * i, m_new.interleave(v_new))
            var m_hat = m_new / bc1_v
            var v_hat = v_new / bc2_v
            p_p.unsafe_store(i, p - lr_v * m_hat / (sqrt(v_hat) + eps_v))
            i += W
        while i < PARAM_SIZE:
            var g = grads[i] + wd * params[i]
            var m = state[i, 0]
            var v = state[i, 1]
            var m_new = beta1 * m + one_minus_beta1 * g
            var v_new = beta2 * v + one_minus_beta2 * g * g
            state[i, 0] = m_new
            state[i, 1] = v_new
            var m_hat = m_new / bias_correction1
            var v_hat = v_new / bias_correction2
            params[i] -= lr * m_hat / (sqrt(v_hat) + eps)
            i += 1

    # =========================================================================
    # GPU kernel implementation
    # =========================================================================

    @always_inline
    @staticmethod
    def step_kernel_impl[
        PARAM_SIZE: Int, dtype: DType = DType.float32
    ](
        params: Tile[dtype, PARAM_SIZE],
        grads: Tile[dtype, PARAM_SIZE],
        state: Tile[dtype, PARAM_SIZE, 2],
        counter: Tile[DType.uint32, 1],
        lr_scale_view: Tile[dtype, 1],
        base_lr: Scalar[dtype],
        beta1: Scalar[dtype],
        beta2: Scalar[dtype],
        eps: Scalar[dtype],
        log_beta1: Scalar[dtype],
        log_beta2: Scalar[dtype],
    ):
        """Adam optimizer kernel.

        state layout: (PARAM_SIZE, 2) where state[i, 0] = m, state[i, 1] = v.

        Both the bias-correction step counter and `lr_scale` are read from
        device-side views so they stay fresh under CUDA-graph replay.
        """
        comptime assert dtype.is_floating_point(), "dtype must be floating point"
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= PARAM_SIZE:
            return

        var step_f = rebind[Scalar[DType.uint32]](counter[0]).cast[dtype]()
        var one = Scalar[dtype](1.0)
        var bc1 = one - exp(log_beta1 * step_f)
        var bc2 = one - exp(log_beta2 * step_f)
        var lr = base_lr * rebind[Scalar[dtype]](lr_scale_view[0])
        var wd = Scalar[dtype](Self.WEIGHT_DECAY)
        var p_val = rebind[Scalar[dtype]](params[idx])

        # L2-in-gradient: g = grad + WEIGHT_DECAY * param (PyTorch-style)
        var g = rebind[Scalar[dtype]](grads[idx]) + wd * p_val
        var m_val = rebind[Scalar[dtype]](state[idx, 0])
        var v_val = rebind[Scalar[dtype]](state[idx, 1])

        var m_new = beta1 * m_val + (one - beta1) * g
        var v_new = beta2 * v_val + (one - beta2) * g * g

        state[idx, 0] = m_new
        state[idx, 1] = v_new

        var m_hat = m_new / bc1
        var v_hat = v_new / bc2

        params[idx] = p_val - lr * m_hat / (sqrt(v_hat) + eps)

    # =========================================================================
    # GPU launcher
    # =========================================================================

    @staticmethod
    def step_gpu[
        PARAM_SIZE: Int, dtype: DType = DType.float32
    ](
        ctx: DeviceContext,
        params: Tile[dtype, PARAM_SIZE],
        grads: Tile[dtype, PARAM_SIZE],
        state: Tile[dtype, PARAM_SIZE, Self.STATE_PER_PARAM],
        opt_global_state: Tile[dtype, Self.GLOBAL_STATE_SIZE],
        step_num: Int,
    ) raises:
        """Launch Adam optimization step on GPU.

        Bumps the device-side step counter (preamble kernel) and then runs the
        Adam update kernel, which reads the post-bump counter for bias
        correction and `lr_scale` from `opt_global_state[1]`. Both reads
        survive CUDA-graph replay; the host `step_num` argument is preserved
        as a bridge but unused on this path.

        Args:
            ctx: GPU device context.
            params: Parameters [PARAM_SIZE] (modified in place).
            grads: Gradients [PARAM_SIZE].
            state: State [PARAM_SIZE, 2] (m and v moments).
            opt_global_state: Two-slot global state.
                Slot 0 bit-patterns a UInt32 step counter (bumped by the
                preamble kernel). Slot 1 holds `lr_scale: Scalar[dtype]`
                (managed by GPUNetworkState).
            step_num: Unused on the GPU path (kept for trait-signature compat
                with optimizers that still rely on a host counter).
        """
        var base_lr = Scalar[dtype](Self.LR)
        var beta1 = Scalar[dtype](Self.BETA1)
        var beta2 = Scalar[dtype](Self.BETA2)
        var eps = Scalar[dtype](Self.EPS)
        var log_beta1 = Scalar[dtype](log(Self.BETA1))
        var log_beta2 = Scalar[dtype](log(Self.BETA2))

        # Slot 0 (Float32) → bit-cast UInt32 step counter view.
        var counter_t = tile[1](opt_global_state.ptr.unsafe_bitcast[Scalar[DType.uint32]]())
        # Slot 1 → lr_scale view.
        var lr_scale_view = tile[1](opt_global_state.ptr.unsafe_offset(1))

        @always_inline
        def bump_kernel(
            c: Tile[DType.uint32, 1],
        ):
            if Int(thread_idx.x) == 0:
                c[0] = c[0] + UInt32(1)

        ctx.enqueue_function[bump_kernel](
            counter_t,
            grid_dim=(1,),
            block_dim=(1,),
        )

        @always_inline
        def kernel_wrapper(
            params: Tile[dtype, PARAM_SIZE],
            grads: Tile[dtype, PARAM_SIZE],
            state: Tile[dtype, PARAM_SIZE, 2],
            counter: Tile[DType.uint32, 1],
            lr_scale_view: Tile[dtype, 1],
            base_lr: Scalar[dtype],
            beta1: Scalar[dtype],
            beta2: Scalar[dtype],
            eps: Scalar[dtype],
            log_beta1: Scalar[dtype],
            log_beta2: Scalar[dtype],
        ):
            Self.step_kernel_impl[PARAM_SIZE, dtype](
                params,
                grads,
                state,
                counter,
                lr_scale_view,
                base_lr,
                beta1,
                beta2,
                eps,
                log_beta1,
                log_beta2,
            )

        comptime grid_size = (PARAM_SIZE + TPB - 1) // TPB

        ctx.enqueue_function[kernel_wrapper](
            params,
            grads,
            state,
            counter_t,
            lr_scale_view,
            base_lr,
            beta1,
            beta2,
            eps,
            log_beta1,
            log_beta2,
            grid_dim=(grid_size,),
            block_dim=(TPB,),
        )


struct PCAdamW[
    LR: Float64 = 0.001,
    BETA1: Float64 = 0.9,
    BETA2: Float64 = 0.999,
    EPS: Float64 = 1e-8,
    WEIGHT_DECAY: Float64 = 0.01,
]:
    """AdamW optimizer - Adam with decoupled weight decay.

    The key difference from Adam: weight decay is applied directly to parameters,
    not through the gradient. This leads to better generalization.

    Update rule:
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad^2
        m_hat = m / (1 - beta1^step)
        v_hat = v / (1 - beta2^step)
        param = param * (1 - lr * weight_decay) - lr * m_hat / (sqrt(v_hat) + eps)

    STATE_PER_PARAM = 2:
        - state[i, 0] = m (first moment)
        - state[i, 1] = v (second moment)

    GLOBAL_STATE_SIZE = 2:
        - slot 0: UInt32 step counter (bit-patterned), bumped by a preamble
          kernel inside `step_gpu` (Phase 4 of docs/STATE_SIZE_DESIGN.md).
        - slot 1: Scalar[dtype] lr_scale, written by GPUNetworkState.set_lr_scale
          and read by the kernel each step so LR schedules survive
          CUDA-graph replay.
    The CPU `step()` path keeps using the host `step_num` parameter directly.

    All hyperparameters are compile-time struct parameters.
    """

    comptime STATE_PER_PARAM: Int = 2
    comptime GLOBAL_STATE_SIZE: Int = 2

    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    def __init__(out self, *, deinit move: Self):
        pass

    @staticmethod
    def step[
        PARAM_SIZE: Int, dtype: DType = DType.float32
    ](
        params: Tile[dtype, PARAM_SIZE],
        grads: Tile[dtype, PARAM_SIZE],
        state: Tile[dtype, PARAM_SIZE, Self.STATE_PER_PARAM],
        opt_global_state: Tile[dtype, Self.GLOBAL_STATE_SIZE],
        step_num: Int,
        lr_scale: Float64 = 1.0,
    ):
        """AdamW update step."""
        var bias_correction1 = Scalar[dtype](1.0 - (Self.BETA1**step_num))
        var bias_correction2 = Scalar[dtype](1.0 - (Self.BETA2**step_num))
        var one_minus_beta1 = Scalar[dtype](1.0 - Self.BETA1)
        var one_minus_beta2 = Scalar[dtype](1.0 - Self.BETA2)
        var beta1 = Scalar[dtype](Self.BETA1)
        var beta2 = Scalar[dtype](Self.BETA2)
        var lr = Scalar[dtype](Self.LR * lr_scale)
        var eps = Scalar[dtype](Self.EPS)
        var wd_factor = Scalar[dtype](1.0 - Self.LR * Self.WEIGHT_DECAY)

        comptime W = _CPU_SIMD_W
        var p_p = params.ptr
        var g_p = grads.ptr
        var s_p = state.ptr
        var bc1_v = SIMD[dtype, W](bias_correction1)
        var bc2_v = SIMD[dtype, W](bias_correction2)
        var omb1_v = SIMD[dtype, W](one_minus_beta1)
        var omb2_v = SIMD[dtype, W](one_minus_beta2)
        var b1_v = SIMD[dtype, W](beta1)
        var b2_v = SIMD[dtype, W](beta2)
        var lr_v = SIMD[dtype, W](lr)
        var eps_v = SIMD[dtype, W](eps)
        var wdf_v = SIMD[dtype, W](wd_factor)
        var i = 0
        while i + W <= PARAM_SIZE:
            var g = g_p.unsafe_load[width=W](i)
            var mv = s_p.unsafe_load[width=2 * W](2 * i).deinterleave()
            var m = rebind[SIMD[dtype, W]](mv[0])
            var v = rebind[SIMD[dtype, W]](mv[1])
            var m_new = b1_v * m + omb1_v * g
            var v_new = b2_v * v + omb2_v * g * g
            s_p.unsafe_store(2 * i, m_new.interleave(v_new))
            var m_hat = m_new / bc1_v
            var v_hat = v_new / bc2_v
            var p = p_p.unsafe_load[width=W](i)
            p_p.unsafe_store(i, p * wdf_v - lr_v * m_hat / (sqrt(v_hat) + eps_v))
            i += W
        while i < PARAM_SIZE:
            var g = grads[i]
            var m = state[i, 0]
            var v = state[i, 1]
            var m_new = beta1 * m + one_minus_beta1 * g
            var v_new = beta2 * v + one_minus_beta2 * g * g
            state[i, 0] = m_new
            state[i, 1] = v_new
            var m_hat = m_new / bias_correction1
            var v_hat = v_new / bias_correction2
            params[i] = params[i] * wd_factor - lr * m_hat / (
                sqrt(v_hat) + eps
            )
            i += 1

    # =========================================================================
    # GPU kernel implementation
    # =========================================================================

    @always_inline
    @staticmethod
    def step_kernel_impl[
        PARAM_SIZE: Int, dtype: DType = DType.float32
    ](
        params: Tile[dtype, PARAM_SIZE],
        grads: Tile[dtype, PARAM_SIZE],
        state: Tile[dtype, PARAM_SIZE, 2],
        counter: Tile[DType.uint32, 1],
        lr_scale_view: Tile[dtype, 1],
        base_lr: Scalar[dtype],
        beta1: Scalar[dtype],
        beta2: Scalar[dtype],
        eps: Scalar[dtype],
        log_beta1: Scalar[dtype],
        log_beta2: Scalar[dtype],
        wd_factor: Scalar[dtype],
    ):
        """AdamW kernel. Bias corrections are computed inside the kernel from
        the post-bump device counter and `lr_scale` is read from a 1-element
        device view so both survive CUDA-graph replay.
        """
        comptime assert dtype.is_floating_point(), "dtype must be floating point"
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= PARAM_SIZE:
            return

        var step_f = rebind[Scalar[DType.uint32]](counter[0]).cast[dtype]()
        var one = Scalar[dtype](1.0)
        var bc1 = one - exp(log_beta1 * step_f)
        var bc2 = one - exp(log_beta2 * step_f)
        var lr = base_lr * rebind[Scalar[dtype]](lr_scale_view[0])

        var g = rebind[Scalar[dtype]](grads[idx])
        var m_val = rebind[Scalar[dtype]](state[idx, 0])
        var v_val = rebind[Scalar[dtype]](state[idx, 1])

        var m_new = beta1 * m_val + (one - beta1) * g
        var v_new = beta2 * v_val + (one - beta2) * g * g

        state[idx, 0] = m_new
        state[idx, 1] = v_new

        var m_hat = m_new / bc1
        var v_hat = v_new / bc2

        var p = rebind[Scalar[dtype]](params[idx])
        params[idx] = p * wd_factor - lr * m_hat / (sqrt(v_hat) + eps)

    # =========================================================================
    # GPU launcher
    # =========================================================================

    @staticmethod
    def step_gpu[
        PARAM_SIZE: Int, dtype: DType = DType.float32
    ](
        ctx: DeviceContext,
        params: Tile[dtype, PARAM_SIZE],
        grads: Tile[dtype, PARAM_SIZE],
        state: Tile[dtype, PARAM_SIZE, Self.STATE_PER_PARAM],
        opt_global_state: Tile[dtype, Self.GLOBAL_STATE_SIZE],
        step_num: Int,
    ) raises:
        """Launch AdamW optimization step on GPU. See Adam.step_gpu — same
        preamble-bump-then-step pattern with `lr_scale` read from
        opt_global_state[1]; `step_num` is unused on this path.
        """
        var base_lr = Scalar[dtype](Self.LR)
        var beta1 = Scalar[dtype](Self.BETA1)
        var beta2 = Scalar[dtype](Self.BETA2)
        var eps = Scalar[dtype](Self.EPS)
        var log_beta1 = Scalar[dtype](log(Self.BETA1))
        var log_beta2 = Scalar[dtype](log(Self.BETA2))
        var wd_factor = Scalar[dtype](1.0 - Self.LR * Self.WEIGHT_DECAY)

        # Slot 0 → UInt32 step counter view; slot 1 → lr_scale view.
        var counter_t = tile[1](opt_global_state.ptr.unsafe_bitcast[Scalar[DType.uint32]]())
        var lr_scale_view = tile[1](opt_global_state.ptr.unsafe_offset(1))

        @always_inline
        def bump_kernel(
            c: Tile[DType.uint32, 1],
        ):
            if Int(thread_idx.x) == 0:
                c[0] = c[0] + UInt32(1)

        ctx.enqueue_function[bump_kernel](
            counter_t,
            grid_dim=(1,),
            block_dim=(1,),
        )

        @always_inline
        def kernel_wrapper(
            params: Tile[dtype, PARAM_SIZE],
            grads: Tile[dtype, PARAM_SIZE],
            state: Tile[dtype, PARAM_SIZE, 2],
            counter: Tile[DType.uint32, 1],
            lr_scale_view: Tile[dtype, 1],
            base_lr: Scalar[dtype],
            beta1: Scalar[dtype],
            beta2: Scalar[dtype],
            eps: Scalar[dtype],
            log_beta1: Scalar[dtype],
            log_beta2: Scalar[dtype],
            wd_factor: Scalar[dtype],
        ):
            Self.step_kernel_impl[PARAM_SIZE, dtype](
                params,
                grads,
                state,
                counter,
                lr_scale_view,
                base_lr,
                beta1,
                beta2,
                eps,
                log_beta1,
                log_beta2,
                wd_factor,
            )

        comptime grid_size = (PARAM_SIZE + TPB - 1) // TPB

        ctx.enqueue_function[kernel_wrapper](
            params,
            grads,
            state,
            counter_t,
            lr_scale_view,
            base_lr,
            beta1,
            beta2,
            eps,
            log_beta1,
            log_beta2,
            wd_factor,
            grid_dim=(grid_size,),
            block_dim=(TPB,),
        )
