"""PCN traits and activations — Bogacz canonical.

Direction: bottom-up. A PCBlock represents one PCN level:

    μ_above = W · act(x_below) + b              # prediction (bottom → top)
    ε_above = x_above − μ_above                  # local prediction error
    z_below = W^T · ε_above                      # pull-back through W^T
    dE/dx_below ⊃ −act'(x_below) ⊙ z_below       # contribution to x_below update

Convention (mirrors `nn.Linear[in, out]` shape):

  - W shape: [IN_DIM, OUT_DIM]
  - IN_DIM   = below side (predicting block reads this)
  - OUT_DIM  = above side (latent predicted by this block)
  - PARAM_SIZE = IN_DIM * OUT_DIM + OUT_DIM      (W flat | b)

Layers:
  - PCBlock[IN, OUT, ACT=PCReLU]   — one PCN level (W + b + bundled activation)
  - For the readout, use ACT=PCIdentity: PCBlock[HIDDEN_LAST, NUM_CLASSES, PCIdentity]

Activations:
  - PCReLU, PCIdentity, PCTanh, PCSwish (PCActivation trait — extensible).

The activation is bundled into the block (comptime param). It acts on the
*below* side: act(x_below) is what gets multiplied by W. For the readout,
PCIdentity means the readout output is W·x_last + b, directly compared to
target via output loss (no extra nonlinearity on top).
"""

from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceContext
from std.math import exp, tanh

from .pc_constants import TPB
from .pc_initializer import PCInitializer
from noeira.core.tiles import Tile


# =============================================================================
# Activation trait
# =============================================================================


trait PCActivation(ImplicitlyCopyable):
    """Activation function used inside a PCBlock.

    Provides CPU + GPU variants of:
      - apply: a = f(x)
      - apply_derivative_mul: out = z · f'(x)   (gain-modulated z by f'(x))

    For Bogacz canonical, `apply_derivative_mul` is evaluated AT x_below
    (the latent itself, not a pre-activation) — consistent with f'(x) for
    ReLU which uses `x > 0` as the mask.
    """

    @staticmethod
    def apply[
        BATCH: Int, DIM: Int, dtype: DType = DType.float32
    ](
        x: Tile[dtype, BATCH, DIM],
        a: Tile[dtype, BATCH, DIM],
    ):
        ...

    @staticmethod
    def apply_derivative_mul[
        BATCH: Int, DIM: Int, dtype: DType = DType.float32
    ](
        x: Tile[dtype, BATCH, DIM],
        z_in: Tile[dtype, BATCH, DIM],
        z_out: Tile[dtype, BATCH, DIM],
    ):
        ...

    @staticmethod
    def apply_gpu[
        BATCH: Int, DIM: Int, dtype: DType = DType.float32
    ](
        ctx: DeviceContext,
        x: Tile[dtype, BATCH, DIM],
        a: Tile[dtype, BATCH, DIM],
    ) raises:
        ...

    @staticmethod
    def apply_derivative_mul_gpu[
        BATCH: Int, DIM: Int, dtype: DType = DType.float32
    ](
        ctx: DeviceContext,
        x: Tile[dtype, BATCH, DIM],
        z_in: Tile[dtype, BATCH, DIM],
        z_out: Tile[dtype, BATCH, DIM],
    ) raises:
        ...


struct PCReLU(PCActivation):
    """ReLU: f(x) = max(0, x); f'(x) = 1 if x > 0 else 0."""

    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    def __init__(out self, *, deinit move: Self):
        pass

    @staticmethod
    def apply[
        BATCH: Int, DIM: Int, dtype: DType = DType.float32
    ](
        x: Tile[dtype, BATCH, DIM],
        a: Tile[dtype, BATCH, DIM],
    ):
        # ReLU elementwise via direct LayoutTensor indexing (no raw pointers).
        for b in range(BATCH):
            for i in range(DIM):
                var v = rebind[Scalar[dtype]](x[b, i])
                a[b, i] = v if v > 0 else Scalar[dtype](0)

    @staticmethod
    def apply_derivative_mul[
        BATCH: Int, DIM: Int, dtype: DType = DType.float32
    ](
        x: Tile[dtype, BATCH, DIM],
        z_in: Tile[dtype, BATCH, DIM],
        z_out: Tile[dtype, BATCH, DIM],
    ):
        # ReLU derivative gating via direct LayoutTensor indexing.
        for b in range(BATCH):
            for i in range(DIM):
                var xv = rebind[Scalar[dtype]](x[b, i])
                z_out[b, i] = rebind[Scalar[dtype]](
                    z_in[b, i]
                ) if xv > 0 else Scalar[dtype](0)

    # ── GPU kernels (naive: one thread per element) ──────────────────────────

    @staticmethod
    def _relu_apply_kernel[
        BATCH: Int,
        DIM: Int,
        dtype: DType,
    ](
        x: Tile[dtype, BATCH, DIM],
        a: Tile[dtype, BATCH, DIM],
    ):
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * DIM:
            return
        var b = idx // DIM
        var i = idx % DIM
        var v = rebind[Scalar[dtype]](x[b, i])
        a[b, i] = v if v > 0 else Scalar[dtype](0)

    @staticmethod
    def _relu_deriv_mul_kernel[
        BATCH: Int,
        DIM: Int,
        dtype: DType,
    ](
        x: Tile[dtype, BATCH, DIM],
        z_in: Tile[dtype, BATCH, DIM],
        z_out: Tile[dtype, BATCH, DIM],
    ):
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * DIM:
            return
        var b = idx // DIM
        var i = idx % DIM
        var xv = rebind[Scalar[dtype]](x[b, i])
        z_out[b, i] = rebind[Scalar[dtype]](z_in[b, i]) if xv > 0 else Scalar[
            dtype
        ](0)

    @staticmethod
    def apply_gpu[
        BATCH: Int, DIM: Int, dtype: DType = DType.float32
    ](
        ctx: DeviceContext,
        x: Tile[dtype, BATCH, DIM],
        a: Tile[dtype, BATCH, DIM],
    ) raises:
        comptime k = Self._relu_apply_kernel[BATCH, DIM, dtype]
        var threads = BATCH * DIM
        var blocks = (threads + TPB - 1) // TPB
        ctx.enqueue_function[k](x, a, grid_dim=(blocks,), block_dim=(TPB,))

    @staticmethod
    def apply_derivative_mul_gpu[
        BATCH: Int, DIM: Int, dtype: DType = DType.float32
    ](
        ctx: DeviceContext,
        x: Tile[dtype, BATCH, DIM],
        z_in: Tile[dtype, BATCH, DIM],
        z_out: Tile[dtype, BATCH, DIM],
    ) raises:
        comptime k = Self._relu_deriv_mul_kernel[BATCH, DIM, dtype]
        var threads = BATCH * DIM
        var blocks = (threads + TPB - 1) // TPB
        ctx.enqueue_function[k](
            x, z_in, z_out, grid_dim=(blocks,), block_dim=(TPB,)
        )


struct PCIdentity(PCActivation):
    """Identity: f(x) = x; f'(x) = 1.  Used for the readout block."""

    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    def __init__(out self, *, deinit move: Self):
        pass

    @staticmethod
    def apply[
        BATCH: Int, DIM: Int, dtype: DType = DType.float32
    ](
        x: Tile[dtype, BATCH, DIM],
        a: Tile[dtype, BATCH, DIM],
    ):
        # Identity: a = x (elementwise copy via direct LayoutTensor indexing).
        for b in range(BATCH):
            for i in range(DIM):
                a[b, i] = x[b, i]

    @staticmethod
    def apply_derivative_mul[
        BATCH: Int, DIM: Int, dtype: DType = DType.float32
    ](
        x: Tile[dtype, BATCH, DIM],
        z_in: Tile[dtype, BATCH, DIM],
        z_out: Tile[dtype, BATCH, DIM],
    ):
        # Identity derivative = 1: z_out = z_in (copy via LayoutTensor indexing).
        for b in range(BATCH):
            for i in range(DIM):
                z_out[b, i] = z_in[b, i]

    # ── GPU kernels (naive copy) ─────────────────────────────────────────────

    @staticmethod
    def _identity_copy_kernel[
        BATCH: Int,
        DIM: Int,
        dtype: DType,
    ](
        src: Tile[dtype, BATCH, DIM],
        dst: Tile[dtype, BATCH, DIM],
    ):
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * DIM:
            return
        var b = idx // DIM
        var i = idx % DIM
        dst[b, i] = src[b, i]

    @staticmethod
    def apply_gpu[
        BATCH: Int, DIM: Int, dtype: DType = DType.float32
    ](
        ctx: DeviceContext,
        x: Tile[dtype, BATCH, DIM],
        a: Tile[dtype, BATCH, DIM],
    ) raises:
        comptime k = Self._identity_copy_kernel[BATCH, DIM, dtype]
        var threads = BATCH * DIM
        var blocks = (threads + TPB - 1) // TPB
        ctx.enqueue_function[k](x, a, grid_dim=(blocks,), block_dim=(TPB,))

    @staticmethod
    def apply_derivative_mul_gpu[
        BATCH: Int, DIM: Int, dtype: DType = DType.float32
    ](
        ctx: DeviceContext,
        x: Tile[dtype, BATCH, DIM],
        z_in: Tile[dtype, BATCH, DIM],
        z_out: Tile[dtype, BATCH, DIM],
    ) raises:
        # Identity derivative = 1: z_out = z_in (copy)
        comptime k = Self._identity_copy_kernel[BATCH, DIM, dtype]
        var threads = BATCH * DIM
        var blocks = (threads + TPB - 1) // TPB
        ctx.enqueue_function[k](
            z_in, z_out, grid_dim=(blocks,), block_dim=(TPB,)
        )


struct PCTanh(PCActivation):
    """Tanh: f(x) = tanh(x); f'(x) = 1 − tanh²(x).

    Common in recurrent PC (tPC) where the post-activation feeds back as input
    to the next time step. Bounded output prevents the recurrent dynamics
    from drifting to infinity.
    """

    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    def __init__(out self, *, deinit move: Self):
        pass

    @staticmethod
    def apply[
        BATCH: Int, DIM: Int, dtype: DType = DType.float32
    ](
        x: Tile[dtype, BATCH, DIM],
        a: Tile[dtype, BATCH, DIM],
    ):
        comptime assert (
            dtype.is_floating_point()
        ), "PCTanh requires floating-point dtype"
        for b in range(BATCH):
            for i in range(DIM):
                var v = rebind[Scalar[dtype]](x[b, i])
                a[b, i] = tanh(v)

    @staticmethod
    def apply_derivative_mul[
        BATCH: Int, DIM: Int, dtype: DType = DType.float32
    ](
        x: Tile[dtype, BATCH, DIM],
        z_in: Tile[dtype, BATCH, DIM],
        z_out: Tile[dtype, BATCH, DIM],
    ):
        comptime assert (
            dtype.is_floating_point()
        ), "PCTanh requires floating-point dtype"
        for b in range(BATCH):
            for i in range(DIM):
                var t = tanh(rebind[Scalar[dtype]](x[b, i]))
                z_out[b, i] = rebind[Scalar[dtype]](z_in[b, i]) * (
                    Scalar[dtype](1) - t * t
                )

    # ── GPU kernels (naive: one thread per element) ──────────────────────────

    @staticmethod
    def _tanh_apply_kernel[
        BATCH: Int,
        DIM: Int,
        dtype: DType,
    ](
        x: Tile[dtype, BATCH, DIM],
        a: Tile[dtype, BATCH, DIM],
    ):
        comptime assert (
            dtype.is_floating_point()
        ), "PCTanh requires floating-point dtype"
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * DIM:
            return
        var b = idx // DIM
        var i = idx % DIM
        var v = rebind[Scalar[dtype]](x[b, i])
        a[b, i] = tanh(v)

    @staticmethod
    def _tanh_deriv_mul_kernel[
        BATCH: Int,
        DIM: Int,
        dtype: DType,
    ](
        x: Tile[dtype, BATCH, DIM],
        z_in: Tile[dtype, BATCH, DIM],
        z_out: Tile[dtype, BATCH, DIM],
    ):
        comptime assert (
            dtype.is_floating_point()
        ), "PCTanh requires floating-point dtype"
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * DIM:
            return
        var b = idx // DIM
        var i = idx % DIM
        var t = tanh(rebind[Scalar[dtype]](x[b, i]))
        z_out[b, i] = rebind[Scalar[dtype]](z_in[b, i]) * (
            Scalar[dtype](1) - t * t
        )

    @staticmethod
    def apply_gpu[
        BATCH: Int, DIM: Int, dtype: DType = DType.float32
    ](
        ctx: DeviceContext,
        x: Tile[dtype, BATCH, DIM],
        a: Tile[dtype, BATCH, DIM],
    ) raises:
        comptime k = Self._tanh_apply_kernel[BATCH, DIM, dtype]
        var threads = BATCH * DIM
        var blocks = (threads + TPB - 1) // TPB
        ctx.enqueue_function[k](x, a, grid_dim=(blocks,), block_dim=(TPB,))

    @staticmethod
    def apply_derivative_mul_gpu[
        BATCH: Int, DIM: Int, dtype: DType = DType.float32
    ](
        ctx: DeviceContext,
        x: Tile[dtype, BATCH, DIM],
        z_in: Tile[dtype, BATCH, DIM],
        z_out: Tile[dtype, BATCH, DIM],
    ) raises:
        comptime k = Self._tanh_deriv_mul_kernel[BATCH, DIM, dtype]
        var threads = BATCH * DIM
        var blocks = (threads + TPB - 1) // TPB
        ctx.enqueue_function[k](
            x, z_in, z_out, grid_dim=(blocks,), block_dim=(TPB,)
        )


# =============================================================================
# PCSwish (= SiLU)
# =============================================================================


struct PCSwish(PCActivation):
    """Swish / SiLU: f(x) = x · σ(x); f'(x) = σ(x) · (1 + x · (1 − σ(x))).

    Same activation vanilla MBPO uses for its 4-LinearSwish dynamics
    ensemble. Smooth (unlike ReLU), unbounded above (unlike PCTanh — won't
    saturate on large hidden activations), self-gating. Used by the
    Variant-B "activation match" experiment: hold depth at 5 and swap
    PCTanh → PCSwish to test whether activation choice closes the
    remaining MSE gap to vanilla MBPO.
    """

    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    def __init__(out self, *, deinit move: Self):
        pass

    @staticmethod
    def apply[
        BATCH: Int, DIM: Int, dtype: DType = DType.float32
    ](
        x: Tile[dtype, BATCH, DIM],
        a: Tile[dtype, BATCH, DIM],
    ):
        comptime assert (
            dtype.is_floating_point()
        ), "PCSwish requires floating-point dtype"
        for b in range(BATCH):
            for i in range(DIM):
                var v = rebind[Scalar[dtype]](x[b, i])
                var s = Scalar[dtype](1.0) / (Scalar[dtype](1.0) + exp(-v))
                a[b, i] = v * s

    @staticmethod
    def apply_derivative_mul[
        BATCH: Int, DIM: Int, dtype: DType = DType.float32
    ](
        x: Tile[dtype, BATCH, DIM],
        z_in: Tile[dtype, BATCH, DIM],
        z_out: Tile[dtype, BATCH, DIM],
    ):
        comptime assert (
            dtype.is_floating_point()
        ), "PCSwish requires floating-point dtype"
        for b in range(BATCH):
            for i in range(DIM):
                var v = rebind[Scalar[dtype]](x[b, i])
                var s = Scalar[dtype](1.0) / (Scalar[dtype](1.0) + exp(-v))
                var deriv = s * (
                    Scalar[dtype](1.0) + v * (Scalar[dtype](1.0) - s)
                )
                z_out[b, i] = rebind[Scalar[dtype]](z_in[b, i]) * deriv

    # ── GPU kernels (naive: one thread per element) ──────────────────────────

    @staticmethod
    def _swish_apply_kernel[
        BATCH: Int,
        DIM: Int,
        dtype: DType,
    ](
        x: Tile[dtype, BATCH, DIM],
        a: Tile[dtype, BATCH, DIM],
    ):
        comptime assert (
            dtype.is_floating_point()
        ), "PCSwish requires floating-point dtype"
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * DIM:
            return
        var b = idx // DIM
        var i = idx % DIM
        var v = rebind[Scalar[dtype]](x[b, i])
        var s = Scalar[dtype](1.0) / (Scalar[dtype](1.0) + exp(-v))
        a[b, i] = v * s

    @staticmethod
    def _swish_deriv_mul_kernel[
        BATCH: Int,
        DIM: Int,
        dtype: DType,
    ](
        x: Tile[dtype, BATCH, DIM],
        z_in: Tile[dtype, BATCH, DIM],
        z_out: Tile[dtype, BATCH, DIM],
    ):
        comptime assert (
            dtype.is_floating_point()
        ), "PCSwish requires floating-point dtype"
        var idx = Int(block_dim.x * block_idx.x + thread_idx.x)
        if idx >= BATCH * DIM:
            return
        var b = idx // DIM
        var i = idx % DIM
        var v = rebind[Scalar[dtype]](x[b, i])
        var s = Scalar[dtype](1.0) / (Scalar[dtype](1.0) + exp(-v))
        var deriv = s * (Scalar[dtype](1.0) + v * (Scalar[dtype](1.0) - s))
        z_out[b, i] = rebind[Scalar[dtype]](z_in[b, i]) * deriv

    @staticmethod
    def apply_gpu[
        BATCH: Int, DIM: Int, dtype: DType = DType.float32
    ](
        ctx: DeviceContext,
        x: Tile[dtype, BATCH, DIM],
        a: Tile[dtype, BATCH, DIM],
    ) raises:
        comptime k = Self._swish_apply_kernel[BATCH, DIM, dtype]
        var threads = BATCH * DIM
        var blocks = (threads + TPB - 1) // TPB
        ctx.enqueue_function[k](x, a, grid_dim=(blocks,), block_dim=(TPB,))

    @staticmethod
    def apply_derivative_mul_gpu[
        BATCH: Int, DIM: Int, dtype: DType = DType.float32
    ](
        ctx: DeviceContext,
        x: Tile[dtype, BATCH, DIM],
        z_in: Tile[dtype, BATCH, DIM],
        z_out: Tile[dtype, BATCH, DIM],
    ) raises:
        comptime k = Self._swish_deriv_mul_kernel[BATCH, DIM, dtype]
        var threads = BATCH * DIM
        var blocks = (threads + TPB - 1) // TPB
        ctx.enqueue_function[k](
            x, z_in, z_out, grid_dim=(blocks,), block_dim=(TPB,)
        )


# =============================================================================
# PCBlockTrait — composable element of a PCSequential
# =============================================================================


trait PCBlockTrait(ImplicitlyCopyable):
    """One PCN level (Bogacz canonical, bottom-up direction).

    A PCBlock owns a weight matrix W, a bias vector b, and a bundled
    activation. The block predicts the *above* latent from the *below* latent:

        μ_above = W · act(x_below) + b

    Compile-time constants:
      - IN_DIM:  below side (= dim of x_below, also rows of W)
      - OUT_DIM: above side (= dim of x_above predicted by this block, cols of W)
      - PARAM_SIZE: IN_DIM * OUT_DIM + OUT_DIM   (W flat | b)
    """

    comptime IN_DIM: Int
    comptime OUT_DIM: Int
    comptime PARAM_SIZE: Int

    @staticmethod
    def pc_init_params[
        INIT: PCInitializer, dtype: DType = DType.float32
    ](
        params: Tile[dtype, Self.PARAM_SIZE],
    ) raises:
        """Same contract as `initialize_params`
        but driven by the vendored `PCInitializer` (legacy-`nn`-free), so
        `PCModule.make_pcn` initializes any block type without the legacy
        `nn.initializer` trait. Default raises — every `PCBlockTrait`
        conformer used under `PCModule` overrides this."""
        raise Error(
            "pc_init_params: this PCBlockTrait conformer has no nn init"
            " override — add one to use it under PCModule."
        )

    @staticmethod
    def predict[
        BATCH: Int, dtype: DType = DType.float32
    ](
        x_below: Tile[dtype, BATCH, Self.IN_DIM],
        params: Tile[dtype, Self.PARAM_SIZE],
        mu: Tile[dtype, BATCH, Self.OUT_DIM],
        a_below: Tile[dtype, BATCH, Self.IN_DIM],
    ):
        ...

    @staticmethod
    def eps_compute[
        BATCH: Int, dtype: DType = DType.float32
    ](
        x_above: Tile[dtype, BATCH, Self.OUT_DIM],
        mu: Tile[dtype, BATCH, Self.OUT_DIM],
        eps: Tile[dtype, BATCH, Self.OUT_DIM],
    ):
        ...

    @staticmethod
    def pull_back[
        BATCH: Int, dtype: DType = DType.float32
    ](
        eps_above: Tile[dtype, BATCH, Self.OUT_DIM],
        params: Tile[dtype, Self.PARAM_SIZE],
        z_below: Tile[dtype, BATCH, Self.IN_DIM],
    ):
        ...

    @staticmethod
    def act_derivative_mul[
        BATCH: Int, dtype: DType = DType.float32
    ](
        x_below: Tile[dtype, BATCH, Self.IN_DIM],
        z_in: Tile[dtype, BATCH, Self.IN_DIM],
        z_out: Tile[dtype, BATCH, Self.IN_DIM],
    ):
        ...

    @staticmethod
    def weight_grad[
        BATCH: Int, dtype: DType = DType.float32
    ](
        eps_above: Tile[dtype, BATCH, Self.OUT_DIM],
        a_below: Tile[dtype, BATCH, Self.IN_DIM],
        grads: Tile[dtype, Self.PARAM_SIZE],
    ):
        ...

    # ── GPU dispatch declarations ────────────────────────────────────────────

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
        ...

    @staticmethod
    def eps_compute_gpu[
        BATCH: Int, dtype: DType = DType.float32
    ](
        ctx: DeviceContext,
        x_above: Tile[dtype, BATCH, Self.OUT_DIM],
        mu: Tile[dtype, BATCH, Self.OUT_DIM],
        eps: Tile[dtype, BATCH, Self.OUT_DIM],
    ) raises:
        ...

    @staticmethod
    def pull_back_gpu[
        BATCH: Int, dtype: DType = DType.float32
    ](
        ctx: DeviceContext,
        eps_above: Tile[dtype, BATCH, Self.OUT_DIM],
        params: Tile[dtype, Self.PARAM_SIZE],
        z_below: Tile[dtype, BATCH, Self.IN_DIM],
    ) raises:
        ...

    @staticmethod
    def act_derivative_mul_gpu[
        BATCH: Int, dtype: DType = DType.float32
    ](
        ctx: DeviceContext,
        x_below: Tile[dtype, BATCH, Self.IN_DIM],
        z_in: Tile[dtype, BATCH, Self.IN_DIM],
        z_out: Tile[dtype, BATCH, Self.IN_DIM],
    ) raises:
        ...

    @staticmethod
    def weight_grad_gpu[
        BATCH: Int, dtype: DType = DType.float32
    ](
        ctx: DeviceContext,
        eps_above: Tile[dtype, BATCH, Self.OUT_DIM],
        a_below: Tile[dtype, BATCH, Self.IN_DIM],
        grads: Tile[dtype, Self.PARAM_SIZE],
    ) raises:
        ...
