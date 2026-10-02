"""GELUOp — `ElementOp` for the EXACT (erf) GELU.

    y      = 0.5 · x · (1 + erf(x / √2))          = x · Φ(x)
    dy/dx  = Φ(x) + x · φ(x),   φ(x) = exp(-x²/2) / √(2π)

This is `torch.nn.GELU()` (approximate='none') and HuggingFace's `"gelu"`:
the activation of the LeWorldModel ViT encoder, its projectors and its
predictor FFNs. `GELUTanhOp` (gelu_tanh_op.mojo) is the TANH approximation — jax's
default, what DreamerV3 uses — and differs from this by up to ~5e-4 at
|x| ~ 2: far outside a float32 parity gate. Pick by the reference, never by
habit.

Backward reads the cached INPUT (`owns_cache=False`).
"""

from std.math import erf, exp

from ...constants import DT
from ...core.element_op import ElementOp


comptime _INV_SQRT2: Scalar[DT] = 0.7071067811865476
comptime _INV_SQRT_2PI: Scalar[DT] = 0.3989422804014327


struct GELUOp(ElementOp):
    """GELU (erf) with input-cache backward."""

    comptime owns_cache = False

    @staticmethod
    def display_label() -> String:
        return String("GELU")

    @staticmethod
    def forward_scalar(x: Scalar[DT]) -> Scalar[DT]:
        return Scalar[DT](0.5) * x * (Scalar[DT](1.0) + erf(x * _INV_SQRT2))

    @staticmethod
    def forward_simd[W: Int](x: SIMD[DT, W]) -> SIMD[DT, W]:
        return (
            SIMD[DT, W](0.5) * x
            * (SIMD[DT, W](1.0) + erf(x * SIMD[DT, W](_INV_SQRT2)))
        )

    @staticmethod
    def backward_scalar(c: Scalar[DT], go: Scalar[DT]) -> Scalar[DT]:
        var x = c
        var cdf = Scalar[DT](0.5) * (Scalar[DT](1.0) + erf(x * _INV_SQRT2))
        var pdf = _INV_SQRT_2PI * exp(Scalar[DT](-0.5) * x * x)
        return go * (cdf + x * pdf)

    @staticmethod
    def backward_simd[W: Int](
        c: SIMD[DT, W], go: SIMD[DT, W]
    ) -> SIMD[DT, W]:
        var x = c
        var cdf = SIMD[DT, W](0.5) * (
            SIMD[DT, W](1.0) + erf(x * SIMD[DT, W](_INV_SQRT2))
        )
        var pdf = SIMD[DT, W](_INV_SQRT_2PI) * exp(SIMD[DT, W](-0.5) * x * x)
        return go * (cdf + x * pdf)
