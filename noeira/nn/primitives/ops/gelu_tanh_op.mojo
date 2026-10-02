"""GELUTanhOp — `ElementOp` for the GELU activation (tanh approximation).

Matches `jax.nn.gelu(x, approximate=True)`, which is what the DreamerV3
reference `embodied/jax/nets.py:act('gelu')` resolves to:

    c   = sqrt(2/π)
    u   = c · (x + 0.044715 · x³)
    y   = 0.5 · x · (1 + tanh(u))

Backward (input-cache, `owns_cache=False` — needs original x):

    t      = tanh(u)
    dy/dx  = 0.5·(1 + t) + 0.5·x·(1 - t²)·c·(1 + 3·0.044715·x²)
           = 0.5·(1 + t) + 0.5·x·(1 - t²)·c·(1 + 0.134145·x²)

Used by the RSSM / Encoder / Decoder trunks (`Linear → Norm → GELU`).
"""

from std.math import tanh

from ...constants import DT
from ...core.element_op import ElementOp


# sqrt(2/π) and 3·0.044715 as DT-precision constants.
comptime _GELU_C: Scalar[DT] = 0.7978845608028654
comptime _GELU_A: Scalar[DT] = 0.044715
comptime _GELU_3A: Scalar[DT] = 0.134145


struct GELUTanhOp(ElementOp):
    """GELU (tanh approximation) with input-cache backward."""

    comptime owns_cache = False

    @staticmethod
    def display_label() -> String:
        return String("GELUTanh")

    @staticmethod
    def forward_scalar(x: Scalar[DT]) -> Scalar[DT]:
        var u = _GELU_C * (x + _GELU_A * x * x * x)
        return Scalar[DT](0.5) * x * (Scalar[DT](1.0) + tanh(u))

    @staticmethod
    def forward_simd[W: Int](x: SIMD[DT, W]) -> SIMD[DT, W]:
        var half = SIMD[DT, W](0.5)
        var one = SIMD[DT, W](1.0)
        var c = SIMD[DT, W](_GELU_C)
        var a = SIMD[DT, W](_GELU_A)
        var u = c * (x + a * x * x * x)
        return half * x * (one + tanh(u))

    @staticmethod
    def backward_scalar(c: Scalar[DT], go: Scalar[DT]) -> Scalar[DT]:
        # c is the cached INPUT (x).
        var x = c
        var u = _GELU_C * (x + _GELU_A * x * x * x)
        var t = tanh(u)
        var half = Scalar[DT](0.5)
        var one = Scalar[DT](1.0)
        var sech2 = one - t * t
        var dgelu = (
            half * (one + t)
            + half * x * sech2 * _GELU_C * (one + _GELU_3A * x * x)
        )
        return go * dgelu

    @staticmethod
    def backward_simd[W: Int](
        c: SIMD[DT, W], go: SIMD[DT, W]
    ) -> SIMD[DT, W]:
        var x = c
        var half = SIMD[DT, W](0.5)
        var one = SIMD[DT, W](1.0)
        var cc = SIMD[DT, W](_GELU_C)
        var a = SIMD[DT, W](_GELU_A)
        var a3 = SIMD[DT, W](_GELU_3A)
        var u = cc * (x + a * x * x * x)
        var t = tanh(u)
        var sech2 = one - t * t
        var dgelu = half * (one + t) + half * x * sech2 * cc * (one + a3 * x * x)
        return go * dgelu
