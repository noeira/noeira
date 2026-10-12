"""ELUOp — `ElementOp` for the ELU activation (alpha = 1), torch's `nn.ELU()`.

`y = x` for x > 0, `exp(x) - 1` otherwise;  `dy/dx = 1` for x > 0, `exp(x)`
otherwise.

Input-cache op (`owns_cache = False`): the backward branches on the sign of
the original `x`. rsl_rl's default actor / critic activation (RoboParty's
walker uses it), hence this op.
"""

from std.math import exp

from ...constants import DT
from ...core.element_op import ElementOp


struct ELUOp(ElementOp):
    """ELU, alpha 1, input-cache backward (`owns_cache=False`)."""

    comptime owns_cache = False

    @staticmethod
    def display_label() -> String:
        return String("ELU")

    @staticmethod
    def forward_scalar(x: Scalar[DT]) -> Scalar[DT]:
        return x if x > Scalar[DT](0.0) else exp(x) - Scalar[DT](1.0)

    @staticmethod
    def forward_simd[W: Int](x: SIMD[DT, W]) -> SIMD[DT, W]:
        var zero = SIMD[DT, W](0.0)
        return x.gt(zero).select(x, exp(x) - SIMD[DT, W](1.0))

    @staticmethod
    def backward_scalar(c: Scalar[DT], go: Scalar[DT]) -> Scalar[DT]:
        return go if c > Scalar[DT](0.0) else go * exp(c)

    @staticmethod
    def backward_simd[W: Int](
        c: SIMD[DT, W], go: SIMD[DT, W]
    ) -> SIMD[DT, W]:
        var zero = SIMD[DT, W](0.0)
        return c.gt(zero).select(go, go * exp(c))
