"""ReLUOp — `ElementOp` for the rectified linear activation.

Input-cache op: `owns_cache = False`. Forward writes `y = max(x, 0)` to
output; the orchestrator's input slab survives until backward, so the
backward kernel reads `x` back as `c` and gates the gradient with the
sign of the cached input. Mirrors the math the hand-written
`ReLU[DIM]` leaf used pre-Phase-2 migration.

The SIMD path uses `.gt()` to obtain a lanewise Bool mask + `.select()`
to gate; see `feedback_simd_gt_scalar_bool` for why bare `>` returns a
scalar all-equal Bool in Mojo nightly.
"""

from ...constants import DT
from ...core.element_op import ElementOp


struct ReLUOp(ElementOp):
    """ReLU activation with input-cache backward (`owns_cache=False`)."""

    comptime owns_cache = False
    comptime is_relu = True

    @staticmethod
    def display_label() -> String:
        return String("ReLU")

    @staticmethod
    def forward_scalar(x: Scalar[DT]) -> Scalar[DT]:
        var zero: Scalar[DT] = 0.0
        return x if x > zero else zero

    @staticmethod
    def forward_simd[W: Int](x: SIMD[DT, W]) -> SIMD[DT, W]:
        var zero = SIMD[DT, W](0.0)
        return x.gt(zero).select(x, zero)

    @staticmethod
    def backward_scalar(c: Scalar[DT], go: Scalar[DT]) -> Scalar[DT]:
        # c is the cached INPUT (x). dy/dx = 1 if x > 0 else 0.
        var zero: Scalar[DT] = 0.0
        return go if c > zero else zero

    @staticmethod
    def backward_simd[W: Int](
        c: SIMD[DT, W], go: SIMD[DT, W]
    ) -> SIMD[DT, W]:
        var zero = SIMD[DT, W](0.0)
        return c.gt(zero).select(go, zero)
