"""PCN-local weight initializers (buffer-fill API).

Vendored during the nn re-architecture (Phase A) so PCN no longer depends
on `noeira.nn.initializer`. The buffer-fill signature
(`fill[SIZE, FAN_IN, FAN_OUT, dtype](ptr_view)`) is deliberately kept
identical to the legacy `nn.initializer.Initializer.init[...]` so the
existing per-block init logic (`PCBlock.initialize_params`) maps over with
no math change — only the trait name differs.

These are leaf utilities, not architecture: a small Xavier/Zeros pair is
all PCN's blocks use. RNG is stdlib `random_float64()` in [0, 1) mapped to
the legacy `(2u − 1)·limit` form (bit-identical mapping; only the
underlying PRNG differs from legacy Philox — seed via `std.random.seed`
for reproducibility).
"""

from std.math import sqrt
from std.random import random_float64
from noeira.core.tiles import Tile


trait PCInitializer(ImplicitlyCopyable):
    """Buffer-fill initializer for a flat [SIZE] parameter view.

    Mirrors the legacy `nn.initializer.Initializer` surface so PCN block
    init carries over unchanged. FAN_IN / FAN_OUT scope the variance; SIZE
    is the flat element count of the slice being filled (typically the W
    block `in*out`)."""

    @staticmethod
    def fill[
        SIZE: Int, FAN_IN: Int, FAN_OUT: Int, dtype: DType = DType.float32
    ](params: Tile[dtype, SIZE]):
        ...


struct PCXavier(PCInitializer):
    """Xavier/Glorot uniform: U(-limit, limit), limit = sqrt(6/(fan_in+fan_out)).

    Optimal for linear/tanh PC levels. Matches legacy `Xavier`'s scaling
    and `(2u − 1)·limit` mapping (the PRNG differs)."""

    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    def __init__(out self, *, deinit move: Self):
        pass

    @staticmethod
    def fill[
        SIZE: Int, FAN_IN: Int, FAN_OUT: Int, dtype: DType = DType.float32
    ](params: Tile[dtype, SIZE]):
        comptime assert (
            dtype.is_floating_point()
        ), "PCXavier requires floating-point dtype"
        var limit = sqrt(
            Scalar[dtype](6.0) / Scalar[dtype](FAN_IN + FAN_OUT)
        )
        for i in range(SIZE):
            var u = Scalar[dtype](random_float64())
            params[i] = (u * Scalar[dtype](2.0) - Scalar[dtype](1.0)) * limit


struct PCZeros(PCInitializer):
    """Fill with zeros (biases, or zero-init weight experiments)."""

    def __init__(out self):
        pass

    def __init__(out self, *, copy: Self):
        pass

    def __init__(out self, *, deinit move: Self):
        pass

    @staticmethod
    def fill[
        SIZE: Int, FAN_IN: Int, FAN_OUT: Int, dtype: DType = DType.float32
    ](params: Tile[dtype, SIZE]):
        for i in range(SIZE):
            params[i] = Scalar[dtype](0)
