"""The Newton line search's derivative threshold is floored at float32 only.

    pixi run mojo run -I . tests/physics3d/test_ls_gtol_dtype_floor.mojo

`primal.ls_gtol_dtype_floor` is the one rule every Newton line search calls
after its alpha=0 evaluation (the per-env pyramidal `pyramidal_linesearch`,
the per-env elliptic leg and the blocked kernel). Why it exists and what it
bought is in `primal.LS_SLOPE_DROP_F32`'s note: at float32 MuJoCo's threshold
sits below the dtype's resolution and the search brackets to its budget.

What is pinned here is the contract, not the speed:
- float64 is returned UNTOUCHED, bit for bit, so no MuJoCo-parity gate (all
  float64) can move;
- float32 is floored at `LS_SLOPE_DROP_F32 * |d0(0)|`, whatever the sign of
  the slope;
- the floor only RAISES — a threshold already above it is kept.
"""

from std.testing import assert_equal, assert_true, TestSuite

from noeira.physics3d.solver.primal import (
    ls_gtol_dtype_floor, LS_SLOPE_DROP_F32,
)


def test_float64_is_untouched() raises:
    var g = Scalar[DType.float64](1.0e-12)
    # A slope whose floor would be 1e-5, seven orders above `g`.
    var out = ls_gtol_dtype_floor[DType.float64](g, Scalar[DType.float64](-1.0))
    assert_equal(out, g)


def test_float32_is_floored_at_the_slope_ratio() raises:
    var g = Scalar[DType.float32](1.0e-10)
    var d0 = Scalar[DType.float32](-3.0)
    var out = ls_gtol_dtype_floor[DType.float32](g, d0)
    assert_equal(out, Scalar[DType.float32](LS_SLOPE_DROP_F32) * Scalar[DType.float32](3.0))
    # The sign of the slope does not matter: it is a magnitude.
    assert_equal(ls_gtol_dtype_floor[DType.float32](g, -d0), out)


def test_float32_floor_only_raises() raises:
    var g = Scalar[DType.float32](1.0)
    var out = ls_gtol_dtype_floor[DType.float32](g, Scalar[DType.float32](-2.0))
    assert_equal(out, g)
    assert_true(
        Scalar[DType.float32](LS_SLOPE_DROP_F32) * Scalar[DType.float32](2.0) < g,
        "fixture: the floor must sit below the threshold it is meant to keep",
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
