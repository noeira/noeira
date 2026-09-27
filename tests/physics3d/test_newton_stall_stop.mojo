"""The Newton solve's float32 stall exit fires on a RUN of zero improvements only.

    pixi run mojo run -I . tests/physics3d/test_newton_stall_stop.mojo

`primal.newton_stall_stop` is the one rule the three Newton legs (per-env
pyramidal, per-env elliptic, the blocked kernel) call after each line search,
next to MuJoCo's `improvement > 0 && improvement < tolerance`. Why it exists
is in `primal.NEWTON_STALL_RUN_F32`'s note: at float32 a line search can take
a step too small to move the cost, and the solve then repeats that iteration
to its cap.

What is pinned is the contract:
- float64 NEVER stops on it, so no MuJoCo-parity gate (all float64) can move;
- float32 stops on the `NEWTON_STALL_RUN_F32`-th CONSECUTIVE zero, not before;
- any nonzero improvement — progress, or a cost that went UP — resets the run.
"""

from std.testing import assert_true, assert_false, TestSuite

from noeira.physics3d.solver.primal import (
    newton_stall_stop, NEWTON_STALL_RUN_F32,
)


def test_float64_never_stops() raises:
    var run = 0
    for _ in range(10):
        assert_false(
            newton_stall_stop[DType.float64](Scalar[DType.float64](0), run)
        )


def test_float32_stops_on_the_run_not_the_first_zero() raises:
    assert_true(NEWTON_STALL_RUN_F32 >= 2, "fixture: the run must be longer than one")
    var run = 0
    for _ in range(NEWTON_STALL_RUN_F32 - 1):
        assert_false(
            newton_stall_stop[DType.float32](Scalar[DType.float32](0), run),
            "stopped before the run was complete",
        )
    assert_true(
        newton_stall_stop[DType.float32](Scalar[DType.float32](0), run),
        "a full run of zero improvements must stop the solve",
    )


def test_float32_nonzero_resets_the_run() raises:
    var run = 0
    for _ in range(NEWTON_STALL_RUN_F32 - 1):
        _ = newton_stall_stop[DType.float32](Scalar[DType.float32](0), run)
    # progress in between: the next zero starts a new run
    assert_false(
        newton_stall_stop[DType.float32](Scalar[DType.float32](1.0e-3), run)
    )
    assert_false(
        newton_stall_stop[DType.float32](Scalar[DType.float32](0), run),
        "a zero after progress is the first of a new run",
    )
    # a cost that went UP is not a stall either
    var run2 = 0
    for _ in range(NEWTON_STALL_RUN_F32 - 1):
        _ = newton_stall_stop[DType.float32](Scalar[DType.float32](0), run2)
    assert_false(
        newton_stall_stop[DType.float32](Scalar[DType.float32](-0.25), run2)
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
