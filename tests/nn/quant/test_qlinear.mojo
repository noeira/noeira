"""QLinear against a Linear holding the same dequantized weight (CPU).

    pixi run mojo run -I . tests/nn/quant/test_qlinear.mojo

At SmolVLA's four text shapes (q / o, k / v, gate / up, down): forward and
grad_input agree to fp32 rounding (both are fp32 GEMMs over deq(W) here), and
the forward matches the fp64 product x deq(W)ᵀ. The int4 GEMM's comparison is
the GPU gate's.
"""

from std.testing import assert_true

from tests.nn.quant.qlinear_common import check_shape

comptime TOL = 1e-5


def main() raises:
    print("QLinear vs Linear holding deq(W) (CPU)")

    var worst = Float64(0)
    var v0 = check_shape["cpu", 960, 960, 7](None, String("q / o   "))
    var v1 = check_shape["cpu", 960, 320, 7](None, String("k / v   "))
    var v2 = check_shape["cpu", 960, 2560, 7](None, String("gate/up "))
    var v3 = check_shape["cpu", 2560, 960, 7](None, String("down    "))
    for v in [v0.copy(), v1.copy(), v2.copy(), v3.copy()]:
        worst = max(worst, max(v.fwd, max(v.dx, v.fwd_vs_exact)))
    print("  worst relative difference:", worst, "(band", TOL, ")")
    assert_true(worst < TOL, "QLinear differs from Linear over deq(W)")
    print("PASSED")
