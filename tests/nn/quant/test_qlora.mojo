"""QLoRA on the CPU: `QLoRALinear` against the bare `QLinear` (B = 0) and
against LoRA over deq(W) (`qlora_common.mojo` says what and why).

    pixi run mojo run -I . tests/nn/quant/test_qlora.mojo
"""

from std.testing import assert_true

from tests.nn.quant.qlora_common import check_qlora


def main() raises:
    print("QLoRA (CPU): the dequantize path on both sides, band 1e-5")
    var vs = [
        check_qlora["cpu", 64, 96, 7, 4](None, String("small   ")),
        check_qlora["cpu", 960, 320, 7, 16](None, String("k / v   ")),
        check_qlora["cpu", 2560, 960, 7, 16](None, String("down    ")),
    ]
    for v in vs:
        assert_true(v.b0_fwd_diff == 0 and v.b0_dx_diff == 0, "B = 0 is not the bare QLinear")
        assert_true(v.grad_mag > 0, "the reference's dA or dB is zero: the gradient rows are vacuous")
        assert_true(v.fwd < 1e-5 and v.dx < 1e-5 and v.da < 1e-5 and v.db < 1e-5,
                    "QLoRA differs from LoRA over deq(W)")
    print("PASSED")
