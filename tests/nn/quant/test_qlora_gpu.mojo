"""QLoRA on a GPU (`qlora_common.mojo` says what and why), SmolVLA's four
text shapes at the prefill's T = 135, rank 16.

    pixi run -e apple  mojo run -I . tests/nn/quant/test_qlora_gpu.mojo
    pixi run -e jetson mojo run -I . --target-accelerator sm_87 \\
        tests/nn/quant/test_qlora_gpu.mojo

Bands: Metal runs fp32 GEMMs on both sides (1e-5). On NVIDIA the forward is
MAX's int4 GEMM with bf16 activations against an fp32 GEMM, and both
grad_inputs and adapter gradients are TF32 cuBLAS GEMMs: 1e-2.
"""

from std.sys import default_accelerator
from std.testing import assert_true
from max.gpu.host import DeviceContext

from tests.nn.quant.qlora_common import check_qlora

comptime TOL = 1e-2 if default_accelerator().is_nvidia_gpu() else 1e-5


def main() raises:
    var ctx = DeviceContext()
    var c = Optional[DeviceContext](ctx)
    print("QLoRA on", ctx.name(), "band", TOL)
    var vs = [
        check_qlora["gpu", 960, 960, 135, 16](c, String("q / o   ")),
        check_qlora["gpu", 960, 320, 135, 16](c, String("k / v   ")),
        check_qlora["gpu", 960, 2560, 135, 16](c, String("gate/up ")),
        check_qlora["gpu", 2560, 960, 135, 16](c, String("down    ")),
    ]
    for v in vs:
        assert_true(v.b0_fwd_diff == 0 and v.b0_dx_diff == 0, "B = 0 is not the bare QLinear")
        assert_true(v.grad_mag > 0, "the reference's dA or dB is zero: the gradient rows are vacuous")
        assert_true(v.fwd < TOL and v.dx < TOL and v.da < TOL and v.db < TOL,
                    "QLoRA differs from LoRA over deq(W)")
    print("PASSED")
