"""The SmolVLA int4 text tower against its fp32 twin over deq(W), on a GPU
(`q4_tower_common.mojo` says what is compared and why): the dequantize path on
Metal, MAX's int4 GEMM on NVIDIA.

    pixi run -e apple mojo run -I . tests/deep_agents/smolvla/test_q4_tower_gpu.mojo
    pixi run -e jetson mojo run -I . --target-accelerator sm_87 \\
        tests/deep_agents/smolvla/test_q4_tower_gpu.mojo
"""

from max.gpu.host import DeviceContext

from tests.deep_agents.smolvla.q4_tower_common import run_q4_tower


def main() raises:
    var ctx = DeviceContext()
    print("SmolVLA int4 text tower vs fp32 over deq(W) on", ctx.name())
    run_q4_tower["gpu"](Optional[DeviceContext](ctx))
    print("PASSED")
