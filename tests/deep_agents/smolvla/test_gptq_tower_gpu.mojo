"""Sequential GPTQ calibration of a 2-layer SmolVLA text tower on a GPU
(`gptq_tower_common.mojo` says what is checked and why): the dequantize path
on Metal, MAX's int4 GEMM for the loaded int4 tower on NVIDIA.

    pixi run -e apple mojo run -I . tests/deep_agents/smolvla/test_gptq_tower_gpu.mojo
    pixi run -e jetson mojo run -I . --target-accelerator sm_87 \\
        tests/deep_agents/smolvla/test_gptq_tower_gpu.mojo
"""

from max.gpu.host import DeviceContext

from tests.deep_agents.smolvla.gptq_tower_common import run_gptq_tower


def main() raises:
    var ctx = DeviceContext()
    print("SmolVLA text tower, sequential GPTQ calibration on", ctx.name())
    run_gptq_tower["gpu"](Optional[DeviceContext](ctx))
    print("PASSED")
