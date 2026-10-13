"""The SmolVLA QLoRA text tower against LoRA over deq(W) on a GPU: the
dequantize path on Metal, MAX's int4 GEMM on NVIDIA (`qlora_tower_common.mojo`
says what is compared and why).

    pixi run -e apple mojo run -I . tests/deep_agents/smolvla/test_qlora_tower_gpu.mojo
    pixi run -e jetson mojo run -I . --target-accelerator sm_87 \\
        tests/deep_agents/smolvla/test_qlora_tower_gpu.mojo
"""

from max.gpu.host import DeviceContext

from tests.deep_agents.smolvla.qlora_tower_common import run_qlora_tower


def main() raises:
    print("SmolVLA QLoRA text tower vs LoRA over deq(W)")
    var ctx = DeviceContext()
    run_qlora_tower["gpu"](Optional[DeviceContext](ctx))
    print("PASSED")
