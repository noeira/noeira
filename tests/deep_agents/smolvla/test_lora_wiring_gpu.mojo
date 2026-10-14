# +--------------------------------------------------------------------------+ #
# | SmolVLA LoRA wiring: an adapted model against its merged twin (GPU)
# +--------------------------------------------------------------------------+ #
"""`test_lora_wiring.mojo` on the GPU: the adapters' device path inside the
fused loop (`lora_wiring_common.mojo`).

    pixi run -e apple  mojo run -I . tests/deep_agents/smolvla/test_lora_wiring_gpu.mojo
    pixi run -e nvidia mojo run -I . tests/deep_agents/smolvla/test_lora_wiring_gpu.mojo
"""

from max.gpu.host import DeviceContext

from tests.deep_agents.smolvla.lora_wiring_common import run_wiring


def main() raises:
    print("SmolVLA LoRA wiring, adapted vs merged (GPU)")
    var c = DeviceContext()
    run_wiring["gpu"](Optional(c))
    print("PASSED")
