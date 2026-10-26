"""The SmolVLA QLoRA text tower against LoRA over deq(W) on the
CPU (`qlora_tower_common.mojo` says what is compared and why).

    pixi run mojo run -I . tests/deep_agents/smolvla/test_qlora_tower.mojo
"""

from max.gpu.host import DeviceContext

from tests.deep_agents.smolvla.qlora_tower_common import run_qlora_tower


def main() raises:
    print("SmolVLA QLoRA text tower vs LoRA over deq(W)")
    run_qlora_tower["cpu"](None)
    print("PASSED")
