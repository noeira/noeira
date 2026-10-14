"""Sequential GPTQ calibration of a 2-layer SmolVLA text tower on the CPU
(`gptq_tower_common.mojo` says what is checked and why).

    pixi run mojo run -I . tests/deep_agents/smolvla/test_gptq_tower.mojo
"""

from max.gpu.host import DeviceContext

from tests.deep_agents.smolvla.gptq_tower_common import run_gptq_tower


def main() raises:
    print("SmolVLA text tower, sequential GPTQ calibration (CPU)")
    run_gptq_tower["cpu"](None)
    print("PASSED")
