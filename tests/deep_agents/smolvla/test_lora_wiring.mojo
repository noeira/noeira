# +--------------------------------------------------------------------------+ #
# | SmolVLA LoRA wiring: an adapted model against its merged twin (CPU)
# +--------------------------------------------------------------------------+ #
"""Every adapter of both towers, through prefill, cache, denoise and both
backwards, against the same towers with the adapters merged into their
weights (`lora_wiring_common.mojo` says what is compared and why).

    pixi run mojo run -I . tests/deep_agents/smolvla/test_lora_wiring.mojo
"""

from tests.deep_agents.smolvla.lora_wiring_common import run_wiring


def main() raises:
    print("SmolVLA LoRA wiring, adapted vs merged (CPU)")
    run_wiring["cpu"](None)
    print("PASSED")
