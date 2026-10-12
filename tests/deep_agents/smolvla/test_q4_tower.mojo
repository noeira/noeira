"""The SmolVLA int4 text tower against its fp32 twin over deq(W), CPU
(`q4_tower_common.mojo` says what is compared and why).

    pixi run mojo run -I . tests/deep_agents/smolvla/test_q4_tower.mojo
"""

from tests.deep_agents.smolvla.q4_tower_common import run_q4_tower


def main() raises:
    print("SmolVLA int4 text tower vs fp32 over deq(W) (CPU)")
    run_q4_tower["cpu"](None)
    print("PASSED")
