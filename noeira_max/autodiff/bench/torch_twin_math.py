"""The torch twin (``tools/nn/torch_nn_reference.py``) with flash attention
off: SDPA restricted to its MATH backend, which materialises the scores as
the MAX step does. This column separates the missing flash-attention backward
(RFC K3) from the cost of the AD transform itself.

    env -u LD_PRELOAD .pixi/envs/act-ref/bin/python \\
        noeira_max/autodiff/bench/torch_twin_math.py gpt --mode compile --bench-steps 50

Arguments are the twin's.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from torch.nn.attention import SDPBackend, sdpa_kernel

TWIN = Path(__file__).resolve().parents[3] / "tools" / "nn" / "torch_nn_reference.py"


def main() -> None:
    spec = importlib.util.spec_from_file_location("torch_nn_reference", TWIN)
    twin = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = twin
    spec.loader.exec_module(twin)
    print(f"[torch gpt] SDPA backend: MATH only ({TWIN.name})", flush=True)
    with sdpa_kernel(SDPBackend.MATH):
        twin.main()


if __name__ == "__main__":
    main()
