"""Every case against ``torch.func.vjp``, in float64, on the same inputs.

The MAX side dumps inputs, ``r``, the forward output and the VJP to an
``.npz``; ``golden_torch.py`` recomputes both in torch (``act-ref`` env, a
separate process) and reports elementwise agreement. Skipped when the
``act-ref`` env is not installed.

    noeira_max/autodiff/run.sh -m unittest noeira_max.autodiff.tests.test_golden -v
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from max.driver import CPU
from max.engine import InferenceSession

from .cases import FAMILIES
from .harness import FamilyGraph

TORCH_PYTHON = Path(sys.prefix).parent / "act-ref" / "bin" / "python"
TORCH_SIDE = Path(__file__).with_name("golden_torch.py")


def dump(path: Path) -> None:
    session = InferenceSession(devices=[CPU()])
    arrays = {}
    for family in FAMILIES:
        graph = FamilyGraph(family, session)
        for b, binding in enumerate(family.bindings):
            per_case = graph.inputs(binding, seed=100 + b)
            for case, ins, out in zip(family.cases, per_case, graph.run(per_case)):
                prefix = f"{family.name}.{case.name}.{b}"
                for j, x in enumerate(ins[:-1]):
                    arrays[f"{prefix}.x{j}"] = x
                arrays[f"{prefix}.r"] = ins[-1]
                arrays[f"{prefix}.y"] = out[1]
                for k, g in enumerate(out[2:]):
                    arrays[f"{prefix}.g{k}"] = g
    np.savez(path, **arrays)


@unittest.skipUnless(TORCH_PYTHON.exists(), f"no act-ref env at {TORCH_PYTHON}")
class GoldenTest(unittest.TestCase):
    def test_against_torch(self):
        with tempfile.TemporaryDirectory() as tmp:
            npz, report = Path(tmp) / "golden.npz", Path(tmp) / "torch.json"
            dump(npz)
            env = {k: v for k, v in os.environ.items() if k != "LD_PRELOAD"}
            proc = subprocess.run(
                [str(TORCH_PYTHON), str(TORCH_SIDE), str(npz), str(report)],
                env=env, capture_output=True, text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr[-3000:])
            results = json.loads(report.read_text())

        self.assertGreater(len(results), 0)
        worst_y: dict[str, float] = {}
        failures = []
        for r in results:
            name = f"{r['family']}.{r['case']}"
            worst_y[name] = max(worst_y.get(name, 0.0), r["y"].get("normwise", np.inf))
            for k, g in enumerate(r["grads"]):
                if not g["ratio"] <= 1.0:
                    failures.append(f"{name} binding {r['binding']} grad {k}: {g}")

        print("\nForward agreement with torch (worst normwise relative):")
        for name, err in sorted(worst_y.items(), key=lambda kv: -kv[1])[:12]:
            print(f"  {name:40s} {err:.1e}")
        self.assertFalse(failures, "\n".join(failures))


if __name__ == "__main__":
    unittest.main()
