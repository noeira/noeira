"""Every rule against central finite differences, in float64 on CPU.

    noeira_max/autodiff/run.sh -m unittest noeira_max.autodiff.tests.test_gradcheck -v
"""

from __future__ import annotations

import unittest

from max.driver import CPU
from max.engine import InferenceSession

from .cases import FAMILIES
from .harness import FamilyGraph

_SESSION = None


def _session() -> InferenceSession:
    global _SESSION
    if _SESSION is None:
        _SESSION = InferenceSession(devices=[CPU()])
    return _SESSION


def _make_test(family):
    class FamilyTest(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            cls.graph = FamilyGraph(family, _session())
            print(
                f"\n[{family.name}] {len(family.cases)} cases, compiled in "
                f"{cls.graph.compile_seconds:.1f} s",
                flush=True,
            )

        def test_gradcheck(self):
            for i, binding in enumerate(family.bindings):
                failures, worst = self.graph.gradcheck(binding, seed=i)
                with self.subTest(binding=binding):
                    detail = "\n".join(
                        f"  {f.case}: fd {f.fd:.10g} vs analytic "
                        f"{f.analytic:.10g} (rel. error {f.error:.2e})"
                        for f in failures
                    )
                    self.assertFalse(failures, f"binding {binding}:\n{detail}")

    FamilyTest.__name__ = FamilyTest.__qualname__ = f"Test_{family.name}"
    return FamilyTest


for _family in FAMILIES:
    globals()[f"Test_{_family.name}"] = _make_test(_family)
del _family


if __name__ == "__main__":
    unittest.main()
