"""The gradcheck can fail: inject the defects the rules guard against, and
check that the suite catches each one.

    noeira_max/autodiff/run.sh -m unittest noeira_max.autodiff.tests.test_vacuity -v
"""

from __future__ import annotations

import unittest
from unittest import mock

from max.driver import CPU
from max.engine import InferenceSession

from noeira_max.autodiff import registry, transform

from .cases import COMPOSITE, ELEMENTWISE, Family
from .harness import FamilyGraph

_SESSION = InferenceSession(devices=[CPU()])


def _subset(family: Family, *names: str) -> Family:
    cases = tuple(c for c in family.cases if c.name in names)
    assert len(cases) == len(names), names
    return Family(f"{family.name}_vacuity", cases, family.bindings[1:2])


def _failing_cases(family: Family) -> set[str]:
    graph = FamilyGraph(family, _SESSION)
    failures, _ = graph.gradcheck(family.bindings[0])
    return {f.case for f in failures}


class VacuityTest(unittest.TestCase):
    def test_overwriting_a_cotangent_is_caught(self):
        # The bug that broke BPTT in noeira's first framework, silently.
        def overwrite(cts, k, ct):
            cts[k] = ct

        family = _subset(COMPOSITE, "used_twice", "used_thrice_through_exp")
        with mock.patch.object(transform, "_add_ct", overwrite):
            failing = _failing_cases(family)
        self.assertEqual(failing, {"used_twice", "used_thrice_through_exp"})

    def test_a_wrong_rule_is_caught(self):
        # exp's derivative without its exp(x) factor.
        family = _subset(ELEMENTWISE, "exp", "tanh")
        with mock.patch.dict(
            registry._RULES, {"rmo.mo.exp": lambda ctx: [ctx.ct]}
        ):
            failing = _failing_cases(family)
        self.assertEqual(failing, {"exp"})

    def test_a_missing_unbroadcast_fails_at_trace_time(self):
        family = _subset(ELEMENTWISE, "add_bcast_rank")
        with mock.patch.dict(
            registry._RULES, {"rmo.add": lambda ctx: [ctx.ct, ctx.ct]}
        ):
            with self.assertRaisesRegex(ValueError, "rmo.add.*unbroadcast"):
                FamilyGraph(family, _SESSION)


if __name__ == "__main__":
    unittest.main()
