"""Reverse-mode autodiff for MAX graphs, as a graph transform.

A prototype (see ``docs/PROTOTYPE_MAX_AUTODIFF_PLAN.md``): ``vjp``,
``value_and_grad`` and ``grad`` work while a graph is being built, inside a
``max.graph.Graph`` or a function given to
``max.experimental.compilation.compile``, and emit the backward pass into
that same graph.
"""

from .registry import RuleContext, defvjp, nondiff
from .transform import grad, value_and_grad, vjp

__all__ = ["RuleContext", "defvjp", "grad", "nondiff", "value_and_grad", "vjp"]
