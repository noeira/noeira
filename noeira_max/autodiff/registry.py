"""The VJP rule registry, keyed by MLIR op name (plan §3.3).

A rule receives a :class:`RuleContext` and returns one cotangent per op input,
``None`` where it contributes nothing. Rules emit their ops through the public
builder (``max.graph.ops``) into the graph being differentiated.

Keys are op names as the builder emits them (``rmo.matmul``,
``mo.reduce.layer_norm``), never Python classes: ``rmo.concat`` has no class,
and nanobind may make several class objects for one op.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from max.graph import TensorValue, Value

from . import _graph

Cotangent = TensorValue | None
Rule = Callable[["RuleContext"], Sequence[Cotangent]]

_RULES: dict[str, Rule] = {}
_NONDIFF: set[str] = set()


@dataclass
class RuleContext:
    """What a rule sees of the op it differentiates."""

    name: str
    """The op's MLIR name."""
    op: Any
    """The typed binding, for parameters such as ``.axis``."""
    inputs: list[Value]
    """The op's operands, as builder values."""
    outputs: list[Value]
    """The op's results, as builder values."""
    cts: list[Cotangent]
    """One cotangent per result; ``None`` where none arrived."""
    needs: list[bool]
    """Whether each input needs a cotangent. A rule may skip the others."""
    _keys: list[_graph.Key]

    @property
    def ct(self) -> TensorValue:
        """The cotangent of the op's single (or first) result."""
        ct = self.cts[0]
        assert ct is not None, f"{self.name}: no cotangent for result 0"
        return ct

    def const(self, i: int) -> np.ndarray:
        """The value of input ``i``, which must be a constant."""
        value = _graph.constant_value(self._keys[i])
        if value is None:
            raise NotImplementedError(
                f"{self.name}: input {i} is not a constant; this rule only "
                "handles a constant there"
            )
        return value

    def attr(self, name: str) -> Any:
        """An attribute of an op without a typed binding (``rmo.concat``)."""
        return _graph.attr(self.op, name)


def defvjp(*names: str) -> Callable[[Rule], Rule]:
    """Registers the decorated function as the VJP rule of ``names``."""

    def register(rule: Rule) -> Rule:
        for name in names:
            _claim(name)
            _RULES[name] = rule
        return rule

    return register


def nondiff(*names: str) -> None:
    """Declares ops that pass no cotangent to any input (comparisons,
    ``argmax``, ``floor``, random draws, integer casts)."""
    for name in names:
        _claim(name)
        _NONDIFF.add(name)


def _claim(name: str) -> None:
    if name in _RULES or name in _NONDIFF:
        raise ValueError(f"a VJP rule for '{name}' is already registered")


def is_nondiff(name: str) -> bool:
    return name in _NONDIFF


def lookup(name: str, op: Any) -> Rule:
    """The rule for ``name``. An op without one is an error, never a zero."""
    rule = _RULES.get(name)
    if rule is not None:
        return rule
    where = _graph.location(op)
    hint = (
        f" Built at {where}."
        if where
        else " Set `Graph.debug.source_tracebacks = True` before building the"
        " graph to see where it was built."
    )
    raise NotImplementedError(
        f"No VJP rule for '{name}', which is on the path from the inputs to "
        f"the output.{hint} Register one with @defvjp('{name}'), or "
        f"nondiff('{name}') if it has no gradient."
    )


def coverage() -> tuple[list[str], list[str]]:
    """The op names with a rule, and the ones declared non-differentiable."""
    return sorted(_RULES), sorted(_NONDIFF)
