"""Reverse-mode AD as a graph transform (plan §3.2).

``fun`` runs in the graph being built, so its ops land in the enclosing block.
The transform then walks those ops backwards and emits each one's VJP into
the same graph, through the public builder. Forward, backward and whatever
follows (an optimizer update) end up in one graph, compiled as one model.

Works inside ``max.graph.Graph`` (on ``TensorValue``) and inside
``max.experimental.compilation.stage`` / ``compile`` (on ``Tensor``).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from max.experimental import tree_utils as tree
from max.experimental.tensor import Tensor
from max.graph import BufferValue, Graph, StaticDim, TensorValue, ops

from . import _graph, registry, rules  # noqa: F401  (rules: registers them)
from ._ops import is_one, ones_like, zeros_like
from .registry import RuleContext

# What a primal, an output or a cotangent may be: the walk stops there.
_LEAF = (TensorValue, BufferValue, Tensor)


def vjp(fun: Callable[..., Any], *primals: Any, has_aux: bool = False):
    """Calls ``fun(*primals)`` in the current graph, like ``jax.vjp``.

    Returns ``(out, vjp_fn)``, or ``(out, vjp_fn, aux)`` with ``has_aux``
    (``fun`` then returns ``(out, aux)``). ``vjp_fn(cotangents)`` takes one
    cotangent per leaf of ``out``, in ``out``'s structure, emits the backward
    pass into the graph, and returns one cotangent per primal, in the
    primals' structure. Values ``fun`` closes over are constants.
    """
    graph = _current_graph()
    leaves, treedef = tree.flatten(primals, leaf=_LEAF)
    bound = _bind(leaves)
    before = _op_ids(graph)

    out = fun(*tree.unflatten(treedef, [b.argument for b in bound]))
    aux = None
    if has_aux:
        out, aux = out
    out_leaves, out_def = tree.flatten(out, leaf=_LEAF)
    tape = _Tape(
        graph,
        before,
        inputs=[b.value for b in bound],
        outputs=[_tensor_value(o) for o in out_leaves],
    )

    def vjp_fn(cotangents: Any) -> Any:
        ct_leaves, ct_def = tree.flatten(cotangents, leaf=_LEAF)
        if ct_def != out_def:
            raise ValueError(
                "vjp_fn needs one cotangent per output leaf, in the output's "
                f"structure; got {ct_def}, expected {out_def}"
            )
        grads = tape.backward([_tensor_value(c) for c in ct_leaves])
        return tree.unflatten(
            treedef, [_like(leaf, g) for leaf, g in zip(leaves, grads)]
        )

    return (out, vjp_fn, aux) if has_aux else (out, vjp_fn)


def value_and_grad(
    fun: Callable[..., Any],
    argnums: int | Sequence[int] = 0,
    *,
    has_aux: bool = False,
) -> Callable[..., Any]:
    """``jax.value_and_grad``: the value of a scalar-valued ``fun`` and its
    gradient with respect to the arguments in ``argnums``.

    The gradient has the structure of that argument (or a tuple of them when
    ``argnums`` is a sequence). A scalar is any tensor whose dims are all 1,
    since MAX reductions keep the reduced axes.
    """
    single = isinstance(argnums, int)
    nums = (argnums,) if single else tuple(argnums)

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        def partial(*diff: Any) -> Any:
            full = list(args)
            for i, value in zip(nums, diff):
                full[i] = value
            return fun(*full, **kwargs)

        result = vjp(partial, *(args[i] for i in nums), has_aux=has_aux)
        out, vjp_fn = result[0], result[1]
        value = _tensor_value(out)
        if not all(is_one(d) for d in value.shape):
            raise TypeError(
                "value_and_grad needs a scalar output (every dim 1); got "
                f"shape {value.shape}. Use vjp for other outputs."
            )
        grads = vjp_fn(_like(out, ones_like(value)))
        grads = grads[0] if single else grads
        return ((out, result[2]) if has_aux else out), grads

    return wrapper


def grad(
    fun: Callable[..., Any],
    argnums: int | Sequence[int] = 0,
    *,
    has_aux: bool = False,
) -> Callable[..., Any]:
    """``jax.grad``: like :func:`value_and_grad`, without the value."""
    vg = value_and_grad(fun, argnums, has_aux=has_aux)

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        value, grads = vg(*args, **kwargs)
        return (grads, value[1]) if has_aux else grads

    return wrapper


class _Bound:
    """A primal leaf: what ``fun`` receives, and the value the walk tracks."""

    def __init__(self, argument: Any, value: TensorValue) -> None:
        self.argument = argument
        self.value = value


def _bind(leaves: list[Any]) -> list[_Bound]:
    bound = []
    seen: set[_graph.Key] = set()
    for leaf in leaves:
        value = _tensor_value(leaf)
        if _graph.key(value) in seen:
            # The same value passed twice: give each position its own SSA
            # value, so each gets its own partial derivative.
            value = ops.rebind(value, value.shape)
        seen.add(_graph.key(value))
        argument = (
            Tensor.from_graph_value(value) if isinstance(leaf, Tensor) else value
        )
        bound.append(_Bound(argument, value))
    return bound


class _Tape:
    """The ops ``fun`` emitted that depend on the primals, in program order."""

    def __init__(
        self,
        graph: Graph,
        before: set[_graph.Key],
        inputs: list[TensorValue],
        outputs: list[TensorValue],
    ) -> None:
        self.inputs = inputs
        self.outputs = outputs
        self.input_keys = [_graph.key(v) for v in inputs]
        active = set(self.input_keys)
        self.ops = []
        for op in _graph.block_ops(graph):
            results = _graph.results(op)
            if results and results[0] in before:
                continue  # built before ``fun`` ran: a constant to ``fun``
            if any(k in active for k in _graph.operands(op)):
                self.ops.append(op)
                active.update(results)
        self.active = active

    def backward(self, seeds: list[TensorValue]) -> list[TensorValue]:
        cts: dict[_graph.Key, TensorValue] = {}
        for out, seed in zip(self.outputs, seeds):
            _add_ct(cts, _graph.key(out), _conform("seed", out, seed))

        for op in reversed(self.ops):
            result_keys = _graph.results(op)
            op_cts = [cts.get(k) for k in result_keys]
            if all(c is None for c in op_cts):
                continue
            name = _graph.op_name(op)
            if registry.is_nondiff(name):
                continue
            input_keys = _graph.operands(op)
            inputs = [_graph.wrap(k) for k in input_keys]
            needs = [
                k in self.active and _is_float(v)
                for k, v in zip(input_keys, inputs)
            ]
            if not any(needs):
                continue
            rule = registry.lookup(name, op)
            ctx = RuleContext(
                name=name,
                op=op,
                inputs=inputs,
                outputs=[_graph.wrap(k) for k in result_keys],
                cts=op_cts,
                needs=needs,
                _keys=input_keys,
            )
            in_cts = list(rule(ctx))
            if len(in_cts) != len(inputs):
                raise RuntimeError(
                    f"the rule for '{name}' returned {len(in_cts)} cotangents "
                    f"for {len(inputs)} inputs"
                )
            for k, value, need, ct in zip(input_keys, inputs, needs, in_cts):
                if need and ct is not None:
                    _add_ct(cts, k, _conform(name, value, ct))

        return [
            cts[k] if k in cts else zeros_like(v)
            for k, v in zip(self.input_keys, self.inputs)
        ]


def _add_ct(
    cts: dict[_graph.Key, TensorValue], k: _graph.Key, ct: TensorValue
) -> None:
    """Accumulates: a value used twice receives the sum of both cotangents."""
    cts[k] = ops.add(cts[k], ct) if k in cts else ct


def _conform(name: str, value: TensorValue, ct: TensorValue) -> TensorValue:
    """Checks a cotangent against its primal, so a rule that forgets to
    unbroadcast fails here, at trace time, naming the op."""
    if ct.dtype != value.dtype:
        raise TypeError(
            f"the rule for '{name}' returned a {ct.dtype} cotangent for a "
            f"{value.dtype} input"
        )
    want, have = list(value.shape), list(ct.shape)
    if have == want:
        return ct
    static_mismatch = len(have) != len(want) or any(
        h != w for h, w in zip(have, want) if _static(h) and _static(w)
    )
    if static_mismatch:
        raise ValueError(
            f"the rule for '{name}' returned a cotangent of shape {ct.shape} "
            f"for an input of shape {value.shape} (missing unbroadcast?)"
        )
    return ops.rebind(ct, want)


def _static(dim: Any) -> bool:
    return isinstance(dim, StaticDim)


def _is_float(value: Any) -> bool:
    return isinstance(value, TensorValue) and value.dtype.is_float()


def _tensor_value(x: Any) -> TensorValue:
    if isinstance(x, TensorValue):
        return x
    if isinstance(x, Tensor):
        return x.__tensorvalue__()
    if isinstance(x, BufferValue):
        return ops.buffer_load(x)
    raise TypeError(
        f"expected a TensorValue, BufferValue or Tensor; got {type(x).__name__}"
    )


def _like(template: Any, value: TensorValue) -> Any:
    """``value`` as the same kind of object as ``template``."""
    return Tensor.from_graph_value(value) if isinstance(template, Tensor) else value


def _op_ids(graph: Graph) -> set[_graph.Key]:
    """Identifies the ops already in ``graph`` by their first result."""
    ids = set()
    for op in _graph.block_ops(graph):
        results = _graph.results(op)
        if results:
            ids.add(results[0])
    return ids


def _current_graph() -> Graph:
    try:
        return Graph.current
    except LookupError:
        raise NotImplementedError(
            "call vjp / value_and_grad while a graph is being built (inside "
            "max.graph.Graph, or a function given to "
            "max.experimental.compilation.compile or stage). Eager mode is "
            "not supported yet."
        ) from None
