"""Every private MAX API the transform touches, in one place.

This module *is* the list of what MAX would need to make public for the
transform to live outside it. Nothing else in the package may import
``max._core``, ``max._mlir`` or read an underscore attribute of a MAX object.

What it needs, and why:

- ``Graph._mlir_op``: the ``mo.graph`` op whose block holds the ops to walk.
- ``max._core.Operation._from_cmlir``: turn that op into the typed bindings.
- ``max._core`` ``Operation`` / ``Block`` / ``Value``: walk the ops, name
  them (``asm``), and find the op that defines a value (``owner``).
- ``max._core.dialects.mo.ConstantOp`` and its ``#M.dense_array`` value:
  several ops take their parameters as constant operands (a transpose's
  permutation, ``split``'s sizes, ``layer_norm``'s epsilon).
- ``max._mlir.ir.Operation._CAPICreate``: read an attribute of an op that has
  no typed binding (``rmo.concat``, and a custom op's ``symbol``).
- ``Graph._import_kernels``: add a Mojo custom-op package to a graph that is
  already being built (the public route, ``custom_extensions=``, is fixed
  when the graph is created).
- ``TensorValue._mlir_value``: the value identity the walk keys on.

Reflecting generically over the typed bindings' properties can segfault
(``probes/probe_ops2.py``), so only the properties named here are ever read.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from max import _core
from max._core.dialects import mo
from max._mlir import ir
from max.graph import Graph, Value

# The type the walk keys on: hashable, and equal across the different Python
# wrappers of one SSA value.
Key = _core.Value

# Op names are read from the op's assembly. The first token after the result
# list is the name, quoted in the generic form.
_NAME = re.compile(r'\s*(?:%[^=]*=\s*)?"?([A-Za-z_][\w.$]*)')

# A typed binding class maps to exactly one op name, so the name is cached per
# class: printing a constant prints its whole array. The bare
# ``_core.Operation`` (an op without a typed binding) is named every time.
_NAME_BY_CLASS: dict[type, str] = {}


def block_ops(graph: Graph) -> list[_core.Operation]:
    """The ops of ``graph``'s body, in program order (which is topological)."""
    graph_op = _core.Operation._from_cmlir(graph._mlir_op)
    return list(graph_op.regions[0].front)


def key(value: Any) -> Key:
    """The identity of a graph value (a ``TensorValue``, ``BufferValue`` or a
    raw ``max._core.Value``) as the walk tracks it."""
    if isinstance(value, _core.Value):
        return value
    return value._mlir_value


def wrap(k: Key) -> Value:
    """The builder-level value (``TensorValue``, …) for a walk key."""
    return Value.from_mlir(k)


def operands(op: _core.Operation) -> list[Key]:
    # A typed op lists `OpOperand`s; an op without a typed binding (a custom
    # op's `mo.custom`) lists the `Value`s themselves.
    return [o if isinstance(o, _core.Value) else o.value for o in op.operands]


def results(op: _core.Operation) -> list[Key]:
    return list(op.results)


def op_name(op: _core.Operation) -> str:
    """The MLIR name of ``op``, as the builder emitted it (``rmo.matmul``)."""
    cls = type(op)
    if cls is not _core.Operation:
        name = _NAME_BY_CLASS.get(cls)
        if name is not None:
            return name
    match = _NAME.match(op.asm(skip_regions=True, assume_verified=True))
    if match is None:
        raise RuntimeError(f"cannot read the name of op {cls.__name__}")
    name = match.group(1)
    if cls is not _core.Operation:
        _NAME_BY_CLASS[cls] = name
    return name


def location(op: _core.Operation) -> str | None:
    """The op's source location, or ``None`` when MAX did not record one.

    MAX records Python tracebacks only with ``Graph.debug.source_tracebacks``
    or ``MODULAR_DEBUG`` set; otherwise every op carries ``loc(unknown)``.
    """
    asm = op.asm(
        skip_regions=True, assume_verified=True, enable_debug_info=True
    )
    start = asm.rfind(" loc(")
    if start < 0 or asm.endswith("loc(unknown)"):
        return None
    return asm[start + 1 :]


def is_constant(k: Key) -> bool:
    return isinstance(k.owner, mo.ConstantOp)


def constant_value(k: Key) -> np.ndarray | None:
    """The value of a constant operand, or ``None`` if ``k`` is not one."""
    owner = k.owner
    if not isinstance(owner, mo.ConstantOp):
        return None
    tv = wrap(k)
    dtype = tv.dtype.to_numpy()
    shape = tuple(int(d) for d in tv.shape)
    # ``#M.dense_array`` -> ``#M.primitives_array`` -> the raw bytes.
    raw = np.frombuffer(bytes(owner.value.data.data), dtype=dtype)
    return raw.reshape(shape).copy()


def shape_attr(op: _core.Operation, name: str) -> list[int | None]:
    """A ``#mosh.ape`` shape attribute of a typed op (``rmo.slice``'s
    ``starts``), as ints; ``None`` for an entry that is a symbolic expression.
    """
    text = getattr(op, name).asm
    inner = text[text.index("[") + 1 : text.index("]")]
    entries: list[int | None] = []
    for token in inner.split(","):
        try:
            entries.append(int(token))
        except ValueError:
            entries.append(None)
    return entries


def attr(op: _core.Operation, name: str) -> Any:
    """A named attribute of ``op``, read through the upstream MLIR bindings.

    Only for ops without a typed binding; typed ops expose their attributes
    as properties (``.axis``, ``.new_shape``), which rules read directly.
    """
    attribute = ir.Operation._CAPICreate(op._CAPIPtr).attributes[name]
    if isinstance(attribute, ir.IntegerAttr | ir.StringAttr):
        return attribute.value
    return attribute


def import_kernels(graph: Graph, path: Any) -> None:
    """Makes the Mojo custom ops in ``path`` available to ``graph``, once."""
    loaded = graph.__dict__.setdefault("_autodiff_kernel_paths", set())
    if str(path) not in loaded:
        graph._import_kernels([path])
        loaded.add(str(path))
