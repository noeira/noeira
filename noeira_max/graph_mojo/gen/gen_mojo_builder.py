"""Generates ``max_graph_gen/ops.mojo`` from ``rmo_schema.json``: every
``rmo`` op MAX's stubs describe, as a typed Mojo function over a
``GraphBackend``.

    pixi run -e default python noeira_max/graph_mojo/gen/extract_schema.py
    pixi run -e default python noeira_max/graph_mojo/gen/gen_mojo_builder.py

Each function takes the op's operands as ``Value``s, its attributes as Mojo
types, and, for ops whose stubs have no inferring constructor, its result
types. It fills an ``OpArgs`` in the stub's argument order and calls
``add_op``. The op's form is the stub's own: an op is usable from Mojo as
soon as MAX's stub lists it, with nothing written by hand.
"""

from __future__ import annotations

import json
import re
import sys
import textwrap
import time
from pathlib import Path

HERE = Path(__file__).parent
SCHEMA = HERE / "rmo_schema.json"
OUT = HERE.parent / "max_graph_gen" / "ops.mojo"

# Words a Mojo function or argument cannot be called.
RESERVED = {
    "and", "or", "not", "in", "is", "if", "else", "elif", "for", "while", "def",
    "struct", "trait", "var", "ref", "return", "raise", "raises", "try", "except",
    "finally", "with", "as", "from", "import", "pass", "break", "continue",
    "comptime", "del", "lambda", "global", "nonlocal", "yield", "assert", "async",
    "await", "class", "fn", "let", "alias", "inout", "owned", "borrowed", "out",
    "mut", "imm", "read", "deinit", "self", "Self", "None", "True", "False",
    "case", "match",
}
# Attribute kinds: the Mojo type, and the `OpArgs` method that records it.
ATTRS = {
    "index": ("Int", "attr_index"),
    "int": ("Int", "attr_int"),
    "bool": ("Bool", "attr_bool"),
    "bool_attr": ("Bool", "attr_bool_attr"),
    "string": ("String", "attr_string"),
    "string_attr": ("String", "attr_string_attr"),
    "shape": ("List[Dim]", "attr_shape"),
    "dtype": ("DType", "attr_dtype"),
}


def mojo_name(name: str) -> str:
    return name + "_" if name in RESERVED else name


def function_name(cls: str) -> str:
    stem = cls.removesuffix("Op")
    return mojo_name(re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", stem).lower())


def mappable(form: dict) -> bool:
    return all(p["role"] != "unmapped" for p in form["params"])


# Attribute kinds passed as MLIR attribute objects. The stubs say
# `IntegerAttr`, not which integer type: `index` is a guess (right for
# axes), while a form taking a plain `int` lets MAX's C++ builder choose
# (`top_k` needs a 64-bit signed `k`).
ATTR_OBJECTS = {"index", "bool_attr", "string_attr"}


def choose_form(op: dict) -> dict | None:
    """The inferring form when the stub has one, else the explicit one; and
    among those, the one taking the most plain Python types."""
    forms = [f for f in op["forms"] if mappable(f)]
    if not forms:
        return None
    return min(forms, key=lambda f: (f["kind"] != "inferred",
                                     sum(p["kind"] in ATTR_OBJECTS for p in f["params"])))


def docstring(op: dict, form: dict) -> list[str]:
    summary = op["summary"].replace("\\", "\\\\").replace('"""', "'''").strip()
    if not summary:
        summary = f"The `{op['mlir_name']}` op."
    if summary[0].isalpha() and summary[0].islower():
        summary = summary[0].upper() + summary[1:]
    note = (f"`{op['mlir_name']}` (`{op['class']}`); "
            + ("result types inferred." if form["kind"] == "inferred"
               else "the caller supplies the result type."))
    lines = textwrap.wrap(summary, 72) + [""] + textwrap.wrap(note, 72)
    out = ['    """' + lines[0]]
    out += [("    " + l) if l else "" for l in lines[1:]]
    out[-1] += '"""' if len(lines) > 1 else ""
    if len(lines) == 1:
        out[0] += '"""'
    return out


def generate_op(op: dict, form: dict) -> list[str]:
    params, body = ["mut b: B"], [f'    var a = OpArgs("{op["class"]}", "{op["mlir_name"]}")']
    n_results = 0
    for p in form["params"]:
        name, role, kind = mojo_name(p["name"]), p["role"], p["kind"]
        if role == "operand":
            if p.get("optional"):
                params.append(f"{name}: Optional[Value] = None")
                body.append(f"    if {name}:")
                body.append(f'        a.operand("{p["name"]}", {name}.value())')
            else:
                params.append(f"{name}: Value")
                body.append(f'    a.operand("{p["name"]}", {name})')
        elif role == "result":
            n_results += 1
            if kind == "chain":
                body.append(f'    a.result_chain("{p["name"]}")')
            else:
                params.append(f"{name}: TensorType")
                body.append(f'    a.result("{p["name"]}", {name})')
        elif role == "param_decls":
            if not p.get("default"):  # a defaulted one is left to its default
                body.append(f'    a.param_decls("{p["name"]}")')
        elif role == "attribute":
            mojo_type, method = ATTRS[kind]
            if p.get("default"):
                params.append(f"{name}: Optional[{mojo_type}] = None")
                body.append(f"    if {name}:")
                body.append(f'        a.{method}("{p["name"]}", {name}.value())')
            else:
                params.append(f"{name}: {mojo_type}")
                body.append(f'    a.{method}("{p["name"]}", {name})')
    n_results = n_results or op["n_results"]
    returns = "Value" if n_results == 1 else "List[Value]"
    body.append("    var results = b.add_op(a^)")
    body.append("    return results[0].copy()" if n_results == 1 else "    return results^")
    head = f"def {function_name(op['class'])}[B: GraphBackend]("
    signature = head + ", ".join(params) + f") raises -> {returns}:"
    if len(signature) > 88:
        signature = head + "\n" + ",\n".join("    " + p for p in params) + f",\n) raises -> {returns}:"
    return [signature] + docstring(op, form) + body


def main() -> None:
    start = time.perf_counter()
    schema = json.loads(SCHEMA.read_text())
    chunks, skipped, names = [], [], set()
    for op in schema["ops"]:
        form = choose_form(op)
        if form is None:
            skipped.append(op["class"])
            continue
        name = function_name(op["class"])
        if name in names:
            raise SystemExit(f"two ops map to the Mojo name {name}")
        names.add(name)
        chunks.append("\n".join(generate_op(op, form)))
    header = [
        "# GENERATED by noeira_max/graph_mojo/gen/gen_mojo_builder.py from",
        f"# {Path(schema['source']).as_posix().split('site-packages/')[-1]} (via rmo_schema.json).",
        "# DO NOT EDIT: rerun extract_schema.py, then gen_mojo_builder.py.",
        '"""Every `rmo` op that MAX\'s stubs describe, as a typed Mojo function.',
        "",
        "Each function takes a `GraphBackend` first, then the op's arguments in the",
        "stub's order: operands as `Value`s, attributes as Mojo types, and, where",
        "the stub has no inferring constructor, the result type. "
        f"{len(chunks)} ops;",
        "skipped (no form a typed generator can map): "
        + (", ".join(skipped) or "none") + '.',
        '"""',
        "",
        "from .backend import Dim, GraphBackend, OpArgs, TensorType, Value",
        "",
    ]
    text = "\n".join(header) + "\n\n" + "\n\n\n".join(chunks) + "\n"
    OUT.write_text(text)
    seconds = time.perf_counter() - start
    print(f"wrote {OUT}: {len(chunks)} functions, {text.count(chr(10))} lines, "
          f"skipped {skipped}, in {seconds * 1000:.0f} ms")
    if "--names" in sys.argv:
        print(" ".join(sorted(names)))


if __name__ == "__main__":
    main()
