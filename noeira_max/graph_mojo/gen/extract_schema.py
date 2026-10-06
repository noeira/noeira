"""MAX's ``rmo`` op schema, read from the installed stubs.

MAX ships its graph ops as nanobind classes with a generated ``.pyi``
(``max/_core/dialects/rmo/__init__.pyi``): one class per op, typed
``__init__`` overloads, docstrings. This script parses that file with
``ast`` (nothing is imported from MAX) and writes ``rmo_schema.json``: per
op, its class, MLIR name, summary, and every constructor form with each
parameter's role (operand, result type, attribute) and kind.

    pixi run -e default python noeira_max/graph_mojo/gen/extract_schema.py [OUT.json]

It also prints a coverage report: which ops a generator can expose as typed
functions, and which stub constructs it cannot (``--report`` only prints).
"""

from __future__ import annotations

import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path

DEFAULT_OUT = Path(__file__).with_name("rmo_schema.json")

# Parameter kinds, by annotation. Operands are SSA values; results are types
# the caller supplies in the explicit forms; attributes are compile-time data.
KINDS = {
    "max._core.Value[max._core.dialects.mo.TensorType]": ("operand", "tensor"),
    "max._core.Value[max._core.dialects.mo.BufferType]": ("operand", "buffer"),
    "max._core.Value[max._core.dialects.mo.ChainType]": ("operand", "chain"),
    "max._core.dialects.mo.TensorType": ("result", "tensor"),
    "max._core.dialects.mo.ChainType": ("result", "chain"),
    "max._core.dialects.kgen.ParamDeclArrayAttr": ("param_decls", "param_decls"),
    "Sequence[max._core.dialects.kgen.ParamDeclAttr]": ("param_decls", "param_decls"),
    "max._core.dialects.mosh.ShapeAttr": ("attribute", "shape"),
    "max._core.dialects.builtin.IntegerAttr": ("attribute", "index"),
    "max._core.dialects.builtin.BoolAttr": ("attribute", "bool_attr"),
    "max._core.dialects.builtin.StringAttr": ("attribute", "string_attr"),
    "max._core.dtype.DType": ("attribute", "dtype"),
    "bool": ("attribute", "bool"),
    "int": ("attribute", "int"),
    "str": ("attribute", "string"),
}
# Constructs the stubs name but a typed generator cannot map: an enum
# attribute whose values the stub does not list, and the generic forms that
# take raw operand lists and attribute dictionaries.
UNMAPPED = {
    "max._core.dialects.mo.CoordinateTransformModeAttr": "enum attribute, values not in the stub",
}
GENERIC_FORMS = (
    {"operands", "attributes"},
    {"operands", "properties", "discardable_attributes"},
    {"input_values", "graph_op"},
)


def mlir_name(cls: str, doc: str) -> tuple[str, str]:
    """The op's MLIR name: from the docstring's MLIR example when it has one,
    else derived from the class name (``MoReduceArgMaxOp`` -> ``rmo.mo.reduce_arg_max``,
    which may misplace a dot)."""
    stem = cls.removesuffix("Op")
    dotted = "rmo." + ("mo." if stem.startswith("Mo") else "")
    snake = re.sub(r"(?<!^)(?=[A-Z])", "_", stem.removeprefix("Mo")).lower()
    candidates = set(re.findall(r"\b(rmo\.[a-z0-9_.]+)\(", doc))
    # Keep the example that ends with this op's own words (docstrings also
    # show the ops that feed it, e.g. mo.constant).
    for name in sorted(candidates, key=len, reverse=True):
        if name.replace(".", "_").endswith(snake.split("_")[-1]):
            return name, "docstring"
    return dotted + snake, "derived"


def classify(name: str, annotation: str) -> dict:
    if annotation in KINDS:
        role, kind = KINDS[annotation]
        return {"name": name, "role": role, "kind": kind}
    if annotation.startswith("Sequence[max._core.Value["):
        return {"name": name, "role": "operand", "kind": "tensor", "variadic": True}
    if annotation.endswith("| None") and annotation.removesuffix(" | None") in KINDS:
        role, kind = KINDS[annotation.removesuffix(" | None")]
        return {"name": name, "role": role, "kind": kind, "optional": True}
    return {"name": name, "role": "unmapped", "kind": annotation,
            "why": UNMAPPED.get(annotation, "annotation not in the kind table")}


def extract(stub: Path) -> dict:
    tree = ast.parse(stub.read_text())
    ops = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        doc = ast.get_docstring(node) or ""
        forms = []
        for f in node.body:
            if not (isinstance(f, ast.FunctionDef) and f.name == "__init__"):
                continue
            args = f.args.args[3:]  # self, builder, location
            names = {a.arg for a in args}
            if any(names == g for g in GENERIC_FORMS):
                continue
            defaults = {a.arg for a in args[len(args) - len(f.args.defaults):]} if f.args.defaults else set()
            params = []
            for a in args:
                p = classify(a.arg, ast.unparse(a.annotation))
                if a.arg in defaults:
                    p["default"] = True
                params.append(p)
            explicit = any(p["role"] == "result" for p in params)
            form = {"kind": "explicit" if explicit else "inferred", "params": params}
            if form not in forms:  # overloads that differ only in a C++ type
                forms.append(form)
        name, source = mlir_name(node.name, doc)
        results = max((sum(p["role"] == "result" for p in fm["params"]) for fm in forms), default=0)
        ops.append({
            "class": node.name,
            "mlir_name": name,
            "mlir_name_source": source,
            "summary": doc.strip().split("\n\n")[0].replace("\n", " ") if doc else "",
            "doc": doc,
            "forms": forms,
            "n_results": results or 1,
            "n_results_known": results > 0,
        })
    # The stub's path inside the environment, not on this machine.
    return {"source": stub.as_posix().split("site-packages/")[-1], "ops": ops}


def report(schema: dict) -> str:
    ops = schema["ops"]
    lines = [f"{len(ops)} op classes in {Path(schema['source']).name}"]
    def usable(form: dict) -> bool:
        return all(p["role"] != "unmapped" for p in form["params"])
    inferred = [o for o in ops if any(f["kind"] == "inferred" and usable(f) for f in o["forms"])]
    explicit_only = [o for o in ops if o not in inferred and any(usable(f) for f in o["forms"])]
    none = [o for o in ops if not any(usable(f) for f in o["forms"])]
    lines.append(f"  {len(inferred)} have a form with inferred result types (operands and attributes only)")
    lines.append(f"  {len(explicit_only)} only have forms where the caller supplies the result types")
    lines.append(f"  {len(none)} have no form a typed generator can map: "
                 + ", ".join(o["class"] for o in none))
    unmapped = Counter((p["kind"], p.get("why", "")) for o in ops for f in o["forms"]
                       for p in f["params"] if p["role"] == "unmapped")
    for (kind, why), n in unmapped.items():
        lines.append(f"  unmapped parameter type ({n}x): {kind} ({why})")
    named = Counter(o["mlir_name_source"] for o in ops)
    lines.append(f"  MLIR names: {named['docstring']} from docstring examples, "
                 f"{named['derived']} derived from the class name")
    unknown = [o["class"] for o in ops if not o["n_results_known"]]
    lines.append(f"  result count not in any form ({len(unknown)}): assumed 1 for "
                 + ", ".join(unknown[:8]) + (" ..." if len(unknown) > 8 else ""))
    kinds = Counter(p["kind"] for o in ops for f in o["forms"] for p in f["params"]
                    if p["role"] == "attribute")
    lines.append("  attribute kinds: " + ", ".join(f"{k} {n}" for k, n in kinds.most_common()))
    return "\n".join(lines)


def stub_path() -> Path:
    import importlib.util

    spec = importlib.util.find_spec("max")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("cannot find the installed `max` package")
    return Path(list(spec.submodule_search_locations)[0]) / "_core/dialects/rmo/__init__.pyi"


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    schema = extract(stub_path())
    print(report(schema))
    if "--report" not in sys.argv:
        out = Path(args[0]) if args else DEFAULT_OUT
        out.write_text(json.dumps(schema, indent=1) + "\n")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
