#!/usr/bin/env python3
"""Rewrite LayoutTensor uses to the `noeira.core.tiles` TileTensor spelling.

MAX 26.7 deprecates LayoutTensor and modular a65e584 removes it. This handles
the mechanical part of the migration (docs/MOJO_1_2_MAX_26_7_UPGRADE.md §5):

    LayoutTensor[DT, Layout.row_major(B, N), MutAnyOrigin]            -> Tile[DT, B, N]
    LayoutTensor[DT, Layout.row_major(B, N), ImmutAnyOrigin]          -> ImmTile[DT, B, N]
    LayoutTensor[DT, Layout.row_major(N), MutAnyOrigin,
                 address_space=AddressSpace.SHARED]                   -> SharedTile[DT, N]
    LayoutTensor[DT, Layout.row_major(B, N), MutAnyOrigin](buf)       -> tile[B, N](buf)
    LayoutTensor[DT, Layout.row_major(B, N)](buf)                     -> tile[B, N](buf)
    LayoutTensor[..SHARED].stack_allocation()                         -> shared_tile[DT, N]()
    LayoutTensor[.., MutAnyOrigin].stack_allocation()                 -> local_tile[DT, N]()
    t.lt[T, Layout.row_major(B, N)]()                                 -> t.tt[T, B, N]()
    t.lt_at[T, Layout.row_major(B, N)](off)                           -> t.tt_at[T, B, N](off)

A layout may also be a NAME bound by `comptime NAME = Layout.row_major(...)`
(the nearest preceding binding in the file, or `Self.NAME` / a module-level
one). Everything else is left alone and reported: runtime layouts
(`UNKNOWN_VALUE`, `RuntimeLayout`), generic `L: Layout` parameters, other
origins and keyword arguments. Comments and docstrings are not touched.

The rewrite is type-checked by the compiler: every Tile carries static dims,
so a wrongly resolved name shows up as a type mismatch, not a silent change.
Element reads such as `rebind[Scalar[DT]](t[i, j])` compile unchanged.

Usage: lt_to_tile.py [--dry-run] FILE_OR_DIR...
"""

import re
import sys
from pathlib import Path

TILE_NAMES = (
    "RowMajor", "Tile", "ImmTile", "SharedTile", "tile", "shared_tile", "local_tile"
)


def code_mask(s):
    """True for each char that is code (not in a comment or a string)."""
    mask = [True] * len(s)
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if s.startswith('"""', i) or s.startswith("'''", i):
            q = s[i : i + 3]
            j = s.find(q, i + 3)
            j = n if j < 0 else j + 3
            for k in range(i, j):
                mask[k] = False
            i = j
        elif c in "\"'":
            j = i + 1
            while j < n and s[j] != c and s[j] != "\n":
                j += 2 if s[j] == "\\" else 1
            for k in range(i, min(j + 1, n)):
                mask[k] = False
            i = j + 1
        elif c == "#":
            j = s.find("\n", i)
            j = n if j < 0 else j
            for k in range(i, j):
                mask[k] = False
            i = j
        else:
            i += 1
    return mask


def match_bracket(s, i):
    """Index of the bracket closing the one at s[i], or -1."""
    pairs = {"[": "]", "(": ")"}
    stack = []
    for j in range(i, len(s)):
        c = s[j]
        if c in pairs:
            stack.append(pairs[c])
        elif c in ")]":
            if not stack or stack.pop() != c:
                return -1
            if not stack:
                return j
    return -1


def split_top(body):
    """Split on commas at bracket depth 0; drop a trailing empty part."""
    parts, depth, cur = [], 0, []
    for c in body:
        if c in "([":
            depth += 1
        elif c in ")]":
            depth -= 1
        if c == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(c)
    parts.append("".join(cur))
    parts = [" ".join(p.split()) for p in parts]
    if parts and parts[-1] == "":
        parts.pop()
    return parts


ROW_MAJOR = re.compile(r"^Layout\.row_major\((.*)\)$", re.S)
ROW_MAJOR_T = re.compile(r"^RowMajor\[(.*)\]$", re.S)


def static_row_major(expr):
    """`Layout.row_major(a, b)` -> 'a, b', or None if not a static row_major."""
    m = ROW_MAJOR.match(expr.strip()) or ROW_MAJOR_T.match(expr.strip())
    if not m:
        return None
    args = split_top(m.group(1))
    if not args or any(
        "UNKNOWN_VALUE" in a or "IndexList" in a or "Index(" in a or "=" in a
        for a in args
    ):
        return None
    return ", ".join(args)


BINDING = re.compile(
    r"^[ \t]*comptime[ \t]+(\w+)(?:[ \t]*:[ \t]*Layout)?[ \t]*=[ \t]*(Layout\.row_major\()",
    re.M,
)


def layout_bindings(s, mask):
    """[(pos, name, dims-or-None)] for every `comptime NAME = Layout.row_major(..)`."""
    out = []
    for m in BINDING.finditer(s):
        if not mask[m.start(1)]:
            continue
        p = m.start(2) + len("Layout.row_major")
        q = match_bracket(s, p)
        if q < 0:
            continue
        out.append((m.start(), m.group(1), static_row_major(s[m.start(2) : q + 1])))
    return out


GENERIC = re.compile(r"\b(\w+)[ \t]*:[ \t]*(Tensor)?Layout\b(?!\.)")


def generic_params(s, mask):
    """[(pos, name)] for every `NAME: Layout` parameter declaration."""
    return [(m.start(), m.group(1)) for m in GENERIC.finditer(s) if mask[m.start()]]


def name_kind(name, pos, bindings, generics):
    """'binding' / 'generic' / None for a layout NAME used at `pos`: the
    nearest preceding declaration of either kind decides."""
    b = [x[0] for x in bindings if x[1] == name and x[0] < pos]
    g = [x[0] for x in generics if x[1] == name and x[0] < pos]
    if not b and not g:
        if any(x[1] == name for x in bindings):
            return "binding"
        return "generic" if any(x[1] == name for x in generics) else None
    return "binding" if (b[-1] if b else -1) > (g[-1] if g else -1) else "generic"


def resolve_layout(expr, pos, bindings):
    """Dims string for a layout expression at `pos`, or None."""
    dims = static_row_major(expr)
    if dims is not None:
        return dims
    name = expr.strip()
    if name.startswith("Self."):
        name = name[5:]
    if not re.fullmatch(r"\w+", name):
        return None
    before = [b for b in bindings if b[1] == name and b[0] < pos]
    cands = before[-1:] or [b for b in bindings if b[1] == name]
    if len(cands) != 1:
        return None
    # A later binding with a different value makes the name ambiguous only
    # when nothing precedes the use; the nearest preceding one wins otherwise.
    if not before and len({b[2] for b in bindings if b[1] == name}) > 1:
        return None
    return cands[0][2]


def convert(s):
    mask = code_mask(s)
    bindings = layout_bindings(s, mask)
    generics = generic_params(s, mask)
    edits = []  # (start, end, text)
    skipped = []

    def line_of(pos):
        return s.count("\n", 0, pos) + 1

    def classify(expr, pos):
        """(kind, dims, name) for a layout expression: kind is 'inline'
        (a static Layout.row_major), 'binding' (a comptime name bound to one;
        dims resolved), 'generic' (a `NAME: Layout` parameter), or None."""
        dims = static_row_major(expr)
        if dims is not None:
            return "inline", dims, None
        name = expr.strip()
        # A layout read off a type: `type_of(self.d).L_B3`,
        # `Data[DT, MD, N].L_CONTACTS`. Its binding becomes a RowMajor type,
        # so the expression is a layout type like a generic parameter.
        if re.fullmatch(r"(type_of\(.*\)|\w+\[.*\])\.\w+", name, re.S):
            return "generic", None, name
        bare = name[5:] if name.startswith("Self.") else name
        if not re.fullmatch(r"\w+", bare):
            return None, None, None
        if bare in ("DYN1", "DYN2"):  # runtime-extent layout TYPES (core.tiles)
            return "generic", None, name
        kind = name_kind(bare, pos, bindings, generics)
        if kind == "binding":
            dims = resolve_layout(name, pos, bindings)
            return ("binding", dims, name) if dims is not None else (None, None, None)
        if kind == "generic":
            return "generic", None, name
        return None, None, None

    # 1. LayoutTensor[...] types, constructors, stack allocations.
    for m in re.finditer(r"\bLayoutTensor\[", s):
        if not mask[m.start()]:
            continue
        lb = m.end() - 1
        rb = match_bracket(s, lb)
        if rb < 0:
            skipped.append((line_of(m.start()), "unbalanced"))
            continue
        body = "".join(c if mask[lb + 1 + k] or c == "\n" else " " for k, c in enumerate(s[lb + 1 : rb]))
        parts = split_top(body)
        pos_args = [x for x in parts if not re.match(r"^\w+\s*=", x)]
        kw = [re.sub(r"\s*=\s*", "=", x, count=1) for x in parts if re.match(r"^\w+\s*=", x)]
        if len(pos_args) < 2 or len(pos_args) > 3:
            skipped.append((line_of(m.start()), "param count"))
            continue
        dt, lay = pos_args[0], pos_args[1]
        origin = pos_args[2] if len(pos_args) > 2 else None
        kind, dims, name = classify(lay, m.start())
        if kind is None:
            skipped.append((line_of(m.start()), f"layout {lay[:50]}"))
            continue
        shared = kw == ["address_space=AddressSpace.SHARED"]
        extra = "".join(", " + k for k in kw)

        end = rb + 1
        tail = s[end:]
        if tail.startswith(".stack_allocation()"):
            if origin != "MutAnyOrigin" or dims is None or (kw and not shared):
                skipped.append((line_of(m.start()), "stack_allocation form"))
                continue
            fn = "shared_tile" if shared else "local_tile"
            edits.append(
                (m.start(), end + len(".stack_allocation()"), f"{fn}[{dt}, {dims}]()")
            )
        elif tail.startswith("("):
            cp = match_bracket(s, end)
            args = split_top(s[end + 1 : cp]) if cp > 0 else []
            if (
                cp < 0
                or len(args) != 1
                or kw
                or dims is None
            ):
                skipped.append((line_of(m.start()), "constructor form"))
                continue
            edits.append((m.start(), cp + 1, f"tile[{dims}]({args[0]})"))
        elif kind == "inline" and kw and not shared:
            if origin is None:
                skipped.append((line_of(m.start()), "kwargs, no origin"))
                continue
            edits.append(
                (m.start(), end, f"TileTensor[{dt}, RowMajor[{dims}], {origin}{extra}]")
            )
        elif kind == "inline":
            if origin == "MutAnyOrigin":
                tname = "SharedTile" if shared else "Tile"
            elif origin == "ImmutAnyOrigin" and not shared:
                tname = "ImmTile"
            else:
                skipped.append((line_of(m.start()), f"type origin {origin}"))
                continue
            edits.append((m.start(), end, f"{tname}[{dt}, {dims}]"))
        else:
            # A named layout keeps its name: it becomes a layout TYPE
            # (binding -> RowMajor[...], parameter -> TensorLayout).
            if origin is None:
                skipped.append((line_of(m.start()), "named type, no origin"))
                continue
            edits.append((m.start(), end, f"TileTensor[{dt}, {name}, {origin}{extra}]"))

    # 2. Tensor.lt / lt_at accessors.
    for m in re.finditer(r"\.(lt|lt_at)\[", s):
        if not mask[m.start()]:
            continue
        lb = m.end() - 1
        rb = match_bracket(s, lb)
        if rb < 0:
            continue
        parts = split_top(s[lb + 1 : rb])
        if len(parts) != 2:
            skipped.append((line_of(m.start()), f".{m.group(1)} params"))
            continue
        kind, dims, name = classify(parts[1], m.start())
        if kind in ("inline", "binding"):
            new = "tt" if m.group(1) == "lt" else "tt_at"
            edits.append((m.start(), rb + 1, f".{new}[{parts[0]}, {dims}]"))
        elif kind == "generic" and m.group(1) == "lt" and s.startswith("()", rb + 1):
            # A layout TYPE: pass its value to the runtime-layout accessor.
            # Needs a concrete type (`type_of(self.d).L_B3`); a bare generic
            # `L` parameter cannot be default-constructed, which the
            # compiler reports at the call.
            edits.append((m.start(), rb + 3, f".tt_dyn[{parts[0]}, {name}]({name}())"))
        else:
            skipped.append((line_of(m.start()), f".{m.group(1)} layout {parts[1][:40]}"))

    # 2b. `.lt_dyn[T, L](rl)` -> `.tt_dyn[T, L](rl)`: same parameters, the
    #     runtime layout value is now a TileTensor layout (`rl1` / `rl2`).
    for m in re.finditer(r"\.lt_dyn\[", s):
        if mask[m.start()]:
            edits.append((m.start(), m.end(), ".tt_dyn["))

    # 2c. LayoutTensor's `.element_type` is TileTensor's `.ElementType`.
    for m in re.finditer(r"\.element_type\b", s):
        if mask[m.start()]:
            edits.append((m.start(), m.end(), ".ElementType"))

    # 2d. `mut x: Tile[...]` -> `x: Tile[...]`. A MutAnyOrigin view writes
    #     through its pointer; `mut` only made the parameter demand an
    #     argument of EXACTLY that type, which a narrow-origin view (what
    #     `tile(list)` returns) is not. A body that reassigns the view
    #     itself fails to compile, so nothing changes silently.
    for m in re.finditer(r"\bmut (\w+): (Tile|SharedTile|TileTensor)\[", s):
        if not mask[m.start()]:
            continue
        rb = match_bracket(s, m.end() - 1)
        if rb < 0:
            continue
        if m.group(2) == "TileTensor" and "MutAnyOrigin" not in s[m.end() : rb]:
            continue
        edits.append((m.start(), m.start() + 4, ""))

    # 3. Layout declarations become layout types: `NAME: Layout` parameters
    #    -> `NAME: TensorLayout`; `comptime NAME = Layout.row_major(...)`
    #    -> `comptime NAME = RowMajor[...]`.
    for pos, name in generics:
        m = GENERIC.match(s, pos)
        if m.group(2):  # already TensorLayout (a re-run)
            continue
        a = m.start(0) + len(name)
        colon = s.index("Layout", a)
        edits.append((colon, colon + len("Layout"), "TensorLayout"))
    for m in BINDING.finditer(s):
        if not mask[m.start(1)]:
            continue
        p = m.start(2) + len("Layout.row_major")
        q = match_bracket(s, p)
        dims = static_row_major(s[m.start(2) : q + 1]) if q > 0 else None
        if dims is None:
            continue
        # drop an explicit `: Layout` annotation along with the value
        head = s[m.start(1) + len(m.group(1)) : m.start(2)]
        if ":" in head:
            edits.append((m.start(1) + len(m.group(1)), m.start(2), " = "))
        edits.append((m.start(2), q + 1, f"RowMajor[{dims}]"))

    # 4. Any other static `Layout.row_major(...)` left in code (a layout passed
    #    as a parameter) -> `RowMajor[...]`.
    for m in re.finditer(r"\bLayout\.row_major\(", s):
        if not mask[m.start()]:
            continue
        q = match_bracket(s, m.end() - 1)
        dims = static_row_major(s[m.start() : q + 1]) if q > 0 else None
        if dims is not None:
            edits.append((m.start(), q + 1, f"RowMajor[{dims}]"))

    # Apply, outermost first; drop edits nested inside another edit.
    edits.sort(key=lambda e: (e[0], -e[1]))
    kept, last_end = [], -1
    for e in edits:
        if e[0] >= last_end:
            kept.append(e)
            last_end = e[1]
    out, prev = [], 0
    for a, b, text in kept:
        out.append(s[prev:a])
        out.append(text)
        prev = b
    out.append(s[prev:])
    return "".join(out), len(kept), skipped


IMPORT = re.compile(r"^from ([\w.]+) import (\([^)]*\)|[^\n]*)\n", re.M)


def _names(group):
    return [x.strip() for x in group.strip("()").replace("\n", " ").split(",") if x.strip()]


def _code(s):
    mask = code_mask(s)
    return "".join(c if k else " " for c, k in zip(s, mask))


def _uses(code, name):
    """Uses of `name` outside import statements."""
    body = IMPORT.sub(lambda m: "\n", code)
    return re.search(rf"(?<![\w.]){name}\b", body) is not None


LEGACY = ("LayoutTensor", "Layout", "RuntimeLayout", "UNKNOWN_VALUE")


def fix_imports(s):
    code = _code(s)
    tiles = [n for n in TILE_NAMES if re.search(rf"(?<![\w.]){n}\[", IMPORT.sub("\n", code))]
    layout_need = [n for n in ("TileTensor", "TensorLayout") if _uses(code, n)]

    imported = set()
    for m in IMPORT.finditer(s):
        if m.group(1).startswith("layout") or m.group(1) == "noeira.core.tiles":
            imported |= set(_names(m.group(2)))

    def rewrite(m):
        mod, names = m.group(1), _names(m.group(2))
        if mod == "layout":
            names = [n for n in names if n not in LEGACY or _uses(code, n)]
            names += [n for n in layout_need if n not in imported and n not in names]
            layout_need.clear()
            # The tiles import goes right here, so it stays in the import
            # block even when this line ends up empty and disappears.
            if tiles and "noeira.core.tiles" not in s:
                extra = "from noeira.core.tiles import " + ", ".join(tiles) + "\n"
                tiles.clear()
                head = ""
                if names:
                    head = f"from {mod} import " + ", ".join(names)
                    if len(head) > 80:
                        head = f"from {mod} import (\n" + "".join(f"    {n},\n" for n in names) + ")"
                    head += "\n"
                return head + extra
        elif mod == "noeira.core.tiles":
            names += [n for n in tiles if n not in names]
            tiles.clear()
        elif mod.startswith("layout."):
            names = [n for n in names if n not in LEGACY or _uses(code, n)]
        else:
            return m.group(0)
        if not names:
            return ""
        line = f"from {mod} import " + ", ".join(names)
        if len(line) > 80:
            line = f"from {mod} import (\n" + "".join(f"    {n},\n" for n in names) + ")"
        return line + "\n"

    s = IMPORT.sub(rewrite, s)
    add = ""
    if layout_need:
        add += "from layout import " + ", ".join(layout_need) + "\n"
    if tiles:
        add += "from noeira.core.tiles import " + ", ".join(tiles) + "\n"
    if add:
        ims = list(IMPORT.finditer(s))
        lay = [m for m in ims if m.group(1).startswith("layout")]
        at = (lay[-1] if lay else ims[-1]).end() if ims else 0
        s = s[:at] + add + s[at:]
    return s


def main(argv):
    dry = "--dry-run" in argv
    paths = [Path(a) for a in argv if a != "--dry-run"]
    files = []
    for p in paths:
        files += sorted(p.rglob("*.mojo")) if p.is_dir() else [p]
    total_edits = total_skips = changed = 0
    for f in files:
        s = f.read_text()
        if not re.search(r"LayoutTensor|\.lt(_at|_dyn)?\[|Layout\.row_major|: Layout\b|^from layout|noeira\.core\.tiles", s, re.M):
            continue
        t, n, skipped = convert(s)
        t = fix_imports(t)
        total_edits += n
        total_skips += len(skipped)
        for line, why in skipped:
            print(f"SKIP {f}:{line}: {why}")
        if t != s:
            changed += 1
            if not dry:
                f.write_text(t)
    print(f"{changed} files changed, {total_edits} rewrites, {total_skips} left for hand migration")


if __name__ == "__main__":
    main(sys.argv[1:])
