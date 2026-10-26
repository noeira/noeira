#!/usr/bin/env python3
"""Kernel launches that pass a TileTensor view where the kernel declares a Pointer.

A LayoutTensor argument was accepted for a `Pointer[...]` kernel parameter, so
`enqueue_function[k](t.lt[...](), ...)` compiled against a pointer-typed `k`.
A TileTensor is NOT accepted there. The error appears only when the launch is
instantiated, which `mojo precompile` never does, so the package build stays
green. This finds them statically: each `enqueue_function[NAME](args)` is
resolved to NAME's definition (same file, an import, or a `comptime` alias),
and every argument that is a `.tt*[...]()` / `tile[...](...)` view at a
`Pointer` parameter position is reported. Fix: pass the view's `.ptr`.

Run from the repo root: python3 tools/migrate/check_pointer_kernel_args.py
Exit status 1 when anything is found.
"""
import re, sys, pathlib
sys.path.insert(0, 'tools/migrate')
from lt_to_tile import match_bracket, split_top, code_mask
roots = ['noeira', 'tests', 'examples', 'benchmarks']
files = [f for r in roots for f in pathlib.Path(r).rglob('*.mojo')]
defs = {}  # (file, name) -> set of pointer positions
def kernel_defs(f, s):
    for m in re.finditer(r'^def (\w+)(\[)?', s, re.M):
        i = m.end()
        if m.group(2):
            j = match_bracket(s, i - 1)
            if j < 0: continue
            i = j + 1
        if i >= len(s) or s[i] != '(': continue
        k = match_bracket(s, i)
        if k < 0: continue
        params = [p for p in split_top(s[i + 1:k]) if p]
        defs[(str(f), m.group(1))] = {pos for pos, p in enumerate(params) if re.search(r':\s*(Unsafe)?Pointer\[', p)}
texts = {}
for f in files:
    s = f.read_text(); texts[str(f)] = s; kernel_defs(f, s)
def module_file(mod, cur):
    if mod.startswith('.'):
        base = pathlib.Path(cur).parent
        dots = len(mod) - len(mod.lstrip('.'))
        for _ in range(dots - 1): base = base.parent
        rest = mod.lstrip('.')
        p = base / (rest.replace('.', '/') + '.mojo') if rest else base / '__init__.mojo'
    else:
        p = pathlib.Path(mod.replace('.', '/') + '.mojo')
    return str(p)
VIEW = re.compile(r'(\.tt(_at|_dyn)?\[.*\]\(.*\)|^tile\[.*\]\(.*\))\s*$', re.S)
hits = 0
for f in files:
    s = texts[str(f)]; mask = code_mask(s)
    imports = {}
    for m in re.finditer(r'^\s*from ([\w.]+) import (\([^)]*\)|[^\n]*)', s, re.M):
        for n in m.group(2).strip('()').replace('\n', ' ').split(','):
            n = n.strip().split(' as ')
            if n[0]: imports[n[-1].strip()] = (module_file(m.group(1), str(f)), n[0].strip())
    aliases = {m.group(1): m.group(2) for m in re.finditer(r'^\s*comptime (\w+) = ([\w]+)\[', s, re.M)}
    for m in re.finditer(r'enqueue_function\[\s*([\w]+)', s):
        if not mask[m.start()]: continue
        name = m.group(1)
        for _ in range(3):
            if name in aliases and (str(f), name) not in defs: name = aliases[name]
        key = (str(f), name) if (str(f), name) in defs else (imports.get(name, (None, None)))
        pos_set = defs.get(key if key[0] else ('', ''), set()) if key else set()
        if not pos_set: continue
        lb = s.index('[', m.start()); rb = match_bracket(s, lb)
        if rb < 0 or s[rb + 1:rb + 2] != '(': continue
        cp = match_bracket(s, rb + 1)
        args = [a for a in split_top(s[rb + 2:cp]) if not re.match(r'^\w+\s*=', a)]
        bad = [p for p in sorted(pos_set) if p < len(args) and VIEW.search(args[p])]
        if bad:
            hits += 1
            print(f'{f}:{s.count(chr(10), 0, m.start()) + 1} {name} <- {key[0]} args{bad}')
print(hits, 'launches')
sys.exit(1 if hits else 0)
