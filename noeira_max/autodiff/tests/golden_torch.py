"""Torch side of the golden test: ``torch.func.vjp`` on the inputs MAX used.

Runs in the ``act-ref`` env and never imports MAX; ``test_golden.py`` drives
it. It loads ``cases.py`` by path, so the package's ``__init__`` (which does
import MAX) never runs here.

    env -u LD_PRELOAD .pixi/envs/act-ref/bin/python \\
        noeira_max/autodiff/tests/golden_torch.py IN.npz OUT.json
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

RTOL = 1e-6


def load_cases():
    path = Path(__file__).with_name("cases.py")
    spec = importlib.util.spec_from_file_location("autodiff_cases", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up
    spec.loader.exec_module(module)
    return module


def compare(ours: np.ndarray, theirs: torch.Tensor) -> dict:
    """Elementwise: ``|a - b| <= 1e-9 * max|b| + RTOL * |b|`` passes when
    ``ratio <= 1``. ``normwise`` is ``max|a - b| / max|b|``, for reporting."""
    b = theirs.detach().numpy()
    if ours.shape != b.shape:
        return {"shape_mismatch": [list(ours.shape), list(b.shape)], "ratio": float("inf")}
    scale = max(float(np.abs(b).max(initial=0.0)), 1e-300)
    diff = np.abs(ours - b)
    ratio = float((diff / (1e-9 * scale + RTOL * np.abs(b))).max(initial=0.0))
    return {
        "ratio": ratio,
        "normwise": float(diff.max(initial=0.0) / scale),
        "max_abs": float(diff.max(initial=0.0)),
    }


def main(npz_path: str, out_path: str) -> None:
    cases = load_cases()
    data = np.load(npz_path)
    results = []
    for family in cases.FAMILIES:
        for case in family.cases:
            if case.torch_fn is None:
                continue
            diff = [j for j, a in enumerate(case.args) if a.diff]
            b = 0
            while f"{family.name}.{case.name}.{b}.r" in data:
                prefix = f"{family.name}.{case.name}.{b}"
                xs = [
                    torch.from_numpy(data[f"{prefix}.x{j}"])
                    for j in range(len(case.args))
                ]

                def f(*values, xs=xs):
                    full = list(xs)
                    for j, value in zip(diff, values):
                        full[j] = value
                    return case.torch_fn(torch, *full)

                y, vjp_fn = torch.func.vjp(f, *(xs[j] for j in diff))
                r = torch.from_numpy(data[f"{prefix}.r"]).reshape(y.shape)
                grads = vjp_fn(r)
                results.append({
                    "family": family.name,
                    "case": case.name,
                    "binding": b,
                    "y": compare(data[f"{prefix}.y"], y),
                    "grads": [
                        compare(data[f"{prefix}.g{k}"], g)
                        for k, g in enumerate(grads)
                    ],
                })
                b += 1
    Path(out_path).write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    main(sys.argv[1], sys.argv[2])
