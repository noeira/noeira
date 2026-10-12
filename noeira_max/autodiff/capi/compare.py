"""Compares the Mojo driver's run (``mojo.txt``) with the same compiled step
driven from Python (``python_reference.txt``). Exits 1 on any mismatch.

    noeira_max/autodiff/run.sh noeira_max/autodiff/capi/compare.py OUT_DIR STEPS
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def main(out: Path, steps: int) -> int:
    ref_lines = (out / "python_reference.txt").read_text().split("\n")
    python = np.array([float(l) for l in ref_lines if l and not l.startswith("median")], np.float32)
    python_us = float(next(l for l in ref_lines if l.startswith("median")).split()[1])

    mojo, info = [], {}
    for line in (out / "mojo.txt").read_text().splitlines():
        key, _, rest = line.partition(" ")
        if key == "loss":
            mojo.append(float(rest.split()[1]))
        elif key:
            info[key] = rest
    mojo = np.array(mojo, np.float32)

    failures = []
    if len(mojo) != steps or len(python) != steps:
        failures.append(f"step counts: mojo {len(mojo)}, python {len(python)}, wanted {steps}")
    n = min(len(mojo), len(python))
    exact = int((mojo[:n] == python[:n]).sum())
    worst = float(np.max(np.abs(mojo[:n] - python[:n]) / np.abs(python[:n]))) if n else 0.0
    if worst > 1e-6:
        failures.append(f"losses differ: worst relative difference {worst:.2e}")
    counter = info.get("step_counter_in_mojo_memory")
    if counter is not None and float(counter) != steps:
        failures.append(f"step counter in Mojo's memory is {counter}, not {steps}")

    mojo_us = float(info.get("mojo_median_step_us", "nan"))
    print(f"losses: {exact}/{n} bit-identical, worst relative difference {worst:.1e}; "
          f"{mojo[0]:.4f} -> {mojo[-1]:.4f}")
    print(f"median step: Mojo -> C API {mojo_us:.1f} us, Python -> MAX {python_us:.1f} us "
          f"({python_us - mojo_us:+.1f} us per step for Python)")
    print(f"MEF load from Mojo: {float(info.get('mef_load_ms', 'nan')):.1f} ms")
    if counter is not None:
        print(f"step counter read from Mojo's own memory: {counter} (in place across calls)")
    if "capture_error" in info:
        print(f"capture: error: {info['capture_error']}")
    elif "capture_ok" in info:
        print(f"capture: ok, {info['capture_ok']}")
    for f in failures:
        print("FAIL:", f)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), int(sys.argv[2])))
