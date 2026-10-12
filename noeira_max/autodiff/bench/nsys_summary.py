"""Kernels per train step from an nsys profile of a MAX step (``mlp_step.py``,
``bench_gpt_max.py``): what ran, by kind, how many kernels per step and for how
long, which GEMM path each GEMM took, and the memsets and copies per step.

    nsys profile -o PREFIX --cuda-graph-trace=node <the command that runs the step>
    nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum \\
        --format csv --force-export=true --force-overwrite=true -o PREFIX PREFIX.nsys-rep
    python noeira_max/autodiff/bench/nsys_summary.py PREFIX [--steps N]

``--steps`` defaults to ``steps_done`` from ``PREFIX.json`` (``mlp_step.py
--json``). ``--cuda-graph-trace=node`` makes nsys list the kernels of a
replayed CUDA graph one by one; without it, a captured step shows none.

GEMM paths, by kernel name, on NVIDIA:
- ``multistage_gemm`` (split-K included): MAX's tiled path, taken when
  m > 1, n % 128 == 0, k % 32 == 0 and k >= 128. It runs float32 in TF32.
- cuBLAS or CUTLASS kernels: MAX's vendor path, everything else. Its
  precision cannot be read off the name; ``mlp_step.py``'s first-loss check
  against float64 measures it.
- ``naive``: MAX's untiled fallback (batched products), float32.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def category(name: str) -> str:
    n = name.lower()
    if n.startswith("mojo_pkg"):
        return "Mojo custom ops"
    if "multistage_gemm" in n:
        return "GEMM, MAX multistage (TF32)"
    if "naive" in n and ("gemm" in n or "matmul" in n):
        return "GEMM, MAX naive"
    if any(s in n for s in ("cublas", "cutlass", "sgemm", "xmma", "gemv", "gemm")):
        return "GEMM, cuBLAS / vendor"
    if "matmul" in n:
        return "GEMM, other MAX matmul"
    if "transpose" in n:
        return "transposes"
    if "mutable_store" in n:
        return "fused updates ending in a buffer store"
    if "rowwise" in n or "reduce" in n:
        return "reductions"
    return "elementwise and the rest"


def column(row: dict, prefix: str) -> str:
    """The value of the first column whose name starts with ``prefix``."""
    for key, value in row.items():
        if key and key.startswith(prefix):
            return value
    raise KeyError(prefix)


def read(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open())) if path.exists() else []


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("prefix", type=Path, help="the -o prefix given to nsys profile and nsys stats")
    ap.add_argument("--steps", type=int, help="steps in the profile (default: PREFIX.json's steps_done)")
    args = ap.parse_args()
    prefix = args.prefix
    steps = args.steps
    if steps is None:
        steps = json.loads(Path(f"{prefix}.json").read_text())["steps_done"]

    kernels = read(Path(f"{prefix}_cuda_gpu_kern_sum.csv"))
    if not kernels:
        raise SystemExit(f"no kernels in {prefix}_cuda_gpu_kern_sum.csv")
    by_kind = defaultdict(lambda: [0, 0.0])
    irregular = []
    gemms = []
    total_us = 0.0
    for row in kernels:
        name = column(row, "Name")
        instances = int(column(row, "Instances"))
        us = float(column(row, "Total Time")) / 1000.0 / steps
        kind = category(name)
        by_kind[kind][0] += instances
        by_kind[kind][1] += us
        total_us += us
        if instances % steps:
            irregular.append((instances, name))
        if kind.startswith("GEMM"):
            gemms.append((instances / steps, float(column(row, "Avg")) / 1000.0, kind, name))
    launches = sum(count for count, _ in by_kind.values()) / steps

    print(f"{prefix.name}: {launches:.1f} kernel launches and {total_us:.1f} µs of kernel time "
          f"per step ({steps} steps)")
    print(f"  {'kind':42s} {'per step':>9s} {'µs/step':>9s} {'share':>6s}")
    for kind, (count, us) in sorted(by_kind.items(), key=lambda kv: -kv[1][1]):
        print(f"  {kind:42s} {count / steps:9.1f} {us:9.1f} {100 * us / total_us:5.1f}%")
    print("  GEMM kernels (per step, average µs, path):")
    for per_step, avg_us, kind, name in sorted(gemms, key=lambda g: -g[0] * g[1]):
        print(f"    {per_step:5.1f} x {avg_us:8.2f} µs  {kind[6:]:20s} {name[:110]}")
    if irregular:
        print("  not once per step (one-time, or a count that does not divide by the steps):")
        for instances, name in sorted(irregular, reverse=True)[:10]:
            print(f"    {instances:6d}  {name[:110]}")

    memory = {}
    times = read(Path(f"{prefix}_cuda_gpu_mem_time_sum.csv"))
    sizes = {column(r, "Operation"): float(column(r, "Total")) for r in read(Path(f"{prefix}_cuda_gpu_mem_size_sum.csv"))}
    if times:
        print("  memory operations per step:")
        for row in times:
            op = column(row, "Operation")
            count = int(column(row, "Count")) / steps
            us = float(column(row, "Total Time")) / 1000.0 / steps
            mb = sizes.get(op, 0.0) / steps
            memory[op] = dict(per_step=count, us_per_step=us, mb_per_step=mb)
            print(f"    {op:36s} {count:7.2f} per step  {mb:9.2f} MB  {us:8.1f} µs")

    print("RESULT " + json.dumps(dict(
        profile=prefix.name, steps=steps, launches_per_step=launches, kernel_us_per_step=total_us,
        kinds={k: dict(per_step=c / steps, us_per_step=u) for k, (c, u) in by_kind.items()},
        gemms=[dict(per_step=p, avg_us=a, path=k[6:], name=n) for p, a, k, n in gemms],
        memory=memory,
    )), flush=True)


if __name__ == "__main__":
    main()
