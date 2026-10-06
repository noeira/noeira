"""Test models for maxrt, exported as MEFs into OUT_DIR.

- ``add.mef``: ``(a + b) * 2`` over a symbolic dim ``n``.
- ``inplace.mef``: a ``[8]`` buffer stored in place, ``b += x``; outputs
  ``sum(b)`` and ``b * 2``.
- ``add_gpu.mef``, ``inplace_gpu.mef``: the same on the accelerator, when
  there is one.

    pixi run -e default python noeira_max/capi_mojo/maxrt_tests/make_mefs.py OUT_DIR
"""

from __future__ import annotations

import sys
from pathlib import Path

from max.driver import CPU, Accelerator, accelerator_count
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import BufferType, DeviceRef, Graph, TensorType, ops


def add_graph(dev: DeviceRef) -> Graph:
    spec = TensorType(DType.float32, ["n"], dev)
    with Graph("add", input_types=[spec, spec]) as g:
        a, b = g.inputs
        g.output((a + b) * 2.0)
    return g


def inplace_graph(dev: DeviceRef) -> Graph:
    types = [BufferType(DType.float32, [8], dev), TensorType(DType.float32, [8], dev)]
    with Graph("inplace", input_types=types) as g:
        b, x = g.inputs
        new = ops.buffer_load(b) + x
        ops.buffer_store(b, new)
        g.output(ops.sum(new, axis=0), new * 2.0)
    return g


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    targets = [("", CPU(), DeviceRef.CPU())]
    if accelerator_count():
        targets.append(("_gpu", Accelerator(), DeviceRef.GPU()))
    for suffix, device, ref in targets:
        session = InferenceSession(devices=[device])
        for name, build in (("add", add_graph), ("inplace", inplace_graph)):
            session.compile(build(ref)).export_mef(out / f"{name}{suffix}.mef")
            print(f"wrote {name}{suffix}.mef", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
