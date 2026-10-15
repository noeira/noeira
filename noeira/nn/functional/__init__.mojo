"""The functional mode: networks whose shapes and wiring are RUNTIME values.

`Tape` records ops as they run and differentiates them in reverse; `FTensor`
is a handle on one node, with operators (`a + b`, `x @ w`, `(e ** 2).mean()`).
See `ftensor.mojo` for the surface, `tape.mojo` for the engine and `gpu.mojo`
for the device kernels. `Tape()` runs on the CPU, `Tape(ctx)` on the GPU —
one binary, any shape. A prototype (branch `proto/functional-nn`).
"""

from .ftensor import Tape, FTensor
