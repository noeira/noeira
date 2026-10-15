"""The functional mode: networks whose shapes and wiring are RUNTIME values.

`Tape` records ops as they run and differentiates them in reverse; `FTensor`
is a handle on one node, with operators (`a + b`, `x @ w`, `(e ** 2).mean()`).
See `ftensor.mojo` for the surface and `tape.mojo` for the engine. CPU only —
a prototype (branch `proto/functional-nn`).
"""

from .ftensor import Tape, FTensor
