"""The functional mode: networks whose shapes and wiring are RUNTIME values.

`Tape` records ops as they run and differentiates them in reverse; `FTensor`
is a handle on one node, with operators (`a + b`, `x @ w`, `(e ** 2).mean()`).
Comptime nn modules join a tape as ops (`tape.module[B](m)`), nn's
optimizers take the tape as a model, and on NVIDIA a step can be captured
into a CUDA graph. See `README.md`; `ftensor.mojo` for the surface,
`tape.mojo` for the engine, `gpu.mojo` for the device kernels,
`module_op.mojo` for the module bridge. `Tape()` runs on the CPU,
`Tape(ctx)` on the GPU — one binary, any shape. Experimental.
"""

from .ftensor import Tape, FTensor, FModule, NoGrad
