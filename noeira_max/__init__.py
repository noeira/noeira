"""noeira_max — a small MAX-based package for evaluating MAX as an inference backend
for noeira, driven from Mojo via Python interop.

v1 scope: MLP *inference* only. Training on MAX is prototyped separately, in
``noeira_max/autodiff``.

The public surface is :class:`noeira_max.mlp_inference.MLPInference`, a configurable
MLP whose dims/batch/device are all variables so multiple shapes can be swept,
plus timing primitives that let the Mojo caller attribute latency across:

  * pure MAX device compute,
  * host<->device data transfer (H2D / D2H),
  * the Mojo<->Python interop bridge itself.
"""

from .mlp_inference import MLPInference

__all__ = ["MLPInference"]
