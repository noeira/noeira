"""Train steps defined in Mojo, compiled by MAX, run by `maxrt`.

- `train`: `Tape`, the gradient of the ops a Mojo program emitted, through
  the autodiff prototype's transform (Python, at setup only); `mse` and
  `AdamW`, emitted with `max_graph_gen`;
- `mlp`: an MLP train step at the shapes of noeira's RL networks;
- `gate`: lending the step's buffers to `maxrt`, and checking a run against
  the Python prototype's.

Build with `-I noeira_max/train_from_mojo -I noeira_max/graph_mojo
-I noeira_max/capi_mojo`, link `-lmax`, and run inside the pixi env with the
repo root on PYTHONPATH.
"""

from .gate import check, host_copy, lend, median_us, pipelined_us, read, train
from .mlp import (
    Shape, build_mlp_step, input_shape, numbers, param_dims, param_name, shape_named, state_names,
    step_name,
)
from .train import AdamW, Tape, cross_entropy, mse
