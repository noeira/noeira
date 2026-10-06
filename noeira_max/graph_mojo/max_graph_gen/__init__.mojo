"""MAX graphs built from Mojo, with ops generated from MAX's own op stubs.

- `ops`: GENERATED, one typed function per `rmo` op (`gen/gen_mojo_builder.py`);
- `backend`: `Value`, `TensorType`, `Dim`, the `GraphBackend` trait, the
  Python backend (MAX 26.6 builds graphs only in Python) and the C backend's
  interface;
- `api`: `Graph` (build, output, compile to a `maxrt` model) and the
  hand-written functions whose result types must be computed.

Build with `-I noeira_max/graph_mojo -I noeira_max/capi_mojo` and link
`-lmax`; run inside the pixi env with the repo root on PYTHONPATH.
"""

from .api import (
    Graph,
    add,
    cast,
    constant,
    div,
    exp,
    matmul,
    mul,
    reduce_max,
    reduce_mean,
    reduce_sum,
    relu,
    reshape,
    softmax,
    sqrt,
    sub,
    tanh,
    transpose,
)
from .backend import CBackend, Dim, GraphBackend, OpArgs, PythonBackend, TensorType, Value
