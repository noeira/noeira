# noeira_max.autodiff — reverse-mode AD for MAX graphs, as a graph transform

A prototype (MAX 26.6). `value_and_grad` runs while a graph is being built,
inside a `max.graph.Graph` or a function given to
`max.experimental.compilation.compile`. It walks the ops the function emitted
and emits each one's VJP into the same graph, so forward, backward and update
compile into one model.

```python
from max.graph import ops
from noeira_max.autodiff import value_and_grad

def loss(params, x, y):
    return cross_entropy(ops.matmul(x, params["w"]) + params["b"], y)

# inside the graph being built:
value, grads = value_and_grad(loss)(params, x, y)   # grads: same keys as params
```

| Path | What |
|---|---|
| `_graph.py` | Every private MAX API the transform touches, in one module |
| `transform.py` | `vjp`, `value_and_grad`, `grad` |
| `registry.py` | `defvjp(name)` / `nondiff(name)`; rules are keyed by MLIR op name |
| `rules/` | 42 VJP rules: linalg, elementwise, reduction, shape, indexing, nn |
| `models/` | MLP and character GPT as functions of a parameter dict |
| `tests/` | Gradcheck, torch golden VJPs, structural, vacuity, model parity, MAX findings |
| `m0/` | The go/no-go probes |

Run the tests from the repo root (the torch side needs the `act-ref` env):

```bash
noeira_max/autodiff/run.sh -m unittest discover -s noeira_max/autodiff/tests -t .
```

`run.sh` runs Python through the main checkout's pixi env, so a git worktree
shares its MAX compile cache.
