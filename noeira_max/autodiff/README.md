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
| `rules/` | 42 VJP rules (linalg, elementwise, reduction, shape, indexing, nn), and the rules of Mojo custom ops (`custom.py`) |
| `kernels/` | Mojo custom ops for kernel-backed rules: LayerNorm, and noeira's fused attention. Each is a forward that returns its residuals and a backward that takes them |
| `optim.py` | SGD and AdamW emitted into the graph: schedule, bias correction and clipping from a device step counter |
| `train.py` | `build_train_step`: forward + backward + update as one graph, parameters and state as buffers updated in place; MEF export / load |
| `models/` | MLP and character GPT (dropout, in-graph batch sampling) as functions of a parameter dict |
| `bench/` | The GPT train step against torch (`bench_gpt_max.py`, `torch_twin_math.py`), compile time (`compile_scaling.py`, `buffer_order_compile_time.py`), kernel-backed rules (`layer_norm_kernel.py`, `attention_kernel.py`), single ops (`kernel_probe.py`), small MLP train steps at RL shapes (`mlp_step.py`, profiled with `nsys_summary.py`); `run_5090.sh` runs them on one GPU |
| `capi/` | The exported step, trained from a Mojo binary through the MAX C API with no Python in the process (`run.sh` exports, builds, runs and compares) |
| `tests/` | Gradcheck, torch golden VJPs, structural, vacuity, model parity, MAX findings |
| `probes/` | The first probes of MAX's graph internals: op names, buffers, the compile cache, a 40-line reverse walk |

Run the tests from the repo root (the torch side needs the `act-ref` env):

```bash
noeira_max/autodiff/run.sh -m unittest discover -s noeira_max/autodiff/tests -t .
```

`run.sh` runs Python through the main checkout's pixi env, so a git worktree
shares its MAX compile cache.
