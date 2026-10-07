# train_from_mojo — train steps defined in Mojo, trained on MAX

A Mojo program builds its whole train step as one MAX graph:
- the parameter and Adam-moment buffers;
- the model's forward pass, Mojo custom-op kernels included;
- the loss;
- AdamW's update, stored into the buffers in place.

It builds the graph with the generated builder (`../graph_mojo/`). MAX compiles the step once, and `maxrt` (`../capi_mojo/`) runs every step with no Python, updating memory the program owns.

One step is still Python, at setup: differentiation. `max_train.Tape` records the graph's ops before the forward pass. `Tape.gradients` then hands the ops emitted since to the autodiff prototype's transform (`../autodiff/transform.py`), which emits their VJPs into the same graph. The VJP rules live there, in Python. A rule registry in MAX's graph layer would make this one C call.

```mojo
var tape = Tape()                                      # after loading the parameters
var loss = cross_entropy(forward(p, idx, layers, kernels), targets)
var grads = tape.gradients(primals, loss)              # the one call into Python
AdamW(3e-4, 0.9, 0.999, 1e-8).apply(params, m, v, step, grads)
g.output([loss^])
var model = g.compile(rt, "gpt_train_step.mef")        # then: model.execute(inputs), no Python
```

## What is checked

Each program builds a step in Mojo and the same step with the Python prototype (`py_grad.py`), from the same weights and batch, then compares:
- **the MLIR text of the two graphs**, which must be identical: the same graph, so MAX's compile cache serves the second from the first;
- **every step's loss**, bit for bit;
- **every final parameter, moment and the step counter**, bit for bit.

Every program passes on the Apple M1's CPU and on an RTX 5090 (`--gpu`, buffers on the device, lent by address). On CUDA both sides of the gate run in TF32, and they still agree bit for bit.

| Program | Step | Result (M1 CPU; RTX 5090) |
|---|---|---|
| `train_mlp.mojo --shape small/ppo/sac` | MLP, MSE, AdamW: 8-16-16-4 ReLU; PPO 8-64-64-4 tanh; SAC critic 23-256-256-1 ReLU | identical text; 100 of 100 losses and every buffer bit-identical |
| `train_mlp.mojo --mutate` | the same with twice the learning rate | caught: text, losses and buffers all differ |
| `train_ln_mlp.mojo` | dense, noeira's LayerNorm kernel pair (custom ops), ReLU, dense | identical text, both kernels in it; bit-identical |
| `train_gpt.mojo` | the autodiff prototype's 2-layer GPT, with noeira's LayerNorm and attention kernel pairs, a fixed batch | identical text, all four kernels in it; 30 of 30 losses bit-identical |
| `train_nn_mlp.mojo` | the SAC critic as three noeira nn layers; the step's parameter buffers are the layers' own `Param` memory | bit-identical to the Python step; nn's forward on the trained weights agrees with a float64 forward to 5.7e-7, where training moved the output by 2.56 (relative) |

```bash
noeira_max/train_from_mojo/run.sh [--shape small|ppo|sac] [--steps N] [--gpu]   # MLP, mutant, LayerNorm pair, GPT
noeira_max/train_from_mojo/run.sh --nn [--steps N] [--gpu] [--no-bump]          # the nn hybrid
```

`--gpu` copies every buffer to the device once and lends its device address (the nn hybrid lends nn's own device buffers). On Metal, MAX 26.6 cannot share device memory through the C API (`../capi_mojo/`), so these programs cannot keep their state on an Apple GPU.

## Files

| File | What |
|---|---|
| `max_train/train.mojo` | `Tape`; `mse`, `cross_entropy`; `AdamW`. Every op is emitted in the order the Python prototype emits it |
| `max_train/mlp.mojo` | The MLP step at the shapes of noeira's RL networks |
| `max_train/gate.mojo` | Lending buffers to `maxrt`, reading them back, comparing a run with the reference |
| `py_grad.py` | The Python half: `snapshot` and `backward` (the transform), the problems, and the reference runs |
| `train_mlp.mojo`, `train_ln_mlp.mojo`, `train_gpt.mojo`, `train_nn_mlp.mojo` | The programs above |

## Notes

- **The builder needed graph-level state, not only ops.**
  - Buffer loads and stores are chained, and `max.graph` keeps the chain on the graph, one per device. `max_graph_gen.buffer_load` and `buffer_store` use the generated `mo_mutable_load` / `mo_mutable_store` ops, and read and advance that chain through the Python backend.
  - Custom ops (`mo.custom`) and casts (`mo.cast`) come from the Python backend, like constants. `max.graph.ops.cast` emits the `mo` dialect's `mo.cast`. The generated `rmo.mo.cast` is a different op that computes the same values, so a graph built with it trains identically but has different text, and so a different compile-cache key.
  - A rank-0 constant must be made from a Python number: a 0-d NumPy array becomes shape `[1]`.
- **The transform loads a dict of parameters in sorted key order** (`tree_utils.flatten` sorts keys). The Mojo builders load in that order, so that the graphs' text matches.
- **nn's derived weights.** nn caches derived copies of a weight (zero-padded on the GPU, bf16) and refreshes them when the optimizer bumps `Param.version`. A MAX step writes the weights behind nn's back, so `train_nn_mlp` bumps the versions after training. The bump changed nothing on either device: on the CPU no copy is cached, and on the 5090 nn's forward for these shapes goes through cuBLAS with no padded copy. `--no-bump` skips it, for shapes and paths that do cache one.
- **Mojo frees a value at its last use, before a call that only received its address.** This happened twice at the Python boundary:
  - a NumPy array was released inside the argument list of `HostBuffer(copy_from=address(a), nbytes=a.nbytes)`, before the copy ran;
  - an nn output tensor was released before Python copied it.

  Each read stale bytes, with no error: small arrays survived, and the first element of the large one did not. Every such read now goes through a helper that receives the owner (`host_copy`, `to_numpy`).
- **A conditional expression with a compile-time condition returned a dead `DeviceContext`.** `Optional(DeviceContext()) if gpu else None`, with `gpu` a `comptime` value, gave a context whose first buffer crashed inside AsyncRT: SIGSEGV on CUDA, SIGTRAP on Metal. The same expression with a runtime condition works, and so does `comptime if`, which `train_nn_mlp` now uses (Mojo 1.1.0).
