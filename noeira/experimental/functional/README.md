# `experimental/functional/` — an eager mode for `nn`

The comptime `nn` fixes every shape and the network's wiring at compile time:
`Sequential[Linear[64, 256], …]` is a type, and its backward pass is the
composition of its children's VJPs, also at compile time. This package is the
other mode: shapes are runtime values, the network is whatever code ran, and
a tape records the ops to differentiate them in reverse.

```mojo
from noeira.experimental.functional import Tape, FTensor, FModule
import noeira.experimental.functional.F as F

var tape = Tape()                                 # or Tape(ctx): the GPU
var w = tape.param[Kaiming](4, 16, name="w")      # runtime shapes; names key checkpoints
var b = tape.zeros(1, 16, name="b")
var ln = tape.module[B](LayerNorm[16].make["cpu", Kaiming](), name="ln")  # a comptime module as an op
var h = ln(F.linear(x, w, b, "relu"))
var loss = ((h @ v - y) ** 2).mean()
loss.backward()
adam.step["cpu"](tape)                            # nn's optimizers take the tape
save_params["cpu"](tape, path)                    # so do nn's checkpoints
with tape.no_grad():                              # inference: no grad buffers, no backward
    var act = F.linear(obs, w, b, "relu")
```

## What it is

| Piece | File | |
|---|---|---|
| N-d ops | `nd.mojo` | rank <= 4: broadcasting, batched matmul, `reshape`, `permute`, `transpose`, `sum / mean / max(axis)`, `narrow`, `F.concat`, `F.gather` |
| more ops | `extra.mojo` | `exp`, `log`, `sqrt`, `clamp`, `softmax`, `log_softmax`, `F.cross_entropy` (fused), `F.dropout` (seeded) |
| `TapeData` | `tape.mojo` | nodes with runtime shapes (rank <= 4; a 2-D view `rows × cols` for the matrix ops), numpy broadcasting; reverse pass by op tag; parameters persist across `reset`, which reuses every node buffer (no allocation in a steady-state step) |
| `Tape`, `FTensor` | `ftensor.mojo` | handles; `+ - * / @ ** -`, `.T()`, `.sum() .mean() .sum_rows()`, activations; `ParamWalkable`; graph capture |
| `F` | `F.mojo` | `F.linear(x, w, b[, act])` — one fused node with `Linear` / `LinearAct`'s launches — and `F.mse` |
| GPU kernels | `gpu.mojo` | every size a runtime argument; cuBLAS / cuBLASLt on NVIDIA at the comptime nn's TF32 rule, MAX's matmul over runtime views on Metal |
| module bridge | `module_op.mojo` | a comptime `Module` (1 to 3 inputs, up to 3 batch sizes) owned by the tape, called through thin function pointers to per-type thunks |

Activations are the comptime nn's `ElementOp` structs and the CPU GEMMs are
`gemm_cpu_ptr` (`nn/core/cpu_gemm.mojo`), so the two modes share that math.
The one change to `nn` core is the `IsParamVisitorRef` marker
(`nn/core/param.mojo`), which lets a walk hand runtime-sized parameters to the
erased visitor nn's optimizers and checkpoints use.

## Status (2026-10-16)

Gated on the CPU, Metal (M1 Pro) and an RTX 5090:

- bit-identical to the comptime nn: `Sequential[Linear, ReLU|Tanh, Linear]`
  forward and every gradient on the CPU, forward and dW on the GPU (db to
  ~3e-8, summation order); `Sequential[Linear, LayerNorm, Tanh, Linear]` vs
  the same modules on a tape, every gradient and the parameters after an Adam
  step, CPU and GPU; fused `F.linear` vs `Linear` / `LinearAct`;
- every op against central differences, with a mutant that must fail;
- a captured training step (forward, backward, nn's Adam ADOPTED into its
  arena) replayed on new inputs == eager, 20 steps, bit for bit; an
  un-adopted Adam captured the same way (host bias correction frozen) is
  caught by the gate.

Speed, functional / comptime (best of interleaved rounds):

| | CPU (M1) | Metal | 5090 eager | 5090 captured |
|---|---|---|---|---|
| acting, B=1 | 1.1-1.3 | 0.90 | 1.58 | 1.00 |
| PPO B=64 train (+Adam) | 1.04 | 0.88 | 1.16 | 1.13 |
| SAC B=256 train (+Adam) | 1.01 | 0.98 | 1.17 | 1.08 |
| B=4096 64-512-512-64 train | — | 1.08 | 1.14 | 1.15 |

Build: one functional MLP binary (20.5 s, Metal kernels included) serves any
width; the comptime one is ~11 s per width. Modules used through the bridge
are still compiled per shape.

Evidence: `noeira-docs/functional_nn_5090_2026-10-16/`.

## Limits

- Rank <= 4. `reshape`, `permute`, `narrow` and `concat` copy (a node owns
  its storage). Softmax and cross-entropy run one thread per row (simple,
  deterministic, not a tuned reduction); a long last axis is slow.
- `gather` indices are floats holding integers (exact to 2^24); out of range
  raises on the CPU and reads NaN on the GPU. Its gradient runs one thread
  per column, deterministic.
- A module: up to three batch sizes (`tape.module[64, 1](m)`), 1 to 3
  inputs, fp32 activations, ONE call per step with gradients on (its caches
  hold one call; under `no_grad` any number).
- The op set is closed (a tag + forward + backward per op): Mojo has no trait
  objects. Activations reuse `ElementOp`.
- Capture: NVIDIA only; one eager step first; no host reads or host inputs
  inside the step (`input_device`); the optimizer must be ADOPTED;
  `capture_end()` replays the recorded step once (capture does not run it);
  `F.dropout`'s seed is baked into the graph (every replay draws the
  captured mask).

## N-d tensors, more ops, multi-input modules (2026-10-16)

```mojo
var x = F.gather(emb, tokens) + pos                      # [B, T, D] + [T, D]
var q = F.linear(x, wq, bq).reshape([B, T, H, Dh]).permute([0, 2, 1, 3])
var att = ((q @ k.T()) * scale).softmax()                # batched, [B, H, T, T]
var loss = F.cross_entropy(logits, targets)              # [B, T, V] logits
var h = gru(x, h)                                        # a 2-input nn module
```

The integration gate (`test_functional_transformer.mojo`) is a transformer
written with these ops — embedding, positions, two attention heads, a
written-out LayerNorm, an MLP, cross-entropy, nn's Adam — that learns to
reverse 6 tokens of 8 (held-out accuracy 1.0) and, on NVIDIA, trains from a
captured graph bit-identically to eager.

## Safety and basics (2026-10-16)

- **Stale handles raise.** `reset` bumps the tape's step; an activation
  handle from an earlier step raises on any use instead of reading the node
  that now owns its slot. Parameters stay valid.
- **Names.** Parameters and modules take `name=` (defaults `p<i>`, `m<j>`;
  no `.`, no duplicates). The parameter walk runs in NAME order, so
  `save_params(tape)` / `load_params(tape)` work across a reordering of the
  code that creates them, and nn's reader refuses a renamed parameter.
- **`no_grad`.** `with tape.no_grad():` records plain data: no gradient
  buffer, no saved pre-activation, no backward, and a module may be called
  any number of times.

## What was learned

- `def(...) thin` function values are storable; `def(...)` is a trait.
- A handle returning an untracked pointer into tape-owned memory is a
  use-after-free once the handle is the owner's last reference (Mojo ends a
  value at its last use): return a `ref` whose origin is the handle.
- Fresh device buffers read as zero: a "stale memory" gate must warm up with
  the SAME shapes, or slots reallocate and the hazard hides.
- Stream capture records without running: a captured stateful step must be
  replayed once, and its gate compared step by step against eager.
- A module's input pack (`TensorRefs[N]`) needs its entries to share ONE
  origin: two locals cannot form one (nn uses a `TensorPack`); the bridge
  passes node storages as pointers with the same untracked origin.
- Two modules in one package may import each other (`tape.mojo` <->
  `nd.mojo`), which keeps the N-d ops out of the engine file.

## Gates

```
pixi run -e apple   mojo run -I . tests/experimental/functional/test_functional_mode.mojo        # CPU
pixi run -e apple   mojo run -I . tests/experimental/functional/test_functional_gpu.mojo         # GPU vs CPU, vs Sequential
pixi run -e apple   mojo run -I . tests/experimental/functional/test_functional_module.mojo      # module bridge, Adam
pixi run -e apple   mojo run -I . tests/experimental/functional/test_functional_fused.mojo       # fused linear, capture
pixi run -e default mojo run -I . tests/experimental/functional/test_functional_capture_adam.mojo  # captured training
pixi run -e apple   mojo run -I . tests/experimental/functional/test_functional_safety.mojo      # handles, names, checkpoint, no_grad
pixi run -e apple   mojo run -I . tests/experimental/functional/test_functional_nd.mojo          # N-d ops
pixi run -e apple   mojo run -I . tests/experimental/functional/test_functional_ops.mojo         # exp … dropout
pixi run -e apple   mojo run -I . tests/experimental/functional/test_functional_module_arity.mojo  # multi-input modules, batches
pixi run -e apple   mojo run -I . tests/experimental/functional/test_functional_transformer.mojo   # integration
```
