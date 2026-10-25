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
| backend | `backend.mojo` | `Dev`: launch through MAX's context or noeira's `metal_rt` launcher (`Tape.metal_rt()`) |
| `TapeAdam` | `optim.mojo` | nn's per-parameter Adam, launched through the tape's `Dev` (runs on a metal_rt tape) |
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

## On Metal without MAX's runtime: `Tape.metal_rt` (2026-10-25)

```mojo
var tape = Tape.metal_rt()                     # COMMIT_EVERY_N, N = 16
var w = tape.param[Kaiming](23, 256, name="w") # same API from here on
...
loss.backward()
opt.step(tape)                                 # TapeAdam: nn's Adam kernel, launched by the tape
var l = loss.item()                            # a read is the sync
```

A GPU tape whose every kernel launch, buffer and copy goes through noeira's
own Metal launcher (`experimental/metal_rt`) instead of MAX's runtime.

**How.** The GPU functions (`gpu.mojo`, `nd.mojo`, `extra.mojo`) take a `Dev`
(`backend.mojo`) where they took a `DeviceContext`, and call the same
`enqueue_function` / `enqueue_copy` / `synchronize` on it, so not one kernel
call site changed. `Dev` forwards to MAX, or to `MetalRT`. On a metal_rt
tape:

- **Storage.** It is MetalRT's MTLBuffers, held as NON-OWNING `DeviceBuffer`s
  over their GPU address (`DeviceBuffer(ctx, ptr, n, owning=False)`), so
  `Tensor`, `create_sub_buffer` views and `unsafe_ptr()` work unchanged. A
  `DeviceBuffer` kernel argument is bound as its address. Other arguments
  are bound as their own bytes.
- **Copies.** Host -> device goes through a staging ring plus a copy kernel,
  ordered like a launch. Device -> host syncs, then reads shared memory.
- **Matmul.** It is the tape's own GEMM (`ctx.own`): a new
  `_tiled_mm_kernel` (64 x 64 tiles, 4 x 4 per thread, k in order), with
  `_naive_mm_kernel` below 8 tiles or with a dimension of 1. MAX's matmul
  launches through MAX and cannot run here.
- **Bias gradient.** It is nn's `enqueue_bias_grad`, re-issued through `Dev`:
  the same two kernels and grids.
- **Optimizer.** It is `TapeAdam` (`optim.mojo`): nn's per-parameter Adam,
  with the same host step state and the same kernel
  (`_adam_update_kernel_rt`), launched through `Dev`.
- **Refused, with a clear error:** module ops, nn's optimizers, checkpoint
  walks, and capture. They launch through MAX.

`Tape(ctx, own_kernels=True)` runs the same replacements on MAX's launch
path. The bit-identity gate uses it.

What still touches MAX on a metal_rt tape:
- the `DeviceContext` object, which types the buffer handles but runs no
  GPU work;
- the host-side `DeviceBuffer` handle API;
- `_compile_code`, the compiler entry point that produces the metallib.

Nothing GPU-side touches MAX.

**Gates** (`tests/experimental/functional/test_functional_metal_rt.mojo`):

- **(a) Bit identity: 18 of 18 runs.** Three networks: PPO 8-64-64-4 tanh,
  SAC 23-256-256-1 relu, and an unfused graph (matmul, broadcast, tanh,
  softmax, sigmoid, log, transpose, sum_rows, mean, pow). Each runs under
  three commit policies (each, every-16, adaptive), with the loss read every
  step or every 3. Against the own-kernel MAX tape they agree BIT FOR BIT on
  the forward output, every step-0 gradient, every loss read, and every
  parameter after 8 Adam steps.
- **(b) Tolerance vs the normal MAX path** (MAX matmul, nn's Adam adopted).
  Measured, relative to the largest reference magnitude: forward 0
  (identical), gradients ≤ 4e-8, losses ≤ 8e-8, parameters ≤ 6e-5 of the
  run's total `lr`. The gate's bounds are 1e-5 / 1e-5 / 1e-4 / 1e-3.
- **(c) Refusals:** nn's Adam and a module op both raise on a metal_rt tape.
- **Negative control:** the same runs with no ordering between command
  buffers differ from MAX, and the gate sees it. This control is reported,
  not gated, because the failure is a race.
- **The MAX path did not regress.** `test_functional_gpu`, `_fused`,
  `_module`, `_nd`, `_ops`, `_safety`, `_mode` and `_transformer` pass.
  `test_functional_module_arity` aborts after its CPU part ("tcmalloc: Attempt
  to free invalid pointer 0x1"); it aborts the same way on the commit before
  this change.

**Speed** (`benchmarks/functional/bench_functional_metal_rt.mojo`, M1 Pro):

- Units are us per step, median of 7 rounds.
- Each arm runs in its OWN process (`--arm K`). MAX's per-launch cost grows
  with every live MAX buffer, and the interleaved run, with all six arms
  alive, slows the MAX arms by 10–60%.
- "tape MAX" is the normal path: MAX matmul and nn's adopted Adam.
- The metal_rt arm shown is the best of the three policies: every-64. The
  only syncs are the reads; no `flush()`.

| workload | loop | comptime nn (MAX) | tape MAX | tape on metal_rt (every-64) | vs tape MAX |
|---|---|---|---|---|---|
| ACT B=1 23-256-256-1 fwd | end | 222 | 235 | 162 | 1.4x |
| ACT B=1 | read every step | 607 | 698 | 324 (every-16) | 2.2x |
| PPO B=64 8-64-64-4 train | end | 1606 | 1088 | 244 | 4.5x |
| PPO B=64 train | read/20 | 1624 | 1112 | 245 | 4.5x |
| PPO B=64 train | read/1 | 2067 | 1537 | 408 | 3.8x |
| PPO B=64 train + Adam | read/20 | 1739 | 1149 | 267 | 4.3x |
| PPO B=64 train + Adam | read/1 | 2187 | 1590 | 403 | 3.9x |
| SAC B=256 23-256-256-1 train | read/20 | 1280 | 1078 | 782 | 1.4x |
| SAC B=256 train + Adam | read/20 | 1368 | 1103 | 830 | 1.3x |
| SAC B=256 train + Adam | read/1 | 1943 | 1703 | 966 | 1.8x |
| B=4096 64-512-512-64 train | read/20 | 8981 | 9857 | 16386 | 0.6x |

- **Small steps.** The metal_rt tape runs PPO-sized steps 4–4.5x faster than
  the tape on MAX, and 6.6x faster than the comptime nn on MAX.
- **B=4096 is GEMM-bound and slower** (0.6x). The tape's GEMM runs at
  0.73 TFLOP/s on [4096, 512] @ [512, 512], against 2.5 for MAX's matmul.
  The own-kernel MAX arm, 17.8 ms, shows it is the kernel and not the
  launcher. 8 x 8 outputs per thread and 32-deep k-slabs were both slower.
  MAX's speed most likely needs Apple's simdgroup matrix instructions, which
  were not attempted here.
- **Noise.** The read-every-step rows spread ±20–40% between rounds; see the
  brackets in the bench output.

**Pitfalls found on the way** (details in `metal_rt/README.md`):

- **An autorelease pool spanning an open command buffer.** It crashed as soon
  as two launchers interleaved. Command buffers are now retained instead.
- **A buffer allocated while a command buffer is open.** The residency set
  did not cover it yet. Probable cause of a rare loss/parameter divergence
  at step 2; fixed by committing first.
- **A second `DeviceContext` is a second queue.** nn's Adam on another
  context than the tape's raced the tape in the first version of gate (b).

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
pixi run -e apple   mojo run -I . tests/experimental/functional/test_functional_metal_rt.mojo      # the tape on metal_rt
```
