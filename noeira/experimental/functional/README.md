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
var w = tape.param[Kaiming](4, 16)                # runtime shapes
var b = tape.zeros(1, 16)
var ln = tape.module[B](LayerNorm[16].make["cpu", Kaiming]())  # a comptime module as an op
var h = ln(F.linear(x, w, b, "relu"))
var loss = ((h @ v - y) ** 2).mean()
loss.backward()
adam.step["cpu"](tape)                            # nn's optimizers take the tape
```

## What it is

| Piece | File | |
|---|---|---|
| `TapeData` | `tape.mojo` | nodes with runtime rank-2 shapes, numpy broadcasting; reverse pass by op tag; parameters persist across `reset`, which reuses every node buffer (no allocation in a steady-state step) |
| `Tape`, `FTensor` | `ftensor.mojo` | handles; `+ - * / @ ** -`, `.T()`, `.sum() .mean() .sum_rows()`, activations; `ParamWalkable`; graph capture |
| `F` | `F.mojo` | `F.linear(x, w, b[, act])` — one fused node with `Linear` / `LinearAct`'s launches — and `F.mse` |
| GPU kernels | `gpu.mojo` | every size a runtime argument; cuBLAS / cuBLASLt on NVIDIA at the comptime nn's TF32 rule, MAX's matmul over runtime views on Metal |
| module bridge | `module_op.mojo` | a comptime `Module` owned by the tape, called through thin function pointers to per-type thunks |

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

- Rank 2 only (no attention / conv as native ops — use them as modules).
- A module: batch fixed when handed over, ONE call per step (its caches hold
  one call), ARITY 1, fp32 activations.
- The op set is closed (a tag + forward + backward per op): Mojo has no trait
  objects. Activations reuse `ElementOp`.
- An `FTensor` from a previous step points at a slot `reset` reused.
- Capture: NVIDIA only; one eager step first; no host reads or host inputs
  inside the step (`input_device`); the optimizer must be ADOPTED;
  `capture_end()` replays the recorded step once (capture does not run it).

## What was learned

- `def(...) thin` function values are storable; `def(...)` is a trait.
- A handle returning an untracked pointer into tape-owned memory is a
  use-after-free once the handle is the owner's last reference (Mojo ends a
  value at its last use): return a `ref` whose origin is the handle.
- Fresh device buffers read as zero: a "stale memory" gate must warm up with
  the SAME shapes, or slots reallocate and the hazard hides.
- Stream capture records without running: a captured stateful step must be
  replayed once, and its gate compared step by step against eager.

## Gates

```
pixi run -e apple   mojo run -I . tests/experimental/functional/test_functional_mode.mojo        # CPU
pixi run -e apple   mojo run -I . tests/experimental/functional/test_functional_gpu.mojo         # GPU vs CPU, vs Sequential
pixi run -e apple   mojo run -I . tests/experimental/functional/test_functional_module.mojo      # module bridge, Adam
pixi run -e apple   mojo run -I . tests/experimental/functional/test_functional_fused.mojo       # fused linear, capture
pixi run -e default mojo run -I . tests/experimental/functional/test_functional_capture_adam.mojo  # captured training
```
