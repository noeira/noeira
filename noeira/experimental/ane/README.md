# ane — the functional tape's training step on the Apple Neural Engine

A spike (2026-10-25). It runs MIL programs on the ANE from pure Mojo, through
the Objective-C runtime (`experimental/metal_rt/objc.mojo`) and Apple's
**private** `AppleNeuralEngine.framework`. Any macOS update may break it. The
call sequence follows maderix/ANE (MIT, github.com/maderix/ANE).

The training step is split between the two:

- **ANE:** one recorded forward of `experimental/functional` becomes one MIL
  program (`emit.mojo`). Its inputs are the parameters, the data and a
  loss-scale seed. Its outputs are the loss and every parameter gradient.
- **Host:** the optimizer runs there, on fp32 master weights. Any CPU
  optimizer works, e.g. nn's `Adam`.

| File | What |
|---|---|
| `runtime.mojo` | `ANE` (frameworks, IOSurfaces), `Program` (compile, load, run; reads each I/O layout from the loaded model), `Layout` |
| `emit.mojo` | `emit_grad_step(tape, loss)`: the rank-2 ops of an MLP and its loss, with tape_to_mo's VJPs written as MIL; dead-code elimination |
| `step.mojo` | `AneGrad`: emit + compile, and per step: write the parameters, evaluate, write the gradients into the tape |
| `probe_ane.mojo` | Step 0: a matmul with its weights in the input, timed |
| `probe_io.mojo` | Which input/output shapes work (several I/O, odd sizes, both tensor forms) |
| `probe_ops.mojo` | Which ops compile, and the accuracy of exp/log/sqrt/sigmoid/tanh/reciprocal |
| `probe_mil.mojo` | Bisects a refused program (`ANECCompile() FAILED` names nothing) |
| `bench_ane_step.mojo` | Step 3: CPU tape vs ANE vs the metal_rt tape |
| `tests/experimental/ane/test_ane_step.mojo` | The gates |

```bash
pixi run -e apple mojo run -I . tests/experimental/ane/test_ane_step.mojo
pixi run -e apple mojo build -I . noeira/experimental/ane/bench_ane_step.mojo -o /tmp/bench_ane && /tmp/bench_ane
```

## What the M1 Pro's ANE (h13g, 16 cores) takes

- **I/O format.** Every input and output must be fp16: a program with fp32
  I/O does not compile. Tensors are rank 4. The emitter uses `[1, 1, R, K]`
  for the tape's `[R, K]`.
- **Row padding.** Each row is padded to 64 bytes. `Program` reads the
  strides from `modelAttributes` (`NetworkStatusList[0].LiveInputList`).
  A densely written surface of an odd width fails with "IOSurface smaller
  than the model expects".
- **Surface order.** Inputs and outputs are ordered by SYMBOL NAME, not by
  argument position. The emitter names them `i000…` and `o000…` so the two
  orders agree.
- **Refused programs.** The compile fails on:
  - `log` without an explicit `epsilon`: MIL's default, 1e-45, is not an
    fp16;
  - any fp16 constant above 65504 (a mean's 1/count, say);
  - a dead op whose operand also feeds a live reduce.

  The emitter avoids all three.
- **Accuracy of the ANE's own functions**, worst over 256 points, in fp16
  ulps of the result:

  | Function | Worst error | Where |
  |---|---|---|
  | sigmoid | 65 | in its tail |
  | tanh | 11 | near 0 |
  | log | 10 | near 1 |
  | exp | 5 | |
  | sqrt | 1.5 | |
  | reciprocal | 1.5 | |

  The matmul accumulates well (step 0: ≤ 0.5 ulp of the output).
- **Loss scale.** The seed is an input, so the scale changes without a
  recompile. When the loss is a mean, its 1/count is applied on the host,
  so the effective scale (seed × count) can go past 65504. A 1024 scale
  lost 10% of the every-op graph's first-layer gradient to underflow; at
  32768 the loss was 1.1%.

## Gates (`test_ane_step`): all pass

Each gate compares the ANE's gradients with the fp32 CPU tape's, at the same
weights, on three batches. The error is max |g_ane − g| / max |g| per
parameter; the bound is 3e-2. The "noise floor" column runs the fp32 tape
alone with every parameter and input perturbed by 2⁻¹¹, and reports how far
its own gradients move: no fp16 evaluation can do better than that.

| Gate | Worst gradient | fp32 noise floor (same parameter) |
|---|---|---|
| a. MLP 17-64relu-64tanh-6, B=64 | 4.5e-3 (w3) | 1.3e-2 (w1) |
| b. every other rank-2 op | 1.1e-2 (w1) | 6.7e-4 |
| d. MLP B=1024 64-256-256-64 | 2.1e-2 (w1) | 2.5e-2 (w1: relu kinks) |
| c. mutant: tanh's backward 1 + y² | 0.41: caught | |

**Training:** a sine fit (1-64tanh-64tanh-1, B=256, 2000 steps of Adam) from
the same init and batches. The ANE run ends at loss 4.7e-5, the CPU tape at
1.7e-4 (mean of the last 100 steps). Two ANE steps were skipped for overflow.

## Step 3: one training step (forward, backward, Adam), loss read every step

Median µs/step [min-max] over 5 rounds, in one process, with one fixed batch
already in place. The breakdown column is the ANE arm's mean per step, in µs:
fp16 writes / eval / gradient reads / Adam.

| Workload | CPU tape | ANE + host Adam | metal_rt tape | ANE breakdown | ANE eval TFLOP/s |
|---|---|---|---|---|---|
| PPO B=64 8-64-64-4 tanh | 66 | 458 | 562 | 12 / 422 / 16 / 5 | 0.004 |
| SAC B=256 23-256-256-1 relu | 348 | 859 | 856 | 42 / 645 / 96 / 44 | 0.17 |
| BIG B=4096 64-512-512-64 relu | 16378 | **5328** | 7484 | 136 / 4642 / 366 / 173 | 1.7 |
| WIDE B=2048 512-2048-2048-512 relu | — | **30132** | 36608 | 2242 / 17870 / 6554 / 3407 | 4.3 |

- **Small networks: the ANE loses.** One evaluation of a program with 9
  inputs and 7 outputs costs ~420 µs, even with almost nothing to compute.
  Step 0's 1-in/1-out matmul cost ~240 µs. The CPU tape is 3–7× faster here.
- **Large networks: the ANE wins.** It is 1.4× faster than the metal_rt tape
  at BIG and 1.2× at WIDE. The evaluation alone, at 4.3 TFLOP/s, would be
  2× faster than metal_rt at WIDE.
- **The host is the rest.** Its fp16 conversions and Adam are plain scalar
  loops: 12 ms of WIDE's 30 ms.

Not measured: power. It needs `sudo powermetrics` during a run.
