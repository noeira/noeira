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
| `benchmarks/ane/probe_ane.mojo` | Step 0: a matmul with its weights in the input, timed |
| `benchmarks/ane/probe_io.mojo` | Which input/output shapes work (several I/O, odd sizes, both tensor forms) |
| `benchmarks/ane/probe_ops.mojo` | Which ops compile, and the accuracy of exp/log/sqrt/sigmoid/tanh/reciprocal |
| `benchmarks/ane/probe_mil.mojo` | Bisects a refused program (`ANECCompile() FAILED` names nothing) |
| `benchmarks/ane/probe_zero_copy.mojo` | GPU (metal_rt) and ANE sharing the surfaces with no copy |
| `gpu_adam.mojo` | `AneAdamGpu`: the step with fp32 masters and Adam on the GPU, zero-copy to and from the surfaces |
| `tests/experimental/ane/test_ane_gpu_adam.mojo` | Its gates |
| `benchmarks/ane/bench_ane_step.mojo` | Step 3: CPU tape vs ANE vs the metal_rt tape |
| `tests/experimental/ane/test_ane_step.mojo` | The gates |

```bash
pixi run -e apple mojo run -I . tests/experimental/ane/test_ane_step.mojo
pixi run -e apple mojo build -I . benchmarks/ane/bench_ane_step.mojo -o /tmp/bench_ane && /tmp/bench_ane
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

## Zero copy between the GPU (metal_rt) and the ANE (`probe_zero_copy.mojo`)

Can a GPU kernel write the ANE's input surfaces and read its output surface
in place, with no CPU copy? **Yes.**

**The route: a buffer, not a texture.** `MetalRT.wrap_external(addr, nbytes)`
wraps a surface's memory with `newBufferWithBytesNoCopy:length:options:
deallocator:`:
- shared storage and a nil deallocator, on the surface's page-aligned base
  and `IOSurfaceGetAllocSize` (whole 16 KB pages since f23f04c99);
- `contents` comes back equal to the IOSurface's base address, so no copy;
- the buffer gets a `gpuAddress` and joins the residency set like any other
  metal_rt buffer;
- `free` releases the MTLBuffer object only; the IOSurface keeps owning the
  memory.

The texture route (`newTextureWithDescriptor:iosurface:plane:`) was not
needed and was not tried.

**Checks** (linear program `y = x @ w + b`, B×I @ I×O; all pass):

| Check | Result |
|---|---|
| a. GPU -> ANE: GPU kernels write x, w, b (fp16, padded row stride); CPU reads y vs float64 | 0 bad elements, 20 iterations per shape (3 at B=4096) |
| b. ANE -> GPU: GPU kernel reads y (padded) into an fp32 buffer | 0 bad elements |
| d. shapes | B17 I5 O6 (every row padded to 32 elements), B256 I64 O64, B64 I32 O48 (mixed), B4096 I512 O512 |
| c. coherency: 200 iterations of y = a + b, values changed every iteration, every element checked exactly, at [64,40], [17,6] and [512,512] | 0 bad **without any IOSurfaceLock**; 0 bad with lock/unlock around each handoff too |
| f. mutant: inputs written at the DENSE row stride | B17 I5 O6: 306 / 306 elements wrong, caught. At B256 I64 O64 the rows are not padded, so there is nothing to catch. |

The tolerance in (a) and (b) is 2⁻⁹ · max(|ref|, 1), about 2 fp16 ulps. A
stale or misplaced element is off by 0.1 or more.

**Locks: not needed.** metal_rt's `sync()` (command buffer completed) before
the ANE evaluation, and the synchronous `evaluateWithQoS` before the GPU
reads, are enough on the M1 Pro.

**Timing**, median µs per iteration, two runs (the ANE evaluation itself
varies 640–1970 µs at B=4096 between runs):

| Shape | Zero-copy, 2 GPU syncs (write+sync / ANE / read+sync) | Zero-copy, 1 GPU sync per iteration | CPU path (generate / write / ANE / read) |
|---|---|---|---|
| B17 I5 O6 | 590 (183 / 243 / 162) | 429 | 248 (0 / 5 / 239 / 2) |
| B256 I64 O64 | 575–605 (181 / 245 / 160) | 430–451 | 286–300 (24 / 15 / 246 / 9) |
| B4096 I512 O512 | 1255–1628 (~360 / 640–930 / ~270) | 1451–1924 | 4845–5833 (2290 / 840 / 990–1970 / 710) |

- **Large tensors: zero copy wins, about 2×.** (Corrected: an earlier
  version said 3–3.5×, but that counted the CPU path's host value-generator
  loop, ~2.3 ms, which is test scaffolding, not I/O.) At B=4096:
  - the I/O alone is about 2.5× faster: the CPU's fp16 write + read
    (≈ 1.55 ms) against the GPU's write + read with both syncs (≈ 0.63 ms);
  - a whole iteration without the generator is about 2× faster: ≈ 2.5 ms on
    the CPU path against 1.25–1.6 ms zero-copy.
- **Small tensors: zero copy loses.** One GPU round trip (commit, run, sync)
  costs ~160–180 µs. The CPU writes a few hundred fp16 values in 5 µs.
- **The training step needs one sync.** The planned step (ANE eval -> one
  GPU kernel that reads the gradients, runs Adam on fp32 masters and writes
  fp16 weights -> ANE eval) needs ONE GPU sync per step, the "1 GPU sync"
  column. The GPU round trip is then the fixed cost to beat.

**Surprises:**

- **64-bit index math was slow.** The first write kernel used `Int` (64-bit)
  indices and divided a 1-D thread index into (row, col). It took 2–5 ms at
  B=4096, on an ordinary metal_rt buffer as much as on a surface: the GPU
  emulates 64-bit division. With 32-bit math and a 2-D grid (columns in x,
  rows in y) the same writes take ~0.36 ms with the sync. The surface
  memory itself is as fast as a normal buffer: a trivial fill of 4 MB runs
  in ~250 µs, sync included, on both.
- **The ANE evaluation is noisy.** At B=4096 it ran in 640–930 µs after the
  GPU path and 990–1970 µs after the CPU path. These are single runs, not a
  controlled comparison.
- **No locks needed** (see above).

## The training step with Adam on the GPU (`gpu_adam.mojo`)

`AneAdamGpu` keeps the fp32 master weights and Adam's m and v in ordinary
metal_rt buffers. A step:

1. the CPU writes the seed if the scale changed;
2. the ANE evaluates;
3. ONE command buffer runs on the GPU:
   - per gradient, a check kernel reads the fp16 gradient straight from its
     output surface (wrapped, padded rows) and raises a flag on Inf/NaN;
   - per parameter, an Adam kernel returns at once if the flag is up.
     Otherwise it applies nn Adam's CPU expression: the gradient
     × 1/(scale · grad_div), then `p - lr * (m / bc1) / (sqrt(v / bc2) + eps)`
     with bc = 1 − βᵗ from fp32 running products. It updates w, m and v and
     writes fp16(w) straight into the input surface;
4. one sync;
5. the CPU reads the flag and the loss.

A non-finite step changes nothing: not the masters, not m or v, not the step
count. The data inputs are still written by the CPU (`set_data`).

**Gates** (`tests/experimental/ane/test_ane_gpu_adam.mojo`, all pass). The
network is the sine fit 1-64tanh-64tanh-1 at B=256; its w3 [64, 1] has
padded rows.

| Gate | Result |
|---|---|
| 1. five steps vs `AneGrad.run` + nn's Adam on the host | Losses identical every step. fp32 masters within 0 / 2 / 5 / 6.5 / 6.75 ulps (of max(\|p\|, lr)) after steps 1–5; bound 8. fp16 weights in the surfaces bit-equal to fp16(downloaded masters): 0 mismatches. |
| 2. skip (seed 65504: the gradients overflow) | Both arms return NaN. Masters, m, v bit-identical (13059 / 13059). Step count 5 → 5. The next good step matches the host arm (6.75 ulps, identical loss). |
| 3. sine-fit training, 2000 steps, dynamic scaler | Ensembles of 12 per arm (inits 0..11 fp32 ulps apart). Median of the last-100-step medians: host 5.2e-5, GPU 6.9e-5, ratio 1.32; bound: within a factor 2. The CPU tape's mean is 1.7e-4. |
| 4a. mutant: weights written at the dense row stride | 63 surface elements wrong per step, masters off by ~2.6e7 ulps: FAILS gate 1 (caught) |
| 4b. mutant: no bias correction | masters off by ~2.8e7 ulps: FAILS gate 1 (caught) |

Why the masters differ at all:
- the two arms see the same fp16 weights, so the same gradients, bit for
  bit; only Adam's fp32 arithmetic differs. The GPU's division and sqrt are
  not correctly rounded, the GPU flushes fp32 denormals (measured: 1e-20²
  gives 0 on the GPU), and fma contraction may differ;
- each update (size ≲ lr) differs by a few ulps, and m and v carry the
  difference into later steps.

Why ensembles in gate 3:
- the final loss is chaotic in the init's last bit. From inits 1..11 ulps
  apart, host Adam alone ends anywhere from 2.1e-5 to 2.5e-4;
- with 5 runs per arm, the two medians differed 3.4× by chance; with 12,
  1.3×.

**Bench** (`bench_ane_step.mojo`, median µs/step [min-max], loss read every
step):

| Workload | CPU tape | ANE + host Adam | **ANE + GPU Adam** | metal_rt tape | host-Adam arm: write / eval / read / Adam | GPU-Adam arm: eval / GPU cb incl. sync |
|---|---|---|---|---|---|---|
| PPO B=64 8-64-64-4 tanh | 71 | 400 | 688 | 624 | 15 / 509 / 27 / 10 | 380 / 308 |
| SAC B=256 23-256-256-1 relu | 740 | 794 | 1274 | 1010 | 56 / 547 / 130 / 63 | 797 / 468 |
| BIG B=4096 64-512-512-64 relu | 19417 | 5554 | 5617 | 7598 | 244 / 4871 / 543 / 275 | 5326 / 993 |
| WIDE B=2048 512-2048-2048-512 relu | — | 34884 | **19669** | 38744 | 3310 / 18293 / 10488 / 9465 | 17797 / 2410 |

- **PPO and SAC: it loses** (1.7× and 1.6× slower than host Adam). The GPU
  command buffer costs 300–470 µs: the round trip, plus about 12 serial
  dispatches. The host did the whole I/O and Adam in under 0.25 ms.
- **BIG: a tie** (5617 vs 5554). The GPU part costs 0.5–1.3 ms, depending
  on the GPU's power state. Back to back it is 473 µs; after 5 ms idle
  (the ANE's evaluation) it is ~650 µs. The host's write + read + Adam
  cost 1.06 ms.
  - Per launch on [512, 512], the Adam kernel takes ~106 µs and the check
    ~16 µs. The 12 dispatches total ~250 µs plus the sync.
- **WIDE: 1.77× faster.** The step drops from 34.9 to 19.7 ms: the GPU part
  takes 2.4 ms in place of the host's 23 ms of fp16 conversion and scalar
  Adam. It is now 2× faster than the metal_rt tape.
- **Not done.** Fusing the per-parameter checks (or all the kernels) into
  one dispatch would cut the serial dispatches; Metal's 31-buffer argument
  table allows it at these sizes.

**Surprises:**

- **A NaN in the DATA does not produce a non-finite gradient.** The ANE's
  tanh returns a finite value for a NaN input, so the skip gate forces an
  overflow with the largest fp16 seed (65504) instead.
- **The GPU flushes fp32 denormals.** fp16 denormals survive both casts on
  the GPU.
