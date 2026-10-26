# tape_to_mo — the functional tape compiled by MAX's graph compiler, from Mojo

A spike (2026-10-23). One eager forward of `noeira.experimental.functional`
is recorded. `emit_train_step` turns it into a textual MO graph: the
forward, its backward (each op's VJP written as MO ops) and Adam updating
the parameters, moments and bias-correction powers in place. MAX's C API
compiles that text (`M_setModelPath` on a file containing `mo.graph`) and
`maxrt` runs it. Python is involved nowhere: not to build the graph, not to
differentiate it, not to run it. Every kernel is MAX's Mojo kernels, fused
by MAX.

| File | What |
|---|---|
| `mo_text.mojo` | `MoGraph`: SSA values, MO types, the ops the step needs, buffer load/store on one chain, the `mo.graph` header as the Python API prints it |
| `emit.mojo` | `emit_train_step(tape, loss, lr, …, device="" or "gpu:0")`: rank-2 graphs (add/sub/mul/div with broadcasting, matmul, scalar ops, relu/tanh/sigmoid/exp/log/sqrt, sum/mean, transpose, sum_rows, `F.linear` ± activation) |
| `test_tape_to_mo.mojo` | The gates and the timings below |
| `run.sh` | Build + run (`TAPE_MO_ARGS=--gpu` adds the Metal/CUDA arm) |

## Gates (Mac M1 Pro, Mojo 1.2.0rc2, MAX 26.7.0rc2): all pass

These compare 20 Adam steps of the compiled step with the eager tape plus
nn's Adam, from the same weights:

| Gate | Worst loss (relative) | Worst parameter (absolute) |
|---|---|---|
| MLP (relu, tanh, linear, MSE), CPU, a new batch each step | 7.1e-7 | 6.0e-8 (all 5,702) |
| Every other rank-2 op, CPU | 2.4e-7 | 6.0e-8 |
| MLP on Metal, batch 0 each step | 6.3e-7 | 1.3e-7 |
| Mutant: tanh's backward using 1 + y² | caught (1.8e-3) | 0.025 |

## Timings: MAX loses to the eager tape on the Mac

Each figure is the fastest 40-step chunk of 200. The loss is read back
every step in both arms. On Metal both arms run on the GPU with the batch
on the device and the adopted Adam. Another session was compiling during
the run, so the CPU column is loose.

| Batch, layers | CPU: eager / MAX | Metal: eager tape / MAX |
|---|---|---|
| 1, 17→256→256→6 | 60 / 580 µs | 2.04 / 2.09 ms |
| 64, 17→256→256→6 | 160 / 644 µs | 1.65 / 2.49 ms |
| 256, 64→256→256→8 | 465 / 1010 µs | 1.87 / 2.83 ms |
| 1024, 128→512→512→16 | 4.37 / 3.82 ms | 3.58 / 8.61 ms |

**Compile time:**

| Case | Time |
|---|---|
| Cold, CPU | 31–40 s per graph |
| Cold, Metal | 52–72 s per graph |
| From MAX's MEF cache | 7–18 ms |

## RTX 5090: MAX loses to the captured tape at every size

`run.sh` does not cover the box: build with `-e default`, run under `pixi run`
WITH the intercept preloaded, and symlink `noeira/cuda/libcuda_intercept.so`
into the worktree (the tape's capture dlopens it relatively). Then
`<binary> <dir> --nvidia [--captured-only]`.

| µs per step | B1 | B1024 |
|---|---|---|
| captured tape | 43-46 | 98-99 |
| MAX executed | 448-469 | 474-584 |
| MAX captured (`capture=True`) | 106-110 | 312-315 |
| loss read every 20 steps: captured tape / MAX captured | 33 / 84-85 | 92 / 214-226 |

- MAX's CUDA matmul is TF32: the strict fp32 gate fails at 1.4e-4 (loss) /
  5.9e-3 (param), already 4.8e-5 at step 0. Captured MAX equals executed MAX
  bit for bit.
- `capture` records without executing (replay once after it), and `replay`
  returns before the GPU is done (only a read waits).

**Where MAX's GPU time goes** (`--profile tape|max 1|1024` under
`nsys profile --cuda-graph-trace=node`). Kernel counts are equal (MAX 37-46
per step, the tape 36-40) and gaps inside both graphs are < 0.5 µs. MAX
loses on three things, all inside MAX 26.7:

- Each vendor cuBLAS call creates and zeroes a fresh 32 MiB workspace
  (`linalg/matmul/vendor/blas.mojo`): 3 memsets of ~9.8 µs per step,
  replayed by the device graph too.
- On sm_120, fp32 GEMMs take the Ampere-tuned `multistage_gemm` configs:
  4-32 CTAs and no split-K. The K = 1024 weight gradients at B1024 take
  58 µs each, where cuBLASLt takes 8 µs plus a 1.7 µs split-K reduce. In
  all, MAX's matmuls take 200 µs per step against the tape's 66. At B1, one
  matmul runs on a single 16-thread block of `matmul_kernel_naive`: 10 µs.
- Adam is ~18 per-parameter kernels (nn's grouped Adam: 2), and each store
  is a small device-to-device copy (~15 per step).

The tape is TF32 too at B1024 (cuBLASLt `s1688gemm`). Only the gate's B64
shapes are fp32 on the tape.

## Traps (all were harness bugs, not MAX's)

- **Indexing a `Tuple` returns a temporary copy.** `HostBuffer(copy_from=Int(bt[1].unsafe_ptr()))`
  copies from a list that is already freed: `y` reached the GPU as 2.3e-38
  in element 0, and the loss was off by 0.4 % while every layer matched.
  Bind the list to a variable and keep it alive past the copy.
- **Reading back parameters on Metal.** The `Tensor` that `to_device`
  returned still shows its initial bytes after the model has stored into
  that memory. Read through the map instead:
  `inputs.tensor("p0").to_host()`.
- **The MO text is not a supported contract.** The Graph API is MAX's only
  supported IR producer. Pin the version, and keep this gate.
