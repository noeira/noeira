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
| simdgroup GEMM | `sg_gemm.mojo` | Apple's 8 x 8 simdgroup matrix units through the AIR functions; the tape's own GEMM on Metal |
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
- **Matmul.** It is the tape's own GEMM (`ctx.own`): Apple's simdgroup
  matrix hardware, through `sg_gemm.mojo` (see "A GEMM on the simdgroup
  matrix hardware" below), for outputs of at least 64 x 64 and k > 1;
  otherwise `_naive_mm_kernel`. MAX's matmul launches through MAX and cannot
  run here. (The first version used a scalar `_tiled_mm_kernel`, which is
  kept.)
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

This table is the FIRST version, with the scalar tiled GEMM. The simdgroup
GEMM below takes B=4096 from 16.4 ms to 6.9 ms (1.44x faster than the tape
on MAX), SAC from 782 to 622 us, and PPO from 245 to 234 us.

- **Small steps.** The metal_rt tape runs PPO-sized steps 4–4.5x faster than
  the tape on MAX, and 6.6x faster than the comptime nn on MAX.
- **B=4096 was GEMM-bound and slower** (0.6x). The scalar GEMM ran at
  0.73 TFLOP/s on [4096, 512] @ [512, 512], against 2.5 for MAX's matmul.
  The own-kernel MAX arm, 17.8 ms, showed it was the kernel and not the
  launcher. 8 x 8 outputs per thread and 32-deep k-slabs were both slower.
  The simdgroup GEMM (next section) closed it.
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

## A GEMM on the simdgroup matrix hardware (2026-10-25)

`sg_gemm.mojo` is a from-scratch GEMM on Apple's 8 x 8 simdgroup matrix
units. The tape uses it for every Metal product with an output of at least
64 x 64 and k > 1, on a metal_rt tape and on `own_kernels=True`.

**The route to the hardware.** MSL's `simdgroup_load`,
`simdgroup_multiply_accumulate`, `simdgroup_store` and
`make_filled_simdgroup_matrix` compile to four AIR functions. I compiled a
15-line MSL kernel of my own with `xcrun metal -S -emit-llvm` and read their
names and signatures off the IR:

```
<64 x float> air.simdgroup_matrix_8x8_load.v64f32.p{1,3}f32(ptr, <2 x i64> (8, 8), <2 x i64> strides, <2 x i64> origin)
<64 x float> air.simdgroup_matrix_8x8_multiply_accumulate.v64f32.v64f32.v64f32.v64f32(a, b, c)   // a·b + c
void         air.simdgroup_matrix_8x8_store.v64f32.p{1,3}f32(m, ptr, …)
<64 x float> air.simdgroup_matrix_8x8_init_filled.v64f32.f32(float)
```

- **Strides.** They are `(1, ld)` for a row-major load and `(ld, 1)` for the
  transposed one.
- **Calling them.** `llvm_intrinsic` rejects these names ("could not find
  LLVM intrinsic") because they are external functions that the Metal
  backend lowers, not LLVM intrinsics. `external_call` reaches them.
- **Proof.** `metal_rt/probe_simdgroup.mojo` runs an 8x8x8 product, plain
  and with B loaded transposed: exact against a CPU reference.

**The kernels**, f32 in and out, 128 threads (4 simdgroups in 2 x 2) per
threadgroup:

- **Direct kernels** run when M and N are multiples of the tile and K a
  multiple of 8. Fragments load straight from device memory; the tape's
  transposes are the hardware's transposed load. There is no staging and no
  barrier. Each simdgroup holds RA x CB accumulators, one variant per form:
  - 4 x 8 (64 x 128 tile) for plain products from 32 tiles up;
  - 2 x 8 (32 x 128) for aᵀ·b;
  - 4 x 4 (64 x 64) for a·bᵀ and small outputs.
- **The staged kernel** handles everything else (edge tiles, any K). Its
  64 x 64 tile is staged 16-deep through threadgroup memory: in canonical
  form, zero-filled past M, N and K, with the transposes absorbed by the
  staging loads. Interior tiles store their fragments directly. Edge tiles
  go through threadgroup memory and a bounds-checked copy. With ACC,
  interior accumulators start from `o`.
- **Determinism.** Every output sums k in hardware order, so results are the
  same run to run and launcher to launcher.

**Gates:**

- **(a) `tests/experimental/functional/test_sg_gemm.mojo` vs a float64 CPU
  reference.** 16 shapes × 8 (TA, TB, ACC) forms × 2 paths: the tape's
  dispatcher and the staged kernel forced. That is 256 cases. Shapes cover
  tile multiples, non-multiples of 64 and of 8 in M, N and K, dimensions of 1
  everywhere, and the MLP shapes.
  - Bound per element: (K + 2) u (Σ|a b| + |o0|), where u = 2⁻²⁴, the fp32
    summation bound.
  - **All 256 pass.** The worst error is 0.50 of the bound, at K = 1, where
    the bound is a single rounding. At K ≥ 64 the worst is 0.06 of the
    bound.
  - The worst relative error on outputs with |ref| > 1e-3 is 3.9e-3, and
    that is cancellation.
- **(b) `test_functional_metal_rt.mojo`: 23 of 23,** three runs.
  Bit-identity against `Tape(ctx, own_kernels=True)` still holds, since both
  run the same kernels.
- **(c) Tolerance vs the normal MAX path,** unchanged and well inside its
  bounds: forward 0, gradients ≤ 4e-8, losses ≤ 8e-8, parameters ≤ 6e-5 of
  the run's lr. `test_functional_gpu` and `_fused` still pass.

**TFLOP/s** (`benchmarks/functional/bench_tape_gemm.mojo`, best of 5, 20
launches per sync). MAX's dW runs as a transpose plus matmul, which is how
the tape runs it on MAX, because MAX's matmul has no transpose_a.

| product | M x N x K | naive | old tiled | simdgroup (tape) | MAX matmul | vs MAX |
|---|---|---|---|---|---|---|
| L1 fwd x@w | 4096 x 512 x 64 | 0.08 | 0.67 | 2.46 | 1.75 | 1.40 |
| L2 fwd x@w | 4096 x 512 x 512 | 0.10 | 0.74 | 3.03 | 2.54 | 1.19 |
| L3 fwd x@w | 4096 x 64 x 512 | 0.09 | 0.59 | 2.10 | 1.23 | 1.70 |
| L3 dX = dO@wᵀ | 4096 x 512 x 64 | 0.02 | 0.67 | 1.86 | 1.38 | 1.34 |
| L2 dX = dO@wᵀ | 4096 x 512 x 512 | 0.02 | 0.73 | 2.18 | 2.26 | 0.96 |
| L3 dW = xᵀ@dO | 512 x 64 x 4096 | 0.07 | 0.18 | 0.56 | 0.27 | 2.07 |
| L2 dW = xᵀ@dO | 512 x 512 x 4096 | 0.08 | 0.60 | 2.46 | 1.01 | 2.42 |
| L1 dW = xᵀ@dO | 64 x 512 x 4096 | 0.07 | 0.18 | 0.61 | 0.67 | 0.91 |
| SAC fwd | 256 x 256 x 256 | 0.09 | 0.33 | 0.81 | 0.66 | 1.22 |
| SAC L1 fwd (K = 23: staged) | 256 x 256 x 23 | 0.05 | 0.13 | 0.12 | 0.05 | 2.47 |
| PPO fwd | 64 x 64 x 64 | 0.02 | 0.02 | 0.03 | 0.02 | 2.15 |

**The tape** (`bench_functional_metal_rt.mojo`, each arm in its own process,
median of 7, us/step; metal_rt = best of the three policies):

| workload | loop | comptime nn (MAX) | tape MAX | tape metal_rt, scalar GEMM | tape metal_rt, simdgroup GEMM | vs tape MAX |
|---|---|---|---|---|---|---|
| B=4096 64-512-512-64 train | end | 9006 | 9844 | 16622 | 6860 | 1.44x |
| B=4096 train | read/20 | 9028 | 9920 | 16386 | 6896 | 1.44x |
| B=4096 train | read/1 | 9812 | 10295 | 16850 | 7230 | 1.42x |
| SAC B=256 train | read/20 | 1305 | 1110 | 782 | 622 | 1.79x |
| SAC B=256 train + Adam | read/20 | 1642 | 1096 | 830 | 636 | 1.72x |
| PPO B=64 train | read/20 | 1685 | 1122 | 245 | 234 | 4.79x |
| PPO B=64 train + Adam | read/20 | 1860 | 1160 | 267 | 255 | 4.55x |
| ACT B=1 fwd | end | 225 | 240 | 162 | 162 | 1.48x |

- **ACT never reaches the GEMM.** Its products have M = 1 and stay naive.
  Its read-every-step row (442 now, 324 in the first table) is noise: the
  pre-simdgroup binary measures 407–467 on the same run.
- **Thresholds.** Simdgroup kernels run from a 64 x 64 output up (k > 1);
  below that the naive kernel runs. At 64 x 64 x 64 the simdgroup kernel
  already beats naive (0.03 vs 0.02 TFLOP/s), and the PPO step went from
  243 to 227 us when the threshold dropped from 8 tiles to 1. Plain products
  take the 64 x 128 tile only from 32 tiles up: on SAC's 256 x 256 output
  it leaves 8 threadgroups and lost to 4 x 4 (0.57 vs 0.81).

**What did not help:**

- **Accumulators in an `Array[SIMD[f32, 64], 16]`:** 0.04 TFLOP/s, 20x
  slower than scalar. In memory each `<64 x float>` is materialised whole
  per thread. They must be named SSA locals, and the kernels are written out
  that way.
- **Staging every tile through threadgroup memory:** 0.7 TFLOP/s, no faster
  than the scalar kernel. Staging and barriers dominated. Loading straight
  from device memory gave 2.0, and a lean kernel with no staging code at all
  gave 2.7–3.0.
- **A 2x unroll of the k loop:** +2%, not kept.
- **An 8 x 4 tile (128 x 64):** slower than 4 x 8 on every shape.
- **half / bf16 inputs:** not tried; f32 only.

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
pixi run -e apple   mojo run -I . tests/experimental/functional/test_sg_gemm.mojo                  # simdgroup GEMM vs float64
```
