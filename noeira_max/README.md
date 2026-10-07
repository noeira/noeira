# noeira_max — MAX as a backend for noeira

Prototypes that evaluate Modular's MAX (26.6.0, with Mojo 1.1.0) as a backend for
noeira: running MAX models from Mojo, building MAX graphs from Mojo, and training on
MAX. Each prototype is tested and measured; none of them is used by the `noeira`
package.

| Path | What |
|---|---|
| `autodiff/` | Training on MAX: reverse-mode autodiff as a graph transform, the whole train step (forward, backward, AdamW) as one compiled graph, and kernel-backed rules through Mojo custom ops. See `autodiff/README.md`. |
| `capi_mojo/maxrt/` | `maxrt`, a Mojo binding to the MAX C API: load a MEF, lend host or device buffers, execute, capture and replay, on CPU, Metal and CUDA, with no Python in the process. See `capi_mojo/maxrt/README.md`. |
| `capi_mojo/maxrt_tests/` | `maxrt`'s tests, inside pixi and outside it; on NVIDIA, the Mojo ↔ MAX device round trip. |
| `capi_mojo/bench/` | MLP inference through the C API, noeira's nn with CUDA-graph capture, and Mojo kernels on MAX's own stream. |
| `capi_mojo/examples/train_from_mef.mojo` | A train step exported by `autodiff/`, trained from Mojo on `maxrt` (CPU, or GPU with capture). |
| `capi_mojo/run_mef.mojo`, `capi_mojo/build_mef.py` | The smallest Mojo program on the C API: raw `external_call`s. See `capi_mojo/README.md`. |
| `graph_mojo/` | A Mojo graph builder generated from MAX's op stubs (121 ops), with parity tests against `max.graph`. |
| `train_from_mojo/` | Train steps defined in Mojo (model, loss, AdamW; noeira's kernel pairs as custom ops), differentiated by `autodiff/`'s transform at setup, trained on MAX through `maxrt`. Gated against the same steps built in Python. See `train_from_mojo/README.md`. |
| `staged_vs_eager/` | One MLP of noeira layers, run eagerly on noeira's kernels or staged as a MAX graph on the same memory, and the latency crossover between the two. |
| `mlp_inference.py`, `benchmark_interop.mojo` | MAX from Mojo through Python interop: `MLPInference`, a configurable MLP compiled once on MAX, and the benchmark that splits its cost into compute, transfers and Python glue. |
| `benchmark_nn_baseline.mojo` | noeira's nn on the same MLPs: the baseline. |
| `probe_c_api.sh` | Checks that the C API links and that a MEF can be exported. On macOS it reports a false NO-GO (see the notes). |
| `graph_mlp_example.py`, `graph_relu_example.py` | MAX reference snippets. |

## Results

MAX 26.6.0 and Mojo 1.1.0 throughout. GPU numbers are from an RTX 5090 (driver 580.173.02).

### Running MAX models from Mojo

An RL actor MLP (17 → 256 → 256 → 6) and a wider one (256 → 512 → 512 → 64), in µs per
call on an RTX 5090:

| Shape | (a) MAX compute only, timed from Python | (b) Through Python, from Mojo | (c) C API, host buffers | (d) C API, device buffers | (e) C API, device buffers, captured | (f) noeira nn | (g) noeira nn, captured* |
|---|---|---|---|---|---|---|---|
| actor, batch 1 | 26.8 | 48.5 | 36.5 | 24.1 | **8.2** | 21.5 | 12.7 |
| actor, batch 64 | 52.1 | 86.2 | 74.8 | 69.1 | **48.4** | 53.2 | 49.4 |
| actor, batch 1024 | 53.0 | 86.9 | 85.2 | 69.4 | **49.9** | 55.3 | 53.2 |
| wide, batch 1 | 19.0 | 36.4 | 35.6 | 24.0 | **9.1** | 20.5 | 13.0 |
| wide, batch 1024 | 64.4 | 144.2 | 131.4 | 79.6 | **65.3** | 91.2 | 82.7 |

What each column runs:
- **(a)** MAX's device compute alone, calls back to back, timed from Python: a floor, not
  the latency a caller sees.
- **(b)** The whole call from Mojo through Python interop: the input copied to the device,
  the model executed, the output copied back (`benchmark_interop.mojo`).
- **(c)** The same through the C API, with no Python in the process (`capi_mojo/bench/`).
- **(d)** The input in a Mojo device buffer, lent to MAX once by address: no copies.
- **(e)** (d) recorded once as a CUDA graph, and that graph replayed on every call.
- **(f)** noeira's nn forward on the same MLPs (`benchmark_nn_baseline.mojo`); **(g)** that
  forward recorded once as a CUDA graph and replayed (`capi_mojo/bench/bench_nn_capture.mojo`).

(a) and (b) are means over 2,000 calls; (c) to (g) are medians over 1,000 or 2,000 calls,
each synchronised. (a) to (f) come from one session. *(g) comes from a later session on the same GPU model: there, (e) reproduced
within 1% (8.3 µs at actor batch 1) and (f), on a newer noeira, measured 18.4 µs.

- **Each layer of overhead can be removed.** At actor batch 1, the C API removes Python's
  glue (48.5 → 36.5 µs), lending device buffers removes the copies (24.1), and capture
  removes MAX's per-call host cost (8.2; 4.1 pipelined).
- **With device buffers and capture, MAX runs these MLPs faster than noeira's nn, captured
  or not**: 8.2 against 12.7 µs at batch 1. Where both are GPU-bound (actor, batch 64 and
  1024), they are within a few µs.
- **Mojo and MAX share the CUDA context and its allocator**, so device buffers pass both
  ways by address, but not a stream: each hand-off is a host synchronisation, about 3 µs.
  With Mojo's kernels on MAX's stream (`capi_mojo/bench/run.sh --stream`), a "Mojo kernel →
  MAX → Mojo kernel" iteration drops from 18.3 to 12.3 µs, or 8.2 µs pipelined, with exact
  results; on two streams without the synchronisations, almost every result is wrong. The
  C API does not expose its stream, so that benchmark finds it through noeira's CUDA
  interposer.
- MAX compiles each MLP in about 30 s on CUDA, about 1 s once cached.
- (c) and (d) vary by up to 8 µs between processes (MAX's host cost per call; 28.6 and
  15.9 µs at actor batch 1 in the later session); replays do not.

The first measurements, in June 2026, went through Python only. MAX's delivered latency
was then 2.7–4× nn's on NVIDIA. That gap was Python's glue and the host copies (the
Mojo ↔ Python crossing itself costs about 0.15 µs), which the C API path above removes.

### Building MAX graphs from Mojo

- `graph_mojo/gen/` reads MAX's op stubs (`max/_core/dialects/rmo/__init__.pyi`, 123 op
  classes) and writes a typed Mojo builder for 121 of them, in under 0.1 s. Graphs built
  with it are bit-identical to `max.graph`'s on 7 cases, including ops that no hand-written
  code names.
- The stubs are not a full op schema: 93 ops need the caller to supply the result type,
  integer attributes do not say their width, and MLIR op names are missing.
- Python's `max.graph` and the C API share one `libmax` in a Mojo process. Process start to
  first inference takes 1.65 s with a warm compile cache.

### Eager or staged, from one definition

`staged_vs_eager/` runs the same MLP eagerly (noeira's layers) or as a MAX graph that
borrows the eager layers' weight memory, compiled once per width (the batch is symbolic).
The crossover follows MAX's executor cost per call:

| | MAX executor, per call | Eager wins | Staged wins (staged / eager) |
|---|---|---|---|
| Apple M1 CPU | 100–300 µs | below ~2 ms of work per call (by up to 48×) | above it (0.61–0.83) |
| x86 CPU (EPYC 9254) | ~24 µs | below 50–100 µs of work per call | above it (0.63–0.88) |
| RTX 5090 | | at batch 1 (by up to 5.5×), and at width 256 | widths 1024 and 4096 from batch 4 (0.83–0.94) |

- On the CPU, the two paths' outputs are bit-identical. On CUDA they differ by TF32
  rounding from batch 64: MAX's multistage GEMM runs float32 in TF32 there.
- On Metal, lending a Mojo device buffer to MAX crashes inside `M_borrowTensorInto`. One
  x86 run in three hung after a compile, every thread waiting; the cause is not isolated.

### Training on MAX

`autodiff/` (see its README):
- `value_and_grad` is a graph transform: it walks the ops a function emitted and emits their
  VJPs into the same graph (42 rules). In float64, training an MLP and a small GPT matches
  PyTorch to 3e-9 or better over 30–50 steps.
- The whole train step (forward, backward, and AdamW with its schedule and clipping on the
  device) is one MAX graph whose parameters and optimizer state are updated in place. It
  exports to a MEF and trains from Mojo with the same losses, bit for bit, on the CPU and on
  CUDA with capture.
- On the RTX 5090, the char-GPT recipe of `tools/nn/torch_nn_reference.py` (6 layers × 384,
  batch 64, sequence 256) takes 52.4 ms per step, and 27.5 ms with noeira's fused attention
  and LayerNorm as Mojo custom-op pairs. `torch.compile` takes 20.4 ms and noeira's nn
  20.3 ms on the same recipe.
- Compile time is quadratic in depth: 122 s at 1 layer, 354 s at 6, 1015 s at 12; 1.2 s
  once cached.
- MAX 26.6 behaviours found along the way are pinned as expected failures in
  `autodiff/tests/test_max_findings.py`.

### Training from Mojo

`train_from_mojo/` joins the two halves above. A Mojo program builds its whole train step with the generated builder:
- the parameter and moment buffers;
- the forward pass, with noeira's LayerNorm and attention kernel pairs as custom ops;
- the loss;
- AdamW's in-place update.

One call hands the ops it emitted to `autodiff/`'s transform, which adds the backward pass. MAX compiles the step, and `maxrt` runs every step with no Python.

- On the M1's CPU, each step tested prints the same MLIR as the step the Python prototype builds, so MAX's compile cache serves one from the other. It trains bit for bit like it, every loss and every final buffer:
  - three RL-sized MLPs;
  - an MLP with the LayerNorm pair;
  - the 2-layer GPT with both kernel pairs.
- The step can train noeira nn layers' own weights: their `Param` memory is lent to MAX as the parameter buffers, and nn's forward then runs on the trained weights.

## How to run

From the repo root. The scripts use the main checkout's pixi env (`default`; `MAXRT_ENV`
or `AUTODIFF_ENV` select another), so they also work from a git worktree.

```bash
noeira_max/capi_mojo/bench/run.sh [--all | --stream]   # NVIDIA: (c)-(e); --all adds (a), (b), (f); --stream the shared stream
noeira_max/capi_mojo/maxrt_tests/run.sh                 # maxrt's tests (MAXRT_ENV=apple: Metal too)
noeira_max/graph_mojo/run.sh                            # regenerate the builder, parity tests, example
noeira_max/staged_vs_eager/run.sh [--gpu]               # the crossover sweep, CPU or CUDA
noeira_max/train_from_mojo/run.sh [--nn] [--gpu]         # train steps built in Mojo, against Python's
noeira_max/autodiff/run.sh -m unittest discover -s noeira_max/autodiff/tests -t .

# Through Python interop: build to a binary, and run it inside pixi
pixi run -e apple  bash -c 'mojo build -I . noeira_max/benchmark_interop.mojo -o /tmp/bench && /tmp/bench'
pixi run -e nvidia bash -c 'mojo build -I . noeira_max/benchmark_interop.mojo -o /tmp/bench && /tmp/bench'
pixi run -e nvidia mojo run -I . noeira_max/benchmark_nn_baseline.mojo
pixi run -e nvidia mojo run -I . noeira_max/capi_mojo/bench/bench_nn_capture.mojo   # (f) and (g)
```

## Notes

- **Build, don't `mojo run`.** A JIT `mojo run` clashes with MAX's Python engine
  (`Init::getOrCreateContext() requested an M::Context with different Init::Options`), and
  it cannot resolve `libmax`'s symbols. `mojo build ... -Xlinker -lmax` works.
- **Run built binaries inside `pixi run`.** The embedded Python needs `MOJO_PYTHON_LIBRARY`,
  and the C API finds its runtime through `MODULAR_HOME`; outside pixi, `maxrt` writes its
  own `modular.cfg`.
- **Mojo frees a value at its last use.** Memory lent to MAX by address must stay alive
  until MAX has read it, and a pointer into a MAX output must not outlive its owner (`maxrt`
  ties each pointer to its owner's origin).
- **`M_copyTensorToDevice` returns before the copy is done.** `maxrt` synchronises after it.
- **noeira's CUDA interposer.** On Linux, `pixi run` preloads it; the scripts run MAX without
  it (`env -u LD_PRELOAD`), except the shared-stream benchmark, which needs it.
- **macOS.** `libmax.dylib` exports the C API too: its 70 `M_*` functions are in the Mach-O
  export trie, which `nm` and `grep` miss, hence `probe_c_api.sh`'s false NO-GO.
