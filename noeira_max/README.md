# noeira_max — MAX-as-inference-backend prototype

A small prototype probing **MAX as an *inference* backend for noeira**, driven from Mojo
via Python interop. *Training* on MAX is prototyped separately, in `autodiff/`. v1
scope is **MLP inference only**.

It answers three questions for "should noeira incorporate MAX?":

1. How fast is MAX device compute on a realistic RL MLP?
2. What is the host↔device data-transfer cost (H2D / D2H)?
3. What does the Mojo↔Python interop bridge actually cost on the real call path?

## Layout

| File | What |
|---|---|
| `mlp_inference.py` | `MLPInference` — configurable MLP (dims/batch/device all variables) built+compiled once on MAX, plus timing primitives. Inference only; weights are random. |
| `benchmark_interop.mojo` | Mojo driver that imports the package via Python interop and times the MAX decomposition across a batch/shape sweep. |
| `benchmark_nn_baseline.mojo` | Pure-nn native GPU forward on the SAME shapes — the apples-to-apples "why incorporate MAX?" baseline. |
| `probe_c_api.sh` | Path-B feasibility probe: is the MAX C API linkable + is there a MEF-export path? Prints GO/NO-GO. Run under `-e nvidia`; on macOS it reports a false NO-GO (see Path B). |
| `capi_mojo/maxrt/` | Path B: a Mojo binding to the MAX C API (load a MEF, lend host or device buffers, execute, capture, replay). Tests in `capi_mojo/maxrt_tests/`, a training example in `capi_mojo/examples/`. |
| `capi_mojo/bench/` | Columns (c)–(e) of the CUDA table: `run.sh` exports the MLP MEFs, builds and runs `bench_capi.mojo`; `run.sh --all` reruns every column on the same box. |
| `graph_mlp_example.py`, `graph_relu_example.py` | Original MAX reference snippets (kept for reference). |

`MLPInference(input_dim, hidden, output_dim, batch, device="gpu", seed=0)` — `hidden` is a
list `[256, 256]` or a Mojo-friendly string `"256,256"`. `device="gpu"` → Metal on Apple,
CUDA on NVIDIA (same Python, different backend); falls back to CPU if no accelerator.

## How to run

**Build to a binary — do NOT `mojo run`** (JIT triggers an `M::Context` clash with MAX's
Python engine; a compiled binary has no JIT context). **And run the binary *inside* the
activated env** — the embedded Python needs `MOJO_PYTHON_LIBRARY` set, which `pixi run` only
provides for the command it wraps. Build + run in one pixi invocation:

```bash
# Apple (Metal)
pixi run -e apple  bash -c 'mojo build -I . noeira_max/benchmark_interop.mojo -o /tmp/bench && /tmp/bench'
# NVIDIA (CUDA)
pixi run -e nvidia bash -c 'mojo build -I . noeira_max/benchmark_interop.mojo -o /tmp/bench && /tmp/bench'
```
(Running the bare `/tmp/bench` outside `pixi run` fails with "No module named 'max'", because
activation env vars aren't set in your shell.)

nn baseline (pure nn, no Python — plain `mojo run` is fine):
```bash
pixi run -e apple  mojo run -I . noeira_max/benchmark_nn_baseline.mojo
pixi run -e nvidia mojo run -I . noeira_max/benchmark_nn_baseline.mojo
```

You can also drive the Python package directly:
```bash
pixi run -e apple python -c "from noeira_max import MLPInference; m=MLPInference(17,'256,256',6,64); print(m.info())"
```

## Findings so far (Apple M-series / Metal, 2026-06-03)

### ⚠️ Footgun: `mojo run` + MAX Python engine clash at the runtime-context level
JIT `mojo run` creates an `M::Context` whose `Init::Options` conflict with the one
`max.engine` wants → `LLVM ERROR: Init::getOrCreateContext() requested an M::Context with
different Init::Options`. **A compiled binary has no JIT context, so the Python engine
initializes cleanly.** Always `mojo build` then run the executable. (Worth re-checking on
NVIDIA — this is your part.)

### The Mojo↔Python FFI crossing is essentially free
The per-call interop floor (a Mojo loop over a Python `noop()`) is **~0.15 µs/call**. The
"Python in the hot loop" tax people worry about is *not* the FFI boundary. Mojo-side
end-to-end `infer()` ≈ Python-side end-to-end `full()` to within run-to-run noise.

### The real costs are MAX compute, data transfer, and Python *glue*
For each call the decomposition is: `MAX device compute` + `H2D` + `D2H` + `Python glue`
(numpy contiguity checks, `Buffer` object creation, attribute lookups). On Metal the
**Python glue per call (~130–170 µs)** dwarfs the FFI crossing (0.15 µs) — i.e. *what you
do in Python per call matters far more than crossing the Mojo/Python line.* This is the
argument for the production path (B): Mojo → MAX **C API** on a precompiled MEF, which
removes the Python glue entirely.

### Numbers are path (A), a pessimistic upper bound
This prototype is **path (A): Mojo → CPython → MAX**, Python in the hot loop. The
production-realistic **path (B): Mojo → MAX C API** (`M_executeModelSync` on a precompiled
MEF) removes Python from the loop and is strictly faster. Read (A) as an upper bound: *if
MAX wins even here, path (B) wins by more.*

### Metal compile time is high (~15 s even for a tiny MLP)
One-time per graph shape, excluded from per-call numbers, but relevant for research
iteration that sweeps many shapes. NVIDIA compile times are the ones that matter for you.

### nn vs MAX head-to-head (Apple/Metal, µs per call)

| Shape | nn forward (compute) | MAX device compute | MAX end-to-end from Mojo |
|---|---|---|---|
| actor-b1    (17→256→256→6, b=1)     | 658  | **284**   | 507   |
| actor-b64   (b=64)                  | 618  | **309**   | 978   |
| actor-b1024 (b=1024)                | **1357** | 1663  | 4484  |
| wide-b1     (256→512→512→64, b=1)   | 699  | **287**   | 868   |
| wide-b1024  (b=1024)                | **8913** | 12455 | 15050 |

### nn vs MAX head-to-head (NVIDIA / CUDA, µs per call) — the decisive run

| Shape | nn forward (delivered) | MAX raw compute | MAX delivered (e2e) | nn vs MAX-delivered |
|---|---|---|---|---|
| actor-b1    | **26.6** | 45.3 | 73.8  | nn 2.8× |
| actor-b64   | **36.9** | 71.1 | 99.3  | nn 2.7× |
| actor-b1024 | **38.9** | 55.5 | 104.5 | nn 2.7× |
| wide-b1     | **18.7** | 30.8 | 75.2  | nn 4.0× |
| wide-b1024  | **68.0** | 51.2 | 202.5 | nn 3.0× |

Reading (CUDA — **this is the verdict**):
- **nn wins delivered latency everywhere, ~2.7–4×.** H2D+D2H+Python glue (30–150 µs) dwarfs
  compute at RL-MLP scale.
- **nn wins even raw compute in 4/5 shapes.** MAX's compiler only leads at the widest matmul
  (wide-b1024) — the large/transformer regime it's built for, not small RL MLPs.
- **Interop bridge is free (0.18 µs/call)** on CUDA too — the cost is transfer + Python glue.
- nn here is **unoptimized** (plain `Linear+ReLU`, no fused `LinearReLU`, no CUDA-graph
  capture) — a *ceiling*; the real nn is faster still.
- **MAX compile cost on CUDA is ~46–52 s per shape** (vs ~15 s Metal) — a real RL shape-sweep tax.
- **This bounds path B too:** path B's best case ≈ MAX raw compute (45–71 µs for actor) still
  loses to nn delivered (27–39 µs) except at wide-b1024 (where nn is unoptimized). A perfect
  no-Python path B can't flip the RL-scale verdict.

**Bottom line for "why don't I incorporate MAX?": at RL-MLP inference scale on NVIDIA, nn is
~3× faster delivered and competitive-to-better on raw compute, with no interop tax and no
per-shape compile wall. MAX pays off at large/transformer-scale graphs, not here.**
*(Path A only; path B with device buffers and capture reverses it at these shapes, next
section.)*

### Path B measured: the MAX C API from Mojo (RTX 5090, 2026-10-05)

Every column from one box (RTX 5090, MAX 26.6.0, Mojo 1.1.0), with
`capi_mojo/bench/run.sh --all`. µs per call:

| Shape | (a) MAX compute, from Python, pipelined | (b) MAX through Python from Mojo | (c) C API, host in/out | (d) C API, device buffers | (e) (d) + capture | (e) pipelined | (f) nn | (f) nn pipelined |
|---|---|---|---|---|---|---|---|---|
| actor-b1    | 26.8 | 48.5  | 36.5  | 24.1 | **8.2**  | 4.1  | 21.5 | 18.4 |
| actor-b64   | 52.1 | 86.2  | 74.8  | 69.1 | **48.4** | 44.6 | 53.2 | 49.2 |
| actor-b1024 | 53.0 | 86.9  | 85.2  | 69.4 | **49.9** | 45.1 | 55.3 | 51.2 |
| wide-b1     | 19.0 | 36.4  | 35.6  | 24.0 | **9.1**  | 6.1  | 20.5 | 16.4 |
| wide-b1024  | 64.4 | 144.2 | 131.4 | 79.6 | **65.3** | 61.4 | 91.2 | 86.2 |

- (a), (b): `benchmark_interop.mojo` (path A, as above, rerun on this box).
- (c)–(e): `capi_mojo/bench/bench_capi.mojo`, through `capi_mojo/maxrt/`, no Python in the
  process. (c) lends the host input, copies it to the device (`M_copyTensorToDevice`),
  executes and copies the output back, every call. (d) lends a Mojo `DeviceBuffer` to MAX
  once, by address, and per call synchronises Mojo's context, executes and synchronises
  MAX's device. (e) captures (d) once and replays it. Synchronised columns: median of 1,000
  calls; pipelined: mean of 1,000 back-to-back calls. Every output is checked bit for bit
  against MAX's from Python.
- (f): `benchmark_nn_baseline.mojo`, now with a synchronised per-call median. Still plain
  `Linear+ReLU` without CUDA-graph capture.

Reading:
- **Device buffers work both ways on CUDA.** MAX reads a Mojo `DeviceBuffer` lent by
  address, and Mojo reads MAX's output by address: the two share the CUDA context and its
  allocator. They do not share a stream (nsys: MAX on one, Mojo's `DeviceContext` on
  another), so each hand-off is a host synchronisation; the C API has no stream parameter.
- **Capture removes MAX's per-call host cost.** At batch 1, `M_executeModelSync` costs about
  20 µs of host time whatever the GPU does ((d) pipelined stays near 24 µs); a replay costs
  about 4 µs.
- **With device buffers and capture, MAX beats the nn baseline at every shape** (8.2 against
  21.5 µs at batch 1; 65.3 against 91.2 at wide-b1024). The fair next comparison is nn with
  its own CUDA-graph capture.
- MAX compile on CUDA: 29–31 s per shape cold, 1.1 s cached.

Path B on CPU and Metal: `capi_mojo/maxrt/README.md`.

### (Earlier) Apple/Metal numbers — for reference

Reading (Metal only):
- **MAX raw compute wins at small batch** (~2× faster, 284 vs 658 µs at b=1) but **loses
  end-to-end** once you add H2D+D2H+Python glue — the *delivered* MAX latency to a Mojo
  caller (507–978 µs) is at or above nn's (618–699 µs).
- **nn wins outright at large batch**, on raw compute *and* end-to-end (its native kernels
  beat MAX's here on Metal, and it pays zero transfer/Python tax).
- nn's number is the **full delivered latency** to a Mojo caller (data already in Mojo GPU
  buffers); MAX must overcome its transfer+glue tax to be worth it. On Metal it generally
  isn't; whether MAX's compute edge widens enough on CUDA to flip the end-to-end verdict is
  exactly what the NVIDIA run answers.
- Caveat: nn small-batch numbers include nn's per-call overhead (Sequential mid-buffer
  handling); both columns are Metal and not the target platform. Treat the *shape of the
  story* (compute vs delivered, small vs large batch) as the takeaway, not absolute µs.

## Path B (no-Python hot path): works on Apple and NVIDIA

Path B = Mojo → MAX **C API** on a precompiled MEF, removing Python from the hot loop:
```
M_newStatus → M_newRuntimeConfig → M_newDevice (host, accelerator) → M_runtimeConfigAddDevice →
M_newRuntimeContext → M_newCompileConfig → M_setModelPath(<.mef>) → M_compileModelSync (a load) →
M_initModel →
[hot loop] M_newAsyncTensorMap → M_newTensorSpec → M_borrowTensorInto →
           M_executeModelSync (or M_replayModelSync) → M_getTensorByNameFrom → M_getTensorData
```
`capi_mojo/maxrt/` wraps it as a Mojo binding; the table above measures it on CUDA.

**Correction (2026-10-05) of the June finding "blocked on Apple at the linker level":**
- The C API is **exported on macOS too**. `libmax.dylib` lists all 70 `M_*` functions in
  its export trie (`xcrun dyld_info -exports`). `nm -gU` finds none because the symbol
  table is stripped, which is also why `probe_c_api.sh` reports a false NO-GO on macOS.
- **A MEF can be exported from Python:** `InferenceSession(...).compile(graph).export_mef(path)`
  (MAX 26.6).
- Build with `mojo build ... -Xlinker -lmax`; `mojo run` cannot resolve the symbols.

## Not yet done (next steps)
- nn with its own CUDA-graph capture, against column (e).
- bf16 / fp16.

## Done
- ✅ `MLPInference` MAX package (configurable dims/batch/device) + interop decomposition.
- ✅ nn native baseline on identical shapes (`benchmark_nn_baseline.mojo`).
- ✅ Path B: a Mojo binding to the C API (`capi_mojo/maxrt/`), measured on CUDA with device
  buffers and capture (columns (c)–(e)).
