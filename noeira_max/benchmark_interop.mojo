"""Benchmark MAX MLP inference driven from Mojo via Python interop.

Answers three questions for "should noeira incorporate MAX as an inference backend":
  1. How fast is MAX device compute on a realistic RL MLP?
  2. How much is the host<->device data-transfer cost (H2D / D2H)?
  3. How much does the Mojo<->Python interop bridge cost on the real call path?

Attribution strategy
--------------------
The Python package runs `bench_*(iters)` loops *inside one* Mojo->Python call, so those
numbers are ~interop-free (pure MAX compute / transfer). The Mojo side separately times:
  * `noop()` in a Mojo loop      -> per-crossing interop FLOOR
  * `infer(x)` in a Mojo loop    -> realistic per-call cost a Mojo caller actually pays
The gap between (Mojo infer) and (Python full) is the bridge tax on the real path.

NOTE: this is integration path (A) — Mojo -> CPython -> MAX, Python in the hot loop. The
production-realistic path (B) is Mojo -> MAX C API on a precompiled MEF (no Python in the
loop). So the interop numbers here are a PESSIMISTIC upper bound, not the (B) floor.

IMPORTANT — build, do NOT `mojo run`:
  `mojo run` (JIT) creates an M::Context with compiler options that conflict with the
  one MAX's Python `max.engine` wants ("Init::getOrCreateContext() requested an
  M::Context with different Init::Options"). A *compiled binary* has no JIT context, so
  the Python engine initializes cleanly. Always build then run the executable:

  pixi run -e apple  mojo build -I . noeira_max/benchmark_interop.mojo -o /tmp/bench && /tmp/bench   # Metal
  pixi run -e nvidia mojo build -I . noeira_max/benchmark_interop.mojo -o /tmp/bench && /tmp/bench   # CUDA
"""

from std.python import Python, PythonObject
from std.time import perf_counter_ns


def f2(x: Float64) -> String:
    """Format a Float64 to 2 decimals without f-strings."""
    var neg = x < 0.0
    var v = -x if neg else x
    var scaled = Int(v * 100.0 + 0.5)
    var whole = scaled // 100
    var frac = scaled % 100
    var fs = String(frac)
    if frac < 10:
        fs = "0" + fs
    var s = String(whole) + "." + fs
    return "-" + s if neg else s


def pad(s: String, width: Int) -> String:
    var out = s
    while out.byte_length() < width:
        out = out + " "
    return out


def run_shape(
    mlp_module: PythonObject,
    name: String,
    in_dim: Int,
    hidden: String,
    out_dim: Int,
    batch: Int,
    device: String,
    py_iters: Int,
    mojo_iters: Int,
) raises:
    print("")
    print("================================================================")
    print(
        "SHAPE: "
        + name
        + "  (in="
        + String(in_dim)
        + " hidden=["
        + hidden
        + "] out="
        + String(out_dim)
        + " batch="
        + String(batch)
        + ")"
    )
    print("================================================================")

    var m = mlp_module.MLPInference(
        in_dim, PythonObject(hidden), out_dim, batch, PythonObject(device)
    )
    print(String(m.info()))

    # ---- Python-internal benches (interop-free attribution) ----
    # warmup
    _ = m.bench_full(50)
    var us = 1.0e6
    var comp = Float64(py=m.bench_compute(py_iters)) / Float64(py_iters) * us
    var h2d = Float64(py=m.bench_h2d(py_iters)) / Float64(py_iters) * us
    var d2h = Float64(py=m.bench_d2h(py_iters)) / Float64(py_iters) * us
    var full = Float64(py=m.bench_full(py_iters)) / Float64(py_iters) * us

    # ---- Mojo-side interop floor: noop() in a Mojo loop ----
    var x = m.make_host_input()
    for _ in range(100):
        _ = m.noop()
    var t0 = perf_counter_ns()
    for _ in range(mojo_iters):
        _ = m.noop()
    var interop_floor = (
        Float64(perf_counter_ns() - t0) / Float64(mojo_iters) / 1000.0
    )

    # ---- Mojo-side realistic per-call: infer(x) in a Mojo loop ----
    for _ in range(50):
        _ = m.infer(x)
    t0 = perf_counter_ns()
    for _ in range(mojo_iters):
        _ = m.infer(x)
    var infer_mojo = (
        Float64(perf_counter_ns() - t0) / Float64(mojo_iters) / 1000.0
    )

    # Device+transfer is the irreducible work; Python glue is everything else the
    # Python loop spends (numpy contiguity, Buffer object creation, attr lookups).
    var device_plus_transfer = comp + h2d + d2h
    var python_glue = full - device_plus_transfer

    print("")
    print("  stage                                us/call")
    print("  ------------------------------------------------")
    print("  MAX device compute (no interop)      " + pad(f2(comp), 10))
    print("  H2D transfer (host->device)          " + pad(f2(h2d), 10))
    print("  D2H transfer (device->host)          " + pad(f2(d2h), 10))
    print("  = device + transfer subtotal         " + pad(f2(device_plus_transfer), 10))
    print("  Python glue per call (numpy/Buffer)  " + pad(f2(python_glue), 10))
    print("  Python end-to-end (full)             " + pad(f2(full), 10))
    print("  ------------------------------------------------")
    print("  Mojo->Python interop FLOOR (noop)    " + pad(f2(interop_floor), 10))
    print("  Mojo end-to-end infer() per call     " + pad(f2(infer_mojo), 10))
    print(
        "  (Mojo infer vs Python full = "
        + f2(infer_mojo)
        + " vs "
        + f2(full)
        + " -> bridge adds ~interop floor; any gap is run-to-run noise)"
    )


def main() raises:
    # Make the local noeira_max package importable from the embedded Python.
    var sys = Python.import_module("sys")
    _ = sys.path.insert(0, PythonObject("."))
    var mlp_module = Python.import_module("noeira_max.mlp_inference")

    var device = String("gpu")  # "gpu" -> Metal on Apple, CUDA on NVIDIA
    var py_iters = 2000
    var mojo_iters = 2000

    print("MAX MLP inference interop benchmark | device=" + device)
    print("(compile time is one-time; all us/call numbers are warm steady-state)")

    # RL-realistic actor MLP across the batch regimes that matter:
    #   batch=1     -> single-action selection (interop/launch dominated)
    #   batch=64    -> small batched envs
    #   batch=1024  -> large batched envs (device-compute dominated)
    run_shape(mlp_module, "actor-b1", 17, "256,256", 6, 1, device, py_iters, mojo_iters)
    run_shape(mlp_module, "actor-b64", 17, "256,256", 6, 64, device, py_iters, mojo_iters)
    run_shape(mlp_module, "actor-b1024", 17, "256,256", 6, 1024, device, py_iters, mojo_iters)

    # A wider net to show the compute/interop crossover.
    run_shape(mlp_module, "wide-b1", 256, "512,512", 64, 1, device, py_iters, mojo_iters)
    run_shape(mlp_module, "wide-b1024", 256, "512,512", 64, 1024, device, py_iters, mojo_iters)

    print("")
    print("Done. Reminder: numbers above are path (A) Mojo->CPython->MAX (Python in")
    print("the hot loop) — a pessimistic upper bound vs path (B) Mojo->MAX C API.")
