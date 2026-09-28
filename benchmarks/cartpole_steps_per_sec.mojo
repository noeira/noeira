"""CartPole-v1 environment throughput: native Mojo vs Gymnasium.

Three rows, all single-env on one CPU core, random actions, auto-reset on
done, no agent in the loop:

  1. Gymnasium, pure Python      — `benchmarks/cartpole_sps_gym.py:run`
  2. Gymnasium via Mojo wrapper  — `GymCartPoleEnv.step_raw` (Python interop)
  3. Native Mojo                 — `CartPoleEnv[DType.float64].step_raw`

Each row is the best of REPEATS timed runs after a warm-up. A checksum of
reward + cart position is carried out of every loop so no path can be
optimised away; rows 1-2 and 3 draw different reset states, so their
checksums differ.

    pixi run mojo run -I . benchmarks/cartpole_steps_per_sec.mojo
"""

from std.python import Python
from std.random import random_si64, seed
from std.time import perf_counter_ns

from noeira.envs import CartPoleEnv
from noeira.envs.gymnasium import GymCartPoleEnv

comptime REPEATS = 5
comptime NATIVE_STEPS = 50_000_000
comptime PYTHON_STEPS = 500_000
comptime ACTION_BUF = 1 << 20  # power of two: index with `i & (ACTION_BUF-1)`


def make_actions() -> List[Int]:
    var actions = List[Int](capacity=ACTION_BUF)
    for _ in range(ACTION_BUF):
        actions.append(Int(random_si64(0, 1)))
    return actions^


def bench_native(
    num_steps: Int, actions: List[Int]
) -> Tuple[Float64, Float64]:
    var env = CartPoleEnv[DType.float64]()
    _ = env.reset_obs()
    var checksum: Float64 = 0.0
    var t0 = perf_counter_ns()
    for i in range(num_steps):
        var r = env.step_raw(actions[i & (ACTION_BUF - 1)])
        checksum += Float64(r[1]) + r[0][0]
        if r[2]:
            _ = env.reset_obs()
    var dt = Float64(perf_counter_ns() - t0) / 1e9
    return (dt, checksum)


def bench_wrapper(
    num_steps: Int, actions: List[Int]
) raises -> Tuple[Float64, Float64]:
    var env = GymCartPoleEnv()
    _ = env.reset_obs()
    var checksum: Float64 = 0.0
    var t0 = perf_counter_ns()
    for i in range(num_steps):
        var r = env.step_raw(actions[i & (ACTION_BUF - 1)])
        checksum += r[1] + r[0][0]
        if r[2]:
            _ = env.reset_obs()
    var dt = Float64(perf_counter_ns() - t0) / 1e9
    env.close()
    return (dt, checksum)


def bench_python(num_steps: Int, run_seed: Int) raises -> Tuple[Float64, Float64]:
    var sys = Python.import_module("sys")
    _ = sys.path.insert(0, "benchmarks")
    var mod = Python.import_module("cartpole_sps_gym")
    var r = mod.run(num_steps, run_seed)
    return (Float64(py=r[0]), Float64(py=r[1]))


def with_commas(n: Int) -> String:
    var s = String(n)
    var out = String()
    var count = 0
    for i in range(s.byte_length() - 1, -1, -1):
        if count > 0 and count % 3 == 0:
            out = "," + out
        out = s[byte=i] + out
        count += 1
    return out


def fixed1(x: Float64) -> String:
    var t = Int(x * 10.0 + 0.5)
    return String(t // 10) + "." + String(t % 10)


def report(name: String, steps: Int, best: Float64, baseline_sps: Float64):
    var sps = Float64(steps) / best
    var ns_per_step = best / Float64(steps) * 1e9
    print(
        "  ",
        name,
        "|",
        with_commas(Int(sps)),
        "steps/s |",
        fixed1(ns_per_step),
        "ns/step |",
        fixed1(sps / baseline_sps),
        "x vs pure Python",
    )


def main() raises:
    seed(0)
    var actions = make_actions()

    print("CartPole-v1 throughput (single env, 1 CPU core, random actions)")
    print("  best of", REPEATS, "runs after a warm-up")
    print()

    # Warm-ups: first Python import, first gym.make, page-in of the loops.
    _ = bench_python(10_000, 0)
    _ = bench_wrapper(10_000, actions)
    _ = bench_native(1_000_000, actions)

    var best_py = Float64.MAX
    var best_wrap = Float64.MAX
    var best_nat = Float64.MAX
    var sink: Float64 = 0.0
    for rep in range(REPEATS):
        var p = bench_python(PYTHON_STEPS, rep)
        var w = bench_wrapper(PYTHON_STEPS, actions)
        var n = bench_native(NATIVE_STEPS, actions)
        best_py = min(best_py, p[0])
        best_wrap = min(best_wrap, w[0])
        best_nat = min(best_nat, n[0])
        sink += p[1] + w[1] + n[1]
        print(
            "  run",
            rep + 1,
            ": python",
            Int(Float64(PYTHON_STEPS) / p[0]),
            "| wrapper",
            Int(Float64(PYTHON_STEPS) / w[0]),
            "| native",
            Int(Float64(NATIVE_STEPS) / n[0]),
            "steps/s",
        )

    var py_sps = Float64(PYTHON_STEPS) / best_py
    print()
    report("Gymnasium (pure Python)    ", PYTHON_STEPS, best_py, py_sps)
    report("Gymnasium via Mojo wrapper ", PYTHON_STEPS, best_wrap, py_sps)
    report("noeira native Mojo         ", NATIVE_STEPS, best_nat, py_sps)
    print()
    print("  (checksum", sink, ")")
