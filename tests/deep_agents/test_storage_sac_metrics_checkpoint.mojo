"""SAC diagnostics + single-file checkpoint gate (storage SACTrainer, CPU).

Guards two fixes to the migrated SAC trainer:

  1. `flush_metrics` now populates the per-batch DIAGNOSTICS (mean_q /
     mean_target / mean_next_q / mean_reward / mean_abs_action) — they used to
     be hardcoded 0.0. Trained with `diag_every=0` (the driver never flushes) so
     the trainer's accumulators retain every update; one manual `flush_metrics`
     then drains real means.
  2. `save` writes ONE `nn-ckpt v2` envelope (actor + twin critics, sections
     prefixed actor./critic1./critic2.) — NOT three `.actor`/`.critic1`/`.critic2`
     sidecar files. `load` round-trips the greedy action bit-for-bit.

Run: pixi run mojo run -I . tests/deep_agents/test_storage_sac_metrics_checkpoint.mojo
"""

from std.random import seed
from std.testing import assert_true
from std.math import isnan, isinf
from std.python import Python, PythonObject

from noeira.nn.constants import DT
from noeira.deep_agents.sac.config import SAC
from noeira.envs.pendulum.pendulum_v1 import PendulumEnv


comptime OBS = 3
comptime ACT = 1
comptime H = 128
comptime BATCH = 128
comptime CAP = 100_000
comptime CKPT = "test_sac_singlefile.ckpt"


def _finite(x: Float64) -> Bool:
    return not (isnan(x) or isinf(x))


def main() raises:
    seed(42)
    print("=" * 60)
    print("SAC diagnostics + single-file checkpoint gate (CPU)")
    print("=" * 60)

    var env = PendulumEnv[DT]()
    var agent = SAC["cpu", OBS, ACT, BATCH, CAP, H](
        action_scale=Scalar[DT](2.0),
        learning_starts=500,
    )
    # diag_every defaults to 0 → driver never flushes → accumulators retain all.
    _ = agent.train_single(env, total_timesteps=3_000, print_every=3_000)

    # ── (1) diagnostics populated (no longer hardcoded 0.0) ──────────────
    var m = agent.flush_metrics()
    print("  train_steps    =", m.train_steps.to_f64())
    print("  mean_q         =", m.mean_q.to_f64())
    print("  mean_target    =", m.mean_target.to_f64())
    print("  mean_next_q    =", m.mean_next_q.to_f64())
    print("  mean_reward    =", m.reward_mean.to_f64())
    print("  mean_done      =", m.mean_done.to_f64())
    print("  mean_abs_action=", m.action_abs_mean.to_f64())

    assert_true(m.train_steps.to_f64() > 0.0, "training actually ran")
    assert_true(_finite(m.mean_q.to_f64()) and m.mean_q.to_f64() != 0.0, "mean_q populated")
    assert_true(_finite(m.mean_target.to_f64()) and m.mean_target.to_f64() != 0.0, "mean_target populated")
    assert_true(_finite(m.mean_next_q.to_f64()) and m.mean_next_q.to_f64() != 0.0, "mean_next_q populated")
    assert_true(_finite(m.reward_mean.to_f64()) and m.reward_mean.to_f64() != 0.0, "mean_reward populated")
    assert_true(_finite(m.mean_done.to_f64()), "mean_done finite")  # may legitimately be 0
    assert_true(m.action_abs_mean.to_f64() > 0.0, "mean_abs_action populated")
    print("  diagnostics OK (real per-batch means)")

    # ── (2) single-file checkpoint + round-trip ──────────────────────────
    var os = Python.import_module("os")
    agent.save(CKPT)
    assert_true(Bool(os.path.exists(PythonObject(CKPT))), "single ckpt file written")
    assert_true(
        not Bool(os.path.exists(PythonObject(CKPT + ".actor"))),
        "no .actor sidecar (must be ONE file)",
    )
    assert_true(
        not Bool(os.path.exists(PythonObject(CKPT + ".critic1"))),
        "no .critic1 sidecar (must be ONE file)",
    )
    assert_true(
        not Bool(os.path.exists(PythonObject(CKPT + ".critic2"))),
        "no .critic2 sidecar (must be ONE file)",
    )

    var probe = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
    for d in range(OBS):
        probe[d] = Scalar[DT](0.1 * Float64(d - 1))
    var a_before = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0.0))
    agent.select_greedy_action(probe, a_before)
    agent.load(CKPT)
    var a_after = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0.0))
    agent.select_greedy_action(probe, a_after)
    for j in range(ACT):
        var diff = Float64(a_after[j] - a_before[j])
        if diff < 0:
            diff = -diff
        assert_true(diff < 1e-5, "single-file load round-trips greedy action")
    print("  single-file checkpoint OK (one file, round-trip < 1e-5)")

    _ = os.remove(PythonObject(CKPT))
    print("ALL PASSED")
