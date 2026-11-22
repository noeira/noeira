"""Phase 2 planners: MPPICPU on LinearQuadratic.

LQ has a closed-form LQR optimum, so MPPI's first action under enough
samples / iters should match ``a* = -K·z`` within a loose tolerance.

Test plan:
  1. ``test_mppi_recovers_lqr_action`` — MPPICPU on a stable LQ
     problem (A=0.9, B=1.0, Q=1.0, R=0.1) starting from ``z=1.0``;
     check the planned first action sits within 25% of the LQR
     argmax. (Tight tolerance is infeasible at modest sample counts
     because MPPI's softmax is a soft argmax; we only need to verify
     "in the right direction and roughly right magnitude".)
  2. ``test_mppi_zero_state_no_op`` — at ``z=0`` (zero-cost equilibrium),
     LQR says ``a*=0``; MPPI should likewise pull toward 0 — assert
     ``|a| < 0.15``.
  3. ``test_mppi_deterministic_same_seed`` — same seed + same z0
     produces bit-identical first action across two calls (with the
     planner reset in between).

The callback (``LQRolloutCallback``) is a tiny 1D adapter implementing
``RolloutCallbackCPU`` against a scalar ``LinearQuadratic1D`` stub.
Policy mean = 0 (no learned π); terminal value = 0 (no Q bootstrap) —
that's fine for short-horizon MPPI dominated by rollout reward.
"""

from std.math import abs as math_abs
from std.random import random_float64, seed as _set_seed
from std.testing import assert_true

from noeira.planners.trajectory import MPPICPU, RolloutCallbackCPU
from noeira.planners.testing import LinearQuadratic1D


@fieldwise_init
struct LQRolloutCallback(
    Movable, Deinitable, RolloutCallbackCPU
):
    """1D LinearQuadratic adapter for ``RolloutCallbackCPU``.

    LATENT_DIM = ACTION_DIM = 1: just a thin wrapper around the
    scalar ``LinearQuadratic1D`` stub so MPPICPU can exercise the
    trait at the smallest interesting dim.
    """

    comptime LATENT_DIM: Int = 1
    comptime ACTION_DIM: Int = 1

    var lq: LinearQuadratic1D

    def policy_action_cpu(
        mut self,
        z: List[Float64],
        mut action_out: List[Float64],
    ) raises:
        # No learned policy in the stub — return zero. MPPI's
        # pi-traj seeding then degenerates to "zero-seeded with
        # noise" warm-starts, which is harmless.
        action_out[0] = 0.0

    def rollout_step_cpu(
        mut self,
        z: List[Float64],
        a: List[Float64],
        mut z_next_out: List[Float64],
    ) raises -> Float64:
        z_next_out[0] = self.lq.step(z[0], a[0])
        return self.lq.reward(z[0], a[0])

    def terminal_value_cpu(
        mut self,
        z: List[Float64],
    ) raises -> Float64:
        # No Q-bootstrap for the stub. Short horizon + cost-to-go
        # decay makes the terminal term small.
        return 0.0


comptime LATENT_DIM: Int = 1
comptime ACTION_DIM: Int = 1
comptime HORIZON: Int = 5
comptime NUM_SAMPLES: Int = 256
comptime NUM_PI_TRAJS: Int = 0
comptime NUM_ELITES: Int = 32  # top-K elite filter (reference TD-MPC2 recipe)
comptime NUM_ITERATIONS: Int = 6


def test_mppi_recovers_lqr_action() raises:
    """MPPI's planned first action on a stable LQ problem starts
    from z=1.0 and should point in the LQR-argmax direction with
    roughly LQR-like magnitude.

    LQR for A=0.9, B=1.0, Q=1.0, R=0.1 gives K≈0.873, so the
    expected optimal action at z=1 is ``a*≈-0.873``. Asserting
    within 25% of that absolute value is a generous-but-meaningful
    bound — MPPI's softmax is a soft argmax, and 6 iters with
    256 samples at NUM_PI_TRAJS=0 doesn't fully concentrate.
    """
    var lq = LinearQuadratic1D(A=0.9, B=1.0, Q=1.0, R=0.1)
    var cb = LQRolloutCallback(lq=lq.copy())
    var planner = MPPICPU[
        LATENT_DIM,
        ACTION_DIM,
        HORIZON,
        NUM_SAMPLES,
        NUM_PI_TRAJS,
        NUM_ITERATIONS,
        NUM_ELITES,
    ](0xA1B2)
    var z0: List[Float64] = [1.0]
    var action = planner.plan(
        cb,
        z0,
        gamma=0.95,
        temperature=10.0,
        action_scale=1.0,
        deterministic=True,
    )

    var K = lq.lqr_gain_infinite_horizon()
    var lqr_action = -K * 1.0  # expected ~-0.873

    # Sign: MPPI must pull toward 0 (i.e. produce a negative action
    # for positive z).
    assert_true(
        action[0] < 0.0,
        "MPPI first action should be negative for z=1, got "
        + String(action[0]),
    )
    # Magnitude: |a| within 25% of |a*|.
    var rel_err = math_abs(action[0] - lqr_action) / math_abs(
        lqr_action
    )
    assert_true(
        rel_err < 0.25,
        "MPPI action "
        + String(action[0])
        + " not within 25% of LQR optimum "
        + String(lqr_action)
        + " (rel_err="
        + String(rel_err)
        + ")",
    )


def test_mppi_zero_state_no_op() raises:
    """At z=0, the cost is zero whatever a we pick, but the optimal
    action is also 0 (LQR says ``a*=-K·0=0``). MPPI should pull
    toward zero on average. With deterministic=True (no per-action
    exploration noise) at moderate iters the result should be
    small in absolute value.
    """
    var lq = LinearQuadratic1D(A=0.9, B=1.0, Q=1.0, R=0.1)
    var cb = LQRolloutCallback(lq=lq.copy())
    var planner = MPPICPU[
        LATENT_DIM,
        ACTION_DIM,
        HORIZON,
        NUM_SAMPLES,
        NUM_PI_TRAJS,
        NUM_ITERATIONS,
        NUM_ELITES,
    ](0xBEEF)
    var z0: List[Float64] = [0.0]
    var action = planner.plan(
        cb,
        z0,
        gamma=0.95,
        temperature=10.0,
        action_scale=1.0,
        deterministic=True,
    )
    assert_true(
        math_abs(action[0]) < 0.15,
        "MPPI at z=0 should be near 0, got " + String(action[0]),
    )


def test_mppi_deterministic_same_seed() raises:
    """Same seed + same inputs + freshly-constructed planner →
    bit-identical first action.
    """
    var lq = LinearQuadratic1D(A=0.9, B=1.0, Q=1.0, R=0.1)
    var z0: List[Float64] = [1.0]

    var cb1 = LQRolloutCallback(lq=lq.copy())
    var planner1 = MPPICPU[
        LATENT_DIM,
        ACTION_DIM,
        HORIZON,
        NUM_SAMPLES,
        NUM_PI_TRAJS,
        NUM_ITERATIONS,
        NUM_ELITES,
    ](0xC0DE)
    var a1 = planner1.plan(
        cb1,
        z0,
        gamma=0.95,
        temperature=10.0,
        action_scale=1.0,
        deterministic=True,
    )

    var cb2 = LQRolloutCallback(lq=lq.copy())
    var planner2 = MPPICPU[
        LATENT_DIM,
        ACTION_DIM,
        HORIZON,
        NUM_SAMPLES,
        NUM_PI_TRAJS,
        NUM_ITERATIONS,
        NUM_ELITES,
    ](0xC0DE)
    var a2 = planner2.plan(
        cb2,
        z0,
        gamma=0.95,
        temperature=10.0,
        action_scale=1.0,
        deterministic=True,
    )

    # Identical seed + identical planner state → bit-identical
    # PRNG draws → bit-identical output.
    assert_true(
        a1[0] == a2[0],
        "Same-seed runs should be bit-identical: "
        + String(a1[0])
        + " vs "
        + String(a2[0]),
    )


def _plan_once(rng_seed: UInt64, disturb: Bool) raises -> Float64:
    """The first action of a fresh planner built with `rng_seed`, on the
    LQ problem at z = 1; `disturb` draws from the GLOBAL generator first."""
    if disturb:
        for _ in range(7):
            _ = random_float64()
    var lq = LinearQuadratic1D(A=0.9, B=1.0, Q=1.0, R=0.1)
    var cb = LQRolloutCallback(lq=lq.copy())
    var planner = MPPICPU[
        LATENT_DIM,
        ACTION_DIM,
        HORIZON,
        NUM_SAMPLES,
        NUM_PI_TRAJS,
        NUM_ITERATIONS,
        NUM_ELITES,
    ](rng_seed)
    var z0: List[Float64] = [1.0]
    var a = planner.plan(
        cb, z0, gamma=0.95, temperature=10.0, action_scale=1.0, deterministic=False,
    )
    return a[0]


def test_mppi_noise_is_its_own_stream() raises:
    """The planner's draws come from its seed alone, and it leaves the
    global `std.random` generator where it found it.

    Until Nov 2026 `MPPICPU` drew from the global generator: whatever else
    the process drew changed the plan, and the plan's draws shifted every
    later draw of the process.
    """
    # 1. the global generator's state does not enter the plan
    _set_seed(1)
    var a = _plan_once(0xC0DE, False)
    _set_seed(2)
    var b = _plan_once(0xC0DE, True)
    assert_true(a == b, "the global generator changed the plan: " + String(a) + " vs " + String(b))

    # 2. the seed does: another seed, another plan
    var c = _plan_once(0xC0DF, False)
    assert_true(a != c, "two seeds gave the same action: " + String(a))

    # 3. a plan does not move the global generator
    _set_seed(77)
    var r1 = random_float64()
    _set_seed(77)
    _ = _plan_once(0xC0DE, False)
    var r2 = random_float64()
    assert_true(r1 == r2, "a plan drew from the global generator: " + String(r1) + " vs " + String(r2))

    # 4. `reseed` starts the stream again; without it the second plan differs
    var lq = LinearQuadratic1D(A=0.9, B=1.0, Q=1.0, R=0.1)
    var cb = LQRolloutCallback(lq=lq.copy())
    var planner = MPPICPU[
        LATENT_DIM,
        ACTION_DIM,
        HORIZON,
        NUM_SAMPLES,
        NUM_PI_TRAJS,
        NUM_ITERATIONS,
        NUM_ELITES,
    ](0xC0DE)
    var z0: List[Float64] = [1.0]
    var p1 = planner.plan(cb, z0, gamma=0.95, temperature=10.0, action_scale=1.0, deterministic=False)
    planner.start_episode()
    var p2 = planner.plan(cb, z0, gamma=0.95, temperature=10.0, action_scale=1.0, deterministic=False)
    planner.start_episode()
    planner.reseed(0xC0DE)
    var p3 = planner.plan(cb, z0, gamma=0.95, temperature=10.0, action_scale=1.0, deterministic=False)
    assert_true(p1[0] == a, "a planner built with the seed differs from _plan_once: " + String(p1[0]) + " vs " + String(a))
    assert_true(p2[0] != p1[0], "the second plan drew the first plan's numbers: " + String(p2[0]))
    assert_true(p3[0] == p1[0], "reseed did not start the stream again: " + String(p3[0]) + " vs " + String(p1[0]))


def main() raises:
    print("=== Phase 2 planners: MPPICPU on LinearQuadratic ===")
    test_mppi_recovers_lqr_action()
    print("  PASS MPPI recovers LQR-argmax direction within 25%")
    test_mppi_zero_state_no_op()
    print("  PASS MPPI at zero state stays near zero")
    test_mppi_deterministic_same_seed()
    print("  PASS same-seed runs bit-identical")
    test_mppi_noise_is_its_own_stream()
    print("  PASS the noise is the planner's own stream (seed in, global generator untouched)")
    print("OK")
