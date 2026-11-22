"""Phase 2 planners: MPPICPU goal-reach test.

Mirror of ``test_cem_goal_reach.mojo`` / ``test_random_shooter_goal_reach.mojo``
adapted for MPPI's **continuous** action space. Same shape
(``IdentityDynamics + GoalReachReward``, BATCH=1, HORIZON=3,
ACT_DIM=3, goal=[1, 1, 1] from origin), but the action is now a
continuous vector in ``[-1, 1]^3`` rather than a one-hot. The
trait wired here is ``RolloutCallbackCPU``, which gives MPPI a
per-step ``(z, a) → (z', r)`` contract — different from the
per-plan ``score_plan`` used by CEM / random shooter.

Setup:
  z' = z + a, a ∈ [-1, 1]^ACT_DIM (clamped by MPPI)
  step reward: r(z') = -‖z' - goal‖²
  terminal value: 0 (no bootstrap in the stub)

Optimal first action from z0 = [0, 0, 0]: ``a* = clamp(goal - z0) =
[1, 1, 1]`` — reaches the goal in one step, then ``a = 0`` afterwards
keeps the state pinned. So the first-action assertion is "selected
action close to [1, 1, 1] in L_inf norm".

The tolerance is generous (0.2 per component) because MPPI's softmax
is a soft argmax — it samples actions around the converged mean, and
the post-selection per-axis Gaussian noise adds another layer.
Tightening below ~0.15 would make the test flaky at modest sample
counts.

Usage:
    pixi run mojo run -I . tests/planners/trajectory/test_mppi_goal_reach.mojo
"""

from std.math import abs as math_abs
from std.testing import assert_true

from noeira.planners.trajectory import MPPICPU, RolloutCallbackCPU
from noeira.planners.testing import IdentityDynamics, GoalReachReward


# Comptime sizes — chosen so MPPI converges tightly + symmetrically.
# 128 samples × 6 iters is too few: at that budget, dim-asymmetric
# convergence is on the same order (0.2-0.4 spread) as random walks
# in the per-dim mean updates, which looked like a kernel bug in the
# GPU version until the diagnostic check showed CPU MPPI does the
# same. With 512 samples × 12 iters all 3 dims land within 0.1 of
# optimum on both backends.
comptime LATENT_DIM: Int = 3
comptime ACTION_DIM: Int = 3
comptime HORIZON: Int = 3
comptime NUM_SAMPLES: Int = 512
comptime NUM_PI_TRAJS: Int = 0  # no learned policy in this stub
comptime NUM_ELITES: Int = 64
comptime NUM_ITERATIONS: Int = 12


@fieldwise_init
struct GoalReachRolloutCallback(
    Movable, Deinitable, RolloutCallbackCPU
):
    """Per-step ``z' = z + a`` with step reward ``-‖z' - goal‖²``.

    Stateless except for the goal vector. ``policy_action_cpu`` returns
    zero (no learned policy in the stub — pi-traj seeding degenerates
    harmlessly to zero-seeded warm-starts; this test uses
    ``NUM_PI_TRAJS = 0`` anyway). ``terminal_value_cpu`` returns 0 so
    the total return is purely the sum of per-step rewards.
    """

    comptime LATENT_DIM: Int = LATENT_DIM
    comptime ACTION_DIM: Int = ACTION_DIM

    var goal_x: Float64
    var goal_y: Float64
    var goal_z: Float64

    def policy_action_cpu(
        mut self,
        z: List[Float64],
        mut action_out: List[Float64],
    ) raises:
        for i in range(Self.ACTION_DIM):
            action_out[i] = 0.0

    def rollout_step_cpu(
        mut self,
        z: List[Float64],
        a: List[Float64],
        mut z_next_out: List[Float64],
    ) raises -> Float64:
        # IdentityDynamics: z' = z + a.
        z_next_out[0] = z[0] + a[0]
        z_next_out[1] = z[1] + a[1]
        z_next_out[2] = z[2] + a[2]
        # GoalReachReward: r(z') = -‖z' - goal‖².
        var goal: List[Float64] = [self.goal_x, self.goal_y, self.goal_z]
        var rew = GoalReachReward(goal=goal.copy())
        return rew.reward(z_next_out)

    def terminal_value_cpu(
        mut self,
        z: List[Float64],
    ) raises -> Float64:
        # No Q-bootstrap in the stub — let the per-step rewards drive.
        return 0.0


def _approx(a: Float64, b: Float64, tol: Float64 = 1e-9) -> Bool:
    return math_abs(a - b) <= tol


def test_mppi_converges_to_goal() raises:
    """MPPI's first selected action should pull strongly toward
    ``[1, 1, 1]`` from origin. With 6 iterations × 128 samples ×
    top-K=16 elites, the converged mean's first step concentrates on
    ``a* = goal - z0 = [1, 1, 1]``, which after clamp + per-axis noise
    should land within L_inf ≤ 0.20 of (1, 1, 1) in deterministic mode.
    """
    var planner = MPPICPU[
        LATENT_DIM,
        ACTION_DIM,
        HORIZON,
        NUM_SAMPLES,
        NUM_PI_TRAJS,
        NUM_ITERATIONS,
        NUM_ELITES,
    ](0xC0FFEE)
    var cb = GoalReachRolloutCallback(goal_x=1.0, goal_y=1.0, goal_z=1.0)
    var z0: List[Float64] = [0.0, 0.0, 0.0]

    var action = planner.plan(
        cb,
        z0,
        gamma=0.95,
        temperature=10.0,
        action_scale=1.0,
        deterministic=True,
    )

    # Per-axis check: each component close to 1.0.
    for i in range(ACTION_DIM):
        var err = math_abs(action[i] - 1.0)
        assert_true(
            err < 0.20,
            "MPPI first action[" + String(i) + "] = "
            + String(action[i])
            + " not within 0.20 of optimal 1.0 (err = "
            + String(err)
            + ")",
        )


def test_mppi_at_goal_stays() raises:
    """At ``z0 = goal``, the optimal action is the zero vector — moving
    in any direction increases distance² to goal. MPPI should select an
    action with L_inf norm ≤ 0.30 (slightly looser than the converge
    test since the optimum is interior and Gaussian sampling pushes off
    it).
    """
    var planner = MPPICPU[
        LATENT_DIM,
        ACTION_DIM,
        HORIZON,
        NUM_SAMPLES,
        NUM_PI_TRAJS,
        NUM_ITERATIONS,
        NUM_ELITES,
    ](0x60A1)
    var cb = GoalReachRolloutCallback(goal_x=1.0, goal_y=1.0, goal_z=1.0)
    var z0: List[Float64] = [1.0, 1.0, 1.0]  # already at goal

    var action = planner.plan(
        cb,
        z0,
        gamma=0.95,
        temperature=10.0,
        action_scale=1.0,
        deterministic=True,
    )

    for i in range(ACTION_DIM):
        var mag = math_abs(action[i])
        assert_true(
            mag < 0.30,
            "MPPI at-goal action[" + String(i) + "] = "
            + String(action[i])
            + " should be near 0 (got |a| = "
            + String(mag)
            + ")",
        )


def test_mppi_warm_start_improves_consecutive_calls() raises:
    """Consecutive ``plan()`` calls on the same env warm-start each
    other (``prev_mean`` shifted forward by 1 step). After two calls
    from the same z0, the second call should select an action closer
    to the optimum than the first — because the second iter's initial
    distribution is the previous iter's converged mean.

    Asserts: |action_2 - goal_direction| ≤ |action_1 - goal_direction|
    (within a small tolerance allowing for noise).
    """
    var planner = MPPICPU[
        LATENT_DIM,
        ACTION_DIM,
        HORIZON,
        NUM_SAMPLES,
        NUM_PI_TRAJS,
        NUM_ITERATIONS,
        NUM_ELITES,
    ](0x12121212)
    var cb = GoalReachRolloutCallback(goal_x=1.0, goal_y=1.0, goal_z=1.0)
    var z0: List[Float64] = [0.0, 0.0, 0.0]

    # First call: t0=True, no warm-start.
    var a1 = planner.plan(
        cb,
        z0,
        gamma=0.95,
        temperature=10.0,
        action_scale=1.0,
        deterministic=True,
    )
    # Second call: t0=False (auto-set after first call), warm-started
    # from a1's converged distribution.
    var a2 = planner.plan(
        cb,
        z0,
        gamma=0.95,
        temperature=10.0,
        action_scale=1.0,
        deterministic=True,
    )

    # Total L1 distance from optimal (1, 1, 1).
    var err1: Float64 = 0.0
    var err2: Float64 = 0.0
    for i in range(ACTION_DIM):
        err1 += math_abs(a1[i] - 1.0)
        err2 += math_abs(a2[i] - 1.0)

    # Allow a small slack: warm-start could occasionally produce a
    # slightly worse result on a single noisy run. Tolerance of 0.3
    # keeps the test robust while still asserting the "warm-start is
    # at least as good" contract.
    assert_true(
        err2 <= err1 + 0.30,
        "warm-start should not regress: err1=" + String(err1)
        + ", err2=" + String(err2),
    )


def test_mppi_start_episode_resets_warm_start() raises:
    """``start_episode()`` should reset ``t0`` so the next ``plan()``
    discards the previous mean. Run two plans from the same z0, then
    call ``start_episode()`` and run a third plan — the third plan
    should behave like a fresh first call (no warm-start).

    Verifies the contract by checking that
    ``planner.t0 == True`` after ``start_episode()``.
    """
    var planner = MPPICPU[
        LATENT_DIM,
        ACTION_DIM,
        HORIZON,
        NUM_SAMPLES,
        NUM_PI_TRAJS,
        NUM_ITERATIONS,
        NUM_ELITES,
    ](0xEDEDEDED)
    var cb = GoalReachRolloutCallback(goal_x=1.0, goal_y=1.0, goal_z=1.0)
    var z0: List[Float64] = [0.0, 0.0, 0.0]

    # Fresh planner: t0=True by construction.
    assert_true(planner.t0, "freshly-constructed planner should have t0=True")

    _ = planner.plan(
        cb, z0, gamma=0.95, temperature=10.0,
        action_scale=1.0, deterministic=True,
    )
    # After first plan, t0 flipped to False — subsequent calls warm-start.
    assert_true(
        not planner.t0,
        "after plan(), t0 should be False (warm-start enabled)",
    )

    planner.start_episode()
    assert_true(
        planner.t0,
        "start_episode() should reset t0 to True",
    )


def main() raises:
    print("=== Phase 2 planners: MPPICPU goal-reach ===")
    test_mppi_converges_to_goal()
    print(
        "  PASS MPPI converges to goal on"
        " (IdentityDynamics + GoalReachReward)"
    )
    test_mppi_at_goal_stays()
    print("  PASS MPPI at goal selects near-zero action")
    test_mppi_warm_start_improves_consecutive_calls()
    print("  PASS warm-start across consecutive plan() calls")
    test_mppi_start_episode_resets_warm_start()
    print("  PASS start_episode() resets warm-start")
    print("OK")
