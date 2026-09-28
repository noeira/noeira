"""`FBTrainer` smoke gate — does a step run, and does `B` avoid collapsing?

A trainer with five networks and three accumulating vjps into one of them has
many ways to be wrong that produce no error. This gate checks the two that a
loss curve cannot show:

  [2] `B` does not collapse. `L_ortho` exists to prevent every row of `B` from
      shrinking to the same direction, and §11's warning is that `L_FB` keeps
      descending while it happens. So the assertion is on the ROW SPREAD of
      `B` after training, not on the loss.

  [3] the three `B` vjps really do accumulate. Checked directly: a step with
      `ortho_weight = 0` and a step with `ortho_weight = 1` must move `B`'s
      parameters DIFFERENTLY. If the third vjp were overwriting rather than
      accumulating — or if `zero_grad` ran mid-step — the ortho contribution
      would be dropped and the two would coincide.

⚠ This is a smoke gate on random data, not a convergence test. It runs a few
dozen steps on a synthetic dataset and asserts structural properties. Whether
FB learns anything USEFUL is milestone 1's `point_mass` evaluation, which needs
a real dataset — see `docs/BFM_ZERO_SHOT_RL.md` §13 step 4a.

Run:
    pixi run mojo run -I . tests/deep_agents/test_fb_trainer_smoke.mojo
"""

from std.math import abs, sqrt
from std.random import random_float64, seed
from std.testing import assert_true

from noeira.nn.constants import DT
from noeira.nn.core.tensor import Tensor
from noeira.nn.combinators.sequential import Sequential
from noeira.nn.primitives.linear import Linear
from noeira.nn.primitives.activations import Tanh, ReLU
from noeira.deep_agents.fb.trainer import FBTrainer
from noeira.deep_agents.fb import sample_z_uniform


comptime OBS: Int = 4
comptime ACT: Int = 2
comptime D: Int = 6
comptime BATCH: Int = 8
comptime HID: Int = 32
comptime STEPS: Int = 40
comptime SEED: Int = 20260805

comptime F_IN = OBS + ACT + D
comptime A_IN = OBS + D

comptime FNet = Sequential[
    Linear[F_IN, HID], ReLU[HID], Linear[HID, D]
]
comptime BNet = Sequential[
    Linear[OBS, HID], ReLU[HID], Linear[HID, D]
]
comptime ANet = Sequential[
    Linear[A_IN, HID], ReLU[HID], Linear[HID, ACT], Tanh[ACT]
]

comptime Trainer = FBTrainer[FNet, BNet, ANet, OBS, ACT, D, BATCH]


def _rand_tensor(n: Int, scale: Float64) raises -> Tensor:
    var t = Tensor.alloc(n)
    for i in range(n):
        t.data[i] = Scalar[DT]((random_float64() * 2.0 - 1.0) * scale)
    return t^


def _z_tensor(batch: Int) raises -> Tensor:
    var z = sample_z_uniform[D](batch)
    var t = Tensor.alloc(batch * D)
    for i in range(batch * D):
        t.data[i] = z[i]
    return t^


def _row_spread(ref b: Tensor, rows: Int) -> Float64:
    """Mean pairwise distance between normalised rows of `B`.

    Normalising first is what makes this a COLLAPSE metric rather than a scale
    metric: a `B` whose rows all shrink together keeps its directions and
    should not be flagged, while a `B` whose rows converge in DIRECTION should
    be, at any magnitude.
    """
    var acc = Float64(0)
    var pairs = 0
    for i in range(rows):
        for j in range(i + 1, rows):
            var ni = Float64(0)
            var nj = Float64(0)
            for k in range(D):
                ni += Float64(b.data[i * D + k]) * Float64(b.data[i * D + k])
                nj += Float64(b.data[j * D + k]) * Float64(b.data[j * D + k])
            ni = sqrt(ni)
            nj = sqrt(nj)
            if ni < 1e-9 or nj < 1e-9:
                continue
            var d2 = Float64(0)
            for k in range(D):
                var u = Float64(b.data[i * D + k]) / ni
                var v = Float64(b.data[j * D + k]) / nj
                d2 += (u - v) * (u - v)
            acc += sqrt(d2)
            pairs += 1
    return acc / Float64(pairs) if pairs > 0 else 0.0


def test_step_runs_and_reports() raises:
    print("[1] a train step runs end to end ...")
    seed(SEED)
    var t = Trainer.make(lr=1e-3)

    var s = _rand_tensor(BATCH * OBS, 1.0)
    var a = _rand_tensor(BATCH * ACT, 1.0)
    var sn = _rand_tensor(BATCH * OBS, 1.0)
    var sp = _rand_tensor(BATCH * OBS, 1.0)
    var z = _z_tensor(BATCH)

    t.load_batch(s, a, sn, sp, z)
    var l = t.train_step()
    print("      measure", l.measure, " ortho", l.ortho, " actor", l.actor)
    print("      |F|", l.f_norm, " |B|", l.b_norm)
    assert_true(l.measure == l.measure, "measure loss is NaN")
    assert_true(l.ortho == l.ortho, "ortho loss is NaN")
    assert_true(l.actor == l.actor, "actor loss is NaN")
    assert_true(l.b_norm > 1e-9, "B output is identically zero after one step")


def test_b_does_not_collapse() raises:
    print("[2] B keeps distinct row directions over", STEPS, "steps ...")
    seed(SEED)
    var t = Trainer.make(lr=1e-3)

    var probe = _rand_tensor(BATCH * OBS, 1.0)
    var b0 = Tensor()
    t.backward_embed[BATCH](probe, b0)
    var spread0 = _row_spread(b0, BATCH)

    for _ in range(STEPS):
        var s = _rand_tensor(BATCH * OBS, 1.0)
        var a = _rand_tensor(BATCH * ACT, 1.0)
        var sn = _rand_tensor(BATCH * OBS, 1.0)
        var sp = _rand_tensor(BATCH * OBS, 1.0)
        var z = _z_tensor(BATCH)
        t.load_batch(s, a, sn, sp, z)
        _ = t.train_step()

    var b1 = Tensor()
    t.backward_embed[BATCH](probe, b1)
    var spread1 = _row_spread(b1, BATCH)
    print("      mean pairwise direction distance:", spread0, "->", spread1)
    assert_true(
        spread1 > 0.1,
        "B's rows collapsed onto one direction (spread " + String(spread1)
        + "). L_ortho is not doing its job, and the measure loss would have"
        " kept descending regardless — see this file's docstring.",
    )


def test_ortho_weight_changes_the_update() raises:
    """The third `B` vjp must actually reach `B`'s parameters.

    Two trainers, identical seed and identical data, differing only in
    `ortho_weight`. If the ortho gradient were dropped — overwritten by a later
    vjp, or zeroed mid-step — the two would produce the SAME `B`.
    """
    print("[3] ortho_weight changes B's update (the vjps accumulate) ...")

    var probe = Tensor.alloc(BATCH * OBS)
    for i in range(BATCH * OBS):
        probe.data[i] = Scalar[DT](0.21 * Float64(i % 9) - 0.7)

    var outs = List[Float64]()
    for variant in range(2):
        var w = 0.0 if variant == 0 else 1.0
        seed(SEED)
        var t = Trainer.make(lr=1e-3, ortho_weight=w)
        seed(SEED + 1)
        for _ in range(5):
            var s = _rand_tensor(BATCH * OBS, 1.0)
            var a = _rand_tensor(BATCH * ACT, 1.0)
            var sn = _rand_tensor(BATCH * OBS, 1.0)
            var sp = _rand_tensor(BATCH * OBS, 1.0)
            var z = _z_tensor(BATCH)
            t.load_batch(s, a, sn, sp, z)
            _ = t.train_step()
        var b = Tensor()
        t.backward_embed[BATCH](probe, b)
        var acc = Float64(0)
        for i in range(BATCH * D):
            acc += Float64(b.data[i]) * Float64(b.data[i])
        outs.append(sqrt(acc))

    print("      ||B(probe)||: ortho_weight=0 ->", outs[0],
          "  ortho_weight=1 ->", outs[1])
    assert_true(
        abs(outs[0] - outs[1]) > 1e-4,
        "ortho_weight had no effect on B (" + String(outs[0]) + " vs "
        + String(outs[1]) + "). The L_ortho gradient is not reaching B's"
        " parameters — most likely a vjp overwrote it or zero_grad ran"
        " mid-step.",
    )


def test_bc_weight_curbs_action_saturation() raises:
    """`bc_weight` must pull `pi_z` toward the DATA actions.

    Without it the actor maximises `F(s, pi, z)·z` over the box [-1, 1], `F` is
    near-linear in `a`, and the maximiser of a linear function on a box is a
    CORNER — so the policy goes bang-bang. Measured on walker at 200 k steps
    with `bc_weight = 0`: 95-98% of actions had |a| > 0.99 and the policy
    scored WORSE than random on all three tasks.

    The gate feeds SMALL data actions (+-0.15) and checks the BC trainer's
    policy stays closer to them. ⚠ Small data actions are the point: if the
    dataset itself were bang-bang, "near the data" and "saturated" would be the
    same place and this would measure nothing.
    """
    print("[4] bc_weight pulls pi_z toward the data actions ...")

    var probe = Tensor.alloc(BATCH * OBS)
    for i in range(BATCH * OBS):
        probe.data[i] = Scalar[DT](0.17 * Float64(i % 11) - 0.8)
    var zp = _z_tensor(BATCH)

    var sat = List[Float64]()
    for variant in range(2):
        var w = 0.0 if variant == 0 else 2.0
        seed(SEED)
        var t = Trainer.make(lr=3e-3, bc_weight=w)
        seed(SEED + 5)
        for _ in range(60):
            var s = _rand_tensor(BATCH * OBS, 1.0)
            # ⚠ SMALL data actions — see the docstring.
            var a = _rand_tensor(BATCH * ACT, 0.15)
            var sn = _rand_tensor(BATCH * OBS, 1.0)
            var sp = _rand_tensor(BATCH * OBS, 1.0)
            var z = _z_tensor(BATCH)
            t.load_batch(s, a, sn, sp, z)
            _ = t.train_step(want_loss=True)
        var out = Tensor()
        t.act[BATCH](probe, zp, out)
        var acc = Float64(0)
        for i in range(BATCH * ACT):
            acc += abs(Float64(out.data[i]))
        sat.append(acc / Float64(BATCH * ACT))

    print("      mean|a|:  bc_weight=0 ->", sat[0], "  bc_weight=2 ->", sat[1])
    assert_true(
        sat[1] < sat[0],
        "bc_weight did not reduce |action| (" + String(sat[0]) + " -> "
        + String(sat[1]) + "). The BC gradient is not reaching the actor, and"
        " the policy will go bang-bang on any offline dataset.",
    )


def test_bc_mask_selects_rows() raises:
    """[4b] `fill_bc_mask(0)` must switch BC off on every row while leaving
    the adaptive scale on, and `fill_bc_mask(BATCH)` must be the full pull.
    Data actions are a CONSTANT +0.9 so the pull's direction is unmistakable,
    and the check is on the SIGNED mean action: the value term alone drives
    this actor negative on the probe (~-0.5), BC toward +0.9 must drive it
    positive. (|a| would not do: it conflates the two directions — the first
    draft of this gate failed for exactly that reason.)"""
    print("[4b] bc_mask: all-zero mask = no clone, all-one mask = full clone ...")
    var probe = Tensor.alloc(BATCH * OBS)
    for i in range(BATCH * OBS):
        probe.data[i] = Scalar[DT](0.17 * Float64(i % 11) - 0.8)
    var zp = _z_tensor(BATCH)
    var res = List[Float64]()
    for variant in range(2):
        seed(SEED)
        var t = Trainer.make(lr=3e-3, bc_weight=2.0)
        t.fill_bc_mask(0 if variant == 0 else BATCH)
        seed(SEED + 5)
        for _ in range(60):
            var s = _rand_tensor(BATCH * OBS, 1.0)
            var a = Tensor.alloc(BATCH * ACT)
            for i in range(BATCH * ACT):
                a.data[i] = Scalar[DT](0.9)
            var sn = _rand_tensor(BATCH * OBS, 1.0)
            var sp = _rand_tensor(BATCH * OBS, 1.0)
            var z = _z_tensor(BATCH)
            t.load_batch(s, a, sn, sp, z)
            _ = t.train_step(want_loss=False)
        var out = Tensor()
        t.act[BATCH](probe, zp, out)
        var acc = Float64(0)
        for i in range(BATCH * ACT):
            acc += Float64(out.data[i])
        res.append(acc / Float64(BATCH * ACT))
    print("      mean(a):  mask all-zero ->", res[0], "  mask all-one ->", res[1])
    assert_true(
        res[1] > res[0] + 0.3,
        "the BC mask did not select rows (all-one " + String(res[1])
        + " vs all-zero " + String(res[0]) + ")",
    )


def test_actor_update_independent_of_want_loss() raises:
    """[5] The actor update must be IDENTICAL with and without `want_loss`.

    ⚠⚠ It was not. TD3+BC's adaptive scale read `abs(loss) if want_loss else
    1.0`, so the value term was normalised on one step in `LOG_EVERY` and left
    raw on the rest — and under CUDA-graph capture (`want_loss=False` always)
    it was never normalised at all. `||F||` then grew 2.65x over 214 k steps
    while `bc_weight` stayed fixed, the value term outgrew BC, and `pi_z` went
    bang-bang: 70-79% saturated, WORSE than random on all three walker tasks.

    A loss flag silently changing the OBJECTIVE is the worst shape this can
    take, because every diagnostic that would reveal it is computed on exactly
    the steps where the bug is absent.
    """
    print("[5] actor update is independent of want_loss ...")
    seed(SEED)
    var a = Trainer.make(lr=1e-2, bc_weight=1.0)
    seed(SEED)
    var b = Trainer.make(lr=1e-2, bc_weight=1.0)

    var s_ = _rand_tensor(BATCH * OBS, 1.0)
    var ac = _rand_tensor(BATCH * ACT, 1.0)
    var sn = _rand_tensor(BATCH * OBS, 1.0)
    var sp = _rand_tensor(BATCH * OBS, 1.0)
    var z = _z_tensor(BATCH)

    a.load_batch(s_, ac, sn, sp, z)
    b.load_batch(s_, ac, sn, sp, z)
    # ⚠ Reseed before EACH step. `gaussian_t` draws the target-smoothing noise
    # from the GLOBAL host RNG on the cpu target, so running `a` then `b`
    # otherwise hands them different noise and the comparison measures the RNG
    # rather than `want_loss`. That is exactly how this test first "failed"
    # (0.055) against a fix that was already correct.
    seed(SEED + 77)
    _ = a.train_step(want_loss=True)
    seed(SEED + 77)
    _ = b.train_step(want_loss=False)

    var probe = _rand_tensor(BATCH * OBS, 1.0)
    var zp = _z_tensor(BATCH)
    var pa = Tensor()
    var pb = Tensor()
    a.act[BATCH](probe, zp, pa)
    b.act[BATCH](probe, zp, pb)

    var worst = Float64(0)
    for i in range(BATCH * ACT):
        var e = abs(Float64(pa.data[i]) - Float64(pb.data[i]))
        if e > worst:
            worst = e
    print("      worst |pi_z(want_loss=True) - pi_z(want_loss=False)| =", worst)
    assert_true(
        worst < 1e-6,
        "the actor update DIFFERS by " + String(worst) + " depending on"
        " want_loss — the logging cadence is changing the objective",
    )
    print("      OK")


def test_act_l2_curbs_action_saturation() raises:
    """[6] `act_l2_weight` must pull `pi_z` toward ZERO — the online stand-in
    for BC, where the replay's own actions are already saturated and BC
    toward them would be circular. Same construction as [4] with the data
    actions LARGE (+-1), so that BC could not have produced the effect."""
    print("[6] act_l2_weight pulls pi_z toward zero ...")
    var probe = Tensor.alloc(BATCH * OBS)
    for i in range(BATCH * OBS):
        probe.data[i] = Scalar[DT](0.17 * Float64(i % 11) - 0.8)
    var zp = _z_tensor(BATCH)
    var sat = List[Float64]()
    for variant in range(2):
        var w = 0.0 if variant == 0 else 2.0
        seed(SEED)
        var t = Trainer.make(lr=3e-3, act_l2_weight=w)
        seed(SEED + 9)
        for _ in range(60):
            var s = _rand_tensor(BATCH * OBS, 1.0)
            var a = _rand_tensor(BATCH * ACT, 1.0)
            var sn = _rand_tensor(BATCH * OBS, 1.0)
            var sp = _rand_tensor(BATCH * OBS, 1.0)
            var z = _z_tensor(BATCH)
            t.load_batch(s, a, sn, sp, z)
            _ = t.train_step(want_loss=False)
        var out = Tensor()
        t.act[BATCH](probe, zp, out)
        var acc = Float64(0)
        for i in range(BATCH * ACT):
            acc += abs(Float64(out.data[i]))
        sat.append(acc / Float64(BATCH * ACT))
    print("      mean|a|:  act_l2=0 ->", sat[0], "  act_l2=2 ->", sat[1])
    assert_true(
        sat[1] < sat[0],
        "act_l2_weight did not reduce |action| (" + String(sat[0]) + " -> "
        + String(sat[1]) + ") — the penalty gradient is not reaching the actor",
    )


def test_act_l2_margin_leaves_the_band_alone() raises:
    """[6b] The HINGED penalty must leave the interior to the value term.
    Three runs on identical data: no penalty, hinge at 0.8, hinge at 0.999.
      * a margin no action crosses must be INVISIBLE: the 0.999 run must
        match the no-penalty run. This is the check that failed silently
        before — the adaptive scale had been switched on by the penalty and
        held mean|a| at 0.27 against 0.73, with the hinge itself doing
        nothing (§18.7.1).
      * the CROSSED hinge must not raise mean|a| and must leave fewer actions
        above the margin than no penalty.

    ⚠ The crossed arm used margin 0.8 and counted actions above 0.9. The
    actor's outputs sit at mean|a| ~= 0.54, so exactly ONE element of
    `BATCH * ACT` was ever above 0.9 — the leg passed or failed on a single
    element flipping, and it flipped the moment `L_ortho` moved to the
    reference's scale (docs §12.28) and shifted the trajectory. The hinge code
    was not involved. The margin is now 0.3, BELOW where the actions sit, so
    the penalty demonstrably engages, and `above[0]` is asserted non-trivial
    so the leg can still fail.
    """
    comptime HINGE_M = 0.3
    print("[6b] act_l2_margin: an uncrossed margin is invisible; a crossed one caps ...")
    var probe = Tensor.alloc(BATCH * OBS)
    for i in range(BATCH * OBS):
        probe.data[i] = Scalar[DT](0.17 * Float64(i % 11) - 0.8)
    var zp = _z_tensor(BATCH)
    var means = List[Float64]()
    var above = List[Int]()
    for variant in range(3):
        var w = 0.0 if variant == 0 else 2.0
        var m = 0.999 if variant == 1 else HINGE_M
        seed(SEED)
        var t = Trainer.make(lr=3e-3, act_l2_weight=w, act_l2_margin=m)
        seed(SEED + 9)
        for _ in range(60):
            var s = _rand_tensor(BATCH * OBS, 1.0)
            var a = _rand_tensor(BATCH * ACT, 1.0)
            var sn = _rand_tensor(BATCH * OBS, 1.0)
            var sp = _rand_tensor(BATCH * OBS, 1.0)
            var z = _z_tensor(BATCH)
            t.load_batch(s, a, sn, sp, z)
            _ = t.train_step(want_loss=False)
        var out = Tensor()
        t.act[BATCH](probe, zp, out)
        var acc = Float64(0)
        var n_above = 0
        for i in range(BATCH * ACT):
            var v = abs(Float64(out.data[i]))
            acc += v
            if v > HINGE_M:
                n_above += 1
        means.append(acc / Float64(BATCH * ACT))
        above.append(n_above)
    print("      mean|a|:  none ->", means[0], "  hinge@0.999 ->", means[1],
          "  hinge@", HINGE_M, "->", means[2])
    print("      |a|>", HINGE_M, ":  none ->", above[0], "  hinge@0.999 ->",
          above[1], "  hinge@", HINGE_M, "->", above[2])
    assert_true(
        above[0] >= (BATCH * ACT) // 10,
        "vacuous: only " + String(above[0]) + " of " + String(BATCH * ACT)
        + " actions clear the margin without a penalty, so 'the hinge caps'"
        " is decided by a handful of elements and flips on any trajectory"
        " change",
    )
    assert_true(
        abs(means[1] - means[0]) < 1e-6,
        "a margin no action crosses CHANGED the actor (" + String(means[0])
        + " vs " + String(means[1]) + ") — something other than the hinge is"
        " reacting to act_l2_weight > 0",
    )
    assert_true(means[2] <= means[0] + 1e-6, "the crossed hinge raised mean|a|")
    assert_true(
        above[2] <= above[0],
        "the crossed hinge left more actions above the margin",
    )


def main() raises:
    test_step_runs_and_reports()
    test_b_does_not_collapse()
    test_ortho_weight_changes_the_update()
    test_bc_weight_curbs_action_saturation()
    test_bc_mask_selects_rows()
    test_actor_update_independent_of_want_loss()
    test_act_l2_curbs_action_saturation()
    test_act_l2_margin_leaves_the_band_alone()
    print("\n[PASS] FB trainer smoke gate")
