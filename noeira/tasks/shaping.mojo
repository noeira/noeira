"""THE SHAPED REWARD'S PARAMETERS, PER LANE — the host half.

    var sw = shaping_words(w_goal, w_reach, goal_margin, reach_margin)
    for j in range(len(sw)):
        meta[env * METADATA_SIZE + META_IDX_SHAPE_W_GOAL + j] = sw[j]

## ⚠⚠ WHY PER LANE AND NOT PER RUN

They lived in `curriculum`, which is `[1, MODEL_CURRICULUM_SIZE]` — one row for
the whole batch. That is right for the region table, because a region belongs
to the FAMILY, and wrong for shaping, because what a weight is WORTH depends on
the task's own distance scale. Measured on `so101_tabletop` at identical
weights (1.0 / 0.5) and identical margins (0.10 / 0.20):

    task     op      goal dist   goal term   total reward
    gather   Near        0.139       0.011          0.292
    lift     Above       0.030       0.811          1.171
    settle   On          0.000       1.000          1.360

A 4.7x spread in the reward and 91x in the goal term, from the same four
numbers. `Near`'s distance is a separation between two props that starts near
0.14 m; `Above`'s is a z-shortfall of 0.03 m; `On`'s is zero at reset by
construction. One margin cannot serve all three, and a two-task batch under one
would hand its lanes a bimodal reward — which on this family is how a critic
gets destabilised.

## ⚠ ZERO IS "NO SHAPING", WHICH IS WHAT AN UNTOUCHED `meta` HOLDS

`Data` uploads a zero-filled `meta`, so a driver that never writes these gets
the SPARSE reward — the goal bit and nothing else — instead of a shaped one
with meaningless parameters. Same bias-toward-safe as the init-region words.
"""

from std.math import sqrt

from noeira.physics3d.gpu.constants import (
    META_IDX_SHAPE_W_GOAL, META_IDX_SHAPE_W_REACH,
    META_IDX_GOAL_MARGIN, META_IDX_REACH_MARGIN,
)


comptime SHAPING_WORDS: Int = 4
"""How many `meta` words `shaping_words` returns, starting at
`META_IDX_SHAPE_W_GOAL`. ⚠ THE FOUR ARE CONTIGUOUS AND IN THIS ORDER so a
caller can write them with one loop; `constants.mojo` is where that adjacency
is declared and this only restates the count."""


def shaping_words(
    w_goal: Float64, w_reach: Float64,
    goal_margin: Float64, reach_margin: Float64,
) raises -> List[Float64]:
    """`[w_goal, w_reach, goal_margin, reach_margin]`, validated.

    ⚠ NEGATIVE WEIGHTS ARE REFUSED. They multiply a `tolerance` that is LARGER
    nearer the goal, so a negative one pays the policy to move away — and it
    would train toward exactly that, with a perfectly healthy critic.

    ⚠⚠ A NONZERO WEIGHT WITH A ZERO MARGIN IS REFUSED, and it is the trap this
    function exists for. `tolerance` with `margin == 0` is a HARD INDICATOR: 1
    inside the bounds, 0 outside. So the term silently becomes SPARSE — no
    gradient anywhere — which is the one thing the shaping was added to avoid,
    and it reads in a log as a shaped run that would not learn.

    ⚠ A ZERO WEIGHT WITH ANY MARGIN IS FINE: the term is off, which is how a
    caller turns one half of the shaping off deliberately.
    """
    if w_goal < 0.0 or w_reach < 0.0:
        raise Error(
            "tasks: negative shaping weight (" + String(w_goal) + ", "
            + String(w_reach) + "). These multiply a `tolerance` that REWARDS"
            " proximity, so a negative one pays the policy to move away from"
            " the goal — and it would learn that."
        )
    if w_goal > 0.0 and goal_margin <= 0.0:
        raise Error(
            "tasks: goal weight " + String(w_goal) + " with margin "
            + String(goal_margin) + ". `tolerance` with a zero margin is a"
            " HARD INDICATOR — 1 inside the bounds, 0 outside — so the term"
            " has no gradient anywhere and the run is sparse while looking"
            " shaped."
        )
    if w_reach > 0.0 and reach_margin <= 0.0:
        raise Error(
            "tasks: reach weight " + String(w_reach) + " with margin "
            + String(reach_margin) + ". See the goal-margin error above:"
            " a zero margin makes the term a hard indicator."
        )
    var out = List[Float64]()
    out.append(w_goal)
    out.append(w_reach)
    out.append(goal_margin)
    out.append(reach_margin)
    return out^


comptime REWARD_MODE_WORDS: Int = 2
"""How many `meta` words `reward_mode_words` returns, starting at
`META_IDX_REWARD_MODE`: the mode and the success bonus. (The block's other
two words, `META_IDX_PHI_PREV` / `META_IDX_EPISODE_FLAGS`, are episode state the
hook and the reset own — a driver never writes them.)"""


def reward_mode_words(
    potential: Bool, success_bonus: Float64 = 0.0
) raises -> List[Float64]:
    """`[mode, success_bonus]` for `meta[META_IDX_REWARD_MODE ..]`.

    `potential=False` is the LEGACY reward (every shaped term a raw per-step
    value) — also what an untouched `meta` gives. `potential=True` pays the
    change of the staged potential and the full budget while the goal holds
    (`family_config.compute_reward_and_done_gpu`, the potential-based mode).

    ⚠ A SUCCESS BONUS WITHOUT THE POTENTIAL MODE IS REFUSED: the legacy hook
    never reads the word, so the bonus would be silently absent from a run
    that asked for it.
    """
    if success_bonus < 0.0:
        raise Error("tasks: negative success bonus " + String(success_bonus))
    if success_bonus > 0.0 and not potential:
        raise Error(
            "tasks: a success bonus (" + String(success_bonus) + ") needs the"
            " potential-based reward mode; the legacy hook never reads it."
        )
    var out = List[Float64]()
    out.append(1.0 if potential else 0.0)
    out.append(success_bonus)
    return out^


@always_inline
def near_transport_shortfall[
    DTYPE: DType
](
    ex: Scalar[DTYPE], ey: Scalar[DTYPE], ez: Scalar[DTYPE],
    radius: Scalar[DTYPE],
) -> Scalar[DTYPE]:
    """The `Near(a, b, radius)` shortfall the POTENTIAL-BASED reward measures:
    `max(|horizontal|, |vertical|) - radius`, floored at 0.

    ⚠⚠ NOT THE 3D DISTANCE, BECAUSE THE 3D DISTANCE PAYS NEGATIVE FOR THE LIFT
    THE TASK NEEDS. `so101_tower_cube_in_bowl` is `Near(brick, bowl, 0.045)`:
    with the 3D shortfall, raising the brick off the desk moves it AWAY from
    the bowl's origin, so the goal potential DROPS on the one move that can
    get the brick over the rim, and the cheap optimum is to push it along the
    desk against the bowl (0.078 away at the rim). PPO sat at ~2 % success for
    20M steps (26 Sep). so101-nexus's reward notes name this trap ("an undo
    factor pays negative on the lift the task requires") and fix it with a
    Chebyshev transport distance.

    Along the ideal trajectory it never increases: lifting is FREE while the
    horizontal gap exceeds the height gap, carrying reduces it, and lowering
    into the bowl reduces it. It is <= the 3D shortfall, so it reaches 0 no
    later than the predicate. The SUCCESS test is still the 3D `Near`.
    """
    var h = sqrt(ex * ex + ey * ey)
    var v = ez if ez > Scalar[DTYPE](0) else -ez
    var m = h if h > v else v
    var d = m - radius
    return d if d > Scalar[DTYPE](0) else Scalar[DTYPE](0)


comptime OPTIMAL_MARGIN_PER_METRE: Float64 = 1.5174271293851465
"""`margin / shortfall` at which a `tolerance` term's gradient is greatest.

`k / sqrt(2)` where `k = sqrt(-2 ln(DEFAULT_VALUE_AT_MARGIN))`. ⚠ IT IS TIED
TO `DEFAULT_VALUE_AT_MARGIN`; a term built at a different `value_at_margin`
has a different constant, which is why `optimal_margin` is the only place it
is written.
"""


def optimal_margin(shortfall: Float64) raises -> Float64:
    """The margin at which a `tolerance` term starting at `shortfall` has the
    steepest gradient — and, at the same time, 0.632 of headroom.

    ## ⚠⚠ THE GRADIENT IS NOT MONOTONE IN THE MARGIN

    Both ends are flat: a narrow band is flat everywhere outside itself, a
    wide one is flat near zero. Measured on `lift`'s 0.030 m z-shortfall
    against the real `tolerance`:

        margin   term@reset   gradient/m
        0.020        0.006          1.9      too narrow — dead band
        0.046        0.376         24.5      the peak
        0.050        0.437         24.1
        0.100        0.813         11.2      too wide — mostly free
        0.150        0.912          5.6

    Both wrong margins shipped. 0.02 gave a run whose return never moved;
    0.10 gave a run that bought +0.034 reward per step over doing nothing and
    left the brick on the table, because 0.81 of the goal term was paid
    before the policy acted.

    ⚠ THE ARGUMENT IS THE SHORTFALL PAST THE RADIUS, not the raw distance.
    `tolerance` is 1 inside `[lower, upper]`, so a term with a nonzero radius
    has less to close than its distance suggests.

    ## ⚠⚠ IT HAS LOST EVERY TIME IT HAS BEEN TESTED. DO NOT FOLLOW IT BLIND.

    The number below is the gradient peak AT THE RESET DISTANCE, and that has
    turned out not to predict learning. Measured on `so101_gather_bricks`,
    1M steps, same seed, same update budget, ONLY the margins differing —
    128-episode greedy evaluation against a baseline of 0 in 256:

        goal margin  x goal dist  success (128 ep)  gradient at reset
        0.070        0.50x        0.070  ( 9/128)    0.29/m
        0.100        0.72x        0.203  (26/128)    0.73/m   <- the peak
        0.211        1.52x        0.047  ( 6/128)    5.25/m   <- THIS FUNCTION

    0.10 beats 0.07 at Fisher one-sided p = 1.6e-3 and 0.211 at p = 1.1e-4;
    0.07 and 0.211 are indistinguishable (p = 0.30). So the success curve has
    an INTERIOR maximum around 0.72x the goal distance and falls away on both
    sides, while the gradient this function maximises rises monotonically
    across the whole range — the two are not the same shape, and the margin
    with ONE SEVENTH the gradient of the peak is 4.3x better. `so101_lift_brick` lost the other way — its recommendation
    (0.058) was the only lift run that learned nothing at all, while 0.10 and
    0.211 both moved — though lift's goal is unreachable for other reasons,
    so that leg is weaker.

    The likely reason: learning is not decided at the reset distance. A
    narrow band keeps the term near zero until the predicate is nearly
    satisfied and then rises sharply, so the return discriminates ACHIEVING
    the goal from hovering near it. A wide band pays generously for being
    roughly in the area, and a policy can collect most of it without ever
    closing. Maximising the slope at reset optimises the wrong end of the
    trajectory.

    ⚠ SO TREAT THIS AS ONE COORDINATE, not the answer: it says where the
    reset-distance gradient peaks, which is worth knowing and is not worth
    obeying. The measured starting point on this family is **0.7x the goal
    distance**, and the only honest way to place a margin is still to run two
    and compare — the curve above cost three 16-minute runs and is one task's.
    """
    if shortfall <= 0.0:
        raise Error(
            "tasks: optimal_margin(" + String(shortfall) + "). The term is"
            " already inside its radius at reset, so there is no distance to"
            " close and no margin makes it train."
        )
    return OPTIMAL_MARGIN_PER_METRE * shortfall
