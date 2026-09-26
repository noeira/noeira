"""Shared primal-solver fragments over `Scratch` working sets (Stage-S
refactor). These are the reusable leaf computations extracted VERBATIM from
the inlined fields-Newton kernel (`solver/newton_solve.mojo`).

All helpers are `@always_inline` and operate on the per-env `Scratch`
working set the primal solvers already build (Je/De/bias_e/qacc/...), so
inlining them at the Newton call sites is codegen- (and thus bit-) identical
to the previous inline code — the Newton golden gates re-validate this after
each extraction.

PYRAMIDAL cone only (the pyramidal primal path is what the Walker2D Newton
gates exercise; the elliptic path is inlined in `newton_solve.mojo`).

ROW KINDS. The row list is no longer homogeneous. Contact edges and joint
limits are ONE-SIDED (`SROW_LIMIT`: inactive once jar >= 0), while
dry-friction dof rows are BOX-clamped to +-frictionloss (`SROW_FRICTION`),
which has a LINEAR regime where the force is constant and the row contributes
NOTHING to the Hessian. Classifying with `force > 0` — as this file did while
every row was one-sided — silently mis-handles a box row: a saturated negative
row has force > 0 and would wrongly add curvature. Callers therefore read the
per-row `state_e` that `pyramidal_edge_forces` now writes."""

from std.math import sqrt, abs

from ..fields.scratch import Scratch
from ..constraints.scalar_rows import (
    scalar_row_state,
    scalar_row_force,
    scalar_row_cost,
    SROW_QUADRATIC,
)


@always_inline
def pyramidal_edge_forces[
    DTYPE: DType,
    E_CAP: Int,
    V_CAP: Int,
    N_CAP: Int = 1,
    IX_CAP: Int = 1,
    SPARSE: Bool = False,
](
    num_edges: Int,
    Je: Scratch[Scalar[DTYPE], E_CAP * V_CAP],
    De: Scratch[Scalar[DTYPE], E_CAP],
    bias_e: Scratch[Scalar[DTYPE], E_CAP],
    kind_e: Scratch[Int, E_CAP],
    R_e: Scratch[Scalar[DTYPE], E_CAP],
    floss_e: Scratch[Scalar[DTYPE], E_CAP],
    qacc: Scratch[Scalar[DTYPE], V_CAP],
    mut jar: Scratch[Scalar[DTYPE], E_CAP],
    mut force: Scratch[Scalar[DTYPE], E_CAP],
    mut state_e: Scratch[Int, E_CAP],
    mut qfrc: Scratch[Scalar[DTYPE], V_CAP],
    nv: Int,
    je_n: Scratch[Int, N_CAP],
    je_ix: Scratch[Int, IX_CAP],
):
    """Primal row forces given the current qacc.

        qfrc = 0
        for each row e:
            jar[e]   = bias_e[e] + Je[e]·qacc
            state[e] = branch(kind_e[e], jar[e], R_e[e], floss_e[e])
            force[e] = f(state[e])       one-sided or box, see scalar_rows
            qfrc    += Je[e] * force[e]

    Je is row-major [num_edges, nv], STRIDE `nv` — the live dof count, not
    `V_CAP`. The array is sized `E_CAP * V_CAP`, and on the static leg the cap
    and the stride are the same integer, which is why a mix-up here survives
    every gate in the tree; see `tests/physics3d/test_cholesky_both_legs.mojo`.
    Writes jar/force/state_e (per-row) and qfrc (per-dof).

    ⚠ `SPARSE`: walk only each row's NONZERO dofs, listed in
    `je_ix[e*nv + 0 .. je_n[e])`. A contact row is nonzero on the two bodies'
    ancestor chains only — a dozen of humanoid_CMU's 62 dofs — and a limit or
    friction row on ONE. Every term this skips has `Je == 0` exactly, and
    adding an exact zero product changes no accumulator, so the sparse walk
    is BIT-IDENTICAL to the dense one; `PERFORMANCE.md` §13 measured the
    dense walks at most of a 62-dof step. Off (the default), the lists are
    unused and the loops are the dense originals, byte for byte."""
    for i in range(nv):
        qfrc[i] = Scalar[DTYPE](0)
    for e_idx in range(num_edges):
        jar[e_idx] = bias_e[e_idx]
        comptime if SPARSE:
            for a in range(je_n[e_idx]):
                var i = je_ix[e_idx * nv + a]
                jar[e_idx] += Je[e_idx * nv + i] * qacc[i]
        else:
            for i in range(nv):
                jar[e_idx] += Je[e_idx * nv + i] * qacc[i]
        var st = scalar_row_state[DTYPE](
            kind_e[e_idx], jar[e_idx], R_e[e_idx], floss_e[e_idx]
        )
        state_e[e_idx] = st
        force[e_idx] = scalar_row_force[DTYPE](
            st, jar[e_idx], De[e_idx], floss_e[e_idx]
        )
        comptime if SPARSE:
            for a in range(je_n[e_idx]):
                var i = je_ix[e_idx * nv + a]
                qfrc[i] += Je[e_idx * nv + i] * force[e_idx]
        else:
            for i in range(nv):
                qfrc[i] += Je[e_idx * nv + i] * force[e_idx]


# ⚠⚠ MuJoCo'S LINE-SEARCH THRESHOLD IS BELOW FLOAT32'S RESOLUTION. `gtol` is
# `tolerance * ls_tolerance * snorm / scale`; at the float32 tolerance floor
# (1e-6, `NEWTON_TOL_GPU`) that is ~1e-8 of the slope, while a `d0` summed in
# float32 carries rounding near 1e-7 of its terms. So `|d0| < gtol` is often
# unreachable and the search brackets until the interval stops moving or the
# budget runs out.
#
# MEASURED 2026-09-26, so101_tower (elliptic, impratio 10), 1024 lanes, RTX
# 5090: 31.2 evaluations per solve at 2.0 iterations (15.6 per iteration)
# against CPU MuJoCo 3.12's 3.2 per iteration on the same protocol. With the
# threshold floored at `LS_SLOPE_DROP_F32 * |d0(0)|`: 9.4 per solve, iterations
# unchanged (2.01), the blocked Newton launch 3778 -> 2605 us and the control
# step 99.2 -> 79.4 ms. 1e-4 bought 1.5% more; 1e-5 is the conservative one.
#
# ⚠ RELATIVE TO THE SLOPE AT alpha=0, NOT ABSOLUTE. MuJoCo Warp floors the same
# threshold at an absolute 1e-6 (`solver.py:1023`), but it forms `snorm * scale`
# where MuJoCo and this tree form `snorm / scale`, so the number does not
# transplant: on the tower it cut only 8% and moved the dynamics (collision
# +13% at the same contact count). A slope ratio has no units to get wrong.
#
# ⚠ FLOAT64 IS UNTOUCHED — the same scoping as `NEWTON_TOL_GPU`, so every
# MuJoCo-parity gate (all float64) is bit-identical across this.
comptime LS_SLOPE_DROP_F32: Float64 = 1e-5


@always_inline
def ls_gtol_dtype_floor[
    DTYPE: DType
](gtol: Scalar[DTYPE], d0_at_zero: Scalar[DTYPE]) -> Scalar[DTYPE]:
    """The line search's derivative threshold, floored where the dtype cannot
    resolve MuJoCo's. Call it once, after the alpha=0 `PrimalEval`, at every
    Newton line search — they must not disagree."""
    comptime if DTYPE == DType.float64:
        return gtol
    else:
        return max(gtol, Scalar[DTYPE](LS_SLOPE_DROP_F32) * abs(d0_at_zero))


@always_inline
def pyramidal_linesearch[
    DTYPE: DType,
    E_CAP: Int,
    V_CAP: Int,
    LINESEARCH_ITER: Int,
    PRIMAL_MINVAL: Float64,
    N_CAP: Int = 1,
    IX_CAP: Int = 1,
    SPARSE: Bool = False,
](
    num_edges: Int,
    Je: Scratch[Scalar[DTYPE], E_CAP * V_CAP],
    De: Scratch[Scalar[DTYPE], E_CAP],
    kind_e: Scratch[Int, E_CAP],
    R_e: Scratch[Scalar[DTYPE], E_CAP],
    floss_e: Scratch[Scalar[DTYPE], E_CAP],
    search: Scratch[Scalar[DTYPE], V_CAP],
    Mv: Scratch[Scalar[DTYPE], V_CAP],
    Ma: Scratch[Scalar[DTYPE], V_CAP],
    f_smooth: Scratch[Scalar[DTYPE], V_CAP],
    qacc: Scratch[Scalar[DTYPE], V_CAP],
    qacc_smooth: Scratch[Scalar[DTYPE], V_CAP],
    jar: Scratch[Scalar[DTYPE], E_CAP],
    nv: Int,
    # Row sparsity lists, read only under `SPARSE` — see
    # `pyramidal_edge_forces`.
    je_n: Scratch[Int, N_CAP],
    je_ix: Scratch[Int, IX_CAP],
    # The number of `peval` calls this search made — the line search's unit
    # of work, and the count the blocked kernel reports as `[lseval]`. The
    # caller zeroes it; MuJoCo's `neval` is the same quantity per iteration.
    mut lsiter: Int,
    # `<option ls_iterations>`. ⚠ THE COMPTIME `LINESEARCH_ITER` IS THE CEILING
    # A `range()` NEEDS, NOT THE BUDGET — a model asking for fewer iterations
    # (apollo asks for 10, so101 for 20) must get them. A non-positive value
    # means "use the ceiling".
    max_ls: Int = -1,
    # `opt.tolerance * opt.ls_tolerance / scale`. `PrimalSearch` converges on
    # `|deriv[0]| < tolerance * snorm / scale` (engine_solver.c:1711) where its
    # `tolerance` argument is already the PRODUCT of the two options
    # (engine_solver.c:2236). `snorm` is `|search|`, computed here because
    # this is where the direction lives. A non-positive value falls back to
    # the old single-step behaviour — no caller should pass one.
    gtol_scale: Scalar[DTYPE] = Scalar[DTYPE](0),
) -> Scalar[DTYPE]:
    """`PrimalSearch` for the pyramidal primal cost (engine_solver.c:1692).

    Returns the step length `alpha` minimising the Gauss + row-constraint cost
    along `search`, or 0 when there is no improvement to be had.

    ⚠⚠ THIS ITERATES; IT USED TO TAKE ONE STEP. The old body computed
    `alpha = -d1/d2` from the derivatives AT alpha=0 and then halved until the
    cost stopped rising. That is a Newton step on the line plus a backtrack,
    and on a PIECEWISE-quadratic line — rows cross their zone boundaries
    between alpha=0 and the minimum, which is the whole reason a line search
    exists here — it lands SHORT and halving only guards the other direction.

    ⚠ THE COLD START HID IT FOR AS LONG AS WE HAD ONE. Starting every solve at
    `qacc_smooth` leaves the iterate far enough from the optimum that the NEXT
    Newton iteration absorbs the shortfall. `qacc_warmstart` puts the iterate
    where there is no next iteration, and the engine's answer became a
    function of where it started while MuJoCo's is not: on
    `test_frictionless_contact_pyramidal` MuJoCo reaches `gradient` 1e-34 in
    ONE iteration and is start-independent to 1.1e-16, where this plateaued at
    1.07e-08 with `alpha` collapsing to 5e-03.

    THE THREE PHASES ARE MuJoCo'S, and are the same three the ELLIPTIC leg in
    `newton_solve.mojo` already runs — that leg was ported faithfully and this
    one was not, which is why the two cones stopped agreeing:

      1. one Newton step on the line from alpha=0; accept on `|d0| < gtol`;
      2. a ONE-SIDED Newton search in the descent direction until the
         derivative changes sign (a bracket) or the budget runs out;
      3. a BRACKETED search over three candidates — `p1next`, `p2next` and the
         bracket midpoint — accepting the CHEAPEST of those under `gtol`.

    ⚠⚠ PHASE 3 IS NOT A BISECTION, and believing it was cost apollo four
    orders. This body used to bisect the bracket and stop on `|d0| < gtol`,
    on the reasoning that both schemes converge to the same root of `d0` and
    both stop at the same tolerance, so they agree to within it. They do not.
    MuJoCo's phase 3 keeps a NEWTON step off each bracket end (`p1next`,
    `p2next`) alongside the midpoint, and on apollo's second Newton iteration
    the very first `p1next` lands the root — `d0` = -3.6e-11 against a `gtol`
    of 7.3e-05 — and returns in ONE evaluation. Bisection starts from the
    midpoint, which is at the WRONG END of a bracket whose root sits hard
    against the low edge, and spends the entire `ls_iterations` budget
    halving from 0.875 down to 0.7515 without ever getting under `gtol`; it
    then returns the stale endpoint. The returned alpha is not "the same root
    within tolerance" — it does not satisfy the criterion AT ALL:

        MuJoCo  alpha 0.75053713815071321  (5 evaluations, converged)
        bisect  alpha 0.7504804293368179   (10 evaluations, budget exhausted)

    That is 7.6e-05 of relative alpha, and it is the whole of apollo's
    residual against MuJoCo — `|d qpos|` 1.7e-06 after two Newton iterations
    where one iteration agrees to 4.4e-16.

    ⚠ `p2update` (engine_solver.c:1787, LSresult 6) is NOT ported: the
    reference initialises it to 1 and only ever assigns 1, so its branch is
    unreachable in 3.10.0. Ported code that cannot run is a second thing to
    keep in step for no behaviour.
    """
    comptime ZERO = Scalar[DTYPE](0)
    comptime MINVAL = Scalar[DTYPE](PRIMAL_MINVAL)

    var budget = LINESEARCH_ITER
    if max_ls > 0 and max_ls < budget:
        budget = max_ls

    # Precompute Jv_e = Je · search for each row.
    var Jv_e = Scratch[Scalar[DTYPE], E_CAP](
        num_edges, uninitialized=Scalar[DTYPE](0)
    )
    for e_idx in range(num_edges):
        Jv_e[e_idx] = Scalar[DTYPE](0)
        comptime if SPARSE:
            for a in range(je_n[e_idx]):
                var i = je_ix[e_idx * nv + a]
                Jv_e[e_idx] += Je[e_idx * nv + i] * search[i]
        else:
            for i in range(nv):
                Jv_e[e_idx] += Je[e_idx * nv + i] * search[i]

    var gauss_a: Scalar[DTYPE] = 0
    var gauss_b: Scalar[DTYPE] = 0
    var snorm_sq: Scalar[DTYPE] = 0
    for i in range(nv):
        gauss_a += Mv[i] * search[i]
        gauss_b += (Ma[i] - f_smooth[i]) * search[i]
        snorm_sq += search[i] * search[i]

    # `PrimalSearch` bails on a degenerate direction before anything else
    # (engine_solver.c:1705, LSresult 1).
    var snorm = sqrt(snorm_sq)
    if snorm < MINVAL:
        return ZERO

    var gtol = gtol_scale * snorm

    # ── `PrimalEval` (engine_solver.c:1511) ──────────────────────────────
    # The SHIFTED line cost `cost(alpha) - cost(0)` and its two derivatives,
    # in ONE pass. ⚠ THE COST IS CARRIED AT EVERY POINT, not computed on
    # demand: phase 3 selects among its candidates by COST among those under
    # `gtol`, and the two bracket ends are compared by cost on the way out.
    #
    # `d(cost)/dalpha = -f*Jv` in EVERY state; the second derivative is
    # `D*Jv^2` only where the row is quadratic. Rows are RE-CLASSIFIED at the
    # trial point — a step can move a row across a zone boundary, and not
    # re-classifying is what makes a line search a fixed quadratic model.
    #
    # The Gauss term's shifted cost is `0.5*a^2*(Mv·s) + a*((Ma - f)·s)`. The
    # cross term `Mv·(qacc - qacc_smooth)` equals `(Ma - f_smooth)·s` because
    # M is symmetric and `f_smooth = M*qacc_smooth`, which is why MuJoCo
    # stores two coefficients rather than three (`quadGauss[1..2]`).
    #
    # `it` is `ctx->LSiter`, and it counts EVERY evaluation including the two
    # before the one-sided search — the budget `ls_iterations` is a count of
    # `PrimalEval` calls, not of bracket steps.
    @always_inline
    def peval(
        a: Scalar[DTYPE],
        mut c: Scalar[DTYPE],
        mut d0: Scalar[DTYPE],
        mut d1: Scalar[DTYPE],
        mut it: Int,
    ) {imm}:
        c = Scalar[DTYPE](0.5) * gauss_a * a * a + gauss_b * a
        d0 = gauss_a * a + gauss_b
        d1 = gauss_a
        for e_idx in range(num_edges):
            var jt = jar[e_idx] + a * Jv_e[e_idx]
            var st = scalar_row_state[DTYPE](
                kind_e[e_idx], jt, R_e[e_idx], floss_e[e_idx]
            )
            var st0 = scalar_row_state[DTYPE](
                kind_e[e_idx], jar[e_idx], R_e[e_idx], floss_e[e_idx]
            )
            c += scalar_row_cost[DTYPE](
                st, jt, De[e_idx], R_e[e_idx], floss_e[e_idx]
            ) - scalar_row_cost[DTYPE](
                st0, jar[e_idx], De[e_idx], R_e[e_idx], floss_e[e_idx]
            )
            d0 += (
                -scalar_row_force[DTYPE](st, jt, De[e_idx], floss_e[e_idx])
                * Jv_e[e_idx]
            )
            if st == SROW_QUADRATIC:
                d1 += De[e_idx] * Jv_e[e_idx] * Jv_e[e_idx]
        # ⚠ FLOOR ONLY A NON-POSITIVE SECOND DERIVATIVE, AND ONLY TO
        # `mjMINVAL`. `PrimalEval` (engine_solver.c:1643) reads
        # `if (deriv[1] <= 0) { mju_warning("not convex"); deriv[1] = mjMINVAL; }`
        # — a guard against a SHOULD-NOT-OCCUR convexity violation. Testing
        # `< PRIMAL_MINVAL` instead floors a legitimately SMALL POSITIVE
        # curvature, which is what this IS near the optimum, and crushes
        # `alpha = -d0/d1` by up to three orders.
        if d1 <= ZERO:
            d1 = MINVAL
        it += 1

    var p0_a = ZERO
    var p0_c = ZERO
    var p0_d0 = ZERO
    var p0_d1 = ZERO
    peval(p0_a, p0_c, p0_d0, p0_d1, lsiter)
    gtol = ls_gtol_dtype_floor[DTYPE](gtol, p0_d0)

    # ⚠ `PrimalSearch` ALWAYS ATTEMPTS ONE NEWTON STEP (engine_solver.c:1733),
    # including when `d0 >= 0`. The old body returned 0 on a non-descent
    # direction; the caller's `alpha < 1e-10` break then ended the whole solve.
    var p1_a = p0_a - p0_d0 / p0_d1
    var p1_c = ZERO
    var p1_d0 = ZERO
    var p1_d1 = ZERO
    peval(p1_a, p1_c, p1_d0, p1_d1, lsiter)

    if abs(p1_d0) < gtol:
        return p1_a

    var dir = Scalar[DTYPE](1) if p1_d0 < ZERO else Scalar[DTYPE](-1)

    # ── phase 2: one-sided Newton search ────────────────────────────────
    var p2_a = p0_a
    var p2_c = p0_c
    var p2_d0 = p0_d0
    var p2_d1 = p0_d1
    while p1_d0 * dir <= -gtol and lsiter < budget:
        p2_a = p1_a
        p2_c = p1_c
        p2_d0 = p1_d0
        p2_d1 = p1_d1
        p1_a = p1_a - p1_d0 / p1_d1
        peval(p1_a, p1_c, p1_d0, p1_d1, lsiter)
        if abs(p1_d0) < gtol:
            return p1_a

    # Could not bracket within the budget (LSresult 3).
    if lsiter >= budget:
        return p1_a

    # ── phase 3: bracketed search over {p1next, p2next, pmid} ───────────
    # `p2next` starts as the point that ENDED phase 2 and `p1next` as one
    # Newton step off it. On apollo that first `p1next` is already the root.
    var n2_a = p1_a
    var n2_c = p1_c
    var n2_d0 = p1_d0
    var n2_d1 = p1_d1
    var n1_a = p1_a - p1_d0 / p1_d1
    var n1_c = ZERO
    var n1_d0 = ZERO
    var n1_d1 = ZERO
    peval(n1_a, n1_c, n1_d0, n1_d1, lsiter)

    var pm_a: Scalar[DTYPE]
    var pm_c = ZERO
    var pm_d0 = ZERO
    var pm_d1 = ZERO

    while lsiter < budget:
        pm_a = Scalar[DTYPE](0.5) * (p1_a + p2_a)
        peval(pm_a, pm_c, pm_d0, pm_d1, lsiter)

        # Cheapest candidate that is under `gtol`, scanned in MuJoCo's order
        # (`p1next`, `p2next`, `pmid`) so a cost tie resolves as it does there.
        var best_a = ZERO
        var best_c = ZERO
        var has_best = False
        if abs(n1_d0) < gtol:
            best_a = n1_a
            best_c = n1_c
            has_best = True
        if abs(n2_d0) < gtol and (not has_best or n2_c < best_c):
            best_a = n2_a
            best_c = n2_c
            has_best = True
        # ⚠ NO `best_c = pm_c` HERE, and the reference is not being
        # deviated from. `engine_solver.c:1842` writes `bestcost` inside a
        # LOOP over `candidates[3] = {p1next, p2next, pmid}`; we unrolled the
        # loop, and `pmid` is the last candidate, so that write is read by
        # nothing. `newton_solve.mojo` drops it at the same place — the two
        # copies of this linesearch stay identical. A FOURTH candidate would
        # need it back.
        if abs(pm_d0) < gtol and (not has_best or pm_c < best_c):
            best_a = pm_a
            has_best = True
        if has_best:
            return best_a

        # ── `updateBracket` (engine_solver.c:1665), once per bracket end ──
        # A candidate replaces the end when it has the SAME derivative sign
        # and is closer to zero; the end then gets a fresh Newton next-point.
        var b1 = False
        if p1_d0 < ZERO and n1_d0 < ZERO and p1_d0 < n1_d0:
            p1_a = n1_a; p1_c = n1_c; p1_d0 = n1_d0; p1_d1 = n1_d1; b1 = True
        elif p1_d0 > ZERO and n1_d0 > ZERO and p1_d0 > n1_d0:
            p1_a = n1_a; p1_c = n1_c; p1_d0 = n1_d0; p1_d1 = n1_d1; b1 = True
        if p1_d0 < ZERO and n2_d0 < ZERO and p1_d0 < n2_d0:
            p1_a = n2_a; p1_c = n2_c; p1_d0 = n2_d0; p1_d1 = n2_d1; b1 = True
        elif p1_d0 > ZERO and n2_d0 > ZERO and p1_d0 > n2_d0:
            p1_a = n2_a; p1_c = n2_c; p1_d0 = n2_d0; p1_d1 = n2_d1; b1 = True
        if p1_d0 < ZERO and pm_d0 < ZERO and p1_d0 < pm_d0:
            p1_a = pm_a; p1_c = pm_c; p1_d0 = pm_d0; p1_d1 = pm_d1; b1 = True
        elif p1_d0 > ZERO and pm_d0 > ZERO and p1_d0 > pm_d0:
            p1_a = pm_a; p1_c = pm_c; p1_d0 = pm_d0; p1_d1 = pm_d1; b1 = True

        var b2 = False
        if p2_d0 < ZERO and n1_d0 < ZERO and p2_d0 < n1_d0:
            p2_a = n1_a; p2_c = n1_c; p2_d0 = n1_d0; p2_d1 = n1_d1; b2 = True
        elif p2_d0 > ZERO and n1_d0 > ZERO and p2_d0 > n1_d0:
            p2_a = n1_a; p2_c = n1_c; p2_d0 = n1_d0; p2_d1 = n1_d1; b2 = True
        if p2_d0 < ZERO and n2_d0 < ZERO and p2_d0 < n2_d0:
            p2_a = n2_a; p2_c = n2_c; p2_d0 = n2_d0; p2_d1 = n2_d1; b2 = True
        elif p2_d0 > ZERO and n2_d0 > ZERO and p2_d0 > n2_d0:
            p2_a = n2_a; p2_c = n2_c; p2_d0 = n2_d0; p2_d1 = n2_d1; b2 = True
        if p2_d0 < ZERO and pm_d0 < ZERO and p2_d0 < pm_d0:
            p2_a = pm_a; p2_c = pm_c; p2_d0 = pm_d0; p2_d1 = pm_d1; b2 = True
        elif p2_d0 > ZERO and pm_d0 > ZERO and p2_d0 > pm_d0:
            p2_a = pm_a; p2_c = pm_c; p2_d0 = pm_d0; p2_d1 = pm_d1; b2 = True

        # ⚠ THE NEXT-POINTS ARE RECOMPUTED ONLY FOR AN END THAT MOVED, and an
        # end that did not move keeps the next-point it already had — which is
        # what lets a converged `p1next` survive to the candidate scan above.
        if b1:
            n1_a = p1_a - p1_d0 / p1_d1
            peval(n1_a, n1_c, n1_d0, n1_d1, lsiter)
        if b2:
            n2_a = p2_a - p2_d0 / p2_d1
            peval(n2_a, n2_c, n2_d0, n2_d1, lsiter)

        # Neither end could be improved: numerical accuracy reached, take the
        # midpoint (LSresult 0 when it improves, 7 when it does not).
        if not b1 and not b2:
            return pm_a

    # ⚠ NO IMPROVEMENT MEANS ZERO, NOT A SMALL STEP. `PrimalSearch` returns
    # the bracket with the lower cost only when that cost is BELOW alpha=0's,
    # and otherwise returns exactly 0 (LSresult 5) so the caller's loop breaks
    # without moving `qacc`. Returning a tiny non-improving alpha instead — as
    # the halving body did — nudged `qacc` by ~1e-12 on EVERY step of a
    # rollout once the warm start put the iterate near its optimum.
    if p1_c <= p2_c and p1_c < ZERO:
        return p1_a
    if p2_c <= p1_c and p2_c < ZERO:
        return p2_a
    return ZERO
