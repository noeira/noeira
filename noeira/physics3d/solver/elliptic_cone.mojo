"""The ELLIPTIC friction cone at any condim: state, force, Hessian, linesearch.

Transcribed from MuJoCo's `mjCNSTR_CONTACT_ELLIPTIC` branches:

    state + force + cone Hessian   engine_core_constraint.c  mj_constraintUpdate_impl
    linesearch quadratics          engine_solver.c           PrimalPrepare / PrimalEval

⚠⚠ WRITTEN IN MuJoCo's U-SPACE, WHICH IS WHY IT GENERALIZES. The solver used to
carry the same arithmetic with the friction coefficient FACTORED OUT: one
`mu` per contact, `T = |jar_t|` unscaled, two hard-coded tangents. That
factoring is algebraically identical to MuJoCo — but ONLY when every
tangential row shares one coefficient, which is exactly the condim-3
isotropic-slide case. The moment a contact has a torsional row
(`friction = "1 1 0.005"`) the cone stops being circular in `jar` and the
factored form has no way to say so. So the rows are kept RAW here and mapped
the way MuJoCo maps them:

    U[0] = jar_n * mu            N = U[0]
    U[t] = jar_t[t] * fr[t]      T = |U[1..]|

with `mu = con->mu` the REGULARIZED coefficient and `fr[t] = con->friction[t]`
the raw per-direction one. The three zones are then `N >= mu*T` (top, no
force), `mu*N + T <= 0` (bottom, unconstrained quadratic) and the cone surface
between them. At condim 3 with `fr[0] == fr[1]` this reduces algebraically to
the old expressions — verified term by term against them before the switch, so
condim-3 gates do not move.

⚠ THE `nt` ARGUMENT IS PER CONTACT AND RUNTIME. `NT` sizes the arrays for the
model's worst condim; `nt` is THIS contact's `dim-1`, because condim is a
property of the geom pair. Rows `t >= nt` are zeroed by the producer, but every
loop here is bounded by `nt` rather than relying on that — a zero row would
still contribute a `0*0` term to `T` and change nothing, whereas a zero
`fr[t]` in a denominator would not be so forgiving.

`nt == 0` is a FRICTIONLESS contact (`condim="1"`). It is not a special case:
`T` is 0, so the zone test collapses to `N >= 0` versus `N < 0`, which is the
one-sided normal constraint MuJoCo emits as `mjCNSTR_CONTACT_FRICTIONLESS`.
"""

from std.math import sqrt

from layout import TileTensor, TensorLayout

from ..fields.scratch import Scratch


# Constraint states, matching `mjCNSTRSTATE_*` for the values this cone can
# produce. The solver stores these per contact in `cs_arr`.
comptime ELL_SATISFIED: Int = 0
comptime ELL_QUADRATIC: Int = 1
comptime ELL_CONE: Int = 2

# `mjMINVAL`. Used only where MuJoCo's own guarantee (`T > 0` strictly in the
# middle zone) is one float32 rounding away from failing.
comptime ELL_MINVAL: Float64 = 1e-12


@always_inline
def ell_state_force[
    DTYPE: DType, NT: Int, T_CAP: Int
](
    nt: Int,
    base: Int,
    jar_n: Scalar[DTYPE],
    jar_t: Scratch[Scalar[DTYPE], T_CAP],
    mu: Scalar[DTYPE],
    D_n: Scalar[DTYPE],
    D_t: Scratch[Scalar[DTYPE], T_CAP],
    fr: Scratch[Scalar[DTYPE], T_CAP],
    mut f_n: Scalar[DTYPE],
    mut f_t: Scratch[Scalar[DTYPE], T_CAP],
) -> Int:
    """Zone, normal force and tangential forces for one contact.

    `base` indexes the contact's row block (`c*NT`) in the flat arrays;
    `jar_t[base+t]` is row `t`. Returns one of `ELL_SATISFIED` /
    `ELL_QUADRATIC` / `ELL_CONE` and writes `f_n` / `f_t[base..base+nt)`.
    """
    comptime ZERO = Scalar[DTYPE](0)
    comptime ONE = Scalar[DTYPE](1)
    comptime MINVAL = Scalar[DTYPE](ELL_MINVAL)

    var N = jar_n * mu
    var T_sq = ZERO
    for t in range(nt):
        var u = jar_t[base + t] * fr[base + t]
        T_sq += u * u
    var T = sqrt(T_sq)

    # top zone: no force at all
    if N >= mu * T or (T <= ZERO and N >= ZERO):
        f_n = ZERO
        for t in range(nt):
            f_t[base + t] = ZERO
        return ELL_SATISFIED

    # bottom zone: the unconstrained quadratic, one independent row each
    if mu * N + T <= ZERO or (T <= ZERO and N < ZERO):
        f_n = -D_n * jar_n
        for t in range(nt):
            f_t[base + t] = -D_t[base + t] * jar_t[base + t]
        return ELL_QUADRATIC

    # middle zone: on the cone surface.
    # `T > 0` STRICTLY here — `T == 0` forces `mu*N + T = mu*N <= 0` given
    # `N < mu*T = 0`, which is the bottom zone. The floor is float32 paranoia,
    # not a behaviour, and MuJoCo does not carry one.
    var T_s = T if T > MINVAL else MINVAL
    var Dm = D_n / (mu * mu * (ONE + mu * mu))
    var NmT = N - mu * T
    f_n = -Dm * NmT * mu
    for t in range(nt):
        var u = jar_t[base + t] * fr[base + t]
        f_t[base + t] = -f_n / T_s * u * fr[base + t]
    return ELL_CONE


@always_inline
def ell_row_cost[
    DTYPE: DType, NT: Int, T_CAP: Int
](
    state: Int,
    nt: Int,
    base: Int,
    jar_n: Scalar[DTYPE],
    jar_t: Scratch[Scalar[DTYPE], T_CAP],
    mu: Scalar[DTYPE],
    D_n: Scalar[DTYPE],
    D_t: Scratch[Scalar[DTYPE], T_CAP],
    fr: Scratch[Scalar[DTYPE], T_CAP],
) -> Scalar[DTYPE]:
    """One elliptic contact's primal cost (engine_core_constraint.c:3255-3277).

    ⚠ IT TAKES THE ZONE RATHER THAN RE-DERIVING IT. `ell_state_force` already
    classifies the contact and the classification is three float comparisons
    on a knife edge; a second copy of that rule beside this one is exactly the
    shape that has drifted in this tree before. Call them as a pair.
    """
    comptime ZERO = Scalar[DTYPE](0)
    comptime ONE = Scalar[DTYPE](1)

    if state == ELL_SATISFIED:
        return ZERO

    if state == ELL_QUADRATIC:
        # The unconstrained quadratic, one independent row per cone dimension
        # — the NORMAL row included.
        var s = Scalar[DTYPE](0.5) * D_n * jar_n * jar_n
        for t in range(nt):
            var jt = jar_t[base + t]
            s += Scalar[DTYPE](0.5) * D_t[base + t] * jt * jt
        return s

    # Middle zone: 0.5 * D0/(mu^2 (1+mu^2)) * (N - mu*T)^2.
    var N = jar_n * mu
    var T_sq = ZERO
    for t in range(nt):
        var u = jar_t[base + t] * fr[base + t]
        T_sq += u * u
    var T = sqrt(T_sq)
    var Dm = D_n / (mu * mu * (ONE + mu * mu))
    var NmT = N - mu * T
    return Scalar[DTYPE](0.5) * Dm * NmT * NmT


@always_inline
def ell_hessian_block[
    DTYPE: DType, NT: Int, T_CAP: Int, HN: Int
](
    state: Int,
    nt: Int,
    base: Int,
    jar_n: Scalar[DTYPE],
    jar_t: Scratch[Scalar[DTYPE], T_CAP],
    mu: Scalar[DTYPE],
    D_n: Scalar[DTYPE],
    D_t: Scratch[Scalar[DTYPE], T_CAP],
    fr: Scratch[Scalar[DTYPE], T_CAP],
    mut Hb: Array[Scalar[DTYPE], HN],
):
    """The contact's `dim x dim` Hessian block in ROW space, row-major over
    `(n, t_0, ..., t_{nt-1})` with stride `NT+1`.

    The caller turns this into the `nv x nv` contribution as
    `sum_{k,j} Hb[k,j] * J_k J_j^T`. Splitting it out is what lets QUADRATIC
    and CONE share one accumulation loop — they used to be two hand-fused
    copies of the same six outer products, written out twice more inside the
    Newton loop for the state-change rebuild.

    ⚠ `Hb` IS ZEROED HERE FOR THE FULL `(NT+1)^2`, not just the live `dim`
    rows. The caller loops to `nt+1`, but a stale entry from a contact with
    more rows would otherwise be read if that ever changed.
    """
    comptime ZERO = Scalar[DTYPE](0)
    comptime ONE = Scalar[DTYPE](1)
    comptime MINVAL = Scalar[DTYPE](ELL_MINVAL)
    comptime DIM = NT + 1

    for k in range(HN):
        Hb[k] = ZERO
    if state == ELL_SATISFIED:
        return

    if state == ELL_QUADRATIC:
        # Independent rows: diag(D_n, D_t...).
        Hb[0] = D_n
        for t in range(nt):
            Hb[(t + 1) * DIM + (t + 1)] = D_t[base + t]
        return

    # CONE. Verbatim from `mj_constraintUpdate_impl`'s `flg_coneHessian`
    # block, including the order of operations: build in U-space, then pre-
    # and post-multiply by `diag(mu, friction)` and scale by `Dm`.
    var N = jar_n * mu
    var T_sq = ZERO
    for t in range(nt):
        var u = jar_t[base + t] * fr[base + t]
        T_sq += u * u
    var T = sqrt(T_sq)
    var T_s = T if T > MINVAL else MINVAL
    var Dm = D_n / (mu * mu * (ONE + mu * mu))

    # first row: (1, -mu/T * U)
    Hb[0] = ONE
    var scl = -mu / T_s
    for t in range(nt):
        Hb[t + 1] = scl * (jar_t[base + t] * fr[base + t])

    # upper block: mu*N/T^3 * U U'
    scl = mu * N / (T_s * T_s * T_s)
    for k in range(nt):
        var uk = jar_t[base + k] * fr[base + k]
        for j in range(k, nt):
            var uj = jar_t[base + j] * fr[base + j]
            Hb[(k + 1) * DIM + (j + 1)] = scl * uj * uk

    # diagonal: += (mu^2 - mu*N/T)
    scl = mu * mu - mu * N / T_s
    for t in range(nt):
        Hb[(t + 1) * DIM + (t + 1)] += scl

    # pre/post multiply by diag(mu, friction), scale by Dm
    for k in range(nt + 1):
        var sk = Dm * (mu if k == 0 else fr[base + k - 1])
        for j in range(k, nt + 1):
            Hb[k * DIM + j] *= sk * (mu if j == 0 else fr[base + j - 1])

    # symmetrize
    for k in range(nt + 1):
        for j in range(k + 1, nt + 1):
            Hb[j * DIM + k] = Hb[k * DIM + j]

    # ── PSD PROJECTION — `HessianCone` (engine_solver.c:2052) ───────────────
    #
    # ⚠⚠ THE CONE HESSIAN IS INDEFINITE AND MuJoCo NEVER ADDS IT RAW. Its
    # middle-zone cost is `0.5*Dm*(N - mu*T)^2`, whose Hessian carries the term
    # `Dm*(N - mu*T)*Hess(N - mu*T)` — and `N - mu*T < 0` is the very condition
    # that DEFINES the middle zone, so that term is negative-curvature by
    # construction. `HessianCone` factors the block with
    # `mju_cholFactor(local, dim, mjMINVAL)`, whose diagonal CLAMP turns it
    # into a PSD matrix, and then applies `dim` rank-1 `mju_cholUpdate`s of
    # `L' J`. The Hessian the reference's Newton direction is computed against
    # is therefore `J' (L L') J`, not `J' H J`.
    #
    # ⚠ THE SYMPTOM OF ADDING IT RAW IS A ZIG-ZAG, NOT A BLOW-UP. An indefinite
    # Hessian gives a direction the linesearch has to cut, and the solver
    # enters a period-2 cycle: measured on `unitree_go1` at `impratio="100"`,
    # `alpha` alternated 0.0895 / 0.2317 for hundreds of iterations while
    # `scale*|grad|` decayed ~5% per PAIR of steps. MuJoCo converges that pose
    # in **6** iterations and we needed ~800 — the residual on board row
    # `unitree_go1` was this, not the cone algebra.
    #
    # `L L'` is reconstructed here rather than threaded out as a factor,
    # because the accumulation below already forms `J' Hb J` and the two are
    # the same matrix.
    var n = nt + 1
    for j in range(n):
        var tj = Hb[j * DIM + j]
        for k in range(j):
            tj -= Hb[j * DIM + k] * Hb[j * DIM + k]
        # `mjMINVAL`, and the clamp IS the projection — a pivot at or below it
        # is a direction of non-positive curvature, and MuJoCo keeps the
        # matrix usable rather than declaring the factorization failed.
        if tj < Scalar[DTYPE](1e-15):
            tj = Scalar[DTYPE](1e-15)
        Hb[j * DIM + j] = sqrt(tj)
        var inv = ONE / Hb[j * DIM + j]
        for i in range(j + 1, n):
            var v = Hb[i * DIM + j]
            for k in range(j):
                v -= Hb[i * DIM + k] * Hb[j * DIM + k]
            Hb[i * DIM + j] = v * inv
    # `Hb <- L L'`, lower factor read in place. Walk k from the LAST column
    # backwards so a row's own factor entries are still intact when read.
    for i in range(n - 1, -1, -1):
        for j in range(i, -1, -1):
            var acc = ZERO
            for k in range(j + 1):
                acc += Hb[i * DIM + k] * Hb[j * DIM + k]
            Hb[i * DIM + j] = acc
    for k in range(n):
        for j in range(k + 1, n):
            Hb[k * DIM + j] = Hb[j * DIM + k]


@always_inline
def _cn_len[
    SPARSE: Bool, N_CAP: Int
](cn_n: Scratch[Int, N_CAP], c: Int, nv: Int) -> Int:
    """How many dofs contact `c`'s rows touch: its nonzero list's length under
    `SPARSE`, all `nv` otherwise. Paired with `_cn_dof`, it lets one loop body
    serve both walks: `for a in range(_cn_len(...)): var i = _cn_dof(...)`
    is the dense `for i in range(nv)` when `SPARSE` is off."""
    comptime if SPARSE:
        return cn_n[c]
    else:
        return nv


@always_inline
def _cn_dof[
    SPARSE: Bool, IX_CAP: Int
](cn_ix: Scratch[Int, IX_CAP], c: Int, a: Int, nv: Int) -> Int:
    """The `a`-th dof contact `c` touches (`cn_ix[c*nv + a]`), or `a` itself
    when `SPARSE` is off. See `_cn_len`."""
    comptime if SPARSE:
        return cn_ix[c * nv + a]
    else:
        return a


@always_inline
def ell_add_contact_hessian[
    DTYPE: DType,
    MC_CAP: Int,
    NT: Int,
    T_CAP: Int,
    V_CAP: Int,
    M_CAP: Int,
    HN: Int,
    JT_CAP: Int,
    L_WS: TensorLayout,
    N_CAP: Int = 1,
    IX_CAP: Int = 1,
    SPARSE: Bool = False,
    JT_PC: Bool = False,
](
    nc: Int,
    cs_arr: Scratch[Int, MC_CAP],
    nt_cache: Scratch[Int, MC_CAP],
    Jn_c: Scratch[Scalar[DTYPE], MC_CAP * V_CAP],
    Jt_c: Scratch[Scalar[DTYPE], JT_CAP],
    jar_n_arr: Scratch[Scalar[DTYPE], MC_CAP],
    jar_t_arr: Scratch[Scalar[DTYPE], T_CAP],
    mu_cache: Scratch[Scalar[DTYPE], MC_CAP],
    D_n_cache: Scratch[Scalar[DTYPE], MC_CAP],
    D_t_cache: Scratch[Scalar[DTYPE], T_CAP],
    fr_cache: Scratch[Scalar[DTYPE], T_CAP],
    mut H: Scratch[Scalar[DTYPE], M_CAP],
    nv: Int,
    cn_n: Scratch[Int, N_CAP],
    cn_ix: Scratch[Int, IX_CAP],
    solver: TileTensor[DTYPE, L_WS, MutAnyOrigin],
    env: Int,
    ws_Jt_idx: Int,
    mc: Int,
):
    """Add every contact's `J^T Hb J` to the `nv x nv` Newton Hessian.

    ⚠ `SPARSE`: every walk over `nv` becomes a walk over the contact's
    nonzero-dof list (`cn_n` / `cn_ix`, the union of its normal and tangent
    rows' supports). A contact row is nonzero on its two bodies' ancestor
    chains only — 6 or 12 of reassemble5's 33 dofs — so `J^T Hb J` goes from
    `dim * nv^2` to `dim * nnz^2`. BIT-IDENTICAL to the dense walk: every
    skipped term is a product with an exact zero and adds nothing, and the
    `JH` entries outside the support are never read once both outer loops are
    restricted. Off, the loops are the dense originals.

    Two-stage — `JH[k] = sum_j Hb[k,j] J_j`, then `H += sum_k J_k JH[k]^T` —
    which is `O(dim^2 nv + dim nv^2)` rather than the `O(dim^2 nv^2)` a naive
    double loop would cost. At condim 3 that is THREE rank-1 outer products
    where the hand-fused two-tangent version it replaces did six, so the
    generalization is not a slowdown even before the extra rows.
    """
    comptime ZERO = Scalar[DTYPE](0)
    comptime DIM = NT + 1
    # `Hb` is (NT+1)^2 -- CONDIM-derived, so it stays a real Array and
    # keeps its comptime bound. `JH` is DIM rows of `nv`: the row COUNT is
    # condim, the row LENGTH is the dof count, and only the latter goes
    # dynamic. Not every comptime size in this file is a model dimension.
    var Hb = Array[Scalar[DTYPE], HN](fill=ZERO)
    var JH = Scratch[Scalar[DTYPE], DIM * V_CAP](DIM * nv, fill=ZERO)
    # ⚠ `JT_PC`: this contact's NT tangent rows, re-read from the workspace
    # per contact instead of indexing an all-contact stack cache. `NT*nv`
    # floats against the caller's `T_CAP*V_CAP` — see `_newton_solve_env`'s
    # note. Declared once, refilled at the top of each contact.
    comptime JTL_CAP = NT * V_CAP if JT_PC else 1
    var jt_l = Scratch[Scalar[DTYPE], JTL_CAP](
        NT * nv if JT_PC else 1, fill=ZERO
    )

    for c in range(nc):
        var cs = cs_arr[c]
        if cs == ELL_SATISFIED:
            continue
        var nt_c = nt_cache[c]
        comptime if JT_PC:
            for t in range(nt_c):
                for i in range(nv):
                    jt_l[t * nv + i] = rebind[Scalar[DTYPE]](
                        solver[env, ws_Jt_idx + t * mc * nv + c * nv + i]
                    )
        ell_hessian_block[DTYPE, NT, T_CAP, HN](
            cs, nt_c, c * NT, jar_n_arr[c], jar_t_arr,
            mu_cache[c], D_n_cache[c], D_t_cache, fr_cache, Hb,
        )

        var n_c = _cn_len[SPARSE](cn_n, c, nv)
        for k in range(nt_c + 1):
            for a in range(n_c):
                var i = _cn_dof[SPARSE](cn_ix, c, a, nv)
                JH[k * nv + i] = ZERO
            for j in range(nt_c + 1):
                var h = Hb[k * DIM + j]
                if h == ZERO:
                    continue
                if j == 0:
                    for a in range(n_c):
                        var i = _cn_dof[SPARSE](cn_ix, c, a, nv)
                        JH[k * nv + i] += h * Jn_c[c * nv + i]
                else:
                    var jb = (j - 1) * nv if JT_PC else (c * NT + j - 1) * nv
                    for a in range(n_c):
                        var i = _cn_dof[SPARSE](cn_ix, c, a, nv)
                        comptime if JT_PC:
                            JH[k * nv + i] += h * jt_l[jb + i]
                        else:
                            JH[k * nv + i] += h * Jt_c[jb + i]

        for k in range(nt_c + 1):
            var kb = c * nv if k == 0 else (
                (k - 1) * nv if JT_PC else (c * NT + k - 1) * nv
            )
            for a in range(n_c):
                var i = _cn_dof[SPARSE](cn_ix, c, a, nv)
                var jki: Scalar[DTYPE]
                if k == 0:
                    jki = Jn_c[kb + i]
                else:
                    comptime if JT_PC:
                        jki = jt_l[kb + i]
                    else:
                        jki = Jt_c[kb + i]
                if jki == ZERO:
                    continue
                # ⚠ LOWER TRIANGLE ONLY. Every reader of `H` is a Cholesky
                # (`chol_factor_inline` / `chol_factor_seg`) that reads
                # `H[i, j]` for `j <= i` and nothing else; the block is
                # symmetric, so the upper half was half the work for nobody
                # (PERFORMANCE.md §13.19). Same rule in `newton_solve.mojo`'s
                # equality rows and pyramidal outer products.
                # The dof list is ascending (dense: the index itself; sparse:
                # built by an ascending scan), so `b <= a` IS `j <= i` — no
                # test per term (§13.24).
                for b in range(a + 1):
                    var j = _cn_dof[SPARSE](cn_ix, c, b, nv)
                    H[i * nv + j] += jki * JH[k * nv + j]


@always_inline
def ell_line_eval[
    DTYPE: DType, NT: Int, T_CAP: Int
](
    nt: Int,
    base: Int,
    alpha: Scalar[DTYPE],
    jar_n: Scalar[DTYPE],
    jar_t: Scratch[Scalar[DTYPE], T_CAP],
    Js_n: Scalar[DTYPE],
    Js_t: Scratch[Scalar[DTYPE], T_CAP],
    mu: Scalar[DTYPE],
    D_n: Scalar[DTYPE],
    D_t: Scratch[Scalar[DTYPE], T_CAP],
    fr: Scratch[Scalar[DTYPE], T_CAP],
    mut cost: Scalar[DTYPE],
    mut d1: Scalar[DTYPE],
    mut d2: Scalar[DTYPE],
):
    """Add this contact's SHIFTED cost and two derivatives at `alpha`.

    `PrimalEval`'s elliptic branch (engine_solver.c:1511) fused with the
    `quad` terms `PrimalPrepare` builds for it — now including the cost, which
    this used to omit because the elliptic line search had nothing to spend it
    on (AUD-40).

    ⚠⚠ THE COST IS A ZONE-TRANSITION TABLE, NOT A FUNCTION OF THE END POINT.
    `ellipticCostDif` (engine_solver.c) branches on the pair
    (zone at alpha=0, zone at alpha) — nine cases — because the cone cost is
    only PIECEWISE smooth and the difference across a zone boundary is not
    the difference of the two pieces' formulas. Two of the cases matter for
    accuracy rather than bookkeeping:

      * MIDDLE -> MIDDLE is RATIONALIZED. Writing it as
        `0.5*Dm*(r^2 - r0^2)` cancels catastrophically when the step is small
        and `r ~ r0`, which is exactly the regime phase 3 lives in — it
        compares candidate costs that differ in the last digits. MuJoCo
        forms `T_delta = Tsqr_delta/(T + T0)` and
        `0.5*Dm*r_delta*(2*r0 + r_delta)` instead, which never subtracts two
        near-equal large numbers.
      * the BOUNDARY-CROSSING cases (3->2, 2->3) add or subtract the
        half-gap `0.5*Dm*(mu*N + T)^2` at the crossing, because the quadratic
        and cone pieces do not meet at the same value.

    The derivatives are unchanged, and deliberately still go through
    `_ell_quad_deriv` in the bottom zone rather than through the `q1`/`q2`
    form MuJoCo uses (`2*alpha*quad[2] + quad[1]`). The two are algebraically
    the same sum in a different association; keeping the existing one means
    this change moves the COST only, and a gate that moves is telling you
    about phase 3 rather than about a re-associated accumulation.

    ⚠ `nt == 0` (a frictionless contact) reaches this with `UU == VV == 0`,
    so both zones are decided by the sign of `N` alone and the table collapses
    to the one-sided quadratic — the same answer `scalar_row_cost` gives for
    the row MuJoCo would emit as `mjCNSTR_CONTACT_FRICTIONLESS`.
    """
    comptime ZERO = Scalar[DTYPE](0)
    comptime ONE = Scalar[DTYPE](1)
    comptime TWO = Scalar[DTYPE](2)
    comptime HALF = Scalar[DTYPE](0.5)
    comptime MINVAL = Scalar[DTYPE](ELL_MINVAL)

    # `PrimalPrepare`'s `quad[0..2]` (engine_solver.c:1466-1489) — the
    # BOTTOM-ZONE quadratic over this contact's `dim` rows:
    #
    #     q0 = 0.5 * sum_k D_k * jar_k^2
    #     q1 =       sum_k D_k * jar_k * Js_k
    #     q2 = 0.5 * sum_k D_k * Js_k^2
    #
    # ⚠ `q1` IS NOT HALVED AND THE OTHER TWO ARE. The asymmetry is what makes
    # `cost = alpha^2*q2 + alpha*q1` and `deriv = 2*alpha*q2 + q1` the same
    # polynomial's value and slope.
    #
    # ⚠ ALPHA-INDEPENDENT AND RECOMPUTED ANYWAY. MuJoCo hoists these out of
    # the search; hoisting them here means three `Scratch[MAX_CONTACTS]`
    # arrays in the caller, and those take the per-env Newton kernel over
    # Metal's per-thread stack — a failure that reports no diagnostic at all.
    # See the caller's note.
    var q0 = jar_n * D_n * jar_n
    var q1 = Js_n * D_n * jar_n
    var q2 = Js_n * D_n * Js_n
    for t in range(nt):
        var dq = D_t[base + t]
        var jr = jar_t[base + t]
        var js = Js_t[base + t]
        q0 += jr * dq * jr
        q1 += js * dq * jr
        q2 += js * dq * js
    q0 *= HALF
    q2 *= HALF

    # U/V: the ray `jar + alpha*Js` mapped into the space where the cone is
    # circular. `UU/UV/VV` are `PrimalPrepare`'s quad[5..7].
    var U0 = jar_n * mu
    var V0 = Js_n * mu
    var UU = ZERO
    var UV = ZERO
    var VV = ZERO
    for t in range(nt):
        var u = jar_t[base + t] * fr[base + t]
        var v = Js_t[base + t] * fr[base + t]
        UU += u * u
        UV += u * v
        VV += v * v
    var Dm = D_n / (mu * mu * (ONE + mu * mu))

    # ── the zone at alpha = 0 ────────────────────────────────────────────
    var zone0: Int
    var T0 = ZERO
    if UU <= ZERO:
        zone0 = 2 if U0 < ZERO else 1
    else:
        T0 = sqrt(UU)
        if U0 >= mu * T0:
            zone0 = 1
        elif mu * U0 + T0 <= ZERO:
            zone0 = 2
        else:
            zone0 = 3

    # ── the zone at alpha ────────────────────────────────────────────────
    var N = U0 + alpha * V0
    var T_sq = UU + alpha * (TWO * UV + alpha * VV)
    var zone_a: Int
    var T = ZERO
    if T_sq <= ZERO:
        zone_a = 2 if N < ZERO else 1
    else:
        T = sqrt(T_sq)
        if N >= mu * T:
            zone_a = 1
        elif mu * N + T <= ZERO:
            zone_a = 2
        else:
            zone_a = 3

    # ── cost: `ellipticCostDif`, case for case ───────────────────────────
    if zone0 == 1 and zone_a == 1:
        pass  # both top: no cost at either end
    elif zone0 == 2 and zone_a == 2:
        cost += alpha * alpha * q2 + alpha * q1
    elif zone0 == 3 and zone_a == 3:
        var Tsq_delta = alpha * (TWO * UV + alpha * VV)
        var T_delta = Tsq_delta / (T + T0)
        var r_delta = alpha * V0 - mu * T_delta
        var r0 = U0 - mu * T0
        cost += HALF * Dm * r_delta * (TWO * r0 + r_delta)
    elif zone0 == 3 and zone_a == 2:
        var b0 = mu * U0 + T0
        cost += alpha * (alpha * q2 + q1) + HALF * Dm * b0 * b0
    elif zone0 == 2 and zone_a == 3:
        var b = mu * N + T
        cost += alpha * (alpha * q2 + q1) - HALF * Dm * b * b
    elif zone0 == 1 and zone_a == 2:
        cost += alpha * alpha * q2 + alpha * q1 + q0
    elif zone0 == 1 and zone_a == 3:
        var r = N - mu * T
        cost += HALF * Dm * r * r
    elif zone0 == 3 and zone_a == 1:
        var r0 = U0 - mu * T0
        cost += -HALF * Dm * r0 * r0
    elif zone0 == 2 and zone_a == 1:
        cost += -q0

    # ── derivatives, unchanged ───────────────────────────────────────────
    if zone_a == 2:
        _ell_quad_deriv[DTYPE, NT, T_CAP](
            nt, base, alpha, jar_n, jar_t, Js_n, Js_t, D_n, D_t, d1, d2
        )
    elif zone_a == 3:
        var T_s = T if T > MINVAL else MINVAL
        var N1 = V0
        var T1 = (UV + alpha * VV) / T_s
        var T2 = VV / T_s - (UV + alpha * VV) * T1 / (T_s * T_s)
        var NmT = N - mu * T
        var dN = N1 - mu * T1
        d1 += Dm * NmT * dN
        d2 += Dm * (dN * dN + NmT * (-mu * T2))


@always_inline
def _ell_quad_deriv[
    DTYPE: DType, NT: Int, T_CAP: Int
](
    nt: Int,
    base: Int,
    alpha: Scalar[DTYPE],
    jar_n: Scalar[DTYPE],
    jar_t: Scratch[Scalar[DTYPE], T_CAP],
    Js_n: Scalar[DTYPE],
    Js_t: Scratch[Scalar[DTYPE], T_CAP],
    D_n: Scalar[DTYPE],
    D_t: Scratch[Scalar[DTYPE], T_CAP],
    mut d1: Scalar[DTYPE],
    mut d2: Scalar[DTYPE],
):
    """Bottom zone: every row is an independent quadratic `0.5*D*jar^2`."""
    var tN = jar_n + alpha * Js_n
    d1 += D_n * tN * Js_n
    d2 += D_n * Js_n * Js_n
    for t in range(nt):
        var d = D_t[base + t]
        var js = Js_t[base + t]
        d1 += d * (jar_t[base + t] + alpha * js) * js
        d2 += d * js * js
