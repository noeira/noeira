"""Scalar (single-dof) constraint rows — joint limits and dry-friction dofs —
built in the shape the primal Newton solvers consume, so they can be solved
*with* the contact rows instead of after them.

WHY THIS EXISTS
---------------
MuJoCo puts every constraint row in ONE system (`mj_solNewton` over the whole
`efc` block) so each force is solved knowing about the others. We used to solve
contacts in the Newton core and then apply limits / frictionloss / equality /
tendons as SEPARATE PGS post-passes. Each pass saw the ones before it and never
the ones after, so the contact force was computed as if the other rows were not
there.

Measured on dm_control's finger `turn`, where a frictionloss row on the spinner
dof is live at the same instant as the three elliptic contact rows (nefc=4,
nf=1) and acts on a dof the contact rows also act on: solving MuJoCo's own
primal problem from MuJoCo's own efc data reproduces MuJoCo's contact force to
~1e-14 with all 4 rows, and reproduces OUR force to ~1e-7 with the 3 contact
rows alone. The solver was right; the system it was handed was missing a row.

A scalar row is `J = sign * e_dof` — a single nonzero — so it is stored as
`(dof, sign)` rather than a dense NV-vector. That keeps the added local storage
at O(rows) instead of O(rows*NV), which matters: the elliptic Newton core is
already close to the Metal local-memory ceiling (see the RK4/elliptic OOM note
in the solver docstring). Equality rows need a dense J, so they are not
built here — the PYRAMIDAL Newton path, which already carries a dense-J edge
list, gets fixed-tendon equality rows from
`constraints/tendon_limit.build_tendon_equality_rows` instead (2026-07-31);
the elliptic and CG paths, which consume THIS builder, still run them as
post-passes. `SROW_EQ_BILATERAL` lives here because the row-state / force /
cost helpers below are shared by both.

ROW SEMANTICS (MuJoCo `mj_constraintUpdate`, engine_core_constraint.c:2296+)

    jar   = sign * qacc[dof] + bias          (bias = -aref)
    LIMIT     one-sided:  jar >= 0        -> SATISFIED, f = 0
                          else            -> QUADRATIC, f = -D*jar
    FRICTION  box:        jar <= -R*floss -> LINEARNEG, f = +floss
                          jar >=  R*floss -> LINEARPOS, f = -floss
                          else            -> QUADRATIC, f = -D*jar

The cost derivative along a search direction is `-f * Jv` in EVERY state (for
QUADRATIC that is `D*jar*Jv`, for the linear branches `-/+ floss*Jv`, and for
SATISFIED it is zero because f is), which is why the line-search sites below
need only the force, not the branch.

CONSUMERS: `solver/newton_solve.mojo` (elliptic + pyramidal) and
`solver/cg_solve.mojo`. The two PGS solvers (`constraints/contact_solve.mojo`,
`solver/island_pgs_solve.mojo`) still run these as post-passes.

⚠ WHY THE PGS PAIR IS NOT DONE (attempted and reverted 2026-07-30). A PGS
solver cannot take these as a Hessian block, so the integration is instead one
Gauss-Seidel sweep of the rows inside the coupled contact loop. That much works.
What does not: `island_pgs_solve` freezes each island once its contacts settle
and breaks out of the loop when all islands are frozen, and a scalar row belongs
to no island. Two measured failures of `test_island_pgs_fields` Part A
(IslandPGS vs PGS, which must agree):

  * naive interleave                      -> 0.372 rel
  * + hold the loop open until the rows
    have also stopped moving              -> 0.065 rel

The residue is that contacts in a FROZEN island stop updating while the rows
keep pushing those same dofs, so the island never reacts. Closing it needs a
dof -> island map and un-freeze-on-row-update inside the island bookkeeping —
a change to a performance layer that deserves its own validation pass, not a
tail-end of this one.
"""

from std.math import pow
from layout import TileTensor, TensorLayout

from ..joint_types import JNT_HINGE, JNT_SLIDE, JNT_FREE, JNT_BALL
from ..gpu.constants import (
    MODEL_META_IDX_TIMESTEP,
    MODEL_JOINT_SIZE,
    MODEL_META_SIZE,
    MODEL_META_IDX_SOLREF_LIMIT_0,
    MODEL_META_IDX_SOLREF_LIMIT_1,
    MODEL_META_IDX_SOLIMP_LIMIT_0,
    MODEL_META_IDX_SOLIMP_LIMIT_1,
    MODEL_META_IDX_SOLIMP_LIMIT_2,
    MODEL_META_IDX_SOLIMP_LIMIT_3,
    MODEL_META_IDX_SOLIMP_LIMIT_4,
    JOINT_IDX_TYPE,
    JOINT_IDX_DOF_ADR,
    JOINT_IDX_QPOS_ADR,
    JOINT_IDX_RANGE_MIN,
    JOINT_IDX_RANGE_MAX,
    JOINT_IDX_MARGIN,
    JOINT_IDX_FRICTIONLOSS,
    JOINT_IDX_SOLREF_LIMIT_0,
    JOINT_IDX_SOLREF_LIMIT_1,
    JOINT_IDX_SOLIMP_LIMIT_0,
    JOINT_IDX_SOLIMP_LIMIT_1,
    JOINT_IDX_SOLIMP_LIMIT_2,
    JOINT_IDX_SOLIMP_LIMIT_3,
    JOINT_IDX_SOLIMP_LIMIT_4,
)

# Row kinds
from .constraint_data import (
    solref_spring_damper,
    refsafe_timeconst,
)
from ..fields import DimsLike
from ..fields.scratch import Scratch, cap

comptime SROW_LIMIT: Int = 0
comptime SROW_FRICTION: Int = 1
# Bilateral: an equality row. Always active, never clamped — its state is
# unconditionally QUADRATIC, so `R`/`floss` are never read for it. Built by
# `constraints/tendon_limit.build_tendon_equality_rows` for the pyramidal edge
# list; `build_scalar_rows` below does NOT emit one (see the docstring note on
# equality rows in the elliptic/CG paths).
comptime SROW_EQ_BILATERAL: Int = 2

# Row states (MuJoCo mjCNSTRSTATE_*)
comptime SROW_SATISFIED: Int = 0
comptime SROW_QUADRATIC: Int = 1
comptime SROW_LINEARNEG: Int = 2
comptime SROW_LINEARPOS: Int = 3

# MuJoCo dof_solref / dof_solimp defaults for friction rows (MJCF
# `solreffriction` / `solimpfriction`). Kept in sync with
# constraints/friction_dof.mojo — see that module's docstring for why these are
# DISTINCT from the limit pair and must not be conflated with it.
comptime DOF_SOLREF_TIMECONST: Float64 = 0.02
comptime DOF_SOLIMP_DMIN: Float64 = 0.9
comptime DOF_SOLIMP_DMAX: Float64 = 0.95

# engine_core_constraint.c:1284-1287
comptime MJ_MINIMP: Float64 = 0.0001
comptime MJ_MAXIMP: Float64 = 0.9999


@always_inline
def max_scalar_rows(nv: Int, njoint: Int) -> Int:
    """One row per frictional dof plus up to two limit rows per joint.

    ⚠ THIS IS THE CAPACITY GUARD'S BOUND, and it must be the LIVE one. It is
    what `n >= max_rows` compares against in the builder below, i.e. the point
    at which rows start being DROPPED. Feed it a cap and on the dynamic leg it
    reads 0, every row is dropped, and the solver runs on an empty constraint
    set — a silent physics change, not a crash.
    """
    var n = 2 * njoint + nv
    return n if n > 0 else 1


@always_inline
def max_scalar_rows_cap[NV: Int, NJOINT: Int]() -> Int:
    """The same count as a scratch CAP: 0 when either dimension is dynamic.

    Deliberately UNCLAMPED, unlike the runtime form above. `Scratch` reads 0
    as "use the heap"; the old `n if n > 0 else 1` clamp existed to keep
    `Array` legal at size zero, and `_slot[]` now does that job inside
    the container. Clamping here would hand the dynamic leg a ONE-element
    stack array instead — see `fields/scratch.mojo` on why caps poison to 0.
    """
    return cap[2 * NJOINT + NV]()


@always_inline
def scalar_row_state[
    DTYPE: DType
](
    kind: Int,
    jar: Scalar[DTYPE],
    R: Scalar[DTYPE],
    floss: Scalar[DTYPE],
) -> Int:
    """MuJoCo's per-row branch, given the current `jar`."""
    if kind == SROW_EQ_BILATERAL:
        return SROW_QUADRATIC
    if kind == SROW_LIMIT:
        if jar >= Scalar[DTYPE](0):
            return SROW_SATISFIED
        return SROW_QUADRATIC
    if jar <= -R * floss:
        return SROW_LINEARNEG
    if jar >= R * floss:
        return SROW_LINEARPOS
    return SROW_QUADRATIC


@always_inline
def scalar_row_force[
    DTYPE: DType
](
    state: Int,
    jar: Scalar[DTYPE],
    D: Scalar[DTYPE],
    floss: Scalar[DTYPE],
) -> Scalar[DTYPE]:
    """Constraint force for a row already classified by `scalar_row_state`."""
    if state == SROW_QUADRATIC:
        return -D * jar
    if state == SROW_LINEARNEG:
        return floss
    if state == SROW_LINEARPOS:
        return -floss
    return Scalar[DTYPE](0)


@always_inline
def scalar_row_cost[
    DTYPE: DType
](
    state: Int,
    jar: Scalar[DTYPE],
    D: Scalar[DTYPE],
    R: Scalar[DTYPE],
    floss: Scalar[DTYPE],
) -> Scalar[DTYPE]:
    """Primal cost contribution (engine_core_constraint.c:2306-2334)."""
    if state == SROW_QUADRATIC:
        return Scalar[DTYPE](0.5) * D * jar * jar
    if state == SROW_LINEARNEG:
        return -Scalar[DTYPE](0.5) * R * floss * floss - floss * jar
    if state == SROW_LINEARPOS:
        return -Scalar[DTYPE](0.5) * R * floss * floss + floss * jar
    return Scalar[DTYPE](0)


@always_inline
def _clamp_imp[DTYPE: DType](v: Scalar[DTYPE]) -> Scalar[DTYPE]:
    if v < Scalar[DTYPE](MJ_MINIMP):
        return Scalar[DTYPE](MJ_MINIMP)
    if v > Scalar[DTYPE](MJ_MAXIMP):
        return Scalar[DTYPE](MJ_MAXIMP)
    return v


@always_inline
def build_scalar_rows[
    DTYPE: DType,
    S_CAP: Int,
    D: DimsLike,
    L_QPOS: TensorLayout,
    L_QVEL: TensorLayout,
    L_JOINTS: TensorLayout,
    L_MMETA: TensorLayout,
    L_DOF_INVWEIGHT0: TensorLayout,
    L_M_INV: TensorLayout,
](
    env: Int,
    dims: D,
    qpos: TileTensor[DTYPE, L_QPOS, MutAnyOrigin],
    qvel: TileTensor[DTYPE, L_QVEL, MutAnyOrigin],
    joints: TileTensor[DTYPE, L_JOINTS, MutAnyOrigin],
    mmeta: TileTensor[DTYPE, L_MMETA, MutAnyOrigin],
    dof_invweight0: TileTensor[DTYPE, L_DOF_INVWEIGHT0, MutAnyOrigin],
    m_inv: TileTensor[DTYPE, L_M_INV, MutAnyOrigin],
    mut sr_dof: Scratch[Int, S_CAP],
    mut sr_kind: Scratch[Int, S_CAP],
    mut sr_sign: Scratch[Scalar[DTYPE], S_CAP],
    mut sr_D: Scratch[Scalar[DTYPE], S_CAP],
    mut sr_R: Scratch[Scalar[DTYPE], S_CAP],
    mut sr_bias: Scratch[Scalar[DTYPE], S_CAP],
    mut sr_floss: Scratch[Scalar[DTYPE], S_CAP],
) -> Int:
    """Build the active limit + friction rows for one env. Returns the count.

    Limit rows reproduce the PYRAMIDAL Newton path's existing limit builder
    verbatim (per-joint solref/solimp with model-level fallback, and the
    `D = 1/(1/inv_K - K)` round-trip that matches the CPU `primal_D`), so
    folding them into the ELLIPTIC path does not perturb pyramidal goldens.
    Friction rows reproduce `constraints/friction_dof.mojo`.
    """
    var nv = dims.get_nv()
    var njoint = dims.get_njoint()
    # ⚠ THE LIVE BOUND, not `S_CAP`. S_CAP is 0 on a dynamic provider, so
    # guarding with it would drop every row and hand the solver an empty
    # constraint set — the failure this whole phase exists to prevent.
    var max_rows = max_scalar_rows(nv, njoint)
    var n = 0

    # ---- joint limits -----------------------------------------------------
    var lr_tc_def = rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_SOLREF_LIMIT_0])
    var lr_dr_def = rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_SOLREF_LIMIT_1])
    var li_dmin_def = rebind[Scalar[DTYPE]](
        mmeta[MODEL_META_IDX_SOLIMP_LIMIT_0]
    )
    var li_dmax_def = rebind[Scalar[DTYPE]](
        mmeta[MODEL_META_IDX_SOLIMP_LIMIT_1]
    )
    var li_width_def = rebind[Scalar[DTYPE]](
        mmeta[MODEL_META_IDX_SOLIMP_LIMIT_2]
    )
    var li_mid_def = rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_SOLIMP_LIMIT_3])
    var li_pow_def = rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_SOLIMP_LIMIT_4])

    for j in range(njoint):
        var jtype = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_TYPE]))
        if jtype != JNT_HINGE and jtype != JNT_SLIDE:
            continue
        var rmin = rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_RANGE_MIN])
        var rmax = rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_RANGE_MAX])
        if rmin < Scalar[DTYPE](-1e9) or rmax > Scalar[DTYPE](1e9):
            continue
        var dof = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_DOF_ADR]))
        var qadr = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_QPOS_ADR]))
        var pos = rebind[Scalar[DTYPE]](qpos[env, qadr])
        # `jnt_margin` (AUD-03): row when `dist < margin`, everything
        # downstream sees `dist - margin` (engine_core_constraint.c:1394-1425).
        var jmargin = rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_MARGIN])

        # Per-joint solref/solimp with model-level fallback.
        var lr_tc = rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_SOLREF_LIMIT_0])
        var lr_dr = rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_SOLREF_LIMIT_1])
        # ⚠ `== 0` NOT `<= 0` (AUD-29): a NEGATIVE solref is MuJoCo's direct
        # stiffness/damping form and must reach `solref_spring_damper`. Only
        # an unset (zero) record falls back to the model default.
        if lr_tc == Scalar[DTYPE](0):
            lr_tc = lr_tc_def
        if lr_dr == Scalar[DTYPE](0):
            lr_dr = lr_dr_def
        var li_dmin = rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_SOLIMP_LIMIT_0])
        var li_dmax = rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_SOLIMP_LIMIT_1])
        var li_width = rebind[Scalar[DTYPE]](
            joints[j, JOINT_IDX_SOLIMP_LIMIT_2]
        )
        var li_mid = rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_SOLIMP_LIMIT_3])
        var li_pow = rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_SOLIMP_LIMIT_4])
        if li_dmax <= Scalar[DTYPE](0) and li_width <= Scalar[DTYPE](0):
            li_dmin = li_dmin_def
            li_dmax = li_dmax_def
            li_width = li_width_def
            li_mid = li_mid_def
            li_pow = li_pow_def
        if li_width < Scalar[DTYPE](1e-6):
            li_width = Scalar[DTYPE](1e-6)
        li_dmin = _clamp_imp[DTYPE](li_dmin)
        li_dmax = _clamp_imp[DTYPE](li_dmax)
        if li_pow < Scalar[DTYPE](1):
            li_pow = Scalar[DTYPE](1)
        # solref -> (K, B), including MuJoCo's DIRECT form for a NEGATIVE
        # solref. See `constraints/constraint_data.solref_spring_damper` — the
        # formula lived in twelve copy-pasted sites until 2026-08-03.
        var (K_spring, B_damp) = solref_spring_damper[DTYPE](
            lr_tc, lr_dr, li_dmax,
            rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_TIMESTEP]),
        )

        for side in range(2):
            var sign = Scalar[DTYPE](1) if side == 0 else Scalar[DTYPE](-1)
            var dist = (pos - rmin) if side == 0 else (rmax - pos)
            if dist >= jmargin or n >= max_rows:
                continue
            var pen = jmargin - dist

            # getimpedance (engine_core_constraint.c:1361-1379)
            var imp: Scalar[DTYPE]
            if li_dmin == li_dmax or li_width <= Scalar[DTYPE](0):
                imp = Scalar[DTYPE](0.5) * (li_dmin + li_dmax)
            else:
                var x = pen / li_width
                var y: Scalar[DTYPE]
                if x <= Scalar[DTYPE](0):
                    y = Scalar[DTYPE](0)
                elif x >= Scalar[DTYPE](1):
                    y = Scalar[DTYPE](1)
                elif li_pow == Scalar[DTYPE](1):
                    y = x
                elif x <= li_mid:
                    var a = Scalar[DTYPE](1) / pow(
                        li_mid, li_pow - Scalar[DTYPE](1)
                    )
                    y = a * pow(x, li_pow)
                else:
                    var b = Scalar[DTYPE](1) / pow(
                        Scalar[DTYPE](1) - li_mid, li_pow - Scalar[DTYPE](1)
                    )
                    y = Scalar[DTYPE](1) - b * pow(
                        Scalar[DTYPE](1) - x, li_pow
                    )
                imp = li_dmin + y * (li_dmax - li_dmin)
            if imp < Scalar[DTYPE](1e-6):
                imp = Scalar[DTYPE](1e-6)

            # `dof_invweight0`, MuJoCo's `mj_diagApprox` for a dof row; the
            # per-step `diag(M^-1)` this read is no longer formed under Newton
            # (see the pyramidal builder in `solver/newton_solve.mojo`).
            var diag = rebind[Scalar[DTYPE]](dof_invweight0[dof])
            var R_lim = (Scalar[DTYPE](1) - imp) / imp * diag
            if R_lim < Scalar[DTYPE](1e-14):
                R_lim = Scalar[DTYPE](1e-14)

            var v = sign * rebind[Scalar[DTYPE]](qvel[env, dof])
            sr_dof[n] = dof
            sr_kind[n] = SROW_LIMIT
            sr_sign[n] = sign
            sr_R[n] = R_lim
            sr_D[n] = Scalar[DTYPE](1) / R_lim
            sr_bias[n] = B_damp * v - K_spring * imp * pen
            sr_floss[n] = Scalar[DTYPE](0)
            n += 1

    # ---- dry-friction dofs ------------------------------------------------
    # K = 0 for a friction row, so only B survives: aref = -B*vel. `pos` is
    # identically 0, so the impedance sits on the saturated branch at dmin.
    var f_imp = _clamp_imp[DTYPE](Scalar[DTYPE](DOF_SOLIMP_DMIN))
    var f_dmax = _clamp_imp[DTYPE](Scalar[DTYPE](DOF_SOLIMP_DMAX))
    # REFSAFE applies to the hardcoded friction default too — see
    # `refsafe_timeconst` and friction_dof.mojo.
    var f_tc = refsafe_timeconst[DTYPE](
        Scalar[DTYPE](DOF_SOLREF_TIMECONST),
        rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_TIMESTEP]),
    )
    var f_B = Scalar[DTYPE](2.0) / (f_dmax * f_tc)

    for j in range(njoint):
        var floss = rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_FRICTIONLOSS])
        if floss <= Scalar[DTYPE](0):
            continue
        var jtype = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_TYPE]))
        var dof_adr = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_DOF_ADR]))
        var nd = 1
        if jtype == JNT_FREE:
            nd = 6
        elif jtype == JNT_BALL:
            nd = 3
        for k in range(nd):
            if n >= max_rows:
                break
            var dof = dof_adr + k
            # `dof_invweight0`, MuJoCo's `mj_diagApprox` for a dof row; the
            # per-step `diag(M^-1)` this read is no longer formed under Newton
            # (see the pyramidal builder in `solver/newton_solve.mojo`).
            var diag = rebind[Scalar[DTYPE]](dof_invweight0[dof])
            var R_f = (Scalar[DTYPE](1.0) - f_imp) / f_imp * diag
            if R_f < Scalar[DTYPE](1e-14):
                R_f = Scalar[DTYPE](1e-14)
            sr_dof[n] = dof
            sr_kind[n] = SROW_FRICTION
            sr_sign[n] = Scalar[DTYPE](1)
            sr_R[n] = R_f
            sr_D[n] = Scalar[DTYPE](1) / R_f
            sr_bias[n] = f_B * rebind[Scalar[DTYPE]](qvel[env, dof])
            sr_floss[n] = floss
            n += 1

    return n
