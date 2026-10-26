"""Tendon limit rows — MuJoCo's `mjCNSTR_LIMIT_TENDON`.

A tendon whose `<spatial>`/`<fixed>` element carries `limited="true"` and a
`range` gets a one-sided row per violated side, exactly like a joint limit but
with a DENSE Jacobian:

    dist = side * (range[(side+1)/2] - ten_length)      side = -1 (lo), +1 (hi)
    jac  = -side * ten_J                                (engine_core_constraint.c:814)
    D    = 1 / R,  R = (1-imp)/imp * tendon_invweight0

`tendon_invweight0` is `J M^-1 J^T` at qpos0 and is the row's `diagApprox`
(`engine_core_constraint.c:1130`) — the same slot `dof_invweight0` fills for a
joint limit. Getting it wrong is a silent multiplicative error on the whole
constraint force, which is exactly how bug 20 hid for weeks; the ball_in_cup
gate diffs it against MuJoCo directly.

WHY THIS IS A ROW AND NOT A POST-PASS. Commit 04a7c508 moved joint limits and
dry friction INTO the Newton/CG systems after proving that solving them
sequentially after the contacts was the whole of dm_control finger's contact
residual (0.216 N of force error, 0.0390 gated). ball_in_cup has exactly the
same shape — a caught ball rests on the cup's capsules (contacts) while the
string is taut (this row), on shared dofs — so building it as a post-pass
would have reproduced the bug we just finished removing.

SCOPE: PYRAMIDAL only. The pyramidal solvers already carry a dense `Je` edge
list, so a dense row costs nothing structurally. The ELLIPTIC core stores its
scalar rows as `(dof, sign)` precisely to stay under Metal's local-memory
ceiling (see constraints/scalar_rows.mojo), and `MAX_TENDON * NV` extra floats
there is a change that needs its own measurement. No elliptic model has a
limited tendon; `ModelDefFromXML` RAISES on the combination rather than
letting it pass silently.

Also SOLVER-scoped: the rows are built in `solver/newton_solve.mojo`, which is
the env default (`Phyics3dEnv.SOLVER = "newton"`). `cg_solve` is elliptic-only
and so is covered by the same raise. The PGS solvers (`contact_solve`,
`island_pgs_solve`) would silently omit the row; they are already the two
paths that omit the joint-limit/friction rows for the same structural reason
(see constraints/scalar_rows.mojo), and nothing gated or trained selects them.
"""

from std.collections import Array
from std.math import pow
from layout import TileTensor, TensorLayout
from ..fields.scratch import Scratch


from ..gpu.constants import (
    MODEL_META_IDX_TIMESTEP,
    MODEL_BODY_SIZE,
    MODEL_JOINT_SIZE,
    MODEL_META_SIZE,
    MODEL_SITE_SIZE,
    MODEL_GEOM_SIZE,
    MODEL_TENDON_SIZE,
    MODEL_META_IDX_NTENDON,
    JOINT_IDX_DOF_ADR,
    TENDON_KIND_SPATIAL,
    TENDON_IDX_KIND,
    TENDON_IDX_LIMITED,
    TENDON_IDX_RANGE_MIN,
    TENDON_IDX_RANGE_MAX,
    TENDON_IDX_MARGIN,
    TENDON_IDX_INVWEIGHT0,
    TENDON_IDX_NUM_JOINTS,
    TENDON_IDX_JOINT_0,
    TENDON_MAX_WRAPS,
    TENDON_IDX_COEF_0,
    TENDON_IDX_SOLREF_LIM_0,
    TENDON_IDX_SOLREF_LIM_1,
    TENDON_IDX_SOLIMP_LIM_0,
    TENDON_IDX_SOLIMP_LIM_1,
    TENDON_IDX_SOLIMP_LIM_2,
    TENDON_IDX_SOLIMP_LIM_3,
    TENDON_IDX_SOLIMP_LIM_4,
    TENDON_IDX_IS_EQUALITY,
    TENDON_IDX_LENGTH_REF,
    TENDON_IDX_SOLREF_0,
    TENDON_IDX_SOLREF_1,
    TENDON_IDX_SOLIMP_0,
    TENDON_IDX_SOLIMP_1,
    TENDON_IDX_SOLIMP_2,
    TENDON_IDX_SOLIMP_3,
    TENDON_IDX_SOLIMP_4,
    JOINT_IDX_QPOS_ADR,
)
from ..dynamics.tendon import spatial_tendon_length_jac
from .scalar_rows import SROW_EQ_BILATERAL
from .constraint_data import solref_spring_damper

# How many (joint, coef) pairs a fixed tendon stores.
from ..fields import Dims, DimsLike, may_exist
#
# ⚠ WAS A LOCAL `4`, ON THE REASONING THAT REUSING `TENDON_MAX_SITES` WOULD BE
# COINCIDENTAL. The reasoning was sound and the outcome was not: the cap ended
# up written down in five places that all had to be changed together, and when
# dog arrived with an 11-joint tendon four of them were missed. They are the
# same quantity — how wide one tendon's wrap list may be — so they now share
# one constant, and `TENDON_MAX_SITES` is an alias of it rather than a
# coincidence.
comptime TENDON_MAX_JOINTS: Int = TENDON_MAX_WRAPS


comptime MJ_MINIMP: Float64 = 0.0001
comptime MJ_MAXIMP: Float64 = 0.9999


@always_inline
def _clamp_imp[DTYPE: DType](v: Scalar[DTYPE]) -> Scalar[DTYPE]:
    """MuJoCo clamps every impedance to [mjMINIMP, mjMAXIMP]
    (engine_core_constraint.c). Omitting this was bug 21."""
    if v < Scalar[DTYPE](MJ_MINIMP):
        return Scalar[DTYPE](MJ_MINIMP)
    if v > Scalar[DTYPE](MJ_MAXIMP):
        return Scalar[DTYPE](MJ_MAXIMP)
    return v


@always_inline
def _solimp[
    DTYPE: DType
](
    pen: Scalar[DTYPE],
    dmin: Scalar[DTYPE],
    dmax: Scalar[DTYPE],
    width: Scalar[DTYPE],
    midpoint: Scalar[DTYPE],
    power: Scalar[DTYPE],
) -> Scalar[DTYPE]:
    """MuJoCo's piecewise-power impedance ramp over the violation depth."""
    if dmin == dmax or width <= Scalar[DTYPE](0):
        return _clamp_imp[DTYPE](Scalar[DTYPE](0.5) * (dmin + dmax))
    var x = pen / width
    if x <= Scalar[DTYPE](0):
        return _clamp_imp[DTYPE](dmin)
    if x >= Scalar[DTYPE](1):
        return _clamp_imp[DTYPE](dmax)
    var y: Scalar[DTYPE]
    if power == Scalar[DTYPE](1):
        y = x
    elif x <= midpoint:
        y = pow(x, power) / pow(midpoint, power - Scalar[DTYPE](1))
    else:
        y = Scalar[DTYPE](1) - pow(Scalar[DTYPE](1) - x, power) / pow(
            Scalar[DTYPE](1) - midpoint, power - Scalar[DTYPE](1)
        )
    return _clamp_imp[DTYPE](dmin + y * (dmax - dmin))


def build_tendon_limit_rows[
    DTYPE: DType,
    V_CAP: Int,
    E_CAP: Int,
    BATCH: Int,
    D: DimsLike,
    L_QPOS: TensorLayout,
    L_QVEL: TensorLayout,
    L_TENDONS: TensorLayout,
    L_SITES: TensorLayout,
    L_GEOMS: TensorLayout,
    L_BODIES: TensorLayout,
    L_JOINTS: TensorLayout,
    L_MMETA: TensorLayout,
    L_SUBTREE_COM: TensorLayout,
    L_CDOF: TensorLayout,
    L_XQUAT: TensorLayout,
    L_M_INV: TensorLayout,
](
    env: Int,
    dims: D,
    qpos: TileTensor[DTYPE, L_QPOS, MutAnyOrigin],
    qvel: TileTensor[DTYPE, L_QVEL, MutAnyOrigin],
    tendons: TileTensor[DTYPE, L_TENDONS, MutAnyOrigin],
    sites: TileTensor[DTYPE, L_SITES, MutAnyOrigin],
    geoms: TileTensor[DTYPE, L_GEOMS, MutAnyOrigin],
    bodies: TileTensor[DTYPE, L_BODIES, MutAnyOrigin],
    joints: TileTensor[DTYPE, L_JOINTS, MutAnyOrigin],
    mmeta: TileTensor[DTYPE, L_MMETA, MutAnyOrigin],
    subtree_com: TileTensor[DTYPE, L_SUBTREE_COM, MutAnyOrigin],
    cdof: TileTensor[DTYPE, L_CDOF, MutAnyOrigin],
    xpos: TileTensor[DTYPE, L_SUBTREE_COM, MutAnyOrigin],
    xquat: TileTensor[DTYPE, L_XQUAT, MutAnyOrigin],
    m_inv: TileTensor[DTYPE, L_M_INV, MutAnyOrigin],
    mut Je: Scratch[Scalar[DTYPE], E_CAP * V_CAP],
    mut De: Scratch[Scalar[DTYPE], E_CAP],
    mut bias_e: Scratch[Scalar[DTYPE], E_CAP],
    # ⚠⚠ THE LIVE ROW BUDGET, NOT `E_CAP`. `E_CAP` is the COMPTIME array
    # capacity and is **0** on a dynamic dimension provider — the studio,
    # every runtime-loaded model, every Menagerie scene. Guarding the emit
    # loop with it meant `0 >= 0` on the first tendon, so every limit and
    # every tendon equality was dropped and the solver ran an
    # unconstrained system. The caller documents this exact trap twenty
    # lines above the call (`newton_solve.mojo:836`, "TWO SPELLINGS OF THE
    # ROW BUDGET") for the rows it builds INLINE, and then handed the cap
    # to the two builders it calls out to. A comptime caller passes its
    # cap here, which is also its live budget; a runtime caller passes the
    # live count. See `test_tendon_rows_live_budget_vs_mujoco`.
    max_rows: Int,
    mut num_edges: Int,
):
    """Append a row per violated tendon limit side to the pyramidal edge list.

    Rows are ONE-SIDED (force >= 0), so the caller leaves `kind_e` at
    SROW_LIMIT and `R_e`/`floss_e` at 0, exactly as it does for joint limits.
    """
    var nv = dims.get_nv()
    var nbody = dims.get_nbody()
    var njoint = dims.get_njoint()
    var nsite = dims.get_nsite()
    var ntendon = dims.get_ntendon()
    comptime if not may_exist[D.NTENDON]():
        return

    var nten = Int(rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_NTENDON]))
    if nten == 0:
        return
    if nten > ntendon:
        nten = ntendon

    var tJ = Scratch[Scalar[DTYPE], V_CAP](nv, fill=Scalar[DTYPE](0))

    for t in range(nten):
        if Int(rebind[Scalar[DTYPE]](tendons[t, TENDON_IDX_LIMITED])) == 0:
            continue

        # --- length + moment arm ------------------------------------------
        var ten_len = Scalar[DTYPE](0)
        var kind = Int(rebind[Scalar[DTYPE]](tendons[t, TENDON_IDX_KIND]))
        if kind == TENDON_KIND_SPATIAL:
            ten_len = spatial_tendon_length_jac[
                DTYPE, V_CAP, BATCH
            ](
                env, t, dims, tendons, sites, geoms, bodies, joints, mmeta,
                subtree_com,
                cdof, xpos, xquat, tJ,
            )
        else:
            # ⚠⚠ A FIXED TENDON'S LIMIT IS A ROW TOO. This branch was
            # `continue` until 2026-09-06 ("this builder is not given
            # qpos"): every `<fixed limited="true">` tendon fell through and
            # its range never produced a row, on every solver path, while the
            # equality builder below handled both kinds. MuJoCo's
            # `mj_instantiateLimit` does not distinguish them — `ten_length`
            # and `ten_J` are whatever `mj_tendon` computed, `sum(coef*qpos)`
            # and `coef` per dof for a fixed one. ToddlerBot's waist is
            # coupled by two such tendons with `range="-0.001 0.001"`; without
            # their rows `waist_yaw` walked 4e-02 from MuJoCo in 100 steps
            # (PERFORMANCE.md §13.29). Same length/Jacobian as
            # `build_tendon_equality_rows`.
            for i in range(nv):
                tJ[i] = Scalar[DTYPE](0)
            var njnt_f = Int(
                rebind[Scalar[DTYPE]](tendons[t, TENDON_IDX_NUM_JOINTS])
            )
            for k in range(TENDON_MAX_JOINTS):
                if k >= njnt_f:
                    break
                var jf = Int(
                    rebind[Scalar[DTYPE]](tendons[t, TENDON_IDX_JOINT_0 + k])
                )
                if jf < 0 or jf >= njoint:
                    continue
                var coef_f = rebind[Scalar[DTYPE]](
                    tendons[t, TENDON_IDX_COEF_0 + k]
                )
                var qadr_f = Int(
                    rebind[Scalar[DTYPE]](joints[jf, JOINT_IDX_QPOS_ADR])
                )
                var dadr_f = Int(
                    rebind[Scalar[DTYPE]](joints[jf, JOINT_IDX_DOF_ADR])
                )
                ten_len += coef_f * rebind[Scalar[DTYPE]](qpos[env, qadr_f])
                tJ[dadr_f] = tJ[dadr_f] + coef_f

        # --- ten_J . qvel --------------------------------------------------
        var ten_vel = Scalar[DTYPE](0)
        for i in range(nv):
            ten_vel += tJ[i] * rebind[Scalar[DTYPE]](qvel[env, i])

        # --- K = J M^-1 J^T at the CURRENT pose ----------------------------
        # ⚠ NO `K = J M^-1 J^T` HERE ANY MORE. MuJoCo prices a tendon row with
        # `tendon_invweight0` (`mj_diagApprox`, engine_core_constraint.c:1720)
        # and sets `efc_D = 1/R` outright (:2259). The M^-1 product this formed
        # fed a round-trip of R — `1/(1/(K+R)) - K` — and nothing else, and it
        # was one of the last readers keeping the integrator forming a dense
        # `M^-1` under Newton (`PERFORMANCE.md` §13).

        var rmin = rebind[Scalar[DTYPE]](tendons[t, TENDON_IDX_RANGE_MIN])
        var rmax = rebind[Scalar[DTYPE]](tendons[t, TENDON_IDX_RANGE_MAX])
        var margin = rebind[Scalar[DTYPE]](tendons[t, TENDON_IDX_MARGIN])

        var tc = rebind[Scalar[DTYPE]](tendons[t, TENDON_IDX_SOLREF_LIM_0])
        var dr = rebind[Scalar[DTYPE]](tendons[t, TENDON_IDX_SOLREF_LIM_1])
        var dmin = rebind[Scalar[DTYPE]](tendons[t, TENDON_IDX_SOLIMP_LIM_0])
        var dmax = rebind[Scalar[DTYPE]](tendons[t, TENDON_IDX_SOLIMP_LIM_1])
        var width = rebind[Scalar[DTYPE]](tendons[t, TENDON_IDX_SOLIMP_LIM_2])
        var midpt = rebind[Scalar[DTYPE]](tendons[t, TENDON_IDX_SOLIMP_LIM_3])
        var power = rebind[Scalar[DTYPE]](tendons[t, TENDON_IDX_SOLIMP_LIM_4])

        # ⚠ THROUGH THE SHARED HELPER, NOT INLINE. `solref_spring_damper`
        # carries MuJoCo's REFSAFE clamp `timeconst = max(timeconst,
        # 2*timestep)` (engine_core_constraint.c:2029) and the negative
        # (direct) solref form; the inline `1/(d² tc² dr²)` this replaces had
        # neither. It is what put dm_control's quadruped at 2.9e-2 on the
        # three-tree board (2026-09-06): its coupling equalities say
        # `solref=".005 .5"` at `timestep="0.005"`, MuJoCo clamps that
        # timeconst to 0.01 before it becomes a stiffness, and the inline
        # formula built a row four times too stiff. With the clamp the scene
        # tracks MuJoCo to 4.9e-17 over 20 steps. Same rule at both tendon
        # sites (limit and equality) — a rule written inline twice drifts.
        var _kb = solref_spring_damper[DTYPE](
            tc, dr, dmax, rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_TIMESTEP])
        )
        var K_spring = _kb[0]
        var B_damp = _kb[1]

        var diag = rebind[Scalar[DTYPE]](tendons[t, TENDON_IDX_INVWEIGHT0])

        # side = -1 (lower), +1 (upper); `sign` below is MuJoCo's -side, so
        # the row's Jacobian is `sign * ten_J`.
        for s in range(2):
            if num_edges >= max_rows:
                break
            var side = Scalar[DTYPE](-1) if s == 0 else Scalar[DTYPE](1)
            var bound = rmin if s == 0 else rmax
            var dist = side * (bound - ten_len)
            if dist >= margin:
                continue

            var sign = -side
            # ⚠ AUD-38: `margin - dist`, MuJoCo's `-(pos - margin)`
            # (engine_core_constraint.c:2109, :3255). This was `-dist`: inside
            # the band (0 < dist < margin) the impedance saturated at dmin and
            # the reference acceleration pulled the tendon INTO the limit.
            var pen = margin - dist
            var v_lim = sign * ten_vel

            var imp = _solimp[DTYPE](pen, dmin, dmax, width, midpt, power)
            var R = (Scalar[DTYPE](1) - imp) / imp * diag
            if R < Scalar[DTYPE](1e-14):
                R = Scalar[DTYPE](1e-14)

            for i in range(nv):
                Je[num_edges * nv + i] = sign * tJ[i]

            De[num_edges] = Scalar[DTYPE](1) / R
            bias_e[num_edges] = B_damp * v_lim - K_spring * imp * pen
            num_edges += 1


def build_tendon_equality_rows[
    DTYPE: DType,
    V_CAP: Int,
    E_CAP: Int,
    BATCH: Int,
    D: DimsLike,
    L_QPOS: TensorLayout,
    L_QVEL: TensorLayout,
    L_TENDONS: TensorLayout,
    L_SITES: TensorLayout,
    L_GEOMS: TensorLayout,
    L_BODIES: TensorLayout,
    L_JOINTS: TensorLayout,
    L_MMETA: TensorLayout,
    L_SUBTREE_COM: TensorLayout,
    L_CDOF: TensorLayout,
    L_XQUAT: TensorLayout,
    L_M_INV: TensorLayout,
](
    env: Int,
    dims: D,
    qpos: TileTensor[DTYPE, L_QPOS, MutAnyOrigin],
    qvel: TileTensor[DTYPE, L_QVEL, MutAnyOrigin],
    tendons: TileTensor[DTYPE, L_TENDONS, MutAnyOrigin],
    sites: TileTensor[DTYPE, L_SITES, MutAnyOrigin],
    geoms: TileTensor[DTYPE, L_GEOMS, MutAnyOrigin],
    bodies: TileTensor[DTYPE, L_BODIES, MutAnyOrigin],
    joints: TileTensor[DTYPE, L_JOINTS, MutAnyOrigin],
    mmeta: TileTensor[DTYPE, L_MMETA, MutAnyOrigin],
    subtree_com: TileTensor[DTYPE, L_SUBTREE_COM, MutAnyOrigin],
    cdof: TileTensor[DTYPE, L_CDOF, MutAnyOrigin],
    xpos: TileTensor[DTYPE, L_SUBTREE_COM, MutAnyOrigin],
    xquat: TileTensor[DTYPE, L_XQUAT, MutAnyOrigin],
    m_inv: TileTensor[DTYPE, L_M_INV, MutAnyOrigin],
    mut Je: Scratch[Scalar[DTYPE], E_CAP * V_CAP],
    mut De: Scratch[Scalar[DTYPE], E_CAP],
    mut bias_e: Scratch[Scalar[DTYPE], E_CAP],
    mut kind_e: Scratch[Int, E_CAP],
    # ⚠⚠ THE LIVE ROW BUDGET, NOT `E_CAP`. `E_CAP` is the COMPTIME array
    # capacity and is **0** on a dynamic dimension provider — the studio,
    # every runtime-loaded model, every Menagerie scene. Guarding the emit
    # loop with it meant `0 >= 0` on the first tendon, so every limit and
    # every tendon equality was dropped and the solver ran an
    # unconstrained system. The caller documents this exact trap twenty
    # lines above the call (`newton_solve.mojo:836`, "TWO SPELLINGS OF THE
    # ROW BUDGET") for the rows it builds INLINE, and then handed the cap
    # to the two builders it calls out to. A comptime caller passes its
    # cap here, which is also its live budget; a runtime caller passes the
    # live count. See `test_tendon_rows_live_budget_vs_mujoco`.
    max_rows: Int,
    mut num_edges: Int,
):
    """Append one BILATERAL row per `<equality><tendon>` to the pyramidal edge
    list — `mjEQ_TENDON`, on a FIXED or a SPATIAL tendon.

    WHY THIS IS A ROW AND NOT A POST-PASS. `_tendon_env` solved these after
    the Newton solve returned, as a separate Gauss-Seidel sweep. With only
    equality rows live that converges — a quadruped in free flight matched
    MuJoCo to 1e-7. With contacts live it does not: both act on the same leg
    dofs, the contact force is computed as if the coupling were absent, and a
    standing quadruped's toes carried about a THIRD of the ground reaction
    force MuJoCo computes. Exactly the finding that made joint limits and
    frictionloss rows (see constraints/scalar_rows.mojo), one constraint type
    later.

    Bilateral means the row is ALWAYS active and never clamped: `kind_e` is
    `SROW_EQ_BILATERAL`, whose state is unconditionally QUADRATIC, and
    `R_e`/`floss_e` stay 0 because only the box branch reads them.

    BOTH KINDS, since 2026-08-12. This used to skip SPATIAL tendons, on the
    written grounds that "the `_tendon_env` post-pass still covers it". IT DID
    NOT. `_tendon_env` has no spatial branch at all — it computes a FIXED
    tendon's length from joint coefficients, and a spatial tendon has
    `num_joints == 0`, so it built a row with a ZERO Jacobian, converged it,
    and applied `qacc += M^-1 J^T dlambda == 0`. A spatial `<equality><tendon>`
    was therefore not constrained ANYWHERE in the engine: on the specimen in
    `tests/physics3d/test_spatial_tendon_equality_vs_mujoco.mojo` (a bob whose
    only support is the constraint) we reproduced MuJoCo's FREE FALL to 13
    digits, -3.1470479999999923 against a true answer of -0.000367181842.

    Four comments asserted the handoff and they cited each other in a ring:
    this one skipped spatial BECAUSE the post-pass covered it, and the
    post-pass's guard tested `kind != SPATIAL` specifically TO LET IT THROUGH,
    into code that could not express it. Nobody ever ran the pair — no model
    in the tree pairs a spatial tendon with an equality, so the whole
    arrangement was inert. See `docs/DM_CONTROL_PORT_PHASE2.md` section 25.

    The residual is the same for both kinds (`ten_length - tendon_length0`);
    only the length and its moment arm differ, and `spatial_tendon_length_jac`
    already computes those for the limit builder above.
    """
    var nq = dims.get_nq()
    var nv = dims.get_nv()
    var nbody = dims.get_nbody()
    var njoint = dims.get_njoint()
    var nsite = dims.get_nsite()
    var ntendon = dims.get_ntendon()
    comptime if not may_exist[D.NTENDON]():
        return

    var nten = Int(rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_NTENDON]))
    if nten == 0:
        return
    if nten > ntendon:
        nten = ntendon

    var eqJ = Scratch[Scalar[DTYPE], V_CAP](nv, fill=Scalar[DTYPE](0))

    for t in range(nten):
        if Int(rebind[Scalar[DTYPE]](tendons[t, TENDON_IDX_IS_EQUALITY])) == 0:
            continue
        if num_edges >= max_rows:
            break

        # --- length, rate and moment arm, per kind -------------------------
        for i in range(nv):
            Je[num_edges * nv + i] = Scalar[DTYPE](0)
        var ten_len = Scalar[DTYPE](0)

        if (
            Int(rebind[Scalar[DTYPE]](tendons[t, TENDON_IDX_KIND]))
            == TENDON_KIND_SPATIAL
        ):
            # The polyline length and its dense moment arm — the same call the
            # limit builder above makes, and the piece whose absence made this
            # constraint a no-op.
            ten_len = spatial_tendon_length_jac[
                DTYPE, V_CAP, BATCH
            ](
                env, t, dims, tendons, sites, geoms, bodies, joints, mmeta,
                subtree_com,
                cdof, xpos, xquat, eqJ,
            )
            for i in range(nv):
                Je[num_edges * nv + i] = eqJ[i]
        else:
            var njnt = Int(
                rebind[Scalar[DTYPE]](tendons[t, TENDON_IDX_NUM_JOINTS])
            )
            for k in range(TENDON_MAX_JOINTS):
                if k >= njnt:
                    break
                var j = Int(
                    rebind[Scalar[DTYPE]](tendons[t, TENDON_IDX_JOINT_0 + k])
                )
                if j < 0 or j >= njoint:
                    continue
                var coef = rebind[Scalar[DTYPE]](
                    tendons[t, TENDON_IDX_COEF_0 + k]
                )
                var qadr = Int(
                    rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_QPOS_ADR])
                )
                var dadr = Int(
                    rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_DOF_ADR])
                )
                ten_len += coef * rebind[Scalar[DTYPE]](qpos[env, qadr])
                Je[num_edges * nv + dadr] = (
                    Je[num_edges * nv + dadr] + coef
                )

        # `ten_vel` comes off the ASSEMBLED row rather than being accumulated
        # alongside it, so the two kinds share one expression. Identical for a
        # fixed tendon, which had `J[dof] = coef` by construction.
        var ten_vel = Scalar[DTYPE](0)
        for i in range(nv):
            ten_vel += Je[num_edges * nv + i] * rebind[Scalar[DTYPE]](
                qvel[env, i]
            )

        # ⚠ NO `K = J M^-1 J^T` HERE ANY MORE. MuJoCo prices a tendon row with
        # `tendon_invweight0` (`mj_diagApprox`, engine_core_constraint.c:1720)
        # and sets `efc_D = 1/R` outright (:2259). The M^-1 product this formed
        # fed a round-trip of R — `1/(1/(K+R)) - K` — and nothing else, and it
        # was one of the last readers keeping the integrator forming a dense
        # `M^-1` under Newton (`PERFORMANCE.md` §13).

        # `pos` is SIGNED for a bilateral row — not a penetration depth. Only
        # the impedance lookup takes its magnitude.
        var pos_err = ten_len - rebind[Scalar[DTYPE]](
            tendons[t, TENDON_IDX_LENGTH_REF]
        )

        var tc = rebind[Scalar[DTYPE]](tendons[t, TENDON_IDX_SOLREF_0])
        var dr = rebind[Scalar[DTYPE]](tendons[t, TENDON_IDX_SOLREF_1])
        var dmin = rebind[Scalar[DTYPE]](tendons[t, TENDON_IDX_SOLIMP_0])
        var dmax = rebind[Scalar[DTYPE]](tendons[t, TENDON_IDX_SOLIMP_1])
        var width = rebind[Scalar[DTYPE]](tendons[t, TENDON_IDX_SOLIMP_2])
        var midpt = rebind[Scalar[DTYPE]](tendons[t, TENDON_IDX_SOLIMP_3])
        var power = rebind[Scalar[DTYPE]](tendons[t, TENDON_IDX_SOLIMP_4])

        # ⚠ THROUGH THE SHARED HELPER, NOT INLINE. `solref_spring_damper`
        # carries MuJoCo's REFSAFE clamp `timeconst = max(timeconst,
        # 2*timestep)` (engine_core_constraint.c:2029) and the negative
        # (direct) solref form; the inline `1/(d² tc² dr²)` this replaces had
        # neither. It is what put dm_control's quadruped at 2.9e-2 on the
        # three-tree board (2026-09-06): its coupling equalities say
        # `solref=".005 .5"` at `timestep="0.005"`, MuJoCo clamps that
        # timeconst to 0.01 before it becomes a stiffness, and the inline
        # formula built a row four times too stiff. With the clamp the scene
        # tracks MuJoCo to 4.9e-17 over 20 steps. Same rule at both tendon
        # sites (limit and equality) — a rule written inline twice drifts.
        var _kb = solref_spring_damper[DTYPE](
            tc, dr, dmax, rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_TIMESTEP])
        )
        var K_spring = _kb[0]
        var B_damp = _kb[1]

        var pen = pos_err if pos_err >= Scalar[DTYPE](0) else -pos_err
        var imp = _solimp[DTYPE](pen, dmin, dmax, width, midpt, power)

        # diagApprox is the tendon's OWN invweight0 — one number, not a sum
        # over its joints (engine_core_constraint.c:1091). See
        # `_tendon_env`'s note for the bug that rule replaced.
        var diag = rebind[Scalar[DTYPE]](tendons[t, TENDON_IDX_INVWEIGHT0])
        var R = (Scalar[DTYPE](1) - imp) / imp * diag
        if R < Scalar[DTYPE](1e-14):
            R = Scalar[DTYPE](1e-14)

        De[num_edges] = Scalar[DTYPE](1) / R
        bias_e[num_edges] = B_damp * ten_vel + K_spring * imp * pos_err
        kind_e[num_edges] = SROW_EQ_BILATERAL
        num_edges += 1
