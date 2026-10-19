"""CG constraint solve over per-field tensors (Stage-S, single-source).

Per-field port of `CGSolver.solve_gpu` (solver/cg_solver.mojo:503) — the
M-preconditioned nonlinear conjugate-gradient primal solver (Polak-Ribiere
beta, Armijo line-search), ELLIPTIC 3-zone friction cone. Arithmetic,
iteration order, and constants (CG_ITER_GPU=100, CG_TOL_GPU=1e-6,
LINESEARCH_ITER=10, ARMIJO=1e-4, PRIMAL_MINVAL_GPU=1e-12) are verbatim from
the legacy kernel.

CG differs from Newton only in the inner recurrence: Newton builds and
Cholesky-factorizes the full Hessian H = M + JᵀDJ each iteration; CG
Cholesky-factorizes M ONCE (the preconditioner) and forms the search
direction as -M⁻¹grad + β·search_old (β via Polak-Ribiere). The contact
setup (normal + friction precompute), the 3-zone cone force update, and the
limits/equality/tendon tail are SHARED with the fields-Newton elliptic path,
so this port reuses the exact same helpers
(`_precompute_contact_normal` / `_precompute_contact_friction`
/ `_limits_env` / `_equality_env` / `_tendon_env`) that
the golden-validated fields-Newton solve uses.

Structural transformation (same as newton_solve.mojo): the legacy
kernel is 2D-threaded (thread_y = contact slot) with barriers; this port
SERIALIZES it per env. The init + normal-precompute parallel phases become
`for contact_tid in range(max_contacts)` loops; the friction phase becomes
`for contact_tid in range(nc)`. All phases write disjoint per-contact slots,
so serialization is value-identical. The CG core after the legacy
`if not valid_env or contact_tid != 0: return` gate runs single-thread and
is ported as-is.

qfrc_smooth convention: the legacy GPU kernel reads qfrc_smooth from the
`ws_fnet` workspace; this port captures qfrc_smooth = Ma = M·qacc_smooth at
entry (qacc = qacc_smooth on entry), identical to the fields-Newton elliptic
solve (`qfrc_sm[i] = Ma[i]`). Both are the smooth net force at the entry
point; using Ma keeps CG solving the *same* convex problem as fields-Newton
(so the two agree to solver tolerance — the gate) and avoids plumbing a
separate fnet operand.

Workspace: uses a PREFIX (35*MC + 6*MC*NV) of the PGS-sized `cscratch.solver`
tensor (81*MC + 12*MC*NV), exactly as the fields-Newton solve does.

ELLIPTIC cone only — the legacy `CGSolver` has no pyramidal path. CONE_TYPE
is kept in the signature for parity with `solve_newton`; the friction
precompute is invoked with it (the shared builder branches internally).
"""

from std.math import sqrt, abs
from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceContext
from std.sys import default_accelerator
from layout import TileTensor, TensorLayout
from noeira.core.tiles import RowMajor, Tile

from ..types import _max_one, ConeType
from .cholesky import chol_factor_inline, chol_solve_inline
from ..constraints.solver_ws import (
    ws_budget,
    _max_one_rt,
    ws_c_dist,
    ws_pos_bias,
    ws_j_n,
    ws_ell_jt,
    ws_ell_mu,
    ws_ell_dn,
    ws_ell_dt,
    ws_ell_bt,
    ws_ell_ntc,
    ws_end_elliptic,
)
from ..constraints.contact_solve import (
    _init_common_normal_ws,
    _precompute_contact_normal,
    _precompute_contact_friction,
)
from ..constraints.limits import _limits_env
from ..constraints.friction_dof import _friction_env
from ..constraints.scalar_rows import (
    build_scalar_rows,
    max_scalar_rows,
    max_scalar_rows_cap,
    scalar_row_state,
    scalar_row_force,
    scalar_row_cost,
)
from ..constraints.equality_tendon import (
    _equality_env,
    _tendon_env,
)
from ..fields import (
    Data,
    Model,
    DynamicsScratch,
    ContactScratch,
    Dims,
    DimsLike,
    AsStatic,
    DYN1,
    DYN2,
    rl1,
    rl2,
)
from ..fields.scratch import Scratch, cap
from ..fields.dims import may_exist
from ..gpu.constants import (
    MODEL_META_IDX_TIMESTEP,
    MODEL_BODY_SIZE,
    MODEL_JOINT_SIZE,
    MODEL_META_SIZE,
    MODEL_EQ_SIZE,
    MODEL_TENDON_SIZE,
    MODEL_SITE_SIZE,
    MODEL_GEOM_SIZE,
    METADATA_SIZE,
    CONTACT_SIZE,
    CONTACT_IDX_FORCE_N,
    CONTACT_IDX_FORCE_T1,
    CONTACT_IDX_FORCE_T2,
    META_IDX_NUM_CONTACTS,
    MODEL_META_IDX_SOLREF_CONTACT_0,
    MODEL_META_IDX_SOLREF_CONTACT_1,
    MODEL_META_IDX_SOLIMP_CONTACT_0,
    MODEL_META_IDX_SOLIMP_CONTACT_1,
    MODEL_META_IDX_SOLIMP_CONTACT_2,
    MODEL_META_IDX_SOLIMP_CONTACT_3,
    MODEL_META_IDX_SOLIMP_CONTACT_4,
    MODEL_META_IDX_IMPRATIO,
)

# One env per block (see newton_solve.mojo:115): the per-env CG solve
# stack-allocates a modest local frame (a handful of V_SIZE/M_SIZE arrays),
# so one thread per block keeps the local reservation tight.
from ..constraints.constraint_data import solref_spring_damper

comptime CG_TPB: Int = 1


def _cg_solve_env[
    DTYPE: DType,
    CONE_TYPE: Int,
    BATCH: Int,
    SOLVER_WS: Int,
    D: DimsLike,
    L_QPOS: TensorLayout,
    L_QVEL: TensorLayout,
    L_XPOS: TensorLayout,
    L_XQUAT: TensorLayout,
    L_CONTACTS: TensorLayout,
    L_SMETA: TensorLayout,
    L_JOINTS: TensorLayout,
    L_BODIES: TensorLayout,
    L_MMETA: TensorLayout,
    L_EQUALITY: TensorLayout,
    L_TENDONS: TensorLayout,
    L_SITES: TensorLayout,
    L_GEOMS_W: TensorLayout,
    L_BODY_INVWEIGHT0: TensorLayout,
    L_DOF_INVWEIGHT0: TensorLayout,
    L_CDOF: TensorLayout,
    L_M: TensorLayout,
    L_SOLVER: TensorLayout,
](
    env: Int,
    dims: D,
    qpos: TileTensor[DTYPE, L_QPOS, MutAnyOrigin],
    qvel: TileTensor[DTYPE, L_QVEL, MutAnyOrigin],
    xpos: TileTensor[DTYPE, L_XPOS, MutAnyOrigin],
    xquat: TileTensor[DTYPE, L_XQUAT, MutAnyOrigin],
    subtree_com: TileTensor[DTYPE, L_XPOS, MutAnyOrigin],
    contacts: TileTensor[DTYPE, L_CONTACTS, MutAnyOrigin],
    smeta: TileTensor[DTYPE, L_SMETA, MutAnyOrigin],
    joints: TileTensor[DTYPE, L_JOINTS, MutAnyOrigin],
    bodies: TileTensor[DTYPE, L_BODIES, MutAnyOrigin],
    mmeta: TileTensor[DTYPE, L_MMETA, MutAnyOrigin],
    equality: TileTensor[DTYPE, L_EQUALITY, MutAnyOrigin],
    tendons: TileTensor[DTYPE, L_TENDONS, MutAnyOrigin],
    sites: TileTensor[DTYPE, L_SITES, MutAnyOrigin],
    # ⚠ FOR TENDON WRAP GEOMS ONLY — see `dynamics/tendon._geom_world_frame`.
    geoms_w: TileTensor[DTYPE, L_GEOMS_W, MutAnyOrigin],
    body_invweight0: TileTensor[DTYPE, L_BODY_INVWEIGHT0, MutAnyOrigin],
    dof_invweight0: TileTensor[DTYPE, L_DOF_INVWEIGHT0, MutAnyOrigin],
    cdof: TileTensor[DTYPE, L_CDOF, MutAnyOrigin],
    M: TileTensor[DTYPE, L_M, MutAnyOrigin],
    m_inv: TileTensor[DTYPE, L_M, MutAnyOrigin],
    qacc_constrained: TileTensor[DTYPE, L_QVEL, MutAnyOrigin],
    solver: TileTensor[DTYPE, L_SOLVER, MutAnyOrigin],
):
    """Full primal CG contact solve for one env (verbatim from
    CGSolver.solve_gpu, serialized per env — see module docstring)."""
    var nq = dims.get_nq()
    var nv = dims.get_nv()
    var nbody = dims.get_nbody()
    var njoint = dims.get_njoint()
    var max_contacts = dims.get_max_contacts()
    var ngeom = dims.get_ngeom()
    var nequality = dims.get_nequality()
    var ntendon = dims.get_ntendon()
    var nsite = dims.get_nsite()
    # ⚠ CAPS SIZE CONTAINERS AND NOTHING ELSE. Every offset, stride and
    # loop bound below reads the LIVE `max_contacts` / `nv` above. On the
    # static leg the two are the same integer, which is exactly why a mix-up
    # here is invisible to the whole 124-file suite.
    comptime MC_CAP = cap[D.CAP_MAX_CONTACTS]()
    comptime V_CAP = cap[D.CAP_NV]()
    comptime M_CAP = cap[D.CAP_NV * D.CAP_NV]()

    # Common normal block offsets (row-relative; the legacy `solver_ws_idx`
    # base is gone — same layout as newton_solve.mojo)
    var ws_c_dist_idx = ws_c_dist(max_contacts)
    var ws_pos_bias_idx = ws_pos_bias(max_contacts)
    var ws_J_n_idx = ws_j_n(max_contacts)

    # Primal-specific offsets, from `solver/elliptic_layout` — the ONE source
    # of truth shared with the producer and with `newton_solve`.
    #
    # ⚠ THIS PATH IS CONDIM-3 BY CONSTRUCTION AND SAYS SO. `solve_cg` takes no
    # `MAX_CONDIM` parameter — its cone math below is written for exactly two
    # tangents — so it pins the layout at 3 and `_precompute_contact_friction`
    # clamps each contact's `condim` to it. A condim-4 geom therefore loses its
    # torsional row HERE even though the Newton path now keeps it. That is not
    # a regression (this path never had it) but it IS a real difference between
    # the two solvers on the same model; Newton is the default, and the CG path
    # is a legacy alternate selected only by `<option solver="CG">`.
    comptime CG_MAX_CONDIM = 3
    var ws_Jt1_idx = ws_ell_jt(max_contacts, nv) + 0 * max_contacts * nv
    var ws_Jt2_idx = ws_ell_jt(max_contacts, nv) + 1 * max_contacts * nv
    var ws_mu_idx = ws_ell_mu(max_contacts, nv, CG_MAX_CONDIM)
    var ws_D_n_idx = ws_ell_dn(max_contacts, nv, CG_MAX_CONDIM)
    # Tangent row 0's `D`. Row 1 has its own slot now (the two differ once
    # slide friction is anisotropic), but this path's cone math carries a
    # single `D_f`, so it reads row 0's and applies it to both — which is
    # exact for the isotropic slide the contact record can express.
    var ws_D_f_idx = ws_ell_dt(max_contacts, nv, CG_MAX_CONDIM) + 0 * max_contacts
    var ws_bt1_idx = ws_ell_bt(max_contacts, nv, CG_MAX_CONDIM) + 0 * max_contacts
    var ws_bt2_idx = ws_ell_bt(max_contacts, nv, CG_MAX_CONDIM) + 1 * max_contacts
    var ws_ntc_idx = ws_ell_ntc(max_contacts, nv, CG_MAX_CONDIM)
    # Per-contact solve state. ⚠ THESE ARE LIVE HERE, unlike on the Newton
    # path, which keeps the same quantities in InlineArrays and left these
    # slots write-only; that is why they hang off `ell_end` rather than being
    # part of the shared layout.
    var CVS = ws_end_elliptic(max_contacts, nv, CG_MAX_CONDIM)
    var ws_jar_n_idx = CVS + 0 * max_contacts
    var ws_jar_t1_idx = CVS + 1 * max_contacts
    var ws_jar_t2_idx = CVS + 2 * max_contacts
    var ws_fn_idx = CVS + 3 * max_contacts
    var ws_ft1_idx = CVS + 4 * max_contacts
    var ws_ft2_idx = CVS + 5 * max_contacts
    var ws_cstate_idx = CVS + 6 * max_contacts
    # ⚠ Overrunning `SOLVER_WS` would not crash — `solver` is
    # `[BATCH, SOLVER_WS]`, so a write past the row lands in the NEXT ENV's
    # workspace. Caught at compile time instead.
    # ⚠ WAS A `comptime assert`, AND COULD NOT STAY ONE: `CVS` is a runtime
    # value on the dynamic leg, and a comptime assert over caps degrades to
    # `0 <= SOLVER_WS` — it would assert nothing on the leg that needs it.
    # Overrunning does not fault: `solver` is `[BATCH, SOLVER_WS]`, so a write
    # past the row lands in the NEXT env's workspace.
    if CVS + 7 * max_contacts > SOLVER_WS:
        print(
            "FATAL: the ELLIPTIC contact region plus CG's per-contact state"
            " does not fit ContactScratch.solver — raise SOLVER_WS in"
            " fields/contact_scratch.mojo and the four files that recompute it"
        )
        return

    # === Initialize workspace (legacy: parallel, one thread per slot) ===
    for contact_tid in range(max_contacts):
        _init_common_normal_ws[
            DTYPE](env, contact_tid, dims, solver)
        for d in range(nv):
            solver[env, ws_Jt1_idx + contact_tid * nv + d] = 0
            solver[env, ws_Jt2_idx + contact_tid * nv + d] = 0
        solver[env, ws_mu_idx + contact_tid] = 0
        solver[env, ws_D_n_idx + contact_tid] = 0
        solver[env, ws_D_f_idx + contact_tid] = 0
        solver[env, ws_bt1_idx + contact_tid] = 0
        solver[env, ws_bt2_idx + contact_tid] = 0
        solver[env, ws_ntc_idx + contact_tid] = 0
        solver[env, ws_jar_n_idx + contact_tid] = 0
        solver[env, ws_jar_t1_idx + contact_tid] = 0
        solver[env, ws_jar_t2_idx + contact_tid] = 0
        solver[env, ws_fn_idx + contact_tid] = 0
        solver[env, ws_ft1_idx + contact_tid] = 0
        solver[env, ws_ft2_idx + contact_tid] = 0
        solver[env, ws_cstate_idx + contact_tid] = 0

    # Read metadata (legacy `dt` read dropped — only the unused-arg limits
    # call consumed it)
    var nc = 0
    var K_spring: Scalar[DTYPE] = 0
    var B_damp: Scalar[DTYPE] = 0
    var si_dmin: Scalar[DTYPE] = 0
    var si_dmax: Scalar[DTYPE] = 0
    var si_width: Scalar[DTYPE] = 1
    var si_midpoint: Scalar[DTYPE] = Scalar[DTYPE](0.5)
    var si_power: Scalar[DTYPE] = Scalar[DTYPE](2.0)
    var impratio: Scalar[DTYPE] = Scalar[DTYPE](1.0)

    nc = Int(rebind[Scalar[DTYPE]](smeta[env, META_IDX_NUM_CONTACTS]))
    if nc > max_contacts:
        nc = max_contacts
    var sr_tc = rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_SOLREF_CONTACT_0])
    var sr_dr = rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_SOLREF_CONTACT_1])
    si_dmin = rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_SOLIMP_CONTACT_0])
    si_dmax = rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_SOLIMP_CONTACT_1])
    si_width = rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_SOLIMP_CONTACT_2])
    si_midpoint = rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_SOLIMP_CONTACT_3])
    si_power = rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_SOLIMP_CONTACT_4])
    if si_width < Scalar[DTYPE](1e-6):
        si_width = Scalar[DTYPE](1e-6)
    # MuJoCo clamps BOTH ends of solimp to [mjMINIMP, mjMAXIMP] before
    # interpolating (engine_core_constraint.c:1284-1287). The dmin floor is
    # the one that bites: R = (1-imp)/imp * diagApprox, so dmin=0 asks for an
    # infinitely soft contact at first touch. dm_control's finger is the first
    # model here to set it (`solimp="0 0.9 0.01"`); everything before used the
    # 0.9 default, which is why clamping only dmax survived.
    comptime MJ_MINIMP = Scalar[DTYPE](0.0001)
    comptime MJ_MAXIMP = Scalar[DTYPE](0.9999)
    if si_dmin < MJ_MINIMP:
        si_dmin = MJ_MINIMP
    elif si_dmin > MJ_MAXIMP:
        si_dmin = MJ_MAXIMP
    if si_dmax < MJ_MINIMP:
        si_dmax = MJ_MINIMP
    elif si_dmax > MJ_MAXIMP:
        si_dmax = MJ_MAXIMP
    if si_power < Scalar[DTYPE](1):
        si_power = Scalar[DTYPE](1)
    # solref -> (K, B), including MuJoCo's DIRECT form for a NEGATIVE
    # solref. See `constraints/constraint_data.solref_spring_damper` — the
    # formula lived in twelve copy-pasted sites until 2026-08-03.
    (K_spring, B_damp) = solref_spring_damper[DTYPE](
        sr_tc, sr_dr, si_dmax,
        rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_TIMESTEP]),
    )
    impratio = rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_IMPRATIO])
    if impratio < Scalar[DTYPE](1e-6):
        impratio = Scalar[DTYPE](1.0)

    # === PHASE 1: normal precompute (shared with Newton) ===
    for contact_tid in range(max_contacts):
        _precompute_contact_normal[
            DTYPE, V_CAP](
            env,
            contact_tid,
            nc,
            dims,
            qvel,
            subtree_com,
            contacts,
            joints,
            bodies,
            mmeta,
            body_invweight0,
            cdof,
            m_inv,
            qacc_constrained,
            solver,
            K_spring,
            B_damp,
            si_dmin,
            si_dmax,
            si_width,
            si_midpoint,
            si_power,
        )

    # === PHASE 2: Tangent frame + friction data (shared with Newton) ===
    for contact_tid in range(nc):
        _precompute_contact_friction[
            DTYPE,
            V_CAP, CONE_TYPE=CONE_TYPE, MAX_CONDIM=CG_MAX_CONDIM](
            env,
            contact_tid,
            nc,
            dims,
            qvel,
            subtree_com,
            contacts,
            joints,
            bodies,
            mmeta,
            cdof,
            solver,
            B_damp,
            impratio,
            K_spring,
        )

    # === SEQUENTIAL: primal CG (legacy: thread 0) ===
    comptime CG_ITER_GPU: Int = 100
    comptime CG_TOL_GPU: Float64 = 1e-6
    comptime LINESEARCH_ITER: Int = 10
    comptime ARMIJO: Float64 = 1e-4
    comptime PRIMAL_MINVAL_GPU: Float64 = 1e-12

    # === Step 2: Cholesky factorize M (preconditioner) ===
    var M_chol = Scratch[Scalar[DTYPE], M_CAP](
        nv * nv, uninitialized=Scalar[DTYPE](0)
    )
    var L_prec = Scratch[Scalar[DTYPE], M_CAP](
        nv * nv, uninitialized=Scalar[DTYPE](0)
    )
    for k in range(nv * nv):
        M_chol[k] = rebind[Scalar[DTYPE]](M[env, k])
    _ = chol_factor_inline[DTYPE, M_CAP](M_chol, L_prec, nv)

    # === Step 3: Initialize qacc, qacc_sm, qfrc_sm (= Ma), Ma, scale ===
    var qacc = Scratch[Scalar[DTYPE], V_CAP](
        nv, uninitialized=Scalar[DTYPE](0)
    )
    var qacc_sm = Scratch[Scalar[DTYPE], V_CAP](
        nv, uninitialized=Scalar[DTYPE](0)
    )
    var qfrc_sm = Scratch[Scalar[DTYPE], V_CAP](
        nv, uninitialized=Scalar[DTYPE](0)
    )
    var Ma = Scratch[Scalar[DTYPE], V_CAP](
        nv, uninitialized=Scalar[DTYPE](0)
    )
    var grad = Scratch[Scalar[DTYPE], V_CAP](
        nv, uninitialized=Scalar[DTYPE](0)
    )
    var Mgrad = Scratch[Scalar[DTYPE], V_CAP](
        nv, uninitialized=Scalar[DTYPE](0)
    )
    var gradold = Scratch[Scalar[DTYPE], V_CAP](
        nv, uninitialized=Scalar[DTYPE](0)
    )
    var Mgradold = Scratch[Scalar[DTYPE], V_CAP](
        nv, uninitialized=Scalar[DTYPE](0)
    )
    var search = Scratch[Scalar[DTYPE], V_CAP](
        nv, uninitialized=Scalar[DTYPE](0)
    )
    var Mv = Scratch[Scalar[DTYPE], V_CAP](
        nv, uninitialized=Scalar[DTYPE](0)
    )

    # ⚠ A COLD START, AND MuJoCo'S CG IS WARM-STARTED. `warmstart()`
    # (engine_forward.c:786) runs in `mj_fwdConstraint` BEFORE the solver is
    # chosen, so `mjSOL_CG` takes the same non-PGS branch `mjSOL_NEWTON` does:
    # start at the cheaper of `qacc_warmstart` and `qacc_smooth` by primal
    # cost. `newton_solve.mojo` implements that; this does not.
    #
    # NOT AN OVERSIGHT AND NOT DONE: no model in either reference tree selects
    # CG — every `<option solver=>` in Menagerie and dm_control says Newton —
    # so the gap is unreachable from anything gated here, and porting it needs
    # CG's own cost evaluation rather than a copy of the Newton block. It
    # costs a converged solve NOTHING (both starts reach the same fixed point)
    # and costs a TRUNCATED one a different iterate. `d.qacc_warmstart` is
    # written after every solve whatever the solver, so the field this would
    # read is already live and correct.
    for i in range(nv):
        var q_i = rebind[Scalar[DTYPE]](qacc_constrained[env, i])
        qacc[i] = q_i
        qacc_sm[i] = q_i

    # Ma = M * qacc
    for i in range(nv):
        var s: Scalar[DTYPE] = 0
        for j in range(nv):
            s += rebind[Scalar[DTYPE]](M[env, i * nv + j]) * qacc[j]
        Ma[i] = s

    # qfrc_sm = M * qacc_sm = Ma at entry (matches fields-Newton convention)
    for i in range(nv):
        qfrc_sm[i] = Ma[i]

    # Scale = 1/trace(M) for convergence check
    var scale: Scalar[DTYPE] = 0
    for i in range(nv):
        scale += rebind[Scalar[DTYPE]](M[env, i * nv + i])
    if scale > Scalar[DTYPE](1e-10):
        scale = Scalar[DTYPE](1.0) / scale
    else:
        scale = Scalar[DTYPE](1.0)

    # === Scalar rows: joint limits + dry-friction dofs ===
    # Rows of THIS system, not post-passes — see constraints/scalar_rows.mojo.
    comptime S_CAP = max_scalar_rows_cap[D.CAP_NV, D.CAP_NJOINT]()
    var max_srows = max_scalar_rows(nv, njoint)
    var sr_dof = Scratch[Int, S_CAP](max_srows, fill=0)
    var sr_kind = Scratch[Int, S_CAP](max_srows, fill=0)
    var sr_sign = Scratch[Scalar[DTYPE], S_CAP](max_srows, fill=Scalar[DTYPE](0))
    var sr_D = Scratch[Scalar[DTYPE], S_CAP](max_srows, fill=Scalar[DTYPE](0))
    var sr_R = Scratch[Scalar[DTYPE], S_CAP](max_srows, fill=Scalar[DTYPE](0))
    var sr_bias = Scratch[Scalar[DTYPE], S_CAP](max_srows, fill=Scalar[DTYPE](0))
    var sr_floss = Scratch[Scalar[DTYPE], S_CAP](max_srows, fill=Scalar[DTYPE](0))
    var ns = build_scalar_rows[DTYPE, S_CAP](
        env, dims, qpos, qvel, joints, mmeta, dof_invweight0, m_inv,
        sr_dof, sr_kind, sr_sign, sr_D, sr_R, sr_bias, sr_floss,
    )
    var sr_jar = Scratch[Scalar[DTYPE], S_CAP](max_srows, fill=Scalar[DTYPE](0))
    var sr_f = Scratch[Scalar[DTYPE], S_CAP](max_srows, fill=Scalar[DTYPE](0))
    var sr_st = Scratch[Int, S_CAP](max_srows, fill=0)
    var sr_Js = Scratch[Scalar[DTYPE], S_CAP](max_srows, fill=Scalar[DTYPE](0))

    # === Step 4: Compute initial jar and forces via 3-zone cone logic ===
    for c in range(nc):
        if rebind[Scalar[DTYPE]](solver[env, ws_c_dist_idx + c]) >= Scalar[
            DTYPE
        ](0):
            solver[env, ws_fn_idx + c] = 0
            solver[env, ws_ft1_idx + c] = 0
            solver[env, ws_ft2_idx + c] = 0
            solver[env, ws_cstate_idx + c] = 0
            continue

        var jar_n_c: Scalar[DTYPE] = rebind[Scalar[DTYPE]](
            solver[env, ws_pos_bias_idx + c]
        )
        var jar_t1_c: Scalar[DTYPE] = rebind[Scalar[DTYPE]](
            solver[env, ws_bt1_idx + c]
        )
        var jar_t2_c: Scalar[DTYPE] = rebind[Scalar[DTYPE]](
            solver[env, ws_bt2_idx + c]
        )
        for i in range(nv):
            var qa_i = qacc[i]
            jar_n_c += (
                rebind[Scalar[DTYPE]](solver[env, ws_J_n_idx + c * nv + i])
                * qa_i
            )
            jar_t1_c += (
                rebind[Scalar[DTYPE]](solver[env, ws_Jt1_idx + c * nv + i])
                * qa_i
            )
            jar_t2_c += (
                rebind[Scalar[DTYPE]](solver[env, ws_Jt2_idx + c * nv + i])
                * qa_i
            )
        solver[env, ws_jar_n_idx + c] = jar_n_c
        solver[env, ws_jar_t1_idx + c] = jar_t1_c
        solver[env, ws_jar_t2_idx + c] = jar_t2_c

        var mu_c = rebind[Scalar[DTYPE]](solver[env, ws_mu_idx + c])
        var D_n_c = rebind[Scalar[DTYPE]](solver[env, ws_D_n_idx + c])
        var D_f_c = rebind[Scalar[DTYPE]](solver[env, ws_D_f_idx + c])
        var T = sqrt(jar_t1_c * jar_t1_c + jar_t2_c * jar_t2_c)
        var T_safe = T
        if T_safe < Scalar[DTYPE](PRIMAL_MINVAL_GPU):
            T_safe = Scalar[DTYPE](PRIMAL_MINVAL_GPU)

        if jar_n_c >= mu_c * T_safe:
            solver[env, ws_fn_idx + c] = 0
            solver[env, ws_ft1_idx + c] = 0
            solver[env, ws_ft2_idx + c] = 0
            solver[env, ws_cstate_idx + c] = 0
        elif mu_c * jar_n_c + T <= Scalar[DTYPE](0):
            solver[env, ws_fn_idx + c] = -D_n_c * jar_n_c
            solver[env, ws_ft1_idx + c] = -D_f_c * jar_t1_c
            solver[env, ws_ft2_idx + c] = -D_f_c * jar_t2_c
            solver[env, ws_cstate_idx + c] = 1
        else:
            var s = jar_n_c - mu_c * T_safe
            var Dm = D_n_c / (Scalar[DTYPE](1.0) + mu_c * mu_c)
            solver[env, ws_fn_idx + c] = -Dm * s
            solver[env, ws_ft1_idx + c] = Dm * mu_c * s * jar_t1_c / T_safe
            solver[env, ws_ft2_idx + c] = Dm * mu_c * s * jar_t2_c / T_safe
            solver[env, ws_cstate_idx + c] = 2

    for s_i in range(ns):
        var jr = sr_bias[s_i] + sr_sign[s_i] * qacc[sr_dof[s_i]]
        sr_jar[s_i] = jr
        var st = scalar_row_state[DTYPE](
            sr_kind[s_i], jr, sr_R[s_i], sr_floss[s_i]
        )
        sr_st[s_i] = st
        sr_f[s_i] = scalar_row_force[DTYPE](st, jr, sr_D[s_i], sr_floss[s_i])

    # === Step 5: Compute initial gradient and preconditioned gradient ===
    var grad_norm_sq: Scalar[DTYPE] = 0
    for i in range(nv):
        var g: Scalar[DTYPE] = Ma[i] - qfrc_sm[i]
        for c in range(nc):
            var cs = Int(rebind[Scalar[DTYPE]](solver[env, ws_cstate_idx + c]))
            if cs == 0:
                continue
            g -= (
                rebind[Scalar[DTYPE]](solver[env, ws_J_n_idx + c * nv + i])
                * rebind[Scalar[DTYPE]](solver[env, ws_fn_idx + c])
                + rebind[Scalar[DTYPE]](solver[env, ws_Jt1_idx + c * nv + i])
                * rebind[Scalar[DTYPE]](solver[env, ws_ft1_idx + c])
                + rebind[Scalar[DTYPE]](solver[env, ws_Jt2_idx + c * nv + i])
                * rebind[Scalar[DTYPE]](solver[env, ws_ft2_idx + c])
            )
        for s_i in range(ns):
            if sr_dof[s_i] == i:
                g -= sr_sign[s_i] * sr_f[s_i]
        grad[i] = g
        grad_norm_sq += g * g

    # Initial preconditioned gradient: Mgrad = M⁻¹ · grad (Cholesky solve)
    chol_solve_inline[DTYPE, M_CAP, V_CAP](L_prec, grad, Mgrad, nv)

    # Initial search direction: search = -Mgrad
    for i in range(nv):
        search[i] = -Mgrad[i]

    # === Step 6: CG iteration loop ===
    for _iter in range(CG_ITER_GPU):
        # Convergence check
        if scale * sqrt(grad_norm_sq) < Scalar[DTYPE](CG_TOL_GPU):
            break

        # Mv = M * search (for linesearch Gauss cost)
        for i in range(nv):
            var s: Scalar[DTYPE] = 0
            for j in range(nv):
                s += rebind[Scalar[DTYPE]](M[env, i * nv + j]) * search[j]
            Mv[i] = s

        # Precompute J · search per contact direction (for linesearch)
        var Js_n = Scratch[Scalar[DTYPE], MC_CAP](
        max_contacts, uninitialized=Scalar[DTYPE](0)
    )
        var Js_t1 = Scratch[Scalar[DTYPE], MC_CAP](
        max_contacts, uninitialized=Scalar[DTYPE](0)
    )
        var Js_t2 = Scratch[Scalar[DTYPE], MC_CAP](
        max_contacts, uninitialized=Scalar[DTYPE](0)
    )
        for c in range(nc):
            var js_n_c: Scalar[DTYPE] = 0
            var js_t1_c: Scalar[DTYPE] = 0
            var js_t2_c: Scalar[DTYPE] = 0
            if rebind[Scalar[DTYPE]](solver[env, ws_c_dist_idx + c]) < Scalar[
                DTYPE
            ](0):
                for i in range(nv):
                    var s_i = search[i]
                    js_n_c += (
                        rebind[Scalar[DTYPE]](
                            solver[env, ws_J_n_idx + c * nv + i]
                        )
                        * s_i
                    )
                    js_t1_c += (
                        rebind[Scalar[DTYPE]](
                            solver[env, ws_Jt1_idx + c * nv + i]
                        )
                        * s_i
                    )
                    js_t2_c += (
                        rebind[Scalar[DTYPE]](
                            solver[env, ws_Jt2_idx + c * nv + i]
                        )
                        * s_i
                    )
            Js_n[c] = js_n_c
            Js_t1[c] = js_t1_c
            Js_t2[c] = js_t2_c
        for s_i in range(ns):
            sr_Js[s_i] = sr_sign[s_i] * search[sr_dof[s_i]]

        # Compute current total cost and gradient-direction dot product
        var gauss_0: Scalar[DTYPE] = 0
        var g1: Scalar[DTYPE] = 0
        var g2: Scalar[DTYPE] = 0
        var gtd: Scalar[DTYPE] = 0
        for i in range(nv):
            var Ma_diff_i = Ma[i] - qfrc_sm[i]
            var qa_diff_i = qacc[i] - qacc_sm[i]
            gauss_0 += Ma_diff_i * qa_diff_i
            g1 += Ma_diff_i * search[i] + Mv[i] * qa_diff_i
            g2 += Mv[i] * search[i]
            gtd += grad[i] * search[i]
        gauss_0 = Scalar[DTYPE](0.5) * gauss_0
        g1 = Scalar[DTYPE](0.5) * g1
        g2 = Scalar[DTYPE](0.5) * g2

        # Current constraint cost
        var c_cost_0: Scalar[DTYPE] = 0
        for c in range(nc):
            if rebind[Scalar[DTYPE]](
                solver[env, ws_c_dist_idx + c]
            ) >= Scalar[DTYPE](0):
                continue
            var cs = Int(rebind[Scalar[DTYPE]](solver[env, ws_cstate_idx + c]))
            var N = rebind[Scalar[DTYPE]](solver[env, ws_jar_n_idx + c])
            var T1 = rebind[Scalar[DTYPE]](solver[env, ws_jar_t1_idx + c])
            var T2 = rebind[Scalar[DTYPE]](solver[env, ws_jar_t2_idx + c])
            var mu_c = rebind[Scalar[DTYPE]](solver[env, ws_mu_idx + c])
            var D_n_c = rebind[Scalar[DTYPE]](solver[env, ws_D_n_idx + c])
            var D_f_c = rebind[Scalar[DTYPE]](solver[env, ws_D_f_idx + c])
            if cs == 1:
                c_cost_0 += Scalar[DTYPE](0.5) * (
                    D_n_c * N * N + D_f_c * (T1 * T1 + T2 * T2)
                )
            elif cs == 2:
                var T_s = sqrt(T1 * T1 + T2 * T2)
                if T_s < Scalar[DTYPE](PRIMAL_MINVAL_GPU):
                    T_s = Scalar[DTYPE](PRIMAL_MINVAL_GPU)
                var s = N - mu_c * T_s
                var Dm = D_n_c / (Scalar[DTYPE](1.0) + mu_c * mu_c)
                c_cost_0 += Scalar[DTYPE](0.5) * Dm * s * s

        for s_i in range(ns):
            c_cost_0 += scalar_row_cost[DTYPE](
                sr_st[s_i], sr_jar[s_i], sr_D[s_i], sr_R[s_i], sr_floss[s_i]
            )

        var current_cost = gauss_0 + c_cost_0

        # Armijo linesearch
        var alpha = Scalar[DTYPE](1.0)
        var armijo_c = Scalar[DTYPE](ARMIJO)
        for _ in range(LINESEARCH_ITER):
            var trial_gauss = gauss_0 + alpha * g1 + alpha * alpha * g2
            var trial_c_cost: Scalar[DTYPE] = 0
            for c in range(nc):
                if rebind[Scalar[DTYPE]](
                    solver[env, ws_c_dist_idx + c]
                ) >= Scalar[DTYPE](0):
                    continue
                var trial_N = (
                    rebind[Scalar[DTYPE]](solver[env, ws_jar_n_idx + c])
                    + alpha * Js_n[c]
                )
                var trial_T1 = (
                    rebind[Scalar[DTYPE]](solver[env, ws_jar_t1_idx + c])
                    + alpha * Js_t1[c]
                )
                var trial_T2 = (
                    rebind[Scalar[DTYPE]](solver[env, ws_jar_t2_idx + c])
                    + alpha * Js_t2[c]
                )
                var mu_c = rebind[Scalar[DTYPE]](solver[env, ws_mu_idx + c])
                var D_n_c = rebind[Scalar[DTYPE]](solver[env, ws_D_n_idx + c])
                var D_f_c = rebind[Scalar[DTYPE]](solver[env, ws_D_f_idx + c])
                var trial_T = sqrt(trial_T1 * trial_T1 + trial_T2 * trial_T2)
                var trial_T_safe = trial_T
                if trial_T_safe < Scalar[DTYPE](PRIMAL_MINVAL_GPU):
                    trial_T_safe = Scalar[DTYPE](PRIMAL_MINVAL_GPU)
                if trial_N >= mu_c * trial_T_safe:
                    pass
                elif mu_c * trial_N + trial_T <= Scalar[DTYPE](0):
                    trial_c_cost += Scalar[DTYPE](0.5) * (
                        D_n_c * trial_N * trial_N
                        + D_f_c * (trial_T1 * trial_T1 + trial_T2 * trial_T2)
                    )
                else:
                    var trial_s = trial_N - mu_c * trial_T_safe
                    var Dm = D_n_c / (Scalar[DTYPE](1.0) + mu_c * mu_c)
                    trial_c_cost += Scalar[DTYPE](0.5) * Dm * trial_s * trial_s
            for s_i in range(ns):
                var tj = sr_jar[s_i] + alpha * sr_Js[s_i]
                var tst = scalar_row_state[DTYPE](
                    sr_kind[s_i], tj, sr_R[s_i], sr_floss[s_i]
                )
                trial_c_cost += scalar_row_cost[DTYPE](
                    tst, tj, sr_D[s_i], sr_R[s_i], sr_floss[s_i]
                )
            var trial_cost = trial_gauss + trial_c_cost
            if trial_cost <= current_cost + armijo_c * alpha * gtd:
                break
            alpha = alpha * Scalar[DTYPE](0.5)

        if alpha < Scalar[DTYPE](1e-12):
            break

        # Update qacc and Ma
        for i in range(nv):
            qacc[i] = qacc[i] + alpha * search[i]
            Ma[i] = Ma[i] + alpha * Mv[i]

        # Recompute jar and forces (3-zone cone logic)
        for c in range(nc):
            if rebind[Scalar[DTYPE]](
                solver[env, ws_c_dist_idx + c]
            ) >= Scalar[DTYPE](0):
                continue
            var jar_n_c: Scalar[DTYPE] = rebind[Scalar[DTYPE]](
                solver[env, ws_pos_bias_idx + c]
            )
            var jar_t1_c: Scalar[DTYPE] = rebind[Scalar[DTYPE]](
                solver[env, ws_bt1_idx + c]
            )
            var jar_t2_c: Scalar[DTYPE] = rebind[Scalar[DTYPE]](
                solver[env, ws_bt2_idx + c]
            )
            for i in range(nv):
                var qa_i = qacc[i]
                jar_n_c += (
                    rebind[Scalar[DTYPE]](solver[env, ws_J_n_idx + c * nv + i])
                    * qa_i
                )
                jar_t1_c += (
                    rebind[Scalar[DTYPE]](solver[env, ws_Jt1_idx + c * nv + i])
                    * qa_i
                )
                jar_t2_c += (
                    rebind[Scalar[DTYPE]](solver[env, ws_Jt2_idx + c * nv + i])
                    * qa_i
                )
            solver[env, ws_jar_n_idx + c] = jar_n_c
            solver[env, ws_jar_t1_idx + c] = jar_t1_c
            solver[env, ws_jar_t2_idx + c] = jar_t2_c

            var mu_c = rebind[Scalar[DTYPE]](solver[env, ws_mu_idx + c])
            var D_n_c = rebind[Scalar[DTYPE]](solver[env, ws_D_n_idx + c])
            var D_f_c = rebind[Scalar[DTYPE]](solver[env, ws_D_f_idx + c])
            var T = sqrt(jar_t1_c * jar_t1_c + jar_t2_c * jar_t2_c)
            var T_safe = T
            if T_safe < Scalar[DTYPE](PRIMAL_MINVAL_GPU):
                T_safe = Scalar[DTYPE](PRIMAL_MINVAL_GPU)
            if jar_n_c >= mu_c * T_safe:
                solver[env, ws_fn_idx + c] = 0
                solver[env, ws_ft1_idx + c] = 0
                solver[env, ws_ft2_idx + c] = 0
                solver[env, ws_cstate_idx + c] = 0
            elif mu_c * jar_n_c + T <= Scalar[DTYPE](0):
                solver[env, ws_fn_idx + c] = -D_n_c * jar_n_c
                solver[env, ws_ft1_idx + c] = -D_f_c * jar_t1_c
                solver[env, ws_ft2_idx + c] = -D_f_c * jar_t2_c
                solver[env, ws_cstate_idx + c] = 1
            else:
                var s = jar_n_c - mu_c * T_safe
                var Dm = D_n_c / (Scalar[DTYPE](1.0) + mu_c * mu_c)
                solver[env, ws_fn_idx + c] = -Dm * s
                solver[env, ws_ft1_idx + c] = Dm * mu_c * s * jar_t1_c / T_safe
                solver[env, ws_ft2_idx + c] = Dm * mu_c * s * jar_t2_c / T_safe
                solver[env, ws_cstate_idx + c] = Scalar[DTYPE](2)

        for s_i in range(ns):
            var jr = sr_bias[s_i] + sr_sign[s_i] * qacc[sr_dof[s_i]]
            sr_jar[s_i] = jr
            var st = scalar_row_state[DTYPE](
                sr_kind[s_i], jr, sr_R[s_i], sr_floss[s_i]
            )
            sr_st[s_i] = st
            sr_f[s_i] = scalar_row_force[DTYPE](
                st, jr, sr_D[s_i], sr_floss[s_i]
            )

        # Save old gradient for Polak-Ribiere
        for i in range(nv):
            gradold[i] = grad[i]
            Mgradold[i] = Mgrad[i]

        # Compute new gradient
        grad_norm_sq = 0
        for i in range(nv):
            var g: Scalar[DTYPE] = Ma[i] - qfrc_sm[i]
            for c in range(nc):
                var cs = Int(
                    rebind[Scalar[DTYPE]](solver[env, ws_cstate_idx + c])
                )
                if cs == 0:
                    continue
                g -= (
                    rebind[Scalar[DTYPE]](solver[env, ws_J_n_idx + c * nv + i])
                    * rebind[Scalar[DTYPE]](solver[env, ws_fn_idx + c])
                    + rebind[Scalar[DTYPE]](
                        solver[env, ws_Jt1_idx + c * nv + i]
                    )
                    * rebind[Scalar[DTYPE]](solver[env, ws_ft1_idx + c])
                    + rebind[Scalar[DTYPE]](
                        solver[env, ws_Jt2_idx + c * nv + i]
                    )
                    * rebind[Scalar[DTYPE]](solver[env, ws_ft2_idx + c])
                )
            for s_i in range(ns):
                if sr_dof[s_i] == i:
                    g -= sr_sign[s_i] * sr_f[s_i]
            grad[i] = g
            grad_norm_sq += g * g

        # Compute new preconditioned gradient: Mgrad = M⁻¹ · grad
        chol_solve_inline[DTYPE, M_CAP, V_CAP](L_prec, grad, Mgrad, nv)

        # Polak-Ribiere beta
        var num: Scalar[DTYPE] = 0
        var den: Scalar[DTYPE] = 0
        for i in range(nv):
            num += grad[i] * (Mgrad[i] - Mgradold[i])
            den += gradold[i] * Mgradold[i]
        if den < Scalar[DTYPE](PRIMAL_MINVAL_GPU):
            den = Scalar[DTYPE](PRIMAL_MINVAL_GPU)
        var beta = num / den
        if beta < Scalar[DTYPE](0):
            beta = Scalar[DTYPE](0)

        # Update search direction: search = -Mgrad + beta * search
        for i in range(nv):
            search[i] = -Mgrad[i] + beta * search[i]

    # Write solved qacc back
    for i in range(nv):
        qacc_constrained[env, i] = qacc[i]

    # Write forces to state buffer for display/warmstart
    for c in range(nc):
        var c_off = c * CONTACT_SIZE
        contacts[env, c_off + CONTACT_IDX_FORCE_N] = solver[env, ws_fn_idx + c]
        contacts[env, c_off + CONTACT_IDX_FORCE_T1] = solver[
            env, ws_ft1_idx + c
        ]
        contacts[env, c_off + CONTACT_IDX_FORCE_T2] = solver[
            env, ws_ft2_idx + c
        ]

    # === Post-solve: equality, tendon (shared with Newton elliptic) ===
    # Limits and dry-friction dofs are rows of the CG system above, matching
    # the Newton path — see constraints/scalar_rows.mojo. Equality and tendon
    # rows need a dense Jacobian and remain post-passes for now.
    comptime SOLVER_ITER_GPU: Int = 50

    comptime if may_exist[D.NEQUALITY]():
        _equality_env[
            DTYPE, V_CAP, SOLVER_ITER_GPU](
            env, dims, qpos, qvel, xpos, xquat, subtree_com, joints, bodies, mmeta,
            equality, body_invweight0, dof_invweight0, cdof,
            m_inv, qacc_constrained,
        )

    comptime if may_exist[D.NTENDON]():
        _tendon_env[
            DTYPE, BATCH,
            SOLVER_ITER_GPU](
            env, dims, qpos, qvel, joints, mmeta, tendons, sites, geoms_w, bodies,
            subtree_com, cdof, xpos, xquat, m_inv, qacc_constrained,
        )


def _cg_solve_fields_kernel[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    NGEOM: Int,
    NEQUALITY: Int,
    NTENDON: Int,
    NSITE: Int,
    CONE_TYPE: Int,
    BATCH: Int,
    SOLVER_WS: Int,
](
    qpos: Tile[DTYPE, BATCH, NQ],
    qvel: Tile[DTYPE, BATCH, NV],
    xpos: Tile[DTYPE, BATCH, NBODY * 3],
    xquat: Tile[DTYPE, BATCH, NBODY * 4],
    subtree_com: Tile[DTYPE, BATCH, NBODY * 3],
    contacts: Tile[DTYPE, BATCH, MAX_CONTACTS * CONTACT_SIZE],
    smeta: Tile[DTYPE, BATCH, METADATA_SIZE],
    joints: Tile[DTYPE, NJOINT, MODEL_JOINT_SIZE],
    bodies: Tile[DTYPE, NBODY, MODEL_BODY_SIZE],
    mmeta: Tile[DTYPE, MODEL_META_SIZE],
    equality: Tile[DTYPE, NEQUALITY, MODEL_EQ_SIZE],
    tendons: Tile[DTYPE, NTENDON, MODEL_TENDON_SIZE],
    sites: Tile[DTYPE, NSITE, MODEL_SITE_SIZE],
    geoms_w: Tile[DTYPE, NGEOM, MODEL_GEOM_SIZE],
    body_invweight0: Tile[DTYPE, NBODY, 2],
    dof_invweight0: Tile[DTYPE, NV],
    cdof: Tile[DTYPE, BATCH, NV * 6],
    M: Tile[DTYPE, BATCH, NV * NV],
    m_inv: Tile[DTYPE, BATCH, NV * NV],
    qacc_constrained: Tile[DTYPE, BATCH, NV],
    solver: Tile[DTYPE, BATCH, SOLVER_WS],
):
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    _cg_solve_env[
        DTYPE,
        CONE_TYPE,
        BATCH,
        SOLVER_WS](
        env, Dims[nq=NQ, nv=NV, nbody=NBODY, njoint=NJOINT, max_contacts=MAX_CONTACTS, ngeom=NGEOM, nequality=NEQUALITY, ntendon=NTENDON, nsite=NSITE](), qpos, qvel, xpos, xquat, subtree_com, contacts, smeta, joints,
        bodies, mmeta, equality, tendons, sites, geoms_w, body_invweight0,
        dof_invweight0, cdof, M, m_inv, qacc_constrained, solver,
    )


def solve_cg[

    target: StaticString,
    DTYPE: DType,
    D: DimsLike,
    CONE_TYPE: Int = ConeType.ELLIPTIC,
    BATCH: Int = 1,
    # Appended, not grouped with NEXCLUDE — see `fields.Model`.
](
    mut d: Data[DTYPE, D, BATCH],
    mut m: Model[DTYPE, D],
    mut scratch: DynamicsScratch[DTYPE, D, BATCH],
    mut cscratch: ContactScratch[DTYPE, D, BATCH, _],
    ctx: Optional[DeviceContext] = None,
) raises:
    """Primal CG contact solve into `scratch.qacc_constrained` (+ solved
    forces back into `d.contacts`), both targets, one body. Same signature
    family as `solve_newton`/`solve_contacts` so callers can
    swap solvers.

    ELLIPTIC cone only. Joint limits, equality constraints and fixed tendons
    run after the CG core (50 iterations), at the legacy positions.

    Uses a PREFIX (35*MC + 6*MC*NV) of the PGS-sized `cscratch.solver`
    tensor (81*MC + 12*MC*NV) — no separate scratch.
    """
    comptime MC = _max_one[D.MAX_CONTACTS]()
    comptime SOLVER_WS = 81 * MC + 12 * MC * D.NV

    comptime L_NV = RowMajor[BATCH, D.NV]
    comptime L_B3 = RowMajor[BATCH, D.NBODY * 3]
    comptime L_B4 = RowMajor[BATCH, D.NBODY * 4]
    comptime L_CON = RowMajor[BATCH, D.MAX_CONTACTS * CONTACT_SIZE]
    comptime L_SMETA = RowMajor[BATCH, METADATA_SIZE]
    comptime L_JOINT = RowMajor[D.NJOINT, MODEL_JOINT_SIZE]
    comptime L_BODY = RowMajor[D.NBODY, MODEL_BODY_SIZE]
    comptime L_MMETA = RowMajor[MODEL_META_SIZE]
    comptime L_EQ = RowMajor[D.NEQUALITY, MODEL_EQ_SIZE]
    comptime L_TEN = RowMajor[D.NTENDON, MODEL_TENDON_SIZE]
    comptime L_SITE = RowMajor[D.NSITE, MODEL_SITE_SIZE]
    comptime L_GEOM_W = RowMajor[D.NGEOM, MODEL_GEOM_SIZE]
    comptime L_BW = RowMajor[D.NBODY, 2]
    comptime L_CDOF = RowMajor[BATCH, D.NV * 6]
    comptime L_prec = RowMajor[BATCH, D.NV * D.NV]
    comptime L_SOLVER = RowMajor[BATCH, SOLVER_WS]

    comptime L_QPOS = RowMajor[BATCH, D.NQ]
    comptime L_DW = RowMajor[D.NV]

    comptime if target == "cpu":
        var dm = d.dims
        var rl_QPOS = rl2(BATCH, dm.get_nq())
        var rl_NV = rl2(BATCH, dm.get_nv())
        var rl_B3 = rl2(BATCH, dm.get_nbody() * 3)
        var rl_B4 = rl2(BATCH, dm.get_nbody() * 4)
        var rl_CON = rl2(BATCH, dm.get_max_contacts() * CONTACT_SIZE)
        var rl_SMETA = rl2(BATCH, METADATA_SIZE)
        var rl_JOINT = rl2(dm.get_njoint(), MODEL_JOINT_SIZE)
        var rl_BODY = rl2(dm.get_nbody(), MODEL_BODY_SIZE)
        var rl_MMETA = rl1(MODEL_META_SIZE)
        var rl_EQ = rl2(dm.get_nequality(), MODEL_EQ_SIZE)
        var rl_TEN = rl2(dm.get_ntendon(), MODEL_TENDON_SIZE)
        var rl_SITE = rl2(dm.get_nsite(), MODEL_SITE_SIZE)
        var rl_GEOM_W = rl2(dm.get_ngeom(), MODEL_GEOM_SIZE)
        var rl_BW = rl2(dm.get_nbody(), 2)
        var rl_DW = rl1(dm.get_nv())
        var rl_CDOF = rl2(BATCH, dm.get_nv() * 6)
        var rl_prec = rl2(BATCH, dm.get_nv() * dm.get_nv())
        # ⚠⚠ RUNTIME BUDGET, NOT THE COMPTIME `SOLVER_WS`. On a dynamic
        # provider `D.NV` is DIM_POISON and `D.MAX_CONTACTS` floors to 1,
        # so the comptime literal is 81 - 12 = 69 scalars for EVERY model
        # while the ws_* offsets below are computed from the RUNTIME nv/mc.
        # The spelling was swept to `rl2`/`lt_dyn` in 3a; the VALUE was not.
        var rl_SOLVER = rl2(
            BATCH, ws_budget(_max_one_rt(dm.get_max_contacts()), dm.get_nv())
        )
        var qpos_v = d.qpos.tt_dyn["cpu", DYN2](rl_QPOS)
        var qvel_v = d.qvel.tt_dyn["cpu", DYN2](rl_NV)
        var xpos_v = d.xpos.tt_dyn["cpu", DYN2](rl_B3)
        var xquat_v = d.xquat.tt_dyn["cpu", DYN2](rl_B4)
        var stcom_v = d.subtree_com.tt_dyn["cpu", DYN2](rl_B3)
        var con_v = d.contacts.tt_dyn["cpu", DYN2](rl_CON)
        var smeta_v = d.meta.tt_dyn["cpu", DYN2](rl_SMETA)
        var joints_v = m.joints.tt_dyn["cpu", DYN2](rl_JOINT)
        var bodies_v = m.bodies.tt_dyn["cpu", DYN2](rl_BODY)
        var mmeta_v = m.meta.tt_dyn["cpu", DYN1](rl_MMETA)
        var eq_v = m.equality.tt_dyn["cpu", DYN2](rl_EQ)
        var ten_v = m.tendons.tt_dyn["cpu", DYN2](rl_TEN)
        var site_v = m.sites.tt_dyn["cpu", DYN2](rl_SITE)
        var geomw_v = m.geoms.tt_dyn["cpu", DYN2](rl_GEOM_W)
        var bw_v = m.body_invweight0.tt_dyn["cpu", DYN2](rl_BW)
        var dw_v = m.dof_invweight0.tt_dyn["cpu", DYN1](rl_DW)
        var cdof_v = scratch.cdof.tt_dyn["cpu", DYN2](rl_CDOF)
        var M_v = scratch.M.tt_dyn["cpu", DYN2](rl_prec)
        var mi_v = scratch.m_inv.tt_dyn["cpu", DYN2](rl_prec)
        var qc_v = scratch.qacc_constrained.tt_dyn["cpu", DYN2](rl_NV)
        var sol_v = cscratch.solver.tt_dyn["cpu", DYN2](rl_SOLVER)
        for e in range(BATCH):
            _cg_solve_env[
                DTYPE,
                CONE_TYPE,
                BATCH,
                SOLVER_WS](
                e, dm, qpos_v, qvel_v, xpos_v, xquat_v, stcom_v, con_v, smeta_v,
                joints_v, bodies_v, mmeta_v, eq_v, ten_v, site_v, geomw_v, bw_v, dw_v,
                cdof_v, M_v, mi_v, qc_v, sol_v,
            )
    else:
        var c = ctx.value()
        comptime BLOCKS = (BATCH + CG_TPB - 1) // CG_TPB
        c.enqueue_function[
            _cg_solve_fields_kernel[
                DTYPE,
                D.NQ,
                D.NV,
                D.NBODY,
                D.NJOINT,
                D.MAX_CONTACTS,
                D.NGEOM,
                D.NEQUALITY,
                D.NTENDON,
                D.NSITE,
                CONE_TYPE,
                BATCH,
                SOLVER_WS,
            ]
        ](
            d.qpos.tt["gpu", BATCH, D.NQ](),
            d.qvel.tt["gpu", BATCH, D.NV](),
            d.xpos.tt["gpu", BATCH, D.NBODY * 3](),
            d.xquat.tt["gpu", BATCH, D.NBODY * 4](),
            d.subtree_com.tt["gpu", BATCH, D.NBODY * 3](),
            d.contacts.tt["gpu", BATCH, D.MAX_CONTACTS * CONTACT_SIZE](),
            d.meta.tt["gpu", BATCH, METADATA_SIZE](),
            m.joints.tt["gpu", D.NJOINT, MODEL_JOINT_SIZE](),
            m.bodies.tt["gpu", D.NBODY, MODEL_BODY_SIZE](),
            m.meta.tt["gpu", MODEL_META_SIZE](),
            m.equality.tt["gpu", D.NEQUALITY, MODEL_EQ_SIZE](),
            m.tendons.tt["gpu", D.NTENDON, MODEL_TENDON_SIZE](),
            m.sites.tt["gpu", D.NSITE, MODEL_SITE_SIZE](),
                m.geoms.tt["gpu", D.NGEOM, MODEL_GEOM_SIZE](),
            m.body_invweight0.tt["gpu", D.NBODY, 2](),
            m.dof_invweight0.tt["gpu", D.NV](),
            scratch.cdof.tt["gpu", BATCH, D.NV * 6](),
            scratch.M.tt["gpu", BATCH, D.NV * D.NV](),
            scratch.m_inv.tt["gpu", BATCH, D.NV * D.NV](),
            scratch.qacc_constrained.tt["gpu", BATCH, D.NV](),
            cscratch.solver.tt["gpu", BATCH, SOLVER_WS](),
            grid_dim=(BLOCKS,),
            block_dim=(CG_TPB,),
        )
