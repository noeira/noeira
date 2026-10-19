"""Fields-native constraint inverse-weights (body_invweight0 / dof_invweight0).

Port of legacy `dynamics/mass_matrix.compute_body_invweight0` (MuJoCo
`mj_setConst`) onto the per-field tensors — a ONE-TIME CPU build step run by the
fields model build (`init_fields`), replacing the CPU-`Model` invweight0 path.

It seeds `d.qpos` to qpos0 itself (see below), then runs the fields
FK → subtree_com → cdof → CRBA(+armature) → LDL → M^-1 pipeline:
  body_invweight0[2i]   = avg(A[0,0], A[1,1], A[2,2])   (translation)
  body_invweight0[2i+1] = avg(A[3,3], A[4,4], A[5,5])   (rotation)
where A = J M^-1 J^T for the 6xNV body-CoM Jacobian J, and
  dof_invweight0[d]     = (M^-1)[d,d],
the latter then averaged within each free/ball joint's translation and
rotation dof groups exactly as `mj_setConst` does.

These feed every constraint/solver inverse-weight (`mj_diagApprox` reads
dof_invweight0 for joint limits, body_invweight0 for contacts), so a wrong
value here is a silent multiplicative error on every constraint force.
Gated against MuJoCo's own m.dof_invweight0 / m.body_invweight0 by
tests/physics3d/test_constraints_vs_mujoco.mojo (ant, humanoid, hopper — all
FREE-ROOTED) and by tests/dm_control/test_finger_vs_dm_control.mojo, which
covers the case those three structurally cannot: a body with NO translational
freedom, whose translational weight is legitimately zero. Add the same
few-line check to each newly ported model's gate — it is what caught bug 20.

`body_simple == 2` bodies (all joints axis-aligned SLIDEs, inertial frame ==
body frame, not a parent, anchored) take MuJoCo's short-circuit
`invweight0 = (1/mass, 0)` rather than the general path — see the comment at
the branch. This USED to be an unimplemented "known deviation" documented as
harmless; it is not. The general path averages over three translational axes,
so a body with fewer than three slide dofs comes out short by exactly the
missing fraction. Fixed 2026-07-31 after dm_control's ball_in_cup measured
10.1859 against MuJoCo's 15.2789 (= 2/3) on its two-slide ball.

CPU-only (build-time); no GPU kernels needed.
"""


from noeira.physics3d.fields import (
    Data,
    Model,
    DynamicsScratch,
    Dims,
    DimsLike,
    AsStatic,
    Scratch,
    cap,
    DYN1,
    DYN2,
    rl1,
    rl2,
)
from noeira.physics3d.kinematics.forward_kinematics import (
    forward_kinematics,
)
from noeira.physics3d.dynamics.subtree_com import (
    compute_subtree_com,
)
from noeira.physics3d.dynamics.cdof import compute_cdof
from noeira.physics3d.dynamics.mass_matrix import (
    compute_mass_matrix,
)
from noeira.physics3d.dynamics.ldl import (
    ldl_factor,
    ldl_solve,
)
from noeira.physics3d.joint_types import JNT_FREE, JNT_BALL, JNT_SLIDE
from noeira.physics3d.gpu.constants import (
    MODEL_BODY_SIZE,
    MODEL_JOINT_SIZE,
    MODEL_META_IDX_MEANINERTIA,
    BODY_IDX_PARENT,
    BODY_IDX_ROOTID,
    BODY_IDX_POS_X,
    BODY_IDX_POS_Y,
    BODY_IDX_POS_Z,
    BODY_IDX_QUAT_X,
    BODY_IDX_QUAT_Y,
    BODY_IDX_QUAT_Z,
    BODY_IDX_QUAT_W,
    BODY_IDX_IPOS_X,
    BODY_IDX_IPOS_Y,
    BODY_IDX_IPOS_Z,
    BODY_IDX_IQUAT_X,
    BODY_IDX_IQUAT_Y,
    BODY_IDX_IQUAT_Z,
    BODY_IDX_IQUAT_W,
    BODY_IDX_MASS,
    JOINT_IDX_POS_X,
    JOINT_IDX_POS_Y,
    JOINT_IDX_POS_Z,
    JOINT_IDX_AXIS_X,
    JOINT_IDX_AXIS_Y,
    JOINT_IDX_AXIS_Z,
    JOINT_IDX_TYPE,
    JOINT_IDX_BODY_ID,
    JOINT_IDX_QPOS_ADR,
    JOINT_IDX_DOF_ADR,
    JOINT_IDX_ARMATURE,
    JOINT_IDX_QPOS0,
    MODEL_META_SIZE,
    MODEL_SITE_SIZE,
    MODEL_GEOM_SIZE,
    MODEL_TENDON_SIZE,
    TENDON_KIND_SPATIAL,
    TENDON_IDX_KIND,
    TENDON_IDX_NUM_JOINTS,
    TENDON_IDX_JOINT_0,
    TENDON_IDX_COEF_0,
    TENDON_IDX_INVWEIGHT0,
    TENDON_IDX_LENGTH_REF,
    MODEL_EQ_SIZE,
    EQ_IDX_TYPE,
    EQ_IDX_BODY_A,
    EQ_IDX_BODY_B,
    EQ_IDX_ANCHOR_AX,
    EQ_IDX_ANCHOR_AY,
    EQ_IDX_ANCHOR_AZ,
    EQ_IDX_ANCHOR_BX,
    EQ_IDX_ANCHOR_BY,
    EQ_IDX_ANCHOR_BZ,
    EQ_IDX_OBJTYPE,
    EQ_IDX_RELPOSE_X,
    EQ_IDX_RELPOSE_Y,
    EQ_IDX_RELPOSE_Z,
    EQ_IDX_RELPOSE_W,
)
from noeira.physics3d.types import EQ_WELD, EQ_CONNECT, EQ_OBJ_SITE
from noeira.physics3d.kinematics.quat_math import (
    quat_mul,
    quat_conjugate,
    quat_rotate,
)
from .tendon import spatial_tendon_length_jac


def compute_invweight0[

    DTYPE: DType,
    # Appended, not grouped with NEXCLUDE — see `fields.Model`.

    D: DimsLike,
](
    mut d: Data[DTYPE, D, 1],
    mut mf: Model[DTYPE, D],
    mut sc: DynamicsScratch[DTYPE, D, 1],
) raises:
    """Compute mf.body_invweight0 / mf.dof_invweight0 at qpos0 — see module
    docstring. Build-time, single-env (BATCH=1); overwrites `d`."""
    # ⚠ READ ONCE, FROM THE PROVIDER THE MODEL CARRIES (3c-c). These were
    # `D.NQ`/`D.NV`/… — comptime, hence `DIM_POISON` on a runtime-loaded
    # model, hence every loop below running zero times and the inverse
    # weights coming out silently ZERO.
    var nq = mf.dims.get_nq()
    var nv = mf.dims.get_nv()
    var nbody = mf.dims.get_nbody()
    var njoint = mf.dims.get_njoint()
    var ntendon = mf.dims.get_ntendon()
    # ── reference pose = MuJoCo qpos0 ────────────────────────────────────────
    # `mj_setConst` evaluates the inverse weights at qpos0: the COMPILER's
    # reference configuration (joint `ref`, free joints at their body's pose),
    # NOT the env reset pose.  Those used to differ for Gym-derived models —
    # ant's <custom><numeric name="init_qpos"> parked its ankles at ±1 rad
    # until the parser stopped applying it (MuJoCo never did) — and M is
    # configuration-dependent, so seeding from a reset pose skewed every
    # inverse weight (ant: 0.75% on the hinges, 32% on the free root). Built
    # from the joint records here so that can never recur.
    for i in range(nq):
        d.qpos.data[i] = Scalar[DTYPE](0)
    for j in range(njoint):
        var jo = j * MODEL_JOINT_SIZE
        var jt = Int(mf.joints.data[jo + JOINT_IDX_TYPE])
        var qadr = Int(mf.joints.data[jo + JOINT_IDX_QPOS_ADR])
        if jt == JNT_FREE:
            # qpos = [tx, ty, tz, qw, qx, qy, qz] from the body's own pose.
            var bo = (
                Int(mf.joints.data[jo + JOINT_IDX_BODY_ID]) * MODEL_BODY_SIZE
            )
            d.qpos.data[qadr + 0] = mf.bodies.data[bo + BODY_IDX_POS_X]
            d.qpos.data[qadr + 1] = mf.bodies.data[bo + BODY_IDX_POS_Y]
            d.qpos.data[qadr + 2] = mf.bodies.data[bo + BODY_IDX_POS_Z]
            d.qpos.data[qadr + 3] = mf.bodies.data[bo + BODY_IDX_QUAT_W]
            d.qpos.data[qadr + 4] = mf.bodies.data[bo + BODY_IDX_QUAT_X]
            d.qpos.data[qadr + 5] = mf.bodies.data[bo + BODY_IDX_QUAT_Y]
            d.qpos.data[qadr + 6] = mf.bodies.data[bo + BODY_IDX_QUAT_Z]
        elif jt == JNT_BALL:
            d.qpos.data[qadr] = Scalar[DTYPE](1)  # identity (w, x, y, z)
        else:
            d.qpos.data[qadr] = mf.joints.data[jo + JOINT_IDX_QPOS0]
    for i in range(nv):
        d.qvel.data[i] = Scalar[DTYPE](0)

    # ── fields pipeline: FK -> subtree_com -> cdof -> CRBA -> +armature -> LDL -> M^-1
    forward_kinematics["cpu", DTYPE, BATCH=1](d, mf, None)
    compute_subtree_com["cpu", DTYPE, BATCH=1](d, mf, None)
    compute_cdof["cpu", DTYPE, BATCH=1](d, mf, sc, None)
    compute_mass_matrix["cpu", DTYPE, BATCH=1](d, mf, sc, None)

    # Add armature to the diagonal (matches legacy mass_matrix.mojo:447-457).
    for j in range(njoint):
        var jo = j * MODEL_JOINT_SIZE
        var jtype = Int(mf.joints.data[jo + JOINT_IDX_TYPE])
        var dof_adr = Int(mf.joints.data[jo + JOINT_IDX_DOF_ADR])
        var arm = mf.joints.data[jo + JOINT_IDX_ARMATURE]
        var ndof = 1
        if jtype == JNT_FREE:
            ndof = 6
        elif jtype == JNT_BALL:
            ndof = 3
        for dd in range(ndof):
            sc.M.data[(dof_adr + dd) * nv + (dof_adr + dd)] += arm

    # ── stat.meaninertia, and it MUST be read here ──────────────────────────
    # `ldl_factor` overwrites `sc.M` IN PLACE on the very next line, so after
    # it the "diagonal" is the LDL factor's D, not the mass matrix's. MuJoCo
    # takes this from `d->qM` inside `mj_setConst`, i.e. the same pre-
    # factorization matrix, armature included.
    #
    # Consumed only by `mj_solNoSlip`'s convergence test — see
    # `MODEL_META_IDX_MEANINERTIA`.
    # ⚠ `dof_M0` COMES FROM THE SAME PRE-FACTORIZATION DIAGONAL, and for the
    # same reason: `ldl_factor` below overwrites `sc.M` in place, so after it
    # the diagonal is the factor's D. MuJoCo computes `dof_M0` in its own CRB
    # pass (`engine_setconst.c:36`) as `armature + cdof_i . (crb_i * cdof_i)`,
    # which is exactly this diagonal — armature was added a few lines up.
    # Read here rather than recomputed because a second CRBA is a second
    # implementation to keep in step.
    var _mi_sum = Float64(0)
    for i in range(nv):
        var _mii = Float64(sc.M.data[i * nv + i])
        mf.dof_M0.data[i] = Scalar[DTYPE](_mii)
        _mi_sum += _mii
    mf.meta.data[MODEL_META_IDX_MEANINERTIA] = Scalar[DTYPE](
        _mi_sum / Float64(nv) if nv > 0 else Float64(0)
    )

    ldl_factor["cpu", DTYPE, BATCH=1](mf, sc)

    # ── dof->body map (matches legacy :461-476) ──────────────────────────────
    var dof_body = List[Int](length=nv, fill=0)
    for j in range(njoint):
        var jo = j * MODEL_JOINT_SIZE
        var jtype = Int(mf.joints.data[jo + JOINT_IDX_TYPE])
        var body = Int(mf.joints.data[jo + JOINT_IDX_BODY_ID])
        var dof_adr = Int(mf.joints.data[jo + JOINT_IDX_DOF_ADR])
        var ndof = 1
        if jtype == JNT_FREE:
            ndof = 6
        elif jtype == JNT_BALL:
            ndof = 3
        for dd in range(ndof):
            dof_body[dof_adr + dd] = body

    # World body: zero weights.
    mf.body_invweight0.data[0] = Scalar[DTYPE](0)
    mf.body_invweight0.data[1] = Scalar[DTYPE](0)

    # ── body_simple == 2 (mj_setConst's short-circuit) ───────────────────────
    #
    # MuJoCo does NOT take the A = J M^-1 J^T path for a "simple level 2" body
    # — one whose joints are ALL axis-aligned SLIDEs through the body origin,
    # whose inertial frame is the body frame, and which is neither a parent nor
    # a child of a moving body. It assigns
    #
    #     body_invweight0[2i] = 1/mass,   body_invweight0[2i+1] = 0
    #
    # (engine_setconst.c:138-145, predicate at user_model.cc:2657-2766).
    #
    # ⚠ THIS IS NOT AN OPTIMIZATION. The two paths agree only when the body has
    # all THREE translational dofs, because the general path averages
    # (A[0,0] + A[1,1] + A[2,2]) / 3 and a missing axis contributes a ZERO.
    # A body with two slides therefore gets 2/3 of the right value, and one
    # with a single slide gets 1/3.
    #
    # This module's docstring claimed the deviation "agrees to float tolerance
    # on the models gated here", which was true only because every gated model
    # was FREE-ROOTED (3 translational dofs) or contact-free. dm_control's
    # ball_in_cup is the first with a contacting 2-slide body: its ball came
    # out at 10.1859 against MuJoCo's 15.2789 — exactly 2/3 — making every
    # ball contact 1.5x too soft. Same failure shape as bug 20, on the other
    # half of the same function, and hidden for the same reason.
    var body_dofnum = List[Int](length=nbody, fill=0)
    var body_is_parent = List[Bool](length=nbody, fill=False)
    for i in range(nbody):
        var par = Int(mf.bodies.data[i * MODEL_BODY_SIZE + BODY_IDX_PARENT])
        if par > 0:
            body_is_parent[par] = True
    for dd in range(nv):
        body_dofnum[dof_body[dd]] += 1

    var simple2 = List[Bool](length=nbody, fill=False)
    for i in range(1, nbody):
        if body_dofnum[i] == 0 or body_is_parent[i]:
            continue
        var bo = i * MODEL_BODY_SIZE
        # sameframe: inertial frame == body frame.
        if (
            abs(mf.bodies.data[bo + BODY_IDX_IPOS_X]) > 1e-12
            or abs(mf.bodies.data[bo + BODY_IDX_IPOS_Y]) > 1e-12
            or abs(mf.bodies.data[bo + BODY_IDX_IPOS_Z]) > 1e-12
            or abs(mf.bodies.data[bo + BODY_IDX_IQUAT_X]) > 1e-12
            or abs(mf.bodies.data[bo + BODY_IDX_IQUAT_Y]) > 1e-12
            or abs(mf.bodies.data[bo + BODY_IDX_IQUAT_Z]) > 1e-12
            or abs(abs(mf.bodies.data[bo + BODY_IDX_IQUAT_W]) - 1) > 1e-12
        ):
            continue
        # self-root, or parent is a fixed child of world.
        var rootid = Int(mf.bodies.data[bo + BODY_IDX_ROOTID])
        var par = Int(mf.bodies.data[bo + BODY_IDX_PARENT])
        var anchored = rootid == i
        if not anchored and par > 0:
            var gp = Int(
                mf.bodies.data[par * MODEL_BODY_SIZE + BODY_IDX_PARENT]
            )
            anchored = gp == 0 and body_dofnum[par] == 0
        if not anchored:
            continue
        # every joint an axis-aligned SLIDE through the body origin.
        var ok = True
        for j in range(njoint):
            var jo = j * MODEL_JOINT_SIZE
            if Int(mf.joints.data[jo + JOINT_IDX_BODY_ID]) != i:
                continue
            if Int(mf.joints.data[jo + JOINT_IDX_TYPE]) != JNT_SLIDE:
                ok = False
                break
            if (
                abs(mf.joints.data[jo + JOINT_IDX_POS_X]) > 1e-12
                or abs(mf.joints.data[jo + JOINT_IDX_POS_Y]) > 1e-12
                or abs(mf.joints.data[jo + JOINT_IDX_POS_Z]) > 1e-12
            ):
                ok = False
                break
            var naxis = 0
            if abs(mf.joints.data[jo + JOINT_IDX_AXIS_X]) > 1e-14:
                naxis += 1
            if abs(mf.joints.data[jo + JOINT_IDX_AXIS_Y]) > 1e-14:
                naxis += 1
            if abs(mf.joints.data[jo + JOINT_IDX_AXIS_Z]) > 1e-14:
                naxis += 1
            if naxis != 1:
                ok = False
                break
        if ok:
            simple2[i] = True

    # ── per-body invweight0 via A = J M^-1 J^T diagonal (legacy :482-586) ─────
    for i in range(nbody):
        if simple2[i]:
            var mass = mf.bodies.data[i * MODEL_BODY_SIZE + BODY_IDX_MASS]
            if mass < Scalar[DTYPE](1e-15):
                mass = Scalar[DTYPE](1e-15)
            mf.body_invweight0.data[2 * i] = Scalar[DTYPE](1) / mass
            mf.body_invweight0.data[2 * i + 1] = Scalar[DTYPE](0)
            continue
        var ti_x = d.xipos.data[i * 3 + 0]
        var ti_y = d.xipos.data[i * 3 + 1]
        var ti_z = d.xipos.data[i * 3 + 2]

        var A_diag = Array[Scalar[DTYPE], 6](fill=Scalar[DTYPE](0))
        for k in range(6):
            var J_row = List[Scalar[DTYPE]](length=nv, fill=Scalar[DTYPE](0))
            for dd in range(nv):
                var b = dof_body[dd]
                # Does DOF dd affect body i (dd's body == i or an ancestor)?
                var affects = b == i
                if not affects:
                    var current = i
                    while current > 0:
                        var par = Int(
                            mf.bodies.data[current * MODEL_BODY_SIZE + BODY_IDX_PARENT]
                        )
                        if par == b:
                            affects = True
                            break
                        current = par
                if not affects:
                    continue

                var ang_x = sc.cdof.data[dd * 6 + 0]
                var ang_y = sc.cdof.data[dd * 6 + 1]
                var ang_z = sc.cdof.data[dd * 6 + 2]
                var lin_x = sc.cdof.data[dd * 6 + 3]
                var lin_y = sc.cdof.data[dd * 6 + 4]
                var lin_z = sc.cdof.data[dd * 6 + 5]

                # Fields cdof.lin is referenced at subtree_com[rootid[b]]
                # (MuJoCo convention, cdof.mojo:108), so shift the CoM
                # Jacobian from THAT point — not the body CoM.
                var rb = Int(
                    mf.bodies.data[b * MODEL_BODY_SIZE + BODY_IDX_ROOTID]
                )
                var dx = ti_x - d.subtree_com.data[rb * 3 + 0]
                var dy = ti_y - d.subtree_com.data[rb * 3 + 1]
                var dz = ti_z - d.subtree_com.data[rb * 3 + 2]

                if k == 0:
                    J_row[dd] = lin_x + ang_y * dz - ang_z * dy
                elif k == 1:
                    J_row[dd] = lin_y + ang_z * dx - ang_x * dz
                elif k == 2:
                    J_row[dd] = lin_z + ang_x * dy - ang_y * dx
                elif k == 3:
                    J_row[dd] = ang_x
                elif k == 4:
                    J_row[dd] = ang_y
                else:
                    J_row[dd] = ang_z

            # A[k,k] = J_row . M^-1 . J_row via a direct LDL solve (matches
            # legacy compute_body_invweight0 arithmetic bit-for-bit).
            for q in range(nv):
                sc.fnet.data[q] = J_row[q]
            ldl_solve["cpu", DTYPE, BATCH=1](mf, sc)
            var dot_val = Scalar[DTYPE](0)
            for q in range(nv):
                dot_val += J_row[q] * sc.qacc_ws.data[q]
            A_diag[k] = dot_val

        var tran = (A_diag[0] + A_diag[1] + A_diag[2]) / Scalar[DTYPE](3)
        var rot = (A_diag[3] + A_diag[4] + A_diag[5]) / Scalar[DTYPE](3)
        # NO cross-fallback between the two halves. `mj_setConst` assigns the
        # two averages independently (engine_setconst.c:157-158); its only
        # special cases are a world-welded body (both 0) and `body_simple == 2`
        # (1/mass, 0) — neither substitutes one half for the other. A ZERO
        # translational weight is a correct, meaningful answer: a body whose
        # CoM lies on its only rotation axis genuinely cannot be translated,
        # so J_tran is exactly 0 and so is A's translational block.
        #
        # This used to read "if tran ~ 0: tran = rot" (and vice versa), which
        # silently substituted a value MuJoCo never produces. dm_control's
        # finger spinner is exactly that body — a symmetric two-capsule wheel
        # on one hinge, CoM on the axis — so its `tran` became 26.24 instead
        # of 0. Contacts read diagApprox = body_invweight0[a] + [b], giving
        # 26.65 where MuJoCo has 0.4166, hence R = (1-imp)/imp*diagApprox was
        # 64x too large and every fingertip/spinner contact was 64x too soft
        # (measured normal force 29.1 N vs MuJoCo's 1860.2 N at the same pose).
        # Invisible until now because every previously gated model is
        # free-rooted, where no body has a zero translational weight.
        mf.body_invweight0.data[2 * i] = tran
        mf.body_invweight0.data[2 * i + 1] = rot

    # ── dof_invweight0[d] = (M^-1)[d,d] via solve M x = e_d (legacy :588-601) ─
    for dd in range(nv):
        for q in range(nv):
            sc.fnet.data[q] = Scalar[DTYPE](1) if q == dd else Scalar[DTYPE](0)
        ldl_solve["cpu", DTYPE, BATCH=1](mf, sc)
        mf.dof_invweight0.data[dd] = sc.qacc_ws.data[dd]

    # ── average within each free/ball dof group (engine_setconst.c:199-209) ──
    # MuJoCo assigns one value to a free joint's 3 translation dofs and one to
    # its 3 rotation dofs (a ball joint: one across its 3), so the weight is
    # axis-independent.  Skipping this leaves per-axis values MuJoCo never
    # produces — 32% off on ant's free root even at the correct pose.
    for j in range(njoint):
        var jo = j * MODEL_JOINT_SIZE
        var jt = Int(mf.joints.data[jo + JOINT_IDX_TYPE])
        var dadr = Int(mf.joints.data[jo + JOINT_IDX_DOF_ADR])
        var ngroup = 2 if jt == JNT_FREE else (1 if jt == JNT_BALL else 0)
        for g in range(ngroup):
            var b = dadr + 3 * g
            var avg = (
                mf.dof_invweight0.data[b]
                + mf.dof_invweight0.data[b + 1]
                + mf.dof_invweight0.data[b + 2]
            ) / Scalar[DTYPE](3)
            mf.dof_invweight0.data[b] = avg
            mf.dof_invweight0.data[b + 1] = avg
            mf.dof_invweight0.data[b + 2] = avg

    # ── tendon_invweight0[i] = J_i M^-1 J_i^T (engine_setconst.c:256-271) ─────
    # The tendon limit row's diagApprox, and the only place a spatial tendon's
    # stiffness is set. Evaluated at the SAME qpos0 pose as everything above —
    # a spatial tendon's Jacobian is configuration-dependent, so this is a
    # reference value, exactly like body_invweight0.
    #
    # ⚠ AND `tendon_length0` (LENGTH_REF), which nothing assigned until
    # 2026-08-12. `TendonData.length_ref` defaulted to 0.0 and no parser ever
    # wrote it, so every `<equality><tendon>` was solved against a target of
    # ZERO rather than the tendon's rest length. It went unnoticed because
    # `m.tendon_length0` is 0.0 for every equality tendon in the tree —
    # quadruped's four leg couplings and manipulator/stacker's `coupling` all
    # constrain a signed sum of joint angles that vanishes at qpos0 — so the
    # default was accidentally right on the only models that read it. The
    # first tendon with a nonzero rest length would have been welded to zero.
    # This is the same pose the invweights use, which is what qpos0 means.
    # ⚠ A RUNTIME `if`, AND STILL A GUARD (3c-b). Unlike the equality passes
    # below, this block BINDS TENSOR VIEWS before it loops, and a model with
    # no tendons allocates those record tensors at length 0 — the zero-extent
    # operand this tree has crashed on. So the gate survives; only the
    # dimension it reads changes from comptime (`DIM_POISON` on a dynamic
    # provider, hence silently skipped) to live.
    if d.dims.get_ntendon() > 0:
        var dm = d.dims
        var rl_META = rl1(MODEL_META_SIZE)
        var rl_TEN = rl2(dm.get_ntendon(), MODEL_TENDON_SIZE)
        var rl_SITE = rl2(dm.get_nsite(), MODEL_SITE_SIZE)
        # ⚠ NOT CLAMPED TO 1. `rl_SITE` above is built the same way and a
        # FIXED-tendon-only model reaches here with nsite 0, so a zero-extent
        # model view is already exercised on this path. The tensor is indexed
        # only where a wrap entry names a geom, which cannot happen at ngeom 0.
        var rl_GEOM_V = rl2(dm.get_ngeom(), MODEL_GEOM_SIZE)
        var rl_BODY_V = rl2(dm.get_nbody(), MODEL_BODY_SIZE)
        var rl_JOINT_V = rl2(dm.get_njoint(), MODEL_JOINT_SIZE)
        var rl_B3_V = rl2(1, dm.get_nbody() * 3)
        var rl_B4_V = rl2(1, dm.get_nbody() * 4)
        var rl_CDOF_V = rl2(1, dm.get_nv() * 6)

        var meta_v = mf.meta.tt_dyn["cpu", DYN1](rl_META)
        var ten_v = mf.tendons.tt_dyn["cpu", DYN2](rl_TEN)
        var site_v = mf.sites.tt_dyn["cpu", DYN2](rl_SITE)
        var geoms_v = mf.geoms.tt_dyn["cpu", DYN2](rl_GEOM_V)
        var bodies_v = mf.bodies.tt_dyn["cpu", DYN2](rl_BODY_V)
        var joints_v = mf.joints.tt_dyn["cpu", DYN2](rl_JOINT_V)
        var stcom_v = d.subtree_com.tt_dyn["cpu", DYN2](rl_B3_V)
        var xpos_v = d.xpos.tt_dyn["cpu", DYN2](rl_B3_V)
        var xquat_v = d.xquat.tt_dyn["cpu", DYN2](rl_B4_V)
        var cdof_v = sc.cdof.tt_dyn["cpu", DYN2](rl_CDOF_V)

        var tJ = Scratch[Scalar[DTYPE], cap[D.NV]()](nv, fill=Scalar[DTYPE](0))
        for t in range(ntendon):
            var kind = Int(mf.tendons.data[t * MODEL_TENDON_SIZE + TENDON_IDX_KIND])
            var len0 = Scalar[DTYPE](0)
            if kind == TENDON_KIND_SPATIAL:
                len0 = spatial_tendon_length_jac[
                    DTYPE, cap[D.NV](), 1
                ](
                    0, t, dm, ten_v, site_v, geoms_v, bodies_v, joints_v,
                    meta_v,
                    stcom_v, cdof_v, xpos_v, xquat_v, tJ,
                )
            else:
                # Fixed tendon: J[dof_adr(j)] = coef_j, length = sum coef*qpos0.
                for i in range(nv):
                    tJ[i] = Scalar[DTYPE](0)
                var nj = Int(
                    mf.tendons.data[t * MODEL_TENDON_SIZE + TENDON_IDX_NUM_JOINTS]
                )
                for k in range(nj):
                    var jid = Int(
                        mf.tendons.data[
                            t * MODEL_TENDON_SIZE + TENDON_IDX_JOINT_0 + k
                        ]
                    )
                    if jid < 0 or jid >= njoint:
                        continue
                    var dadr = Int(
                        mf.joints.data[jid * MODEL_JOINT_SIZE + JOINT_IDX_DOF_ADR]
                    )
                    var qadr_t = Int(
                        mf.joints.data[
                            jid * MODEL_JOINT_SIZE + JOINT_IDX_QPOS_ADR
                        ]
                    )
                    var coef_t = mf.tendons.data[
                        t * MODEL_TENDON_SIZE + TENDON_IDX_COEF_0 + k
                    ]
                    tJ[dadr] += coef_t
                    len0 += coef_t * d.qpos.data[qadr_t]

            mf.tendons.data[
                t * MODEL_TENDON_SIZE + TENDON_IDX_LENGTH_REF
            ] = len0

            for q in range(nv):
                sc.fnet.data[q] = tJ[q]
            ldl_solve["cpu", DTYPE, BATCH=1](mf, sc)
            var iw = Scalar[DTYPE](0)
            for q in range(nv):
                iw += tJ[q] * sc.qacc_ws.data[q]
            mf.tendons.data[
                t * MODEL_TENDON_SIZE + TENDON_IDX_INVWEIGHT0
            ] = iw

    # ── weld relpose = the qpos0 relative pose (mjCEquality::Compile) ─────────
    # MJCF's default `relpose` is `0 0 0 0 0 0 0`, and a ZERO QUATERNION means
    # "derive it": MuJoCo's compiler writes the relative pose the two bodies
    # already have at the reference configuration. Verified against the
    # runtime (MuJoCo 3.10.0) — a body at z = 0.3 welded to the world compiles
    # to `(0, 0, -0.3, 1, 0, 0, 0)`, and an EXPLICIT identity quaternion is
    # kept as identity, so the test really is on the quaternion and not on
    # whether the attribute was written.
    #
    # ⚠ WE DEFAULTED TO IDENTITY UNTIL 2026-08-12, which welds the two bodies
    # COINCIDENT instead of holding their initial offset. On a body welded to
    # the world at z = 0.3 that dragged it to the origin until the floor
    # stopped it. Invisible because sawyer — the only model in the tree with a
    # weld — has mocap and hand at the SAME pose at qpos0, so identity was
    # accidentally the right answer. Third such default in this arc, after the
    # phantom body mass and `tendon_length0`.
    #
    # Here rather than in the parser because it needs the WORLD poses, which
    # exist only after FK; this function has already run FK at qpos0 for the
    # inverse weights above, which is the same reference configuration
    # `mjCEquality::Compile` uses.
    # ── connect anchor_b = the qpos0 anchor in body2's frame (mj_setConst) ───
    # MuJoCo stores a connect's anchor TWICE — `eq_data[0:3]` in body1's frame
    # (the MJCF `anchor` attribute) and `eq_data[3:6]` in body2's — and derives
    # the second at the reference configuration:
    #
    #   pos = xpos[b1] + xmat[b1] * data[0:3]          (mj_local2Global)
    #   data[3:6] = xmat[b2]^T * (pos - xpos[b2])
    #
    # `engine_setconst.c`, "compute missing eq_data for body constraints".
    # BYTE-IDENTICAL in MuJoCo 3.3.6, 3.6.0, 3.11.0 and `mujoco-main`, so
    # there is no version risk here even though none of those trees is the
    # 3.10.0 runtime. Confirmed against that runtime directly: bodies at
    # (0.1,0.2,0.3) and (0.7,-0.3,0.4) with `anchor="0.05 0.06 0.07"` compile
    # to `eq_data[3:6] = (-0.55, 0.56, -0.03)`.
    #
    # ⚠ UNCONDITIONAL, unlike the weld derivation below — there is no "the
    # user already wrote it" escape, because MJCF gives a connect no attribute
    # that lands in `eq_data[3:6]`. Do NOT copy the weld's zero-quaternion
    # guard here.
    #
    # ⚠ SKIPPED FOR SITE SEMANTICS. MuJoCo zeroes `eq_data` for a site-based
    # connect and reads `site_xpos` instead; we store the site offsets in the
    # anchor slots (see `_fill_equality`), so deriving would overwrite site2's
    # offset with a value MuJoCo never computes.
    # The live equality count, for the two passes below.
    var eq_n = mf.dims.get_nequality()
    # ⚠ THE `comptime if eq_n > 0:` HERE IS GONE, NOT CONVERTED (3c-b).
    # It wrapped nothing but `for e in range(eq_n)`, which already runs
    # zero times at zero equalities — the gate only ever removed dead code.
    # On a dynamic provider it read `-1 > 0` and skipped the block outright,
    # which is a silent behaviour change rather than an optimisation.
    for e in range(eq_n):
        var eo = e * MODEL_EQ_SIZE
        if Int(mf.equality.data[eo + EQ_IDX_TYPE]) != EQ_CONNECT:
            continue
        if Int(mf.equality.data[eo + EQ_IDX_OBJTYPE]) == EQ_OBJ_SITE:
            continue

        var cba = Int(mf.equality.data[eo + EQ_IDX_BODY_A])
        var cbb = Int(mf.equality.data[eo + EQ_IDX_BODY_B])
        if cba < 0 or cba >= nbody or cbb < 0 or cbb >= nbody:
            continue

        # world anchor = xpos[b1] + R(xquat[b1]) * anchor_a
        var wrot = quat_rotate[DTYPE](
            d.xquat.data[cba * 4 + 0],
            d.xquat.data[cba * 4 + 1],
            d.xquat.data[cba * 4 + 2],
            d.xquat.data[cba * 4 + 3],
            mf.equality.data[eo + EQ_IDX_ANCHOR_AX],
            mf.equality.data[eo + EQ_IDX_ANCHOR_AY],
            mf.equality.data[eo + EQ_IDX_ANCHOR_AZ],
        )
        var wax = d.xpos.data[cba * 3 + 0] + wrot[0]
        var way = d.xpos.data[cba * 3 + 1] + wrot[1]
        var waz = d.xpos.data[cba * 3 + 2] + wrot[2]

        # anchor_b = R(xquat[b2])^T * (world anchor - xpos[b2])
        var qb = quat_conjugate[DTYPE](
            d.xquat.data[cbb * 4 + 0],
            d.xquat.data[cbb * 4 + 1],
            d.xquat.data[cbb * 4 + 2],
            d.xquat.data[cbb * 4 + 3],
        )
        var ab = quat_rotate[DTYPE](
            qb[0], qb[1], qb[2], qb[3],
            wax - d.xpos.data[cbb * 3 + 0],
            way - d.xpos.data[cbb * 3 + 1],
            waz - d.xpos.data[cbb * 3 + 2],
        )
        mf.equality.data[eo + EQ_IDX_ANCHOR_BX] = ab[0]
        mf.equality.data[eo + EQ_IDX_ANCHOR_BY] = ab[1]
        mf.equality.data[eo + EQ_IDX_ANCHOR_BZ] = ab[2]

    # Same as the CONNECT pass above: the gate was the loop bound.
    for e in range(eq_n):
        var eo = e * MODEL_EQ_SIZE
        if Int(mf.equality.data[eo + EQ_IDX_TYPE]) != EQ_WELD:
            continue

        var rq_x = mf.equality.data[eo + EQ_IDX_RELPOSE_X]
        var rq_y = mf.equality.data[eo + EQ_IDX_RELPOSE_Y]
        var rq_z = mf.equality.data[eo + EQ_IDX_RELPOSE_Z]
        var rq_w = mf.equality.data[eo + EQ_IDX_RELPOSE_W]
        if (
            rq_x * rq_x + rq_y * rq_y + rq_z * rq_z + rq_w * rq_w
            > Scalar[DTYPE](1e-12)
        ):
            continue  # written explicitly — leave it alone

        var ba = Int(mf.equality.data[eo + EQ_IDX_BODY_A])
        var bb = Int(mf.equality.data[eo + EQ_IDX_BODY_B])
        if ba < 0 or ba >= nbody or bb < 0 or bb >= nbody:
            continue

        var qa = quat_conjugate[DTYPE](
            d.xquat.data[ba * 4 + 0],
            d.xquat.data[ba * 4 + 1],
            d.xquat.data[ba * 4 + 2],
            d.xquat.data[ba * 4 + 3],
        )
        # Pose of body B expressed in body A's frame — the direction
        # `mj_equalityAnchors` then reads back as body A's anchor.
        var rel = quat_rotate[DTYPE](
            qa[0], qa[1], qa[2], qa[3],
            d.xpos.data[bb * 3 + 0] - d.xpos.data[ba * 3 + 0],
            d.xpos.data[bb * 3 + 1] - d.xpos.data[ba * 3 + 1],
            d.xpos.data[bb * 3 + 2] - d.xpos.data[ba * 3 + 2],
        )
        var relq = quat_mul[DTYPE](
            qa[0], qa[1], qa[2], qa[3],
            d.xquat.data[bb * 4 + 0],
            d.xquat.data[bb * 4 + 1],
            d.xquat.data[bb * 4 + 2],
            d.xquat.data[bb * 4 + 3],
        )

        mf.equality.data[eo + EQ_IDX_ANCHOR_AX] = rel[0]
        mf.equality.data[eo + EQ_IDX_ANCHOR_AY] = rel[1]
        mf.equality.data[eo + EQ_IDX_ANCHOR_AZ] = rel[2]
        mf.equality.data[eo + EQ_IDX_RELPOSE_X] = relq[0]
        mf.equality.data[eo + EQ_IDX_RELPOSE_Y] = relq[1]
        mf.equality.data[eo + EQ_IDX_RELPOSE_Z] = relq[2]
        mf.equality.data[eo + EQ_IDX_RELPOSE_W] = relq[3]
