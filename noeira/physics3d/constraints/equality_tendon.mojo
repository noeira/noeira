"""Equality (connect/weld) + fixed-tendon constraints over per-field tensors
(migration P4, single-source).

Per-field ports of `build_and_solve_equality_gpu`
(constraints/constraint_builder_gpu.mojo:1030) and
`build_and_solve_tendon_gpu` (:1573), plus the weld/angular Jacobian rows
they call (dynamics/jacobian.mojo: `compute_weld_jacobian_row_gpu`,
`compute_angular_jacobian_row_gpu`) — arithmetic verbatim. Both are per-env
functions called by the fields PGS contact solve
(constraints/contact_solve.mojo) at the exact legacy position: after
the joint-limits pass, before the friction phase, with the legacy
PGS_ITERATIONS iteration count.

✅ THE LEGACY invweight0-OFFSET QUIRK IS GONE FROM BOTH BUILDERS, and
`_legacy_invw_read` / `_legacy_tendon_col` are deleted. Recorded because the
quirk was NOT a harmless addressing detail, and because this docstring itself
outlived it by long enough to mislead.

What it was: the legacy builders computed their diagApprox offsets with
NTENDON/NSITE left at their 0 defaults, so on any model with tendons and/or
sites the reads landed NTENDON*MODEL_TENDON_SIZE + NSITE*MODEL_SITE_SIZE BEFORE
the invweight0 records — inside the tendon / site records. A helper reproduced
that addressing BIT-EXACTLY, "including the misreads".

What it cost: on sawyer (8 sites, 0 tendons) the weld's `delta = 2*24 = 48`
landed in the SITE records and returned `sites[6, 0]` = **27.0** — column 0 of a
site record is THE BODY INDEX THE SITE IS ATTACHED TO. A body id served as an
inverse inertia where the correct diagApprox is 6.1056, so R was 4.4x too large
and the mocap weld 4.4x too soft: half of defect 28
(`docs/DM_CONTROL_PORT_PHASE2.md` §23). The other half was three phantom
kilograms in the mass matrix.

⚠ "Port the CODE, not the algorithm" is the right instinct for a reference whose
quirks are load-bearing — but the reference is MuJoCo, not our own deleted
legacy. Reproducing OUR bug bit-exactly preserved nothing MuJoCo does.

Both builders now read the quantity MuJoCo reads:
  * connect/weld -> `body_invweight0[b, 0]` (translation rows) and
    `[b, 1]` (weld orientation rows), summed over the two bodies
    (engine_core_constraint.c:1447 / :1461).
  * equality TENDON -> `tendon_invweight0[id]`, ONE number, the tendon's own
    J M^-1 J^T at qpos0 (:1091) — NOT the sum of `dof_invweight0` over its
    joints, which is the mjEQ_JOINT rule (:1090) applied to the wrong
    constraint type. Fixed earlier in `01f7b62f`; see the note at the tendon
    R computation below.

⚠⚠ THIS PARAGRAPH USED TO SAY THE TENDON PATH STILL HAD THE BUG. It did not —
`01f7b62f` had already fixed it — and the stale claim was read off this
docstring and reported as an open defect. A comment describing code is a
hypothesis about the code. GREP THE CALL SITES."""

from ..fields.scratch import Scratch
from ..dynamics.jac_contact_row import _body_chain, _affects, CHAIN_CAP
from std.math import abs, pow, sqrt
from layout import TileTensor, TensorLayout

from ..types import _max_one, EQ_CONNECT, EQ_WELD, EQ_JOINT
from ..joint_types import JNT_FREE, JNT_BALL
from ..kinematics.quat_math import quat_mul, quat_conjugate, quat_rotate
from ..gpu.constants import (
    MODEL_META_IDX_TIMESTEP,
    MODEL_BODY_SIZE,
    MODEL_JOINT_SIZE,
    MODEL_META_SIZE,
    MODEL_EQ_SIZE,
    MODEL_TENDON_SIZE,
    MODEL_SITE_SIZE,
    MODEL_GEOM_SIZE,
    MODEL_META_IDX_NJOINT,
    MODEL_META_IDX_NEQUALITY,
    MODEL_META_IDX_NTENDON,
    BODY_IDX_PARENT,
    BODY_IDX_ROOTID,
    JOINT_IDX_TYPE,
    JOINT_IDX_BODY_ID,
    JOINT_IDX_DOF_ADR,
    JOINT_IDX_QPOS_ADR,
    JOINT_IDX_QPOS0,
    EQ_IDX_TYPE,
    EQ_IDX_BODY_A,
    EQ_IDX_BODY_B,
    EQ_IDX_ANCHOR_AX,
    EQ_IDX_ANCHOR_AY,
    EQ_IDX_ANCHOR_AZ,
    EQ_IDX_ANCHOR_BX,
    EQ_IDX_ANCHOR_BY,
    EQ_IDX_ANCHOR_BZ,
    EQ_IDX_RELPOSE_X,
    EQ_IDX_RELPOSE_Y,
    EQ_IDX_RELPOSE_Z,
    EQ_IDX_RELPOSE_W,
    EQ_IDX_SOLREF_0,
    EQ_IDX_SOLREF_1,
    EQ_IDX_SOLIMP_0,
    EQ_IDX_SOLIMP_1,
    EQ_IDX_SOLIMP_2,
    EQ_IDX_SOLIMP_3,
    EQ_IDX_SOLIMP_4,
    EQ_IDX_TORQUESCALE,
    TENDON_IDX_NUM_JOINTS,
    TENDON_IDX_IS_EQUALITY,
    TENDON_IDX_JOINT_0,
    TENDON_IDX_COEF_0,
    TENDON_MAX_WRAPS,
    TENDON_IDX_LENGTH_REF,
    TENDON_IDX_SOLREF_0,
    TENDON_IDX_SOLREF_1,
    TENDON_IDX_SOLIMP_0,
    TENDON_IDX_SOLIMP_1,
    TENDON_IDX_SOLIMP_2,
    TENDON_IDX_SOLIMP_3,
    TENDON_IDX_SOLIMP_4,
    TENDON_IDX_INVWEIGHT0,
    TENDON_IDX_KIND,
    TENDON_KIND_SPATIAL,
)


from .constraint_data import solref_spring_damper
from ..dynamics.tendon import spatial_tendon_length_jac


# =============================================================================
from ..fields import Dims, DimsLike, Scratch, cap, may_exist
# Weld / angular Jacobian rows (ports of dynamics/jacobian.mojo GPU rows)
# =============================================================================


@always_inline
def _weld_jacobian_row[
    DTYPE: DType,
    V_SIZE: Int,
    L_SUBTREE_COM: TensorLayout,
    L_JOINTS: TensorLayout,
    L_BODIES: TensorLayout,
    L_MMETA: TensorLayout,
    L_CDOF: TensorLayout,
](
    env: Int,
    subtree_com: TileTensor[DTYPE, L_SUBTREE_COM, MutAnyOrigin],
    joints: TileTensor[DTYPE, L_JOINTS, MutAnyOrigin],
    bodies: TileTensor[DTYPE, L_BODIES, MutAnyOrigin],
    mmeta: TileTensor[DTYPE, L_MMETA, MutAnyOrigin],
    cdof: TileTensor[DTYPE, L_CDOF, MutAnyOrigin],
    body_a: Int,
    body_b: Int,
    pos_a_x: Scalar[DTYPE],
    pos_a_y: Scalar[DTYPE],
    pos_a_z: Scalar[DTYPE],
    pos_b_x: Scalar[DTYPE],
    pos_b_y: Scalar[DTYPE],
    pos_b_z: Scalar[DTYPE],
    dir_x: Scalar[DTYPE],
    dir_y: Scalar[DTYPE],
    dir_z: Scalar[DTYPE],
    nv: Int,
    mut J_row: Scratch[Scalar[DTYPE], V_SIZE],
):
    """Weld/connect Jacobian row: J = J_a(at pos_a) - J_b(at pos_b)
    (verbatim from compute_weld_jacobian_row_gpu).

    Each body's Jacobian uses its OWN anchor position, unlike the contact
    Jacobian which uses a single shared contact point.

    ⚠⚠ `nv`, NOT `V_SIZE`, AND `Scratch`, NOT `Array`. `V_SIZE` is
    `cap[D.NV]()`, which is **0 on a dynamic provider** — an `Array` of
    length 0 that this loop then failed to zero and the body below indexed,
    giving "index 0 is out of bounds, valid range is 0 to -1" from a runtime
    model with a weld. It survived the 3a/3b sweeps because the ONLY caller
    sat behind `comptime if D.CAP_NEQUALITY > 0`, false on every dynamic
    provider, so the path was unreachable until `may_exist` opened it. A
    latent zero-size container behind a dead gate is invisible to every audit
    that greps for reachable code.
    """
    for i in range(nv):
        J_row[i] = 0

    var num_joints = Int(
        rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_NJOINT])
    )

    for j_idx in range(num_joints):
        var jnt_type = Int(
            rebind[Scalar[DTYPE]](joints[j_idx, JOINT_IDX_TYPE])
        )
        var joint_body = Int(
            rebind[Scalar[DTYPE]](joints[j_idx, JOINT_IDX_BODY_ID])
        )
        var dof_adr = Int(
            rebind[Scalar[DTYPE]](joints[j_idx, JOINT_IDX_DOF_ADR])
        )

        # Check if this joint affects body_a
        var affects_a = False
        if body_a == joint_body:
            affects_a = True
        else:
            var current = body_a
            while current > 0:
                var current_parent = Int(
                    rebind[Scalar[DTYPE]](bodies[current, BODY_IDX_PARENT])
                )
                if current_parent == joint_body:
                    affects_a = True
                    break
                current = current_parent

        # Check if this joint affects body_b
        var affects_b = False
        if body_b > 0:
            if body_b == joint_body:
                affects_b = True
            else:
                var current_b = body_b
                while current_b > 0:
                    var current_parent_b = Int(
                        rebind[Scalar[DTYPE]](
                            bodies[current_b, BODY_IDX_PARENT]
                        )
                    )
                    if current_parent_b == joint_body:
                        affects_b = True
                        break
                    current_b = current_parent_b

        if not affects_a and not affects_b:
            continue

        var num_dof = 1
        if jnt_type == JNT_FREE:
            num_dof = 6
        elif jnt_type == JNT_BALL:
            num_dof = 3

        # Reference point for cdof cross product
        var jb_rootid = Int(
            rebind[Scalar[DTYPE]](bodies[joint_body, BODY_IDX_ROOTID])
        )
        var ref_x = rebind[Scalar[DTYPE]](
            subtree_com[env, jb_rootid * 3 + 0]
        )
        var ref_y = rebind[Scalar[DTYPE]](
            subtree_com[env, jb_rootid * 3 + 1]
        )
        var ref_z = rebind[Scalar[DTYPE]](
            subtree_com[env, jb_rootid * 3 + 2]
        )

        for d in range(num_dof):
            var dof_idx = dof_adr + d
            var ang_x = cdof[env, dof_idx * 6 + 0]
            var ang_y = cdof[env, dof_idx * 6 + 1]
            var ang_z = cdof[env, dof_idx * 6 + 2]
            var lin_x = cdof[env, dof_idx * 6 + 3]
            var lin_y = cdof[env, dof_idx * 6 + 4]
            var lin_z = cdof[env, dof_idx * 6 + 5]

            if affects_a:
                # Jacobian at body_a's anchor point
                var ra_x = pos_a_x - ref_x
                var ra_y = pos_a_y - ref_y
                var ra_z = pos_a_z - ref_z
                var cx = ang_y * ra_z - ang_z * ra_y
                var cy = ang_z * ra_x - ang_x * ra_z
                var cz = ang_x * ra_y - ang_y * ra_x
                var val = (lin_x + cx) * dir_x + (lin_y + cy) * dir_y + (lin_z + cz) * dir_z
                J_row[dof_idx] += rebind[Scalar[DTYPE]](val)

            if affects_b:
                # Jacobian at body_b's anchor point (separate!)
                var rb_x = pos_b_x - ref_x
                var rb_y = pos_b_y - ref_y
                var rb_z = pos_b_z - ref_z
                var cx = ang_y * rb_z - ang_z * rb_y
                var cy = ang_z * rb_x - ang_x * rb_z
                var cz = ang_x * rb_y - ang_y * rb_x
                var val = (lin_x + cx) * dir_x + (lin_y + cy) * dir_y + (lin_z + cz) * dir_z
                J_row[dof_idx] -= rebind[Scalar[DTYPE]](val)


@always_inline
def _angular_jacobian_row_eq[
    DTYPE: DType,
    V_SIZE: Int,
    L_JOINTS: TensorLayout,
    L_BODIES: TensorLayout,
    L_MMETA: TensorLayout,
    L_CDOF: TensorLayout,
](
    env: Int,
    joints: TileTensor[DTYPE, L_JOINTS, MutAnyOrigin],
    bodies: TileTensor[DTYPE, L_BODIES, MutAnyOrigin],
    mmeta: TileTensor[DTYPE, L_MMETA, MutAnyOrigin],
    cdof: TileTensor[DTYPE, L_CDOF, MutAnyOrigin],
    contact_body_a: Int,
    contact_body_b: Int,
    dir_x: Scalar[DTYPE],
    dir_y: Scalar[DTYPE],
    dir_z: Scalar[DTYPE],
    nv: Int,
    mut J_row: Scratch[Scalar[DTYPE], V_SIZE],
):
    """Angular-only Jacobian row (verbatim from
    compute_angular_jacobian_row_gpu; duplicate of
    contact_solve._angular_jacobian_row — that module imports
    THIS one, so importing back would be circular).

    J[dof] = cdof_angular[dof] . dir (bilateral: body_a - body_b).

    ⚠ `nv`, NOT `V_SIZE` — see `_weld_jacobian_row` above for the failure.
    """
    for i in range(nv):
        J_row[i] = 0

    var num_joints = Int(
        rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_NJOINT])
    )

    var chain_a = Scratch[Int, CHAIN_CAP](CHAIN_CAP, uninitialized=0)
    var chain_b = Scratch[Int, CHAIN_CAP](CHAIN_CAP, uninitialized=0)
    var n_a = _body_chain[DTYPE](bodies, contact_body_a, chain_a)
    var n_b = _body_chain[DTYPE](bodies, contact_body_b, chain_b)
    for j_idx in range(num_joints):
        var jnt_type = Int(
            rebind[Scalar[DTYPE]](joints[j_idx, JOINT_IDX_TYPE])
        )
        var joint_body = Int(
            rebind[Scalar[DTYPE]](joints[j_idx, JOINT_IDX_BODY_ID])
        )
        var dof_adr = Int(
            rebind[Scalar[DTYPE]](joints[j_idx, JOINT_IDX_DOF_ADR])
        )

        # Check if this joint affects body_a
        var affects_a = _affects[DTYPE](
            bodies, chain_a, n_a, contact_body_a, joint_body
        )
        var affects_b = _affects[DTYPE](
            bodies, chain_b, n_b, contact_body_b, joint_body
        )

        if not affects_a and not affects_b:
            continue

        var num_dof = 1
        if jnt_type == JNT_FREE:
            num_dof = 6
        elif jnt_type == JNT_BALL:
            num_dof = 3

        for d in range(num_dof):
            var dof_idx = dof_adr + d

            # Angular-only: just dot product of angular cdof with direction
            var ang_x = cdof[env, dof_idx * 6 + 0]
            var ang_y = cdof[env, dof_idx * 6 + 1]
            var ang_z = cdof[env, dof_idx * 6 + 2]

            var val = ang_x * dir_x + ang_y * dir_y + ang_z * dir_z

            if affects_a:
                J_row[dof_idx] += rebind[Scalar[DTYPE]](val)
            if affects_b:
                J_row[dof_idx] -= rebind[Scalar[DTYPE]](val)


# =============================================================================
# Equality constraints (port of build_and_solve_equality_gpu)
# =============================================================================


@always_inline
@always_inline
def _cross3[DTYPE: DType](
    ax: Scalar[DTYPE], ay: Scalar[DTYPE], az: Scalar[DTYPE],
    bx: Scalar[DTYPE], by: Scalar[DTYPE], bz: Scalar[DTYPE],
) -> Array[Scalar[DTYPE], 3]:
    var r = Array[Scalar[DTYPE], 3](fill=Scalar[DTYPE](0))
    r[0] = ay * bz - az * by
    r[1] = az * bx - ax * bz
    r[2] = ax * by - ay * bx
    return r^


def _comvel_and_cdof_dot[
    DTYPE: DType,
    CV_CAP: Int,
    CDD_CAP: Int,
    L_JOINTS: TensorLayout,
    L_BODIES: TensorLayout,
    L_MMETA: TensorLayout,
    L_CDOF: TensorLayout,
    L_QVEL: TensorLayout,
](
    env: Int,
    nbody: Int,
    nv: Int,
    joints: TileTensor[DTYPE, L_JOINTS, MutAnyOrigin],
    bodies: TileTensor[DTYPE, L_BODIES, MutAnyOrigin],
    mmeta: TileTensor[DTYPE, L_MMETA, MutAnyOrigin],
    cdof: TileTensor[DTYPE, L_CDOF, MutAnyOrigin],
    qvel: TileTensor[DTYPE, L_QVEL, MutAnyOrigin],
    mut cv_b: Scratch[Scalar[DTYPE], CV_CAP],
    mut cdd: Scratch[Scalar[DTYPE], CDD_CAP],
):
    """`mj_comVel`: per-body spatial velocity `cvel` (ang 0:3, lin 3:6, in
    the c-frame) and per-dof `cdof_dot = cvel_before_dof x cdof`, the same
    recurrence `rne.mojo` runs inline. Bodies are visited in index order,
    which is parent-first for every model the parser builds."""
    var njoint = Int(rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_NJOINT]))
    for k in range(nbody * 6):
        cv_b[k] = Scalar[DTYPE](0)
    for k in range(nv * 6):
        cdd[k] = Scalar[DTYPE](0)
    for b in range(1, nbody):
        var par = Int(rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_PARENT]))
        var wx = Scalar[DTYPE](0)
        var wy = Scalar[DTYPE](0)
        var wz = Scalar[DTYPE](0)
        var vx = Scalar[DTYPE](0)
        var vy = Scalar[DTYPE](0)
        var vz = Scalar[DTYPE](0)
        if par > 0:
            wx = cv_b[par * 6 + 0]
            wy = cv_b[par * 6 + 1]
            wz = cv_b[par * 6 + 2]
            vx = cv_b[par * 6 + 3]
            vy = cv_b[par * 6 + 4]
            vz = cv_b[par * 6 + 5]
        for j in range(njoint):
            if Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_BODY_ID])) != b:
                continue
            var jt = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_TYPE]))
            var adr = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_DOF_ADR]))
            var d_lo = 0
            var nd = 1
            if jt == JNT_FREE:
                # translation dofs: cdof_dot = 0, cvel accumulates
                for dd in range(3):
                    var dof = adr + dd
                    var qd = rebind[Scalar[DTYPE]](qvel[env, dof])
                    wx += rebind[Scalar[DTYPE]](cdof[env, dof * 6 + 0]) * qd
                    wy += rebind[Scalar[DTYPE]](cdof[env, dof * 6 + 1]) * qd
                    wz += rebind[Scalar[DTYPE]](cdof[env, dof * 6 + 2]) * qd
                    vx += rebind[Scalar[DTYPE]](cdof[env, dof * 6 + 3]) * qd
                    vy += rebind[Scalar[DTYPE]](cdof[env, dof * 6 + 4]) * qd
                    vz += rebind[Scalar[DTYPE]](cdof[env, dof * 6 + 5]) * qd
                d_lo = 3
                nd = 6
            elif jt == JNT_BALL:
                nd = 3
            # rotational / scalar dofs: cdof_dot from the PRE-dof cvel, then
            # accumulate (all of a ball's three read the same cvel)
            for dd in range(d_lo, nd):
                var dof = adr + dd
                var sax = rebind[Scalar[DTYPE]](cdof[env, dof * 6 + 0])
                var say = rebind[Scalar[DTYPE]](cdof[env, dof * 6 + 1])
                var saz = rebind[Scalar[DTYPE]](cdof[env, dof * 6 + 2])
                var slx = rebind[Scalar[DTYPE]](cdof[env, dof * 6 + 3])
                var sly = rebind[Scalar[DTYPE]](cdof[env, dof * 6 + 4])
                var slz = rebind[Scalar[DTYPE]](cdof[env, dof * 6 + 5])
                var ca = _cross3[DTYPE](wx, wy, wz, sax, say, saz)
                var cl1 = _cross3[DTYPE](wx, wy, wz, slx, sly, slz)
                var cl2 = _cross3[DTYPE](vx, vy, vz, sax, say, saz)
                cdd[dof * 6 + 0] = ca[0]
                cdd[dof * 6 + 1] = ca[1]
                cdd[dof * 6 + 2] = ca[2]
                cdd[dof * 6 + 3] = cl1[0] + cl2[0]
                cdd[dof * 6 + 4] = cl1[1] + cl2[1]
                cdd[dof * 6 + 5] = cl1[2] + cl2[2]
            for dd in range(d_lo, nd):
                var dof = adr + dd
                var qd = rebind[Scalar[DTYPE]](qvel[env, dof])
                wx += rebind[Scalar[DTYPE]](cdof[env, dof * 6 + 0]) * qd
                wy += rebind[Scalar[DTYPE]](cdof[env, dof * 6 + 1]) * qd
                wz += rebind[Scalar[DTYPE]](cdof[env, dof * 6 + 2]) * qd
                vx += rebind[Scalar[DTYPE]](cdof[env, dof * 6 + 3]) * qd
                vy += rebind[Scalar[DTYPE]](cdof[env, dof * 6 + 4]) * qd
                vz += rebind[Scalar[DTYPE]](cdof[env, dof * 6 + 5]) * qd
        cv_b[b * 6 + 0] = wx
        cv_b[b * 6 + 1] = wy
        cv_b[b * 6 + 2] = wz
        cv_b[b * 6 + 3] = vx
        cv_b[b * 6 + 4] = vy
        cv_b[b * 6 + 5] = vz


def _jdotv_point[
    DTYPE: DType,
    CV_CAP: Int,
    CDD_CAP: Int,
    L_SUBTREE_COM: TensorLayout,
    L_JOINTS: TensorLayout,
    L_BODIES: TensorLayout,
    L_MMETA: TensorLayout,
    L_CDOF: TensorLayout,
    L_QVEL: TensorLayout,
](
    env: Int,
    subtree_com: TileTensor[DTYPE, L_SUBTREE_COM, MutAnyOrigin],
    joints: TileTensor[DTYPE, L_JOINTS, MutAnyOrigin],
    bodies: TileTensor[DTYPE, L_BODIES, MutAnyOrigin],
    mmeta: TileTensor[DTYPE, L_MMETA, MutAnyOrigin],
    cdof: TileTensor[DTYPE, L_CDOF, MutAnyOrigin],
    qvel: TileTensor[DTYPE, L_QVEL, MutAnyOrigin],
    cv_b: Scratch[Scalar[DTYPE], CV_CAP],
    cdd: Scratch[Scalar[DTYPE], CDD_CAP],
    body: Int,
    px: Scalar[DTYPE],
    py: Scalar[DTYPE],
    pz: Scalar[DTYPE],
) -> Array[Scalar[DTYPE], 6]:
    """`mj_jacDot(point, body) * qvel` — the translational (0:3) and rotational
    (3:6) parts of the point Jacobian's time derivative applied to `qvel`.
    Port of engine_core_util.c `mj_jacDot` folded with the matvec: over the
    dof chain of `body`, `jacp_dot[i] = cdof_dot_lin + cdof_dot_ang x offset
    + cdof_ang x pvel_lin`, `jacr_dot[i] = cdof_dot_ang`, with `cdof_dot`
    of a quaternion dof (ball, free rotation) recomputed from the dof body's
    FULL cvel as MuJoCo does. Body 0 (world) returns zeros."""
    var out = Array[Scalar[DTYPE], 6](fill=Scalar[DTYPE](0))
    if body <= 0:
        return out^
    var root = Int(rebind[Scalar[DTYPE]](bodies[body, BODY_IDX_ROOTID]))
    var ox = px - rebind[Scalar[DTYPE]](subtree_com[env, root * 3 + 0])
    var oy = py - rebind[Scalar[DTYPE]](subtree_com[env, root * 3 + 1])
    var oz = pz - rebind[Scalar[DTYPE]](subtree_com[env, root * 3 + 2])
    # point linear velocity: cvel_lin + cvel_ang x offset (mju_transformSpatial)
    var bw = _cross3[DTYPE](
        cv_b[body * 6 + 0], cv_b[body * 6 + 1], cv_b[body * 6 + 2], ox, oy, oz
    )
    var pvx = cv_b[body * 6 + 3] + bw[0]
    var pvy = cv_b[body * 6 + 4] + bw[1]
    var pvz = cv_b[body * 6 + 5] + bw[2]
    var njoint = Int(rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_NJOINT]))
    for j in range(njoint):
        var jb = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_BODY_ID]))
        var affects = jb == body
        if not affects:
            var cur = body
            while cur > 0:
                var par = Int(rebind[Scalar[DTYPE]](bodies[cur, BODY_IDX_PARENT]))
                if par == jb:
                    affects = True
                    break
                cur = par
        if not affects:
            continue
        var jt = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_TYPE]))
        var adr = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_DOF_ADR]))
        var nd = 1
        if jt == JNT_FREE:
            nd = 6
        elif jt == JNT_BALL:
            nd = 3
        for dd in range(nd):
            var dof = adr + dd
            var qd = rebind[Scalar[DTYPE]](qvel[env, dof])
            var sax = rebind[Scalar[DTYPE]](cdof[env, dof * 6 + 0])
            var say = rebind[Scalar[DTYPE]](cdof[env, dof * 6 + 1])
            var saz = rebind[Scalar[DTYPE]](cdof[env, dof * 6 + 2])
            var cax = cdd[dof * 6 + 0]
            var cay = cdd[dof * 6 + 1]
            var caz = cdd[dof * 6 + 2]
            var clx = cdd[dof * 6 + 3]
            var cly = cdd[dof * 6 + 4]
            var clz = cdd[dof * 6 + 5]
            var is_quat = jt == JNT_BALL or (jt == JNT_FREE and dd >= 3)
            if is_quat:
                # crossMotion(cvel[dof body], cdof) — the FULL body velocity
                var slx = rebind[Scalar[DTYPE]](cdof[env, dof * 6 + 3])
                var sly = rebind[Scalar[DTYPE]](cdof[env, dof * 6 + 4])
                var slz = rebind[Scalar[DTYPE]](cdof[env, dof * 6 + 5])
                var fwx = cv_b[jb * 6 + 0]
                var fwy = cv_b[jb * 6 + 1]
                var fwz = cv_b[jb * 6 + 2]
                var fvx = cv_b[jb * 6 + 3]
                var fvy = cv_b[jb * 6 + 4]
                var fvz = cv_b[jb * 6 + 5]
                var ca = _cross3[DTYPE](fwx, fwy, fwz, sax, say, saz)
                var cl1 = _cross3[DTYPE](fwx, fwy, fwz, slx, sly, slz)
                var cl2 = _cross3[DTYPE](fvx, fvy, fvz, sax, say, saz)
                cax = ca[0]
                cay = ca[1]
                caz = ca[2]
                clx = cl1[0] + cl2[0]
                cly = cl1[1] + cl2[1]
                clz = cl1[2] + cl2[2]
            var t1 = _cross3[DTYPE](cax, cay, caz, ox, oy, oz)
            var t2 = _cross3[DTYPE](sax, say, saz, pvx, pvy, pvz)
            out[0] += (clx + t1[0] + t2[0]) * qd
            out[1] += (cly + t1[1] + t2[1]) * qd
            out[2] += (clz + t1[2] + t2[2]) * qd
            out[3] += cax * qd
            out[4] += cay * qd
            out[5] += caz * qd
    return out^


@always_inline
def _qmul_wxyz[DTYPE: DType](
    a: Array[Scalar[DTYPE], 4], b: Array[Scalar[DTYPE], 4]
) -> Array[Scalar[DTYPE], 4]:
    """`mju_mulQuat` in MuJoCo's (w,x,y,z) order."""
    var r = Array[Scalar[DTYPE], 4](fill=Scalar[DTYPE](0))
    r[0] = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]
    r[1] = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2]
    r[2] = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1]
    r[3] = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]
    return r^


@always_inline
def _qaxis_wxyz[DTYPE: DType](
    q: Array[Scalar[DTYPE], 4],
    vx: Scalar[DTYPE], vy: Scalar[DTYPE], vz: Scalar[DTYPE],
) -> Array[Scalar[DTYPE], 4]:
    """`mju_mulQuatAxis`: q * (0, v), (w,x,y,z) order."""
    var r = Array[Scalar[DTYPE], 4](fill=Scalar[DTYPE](0))
    r[0] = -q[1] * vx - q[2] * vy - q[3] * vz
    r[1] = q[0] * vx + q[2] * vz - q[3] * vy
    r[2] = q[0] * vy + q[3] * vx - q[1] * vz
    r[3] = q[0] * vz + q[1] * vy - q[2] * vx
    return r^


@always_inline
def _qderiv_wxyz[DTYPE: DType](
    q: Array[Scalar[DTYPE], 4],
    vx: Scalar[DTYPE], vy: Scalar[DTYPE], vz: Scalar[DTYPE],
) -> Array[Scalar[DTYPE], 4]:
    """`mju_derivQuat`: 0.5 * (0, v) * q, (w,x,y,z) order."""
    var r = Array[Scalar[DTYPE], 4](fill=Scalar[DTYPE](0))
    var h = Scalar[DTYPE](0.5)
    r[0] = h * (-vx * q[1] - vy * q[2] - vz * q[3])
    r[1] = h * (vx * q[0] + vy * q[3] - vz * q[2])
    r[2] = h * (-vx * q[3] + vy * q[0] + vz * q[1])
    r[3] = h * (vx * q[2] - vy * q[1] + vz * q[0])
    return r^


@always_inline
def _qneg_wxyz[DTYPE: DType](
    q: Array[Scalar[DTYPE], 4]
) -> Array[Scalar[DTYPE], 4]:
    var r = Array[Scalar[DTYPE], 4](fill=Scalar[DTYPE](0))
    r[0] = q[0]
    r[1] = -q[1]
    r[2] = -q[2]
    r[3] = -q[3]
    return r^


def build_weld_equality_rows[
    DTYPE: DType,
    V_SIZE: Int,
    EQ_CAP: Int,
    EQJ_CAP: Int,
    D: DimsLike,
    L_QPOS: TensorLayout,
    L_QVEL: TensorLayout,
    L_XPOS: TensorLayout,
    L_XQUAT: TensorLayout,
    L_JOINTS: TensorLayout,
    L_BODIES: TensorLayout,
    L_MMETA: TensorLayout,
    L_EQUALITY: TensorLayout,
    L_BODY_INVWEIGHT0: TensorLayout,
    L_DOF_INVWEIGHT0: TensorLayout,
    L_CDOF: TensorLayout,
    L_M_INV: TensorLayout,
](
    env: Int,
    dims: D,
    qpos: TileTensor[DTYPE, L_QPOS, MutAnyOrigin],
    qvel: TileTensor[DTYPE, L_QVEL, MutAnyOrigin],
    xpos: TileTensor[DTYPE, L_XPOS, MutAnyOrigin],
    xquat: TileTensor[DTYPE, L_XQUAT, MutAnyOrigin],
    subtree_com: TileTensor[DTYPE, L_XPOS, MutAnyOrigin],
    joints: TileTensor[DTYPE, L_JOINTS, MutAnyOrigin],
    bodies: TileTensor[DTYPE, L_BODIES, MutAnyOrigin],
    mmeta: TileTensor[DTYPE, L_MMETA, MutAnyOrigin],
    equality: TileTensor[DTYPE, L_EQUALITY, MutAnyOrigin],
    body_invweight0: TileTensor[DTYPE, L_BODY_INVWEIGHT0, MutAnyOrigin],
    dof_invweight0: TileTensor[DTYPE, L_DOF_INVWEIGHT0, MutAnyOrigin],
    cdof: TileTensor[DTYPE, L_CDOF, MutAnyOrigin],
    m_inv: TileTensor[DTYPE, L_M_INV, MutAnyOrigin],
    mut eq_K: Scratch[Scalar[DTYPE], EQ_CAP],
    mut eq_bias: Scratch[Scalar[DTYPE], EQ_CAP],
    mut eq_inv_K_imp: Scratch[Scalar[DTYPE], EQ_CAP],
    mut eq_J: Scratch[Scalar[DTYPE], EQJ_CAP],
    mut eq_MinvJ: Scratch[Scalar[DTYPE], EQJ_CAP],
) -> Int:
    """Build the connect/weld equality ROWS — J, bias, K and 1/(K+R). NO SOLVE.

    Split out so the Newton solver can put these rows in its system (defect
    29a). Body lifted VERBATIM from `_equality_env`, so the PGS post-pass
    (still used by the PGS / CG / island solvers) and the Newton rows stay
    bit-identical by construction rather than by review.
    """

    var nv = dims.get_nv()
    var nbody = dims.get_nbody()
    var njoint = dims.get_njoint()
    var nequality = dims.get_nequality()
    # The row budget is 6 per equality (weld); this is the live bound the
    # `num_eq_rows >= max_eq_rows` guards below compare against. It is NOT
    # `EQ_CAP`, which is 0 on a dynamic provider.
    var max_eq_rows = 6 * nequality
    if nequality == 0:
        return 0

    var neq = Int(rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_NEQUALITY]))
    if neq == 0:
        return 0
    if neq > nequality:
        neq = nequality

    # ⚠ THE LENGTH IS `nv`, THE CAP IS `V_SIZE`. `Scratch` puts this on the
    # stack when the cap is non-zero and on the heap when it is 0 (a dynamic
    # provider); the constructor's `nv` is load-bearing on the heap leg and
    # inert on the stack one, which is exactly why getting it wrong is
    # invisible to every static-leg gate.
    var J_row = Scratch[Scalar[DTYPE], V_SIZE](nv, fill=Scalar[DTYPE](0))
    var num_eq_rows = 0

    # ── `J̇·v` for connect/weld rows (mj_Jdotv, MuJoCo 3.10) ──────────────
    #
    # ⚠⚠ THIS TERM DID NOT EXIST IN THE TREES THIS FILE WAS TRANSCRIBED FROM.
    # MuJoCo 3.10's `mj_referenceConstraint` ends with `mj_Jdotv`, which
    # subtracts the anchor's `J̇·qvel` (the centripetal/Coriolis part of its
    # acceleration) from `aref` on every connect and weld row; 3.3.6, 3.5.1
    # and 3.6.0 have no such routine. The term is exactly zero at rest, so
    # every connect/weld gate written at rest was green without it. It was
    # found on ToddlerBot (PERFORMANCE.md §13.29): its neck is a closed loop
    # of four near-hard connects whose anchors move while the robot sways,
    # and `waist_yaw` walked 4e-02 from MuJoCo in 100 steps with J, D, pos,
    # vel, KBIP, invweights and M all matched to 1e-14 — `aref` was the one
    # number left, off by exactly this. Our `bias` is `-aref`, so it ADDS.
    #
    # The pass below is `mj_comVel` (per-body cvel, per-dof cdof_dot), run
    # once per call and only when a connect or weld is present; the two
    # `_jdotv_point` calls per row are `mj_jacDot` folded with the matvec.
    var _has_cw = False
    for _e in range(neq):
        var _t = Int(rebind[Scalar[DTYPE]](equality[_e, EQ_IDX_TYPE]))
        if _t == EQ_CONNECT or _t == EQ_WELD:
            _has_cw = True
    # ⚠ SIZED BY THE PROVIDER'S CAPS, NOT THE HEAP: this builder also runs
    # inside the GPU Newton kernels, where a heap `Scratch[.., 0]` is
    # undefined (it was — the tendon-fields gate's CPU/GPU parity broke on
    # it). `cap[]` is the static extent on a compile-time model and 0 on a
    # dynamic one, where the heap leg is the right one anyway.
    comptime CV_CAP = 6 * cap[D.NBODY]()
    comptime CDD_CAP = 6 * cap[D.NV]()
    var cv_b = Scratch[Scalar[DTYPE], CV_CAP](
        (nbody if _has_cw else 1) * 6, fill=Scalar[DTYPE](0)
    )
    var cdd = Scratch[Scalar[DTYPE], CDD_CAP](
        (nv if _has_cw else 1) * 6, fill=Scalar[DTYPE](0)
    )
    if _has_cw:
        _comvel_and_cdof_dot[DTYPE, CV_CAP, CDD_CAP](
            env, nbody, nv, joints, bodies, mmeta, cdof, qvel, cv_b, cdd
        )


    for eq_i in range(neq):
        var eq_type = Int(rebind[Scalar[DTYPE]](equality[eq_i, EQ_IDX_TYPE]))
        var body_a = Int(
            rebind[Scalar[DTYPE]](equality[eq_i, EQ_IDX_BODY_A])
        )
        var body_b = Int(
            rebind[Scalar[DTYPE]](equality[eq_i, EQ_IDX_BODY_B])
        )

        # Read anchors
        var anc_ax = rebind[Scalar[DTYPE]](equality[eq_i, EQ_IDX_ANCHOR_AX])
        var anc_ay = rebind[Scalar[DTYPE]](equality[eq_i, EQ_IDX_ANCHOR_AY])
        var anc_az = rebind[Scalar[DTYPE]](equality[eq_i, EQ_IDX_ANCHOR_AZ])
        var anc_bx = rebind[Scalar[DTYPE]](equality[eq_i, EQ_IDX_ANCHOR_BX])
        var anc_by = rebind[Scalar[DTYPE]](equality[eq_i, EQ_IDX_ANCHOR_BY])
        var anc_bz = rebind[Scalar[DTYPE]](equality[eq_i, EQ_IDX_ANCHOR_BZ])

        # Read solref/solimp
        var sr_tc = rebind[Scalar[DTYPE]](equality[eq_i, EQ_IDX_SOLREF_0])
        var sr_dr = rebind[Scalar[DTYPE]](equality[eq_i, EQ_IDX_SOLREF_1])
        var si_dmin = rebind[Scalar[DTYPE]](equality[eq_i, EQ_IDX_SOLIMP_0])
        var si_dmax = rebind[Scalar[DTYPE]](equality[eq_i, EQ_IDX_SOLIMP_1])
        var si_width = rebind[Scalar[DTYPE]](equality[eq_i, EQ_IDX_SOLIMP_2])
        var si_midpoint = rebind[Scalar[DTYPE]](
            equality[eq_i, EQ_IDX_SOLIMP_3]
        )
        var si_power = rebind[Scalar[DTYPE]](equality[eq_i, EQ_IDX_SOLIMP_4])
        if si_width < Scalar[DTYPE](1e-6):
            si_width = Scalar[DTYPE](1e-6)
        # Clamp BOTH ends to [mjMINIMP, mjMAXIMP] as MuJoCo does before
        # interpolating (engine_core_constraint.c:1284-1287).
        comptime MJE_MINIMP = Scalar[DTYPE](0.0001)
        comptime MJE_MAXIMP = Scalar[DTYPE](0.9999)
        if si_dmin < MJE_MINIMP:
            si_dmin = MJE_MINIMP
        elif si_dmin > MJE_MAXIMP:
            si_dmin = MJE_MAXIMP
        if si_dmax < MJE_MINIMP:
            si_dmax = MJE_MINIMP
        elif si_dmax > MJE_MAXIMP:
            si_dmax = MJE_MAXIMP
        if si_power < Scalar[DTYPE](1):
            si_power = Scalar[DTYPE](1)
        # solref -> (K, B), including MuJoCo's DIRECT form for a NEGATIVE
        # solref. See `constraints/constraint_data.solref_spring_damper` — the
        # formula lived in twelve copy-pasted sites until 2026-08-03.
        var (eq_K_spring, eq_B_damp) = solref_spring_damper[DTYPE](
            sr_tc, sr_dr, si_dmax,
            rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_TIMESTEP]),
        )

        # ── mjEQ_JOINT — ONE row coupling two scalar joints ──────────────────
        #
        # engine_core_constraint.c:556. With `dif = q2 - q2_ref`:
        #
        #   cpos  = q1 - q1_ref - p0 - (p1*dif + p2*dif^2 + p3*dif^3 + p4*dif^4)
        #   deriv = p1 + 2*p2*dif + 3*p3*dif^2 + 4*p4*dif^3
        #   J     = e_dof1 - deriv * e_dof2
        #
        # With joint2 absent (`eq_obj2id < 0`) the polynomial drops out and the
        # row is `q1 - q1_ref - p0` against `e_dof1` alone.
        #
        # ⚠ THE REFERENCE IS `qpos0`, NOT ZERO. MuJoCo subtracts
        # `m->qpos0[jnt_qposadr]` from BOTH joints; on a joint with a nonzero
        # `ref` the two differ and the pair would be held at the wrong offset.
        #
        # ⚠ `body_a`/`body_b` ARE JOINT INDICES HERE — the slots are reused
        # per type exactly as MuJoCo reuses `eq_obj1id`. See `EQ_IDX_TYPE`.
        #
        # ⚠ `eq_K` IS `J M^-1 J^T`, NOT the spring constant. The callers
        # recover `R = 1/eq_inv_K_imp - eq_K` and use `1/R` as `efc_D`, so
        # putting the stiffness here silently changes every consumer's R. The
        # weld path above has the same contract; this branch mirrors it.
        # diagApprox for a joint equality is `dof_invweight0` summed over the
        # one or two dofs (engine_core_constraint.c, mj_diagApprox).
        if eq_type == EQ_JOINT:
            if num_eq_rows >= max_eq_rows:
                break
            if body_a < 0 or body_a >= njoint:
                continue
            var jdadr1 = Int(
                rebind[Scalar[DTYPE]](joints[body_a, JOINT_IDX_DOF_ADR])
            )
            if jdadr1 < 0 or jdadr1 >= nv:
                continue

            var p0 = rebind[Scalar[DTYPE]](equality[eq_i, EQ_IDX_ANCHOR_AX])
            var p1 = rebind[Scalar[DTYPE]](equality[eq_i, EQ_IDX_ANCHOR_AY])
            var p2 = rebind[Scalar[DTYPE]](equality[eq_i, EQ_IDX_ANCHOR_AZ])
            var p3 = rebind[Scalar[DTYPE]](equality[eq_i, EQ_IDX_ANCHOR_BX])
            var p4 = rebind[Scalar[DTYPE]](equality[eq_i, EQ_IDX_ANCHOR_BY])

            var jqadr1 = Int(
                rebind[Scalar[DTYPE]](joints[body_a, JOINT_IDX_QPOS_ADR])
            )
            var jres = (
                rebind[Scalar[DTYPE]](qpos[env, jqadr1])
                - rebind[Scalar[DTYPE]](joints[body_a, JOINT_IDX_QPOS0])
                - p0
            )

            # ⚠⚠ `nv`, NOT `V_SIZE` — THE THIRD TIME IN THIS FILE, AND THE
            # ONLY COPY THAT WAS LEFT. `V_SIZE` is `cap[D.NV]()`, which is
            # **0 on a dynamic provider**, so on the runtime path this loop
            # zeroed NOTHING and the row inherited the PREVIOUS row's
            # Jacobian. `_weld_jacobian_row` (:171) and the weld branch
            # (:935) both carry a warning about exactly this; the joint
            # branch was written the same way and never converted.
            #
            # ⚠ IT IS INVISIBLE WITH ONE EQUALITY. `J_row` is allocated
            # `fill=0`, so the FIRST row is correct whatever this loop does —
            # a model with a single equality agrees with MuJoCo to 1e-18 and
            # nothing is wrong. It takes a SECOND joint equality for the
            # stale entries to appear, and then they appear in the second
            # row's `J`: measured on a two-chain fixture, row 1 got
            # `J = [1, -1, 0, 0]` and row 2 got `[1, -1, 1, -1]` where MuJoCo
            # has `[0, 0, 1, -1]` — a row coupling two chains that share no
            # dof. aloha, whose two arms each weld their gripper fingers, is
            # the model in this tree that has two.
            for i in range(nv):
                J_row[i] = Scalar[DTYPE](0)
            J_row[jdadr1] = Scalar[DTYPE](1)
            var jdA = rebind[Scalar[DTYPE]](dof_invweight0[jdadr1])

            if body_b >= 0 and body_b < njoint:
                var jqadr2 = Int(
                    rebind[Scalar[DTYPE]](joints[body_b, JOINT_IDX_QPOS_ADR])
                )
                var jdadr2 = Int(
                    rebind[Scalar[DTYPE]](joints[body_b, JOINT_IDX_DOF_ADR])
                )
                var jdif = (
                    rebind[Scalar[DTYPE]](qpos[env, jqadr2])
                    - rebind[Scalar[DTYPE]](joints[body_b, JOINT_IDX_QPOS0])
                )
                var jd2 = jdif * jdif
                jres -= p1 * jdif + p2 * jd2 + p3 * jd2 * jdif + p4 * jd2 * jd2
                var jderiv = (
                    p1
                    + Scalar[DTYPE](2) * p2 * jdif
                    + Scalar[DTYPE](3) * p3 * jd2
                    + Scalar[DTYPE](4) * p4 * jd2 * jdif
                )
                if jdadr2 >= 0 and jdadr2 < nv:
                    J_row[jdadr2] = J_row[jdadr2] - jderiv
                    jdA += rebind[Scalar[DTYPE]](dof_invweight0[jdadr2])

            # Impedance on the row's own residual. `dim` is 1 here, so
            # MuJoCo's `mju_norm(efc_pos+i, dim)` is exactly |jres| — none of
            # the weld's norm-over-six-rows subtlety applies.
            var jimp: Scalar[DTYPE]
            if si_dmin == si_dmax or si_width <= Scalar[DTYPE](0):
                jimp = Scalar[DTYPE](0.5) * (si_dmin + si_dmax)
            else:
                var jxe = abs(jres) / si_width
                var jye: Scalar[DTYPE]
                if jxe <= Scalar[DTYPE](0):
                    jye = Scalar[DTYPE](0)
                elif jxe >= Scalar[DTYPE](1):
                    jye = Scalar[DTYPE](1)
                elif si_power == Scalar[DTYPE](1):
                    jye = jxe
                elif jxe <= si_midpoint:
                    jye = pow(jxe, si_power) / pow(
                        si_midpoint, si_power - Scalar[DTYPE](1)
                    )
                else:
                    jye = Scalar[DTYPE](1) - pow(
                        Scalar[DTYPE](1) - jxe, si_power
                    ) / pow(
                        Scalar[DTYPE](1) - si_midpoint,
                        si_power - Scalar[DTYPE](1),
                    )
                jimp = si_dmin + jye * (si_dmax - si_dmin)
            if jimp < Scalar[DTYPE](1e-6):
                jimp = Scalar[DTYPE](1e-6)

            var jk = Scalar[DTYPE](0)
            var jv = Scalar[DTYPE](0)
            for i in range(nv):
                eq_J[num_eq_rows * nv + i] = J_row[i]
                var jmij = Scalar[DTYPE](0)
                for k2 in range(nv):
                    jmij += (
                        rebind[Scalar[DTYPE]](m_inv[env, i * nv + k2])
                        * J_row[k2]
                    )
                eq_MinvJ[num_eq_rows * nv + i] = jmij
                jk += J_row[i] * jmij
                jv += J_row[i] * rebind[Scalar[DTYPE]](qvel[env, i])
            if jk < Scalar[DTYPE](1e-10):
                jk = Scalar[DTYPE](1e-10)

            var jR = (Scalar[DTYPE](1) - jimp) / jimp * jdA
            if jR < Scalar[DTYPE](1e-14):
                jR = Scalar[DTYPE](1e-14)

            eq_K[num_eq_rows] = jk
            eq_bias[num_eq_rows] = eq_K_spring * jimp * jres + eq_B_damp * jv
            eq_inv_K_imp[num_eq_rows] = Scalar[DTYPE](1) / (jk + jR)
            num_eq_rows += 1
            continue

        # Compute world anchor A: xpos[body_a] + quat_rotate(xquat[body_a], anchor_a)
        var xpos_a_x = rebind[Scalar[DTYPE]](
            xpos[env, body_a * 3 + 0]
        )
        var xpos_a_y = rebind[Scalar[DTYPE]](
            xpos[env, body_a * 3 + 1]
        )
        var xpos_a_z = rebind[Scalar[DTYPE]](
            xpos[env, body_a * 3 + 2]
        )
        var xquat_a_x = rebind[Scalar[DTYPE]](
            xquat[env, body_a * 4 + 0]
        )
        var xquat_a_y = rebind[Scalar[DTYPE]](
            xquat[env, body_a * 4 + 1]
        )
        var xquat_a_z = rebind[Scalar[DTYPE]](
            xquat[env, body_a * 4 + 2]
        )
        var xquat_a_w = rebind[Scalar[DTYPE]](
            xquat[env, body_a * 4 + 3]
        )
        var rot_a = quat_rotate[DTYPE](
            xquat_a_x, xquat_a_y, xquat_a_z, xquat_a_w, anc_ax, anc_ay, anc_az
        )
        var world_ax = xpos_a_x + rot_a[0]
        var world_ay = xpos_a_y + rot_a[1]
        var world_az = xpos_a_z + rot_a[2]

        # Compute world anchor B
        var world_bx: Scalar[DTYPE]
        var world_by: Scalar[DTYPE]
        var world_bz: Scalar[DTYPE]
        if body_b > 0:
            var xpos_b_x = rebind[Scalar[DTYPE]](
                xpos[env, body_b * 3 + 0]
            )
            var xpos_b_y = rebind[Scalar[DTYPE]](
                xpos[env, body_b * 3 + 1]
            )
            var xpos_b_z = rebind[Scalar[DTYPE]](
                xpos[env, body_b * 3 + 2]
            )
            var xquat_b_x = rebind[Scalar[DTYPE]](
                xquat[env, body_b * 4 + 0]
            )
            var xquat_b_y = rebind[Scalar[DTYPE]](
                xquat[env, body_b * 4 + 1]
            )
            var xquat_b_z = rebind[Scalar[DTYPE]](
                xquat[env, body_b * 4 + 2]
            )
            var xquat_b_w = rebind[Scalar[DTYPE]](
                xquat[env, body_b * 4 + 3]
            )
            var rot_b = quat_rotate[DTYPE](
                xquat_b_x,
                xquat_b_y,
                xquat_b_z,
                xquat_b_w,
                anc_bx,
                anc_by,
                anc_bz,
            )
            world_bx = xpos_b_x + rot_b[0]
            world_by = xpos_b_y + rot_b[1]
            world_bz = xpos_b_z + rot_b[2]
        else:
            world_bx = anc_bx
            world_by = anc_by
            world_bz = anc_bz

        # J̇·v of the two anchors (zeros for the world) — see the note above
        var jdv_a = _jdotv_point[DTYPE, CV_CAP, CDD_CAP](
            env, subtree_com, joints, bodies, mmeta, cdof, qvel, cv_b, cdd,
            body_a, world_ax, world_ay, world_az,
        )
        var jdv_b = _jdotv_point[DTYPE, CV_CAP, CDD_CAP](
            env, subtree_com, joints, bodies, mmeta, cdof, qvel, cv_b, cdd,
            body_b, world_bx, world_by, world_bz,
        )
        var pos_err_x = world_ax - world_bx
        var pos_err_y = world_ay - world_by
        var pos_err_z = world_az - world_bz

        # --- 3 position rows (connect + weld) ---
        var dirs = Array[Scalar[DTYPE], 9](fill=Scalar[DTYPE](0))
        dirs[0] = Scalar[DTYPE](1)  # x-axis: (1,0,0)
        dirs[4] = Scalar[DTYPE](1)  # y-axis: (0,1,0)
        dirs[8] = Scalar[DTYPE](1)  # z-axis: (0,0,1)

        var pos_errs = Array[Scalar[DTYPE], 3](fill=Scalar[DTYPE](0))
        pos_errs[0] = pos_err_x
        pos_errs[1] = pos_err_y
        pos_errs[2] = pos_err_z

        # ── weld orientation residual, hoisted, and ONE impedance ────────────
        #
        # MuJoCo's impedance argument for an equality is the NORM OVER ALL ITS
        # ROWS, not the per-row residual: `mj_constraintUpdate` sets
        # `*pos = mju_norm(efc_pos+i, 6)` for a weld and `norm(..., 3)` for a
        # connect (engine_core_constraint.c:2071), with `dim` 6 and 3.
        #
        # ⚠ WE USED THE PER-ROW VALUE, AND THE BIAS CANNOT SEE IT: a row whose
        # own residual is 0 contributes `K*imp*0 = 0` for any `imp`. It shows
        # up only through `R = (1-imp)/imp * diagApprox`. On a tilted weld,
        # rows 3 and 5 have zero individual residual and so took imp = dmin =
        # 0.9 against MuJoCo's 0.95 — `efc_D` 0.0300 against 0.0633, 53% off on
        # the compliance of two of the six rows, on every weld in the repo.
        # Found only once the rows were diffed against `efc_D` directly;
        # `efc_J` and `efc_aref` both matched exactly with the bug present.
        var rot_errs = Array[Scalar[DTYPE], 3](fill=Scalar[DTYPE](0))
        var rot_jdv = Array[Scalar[DTYPE], 3](fill=Scalar[DTYPE](0))
        var ts = Scalar[DTYPE](1)
        var cqb = Array[Scalar[DTYPE], 4](fill=Scalar[DTYPE](0))
        var qrel = Array[Scalar[DTYPE], 4](fill=Scalar[DTYPE](0))
        cqb[3] = Scalar[DTYPE](1)
        qrel[3] = Scalar[DTYPE](1)
        if eq_type == EQ_WELD:
            # Read relpose
            var rp_x = rebind[Scalar[DTYPE]](
                equality[eq_i, EQ_IDX_RELPOSE_X]
            )
            var rp_y = rebind[Scalar[DTYPE]](
                equality[eq_i, EQ_IDX_RELPOSE_Y]
            )
            var rp_z = rebind[Scalar[DTYPE]](
                equality[eq_i, EQ_IDX_RELPOSE_Z]
            )
            var rp_w = rebind[Scalar[DTYPE]](
                equality[eq_i, EQ_IDX_RELPOSE_W]
            )

            # Compute orientation error: 0.5 * imag(conj(quat_b) * quat_a * relpose)
            var qa_x = xquat_a_x
            var qa_y = xquat_a_y
            var qa_z = xquat_a_z
            var qa_w = xquat_a_w

            var qb_x: Scalar[DTYPE]
            var qb_y: Scalar[DTYPE]
            var qb_z: Scalar[DTYPE]
            var qb_w: Scalar[DTYPE]
            if body_b > 0:
                qb_x = rebind[Scalar[DTYPE]](
                    xquat[env, body_b * 4 + 0]
                )
                qb_y = rebind[Scalar[DTYPE]](
                    xquat[env, body_b * 4 + 1]
                )
                qb_z = rebind[Scalar[DTYPE]](
                    xquat[env, body_b * 4 + 2]
                )
                qb_w = rebind[Scalar[DTYPE]](
                    xquat[env, body_b * 4 + 3]
                )
            else:
                qb_x = Scalar[DTYPE](0)
                qb_y = Scalar[DTYPE](0)
                qb_z = Scalar[DTYPE](0)
                qb_w = Scalar[DTYPE](1)

            # MuJoCo's two working quaternions (engine_core_constraint.c:686):
            #   quat  = q0 (x) relpose        `qrel` here
            #   quat1 = neg(q1)               `cqb` here
            var cqb_t = quat_conjugate[DTYPE](qb_x, qb_y, qb_z, qb_w)
            var qrel_t = quat_mul[DTYPE](
                qa_x, qa_y, qa_z, qa_w, rp_x, rp_y, rp_z, rp_w
            )
            cqb[0] = cqb_t[0]
            cqb[1] = cqb_t[1]
            cqb[2] = cqb_t[2]
            cqb[3] = cqb_t[3]
            qrel[0] = qrel_t[0]
            qrel[1] = qrel_t[1]
            qrel[2] = qrel_t[2]
            qrel[3] = qrel_t[3]
            var err_q = quat_mul[DTYPE](
                cqb[0], cqb[1], cqb[2], cqb[3],
                qrel[0], qrel[1], qrel[2], qrel[3],
            )
            # imaginary part, SCALED BY TORQUESCALE.
            #
            # ⚠ NO 0.5 HERE. MuJoCo's residual is `mju_scl3(cpos+3, quat2+1,
            # torquescale)` — the 0.5 belongs to the JACOBIAN
            # (`jac = 0.5*quat3[1..3]`), not to the error. We had it on the
            # residual and not on the Jacobian, which is a different constraint:
            # scaling one side of `J qacc + bias = 0` without the other moves
            # the row's effective stiffness.
            #
            # MuJoCo applies `eq_data[10]` twice — to the residual
            # (`mju_scl3(cpos+3, quat2+1, torquescale)`,
            # engine_core_constraint.c:701) and to the rotational Jacobian
            # (`mju_scl(jac[0]+3*nv, ..., torquescale, 3*nv)`, :721) — so it
            # scales the whole rotational half of the weld, not just its error.
            # Both are needed: scaling only the residual changes the target
            # without changing the row's effective stiffness, which is a
            # different constraint from MuJoCo's.
            #
            # ⚠ UNIMPLEMENTED UNTIL 2026-08-12. MJCF defaults it to 1, so this
            # was inert on every dm_control model — but MetaWorld's
            # `reset_mocap_welds` sets 5.0, so sawyer's mocap weld was 5x too
            # soft in orientation against the environment we claim to port. It
            # stayed invisible because the gate's REFERENCE side also set 1.0:
            # the protocol had been written to match our limitation.
            ts = rebind[Scalar[DTYPE]](
                equality[eq_i, EQ_IDX_TORQUESCALE]
            )
            rot_errs[0] = err_q[0] * ts
            rot_errs[1] = err_q[1] * ts
            rot_errs[2] = err_q[2] * ts

        var pn_sq = (
            pos_errs[0] * pos_errs[0]
            + pos_errs[1] * pos_errs[1]
            + pos_errs[2] * pos_errs[2]
        )
        if eq_type == EQ_WELD:
            pn_sq += (
                rot_errs[0] * rot_errs[0]
                + rot_errs[1] * rot_errs[1]
                + rot_errs[2] * rot_errs[2]
            )
        var pos_norm = sqrt(pn_sq)
        var imp_eq: Scalar[DTYPE]
        if si_dmin == si_dmax or si_width <= Scalar[DTYPE](0):
            imp_eq = Scalar[DTYPE](0.5) * (si_dmin + si_dmax)
        else:
            var xe = pos_norm / si_width
            var ye: Scalar[DTYPE]
            if xe <= Scalar[DTYPE](0):
                ye = Scalar[DTYPE](0)
            elif xe >= Scalar[DTYPE](1):
                ye = Scalar[DTYPE](1)
            elif si_power == Scalar[DTYPE](1):
                ye = xe
            elif xe <= si_midpoint:
                ye = pow(xe, si_power) / pow(
                    si_midpoint, si_power - Scalar[DTYPE](1)
                )
            else:
                ye = Scalar[DTYPE](1) - pow(
                    Scalar[DTYPE](1) - xe, si_power
                ) / pow(
                    Scalar[DTYPE](1) - si_midpoint,
                    si_power - Scalar[DTYPE](1),
                )
            imp_eq = si_dmin + ye * (si_dmax - si_dmin)
        if imp_eq < Scalar[DTYPE](1e-6):
            imp_eq = Scalar[DTYPE](1e-6)


        for d in range(3):
            if num_eq_rows >= max_eq_rows:
                break
            var dx = dirs[d * 3 + 0]
            var dy = dirs[d * 3 + 1]
            var dz = dirs[d * 3 + 2]

            # Compute Jacobian: J = J_a(at world_a) - J_b(at world_b)
            # Each body's Jacobian uses its OWN anchor point (MuJoCo convention)
            # ⚠ `nv`, NOT `V_SIZE` — the cap is 0 on a dynamic provider, so
            # this loop zeroed nothing there. `_weld_jacobian_row` zeroes it
            # again from `nv` anyway; the redundancy is kept because the two
            # were written apart and a future edit could drop either.
            for i in range(nv):
                J_row[i] = 0
            _weld_jacobian_row[
                DTYPE, V_SIZE](
                env,
                subtree_com,
                joints,
                bodies,
                mmeta,
                cdof,
                body_a,
                body_b,
                world_ax,
                world_ay,
                world_az,
                world_bx,
                world_by,
                world_bz,
                dx,
                dy,
                dz,
                nv,
                J_row,
            )

            # Compute K = J @ M_inv @ J^T, store J and MinvJ
            var k: Scalar[DTYPE] = 0
            var v_n: Scalar[DTYPE] = 0
            for i in range(nv):
                eq_J[num_eq_rows * nv + i] = J_row[i]
                var mi_j_sum: Scalar[DTYPE] = 0
                for j_idx in range(nv):
                    mi_j_sum += (
                        rebind[Scalar[DTYPE]](
                            m_inv[env, i * nv + j_idx]
                        )
                        * J_row[j_idx]
                    )
                eq_MinvJ[num_eq_rows * nv + i] = mi_j_sum
                k += J_row[i] * mi_j_sum
                v_n += J_row[i] * rebind[Scalar[DTYPE]](
                    qvel[env, i]
                )

            if k < Scalar[DTYPE](1e-10):
                k = Scalar[DTYPE](1e-10)
            eq_K[num_eq_rows] = k

            # ONE impedance for the whole constraint — see the hoisted block.
            var err_d = pos_errs[d]
            var imp = imp_eq

            # MuJoCo equality bias: bias = -aref = B*vel + K*I*pos
            # where pos is the SIGNED error (not abs). Contact formula uses
            # -K*I*pen because contact pos = -penetration, but equality pos
            # is signed directly.
            var bias = eq_K_spring * imp * err_d + eq_B_damp * v_n
            # + J̇·v (MuJoCo: aref -= jdv1 - jdv2; bias is -aref, so it adds)
            bias += jdv_a[d] - jdv_b[d]
            eq_bias[num_eq_rows] = bias
            # MuJoCo: R = (1-imp)/imp * diagApprox, and diagApprox for a
            # connect/weld row is
            #     body_invweight0[2*b1] + body_invweight0[2*b2]
            # (engine_core_constraint.c:1447 / :1461). Read STRAIGHT out of
            # body_invweight0 — see the ⚠ below for why this no longer goes
            # through `_legacy_invw_read`.
            var diag_eq: Scalar[DTYPE] = 0
            if body_a > 0 and body_a < nbody:
                diag_eq += rebind[Scalar[DTYPE]](body_invweight0[body_a, 0])
            if body_b > 0 and body_b < nbody:
                diag_eq += rebind[Scalar[DTYPE]](body_invweight0[body_b, 0])
            if diag_eq < Scalar[DTYPE](1e-10):
                diag_eq = rebind[Scalar[DTYPE]](k)
            var R_eq = (Scalar[DTYPE](1.0) - imp) / imp * diag_eq
            eq_inv_K_imp[num_eq_rows] = Scalar[DTYPE](1.0) / (
                rebind[Scalar[DTYPE]](k) + R_eq
            )

            num_eq_rows += 1

        # --- 3 orientation rows (weld only) ---
        if eq_type == EQ_WELD:

            # ── the ROTATIONAL JACOBIAN, ported exactly ──────────────────────
            #
            # MuJoCo (engine_core_constraint.c:704-721), per dof j:
            #     axis  = [jac0 - jac1]_col(j)         angular difference
            #     quat2 = neg(q1) (x) axis             `mju_mulQuatAxis`
            #     quat3 = quat2 (x) q0 (x) relpose
            #     jac[3+k][j] = 0.5 * quat3[k+1]
            # then the whole 3xNV block is scaled by torquescale.
            #
            # ⚠⚠ WE USED TO BUILD THREE WORLD-AXIS ROWS INSTEAD —
            # `J = (w_a - w_b) . e_k` for k = x, y, z — which is the same thing
            # ONLY to first order in the orientation error and in the relative
            # orientation. Every gated weld sat at ~0 orientation error
            # (sawyer's mocap tracks the hand), so nothing measured it. The
            # first specimen to put a real moment on a weld disagreed with
            # MuJoCo by 0.086 rad of tilt AT torquescale 1, where the
            # torquescale code above is a no-op.
            #
            # Written straight into `eq_J` rather than through a staging row:
            # the three rows share one pass over the joints, and a per-row
            # buffer would be a fourth V_SIZE array in a Metal-compiled kernel.
            if num_eq_rows + 3 > max_eq_rows:
                continue
            for r in range(3):
                for i in range(nv):
                    eq_J[(num_eq_rows + r) * nv + i] = Scalar[DTYPE](0)

            var n_j = Int(rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_NJOINT]))
            for j_idx in range(n_j):
                var jt_o = Int(
                    rebind[Scalar[DTYPE]](joints[j_idx, JOINT_IDX_TYPE])
                )
                var jb_o = Int(
                    rebind[Scalar[DTYPE]](joints[j_idx, JOINT_IDX_BODY_ID])
                )
                var ja_o = Int(
                    rebind[Scalar[DTYPE]](joints[j_idx, JOINT_IDX_DOF_ADR])
                )

                var aff_a = body_a == jb_o
                if not aff_a:
                    var cur = body_a
                    while cur > 0:
                        var par = Int(
                            rebind[Scalar[DTYPE]](bodies[cur, BODY_IDX_PARENT])
                        )
                        if par == jb_o:
                            aff_a = True
                            break
                        cur = par
                var aff_b = False
                if body_b > 0:
                    aff_b = body_b == jb_o
                    if not aff_b:
                        var curb = body_b
                        while curb > 0:
                            var parb = Int(
                                rebind[Scalar[DTYPE]](
                                    bodies[curb, BODY_IDX_PARENT]
                                )
                            )
                            if parb == jb_o:
                                aff_b = True
                                break
                            curb = parb

                # A dof reachable from BOTH bodies cancels, exactly as the
                # +=/-= in `_angular_jacobian_row_eq` did.
                var sgn = Scalar[DTYPE](0)
                if aff_a:
                    sgn += Scalar[DTYPE](1)
                if aff_b:
                    sgn -= Scalar[DTYPE](1)
                if sgn == Scalar[DTYPE](0):
                    continue

                var ndof_o = 1
                if jt_o == JNT_FREE:
                    ndof_o = 6
                elif jt_o == JNT_BALL:
                    ndof_o = 3

                for dd in range(ndof_o):
                    var dof_i = ja_o + dd
                    var wx = (
                        rebind[Scalar[DTYPE]](cdof[env, dof_i * 6 + 0]) * sgn
                    )
                    var wy = (
                        rebind[Scalar[DTYPE]](cdof[env, dof_i * 6 + 1]) * sgn
                    )
                    var wz = (
                        rebind[Scalar[DTYPE]](cdof[env, dof_i * 6 + 2]) * sgn
                    )

                    # quat2 = cqb (x) (0, w) — MuJoCo's `mju_mulQuatAxis`,
                    # in our (x, y, z, w) storage.
                    var q2w = -(cqb[0] * wx + cqb[1] * wy + cqb[2] * wz)
                    var q2x = cqb[3] * wx + cqb[1] * wz - cqb[2] * wy
                    var q2y = cqb[3] * wy + cqb[2] * wx - cqb[0] * wz
                    var q2z = cqb[3] * wz + cqb[0] * wy - cqb[1] * wx

                    var q3 = quat_mul[DTYPE](
                        q2x, q2y, q2z, q2w,
                        qrel[0], qrel[1], qrel[2], qrel[3],
                    )
                    var half_ts = Scalar[DTYPE](0.5) * ts
                    eq_J[(num_eq_rows + 0) * nv + dof_i] = half_ts * q3[0]
                    eq_J[(num_eq_rows + 1) * nv + dof_i] = half_ts * q3[1]
                    eq_J[(num_eq_rows + 2) * nv + dof_i] = half_ts * q3[2]

            for d in range(3):
                for i in range(nv):
                    J_row[i] = eq_J[num_eq_rows * nv + i]

                # K, store J and MinvJ
                var k: Scalar[DTYPE] = 0
                var v_n: Scalar[DTYPE] = 0
                for i in range(nv):
                    eq_J[num_eq_rows * nv + i] = J_row[i]
                    var mi_j_sum: Scalar[DTYPE] = 0
                    for j_idx in range(nv):
                        mi_j_sum += (
                            rebind[Scalar[DTYPE]](
                                m_inv[env, i * nv + j_idx]
                            )
                            * J_row[j_idx]
                        )
                    eq_MinvJ[num_eq_rows * nv + i] = mi_j_sum
                    k += J_row[i] * mi_j_sum
                    v_n += J_row[i] * rebind[Scalar[DTYPE]](
                        qvel[env, i]
                    )

                if k < Scalar[DTYPE](1e-10):
                    k = Scalar[DTYPE](1e-10)
                eq_K[num_eq_rows] = k

                # ONE impedance for the whole constraint — see above.
                var err_d = rot_errs[d]
                var imp = imp_eq

                # MuJoCo equality bias: bias = K*I*pos + B*vel (signed pos)
                var bias = eq_K_spring * imp * err_d + eq_B_damp * v_n
                # + rotational J̇·v (mj_Jdotv's weld branch): with q0, q1 the
                # two body quaternions (w,x,y,z) and q0r = q0*relpose, the
                # row's J is 0.5*ts*vec(neg(q1)*(J0-J1)*q0r); the product
                # rule gives three terms and MuJoCo subtracts their sum from
                # aref — so it ADDS to bias. Computed once, on d == 0.
                if d == 0:
                    var q0 = Array[Scalar[DTYPE], 4](fill=Scalar[DTYPE](0))
                    q0[0] = rebind[Scalar[DTYPE]](xquat[env, body_a * 4 + 3])
                    q0[1] = rebind[Scalar[DTYPE]](xquat[env, body_a * 4 + 0])
                    q0[2] = rebind[Scalar[DTYPE]](xquat[env, body_a * 4 + 1])
                    q0[3] = rebind[Scalar[DTYPE]](xquat[env, body_a * 4 + 2])
                    var q1 = Array[Scalar[DTYPE], 4](fill=Scalar[DTYPE](0))
                    q1[0] = Scalar[DTYPE](1)
                    if body_b > 0:
                        q1[0] = rebind[Scalar[DTYPE]](xquat[env, body_b * 4 + 3])
                        q1[1] = rebind[Scalar[DTYPE]](xquat[env, body_b * 4 + 0])
                        q1[2] = rebind[Scalar[DTYPE]](xquat[env, body_b * 4 + 1])
                        q1[3] = rebind[Scalar[DTYPE]](xquat[env, body_b * 4 + 2])
                    # ⚠ `qrel` above is ALREADY q_a * relpose; the raw relpose
                    # is `rp_*`. Using `qrel` here applied the body rotation
                    # twice — exact at identity, 0.5% off once rotated.
                    var rp = Array[Scalar[DTYPE], 4](fill=Scalar[DTYPE](0))
                    rp[0] = rebind[Scalar[DTYPE]](equality[eq_i, EQ_IDX_RELPOSE_W])
                    rp[1] = rebind[Scalar[DTYPE]](equality[eq_i, EQ_IDX_RELPOSE_X])
                    rp[2] = rebind[Scalar[DTYPE]](equality[eq_i, EQ_IDX_RELPOSE_Y])
                    rp[3] = rebind[Scalar[DTYPE]](equality[eq_i, EQ_IDX_RELPOSE_Z])
                    var q0r = _qmul_wxyz[DTYPE](q0, rp)
                    var negq1 = _qneg_wxyz[DTYPE](q1)
                    var w1x = cv_b[body_a * 6 + 0]
                    var w1y = cv_b[body_a * 6 + 1]
                    var w1z = cv_b[body_a * 6 + 2]
                    var w2x = Scalar[DTYPE](0)
                    var w2y = Scalar[DTYPE](0)
                    var w2z = Scalar[DTYPE](0)
                    if body_b > 0:
                        w2x = cv_b[body_b * 6 + 0]
                        w2y = cv_b[body_b * 6 + 1]
                        w2z = cv_b[body_b * 6 + 2]
                    var dwx = w1x - w2x
                    var dwy = w1y - w2y
                    var dwz = w1z - w2z
                    var qdot0 = _qderiv_wxyz[DTYPE](q0, w1x, w1y, w1z)
                    var qdot0r = _qmul_wxyz[DTYPE](qdot0, rp)
                    var qdot1 = _qderiv_wxyz[DTYPE](q1, w2x, w2y, w2z)
                    var negqdot1 = _qneg_wxyz[DTYPE](qdot1)
                    var djx = jdv_a[3] - jdv_b[3]
                    var djy = jdv_a[4] - jdv_b[4]
                    var djz = jdv_a[5] - jdv_b[5]
                    var t1 = _qmul_wxyz[DTYPE](
                        _qaxis_wxyz[DTYPE](negqdot1, dwx, dwy, dwz), q0r
                    )
                    var t2 = _qmul_wxyz[DTYPE](
                        _qaxis_wxyz[DTYPE](negq1, djx, djy, djz), q0r
                    )
                    var t3 = _qmul_wxyz[DTYPE](
                        _qaxis_wxyz[DTYPE](negq1, dwx, dwy, dwz), qdot0r
                    )
                    var half_ts_r = Scalar[DTYPE](0.5) * ts
                    rot_jdv[0] = half_ts_r * (t1[1] + t2[1] + t3[1])
                    rot_jdv[1] = half_ts_r * (t1[2] + t2[2] + t3[2])
                    rot_jdv[2] = half_ts_r * (t1[3] + t2[3] + t3[3])
                bias += rot_jdv[d]
                eq_bias[num_eq_rows] = bias
                # MuJoCo takes the ROTATION half of the pair for the weld's
                # orientation rows — `body_invweight0[2*b + (weldcnt > 2)]`,
                # engine_core_constraint.c:1461. Same direct read as the
                # translation rows above.
                var diag_rot: Scalar[DTYPE] = 0
                if body_a > 0 and body_a < nbody:
                    diag_rot += rebind[Scalar[DTYPE]](
                        body_invweight0[body_a, 1]
                    )
                if body_b > 0 and body_b < nbody:
                    diag_rot += rebind[Scalar[DTYPE]](
                        body_invweight0[body_b, 1]
                    )
                if diag_rot < Scalar[DTYPE](1e-10):
                    diag_rot = rebind[Scalar[DTYPE]](k)
                var R_rot = (Scalar[DTYPE](1.0) - imp) / imp * diag_rot
                eq_inv_K_imp[num_eq_rows] = Scalar[DTYPE](1.0) / (
                    rebind[Scalar[DTYPE]](k) + R_rot
                )

                num_eq_rows += 1

    return num_eq_rows


@always_inline
def _equality_env[
    DTYPE: DType,
    V_SIZE: Int,
    NUM_ITERATIONS: Int,
    D: DimsLike,
    L_QPOS: TensorLayout,
    L_QVEL: TensorLayout,
    L_XPOS: TensorLayout,
    L_XQUAT: TensorLayout,
    L_JOINTS: TensorLayout,
    L_BODIES: TensorLayout,
    L_MMETA: TensorLayout,
    L_EQUALITY: TensorLayout,
    L_BODY_INVWEIGHT0: TensorLayout,
    L_DOF_INVWEIGHT0: TensorLayout,
    L_CDOF: TensorLayout,
    L_M_INV: TensorLayout,
](
    env: Int,
    dims: D,
    qpos: TileTensor[DTYPE, L_QPOS, MutAnyOrigin],
    qvel: TileTensor[DTYPE, L_QVEL, MutAnyOrigin],
    xpos: TileTensor[DTYPE, L_XPOS, MutAnyOrigin],
    xquat: TileTensor[DTYPE, L_XQUAT, MutAnyOrigin],
    subtree_com: TileTensor[DTYPE, L_XPOS, MutAnyOrigin],
    joints: TileTensor[DTYPE, L_JOINTS, MutAnyOrigin],
    bodies: TileTensor[DTYPE, L_BODIES, MutAnyOrigin],
    mmeta: TileTensor[DTYPE, L_MMETA, MutAnyOrigin],
    equality: TileTensor[DTYPE, L_EQUALITY, MutAnyOrigin],
    body_invweight0: TileTensor[DTYPE, L_BODY_INVWEIGHT0, MutAnyOrigin],
    dof_invweight0: TileTensor[DTYPE, L_DOF_INVWEIGHT0, MutAnyOrigin],
    cdof: TileTensor[DTYPE, L_CDOF, MutAnyOrigin],
    m_inv: TileTensor[DTYPE, L_M_INV, MutAnyOrigin],
    qacc_constrained: TileTensor[DTYPE, L_QVEL, MutAnyOrigin],
):
    """Build and solve equality constraints (connect + weld) for one env.

    Reads equality constraint definitions from the equality record tensor,
    computes world anchors, Jacobians, impedance, and runs bilateral PGS
    iterations (no lambda >= 0 clamping) on `qacc_constrained`.

    ⚠ IT DOES NOT TAKE `tendons` / `sites` / `dof_invweight0`, and must not be
    given them again. It used to, purely so the deleted `_legacy_invw_read`
    could index across the concatenated record slab — which is how a SITE
    record came to supply a weld's diagApprox (see the module docstring). The
    equality solver has no business reading a tendon or a site; dropping the
    parameters makes that unrepresentable rather than merely unused.
    """


    var nv = dims.get_nv()
    var nequality = dims.get_nequality()
    if nequality == 0:
        return

    comptime EQ_CAP = 6 * cap[D.NEQUALITY]()
    comptime EQJ_CAP = 6 * cap[D.NEQUALITY]() * cap[D.NV]()
    var eq_rows = 6 * nequality

    var eq_K = Scratch[Scalar[DTYPE], EQ_CAP](eq_rows, Scalar[DTYPE](1))
    var eq_bias = Scratch[Scalar[DTYPE], EQ_CAP](eq_rows, Scalar[DTYPE](0))
    var eq_inv_K_imp = Scratch[Scalar[DTYPE], EQ_CAP](eq_rows, Scalar[DTYPE](0))
    var eq_lambda = Scratch[Scalar[DTYPE], EQ_CAP](eq_rows, Scalar[DTYPE](0))
    var eq_J = Scratch[Scalar[DTYPE], EQJ_CAP](eq_rows * nv, Scalar[DTYPE](0))
    var eq_MinvJ = Scratch[Scalar[DTYPE], EQJ_CAP](eq_rows * nv, Scalar[DTYPE](0))

    var num_eq_rows = build_weld_equality_rows[DTYPE, V_SIZE](
        env, dims, qpos, qvel, xpos, xquat, subtree_com, joints, bodies, mmeta,
        equality, body_invweight0, dof_invweight0, cdof, m_inv,
        eq_K, eq_bias, eq_inv_K_imp, eq_J, eq_MinvJ,
    )

    if num_eq_rows == 0:
        return

    # Bilateral PGS iterations (no clamping)
    for _ in range(NUM_ITERATIONS):
        var max_delta: Scalar[DTYPE] = 0
        for r in range(num_eq_rows):
            # a_eq = J @ qacc
            var a_eq: Scalar[DTYPE] = 0
            for i in range(nv):
                a_eq += eq_J[r * nv + i] * rebind[Scalar[DTYPE]](
                    qacc_constrained[env, i]
                )

            var R_eq = Scalar[DTYPE](1.0) / eq_inv_K_imp[r] - eq_K[r]
            var residual = a_eq + eq_bias[r] + R_eq * eq_lambda[r]
            var delta = -residual * eq_inv_K_imp[r]
            var old_lambda = eq_lambda[r]
            eq_lambda[r] = eq_lambda[r] + delta
            # Bilateral: no clamping (force can push or pull)
            var actual = eq_lambda[r] - old_lambda
            var abs_d = abs(actual)
            if abs_d > max_delta:
                max_delta = abs_d
            # qacc += MinvJ * delta
            for i in range(nv):
                qacc_constrained[env, i] = (
                    rebind[Scalar[DTYPE]](qacc_constrained[env, i])
                    + eq_MinvJ[r * nv + i] * actual
                )

        if max_delta < Scalar[DTYPE](1e-4):
            break


# =============================================================================
# Fixed tendons (port of build_and_solve_tendon_gpu)
# =============================================================================


@always_inline
def _tendon_env[
    DTYPE: DType,
    BATCH: Int,
    NUM_ITERATIONS: Int,
    D: DimsLike,
    L_QPOS: TensorLayout,
    L_QVEL: TensorLayout,
    L_JOINTS: TensorLayout,
    L_MMETA: TensorLayout,
    L_TENDONS: TensorLayout,
    L_SITES: TensorLayout,
    L_GEOMS: TensorLayout,
    L_BODIES: TensorLayout,
    L_SUBTREE_COM: TensorLayout,
    L_CDOF: TensorLayout,
    L_XQUAT: TensorLayout,
    L_M_INV: TensorLayout,
](
    env: Int,
    dims: D,
    qpos: TileTensor[DTYPE, L_QPOS, MutAnyOrigin],
    qvel: TileTensor[DTYPE, L_QVEL, MutAnyOrigin],
    joints: TileTensor[DTYPE, L_JOINTS, MutAnyOrigin],
    mmeta: TileTensor[DTYPE, L_MMETA, MutAnyOrigin],
    tendons: TileTensor[DTYPE, L_TENDONS, MutAnyOrigin],
    sites: TileTensor[DTYPE, L_SITES, MutAnyOrigin],
    geoms: TileTensor[DTYPE, L_GEOMS, MutAnyOrigin],
    bodies: TileTensor[DTYPE, L_BODIES, MutAnyOrigin],
    subtree_com: TileTensor[DTYPE, L_SUBTREE_COM, MutAnyOrigin],
    cdof: TileTensor[DTYPE, L_CDOF, MutAnyOrigin],
    xpos: TileTensor[DTYPE, L_SUBTREE_COM, MutAnyOrigin],
    xquat: TileTensor[DTYPE, L_XQUAT, MutAnyOrigin],
    m_inv: TileTensor[DTYPE, L_M_INV, MutAnyOrigin],
    qacc_constrained: TileTensor[DTYPE, L_QVEL, MutAnyOrigin],
):
    """Build and solve `<equality><tendon>` rows for one env, as a POST-PASS.

    The constraint is `ten_length - length_ref == 0`, bilateral. A FIXED
    tendon's length is `Σ coef_i * qpos[qposadr_i]`; a SPATIAL one's is the
    site polyline, from `spatial_tendon_length_jac`.

    ⚠ THE SPATIAL BRANCH DID NOT EXIST UNTIL 2026-08-12, and four comments
    across the solvers said this pass covered spatial tendons. It could not:
    it read `num_joints`, which is 0 for a spatial tendon, so it built a row
    with a ZERO Jacobian and applied `qacc += M^-1 J^T dlambda == 0`. A
    spatial `<equality><tendon>` was silently unconstrained. See
    `constraints/tendon_limit.build_tendon_equality_rows` for the measurement.

    ⚠ THIS IS THE WRONG PLACE FOR THESE ROWS and the Newton paths no longer
    call it — a post-pass computes the contact force as if the coupling were
    absent, which cost a standing quadruped two thirds of its ground reaction
    force (defect 29a's finding, one constraint type earlier). It remains only
    for the CG, island-PGS and PGS contact solvers, which have no row list to
    append to. `body_invweight0`/`dof_invweight0` are gone from the signature
    with the Newton call sites that carried them; the row's diagApprox is
    `TENDON_IDX_INVWEIGHT0`.
    """

    var nq = dims.get_nq()
    var nv = dims.get_nv()
    var nbody = dims.get_nbody()
    var njoint = dims.get_njoint()
    var ntendon = dims.get_ntendon()
    var nsite = dims.get_nsite()
    comptime if not may_exist[D.NTENDON]():
        return

    # Read number of tendons from model metadata
    var nten = Int(
        rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_NTENDON])
    )
    if nten == 0:
        return
    if nten > ntendon:
        nten = ntendon

    # One bilateral row per tendon
    comptime TEN_CAP = cap[D.NTENDON]()
    comptime TENJ_CAP = cap[D.NTENDON]() * cap[D.NV]()

    var ten_K = Scratch[Scalar[DTYPE], TEN_CAP](ntendon, Scalar[DTYPE](1))
    var ten_bias = Scratch[Scalar[DTYPE], TEN_CAP](ntendon, Scalar[DTYPE](0))
    var ten_inv_K_imp = Scratch[Scalar[DTYPE], TEN_CAP](ntendon, Scalar[DTYPE](0))
    var ten_lambda = Scratch[Scalar[DTYPE], TEN_CAP](ntendon, Scalar[DTYPE](0))
    var ten_J = Scratch[Scalar[DTYPE], TENJ_CAP](ntendon * nv, Scalar[DTYPE](0))
    var ten_MinvJ = Scratch[Scalar[DTYPE], TENJ_CAP](ntendon * nv, Scalar[DTYPE](0))

    var num_ten_rows = 0

    for t_i in range(nten):
        if num_ten_rows >= ntendon:
            break

        # A <tendon> DECLARATION is not a constraint. This pass imposes a
        # bilateral `ten_length == LENGTH_REF`, which only <equality><tendon>
        # asks for; a plain <fixed>/<spatial> is just a length definition used
        # by transmissions, springs and limits.
        #
        # This guard became load-bearing on 2026-07-31, when `fields_build`
        # stopped hardcoding `ntendon = 0`. humanoid and humanoid_standup each
        # declare two <fixed> hip-knee tendons that MuJoCo constrains in no
        # way; without it, populating the count would have welded them.
        if (
            Int(rebind[Scalar[DTYPE]](tendons[t_i, TENDON_IDX_IS_EQUALITY]))
            == 0
        ):
            continue

        # ⚠ NO `SKIP_FIXED` GUARD ANY MORE. Every Newton path used to pass
        # `SKIP_FIXED=True` and this guard then let SPATIAL tendons through
        # into the fixed-only code below. The Newton paths now build rows for
        # both kinds and do not call this function at all, so a caller
        # reaching here owns every equality tendon — adding a kind test back
        # would drop one on the floor exactly as before.
        var length_ref = rebind[Scalar[DTYPE]](
            tendons[t_i, TENDON_IDX_LENGTH_REF]
        )

        # Compute tendon length and velocity, build trivial Jacobian.
        #
        # ⚠ THE SOLVER HELD ITS OWN COPY OF THE 4-WRAP CAP: four unrolled reads
        # into a pair of `Array[..., 4]` locals. Widening the parser and
        # the record alone would have left a tendon with more than four wraps
        # silently short HERE instead — the same defect one layer down.
        #
        # ⚠ BUT WIDENING THOSE LOCALS TO 16 CORRUPTED THE SOLVE. This kernel is
        # Metal-compiled, and two 16-wide per-thread arrays moved Part A's
        # golden by 33% while the RECORD at 16 was provably fine — bisected as:
        #
        #     record 16, locals 16  ->  FAIL  -664336.715
        #     record 16, locals  4  ->  PASS
        #     record  4, locals  4  ->  PASS
        #
        # It is the local-memory hazard this codebase has hit before (the RK4
        # elliptic OOM, the two-non-owning-DeviceBuffer miscompile), and it
        # shows up as WRONG NUMBERS rather than a crash.
        #
        # So the wraps are read INLINE and never buffered, which is what
        # `tendon_limit.mojo` and `invweight.mojo` already do. The kernel now
        # holds less local state than before this change, and the cap it obeys
        # is the record's rather than one of its own.
        var ten_length: Scalar[DTYPE] = 0
        var r = num_ten_rows

        for i in range(nv):
            ten_J[r * nv + i] = Scalar[DTYPE](0)

        if (
            Int(rebind[Scalar[DTYPE]](tendons[t_i, TENDON_IDX_KIND]))
            == TENDON_KIND_SPATIAL
        ):
            # The site polyline and its dense moment arm. `sp_J` is the one
            # per-thread buffer this branch costs; see the Metal local-memory
            # warning above before adding a second.
            var sp_J = Scratch[Scalar[DTYPE], cap[D.CAP_NV]()](
                nv, fill=Scalar[DTYPE](0)
            )
            ten_length = spatial_tendon_length_jac[
                DTYPE, cap[D.CAP_NV](), BATCH
            ](
                env, t_i, dims, tendons, sites, geoms, bodies, joints, mmeta,
                subtree_com,
                cdof, xpos, xquat, sp_J,
            )
            for i in range(nv):
                ten_J[r * nv + i] = sp_J[i]
        else:
            var num_joints = Int(
                rebind[Scalar[DTYPE]](tendons[t_i, TENDON_IDX_NUM_JOINTS])
            )
            for ji in range(TENDON_MAX_WRAPS):
                if ji >= num_joints:
                    break
                var jnt_idx = Int(
                    rebind[Scalar[DTYPE]](
                        tendons[t_i, TENDON_IDX_JOINT_0 + ji]
                    )
                )
                if jnt_idx < 0 or jnt_idx >= njoint:
                    continue
                # Read joint's qpos_adr and dof_adr from the joint records
                var qpos_adr = Int(
                    rebind[Scalar[DTYPE]](joints[jnt_idx, JOINT_IDX_QPOS_ADR])
                )
                var dof_adr = Int(
                    rebind[Scalar[DTYPE]](joints[jnt_idx, JOINT_IDX_DOF_ADR])
                )
                var c = rebind[Scalar[DTYPE]](
                    tendons[t_i, TENDON_IDX_COEF_0 + ji]
                )
                ten_length += c * rebind[Scalar[DTYPE]](
                    qpos[env, qpos_adr]
                )
                # ⚠ ACCUMULATE. This was a bare `=`, so a fixed tendon naming
                # the same joint twice kept only the last coefficient instead
                # of their sum. No model in the tree does that, and
                # `build_tendon_equality_rows` already accumulated.
                ten_J[r * nv + dof_adr] = ten_J[r * nv + dof_adr] + c

        # Off the ASSEMBLED row, so both kinds share one expression — and
        # identical to the old per-wrap accumulation for a fixed tendon.
        var ten_vel: Scalar[DTYPE] = 0
        for i in range(nv):
            ten_vel += ten_J[r * nv + i] * rebind[Scalar[DTYPE]](
                qvel[env, i]
            )

        # Tendon position error (bilateral)
        var pos_err = ten_length - length_ref

        # Compute K = J @ M_inv @ J^T and MinvJ
        var k: Scalar[DTYPE] = 0
        for i in range(nv):
            var mi_j_sum: Scalar[DTYPE] = 0
            for j_idx in range(nv):
                mi_j_sum += (
                    rebind[Scalar[DTYPE]](
                        m_inv[env, i * nv + j_idx]
                    )
                    * ten_J[r * nv + j_idx]
                )
            ten_MinvJ[r * nv + i] = mi_j_sum
            k += ten_J[r * nv + i] * mi_j_sum

        if k < Scalar[DTYPE](1e-10):
            k = Scalar[DTYPE](1e-10)
        ten_K[r] = k

        # Read solref/solimp
        var sr_tc = rebind[Scalar[DTYPE]](tendons[t_i, TENDON_IDX_SOLREF_0])
        var sr_dr = rebind[Scalar[DTYPE]](tendons[t_i, TENDON_IDX_SOLREF_1])
        var si_dmin = rebind[Scalar[DTYPE]](
            tendons[t_i, TENDON_IDX_SOLIMP_0]
        )
        var si_dmax = rebind[Scalar[DTYPE]](
            tendons[t_i, TENDON_IDX_SOLIMP_1]
        )
        var si_width = rebind[Scalar[DTYPE]](
            tendons[t_i, TENDON_IDX_SOLIMP_2]
        )
        var si_midpoint = rebind[Scalar[DTYPE]](
            tendons[t_i, TENDON_IDX_SOLIMP_3]
        )
        var si_power = rebind[Scalar[DTYPE]](
            tendons[t_i, TENDON_IDX_SOLIMP_4]
        )
        if si_width < Scalar[DTYPE](1e-6):
            si_width = Scalar[DTYPE](1e-6)
        # Clamp BOTH ends to [mjMINIMP, mjMAXIMP] as MuJoCo does before
        # interpolating (engine_core_constraint.c:1284-1287).
        comptime MJE_MINIMP = Scalar[DTYPE](0.0001)
        comptime MJE_MAXIMP = Scalar[DTYPE](0.9999)
        if si_dmin < MJE_MINIMP:
            si_dmin = MJE_MINIMP
        elif si_dmin > MJE_MAXIMP:
            si_dmin = MJE_MAXIMP
        if si_dmax < MJE_MINIMP:
            si_dmax = MJE_MINIMP
        elif si_dmax > MJE_MAXIMP:
            si_dmax = MJE_MAXIMP
        if si_power < Scalar[DTYPE](1):
            si_power = Scalar[DTYPE](1)
        # MuJoCo: K = 1/(dmax² * timeconst² * dampratio²)
        # solref -> (K, B), including MuJoCo's DIRECT form for a NEGATIVE
        # solref. See `constraints/constraint_data.solref_spring_damper` — the
        # formula lived in twelve copy-pasted sites until 2026-08-03.
        var (t_K_spring, t_B_damp) = solref_spring_damper[DTYPE](
            sr_tc, sr_dr, si_dmax,
            rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_TIMESTEP]),
        )

        # Impedance: MuJoCo piecewise power formula on |pos_err|
        var penetration = abs(pos_err)
        var imp: Scalar[DTYPE]
        if si_dmin == si_dmax or si_width <= Scalar[DTYPE](0):
            imp = Scalar[DTYPE](0.5) * (si_dmin + si_dmax)
        else:
            var x = penetration / si_width
            var y: Scalar[DTYPE]
            if x <= Scalar[DTYPE](0):
                y = Scalar[DTYPE](0)
            elif x >= Scalar[DTYPE](1):
                y = Scalar[DTYPE](1)
            elif si_power == Scalar[DTYPE](1):
                y = x
            elif x <= si_midpoint:
                var a = Scalar[DTYPE](1) / pow(
                    si_midpoint, si_power - Scalar[DTYPE](1)
                )
                y = a * pow(x, si_power)
            else:
                var b = Scalar[DTYPE](1) / pow(
                    Scalar[DTYPE](1) - si_midpoint, si_power - Scalar[DTYPE](1)
                )
                y = Scalar[DTYPE](1) - b * pow(Scalar[DTYPE](1) - x, si_power)
            imp = si_dmin + y * (si_dmax - si_dmin)
        if imp < Scalar[DTYPE](1e-6):
            imp = Scalar[DTYPE](1e-6)

        # Bilateral equality constraint bias (MuJoCo formula):
        #   aref = -B*vel - K*imp*pos  (pos = pos_err, signed)
        #   bias = -aref = B*vel + K*imp*pos
        var bias = t_B_damp * ten_vel + t_K_spring * imp * pos_err
        ten_bias[r] = bias

        # R = (1-imp)/imp * diagApprox.
        #
        # ⚠ diagApprox for a TENDON equality is `tendon_invweight0[id]` — ONE
        # number, the tendon's own J M^-1 J^T at qpos0 (engine_core_constraint
        # .c:1091, `m->tendon_invweight0[m->eq_obj1id[id]]`). It is NOT the sum
        # of `dof_invweight0` over the tendon's joints, which is what this used
        # to compute: that is the mjEQ_JOINT branch's rule (:1090), applied to
        # the wrong constraint type. The two disagree by more than a constant —
        # a fixed tendon's coefficients and the joints' cross-coupling both
        # enter the real quantity and neither enters the sum.
        #
        # Silent, as usual: R only sets the row's COMPLIANCE, so the constraint
        # still held approximately and nothing errored. quadruped's four
        # coupling tendons are the first equality tendons any model here builds
        # through the parser, and they put ~7% on qacc.
        var diag_ten = rebind[Scalar[DTYPE]](
            tendons[t_i, TENDON_IDX_INVWEIGHT0]
        )
        if diag_ten < Scalar[DTYPE](1e-10):
            diag_ten = k  # Fallback to exact K
        var R_ten = (Scalar[DTYPE](1.0) - imp) / imp * diag_ten
        ten_inv_K_imp[r] = Scalar[DTYPE](1.0) / (k + R_ten)

        num_ten_rows += 1

    if num_ten_rows == 0:
        return

    # Bilateral PGS iterations (no clamping — bilateral constraint)
    for _ in range(NUM_ITERATIONS):
        var max_delta: Scalar[DTYPE] = 0
        for r in range(num_ten_rows):
            # a_ten = J @ qacc
            var a_ten: Scalar[DTYPE] = 0
            for i in range(nv):
                a_ten += ten_J[r * nv + i] * rebind[Scalar[DTYPE]](
                    qacc_constrained[env, i]
                )

            var R_ten = Scalar[DTYPE](1.0) / ten_inv_K_imp[r] - ten_K[r]
            var residual = a_ten + ten_bias[r] + R_ten * ten_lambda[r]
            var delta = -residual * ten_inv_K_imp[r]
            var old_lambda = ten_lambda[r]
            ten_lambda[r] = ten_lambda[r] + delta
            # Bilateral: no clamping
            var actual = ten_lambda[r] - old_lambda
            var abs_d = abs(actual)
            if abs_d > max_delta:
                max_delta = abs_d
            # qacc += MinvJ * delta
            for i in range(nv):
                qacc_constrained[env, i] = (
                    rebind[Scalar[DTYPE]](qacc_constrained[env, i])
                    + ten_MinvJ[r * nv + i] * actual
                )

        if max_delta < Scalar[DTYPE](1e-4):
            break
