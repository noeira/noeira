"""`mj_jac` / `mj_jacSite` — the full 3-by-nv translational and rotational
Jacobians of a world point attached to a body.

WHY THIS EXISTS SEPARATELY FROM THE CONSTRAINT ROWS. The same arithmetic is
already in the tree twice, but both copies project onto a direction and return
a SCALAR row: `constraints/contact_solve._contact_jacobian_row` (translation,
dotted with the contact normal) and `_angular_jacobian_row` (rotation, dotted
with a torsion axis). Inverse kinematics needs the UNPROJECTED 6 x nv, because
the damped-least-squares update solves against the whole matrix rather than one
direction of it. This module is that same arithmetic with the dot product
removed — not a re-derivation.

TRANSCRIBED FROM `engine_core_util.c:mj_jac`. That function is BYTE-IDENTICAL
in the 3.6.0 and 3.11.0 reference trees, and the runtime here is 3.10.0, so the
usual version-drift caveat (`references/` never matches the runtime) does not
bite for once.

Two details differ from the contact-row copies. Only the second changes a
number; the first is recorded so nobody "fixes" it back:

1. THE OFFSET IS HOISTED. MuJoCo computes
   `offset = point - subtree_com[body_rootid[body]]` ONCE before the loop;
   `_contact_jacobian_row` recomputes it inside the loop from each joint's
   body's rootid. ⚠ THESE ARE THE SAME VALUE, ALWAYS — `body_rootid` is by
   construction the child-of-world on a body's chain, so every ancestor of the
   target shares the target's root, and only ancestors are ever visited. This
   was checked rather than assumed: over all of quadruped's bodies, the number
   of ancestors carrying a different rootid is 0. Hoisting is MuJoCo's
   formulation and one fewer lookup per DOF, not a correctness fix.

2. ⚠ THE BODY IS REMAPPED THROUGH `body_weldid` AND MAY TERMINATE THE WHOLE
   THING. A body with no joints of its own is welded to its nearest jointed
   ancestor; if that resolves to the world, MuJoCo returns a ZERO Jacobian
   rather than walking anything. A site on a static decoration is not an
   error, it is a site that cannot be moved. quadruped fetch has exactly one
   such site, so the branch is exercised — though ⚠ NOT DISCRIMINATINGLY:
   deleting the early return gives zero anyway, because the ancestor walk then
   finds no joints. It is transcription faithfulness, not a guard whose
   removal a test would catch.

LAYOUT. MuJoCo writes `jacp[i + k*nv]` — row-major 3 x nv, row `k` is the world
axis and column `i` the DOF. `jacp[k * nv + i]` here, same thing. ⚠ Reading it
as nv x 3 transposes silently and still has the right norm, so a gate that only
checks magnitudes will not see it.
"""

from std.collections import Array
from layout import TileTensor, TensorLayout
from ..fields.scratch import Scratch

from ..joint_types import JNT_FREE, JNT_BALL
from ..gpu.constants import (
    MODEL_BODY_SIZE,
    MODEL_JOINT_SIZE,
    MODEL_META_SIZE,
    MODEL_META_IDX_NJOINT,
    MODEL_SITE_SIZE,
    BODY_IDX_PARENT,
    BODY_IDX_ROOTID,
    BODY_IDX_WELDID,
    JOINT_IDX_TYPE,
    JOINT_IDX_BODY_ID,
    JOINT_IDX_DOF_ADR,
    SITE_IDX_BODY,
)


@always_inline
def _is_self_or_ancestor[
    DTYPE: DType,
    L_BODIES: TensorLayout](
    bodies: TileTensor[DTYPE, L_BODIES, MutAnyOrigin],
    body: Int,
    candidate: Int,
) -> Bool:
    """Is `candidate` `body` itself, or one of its ancestors?

    MuJoCo walks `dof_parentid` instead, which we do not store. The two select
    the SAME DOF SET: the `dof_parentid` chain from a body's last DOF visits
    that body's remaining DOFs and then jumps to the parent body's last DOF, so
    the union over the chain is exactly "every DOF of every self-or-ancestor
    body".
    """
    if candidate <= 0:
        # The world carries no joints (MuJoCo rejects them), so a joint can
        # never name body 0 and this is unreachable for real models. Answering
        # False keeps a malformed model from walking off the parent chain.
        return False
    var cur = body
    while cur > 0:
        if cur == candidate:
            return True
        cur = Int(rebind[Scalar[DTYPE]](bodies[cur, BODY_IDX_PARENT]))
    return False


@always_inline
def jac_point[
    DTYPE: DType,
    V_CAP: Int,
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
    body: Int,
    point_x: Scalar[DTYPE],
    point_y: Scalar[DTYPE],
    point_z: Scalar[DTYPE],
    mut jacp: Scratch[Scalar[DTYPE], 3 * V_CAP],
    mut jacr: Scratch[Scalar[DTYPE], 3 * V_CAP],
    nv: Int,
):
    """`mj_jac(m, d, jacp, jacr, point, body)` — both blocks, always.

    MuJoCo lets either output be NULL and skips its work; we always fill both,
    because every caller here (IK) wants both and the saving is a handful of
    multiply-adds per DOF.
    """
    for i in range(3 * nv):
        jacp[i] = 0
        jacr[i] = 0

    # ⚠ ROOT OF THE ORIGINAL BODY, BEFORE THE WELD REMAP BELOW — that is the
    # order in `mj_jac`. Welding never crosses a tree, so the two orders agree;
    # transcribed in MuJoCo's order anyway so a future reader diffing against
    # the C file finds the same statement sequence.
    var root = Int(rebind[Scalar[DTYPE]](bodies[body, BODY_IDX_ROOTID]))
    var off_x = point_x - rebind[Scalar[DTYPE]](subtree_com[env, root * 3 + 0])
    var off_y = point_y - rebind[Scalar[DTYPE]](subtree_com[env, root * 3 + 1])
    var off_z = point_z - rebind[Scalar[DTYPE]](subtree_com[env, root * 3 + 2])

    # Skip fixed bodies; a body welded to the world has no DOFs at all.
    var wbody = Int(rebind[Scalar[DTYPE]](bodies[body, BODY_IDX_WELDID]))
    if wbody == 0:
        return

    var num_joints = Int(rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_NJOINT]))

    for j_idx in range(num_joints):
        var joint_body = Int(
            rebind[Scalar[DTYPE]](joints[j_idx, JOINT_IDX_BODY_ID])
        )
        if not _is_self_or_ancestor[DTYPE](bodies, wbody, joint_body):
            continue

        var jnt_type = Int(
            rebind[Scalar[DTYPE]](joints[j_idx, JOINT_IDX_TYPE])
        )
        var dof_adr = Int(
            rebind[Scalar[DTYPE]](joints[j_idx, JOINT_IDX_DOF_ADR])
        )
        var num_dof = 1
        if jnt_type == JNT_FREE:
            num_dof = 6
        elif jnt_type == JNT_BALL:
            num_dof = 3

        for d in range(num_dof):
            var i = dof_adr + d

            # cdof is [angular(3), linear(3)] per DOF, matching MuJoCo's
            # `cdof+6*i` with rotation first.
            var ang_x = rebind[Scalar[DTYPE]](cdof[env, i * 6 + 0])
            var ang_y = rebind[Scalar[DTYPE]](cdof[env, i * 6 + 1])
            var ang_z = rebind[Scalar[DTYPE]](cdof[env, i * 6 + 2])

            jacr[0 * nv + i] = ang_x
            jacr[1 * nv + i] = ang_y
            jacr[2 * nv + i] = ang_z

            # jacp = cdof_linear + cdof_angular x offset
            jacp[0 * nv + i] = (
                rebind[Scalar[DTYPE]](cdof[env, i * 6 + 3])
                + ang_y * off_z
                - ang_z * off_y
            )
            jacp[1 * nv + i] = (
                rebind[Scalar[DTYPE]](cdof[env, i * 6 + 4])
                + ang_z * off_x
                - ang_x * off_z
            )
            jacp[2 * nv + i] = (
                rebind[Scalar[DTYPE]](cdof[env, i * 6 + 5])
                + ang_x * off_y
                - ang_y * off_x
            )


@always_inline
def jac_site[
    DTYPE: DType,
    V_CAP: Int,
    L_SUBTREE_COM: TensorLayout,
    L_JOINTS: TensorLayout,
    L_BODIES: TensorLayout,
    L_MMETA: TensorLayout,
    L_CDOF: TensorLayout,
    L_SITES: TensorLayout,
    L_SITE_XPOS: TensorLayout,
](
    env: Int,
    subtree_com: TileTensor[DTYPE, L_SUBTREE_COM, MutAnyOrigin],
    joints: TileTensor[DTYPE, L_JOINTS, MutAnyOrigin],
    bodies: TileTensor[DTYPE, L_BODIES, MutAnyOrigin],
    mmeta: TileTensor[DTYPE, L_MMETA, MutAnyOrigin],
    cdof: TileTensor[DTYPE, L_CDOF, MutAnyOrigin],
    sites: TileTensor[DTYPE, L_SITES, MutAnyOrigin],
    site_xpos: TileTensor[DTYPE, L_SITE_XPOS, MutAnyOrigin],
    site: Int,
    mut jacp: Scratch[Scalar[DTYPE], 3 * V_CAP],
    mut jacr: Scratch[Scalar[DTYPE], 3 * V_CAP],
    nv: Int,
):
    """`mj_jacSite` — `jac_point` at the site's world position and body."""
    jac_point[DTYPE, V_CAP](
        env,
        subtree_com,
        joints,
        bodies,
        mmeta,
        cdof,
        Int(rebind[Scalar[DTYPE]](sites[site, SITE_IDX_BODY])),
        rebind[Scalar[DTYPE]](site_xpos[env, site * 3 + 0]),
        rebind[Scalar[DTYPE]](site_xpos[env, site * 3 + 1]),
        rebind[Scalar[DTYPE]](site_xpos[env, site * 3 + 2]),
        jacp,
        jacr,
        nv,
    )
