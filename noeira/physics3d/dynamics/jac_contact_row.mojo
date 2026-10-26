"""One row of the contact Jacobian — moved out of `constraints/` (phase 2.0).

⚠ THIS LIVES IN `dynamics/` FOR A STRUCTURAL REASON, not a taxonomic one.
`dynamics/tendon.mojo` needs it, and while it sat in
`constraints/contact_solve.mojo` that need was the single import making
`dynamics` depend on `constraints`. Together with `constraints -> solver` (the
`qcqp`/`elliptic_layout` leaves, moved in the same step) it closed a cycle that
put {constraints, dynamics, solver} into one 22.5k-line strongly-connected
component — 136 of the sweep's 238 dim-carrying declarations with no
intermediate green state between them. That is what made §5.4's
package-at-a-time gating protocol unexecutable. See docs §11.2/§11.3.

The body is VERBATIM from `contact_solve.mojo` — a relocation, not a rewrite.
Its only dependencies were `gpu.constants` offsets and the joint-type
constants, which is why it moved cleanly.

⚠ ITS SIBLING `_angular_jacobian_row` DELIBERATELY DID NOT MOVE. Nothing in
`dynamics` imports it, so it creates no cycle, and moving code that is not in
the way would widen this step past what its gate covers.
"""

from layout import TileTensor, TensorLayout
from ..fields.scratch import Scratch

from ..joint_types import JNT_FREE, JNT_BALL
from ..gpu.constants import (
    MODEL_BODY_SIZE,
    MODEL_JOINT_SIZE,
    MODEL_META_SIZE,
    MODEL_META_IDX_NJOINT,
    BODY_IDX_PARENT,
    BODY_IDX_ROOTID,
    JOINT_IDX_TYPE,
    JOINT_IDX_BODY_ID,
    JOINT_IDX_DOF_ADR,
)



# Ancestor chains of the two contact bodies, walked ONCE per row.
#
# ⚠ The three Jacobian-row builders (`_contact_jacobian_row` here, the two
# `_angular_jacobian_row`s in `constraints/`) used to decide "does joint j
# move body a" by walking a's parent chain for EVERY joint: `njoint × depth`
# dependent `bodies[.., BODY_IDX_PARENT]` loads per row — ~650 on dog for a
# 50-joint model, three rows per contact. `mj_jac` walks the chain once. The
# chain is recorded in a small fixed array and each joint's body is tested
# against it (a handful of register compares). Same per-dof arithmetic in the
# same order, so the rows are bit-identical (PERFORMANCE.md §13.23).
#
# `CHAIN_CAP` bounds the depth this fast path handles; a deeper chain returns
# -1 and `_affects` falls back to the walk, so nothing is ever wrong, only
# slower.
comptime CHAIN_CAP: Int = 32


@always_inline
def _body_chain[
    DTYPE: DType,
    L_BODIES: TensorLayout,
](
    bodies: TileTensor[DTYPE, L_BODIES, MutAnyOrigin],
    body: Int,
    mut chain: Scratch[Int, CHAIN_CAP],
) -> Int:
    """`chain[0..n)` = `body`, its parent, … up to the root body; returns `n`,
    or -1 if the chain does not fit (`body <= 0` gives 0)."""
    var n = 0
    var cur = body
    while cur > 0:
        if n >= CHAIN_CAP:
            return -1
        chain[n] = cur
        n += 1
        cur = Int(rebind[Scalar[DTYPE]](bodies[cur, BODY_IDX_PARENT]))
    return n


@always_inline
def _affects[
    DTYPE: DType,
    L_BODIES: TensorLayout,
](
    bodies: TileTensor[DTYPE, L_BODIES, MutAnyOrigin],
    chain: Scratch[Int, CHAIN_CAP],
    n: Int,
    body: Int,
    joint_body: Int,
) -> Bool:
    """Does the joint on `joint_body` move `body`? `body` itself or one of its
    ancestors. Reads the recorded chain; walks only when it overflowed."""
    if n >= 0:
        for a in range(n):
            if chain[a] == joint_body:
                return True
        return False
    if body <= 0:
        return False
    if body == joint_body:
        return True
    var cur = body
    while cur > 0:
        var par = Int(rebind[Scalar[DTYPE]](bodies[cur, BODY_IDX_PARENT]))
        if par == joint_body:
            return True
        cur = par
    return False


@always_inline
def _joint_row_add[
    DTYPE: DType,
    V_CAP: Int,
    L_SUBTREE_COM: TensorLayout,
    L_JOINTS: TensorLayout,
    L_BODIES: TensorLayout,
    L_CDOF: TensorLayout,
](
    env: Int,
    j_idx: Int,
    affects_a: Bool,
    affects_b: Bool,
    subtree_com: TileTensor[DTYPE, L_SUBTREE_COM, MutAnyOrigin],
    joints: TileTensor[DTYPE, L_JOINTS, MutAnyOrigin],
    bodies: TileTensor[DTYPE, L_BODIES, MutAnyOrigin],
    cdof: TileTensor[DTYPE, L_CDOF, MutAnyOrigin],
    contact_pos_x: Scalar[DTYPE],
    contact_pos_y: Scalar[DTYPE],
    contact_pos_z: Scalar[DTYPE],
    dir_x: Scalar[DTYPE],
    dir_y: Scalar[DTYPE],
    dir_z: Scalar[DTYPE],
    mut J_row: Scratch[Scalar[DTYPE], V_CAP],
):
    """One joint's dofs into the row: `+val` for body a, `-val` for body b —
    the per-joint body of `_contact_jacobian_row`, shared by its two joint
    walks so the arithmetic is written once."""
    var jnt_type = Int(rebind[Scalar[DTYPE]](joints[j_idx, JOINT_IDX_TYPE]))
    var joint_body = Int(
        rebind[Scalar[DTYPE]](joints[j_idx, JOINT_IDX_BODY_ID])
    )
    var dof_adr = Int(rebind[Scalar[DTYPE]](joints[j_idx, JOINT_IDX_DOF_ADR]))

    var num_dof = 1
    if jnt_type == JNT_FREE:
        num_dof = 6
    elif jnt_type == JNT_BALL:
        num_dof = 3

    # Reference = subtree_com[rootid] (must match cdof computation)
    var jb_rootid = Int(
        rebind[Scalar[DTYPE]](bodies[joint_body, BODY_IDX_ROOTID])
    )
    var b_x = rebind[Scalar[DTYPE]](
        subtree_com[env, jb_rootid * 3 + 0]
    )
    var b_y = rebind[Scalar[DTYPE]](
        subtree_com[env, jb_rootid * 3 + 1]
    )
    var b_z = rebind[Scalar[DTYPE]](
        subtree_com[env, jb_rootid * 3 + 2]
    )

    var rx = contact_pos_x - b_x
    var ry = contact_pos_y - b_y
    var rz = contact_pos_z - b_z

    for d in range(num_dof):
        var dof_idx = dof_adr + d

        var ang_x = cdof[env, dof_idx * 6 + 0]
        var ang_y = cdof[env, dof_idx * 6 + 1]
        var ang_z = cdof[env, dof_idx * 6 + 2]
        var lin_x = cdof[env, dof_idx * 6 + 3]
        var lin_y = cdof[env, dof_idx * 6 + 4]
        var lin_z = cdof[env, dof_idx * 6 + 5]

        # J_trans = cdof_lin + cdof_ang x r
        var cross_x = ang_y * rz - ang_z * ry
        var cross_y = ang_z * rx - ang_x * rz
        var cross_z = ang_x * ry - ang_y * rx

        var jt_x = lin_x + cross_x
        var jt_y = lin_y + cross_y
        var jt_z = lin_z + cross_z

        var val = jt_x * dir_x + jt_y * dir_y + jt_z * dir_z

        # Body A contributes positively, body B negatively
        if affects_a:
            J_row[dof_idx] += rebind[Scalar[DTYPE]](val)
        if affects_b:
            J_row[dof_idx] -= rebind[Scalar[DTYPE]](val)


@always_inline
def _contact_jacobian_row[
    DTYPE: DType,
    V_CAP: Int,
    L_SUBTREE_COM: TensorLayout,
    L_JOINTS: TensorLayout,
    L_BODIES: TensorLayout,
    L_MMETA: TensorLayout,
    L_CDOF: TensorLayout,
    JM_CAP: Int = 1,
](
    env: Int,
    subtree_com: TileTensor[DTYPE, L_SUBTREE_COM, MutAnyOrigin],
    joints: TileTensor[DTYPE, L_JOINTS, MutAnyOrigin],
    bodies: TileTensor[DTYPE, L_BODIES, MutAnyOrigin],
    mmeta: TileTensor[DTYPE, L_MMETA, MutAnyOrigin],
    cdof: TileTensor[DTYPE, L_CDOF, MutAnyOrigin],
    contact_body_a: Int,
    contact_body_b: Int,
    contact_pos_x: Scalar[DTYPE],
    contact_pos_y: Scalar[DTYPE],
    contact_pos_z: Scalar[DTYPE],
    dir_x: Scalar[DTYPE],
    dir_y: Scalar[DTYPE],
    dir_z: Scalar[DTYPE],
    mut J_row: Scratch[Scalar[DTYPE], V_CAP],
    nv: Int,
    # Body → joint map (`body_jntadr` / `body_jntnum`), built once per solve
    # by the CPU Newton from the joint table and passed down: with it the
    # row visits only the joints on the two contact bodies' chains instead
    # of testing all `njoint` (PERFORMANCE.md §13.25). Defaults = no map,
    # the scanning form — every other caller, the GPU kernels included.
    jnt_adr: Scratch[Int, JM_CAP] = Scratch[Int, JM_CAP](1, fill=0),
    jnt_num: Scratch[Int, JM_CAP] = Scratch[Int, JM_CAP](1, fill=0),
    map_ok: Bool = False,
):
    """One row of the contact Jacobian (verbatim from
    compute_contact_jacobian_row_gpu; the legacy body computed an unused
    xpos offset, dropped here).

    Bilateral: J_row[i] = J_a[i] - J_b[i] for body-body contacts.
    For ground contacts (body_b = 0, worldbody), only body_a contributes.
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
    if map_ok and n_a >= 0 and n_b >= 0:
        # The joints on the two chains, in chain order. A joint on BOTH
        # chains adds `val` in the first pass and subtracts the same `val`
        # in the second — exactly what the scanning walk did in one visit.
        for sa in range(n_a):
            var bb = chain_a[sa]
            var j0 = jnt_adr[bb]
            for jj in range(jnt_num[bb]):
                _joint_row_add[DTYPE, V_CAP](
                    env, j0 + jj, True, False, subtree_com, joints, bodies,
                    cdof, contact_pos_x, contact_pos_y, contact_pos_z,
                    dir_x, dir_y, dir_z, J_row,
                )
        for sb in range(n_b):
            var bb = chain_b[sb]
            var j0 = jnt_adr[bb]
            for jj in range(jnt_num[bb]):
                _joint_row_add[DTYPE, V_CAP](
                    env, j0 + jj, False, True, subtree_com, joints, bodies,
                    cdof, contact_pos_x, contact_pos_y, contact_pos_z,
                    dir_x, dir_y, dir_z, J_row,
                )
        return
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
        _joint_row_add[DTYPE, V_CAP](
            env, j_idx, affects_a, affects_b, subtree_com, joints, bodies,
            cdof, contact_pos_x, contact_pos_y, contact_pos_z,
            dir_x, dir_y, dir_z, J_row,
        )
