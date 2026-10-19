"""`cdof` (spatial motion axes per DOF) over per-field tensors (migration P2,
single-source). Per-field port of `cdof_body_gpu`/`compute_cdof_gpu`
(dynamics/jacobian.mojo) — arithmetic verbatim; addressing per-field; the
cdof output lives in an owned `DynamicsScratch.cdof` tensor instead of the
`ws_cdof_offset` region of a caller-provided workspace slab (the first
workspace array to move into stateful scratch).

Operands: qpos, xpos, xquat, subtree_com + body/joint records -> cdof
(7 operands). One formula body for both targets. As with FK, `num_joints`
is the comptime NJOINT (no metadata read)."""

from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from layout import TileTensor, TensorLayout
from noeira.core.tiles import RowMajor, Tile

from ..kinematics.quat_math import (
    gpu_quat_mul,
    gpu_quat_rotate,
    gpu_axis_angle_to_quat,
)
from ..joint_types import JNT_FREE, JNT_SLIDE, JNT_HINGE, JNT_BALL
from ..fields import (
    Data,
    Model,
    DynamicsScratch,
    Dims,
    DimsLike,
    AsStatic,
    DYN2,
    rl2,
)
from ..gpu.constants import (
    MODEL_BODY_SIZE,
    MODEL_JOINT_SIZE,
    BODY_IDX_PARENT,
    BODY_IDX_POS_X,
    BODY_IDX_POS_Y,
    BODY_IDX_POS_Z,
    BODY_IDX_QUAT_X,
    BODY_IDX_QUAT_Y,
    BODY_IDX_QUAT_Z,
    BODY_IDX_QUAT_W,
    BODY_IDX_ROOTID,
    JOINT_IDX_TYPE,
    JOINT_IDX_BODY_ID,
    JOINT_IDX_QPOS_ADR,
    JOINT_IDX_DOF_ADR,
    JOINT_IDX_POS_X,
    JOINT_IDX_POS_Y,
    JOINT_IDX_POS_Z,
    JOINT_IDX_AXIS_X,
    JOINT_IDX_AXIS_Y,
    JOINT_IDX_AXIS_Z,
    JOINT_IDX_QPOS0,
)

comptime CDOF_TPB: Int = 64


@always_inline
def _cdof_body[
    DTYPE: DType,
    D: DimsLike,
    L_QPOS: TensorLayout,
    L_XPOS: TensorLayout,
    L_XQUAT: TensorLayout,
    L_BODIES: TensorLayout,
    L_JOINTS: TensorLayout,
    L_CDOF: TensorLayout,
](
    env: Int,
    body: Int,
    dims: D,
    qpos: TileTensor[DTYPE, L_QPOS, MutAnyOrigin],
    xpos: TileTensor[DTYPE, L_XPOS, MutAnyOrigin],
    xquat: TileTensor[DTYPE, L_XQUAT, MutAnyOrigin],
    subtree_com: TileTensor[DTYPE, L_XPOS, MutAnyOrigin],
    bodies: TileTensor[DTYPE, L_BODIES, MutAnyOrigin],
    joints: TileTensor[DTYPE, L_JOINTS, MutAnyOrigin],
    cdof: TileTensor[DTYPE, L_CDOF, MutAnyOrigin],
):
    """One body's cdof for its DOFs (arithmetic verbatim from
    cdof_body_gpu)."""
    var njoint = dims.get_njoint()
    var parent = Int(rebind[Scalar[DTYPE]](bodies[body, BODY_IDX_PARENT]))

    # Parent world orientation (worldbody=0 has identity)
    var acc_qx = rebind[Scalar[DTYPE]](xquat[env, parent * 4 + 0])
    var acc_qy = rebind[Scalar[DTYPE]](xquat[env, parent * 4 + 1])
    var acc_qz = rebind[Scalar[DTYPE]](xquat[env, parent * 4 + 2])
    var acc_qw = rebind[Scalar[DTYPE]](xquat[env, parent * 4 + 3])

    # acc = parent_quat * body_quat
    var bq_x = rebind[Scalar[DTYPE]](bodies[body, BODY_IDX_QUAT_X])
    var bq_y = rebind[Scalar[DTYPE]](bodies[body, BODY_IDX_QUAT_Y])
    var bq_z = rebind[Scalar[DTYPE]](bodies[body, BODY_IDX_QUAT_Z])
    var bq_w = rebind[Scalar[DTYPE]](bodies[body, BODY_IDX_QUAT_W])
    var pre_q = gpu_quat_mul(
        acc_qx,
        acc_qy,
        acc_qz,
        acc_qw,
        bq_x,
        bq_y,
        bq_z,
        bq_w,
    )
    acc_qx = pre_q[0]
    acc_qy = pre_q[1]
    acc_qz = pre_q[2]
    acc_qw = pre_q[3]

    # Reference point: subtree_com[rootid[body]] (MuJoCo convention)
    var rootid = Int(rebind[Scalar[DTYPE]](bodies[body, BODY_IDX_ROOTID]))
    var ref_x = rebind[Scalar[DTYPE]](subtree_com[env, rootid * 3 + 0])
    var ref_y = rebind[Scalar[DTYPE]](subtree_com[env, rootid * 3 + 1])
    var ref_z = rebind[Scalar[DTYPE]](subtree_com[env, rootid * 3 + 2])

    # xpos_initial: xpos[parent] + R(xquat[parent]) * body_pos
    var body_pos_x = rebind[Scalar[DTYPE]](bodies[body, BODY_IDX_POS_X])
    var body_pos_y = rebind[Scalar[DTYPE]](bodies[body, BODY_IDX_POS_Y])
    var body_pos_z = rebind[Scalar[DTYPE]](bodies[body, BODY_IDX_POS_Z])
    var par_qx = rebind[Scalar[DTYPE]](xquat[env, parent * 4 + 0])
    var par_qy = rebind[Scalar[DTYPE]](xquat[env, parent * 4 + 1])
    var par_qz = rebind[Scalar[DTYPE]](xquat[env, parent * 4 + 2])
    var par_qw = rebind[Scalar[DTYPE]](xquat[env, parent * 4 + 3])
    var bpos_w = gpu_quat_rotate(
        par_qx, par_qy, par_qz, par_qw, body_pos_x, body_pos_y, body_pos_z
    )
    var cx = rebind[Scalar[DTYPE]](xpos[env, parent * 3 + 0]) + bpos_w[0]
    var cy = rebind[Scalar[DTYPE]](xpos[env, parent * 3 + 1]) + bpos_w[1]
    var cz = rebind[Scalar[DTYPE]](xpos[env, parent * 3 + 2]) + bpos_w[2]

    for j in range(njoint):
        var joint_body = Int(
            rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_BODY_ID])
        )
        if joint_body != body:
            continue

        var jnt_type = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_TYPE]))
        var dof_adr = Int(
            rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_DOF_ADR])
        )

        if jnt_type == JNT_HINGE:
            var axis_lx = rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_AXIS_X])
            var axis_ly = rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_AXIS_Y])
            var axis_lz = rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_AXIS_Z])
            var jpos_lx = rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_POS_X])
            var jpos_ly = rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_POS_Y])
            var jpos_lz = rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_POS_Z])

            var a_w = gpu_quat_rotate(
                acc_qx,
                acc_qy,
                acc_qz,
                acc_qw,
                axis_lx,
                axis_ly,
                axis_lz,
            )
            var ax = a_w[0]
            var ay = a_w[1]
            var az = a_w[2]

            var jp = gpu_quat_rotate(
                acc_qx,
                acc_qy,
                acc_qz,
                acc_qw,
                jpos_lx,
                jpos_ly,
                jpos_lz,
            )
            var anc_x = cx + jp[0]
            var anc_y = cy + jp[1]
            var anc_z = cz + jp[2]

            var ox = ref_x - anc_x
            var oy = ref_y - anc_y
            var oz = ref_z - anc_z

            cdof[env, dof_adr * 6 + 0] = ax
            cdof[env, dof_adr * 6 + 1] = ay
            cdof[env, dof_adr * 6 + 2] = az
            cdof[env, dof_adr * 6 + 3] = ay * oz - az * oy
            cdof[env, dof_adr * 6 + 4] = az * ox - ax * oz
            cdof[env, dof_adr * 6 + 5] = ax * oy - ay * ox

            var qpos_adr_val = Int(
                rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_QPOS_ADR])
            )
            var qpos0_val = rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_QPOS0])
            var angle = (
                rebind[Scalar[DTYPE]](qpos[env, qpos_adr_val]) - qpos0_val
            )
            var hinge_q = gpu_axis_angle_to_quat(ax, ay, az, angle)
            var new_q = gpu_quat_mul(
                hinge_q[0],
                hinge_q[1],
                hinge_q[2],
                hinge_q[3],
                acc_qx,
                acc_qy,
                acc_qz,
                acc_qw,
            )
            acc_qx = new_q[0]
            acc_qy = new_q[1]
            acc_qz = new_q[2]
            acc_qw = new_q[3]

            var vec = gpu_quat_rotate(
                acc_qx, acc_qy, acc_qz, acc_qw, jpos_lx, jpos_ly, jpos_lz
            )
            cx = anc_x - vec[0]
            cy = anc_y - vec[1]
            cz = anc_z - vec[2]

        elif jnt_type == JNT_SLIDE:
            var axis_lx = rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_AXIS_X])
            var axis_ly = rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_AXIS_Y])
            var axis_lz = rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_AXIS_Z])

            var a_w = gpu_quat_rotate(
                acc_qx,
                acc_qy,
                acc_qz,
                acc_qw,
                axis_lx,
                axis_ly,
                axis_lz,
            )

            cdof[env, dof_adr * 6 + 3] = a_w[0]
            cdof[env, dof_adr * 6 + 4] = a_w[1]
            cdof[env, dof_adr * 6 + 5] = a_w[2]

            var qpos_adr_val2 = Int(
                rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_QPOS_ADR])
            )
            var qpos0_val2 = rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_QPOS0])
            var disp = (
                rebind[Scalar[DTYPE]](qpos[env, qpos_adr_val2]) - qpos0_val2
            )
            cx += disp * a_w[0]
            cy += disp * a_w[1]
            cz += disp * a_w[2]

        elif jnt_type == JNT_BALL:
            # ⚠⚠ THIS BRANCH DID NOT EXIST, AND `JNT_BALL` WAS NOT EVEN
            # IMPORTED HERE. A ball joint therefore had NO motion subspace:
            # its three `cdof` rows stayed zero, so the mass matrix rows and
            # every Jacobian column built from them were zero too, no force
            # could reach the joint, `qacc` was zero and `qvel` stayed
            # EXACTLY 0.0 forever. The joint is free in the model and frozen
            # in the simulation — measured on cassie, whose achilles rods
            # never rotated while MuJoCo drove them to -0.797 rad/s.
            #
            # `mass_matrix.mojo` and `rne.mojo` both already had their
            # `JNT_BALL` cases; they were consuming a `cdof` nobody wrote.
            #
            # MuJoCo (`engine_core_smooth.c:329`) FALLS THROUGH from FREE into
            # BALL with `skip = 18`, so these three rows and the free joint's
            # rotational three are the SAME construction: axis k is column k
            # of the body's world orientation, and the linear part is
            # `axis x (subtree_com - xanchor)`.
            #
            # ⚠ THE OFFSET IS FROM THE **ANCHOR**, NOT THE BODY ORIGIN. They
            # coincide for a free joint (its anchor IS the origin) and for any
            # ball joint that omits `pos`, which is why the free branch below
            # can use `xpos` and still be right. cassie omits it; a model that
            # does not would be off by the anchor offset.
            var b_jpx = rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_POS_X])
            var b_jpy = rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_POS_Y])
            var b_jpz = rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_POS_Z])
            var b_jp = gpu_quat_rotate(
                acc_qx, acc_qy, acc_qz, acc_qw, b_jpx, b_jpy, b_jpz
            )
            var b_ox = ref_x - (cx + b_jp[0])
            var b_oy = ref_y - (cy + b_jp[1])
            var b_oz = ref_z - (cz + b_jp[2])

            var bbqx = rebind[Scalar[DTYPE]](xquat[env, body * 4 + 0])
            var bbqy = rebind[Scalar[DTYPE]](xquat[env, body * 4 + 1])
            var bbqz = rebind[Scalar[DTYPE]](xquat[env, body * 4 + 2])
            var bbqw = rebind[Scalar[DTYPE]](xquat[env, body * 4 + 3])
            var two = Scalar[DTYPE](2)
            var one = Scalar[DTYPE](1)
            # Columns of the rotation matrix built from the body quaternion.
            var bx0 = one - two * (bbqy * bbqy + bbqz * bbqz)
            var by0 = two * (bbqx * bbqy + bbqw * bbqz)
            var bz0 = two * (bbqx * bbqz - bbqw * bbqy)
            var bx1 = two * (bbqx * bbqy - bbqw * bbqz)
            var by1 = one - two * (bbqx * bbqx + bbqz * bbqz)
            var bz1 = two * (bbqy * bbqz + bbqw * bbqx)
            var bx2 = two * (bbqx * bbqz + bbqw * bbqy)
            var by2 = two * (bbqy * bbqz - bbqw * bbqx)
            var bz2 = one - two * (bbqx * bbqx + bbqy * bbqy)

            cdof[env, (dof_adr + 0) * 6 + 0] = bx0
            cdof[env, (dof_adr + 0) * 6 + 1] = by0
            cdof[env, (dof_adr + 0) * 6 + 2] = bz0
            cdof[env, (dof_adr + 0) * 6 + 3] = by0 * b_oz - bz0 * b_oy
            cdof[env, (dof_adr + 0) * 6 + 4] = bz0 * b_ox - bx0 * b_oz
            cdof[env, (dof_adr + 0) * 6 + 5] = bx0 * b_oy - by0 * b_ox
            cdof[env, (dof_adr + 1) * 6 + 0] = bx1
            cdof[env, (dof_adr + 1) * 6 + 1] = by1
            cdof[env, (dof_adr + 1) * 6 + 2] = bz1
            cdof[env, (dof_adr + 1) * 6 + 3] = by1 * b_oz - bz1 * b_oy
            cdof[env, (dof_adr + 1) * 6 + 4] = bz1 * b_ox - bx1 * b_oz
            cdof[env, (dof_adr + 1) * 6 + 5] = bx1 * b_oy - by1 * b_ox
            cdof[env, (dof_adr + 2) * 6 + 0] = bx2
            cdof[env, (dof_adr + 2) * 6 + 1] = by2
            cdof[env, (dof_adr + 2) * 6 + 2] = bz2
            cdof[env, (dof_adr + 2) * 6 + 3] = by2 * b_oz - bz2 * b_oy
            cdof[env, (dof_adr + 2) * 6 + 4] = bz2 * b_ox - bx2 * b_oz
            cdof[env, (dof_adr + 2) * 6 + 5] = bx2 * b_oy - by2 * b_ox

        elif jnt_type == JNT_FREE:
            # Translation DOFs: pure linear
            cdof[env, (dof_adr + 0) * 6 + 3] = Scalar[DTYPE](1)
            cdof[env, (dof_adr + 1) * 6 + 4] = Scalar[DTYPE](1)
            cdof[env, (dof_adr + 2) * 6 + 5] = Scalar[DTYPE](1)

            # Rotation DOFs: body xmat columns + subtree_com offset
            var bqx = rebind[Scalar[DTYPE]](xquat[env, body * 4 + 0])
            var bqy = rebind[Scalar[DTYPE]](xquat[env, body * 4 + 1])
            var bqz = rebind[Scalar[DTYPE]](xquat[env, body * 4 + 2])
            var bqw = rebind[Scalar[DTYPE]](xquat[env, body * 4 + 3])
            var ax0_x = Scalar[DTYPE](1) - Scalar[DTYPE](2) * (
                bqy * bqy + bqz * bqz
            )
            var ax0_y = Scalar[DTYPE](2) * (bqx * bqy + bqw * bqz)
            var ax0_z = Scalar[DTYPE](2) * (bqx * bqz - bqw * bqy)
            var ax1_x = Scalar[DTYPE](2) * (bqx * bqy - bqw * bqz)
            var ax1_y = Scalar[DTYPE](1) - Scalar[DTYPE](2) * (
                bqx * bqx + bqz * bqz
            )
            var ax1_z = Scalar[DTYPE](2) * (bqy * bqz + bqw * bqx)
            var ax2_x = Scalar[DTYPE](2) * (bqx * bqz + bqw * bqy)
            var ax2_y = Scalar[DTYPE](2) * (bqy * bqz - bqw * bqx)
            var ax2_z = Scalar[DTYPE](1) - Scalar[DTYPE](2) * (
                bqx * bqx + bqy * bqy
            )
            var f_xpos_x = rebind[Scalar[DTYPE]](xpos[env, body * 3 + 0])
            var f_xpos_y = rebind[Scalar[DTYPE]](xpos[env, body * 3 + 1])
            var f_xpos_z = rebind[Scalar[DTYPE]](xpos[env, body * 3 + 2])
            var f_off_x = ref_x - f_xpos_x
            var f_off_y = ref_y - f_xpos_y
            var f_off_z = ref_z - f_xpos_z
            cdof[env, (dof_adr + 3) * 6 + 0] = ax0_x
            cdof[env, (dof_adr + 3) * 6 + 1] = ax0_y
            cdof[env, (dof_adr + 3) * 6 + 2] = ax0_z
            cdof[env, (dof_adr + 3) * 6 + 3] = ax0_y * f_off_z - ax0_z * f_off_y
            cdof[env, (dof_adr + 3) * 6 + 4] = ax0_z * f_off_x - ax0_x * f_off_z
            cdof[env, (dof_adr + 3) * 6 + 5] = ax0_x * f_off_y - ax0_y * f_off_x
            cdof[env, (dof_adr + 4) * 6 + 0] = ax1_x
            cdof[env, (dof_adr + 4) * 6 + 1] = ax1_y
            cdof[env, (dof_adr + 4) * 6 + 2] = ax1_z
            cdof[env, (dof_adr + 4) * 6 + 3] = ax1_y * f_off_z - ax1_z * f_off_y
            cdof[env, (dof_adr + 4) * 6 + 4] = ax1_z * f_off_x - ax1_x * f_off_z
            cdof[env, (dof_adr + 4) * 6 + 5] = ax1_x * f_off_y - ax1_y * f_off_x
            cdof[env, (dof_adr + 5) * 6 + 0] = ax2_x
            cdof[env, (dof_adr + 5) * 6 + 1] = ax2_y
            cdof[env, (dof_adr + 5) * 6 + 2] = ax2_z
            cdof[env, (dof_adr + 5) * 6 + 3] = ax2_y * f_off_z - ax2_z * f_off_y
            cdof[env, (dof_adr + 5) * 6 + 4] = ax2_z * f_off_x - ax2_x * f_off_z
            cdof[env, (dof_adr + 5) * 6 + 5] = ax2_x * f_off_y - ax2_y * f_off_x

            # FREE joint sets orientation from qpos
            var qpos_adr_val = Int(
                rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_QPOS_ADR])
            )
            acc_qx = rebind[Scalar[DTYPE]](qpos[env, qpos_adr_val + 3])
            acc_qy = rebind[Scalar[DTYPE]](qpos[env, qpos_adr_val + 4])
            acc_qz = rebind[Scalar[DTYPE]](qpos[env, qpos_adr_val + 5])
            acc_qw = rebind[Scalar[DTYPE]](qpos[env, qpos_adr_val + 6])


@always_inline
def _cdof_env[
    DTYPE: DType,
    D: DimsLike,
    L_QPOS: TensorLayout,
    L_XPOS: TensorLayout,
    L_XQUAT: TensorLayout,
    L_BODIES: TensorLayout,
    L_JOINTS: TensorLayout,
    L_CDOF: TensorLayout,
](
    env: Int,
    dims: D,
    qpos: TileTensor[DTYPE, L_QPOS, MutAnyOrigin],
    xpos: TileTensor[DTYPE, L_XPOS, MutAnyOrigin],
    xquat: TileTensor[DTYPE, L_XQUAT, MutAnyOrigin],
    subtree_com: TileTensor[DTYPE, L_XPOS, MutAnyOrigin],
    bodies: TileTensor[DTYPE, L_BODIES, MutAnyOrigin],
    joints: TileTensor[DTYPE, L_JOINTS, MutAnyOrigin],
    cdof: TileTensor[DTYPE, L_CDOF, MutAnyOrigin],
):
    """Full cdof for one env: zero + per-body walk."""
    var nv = dims.get_nv()
    var nbody = dims.get_nbody()
    for i in range(nv * 6):
        cdof[env, i] = 0

    for body in range(1, nbody):
        _cdof_body[DTYPE](
            env, body, dims, qpos, xpos, xquat, subtree_com, bodies, joints, cdof
        )


def _cdof_fields_kernel[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    BATCH: Int,
](
    qpos: Tile[DTYPE, BATCH, NQ],
    xpos: Tile[DTYPE, BATCH, NBODY * 3],
    xquat: Tile[DTYPE, BATCH, NBODY * 4],
    subtree_com: Tile[DTYPE, BATCH, NBODY * 3],
    bodies: Tile[DTYPE, NBODY, MODEL_BODY_SIZE],
    joints: Tile[DTYPE, NJOINT, MODEL_JOINT_SIZE],
    cdof: Tile[DTYPE, BATCH, NV * 6],
):
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    _cdof_env[DTYPE](
        env, Dims[nq=NQ, nv=NV, nbody=NBODY, njoint=NJOINT](), qpos, xpos, xquat, subtree_com, bodies, joints, cdof
    )


# ── Cooperative (_mt) kernel — schedule from the legacy
# `compute_cdof_gpu_mt` (dynamics/jacobian.mojo): bodies are independent
# (each writes only its own DOFs from FK state), so threads stripe over
# bodies with no level ordering; one barrier between the zero-init and the
# body sweep. Per-body arithmetic is the SAME `_cdof_body` helper as
# the serial kernel -> bit-exact. Grid is exact (one block per env) ->
# legacy valid_env guards dropped; the legacy trailing barrier is dropped
# too (kernel end is the sync point in this standalone launch).
def _cdof_fields_mt_kernel[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    BATCH: Int,
    N_THREADS: Int,
](
    qpos: Tile[DTYPE, BATCH, NQ],
    xpos: Tile[DTYPE, BATCH, NBODY * 3],
    xquat: Tile[DTYPE, BATCH, NBODY * 4],
    subtree_com: Tile[DTYPE, BATCH, NBODY * 3],
    bodies: Tile[DTYPE, NBODY, MODEL_BODY_SIZE],
    joints: Tile[DTYPE, NJOINT, MODEL_JOINT_SIZE],
    cdof: Tile[DTYPE, BATCH, NV * 6],
):
    var env = Int(block_idx.x)
    var tid = Int(thread_idx.x)

    # Zero cdof (NV*6), distributed across threads.
    for i in range(tid, NV * 6, N_THREADS):
        cdof[env, i] = 0
    barrier()

    # Per-body, independent -> stripe across threads, no per-body barrier.
    for body in range(1 + tid, NBODY, N_THREADS):
        _cdof_body[DTYPE](
            env, body, Dims[nq=NQ, nv=NV, nbody=NBODY, njoint=NJOINT](), qpos, xpos, xquat, subtree_com, bodies, joints, cdof
        )


def compute_cdof[

    target: StaticString,
    DTYPE: DType,
    D: DimsLike,
    BATCH: Int = 1,
    PARALLEL: Bool = False,
    # Appended, not grouped with NEXCLUDE — see `fields.Model`.
](
    mut d: Data[DTYPE, D, BATCH],
    mut m: Model[DTYPE, D],
    mut scratch: DynamicsScratch[DTYPE, D, BATCH],
    ctx: Optional[DeviceContext] = None,
) raises:
    """`cdof` from FK products, both targets, one body. Output goes to the
    owned `scratch.cdof` tensor. PARALLEL=True (GPU only): cooperative
    flat-parallel kernel, bit-exact vs serial. CPU ignores PARALLEL."""
    comptime L_QPOS = RowMajor[BATCH, D.NQ]
    comptime L_B3 = RowMajor[BATCH, D.NBODY * 3]
    comptime L_B4 = RowMajor[BATCH, D.NBODY * 4]
    comptime L_BODY = RowMajor[D.NBODY, MODEL_BODY_SIZE]
    comptime L_JOINT = RowMajor[D.NJOINT, MODEL_JOINT_SIZE]
    comptime L_CDOF = RowMajor[BATCH, D.NV * 6]

    comptime if target == "cpu":
        var dm = d.dims
        var rl_QPOS = rl2(BATCH, dm.get_nq())
        var rl_B3 = rl2(BATCH, dm.get_nbody() * 3)
        var rl_B4 = rl2(BATCH, dm.get_nbody() * 4)
        var rl_BODY = rl2(dm.get_nbody(), MODEL_BODY_SIZE)
        var rl_JOINT = rl2(dm.get_njoint(), MODEL_JOINT_SIZE)
        var rl_CDOF = rl2(BATCH, dm.get_nv() * 6)
        var qpos_v = d.qpos.tt_dyn["cpu", DYN2](rl_QPOS)
        var xpos_v = d.xpos.tt_dyn["cpu", DYN2](rl_B3)
        var xquat_v = d.xquat.tt_dyn["cpu", DYN2](rl_B4)
        var stcom_v = d.subtree_com.tt_dyn["cpu", DYN2](rl_B3)
        var bodies_v = m.bodies.tt_dyn["cpu", DYN2](rl_BODY)
        var joints_v = m.joints.tt_dyn["cpu", DYN2](rl_JOINT)
        var cdof_v = scratch.cdof.tt_dyn["cpu", DYN2](rl_CDOF)
        for e in range(BATCH):
            _cdof_env[DTYPE](
                e, dm, qpos_v, xpos_v, xquat_v, stcom_v, bodies_v, joints_v, cdof_v
            )
    elif PARALLEL:
        var c = ctx.value()
        comptime MT_T = D.NV
        c.enqueue_function[
            _cdof_fields_mt_kernel[DTYPE, D.NQ, D.NV, D.NBODY, D.NJOINT, BATCH, MT_T]
        ](
            d.qpos.tt["gpu", BATCH, D.NQ](),
            d.xpos.tt["gpu", BATCH, D.NBODY * 3](),
            d.xquat.tt["gpu", BATCH, D.NBODY * 4](),
            d.subtree_com.tt["gpu", BATCH, D.NBODY * 3](),
            m.bodies.tt["gpu", D.NBODY, MODEL_BODY_SIZE](),
            m.joints.tt["gpu", D.NJOINT, MODEL_JOINT_SIZE](),
            scratch.cdof.tt["gpu", BATCH, D.NV * 6](),
            grid_dim=(BATCH,),
            block_dim=(MT_T,),
        )
    else:
        var c = ctx.value()
        comptime BLOCKS = (BATCH + CDOF_TPB - 1) // CDOF_TPB
        c.enqueue_function[
            _cdof_fields_kernel[DTYPE, D.NQ, D.NV, D.NBODY, D.NJOINT, BATCH]
        ](
            d.qpos.tt["gpu", BATCH, D.NQ](),
            d.xpos.tt["gpu", BATCH, D.NBODY * 3](),
            d.xquat.tt["gpu", BATCH, D.NBODY * 4](),
            d.subtree_com.tt["gpu", BATCH, D.NBODY * 3](),
            m.bodies.tt["gpu", D.NBODY, MODEL_BODY_SIZE](),
            m.joints.tt["gpu", D.NJOINT, MODEL_JOINT_SIZE](),
            scratch.cdof.tt["gpu", BATCH, D.NV * 6](),
            grid_dim=(BLOCKS,),
            block_dim=(CDOF_TPB,),
        )
