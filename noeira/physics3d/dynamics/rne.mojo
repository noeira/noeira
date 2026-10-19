"""RNE bias forces over per-field tensors (migration P2, single-source).

Per-field port of `compute_bias_forces_rne_gpu` + `rne_fwd_body`
(dynamics/bias_forces.mojo) — arithmetic verbatim. Computes
b(q, qvel) = C(q, qvel)·qvel + g(q) into `scratch.bias`.

Operands (12): qvel, xquat, xipos, subtree_com (data) + bodies, joints,
meta (model; gravity lives in the meta record) + cdof, crb, rne_cacc,
rne_cfrc, bias (scratch). As in the legacy code, the `crb` scratch tensor
doubles as per-body cvel storage during RNE (b*6 indexing within the
NBODY*10 tensor); `cinert` stays a per-thread Array."""

from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.sync import barrier
from max.gpu.memory import AddressSpace
from max.gpu.host import DeviceContext
from layout import TileTensor, TensorLayout
from noeira.core.tiles import RowMajor, Tile, shared_tile

from ..kinematics.quat_math import gpu_quat_mul
from ..joint_types import JNT_FREE, JNT_BALL
from .body_joint_map import body_joint_map
from ..fields import (
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
from ..gpu.constants import (
    MODEL_BODY_SIZE,
    MODEL_JOINT_SIZE,
    MODEL_META_SIZE,
    MODEL_META_IDX_GRAVITY_X,
    MODEL_META_IDX_GRAVITY_Y,
    MODEL_META_IDX_GRAVITY_Z,
    BODY_IDX_MASS,
    BODY_IDX_IXX,
    BODY_IDX_IYY,
    BODY_IDX_IZZ,
    BODY_IDX_PARENT,
    BODY_IDX_IQUAT_X,
    BODY_IDX_IQUAT_Y,
    BODY_IDX_IQUAT_Z,
    BODY_IDX_IQUAT_W,
    BODY_IDX_ROOTID,
    JOINT_IDX_TYPE,
    JOINT_IDX_BODY_ID,
    JOINT_IDX_DOF_ADR,
)

comptime RNE_TPB: Int = 64


@always_inline
def _max_one[N: Int]() -> Int:
    return N if N > 0 else 1


@always_inline
def _rne_fwd_body[
    DTYPE: DType,
    D: DimsLike,
    L_QVEL: TensorLayout,
    L_BODIES: TensorLayout,
    L_JOINTS: TensorLayout,
    L_CDOF: TensorLayout,
    L_CVEL: TensorLayout,
    L_CACC: TensorLayout,
](
    env: Int,
    b: Int,
    gx: Scalar[DTYPE],
    gy: Scalar[DTYPE],
    gz: Scalar[DTYPE],
    dims: D,
    qvel: TileTensor[DTYPE, L_QVEL, MutAnyOrigin],
    bodies: TileTensor[DTYPE, L_BODIES, MutAnyOrigin],
    joints: TileTensor[DTYPE, L_JOINTS, MutAnyOrigin],
    cdof: TileTensor[DTYPE, L_CDOF, MutAnyOrigin],
    cvel: TileTensor[DTYPE, L_CVEL, MutAnyOrigin],
    cacc: TileTensor[DTYPE, L_CACC, MutAnyOrigin],
    # This body's joints as a contiguous run `[j_lo, j_hi)` of the joint
    # table (`body_joint_map` on the CPU, the block's shared map on the
    # cooperative kernel). With `map_ok` False the loop scans all `njoint`
    # rows for `JOINT_IDX_BODY_ID == b` instead. Same joints, same order.
    j_lo: Int = 0,
    j_hi: Int = -1,
    map_ok: Bool = False,
):
    """Forward-pass cvel/cacc for one body (verbatim from rne_fwd_body;
    `cvel` is the crb scratch tensor, b*6 indexing)."""
    var njoint = dims.get_njoint()
    var parent = Int(rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_PARENT]))

    var cv_wx = rebind[Scalar[DTYPE]](cvel[env, parent * 6 + 0])
    var cv_wy = rebind[Scalar[DTYPE]](cvel[env, parent * 6 + 1])
    var cv_wz = rebind[Scalar[DTYPE]](cvel[env, parent * 6 + 2])
    var cv_vx = rebind[Scalar[DTYPE]](cvel[env, parent * 6 + 3])
    var cv_vy = rebind[Scalar[DTYPE]](cvel[env, parent * 6 + 4])
    var cv_vz = rebind[Scalar[DTYPE]](cvel[env, parent * 6 + 5])

    if parent == 0:
        cacc[env, b * 6 + 0] = Scalar[DTYPE](0)
        cacc[env, b * 6 + 1] = Scalar[DTYPE](0)
        cacc[env, b * 6 + 2] = Scalar[DTYPE](0)
        cacc[env, b * 6 + 3] = -gx
        cacc[env, b * 6 + 4] = -gy
        cacc[env, b * 6 + 5] = -gz
    else:
        for k in range(6):
            cacc[env, b * 6 + k] = cacc[env, parent * 6 + k]

    var lo = j_lo if map_ok else 0
    var hi = j_hi if map_ok else njoint
    for j in range(lo, hi):
        if not map_ok:
            var jnt_body = Int(
                rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_BODY_ID])
            )
            if jnt_body != b:
                continue

        var jnt_type = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_TYPE]))
        var dof_adr = Int(
            rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_DOF_ADR])
        )

        if jnt_type == JNT_FREE:
            # Translation DOFs: cdof_dot = 0, just update cvel
            for d in range(3):
                var dof = dof_adr + d
                var qdot = rebind[Scalar[DTYPE]](qvel[env, dof])
                cv_wx = cv_wx + rebind[Scalar[DTYPE]](
                    cdof[env, dof * 6 + 0]
                ) * qdot
                cv_wy = cv_wy + rebind[Scalar[DTYPE]](
                    cdof[env, dof * 6 + 1]
                ) * qdot
                cv_wz = cv_wz + rebind[Scalar[DTYPE]](
                    cdof[env, dof * 6 + 2]
                ) * qdot
                cv_vx = cv_vx + rebind[Scalar[DTYPE]](
                    cdof[env, dof * 6 + 3]
                ) * qdot
                cv_vy = cv_vy + rebind[Scalar[DTYPE]](
                    cdof[env, dof * 6 + 4]
                ) * qdot
                cv_vz = cv_vz + rebind[Scalar[DTYPE]](
                    cdof[env, dof * 6 + 5]
                ) * qdot

            # Rotation DOFs: all 3 cdof_dots use pre-rotation cvel
            for d in range(3, 6):
                var dof = dof_adr + d
                var qdot = rebind[Scalar[DTYPE]](qvel[env, dof])
                var s_ang_x = rebind[Scalar[DTYPE]](cdof[env, dof * 6 + 0])
                var s_ang_y = rebind[Scalar[DTYPE]](cdof[env, dof * 6 + 1])
                var s_ang_z = rebind[Scalar[DTYPE]](cdof[env, dof * 6 + 2])
                var s_lin_x = rebind[Scalar[DTYPE]](cdof[env, dof * 6 + 3])
                var s_lin_y = rebind[Scalar[DTYPE]](cdof[env, dof * 6 + 4])
                var s_lin_z = rebind[Scalar[DTYPE]](cdof[env, dof * 6 + 5])

                var cdot_ang_x = cv_wy * s_ang_z - cv_wz * s_ang_y
                var cdot_ang_y = cv_wz * s_ang_x - cv_wx * s_ang_z
                var cdot_ang_z = cv_wx * s_ang_y - cv_wy * s_ang_x
                var cdot_lin_x = (cv_wy * s_lin_z - cv_wz * s_lin_y) + (
                    cv_vy * s_ang_z - cv_vz * s_ang_y
                )
                var cdot_lin_y = (cv_wz * s_lin_x - cv_wx * s_lin_z) + (
                    cv_vz * s_ang_x - cv_vx * s_ang_z
                )
                var cdot_lin_z = (cv_wx * s_lin_y - cv_wy * s_lin_x) + (
                    cv_vx * s_ang_y - cv_vy * s_ang_x
                )

                cacc[env, b * 6 + 0] = cacc[env, b * 6 + 0] + cdot_ang_x * qdot
                cacc[env, b * 6 + 1] = cacc[env, b * 6 + 1] + cdot_ang_y * qdot
                cacc[env, b * 6 + 2] = cacc[env, b * 6 + 2] + cdot_ang_z * qdot
                cacc[env, b * 6 + 3] = cacc[env, b * 6 + 3] + cdot_lin_x * qdot
                cacc[env, b * 6 + 4] = cacc[env, b * 6 + 4] + cdot_lin_y * qdot
                cacc[env, b * 6 + 5] = cacc[env, b * 6 + 5] + cdot_lin_z * qdot

            # Update cvel AFTER all 3 rotational cdof_dots
            for d in range(3, 6):
                var dof = dof_adr + d
                var qdot = rebind[Scalar[DTYPE]](qvel[env, dof])
                cv_wx = cv_wx + rebind[Scalar[DTYPE]](
                    cdof[env, dof * 6 + 0]
                ) * qdot
                cv_wy = cv_wy + rebind[Scalar[DTYPE]](
                    cdof[env, dof * 6 + 1]
                ) * qdot
                cv_wz = cv_wz + rebind[Scalar[DTYPE]](
                    cdof[env, dof * 6 + 2]
                ) * qdot
                cv_vx = cv_vx + rebind[Scalar[DTYPE]](
                    cdof[env, dof * 6 + 3]
                ) * qdot
                cv_vy = cv_vy + rebind[Scalar[DTYPE]](
                    cdof[env, dof * 6 + 4]
                ) * qdot
                cv_vz = cv_vz + rebind[Scalar[DTYPE]](
                    cdof[env, dof * 6 + 5]
                ) * qdot

        elif jnt_type == JNT_BALL:
            for d in range(3):
                var dof = dof_adr + d
                var qdot = rebind[Scalar[DTYPE]](qvel[env, dof])
                var s_ang_x = rebind[Scalar[DTYPE]](cdof[env, dof * 6 + 0])
                var s_ang_y = rebind[Scalar[DTYPE]](cdof[env, dof * 6 + 1])
                var s_ang_z = rebind[Scalar[DTYPE]](cdof[env, dof * 6 + 2])
                var s_lin_x = rebind[Scalar[DTYPE]](cdof[env, dof * 6 + 3])
                var s_lin_y = rebind[Scalar[DTYPE]](cdof[env, dof * 6 + 4])
                var s_lin_z = rebind[Scalar[DTYPE]](cdof[env, dof * 6 + 5])

                var cdot_ang_x = cv_wy * s_ang_z - cv_wz * s_ang_y
                var cdot_ang_y = cv_wz * s_ang_x - cv_wx * s_ang_z
                var cdot_ang_z = cv_wx * s_ang_y - cv_wy * s_ang_x
                var cdot_lin_x = (cv_wy * s_lin_z - cv_wz * s_lin_y) + (
                    cv_vy * s_ang_z - cv_vz * s_ang_y
                )
                var cdot_lin_y = (cv_wz * s_lin_x - cv_wx * s_lin_z) + (
                    cv_vz * s_ang_x - cv_vx * s_ang_z
                )
                var cdot_lin_z = (cv_wx * s_lin_y - cv_wy * s_lin_x) + (
                    cv_vx * s_ang_y - cv_vy * s_ang_x
                )

                cacc[env, b * 6 + 0] = cacc[env, b * 6 + 0] + cdot_ang_x * qdot
                cacc[env, b * 6 + 1] = cacc[env, b * 6 + 1] + cdot_ang_y * qdot
                cacc[env, b * 6 + 2] = cacc[env, b * 6 + 2] + cdot_ang_z * qdot
                cacc[env, b * 6 + 3] = cacc[env, b * 6 + 3] + cdot_lin_x * qdot
                cacc[env, b * 6 + 4] = cacc[env, b * 6 + 4] + cdot_lin_y * qdot
                cacc[env, b * 6 + 5] = cacc[env, b * 6 + 5] + cdot_lin_z * qdot

            for d in range(3):
                var dof = dof_adr + d
                var qdot = rebind[Scalar[DTYPE]](qvel[env, dof])
                cv_wx = cv_wx + rebind[Scalar[DTYPE]](
                    cdof[env, dof * 6 + 0]
                ) * qdot
                cv_wy = cv_wy + rebind[Scalar[DTYPE]](
                    cdof[env, dof * 6 + 1]
                ) * qdot
                cv_wz = cv_wz + rebind[Scalar[DTYPE]](
                    cdof[env, dof * 6 + 2]
                ) * qdot
                cv_vx = cv_vx + rebind[Scalar[DTYPE]](
                    cdof[env, dof * 6 + 3]
                ) * qdot
                cv_vy = cv_vy + rebind[Scalar[DTYPE]](
                    cdof[env, dof * 6 + 4]
                ) * qdot
                cv_vz = cv_vz + rebind[Scalar[DTYPE]](
                    cdof[env, dof * 6 + 5]
                ) * qdot

        else:
            # HINGE or SLIDE (1 DOF)
            var dof = dof_adr
            var qdot = rebind[Scalar[DTYPE]](qvel[env, dof])
            var s_ang_x = rebind[Scalar[DTYPE]](cdof[env, dof * 6 + 0])
            var s_ang_y = rebind[Scalar[DTYPE]](cdof[env, dof * 6 + 1])
            var s_ang_z = rebind[Scalar[DTYPE]](cdof[env, dof * 6 + 2])
            var s_lin_x = rebind[Scalar[DTYPE]](cdof[env, dof * 6 + 3])
            var s_lin_y = rebind[Scalar[DTYPE]](cdof[env, dof * 6 + 4])
            var s_lin_z = rebind[Scalar[DTYPE]](cdof[env, dof * 6 + 5])

            var cdot_ang_x = cv_wy * s_ang_z - cv_wz * s_ang_y
            var cdot_ang_y = cv_wz * s_ang_x - cv_wx * s_ang_z
            var cdot_ang_z = cv_wx * s_ang_y - cv_wy * s_ang_x
            var cdot_lin_x = (cv_wy * s_lin_z - cv_wz * s_lin_y) + (
                cv_vy * s_ang_z - cv_vz * s_ang_y
            )
            var cdot_lin_y = (cv_wz * s_lin_x - cv_wx * s_lin_z) + (
                cv_vz * s_ang_x - cv_vx * s_ang_z
            )
            var cdot_lin_z = (cv_wx * s_lin_y - cv_wy * s_lin_x) + (
                cv_vx * s_ang_y - cv_vy * s_ang_x
            )

            cacc[env, b * 6 + 0] = cacc[env, b * 6 + 0] + cdot_ang_x * qdot
            cacc[env, b * 6 + 1] = cacc[env, b * 6 + 1] + cdot_ang_y * qdot
            cacc[env, b * 6 + 2] = cacc[env, b * 6 + 2] + cdot_ang_z * qdot
            cacc[env, b * 6 + 3] = cacc[env, b * 6 + 3] + cdot_lin_x * qdot
            cacc[env, b * 6 + 4] = cacc[env, b * 6 + 4] + cdot_lin_y * qdot
            cacc[env, b * 6 + 5] = cacc[env, b * 6 + 5] + cdot_lin_z * qdot

            cv_wx = cv_wx + s_ang_x * qdot
            cv_wy = cv_wy + s_ang_y * qdot
            cv_wz = cv_wz + s_ang_z * qdot
            cv_vx = cv_vx + s_lin_x * qdot
            cv_vy = cv_vy + s_lin_y * qdot
            cv_vz = cv_vz + s_lin_z * qdot

    cvel[env, b * 6 + 0] = cv_wx
    cvel[env, b * 6 + 1] = cv_wy
    cvel[env, b * 6 + 2] = cv_wz
    cvel[env, b * 6 + 3] = cv_vx
    cvel[env, b * 6 + 4] = cv_vy
    cvel[env, b * 6 + 5] = cv_vz


@always_inline
def _rne_cinert_body[
    DTYPE: DType,
    CIN_CAP: Int,
    L_XQUAT: TensorLayout,
    L_XIPOS: TensorLayout,
    L_BODIES: TensorLayout,
](
    env: Int,
    b: Int,
    xquat: TileTensor[DTYPE, L_XQUAT, MutAnyOrigin],
    xipos: TileTensor[DTYPE, L_XIPOS, MutAnyOrigin],
    subtree_com: TileTensor[DTYPE, L_XIPOS, MutAnyOrigin],
    bodies: TileTensor[DTYPE, L_BODIES, MutAnyOrigin],
    mut cinert_g: Scratch[Scalar[DTYPE], CIN_CAP],
):
    """One body's cinert (spatial inertia at subtree_com, mj_inertCom).
    Extracted verbatim from the `_rne_env` step-0 loop body so
    serial and _mt schedules share identical arithmetic."""
    var Ixx_local = rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_IXX])
    var Iyy_local = rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_IYY])
    var Izz_local = rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_IZZ])

    var bqx = rebind[Scalar[DTYPE]](xquat[env, b * 4 + 0])
    var bqy = rebind[Scalar[DTYPE]](xquat[env, b * 4 + 1])
    var bqz = rebind[Scalar[DTYPE]](xquat[env, b * 4 + 2])
    var bqw = rebind[Scalar[DTYPE]](xquat[env, b * 4 + 3])
    var iqx = rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_IQUAT_X])
    var iqy = rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_IQUAT_Y])
    var iqz = rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_IQUAT_Z])
    var iqw = rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_IQUAT_W])
    var iq = gpu_quat_mul(bqx, bqy, bqz, bqw, iqx, iqy, iqz, iqw)
    var qx = iq[0]
    var qy = iq[1]
    var qz = iq[2]
    var qw = iq[3]

    var r00 = Scalar[DTYPE](1) - Scalar[DTYPE](2) * (qy * qy + qz * qz)
    var r10 = Scalar[DTYPE](2) * (qx * qy + qw * qz)
    var r20 = Scalar[DTYPE](2) * (qx * qz - qw * qy)
    var r01 = Scalar[DTYPE](2) * (qx * qy - qw * qz)
    var r11 = Scalar[DTYPE](1) - Scalar[DTYPE](2) * (qx * qx + qz * qz)
    var r21 = Scalar[DTYPE](2) * (qy * qz + qw * qx)
    var r02 = Scalar[DTYPE](2) * (qx * qz + qw * qy)
    var r12 = Scalar[DTYPE](2) * (qy * qz - qw * qx)
    var r22 = Scalar[DTYPE](1) - Scalar[DTYPE](2) * (qx * qx + qy * qy)

    var mass_b = rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_MASS])
    cinert_g[b * 10 + 0] = (
        Ixx_local * r00 * r00 + Iyy_local * r01 * r01 + Izz_local * r02 * r02
    )
    cinert_g[b * 10 + 1] = (
        Ixx_local * r10 * r10 + Iyy_local * r11 * r11 + Izz_local * r12 * r12
    )
    cinert_g[b * 10 + 2] = (
        Ixx_local * r20 * r20 + Iyy_local * r21 * r21 + Izz_local * r22 * r22
    )
    cinert_g[b * 10 + 3] = (
        Ixx_local * r00 * r10 + Iyy_local * r01 * r11 + Izz_local * r02 * r12
    )
    cinert_g[b * 10 + 4] = (
        Ixx_local * r00 * r20 + Iyy_local * r01 * r21 + Izz_local * r02 * r22
    )
    cinert_g[b * 10 + 5] = (
        Ixx_local * r10 * r20 + Iyy_local * r11 * r21 + Izz_local * r12 * r22
    )
    var rootid_b = Int(rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_ROOTID]))
    var dx_b = rebind[Scalar[DTYPE]](
        xipos[env, b * 3 + 0]
    ) - rebind[Scalar[DTYPE]](subtree_com[env, rootid_b * 3 + 0])
    var dy_b = rebind[Scalar[DTYPE]](
        xipos[env, b * 3 + 1]
    ) - rebind[Scalar[DTYPE]](subtree_com[env, rootid_b * 3 + 1])
    var dz_b = rebind[Scalar[DTYPE]](
        xipos[env, b * 3 + 2]
    ) - rebind[Scalar[DTYPE]](subtree_com[env, rootid_b * 3 + 2])
    cinert_g[b * 10 + 0] = cinert_g[b * 10 + 0] + mass_b * (
        dy_b * dy_b + dz_b * dz_b
    )
    cinert_g[b * 10 + 1] = cinert_g[b * 10 + 1] + mass_b * (
        dx_b * dx_b + dz_b * dz_b
    )
    cinert_g[b * 10 + 2] = cinert_g[b * 10 + 2] + mass_b * (
        dx_b * dx_b + dy_b * dy_b
    )
    cinert_g[b * 10 + 3] = cinert_g[b * 10 + 3] - mass_b * dx_b * dy_b
    cinert_g[b * 10 + 4] = cinert_g[b * 10 + 4] - mass_b * dx_b * dz_b
    cinert_g[b * 10 + 5] = cinert_g[b * 10 + 5] - mass_b * dy_b * dz_b
    cinert_g[b * 10 + 6] = mass_b * dx_b
    cinert_g[b * 10 + 7] = mass_b * dy_b
    cinert_g[b * 10 + 8] = mass_b * dz_b
    cinert_g[b * 10 + 9] = mass_b


@always_inline
def _rne_cfrc_body[
    DTYPE: DType,
    CIN_CAP: Int,
    L_CRB: TensorLayout,
    L_RNE_CACC: TensorLayout,
](
    env: Int,
    b: Int,
    cinert_g: Scratch[Scalar[DTYPE], CIN_CAP],
    crb: TileTensor[DTYPE, L_CRB, MutAnyOrigin],
    rne_cacc: TileTensor[DTYPE, L_RNE_CACC, MutAnyOrigin],
    rne_cfrc: TileTensor[DTYPE, L_RNE_CACC, MutAnyOrigin],
):
    """One body's spatial force cfrc = I*cacc + cvel x* (I*cvel). Extracted
    verbatim from the `_rne_env` step-2 loop body. Reads only this
    body's cinert_g slot (callers must use the same body->thread mapping
    for cinert and cfrc, like the legacy _mt)."""
    var wx = rebind[Scalar[DTYPE]](crb[env, b * 6 + 0])
    var wy = rebind[Scalar[DTYPE]](crb[env, b * 6 + 1])
    var wz = rebind[Scalar[DTYPE]](crb[env, b * 6 + 2])
    var vx = rebind[Scalar[DTYPE]](crb[env, b * 6 + 3])
    var vy = rebind[Scalar[DTYPE]](crb[env, b * 6 + 4])
    var vz = rebind[Scalar[DTYPE]](crb[env, b * 6 + 5])

    var ci0 = cinert_g[b * 10 + 0]
    var ci1 = cinert_g[b * 10 + 1]
    var ci2 = cinert_g[b * 10 + 2]
    var ci3 = cinert_g[b * 10 + 3]
    var ci4 = cinert_g[b * 10 + 4]
    var ci5 = cinert_g[b * 10 + 5]
    var ci6 = cinert_g[b * 10 + 6]
    var ci7 = cinert_g[b * 10 + 7]
    var ci8 = cinert_g[b * 10 + 8]
    var ci9 = cinert_g[b * 10 + 9]
    var ax = rne_cacc[env, b * 6 + 0]
    var ay = rne_cacc[env, b * 6 + 1]
    var az = rne_cacc[env, b * 6 + 2]
    var alx = rne_cacc[env, b * 6 + 3]
    var aly = rne_cacc[env, b * 6 + 4]
    var alz = rne_cacc[env, b * 6 + 5]

    var Ia0 = ci0 * ax + ci3 * ay + ci4 * az - ci8 * aly + ci7 * alz
    var Ia1 = ci3 * ax + ci1 * ay + ci5 * az + ci8 * alx - ci6 * alz
    var Ia2 = ci4 * ax + ci5 * ay + ci2 * az - ci7 * alx + ci6 * aly
    var Ia3 = ci8 * ay - ci7 * az + ci9 * alx
    var Ia4 = ci6 * az - ci8 * ax + ci9 * aly
    var Ia5 = ci7 * ax - ci6 * ay + ci9 * alz

    var Iv0 = ci0 * wx + ci3 * wy + ci4 * wz - ci8 * vy + ci7 * vz
    var Iv1 = ci3 * wx + ci1 * wy + ci5 * wz + ci8 * vx - ci6 * vz
    var Iv2 = ci4 * wx + ci5 * wy + ci2 * wz - ci7 * vx + ci6 * vy
    var Iv3 = ci8 * wy - ci7 * wz + ci9 * vx
    var Iv4 = ci6 * wz - ci8 * wx + ci9 * vy
    var Iv5 = ci7 * wx - ci6 * wy + ci9 * vz

    var xf0 = wy * Iv2 - wz * Iv1 + vy * Iv5 - vz * Iv4
    var xf1 = wz * Iv0 - wx * Iv2 + vz * Iv3 - vx * Iv5
    var xf2 = wx * Iv1 - wy * Iv0 + vx * Iv4 - vy * Iv3
    var xf3 = wy * Iv5 - wz * Iv4
    var xf4 = wz * Iv3 - wx * Iv5
    var xf5 = wx * Iv4 - wy * Iv3

    rne_cfrc[env, b * 6 + 0] = Ia0 + xf0
    rne_cfrc[env, b * 6 + 1] = Ia1 + xf1
    rne_cfrc[env, b * 6 + 2] = Ia2 + xf2
    rne_cfrc[env, b * 6 + 3] = Ia3 + xf3
    rne_cfrc[env, b * 6 + 4] = Ia4 + xf4
    rne_cfrc[env, b * 6 + 5] = Ia5 + xf5


@always_inline
def _rne_backward_env[
    DTYPE: DType,
    D: DimsLike,
    L_BODIES: TensorLayout,
    L_RNE_CFRC: TensorLayout,
](
    env: Int,
    dims: D,
    bodies: TileTensor[DTYPE, L_BODIES, MutAnyOrigin],
    rne_cfrc: TileTensor[DTYPE, L_RNE_CFRC, MutAnyOrigin],
):
    """Backward cfrc accumulation (leaves to root; strictly sequential, one
    caller thread). Extracted verbatim from `_rne_env` step 3."""
    var nbody = dims.get_nbody()
    for b in range(nbody - 1, 0, -1):
        var parent = Int(rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_PARENT]))
        if parent > 0:
            for k in range(6):
                rne_cfrc[env, parent * 6 + k] = (
                    rne_cfrc[env, parent * 6 + k] + rne_cfrc[env, b * 6 + k]
                )


@always_inline
def _rne_project_joint[
    DTYPE: DType,
    L_JOINTS: TensorLayout,
    L_CDOF: TensorLayout,
    L_RNE_CFRC: TensorLayout,
    L_BIAS: TensorLayout,
](
    env: Int,
    j: Int,
    joints: TileTensor[DTYPE, L_JOINTS, MutAnyOrigin],
    cdof: TileTensor[DTYPE, L_CDOF, MutAnyOrigin],
    rne_cfrc: TileTensor[DTYPE, L_RNE_CFRC, MutAnyOrigin],
    bias: TileTensor[DTYPE, L_BIAS, MutAnyOrigin],
):
    """Project one joint's DOFs to joint space:
    bias[d] = cdof[d] . cfrc[body_of_dof]. Extracted verbatim from the
    `_rne_env` step-4 loop body (joints own disjoint DOFs ->
    striping joints across threads is race-free)."""
    var jnt_type = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_TYPE]))
    var body = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_BODY_ID]))
    var dof_adr = Int(
        rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_DOF_ADR])
    )
    var num_dof = 1
    if jnt_type == JNT_FREE:
        num_dof = 6
    elif jnt_type == JNT_BALL:
        num_dof = 3

    for d in range(num_dof):
        var dof = dof_adr + d
        bias[env, dof] = 0
        for k in range(6):
            bias[env, dof] = (
                bias[env, dof]
                + cdof[env, dof * 6 + k] * rne_cfrc[env, body * 6 + k]
            )


@always_inline
def _rne_env[
    DTYPE: DType,
    D: DimsLike,
    L_QVEL: TensorLayout,
    L_XQUAT: TensorLayout,
    L_XIPOS: TensorLayout,
    L_BODIES: TensorLayout,
    L_JOINTS: TensorLayout,
    L_META: TensorLayout,
    L_CDOF: TensorLayout,
    L_CRB: TensorLayout,
    L_RNE_CACC: TensorLayout,
    # CPU dispatcher only: derive the body → joint map once per env and hand
    # it to the per-body forward pass (see `body_joint_map`). The serial GPU
    # kernel keeps the scan (no per-thread table); the cooperative kernel
    # builds the map once per block in threadgroup memory.
    JMAP: Bool = False,
](
    env: Int,
    dims: D,
    qvel: TileTensor[DTYPE, L_QVEL, MutAnyOrigin],
    xquat: TileTensor[DTYPE, L_XQUAT, MutAnyOrigin],
    xipos: TileTensor[DTYPE, L_XIPOS, MutAnyOrigin],
    subtree_com: TileTensor[DTYPE, L_XIPOS, MutAnyOrigin],
    bodies: TileTensor[DTYPE, L_BODIES, MutAnyOrigin],
    joints: TileTensor[DTYPE, L_JOINTS, MutAnyOrigin],
    meta: TileTensor[DTYPE, L_META, MutAnyOrigin],
    cdof: TileTensor[DTYPE, L_CDOF, MutAnyOrigin],
    crb: TileTensor[DTYPE, L_CRB, MutAnyOrigin],
    rne_cacc: TileTensor[DTYPE, L_RNE_CACC, MutAnyOrigin],
    rne_cfrc: TileTensor[DTYPE, L_RNE_CACC, MutAnyOrigin],
    bias: TileTensor[DTYPE, L_QVEL, MutAnyOrigin],
):
    """Full RNE for one env (verbatim from compute_bias_forces_rne_gpu)."""
    var nv = dims.get_nv()
    var nbody = dims.get_nbody()
    var njoint = dims.get_njoint()
    for i in range(nv):
        bias[env, i] = 0

    var gx = rebind[Scalar[DTYPE]](meta[MODEL_META_IDX_GRAVITY_X])
    var gy = rebind[Scalar[DTYPE]](meta[MODEL_META_IDX_GRAVITY_Y])
    var gz = rebind[Scalar[DTYPE]](meta[MODEL_META_IDX_GRAVITY_Z])

    # ⚠ `nbody * 6`, NOT a cap. These bound writes into TENSORS, so the cap
    # never sized anything here -- and `_max_one` collapses to 1 on a dynamic
    # provider, which would leave `rne_cacc`/`rne_cfrc` holding the PREVIOUS
    # step's values from index 1 on. Silent: the static leg has cap == exact,
    # so no gate in the tree can see it. Same defect as `rne_post`; found by
    # `audit_caps.py`, and the `range(nbody * 10)` three lines below is what
    # the correct form looks like.
    for i in range(nbody * 6):
        rne_cacc[env, i] = Scalar[DTYPE](0)
    for i in range(nbody * 6):
        rne_cfrc[env, i] = Scalar[DTYPE](0)
    comptime CINERT_GPU_SIZE = cap[D.NBODY]() * 10
    var cinert_g = Scratch[Scalar[DTYPE], CINERT_GPU_SIZE](
        nbody * 10, uninitialized=0
    )
    for i in range(nbody * 10):
        cinert_g[i] = Scalar[DTYPE](0)

    # Step 0: cinert — spatial inertia at subtree_com (mj_inertCom)
    for b in range(nbody):
        _rne_cinert_body[DTYPE](
            env, b, xquat, xipos, subtree_com, bodies, cinert_g
        )

    # Per-body spatial velocity stored in the crb tensor (b*6 indexing)
    for i in range(nbody * 6):
        crb[env, i] = 0

    # Step 1: Forward pass — cvel and cacc (root to leaves)
    comptime JM_CAP = cap[D.NBODY]() if JMAP else 1
    var jnt_adr = Scratch[Int, JM_CAP](nbody if JMAP else 1, fill=-1)
    var jnt_num = Scratch[Int, JM_CAP](nbody if JMAP else 1, fill=0)
    var map_ok = False
    comptime if JMAP:
        map_ok = body_joint_map[DTYPE, JM_CAP](
            njoint, nbody, joints, jnt_adr, jnt_num
        )
    for b in range(1, nbody):
        var j_lo = jnt_adr[b] if map_ok else 0
        var j_hi = j_lo + jnt_num[b] if map_ok else njoint
        _rne_fwd_body[DTYPE](
            env, b, gx, gy, gz, dims, qvel, bodies, joints, cdof, crb, rne_cacc,
            j_lo, j_hi, map_ok,
        )

    # Step 2: Spatial forces per body: cfrc = I*cacc + cvel x* (I*cvel)
    for b in range(nbody):
        _rne_cfrc_body[DTYPE](
            env, b, cinert_g, crb, rne_cacc, rne_cfrc
        )

    # Step 3: Backward pass — simple addition
    _rne_backward_env[DTYPE](env, dims, bodies, rne_cfrc)

    # Step 4: Project to joint space: bias[d] = cdof[d] . cfrc[body_of_dof]
    for j in range(njoint):
        _rne_project_joint[DTYPE](
            env, j, joints, cdof, rne_cfrc, bias
        )


def _rne_fields_kernel[
    DTYPE: DType,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    BATCH: Int,
](
    qvel: Tile[DTYPE, BATCH, NV],
    xquat: Tile[DTYPE, BATCH, NBODY * 4],
    xipos: Tile[DTYPE, BATCH, NBODY * 3],
    subtree_com: Tile[DTYPE, BATCH, NBODY * 3],
    bodies: Tile[DTYPE, NBODY, MODEL_BODY_SIZE],
    joints: Tile[DTYPE, NJOINT, MODEL_JOINT_SIZE],
    meta: Tile[DTYPE, MODEL_META_SIZE],
    cdof: Tile[DTYPE, BATCH, NV * 6],
    crb: Tile[DTYPE, BATCH, NBODY * 10],
    rne_cacc: Tile[DTYPE, BATCH, NBODY * 6],
    rne_cfrc: Tile[DTYPE, BATCH, NBODY * 6],
    bias: Tile[DTYPE, BATCH, NV],
):
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    _rne_env[DTYPE](
        env, Dims[nv=NV, nbody=NBODY, njoint=NJOINT](), qvel, xquat, xipos, subtree_com, bodies, joints, meta,
        cdof, crb, rne_cacc, rne_cfrc, bias,
    )


# ── Cooperative (_mt) kernel — schedule from the legacy
# `compute_bias_forces_rne_gpu_mt` (dynamics/bias_forces.mojo): striped
# init; cinert flat-striped into a PER-THREAD Array; forward
# cvel/cacc level-parallel (same `_rne_fwd_body` helper, barrier per
# level); cfrc flat-striped with the SAME body->thread mapping as cinert
# (so cinert_g[b] is thread-local); backward pass level-parallel in gather
# form (§13.47; serial on tid 0 before); joint projection flat-striped
# (joints own disjoint DOFs). The forward/cfrc/projection helpers are the
# ones the serial kernel calls; the backward pass reproduces the serial
# summation order -> bit-exact. Grid is exact (one block per
# env) -> legacy valid_env guards dropped; trailing barrier dropped.
def _rne_fields_mt_kernel[
    DTYPE: DType,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    BATCH: Int,
    N_THREADS: Int,
](
    qvel: Tile[DTYPE, BATCH, NV],
    xquat: Tile[DTYPE, BATCH, NBODY * 4],
    xipos: Tile[DTYPE, BATCH, NBODY * 3],
    subtree_com: Tile[DTYPE, BATCH, NBODY * 3],
    bodies: Tile[DTYPE, NBODY, MODEL_BODY_SIZE],
    joints: Tile[DTYPE, NJOINT, MODEL_JOINT_SIZE],
    meta: Tile[DTYPE, MODEL_META_SIZE],
    cdof: Tile[DTYPE, BATCH, NV * 6],
    crb: Tile[DTYPE, BATCH, NBODY * 10],
    rne_cacc: Tile[DTYPE, BATCH, NBODY * 6],
    rne_cfrc: Tile[DTYPE, BATCH, NBODY * 6],
    bias: Tile[DTYPE, BATCH, NV],
):
    var env = Int(block_idx.x)
    var tid = Int(thread_idx.x)

    var gx = rebind[Scalar[DTYPE]](meta[MODEL_META_IDX_GRAVITY_X])
    var gy = rebind[Scalar[DTYPE]](meta[MODEL_META_IDX_GRAVITY_Y])
    var gz = rebind[Scalar[DTYPE]](meta[MODEL_META_IDX_GRAVITY_Z])

    # ⚠ THE MODEL'S TOPOLOGY, ONCE PER BLOCK, IN THREADGROUP MEMORY. Before
    # §13.47 every thread rebuilt the body-level table from `NBODY` global
    # parent reads, every body scanned all `NJOINT` rows of the joint table
    # for its joints on every level of the forward pass, and the backward
    # accumulation ran on thread 0 as `NBODY` chained read-modify-writes of
    # GLOBAL memory — the CRBA kernel's shape (§13.46), which paid 18×.
    # Layout: `[0, NBODY)` parent body, `[NBODY, 2·NBODY)` first joint of
    # the body, `[2·NBODY, 3·NBODY)` its joint count — or -1 when the
    # body's joints are not one contiguous run, in which case that body
    # scans (`body_joint_map`'s rule, decided per body). Built by a thread
    # per body: one parent read, one pass over the joint table.
    var topo_sh = shared_tile[DType.int32, 3 * NBODY]()
    var topo = topo_sh.ptr
    for b in range(tid, NBODY, N_THREADS):
        topo[unsafe_offset=b] = Int32(Int(rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_PARENT])))
        var adr = -1
        var num = 0
        var contiguous = True
        for j in range(NJOINT):
            var jb = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_BODY_ID]))
            if jb == b:
                if adr < 0:
                    adr = j
                elif adr + num != j:
                    contiguous = False
                num += 1
        topo[unsafe_offset=NBODY + b] = Int32(adr)
        topo[unsafe_offset=2 * NBODY + b] = Int32(num if contiguous else -1)

    # Init bias / cvel(crb) / cacc / cfrc (distributed).
    for i in range(tid, NV, N_THREADS):
        bias[env, i] = Scalar[DTYPE](0)
    for i in range(tid, NBODY * 6, N_THREADS):
        crb[env, i] = Scalar[DTYPE](0)
        rne_cacc[env, i] = Scalar[DTYPE](0)
        rne_cfrc[env, i] = Scalar[DTYPE](0)
    barrier()

    # Body tree depth (level) — from the shared parent table, identical in
    # every thread -> identical barrier count.
    var level = Array[Int, NBODY](fill=0)
    var max_level = 0
    for b in range(1, NBODY):
        level[b] = level[Int(topo[unsafe_offset=b])] + 1
        if level[b] > max_level:
            max_level = level[b]

    # Step 0: cinert (flat, my bodies -> per-thread cinert_g slots).
    comptime CINERT_GPU_SIZE = cap[NBODY]() * 10
    var cinert_g = Scratch[Scalar[DTYPE], CINERT_GPU_SIZE](
        NBODY * 10, uninitialized=0
    )
    for b in range(tid, NBODY, N_THREADS):
        _rne_cinert_body[DTYPE](
            env, b, xquat, xipos, subtree_com, bodies, cinert_g
        )

    # Step 1: forward cvel/cacc (level-parallel, shared helper; each body's
    # joints from the shared map).
    for lvl in range(1, max_level + 1):
        for b in range(1 + tid, NBODY, N_THREADS):
            if level[b] == lvl:
                var num = Int(topo[unsafe_offset=2 * NBODY + b])
                var j_lo = Int(topo[unsafe_offset=NBODY + b])
                _rne_fwd_body[DTYPE](
                    env, b, gx, gy, gz, Dims[nv=NV, nbody=NBODY, njoint=NJOINT](), qvel, bodies, joints, cdof, crb,
                    rne_cacc, j_lo, j_lo + num, num >= 0,
                )
        barrier()

    # Step 2: cfrc (flat, SAME mapping as cinert so cinert_g[b] is local).
    for b in range(tid, NBODY, N_THREADS):
        _rne_cfrc_body[DTYPE](
            env, b, cinert_g, crb, rne_cacc, rne_cfrc
        )
    barrier()

    # Step 3: backward cfrc accumulation, leaves to root, level-parallel in
    # GATHER form: a body at level `lvl` adds its children (all at `lvl+1`,
    # complete after the previous barrier) into its own row, in DECREASING
    # child index — the order in which the serial pass (`_rne_backward_env`:
    # `for b in nbody-1..1: cfrc[parent[b]] += cfrc[b]`) delivered them to
    # this parent, each child's row final at that moment there as here. Same
    # additions, same order, same rounding: bit-exact. World (body 0) is
    # never a target, as in the serial pass's `parent > 0`.
    for lvl in range(max_level, 0, -1):
        for b in range(1 + tid, NBODY, N_THREADS):
            if level[b] == lvl:
                var f0 = rne_cfrc[env, b * 6 + 0]
                var f1 = rne_cfrc[env, b * 6 + 1]
                var f2 = rne_cfrc[env, b * 6 + 2]
                var f3 = rne_cfrc[env, b * 6 + 3]
                var f4 = rne_cfrc[env, b * 6 + 4]
                var f5 = rne_cfrc[env, b * 6 + 5]
                for c in range(NBODY - 1, b, -1):
                    if Int(topo[unsafe_offset=c]) == b:
                        f0 = f0 + rne_cfrc[env, c * 6 + 0]
                        f1 = f1 + rne_cfrc[env, c * 6 + 1]
                        f2 = f2 + rne_cfrc[env, c * 6 + 2]
                        f3 = f3 + rne_cfrc[env, c * 6 + 3]
                        f4 = f4 + rne_cfrc[env, c * 6 + 4]
                        f5 = f5 + rne_cfrc[env, c * 6 + 5]
                rne_cfrc[env, b * 6 + 0] = f0
                rne_cfrc[env, b * 6 + 1] = f1
                rne_cfrc[env, b * 6 + 2] = f2
                rne_cfrc[env, b * 6 + 3] = f3
                rne_cfrc[env, b * 6 + 4] = f4
                rne_cfrc[env, b * 6 + 5] = f5
        barrier()

    # Step 4: qfrc projection (flat per joint; disjoint DOFs).
    for j in range(tid, NJOINT, N_THREADS):
        _rne_project_joint[DTYPE](
            env, j, joints, cdof, rne_cfrc, bias
        )


def compute_bias_forces_rne[

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
    """RNE bias forces, both targets, one body. Reads FK products + qvel +
    `scratch.cdof`; writes `scratch.bias` (+ crb/rne_cacc/rne_cfrc temps).
    PARALLEL=True (GPU only): cooperative level-parallel kernel, bit-exact
    vs serial. CPU ignores PARALLEL."""
    comptime L_NV = RowMajor[BATCH, D.NV]
    comptime L_B3 = RowMajor[BATCH, D.NBODY * 3]
    comptime L_B4 = RowMajor[BATCH, D.NBODY * 4]
    comptime L_BODY = RowMajor[D.NBODY, MODEL_BODY_SIZE]
    comptime L_JOINT = RowMajor[D.NJOINT, MODEL_JOINT_SIZE]
    comptime L_META = RowMajor[MODEL_META_SIZE]
    comptime L_CDOF = RowMajor[BATCH, D.NV * 6]
    comptime L_CRB = RowMajor[BATCH, D.NBODY * 10]
    comptime L_B6 = RowMajor[BATCH, D.NBODY * 6]

    comptime if target == "cpu":
        var dm = d.dims
        var rl_NV = rl2(BATCH, dm.get_nv())
        var rl_B4 = rl2(BATCH, dm.get_nbody() * 4)
        var rl_B3 = rl2(BATCH, dm.get_nbody() * 3)
        var rl_BODY = rl2(dm.get_nbody(), MODEL_BODY_SIZE)
        var rl_JOINT = rl2(dm.get_njoint(), MODEL_JOINT_SIZE)
        var rl_META = rl1(MODEL_META_SIZE)
        var rl_CDOF = rl2(BATCH, dm.get_nv() * 6)
        var rl_CRB = rl2(BATCH, dm.get_nbody() * 10)
        var rl_B6 = rl2(BATCH, dm.get_nbody() * 6)
        var qvel_v = d.qvel.tt_dyn["cpu", DYN2](rl_NV)
        var xquat_v = d.xquat.tt_dyn["cpu", DYN2](rl_B4)
        var xipos_v = d.xipos.tt_dyn["cpu", DYN2](rl_B3)
        var stcom_v = d.subtree_com.tt_dyn["cpu", DYN2](rl_B3)
        var bodies_v = m.bodies.tt_dyn["cpu", DYN2](rl_BODY)
        var joints_v = m.joints.tt_dyn["cpu", DYN2](rl_JOINT)
        var meta_v = m.meta.tt_dyn["cpu", DYN1](rl_META)
        var cdof_v = scratch.cdof.tt_dyn["cpu", DYN2](rl_CDOF)
        var crb_v = scratch.crb.tt_dyn["cpu", DYN2](rl_CRB)
        var cacc_v = scratch.rne_cacc.tt_dyn["cpu", DYN2](rl_B6)
        var cfrc_v = scratch.rne_cfrc.tt_dyn["cpu", DYN2](rl_B6)
        var bias_v = scratch.bias.tt_dyn["cpu", DYN2](rl_NV)
        for e in range(BATCH):
            _rne_env[DTYPE, JMAP=True](
                e, dm, qvel_v, xquat_v, xipos_v, stcom_v, bodies_v, joints_v,
                meta_v, cdof_v, crb_v, cacc_v, cfrc_v, bias_v,
            )
    elif PARALLEL:
        var c = ctx.value()
        comptime MT_T = D.NV
        c.enqueue_function[
            _rne_fields_mt_kernel[DTYPE, D.NV, D.NBODY, D.NJOINT, BATCH, MT_T]
        ](
            d.qvel.tt["gpu", BATCH, D.NV](),
            d.xquat.tt["gpu", BATCH, D.NBODY * 4](),
            d.xipos.tt["gpu", BATCH, D.NBODY * 3](),
            d.subtree_com.tt["gpu", BATCH, D.NBODY * 3](),
            m.bodies.tt["gpu", D.NBODY, MODEL_BODY_SIZE](),
            m.joints.tt["gpu", D.NJOINT, MODEL_JOINT_SIZE](),
            m.meta.tt["gpu", MODEL_META_SIZE](),
            scratch.cdof.tt["gpu", BATCH, D.NV * 6](),
            scratch.crb.tt["gpu", BATCH, D.NBODY * 10](),
            scratch.rne_cacc.tt["gpu", BATCH, D.NBODY * 6](),
            scratch.rne_cfrc.tt["gpu", BATCH, D.NBODY * 6](),
            scratch.bias.tt["gpu", BATCH, D.NV](),
            grid_dim=(BATCH,),
            block_dim=(MT_T,),
        )
    else:
        var c = ctx.value()
        comptime BLOCKS = (BATCH + RNE_TPB - 1) // RNE_TPB
        c.enqueue_function[
            _rne_fields_kernel[DTYPE, D.NV, D.NBODY, D.NJOINT, BATCH]
        ](
            d.qvel.tt["gpu", BATCH, D.NV](),
            d.xquat.tt["gpu", BATCH, D.NBODY * 4](),
            d.xipos.tt["gpu", BATCH, D.NBODY * 3](),
            d.subtree_com.tt["gpu", BATCH, D.NBODY * 3](),
            m.bodies.tt["gpu", D.NBODY, MODEL_BODY_SIZE](),
            m.joints.tt["gpu", D.NJOINT, MODEL_JOINT_SIZE](),
            m.meta.tt["gpu", MODEL_META_SIZE](),
            scratch.cdof.tt["gpu", BATCH, D.NV * 6](),
            scratch.crb.tt["gpu", BATCH, D.NBODY * 10](),
            scratch.rne_cacc.tt["gpu", BATCH, D.NBODY * 6](),
            scratch.rne_cfrc.tt["gpu", BATCH, D.NBODY * 6](),
            scratch.bias.tt["gpu", BATCH, D.NV](),
            grid_dim=(BLOCKS,),
            block_dim=(RNE_TPB,),
        )
