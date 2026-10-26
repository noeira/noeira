"""`mj_integratePos` — advance `qpos` by a velocity-space vector, standalone.

WHY THIS EXISTS WHEN `EulerIntegrator` ALREADY DOES IT. The same loop is
inlined in `integrator/euler.mojo::_finalize_env`, fused with the qvel update
and clamping that belong to a physics step. Inverse kinematics needs the
position update ALONE, applied to a Newton-style search direction that is not
a velocity and has nothing to do with a timestep — `qpos_from_site_pose` calls
`mj_integratePos(m, qpos, update_nv, 1)` with `dt = 1`. Extracting it keeps IK
from either duplicating quaternion bookkeeping or dragging a whole integrator
step in behind it.

Transcribed from `engine_support.c:mj_integratePosInd` (`mj_integratePos` is
that with `index = NULL`, `nbody = m->nbody`).

MuJoCo walks bodies and then each body's joints; we walk the joint array
directly. Same set of joints, and each writes a disjoint slice of `qpos`, so
the order cannot matter.

⚠ THE FREE-JOINT ROTATION IS NOT A SEPARATE CASE IN MuJoCo. `mjJNT_FREE` falls
THROUGH into `mjJNT_BALL` after advancing the three translational entries —
which is why the quaternion update below is shared rather than duplicated. A
transcription that gives FREE its own quaternion branch and stops is the same
code today and drifts the moment the ball branch is touched.

⚠ THE BALL BRANCH IS UNEXERCISED. Nothing in the tree has a ball joint (the
gap `envs/dm_control/gpu_reset.mojo` records), so the `JNT_BALL` entry point
here has never run against MuJoCo, and neither has the qpos layout it assumes
(w first, matching FREE). It is written for faithfulness, not because it is
known to be right. Anything relying on it must gate it first.
"""

from layout import TileTensor, TensorLayout

from .quat_math import quat_integrate
from ..fields import DimsLike
from ..joint_types import JNT_FREE, JNT_BALL, JNT_HINGE, JNT_SLIDE
from ..gpu.constants import (
    MODEL_JOINT_SIZE,
    JOINT_IDX_TYPE,
    JOINT_IDX_QPOS_ADR,
    JOINT_IDX_DOF_ADR,
)


@always_inline
def integrate_pos[
    DTYPE: DType,
    D: DimsLike,
    L_QPOS: TensorLayout,
    L_DQ: TensorLayout,
    L_JOINTS: TensorLayout,
](
    env: Int,
    dims: D,
    qpos: TileTensor[DTYPE, L_QPOS, MutAnyOrigin],
    dq: TileTensor[DTYPE, L_DQ, MutAnyOrigin],
    joints: TileTensor[DTYPE, L_JOINTS, MutAnyOrigin],
    dt: Scalar[DTYPE],
):
    """`qpos <- qpos (+) dt * dq`, quaternion-aware. Mutates `qpos` in place.

    `dq` is indexed in velocity space (nv), `qpos` in position space (nq); the
    two differ wherever a joint has a quaternion.
    """
    var njoint = dims.get_njoint()
    for j in range(njoint):
        var jnt_type = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_TYPE]))
        var padr = Int(
            rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_QPOS_ADR])
        )
        var vadr = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_DOF_ADR]))

        if jnt_type == JNT_FREE:
            for i in range(3):
                qpos[env, padr + i] = (
                    rebind[Scalar[DTYPE]](qpos[env, padr + i])
                    + dt * rebind[Scalar[DTYPE]](dq[env, vadr + i])
                )
            # ⚠ FALLTHROUGH in MuJoCo — the rotation is the BALL update on the
            # shifted addresses, not a free-joint-specific formula.
            padr += 3
            vadr += 3

        if jnt_type == JNT_FREE or jnt_type == JNT_BALL:
            # FREE/BALL qpos holds the quaternion w FIRST, unlike the
            # [x, y, z, w] convention `quat_math` takes and returns.
            var qw = rebind[Scalar[DTYPE]](qpos[env, padr + 0])
            var qx = rebind[Scalar[DTYPE]](qpos[env, padr + 1])
            var qy = rebind[Scalar[DTYPE]](qpos[env, padr + 2])
            var qz = rebind[Scalar[DTYPE]](qpos[env, padr + 3])
            var r = quat_integrate(
                qx,
                qy,
                qz,
                qw,
                rebind[Scalar[DTYPE]](dq[env, vadr + 0]),
                rebind[Scalar[DTYPE]](dq[env, vadr + 1]),
                rebind[Scalar[DTYPE]](dq[env, vadr + 2]),
                dt,
            )
            qpos[env, padr + 0] = r[3]
            qpos[env, padr + 1] = r[0]
            qpos[env, padr + 2] = r[1]
            qpos[env, padr + 3] = r[2]

        elif jnt_type == JNT_HINGE or jnt_type == JNT_SLIDE:
            # One scalar, same for a rotation and a translation.
            qpos[env, padr] = (
                rebind[Scalar[DTYPE]](qpos[env, padr])
                + dt * rebind[Scalar[DTYPE]](dq[env, vadr])
            )
