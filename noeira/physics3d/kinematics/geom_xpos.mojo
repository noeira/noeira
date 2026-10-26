"""MuJoCo-compatible geom world positions (`geom_xpos`) from `Data.xpos/xquat`.

MuJoCo materializes `data.geom_xpos` (NGEOM x 3) during `mj_kinematics`. Our
`Data` does not, for the same reason it does not materialize `xmat` (see
`xmat.mojo`): it would be another per-geom field to allocate, upload, and
thread through every FK kernel variant and GPU hook signature, for a quantity
that task code reads one or two entries of.

So it is derived on demand here, from the body pose the FK already produced.
The arithmetic is the same as `collision.contact_detection._geom_world_pose`,
including its `body == 0` shortcut (world-attached geoms carry their local pos
directly, since the worldbody frame is the identity).

Used by the dm_control ports whose reward is a geom-to-geom distance —
`point_mass` reads `named.data.geom_xpos[['target', 'pointmass']]`.
"""

from ..fields import Data, Dims, DimsLike
from ..gpu.constants import (
    MODEL_GEOM_SIZE,
    GEOM_IDX_BODY,
    GEOM_IDX_POS_X,
    GEOM_IDX_POS_Y,
    GEOM_IDX_POS_Z,
)
from layout import TileTensor, TensorLayout

from .quat_math import gpu_quat_rotate


@always_inline
def geom_xpos[DTYPE: DType, D: DimsLike](
    d: Data[DTYPE, D, 1],
    m_geoms: List[Scalar[DTYPE]],
    geom: Int,
) -> Tuple[Float64, Float64, Float64]:
    """`data.geom_xpos[geom]` for the single-env (BATCH=1) CPU path.

    `m_geoms` is the flat geom record slab the CPU config hooks are handed.
    """
    var base = geom * MODEL_GEOM_SIZE
    var body = Int(m_geoms[base + GEOM_IDX_BODY])
    var lx = Float64(m_geoms[base + GEOM_IDX_POS_X])
    var ly = Float64(m_geoms[base + GEOM_IDX_POS_Y])
    var lz = Float64(m_geoms[base + GEOM_IDX_POS_Z])

    # World-attached geoms: the worldbody frame is the identity, so the
    # local offset IS the world position.
    if body == 0:
        return (lx, ly, lz)

    var qx = Float64(d.xquat.data[body * 4 + 0])
    var qy = Float64(d.xquat.data[body * 4 + 1])
    var qz = Float64(d.xquat.data[body * 4 + 2])
    var qw = Float64(d.xquat.data[body * 4 + 3])
    var rot = gpu_quat_rotate(qx, qy, qz, qw, lx, ly, lz)
    return (
        Float64(d.xpos.data[body * 3 + 0]) + rot[0],
        Float64(d.xpos.data[body * 3 + 1]) + rot[1],
        Float64(d.xpos.data[body * 3 + 2]) + rot[2],
    )


# =============================================================================
# GPU-batched counterpart
# =============================================================================


@always_inline
def geom_xpos_gpu[
    DTYPE: DType,
    L_XPOS: TensorLayout,
    L_XQUAT: TensorLayout,
    L_GEOMS: TensorLayout,
](
    xpos: TileTensor[DTYPE, L_XPOS, MutAnyOrigin],
    xquat: TileTensor[DTYPE, L_XQUAT, MutAnyOrigin],
    geoms: TileTensor[DTYPE, L_GEOMS, MutAnyOrigin],
    env: Int,
    geom: Int,
) -> Tuple[Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE]]:
    """`data.geom_xpos[geom]` for one lane of the batched path.

    ⚠ DERIVED, NOT STORED — and that is a correction to the plan. G2 in
    `docs/DM_CONTROL_PORT.md` specified `geom_xpos` as "a new `Data` field
    `[BATCH, NGEOM*3]` + fill in FK, CPU **and** GPU". It does not need to be:
    a geom's world position is its body's `xpos` plus its local offset rotated
    by the body's `xquat`, which are both already hook operands. Deriving it
    costs a quaternion rotate at the two or three call sites a task actually
    has, against an NGEOM*3 tensor allocated, filled and uploaded every step
    for every model. Same argument as `xmat_elem_gpu` vs an `xmat` field.

    World-attached geoms (body 0) short-circuit: the worldbody frame is the
    identity, so the local offset IS the world position. dm_control's targets
    are usually world-attached, so this branch is the common one, not an edge
    case.
    """
    var body = Int(rebind[Scalar[DTYPE]](geoms[geom, GEOM_IDX_BODY]))
    var lx = rebind[Scalar[DTYPE]](geoms[geom, GEOM_IDX_POS_X])
    var ly = rebind[Scalar[DTYPE]](geoms[geom, GEOM_IDX_POS_Y])
    var lz = rebind[Scalar[DTYPE]](geoms[geom, GEOM_IDX_POS_Z])
    if body == 0:
        return (lx, ly, lz)

    var qx = rebind[Scalar[DTYPE]](xquat[env, body * 4 + 0])
    var qy = rebind[Scalar[DTYPE]](xquat[env, body * 4 + 1])
    var qz = rebind[Scalar[DTYPE]](xquat[env, body * 4 + 2])
    var qw = rebind[Scalar[DTYPE]](xquat[env, body * 4 + 3])
    var rot = gpu_quat_rotate[DTYPE](qx, qy, qz, qw, lx, ly, lz)
    return (
        rebind[Scalar[DTYPE]](xpos[env, body * 3 + 0]) + rot[0],
        rebind[Scalar[DTYPE]](xpos[env, body * 3 + 1]) + rot[1],
        rebind[Scalar[DTYPE]](xpos[env, body * 3 + 2]) + rot[2],
    )
