"""MuJoCo-compatible geom world orientation (`geom_xquat` / `geom_xmat`).

The geom twin of `xmat.mojo`. MuJoCo materializes `data.geom_xmat` (NGEOM x 9)
during `mj_kinematics`; our `Data` stores neither that nor `data.xmat`, for the
reason both files give — it is another per-geom field to allocate, upload and
thread through every FK kernel variant, for a quantity task code reads one
entry of. So it is derived on demand from the body pose FK already produced:

    geom world quat = body xquat (x) geom local quat

Used by dm_control's `fish`, whose `mouth_to_target` expresses a world vector
in the MOUTH GEOM's frame (`v.dot(geom_xmat['mouth'].reshape(3, 3))`, i.e.
R^T v) rather than in a body frame.

⚠ THE GEOM'S LOCAL QUATERNION IS OFTEN NOT WRITTEN IN THE MJCF. A `fromto`
capsule/cylinder/ellipsoid has its frame DERIVED by the compiler — z aligned
to the segment — and fish's `mouth` is exactly that
(`fromto="0 .079 0 0 .07 0"` compiles to `quat = (.7071, -.7071, 0, 0)`).
So this accessor is only as correct as the parser's fromto->quat convention;
`test_fish_vs_dm_control` pins our `GEOM_IDX_QUAT_*` against MuJoCo's
`geom_quat` for every geom rather than trusting it.

QUATERNION ORDER: `Data.xquat` and the packed geom record are both
[x, y, z, w], NOT MuJoCo's [w, x, y, z].
"""

from std.collections import Array
from layout import TileTensor, TensorLayout

from ..fields import Data, Dims, DimsLike
from ..gpu.constants import (
    MODEL_GEOM_SIZE,
    GEOM_IDX_BODY,
    GEOM_IDX_QUAT_X,
    GEOM_IDX_QUAT_Y,
    GEOM_IDX_QUAT_Z,
    GEOM_IDX_QUAT_W,
)
from .quat_math import gpu_quat_mul
from .xmat import quat_xmat_elem


@always_inline
def geom_xquat[DTYPE: DType, D: DimsLike](
    d: Data[DTYPE, D, 1],
    m_geoms: List[Scalar[DTYPE]],
    geom: Int,
) -> Tuple[Float64, Float64, Float64, Float64]:
    """The geom's world orientation as (x, y, z, w).

    `m_geoms` is the flat geom record slab the CPU config hooks are handed.
    World-attached geoms (`body == 0`) carry their local quaternion directly,
    since the worldbody frame is the identity — the same shortcut
    `geom_xpos` takes.
    """
    var base = geom * MODEL_GEOM_SIZE
    var gx = Float64(m_geoms[base + GEOM_IDX_QUAT_X])
    var gy = Float64(m_geoms[base + GEOM_IDX_QUAT_Y])
    var gz = Float64(m_geoms[base + GEOM_IDX_QUAT_Z])
    var gw = Float64(m_geoms[base + GEOM_IDX_QUAT_W])

    var body = Int(m_geoms[base + GEOM_IDX_BODY])
    if body == 0:
        return (gx, gy, gz, gw)

    var bx = Float64(d.xquat.data[body * 4 + 0])
    var by = Float64(d.xquat.data[body * 4 + 1])
    var bz = Float64(d.xquat.data[body * 4 + 2])
    var bw = Float64(d.xquat.data[body * 4 + 3])
    var q = gpu_quat_mul(bx, by, bz, bw, gx, gy, gz, gw)
    return (q[0], q[1], q[2], q[3])


@always_inline
def geom_xmat_elem[DTYPE: DType, D: DimsLike](
    d: Data[DTYPE, D, 1],
    m_geoms: List[Scalar[DTYPE]],
    geom: Int,
    idx: Int,
) -> Float64:
    """`data.geom_xmat[geom, idx]`; `idx` is an `XMAT_*` constant."""
    var q = geom_xquat(d, m_geoms, geom)
    return quat_xmat_elem(q[0], q[1], q[2], q[3], idx)


@always_inline
def geom_xquat_gpu[
    DTYPE: DType,
    L_XQUAT: TensorLayout,
    L_GEOMS: TensorLayout,
](
    xquat: TileTensor[DTYPE, L_XQUAT, MutAnyOrigin],
    geoms: TileTensor[DTYPE, L_GEOMS, MutAnyOrigin],
    env: Int,
    geom: Int,
) -> Array[Scalar[DTYPE], 4]:
    """`geom_xquat` for one lane of the batched path, as (x, y, z, w).

    DERIVED, not stored — same call as the CPU form, and for the same reason
    `geom_xpos_gpu` is derived: `Data` carries no `geom_xmat`, and this is one
    quaternion multiply over FK output the batched path already has.

    All arithmetic in `DTYPE`. The CPU twin widens to Float64; Metal rejects a
    kernel containing `double`, so the two agree to float32 rounding rather
    than bitwise.

    World-attached geoms (`body == 0`) carry their local quaternion directly —
    the worldbody frame is the identity. ⚠ That branch is RUNTIME here, not
    comptime: `geom` is an ordinary argument so a caller can loop over geoms.
    """
    var gx = rebind[Scalar[DTYPE]](geoms[geom, GEOM_IDX_QUAT_X])
    var gy = rebind[Scalar[DTYPE]](geoms[geom, GEOM_IDX_QUAT_Y])
    var gz = rebind[Scalar[DTYPE]](geoms[geom, GEOM_IDX_QUAT_Z])
    var gw = rebind[Scalar[DTYPE]](geoms[geom, GEOM_IDX_QUAT_W])

    var body = Int(rebind[Scalar[DTYPE]](geoms[geom, GEOM_IDX_BODY]))
    var out = Array[Scalar[DTYPE], 4](fill=Scalar[DTYPE](0))
    if body == 0:
        out[0] = gx
        out[1] = gy
        out[2] = gz
        out[3] = gw
        return out^

    var q = gpu_quat_mul[DTYPE](
        rebind[Scalar[DTYPE]](xquat[env, body * 4 + 0]),
        rebind[Scalar[DTYPE]](xquat[env, body * 4 + 1]),
        rebind[Scalar[DTYPE]](xquat[env, body * 4 + 2]),
        rebind[Scalar[DTYPE]](xquat[env, body * 4 + 3]),
        gx, gy, gz, gw,
    )
    out[0] = q[0]
    out[1] = q[1]
    out[2] = q[2]
    out[3] = q[3]
    return out^
