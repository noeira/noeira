"""World orientation of a site — MuJoCo's `d->site_xmat`, composed on demand.

MuJoCo materialises `site_xmat` in `mjData` alongside `site_xpos`. We do NOT,
and that is a deliberate scope call rather than an omission: the world site
frame is `xquat[site_body] * site_quat`, one quaternion multiply, and the three
consumers that need it (`sensors/touch.mojo`, `sensors/frame_vel.mojo`,
`sensors/site_acc.mojo`) all already have the body quaternion in hand. Storing
it would add a `[BATCH, NSITE*9]` tensor to `Data`, a write to every forward
kinematics path (serial, multithreaded, and two GPU kernels), and an operand to
the kernels that bind `Data` — for a quantity nothing reads inside the
dynamics.

Until 2026-08-01 those three consumers substituted the BODY quaternion for the
site's own and said so in their docstrings, because every site in every ported
model was either a sphere (orientation-free) or axis-aligned in its body frame.
manipulator ended that: `thumb_touch` and `finger_touch` are BOX zones carrying
`euler="0 15 0"` from `class="hand"`, so the substitution is wrong by 15
degrees on the two zones that decide whether a grasp registers.

The site record's quaternion is (x, y, z, w), matching `BODY_IDX_QUAT_*`.
"""

from std.collections import Array
from layout import TileTensor, TensorLayout

from ..gpu.constants import (
    MODEL_SITE_SIZE,
    SITE_IDX_BODY,
    SITE_IDX_QUAT_X,
    SITE_IDX_QUAT_Y,
    SITE_IDX_QUAT_Z,
    SITE_IDX_QUAT_W,
)
from .quat_math import gpu_quat_mul


@always_inline
def site_world_quat[
    DTYPE: DType,
    L_SITES: TensorLayout,
    L_XQUAT: TensorLayout,
](
    env: Int,
    site_idx: Int,
    sites: TileTensor[DTYPE, L_SITES, MutAnyOrigin],
    xquat: TileTensor[DTYPE, L_XQUAT, MutAnyOrigin],
) -> Array[Scalar[DTYPE], 4]:
    """`xquat[site_body] * site_quat` as (x, y, z, w) — MuJoCo's `site_xmat`.

    Both operands are unit quaternions, so the product is too; no
    renormalisation.
    """
    var s_body = Int(rebind[Scalar[DTYPE]](sites[site_idx, SITE_IDX_BODY]))
    return gpu_quat_mul[DTYPE](
        rebind[Scalar[DTYPE]](xquat[env, s_body * 4 + 0]),
        rebind[Scalar[DTYPE]](xquat[env, s_body * 4 + 1]),
        rebind[Scalar[DTYPE]](xquat[env, s_body * 4 + 2]),
        rebind[Scalar[DTYPE]](xquat[env, s_body * 4 + 3]),
        rebind[Scalar[DTYPE]](sites[site_idx, SITE_IDX_QUAT_X]),
        rebind[Scalar[DTYPE]](sites[site_idx, SITE_IDX_QUAT_Y]),
        rebind[Scalar[DTYPE]](sites[site_idx, SITE_IDX_QUAT_Z]),
        rebind[Scalar[DTYPE]](sites[site_idx, SITE_IDX_QUAT_W]),
    )


@always_inline
def site_world_quat_list[
    DTYPE: DType
](
    m_sites: List[Scalar[DTYPE]],
    xquat: List[Scalar[DTYPE]],
    body: Int,
    site: Int,
) -> Tuple[Float64, Float64, Float64, Float64]:
    """`site_world_quat` over the host `.data` buffers, as (x, y, z, w).

    The sensor modules take `List` buffers rather than LayoutTensors, so they
    need this rather than the kernel-facing form above. Same composition:
    `xquat[body] * site_quat`.
    """
    var sb = site * MODEL_SITE_SIZE
    var bx = Float64(xquat[body * 4 + 0])
    var by = Float64(xquat[body * 4 + 1])
    var bz = Float64(xquat[body * 4 + 2])
    var bw = Float64(xquat[body * 4 + 3])
    var sx = Float64(m_sites[sb + SITE_IDX_QUAT_X])
    var sy = Float64(m_sites[sb + SITE_IDX_QUAT_Y])
    var sz = Float64(m_sites[sb + SITE_IDX_QUAT_Z])
    var sw = Float64(m_sites[sb + SITE_IDX_QUAT_W])
    return (
        bw * sx + bx * sw + by * sz - bz * sy,
        bw * sy - bx * sz + by * sw + bz * sx,
        bw * sz + bx * sy - by * sx + bz * sw,
        bw * sw - bx * sx - by * sy - bz * sz,
    )
