"""MuJoCo-compatible body rotation matrices (`xmat`) from `Data.xquat`.

MuJoCo exposes each body's world orientation twice: as `data.xquat` (a
quaternion) and as `data.xmat` (the same rotation as a flattened row-major
3x3). Our `Data` stores only the quaternion, but a lot of task code — most of
the dm_control suite, which reads `named.data.xmat['torso', 'zz']` for upright
/ orientation terms — is written against the matrix form.

Rather than materialize an `xmat` tensor (another NBODY*9 field to allocate,
upload, and thread through every GPU hook signature), this derives the
element on demand. Each accessor is a handful of flops on values already in
registers, and reward hooks typically want one or two specific entries.

Column naming matches MuJoCo's `named` indexer: 'xx' 'xy' 'xz' 'yx' 'yy' 'yz'
'zx' 'zy' 'zz', row-major — so `XMAT_ZZ` is index 8, `XMAT_XZ` is index 2.

QUATERNION ORDER: `Data.xquat` is [x, y, z, w] (see `forward_kinematics`),
NOT MuJoCo's [w, x, y, z]. The accessors below take the components by name to
keep that from biting at the call site.
"""

from layout import TileTensor, TensorLayout

from ..fields import Data, Dims, DimsLike


# Row-major indices into a flattened 3x3, matching MuJoCo's named columns.
comptime XMAT_XX: Int = 0
comptime XMAT_XY: Int = 1
comptime XMAT_XZ: Int = 2
comptime XMAT_YX: Int = 3
comptime XMAT_YY: Int = 4
comptime XMAT_YZ: Int = 5
comptime XMAT_ZX: Int = 6
comptime XMAT_ZY: Int = 7
comptime XMAT_ZZ: Int = 8


@always_inline
def quat_xmat_elem[
    DTYPE: DType = DType.float64
](
    qx: Scalar[DTYPE],
    qy: Scalar[DTYPE],
    qz: Scalar[DTYPE],
    qw: Scalar[DTYPE],
    idx: Int,
) -> Scalar[DTYPE]:
    """One element of the row-major rotation matrix for quaternion (x,y,z,w).

    `idx` is an `XMAT_*` constant. Standard quaternion-to-matrix identities;
    gated against MuJoCo's own `data.xmat` in
    `tests/dm_control/test_xmat_vs_mujoco.mojo`.

    `DTYPE` is INFERRED from the components and defaults to float64, so the CPU
    call sites are unchanged. ⚠ It exists because **Metal has no `double`**: a
    GPU hook that called the float64 form would make its enclosing kernel
    return a double and Metal rejects the module outright — the same failure
    documented at `physics3d/solver/newton_solve.mojo:1046`. GPU callers go
    through `xmat_elem_gpu` below, which instantiates this at the env dtype.
    """
    comptime ONE = Scalar[DTYPE](1.0)
    comptime TWO = Scalar[DTYPE](2.0)
    if idx == XMAT_XX:
        return ONE - TWO * (qy * qy + qz * qz)
    if idx == XMAT_XY:
        return TWO * (qx * qy - qw * qz)
    if idx == XMAT_XZ:
        return TWO * (qx * qz + qw * qy)
    if idx == XMAT_YX:
        return TWO * (qx * qy + qw * qz)
    if idx == XMAT_YY:
        return ONE - TWO * (qx * qx + qz * qz)
    if idx == XMAT_YZ:
        return TWO * (qy * qz - qw * qx)
    if idx == XMAT_ZX:
        return TWO * (qx * qz - qw * qy)
    if idx == XMAT_ZY:
        return TWO * (qy * qz + qw * qx)
    return ONE - TWO * (qx * qx + qy * qy)  # XMAT_ZZ


@always_inline
def xmat_elem[DTYPE: DType, D: DimsLike](
    d: Data[DTYPE, D, 1],
    body: Int,
    idx: Int,
) -> Float64:
    """`data.xmat[body, idx]` for the single-env (BATCH=1) CPU path.

    Equivalent to MuJoCo's `named.data.xmat[body_name][idx]`.
    """
    var qx = Float64(d.xquat.data[body * 4 + 0])
    var qy = Float64(d.xquat.data[body * 4 + 1])
    var qz = Float64(d.xquat.data[body * 4 + 2])
    var qw = Float64(d.xquat.data[body * 4 + 3])
    return quat_xmat_elem(qx, qy, qz, qw, idx)


@always_inline
def xmat_elem_gpu[
    DTYPE: DType,
    L_XQUAT: TensorLayout,
](
    xquat: TileTensor[DTYPE, L_XQUAT, MutAnyOrigin],
    env: Int,
    body: Int,
    idx: Int,
) -> Scalar[DTYPE]:
    """`data.xmat[body, idx]` for one lane of the GPU-batched field tensors.

    The batched counterpart of `xmat_elem`, which is pinned to `BATCH=1` `Data`
    and float64 and so cannot be called from a kernel. Same identities, same
    `[x, y, z, w]` component order, all arithmetic in `DTYPE`.
    """
    var qx = rebind[Scalar[DTYPE]](xquat[env, body * 4 + 0])
    var qy = rebind[Scalar[DTYPE]](xquat[env, body * 4 + 1])
    var qz = rebind[Scalar[DTYPE]](xquat[env, body * 4 + 2])
    var qw = rebind[Scalar[DTYPE]](xquat[env, body * 4 + 3])
    return quat_xmat_elem[DTYPE](qx, qy, qz, qw, idx)
