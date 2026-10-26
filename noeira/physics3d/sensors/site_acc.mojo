"""Acceleration-stage site sensors — MuJoCo `accelerometer`, `force`, `torque`.

All three read the `mj_rnePostConstraint` products (`dynamics/rne_post.mojo`)
and transport them from the subtree CoM of the body's kinematic root to the
site, then rotate into the site frame — `mju_transformSpatial`:

    flg_force = 0 (motion):  ang' = ang            lin' = lin - dif x ang
    flg_force = 1 (force):   trq' = trq - dif x f  f'   = f
    dif = site_xpos - subtree_com[rootid]

    accelerometer -> objectAcceleration(site).lin, with the
                     `vel_ang x vel_lin` correction MuJoCo adds afterwards
                     (engine_core_util.c:872) — the site is a POINT on a
                     rotating body, so its linear acceleration is not the
                     transported spatial acceleration alone.
    force         -> transformSpatial(cfrc_int[body], flg_force=1)[3:6]
    torque        -> same tmp, [0:3]

Sign convention for force/torque: `cfrc_int` is the interaction force
between the body and its PARENT. For a toe standing on the floor this comes
out NEGATIVE in z (the leg pulls down on the shin), which is what MuJoCo
reports too — dm_control's quadruped feeds it through `arcsinh` and lets the
policy sort it out.

THE SITE FRAME is `xquat[body] * site_quat`, composed by
`kinematics/site_frame.site_world_quat_list`. `Data` deliberately stores no
`site_xmat` — it is one quaternion multiply and every consumer already holds
the body quaternion, so materialising a `[BATCH, NSITE*9]` tensor and writing
it in four forward-kinematics paths would buy nothing the dynamics reads. See
that module for the reasoning.

⚠ This USED TO substitute the body's quaternion for the site's, which is exact
only for an identity-oriented site. Every model that reached here was of that
form, so the substitution was invisible; manipulator's rotated box touch zones
are what forced the site quaternion into the model record, and this was fixed
with it. Fixing it moved NO existing gate, which is the evidence that the old
scope really was as narrow as it claimed.
"""

from std.collections import Array
from layout import TileTensor, TensorLayout

from ..kinematics.site_frame import site_world_quat, site_world_quat_list
from ..kinematics.quat_math import quat_rotate_inverse
from ..gpu.constants import (
    MODEL_BODY_SIZE,
    MODEL_SITE_SIZE,
    BODY_IDX_ROOTID,
)


@always_inline
def _site_transport(
    v0: Float64, v1: Float64, v2: Float64,
    v3: Float64, v4: Float64, v5: Float64,
    dx: Float64, dy: Float64, dz: Float64,
    flg_force: Bool,
) -> Tuple[Float64, Float64, Float64, Float64, Float64, Float64]:
    """`mju_transformSpatial` translation half (no rotation), `d = new - old`.
    Returns the packed 6-vector in MuJoCo order (rotational first)."""
    if flg_force:
        return (
            v0 - (dy * v5 - dz * v4),
            v1 - (dz * v3 - dx * v5),
            v2 - (dx * v4 - dy * v3),
            v3,
            v4,
            v5,
        )
    return (
        v0,
        v1,
        v2,
        v3 - (dy * v2 - dz * v1),
        v4 - (dz * v0 - dx * v2),
        v5 - (dx * v1 - dy * v0),
    )


def site_accelerometer[
    DTYPE: DType
](
    cvel: List[Scalar[DTYPE]],
    cacc: List[Scalar[DTYPE]],
    subtree_com: List[Scalar[DTYPE]],
    site_xpos: List[Scalar[DTYPE]],
    xquat: List[Scalar[DTYPE]],
    m_bodies: List[Scalar[DTYPE]],
    m_sites: List[Scalar[DTYPE]],
    body: Int,
    site: Int,
) raises -> Tuple[Float64, Float64, Float64]:
    """`sensordata` for `<accelerometer site=...>` — linear acceleration of
    the site point, in the site frame.

    All lists are the single-env `Data` / `Model` host buffers (`.data`).
    Requires the integrator to have run with `RNE_POST=True`.
    """
    var root = Int(m_bodies[body * MODEL_BODY_SIZE + BODY_IDX_ROOTID])
    var dx = Float64(site_xpos[site * 3 + 0]) - Float64(
        subtree_com[root * 3 + 0]
    )
    var dy = Float64(site_xpos[site * 3 + 1]) - Float64(
        subtree_com[root * 3 + 1]
    )
    var dz = Float64(site_xpos[site * 3 + 2]) - Float64(
        subtree_com[root * 3 + 2]
    )

    var v = _site_transport(
        Float64(cvel[body * 6 + 0]), Float64(cvel[body * 6 + 1]),
        Float64(cvel[body * 6 + 2]), Float64(cvel[body * 6 + 3]),
        Float64(cvel[body * 6 + 4]), Float64(cvel[body * 6 + 5]),
        dx, dy, dz, False,
    )
    var a = _site_transport(
        Float64(cacc[body * 6 + 0]), Float64(cacc[body * 6 + 1]),
        Float64(cacc[body * 6 + 2]), Float64(cacc[body * 6 + 3]),
        Float64(cacc[body * 6 + 4]), Float64(cacc[body * 6 + 5]),
        dx, dy, dz, False,
    )

    # The SITE's world frame, not the body's — `xquat[body] * site_quat`.
    var sq = site_world_quat_list[DTYPE](m_sites, xquat, body, site)
    var qx = Scalar[DTYPE](sq[0])
    var qy = Scalar[DTYPE](sq[1])
    var qz = Scalar[DTYPE](sq[2])
    var qw = Scalar[DTYPE](sq[3])

    # Rotate into the site frame BEFORE the correction — MuJoCo builds both
    # `vel` and `res` in the local frame and only then adds vel_ang x vel_lin.
    var vl = quat_rotate_inverse[DType.float64](
        Float64(qx), Float64(qy), Float64(qz), Float64(qw), v[0], v[1], v[2]
    )
    var vv = quat_rotate_inverse[DType.float64](
        Float64(qx), Float64(qy), Float64(qz), Float64(qw), v[3], v[4], v[5]
    )
    var al = quat_rotate_inverse[DType.float64](
        Float64(qx), Float64(qy), Float64(qz), Float64(qw), a[3], a[4], a[5]
    )

    return (
        al[0] + (vl[1] * vv[2] - vl[2] * vv[1]),
        al[1] + (vl[2] * vv[0] - vl[0] * vv[2]),
        al[2] + (vl[0] * vv[1] - vl[1] * vv[0]),
    )


def site_force_torque[
    DTYPE: DType
](
    cfrc_int: List[Scalar[DTYPE]],
    subtree_com: List[Scalar[DTYPE]],
    site_xpos: List[Scalar[DTYPE]],
    xquat: List[Scalar[DTYPE]],
    m_bodies: List[Scalar[DTYPE]],
    m_sites: List[Scalar[DTYPE]],
    body: Int,
    site: Int,
) raises -> Tuple[Float64, Float64, Float64, Float64, Float64, Float64]:
    """`sensordata` for `<force site=...>` then `<torque site=...>` on the
    same site — the body/parent interaction wrench in the site frame.

    Returns force first, then torque (the opposite of MuJoCo's packed order,
    which puts the rotational half first; callers here want the force far
    more often).
    """
    var root = Int(m_bodies[body * MODEL_BODY_SIZE + BODY_IDX_ROOTID])
    var dx = Float64(site_xpos[site * 3 + 0]) - Float64(
        subtree_com[root * 3 + 0]
    )
    var dy = Float64(site_xpos[site * 3 + 1]) - Float64(
        subtree_com[root * 3 + 1]
    )
    var dz = Float64(site_xpos[site * 3 + 2]) - Float64(
        subtree_com[root * 3 + 2]
    )

    var t = _site_transport(
        Float64(cfrc_int[body * 6 + 0]), Float64(cfrc_int[body * 6 + 1]),
        Float64(cfrc_int[body * 6 + 2]), Float64(cfrc_int[body * 6 + 3]),
        Float64(cfrc_int[body * 6 + 4]), Float64(cfrc_int[body * 6 + 5]),
        dx, dy, dz, True,
    )

    # The SITE's world frame, not the body's — `xquat[body] * site_quat`.
    var sq = site_world_quat_list[DTYPE](m_sites, xquat, body, site)
    var qx = sq[0]
    var qy = sq[1]
    var qz = sq[2]
    var qw = sq[3]

    var trq = quat_rotate_inverse[DType.float64](
        qx, qy, qz, qw, t[0], t[1], t[2]
    )
    var frc = quat_rotate_inverse[DType.float64](
        qx, qy, qz, qw, t[3], t[4], t[5]
    )

    return (frc[0], frc[1], frc[2], trq[0], trq[1], trq[2])


# ── GPU forms ────────────────────────────────────────────────────────────
#
# ⚠ EVERYTHING BELOW COMPUTES IN `DTYPE`, NOT Float64. The CPU functions above
# widen to Float64 internally; Metal REJECTS a kernel module containing
# `double` outright, so the GPU twins cannot. The two therefore agree to
# float32 rounding rather than bitwise, which is what the GPU-vs-CPU gates'
# `atol + rtol*|cpu|` bound is sized for.
#
# ⚠ AND THEY MUST BE FED THE `*_acc` SNAPSHOTS. `site_xpos` / `xquat` here are
# the pose AS IT STOOD WHEN `cacc`/`cfrc_int` WERE WRITTEN — `Data.site_xpos_acc`
# and `Data.xquat_acc`, not the live FK products. Passing the live ones mixes
# integration stages and is defect 19: dog's accelerometer read 1.484 against
# dm_control's -6.386 that way, with `cacc` itself exact to 4.5e-10. The
# parameters are named `site_xpos` / `xquat` to mirror the CPU signature, which
# is exactly why this warning is here.


@always_inline
def _site_transport_gpu[
    DTYPE: DType
](
    v0: Scalar[DTYPE], v1: Scalar[DTYPE], v2: Scalar[DTYPE],
    v3: Scalar[DTYPE], v4: Scalar[DTYPE], v5: Scalar[DTYPE],
    dx: Scalar[DTYPE], dy: Scalar[DTYPE], dz: Scalar[DTYPE],
    flg_force: Bool,
) -> Array[Scalar[DTYPE], 6]:
    """`_site_transport` in DTYPE. Same expressions, same association."""
    var o = Array[Scalar[DTYPE], 6](fill=Scalar[DTYPE](0))
    if flg_force:
        o[0] = v0 - (dy * v5 - dz * v4)
        o[1] = v1 - (dz * v3 - dx * v5)
        o[2] = v2 - (dx * v4 - dy * v3)
        o[3] = v3
        o[4] = v4
        o[5] = v5
    else:
        o[0] = v0
        o[1] = v1
        o[2] = v2
        o[3] = v3 - (dy * v2 - dz * v1)
        o[4] = v4 - (dz * v0 - dx * v2)
        o[5] = v5 - (dx * v1 - dy * v0)
    return o^


@always_inline
def _site_com_offset_gpu[
    DTYPE: DType,
    L_SITE_XPOS: TensorLayout,
    L_SUBTREE_COM: TensorLayout,
    L_BODIES: TensorLayout](
    site_xpos: TileTensor[DTYPE, L_SITE_XPOS, MutAnyOrigin],
    subtree_com: TileTensor[DTYPE, L_SUBTREE_COM, MutAnyOrigin],
    bodies: TileTensor[DTYPE, L_BODIES, MutAnyOrigin],
    env: Int,
    body: Int,
    site: Int,
) -> Array[Scalar[DTYPE], 3]:
    """`site_xpos[site] - subtree_com[rootid[body]]` — the transport vector
    both acceleration-stage sensors need."""
    var root = Int(rebind[Scalar[DTYPE]](bodies[body, BODY_IDX_ROOTID]))
    var d = Array[Scalar[DTYPE], 3](fill=Scalar[DTYPE](0))
    for k in range(3):
        d[k] = rebind[Scalar[DTYPE]](site_xpos[env, site * 3 + k]) - rebind[
            Scalar[DTYPE]
        ](subtree_com[env, root * 3 + k])
    return d^


@always_inline
def site_accelerometer_gpu[
    DTYPE: DType,
    L_CVEL: TensorLayout,
    L_SUBTREE_COM: TensorLayout,
    L_SITE_XPOS: TensorLayout,
    L_XQUAT: TensorLayout,
    L_BODIES: TensorLayout,
    L_SITES: TensorLayout,
](
    cvel: TileTensor[DTYPE, L_CVEL, MutAnyOrigin],
    cacc: TileTensor[DTYPE, L_CVEL, MutAnyOrigin],
    subtree_com: TileTensor[DTYPE, L_SUBTREE_COM, MutAnyOrigin],
    site_xpos: TileTensor[DTYPE, L_SITE_XPOS, MutAnyOrigin],
    xquat: TileTensor[DTYPE, L_XQUAT, MutAnyOrigin],
    bodies: TileTensor[DTYPE, L_BODIES, MutAnyOrigin],
    sites: TileTensor[DTYPE, L_SITES, MutAnyOrigin],
    env: Int,
    body: Int,
    site: Int,
) -> Array[Scalar[DTYPE], 3]:
    """`site_accelerometer` against the batched field tensors.

    Pass `site_xpos_acc` / `xquat_acc`, not the live products — see the
    section note above. Requires the integrator to have run with
    `RNE_POST=True`; without it `cacc` is all zeros and this returns the
    `vel_ang x vel_lin` correction alone, which is finite, plausible and
    wrong. `Phyics3dBatchedEnv.__init__` asserts the Euler pairing that
    `RNE_POST` needs, but nothing can assert that a config asked for it.
    """
    var d = _site_com_offset_gpu[DTYPE](
        site_xpos, subtree_com, bodies, env, body, site
    )

    var v = _site_transport_gpu[DTYPE](
        rebind[Scalar[DTYPE]](cvel[env, body * 6 + 0]),
        rebind[Scalar[DTYPE]](cvel[env, body * 6 + 1]),
        rebind[Scalar[DTYPE]](cvel[env, body * 6 + 2]),
        rebind[Scalar[DTYPE]](cvel[env, body * 6 + 3]),
        rebind[Scalar[DTYPE]](cvel[env, body * 6 + 4]),
        rebind[Scalar[DTYPE]](cvel[env, body * 6 + 5]),
        d[0], d[1], d[2], False,
    )
    var a = _site_transport_gpu[DTYPE](
        rebind[Scalar[DTYPE]](cacc[env, body * 6 + 0]),
        rebind[Scalar[DTYPE]](cacc[env, body * 6 + 1]),
        rebind[Scalar[DTYPE]](cacc[env, body * 6 + 2]),
        rebind[Scalar[DTYPE]](cacc[env, body * 6 + 3]),
        rebind[Scalar[DTYPE]](cacc[env, body * 6 + 4]),
        rebind[Scalar[DTYPE]](cacc[env, body * 6 + 5]),
        d[0], d[1], d[2], False,
    )

    var sq = site_world_quat[DTYPE](
        env, site, sites, xquat
    )
    # Rotate into the site frame BEFORE the correction — MuJoCo builds both
    # `vel` and `res` locally and only then adds vel_ang x vel_lin.
    var vl = quat_rotate_inverse[DTYPE](
        sq[0], sq[1], sq[2], sq[3], v[0], v[1], v[2]
    )
    var vv = quat_rotate_inverse[DTYPE](
        sq[0], sq[1], sq[2], sq[3], v[3], v[4], v[5]
    )
    var al = quat_rotate_inverse[DTYPE](
        sq[0], sq[1], sq[2], sq[3], a[3], a[4], a[5]
    )

    var out = Array[Scalar[DTYPE], 3](fill=Scalar[DTYPE](0))
    out[0] = al[0] + (vl[1] * vv[2] - vl[2] * vv[1])
    out[1] = al[1] + (vl[2] * vv[0] - vl[0] * vv[2])
    out[2] = al[2] + (vl[0] * vv[1] - vl[1] * vv[0])
    return out^


@always_inline
def site_force_torque_gpu[
    DTYPE: DType,
    L_CFRC_INT: TensorLayout,
    L_SUBTREE_COM: TensorLayout,
    L_SITE_XPOS: TensorLayout,
    L_XQUAT: TensorLayout,
    L_BODIES: TensorLayout,
    L_SITES: TensorLayout,
](
    cfrc_int: TileTensor[DTYPE, L_CFRC_INT, MutAnyOrigin],
    subtree_com: TileTensor[DTYPE, L_SUBTREE_COM, MutAnyOrigin],
    site_xpos: TileTensor[DTYPE, L_SITE_XPOS, MutAnyOrigin],
    xquat: TileTensor[DTYPE, L_XQUAT, MutAnyOrigin],
    bodies: TileTensor[DTYPE, L_BODIES, MutAnyOrigin],
    sites: TileTensor[DTYPE, L_SITES, MutAnyOrigin],
    env: Int,
    body: Int,
    site: Int,
) -> Array[Scalar[DTYPE], 6]:
    """`site_force_torque` against the batched field tensors.

    Returns `[force(3), torque(3)]` — force first, matching the CPU twin
    rather than MuJoCo's packed order. Pass `site_xpos_acc` / `xquat_acc`.
    """
    var d = _site_com_offset_gpu[DTYPE](
        site_xpos, subtree_com, bodies, env, body, site
    )

    var t = _site_transport_gpu[DTYPE](
        rebind[Scalar[DTYPE]](cfrc_int[env, body * 6 + 0]),
        rebind[Scalar[DTYPE]](cfrc_int[env, body * 6 + 1]),
        rebind[Scalar[DTYPE]](cfrc_int[env, body * 6 + 2]),
        rebind[Scalar[DTYPE]](cfrc_int[env, body * 6 + 3]),
        rebind[Scalar[DTYPE]](cfrc_int[env, body * 6 + 4]),
        rebind[Scalar[DTYPE]](cfrc_int[env, body * 6 + 5]),
        d[0], d[1], d[2], True,
    )

    var sq = site_world_quat[DTYPE](
        env, site, sites, xquat
    )
    var trq = quat_rotate_inverse[DTYPE](
        sq[0], sq[1], sq[2], sq[3], t[0], t[1], t[2]
    )
    var frc = quat_rotate_inverse[DTYPE](
        sq[0], sq[1], sq[2], sq[3], t[3], t[4], t[5]
    )

    var out = Array[Scalar[DTYPE], 6](fill=Scalar[DTYPE](0))
    out[0] = frc[0]
    out[1] = frc[1]
    out[2] = frc[2]
    out[3] = trq[0]
    out[4] = trq[1]
    out[5] = trq[2]
    return out^
