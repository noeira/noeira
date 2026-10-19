"""Site-frame velocity sensors — MuJoCo `velocimeter` and `gyro`.

Both are thin wrappers over `mj_objectVelocity(m, d, mjOBJ_SITE, id, res, 1)`
(engine_support.c), which transports the body's spatial velocity to the site
point and rotates it into the site frame:

    mju_transformSpatial(res, d->cvel + 6*bodyid, 0,
                         d->site_xpos + 3*siteid,          <- to here
                         d->subtree_com + 3*body_rootid,   <- from here
                         d->site_xmat + 9*siteid)          <- into this frame

    mjSENS_VELOCIMETER -> res[3:6]   (linear)
    mjSENS_GYRO        -> res[0:3]   (angular)

Our `Data` carries the same content in a different parameterisation:
`xangvel[b]` is the body's world angular velocity, and `xvel[b]` is its world
linear velocity AT `xipos[b]` (the body CoM — see `_vel_body`, which propagates
`v = v_parent + w_parent x (xipos - xipos_parent)`). So the transport is a
single cross product and the rotation is one inverse quaternion rotation:

    v_site_world = xvel[b] + xangvel[b] x (site_xpos - xipos[b])
    velocimeter  = R_site^T v_site_world
    gyro         = R_site^T xangvel[b]

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

Used by dm_control's swimmer, whose `body_velocities()` observation is the
`[vx, vy, wz]` slice of one velocimeter/gyro pair per link.
"""

from std.collections import Array
from layout import TileTensor, TensorLayout

from ..kinematics.quat_math import quat_rotate_inverse
from ..kinematics.site_frame import site_world_quat, site_world_quat_list
from ..gpu.constants import MODEL_SITE_SIZE


@always_inline
def point_velocity_world[
    DTYPE: DType
](
    xvel: List[Scalar[DTYPE]],
    xangvel: List[Scalar[DTYPE]],
    xipos: List[Scalar[DTYPE]],
    body: Int,
    px: Scalar[DTYPE],
    py: Scalar[DTYPE],
    pz: Scalar[DTYPE],
) -> Tuple[Scalar[DTYPE], Scalar[DTYPE], Scalar[DTYPE]]:
    """World-frame linear velocity of a point rigidly attached to `body`.

    `mj_objectVelocity(..., flg_local=0)`'s linear half, for ANY point — a
    site, a geom, a body landmark. `xvel[b]` is the body's world linear
    velocity at its CoM (`xipos[b]`), so the point velocity is that plus
    `omega x (p - xipos)`.

    ⚠ THE ARITHMETIC IS `site_frame_velocity`'s, MOVED HERE UNCHANGED — same
    expressions, same order, same association. That function's velocimeter is
    gated exact on dog (2.66e-15 across the observation) and on swimmer, so a
    reassociation here would be a silent regression in a passing gate. It is
    `@always_inline`, so this is a factoring, not a call.

    Extracted for dog `fetch`, which needs the same transport in WORLD frame
    for a GEOM (the ball) and a SITE (the head) — `ball_in_head_frame`
    subtracts the two. Writing it a second time is how a quantity ends up
    computed five slightly different ways; today's tendon cap was exactly that.
    """
    var wx = xangvel[body * 3 + 0]
    var wy = xangvel[body * 3 + 1]
    var wz = xangvel[body * 3 + 2]

    var rx = px - xipos[body * 3 + 0]
    var ry = py - xipos[body * 3 + 1]
    var rz = pz - xipos[body * 3 + 2]

    return (
        xvel[body * 3 + 0] + (wy * rz - wz * ry),
        xvel[body * 3 + 1] + (wz * rx - wx * rz),
        xvel[body * 3 + 2] + (wx * ry - wy * rx),
    )


def site_frame_velocity[
    DTYPE: DType
](
    xvel: List[Scalar[DTYPE]],
    xangvel: List[Scalar[DTYPE]],
    xipos: List[Scalar[DTYPE]],
    xquat: List[Scalar[DTYPE]],
    site_xpos: List[Scalar[DTYPE]],
    m_sites: List[Scalar[DTYPE]],
    body: Int,
    site: Int,
) raises -> Tuple[Float64, Float64, Float64, Float64, Float64, Float64]:
    """`(velocimeter[0:3], gyro[0:3])` for `site` mounted on `body`.

    All six lists are the corresponding `Data` host buffers (`.data`), single
    env. Returns linear first, then angular — the opposite of MuJoCo's packed
    `res` ordering, because callers here read the linear part far more often.
    """
    var wx = xangvel[body * 3 + 0]
    var wy = xangvel[body * 3 + 1]
    var wz = xangvel[body * 3 + 2]

    # Transport the CoM velocity to the site point. Shared with `fetch`'s
    # geom/site velocities — see `point_velocity_world` for why it is one
    # function and why its arithmetic must not be rearranged.
    var v = point_velocity_world[DTYPE](
        xvel,
        xangvel,
        xipos,
        body,
        site_xpos[site * 3 + 0],
        site_xpos[site * 3 + 1],
        site_xpos[site * 3 + 2],
    )
    var vx = v[0]
    var vy = v[1]
    var vz = v[2]

    # R_site is the SITE's world frame, `xquat[body] * site_quat` — not the
    # body's. Those coincide only for a site with no orientation attribute,
    # which was true of every model that reached here before manipulator.
    var sq = site_world_quat_list[DTYPE](m_sites, xquat, body, site)
    var qx = Scalar[DTYPE](sq[0])
    var qy = Scalar[DTYPE](sq[1])
    var qz = Scalar[DTYPE](sq[2])
    var qw = Scalar[DTYPE](sq[3])

    var vl = quat_rotate_inverse[DTYPE](qx, qy, qz, qw, vx, vy, vz)
    var wl = quat_rotate_inverse[DTYPE](qx, qy, qz, qw, wx, wy, wz)

    return (
        Float64(vl[0]),
        Float64(vl[1]),
        Float64(vl[2]),
        Float64(wl[0]),
        Float64(wl[1]),
        Float64(wl[2]),
    )


@always_inline
def site_frame_velocity_gpu[
    DTYPE: DType,
    L_XVEL: TensorLayout,
    L_XQUAT: TensorLayout,
    L_SITE_XPOS: TensorLayout,
    L_SITES: TensorLayout,
](
    xvel: TileTensor[DTYPE, L_XVEL, MutAnyOrigin],
    xangvel: TileTensor[DTYPE, L_XVEL, MutAnyOrigin],
    xipos: TileTensor[DTYPE, L_XVEL, MutAnyOrigin],
    xquat: TileTensor[DTYPE, L_XQUAT, MutAnyOrigin],
    site_xpos: TileTensor[DTYPE, L_SITE_XPOS, MutAnyOrigin],
    sites: TileTensor[DTYPE, L_SITES, MutAnyOrigin],
    env: Int,
    body: Int,
    site: Int,
) -> Array[Scalar[DTYPE], 6]:
    """`site_frame_velocity` against the batched field tensors.

    Returns `[velocimeter(3), gyro(3)]` — linear first, matching the CPU
    twin's tuple order rather than MuJoCo's packed `res` (rotational first).

    ⚠ THE ARITHMETIC IS THE CPU FUNCTION'S, TRANSCRIBED EXPRESSION FOR
    EXPRESSION — same cross products, same association, same order. The two
    are diffed element-wise by the GPU-vs-CPU gates, so a reassociation here
    reads as a physics divergence. It is a transcription rather than a call
    because the CPU form takes `List` host buffers, which no kernel has.

    Everything is computed in DTYPE (float32 in production). The CPU twin
    widens to Float64 internally, so the two agree to float32 rounding, not
    bitwise — which is exactly what the gates' `atol + rtol*|cpu|` bound is
    sized for. Metal rejects a kernel containing `double` outright, so
    widening here is not an option (`feedback_metal_nested_generics`).
    """
    var wx = rebind[Scalar[DTYPE]](xangvel[env, body * 3 + 0])
    var wy = rebind[Scalar[DTYPE]](xangvel[env, body * 3 + 1])
    var wz = rebind[Scalar[DTYPE]](xangvel[env, body * 3 + 2])

    # Transport the CoM velocity to the site point: v + w x (p - xipos).
    var rx = rebind[Scalar[DTYPE]](site_xpos[env, site * 3 + 0]) - rebind[
        Scalar[DTYPE]
    ](xipos[env, body * 3 + 0])
    var ry = rebind[Scalar[DTYPE]](site_xpos[env, site * 3 + 1]) - rebind[
        Scalar[DTYPE]
    ](xipos[env, body * 3 + 1])
    var rz = rebind[Scalar[DTYPE]](site_xpos[env, site * 3 + 2]) - rebind[
        Scalar[DTYPE]
    ](xipos[env, body * 3 + 2])

    var vx = rebind[Scalar[DTYPE]](xvel[env, body * 3 + 0]) + (
        wy * rz - wz * ry
    )
    var vy = rebind[Scalar[DTYPE]](xvel[env, body * 3 + 1]) + (
        wz * rx - wx * rz
    )
    var vz = rebind[Scalar[DTYPE]](xvel[env, body * 3 + 2]) + (
        wx * ry - wy * rx
    )

    # R_site is the SITE's world frame, `xquat[body] * site_quat`.
    var sq = site_world_quat[DTYPE](
        env, site, sites, xquat
    )
    var vl = quat_rotate_inverse[DTYPE](sq[0], sq[1], sq[2], sq[3], vx, vy, vz)
    var wl = quat_rotate_inverse[DTYPE](sq[0], sq[1], sq[2], sq[3], wx, wy, wz)

    var out = Array[Scalar[DTYPE], 6](fill=Scalar[DTYPE](0))
    out[0] = vl[0]
    out[1] = vl[1]
    out[2] = vl[2]
    out[3] = wl[0]
    out[4] = wl[1]
    out[5] = wl[2]
    return out^
