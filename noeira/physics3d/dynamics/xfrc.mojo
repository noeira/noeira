"""`mj_xfrcAccumulate` — a per-lane external wrench on a body, into the smooth force.

WHAT IT IS. MuJoCo's `xfrc_applied`: for each body a force and a torque, both
in the WORLD frame, the force acting at the body's centre of mass.
`engine_support.c:497`:

    for (i = 1; i < nbody; i++)
      if (!mju_isZero(xfrc + 6*i, 6))
        mj_applyFT(m, d, xfrc + 6*i, xfrc + 6*i + 3, d->xipos + 3*i, i, qfrc);

called from `mj_fwdAcceleration` on `qfrc_smooth`, after the passive, bias,
applied and actuator terms (`engine_forward.c:1028`). Here it is
`Data.xfrc_applied`, `[BATCH, NBODY * 6]`, one record per LANE and body: the
`Model` is shared by the batch, this is state, so each lane carries its own
wrench. Body b of lane e sits at `(e * NBODY + b) * 6`: force x, y, z, then
torque x, y, z.

A FORCE AT ANOTHER POINT OF THE BODY is the caller's to convert, as with
MuJoCo: force `f` at the world point `p` is the record `(f, (p - xipos) x f)`.
`xfrc_at_point` below does that on the host.

WHO READS IT. Only an integrator built with `XFRC=True` (the Euler one; every
other build compiles exactly as before and never binds the buffer). The caller
owns the contents: nothing here or in a reset clears them, a wrench stays until
it is overwritten. `mj_resetData` zeroes MuJoCo's; a caller who wants that
writes the zeros.

TRANSCRIPTION NOTES

1. THE JACOBIAN IS `gravcomp.mojo`'s, WHICH IS MuJoCo'S, TERM FOR TERM:
   `jacp[k][i] = cdof[3+k] + (cdof[0:3] x offset)[k]` with
   `offset = xipos[b] - subtree_com[body_rootid[b]]`, and `jacr[k][i] =
   cdof[k]`. The force's product is added first, then the torque's, as
   `mj_applyFT` does.

2. A BODY WHOSE SIX ENTRIES ARE ALL ZERO IS SKIPPED, as in MuJoCo, so a build
   with `XFRC=True` and an all-zero buffer gives the build without it bit for
   bit (adding `0 * j` to a `-0.0` would not).

3. THE GROUPING IS THE PASSIVE SEAM'S, not MuJoCo's: each body is added
   straight into `scratch.fnet`, which already holds `qfrc - bias - passive`
   (and fluid, and gravcomp). Same terms, another association: 1e-16 to
   1e-13 from MuJoCo in float64, not 0.0 (`gravcomp.mojo`, same note).

WHAT IS NOT PORTED. `mj_rnePostConstraint` starts `cfrc_ext` from
`xfrc_applied` (`engine_core_smooth.c`); `rne_post.mojo` does not read this
field, so `cacc`, `cfrc_int`, `cfrc_ext` and the sensors on them would miss
the wrench. The Euler integrator therefore refuses `XFRC` with `RNE_POST` at
compile time.

CALL SITE. The Euler integrator's passive seam, after
`compute_gravcomp_forces`, before the LDL solve that gives `qacc_smooth`.
Operands (7): xfrc_applied, xipos, subtree_com (data) + bodies, joints (model)
+ cdof, fnet (scratch).
"""

from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceContext
from layout import TileTensor, TensorLayout
from noeira.core.tiles import RowMajor, Tile

from ..joint_types import JNT_FREE, JNT_BALL
from ..fields import (
    Data,
    Model,
    DynamicsScratch,
    Dims,
    DimsLike,
    DYN2,
    rl2,
)
from ..gpu.constants import (
    MODEL_BODY_SIZE,
    MODEL_JOINT_SIZE,
    BODY_IDX_PARENT,
    BODY_IDX_ROOTID,
    JOINT_IDX_TYPE,
    JOINT_IDX_BODY_ID,
    JOINT_IDX_DOF_ADR,
)

comptime XFRC_TPB: Int = 64
comptime XFRC_STRIDE: Int = 6


def _xfrc_forces_env[
    DTYPE: DType,
    D: DimsLike,
    L_X6: TensorLayout,
    L_B3: TensorLayout,
    L_BODIES: TensorLayout,
    L_JOINTS: TensorLayout,
    L_CDOF: TensorLayout,
    L_FNET: TensorLayout,
](
    env: Int,
    dims: D,
    xfrc: TileTensor[DTYPE, L_X6, MutAnyOrigin],
    xipos: TileTensor[DTYPE, L_B3, MutAnyOrigin],
    subtree_com: TileTensor[DTYPE, L_B3, MutAnyOrigin],
    bodies: TileTensor[DTYPE, L_BODIES, MutAnyOrigin],
    joints: TileTensor[DTYPE, L_JOINTS, MutAnyOrigin],
    cdof: TileTensor[DTYPE, L_CDOF, MutAnyOrigin],
    fnet: TileTensor[DTYPE, L_FNET, MutAnyOrigin],
):
    """`mj_xfrcAccumulate` for one env, accumulated into `fnet`."""
    var nbody = dims.get_nbody()
    var njoint = dims.get_njoint()

    for b in range(1, nbody):
        var fx = rebind[Scalar[DTYPE]](xfrc[env, b * XFRC_STRIDE + 0])
        var fy = rebind[Scalar[DTYPE]](xfrc[env, b * XFRC_STRIDE + 1])
        var fz = rebind[Scalar[DTYPE]](xfrc[env, b * XFRC_STRIDE + 2])
        var tx = rebind[Scalar[DTYPE]](xfrc[env, b * XFRC_STRIDE + 3])
        var ty = rebind[Scalar[DTYPE]](xfrc[env, b * XFRC_STRIDE + 4])
        var tz = rebind[Scalar[DTYPE]](xfrc[env, b * XFRC_STRIDE + 5])
        # `if (!mju_isZero(xfrc + 6*i, 6))`. Note 2 in the header.
        if fx == 0 and fy == 0 and fz == 0 and tx == 0 and ty == 0 and tz == 0:
            continue

        # `offset = point - subtree_com[body_rootid[body]]`, point == xipos[b]
        var root = Int(rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_ROOTID]))
        var ox = (
            rebind[Scalar[DTYPE]](xipos[env, b * 3 + 0])
            - rebind[Scalar[DTYPE]](subtree_com[env, root * 3 + 0])
        )
        var oy = (
            rebind[Scalar[DTYPE]](xipos[env, b * 3 + 1])
            - rebind[Scalar[DTYPE]](subtree_com[env, root * 3 + 1])
        )
        var oz = (
            rebind[Scalar[DTYPE]](xipos[env, b * 3 + 2])
            - rebind[Scalar[DTYPE]](subtree_com[env, root * 3 + 2])
        )

        # This body and its ancestors: the dof set of MuJoCo's chain from
        # `body_weldid[b]` (`gravcomp.mojo`, `jac_point.mojo`).
        var body = b
        while body > 0:
            for j in range(njoint):
                if Int(
                    rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_BODY_ID])
                ) != body:
                    continue
                var dof_adr = Int(
                    rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_DOF_ADR])
                )
                var jtype = Int(
                    rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_TYPE])
                )
                var ndof = 1
                if jtype == JNT_FREE:
                    ndof = 6
                elif jtype == JNT_BALL:
                    ndof = 3

                for k in range(ndof):
                    var di = dof_adr + k
                    # cdof per dof: [ang(3), lin(3)] — MuJoCo's cdof[0:6].
                    var ca0 = rebind[Scalar[DTYPE]](cdof[env, di * 6 + 0])
                    var ca1 = rebind[Scalar[DTYPE]](cdof[env, di * 6 + 1])
                    var ca2 = rebind[Scalar[DTYPE]](cdof[env, di * 6 + 2])
                    var cl0 = rebind[Scalar[DTYPE]](cdof[env, di * 6 + 3])
                    var cl1 = rebind[Scalar[DTYPE]](cdof[env, di * 6 + 4])
                    var cl2 = rebind[Scalar[DTYPE]](cdof[env, di * 6 + 5])

                    # `jacp[k][i] = cdof[3+k] + (cdof[0:3] x offset)[k]`
                    var jp0 = cl0 + (ca1 * oz - ca2 * oy)
                    var jp1 = cl1 + (ca2 * ox - ca0 * oz)
                    var jp2 = cl2 + (ca0 * oy - ca1 * ox)

                    # `mj_applyFT`: the force's `J^T f` is added, then the
                    # torque's, whose Jacobian is `jacr[k][i] = cdof[k]`.
                    var acc = (
                        rebind[Scalar[DTYPE]](fnet[env, di])
                        + (jp0 * fx + jp1 * fy + jp2 * fz)
                    )
                    fnet[env, di] = acc + (ca0 * tx + ca1 * ty + ca2 * tz)

            body = Int(rebind[Scalar[DTYPE]](bodies[body, BODY_IDX_PARENT]))


def _xfrc_forces_kernel[
    DTYPE: DType,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    BATCH: Int,
](
    xfrc: Tile[DTYPE, BATCH, NBODY * XFRC_STRIDE],
    xipos: Tile[DTYPE, BATCH, NBODY * 3],
    subtree_com: Tile[DTYPE, BATCH, NBODY * 3],
    bodies: Tile[DTYPE, NBODY, MODEL_BODY_SIZE],
    joints: Tile[DTYPE, NJOINT, MODEL_JOINT_SIZE],
    cdof: Tile[DTYPE, BATCH, NV * 6],
    fnet: Tile[DTYPE, BATCH, NV],
):
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    _xfrc_forces_env[DTYPE](
        env,
        Dims[nv=NV, nbody=NBODY, njoint=NJOINT](),
        xfrc,
        xipos,
        subtree_com,
        bodies,
        joints,
        cdof,
        fnet,
    )


def compute_xfrc_forces[
    target: StaticString,
    DTYPE: DType,
    D: DimsLike,
    BATCH: Int = 1,
](
    mut d: Data[DTYPE, D, BATCH],
    mut m: Model[DTYPE, D],
    mut scratch: DynamicsScratch[DTYPE, D, BATCH],
    ctx: Optional[DeviceContext] = None,
) raises:
    """Accumulate `d.xfrc_applied` into `scratch.fnet`, both targets.

    Call in the passive seam after `compute_gravcomp_forces` (the kinematics,
    `subtree_com` and `cdof` of this step are then in place) and before the
    solve for `qacc_smooth`.
    """
    comptime if target == "cpu":
        var dm = d.dims
        var rl_X6 = rl2(BATCH, dm.get_nbody() * XFRC_STRIDE)
        var rl_B3 = rl2(BATCH, dm.get_nbody() * 3)
        var rl_BODY = rl2(dm.get_nbody(), MODEL_BODY_SIZE)
        var rl_JOINT = rl2(dm.get_njoint(), MODEL_JOINT_SIZE)
        var rl_CDOF = rl2(BATCH, dm.get_nv() * 6)
        var rl_NV = rl2(BATCH, dm.get_nv())
        var xfrc_v = d.xfrc_applied.tt_dyn["cpu", DYN2](rl_X6)
        var xipos_v = d.xipos.tt_dyn["cpu", DYN2](rl_B3)
        var stcom_v = d.subtree_com.tt_dyn["cpu", DYN2](rl_B3)
        var bodies_v = m.bodies.tt_dyn["cpu", DYN2](rl_BODY)
        var joints_v = m.joints.tt_dyn["cpu", DYN2](rl_JOINT)
        var cdof_v = scratch.cdof.tt_dyn["cpu", DYN2](rl_CDOF)
        var fnet_v = scratch.fnet.tt_dyn["cpu", DYN2](rl_NV)
        for e in range(BATCH):
            _xfrc_forces_env[DTYPE](
                e, dm, xfrc_v, xipos_v, stcom_v, bodies_v, joints_v, cdof_v,
                fnet_v,
            )
    else:
        var c = ctx.value()
        comptime BLOCKS = (BATCH + XFRC_TPB - 1) // XFRC_TPB
        c.enqueue_function[
            _xfrc_forces_kernel[DTYPE, D.NV, D.NBODY, D.NJOINT, BATCH]
        ](
            d.xfrc_applied.tt["gpu", BATCH, D.NBODY * XFRC_STRIDE](),
            d.xipos.tt["gpu", BATCH, D.NBODY * 3](),
            d.subtree_com.tt["gpu", BATCH, D.NBODY * 3](),
            m.bodies.tt["gpu", D.NBODY, MODEL_BODY_SIZE](),
            m.joints.tt["gpu", D.NJOINT, MODEL_JOINT_SIZE](),
            scratch.cdof.tt["gpu", BATCH, D.NV * 6](),
            scratch.fnet.tt["gpu", BATCH, D.NV](),
            grid_dim=(BLOCKS,),
            block_dim=(XFRC_TPB,),
        )


def xfrc_at_point[DTYPE: DType, D: DimsLike, BATCH: Int](
    mut d: Data[DTYPE, D, BATCH],
    env: Int,
    body: Int,
    fx: Float64, fy: Float64, fz: Float64,
    px: Float64, py: Float64, pz: Float64,
):
    """Host: lane `env`, body `body` takes the world force `f` acting at the
    world point `p`, written as MuJoCo's record `(f, (p - xipos) x f)` over
    whatever the record held. Reads the HOST `d.xipos` (current kinematics:
    after a CPU step or FK, or a download); a device run then uploads
    `d.xfrc_applied`."""
    var nb = d.dims.get_nbody()
    var rx = px - Float64(d.xipos.data[(env * nb + body) * 3 + 0])
    var ry = py - Float64(d.xipos.data[(env * nb + body) * 3 + 1])
    var rz = pz - Float64(d.xipos.data[(env * nb + body) * 3 + 2])
    var o = (env * nb + body) * XFRC_STRIDE
    d.xfrc_applied.data[o + 0] = Scalar[DTYPE](fx)
    d.xfrc_applied.data[o + 1] = Scalar[DTYPE](fy)
    d.xfrc_applied.data[o + 2] = Scalar[DTYPE](fz)
    d.xfrc_applied.data[o + 3] = Scalar[DTYPE](ry * fz - rz * fy)
    d.xfrc_applied.data[o + 4] = Scalar[DTYPE](rz * fx - rx * fz)
    d.xfrc_applied.data[o + 5] = Scalar[DTYPE](rx * fy - ry * fx)
