"""`mj_gravcomp` — per-body gravity compensation, into the passive seam.

WHAT IT IS. `<body gravcomp="F">` holds a body up against `F` times its own
weight. `engine_passive.c:817`:

    force = gravity * -(body_mass[i] * body_gravcomp[i])       (mji_scl3)
    mj_applyFT(m, d, force, /*torque=*/0, d->xipos + 3*i, i, d->qfrc_gravcomp)

and `mj_passive` then folds `qfrc_gravcomp` into `qfrc_passive` for every dof
whose joint does NOT set `actuatorgravcomp` (`engine_passive.c:1012-1022`).

⚠⚠ WHY THIS MATTERS MORE THAN ITS ONE-LINE FORCE LAW SUGGESTS. Eight models in
`mujoco_menagerie-main` declare it — flexiv_rizon4, agilex_piper, i2rt_yam,
arx_l5, google_robot, shadow_dexee and both hello_robot_stretch generations —
and on every one of them it is the ENTIRE contents of `qfrc_passive`:
`|qfrc_gravcomp|max == |qfrc_passive|max` to the last digit, because those
models carry no springs and no joint damping. Until 2026-08-21 this engine
computed none of it, so on stretch_3 a **46.9 N·m** hold-up force was simply
missing and every gravcomp link sagged under its own weight from step one.
Those eight scenes were 8 of the 16 that had not reached 1e-9, and they
included the first, second and fourth entries on the board.

⚠ IT IS NOT AN ACTUATOR AND IT IS NOT A SPRING. It reads as one — an arm that
holds its pose is what a well-tuned position servo looks like — which is
exactly why the absence was invisible for so long: `|d qfrc_actuator|` was
0.000e+00 on all of them (the actuators were right), `nefc` matched, the mass
matrix matched, and the residual still would not go away. **The tell was
`|qfrc_passive|max` in `tri.py`'s triage row**, a column that had been printed
for weeks next to models we had no passive forces for at all.

TRANSCRIPTION NOTES

1. THE JACOBIAN IS MuJoCo'S FORM, TERM FOR TERM. `mj_jacSparse`
   (`engine_core_util.c:359-365`) builds

       offset = point - subtree_com[body_rootid[body]]
       tmp    = cdof[0:3] x offset
       jacp[k][i] = cdof[3+k] + tmp[k]

   and `mju_mulMatTVec` then forms `qforce[i] = sum_k jacp[k][i] * force[k]`.
   ⚠ The algebraically equal form — transport the wrench to the com reference
   first, then dot with `(cdof_lin, cdof_ang)` — is what `fluid_forces.mojo`
   next door does, and it is NOT the same in floating point. Written the
   reference's way here because this term lands on models we want at 1e-16.

2. THE COUNT IS `> 0`, NOT `!= 0`. `engine_setconst.c:102` counts
   `body_gravcomp[i] > 0` strictly and `mj_gravcomp` early-outs on
   `!m->ngravcomp`, so a negative `gravcomp` (which the compiler does not
   reject) disables the whole pass rather than pushing the body down. Ported
   with the same comparison so a nonsense model is nonsense identically.

3. GRAVITY DISABLED IS ALREADY ZERO GRAVITY HERE. MuJoCo tests
   `mjDISABLED(mjDSBL_GRAVITY) || mju_norm3(m->opt.gravity) == 0`;
   `full_parser` zeroes the vector when `<flag gravity="disable"/>` is set, so
   the norm test alone covers both.

⚠ WHAT IS DELIBERATELY NOT PORTED: `jnt_actgravcomp`. That flag moves a
joint's share out of `qfrc_passive` and into `qfrc_actuator`, where
`jnt_actfrcrange` can clamp it. Nothing in this tree sets it; the parser
counts the declarations and prints, so a model that does will say so instead
of silently getting the unclamped answer. See
`FlatModelDef.act_gravcomp_joints`.

⚠ GROUPING, AND WHY IT IS NOT BIT-EXACT. MuJoCo sums every body into its own
`qfrc_gravcomp` vector and adds that to `qfrc_passive` once; we add each body
straight into `scratch.fnet`, which already holds
`qfrc - bias - damping - stiffness - frictionloss`. Same terms, different
association, so the models here land at 1e-16..1e-13 rather than at 0.0. That
is a pre-existing property of the whole passive seam — `_fnet_passive_kernel`
never had a separate `qfrc_passive` to group into — and not something this
module introduced.

CALL SITE. The passive seam of all three integrators, immediately AFTER
`compute_fluid_forces`, because `mj_passive` adds fluid before gravcomp.
Operands (8): xipos, subtree_com (data) + bodies, joints, meta (model)
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
    DYN1,
    DYN2,
    rl1,
    rl2,
)
from ..gpu.constants import (
    MODEL_BODY_SIZE,
    MODEL_JOINT_SIZE,
    MODEL_META_SIZE,
    MODEL_META_IDX_NGRAVCOMP,
    MODEL_META_IDX_GRAVITY_X,
    MODEL_META_IDX_GRAVITY_Y,
    MODEL_META_IDX_GRAVITY_Z,
    BODY_IDX_MASS,
    BODY_IDX_GRAVCOMP,
    BODY_IDX_PARENT,
    BODY_IDX_ROOTID,
    JOINT_IDX_TYPE,
    JOINT_IDX_BODY_ID,
    JOINT_IDX_DOF_ADR,
)

comptime GRAVCOMP_TPB: Int = 64


def _gravcomp_forces_env[
    DTYPE: DType,
    D: DimsLike,
    L_B3: TensorLayout,
    L_BODIES: TensorLayout,
    L_JOINTS: TensorLayout,
    L_MMETA: TensorLayout,
    L_CDOF: TensorLayout,
    L_FNET: TensorLayout,
](
    env: Int,
    dims: D,
    xipos: TileTensor[DTYPE, L_B3, MutAnyOrigin],
    subtree_com: TileTensor[DTYPE, L_B3, MutAnyOrigin],
    bodies: TileTensor[DTYPE, L_BODIES, MutAnyOrigin],
    joints: TileTensor[DTYPE, L_JOINTS, MutAnyOrigin],
    mmeta: TileTensor[DTYPE, L_MMETA, MutAnyOrigin],
    cdof: TileTensor[DTYPE, L_CDOF, MutAnyOrigin],
    fnet: TileTensor[DTYPE, L_FNET, MutAnyOrigin],
):
    """`mj_gravcomp` for one env, accumulated into `fnet`."""
    # ── the two early-outs, in MuJoCo's order ────────────────────────────
    # `if (!m->ngravcomp || mjDISABLED(mjDSBL_GRAVITY) || norm3(gravity) == 0)`
    if mmeta[MODEL_META_IDX_NGRAVCOMP] <= Scalar[DTYPE](0):
        return
    var gx = rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_GRAVITY_X])
    var gy = rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_GRAVITY_Y])
    var gz = rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_GRAVITY_Z])
    if gx == 0 and gy == 0 and gz == 0:
        return

    var nbody = dims.get_nbody()
    var njoint = dims.get_njoint()

    for b in range(1, nbody):
        var gc = rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_GRAVCOMP])
        # `if (m->body_gravcomp[i])` — a plain truth test, so a body carrying
        # a negative value IS compensated (downwards) once `ngravcomp` has
        # let the pass run at all. Note 2 in the header.
        if gc == 0:
            continue
        var mass = rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_MASS])

        # `mji_scl3(force, m->opt.gravity, -(body_mass * body_gravcomp))`
        var s = -(mass * gc)
        var fx = gx * s
        var fy = gy * s
        var fz = gz * s

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

        # Walk this body and its ancestors, exactly as `fluid_forces` does.
        # ⚠ That walk visits the SAME dof set as MuJoCo's `dof_parentid`
        # chain from `body_weldid[b]`'s last dof — see `jac_point.mojo`'s
        # header, which checked it rather than assuming it. A body with no
        # jointed ancestor contributes nothing here and MuJoCo returns a zero
        # Jacobian for it, which is the same answer.
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

                    # `mji_cross(tmp, cdof, offset)` then
                    # `jacp[k][i] = cdof[3+k] + tmp[k]`.
                    var jp0 = cl0 + (ca1 * oz - ca2 * oy)
                    var jp1 = cl1 + (ca2 * ox - ca0 * oz)
                    var jp2 = cl2 + (ca0 * oy - ca1 * ox)

                    # `mju_mulMatTVec(qforce, jacp, force, 3, NV)`.
                    fnet[env, di] = (
                        rebind[Scalar[DTYPE]](fnet[env, di])
                        + jp0 * fx
                        + jp1 * fy
                        + jp2 * fz
                    )

            body = Int(rebind[Scalar[DTYPE]](bodies[body, BODY_IDX_PARENT]))


def _gravcomp_forces_kernel[
    DTYPE: DType,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    BATCH: Int,
](
    xipos: Tile[DTYPE, BATCH, NBODY * 3],
    subtree_com: Tile[DTYPE, BATCH, NBODY * 3],
    bodies: Tile[DTYPE, NBODY, MODEL_BODY_SIZE],
    joints: Tile[DTYPE, NJOINT, MODEL_JOINT_SIZE],
    mmeta: Tile[DTYPE, MODEL_META_SIZE],
    cdof: Tile[DTYPE, BATCH, NV * 6],
    fnet: Tile[DTYPE, BATCH, NV],
):
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    _gravcomp_forces_env[DTYPE](
        env,
        Dims[nv=NV, nbody=NBODY, njoint=NJOINT](),
        xipos,
        subtree_com,
        bodies,
        joints,
        mmeta,
        cdof,
        fnet,
    )


def compute_gravcomp_forces[
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
    """Accumulate `mj_gravcomp` into `scratch.fnet`, both targets.

    No-op when the model has no `<body gravcomp>` or gravity is zero (both
    early-outs live inside the env helper, so the GPU path pays one launch and
    nothing else). Call in the passive seam AFTER `compute_fluid_forces` —
    `mj_passive` adds fluid before gravcomp.
    """
    comptime L_NV = RowMajor[BATCH, D.NV]
    comptime L_B3 = RowMajor[BATCH, D.NBODY * 3]
    comptime L_BODY = RowMajor[D.NBODY, MODEL_BODY_SIZE]
    comptime L_JOINT = RowMajor[D.NJOINT, MODEL_JOINT_SIZE]
    comptime L_META = RowMajor[MODEL_META_SIZE]
    comptime L_CDOF = RowMajor[BATCH, D.NV * 6]

    comptime if target == "cpu":
        var dm = d.dims
        var rl_B3 = rl2(BATCH, dm.get_nbody() * 3)
        var rl_BODY = rl2(dm.get_nbody(), MODEL_BODY_SIZE)
        var rl_JOINT = rl2(dm.get_njoint(), MODEL_JOINT_SIZE)
        var rl_META = rl1(MODEL_META_SIZE)
        var rl_CDOF = rl2(BATCH, dm.get_nv() * 6)
        var rl_NV = rl2(BATCH, dm.get_nv())
        var xipos_v = d.xipos.tt_dyn["cpu", DYN2](rl_B3)
        var stcom_v = d.subtree_com.tt_dyn["cpu", DYN2](rl_B3)
        var bodies_v = m.bodies.tt_dyn["cpu", DYN2](rl_BODY)
        var joints_v = m.joints.tt_dyn["cpu", DYN2](rl_JOINT)
        var meta_v = m.meta.tt_dyn["cpu", DYN1](rl_META)
        var cdof_v = scratch.cdof.tt_dyn["cpu", DYN2](rl_CDOF)
        var fnet_v = scratch.fnet.tt_dyn["cpu", DYN2](rl_NV)
        for e in range(BATCH):
            _gravcomp_forces_env[DTYPE](
                e, dm, xipos_v, stcom_v, bodies_v, joints_v, meta_v, cdof_v,
                fnet_v,
            )
    else:
        var c = ctx.value()
        comptime BLOCKS = (BATCH + GRAVCOMP_TPB - 1) // GRAVCOMP_TPB
        c.enqueue_function[
            _gravcomp_forces_kernel[
                DTYPE, D.NV, D.NBODY, D.NJOINT, BATCH
            ]
        ](
            d.xipos.tt["gpu", BATCH, D.NBODY * 3](),
            d.subtree_com.tt["gpu", BATCH, D.NBODY * 3](),
            m.bodies.tt["gpu", D.NBODY, MODEL_BODY_SIZE](),
            m.joints.tt["gpu", D.NJOINT, MODEL_JOINT_SIZE](),
            m.meta.tt["gpu", MODEL_META_SIZE](),
            scratch.cdof.tt["gpu", BATCH, D.NV * 6](),
            scratch.fnet.tt["gpu", BATCH, D.NV](),
            grid_dim=(BLOCKS,),
            block_dim=(GRAVCOMP_TPB,),
        )
