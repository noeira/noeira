"""`mj_rnePostConstraint` over per-field tensors — `cacc` and `cfrc_int`.

MuJoCo runs this once per step, inside `mj_sensorAcc`, whenever the model
declares an acceleration-stage sensor (`engine_sensor.c:908` gates it on
`accelerometer` / `force` / `torque` / `framelinacc` / `frameangacc` /
`subtreeangmom`). It is NOT part of the dynamics: nothing it writes feeds
back into `qacc`. That is why this stage is comptime-gated on the
integrator's `RNE_POST` parameter — only quadruped pays for it.

    cacc[0]      = (0, -gravity)                      (world)
    cacc[b]      = cacc[parent] + cdof_dot*qvel + cdof*qacc
    cfrc_body[b] = cinert[b]*cacc[b] + cvel[b] x* (cinert[b]*cvel[b])
    cfrc_int[b]  = cfrc_body[b] - cfrc_ext[b]
    cfrc_int[parent] += cfrc_int[b]                   (leaves -> root)

Every quantity is world-oriented torque:force, referenced at the subtree
CoM of the body's kinematic root — the same convention as `Data.cfrc_ext`.

WHY THIS IS NOT A COPY OF THE RNE BIAS PASS. It is the same recursion with
one extra term, and the recursion is LINEAR in it, so the `cdof*qacc`
contribution can be accumulated in its own forward sweep afterwards:

    extra[b] = extra[parent] + sum_{dof in b} cdof[dof] * qacc[dof]
    cacc[b] += extra[b]

That lets `_rne_fwd_body` / `_rne_cinert_body` / `_rne_cfrc_body` /
`_rne_backward_env` be reused VERBATIM from `dynamics/rne.mojo` — the bias
pass and this one cannot drift apart, and the hot RNE path is untouched.
`_rne_fwd_body` also refills the `crb` scratch with per-body `cvel`, so
this stage does not depend on what the constraint solvers left behind.

WHICH `qacc`. MuJoCo's `mj_sensorAcc` reads `d->qacc` as written by
`mj_fwdConstraint` — BEFORE the Euler integrator's implicit-damping
re-solve. Ours is `scratch.qacc_constrained`; `Data.qacc` is the damped one
(`euler._finalize_env` overwrites it). Passing `d.qacc` here would be
silently wrong on any model with joint damping — which is every model that
wants these sensors.

EQUALITY CONSTRAINTS ARE NOT MAPPED INTO `cfrc_ext` (AUD-48). MuJoCo walks
the equality rows and adds `connect`/`weld` forces (`mjEQ_JOINT` and
`mjEQ_TENDON` contribute nothing — they only advance the row cursor).
quadruped's four equalities are all `<equality><tendon>`, so the walk is a
no-op for it, which is why this stage could be written against that model and
be exact. A model with `connect`/`weld` equalities plus an acceleration-stage
sensor would read LOW by the whole loop-closure load.

⚠⚠ THIS PARAGRAPH USED TO SAY `compute_rne_post` RAISED ON THAT COMBINATION.
It did not — there was no raise anywhere in this file, and a stale claim in a
docstring is worse than no claim, because it is what a reader checks INSTEAD
of the code. The refusal is real as of 2026-09-13 and lives at LOAD, in
`full_parser._refuse_wrong_physics`, where both halves of the condition are in
hand and it costs nothing per step. What blocks the implementation is that
those rows' `efc_force` is not retained past the solve.

`xfrc_applied` is likewise absent: we have no such field.
"""

from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceContext
from layout import TileTensor, TensorLayout
from noeira.core.tiles import RowMajor, Tile

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
from ..joint_types import JNT_FREE, JNT_BALL
from ..collision.contact_frame import contact_tangent_frame
from .body_joint_map import body_joint_map
from .rne import (
    _max_one,
    _rne_fwd_body,
    _rne_cinert_body,
    _rne_cfrc_body,
    _rne_backward_env,
)
from std.math import nan
from ..constraints.solver_ws import _max_one_rt
from ..types import EQ_CONNECT, EQ_WELD, EQ_JOINT
from ..kinematics.quat_math import gpu_quat_rotate
from ..gpu.constants import (
    CONTACT_SIZE,
    METADATA_SIZE,
    META_IDX_NUM_CONTACTS,
    MODEL_BODY_SIZE,
    MODEL_JOINT_SIZE,
    MODEL_META_SIZE,
    MODEL_META_IDX_GRAVITY_X,
    MODEL_META_IDX_GRAVITY_Y,
    MODEL_META_IDX_GRAVITY_Z,
    BODY_IDX_PARENT,
    BODY_IDX_ROOTID,
    MODEL_META_IDX_NEQUALITY,
    MODEL_EQ_SIZE,
    EQ_IDX_TYPE,
    EQ_IDX_BODY_A,
    EQ_IDX_BODY_B,
    EQ_IDX_ANCHOR_AX,
    EQ_IDX_ANCHOR_AY,
    EQ_IDX_ANCHOR_AZ,
    EQ_IDX_ANCHOR_BX,
    EQ_IDX_ANCHOR_BY,
    EQ_IDX_ANCHOR_BZ,
    META_IDX_EQ_FORCE_LIVE,
    JOINT_IDX_TYPE,
    JOINT_IDX_BODY_ID,
    JOINT_IDX_DOF_ADR,
    CONTACT_IDX_BODY_A,
    CONTACT_IDX_BODY_B,
    CONTACT_IDX_POS_X,
    CONTACT_IDX_POS_Y,
    CONTACT_IDX_POS_Z,
    CONTACT_IDX_NX,
    CONTACT_IDX_NY,
    CONTACT_IDX_NZ,
    CONTACT_IDX_FRAME_T1_X,
    CONTACT_IDX_FRAME_T1_Y,
    CONTACT_IDX_FRAME_T1_Z,
    CONTACT_IDX_FORCE_N,
    CONTACT_IDX_FORCE_T1,
    CONTACT_IDX_FORCE_T2,
    CONTACT_IDX_FORCE_TORSION,
    CONTACT_IDX_FORCE_ROLL1,
    CONTACT_IDX_FORCE_ROLL2,
)

comptime RNE_POST_TPB: Int = 64


@always_inline
def _cfrc_ext_env[
    DTYPE: DType,
    D: DimsLike,
    L_CONTACTS: TensorLayout,
    L_DMETA: TensorLayout,
    L_SUBTREE_COM: TensorLayout,
    L_BODIES: TensorLayout,
    L_CFRC_EXT: TensorLayout,
    L_XQUAT_E: TensorLayout,
    L_EQUALITY: TensorLayout,
    L_EQFORCE: TensorLayout,
    L_MMETA_E: TensorLayout,
](
    env: Int,
    dims: D,
    contacts: TileTensor[DTYPE, L_CONTACTS, MutAnyOrigin],
    dmeta: TileTensor[DTYPE, L_DMETA, MutAnyOrigin],
    subtree_com: TileTensor[DTYPE, L_SUBTREE_COM, MutAnyOrigin],
    bodies: TileTensor[DTYPE, L_BODIES, MutAnyOrigin],
    cfrc_ext: TileTensor[DTYPE, L_CFRC_EXT, MutAnyOrigin],
    xpos: TileTensor[DTYPE, L_SUBTREE_COM, MutAnyOrigin],
    xquat: TileTensor[DTYPE, L_XQUAT_E, MutAnyOrigin],
    equality: TileTensor[DTYPE, L_EQUALITY, MutAnyOrigin],
    eq_force: TileTensor[DTYPE, L_EQFORCE, MutAnyOrigin],
    mmeta: TileTensor[DTYPE, L_MMETA_E, MutAnyOrigin],
):
    """Contact forces accumulated per body at the root's subtree CoM.

    Same arithmetic as `gpu/cfrc_ext_gpu.mojo`, but reading the already
    computed `Data.subtree_com` and the `BODY_IDX_ROOTID` column instead of
    rebuilding both from body masses and parents.

    SIGN. `contacts[BODY_A]` is the body of geom[0] — MuJoCo's "body 1",
    which it SUBTRACTS (`mju_subFrom`), adding to body 2. Our stored contact
    force is the force on A, so A adds and B subtracts. (The one existing
    consumer, Ant's contact_cost, takes a norm and could not have caught a
    flipped sign; the quadruped force-sensor gate can.)
    """
    var nbody = dims.get_nbody()
    var max_contacts = dims.get_max_contacts()
    for i in range(nbody * 6):
        cfrc_ext[env, i] = Scalar[DTYPE](0)

    var ncon = Int(rebind[Scalar[DTYPE]](dmeta[env, META_IDX_NUM_CONTACTS]))

    for ci in range(max_contacts):
        if ci >= ncon:
            break
        var cb = ci * CONTACT_SIZE

        var nx = rebind[Scalar[DTYPE]](contacts[env, cb + CONTACT_IDX_NX])
        var ny = rebind[Scalar[DTYPE]](contacts[env, cb + CONTACT_IDX_NY])
        var nz = rebind[Scalar[DTYPE]](contacts[env, cb + CONTACT_IDX_NZ])
        # FRAME_T1 is a HINT, not a tangent — it has had no fallback, no
        # Gram-Schmidt and no normalization applied, and non-capsule pairs
        # never write it at all. Reading it raw gave the tangential force a
        # garbage direction while the normal component stayed right, because
        # that one only needs `n`. See collision/contact_frame.mojo.
        var frame = contact_tangent_frame[DTYPE](
            nx,
            ny,
            nz,
            rebind[Scalar[DTYPE]](contacts[env, cb + CONTACT_IDX_FRAME_T1_X]),
            rebind[Scalar[DTYPE]](contacts[env, cb + CONTACT_IDX_FRAME_T1_Y]),
            rebind[Scalar[DTYPE]](contacts[env, cb + CONTACT_IDX_FRAME_T1_Z]),
        )
        var t1x = frame[0]
        var t1y = frame[1]
        var t1z = frame[2]
        var t2x = frame[3]
        var t2y = frame[4]
        var t2z = frame[5]

        var f_n = rebind[Scalar[DTYPE]](contacts[env, cb + CONTACT_IDX_FORCE_N])
        var f_t1 = rebind[Scalar[DTYPE]](
            contacts[env, cb + CONTACT_IDX_FORCE_T1]
        )
        var f_t2 = rebind[Scalar[DTYPE]](
            contacts[env, cb + CONTACT_IDX_FORCE_T2]
        )
        var f_tors = rebind[Scalar[DTYPE]](
            contacts[env, cb + CONTACT_IDX_FORCE_TORSION]
        )
        var f_r1 = rebind[Scalar[DTYPE]](
            contacts[env, cb + CONTACT_IDX_FORCE_ROLL1]
        )
        var f_r2 = rebind[Scalar[DTYPE]](
            contacts[env, cb + CONTACT_IDX_FORCE_ROLL2]
        )

        var fw_x = f_n * nx + f_t1 * t1x + f_t2 * t2x
        var fw_y = f_n * ny + f_t1 * t1y + f_t2 * t2y
        var fw_z = f_n * nz + f_t1 * t1z + f_t2 * t2z
        var tw_x = f_tors * nx + f_r1 * t1x + f_r2 * t2x
        var tw_y = f_tors * ny + f_r1 * t1y + f_r2 * t2y
        var tw_z = f_tors * nz + f_r1 * t1z + f_r2 * t2z

        var px = rebind[Scalar[DTYPE]](contacts[env, cb + CONTACT_IDX_POS_X])
        var py = rebind[Scalar[DTYPE]](contacts[env, cb + CONTACT_IDX_POS_Y])
        var pz = rebind[Scalar[DTYPE]](contacts[env, cb + CONTACT_IDX_POS_Z])

        var ka = Int(rebind[Scalar[DTYPE]](contacts[env, cb + CONTACT_IDX_BODY_A]))
        var kb = Int(rebind[Scalar[DTYPE]](contacts[env, cb + CONTACT_IDX_BODY_B]))

        for side in range(2):
            var k = ka if side == 0 else kb
            if k <= 0:
                continue
            var rid = Int(rebind[Scalar[DTYPE]](bodies[k, BODY_IDX_ROOTID]))
            var dx = rebind[Scalar[DTYPE]](subtree_com[env, rid * 3 + 0]) - px
            var dy = rebind[Scalar[DTYPE]](subtree_com[env, rid * 3 + 1]) - py
            var dz = rebind[Scalar[DTYPE]](subtree_com[env, rid * 3 + 2]) - pz
            # transformSpatial(flg_force=1): torque -= (newpos-oldpos) x force
            var mx = tw_x - (dy * fw_z - dz * fw_y)
            var my = tw_y - (dz * fw_x - dx * fw_z)
            var mz = tw_z - (dx * fw_y - dy * fw_x)

            var s = Scalar[DTYPE](1) if side == 0 else Scalar[DTYPE](-1)
            var o = k * 6
            cfrc_ext[env, o + 0] = (
                rebind[Scalar[DTYPE]](cfrc_ext[env, o + 0]) + s * mx
            )
            cfrc_ext[env, o + 1] = (
                rebind[Scalar[DTYPE]](cfrc_ext[env, o + 1]) + s * my
            )
            cfrc_ext[env, o + 2] = (
                rebind[Scalar[DTYPE]](cfrc_ext[env, o + 2]) + s * mz
            )
            cfrc_ext[env, o + 3] = (
                rebind[Scalar[DTYPE]](cfrc_ext[env, o + 3]) + s * fw_x
            )
            cfrc_ext[env, o + 4] = (
                rebind[Scalar[DTYPE]](cfrc_ext[env, o + 4]) + s * fw_y
            )
            cfrc_ext[env, o + 5] = (
                rebind[Scalar[DTYPE]](cfrc_ext[env, o + 5]) + s * fw_z
            )

    # ── cfrc_ext += connect / weld equality forces (AUD-48) ───────────────
    #
    # `mj_rnePostConstraint`'s third block (engine_core_smooth.c:2464-2523).
    # `mjEQ_JOINT` and `mjEQ_TENDON` contribute NOTHING — they only advance
    # the row cursor — which is why quadruped, whose four equalities are all
    # tendons, was exact before this landed and is unchanged by it.
    #
    #   cfrc = (torque, force),  force = efc_force[i..i+3]
    #                            torque = efc_force[i+3..i+6] for a WELD, else 0
    #   body1: pos = xpos[b1] + R(xquat[b1]) * anchor_a ;  cfrc_ext[b1] += T(cfrc)
    #   body2: pos = xpos[b2] + R(xquat[b2]) * anchor_b ;  cfrc_ext[b2] -= T(cfrc)
    #
    # with `T` the force-transform to the body root's subtree CoM — the same
    # `torque -= (newpos - oldpos) x force` the contact block above applies.
    #
    # ⚠ THE ANCHORS ARE ALREADY NORMALISED. MuJoCo swaps which half of
    # `eq_data` belongs to which body between connect and weld, and reduces a
    # site-based equality to `site_xpos`; `_fill_equality` and
    # `compute_invweight0` between them leave `ANCHOR_A` as body1's local
    # anchor and `ANCHOR_B` as body2's for BOTH types and BOTH semantics. So
    # this walk needs neither the swap nor a site branch.
    # ⚠⚠ THE LIVE COUNT, NOT THE TABLE SIZE. `NEQUALITY` is a CAPACITY — a
    # model with one connect routinely declares 3 or 6 so the ROW budget fits
    # — and the unused slots are zero-filled. `EQ_IDX_TYPE == 0` is
    # `EQ_CONNECT`, so a loop bounded by the capacity reads every empty slot
    # as a connect between body 0 and body 0. Measured on this file's own
    # fixture before the fix: one connect, `n_cw = 3`, `want_rows = 9`
    # against the solver's 3, and the walk refused itself.
    # `build_weld_equality_rows` reads the same word for the same reason.
    var nequality = dims.get_nequality()
    var neq_live = Int(rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_NEQUALITY]))
    if neq_live < nequality:
        nequality = neq_live
    if nequality > 0:
        # ⚠⚠ THE CURSOR IS RECOMPUTED HERE AND CHECKED AGAINST THE SOLVER'S
        # OWN COUNT. `build_weld_equality_rows` emits 1 row per joint
        # equality, 3 per connect and 6 per weld — MuJoCo's own advance
        # (`i += type == mjEQ_WELD ? 6 : 3`, with `i++` for joint/tendon) —
        # but it also has malformed-model `continue`s that emit none. If the
        # two disagree the rows are off by an unknown amount and every force
        # after the gap belongs to another constraint, so the walk REFUSES
        # and NaNs the bodies instead. A silent desync here is a force sensor
        # reading another equality's load.
        var want_rows = 0
        var n_cw = 0
        for e in range(nequality):
            var t = Int(rebind[Scalar[DTYPE]](equality[e, EQ_IDX_TYPE]))
            if t == EQ_CONNECT:
                want_rows += 3
                n_cw += 1
            elif t == EQ_WELD:
                want_rows += 6
                n_cw += 1
            elif t == EQ_JOINT:
                want_rows += 1
        if n_cw > 0:
            var live = Int(
                rebind[Scalar[DTYPE]](dmeta[env, META_IDX_EQ_FORCE_LIVE])
            )
            if live != want_rows:
                # The solver that ran this step did not retain the equality
                # forces (PGS, CG, the island path) or emitted a different
                # row set. Poison exactly the bodies whose `cfrc_int` would
                # be short the constraint; `cfrc_int` sums leaves to root, so
                # this reaches every force sensor above them and no other.
                var qnan = nan[DTYPE]()
                for e in range(nequality):
                    var t2 = Int(
                        rebind[Scalar[DTYPE]](equality[e, EQ_IDX_TYPE])
                    )
                    if t2 != EQ_CONNECT and t2 != EQ_WELD:
                        continue
                    for side in range(2):
                        var kk = Int(rebind[Scalar[DTYPE]](
                            equality[e, EQ_IDX_BODY_A if side == 0
                                     else EQ_IDX_BODY_B]
                        ))
                        if kk <= 0 or kk >= nbody:
                            continue
                        for c in range(6):
                            cfrc_ext[env, kk * 6 + c] = qnan
            else:
                var row = 0
                for e in range(nequality):
                    var t2 = Int(
                        rebind[Scalar[DTYPE]](equality[e, EQ_IDX_TYPE])
                    )
                    if t2 == EQ_JOINT:
                        row += 1
                        continue
                    if t2 != EQ_CONNECT and t2 != EQ_WELD:
                        continue

                    var efx = rebind[Scalar[DTYPE]](eq_force[env, row + 0])
                    var efy = rebind[Scalar[DTYPE]](eq_force[env, row + 1])
                    var efz = rebind[Scalar[DTYPE]](eq_force[env, row + 2])
                    var etx = Scalar[DTYPE](0)
                    var ety = Scalar[DTYPE](0)
                    var etz = Scalar[DTYPE](0)
                    if t2 == EQ_WELD:
                        etx = rebind[Scalar[DTYPE]](eq_force[env, row + 3])
                        ety = rebind[Scalar[DTYPE]](eq_force[env, row + 4])
                        etz = rebind[Scalar[DTYPE]](eq_force[env, row + 5])
                    row += 6 if t2 == EQ_WELD else 3

                    for side in range(2):
                        var kk = Int(rebind[Scalar[DTYPE]](
                            equality[e, EQ_IDX_BODY_A if side == 0
                                     else EQ_IDX_BODY_B]
                        ))
                        if kk <= 0 or kk >= nbody:
                            continue
                        var ax = rebind[Scalar[DTYPE]](
                            equality[e, EQ_IDX_ANCHOR_AX if side == 0
                                     else EQ_IDX_ANCHOR_BX]
                        )
                        var ay = rebind[Scalar[DTYPE]](
                            equality[e, EQ_IDX_ANCHOR_AY if side == 0
                                     else EQ_IDX_ANCHOR_BY]
                        )
                        var az = rebind[Scalar[DTYPE]](
                            equality[e, EQ_IDX_ANCHOR_AZ if side == 0
                                     else EQ_IDX_ANCHOR_BZ]
                        )
                        var rot = gpu_quat_rotate[DTYPE](
                            rebind[Scalar[DTYPE]](xquat[env, kk * 4 + 0]),
                            rebind[Scalar[DTYPE]](xquat[env, kk * 4 + 1]),
                            rebind[Scalar[DTYPE]](xquat[env, kk * 4 + 2]),
                            rebind[Scalar[DTYPE]](xquat[env, kk * 4 + 3]),
                            ax, ay, az,
                        )
                        var px2 = rebind[Scalar[DTYPE]](
                            xpos[env, kk * 3 + 0]
                        ) + rot[0]
                        var py2 = rebind[Scalar[DTYPE]](
                            xpos[env, kk * 3 + 1]
                        ) + rot[1]
                        var pz2 = rebind[Scalar[DTYPE]](
                            xpos[env, kk * 3 + 2]
                        ) + rot[2]

                        var rid2 = Int(rebind[Scalar[DTYPE]](
                            bodies[kk, BODY_IDX_ROOTID]
                        ))
                        var ddx = rebind[Scalar[DTYPE]](
                            subtree_com[env, rid2 * 3 + 0]
                        ) - px2
                        var ddy = rebind[Scalar[DTYPE]](
                            subtree_com[env, rid2 * 3 + 1]
                        ) - py2
                        var ddz = rebind[Scalar[DTYPE]](
                            subtree_com[env, rid2 * 3 + 2]
                        ) - pz2
                        var tx2 = etx - (ddy * efz - ddz * efy)
                        var ty2 = ety - (ddz * efx - ddx * efz)
                        var tz2 = etz - (ddx * efy - ddy * efx)

                        # ⚠ BODY 1 ADDS AND BODY 2 SUBTRACTS. The reference's
                        # own comment on the body-1 branch says "opposite for
                        # body 1" and the code there is `mju_addTo`
                        # (engine_core_smooth.c:2503); the comment is stale
                        # and the code is the contract. It is the OPPOSITE
                        # sign convention to the contact block above, where
                        # our stored force is the force on A.
                        var sg = Scalar[DTYPE](1) if side == 0 else Scalar[DTYPE](-1)
                        var oo = kk * 6
                        cfrc_ext[env, oo + 0] = rebind[Scalar[DTYPE]](
                            cfrc_ext[env, oo + 0]) + sg * tx2
                        cfrc_ext[env, oo + 1] = rebind[Scalar[DTYPE]](
                            cfrc_ext[env, oo + 1]) + sg * ty2
                        cfrc_ext[env, oo + 2] = rebind[Scalar[DTYPE]](
                            cfrc_ext[env, oo + 2]) + sg * tz2
                        cfrc_ext[env, oo + 3] = rebind[Scalar[DTYPE]](
                            cfrc_ext[env, oo + 3]) + sg * efx
                        cfrc_ext[env, oo + 4] = rebind[Scalar[DTYPE]](
                            cfrc_ext[env, oo + 4]) + sg * efy
                        cfrc_ext[env, oo + 5] = rebind[Scalar[DTYPE]](
                            cfrc_ext[env, oo + 5]) + sg * efz


@always_inline
def _rne_post_env[
    DTYPE: DType,
    D: DimsLike,
    L_QVEL: TensorLayout,
    L_XQUAT: TensorLayout,
    L_XIPOS: TensorLayout,
    L_CONTACTS: TensorLayout,
    L_DMETA: TensorLayout,
    L_BODIES: TensorLayout,
    L_JOINTS: TensorLayout,
    L_MMETA: TensorLayout,
    L_CDOF: TensorLayout,
    L_CRB: TensorLayout,
    L_CVEL: TensorLayout,
    L_EQUALITY: TensorLayout,
    L_EQFORCE: TensorLayout,
    # CPU dispatcher only — see `_rne_env`.
    JMAP: Bool = False,
](
    env: Int,
    dims: D,
    qvel: TileTensor[DTYPE, L_QVEL, MutAnyOrigin],
    qacc: TileTensor[DTYPE, L_QVEL, MutAnyOrigin],
    xquat: TileTensor[DTYPE, L_XQUAT, MutAnyOrigin],
    xipos: TileTensor[DTYPE, L_XIPOS, MutAnyOrigin],
    subtree_com: TileTensor[DTYPE, L_XIPOS, MutAnyOrigin],
    contacts: TileTensor[DTYPE, L_CONTACTS, MutAnyOrigin],
    dmeta: TileTensor[DTYPE, L_DMETA, MutAnyOrigin],
    bodies: TileTensor[DTYPE, L_BODIES, MutAnyOrigin],
    joints: TileTensor[DTYPE, L_JOINTS, MutAnyOrigin],
    mmeta: TileTensor[DTYPE, L_MMETA, MutAnyOrigin],
    cdof: TileTensor[DTYPE, L_CDOF, MutAnyOrigin],
    crb: TileTensor[DTYPE, L_CRB, MutAnyOrigin],
    cvel: TileTensor[DTYPE, L_CVEL, MutAnyOrigin],
    cacc: TileTensor[DTYPE, L_CVEL, MutAnyOrigin],
    cfrc_ext: TileTensor[DTYPE, L_CVEL, MutAnyOrigin],
    cfrc_int: TileTensor[DTYPE, L_CVEL, MutAnyOrigin],
    # The three operands the equality walk needs on top of the contact one:
    # the body poses the anchors are expressed against, the equality records,
    # and the row forces the solver retained. See `_cfrc_ext_env`.
    xpos: TileTensor[DTYPE, L_XIPOS, MutAnyOrigin],
    equality: TileTensor[DTYPE, L_EQUALITY, MutAnyOrigin],
    eq_force: TileTensor[DTYPE, L_EQFORCE, MutAnyOrigin],
):
    """One env's `mj_rnePostConstraint`. See the module docstring."""
    var nbody = dims.get_nbody()
    var njoint = dims.get_njoint()
    var gx = rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_GRAVITY_X])
    var gy = rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_GRAVITY_Y])
    var gz = rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_GRAVITY_Z])

    comptime B6 = cap[D.NBODY]() * 6
    # ⚠ `nbody * 6`, NOT `B6`. `B6` is a CAP — 0 on a dynamic provider — so
    # bounding this loop with it would leave `cacc` holding the previous
    # step's values on that leg, silently. A cap belongs in a `Scratch[...]`
    # size and nowhere else; `scratchpad/p2b2/audit_caps.py` checks that.
    for i in range(nbody * 6):
        cacc[env, i] = Scalar[DTYPE](0)
    # World acceleration = -gravity. `_rne_fwd_body` writes this into every
    # body whose parent is 0 rather than reading it from here, so body 0's own
    # row is never used by the recursion — but MuJoCo sets it, and leaving it
    # zero would make `d.cacc` disagree with `mjData.cacc` on the one row a
    # parity gate is most likely to print first.
    cacc[env, 3] = -gx
    cacc[env, 4] = -gy
    cacc[env, 5] = -gz
    for i in range(nbody * 6):
        crb[env, i] = Scalar[DTYPE](0)

    comptime CIN = cap[D.NBODY]() * 10
    var cinert_g = Scratch[Scalar[DTYPE], CIN](nbody * 10, uninitialized=0)
    for i in range(nbody * 10):
        cinert_g[i] = Scalar[DTYPE](0)
    for b in range(nbody):
        _rne_cinert_body[DTYPE](
            env, b, xquat, xipos, subtree_com, bodies, cinert_g
        )

    comptime JM_CAP = cap[D.NBODY]() if JMAP else 1
    var jnt_adr = Scratch[Int, JM_CAP](nbody if JMAP else 1, fill=-1)
    var jnt_num = Scratch[Int, JM_CAP](nbody if JMAP else 1, fill=0)
    var map_ok = False
    comptime if JMAP:
        map_ok = body_joint_map[DTYPE, JM_CAP](
            njoint, nbody, joints, jnt_adr, jnt_num
        )
    # 1. cvel (into crb) + the qacc-free part of cacc, verbatim from RNE.
    for b in range(1, nbody):
        var j_lo = jnt_adr[b] if map_ok else 0
        var j_hi = j_lo + jnt_num[b] if map_ok else njoint
        _rne_fwd_body[DTYPE](
            env, b, gx, gy, gz, dims, qvel, bodies, joints, cdof, crb, cacc,
            j_lo, j_hi, map_ok,
        )

    # 2. The cdof*qacc term, as its own forward sweep (see docstring).
    var extra = Scratch[Scalar[DTYPE], B6](nbody * 6, uninitialized=0)
    for i in range(nbody * 6):
        extra[i] = Scalar[DTYPE](0)
    for b in range(1, nbody):
        var parent = Int(rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_PARENT]))
        for k in range(6):
            extra[b * 6 + k] = extra[parent * 6 + k]
        var j_lo = 0
        var j_hi = njoint
        if map_ok:
            j_lo = jnt_adr[b]
            j_hi = j_lo + jnt_num[b]
        for j in range(j_lo, j_hi):
            if not map_ok and Int(
                rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_BODY_ID])
            ) != b:
                continue
            var jt = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_TYPE]))
            var adr = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_DOF_ADR]))
            var ndof = 1
            if jt == JNT_FREE:
                ndof = 6
            elif jt == JNT_BALL:
                ndof = 3
            for dd in range(ndof):
                var dof = adr + dd
                var a = rebind[Scalar[DTYPE]](qacc[env, dof])
                for k in range(6):
                    extra[b * 6 + k] = extra[b * 6 + k] + rebind[
                        Scalar[DTYPE]
                    ](cdof[env, dof * 6 + k]) * a
        for k in range(6):
            cacc[env, b * 6 + k] = (
                rebind[Scalar[DTYPE]](cacc[env, b * 6 + k]) + extra[b * 6 + k]
            )

    # 3. External (contact) forces per body.
    _cfrc_ext_env[DTYPE](
        env, dims, contacts, dmeta, subtree_com, bodies, cfrc_ext,
        xpos, xquat, equality, eq_force, mmeta,
    )

    # 4. cfrc_int = cfrc_body - cfrc_ext, then accumulate leaves -> root.
    for b in range(nbody):
        _rne_cfrc_body[DTYPE](
            env, b, cinert_g, crb, cacc, cfrc_int
        )
    for i in range(nbody * 6):
        cfrc_int[env, i] = rebind[Scalar[DTYPE]](
            cfrc_int[env, i]
        ) - rebind[Scalar[DTYPE]](cfrc_ext[env, i])
    # ⚠ `_rne_backward_env` stops at parent > 0, where MuJoCo accumulates into
    # body 0 as well. `cfrc_int[0]` is therefore ours-only; no sensor reads it
    # (`site_bodyid` is never 0), and for a system with no external wrench the
    # two agree anyway, because the sum it would hold is the net wrench.
    _rne_backward_env[DTYPE](env, dims, bodies, cfrc_int)

    # 5. Publish cvel out of the crb scratch (the accelerometer needs it).
    for b in range(nbody):
        for k in range(6):
            cvel[env, b * 6 + k] = rebind[Scalar[DTYPE]](
                crb[env, b * 6 + k]
            )


def _rne_post_kernel[
    DTYPE: DType,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    MAX_CONTACTS: Int,
    NEQUALITY: Int,
    BATCH: Int,
](
    qvel: Tile[DTYPE, BATCH, NV],
    qacc: Tile[DTYPE, BATCH, NV],
    xquat: Tile[DTYPE, BATCH, NBODY * 4],
    xipos: Tile[DTYPE, BATCH, NBODY * 3],
    subtree_com: Tile[DTYPE, BATCH, NBODY * 3],
    contacts: Tile[DTYPE, BATCH, MAX_CONTACTS * CONTACT_SIZE],
    dmeta: Tile[DTYPE, BATCH, METADATA_SIZE],
    bodies: Tile[DTYPE, NBODY, MODEL_BODY_SIZE],
    joints: Tile[DTYPE, NJOINT, MODEL_JOINT_SIZE],
    mmeta: Tile[DTYPE, MODEL_META_SIZE],
    cdof: Tile[DTYPE, BATCH, NV * 6],
    crb: Tile[DTYPE, BATCH, NBODY * 10],
    cvel: Tile[DTYPE, BATCH, NBODY * 6],
    cacc: Tile[DTYPE, BATCH, NBODY * 6],
    cfrc_ext: Tile[DTYPE, BATCH, NBODY * 6],
    cfrc_int: Tile[DTYPE, BATCH, NBODY * 6],
    xpos: Tile[DTYPE, BATCH, NBODY * 3],
    equality: Tile[DTYPE, _max_one[NEQUALITY](), MODEL_EQ_SIZE],
    eq_force: Tile[DTYPE, BATCH, _max_one[6 * NEQUALITY]()],
):
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    _rne_post_env[DTYPE](
        env, Dims[nv=NV, nbody=NBODY, njoint=NJOINT, max_contacts=MAX_CONTACTS, nequality=NEQUALITY](), qvel, qacc, xquat, xipos, subtree_com, contacts, dmeta, bodies,
        joints, mmeta, cdof, crb, cvel, cacc, cfrc_ext, cfrc_int,
        xpos, equality, eq_force,
    )


def compute_rne_post[

    target: StaticString,
    DTYPE: DType,
    D: DimsLike,
    BATCH: Int = 1,
    # Appended, not grouped with NEXCLUDE — see `fields.Model`.
](
    mut d: Data[DTYPE, D, BATCH],
    mut m: Model[DTYPE, D],
    mut scratch: DynamicsScratch[DTYPE, D, BATCH],
    ctx: Optional[DeviceContext] = None,
) raises:
    """Fill `d.cacc` / `d.cfrc_int` (and `d.cvel` / `d.cfrc_ext`) for the
    CURRENT state. Run between the constraint solve and the integration —
    that is where MuJoCo's `mj_sensorAcc` sits, and the FK products, the
    contact forces and `scratch.qacc_constrained` are all valid there."""
    comptime L_NV = RowMajor[BATCH, D.NV]
    comptime L_B3 = RowMajor[BATCH, D.NBODY * 3]
    comptime L_B4 = RowMajor[BATCH, D.NBODY * 4]
    comptime L_B6 = RowMajor[BATCH, D.NBODY * 6]
    comptime L_B10 = RowMajor[BATCH, D.NBODY * 10]
    comptime L_CON = RowMajor[BATCH, D.MAX_CONTACTS * CONTACT_SIZE]
    comptime L_EQ_RP = RowMajor[_max_one[D.NEQUALITY](), MODEL_EQ_SIZE]
    comptime L_EQF_RP = RowMajor[BATCH, _max_one[6 * D.NEQUALITY]()]
    comptime L_DMETA = RowMajor[BATCH, METADATA_SIZE]
    comptime L_BODY = RowMajor[D.NBODY, MODEL_BODY_SIZE]
    comptime L_JOINT = RowMajor[D.NJOINT, MODEL_JOINT_SIZE]
    comptime L_MMETA = RowMajor[MODEL_META_SIZE]
    comptime L_CDOF = RowMajor[BATCH, D.NV * 6]

    comptime if target == "cpu":
        var dm = d.dims
        var rl_NV = rl2(BATCH, dm.get_nv())
        var rl_B4 = rl2(BATCH, dm.get_nbody() * 4)
        var rl_B3 = rl2(BATCH, dm.get_nbody() * 3)
        var rl_CON = rl2(BATCH, dm.get_max_contacts() * CONTACT_SIZE)
        var rl_DMETA = rl2(BATCH, METADATA_SIZE)
        var rl_BODY = rl2(dm.get_nbody(), MODEL_BODY_SIZE)
        var rl_JOINT = rl2(dm.get_njoint(), MODEL_JOINT_SIZE)
        var rl_MMETA = rl1(MODEL_META_SIZE)
        var rl_CDOF = rl2(BATCH, dm.get_nv() * 6)
        var rl_B10 = rl2(BATCH, dm.get_nbody() * 10)
        var rl_B6 = rl2(BATCH, dm.get_nbody() * 6)
        var qvel_v = d.qvel.tt_dyn["cpu", DYN2](rl_NV)
        var qacc_v = scratch.qacc_constrained.tt_dyn["cpu", DYN2](rl_NV)
        var xquat_v = d.xquat.tt_dyn["cpu", DYN2](rl_B4)
        var xipos_v = d.xipos.tt_dyn["cpu", DYN2](rl_B3)
        var stcom_v = d.subtree_com.tt_dyn["cpu", DYN2](rl_B3)
        var con_v = d.contacts.tt_dyn["cpu", DYN2](rl_CON)
        var dmeta_v = d.meta.tt_dyn["cpu", DYN2](rl_DMETA)
        var bodies_v = m.bodies.tt_dyn["cpu", DYN2](rl_BODY)
        var joints_v = m.joints.tt_dyn["cpu", DYN2](rl_JOINT)
        var mmeta_v = m.meta.tt_dyn["cpu", DYN1](rl_MMETA)
        var cdof_v = scratch.cdof.tt_dyn["cpu", DYN2](rl_CDOF)
        var crb_v = scratch.crb.tt_dyn["cpu", DYN2](rl_B10)
        var cvel_v = d.cvel.tt_dyn["cpu", DYN2](rl_B6)
        var cacc_v = d.cacc.tt_dyn["cpu", DYN2](rl_B6)
        var cfrc_ext_v = d.cfrc_ext.tt_dyn["cpu", DYN2](rl_B6)
        var cfrc_int_v = d.cfrc_int.tt_dyn["cpu", DYN2](rl_B6)
        var xpos_v = d.xpos.tt_dyn["cpu", DYN2](rl_B3)
        var eq_v = m.equality.tt_dyn["cpu", DYN2](
            rl2(_max_one_rt(dm.get_nequality()), MODEL_EQ_SIZE)
        )
        var eqf_v = d.efc_eq_force.tt_dyn["cpu", DYN2](
            rl2(BATCH, _max_one_rt(dm.get_nequality() * 6))
        )
        for e in range(BATCH):
            _rne_post_env[DTYPE, JMAP=True](
                e, dm, qvel_v, qacc_v, xquat_v, xipos_v, stcom_v, con_v, dmeta_v,
                bodies_v, joints_v, mmeta_v, cdof_v, crb_v, cvel_v, cacc_v,
                cfrc_ext_v, cfrc_int_v, xpos_v, eq_v, eqf_v,
            )
    else:
        var c = ctx.value()
        comptime BLOCKS = (BATCH + RNE_POST_TPB - 1) // RNE_POST_TPB
        c.enqueue_function[
            _rne_post_kernel[
                DTYPE, D.NV, D.NBODY, D.NJOINT, D.MAX_CONTACTS, D.NEQUALITY,
                BATCH,
            ]
        ](
            d.qvel.tt["gpu", BATCH, D.NV](),
            scratch.qacc_constrained.tt["gpu", BATCH, D.NV](),
            d.xquat.tt["gpu", BATCH, D.NBODY * 4](),
            d.xipos.tt["gpu", BATCH, D.NBODY * 3](),
            d.subtree_com.tt["gpu", BATCH, D.NBODY * 3](),
            d.contacts.tt["gpu", BATCH, D.MAX_CONTACTS * CONTACT_SIZE](),
            d.meta.tt["gpu", BATCH, METADATA_SIZE](),
            m.bodies.tt["gpu", D.NBODY, MODEL_BODY_SIZE](),
            m.joints.tt["gpu", D.NJOINT, MODEL_JOINT_SIZE](),
            m.meta.tt["gpu", MODEL_META_SIZE](),
            scratch.cdof.tt["gpu", BATCH, D.NV * 6](),
            scratch.crb.tt["gpu", BATCH, D.NBODY * 10](),
            d.cvel.tt["gpu", BATCH, D.NBODY * 6](),
            d.cacc.tt["gpu", BATCH, D.NBODY * 6](),
            d.cfrc_ext.tt["gpu", BATCH, D.NBODY * 6](),
            d.cfrc_int.tt["gpu", BATCH, D.NBODY * 6](),
            d.xpos.tt["gpu", BATCH, D.NBODY * 3](),
            m.equality.tt["gpu", _max_one[D.NEQUALITY](), MODEL_EQ_SIZE](),
            d.efc_eq_force.tt["gpu", BATCH, _max_one[6 * D.NEQUALITY]()](),
            grid_dim=(BLOCKS,),
            block_dim=(RNE_POST_TPB,),
        )
