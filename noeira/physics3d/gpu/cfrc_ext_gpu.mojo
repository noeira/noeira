"""GPU kernel for computing cfrc_ext: external contact forces per body.

Mirrors the CPU compute_cfrc_ext() in physics3d/dynamics/cfrc_ext.mojo.

Algorithm (one GPU thread per environment):
1. Zero cfrc_ext region in state buffer.
2. Compute subtree CoM for each body (backward pass using body_mass/body_parent from model).
3. Compute body_rootid for each body.
4. For each active contact: reconstruct world-frame force/torque, apply moment-arm
   correction, accumulate into cfrc_ext[body_a] and subtract from cfrc_ext[body_b].

cfrc_ext[b*6 + 0..5] = [torque_x, torque_y, torque_z, force_x, force_y, force_z]
expressed in world frame at subtree CoM of the body's kinematic root.
"""

from std.collections import Array

from max.gpu.host import DeviceContext, DeviceBuffer
from max.gpu import thread_idx, block_idx, block_dim
from layout import TileTensor, TensorLayout
from noeira.core.tiles import Tile

from .constants import (
    METADATA_SIZE,
    TPB,
    META_IDX_NUM_CONTACTS,
    CONTACT_SIZE,
    CONTACT_IDX_BODY_A,
    CONTACT_IDX_BODY_B,
    CONTACT_IDX_POS_X,
    CONTACT_IDX_POS_Y,
    CONTACT_IDX_POS_Z,
    CONTACT_IDX_NX,
    CONTACT_IDX_NY,
    CONTACT_IDX_NZ,
    CONTACT_IDX_FORCE_N,
    CONTACT_IDX_FORCE_T1,
    CONTACT_IDX_FORCE_T2,
    CONTACT_IDX_FORCE_TORSION,
    CONTACT_IDX_FORCE_ROLL1,
    CONTACT_IDX_FORCE_ROLL2,
    CONTACT_IDX_FRAME_T1_X,
    CONTACT_IDX_FRAME_T1_Y,
    CONTACT_IDX_FRAME_T1_Z,
    BODY_IDX_MASS,
    BODY_IDX_PARENT,
    MODEL_BODY_SIZE,
)
from ..collision.contact_frame import contact_tangent_frame
from ..fields import DimsLike


def compute_cfrc_ext[
    DTYPE: DType,
    BATCH_SIZE: Int,
    D: DimsLike,
    L_XIPOS: TensorLayout,
    L_CONTACTS: TensorLayout,
    L_META: TensorLayout,
    L_CFRC_EXT: TensorLayout,
    L_BODIES: TensorLayout,
](
    ctx: DeviceContext,
    dims: D,
    xipos: TileTensor[DTYPE, L_XIPOS, MutAnyOrigin],
    contacts: TileTensor[DTYPE, L_CONTACTS, MutAnyOrigin],
    meta: TileTensor[DTYPE, L_META, MutAnyOrigin],
    cfrc_ext: TileTensor[DTYPE, L_CFRC_EXT, MutAnyOrigin],
    bodies: TileTensor[DTYPE, L_BODIES, MutAnyOrigin],
) raises:
    """Per-field cfrc_ext (G5 — no state slab): contact forces accumulated
    into per-root-subtree spatial force records, arithmetic verbatim from the
    legacy slab kernel. body_mass / body_parent come from the packed
    `Model.bodies` records; xipos/contacts/meta/cfrc_ext are the
    Data tensors.
    """
    var nbody = dims.get_nbody()
    var max_contacts = dims.get_max_contacts()
    comptime BLOCKS = (BATCH_SIZE + TPB - 1) // TPB

    comptime EPS = Scalar[DTYPE](1e-10)

    @always_inline
    # ⚠⚠ COMPTIME INSIDE THE KERNEL. `dims` is a host value and
    # `Dims` is not `DevicePassable` — reading it here yields 0,
    # every loop bound collapses and the output comes back zeroed
    # (gated by `test_cfrc_ext_batched_vs_cpu`, which caught
    # exactly that as rel = 1.0). Decision 3 keeps this leg
    # comptime, so the caps are the right and only source here.
    def cfrc_ext_fields_kernel(
        xipos: Tile[DTYPE, BATCH_SIZE, D.CAP_NBODY * 3],
        contacts: Tile[DTYPE, BATCH_SIZE, D.CAP_MAX_CONTACTS * CONTACT_SIZE],
        meta: Tile[DTYPE, BATCH_SIZE, METADATA_SIZE],
        cfrc_ext: Tile[DTYPE, BATCH_SIZE, D.CAP_NBODY * 6],
        bodies: Tile[DTYPE, D.CAP_NBODY, MODEL_BODY_SIZE],
    ):
        var env = Int(block_dim.x * block_idx.x + thread_idx.x)
        if env >= BATCH_SIZE:
            return

        # --- 1. Zero cfrc_ext ---
        for i in range(D.CAP_NBODY * 6):
            cfrc_ext[env, i] = Scalar[DTYPE](0)

        # --- 2. Compute subtree_com for each body ---
        var stmass = Array[Scalar[DTYPE], D.CAP_NBODY](uninitialized=True)
        var stcom = Array[Scalar[DTYPE], D.CAP_NBODY * 3](uninitialized=True)

        for i in range(D.CAP_NBODY):
            var m = rebind[Scalar[DTYPE]](bodies[i, BODY_IDX_MASS])
            stmass[i] = m
            stcom[i * 3 + 0] = m * rebind[Scalar[DTYPE]](
                xipos[env, i * 3 + 0]
            )
            stcom[i * 3 + 1] = m * rebind[Scalar[DTYPE]](
                xipos[env, i * 3 + 1]
            )
            stcom[i * 3 + 2] = m * rebind[Scalar[DTYPE]](
                xipos[env, i * 3 + 2]
            )

        # Backward sweep: add child contribution to parent
        for i in range(D.CAP_NBODY - 1, 0, -1):
            var p = Int(rebind[Scalar[DTYPE]](bodies[i, BODY_IDX_PARENT]))
            stmass[p] += stmass[i]
            stcom[p * 3 + 0] += stcom[i * 3 + 0]
            stcom[p * 3 + 1] += stcom[i * 3 + 1]
            stcom[p * 3 + 2] += stcom[i * 3 + 2]

        # Normalize to get CoM position
        for i in range(D.CAP_NBODY):
            var sm = stmass[i]
            if sm > EPS:
                stcom[i * 3 + 0] = stcom[i * 3 + 0] / sm
                stcom[i * 3 + 1] = stcom[i * 3 + 1] / sm
                stcom[i * 3 + 2] = stcom[i * 3 + 2] / sm
            else:
                stcom[i * 3 + 0] = rebind[Scalar[DTYPE]](
                    xipos[env, i * 3 + 0]
                )
                stcom[i * 3 + 1] = rebind[Scalar[DTYPE]](
                    xipos[env, i * 3 + 1]
                )
                stcom[i * 3 + 2] = rebind[Scalar[DTYPE]](
                    xipos[env, i * 3 + 2]
                )

        # --- 3. Compute body_rootid ---
        var rootid = Array[Int, D.CAP_NBODY](uninitialized=True)
        rootid[0] = 0
        for i in range(1, D.CAP_NBODY):
            var p = Int(rebind[Scalar[DTYPE]](bodies[i, BODY_IDX_PARENT]))
            if p == 0:
                rootid[i] = i
            else:
                rootid[i] = rootid[p]

        # --- 4. Accumulate contact forces ---
        var num_contacts = Int(
            rebind[Scalar[DTYPE]](meta[env, META_IDX_NUM_CONTACTS])
        )

        for ci in range(D.CAP_MAX_CONTACTS):
            if ci >= num_contacts:
                break

            var con_base = ci * CONTACT_SIZE

            # Contact frame axes
            var nx = rebind[Scalar[DTYPE]](
                contacts[env, con_base + CONTACT_IDX_NX]
            )
            var ny = rebind[Scalar[DTYPE]](
                contacts[env, con_base + CONTACT_IDX_NY]
            )
            var nz = rebind[Scalar[DTYPE]](
                contacts[env, con_base + CONTACT_IDX_NZ]
            )
            # FRAME_T1 is a HINT, not a tangent — unnormalized, not orthogonal
            # to the normal, and written only by the capsule narrow phases.
            # Reading it raw left the tangential force pointing somewhere
            # arbitrary while the normal component stayed correct, so
            # `contact_cost` (a squared norm over this) read wrong on every
            # model whose contacts are not capsule-vs-something.
            # See collision/contact_frame.mojo.
            var frame = contact_tangent_frame[DTYPE](
                nx,
                ny,
                nz,
                rebind[Scalar[DTYPE]](
                    contacts[env, con_base + CONTACT_IDX_FRAME_T1_X]
                ),
                rebind[Scalar[DTYPE]](
                    contacts[env, con_base + CONTACT_IDX_FRAME_T1_Y]
                ),
                rebind[Scalar[DTYPE]](
                    contacts[env, con_base + CONTACT_IDX_FRAME_T1_Z]
                ),
            )
            var t1x = frame[0]
            var t1y = frame[1]
            var t1z = frame[2]
            var t2x = frame[3]
            var t2y = frame[4]
            var t2z = frame[5]

            # Contact forces in contact-local frame
            var f_n = rebind[Scalar[DTYPE]](
                contacts[env, con_base + CONTACT_IDX_FORCE_N]
            )
            var f_t1 = rebind[Scalar[DTYPE]](
                contacts[env, con_base + CONTACT_IDX_FORCE_T1]
            )
            var f_t2 = rebind[Scalar[DTYPE]](
                contacts[env, con_base + CONTACT_IDX_FORCE_T2]
            )
            var f_tors = rebind[Scalar[DTYPE]](
                contacts[env, con_base + CONTACT_IDX_FORCE_TORSION]
            )
            var f_roll1 = rebind[Scalar[DTYPE]](
                contacts[env, con_base + CONTACT_IDX_FORCE_ROLL1]
            )
            var f_roll2 = rebind[Scalar[DTYPE]](
                contacts[env, con_base + CONTACT_IDX_FORCE_ROLL2]
            )

            # World-frame force: f_n*N + f_t1*T1 + f_t2*T2
            var fw_x = f_n * nx + f_t1 * t1x + f_t2 * t2x
            var fw_y = f_n * ny + f_t1 * t1y + f_t2 * t2y
            var fw_z = f_n * nz + f_t1 * t1z + f_t2 * t2z
            # World-frame torque: f_tors*N + f_roll1*T1 + f_roll2*T2
            var tw_x = f_tors * nx + f_roll1 * t1x + f_roll2 * t2x
            var tw_y = f_tors * ny + f_roll1 * t1y + f_roll2 * t2y
            var tw_z = f_tors * nz + f_roll1 * t1z + f_roll2 * t2z

            # Contact point
            var cx = rebind[Scalar[DTYPE]](
                contacts[env, con_base + CONTACT_IDX_POS_X]
            )
            var cy = rebind[Scalar[DTYPE]](
                contacts[env, con_base + CONTACT_IDX_POS_Y]
            )
            var cz = rebind[Scalar[DTYPE]](
                contacts[env, con_base + CONTACT_IDX_POS_Z]
            )

            var ka = Int(
                rebind[Scalar[DTYPE]](
                    contacts[env, con_base + CONTACT_IDX_BODY_A]
                )
            )
            var kb = Int(
                rebind[Scalar[DTYPE]](
                    contacts[env, con_base + CONTACT_IDX_BODY_B]
                )
            )

            # body_a: add direct force
            if ka > 0:
                var rid = rootid[ka]
                var dx = stcom[rid * 3 + 0] - cx
                var dy = stcom[rid * 3 + 1] - cy
                var dz = stcom[rid * 3 + 2] - cz
                # moment arm correction: tw_corrected = tw - (d × fw)
                var cx_ = dy * fw_z - dz * fw_y
                var cy_ = dz * fw_x - dx * fw_z
                var cz_ = dx * fw_y - dy * fw_x
                var base_off = ka * 6
                cfrc_ext[env, base_off + 0] = rebind[Scalar[DTYPE]](
                    cfrc_ext[env, base_off + 0]
                ) + (tw_x - cx_)
                cfrc_ext[env, base_off + 1] = rebind[Scalar[DTYPE]](
                    cfrc_ext[env, base_off + 1]
                ) + (tw_y - cy_)
                cfrc_ext[env, base_off + 2] = rebind[Scalar[DTYPE]](
                    cfrc_ext[env, base_off + 2]
                ) + (tw_z - cz_)
                cfrc_ext[env, base_off + 3] = (
                    rebind[Scalar[DTYPE]](cfrc_ext[env, base_off + 3]) + fw_x
                )
                cfrc_ext[env, base_off + 4] = (
                    rebind[Scalar[DTYPE]](cfrc_ext[env, base_off + 4]) + fw_y
                )
                cfrc_ext[env, base_off + 5] = (
                    rebind[Scalar[DTYPE]](cfrc_ext[env, base_off + 5]) + fw_z
                )

            # body_b: subtract reaction force (Newton's 3rd law)
            if kb > 0:
                var rid = rootid[kb]
                var dx = stcom[rid * 3 + 0] - cx
                var dy = stcom[rid * 3 + 1] - cy
                var dz = stcom[rid * 3 + 2] - cz
                var cx_ = dy * fw_z - dz * fw_y
                var cy_ = dz * fw_x - dx * fw_z
                var cz_ = dx * fw_y - dy * fw_x
                var base_off = kb * 6
                cfrc_ext[env, base_off + 0] = rebind[Scalar[DTYPE]](
                    cfrc_ext[env, base_off + 0]
                ) - (tw_x - cx_)
                cfrc_ext[env, base_off + 1] = rebind[Scalar[DTYPE]](
                    cfrc_ext[env, base_off + 1]
                ) - (tw_y - cy_)
                cfrc_ext[env, base_off + 2] = rebind[Scalar[DTYPE]](
                    cfrc_ext[env, base_off + 2]
                ) - (tw_z - cz_)
                cfrc_ext[env, base_off + 3] = (
                    rebind[Scalar[DTYPE]](cfrc_ext[env, base_off + 3]) - fw_x
                )
                cfrc_ext[env, base_off + 4] = (
                    rebind[Scalar[DTYPE]](cfrc_ext[env, base_off + 4]) - fw_y
                )
                cfrc_ext[env, base_off + 5] = (
                    rebind[Scalar[DTYPE]](cfrc_ext[env, base_off + 5]) - fw_z
                )

    ctx.enqueue_function[cfrc_ext_fields_kernel](
        xipos,
        contacts,
        meta,
        cfrc_ext,
        bodies,
        grid_dim=(BLOCKS,),
        block_dim=(TPB,),
    )
