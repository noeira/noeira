"""CRBA mass matrix over per-field tensors (migration P2, single-source).

Per-field port of `compute_mass_matrix_full_gpu` (dynamics/mass_matrix.mojo)
— arithmetic verbatim. Reads `scratch.cdof`, writes `scratch.M` (owned
tensors, replacing the ws_cdof/ws_M regions). Per-thread scratch (dof_body,
world-frame inertia, subtree mask) stays in `Scratch` (2b.2).

Operands: xquat, xipos, subtree_com + body/joint records + cdof -> M
(7 operands). `num_joints` is the comptime NJOINT (no metadata read)."""

from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.sync import barrier
from max.gpu.memory import AddressSpace
from max.gpu.host import DeviceContext
from layout import TileTensor, TensorLayout
from noeira.core.tiles import RowMajor, Tile, shared_tile

from ..kinematics.quat_math import gpu_quat_mul
from ..joint_types import JNT_FREE, JNT_BALL
from ..fields import (
    Data,
    Model,
    DynamicsScratch,
    Dims,
    DimsLike,
    AsStatic,
    Scratch,
    cap,
    DYN2,
    rl2,
)
from ..gpu.constants import (
    MODEL_BODY_SIZE,
    MODEL_JOINT_SIZE,
    BODY_IDX_MASS,
    BODY_IDX_IXX,
    BODY_IDX_IYY,
    BODY_IDX_IZZ,
    BODY_IDX_PARENT,
    BODY_IDX_IQUAT_X,
    BODY_IDX_IQUAT_Y,
    BODY_IDX_IQUAT_Z,
    BODY_IDX_IQUAT_W,
    BODY_IDX_ROOTID,
    JOINT_IDX_TYPE,
    JOINT_IDX_BODY_ID,
    JOINT_IDX_DOF_ADR,
)

comptime MM_TPB: Int = 64


@always_inline
def _mm_setup_env[
    DTYPE: DType,
    D: DimsLike,
    V_CAP: Int,
    B6_CAP: Int,
    MASK_CAP: Int,
    L_XQUAT: TensorLayout,
    L_BODIES: TensorLayout,
    L_JOINTS: TensorLayout,
](
    env: Int,
    dims: D,
    xquat: TileTensor[DTYPE, L_XQUAT, MutAnyOrigin],
    bodies: TileTensor[DTYPE, L_BODIES, MutAnyOrigin],
    joints: TileTensor[DTYPE, L_JOINTS, MutAnyOrigin],
    mut dof_body: Scratch[Int, V_CAP],
    mut I_world: Scratch[Scalar[DTYPE], B6_CAP],
    mut subtree_mask: Scratch[Bool, MASK_CAP],
):
    """CRBA setup (dof->body map, per-body world inertia, subtree mask).
    Extracted verbatim from `_mass_matrix_env` so the serial and _mt
    schedules share identical arithmetic (model + FK-state reads only ->
    every thread computes the same values)."""
    var njoint = dims.get_njoint()
    var nv = dims.get_nv()
    var nbody = dims.get_nbody()
    for i in range(nv):
        dof_body[i] = 0

    for j in range(njoint):
        var jnt_type = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_TYPE]))
        var body_id = Int(
            rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_BODY_ID])
        )
        var dof_adr = Int(
            rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_DOF_ADR])
        )

        var ndof = 1
        if jnt_type == JNT_FREE:
            ndof = 6
        elif jnt_type == JNT_BALL:
            ndof = 3
        for d in range(ndof):
            dof_body[dof_adr + d] = body_id

    # Per-body world-frame inertia tensor
    for b in range(nbody):
        var Ixx_l = rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_IXX])
        var Iyy_l = rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_IYY])
        var Izz_l = rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_IZZ])

        var bqx = rebind[Scalar[DTYPE]](xquat[env, b * 4 + 0])
        var bqy = rebind[Scalar[DTYPE]](xquat[env, b * 4 + 1])
        var bqz = rebind[Scalar[DTYPE]](xquat[env, b * 4 + 2])
        var bqw = rebind[Scalar[DTYPE]](xquat[env, b * 4 + 3])
        var iqx = rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_IQUAT_X])
        var iqy = rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_IQUAT_Y])
        var iqz = rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_IQUAT_Z])
        var iqw = rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_IQUAT_W])
        var iq = gpu_quat_mul(bqx, bqy, bqz, bqw, iqx, iqy, iqz, iqw)
        var qx = iq[0]
        var qy = iq[1]
        var qz = iq[2]
        var qw = iq[3]

        var r00 = Scalar[DTYPE](1) - Scalar[DTYPE](2) * (qy * qy + qz * qz)
        var r10 = Scalar[DTYPE](2) * (qx * qy + qw * qz)
        var r20 = Scalar[DTYPE](2) * (qx * qz - qw * qy)
        var r01 = Scalar[DTYPE](2) * (qx * qy - qw * qz)
        var r11 = Scalar[DTYPE](1) - Scalar[DTYPE](2) * (qx * qx + qz * qz)
        var r21 = Scalar[DTYPE](2) * (qy * qz + qw * qx)
        var r02 = Scalar[DTYPE](2) * (qx * qz + qw * qy)
        var r12 = Scalar[DTYPE](2) * (qy * qz - qw * qx)
        var r22 = Scalar[DTYPE](1) - Scalar[DTYPE](2) * (qx * qx + qy * qy)

        I_world[b * 6 + 0] = (
            Ixx_l * r00 * r00 + Iyy_l * r01 * r01 + Izz_l * r02 * r02
        )
        I_world[b * 6 + 1] = (
            Ixx_l * r10 * r10 + Iyy_l * r11 * r11 + Izz_l * r12 * r12
        )
        I_world[b * 6 + 2] = (
            Ixx_l * r20 * r20 + Iyy_l * r21 * r21 + Izz_l * r22 * r22
        )
        I_world[b * 6 + 3] = (
            Ixx_l * r00 * r10 + Iyy_l * r01 * r11 + Izz_l * r02 * r12
        )
        I_world[b * 6 + 4] = (
            Ixx_l * r00 * r20 + Iyy_l * r01 * r21 + Izz_l * r02 * r22
        )
        I_world[b * 6 + 5] = (
            Ixx_l * r10 * r20 + Iyy_l * r11 * r21 + Izz_l * r12 * r22
        )

    # Subtree membership mask (O(1) lookups in the inner loop)
    for k in range(nbody):
        subtree_mask[k * nbody + k] = True
        var current = k
        while current > 0:
            var parent = Int(
                rebind[Scalar[DTYPE]](bodies[current, BODY_IDX_PARENT])
            )
            subtree_mask[k * nbody + parent] = True
            current = parent


@always_inline
def _mm_row_env[
    DTYPE: DType,
    D: DimsLike,
    V_CAP: Int,
    B6_CAP: Int,
    MASK_CAP: Int,
    L_XIPOS: TensorLayout,
    L_BODIES: TensorLayout,
    L_CDOF: TensorLayout,
    L_M: TensorLayout,
](
    env: Int,
    i: Int,
    dims: D,
    xipos: TileTensor[DTYPE, L_XIPOS, MutAnyOrigin],
    subtree_com: TileTensor[DTYPE, L_XIPOS, MutAnyOrigin],
    bodies: TileTensor[DTYPE, L_BODIES, MutAnyOrigin],
    cdof: TileTensor[DTYPE, L_CDOF, MutAnyOrigin],
    M: TileTensor[DTYPE, L_M, MutAnyOrigin],
    dof_body: Scratch[Int, V_CAP],
    I_world: Scratch[Scalar[DTYPE], B6_CAP],
    subtree_mask: Scratch[Bool, MASK_CAP],
):
    """One CRBA row i (M[i,j] for j>=i + symmetric writes). Extracted
    verbatim from the `_mass_matrix_env` row loop so serial and _mt
    share identical arithmetic."""
    var nv = dims.get_nv()
    var nbody = dims.get_nbody()
    var body_i = dof_body[i]
    var ai0 = cdof[env, i * 6 + 0]
    var ai1 = cdof[env, i * 6 + 1]
    var ai2 = cdof[env, i * 6 + 2]
    var li0 = cdof[env, i * 6 + 3]
    var li1 = cdof[env, i * 6 + 4]
    var li2 = cdof[env, i * 6 + 5]

    for j in range(i, nv):
            var body_j = dof_body[j]
            var aj0 = cdof[env, j * 6 + 0]
            var aj1 = cdof[env, j * 6 + 1]
            var aj2 = cdof[env, j * 6 + 2]
            var lj0 = cdof[env, j * 6 + 3]
            var lj1 = cdof[env, j * 6 + 4]
            var lj2 = cdof[env, j * 6 + 5]

            var mij: M.ElementType = 0

            for k in range(nbody):
                if not subtree_mask[k * nbody + body_i]:
                    continue
                if not subtree_mask[k * nbody + body_j]:
                    continue

                var mk = rebind[Scalar[DTYPE]](bodies[k, BODY_IDX_MASS])
                var pk0 = rebind[Scalar[DTYPE]](xipos[env, k * 3 + 0])
                var pk1 = rebind[Scalar[DTYPE]](xipos[env, k * 3 + 1])
                var pk2 = rebind[Scalar[DTYPE]](xipos[env, k * 3 + 2])

                var ri_root = Int(
                    rebind[Scalar[DTYPE]](bodies[body_i, BODY_IDX_ROOTID])
                )
                var pi0 = rebind[Scalar[DTYPE]](
                    subtree_com[env, ri_root * 3 + 0]
                )
                var pi1 = rebind[Scalar[DTYPE]](
                    subtree_com[env, ri_root * 3 + 1]
                )
                var pi2 = rebind[Scalar[DTYPE]](
                    subtree_com[env, ri_root * 3 + 2]
                )
                var di0 = pk0 - pi0
                var di1 = pk1 - pi1
                var di2 = pk2 - pi2
                var vki0 = li0 + ai1 * di2 - ai2 * di1
                var vki1 = li1 + ai2 * di0 - ai0 * di2
                var vki2 = li2 + ai0 * di1 - ai1 * di0

                var rj_root = Int(
                    rebind[Scalar[DTYPE]](bodies[body_j, BODY_IDX_ROOTID])
                )
                var pj0 = rebind[Scalar[DTYPE]](
                    subtree_com[env, rj_root * 3 + 0]
                )
                var pj1 = rebind[Scalar[DTYPE]](
                    subtree_com[env, rj_root * 3 + 1]
                )
                var pj2 = rebind[Scalar[DTYPE]](
                    subtree_com[env, rj_root * 3 + 2]
                )
                var dj0 = pk0 - pj0
                var dj1 = pk1 - pj1
                var dj2 = pk2 - pj2
                var vkj0 = lj0 + aj1 * dj2 - aj2 * dj1
                var vkj1 = lj1 + aj2 * dj0 - aj0 * dj2
                var vkj2 = lj2 + aj0 * dj1 - aj1 * dj0

                mij = mij + mk * (vki0 * vkj0 + vki1 * vkj1 + vki2 * vkj2)

                var Ik_xx = I_world[k * 6 + 0]
                var Ik_yy = I_world[k * 6 + 1]
                var Ik_zz = I_world[k * 6 + 2]
                var Ik_xy = I_world[k * 6 + 3]
                var Ik_xz = I_world[k * 6 + 4]
                var Ik_yz = I_world[k * 6 + 5]

                var Iaj0 = Ik_xx * aj0 + Ik_xy * aj1 + Ik_xz * aj2
                var Iaj1 = Ik_xy * aj0 + Ik_yy * aj1 + Ik_yz * aj2
                var Iaj2 = Ik_xz * aj0 + Ik_yz * aj1 + Ik_zz * aj2

                mij = mij + ai0 * Iaj0 + ai1 * Iaj1 + ai2 * Iaj2

            M[env, i * nv + j] = mij
            if i != j:
                M[env, j * nv + i] = mij


@always_inline
def _mass_matrix_env[
    DTYPE: DType,
    D: DimsLike,
    L_XQUAT: TensorLayout,
    L_XIPOS: TensorLayout,
    L_BODIES: TensorLayout,
    L_JOINTS: TensorLayout,
    L_CDOF: TensorLayout,
    L_M: TensorLayout,
](
    env: Int,
    dims: D,
    xquat: TileTensor[DTYPE, L_XQUAT, MutAnyOrigin],
    xipos: TileTensor[DTYPE, L_XIPOS, MutAnyOrigin],
    subtree_com: TileTensor[DTYPE, L_XIPOS, MutAnyOrigin],
    bodies: TileTensor[DTYPE, L_BODIES, MutAnyOrigin],
    joints: TileTensor[DTYPE, L_JOINTS, MutAnyOrigin],
    cdof: TileTensor[DTYPE, L_CDOF, MutAnyOrigin],
    M: TileTensor[DTYPE, L_M, MutAnyOrigin],
):
    """Full nv x nv mass matrix for one env (arithmetic verbatim from
    compute_mass_matrix_full_gpu; setup/row bodies now live in the shared
    `_mm_setup_env` / `_mm_row_env` helpers — pure refactor,
    gated bit-exact by tests/physics3d/test_fk_fields.mojo)."""
    var nv = dims.get_nv()
    var nbody = dims.get_nbody()
    for i in range(nv * nv):
        M[env, i] = 0

    # ⚠ `cap[]`, NOT the dimension: on a dynamic provider `D.NBODY` is
    # DIM_POISON (-1) and `NBODY * NBODY` would come out POSITIVE 1, which
    # silently selects the stack leg with a one-element mask. `cap[]` maps
    # poison to 0 so every product containing it is 0. See fields/scratch.mojo.
    comptime V_CAP = cap[D.NV]()
    comptime B6_CAP = cap[D.NBODY]() * 6
    comptime MASK_CAP = cap[D.NBODY]() * cap[D.NBODY]()
    var dof_body = Scratch[Int, V_CAP](nv, uninitialized=0)
    var I_world = Scratch[Scalar[DTYPE], B6_CAP](nbody * 6, uninitialized=0)
    var subtree_mask = Scratch[Bool, MASK_CAP](nbody * nbody, False)
    _mm_setup_env[DTYPE](
        env, dims, xquat, bodies, joints, dof_body, I_world, subtree_mask
    )

    # M[i,j] via direct body summation with subtree mask lookup
    for i in range(nv):
        _mm_row_env[DTYPE](
            env, i, dims, xipos, subtree_com, bodies, cdof, M,
            dof_body, I_world, subtree_mask,
        )


# ── Cooperative (_mt) kernel — schedule from the legacy
# `compute_mass_matrix_full_gpu_mt` (dynamics/mass_matrix.mojo): every
# thread redundantly computes the setup (model/FK-state reads only), then
# rows are striped i = tid, tid+n, ... Per-row arithmetic is the SAME
# `_mm_row_env` helper as the serial kernel -> bit-exact. One
# barrier after the distributed zero-init (the legacy variant relied on
# caller-side barriers; here the kernel is standalone). Grid is exact ->
# no valid_env guards. NOTE: this ports the DENSE full-CRBA schedule, not
# `compute_mass_matrix_treewalk_gpu_mt` — the treewalk is a different
# algorithm (composite-inertia ancestor walk), only float-tolerance-equal
# to the dense serial kernel, so it cannot satisfy the bit-exact gate.
# The treewalk lives in `_mass_matrix_treewalk_fields_mt_kernel` below,
# behind the dispatcher's TREEWALK=True (gated bit-exact vs the LEGACY
# treewalk instead).
def _mass_matrix_fields_mt_kernel[
    DTYPE: DType,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    BATCH: Int,
    N_THREADS: Int,
](
    xquat: Tile[DTYPE, BATCH, NBODY * 4],
    xipos: Tile[DTYPE, BATCH, NBODY * 3],
    subtree_com: Tile[DTYPE, BATCH, NBODY * 3],
    bodies: Tile[DTYPE, NBODY, MODEL_BODY_SIZE],
    joints: Tile[DTYPE, NJOINT, MODEL_JOINT_SIZE],
    cdof: Tile[DTYPE, BATCH, NV * 6],
    M: Tile[DTYPE, BATCH, NV * NV],
):
    var env = Int(block_idx.x)
    var tid = Int(thread_idx.x)

    # Zero M — distributed across threads, published before the row sweep
    # (every entry is overwritten by the sweep; the barrier just orders the
    # dead zero-stores before the live writes).
    for i in range(tid, NV * NV, N_THREADS):
        M[env, i] = 0
    barrier()

    # Setup redundantly per thread (identical values in every thread).
    # ⚠ CONSTRUCTED IN THE KERNEL, never captured from the host — a `Dims`
    # is not `DevicePassable`, and a captured one reads 0 on device (every
    # loop bound collapses and the output comes back zeroed, which compiles).
    var kdims = Dims[nv=NV, nbody=NBODY, njoint=NJOINT]()
    comptime V_CAP = cap[NV]()
    comptime B6_CAP = cap[NBODY]() * 6
    comptime MASK_CAP = cap[NBODY]() * cap[NBODY]()
    var dof_body = Scratch[Int, V_CAP](NV, uninitialized=0)
    var I_world = Scratch[Scalar[DTYPE], B6_CAP](NBODY * 6, uninitialized=0)
    var subtree_mask = Scratch[Bool, MASK_CAP](NBODY * NBODY, False)
    _mm_setup_env[DTYPE](
        env, kdims, xquat, bodies, joints, dof_body, I_world, subtree_mask
    )

    # Each thread handles rows i where i % N_THREADS == tid.
    for i in range(tid, NV, N_THREADS):
        _mm_row_env[DTYPE](
            env, i, kdims, xipos, subtree_com, bodies, cdof, M,
            dof_body, I_world, subtree_mask,
        )


# ── Tree-walk CRBA (cooperative) — legacy PRODUCTION mass matrix ─────────
# Verbatim port of `compute_mass_matrix_treewalk_gpu_mt`
# (dynamics/mass_matrix.mojo:2017): per-DOF row, walk ancestor DOFs, using
# a composite spatial inertia about P = subtree_com[rootid] (O(NV·depth)
# vs the dense O(NV²·NBODY)). Same launch shape as the dense _mt kernel:
# grid=(BATCH,), block=(N_THREADS,) — exact grid, so the legacy valid_env
# guards collapse to unconditional; barrier placement is identical
# (distributed zero -> barrier -> striped row loop -> barrier). Setup
# (dof maps + composite) is per-thread redundant, exactly like legacy.
#
# NOTE on the legacy composite-inertia skip: when the legacy integrator
# selects the treewalk (USE_TREEWALK_MM, rk4_integrator.mojo:1536-1541) it
# SKIPS `compute_composite_inertia_gpu` (legacy step 5) — the treewalk
# builds its own composite in registers and never reads the `crb` slot
# (RNE later overwrites it with cvel). The fields path has NO standalone
# composite-inertia pass at all (the dense fields kernel also builds its
# per-body I_world in registers), so there is nothing to skip here — this
# kernel simply does not do that dead work, mirroring legacy production.
@always_inline
def _mm_treewalk_env[
    # ⚠ FIRST, AND `//` MARKS IT INFERRED-ONLY, exactly as `noslip_pyramidal`
    # declares the one array it writes. This routine WRITES the composite
    # through `comp`, and an inferred `_` origin comes out IMMUTABLE
    # ("expression must be mutable in assignment"). Putting the marker after
    # the explicit parameters instead makes ALL of them inferred-only and every
    # call site stops compiling.
    CO: MutOrigin,
    TO: MutOrigin, //,
    DTYPE: DType,
    N_THREADS: Int,
    GPU: Bool,
    D: DimsLike,
    L_XQUAT: TensorLayout,
    L_XIPOS: TensorLayout,
    L_BODIES: TensorLayout,
    L_JOINTS: TensorLayout,
    L_CDOF: TensorLayout,
    L_M: TensorLayout,
    CMP_AS: AddressSpace = AddressSpace.GENERIC,
](
    env: Int,
    tid: Int,
    dims: D,
    xquat: TileTensor[DTYPE, L_XQUAT, MutAnyOrigin],
    xipos: TileTensor[DTYPE, L_XIPOS, MutAnyOrigin],
    subtree_com: TileTensor[DTYPE, L_XIPOS, MutAnyOrigin],
    bodies: TileTensor[DTYPE, L_BODIES, MutAnyOrigin],
    joints: TileTensor[DTYPE, L_JOINTS, MutAnyOrigin],
    cdof: TileTensor[DTYPE, L_CDOF, MutAnyOrigin],
    M: TileTensor[DTYPE, L_M, MutAnyOrigin],
    # ⚠⚠ THE COMPOSITE INERTIA, OWNED BY THE CALLER AND SHARED ACROSS THE
    # BLOCK ON GPU. It used to be a per-thread `Scratch`, so every one of the
    # `N_THREADS = NV` threads rebuilt the whole `nbody`-body composite —
    # quaternion rotation, parallel-axis shift, ten accumulators each — and
    # then threw it away. Modelled against the measured kernel that redundancy
    # is 93% of the launch at k=13 (119,532 of 128,568 units, reproducing the
    # observed 32x growth from k=0 to k=13 to within 6%), while the dense
    # `nv*nv` zeroing this term was ASSUMED to be about is 5%.
    comp: Pointer[Scalar[DTYPE], CO, address_space=CMP_AS],
    topo: Pointer[Int32, TO, address_space=CMP_AS],
):
    """Tree-walk CRBA for ONE env — O(nv*depth), shared by CPU and GPU.

    ⚠ THE ONLY DIFFERENCE BETWEEN THE TWO TARGETS IS `GPU`, WHICH GATES THE
    BARRIERS. Everything arithmetic is one copy, so the CPU result is
    bit-exact against the cooperative GPU kernel (N_THREADS=1, tid=0 simply
    collapses the striding); it stays float-tolerance-equal, not bit-exact,
    against the DENSE kernels, which sum the same terms in a different order.
    """

    var nv = dims.get_nv()
    var nbody = dims.get_nbody()
    var njoint = dims.get_njoint()
    # ── The dof topology, built COOPERATIVELY into `topo` (2026-09-08).
    # `topo` holds four int tables back to back: dof -> body [nv], dof ->
    # parent dof [nv], body -> first dof [nbody], body -> last dof [nbody].
    # They are model constants, and this body used to rebuild them on EVERY
    # thread: a loop over all joints (three global loads each — 97 joints at
    # k=13) and a parent walk per dof, before any arithmetic, in a kernel
    # that runs a quarter of a wave and so pays its per-block latency in
    # full (PERFORMANCE.md §13.46). The same rule, split by what can be
    # written without a race: a thread per JOINT writes the dofs it owns;
    # a thread per BODY scans that map for its first and last dof (the
    # serial min/max over the body's joints, read off the map instead); a
    # thread per DOF walks its parent. `N_THREADS == 1` runs the same code
    # serially, so the CPU leg produces the same tables and the same `M`.
    var t_body = 0
    var t_parent = nv
    var t_first = 2 * nv
    var t_last = 2 * nv + nbody
    for d in range(tid, nv, N_THREADS):
        topo[unsafe_offset=t_body + d] = 0
        topo[unsafe_offset=t_parent + d] = -1
    for b in range(tid, nbody, N_THREADS):
        topo[unsafe_offset=t_first + b] = -1
        topo[unsafe_offset=t_last + b] = -1
    comptime if GPU:
        barrier()
    for j in range(tid, njoint, N_THREADS):
        var jb = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_BODY_ID]))
        var dadr = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_DOF_ADR]))
        var jt = Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_TYPE]))
        var ndof = 1
        if jt == JNT_FREE:
            ndof = 6
        elif jt == JNT_BALL:
            ndof = 3
        for d in range(ndof):
            topo[unsafe_offset=t_body + dadr + d] = Int32(jb)
    comptime if GPU:
        barrier()
    for b in range(tid, nbody, N_THREADS):
        var f = -1
        var l = -1
        for d in range(nv):
            if Int(topo[unsafe_offset=t_body + d]) == b:
                if f < 0:
                    f = d
                l = d
        topo[unsafe_offset=t_first + b] = Int32(f)
        topo[unsafe_offset=t_last + b] = Int32(l)
    comptime if GPU:
        barrier()
    for d in range(tid, nv, N_THREADS):
        var b = Int(topo[unsafe_offset=t_body + d])
        if d > Int(topo[unsafe_offset=t_first + b]):
            topo[unsafe_offset=t_parent + d] = Int32(d - 1)
        else:
            var p = Int(rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_PARENT]))
            while p > 0:
                if Int(topo[unsafe_offset=t_last + p]) >= 0:
                    topo[unsafe_offset=t_parent + d] = topo[
                        unsafe_offset=t_last + p
                    ]
                    break
                p = Int(rebind[Scalar[DTYPE]](bodies[p, BODY_IDX_PARENT]))
    comptime if GPU:
        barrier()

    # --- per-body composite contribution about P = stcom[rootid] ---
    # ⚠ STRIDED, AND BIT-EXACT BECAUSE IT IS. Each `b` writes only its own ten
    # slots and reads nothing another `b` wrote, so distributing the loop
    # changes which thread does the work and nothing else. On CPU
    # `N_THREADS == 1` and `tid == 0`, so this reads `range(nbody)` exactly.
    for b in range(tid, nbody, N_THREADS):
        var mass = rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_MASS])
        # rotated body inertia (world-aligned, about body COM)
        var Ixx_l = rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_IXX])
        var Iyy_l = rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_IYY])
        var Izz_l = rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_IZZ])
        var bqx = rebind[Scalar[DTYPE]](xquat[env, b * 4 + 0])
        var bqy = rebind[Scalar[DTYPE]](xquat[env, b * 4 + 1])
        var bqz = rebind[Scalar[DTYPE]](xquat[env, b * 4 + 2])
        var bqw = rebind[Scalar[DTYPE]](xquat[env, b * 4 + 3])
        var iqx = rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_IQUAT_X])
        var iqy = rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_IQUAT_Y])
        var iqz = rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_IQUAT_Z])
        var iqw = rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_IQUAT_W])
        var iq = gpu_quat_mul(bqx, bqy, bqz, bqw, iqx, iqy, iqz, iqw)
        var qx = iq[0]
        var qy = iq[1]
        var qz = iq[2]
        var qw = iq[3]
        var r00 = Scalar[DTYPE](1) - Scalar[DTYPE](2) * (qy * qy + qz * qz)
        var r10 = Scalar[DTYPE](2) * (qx * qy + qw * qz)
        var r20 = Scalar[DTYPE](2) * (qx * qz - qw * qy)
        var r01 = Scalar[DTYPE](2) * (qx * qy - qw * qz)
        var r11 = Scalar[DTYPE](1) - Scalar[DTYPE](2) * (qx * qx + qz * qz)
        var r21 = Scalar[DTYPE](2) * (qy * qz + qw * qx)
        var r02 = Scalar[DTYPE](2) * (qx * qz + qw * qy)
        var r12 = Scalar[DTYPE](2) * (qy * qz - qw * qx)
        var r22 = Scalar[DTYPE](1) - Scalar[DTYPE](2) * (qx * qx + qy * qy)
        var Iw_xx = Ixx_l * r00 * r00 + Iyy_l * r01 * r01 + Izz_l * r02 * r02
        var Iw_yy = Ixx_l * r10 * r10 + Iyy_l * r11 * r11 + Izz_l * r12 * r12
        var Iw_zz = Ixx_l * r20 * r20 + Iyy_l * r21 * r21 + Izz_l * r22 * r22
        var Iw_xy = Ixx_l * r00 * r10 + Iyy_l * r01 * r11 + Izz_l * r02 * r12
        var Iw_xz = Ixx_l * r00 * r20 + Iyy_l * r01 * r21 + Izz_l * r02 * r22
        var Iw_yz = Ixx_l * r10 * r20 + Iyy_l * r11 * r21 + Izz_l * r12 * r22
        # d = xipos[b] - stcom[rootid[b]]
        var rootb = Int(rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_ROOTID]))
        var dx = rebind[Scalar[DTYPE]](
            xipos[env, b * 3 + 0]
        ) - rebind[Scalar[DTYPE]](subtree_com[env, rootb * 3 + 0])
        var dy = rebind[Scalar[DTYPE]](
            xipos[env, b * 3 + 1]
        ) - rebind[Scalar[DTYPE]](subtree_com[env, rootb * 3 + 1])
        var dz = rebind[Scalar[DTYPE]](
            xipos[env, b * 3 + 2]
        ) - rebind[Scalar[DTYPE]](subtree_com[env, rootb * 3 + 2])
        var dd = dx * dx + dy * dy + dz * dz
        comp[unsafe_offset=b * 10 + 0] = mass
        comp[unsafe_offset=b * 10 + 1] = mass * dx
        comp[unsafe_offset=b * 10 + 2] = mass * dy
        comp[unsafe_offset=b * 10 + 3] = mass * dz
        comp[unsafe_offset=b * 10 + 4] = Iw_xx + mass * (dd - dx * dx)
        comp[unsafe_offset=b * 10 + 5] = Iw_yy + mass * (dd - dy * dy)
        comp[unsafe_offset=b * 10 + 6] = Iw_zz + mass * (dd - dz * dz)
        comp[unsafe_offset=b * 10 + 7] = Iw_xy - mass * dx * dy
        comp[unsafe_offset=b * 10 + 8] = Iw_xz - mass * dx * dz
        comp[unsafe_offset=b * 10 + 9] = Iw_yz - mass * dy * dz

    comptime if GPU:
        barrier()

    # leaf→root accumulate (common P within a tree → additive; stop at roots)
    #
    # ⚠⚠ SERIAL ON THREAD 0, AND IT MUST STAY THAT WAY. `comp[parent] +=
    # comp[child]` walks bodies in DESCENDING index order and a parent
    # accumulates several children, so the summation order is fixed by the
    # loop. Distributing it would reorder a floating-point reduction and break
    # the CPU/GPU bit-exactness this function's docstring promises — for
    # O(nbody) work that is nothing beside the build above.
    if tid == 0:
        for b in range(nbody - 1, 0, -1):
            var p = Int(rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_PARENT]))
            if p > 0:
                for e in range(10):
                    comp[unsafe_offset=p * 10 + e] = (
                        comp[unsafe_offset=p * 10 + e]
                        + comp[unsafe_offset=b * 10 + e]
                    )
    comptime if GPU:
        barrier()

    # zero M (distributed)
    for idx in range(tid, nv * nv, N_THREADS):
        M[env, idx] = Scalar[DTYPE](0)
    comptime if GPU:
        barrier()

    # per-DOF row, distributed: f_i = comp[body_i]·cdof_i, walk ancestor DOFs
    for i in range(tid, nv, N_THREADS):
        var bi = Int(topo[unsafe_offset=t_body + i])
        var ai0 = cdof[env, i * 6 + 0]
        var ai1 = cdof[env, i * 6 + 1]
        var ai2 = cdof[env, i * 6 + 2]
        var li0 = cdof[env, i * 6 + 3]
        var li1 = cdof[env, i * 6 + 4]
        var li2 = cdof[env, i * 6 + 5]
        var Mc = comp[unsafe_offset=bi * 10 + 0]
        var hx = comp[unsafe_offset=bi * 10 + 1]
        var hy = comp[unsafe_offset=bi * 10 + 2]
        var hz = comp[unsafe_offset=bi * 10 + 3]
        var Cxx = comp[unsafe_offset=bi * 10 + 4]
        var Cyy = comp[unsafe_offset=bi * 10 + 5]
        var Czz = comp[unsafe_offset=bi * 10 + 6]
        var Cxy = comp[unsafe_offset=bi * 10 + 7]
        var Cxz = comp[unsafe_offset=bi * 10 + 8]
        var Cyz = comp[unsafe_offset=bi * 10 + 9]
        # f_ang = Ic_rot·a_i + hc×l_i
        var fa0 = Cxx * ai0 + Cxy * ai1 + Cxz * ai2 + (hy * li2 - hz * li1)
        var fa1 = Cxy * ai0 + Cyy * ai1 + Cyz * ai2 + (hz * li0 - hx * li2)
        var fa2 = Cxz * ai0 + Cyz * ai1 + Czz * ai2 + (hx * li1 - hy * li0)
        # f_lin = Mc·l_i + a_i×hc
        var fl0 = Mc * li0 + (ai1 * hz - ai2 * hy)
        var fl1 = Mc * li1 + (ai2 * hx - ai0 * hz)
        var fl2 = Mc * li2 + (ai0 * hy - ai1 * hx)
        var j = i
        while j >= 0:
            var aj0 = cdof[env, j * 6 + 0]
            var aj1 = cdof[env, j * 6 + 1]
            var aj2 = cdof[env, j * 6 + 2]
            var lj0 = cdof[env, j * 6 + 3]
            var lj1 = cdof[env, j * 6 + 4]
            var lj2 = cdof[env, j * 6 + 5]
            var mij = (
                aj0 * fa0 + aj1 * fa1 + aj2 * fa2
                + lj0 * fl0 + lj1 * fl1 + lj2 * fl2
            )
            M[env, i * nv + j] = mij
            if i != j:
                M[env, j * nv + i] = mij
            j = Int(topo[unsafe_offset=t_parent + j])
    comptime if GPU:
        barrier()


def _mass_matrix_treewalk_fields_mt_kernel[
    DTYPE: DType,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    BATCH: Int,
    N_THREADS: Int,
](
    xquat: Tile[DTYPE, BATCH, NBODY * 4],
    xipos: Tile[DTYPE, BATCH, NBODY * 3],
    subtree_com: Tile[DTYPE, BATCH, NBODY * 3],
    bodies: Tile[DTYPE, NBODY, MODEL_BODY_SIZE],
    joints: Tile[DTYPE, NJOINT, MODEL_JOINT_SIZE],
    cdof: Tile[DTYPE, BATCH, NV * 6],
    M: Tile[DTYPE, BATCH, NV * NV],
):
    # ⚠ ONE COMPOSITE INERTIA PER BLOCK, not one per thread. `NBODY*10`
    # scalars — 880 B at k=13 against this kernel's otherwise trivial
    # threadgroup use, and it replaces `N_THREADS` private copies of the same
    # array. Allocated HERE rather than inside `_mm_treewalk_env` because
    # shared memory belongs to the kernel scope; the env function takes a
    # pointer so the CPU leg can hand it a `Scratch` instead.
    var comp_sh = shared_tile[DTYPE, NBODY * 10]()
    var topo_sh = shared_tile[DType.int32, 2 * NV + 2 * NBODY]()
    _mm_treewalk_env[DTYPE, N_THREADS, True, CMP_AS = AddressSpace.SHARED](
        Int(block_idx.x), Int(thread_idx.x),
        Dims[nv=NV, nbody=NBODY, njoint=NJOINT](),
        xquat, xipos, subtree_com, bodies, joints, cdof, M, comp_sh.ptr,
        topo_sh.ptr,
    )


def _mass_matrix_fields_kernel[
    DTYPE: DType,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    BATCH: Int,
](
    xquat: Tile[DTYPE, BATCH, NBODY * 4],
    xipos: Tile[DTYPE, BATCH, NBODY * 3],
    subtree_com: Tile[DTYPE, BATCH, NBODY * 3],
    bodies: Tile[DTYPE, NBODY, MODEL_BODY_SIZE],
    joints: Tile[DTYPE, NJOINT, MODEL_JOINT_SIZE],
    cdof: Tile[DTYPE, BATCH, NV * 6],
    M: Tile[DTYPE, BATCH, NV * NV],
):
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    _mass_matrix_env[DTYPE](
        env, Dims[nv=NV, nbody=NBODY, njoint=NJOINT](), xquat, xipos, subtree_com, bodies, joints, cdof, M
    )


def compute_mass_matrix[

    target: StaticString,
    DTYPE: DType,
    D: DimsLike,
    BATCH: Int = 1,
    PARALLEL: Bool = False,
    TREEWALK: Bool = False,
    # Appended, not grouped with NEXCLUDE — see `fields.Model`.
](
    mut d: Data[DTYPE, D, BATCH],
    mut m: Model[DTYPE, D],
    mut scratch: DynamicsScratch[DTYPE, D, BATCH],
    ctx: Optional[DeviceContext] = None,
) raises:
    """CRBA mass matrix from FK products + cdof, both targets, one body.
    Reads `scratch.cdof`, writes `scratch.M`. PARALLEL=True (GPU only):
    cooperative row-striped kernel, bit-exact vs serial. CPU ignores
    PARALLEL. TREEWALK=True: the TREE-WALK CRBA (O(NV·depth)) —
    float-tolerance-equal to the dense kernels, NOT bit-exact vs them.

    ⚠ THE DENSE KERNEL IS O(NV²·NBODY) AND THAT IS NOT A CONSTANT FACTOR.
    It evaluates every (i, j) dof pair against every body through a subtree
    mask, so it costs 4 080 inner iterations on Sawyer (NV=15, NBODY=34)
    where the treewalk costs ~110 — measured at **5.6 µs vs 0.44 µs**, i.e.
    12.6× — against MuJoCo's `mj_crb` at 0.81 µs. On SO-ARM100 (NV=6,
    NBODY=8) the same gap is worth 0.3 µs, which is why it hid for so long:
    **this defect is invisible on small models and grows as NV·NBODY.**

    ⚠ TREEWALK USED TO REQUIRE PARALLEL, so every CPU caller silently got
    the dense kernel — the viewer, every test and every single-env rollout.
    The requirement was never real: the cooperative kernel's only
    parallelism is two strided loops, and N_THREADS=1 collapses them."""
    comptime L_B3 = RowMajor[BATCH, D.NBODY * 3]
    comptime L_B4 = RowMajor[BATCH, D.NBODY * 4]
    comptime L_BODY = RowMajor[D.NBODY, MODEL_BODY_SIZE]
    comptime L_JOINT = RowMajor[D.NJOINT, MODEL_JOINT_SIZE]
    comptime L_CDOF = RowMajor[BATCH, D.NV * 6]
    comptime L_M = RowMajor[BATCH, D.NV * D.NV]

    comptime if target == "cpu":
        var dm = d.dims
        var rl_B4 = rl2(BATCH, dm.get_nbody() * 4)
        var rl_B3 = rl2(BATCH, dm.get_nbody() * 3)
        var rl_BODY = rl2(dm.get_nbody(), MODEL_BODY_SIZE)
        var rl_JOINT = rl2(dm.get_njoint(), MODEL_JOINT_SIZE)
        var rl_CDOF = rl2(BATCH, dm.get_nv() * 6)
        var rl_M = rl2(BATCH, dm.get_nv() * dm.get_nv())
        var xquat_v = d.xquat.tt_dyn["cpu", DYN2](rl_B4)
        var xipos_v = d.xipos.tt_dyn["cpu", DYN2](rl_B3)
        var stcom_v = d.subtree_com.tt_dyn["cpu", DYN2](rl_B3)
        var bodies_v = m.bodies.tt_dyn["cpu", DYN2](rl_BODY)
        var joints_v = m.joints.tt_dyn["cpu", DYN2](rl_JOINT)
        var cdof_v = scratch.cdof.tt_dyn["cpu", DYN2](rl_CDOF)
        var M_v = scratch.M.tt_dyn["cpu", DYN2](rl_M)
        comptime if TREEWALK:
            # The CPU twin of the kernel's shared array. `N_THREADS == 1`, so
            # one buffer serves the single "thread"; hoisted out of the env
            # loop because it is overwritten wholesale each time anyway.
            var comp_s = Scratch[Scalar[DTYPE], cap[D.NBODY]() * 10](
                dm.get_nbody() * 10, 0
            )
            var topo_s = Scratch[Int32, 2 * cap[D.NV]() + 2 * cap[D.NBODY]()](
                2 * dm.get_nv() + 2 * dm.get_nbody(), 0
            )
            for e in range(BATCH):
                _mm_treewalk_env[DTYPE, 1, False](
                    e, 0,
                    dm,
                    xquat_v, xipos_v, stcom_v, bodies_v, joints_v, cdof_v, M_v,
                    comp_s.unsafe_ptr(),
                    topo_s.unsafe_ptr(),
                )
        else:
            for e in range(BATCH):
                _mass_matrix_env[DTYPE](
                    e, dm, xquat_v, xipos_v, stcom_v, bodies_v, joints_v, cdof_v,
                    M_v,
                )
    elif PARALLEL and TREEWALK:
        var c = ctx.value()
        comptime MT_T = D.NV
        c.enqueue_function[
            _mass_matrix_treewalk_fields_mt_kernel[
                DTYPE, D.NV, D.NBODY, D.NJOINT, BATCH, MT_T
            ]
        ](
            d.xquat.tt["gpu", BATCH, D.NBODY * 4](),
            d.xipos.tt["gpu", BATCH, D.NBODY * 3](),
            d.subtree_com.tt["gpu", BATCH, D.NBODY * 3](),
            m.bodies.tt["gpu", D.NBODY, MODEL_BODY_SIZE](),
            m.joints.tt["gpu", D.NJOINT, MODEL_JOINT_SIZE](),
            scratch.cdof.tt["gpu", BATCH, D.NV * 6](),
            scratch.M.tt["gpu", BATCH, D.NV * D.NV](),
            grid_dim=(BATCH,),
            block_dim=(MT_T,),
        )
    elif PARALLEL:
        var c = ctx.value()
        comptime MT_T = D.NV
        c.enqueue_function[
            _mass_matrix_fields_mt_kernel[
                DTYPE, D.NV, D.NBODY, D.NJOINT, BATCH, MT_T
            ]
        ](
            d.xquat.tt["gpu", BATCH, D.NBODY * 4](),
            d.xipos.tt["gpu", BATCH, D.NBODY * 3](),
            d.subtree_com.tt["gpu", BATCH, D.NBODY * 3](),
            m.bodies.tt["gpu", D.NBODY, MODEL_BODY_SIZE](),
            m.joints.tt["gpu", D.NJOINT, MODEL_JOINT_SIZE](),
            scratch.cdof.tt["gpu", BATCH, D.NV * 6](),
            scratch.M.tt["gpu", BATCH, D.NV * D.NV](),
            grid_dim=(BATCH,),
            block_dim=(MT_T,),
        )
    else:
        var c = ctx.value()
        comptime BLOCKS = (BATCH + MM_TPB - 1) // MM_TPB
        c.enqueue_function[
            _mass_matrix_fields_kernel[DTYPE, D.NV, D.NBODY, D.NJOINT, BATCH]
        ](
            d.xquat.tt["gpu", BATCH, D.NBODY * 4](),
            d.xipos.tt["gpu", BATCH, D.NBODY * 3](),
            d.subtree_com.tt["gpu", BATCH, D.NBODY * 3](),
            m.bodies.tt["gpu", D.NBODY, MODEL_BODY_SIZE](),
            m.joints.tt["gpu", D.NJOINT, MODEL_JOINT_SIZE](),
            scratch.cdof.tt["gpu", BATCH, D.NV * 6](),
            scratch.M.tt["gpu", BATCH, D.NV * D.NV](),
            grid_dim=(BLOCKS,),
            block_dim=(MM_TPB,),
        )
