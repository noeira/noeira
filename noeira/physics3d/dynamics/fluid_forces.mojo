"""Passive fluid forces over per-field tensors (Stage-A, single-source).

Per-field port of `compute_fluid_forces` (dynamics/fluid_forces.mojo) — the
MuJoCo inertia-box fluid model (mj_inertiaBoxFluidModel, engine_passive.c):
per non-world body, approximate the shape as an equivalent box from the
diagonal inertia, compute viscous (Stokes) + pressure (quadratic) drag in the
body local frame, rotate back to world, transport the wrench to
subtree_com[rootid], and accumulate into `scratch.fnet` via Jᵀ (walking the
kinematic tree). Arithmetic verbatim from the legacy CPU routine.

Enabled only when meta density > 0 OR viscosity > 0 (early-out otherwise);
in this codebase that is the Swimmer environment (density=4000, viscosity=0.1).
Every other env has both zero, so this is a no-op there.

Inserted in the integrator passive seam right after fnet assembly
(qfrc - bias - damping - stiffness - frictionloss) and before the LDL solve,
matching the legacy euler/rk4/implicit integrators (`compute_fluid_forces`
call site, into f_net). The fields path already has all inputs current at the
seam: d.xvel/xangvel/xquat/xipos/subtree_com, m.bodies/joints/meta,
scratch.cdof, scratch.fnet.

Structural transformation (same as the other fields ports): serialized per
env. Operands (10): xvel, xangvel, xquat, xipos, subtree_com (data) + bodies,
joints, meta (model) + cdof, fnet (scratch)."""

from std.math import sqrt, abs
from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceContext
from layout import TileTensor, TensorLayout
from noeira.core.tiles import RowMajor, Tile

from ..kinematics.quat_math import quat_rotate, quat_mul
from ..joint_types import JNT_FREE, JNT_BALL
from ..fields import (
    Data,
    Model,
    DynamicsScratch,
    Dims,
    DimsLike,
    AsStatic,
    DYN1,
    DYN2,
    rl1,
    rl2,
)
from ..gpu.constants import (
    MODEL_BODY_SIZE,
    MODEL_JOINT_SIZE,
    MODEL_META_SIZE,
    MODEL_META_IDX_DENSITY,
    MODEL_META_IDX_VISCOSITY,
    MODEL_META_IDX_WIND_X,
    MODEL_META_IDX_WIND_Y,
    MODEL_META_IDX_WIND_Z,
    BODY_IDX_MASS,
    BODY_IDX_IXX,
    BODY_IDX_IYY,
    BODY_IDX_IZZ,
    BODY_IDX_IQUAT_X,
    BODY_IDX_IQUAT_Y,
    BODY_IDX_IQUAT_Z,
    BODY_IDX_IQUAT_W,
    BODY_IDX_PARENT,
    BODY_IDX_ROOTID,
    JOINT_IDX_TYPE,
    JOINT_IDX_BODY_ID,
    JOINT_IDX_DOF_ADR,
)

comptime FLUID_TPB: Int = 64


def _fluid_forces_env[
    DTYPE: DType,
    D: DimsLike,
    L_XVEL: TensorLayout,
    L_XQUAT: TensorLayout,
    L_BODIES: TensorLayout,
    L_JOINTS: TensorLayout,
    L_MMETA: TensorLayout,
    L_CDOF: TensorLayout,
    L_FNET: TensorLayout,
](
    env: Int,
    dims: D,
    xvel: TileTensor[DTYPE, L_XVEL, MutAnyOrigin],
    xangvel: TileTensor[DTYPE, L_XVEL, MutAnyOrigin],
    xquat: TileTensor[DTYPE, L_XQUAT, MutAnyOrigin],
    xipos: TileTensor[DTYPE, L_XVEL, MutAnyOrigin],
    subtree_com: TileTensor[DTYPE, L_XVEL, MutAnyOrigin],
    bodies: TileTensor[DTYPE, L_BODIES, MutAnyOrigin],
    joints: TileTensor[DTYPE, L_JOINTS, MutAnyOrigin],
    mmeta: TileTensor[DTYPE, L_MMETA, MutAnyOrigin],
    cdof: TileTensor[DTYPE, L_CDOF, MutAnyOrigin],
    fnet: TileTensor[DTYPE, L_FNET, MutAnyOrigin],
):
    """Inertia-box fluid drag for one env (verbatim from compute_fluid_forces,
    serialized per env)."""
    var nbody = dims.get_nbody()
    var njoint = dims.get_njoint()
    var rho = rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_DENSITY])
    var mu = rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_VISCOSITY])
    var wind_x = rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_WIND_X])
    var wind_y = rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_WIND_Y])
    var wind_z = rebind[Scalar[DTYPE]](mmeta[MODEL_META_IDX_WIND_Z])

    # Early-out: no fluid forces when both density and viscosity are zero
    # ⚠ AND THE WIND DOES NOT CHANGE THAT. `mj_passive` calls neither fluid
    # model unless `density` or `viscosity` is non-zero
    # (engine_passive.c), so `<option wind>` in a vacuum is inert in MuJoCo
    # too — it is a fluid VELOCITY, and a fluid with no density and no
    # viscosity exerts nothing however fast it moves.
    if rho <= Scalar[DTYPE](0) and mu <= Scalar[DTYPE](0):
        return

    comptime PI: Scalar[DTYPE] = 3.14159265358979323846

    for b in range(1, nbody):
        var mass = rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_MASS])
        if mass <= Scalar[DTYPE](1e-10):
            continue

        # --- 1. Equivalent box dimensions from diagonal inertia ---
        var Ixx = rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_IXX])
        var Iyy = rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_IYY])
        var Izz = rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_IZZ])

        var bx2 = Scalar[DTYPE](6) * (Iyy + Izz - Ixx) / mass
        var by2 = Scalar[DTYPE](6) * (Ixx + Izz - Iyy) / mass
        var bz2 = Scalar[DTYPE](6) * (Ixx + Iyy - Izz) / mass

        var bx = sqrt(max(bx2, Scalar[DTYPE](0)))
        var by = sqrt(max(by2, Scalar[DTYPE](0)))
        var bz = sqrt(max(bz2, Scalar[DTYPE](0)))

        # --- 2. Body world velocity (at body frame origin) ---
        var vx_w = rebind[Scalar[DTYPE]](xvel[env, b * 3 + 0])
        var vy_w = rebind[Scalar[DTYPE]](xvel[env, b * 3 + 1])
        var vz_w = rebind[Scalar[DTYPE]](xvel[env, b * 3 + 2])
        var wx_w = rebind[Scalar[DTYPE]](xangvel[env, b * 3 + 0])
        var wy_w = rebind[Scalar[DTYPE]](xangvel[env, b * 3 + 1])
        var wz_w = rebind[Scalar[DTYPE]](xangvel[env, b * 3 + 2])

        # --- 3. Rotate velocity into the INERTIAL frame (conjugate quat) ---
        #
        # ⚠ `ximat`, NOT `xmat`. `mj_inertiaBoxFluidModel` builds the box from
        # `body_inertia`, which is the diagonal of the inertia tensor in the
        # body's INERTIAL frame (`xquat ⊗ iquat`), and reads the velocity in
        # that same frame (`mj_objectVelocity(..., flg_local=1)` uses
        # `ximat`). `BODY_IDX_IXX..IZZ` are those principal moments, so the
        # box's x face is the inertial frame's x face. Rotating the velocity
        # into the BODY frame instead pairs each face with the wrong velocity
        # component whenever `iquat` is not the identity — a `fromto` capsule
        # along the body's x axis has its principal z along that axis. The
        # viscous term is symmetric (one `diam`) and never noticed; the
        # pressure-drag term was 0.18% off on Gymnasium's swimmer at step 1
        # and 5.6e-2 in qpos by step 50 (bigsweep, 2026-09-06). Velocity-only
        # and density-only ablations pinned it to the density term.
        var bqx = rebind[Scalar[DTYPE]](xquat[env, b * 4 + 0])
        var bqy = rebind[Scalar[DTYPE]](xquat[env, b * 4 + 1])
        var bqz = rebind[Scalar[DTYPE]](xquat[env, b * 4 + 2])
        var bqw = rebind[Scalar[DTYPE]](xquat[env, b * 4 + 3])
        var iqx = rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_IQUAT_X])
        var iqy = rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_IQUAT_Y])
        var iqz = rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_IQUAT_Z])
        var iqw = rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_IQUAT_W])
        var qi = quat_mul[DTYPE](bqx, bqy, bqz, bqw, iqx, iqy, iqz, iqw)
        var qx = qi[0]
        var qy = qi[1]
        var qz = qi[2]
        var qw = qi[3]

        var vloc = quat_rotate[DTYPE](-qx, -qy, -qz, qw, vx_w, vy_w, vz_w)
        var wloc = quat_rotate[DTYPE](-qx, -qy, -qz, qw, wx_w, wy_w, wz_w)
        var vx = vloc[0]
        var vy = vloc[1]
        var vz = vloc[2]
        var wx = wloc[0]
        var wy = wloc[1]
        var wz = wloc[2]

        # --- 3b. Subtract the wind (AUD-27) --------------------------------
        #
        # ⚠ EVERY DRAG TERM BELOW IS A FUNCTION OF THE VELOCITY RELATIVE TO
        # THE FLUID, and until now it was handed the velocity relative to the
        # WORLD. MuJoCo builds a 6-vector `wind = [0, 0, 0, wx, wy, wz]`,
        # rotates it into this same inertial frame and does
        # `lvel[3..5] -= lwind[3..5]` (engine_passive.c:1167-1174, and the
        # identical block at :1239-1248 in the ellipsoid model). The rotation
        # is the SAME conjugate quaternion the velocity above just used, and
        # `mju_transformSpatial`'s position terms drop out because the wind's
        # angular half is zero — a pure rotation is the whole transform.
        #
        # ⚠ ANGULAR VELOCITY IS UNTOUCHED. A uniform wind is a translation of
        # the fluid, so it shifts the linear velocity and leaves the spin
        # alone; MuJoCo subtracts from `lvel+3` only. Subtracting it from
        # `wx/wy/wz` here would be dimensionally wrong as well as wrong.
        if (
            wind_x != Scalar[DTYPE](0)
            or wind_y != Scalar[DTYPE](0)
            or wind_z != Scalar[DTYPE](0)
        ):
            var wnd = quat_rotate[DTYPE](
                -qx, -qy, -qz, qw, wind_x, wind_y, wind_z
            )
            vx -= wnd[0]
            vy -= wnd[1]
            vz -= wnd[2]

        # --- 4. Equivalent sphere diameter for Stokes drag ---
        var diam = (bx + by + bz) / Scalar[DTYPE](3)

        # --- 5. Accumulate local-frame forces and torques ---
        var lfx = Scalar[DTYPE](0)
        var lfy = Scalar[DTYPE](0)
        var lfz = Scalar[DTYPE](0)
        var ltx = Scalar[DTYPE](0)
        var lty = Scalar[DTYPE](0)
        var ltz = Scalar[DTYPE](0)

        # Viscous (Stokes) drag — linear in velocity
        if mu > Scalar[DTYPE](0):
            var visc_lin = Scalar[DTYPE](3) * PI * diam * mu
            lfx = lfx - visc_lin * vx
            lfy = lfy - visc_lin * vy
            lfz = lfz - visc_lin * vz

            var d3 = diam * diam * diam
            var visc_ang = PI * d3 * mu
            ltx = ltx - visc_ang * wx
            lty = lty - visc_ang * wy
            ltz = ltz - visc_ang * wz

        # Pressure (quadratic) drag
        if rho > Scalar[DTYPE](0):
            var half_rho = Scalar[DTYPE](0.5) * rho
            lfx = lfx - half_rho * by * bz * abs(vx) * vx
            lfy = lfy - half_rho * bx * bz * abs(vy) * vy
            lfz = lfz - half_rho * bx * by * abs(vz) * vz

            var bx4 = bx * bx * bx * bx
            var by4 = by * by * by * by
            var bz4 = bz * bz * bz * bz
            ltx = ltx - rho * bx * (by4 + bz4) * abs(wx) * wx / Scalar[DTYPE](
                64
            )
            lty = lty - rho * by * (bx4 + bz4) * abs(wy) * wy / Scalar[DTYPE](
                64
            )
            ltz = ltz - rho * bz * (bx4 + by4) * abs(wz) * wz / Scalar[DTYPE](
                64
            )

        # --- 6. Rotate forces/torques back to world frame ---
        var fw = quat_rotate[DTYPE](qx, qy, qz, qw, lfx, lfy, lfz)
        var tw = quat_rotate[DTYPE](qx, qy, qz, qw, ltx, lty, ltz)
        var fx_w = fw[0]
        var fy_w = fw[1]
        var fz_w = fw[2]
        var tx_w = tw[0]
        var ty_w = tw[1]
        var tz_w = tw[2]

        # --- 7. Apply wrench at xipos via Jacobian transpose ---
        var px = rebind[Scalar[DTYPE]](xipos[env, b * 3 + 0])
        var py = rebind[Scalar[DTYPE]](xipos[env, b * 3 + 1])
        var pz = rebind[Scalar[DTYPE]](xipos[env, b * 3 + 2])

        # subtree_com reference point for this body's root (fields path always
        # has subtree_com computed — the legacy has_stcom branch)
        var root = Int(rebind[Scalar[DTYPE]](bodies[b, BODY_IDX_ROOTID]))
        var ref_x = rebind[Scalar[DTYPE]](subtree_com[env, root * 3 + 0])
        var ref_y = rebind[Scalar[DTYPE]](subtree_com[env, root * 3 + 1])
        var ref_z = rebind[Scalar[DTYPE]](subtree_com[env, root * 3 + 2])

        # Transport torque from xipos to subtree_com reference
        var dx = px - ref_x
        var dy = py - ref_y
        var dz = pz - ref_z
        var tau_ox = tx_w + dy * fz_w - dz * fy_w
        var tau_oy = ty_w + dz * fx_w - dx * fz_w
        var tau_oz = tz_w + dx * fy_w - dy * fx_w

        # Walk the kinematic tree from body b to root, accumulating via cdof
        var body = b
        while body > 0:
            for j in range(njoint):
                if Int(rebind[Scalar[DTYPE]](joints[j, JOINT_IDX_BODY_ID])) != (
                    body
                ):
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

                for d in range(ndof):
                    var di = dof_adr + d
                    # cdof layout per DOF: [ang_x, ang_y, ang_z, lin_x, lin_y, lin_z]
                    var ca0 = rebind[Scalar[DTYPE]](cdof[env, di * 6 + 0])
                    var ca1 = rebind[Scalar[DTYPE]](cdof[env, di * 6 + 1])
                    var ca2 = rebind[Scalar[DTYPE]](cdof[env, di * 6 + 2])
                    var cl0 = rebind[Scalar[DTYPE]](cdof[env, di * 6 + 3])
                    var cl1 = rebind[Scalar[DTYPE]](cdof[env, di * 6 + 4])
                    var cl2 = rebind[Scalar[DTYPE]](cdof[env, di * 6 + 5])
                    fnet[env, di] = (
                        rebind[Scalar[DTYPE]](fnet[env, di])
                        + cl0 * fx_w
                        + cl1 * fy_w
                        + cl2 * fz_w
                        + ca0 * tau_ox
                        + ca1 * tau_oy
                        + ca2 * tau_oz
                    )

            body = Int(rebind[Scalar[DTYPE]](bodies[body, BODY_IDX_PARENT]))


def _fluid_forces_fields_kernel[
    DTYPE: DType,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    BATCH: Int,
](
    xvel: Tile[DTYPE, BATCH, NBODY * 3],
    xangvel: Tile[DTYPE, BATCH, NBODY * 3],
    xquat: Tile[DTYPE, BATCH, NBODY * 4],
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
    _fluid_forces_env[DTYPE](
        env, Dims[nv=NV, nbody=NBODY, njoint=NJOINT](), xvel, xangvel, xquat, xipos, subtree_com, bodies, joints, mmeta,
        cdof, fnet,
    )


def compute_fluid_forces[

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
    """Accumulate inertia-box fluid drag into `scratch.fnet`, both targets, one
    body. No-op when meta density and viscosity are both zero (early-out inside
    the env helper). Call in the passive seam after fnet assembly, before the
    LDL solve."""
    comptime L_NV = RowMajor[BATCH, D.NV]
    comptime L_B3 = RowMajor[BATCH, D.NBODY * 3]
    comptime L_B4 = RowMajor[BATCH, D.NBODY * 4]
    comptime L_BODY = RowMajor[D.NBODY, MODEL_BODY_SIZE]
    comptime L_JOINT = RowMajor[D.NJOINT, MODEL_JOINT_SIZE]
    comptime L_META = RowMajor[MODEL_META_SIZE]
    comptime L_CDOF = RowMajor[BATCH, D.NV * 6]

    comptime if target == "cpu":
        var dm = d.dims
        var rl_B3 = rl2(BATCH, dm.get_nbody() * 3)
        var rl_B4 = rl2(BATCH, dm.get_nbody() * 4)
        var rl_BODY = rl2(dm.get_nbody(), MODEL_BODY_SIZE)
        var rl_JOINT = rl2(dm.get_njoint(), MODEL_JOINT_SIZE)
        var rl_META = rl1(MODEL_META_SIZE)
        var rl_CDOF = rl2(BATCH, dm.get_nv() * 6)
        var rl_NV = rl2(BATCH, dm.get_nv())
        var xvel_v = d.xvel.tt_dyn["cpu", DYN2](rl_B3)
        var xangvel_v = d.xangvel.tt_dyn["cpu", DYN2](rl_B3)
        var xquat_v = d.xquat.tt_dyn["cpu", DYN2](rl_B4)
        var xipos_v = d.xipos.tt_dyn["cpu", DYN2](rl_B3)
        var stcom_v = d.subtree_com.tt_dyn["cpu", DYN2](rl_B3)
        var bodies_v = m.bodies.tt_dyn["cpu", DYN2](rl_BODY)
        var joints_v = m.joints.tt_dyn["cpu", DYN2](rl_JOINT)
        var meta_v = m.meta.tt_dyn["cpu", DYN1](rl_META)
        var cdof_v = scratch.cdof.tt_dyn["cpu", DYN2](rl_CDOF)
        var fnet_v = scratch.fnet.tt_dyn["cpu", DYN2](rl_NV)
        for e in range(BATCH):
            _fluid_forces_env[DTYPE](
                e, dm, xvel_v, xangvel_v, xquat_v, xipos_v, stcom_v, bodies_v,
                joints_v, meta_v, cdof_v, fnet_v,
            )
    else:
        var c = ctx.value()
        comptime BLOCKS = (BATCH + FLUID_TPB - 1) // FLUID_TPB
        c.enqueue_function[
            _fluid_forces_fields_kernel[DTYPE, D.NV, D.NBODY, D.NJOINT, BATCH]
        ](
            d.xvel.tt["gpu", BATCH, D.NBODY * 3](),
            d.xangvel.tt["gpu", BATCH, D.NBODY * 3](),
            d.xquat.tt["gpu", BATCH, D.NBODY * 4](),
            d.xipos.tt["gpu", BATCH, D.NBODY * 3](),
            d.subtree_com.tt["gpu", BATCH, D.NBODY * 3](),
            m.bodies.tt["gpu", D.NBODY, MODEL_BODY_SIZE](),
            m.joints.tt["gpu", D.NJOINT, MODEL_JOINT_SIZE](),
            m.meta.tt["gpu", MODEL_META_SIZE](),
            scratch.cdof.tt["gpu", BATCH, D.NV * 6](),
            scratch.fnet.tt["gpu", BATCH, D.NV](),
            grid_dim=(BLOCKS,),
            block_dim=(FLUID_TPB,),
        )
