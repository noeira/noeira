"""The `<sensor>` evaluation passes — `mj_sensorPos` / `Vel` / `Acc` (AUD-23).

The six kernels beside this file each compute one sensor, addressed by
`(body, site)` or `(root)`. This is the front end over them: it walks the
model's sensor table, dispatches by `mjtSensor`, writes the values into
`d.sensordata` at each sensor's own `adr`, and applies `cutoff`.

⚠ TWO SENSORS ARE COMPUTED HERE RATHER THAN IN A KERNEL FILE, and that is
deliberate. `jointpos` and `jointvel` are `d->qpos[m->jnt_qposadr[objid]]` and
`d->qvel[m->jnt_dofadr[objid]]` — the whole of the reference's case arms
(`engine_sensor.c:644, 873`). A file per sensor would wrap a subscript in a
call boundary. Their `objid` is a JOINT index, not a site or a body, which is
the only place in this pass where that is true.

⚠⚠ THREE PASSES, BECAUSE MuJoCo HAS THREE. `mj_sensorPos` runs after forward
kinematics, `mj_sensorVel` after the velocity stage, `mj_sensorAcc` after the
constraint solve (`engine_forward.c:1797, 1814, 1832`). A sensor's stage is not
a scheduling detail — it is what decides whether the quantity it reads exists
yet. Collapsing them into one pass would compute the velocimeter from last
step's velocities on every integrator that separates the stages, which is all
of them.

⚠ THE ACCELERATION PASS READS THE `*_acc` SNAPSHOT, NOT THE LIVE FK PRODUCTS.
`site_xpos_acc` / `xquat_acc` are the pre-integration pose (`fields/data.mojo`,
"defect 19"); the live `site_xpos` has already moved by the time the step ends.
Every acceleration-stage kernel here takes the snapshot, matching what the
hand-written env hooks already do.

⚠ CPU, BATCH=1. Every existing CPU sensor caller in the tree is BATCH=1 and
these kernels take host `List`s; the `_gpu` twins take `LayoutTensor`s and want
a kernel of their own rather than a loop over envs. `sensordata` is allocated
`[BATCH, NSENSORDATA]` so the batched pass has somewhere to write when it
lands.

⚠ AN UNSERVED SENSOR IS SKIPPED, AND ITS SLICE STAYS ZERO. It still holds its
`adr`, so every sensor after it lands in the right place — that is the whole
reason `_fill_sensors` gives it a row. Reading one by name raises
(`FlatModelDef._require_served`) rather than handing back zeros that look like
a measurement.
"""

from max.gpu import thread_idx, block_idx, block_dim
from max.gpu.host import DeviceContext
from layout import TileTensor, TensorLayout
from noeira.core.tiles import RowMajor, Tile

from noeira.physics3d.fields import (
    Data, Model, Dims, DimsLike, DYN1, DYN2, rl1, rl2,
)
from noeira.physics3d.types import _max_one
from noeira.physics3d.constants import (
    SENS_TOUCH,
    SENS_ACCELEROMETER,
    SENS_VELOCIMETER,
    SENS_GYRO,
    SENS_FORCE,
    SENS_TORQUE,
    SENS_RANGEFINDER,
    SENS_JOINTPOS,
    SENS_JOINTVEL,
    SENS_TENDONPOS,
    SENS_ACTUATORPOS,
    SENS_JOINTACTFRC,
    SENS_FRAMEPOS,
    SENS_FRAMEQUAT,
    SENS_FRAMEXAXIS,
    SENS_FRAMEYAXIS,
    SENS_FRAMEZAXIS,
    SENS_FRAMELINVEL,
    SENS_FRAMEANGVEL,
    SENS_SUBTREECOM,
    SENSOBJ_UNKNOWN,
    SENS_SUBTREELINVEL,
    SENS_SUBTREEANGMOM,
    SENS_MAGNETOMETER,
    SENSDATA_REAL,
    SENSDATA_POSITIVE,
    SENSSTAGE_POS,
    SENSSTAGE_VEL,
    SENSSTAGE_ACC,
)
from noeira.physics3d.gpu.constants import (
    MODEL_JOINT_SIZE,
    MODEL_BODY_SIZE,
    MODEL_MESH_META_SIZE,
    MAX_GPU_MESHES,
    MESH_ARENA_FLOATS_PER_TRI,
    MODEL_HFIELD_META_SIZE,
    MAX_GPU_HFIELDS,
    MODEL_SITE_SIZE,
    SITE_IDX_QUAT_X,
    SITE_IDX_QUAT_Y,
    SITE_IDX_QUAT_Z,
    SITE_IDX_QUAT_W,
    SENSOR_IDX_MAG_X,
    SENSOR_IDX_MAG_Y,
    SENSOR_IDX_MAG_Z,
    MODEL_GEOM_SIZE,
    CONTACT_SIZE,
    METADATA_SIZE,
    JOINT_IDX_QPOS_ADR,
    JOINT_IDX_DOF_ADR,
    MODEL_SENSOR_SIZE,
    SENSOR_IDX_TYPE,
    SENSOR_IDX_OBJTYPE,
    SENSOR_IDX_OBJID,
    SENSOR_IDX_DIM,
    SENSOR_IDX_ADR,
    SENSOR_IDX_DATATYPE,
    SENSOR_IDX_NEEDSTAGE,
    SENSOR_IDX_CUTOFF,
    SENSOR_IDX_BODY,
    SENSOR_IDX_SERVED,
    SENSOR_IDX_REFTYPE,
    SENSOR_IDX_REFID,
)
from noeira.physics3d.kinematics.quat_math import gpu_quat_rotate
from .frame import (
    frame_object_pose,
    frame_object_body,
    frame_pos_sensor,
    frame_axis_sensor,
    frame_quat_sensor,
    frame_vel_sensor,
)
from .frame_vel import site_frame_velocity_gpu
from .site_acc import site_accelerometer_gpu, site_force_torque_gpu
from .subtree import subtree_linvel_gpu, subtree_angmom_gpu
from .touch import touch_sphere_site_gpu
from .rangefinder import rangefinder_ray


@always_inline
def _atleast1(n: Int) -> Int:
    """A zero-extent LayoutTensor cannot be bound; an optional table floors to
    one row exactly as `Data` allocates it."""
    return n if n > 0 else 1


@always_inline
def _apply_cutoff[
    DTYPE: DType, L_SD: TensorLayout
](
    sensordata: TileTensor[DTYPE, L_SD, MutAnyOrigin],
    env: Int,
    adr: Int,
    dim: Int,
    datatype: Int,
    # ⚠ `Scalar[DTYPE]`, NOT `Float64`. Metal has no `double`: a host-side
    # `Float64` here compiles to an `air.convert.f.f32.f.f64` the Metal
    # verifier rejects outright, which is how the first device build of this
    # pass failed. Every scalar that crosses into the kernel is DTYPE-wide.
    cutoff: Scalar[DTYPE],
):
    """`apply_cutoff` (engine_sensor.c:198-224), verbatim.

    ⚠ `cutoff <= 0` DISABLES CLAMPING; it does not clamp to zero. That is the
    reference's first line and the reason `SensorData.cutoff` is a plain value
    with no `has_cutoff` flag beside it.

    REAL clips to `[-cutoff, +cutoff]`; POSITIVE takes `min(cutoff, x)` and
    leaves the negative side alone. ⚠ The rangefinder is REAL, not POSITIVE —
    the audit's AUD-47 said otherwise and was wrong; only touch and insidesite
    are positive in 3.10, 3.11 and 3.12 alike.
    """
    if cutoff <= Scalar[DTYPE](0):
        return
    var c = cutoff
    for k in range(dim):
        var v = rebind[Scalar[DTYPE]](sensordata[env, adr + k])
        if datatype == SENSDATA_REAL:
            if v > c:
                v = c
            elif v < -c:
                v = -c
        elif datatype == SENSDATA_POSITIVE:
            if v > c:
                v = c
        sensordata[env, adr + k] = v


def _eval_stage[
    target: StaticString, DTYPE: DType, D: DimsLike, BATCH: Int = 1
](
    mut d: Data[DTYPE, D, BATCH],
    mut m: Model[DTYPE, D],
    stage: Int,
    have_rne_post: Bool = True,
    ctx: Optional[DeviceContext] = None,
) raises:
    """Evaluate every SERVED sensor whose `needstage` is `stage`.

    One function rather than three near-identical ones: the stage is a filter
    over the same table, and the three public entry points below name the
    stages so a caller reads like `mj_sensorPos` and cannot pass a number.

    ⚠⚠ `have_rne_post` IS NOT THE SAME AS "THE ACCELERATION STAGE RUNS", and
    conflating them was a real bug. MuJoCo classes TOUCH as an acceleration
    sensor, so the first version of this gated the whole stage on `RNE_POST`
    and hopper — which declares two touch sensors and runs `RNE_POST=False` —
    stopped stepping. Touch reads CONTACTS and the live `site_xpos`; it never
    touches `cacc` or `cfrc_int`. Only the accelerometer and the force/torque
    pair need the post-constraint RNE, and only those are skipped when it has
    not run.
    """
    var nsensor = m.dims.get_nsensor()
    if nsensor == 0:
        return

    var nbody = m.dims.get_nbody()
    var dm = d.dims
    var mdm = m.dims

    # The GPU bind needs STATIC layouts; the CPU one needs runtime ones,
    # because a dynamic dimension provider leaves every `D.N*` poisoned.
    # Both spellings describe the same buffers — see `fields.Model`.
    comptime G_SENS = RowMajor[_max_one[D.NSENSOR](), MODEL_SENSOR_SIZE]
    comptime G_SD = RowMajor[BATCH, _max_one[D.NSENSORDATA]()]
    comptime G_B3 = RowMajor[BATCH, D.NBODY * 3]
    comptime G_B4 = RowMajor[BATCH, D.NBODY * 4]
    comptime G_B6 = RowMajor[BATCH, D.NBODY * 6]
    comptime G_S3 = RowMajor[BATCH, _max_one[D.NSITE * 3]()]
    comptime G_NQ = RowMajor[BATCH, D.NQ]
    comptime G_NV = RowMajor[BATCH, D.NV]
    comptime G_TEN = RowMajor[BATCH, _max_one[D.NTENDON]()]
    comptime G_ACT = RowMajor[BATCH, _max_one[D.NACT]()]
    comptime G_CON = RowMajor[BATCH, D.MAX_CONTACTS * CONTACT_SIZE]
    comptime G_META = RowMajor[BATCH, METADATA_SIZE]
    comptime G_BODY = RowMajor[D.NBODY, MODEL_BODY_SIZE]
    comptime G_SITE = RowMajor[_max_one[D.NSITE](), MODEL_SITE_SIZE]
    comptime G_GEOM = RowMajor[_max_one[D.NGEOM](), MODEL_GEOM_SIZE]
    comptime G_JOINT = RowMajor[D.NJOINT, MODEL_JOINT_SIZE]
    comptime G_MESH_META = RowMajor[MAX_GPU_MESHES * MODEL_MESH_META_SIZE]
    comptime G_TRI = RowMajor[_max_one[D.NMESH_TRI * MESH_ARENA_FLOATS_PER_TRI]()]
    comptime G_HF_META = RowMajor[MAX_GPU_HFIELDS * MODEL_HFIELD_META_SIZE]
    comptime G_HF = RowMajor[BATCH * _max_one[D.NHFIELD_DATA]()]

    # ── the batched views, bound ONCE for the whole pass ──────────────────
    #
    # ⚠⚠ EVERY KERNEL BELOW IS THE `_gpu` TWIN, EVEN ON THE CPU, AND THAT IS
    # WHAT MAKES `BATCH > 1` WORK AT ALL (AUD-53). The `List` forms beside
    # them index from 0 and cannot see an env; their `_gpu` twins take
    # `(tensor..., env)` and were written for the device. Calling those with
    # `lt_dyn["cpu", ...]` views is the idiom every `dynamics/` dispatcher
    # already uses, and it means the single-env and batched legs run ONE
    # implementation rather than two. The `List` forms stay for the env config
    # hooks that predate the framework and call them directly.
    comptime if target == "cpu":
        var rl_B3 = rl2(BATCH, dm.get_nbody() * 3)
        var rl_B4 = rl2(BATCH, dm.get_nbody() * 4)
        var rl_B6 = rl2(BATCH, dm.get_nbody() * 6)
        var rl_S3 = rl2(BATCH, _atleast1(dm.get_nsite() * 3))
        var rl_NQ = rl2(BATCH, dm.get_nq())
        var rl_NV = rl2(BATCH, dm.get_nv())
        var rl_SD = rl2(BATCH, _atleast1(mdm.get_nsensordata()))
        var rl_TEN = rl2(BATCH, _atleast1(mdm.get_ntendon()))
        var rl_ACT = rl2(BATCH, _atleast1(mdm.get_nact()))
        var rl_CON = rl2(BATCH, dm.get_max_contacts() * CONTACT_SIZE)
        var rl_META = rl2(BATCH, METADATA_SIZE)
        var rl_BODY = rl2(mdm.get_nbody(), MODEL_BODY_SIZE)
        var rl_SITE = rl2(_atleast1(mdm.get_nsite()), MODEL_SITE_SIZE)
        var rl_GEOM = rl2(_atleast1(mdm.get_ngeom()), MODEL_GEOM_SIZE)
        var rl_JOINT = rl2(mdm.get_njoint(), MODEL_JOINT_SIZE)

        var xpos_v = d.xpos.tt_dyn["cpu", DYN2](rl_B3)
        var xquat_v = d.xquat.tt_dyn["cpu", DYN2](rl_B4)
        var xipos_v = d.xipos.tt_dyn["cpu", DYN2](rl_B3)
        var xvel_v = d.xvel.tt_dyn["cpu", DYN2](rl_B3)
        var xangvel_v = d.xangvel.tt_dyn["cpu", DYN2](rl_B3)
        var site_xpos_v = d.site_xpos.tt_dyn["cpu", DYN2](rl_S3)
        var site_xpos_acc_v = d.site_xpos_acc.tt_dyn["cpu", DYN2](rl_S3)
        var xquat_acc_v = d.xquat_acc.tt_dyn["cpu", DYN2](rl_B4)
        var cvel_v = d.cvel.tt_dyn["cpu", DYN2](rl_B6)
        var cacc_v = d.cacc.tt_dyn["cpu", DYN2](rl_B6)
        var cfrc_int_v = d.cfrc_int.tt_dyn["cpu", DYN2](rl_B6)
        var stcom_v = d.subtree_com.tt_dyn["cpu", DYN2](rl_B3)
        var qpos_v = d.qpos.tt_dyn["cpu", DYN2](rl_NQ)
        var qvel_v = d.qvel.tt_dyn["cpu", DYN2](rl_NV)
        var qfrc_v = d.qfrc.tt_dyn["cpu", DYN2](rl_NV)
        var tenlen_v = d.ten_length.tt_dyn["cpu", DYN2](rl_TEN)
        var actlen_v = d.actuator_length.tt_dyn["cpu", DYN2](rl_ACT)
        var sdata_v = d.sensordata.tt_dyn["cpu", DYN2](rl_SD)
        var con_v = d.contacts.tt_dyn["cpu", DYN2](rl_CON)
        var dmeta_v = d.meta.tt_dyn["cpu", DYN2](rl_META)
        var bodies_v = m.bodies.tt_dyn["cpu", DYN2](rl_BODY)
        var sites_v = m.sites.tt_dyn["cpu", DYN2](rl_SITE)
        var geoms_v = m.geoms.tt_dyn["cpu", DYN2](rl_GEOM)
        var joints_v = m.joints.tt_dyn["cpu", DYN2](rl_JOINT)
        var sensors_v = m.sensors.tt_dyn["cpu", DYN2](
            rl2(_atleast1(nsensor), MODEL_SENSOR_SIZE)
        )
        # The rangefinder's four extra tables. ⚠ BOUND EVEN WHEN NO MODEL HAS ONE:
        # every one is a fixed-capacity arena (`MAX_GPU_MESHES`, `MAX_GPU_HFIELDS`)
        # or floors to a single element, so there is nothing to guard and a guard
        # would be a second predicate that can disagree with the sensor table.
        var hfn = _atleast1(dm.get_nhfield_data())
        var mesh_meta_v = m.mesh_meta.tt_dyn["cpu", DYN1](
            rl1(MAX_GPU_MESHES * MODEL_MESH_META_SIZE)
        )
        var mesh_tris_v = m.mesh_tris.tt_dyn["cpu", DYN1](
            rl1(_atleast1(mdm.get_nmesh_tri() * MESH_ARENA_FLOATS_PER_TRI))
        )
        var hf_meta_v = m.hfield_meta.tt_dyn["cpu", DYN1](
            rl1(MAX_GPU_HFIELDS * MODEL_HFIELD_META_SIZE)
        )
        var hf_v = d.hfield_data.tt_dyn["cpu", DYN1](rl1(BATCH * hfn))

        for env in range(BATCH):
            _eval_sensor_env[DTYPE, D](
                sensors_v, sdata_v, xpos_v, xquat_v, xipos_v, xvel_v, xangvel_v,
                site_xpos_v, site_xpos_acc_v, xquat_acc_v,
                cvel_v, cacc_v, cfrc_int_v, stcom_v,
                qpos_v, qvel_v, qfrc_v, tenlen_v, actlen_v,
                con_v, dmeta_v, bodies_v, sites_v, geoms_v, joints_v,
                dm, nsensor, env, stage, have_rne_post,
            )
            _eval_rangefinder_env[DTYPE, D](
                sensors_v, sdata_v, sites_v, site_xpos_v, xpos_v, xquat_v,
                geoms_v, bodies_v,
                mesh_meta_v, mesh_tris_v, hf_meta_v, hf_v,
                mdm.get_ngeom(), dm.get_nhfield_data(), nsensor, env, stage,
            )
    else:
        var c = ctx.value()
        comptime BLOCKS = (BATCH + SENSOR_TPB - 1) // SENSOR_TPB
        c.enqueue_function[
            _sensor_stage_kernel[
                DTYPE, D.NQ, D.NV, D.NBODY, D.NJOINT, D.NSITE, D.NGEOM,
                D.MAX_CONTACTS, D.NSENSOR, D.NSENSORDATA, D.NTENDON, D.NACT,
                D.NMESH_TRI, D.NHFIELD_DATA, BATCH,
            ]
        ](
            m.sensors.tt["gpu", _max_one[D.NSENSOR](), MODEL_SENSOR_SIZE](),
            d.sensordata.tt["gpu", BATCH, _max_one[D.NSENSORDATA]()](),
            d.xpos.tt["gpu", BATCH, D.NBODY * 3](),
            d.xquat.tt["gpu", BATCH, D.NBODY * 4](),
            d.xipos.tt["gpu", BATCH, D.NBODY * 3](),
            d.xvel.tt["gpu", BATCH, D.NBODY * 3](),
            d.xangvel.tt["gpu", BATCH, D.NBODY * 3](),
            d.site_xpos.tt["gpu", BATCH, _max_one[D.NSITE * 3]()](),
            d.site_xpos_acc.tt["gpu", BATCH, _max_one[D.NSITE * 3]()](),
            d.xquat_acc.tt["gpu", BATCH, D.NBODY * 4](),
            d.cvel.tt["gpu", BATCH, D.NBODY * 6](),
            d.cacc.tt["gpu", BATCH, D.NBODY * 6](),
            d.cfrc_int.tt["gpu", BATCH, D.NBODY * 6](),
            d.subtree_com.tt["gpu", BATCH, D.NBODY * 3](),
            d.qpos.tt["gpu", BATCH, D.NQ](),
            d.qvel.tt["gpu", BATCH, D.NV](),
            d.qfrc.tt["gpu", BATCH, D.NV](),
            d.ten_length.tt["gpu", BATCH, _max_one[D.NTENDON]()](),
            d.actuator_length.tt["gpu", BATCH, _max_one[D.NACT]()](),
            d.contacts.tt["gpu", BATCH, D.MAX_CONTACTS * CONTACT_SIZE](),
            d.meta.tt["gpu", BATCH, METADATA_SIZE](),
            m.bodies.tt["gpu", D.NBODY, MODEL_BODY_SIZE](),
            m.sites.tt["gpu", _max_one[D.NSITE](), MODEL_SITE_SIZE](),
            m.geoms.tt["gpu", _max_one[D.NGEOM](), MODEL_GEOM_SIZE](),
            m.joints.tt["gpu", D.NJOINT, MODEL_JOINT_SIZE](),
            Int32(stage),
            Int32(1) if have_rne_post else Int32(0),
            grid_dim=(BLOCKS,),
            block_dim=(SENSOR_TPB,),
        )
        c.enqueue_function[
            _sensor_rangefinder_kernel[
                DTYPE, D.NBODY, D.NSITE, D.NGEOM, D.NSENSOR, D.NSENSORDATA,
                D.NMESH_TRI, D.NHFIELD_DATA, BATCH,
            ]
        ](
            m.sensors.tt["gpu", _max_one[D.NSENSOR](), MODEL_SENSOR_SIZE](),
            d.sensordata.tt["gpu", BATCH, _max_one[D.NSENSORDATA]()](),
            m.sites.tt["gpu", _max_one[D.NSITE](), MODEL_SITE_SIZE](),
            d.site_xpos.tt["gpu", BATCH, _max_one[D.NSITE * 3]()](),
            d.xpos.tt["gpu", BATCH, D.NBODY * 3](),
            d.xquat.tt["gpu", BATCH, D.NBODY * 4](),
            m.geoms.tt["gpu", _max_one[D.NGEOM](), MODEL_GEOM_SIZE](),
            m.bodies.tt["gpu", D.NBODY, MODEL_BODY_SIZE](),
            m.mesh_meta.tt["gpu", MAX_GPU_MESHES * MODEL_MESH_META_SIZE](),
            m.mesh_tris.tt["gpu", _max_one[D.NMESH_TRI * MESH_ARENA_FLOATS_PER_TRI]()](),
            m.hfield_meta.tt["gpu", MAX_GPU_HFIELDS * MODEL_HFIELD_META_SIZE](),
            d.hfield_data.tt["gpu", BATCH * _max_one[D.NHFIELD_DATA]()](),
            Int32(D.NGEOM),
            Int32(D.NHFIELD_DATA),
            Int32(stage),
            grid_dim=(BLOCKS,),
            block_dim=(SENSOR_TPB,),
        )


def _eval_sensor_env[
    DTYPE: DType,
    D: DimsLike,
    L_SENS: TensorLayout,
    L_SD: TensorLayout,
    L_B3: TensorLayout,
    L_B4: TensorLayout,
    L_B6: TensorLayout,
    L_S3: TensorLayout,
    L_NQ: TensorLayout,
    L_NV: TensorLayout,
    L_TEN: TensorLayout,
    L_ACT: TensorLayout,
    L_CON: TensorLayout,
    L_META: TensorLayout,
    L_BODY: TensorLayout,
    L_SITE: TensorLayout,
    L_GEOM: TensorLayout,
    L_JOINT: TensorLayout,
](
    sensors: TileTensor[DTYPE, L_SENS, MutAnyOrigin],
    sensordata: TileTensor[DTYPE, L_SD, MutAnyOrigin],
    xpos: TileTensor[DTYPE, L_B3, MutAnyOrigin],
    xquat: TileTensor[DTYPE, L_B4, MutAnyOrigin],
    xipos: TileTensor[DTYPE, L_B3, MutAnyOrigin],
    xvel: TileTensor[DTYPE, L_B3, MutAnyOrigin],
    xangvel: TileTensor[DTYPE, L_B3, MutAnyOrigin],
    site_xpos: TileTensor[DTYPE, L_S3, MutAnyOrigin],
    site_xpos_acc: TileTensor[DTYPE, L_S3, MutAnyOrigin],
    xquat_acc: TileTensor[DTYPE, L_B4, MutAnyOrigin],
    cvel: TileTensor[DTYPE, L_B6, MutAnyOrigin],
    cacc: TileTensor[DTYPE, L_B6, MutAnyOrigin],
    cfrc_int: TileTensor[DTYPE, L_B6, MutAnyOrigin],
    subtree_com: TileTensor[DTYPE, L_B3, MutAnyOrigin],
    qpos: TileTensor[DTYPE, L_NQ, MutAnyOrigin],
    qvel: TileTensor[DTYPE, L_NV, MutAnyOrigin],
    qfrc: TileTensor[DTYPE, L_NV, MutAnyOrigin],
    ten_length: TileTensor[DTYPE, L_TEN, MutAnyOrigin],
    actuator_length: TileTensor[DTYPE, L_ACT, MutAnyOrigin],
    contacts: TileTensor[DTYPE, L_CON, MutAnyOrigin],
    dmeta: TileTensor[DTYPE, L_META, MutAnyOrigin],
    bodies: TileTensor[DTYPE, L_BODY, MutAnyOrigin],
    sites: TileTensor[DTYPE, L_SITE, MutAnyOrigin],
    geoms: TileTensor[DTYPE, L_GEOM, MutAnyOrigin],
    joints: TileTensor[DTYPE, L_JOINT, MutAnyOrigin],
    dims: D,
    nsensor: Int,
    env: Int,
    stage: Int,
    have_rne_post: Bool,
):
    """Every SERVED sensor of one env whose `needstage` is `stage`.

    ⚠ NOT `raises`, AND THAT IS A REQUIREMENT RATHER THAN A TIDY-UP. A GPU
    kernel body cannot propagate an error, so every kernel this calls had to
    be non-raising already — they are, because each is the `_gpu` twin.

    ⚠⚠ TWENTY-NINE TENSORS, AND THAT IS THE WHOLE REASON THIS FUNCTION
    EXISTS SEPARATELY FROM ITS DISPATCHER. A GPU kernel is handed
    tensors, not a `Data`/`Model` pair, and cannot manufacture one; a
    CPU dispatcher has the structs and binds the views. Splitting the
    per-env body out is what lets the two legs run the SAME code, which
    is the shape every other pass in `dynamics/` already has
    (`_rne_post_env` + `_rne_post_kernel` + `compute_rne_post`).

    ⚠ NOTHING HERE IS ON THE STACK, DELIBERATELY. The kernels it calls
    return `Array`s of 3 to 6 scalars — registers — and the mesh BVH
    traversal is stackless by construction (pre-order + escape index,
    `ray/mesh.mojo`). There is no `Scratch` anywhere under `sensors/`
    or `ray/`. That is what makes a device kernel possible at all: the
    thing that blows Metal's per-thread frame is a runtime-indexed
    local array, and this pass has none.
    """
    var nbody = dims.get_nbody()
    for i in range(nsensor):
        if Int(rebind[Scalar[DTYPE]](sensors[i, SENSOR_IDX_SERVED])) != 1:
            continue
        if Int(rebind[Scalar[DTYPE]](sensors[i, SENSOR_IDX_NEEDSTAGE])) != stage:
            continue

        var st = Int(rebind[Scalar[DTYPE]](sensors[i, SENSOR_IDX_TYPE]))
        # The three that genuinely read `cacc` / `cfrc_int`. Without the
        # post-constraint RNE they would report whatever those buffers hold,
        # so they are skipped and their slots keep the NaN `Data` filled them
        # with — loud to a reader, inert to the seven manipulation envs that
        # declare force/torque sensors and never read `sensordata`.
        if not have_rne_post and _needs_rne_post(st):
            continue
        var objid = Int(rebind[Scalar[DTYPE]](sensors[i, SENSOR_IDX_OBJID]))
        var body = Int(rebind[Scalar[DTYPE]](sensors[i, SENSOR_IDX_BODY]))
        var adr = Int(rebind[Scalar[DTYPE]](sensors[i, SENSOR_IDX_ADR]))
        var dim = Int(rebind[Scalar[DTYPE]](sensors[i, SENSOR_IDX_DIM]))

        # ⚠⚠ THE RANGEFINDER IS NOT HERE — IT HAS ITS OWN PASS. The four
        # ray tables it needs (mesh meta/tris, hfield meta/data) are four
        # buffers no other sensor reads, and Metal's ARGUMENT TABLE is the
        # binding limit on this kernel: at twenty-nine buffers the Metal
        # compiler fails with no diagnostic at all ('failed to compile
        # metallib'), and it still fails with the ray CALL removed, so the
        # count is the cause rather than the ray code. Splitting the one
        # sensor that needs them into `_eval_rangefinder_env` leaves this
        # pass at twenty-five and that one at twelve.
        if st == SENS_RANGEFINDER:
            continue

        if st == SENS_VELOCIMETER or st == SENS_GYRO:
            # ⚠ ONE KERNEL, TWO SENSORS, AND IT RETURNS LINEAR FIRST.
            # `site_frame_velocity` gives (lin, ang) — the opposite order to
            # MuJoCo's packed `res` — so the velocimeter takes [0..2] and the
            # gyro [3..5]. Getting this backwards is silent: both are 3-vectors
            # of plausible magnitude.
            var fv = site_frame_velocity_gpu[DTYPE](
                xvel, xangvel, xipos, xquat, site_xpos,
                sites, env, body, objid,
            )
            # ⚠ UNROLLED, AND NOT BY PREFERENCE: `site_frame_velocity`
            # returns a TUPLE, whose index must be a compile-time constant.
            # `fv[base + k]` with a runtime `base` does not compile.
            var vbase = 0 if st == SENS_VELOCIMETER else 3
            for k in range(3):
                sensordata[env, adr + k] = fv[vbase + k]

        elif st == SENS_JOINTPOS:
            # `mjSENS_JOINTPOS` (engine_sensor.c:644), verbatim:
            # `sensordata[0] = d->qpos[m->jnt_qposadr[objid]]`.
            #
            # ⚠ `objid` IS A JOINT INDEX HERE, NOT A SITE. The other seven
            # served types resolve to a site or a body; these two are the only
            # ones whose `objtype` is `mjOBJ_JOINT`, and `_fill_sensors` is
            # where that is decided. ⚠ The joint is guaranteed slide or hinge
            # — the loader refuses anything else, as MuJoCo's compiler does —
            # so `qpos_adr` names exactly one scalar.
            sensordata[env, adr] = rebind[Scalar[DTYPE]](qpos[
                env,
                Int(rebind[Scalar[DTYPE]](
                    joints[objid, JOINT_IDX_QPOS_ADR]))
            ])

        elif st == SENS_JOINTVEL:
            # `mjSENS_JOINTVEL` (engine_sensor.c:873):
            # `sensordata[0] = d->qvel[m->jnt_dofadr[objid]]`.
            #
            # ⚠ `dof_adr`, NOT `qpos_adr`. They coincide on a model whose
            # joints are all scalar and diverge the moment a free or ball
            # joint appears anywhere BEFORE this one — which is most models
            # that declare these sensors at all. Reading the wrong table gives
            # a plausible number from the wrong joint.
            sensordata[env, adr] = rebind[Scalar[DTYPE]](qvel[
                env,
                Int(rebind[Scalar[DTYPE]](
                    joints[objid, JOINT_IDX_DOF_ADR]))
            ])

        elif (
            st == SENS_FRAMEPOS
            or st == SENS_FRAMEQUAT
            or st == SENS_FRAMEXAXIS
            or st == SENS_FRAMEYAXIS
            or st == SENS_FRAMEZAXIS
            or st == SENS_FRAMELINVEL
            or st == SENS_FRAMEANGVEL
        ):
            # ⚠ FIVE SENSORS, ONE POSE LOOKUP, AND THE REFERENCE FRAME IS
            # PART OF IT. `objtype` selects among body / xbody / geom / site
            # (the four this loader resolves); `reftype`/`refid` are MuJoCo's
            # optional relative form and `-1` is its own "absent". A frame
            # sensor whose reference were ignored would report the GLOBAL
            # quantity — right units, right magnitude, wrong frame — so the
            # branch is here rather than in the parser.
            var otype = Int(rebind[Scalar[DTYPE]](sensors[i, SENSOR_IDX_OBJTYPE]))
            var op = frame_object_pose[DTYPE](
                xpos, xquat, xipos, site_xpos,
                bodies, geoms, sites, env, otype, objid,
            )
            var rtype = Int(rebind[Scalar[DTYPE]](sensors[i, SENSOR_IDX_REFTYPE]))
            var refid = Int(rebind[Scalar[DTYPE]](sensors[i, SENSOR_IDX_REFID]))
            var has_ref = refid >= 0 and rtype != SENSOBJ_UNKNOWN
            # ⚠ THE REFERENCE POSE IS FETCHED UNCONDITIONALLY and ignored
            # when absent. `frame_object_pose` is total (its own docstring
            # says so) and a `Tuple` cannot be declared and filled later in
            # two branches without the compiler losing track of it; the cost
            # is one pose lookup on models that declare no reference, which
            # is every model in this tree today.
            var rp = frame_object_pose[DTYPE](
                xpos, xquat, xipos, site_xpos,
                bodies, geoms, sites, env,
                rtype if has_ref else SENSOBJ_UNKNOWN,
                refid if has_ref else 0,
            )

            if st == SENS_FRAMEPOS:
                var v = frame_pos_sensor(
                    op[0], op[1], op[2], has_ref,
                    rp[0], rp[1], rp[2], rp[3], rp[4], rp[5], rp[6],
                )
                # ⚠ UNROLLED: `frame_pos_sensor` / `frame_axis_sensor`
                # return a TUPLE, whose index must be a compile-time
                # constant. The `_gpu` kernels return `Array` and can be
                # indexed by a loop variable; these cannot.
                sensordata[env, adr + 0] = Scalar[DTYPE](v[0])
                sensordata[env, adr + 1] = Scalar[DTYPE](v[1])
                sensordata[env, adr + 2] = Scalar[DTYPE](v[2])
            elif st == SENS_FRAMEQUAT:
                # ⚠ (w, x, y, z) COMES BACK, because that is what
                # `sensordata` holds. See `frame.mojo`'s module note.
                var q = frame_quat_sensor(
                    op[3], op[4], op[5], op[6], has_ref,
                    rp[3], rp[4], rp[5], rp[6],
                )
                sensordata[env, adr + 0] = Scalar[DTYPE](q[0])
                sensordata[env, adr + 1] = Scalar[DTYPE](q[1])
                sensordata[env, adr + 2] = Scalar[DTYPE](q[2])
                sensordata[env, adr + 3] = Scalar[DTYPE](q[3])
            elif st == SENS_FRAMELINVEL or st == SENS_FRAMEANGVEL:
                # ⚠ THE VELOCITY STAGE, AND `needstage` IS WHAT PUTS IT
                # THERE. These two are the only frame sensors MuJoCo
                # evaluates in `mj_sensorVel`; the stage filter at the top of
                # this loop is what keeps them from reading `xvel`/`xangvel`
                # a stage too early, when they still describe the previous
                # step.
                var fb = frame_object_body[DTYPE](
                    geoms, sites, otype, objid
                )
                var rb = frame_object_body[DTYPE](
                    geoms, sites,
                    rtype if has_ref else SENSOBJ_UNKNOWN,
                    refid if has_ref else 0,
                )
                var fv = frame_vel_sensor[DTYPE](
                    xvel, xangvel, xipos, env, fb,
                    op[0], op[1], op[2],
                    has_ref, rb, rp[0], rp[1], rp[2],
                    rp[3], rp[4], rp[5], rp[6],
                )
                # ⚠ ANGULAR FIRST, THEN LINEAR — `frame_vel_sensor` returns
                # MuJoCo's packed order, not `site_frame_velocity`'s. Both
                # halves are three plausible floats, so a swap is silent.
                if st == SENS_FRAMELINVEL:
                    sensordata[env, adr + 0] = Scalar[DTYPE](fv[3])
                    sensordata[env, adr + 1] = Scalar[DTYPE](fv[4])
                    sensordata[env, adr + 2] = Scalar[DTYPE](fv[5])
                else:
                    sensordata[env, adr + 0] = Scalar[DTYPE](fv[0])
                    sensordata[env, adr + 1] = Scalar[DTYPE](fv[1])
                    sensordata[env, adr + 2] = Scalar[DTYPE](fv[2])
            else:
                # `mjSENS_FRAMEXAXIS` is 28 and the three are consecutive, so
                # the axis index is the type offset — MuJoCo's own
                # `type - mjSENS_FRAMEXAXIS` (engine_sensor.c:694).
                var v = frame_axis_sensor(
                    op[3], op[4], op[5], op[6], st - SENS_FRAMEXAXIS,
                    has_ref, rp[3], rp[4], rp[5], rp[6],
                )
                # ⚠ UNROLLED: `frame_pos_sensor` / `frame_axis_sensor`
                # return a TUPLE, whose index must be a compile-time
                # constant. The `_gpu` kernels return `Array` and can be
                # indexed by a loop variable; these cannot.
                sensordata[env, adr + 0] = Scalar[DTYPE](v[0])
                sensordata[env, adr + 1] = Scalar[DTYPE](v[1])
                sensordata[env, adr + 2] = Scalar[DTYPE](v[2])

        elif st == SENS_ACTUATORPOS:
            # `mjSENS_ACTUATORPOS` (engine_sensor.c:652):
            # `d->actuator_length[actuator_outadr[objid]]`, `sensor_dim`
            # values. Every transmission MuJoCo currently ships has an output
            # block of ONE row (`mj_transmission`'s own comment at :1299), so
            # `outadr[i] == i` and the dim is 1.
            #
            # ⚠ FILLED BY `compute_actuator_lengths`, which is guarded on THIS
            # row existing — and which leaves the slot NaN for a transmission
            # it cannot express. Such a row is marked unserved at load, so the
            # pass above skips it and this line is never reached for one.
            sensordata[env, adr] = rebind[Scalar[DTYPE]](
                actuator_length[env, objid])

        elif st == SENS_TENDONPOS:
            # `mjSENS_TENDONPOS` (engine_sensor.c:648):
            # `sensordata[0] = d->ten_length[objid]`.
            #
            # ⚠ `d.ten_length` IS FILLED ONLY WHEN A SENSOR ASKS, by
            # `dynamics/tendon_lengths.compute_tendon_lengths`, which runs at
            # MuJoCo's `mj_tendon` point in the step. Its guard is THIS row's
            # existence, so the two cannot drift: no `<tendonpos>`, no pass,
            # and the array keeps the NaN `Data` allocated it with.
            sensordata[env, adr] = rebind[Scalar[DTYPE]](
                ten_length[env, objid])

        elif st == SENS_JOINTACTFRC:
            # `mjSENS_JOINTACTFRC` (engine_sensor.c:1309):
            # `sensordata[0] = d->qfrc_actuator[m->jnt_dofadr[objid]]`.
            #
            # ⚠⚠ `d.qfrc` IS THIS TREE'S `qfrc_actuator`, AND `d.qfrc_actuator`
            # IS NOT. The field of that name in `Data` is allocated, uploaded,
            # downloaded and never written by anything in `physics3d`. What
            # `apply_actions_fields` fills — and what
            # `_clamp_joint_actfrc` clamps against `jnt_actfrcrange`, exactly
            # as `mj_fwdActuation` clamps `qfrc_actuator`
            # (engine_forward.c:722-738) — is `d.qfrc`. Reading the
            # same-named buffer would have returned zeros forever.
            #
            # ⚠ IT IS SHORT `actuatorgravcomp` (AUD-30). MuJoCo routes a
            # gravcomp body's share into `qfrc_actuator` when the actuator
            # asks for it; this engine leaves it in the passive term. No model
            # in this tree declares both, and the audit tracks it separately.
            sensordata[env, adr] = rebind[Scalar[DTYPE]](qfrc[
                env,
                Int(rebind[Scalar[DTYPE]](
                    joints[objid, JOINT_IDX_DOF_ADR]))
            ])

        elif st == SENS_SUBTREECOM:
            # `mjSENS_SUBTREECOM` (engine_sensor.c:737):
            # `mju_copy3(sensordata, d->subtree_com + 3*objid)`.
            #
            # ⚠ THIS READ IS WHY `compute_subtree_com` MOVED AHEAD OF
            # `sensor_pos` IN THE STEP. It is `mj_comPos`, upstream of
            # `mj_sensorPos` in the reference; evaluated where it used to sit
            # this would have reported LAST step's centre of mass.
            for k in range(3):
                sensordata[env, adr + k] = rebind[Scalar[DTYPE]](
                    subtree_com[env, objid * 3 + k])

        elif st == SENS_SUBTREELINVEL:
            var svx = Scalar[DTYPE](0)
            var svy = Scalar[DTYPE](0)
            var svz = Scalar[DTYPE](0)
            subtree_linvel_gpu[DTYPE](
                dims, xvel, bodies, env, objid, svx, svy, svz
            )
            sensordata[env, adr + 0] = svx
            sensordata[env, adr + 1] = svy
            sensordata[env, adr + 2] = svz

        elif st == SENS_MAGNETOMETER:
            # `mjSENS_MAGNETOMETER` (engine_sensor.c:538):
            # `mju_mulMatTVec(sensordata, d->site_xmat+9*objid,
            #                 m->opt.magnetic, 3, 3)`
            # — the world-frame field rotated INTO the site frame.
            #
            # ⚠ THE FIELD IS ON THE SENSOR ROW, NOT IN `Model.meta`. It is a
            # document global with exactly one consumer in the reference, and
            # `mmeta` is not among this kernel's 25 buffers; binding it for
            # three floats would spend a 26th against an argument table that
            # fails with NO DIAGNOSTIC at 29. See `SENSOR_IDX_MAG_X`.
            #
            # ⚠ `R_site` IS COMPOSED, NOT READ, and the transpose is taken by
            # rotating with the CONJUGATES in the reverse order:
            # `R_site^T v = R_local^T (R_body^T v)`. `Data` has no
            # `site_xmat`, and `site_xpos` is empty on a site-less model —
            # the operand that crashed three solver tests when `tendon.mojo`
            # bound it (see `_site_world`).
            var mgx = rebind[Scalar[DTYPE]](sensors[i, SENSOR_IDX_MAG_X])
            var mgy = rebind[Scalar[DTYPE]](sensors[i, SENSOR_IDX_MAG_Y])
            var mgz = rebind[Scalar[DTYPE]](sensors[i, SENSOR_IDX_MAG_Z])
            var mbqx = rebind[Scalar[DTYPE]](xquat[env, body * 4 + 0])
            var mbqy = rebind[Scalar[DTYPE]](xquat[env, body * 4 + 1])
            var mbqz = rebind[Scalar[DTYPE]](xquat[env, body * 4 + 2])
            var mbqw = rebind[Scalar[DTYPE]](xquat[env, body * 4 + 3])
            var msqx = rebind[Scalar[DTYPE]](sites[objid, SITE_IDX_QUAT_X])
            var msqy = rebind[Scalar[DTYPE]](sites[objid, SITE_IDX_QUAT_Y])
            var msqz = rebind[Scalar[DTYPE]](sites[objid, SITE_IDX_QUAT_Z])
            var msqw = rebind[Scalar[DTYPE]](sites[objid, SITE_IDX_QUAT_W])
            var mb = gpu_quat_rotate(
                -mbqx, -mbqy, -mbqz, mbqw, mgx, mgy, mgz
            )
            var ms = gpu_quat_rotate(
                -msqx, -msqy, -msqz, msqw, mb[0], mb[1], mb[2]
            )
            for k in range(3):
                sensordata[env, adr + k] = ms[k]

        elif st == SENS_SUBTREEANGMOM:
            # `mjSENS_SUBTREEANGMOM` (engine_sensor.c). MuJoCo reads
            # `d->subtree_angmom + 3*objid`, which `mj_subtreeVel` filled
            # with two reverse passes. We evaluate the same identity
            # directly over the subtree — see `subtree_angmom_gpu` for the
            # measurement that licensed the substitution, and for why it is
            # NOT bit-identical to the reference.
            #
            # ⚠ VEL STAGE, and every operand it reads is already bound here:
            # `xipos`, `xvel`, `xangvel`, `xquat`, `subtree_com`, `bodies`.
            # That is the whole reason this is a sensor kernel rather than a
            # pass in the step — the stage kernel is at 25 buffers and fails
            # with NO DIAGNOSTIC at 29.
            var alx = Scalar[DTYPE](0)
            var aly = Scalar[DTYPE](0)
            var alz = Scalar[DTYPE](0)
            subtree_angmom_gpu[DTYPE](
                dims, xipos, xvel, xangvel, xquat, subtree_com, bodies,
                env, objid, alx, aly, alz,
            )
            sensordata[env, adr + 0] = alx
            sensordata[env, adr + 1] = aly
            sensordata[env, adr + 2] = alz

        elif st == SENS_ACCELEROMETER:
            # `*_acc`, not the live FK products — see the module note.
            var a = site_accelerometer_gpu[DTYPE](
                cvel, cacc, subtree_com, site_xpos_acc, xquat_acc,
                bodies, sites, env, body, objid,
            )
            for k in range(3):
                sensordata[env, adr + k] = a[k]

        elif st == SENS_FORCE or st == SENS_TORQUE:
            # ⚠ ONE KERNEL, TWO SENSORS AGAIN: force first, then torque.
            var ft = site_force_torque_gpu[DTYPE](
                cfrc_int, subtree_com, site_xpos_acc, xquat_acc,
                bodies, sites, env, body, objid,
            )
            # Tuple again — see the velocimeter note above.
            var fbase = 0 if st == SENS_FORCE else 3
            for k in range(3):
                sensordata[env, adr + k] = ft[fbase + k]

        elif st == SENS_TOUCH:
            # `scale` 1.0 is the MuJoCo-matching value; the parameter exists
            # for callers wanting impulses, and a sensor is not one of them.
            sensordata[env, adr] = touch_sphere_site_gpu[DTYPE](
                dims, contacts, site_xpos, sites, dmeta, xquat,
                env, objid, Scalar[DTYPE](1.0),
            )

        _apply_cutoff[DTYPE](
            sensordata,
            env,
            adr,
            dim,
            Int(rebind[Scalar[DTYPE]](sensors[i, SENSOR_IDX_DATATYPE])),
            rebind[Scalar[DTYPE]](sensors[i, SENSOR_IDX_CUTOFF]),
        )


def _eval_rangefinder_env[
    DTYPE: DType,
    D: DimsLike,
    L_SENS: TensorLayout,
    L_SD: TensorLayout,
    L_SITE: TensorLayout,
    L_S3: TensorLayout,
    L_B3: TensorLayout,
    L_B4: TensorLayout,
    L_GEOM: TensorLayout,
    L_BODY: TensorLayout,
    L_MESH_META: TensorLayout,
    L_TRI: TensorLayout,
    L_HF_META: TensorLayout,
    L_HF: TensorLayout,
](
    sensors: TileTensor[DTYPE, L_SENS, MutAnyOrigin],
    sensordata: TileTensor[DTYPE, L_SD, MutAnyOrigin],
    sites: TileTensor[DTYPE, L_SITE, MutAnyOrigin],
    site_xpos: TileTensor[DTYPE, L_S3, MutAnyOrigin],
    xpos: TileTensor[DTYPE, L_B3, MutAnyOrigin],
    xquat: TileTensor[DTYPE, L_B4, MutAnyOrigin],
    geoms: TileTensor[DTYPE, L_GEOM, MutAnyOrigin],
    bodies: TileTensor[DTYPE, L_BODY, MutAnyOrigin],
    mesh_meta: TileTensor[DTYPE, L_MESH_META, MutAnyOrigin],
    mesh_tris: TileTensor[DTYPE, L_TRI, MutAnyOrigin],
    hfield_meta: TileTensor[DTYPE, L_HF_META, MutAnyOrigin],
    hfield_data: TileTensor[DTYPE, L_HF, MutAnyOrigin],
    ngeom: Int,
    hf_stride: Int,
    nsensor: Int,
    env: Int,
    stage: Int,
):
    """The `<rangefinder>` rows of one env — a pass of its own, TWELVE buffers.

    ⚠⚠ IT IS SPLIT OFF FOR THE ARGUMENT TABLE, NOT FOR TIDINESS. The ray
    tables (mesh meta/tris, hfield meta/data) are four buffers no other sensor
    touches. Folded into `_eval_sensor_env` they take that kernel to
    twenty-nine, and the Metal compiler then fails with NO diagnostic —
    "Metal Compiler failed to compile metallib. Please submit a bug report."
    It fails identically with the ray CALL removed and the four buffers left
    bound, which is the measurement that says the COUNT is the cause. Metal's
    documented ceiling is 31 buffers; twenty-nine plus whatever the runtime
    reserves is evidently over it.

    ⚠ THE ALTERNATIVE WAS PACKING, AND IT WOULD HAVE BEEN WORSE HERE. The
    tree's other answer to an operand crunch is one workspace tensor with
    comptime offsets (`constraints/solver_ws.mojo`). It suits a solver, whose
    regions are all the same scalar type and all written by one function; it
    does not suit four independent model arenas that other kernels already
    bind separately and by name.

    Position stage only — a rangefinder is `mjSTAGE_POS`.
    """
    if stage != SENSSTAGE_POS:
        return
    for i in range(nsensor):
        if Int(rebind[Scalar[DTYPE]](sensors[i, SENSOR_IDX_SERVED])) != 1:
            continue
        if Int(rebind[Scalar[DTYPE]](sensors[i, SENSOR_IDX_TYPE])) != SENS_RANGEFINDER:
            continue
        var objid = Int(rebind[Scalar[DTYPE]](sensors[i, SENSOR_IDX_OBJID]))
        var adr = Int(rebind[Scalar[DTYPE]](sensors[i, SENSOR_IDX_ADR]))
        # ⚠⚠ A COMPTIME DType SPLIT, AND NOT BY PREFERENCE. Only
        # `rangefinder_ray` among the kernels carries
        # `where DTYPE.is_floating_point()` (it reaches `ray_model`), and
        # this pass cannot: it is called from `EulerIntegrator.step`,
        # whose `DTYPE` is unconstrained because the env-config trait that
        # reaches it is. Adding the constraint upward makes the compiler
        # reject the whole conformance — the same wall `touch.mojo`'s GPU
        # twin hit, and the same fix: name the two concrete float types
        # here rather than widen a trait every environment implements.
        #
        # ⚠ `rangefinder_ray`, NOT `rangefinder_site`. The latter binds
        # its ten views from a `Data`/`Model` pair, which a device
        # kernel does not have. Same arithmetic; the wrapper is what
        # the CPU config hooks still call.
        comptime if DTYPE == DType.float32:
            sensordata[env, adr] = rebind[Scalar[DTYPE]](
                rangefinder_ray[DType.float32](
                    rebind[TileTensor[DType.float32, L_SITE, MutAnyOrigin]](sites),
                    rebind[TileTensor[DType.float32, L_S3, MutAnyOrigin]](site_xpos),
                    rebind[TileTensor[DType.float32, L_B3, MutAnyOrigin]](xpos),
                    rebind[TileTensor[DType.float32, L_B4, MutAnyOrigin]](xquat),
                    rebind[TileTensor[DType.float32, L_GEOM, MutAnyOrigin]](geoms),
                    rebind[TileTensor[DType.float32, L_BODY, MutAnyOrigin]](bodies),
                    rebind[TileTensor[DType.float32, L_MESH_META, MutAnyOrigin]](mesh_meta),
                    rebind[TileTensor[DType.float32, L_TRI, MutAnyOrigin]](mesh_tris),
                    rebind[TileTensor[DType.float32, L_HF_META, MutAnyOrigin]](hfield_meta),
                    rebind[TileTensor[DType.float32, L_HF, MutAnyOrigin]](hfield_data),
                    ngeom, hf_stride, env, objid,
                )
            )
        else:
            comptime if DTYPE == DType.float64:
                sensordata[env, adr] = rebind[Scalar[DTYPE]](
                    rangefinder_ray[DType.float64](
                        rebind[TileTensor[DType.float64, L_SITE, MutAnyOrigin]](sites),
                        rebind[TileTensor[DType.float64, L_S3, MutAnyOrigin]](site_xpos),
                        rebind[TileTensor[DType.float64, L_B3, MutAnyOrigin]](xpos),
                        rebind[TileTensor[DType.float64, L_B4, MutAnyOrigin]](xquat),
                        rebind[TileTensor[DType.float64, L_GEOM, MutAnyOrigin]](geoms),
                        rebind[TileTensor[DType.float64, L_BODY, MutAnyOrigin]](bodies),
                        rebind[TileTensor[DType.float64, L_MESH_META, MutAnyOrigin]](mesh_meta),
                        rebind[TileTensor[DType.float64, L_TRI, MutAnyOrigin]](mesh_tris),
                        rebind[TileTensor[DType.float64, L_HF_META, MutAnyOrigin]](hfield_meta),
                        rebind[TileTensor[DType.float64, L_HF, MutAnyOrigin]](hfield_data),
                        ngeom, hf_stride, env, objid,
                    )
                )

        _apply_cutoff[DTYPE](
            sensordata,
            env,
            adr,
            1,
            Int(rebind[Scalar[DTYPE]](sensors[i, SENSOR_IDX_DATATYPE])),
            rebind[Scalar[DTYPE]](sensors[i, SENSOR_IDX_CUTOFF]),
        )



comptime SENSOR_TPB: Int = 64


def _sensor_stage_kernel[
    DTYPE: DType,
    NQ: Int,
    NV: Int,
    NBODY: Int,
    NJOINT: Int,
    NSITE: Int,
    NGEOM: Int,
    MAX_CONTACTS: Int,
    NSENSOR: Int,
    NSENSORDATA: Int,
    NTENDON: Int,
    NACT: Int,
    NMESH_TRI: Int,
    NHF: Int,
    BATCH: Int,
](
    sensors: Tile[DTYPE, _max_one[NSENSOR](), MODEL_SENSOR_SIZE],
    sensordata: Tile[DTYPE, BATCH, _max_one[NSENSORDATA]()],
    xpos: Tile[DTYPE, BATCH, NBODY * 3],
    xquat: Tile[DTYPE, BATCH, NBODY * 4],
    xipos: Tile[DTYPE, BATCH, NBODY * 3],
    xvel: Tile[DTYPE, BATCH, NBODY * 3],
    xangvel: Tile[DTYPE, BATCH, NBODY * 3],
    site_xpos: Tile[DTYPE, BATCH, _max_one[NSITE * 3]()],
    site_xpos_acc: Tile[DTYPE, BATCH, _max_one[NSITE * 3]()],
    xquat_acc: Tile[DTYPE, BATCH, NBODY * 4],
    cvel: Tile[DTYPE, BATCH, NBODY * 6],
    cacc: Tile[DTYPE, BATCH, NBODY * 6],
    cfrc_int: Tile[DTYPE, BATCH, NBODY * 6],
    subtree_com: Tile[DTYPE, BATCH, NBODY * 3],
    qpos: Tile[DTYPE, BATCH, NQ],
    qvel: Tile[DTYPE, BATCH, NV],
    qfrc: Tile[DTYPE, BATCH, NV],
    ten_length: Tile[DTYPE, BATCH, _max_one[NTENDON]()],
    actuator_length: Tile[DTYPE, BATCH, _max_one[NACT]()],
    contacts: Tile[DTYPE, BATCH, MAX_CONTACTS * CONTACT_SIZE],
    dmeta: Tile[DTYPE, BATCH, METADATA_SIZE],
    bodies: Tile[DTYPE, NBODY, MODEL_BODY_SIZE],
    sites: Tile[DTYPE, _max_one[NSITE](), MODEL_SITE_SIZE],
    geoms: Tile[DTYPE, _max_one[NGEOM](), MODEL_GEOM_SIZE],
    joints: Tile[DTYPE, NJOINT, MODEL_JOINT_SIZE],
    # ⚠⚠ `Int32`, NOT `Int`, AND NOT `Bool`. `Int` and `UInt` do not conform
    # to `DevicePassable` — the launch fails to compile with "use a
    # fixed-width type such as Int32 or Int64 instead" — and there is no
    # `Bool` overload of `enqueue_function` either. Every scalar a kernel
    # takes in this tree is a fixed-width integer for exactly this reason.
    stage: Int32,
    have_rne_post: Int32,
):
    """One thread per env; the body is `_eval_sensor_env` verbatim.

    ⚠⚠ TWENTY-NINE BUFFERS, AND THE LIMIT THAT MATTERS IS METAL'S
    ARGUMENT TABLE (31), not per-thread stack. This tree already ships
    a 27-buffer Metal kernel (`_newton_solve_fields_kernel`) and a
    14-buffer one that binds the whole ray/mesh/hfield set
    (`raytrace/batch.mojo`), so the count is known to be reachable —
    but there are only two slots spare. ⚠ ADDING AN OPERAND HERE IS A
    DECISION, not a detail. If one is needed, pack the cold tables the
    way `constraints/solver_ws.mojo` packs the solver workspace: one
    tensor, comptime offsets, named accessors.

    ⚠ NO SHARED MEMORY AND NO PER-THREAD SCRATCH. See
    `_eval_sensor_env` for why there is nothing to move off the stack:
    every kernel it calls returns a handful of scalars and the mesh
    BVH walk is stackless by construction.
    """
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    _eval_sensor_env[DTYPE](
        sensors, sensordata, xpos, xquat, xipos, xvel, xangvel, site_xpos, site_xpos_acc, xquat_acc, cvel, cacc, cfrc_int, subtree_com, qpos, qvel, qfrc, ten_length, actuator_length, contacts, dmeta, bodies, sites, geoms, joints,
        Dims[
            nq=NQ, nv=NV, nbody=NBODY, njoint=NJOINT, nsite=NSITE,
            ngeom=NGEOM, max_contacts=MAX_CONTACTS, nsensor=NSENSOR,
            nsensordata=NSENSORDATA, ntendon=NTENDON, nact=NACT,
            nmesh_tri=NMESH_TRI, nhfield_data=NHF,
        ](),
        NSENSOR, env, Int(stage), have_rne_post != 0,
    )

def _sensor_rangefinder_kernel[
    DTYPE: DType,
    NBODY: Int, NSITE: Int, NGEOM: Int, NSENSOR: Int, NSENSORDATA: Int,
    NMESH_TRI: Int, NHF: Int, BATCH: Int,
](
    sensors: Tile[DTYPE, _max_one[NSENSOR](), MODEL_SENSOR_SIZE],
    sensordata: Tile[DTYPE, BATCH, _max_one[NSENSORDATA]()],
    sites: Tile[DTYPE, _max_one[NSITE](), MODEL_SITE_SIZE],
    site_xpos: Tile[DTYPE, BATCH, _max_one[NSITE * 3]()],
    xpos: Tile[DTYPE, BATCH, NBODY * 3],
    xquat: Tile[DTYPE, BATCH, NBODY * 4],
    geoms: Tile[DTYPE, _max_one[NGEOM](), MODEL_GEOM_SIZE],
    bodies: Tile[DTYPE, NBODY, MODEL_BODY_SIZE],
    mesh_meta: Tile[DTYPE, MAX_GPU_MESHES * MODEL_MESH_META_SIZE],
    mesh_tris: Tile[DTYPE, _max_one[NMESH_TRI * MESH_ARENA_FLOATS_PER_TRI]()],
    hfield_meta: Tile[DTYPE, MAX_GPU_HFIELDS * MODEL_HFIELD_META_SIZE],
    hfield_data: Tile[DTYPE, BATCH * _max_one[NHF]()],
    ngeom: Int32,
    hf_stride: Int32,
    stage: Int32,
):
    """One thread per env, `<rangefinder>` rows only. TWELVE buffers.

    Its own kernel for the reason `_eval_rangefinder_env` gives: the four ray
    tables push the main pass past Metal's argument table. Two launches on the
    position stage; the velocity and acceleration stages return at the stage
    test without doing anything.
    """
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    _eval_rangefinder_env[DTYPE, Dims[nbody=NBODY, nsite=NSITE, ngeom=NGEOM, nsensor=NSENSOR, nsensordata=NSENSORDATA, nmesh_tri=NMESH_TRI, nhfield_data=NHF]](
        sensors, sensordata, sites, site_xpos, xpos, xquat, geoms, bodies,
        mesh_meta, mesh_tris, hfield_meta, hfield_data,
        Int(ngeom), Int(hf_stride), NSENSOR, env, Int(stage),
    )



@always_inline
def _needs_rne_post(sensor_type: Int) -> Bool:
    """Does this sensor read `cacc` / `cfrc_int`?

    ⚠ TOUCH IS AN ACCELERATION-STAGE SENSOR AND IS NOT ONE OF THESE. It sums
    contact normal forces over a zone — contacts and `site_xpos`, both valid
    without the post-constraint RNE. Putting it in this set is what broke
    hopper.
    """
    return (
        sensor_type == SENS_ACCELEROMETER
        or sensor_type == SENS_FORCE
        or sensor_type == SENS_TORQUE
    )


@always_inline
def sensor_pos[
    target: StaticString, DTYPE: DType, D: DimsLike, BATCH: Int = 1
](
    mut d: Data[DTYPE, D, BATCH],
    mut m: Model[DTYPE, D],
    ctx: Optional[DeviceContext] = None,
) raises:
    """`mj_sensorPos` — the position stage. Call after forward kinematics."""
    _eval_stage[target, DTYPE, D, BATCH](d, m, SENSSTAGE_POS, True, ctx)


@always_inline
def sensor_vel[
    target: StaticString, DTYPE: DType, D: DimsLike, BATCH: Int = 1
](
    mut d: Data[DTYPE, D, BATCH],
    mut m: Model[DTYPE, D],
    ctx: Optional[DeviceContext] = None,
) raises:
    """`mj_sensorVel` — the velocity stage. Call after body velocities."""
    _eval_stage[target, DTYPE, D, BATCH](d, m, SENSSTAGE_VEL, True, ctx)


@always_inline
def sensor_acc[
    target: StaticString, DTYPE: DType, D: DimsLike, BATCH: Int = 1
](
    mut d: Data[DTYPE, D, BATCH],
    mut m: Model[DTYPE, D],
    have_rne_post: Bool = True,
    ctx: Optional[DeviceContext] = None,
) raises:
    """`mj_sensorAcc` — the acceleration stage.

    Call after the constraint solve and after `rne_post` has filled `cacc` /
    `cfrc_int`, at the point `EulerIntegrator.step` already evaluates the
    hand-written acceleration-stage hooks.
    """
    _eval_stage[target, DTYPE, D, BATCH](
        d, m, SENSSTAGE_ACC, have_rne_post, ctx
    )
