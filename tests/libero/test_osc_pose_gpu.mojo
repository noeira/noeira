"""The OSC_POSE kernel over MANY LANES — L4b's gate.

    pixi run mojo run -I . tests/libero/test_osc_pose_gpu.mojo            # host legs
    pixi run -e apple mojo run -I . tests/libero/test_osc_pose_gpu.mojo   # + the device leg

`osc_pose_gpu.osc_run_gpu` is the controller; `osc_pose.OscPose` is one lane
of it, and `tools/libero/libero_demo_replay.py` gates THAT against MuJoCo to
1.5e-5 m over a recorded demo. What a demo replay cannot see is everything
that only appears with more than one lane, and that is what this file is:

1. **eight lanes, eight different arm poses and eight different actions**,
   evaluated by `osc_run_gpu` on `[BATCH, ...]` tensors, against the same
   eight states run one at a time through `OscPose`. At float64 the two are
   the SAME CODE on the same numbers, so the result must be **bit-exact** —
   any difference is a lane-indexing bug, the defect
   `test_tape_gpu_parity`'s two-lane arrangement exists to catch and the one
   a single-lane test can never show.
2. **the lanes must disagree.** Eight lanes that happen to produce the same
   torque would pass check 1 with `env` ignored entirely.
3. **the device leg**, with an accelerator: the same eight lanes at float32
   inside a real kernel, against leg 1. ⚠ THIS IS THE ONLY LEG THAT
   COMPILES FOR METAL, and compiling is half of what it proves — every
   matrix in the controller lives in a `[BATCH, OSC_WORK_WORDS]` GLOBAL
   tensor precisely because a per-thread array indexed by a runtime value
   reads back wrong there (`osc_pose_gpu`'s header). The band is float32's,
   stated and printed, not a tolerance tuned until it passed.

⚠ THE DYNAMICS INPUTS ARE COMPUTED ONCE, ON THE CPU, AND SHIPPED TO BOTH
LEGS. This gates the CONTROLLER, not the engine: `subtree_com`, `cdof`, `M`
and `bias` are the engine's, they already have their own gates (the
fifty-step board, `bcmp.py`), and recomputing them per leg would fold two
questions into one number.
"""

from std.os.path import exists
from std.sys import has_accelerator
from max.gpu import thread_idx, block_idx, block_dim
from layout import TileTensor
from noeira.core.tiles import RowMajor
from max.gpu.host import DeviceContext

from noeira.nn.core.tensor import TensorImpl
from noeira.physics3d.fields import (
    Data, Model, DynDims, DynamicsScratch, DYN1, DYN2, rl1, rl2,
)
from noeira.physics3d.parser.runtime_load import (
    parse_model_runtime, dims_from_flat, build_model_runtime,
)
from noeira.physics3d.gpu.constants import (
    MODEL_BODY_SIZE, MODEL_JOINT_SIZE, MODEL_META_SIZE, MODEL_SITE_SIZE,
)
from noeira.physics3d.dynamics.osc_pose import (
    OscPose, OscPoseConfig, ARM_DOF, OSC_ACTION_DIM,
)
from noeira.physics3d.dynamics.osc_pose_gpu import (
    OSC_REF_WORDS, OSC_STATE_WORDS, OSC_WORK_WORDS, osc_run_gpu,
)
from noeira.tasks.spec import load_family
from noeira.tasks.family import scene_path
from noeira.envs.libero.models.libero_goal_dims import LIBERO_GOAL_DIMS
from noeira.envs.libero.models.libero_goal_xml import LIBERO_GOAL_MAX_CONTACTS


comptime DT = DType.float64
comptime F32 = DType.float32
comptime FAMILY = "noeira/envs/libero/families/libero_goal.family"
comptime PACK = "noeira/envs/libero/assets"
comptime BATCH = 8

comptime NB = LIBERO_GOAL_DIMS.NBODY
comptime NS = LIBERO_GOAL_DIMS.NSITE
comptime NQ = LIBERO_GOAL_DIMS.NQ
comptime NV = LIBERO_GOAL_DIMS.NV
comptime NJ = LIBERO_GOAL_DIMS.NJOINT
comptime NACT = LIBERO_GOAL_DIMS.NACT

comptime L_STATE = RowMajor[BATCH, OSC_STATE_WORDS]
comptime L_WORK = RowMajor[BATCH, OSC_WORK_WORDS]
comptime L_REFS = RowMajor[OSC_REF_WORDS]
comptime L_CTRL = RowMajor[BATCH, NACT]
comptime L_QPOS = RowMajor[BATCH, NQ]
comptime L_NV = RowMajor[BATCH, NV]
comptime L_B4 = RowMajor[BATCH, NB * 4]
comptime L_SX = RowMajor[BATCH, NS * 3]
comptime L_B3 = RowMajor[BATCH, NB * 3]
comptime L_CDOF = RowMajor[BATCH, NV * 6]
comptime L_M = RowMajor[BATCH, NV * NV]
comptime L_JOINTS = RowMajor[NJ, MODEL_JOINT_SIZE]
comptime L_BODIES = RowMajor[NB, MODEL_BODY_SIZE]
comptime L_SITES = RowMajor[NS, MODEL_SITE_SIZE]
comptime L_MMETA = RowMajor[MODEL_META_SIZE]


struct Tally(Copyable, ImplicitlyCopyable, Movable):
    var checks: Int
    var failures: Int

    def __init__(out self):
        self.checks = 0
        self.failures = 0

    def check(mut self, ok: Bool, what: String):
        self.checks += 1
        if ok:
            print("  ok:", what)
        else:
            self.failures += 1
            print("  FAIL:", what)


def _index(names: List[String], want: String) raises -> Int:
    for i in range(len(names)):
        if String(names[i]) == want:
            return i
    raise Error("no '" + want + "' in the composed scene")


def _osc_kernel(
    state: TileTensor[F32, L_STATE, MutAnyOrigin],
    work: TileTensor[F32, L_WORK, MutAnyOrigin],
    refs: TileTensor[F32, L_REFS, MutAnyOrigin],
    ctrl: TileTensor[F32, L_CTRL, MutAnyOrigin],
    qpos: TileTensor[F32, L_QPOS, MutAnyOrigin],
    qvel: TileTensor[F32, L_NV, MutAnyOrigin],
    xquat: TileTensor[F32, L_B4, MutAnyOrigin],
    sxp: TileTensor[F32, L_SX, MutAnyOrigin],
    stcom: TileTensor[F32, L_B3, MutAnyOrigin],
    cdof: TileTensor[F32, L_CDOF, MutAnyOrigin],
    mass: TileTensor[F32, L_M, MutAnyOrigin],
    bias: TileTensor[F32, L_NV, MutAnyOrigin],
    joints: TileTensor[F32, L_JOINTS, MutAnyOrigin],
    bodies: TileTensor[F32, L_BODIES, MutAnyOrigin],
    sites: TileTensor[F32, L_SITES, MutAnyOrigin],
    mmeta: TileTensor[F32, L_MMETA, MutAnyOrigin],
):
    var env = Int(block_dim.x * block_idx.x + thread_idx.x)
    if env >= BATCH:
        return
    osc_run_gpu[F32](
        state, work, refs, ctrl, qpos, qvel, xquat, sxp, stcom, cdof,
        mass, bias, joints, bodies, sites, mmeta, env, NV,
    )


def main() raises:
    print("=== OSC_POSE over", BATCH, "lanes — L4b ===")
    if not exists(PACK):
        print("  SKIPPED: no LIBERO pack at", PACK)
        print("=== SKIPPED (no pack — this is not a pass) ===")
        return
    var ta = Tally()

    # ── the scene, on the CPU ─────────────────────────────────────────────
    var f = load_family(FAMILY)
    var fmd = parse_model_runtime(scene_path(f))
    var verts = 32768
    var dims = dims_from_flat(
        fmd, max_contacts=LIBERO_GOAL_MAX_CONTACTS, nmesh_verts=verts
    )
    var m = Model[DT, DynDims](dims)
    while True:
        try:
            build_model_runtime[DT](fmd, dims, m)
            break
        except e:
            if String(e).find("mesh vertex capacity") < 0:
                raise e
            verts *= 2
            dims = dims_from_flat(
                fmd, max_contacts=LIBERO_GOAL_MAX_CONTACTS, nmesh_verts=verts
            )
            m = Model[DT, DynDims](dims)
    var d = Data[DT, DynDims, 1](dims)
    var scratch = DynamicsScratch[DT, DynDims, 1](dims)
    var nq = dims.get_nq()
    var nv = dims.get_nv()
    var nb = dims.get_nbody()
    var ns = dims.get_nsite()
    var nact = dims.get_nact()
    if nq != NQ or nv != NV or nb != NB or ns != NS or nact != NACT:
        raise Error("osc gpu: the scene disagrees with LIBERO_GOAL_DIMS")

    var qadr_all = List[Int]()
    var dadr_all = List[Int]()
    var qa = 0
    var da = 0
    for i in range(len(fmd.joints)):
        qadr_all.append(qa)
        dadr_all.append(da)
        qa += fmd.joints[i].nq
        da += fmd.joints[i].nv
    var dof = List[Int]()
    var qadr = List[Int]()
    var jidx = List[Int]()
    var act_idx = List[Int]()
    var tmin = List[Float64]()
    var tmax = List[Float64]()
    for j in range(ARM_DOF):
        var ji = _index(fmd.joint_names, String("robot_joint") + String(j + 1))
        dof.append(dadr_all[ji])
        qadr.append(qadr_all[ji])
        jidx.append(ji)
        var ai = _index(fmd.actuator_names, String("robot_torq_j") + String(j + 1))
        act_idx.append(ai)
        tmin.append(fmd.actuators[ai].ctrl_min)
        tmax.append(fmd.actuators[ai].ctrl_max)
    var site = _index(fmd.site_names, String("robot_grip_site"))
    var site_body = fmd.sites[site].body_id
    var ga1 = _index(fmd.actuator_names, String("robot_gripper_finger_joint1"))
    var ga2 = _index(fmd.actuator_names, String("robot_gripper_finger_joint2"))

    # ── the staging tensors both legs read ────────────────────────────────
    var t_state = TensorImpl[DT].alloc(BATCH * OSC_STATE_WORDS)
    var t_work = TensorImpl[DT].alloc(BATCH * OSC_WORK_WORDS)
    var t_refs = TensorImpl[DT].alloc(OSC_REF_WORDS)
    var t_ctrl = TensorImpl[DT].alloc(BATCH * NACT)
    var t_qpos = TensorImpl[DT].alloc(BATCH * NQ)
    var t_qvel = TensorImpl[DT].alloc(BATCH * NV)
    var t_xq = TensorImpl[DT].alloc(BATCH * NB * 4)
    var t_sx = TensorImpl[DT].alloc(BATCH * NS * 3)
    var t_st = TensorImpl[DT].alloc(BATCH * NB * 3)
    var t_cdof = TensorImpl[DT].alloc(BATCH * NV * 6)
    var t_m = TensorImpl[DT].alloc(BATCH * NV * NV)
    var t_bias = TensorImpl[DT].alloc(BATCH * NV)
    var t_joints = TensorImpl[DT].alloc(NJ * MODEL_JOINT_SIZE)
    var t_bodies = TensorImpl[DT].alloc(NB * MODEL_BODY_SIZE)
    var t_sites = TensorImpl[DT].alloc(NS * MODEL_SITE_SIZE)
    var t_meta = TensorImpl[DT].alloc(MODEL_META_SIZE)
    for i in range(NJ * MODEL_JOINT_SIZE):
        t_joints.data[i] = m.joints.data[i]
    for i in range(NB * MODEL_BODY_SIZE):
        t_bodies.data[i] = m.bodies.data[i]
    for i in range(NS * MODEL_SITE_SIZE):
        t_sites.data[i] = m.sites.data[i]
    for i in range(MODEL_META_SIZE):
        t_meta.data[i] = m.meta.data[i]
    for i in range(BATCH * OSC_WORK_WORDS):
        t_work.data[i] = Scalar[DT](0)
    for i in range(BATCH * NACT):
        t_ctrl.data[i] = Scalar[DT](0)

    # ── eight lanes, eight states, eight actions, one at a time ───────────
    print("--- the reference: one lane at a time through OscPose ---")
    var ref_ctrl = List[List[Float64]]()
    for e in range(BATCH):
        # a different arm pose per lane, and the free objects parked high
        for i in range(nq):
            d.qpos.data[i] = Scalar[DT](0)
        for i in range(nv):
            d.qvel.data[i] = Scalar[DT](0)
        for i in range(len(f.base_qpos)):
            d.qpos.data[i] = Scalar[DT](f.base_qpos[i])
        for j in range(ARM_DOF):
            # ⚠ EVERY LANE A DIFFERENT POSE AND A DIFFERENT VELOCITY, or a
            # kernel that ignored `env` would agree with the reference.
            d.qpos.data[qadr[j]] = Scalar[DT](
                Float64(d.qpos.data[qadr[j]]) + 0.03 * Float64((e * 3 + j) % 5 - 2)
            )
            d.qvel.data[dof[j]] = Scalar[DT](0.02 * Float64((e * 5 + j) % 7 - 3))
        for i in range(len(fmd.joints)):
            if fmd.joints[i].nq == 7:
                d.qpos.data[qadr_all[i]] = Scalar[DT](2.0)
                d.qpos.data[qadr_all[i] + 2] = Scalar[DT](3.0)
                d.qpos.data[qadr_all[i] + 3] = Scalar[DT](1.0)

        var osc = OscPose(
            dof.copy(), qadr.copy(), jidx.copy(), tmin.copy(), tmax.copy(),
            act_idx.copy(), site, site_body, ga1, ga2,
            fmd.actuators[ga1].ctrl_min, fmd.actuators[ga1].ctrl_max,
            fmd.actuators[ga2].ctrl_min, fmd.actuators[ga2].ctrl_max,
            OscPoseConfig(), nact, nq, nv,
        )
        osc.update(d, m, scratch)
        osc.reset(d, m)
        var action = List[Float64]()
        for k in range(OSC_ACTION_DIM):
            action.append(0.25 * Float64((e * 2 + k) % 7 - 3))
        osc.set_goal(action, d, m)
        # a second update, as a substep would: the goal is held, the state read
        osc.update(d, m, scratch)
        var c = osc.run(action, d, m, scratch)
        ref_ctrl.append(c^)

        # stage this lane's inputs and its controller state
        var sw = osc.state_words()
        for k in range(OSC_STATE_WORDS):
            t_state.data[e * OSC_STATE_WORDS + k] = Scalar[DT](sw[k])
        for k in range(NQ):
            t_qpos.data[e * NQ + k] = d.qpos.data[k]
        for k in range(NV):
            t_qvel.data[e * NV + k] = d.qvel.data[k]
            t_bias.data[e * NV + k] = scratch.bias.data[k]
        for k in range(NB * 4):
            t_xq.data[e * NB * 4 + k] = d.xquat.data[k]
        for k in range(NS * 3):
            t_sx.data[e * NS * 3 + k] = d.site_xpos.data[k]
        for k in range(NB * 3):
            t_st.data[e * NB * 3 + k] = d.subtree_com.data[k]
        for k in range(NV * 6):
            t_cdof.data[e * NV * 6 + k] = scratch.cdof.data[k]
        for k in range(NV * NV):
            t_m.data[e * NV * NV + k] = scratch.M.data[k]
        if e == 0:
            for k in range(OSC_REF_WORDS):
                t_refs.data[k] = osc.refs_t.data[k]
    print("  staged", BATCH, "lanes")

    var spread = 0.0
    for e in range(1, BATCH):
        for j in range(ARM_DOF):
            var dd = ref_ctrl[e][act_idx[j]] - ref_ctrl[0][act_idx[j]]
            if dd < 0.0:
                dd = -dd
            if dd > spread:
                spread = dd
    print("  lane torques differ from lane 0 by up to", spread, "N m")
    ta.check(spread > 1.0, "the eight lanes really do carry different states")

    # ── leg 1: the kernel loop over BATCH lanes, float64 ──────────────────
    print("--- leg 1: osc_run_gpu over", BATCH, "lanes at float64 ---")
    for e in range(BATCH):
        osc_run_gpu[DT](
            t_state.tt_dyn["cpu", DYN2](rl2(BATCH, OSC_STATE_WORDS)),
            t_work.tt_dyn["cpu", DYN2](rl2(BATCH, OSC_WORK_WORDS)),
            t_refs.tt_dyn["cpu", DYN1](rl1(OSC_REF_WORDS)),
            t_ctrl.tt_dyn["cpu", DYN2](rl2(BATCH, NACT)),
            t_qpos.tt_dyn["cpu", DYN2](rl2(BATCH, NQ)),
            t_qvel.tt_dyn["cpu", DYN2](rl2(BATCH, NV)),
            t_xq.tt_dyn["cpu", DYN2](rl2(BATCH, NB * 4)),
            t_sx.tt_dyn["cpu", DYN2](rl2(BATCH, NS * 3)),
            t_st.tt_dyn["cpu", DYN2](rl2(BATCH, NB * 3)),
            t_cdof.tt_dyn["cpu", DYN2](rl2(BATCH, NV * 6)),
            t_m.tt_dyn["cpu", DYN2](rl2(BATCH, NV * NV)),
            t_bias.tt_dyn["cpu", DYN2](rl2(BATCH, NV)),
            t_joints.tt_dyn["cpu", DYN2](rl2(NJ, MODEL_JOINT_SIZE)),
            t_bodies.tt_dyn["cpu", DYN2](rl2(NB, MODEL_BODY_SIZE)),
            t_sites.tt_dyn["cpu", DYN2](rl2(NS, MODEL_SITE_SIZE)),
            t_meta.tt_dyn["cpu", DYN1](rl1(MODEL_META_SIZE)),
            e,
            NV,
        )
    var worst = 0.0
    var worst_lane = -1
    for e in range(BATCH):
        for j in range(ARM_DOF):
            var a = Float64(t_ctrl.data[e * NACT + act_idx[j]])
            var b = ref_ctrl[e][act_idx[j]]
            var dd = a - b
            if dd < 0.0:
                dd = -dd
            if dd > worst:
                worst = dd
                worst_lane = e
    print("  worst |d tau| vs the one-lane reference:", worst,
          "N m (lane", worst_lane, ")")
    ta.check(worst == 0.0, "BATCH lanes == one lane at a time, BIT-EXACT")

    # ── leg 2: the device ─────────────────────────────────────────────────
    comptime if has_accelerator():
        print("--- leg 2: the same eight lanes in a kernel, float32 ---")
        var ctx = DeviceContext()
        var g_state = TensorImpl[F32].alloc(BATCH * OSC_STATE_WORDS)
        var g_work = TensorImpl[F32].alloc(BATCH * OSC_WORK_WORDS)
        var g_refs = TensorImpl[F32].alloc(OSC_REF_WORDS)
        var g_ctrl = TensorImpl[F32].alloc(BATCH * NACT)
        var g_qpos = TensorImpl[F32].alloc(BATCH * NQ)
        var g_qvel = TensorImpl[F32].alloc(BATCH * NV)
        var g_xq = TensorImpl[F32].alloc(BATCH * NB * 4)
        var g_sx = TensorImpl[F32].alloc(BATCH * NS * 3)
        var g_st = TensorImpl[F32].alloc(BATCH * NB * 3)
        var g_cdof = TensorImpl[F32].alloc(BATCH * NV * 6)
        var g_m = TensorImpl[F32].alloc(BATCH * NV * NV)
        var g_bias = TensorImpl[F32].alloc(BATCH * NV)
        var g_joints = TensorImpl[F32].alloc(NJ * MODEL_JOINT_SIZE)
        var g_bodies = TensorImpl[F32].alloc(NB * MODEL_BODY_SIZE)
        var g_sites = TensorImpl[F32].alloc(NS * MODEL_SITE_SIZE)
        var g_meta = TensorImpl[F32].alloc(MODEL_META_SIZE)
        for i in range(BATCH * OSC_STATE_WORDS):
            g_state.data[i] = Scalar[F32](Float64(t_state.data[i]))
        for i in range(BATCH * OSC_WORK_WORDS):
            g_work.data[i] = Scalar[F32](0)
        for i in range(OSC_REF_WORDS):
            g_refs.data[i] = Scalar[F32](Float64(t_refs.data[i]))
        for i in range(BATCH * NACT):
            g_ctrl.data[i] = Scalar[F32](0)
        for i in range(BATCH * NQ):
            g_qpos.data[i] = Scalar[F32](Float64(t_qpos.data[i]))
        for i in range(BATCH * NV):
            g_qvel.data[i] = Scalar[F32](Float64(t_qvel.data[i]))
            g_bias.data[i] = Scalar[F32](Float64(t_bias.data[i]))
        for i in range(BATCH * NB * 4):
            g_xq.data[i] = Scalar[F32](Float64(t_xq.data[i]))
        for i in range(BATCH * NS * 3):
            g_sx.data[i] = Scalar[F32](Float64(t_sx.data[i]))
        for i in range(BATCH * NB * 3):
            g_st.data[i] = Scalar[F32](Float64(t_st.data[i]))
        for i in range(BATCH * NV * 6):
            g_cdof.data[i] = Scalar[F32](Float64(t_cdof.data[i]))
        for i in range(BATCH * NV * NV):
            g_m.data[i] = Scalar[F32](Float64(t_m.data[i]))
        for i in range(NJ * MODEL_JOINT_SIZE):
            g_joints.data[i] = Scalar[F32](Float64(t_joints.data[i]))
        for i in range(NB * MODEL_BODY_SIZE):
            g_bodies.data[i] = Scalar[F32](Float64(t_bodies.data[i]))
        for i in range(NS * MODEL_SITE_SIZE):
            g_sites.data[i] = Scalar[F32](Float64(t_sites.data[i]))
        for i in range(MODEL_META_SIZE):
            g_meta.data[i] = Scalar[F32](Float64(t_meta.data[i]))
        g_state.upload(ctx)
        g_work.upload(ctx)
        g_refs.upload(ctx)
        g_ctrl.upload(ctx)
        g_qpos.upload(ctx)
        g_qvel.upload(ctx)
        g_xq.upload(ctx)
        g_sx.upload(ctx)
        g_st.upload(ctx)
        g_cdof.upload(ctx)
        g_m.upload(ctx)
        g_bias.upload(ctx)
        g_joints.upload(ctx)
        g_bodies.upload(ctx)
        g_sites.upload(ctx)
        g_meta.upload(ctx)
        ctx.enqueue_function[_osc_kernel](
            g_state.tt["gpu", BATCH, OSC_STATE_WORDS](), g_work.tt["gpu", BATCH, OSC_WORK_WORDS](),
            g_refs.tt["gpu", OSC_REF_WORDS](), g_ctrl.tt["gpu", BATCH, NACT](),
            g_qpos.tt["gpu", BATCH, NQ](), g_qvel.tt["gpu", BATCH, NV](),
            g_xq.tt["gpu", BATCH, NB * 4](), g_sx.tt["gpu", BATCH, NS * 3](),
            g_st.tt["gpu", BATCH, NB * 3](), g_cdof.tt["gpu", BATCH, NV * 6](),
            g_m.tt["gpu", BATCH, NV * NV](), g_bias.tt["gpu", BATCH, NV](),
            g_joints.tt["gpu", NJ, MODEL_JOINT_SIZE](), g_bodies.tt["gpu", NB, MODEL_BODY_SIZE](),
            g_sites.tt["gpu", NS, MODEL_SITE_SIZE](), g_meta.tt["gpu", MODEL_META_SIZE](),
            grid_dim=(1,), block_dim=(BATCH,),
        )
        g_ctrl.download(ctx)
        g_state.download(ctx)
        ctx.synchronize()
        var dworst = 0.0
        var dlane = -1
        var biggest = 0.0
        for e in range(BATCH):
            for j in range(ARM_DOF):
                var a = Float64(g_ctrl.data[e * NACT + act_idx[j]])
                var b = ref_ctrl[e][act_idx[j]]
                var ab = b if b > 0.0 else -b
                if ab > biggest:
                    biggest = ab
                var dd = a - b
                if dd < 0.0:
                    dd = -dd
                if dd > dworst:
                    dworst = dd
                    dlane = e
        var sing = 0
        for e in range(BATCH):
            if g_state.data[e * OSC_STATE_WORDS + 21] != Scalar[F32](0):
                sing += 1
        print("  worst |d tau| device(f32) vs host(f64):", dworst,
              "N m (lane", dlane, "), largest torque", biggest, "N m")
        ta.check(sing == 0, "no lane hit the singular flag")
        # ⚠ THE BAND IS float32's ON THIS ARITHMETIC, not a tuned number: a
        # 7x7 inverse and four matrix products at ~1e-7 relative, against
        # torques of tens of N m.
        ta.check(
            dworst < 2e-2,
            "the device agrees with the host to float32 (" + String(dworst)
            + " N m on " + String(biggest) + ")",
        )
        var dspread = 0.0
        for e in range(1, BATCH):
            for j in range(ARM_DOF):
                var dd = (
                    Float64(g_ctrl.data[e * NACT + act_idx[j]])
                    - Float64(g_ctrl.data[0 * NACT + act_idx[j]])
                )
                if dd < 0.0:
                    dd = -dd
                if dd > dspread:
                    dspread = dd
        ta.check(dspread > 1.0, "and the device's lanes differ from each other")
    else:
        print("--- leg 2: SKIPPED (no accelerator; run with -e apple / -e nvidia) ---")

    print()
    print("--- ran", ta.checks, "checks,", ta.failures, "failed ---")
    if ta.failures != 0:
        raise Error(
            "osc gpu: " + String(ta.failures) + " of " + String(ta.checks)
            + " failed"
        )
    print("=== PASS ===")
