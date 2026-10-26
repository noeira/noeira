"""OSC_POSE on the host — ONE LANE OF THE DEVICE CONTROLLER, nothing more.

    var osc = OscPose(dof, qadr, joint, tmin, tmax, act, site, site_body,
                      ga0, ga1, g0min, g0max, g1min, g1max, cfg, nact, nq, nv)
    osc.update(d, m, scratch)          # every substep: sim.forward()'s half
    osc.reset(d, m)                    # once, after the first update
    osc.set_goal(action, d, m)         # on a POLICY step (25 substeps apart)
    var ctrl = osc.run(action, d, m, scratch)   # nact ctrl values

## ⚠⚠ THE ARITHMETIC IS IN `osc_pose_gpu.mojo` AND IS NOT REPEATED HERE

This struct owns four one-lane `TensorImpl`s and calls `osc_reset_gpu`,
`osc_set_goal_gpu`, `osc_run_gpu` and `osc_gripper_gpu` on views of them.
Every number the controller computes is computed by those functions, at
`DType.float64`, from this file exactly as from a kernel — the pattern
`tests/tasks/test_tape_gpu_parity.mojo` uses for the goal tape, and the
reason the demo replay gate (`tools/libero/libero_demo_replay.py`, 1.5e-5 m
against MuJoCo over 80 policy steps) is a gate on the DEVICE code too.

An earlier version of this file spelled the whole law a second time over
`List[Float64]`. That is `_a_rule_written_inline_twice_drifts`, and for a
controller it would have drifted silently — both legs still hold the arm up.

## WHAT IS QUOTED, AND FROM WHERE

`robosuite-1.4.0/robosuite/controllers/osc.py` + `base_controller.py` +
`utils/control_utils.py` — the version LIBERO pins (`requirements.txt`).
The 1.5 tree beside LIBERO is a different controller (base-frame goals,
`_goal_update_mode`, a `_center` site) and did not record the demos.

    F  = kp (goal_pos - ee_pos) - kd ee_vel_pos
    T  = kp ori_error(goal_R, ee_R) - kd ee_vel_ori
    Lp = pinv(Jp M^-1 Jp^T)   Lo = pinv(Jr M^-1 Jr^T)   (uncouple_pos_ori)
    L  = pinv(J M^-1 J^T)     Jbar = M^-1 J^T L    N = I - Jbar J
    tau = J^T [Lp F; Lo T] + qfrc_bias + N^T M (10 (q0 - q) - 2sqrt10 qd)
    tau = clip(tau, ctrlrange)

`kp = 150`, `kd = 2 sqrt(kp)`, M and J the 7-dof arm BLOCKS, `ee_vel =
J @ qvel` over the full dof vector (`get_site_xvelp/xvelr` are exactly
that, not `mj_objectVelocity`), `q0` the configuration at controller
construction — which `Robot.reset` does after writing `init_qpos`.

⚠ `pinv` IS `inv` HERE. `np.linalg.pinv` (SVD, rcond 1e-15) and an inverse
agree wherever the operational-space inertia is full rank, which is every
configuration a manipulation demo visits; `osc_pose_gpu` raises a flag
instead of returning a pseudo-inverse and `run()` turns that into an error
rather than a plausible torque.

⚠ `T.quat2mat` CASTS TO float32 before forming the goal rotation, so
robosuite's goal carries ~1e-7 of relative error that this does not. That
difference IS most of the 1.5e-5 m the replay measures.

## THE GRIPPER

`PandaGripper.format_action` + `Manipulator.grip_action`: `current +=
[-1, +1] * 0.01 * sign(a)` clipped to +-1, EVERY SIM STEP (`Robot.control`
runs per substep; only `set_goal` is gated on `policy_step`), then
`ctrl = bias + weight * current`. 0.25 per policy step — four policy steps
from neutral to an end, eight end to end. `PandaGripperRamp` is the same
`osc_gripper_gpu` on a one-lane tensor.
"""

from std.math import sqrt

from noeira.nn.core.tensor import TensorImpl

from ..fields import (
    Data, Model, DynamicsScratch, DynDims, DYN1, DYN2, rl1, rl2,
)
from ..gpu.constants import (
    MODEL_BODY_SIZE, MODEL_JOINT_SIZE, MODEL_META_SIZE, MODEL_SITE_SIZE,
)
from .osc_pose_gpu import (
    OSC_ACTION_DIM,
    OSC_ARM, OSC_WRENCH, OSC_REF_WORDS, OSC_STATE_WORDS, OSC_WORK_WORDS,
    OSC_IDX_GOAL_POS, OSC_IDX_GOAL_MAT, OSC_IDX_Q0, OSC_IDX_SINGULAR,
    OSC_IDX_READY, OSC_IDX_GRIP0, OSC_IDX_GRIP1, OSC_REF_SITE,
    OSC_W_JAC, OSC_W_MBLK, OSC_W_VEC,
    build_osc_refs, osc_refresh_dynamics,
    osc_reset_gpu, osc_set_goal_gpu, osc_run_gpu, osc_gripper_gpu,
)


comptime DT = DType.float64
comptime ARM_DOF: Int = OSC_ARM
# `OSC_ACTION_DIM` is re-exported from `osc_pose_gpu`, where it sits beside the
# kernel that indexes by it. Importers of this module keep working.


struct OscPoseConfig(Copyable, ImplicitlyCopyable, Movable):
    """`controllers/config/osc_pose.json`, the file LIBERO loads."""

    var kp: Float64
    var damping_ratio: Float64
    var output_max_pos: Float64
    var output_max_ori: Float64
    var nullspace_kp: Float64
    var gripper_speed: Float64

    def __init__(out self):
        self.kp = 150.0
        self.damping_ratio = 1.0
        self.output_max_pos = 0.05
        self.output_max_ori = 0.5
        self.nullspace_kp = 10.0
        self.gripper_speed = 0.01


struct OscPose(Movable & Deinitable):
    var refs_t: TensorImpl[DT]
    var state_t: TensorImpl[DT]
    var work_t: TensorImpl[DT]
    var ctrl_t: TensorImpl[DT]
    var act_t: TensorImpl[DT]
    var nact: Int
    var nq: Int
    var nv: Int
    var dof: List[Int]
    var qadr: List[Int]

    def __init__(
        out self,
        dof: List[Int], qadr: List[Int], joint: List[Int],
        tmin: List[Float64], tmax: List[Float64], act: List[Int],
        site: Int, site_body: Int,
        grip_act0: Int, grip_act1: Int,
        g0_min: Float64, g0_max: Float64, g1_min: Float64, g1_max: Float64,
        cfg: OscPoseConfig,
        nact: Int, nq: Int, nv: Int,
    ) raises:
        var refs = build_osc_refs(
            dof, qadr, joint, tmin, tmax, act, site, site_body,
            grip_act0, grip_act1, g0_min, g0_max, g1_min, g1_max,
            cfg.kp, cfg.damping_ratio, cfg.output_max_pos, cfg.output_max_ori,
            cfg.nullspace_kp, cfg.gripper_speed,
        )
        self.refs_t = TensorImpl[DT].alloc(OSC_REF_WORDS)
        for i in range(OSC_REF_WORDS):
            self.refs_t.data[i] = Scalar[DT](refs[i])
        self.state_t = TensorImpl[DT].alloc(OSC_STATE_WORDS)
        for i in range(OSC_STATE_WORDS):
            self.state_t.data[i] = Scalar[DT](0)
        self.work_t = TensorImpl[DT].alloc(OSC_WORK_WORDS)
        for i in range(OSC_WORK_WORDS):
            self.work_t.data[i] = Scalar[DT](0)
        var na = nact if nact > 0 else 1
        self.ctrl_t = TensorImpl[DT].alloc(na)
        for i in range(na):
            self.ctrl_t.data[i] = Scalar[DT](0)
        self.act_t = TensorImpl[DT].alloc(OSC_ACTION_DIM)
        for i in range(OSC_ACTION_DIM):
            self.act_t.data[i] = Scalar[DT](0)
        self.nact = nact
        self.nq = nq
        self.nv = nv
        self.dof = dof.copy()
        self.qadr = qadr.copy()

    def __init__(out self, *, deinit move: Self):
        self.refs_t = move.refs_t^
        self.state_t = move.state_t^
        self.work_t = move.work_t^
        self.ctrl_t = move.ctrl_t^
        self.act_t = move.act_t^
        self.nact = move.nact
        self.nq = move.nq
        self.nv = move.nv
        self.dof = move.dof^
        self.qadr = move.qadr^

    def _write_action(mut self, action: List[Float64]):
        for i in range(OSC_ACTION_DIM):
            self.act_t.data[i] = Scalar[DT](
                action[i] if i < len(action) else 0.0
            )

    def update(
        mut self,
        mut d: Data[DT, DynDims, 1],
        mut m: Model[DT, DynDims],
        mut scratch: DynamicsScratch[DT, DynDims, 1],
    ) raises:
        """`BaseController.update`'s `sim.forward()` — FK products, body
        velocities, `subtree_com`, `cdof`, CRBA and the RNE bias, at the
        current state. Everything `run()` reads."""
        osc_refresh_dynamics["cpu", DT, DynDims, 1](d, m, scratch)

    def reset(
        mut self,
        mut d: Data[DT, DynDims, 1],
        mut m: Model[DT, DynDims],
    ) raises:
        """Goal = the current eef pose, nullspace target = the current
        joints. Needs `update()` first."""
        osc_reset_gpu[DT](
            self.state_t.tt_dyn["cpu", DYN2](rl2(1, OSC_STATE_WORDS)),
            self.work_t.tt_dyn["cpu", DYN2](rl2(1, OSC_WORK_WORDS)),
            self.refs_t.tt_dyn["cpu", DYN1](rl1(OSC_REF_WORDS)),
            d.qpos.tt_dyn["cpu", DYN2](rl2(1, self.nq)),
            d.xquat.tt_dyn["cpu", DYN2](rl2(1, d.dims.get_nbody() * 4)),
            d.site_xpos.tt_dyn["cpu", DYN2](rl2(1, d.dims.get_nsite() * 3)),
            m.sites.tt_dyn["cpu", DYN2](
                rl2(d.dims.get_nsite(), MODEL_SITE_SIZE)
            ),
            0,
        )

    def set_goal(
        mut self,
        action: List[Float64],
        mut d: Data[DT, DynDims, 1],
        mut m: Model[DT, DynDims],
    ) raises:
        """`set_goal(action)`, on a policy step, after `update()`."""
        if len(action) < 6:
            raise Error("osc: set_goal needs at least six numbers")
        self._write_action(action)
        osc_set_goal_gpu[DT](
            self.state_t.tt_dyn["cpu", DYN2](rl2(1, OSC_STATE_WORDS)),
            self.work_t.tt_dyn["cpu", DYN2](rl2(1, OSC_WORK_WORDS)),
            self.refs_t.tt_dyn["cpu", DYN1](rl1(OSC_REF_WORDS)),
            self.act_t.tt_dyn["cpu", DYN2](rl2(1, OSC_ACTION_DIM)),
            d.xquat.tt_dyn["cpu", DYN2](rl2(1, d.dims.get_nbody() * 4)),
            d.site_xpos.tt_dyn["cpu", DYN2](rl2(1, d.dims.get_nsite() * 3)),
            m.sites.tt_dyn["cpu", DYN2](
                rl2(d.dims.get_nsite(), MODEL_SITE_SIZE)
            ),
            0,
        )

    def run(
        mut self,
        action: List[Float64],
        mut d: Data[DT, DynDims, 1],
        mut m: Model[DT, DynDims],
        mut scratch: DynamicsScratch[DT, DynDims, 1],
    ) raises -> List[Float64]:
        """`run_controller` + `clip_torques` + `grip_action` — the whole
        `ctrl` vector for this substep. RAISES on a singular
        operational-space inertia, which is what the kernel's flag means."""
        var nb = d.dims.get_nbody()
        var ns = d.dims.get_nsite()
        var nj = d.dims.get_njoint()
        self._write_action(action)
        osc_run_gpu[DT](
            self.state_t.tt_dyn["cpu", DYN2](rl2(1, OSC_STATE_WORDS)),
            self.work_t.tt_dyn["cpu", DYN2](rl2(1, OSC_WORK_WORDS)),
            self.refs_t.tt_dyn["cpu", DYN1](rl1(OSC_REF_WORDS)),
            self.ctrl_t.tt_dyn["cpu", DYN2](rl2(1, self.nact)),
            d.qpos.tt_dyn["cpu", DYN2](rl2(1, self.nq)),
            d.qvel.tt_dyn["cpu", DYN2](rl2(1, self.nv)),
            d.xquat.tt_dyn["cpu", DYN2](rl2(1, nb * 4)),
            d.site_xpos.tt_dyn["cpu", DYN2](rl2(1, ns * 3)),
            d.subtree_com.tt_dyn["cpu", DYN2](rl2(1, nb * 3)),
            scratch.cdof.tt_dyn["cpu", DYN2](rl2(1, self.nv * 6)),
            scratch.M.tt_dyn["cpu", DYN2](rl2(1, self.nv * self.nv)),
            scratch.bias.tt_dyn["cpu", DYN2](rl2(1, self.nv)),
            m.joints.tt_dyn["cpu", DYN2](rl2(nj, MODEL_JOINT_SIZE)),
            m.bodies.tt_dyn["cpu", DYN2](rl2(nb, MODEL_BODY_SIZE)),
            m.sites.tt_dyn["cpu", DYN2](rl2(ns, MODEL_SITE_SIZE)),
            m.meta.tt_dyn["cpu", DYN1](rl1(MODEL_META_SIZE)),
            0,
            self.nv,
        )
        if self.state_t.data[OSC_IDX_SINGULAR] != Scalar[DT](0):
            raise Error(
                "osc: the operational-space inertia is singular at this"
                " configuration. robosuite's `pinv` would return a"
                " pseudo-inverse and a plausible torque; this refuses."
            )
        osc_gripper_gpu[DT, ACT_DIM=OSC_ACTION_DIM](
            self.state_t.tt_dyn["cpu", DYN2](rl2(1, OSC_STATE_WORDS)),
            self.refs_t.tt_dyn["cpu", DYN1](rl1(OSC_REF_WORDS)),
            self.ctrl_t.tt_dyn["cpu", DYN2](rl2(1, self.nact)),
            self.act_t.tt_dyn["cpu", DYN2](rl2(1, OSC_ACTION_DIM)),
            0,
        )
        var out = List[Float64]()
        for i in range(self.nact):
            out.append(Float64(self.ctrl_t.data[i]))
        return out^

    # ── read-back, for a trace or a gate ──────────────────────────────────

    def goal_pos(self) -> List[Float64]:
        var out = List[Float64]()
        for k in range(3):
            out.append(Float64(self.state_t.data[OSC_IDX_GOAL_POS + k]))
        return out^

    def goal_mat(self) -> List[Float64]:
        var out = List[Float64]()
        for k in range(9):
            out.append(Float64(self.state_t.data[OSC_IDX_GOAL_MAT + k]))
        return out^

    def initial_joint(self) -> List[Float64]:
        var out = List[Float64]()
        for k in range(OSC_ARM):
            out.append(Float64(self.state_t.data[OSC_IDX_Q0 + k]))
        return out^

    def ee_pos(self, d: Data[DT, DynDims, 1]) -> List[Float64]:
        """The grip site's world position, straight from `Data` — the same
        words `osc_run_gpu` reads."""
        var site = Int(Float64(self.refs_t.data[OSC_REF_SITE]))
        var out = List[Float64]()
        for k in range(3):
            out.append(Float64(d.site_xpos.data[site * 3 + k]))
        return out^

    def jac(self) -> List[Float64]:
        """The 6 x 7 arm Jacobian block of the last `run()`."""
        var out = List[Float64]()
        for k in range(OSC_WRENCH * OSC_ARM):
            out.append(Float64(self.work_t.data[OSC_W_JAC + k]))
        return out^

    def mass_block(self) -> List[Float64]:
        """The 7 x 7 arm mass block of the last `run()`, armature INCLUDED."""
        var out = List[Float64]()
        for k in range(OSC_ARM * OSC_ARM):
            out.append(Float64(self.work_t.data[OSC_W_MBLK + k]))
        return out^

    def is_ready(self) -> Bool:
        return self.state_t.data[OSC_IDX_READY] != Scalar[DT](0)

    def state_words(self) -> List[Float64]:
        var out = List[Float64]()
        for k in range(OSC_STATE_WORDS):
            out.append(Float64(self.state_t.data[k]))
        return out^


struct PandaGripperRamp(Movable & Deinitable):
    """`osc_gripper_gpu` on a one-lane tensor — the ramp, standalone, for a
    caller that drives the gripper without the arm (and for its gate)."""

    var refs_t: TensorImpl[DT]
    var state_t: TensorImpl[DT]
    var ctrl_t: TensorImpl[DT]
    var act_t: TensorImpl[DT]

    def __init__(
        out self,
        ctrl0_min: Float64, ctrl0_max: Float64,
        ctrl1_min: Float64, ctrl1_max: Float64,
        speed: Float64 = 0.01,
    ) raises:
        var zeros = List[Int]()
        var fzeros = List[Float64]()
        for _ in range(OSC_ARM):
            zeros.append(0)
            fzeros.append(0.0)
        var refs = build_osc_refs(
            zeros, zeros, zeros, fzeros, fzeros, zeros, 0, 0, 0, 1,
            ctrl0_min, ctrl0_max, ctrl1_min, ctrl1_max,
            150.0, 1.0, 0.05, 0.5, 10.0, speed,
        )
        self.refs_t = TensorImpl[DT].alloc(OSC_REF_WORDS)
        for i in range(OSC_REF_WORDS):
            self.refs_t.data[i] = Scalar[DT](refs[i])
        self.state_t = TensorImpl[DT].alloc(OSC_STATE_WORDS)
        for i in range(OSC_STATE_WORDS):
            self.state_t.data[i] = Scalar[DT](0)
        self.ctrl_t = TensorImpl[DT].alloc(2)
        self.ctrl_t.data[0] = Scalar[DT](0)
        self.ctrl_t.data[1] = Scalar[DT](0)
        self.act_t = TensorImpl[DT].alloc(OSC_ACTION_DIM)
        for i in range(OSC_ACTION_DIM):
            self.act_t.data[i] = Scalar[DT](0)

    def __init__(out self, *, deinit move: Self):
        self.refs_t = move.refs_t^
        self.state_t = move.state_t^
        self.ctrl_t = move.ctrl_t^
        self.act_t = move.act_t^

    def step(mut self, action: Float64) raises -> List[Float64]:
        """One SIM step of the ramp; returns the two finger ctrls."""
        self.act_t.data[OSC_ACTION_DIM - 1] = Scalar[DT](action)
        osc_gripper_gpu[DT, ACT_DIM=OSC_ACTION_DIM](
            self.state_t.tt_dyn["cpu", DYN2](rl2(1, OSC_STATE_WORDS)),
            self.refs_t.tt_dyn["cpu", DYN1](rl1(OSC_REF_WORDS)),
            self.ctrl_t.tt_dyn["cpu", DYN2](rl2(1, 2)),
            self.act_t.tt_dyn["cpu", DYN2](rl2(1, OSC_ACTION_DIM)),
            0,
        )
        var out = List[Float64]()
        out.append(Float64(self.ctrl_t.data[0]))
        out.append(Float64(self.ctrl_t.data[1]))
        return out^

    def current(self) -> List[Float64]:
        var out = List[Float64]()
        out.append(Float64(self.state_t.data[OSC_IDX_GRIP0]))
        out.append(Float64(self.state_t.data[OSC_IDX_GRIP1]))
        return out^
