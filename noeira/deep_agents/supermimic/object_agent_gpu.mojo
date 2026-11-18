"""SMObjAgentGpu — the object interface's loop, everything on the device.

`agent_gpu.SuperMimicAgentGpu`'s loop over a simulator that carries one free
object (`object_env_gpu.SMObjTrackingEnvGpu`), with the object interface's
trainer (`object_trainer.SMObjTrainer`). `docs/OBJECT_INTERACTION_PLAN.md`
§4, step O1.4. Per iteration:

    goals, the object's reference row of each lane's frame
    a = dec([enc_motion(goal) + gain · enc_obj(features, hist), hist])
    one control step, both rings written
    every 1 / utd steps:
        DEF_WM_BATCHES batches of k-step windows → the object network's step
        (the tracker's dynamics frozen beside it)
        one batch of starts → the object encoder's step through the imagined
        rollout, then the interface critic's fit on it

    var base = SuperMimicTrainer[OC.BASE].for_robot(R.SIM.create(), Optional(ctx))
    _ = base.load_state(tracker_ckpt)
    var orefs = sm_object_ref_rows[R](base.tables, lib_rows, track)     # before the tables go to the device
    var tr = SMObjTrainer[OC](base^)
    tr.load_object(object_fit_ckpt)
    var agent = SMObjAgentGpu[OC, BS](ctx, lib^, track, orefs, BS.create(ctx), tr^, sched^)
    agent.train(iters, log_every, ...)

What stays on the host per step: the reset sampler's small download (as the
tracker's agent), and on a logged step the object's tracking error per lane.
"""

from std.math import acos, ceildiv, sqrt
from std.random import random_ui64
from std.time import perf_counter_ns

from max.gpu import global_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from noeira.deep_agents.supermimic.agent import SMSchedule, sm_cosine_lr
from noeira.deep_agents.supermimic.agent_gpu import _k_ref_root, _k_start_inputs
from noeira.deep_agents.supermimic.env_gpu import SMReplayGpuOf
from noeira.deep_agents.supermimic.kinematics import FP
from noeira.deep_agents.supermimic.lanes import talloc, tp
from noeira.deep_agents.supermimic.motion_library import SM_REF_SCRATCH, SMMotionLibrary
from noeira.deep_agents.supermimic.object_env_gpu import SMObjReplayGpu, SMObjTrackingEnvGpu
from noeira.deep_agents.supermimic.object_imagination import SMObjImagBatch
from noeira.deep_agents.supermimic.object_model import SMObjLoss, SMObjWindowBatch
from noeira.deep_agents.supermimic.object_policy import SMObjConfigT
from noeira.deep_agents.supermimic.object_spec import SM_OBJ, SM_OBJ_CTX
from noeira.deep_agents.supermimic.object_task import (
    SM_OBJ_NEAR, SM_OREF, SM_ORW, sm_obj_reward_params, sm_squeeze_lane, sm_squeeze_weights,
)
from noeira.deep_agents.supermimic.object_trainer import SMObjTrainer, SMObjUpdate
from noeira.deep_agents.supermimic.robot import SMDims, TrackingBatchedSim, TrackingRobot
from noeira.deep_agents.supermimic.world_model import SMWindowBatch
from noeira.nn.constants import DT
from noeira.nn.core.ptr import mptr
from noeira.nn.core.tensor import TensorImpl


comptime FPi = Pointer[Int32, MutAnyOrigin]
comptime _TPB: Int = 64
comptime SM_OBJ_OM_LR: Float64 = 5e-5
"""The object network starts fitted (O1.3) and keeps fitting on the
interface's replay at this rate (the prompt interface's choice for its world
model, `prompt_agent.SM_P_WM_LR`, is the precedent)."""
comptime SM_OBJ_TERMS_LEAD: Int = 8          # steps before a logged one on which an update also returns the object terms


def _k_oref_lanes(nl: Int64, frame: FPi, orows: FP[DT], dst: FP[DT]):
    """dst[l] = the object's reference row of lane l's frame."""
    var l = Int(global_idx.x)
    if l >= Int(nl):
        return
    var f = Int(frame[unsafe_offset=l])
    for k in range(SM_OREF):
        dst[unsafe_offset=l * SM_OREF + k] = orows[unsafe_offset=f * SM_OREF + k]


def _k_oref_starts(nb: Int64, h: Int64, last: FPi, frame: FPi, orows: FP[DT], orefs: FP[DT]):
    """Start b: the object's reference rows of frames min(frame + t, clip end),
    t = 0..h (`agent_gpu._k_start_inputs`'s frames)."""
    var b = Int(global_idx.x)
    var N = Int(nb)
    if b >= N:
        return
    var f0 = Int(frame[unsafe_offset=b])
    var lf = Int(last[unsafe_offset=f0])
    for t in range(Int(h) + 1):
        var f = min(f0 + t, lf)
        for k in range(SM_OREF):
            orefs[unsafe_offset=(t * N + b) * SM_OREF + k] = orows[unsafe_offset=f * SM_OREF + k]


def _k_obj_err(nl: Int64, o: FP[DT], oref: FP[DT], res: FP[DT]):
    """res[l] = (|p − p_ref|, |q · q_ref|) of lane l's object after the step
    against the row the step aimed at."""
    var l = Int(global_idx.x)
    if l >= Int(nl):
        return
    var d2 = Scalar[DT](0)
    var dot = Scalar[DT](0)
    var n1 = Scalar[DT](0)
    var n2 = Scalar[DT](0)
    for k in range(3):
        var d = o[unsafe_offset=l * SM_OBJ + k] - oref[unsafe_offset=l * SM_OREF + k]
        d2 += d * d
    for k in range(4):
        var a = o[unsafe_offset=l * SM_OBJ + 3 + k]
        var b = oref[unsafe_offset=l * SM_OREF + 3 + k]
        dot += a * b
        n1 += a * a
        n2 += b * b
    res[unsafe_offset=2 * l] = sqrt(d2)
    res[unsafe_offset=2 * l + 1] = abs(dot) / sqrt(max(n1 * n2, Scalar[DT](1e-12)))


def _k_squeeze[R: TrackingRobot](
    nl: Int64, arm0: Int64, scale: Scalar[DT], tree: FP[DT], table: FP[DT], frame: FPi, weight: FP[DT], force: FP[DT], ks: FP[DT],
    s: FP[DT], o: FP[DT], scratch: FP[DT], actions: FP[DT],
):
    """Lane l's action += `object_task.sm_squeeze_lane` with scale · force[l] ·
    weight[frame[l]] newtons (`SMObjSqueeze`). scratch: S + 2 · FK_BLOCK + 6
    per lane."""
    comptime D = SMDims[R]
    comptime SC = D.S + 2 * D.FK_BLOCK + 6
    var l = Int(global_idx.x)
    if l >= Int(nl):
        return
    sm_squeeze_lane[R](
        Int(arm0), scale * force[unsafe_offset=l] * weight[unsafe_offset=Int(frame[unsafe_offset=l])], tree, table, ks,
        s.unsafe_offset(l * D.S), o.unsafe_offset(l * SM_OBJ), scratch.unsafe_offset(l * SC), actions.unsafe_offset(l * R.NDOF),
    )


struct SMObjSqueeze[R: TrackingRobot](Movable):
    """A scripted squeeze on the device: an EXPLORATION PRIOR, not a policy.

    While a lane's reference box is lifted (`object_task.sm_squeeze_weights`),
    each hand within `SM_SQ_REACH` of the SIMULATED box's centre is pushed
    toward that centre with the lane's force, as joint-target offsets on the
    joints `arm0 ..`: Jᵀ F over (stiffness · action scale), the hand's
    Jacobian by finite differences of the FK. `supermimic_eval_cpu --squeeze`
    is the same rule on the CPU (`docs/OBJECT_INTERACTION_PLAN.md` §6: it
    lifts the box on half the clips at 40 N; the blind tracker on none).

    Why: the reference's hands touch the box and do not press, the training
    noise is a tenth of the action a hold needs, so no replay row shows a held
    box and the object network cannot give a gradient toward one. Lanes with
    DIFFERENT forces show it how the box's slip depends on the squeeze.

    `ks[j]`: N·m per action unit of joint j (stiffness · action scale)."""
    var arm0: Int
    var weight: DeviceBuffer[DT]          # [frames]
    var force: DeviceBuffer[DT]           # [lanes] newtons
    var ks: DeviceBuffer[DT]              # [NDOF]
    var scratch: DeviceBuffer[DT]
    var lanes: Int
    var on: Bool

    def __init__(out self, ctx: DeviceContext) raises:
        """Off: `apply` does nothing."""
        self.arm0 = 0
        self.lanes = 0
        self.on = False
        self.weight = ctx.enqueue_create_buffer[DT](1)
        self.force = ctx.enqueue_create_buffer[DT](1)
        self.ks = ctx.enqueue_create_buffer[DT](1)
        self.scratch = ctx.enqueue_create_buffer[DT](1)

    def __init__(
        out self, ctx: DeviceContext, lanes: Int, lens: List[Int], track: List[Scalar[DT]], stride: Int, forces: List[Float64],
        ks: List[Float64], arm0: Int,
    ) raises:
        comptime D = SMDims[Self.R]
        self.on = True
        if len(forces) != lanes or len(ks) != Self.R.NDOF or arm0 < 0 or arm0 >= Self.R.NDOF:
            raise Error("SMObjSqueeze: " + String(lanes) + " lanes need as many forces, " + String(Self.R.NDOF) + " joints as many `ks`")
        var w = sm_squeeze_weights(lens, track, stride)
        self.arm0 = arm0
        self.lanes = lanes
        self.weight = ctx.enqueue_create_buffer[DT](len(w))
        var hw = ctx.enqueue_create_host_buffer[DT](len(w))
        for i in range(len(w)):
            hw[i] = Scalar[DT](w[i])
        ctx.enqueue_copy(self.weight, hw)
        self.force = ctx.enqueue_create_buffer[DT](lanes)
        var hf = ctx.enqueue_create_host_buffer[DT](lanes)
        for i in range(lanes):
            hf[i] = Scalar[DT](forces[i])
        ctx.enqueue_copy(self.force, hf)
        self.ks = ctx.enqueue_create_buffer[DT](Self.R.NDOF)
        var hk = ctx.enqueue_create_host_buffer[DT](Self.R.NDOF)
        for i in range(Self.R.NDOF):
            hk[i] = Scalar[DT](ks[i])
        ctx.enqueue_copy(self.ks, hk)
        self.scratch = ctx.enqueue_create_buffer[DT](lanes * (D.S + 2 * D.FK_BLOCK + 6))
        ctx.synchronize()

    def __init__(out self, *, deinit move: Self):
        self.arm0 = move.arm0
        self.weight = move.weight^
        self.force = move.force^
        self.ks = move.ks^
        self.scratch = move.scratch^
        self.lanes = move.lanes
        self.on = move.on

    def apply(
        mut self, ctx: DeviceContext, tree: FP[DT], table: FP[DT], frame: FPi, s: FP[DT], o: FP[DT], actions: FP[DT],
        scale: Float64 = 1.0,
    ) raises:
        """actions [lanes · NDOF] (device) += the squeeze, in place. `frame`,
        `s`, `o`: every lane's frame, state and object BEFORE the step.
        `scale` multiplies every lane's force (a prior that fades)."""
        if not self.on or scale <= 0.0:
            return
        ctx.enqueue_function[_k_squeeze[Self.R]](
            Int64(self.lanes), Int64(self.arm0), Scalar[DT](scale), tree, table, frame, mptr(self.weight.unsafe_ptr()), mptr(self.force.unsafe_ptr()),
            mptr(self.ks.unsafe_ptr()), s, o, mptr(self.scratch.unsafe_ptr()), actions, grid_dim=ceildiv(self.lanes, _TPB), block_dim=_TPB,
        )


@fieldwise_init
struct SMObjLog(Copyable, Movable):
    var updated: Bool
    var sim_reward: Float64       # the tracker's eight terms in the simulator, mean per lane
    var falls: Int
    var ended: Int
    # the object in the simulator, on a logged step (−1 otherwise): mean
    # distance to its reference (m), mean angle (rad), share of lanes within
    # `SM_OBJ_NEAR`
    var obj_pos_err: Float64
    var obj_ang_err: Float64
    var obj_near: Float64
    var om: SMObjLoss
    var pi: SMObjUpdate


struct SMObjAgentGpu[OC: SMObjConfigT, BS: TrackingBatchedSim](Movable):
    var ctx: DeviceContext
    var env: SMObjTrackingEnvGpu[Self.OC.BASE.ROBOT, Self.BS]
    var rb: SMReplayGpuOf[Self.OC.BASE.ROBOT]
    var orb: SMObjReplayGpu
    var trainer: SMObjTrainer[Self.OC]
    var sched: SMSchedule
    var om_lr: Float64
    var iteration: Int
    var resumed_at: Int
    var updates: Int
    var utd_acc: Float64
    var orows: DeviceBuffer[DT]           # the object's reference rows, [frames · 19]
    var oref_l: TensorImpl[DT]            # [L · 19] each lane's row for the step being taken
    var oerr: TensorImpl[DT]              # [L · 2]
    var goals: TensorImpl[DT]
    var wb: SMWindowBatch[Self.OC.BASE.ROBOT]
    var wob: SMObjWindowBatch
    var ib: SMObjImagBatch[Self.OC.BASE.ROBOT]
    var w_idx: DeviceBuffer[DType.int32]
    var w_frame: DeviceBuffer[DType.int32]
    var w_aclean: DeviceBuffer[DT]
    var w_reward: DeviceBuffer[DT]
    var s_idx: DeviceBuffer[DType.int32]
    var s_frame: DeviceBuffer[DType.int32]
    var s_hs: DeviceBuffer[DT]
    var s_ha: DeviceBuffer[DT]
    var s_scratch: DeviceBuffer[DT]
    var squeeze: SMObjSqueeze[Self.OC.BASE.ROBOT]     # an exploration prior on the executed actions (off as made)
    # > 0: the squeeze's forces fade linearly to zero at this iteration. Once the interface squeezes by itself,
    # a scripted squeeze on top is too much, and the interface cannot tell a pushed lane from a plain one.
    var squeeze_until: Int

    def __init__(
        out self, ctx: DeviceContext, var lib: SMMotionLibrary[Self.OC.BASE.ROBOT], track: List[Scalar[DT]],
        orefs: List[Scalar[DT]], var sim: Self.BS, var trainer: SMObjTrainer[Self.OC], var sched: SMSchedule,
    ) raises:
        """`track`: `sm_object_track`'s rows for `lib`'s frames; `orefs`:
        `object_task.sm_object_ref_rows` of the same frames."""
        comptime B = Self.OC.BASE
        comptime R = B.ROBOT
        comptime D = SMDims[R]
        comptime K = B.DEF_WM_K
        comptime NW = B.WM_BATCH
        comptime BP = B.PI_BATCH
        comptime H = B.HORIZON
        comptime L = B.LANES
        comptime assert B.TARGET == "gpu", "SMObjAgentGpu needs a GPU config"
        comptime assert L == Self.BS.N, "the config's lanes must be the simulator's"
        if len(orefs) != lib.num_frames * SM_OREF:
            raise Error("SMObjAgentGpu: " + String(lib.num_frames) + " library frames, " + String(len(orefs) // SM_OREF) + " object reference rows")
        self.ctx = ctx
        var noise = sched.noise
        self.orows = ctx.enqueue_create_buffer[DT](len(orefs))
        var ht = ctx.enqueue_create_host_buffer[DT](len(orefs))
        for i in range(len(orefs)):
            ht[i] = orefs[i]
        ctx.enqueue_copy(self.orows, ht)
        self.env = SMObjTrackingEnvGpu[R, Self.BS](
            ctx, lib^, track, sim^, dr=noise, obs_noise=noise, reset_noise=noise,
            action_noise=0.1 if noise else 0.0, rng_seed=sched.seed, friction_dr=noise,
        )
        self.rb = SMReplayGpuOf[R](ctx, L, sched.replay_capacity)
        self.orb = SMObjReplayGpu(ctx, L, sched.replay_capacity)
        self.trainer = trainer^
        self.sched = sched^
        self.om_lr = SM_OBJ_OM_LR
        self.iteration = 0
        self.resumed_at = 0
        self.updates = 0
        self.utd_acc = 0.0
        self.oref_l = talloc["gpu"](L * SM_OREF, Optional(ctx))
        self.oerr = talloc["gpu"](L * 2, Optional(ctx))
        self.goals = talloc["gpu"](L * D.GOAL, Optional(ctx))
        self.wb = SMWindowBatch[R](K, NW)
        self.wb.states.ensure_gpu(ctx, (K + 1) * NW * D.S)
        self.wb.a_exec.ensure_gpu(ctx, K * NW * R.NDOF)
        self.wb.xi.ensure_gpu(ctx, NW * D.X)
        self.wb.done.ensure_gpu(ctx, K * NW)
        self.wb.ref_root.ensure_gpu(ctx, K * NW * 7)
        self.wob = SMObjWindowBatch(K, NW)
        self.wob.states.ensure_gpu(ctx, (K + 1) * NW * SM_OBJ)
        self.wob.ctx.ensure_gpu(ctx, NW * SM_OBJ_CTX)
        self.ib = SMObjImagBatch[R](BP, H)
        self.ib.to_device(ctx)
        self.w_idx = ctx.enqueue_create_buffer[DType.int32](NW)
        self.w_frame = ctx.enqueue_create_buffer[DType.int32](K * NW)
        self.w_aclean = ctx.enqueue_create_buffer[DT](K * NW * R.NDOF)
        self.w_reward = ctx.enqueue_create_buffer[DT](K * NW)
        self.s_idx = ctx.enqueue_create_buffer[DType.int32](BP)
        self.s_frame = ctx.enqueue_create_buffer[DType.int32](BP)
        self.s_hs = ctx.enqueue_create_buffer[DT](D.HIST * BP * D.S)
        self.s_ha = ctx.enqueue_create_buffer[DT](D.HIST * BP * R.NDOF)
        self.s_scratch = ctx.enqueue_create_buffer[DT](BP * SM_REF_SCRATCH[R])
        self.squeeze = SMObjSqueeze[R](ctx)
        self.squeeze_until = 0
        ctx.synchronize()
        self.set_reward(sm_obj_reward_params())

    def __init__(out self, *, deinit move: Self):
        self.ctx = move.ctx
        self.env = move.env^
        self.rb = move.rb^
        self.orb = move.orb^
        self.trainer = move.trainer^
        self.sched = move.sched^
        self.om_lr = move.om_lr
        self.iteration = move.iteration
        self.resumed_at = move.resumed_at
        self.updates = move.updates
        self.utd_acc = move.utd_acc
        self.orows = move.orows^
        self.oref_l = move.oref_l^
        self.oerr = move.oerr^
        self.goals = move.goals^
        self.wb = move.wb^
        self.wob = move.wob^
        self.ib = move.ib^
        self.w_idx = move.w_idx^
        self.w_frame = move.w_frame^
        self.w_aclean = move.w_aclean^
        self.w_reward = move.w_reward^
        self.squeeze = move.squeeze^
        self.squeeze_until = move.squeeze_until
        self.s_idx = move.s_idx^
        self.s_frame = move.s_frame^
        self.s_hs = move.s_hs^
        self.s_ha = move.s_ha^
        self.s_scratch = move.s_scratch^

    def set_reward(mut self, rw: List[Scalar[DT]]) raises:
        """The object reward's data (`object_task.sm_obj_reward_params`)."""
        if len(rw) != SM_ORW:
            raise Error("SMObjAgentGpu.set_reward: " + String(SM_ORW) + " numbers, got " + String(len(rw)))
        for k in range(SM_ORW):
            self.ib.rw.data[k] = rw[k]
        self.ib.rw.upload_resident(self.ctx)

    def resume(mut self, iteration: Int, updates: Int):
        """Go on from a saved run: the counts and the cosine schedule continue;
        the replay is empty again, so the warm-up runs again."""
        self.iteration = iteration
        self.resumed_at = iteration
        self.updates = updates

    def _lane_refs(mut self) raises:
        comptime L = Self.OC.BASE.LANES
        self.ctx.enqueue_function[_k_oref_lanes](
            Int64(L), mptr(self.env.env.frame.unsafe_ptr()), mptr(self.orows.unsafe_ptr()), tp["gpu"](self.oref_l),
            grid_dim=ceildiv(L, _TPB), block_dim=_TPB,
        )

    def _window_batch(mut self) raises:
        comptime R = Self.OC.BASE.ROBOT
        comptime K = Self.OC.BASE.DEF_WM_K
        comptime NW = Self.OC.BASE.WM_BATCH
        var ctx = self.ctx
        self.rb.sample_windows(ctx, K, NW, random_ui64(0, 0xFFFFFFFFFFFF), mptr(self.w_idx.unsafe_ptr()))
        if self.rb.eligible_total(ctx) == 0:
            raise Error("SMObjAgentGpu: no sampleable windows yet (raise the warm-up)")
        self.rb.windows_at(
            ctx, mptr(self.w_idx.unsafe_ptr()), NW, K, tp["gpu"](self.wb.states), tp["gpu"](self.wb.a_exec),
            mptr(self.w_aclean.unsafe_ptr()), tp["gpu"](self.wb.xi), mptr(self.w_reward.unsafe_ptr()),
            tp["gpu"](self.wb.done), mptr(self.w_frame.unsafe_ptr()),
        )
        ctx.enqueue_function[_k_ref_root[SMDims[R].S]](
            Int64(K * NW), mptr(self.env.env.lib.rows_dev.value().unsafe_ptr()), mptr(self.w_frame.unsafe_ptr()), tp["gpu"](self.wb.ref_root),
            grid_dim=ceildiv(K * NW, _TPB), block_dim=_TPB,
        )
        self.orb.windows_at(ctx, mptr(self.w_idx.unsafe_ptr()), NW, K, tp["gpu"](self.wob.states), tp["gpu"](self.wob.ctx))

    def _start_batch(mut self) raises:
        comptime R = Self.OC.BASE.ROBOT
        comptime D = SMDims[R]
        comptime BP = Self.OC.BASE.PI_BATCH
        comptime H = Self.OC.BASE.HORIZON
        var ctx = self.ctx
        self.rb.sample_starts(ctx, D.HIST, BP, random_ui64(0, 0xFFFFFFFFFFFF), mptr(self.s_idx.unsafe_ptr()))
        if self.rb.eligible_total(ctx) == 0:
            raise Error("SMObjAgentGpu: no sampleable starts yet (raise the warm-up)")
        self.rb.starts_at(
            ctx, mptr(self.s_idx.unsafe_ptr()), BP, D.HIST, tp["gpu"](self.ib.base.state), tp["gpu"](self.ib.base.xi),
            mptr(self.s_frame.unsafe_ptr()), tp["gpu"](self.ib.base.prev_action), mptr(self.s_hs.unsafe_ptr()), mptr(self.s_ha.unsafe_ptr()),
        )
        ctx.enqueue_function[_k_start_inputs[R]](
            Int64(BP), Int64(H), UInt32(random_ui64(1, 0xFFFFFFFF)), Int64(1 if self.sched.noise else 0),
            mptr(self.env.env.lib.rows_dev.value().unsafe_ptr()), self.trainer.base.tables.tree_p["gpu"](), self.trainer.base.tables.head_p["gpu"](),
            mptr(self.env.env.lib.last_dev.value().unsafe_ptr()), mptr(self.s_frame.unsafe_ptr()),
            mptr(self.s_hs.unsafe_ptr()), mptr(self.s_ha.unsafe_ptr()), tp["gpu"](self.ib.base.xi), mptr(self.s_scratch.unsafe_ptr()),
            tp["gpu"](self.ib.base.hist0), tp["gpu"](self.ib.base.goals), tp["gpu"](self.ib.base.refs),
            grid_dim=ceildiv(BP, _TPB), block_dim=_TPB,
        )
        # the object at the start row (a window of 0 steps), and its reference rows
        self.orb.windows_at(ctx, mptr(self.s_idx.unsafe_ptr()), BP, 0, tp["gpu"](self.ib.obj0), tp["gpu"](self.ib.octx))
        ctx.enqueue_function[_k_oref_starts](
            Int64(BP), Int64(H), mptr(self.env.env.lib.last_dev.value().unsafe_ptr()), mptr(self.s_frame.unsafe_ptr()),
            mptr(self.orows.unsafe_ptr()), tp["gpu"](self.ib.orefs), grid_dim=ceildiv(BP, _TPB), block_dim=_TPB,
        )

    def iterate(mut self, log: Bool = False, terms: Bool = False) raises -> SMObjLog:
        """One env step, and the updates it is owed. `log`: also measure the
        object's tracking error in the simulator (a small download); `log` or
        `terms`: an update of this step also returns the imagined rollout's
        object terms (another).

        ⚠ Updates fall on 0.4 of the steps, summed in floats: a logged step
        need not be one of them. `train` asks for the terms on the steps
        leading to a logged one and keeps the last it got."""
        comptime B = Self.OC.BASE
        comptime L = B.LANES
        var ctx = self.ctx
        self.env.env.policy_inputs(ctx, self.goals)
        self._lane_refs()
        var a = self.trainer.act[L](self.env.env.hist, self.goals, self.env.env.s, self.env.prev_o, self.oref_l, resident=True)
        self.squeeze.apply(
            ctx, self.trainer.base.tables.tree_p["gpu"](), self.trainer.base.tables.head_p["gpu"](), mptr(self.env.env.frame.unsafe_ptr()),
            tp["gpu"](self.env.env.s), tp["gpu"](self.env.prev_o), tp["gpu"](a),
            1.0 if self.squeeze_until <= 0 else max(0.0, 1.0 - Float64(self.iteration) / Float64(self.squeeze_until)),
        )
        var si = self.env.step(ctx, a, self.rb, self.orb)
        var info = SMObjLog(
            False, si.reward_mean, si.n_terminated, si.n_terminated + si.n_truncated, -1.0, -1.0, -1.0,
            SMObjLoss(0, 0, 0, 0, 0, 0, 0, 0, -1.0, -1.0, -1.0, -1.0), SMObjUpdate(0, 0, 0, 0, 0, 0, -1.0, -1.0, -1.0),
        )
        if log:
            # `env.o` is every lane's object after the step and before any reset; `oref_l` the row it aimed at
            ctx.enqueue_function[_k_obj_err](
                Int64(L), tp["gpu"](self.env.o), tp["gpu"](self.oref_l), tp["gpu"](self.oerr), grid_dim=ceildiv(L, _TPB), block_dim=_TPB,
            )
            self.oerr.download(ctx)
            var pe = 0.0
            var ae = 0.0
            var near = 0
            for l in range(L):
                var d = Float64(self.oerr.data[2 * l])
                pe += d
                ae += 2.0 * acos(min(Float64(self.oerr.data[2 * l + 1]), 1.0))
                if d <= SM_OBJ_NEAR:
                    near += 1
            info.obj_pos_err = pe / Float64(L)
            info.obj_ang_err = ae / Float64(L)
            info.obj_near = Float64(near) / Float64(L)
        self.iteration += 1
        if self.iteration - self.resumed_at >= self.sched.warmup:
            self.utd_acc += self.sched.utd
            while self.utd_acc >= 1.0:
                self.utd_acc -= 1.0
                self.trainer.set_lr(sm_cosine_lr(self.iteration, self.sched.total_iterations, self.sched.lr), self.om_lr)
                var acc = SMObjLoss(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
                for _ in range(B.DEF_WM_BATCHES):
                    self._window_batch()
                    acc.add_scaled(self.trainer.om_update[B.WM_BATCH](self.wb, self.wob, resident=True), 1.0 / Float64(B.DEF_WM_BATCHES))
                self._start_batch()
                var vals = List[Float64]()
                var tgts = List[Float64]()
                var none = List[Int]()
                info.pi = self.trainer.policy_update[B.PI_BATCH](self.ib, none, vals, tgts, self.sched.noise, resident=True, terms=log or terms)
                info.om = acc.copy()
                self.updates += 1
                info.updated = True
        return info.copy()

    def train(
        mut self, iters: Int, log_every: Int = 200, checkpoint_every: Int = 0, checkpoint_path: String = "", verbose: Bool = True,
    ) raises:
        var t0 = perf_counter_ns()
        var rs = 0.0
        var fs = 0
        var es = 0
        var n = 0
        var last_pi = SMObjUpdate(0, 0, 0, 0, 0, 0, -1.0, -1.0, -1.0)
        var last_om = SMObjLoss(0, 0, 0, 0, 0, 0, 0, 0, -1.0, -1.0, -1.0, -1.0)
        for it in range(self.iteration + 1, iters + 1):
            var logged = log_every > 0 and (it % log_every == 0 or it == iters)
            # the steps just before a logged one: one of them updates (`iterate`'s note)
            var lead = log_every > 0 and (it % log_every > log_every - 1 - SM_OBJ_TERMS_LEAD or it > iters - SM_OBJ_TERMS_LEAD)
            var r = self.iterate(logged, lead)
            rs += r.sim_reward
            fs += r.falls
            es += r.ended
            n += 1
            if r.updated:
                last_om = r.om.copy()
                # a logged step's update carries the object terms; keep them until the next one
                if r.pi.r_obj_pos >= 0.0 or last_pi.r_obj_pos < 0.0:
                    last_pi = r.pi.copy()
                else:
                    var keep = last_pi.copy()
                    last_pi = r.pi.copy()
                    last_pi.r_obj_pos = keep.r_obj_pos
                    last_pi.r_obj_ori = keep.r_obj_ori
                    last_pi.r_obj_hands = keep.r_obj_hands
            if logged:
                var dt_s = Float64(perf_counter_ns() - t0) * 1e-9
                if verbose:
                    print(
                        "it", it, "| sim: reward/step", rs / Float64(n), "fell", fs, "ended", es,
                        "| box: off its reference by", r.obj_pos_err, "m,", r.obj_ang_err, "rad, within 10 cm in", r.obj_near, "of the lanes",
                        "| imagined: J", last_pi.imag_objective, "return", last_pi.imag_return_undisc,
                        "box terms/step (pos, ori, hands)", last_pi.r_obj_pos, last_pi.r_obj_ori, last_pi.r_obj_hands,
                        "| |g|", last_pi.policy_grad_norm, "critic loss", last_pi.critic_loss, "V", last_pi.critic_value_mean,
                        "| object net: k-step", last_om.obj_nse_kstep, "robot k-step", last_om.nse_kstep,
                        "| updates", self.updates, "|", Float64(n) / max(dt_s, 1e-9), "it/s",
                    )
                rs = 0.0
                fs = 0
                es = 0
                n = 0
                t0 = perf_counter_ns()
            if checkpoint_path.byte_length() > 0 and ((checkpoint_every > 0 and it % checkpoint_every == 0) or it == iters):
                self.trainer.save_state(checkpoint_path, it, self.updates)
