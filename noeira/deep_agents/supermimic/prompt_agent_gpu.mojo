"""SMPromptAgentGpu — `prompt_agent.mojo`'s loop with everything on the device.

The same rules, as kernels: each prompt lane draws a command at a reset and
every `DEF_HOLD` steps (`_k_prompt_cmd`, a counter hash of (seed, lane)); an
imagined rollout starts from a row a prompt lane logged, the first share of the
batch reading reference commands from the row's frame, the rest holding a bank
row (`_k_prompt_start`). What stays on the host per step: the reset sampler's
small download, as in `agent_gpu.mojo`, plus the prompt lanes' task reward for
the log.

    var tr = SMPromptTrainer[P](SuperMimicTrainer[P.BASE].for_robot(R.SIM.create(), Optional(ctx)))
    _ = tr.base.load_state(tracker_ckpt)
    var data = sm_prompt_data[P.BASE, P.CMD](tr.base, lib, bank_n)
    var agent = SMPromptAgentGpu[P, BS](ctx, lib^, BS.create(ctx), tr^, data^, sched^, split)
    agent.train(iters, log_every, checkpoint_every=..., checkpoint_path=...)
"""

from std.math import ceildiv
from std.random import random_ui64
from std.time import perf_counter_ns

from max.gpu import global_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from noeira.core.logger import Logger, NoOpLogger
from noeira.deep_agents.supermimic.agent import SMSchedule, sm_cosine_lr
from noeira.deep_agents.supermimic.agent_gpu import _k_ref_root
from noeira.deep_agents.supermimic.env_gpu import SMReplayGpuOf, SMTrackingEnvGpu, ST_TERMINATED
from noeira.deep_agents.supermimic.heads import SMHeadLayout, sm_observation_lane
from noeira.deep_agents.supermimic.kinematics import FP
from noeira.deep_agents.supermimic.lanes import sm_hash_u01, talloc, tp
from noeira.deep_agents.supermimic.motion_library import SMMotionLibrary
from noeira.deep_agents.supermimic.policy import sm_obs_noise_half
from noeira.deep_agents.supermimic.prompt import SMSpec, sm_spec_velocity
from noeira.deep_agents.supermimic.prompt_agent import SM_P_ALIGN_WARM_GAIN, SM_P_WM_LR, SMPromptLog, sm_print_prompt
from noeira.deep_agents.supermimic.prompt_bank import SMPromptData
from noeira.deep_agents.supermimic.prompt_imagination import SMPromptBatch
from noeira.deep_agents.supermimic.prompt_policy import SMPromptConfigT
from noeira.deep_agents.supermimic.prompt_trainer import SMPromptTrainer, SMPromptUpdate
from noeira.deep_agents.supermimic.robot import SMDims, TrackingBatchedSim, TrackingRobot
from noeira.deep_agents.supermimic.world_model import SMWindowBatch, SMWMLoss
from noeira.nn.constants import DT
from noeira.nn.core.ptr import mptr
from noeira.nn.core.tensor import TensorImpl


comptime FPi = Pointer[Int32, MutAnyOrigin]
comptime _TPB: Int = 64


def _k_prompt_cmd[CMD: Int](
    nl: Int64, split: Int64, hold_steps: Int64, ref_share: Scalar[DT], seed: UInt32, bank_n: Int64,
    frame: FPi, ep_len: FPi, src: FPi, hold: FPi,
    cmd_ref: FP[DT], vel_ref: FP[DT], bank: FP[DT], cmds: FP[DT], specs: FP[DT],
):
    """Lane l's command and spec for this step (`SMPromptAgent._commands`)."""
    comptime RW = CMD + SMSpec.SIZE
    var l = Int(global_idx.x)
    if l >= Int(nl) or l < Int(split):
        return
    if ep_len[unsafe_offset=l] == 0 or hold[unsafe_offset=l] <= 0:
        hold[unsafe_offset=l] = Int32(hold_steps)
        if Scalar[DT](sm_hash_u01(seed, UInt32(2 * l))) < ref_share:
            src[unsafe_offset=l] = -1
        else:
            var k = Int(Scalar[DT](sm_hash_u01(seed, UInt32(2 * l + 1))) * Scalar[DT](bank_n))
            src[unsafe_offset=l] = Int32(min(k, Int(bank_n) - 1))
    hold[unsafe_offset=l] = hold[unsafe_offset=l] - 1
    var s = Int(src[unsafe_offset=l])
    var f = Int(frame[unsafe_offset=l])
    if s < 0:
        for k in range(CMD):
            cmds[unsafe_offset=l * CMD + k] = cmd_ref[unsafe_offset=f * CMD + k]
        sm_spec_velocity[DT](
            specs.unsafe_offset(l * SMSpec.SIZE), vel_ref[unsafe_offset=f * 3], vel_ref[unsafe_offset=f * 3 + 1], vel_ref[unsafe_offset=f * 3 + 2],
        )
    else:
        for k in range(CMD):
            cmds[unsafe_offset=l * CMD + k] = bank[unsafe_offset=s * RW + k]
        for k in range(SMSpec.SIZE):
            specs[unsafe_offset=l * SMSpec.SIZE + k] = bank[unsafe_offset=s * RW + CMD + k]


def _k_prompt_start[R: TrackingRobot, CMD: Int](
    nb: Int64, h: Int64, nref: Int64, seed: UInt32, noise: Int64, bank_n: Int64, table: FP[DT], last: FPi,
    frame: FPi, hist_states: FP[DT], hist_actions: FP[DT], xi: FP[DT], cmd_ref: FP[DT], vel_ref: FP[DT], lat: FP[DT], bank: FP[DT],
    hist0: FP[DT], cmds: FP[DT], specs: FP[DT], lats: FP[DT],
):
    """Start b: its observation history, then its commands, specs and motion
    latents (`SMPromptAgent.start_batch`)."""
    comptime D = SMDims[R]
    comptime n = R.NDOF
    comptime LAT = D.LATENT
    comptime RW = CMD + SMSpec.SIZE
    var b = Int(global_idx.x)
    var N = Int(nb)
    if b >= N:
        return
    for hh in range(D.HIST):
        var o = hist0.unsafe_offset((hh * N + b) * D.OBS)
        sm_observation_lane[DT, R](
            table, hist_states.unsafe_offset((hh * N + b) * D.S), hist_actions.unsafe_offset((hh * N + b) * n),
            xi.unsafe_offset(b * D.X), o,
        )
        if noise != 0:
            for k in range(D.OBS):
                var hw = sm_obs_noise_half[n, DT](k)
                if hw > 0:
                    var u = Scalar[DT](sm_hash_u01(seed, UInt32(((hh * N + b) * D.OBS + k))))
                    o[unsafe_offset=k] += hw * (Scalar[DT](2) * u - 1)
    var H = Int(h)
    if b < Int(nref):
        var f0 = Int(frame[unsafe_offset=b])
        var lf = Int(last[unsafe_offset=f0])
        for t in range(H + 1):
            var f = min(f0 + t, lf)
            for k in range(CMD):
                cmds[unsafe_offset=(t * N + b) * CMD + k] = cmd_ref[unsafe_offset=f * CMD + k]
            if t < H:
                sm_spec_velocity[DT](
                    specs.unsafe_offset((t * N + b) * SMSpec.SIZE), vel_ref[unsafe_offset=f * 3], vel_ref[unsafe_offset=f * 3 + 1], vel_ref[unsafe_offset=f * 3 + 2],
                )
                for k in range(LAT):
                    lats[unsafe_offset=(t * N + b) * LAT + k] = lat[unsafe_offset=f * LAT + k]
    else:
        var r = min(Int(Scalar[DT](sm_hash_u01(seed, UInt32(0x7FFFFF00) + UInt32(b))) * Scalar[DT](bank_n)), Int(bank_n) - 1) * RW
        for t in range(H + 1):
            for k in range(CMD):
                cmds[unsafe_offset=(t * N + b) * CMD + k] = bank[unsafe_offset=r + k]
            if t < H:
                for k in range(SMSpec.SIZE):
                    specs[unsafe_offset=(t * N + b) * SMSpec.SIZE + k] = bank[unsafe_offset=r + CMD + k]


struct SMPromptAgentGpu[P: SMPromptConfigT, BS: TrackingBatchedSim](Movable):
    var ctx: DeviceContext
    var env: SMTrackingEnvGpu[Self.P.BASE.ROBOT, Self.BS]
    var rb: SMReplayGpuOf[Self.P.BASE.ROBOT]
    var trainer: SMPromptTrainer[Self.P]
    var data: SMPromptData[Self.P.CMD]
    var sched: SMSchedule
    var split: Int
    var align_warm: Int           # updates at the alignment's raised weight
    var nref: Int
    var iteration: Int
    var updates: Int
    var utd_acc: Float64
    var wb: SMWindowBatch[Self.P.BASE.ROBOT]
    var ib: SMPromptBatch[Self.P.BASE.ROBOT, Self.P.CMD]
    var goals: TensorImpl[DT]
    var cmds: TensorImpl[DT]
    var specs: TensorImpl[DT]
    var src: DeviceBuffer[DType.int32]
    var hold: DeviceBuffer[DType.int32]
    var w_idx: DeviceBuffer[DType.int32]
    var w_frame: DeviceBuffer[DType.int32]
    var w_aclean: DeviceBuffer[DT]
    var w_reward: DeviceBuffer[DT]
    var s_idx: DeviceBuffer[DType.int32]
    var s_frame: DeviceBuffer[DType.int32]
    var s_hs: DeviceBuffer[DT]
    var s_ha: DeviceBuffer[DT]

    def __init__(
        out self, ctx: DeviceContext, var lib: SMMotionLibrary[Self.P.BASE.ROBOT], var sim: Self.BS,
        var trainer: SMPromptTrainer[Self.P], var data: SMPromptData[Self.P.CMD], var sched: SMSchedule, split: Int,
    ) raises:
        comptime B = Self.P.BASE
        comptime R = B.ROBOT
        comptime D = SMDims[R]
        comptime K = B.DEF_WM_K
        comptime NW = B.WM_BATCH
        comptime BP = B.PI_BATCH
        comptime H = B.HORIZON
        comptime L = B.LANES
        comptime assert B.TARGET == "gpu", "SMPromptAgentGpu needs a GPU config"
        comptime assert L == Self.BS.N, "the config's lanes must be the simulator's"
        self.ctx = ctx
        var noise = sched.noise
        self.env = SMTrackingEnvGpu[R, Self.BS](
            ctx, lib^, sim^, dr=noise, obs_noise=noise, reset_noise=noise,
            action_noise=0.1 if noise else 0.0, rng_seed=sched.seed,
        )
        self.env.prompt_from = split
        self.rb = SMReplayGpuOf[R](ctx, L, sched.replay_capacity)
        self.trainer = trainer^
        data.to_device(ctx)
        self.data = data^
        self.sched = sched^
        self.split = split
        self.align_warm = 0
        self.nref = Int(Float64(BP) * Self.P.DEF_REF_SHARE)
        self.iteration = 0
        self.updates = 0
        self.utd_acc = 0.0
        self.wb = SMWindowBatch[R](K, NW)
        self.wb.states.ensure_gpu(ctx, (K + 1) * NW * D.S)
        self.wb.a_exec.ensure_gpu(ctx, K * NW * R.NDOF)
        self.wb.xi.ensure_gpu(ctx, NW * D.X)
        self.wb.done.ensure_gpu(ctx, K * NW)
        self.wb.ref_root.ensure_gpu(ctx, K * NW * 7)
        self.ib = SMPromptBatch[R, Self.P.CMD](BP, H, self.nref)
        self.ib.ensure_device(ctx)
        self.goals = talloc["gpu"](L * D.GOAL, Optional(ctx))
        self.cmds = talloc["gpu"](L * Self.P.CMD, Optional(ctx))
        self.specs = talloc["gpu"](L * SMSpec.SIZE, Optional(ctx))
        self.src = ctx.enqueue_create_buffer[DType.int32](L)
        self.hold = ctx.enqueue_create_buffer[DType.int32](L)
        self.src.enqueue_fill(Int32(-1))
        self.hold.enqueue_fill(Int32(0))
        self.w_idx = ctx.enqueue_create_buffer[DType.int32](NW)
        self.w_frame = ctx.enqueue_create_buffer[DType.int32](K * NW)
        self.w_aclean = ctx.enqueue_create_buffer[DT](K * NW * R.NDOF)
        self.w_reward = ctx.enqueue_create_buffer[DT](K * NW)
        self.s_idx = ctx.enqueue_create_buffer[DType.int32](BP)
        self.s_frame = ctx.enqueue_create_buffer[DType.int32](BP)
        self.s_hs = ctx.enqueue_create_buffer[DT](D.HIST * BP * D.S)
        self.s_ha = ctx.enqueue_create_buffer[DT](D.HIST * BP * R.NDOF)
        ctx.synchronize()

    def __init__(out self, *, deinit move: Self):
        self.ctx = move.ctx
        self.env = move.env^
        self.rb = move.rb^
        self.trainer = move.trainer^
        self.data = move.data^
        self.sched = move.sched^
        self.split = move.split
        self.align_warm = move.align_warm
        self.nref = move.nref
        self.iteration = move.iteration
        self.updates = move.updates
        self.utd_acc = move.utd_acc
        self.wb = move.wb^
        self.ib = move.ib^
        self.goals = move.goals^
        self.cmds = move.cmds^
        self.specs = move.specs^
        self.src = move.src^
        self.hold = move.hold^
        self.w_idx = move.w_idx^
        self.w_frame = move.w_frame^
        self.w_aclean = move.w_aclean^
        self.w_reward = move.w_reward^
        self.s_idx = move.s_idx^
        self.s_frame = move.s_frame^
        self.s_hs = move.s_hs^
        self.s_ha = move.s_ha^

    def _commands(mut self) raises:
        comptime L = Self.P.BASE.LANES
        var ctx = self.ctx
        ctx.enqueue_function[_k_prompt_cmd[Self.P.CMD]](
            Int64(L), Int64(self.split), Int64(Self.P.DEF_HOLD), Scalar[DT](Self.P.DEF_REF_SHARE),
            UInt32(random_ui64(1, 0xFFFFFFFF)), Int64(self.data.bank_n),
            mptr(self.env.frame.unsafe_ptr()), mptr(self.env.ep_len.unsafe_ptr()), mptr(self.src.unsafe_ptr()), mptr(self.hold.unsafe_ptr()),
            tp["gpu"](self.data.cmd_ref), tp["gpu"](self.data.vel_ref), tp["gpu"](self.data.bank), tp["gpu"](self.cmds), tp["gpu"](self.specs),
            grid_dim=ceildiv(L, _TPB), block_dim=_TPB,
        )

    def _window_batch(mut self) raises:
        comptime R = Self.P.BASE.ROBOT
        comptime K = Self.P.BASE.DEF_WM_K
        comptime NW = Self.P.BASE.WM_BATCH
        var ctx = self.ctx
        self.rb.sample_windows(ctx, K, NW, random_ui64(0, 0xFFFFFFFFFFFF), mptr(self.w_idx.unsafe_ptr()))
        if self.rb.eligible_total(ctx) == 0:
            raise Error("SMPromptAgentGpu: no sampleable windows yet (raise the warm-up)")
        self.rb.windows_at(
            ctx, mptr(self.w_idx.unsafe_ptr()), NW, K, tp["gpu"](self.wb.states), tp["gpu"](self.wb.a_exec),
            mptr(self.w_aclean.unsafe_ptr()), tp["gpu"](self.wb.xi), mptr(self.w_reward.unsafe_ptr()),
            tp["gpu"](self.wb.done), mptr(self.w_frame.unsafe_ptr()),
        )
        ctx.enqueue_function[_k_ref_root[SMDims[R].S]](
            Int64(K * NW), mptr(self.env.lib.rows_dev.value().unsafe_ptr()), mptr(self.w_frame.unsafe_ptr()), tp["gpu"](self.wb.ref_root),
            grid_dim=ceildiv(K * NW, _TPB), block_dim=_TPB,
        )

    def _start_batch(mut self) raises:
        comptime R = Self.P.BASE.ROBOT
        comptime D = SMDims[R]
        comptime BP = Self.P.BASE.PI_BATCH
        var ctx = self.ctx
        self.rb.sample_starts(ctx, D.HIST, BP, random_ui64(0, 0xFFFFFFFFFFFF), mptr(self.s_idx.unsafe_ptr()), first_env=self.split)
        if self.rb.eligible_total(ctx) == 0:
            raise Error("SMPromptAgentGpu: no sampleable starts yet (raise the warm-up)")
        self.rb.starts_at(
            ctx, mptr(self.s_idx.unsafe_ptr()), BP, D.HIST, tp["gpu"](self.ib.state), tp["gpu"](self.ib.xi),
            mptr(self.s_frame.unsafe_ptr()), tp["gpu"](self.ib.prev_action), mptr(self.s_hs.unsafe_ptr()), mptr(self.s_ha.unsafe_ptr()),
        )
        ctx.enqueue_function[_k_prompt_start[R, Self.P.CMD]](
            Int64(BP), Int64(Self.P.BASE.HORIZON), Int64(self.nref), UInt32(random_ui64(1, 0xFFFFFFFF)), Int64(1 if self.sched.noise else 0),
            Int64(self.data.bank_n), self.trainer.base.tables.head_p["gpu"](), mptr(self.env.lib.last_dev.value().unsafe_ptr()),
            mptr(self.s_frame.unsafe_ptr()), mptr(self.s_hs.unsafe_ptr()), mptr(self.s_ha.unsafe_ptr()), tp["gpu"](self.ib.xi),
            tp["gpu"](self.data.cmd_ref), tp["gpu"](self.data.vel_ref), tp["gpu"](self.data.lat), tp["gpu"](self.data.bank),
            tp["gpu"](self.ib.hist0), tp["gpu"](self.ib.cmds), tp["gpu"](self.ib.specs), tp["gpu"](self.ib.lat),
            grid_dim=ceildiv(BP, _TPB), block_dim=_TPB,
        )

    def iterate(mut self) raises -> SMPromptLog:
        """One env step, and the updates it is owed."""
        comptime B = Self.P.BASE
        var ctx = self.ctx
        if self.split > 0:
            self.env.policy_inputs(ctx, self.goals)
        self._commands()
        var a = self.trainer.act[B.LANES](self.env.hist, self.goals, self.cmds, self.split, resident=True)
        _ = self.env.step(ctx, a, self.rb, tp["gpu"](self.specs))
        self.env.prompt_r.download(ctx)
        var rsum = 0.0
        var falls = 0
        for l in range(self.split, B.LANES):
            rsum += Float64(self.env.prompt_r.data[l])
            if (self.env.h_out_i[2 * l + 1] & ST_TERMINATED) != 0:
                falls += 1
        var info = SMPromptLog(
            False, rsum / Float64(max(B.LANES - self.split, 1)), falls, SMWMLoss(0, 0, 0, 0, 0), SMPromptUpdate(0, 0, 0, 0, 0, 0, 0),
        )
        self.iteration += 1
        if self.iteration >= self.sched.warmup:
            self.utd_acc += self.sched.utd
            while self.utd_acc >= 1.0:
                self.utd_acc -= 1.0
                self.trainer.set_lr(sm_cosine_lr(self.iteration, self.sched.total_iterations, self.sched.lr), SM_P_WM_LR)
                self.trainer.w_align = Self.P.DEF_ALIGN * (SM_P_ALIGN_WARM_GAIN if self.updates < self.align_warm else 1.0)
                var acc = SMWMLoss(0, 0, 0, 0, 0)
                for _ in range(B.DEF_WM_BATCHES):
                    self._window_batch()
                    acc.add_scaled(self.trainer.wm_update[B.WM_BATCH](self.wb, resident=True), 1.0 / Float64(B.DEF_WM_BATCHES))
                self._start_batch()
                var vals = List[Float64]()
                var tgts = List[Float64]()
                var none = List[Int]()
                info.pi = self.trainer.policy_update[B.PI_BATCH](self.ib, none, vals, tgts, self.sched.noise, resident=True)
                info.wm = acc^
                self.updates += 1
                info.updated = True
        return info^

    def train(
        mut self, iters: Int, log_every: Int = 200, checkpoint_every: Int = 0, checkpoint_path: String = "", verbose: Bool = True,
    ) raises:
        var t0 = perf_counter_ns()
        var rs = 0.0
        var fs = 0
        var n = 0
        var last = SMPromptUpdate(0, 0, 0, 0, 0, 0, 0)
        var lw = SMWMLoss(0, 0, 0, 0, 0)
        for it in range(1, iters + 1):
            var r = self.iterate()
            rs += r.sim_r
            fs += r.falls
            n += 1
            if r.updated:
                last = r.pi.copy()
                lw = r.wm.copy()
            if log_every > 0 and (it % log_every == 0 or it == iters):
                var dt_s = Float64(perf_counter_ns() - t0) * 1e-9
                if verbose:
                    sm_print_prompt(
                        it, self.updates, rs / Float64(n), Float64(fs) / (Float64(n) * Float64(max(Self.P.BASE.LANES - self.split, 1))),
                        lw, last, Self.P.BASE.HORIZON, Float64(n) / max(dt_s, 1e-9),
                    )
                rs = 0.0
                fs = 0
                n = 0
                t0 = perf_counter_ns()
            if checkpoint_path.byte_length() > 0 and ((checkpoint_every > 0 and it % checkpoint_every == 0) or it == iters):
                self.trainer.save_state(checkpoint_path, it, self.updates)
