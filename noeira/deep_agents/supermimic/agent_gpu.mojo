"""SuperMimicAgentGpu — the agent with everything on the device.

`agent.mojo`'s loop (Algorithm A.1, same schedule, same update cadence) over
the device env (`env_gpu.SMTrackingEnvGpu`) and the device replay
(`EpisodicReplayGpu`), feeding the trainer device-resident batches:

    var agent = sm_agent_gpu[SMPaperG1["gpu"], G1BatchedSim[4096]](
        ctx, lib^, G1BatchedSim[4096].create(ctx), total_iterations=iters)
    agent.train(iters, log_every, logger=..., checkpoint_every=..., ...)   # as SuperMimicAgent

What stays on the host per step: the reset sampler's per-lane log (one small
download, see `env_gpu.mojo`) and the scalars the trainer reports.

The batch builders are kernels over the same per-frame functions as the CPU
agent's: window ref roots gathered from the library rows; a start's
observation history rebuilt by `sm_observation_lane` from the logged states
and clean actions (noise from a counter hash); its goals and reference rows
for frames min(frame + t, clip end), t = 0..H, by `sm_goal_window_lane` and
`sm_reference_row_lane`.
"""

from std.math import ceildiv
from std.random import random_ui64
from std.time import perf_counter_ns

from max.gpu import global_idx
from max.gpu.host import DeviceBuffer, DeviceContext

from noeira.core.logger import Logger, NoOpLogger
from noeira.deep_agents.supermimic.agent import SMSchedule, sm_cosine_lr, sm_schedule
from noeira.deep_agents.supermimic.metrics import SMIterInfo, SMTrainLog, sm_tick
from noeira.io.artifact_sink import ArtifactSink
from noeira.deep_agents.supermimic.config import SMConfigT
from noeira.deep_agents.supermimic.eval import SMInRunBench
from noeira.deep_agents.supermimic.env_gpu import SMReplayGpuOf, SMTrackingEnvGpu
from noeira.deep_agents.supermimic.heads import SMHeadLayout, sm_observation_lane
from noeira.deep_agents.supermimic.imagination import SMImagBatch
from noeira.deep_agents.supermimic.kinematics import FP
from noeira.deep_agents.supermimic.lanes import sm_hash_u01, talloc, tp
from noeira.deep_agents.supermimic.motion_library import (
    SM_REF_SCRATCH, SMMotionLibrary, sm_goal_window_lane, sm_reference_row_lane,
)
from noeira.deep_agents.supermimic.policy import sm_obs_noise_half
from noeira.deep_agents.supermimic.robot import SMDims, TrackingBatchedSim, TrackingRobot
from noeira.deep_agents.supermimic.trainer import SMUpdateMetrics, SuperMimicTrainer
from noeira.deep_agents.supermimic.world_model import SMWindowBatch, SMWMLoss
from noeira.nn.constants import DT
from noeira.nn.core.ptr import mptr
from noeira.nn.core.tensor import TensorImpl


comptime FPi = Pointer[Int32, MutAnyOrigin]
comptime _TPB: Int = 64


def _k_ref_root[S: Int](n: Int64, rows: FP[DT], frame: FPi, dst: FP[DT]):
    """dst[i] = the library's root pos + quat at frame[i] (7)."""
    var i = Int(global_idx.x)
    if i >= Int(n):
        return
    var f = Int(frame[unsafe_offset=i])
    comptime for c in range(7):
        dst[unsafe_offset=i * 7 + c] = rows[unsafe_offset=f * S + c]


def _k_start_inputs[R: TrackingRobot](
    nb: Int64, h: Int64, seed: UInt32, noise: Int64, rows: FP[DT], tree: FP[DT], table: FP[DT], last: FPi,
    frame: FPi, hist_states: FP[DT], hist_actions: FP[DT], xi: FP[DT], scratch: FP[DT],
    hist0: FP[DT], goals: FP[DT], refs: FP[DT],
):
    """Start b: its observation history and the goals / reference rows of
    frames min(frame + t, clip end), t = 0..h."""
    comptime D = SMDims[R]
    comptime n = R.NDOF
    comptime L = SMHeadLayout[R]
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
    var f0 = Int(frame[unsafe_offset=b])
    var lf = Int(last[unsafe_offset=f0])
    var sc = scratch.unsafe_offset(b * SM_REF_SCRATCH[R])
    for t in range(Int(h) + 1):
        var f = min(f0 + t, lf)
        sm_goal_window_lane[R](rows, f, lf, table.unsafe_offset(L.Q0), goals.unsafe_offset((t * N + b) * D.GOAL))
        sm_reference_row_lane[R](rows, f, tree, table, sc, refs.unsafe_offset((t * N + b) * D.REF))


struct SuperMimicAgentGpu[C: SMConfigT, BS: TrackingBatchedSim](Movable):
    var ctx: DeviceContext
    var env: SMTrackingEnvGpu[Self.C.ROBOT, Self.BS]
    var rb: SMReplayGpuOf[Self.C.ROBOT]
    var trainer: SuperMimicTrainer[Self.C]
    var sched: SMSchedule
    var iteration: Int
    var updates: Int
    var utd_acc: Float64
    var ep_sum: List[Float64]
    var ep_n: List[Int]
    var finished: List[Float64]
    var finished_len: List[Int]     # their lengths (steps)
    var t_sim: Float64
    var t_wm: Float64
    var t_pi: Float64
    var lr: Float64
    var last_avg_reward: Float64      # the last log window's avg_reward (the run's outcome)
    # device batches, allocated once
    var wb: SMWindowBatch[Self.C.ROBOT]
    var ib: SMImagBatch[Self.C.ROBOT]
    var goals: TensorImpl[DT]
    var w_idx: DeviceBuffer[DType.int32]
    var w_frame: DeviceBuffer[DType.int32]
    var w_aclean: DeviceBuffer[DT]
    var w_reward: DeviceBuffer[DT]
    var s_idx: DeviceBuffer[DType.int32]
    var s_frame: DeviceBuffer[DType.int32]
    var s_hs: DeviceBuffer[DT]
    var s_ha: DeviceBuffer[DT]
    var s_scratch: DeviceBuffer[DT]

    def __init__(
        out self, ctx: DeviceContext, var lib: SMMotionLibrary[Self.C.ROBOT], var sim: Self.BS, var sched: SMSchedule
    ) raises:
        comptime R = Self.C.ROBOT
        comptime D = SMDims[R]
        comptime assert Self.C.TARGET == "gpu", "SuperMimicAgentGpu needs a GPU config"
        comptime assert Self.C.LANES == Self.BS.N, "the config's lanes must be the simulator's"
        comptime K = Self.C.DEF_WM_K
        comptime B = Self.C.WM_BATCH
        comptime BP = Self.C.PI_BATCH
        comptime H = Self.C.HORIZON
        self.ctx = ctx
        var noise = sched.noise
        self.env = SMTrackingEnvGpu[R, Self.BS](
            ctx, lib^, sim^, dr=noise, obs_noise=noise, reset_noise=noise,
            action_noise=0.1 if noise else 0.0, rng_seed=sched.seed,
        )
        self.rb = SMReplayGpuOf[R](ctx, Self.C.LANES, sched.replay_capacity)
        self.trainer = SuperMimicTrainer[Self.C].for_robot(R.SIM.create(), Optional(ctx))
        self.sched = sched^
        self.iteration = 0
        self.updates = 0
        self.utd_acc = 0.0
        self.ep_sum = List[Float64](length=Self.C.LANES, fill=0.0)
        self.ep_n = List[Int](length=Self.C.LANES, fill=0)
        self.finished = List[Float64]()
        self.finished_len = List[Int]()
        self.t_sim = 0.0
        self.t_wm = 0.0
        self.t_pi = 0.0
        self.lr = self.sched.lr
        self.last_avg_reward = 0.0
        self.wb = SMWindowBatch[R](K, B)
        self.wb.states.ensure_gpu(ctx, (K + 1) * B * D.S)
        self.wb.a_exec.ensure_gpu(ctx, K * B * R.NDOF)
        self.wb.xi.ensure_gpu(ctx, B * D.X)
        self.wb.done.ensure_gpu(ctx, K * B)
        self.wb.ref_root.ensure_gpu(ctx, K * B * 7)
        self.ib = SMImagBatch[R](BP, H)
        self.ib.state.ensure_gpu(ctx, BP * D.S)
        self.ib.prev_action.ensure_gpu(ctx, BP * R.NDOF)
        self.ib.xi.ensure_gpu(ctx, BP * D.X)
        self.ib.hist0.ensure_gpu(ctx, D.HIST * BP * D.OBS)
        self.ib.goals.ensure_gpu(ctx, (H + 1) * BP * D.GOAL)
        self.ib.refs.ensure_gpu(ctx, (H + 1) * BP * D.REF)
        self.goals = talloc["gpu"](Self.C.LANES * D.GOAL, Optional(ctx))
        self.w_idx = ctx.enqueue_create_buffer[DType.int32](B)
        self.w_frame = ctx.enqueue_create_buffer[DType.int32](K * B)
        self.w_aclean = ctx.enqueue_create_buffer[DT](K * B * R.NDOF)
        self.w_reward = ctx.enqueue_create_buffer[DT](K * B)
        self.s_idx = ctx.enqueue_create_buffer[DType.int32](BP)
        self.s_frame = ctx.enqueue_create_buffer[DType.int32](BP)
        self.s_hs = ctx.enqueue_create_buffer[DT](D.HIST * BP * D.S)
        self.s_ha = ctx.enqueue_create_buffer[DT](D.HIST * BP * R.NDOF)
        self.s_scratch = ctx.enqueue_create_buffer[DT](BP * SM_REF_SCRATCH[R])
        ctx.synchronize()

    def __init__(out self, *, deinit move: Self):
        self.ctx = move.ctx
        self.env = move.env^
        self.rb = move.rb^
        self.trainer = move.trainer^
        self.sched = move.sched^
        self.iteration = move.iteration
        self.updates = move.updates
        self.utd_acc = move.utd_acc
        self.ep_sum = move.ep_sum^
        self.ep_n = move.ep_n^
        self.finished = move.finished^
        self.finished_len = move.finished_len^
        self.t_sim = move.t_sim
        self.t_wm = move.t_wm
        self.t_pi = move.t_pi
        self.lr = move.lr
        self.last_avg_reward = move.last_avg_reward
        self.wb = move.wb^
        self.ib = move.ib^
        self.goals = move.goals^
        self.w_idx = move.w_idx^
        self.w_frame = move.w_frame^
        self.w_aclean = move.w_aclean^
        self.w_reward = move.w_reward^
        self.s_idx = move.s_idx^
        self.s_frame = move.s_frame^
        self.s_hs = move.s_hs^
        self.s_ha = move.s_ha^
        self.s_scratch = move.s_scratch^

    # ── data plumbing: device replay → the trainer's device batches ──
    def _window_batch(mut self) raises:
        comptime R = Self.C.ROBOT
        comptime K = Self.C.DEF_WM_K
        comptime B = Self.C.WM_BATCH
        var ctx = self.ctx
        self.rb.sample_windows(ctx, K, B, random_ui64(0, 0xFFFFFFFFFFFF), mptr(self.w_idx.unsafe_ptr()))
        if self.rb.eligible_total(ctx) == 0:
            raise Error("SuperMimicAgentGpu: no sampleable windows yet (raise the warm-up)")
        self.rb.windows_at(
            ctx, mptr(self.w_idx.unsafe_ptr()), B, K, tp["gpu"](self.wb.states), tp["gpu"](self.wb.a_exec),
            mptr(self.w_aclean.unsafe_ptr()), tp["gpu"](self.wb.xi), mptr(self.w_reward.unsafe_ptr()),
            tp["gpu"](self.wb.done), mptr(self.w_frame.unsafe_ptr()),
        )
        ctx.enqueue_function[_k_ref_root[SMDims[R].S]](
            Int64(K * B), mptr(self.env.lib.rows_dev.value().unsafe_ptr()), mptr(self.w_frame.unsafe_ptr()), tp["gpu"](self.wb.ref_root),
            grid_dim=ceildiv(K * B, _TPB), block_dim=_TPB,
        )

    def _start_batch(mut self) raises:
        comptime R = Self.C.ROBOT
        comptime D = SMDims[R]
        comptime BP = Self.C.PI_BATCH
        var ctx = self.ctx
        self.rb.sample_starts(ctx, D.HIST, BP, random_ui64(0, 0xFFFFFFFFFFFF), mptr(self.s_idx.unsafe_ptr()))
        if self.rb.eligible_total(ctx) == 0:
            raise Error("SuperMimicAgentGpu: no sampleable starts yet (raise the warm-up)")
        self.rb.starts_at(
            ctx, mptr(self.s_idx.unsafe_ptr()), BP, D.HIST, tp["gpu"](self.ib.state), tp["gpu"](self.ib.xi),
            mptr(self.s_frame.unsafe_ptr()), tp["gpu"](self.ib.prev_action), mptr(self.s_hs.unsafe_ptr()), mptr(self.s_ha.unsafe_ptr()),
        )
        ctx.enqueue_function[_k_start_inputs[R]](
            Int64(BP), Int64(Self.C.HORIZON), UInt32(random_ui64(1, 0xFFFFFFFF)), Int64(1 if self.sched.noise else 0),
            mptr(self.env.lib.rows_dev.value().unsafe_ptr()), self.trainer.tables.tree_p["gpu"](), self.trainer.tables.head_p["gpu"](),
            mptr(self.env.lib.last_dev.value().unsafe_ptr()), mptr(self.s_frame.unsafe_ptr()),
            mptr(self.s_hs.unsafe_ptr()), mptr(self.s_ha.unsafe_ptr()), tp["gpu"](self.ib.xi), mptr(self.s_scratch.unsafe_ptr()),
            tp["gpu"](self.ib.hist0), tp["gpu"](self.ib.goals), tp["gpu"](self.ib.refs),
            grid_dim=ceildiv(BP, _TPB), block_dim=_TPB,
        )

    # ── the loop ──
    def iterate(mut self) raises -> SMIterInfo:
        """One env step, and the updates it is owed (as `SuperMimicAgent`)."""
        var ctx = self.ctx
        var t0 = perf_counter_ns()
        self.env.policy_inputs(ctx, self.goals)
        var a = self.trainer.act[Self.C.LANES](self.env.hist, self.goals, resident=True)
        var si = self.env.step(ctx, a, self.rb)
        for i in range(Self.C.LANES):
            self.ep_sum[i] += self.env.last_reward[i]
            self.ep_n[i] += 1
        # a lane whose episode ended this step logged its last reward; the
        # env's logs were reset, so read the ending from them
        for i in range(Self.C.LANES):
            if len(self.env.logs[i].rewards) == 0:
                self.finished.append(self.ep_sum[i] / (0.1 * Float64(max(self.ep_n[i], 1))))
                self.finished_len.append(self.ep_n[i])
                self.ep_sum[i] = 0.0
                self.ep_n[i] = 0
        self.t_sim += Float64(perf_counter_ns() - t0) * 1e-9
        self.iteration += 1
        var info = SMIterInfo(False, SMWMLoss(0, 0, 0, 0, 0), SMUpdateMetrics(0, 0, 0, 0, 0, 0), si.reward_mean, si.n_terminated)
        if self.iteration >= self.sched.warmup:
            self.utd_acc += self.sched.utd
            while self.utd_acc >= 1.0:
                self.utd_acc -= 1.0
                self.lr = sm_cosine_lr(self.iteration, self.sched.total_iterations, self.sched.lr)
                self.trainer.set_lr(self.lr)
                var t1 = perf_counter_ns()
                var acc = SMWMLoss(0, 0, 0, 0, 0)
                for _ in range(Self.C.DEF_WM_BATCHES):
                    self._window_batch()
                    acc.add_scaled(self.trainer.wm_update[Self.C.WM_BATCH](self.wb, resident=True), 1.0 / Float64(Self.C.DEF_WM_BATCHES))
                var t2 = perf_counter_ns()
                self._start_batch()
                var vals = List[Float64]()
                var tgts = List[Float64]()
                var none = List[Int]()
                info.pi = self.trainer.policy_update[Self.C.PI_BATCH](self.ib, none, vals, tgts, self.sched.noise, resident=True)
                info.wm = acc^
                self.t_wm += Float64(t2 - t1) * 1e-9
                self.t_pi += Float64(perf_counter_ns() - t2) * 1e-9
                self.updates += 1
                info.updated = True
        return info^

    def train[L: Logger = NoOpLogger, BB: TrackingBatchedSim = Self.BS](
        mut self,
        iters: Int,
        log_every: Int = 50,
        *,
        logger: Optional[Pointer[L, MutAnyOrigin]] = None,
        verbose: Bool = True,
        checkpoint_every: Int = 0,
        checkpoint_path: String = "",
        artifacts: Optional[ArtifactSink] = None,
        run_dir: String = "",
        bench: Optional[Pointer[SMInRunBench[Self.C, BB], MutAnyOrigin]] = None,
    ) raises:
        """As `SuperMimicAgent.train`; with `bench`, the fixed-clip benchmark
        every `bench.every` iterations and at the end (`SMInRunBench`)."""
        var lg = SMTrainLog(Self.C.HORIZON, Self.C.LANES)
        for it in range(1, iters + 1):
            lg.add(self.iterate())
            if sm_tick[Self.C, L](
                self.trainer, lg, self.env.sampler, it, iters, self.updates, self.finished, self.finished_len, self.lr, self.t_sim, self.t_wm, self.t_pi,
                log_every, logger, verbose, checkpoint_every, checkpoint_path, artifacts, run_dir,
            ):
                self.finished = List[Float64]()
                self.finished_len = List[Int]()
            if bench and bench.value()[].due(it, iters):
                var bp = bench.value()
                var tb = perf_counter_ns()
                var sm = bp[].run(self.ctx, self.trainer)
                var fail = Float64(sm.failed) / Float64(max(sm.clips, 1))
                if verbose:
                    print(
                        "  bench @", it, ":", sm.clips, "clips | norm return", sm.mean_norm_return, "| failed", sm.failed,
                        "|", Float64(perf_counter_ns() - tb) * 1e-9, "s",
                    )
                if logger:
                    var lp = logger.value()
                    lp[].log_scalar("bench_norm_return", sm.mean_norm_return, it)
                    lp[].log_scalar("bench_failed", fail, it)
                    lp[].flush()
        self.last_avg_reward = lg.last_avg


def sm_agent_gpu[C: SMConfigT, BS: TrackingBatchedSim](
    ctx: DeviceContext,
    var lib: SMMotionLibrary[C.ROBOT],
    var sim: BS,
    total_iterations: Int,
    noise: Bool = True,
    seed: Int = 0,
    utd: Float64 = C.DEF_UTD,
    warmup: Int = C.DEF_WARMUP,
    lr: Float64 = C.DEF_LR,
    replay_capacity: Int = C.DEF_REPLAY_CAP,
) raises -> SuperMimicAgentGpu[C, BS]:
    """`SuperMimicAgentGpu[C, BS]` with the config's tuned defaults."""
    return SuperMimicAgentGpu[C, BS](ctx, lib^, sim^, sm_schedule[C](total_iterations, noise, seed, utd, warmup, lr, replay_capacity))
