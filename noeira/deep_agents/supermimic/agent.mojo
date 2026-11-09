"""SuperMimicAgent — user-facing facade: env + replay + trainer + schedule.

The reproduction's `training/loop.py` (paper Algorithm A.1). The trainer owns
the algorithm (`trainer.mojo`); the agent owns the data side and the loop:

    var agent = sm_agent[SMSmokeG1["cpu"]](library^, total_iterations=400)
    agent.train(400, log_every=50, logger=logger_ptr, checkpoint_every=100,
                checkpoint_path=run.checkpoint_path("last"), artifacts=artifacts, run_dir=run.dir)

Every iteration: the policy acts on every lane (`trainer.act`), one env step
(rows into the episodic replay ring); after a warm-up and at the UTD ratio,
the world model's 10 batches of k = 8 windows and one policy update from
replay starts (their 10-step observation history rebuilt from the logged
states and clean actions). One cosine lr for all three networks.

The batches handed to the trainer are `nn` tensors; on the GPU path the
replay's device sibling fills them on the device instead.
"""

from std.math import cos, pi
from std.random import random_float64
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext

from noeira.core.logger import Logger, NoOpLogger
from noeira.io.artifact_sink import ArtifactSink

from noeira.deep_agents.data import EpisodicReplay
from noeira.deep_agents.supermimic.config import SMConfigT
from noeira.deep_agents.supermimic.env import SMReplayOf, SMTrackingEnv
from noeira.deep_agents.supermimic.heads import sm_observation_lane
from noeira.deep_agents.supermimic.imagination import SMImagBatch
from noeira.deep_agents.supermimic.metrics import SMIterInfo, SMTrainLog, sm_tick
from noeira.deep_agents.supermimic.motion_library import SMMotionLibrary
from noeira.deep_agents.supermimic.policy import sm_obs_noise_half
from noeira.deep_agents.supermimic.robot import SMDims
from noeira.deep_agents.supermimic.trainer import SMUpdateMetrics, SuperMimicTrainer
from noeira.deep_agents.supermimic.world_model import SMWindowBatch, SMWMLoss
from noeira.nn.constants import DT
from noeira.nn.core.tensor import TensorImpl


def sm_cosine_lr(step: Int, total: Int, base: Float64, final_ratio: Float64 = 0.03, decay_fraction: Float64 = 250000.0 / 320000.0) -> Float64:
    """`schedule.cosine_lr`."""
    var decay = max(1, Int(Float64(total) * decay_fraction))
    if step >= decay:
        return base * final_ratio
    var c = 0.5 * (1.0 + cos(pi * Float64(step) / Float64(decay)))
    return base * (final_ratio + (1.0 - final_ratio) * c)


def sm_restart_lr(step: Int, decay: Int, base: Float64, ratio: Float64, final_ratio: Float64 = 0.03) -> Float64:
    """realmimic's warm restart (`training/schedule.py` `scheduled_lr`): a cosine
    from ratio x base down to final_ratio x base over `decay` steps, then flat."""
    return sm_cosine_lr(step, decay, base * ratio, final_ratio / ratio, 1.0)


@fieldwise_init
struct SMSchedule(Copyable, Movable):
    """`ScheduleCfg` + the env's noise flags. `sm_schedule[C](...)` fills the
    tuned values from the config's `DEF_*`."""
    var total_iterations: Int
    var utd: Float64              # updates per env step
    var warmup: Int               # env steps before the first update
    var lr: Float64
    var replay_capacity: Int      # rows per lane
    var noise: Bool               # obs / action / reset noise and domain randomisation
    var seed: Int


def sm_schedule[C: SMConfigT](
    total_iterations: Int, noise: Bool = True, seed: Int = 0,
    utd: Float64 = C.DEF_UTD, warmup: Int = C.DEF_WARMUP, lr: Float64 = C.DEF_LR,
    replay_capacity: Int = C.DEF_REPLAY_CAP,
) -> SMSchedule:
    """The schedule from the config's tuned defaults, any of them overridable."""
    return SMSchedule(total_iterations, utd, warmup, lr, replay_capacity, noise, seed)


struct SuperMimicAgent[C: SMConfigT](Movable):
    var env: SMTrackingEnv[Self.C.ROBOT]
    var rb: SMReplayOf[Self.C.ROBOT]
    var trainer: SuperMimicTrainer[Self.C]
    var sched: SMSchedule
    var iteration: Int
    var updates: Int
    var utd_acc: Float64
    var ep_sum: List[Float64]
    var ep_n: List[Int]
    var finished: List[Float64]     # normalised returns Σr / (0.1·T) of finished episodes
    var finished_len: List[Int]     # their lengths (steps)
    var t_sim: Float64
    var t_wm: Float64
    var t_pi: Float64
    var lr: Float64
    var last_avg_reward: Float64      # the last log window's avg_reward (the run's outcome)
    var resumed_at: Int               # the iteration this process started from (the warm-up counts from it)
    var restart_at: Int               # a warm restart of the lr from this iteration (-1: none)
    var restart_ratio: Float64
    var restart_decay: Int

    def __init__(
        out self, var lib: SMMotionLibrary[Self.C.ROBOT], var sched: SMSchedule, ctx: Optional[DeviceContext] = None
    ) raises:
        """`ctx` is required when the config's target is "gpu" (the networks
        and updates run there; the env and replay stay on the host)."""
        var noise = sched.noise
        self.env = SMTrackingEnv[Self.C.ROBOT](
            Self.C.LANES, lib^, dr=noise, obs_noise=noise, reset_noise=noise,
            action_noise=0.1 if noise else 0.0, rng_seed=sched.seed,
        )
        self.rb = SMReplayOf[Self.C.ROBOT](Self.C.LANES, sched.replay_capacity)
        self.trainer = SuperMimicTrainer[Self.C].for_robot(self.env.lanes[0].sim, ctx)
        self.sched = sched^
        self.iteration = 0
        self.resumed_at = 0
        self.restart_at = -1
        self.restart_ratio = 0.3
        self.restart_decay = 0
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

    def __init__(out self, *, deinit move: Self):
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
        self.resumed_at = move.resumed_at
        self.restart_at = move.restart_at
        self.restart_ratio = move.restart_ratio
        self.restart_decay = move.restart_decay

    # ── data plumbing: replay → the trainer's batches ──
    def window_batch(mut self) raises -> SMWindowBatch[Self.C.ROBOT]:
        comptime S = SMDims[Self.C.ROBOT].S
        comptime K = Self.C.DEF_WM_K
        var idx = SMReplayOf[Self.C.ROBOT].sample_indices(self.rb.window_mask(K), Self.C.WM_BATCH)
        var w = self.rb.windows_at(idx, K)
        var wb = SMWindowBatch[Self.C.ROBOT](K, Self.C.WM_BATCH)
        for i in range(len(w.states)):
            wb.states.data[i] = w.states[i]
        for i in range(len(w.a_exec)):
            wb.a_exec.data[i] = w.a_exec[i]
        for i in range(len(w.xi)):
            wb.xi.data[i] = w.xi[i]
        for i in range(len(w.done)):
            wb.done.data[i] = w.done[i]
            var f = w.frame[i]
            for c in range(7):
                wb.ref_root.data[i * 7 + c] = self.env.lib.rows[f * S + c]
        return wb^

    def start_batch(mut self) raises -> SMImagBatch[Self.C.ROBOT]:
        comptime R = Self.C.ROBOT
        comptime D = SMDims[R]
        comptime BP = Self.C.PI_BATCH
        var idx = SMReplayOf[R].sample_indices(self.rb.start_mask(D.HIST), BP)
        var st = self.rb.starts_at(idx, D.HIST)
        var ib = SMImagBatch[R](BP, Self.C.HORIZON)
        for i in range(len(st.state)):
            ib.state.data[i] = st.state[i]
        for i in range(len(st.prev_action)):
            ib.prev_action.data[i] = st.prev_action[i]
        for i in range(len(st.xi)):
            ib.xi.data[i] = st.xi[i]
        for hh in range(D.HIST):
            for b in range(BP):
                var o = (hh * BP + b) * D.OBS
                sm_observation_lane[DT, R](
                    self.trainer.tables.head_ptr(),
                    st.hist_states.unsafe_ptr().unsafe_offset((hh * BP + b) * D.S).as_unsafe_any_origin(),
                    st.hist_actions.unsafe_ptr().unsafe_offset((hh * BP + b) * R.NDOF).as_unsafe_any_origin(),
                    st.xi.unsafe_ptr().unsafe_offset(b * D.X).as_unsafe_any_origin(),
                    ib.hist0.data.unsafe_ptr().unsafe_offset(o).as_unsafe_any_origin(),
                )
                if self.sched.noise:
                    for k in range(D.OBS):
                        var hw = sm_obs_noise_half[R.NDOF](k)
                        if hw > 0:
                            ib.hist0.data[o + k] += Scalar[DT](random_float64(-hw, hw))
        for b in range(BP):
            var last = self.env.lib.last_frame_of(st.frame[b])
            for t in range(Self.C.HORIZON + 1):
                var f = min(st.frame[b] + t, last)
                self.env.lib.goal_window(f, ib.goals.data.unsafe_ptr().unsafe_offset((t * BP + b) * D.GOAL).as_unsafe_any_origin())
                self.env.lib.reference_row(
                    f, self.trainer.tables, ib.refs.data.unsafe_ptr().unsafe_offset((t * BP + b) * D.REF).as_unsafe_any_origin(),
                )
        return ib^

    # ── the loop ──
    def iterate(mut self) raises -> SMIterInfo:
        """One env step, and the updates it is owed."""
        comptime R = Self.C.ROBOT
        comptime D = SMDims[R]
        var t0 = perf_counter_ns()
        var hist = TensorImpl[DT].alloc(Self.C.LANES * D.HIST * D.OBS)
        var goals = TensorImpl[DT].alloc(Self.C.LANES * D.GOAL)
        self.env.policy_inputs(hist, goals)
        var a = self.trainer.act[Self.C.LANES](hist, goals)
        var si = self.env.step(a.data, self.rb)
        for i in range(Self.C.LANES):
            self.ep_sum[i] += self.env.last_reward[i]
            self.ep_n[i] += 1
            if self.env.lanes[i].ep_len == 0:      # reset this step
                self.finished.append(self.ep_sum[i] / (0.1 * Float64(self.ep_n[i])))
                self.finished_len.append(self.ep_n[i])
                self.ep_sum[i] = 0.0
                self.ep_n[i] = 0
        self.t_sim += Float64(perf_counter_ns() - t0) * 1e-9
        self.iteration += 1
        var info = SMIterInfo(False, SMWMLoss(0, 0, 0, 0, 0), SMUpdateMetrics(0, 0, 0, 0, 0, 0), si.reward_mean, si.n_terminated)
        if self.iteration - self.resumed_at >= self.sched.warmup:     # a resumed replay starts empty
            self.utd_acc += self.sched.utd
            while self.utd_acc >= 1.0:
                self.utd_acc -= 1.0
                self.lr = self._lr()
                self.trainer.set_lr(self.lr)
                var t1 = perf_counter_ns()
                var acc = SMWMLoss(0, 0, 0, 0, 0)
                for _ in range(Self.C.DEF_WM_BATCHES):
                    var wb = self.window_batch()
                    acc.add_scaled(self.trainer.wm_update[Self.C.WM_BATCH](wb), 1.0 / Float64(Self.C.DEF_WM_BATCHES))
                var t2 = perf_counter_ns()
                var ib = self.start_batch()
                var vals = List[Float64]()
                var tgts = List[Float64]()
                var none = List[Int]()
                info.pi = self.trainer.policy_update[Self.C.PI_BATCH](ib, none, vals, tgts, self.sched.noise)
                info.wm = acc^
                self.t_wm += Float64(t2 - t1) * 1e-9
                self.t_pi += Float64(perf_counter_ns() - t2) * 1e-9
                self.updates += 1
                info.updated = True
        return info^

    def resume(mut self, iteration: Int, updates: Int, lr_restart_ratio: Float64 = 0.0, lr_restart_decay: Int = 0):
        """Go on from a checkpoint's counters (`SuperMimicTrainer.load_state`):
        the iterations count on to the run's total, the warm-up runs again (the
        replay is empty) and Adam re-warms (its moments are not saved). With a
        restart ratio and decay, realmimic's warm restart for resumed runs: the
        lr restarts at ratio x lr and decays by a second cosine to the first's
        final ratio over `lr_restart_decay` iterations."""
        self.iteration = iteration
        self.updates = updates
        self.resumed_at = iteration
        if lr_restart_ratio > 0.0 and lr_restart_decay > 0:
            self.restart_at = iteration
            self.restart_ratio = lr_restart_ratio
            self.restart_decay = lr_restart_decay

    def _lr(self) -> Float64:
        if self.restart_at >= 0 and self.iteration >= self.restart_at:
            return sm_restart_lr(self.iteration - self.restart_at, self.restart_decay, self.sched.lr, self.restart_ratio)
        return sm_cosine_lr(self.iteration, self.sched.total_iterations, self.sched.lr)

    def train[L: Logger = NoOpLogger](
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
    ) raises:
        """`iters` iterations. Every `log_every`: one `SMMetrics` bundle to
        `logger` (the run's CSV + monitor, `run_session.run_logger`) and a
        terminal line; every `checkpoint_every` and at the end: the
        trainer's one-file checkpoint at `checkpoint_path`, offered to
        `artifacts` (SAC's `train` arguments)."""
        var lg = SMTrainLog(Self.C.HORIZON, Self.C.LANES)
        for it in range(self.iteration + 1, iters + 1):
            lg.add(self.iterate())
            if sm_tick[Self.C, L](
                self.trainer, lg, self.env.sampler, it, iters, self.updates, self.finished, self.finished_len, self.lr, self.t_sim, self.t_wm, self.t_pi,
                log_every, logger, verbose, checkpoint_every, checkpoint_path, artifacts, run_dir,
            ):
                self.finished = List[Float64]()
                self.finished_len = List[Int]()
        self.last_avg_reward = lg.last_avg


def sm_agent[C: SMConfigT](
    var lib: SMMotionLibrary[C.ROBOT],
    total_iterations: Int,
    noise: Bool = True,
    seed: Int = 0,
    utd: Float64 = C.DEF_UTD,
    warmup: Int = C.DEF_WARMUP,
    lr: Float64 = C.DEF_LR,
    replay_capacity: Int = C.DEF_REPLAY_CAP,
    ctx: Optional[DeviceContext] = None,
) raises -> SuperMimicAgent[C]:
    """`SuperMimicAgent[C]` with the config's tuned defaults (SAC's
    `agent_from_config`); `ctx` for a "gpu" config."""
    return SuperMimicAgent[C](lib^, sm_schedule[C](total_iterations, noise, seed, utd, warmup, lr, replay_capacity), ctx)
