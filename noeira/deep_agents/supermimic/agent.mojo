"""SuperMimicAgent — user-facing facade: env + replay + trainer + schedule.

The reproduction's `training/loop.py` (paper Algorithm A.1). The trainer owns
the algorithm (`trainer.mojo`); the agent owns the data side and the loop:

    var agent = SuperMimicAgent[SMSmokeG1](library^, SMSchedule(...))
    agent.train(iters=400, log_every=50)

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

from noeira.deep_agents.data import EpisodicReplay
from noeira.deep_agents.supermimic.config import SMConfigT
from noeira.deep_agents.supermimic.env import SMReplayOf, SMTrackingEnv
from noeira.deep_agents.supermimic.heads import sm_observation_lane
from noeira.deep_agents.supermimic.imagination import SMImagBatch
from noeira.deep_agents.supermimic.motion_library import SMMotionLibrary
from noeira.deep_agents.supermimic.policy import sm_obs_noise_half
from noeira.deep_agents.supermimic.robot import SMDims
from noeira.deep_agents.supermimic.trainer import SMUpdateMetrics, SuperMimicTrainer
from noeira.deep_agents.supermimic.world_model import SMWindowBatch, SMWMLoss
from noeira.nn.constants import DT
from noeira.nn.core.tensor import TensorImpl


comptime SM_WM_K: Int = 8
comptime SM_WM_BATCHES: Int = 10


def sm_cosine_lr(step: Int, total: Int, base: Float64, final_ratio: Float64 = 0.03, decay_fraction: Float64 = 250000.0 / 320000.0) -> Float64:
    """`schedule.cosine_lr`."""
    var decay = max(1, Int(Float64(total) * decay_fraction))
    if step >= decay:
        return base * final_ratio
    var c = 0.5 * (1.0 + cos(pi * Float64(step) / Float64(decay)))
    return base * (final_ratio + (1.0 - final_ratio) * c)


@fieldwise_init
struct SMSchedule(Copyable, Movable):
    """`ScheduleCfg` + the env's noise flags."""
    var total_iterations: Int
    var utd: Float64              # updates per env step (0.4)
    var warmup: Int               # env steps before the first update (40)
    var lr: Float64               # 2e-4
    var replay_capacity: Int      # rows per lane (1000)
    var noise: Bool               # obs / action / reset noise and domain randomisation
    var seed: Int


@fieldwise_init
struct SMIterInfo(Copyable, Movable):
    var updated: Bool
    var wm: SMWMLoss
    var pi: SMUpdateMetrics


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
    var t_sim: Float64
    var t_wm: Float64
    var t_pi: Float64

    def __init__(out self, var lib: SMMotionLibrary[Self.C.ROBOT], var sched: SMSchedule) raises:
        var noise = sched.noise
        self.env = SMTrackingEnv[Self.C.ROBOT](
            Self.C.LANES, lib^, dr=noise, obs_noise=noise, reset_noise=noise,
            action_noise=0.1 if noise else 0.0, rng_seed=sched.seed,
        )
        self.rb = SMReplayOf[Self.C.ROBOT](Self.C.LANES, sched.replay_capacity)
        self.trainer = SuperMimicTrainer[Self.C].for_robot(self.env.lanes[0].sim)
        self.sched = sched^
        self.iteration = 0
        self.updates = 0
        self.utd_acc = 0.0
        self.ep_sum = List[Float64](length=Self.C.LANES, fill=0.0)
        self.ep_n = List[Int](length=Self.C.LANES, fill=0)
        self.finished = List[Float64]()
        self.t_sim = 0.0
        self.t_wm = 0.0
        self.t_pi = 0.0

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
        self.t_sim = move.t_sim
        self.t_wm = move.t_wm
        self.t_pi = move.t_pi

    # ── data plumbing: replay → the trainer's batches ──
    def window_batch(mut self) raises -> SMWindowBatch[Self.C.ROBOT]:
        comptime S = SMDims[Self.C.ROBOT].S
        var idx = SMReplayOf[Self.C.ROBOT].sample_indices(self.rb.window_mask(SM_WM_K), Self.C.WM_BATCH)
        var w = self.rb.windows_at(idx, SM_WM_K)
        var wb = SMWindowBatch[Self.C.ROBOT](SM_WM_K, Self.C.WM_BATCH)
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
        _ = self.env.step(a.data, self.rb)
        for i in range(Self.C.LANES):
            self.ep_sum[i] += self.env.last_reward[i]
            self.ep_n[i] += 1
            if self.env.lanes[i].ep_len == 0:      # reset this step
                self.finished.append(self.ep_sum[i] / (0.1 * Float64(self.ep_n[i])))
                self.ep_sum[i] = 0.0
                self.ep_n[i] = 0
        self.t_sim += Float64(perf_counter_ns() - t0) * 1e-9
        self.iteration += 1
        var info = SMIterInfo(False, SMWMLoss(0, 0, 0, 0, 0), SMUpdateMetrics(0, 0, 0, 0, 0, 0))
        if self.iteration >= self.sched.warmup:
            self.utd_acc += self.sched.utd
            while self.utd_acc >= 1.0:
                self.utd_acc -= 1.0
                self.trainer.set_lr(sm_cosine_lr(self.iteration, self.sched.total_iterations, self.sched.lr))
                var t1 = perf_counter_ns()
                var acc = SMWMLoss(0, 0, 0, 0, 0)
                for _ in range(SM_WM_BATCHES):
                    var wb = self.window_batch()
                    var l = self.trainer.wm_update[Self.C.WM_BATCH](wb)
                    acc.loss += l.loss / Float64(SM_WM_BATCHES)
                    acc.state_loss += l.state_loss / Float64(SM_WM_BATCHES)
                    acc.term_bce += l.term_bce / Float64(SM_WM_BATCHES)
                    acc.nse_1step += l.nse_1step / Float64(SM_WM_BATCHES)
                    acc.nse_kstep += l.nse_kstep / Float64(SM_WM_BATCHES)
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

    def train(mut self, iters: Int, log_every: Int) raises:
        """`iters` iterations, a log line every `log_every`."""
        var n_up = 0
        var wl = 0.0
        var w1 = 0.0
        var w8 = 0.0
        var ij = 0.0
        var iu = 0.0
        var cl = 0.0
        var vm = 0.0
        var tmn = 0.0
        for it in range(iters):
            var r = self.iterate()
            if r.updated:
                n_up += 1
                wl += r.wm.loss
                w1 += r.wm.nse_1step
                w8 += r.wm.nse_kstep
                ij += r.pi.imag_objective
                iu += r.pi.imag_return_undisc
                cl += r.pi.critic_loss
                vm += r.pi.critic_value_mean
                tmn += r.pi.critic_target_mean
            if (it + 1) % log_every == 0:
                var tr = 0.0
                for k in range(len(self.finished)):
                    tr += self.finished[k]
                var nf = len(self.finished)
                var d = Float64(max(n_up, 1))
                print(
                    "it", it + 1, "| updates", self.updates, "| train_ret", (tr / Float64(nf)) if nf > 0 else -1.0, "(n=", nf, ")",
                    "| wm loss", wl / d, "nse1", w1 / d, "nse8", w8 / d,
                    "| imag J", ij / d, "ret", iu / d, "| critic", cl / d, "V", vm / d, "R", tmn / d,
                    "| t sim/wm/pi", self.t_sim, self.t_wm, self.t_pi,
                )
                self.finished = List[Float64]()
                n_up = 0
                wl = 0.0
                w1 = 0.0
                w8 = 0.0
                ij = 0.0
                iu = 0.0
                cl = 0.0
                vm = 0.0
                tmn = 0.0
