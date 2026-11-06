"""SMPromptAgent — a tracker's lanes and prompt lanes, on the CPU env.

Appendix G's loop over a tracker that no longer trains: lanes below `split` keep
tracking clips (they feed the world model what it was fit on), the others run
the prompt interface under fall-only termination. Every lane draws a command at
each reset and every `DEF_HOLD` steps: a reference command read from its clip,
or a held one from the bank. Updates: the world model on every lane's rows, the
encoder and its critic from starts logged by the prompt lanes, whose commands are
drawn afresh (nothing about a command is logged: a held z is relabelled at will,
FB's habit, which also trains a change of command). The first `align_warm`
updates raise the alignment's weight, so the prompt lanes reach the tracker's
latents before they feed the world model much.

`prompt_agent_gpu.mojo` is this loop on the device env. This one runs where the
batched G1 does not (the Mac), at smoke scale: it is the loop's logic under test.
"""

from std.math import cos, pi
from std.random import random_float64, random_ui64
from std.time import perf_counter_ns

from noeira.deep_agents.supermimic.agent import SMSchedule, sm_cosine_lr
from noeira.deep_agents.supermimic.env import SMReplayOf, SMTrackingEnv
from noeira.deep_agents.supermimic.heads import sm_observation_lane
from noeira.deep_agents.supermimic.motion_library import SMMotionLibrary
from noeira.deep_agents.supermimic.policy import sm_obs_noise_half
from noeira.deep_agents.supermimic.prompt import SMSpec, sm_spec_velocity
from noeira.deep_agents.supermimic.prompt_bank import SMPromptData
from noeira.deep_agents.supermimic.prompt_imagination import SMPromptBatch
from noeira.deep_agents.supermimic.prompt_policy import SMPromptConfigT
from noeira.deep_agents.supermimic.prompt_trainer import SMPromptTrainer, SMPromptUpdate
from noeira.deep_agents.supermimic.robot import SMDims
from noeira.deep_agents.supermimic.world_model import SMWindowBatch, SMWMLoss
from noeira.nn.constants import DT
from noeira.nn.core.tensor import TensorImpl


comptime SM_P_WM_LR: Float64 = 2e-5       # the world model is trained already: it keeps fitting gently
comptime SM_P_ALIGN_WARM_GAIN: Float64 = 100.0   # the first updates only learn to read a reference command


@fieldwise_init
struct SMPromptLog(Copyable, Movable):
    """One iteration's numbers; `sim_r` and `falls` over the prompt lanes."""
    var updated: Bool
    var sim_r: Float64
    var falls: Int
    var wm: SMWMLoss
    var pi: SMPromptUpdate


def sm_print_prompt(it: Int, up: Int, sim_r: Float64, fall: Float64, wm: SMWMLoss, pi: SMPromptUpdate, h: Int, steps_s: Float64):
    """`sim_r` and `imag_r` are the same quantity, the reward per step, simulated
    and imagined: the honesty gate reads them side by side."""
    print(
        "it", it, "| up", up, "| sim_r", sim_r, "imag_r", pi.imag_return_undisc / Float64(h), "| fall/step", fall,
        "| align", pi.align_mse, "|g|", pi.policy_grad_norm, "| critic", pi.critic_loss, "V", pi.critic_value_mean,
        "R", pi.critic_target_mean, "| wm", wm.loss, "nse1", wm.nse_1step, "nse8", wm.nse_kstep, "|", steps_s, "steps/s",
    )


struct SMPromptAgent[P: SMPromptConfigT](Movable):
    var env: SMTrackingEnv[Self.P.BASE.ROBOT]
    var rb: SMReplayOf[Self.P.BASE.ROBOT]
    var trainer: SMPromptTrainer[Self.P]
    var data: SMPromptData[Self.P.CMD]
    var sched: SMSchedule
    var split: Int
    var align_warm: Int           # updates at the alignment's raised weight
    var iteration: Int
    var updates: Int
    var utd_acc: Float64
    var src: List[Int]            # per lane: −1 a reference command, else a bank row
    var hold: List[Int]           # steps left on the lane's command
    var specs: List[Scalar[DT]]   # [L·SPEC]

    def __init__(
        out self, var lib: SMMotionLibrary[Self.P.BASE.ROBOT], var trainer: SMPromptTrainer[Self.P],
        var data: SMPromptData[Self.P.CMD], var sched: SMSchedule, split: Int,
    ) raises:
        comptime L = Self.P.BASE.LANES
        var noise = sched.noise
        self.env = SMTrackingEnv[Self.P.BASE.ROBOT](
            L, lib^, dr=noise, obs_noise=noise, reset_noise=noise, action_noise=0.1 if noise else 0.0, rng_seed=sched.seed,
        )
        self.env.prompt_from = split
        self.rb = SMReplayOf[Self.P.BASE.ROBOT](L, sched.replay_capacity)
        self.trainer = trainer^
        self.data = data^
        self.sched = sched^
        self.split = split
        self.align_warm = 0
        self.iteration = 0
        self.updates = 0
        self.utd_acc = 0.0
        self.src = List[Int](length=L, fill=-1)
        self.hold = List[Int](length=L, fill=0)
        self.specs = List[Scalar[DT]](length=L * SMSpec.SIZE, fill=0)

    def __init__(out self, *, deinit move: Self):
        self.env = move.env^
        self.rb = move.rb^
        self.trainer = move.trainer^
        self.data = move.data^
        self.sched = move.sched^
        self.split = move.split
        self.align_warm = move.align_warm
        self.iteration = move.iteration
        self.updates = move.updates
        self.utd_acc = move.utd_acc
        self.src = move.src^
        self.hold = move.hold^
        self.specs = move.specs^

    def _commands(mut self, mut cmds: TensorImpl[DT]):
        """Each prompt lane's command and spec for this step."""
        comptime CMD = Self.P.CMD
        comptime RW = CMD + SMSpec.SIZE
        var sp = self.specs.unsafe_ptr().as_unsafe_any_origin()
        for l in range(self.split, Self.P.BASE.LANES):
            if self.env.lanes[l].ep_len == 0 or self.hold[l] <= 0:
                self.hold[l] = Self.P.DEF_HOLD
                if random_float64(0.0, 1.0) < Self.P.DEF_REF_SHARE:
                    self.src[l] = -1
                else:
                    self.src[l] = Int(random_ui64(0, UInt64(self.data.bank_n - 1)))
            self.hold[l] -= 1
            var f = self.env.lanes[l].frame
            if self.src[l] < 0:
                for k in range(CMD):
                    cmds.data[l * CMD + k] = self.data.cmd_ref.data[f * CMD + k]
                sm_spec_velocity[DT](
                    sp.unsafe_offset(l * SMSpec.SIZE), self.data.vel_ref.data[f * 3], self.data.vel_ref.data[f * 3 + 1], self.data.vel_ref.data[f * 3 + 2],
                )
            else:
                var row = self.src[l] * RW
                for k in range(CMD):
                    cmds.data[l * CMD + k] = self.data.bank.data[row + k]
                for k in range(SMSpec.SIZE):
                    self.specs[l * SMSpec.SIZE + k] = self.data.bank.data[row + CMD + k]

    # ── data plumbing ──
    def window_batch(mut self) raises -> SMWindowBatch[Self.P.BASE.ROBOT]:
        comptime R = Self.P.BASE.ROBOT
        comptime S = SMDims[R].S
        comptime K = Self.P.BASE.DEF_WM_K
        var idx = SMReplayOf[R].sample_indices(self.rb.window_mask(K), Self.P.BASE.WM_BATCH)
        var w = self.rb.windows_at(idx, K)
        var wb = SMWindowBatch[R](K, Self.P.BASE.WM_BATCH)
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

    def start_batch(mut self) raises -> SMPromptBatch[Self.P.BASE.ROBOT, Self.P.CMD]:
        """Starts logged by the prompt lanes. The first half reads reference
        commands from the start's frame, the second holds a bank row."""
        comptime R = Self.P.BASE.ROBOT
        comptime D = SMDims[R]
        comptime BP = Self.P.BASE.PI_BATCH
        comptime H = Self.P.BASE.HORIZON
        comptime CMD = Self.P.CMD
        comptime RW = CMD + SMSpec.SIZE
        var mask = self.rb.start_mask(D.HIST)
        var cap = self.rb.capacity()
        for i in range(self.split * cap):
            mask[i] = False
        var idx = SMReplayOf[R].sample_indices(mask, BP)
        var st = self.rb.starts_at(idx, D.HIST)
        var nref = Int(Float64(BP) * Self.P.DEF_REF_SHARE)
        var ib = SMPromptBatch[R, CMD](BP, H, nref)
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
                    self.trainer.base.tables.head_ptr(),
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
        var sp = ib.specs.data.unsafe_ptr().as_unsafe_any_origin()
        for b in range(BP):
            if b < nref:
                var last = self.env.lib.last_frame_of(st.frame[b])
                for t in range(H + 1):
                    var f = min(st.frame[b] + t, last)
                    for k in range(CMD):
                        ib.cmds.data[(t * BP + b) * CMD + k] = self.data.cmd_ref.data[f * CMD + k]
                    if t < H:
                        sm_spec_velocity[DT](
                            sp.unsafe_offset((t * BP + b) * SMSpec.SIZE), self.data.vel_ref.data[f * 3],
                            self.data.vel_ref.data[f * 3 + 1], self.data.vel_ref.data[f * 3 + 2],
                        )
                        for k in range(D.LATENT):
                            ib.lat.data[(t * BP + b) * D.LATENT + k] = self.data.lat.data[f * D.LATENT + k]
            else:
                var row = Int(random_ui64(0, UInt64(self.data.bank_n - 1))) * RW
                for t in range(H + 1):
                    for k in range(CMD):
                        ib.cmds.data[(t * BP + b) * CMD + k] = self.data.bank.data[row + k]
                    if t < H:
                        for k in range(SMSpec.SIZE):
                            ib.specs.data[(t * BP + b) * SMSpec.SIZE + k] = self.data.bank.data[row + CMD + k]
        return ib^

    # ── the loop ──
    def iterate(mut self) raises -> SMPromptLog:
        """One env step, and the updates it is owed."""
        comptime B = Self.P.BASE
        comptime R = B.ROBOT
        comptime D = SMDims[R]
        var hist = TensorImpl[DT].alloc(B.LANES * D.HIST * D.OBS)
        var goals = TensorImpl[DT].alloc(B.LANES * D.GOAL)
        var cmds = TensorImpl[DT].alloc(B.LANES * Self.P.CMD)
        self.env.policy_inputs(hist, goals)
        self._commands(cmds)
        var a = self.trainer.act[B.LANES](hist, goals, cmds, self.split)
        var si = self.env.step(a.data, self.rb, self.specs.unsafe_ptr().as_unsafe_any_origin())
        var rsum = 0.0
        var falls = 0
        for l in range(self.split, B.LANES):
            rsum += self.env.last_prompt_r[l]
            falls += 1 if self.env.last_terminated[l] else 0
        var info = SMPromptLog(
            False, rsum / Float64(max(B.LANES - self.split, 1)), falls, SMWMLoss(0, 0, 0, 0, 0), SMPromptUpdate(0, 0, 0, 0, 0, 0, 0),
        )
        _ = si^
        self.iteration += 1
        if self.iteration >= self.sched.warmup:
            self.utd_acc += self.sched.utd
            while self.utd_acc >= 1.0:
                self.utd_acc -= 1.0
                self.trainer.set_lr(sm_cosine_lr(self.iteration, self.sched.total_iterations, self.sched.lr), SM_P_WM_LR)
                self.trainer.w_align = Self.P.DEF_ALIGN * (SM_P_ALIGN_WARM_GAIN if self.updates < self.align_warm else 1.0)
                var acc = SMWMLoss(0, 0, 0, 0, 0)
                for _ in range(B.DEF_WM_BATCHES):
                    var wb = self.window_batch()
                    acc.add_scaled(self.trainer.wm_update[B.WM_BATCH](wb), 1.0 / Float64(B.DEF_WM_BATCHES))
                var ib = self.start_batch()
                var vals = List[Float64]()
                var tgts = List[Float64]()
                var none = List[Int]()
                info.pi = self.trainer.policy_update[B.PI_BATCH](ib, none, vals, tgts, self.sched.noise)
                info.wm = acc^
                self.updates += 1
                info.updated = True
        return info^

    def train(mut self, iters: Int, log_every: Int = 50, checkpoint_path: String = "", verbose: Bool = True) raises:
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
        if checkpoint_path.byte_length() > 0:
            self.trainer.save_state(checkpoint_path, iters, self.updates)
