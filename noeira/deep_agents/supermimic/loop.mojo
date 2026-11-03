# +--------------------------------------------------------------------------+ #
# | SuperMimic: the training loop (paper Algorithm A.1)
# +--------------------------------------------------------------------------+ #
"""The reproduction's `training/loop.py` `Trainer.run`: collect one vectorised
env step, then, after a warm-up, updates at the update-to-data ratio:

    every iteration   a = policy(history, goal) for every lane; env.step(a)
    UTD (0.4/step)    world model: 10 batches of k = 8 replay windows, Eq. 1,
                      clip 1.0, Adam; then one `PolicyTrainer.update` from
                      replay starts (their 10-step history rebuilt from the
                      logged states and actions)
    lr                cosine from 2e-4 to 3 % over 250k / 320k of training,
                      the same for world model, policy and critic

Logged: the training return Σ r / (0.1 · T) of finished episodes (the
paper's normalised return), the world model's 1- and 8-step errors, the
imagined return and survival, and wall-clock per phase.

CPU for now: every network on `nn`'s CPU path, one simulator per lane.
"""

from std.math import cos, pi
from std.time import perf_counter_ns

from max.gpu.host import DeviceContext

from noeira.deep_agents.supermimic.env import SMG1Env
from noeira.deep_agents.supermimic.heads import SM_REF_DIM, sm_observation_lane
from noeira.deep_agents.supermimic.imagination import SMImagBatch
from noeira.deep_agents.supermimic.motion_library import SMMotionLibrary
from noeira.deep_agents.supermimic.policy import (
    SMCritic, SMPolicy, SM_ACT, SM_DEC_IN, SM_GOAL_DIM, SM_HIST, SM_LATENT, SM_OBS, sm_goal_scale,
    sm_hist_flat_index, sm_obs_noise_half, sm_obs_scale,
)
from noeira.deep_agents.supermimic.replay import SMReplayBuffer
from noeira.deep_agents.supermimic.trainer import SMPolicyTrainer, SMUpdateMetrics
from noeira.deep_agents.supermimic.world_model import SMWindowBatch, SMWorldModel, SMWMLoss, sm_wm_window_loss
from noeira.nn.constants import DT
from noeira.nn.core.initializer import Kaiming
from noeira.nn.core.param import ParamVisitor
from noeira.nn.core.tensor import Tensor, TensorImpl
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.optimizer.adam import Adam
from noeira.nn.optimizer.grad_clip import clip_grad_norm
from std.random import random_float64


comptime SM_WM_K: Int = 8
comptime SM_WM_BATCHES: Int = 10


def sm_cosine_lr(step: Int, total: Int, base: Float64, final_ratio: Float64 = 0.03, decay_fraction: Float64 = 250000.0 / 320000.0) -> Float64:
    """`schedule.cosine_lr`."""
    var decay = max(1, Int(Float64(total) * decay_fraction))
    if step >= decay:
        return base * final_ratio
    var c = 0.5 * (1.0 + cos(pi * Float64(step) / Float64(decay)))
    return base * (final_ratio + (1.0 - final_ratio) * c)


struct _ScaleParam(ParamVisitor):
    """Multiply one named param by `k` (the decoder's last layer starts at gain 0.01)."""
    var name: String
    var k: Scalar[DT]
    var hit: Int

    def __init__(out self, var name: String, k: Scalar[DT]):
        self.name = name^
        self.k = k
        self.hit = 0

    def visit[target: StaticString, N: Int](
        mut self, name: String, mut param: Tensor, mut grad: Tensor,
        mut m: Tensor, mut v: Tensor, apply_decay: Bool, ctx: Optional[DeviceContext],
    ) raises:
        if name == self.name:
            for i in range(N):
                param.data[i] *= self.k
            self.hit += 1


@fieldwise_init
struct SMTrainCfg(Copyable, Movable):
    var total_iterations: Int
    var utd: Float64
    var warmup: Int
    var lr: Float64
    var obs_noise: Bool
    var log_interval: Int


struct SMTrainer[
    HW: Int, TH: Int, E1: Int, E2: Int, E3: Int, E4: Int, D1: Int, D2: Int, D3: Int, D4: Int,
    C1: Int, C2: Int, NL: Int, BW: Int, BP: Int, H: Int,
](Movable):
    var env: SMG1Env
    var rb: SMReplayBuffer
    var wm: SMWorldModel[Self.HW, Self.TH]
    var pol: SMPolicy[Self.E1, Self.E2, Self.E3, Self.E4, Self.D1, Self.D2, Self.D3, Self.D4]
    var critic: SMCritic[Self.C1, Self.C2]
    var wm_opt: Adam
    var pt: SMPolicyTrainer
    var cfg: SMTrainCfg
    var iteration: Int
    var updates: Int
    var utd_acc: Float64
    var ep_sum: List[Float64]
    var ep_n: List[Int]
    var finished: List[Float64]
    var t_sim: Float64
    var t_wm: Float64
    var t_pi: Float64

    def __init__(out self, var env: SMG1Env, capacity: Int, var cfg: SMTrainCfg) raises:
        if env.num_lanes() != Self.NL:
            raise Error("SMTrainer: " + String(env.num_lanes()) + " lanes, compiled for " + String(Self.NL))
        self.env = env^
        self.rb = SMReplayBuffer(Self.NL, capacity)
        self.wm = SMWorldModel[Self.HW, Self.TH].make[Kaiming]()
        self.pol = SMPolicy[Self.E1, Self.E2, Self.E3, Self.E4, Self.D1, Self.D2, Self.D3, Self.D4].make[Kaiming]()
        self.critic = SMCritic[Self.C1, Self.C2].make["cpu", Kaiming](None)
        var sc = _ScaleParam(String("decoder.4.weight"), Scalar[DT](0.01))
        self.pol.for_each_param["cpu"](sc, None)
        if sc.hit != 1:
            raise Error("SMTrainer: the decoder's last layer was not found to rescale")
        self.wm_opt = Adam(lr=Scalar[DT](cfg.lr))
        self.pt = SMPolicyTrainer()
        self.cfg = cfg^
        self.iteration = 0
        self.updates = 0
        self.utd_acc = 0.0
        self.ep_sum = List[Float64](length=Self.NL, fill=0.0)
        self.ep_n = List[Int](length=Self.NL, fill=0)
        self.finished = List[Float64]()
        self.t_sim = 0.0
        self.t_wm = 0.0
        self.t_pi = 0.0

    def __init__(out self, *, deinit move: Self):
        self.env = move.env^
        self.rb = move.rb^
        self.wm = move.wm^
        self.pol = move.pol^
        self.critic = move.critic^
        self.wm_opt = move.wm_opt^
        self.pt = move.pt^
        self.cfg = move.cfg^
        self.iteration = move.iteration
        self.updates = move.updates
        self.utd_acc = move.utd_acc
        self.ep_sum = move.ep_sum^
        self.ep_n = move.ep_n^
        self.finished = move.finished^
        self.t_sim = move.t_sim
        self.t_wm = move.t_wm
        self.t_pi = move.t_pi

    def act(mut self) raises -> List[Scalar[DT]]:
        var ein = TensorImpl[DT].alloc(Self.NL * SM_GOAL_DIM)
        var g = List[Scalar[DT]](length=SM_GOAL_DIM, fill=0)
        for i in range(Self.NL):
            self.env.goal(i, g)
            for k in range(SM_GOAL_DIM):
                ein.data[i * SM_GOAL_DIM + k] = g[k] * Scalar[DT](sm_goal_scale(k))
        var z = TensorImpl[DT].alloc(Self.NL * SM_LATENT)
        self.pol.enc.forward["cpu", Self.NL](TensorRefs[1, _, DT](ein), z, None)
        var din = TensorImpl[DT].alloc(Self.NL * SM_DEC_IN)
        for i in range(Self.NL):
            for k in range(SM_LATENT):
                din.data[i * SM_DEC_IN + k] = z.data[i * SM_LATENT + k]
            for hh in range(SM_HIST):
                for k in range(SM_OBS):
                    din.data[i * SM_DEC_IN + SM_LATENT + sm_hist_flat_index(hh, k)] = (
                        self.env.lanes[i].hist[hh * SM_OBS + k] * Scalar[DT](sm_obs_scale(k))
                    )
        var a = TensorImpl[DT].alloc(Self.NL * SM_ACT)
        self.pol.dec.forward["cpu", Self.NL](TensorRefs[1, _, DT](din), a, None)
        return a.data.copy()

    def wm_update(mut self) raises -> SMWMLoss:
        var acc = SMWMLoss(0, 0, 0, 0, 0)
        for _ in range(SM_WM_BATCHES):
            var idx = SMReplayBuffer.sample_indices(self.rb.window_mask(SM_WM_K), Self.BW)
            var w = self.rb.windows_at(idx, SM_WM_K)
            var wb = SMWindowBatch(SM_WM_K, Self.BW)
            for i in range(len(w.states)):
                wb.states[i] = w.states[i]
            for i in range(len(w.a_exec)):
                wb.a_exec[i] = w.a_exec[i]
            for i in range(len(w.xi)):
                wb.xi[i] = w.xi[i]
            for i in range(len(w.done)):
                wb.done[i] = w.done[i]
                var f = w.frame[i]
                for c in range(7):
                    wb.ref_root[i * 7 + c] = self.env.lib.rows[f * 71 + c]
            self.wm.zero_grad["cpu"](None)
            var l = sm_wm_window_loss[Self.HW, Self.TH, Self.BW](self.wm, wb, True)
            _ = clip_grad_norm["cpu"](self.wm, Scalar[DT](1.0), None)
            self.wm_opt.step["cpu"](self.wm, None)
            acc.loss += l.loss / Float64(SM_WM_BATCHES)
            acc.state_loss += l.state_loss / Float64(SM_WM_BATCHES)
            acc.term_bce += l.term_bce / Float64(SM_WM_BATCHES)
            acc.nse_1step += l.nse_1step / Float64(SM_WM_BATCHES)
            acc.nse_kstep += l.nse_kstep / Float64(SM_WM_BATCHES)
        return acc^

    def policy_update(mut self) raises -> SMUpdateMetrics:
        var idx = SMReplayBuffer.sample_indices(self.rb.start_mask(SM_HIST), Self.BP)
        var st = self.rb.starts_at(idx, SM_HIST)
        var ib = SMImagBatch(Self.BP, Self.H)
        for i in range(len(st.state)):
            ib.state[i] = st.state[i]
        for i in range(len(st.prev_action)):
            ib.prev_action[i] = st.prev_action[i]
        for i in range(len(st.xi)):
            ib.xi[i] = st.xi[i]
        for hh in range(SM_HIST):
            for b in range(Self.BP):
                var o = (hh * Self.BP + b) * SM_OBS
                sm_observation_lane[DT](
                    self.env.table.ptr(), st.hist_states.unsafe_ptr().unsafe_offset((hh * Self.BP + b) * 71).as_unsafe_any_origin(),
                    st.hist_actions.unsafe_ptr().unsafe_offset((hh * Self.BP + b) * 29).as_unsafe_any_origin(),
                    st.xi.unsafe_ptr().unsafe_offset(b * 33).as_unsafe_any_origin(),
                    ib.hist0.unsafe_ptr().unsafe_offset(o).as_unsafe_any_origin(),
                )
                if self.cfg.obs_noise:
                    for k in range(SM_OBS):
                        var hw = sm_obs_noise_half(k)
                        if hw > 0:
                            ib.hist0[o + k] += Scalar[DT](random_float64(-hw, hw))
        for b in range(Self.BP):
            var last = self.env.lib.last_frame_of(st.frame[b])
            for t in range(Self.H + 1):
                var f = min(st.frame[b] + t, last)
                self.env.lib.goal_window(f, ib.goals.unsafe_ptr().unsafe_offset((t * Self.BP + b) * SM_GOAL_DIM).as_unsafe_any_origin())
                self.env.lib.reference_row(
                    f, self.env.tree, self.env.table,
                    ib.refs.unsafe_ptr().unsafe_offset((t * Self.BP + b) * SM_REF_DIM).as_unsafe_any_origin(),
                )
        var vals = List[Float64]()
        var tgts = List[Float64]()
        var none = List[Int]()
        return self.pt.update[
            Self.HW, Self.TH, Self.E1, Self.E2, Self.E3, Self.E4, Self.D1, Self.D2, Self.D3, Self.D4,
            Self.C1, Self.C2, Self.BP, Self.H,
        ](self.wm, self.pol, self.critic, self.env.tree, self.env.table, ib, none, vals, tgts, self.cfg.obs_noise)

    def iterate(mut self) raises -> Tuple[Bool, SMWMLoss, SMUpdateMetrics]:
        """One env step, and the updates it is owed. Returns (updated, ...)."""
        var t0 = perf_counter_ns()
        var a = self.act()
        _ = self.env.step(a, self.rb)
        for i in range(Self.NL):
            self.ep_sum[i] += self.env.last_reward[i]
            self.ep_n[i] += 1
            if self.env.lanes[i].ep_len == 0:      # reset this step
                self.finished.append(self.ep_sum[i] / (0.1 * Float64(self.ep_n[i])))
                self.ep_sum[i] = 0.0
                self.ep_n[i] = 0
        self.t_sim += Float64(perf_counter_ns() - t0) * 1e-9
        self.iteration += 1
        var lw = SMWMLoss(0, 0, 0, 0, 0)
        var lp = SMUpdateMetrics(0, 0, 0, 0, 0, 0)
        var updated = False
        if self.iteration >= self.cfg.warmup:
            self.utd_acc += self.cfg.utd
            while self.utd_acc >= 1.0:
                self.utd_acc -= 1.0
                var lr = sm_cosine_lr(self.iteration, self.cfg.total_iterations, self.cfg.lr)
                self.wm_opt.lr = Scalar[DT](lr)
                self.pt.pol_opt.lr = Scalar[DT](lr)
                self.pt.critic_opt.lr = Scalar[DT](lr)
                var t1 = perf_counter_ns()
                lw = self.wm_update()
                var t2 = perf_counter_ns()
                lp = self.policy_update()
                self.t_wm += Float64(t2 - t1) * 1e-9
                self.t_pi += Float64(perf_counter_ns() - t2) * 1e-9
                self.updates += 1
                updated = True
        return (updated, lw^, lp^)
