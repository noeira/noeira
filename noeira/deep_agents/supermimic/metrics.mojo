"""SuperMimic's metrics: one bundle per log line, dashboard names.

`SMMetrics` is a `@fieldwise_init` struct of `LogScalar`s, walked by
`log_bundle` (one `log_scalar(name, value, step)` per field), so a FIELD NAME
IS THE METRIC NAME the monitor groups by (`noeira-cloud`'s
`metric-groups.ts`). Names reuse the existing groups wherever the quantity
is the same one:

    avg_reward          finished episodes' return / (0.1 · length): the
                        fraction of the maximum per-step reward (Episode Reward);
                        a window where none finished repeats the last value
    episodes            episodes finished in the window (Training Progress)
    reward_mean         per-step simulator reward, all lanes (Reward Statistics)
    mean_done           fraction of lane-steps that terminated (Done Ratio)
    wm_loss             Eq. 1, the 10 world-model batches' mean (Loss)
    dyn_loss            its state term (Dynamics Loss)
    termination_loss    its termination BCE (World Model Losses)
    policy_loss         −J / H, the actor's loss (Policy Loss)
    imagined_reward_mean  the undiscounted imagined return / H (Imagined Returns)
    actor_grad_norm     the policy gradient's pre-clip norm (Gradient Norms)
    critic_loss         the critic fit's last minibatch MSE (Critic Loss)
    value_mean          V on the imagined states (Value Head)
    value_target_mean   its TD(λ) targets (Value Head)
    train_steps / n_updates   updates so far / in the window (Training Progress)
    steps_per_s / wall_s      env steps per second / elapsed (Runtime Throughput)

SuperMimic's own (no dashboard group yet): `wm_nse_1step` / `wm_nse_8step`
(the world model's open-loop error at 1 and k steps, the paper's diagnostic),
`lr`, and the time split `sim_s` / `wm_s` / `pi_s`.

THE CURRICULUM'S (`SMSamplerStats`, in units of the sampler's G_MAX):
`episode_len` (the finished episodes' mean length), `reset_bin_return`
(the mean estimated return of the bins the window's resets drew),
`bin_return_mean` (the estimate over all bins) and `reset_hard_share` (the
share of resets in the previous window's hardest decile of bins; 0.1 =
uniform). `avg_reward` is an average over FINISHED episodes whose start the
sampler chooses: read it against these.
"""

from std.time import perf_counter_ns

from noeira.core.logger import Logger
from noeira.deep_agents.supermimic.config import SMConfigT
from noeira.deep_agents.supermimic.motion_library import SMResetSampler, SMSamplerStats
from noeira.deep_agents.supermimic.trainer import SMUpdateMetrics, SuperMimicTrainer
from noeira.deep_agents.training.checkpoint import announce_checkpoint
from noeira.io.artifact_sink import ArtifactSink
from noeira.deep_agents.supermimic.world_model import SMWMLoss
from noeira.nn.constants import DT
from noeira.nn.core.log_bundle import log_bundle
from noeira.nn.core.metric import LogScalar


@fieldwise_init
struct SMIterInfo(Copyable, Movable):
    """One iteration: the env step's reward / terminations and, when updates
    ran, their metrics."""
    var updated: Bool
    var wm: SMWMLoss
    var pi: SMUpdateMetrics
    var reward_mean: Float64
    var n_terminated: Int


@fieldwise_init
struct SMMetrics(Copyable, Movable):
    var avg_reward: LogScalar[DT]
    var episodes: LogScalar[DT]
    var reward_mean: LogScalar[DT]
    var mean_done: LogScalar[DT]
    var wm_loss: LogScalar[DT]
    var dyn_loss: LogScalar[DT]
    var termination_loss: LogScalar[DT]
    var wm_nse_1step: LogScalar[DT]
    var wm_nse_8step: LogScalar[DT]
    var policy_loss: LogScalar[DT]
    var imagined_reward_mean: LogScalar[DT]
    var actor_grad_norm: LogScalar[DT]
    var critic_loss: LogScalar[DT]
    var value_mean: LogScalar[DT]
    var value_target_mean: LogScalar[DT]
    var lr: LogScalar[DT]
    var train_steps: LogScalar[DT]
    var n_updates: LogScalar[DT]
    var steps_per_s: LogScalar[DT]
    var wall_s: LogScalar[DT]
    var sim_s: LogScalar[DT]
    var wm_s: LogScalar[DT]
    var pi_s: LogScalar[DT]
    var episode_len: LogScalar[DT]
    var reset_bin_return: LogScalar[DT]
    var bin_return_mean: LogScalar[DT]
    var reset_hard_share: LogScalar[DT]


@always_inline
def _ls(x: Float64) -> LogScalar[DT]:
    return LogScalar[DT](Scalar[DT](x))


struct SMTrainLog(Movable):
    """Means between two log lines; both agents feed it the same way."""
    var horizon: Int
    var lanes: Int
    var n_it: Int
    var n_up: Int
    var rsum: Float64
    var nterm: Int
    var wl: Float64
    var sl: Float64
    var tl: Float64
    var w1: Float64
    var w8: Float64
    var ij: Float64
    var iu: Float64
    var gn: Float64
    var cl: Float64
    var vm: Float64
    var tmn: Float64
    var t0: Int
    var t_win: Int
    var last_avg: Float64
    var last_len: Float64

    def __init__(out self, horizon: Int, lanes: Int):
        self.horizon = horizon
        self.lanes = lanes
        self.n_it = 0
        self.n_up = 0
        self.rsum = 0.0
        self.nterm = 0
        self.wl = 0.0
        self.sl = 0.0
        self.tl = 0.0
        self.w1 = 0.0
        self.w8 = 0.0
        self.ij = 0.0
        self.iu = 0.0
        self.gn = 0.0
        self.cl = 0.0
        self.vm = 0.0
        self.tmn = 0.0
        self.t0 = perf_counter_ns()
        self.t_win = self.t0
        self.last_avg = 0.0
        self.last_len = 0.0

    def __init__(out self, *, deinit move: Self):
        self.horizon = move.horizon
        self.lanes = move.lanes
        self.n_it = move.n_it
        self.n_up = move.n_up
        self.rsum = move.rsum
        self.nterm = move.nterm
        self.wl = move.wl
        self.sl = move.sl
        self.tl = move.tl
        self.w1 = move.w1
        self.w8 = move.w8
        self.ij = move.ij
        self.iu = move.iu
        self.gn = move.gn
        self.cl = move.cl
        self.vm = move.vm
        self.tmn = move.tmn
        self.t0 = move.t0
        self.t_win = move.t_win
        self.last_avg = move.last_avg
        self.last_len = move.last_len

    def add(mut self, r: SMIterInfo):
        self.n_it += 1
        self.rsum += r.reward_mean
        self.nterm += r.n_terminated
        if not r.updated:
            return
        self.n_up += 1
        self.wl += r.wm.loss
        self.sl += r.wm.state_loss
        self.tl += r.wm.term_bce
        self.w1 += r.wm.nse_1step
        self.w8 += r.wm.nse_kstep
        self.ij += r.pi.imag_objective
        self.iu += r.pi.imag_return_undisc
        self.gn += r.pi.policy_grad_norm
        self.cl += r.pi.critic_loss
        self.vm += r.pi.critic_value_mean
        self.tmn += r.pi.critic_target_mean

    def flush(
        mut self, updates: Int, finished: List[Float64], finished_len: List[Int], ss: SMSamplerStats, lr: Float64,
        t_sim: Float64, t_wm: Float64, t_pi: Float64,
    ) -> SMMetrics:
        """The window's bundle; starts the next window."""
        var tr = 0.0
        var tl = 0
        for k in range(len(finished)):
            tr += finished[k]
        for k in range(len(finished_len)):
            tl += finished_len[k]
        var nf = len(finished)
        var d = Float64(max(self.n_up, 1))
        var di = Float64(max(self.n_it, 1))
        var now = perf_counter_ns()
        var win_s = Float64(now - self.t_win) * 1e-9
        var h = Float64(self.horizon)
        if nf > 0:            # a window with no finished episode repeats the last value
            self.last_avg = tr / Float64(nf)
        if len(finished_len) > 0:
            self.last_len = Float64(tl) / Float64(len(finished_len))
        var m = SMMetrics(
            _ls(self.last_avg), _ls(Float64(nf)),
            _ls(self.rsum / di), _ls(Float64(self.nterm) / (di * Float64(self.lanes))),
            _ls(self.wl / d), _ls(self.sl / d), _ls(self.tl / d), _ls(self.w1 / d), _ls(self.w8 / d),
            _ls(-self.ij / d / h), _ls(self.iu / d / h), _ls(self.gn / d),
            _ls(self.cl / d), _ls(self.vm / d), _ls(self.tmn / d),
            _ls(lr), _ls(Float64(updates)), _ls(Float64(self.n_up)),
            _ls(Float64(self.n_it) / max(win_s, 1e-9)), _ls(Float64(now - self.t0) * 1e-9),
            _ls(t_sim), _ls(t_wm), _ls(t_pi),
            _ls(self.last_len), _ls(ss.reset_bin_return), _ls(ss.bin_return_mean), _ls(ss.reset_hard_share),
        )
        self.n_it = 0
        self.n_up = 0
        self.rsum = 0.0
        self.nterm = 0
        self.wl = 0.0
        self.sl = 0.0
        self.tl = 0.0
        self.w1 = 0.0
        self.w8 = 0.0
        self.ij = 0.0
        self.iu = 0.0
        self.gn = 0.0
        self.cl = 0.0
        self.vm = 0.0
        self.tmn = 0.0
        self.t_win = now
        return m^


def sm_print_metrics(it: Int, m: SMMetrics):
    """The terminal's line (the same numbers the logger gets)."""
    print(
        "it", it, "| updates", Int(m.train_steps.v), "| avg_reward", m.avg_reward.v, "(n=", Int(m.episodes.v), ")",
        "| r", m.reward_mean.v, "done", m.mean_done.v,
        "| wm", m.wm_loss.v, "nse1", m.wm_nse_1step.v, "nse8", m.wm_nse_8step.v,
        "| pi_loss", m.policy_loss.v, "imag_r", m.imagined_reward_mean.v, "|g|", m.actor_grad_norm.v,
        "| critic", m.critic_loss.v, "V", m.value_mean.v, "R", m.value_target_mean.v,
        "| len", m.episode_len.v, "G_reset", m.reset_bin_return.v, "G_bins", m.bin_return_mean.v, "hard", m.reset_hard_share.v,
        "|", m.steps_per_s.v, "steps/s | sim/wm/pi", m.sim_s.v, m.wm_s.v, m.pi_s.v, "s",
    )


def sm_emit[L: Logger](logger: Optional[Pointer[L, MutAnyOrigin]], m: SMMetrics, step: Int) raises:
    """The bundle to the run's logger (CSV + monitor), if there is one."""
    if logger:
        var lp = logger.value()
        log_bundle(lp[], m, step)
        lp[].flush()


def sm_tick[C: SMConfigT, L: Logger](
    mut trainer: SuperMimicTrainer[C], mut lg: SMTrainLog, mut sampler: SMResetSampler, it: Int, total: Int, updates: Int,
    finished: List[Float64], finished_len: List[Int], lr: Float64, t_sim: Float64, t_wm: Float64, t_pi: Float64,
    log_every: Int, logger: Optional[Pointer[L, MutAnyOrigin]], verbose: Bool,
    checkpoint_every: Int, checkpoint_path: String, artifacts: Optional[ArtifactSink], run_dir: String,
) raises -> Bool:
    """After iteration `it` (1-based) of `total`: the log line and bundle
    every `log_every`, the checkpoint every `checkpoint_every` and at the
    end (offered to the run's artifact sink). Returns True when it logged
    (the caller then starts a new window of finished episodes)."""
    var logged = False
    if log_every > 0 and (it % log_every == 0 or it == total):
        var m = lg.flush(updates, finished, finished_len, sampler.take_stats(), lr, t_sim, t_wm, t_pi)
        if verbose:
            sm_print_metrics(it, m)
            if sampler.count:
                print("  sampler: count estimator, untouched bins", sampler.untouched_share())
        sm_emit(logger, m, it)
        logged = True
    if checkpoint_path.byte_length() > 0 and ((checkpoint_every > 0 and it % checkpoint_every == 0) or it == total):
        trainer.save_state(checkpoint_path, it, updates)
        announce_checkpoint(checkpoint_path, artifacts, run_dir)
    return logged
