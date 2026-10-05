"""Continuous PPO on any GPU-batched env — the rollout on the device, the env
step / reset and the K-epoch update optionally captured in CUDA graphs.

    from noeira.deep_agents.training.ppo_vec_driver import (
        PPOVecConfig, run_ppo_vec,
    )
    var env = MyBatchedEnv[N_ENVS](ctx)
    env.reset_batch[N_ENVS](ctx=ctx, rng_seed=UInt64(seed))   # the CALLER resets
    var agent = PPOAgent["gpu", Actor, Critic, OBS, ACT, ROLLOUT, MB, EPOCHS, N_ENVS](
        ctx=ctx, ..., action_scale=4.0,
    )
    var obs_rms = RunningMeanStd(OBS)          # or `.load(<init run>/obs_norm.txt)`
    var res = run_ppo_vec[HIST=3, ENV_GRAPH=True, TRAIN_GRAPH=True](
        agent, env, ctx, cfg, obs_rms, logger, run.checkpoint_path("last"),
        run.dir + "/obs_norm.txt",
    )

The family driver (`tasks/ppo_family_driver.mojo`) without its family: no
servo model, no delta / target actions, no `--repeat`, no success bit. What
it keeps, each a `PPOVecConfig` field:

  - the policy's observation = the env's row, then the last `HIST` EXECUTED
    actions (newest first; zeroed when an episode starts);
  - running observation normalisation and CleanRL's `NormalizeReward`
    (`obs_norm.mojo`), each switchable (off = the identity);
  - the lr annealed to 0 and the entropy coefficient `ent0 -> ent1`, linearly
    over `anneal_steps`, or both fixed (`anneal=False` — the update graph is
    then captured ONCE instead of after every update);
  - an action-rate penalty `w * |a_t - a_{t-1}|^2` on the executed action,
    subtracted from the reward PPO learns from (not from the logged return);
  - the divergence guard: a lane whose observation has a non-finite word or
    one beyond `obs_bound` (or whose reward is beyond `rew_bound`) is ended
    as TERMINATED (its next observation is garbage — no bootstrap), its
    reward zeroed, its row kept out of the statistics, and counted;
  - the env's own `terminated_ptr()`: a termination bootstraps from 0, a
    time-limit truncation from V(its own final observation) (`gae_step.mojo`).

THE EXECUTED ACTION is the trainer's `clamp(sample, -action_scale,
action_scale)`, written straight into the env's action slab; the history and
the penalty read it there. The trainer records the UNclipped sample.

THE CHECKPOINT is two files: the trainer's v3 `storage-ckpt` (actor + critic,
`save_state`) and the observation statistics in `RunningMeanStd.save`'s text
format (`count c / mean ... / var ...`, OBS words each — the env's words,
then the history's). An eval normalises with them, clip `obs_clip`.

THE GREEDY EVALUATION is the caller's (it is the env's business what to
measure): `obs_rms` holds the final statistics when `run_ppo_vec` returns.

⚠ NOT BIT-IDENTICAL BETWEEN RUNS OF DIFFERENT GRAPH FLAGS? It IS: the env
graph and the train graph replay the eager kernels (`maybe_capture_replay`),
so `ENV_GRAPH` / `TRAIN_GRAPH` on or off give the same checkpoint
(`tests/deep_agents/test_ppo_vec_driver.mojo`).
"""

from layout import Layout, LayoutTensor
from max.gpu import global_idx
from max.gpu.host import DeviceContext, HostBuffer
from std.time import perf_counter_ns

from noeira.core.logger import Logger
from noeira.cuda import CUDAGraph, maybe_capture_replay
from noeira.deep_agents.ppo import PPOAgent
from noeira.deep_agents.training.batched_env import BatchedEnv
from noeira.deep_agents.training.driver_onpolicy import (
    OnPolicyBatchedCore, onpolicy_update_device,
)
from noeira.deep_agents.training.obs_norm import (
    RunningMeanStd, augment_k, norm_reward, update_rms_device, normalize_device,
)
from noeira.nn.constants import DT, TPB
from noeira.nn.core.fill import fill_dev
from noeira.nn.core.module import Module
from noeira.nn.core.ptr import mptr
from noeira.nn.core.tensor import Tensor


comptime _V[n: Int] = LayoutTensor[DT, Layout.row_major(n), MutAnyOrigin]
comptime _Ptr = Pointer[Scalar[DT], MutAnyOrigin]

# the episode record per lane per step
comptime _EP_DONE = 0
comptime _EP_RET = 1
comptime _EP_LEN = 2
comptime _EP_W = 3
# the per-lane counters read at log cadence
comptime _S_DIVERGED = 0
comptime _S_PEN_SUM = 1
comptime _S_PEN_N = 2
comptime _S_SIZE = 3

comptime _NO_CLIP = 1.0e30
"""The clip of a switched-off normalisation: (x - 0) / sqrt(1 + 1e-8) is x in
Float32, so mean 0 / variance 1 / this clip is the identity."""


# ═══════════════════════════════════════════════════════════════════════════
# the configuration and the result
# ═══════════════════════════════════════════════════════════════════════════


struct PPOVecConfig(Copyable, Movable):
    """`run_ppo_vec`'s run-time options. PPO's own (gamma, lambda, clip, grad
    clip, action scale, log-std init) are the agent's, set at construction."""

    var total_steps: Int
    var lr: Float64
    """The initial learning rate (actor and critic); also the fixed one."""
    var ent0: Float64
    var ent1: Float64
    """The entropy coefficient's floor when annealed (ignored when fixed)."""
    var anneal: Bool
    var anneal_steps: Int
    """The schedule's length in env steps; 0 = `total_steps`."""
    var norm_obs: Bool
    var norm_reward: Bool
    var obs_clip: Float64
    var rew_clip: Float64
    var act_rate_w: Float64
    var obs_bound: Float64
    var rew_bound: Float64
    var seed: Int
    var ckpt_every: Int
    """Env steps between checkpoints (0 = only at the end)."""
    var log_every: Int
    """Updates between logger flushes."""
    var print_every: Int
    """Updates between console lines."""
    var window: Int
    """Completed episodes the logged return / length average over."""
    var episode_sync_every: Int
    """Policy steps of episode records buffered on the host before one sync."""

    def __init__(
        out self,
        total_steps: Int,
        lr: Float64 = 3e-4,
        ent0: Float64 = 0.0,
        ent1: Float64 = 0.0,
        anneal: Bool = False,
        anneal_steps: Int = 0,
        norm_obs: Bool = True,
        norm_reward: Bool = True,
        obs_clip: Float64 = 10.0,
        rew_clip: Float64 = 10.0,
        act_rate_w: Float64 = 0.0,
        obs_bound: Float64 = 1.0e3,
        rew_bound: Float64 = 1.0e3,
        seed: Int = 1,
        ckpt_every: Int = 0,
        log_every: Int = 1,
        print_every: Int = 10,
        window: Int = 100,
        episode_sync_every: Int = 32,
    ):
        self.total_steps = total_steps
        self.lr = lr
        self.ent0 = ent0
        self.ent1 = ent1
        self.anneal = anneal
        self.anneal_steps = anneal_steps
        self.norm_obs = norm_obs
        self.norm_reward = norm_reward
        self.obs_clip = obs_clip
        self.rew_clip = rew_clip
        self.act_rate_w = act_rate_w
        self.obs_bound = obs_bound
        self.rew_bound = rew_bound
        self.seed = seed
        self.ckpt_every = ckpt_every
        self.log_every = max(log_every, 1)
        self.print_every = max(print_every, 1)
        self.window = max(window, 1)
        self.episode_sync_every = max(episode_sync_every, 1)


@fieldwise_init
struct PPOVecResult(Copyable, Movable):
    var steps: Int
    var seconds: Float64
    var episodes: Int
    var mean_return: Float64
    """Over the last `window` completed episodes (the env's reward, raw)."""
    var mean_length: Float64
    var diverged: Int


# ═══════════════════════════════════════════════════════════════════════════
# the per-step kernels (one thread per lane)
# ═══════════════════════════════════════════════════════════════════════════


def _pre_k[N: Int, ACT: Int, W: Int](
    act: _V[N * ACT],
    a_prev: _V[N * ACT],
    has_prev: _V[N],
    pen: _V[N],
    hist: _V[N * W + 1],
    stats: _V[N * _S_SIZE],
    w: Scalar[DT],
):
    """After the policy acted (`act` = the executed action): the action-rate
    penalty against the lane's previous action (0 on an episode's first
    step), then the action pushed into the history, newest first."""
    var e = Int(global_idx.x)
    if e >= N:
        return
    var hp = rebind[Scalar[DT]](has_prev[e]) > Scalar[DT](0.5)
    var d2: Scalar[DT] = 0.0
    for j in range(ACT):
        var a = rebind[Scalar[DT]](act[e * ACT + j])
        if hp:
            var dd = a - rebind[Scalar[DT]](a_prev[e * ACT + j])
            d2 += dd * dd
        a_prev[e * ACT + j] = a
    has_prev[e] = Scalar[DT](1.0)
    var p = w * d2
    pen[e] = p
    if w > Scalar[DT](0.0):
        stats[e * _S_SIZE + _S_PEN_SUM] = (
            rebind[Scalar[DT]](stats[e * _S_SIZE + _S_PEN_SUM]) + p
        )
        stats[e * _S_SIZE + _S_PEN_N] = (
            rebind[Scalar[DT]](stats[e * _S_SIZE + _S_PEN_N]) + Scalar[DT](1.0)
        )
    comptime if W > 0:
        for k in range(W - 1, ACT - 1, -1):
            hist[e * W + k] = hist[e * W + k - ACT]
        for j in range(ACT):
            hist[e * W + j] = act[e * ACT + j]


def _post_k[N: Int, E_OBS: Int](
    obs: _V[N * E_OBS],
    rew: _V[N],
    done: _V[N],
    term: _V[N],
    pen: _V[N],
    diverged: _V[N],
    rew_raw: _V[N],
    done_out: _V[N],
    term_out: _V[N],
    raw_ret: _V[N],
    ep_len: _V[N],
    stats: _V[N * _S_SIZE],
    obs_bound: Scalar[DT],
    rew_bound: Scalar[DT],
):
    """After the env step: the divergence guard, the transition's reward
    (the env's, less the penalty), done and terminated. A diverged lane is
    ended as terminated and its done written back into the env's slab, so
    `selective_reset_batch` resets it."""
    var e = Int(global_idx.x)
    if e >= N:
        return
    var r = rebind[Scalar[DT]](rew[e])
    var bad = not (r == r) or abs(r) > rew_bound
    if not bad:
        for k in range(E_OBS):
            var v = rebind[Scalar[DT]](obs[e * E_OBS + k])
            if not (v == v) or abs(v) > obs_bound:
                bad = True
                break
    if bad:
        diverged[e] = Scalar[DT](1.0)
        done_out[e] = Scalar[DT](1.0)
        term_out[e] = Scalar[DT](1.0)
        done[e] = Scalar[DT](1.0)
        rew_raw[e] = Scalar[DT](0.0)
        stats[e * _S_SIZE + _S_DIVERGED] = (
            rebind[Scalar[DT]](stats[e * _S_SIZE + _S_DIVERGED]) + Scalar[DT](1.0)
        )
    else:
        diverged[e] = Scalar[DT](0.0)
        done_out[e] = done[e]
        term_out[e] = term[e]
        rew_raw[e] = r - rebind[Scalar[DT]](pen[e])
        raw_ret[e] = rebind[Scalar[DT]](raw_ret[e]) + r
    ep_len[e] = rebind[Scalar[DT]](ep_len[e]) + Scalar[DT](1.0)


def _ret_k[N: Int](
    rew_raw: _V[N], ret_acc: _V[N], rets: _V[N], gamma: Scalar[DT],
):
    """The discounted return `NormalizeReward` scales by."""
    var e = Int(global_idx.x)
    if e >= N:
        return
    var ra = rebind[Scalar[DT]](ret_acc[e]) * gamma + rebind[Scalar[DT]](rew_raw[e])
    ret_acc[e] = ra
    rets[e] = ra


def _rew_k[N: Int](
    rew_raw: _V[N],
    done_out: _V[N],
    ret_var: _V[1],
    rew_n: _V[N],
    raw_ret: _V[N],
    ret_acc: _V[N],
    ep_len: _V[N],
    ep: _V[N * _EP_W],
    clip: Scalar[DT],
):
    """The normalised reward, and the record of a lane whose episode ended
    (its raw return, its length) — then its accumulators start over."""
    var e = Int(global_idx.x)
    if e >= N:
        return
    rew_n[e] = norm_reward(
        rebind[Scalar[DT]](rew_raw[e]), rebind[Scalar[DT]](ret_var[0]), clip
    )
    if rebind[Scalar[DT]](done_out[e]) > Scalar[DT](0.5):
        ep[e * _EP_W + _EP_DONE] = Scalar[DT](1.0)
        ep[e * _EP_W + _EP_RET] = raw_ret[e]
        ep[e * _EP_W + _EP_LEN] = ep_len[e]
        raw_ret[e] = Scalar[DT](0.0)
        ret_acc[e] = Scalar[DT](0.0)
        ep_len[e] = Scalar[DT](0.0)
    else:
        ep[e * _EP_W + _EP_DONE] = Scalar[DT](0.0)


def _reset_k[N: Int, W: Int](
    done_out: _V[N], hist: _V[N * W + 1], has_prev: _V[N],
):
    """A lane whose episode ended: no history, no previous action."""
    var e = Int(global_idx.x)
    if e >= N:
        return
    if rebind[Scalar[DT]](done_out[e]) <= Scalar[DT](0.5):
        return
    for k in range(W):
        hist[e * W + k] = Scalar[DT](0.0)
    has_prev[e] = Scalar[DT](0.0)


# ═══════════════════════════════════════════════════════════════════════════
# the rollout's device state and its step
# ═══════════════════════════════════════════════════════════════════════════


struct PPOVecRollout[N_: Int, E_OBS_: Int, ACT_: Int, HIST_: Int](Movable):
    """The per-lane state of `run_ppo_vec` on the device. `OBS` is the
    policy's observation: the env's `E_OBS` words, then `HIST` actions."""

    comptime N = Self.N_
    comptime E_OBS = Self.E_OBS_
    comptime ACT = Self.ACT_
    comptime W = Self.HIST_ * Self.ACT_
    comptime OBS = Self.E_OBS + Self.W

    var ctx: DeviceContext
    var cur_n: Tensor
    var next_n: Tensor
    var aug: Tensor
    var hist: Tensor
    var a_prev: Tensor
    var has_prev: Tensor
    var pen: Tensor
    var tprev: Tensor  # `augment_k`'s target-lead input, unread (T = 0)
    var qa: Tensor
    var diverged: Tensor
    var rew_raw: Tensor
    var done_out: Tensor
    var term_out: Tensor
    var raw_ret: Tensor
    var ep_len: Tensor
    var ret_acc: Tensor
    var rets: Tensor
    var rew_n: Tensor
    var ep: Tensor
    var stats: Tensor
    var obs_mean: Tensor
    var obs_var: Tensor
    var obs_count: Tensor
    var ret_mean: Tensor
    var ret_var: Tensor
    var ret_count: Tensor
    var _ring_bufs: List[HostBuffer[DT]]
    var _pending: Int
    var _sync_every: Int

    def __init__(out self, ctx: DeviceContext, sync_every: Int) raises:
        comptime N = Self.N
        self.ctx = ctx
        self.cur_n = Tensor.alloc_gpu(ctx, N * Self.OBS)
        self.next_n = Tensor.alloc_gpu(ctx, N * Self.OBS)
        self.aug = Tensor.alloc_gpu(ctx, N * Self.OBS)
        self.hist = Tensor.alloc_gpu(ctx, N * Self.W + 1)
        self.a_prev = Tensor.alloc_gpu(ctx, N * Self.ACT)
        self.has_prev = Tensor.alloc_gpu(ctx, N)
        self.pen = Tensor.alloc_gpu(ctx, N)
        self.tprev = Tensor.alloc_gpu(ctx, N * Self.ACT)
        self.qa = Tensor.alloc_gpu(ctx, Self.ACT)
        self.diverged = Tensor.alloc_gpu(ctx, N)
        self.rew_raw = Tensor.alloc_gpu(ctx, N)
        self.done_out = Tensor.alloc_gpu(ctx, N)
        self.term_out = Tensor.alloc_gpu(ctx, N)
        self.raw_ret = Tensor.alloc_gpu(ctx, N)
        self.ep_len = Tensor.alloc_gpu(ctx, N)
        self.ret_acc = Tensor.alloc_gpu(ctx, N)
        self.rets = Tensor.alloc_gpu(ctx, N)
        self.rew_n = Tensor.alloc_gpu(ctx, N)
        self.ep = Tensor.alloc_gpu(ctx, N * _EP_W)
        self.stats = Tensor.alloc_gpu(ctx, N * _S_SIZE)
        self.obs_mean = Tensor.alloc_gpu(ctx, Self.OBS)
        self.obs_var = Tensor.alloc_gpu(ctx, Self.OBS)
        fill_dev(self.obs_var.dev.value(), Self.OBS, Scalar[DT](1.0), ctx)
        self.obs_count = Tensor.alloc_gpu(ctx, 1)
        fill_dev(self.obs_count.dev.value(), 1, Scalar[DT](1e-4), ctx)
        self.ret_mean = Tensor.alloc_gpu(ctx, 1)
        self.ret_var = Tensor.alloc_gpu(ctx, 1)
        fill_dev(self.ret_var.dev.value(), 1, Scalar[DT](1.0), ctx)
        self.ret_count = Tensor.alloc_gpu(ctx, 1)
        fill_dev(self.ret_count.dev.value(), 1, Scalar[DT](1e-4), ctx)
        self._sync_every = max(sync_every, 1)
        self._ring_bufs = List[HostBuffer[DT]]()
        for _ in range(self._sync_every):
            self._ring_bufs.append(ctx.enqueue_create_host_buffer[DT](N * _EP_W))
        self._pending = 0

    @staticmethod
    def _g(n: Int) -> Int:
        return (n + TPB - 1) // TPB

    # ── the observation statistics in and out (the host struct) ──────────

    def stats_from_host(mut self, ref rms: RunningMeanStd) raises:
        self.obs_mean.ensure(Self.OBS)
        self.obs_var.ensure(Self.OBS)
        for k in range(Self.OBS):
            self.obs_mean.data[k] = Scalar[DT](rms.mean[k])
            self.obs_var.data[k] = Scalar[DT](rms.var_[k])
        self.obs_mean.upload_resident(self.ctx)
        self.obs_var.upload_resident(self.ctx)
        fill_dev(self.obs_count.dev.value(), 1, Scalar[DT](rms.count), self.ctx)
        self.ctx.synchronize()

    def stats_to_host(mut self, mut rms: RunningMeanStd) raises:
        self.obs_mean.ensure(Self.OBS)
        self.obs_var.ensure(Self.OBS)
        self.obs_count.ensure(1)
        self.obs_mean.download(self.ctx)
        self.obs_var.download(self.ctx)
        self.obs_count.download(self.ctx)
        for k in range(Self.OBS):
            rms.mean[k] = Float64(self.obs_mean.data[k])
            rms.var_[k] = Float64(self.obs_var.data[k])
        rms.count = Float64(self.obs_count.data[0])

    def read_stats(mut self) raises -> List[Float64]:
        """[diverged (cumulative), penalty sum, penalty count], over lanes."""
        self.stats.ensure(Self.N * _S_SIZE)
        self.stats.download(self.ctx)
        var s = List[Float64](length=_S_SIZE, fill=0.0)
        for e in range(Self.N):
            for k in range(_S_SIZE):
                s[k] += Float64(self.stats.data[e * _S_SIZE + k])
        return s^

    def reset_penalty_stats(mut self) raises:
        self.stats.ensure(Self.N * _S_SIZE)
        self.stats.download(self.ctx)
        for e in range(Self.N):
            self.stats.data[e * _S_SIZE + _S_PEN_SUM] = Scalar[DT](0)
            self.stats.data[e * _S_SIZE + _S_PEN_N] = Scalar[DT](0)
        self.stats.upload_resident(self.ctx)

    # ── the episode records ─────────────────────────────────────────────

    def drain(
        mut self, mut ep_ret: List[Float64], mut ep_len: List[Float64],
    ) raises:
        """ONE sync, then every buffered step's records, lanes in order."""
        if self._pending == 0:
            return
        self.ctx.synchronize()
        for s in range(self._pending):
            var p = self._ring_bufs[s].unsafe_ptr()
            for e in range(Self.N):
                if p[unsafe_offset = e * _EP_W + _EP_DONE] > Scalar[DT](0.5):
                    ep_ret.append(Float64(p[unsafe_offset = e * _EP_W + _EP_RET]))
                    ep_len.append(Float64(p[unsafe_offset = e * _EP_W + _EP_LEN]))
        self._pending = 0

    def ring_full(self) -> Bool:
        return self._pending >= self._sync_every

    # ── the pieces ──────────────────────────────────────────────────────

    def _augment(mut self, raw_ptr: _Ptr) raises:
        comptime N = Self.N
        self.ctx.enqueue_function[
            augment_k[N, Self.E_OBS, Self.W, 0, Self.ACT]
        ](
            _V[N * Self.E_OBS](raw_ptr),
            self.hist.lt["gpu", Layout.row_major(N * Self.W + 1)](),
            self.tprev.lt["gpu", Layout.row_major(N * Self.ACT)](),
            self.qa.lt["gpu", Layout.row_major(Self.ACT)](),
            self.aug.lt["gpu", Layout.row_major(N * Self.OBS)](),
            grid_dim=Self._g(N), block_dim=TPB,
        )

    def start(mut self, raw_ptr: _Ptr, ref cfg: PPOVecConfig) raises:
        """The first observation: the statistics updated with every lane
        (when on), then normalised."""
        self._augment(raw_ptr)
        if cfg.norm_obs:
            update_rms_device[Self.N, Self.OBS](
                self.ctx, self.aug, self.diverged, self.obs_mean, self.obs_var,
                self.obs_count, False,
            )
        normalize_device[Self.N, Self.OBS](
            self.ctx, self.aug, self.cur_n, self.obs_mean, self.obs_var,
            self.diverged,
            cfg.obs_clip if cfg.norm_obs else _NO_CLIP, False,
        )

    def step[
        A: OnPolicyBatchedCore, E: BatchedEnv, USE_ENV_GRAPH: Bool
    ](
        mut self,
        mut trainer: A,
        mut env: E,
        ref cfg: PPOVecConfig,
        gamma: Scalar[DT],
        mut env_graph: Optional[CUDAGraph],
        mut reset_graph: Optional[CUDAGraph],
    ) raises:
        """One control step: act, penalty and history, the env step, the
        transition, normalisation, the record, the reset of the lanes that
        ended, the next observation."""
        comptime N = Self.N
        comptime EO = Self.E_OBS
        var c = self.ctx
        var obs_clip = cfg.obs_clip if cfg.norm_obs else _NO_CLIP
        # 1. act — the executed (clamped) action straight into the env
        trainer.select_action_device(
            mptr(self.cur_n.dev.value().unsafe_ptr()), env.action_ptr()
        )
        c.enqueue_function[_pre_k[N, Self.ACT, Self.W]](
            _V[N * Self.ACT](env.action_ptr()),
            self.a_prev.lt["gpu", Layout.row_major(N * Self.ACT)](),
            self.has_prev.lt["gpu", Layout.row_major(N)](),
            self.pen.lt["gpu", Layout.row_major(N)](),
            self.hist.lt["gpu", Layout.row_major(N * Self.W + 1)](),
            self.stats.lt["gpu", Layout.row_major(N * _S_SIZE)](),
            Scalar[DT](cfg.act_rate_w),
            grid_dim=Self._g(N), block_dim=TPB,
        )

        # 2. the env step (its randomness is a device counter: capture-safe)
        def _env_step() capturing raises -> None:
            env.step_batch[N](ctx=Optional(c), rng_seed=UInt64(0))

        comptime if USE_ENV_GRAPH:
            maybe_capture_replay[_env_step, VERBOSE=False](env_graph, c)
        else:
            _env_step()
        # 3. the transition, the next observation and its statistics
        c.enqueue_function[_post_k[N, EO]](
            _V[N * EO](env.obs_ptr()),
            _V[N](env.reward_ptr()),
            _V[N](env.done_ptr()),
            _V[N](env.terminated_ptr()),
            self.pen.lt["gpu", Layout.row_major(N)](),
            self.diverged.lt["gpu", Layout.row_major(N)](),
            self.rew_raw.lt["gpu", Layout.row_major(N)](),
            self.done_out.lt["gpu", Layout.row_major(N)](),
            self.term_out.lt["gpu", Layout.row_major(N)](),
            self.raw_ret.lt["gpu", Layout.row_major(N)](),
            self.ep_len.lt["gpu", Layout.row_major(N)](),
            self.stats.lt["gpu", Layout.row_major(N * _S_SIZE)](),
            Scalar[DT](cfg.obs_bound),
            Scalar[DT](cfg.rew_bound),
            grid_dim=Self._g(N), block_dim=TPB,
        )
        self._augment(env.obs_ptr())
        if cfg.norm_obs:
            update_rms_device[Self.N, Self.OBS](
                c, self.aug, self.diverged, self.obs_mean, self.obs_var,
                self.obs_count, True,
            )
        normalize_device[Self.N, Self.OBS](
            c, self.aug, self.next_n, self.obs_mean, self.obs_var,
            self.diverged, obs_clip, True,
        )
        # 4. reward normalisation and the episode records
        c.enqueue_function[_ret_k[N]](
            self.rew_raw.lt["gpu", Layout.row_major(N)](),
            self.ret_acc.lt["gpu", Layout.row_major(N)](),
            self.rets.lt["gpu", Layout.row_major(N)](),
            gamma,
            grid_dim=Self._g(N), block_dim=TPB,
        )
        if cfg.norm_reward:
            update_rms_device[Self.N, 1](
                c, self.rets, self.diverged, self.ret_mean, self.ret_var,
                self.ret_count, False,
            )
        c.enqueue_function[_rew_k[N]](
            self.rew_raw.lt["gpu", Layout.row_major(N)](),
            self.done_out.lt["gpu", Layout.row_major(N)](),
            self.ret_var.lt["gpu", Layout.row_major(1)](),
            self.rew_n.lt["gpu", Layout.row_major(N)](),
            self.raw_ret.lt["gpu", Layout.row_major(N)](),
            self.ret_acc.lt["gpu", Layout.row_major(N)](),
            self.ep_len.lt["gpu", Layout.row_major(N)](),
            self.ep.lt["gpu", Layout.row_major(N * _EP_W)](),
            Scalar[DT](cfg.rew_clip if cfg.norm_reward else _NO_CLIP),
            grid_dim=Self._g(N), block_dim=TPB,
        )
        if self._pending >= self._sync_every:
            raise Error(
                "PPOVecRollout: the episode ring is full — `drain` when"
                " `ring_full()`"
            )
        c.enqueue_copy(self._ring_bufs[self._pending], self.ep.dev.value())
        self._pending += 1
        # 5. record: the normalised reward and PRE-reset next observation,
        # done, and the true terminal (GAE bootstraps a truncation from V of
        # that observation, a termination from 0)
        trainer.record_device(
            mptr(self.rew_n.dev.value().unsafe_ptr()),
            mptr(self.next_n.dev.value().unsafe_ptr()),
            mptr(self.done_out.dev.value().unsafe_ptr()),
            mptr(self.term_out.dev.value().unsafe_ptr()),
        )

        # 6. reset the finished lanes; the observation they restart from
        def _env_reset() capturing raises -> None:
            env.selective_reset_batch[N](ctx=Optional(c), rng_seed=UInt64(0))

        comptime if USE_ENV_GRAPH:
            maybe_capture_replay[_env_reset, VERBOSE=False](reset_graph, c)
        else:
            _env_reset()
        c.enqueue_function[_reset_k[N, Self.W]](
            self.done_out.lt["gpu", Layout.row_major(N)](),
            self.hist.lt["gpu", Layout.row_major(N * Self.W + 1)](),
            self.has_prev.lt["gpu", Layout.row_major(N)](),
            grid_dim=Self._g(N), block_dim=TPB,
        )
        self._augment(env.obs_ptr())
        normalize_device[Self.N, Self.OBS](
            c, self.aug, self.cur_n, self.obs_mean, self.obs_var,
            self.diverged, obs_clip, False,
        )


# ═══════════════════════════════════════════════════════════════════════════
# the driver
# ═══════════════════════════════════════════════════════════════════════════


def _window_mean(ref xs: List[Float64], window: Int) -> Float64:
    var n = len(xs)
    var lo = n - window if n > window else 0
    if n - lo == 0:
        return 0.0
    var s = 0.0
    for k in range(lo, n):
        s += xs[k]
    return s / Float64(n - lo)


def run_ppo_vec[
    E: BatchedEnv,
    ACTOR: Module,
    CRITIC: Module,
    OBS: Int,
    ACT: Int,
    ROLLOUT: Int,
    MINIBATCH: Int,
    N_EPOCHS: Int,
    N_ENVS: Int,
    L: Logger,
    //,
    HIST: Int = 0,
    ENV_GRAPH: Bool = False,
    TRAIN_GRAPH: Bool = False,
](
    mut agent: PPOAgent[
        "gpu", ACTOR, CRITIC, OBS, ACT, ROLLOUT, MINIBATCH, N_EPOCHS, N_ENVS
    ],
    mut env: E,
    ctx: DeviceContext,
    cfg: PPOVecConfig,
    mut obs_rms: RunningMeanStd,
    mut logger: L,
    ckpt_path: String,
    obs_norm_path: String,
) raises -> PPOVecResult:
    """Train `agent` on `env` for `cfg.total_steps` env steps.

    The env must already be reset (`reset_batch`): a caller that staggers
    its lanes' episode clocks does it between the two. `obs_rms` seeds the
    observation statistics (a fresh one, or an `--init` run's) and holds the
    final ones on return; both files are written every `cfg.ckpt_every`
    steps and at the end."""
    comptime assert OBS == E.OBS_DIM + HIST * E.ACT_DIM, (
        "run_ppo_vec: the agent's OBS must be the env's plus HIST actions"
    )
    comptime assert ACT == E.ACT_DIM, "run_ppo_vec: the agent's ACT is not the env's"
    if len(obs_rms.mean) != OBS:
        raise Error(
            "run_ppo_vec: obs_rms has " + String(len(obs_rms.mean))
            + " words, the policy's observation " + String(OBS)
        )
    comptime N = N_ENVS
    var per_update = N * ROLLOUT
    var n_updates_total = (
        cfg.anneal_steps if cfg.anneal_steps > 0 else cfg.total_steps
    ) // per_update
    var gamma = agent.trainer.gamma

    var dev = PPOVecRollout[N, E.OBS_DIM, ACT, HIST](ctx, cfg.episode_sync_every)
    dev.stats_from_host(obs_rms)
    agent.trainer.enable_device_rollout(
        UInt64(cfg.seed) * UInt64(2654435761) + UInt64(1)
    )
    agent.trainer.actor_opt.set_lr(Scalar[DT](cfg.lr))
    agent.trainer.critic_opt.set_lr(Scalar[DT](cfg.lr))
    agent.trainer.actor_train.set_entropy_coef(Scalar[DT](cfg.ent0))
    dev.start(env.obs_ptr(), cfg)

    var ep_ret = List[Float64]()
    var ep_len = List[Float64]()
    var env_graph: Optional[CUDAGraph] = None
    var reset_graph: Optional[CUDAGraph] = None
    var train_graph: Optional[CUDAGraph] = None
    var logger_ptr = Optional(Pointer(to=logger).as_unsafe_any_origin())
    var step = 0
    var n_updates = 0
    var next_ckpt = cfg.ckpt_every if cfg.ckpt_every > 0 else -1
    var t0 = perf_counter_ns()
    while step < cfg.total_steps:
        dev.step[USE_ENV_GRAPH=ENV_GRAPH](
            agent.trainer, env, cfg, gamma, env_graph, reset_graph
        )
        if dev.ring_full():
            dev.drain(ep_ret, ep_len)
        step += N
        var updated = onpolicy_update_device[
            type_of(agent.trainer), TRAIN_GRAPH
        ](agent.trainer, Optional(ctx), step, train_graph, quiet=n_updates > 0)
        if updated:
            n_updates += 1
            var lr = cfg.lr
            var ent = cfg.ent0
            if cfg.anneal:
                var frac = 1.0 - Float64(n_updates) / Float64(max(n_updates_total, 1))
                if frac < 0.0:
                    frac = 0.0
                lr = cfg.lr * frac
                ent = cfg.ent1 + (cfg.ent0 - cfg.ent1) * frac
                agent.trainer.actor_opt.set_lr(Scalar[DT](lr))
                agent.trainer.critic_opt.set_lr(Scalar[DT](lr))
                agent.trainer.actor_train.set_entropy_coef(Scalar[DT](ent))
                # the new lr / entropy are kernel arguments: re-capture
                train_graph = None
            var log_now = n_updates % cfg.log_every == 0
            var print_now = n_updates % cfg.print_every == 0
            if log_now or print_now:
                dev.drain(ep_ret, ep_len)
                var st = dev.read_stats()
                var diverged = Int(st[_S_DIVERGED])
                var pen = st[_S_PEN_SUM] / st[_S_PEN_N] if st[_S_PEN_N] > 0 else 0.0
                var mret = _window_mean(ep_ret, cfg.window)
                var mlen = _window_mean(ep_len, cfg.window)
                var secs = Float64(perf_counter_ns() - t0) / 1e9
                if log_now:
                    logger.log_scalar("episode_return", mret, step)
                    logger.log_scalar("episode_length", mlen, step)
                    logger.log_scalar("episodes", Float64(len(ep_ret)), step)
                    logger.log_scalar("diverged", Float64(diverged), step)
                    logger.log_scalar("sps", Float64(step) / secs, step)
                    logger.log_scalar("lr", lr, step)
                    logger.log_scalar("ent_coef", ent, step)
                    if cfg.act_rate_w > 0.0:
                        logger.log_scalar("act_rate_penalty", pen, step)
                    agent.trainer.flush_metrics_through_logger[L](
                        logger_ptr, step
                    )
                if print_now:
                    print("  step", step, "| return", mret, "| length", mlen,
                          "| episodes", len(ep_ret), "| diverged", diverged,
                          "| act-rate penalty", pen,
                          "|", Int(Float64(step) / secs), "steps/s")
                dev.reset_penalty_stats()
        if next_ckpt > 0 and step >= next_ckpt:
            agent.trainer.save_state(ckpt_path)
            dev.stats_to_host(obs_rms)
            obs_rms.save(obs_norm_path)
            next_ckpt += cfg.ckpt_every
    dev.drain(ep_ret, ep_len)
    dev.stats_to_host(obs_rms)
    var diverged = Int(dev.read_stats()[_S_DIVERGED])
    if cfg.total_steps > 0:
        agent.trainer.save_state(ckpt_path)
        obs_rms.save(obs_norm_path)
    var secs = Float64(perf_counter_ns() - t0) / 1e9
    return PPOVecResult(
        step, secs, len(ep_ret), _window_mean(ep_ret, cfg.window),
        _window_mean(ep_len, cfg.window), diverged,
    )
