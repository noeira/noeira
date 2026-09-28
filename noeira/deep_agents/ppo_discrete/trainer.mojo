"""PPODiscreteTrainer — block-composed categorical PPO trainer (STORAGE).

Discrete sibling of `ppo/trainer.mojo`. Same 6-block composition over
`OnPolicyState`, but the policy is a softmax categorical over
`N_ACTIONS` logits instead of a diagonal Gaussian:

  PPODiscreteActStep      — per env-step: actor.forward → logits +
                            categorical sample + critic.forward
  PPORecordStep           — per env-step: push cached → rollout buffer
  PPOGAEStep              — per rollout:  bootstrap + per-env GAE
  PPOMinibatchGatherStep  — per epoch:    Fisher-Yates shuffle
                            per minibatch: gather + normalise mb_adv
  PPODiscreteActorTrainStep — per minibatch: categorical clipped surrogate
  PPOCriticTrainStep      — per minibatch: critic MSE update

The GAE / gather / record / critic blocks are SHARED verbatim with the
continuous trainer (they are policy-agnostic and only need ACT=1 — the
action slot holds a discrete index). The act-step and actor-loss are
the only discrete-specific blocks.

Exposes both the single-env (host-list) surface and the N_ENVS-wide
batched pointer surface. Dual-target (CPU/GPU via `train_target`):
per-step actor/critic forwards run on device (the act-step H2Ds obs and
D2Hs logits/value); rollout buffers live host-only; the K-epoch
minibatch is H2D-uploaded before each train step.

STORAGE migration: nets are storage `Module`s, optimizers are storage `Adam`
(arena-adopted on GPU), every block passes storage `Tensor`s. Checkpoint is v3,
the train-step counter a `K` scalar (legacy v2 files still load).
The GPU diag forward writes into an owned device `Tensor` scratch (`_diag_logits`);
the per-sample / EV kernels read/write owned `Tensor`s via `.lt["gpu", layout]()`
views (no raw pointers). The EV kernel is reused from `ppo.trainer`.
"""

from noeira.nn.core.param import walk_params, ParamVisitorRef
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.math import exp as fexp, log as flog
from std.memory import alloc
from std.time import perf_counter_ns
from layout import Layout, LayoutTensor

from noeira.core.logger import Logger, NoOpLogger
from noeira.nn.constants import DT, TPB, TPB_REDUCE
from noeira.nn.core.module import Module
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.call import call_forward, call_vjp
from noeira.nn.core.initializer import Xavier
from noeira.nn.optimizer.adam import Adam
from noeira.nn.core.checkpoint import (
    BinaryCheckpointWriter, BinaryCheckpointReader, CheckpointReader,
    CheckpointScalars, write_model, read_model, _split_lines, _is_v3_header,
    _read_file_bytes, _write_file_bytes,
)

from noeira.nn.core.log_bundle import log_bundle
from noeira.nn.core.metric import LogScalar
from noeira.nn.training.timer import Timer

from ..training.device_mean_accum import DeviceMeanAccum
from ..ppo.trainer import _ppo_ev_kernel
from ..training.episode_tracker import EpisodeTracker
from ..training.onpolicy_state import OnPolicyState
from ..training.driver_onpolicy_discrete import (
    OnPolicyDiscreteAgent,
    OnPolicyDiscreteAgentBatched,
)
from ..ppo.blocks.record_step import PPORecordStep
from ..ppo.blocks.gae_step import PPOGAEStep
from ..ppo.blocks.minibatch_gather_step import PPOMinibatchGatherStep
from ..ppo.blocks.critic_train_step import PPOCriticTrainStep
from ..ppo.metrics import PPOMetrics
from .blocks.act_step import PPODiscreteActStep
from .blocks.actor_train_step import PPODiscreteActorTrainStep


comptime _DIAG_LOG_PROB_DIFF_MAX: Scalar[DT] = 20.0


# ──────────────────────────────────────────────────────────────────────────
# GPU diag kernel — one thread per minibatch row; mirrors the CPU walk in
# `_accumulate_diag` (log-softmax entropy + log-prob at the taken index +
# Schulman-2020 KL + clip indicator). Writes per-sample ent / kl / clip into
# three `[MB]` buffers; explained variance reuses `ppo.trainer._ppo_ev_kernel`.
# ──────────────────────────────────────────────────────────────────────────
def _ppo_discrete_diag_kernel[MB: Int, N: Int](
    logits: LayoutTensor[DT, Layout.row_major(MB, N), MutAnyOrigin],
    act: LayoutTensor[DT, Layout.row_major(MB), MutAnyOrigin],
    olp: LayoutTensor[DT, Layout.row_major(MB), MutAnyOrigin],
    clip_eps: Scalar[DT],
    ent_out: LayoutTensor[DT, Layout.row_major(MB), MutAnyOrigin],
    kl_out: LayoutTensor[DT, Layout.row_major(MB), MutAnyOrigin],
    clip_out: LayoutTensor[DT, Layout.row_major(MB), MutAnyOrigin],
):
    var b = Int(global_idx.x)
    if b >= MB:
        return
    var max_l = rebind[Scalar[DT]](logits[b, 0])
    for j in range(1, N):
        var lj = rebind[Scalar[DT]](logits[b, j])
        if lj > max_l:
            max_l = lj
    var sum_exp: Scalar[DT] = 0.0
    for j in range(N):
        sum_exp += fexp(rebind[Scalar[DT]](logits[b, j]) - max_l)
    var log_sum = flog(sum_exp)
    var ent: Scalar[DT] = 0.0
    for j in range(N):
        var lp_j = (rebind[Scalar[DT]](logits[b, j]) - max_l) - log_sum
        var p_j = fexp(lp_j)
        ent += -p_j * lp_j
    var a_idx = Int(rebind[Scalar[DT]](act[b]))
    var nlp = (rebind[Scalar[DT]](logits[b, a_idx]) - max_l) - log_sum
    var diff = nlp - rebind[Scalar[DT]](olp[b])
    if diff > _DIAG_LOG_PROB_DIFF_MAX:
        diff = _DIAG_LOG_PROB_DIFF_MAX
    elif diff < -_DIAG_LOG_PROB_DIFF_MAX:
        diff = -_DIAG_LOG_PROB_DIFF_MAX
    var ratio = fexp(diff)
    kl_out[b] = (ratio - Scalar[DT](1.0)) - diff
    var dev = ratio - Scalar[DT](1.0)
    if dev < Scalar[DT](0.0):
        dev = -dev
    clip_out[b] = Scalar[DT](1.0) if dev > clip_eps else Scalar[DT](0.0)
    ent_out[b] = ent


struct PPODiscreteTrainer[
    train_target: StaticString,
    ACTOR: Module,
    CRITIC: Module,
    OBS_DIM: Int,
    N_ACTIONS: Int,
    ROLLOUT_LEN: Int,
    MINIBATCH: Int,
    N_EPOCHS: Int,
    N_ENVS: Int = 1,
](OnPolicyDiscreteAgentBatched):
    """CleanRL-style categorical PPO trainer. Exposes both the single-env
    (host-list) surface and the N_ENVS-wide batched pointer surface;
    `N_ENVS` sizes the per-step scratches (defaults to 1). The GAE +
    K-epoch update (`train_step`) is already N_ENVS-generic — the batched
    methods just feed it N_ENVS-wide pointers."""

    # OnPolicyDiscreteAgent(Batched) trait-visible comptime aliases.
    comptime AGENT_TRAIN_TARGET = Self.train_target
    comptime AGENT_OBS_DIM      = Self.OBS_DIM
    comptime AGENT_NUM_ACTIONS  = Self.N_ACTIONS
    comptime AGENT_N_ENVS       = Self.N_ENVS

    comptime N_MINIBATCHES = (Self.ROLLOUT_LEN * Self.N_ENVS) // Self.MINIBATCH

    comptime _T_GAE = 0
    comptime _T_UPDATE = 1
    comptime _T_DIAG = 2

    # ── Networks + optimisers ────────────────────────────────────────
    var actor: Self.ACTOR
    var critic: Self.CRITIC
    var actor_opt: Adam
    var critic_opt: Adam

    # ── Blocks ───────────────────────────────────────────────────────
    var act_step: PPODiscreteActStep[
        Self.OBS_DIM, Self.N_ACTIONS, Self.N_ENVS, Self.ACTOR, Self.CRITIC,
    ]
    var record_step: PPORecordStep[Self.OBS_DIM, 1, Self.ROLLOUT_LEN]
    var gae_step: PPOGAEStep[Self.OBS_DIM, Self.ROLLOUT_LEN, Self.CRITIC]
    var gather_step: PPOMinibatchGatherStep[
        Self.OBS_DIM, 1, Self.ROLLOUT_LEN, Self.MINIBATCH,
    ]
    var actor_train: PPODiscreteActorTrainStep[
        Self.OBS_DIM, Self.MINIBATCH, Self.ACTOR,
    ]
    var critic_train: PPOCriticTrainStep[
        Self.OBS_DIM, Self.MINIBATCH, Self.CRITIC,
    ]

    # ── State (ACT=1: the action slot holds a discrete index) ─────────
    var state: OnPolicyState[
        Self.OBS_DIM, 1, Self.ROLLOUT_LEN, Self.MINIBATCH, Self.N_ENVS,
    ]

    # Host-side staging for the N=1 host-list wrapper paths.
    var _obs1: Optional[Pointer[Scalar[DT], MutUntrackedOrigin]]
    var _act1: Optional[Pointer[Scalar[DT], MutUntrackedOrigin]]
    var _rew1: Optional[Pointer[Scalar[DT], MutUntrackedOrigin]]
    var _done1: Optional[Pointer[Scalar[DT], MutUntrackedOrigin]]
    var _nobs1: Optional[Pointer[Scalar[DT], MutUntrackedOrigin]]

    # ── Hyperparameters ──────────────────────────────────────────────
    var gamma: Scalar[DT]
    var gae_lambda: Scalar[DT]
    var clip_eps: Scalar[DT]
    var entropy_coef: Scalar[DT]
    var max_grad_norm: Scalar[DT]

    # ── Episode tracker ──────────────────────────────────────────────
    var tracker: EpisodeTracker
    var _ep_returns: Optional[Pointer[Scalar[DT], MutUntrackedOrigin]]  # N_ENVS

    # ── Train-step accumulators ──────────────────────────────────────
    var _actor_L_accum: Scalar[DT]
    var _critic_L_accum: Scalar[DT]
    var _entropy_accum: Scalar[DT]
    var _kl_accum: Scalar[DT]
    var _clip_accum: Scalar[DT]
    var _ev_accum: Scalar[DT]
    # Diag actor forward scratch (MINIBATCH * N_ACTIONS) — host on CPU,
    # device-resident on GPU (the diag kernels read its `.dev` buffer).
    var _diag_logits: Tensor
    # GPU diag: device-resident mean accumulators + the [MB]/[1] kernel-output
    # scratches the diag kernels write.
    var _entropy_mean_dev: DeviceMeanAccum
    var _kl_mean_dev: DeviceMeanAccum
    var _clip_mean_dev: DeviceMeanAccum
    var _ev_mean_dev: DeviceMeanAccum
    var _diag_ent: Tensor
    var _diag_kl: Tensor
    var _diag_clip: Tensor
    var _diag_ev: Tensor
    var _update_count: Int
    var _total_train_steps: Int

    var timer: Timer
    var ctx: Optional[DeviceContext]

    def __init__(out self):
        comptime assert (
            Self.train_target == "cpu" or Self.train_target == "gpu"
        ), "PPODiscreteTrainer: train_target must be 'cpu' or 'gpu'"
        comptime assert Self.ACTOR.IN_DIMS[0] == Self.OBS_DIM, (
            "PPODiscreteTrainer: ACTOR.IN_DIM must equal OBS_DIM"
        )
        comptime assert Self.ACTOR.OUT_DIM == Self.N_ACTIONS, (
            "PPODiscreteTrainer: ACTOR.OUT_DIM must equal N_ACTIONS"
        )
        comptime assert Self.CRITIC.IN_DIMS[0] == Self.OBS_DIM, (
            "PPODiscreteTrainer: CRITIC.IN_DIM must equal OBS_DIM"
        )
        comptime assert Self.CRITIC.OUT_DIM == 1, (
            "PPODiscreteTrainer: CRITIC.OUT_DIM must equal 1"
        )
        comptime assert (Self.ROLLOUT_LEN * Self.N_ENVS) % Self.MINIBATCH == 0, (
            "PPODiscreteTrainer: ROLLOUT_LEN * N_ENVS must be divisible by MINIBATCH"
        )
        comptime assert Self.N_ENVS >= 1, "PPODiscreteTrainer: N_ENVS must be >= 1"
        self.actor = Self.ACTOR()
        self.critic = Self.CRITIC()
        self.actor_opt = Adam()
        self.critic_opt = Adam()
        self.act_step = PPODiscreteActStep[
            Self.OBS_DIM, Self.N_ACTIONS, Self.N_ENVS, Self.ACTOR, Self.CRITIC,
        ]()
        self.record_step = PPORecordStep[Self.OBS_DIM, 1, Self.ROLLOUT_LEN]()
        self.gae_step = PPOGAEStep[
            Self.OBS_DIM, Self.ROLLOUT_LEN, Self.CRITIC,
        ]()
        self.gather_step = PPOMinibatchGatherStep[
            Self.OBS_DIM, 1, Self.ROLLOUT_LEN, Self.MINIBATCH,
        ]()
        self.actor_train = PPODiscreteActorTrainStep[
            Self.OBS_DIM, Self.MINIBATCH, Self.ACTOR,
        ]()
        self.critic_train = PPOCriticTrainStep[
            Self.OBS_DIM, Self.MINIBATCH, Self.CRITIC,
        ]()
        self.state = OnPolicyState[
            Self.OBS_DIM, 1, Self.ROLLOUT_LEN, Self.MINIBATCH, Self.N_ENVS,
        ]()
        self._obs1  = None
        self._act1  = None
        self._rew1  = None
        self._done1 = None
        self._nobs1 = None
        self.gamma = Scalar[DT](0.99)
        self.gae_lambda = Scalar[DT](0.95)
        self.clip_eps = Scalar[DT](0.2)
        self.entropy_coef = Scalar[DT](0.0)
        self.max_grad_norm = Scalar[DT](0.0)
        self.tracker = EpisodeTracker.new(
            window_size=10, initial_fill=Scalar[DT](0.0),
        )
        self._ep_returns = None
        self._actor_L_accum = Scalar[DT](0.0)
        self._critic_L_accum = Scalar[DT](0.0)
        self._entropy_accum = Scalar[DT](0.0)
        self._kl_accum = Scalar[DT](0.0)
        self._clip_accum = Scalar[DT](0.0)
        self._ev_accum = Scalar[DT](0.0)
        self._diag_logits = Tensor()
        self._entropy_mean_dev = DeviceMeanAccum()
        self._kl_mean_dev = DeviceMeanAccum()
        self._clip_mean_dev = DeviceMeanAccum()
        self._ev_mean_dev = DeviceMeanAccum()
        self._diag_ent = Tensor()
        self._diag_kl = Tensor()
        self._diag_clip = Tensor()
        self._diag_ev = Tensor()
        self._update_count = 0
        self._total_train_steps = 0
        self.timer = Timer.new()
        self.ctx = None

    @staticmethod
    def make(
        actor_lr: Scalar[DT] = Scalar[DT](3e-4),
        critic_lr: Scalar[DT] = Scalar[DT](1e-3),
        gamma: Scalar[DT] = Scalar[DT](0.99),
        gae_lambda: Scalar[DT] = Scalar[DT](0.95),
        clip_eps: Scalar[DT] = Scalar[DT](0.2),
        entropy_coef: Scalar[DT] = Scalar[DT](0.01),
        window_size: Int = 10,
        initial_episode_fill: Scalar[DT] = Scalar[DT](0.0),
        max_grad_norm: Scalar[DT] = Scalar[DT](0.0),
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        comptime assert (
            Self.train_target == "cpu" or Self.train_target == "gpu"
        ), "PPODiscreteTrainer.make: train_target must be 'cpu' or 'gpu'"
        comptime if Self.train_target == "gpu":
            if not ctx:
                raise Error(
                    "PPODiscreteTrainer.make[train_target='gpu']: ctx required"
                )
        var t = Self()
        t.ctx = ctx
        t.actor = Self.ACTOR.make[target=Self.train_target, INIT=Xavier](
            ctx=ctx
        )
        t.critic = Self.CRITIC.make[target=Self.train_target, INIT=Xavier](
            ctx=ctx
        )
        t.actor_opt = Adam(lr=actor_lr)
        t.actor_opt.adopt[Self.train_target, M=Self.ACTOR](t.actor, ctx)
        t.critic_opt = Adam(lr=critic_lr)
        t.critic_opt.adopt[Self.train_target, M=Self.CRITIC](t.critic, ctx)
        t.act_step = PPODiscreteActStep[
            Self.OBS_DIM, Self.N_ACTIONS, Self.N_ENVS, Self.ACTOR, Self.CRITIC,
        ].make[Self.train_target](ctx=ctx)
        t.record_step = PPORecordStep[
            Self.OBS_DIM, 1, Self.ROLLOUT_LEN,
        ].make[Self.train_target](ctx=ctx)
        t.gae_step = PPOGAEStep[
            Self.OBS_DIM, Self.ROLLOUT_LEN, Self.CRITIC,
        ].make[Self.train_target](ctx=ctx)
        t.gather_step = PPOMinibatchGatherStep[
            Self.OBS_DIM, 1, Self.ROLLOUT_LEN, Self.MINIBATCH,
        ].make[Self.train_target](ctx=ctx)
        t.actor_train = PPODiscreteActorTrainStep[
            Self.OBS_DIM, Self.MINIBATCH, Self.ACTOR,
        ].make[Self.train_target](
            ctx=ctx, clip_eps=clip_eps, entropy_coef=entropy_coef,
        )
        t.critic_train = PPOCriticTrainStep[
            Self.OBS_DIM, Self.MINIBATCH, Self.CRITIC,
        ].make[Self.train_target](ctx=ctx)
        t.state = OnPolicyState[
            Self.OBS_DIM, 1, Self.ROLLOUT_LEN, Self.MINIBATCH, Self.N_ENVS,
        ].make[Self.train_target](ctx=ctx)
        t._obs1  = alloc[Scalar[DT]]({count = Self.OBS_DIM}).unsafe_leak()
        t._act1  = alloc[Scalar[DT]]({count = 1}).unsafe_leak()
        t._rew1  = alloc[Scalar[DT]]({count = 1}).unsafe_leak()
        t._done1 = alloc[Scalar[DT]]({count = 1}).unsafe_leak()
        t._nobs1 = alloc[Scalar[DT]]({count = Self.OBS_DIM}).unsafe_leak()
        var ep_returns_p = alloc[Scalar[DT]](
            {count = Self.N_ENVS}
        ).unsafe_leak()
        for e in range(Self.N_ENVS):
            ep_returns_p[unsafe_offset=e] = Scalar[DT](0.0)
        t._ep_returns = ep_returns_p

        # Diag actor-output scratch (MINIBATCH * N_ACTIONS) on the train
        # target; the GPU diag kernels read its device buffer.
        t._diag_logits = Tensor.make[Self.train_target](
            Self.MINIBATCH * Self.N_ACTIONS, ctx
        )

        comptime if Self.train_target == "gpu":
            var c = ctx.value()
            t._diag_ent = Tensor.alloc_gpu(c, Self.MINIBATCH)
            t._diag_kl = Tensor.alloc_gpu(c, Self.MINIBATCH)
            t._diag_clip = Tensor.alloc_gpu(c, Self.MINIBATCH)
            t._diag_ev = Tensor.alloc_gpu(c, 1)
            t._entropy_mean_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)
            t._kl_mean_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)
            t._clip_mean_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)
            t._ev_mean_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)

        t.gamma = gamma
        t.gae_lambda = gae_lambda
        t.clip_eps = clip_eps
        t.entropy_coef = entropy_coef
        t.max_grad_norm = max_grad_norm
        t.tracker = EpisodeTracker.new(
            window_size=window_size, initial_fill=initial_episode_fill,
        )
        t.timer.add_section("gae")
        t.timer.add_section("update")
        t.timer.add_section("diag")
        return t^

    # ──────────────────────────────────────────────────────────────────
    # OnPolicyDiscreteAgent surface (single-env host-list)
    # ──────────────────────────────────────────────────────────────────

    def select_action(
        mut self,
        ref obs: List[Scalar[DT]],
        step_idx: Int,
    ) raises -> Int:
        comptime assert Self.N_ENVS == 1, (
            "PPODiscreteTrainer.select_action: host-list wrapper only "
            "valid at N_ENVS=1"
        )
        _ = step_idx
        var obs_p = self._obs1.value().as_unsafe_any_origin()
        var act_p = self._act1.value().as_unsafe_any_origin()
        for d in range(Self.OBS_DIM):
            obs_p[unsafe_offset=d] = obs[d]
        self.act_step.step[
            Self.train_target, Self.ROLLOUT_LEN, Self.MINIBATCH, Self.N_ENVS,
        ](self.state, self.actor, self.critic, obs_p, act_p)
        return Int(act_p[unsafe_offset=0])

    def select_greedy_action(
        mut self,
        ref obs: List[Scalar[DT]],
    ) raises -> Int:
        return self.act_step.step_greedy_n1[
            Self.train_target, Self.ROLLOUT_LEN, Self.MINIBATCH, Self.N_ENVS,
        ](self.state, self.actor, obs)

    def record_transition(
        mut self,
        ref obs: List[Scalar[DT]],
        action_idx: Int,
        reward: Scalar[DT],
        ref next_obs: List[Scalar[DT]],
        done: Scalar[DT],
    ) raises:
        comptime assert Self.N_ENVS == 1, (
            "PPODiscreteTrainer.record_transition: host-list wrapper only "
            "valid at N_ENVS=1"
        )
        # The action index was cached by select_action; the passed
        # action_idx is ignored (cached sample is authoritative — mirrors
        # the continuous trainer ignoring the env-ready action).
        _ = action_idx
        var obs_p = self._obs1.value().as_unsafe_any_origin()
        var nobs_p = self._nobs1.value().as_unsafe_any_origin()
        var rew_p = self._rew1.value().as_unsafe_any_origin()
        var done_p = self._done1.value().as_unsafe_any_origin()
        for d in range(Self.OBS_DIM):
            obs_p[unsafe_offset=d]  = obs[d]
            nobs_p[unsafe_offset=d] = next_obs[d]
        rew_p[unsafe_offset=0]  = reward
        done_p[unsafe_offset=0] = done
        self.record_step.step[
            Self.train_target, Self.MINIBATCH, Self.N_ENVS,
        ](self.state, obs_p, rew_p, nobs_p, done_p)
        self.tracker.add_reward(reward)

    def mark_terminal(mut self) raises:
        comptime assert Self.N_ENVS == 1, (
            "PPODiscreteTrainer.mark_terminal: host-list wrapper only "
            "valid at N_ENVS=1"
        )
        self.record_step.mark_terminal[
            Self.train_target, Self.MINIBATCH, Self.N_ENVS,
        ](self.state, 0)

    def end_episode(mut self):
        self.tracker.end_episode()

    # ──────────────────────────────────────────────────────────────────
    # OnPolicyDiscreteAgentBatched surface (N_ENVS-wide pointers)
    # ──────────────────────────────────────────────────────────────────

    def select_action_batched(
        mut self,
        obs_ptr: Pointer[Scalar[DT], MutAnyOrigin],
        action_ptr: Pointer[Scalar[DT], MutAnyOrigin],
        step_idx: Int,
    ) raises:
        """N_ENVS-wide action selection. Reads N_ENVS*OBS from obs_ptr,
        writes N_ENVS discrete indices (as floats) into action_ptr, caches
        per-env (index, log_prob, value) for the next record. Same act step
        as the N=1 wrapper, just fed the wide pointers directly."""
        _ = step_idx
        self.act_step.step[
            Self.train_target, Self.ROLLOUT_LEN, Self.MINIBATCH, Self.N_ENVS,
        ](self.state, self.actor, self.critic, obs_ptr, action_ptr)

    def record_batch_cpu(
        mut self,
        obs_ptr: Pointer[Scalar[DT], MutAnyOrigin],
        reward_ptr: Pointer[Scalar[DT], MutAnyOrigin],
        next_obs_ptr: Pointer[Scalar[DT], MutAnyOrigin],
        done_ptr: Pointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        """N_ENVS-wide transition record. Maintains a per-env running
        return (_ep_returns[e]); on done[e] pushes the completed return
        into the EpisodeTracker (add_reward + end_episode), mirroring the
        continuous trainer's record_batch_cpu."""
        self.record_step.step[
            Self.train_target, Self.MINIBATCH, Self.N_ENVS,
        ](self.state, obs_ptr, reward_ptr, next_obs_ptr, done_ptr)
        var ep_ret_p = self._ep_returns.value().as_unsafe_any_origin()
        for e in range(Self.N_ENVS):
            ep_ret_p[unsafe_offset=e] += reward_ptr[unsafe_offset=e]
            if done_ptr[unsafe_offset=e] > Scalar[DT](0.5):
                self.tracker.add_reward(ep_ret_p[unsafe_offset=e])
                self.tracker.end_episode()
                ep_ret_p[unsafe_offset=e] = Scalar[DT](0.0)

    def mark_terminal_env(mut self, env_idx: Int) raises:
        self.record_step.mark_terminal[
            Self.train_target, Self.MINIBATCH, Self.N_ENVS,
        ](self.state, env_idx)

    def train_step(mut self, step_idx: Int) raises -> Bool:
        _ = step_idx
        if self.state.rollout_idx < Self.ROLLOUT_LEN:
            return False

        var t_gae = perf_counter_ns()
        self.gae_step.step[
            Self.train_target, 1, Self.MINIBATCH, Self.N_ENVS,
        ](self.state, self.critic, self.gamma, self.gae_lambda)
        self.timer.accumulate(Self._T_GAE, t_gae)

        var t_upd = perf_counter_ns()
        self.gather_step.reset_indices[Self.train_target, Self.N_ENVS](
            self.state
        )
        for _epoch in range(Self.N_EPOCHS):
            self.gather_step.shuffle_epoch[Self.train_target, Self.N_ENVS](
                self.state
            )
            for mb in range(Self.N_MINIBATCHES):
                self.gather_step.gather[Self.train_target, Self.N_ENVS](
                    self.state, mb
                )
                var aL = self.actor_train.step[
                    Self.train_target, Self.ROLLOUT_LEN, Self.N_ENVS,
                ](self.state, self.actor, self.actor_opt, self.max_grad_norm)
                var cL = self.critic_train.step[
                    Self.train_target, 1, Self.ROLLOUT_LEN, Self.N_ENVS,
                ](self.state, self.critic, self.critic_opt, self.max_grad_norm)
                self._actor_L_accum += aL
                self._critic_L_accum += cL
                self._update_count += 1
                self._total_train_steps += 1
                var t_diag = perf_counter_ns()
                comptime if Self.train_target == "cpu":
                    self._accumulate_diag()
                else:
                    self._accumulate_diag_gpu()
                self.timer.accumulate(Self._T_DIAG, t_diag)
        self.timer.accumulate(Self._T_UPDATE, t_upd)

        self.record_step.reset_rollout[
            Self.train_target, Self.MINIBATCH, Self.N_ENVS,
        ](self.state)
        return True

    def _accumulate_diag(mut self) raises:
        """CPU-only per-minibatch categorical-PPO diagnostics: entropy,
        Schulman-2020 approx_kl, clip_fraction, explained variance.
        Re-runs the (post-update) actor over the minibatch obs and reads
        the critic's value/return scratches. Indexes the `.data` Lists."""
        comptime N = Self.N_ACTIONS
        comptime MB = Self.MINIBATCH

        call_forward["cpu", MB](
            self.actor, TensorRefs[Self.ACTOR.ARITY](self.state.mb_obs),
            self._diag_logits, self.ctx,
        )

        ref lg = self._diag_logits.data
        ref act = self.state.mb_act.data
        ref olp = self.state.mb_olp.data
        ref v = self.state.mb_v.data
        ref ret = self.state.mb_ret.data

        var ent_sum = Scalar[DT](0.0)
        var kl_sum = Scalar[DT](0.0)
        var clip_sum = Scalar[DT](0.0)
        for b in range(MB):
            var base = b * N
            var max_l = lg[base]
            for j in range(1, N):
                if lg[base + j] > max_l:
                    max_l = lg[base + j]
            var sum_exp = Scalar[DT](0.0)
            for j in range(N):
                sum_exp += fexp(lg[base + j] - max_l)
            var log_sum = flog(sum_exp)
            var ent = Scalar[DT](0.0)
            for j in range(N):
                var lp_j = (lg[base + j] - max_l) - log_sum
                var p_j = fexp(lp_j)
                ent += -p_j * lp_j
            var a_idx = Int(act[b])
            var nlp = (lg[base + a_idx] - max_l) - log_sum
            var diff = nlp - olp[b]
            if diff > _DIAG_LOG_PROB_DIFF_MAX:
                diff = _DIAG_LOG_PROB_DIFF_MAX
            elif diff < -_DIAG_LOG_PROB_DIFF_MAX:
                diff = -_DIAG_LOG_PROB_DIFF_MAX
            var ratio = fexp(diff)
            kl_sum += (ratio - Scalar[DT](1.0)) - diff
            var dev = ratio - Scalar[DT](1.0)
            if dev < Scalar[DT](0.0):
                dev = -dev
            if dev > self.clip_eps:
                clip_sum += Scalar[DT](1.0)
            ent_sum += ent

        var inv_mb = Scalar[DT](1.0) / Scalar[DT](MB)
        self._entropy_accum += ent_sum * inv_mb
        self._kl_accum += kl_sum * inv_mb
        self._clip_accum += clip_sum * inv_mb

        # Explained variance (CleanRL): 1 - Var(ret - v) / Var(ret).
        var mean_ret = Scalar[DT](0.0)
        var mean_res = Scalar[DT](0.0)
        for b in range(MB):
            mean_ret += ret[b]
            mean_res += ret[b] - v[b]
        mean_ret *= inv_mb
        mean_res *= inv_mb
        var var_ret = Scalar[DT](0.0)
        var var_res = Scalar[DT](0.0)
        for b in range(MB):
            var dr = ret[b] - mean_ret
            var rr = (ret[b] - v[b]) - mean_res
            var_ret += dr * dr
            var_res += rr * rr
        var ev = Scalar[DT](0.0)
        if var_ret > Scalar[DT](1e-8):
            ev = Scalar[DT](1.0) - var_res / var_ret
        self._ev_accum += ev

    def _accumulate_diag_gpu(mut self) raises:
        """GPU mirror of `_accumulate_diag`: re-run the (post-update) actor on
        the minibatch obs on device, derive categorical entropy / KL / clip
        via `_ppo_discrete_diag_kernel` and explained variance via the shared
        `_ppo_ev_kernel`, then fold them into the device-resident running
        means. No per-minibatch D2H. Device views via `.lt` — no raw pointers."""
        comptime N = Self.N_ACTIONS
        comptime MB = Self.MINIBATCH
        var ctx = self.ctx.value()

        call_forward["gpu", MB](
            self.actor, TensorRefs[Self.ACTOR.ARITY](self.state.mb_obs),
            self._diag_logits, self.ctx,
        )

        comptime n_blocks = (MB + TPB - 1) // TPB
        comptime per_k = _ppo_discrete_diag_kernel[MB, N]
        ctx.enqueue_function[per_k](
            self._diag_logits.lt["gpu", Layout.row_major(MB, N)](),
            self.state.mb_act.lt["gpu", Layout.row_major(MB)](),
            self.state.mb_olp.lt["gpu", Layout.row_major(MB)](),
            self.clip_eps,
            self._diag_ent.lt["gpu", Layout.row_major(MB)](),
            self._diag_kl.lt["gpu", Layout.row_major(MB)](),
            self._diag_clip.lt["gpu", Layout.row_major(MB)](),
            grid_dim=n_blocks, block_dim=TPB,
        )
        self._entropy_mean_dev.accumulate_gpu_lt[MB](
            self._diag_ent.lt["gpu", Layout.row_major(MB)]()
        )
        self._kl_mean_dev.accumulate_gpu_lt[MB](
            self._diag_kl.lt["gpu", Layout.row_major(MB)]()
        )
        self._clip_mean_dev.accumulate_gpu_lt[MB](
            self._diag_clip.lt["gpu", Layout.row_major(MB)]()
        )

        comptime ev_k = _ppo_ev_kernel[MB]
        ctx.enqueue_function[ev_k](
            self.state.mb_ret.lt["gpu", Layout.row_major(MB)](),
            self.state.mb_v.lt["gpu", Layout.row_major(MB)](),
            self._diag_ev.lt["gpu", Layout.row_major(1)](),
            grid_dim=1, block_dim=TPB_REDUCE,
        )
        self._ev_mean_dev.accumulate_gpu_lt[1](
            self._diag_ev.lt["gpu", Layout.row_major(1)]()
        )

    def mean_return(self) -> Scalar[DT]:
        return self.tracker.mean_return()

    def ep_count(self) -> Int:
        return self.tracker.ep_count

    # ─── Logging surface ─────────────────────────────────────────────

    def total_train_steps(self) -> Int:
        return self._total_train_steps

    def flush_metrics[
        L: Logger = NoOpLogger
    ](
        mut self,
        logger: Optional[Pointer[L, MutAnyOrigin]] = None,
        step: Int = 0,
    ) raises -> PPOMetrics:
        """Drain accumulators into a PPOMetrics bundle (reused verbatim —
        `entropy` is the categorical entropy here). Resets per-chunk
        accumulators; `_total_train_steps` is NOT reset."""
        var n = self._update_count if self._update_count > 0 else 1
        var inv = Scalar[DT](1.0) / Scalar[DT](n)
        # entropy / approx_kl / clip_fraction / explained_variance are
        # device-resident on GPU (derived by `_accumulate_diag_gpu`), host
        # scalars on CPU.
        var entropy_mean: Scalar[DT]
        var kl_mean: Scalar[DT]
        var clip_mean: Scalar[DT]
        var ev_mean: Scalar[DT]
        comptime if Self.train_target == "gpu":
            entropy_mean = self._entropy_mean_dev.read["gpu"]()
            kl_mean = self._kl_mean_dev.read["gpu"]()
            clip_mean = self._clip_mean_dev.read["gpu"]()
            ev_mean = self._ev_mean_dev.read["gpu"]()
        else:
            entropy_mean = self._entropy_accum * inv
            kl_mean = self._kl_accum * inv
            clip_mean = self._clip_accum * inv
            ev_mean = self._ev_accum * inv
        var bundle = PPOMetrics(
            policy_loss=LogScalar[DT](self._actor_L_accum * inv),
            critic_loss=LogScalar[DT](self._critic_L_accum * inv),
            train_steps=LogScalar[DT](Scalar[DT](self._total_train_steps)),
            n_updates=LogScalar[DT](Scalar[DT](self._update_count)),
            entropy=LogScalar[DT](entropy_mean),
            approx_kl=LogScalar[DT](kl_mean),
            clip_fraction=LogScalar[DT](clip_mean),
            explained_variance=LogScalar[DT](ev_mean),
        )
        self._actor_L_accum = Scalar[DT](0.0)
        self._critic_L_accum = Scalar[DT](0.0)
        self._entropy_accum = Scalar[DT](0.0)
        self._kl_accum = Scalar[DT](0.0)
        self._clip_accum = Scalar[DT](0.0)
        self._ev_accum = Scalar[DT](0.0)
        self._update_count = 0
        comptime if Self.train_target == "gpu":
            self._entropy_mean_dev.reset["gpu"]()
            self._kl_mean_dev.reset["gpu"]()
            self._clip_mean_dev.reset["gpu"]()
            self._ev_mean_dev.reset["gpu"]()
        if Bool(logger):
            log_bundle(logger.value()[], bundle, step)
        return bundle^

    def flush_metrics_through_logger[L: Logger](
        mut self,
        logger: Optional[Pointer[L, MutAnyOrigin]],
        step: Int,
    ) raises:
        _ = self.flush_metrics[L](logger, step)

    # ─── Checkpoint (ONE v3 file: actor + critic + train-step counter) ───
    def save_state(mut self, path: String) raises:
        """ONE v3 `storage-ckpt` file: actor + critic params + state,
        name-prefixed `actor.` / `critic.`; the train-step counter as a `K`
        section. Atomic and chunked; target-agnostic (GPU params download
        first). Optimizer moments NOT persisted (on-policy resume re-rolls)."""
        var w = BinaryCheckpointWriter(save_moments=False)
        write_model[Self.train_target](w, self.actor, self.ctx, "actor")
        write_model[Self.train_target](w, self.critic, self.ctx, "critic")
        var sc = CheckpointScalars()
        sc.set_int("total_train_steps", self._total_train_steps)
        w.write_scalars(sc)
        _write_file_bytes(path, w.content)

    def load_state(mut self, path: String) raises:
        """Restore actor + critic (v3, or the legacy v2 text this trainer
        wrote before) and the train-step counter. PPO has no target nets, so
        there is no hard copy."""
        var bytes = _read_file_bytes(path)
        if _is_v3_header(bytes):
            var rb = BinaryCheckpointReader(bytes^)
            read_model[Self.train_target](rb, self.actor, self.ctx, "actor")
            read_model[Self.train_target](rb, self.critic, self.ctx, "critic")
            var sc = rb.read_scalars()
            rb.finish()
            self._total_train_steps = sc.get_int(
                "total_train_steps", self._total_train_steps
            )
        else:
            var content: String
            with open(path, "r") as f:
                content = String(f.read())
            var lines = _split_lines(content)
            var body = List[String]()
            for li in range(len(lines)):
                if lines[li].startswith("storage-ckpt"):
                    continue
                body.append(lines[li])
            var r = CheckpointReader(body^)
            r.mode = 0
            walk_params[Self.train_target](self.actor, r, self.ctx, "actor")
            walk_params[Self.train_target](self.critic, r, self.ctx, "critic")
            r.mode = 1
            var _sref3 = ParamVisitorRef.of[type_of(r), Self.train_target](r)
            self.actor.for_each_state[Self.train_target](_sref3, self.ctx, "actor")
            var _sref4 = ParamVisitorRef.of[type_of(r), Self.train_target](r)
            self.critic.for_each_state[Self.train_target](_sref4, self.ctx, "critic")
            self._total_train_steps = Int(
                self._scan_scalar(
                    content, "_total_train_steps=",
                    Scalar[DT](self._total_train_steps),
                )
            )

    @staticmethod
    def _scan_scalar(
        content: String, key: String, default: Scalar[DT],
    ) raises -> Scalar[DT]:
        """Scan `content` lines for `key<value>`; return its float (or
        `default` if absent)."""
        var lines = _split_lines(content)
        for i in range(len(lines)):
            if lines[i].startswith(key):
                var parts = lines[i].split("=")
                if len(parts) >= 2:
                    return Scalar[DT](atof(parts[len(parts) - 1]))
        return default

    def flush_timer_log(mut self) -> String:
        var report = self.timer.format_report()
        self.timer.reset()
        return report
