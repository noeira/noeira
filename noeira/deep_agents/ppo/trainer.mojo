"""PPOTrainer — block-composed PPO continuous trainer (CleanRL-style) (STORAGE).

Composes 6 ref-based step blocks via `OnPolicyState`:

  PPOActStep              — per env-step: actor.forward + sample + critic.forward
  PPORecordStep           — per env-step: push cached → rollout buffer
  PPOGAEStep              — per rollout: bootstrap + per-env GAE
  PPOMinibatchGatherStep  — per epoch:    Fisher-Yates shuffle
                            per minibatch: gather + normalise mb_adv
  PPOActorTrainStep       — per minibatch: actor PPO clipped surrogate update
  PPOCriticTrainStep      — per minibatch: critic MSE update

Dual-target (CPU/GPU via `train_target` struct comptime) × N_ENVS-
parametric (default 1). Single-env (N_ENVS=1) users get a host-list
`OnPolicyAgent` surface (select_action / record_transition / etc.)
consumed by `run_onpolicy_train`. Multi-env (N_ENVS>=1) users get
the pointer-based `OnPolicyAgentBatched` surface consumed by
`run_onpolicy_train_batched` over a `BatchedEnv` adapter.

GPU train_target is a hybrid: per-step actor/critic forwards run on
device (H2D obs + D2H ao/v inside PPOActStep); rollout buffers live
host-only; the K-epoch minibatch is H2D-uploaded into device-side
mb_* scratches before each PPOActorTrainStep / PPOCriticTrainStep.

STORAGE migration: nets are storage `Module`s, optimizers are storage `Adam`
(arena-adopted on GPU), every block passes storage `Tensor`s. Checkpoint is v3,
the train-step counter a `K` scalar (legacy v2 files still load).
The GPU diag forward writes into an owned device `Tensor` scratch (`_diag_ao`);
the per-sample / EV kernels read/write owned `Tensor`s via `.lt["gpu", layout]()`
views (no raw pointers).
"""

from noeira.nn.core.param import walk_params, ParamVisitorRef
from max.gpu import global_idx, thread_idx
from max.gpu.primitives import block
from max.gpu.host import DeviceContext, DeviceBuffer
from std.math import exp as fexp
from std.memory import alloc
from std.time import perf_counter_ns

from noeira.core.logger import Logger, NoOpLogger
from noeira.nn.constants import DT, TPB, TPB_REDUCE
from noeira.nn.core.module import Module
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.call import call_forward, call_vjp
from layout import Layout, LayoutTensor
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
from ..training.episode_tracker import EpisodeTracker
from ..training.onpolicy_state import OnPolicyState
from ..training.driver_onpolicy import OnPolicyAgent, OnPolicyAgentBatched
from .blocks.act_step import PPOActStep
from .blocks.record_step import PPORecordStep
from .blocks.gae_step import PPOGAEStep
from .blocks.minibatch_gather_step import PPOMinibatchGatherStep
from .blocks.actor_train_step import PPOActorTrainStep
from .blocks.critic_train_step import PPOCriticTrainStep
from .blocks.device_rollout import PPODeviceRollout
from .metrics import PPOMetrics
from .objective import (
    LOG_STD_MIN,
    LOG_STD_MAX,
    LOG_PROB_DIFF_MAX,
    EPS_STD,
    LOG_2PI,
)


# Diagnostic constants — aliased from ppo/objective.mojo (S4, 2026-06-07)
# so the recomputed entropy / ratio in the diag path can NEVER drift from
# the clamps the loss kernel actually applied. Previously these were five
# independent literals with a "MUST match" comment; now they are one
# source of truth.
comptime _DIAG_LOG_STD_MIN = LOG_STD_MIN
comptime _DIAG_LOG_STD_MAX = LOG_STD_MAX
comptime _DIAG_LOG_PROB_DIFF_MAX = LOG_PROB_DIFF_MAX
comptime _DIAG_EPS_STD = EPS_STD
comptime _DIAG_LOG_2PI = LOG_2PI


# ──────────────────────────────────────────────────────────────────────────
# GPU diag kernels — mirror the CPU walk in `_accumulate_diag` exactly (same
# clamps + Schulman-2020 KL + CleanRL explained variance).
#
# `_ppo_diag_per_sample_kernel`: one thread per minibatch row; recomputes the
# new log-prob from the (post-update) actor output `ao = [μ, log σ]`, the
# Gaussian entropy, the clamped ratio, and writes per-sample entropy / KL /
# clip-indicator into three `[MB]` buffers (reduced to means by the trainer).
# ──────────────────────────────────────────────────────────────────────────
def _ppo_diag_per_sample_kernel[MB: Int, ACT: Int](
    ao: LayoutTensor[DT, Layout.row_major(MB, 2 * ACT), MutAnyOrigin],
    act: LayoutTensor[DT, Layout.row_major(MB, ACT), MutAnyOrigin],
    olp: LayoutTensor[DT, Layout.row_major(MB), MutAnyOrigin],
    clip_eps: Scalar[DT],
    ent_out: LayoutTensor[DT, Layout.row_major(MB), MutAnyOrigin],
    kl_out: LayoutTensor[DT, Layout.row_major(MB), MutAnyOrigin],
    clip_out: LayoutTensor[DT, Layout.row_major(MB), MutAnyOrigin],
):
    var b = Int(global_idx.x)
    if b >= MB:
        return
    var nlp: Scalar[DT] = 0.0
    var ent: Scalar[DT] = 0.0
    for j in range(ACT):
        var mu = rebind[Scalar[DT]](ao[b, j])
        var ls = rebind[Scalar[DT]](ao[b, ACT + j])
        if ls < _DIAG_LOG_STD_MIN:
            ls = _DIAG_LOG_STD_MIN
        elif ls > _DIAG_LOG_STD_MAX:
            ls = _DIAG_LOG_STD_MAX
        var std = fexp(ls)
        var a = rebind[Scalar[DT]](act[b, j])
        var zz = (a - mu) / (std + _DIAG_EPS_STD)
        nlp += Scalar[DT](-0.5) * (
            _DIAG_LOG_2PI + Scalar[DT](2.0) * ls + zz * zz
        )
        ent += Scalar[DT](0.5) * (
            _DIAG_LOG_2PI + Scalar[DT](1.0) + Scalar[DT](2.0) * ls
        )
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


# `_ppo_ev_kernel`: single-block two-pass CleanRL explained variance over the
# minibatch — `1 − Var(ret − v) / Var(ret)` (mean-centred), or 0 when
# Var(ret) is ~0. Writes the scalar into `ev_out[0]`. Launch grid_dim=1,
# block_dim=TPB_REDUCE.
def _ppo_ev_kernel[MB: Int](
    ret: LayoutTensor[DT, Layout.row_major(MB), MutAnyOrigin],
    v: LayoutTensor[DT, Layout.row_major(MB), MutAnyOrigin],
    ev_out: LayoutTensor[DT, Layout.row_major(1), MutAnyOrigin],
):
    var t = Int(thread_idx.x)
    var s_ret: Scalar[DT] = 0.0
    var s_res: Scalar[DT] = 0.0
    var k = t
    while k < MB:
        s_ret += rebind[Scalar[DT]](ret[k])
        s_res += rebind[Scalar[DT]](ret[k]) - rebind[Scalar[DT]](v[k])
        k += TPB_REDUCE
    var tot_ret = block.sum[block_size=TPB_REDUCE, broadcast=True](val=s_ret)
    var tot_res = block.sum[block_size=TPB_REDUCE, broadcast=True](val=s_res)
    var mean_ret = tot_ret[0] / Scalar[DT](MB)
    var mean_res = tot_res[0] / Scalar[DT](MB)
    var vr: Scalar[DT] = 0.0
    var vs: Scalar[DT] = 0.0
    k = t
    while k < MB:
        var dr = rebind[Scalar[DT]](ret[k]) - mean_ret
        var rr = (rebind[Scalar[DT]](ret[k]) - rebind[Scalar[DT]](v[k])) - mean_res
        vr += dr * dr
        vs += rr * rr
        k += TPB_REDUCE
    var var_ret = block.sum[block_size=TPB_REDUCE, broadcast=False](val=vr)
    var var_res = block.sum[block_size=TPB_REDUCE, broadcast=False](val=vs)
    if t == 0:
        var ev: Scalar[DT] = 0.0
        if var_ret[0] > Scalar[DT](1e-8):
            ev = Scalar[DT](1.0) - var_res[0] / var_ret[0]
        ev_out[0] = ev


struct PPOTrainer[
    train_target: StaticString,
    ACTOR: Module,
    CRITIC: Module,
    OBS_DIM: Int,
    ACT_DIM: Int,
    ROLLOUT_LEN: Int,
    MINIBATCH: Int,
    N_EPOCHS: Int,
    N_ENVS: Int = 1,
](OnPolicyAgent & OnPolicyAgentBatched):
    """CleanRL-style PPO continuous trainer. N_ENVS defaults to 1 for
    single-env consumers (host-list select_action / record_transition
    surface); N_ENVS > 1 uses the pointer-based batched methods
    consumed by `run_onpolicy_train_batched`."""

    # OnPolicyAgentBatched trait-visible comptime aliases.
    comptime AGENT_TRAIN_TARGET = Self.train_target
    comptime AGENT_OBS_DIM      = Self.OBS_DIM
    comptime AGENT_ACT_DIM      = Self.ACT_DIM
    comptime AGENT_N_ENVS       = Self.N_ENVS

    comptime N_MINIBATCHES = (Self.ROLLOUT_LEN * Self.N_ENVS) // Self.MINIBATCH

    # Timer section indices — order matches `add_section` calls in `make`.
    comptime _T_GAE = 0
    comptime _T_UPDATE = 1
    comptime _T_DIAG = 2

    # ── Networks + optimisers ────────────────────────────────────────
    var actor: Self.ACTOR
    var critic: Self.CRITIC
    var actor_opt: Adam
    var critic_opt: Adam

    # ── Blocks ───────────────────────────────────────────────────────
    var act_step: PPOActStep[Self.OBS_DIM, Self.ACT_DIM, Self.ACTOR, Self.CRITIC]
    var record_step: PPORecordStep[Self.OBS_DIM, Self.ACT_DIM, Self.ROLLOUT_LEN]
    var gae_step: PPOGAEStep[Self.OBS_DIM, Self.ROLLOUT_LEN, Self.CRITIC]
    var gather_step: PPOMinibatchGatherStep[
        Self.OBS_DIM, Self.ACT_DIM, Self.ROLLOUT_LEN, Self.MINIBATCH,
    ]
    var actor_train: PPOActorTrainStep[
        Self.OBS_DIM, Self.ACT_DIM, Self.MINIBATCH, Self.ACTOR,
    ]
    var critic_train: PPOCriticTrainStep[
        Self.OBS_DIM, Self.MINIBATCH, Self.CRITIC,
    ]

    # ── State ────────────────────────────────────────────────────────
    var state: OnPolicyState[
        Self.OBS_DIM, Self.ACT_DIM, Self.ROLLOUT_LEN, Self.MINIBATCH,
        Self.N_ENVS,
    ]

    # Host-side staging for the N=1 host-list wrapper paths (so they
    # don't allocate per call). None until `make` allocates real buffers.
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
    var action_scale: Scalar[DT]
    var max_grad_norm: Scalar[DT]

    # ── Episode tracker (per-env running-return + completed-return window) ─
    var tracker: EpisodeTracker
    var _ep_returns: Optional[Pointer[Scalar[DT], MutUntrackedOrigin]]  # N_ENVS

    # ── Train-step accumulators (summed across all minibatch updates) ────
    var _actor_L_accum: Scalar[DT]
    var _critic_L_accum: Scalar[DT]
    # Per-minibatch diagnostics (CPU-only diag walk; averaged at flush).
    var _entropy_accum: Scalar[DT]
    var _kl_accum: Scalar[DT]
    var _clip_accum: Scalar[DT]
    var _ev_accum: Scalar[DT]
    # Diag actor forward scratch (MINIBATCH * 2 * ACT_DIM) — host on CPU,
    # device-resident on GPU (the diag kernels read its `.dev` buffer).
    var _diag_ao: Tensor
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
    var _device_update: Bool
    """True once the update has run through the CUDA-graph surface
    (`begin_update_device`): the losses then live on the device and the
    networks re-pad their weights on every forward (`capture_recast`)."""
    var _t_upd_start: Int
    var device_rollout: PPODeviceRollout
    var _device_rollout: Bool
    """True once `enable_device_rollout` ran: the rollout lives on the
    device and the update runs GAE there (`begin_update_device`)."""
    # Never reset by `flush_*` — emitted as `train_steps` so the
    # downstream monitor can plot cumulative minibatch updates.
    var _total_train_steps: Int

    var timer: Timer

    # Device context (GPU only; None on CPU). Threaded from `make` so the
    # GPU checkpoint path can stage device buffers → host.
    var ctx: Optional[DeviceContext]

    def __init__(out self):
        comptime assert (
            Self.train_target == "cpu" or Self.train_target == "gpu"
        ), "PPOTrainer: train_target must be 'cpu' or 'gpu'"
        comptime assert Self.ACTOR.IN_DIMS[0] == Self.OBS_DIM, (
            "PPOTrainer: ACTOR.IN_DIM must equal OBS_DIM"
        )
        comptime assert Self.ACTOR.OUT_DIM == 2 * Self.ACT_DIM, (
            "PPOTrainer: ACTOR.OUT_DIM must equal 2 * ACT_DIM"
        )
        comptime assert Self.CRITIC.IN_DIMS[0] == Self.OBS_DIM, (
            "PPOTrainer: CRITIC.IN_DIM must equal OBS_DIM"
        )
        comptime assert Self.CRITIC.OUT_DIM == 1, (
            "PPOTrainer: CRITIC.OUT_DIM must equal 1"
        )
        comptime assert (Self.ROLLOUT_LEN * Self.N_ENVS) % Self.MINIBATCH == 0, (
            "PPOTrainer: ROLLOUT_LEN * N_ENVS must be divisible by MINIBATCH"
        )
        comptime assert Self.N_ENVS >= 1, "PPOTrainer: N_ENVS must be >= 1"
        self.actor = Self.ACTOR()
        self.critic = Self.CRITIC()
        self.actor_opt = Adam()
        self.critic_opt = Adam()
        self.act_step = PPOActStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.ACTOR, Self.CRITIC,
        ]()
        self.record_step = PPORecordStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.ROLLOUT_LEN,
        ]()
        self.gae_step = PPOGAEStep[
            Self.OBS_DIM, Self.ROLLOUT_LEN, Self.CRITIC,
        ]()
        self.gather_step = PPOMinibatchGatherStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.ROLLOUT_LEN, Self.MINIBATCH,
        ]()
        self.actor_train = PPOActorTrainStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.MINIBATCH, Self.ACTOR,
        ]()
        self.critic_train = PPOCriticTrainStep[
            Self.OBS_DIM, Self.MINIBATCH, Self.CRITIC,
        ]()
        self.state = OnPolicyState[
            Self.OBS_DIM, Self.ACT_DIM, Self.ROLLOUT_LEN, Self.MINIBATCH,
            Self.N_ENVS,
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
        self.action_scale = Scalar[DT](1.0)
        self.max_grad_norm = Scalar[DT](0.0)
        self.tracker = EpisodeTracker.new(
            window_size=10, initial_fill=Scalar[DT](-1600.0),
        )
        self._ep_returns = None
        self._actor_L_accum = Scalar[DT](0.0)
        self._critic_L_accum = Scalar[DT](0.0)
        self._entropy_accum = Scalar[DT](0.0)
        self._kl_accum = Scalar[DT](0.0)
        self._clip_accum = Scalar[DT](0.0)
        self._ev_accum = Scalar[DT](0.0)
        self._diag_ao = Tensor()
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
        self._device_update = False
        self._t_upd_start = 0
        self.device_rollout = PPODeviceRollout()
        self._device_rollout = False
        self.timer = Timer.new()
        self.ctx = None

    @staticmethod
    def make(
        actor_lr: Scalar[DT] = Scalar[DT](3e-4),
        critic_lr: Scalar[DT] = Scalar[DT](1e-3),
        gamma: Scalar[DT] = Scalar[DT](0.99),
        gae_lambda: Scalar[DT] = Scalar[DT](0.95),
        clip_eps: Scalar[DT] = Scalar[DT](0.2),
        entropy_coef: Scalar[DT] = Scalar[DT](0.0),
        action_scale: Scalar[DT] = Scalar[DT](1.0),
        log_std_init: Scalar[DT] = Scalar[DT](-0.5),
        window_size: Int = 10,
        initial_episode_fill: Scalar[DT] = Scalar[DT](-1600.0),
        # Canonical PPO uses max_grad_norm=0.5 (Schulman 2017 + most
        # implementations). Default 0 keeps bit-identity for callers
        # that previously trained unclipped. Wired to both optimizers
        # below via the explicit `clip_grads` call in the train steps —
        # separate from `clip_eps`, which is the policy-ratio surrogate
        # clip, not the gradient-norm clip.
        max_grad_norm: Scalar[DT] = Scalar[DT](0.0),
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        comptime assert (
            Self.train_target == "cpu" or Self.train_target == "gpu"
        ), "PPOTrainer.make: train_target must be 'cpu' or 'gpu'"
        comptime if Self.train_target == "gpu":
            if not ctx:
                raise Error(
                    "PPOTrainer.make[train_target='gpu']: ctx required"
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
        t.act_step = PPOActStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.ACTOR, Self.CRITIC,
        ].make[Self.train_target](ctx=ctx)
        t.record_step = PPORecordStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.ROLLOUT_LEN,
        ].make[Self.train_target](ctx=ctx)
        t.gae_step = PPOGAEStep[
            Self.OBS_DIM, Self.ROLLOUT_LEN, Self.CRITIC,
        ].make[Self.train_target](ctx=ctx)
        t.gather_step = PPOMinibatchGatherStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.ROLLOUT_LEN, Self.MINIBATCH,
        ].make[Self.train_target](ctx=ctx)
        t.actor_train = PPOActorTrainStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.MINIBATCH, Self.ACTOR,
        ].make[Self.train_target](
            ctx=ctx, clip_eps=clip_eps, entropy_coef=entropy_coef,
        )
        t.critic_train = PPOCriticTrainStep[
            Self.OBS_DIM, Self.MINIBATCH, Self.CRITIC,
        ].make[Self.train_target](ctx=ctx)
        t.state = OnPolicyState[
            Self.OBS_DIM, Self.ACT_DIM, Self.ROLLOUT_LEN, Self.MINIBATCH,
            Self.N_ENVS,
        ].make[Self.train_target](ctx=ctx)
        t._obs1  = alloc[Scalar[DT]]({count = Self.OBS_DIM}).unsafe_leak()
        t._act1  = alloc[Scalar[DT]]({count = Self.ACT_DIM}).unsafe_leak()
        t._rew1  = alloc[Scalar[DT]]({count = 1}).unsafe_leak()
        t._done1 = alloc[Scalar[DT]]({count = 1}).unsafe_leak()
        t._nobs1 = alloc[Scalar[DT]]({count = Self.OBS_DIM}).unsafe_leak()
        var ep_returns_p = alloc[Scalar[DT]](
            {count = Self.N_ENVS}
        ).unsafe_leak()
        for e in range(Self.N_ENVS):
            ep_returns_p[unsafe_offset=e] = Scalar[DT](0.0)
        t._ep_returns = ep_returns_p

        # Diag actor-output scratch (MINIBATCH * 2 * ACT_DIM) on the train
        # target; the GPU diag kernels read its device buffer.
        t._diag_ao = Tensor.make[Self.train_target](
            Self.MINIBATCH * 2 * Self.ACT_DIM, ctx
        )

        comptime if Self.train_target == "gpu":
            # Device diag scratch (per-sample ent/kl/clip + EV scalar) + mean
            # accumulators. `_diag_ao` (above) holds the recomputed post-update
            # actor output; the per-sample kernel writes ent/kl/clip; the EV
            # kernel writes a [1] scalar.
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
        t.action_scale = action_scale
        t.max_grad_norm = max_grad_norm
        # log_std_init is the caller's responsibility (reaching into the
        # actor's GaussianHead.log_std vector — see the example for the
        # idiom). Kept here for forward-compat / docs.
        _ = log_std_init
        t.tracker = EpisodeTracker.new(
            window_size=window_size, initial_fill=initial_episode_fill,
        )

        # Timer sections — index order MUST match the `_T_*` comptime
        # constants above.
        t.timer.add_section("gae")
        t.timer.add_section("update")
        t.timer.add_section("diag")
        return t^

    # ──────────────────────────────────────────────────────────────────
    # OnPolicyAgent surface
    # ──────────────────────────────────────────────────────────────────

    def select_action(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
        step_idx: Int,
    ) raises:
        """N=1 host-list wrapper — only valid when Self.N_ENVS == 1.
        Stages obs into _obs1, delegates to `select_action_batched`
        (which is N_ENVS=Self.N_ENVS-wide), then copies _act1 out."""
        comptime assert Self.N_ENVS == 1, (
            "PPOTrainer.select_action: host-list wrapper only valid "
            "at N_ENVS=1; use select_action_batched for N_ENVS>1"
        )
        var obs_p = self._obs1.value().as_unsafe_any_origin()
        var act_p = self._act1.value().as_unsafe_any_origin()
        for d in range(Self.OBS_DIM):
            obs_p[unsafe_offset=d] = obs[d]
        self.select_action_batched(obs_p, act_p, step_idx)
        for j in range(Self.ACT_DIM):
            action_out[j] = act_p[unsafe_offset=j]

    def select_action_batched(
        mut self,
        obs_ptr: Pointer[Scalar[DT], MutAnyOrigin],
        action_ptr: Pointer[Scalar[DT], MutAnyOrigin],
        step_idx: Int,
    ) raises:
        """N_ENVS-wide action selection. Reads N_ENVS*OBS from obs_ptr,
        writes N_ENVS*ACT into action_ptr, caches per-env sample /
        log_prob / value into state for the next record."""
        _ = step_idx
        self.act_step.step[
            Self.train_target, Self.ROLLOUT_LEN, Self.MINIBATCH, Self.N_ENVS,
        ](
            self.state, self.actor, self.critic,
            obs_ptr, action_ptr, self.action_scale,
        )

    def select_greedy_action(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
    ) raises:
        """Single-env greedy eval — always BATCH=1 even when state is
        sized for N_ENVS > 1 (eval bypasses the rollout buffer)."""
        self.act_step.step_greedy_n1[
            Self.train_target, Self.ROLLOUT_LEN, Self.MINIBATCH, Self.N_ENVS,
        ](self.state, self.actor, obs, action_out, self.action_scale)

    def select_greedy_action_batched(
        mut self,
        obs_ptr: Pointer[Scalar[DT], MutAnyOrigin],
        action_ptr: Pointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        """N_ENVS-wide deterministic eval: the actor's mean, clamped. Leaves
        the rollout cache alone."""
        self.act_step.step_greedy_batched[
            Self.train_target, Self.ROLLOUT_LEN, Self.MINIBATCH, Self.N_ENVS,
        ](self.state, self.actor, obs_ptr, action_ptr, self.action_scale)

    def record_transition(
        mut self,
        ref obs: List[Scalar[DT]],
        ref action: List[Scalar[DT]],
        reward: Scalar[DT],
        ref next_obs: List[Scalar[DT]],
        done: Scalar[DT],
    ) raises:
        """N=1 host-list wrapper. Only valid when Self.N_ENVS == 1.
        Bypasses `record_batch_cpu` to keep the legacy tracker pattern
        (per-step add_reward + driver-driven end_episode) and stay
        bit-identical to the pre-N_ENVS PPOTrainer at single-env."""
        comptime assert Self.N_ENVS == 1, (
            "PPOTrainer.record_transition: host-list wrapper only "
            "valid at N_ENVS=1; use record_batch_cpu for N_ENVS>1"
        )
        _ = action  # env-ready action ignored (cached unbounded used)
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
        ](
            self.state, obs_p, rew_p, nobs_p, done_p,
        )
        self.tracker.add_reward(reward)

    def record_batch_cpu(
        mut self,
        obs_ptr: Pointer[Scalar[DT], MutAnyOrigin],
        reward_ptr: Pointer[Scalar[DT], MutAnyOrigin],
        next_obs_ptr: Pointer[Scalar[DT], MutAnyOrigin],
        done_ptr: Pointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        """N_ENVS-wide transition record. Maintains a per-env running
        return sum (_ep_returns[e]); when done[e] is set, pushes the
        completed return into the EpisodeTracker via the same
        add_reward + end_episode pattern used by the N=1 wrapper."""
        self.record_step.step[
            Self.train_target, Self.MINIBATCH, Self.N_ENVS,
        ](self.state, obs_ptr, reward_ptr, next_obs_ptr, done_ptr)
        var ep_ret_p = self._ep_returns.value().as_unsafe_any_origin()
        for e in range(Self.N_ENVS):
            ep_ret_p[unsafe_offset=e] += reward_ptr[unsafe_offset=e]
            if done_ptr[unsafe_offset=e] > Scalar[DT](0.5):
                # Push a single completed-episode return into the tracker
                # window using its add_reward + end_episode contract.
                self.tracker.add_reward(ep_ret_p[unsafe_offset=e])
                self.tracker.end_episode()
                ep_ret_p[unsafe_offset=e] = Scalar[DT](0.0)

    def mark_terminal(mut self) raises:
        """N=1 host-list wrapper — env 0 terminal."""
        comptime assert Self.N_ENVS == 1, (
            "PPOTrainer.mark_terminal: host-list wrapper only valid "
            "at N_ENVS=1; pass env_idx via mark_terminal_env"
        )
        self.mark_terminal_env(0)

    def mark_terminal_env(mut self, env_idx: Int) raises:
        self.record_step.mark_terminal[
            Self.train_target, Self.MINIBATCH, Self.N_ENVS,
        ](self.state, env_idx)

    def end_episode(mut self):
        self.tracker.end_episode()

    def train_step(mut self, step_idx: Int) raises -> Bool:
        _ = step_idx
        if self.state.rollout_idx < Self.ROLLOUT_LEN:
            return False

        # ── GAE: bootstrap V(s_T) per env + per-env backward pass.
        var t_gae = perf_counter_ns()
        self.gae_step.step[
            Self.train_target, Self.ACT_DIM, Self.MINIBATCH, Self.N_ENVS,
        ](self.state, self.critic, self.gamma, self.gae_lambda)
        self.timer.accumulate(Self._T_GAE, t_gae)

        # ── K-epoch minibatch SGD. Reset indices ONCE per rollout
        # (matches legacy ordering for bit-identity); epoch shuffles
        # operate on whatever state the previous epoch left behind.
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
                    Self.train_target, Self.ACT_DIM, Self.ROLLOUT_LEN,
                    Self.N_ENVS,
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

        # ── Reset rollout cursor + clear term buf.
        self.record_step.reset_rollout[
            Self.train_target, Self.MINIBATCH, Self.N_ENVS,
        ](self.state)
        return True

    def _accumulate_diag(mut self) raises:
        """CPU-only per-minibatch PPO diagnostics: entropy, Schulman-2020
        approx_kl, clip_fraction, and explained variance. Re-runs the actor
        on the current minibatch obs (post-update policy) and reads the
        critic's value/return scratches written by the critic train step.
        Accumulated once per minibatch; averaged by `_update_count` at flush
        (same denominator as the loss accumulators)."""
        comptime ACT = Self.ACT_DIM
        comptime MB = Self.MINIBATCH

        # Re-run the (post-update) actor on the gathered minibatch obs (host)
        # into the diag scratch Tensor.
        call_forward["cpu", MB](
            self.actor, TensorRefs[Self.ACTOR.ARITY](self.state.mb_obs), self._diag_ao, self.ctx
        )

        ref ao = self._diag_ao.data
        ref act = self.state.mb_act.data
        ref olp = self.state.mb_olp.data
        ref v = self.state.mb_v.data
        ref ret = self.state.mb_ret.data

        var ent_sum = Scalar[DT](0.0)
        var kl_sum = Scalar[DT](0.0)
        var clip_sum = Scalar[DT](0.0)
        for b in range(MB):
            var nlp = Scalar[DT](0.0)
            var ent = Scalar[DT](0.0)
            for j in range(ACT):
                var mu = ao[b * 2 * ACT + j]
                var ls = ao[b * 2 * ACT + ACT + j]
                if ls < _DIAG_LOG_STD_MIN:
                    ls = _DIAG_LOG_STD_MIN
                elif ls > _DIAG_LOG_STD_MAX:
                    ls = _DIAG_LOG_STD_MAX
                var std = fexp(ls)
                var a = act[b * ACT + j]
                var zz = (a - mu) / (std + _DIAG_EPS_STD)
                nlp += Scalar[DT](-0.5) * (
                    _DIAG_LOG_2PI + Scalar[DT](2.0) * ls + zz * zz
                )
                ent += Scalar[DT](0.5) * (
                    _DIAG_LOG_2PI + Scalar[DT](1.0) + Scalar[DT](2.0) * ls
                )
            var diff = nlp - olp[b]
            if diff > _DIAG_LOG_PROB_DIFF_MAX:
                diff = _DIAG_LOG_PROB_DIFF_MAX
            elif diff < -_DIAG_LOG_PROB_DIFF_MAX:
                diff = -_DIAG_LOG_PROB_DIFF_MAX
            var ratio = fexp(diff)
            # Schulman 2020 unbiased KL estimate: (r - 1) - log r.
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

        # Explained variance (CleanRL): 1 - Var(ret - v) / Var(ret), with
        # mean-centred variances. v_p holds the critic's pre-update preds
        # for this minibatch.
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
        the current minibatch obs on device, then derive entropy / KL / clip /
        explained-variance via the diag kernels and fold them into the
        device-resident running means. No per-minibatch D2H — read at flush."""
        comptime ACT = Self.ACT_DIM
        comptime MB = Self.MINIBATCH
        var ctx = self.ctx.value()

        # Recompute the actor output on the gathered minibatch obs (device),
        # into the device-resident `_diag_ao` Tensor.
        call_forward["gpu", MB](
            self.actor, TensorRefs[Self.ACTOR.ARITY](self.state.mb_obs), self._diag_ao, self.ctx
        )

        # Device views (`.lt`) for the diag kernels — no raw pointers.
        comptime n_blocks = (MB + TPB - 1) // TPB
        comptime per_k = _ppo_diag_per_sample_kernel[MB, ACT]
        ctx.enqueue_function[per_k](
            self._diag_ao.lt["gpu", Layout.row_major(MB, 2 * ACT)](),
            self.state.mb_act.lt["gpu", Layout.row_major(MB, ACT)](),
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
        # The EV scalar lives in a [1] buffer; reducing it folds (ev, +1).
        self._ev_mean_dev.accumulate_gpu_lt[1](
            self._diag_ev.lt["gpu", Layout.row_major(1)]()
        )

    # ──────────────────────────────────────────────────────────────────
    # CUDA-graph update surface (`USE_TRAIN_CUDA_GRAPH`, GPU only) — the
    # `train_step` update split so the driver can capture one minibatch step
    # and replay it. Same rollout gate, GAE, index resets and host shuffles
    # (same RNG stream) as `train_step`; the minibatch work runs on device.
    # ──────────────────────────────────────────────────────────────────

    def begin_update_device(mut self, step_idx: Int) raises -> Bool:
        _ = step_idx
        comptime if Self.train_target != "gpu":
            raise Error("begin_update_device: the CUDA-graph update is GPU-only")
        if self.state.rollout_idx < Self.ROLLOUT_LEN:
            return False
        if not self._device_update:
            # Under replay the optimizer's host version bump never runs, so the
            # Linear weight caches must re-pad on EVERY forward (also the
            # rollout's eager ones, which would otherwise keep the pad of the
            # captured step's pre-update weights). Set once, kept on.
            self.actor.set_attr["capture_recast"](Scalar[DT](1.0))
            self.critic.set_attr["capture_recast"](Scalar[DT](1.0))
            self._device_update = True

        var t_gae = perf_counter_ns()
        if self._device_rollout:
            # The pool is on the device: V of every row's next obs, GAE there,
            # and the advantages down once for the host shuffle's
            # normalisation.
            call_forward["gpu", Self.ROLLOUT_LEN * Self.N_ENVS](
                self.critic,
                TensorRefs[Self.CRITIC.ARITY](self.state.next_obs_buf),
                self.state.next_val_buf,
                self.ctx,
            )
            self.device_rollout.gae(self.state, self.gamma, self.gae_lambda)
            self.state.adv_buf.download(self.ctx.value())
        else:
            self.gae_step.step[
                Self.train_target, Self.ACT_DIM, Self.MINIBATCH, Self.N_ENVS,
            ](self.state, self.critic, self.gamma, self.gae_lambda)
        self.timer.accumulate(Self._T_GAE, t_gae)

        self._t_upd_start = perf_counter_ns()
        self.gather_step.reset_indices[Self.train_target, Self.N_ENVS](
            self.state
        )
        for epoch in range(Self.N_EPOCHS):
            self.gather_step.shuffle_epoch[Self.train_target, Self.N_ENVS](
                self.state
            )
            self.gather_step.stage_epoch_indices[Self.N_EPOCHS, Self.N_ENVS](
                self.state, epoch
            )
        self.gather_step.upload_device_update[Self.N_EPOCHS, Self.N_ENVS](
            self.state, pool_on_device=self._device_rollout
        )
        return True

    def minibatches_per_update(self) -> Int:
        return Self.N_EPOCHS * Self.N_MINIBATCHES

    def train_minibatch_device(mut self) raises:
        """The captured step: gather + normalise, actor and critic updates
        with device losses and device clipping, the device diagnostics, and
        the minibatch counter. No host work, no sync."""
        comptime if Self.train_target != "gpu":
            raise Error("train_minibatch_device: GPU-only")
        else:
            self.gather_step.gather_device[Self.N_EPOCHS, Self.N_ENVS](
                self.state
            )
            _ = self.actor_train.step[
                Self.train_target, Self.ROLLOUT_LEN, Self.N_ENVS,
                DEVICE_LOSS=True,
            ](self.state, self.actor, self.actor_opt, self.max_grad_norm)
            _ = self.critic_train.step[
                Self.train_target, Self.ACT_DIM, Self.ROLLOUT_LEN, Self.N_ENVS,
                DEVICE_LOSS=True,
            ](self.state, self.critic, self.critic_opt, self.max_grad_norm)
            self._accumulate_diag_gpu()
            self.gather_step.advance_counter[Self.N_ENVS](self.state)

    def note_minibatch_update(mut self):
        self._update_count += 1
        self._total_train_steps += 1

    def end_update_device(mut self) raises:
        # One sync per update so the "update" timer means what it does on the
        # eager path (which synchronises every minibatch anyway).
        self.ctx.value().synchronize()
        self.timer.accumulate(Self._T_UPDATE, self._t_upd_start)
        if self._device_rollout:
            self.device_rollout.reset_rollout(self.state)
        else:
            self.record_step.reset_rollout[
                Self.train_target, Self.MINIBATCH, Self.N_ENVS,
            ](self.state)

    # ──────────────────────────────────────────────────────────────────
    # Device-resident rollout (`DEVICE_ROLLOUT`, GPU only) — the act /
    # record steps as kernels over the env's device buffers; see
    # `ppo/blocks/device_rollout.mojo`.
    # ──────────────────────────────────────────────────────────────────

    def enable_device_rollout(mut self, seed: UInt64) raises:
        comptime if Self.train_target != "gpu":
            raise Error("enable_device_rollout: GPU-only")
        if not self.device_rollout.ready():
            self.device_rollout.setup(self.ctx.value(), seed)
        self._device_rollout = True

    def select_action_device(
        mut self,
        obs_ptr: Pointer[Scalar[DT], MutAnyOrigin],
        action_ptr: Pointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        comptime if Self.train_target != "gpu":
            raise Error("select_action_device: GPU-only")
        else:
            self.device_rollout.copy_obs(self.state, obs_ptr)
            call_forward["gpu", Self.N_ENVS](
                self.actor,
                TensorRefs[Self.ACTOR.ARITY](self.state.ob1),
                self.state.ao1,
                self.ctx,
            )
            call_forward["gpu", Self.N_ENVS](
                self.critic,
                TensorRefs[Self.CRITIC.ARITY](self.state.ob1),
                self.state.v1,
                self.ctx,
            )
            self.device_rollout.sample_continuous(
                self.state, action_ptr, self.action_scale
            )

    def record_device(
        mut self,
        reward_ptr: Pointer[Scalar[DT], MutAnyOrigin],
        next_obs_ptr: Pointer[Scalar[DT], MutAnyOrigin],
        done_ptr: Pointer[Scalar[DT], MutAnyOrigin],
        terminated_ptr: Pointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        comptime if Self.train_target != "gpu":
            raise Error("record_device: GPU-only")
        else:
            self.device_rollout.record(
                self.state, reward_ptr, next_obs_ptr, done_ptr, terminated_ptr
            )

    def add_episode_return(mut self, ret: Scalar[DT]):
        self.tracker.add_complete_return(ret)

    def mean_return(self) -> Scalar[DT]:
        return self.tracker.mean_return()

    def ep_count(self) -> Int:
        return self.tracker.ep_count

    # ─── Logging surface (parity with SACTrainer) ────────────────────────

    def flush_train_log(
        mut self,
    ) -> Tuple[Scalar[DT], Scalar[DT], Int]:
        """Return (mean_actor_loss, mean_critic_loss, n_updates) since
        last flush. `n_updates` counts minibatch updates across the
        K-epoch SGD inside one train_step. Resets accumulators."""
        var n = self._update_count if self._update_count > 0 else 1
        var inv = Scalar[DT](1.0) / Scalar[DT](n)
        var out = (
            self._actor_L_accum * inv,
            self._critic_L_accum * inv,
            self._update_count,
        )
        self._actor_L_accum = Scalar[DT](0.0)
        self._critic_L_accum = Scalar[DT](0.0)
        # Reset diag accumulators in lock-step (mirrors flush_metrics) so a
        # flush_train_log call doesn't leave them to double-count.
        self._entropy_accum = Scalar[DT](0.0)
        self._kl_accum = Scalar[DT](0.0)
        self._clip_accum = Scalar[DT](0.0)
        self._ev_accum = Scalar[DT](0.0)
        self._update_count = 0
        return out

    def total_train_steps(self) -> Int:
        """Cumulative minibatch updates since trainer was made. Not reset
        by `flush_*`."""
        return self._total_train_steps

    def flush_metrics[
        L: Logger = NoOpLogger
    ](
        mut self,
        logger: Optional[Pointer[L, MutAnyOrigin]] = None,
        step: Int = 0,
    ) raises -> PPOMetrics:
        """Drain accumulators into a PPOMetrics bundle. If a logger
        pointer is wired, also emit one log_scalar per metric field.
        Resets per-chunk accumulators on every call; the cumulative
        `_total_train_steps` counter is NOT reset."""
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
        var policy_loss = self._actor_L_accum * inv
        var critic_loss = self._critic_L_accum * inv
        comptime if Self.train_target == "gpu":
            if self._device_update:
                # The CUDA-graph update keeps its losses on the device.
                policy_loss = self.actor_train.inner.read_device_loss()
                critic_loss = self.critic_train.read_device_loss(self.ctx)
                self.actor_train.inner.reset_device_loss()
                self.critic_train.reset_device_loss()
        var bundle = PPOMetrics(
            policy_loss=LogScalar[DT](policy_loss),
            critic_loss=LogScalar[DT](critic_loss),
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

    # ─── Trait-uniform cadence hooks (consumed by the driver) ─────────

    def flush_metrics_through_logger[L: Logger](
        mut self,
        logger: Optional[Pointer[L, MutAnyOrigin]],
        step: Int,
    ) raises:
        """Trait-uniform passthrough: drains the PPO metric accumulators
        through `flush_metrics` and discards the typed bundle. The
        driver calls this at the user's `diag_every` cadence so no
        chunking is needed."""
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
        `default` if absent). The `key` ends with '=' and the value has no
        '=', so split on '=' and take the tail (nightly String has no
        positional slicing)."""
        var lines = _split_lines(content)
        for i in range(len(lines)):
            if lines[i].startswith(key):
                var parts = lines[i].split("=")
                if len(parts) >= 2:
                    return Scalar[DT](atof(parts[len(parts) - 1]))
        return default

    def flush_timer_log(mut self) -> String:
        """Per-section wall-time report (one line per sub-step:
        gae / update / diag) and reset the accumulators. PPO's train_step only
        fires at rollout-length boundaries, so per-section costs are
        amortised across many env steps."""
        var report = self.timer.format_report()
        self.timer.reset()
        return report
