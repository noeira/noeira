"""SACTrainer — storage-framework SAC trainer (CPU gate; GPU stretch).

Assembles the migrated `noeira.nn` SAC blocks into a single
driver-conforming `SACTrainer` that conforms `OffPolicyAgentGpu` so both
the CPU single-env driver (`run_offpolicy_train`) and the batched drivers
type-check. Both the CPU and GPU paths are wired; the GPU per-update hot
loop is sync-free (loss / α / critic / diagnostics device-resident, read
only at flush), so it supports CUDA-graph capture via
`train_device_kernels` / `note_train_update` / `learning_starts_count`.

Pipeline (per train step, mirrors the proven convergence test):
  state.step_idx = step; state.alpha = exp(alpha_opt.value)
  sample_blk.step(state)                       # fills state.mb_*
  target_y_blk.step(state, actor, tgt1, tgt2)  # writes state.mb_y
  twin_critic_blk.step(state, c1, c1_opt, c2, c2_opt)
  out = actor_loss.forward_backward(actor, actor_opt, c1, c2, mb_s, alpha)
  state.log_prob_mean = out.log_prob_mean
  alpha_blk.step(state, alpha_opt)
  polyak_blk.step(state, pair1, pair2)

α is a HOST scalar on CPU (`state.alpha = exp(alpha_opt.value)`).

Dimensions (OBS / ACT / BATCH) derive from `SAMPLE` so they're specified
once (on the sample block type).
"""

from noeira.nn.core.param import walk_params, ParamVisitorRef
from std.math import exp as fexp, log as flog, tanh as ftanh
from std.random import random_float64
from std.random.philox import Random as PhiloxRandom
from layout import Layout, LayoutTensor

from max.gpu import global_idx
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.core.logger import Logger, NoOpLogger
from noeira.nn.constants import DT, TPB
from noeira.nn.core.module import Module
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.call import call_forward
from noeira.nn.core.initializer import Xavier, Zero
from noeira.nn.primitives.rsample import RSample
from noeira.nn.optimizer.adam import Adam
from noeira.nn.optimizer.scalar_adam import ScalarAdam
from noeira.nn.core.checkpoint import (
    CheckpointReader,
    CheckpointScalars,
    _is_v3_header,
    _read_file_bytes,
    _split_lines,
    load_params_multi,
    save_params_multi,
)

from noeira.nn.core.log_bundle import log_bundle
from noeira.nn.core.metric import LogScalar

from ..data.n_step_replay import GPUNStepBuffer
from ..core.online_target_pair import OnlineTargetPair
from ..training.episode_tracker import EpisodeTracker
from ..training.device_mean_accum import DeviceMeanAccum
from ..training.trainer_block import TrainerState
from ..training.driver_offpolicy import OffPolicyAgentGpu
from ..training.blocks import SampleBlock, TwinCriticStep, PolyakStep
from ..training.blocks.action_select import (
    select_squashed_batched,
    warmup_uniform_batched,
)
from .target_y_block import TargetYBlock
from .actor_loss import SACActorLoss
from .blocks.alpha_update_step import AlphaUpdateStep
from .metrics import SACMetrics


# ──────────────────────────────────────────────────────────────────────
# GPU device kernels for the batched action-selection path. Mirror the
# DDPG trainer's device body (Philox warmup + copy/clamp) adapted to the
# storage actor/rsample surface (which take owned Tensors, so obs is
# COPIED into the trainer's device scratch before the actor forward).
# ──────────────────────────────────────────────────────────────────────


struct SACTrainer[
    train_target: StaticString,
    SAMPLE: SampleBlock,
    ACTOR: Module,
    CRITIC: Module,
](OffPolicyAgentGpu):
    """Storage-framework SAC trainer. Dimensions (OBS / ACT / BATCH) are
    derived from SAMPLE so the user specifies them once on the sample
    block type — symbolic-equality follows for the TrainerState the
    trainer holds vs the one SAMPLE.step expects."""

    comptime OBS_DIM: Int = Self.SAMPLE.OBS
    comptime ACT_DIM: Int = Self.SAMPLE.ACT
    comptime BATCH: Int = Self.SAMPLE.BATCH

    comptime AGENT_OBS_DIM: Int = Self.OBS_DIM
    comptime AGENT_ACT_DIM: Int = Self.ACT_DIM
    comptime AGENT_TRAIN_TARGET: StaticString = Self.train_target

    var actor: Self.ACTOR
    var pair1: OnlineTargetPair[Self.CRITIC]
    var pair2: OnlineTargetPair[Self.CRITIC]
    var actor_opt: Adam
    var critic1_opt: Adam
    var critic2_opt: Adam
    var alpha_opt: ScalarAdam

    var sample_blk: Self.SAMPLE
    var target_y_blk: TargetYBlock[
        Self.ACTOR, Self.CRITIC, Self.BATCH, Self.OBS_DIM, Self.ACT_DIM,
    ]
    var twin_critic_blk: TwinCriticStep[
        Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.CRITIC,
    ]
    var actor_loss_blk: SACActorLoss[Self.ACTOR, Self.CRITIC, Self.BATCH]
    var alpha_blk: AlphaUpdateStep[Self.OBS_DIM, Self.ACT_DIM, Self.BATCH]
    var polyak_blk: PolyakStep[
        Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.CRITIC,
    ]

    # select-action rsample (separate from the loss graphs' own rsamples).
    var sel: RSample[Self.ACT_DIM]

    var state: TrainerState[Self.OBS_DIM, Self.ACT_DIM, Self.BATCH]
    var tracker: EpisodeTracker
    var ctx: Optional[DeviceContext]

    # Owned action-selection scratch Tensors (lazily `.ensure`d per call).
    var _ob_scr: Tensor   # N_ENVS * OBS
    var _ao_scr: Tensor   # N_ENVS * 2*ACT
    var _alp_scr: Tensor  # N_ENVS * (ACT + 1)

    var action_scale: Scalar[DT]
    var learning_starts: Int

    # Philox state for the GPU batched warmup kernel (gpu path only).
    var _warmup_rng_seed: UInt64
    var _warmup_rng_offset: UInt64

    # Host metric accumulators (CPU path; simple scalars like the test).
    var _actor_L_accum: Scalar[DT]
    var _critic_L_accum: Scalar[DT]
    var _alpha_accum: Scalar[DT]
    # Diagnostic accumulators (CPU path): batch means drained at flush_metrics.
    var _mean_q_accum: Scalar[DT]            # online Q1(s, a) over the batch
    # `set_bc_q_ratio`: after every `flush_metrics` the BC weight becomes
    # max(bc_weight_floor, bc_q_ratio · mean|Q|) — TD3+BC's normalisation,
    # applied to λ instead of to Q. 0 = fixed λ.
    var bc_q_ratio: Scalar[DT]
    var bc_weight_floor: Scalar[DT]
    var _mean_target_accum: Scalar[DT]       # Bellman target y
    var _mean_reward_accum: Scalar[DT]       # batch reward
    var _mean_next_q_accum: Scalar[DT]       # min(Q1_t,Q2_t)(s',a') bootstrap
    var _mean_done_accum: Scalar[DT]         # batch done
    var _mean_abs_action_accum: Scalar[DT]   # mean |action|
    # Diagnostic accumulators (GPU path): device-resident running means folded
    # in by a tiny reduction kernel EACH update (no per-step D2H — capture-safe)
    # and read with ONE D2H per `diag_every` flush. Mirrors MBPO/DQN.
    var _mean_q_dev: DeviceMeanAccum
    var _mean_target_dev: DeviceMeanAccum
    var _mean_reward_dev: DeviceMeanAccum
    var _mean_next_q_dev: DeviceMeanAccum
    var _mean_done_dev: DeviceMeanAccum
    var _mean_abs_action_dev: DeviceMeanAccum
    var _update_count: Int
    var _total_train_steps: Int

    def __init__(out self):
        self.actor = Self.ACTOR()
        self.pair1 = OnlineTargetPair[Self.CRITIC]()
        self.pair2 = OnlineTargetPair[Self.CRITIC]()
        self.actor_opt = Adam(lr=Scalar[DT](3e-4))
        self.critic1_opt = Adam(lr=Scalar[DT](1e-3))
        self.critic2_opt = Adam(lr=Scalar[DT](1e-3))
        self.alpha_opt = ScalarAdam.new(flog(Scalar[DT](0.2)), Scalar[DT](3e-4))
        self.sample_blk = Self.SAMPLE()
        self.target_y_blk = TargetYBlock[
            Self.ACTOR, Self.CRITIC, Self.BATCH, Self.OBS_DIM, Self.ACT_DIM,
        ]()
        self.twin_critic_blk = TwinCriticStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.CRITIC,
        ]()
        self.actor_loss_blk = SACActorLoss[
            Self.ACTOR, Self.CRITIC, Self.BATCH
        ]()
        self.alpha_blk = AlphaUpdateStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH
        ]()
        self.polyak_blk = PolyakStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.CRITIC,
        ]()
        self.sel = RSample[Self.ACT_DIM]()
        self.state = TrainerState[Self.OBS_DIM, Self.ACT_DIM, Self.BATCH]()
        self.tracker = EpisodeTracker(
            window=List[Scalar[DT]](),
            window_size=0,
            idx=0,
            current_return=Scalar[DT](0.0),
            ep_count=0,
        )
        self.ctx = None
        self._ob_scr = Tensor()
        self._ao_scr = Tensor()
        self._alp_scr = Tensor()
        self.action_scale = Scalar[DT](1.0)
        self.learning_starts = 1_000
        self._warmup_rng_seed = UInt64(0x5AC_C0FFEE)
        self._warmup_rng_offset = UInt64(0)
        self._actor_L_accum = Scalar[DT](0.0)
        self._critic_L_accum = Scalar[DT](0.0)
        self._alpha_accum = Scalar[DT](0.0)
        self._mean_q_accum = Scalar[DT](0.0)
        self.bc_q_ratio = Scalar[DT](0.0)
        self.bc_weight_floor = Scalar[DT](0.0)
        self._mean_target_accum = Scalar[DT](0.0)
        self._mean_reward_accum = Scalar[DT](0.0)
        self._mean_next_q_accum = Scalar[DT](0.0)
        self._mean_done_accum = Scalar[DT](0.0)
        self._mean_abs_action_accum = Scalar[DT](0.0)
        self._mean_q_dev = DeviceMeanAccum()
        self._mean_target_dev = DeviceMeanAccum()
        self._mean_reward_dev = DeviceMeanAccum()
        self._mean_next_q_dev = DeviceMeanAccum()
        self._mean_done_dev = DeviceMeanAccum()
        self._mean_abs_action_dev = DeviceMeanAccum()
        self._update_count = 0
        self._total_train_steps = 0

    def __init__(
        out self,
        ctx: Optional[DeviceContext] = None,
        actor_lr: Scalar[DT] = Scalar[DT](3e-4),
        critic_lr: Scalar[DT] = Scalar[DT](1e-3),
        alpha_lr: Scalar[DT] = Scalar[DT](3e-4),
        gamma: Scalar[DT] = Scalar[DT](0.99),
        tau: Scalar[DT] = Scalar[DT](0.005),
        action_scale: Scalar[DT] = Scalar[DT](1.0),
        init_alpha: Scalar[DT] = Scalar[DT](0.2),
        target_entropy: Scalar[DT] = Scalar[DT](-1.0),
        learning_starts: Int = 1_000,
        window_size: Int = 10,
        initial_episode_fill: Scalar[DT] = Scalar[DT](-1250.0),
        max_grad_norm: Scalar[DT] = Scalar[DT](0.0),
        per_alpha: Scalar[DT] = Scalar[DT](0.6),
        per_beta: Scalar[DT] = Scalar[DT](0.4),
        per_epsilon: Scalar[DT] = Scalar[DT](1e-6),
        use_bf16: Bool = False,
        use_ere: Bool = False,
        ere_eta: Scalar[DT] = Scalar[DT](0.996),
        ere_c_min: Int = 1,
        ere_k_max: Int = 1000,
    ) raises:
        """Unified factory. PER args applied via the SampleBlock trait's
        `configure_per` (no-op for uniform blocks). `ctx` required for
        train_target='gpu'."""
        comptime assert (
            Self.train_target == "cpu" or Self.train_target == "gpu"
        ), "SACTrainer: target must be 'cpu' or 'gpu'"
        comptime if Self.train_target == "gpu":
            if not ctx:
                raise Error("SACTrainer.make[target='gpu']: ctx required")

        # Fields `make` never sets keep their defaults; the rest are built
        # in place below (docs/COMPILE_TIME_PROFILING.md §3.2: no
        # default-construct-then-move chain).
        self.alpha_opt = ScalarAdam.new(flog(Scalar[DT](0.2)), Scalar[DT](3e-4))
        self.sample_blk = Self.SAMPLE()
        self._ob_scr = Tensor()
        self._ao_scr = Tensor()
        self._alp_scr = Tensor()
        self._warmup_rng_seed = UInt64(0x5AC_C0FFEE)
        self._warmup_rng_offset = UInt64(0)
        self._actor_L_accum = Scalar[DT](0.0)
        self._critic_L_accum = Scalar[DT](0.0)
        self._alpha_accum = Scalar[DT](0.0)
        self._mean_q_accum = Scalar[DT](0.0)
        self.bc_q_ratio = Scalar[DT](0.0)
        self.bc_weight_floor = Scalar[DT](0.0)
        self._mean_target_accum = Scalar[DT](0.0)
        self._mean_reward_accum = Scalar[DT](0.0)
        self._mean_next_q_accum = Scalar[DT](0.0)
        self._mean_done_accum = Scalar[DT](0.0)
        self._mean_abs_action_accum = Scalar[DT](0.0)
        self._mean_q_dev = DeviceMeanAccum()
        self._mean_target_dev = DeviceMeanAccum()
        self._mean_reward_dev = DeviceMeanAccum()
        self._mean_next_q_dev = DeviceMeanAccum()
        self._mean_done_dev = DeviceMeanAccum()
        self._mean_abs_action_dev = DeviceMeanAccum()
        self._update_count = 0
        self._total_train_steps = 0

        self.ctx = ctx

        self.actor = Self.ACTOR.make[Self.train_target, Xavier](ctx)
        self.pair1 = OnlineTargetPair[Self.CRITIC].make[
            Self.train_target, Xavier
        ](ctx)
        self.pair2 = OnlineTargetPair[Self.CRITIC].make[
            Self.train_target, Xavier
        ](ctx)

        self.actor_opt = Adam(lr=actor_lr)
        self.critic1_opt = Adam(lr=critic_lr)
        self.critic2_opt = Adam(lr=critic_lr)
        comptime if Self.train_target == "gpu":
            self.actor_opt.adopt[Self.train_target, Self.ACTOR](self.actor, ctx)
            self.critic1_opt.adopt[Self.train_target, Self.CRITIC](
                self.pair1.online, ctx
            )
            self.critic2_opt.adopt[Self.train_target, Self.CRITIC](
                self.pair2.online, ctx
            )

        self.target_y_blk = TargetYBlock[
            Self.ACTOR, Self.CRITIC, Self.BATCH, Self.OBS_DIM, Self.ACT_DIM,
        ].make[Self.train_target](
            action_scale=action_scale, gamma=gamma, ctx=ctx
        )
        self.twin_critic_blk = TwinCriticStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.CRITIC,
        ].make[Self.train_target](ctx=ctx)
        self.actor_loss_blk = SACActorLoss[
            Self.ACTOR, Self.CRITIC, Self.BATCH
        ].make[Self.train_target](ctx=ctx, action_scale=action_scale)
        self.alpha_blk = AlphaUpdateStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH
        ].make(target_entropy=target_entropy)
        self.polyak_blk = PolyakStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.CRITIC,
        ].make(tau=tau)

        # On GPU the entropy temperature lives in a device buffer updated by a
        # 1-thread kernel; on CPU it stays a host scalar (bit-identity path).
        comptime if Self.train_target == "gpu":
            self.alpha_opt = ScalarAdam.new_device(
                ctx.value(), flog(init_alpha), alpha_lr
            )
            # One-time wiring of the device α buffer into both Scale-consuming
            # blocks (target-y soft-V and actor-loss α·log_prob). After this
            # neither block bakes α as a per-step host scalar; both read it
            # on-device, and the device ScalarAdam refreshes it each step.
            # target-y reads α via a raw GPU-ABI kernel arg (its own kernel);
            # the actor-loss Scale node takes a type-safe device sub-buffer.
            self.target_y_blk.set_alpha_ptr(self.alpha_opt.alpha_dev_ptr())
            self.actor_loss_blk.set_alpha_buf(self.alpha_opt.alpha_dev_buffer())
        else:
            self.alpha_opt = ScalarAdam.new(flog(init_alpha), alpha_lr)

        self.sel = RSample[Self.ACT_DIM].make[Self.train_target, Zero](ctx)
        self.sel.action_scale = action_scale

        self.state = TrainerState[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH
        ].make[Self.train_target](ctx=ctx)

        self.tracker = EpisodeTracker.new(
            window_size=window_size, initial_fill=initial_episode_fill
        )

        self.action_scale = action_scale
        self.learning_starts = learning_starts

        # PER / ERE wiring: no-op default for uniform blocks.
        self.sample_blk.configure_per(
            alpha=per_alpha, beta=per_beta, epsilon=per_epsilon
        )
        self.sample_blk.setup(learning_starts, ctx=ctx)
        self.sample_blk.configure_ere(
            enable=use_ere, eta=ere_eta, c_min=ere_c_min, k_max=ere_k_max
        )

        # Pre-size the action scratch for the single-env path.
        comptime if Self.train_target == "cpu":
            self._ob_scr.ensure(Self.OBS_DIM)
            self._ao_scr.ensure(2 * Self.ACT_DIM)
            self._alp_scr.ensure(Self.ACT_DIM + 1)
        else:
            # Device-resident diagnostic accumulators (no per-step D2H).
            self._mean_q_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)
            self._mean_target_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)
            self._mean_reward_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)
            self._mean_next_q_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)
            self._mean_done_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)
            self._mean_abs_action_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)


    @staticmethod
    def make(
        ctx: Optional[DeviceContext] = None,
        actor_lr: Scalar[DT] = Scalar[DT](3e-4),
        critic_lr: Scalar[DT] = Scalar[DT](1e-3),
        alpha_lr: Scalar[DT] = Scalar[DT](3e-4),
        gamma: Scalar[DT] = Scalar[DT](0.99),
        tau: Scalar[DT] = Scalar[DT](0.005),
        action_scale: Scalar[DT] = Scalar[DT](1.0),
        init_alpha: Scalar[DT] = Scalar[DT](0.2),
        target_entropy: Scalar[DT] = Scalar[DT](-1.0),
        learning_starts: Int = 1_000,
        window_size: Int = 10,
        initial_episode_fill: Scalar[DT] = Scalar[DT](-1250.0),
        max_grad_norm: Scalar[DT] = Scalar[DT](0.0),
        per_alpha: Scalar[DT] = Scalar[DT](0.6),
        per_beta: Scalar[DT] = Scalar[DT](0.4),
        per_epsilon: Scalar[DT] = Scalar[DT](1e-6),
        use_bf16: Bool = False,
        use_ere: Bool = False,
        ere_eta: Scalar[DT] = Scalar[DT](0.996),
        ere_c_min: Int = 1,
        ere_k_max: Int = 1000,
    ) raises -> Self:
        """Unified factory. PER args applied via the SampleBlock trait's
        `configure_per` (no-op for uniform blocks). `ctx` required for
        train_target='gpu'."""
        return Self(
            ctx=ctx,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            alpha_lr=alpha_lr,
            gamma=gamma,
            tau=tau,
            action_scale=action_scale,
            init_alpha=init_alpha,
            target_entropy=target_entropy,
            learning_starts=learning_starts,
            window_size=window_size,
            initial_episode_fill=initial_episode_fill,
            max_grad_norm=max_grad_norm,
            per_alpha=per_alpha,
            per_beta=per_beta,
            per_epsilon=per_epsilon,
            use_bf16=use_bf16,
            use_ere=use_ere,
            ere_eta=ere_eta,
            ere_c_min=ere_c_min,
            ere_k_max=ere_k_max,
        )
    def set_bc(mut self, weight: Scalar[DT], n_demo_rows: Int) raises:
        """Behaviour-cloning penalty on the batch's first `n_demo_rows` rows —
        `SACActorLoss.set_bc`. Call after the demos are pinned."""
        self.actor_loss_blk.set_bc(weight, n_demo_rows, self.ctx)

    def set_bc_q_ratio(mut self, ratio: Scalar[DT]) raises:
        """Track the critic: after every `flush_metrics` the BC weight is set
        to max(λ0, ratio · mean|Q|), λ0 being the weight `set_bc` gave. A
        fixed λ against a growing Q shrinks the BC term's share of the actor
        gradient — the tower lift policy peaked at 25k (Q 58) and lost its
        grasp by 50k (Q 87) that way. 0 turns the tracking off."""
        if ratio < Scalar[DT](0):
            raise Error("set_bc_q_ratio: ratio must be >= 0")
        self.bc_q_ratio = ratio
        self.bc_weight_floor = self.actor_loss_blk.bc_weight

    def bc_weight(self) -> Scalar[DT]:
        """The BC weight in force (`SACActorLoss.bc_weight`)."""
        return self.actor_loss_blk.bc_weight

    def set_q_weight(mut self, weight: Scalar[DT]) raises:
        """Multiplier on the SAC half of the actor loss — 0 = BC only.
        `SACActorLoss.set_q_weight`."""
        self.actor_loss_blk.set_q_weight(weight)

    def set_beta(mut self, beta: Scalar[DT]):
        """PER IS-β anneal hook. No-op for uniform sample blocks."""
        self.sample_blk.set_beta(beta)

    # ─── train_step ────────────────────────────────────────────────────
    def train_step(mut self, step_idx: Int) raises -> Bool:
        self.state.step_idx = step_idx
        self.state.did_step = True
        # CPU bakes the host α scalar into the target-y / actor Scale nodes per
        # step. On GPU α lives on-device (wired once at make) and is refreshed
        # by the device ScalarAdam; `state.alpha` is stale/unused on GPU.
        comptime if Self.train_target == "cpu":
            self.state.alpha = fexp(self.alpha_opt.value)
        else:
            self.state.ctx = self.ctx

        self.sample_blk.step(self.state)
        if not self.state.did_step:
            return False

        self.target_y_blk.step[Self.train_target](
            self.state, self.actor, self.pair1.target_net, self.pair2.target_net
        )
        # ACCUMULATE on GPU: the per-batch critic loss is reduced on-device into
        # each critic's MSELoss accumulator (read at flush) — NO per-step D2H.
        # CPU keeps the host `MSELoss.forward` scalar. Metric-only; the gradient
        # (mse_loss.vjp + critic.vjp) is identical either way.
        self.twin_critic_blk.step[
            Self.train_target, ACCUMULATE = Self.train_target == "gpu"
        ](
            self.state,
            self.pair1.online,
            self.critic1_opt,
            self.pair2.online,
            self.critic2_opt,
        )
        var out = self.actor_loss_blk.forward_backward[Self.train_target](
            self.actor,
            self.actor_opt,
            self.pair1.online,
            self.pair2.online,
            self.state.mb_s,
            self.state.mb_a,
            self.state.alpha,
            self.ctx,
        )
        self.state.log_prob_mean = out.log_prob_mean
        self.state.actor_loss = out.loss
        # CPU: host-scalar grad from state.log_prob_mean. GPU: the device
        # ScalarAdam reads the actor-loss `lp_mean` device buffer directly (no
        # D2H) and refreshes the device α the Scale nodes read.
        comptime if Self.train_target == "cpu":
            self.alpha_blk.step[Self.train_target](self.state, self.alpha_opt)
        else:
            self.alpha_opt.step_device(
                self.ctx.value(),
                self.actor_loss_blk.lp_mean_dev(),
                self.alpha_blk.target_entropy,
            )
        self.polyak_blk.step[Self.train_target](
            self.state, self.pair1, self.pair2
        )

        # PER tail (no-op for uniform blocks).
        self.sample_blk.update_priorities(self.state)

        # Host bookkeeping.
        self._actor_L_accum += out.loss
        self._critic_L_accum += self.state.critic_loss
        self._alpha_accum += fexp(self.alpha_opt.value)
        # Diagnostic batch means. CPU sums the host scratches directly; GPU
        # folds each batch mean into a device-resident accumulator via a tiny
        # reduction kernel — NO per-step D2H (capture-safe), read once per
        # `diag_every` flush. Mirrors the MBPO/DQN diagnostic path.
        comptime B = Self.BATCH
        comptime A = Self.ACT_DIM
        comptime if Self.train_target == "cpu":
            var inv_b = Scalar[DT](1.0) / Scalar[DT](B)
            var sq: Scalar[DT] = 0.0
            var sy: Scalar[DT] = 0.0
            var sr: Scalar[DT] = 0.0
            var snq: Scalar[DT] = 0.0
            var sd: Scalar[DT] = 0.0
            for b in range(B):
                sq += self.twin_critic_blk.inner.c1._mb_q.data[b]
                sy += self.state.mb_y.data[b]
                sr += self.state.mb_r.data[b]
                snq += self.target_y_blk.graph.node_output["min_q"]().data[b]
                sd += self.state.mb_d.data[b]
            var saa: Scalar[DT] = 0.0
            for k in range(B * A):
                var av = self.state.mb_a.data[k]
                saa += av if av >= 0 else -av
            self._mean_q_accum += sq * inv_b
            self._mean_target_accum += sy * inv_b
            self._mean_reward_accum += sr * inv_b
            self._mean_next_q_accum += snq * inv_b
            self._mean_done_accum += sd * inv_b
            self._mean_abs_action_accum += saa / Scalar[DT](B * A)
        else:
            comptime lb = Layout.row_major(B)
            comptime lba = Layout.row_major(B * A)
            self._mean_q_dev.accumulate_gpu_lt[B](
                self.twin_critic_blk.inner.c1._mb_q.lt["gpu", lb]()
            )
            self._mean_target_dev.accumulate_gpu_lt[B](
                self.state.mb_y.lt["gpu", lb]()
            )
            self._mean_reward_dev.accumulate_gpu_lt[B](
                self.state.mb_r.lt["gpu", lb]()
            )
            self._mean_next_q_dev.accumulate_gpu_lt[B](
                self.target_y_blk.graph.node_output["min_q"]().lt["gpu", lb]()
            )
            self._mean_done_dev.accumulate_gpu_lt[B](
                self.state.mb_d.lt["gpu", lb]()
            )
            self._mean_abs_action_dev.accumulate_gpu_abs_lt[B * A](
                self.state.mb_a.lt["gpu", lba]()
            )
        self._update_count += 1
        self._total_train_steps += 1
        return True

    def total_train_steps(self) -> Int:
        return self._total_train_steps

    # ─── CUDA-graph capture surface (OffPolicyAgentGpu) ────────────────
    #
    # The SAC GPU hot loop is sync-free (diagnostics, actor loss, α and critic
    # loss are all device-resident, read only at flush), so the per-update
    # device-kernel sequence is capturable. `train_device_kernels` is that
    # sequence with NO host gate and NO host counters — the body of the driver's
    # capture closure. The driver gates entry on `learning_starts_count()` and
    # advances host counters once per logical update via `note_train_update()`.
    # Mirrors `train_step`'s GPU path (kept in sync by hand — both run sample →
    # target_y → twin_critic[ACCUMULATE] → actor_loss → α step_device → polyak →
    # device diagnostics).
    def train_device_kernels(mut self) raises:
        """Stream-capture entry point: the step on the trainer's own context.

        ⚠ The context is COPIED into a local first. `self.ctx.value()` borrows
        from `self`, and `self` is passed mutably here, so handing the borrow
        straight through is an aliasing error ("passed mutably to 'self' and
        immutably to 'gctx'"). `DeviceContext` is a refcounted handle, so the
        copy is a retain, not a device operation.
        """
        var c = self.ctx.value()
        self.train_device_kernels_on(c)

    def train_device_kernels_on(mut self, gctx: DeviceContext) raises:
        """The same step, enqueued on `gctx` so it can be RECORDED.

        ⚠⚠ EVERY CACHED CONTEXT IN THE BODY IS REDIRECTED BELOW, AND MISSING
        ONE IS SILENT. Helpers that take no `ctx` argument reach for a context
        they stored at `make` time; under a recording pass those enqueue
        EAGERLY instead of recording, so the build pass folds one spurious
        batch in and every replay skips them. Training stays correct while the
        diagnostics quietly stop advancing — nothing raises on either side.
        `DeviceMeanAccum.set_ctx` spells the trap out.

        So the redirect must be COMPLETE. The six accumulators below are every
        `DeviceMeanAccum` this trainer owns (fields at :155-160); `state.ctx`
        carries the context to `sample_blk` / `target_y_blk` / `twin_critic_blk`
        / `polyak_blk`; `actor_loss_blk` and `alpha_opt` take one directly.
        Adding a seventh accumulator, or a block that caches its own context,
        means adding it here — grep both, do not assume.
        """
        comptime assert Self.train_target == "gpu", (
            "train_device_kernels_on is GPU-only (device-graph record path)"
        )
        var octx = Optional[DeviceContext](gctx)

        # ── redirect every cached context onto `gctx` (see the docstring) ──
        self._mean_q_dev.set_ctx(gctx)
        self._mean_target_dev.set_ctx(gctx)
        self._mean_reward_dev.set_ctx(gctx)
        self._mean_next_q_dev.set_ctx(gctx)
        self._mean_done_dev.set_ctx(gctx)
        self._mean_abs_action_dev.set_ctx(gctx)

        # Pin step_idx past warmup so the sample block's gate passes; the driver
        # only enters capture once the buffer is warm.
        self.state.step_idx = self.learning_starts
        self.state.did_step = True
        self.state.ctx = octx
        self.sample_blk.step(self.state)

        self.target_y_blk.step["gpu"](
            self.state, self.actor, self.pair1.target_net, self.pair2.target_net
        )
        self.twin_critic_blk.step["gpu", ACCUMULATE=True](
            self.state,
            self.pair1.online,
            self.critic1_opt,
            self.pair2.online,
            self.critic2_opt,
        )
        _ = self.actor_loss_blk.forward_backward["gpu"](
            self.actor,
            self.actor_opt,
            self.pair1.online,
            self.pair2.online,
            self.state.mb_s,
            self.state.mb_a,
            self.state.alpha,
            octx,
        )
        self.alpha_opt.step_device(
            gctx,
            self.actor_loss_blk.lp_mean_dev(),
            self.alpha_blk.target_entropy,
        )
        self.polyak_blk.step["gpu"](self.state, self.pair1, self.pair2)
        self.sample_blk.update_priorities(self.state)

        # Device diagnostics (fold each batch mean into the device accumulators;
        # captured + replayed → advances identically to the non-captured path).
        comptime lb = Layout.row_major(Self.BATCH)
        comptime lba = Layout.row_major(Self.BATCH * Self.ACT_DIM)
        self._mean_q_dev.accumulate_gpu_lt[Self.BATCH](
            self.twin_critic_blk.inner.c1._mb_q.lt["gpu", lb]()
        )
        self._mean_target_dev.accumulate_gpu_lt[Self.BATCH](
            self.state.mb_y.lt["gpu", lb]()
        )
        self._mean_reward_dev.accumulate_gpu_lt[Self.BATCH](
            self.state.mb_r.lt["gpu", lb]()
        )
        self._mean_next_q_dev.accumulate_gpu_lt[Self.BATCH](
            self.target_y_blk.graph.node_output["min_q"]().lt["gpu", lb]()
        )
        self._mean_done_dev.accumulate_gpu_lt[Self.BATCH](
            self.state.mb_d.lt["gpu", lb]()
        )
        self._mean_abs_action_dev.accumulate_gpu_abs_lt[Self.BATCH * Self.ACT_DIM](
            self.state.mb_a.lt["gpu", lba]()
        )

    def note_train_update(mut self):
        """Advance one logical update's host counters under graph replay (the
        device work is replayed by the captured graph; loss / α / diagnostics
        are device-resident and drained at flush, so only the counters live on
        the host)."""
        self._update_count += 1
        self._total_train_steps += 1

    def learning_starts_count(self) -> Int:
        """Env-step threshold after which the replay is warm enough to train —
        the driver gates the capture path on this."""
        return self.learning_starts

    # ─── Action selection ──────────────────────────────────────────────
    def select_action_batched[
        N_ENVS: Int
    ](
        mut self,
        obs: LayoutTensor[
            DT, Layout.row_major(N_ENVS, Self.AGENT_OBS_DIM), MutAnyOrigin
        ],
        action: LayoutTensor[
            DT, Layout.row_major(N_ENVS, Self.AGENT_ACT_DIM), MutAnyOrigin
        ],
        ao_scratch: LayoutTensor[
            DT, Layout.row_major(N_ENVS, 2 * Self.AGENT_ACT_DIM), MutAnyOrigin
        ],
        alp_scratch: LayoutTensor[
            DT, Layout.row_major(N_ENVS, Self.AGENT_ACT_DIM + 1), MutAnyOrigin
        ],
        step_idx: Int,
    ) raises:
        comptime assert N_ENVS > 0, "N_ENVS must be > 0"
        comptime ACT = Self.ACT_DIM
        comptime OBS = Self.OBS_DIM

        # ── Warmup: uniform random action in [-action_scale, +scale].
        if step_idx < self.learning_starts:
            warmup_uniform_batched[Self.train_target, N_ENVS, ACT](
                action,
                self.action_scale,
                self.ctx,
                self._warmup_rng_seed,
                self._warmup_rng_offset,
            )
            return

        # ── Policy: shared squashed-actor body (see
        # training/blocks/action_select.mojo — one copy for
        # sac/redq/redq_ofe/mbpo).
        select_squashed_batched[
            Self.ACTOR, Self.train_target, N_ENVS, OBS, ACT
        ](
            self.actor,
            self.sel,
            self._ob_scr,
            self._ao_scr,
            self._alp_scr,
            obs,
            action,
            self.action_scale,
            self.ctx,
        )
        # silence unused warnings on the driver-owned scratch views.
        _ = ao_scratch
        _ = alp_scratch

    def select_greedy_action(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
    ) raises:
        comptime ACT = Self.ACT_DIM
        comptime OBS = Self.OBS_DIM
        comptime if Self.train_target == "cpu":
            self._ob_scr.ensure(OBS)
            self._ao_scr.ensure(2 * ACT)
            for d in range(OBS):
                self._ob_scr.data[d] = obs[d]
            call_forward["cpu", 1](
                self.actor, TensorRefs[Self.ACTOR.ARITY](self._ob_scr),
                self._ao_scr,
            )
            for j in range(ACT):
                var a = ftanh(self._ao_scr.data[j]) * self.action_scale
                if a > self.action_scale:
                    a = self.action_scale
                elif a < -self.action_scale:
                    a = -self.action_scale
                action_out[j] = a
        else:
            # Fresh single-env Tensors (NOT the batched _ob/_ao scratch, whose
            # `n` is sized for N_ENVS during training — `upload` walks `n` host
            # elements, so reusing them here would read past the 1-env fill).
            var c = self.ctx.value()
            var ob = Tensor.alloc(OBS)
            for d in range(OBS):
                ob.data[d] = obs[d]
            ob.upload(c)
            var ao = Tensor.alloc_gpu(c, 2 * ACT)
            call_forward["gpu", 1](
                self.actor, TensorRefs[Self.ACTOR.ARITY](ob), ao, self.ctx
            )
            ao.download(c)
            for j in range(ACT):
                var a = ftanh(ao.data[j]) * self.action_scale
                if a > self.action_scale:
                    a = self.action_scale
                elif a < -self.action_scale:
                    a = -self.action_scale
                action_out[j] = a

    def select_action(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
        step_idx: Int,
    ) raises:
        """Host-list stochastic action — user-facing entry for smoke tests
        that bypass the driver. Stages obs into the owned scratch and runs
        the warmup/policy path directly."""
        comptime ACT = Self.ACT_DIM
        comptime OBS = Self.OBS_DIM
        comptime if Self.train_target == "cpu":
            if step_idx < self.learning_starts:
                for j in range(ACT):
                    var u = Scalar[DT](2.0 * random_float64() - 1.0)
                    action_out[j] = u * self.action_scale
                return
            self._ob_scr.ensure(OBS)
            self._ao_scr.ensure(2 * ACT)
            self._alp_scr.ensure(ACT + 1)
            for d in range(OBS):
                self._ob_scr.data[d] = obs[d]
            call_forward["cpu", 1](
                self.actor, TensorRefs[Self.ACTOR.ARITY](self._ob_scr),
                self._ao_scr,
            )
            call_forward["cpu", 1](
                self.sel, TensorRefs[1](self._ao_scr), self._alp_scr
            )
            for j in range(ACT):
                var a = self._alp_scr.data[j]
                if a > self.action_scale:
                    a = self.action_scale
                elif a < -self.action_scale:
                    a = -self.action_scale
                action_out[j] = a
        else:
            if step_idx < self.learning_starts:
                for j in range(ACT):
                    var u = Scalar[DT](2.0 * random_float64() - 1.0)
                    action_out[j] = u * self.action_scale
                return
            # Fresh single-env Tensors (see select_greedy_action for why the
            # batched scratch can't be reused on the host-list path).
            var c = self.ctx.value()
            var ob = Tensor.alloc(OBS)
            for d in range(OBS):
                ob.data[d] = obs[d]
            ob.upload(c)
            var ao = Tensor.alloc_gpu(c, 2 * ACT)
            var alp = Tensor.alloc_gpu(c, ACT + 1)
            call_forward["gpu", 1](
                self.actor, TensorRefs[Self.ACTOR.ARITY](ob), ao, self.ctx
            )
            call_forward["gpu", 1](self.sel, TensorRefs[1](ao), alp, self.ctx)
            alp.download(c)
            for j in range(ACT):
                var a = alp.data[j]
                if a > self.action_scale:
                    a = self.action_scale
                elif a < -self.action_scale:
                    a = -self.action_scale
                action_out[j] = a

    # ─── Record ────────────────────────────────────────────────────────
    def record(
        mut self,
        ref obs: List[Scalar[DT]],
        ref action: List[Scalar[DT]],
        reward: Scalar[DT],
        ref next_obs: List[Scalar[DT]],
        done: Scalar[DT],
    ) raises:
        self.tracker.add_reward(reward)
        self.sample_blk.add(obs, action, reward, next_obs, done, ctx=self.ctx)

    def _replay_add(
        mut self,
        ref obs: List[Scalar[DT]],
        ref action: List[Scalar[DT]],
        reward: Scalar[DT],
        ref next_obs: List[Scalar[DT]],
        done: Scalar[DT],
    ) raises:
        self.sample_blk.add(obs, action, reward, next_obs, done, ctx=self.ctx)

    def _tracker_ptr(self) -> Pointer[EpisodeTracker, MutAnyOrigin]:
        return rebind[Pointer[EpisodeTracker, MutAnyOrigin]](
            Pointer(to=self.tracker)
        )

    # ─── GPU-batched record surface (raise until GPU storage migrated) ─
    def record_batch_gpu[
        N_ENVS: Int
    ](
        mut self,
        ctx: DeviceContext,
        prev_obs_dev: DeviceBuffer[DT],
        action_dev: DeviceBuffer[DT],
        reward_dev: DeviceBuffer[DT],
        obs_dev: DeviceBuffer[DT],
        done_dev: DeviceBuffer[DT],
    ) raises:
        """Delegate the N_ENVS device transitions to the sample block's GPU
        replay (one kernel launch)."""
        self.sample_blk.add_batch_gpu[N_ENVS](
            ctx, prev_obs_dev, action_dev, reward_dev, obs_dev, done_dev,
        )

    def record_batch_gpu_nstep[
        N_ENVS: Int, NS: Int
    ](
        mut self,
        ctx: DeviceContext,
        mut nstep_buf: GPUNStepBuffer[
            NS, Self.AGENT_OBS_DIM, Self.AGENT_ACT_DIM, N_ENVS,
        ],
        prev_obs_dev: DeviceBuffer[DT],
        action_dev: DeviceBuffer[DT],
        reward_dev: DeviceBuffer[DT],
        obs_dev: DeviceBuffer[DT],
        done_dev: DeviceBuffer[DT],
    ) raises:
        """Push the device transitions through the n-step buffer, then store
        matured n-step transitions into the GPU replay via the sample block."""
        nstep_buf.process(
            ctx, prev_obs_dev, action_dev, reward_dev, obs_dev, done_dev,
        )
        self.sample_blk.store_via_block_gpu[N_ENVS, NS](ctx, nstep_buf)

    # ─── Metrics / logging ─────────────────────────────────────────────
    def flush_metrics[
        L: Logger = NoOpLogger
    ](
        mut self,
        logger: Optional[Pointer[L, MutAnyOrigin]] = None,
        step: Int = 0,
    ) raises -> SACMetrics:
        """Drain accumulators into a SACMetrics bundle. The per-batch
        diagnostics (mean_q / mean_target / mean_next_q / mean_reward /
        mean_done / mean_abs_action) are real on BOTH targets: CPU reads the
        host accumulators (averaged over the window); GPU reads the
        device-resident `DeviceMeanAccum`s with ONE D2H each at this flush —
        never in the per-step hot loop."""
        var n = self._update_count if self._update_count > 0 else 1
        var inv = Scalar[DT](1.0) / Scalar[DT](n)
        var mq: Scalar[DT]
        var mtgt: Scalar[DT]
        var mr: Scalar[DT]
        var mnq: Scalar[DT]
        var md: Scalar[DT]
        var maa: Scalar[DT]
        # actor_loss + α: CPU reads the host accumulators; GPU reads the
        # device-resident actor-loss accumulator + the live device α (no
        # per-step D2H — drained once here at flush cadence).
        var actor_val: Scalar[DT]
        var alpha_val: Scalar[DT]
        var critic_val: Scalar[DT]
        comptime if Self.train_target == "gpu":
            mq = self._mean_q_dev.read["gpu"]()
            mtgt = self._mean_target_dev.read["gpu"]()
            mr = self._mean_reward_dev.read["gpu"]()
            mnq = self._mean_next_q_dev.read["gpu"]()
            md = self._mean_done_dev.read["gpu"]()
            maa = self._mean_abs_action_dev.read["gpu"]()
            actor_val = self.actor_loss_blk.read_loss_accum(self.ctx.value())
            alpha_val = self.alpha_opt.read_alpha()
            # critic loss = c1 + c2 window means (matches the CPU `loss1+loss2`
            # the twin block returns); device accumulators, ONE D2H each here.
            critic_val = self.twin_critic_blk.inner.c1.mse_loss.read_accum[
                "gpu"
            ](self.ctx) + self.twin_critic_blk.inner.c2.mse_loss.read_accum[
                "gpu"
            ](self.ctx)
        else:
            mq = self._mean_q_accum * inv
            mtgt = self._mean_target_accum * inv
            mr = self._mean_reward_accum * inv
            mnq = self._mean_next_q_accum * inv
            md = self._mean_done_accum * inv
            maa = self._mean_abs_action_accum * inv
            actor_val = self._actor_L_accum * inv
            alpha_val = self._alpha_accum * inv
            critic_val = self._critic_L_accum * inv
        var bundle = SACMetrics(
            policy_loss=LogScalar[DT](actor_val),
            critic_loss=LogScalar[DT](critic_val),
            alpha=LogScalar[DT](alpha_val),
            mean_q=LogScalar[DT](mq),
            mean_target=LogScalar[DT](mtgt),
            reward_mean=LogScalar[DT](mr),
            mean_next_q=LogScalar[DT](mnq),
            mean_done=LogScalar[DT](md),
            action_abs_mean=LogScalar[DT](maa),
            train_steps=LogScalar[DT](Scalar[DT](self._total_train_steps)),
            n_updates=LogScalar[DT](Scalar[DT](self._update_count)),
        )
        self._actor_L_accum = Scalar[DT](0.0)
        self._critic_L_accum = Scalar[DT](0.0)
        self._alpha_accum = Scalar[DT](0.0)
        self._mean_q_accum = Scalar[DT](0.0)
        self._mean_target_accum = Scalar[DT](0.0)
        self._mean_reward_accum = Scalar[DT](0.0)
        self._mean_next_q_accum = Scalar[DT](0.0)
        self._mean_done_accum = Scalar[DT](0.0)
        self._mean_abs_action_accum = Scalar[DT](0.0)
        comptime if Self.train_target == "gpu":
            self._mean_q_dev.reset["gpu"]()
            self._mean_target_dev.reset["gpu"]()
            self._mean_reward_dev.reset["gpu"]()
            self._mean_next_q_dev.reset["gpu"]()
            self._mean_done_dev.reset["gpu"]()
            self._mean_abs_action_dev.reset["gpu"]()
            self.actor_loss_blk.reset_loss_accum()
            self.twin_critic_blk.inner.c1.mse_loss.reset_accum["gpu"]()
            self.twin_critic_blk.inner.c2.mse_loss.reset_accum["gpu"]()
        self._update_count = 0
        if self.bc_q_ratio > Scalar[DT](0) and n > 0:
            var scaled = self.bc_q_ratio * (mq if mq >= Scalar[DT](0) else -mq)
            var w = scaled if scaled > self.bc_weight_floor else self.bc_weight_floor
            self.actor_loss_blk.set_bc_weight(w, self.ctx)
            if Bool(logger):
                logger.value()[].log_scalar(String("bc_weight"), Float64(w), step)
        if Bool(logger):
            log_bundle(logger.value()[], bundle, step)
        return bundle^

    def flush_metrics_through_logger[
        L: Logger
    ](
        mut self,
        logger: Optional[Pointer[L, MutAnyOrigin]],
        step: Int,
    ) raises:
        _ = self.flush_metrics[L](logger, step)

    def flush_train_log(
        mut self,
    ) raises -> Tuple[Scalar[DT], Scalar[DT], Scalar[DT], Int]:
        """(actor_loss, critic_loss, alpha, n_updates) over the window.
        Secondary log path — `flush_metrics` is the authoritative bundle. On
        GPU actor_loss + α are read from the device accumulator / live device α
        (matching `flush_metrics`); the host scalars are never advanced on the
        GPU path, so reading them here would report 0 / a frozen init-α."""
        var n = self._update_count if self._update_count > 0 else 1
        var inv = Scalar[DT](1.0) / Scalar[DT](n)
        var actor_val: Scalar[DT]
        var alpha_val: Scalar[DT]
        var critic_val: Scalar[DT]
        comptime if Self.train_target == "gpu":
            actor_val = self.actor_loss_blk.read_loss_accum(self.ctx.value())
            alpha_val = self.alpha_opt.read_alpha()
            critic_val = self.twin_critic_blk.inner.c1.mse_loss.read_accum[
                "gpu"
            ](self.ctx) + self.twin_critic_blk.inner.c2.mse_loss.read_accum[
                "gpu"
            ](self.ctx)
        else:
            actor_val = self._actor_L_accum * inv
            alpha_val = self._alpha_accum * inv
            critic_val = self._critic_L_accum * inv
        var out = (
            actor_val,
            critic_val,
            alpha_val,
            self._update_count,
        )
        self._actor_L_accum = Scalar[DT](0.0)
        self._critic_L_accum = Scalar[DT](0.0)
        self._alpha_accum = Scalar[DT](0.0)
        self._update_count = 0
        return out

    # ─── Checkpoint (ONE file: actor + twin critics, v3 binary) ────────────
    def save_state(mut self, path: String) raises:
        """Write actor + the two ONLINE critics into a SINGLE v3 binary
        checkpoint (chunked + atomic tmp-rename I/O — the old v2 text path
        wrote the final file in one `f.write`, non-atomic and silently
        truncated at the ~2 GiB write(2) cap). α's whole optimizer
        (`alpha.*`) and the train-step counter ride as `K` scalars; network
        optimizer moments are NOT persisted (resume re-warms). α is, because
        a resume that restarts it at its initial value undoes the part of the
        run that tuned it."""
        var sc = CheckpointScalars()
        self.alpha_opt.put_state(sc, "alpha")
        sc.set_int("total_train_steps", self._total_train_steps)
        save_params_multi[Self.train_target](
            path,
            self.ctx,
            False,
            self.actor,
            self.pair1.online,
            self.pair2.online,
            scalars=sc,
        )

    def load_state(mut self, path: String) raises:
        """Restore actor + twin online critics (v3 binary, or the legacy
        `actor.`/`critic1.`/`critic2.`-prefixed v2 text envelope this trainer
        used to write), then hard-copy online → target."""
        if _is_v3_header(_read_file_bytes(path)):
            var sc = load_params_multi[Self.train_target](
                path,
                self.ctx,
                self.actor,
                self.pair1.online,
                self.pair2.online,
            )
            self.alpha_opt.take_state(sc, "alpha")
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
            walk_params[Self.train_target](self.pair1.online, r, self.ctx, "critic1"
            )
            walk_params[Self.train_target](self.pair2.online, r, self.ctx, "critic2"
            )
            r.mode = 1
            var _sref1 = ParamVisitorRef.of[type_of(r), Self.train_target](r)
            self.actor.for_each_state[Self.train_target](_sref1, self.ctx, "actor")
            self.pair1.online.for_each_state[Self.train_target](
                r, self.ctx, "critic1"
            )
            self.pair2.online.for_each_state[Self.train_target](
                r, self.ctx, "critic2"
            )
        self.pair1.target_net.polyak_from[Self.train_target](
            self.pair1.online, Scalar[DT](1.0), self.ctx
        )
        self.pair2.target_net.polyak_from[Self.train_target](
            self.pair2.online, Scalar[DT](1.0), self.ctx
        )

    def flush_timer_log(mut self) -> String:
        return String("")
