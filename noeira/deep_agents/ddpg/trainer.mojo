"""DDPGTrainer — storage-framework DDPG trainer (CPU gate; GPU stretch).

Assembles the migrated `noeira.nn` DDPG blocks into a single
driver-conforming `DDPGTrainer` that conforms `OffPolicyAgentGpu` so both
the CPU single-env driver (`run_offpolicy_train`) and the batched drivers
type-check. Structural sibling of `SACTrainer` with the DDPG
simplifications:

  * The actor HAS a target — DDPG polyaks the actor too — so the actor
    lives in an `OnlineTargetPair[ACTOR]` (online + target).
  * SINGLE critic (`OnlineTargetPair[CRITIC]`, `SingleCriticStep`) — no
    twin-critic min, no `mean_next_q`.
  * NO entropy temperature (no α opt / AlphaUpdateStep / device-α wiring).
  * Deterministic action selection with additive Gaussian exploration
    noise (NO rsample). The Tanh-bounded actor output is fed straight to
    the env (clamped to ±action_scale); `action_scale` only bounds the
    warmup uniform + clamp — it is NOT multiplied into the deterministic
    policy output (bit-consistent with the legacy DDPG `ActionSamplingBlock`
    and the storage `DDPGTargetYBlock` / `DDPGActorLoss`, which both feed
    the raw Tanh action to the critic).

Pipeline (per train step):
  state.step_idx = step
  sample_blk.step(state)                                  # fills state.mb_*
  target_y_blk.step(state, actor_t, critic_t)             # writes state.mb_y
  critic_blk.step[ACCUMULATE=(gpu)](state, critic, c_opt)
  actor_blk.step(state, actor, actor_opt, critic)         # DPG step
  polyak_blk.step(state, actor_pair, critic_pair)

Dimensions (OBS / ACT / BATCH) derive from `SAMPLE` so they're specified
once (on the sample block type).

CUDA-graph capture (`train_device_kernels` / `note_train_update` /
`learning_starts_count`) is DEFERRED — the `OffPolicyAgentGpu` trait
defaults raise, never reached with `USE_TRAIN_CUDA_GRAPH=False`.
"""

from noeira.nn.core.param import walk_params
from std.math import tanh as ftanh
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
from noeira.nn.core.initializer import Xavier
from noeira.nn.optimizer.adam import Adam
from noeira.nn.core.checkpoint import (
    BinaryCheckpointWriter, BinaryCheckpointReader, CheckpointReader,
    CheckpointScalars, write_model, read_model, _split_lines, _is_v3_header,
    _read_file_bytes, _write_file_bytes,
)
from noeira.nn.random.box_muller import box_muller_normal, box_muller_normal_gpu

from noeira.nn.core.log_bundle import log_bundle
from noeira.nn.core.metric import LogScalar

from ..data.n_step_replay import GPUNStepBuffer
from ..core.online_target_pair import OnlineTargetPair
from ..training.episode_tracker import EpisodeTracker
from ..training.device_mean_accum import DeviceMeanAccum
from ..training.trainer_block import TrainerState
from ..training.driver_offpolicy import OffPolicyAgentGpu
from ..training.blocks import SampleBlock, SingleCriticStep
from ..training.blocks.action_select import (
    select_deterministic_batched,
    warmup_uniform_batched,
)
from .target_y_block import DDPGTargetYBlock
from .blocks.actor_step import DDPGActorStep
from .blocks.polyak_step import DDPGPolyakStep
from .metrics import DDPGMetrics


# ──────────────────────────────────────────────────────────────────────
# GPU device kernels for the batched action-selection path. Mirror the
# SAC trainer's device body (Philox warmup + obs copy) adapted to the
# storage actor surface (which takes owned Tensors, so obs is COPIED into
# the trainer's device scratch before the actor forward). The noise+clamp
# kernel is DDPG-specific (deterministic actor + additive Gaussian, no
# rsample).
# ──────────────────────────────────────────────────────────────────────


struct DDPGTrainer[
    train_target: StaticString,
    SAMPLE: SampleBlock,
    ACTOR: Module,
    CRITIC: Module,
](OffPolicyAgentGpu):
    """Storage-framework DDPG trainer. Dimensions (OBS / ACT / BATCH) are
    derived from SAMPLE so the user specifies them once on the sample
    block type (mirrors SACTrainer)."""

    comptime OBS_DIM: Int = Self.SAMPLE.OBS
    comptime ACT_DIM: Int = Self.SAMPLE.ACT
    comptime BATCH: Int = Self.SAMPLE.BATCH

    comptime AGENT_OBS_DIM: Int = Self.OBS_DIM
    comptime AGENT_ACT_DIM: Int = Self.ACT_DIM
    comptime AGENT_TRAIN_TARGET: StaticString = Self.train_target

    var actor_pair: OnlineTargetPair[Self.ACTOR]
    var critic_pair: OnlineTargetPair[Self.CRITIC]
    var actor_opt: Adam
    var critic_opt: Adam

    var sample_blk: Self.SAMPLE
    var target_y_blk: DDPGTargetYBlock[
        Self.ACTOR, Self.CRITIC, Self.BATCH, Self.OBS_DIM, Self.ACT_DIM,
    ]
    var critic_blk: SingleCriticStep[
        Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.CRITIC,
    ]
    var actor_blk: DDPGActorStep[
        Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.ACTOR, Self.CRITIC,
    ]
    var polyak_blk: DDPGPolyakStep[
        Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.ACTOR, Self.CRITIC,
    ]

    var state: TrainerState[Self.OBS_DIM, Self.ACT_DIM, Self.BATCH]
    var tracker: EpisodeTracker
    var ctx: Optional[DeviceContext]

    # Owned action-selection scratch Tensors (lazily `.ensure`d per call).
    var _ob_scr: Tensor   # N_ENVS * OBS
    var _ao_scr: Tensor   # N_ENVS * ACT (deterministic actor output)
    var _noise_scr: Tensor  # N_ENVS * ACT (box-muller fill)

    var action_scale: Scalar[DT]
    var noise_scale: Scalar[DT]
    var learning_starts: Int

    # Philox state for batched warmup + exploration noise (gpu path only).
    var _warmup_rng_seed: UInt64
    var _warmup_rng_offset: UInt64
    var _noise_rng_seed: UInt64
    var _noise_rng_offset: UInt64

    # Host metric accumulators (CPU path).
    var _actor_L_accum: Scalar[DT]
    var _critic_L_accum: Scalar[DT]
    # Diagnostic accumulators (CPU path): batch means drained at flush_metrics.
    var _mean_q_accum: Scalar[DT]        # online Q(s, a) over the batch
    var _mean_target_accum: Scalar[DT]   # Bellman target y
    var _mean_reward_accum: Scalar[DT]   # batch reward
    # Diagnostic accumulators (GPU path): device-resident running means.
    var _mean_q_dev: DeviceMeanAccum
    var _mean_target_dev: DeviceMeanAccum
    var _mean_reward_dev: DeviceMeanAccum
    var _update_count: Int
    var _total_train_steps: Int

    def __init__(out self):
        self.actor_pair = OnlineTargetPair[Self.ACTOR]()
        self.critic_pair = OnlineTargetPair[Self.CRITIC]()
        self.actor_opt = Adam(lr=Scalar[DT](1e-4))
        self.critic_opt = Adam(lr=Scalar[DT](1e-3))
        self.sample_blk = Self.SAMPLE()
        self.target_y_blk = DDPGTargetYBlock[
            Self.ACTOR, Self.CRITIC, Self.BATCH, Self.OBS_DIM, Self.ACT_DIM,
        ]()
        self.critic_blk = SingleCriticStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.CRITIC,
        ]()
        self.actor_blk = DDPGActorStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.ACTOR, Self.CRITIC,
        ]()
        self.polyak_blk = DDPGPolyakStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.ACTOR, Self.CRITIC,
        ]()
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
        self._noise_scr = Tensor()
        self.action_scale = Scalar[DT](1.0)
        self.noise_scale = Scalar[DT](0.1)
        self.learning_starts = 1_000
        self._warmup_rng_seed = UInt64(0xC0FFEE_C0DE)
        self._warmup_rng_offset = UInt64(0)
        self._noise_rng_seed = UInt64(0xD15EA5E_D00D)
        self._noise_rng_offset = UInt64(0)
        self._actor_L_accum = Scalar[DT](0.0)
        self._critic_L_accum = Scalar[DT](0.0)
        self._mean_q_accum = Scalar[DT](0.0)
        self._mean_target_accum = Scalar[DT](0.0)
        self._mean_reward_accum = Scalar[DT](0.0)
        self._mean_q_dev = DeviceMeanAccum()
        self._mean_target_dev = DeviceMeanAccum()
        self._mean_reward_dev = DeviceMeanAccum()
        self._update_count = 0
        self._total_train_steps = 0

    @staticmethod
    def make(
        ctx: Optional[DeviceContext] = None,
        actor_lr: Scalar[DT] = Scalar[DT](1e-4),
        critic_lr: Scalar[DT] = Scalar[DT](1e-3),
        gamma: Scalar[DT] = Scalar[DT](0.99),
        tau: Scalar[DT] = Scalar[DT](0.005),
        action_scale: Scalar[DT] = Scalar[DT](1.0),
        noise_scale: Scalar[DT] = Scalar[DT](0.1),
        learning_starts: Int = 1_000,
        window_size: Int = 10,
        initial_episode_fill: Scalar[DT] = Scalar[DT](-1250.0),
        max_grad_norm: Scalar[DT] = Scalar[DT](0.0),
        use_bf16: Bool = False,
    ) raises -> Self:
        """Unified factory. `ctx` required for train_target='gpu'.
        `max_grad_norm` / `use_bf16` accepted for signature parity with the
        agent facade (storage Adam clips internally; bf16 is a GPU stretch)."""
        comptime assert (
            Self.train_target == "cpu" or Self.train_target == "gpu"
        ), "DDPGTrainer: target must be 'cpu' or 'gpu'"
        comptime if Self.train_target == "gpu":
            if not ctx:
                raise Error("DDPGTrainer.make[target='gpu']: ctx required")

        var t = Self()
        t.ctx = ctx

        t.actor_pair = OnlineTargetPair[Self.ACTOR].make[
            Self.train_target, Xavier
        ](ctx)
        t.critic_pair = OnlineTargetPair[Self.CRITIC].make[
            Self.train_target, Xavier
        ](ctx)

        t.actor_opt = Adam(lr=actor_lr)
        t.critic_opt = Adam(lr=critic_lr)
        comptime if Self.train_target == "gpu":
            t.actor_opt.adopt[Self.train_target, Self.ACTOR](
                t.actor_pair.online, ctx
            )
            t.critic_opt.adopt[Self.train_target, Self.CRITIC](
                t.critic_pair.online, ctx
            )

        t.target_y_blk = DDPGTargetYBlock[
            Self.ACTOR, Self.CRITIC, Self.BATCH, Self.OBS_DIM, Self.ACT_DIM,
        ].make[Self.train_target](
            action_scale=action_scale, gamma=gamma, ctx=ctx
        )
        t.critic_blk = SingleCriticStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.CRITIC,
        ].make[Self.train_target](ctx=ctx)
        t.actor_blk = DDPGActorStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.ACTOR, Self.CRITIC,
        ].make[Self.train_target](ctx=ctx)
        t.polyak_blk = DDPGPolyakStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.ACTOR, Self.CRITIC,
        ].make(tau=tau)

        t.state = TrainerState[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH
        ].make[Self.train_target](ctx=ctx)

        t.tracker = EpisodeTracker.new(
            window_size=window_size, initial_fill=initial_episode_fill
        )

        t.action_scale = action_scale
        t.noise_scale = noise_scale
        t.learning_starts = learning_starts

        t.sample_blk.setup(learning_starts, ctx=ctx)

        # Pre-size the action scratch for the single-env path.
        comptime if Self.train_target == "cpu":
            t._ob_scr.ensure(Self.OBS_DIM)
            t._ao_scr.ensure(Self.ACT_DIM)
            t._noise_scr.ensure(Self.ACT_DIM)
        else:
            # Device-resident diagnostic accumulators (no per-step D2H).
            t._mean_q_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)
            t._mean_target_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)
            t._mean_reward_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)
        return t^

    def set_beta(mut self, beta: Scalar[DT]):
        """PER IS-β anneal hook. No-op for uniform sample blocks."""
        self.sample_blk.set_beta(beta)

    # ─── train_step ────────────────────────────────────────────────────
    def train_step(mut self, step_idx: Int) raises -> Bool:
        self.state.step_idx = step_idx
        self.state.did_step = True
        comptime if Self.train_target == "gpu":
            self.state.ctx = self.ctx

        self.sample_blk.step(self.state)
        if not self.state.did_step:
            return False

        self.target_y_blk.step[Self.train_target](
            self.state, self.actor_pair.target_net, self.critic_pair.target_net
        )
        # ACCUMULATE on GPU: the per-batch critic loss is reduced on-device
        # into the critic's MSELoss accumulator (read at flush) — NO per-step
        # D2H. CPU keeps the host scalar.
        self.critic_blk.step[
            Self.train_target, ACCUMULATE = Self.train_target == "gpu"
        ](
            self.state,
            self.critic_pair.online,
            self.critic_opt,
        )
        self.actor_blk.step[Self.train_target](
            self.state,
            self.actor_pair.online,
            self.actor_opt,
            self.critic_pair.online,
        )
        self.polyak_blk.step[Self.train_target](
            self.state, self.actor_pair, self.critic_pair
        )

        # PER tail (no-op for uniform blocks).
        self.sample_blk.update_priorities(self.state)

        # Host bookkeeping.
        self._actor_L_accum += self.state.actor_loss
        self._critic_L_accum += self.state.critic_loss
        # Diagnostic batch means. CPU sums the host scratches directly; GPU
        # folds each batch mean into a device-resident accumulator via a tiny
        # reduction kernel — NO per-step D2H, read once per flush.
        comptime B = Self.BATCH
        comptime if Self.train_target == "cpu":
            var inv_b = Scalar[DT](1.0) / Scalar[DT](B)
            var sq: Scalar[DT] = 0.0
            var sy: Scalar[DT] = 0.0
            var sr: Scalar[DT] = 0.0
            for b in range(B):
                sq += self.critic_blk.inner._mb_q.data[b]
                sy += self.state.mb_y.data[b]
                sr += self.state.mb_r.data[b]
            self._mean_q_accum += sq * inv_b
            self._mean_target_accum += sy * inv_b
            self._mean_reward_accum += sr * inv_b
        else:
            comptime lb = Layout.row_major(B)
            self._mean_q_dev.accumulate_gpu_lt[B](
                self.critic_blk.inner._mb_q.lt["gpu", lb]()
            )
            self._mean_target_dev.accumulate_gpu_lt[B](
                self.state.mb_y.lt["gpu", lb]()
            )
            self._mean_reward_dev.accumulate_gpu_lt[B](
                self.state.mb_r.lt["gpu", lb]()
            )
        self._update_count += 1
        self._total_train_steps += 1
        return True

    def total_train_steps(self) -> Int:
        return self._total_train_steps

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

        # ── Policy: shared deterministic-actor body (see
        # training/blocks/action_select.mojo — one copy for ddpg + td3).
        var sigma = self.noise_scale * self.action_scale
        select_deterministic_batched[
            Self.ACTOR, Self.train_target, N_ENVS, OBS, ACT
        ](
            self.actor_pair.online,
            self._ob_scr,
            self._ao_scr,
            self._noise_scr,
            obs,
            action,
            sigma,
            self.action_scale,
            self.ctx,
            self._noise_rng_seed,
            self._noise_rng_offset,
        )
        # silence unused warnings on the driver-owned scratch views.
        _ = ao_scratch
        _ = alp_scratch

    def select_greedy_action(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
    ) raises:
        """Greedy = deterministic actor output (Tanh-bounded), clamped to
        ±action_scale. No exploration noise, no extra squash (the actor
        already ends in Tanh — legacy parity)."""
        comptime ACT = Self.ACT_DIM
        comptime OBS = Self.OBS_DIM
        comptime if Self.train_target == "cpu":
            self._ob_scr.ensure(OBS)
            self._ao_scr.ensure(ACT)
            for d in range(OBS):
                self._ob_scr.data[d] = obs[d]
            call_forward["cpu", 1](
                self.actor_pair.online,
                TensorRefs[Self.ACTOR.ARITY](self._ob_scr), self._ao_scr,
            )
            for j in range(ACT):
                var a = self._ao_scr.data[j]
                if a > self.action_scale:
                    a = self.action_scale
                elif a < -self.action_scale:
                    a = -self.action_scale
                action_out[j] = a
        else:
            # Fresh single-env Tensors (the batched scratch is sized for
            # N_ENVS; reusing it here would walk past the 1-env fill).
            var c = self.ctx.value()
            var ob = Tensor.alloc(OBS)
            for d in range(OBS):
                ob.data[d] = obs[d]
            ob.upload(c)
            var ao = Tensor.alloc_gpu(c, ACT)
            call_forward["gpu", 1](
                self.actor_pair.online, TensorRefs[Self.ACTOR.ARITY](ob), ao,
                self.ctx,
            )
            ao.download(c)
            for j in range(ACT):
                var a = ao.data[j]
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
        """Host-list deterministic action + Gaussian exploration noise —
        user-facing entry for smoke tests that bypass the driver."""
        comptime ACT = Self.ACT_DIM
        comptime OBS = Self.OBS_DIM
        if step_idx < self.learning_starts:
            for j in range(ACT):
                var u = Scalar[DT](2.0 * random_float64() - 1.0)
                action_out[j] = u * self.action_scale
            return
        var sigma = self.noise_scale * self.action_scale
        comptime if Self.train_target == "cpu":
            self._ob_scr.ensure(OBS)
            self._ao_scr.ensure(ACT)
            self._noise_scr.ensure(ACT)
            for d in range(OBS):
                self._ob_scr.data[d] = obs[d]
            call_forward["cpu", 1](
                self.actor_pair.online,
                TensorRefs[Self.ACTOR.ARITY](self._ob_scr), self._ao_scr,
            )
            box_muller_normal(self._noise_scr.data.unsafe_ptr(), ACT)
            for j in range(ACT):
                var a = self._ao_scr.data[j] + self._noise_scr.data[j] * sigma
                if a > self.action_scale:
                    a = self.action_scale
                elif a < -self.action_scale:
                    a = -self.action_scale
                action_out[j] = a
        else:
            var c = self.ctx.value()
            var ob = Tensor.alloc(OBS)
            for d in range(OBS):
                ob.data[d] = obs[d]
            ob.upload(c)
            var ao = Tensor.alloc_gpu(c, ACT)
            call_forward["gpu", 1](
                self.actor_pair.online, TensorRefs[Self.ACTOR.ARITY](ob), ao,
                self.ctx,
            )
            ao.download(c)
            # Host box-muller noise (matches the host-list CPU path).
            var noise = List[Scalar[DT]](length=ACT, fill=Scalar[DT](0.0))
            box_muller_normal(noise.unsafe_ptr(), ACT)
            for j in range(ACT):
                var a = ao.data[j] + noise[j] * sigma
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

    # ─── GPU-batched record surface ────────────────────────────────────
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
    ) raises -> DDPGMetrics:
        """Drain accumulators into a DDPGMetrics bundle. The per-batch
        diagnostics (mean_q / mean_target / mean_reward) are real on BOTH
        targets: CPU reads the host accumulators (averaged over the window);
        GPU reads the device-resident `DeviceMeanAccum`s with ONE D2H each at
        this flush — never in the per-step hot loop."""
        var n = self._update_count if self._update_count > 0 else 1
        var inv = Scalar[DT](1.0) / Scalar[DT](n)
        var actor_val: Scalar[DT]
        var critic_val: Scalar[DT]
        var mq: Scalar[DT]
        var mtgt: Scalar[DT]
        var mr: Scalar[DT]
        comptime if Self.train_target == "gpu":
            actor_val = self.actor_blk.read_loss_accum(self.ctx.value())
            critic_val = self.critic_blk.inner.mse_loss.read_accum["gpu"](
                self.ctx
            )
            mq = self._mean_q_dev.read["gpu"]()
            mtgt = self._mean_target_dev.read["gpu"]()
            mr = self._mean_reward_dev.read["gpu"]()
        else:
            actor_val = self._actor_L_accum * inv
            critic_val = self._critic_L_accum * inv
            mq = self._mean_q_accum * inv
            mtgt = self._mean_target_accum * inv
            mr = self._mean_reward_accum * inv
        var bundle = DDPGMetrics(
            policy_loss=LogScalar[DT](actor_val),
            critic_loss=LogScalar[DT](critic_val),
            mean_q=LogScalar[DT](mq),
            mean_target=LogScalar[DT](mtgt),
            reward_mean=LogScalar[DT](mr),
            train_steps=LogScalar[DT](Scalar[DT](self._total_train_steps)),
            n_updates=LogScalar[DT](Scalar[DT](self._update_count)),
        )
        self._actor_L_accum = Scalar[DT](0.0)
        self._critic_L_accum = Scalar[DT](0.0)
        self._mean_q_accum = Scalar[DT](0.0)
        self._mean_target_accum = Scalar[DT](0.0)
        self._mean_reward_accum = Scalar[DT](0.0)
        comptime if Self.train_target == "gpu":
            self.critic_blk.inner.mse_loss.reset_accum["gpu"]()
            self.actor_blk.reset_loss_accum()
            self._mean_q_dev.reset["gpu"]()
            self._mean_target_dev.reset["gpu"]()
            self._mean_reward_dev.reset["gpu"]()
        self._update_count = 0
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
    ) -> Tuple[Scalar[DT], Scalar[DT], Int]:
        """(mean_actor_loss, mean_critic_loss, n_updates) over the window.
        CPU host-scalar path; for GPU use `flush_metrics` (device accums)."""
        var n = self._update_count if self._update_count > 0 else 1
        var inv = Scalar[DT](1.0) / Scalar[DT](n)
        var out = (
            self._actor_L_accum * inv,
            self._critic_L_accum * inv,
            self._update_count,
        )
        self._actor_L_accum = Scalar[DT](0.0)
        self._critic_L_accum = Scalar[DT](0.0)
        self._mean_q_accum = Scalar[DT](0.0)
        self._mean_target_accum = Scalar[DT](0.0)
        self._mean_reward_accum = Scalar[DT](0.0)
        self._update_count = 0
        return out

    # ─── Checkpoint (ONE v3 file: actor + critic + train-step counter) ──
    def save_state(mut self, path: String) raises:
        """Write the ONLINE actor + critic into a SINGLE v3 `storage-ckpt`
        file, sections name-prefixed `actor.` / `critic.`, the train-step
        counter as a `K` scalar. Atomic and chunked. Optimizer moments NOT
        persisted (resume re-warms)."""
        var w = BinaryCheckpointWriter(save_moments=False)
        write_model[Self.train_target](w, self.actor_pair.online, self.ctx, "actor")
        write_model[Self.train_target](w, self.critic_pair.online, self.ctx, "critic")
        var sc = CheckpointScalars()
        sc.set_int("total_train_steps", self._total_train_steps)
        w.write_scalars(sc)
        _write_file_bytes(path, w.content)

    def load_state(mut self, path: String) raises:
        """Restore the online actor + critic (v3, or the legacy v2 text this
        trainer wrote before), then hard-copy online → target."""
        var bytes = _read_file_bytes(path)
        if _is_v3_header(bytes):
            var rb = BinaryCheckpointReader(bytes^)
            read_model[Self.train_target](rb, self.actor_pair.online, self.ctx, "actor")
            read_model[Self.train_target](rb, self.critic_pair.online, self.ctx, "critic")
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
            walk_params[Self.train_target](self.actor_pair.online, r, self.ctx, "actor"
            )
            walk_params[Self.train_target](self.critic_pair.online, r, self.ctx, "critic"
            )
            r.mode = 1
            self.actor_pair.online.for_each_state[Self.train_target](
                r, self.ctx, "actor"
            )
            self.critic_pair.online.for_each_state[Self.train_target](
                r, self.ctx, "critic"
            )
        self.actor_pair.target_net.polyak_from[Self.train_target](
            self.actor_pair.online, Scalar[DT](1.0), self.ctx
        )
        self.critic_pair.target_net.polyak_from[Self.train_target](
            self.critic_pair.online, Scalar[DT](1.0), self.ctx
        )

    def flush_timer_log(mut self) -> String:
        return String("")
