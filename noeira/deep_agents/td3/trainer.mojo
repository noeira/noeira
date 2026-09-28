"""TD3Trainer — storage-framework TD3 trainer (CPU gate; GPU stretch).

Assembles the migrated `noeira.nn` TD3 blocks into a single
driver-conforming `TD3Trainer` that conforms `OffPolicyAgentGpu` so both the
CPU single-env driver (`run_offpolicy_train`) and the batched drivers
type-check. Structural sibling of the storage `DDPGTrainer` with the TD3
additions:

  * TWIN critics (`pair1`, `pair2` + the shared `TwinCriticStep`, also used by
    SAC) — the target value takes min(Q1',Q2').
  * `TD3TargetYBlock` — target-policy smoothing (clipped Gaussian noise on the
    target action) + twin-critic min.
  * `TD3DelayedActorPolyakStep` — the actor update (DPG on critic1, reusing
    storage `DDPGActorLoss`) + ALL THREE polyaks (actor + both critics), gated
    by an internal `policy_delay` counter.

Pipeline (per train step):
  state.step_idx = step
  sample_blk.step(state)                                          # fills mb_*
  target_y_blk.step(state, actor_t, critic1_t, critic2_t)         # writes mb_y
  twin_critic_blk.step[ACCUMULATE=(gpu)](state, c1, c1_opt, c2, c2_opt)
  actor_polyak_blk.step(state, actor_opt, actor_pair, pair1, pair2)  # delayed
  diagnostics

Deterministic action selection with additive Gaussian exploration noise (NO
rsample); the Tanh-bounded actor output is fed raw to the env (clamped to
±action_scale), exactly like the storage DDPG trainer.

CUDA-graph capture (`train_device_kernels` / `note_train_update` /
`learning_starts_count`) is DEFERRED — the `OffPolicyAgentGpu` trait defaults
raise, never reached with `USE_TRAIN_CUDA_GRAPH=False`.
"""

from noeira.nn.core.param import walk_params
from std.random import random_float64
from std.random.philox import Random as PhiloxRandom
from layout import Layout, LayoutTensor

from max.gpu import global_idx
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.core.logger import Logger, NoOpLogger
from noeira.nn.constants import DT, TPB
from noeira.nn.core.module import Module
from noeira.nn.core.amp import AMPPolicy, NoAMP, Bf16Compute
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
from ..training.blocks import SampleBlock, TwinCriticStep
from ..training.blocks.action_select import (
    select_deterministic_batched,
    warmup_uniform_batched,
)
from .target_y_block import TD3TargetYBlock
from .blocks.delayed_actor_polyak_step import TD3DelayedActorPolyakStep
from .metrics import TD3Metrics


# ──────────────────────────────────────────────────────────────────────
# GPU device kernels for the batched action-selection path (mirror the
# storage DDPG trainer's body: Philox warmup + obs copy + noise+clamp,
# deterministic actor + additive Gaussian, no rsample).
# ──────────────────────────────────────────────────────────────────────


struct TD3Trainer[
    train_target: StaticString,
    SAMPLE: SampleBlock,
    ACTOR: Module,
    CRITIC: Module,
](OffPolicyAgentGpu):
    """Storage-framework TD3 trainer. Dimensions (OBS / ACT / BATCH) are
    derived from SAMPLE (mirrors SAC/DDPG)."""

    comptime OBS_DIM: Int = Self.SAMPLE.OBS
    comptime ACT_DIM: Int = Self.SAMPLE.ACT
    comptime BATCH: Int = Self.SAMPLE.BATCH

    comptime AGENT_OBS_DIM: Int = Self.OBS_DIM
    comptime AGENT_ACT_DIM: Int = Self.ACT_DIM
    comptime AGENT_TRAIN_TARGET: StaticString = Self.train_target

    var actor_pair: OnlineTargetPair[Self.ACTOR]
    var pair1: OnlineTargetPair[Self.CRITIC]
    var pair2: OnlineTargetPair[Self.CRITIC]
    var actor_opt: Adam
    var critic1_opt: Adam
    var critic2_opt: Adam

    var sample_blk: Self.SAMPLE
    var target_y_blk: TD3TargetYBlock[
        Self.ACTOR, Self.CRITIC, Self.BATCH, Self.OBS_DIM, Self.ACT_DIM,
    ]
    var twin_critic_blk: TwinCriticStep[
        Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.CRITIC,
    ]
    var actor_polyak_blk: TD3DelayedActorPolyakStep[
        Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.ACTOR, Self.CRITIC,
    ]

    var state: TrainerState[Self.OBS_DIM, Self.ACT_DIM, Self.BATCH]
    var tracker: EpisodeTracker
    var ctx: Optional[DeviceContext]

    # Owned action-selection scratch Tensors (lazily `.ensure`d per call).
    var _ob_scr: Tensor     # N_ENVS * OBS
    var _ao_scr: Tensor     # N_ENVS * ACT (deterministic actor output)
    var _noise_scr: Tensor  # N_ENVS * ACT (box-muller fill)

    var action_scale: Scalar[DT]
    var exploration_noise: Scalar[DT]
    var learning_starts: Int

    # Philox state for batched warmup + exploration noise (gpu path only).
    var _warmup_rng_seed: UInt64
    var _warmup_rng_offset: UInt64
    var _noise_rng_seed: UInt64
    var _noise_rng_offset: UInt64

    # Host metric accumulators (CPU path).
    var _actor_L_accum: Scalar[DT]
    var _critic_L_accum: Scalar[DT]
    var _actor_updates: Int
    var _critic_updates: Int
    var _total_train_steps: Int
    # Diagnostic accumulators (CPU path): batch means drained at flush.
    var _q_accum: Scalar[DT]
    var _target_accum: Scalar[DT]
    var _reward_accum: Scalar[DT]
    var _done_accum: Scalar[DT]
    # Diagnostic accumulators (GPU path): device-resident running means.
    var _q_mean_dev: DeviceMeanAccum
    var _target_mean_dev: DeviceMeanAccum
    var _reward_mean_dev: DeviceMeanAccum
    var _done_mean_dev: DeviceMeanAccum
    var _use_bf16: Bool

    def __init__(out self):
        self.actor_pair = OnlineTargetPair[Self.ACTOR]()
        self.pair1 = OnlineTargetPair[Self.CRITIC]()
        self.pair2 = OnlineTargetPair[Self.CRITIC]()
        self.actor_opt = Adam(lr=Scalar[DT](3e-4))
        self.critic1_opt = Adam(lr=Scalar[DT](3e-4))
        self.critic2_opt = Adam(lr=Scalar[DT](3e-4))
        self.sample_blk = Self.SAMPLE()
        self.target_y_blk = TD3TargetYBlock[
            Self.ACTOR, Self.CRITIC, Self.BATCH, Self.OBS_DIM, Self.ACT_DIM,
        ]()
        self.twin_critic_blk = TwinCriticStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.CRITIC,
        ]()
        self.actor_polyak_blk = TD3DelayedActorPolyakStep[
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
        self.exploration_noise = Scalar[DT](0.1)
        self.learning_starts = 1_000
        self._warmup_rng_seed = UInt64(0xC0FFEE_C0DE)
        self._warmup_rng_offset = UInt64(0)
        self._noise_rng_seed = UInt64(0xD15EA5E_D00D)
        self._noise_rng_offset = UInt64(0)
        self._actor_L_accum = Scalar[DT](0.0)
        self._critic_L_accum = Scalar[DT](0.0)
        self._actor_updates = 0
        self._critic_updates = 0
        self._total_train_steps = 0
        self._q_accum = Scalar[DT](0.0)
        self._target_accum = Scalar[DT](0.0)
        self._reward_accum = Scalar[DT](0.0)
        self._done_accum = Scalar[DT](0.0)
        self._q_mean_dev = DeviceMeanAccum()
        self._target_mean_dev = DeviceMeanAccum()
        self._reward_mean_dev = DeviceMeanAccum()
        self._done_mean_dev = DeviceMeanAccum()
        self._use_bf16 = False

    @staticmethod
    def make(
        ctx: Optional[DeviceContext] = None,
        actor_lr: Scalar[DT] = Scalar[DT](3e-4),
        critic_lr: Scalar[DT] = Scalar[DT](3e-4),
        gamma: Scalar[DT] = Scalar[DT](0.99),
        tau: Scalar[DT] = Scalar[DT](0.005),
        action_scale: Scalar[DT] = Scalar[DT](1.0),
        exploration_noise: Scalar[DT] = Scalar[DT](0.1),
        target_policy_noise: Scalar[DT] = Scalar[DT](0.2),
        target_noise_clip: Scalar[DT] = Scalar[DT](0.5),
        policy_delay: Int = 2,
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
        ), "TD3Trainer: target must be 'cpu' or 'gpu'"
        comptime if Self.train_target == "gpu":
            if not ctx:
                raise Error("TD3Trainer.make[target='gpu']: ctx required")

        var t = Self()
        t.ctx = ctx

        t.actor_pair = OnlineTargetPair[Self.ACTOR].make[
            Self.train_target, Xavier
        ](ctx)
        t.pair1 = OnlineTargetPair[Self.CRITIC].make[
            Self.train_target, Xavier
        ](ctx)
        t.pair2 = OnlineTargetPair[Self.CRITIC].make[
            Self.train_target, Xavier
        ](ctx)

        t.actor_opt = Adam(lr=actor_lr)
        t.critic1_opt = Adam(lr=critic_lr)
        t.critic2_opt = Adam(lr=critic_lr)
        comptime if Self.train_target == "gpu":
            t.actor_opt.adopt[Self.train_target, Self.ACTOR](
                t.actor_pair.online, ctx
            )
            t.critic1_opt.adopt[Self.train_target, Self.CRITIC](
                t.pair1.online, ctx
            )
            t.critic2_opt.adopt[Self.train_target, Self.CRITIC](
                t.pair2.online, ctx
            )

        t.target_y_blk = TD3TargetYBlock[
            Self.ACTOR, Self.CRITIC, Self.BATCH, Self.OBS_DIM, Self.ACT_DIM,
        ].make[Self.train_target](
            action_scale=action_scale, gamma=gamma,
            noise_std=target_policy_noise, noise_clip=target_noise_clip,
            ctx=ctx,
        )
        t.twin_critic_blk = TwinCriticStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.CRITIC,
        ].make[Self.train_target](ctx=ctx)
        t.actor_polyak_blk = TD3DelayedActorPolyakStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.ACTOR, Self.CRITIC,
        ].make[Self.train_target](
            policy_delay=policy_delay, tau=tau, ctx=ctx,
        )

        t.state = TrainerState[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH
        ].make[Self.train_target](ctx=ctx)

        t.tracker = EpisodeTracker.new(
            window_size=window_size, initial_fill=initial_episode_fill
        )

        t.action_scale = action_scale
        t.exploration_noise = exploration_noise
        t.learning_starts = learning_starts
        t._use_bf16 = use_bf16

        t.sample_blk.setup(learning_starts, ctx=ctx)

        comptime if Self.train_target == "cpu":
            t._ob_scr.ensure(Self.OBS_DIM)
            t._ao_scr.ensure(Self.ACT_DIM)
            t._noise_scr.ensure(Self.ACT_DIM)
        else:
            t._q_mean_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)
            t._target_mean_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)
            t._reward_mean_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)
            t._done_mean_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)
        return t^

    def set_beta(mut self, beta: Scalar[DT]):
        """PER IS-β anneal hook. No-op for uniform sample blocks."""
        self.sample_blk.set_beta(beta)

    # ─── train_step ────────────────────────────────────────────────────
    def train_step(mut self, step_idx: Int) raises -> Bool:
        comptime if Self.train_target == "cpu":
            return self._train_step_impl[NoAMP](step_idx)
        else:
            if self._use_bf16:
                return self._train_step_impl[Bf16Compute](step_idx)
            return self._train_step_impl[NoAMP](step_idx)

    def _train_step_impl[
        POLICY: AMPPolicy = NoAMP,
    ](mut self, step_idx: Int) raises -> Bool:
        self.state.step_idx = step_idx
        self.state.did_step = True
        comptime if Self.train_target == "gpu":
            self.state.ctx = self.ctx

        self.sample_blk.step(self.state)
        if not self.state.did_step:
            return False

        self.target_y_blk.step[Self.train_target, POLICY](
            self.state,
            self.actor_pair.target_net,
            self.pair1.target_net,
            self.pair2.target_net,
        )
        # ACCUMULATE on GPU: the per-batch twin-critic loss is reduced on-device
        # into the critics' accumulators (read at flush) — NO per-step D2H.
        self.twin_critic_blk.step[
            Self.train_target, POLICY, ACCUMULATE = Self.train_target == "gpu"
        ](
            self.state,
            self.pair1.online,
            self.critic1_opt,
            self.pair2.online,
            self.critic2_opt,
        )
        self._critic_L_accum += self.state.critic_loss
        self._critic_updates += 1

        # Per-batch diagnostics — `mean_q` reads twin_critic c1's Q1(s, a);
        # target/reward/done read the minibatch scratches. Fires every step
        # (critic cadence).
        comptime B = Self.BATCH
        comptime if Self.train_target == "cpu":
            var inv_b = Scalar[DT](1.0) / Scalar[DT](B)
            var sq: Scalar[DT] = 0.0
            var sy: Scalar[DT] = 0.0
            var sr: Scalar[DT] = 0.0
            var sd: Scalar[DT] = 0.0
            for b in range(B):
                sq += self.twin_critic_blk.inner.c1._mb_q.data[b]
                sy += self.state.mb_y.data[b]
                sr += self.state.mb_r.data[b]
                sd += self.state.mb_d.data[b]
            self._q_accum += sq * inv_b
            self._target_accum += sy * inv_b
            self._reward_accum += sr * inv_b
            self._done_accum += sd * inv_b
        else:
            comptime lb = Layout.row_major(B)
            self._q_mean_dev.accumulate_gpu_lt[B](
                self.twin_critic_blk.inner.c1._mb_q.lt["gpu", lb]()
            )
            self._target_mean_dev.accumulate_gpu_lt[B](
                self.state.mb_y.lt["gpu", lb]()
            )
            self._reward_mean_dev.accumulate_gpu_lt[B](
                self.state.mb_r.lt["gpu", lb]()
            )
            self._done_mean_dev.accumulate_gpu_lt[B](
                self.state.mb_d.lt["gpu", lb]()
            )

        # TD3 actor + 3-pair polyak (gated by internal counter). Block reads
        # actor via actor_pair.online + critic1 via pair1.online internally.
        self.actor_polyak_blk.step[Self.train_target, POLICY](
            self.state,
            self.actor_opt,
            self.actor_pair,
            self.pair1,
            self.pair2,
        )
        # Block resets _counter to 0 when it fires; the actor loss is valid
        # (CPU) / accumulated on-device (GPU) only then.
        if self.actor_polyak_blk._counter == 0:
            self._actor_L_accum += self.state.actor_loss
            self._actor_updates += 1

        # PER tail (no-op for uniform blocks).
        self.sample_blk.update_priorities(self.state)

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
        var sigma = self.exploration_noise * self.action_scale
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
        ±action_scale. No exploration noise (the actor already ends in Tanh)."""
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
        user-facing entry for smoke tests / the single-env driver."""
        comptime ACT = Self.ACT_DIM
        comptime OBS = Self.OBS_DIM
        if step_idx < self.learning_starts:
            for j in range(ACT):
                var u = Scalar[DT](2.0 * random_float64() - 1.0)
                action_out[j] = u * self.action_scale
            return
        var sigma = self.exploration_noise * self.action_scale
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
    ) raises -> TD3Metrics:
        """Drain accumulators into a TD3Metrics bundle. The per-batch
        diagnostics are real on BOTH targets: CPU reads the host accumulators
        (averaged over the critic-update window); GPU reads the device-resident
        `DeviceMeanAccum`s with ONE D2H each at this flush."""
        var na = self._actor_updates if self._actor_updates > 0 else 1
        var nc = self._critic_updates if self._critic_updates > 0 else 1
        var inv_a = Scalar[DT](1.0) / Scalar[DT](na)
        var inv_c = Scalar[DT](1.0) / Scalar[DT](nc)
        var actor_mean: Scalar[DT]
        var critic_mean: Scalar[DT]
        var q_mean: Scalar[DT]
        var target_mean: Scalar[DT]
        var reward_mean: Scalar[DT]
        var done_mean: Scalar[DT]
        comptime if Self.train_target == "gpu":
            actor_mean = self.actor_polyak_blk.read_loss_accum(self.ctx.value())
            var cl1 = self.twin_critic_blk.inner.c1.mse_loss.read_accum["gpu"](
                self.ctx
            )
            var cl2 = self.twin_critic_blk.inner.c2.mse_loss.read_accum["gpu"](
                self.ctx
            )
            critic_mean = cl1 + cl2
            q_mean = self._q_mean_dev.read["gpu"]()
            target_mean = self._target_mean_dev.read["gpu"]()
            reward_mean = self._reward_mean_dev.read["gpu"]()
            done_mean = self._done_mean_dev.read["gpu"]()
        else:
            actor_mean = self._actor_L_accum * inv_a
            critic_mean = self._critic_L_accum * inv_c
            q_mean = self._q_accum * inv_c
            target_mean = self._target_accum * inv_c
            reward_mean = self._reward_accum * inv_c
            done_mean = self._done_accum * inv_c
        var bundle = TD3Metrics(
            policy_loss=LogScalar[DT](actor_mean),
            critic_loss=LogScalar[DT](critic_mean),
            mean_q=LogScalar[DT](q_mean),
            mean_target=LogScalar[DT](target_mean),
            reward_mean=LogScalar[DT](reward_mean),
            mean_done=LogScalar[DT](done_mean),
            train_steps=LogScalar[DT](Scalar[DT](self._total_train_steps)),
            n_actor_updates=LogScalar[DT](Scalar[DT](self._actor_updates)),
            n_critic_updates=LogScalar[DT](Scalar[DT](self._critic_updates)),
        )
        self._actor_L_accum = Scalar[DT](0.0)
        self._critic_L_accum = Scalar[DT](0.0)
        self._actor_updates = 0
        self._critic_updates = 0
        self._q_accum = Scalar[DT](0.0)
        self._target_accum = Scalar[DT](0.0)
        self._reward_accum = Scalar[DT](0.0)
        self._done_accum = Scalar[DT](0.0)
        comptime if Self.train_target == "gpu":
            self.twin_critic_blk.inner.c1.mse_loss.reset_accum["gpu"]()
            self.twin_critic_blk.inner.c2.mse_loss.reset_accum["gpu"]()
            self.actor_polyak_blk.reset_loss_accum()
            self._q_mean_dev.reset["gpu"]()
            self._target_mean_dev.reset["gpu"]()
            self._reward_mean_dev.reset["gpu"]()
            self._done_mean_dev.reset["gpu"]()
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
    ) -> Tuple[Scalar[DT], Scalar[DT], Int, Int]:
        """(mean_actor_loss, mean_critic_loss, n_actor_updates,
        n_critic_updates) over the window. CPU host-scalar path."""
        var na = self._actor_updates if self._actor_updates > 0 else 1
        var nc = self._critic_updates if self._critic_updates > 0 else 1
        var inv_a = Scalar[DT](1.0) / Scalar[DT](na)
        var inv_c = Scalar[DT](1.0) / Scalar[DT](nc)
        var out = (
            self._actor_L_accum * inv_a,
            self._critic_L_accum * inv_c,
            self._actor_updates,
            self._critic_updates,
        )
        self._actor_L_accum = Scalar[DT](0.0)
        self._critic_L_accum = Scalar[DT](0.0)
        self._actor_updates = 0
        self._critic_updates = 0
        self._q_accum = Scalar[DT](0.0)
        self._target_accum = Scalar[DT](0.0)
        self._reward_accum = Scalar[DT](0.0)
        self._done_accum = Scalar[DT](0.0)
        return out

    # ─── Checkpoint (ONE v3 file: actor + twin critics + train-step counter) ───
    def save_state(mut self, path: String) raises:
        """ONE v3 `storage-ckpt` file: the ONLINE nets, name-prefixed
        `actor.` / `critic1.` / `critic2.`; training scalars as `K` sections.
        Atomic and chunked. Optimizer moments NOT persisted (resume re-warms)."""
        var w = BinaryCheckpointWriter(save_moments=False)
        write_model[Self.train_target](w, self.actor_pair.online, self.ctx, "actor")
        write_model[Self.train_target](w, self.pair1.online, self.ctx, "critic1")
        write_model[Self.train_target](w, self.pair2.online, self.ctx, "critic2")
        var sc = CheckpointScalars()
        sc.set_int("total_train_steps", self._total_train_steps)
        w.write_scalars(sc)
        _write_file_bytes(path, w.content)

    def load_state(mut self, path: String) raises:
        """Restore the online nets (v3, or the legacy v2 text this trainer
        wrote before) and the training scalars, then hard-copy online →
        target."""
        var bytes = _read_file_bytes(path)
        if _is_v3_header(bytes):
            var rb = BinaryCheckpointReader(bytes^)
            read_model[Self.train_target](rb, self.actor_pair.online, self.ctx, "actor")
            read_model[Self.train_target](rb, self.pair1.online, self.ctx, "critic1")
            read_model[Self.train_target](rb, self.pair2.online, self.ctx, "critic2")
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
            walk_params[Self.train_target](self.pair1.online, r, self.ctx, "critic1"
            )
            walk_params[Self.train_target](self.pair2.online, r, self.ctx, "critic2"
            )
            r.mode = 1
            self.actor_pair.online.for_each_state[Self.train_target](
                r, self.ctx, "actor"
            )
            self.pair1.online.for_each_state[Self.train_target](
                r, self.ctx, "critic1"
            )
            self.pair2.online.for_each_state[Self.train_target](
                r, self.ctx, "critic2"
            )
        self.actor_pair.target_net.polyak_from[Self.train_target](
            self.actor_pair.online, Scalar[DT](1.0), self.ctx
        )
        self.pair1.target_net.polyak_from[Self.train_target](
            self.pair1.online, Scalar[DT](1.0), self.ctx
        )
        self.pair2.target_net.polyak_from[Self.train_target](
            self.pair2.online, Scalar[DT](1.0), self.ctx
        )

    def flush_timer_log(mut self) -> String:
        return String("")
