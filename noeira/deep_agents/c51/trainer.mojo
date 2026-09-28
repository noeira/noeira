"""C51Trainer — distributional DQN trainer (Bellemare et al. 2017) (STORAGE).

Pipeline body mirrors `dqn/trainer.mojo`: sample → target-Y (categorical
projection) → q-update (cross-entropy) → polyak. Differences from DQN:

  - `Q_NET.OUT_DIM == NA · N_ATOMS` (per-atom logits instead of Q-values).
  - Target is a distribution `m [B, N_ATOMS]` — the trainer owns its own `_mb_m`
    Tensor instead of using `state.mb_y`.
  - Action selection picks `argmax_a Σ_k softmax(logits[b, a])_k · z_k`
    (expected Q from the distribution), not a plain argmax over Q.

STORAGE migration (Stage 5): mirrors the storage DQN trainer — `Scratch`/
`TargetStorage`/`init_scratch_auto`/`mptr`/TileTensor/legacy-checkpoint gone;
storage Module/Adam/CrossEntropyLoss; the driver's raw obs/action pointers are
bridged into owned Tensor scratch around the storage Module.forward; checkpoint
via a v3 checkpoint (params+state, ε+counter as `K` scalars; moments not
persisted). The distributional diag kernel takes LayoutTensor views (no
unsafe_ptr). CPU + GPU; CUDA-graph capture surface preserved.

Conforms to `OffPolicyDiscreteAgentGpu`.
"""

from noeira.nn.core.param import walk_params, ParamVisitorRef
from std.math import exp as fexp, log as flog
from std.random import random_float64
from max.gpu import global_idx
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor

from noeira.core.logger import Logger, NoOpLogger
from noeira.nn.constants import DT, TPB
from noeira.nn.core.module import Module
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.amp import AMPPolicy, NoAMP
from noeira.nn.core.call import call_forward
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

from ..core.online_target_pair import OnlineTargetPair
from ..training.episode_tracker import EpisodeTracker
from ..training.device_mean_accum import DeviceMeanAccum
from ..training.trainer_block import TrainerState
from ..training.driver_offpolicy_discrete import (
    OffPolicyDiscreteAgent,
    OffPolicyDiscreteAgentGpu,
)
from ..training.blocks import SampleBlock, SinglePolyakStep
from ..data.n_step_replay import GPUNStepBuffer
from .target_y_block import C51TargetYBlock
from .blocks.q_update_step import C51QUpdateStep
from .metrics import C51Metrics


# ──────────────────────────────────────────────────────────────────────────
# GPU per-sample distributional diag kernel (LayoutTensor; one thread per row).
# Softmax the taken-action logit row → expected Q (Σ p_k z_k), entropy
# (−Σ p_k log p_k), target expected value (Σ m_k z_k) into three [BATCH]
# buffers; the trainer reduces them via DeviceMeanAccum. Mirrors the CPU walk.
# ──────────────────────────────────────────────────────────────────────────
def _c51_diag_kernel[BATCH: Int, NK: Int](
    logits: LayoutTensor[DT, Layout.row_major(BATCH, NK), MutAnyOrigin],
    m: LayoutTensor[DT, Layout.row_major(BATCH, NK), MutAnyOrigin],
    z: LayoutTensor[DT, Layout.row_major(NK), MutAnyOrigin],
    eq_out: LayoutTensor[DT, Layout.row_major(BATCH), MutAnyOrigin],
    ent_out: LayoutTensor[DT, Layout.row_major(BATCH), MutAnyOrigin],
    tq_out: LayoutTensor[DT, Layout.row_major(BATCH), MutAnyOrigin],
):
    var b = Int(global_idx.x)
    if b >= BATCH:
        return
    var maxl = rebind[Scalar[DT]](logits[b, 0])
    for k in range(1, NK):
        var v = rebind[Scalar[DT]](logits[b, k])
        if v > maxl:
            maxl = v
    var sum_exp: Scalar[DT] = 0.0
    for k in range(NK):
        sum_exp += fexp(rebind[Scalar[DT]](logits[b, k]) - maxl)
    var eq: Scalar[DT] = 0.0
    var ent: Scalar[DT] = 0.0
    var tq: Scalar[DT] = 0.0
    for k in range(NK):
        var p = fexp(rebind[Scalar[DT]](logits[b, k]) - maxl) / sum_exp
        eq += p * rebind[Scalar[DT]](z[k])
        if p > Scalar[DT](1e-12):
            ent -= p * flog(p)
        tq += rebind[Scalar[DT]](m[b, k]) * rebind[Scalar[DT]](z[k])
    eq_out[b] = eq
    ent_out[b] = ent
    tq_out[b] = tq


struct C51Trainer[
    train_target: StaticString,
    SAMPLE: SampleBlock,
    Q_NET: Module,
    N_ATOMS: Int = 51,
    NUM_ACTIONS: Int = 2,
    DOUBLE: Bool = False,
](OffPolicyDiscreteAgentGpu):
    """Q_NET.OUT_DIM must equal NUM_ACTIONS · N_ATOMS (per-atom logits).
    Standard Rainbow defaults: N_ATOMS=51, V_min=-10, V_max=+10."""

    comptime OBS_DIM: Int = Self.SAMPLE.OBS
    comptime ACT_DIM: Int = Self.SAMPLE.ACT
    comptime BATCH: Int = Self.SAMPLE.BATCH

    comptime AGENT_TRAIN_TARGET: StaticString = Self.train_target
    comptime AGENT_OBS_DIM: Int = Self.OBS_DIM
    comptime AGENT_ACT_DIM: Int = Self.ACT_DIM
    comptime AGENT_NUM_ACTIONS: Int = Self.NUM_ACTIONS

    comptime _T_SAMPLE = 0
    comptime _T_TARGET_Y = 1
    comptime _T_CRITIC = 2
    comptime _T_POLYAK = 3
    comptime _T_DIAG = 4

    var pair: OnlineTargetPair[Self.Q_NET]
    var q_opt: Adam
    var sample_blk: Self.SAMPLE
    var target_y_blk: C51TargetYBlock[
        Self.Q_NET, Self.BATCH, Self.OBS_DIM, Self.NUM_ACTIONS,
        Self.N_ATOMS, Self.DOUBLE,
    ]
    var q_update_blk: C51QUpdateStep[
        Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.NUM_ACTIONS,
        Self.N_ATOMS, Self.Q_NET,
    ]
    var polyak_blk: SinglePolyakStep[
        Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.Q_NET,
    ]

    var state: TrainerState[Self.OBS_DIM, Self.ACT_DIM, Self.BATCH]

    # Target distribution scratch — replaces `state.mb_y`'s [B,1] role.
    var _mb_m: Tensor  # [B · N_ATOMS]

    # Action-selection scratch (obs bridge + logits output, lazily sized).
    var _ob_scr: Tensor
    var _q_scr: Tensor
    var _act_host: List[Scalar[DT]]
    var _act_n: Int

    var tracker: EpisodeTracker
    var ctx: Optional[DeviceContext]

    var epsilon: Scalar[DT]
    var epsilon_decay: Scalar[DT]
    var epsilon_min: Scalar[DT]
    var learning_starts: Int

    var _action_list: List[Scalar[DT]]

    var _loss_accum: Scalar[DT]
    var _q_accum: Scalar[DT]
    var _target_accum: Scalar[DT]
    var _dist_entropy_accum: Scalar[DT]
    var _reward_accum: Scalar[DT]
    var _done_accum: Scalar[DT]
    var _q_mean_dev: DeviceMeanAccum
    var _target_mean_dev: DeviceMeanAccum
    var _dist_entropy_mean_dev: DeviceMeanAccum
    var _reward_mean_dev: DeviceMeanAccum
    var _done_mean_dev: DeviceMeanAccum
    # Per-sample diag scratch ([BATCH] eq/ent/tq filled by `_c51_diag_kernel`).
    var _diag_eq: Tensor
    var _diag_ent: Tensor
    var _diag_tq: Tensor
    var _update_count: Int
    var _total_train_steps: Int
    var timer: Timer

    def __init__(out self):
        self.pair = OnlineTargetPair[Self.Q_NET]()
        self.q_opt = Adam()
        self.sample_blk = Self.SAMPLE()
        self.target_y_blk = C51TargetYBlock[
            Self.Q_NET, Self.BATCH, Self.OBS_DIM, Self.NUM_ACTIONS,
            Self.N_ATOMS, Self.DOUBLE,
        ]()
        self.q_update_blk = C51QUpdateStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.NUM_ACTIONS,
            Self.N_ATOMS, Self.Q_NET,
        ]()
        self.polyak_blk = SinglePolyakStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.Q_NET,
        ]()
        self.state = TrainerState[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH,
        ]()
        self._mb_m = Tensor()
        self._ob_scr = Tensor()
        self._q_scr = Tensor()
        self._act_host = List[Scalar[DT]]()
        self._act_n = 0
        self.tracker = EpisodeTracker(
            window=List[Scalar[DT]](),
            window_size=0,
            idx=0,
            current_return=Scalar[DT](0.0),
            ep_count=0,
        )
        self.ctx = None
        self.epsilon = Scalar[DT](1.0)
        self.epsilon_decay = Scalar[DT](0.995)
        self.epsilon_min = Scalar[DT](0.01)
        self.learning_starts = 1_000
        self._action_list = List[Scalar[DT]](length=1, fill=Scalar[DT](0.0))
        self._loss_accum = Scalar[DT](0.0)
        self._q_accum = Scalar[DT](0.0)
        self._target_accum = Scalar[DT](0.0)
        self._dist_entropy_accum = Scalar[DT](0.0)
        self._reward_accum = Scalar[DT](0.0)
        self._done_accum = Scalar[DT](0.0)
        self._q_mean_dev = DeviceMeanAccum()
        self._target_mean_dev = DeviceMeanAccum()
        self._dist_entropy_mean_dev = DeviceMeanAccum()
        self._reward_mean_dev = DeviceMeanAccum()
        self._done_mean_dev = DeviceMeanAccum()
        self._diag_eq = Tensor()
        self._diag_ent = Tensor()
        self._diag_tq = Tensor()
        self._update_count = 0
        self._total_train_steps = 0
        self.timer = Timer.new()

    @staticmethod
    def make(
        ctx: Optional[DeviceContext] = None,
        lr: Scalar[DT] = Scalar[DT](1e-4),
        gamma: Scalar[DT] = Scalar[DT](0.99),
        tau: Scalar[DT] = Scalar[DT](0.005),
        epsilon: Scalar[DT] = Scalar[DT](1.0),
        epsilon_decay: Scalar[DT] = Scalar[DT](0.995),
        epsilon_min: Scalar[DT] = Scalar[DT](0.05),
        learning_starts: Int = 1_000,
        target_update_freq: Int = 500,
        window_size: Int = 10,
        initial_episode_fill: Scalar[DT] = Scalar[DT](0.0),
        max_grad_norm: Scalar[DT] = Scalar[DT](0.0),
        per_alpha: Scalar[DT] = Scalar[DT](0.6),
        per_beta: Scalar[DT] = Scalar[DT](0.4),
        per_epsilon: Scalar[DT] = Scalar[DT](1e-6),
        nstep: Int = 1,
        v_min: Scalar[DT] = Scalar[DT](-10.0),
        v_max: Scalar[DT] = Scalar[DT](10.0),
    ) raises -> Self:
        comptime assert (
            Self.train_target == "cpu" or Self.train_target == "gpu"
        ), "C51Trainer: train_target must be 'cpu' or 'gpu'"
        comptime assert (
            Self.ACT_DIM == 1
        ), "C51Trainer: SAMPLE.ACT must be 1 (discrete action index)"
        comptime assert (
            Self.Q_NET.IN_DIMS[0] == Self.OBS_DIM
        ), "C51Trainer: Q_NET.IN_DIM must equal SAMPLE.OBS"
        comptime assert (
            Self.Q_NET.OUT_DIM == Self.NUM_ACTIONS * Self.N_ATOMS
        ), "C51Trainer: Q_NET.OUT_DIM must equal NUM_ACTIONS · N_ATOMS"
        comptime if Self.train_target == "gpu":
            if not ctx:
                raise Error("C51Trainer.make[target='gpu']: ctx required")

        var t = Self()
        t.ctx = ctx
        t.epsilon = epsilon
        t.epsilon_decay = epsilon_decay
        t.epsilon_min = epsilon_min
        t.learning_starts = learning_starts

        t.pair = OnlineTargetPair[Self.Q_NET].make[
            Self.train_target, Xavier,
        ](ctx)
        t.q_opt = Adam(lr=lr)
        # `max_grad_norm` accepted for API compat but not wired (storage Adam
        # clips via a separate call; gated configs use 0.0) — matches SAC/DQN.
        _ = max_grad_norm
        comptime if Self.train_target == "gpu":
            t.q_opt.adopt[Self.train_target, Self.Q_NET](t.pair.online, ctx)

        t.target_y_blk = C51TargetYBlock[
            Self.Q_NET, Self.BATCH, Self.OBS_DIM, Self.NUM_ACTIONS,
            Self.N_ATOMS, Self.DOUBLE,
        ].make[Self.train_target](
            gamma=gamma, nstep=nstep, v_min=v_min, v_max=v_max, ctx=ctx,
        )

        t.q_update_blk = C51QUpdateStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.NUM_ACTIONS,
            Self.N_ATOMS, Self.Q_NET,
        ].make[Self.train_target](ctx=ctx)

        t.polyak_blk = SinglePolyakStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.Q_NET,
        ].make(tau=tau, update_every=target_update_freq)

        t.state = TrainerState[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH,
        ].make[Self.train_target](ctx=ctx)

        t.tracker = EpisodeTracker.new(
            window_size=window_size, initial_fill=initial_episode_fill,
        )

        comptime if Self.train_target == "cpu":
            t._mb_m = Tensor.alloc(Self.BATCH * Self.N_ATOMS)
            t._ob_scr.ensure(Self.OBS_DIM)
            t._q_scr.ensure(Self.NUM_ACTIONS * Self.N_ATOMS)
        else:
            var c = ctx.value()
            t._mb_m = Tensor.alloc_gpu(c, Self.BATCH * Self.N_ATOMS)
            t._ob_scr.ensure_gpu(c, Self.OBS_DIM)
            t._q_scr.ensure_gpu(c, Self.NUM_ACTIONS * Self.N_ATOMS)
            t._q_mean_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)
            t._target_mean_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)
            t._dist_entropy_mean_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)
            t._reward_mean_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)
            t._done_mean_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)
            t._diag_eq = Tensor.alloc_gpu(c, Self.BATCH)
            t._diag_ent = Tensor.alloc_gpu(c, Self.BATCH)
            t._diag_tq = Tensor.alloc_gpu(c, Self.BATCH)

        t.sample_blk.configure_per(
            alpha=per_alpha, beta=per_beta, epsilon=per_epsilon,
        )
        t.sample_blk.configure_gamma(gamma)
        t.sample_blk.setup(learning_starts, ctx=ctx)

        t.timer.add_section("sample")
        t.timer.add_section("target_y")
        t.timer.add_section("critic")
        t.timer.add_section("polyak")
        t.timer.add_section("diag")
        return t^

    # ─── Train step ──────────────────────────────────────────────────

    def _train_step_impl[POLICY: AMPPolicy = NoAMP](
        mut self, step_idx: Int,
    ) raises -> Bool:
        from std.time import perf_counter_ns

        self.state.step_idx = step_idx
        self.state.did_step = True
        comptime if Self.train_target == "gpu":
            self.state.ctx = self.ctx

        var t_sample = perf_counter_ns()
        self.sample_blk.step(self.state)
        if not self.state.did_step:
            return False
        self.timer.accumulate(Self._T_SAMPLE, t_sample)

        self._train_post_sample_kernels[POLICY]()

        # Per-batch distributional diag — CPU host walk (GPU counterpart folded
        # into `_train_post_sample_kernels`). For the taken action: softmax its
        # N_ATOMS logit row → expected Q (Σ p_k z_k) + entropy; `_mb_m`'s
        # expected value Σ m_k z_k is the target-Q.
        var t_diag = perf_counter_ns()
        comptime if Self.train_target == "cpu":
            comptime NK = Self.N_ATOMS
            var inv_b: Scalar[DT] = Scalar[DT](1.0) / Scalar[DT](Self.BATCH)
            var sum_q: Scalar[DT] = 0.0
            var sum_tq: Scalar[DT] = 0.0
            var sum_ent: Scalar[DT] = 0.0
            var sum_r: Scalar[DT] = 0.0
            var sum_d: Scalar[DT] = 0.0
            for b in range(Self.BATCH):
                var base = b * NK
                var maxl = self.q_update_blk.inner._logits_a.data[base]
                for k in range(1, NK):
                    var v = self.q_update_blk.inner._logits_a.data[base + k]
                    if v > maxl:
                        maxl = v
                var sum_exp: Scalar[DT] = 0.0
                for k in range(NK):
                    sum_exp += fexp(
                        self.q_update_blk.inner._logits_a.data[base + k] - maxl
                    )
                var eq: Scalar[DT] = 0.0
                var ent: Scalar[DT] = 0.0
                var tq: Scalar[DT] = 0.0
                for k in range(NK):
                    var p = (
                        fexp(
                            self.q_update_blk.inner._logits_a.data[base + k]
                            - maxl
                        )
                        / sum_exp
                    )
                    eq += p * self.target_y_blk._z.data[k]
                    if p > Scalar[DT](1e-12):
                        ent -= p * flog(p)
                    tq += self._mb_m.data[base + k] * self.target_y_blk._z.data[k]
                sum_q += eq
                sum_tq += tq
                sum_ent += ent
                sum_r += self.state.mb_r.data[b]
                sum_d += self.state.mb_d.data[b]
            self._q_accum += sum_q * inv_b
            self._target_accum += sum_tq * inv_b
            self._dist_entropy_accum += sum_ent * inv_b
            self._reward_accum += sum_r * inv_b
            self._done_accum += sum_d * inv_b
        self.timer.accumulate(Self._T_DIAG, t_diag)

        self.note_train_update()
        return True

    # ─── Shared post-sample kernel sequence ───────────────────────────
    def _train_post_sample_kernels[
        POLICY: AMPPolicy = NoAMP,
    ](mut self) raises:
        from std.time import perf_counter_ns

        var t_ty = perf_counter_ns()
        self.target_y_blk.step[Self.train_target, POLICY](
            self.pair.target_net,
            self.pair.online,
            self.state.mb_sp,
            self.state.mb_r,
            self.state.mb_d,
            self._mb_m,
            ctx=self.state.ctx,
        )
        self.timer.accumulate(Self._T_TARGET_Y, t_ty)

        var t_crit = perf_counter_ns()
        self.q_update_blk.step[
            Self.train_target, POLICY, ACCUMULATE = Self.train_target == "gpu"
        ](
            self.state, self.pair.online, self.q_opt, self._mb_m,
        )
        self.timer.accumulate(Self._T_CRITIC, t_crit)

        var t_poly = perf_counter_ns()
        self.polyak_blk.step[Self.train_target](self.state, self.pair)
        self.timer.accumulate(Self._T_POLYAK, t_poly)

        self.sample_blk.update_priorities(self.state)

        comptime if Self.train_target == "gpu":
            var t_diag = perf_counter_ns()
            var c = self.ctx.value()
            comptime LBNK = Layout.row_major(Self.BATCH, Self.N_ATOMS)
            comptime LNK = Layout.row_major(Self.N_ATOMS)
            comptime LB = Layout.row_major(Self.BATCH)
            comptime n_blocks = (Self.BATCH + TPB - 1) // TPB
            comptime diag_k = _c51_diag_kernel[Self.BATCH, Self.N_ATOMS]
            c.enqueue_function[diag_k](
                self.q_update_blk.inner._logits_a.lt["gpu", LBNK](),
                self._mb_m.lt["gpu", LBNK](),
                self.target_y_blk._z.lt["gpu", LNK](),
                self._diag_eq.lt["gpu", LB](),
                self._diag_ent.lt["gpu", LB](),
                self._diag_tq.lt["gpu", LB](),
                grid_dim=n_blocks, block_dim=TPB,
            )
            self._q_mean_dev.accumulate_gpu_lt[Self.BATCH](
                self._diag_eq.lt["gpu", LB]()
            )
            self._target_mean_dev.accumulate_gpu_lt[Self.BATCH](
                self._diag_tq.lt["gpu", LB]()
            )
            self._dist_entropy_mean_dev.accumulate_gpu_lt[Self.BATCH](
                self._diag_ent.lt["gpu", LB]()
            )
            self._reward_mean_dev.accumulate_gpu_lt[Self.BATCH](
                self.state.mb_r.lt["gpu", LB]()
            )
            self._done_mean_dev.accumulate_gpu_lt[Self.BATCH](
                self.state.mb_d.lt["gpu", LB]()
            )
            self.timer.accumulate(Self._T_DIAG, t_diag)

    def note_train_update(mut self):
        self._loss_accum += self.state.critic_loss
        self._update_count += 1
        self._total_train_steps += 1

    # ─── CUDA-graph capture surface ───────────────────────────────────
    def _train_device_kernels_impl[
        POLICY: AMPPolicy = NoAMP,
    ](mut self) raises:
        self.state.step_idx = self.learning_starts
        self.state.did_step = True
        self.state.ctx = self.ctx
        self.sample_blk.step(self.state)
        self._train_post_sample_kernels[POLICY]()

    def train_device_kernels(mut self) raises:
        comptime assert Self.train_target == "gpu", (
            "train_device_kernels is GPU-only (CUDA-graph capture path)"
        )
        self._train_device_kernels_impl[NoAMP]()

    def learning_starts_count(self) -> Int:
        return self.learning_starts

    def train_step(mut self, step_idx: Int) raises -> Bool:
        return self._train_step_impl[NoAMP](step_idx)

    def set_beta(mut self, beta: Scalar[DT]):
        self.sample_blk.set_beta(beta)

    # ─── Record ──────────────────────────────────────────────────────

    def record(
        mut self,
        ref obs: List[Scalar[DT]],
        action_idx: Int,
        reward: Scalar[DT],
        ref next_obs: List[Scalar[DT]],
        done: Scalar[DT],
    ) raises:
        self.tracker.add_reward(reward)
        self._action_list[0] = Scalar[DT](action_idx)
        self.sample_blk.add(
            obs, self._action_list, reward, next_obs, done, ctx=self.ctx,
        )

    def record_batch_cpu[
        N_ENVS: Int,
    ](
        mut self,
        prev_obs_ptr: Pointer[Scalar[DT], MutAnyOrigin],
        action_ptr: Pointer[Scalar[DT], MutAnyOrigin],
        reward_ptr: Pointer[Scalar[DT], MutAnyOrigin],
        next_obs_ptr: Pointer[Scalar[DT], MutAnyOrigin],
        done_ptr: Pointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        comptime assert (
            Self.train_target == "cpu"
        ), "record_batch_cpu: trainer must be cpu"
        comptime OBS = Self.OBS_DIM
        var obs_lane = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
        var act_lane = List[Scalar[DT]](length=1, fill=Scalar[DT](0.0))
        var nxt_lane = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
        for env_idx in range(N_ENVS):
            for d in range(OBS):
                obs_lane[d] = prev_obs_ptr[unsafe_offset=env_idx * OBS + d]
                nxt_lane[d] = next_obs_ptr[unsafe_offset=env_idx * OBS + d]
            act_lane[0] = action_ptr[unsafe_offset=env_idx]
            self.sample_blk.add(
                obs_lane, act_lane, reward_ptr[unsafe_offset=env_idx], nxt_lane,
                done_ptr[unsafe_offset=env_idx], ctx=self.ctx,
            )

    # ─── Batched device record (GPU-batched-env driver) ──────────────

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
        """1-step batched device record. `done_dev` carries `terminated`."""
        self.sample_blk.add_batch_gpu[N_ENVS](
            ctx, prev_obs_dev, action_dev, reward_dev, obs_dev, done_dev,
        )

    def record_batch_gpu_nstep[
        N_ENVS: Int, NS: Int
    ](
        mut self,
        ctx: DeviceContext,
        mut nstep_buf: GPUNStepBuffer[
            NS, Self.AGENT_OBS_DIM, Self.AGENT_ACT_DIM, N_ENVS
        ],
        prev_obs_dev: DeviceBuffer[DT],
        action_dev: DeviceBuffer[DT],
        reward_dev: DeviceBuffer[DT],
        obs_dev: DeviceBuffer[DT],
        done_dev: DeviceBuffer[DT],
    ) raises:
        """N-step batched device record (Rainbow). Keep `NS` aligned with the
        target-Y γ^N bootstrap."""
        nstep_buf.process(
            ctx, prev_obs_dev, action_dev, reward_dev, obs_dev, done_dev,
        )
        self.sample_blk.store_via_block_gpu[N_ENVS, NS](ctx, nstep_buf)

    # ─── Action selection (expected-Q argmax over softmax·z) ─────────

    def _ensure_action_scratch[
        N_ENVS: Int
    ](mut self) raises:
        comptime NK = Self.NUM_ACTIONS * Self.N_ATOMS
        comptime OBS = Self.OBS_DIM
        comptime if Self.train_target == "cpu":
            self._ob_scr.ensure(N_ENVS * OBS)
            self._q_scr.ensure(N_ENVS * NK)
        else:
            var c = self.ctx.value()
            self._ob_scr.ensure_gpu(c, N_ENVS * OBS)
            self._q_scr.ensure_gpu(c, N_ENVS * NK)
            if self._act_n != N_ENVS:
                self._act_host = List[Scalar[DT]](
                    length=N_ENVS, fill=Scalar[DT](0.0)
                )
                self._act_n = N_ENVS

    def _bridge_obs_and_forward[
        N_ENVS: Int
    ](
        mut self, obs_ptr: Pointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        """Copy raw obs into `_ob_scr`, run ONLINE forward into `_q_scr`, and
        (GPU) D2H the logits. After this `_q_scr.data` holds [N_ENVS, NA·NK]."""
        comptime NK = Self.NUM_ACTIONS * Self.N_ATOMS
        comptime OBS = Self.OBS_DIM
        comptime if Self.train_target == "cpu":
            for i in range(N_ENVS * OBS):
                self._ob_scr.data[i] = obs_ptr[unsafe_offset=i]
            call_forward["cpu", N_ENVS](
                self.pair.online,
                TensorRefs[Self.Q_NET.ARITY](self._ob_scr),
                self._q_scr,
            )
        else:
            var c = self.ctx.value()
            var obs_dev = DeviceBuffer[DT](
                c, obs_ptr, N_ENVS * OBS, owning=False,
            )
            c.enqueue_copy(self._ob_scr.dev.value(), obs_dev)
            call_forward["gpu", N_ENVS](
                self.pair.online,
                TensorRefs[Self.Q_NET.ARITY](self._ob_scr),
                self._q_scr,
                self.ctx,
            )
            self._q_scr.download(c)

    def _expected_q_argmax(self, base: Int) -> Int:
        """argmax_a Σ_k softmax(`_q_scr.data`[base + a·NK + k])·z_k."""
        comptime NA = Self.NUM_ACTIONS
        comptime NK = Self.N_ATOMS
        var best_a: Int = 0
        var best_eq: Scalar[DT] = Scalar[DT](0.0)
        for a in range(NA):
            var ab = base + a * NK
            var mx = self._q_scr.data[ab]
            for i in range(1, NK):
                if self._q_scr.data[ab + i] > mx:
                    mx = self._q_scr.data[ab + i]
            var s_exp: Scalar[DT] = Scalar[DT](0.0)
            for i in range(NK):
                s_exp = s_exp + fexp(self._q_scr.data[ab + i] - mx)
            var eq: Scalar[DT] = Scalar[DT](0.0)
            for i in range(NK):
                var p = fexp(self._q_scr.data[ab + i] - mx) / s_exp
                eq = eq + p * self.target_y_blk._z.data[i]
            if a == 0 or eq > best_eq:
                best_eq = eq
                best_a = a
        return best_a

    def select_action_batched[
        N_ENVS: Int,
    ](
        mut self,
        obs_ptr: Pointer[Scalar[DT], MutAnyOrigin],
        action_ptr: Pointer[Scalar[DT], MutAnyOrigin],
        step_idx: Int,
    ) raises:
        comptime NA = Self.NUM_ACTIONS
        comptime NK = Self.NUM_ACTIONS * Self.N_ATOMS

        # Warmup: uniform random action.
        if step_idx < self.learning_starts:
            comptime if Self.train_target == "cpu":
                for i in range(N_ENVS):
                    action_ptr[unsafe_offset=i] = Scalar[DT](
                        Int(random_float64() * Float64(NA))
                    )
            else:
                var c = self.ctx.value()
                self._ensure_action_scratch[N_ENVS]()
                var act_h = self._act_host.unsafe_ptr()
                for i in range(N_ENVS):
                    act_h[unsafe_offset=i] = Scalar[DT](Int(random_float64() * Float64(NA)))
                var action_dev = DeviceBuffer[DT](
                    c, action_ptr, N_ENVS, owning=False,
                )
                c.enqueue_copy(action_dev, act_h)
            return

        self._ensure_action_scratch[N_ENVS]()
        self._bridge_obs_and_forward[N_ENVS](obs_ptr)

        comptime if Self.train_target == "cpu":
            for i in range(N_ENVS):
                var r = random_float64()
                if r < Float64(self.epsilon):
                    action_ptr[unsafe_offset=i] = Scalar[DT](
                        Int(random_float64() * Float64(NA))
                    )
                else:
                    action_ptr[unsafe_offset=i] = Scalar[DT](self._expected_q_argmax(i * NK))
        else:
            var c = self.ctx.value()
            var act_h = self._act_host.unsafe_ptr()
            for i in range(N_ENVS):
                var r = random_float64()
                if r < Float64(self.epsilon):
                    act_h[unsafe_offset=i] = Scalar[DT](Int(random_float64() * Float64(NA)))
                else:
                    act_h[unsafe_offset=i] = Scalar[DT](self._expected_q_argmax(i * NK))
            var action_dev = DeviceBuffer[DT](
                c, action_ptr, N_ENVS, owning=False,
            )
            c.enqueue_copy(action_dev, act_h)

    def set_noise_scale(mut self, scale: Scalar[DT]) raises:
        """Broadcast `noise_scale` to the online net's NoisyLinear layers (1.0 =
        explore, 0.0 = deterministic mean weights). No-op if no Noisy layers."""
        self.pair.online.set_attr["noise_scale"](scale)

    def select_greedy_action_batched[
        N_ENVS: Int,
    ](
        mut self,
        obs_ptr: Pointer[Scalar[DT], MutAnyOrigin],
        action_ptr: Pointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        """Pure greedy: batched forward + expected-Q argmax per env, no epsilon/
        warmup. Pair with `set_noise_scale(0)` for deterministic eval."""
        comptime NA = Self.NUM_ACTIONS
        comptime NK = Self.NUM_ACTIONS * Self.N_ATOMS
        self._ensure_action_scratch[N_ENVS]()
        self._bridge_obs_and_forward[N_ENVS](obs_ptr)
        comptime if Self.train_target == "cpu":
            for i in range(N_ENVS):
                action_ptr[unsafe_offset=i] = Scalar[DT](self._expected_q_argmax(i * NK))
        else:
            var c = self.ctx.value()
            var act_h = self._act_host.unsafe_ptr()
            for i in range(N_ENVS):
                act_h[unsafe_offset=i] = Scalar[DT](self._expected_q_argmax(i * NK))
            var action_dev = DeviceBuffer[DT](
                c, action_ptr, N_ENVS, owning=False,
            )
            c.enqueue_copy(action_dev, act_h)

    def select_greedy_action(
        mut self,
        ref obs: List[Scalar[DT]],
    ) raises -> Int:
        comptime OBS = Self.OBS_DIM
        self._ensure_action_scratch[1]()
        comptime if Self.train_target == "cpu":
            for d in range(OBS):
                self._ob_scr.data[d] = obs[d]
            call_forward["cpu", 1](
                self.pair.online,
                TensorRefs[Self.Q_NET.ARITY](self._ob_scr),
                self._q_scr,
            )
            return self._expected_q_argmax(0)
        else:
            var c = self.ctx.value()
            var obs_h = List[Scalar[DT]](length=OBS, fill=Scalar[DT](0.0))
            for d in range(OBS):
                obs_h[d] = obs[d]
            c.enqueue_copy(self._ob_scr.dev.value(), obs_h.unsafe_ptr())
            call_forward["gpu", 1](
                self.pair.online,
                TensorRefs[Self.Q_NET.ARITY](self._ob_scr),
                self._q_scr,
                self.ctx,
            )
            self._q_scr.download(c)
            return self._expected_q_argmax(0)

    # ─── Episode tracking ────────────────────────────────────────────

    def end_episode(mut self):
        self.tracker.end_episode()
        self.epsilon = self.epsilon * self.epsilon_decay
        if self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min

    def add_complete_return(mut self, ret: Scalar[DT]):
        self.tracker.add_complete_return(ret)

    def mean_return(self) -> Scalar[DT]:
        return self.tracker.mean_return()

    def ep_count(self) -> Int:
        return self.tracker.ep_count

    # ─── Logging ─────────────────────────────────────────────────────

    def flush_train_log(
        mut self,
    ) -> Tuple[Scalar[DT], Scalar[DT], Int]:
        var n = self._update_count if self._update_count > 0 else 1
        var inv = Scalar[DT](1.0) / Scalar[DT](n)
        var out = (
            self._loss_accum * inv,
            self.epsilon,
            self._update_count,
        )
        self._loss_accum = Scalar[DT](0.0)
        self._q_accum = Scalar[DT](0.0)
        self._target_accum = Scalar[DT](0.0)
        self._dist_entropy_accum = Scalar[DT](0.0)
        self._reward_accum = Scalar[DT](0.0)
        self._done_accum = Scalar[DT](0.0)
        self._update_count = 0
        return out

    def total_train_steps(self) -> Int:
        return self._total_train_steps

    def flush_metrics[
        L: Logger = NoOpLogger
    ](
        mut self,
        logger: Optional[Pointer[L, MutAnyOrigin]] = None,
        step: Int = 0,
    ) raises -> C51Metrics:
        """Drain accumulators into a C51Metrics bundle (+ optional logger
        emit). Resets per-chunk accumulators; `_total_train_steps` not reset."""
        var n = self._update_count if self._update_count > 0 else 1
        var inv = Scalar[DT](1.0) / Scalar[DT](n)
        var q_mean: Scalar[DT]
        var target_mean: Scalar[DT]
        var dist_entropy_mean: Scalar[DT]
        var reward_mean: Scalar[DT]
        var done_mean: Scalar[DT]
        var loss_mean: Scalar[DT]
        comptime if Self.train_target == "gpu":
            loss_mean = self.q_update_blk.inner.ce_loss.read_accum["gpu"](
                self.ctx
            )
            q_mean = self._q_mean_dev.read["gpu"]()
            target_mean = self._target_mean_dev.read["gpu"]()
            dist_entropy_mean = self._dist_entropy_mean_dev.read["gpu"]()
            reward_mean = self._reward_mean_dev.read["gpu"]()
            done_mean = self._done_mean_dev.read["gpu"]()
        else:
            loss_mean = self._loss_accum * inv
            q_mean = self._q_accum * inv
            target_mean = self._target_accum * inv
            dist_entropy_mean = self._dist_entropy_accum * inv
            reward_mean = self._reward_accum * inv
            done_mean = self._done_accum * inv
        var bundle = C51Metrics(
            loss=LogScalar[DT](loss_mean),
            explore_rate=LogScalar[DT](self.epsilon),
            mean_q=LogScalar[DT](q_mean),
            mean_target=LogScalar[DT](target_mean),
            dist_entropy_mean=LogScalar[DT](dist_entropy_mean),
            reward_mean=LogScalar[DT](reward_mean),
            mean_done=LogScalar[DT](done_mean),
            train_steps=LogScalar[DT](Scalar[DT](self._total_train_steps)),
            n_updates=LogScalar[DT](Scalar[DT](self._update_count)),
        )
        self._loss_accum = Scalar[DT](0.0)
        self._q_accum = Scalar[DT](0.0)
        self._target_accum = Scalar[DT](0.0)
        self._dist_entropy_accum = Scalar[DT](0.0)
        self._reward_accum = Scalar[DT](0.0)
        self._done_accum = Scalar[DT](0.0)
        comptime if Self.train_target == "gpu":
            self.q_update_blk.inner.ce_loss.reset_accum["gpu"]()
            self._q_mean_dev.reset["gpu"]()
            self._target_mean_dev.reset["gpu"]()
            self._dist_entropy_mean_dev.reset["gpu"]()
            self._reward_mean_dev.reset["gpu"]()
            self._done_mean_dev.reset["gpu"]()
        self._update_count = 0
        if Bool(logger):
            log_bundle(logger.value()[], bundle, step)
        return bundle^

    def flush_metrics_through_logger[L: Logger](
        mut self,
        logger: Optional[Pointer[L, MutAnyOrigin]],
        step: Int,
    ) raises:
        _ = self.flush_metrics[L](logger, step)

    def save_state(mut self, path: String) raises:
        """One-file v3 checkpoint of the ONLINE Q-net params + state, with ε
        and the counter as `K` scalars. Optimizer moments NOT persisted (resume
        re-warms, matching the storage SAC/DQN checkpoints). Target net
        hard-copied from online on load."""
        var w = BinaryCheckpointWriter(save_moments=False)
        write_model[Self.train_target](w, self.pair.online, self.ctx, "q_net")
        var sc = CheckpointScalars()
        sc.set("eps.epsilon", Float64(self.epsilon))
        sc.set("eps.epsilon_decay", Float64(self.epsilon_decay))
        sc.set("eps.epsilon_min", Float64(self.epsilon_min))
        sc.set_int("total_train_steps", self._total_train_steps)
        w.write_scalars(sc)
        _write_file_bytes(path, w.content)

    def load_state(mut self, path: String) raises:
        """Inverse of `save_state`: restore online params + state + ε + counter,
        then hard-copy online → target. Legacy v2 text files still load."""
        var bytes = _read_file_bytes(path)
        if _is_v3_header(bytes):
            var rb = BinaryCheckpointReader(bytes^)
            read_model[Self.train_target](rb, self.pair.online, self.ctx, "q_net")
            var sc = rb.read_scalars()
            rb.finish()
            self.epsilon = Scalar[DT](sc.get("eps.epsilon", Float64(self.epsilon)))
            self.epsilon_decay = Scalar[DT](
                sc.get("eps.epsilon_decay", Float64(self.epsilon_decay))
            )
            self.epsilon_min = Scalar[DT](
                sc.get("eps.epsilon_min", Float64(self.epsilon_min))
            )
            self._total_train_steps = sc.get_int(
                "total_train_steps", self._total_train_steps
            )
            self.pair.target_net.polyak_from[Self.train_target](
                self.pair.online, Scalar[DT](1.0), self.ctx
            )
            return
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
        walk_params[Self.train_target](self.pair.online, r, self.ctx, "q_net")
        r.mode = 1
        var _sref2 = ParamVisitorRef.of[type_of(r), Self.train_target](r)
        self.pair.online.for_each_state[Self.train_target](_sref2, self.ctx, "q_net")
        self.epsilon = self._scan_scalar(content, "eps.epsilon=", self.epsilon)
        self.epsilon_decay = self._scan_scalar(
            content, "eps.epsilon_decay=", self.epsilon_decay
        )
        self.epsilon_min = self._scan_scalar(
            content, "eps.epsilon_min=", self.epsilon_min
        )
        self._total_train_steps = Int(
            self._scan_scalar(
                content, "_total_train_steps=",
                Scalar[DT](self._total_train_steps),
            )
        )
        self.pair.target_net.polyak_from[Self.train_target](
            self.pair.online, Scalar[DT](1.0), self.ctx
        )

    @staticmethod
    def _scan_scalar(
        content: String, key: String, default: Scalar[DT],
    ) raises -> Scalar[DT]:
        """Scan `content` for `key<value>`; return its float (or `default`)."""
        var lines = _split_lines(content)
        for i in range(len(lines)):
            if lines[i].startswith(key):
                # `key` ends with '=' and the value has no '=', so split on '='
                # and take the tail (nightly String has no positional slicing).
                var parts = lines[i].split("=")
                if len(parts) >= 2:
                    return Scalar[DT](atof(parts[len(parts) - 1]))
        return default

    def flush_timer_log(mut self) -> String:
        var report = self.timer.format_report()
        self.timer.reset()
        return report
