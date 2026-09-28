"""DQNTrainer — DQN family trainer (CPU/GPU × uniform replay) (STORAGE).

Discrete-action off-policy trainer for DQN and variants. Pipeline body mirrors
`sac/trainer.mojo`'s SAC pipeline: each train step is a short sequence of
`<block>.step[target, POLICY]` calls against a shared `TrainerState`, with
target-Y and gather/scatter running on-device.

Block decomposition:
  1. `sample_blk: SAMPLE` (SampleBlock trait — uniform / PER / N-step)
  2. `target_y_blk: DQNTargetYBlock` (forward-only, owns `Q_target.forward
     → ReduceMax → finalize fuse`; Double branch swaps in argmax+gather)
  3. `q_update_blk: DQNQUpdateStep` (owns gather+MSE+scatter+Q.vjp)
  4. `polyak_blk: SinglePolyakStep` (soft τ-update OR hard copy every N)

Driver-trait conformance: `OffPolicyDiscreteAgentGpu` via `train_step`,
`select_action_batched`, `select_greedy_action`, `record`, `record_batch_cpu`,
`add_complete_return`, GPU-batched device record + greedy eval.

STORAGE migration (Stage 5): `Scratch`/`TargetStorage`/`init_scratch_auto`/
`mptr`/TileTensor/legacy-checkpoint gone. The Q nets are storage `Module`s; the
blocks pass storage `Tensor`s; action selection bridges the driver's raw obs/
action pointers into owned `Tensor` scratch around the storage `Module.forward`
(the same bridge the storage SAC trainer uses). Checkpoint: v3, with ε and the
train-step counter as `K` scalars (legacy v2 files still load).
CPU + GPU bodies share one `_train_step_impl[POLICY]`. No D2H/H2D in the GPU
train step (all gather/scatter on-device).
"""

from noeira.nn.core.param import walk_params
from std.random import random_float64
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import Layout, LayoutTensor

from noeira.core.logger import Logger, NoOpLogger
from noeira.nn.constants import DT
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
from noeira.nn.core.save_scalar import _expect_kv_line
from noeira.nn.training.timer import Timer

from ..core.online_target_pair import OnlineTargetPair
from ..data.n_step_replay import GPUNStepBuffer
from ..training.episode_tracker import EpisodeTracker
from ..training.device_mean_accum import DeviceMeanAccum
from ..training.trainer_block import TrainerState
from ..training.driver_offpolicy_discrete import (
    OffPolicyDiscreteAgent,
    OffPolicyDiscreteAgentGpu,
)
from ..training.blocks import SampleBlock, SinglePolyakStep
from .target_y_block import DQNTargetYBlock
from .blocks.q_update_step import DQNQUpdateStep
from .metrics import DQNMetrics


struct DQNTrainer[
    train_target: StaticString,
    SAMPLE: SampleBlock,
    Q_NET: Module,
    DOUBLE: Bool = False,
](OffPolicyDiscreteAgentGpu):
    """Dimensions derived from SAMPLE (OBS, ACT=1, BATCH) and Q_NET
    (OUT_DIM = NUM_ACTIONS). The sample block stores discrete action indices as
    a single Scalar[DT] in ACT=1."""

    comptime OBS_DIM: Int = Self.SAMPLE.OBS
    comptime ACT_DIM: Int = Self.SAMPLE.ACT
    comptime BATCH: Int = Self.SAMPLE.BATCH
    comptime NUM_ACTIONS: Int = Self.Q_NET.OUT_DIM

    comptime AGENT_TRAIN_TARGET: StaticString = Self.train_target
    comptime AGENT_OBS_DIM: Int = Self.OBS_DIM
    comptime AGENT_NUM_ACTIONS: Int = Self.NUM_ACTIONS
    comptime AGENT_ACT_DIM: Int = Self.ACT_DIM

    comptime _T_SAMPLE = 0
    comptime _T_TARGET_Y = 1
    comptime _T_CRITIC = 2
    comptime _T_POLYAK = 3
    comptime _T_DIAG = 4

    var pair: OnlineTargetPair[Self.Q_NET]
    var q_opt: Adam
    var sample_blk: Self.SAMPLE
    var target_y_blk: DQNTargetYBlock[
        Self.Q_NET, Self.BATCH, Self.OBS_DIM, Self.NUM_ACTIONS, Self.DOUBLE,
    ]
    var q_update_blk: DQNQUpdateStep[
        Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.NUM_ACTIONS, Self.Q_NET,
    ]
    var polyak_blk: SinglePolyakStep[
        Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.Q_NET,
    ]

    var state: TrainerState[Self.OBS_DIM, Self.ACT_DIM, Self.BATCH]

    # Action-selection scratch: obs bridge + Q output (owned Tensors, lazily
    # sized for the largest N_ENVS seen). The storage `Module.forward` consumes
    # owned Tensors, so the driver's raw obs pointer is COPIED into `_ob_scr`
    # before the forward, and the Q output read back from `_q_scr`.
    var _ob_scr: Tensor
    var _q_scr: Tensor
    # Host mirror for the GPU action H2D (one index per env). Lazily sized.
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
    # Per-batch diagnostic accumulators (CPU-only diag walk).
    var _q_accum: Scalar[DT]
    var _target_accum: Scalar[DT]
    var _td_error_accum: Scalar[DT]
    var _reward_accum: Scalar[DT]
    var _done_accum: Scalar[DT]
    # GPU device-resident mirrors (CPU keeps the host scalars above).
    var _q_mean_dev: DeviceMeanAccum
    var _target_mean_dev: DeviceMeanAccum
    var _td_error_mean_dev: DeviceMeanAccum
    var _reward_mean_dev: DeviceMeanAccum
    var _done_mean_dev: DeviceMeanAccum
    var _update_count: Int
    # Never reset by `flush_*` — emitted as `train_steps` (cumulative updates).
    var _total_train_steps: Int
    var timer: Timer

    def __init__(out self):
        self.pair = OnlineTargetPair[Self.Q_NET]()
        self.q_opt = Adam()
        self.sample_blk = Self.SAMPLE()
        self.target_y_blk = DQNTargetYBlock[
            Self.Q_NET, Self.BATCH, Self.OBS_DIM, Self.NUM_ACTIONS,
            Self.DOUBLE,
        ]()
        self.q_update_blk = DQNQUpdateStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.NUM_ACTIONS,
            Self.Q_NET,
        ]()
        self.polyak_blk = SinglePolyakStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.Q_NET,
        ]()
        self.state = TrainerState[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH,
        ]()
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
        self._td_error_accum = Scalar[DT](0.0)
        self._reward_accum = Scalar[DT](0.0)
        self._done_accum = Scalar[DT](0.0)
        self._q_mean_dev = DeviceMeanAccum()
        self._target_mean_dev = DeviceMeanAccum()
        self._td_error_mean_dev = DeviceMeanAccum()
        self._reward_mean_dev = DeviceMeanAccum()
        self._done_mean_dev = DeviceMeanAccum()
        self._update_count = 0
        self._total_train_steps = 0
        self.timer = Timer.new()

    @staticmethod
    def make(
        ctx: Optional[DeviceContext] = None,
        lr: Scalar[DT] = Scalar[DT](1e-3),
        gamma: Scalar[DT] = Scalar[DT](0.99),
        tau: Scalar[DT] = Scalar[DT](0.005),
        epsilon: Scalar[DT] = Scalar[DT](1.0),
        epsilon_decay: Scalar[DT] = Scalar[DT](0.995),
        epsilon_min: Scalar[DT] = Scalar[DT](0.01),
        learning_starts: Int = 1_000,
        target_update_freq: Int = 0,
        window_size: Int = 10,
        initial_episode_fill: Scalar[DT] = Scalar[DT](0.0),
        max_grad_norm: Scalar[DT] = Scalar[DT](0.0),
        per_alpha: Scalar[DT] = Scalar[DT](0.6),
        per_beta: Scalar[DT] = Scalar[DT](0.4),
        per_epsilon: Scalar[DT] = Scalar[DT](1e-6),
        nstep: Int = 1,
    ) raises -> Self:
        comptime assert (
            Self.train_target == "cpu" or Self.train_target == "gpu"
        ), "DQNTrainer: target must be 'cpu' or 'gpu'"
        comptime assert (
            Self.ACT_DIM == 1
        ), "DQNTrainer: SAMPLE.ACT must be 1 (discrete action index)"
        comptime assert (
            Self.Q_NET.IN_DIMS[0] == Self.OBS_DIM
        ), "DQNTrainer: Q_NET.IN_DIM must equal SAMPLE.OBS"
        comptime if Self.train_target == "gpu":
            if not ctx:
                raise Error("DQNTrainer.make[target='gpu']: ctx required")

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
        # NOTE: `max_grad_norm` is accepted for API compatibility but not wired
        # — the storage `Adam.step` does not clip (grad clipping is a separate
        # explicit `clip_grads_device` call), matching the storage SAC/TD3
        # migration. The gated CartPole configs use 0.0 (off), so this is
        # behaviorally inert; CNN/Atari grad-clip wiring is a follow-up.
        _ = max_grad_norm
        comptime if Self.train_target == "gpu":
            t.q_opt.adopt[Self.train_target, Self.Q_NET](t.pair.online, ctx)

        t.target_y_blk = DQNTargetYBlock[
            Self.Q_NET, Self.BATCH, Self.OBS_DIM, Self.NUM_ACTIONS,
            Self.DOUBLE,
        ].make[Self.train_target](gamma=gamma, nstep=nstep, ctx=ctx)

        t.q_update_blk = DQNQUpdateStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.NUM_ACTIONS,
            Self.Q_NET,
        ].make[Self.train_target](ctx=ctx)

        t.polyak_blk = SinglePolyakStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.Q_NET,
        ].make(tau=tau, update_every=target_update_freq)

        t.state = TrainerState[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH,
        ].make[Self.train_target](ctx=ctx)

        t.tracker = EpisodeTracker.new(
            window_size=window_size,
            initial_fill=initial_episode_fill,
        )

        # Pre-size the single-env action scratch.
        comptime if Self.train_target == "cpu":
            t._ob_scr.ensure(Self.OBS_DIM)
            t._q_scr.ensure(Self.NUM_ACTIONS)
        else:
            var c = ctx.value()
            t._ob_scr.ensure_gpu(c, Self.OBS_DIM)
            t._q_scr.ensure_gpu(c, Self.NUM_ACTIONS)
            # Device-resident mean accumulators for the GPU diag path.
            t._q_mean_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)
            t._target_mean_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)
            t._td_error_mean_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)
            t._reward_mean_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)
            t._done_mean_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)

        # PER hyperparameter wiring: no-op default for uniform blocks.
        t.sample_blk.configure_per(
            alpha=per_alpha, beta=per_beta, epsilon=per_epsilon,
        )
        # N-step γ alignment: no-op default for non-nstep blocks.
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

        # 1. Sample.
        var t_sample = perf_counter_ns()
        self.sample_blk.step(self.state)
        if not self.state.did_step:
            return False
        self.timer.accumulate(Self._T_SAMPLE, t_sample)

        # Shared device-kernel sequence (target_y → q_update → polyak → PER
        # tail → GPU diag) — the body the CUDA-graph capture path replays.
        self._train_post_sample_kernels[POLICY]()

        # Per-batch diagnostic means — CPU-only host walk (the GPU counterpart
        # is folded into `_train_post_sample_kernels`). Q(s,a) is the gathered
        # Q at the taken action; target/reward/done live in the shared state.
        # `mean_td_error` = mean |Q − y|, the Bellman residual magnitude.
        var t_diag = perf_counter_ns()
        comptime if Self.train_target == "cpu":
            var inv_b: Scalar[DT] = Scalar[DT](1.0) / Scalar[DT](Self.BATCH)
            var sum_q: Scalar[DT] = 0.0
            var sum_y: Scalar[DT] = 0.0
            var sum_te: Scalar[DT] = 0.0
            var sum_r: Scalar[DT] = 0.0
            var sum_d: Scalar[DT] = 0.0
            for i in range(Self.BATCH):
                var qi = self.q_update_blk.inner._mb_q_gath.data[i]
                var yi = self.state.mb_y.data[i]
                var te = qi - yi
                sum_q += qi
                sum_y += yi
                sum_te += te if te >= Scalar[DT](0.0) else -te
                sum_r += self.state.mb_r.data[i]
                sum_d += self.state.mb_d.data[i]
            self._q_accum += sum_q * inv_b
            self._target_accum += sum_y * inv_b
            self._td_error_accum += sum_te * inv_b
            self._reward_accum += sum_r * inv_b
            self._done_accum += sum_d * inv_b
        self.timer.accumulate(Self._T_DIAG, t_diag)

        self.note_train_update()
        return True

    # ─── Shared post-sample kernel sequence ───────────────────────────
    #
    # target_y → q_update (ACCUMULATE on GPU) → polyak → PER tail → GPU diag.
    # Called by BOTH `_train_step_impl` (non-captured) and
    # `train_device_kernels` (the CUDA-graph capture closure body), so the two
    # paths enqueue an identical kernel sequence — bit-identity by construction.
    def _train_post_sample_kernels[
        POLICY: AMPPolicy = NoAMP,
    ](mut self) raises:
        from std.time import perf_counter_ns

        # 2. Target-Y.
        var t_ty = perf_counter_ns()
        self.target_y_blk.step[Self.train_target, POLICY](
            self.pair.target_net,
            self.pair.online,
            self.state.mb_sp,
            self.state.mb_r,
            self.state.mb_d,
            self.state.mb_y,
            ctx=self.state.ctx,
        )
        self.timer.accumulate(Self._T_TARGET_Y, t_ty)

        # 3. Q-update (gather → MSE → scatter → Q.vjp → opt.step). On GPU,
        # accumulate the MSE loss on-device (no per-step D2H, CUDA-graph
        # capturable); the host reads it at flush via read_accum.
        var t_crit = perf_counter_ns()
        self.q_update_blk.step[
            Self.train_target, POLICY, ACCUMULATE = Self.train_target == "gpu"
        ](
            self.state, self.pair.online, self.q_opt,
        )
        self.timer.accumulate(Self._T_CRITIC, t_crit)

        # 4. Polyak / hard-copy.
        var t_poly = perf_counter_ns()
        self.polyak_blk.step[Self.train_target](self.state, self.pair)
        self.timer.accumulate(Self._T_POLYAK, t_poly)

        # PER tail (no-op for uniform blocks).
        self.sample_blk.update_priorities(self.state)

        # GPU per-batch diag — device reductions into device-resident running
        # means (no D2H → capture-safe). The CPU host-walk lives in
        # `_train_step_impl`.
        comptime if Self.train_target == "gpu":
            var t_diag = perf_counter_ns()
            comptime lb = Layout.row_major(Self.BATCH)
            self._q_mean_dev.accumulate_gpu_lt[Self.BATCH](
                self.q_update_blk.inner._mb_q_gath.lt["gpu", lb]()
            )
            self._target_mean_dev.accumulate_gpu_lt[Self.BATCH](
                self.state.mb_y.lt["gpu", lb]()
            )
            self._td_error_mean_dev.accumulate_gpu_abs_diff_lt[Self.BATCH](
                self.q_update_blk.inner._mb_q_gath.lt["gpu", lb](),
                self.state.mb_y.lt["gpu", lb](),
            )
            self._reward_mean_dev.accumulate_gpu_lt[Self.BATCH](
                self.state.mb_r.lt["gpu", lb]()
            )
            self._done_mean_dev.accumulate_gpu_lt[Self.BATCH](
                self.state.mb_d.lt["gpu", lb]()
            )
            self.timer.accumulate(Self._T_DIAG, t_diag)

    # ─── Host bookkeeping (counters + metric accumulator) ─────────────
    def note_train_update(mut self):
        self._loss_accum += self.state.critic_loss
        self._update_count += 1
        self._total_train_steps += 1

    # ─── CUDA-graph capture surface ───────────────────────────────────
    def _train_device_kernels_impl[
        POLICY: AMPPolicy = NoAMP,
    ](mut self) raises:
        # Pin `state.step_idx = learning_starts` so the sample block's warmup
        # gate passes (this method has no step_idx of its own).
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
        """Cumulative env-step threshold after which the replay buffer is warm
        enough to train — the driver gates the capture path on this."""
        return self.learning_starts

    def train_step(mut self, step_idx: Int) raises -> Bool:
        return self._train_step_impl[NoAMP](step_idx)

    def set_beta(mut self, beta: Scalar[DT]):
        """PER IS-β anneal hook (callers ramp 0.4 → 1.0). No-op for uniform
        sample blocks."""
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

    # ─── Action selection ────────────────────────────────────────────

    def _ensure_action_scratch[
        N_ENVS: Int
    ](mut self) raises:
        """Lazily grow the obs/Q scratch + host action mirror for N_ENVS."""
        comptime NA = Self.NUM_ACTIONS
        comptime OBS = Self.OBS_DIM
        comptime if Self.train_target == "cpu":
            self._ob_scr.ensure(N_ENVS * OBS)
            self._q_scr.ensure(N_ENVS * NA)
        else:
            var c = self.ctx.value()
            self._ob_scr.ensure_gpu(c, N_ENVS * OBS)
            self._q_scr.ensure_gpu(c, N_ENVS * NA)
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
        """Copy the driver's raw obs slab into `_ob_scr`, run the ONLINE Q-net
        forward into `_q_scr`, and (GPU) D2H the Q values into `_q_scr.data`.
        After this `_q_scr.data` holds the `[N_ENVS, NA]` Q values on the host
        for the per-env argmax."""
        comptime NA = Self.NUM_ACTIONS
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

    def _argmax_row(self, base: Int) -> Int:
        """argmax_a `_q_scr.data[base + a]` over a flat NUM_ACTIONS row."""
        comptime NA = Self.NUM_ACTIONS
        var best_a = 0
        var best_q = self._q_scr.data[base]
        for a in range(1, NA):
            var q = self._q_scr.data[base + a]
            if q > best_q:
                best_q = q
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

        # Warmup path (random action). CPU writes action_ptr directly; GPU
        # stages through a host buffer then H2D (action_ptr is device-side).
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

        # Policy path: one batched online forward, then ε-greedy argmax per env.
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
                    action_ptr[unsafe_offset=i] = Scalar[DT](self._argmax_row(i * NA))
        else:
            var c = self.ctx.value()
            var act_h = self._act_host.unsafe_ptr()
            for i in range(N_ENVS):
                var r = random_float64()
                if r < Float64(self.epsilon):
                    act_h[unsafe_offset=i] = Scalar[DT](Int(random_float64() * Float64(NA)))
                else:
                    act_h[unsafe_offset=i] = Scalar[DT](self._argmax_row(i * NA))
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
            return self._argmax_row(0)
        else:
            var c = self.ctx.value()
            # Stage the single obs through a host list → device → forward.
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
            return self._argmax_row(0)

    # ─── GPU-batched device record (OffPolicyDiscreteAgentGpu) ────────

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
        """1-step batched device record — forwards N_ENVS device transitions
        to the sample block's `add_batch_gpu`. `done_dev` carries `terminated`."""
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
        """N-step batched device record. Keep `NS` aligned with the target-Y
        γ^N bootstrap."""
        nstep_buf.process(
            ctx, prev_obs_dev, action_dev, reward_dev, obs_dev, done_dev,
        )
        self.sample_blk.store_via_block_gpu[N_ENVS, NS](ctx, nstep_buf)

    def select_greedy_action_batched[
        N_ENVS: Int
    ](
        mut self,
        obs_ptr: Pointer[Scalar[DT], MutAnyOrigin],
        action_ptr: Pointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        """Pure greedy action selection for N_ENVS envs — argmax Q, no epsilon,
        no warmup gate. Writes N_ENVS action indices into `action_ptr`."""
        comptime NA = Self.NUM_ACTIONS
        self._ensure_action_scratch[N_ENVS]()
        self._bridge_obs_and_forward[N_ENVS](obs_ptr)
        comptime if Self.train_target == "cpu":
            for i in range(N_ENVS):
                action_ptr[unsafe_offset=i] = Scalar[DT](self._argmax_row(i * NA))
        else:
            var c = self.ctx.value()
            var act_h = self._act_host.unsafe_ptr()
            for i in range(N_ENVS):
                act_h[unsafe_offset=i] = Scalar[DT](self._argmax_row(i * NA))
            var action_dev = DeviceBuffer[DT](
                c, action_ptr, N_ENVS, owning=False,
            )
            c.enqueue_copy(action_dev, act_h)

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

    def set_noise_scale(mut self, scale: Scalar[DT]) raises:
        """Toggle Noisy-net exploration magnitude on the online net (1.0 =
        explore, 0.0 = deterministic mean weights). No-op for non-Noisy nets —
        the storage NoisyLinear honours it where present."""
        self.pair.online.set_attr["noise_scale"](scale)

    # ─── Logging ─────────────────────────────────────────────────────

    def flush_train_log(
        mut self,
    ) -> Tuple[Scalar[DT], Scalar[DT], Int]:
        """Return (mean_loss, epsilon, n_updates) since last flush."""
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
        self._td_error_accum = Scalar[DT](0.0)
        self._reward_accum = Scalar[DT](0.0)
        self._done_accum = Scalar[DT](0.0)
        self._update_count = 0
        return out

    def total_train_steps(self) -> Int:
        """Cumulative training updates since trainer was made. Not reset by
        `flush_*`."""
        return self._total_train_steps

    def flush_metrics[
        L: Logger = NoOpLogger
    ](
        mut self,
        logger: Optional[Pointer[L, MutAnyOrigin]] = None,
        step: Int = 0,
    ) raises -> DQNMetrics:
        """Drain accumulators into a DQNMetrics bundle. If a logger pointer is
        wired, also emit one log_scalar per metric field. Resets per-chunk
        accumulators; the cumulative `_total_train_steps` is NOT reset."""
        var n = self._update_count if self._update_count > 0 else 1
        var inv = Scalar[DT](1.0) / Scalar[DT](n)
        var q_mean: Scalar[DT]
        var target_mean: Scalar[DT]
        var td_error_mean: Scalar[DT]
        var reward_mean: Scalar[DT]
        var done_mean: Scalar[DT]
        var loss_mean: Scalar[DT]
        comptime if Self.train_target == "gpu":
            loss_mean = self.q_update_blk.inner.mse_loss.read_accum["gpu"](
                self.ctx
            )
            q_mean = self._q_mean_dev.read["gpu"]()
            target_mean = self._target_mean_dev.read["gpu"]()
            td_error_mean = self._td_error_mean_dev.read["gpu"]()
            reward_mean = self._reward_mean_dev.read["gpu"]()
            done_mean = self._done_mean_dev.read["gpu"]()
        else:
            loss_mean = self._loss_accum * inv
            q_mean = self._q_accum * inv
            target_mean = self._target_accum * inv
            td_error_mean = self._td_error_accum * inv
            reward_mean = self._reward_accum * inv
            done_mean = self._done_accum * inv
        var bundle = DQNMetrics(
            loss=LogScalar[DT](loss_mean),
            explore_rate=LogScalar[DT](self.epsilon),
            mean_q=LogScalar[DT](q_mean),
            mean_target=LogScalar[DT](target_mean),
            td_error_abs_mean=LogScalar[DT](td_error_mean),
            reward_mean=LogScalar[DT](reward_mean),
            mean_done=LogScalar[DT](done_mean),
            train_steps=LogScalar[DT](Scalar[DT](self._total_train_steps)),
            n_updates=LogScalar[DT](Scalar[DT](self._update_count)),
        )
        self._loss_accum = Scalar[DT](0.0)
        self._q_accum = Scalar[DT](0.0)
        self._target_accum = Scalar[DT](0.0)
        self._td_error_accum = Scalar[DT](0.0)
        self._reward_accum = Scalar[DT](0.0)
        self._done_accum = Scalar[DT](0.0)
        self._update_count = 0
        comptime if Self.train_target == "gpu":
            self.q_update_blk.inner.mse_loss.reset_accum["gpu"]()
            self._q_mean_dev.reset["gpu"]()
            self._target_mean_dev.reset["gpu"]()
            self._td_error_mean_dev.reset["gpu"]()
            self._reward_mean_dev.reset["gpu"]()
            self._done_mean_dev.reset["gpu"]()
        if Bool(logger):
            log_bundle(logger.value()[], bundle, step)
        return bundle^

    # ─── Trait-uniform cadence hooks (consumed by the driver) ─────────

    def flush_metrics_through_logger[L: Logger](
        mut self,
        logger: Optional[Pointer[L, MutAnyOrigin]],
        step: Int,
    ) raises:
        """Trait-uniform passthrough: drains the DQN metric accumulators
        through `flush_metrics` and discards the typed bundle."""
        _ = self.flush_metrics[L](logger, step)

    def save_state(mut self, path: String) raises:
        """One-file v3 checkpoint of the ONLINE Q-net params + state, plus
        the ε-greedy exploration scalars + the cumulative train-step counter
        as `K` sections. Atomic and chunked. On GPU device params download to
        host first; the on-disk format is target-agnostic (train-on-GPU →
        eval-on-CPU loads). The target net is hard-copied from online on load."""
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
        """Inverse of `save_state`: restore online params + state, ε scalars +
        counter, then hard-copy online → target. Legacy v2 text files (the
        format this trainer wrote before) still load."""
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
        walk_params[Self.train_target](self.pair.online, r, self.ctx, "q_net"
        )
        r.mode = 1
        self.pair.online.for_each_state[Self.train_target](
            r, self.ctx, "q_net"
        )
        # Parse the trailing scalar lines (tolerant of order / absence).
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
        """Scan `content` lines for `key<value>`; return its float (or
        `default` if absent). Order-independent (the checkpoint reader consumed
        only the param/state lines; these trailing scalars are read here)."""
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
