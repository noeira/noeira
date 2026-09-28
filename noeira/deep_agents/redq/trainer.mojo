"""REDQTrainer — storage-framework REDQ trainer (CPU gate; GPU stretch).

Randomized Ensembled Double Q-learning (Chen et al., ICLR 2021). Mirrors the
storage `SACTrainer` shape — same `[train_target, SAMPLE, ACTOR, CRITIC]` family
plus `[N, N_MIN, UTD, POLICY_DELAY, Q_MODE]` ensemble knobs — and runs a UTD
inner critic loop per `train_step`:

    train_step(step_idx):                  # outer = 1 env step
        sample once (gates warmup)
        for inner = 0..UTD-1:
            resample subset                # Fisher-Yates of (N choose N_MIN)
            target-y     (EnsembleTargetYBlock)
            critic       (EnsembleCriticStep)
            polyak       (EnsemblePolyakStep — every inner step, paper-faithful)
            if (_inner_count % POLICY_DELAY) == 0:
                actor    (EnsembleActorStep — mean over N online critics)
                alpha    (AlphaUpdateStep — host ScalarAdam)

STORAGE migration (Stage 5): own scratch as `nn.storage.Tensor`s (not `Scratch`);
`Adam.adopt` on GPU; a one-file v3 checkpoint (α's optimizer and the counter
as `K` scalars); device-resident `DeviceMeanAccum` diagnostics on GPU / host
accumulators on CPU. α is a HOST scalar on both targets (the actor-loss block
D2Hs `log_prob_mean` already; REDQ doesn't capture under CUDA graphs because of
the subset-sampling + policy-delay host control flow — capture DEFERRED via the
trait-default no-ops).

Dimensions (OBS / ACT / BATCH) derive from `SAMPLE`.
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
from noeira.nn.core.call import call_forward
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Xavier, Zero
from noeira.nn.primitives.rsample import RSample
from noeira.nn.optimizer.adam import Adam
from noeira.nn.optimizer.scalar_adam import ScalarAdam
from noeira.nn.core.checkpoint import (
    BinaryCheckpointWriter, BinaryCheckpointReader, CheckpointReader,
    CheckpointScalars, write_model, read_model, _split_lines, _is_v3_header,
    _read_file_bytes, _write_file_bytes,
)

from noeira.nn.core.log_bundle import log_bundle
from noeira.nn.core.metric import LogScalar

from noeira.cuda import CUDAGraph, maybe_capture_replay

from ..data.n_step_replay import GPUNStepBuffer
from ..training.episode_tracker import EpisodeTracker
from ..training.device_mean_accum import DeviceMeanAccum
from ..training.trainer_block import TrainerState
from ..training.driver_offpolicy import OffPolicyAgentGpu
from ..training.blocks import SampleBlock
from ..training.blocks.action_select import (
    select_squashed_batched,
    warmup_uniform_batched,
)
from ..sac.blocks.alpha_update_step import AlphaUpdateStep

from .ensemble import CriticEnsemble
from .ensemble_target_y_block import EnsembleTargetYBlock
from .blocks.ensemble_critic_step import EnsembleCriticStep
from .blocks.ensemble_actor_step import EnsembleActorStep
from .blocks.ensemble_polyak_step import EnsemblePolyakStep
from .metrics import REDQMetrics


# ──────────────────────────────────────────────────────────────────────
# GPU select_action_batched kernels (mirror SAC's warmup + copy + clamp).
# ──────────────────────────────────────────────────────────────────────


struct REDQTrainer[
    train_target: StaticString,
    SAMPLE: SampleBlock,
    ACTOR: Module,
    CRITIC: Module,
    N: Int,
    N_MIN: Int,
    UTD: Int,
    POLICY_DELAY: Int,
    Q_MODE: Int,
    USE_TRAIN_CUDA_GRAPH: Bool = False,
](OffPolicyAgentGpu):
    """Storage-framework REDQ trainer. `N` total critics; `N_MIN` MIN-subset;
    `UTD` inner critic updates per env step; `POLICY_DELAY` actor+α cadence;
    `Q_MODE` ∈ {0=MIN, 1=AVE}.

    `USE_TRAIN_CUDA_GRAPH` (GPU + NoAMP; NVIDIA only, no-op elsewhere) captures
    the whole UTD inner loop into one CUDA graph and replays it per env-step —
    the launch-overhead fix. It switches the GPU inner step to a fully
    device-resident path: on-device subset resample (Philox), device α
    (`ScalarAdam.new_device` + `step_device`), and device loss/log-prob
    reductions (no per-step D2H). Mirrors MBPO's `_run_sac_updates_graph`."""

    comptime OBS_DIM: Int = Self.SAMPLE.OBS
    comptime ACT_DIM: Int = Self.SAMPLE.ACT
    comptime BATCH: Int = Self.SAMPLE.BATCH

    comptime AGENT_OBS_DIM: Int = Self.OBS_DIM
    comptime AGENT_ACT_DIM: Int = Self.ACT_DIM
    comptime AGENT_TRAIN_TARGET: StaticString = Self.train_target

    var actor: Self.ACTOR
    var ensemble: CriticEnsemble[Self.CRITIC, Self.N]
    var actor_opt: Adam
    var alpha_opt: ScalarAdam

    var sample_blk: Self.SAMPLE
    var target_y_blk: EnsembleTargetYBlock[
        Self.ACTOR, Self.CRITIC, Self.N, Self.BATCH, Self.OBS_DIM,
        Self.ACT_DIM, Self.N_MIN, Self.Q_MODE,
    ]
    var critic_blk: EnsembleCriticStep[
        Self.CRITIC, Self.N, Self.OBS_DIM, Self.ACT_DIM, Self.BATCH,
    ]
    var actor_blk: EnsembleActorStep[
        Self.ACTOR, Self.CRITIC, Self.N, Self.OBS_DIM, Self.ACT_DIM, Self.BATCH,
    ]
    var alpha_blk: AlphaUpdateStep[Self.OBS_DIM, Self.ACT_DIM, Self.BATCH]
    var polyak_blk: EnsemblePolyakStep[
        Self.CRITIC, Self.N, Self.OBS_DIM, Self.ACT_DIM, Self.BATCH,
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

    var _warmup_rng_seed: UInt64
    var _warmup_rng_offset: UInt64

    # UTD bookkeeping (drives the actor/α cadence).
    var _inner_count: Int

    # Host metric accumulators (CPU path).
    var _actor_L_accum: Scalar[DT]
    var _critic_L_accum: Scalar[DT]
    var _alpha_accum: Scalar[DT]
    var _q_accum: Scalar[DT]
    var _target_accum: Scalar[DT]
    var _reward_accum: Scalar[DT]
    var _done_accum: Scalar[DT]
    var _abs_action_accum: Scalar[DT]
    # GPU device-resident mirrors.
    var _q_mean_dev: DeviceMeanAccum
    var _target_mean_dev: DeviceMeanAccum
    var _reward_mean_dev: DeviceMeanAccum
    var _done_mean_dev: DeviceMeanAccum
    var _abs_action_mean_dev: DeviceMeanAccum
    var _update_count: Int        # inner steps this chunk
    var _actor_update_count: Int  # actor steps this chunk
    var _total_train_steps: Int   # cumulative inner steps (never reset)

    # CUDA-graph capture slot (USE_TRAIN_CUDA_GRAPH; None until first capture).
    var _train_graph: Optional[CUDAGraph]

    def __init__(out self):
        self.actor = Self.ACTOR()
        self.ensemble = CriticEnsemble[Self.CRITIC, Self.N]()
        self.actor_opt = Adam(lr=Scalar[DT](3e-4))
        self.alpha_opt = ScalarAdam.new(flog(Scalar[DT](0.2)), Scalar[DT](3e-4))
        self.sample_blk = Self.SAMPLE()
        self.target_y_blk = EnsembleTargetYBlock[
            Self.ACTOR, Self.CRITIC, Self.N, Self.BATCH, Self.OBS_DIM,
            Self.ACT_DIM, Self.N_MIN, Self.Q_MODE,
        ]()
        self.critic_blk = EnsembleCriticStep[
            Self.CRITIC, Self.N, Self.OBS_DIM, Self.ACT_DIM, Self.BATCH,
        ]()
        self.actor_blk = EnsembleActorStep[
            Self.ACTOR, Self.CRITIC, Self.N, Self.OBS_DIM, Self.ACT_DIM,
            Self.BATCH,
        ]()
        self.alpha_blk = AlphaUpdateStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH
        ]()
        self.polyak_blk = EnsemblePolyakStep[
            Self.CRITIC, Self.N, Self.OBS_DIM, Self.ACT_DIM, Self.BATCH,
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
        self._warmup_rng_seed = UInt64(0xC0FFEE_C0DE)
        self._warmup_rng_offset = UInt64(0)
        self._inner_count = 0
        self._actor_L_accum = Scalar[DT](0.0)
        self._critic_L_accum = Scalar[DT](0.0)
        self._alpha_accum = Scalar[DT](0.0)
        self._q_accum = Scalar[DT](0.0)
        self._target_accum = Scalar[DT](0.0)
        self._reward_accum = Scalar[DT](0.0)
        self._done_accum = Scalar[DT](0.0)
        self._abs_action_accum = Scalar[DT](0.0)
        self._q_mean_dev = DeviceMeanAccum()
        self._target_mean_dev = DeviceMeanAccum()
        self._reward_mean_dev = DeviceMeanAccum()
        self._done_mean_dev = DeviceMeanAccum()
        self._abs_action_mean_dev = DeviceMeanAccum()
        self._update_count = 0
        self._actor_update_count = 0
        self._total_train_steps = 0
        self._train_graph = None

    @staticmethod
    def make(
        ctx: Optional[DeviceContext] = None,
        actor_lr: Scalar[DT] = Scalar[DT](3e-4),
        critic_lr: Scalar[DT] = Scalar[DT](3e-4),
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
    ) raises -> Self:
        comptime assert (
            Self.train_target == "cpu" or Self.train_target == "gpu"
        ), "REDQTrainer: train_target must be 'cpu' or 'gpu'"
        comptime if Self.train_target == "gpu":
            if not ctx:
                raise Error("REDQTrainer.make[train_target='gpu']: ctx required")
        comptime assert Self.N >= 2, "REDQ: N must be ≥ 2"
        comptime assert Self.N_MIN >= 1, "REDQ: N_MIN must be ≥ 1"
        comptime assert Self.N_MIN <= Self.N, "REDQ: N_MIN must be ≤ N"
        comptime assert Self.UTD >= 1, "REDQ: UTD must be ≥ 1"
        comptime assert Self.POLICY_DELAY >= 1, "REDQ: POLICY_DELAY must be ≥ 1"

        var t = Self()
        t.ctx = ctx

        t.actor = Self.ACTOR.make[Self.train_target, Xavier](ctx)
        t.ensemble = CriticEnsemble[Self.CRITIC, Self.N].make[
            Self.train_target, Xavier
        ](ctx=ctx)

        t.actor_opt = Adam(lr=actor_lr)
        comptime if Self.train_target == "gpu":
            t.actor_opt.adopt[Self.train_target, Self.ACTOR](t.actor, ctx)
        for i in range(Self.N):
            t.ensemble.opts[i].lr = critic_lr

        # α temperature optimizer. The CUDA-graph path needs α on-device
        # (device ScalarAdam + the kernels read it via `alpha_dev_ptr`); the
        # eager path keeps the host ScalarAdam (bit-identical to before).
        comptime if Self.train_target == "gpu":
            if Self.USE_TRAIN_CUDA_GRAPH:
                t.alpha_opt = ScalarAdam.new_device(
                    ctx.value(), flog(init_alpha), alpha_lr
                )
            else:
                t.alpha_opt = ScalarAdam.new(flog(init_alpha), alpha_lr)
        else:
            t.alpha_opt = ScalarAdam.new(flog(init_alpha), alpha_lr)

        t.target_y_blk = EnsembleTargetYBlock[
            Self.ACTOR, Self.CRITIC, Self.N, Self.BATCH, Self.OBS_DIM,
            Self.ACT_DIM, Self.N_MIN, Self.Q_MODE,
        ].make[Self.train_target](
            action_scale=action_scale, gamma=gamma, ctx=ctx
        )
        t.critic_blk = EnsembleCriticStep[
            Self.CRITIC, Self.N, Self.OBS_DIM, Self.ACT_DIM, Self.BATCH,
        ].make[Self.train_target](ctx=ctx)
        t.actor_blk = EnsembleActorStep[
            Self.ACTOR, Self.CRITIC, Self.N, Self.OBS_DIM, Self.ACT_DIM,
            Self.BATCH,
        ].make[Self.train_target](action_scale=action_scale, ctx=ctx)
        t.alpha_blk = AlphaUpdateStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH
        ].make(target_entropy=target_entropy)
        t.polyak_blk = EnsemblePolyakStep[
            Self.CRITIC, Self.N, Self.OBS_DIM, Self.ACT_DIM, Self.BATCH,
        ].make(tau=tau)

        # CUDA-graph device path wiring: point target-y + actor-loss at the
        # on-device α buffer, and switch the subset draw to the on-device Philox
        # kernel — all host work removed from the inner step so it's capturable.
        comptime if Self.train_target == "gpu":
            if Self.USE_TRAIN_CUDA_GRAPH:
                t.target_y_blk.set_alpha_ptr(t.alpha_opt.alpha_dev_ptr())
                t.target_y_blk.enable_device_resample()
                t.actor_blk.inner.set_alpha_ptr(t.alpha_opt.alpha_dev_ptr())

        t.sel = RSample[Self.ACT_DIM].make[Self.train_target, Zero](ctx)
        t.sel.action_scale = action_scale

        t.state = TrainerState[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH
        ].make[Self.train_target](ctx=ctx)

        t.tracker = EpisodeTracker.new(
            window_size=window_size, initial_fill=initial_episode_fill
        )

        t.action_scale = action_scale
        t.learning_starts = learning_starts

        t.sample_blk.setup(learning_starts, ctx=ctx)

        comptime if Self.train_target == "cpu":
            t._ob_scr.ensure(Self.OBS_DIM)
            t._ao_scr.ensure(2 * Self.ACT_DIM)
            t._alp_scr.ensure(Self.ACT_DIM + 1)
        else:
            t._q_mean_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)
            t._target_mean_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)
            t._reward_mean_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)
            t._done_mean_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)
            t._abs_action_mean_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)
        return t^

    def set_beta(mut self, beta: Scalar[DT]):
        self.sample_blk.set_beta(beta)

    # ─── Inner step body — one (target_y + critic + polyak + maybe actor). ──
    def _run_inner_step(mut self) raises:
        self._inner_count += 1
        self.target_y_blk.resample_subset_idxs()

        var alpha_val = fexp(self.alpha_opt.value)
        self.state.alpha = alpha_val
        self.target_y_blk.step[Self.train_target](
            self.state, self.actor, self.ensemble, alpha_val
        )
        self.critic_blk.step[Self.train_target](self.state, self.ensemble)
        self.polyak_blk.step[Self.train_target](self.state, self.ensemble)

        if self._inner_count % Self.POLICY_DELAY == 0:
            self.actor_blk.step[Self.train_target](
                self.state, self.actor, self.actor_opt, self.ensemble
            )
            self.alpha_blk.step["cpu"](self.state, self.alpha_opt)
            self._actor_L_accum += self.state.actor_loss
            self._alpha_accum += fexp(self.alpha_opt.value)
            self._actor_update_count += 1

        # Per-batch diagnostics.
        comptime B = Self.BATCH
        comptime A = Self.ACT_DIM
        comptime if Self.train_target == "cpu":
            var inv_b = Scalar[DT](1.0) / Scalar[DT](B)
            var sq: Scalar[DT] = 0.0
            var sy: Scalar[DT] = 0.0
            var sr: Scalar[DT] = 0.0
            var sd: Scalar[DT] = 0.0
            for b in range(B):
                sq += self.critic_blk.member_step._mb_q.data[b]
                sy += self.state.mb_y.data[b]
                sr += self.state.mb_r.data[b]
                sd += self.state.mb_d.data[b]
            var saa: Scalar[DT] = 0.0
            for k in range(B * A):
                var av = self.state.mb_a.data[k]
                saa += av if av >= 0 else -av
            self._q_accum += sq * inv_b
            self._target_accum += sy * inv_b
            self._reward_accum += sr * inv_b
            self._done_accum += sd * inv_b
            self._abs_action_accum += saa / Scalar[DT](B * A)
        else:
            comptime lb = Layout.row_major(B)
            comptime lba = Layout.row_major(B * A)
            self._q_mean_dev.accumulate_gpu_lt[B](
                self.critic_blk.member_step._mb_q.lt["gpu", lb]()
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
            self._abs_action_mean_dev.accumulate_gpu_abs_lt[B * A](
                self.state.mb_a.lt["gpu", lba]()
            )

        self._critic_L_accum += self.state.critic_loss
        self._update_count += 1
        self._total_train_steps += 1

    # ─── Device inner step (CUDA-graph capture body) ───────────────────────
    def _run_inner_step_device(mut self, i: Int) raises:
        """One inner update as a PURE device-kernel sequence — the body captured
        into the CUDA graph and replayed. Sample (device RNG advances on-device),
        subset resample (device Philox), α read on-device, loss + diagnostics
        device-resident. The actor + α step run only on the policy-delay
        iteration (host `if` evaluated at capture → baked into the graph; valid
        because UTD/POLICY_DELAY are comptime-constant). NO host counters — those
        advance in `_run_inner_loop_graph`."""
        comptime assert Self.train_target == "gpu", (
            "_run_inner_step_device is GPU-only (CUDA-graph capture path)"
        )
        self.state.did_step = True
        self.state.ctx = self.ctx
        self.sample_blk.step(self.state)

        # α arg unused here — target_y reads α from the wired device buffer.
        self.target_y_blk.step["gpu"](
            self.state, self.actor, self.ensemble, self.state.alpha
        )
        self.critic_blk.step["gpu", ACCUMULATE=True](self.state, self.ensemble)
        self.polyak_blk.step["gpu"](self.state, self.ensemble)

        if (i + 1) % Self.POLICY_DELAY == 0:
            self.actor_blk.step["gpu"](
                self.state, self.actor, self.actor_opt, self.ensemble
            )
            self.alpha_opt.step_device(
                self.ctx.value(),
                self.actor_blk.inner.lp_mean_dev(),
                self.alpha_blk.target_entropy,
            )

        # Per-batch diagnostics — device-resident accumulators (capture-safe).
        comptime B = Self.BATCH
        comptime A = Self.ACT_DIM
        comptime lb = Layout.row_major(B)
        comptime lba = Layout.row_major(B * A)
        self._q_mean_dev.accumulate_gpu_lt[B](
            self.critic_blk.member_step._mb_q.lt["gpu", lb]()
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
        self._abs_action_mean_dev.accumulate_gpu_abs_lt[B * A](
            self.state.mb_a.lt["gpu", lba]()
        )

    def _run_inner_loop_graph(mut self) raises -> Bool:
        """Capture-once / replay the whole UTD inner loop as one CUDA graph
        (mirrors MBPO's `_run_sac_updates_graph`). The graph is moved into a
        disjoint local for the capture call so the closure can borrow `self`
        without overlapping the `_train_graph` field's mut borrow. Host counters
        advance once per logical inner update afterwards; losses/α/diagnostics
        drain from device accumulators at flush."""
        var ctx = self.ctx.value()
        var g = self._train_graph^
        self._train_graph = None

        def _captured() capturing raises -> None:
            for i in range(Self.UTD):
                self._run_inner_step_device(i)

        maybe_capture_replay[_captured](g, ctx)
        self._train_graph = g^

        for i in range(Self.UTD):
            self._update_count += 1
            self._total_train_steps += 1
            if (i + 1) % Self.POLICY_DELAY == 0:
                self._actor_update_count += 1
        return True

    # ─── train_step — outer (one env step). Runs UTD inner updates. ─────────
    def train_step(mut self, step_idx: Int) raises -> Bool:
        self.state.step_idx = step_idx
        self.state.did_step = True
        comptime if Self.train_target == "gpu":
            self.state.ctx = self.ctx

        self.sample_blk.step(self.state)
        if not self.state.did_step:
            return False

        # CUDA-graph path: capture/replay the whole UTD loop (GPU only). The
        # probe sample above gates warmup; the captured loop draws its own fresh
        # minibatches per iteration (device RNG advances on replay).
        comptime if Self.train_target == "gpu":
            comptime if Self.USE_TRAIN_CUDA_GRAPH:
                return self._run_inner_loop_graph()

        self._run_inner_step()
        for _ in range(Self.UTD - 1):
            self.state.did_step = True
            self.sample_blk.step(self.state)
            if not self.state.did_step:
                break
            self._run_inner_step()
        return True

    def total_train_steps(self) -> Int:
        return self._total_train_steps

    # ─── CUDA-graph capture surface (DEFERRED — trait-default no-ops) ───────
    # REDQ has host control flow (subset sampling + policy-delay gating), so the
    # per-update device-kernel sequence isn't capturable. The driver gates entry
    # on `learning_starts_count`; the capture body / counter advance are inherited
    # as the OffPolicyAgentGpu trait defaults (train_device_kernels raises;
    # note_train_update is a host no-op). REDQ runs the eager `train_step` path.

    def learning_starts_count(self) -> Int:
        return self.learning_starts

    # ─── Action selection ──────────────────────────────────────────────────
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
                self.actor,
                TensorRefs[Self.ACTOR.ARITY](self._ob_scr), self._ao_scr
            )
            for j in range(ACT):
                var a = ftanh(self._ao_scr.data[j]) * self.action_scale
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
            var ao = Tensor.alloc_gpu(c, 2 * ACT)
            call_forward["gpu", 1](
                self.actor,
                TensorRefs[Self.ACTOR.ARITY](ob), ao, self.ctx
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
                self.actor,
                TensorRefs[Self.ACTOR.ARITY](self._ob_scr), self._ao_scr
            )
            self.sel.forward["cpu", 1](
                TensorRefs[1](self._ao_scr), self._alp_scr
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
            var c = self.ctx.value()
            var ob = Tensor.alloc(OBS)
            for d in range(OBS):
                ob.data[d] = obs[d]
            ob.upload(c)
            var ao = Tensor.alloc_gpu(c, 2 * ACT)
            var alp = Tensor.alloc_gpu(c, ACT + 1)
            call_forward["gpu", 1](
                self.actor,
                TensorRefs[Self.ACTOR.ARITY](ob), ao, self.ctx
            )
            self.sel.forward["gpu", 1](TensorRefs[1](ao), alp, self.ctx)
            alp.download(c)
            for j in range(ACT):
                var a = alp.data[j]
                if a > self.action_scale:
                    a = self.action_scale
                elif a < -self.action_scale:
                    a = -self.action_scale
                action_out[j] = a

    # ─── Record ──────────────────────────────────────────────────────────
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
        raise Error(
            "REDQTrainer.record_batch_gpu_nstep: n-step replay not supported"
            " (uniform 1-step replay only)"
        )

    # ─── Metrics / logging ─────────────────────────────────────────────────
    def flush_metrics[
        L: Logger = NoOpLogger
    ](
        mut self,
        logger: Optional[Pointer[L, MutAnyOrigin]] = None,
        step: Int = 0,
    ) raises -> REDQMetrics:
        """Drain inner-step accumulators into a REDQMetrics bundle. critic_loss /
        mean_* averaged over `_update_count` inner steps; actor_loss / alpha over
        `_actor_update_count` actor steps."""
        var n = self._update_count if self._update_count > 0 else 1
        var inv = Scalar[DT](1.0) / Scalar[DT](n)
        var na = self._actor_update_count if self._actor_update_count > 0 else 1
        var inv_a = Scalar[DT](1.0) / Scalar[DT](na)

        var q_mean: Scalar[DT]
        var target_mean: Scalar[DT]
        var reward_mean: Scalar[DT]
        var done_mean: Scalar[DT]
        var abs_action_mean: Scalar[DT]
        comptime if Self.train_target == "gpu":
            q_mean = self._q_mean_dev.read["gpu"]()
            target_mean = self._target_mean_dev.read["gpu"]()
            reward_mean = self._reward_mean_dev.read["gpu"]()
            done_mean = self._done_mean_dev.read["gpu"]()
            abs_action_mean = self._abs_action_mean_dev.read["gpu"]()
        else:
            q_mean = self._q_accum * inv
            target_mean = self._target_accum * inv
            reward_mean = self._reward_accum * inv
            done_mean = self._done_accum * inv
            abs_action_mean = self._abs_action_accum * inv

        # actor/critic loss + α. CUDA-graph path: drain the device accumulators
        # (no host accumulation ran in the captured loop). Else host scalars.
        var actor_loss_v: Scalar[DT]
        var critic_loss_v: Scalar[DT]
        var alpha_v: Scalar[DT]
        comptime if Self.train_target == "gpu":
            comptime if Self.USE_TRAIN_CUDA_GRAPH:
                var c = self.ctx.value()
                actor_loss_v = self.actor_blk.inner.read_loss_accum(c)
                critic_loss_v = (
                    self.critic_blk.member_step.mse_loss.read_accum["gpu"](c)
                )
                alpha_v = self.alpha_opt.read_alpha()
                self.actor_blk.inner.reset_loss_accum()
                self.critic_blk.member_step.mse_loss.reset_accum["gpu"]()
            else:
                actor_loss_v = self._actor_L_accum * inv_a
                critic_loss_v = self._critic_L_accum * inv
                alpha_v = self._alpha_accum * inv_a
        else:
            actor_loss_v = self._actor_L_accum * inv_a
            critic_loss_v = self._critic_L_accum * inv
            alpha_v = self._alpha_accum * inv_a

        var bundle = REDQMetrics(
            policy_loss=LogScalar[DT](actor_loss_v),
            critic_loss=LogScalar[DT](critic_loss_v),
            alpha=LogScalar[DT](alpha_v),
            mean_q=LogScalar[DT](q_mean),
            mean_target=LogScalar[DT](target_mean),
            reward_mean=LogScalar[DT](reward_mean),
            mean_next_q=LogScalar[DT](Scalar[DT](0.0)),
            mean_done=LogScalar[DT](done_mean),
            action_abs_mean=LogScalar[DT](abs_action_mean),
            train_steps=LogScalar[DT](Scalar[DT](self._total_train_steps)),
            n_updates=LogScalar[DT](Scalar[DT](self._update_count)),
        )
        self._actor_L_accum = Scalar[DT](0.0)
        self._critic_L_accum = Scalar[DT](0.0)
        self._alpha_accum = Scalar[DT](0.0)
        self._q_accum = Scalar[DT](0.0)
        self._target_accum = Scalar[DT](0.0)
        self._reward_accum = Scalar[DT](0.0)
        self._done_accum = Scalar[DT](0.0)
        self._abs_action_accum = Scalar[DT](0.0)
        comptime if Self.train_target == "gpu":
            self._q_mean_dev.reset["gpu"]()
            self._target_mean_dev.reset["gpu"]()
            self._reward_mean_dev.reset["gpu"]()
            self._done_mean_dev.reset["gpu"]()
            self._abs_action_mean_dev.reset["gpu"]()
        self._update_count = 0
        self._actor_update_count = 0
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
        var na = self._actor_update_count if self._actor_update_count > 0 else 1
        var n = self._update_count if self._update_count > 0 else 1
        var out = (
            self._actor_L_accum / Scalar[DT](na),
            self._critic_L_accum / Scalar[DT](n),
            self._alpha_accum / Scalar[DT](na),
            self._update_count,
        )
        self._actor_L_accum = Scalar[DT](0.0)
        self._critic_L_accum = Scalar[DT](0.0)
        self._alpha_accum = Scalar[DT](0.0)
        self._update_count = 0
        self._actor_update_count = 0
        return out

    def flush_timer_log(mut self) -> String:
        return String("")

    # ─── Checkpoint (ONE v3 file: actor + N online critics + α + counter) ─
    def save_state(mut self, path: String) raises:
        """ONE v3 `storage-ckpt` file: the actor and the N ONLINE critics,
        name-prefixed `actor.` / `critic<i>.`, then as `K` sections the
        entropy temperature's whole optimizer (`alpha.*`) and the train-step
        counter. Atomic and chunked. Network optimizer moments NOT persisted
        (resume re-warms); α IS, because a resume that restarts it at the
        initial 0.2 undoes the part of the run that tuned it."""
        var w = BinaryCheckpointWriter(save_moments=False)
        write_model[Self.train_target](w, self.actor, self.ctx, "actor")
        for i in range(Self.N):
            write_model[Self.train_target](
                w, self.ensemble.pairs[i].online, self.ctx, "critic" + String(i)
            )
        var sc = CheckpointScalars()
        self.alpha_opt.put_state(sc, "alpha")
        sc.set_int("total_train_steps", self._total_train_steps)
        w.write_scalars(sc)
        _write_file_bytes(path, w.content)

    def load_state(mut self, path: String) raises:
        """Restore the actor, the N online critics, α and the counter (v3, or
        the legacy v2 text this trainer wrote before, which carries neither
        α nor the counter), then hard-copy online → target."""
        var bytes = _read_file_bytes(path)
        if _is_v3_header(bytes):
            var rb = BinaryCheckpointReader(bytes^)
            read_model[Self.train_target](rb, self.actor, self.ctx, "actor")
            for i in range(Self.N):
                read_model[Self.train_target](
                    rb, self.ensemble.pairs[i].online, self.ctx,
                    "critic" + String(i),
                )
            var sc = rb.read_scalars()
            rb.finish()
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
            for i in range(Self.N):
                walk_params[Self.train_target](self.ensemble.pairs[i].online, r, self.ctx, "critic" + String(i)
                )
            r.mode = 1
            var _sref2 = ParamVisitorRef.of[type_of(r), Self.train_target](r)
            self.actor.for_each_state[Self.train_target](_sref2, self.ctx, "actor")
            for i in range(Self.N):
                self.ensemble.pairs[i].online.for_each_state[Self.train_target](
                    r, self.ctx, "critic" + String(i)
                )
        for i in range(Self.N):
            self.ensemble.pairs[i].target_net.polyak_from[Self.train_target](
                self.ensemble.pairs[i].online, Scalar[DT](1.0), self.ctx
            )
