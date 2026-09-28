"""MBPOTrainer — CPU/GPU model-based policy optimization (storage framework).

Pipeline (per SAC sub-update):
  DualSample → TargetY → TwinCritic → SACActorLoss → AlphaUpdate → Polyak
(5 of 6 reused unchanged from the storage SAC blocks; all carry GPU paths.)

Dynamics ensemble training + synthetic rollouts are NOT pipeline blocks —
they're trainer methods invoked from train_step on a `model_train_freq`
cadence.

STORAGE migration (Stage 5): own scratch as `nn.storage.Tensor` (was legacy
`Scratch`/`TargetStorage`); storage `Adam` (decoupled `wd`); the actor update
reuses `SACActorLoss.forward_backward` DIRECTLY (the `SACActorStep` wrapper is
incompatible — it imports a legacy Adam) plus a separately-owned `RSample` for
select_action (mirrors the storage SAC trainer's `self.sel`). One-file v3
checkpoint (α, elites and the counter as `K` scalars). CUDA-graph capture DEFERRED via the
OffPolicyAgentGpu trait-default no-ops.

CPU is behaviorally equivalent to the prior CPU MBPOTrainer. Conforms to
`OffPolicyAgentGpu`.
"""

from noeira.nn.core.param import walk_params, ParamVisitorRef
from std.math import exp as fexp, sqrt as fsqrt, log as flog, tanh as ftanh
from std.random import random_float64, randn_float64
from std.time import perf_counter_ns
from max.gpu import block_dim, block_idx, thread_idx, global_idx
from max.gpu.host import DeviceContext, DeviceBuffer
from std.random.philox import Random as PhiloxRandom
from layout import Layout, LayoutTensor

from noeira.core.logger import Logger, NoOpLogger
from noeira.nn.constants import DT, TPB
from noeira.nn.core.module import Module
from noeira.nn.core.call import call_forward
from noeira.nn.core.amp import AMPPolicy, NoAMP, Bf16Compute
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Xavier, Kaiming
from noeira.nn.optimizer.adam import Adam
from noeira.nn.optimizer.scalar_adam import ScalarAdam
from noeira.nn.primitives.rsample import RSample
from noeira.nn.core.checkpoint import (
    BinaryCheckpointWriter, BinaryCheckpointReader, CheckpointReader,
    CheckpointScalars, write_model, read_model, _split_lines, _is_v3_header,
    _read_file_bytes, _write_file_bytes,
)

from noeira.nn.core.log_bundle import log_bundle
from noeira.nn.core.metric import LogScalar
from noeira.nn.random.box_muller import _box_muller_kernel
from noeira.cuda import CUDAGraph, maybe_capture_replay

from ..core.online_target_pair import OnlineTargetPair
from ..data.n_step_replay import GPUNStepBuffer
from ..training.episode_tracker import EpisodeTracker
from ..training.device_mean_accum import DeviceMeanAccum
from ..training.trainer_block import TrainerState
from ..training.driver_offpolicy import OffPolicyAgentGpu
from ..training.off_policy_critic import concat_sa_gpu
from ..training.blocks import DualSampleStep, TwinCriticStep, PolyakStep
from ..training.blocks.action_kernels import (
    offpolicy_clamp_action_kernel,
)
from ..training.blocks.action_select import (
    select_squashed_batched,
    warmup_uniform_batched,
)
from ..sac.target_y_block import TargetYBlock
from ..sac.actor_loss import SACActorLoss
from ..sac.blocks.alpha_update_step import AlphaUpdateStep
from .dynamics_ensemble_block import DynamicsEnsembleBlock
from .metrics import MBPOMetrics


# ──────────────────────────────────────────────────────────────────────
# Batched GPU action kernels (mirror SAC) + MBPO rollout/dynamics kernels.
# ──────────────────────────────────────────────────────────────────────


def _rollout_posterior_kernel[
    BATCH: Int, OBS: Int, PRED: Int
](
    obs: LayoutTensor[DT, Layout.row_major(BATCH, OBS), MutAnyOrigin],
    mu: LayoutTensor[DT, Layout.row_major(BATCH, PRED), MutAnyOrigin],
    lv: LayoutTensor[DT, Layout.row_major(BATCH, PRED), MutAnyOrigin],
    noise: LayoutTensor[DT, Layout.row_major(BATCH, PRED), MutAnyOrigin],
    out_nxt: LayoutTensor[DT, Layout.row_major(BATCH, OBS), MutAnyOrigin],
    out_rew: LayoutTensor[DT, Layout.row_major(BATCH), MutAnyOrigin],
):
    """Posterior sample one rollout step: PRED = 1 + OBS (reward + Δobs).
    Reward clamped to [-100, 100], NaN→0."""
    var k = Int(block_dim.x * block_idx.x + thread_idx.x)
    if k >= BATCH:
        return
    var mu_r = rebind[Scalar[DT]](mu[k, 0])
    var lv_r = rebind[Scalar[DT]](lv[k, 0])
    var z_r = rebind[Scalar[DT]](noise[k, 0])
    var r = mu_r + fexp(Scalar[DT](0.5) * lv_r) * z_r
    if r != r:
        r = Scalar[DT](0.0)
    elif r < Scalar[DT](-100.0):
        r = Scalar[DT](-100.0)
    elif r > Scalar[DT](100.0):
        r = Scalar[DT](100.0)
    out_rew[k] = r
    for d in range(OBS):
        var mu_d = rebind[Scalar[DT]](mu[k, 1 + d])
        var lv_d = rebind[Scalar[DT]](lv[k, 1 + d])
        var z_d = rebind[Scalar[DT]](noise[k, 1 + d])
        var delta = mu_d + fexp(Scalar[DT](0.5) * lv_d) * z_d
        out_nxt[k, d] = rebind[Scalar[DT]](obs[k, d]) + delta


def _mbpo_elite_assign_kernel[
    BATCH: Int
](
    slot: LayoutTensor[DT, Layout.row_major(BATCH), MutAnyOrigin],
    n_elites: Int32,
    seed: UInt64,
    offset: UInt64,
):
    """Per-transition random elite slot ∈ [0, n_elites), stored as DT."""
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= BATCH:
        return
    var philox = PhiloxRandom(seed=seed + UInt64(b), offset=offset)
    var u = Float32(philox.step_uniform()[0])
    var ne = Int(n_elites)
    var s = Int(u * Float32(ne))
    if s >= ne:
        s = ne - 1
    if s < 0:
        s = 0
    slot[b] = Scalar[DT](s)


def _mbpo_elite_gather_kernel[
    BATCH: Int, NELITES: Int, PRED: Int
](
    mu_all: LayoutTensor[
        DT, Layout.row_major(NELITES * BATCH, PRED), MutAnyOrigin
    ],
    lv_all: LayoutTensor[
        DT, Layout.row_major(NELITES * BATCH, PRED), MutAnyOrigin
    ],
    slot: LayoutTensor[DT, Layout.row_major(BATCH), MutAnyOrigin],
    out_mu: LayoutTensor[DT, Layout.row_major(BATCH, PRED), MutAnyOrigin],
    out_lv: LayoutTensor[DT, Layout.row_major(BATCH, PRED), MutAnyOrigin],
):
    """Per-transition gather: `out[b,:] = all[slot[b]·BATCH + b, :]`."""
    var b = Int(block_dim.x * block_idx.x + thread_idx.x)
    if b >= BATCH:
        return
    var s = Int(rebind[Scalar[DT]](slot[b]))
    if s < 0:
        s = 0
    if s >= NELITES:
        s = NELITES - 1
    var row = s * BATCH + b
    for d in range(PRED):
        out_mu[b, d] = rebind[Scalar[DT]](mu_all[row, d])
        out_lv[b, d] = rebind[Scalar[DT]](lv_all[row, d])


def _build_dyn_target_kernel[
    BATCH: Int, OBS: Int, PRED: Int
](
    rew: LayoutTensor[DT, Layout.row_major(BATCH), MutAnyOrigin],
    s: LayoutTensor[DT, Layout.row_major(BATCH, OBS), MutAnyOrigin],
    sp: LayoutTensor[DT, Layout.row_major(BATCH, OBS), MutAnyOrigin],
    out_tgt: LayoutTensor[DT, Layout.row_major(BATCH, PRED), MutAnyOrigin],
):
    """Dynamics target = [reward, Δobs = s' − s]. PRED = 1 + OBS."""
    var k = Int(block_dim.x * block_idx.x + thread_idx.x)
    if k >= BATCH:
        return
    out_tgt[k, 0] = rebind[Scalar[DT]](rew[k])
    for d in range(OBS):
        out_tgt[k, 1 + d] = rebind[Scalar[DT]](sp[k, d]) - rebind[Scalar[DT]](
            s[k, d]
        )


def _normalize_input_kernel[
    BATCH: Int, D: Int
](
    data: LayoutTensor[DT, Layout.row_major(BATCH, D), MutAnyOrigin],
    mean: LayoutTensor[DT, Layout.row_major(D), MutAnyOrigin],
    std: LayoutTensor[DT, Layout.row_major(D), MutAnyOrigin],
):
    """In-place per-column z-score of the dynamics input."""
    var i = Int(block_dim.x * block_idx.x + thread_idx.x)
    if i >= BATCH * D:
        return
    var b = i // D
    var c = i % D
    var v = rebind[Scalar[DT]](data[b, c])
    var m = rebind[Scalar[DT]](mean[c])
    var s = rebind[Scalar[DT]](std[c])
    data[b, c] = (v - m) / s


def _mbpo_fit_scaler_kernel[
    OBS: Int, ACT: Int, DYN_IN: Int, CAP: Int
](
    obs: LayoutTensor[DT, Layout.row_major(CAP, OBS), MutAnyOrigin],
    act: LayoutTensor[DT, Layout.row_major(CAP, ACT), MutAnyOrigin],
    mean_out: LayoutTensor[DT, Layout.row_major(DYN_IN), MutAnyOrigin],
    std_out: LayoutTensor[DT, Layout.row_major(DYN_IN), MutAnyOrigin],
    n_data: Int32,
):
    """Per-column z-score fit of the dynamics input over the first `n_data`
    rows of the real buffer (obs ++ act). One thread per DYN_IN column — a
    fully on-device replacement for the old full-buffer D2H + host loop +
    H2D + double `ctx.synchronize()`. `std < 1e-12 → 1` (matches the host fit)."""
    var c = Int(block_dim.x * block_idx.x + thread_idx.x)
    if c >= DYN_IN:
        return
    var n = Int(n_data)
    if n < 1:
        mean_out[c] = Scalar[DT](0.0)
        std_out[c] = Scalar[DT](1.0)
        return
    var inv_n = Scalar[DT](1.0) / Scalar[DT](n)
    var sum = Scalar[DT](0.0)
    for i in range(n):
        if c < OBS:
            sum += rebind[Scalar[DT]](obs[i, c])
        else:
            sum += rebind[Scalar[DT]](act[i, c - OBS])
    var mean = sum * inv_n
    var ss = Scalar[DT](0.0)
    for i in range(n):
        var v: Scalar[DT]
        if c < OBS:
            v = rebind[Scalar[DT]](obs[i, c])
        else:
            v = rebind[Scalar[DT]](act[i, c - OBS])
        var d = v - mean
        ss += d * d
    var std = fsqrt(ss * inv_n)
    if std < Scalar[DT](1e-12):
        std = Scalar[DT](1.0)
    mean_out[c] = mean
    std_out[c] = std


struct MBPOTrainer[
    train_target: StaticString,
    ACTOR: Module,
    CRITIC: Module,
    DynNet: Module,
    OBS_DIM: Int,
    ACT_DIM: Int,
    BATCH: Int,
    REPLAY_CAPACITY: Int,
    SYNTH_CAPACITY: Int,
    N_ENSEMBLE: Int,
    NUM_ELITES: Int,
    REAL_RATIO_PCT: Int = 5,
    LOGVAR_MIN: Float64 = -10.0,
    LOGVAR_MAX: Float64 = -2.0,
    USE_TRAIN_CUDA_GRAPH: Bool = False,
](OffPolicyAgentGpu):
    comptime DYN_IN: Int = Self.OBS_DIM + Self.ACT_DIM
    comptime DYN_PRED: Int = 1 + Self.OBS_DIM
    comptime DYN_OUT: Int = 2 * Self.DYN_PRED
    comptime REAL_BS: Int = (Self.BATCH * Self.REAL_RATIO_PCT) // 100
    comptime SYNTH_BS: Int = Self.BATCH - Self.REAL_BS

    comptime ENSEMBLE = DynamicsEnsembleBlock[
        Self.DynNet,
        Self.N_ENSEMBLE,
        Self.NUM_ELITES,
        Self.DYN_IN,
        Self.DYN_OUT,
        Self.BATCH,
        Self.LOGVAR_MIN,
        Self.LOGVAR_MAX,
    ]

    comptime SampleBlk = DualSampleStep[
        Self.OBS_DIM,
        Self.ACT_DIM,
        Self.BATCH,
        Self.REPLAY_CAPACITY,
        Self.SYNTH_CAPACITY,
        Self.REAL_BS,
        Self.SYNTH_BS,
    ]

    comptime AGENT_OBS_DIM: Int = Self.OBS_DIM
    comptime AGENT_ACT_DIM: Int = Self.ACT_DIM
    comptime AGENT_TRAIN_TARGET: StaticString = Self.train_target

    comptime _T_DYN_TRAIN = 0
    comptime _T_ROLLOUT = 1
    comptime _T_SAMPLE = 2
    comptime _T_TARGET_Y = 3
    comptime _T_CRITIC = 4
    comptime _T_ACTOR = 5
    comptime _T_ALPHA = 6
    comptime _T_POLYAK = 7
    comptime _T_DIAG = 8

    var actor: Self.ACTOR
    var pair1: OnlineTargetPair[Self.CRITIC]
    var pair2: OnlineTargetPair[Self.CRITIC]
    var actor_opt: Adam
    var critic1_opt: Adam
    var critic2_opt: Adam
    var alpha_opt: ScalarAdam

    var sample_blk: Self.SampleBlk
    var target_y_blk: TargetYBlock[
        Self.ACTOR,
        Self.CRITIC,
        Self.BATCH,
        Self.OBS_DIM,
        Self.ACT_DIM,
    ]
    var twin_critic_blk: TwinCriticStep[
        Self.OBS_DIM,
        Self.ACT_DIM,
        Self.BATCH,
        Self.CRITIC,
    ]
    var actor_loss_blk: SACActorLoss[Self.ACTOR, Self.CRITIC, Self.BATCH]
    var alpha_blk: AlphaUpdateStep[Self.OBS_DIM, Self.ACT_DIM, Self.BATCH]
    var polyak_blk: PolyakStep[
        Self.OBS_DIM,
        Self.ACT_DIM,
        Self.BATCH,
        Self.CRITIC,
    ]

    # Shared rsample for select_action (the actor-loss block owns its own).
    var sel: RSample[Self.ACT_DIM]

    var ensemble: Self.ENSEMBLE
    var state: TrainerState[Self.OBS_DIM, Self.ACT_DIM, Self.BATCH]
    var tracker: EpisodeTracker
    var ctx: Optional[DeviceContext]

    # select_action scratch (owned Tensors).
    var _ob1: Tensor   # OBS_DIM
    var _ao1: Tensor   # 2 * ACT_DIM
    var _alp1: Tensor  # ACT_DIM + 1

    # Dynamics training / rollout scratches (owned Tensors, target-resident).
    var _dyn_in: Tensor    # BATCH * DYN_IN
    var _dyn_tgt: Tensor   # BATCH * DYN_PRED
    var _ro_obs: Tensor
    var _ro_act: Tensor
    var _ro_nxt: Tensor
    var _ro_mu: Tensor
    var _ro_lv: Tensor
    var _ro_mu_all: Tensor  # NUM_ELITES * BATCH * DYN_PRED
    var _ro_lv_all: Tensor
    var _ro_slot: Tensor    # BATCH
    var _ro_rew: Tensor
    var _ro_done: Tensor
    var _ro_noise: Tensor   # BATCH * DYN_PRED
    var _ro_ao: Tensor      # BATCH * 2*ACT
    var _ro_alp: Tensor     # BATCH * (ACT + 1)

    # Dynamics input scaler (staging — host mirror + device).
    var _in_mean: Tensor   # DYN_IN
    var _in_std: Tensor    # DYN_IN

    var action_scale: Scalar[DT]
    var learning_starts: Int

    var model_train_freq: Int
    var dyn_epochs_per_round: Int
    var rollout_length: Int
    var num_rollouts_per_step: Int
    var sac_updates_per_step: Int
    var dyn_batch_size: Int
    var last_dyn_step: Int
    var _use_bf16: Bool
    # CUDA-graph capture (gated by the comptime `USE_TRAIN_CUDA_GRAPH`; NVIDIA
    # only). `_train_graph` captures the SAC sub-update loop (one cuGraphLaunch
    # per env-step vs ~100+ launches/update). `_dyn_graphs[m]` captures each
    # dynamics member's `train_member_step` (the batch-build stays eager, so the
    # graph reads the persistent `_dyn_in`/`_dyn_tgt` buffers and replays across
    # rounds with no re-instantiate). Both fix the launch-bound profile.
    var _train_graph: Optional[CUDAGraph]
    var _dyn_graphs: List[Optional[CUDAGraph]]
    var dyn_holdout_ratio: Scalar[DT]
    var dyn_max_epochs: Int
    var dyn_patience: Int
    var dyn_holdout_check_every: Int

    var _warmup_rng_seed: UInt64
    var _warmup_rng_offset: UInt64
    var _roll_rng_seed: UInt64
    var _roll_rng_offset: UInt64
    var _elite_rng_offset: UInt64

    var _actor_L_accum: Scalar[DT]
    var _critic_L_accum: Scalar[DT]
    var _update_count: Int
    var _total_train_steps: Int

    var _q_accum: Scalar[DT]
    var _reward_accum: Scalar[DT]
    var _td_accum: Scalar[DT]
    var _done_accum: Scalar[DT]
    var _action_abs_accum: Scalar[DT]
    var _dyn_loss_accum: Scalar[DT]
    var _dyn_step_count: Int
    var _dyn_loss_last: Scalar[DT]
    var _dyn_holdout_loss: Scalar[DT]
    var _dyn_holdout_min: Scalar[DT]
    var _dyn_holdout_max: Scalar[DT]
    var _dyn_input_std_mean: Scalar[DT]

    var _q_mean_dev: DeviceMeanAccum
    var _reward_mean_dev: DeviceMeanAccum
    var _td_mean_dev: DeviceMeanAccum
    var _done_mean_dev: DeviceMeanAccum
    var _action_abs_mean_dev: DeviceMeanAccum

    def __init__(out self):
        comptime assert (
            Self.DynNet.IN_DIMS[0] == Self.DYN_IN
        ), "MBPOTrainer: DynNet.IN_DIM must equal OBS_DIM + ACT_DIM"
        comptime assert (
            Self.DynNet.OUT_DIM == Self.DYN_OUT
        ), "MBPOTrainer: DynNet.OUT_DIM must equal 2 * (1 + OBS_DIM)"
        comptime assert (
            Self.REAL_RATIO_PCT >= 0 and Self.REAL_RATIO_PCT <= 100
        ), "REAL_RATIO_PCT must be in [0, 100]"
        comptime assert Self.REAL_BS >= 1, "REAL_BS must be >= 1"
        comptime assert Self.SYNTH_BS >= 1, "SYNTH_BS must be >= 1"

        self.actor = Self.ACTOR()
        self.pair1 = OnlineTargetPair[Self.CRITIC]()
        self.pair2 = OnlineTargetPair[Self.CRITIC]()
        self.actor_opt = Adam(lr=Scalar[DT](3e-4))
        self.critic1_opt = Adam(lr=Scalar[DT](3e-4))
        self.critic2_opt = Adam(lr=Scalar[DT](3e-4))
        self.alpha_opt = ScalarAdam.new(flog(Scalar[DT](0.2)), Scalar[DT](3e-4))
        self.sample_blk = Self.SampleBlk()
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
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH,
        ]()
        self.polyak_blk = PolyakStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.CRITIC,
        ]()
        self.sel = RSample[Self.ACT_DIM]()
        self.ensemble = Self.ENSEMBLE()
        self.state = TrainerState[Self.OBS_DIM, Self.ACT_DIM, Self.BATCH]()
        self.tracker = EpisodeTracker(
            window=List[Scalar[DT]](),
            window_size=0,
            idx=0,
            current_return=Scalar[DT](0.0),
            ep_count=0,
        )
        self.ctx = None
        self._ob1 = Tensor()
        self._ao1 = Tensor()
        self._alp1 = Tensor()
        self._dyn_in = Tensor()
        self._dyn_tgt = Tensor()
        self._ro_obs = Tensor()
        self._ro_act = Tensor()
        self._ro_nxt = Tensor()
        self._ro_mu = Tensor()
        self._ro_lv = Tensor()
        self._ro_mu_all = Tensor()
        self._ro_lv_all = Tensor()
        self._ro_slot = Tensor()
        self._ro_rew = Tensor()
        self._ro_done = Tensor()
        self._ro_noise = Tensor()
        self._ro_ao = Tensor()
        self._ro_alp = Tensor()
        self._in_mean = Tensor()
        self._in_std = Tensor()
        self.action_scale = Scalar[DT](1.0)
        self.learning_starts = 1_000
        self.model_train_freq = 250
        self.dyn_epochs_per_round = 4
        self.rollout_length = 1
        self.num_rollouts_per_step = 400
        self.sac_updates_per_step = 20
        self.dyn_batch_size = 256
        self.dyn_holdout_ratio = Scalar[DT](0.2)
        self.dyn_max_epochs = 40
        self.dyn_patience = 5
        self.dyn_holdout_check_every = 5
        self.last_dyn_step = -1
        self._use_bf16 = False
        self._train_graph = None
        self._dyn_graphs = List[Optional[CUDAGraph]]()
        self._warmup_rng_seed = UInt64(0xC0FFEE_C0DE)
        self._warmup_rng_offset = UInt64(0)
        self._roll_rng_seed = UInt64(0xB0A75E_D00D)
        self._roll_rng_offset = UInt64(0)
        self._elite_rng_offset = UInt64(0)
        self._actor_L_accum = Scalar[DT](0.0)
        self._critic_L_accum = Scalar[DT](0.0)
        self._update_count = 0
        self._total_train_steps = 0
        self._q_accum = Scalar[DT](0.0)
        self._reward_accum = Scalar[DT](0.0)
        self._td_accum = Scalar[DT](0.0)
        self._done_accum = Scalar[DT](0.0)
        self._action_abs_accum = Scalar[DT](0.0)
        self._dyn_loss_accum = Scalar[DT](0.0)
        self._dyn_step_count = 0
        self._dyn_loss_last = Scalar[DT](0.0)
        self._dyn_holdout_loss = Scalar[DT](0.0)
        self._dyn_holdout_min = Scalar[DT](0.0)
        self._dyn_holdout_max = Scalar[DT](0.0)
        self._dyn_input_std_mean = Scalar[DT](0.0)
        self._q_mean_dev = DeviceMeanAccum()
        self._reward_mean_dev = DeviceMeanAccum()
        self._td_mean_dev = DeviceMeanAccum()
        self._done_mean_dev = DeviceMeanAccum()
        self._action_abs_mean_dev = DeviceMeanAccum()

    @staticmethod
    def make(
        ctx: Optional[DeviceContext] = None,
        actor_lr: Scalar[DT] = Scalar[DT](3e-4),
        critic_lr: Scalar[DT] = Scalar[DT](3e-4),
        alpha_lr: Scalar[DT] = Scalar[DT](3e-4),
        model_lr: Scalar[DT] = Scalar[DT](1e-3),
        gamma: Scalar[DT] = Scalar[DT](0.99),
        tau: Scalar[DT] = Scalar[DT](0.005),
        action_scale: Scalar[DT] = Scalar[DT](1.0),
        init_alpha: Scalar[DT] = Scalar[DT](0.2),
        target_entropy: Scalar[DT] = Scalar[DT](-1.0),
        learning_starts: Int = 1_000,
        window_size: Int = 10,
        initial_episode_fill: Scalar[DT] = Scalar[DT](-1250.0),
        model_train_freq: Int = 250,
        dyn_epochs_per_round: Int = 4,
        rollout_length: Int = 1,
        num_rollouts_per_step: Int = 400,
        sac_updates_per_step: Int = 20,
        dyn_batch_size: Int = 256,
        dyn_max_epochs: Int = 40,
        dyn_weight_decay: Scalar[DT] = Scalar[DT](5e-5),
        dyn_learnable_bounds: Bool = False,
        use_bf16: Bool = False,
    ) raises -> Self:
        comptime assert (
            Self.train_target == "cpu" or Self.train_target == "gpu"
        ), "MBPOTrainer: target must be 'cpu' or 'gpu'"
        comptime if Self.train_target == "gpu":
            if not ctx:
                raise Error("MBPOTrainer.make[target='gpu']: ctx required")

        var t = Self()
        t.ctx = ctx

        t.actor = Self.ACTOR.make[Self.train_target, Xavier](ctx)
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
            t.actor_opt.adopt[Self.train_target, Self.ACTOR](t.actor, ctx)
            t.critic1_opt.adopt[Self.train_target, Self.CRITIC](
                t.pair1.online, ctx
            )
            t.critic2_opt.adopt[Self.train_target, Self.CRITIC](
                t.pair2.online, ctx
            )

        t.target_y_blk = TargetYBlock[
            Self.ACTOR, Self.CRITIC, Self.BATCH, Self.OBS_DIM, Self.ACT_DIM,
        ].make[Self.train_target](
            action_scale=action_scale, gamma=gamma, ctx=ctx
        )
        t.twin_critic_blk = TwinCriticStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.CRITIC,
        ].make[Self.train_target](ctx=ctx)
        t.actor_loss_blk = SACActorLoss[
            Self.ACTOR, Self.CRITIC, Self.BATCH
        ].make[Self.train_target](ctx=ctx, action_scale=action_scale)

        comptime if Self.train_target == "gpu":
            t.alpha_opt = ScalarAdam.new_device(
                ctx.value(), flog(init_alpha), alpha_lr
            )
            t.target_y_blk.set_alpha_ptr(t.alpha_opt.alpha_dev_ptr())
            t.actor_loss_blk.set_alpha_buf(t.alpha_opt.alpha_dev_buffer())
            t._q_mean_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)
            t._reward_mean_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)
            t._td_mean_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)
            t._done_mean_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)
            t._action_abs_mean_dev = DeviceMeanAccum.make["gpu"](ctx=ctx)
        else:
            t.alpha_opt = ScalarAdam.new(flog(init_alpha), alpha_lr)

        t.alpha_blk = AlphaUpdateStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH,
        ].make(target_entropy=target_entropy)
        t.polyak_blk = PolyakStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH, Self.CRITIC,
        ].make(tau=tau)

        t.sel = RSample[Self.ACT_DIM].make[Self.train_target, Xavier](ctx)
        t.sel.action_scale = action_scale

        t.ensemble = Self.ENSEMBLE.make[Self.train_target, Kaiming](ctx)
        t.ensemble.set_lr(model_lr)
        t.ensemble.set_weight_decay(dyn_weight_decay)
        if dyn_learnable_bounds:
            t.ensemble.enable_learnable_bounds()

        t.tracker = EpisodeTracker.new(
            window_size=window_size, initial_fill=initial_episode_fill
        )
        t.state = TrainerState[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH,
        ].make[Self.train_target](ctx=ctx)

        # Owned scratch (target-resident).
        t._ob1 = Tensor.make[Self.train_target](Self.OBS_DIM, ctx)
        t._ao1 = Tensor.make[Self.train_target](2 * Self.ACT_DIM, ctx)
        t._alp1 = Tensor.make[Self.train_target](Self.ACT_DIM + 1, ctx)
        # _ob1 needs a host mirror for staging on GPU.
        comptime if Self.train_target == "gpu":
            t._ob1.ensure(Self.OBS_DIM)
            t._ao1.ensure(2 * Self.ACT_DIM)
            t._alp1.ensure(Self.ACT_DIM + 1)
        t._dyn_in = Tensor.make[Self.train_target](Self.BATCH * Self.DYN_IN, ctx)
        t._dyn_tgt = Tensor.make[Self.train_target](
            Self.BATCH * Self.DYN_PRED, ctx
        )
        t._ro_obs = Tensor.make[Self.train_target](
            Self.BATCH * Self.OBS_DIM, ctx
        )
        t._ro_act = Tensor.make[Self.train_target](
            Self.BATCH * Self.ACT_DIM, ctx
        )
        t._ro_nxt = Tensor.make[Self.train_target](
            Self.BATCH * Self.OBS_DIM, ctx
        )
        t._ro_mu = Tensor.make[Self.train_target](
            Self.BATCH * Self.DYN_PRED, ctx
        )
        t._ro_lv = Tensor.make[Self.train_target](
            Self.BATCH * Self.DYN_PRED, ctx
        )
        t._ro_mu_all = Tensor.make[Self.train_target](
            Self.NUM_ELITES * Self.BATCH * Self.DYN_PRED, ctx
        )
        t._ro_lv_all = Tensor.make[Self.train_target](
            Self.NUM_ELITES * Self.BATCH * Self.DYN_PRED, ctx
        )
        t._ro_slot = Tensor.make[Self.train_target](Self.BATCH, ctx)
        t._ro_rew = Tensor.make[Self.train_target](Self.BATCH, ctx)
        t._ro_done = Tensor.make[Self.train_target](Self.BATCH, ctx)
        t._ro_noise = Tensor.make[Self.train_target](
            Self.BATCH * Self.DYN_PRED, ctx
        )
        t._ro_ao = Tensor.make[Self.train_target](
            Self.BATCH * 2 * Self.ACT_DIM, ctx
        )
        t._ro_alp = Tensor.make[Self.train_target](
            Self.BATCH * (Self.ACT_DIM + 1), ctx
        )
        # Input scaler — host mirror + device (staging).
        t._in_mean = Tensor.make[Self.train_target](Self.DYN_IN, ctx)
        t._in_std = Tensor.make[Self.train_target](Self.DYN_IN, ctx)
        comptime if Self.train_target == "gpu":
            t._in_mean.ensure(Self.DYN_IN)
            t._in_std.ensure(Self.DYN_IN)

        t.action_scale = action_scale
        t.learning_starts = learning_starts
        t.model_train_freq = model_train_freq
        t.dyn_epochs_per_round = dyn_epochs_per_round
        t.rollout_length = rollout_length
        t.num_rollouts_per_step = num_rollouts_per_step
        t.sac_updates_per_step = sac_updates_per_step
        t.dyn_batch_size = dyn_batch_size
        t.dyn_max_epochs = dyn_max_epochs
        t._use_bf16 = use_bf16
        # Per-member dynamics-train graph slots (capture-once / replay). Always
        # populated so indexing is valid; only used when USE_TRAIN_CUDA_GRAPH.
        comptime if Self.USE_TRAIN_CUDA_GRAPH:
            for _ in range(Self.N_ENSEMBLE):
                t._dyn_graphs.append(None)

        t.sample_blk.setup[Self.train_target](learning_starts, ctx=ctx)
        t._set_scaler_identity[Self.train_target]()
        return t^

    # ─── Direct-callable (host-list) surface ─────────────────────────

    def select_action(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
        step_idx: Int,
    ) raises:
        comptime ACT = Self.ACT_DIM
        comptime OBS = Self.OBS_DIM
        if step_idx < self.learning_starts:
            for j in range(ACT):
                var u = Scalar[DT](2.0 * random_float64() - 1.0)
                action_out[j] = u * self.action_scale
            return
        comptime if Self.train_target == "cpu":
            self._ob1.ensure(OBS)
            self._ao1.ensure(2 * ACT)
            self._alp1.ensure(ACT + 1)
            for d in range(OBS):
                self._ob1.data[d] = obs[d]
            call_forward["cpu", 1](
                self.actor, TensorRefs[Self.ACTOR.ARITY](self._ob1), self._ao1
            )
            self.sel.forward["cpu", 1](
                TensorRefs[1](self._ao1), self._alp1
            )
            for j in range(ACT):
                var a = self._alp1.data[j]
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
            var alp = Tensor.alloc_gpu(c, ACT + 1)
            call_forward["gpu", 1](
                self.actor, TensorRefs[Self.ACTOR.ARITY](ob), ao, self.ctx
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

    def _select_action_into_cpu(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
    ) raises:
        """CPU policy action (no warmup gate) — used by the CPU rollout loop."""
        comptime ACT = Self.ACT_DIM
        comptime OBS = Self.OBS_DIM
        self._ob1.ensure(OBS)
        self._ao1.ensure(2 * ACT)
        self._alp1.ensure(ACT + 1)
        for d in range(OBS):
            self._ob1.data[d] = obs[d]
        call_forward["cpu", 1](
            self.actor, TensorRefs[Self.ACTOR.ARITY](self._ob1), self._ao1
        )
        self.sel.forward["cpu", 1](TensorRefs[1](self._ao1), self._alp1)
        for j in range(ACT):
            var a = self._alp1.data[j]
            if a > self.action_scale:
                a = self.action_scale
            elif a < -self.action_scale:
                a = -self.action_scale
            action_out[j] = a

    def select_greedy_action(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
    ) raises:
        comptime ACT = Self.ACT_DIM
        comptime OBS = Self.OBS_DIM
        comptime if Self.train_target == "cpu":
            self._ob1.ensure(OBS)
            self._ao1.ensure(2 * ACT)
            for d in range(OBS):
                self._ob1.data[d] = obs[d]
            call_forward["cpu", 1](
                self.actor, TensorRefs[Self.ACTOR.ARITY](self._ob1), self._ao1
            )
            for j in range(ACT):
                var a = ftanh(self._ao1.data[j]) * self.action_scale
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

    def record(
        mut self,
        ref obs: List[Scalar[DT]],
        ref action: List[Scalar[DT]],
        reward: Scalar[DT],
        ref next_obs: List[Scalar[DT]],
        done: Scalar[DT],
    ) raises:
        self.tracker.add_reward(reward)
        self.sample_blk.real_add[Self.train_target](
            obs, action, reward, next_obs, done, ctx=self.ctx,
        )

    def _tracker_ptr(self) -> Pointer[EpisodeTracker, MutAnyOrigin]:
        return rebind[Pointer[EpisodeTracker, MutAnyOrigin]](
            Pointer(to=self.tracker)
        )

    def train_step(mut self, step_idx: Int) raises -> Bool:
        if step_idx < self.learning_starts:
            return False

        var should_train_dyn = (
            self.last_dyn_step < 0
            or step_idx - self.last_dyn_step >= self.model_train_freq
        )
        if should_train_dyn:
            comptime if Self.train_target == "gpu":
                self._fit_input_scaler_gpu()
            else:
                self._fit_input_scaler_cpu()

            comptime if Self.train_target == "gpu":
                self._train_dynamics_ensemble_gpu()
            else:
                self._train_dynamics_ensemble()

            comptime if Self.train_target == "gpu":
                self._generate_synthetic_rollouts_gpu()
            else:
                self._generate_synthetic_rollouts()
            self.last_dyn_step = step_idx

        if self.sample_blk.real_count[Self.train_target]() < Self.REAL_BS:
            return False
        if self.sample_blk.synth_count[Self.train_target]() < Self.SYNTH_BS:
            return False

        comptime if Self.train_target == "cpu":
            return self._run_sac_updates[NoAMP](step_idx)
        else:
            if self._use_bf16:
                return self._run_sac_updates[Bf16Compute](step_idx)
            # CUDA-graph capture/replay of the SAC sub-update loop (NoAMP only;
            # AMP + capture has known stdlib issues). Comptime-gated so the
            # eager path is the only one compiled when the flag is off.
            comptime if Self.USE_TRAIN_CUDA_GRAPH:
                return self._run_sac_updates_graph()
            else:
                return self._run_sac_updates[NoAMP](step_idx)

    def _run_sac_updates[
        POLICY: AMPPolicy = NoAMP,
    ](mut self, step_idx: Int) raises -> Bool:
        """The `sac_updates_per_step` inner SAC mini-updates against the mixed
        real+synth buffer."""
        var any = False
        for _ in range(self.sac_updates_per_step):
            self.state.step_idx = step_idx
            self.state.did_step = True
            comptime if Self.train_target == "cpu":
                self.state.alpha = fexp(self.alpha_opt.value)
            else:
                self.state.ctx = self.ctx

            self.sample_blk.step[Self.train_target](self.state)
            if not self.state.did_step:
                continue

            self.target_y_blk.step[Self.train_target, POLICY](
                self.state,
                self.actor,
                self.pair1.target_net,
                self.pair2.target_net,
            )

            self.twin_critic_blk.step[
                Self.train_target,
                POLICY,
                ACCUMULATE=Self.train_target == "gpu",
            ](
                self.state,
                self.pair1.online,
                self.critic1_opt,
                self.pair2.online,
                self.critic2_opt,
            )

            var out = self.actor_loss_blk.forward_backward[
                Self.train_target, POLICY
            ](
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

            comptime if Self.train_target == "cpu":
                self.alpha_blk.step["cpu"](self.state, self.alpha_opt)
            else:
                self.alpha_opt.step_device(
                    self.ctx.value(),
                    self.actor_loss_blk.lp_mean_dev(),
                    self.alpha_blk.target_entropy,
                )

            self.polyak_blk.step[Self.train_target](
                self.state, self.pair1, self.pair2,
            )

            # Per-batch diagnostics.
            comptime B = Self.BATCH
            comptime A = Self.ACT_DIM
            comptime if Self.train_target == "cpu":
                var inv_b: Scalar[DT] = Scalar[DT](1.0) / Scalar[DT](B)
                var sum_q: Scalar[DT] = 0.0
                var sum_r: Scalar[DT] = 0.0
                var sum_y: Scalar[DT] = 0.0
                var sum_d: Scalar[DT] = 0.0
                for i in range(B):
                    sum_q += self.twin_critic_blk.inner.c1._mb_q.data[i]
                    sum_r += self.state.mb_r.data[i]
                    sum_y += self.state.mb_y.data[i]
                    sum_d += self.state.mb_d.data[i]
                var sum_a: Scalar[DT] = 0.0
                for i in range(B * A):
                    var av = self.state.mb_a.data[i]
                    sum_a += av if av >= 0 else -av
                self._q_accum += sum_q * inv_b
                self._reward_accum += sum_r * inv_b
                self._td_accum += sum_y * inv_b
                self._done_accum += sum_d * inv_b
                self._action_abs_accum += sum_a / Scalar[DT](B * A)
            else:
                comptime lb = Layout.row_major(B)
                comptime lba = Layout.row_major(B * A)
                self._q_mean_dev.accumulate_gpu_lt[B](
                    self.twin_critic_blk.inner.c1._mb_q.lt["gpu", lb]()
                )
                self._reward_mean_dev.accumulate_gpu_lt[B](
                    self.state.mb_r.lt["gpu", lb]()
                )
                self._td_mean_dev.accumulate_gpu_lt[B](
                    self.state.mb_y.lt["gpu", lb]()
                )
                self._done_mean_dev.accumulate_gpu_lt[B](
                    self.state.mb_d.lt["gpu", lb]()
                )
                self._action_abs_mean_dev.accumulate_gpu_abs_lt[B * A](
                    self.state.mb_a.lt["gpu", lba]()
                )

            self._actor_L_accum += self.state.actor_loss
            self._critic_L_accum += self.state.critic_loss
            self._update_count += 1
            self._total_train_steps += 1
            any = True
        return any

    # ─── CUDA-graph capture surface (OffPolicyAgentGpu) ───────────────────

    def train_device_kernels(mut self) raises:
        """ONE SAC sub-update as a pure device-kernel sequence (no host work) —
        the body captured into the CUDA graph and replayed. Mirrors SAC's
        `train_device_kernels`. GPU + NoAMP only. All per-iteration state that
        changes between replays is device-resident: sample RNG offset
        (advances on-device), alpha (step_device), critic/actor loss +
        diagnostics (device accumulators). Host counters live in
        `note_train_update`; the loss accums (`_actor_L_accum`/`_critic_L_accum`)
        are GPU-unused sentinels (flush reads device accumulators)."""
        comptime if Self.train_target == "gpu":
            comptime B = Self.BATCH
            comptime A = Self.ACT_DIM
            # Pin past warmup so the sample gate passes; bind device ctx.
            self.state.step_idx = self.learning_starts
            self.state.did_step = True
            self.state.ctx = self.ctx

            self.sample_blk.step["gpu"](self.state)
            self.target_y_blk.step["gpu", NoAMP](
                self.state,
                self.actor,
                self.pair1.target_net,
                self.pair2.target_net,
            )
            self.twin_critic_blk.step["gpu", NoAMP, ACCUMULATE=True](
                self.state,
                self.pair1.online,
                self.critic1_opt,
                self.pair2.online,
                self.critic2_opt,
            )
            var out = self.actor_loss_blk.forward_backward["gpu", NoAMP](
                self.actor,
                self.actor_opt,
                self.pair1.online,
                self.pair2.online,
                self.state.mb_s,
                self.state.mb_a,
                self.state.alpha,
                self.ctx,
            )
            _ = out
            self.alpha_opt.step_device(
                self.ctx.value(),
                self.actor_loss_blk.lp_mean_dev(),
                self.alpha_blk.target_entropy,
            )
            self.polyak_blk.step["gpu"](self.state, self.pair1, self.pair2)

            comptime lb = Layout.row_major(B)
            comptime lba = Layout.row_major(B * A)
            self._q_mean_dev.accumulate_gpu_lt[B](
                self.twin_critic_blk.inner.c1._mb_q.lt["gpu", lb]()
            )
            self._reward_mean_dev.accumulate_gpu_lt[B](
                self.state.mb_r.lt["gpu", lb]()
            )
            self._td_mean_dev.accumulate_gpu_lt[B](
                self.state.mb_y.lt["gpu", lb]()
            )
            self._done_mean_dev.accumulate_gpu_lt[B](
                self.state.mb_d.lt["gpu", lb]()
            )
            self._action_abs_mean_dev.accumulate_gpu_abs_lt[B * A](
                self.state.mb_a.lt["gpu", lba]()
            )
        else:
            raise Error("train_device_kernels: GPU-only (CUDA-graph capture)")

    def note_train_update(mut self):
        """Advance one logical update's host bookkeeping (the device work is the
        replayed graph; loss/alpha/diagnostics drain at flush)."""
        self._update_count += 1
        self._total_train_steps += 1

    def _run_sac_updates_graph(mut self) raises -> Bool:
        """Capture-once / replay the `sac_updates_per_step` SAC loop as one CUDA
        graph. The graph is moved into a disjoint local for the capture call so
        the closure can borrow `self` mutably without overlapping a mut borrow
        of the `_train_graph` field (the exclusivity rule the driver sidesteps
        via separate locals)."""
        var ctx = self.ctx.value()
        var g = self._train_graph^
        self._train_graph = None

        def _captured() capturing raises -> None:
            for _ in range(self.sac_updates_per_step):
                self.train_device_kernels()

        maybe_capture_replay[_captured](g, ctx)
        self._train_graph = g^

        for _ in range(self.sac_updates_per_step):
            self.note_train_update()
        return True

    def _dyn_train_step_graph(mut self, m: Int) raises:
        """Capture-once / replay member `m`'s `train_member_step` (the
        batch-build ran eagerly just before, into _dyn_in/_dyn_tgt). Each member
        keeps its own graph in `_dyn_graphs[m]`. The slot is moved into a
        disjoint local for the capture call (Optional.take leaves it None) so
        the closure can borrow self without overlapping the slot's mut borrow;
        on non-NVIDIA `maybe_capture_replay` runs the step eagerly and leaves
        the local None (slot stays None → eager every call)."""
        var ctx = self.ctx.value()
        var g = Optional[CUDAGraph](None)
        if self._dyn_graphs[m]:
            g = Optional[CUDAGraph](self._dyn_graphs[m].take())

        def _captured() capturing raises -> None:
            _ = self.ensemble.train_member_step["gpu", ACCUMULATE=True](
                m, self._dyn_in, self._dyn_tgt
            )

        maybe_capture_replay[_captured](g, ctx)
        self._dyn_graphs[m] = g^

    # ─── Logging surface ─────────────────────────────────────────────────

    def flush_train_log(
        mut self,
    ) -> Tuple[Scalar[DT], Scalar[DT], Scalar[DT], Int]:
        var n = self._update_count if self._update_count > 0 else 1
        var inv = Scalar[DT](1.0) / Scalar[DT](n)
        var alpha_now = fexp(self.alpha_opt.value)
        var out = (
            self._actor_L_accum * inv,
            self._critic_L_accum * inv,
            alpha_now,
            self._update_count,
        )
        self._actor_L_accum = Scalar[DT](0.0)
        self._critic_L_accum = Scalar[DT](0.0)
        self._update_count = 0
        self._q_accum = Scalar[DT](0.0)
        self._reward_accum = Scalar[DT](0.0)
        self._td_accum = Scalar[DT](0.0)
        self._done_accum = Scalar[DT](0.0)
        self._action_abs_accum = Scalar[DT](0.0)
        self._dyn_loss_accum = Scalar[DT](0.0)
        self._dyn_step_count = 0
        return out

    def total_train_steps(self) -> Int:
        return self._total_train_steps

    def learning_starts_count(self) -> Int:
        return self.learning_starts

    def flush_metrics[
        L: Logger = NoOpLogger
    ](
        mut self,
        logger: Optional[Pointer[L, MutAnyOrigin]] = None,
        step: Int = 0,
    ) raises -> MBPOMetrics:
        var n = self._update_count if self._update_count > 0 else 1
        var inv = Scalar[DT](1.0) / Scalar[DT](n)
        if self._dyn_step_count > 0:
            comptime if Self.train_target == "gpu":
                # Device-accumulated mean (one D2H at this diag cadence, not
                # per gradient step); reset for the next window.
                self._dyn_loss_last = self.ensemble.read_dyn_loss_accum["gpu"]()
                self.ensemble.reset_dyn_loss_accum["gpu"]()
            else:
                self._dyn_loss_last = self._dyn_loss_accum / Scalar[DT](
                    self._dyn_step_count
                )
        var actor_mean: Scalar[DT]
        var critic_mean: Scalar[DT]
        var alpha_val: Scalar[DT]
        var q_mean: Scalar[DT]
        var reward_mean: Scalar[DT]
        var td_mean: Scalar[DT]
        var done_mean: Scalar[DT]
        var act_abs_mean: Scalar[DT]
        comptime if Self.train_target == "gpu":
            actor_mean = self.actor_loss_blk.read_loss_accum(self.ctx.value())
            var cl1 = self.twin_critic_blk.inner.c1.mse_loss.read_accum["gpu"](
                self.ctx
            )
            var cl2 = self.twin_critic_blk.inner.c2.mse_loss.read_accum["gpu"](
                self.ctx
            )
            critic_mean = cl1 + cl2
            alpha_val = self.alpha_opt.read_alpha()
            q_mean = self._q_mean_dev.read["gpu"]()
            reward_mean = self._reward_mean_dev.read["gpu"]()
            td_mean = self._td_mean_dev.read["gpu"]()
            done_mean = self._done_mean_dev.read["gpu"]()
            act_abs_mean = self._action_abs_mean_dev.read["gpu"]()
        else:
            actor_mean = self._actor_L_accum * inv
            critic_mean = self._critic_L_accum * inv
            alpha_val = fexp(self.alpha_opt.value)
            q_mean = self._q_accum * inv
            reward_mean = self._reward_accum * inv
            td_mean = self._td_accum * inv
            done_mean = self._done_accum * inv
            act_abs_mean = self._action_abs_accum * inv
        var bundle = MBPOMetrics(
            policy_loss=LogScalar[DT](actor_mean),
            critic_loss=LogScalar[DT](critic_mean),
            alpha=LogScalar[DT](alpha_val),
            mean_q=LogScalar[DT](q_mean),
            reward_mean=LogScalar[DT](reward_mean),
            mean_target=LogScalar[DT](td_mean),
            mean_done=LogScalar[DT](done_mean),
            action_abs_mean=LogScalar[DT](act_abs_mean),
            dyn_loss=LogScalar[DT](self._dyn_loss_last),
            dyn_holdout_mse_mean=LogScalar[DT](self._dyn_holdout_loss),
            dyn_holdout_mse_min=LogScalar[DT](self._dyn_holdout_min),
            dyn_holdout_mse_max=LogScalar[DT](self._dyn_holdout_max),
            dyn_holdout_mse_spread=LogScalar[DT](
                self._dyn_holdout_max - self._dyn_holdout_min
            ),
            dyn_input_std_mean=LogScalar[DT](self._dyn_input_std_mean),
            train_steps=LogScalar[DT](Scalar[DT](self._total_train_steps)),
            n_updates=LogScalar[DT](Scalar[DT](self._update_count)),
        )
        self._actor_L_accum = Scalar[DT](0.0)
        self._critic_L_accum = Scalar[DT](0.0)
        self._update_count = 0
        self._q_accum = Scalar[DT](0.0)
        self._reward_accum = Scalar[DT](0.0)
        self._td_accum = Scalar[DT](0.0)
        self._done_accum = Scalar[DT](0.0)
        self._action_abs_accum = Scalar[DT](0.0)
        self._dyn_loss_accum = Scalar[DT](0.0)
        self._dyn_step_count = 0
        comptime if Self.train_target == "gpu":
            self.twin_critic_blk.inner.c1.mse_loss.reset_accum["gpu"]()
            self.twin_critic_blk.inner.c2.mse_loss.reset_accum["gpu"]()
            self.actor_loss_blk.reset_loss_accum()
            self._q_mean_dev.reset["gpu"]()
            self._reward_mean_dev.reset["gpu"]()
            self._td_mean_dev.reset["gpu"]()
            self._done_mean_dev.reset["gpu"]()
            self._action_abs_mean_dev.reset["gpu"]()
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
        _ = step
        _ = self.flush_metrics[L](logger, self._total_train_steps)

    def save_state(mut self, path: String) raises:
        """ONE v3 `storage-ckpt` file: the SAC nets (actor + 2 online critics)
        and every dynamics member, then as `K` sections α's whole optimizer
        (`alpha.*`), the elite indices (`elites.n`, `elites.<i>`) and the
        train-step counter. Atomic and chunked. Optimizer moments and the
        dynamics input scaler are NOT persisted: the scaler is refit from the
        replay buffer on every dynamics fit, and the buffer is not saved."""
        var w = BinaryCheckpointWriter(save_moments=False)
        write_model[Self.train_target](w, self.actor, self.ctx, "actor")
        write_model[Self.train_target](w, self.pair1.online, self.ctx, "critic1")
        write_model[Self.train_target](w, self.pair2.online, self.ctx, "critic2")
        for i in range(Self.N_ENSEMBLE):
            write_model[Self.train_target](
                w, self.ensemble.members[i], self.ctx, "dyn_member" + String(i)
            )
        var sc = CheckpointScalars()
        self.alpha_opt.put_state(sc, "alpha")
        sc.set_int("elites.n", len(self.ensemble.elite_indices))
        for e in range(len(self.ensemble.elite_indices)):
            sc.set_int("elites." + String(e), self.ensemble.elite_indices[e])
        sc.set_int("total_train_steps", self._total_train_steps)
        w.write_scalars(sc)
        _write_file_bytes(path, w.content)

    def load_state(mut self, path: String) raises:
        """Restore every net, α, the elites and the counter (v3, or the legacy
        v2 text this trainer wrote before, which carries only the nets), then
        hard-copy online → target."""
        var bytes = _read_file_bytes(path)
        if _is_v3_header(bytes):
            var rb = BinaryCheckpointReader(bytes^)
            read_model[Self.train_target](rb, self.actor, self.ctx, "actor")
            read_model[Self.train_target](rb, self.pair1.online, self.ctx, "critic1")
            read_model[Self.train_target](rb, self.pair2.online, self.ctx, "critic2")
            for i in range(Self.N_ENSEMBLE):
                read_model[Self.train_target](
                    rb, self.ensemble.members[i], self.ctx,
                    "dyn_member" + String(i),
                )
            var sc = rb.read_scalars()
            rb.finish()
            self.alpha_opt.take_state(sc, "alpha")
            if sc.has("elites.n"):
                self.ensemble.elite_indices.clear()
                for e in range(sc.get_int("elites.n", 0)):
                    self.ensemble.elite_indices.append(
                        sc.get_int("elites." + String(e), e)
                    )
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
            for i in range(Self.N_ENSEMBLE):
                walk_params[Self.train_target](self.ensemble.members[i], r, self.ctx, "dyn_member" + String(i)
                )
            r.mode = 1
            var _sref2 = ParamVisitorRef.of[type_of(r), Self.train_target](r)
            self.actor.for_each_state[Self.train_target](_sref2, self.ctx, "actor")
            self.pair1.online.for_each_state[Self.train_target](
                r, self.ctx, "critic1"
            )
            self.pair2.online.for_each_state[Self.train_target](
                r, self.ctx, "critic2"
            )
            for i in range(Self.N_ENSEMBLE):
                self.ensemble.members[i].for_each_state[Self.train_target](
                    r, self.ctx, "dyn_member" + String(i)
                )
        self.pair1.target_net.polyak_from[Self.train_target](
            self.pair1.online, Scalar[DT](1.0), self.ctx
        )
        self.pair2.target_net.polyak_from[Self.train_target](
            self.pair2.online, Scalar[DT](1.0), self.ctx
        )

    def flush_timer_log(mut self) -> String:
        return String("")

    # ─── OffPolicyAgentGpu surface ────────────────────────────────────

    def select_action_batched[
        N_ENVS: Int,
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
        comptime OBS = Self.OBS_DIM
        comptime ACT = Self.ACT_DIM

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
            self._ob1,
            self._ao1,
            self._alp1,
            obs,
            action,
            self.action_scale,
            self.ctx,
        )
        # silence unused warnings on the driver-owned scratch views.
        _ = ao_scratch
        _ = alp_scratch

    def _replay_add(
        mut self,
        ref obs: List[Scalar[DT]],
        ref action: List[Scalar[DT]],
        reward: Scalar[DT],
        ref next_obs: List[Scalar[DT]],
        done: Scalar[DT],
    ) raises:
        self.sample_blk.real_add[Self.train_target](
            obs, action, reward, next_obs, done, ctx=self.ctx,
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
        raise Error(
            "MBPOTrainer.record_batch_gpu: GPU-env batched record not"
            " supported (MBPO uses the cpu-env single-env driver path)"
        )

    def record_batch_gpu_nstep[
        N_ENVS: Int, NS: Int
    ](
        mut self,
        ctx: DeviceContext,
        mut nstep_buf: GPUNStepBuffer[
            NS,
            Self.AGENT_OBS_DIM,
            Self.AGENT_ACT_DIM,
            N_ENVS,
        ],
        prev_obs_dev: DeviceBuffer[DT],
        action_dev: DeviceBuffer[DT],
        reward_dev: DeviceBuffer[DT],
        obs_dev: DeviceBuffer[DT],
        done_dev: DeviceBuffer[DT],
    ) raises:
        raise Error("MBPOTrainer.record_batch_gpu_nstep: not supported")

    # ─── Dynamics input scaler (per-DYN_IN-dim z-score) ───────────────

    def _set_scaler_identity[target: StaticString](mut self) raises:
        """Reset the input scaler to identity (mean=0, std=1)."""
        for c in range(Self.DYN_IN):
            self._in_mean.data[c] = Scalar[DT](0.0)
            self._in_std.data[c] = Scalar[DT](1.0)
        comptime if target == "gpu":
            self._in_mean.dev.value().enqueue_copy_from(
                Span(self._in_mean.data)
            )
            self._in_std.dev.value().enqueue_copy_from(Span(self._in_std.data))

    @staticmethod
    def _compute_scaler_host(
        ref obs: List[Scalar[DT]],
        ref act: List[Scalar[DT]],
        mut mean_out: List[Scalar[DT]],
        mut std_out: List[Scalar[DT]],
        n_data: Int,
    ) -> Scalar[DT]:
        """Fit per-DYN_IN-dim mean/std from `n_data` real transitions. Returns
        the diagnostic mean input-scaler std."""
        for c in range(Self.DYN_IN):
            mean_out[c] = Scalar[DT](0.0)
            std_out[c] = Scalar[DT](0.0)
        var inv_n = Scalar[DT](1.0) / Scalar[DT](n_data)
        for i in range(n_data):
            for d in range(Self.OBS_DIM):
                mean_out[d] += obs[i * Self.OBS_DIM + d]
            for j in range(Self.ACT_DIM):
                mean_out[Self.OBS_DIM + j] += act[i * Self.ACT_DIM + j]
        for c in range(Self.DYN_IN):
            mean_out[c] *= inv_n
        for i in range(n_data):
            for d in range(Self.OBS_DIM):
                var diff = obs[i * Self.OBS_DIM + d] - mean_out[d]
                std_out[d] += diff * diff
            for j in range(Self.ACT_DIM):
                var diff = (
                    act[i * Self.ACT_DIM + j] - mean_out[Self.OBS_DIM + j]
                )
                std_out[Self.OBS_DIM + j] += diff * diff
        var std_sum = Scalar[DT](0.0)
        for c in range(Self.DYN_IN):
            var v = fsqrt(std_out[c] * inv_n)
            if v < Scalar[DT](1e-12):
                v = Scalar[DT](1.0)
            std_out[c] = v
            std_sum += v
        return std_sum / Scalar[DT](Self.DYN_IN)

    def _record_holdout_stats(mut self, ref holdout: List[Scalar[DT]]):
        if len(holdout) == 0:
            return
        var s = holdout[0]
        var mn = holdout[0]
        var mx = holdout[0]
        for i in range(1, len(holdout)):
            var v = holdout[i]
            s += v
            if v < mn:
                mn = v
            if v > mx:
                mx = v
        self._dyn_holdout_loss = s / Scalar[DT](len(holdout))
        self._dyn_holdout_min = mn
        self._dyn_holdout_max = mx

    def _fit_input_scaler_cpu(mut self):
        var n_data = self.sample_blk.real_count["cpu"]()
        if n_data < 2:
            return
        self._dyn_input_std_mean = Self._compute_scaler_host(
            self.sample_blk.real_cpu.value().obs,
            self.sample_blk.real_cpu.value().act,
            self._in_mean.data,
            self._in_std.data,
            n_data,
        )

    def _fit_input_scaler_gpu(mut self) raises:
        var n_data = self.sample_blk.real_count["gpu"]()
        if n_data < 2:
            return
        var ctx = self.ctx.value()
        # On-device per-column mean/std fit over the first `n_data` real rows —
        # replaces the old full-buffer D2H (now 20 MB at REPLAY_CAPACITY=300k) +
        # host loop + H2D + two `ctx.synchronize()`. Writes _in_mean/_in_std on
        # device; the normalize kernels read those on the same stream (no sync).
        comptime fit_kernel = _mbpo_fit_scaler_kernel[
            Self.OBS_DIM, Self.ACT_DIM, Self.DYN_IN, Self.REPLAY_CAPACITY
        ]
        comptime n_blocks = (Self.DYN_IN + TPB - 1) // TPB
        ctx.enqueue_function[fit_kernel](
            LayoutTensor[
                DT,
                Layout.row_major(Self.REPLAY_CAPACITY, Self.OBS_DIM),
                MutAnyOrigin,
            ](self.sample_blk.real_gpu.value().obs),
            LayoutTensor[
                DT,
                Layout.row_major(Self.REPLAY_CAPACITY, Self.ACT_DIM),
                MutAnyOrigin,
            ](self.sample_blk.real_gpu.value().act),
            self._in_mean.lt["gpu", Layout.row_major(Self.DYN_IN)](),
            self._in_std.lt["gpu", Layout.row_major(Self.DYN_IN)](),
            Int32(n_data),
            grid_dim=n_blocks,
            block_dim=TPB,
        )
        # Diagnostic only (mean input-scaler std): a tiny DYN_IN-float D2H once
        # per model-train round — NOT per gradient step.
        self._in_std.download(ctx)
        var s = Scalar[DT](0.0)
        for c in range(Self.DYN_IN):
            s += self._in_std.data[c]
        self._dyn_input_std_mean = s / Scalar[DT](Self.DYN_IN)

    def _normalize_dyn_in_cpu(mut self):
        for k in range(Self.BATCH):
            var base = k * Self.DYN_IN
            for c in range(Self.DYN_IN):
                self._dyn_in.data[base + c] = (
                    self._dyn_in.data[base + c] - self._in_mean.data[c]
                ) / self._in_std.data[c]

    def _normalize_dyn_in_gpu(mut self) raises:
        var ctx = self.ctx.value()
        comptime total = Self.BATCH * Self.DYN_IN
        comptime n_blocks = (total + TPB - 1) // TPB
        comptime norm_kernel = _normalize_input_kernel[Self.BATCH, Self.DYN_IN]
        ctx.enqueue_function[norm_kernel](
            self._dyn_in.lt["gpu", Layout.row_major(Self.BATCH, Self.DYN_IN)](),
            self._in_mean.lt["gpu", Layout.row_major(Self.DYN_IN)](),
            self._in_std.lt["gpu", Layout.row_major(Self.DYN_IN)](),
            grid_dim=n_blocks,
            block_dim=TPB,
        )

    # ─── Dynamics training + synthetic rollouts (CPU) ─────────────────

    def _fill_dyn_batch_cpu(mut self, k: Int, idx: Int):
        """Fill row `k` of the host dyn_in/dyn_tgt scratch from real-buffer
        transition `idx`."""
        var rb_obs = self.sample_blk.real_cpu.value().obs.copy()
        var rb_act = self.sample_blk.real_cpu.value().act.copy()
        var rb_rew = self.sample_blk.real_cpu.value().rew.copy()
        var rb_nxt = self.sample_blk.real_cpu.value().nxt.copy()
        for d in range(Self.OBS_DIM):
            self._dyn_in.data[k * Self.DYN_IN + d] = rb_obs[
                idx * Self.OBS_DIM + d
            ]
        for j in range(Self.ACT_DIM):
            self._dyn_in.data[k * Self.DYN_IN + Self.OBS_DIM + j] = rb_act[
                idx * Self.ACT_DIM + j
            ]
        self._dyn_tgt.data[k * Self.DYN_PRED + 0] = rb_rew[idx]
        for d in range(Self.OBS_DIM):
            self._dyn_tgt.data[k * Self.DYN_PRED + 1 + d] = (
                rb_nxt[idx * Self.OBS_DIM + d] - rb_obs[idx * Self.OBS_DIM + d]
            )

    def _eval_member_holdout_cpu(
        mut self, m: Int, n_train: Int, n_holdout: Int,
    ) raises -> Scalar[DT]:
        var n_chunks = n_holdout // Self.BATCH
        if n_chunks < 1:
            n_chunks = 1
        var total = Scalar[DT](0.0)
        for c in range(n_chunks):
            for k in range(Self.BATCH):
                var idx = n_train + ((c * Self.BATCH + k) % n_holdout)
                self._fill_dyn_batch_cpu(k, idx)
            self._normalize_dyn_in_cpu()
            total += self.ensemble.eval_member_mse["cpu"](
                m, self._dyn_in, self._dyn_tgt
            )
        return total / Scalar[DT](n_chunks)

    def _train_dynamics_ensemble(mut self) raises:
        var n_data = self.sample_blk.real_count["cpu"]()
        if n_data < self.dyn_batch_size:
            return
        var n_holdout = Int(Scalar[DT](n_data) * self.dyn_holdout_ratio)
        if n_holdout < 1:
            n_holdout = 1
        var n_train = n_data - n_holdout
        if n_train < self.dyn_batch_size:
            n_train = n_data
            n_holdout = n_data
        var bs = self.dyn_batch_size
        var steps_per_epoch = n_train // bs
        if steps_per_epoch < 1:
            steps_per_epoch = 1
        var n_checks = self.dyn_max_epochs // self.dyn_holdout_check_every
        if n_checks < 1:
            n_checks = 1

        for m in range(Self.N_ENSEMBLE):
            var best = Scalar[DT](1e30)
            var since = 0
            var stop = False
            for _check in range(n_checks):
                if stop:
                    break
                for _ep in range(self.dyn_holdout_check_every):
                    for _ in range(steps_per_epoch):
                        for k in range(Self.BATCH):
                            var idx = Int(random_float64() * Float64(n_train))
                            if idx >= n_train:
                                idx = n_train - 1
                            self._fill_dyn_batch_cpu(k, idx)
                        self._normalize_dyn_in_cpu()
                        var dyn_loss = self.ensemble.train_member_step["cpu"](
                            m, self._dyn_in, self._dyn_tgt
                        )
                        self._dyn_loss_accum += dyn_loss
                        self._dyn_step_count += 1
                var hl = self._eval_member_holdout_cpu(m, n_train, n_holdout)
                # Reference early-stop: a member "improves" only if its holdout
                # MSE drops by > 1% RELATIVE (bnn.py `_save_best`). An absolute
                # threshold doesn't transfer across the data-dependent MSE
                # scale. `(best - hl) > 0.01·best` ⟺ relative improvement > 1%
                # (best > 0; avoids div-by-zero).
                if (best - hl) > Scalar[DT](0.01) * best:
                    best = hl
                    since = 0
                else:
                    since += 1
                    if since >= self.dyn_patience:
                        stop = True

        var holdout = List[Scalar[DT]]()
        for m in range(Self.N_ENSEMBLE):
            holdout.append(self._eval_member_holdout_cpu(m, n_train, n_holdout))
        self._record_holdout_stats(holdout)
        self.ensemble.update_elites(holdout)

    def _generate_synthetic_rollouts(mut self) raises:
        var real_buf_size = self.sample_blk.real_count["cpu"]()
        if real_buf_size < 1:
            return

        var rollouts_done = 0
        while rollouts_done < self.num_rollouts_per_step:
            var this_batch = Self.BATCH
            var remaining = self.num_rollouts_per_step - rollouts_done
            if remaining < this_batch:
                this_batch = remaining

            var rb_obs = self.sample_blk.real_cpu.value().obs.copy()
            for k in range(this_batch):
                var idx = Int(random_float64() * Float64(real_buf_size))
                if idx >= real_buf_size:
                    idx = real_buf_size - 1
                for d in range(Self.OBS_DIM):
                    self._ro_obs.data[k * Self.OBS_DIM + d] = rb_obs[
                        idx * Self.OBS_DIM + d
                    ]

            for _ in range(self.rollout_length):
                for k in range(this_batch):
                    var obs_list = List[Scalar[DT]](capacity=Self.OBS_DIM)
                    for d in range(Self.OBS_DIM):
                        obs_list.append(self._ro_obs.data[k * Self.OBS_DIM + d])
                    var act_list = List[Scalar[DT]](capacity=Self.ACT_DIM)
                    for _ in range(Self.ACT_DIM):
                        act_list.append(Scalar[DT](0.0))
                    self._select_action_into_cpu(obs_list, act_list)
                    for j in range(Self.ACT_DIM):
                        self._ro_act.data[k * Self.ACT_DIM + j] = act_list[j]

                for k in range(this_batch):
                    for d in range(Self.OBS_DIM):
                        self._dyn_in.data[k * Self.DYN_IN + d] = (
                            self._ro_obs.data[k * Self.OBS_DIM + d]
                        )
                    for j in range(Self.ACT_DIM):
                        self._dyn_in.data[
                            k * Self.DYN_IN + Self.OBS_DIM + j
                        ] = self._ro_act.data[k * Self.ACT_DIM + j]
                self._normalize_dyn_in_cpu()

                var n_elites = len(self.ensemble.elite_indices)
                for e in range(n_elites):
                    # Forward each elite into its slice of the stacked buffers.
                    var off = e * Self.BATCH * Self.DYN_PRED
                    var mu_e = Tensor.alloc(Self.BATCH * Self.DYN_PRED)
                    var lv_e = Tensor.alloc(Self.BATCH * Self.DYN_PRED)
                    self.ensemble.predict_member["cpu"](
                        self.ensemble.elite_indices[e],
                        self._dyn_in, mu_e, lv_e,
                    )
                    for q in range(Self.BATCH * Self.DYN_PRED):
                        self._ro_mu_all.data[off + q] = mu_e.data[q]
                        self._ro_lv_all.data[off + q] = lv_e.data[q]

                var s_list = List[Scalar[DT]](capacity=Self.OBS_DIM)
                var a_list = List[Scalar[DT]](capacity=Self.ACT_DIM)
                var sp_list = List[Scalar[DT]](capacity=Self.OBS_DIM)
                for _ in range(Self.OBS_DIM):
                    s_list.append(Scalar[DT](0.0))
                    sp_list.append(Scalar[DT](0.0))
                for _ in range(Self.ACT_DIM):
                    a_list.append(Scalar[DT](0.0))
                for k in range(this_batch):
                    var e = Int(random_float64() * Float64(n_elites))
                    if e >= n_elites:
                        e = n_elites - 1
                    var base = (
                        e * Self.BATCH * Self.DYN_PRED + k * Self.DYN_PRED
                    )
                    var mu_r = self._ro_mu_all.data[base + 0]
                    var lv_r = self._ro_lv_all.data[base + 0]
                    var std_r = fsqrt(fexp(lv_r))
                    var noise_r = Scalar[DT](randn_float64())
                    var rew = mu_r + std_r * noise_r
                    if rew != rew:
                        rew = Scalar[DT](0.0)
                    elif rew < Scalar[DT](-100.0):
                        rew = Scalar[DT](-100.0)
                    elif rew > Scalar[DT](100.0):
                        rew = Scalar[DT](100.0)
                    for d in range(Self.OBS_DIM):
                        s_list[d] = self._ro_obs.data[k * Self.OBS_DIM + d]
                        var mu_d = self._ro_mu_all.data[base + 1 + d]
                        var lv_d = self._ro_lv_all.data[base + 1 + d]
                        var std_d = fsqrt(fexp(lv_d))
                        var noise = Scalar[DT](randn_float64())
                        var delta = mu_d + std_d * noise
                        var nxt = self._ro_obs.data[k * Self.OBS_DIM + d] + delta
                        sp_list[d] = nxt
                        self._ro_nxt.data[k * Self.OBS_DIM + d] = nxt
                    for j in range(Self.ACT_DIM):
                        a_list[j] = self._ro_act.data[k * Self.ACT_DIM + j]
                    self.sample_blk.synth_add(
                        s_list, a_list, rew, sp_list, Scalar[DT](0.0),
                    )

                for k in range(this_batch * Self.OBS_DIM):
                    self._ro_obs.data[k] = self._ro_nxt.data[k]

            rollouts_done += this_batch

    # ─── Dynamics training + synthetic rollouts (GPU) ─────────────────

    def _build_dyn_batch_gpu(mut self, lo: Int, hi: Int) raises:
        var ctx = self.ctx.value()
        comptime n_blocks = (Self.BATCH + TPB - 1) // TPB
        comptime tgt_kernel = _build_dyn_target_kernel[
            Self.BATCH, Self.OBS_DIM, Self.DYN_PRED
        ]
        self.sample_blk.real_sample_range[Self.BATCH](
            ctx, lo, hi,
            self._ro_obs.dev.value(),
            self._ro_act.dev.value(),
            self._ro_rew.dev.value(),
            self._ro_nxt.dev.value(),
            self._ro_done.dev.value(),
        )
        concat_sa_gpu[Self.OBS_DIM, Self.ACT_DIM, Self.BATCH](
            ctx,
            self._ro_obs.lt["gpu", Layout.row_major(Self.BATCH, Self.OBS_DIM)](),
            self._ro_act.lt["gpu", Layout.row_major(Self.BATCH, Self.ACT_DIM)](),
            self._dyn_in.lt["gpu", Layout.row_major(Self.BATCH, Self.DYN_IN)](),
        )
        self._normalize_dyn_in_gpu()
        ctx.enqueue_function[tgt_kernel](
            self._ro_rew.lt["gpu", Layout.row_major(Self.BATCH)](),
            self._ro_obs.lt["gpu", Layout.row_major(Self.BATCH, Self.OBS_DIM)](),
            self._ro_nxt.lt["gpu", Layout.row_major(Self.BATCH, Self.OBS_DIM)](),
            self._dyn_tgt.lt["gpu", Layout.row_major(Self.BATCH, Self.DYN_PRED)](),
            grid_dim=n_blocks,
            block_dim=TPB,
        )

    def _eval_member_holdout_gpu(
        mut self, m: Int, lo: Int, hi: Int,
    ) raises -> Scalar[DT]:
        var span = hi - lo
        var n_chunks = span // Self.BATCH
        if n_chunks < 1:
            n_chunks = 1
        if n_chunks > 4:
            n_chunks = 4
        var total = Scalar[DT](0.0)
        for _ in range(n_chunks):
            self._build_dyn_batch_gpu(lo, hi)
            total += self.ensemble.eval_member_mse["gpu"](
                m, self._dyn_in, self._dyn_tgt
            )
        return total / Scalar[DT](n_chunks)

    def _train_dynamics_ensemble_gpu(mut self) raises:
        var n_data = self.sample_blk.real_count["gpu"]()
        if n_data < self.dyn_batch_size:
            return
        var n_holdout = Int(Scalar[DT](n_data) * self.dyn_holdout_ratio)
        if n_holdout < 1:
            n_holdout = 1
        var n_train = n_data - n_holdout
        var hold_lo = n_train
        var hold_hi = n_data
        if n_train < self.dyn_batch_size:
            n_train = n_data
            hold_lo = 0
            hold_hi = n_data
        var bs = self.dyn_batch_size
        var steps_per_epoch = n_train // bs
        if steps_per_epoch < 1:
            steps_per_epoch = 1
        var n_checks = self.dyn_max_epochs // self.dyn_holdout_check_every
        if n_checks < 1:
            n_checks = 1

        for m in range(Self.N_ENSEMBLE):
            var best = Scalar[DT](1e30)
            var since = 0
            var stop = False
            for _check in range(n_checks):
                if stop:
                    break
                for _ep in range(self.dyn_holdout_check_every):
                    for _ in range(steps_per_epoch):
                        # Batch-build stays EAGER (fresh sample + correct
                        # [0,n_train) range each step); only the network train
                        # step is captured/replayed (it reads the persistent
                        # _dyn_in/_dyn_tgt buffers). ACCUMULATE=True folds the
                        # loss into the device accumulator (read once at flush).
                        self._build_dyn_batch_gpu(0, n_train)
                        comptime if Self.USE_TRAIN_CUDA_GRAPH:
                            self._dyn_train_step_graph(m)
                        else:
                            _ = self.ensemble.train_member_step[
                                "gpu", ACCUMULATE=True
                            ](m, self._dyn_in, self._dyn_tgt)
                        self._dyn_step_count += 1
                var hl = self._eval_member_holdout_gpu(m, hold_lo, hold_hi)
                # Reference early-stop: relative improvement > 1% on holdout
                # MSE (bnn.py `_save_best`); see the CPU path for the rationale.
                if (best - hl) > Scalar[DT](0.01) * best:
                    best = hl
                    since = 0
                else:
                    since += 1
                    if since >= self.dyn_patience:
                        stop = True

        var holdout = List[Scalar[DT]]()
        for m in range(Self.N_ENSEMBLE):
            holdout.append(self._eval_member_holdout_gpu(m, hold_lo, hold_hi))
        self._record_holdout_stats(holdout)
        self.ensemble.update_elites(holdout)

    def _generate_synthetic_rollouts_gpu(mut self) raises:
        if self.sample_blk.real_count["gpu"]() < 1:
            return
        var ctx = self.ctx.value()

        self._ro_done.dev.value().enqueue_fill(Scalar[DT](0.0))

        comptime n_lane_blocks = (Self.BATCH + TPB - 1) // TPB
        comptime clamp_kernel = offpolicy_clamp_action_kernel[
            Self.BATCH, Self.ACT_DIM
        ]
        comptime post_kernel = _rollout_posterior_kernel[
            Self.BATCH, Self.OBS_DIM, Self.DYN_PRED
        ]
        comptime n_noise = Self.BATCH * Self.DYN_PRED

        var chunks = (self.num_rollouts_per_step + Self.BATCH - 1) // Self.BATCH
        for _ in range(chunks):
            self.sample_blk.real_sample[Self.BATCH](
                ctx,
                self._ro_obs.dev.value(),
                self._ro_act.dev.value(),
                self._ro_rew.dev.value(),
                self._ro_nxt.dev.value(),
                self._ro_done.dev.value(),
            )
            self._ro_done.dev.value().enqueue_fill(Scalar[DT](0.0))

            for _ in range(self.rollout_length):
                # Policy action on imagined obs: actor → rsample → clamp.
                call_forward["gpu", Self.BATCH](
                    self.actor,
                    TensorRefs[Self.ACTOR.ARITY](self._ro_obs),
                    self._ro_ao,
                    self.ctx,
                )
                self.sel.forward["gpu", Self.BATCH](
                    TensorRefs[1](self._ro_ao), self._ro_alp, self.ctx
                )
                ctx.enqueue_function[clamp_kernel](
                    self._ro_alp.lt[
                        "gpu", Layout.row_major(Self.BATCH, Self.ACT_DIM + 1)
                    ](),
                    self._ro_act.lt[
                        "gpu", Layout.row_major(Self.BATCH, Self.ACT_DIM)
                    ](),
                    self.action_scale,
                    grid_dim=n_lane_blocks,
                    block_dim=TPB,
                )

                concat_sa_gpu[Self.OBS_DIM, Self.ACT_DIM, Self.BATCH](
                    ctx,
                    self._ro_obs.lt[
                        "gpu", Layout.row_major(Self.BATCH, Self.OBS_DIM)
                    ](),
                    self._ro_act.lt[
                        "gpu", Layout.row_major(Self.BATCH, Self.ACT_DIM)
                    ](),
                    self._dyn_in.lt[
                        "gpu", Layout.row_major(Self.BATCH, Self.DYN_IN)
                    ](),
                )
                self._normalize_dyn_in_gpu()

                var n_elites = len(self.ensemble.elite_indices)
                for e in range(n_elites):
                    # Forward each elite DIRECTLY into its slice of the stacked
                    # device buffers (predict_member writes via lt_at at the
                    # offset) — no fresh per-elite alloc + D2D copy per chunk.
                    var off = e * Self.BATCH * Self.DYN_PRED
                    self.ensemble.predict_member["gpu"](
                        self.ensemble.elite_indices[e],
                        self._dyn_in,
                        self._ro_mu_all,
                        self._ro_lv_all,
                        out_off=off,
                    )

                comptime assign_kernel = _mbpo_elite_assign_kernel[Self.BATCH]
                ctx.enqueue_function[assign_kernel](
                    self._ro_slot.lt["gpu", Layout.row_major(Self.BATCH)](),
                    Int32(n_elites),
                    self._roll_rng_seed + UInt64(0x5107),
                    self._elite_rng_offset,
                    grid_dim=n_lane_blocks,
                    block_dim=TPB,
                )
                self._elite_rng_offset += UInt64(Self.BATCH)

                comptime gather_kernel = _mbpo_elite_gather_kernel[
                    Self.BATCH, Self.NUM_ELITES, Self.DYN_PRED
                ]
                ctx.enqueue_function[gather_kernel](
                    self._ro_mu_all.lt[
                        "gpu",
                        Layout.row_major(
                            Self.NUM_ELITES * Self.BATCH, Self.DYN_PRED
                        ),
                    ](),
                    self._ro_lv_all.lt[
                        "gpu",
                        Layout.row_major(
                            Self.NUM_ELITES * Self.BATCH, Self.DYN_PRED
                        ),
                    ](),
                    self._ro_slot.lt["gpu", Layout.row_major(Self.BATCH)](),
                    self._ro_mu.lt[
                        "gpu", Layout.row_major(Self.BATCH, Self.DYN_PRED)
                    ](),
                    self._ro_lv.lt[
                        "gpu", Layout.row_major(Self.BATCH, Self.DYN_PRED)
                    ](),
                    grid_dim=n_lane_blocks,
                    block_dim=TPB,
                )

                comptime nb_noise = (n_noise + TPB - 1) // TPB
                ctx.enqueue_function[_box_muller_kernel[n_noise]](
                    self._ro_noise.lt["gpu", Layout.row_major(n_noise)](),
                    self._roll_rng_seed,
                    self._roll_rng_offset,
                    grid_dim=nb_noise,
                    block_dim=TPB,
                )
                self._roll_rng_offset += UInt64(((n_noise + 1) // 2) * 2)
                ctx.enqueue_function[post_kernel](
                    self._ro_obs.lt[
                        "gpu", Layout.row_major(Self.BATCH, Self.OBS_DIM)
                    ](),
                    self._ro_mu.lt[
                        "gpu", Layout.row_major(Self.BATCH, Self.DYN_PRED)
                    ](),
                    self._ro_lv.lt[
                        "gpu", Layout.row_major(Self.BATCH, Self.DYN_PRED)
                    ](),
                    self._ro_noise.lt[
                        "gpu", Layout.row_major(Self.BATCH, Self.DYN_PRED)
                    ](),
                    self._ro_nxt.lt[
                        "gpu", Layout.row_major(Self.BATCH, Self.OBS_DIM)
                    ](),
                    self._ro_rew.lt["gpu", Layout.row_major(Self.BATCH)](),
                    grid_dim=n_lane_blocks,
                    block_dim=TPB,
                )

                self.sample_blk.synth_add_batch[Self.BATCH](
                    ctx,
                    self._ro_obs.dev.value(),
                    self._ro_act.dev.value(),
                    self._ro_rew.dev.value(),
                    self._ro_nxt.dev.value(),
                    self._ro_done.dev.value(),
                )

                ctx.enqueue_copy(
                    self._ro_obs.dev.value(), self._ro_nxt.dev.value()
                )
