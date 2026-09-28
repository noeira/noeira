"""REDQOFETrainer — storage-framework REDQ-OFE trainer (CPU gate; GPU stretch).

REDQ + an Online Feature Extractor (OFENet). DenseNet branches compute φ(s) and
φ(s,a); the RL ensemble (actor/critic) operates on those features, and an
auxiliary next-state-prediction loss trains the feature nets. Mirrors the
storage `REDQTrainer` shape with the OFE blocks inserted.

Per `train_step` (paper-faithful cadence):

    train_step(step_idx):                  # outer = 1 env step
        sample (gates warmup)
        inner tick 1                        # _one_inner_tick
        for _ in 0..UTD-1: sample; inner tick
        ONE aux step on the LAST minibatch  # _run_aux_step

    _one_inner_tick:
        feature pre-pass (φ(s), φ(s'))      # OFEFeatureStep
        resample subset
        target-y  on φ(s')                  # EnsembleTargetYBlockOFE
        critic    on φ(s)                   # EnsembleCriticStepOFE
        polyak    (every inner tick)        # EnsemblePolyakStep (reused)
        if inner % POLICY_DELAY == 0:
            actor on φ(s)                    # EnsembleActorStepOFE
            alpha (host ScalarAdam)          # AlphaUpdateStep (reused)

The three OFE nets (state_branch / action_branch / predictor) are owned by the
trainer and threaded into all OFE blocks; they train ONLY via the aux step.

STORAGE migration (Stage 5): own scratch as `nn.storage.Tensor`s; `Adam.adopt`
on GPU; a one-file v3 checkpoint (α's optimizer and the counter as `K`
scalars); α is a HOST scalar on
both targets; CUDA-graph capture DEFERRED (host control flow). Dimensions
(OBS / ACT / BATCH) derive from `SAMPLE`; PHI_S_DIM from SB; PHI_SA_DIM from AB.
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
    BinaryCheckpointWriter, BinaryCheckpointReader, CheckpointReader,
    CheckpointScalars, write_model, read_model, _split_lines, _is_v3_header,
    _read_file_bytes, _write_file_bytes,
)

from noeira.nn.core.log_bundle import log_bundle
from noeira.nn.core.metric import LogScalar

from ..data.n_step_replay import GPUNStepBuffer
from ..training.episode_tracker import EpisodeTracker
from ..training.trainer_block import TrainerState
from ..training.driver_offpolicy import OffPolicyAgentGpu
from ..training.blocks import SampleBlock
from ..training.blocks.action_select import (
    select_squashed_batched,
    warmup_uniform_batched,
)
from ..sac.blocks.alpha_update_step import AlphaUpdateStep

from ..redq.ensemble import CriticEnsemble
from ..redq.blocks.ensemble_polyak_step import EnsemblePolyakStep

from .feature_step import OFEFeatureStep
from .aux_loss_step import OFEAuxLossStep
from .ensemble_target_y_block_ofe import EnsembleTargetYBlockOFE
from .ensemble_critic_step_ofe import EnsembleCriticStepOFE
from .ensemble_actor_step_ofe import EnsembleActorStepOFE
from .metrics import REDQOFEMetrics


# ──────────────────────────────────────────────────────────────────────
# Result struct for the fixed-batch overfit path / introspection.
# ──────────────────────────────────────────────────────────────────────


@fieldwise_init
struct REDQOFEStepResult(Movable & Deinitable):
    var critic_loss: Scalar[DT]
    var actor_loss: Scalar[DT]
    var log_prob_mean: Scalar[DT]
    var alpha: Scalar[DT]
    var aux_loss: Scalar[DT]
    var did_actor_step: Bool


# ──────────────────────────────────────────────────────────────────────
# GPU select_action_batched kernels (mirror REDQ's warmup + copy + clamp).
# ──────────────────────────────────────────────────────────────────────


struct REDQOFETrainer[
    train_target: StaticString,
    SAMPLE: SampleBlock,
    ACTOR: Module,   # IN=PHI_S_DIM,  OUT=2·ACT
    CRITIC: Module,  # IN=PHI_SA_DIM, OUT=1
    SB: Module,      # IN=OBS,            OUT=PHI_S_DIM
    AB: Module,      # IN=PHI_S_DIM+ACT,  OUT=PHI_SA_DIM
    PRED: Module,    # IN=PHI_SA_DIM,     OUT=OBS
    N: Int,
    N_MIN: Int,
    UTD: Int,
    POLICY_DELAY: Int,
    Q_MODE: Int,
](OffPolicyAgentGpu):
    """Storage-framework REDQ-OFE trainer."""

    comptime OBS_DIM: Int = Self.SAMPLE.OBS
    comptime ACT_DIM: Int = Self.SAMPLE.ACT
    comptime BATCH: Int = Self.SAMPLE.BATCH
    comptime PHI_S_DIM: Int = Self.SB.OUT_DIM
    comptime PHI_SA_DIM: Int = Self.AB.OUT_DIM

    comptime AGENT_OBS_DIM: Int = Self.OBS_DIM
    comptime AGENT_ACT_DIM: Int = Self.ACT_DIM
    comptime AGENT_TRAIN_TARGET: StaticString = Self.train_target

    # ── Owned networks ────────────────────────────────────────────────
    var actor: Self.ACTOR
    var state_branch: Self.SB
    var action_branch: Self.AB
    var predictor: Self.PRED
    var ensemble: CriticEnsemble[Self.CRITIC, Self.N]

    # ── Owned optimizers ──────────────────────────────────────────────
    var actor_opt: Adam
    var sb_opt: Adam
    var ab_opt: Adam
    var pred_opt: Adam
    var alpha_opt: ScalarAdam

    # ── Owned blocks ──────────────────────────────────────────────────
    var feat_blk: OFEFeatureStep[Self.SB, Self.OBS_DIM, Self.ACT_DIM, Self.BATCH]
    var target_y_blk: EnsembleTargetYBlockOFE[
        Self.ACTOR, Self.AB, Self.CRITIC, Self.N, Self.BATCH,
        Self.OBS_DIM, Self.PHI_S_DIM, Self.ACT_DIM, Self.N_MIN, Self.Q_MODE,
    ]
    var critic_blk: EnsembleCriticStepOFE[
        Self.AB, Self.CRITIC, Self.N, Self.OBS_DIM, Self.PHI_S_DIM,
        Self.ACT_DIM, Self.BATCH,
    ]
    var actor_blk: EnsembleActorStepOFE[
        Self.ACTOR, Self.AB, Self.CRITIC, Self.N, Self.BATCH,
        Self.PHI_S_DIM, Self.ACT_DIM,
    ]
    var aux_blk: OFEAuxLossStep[
        Self.SB, Self.AB, Self.PRED, Self.OBS_DIM, Self.ACT_DIM, Self.BATCH,
    ]
    var polyak_blk: EnsemblePolyakStep[
        Self.CRITIC, Self.N, Self.OBS_DIM, Self.ACT_DIM, Self.BATCH,
    ]
    var alpha_blk: AlphaUpdateStep[Self.OBS_DIM, Self.ACT_DIM, Self.BATCH]
    var sample_blk: Self.SAMPLE

    # select-action rsample (separate from the loss graphs' own rsamples).
    var sel: RSample[Self.ACT_DIM]

    var state: TrainerState[Self.OBS_DIM, Self.ACT_DIM, Self.BATCH]
    var tracker: EpisodeTracker
    var ctx: Optional[DeviceContext]

    # Owned action-selection scratch Tensors (lazily `.ensure`d per call).
    var _ob_scr: Tensor    # N_ENVS * OBS
    var _phi_scr: Tensor   # N_ENVS * PHI_S_DIM
    var _ao_scr: Tensor    # N_ENVS * 2*ACT
    var _alp_scr: Tensor   # N_ENVS * (ACT + 1)

    var action_scale: Scalar[DT]
    var learning_starts: Int

    var _warmup_rng_seed: UInt64
    var _warmup_rng_offset: UInt64

    var _inner_count: Int
    var _total_train_steps: Int

    # ── Per-flush-window accumulators (host) ──────────────────────────
    var _acc_critic_loss: Scalar[DT]
    var _acc_actor_loss: Scalar[DT]
    var _acc_alpha: Scalar[DT]
    var _acc_lp_mean: Scalar[DT]
    var _acc_aux_loss: Scalar[DT]
    var _acc_n_updates: Int
    var _acc_n_actor_updates: Int

    def __init__(out self):
        self.actor = Self.ACTOR()
        self.state_branch = Self.SB()
        self.action_branch = Self.AB()
        self.predictor = Self.PRED()
        self.ensemble = CriticEnsemble[Self.CRITIC, Self.N]()
        self.actor_opt = Adam(lr=Scalar[DT](3e-4))
        self.sb_opt = Adam(lr=Scalar[DT](3e-4))
        self.ab_opt = Adam(lr=Scalar[DT](3e-4))
        self.pred_opt = Adam(lr=Scalar[DT](3e-4))
        self.alpha_opt = ScalarAdam.new(flog(Scalar[DT](0.2)), Scalar[DT](3e-4))
        self.feat_blk = OFEFeatureStep[
            Self.SB, Self.OBS_DIM, Self.ACT_DIM, Self.BATCH
        ]()
        self.target_y_blk = EnsembleTargetYBlockOFE[
            Self.ACTOR, Self.AB, Self.CRITIC, Self.N, Self.BATCH,
            Self.OBS_DIM, Self.PHI_S_DIM, Self.ACT_DIM, Self.N_MIN, Self.Q_MODE,
        ]()
        self.critic_blk = EnsembleCriticStepOFE[
            Self.AB, Self.CRITIC, Self.N, Self.OBS_DIM, Self.PHI_S_DIM,
            Self.ACT_DIM, Self.BATCH,
        ]()
        self.actor_blk = EnsembleActorStepOFE[
            Self.ACTOR, Self.AB, Self.CRITIC, Self.N, Self.BATCH,
            Self.PHI_S_DIM, Self.ACT_DIM,
        ]()
        self.aux_blk = OFEAuxLossStep[
            Self.SB, Self.AB, Self.PRED, Self.OBS_DIM, Self.ACT_DIM, Self.BATCH,
        ]()
        self.polyak_blk = EnsemblePolyakStep[
            Self.CRITIC, Self.N, Self.OBS_DIM, Self.ACT_DIM, Self.BATCH,
        ]()
        self.alpha_blk = AlphaUpdateStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH
        ]()
        self.sample_blk = Self.SAMPLE()
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
        self._phi_scr = Tensor()
        self._ao_scr = Tensor()
        self._alp_scr = Tensor()
        self.action_scale = Scalar[DT](1.0)
        self.learning_starts = 1_000
        self._warmup_rng_seed = UInt64(0xC0FFEE_C0DE)
        self._warmup_rng_offset = UInt64(0)
        self._inner_count = 0
        self._total_train_steps = 0
        self._acc_critic_loss = Scalar[DT](0.0)
        self._acc_actor_loss = Scalar[DT](0.0)
        self._acc_alpha = Scalar[DT](0.0)
        self._acc_lp_mean = Scalar[DT](0.0)
        self._acc_aux_loss = Scalar[DT](0.0)
        self._acc_n_updates = 0
        self._acc_n_actor_updates = 0

    @staticmethod
    def make(
        ctx: Optional[DeviceContext] = None,
        actor_lr: Scalar[DT] = Scalar[DT](3e-4),
        critic_lr: Scalar[DT] = Scalar[DT](3e-4),
        ofe_lr: Scalar[DT] = Scalar[DT](3e-4),
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
        ), "REDQOFETrainer: train_target must be 'cpu' or 'gpu'"
        comptime if Self.train_target == "gpu":
            if not ctx:
                raise Error(
                    "REDQOFETrainer.make[train_target='gpu']: ctx required"
                )
        comptime assert Self.N >= 2, "REDQ-OFE: N must be ≥ 2"
        comptime assert Self.N_MIN >= 1, "REDQ-OFE: N_MIN must be ≥ 1"
        comptime assert Self.N_MIN <= Self.N, "REDQ-OFE: N_MIN must be ≤ N"
        comptime assert Self.UTD >= 1, "REDQ-OFE: UTD must be ≥ 1"
        comptime assert Self.POLICY_DELAY >= 1, "REDQ-OFE: POLICY_DELAY ≥ 1"
        comptime assert Self.PRED.OUT_DIM == Self.OBS_DIM, (
            "REDQ-OFE: predictor OUT must equal OBS"
        )

        var t = Self()
        t.ctx = ctx

        t.actor = Self.ACTOR.make[Self.train_target, Xavier](ctx)
        t.state_branch = Self.SB.make[Self.train_target, Xavier](ctx)
        t.action_branch = Self.AB.make[Self.train_target, Xavier](ctx)
        t.predictor = Self.PRED.make[Self.train_target, Xavier](ctx)
        t.ensemble = CriticEnsemble[Self.CRITIC, Self.N].make[
            Self.train_target, Xavier
        ](ctx=ctx)

        t.actor_opt = Adam(lr=actor_lr)
        t.sb_opt = Adam(lr=ofe_lr)
        t.ab_opt = Adam(lr=ofe_lr)
        t.pred_opt = Adam(lr=ofe_lr)
        comptime if Self.train_target == "gpu":
            t.actor_opt.adopt[Self.train_target, Self.ACTOR](t.actor, ctx)
            t.sb_opt.adopt[Self.train_target, Self.SB](t.state_branch, ctx)
            t.ab_opt.adopt[Self.train_target, Self.AB](t.action_branch, ctx)
            t.pred_opt.adopt[Self.train_target, Self.PRED](t.predictor, ctx)
        for i in range(Self.N):
            t.ensemble.opts[i].lr = critic_lr

        t.alpha_opt = ScalarAdam.new(flog(init_alpha), alpha_lr)

        t.feat_blk = OFEFeatureStep[
            Self.SB, Self.OBS_DIM, Self.ACT_DIM, Self.BATCH
        ].make[Self.train_target](ctx=ctx)
        t.target_y_blk = EnsembleTargetYBlockOFE[
            Self.ACTOR, Self.AB, Self.CRITIC, Self.N, Self.BATCH,
            Self.OBS_DIM, Self.PHI_S_DIM, Self.ACT_DIM, Self.N_MIN, Self.Q_MODE,
        ].make[Self.train_target](
            action_scale=action_scale, gamma=gamma, ctx=ctx
        )
        t.critic_blk = EnsembleCriticStepOFE[
            Self.AB, Self.CRITIC, Self.N, Self.OBS_DIM, Self.PHI_S_DIM,
            Self.ACT_DIM, Self.BATCH,
        ].make[Self.train_target](ctx=ctx)
        t.actor_blk = EnsembleActorStepOFE[
            Self.ACTOR, Self.AB, Self.CRITIC, Self.N, Self.BATCH,
            Self.PHI_S_DIM, Self.ACT_DIM,
        ].make[Self.train_target](action_scale=action_scale, ctx=ctx)
        t.aux_blk = OFEAuxLossStep[
            Self.SB, Self.AB, Self.PRED, Self.OBS_DIM, Self.ACT_DIM, Self.BATCH,
        ].make[Self.train_target](ctx=ctx)
        t.polyak_blk = EnsemblePolyakStep[
            Self.CRITIC, Self.N, Self.OBS_DIM, Self.ACT_DIM, Self.BATCH,
        ].make(tau=tau)
        t.alpha_blk = AlphaUpdateStep[
            Self.OBS_DIM, Self.ACT_DIM, Self.BATCH
        ].make(target_entropy=target_entropy)

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
            t._phi_scr.ensure(Self.PHI_S_DIM)
            t._ao_scr.ensure(2 * Self.ACT_DIM)
            t._alp_scr.ensure(Self.ACT_DIM + 1)
        return t^

    def set_beta(mut self, beta: Scalar[DT]):
        self.sample_blk.set_beta(beta)

    # ─── Inner tick — one (feat + target_y + critic + polyak + maybe actor). ──
    def _one_inner_tick(
        mut self,
    ) raises -> Tuple[Scalar[DT], Bool, Scalar[DT], Scalar[DT]]:
        self._inner_count += 1

        var alpha_val = fexp(self.alpha_opt.value)
        self.state.alpha = alpha_val

        # (1) Feature pre-pass — φ(s), φ(s').
        self.feat_blk.step[Self.train_target](self.state_branch, self.state)

        # (2) Target y — on φ(s').
        self.target_y_blk.resample_subset_idxs()
        self.target_y_blk.step[Self.train_target](
            self.state, self.actor, self.action_branch, self.ensemble,
            self.feat_blk.phi_sp, alpha_val,
        )

        # (3) Critic update — on φ(s).
        var cl = self.critic_blk.step[Self.train_target](
            self.state, self.action_branch, self.ensemble, self.feat_blk.phi_s,
        )

        # (4) Polyak every inner tick (paper-faithful).
        self.polyak_blk.step[Self.train_target](self.state, self.ensemble)

        # (5) Actor + α every POLICY_DELAY.
        var did_actor: Bool = False
        var actor_loss: Scalar[DT] = Scalar[DT](0.0)
        var lp_mean: Scalar[DT] = Scalar[DT](0.0)
        if self._inner_count % Self.POLICY_DELAY == 0:
            var res = self.actor_blk.forward_backward[Self.train_target](
                self.actor, self.actor_opt, self.action_branch, self.ensemble,
                self.feat_blk.phi_s, alpha_val, self.state.ctx,
            )
            self.state.actor_loss = res.loss
            self.state.log_prob_mean = res.log_prob_mean
            actor_loss = res.loss
            lp_mean = res.log_prob_mean
            did_actor = True
            self.alpha_blk.step["cpu"](self.state, self.alpha_opt)

        return (cl, did_actor, actor_loss, lp_mean)

    def _run_aux_step(mut self) raises -> Scalar[DT]:
        """One aux loss step on the CURRENT `state.mb_*`. forward+vjp on
        SB+AB+PRED is atomic (zero_grad → forward → vjp → step), so prior RL
        forwards clobbering caches is harmless. Returns the MSE loss."""
        return self.aux_blk.step[Self.train_target](
            self.state_branch, self.action_branch, self.predictor,
            self.sb_opt, self.ab_opt, self.pred_opt, self.state,
        )

    # ─── train_step — outer (one env step). Runs UTD inner ticks + 1 aux. ───
    def train_step(mut self, step_idx: Int) raises -> Bool:
        self.state.step_idx = step_idx
        self.state.did_step = True
        comptime if Self.train_target == "gpu":
            self.state.ctx = self.ctx

        self.sample_blk.step(self.state)
        if not self.state.did_step:
            return False

        var critic_loss_acc: Scalar[DT] = Scalar[DT](0.0)
        var actor_loss_last: Scalar[DT] = Scalar[DT](0.0)
        var lp_mean_last: Scalar[DT] = Scalar[DT](0.0)
        var did_actor_step: Bool = False
        var ticks_fired = 0

        var tick0 = self._one_inner_tick()
        critic_loss_acc += tick0[0]
        if tick0[1]:
            did_actor_step = True
            actor_loss_last = tick0[2]
            lp_mean_last = tick0[3]
        ticks_fired += 1

        for _ in range(Self.UTD - 1):
            self.state.did_step = True
            self.sample_blk.step(self.state)
            if not self.state.did_step:
                break
            var tick = self._one_inner_tick()
            critic_loss_acc += tick[0]
            if tick[1]:
                did_actor_step = True
                actor_loss_last = tick[2]
                lp_mean_last = tick[3]
            ticks_fired += 1

        # Aux loss step — ONCE per outer call, on the LAST sampled minibatch.
        var aux_loss = self._run_aux_step()

        self._total_train_steps += ticks_fired

        self._acc_critic_loss += critic_loss_acc
        self._acc_alpha += fexp(self.alpha_opt.value)
        self._acc_aux_loss += aux_loss
        self._acc_n_updates += 1
        if did_actor_step:
            self._acc_actor_loss += actor_loss_last
            self._acc_lp_mean += lp_mean_last
            self._acc_n_actor_updates += 1

        return True

    def total_train_steps(self) -> Int:
        return self._total_train_steps

    def inner_count(self) -> Int:
        return self._inner_count

    def alpha_value(self) -> Scalar[DT]:
        return fexp(self.alpha_opt.value)

    # ─── CUDA-graph capture surface (DEFERRED — trait-default no-ops) ───────
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
        comptime PHI = Self.PHI_S_DIM

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
        comptime PHI = Self.PHI_S_DIM
        comptime if Self.train_target == "cpu":
            self._ob_scr.ensure(OBS)
            self._phi_scr.ensure(PHI)
            self._ao_scr.ensure(2 * ACT)
            for d in range(OBS):
                self._ob_scr.data[d] = obs[d]
            call_forward["cpu", 1](
                self.state_branch,
                TensorRefs[Self.SB.ARITY](self._ob_scr), self._phi_scr
            )
            call_forward["cpu", 1](
                self.actor,
                TensorRefs[Self.ACTOR.ARITY](self._phi_scr), self._ao_scr
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
            var phi = Tensor.alloc_gpu(c, PHI)
            var ao = Tensor.alloc_gpu(c, 2 * ACT)
            call_forward["gpu", 1](
                self.state_branch,
                TensorRefs[Self.SB.ARITY](ob), phi, self.ctx
            )
            call_forward["gpu", 1](
                self.actor,
                TensorRefs[Self.ACTOR.ARITY](phi), ao, self.ctx
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
        comptime PHI = Self.PHI_S_DIM
        if step_idx < self.learning_starts:
            for j in range(ACT):
                var u = Scalar[DT](2.0 * random_float64() - 1.0)
                action_out[j] = u * self.action_scale
            return
        comptime if Self.train_target == "cpu":
            self._ob_scr.ensure(OBS)
            self._phi_scr.ensure(PHI)
            self._ao_scr.ensure(2 * ACT)
            self._alp_scr.ensure(ACT + 1)
            for d in range(OBS):
                self._ob_scr.data[d] = obs[d]
            call_forward["cpu", 1](
                self.state_branch,
                TensorRefs[Self.SB.ARITY](self._ob_scr), self._phi_scr
            )
            call_forward["cpu", 1](
                self.actor,
                TensorRefs[Self.ACTOR.ARITY](self._phi_scr), self._ao_scr
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
            var c = self.ctx.value()
            var ob = Tensor.alloc(OBS)
            for d in range(OBS):
                ob.data[d] = obs[d]
            ob.upload(c)
            var phi = Tensor.alloc_gpu(c, PHI)
            var ao = Tensor.alloc_gpu(c, 2 * ACT)
            var alp = Tensor.alloc_gpu(c, ACT + 1)
            call_forward["gpu", 1](
                self.state_branch,
                TensorRefs[Self.SB.ARITY](ob), phi, self.ctx
            )
            call_forward["gpu", 1](
                self.actor,
                TensorRefs[Self.ACTOR.ARITY](phi), ao, self.ctx
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
            "REDQOFETrainer.record_batch_gpu_nstep: n-step replay not supported"
            " (uniform 1-step replay only)"
        )

    # ─── Metrics / logging ─────────────────────────────────────────────────
    def flush_metrics(mut self) -> REDQOFEMetrics:
        """Drain per-flush-window accumulators into a `REDQOFEMetrics`
        snapshot and reset. Means use sum/count (0.0 sentinel if no updates)."""
        var inv_n: Scalar[DT] = Scalar[DT](1.0) / Scalar[DT](
            self._acc_n_updates
        ) if self._acc_n_updates > 0 else Scalar[DT](0.0)
        var inv_a: Scalar[DT] = Scalar[DT](1.0) / Scalar[DT](
            self._acc_n_actor_updates
        ) if self._acc_n_actor_updates > 0 else Scalar[DT](0.0)
        var m = REDQOFEMetrics(
            critic_loss=self._acc_critic_loss * inv_n,
            actor_loss=self._acc_actor_loss * inv_a,
            alpha=self._acc_alpha * inv_n,
            log_prob_mean=self._acc_lp_mean * inv_a,
            aux_loss=self._acc_aux_loss * inv_n,
            n_updates=self._acc_n_updates,
            n_actor_updates=self._acc_n_actor_updates,
        )
        self._acc_critic_loss = Scalar[DT](0.0)
        self._acc_actor_loss = Scalar[DT](0.0)
        self._acc_alpha = Scalar[DT](0.0)
        self._acc_lp_mean = Scalar[DT](0.0)
        self._acc_aux_loss = Scalar[DT](0.0)
        self._acc_n_updates = 0
        self._acc_n_actor_updates = 0
        return m^

    def flush_metrics_through_logger[
        L: Logger
    ](
        mut self,
        logger: Optional[Pointer[L, MutAnyOrigin]],
        step: Int,
    ) raises:
        var m = self.flush_metrics()
        if Bool(logger):
            var lg = logger.value()
            lg[].log_scalar("critic_loss", Float64(m.critic_loss), step)
            lg[].log_scalar("policy_loss", Float64(m.actor_loss), step)
            lg[].log_scalar("alpha", Float64(m.alpha), step)
            lg[].log_scalar("log_prob_mean", Float64(m.log_prob_mean), step)
            lg[].log_scalar("aux_loss", Float64(m.aux_loss), step)
            lg[].log_scalar("n_updates", Float64(m.n_updates), step)
            lg[].log_scalar("n_actor_updates", Float64(m.n_actor_updates), step)

    def flush_timer_log(mut self) -> String:
        return String("")

    # ─── Checkpoint (ONE v3 file: actor + N critics + SB + AB + PRED + α) ──
    def save_state(mut self, path: String) raises:
        """ONE v3 `storage-ckpt` file: the actor, the N ONLINE critics and the
        three OFE nets, then as `K` sections α's whole optimizer (`alpha.*`)
        and the train-step counter. Atomic and chunked. Network optimizer
        moments NOT persisted (resume re-warms)."""
        var w = BinaryCheckpointWriter(save_moments=False)
        write_model[Self.train_target](w, self.actor, self.ctx, "actor")
        for i in range(Self.N):
            write_model[Self.train_target](
                w, self.ensemble.pairs[i].online, self.ctx, "critic" + String(i)
            )
        write_model[Self.train_target](w, self.state_branch, self.ctx, "state_branch")
        write_model[Self.train_target](w, self.action_branch, self.ctx, "action_branch")
        write_model[Self.train_target](w, self.predictor, self.ctx, "predictor")
        var sc = CheckpointScalars()
        self.alpha_opt.put_state(sc, "alpha")
        sc.set_int("total_train_steps", self._total_train_steps)
        w.write_scalars(sc)
        _write_file_bytes(path, w.content)

    def load_state(mut self, path: String) raises:
        """Restore every net, α and the counter (v3, or the legacy v2 text
        this trainer wrote before, which carries neither α nor the counter),
        then hard-copy online → target."""
        var bytes = _read_file_bytes(path)
        if _is_v3_header(bytes):
            var rb = BinaryCheckpointReader(bytes^)
            read_model[Self.train_target](rb, self.actor, self.ctx, "actor")
            for i in range(Self.N):
                read_model[Self.train_target](
                    rb, self.ensemble.pairs[i].online, self.ctx,
                    "critic" + String(i),
                )
            read_model[Self.train_target](rb, self.state_branch, self.ctx, "state_branch")
            read_model[Self.train_target](rb, self.action_branch, self.ctx, "action_branch")
            read_model[Self.train_target](rb, self.predictor, self.ctx, "predictor")
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
            walk_params[Self.train_target](self.state_branch, r, self.ctx, "state_branch"
            )
            walk_params[Self.train_target](self.action_branch, r, self.ctx, "action_branch"
            )
            walk_params[Self.train_target](self.predictor, r, self.ctx, "predictor"
            )
            r.mode = 1
            var _sref2 = ParamVisitorRef.of[type_of(r), Self.train_target](r)
            self.actor.for_each_state[Self.train_target](_sref2, self.ctx, "actor")
            for i in range(Self.N):
                self.ensemble.pairs[i].online.for_each_state[Self.train_target](
                    r, self.ctx, "critic" + String(i)
                )
            self.state_branch.for_each_state[Self.train_target](
                r, self.ctx, "state_branch"
            )
            self.action_branch.for_each_state[Self.train_target](
                r, self.ctx, "action_branch"
            )
            self.predictor.for_each_state[Self.train_target](
                r, self.ctx, "predictor"
            )
        for i in range(Self.N):
            self.ensemble.pairs[i].target_net.polyak_from[Self.train_target](
                self.ensemble.pairs[i].online, Scalar[DT](1.0), self.ctx
            )
