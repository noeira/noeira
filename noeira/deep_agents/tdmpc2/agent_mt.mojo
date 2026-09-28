"""TD-MPC2 multi-task agent (deep_agents, storage framework, CPU + GPU, MPC-off).

A parallel struct to `TDMPC2Agent` for the multi-task setting: one agent trained
over a SET of envs (heterogeneous obs/action dims, padded to `MAX_OBS`/`MAX_ACT`),
conditioned on a learned per-task embedding. The single-task `TDMPC2Agent` and all
its blocks are left untouched (bit-identical by construction); everything here
uses the `*MT` variants + the storage `TaskEmbedding` table.

Storage migration (mirrors `agent.mojo`): the 5 online Q heads, 5 target Q heads,
and 5 Q optimizers are DISTINCT FIELDS (q0..q4 / qt0..qt4 / qo0..qo4; NQ=5).
Storage threads externals into ONE forward/vjp call (two `mut` List subscripts
can't alias). The WM step threads all 5 online Q as distinct args; the random
PAIR steps (policy: online (pa,pb); td: target (ta,tb)) use a comptime-unrolled
guarded dispatch so two DISTINCT fields are threaded. Adam via ctor+adopt+step;
polyak via `Module.polyak_from`; a one-file v3 checkpoint (task table last,
its Adam powers + pi_scale + step count as `K` scalars); action/record entry points take `List[Scalar[DT]]`.

Acting: MPC-off (`a = π(encode([obs|task_emb]))`) by default, or MPPI planning
via `select_action_mpc` on the GPU target — the planner is task-conditioned
through `TDMPC2RolloutCallbackGPUMT` (`callback_mt.mojo`), which splices the
task embedding into every net input in the order `wm_graph_mt` uses.

train_step: sample length-T windows + per-window task id (host) → t-major →
gather task embeddings → TD targets (stop-grad) → WM BPTT (accumulates embedding
grad sites 1+2) → policy update on encoded latents (site 3) → ONE embedding Adam
step → Polyak. The embedding table's `zero_grad`/`step` bracket the whole step.
"""

from noeira.io.artifact_sink import ArtifactSink
from noeira.deep_agents.training.checkpoint import announce_checkpoint
from noeira.nn.core.param import walk_params, ParamVisitorRef
from std.math import tanh
from std.random import random_float64
from std.time import perf_counter_ns
from layout import Layout, LayoutTensor
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.nn.constants import DT, TPB
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Kaiming, Zero
from noeira.nn.optimizer.adam import Adam
from noeira.nn.primitives.rsample import RSample
from noeira.nn.core.checkpoint import (
    BinaryCheckpointWriter, BinaryCheckpointReader, CheckpointReader,
    CheckpointScalars, write_model, read_model, _split_lines, _is_v3_header,
    _read_file_bytes, _write_file_bytes,
)

from noeira.deep_agents.data.sequence_replay import SequenceReplay
from noeira.core.logger import Logger, NoOpLogger
from noeira.deep_agents.training.batched_env import BatchedEnv
from noeira.deep_agents.training.blocks.action_select import (
    warmup_uniform_batched,
)
from noeira.planners.trajectory.mppi import MPPIGPUBatched
from max.gpu.host import HostBuffer
from .agent import _crossed
from .callback_mt import TDMPC2RolloutCallbackGPUMT
from .batched_acting_mt import (
    tdmpc2_mt_select_action_batched,
    tdmpc2_mt_encode_batched,
    mt_apply_mask_kernel,
)
from .metrics import TDMPC2Metrics

from .nets_mt import (
    TDMPC2EncoderMT, TDMPC2DynamicsMT, TDMPC2RewardMT, TDMPC2QNetMT,
    TDMPC2PolicyMT, TDMPC2TerminationMT,
)
from .wm_graph import NQ
from .wm_graph_mt import TDMPC2WMGraphMT
from .wm_step_mt import WMStepMT
from .policy_step_mt import PolicyStepMT
from .td_target_step_mt import TDTargetStepMT
from .task_embedding import TaskEmbedding


@fieldwise_init
struct TDMPC2MultiTaskAgent[
    target: StaticString,
    MAX_OBS: Int,
    ENC: Int,
    MAX_ACT: Int,
    LATENT: Int,
    MLP: Int,
    BINS: Int,
    SN: Int,
    VMIN: Int,
    VMAX: Int,
    B: Int,
    H: Int,
    CAP: Int,
    NUM_TASKS: Int,
    TASK_EMB: Int,
    QP: Float64 = 0.0,
    # MPPI planning budget. Defaulted and placed LAST so every existing
    # call site keeps working; only `select_action_mpc` reads them.
    NUM_SAMPLES: Int = 512,
    NUM_PI_TRAJS: Int = 24,
    NUM_ELITES: Int = 64,
    NUM_ITERS: Int = 8 if MAX_ACT >= 20 else 6,
](Movable & Deinitable):
    comptime EncT = TDMPC2EncoderMT[
        Self.MAX_OBS, Self.ENC, Self.LATENT, Self.SN, Self.TASK_EMB
    ]
    comptime DynT = TDMPC2DynamicsMT[
        Self.LATENT, Self.MAX_ACT, Self.MLP, Self.SN, Self.TASK_EMB
    ]
    comptime RewT = TDMPC2RewardMT[
        Self.LATENT, Self.MAX_ACT, Self.MLP, Self.BINS, Self.TASK_EMB
    ]
    comptime QNetT = TDMPC2QNetMT[
        Self.LATENT, Self.MAX_ACT, Self.MLP, Self.BINS, Self.TASK_EMB, Self.QP
    ]
    comptime TermT = TDMPC2TerminationMT[
        Self.LATENT, Self.MAX_ACT, Self.MLP, Self.TASK_EMB
    ]
    comptime PolicyT = TDMPC2PolicyMT[
        Self.LATENT, Self.MAX_ACT, Self.MLP, Self.TASK_EMB
    ]
    comptime GraphT = TDMPC2WMGraphMT[
        Self.LATENT, Self.MAX_ACT, Self.MLP, Self.BINS, Self.SN, Self.VMIN,
        Self.VMAX, Self.TASK_EMB, Self.QP,
    ]
    comptime EmbT = TaskEmbedding[Self.NUM_TASKS, Self.TASK_EMB]
    comptime MPC_BT = Self.NUM_SAMPLES + Self.NUM_PI_TRAJS
    comptime PlannerT = MPPIGPUBatched[
        Self.LATENT, Self.MAX_ACT, Self.H, Self.NUM_SAMPLES,
        Self.NUM_PI_TRAJS, Self.NUM_ELITES, Self.NUM_ITERS, 1,
    ]
    comptime MpcCB = TDMPC2RolloutCallbackGPUMT[
        Self.MAX_ACT, Self.LATENT, Self.MLP, Self.BINS, Self.SN, Self.VMIN,
        Self.VMAX, Self.TASK_EMB, NQ, Self.MPC_BT, Self.QP,
    ]
    comptime PB = (Self.H + 1) * Self.B
    comptime AOBS = Self.MAX_OBS + Self.TASK_EMB
    comptime PIN = Self.LATENT + Self.TASK_EMB
    comptime WMStepT = WMStepMT[
        Self.MAX_OBS, Self.ENC, Self.MAX_ACT, Self.LATENT, Self.MLP, Self.BINS,
        Self.SN, Self.VMIN, Self.VMAX, Self.B, Self.H, Self.NUM_TASKS,
        Self.TASK_EMB, Self.QP,
    ]
    comptime PolStepT = PolicyStepMT[
        Self.LATENT, Self.MAX_ACT, Self.MLP, Self.BINS, Self.VMIN, Self.VMAX,
        Self.PB, Self.NUM_TASKS, Self.TASK_EMB, Self.QP,
    ]
    comptime TDStepT = TDTargetStepMT[
        Self.MAX_OBS, Self.ENC, Self.MAX_ACT, Self.LATENT, Self.MLP, Self.BINS,
        Self.SN, Self.VMIN, Self.VMAX, Self.B, Self.H, Self.NUM_TASKS,
        Self.TASK_EMB, Self.QP,
    ]

    var encoder: Self.EncT
    var dynamics: Self.DynT
    var reward: Self.RewT
    # 5 online Q heads (distinct fields; threaded as externals).
    var q0: Self.QNetT
    var q1: Self.QNetT
    var q2: Self.QNetT
    var q3: Self.QNetT
    var q4: Self.QNetT
    # 5 target Q heads.
    var qt0: Self.QNetT
    var qt1: Self.QNetT
    var qt2: Self.QNetT
    var qt3: Self.QNetT
    var qt4: Self.QNetT
    var policy: Self.PolicyT
    var termination: Self.TermT
    var task_emb: Self.EmbT

    var enc_opt: Adam
    var dyn_opt: Adam
    var rew_opt: Adam
    # 5 Q optimizers.
    var qo0: Adam
    var qo1: Adam
    var qo2: Adam
    var qo3: Adam
    var qo4: Adam
    var pi_opt: Adam
    var term_opt: Adam

    var wm_graph: Self.GraphT
    var wm_step: Self.WMStepT
    var pol_step: Self.PolStepT
    var td_step: Self.TDStepT
    var act_rsample: RSample[Self.MAX_ACT]
    var replay: SequenceReplay[Self.MAX_OBS, Self.MAX_ACT, Self.CAP]

    # Per-task action mask [NUM_TASKS, MAX_ACT] (1=active, 0=unused), applied at
    # acting time (recorded actions are masked by the env wrapper). Default 1s.
    var action_mask: List[Scalar[DT]]
    # Acting task id (set by the driver before select_action / record).
    var cur_task: Int

    # ── batched-acting scratch — allocated ONCE (see batched_acting_mt) ───
    var act_ob: Tensor
    var act_ein: Tensor
    var act_z: Tensor
    var act_pin: Tensor
    var act_pio: Tensor
    var act_alp: Tensor
    var act_tem: Tensor
    var act_mask: Tensor
    var act_tids: Tensor
    # ── MPC (GPU only) — planner + rollout callback + landing buffers ────
    # Built lazily on first plan; `_mpc_ready` rebuilds the callback if the
    # agent has MOVED, since it holds raw pointers into our own modules.
    var planner: Optional[Self.PlannerT]
    var mpc_cb: Optional[Self.MpcCB]
    var mpc_out: Optional[DeviceBuffer[DT]]
    var mpc_host: Optional[HostBuffer[DT]]
    var mpc_row: Tensor
    # ⚠ A DEDICATED single-row task-id tensor. Reusing `act_tids` for both the
    # 1-row gather (the planner's broadcast) and the N-row gather (the encode)
    # resizes one tensor between 1 and N every iteration — and
    # `Tensor.upload_resident` CANNOT GROW an existing device buffer:
    # `ensure_gpu(ctx, self.n)` compares `self.n < n` after the host-side
    # `ensure` already bumped `self.n`, so it silently keeps the old, smaller
    # device buffer and the next copy dies with "Copy size exceeds Metal
    # buffer length". Two tensors, two fixed sizes, no resize.
    var mpc_tids: Tensor
    var mpc_ob: Tensor
    var mpc_z0: Tensor

    var gamma: Scalar[DT]
    var tau: Scalar[DT]
    var bce_coef: Scalar[DT]
    var action_scale: Scalar[DT]
    var learning_starts: Int
    var step_count: Int
    var _last_wm: Scalar[DT]
    var _last_pi: Scalar[DT]
    var _last_cons: Scalar[DT]
    var _last_rew: Scalar[DT]
    var _last_val: Scalar[DT]
    var _last_term: Scalar[DT]
    var _cons_acc: Scalar[DT]
    var _rew_acc: Scalar[DT]
    var _val_acc: Scalar[DT]
    var _term_acc: Scalar[DT]
    var _pi_acc: Scalar[DT]
    var _q_mean_acc: Scalar[DT]
    var _q_min_last: Scalar[DT]
    var _q_max_last: Scalar[DT]
    var _td_mean_acc: Scalar[DT]
    var _td_min_last: Scalar[DT]
    var _td_max_last: Scalar[DT]
    var _n_diag: Int
    var ctx: Optional[DeviceContext]
    var temperature: Scalar[DT]

    # ── comptime accessors: distinct online / target Q field by index ──────
    def get_q[i: Int](mut self) -> ref[MutAnyOrigin] Self.QNetT:
        comptime if i == 0:
            return self.q0
        elif i == 1:
            return self.q1
        elif i == 2:
            return self.q2
        elif i == 3:
            return self.q3
        else:
            return self.q4

    def get_qt[i: Int](mut self) -> ref[MutAnyOrigin] Self.QNetT:
        comptime if i == 0:
            return self.qt0
        elif i == 1:
            return self.qt1
        elif i == 2:
            return self.qt2
        elif i == 3:
            return self.qt3
        else:
            return self.qt4

    @staticmethod
    def make(
        lr: Scalar[DT] = Scalar[DT](3e-4),
        gamma: Scalar[DT] = Scalar[DT](0.99),
        tau: Scalar[DT] = Scalar[DT](0.01),
        action_scale: Scalar[DT] = Scalar[DT](1.0),
        learning_starts: Int = 1000,
        enc_lr_scale: Scalar[DT] = Scalar[DT](0.3),
        temperature: Scalar[DT] = Scalar[DT](0.5),
        bce_coef: Scalar[DT] = Scalar[DT](0.0),
        ctx: Optional[DeviceContext] = None,
    ) raises -> Self:
        comptime tg = Self.target
        var enc = Self.EncT.make[tg, INIT=Kaiming](ctx=ctx)
        var dyn = Self.DynT.make[tg, INIT=Kaiming](ctx=ctx)
        var rew = Self.RewT.make[tg, INIT=Kaiming](ctx=ctx)
        var pol = Self.PolicyT.make[tg, INIT=Kaiming](ctx=ctx)

        var q0 = Self.QNetT.make[tg, INIT=Kaiming](ctx=ctx)
        var q1 = Self.QNetT.make[tg, INIT=Kaiming](ctx=ctx)
        var q2 = Self.QNetT.make[tg, INIT=Kaiming](ctx=ctx)
        var q3 = Self.QNetT.make[tg, INIT=Kaiming](ctx=ctx)
        var q4 = Self.QNetT.make[tg, INIT=Kaiming](ctx=ctx)
        var qt0 = Self.QNetT.make[tg, INIT=Kaiming](ctx=ctx)
        var qt1 = Self.QNetT.make[tg, INIT=Kaiming](ctx=ctx)
        var qt2 = Self.QNetT.make[tg, INIT=Kaiming](ctx=ctx)
        var qt3 = Self.QNetT.make[tg, INIT=Kaiming](ctx=ctx)
        var qt4 = Self.QNetT.make[tg, INIT=Kaiming](ctx=ctx)

        var qo0 = Adam(lr=lr)
        var qo1 = Adam(lr=lr)
        var qo2 = Adam(lr=lr)
        var qo3 = Adam(lr=lr)
        var qo4 = Adam(lr=lr)
        comptime if tg == "gpu":
            qo0.adopt[tg, Self.QNetT](q0, ctx)
            qo1.adopt[tg, Self.QNetT](q1, ctx)
            qo2.adopt[tg, Self.QNetT](q2, ctx)
            qo3.adopt[tg, Self.QNetT](q3, ctx)
            qo4.adopt[tg, Self.QNetT](q4, ctx)

        # hard-copy online → target (tau = 1.0).
        qt0.polyak_from[tg](q0, Scalar[DT](1.0), ctx)
        qt1.polyak_from[tg](q1, Scalar[DT](1.0), ctx)
        qt2.polyak_from[tg](q2, Scalar[DT](1.0), ctx)
        qt3.polyak_from[tg](q3, Scalar[DT](1.0), ctx)
        qt4.polyak_from[tg](q4, Scalar[DT](1.0), ctx)

        # Q-dropout (item D): target Q nets eval (no masking) when QP>0.
        comptime if Self.QP > 0.0:
            qt0.set_attr["training"](Scalar[DT](0.0))
            qt1.set_attr["training"](Scalar[DT](0.0))
            qt2.set_attr["training"](Scalar[DT](0.0))
            qt3.set_attr["training"](Scalar[DT](0.0))
            qt4.set_attr["training"](Scalar[DT](0.0))

        # Termination head (item B): Zero-init at bce_coef=0 (no RNG draw),
        # Kaiming when episodic. Built before the table (RNG discipline).
        var term: Self.TermT
        if bce_coef > Scalar[DT](0.0):
            term = Self.TermT.make[tg, INIT=Kaiming](ctx=ctx)
        else:
            term = Self.TermT.make[tg, INIT=Zero](ctx=ctx)

        var enc_opt = Adam(lr=lr * enc_lr_scale)
        var dyn_opt = Adam(lr=lr)
        var rew_opt = Adam(lr=lr)
        var pi_opt = Adam(lr=lr)
        pi_opt.eps = Scalar[DT](1e-5)
        var term_opt = Adam(lr=lr)
        comptime if tg == "gpu":
            enc_opt.adopt[tg, Self.EncT](enc, ctx)
            dyn_opt.adopt[tg, Self.DynT](dyn, ctx)
            rew_opt.adopt[tg, Self.RewT](rew, ctx)
            pi_opt.adopt[tg, Self.PolicyT](pol, ctx)
            term_opt.adopt[tg, Self.TermT](term, ctx)

        var ar = RSample[Self.MAX_ACT].make[tg, INIT=Zero](ctx=ctx)
        ar.action_scale = action_scale

        # Task embedding built LAST (RNG discipline) — its random init draws from
        # the global RNG, so building it after every net keeps the nets' init
        # stream unperturbed (matches the term-head ordering convention).
        var te = Self.EmbT.make[tg](ctx=ctx, lr=lr)

        var amask = List[Scalar[DT]](
            length=Self.NUM_TASKS * Self.MAX_ACT, fill=Scalar[DT](1.0)
        )

        return Self(
            encoder=enc^, dynamics=dyn^, reward=rew^,
            q0=q0^, q1=q1^, q2=q2^, q3=q3^, q4=q4^,
            qt0=qt0^, qt1=qt1^, qt2=qt2^, qt3=qt3^, qt4=qt4^,
            policy=pol^, termination=term^, task_emb=te^,
            enc_opt=enc_opt^, dyn_opt=dyn_opt^, rew_opt=rew_opt^,
            qo0=qo0^, qo1=qo1^, qo2=qo2^, qo3=qo3^, qo4=qo4^,
            pi_opt=pi_opt^, term_opt=term_opt^,
            wm_graph=Self.GraphT.make[tg, INIT=Kaiming](ctx=ctx),
            wm_step=Self.WMStepT.make[tg](ctx=ctx, termination_coef=bce_coef),
            pol_step=Self.PolStepT.make[tg](ctx=ctx),
            td_step=Self.TDStepT.make[tg](ctx=ctx),
            act_rsample=ar^,
            replay=SequenceReplay[
                Self.MAX_OBS, Self.MAX_ACT, Self.CAP
            ].new(),
            action_mask=amask^, cur_task=0,
            act_ob=Tensor(), act_ein=Tensor(), act_z=Tensor(),
            act_pin=Tensor(), act_pio=Tensor(), act_alp=Tensor(),
            act_tem=Tensor(), act_mask=Tensor(), act_tids=Tensor(),
            planner=None, mpc_cb=None, mpc_out=None, mpc_host=None,
            mpc_row=Tensor(), mpc_tids=Tensor(),
            mpc_ob=Tensor(), mpc_z0=Tensor(),
            gamma=gamma, tau=tau, bce_coef=bce_coef, action_scale=action_scale,
            learning_starts=learning_starts, step_count=0,
            _last_wm=Scalar[DT](0.0), _last_pi=Scalar[DT](0.0),
            _last_cons=Scalar[DT](0.0), _last_rew=Scalar[DT](0.0),
            _last_val=Scalar[DT](0.0), _last_term=Scalar[DT](0.0),
            _cons_acc=Scalar[DT](0.0), _rew_acc=Scalar[DT](0.0),
            _val_acc=Scalar[DT](0.0), _term_acc=Scalar[DT](0.0),
            _pi_acc=Scalar[DT](0.0),
            _q_mean_acc=Scalar[DT](0.0), _q_min_last=Scalar[DT](0.0),
            _q_max_last=Scalar[DT](0.0), _td_mean_acc=Scalar[DT](0.0),
            _td_min_last=Scalar[DT](0.0), _td_max_last=Scalar[DT](0.0),
            _n_diag=0, ctx=ctx, temperature=temperature,
        )

    # ── task / mask plumbing ───────────────────────────────────────────
    def set_task(mut self, t: Int):
        self.cur_task = t

    def task(self) -> Int:
        return self.cur_task

    def set_action_mask(mut self, t: Int, ref mask: List[Scalar[DT]]):
        for j in range(Self.MAX_ACT):
            self.action_mask[t * Self.MAX_ACT + j] = mask[j]

    # ── acting (MPC-off): a = π(encode([obs|task_emb])) ─────────────────
    def select_action(
        mut self,
        ref obs: List[Scalar[DT]],
        mut act_out: List[Scalar[DT]],
        explore: Bool = True,
    ) raises:
        comptime tg = Self.target
        comptime A = Self.MAX_ACT
        comptime LAT = Self.LATENT
        comptime MO = Self.MAX_OBS
        comptime EMB = Self.TASK_EMB
        comptime AOBS = Self.AOBS
        comptime PIN = Self.PIN
        var ctx = self.ctx

        # gather this task's embedding (1 row).
        var tids = Tensor.alloc(1)
        tids.data[0] = Scalar[DT](self.cur_task)
        comptime if tg == "gpu":
            tids.upload(ctx.value())
        var tem = Tensor.make[tg](EMB, ctx)
        self.task_emb.gather[tg, 1](tids, tem, ctx)
        comptime if tg == "gpu":
            tem.download(ctx.value())

        # ein = [obs | tem]
        var ein = Tensor.alloc(AOBS)
        for i in range(MO):
            ein.data[i] = obs[i]
        for e in range(EMB):
            ein.data[MO + e] = tem.data[e]
        comptime if tg == "gpu":
            ein.upload(ctx.value())
        var z = Tensor.make[tg](LAT, ctx)
        self.encoder.forward[tg, 1](TensorRefs[1](ein), z, ctx)
        comptime if tg == "gpu":
            z.download(ctx.value())

        # pin = [z | tem]
        var pin = Tensor.alloc(PIN)
        for k in range(LAT):
            pin.data[k] = z.data[k]
        for e in range(EMB):
            pin.data[LAT + e] = tem.data[e]
        comptime if tg == "gpu":
            pin.upload(ctx.value())
        var pio = Tensor.make[tg](2 * A, ctx)
        self.policy.forward[tg, 1](TensorRefs[1](pin), pio, ctx)
        if explore:
            var alp = Tensor.make[tg](A + 1, ctx)
            self.act_rsample.forward[tg, 1](TensorRefs[1](pio), alp, ctx)
            comptime if tg == "gpu":
                alp.download(ctx.value())
            for j in range(A):
                act_out[j] = alp.data[j] * self.action_mask[
                    self.cur_task * A + j
                ]
        else:
            comptime if tg == "gpu":
                pio.download(ctx.value())
            for j in range(A):
                act_out[j] = tanh(pio.data[j]) * self.action_scale * (
                    self.action_mask[self.cur_task * A + j]
                )

    def select_greedy_action(
        mut self,
        ref obs: List[Scalar[DT]],
        mut act_out: List[Scalar[DT]],
    ) raises:
        self.select_action(obs, act_out, explore=False)

    # ── acting (MPC): plan with the task-conditioned world model ─────────

    def mpc_start_episode(mut self) raises:
        """Drop the planner's warm-start mean — call after an env reset, and
        after a TASK switch: a plan carried over from another task's segment
        is a mean optimized for a different reward."""
        comptime if Self.target == "gpu":
            if self.planner:
                self.planner.value().start_episode(0)

    def _mpc_ready(mut self, ctx: DeviceContext) raises:
        """Build the planner + callback on first use, and REBUILD the callback
        if this agent has moved since.

        Same hazard as the single-task agent: `MpcCB` holds raw
        `MutUntrackedOrigin` pointers into our own modules, which the compiler
        will not flag when they go stale, so a cached callback that survived a
        move would plan against freed memory and return plausible actions. The
        guard compares the cached `dyn` pointer against a fresh one.
        """
        comptime A = Self.MAX_ACT
        if not self.planner:
            self.planner = Optional(Self.PlannerT(ctx))
        var fresh = rebind[Pointer[Self.DynT, MutUntrackedOrigin]](
            Pointer(to=self.dynamics)
        )
        var stale = True
        if self.mpc_cb:
            stale = self.mpc_cb.value().dyn != fresh
        if stale:
            self.mpc_cb = Optional(
                Self.MpcCB.make(
                    self.dynamics, self.reward, self.policy,
                    self.qt0, self.qt1, self.qt2, self.qt3, self.qt4,
                    self.action_scale, ctx,
                )
            )
        if not self.mpc_out:
            self.mpc_out = Optional(ctx.enqueue_create_buffer[DT](A))
            self.mpc_host = Optional(ctx.enqueue_create_host_buffer[DT](A))

    def select_action_mpc(
        mut self,
        ref obs: List[Scalar[DT]],
        mut act_out: List[Scalar[DT]],
        explore: Bool = True,
        num_iters: Int = 0,
    ) raises:
        """MPC acting for `cur_task`: plan in latent space with the
        task-conditioned world model. GPU only.

        `num_iters` overrides the comptime MPPI iteration budget for THIS call
        (0 = use `NUM_ITERS`), mirroring the single-task agent. Cost is
        near-linear in it and it is the only budget knob that can vary at
        runtime — `NUM_SAMPLES` / `NUM_PI_TRAJS` size the planner's device
        buffers at construction. Use it to trade plan quality for frame rate
        in an interactive viewer.

        ⚠ The task embedding is re-gathered and re-broadcast on EVERY call.
        The table is trained, so a row captured once would plan against an
        embedding the world model has already moved away from — the same
        staleness the batched acting path avoids.
        """
        comptime assert Self.target == "gpu", (
            "select_action_mpc requires target='gpu' (CPU MPPI is eval-only)"
        )
        comptime A = Self.MAX_ACT
        comptime MO = Self.MAX_OBS
        comptime LAT = Self.LATENT
        comptime EMB = Self.TASK_EMB
        var ctx = self.ctx.value()
        self._mpc_ready(ctx)

        # tem row for cur_task → broadcast over the planning candidates.
        self._mt_gather_row(self.cur_task, Optional(ctx))
        self.mpc_cb.value().set_task_embedding(self.mpc_row, ctx)

        # z0 = encode([obs | tem]) — the encoder is task-conditioned too, so
        # the root latent needs the embedding before the plan starts.
        self.mpc_row.download(ctx)
        self.mpc_ob.ensure(Self.AOBS)
        for i in range(MO):
            self.mpc_ob.data[i] = obs[i]
        for e in range(EMB):
            self.mpc_ob.data[MO + e] = self.mpc_row.data[e]
        self.mpc_ob.upload_resident(ctx)
        self.mpc_z0.ensure_gpu(ctx, LAT)
        self.encoder.forward[Self.target, 1](
            TensorRefs[1](self.mpc_ob), self.mpc_z0, Optional(ctx)
        )

        var out_lt = LayoutTensor[DT, Layout.row_major(1 * A), MutAnyOrigin](
            self.mpc_out.value()
        )
        self.planner.value().plan_gpu[Self.MpcCB](
            ctx, self.mpc_cb.value(),
            self.mpc_z0.lt["gpu", Layout.row_major(1, LAT)](),
            out_lt,
            gamma=Float64(self.gamma),
            temperature=Float64(self.temperature),
            action_scale=Float64(self.action_scale),
            deterministic=not explore,
            num_iters=num_iters,
        )
        ctx.enqueue_copy(self.mpc_host.value(), self.mpc_out.value())
        ctx.synchronize()
        # The planner knows nothing about per-task action masks, so pad
        # columns are zeroed here (a no-op when every task uses MAX_ACT).
        for j in range(A):
            act_out[j] = (
                self.mpc_host.value().unsafe_ptr()[unsafe_offset=j]
                * self.action_mask[self.cur_task * A + j]
            )

    def record(
        mut self,
        ref obs: List[Scalar[DT]],
        ref act: List[Scalar[DT]],
        reward: Scalar[DT],
        done: Scalar[DT],
    ) raises:
        self.replay.record_task(
            rebind[Pointer[Scalar[DT], MutAnyOrigin]](
                Pointer(to=obs[0])
            ),
            rebind[Pointer[Scalar[DT], MutAnyOrigin]](
                Pointer(to=act[0])
            ),
            reward, done, self.cur_task,
        )

    def last_wm_loss(self) -> Scalar[DT]:
        return self._last_wm

    def last_pi_loss(self) -> Scalar[DT]:
        return self._last_pi

    def last_consistency_loss(self) -> Scalar[DT]:
        return self._last_cons

    def last_reward_loss(self) -> Scalar[DT]:
        return self._last_rew

    def last_value_loss(self) -> Scalar[DT]:
        return self._last_val

    def last_termination_loss(self) -> Scalar[DT]:
        return self._last_term

    def pi_scale(self) -> Scalar[DT]:
        return self.pol_step.scale.value

    def task_pi_scale(self, t: Int) -> Scalar[DT]:
        """Per-task policy-loss scale — 1.0 until `set_per_task_pi_scale`
        turns the table on. Log it: whether the per-task spreads actually
        SEPARATE is what makes the experiment readable."""
        return self.pol_step.task_scale(t)

    def set_per_task_pi_scale(
        mut self, enable: Bool, max_reweight: Scalar[DT] = Scalar[DT](10.0)
    ):
        """⚠ A DEVIATION FROM THE REFERENCE, off by default.

        TD-MPC2 normalizes the policy loss by ONE running scale across every
        task (`tdmpc2/tdmpc2.py:34` — a single `RunningScale`, even for MT80).
        Enabling this gives each task its own, which makes each task's policy
        gradient invariant to its OWN Q spread instead of the mixed-batch one.

        Motivation (`docs/TDMPC2_MULTITASK_VALIDATION.md`): on walker
        stand+walk+run the shared scale was set by the two solved tasks
        (Q ~98) while run sat at ~16 and collapsed to the standing floor — with
        a MATCHED run-weighted gradient budget (104k vs 99k), so it was not a
        data problem.

        `max_reweight` clamps the per-row weight symmetrically; a task whose Q
        spread collapses toward zero would otherwise swamp the batch.

        Leave it off to reproduce the reference and every result measured
        before 2026-08-13."""
        self.pol_step.per_task_scale = enable
        self.pol_step.max_reweight = max_reweight

    def flush_metrics[
        L: Logger
    ](
        mut self,
        logger: Optional[Pointer[L, MutAnyOrigin]],
        step: Int,
    ) raises -> TDMPC2Metrics:
        var n = self._n_diag if self._n_diag > 0 else 1
        var inv = Scalar[DT](1.0) / Scalar[DT](n)
        var m = TDMPC2Metrics(
            consistency_loss=self._cons_acc * inv,
            reward_loss=self._rew_acc * inv,
            value_loss=self._val_acc * inv,
            termination_loss=self._term_acc * inv,
            wm_loss=(
                self._cons_acc + self._rew_acc + self._val_acc + self._term_acc
            ) * inv,
            pi_loss=self._pi_acc * inv,
            pi_scale=self.pol_step.scale.value,
            q_mean=self._q_mean_acc * inv,
            q_min=self._q_min_last,
            q_max=self._q_max_last,
            td_target_mean=self._td_mean_acc * inv,
            td_target_min=self._td_min_last,
            td_target_max=self._td_max_last,
        )
        if Bool(logger):
            var lg = logger.value()
            lg[].log_scalar("consistency_loss", Float64(m.consistency_loss), step)
            lg[].log_scalar("reward_loss", Float64(m.reward_loss), step)
            lg[].log_scalar("value_loss", Float64(m.value_loss), step)
            lg[].log_scalar("termination_loss", Float64(m.termination_loss), step)
            lg[].log_scalar("wm_loss", Float64(m.wm_loss), step)
            lg[].log_scalar("policy_loss", Float64(m.pi_loss), step)
            lg[].log_scalar("pi_scale", Float64(m.pi_scale), step)
            lg[].log_scalar("q_mean", Float64(m.q_mean), step)
            lg[].log_scalar("q_min", Float64(m.q_min), step)
            lg[].log_scalar("q_max", Float64(m.q_max), step)
            lg[].log_scalar("td_target_mean", Float64(m.td_target_mean), step)
            lg[].log_scalar("td_target_min", Float64(m.td_target_min), step)
            lg[].log_scalar("td_target_max", Float64(m.td_target_max), step)
        self._cons_acc = Scalar[DT](0.0)
        self._rew_acc = Scalar[DT](0.0)
        self._val_acc = Scalar[DT](0.0)
        self._term_acc = Scalar[DT](0.0)
        self._pi_acc = Scalar[DT](0.0)
        self._q_mean_acc = Scalar[DT](0.0)
        self._td_mean_acc = Scalar[DT](0.0)
        self._n_diag = 0
        return m^

    def flush_metrics_through_logger[
        L: Logger
    ](
        mut self,
        logger: Optional[Pointer[L, MutAnyOrigin]],
        step: Int,
    ) raises:
        _ = self.flush_metrics[L](logger, step)

    # ── Checkpointing (one v3 file; task table LAST among the tensors) ─
    def save_state(mut self, path: String) raises:
        """Every world-model module, the Q ensemble (online + target), policy
        and termination into ONE v3 storage-ckpt file, then the task table
        (param + its own Adam moments), then as `K` sections the table's
        bias-correction powers, the running Q scale (`pi_scale`) and the step
        counter. Atomic and chunked. Network optimizer moments NOT persisted."""
        comptime tg = Self.target
        var w = BinaryCheckpointWriter(save_moments=False)
        write_model[tg](w, self.encoder, self.ctx, "encoder")
        write_model[tg](w, self.dynamics, self.ctx, "dynamics")
        write_model[tg](w, self.reward, self.ctx, "reward")
        write_model[tg](w, self.policy, self.ctx, "policy")
        write_model[tg](w, self.q0, self.ctx, "q0")
        write_model[tg](w, self.q1, self.ctx, "q1")
        write_model[tg](w, self.q2, self.ctx, "q2")
        write_model[tg](w, self.q3, self.ctx, "q3")
        write_model[tg](w, self.q4, self.ctx, "q4")
        write_model[tg](w, self.qt0, self.ctx, "qt0")
        write_model[tg](w, self.qt1, self.ctx, "qt1")
        write_model[tg](w, self.qt2, self.ctx, "qt2")
        write_model[tg](w, self.qt3, self.ctx, "qt3")
        write_model[tg](w, self.qt4, self.ctx, "qt4")
        write_model[tg](w, self.termination, self.ctx, "termination")
        var sc = CheckpointScalars()
        self.task_emb.write_v3(w, sc, "task_emb")
        sc.set("pi_scale", Float64(self.pol_step.scale.value))
        sc.set_int("step_count", self.step_count)
        w.write_scalars(sc)
        _write_file_bytes(path, w.content)

    def load_state(mut self, path: String) raises:
        """Inverse of `save_state`. Legacy v2 text files (the task table as a
        positional text body, no scalars) still load."""
        comptime tg = Self.target
        var bytes = _read_file_bytes(path)
        if _is_v3_header(bytes):
            var rb = BinaryCheckpointReader(bytes^)
            read_model[tg](rb, self.encoder, self.ctx, "encoder")
            read_model[tg](rb, self.dynamics, self.ctx, "dynamics")
            read_model[tg](rb, self.reward, self.ctx, "reward")
            read_model[tg](rb, self.policy, self.ctx, "policy")
            read_model[tg](rb, self.q0, self.ctx, "q0")
            read_model[tg](rb, self.q1, self.ctx, "q1")
            read_model[tg](rb, self.q2, self.ctx, "q2")
            read_model[tg](rb, self.q3, self.ctx, "q3")
            read_model[tg](rb, self.q4, self.ctx, "q4")
            read_model[tg](rb, self.qt0, self.ctx, "qt0")
            read_model[tg](rb, self.qt1, self.ctx, "qt1")
            read_model[tg](rb, self.qt2, self.ctx, "qt2")
            read_model[tg](rb, self.qt3, self.ctx, "qt3")
            read_model[tg](rb, self.qt4, self.ctx, "qt4")
            read_model[tg](rb, self.termination, self.ctx, "termination")
            self.task_emb.read_v3(rb, "task_emb")
            var sc = rb.read_scalars()
            rb.finish()
            self.task_emb.take_scalars(sc, "task_emb")
            self.pol_step.scale.value = Scalar[DT](
                sc.get("pi_scale", Float64(self.pol_step.scale.value))
            )
            self.step_count = sc.get_int("step_count", self.step_count)
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
        walk_params[tg](self.encoder, r, self.ctx, "encoder")
        walk_params[tg](self.dynamics, r, self.ctx, "dynamics")
        walk_params[tg](self.reward, r, self.ctx, "reward")
        walk_params[tg](self.policy, r, self.ctx, "policy")
        walk_params[tg](self.q0, r, self.ctx, "q0")
        walk_params[tg](self.q1, r, self.ctx, "q1")
        walk_params[tg](self.q2, r, self.ctx, "q2")
        walk_params[tg](self.q3, r, self.ctx, "q3")
        walk_params[tg](self.q4, r, self.ctx, "q4")
        walk_params[tg](self.qt0, r, self.ctx, "qt0")
        walk_params[tg](self.qt1, r, self.ctx, "qt1")
        walk_params[tg](self.qt2, r, self.ctx, "qt2")
        walk_params[tg](self.qt3, r, self.ctx, "qt3")
        walk_params[tg](self.qt4, r, self.ctx, "qt4")
        walk_params[tg](self.termination, r, self.ctx, "termination")
        r.mode = 1
        var _sref16 = ParamVisitorRef.of[type_of(r), tg](r)
        self.encoder.for_each_state[tg](_sref16, self.ctx, "encoder")
        var _sref17 = ParamVisitorRef.of[type_of(r), tg](r)
        self.dynamics.for_each_state[tg](_sref17, self.ctx, "dynamics")
        var _sref18 = ParamVisitorRef.of[type_of(r), tg](r)
        self.reward.for_each_state[tg](_sref18, self.ctx, "reward")
        var _sref19 = ParamVisitorRef.of[type_of(r), tg](r)
        self.policy.for_each_state[tg](_sref19, self.ctx, "policy")
        var _sref20 = ParamVisitorRef.of[type_of(r), tg](r)
        self.q0.for_each_state[tg](_sref20, self.ctx, "q0")
        var _sref21 = ParamVisitorRef.of[type_of(r), tg](r)
        self.q1.for_each_state[tg](_sref21, self.ctx, "q1")
        var _sref22 = ParamVisitorRef.of[type_of(r), tg](r)
        self.q2.for_each_state[tg](_sref22, self.ctx, "q2")
        var _sref23 = ParamVisitorRef.of[type_of(r), tg](r)
        self.q3.for_each_state[tg](_sref23, self.ctx, "q3")
        var _sref24 = ParamVisitorRef.of[type_of(r), tg](r)
        self.q4.for_each_state[tg](_sref24, self.ctx, "q4")
        var _sref25 = ParamVisitorRef.of[type_of(r), tg](r)
        self.qt0.for_each_state[tg](_sref25, self.ctx, "qt0")
        var _sref26 = ParamVisitorRef.of[type_of(r), tg](r)
        self.qt1.for_each_state[tg](_sref26, self.ctx, "qt1")
        var _sref27 = ParamVisitorRef.of[type_of(r), tg](r)
        self.qt2.for_each_state[tg](_sref27, self.ctx, "qt2")
        var _sref28 = ParamVisitorRef.of[type_of(r), tg](r)
        self.qt3.for_each_state[tg](_sref28, self.ctx, "qt3")
        var _sref29 = ParamVisitorRef.of[type_of(r), tg](r)
        self.qt4.for_each_state[tg](_sref29, self.ctx, "qt4")
        var _sref30 = ParamVisitorRef.of[type_of(r), tg](r)
        self.termination.for_each_state[tg](_sref30, self.ctx, "termination")
        # Trailing task-embedding body, read positionally from the reader cursor.
        self.task_emb.load_body(r.lines, r.cur, "task_emb")

    # ── td-target dispatch: thread the random target pair as DISTINCT fields ─
    def _td_dispatch(
        mut self,
        a: Int, b: Int,
        mut task_ids: Tensor,
        mut obs: Tensor, mut rew: Tensor, mut done: Tensor, mut td: Tensor,
        gamma: Scalar[DT],
        ctx: Optional[DeviceContext],
    ) raises:
        comptime tg = Self.target
        comptime for i in range(NQ):
            comptime for j in range(NQ):
                comptime if i < j:
                    if (a == i and b == j) or (a == j and b == i):
                        self.td_step.step[tg](
                            self.encoder, self.policy,
                            self.get_qt[i](), self.get_qt[j](),
                            self.task_emb, task_ids,
                            obs, rew, done, td, gamma, ctx,
                        )

    def _policy_dispatch(
        mut self,
        a: Int, b: Int,
        mut zpol: Tensor,
        mut task_ids: Tensor,
    ) raises -> Scalar[DT]:
        comptime tg = Self.target
        comptime for i in range(NQ):
            comptime for j in range(NQ):
                comptime if i < j:
                    if (a == i and b == j) or (a == j and b == i):
                        return self.pol_step.step[tg](
                            self.policy, self.get_q[i](), self.get_q[j](),
                            self.pi_opt, self.task_emb, zpol, task_ids, self.ctx,
                        )
        return Scalar[DT](0.0)

    def train_step(mut self) raises -> Bool:
        self.step_count += 1
        if not self.replay.can_sample[Self.H]():
            return False
        if self.replay.count() < self.learning_starts:
            return False

        comptime tg = Self.target
        comptime MO = Self.MAX_OBS
        comptime A = Self.MAX_ACT
        comptime LAT = Self.LATENT
        comptime EMB = Self.TASK_EMB
        comptime AOBS = Self.AOBS
        comptime HH = Self.H
        comptime BB = Self.B
        comptime PBP = Self.PB

        # ── sample (b-major) + per-window task ids ──────────────────────
        var ob = List[Scalar[DT]](length=BB * (HH + 1) * MO, fill=0)
        var ab = List[Scalar[DT]](length=BB * HH * A, fill=0)
        var rb = List[Scalar[DT]](length=BB * HH, fill=0)
        var dbf = List[Scalar[DT]](length=BB * HH, fill=0)
        var tk = List[Scalar[DT]](length=BB, fill=0)
        self.replay.sample_batch_task[BB, HH](
            rebind[Pointer[Scalar[DT], MutAnyOrigin]](
                Pointer(to=ob[0])
            ),
            rebind[Pointer[Scalar[DT], MutAnyOrigin]](
                Pointer(to=ab[0])
            ),
            rebind[Pointer[Scalar[DT], MutAnyOrigin]](
                Pointer(to=rb[0])
            ),
            rebind[Pointer[Scalar[DT], MutAnyOrigin]](
                Pointer(to=dbf[0])
            ),
            rebind[Pointer[Scalar[DT], MutAnyOrigin]](
                Pointer(to=tk[0])
            ),
        )

        # ── t-major input Tensors ───────────────────────────────────────
        var ot = Tensor.alloc((HH + 1) * BB * MO)
        var at = Tensor.alloc(HH * BB * A)
        var rt = Tensor.alloc(HH * BB)
        var dt = Tensor.alloc(HH * BB)
        var td = Tensor.alloc(HH * BB)
        var tids = Tensor.alloc(BB)            # [B] per-window task id
        var tids_pb = Tensor.alloc(PBP)        # [PB] per-row task id
        for b in range(BB):
            tids.data[b] = tk[b]
            for t in range(HH + 1):
                for i in range(MO):
                    ot.data[(t * BB + b) * MO + i] = ob[
                        (b * (HH + 1) + t) * MO + i
                    ]
            for t in range(HH):
                for j in range(A):
                    at.data[(t * BB + b) * A + j] = ab[(b * HH + t) * A + j]
                rt.data[t * BB + b] = rb[b * HH + t]
                dt.data[t * BB + b] = dbf[b * HH + t]
        for t in range(HH + 1):
            for b in range(BB):
                tids_pb.data[t * BB + b] = tk[b]

        var ta = Int(random_float64() * Float64(NQ))
        if ta >= NQ:
            ta = NQ - 1
        var tb = (ta + 1) % NQ
        var pa = Int(random_float64() * Float64(NQ))
        if pa >= NQ:
            pa = NQ - 1
        var pb = (pa + 1) % NQ

        var ctx = self.ctx
        comptime if tg == "gpu":
            ot.upload(ctx.value())
            at.upload(ctx.value())
            rt.upload(ctx.value())
            dt.upload(ctx.value())
            td.upload(ctx.value())
            tids.upload(ctx.value())
            tids_pb.upload(ctx.value())

        # bracket the embedding grad accumulation around the whole step.
        self.task_emb.zero_grad[tg]()

        # ── TD targets (stop-grad) ──────────────────────────────────────
        self._td_dispatch(ta, tb, tids, ot, rt, dt, td, self.gamma, ctx)

        # ── WM BPTT (accumulates embedding grad sites 1+2) ──────────────
        var wl = self.wm_step.step[tg](
            self.wm_graph, self.encoder, self.dynamics, self.reward,
            self.q0, self.q1, self.q2, self.q3, self.q4, self.termination,
            self.task_emb,
            self.enc_opt, self.dyn_opt, self.rew_opt,
            self.qo0, self.qo1, self.qo2, self.qo3, self.qo4, self.term_opt,
            tids, ot, at, rt, td, dt, ctx,
        )
        self._last_cons = wl.consistency
        self._last_rew = wl.reward
        self._last_val = wl.value
        self._last_term = wl.termination
        self._last_wm = wl.total()

        # ── policy update on encoded [obs|task_emb] latents (site 3) ────
        var tem_pb = Tensor.make[tg](PBP * EMB, ctx)
        self.task_emb.gather[tg, PBP](tids_pb, tem_pb, ctx)
        comptime if tg == "gpu":
            tem_pb.download(ctx.value())
        var oaug = Tensor.alloc(PBP * AOBS)
        for row in range(PBP):
            for i in range(MO):
                oaug.data[row * AOBS + i] = ot.data[row * MO + i]
            for e in range(EMB):
                oaug.data[row * AOBS + MO + e] = tem_pb.data[row * EMB + e]
        comptime if tg == "gpu":
            oaug.upload(ctx.value())
        var zpol = Tensor.make[tg](PBP * LAT, ctx)
        self.encoder.forward[tg, PBP](TensorRefs[1](oaug), zpol, ctx)
        self._last_pi = self._policy_dispatch(pa, pb, zpol, tids_pb)

        # ── ONE embedding Adam step (sites 1+2+3 accumulated) ───────────
        self.task_emb.step[tg]()

        # ── Polyak (target ← online) ────────────────────────────────────
        self.qt0.polyak_from[tg](self.q0, self.tau, ctx)
        self.qt1.polyak_from[tg](self.q1, self.tau, ctx)
        self.qt2.polyak_from[tg](self.q2, self.tau, ctx)
        self.qt3.polyak_from[tg](self.q3, self.tau, ctx)
        self.qt4.polyak_from[tg](self.q4, self.tau, ctx)

        # ── TD-target stats over the [H*B] targets (host on both paths) ──
        comptime if tg == "gpu":
            td.download(ctx.value())
        var td_sum: Scalar[DT] = 0.0
        var td_mn = td.data[0]
        var td_mx = td.data[0]
        for i in range(HH * BB):
            var v = td.data[i]
            td_sum += v
            if v < td_mn:
                td_mn = v
            if v > td_mx:
                td_mx = v

        self._cons_acc += self._last_cons
        self._rew_acc += self._last_rew
        self._val_acc += self._last_val
        self._term_acc += self._last_term
        self._pi_acc += self._last_pi
        self._q_mean_acc += self.pol_step.q_mean
        self._q_min_last = self.pol_step.q_min
        self._q_max_last = self.pol_step.q_max
        self._td_mean_acc += td_sum / Scalar[DT](HH * BB)
        self._td_min_last = td_mn
        self._td_max_last = td_mx
        self._n_diag += 1
        return True

    # ── batched multi-task drivers ────────────────────────────────────────
    #
    # ⚠ SCOPE: this driver trains ONE task per call — a SEGMENT — and the
    # example loops over tasks. That is genuinely multi-task learning rather
    # than sequential fine-tuning, because the replay keeps every task and
    # `train_step` samples windows uniformly across all of them: the gradient
    # steps taken during task 2's segment still see tasks 0 and 1. What it is
    # NOT is simultaneous collection: at any instant the acting policy is
    # driving one task's envs.
    #
    # The reason is typing, not preference. Each dm_control task is a distinct
    # `Phyics3dEnvConfig` and therefore a distinct env TYPE, so a simultaneous
    # driver would have to be parameterized on one type per task. Segments
    # need one, and the replay does the mixing.
    #
    # ⚠ REQUIRES `E.OBS_DIM == MAX_OBS` and `E.ACT_DIM == MAX_ACT`. The
    # obs/action padding that lets tasks have DIFFERENT dims lives in the agent
    # (and in `action_mask`), but nothing pads an env's slabs yet, so today the
    # tasks must share dims — true for walker's stand/walk/run, which share one
    # model. Heterogeneous bodies need a padding `BatchedEnv` wrapper first.

    def _mt_prepare(
        mut self, task_id: Int, ctx: Optional[DeviceContext]
    ) raises:
        """Point the agent at `task_id` and refresh the acting-side task state.

        ⚠ The embedding is RE-GATHERED every iteration, not once per segment:
        `train_step` updates the task-embedding table, so a row cached at the
        top of a segment would drive the whole segment with an embedding the
        world model has already moved away from.
        """
        comptime tg = Self.target
        comptime A = Self.MAX_ACT
        self.cur_task = task_id
        # The mask is constant for the whole segment (it is a property of the
        # task, not of training), so this uploads once per call rather than
        # per step — unlike the embedding, which the table keeps moving.
        self.act_mask.ensure(A)
        for j in range(A):
            self.act_mask.data[j] = self.action_mask[task_id * A + j]
        comptime if tg == "gpu":
            self.act_mask.upload_resident(ctx.value())

    def _mt_gather_emb[
        N_ENVS: Int
    ](mut self, task_id: Int, ctx: Optional[DeviceContext]) raises:
        """`act_tem[n] = task_emb[task_id]` for all N lanes (one task per
        segment, so every lane gathers the same row — which keeps the concat in
        `batched_acting_mt` a plain elementwise copy)."""
        comptime tg = Self.target
        comptime EMB = Self.TASK_EMB
        self.act_tids.ensure(N_ENVS)
        for e in range(N_ENVS):
            self.act_tids.data[e] = Scalar[DT](task_id)
        comptime if tg == "gpu":
            self.act_tids.upload_resident(ctx.value())
        self.act_tem.ensure[tg](N_ENVS * EMB, ctx)
        self.task_emb.gather[tg, N_ENVS](self.act_tids, self.act_tem, ctx)

    def _mt_gather_row(
        mut self, task_id: Int, ctx: Optional[DeviceContext]
    ) raises:
        """Gather ONE `[TASK_EMB]` embedding row into `mpc_row` — what the
        planner's rollout callback broadcasts over its candidates. Uses its own
        fixed-size tensors (see the `mpc_tids` note)."""
        comptime tg = Self.target
        comptime EMB = Self.TASK_EMB
        self.mpc_tids.ensure(1)
        self.mpc_tids.data[0] = Scalar[DT](task_id)
        comptime if tg == "gpu":
            self.mpc_tids.upload_resident(ctx.value())
        self.mpc_row.ensure[tg](EMB, ctx)
        self.task_emb.gather[tg, 1](self.mpc_tids, self.mpc_row, ctx)

    def evaluate_batched_mt[
        E: BatchedEnv,
        EVAL_ENVS: Int,
        USE_MPC: Bool = False,
    ](
        mut self,
        mut env: E,
        task_id: Int,
        *,
        max_steps: Int = 1_000,
        rng_seed: UInt64 = 12345,
    ) raises -> Scalar[DT]:
        """Deterministic eval of ONE task over `EVAL_ENVS` envs → mean return
        over COMPLETED episodes (0.0 if none finished within `max_steps`).

        `USE_MPC=True` evaluates the MPPI planner (GPU only) instead of the
        policy prior — the two are different controllers and generally score
        differently, so an eval number is only comparable to another taken in
        the SAME mode."""
        comptime tg = Self.target
        comptime MO = Self.MAX_OBS
        comptime A = Self.MAX_ACT
        comptime LAT = Self.LATENT
        comptime EMB = Self.TASK_EMB
        comptime assert E.ENV_TARGET == tg, (
            "evaluate_batched_mt: env target must equal the train target"
        )
        comptime assert E.OBS_DIM == MO and E.ACT_DIM == A, (
            "evaluate_batched_mt: env dims must equal MAX_OBS / MAX_ACT"
        )

        comptime if USE_MPC:
            comptime assert tg == "gpu", "MPC eval is GPU-only"
        var ctx = self.ctx
        self._mt_prepare(task_id, ctx)

        # An EVAL_ENVS-wide planner + callback (the agent's own pair is the
        # N=1 one `select_action_mpc` uses). Built once per eval call, which is
        # periodic — not per step.
        comptime EV_BT = EVAL_ENVS * Self.MPC_BT
        comptime EvPlannerT = MPPIGPUBatched[
            Self.LATENT, Self.MAX_ACT, Self.H, Self.NUM_SAMPLES,
            Self.NUM_PI_TRAJS, Self.NUM_ELITES, Self.NUM_ITERS, EVAL_ENVS,
        ]
        comptime EvCB = TDMPC2RolloutCallbackGPUMT[
            Self.MAX_ACT, Self.LATENT, Self.MLP, Self.BINS, Self.SN,
            Self.VMIN, Self.VMAX, Self.TASK_EMB, NQ, EV_BT, Self.QP,
        ]
        var ev_planner: Optional[EvPlannerT] = None
        var ev_cb: Optional[EvCB] = None
        comptime if USE_MPC:
            var c = ctx.value()
            ev_planner = Optional(EvPlannerT(c))
            ev_cb = Optional(
                EvCB.make(
                    self.dynamics, self.reward, self.policy,
                    self.qt0, self.qt1, self.qt2, self.qt3, self.qt4,
                    self.action_scale, c,
                )
            )
            for e in range(EVAL_ENVS):
                ev_planner.value().start_episode(e)

        env.reset_batch[EVAL_ENVS](ctx=ctx, rng_seed=rng_seed)

        var per_env = List[Scalar[DT]](length=EVAL_ENVS, fill=Scalar[DT](0.0))
        var rew_h = List[Scalar[DT]](length=EVAL_ENVS, fill=Scalar[DT](0.0))
        var done_h = List[Scalar[DT]](length=EVAL_ENVS, fill=Scalar[DT](0.0))
        var returns = List[Scalar[DT]]()

        for s in range(max_steps):
            comptime if USE_MPC:
                # ⚠ Two gathers, and the ORDER matters. `set_task_embedding`
                # broadcasts a ONE-row `[EMB]` tensor across the planning
                # candidates, so `act_tem` must hold a single row at that
                # moment; the encode below then needs it EVAL_ENVS rows wide.
                self._mt_gather_row(task_id, ctx)
                ev_cb.value().set_task_embedding(self.mpc_row, ctx.value())
                self._mt_gather_emb[EVAL_ENVS](task_id, ctx)
                tdmpc2_mt_encode_batched[
                    Self.EncT, tg, EVAL_ENVS, MO, LAT, EMB
                ](
                    self.encoder, self.act_ob, self.act_ein, self.act_z,
                    self.act_tem,
                    LayoutTensor[
                        DT, Layout.row_major(EVAL_ENVS, MO), MutAnyOrigin
                    ](env.obs_ptr()),
                    ctx,
                )
                ev_planner.value().plan_gpu[EvCB](
                    ctx.value(), ev_cb.value(),
                    self.act_z.lt["gpu", Layout.row_major(EVAL_ENVS, LAT)](),
                    LayoutTensor[
                        DT, Layout.row_major(EVAL_ENVS * A), MutAnyOrigin
                    ](env.action_ptr()),
                    gamma=Float64(self.gamma),
                    temperature=Float64(self.temperature),
                    action_scale=Float64(self.action_scale),
                    deterministic=True,
                    rng_base_seed=UInt32(rng_seed) + UInt32(s),
                )
                ctx.value().enqueue_function[
                    mt_apply_mask_kernel[EVAL_ENVS, A]
                ](
                    LayoutTensor[
                        DT, Layout.row_major(EVAL_ENVS, A), MutAnyOrigin
                    ](env.action_ptr()),
                    self.act_mask.lt["gpu", Layout.row_major(A)](),
                    grid_dim=(EVAL_ENVS * A + TPB - 1) // TPB,
                    block_dim=TPB,
                )
            else:
                self._mt_gather_emb[EVAL_ENVS](task_id, ctx)
                tdmpc2_mt_select_action_batched[
                    Self.EncT, Self.PolicyT, tg, EVAL_ENVS, MO, A, LAT, EMB
                ](
                    self.encoder, self.policy, self.act_rsample,
                    self.act_ob, self.act_ein, self.act_z, self.act_pin,
                    self.act_pio, self.act_alp, self.act_tem, self.act_mask,
                    LayoutTensor[
                        DT, Layout.row_major(EVAL_ENVS, MO), MutAnyOrigin
                    ](env.obs_ptr()),
                    LayoutTensor[
                        DT, Layout.row_major(EVAL_ENVS, A), MutAnyOrigin
                    ](env.action_ptr()),
                    self.action_scale, False, ctx,
                )
            env.step_batch[EVAL_ENVS](
                ctx=ctx, rng_seed=rng_seed + UInt64(s + 1)
            )
            # Read done BEFORE selective_reset_batch — it zeroes the slab.
            comptime if E.ENV_TARGET == "cpu":
                var rp = env.reward_ptr()
                var dp = env.done_ptr()
                for e in range(EVAL_ENVS):
                    rew_h[e] = rp[unsafe_offset=e]
                    done_h[e] = dp[unsafe_offset=e]
            else:
                var c = ctx.value()
                var rv = DeviceBuffer[DT](
                    c, env.reward_ptr(), EVAL_ENVS, owning=False
                )
                var dv = DeviceBuffer[DT](
                    c, env.done_ptr(), EVAL_ENVS, owning=False
                )
                c.enqueue_copy(rew_h.unsafe_ptr(), rv)
                c.enqueue_copy(done_h.unsafe_ptr(), dv)
                c.synchronize()
            for e in range(EVAL_ENVS):
                per_env[e] = per_env[e] + rew_h[e]
                if done_h[e] > Scalar[DT](0.5):
                    returns.append(per_env[e])
                    per_env[e] = Scalar[DT](0.0)
            env.selective_reset_batch[EVAL_ENVS](
                ctx=ctx, rng_seed=rng_seed + UInt64(s + 1) * UInt64(7)
            )

        if len(returns) == 0:
            return Scalar[DT](0.0)
        var tot = Scalar[DT](0.0)
        for i in range(len(returns)):
            tot += returns[i]
        return tot / Scalar[DT](len(returns))

    def train_batched_mt[
        E: BatchedEnv,
        N_ENVS: Int = 1,
        L: Logger = NoOpLogger,
        USE_MPC: Bool = False,
        EE: BatchedEnv = E,
        EVAL_ENVS: Int = N_ENVS,
    ](
        mut self,
        mut env: E,
        task_id: Int,
        total_env_steps: Int,
        *,
        rng_seed: UInt64 = 42,
        updates_per_step: Int = 1,
        print_every: Int = 20_000,
        verbose: Bool = True,
        logger: Optional[Pointer[L, MutAnyOrigin]] = None,
        diag_every: Int = 0,
        checkpoint_path: String = "",
        artifacts: Optional[ArtifactSink] = None,
        run_dir: String = String(""),
        checkpoint_every: Int = 0,
        base_step: Int = 0,
        eval_env: Optional[Pointer[EE, MutAnyOrigin]] = None,
        eval_every: Int = 0,
        eval_max_steps: Int = 1_000,
        task_label: String = "",
    ) raises -> Scalar[DT]:
        """Collect ONE task's segment with N envs in lockstep, training on the
        WHOLE replay → best eval return for this task.

        The multi-task sibling of `TDMPC2Agent.train_batched`; see the block
        comment above for what "segment" buys and costs. `total_env_steps`
        counts steps across all envs, so the loop runs `total // N_ENVS`
        iterations. Call once per task, in a loop, on the same agent — nets,
        replay, embedding table and optimizers all persist across calls.

        Every transition is stored with `task_id`, and the strided sampler
        rejects windows that straddle a task change, so the batches this call
        trains on mix tasks without mixing them WITHIN a window.
        """
        comptime tg = Self.target
        comptime env_target = E.ENV_TARGET
        comptime MO = Self.MAX_OBS
        comptime A = Self.MAX_ACT
        comptime LAT = Self.LATENT
        comptime EMB = Self.TASK_EMB

        comptime assert N_ENVS > 0, "N_ENVS must be > 0"
        comptime assert env_target == tg, (
            "train_batched_mt: env target must equal the agent's train target"
        )
        comptime assert E.OBS_DIM == MO and E.ACT_DIM == A, (
            "train_batched_mt: env dims must equal MAX_OBS / MAX_ACT — the"
            " padding wrapper for heterogeneous tasks does not exist yet"
        )
        comptime assert Self.CAP % N_ENVS == 0, (
            "train_batched_mt: replay CAP must be a multiple of N_ENVS"
        )
        if task_id < 0 or task_id >= Self.NUM_TASKS:
            raise Error("train_batched_mt: task_id out of range")

        comptime if USE_MPC:
            comptime assert tg == "gpu", (
                "train_batched_mt[USE_MPC=True] requires target='gpu'"
            )
        var ctx = self.ctx
        # ⚠ Not optional: N envs interleave into one ring, so windows must be
        # walked with stride N or they hop between envs every frame.
        self.replay.set_env_stride(N_ENVS)
        self._mt_prepare(task_id, ctx)

        var iters = total_env_steps // N_ENVS

        # ── MPC: an N_ENVS-wide planner + callback, HOISTED out of the loop ──
        # (`MpcCB.make` allocates nine device tensors; building one per step is
        # the hot-loop allocation footgun.)
        comptime TR_BT = N_ENVS * Self.MPC_BT
        comptime TrPlannerT = MPPIGPUBatched[
            Self.LATENT, Self.MAX_ACT, Self.H, Self.NUM_SAMPLES,
            Self.NUM_PI_TRAJS, Self.NUM_ELITES, Self.NUM_ITERS, N_ENVS,
        ]
        comptime TrCB = TDMPC2RolloutCallbackGPUMT[
            Self.MAX_ACT, Self.LATENT, Self.MLP, Self.BINS, Self.SN,
            Self.VMIN, Self.VMAX, Self.TASK_EMB, NQ, TR_BT, Self.QP,
        ]
        var tr_planner: Optional[TrPlannerT] = None
        var tr_cb: Optional[TrCB] = None
        comptime if USE_MPC:
            var c = ctx.value()
            tr_planner = Optional(TrPlannerT(c))
            tr_cb = Optional(
                TrCB.make(
                    self.dynamics, self.reward, self.policy,
                    self.qt0, self.qt1, self.qt2, self.qt3, self.qt4,
                    self.action_scale, c,
                )
            )
            for e in range(N_ENVS):
                tr_planner.value().start_episode(e)

        var obs_h = List[Scalar[DT]](length=N_ENVS * MO, fill=Scalar[DT](0.0))
        var act_h = List[Scalar[DT]](length=N_ENVS * A, fill=Scalar[DT](0.0))
        var rew_h = List[Scalar[DT]](length=N_ENVS, fill=Scalar[DT](0.0))
        var done_h = List[Scalar[DT]](length=N_ENVS, fill=Scalar[DT](0.0))
        var term_h = List[Scalar[DT]](length=N_ENVS, fill=Scalar[DT](0.0))

        env.reset_batch[N_ENVS](ctx=ctx, rng_seed=rng_seed)
        self._mt_stage_obs[E, N_ENVS](env, obs_h, ctx)
        comptime if env_target != "cpu":
            ctx.value().synchronize()

        var per_env_ret = List[Scalar[DT]](
            length=N_ENVS, fill=Scalar[DT](0.0)
        )
        var window = List[Scalar[DT]](length=100, fill=Scalar[DT](0.0))
        var w_idx = 0
        var w_cnt = 0
        var best: Scalar[DT] = Scalar[DT](-1.0e30)
        var t_start = perf_counter_ns()
        var warm_off: UInt64 = 0
        var warm_seed = rng_seed
        var tag = task_label if task_label else String("task ") + String(task_id)

        for it in range(iters):
            var step = it * N_ENVS
            var gstep = base_step + step

            var act_lt = LayoutTensor[
                DT, Layout.row_major(N_ENVS, A), MutAnyOrigin
            ](env.action_ptr())

            # ── 1. actions ───────────────────────────────────────────────
            # ⚠ Gate on the REPLAY COUNT, not the per-call step counter:
            # these drivers are called once per SEGMENT (a task's turn, or a
            # ladder rung) and `step` restarts at 0 every call, so a step-based
            # gate re-runs the random warmup at the top of every segment,
            # quietly poisoning an agent that is already trained. On the first
            # call the two are the same quantity; `replay.count()` is also
            # exactly what `train_step` itself gates on.
            if self.replay.count() < self.learning_starts:
                warmup_uniform_batched[tg, N_ENVS, A](
                    act_lt, self.action_scale, ctx, warm_seed, warm_off
                )
            else:
                # Re-gather EVERY iteration: the table is being trained.
                comptime if USE_MPC:
                    # One row for the callback's broadcast, then N rows for
                    # the encode — see the note in `evaluate_batched_mt`.
                    self._mt_gather_row(task_id, ctx)
                    tr_cb.value().set_task_embedding(
                        self.mpc_row, ctx.value()
                    )
                    self._mt_gather_emb[N_ENVS](task_id, ctx)
                    tdmpc2_mt_encode_batched[
                        Self.EncT, tg, N_ENVS, MO, LAT, EMB
                    ](
                        self.encoder, self.act_ob, self.act_ein, self.act_z,
                        self.act_tem,
                        LayoutTensor[
                            DT, Layout.row_major(N_ENVS, MO), MutAnyOrigin
                        ](env.obs_ptr()),
                        ctx,
                    )
                    tr_planner.value().plan_gpu[TrCB](
                        ctx.value(), tr_cb.value(),
                        self.act_z.lt["gpu", Layout.row_major(N_ENVS, LAT)](),
                        LayoutTensor[
                            DT, Layout.row_major(N_ENVS * A), MutAnyOrigin
                        ](env.action_ptr()),
                        gamma=Float64(self.gamma),
                        temperature=Float64(self.temperature),
                        action_scale=Float64(self.action_scale),
                        deterministic=False,
                        rng_base_seed=UInt32(rng_seed) + UInt32(it),
                    )
                    ctx.value().enqueue_function[
                        mt_apply_mask_kernel[N_ENVS, A]
                    ](
                        act_lt,
                        self.act_mask.lt["gpu", Layout.row_major(A)](),
                        grid_dim=(N_ENVS * A + TPB - 1) // TPB,
                        block_dim=TPB,
                    )
                else:
                    self._mt_gather_emb[N_ENVS](task_id, ctx)
                    tdmpc2_mt_select_action_batched[
                        Self.EncT, Self.PolicyT, tg, N_ENVS, MO, A, LAT, EMB
                    ](
                        self.encoder, self.policy, self.act_rsample,
                        self.act_ob, self.act_ein, self.act_z, self.act_pin,
                        self.act_pio, self.act_alp, self.act_tem,
                        self.act_mask,
                        LayoutTensor[
                            DT, Layout.row_major(N_ENVS, MO), MutAnyOrigin
                        ](env.obs_ptr()),
                        act_lt,
                        self.action_scale, True, ctx,
                    )

            # ── 2. step ──────────────────────────────────────────────────
            env.step_batch[N_ENVS](ctx=ctx, rng_seed=rng_seed + UInt64(it + 1))

            # ── 3. one D2H (BEFORE selective_reset zeroes `done`) ────────
            comptime if env_target == "cpu":
                var ap = env.action_ptr()
                for k in range(N_ENVS * A):
                    act_h[k] = ap[unsafe_offset=k]
                var rp = env.reward_ptr()
                var dp = env.done_ptr()
                var tp = env.terminated_ptr()
                for e in range(N_ENVS):
                    rew_h[e] = rp[unsafe_offset=e]
                    done_h[e] = dp[unsafe_offset=e]
                    term_h[e] = tp[unsafe_offset=e]
            else:
                var c = ctx.value()
                var av = DeviceBuffer[DT](
                    c, env.action_ptr(), N_ENVS * A, owning=False
                )
                var rv = DeviceBuffer[DT](
                    c, env.reward_ptr(), N_ENVS, owning=False
                )
                var dv = DeviceBuffer[DT](
                    c, env.done_ptr(), N_ENVS, owning=False
                )
                var tv = DeviceBuffer[DT](
                    c, env.terminated_ptr(), N_ENVS, owning=False
                )
                c.enqueue_copy(act_h.unsafe_ptr(), av)
                c.enqueue_copy(rew_h.unsafe_ptr(), rv)
                c.enqueue_copy(done_h.unsafe_ptr(), dv)
                c.enqueue_copy(term_h.unsafe_ptr(), tv)
                c.synchronize()

            # ── 4. record N transitions, lockstep, TAGGED with the task ──
            for e in range(N_ENVS):
                self.replay.record_task(
                    rebind[Pointer[Scalar[DT], MutAnyOrigin]](
                        Pointer(to=obs_h[e * MO])
                    ),
                    rebind[Pointer[Scalar[DT], MutAnyOrigin]](
                        Pointer(to=act_h[e * A])
                    ),
                    rew_h[e],
                    term_h[e],
                    task_id,
                )
                per_env_ret[e] = per_env_ret[e] + rew_h[e]
                if done_h[e] > Scalar[DT](0.5):
                    window[w_idx] = per_env_ret[e]
                    w_idx = (w_idx + 1) % 100
                    if w_cnt < 100:
                        w_cnt += 1
                    per_env_ret[e] = Scalar[DT](0.0)
                    comptime if USE_MPC:
                        tr_planner.value().start_episode(e)

            # ── 5. reset finished lanes, stage the next `s` ──────────────
            env.selective_reset_batch[N_ENVS](
                ctx=ctx, rng_seed=rng_seed + UInt64(it + 1) * UInt64(7)
            )
            self._mt_stage_obs[E, N_ENVS](env, obs_h, ctx)

            # ── 6. updates (over the WHOLE replay — every task) ──────────
            if self.replay.count() >= self.learning_starts:
                for _ in range(updates_per_step):
                    _ = self.train_step()

            # ── 7. logging / checkpoint / eval ──────────────────────────
            var prev = step
            var now = step + N_ENVS
            if diag_every > 0 and _crossed(prev, now, diag_every):
                self.flush_metrics_through_logger[L](logger, gstep)
                if Bool(logger):
                    var lg = logger.value()
                    if w_cnt > 0:
                        var acc: Scalar[DT] = 0.0
                        for k in range(w_cnt):
                            acc += window[k]
                        lg[].log_scalar(
                            String("avg_reward_") + tag,
                            Float64(acc / Scalar[DT](w_cnt)),
                            gstep,
                        )
                    lg[].flush()

            if (
                checkpoint_every > 0
                and checkpoint_path.byte_length() > 0
                and _crossed(prev, now, checkpoint_every)
            ):
                self.save_state(checkpoint_path)
                announce_checkpoint(checkpoint_path, artifacts, run_dir)

            var do_eval = (
                eval_every > 0
                and Bool(eval_env)
                and _crossed(prev, now, eval_every)
            )
            if do_eval:
                var ep = eval_env.value()
                var ret = self.evaluate_batched_mt[EE, EVAL_ENVS, USE_MPC](
                    ep[], task_id,
                    max_steps=eval_max_steps,
                    rng_seed=rng_seed + UInt64(gstep),
                )
                if ret > best:
                    best = ret
                if Bool(logger):
                    var lg = logger.value()
                    lg[].log_scalar(
                        String("eval_return_") + tag, Float64(ret), gstep
                    )
                if verbose:
                    var el = Float64(perf_counter_ns() - t_start) / 1e9
                    print(
                        "  [", tag, "] step", gstep, " eval=", ret,
                        " best=", best, " wm=", self.last_wm_loss(),
                        " pi=", self.last_pi_loss(), " (", el, "s )",
                    )
            elif verbose and print_every > 0 and _crossed(
                prev, now, print_every
            ):
                var el = Float64(perf_counter_ns() - t_start) / 1e9
                var mean_ret: Scalar[DT] = 0.0
                if w_cnt > 0:
                    for k in range(w_cnt):
                        mean_ret += window[k]
                    mean_ret /= Scalar[DT](w_cnt)
                print(
                    "  [", tag, "] step", gstep, " mean_ret(100)=", mean_ret,
                    " wm=", self.last_wm_loss(),
                    " pi=", self.last_pi_loss(), " (", el, "s )",
                )

        if checkpoint_every > 0 and checkpoint_path.byte_length() > 0:
            self.save_state(checkpoint_path)
            announce_checkpoint(checkpoint_path, artifacts, run_dir)
        return best

    def _mt_stage_obs[
        E: BatchedEnv, N_ENVS: Int
    ](
        mut self,
        mut env: E,
        mut obs_h: List[Scalar[DT]],
        ctx: Optional[DeviceContext],
    ) raises:
        """Enqueue the [N, MAX_OBS] obs copy without synchronizing — it lands
        before the next iteration's sync, so the loop pays ONE sync per
        iteration. (Same idiom as the single-task driver.)"""
        comptime MO = Self.MAX_OBS
        comptime if E.ENV_TARGET == "cpu":
            var op = env.obs_ptr()
            for k in range(N_ENVS * MO):
                obs_h[k] = op[unsafe_offset=k]
        else:
            var c = ctx.value()
            var ov = DeviceBuffer[DT](
                c, env.obs_ptr(), N_ENVS * MO, owning=False
            )
            c.enqueue_copy(obs_h.unsafe_ptr(), ov)
