"""TD-MPC2 agent (deep_agents, storage framework, CPU + GPU, MPC-off).

Single `target`-generic struct owning the world model (encoder, dynamics,
reward, Q ensemble online + target, policy) + their optimizers + the WM
ComputeGraph + the training blocks (WMStep BPTT, PolicyStep, TDTargetStep)
+ a SequenceReplay (host). `target` ("cpu"/"gpu") is comptime; `ctx` is
threaded for GPU.

Storage migration (Stage 5): the 5 online Q heads, 5 target Q heads, and 5
Q optimizers are DISTINCT FIELDS (q0..q4 / qt0..qt4 / qo0..qo4; NQ fixed = 5).
Storage threads externals into ONE forward/vjp call (two `mut` subscripts of
one List can't alias). The WM step threads all 5 online Q as distinct args;
the random PAIR steps (policy: online (pa,pb); td: target (ta,tb)) use a
comptime-unrolled guarded dispatch so two DISTINCT fields are threaded.

Acting is MPC-off: `a = π(encode(obs))` (reference `cfg.mpc=False`).

train_step: sample length-T window (host) → transpose to t-major → TD
targets (stop-grad) → WM BPTT → policy update on encoded latents → Polyak.
Replay stays host; the steps upload/download internally via storage Tensors.
"""

from noeira.io.artifact_sink import ArtifactSink
from noeira.deep_agents.training.checkpoint import announce_checkpoint
from noeira.nn.core.param import walk_params, ParamVisitorRef
from std.math import tanh
from std.random import random_float64
from std.time import perf_counter_ns
from layout import Layout, LayoutTensor
from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer

from noeira.nn.constants import DT
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
from noeira.planners.trajectory.mppi import MPPIGPUBatched
from noeira.core.logger import Logger, NoOpLogger
from noeira.core.env_traits import BoxContinuousActionEnv
from noeira.deep_agents.training.batched_env import BatchedEnv
from noeira.deep_agents.training.blocks.action_select import (
    warmup_uniform_batched,
)
from .batched_acting import (
    tdmpc2_select_action_batched,
    tdmpc2_encode_batched,
)
from .callback import TDMPC2RolloutCallbackGPU
from .metrics import TDMPC2Metrics

from .nets import (
    TDMPC2Encoder, TDMPC2Dynamics, TDMPC2Reward, TDMPC2QNet, TDMPC2Policy,
    TDMPC2Termination,
)
from .wm_graph import TDMPC2WMGraph, NQ
from .wm_step import WMStep
from .policy_step import PolicyStep
from .td_target_step import TDTargetStep


@always_inline
def _crossed(prev: Int, now: Int, every: Int) -> Bool:
    """True iff a multiple of `every` lies in `(prev, now]`.

    The batched loop advances the env-step counter by N_ENVS per iteration, so
    `step % every == 0` — the single-env test — fires only when `every` is a
    multiple of N_ENVS and is SILENTLY NEVER TRUE otherwise (e.g.
    `eval_every=10_000` with 32 envs: the counter goes 9984 → 10016 and no
    eval ever runs). Testing for a crossed boundary instead makes every
    cadence work at any N_ENVS."""
    if every <= 0:
        return False
    return (now // every) > (prev // every)


@fieldwise_init
struct TDMPC2Agent[
    target: StaticString,
    OBS: Int,
    ENC: Int,
    ACT: Int,
    LATENT: Int,
    MLP: Int,
    BINS: Int,
    SN: Int,
    VMIN: Int,
    VMAX: Int,
    B: Int,
    H: Int,
    CAP: Int,
    NUM_SAMPLES: Int = 512,
    NUM_PI_TRAJS: Int = 24,
    NUM_ELITES: Int = 64,
    NUM_ITERS: Int = 6,
    QP: Float64 = 0.0,
](Movable & Deinitable):
    comptime EncT = TDMPC2Encoder[Self.OBS, Self.ENC, Self.LATENT, Self.SN]
    comptime DynT = TDMPC2Dynamics[Self.LATENT, Self.ACT, Self.MLP, Self.SN]
    comptime RewT = TDMPC2Reward[Self.LATENT, Self.ACT, Self.MLP, Self.BINS]
    comptime QNetT = TDMPC2QNet[Self.LATENT, Self.ACT, Self.MLP, Self.BINS, Self.QP]
    comptime TermT = TDMPC2Termination[Self.LATENT, Self.ACT, Self.MLP]
    comptime PolicyT = TDMPC2Policy[Self.LATENT, Self.ACT, Self.MLP]
    comptime GraphT = TDMPC2WMGraph[
        Self.LATENT, Self.ACT, Self.MLP, Self.BINS, Self.SN, Self.VMIN,
        Self.VMAX, Self.QP,
    ]
    comptime PB = (Self.H + 1) * Self.B
    comptime WMStepT = WMStep[
        Self.OBS, Self.ENC, Self.ACT, Self.LATENT, Self.MLP, Self.BINS,
        Self.SN, Self.VMIN, Self.VMAX, Self.B, Self.H, Self.QP,
    ]
    comptime PolStepT = PolicyStep[
        Self.LATENT, Self.ACT, Self.MLP, Self.BINS, Self.VMIN, Self.VMAX,
        Self.PB, Self.QP,
    ]
    comptime TDStepT = TDTargetStep[
        Self.OBS, Self.ENC, Self.ACT, Self.LATENT, Self.MLP, Self.BINS,
        Self.SN, Self.VMIN, Self.VMAX, Self.B, Self.H, Self.QP,
    ]
    comptime MPC_BT = Self.NUM_SAMPLES + Self.NUM_PI_TRAJS
    comptime PlannerT = MPPIGPUBatched[
        Self.LATENT, Self.ACT, Self.H, Self.NUM_SAMPLES, Self.NUM_PI_TRAJS,
        Self.NUM_ELITES, Self.NUM_ITERS, 1,
    ]
    comptime MpcCB = TDMPC2RolloutCallbackGPU[
        Self.ACT, Self.LATENT, Self.MLP, Self.BINS, Self.SN, Self.VMIN,
        Self.VMAX, NQ, Self.MPC_BT, Self.QP,
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
    var act_rsample: RSample[Self.ACT]
    var replay: SequenceReplay[Self.OBS, Self.ACT, Self.CAP]

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
    var planner: Optional[Self.PlannerT]
    var temperature: Scalar[DT]

    # ── acting scratch — allocated ONCE, reused every step ────────────────
    # `select_action` / `select_action_mpc` run per ENV-STEP. Building their
    # staging tensors per call meant ~5 (MPC-off) to ~13 (MPC) device/pinned
    # buffer creations per action: at 1 M env-steps that is ~13 M calls into
    # `enqueue_create_buffer`, which is both slow (~5 ms/action measured on
    # Metal) and a known way to hammer the disk on CUDA.
    var act_ob: Tensor
    var act_z: Tensor
    var act_pio: Tensor
    var act_alp: Tensor
    # MPC only: the rollout callback (9 device scratch tensors) + the selected
    # action's device/host landing buffers.
    #
    # ⚠ `MpcCB` stores RAW POINTERS to this agent's own modules, so a cached
    # one is invalidated by MOVING the agent (`var a = TDMPC2[...](...)`, or
    # storing it in a struct field, both of which happen). `_mpc_ready` checks
    # the cached callback still points at OUR `dynamics` and rebuilds if not —
    # the alternative is a use-after-free that reads plausible garbage.
    var mpc_cb: Optional[Self.MpcCB]
    var mpc_out: Optional[DeviceBuffer[DT]]
    var mpc_host: Optional[HostBuffer[DT]]

    # ── train_step scratch — allocated ONCE, reused every gradient step ───
    # The replay sample (b-major host) and its t-major transpose. Previously
    # four Lists + five Tensors per step, each Tensor's `.upload()` creating a
    # fresh device buffer.
    var smp_ob: List[Scalar[DT]]
    var smp_ab: List[Scalar[DT]]
    var smp_rb: List[Scalar[DT]]
    var smp_dbf: List[Scalar[DT]]
    var tr_ot: Tensor
    var tr_at: Tensor
    var tr_rt: Tensor
    var tr_dt: Tensor
    var tr_td: Tensor
    # Policy-update inputs: `tr_zpol` is PB x LATENT — 2 MB at the reference
    # dims, and it was being recreated on every single gradient step.
    var tr_obs_pb: Tensor
    var tr_zpol: Tensor

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

        # Termination head (item B): Zero-init at bce_coef=0 (no RNG draw → other
        # nets bit-identical), Kaiming when episodic. Built last.
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

        var ar = RSample[Self.ACT].make[tg, INIT=Zero](ctx=ctx)
        ar.action_scale = action_scale

        var planner: Optional[Self.PlannerT] = None
        comptime if tg == "gpu":
            planner = Self.PlannerT(ctx.value())

        return Self(
            encoder=enc^, dynamics=dyn^, reward=rew^,
            q0=q0^, q1=q1^, q2=q2^, q3=q3^, q4=q4^,
            qt0=qt0^, qt1=qt1^, qt2=qt2^, qt3=qt3^, qt4=qt4^,
            policy=pol^, termination=term^,
            enc_opt=enc_opt^, dyn_opt=dyn_opt^, rew_opt=rew_opt^,
            qo0=qo0^, qo1=qo1^, qo2=qo2^, qo3=qo3^, qo4=qo4^,
            pi_opt=pi_opt^, term_opt=term_opt^,
            wm_graph=Self.GraphT.make[tg, INIT=Kaiming](ctx=ctx),
            wm_step=Self.WMStepT.make[tg](ctx=ctx, termination_coef=bce_coef),
            pol_step=Self.PolStepT.make[tg](ctx=ctx),
            td_step=Self.TDStepT.make[tg](ctx=ctx),
            act_rsample=ar^,
            replay=SequenceReplay[Self.OBS, Self.ACT, Self.CAP].new(),
            gamma=gamma, tau=tau, bce_coef=bce_coef, action_scale=action_scale,
            learning_starts=learning_starts, step_count=0,
            _last_wm=Scalar[DT](0.0), _last_pi=Scalar[DT](0.0),
            _last_cons=Scalar[DT](0.0), _last_rew=Scalar[DT](0.0),
            _last_val=Scalar[DT](0.0), _last_term=Scalar[DT](0.0),
            _cons_acc=Scalar[DT](0.0),
            _rew_acc=Scalar[DT](0.0), _val_acc=Scalar[DT](0.0),
            _term_acc=Scalar[DT](0.0),
            _pi_acc=Scalar[DT](0.0),
            _q_mean_acc=Scalar[DT](0.0), _q_min_last=Scalar[DT](0.0),
            _q_max_last=Scalar[DT](0.0), _td_mean_acc=Scalar[DT](0.0),
            _td_min_last=Scalar[DT](0.0), _td_max_last=Scalar[DT](0.0),
            _n_diag=0,
            ctx=ctx,
            planner=planner^, temperature=temperature,
            # Acting scratch starts EMPTY and is sized on first use: building
            # it here would allocate MPC buffers for agents that never plan,
            # and the callback in particular must not be built until the agent
            # has stopped moving (see the field comment).
            act_ob=Tensor(), act_z=Tensor(),
            act_pio=Tensor(), act_alp=Tensor(),
            mpc_cb=None, mpc_out=None, mpc_host=None,
            smp_ob=List[Scalar[DT]](), smp_ab=List[Scalar[DT]](),
            smp_rb=List[Scalar[DT]](), smp_dbf=List[Scalar[DT]](),
            tr_ot=Tensor(), tr_at=Tensor(), tr_rt=Tensor(),
            tr_dt=Tensor(), tr_td=Tensor(),
            tr_obs_pb=Tensor(), tr_zpol=Tensor(),
        )

    # ── acting (MPC-off): a = π(encode(obs)) ───────────────────────────
    def select_action(
        mut self,
        ref obs: List[Scalar[DT]],
        mut act_out: List[Scalar[DT]],
        explore: Bool = True,
    ) raises:
        comptime tg = Self.target
        comptime A = Self.ACT
        comptime LAT = Self.LATENT
        var ctx = self.ctx
        # Stage obs into the AGENT-OWNED tensor and refresh the device copy in
        # place. `upload_resident` (not `upload`) is load-bearing: `upload`
        # re-creates `self.dev` on every call, which is exactly the per-step
        # buffer creation this scratch exists to avoid.
        self.act_ob.ensure(Self.OBS)
        for d in range(Self.OBS):
            self.act_ob.data[d] = obs[d]
        comptime if tg == "gpu":
            self.act_ob.upload_resident(ctx.value())
        self.act_z.ensure[tg](LAT, ctx)
        self.encoder.forward[tg, 1](TensorRefs[1](self.act_ob), self.act_z, ctx)
        self.act_pio.ensure[tg](2 * A, ctx)
        self.policy.forward[tg, 1](
            TensorRefs[1](self.act_z), self.act_pio, ctx
        )
        if explore:
            self.act_alp.ensure[tg](A + 1, ctx)
            self.act_rsample.forward[tg, 1](
                TensorRefs[1](self.act_pio), self.act_alp, ctx
            )
            comptime if tg == "gpu":
                self.act_alp.download(ctx.value())
            for j in range(A):
                act_out[j] = self.act_alp.data[j]
        else:
            comptime if tg == "gpu":
                self.act_pio.download(ctx.value())
            for j in range(A):
                act_out[j] = tanh(self.act_pio.data[j]) * self.action_scale

    def select_greedy_action(
        mut self,
        ref obs: List[Scalar[DT]],
        mut act_out: List[Scalar[DT]],
    ) raises:
        self.select_action(obs, act_out, explore=False)

    def mpc_start_episode(mut self) raises:
        comptime if Self.target == "gpu":
            self.planner.value().start_episode(0)

    def select_action_mpc(
        mut self,
        ref obs: List[Scalar[DT]],
        mut act_out: List[Scalar[DT]],
        explore: Bool = True,
        num_iters: Int = 0,
    ) raises:
        """MPC acting: plan in latent space via MPPIGPUBatched (single env).
        GPU only.

        `num_iters` overrides the comptime MPPI iteration budget for THIS call
        (0 = use `NUM_ITERS`). Cost is near-linear in it, and it is the only
        budget knob that can vary at runtime — `NUM_SAMPLES` / `NUM_PI_TRAJS`
        size the planner's device buffers at construction. Use it to trade plan
        quality for frame rate in an interactive viewer."""
        comptime assert Self.target == "gpu", (
            "select_action_mpc requires target='gpu' (CPU MPPI is eval-only)"
        )
        comptime A = Self.ACT
        comptime LAT = Self.LATENT
        var ctx = self.ctx.value()

        self._mpc_ready(ctx)

        self.act_ob.ensure(Self.OBS)
        for d in range(Self.OBS):
            self.act_ob.data[d] = obs[d]
        self.act_ob.upload_resident(ctx)
        self.act_z.ensure_gpu(ctx, LAT)
        self.encoder.forward[Self.target, 1](
            TensorRefs[1](self.act_ob), self.act_z, Optional(ctx)
        )

        var z0_lt = self.act_z.lt["gpu", Layout.row_major(1, LAT)]()
        var out_lt = LayoutTensor[DT, Layout.row_major(1 * A), MutAnyOrigin](
            self.mpc_out.value()
        )
        self.planner.value().plan_gpu[Self.MpcCB](
            ctx, self.mpc_cb.value(), z0_lt, out_lt,
            gamma=Float64(self.gamma),
            temperature=Float64(self.temperature),
            action_scale=Float64(self.action_scale),
            deterministic=not explore,
            num_iters=num_iters,
        )
        ctx.enqueue_copy(self.mpc_host.value(), self.mpc_out.value())
        ctx.synchronize()
        for j in range(A):
            act_out[j] = self.mpc_host.value().unsafe_ptr()[unsafe_offset=j]

    @staticmethod
    def _ensure_list(mut buf: List[Scalar[DT]], n: Int):
        """Lazy-grow a host scratch List to >= n, zero-filled — the `List`
        counterpart of `Tensor.ensure`. Reallocating only on growth is what
        makes the per-step cost zero after the first call; every consumer
        overwrites the whole span before reading it."""
        if len(buf) < n:
            buf = List[Scalar[DT]](length=n, fill=Scalar[DT](0))

    def _mpc_ready(mut self, ctx: DeviceContext) raises:
        """Build the MPC scratch on first use, and REBUILD the callback if this
        agent has moved since.

        `MpcCB` holds raw `MutUntrackedOrigin` pointers to `self.dynamics`,
        `self.reward`, `self.policy` and the five target-Q heads. Those are
        untracked by construction — the compiler will not complain when they go
        stale — so a cached callback that survived a move of the agent would
        plan against freed memory and return plausible-looking actions. The
        guard compares the cached `dyn` pointer with a fresh one: same address
        means the modules have not moved and the whole callback is still valid.
        """
        comptime A = Self.ACT
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
        # These own their memory outright, so a move carries them along.
        if not self.mpc_out:
            self.mpc_out = Optional(ctx.enqueue_create_buffer[DT](A))
            self.mpc_host = Optional(ctx.enqueue_create_host_buffer[DT](A))

    def record(
        mut self,
        ref obs: List[Scalar[DT]],
        ref act: List[Scalar[DT]],
        reward: Scalar[DT],
        done: Scalar[DT],
    ) raises:
        self.replay.record(
            rebind[Pointer[Scalar[DT], MutAnyOrigin]](
                Pointer(to=obs[0])
            ),
            rebind[Pointer[Scalar[DT], MutAnyOrigin]](
                Pointer(to=act[0])
            ),
            reward, done,
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

    # ── Checkpointing (one v3 file) ───────────────────────────────────
    def save_state(mut self, path: String) raises:
        """Save every world-model module + the Q ensemble (online + target) +
        policy + termination into ONE v3 storage-ckpt file, then as `K`
        sections the policy loss's running Q scale (`pi_scale`) and the step
        counter. Atomic and chunked. Optimizer moments are NOT persisted
        (resume re-warms)."""
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
        sc.set("pi_scale", Float64(self.pol_step.scale.value))
        sc.set_int("step_count", self.step_count)
        w.write_scalars(sc)
        _write_file_bytes(path, w.content)

    def load_state(mut self, path: String) raises:
        """Inverse of `save_state` (online + target Q both restored). Legacy
        v2 text files — which carry neither scalar — still load."""
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
            var sc = rb.read_scalars()
            rb.finish()
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

    # ── td-target dispatch: thread the random target pair as DISTINCT fields ─
    def _td_dispatch(
        mut self,
        a: Int, b: Int,
        obs_p: Pointer[Tensor, MutUntrackedOrigin],
        rew_p: Pointer[Tensor, MutUntrackedOrigin],
        done_p: Pointer[Tensor, MutUntrackedOrigin],
        td_p: Pointer[Tensor, MutUntrackedOrigin],
        gamma: Scalar[DT],
        ctx: Optional[DeviceContext],
    ) raises:
        """⚠ The four tensors arrive as POINTERS, not `mut` refs, because the
        caller now passes AGENT-OWNED scratch (`self.tr_ot`, …). A `mut Tensor`
        parameter bound to a field of the same `self` this method already
        borrows mutably is an exclusivity violation — the compiler rejects it.
        Untracked pointers are the sanctioned escape (the same one the rollout
        callback uses for its module handles); they stay valid for the call
        because the caller's `self` outlives it."""
        ref obs = obs_p[]
        ref rew = rew_p[]
        ref done = done_p[]
        ref td = td_p[]
        comptime tg = Self.target
        comptime for i in range(NQ):
            comptime for j in range(NQ):
                comptime if i < j:
                    if (a == i and b == j) or (a == j and b == i):
                        self.td_step.step[tg](
                            self.encoder, self.policy,
                            self.get_qt[i](), self.get_qt[j](),
                            obs, rew, done, td, gamma, ctx,
                        )

    def _policy_dispatch(
        mut self,
        a: Int, b: Int,
        zpol_p: Pointer[Tensor, MutUntrackedOrigin],
    ) raises -> Scalar[DT]:
        """Pointer parameter for the same exclusivity reason as
        `_td_dispatch` — the caller passes `self.tr_zpol`."""
        ref zpol = zpol_p[]
        comptime tg = Self.target
        comptime for i in range(NQ):
            comptime for j in range(NQ):
                comptime if i < j:
                    if (a == i and b == j) or (a == j and b == i):
                        return self.pol_step.step[tg](
                            self.policy, self.get_q[i](), self.get_q[j](),
                            self.pi_opt, zpol, self.ctx,
                        )
        return Scalar[DT](0.0)

    def train_step(mut self) raises -> Bool:
        self.step_count += 1
        if not self.replay.can_sample[Self.H]():
            return False
        if self.replay.count() < self.learning_starts:
            return False

        comptime tg = Self.target
        comptime OBSD = Self.OBS
        comptime ACTD = Self.ACT
        comptime LAT = Self.LATENT
        comptime HH = Self.H
        comptime BB = Self.B

        # ── sample (b-major) + transpose to t-major (host) ─────────────
        # All of this is AGENT-OWNED scratch, sized once. It used to be
        # allocated per train_step: four host Lists plus seven Tensors whose
        # `.upload()` re-created its device buffer every call — ~7 device
        # buffer creations per GRADIENT STEP, `tr_zpol` alone being
        # PB x LATENT (2 MB at the reference dims). See the acting-scratch
        # note on the fields: same footgun, hotter loop.
        #
        # ⚠ Caching is only sound because every one of these is FULLY
        # overwritten before it is read. `ob/ab/rb/dbf` are filled by
        # `sample_batch`, `tr_ot/at/rt/dt` by the transpose below,
        # `tr_obs_pb` by the copy, `tr_zpol` by `encoder.forward`. `tr_td` is
        # the exception — it is written on DEVICE by `_td_dispatch` — so it is
        # explicitly re-zeroed each step, reproducing what `Tensor.alloc` did
        # and keeping this change independent of what that step writes.
        self._ensure_list(self.smp_ob, BB * (HH + 1) * OBSD)
        self._ensure_list(self.smp_ab, BB * HH * ACTD)
        self._ensure_list(self.smp_rb, BB * HH)
        self._ensure_list(self.smp_dbf, BB * HH)
        ref ob = self.smp_ob
        ref ab = self.smp_ab
        ref rb = self.smp_rb
        ref dbf = self.smp_dbf
        self.replay.sample_batch[BB, HH](
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
        )

        # t-major input Tensors (agent-owned; host side sized here).
        self.tr_ot.ensure((HH + 1) * BB * OBSD)
        self.tr_at.ensure(HH * BB * ACTD)
        self.tr_rt.ensure(HH * BB)
        self.tr_dt.ensure(HH * BB)
        self.tr_td.ensure(HH * BB)
        ref ot = self.tr_ot
        ref at = self.tr_at
        ref rt = self.tr_rt
        ref dt = self.tr_dt
        ref td = self.tr_td
        for i in range(HH * BB):
            td.data[i] = Scalar[DT](0.0)   # see the note above
        for b in range(BB):
            for t in range(HH + 1):
                for i in range(OBSD):
                    ot.data[(t * BB + b) * OBSD + i] = ob[
                        (b * (HH + 1) + t) * OBSD + i
                    ]
            for t in range(HH):
                for j in range(ACTD):
                    at.data[(t * BB + b) * ACTD + j] = ab[(b * HH + t) * ACTD + j]
                rt.data[t * BB + b] = rb[b * HH + t]
                dt.data[t * BB + b] = dbf[b * HH + t]

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
            # `upload_resident`, NOT `upload`: the latter re-creates the device
            # buffer on every call, which is the allocation this scratch exists
            # to remove. Same bytes copied either way.
            ot.upload_resident(ctx.value())
            at.upload_resident(ctx.value())
            rt.upload_resident(ctx.value())
            dt.upload_resident(ctx.value())
            td.upload_resident(ctx.value())

        # ── TD targets (stop-grad) ─────────────────────────────────────
        self._td_dispatch(
            ta, tb,
            rebind[Pointer[Tensor, MutUntrackedOrigin]](Pointer(to=ot)),
            rebind[Pointer[Tensor, MutUntrackedOrigin]](Pointer(to=rt)),
            rebind[Pointer[Tensor, MutUntrackedOrigin]](Pointer(to=dt)),
            rebind[Pointer[Tensor, MutUntrackedOrigin]](Pointer(to=td)),
            self.gamma, ctx,
        )

        # ── WM BPTT ─────────────────────────────────────────────────────
        var wl = self.wm_step.step[tg](
            self.wm_graph, self.encoder, self.dynamics, self.reward,
            self.q0, self.q1, self.q2, self.q3, self.q4, self.termination,
            self.enc_opt, self.dyn_opt, self.rew_opt,
            self.qo0, self.qo1, self.qo2, self.qo3, self.qo4, self.term_opt,
            ot, at, rt, td, dt, ctx,
        )
        self._last_cons = wl.consistency
        self._last_rew = wl.reward
        self._last_val = wl.value
        self._last_term = wl.termination
        self._last_wm = wl.total()

        # ── policy update on encoded latents ───────────────────────────
        self.tr_zpol.ensure[tg](Self.PB * LAT, ctx)
        self.tr_obs_pb.ensure(Self.PB * OBSD)
        for i in range(Self.PB * OBSD):
            self.tr_obs_pb.data[i] = ot.data[i]
        comptime if tg == "gpu":
            self.tr_obs_pb.upload_resident(ctx.value())
        self.encoder.forward[tg, Self.PB](
            TensorRefs[1](self.tr_obs_pb), self.tr_zpol, ctx
        )
        self._last_pi = self._policy_dispatch(
            pa, pb,
            rebind[Pointer[Tensor, MutUntrackedOrigin]](
                Pointer(to=self.tr_zpol)
            ),
        )

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

    # ── one-call training / eval drivers ──────────────────────────────────
    # `train` / `evaluate` are the SINGLE-env pair (one `BoxContinuousActionEnv`
    # stepped one action at a time); `train_batched` / `evaluate_batched` below
    # are the N-env pair over the `BatchedEnv` trait. Both internalize the
    # collect → record → train_step loop (+ warmup, periodic eval, logging,
    # checkpoint) so examples don't hand-roll it.

    def evaluate[
        E: BoxContinuousActionEnv,
        USE_MPC: Bool = False,
    ](
        mut self,
        mut env: E,
        *,
        episodes: Int = 2,
        max_steps: Int = 1_000,
    ) raises -> Scalar[DT]:
        """Deterministic eval → mean episode return.

        `USE_MPC=False` (default) acts greedily via `a = π(encode(obs))`;
        `USE_MPC=True` plans with MPPI (`select_action_mpc`, GPU only — a
        comptime assert in that path enforces `target == "gpu"`)."""
        comptime OBSD = Self.OBS
        comptime ACTD = Self.ACT
        var obs_l = List[Scalar[DT]](length=OBSD, fill=Scalar[DT](0.0))
        var act_l = List[Scalar[DT]](length=ACTD, fill=Scalar[DT](0.0))
        var total: Scalar[DT] = 0.0
        for _ep in range(episodes):
            var obs = env.reset_obs_list()
            comptime if USE_MPC:
                self.mpc_start_episode()
            for _s in range(max_steps):
                for d in range(OBSD):
                    obs_l[d] = Scalar[DT](obs[d])
                comptime if USE_MPC:
                    self.select_action_mpc(obs_l, act_l, explore=False)
                else:
                    self.select_greedy_action(obs_l, act_l)
                var env_action = List[Scalar[E.dtype]](capacity=ACTD)
                for j in range(ACTD):
                    env_action.append(Scalar[E.dtype](act_l[j]))
                var r = env.step_continuous_vec[E.dtype](env_action)
                total += Scalar[DT](r[1])
                obs = r[0].copy()
                if r[2]:
                    break
        return total / Scalar[DT](episodes if episodes > 0 else 1)

    def train[
        E: BoxContinuousActionEnv,
        L: Logger = NoOpLogger,
        EE: BoxContinuousActionEnv = E,
        USE_MPC: Bool = False,
    ](
        mut self,
        mut env: E,
        total_timesteps: Int,
        *,
        train_every: Int = 1,
        print_every: Int = 20_000,
        verbose: Bool = True,
        logger: Optional[Pointer[L, MutAnyOrigin]] = None,
        diag_every: Int = 0,
        checkpoint_path: String = "",
        artifacts: Optional[ArtifactSink] = None,
        run_dir: String = String(""),
        checkpoint_every: Int = 0,
        eval_env: Optional[Pointer[EE, MutAnyOrigin]] = None,
        eval_every: Int = 0,
        eval_episodes: Int = 2,
        eval_max_steps: Int = 1_000,
    ) raises -> Scalar[DT]:
        """Single-env TD-MPC2 training driver → best eval return.

        One env step + (after warmup) one `train_step` per iteration:
          * `step < learning_starts` → uniform random actions in [-1, 1]
            (the `learning_starts` passed at construction);
          * else → `USE_MPC ? select_action_mpc : select_action` (explore).

        Bootstrapping: records `done = was_terminated()` (NATURAL termination
        only) so the value bootstrap continues across truncation and drops on
        a real terminal — truncation-only envs (e.g. HalfCheetah with
        `TERMINATE_ON_UNHEALTHY=False`) record `done = 0` throughout.

        Optional streams (all off by default):
          * `logger` + `diag_every > 0` → drain the full TD-MPC2 metric bundle
            (consistency/reward/value/wm/policy losses, q/td stats) every
            `diag_every` env-steps, plus an `avg_reward` training signal;
          * `checkpoint_every > 0` + `checkpoint_path` → `save_state` cadence
            (+ once at the end);
          * `eval_env` (ISOLATED env ptr) + `eval_every > 0` → periodic
            DETERMINISTIC eval logged as `eval/mean_return` (the deployable
            signal; pass `USE_MPC=True` to eval the planner)."""
        comptime OBSD = Self.OBS
        comptime ACTD = Self.ACT

        # Single-stream collection. Raises if this agent's replay already holds
        # frames laid down by `train_batched` — those are interleaved N ways,
        # and continuing to fill them one-env-at-a-time would make both halves
        # of the ring unreadable. No-op for the normal (fresh, or repeatedly
        # single-env) case.
        self.replay.set_env_stride(1)

        var obs_l = List[Scalar[DT]](length=OBSD, fill=Scalar[DT](0.0))
        var act_l = List[Scalar[DT]](length=ACTD, fill=Scalar[DT](0.0))
        var obs = env.reset_obs_list()

        # Ring buffer of the last 100 completed-episode returns (for the
        # `avg_reward` stream + progress prints). Avoids List slicing/pop.
        var window = List[Scalar[DT]](length=100, fill=Scalar[DT](0.0))
        var w_idx = 0
        var w_cnt = 0
        var cur_ret: Scalar[DT] = 0.0
        var best: Scalar[DT] = Scalar[DT](-1.0e30)
        var t_start = perf_counter_ns()

        comptime if USE_MPC:
            self.mpc_start_episode()

        for step in range(total_timesteps):
            for d in range(OBSD):
                obs_l[d] = Scalar[DT](obs[d])

            if self.replay.count() < self.learning_starts:
                for j in range(ACTD):
                    act_l[j] = Scalar[DT](random_float64() * 2.0 - 1.0)
            else:
                comptime if USE_MPC:
                    self.select_action_mpc(obs_l, act_l, explore=True)
                else:
                    self.select_action(obs_l, act_l, explore=True)

            var env_action = List[Scalar[E.dtype]](capacity=ACTD)
            for j in range(ACTD):
                env_action.append(Scalar[E.dtype](act_l[j]))
            var res = env.step_continuous_vec[E.dtype](env_action)
            var reward = Scalar[DT](res[1])
            var done = res[2]
            # Replay stores NATURAL termination only (truncation keeps the
            # bootstrap). `was_terminated()` returns terminated-not-truncated.
            var term: Scalar[DT] = 1.0 if env.was_terminated() else 0.0
            self.record(obs_l, act_l, reward, term)
            cur_ret += reward
            obs = res[0].copy()

            if done:
                obs = env.reset_obs_list()
                window[w_idx] = cur_ret
                w_idx = (w_idx + 1) % 100
                if w_cnt < 100:
                    w_cnt += 1
                cur_ret = 0.0
                comptime if USE_MPC:
                    self.mpc_start_episode()

            if (
                self.replay.count() >= self.learning_starts
                and step % train_every == 0
            ):
                _ = self.train_step()

            if diag_every > 0 and step > 0 and step % diag_every == 0:
                self.flush_metrics_through_logger[L](logger, step)
                if Bool(logger):
                    var lg = logger.value()
                    if w_cnt > 0:
                        var s: Scalar[DT] = 0.0
                        for k in range(w_cnt):
                            s += window[k]
                        lg[].log_scalar(
                            "avg_reward",
                            Float64(s / Scalar[DT](w_cnt)),
                            step,
                        )
                    lg[].flush()

            if (
                checkpoint_every > 0
                and step > 0
                and step % checkpoint_every == 0
                and checkpoint_path.byte_length() > 0
            ):
                self.save_state(checkpoint_path)
                announce_checkpoint(checkpoint_path, artifacts, run_dir)

            var do_eval = (
                eval_every > 0 and step > 0 and step % eval_every == 0
                and Bool(eval_env)
            )
            if do_eval:
                var ep = eval_env.value()
                var ret = self.evaluate[EE, USE_MPC](
                    ep[], episodes=eval_episodes, max_steps=eval_max_steps
                )
                if ret > best:
                    best = ret
                if Bool(logger):
                    var lg = logger.value()
                    lg[].log_scalar("eval_return", Float64(ret), step)
                    lg[].log_scalar("eval_return_best", Float64(best), step)
                if verbose:
                    var elapsed = (
                        Float64(perf_counter_ns() - t_start) / 1e9
                    )
                    print(
                        "  step", step, " eval_return=", ret, " best=", best,
                        " wm=", self.last_wm_loss(),
                        " pi=", self.last_pi_loss(),
                        " (", elapsed, "s )",
                    )
            elif verbose and print_every > 0 and step > 0 and (
                step % print_every == 0
            ):
                var elapsed = Float64(perf_counter_ns() - t_start) / 1e9
                var mean_ret: Scalar[DT] = 0.0
                if w_cnt > 0:
                    for k in range(w_cnt):
                        mean_ret += window[k]
                    mean_ret /= Scalar[DT](w_cnt)
                print(
                    "  step", step, " mean_ret(100)=", mean_ret,
                    " wm=", self.last_wm_loss(),
                    " pi=", self.last_pi_loss(), " (", elapsed, "s )",
                )

        if checkpoint_every > 0 and checkpoint_path.byte_length() > 0:
            self.save_state(checkpoint_path)
            announce_checkpoint(checkpoint_path, artifacts, run_dir)
        return best

    # ── batched drivers (N_ENVS envs stepped in lockstep) ──────────────────

    def evaluate_batched[
        E: BatchedEnv,
        EVAL_ENVS: Int,
        USE_MPC: Bool = False,
    ](
        mut self,
        mut env: E,
        *,
        max_steps: Int = 1_000,
        rng_seed: UInt64 = 12345,
    ) raises -> Scalar[DT]:
        """Deterministic eval over `EVAL_ENVS` envs in lockstep → mean return.

        The batched counterpart of `evaluate`. Runs `max_steps` iterations
        (i.e. one episode per env for a fixed-horizon env) collecting every
        COMPLETED episode return; envs that never finish contribute nothing,
        so a `max_steps` shorter than the episode returns 0.0 rather than a
        truncated return.

        Greedy: `a = tanh(π(encode(obs)).mean)·scale` — no rsample noise — or
        the MPPI plan with `deterministic=True` when `USE_MPC`."""
        comptime tg = Self.target
        comptime env_target = E.ENV_TARGET
        comptime OBSD = Self.OBS
        comptime ACTD = Self.ACT
        comptime LAT = Self.LATENT
        comptime assert env_target == tg, (
            "evaluate_batched: env target must equal the agent's train target"
        )
        comptime assert E.OBS_DIM == OBSD and E.ACT_DIM == ACTD, (
            "evaluate_batched: env dims must match the agent's"
        )
        comptime if USE_MPC:
            comptime assert tg == "gpu", "MPC eval is GPU-only"

        var ctx = self.ctx
        var ob_scr = Tensor()
        var z_scr = Tensor()
        var pio_scr = Tensor()
        var alp_scr = Tensor()

        # A planner for EXACTLY this env count (the agent's own planner is
        # N_ENVS=1). Built per eval call — eval is periodic, not hot.
        comptime EV_BT = EVAL_ENVS * Self.MPC_BT
        comptime EvPlannerT = MPPIGPUBatched[
            Self.LATENT, Self.ACT, Self.H, Self.NUM_SAMPLES,
            Self.NUM_PI_TRAJS, Self.NUM_ELITES, Self.NUM_ITERS, EVAL_ENVS,
        ]
        comptime EvCB = TDMPC2RolloutCallbackGPU[
            Self.ACT, Self.LATENT, Self.MLP, Self.BINS, Self.SN, Self.VMIN,
            Self.VMAX, NQ, EV_BT, Self.QP,
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

        var per_env = List[Scalar[DT]](
            length=EVAL_ENVS, fill=Scalar[DT](0.0)
        )
        var rew_h = List[Scalar[DT]](length=EVAL_ENVS, fill=Scalar[DT](0.0))
        var done_h = List[Scalar[DT]](length=EVAL_ENVS, fill=Scalar[DT](0.0))
        var returns = List[Scalar[DT]]()

        for s in range(max_steps):
            var obs_lt = LayoutTensor[
                DT, Layout.row_major(EVAL_ENVS, OBSD), MutAnyOrigin
            ](env.obs_ptr())
            var act_lt = LayoutTensor[
                DT, Layout.row_major(EVAL_ENVS, ACTD), MutAnyOrigin
            ](env.action_ptr())

            comptime if USE_MPC:
                tdmpc2_encode_batched[
                    Self.EncT, tg, EVAL_ENVS, OBSD, LAT
                ](self.encoder, ob_scr, z_scr, obs_lt, ctx)
                ev_planner.value().plan_gpu[EvCB](
                    ctx.value(),
                    ev_cb.value(),
                    z_scr.lt["gpu", Layout.row_major(EVAL_ENVS, LAT)](),
                    LayoutTensor[
                        DT, Layout.row_major(EVAL_ENVS * ACTD), MutAnyOrigin
                    ](env.action_ptr()),
                    gamma=Float64(self.gamma),
                    temperature=Float64(self.temperature),
                    action_scale=Float64(self.action_scale),
                    deterministic=True,
                    rng_base_seed=UInt32(rng_seed) + UInt32(s),
                )
            else:
                tdmpc2_select_action_batched[
                    Self.EncT, Self.PolicyT, tg, EVAL_ENVS, OBSD, ACTD, LAT
                ](
                    self.encoder, self.policy, self.act_rsample,
                    ob_scr, z_scr, pio_scr, alp_scr,
                    obs_lt, act_lt, self.action_scale, False, ctx,
                )

            env.step_batch[EVAL_ENVS](
                ctx=ctx, rng_seed=rng_seed + UInt64(s + 1)
            )

            # Reward/done must be read BEFORE selective_reset_batch — that
            # call ZEROES the done slab as it resets the finished lanes.
            comptime if env_target == "cpu":
                var rp = env.reward_ptr()
                var dp = env.done_ptr()
                for e in range(EVAL_ENVS):
                    rew_h[e] = rp[unsafe_offset=e]
                    done_h[e] = dp[unsafe_offset=e]
            else:
                var c = ctx.value()
                var rew_view = DeviceBuffer[DT](
                    c, env.reward_ptr(), EVAL_ENVS, owning=False
                )
                var done_view = DeviceBuffer[DT](
                    c, env.done_ptr(), EVAL_ENVS, owning=False
                )
                c.enqueue_copy(rew_h.unsafe_ptr(), rew_view)
                c.enqueue_copy(done_h.unsafe_ptr(), done_view)
                c.synchronize()

            for e in range(EVAL_ENVS):
                per_env[e] = per_env[e] + rew_h[e]
                if done_h[e] > Scalar[DT](0.5):
                    returns.append(per_env[e])
                    per_env[e] = Scalar[DT](0.0)
                    comptime if USE_MPC:
                        ev_planner.value().start_episode(e)

            env.selective_reset_batch[EVAL_ENVS](
                ctx=ctx, rng_seed=rng_seed + UInt64(s + 1) * UInt64(7)
            )

        if len(returns) == 0:
            return Scalar[DT](0.0)
        var tot = Scalar[DT](0.0)
        for i in range(len(returns)):
            tot += returns[i]
        return tot / Scalar[DT](len(returns))

    def train_batched[
        E: BatchedEnv,
        N_ENVS: Int = 1,
        L: Logger = NoOpLogger,
        USE_MPC: Bool = False,
        EE: BatchedEnv = E,
        EVAL_ENVS: Int = N_ENVS,
    ](
        mut self,
        mut env: E,
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
    ) raises -> Scalar[DT]:
        """Multi-env TD-MPC2 training driver → best eval return.

        The batched sibling of `train`. `total_env_steps` counts env-steps
        ACROSS ALL ENVS (SAC's convention), so the loop runs
        `total_env_steps // N_ENVS` iterations and each iteration advances
        every env by one step. `learning_starts`, `print_every`,
        `checkpoint_every` and `eval_every` are all in the same all-env unit.

        What is genuinely batched, and what is not:
          * env stepping — one `step_batch` for all N (the GPU physics win);
          * acting — ONE `encoder → policy → rsample` pass over [N, ·], or
            ONE `plan_gpu` that runs N × (NUM_SAMPLES + NUM_PI_TRAJS) MPPI
            candidates in the same kernel grid;
          * `train_step` — UNCHANGED. It samples B windows from replay, which
            never depended on how the data was collected. `updates_per_step`
            train steps run per ITERATION, so `updates_per_step=N_ENVS`
            reproduces the single-env ratio of one update per env-step.

        ⚠ Replay interleaving: N envs write into one ring round-robin, so a
        contiguous window would hop between envs every frame. The driver sets
        `replay.set_env_stride(N_ENVS)`, which makes the sampler walk lanes of
        stride N — windows are then one env's real trajectory. That call is
        NOT optional and there is no runtime signal if it is missing: the loss
        still falls, on dynamics that never happened.

        MPC (`USE_MPC=True`, GPU only) builds its own `MPPIGPUBatched` sized
        for N_ENVS (the agent's own `planner` field is the N=1 one used by
        `select_action_mpc`) and a matching rollout callback, both hoisted out
        of the loop — `MpcCB.make` allocates device scratch, so making one per
        step is the `enqueue_create_buffer`-in-a-hot-loop footgun.

        Bootstrapping matches `train`: the replay stores `terminated`
        (NATURAL termination), not `done`, so the value bootstrap survives
        time-limit truncation.
        """
        comptime tg = Self.target
        comptime env_target = E.ENV_TARGET
        comptime OBSD = Self.OBS
        comptime ACTD = Self.ACT
        comptime LAT = Self.LATENT

        comptime assert N_ENVS > 0, "N_ENVS must be > 0"
        comptime assert env_target == tg, (
            "train_batched: env target must equal the agent's train target"
            " (cross-target batched collection would D2H every obs)"
        )
        comptime assert E.OBS_DIM == OBSD and E.ACT_DIM == ACTD, (
            "train_batched: env dims must match the agent's"
        )
        comptime if USE_MPC:
            comptime assert tg == "gpu", (
                "train_batched[USE_MPC=True] requires target='gpu'"
            )
        # The strided sampler needs the ring wrap to preserve lane identity.
        comptime assert Self.CAP % N_ENVS == 0, (
            "train_batched: replay CAP must be a multiple of N_ENVS"
        )

        var ctx = self.ctx
        # ⚠ Not optional — see the docstring. Without it every training window
        # is a round-robin of N different envs.
        self.replay.set_env_stride(N_ENVS)

        var iters = total_env_steps // N_ENVS

        # ── acting scratch (allocated ONCE) ──────────────────────────────
        var ob_scr = Tensor()
        var z_scr = Tensor()
        var pio_scr = Tensor()
        var alp_scr = Tensor()

        # ── MPC: N_ENVS-wide planner + its rollout callback, hoisted ─────
        comptime TR_BT = N_ENVS * Self.MPC_BT
        comptime TrPlannerT = MPPIGPUBatched[
            Self.LATENT, Self.ACT, Self.H, Self.NUM_SAMPLES,
            Self.NUM_PI_TRAJS, Self.NUM_ELITES, Self.NUM_ITERS, N_ENVS,
        ]
        comptime TrCB = TDMPC2RolloutCallbackGPU[
            Self.ACT, Self.LATENT, Self.MLP, Self.BINS, Self.SN, Self.VMIN,
            Self.VMAX, NQ, TR_BT, Self.QP,
        ]
        var planner: Optional[TrPlannerT] = None
        var cb: Optional[TrCB] = None
        comptime if USE_MPC:
            var c = ctx.value()
            planner = Optional(TrPlannerT(c))
            cb = Optional(
                TrCB.make(
                    self.dynamics, self.reward, self.policy,
                    self.qt0, self.qt1, self.qt2, self.qt3, self.qt4,
                    self.action_scale, c,
                )
            )
            for e in range(N_ENVS):
                planner.value().start_episode(e)

        # ── host mirrors (the replay is a host ring) ─────────────────────
        # `obs_h` holds the PREVIOUS iteration's post-reset observation — the
        # `s` of the transition recorded after the next step.
        var obs_h = List[Scalar[DT]](
            length=N_ENVS * OBSD, fill=Scalar[DT](0.0)
        )
        var act_h = List[Scalar[DT]](
            length=N_ENVS * ACTD, fill=Scalar[DT](0.0)
        )
        var rew_h = List[Scalar[DT]](length=N_ENVS, fill=Scalar[DT](0.0))
        var done_h = List[Scalar[DT]](length=N_ENVS, fill=Scalar[DT](0.0))
        var term_h = List[Scalar[DT]](length=N_ENVS, fill=Scalar[DT](0.0))

        env.reset_batch[N_ENVS](ctx=ctx, rng_seed=rng_seed)
        self._dl_obs[E, N_ENVS](env, obs_h, ctx)

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

        for it in range(iters):
            var step = it * N_ENVS          # env-steps done BEFORE this iter
            var gstep = base_step + step    # cumulative, for logging

            var obs_lt = LayoutTensor[
                DT, Layout.row_major(N_ENVS, OBSD), MutAnyOrigin
            ](env.obs_ptr())
            var act_lt = LayoutTensor[
                DT, Layout.row_major(N_ENVS, ACTD), MutAnyOrigin
            ](env.action_ptr())

            # ── 1. actions → env.action_ptr() ────────────────────────────
            # ⚠ Gate on the REPLAY COUNT, not the per-call step counter:
            # these drivers are called once per SEGMENT (a task's turn, or a
            # ladder rung) and `step` restarts at 0 every call, so a step-based
            # gate re-runs the random warmup at the top of every segment,
            # quietly poisoning an agent that is already trained. On the first
            # call the two are the same quantity; `replay.count()` is also
            # exactly what `train_step` itself gates on.
            if self.replay.count() < self.learning_starts:
                warmup_uniform_batched[tg, N_ENVS, ACTD](
                    act_lt, self.action_scale, ctx, warm_seed, warm_off
                )
            else:
                comptime if USE_MPC:
                    tdmpc2_encode_batched[
                        Self.EncT, tg, N_ENVS, OBSD, LAT
                    ](self.encoder, ob_scr, z_scr, obs_lt, ctx)
                    planner.value().plan_gpu[TrCB](
                        ctx.value(),
                        cb.value(),
                        z_scr.lt["gpu", Layout.row_major(N_ENVS, LAT)](),
                        LayoutTensor[
                            DT, Layout.row_major(N_ENVS * ACTD), MutAnyOrigin
                        ](env.action_ptr()),
                        gamma=Float64(self.gamma),
                        temperature=Float64(self.temperature),
                        action_scale=Float64(self.action_scale),
                        deterministic=False,
                        rng_base_seed=UInt32(rng_seed) + UInt32(it),
                    )
                else:
                    tdmpc2_select_action_batched[
                        Self.EncT, Self.PolicyT, tg, N_ENVS, OBSD, ACTD, LAT
                    ](
                        self.encoder, self.policy, self.act_rsample,
                        ob_scr, z_scr, pio_scr, alp_scr,
                        obs_lt, act_lt, self.action_scale, True, ctx,
                    )

            # ── 2. step every env ────────────────────────────────────────
            env.step_batch[N_ENVS](
                ctx=ctx, rng_seed=rng_seed + UInt64(it + 1)
            )

            # ── 3. one D2H for the whole batch ───────────────────────────
            # ⚠ BEFORE selective_reset_batch: that call zeroes the done slab.
            comptime if env_target == "cpu":
                var ap = env.action_ptr()
                for k in range(N_ENVS * ACTD):
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
                var act_view = DeviceBuffer[DT](
                    c, env.action_ptr(), N_ENVS * ACTD, owning=False
                )
                var rew_view = DeviceBuffer[DT](
                    c, env.reward_ptr(), N_ENVS, owning=False
                )
                var done_view = DeviceBuffer[DT](
                    c, env.done_ptr(), N_ENVS, owning=False
                )
                var term_view = DeviceBuffer[DT](
                    c, env.terminated_ptr(), N_ENVS, owning=False
                )
                c.enqueue_copy(act_h.unsafe_ptr(), act_view)
                c.enqueue_copy(rew_h.unsafe_ptr(), rew_view)
                c.enqueue_copy(done_h.unsafe_ptr(), done_view)
                c.enqueue_copy(term_h.unsafe_ptr(), term_view)
                # Also drains the obs copy enqueued at the END of the previous
                # iteration, so `obs_h` is the post-reset obs of THIS step's
                # `s`. One sync per iteration for N_ENVS env-steps.
                c.synchronize()

            # ── 4. record N transitions, lockstep (env 0 … N-1) ──────────
            # This order IS the layout the strided sampler assumes.
            for e in range(N_ENVS):
                self.replay.record(
                    rebind[Pointer[Scalar[DT], MutAnyOrigin]](
                        Pointer(to=obs_h[e * OBSD])
                    ),
                    rebind[Pointer[Scalar[DT], MutAnyOrigin]](
                        Pointer(to=act_h[e * ACTD])
                    ),
                    rew_h[e],
                    term_h[e],
                )
                per_env_ret[e] = per_env_ret[e] + rew_h[e]
                if done_h[e] > Scalar[DT](0.5):
                    window[w_idx] = per_env_ret[e]
                    w_idx = (w_idx + 1) % 100
                    if w_cnt < 100:
                        w_cnt += 1
                    per_env_ret[e] = Scalar[DT](0.0)
                    comptime if USE_MPC:
                        planner.value().start_episode(e)

            # ── 5. reset the finished lanes, stage next `s` ──────────────
            env.selective_reset_batch[N_ENVS](
                ctx=ctx, rng_seed=rng_seed + UInt64(it + 1) * UInt64(7)
            )
            self._stage_obs[E, N_ENVS](env, obs_h, ctx)

            # ── 6. updates ───────────────────────────────────────────────
            if self.replay.count() >= self.learning_starts:
                for _ in range(updates_per_step):
                    _ = self.train_step()

            # ── 7. logging / checkpoint / eval (all in all-env steps) ────
            var prev = step
            var now = step + N_ENVS
            if diag_every > 0 and _crossed(prev, now, diag_every):
                self.flush_metrics_through_logger[L](logger, gstep)
                if Bool(logger):
                    var lg = logger.value()
                    if w_cnt > 0:
                        var s: Scalar[DT] = 0.0
                        for k in range(w_cnt):
                            s += window[k]
                        lg[].log_scalar(
                            "avg_reward", Float64(s / Scalar[DT](w_cnt)), gstep
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
                var ret = self.evaluate_batched[EE, EVAL_ENVS, USE_MPC](
                    ep[],
                    max_steps=eval_max_steps,
                    rng_seed=rng_seed + UInt64(gstep),
                )
                if ret > best:
                    best = ret
                if Bool(logger):
                    var lg = logger.value()
                    lg[].log_scalar("eval_return", Float64(ret), gstep)
                    lg[].log_scalar("eval_return_best", Float64(best), gstep)
                if verbose:
                    var elapsed = Float64(perf_counter_ns() - t_start) / 1e9
                    print(
                        "  step", gstep, " eval_return=", ret, " best=", best,
                        " wm=", self.last_wm_loss(),
                        " pi=", self.last_pi_loss(),
                        " (", elapsed, "s )",
                    )
            elif verbose and print_every > 0 and _crossed(
                prev, now, print_every
            ):
                var elapsed = Float64(perf_counter_ns() - t_start) / 1e9
                var mean_ret: Scalar[DT] = 0.0
                if w_cnt > 0:
                    for k in range(w_cnt):
                        mean_ret += window[k]
                    mean_ret /= Scalar[DT](w_cnt)
                print(
                    "  step", gstep, " mean_ret(100)=", mean_ret,
                    " wm=", self.last_wm_loss(),
                    " pi=", self.last_pi_loss(), " (", elapsed, "s )",
                )

        if checkpoint_every > 0 and checkpoint_path.byte_length() > 0:
            self.save_state(checkpoint_path)
            announce_checkpoint(checkpoint_path, artifacts, run_dir)
        return best

    def _stage_obs[
        E: BatchedEnv, N_ENVS: Int
    ](
        mut self,
        mut env: E,
        mut obs_h: List[Scalar[DT]],
        ctx: Optional[DeviceContext],
    ) raises:
        """ENQUEUE the [N_ENVS, OBS] obs copy without synchronizing.

        The copy lands before the next iteration's `synchronize`, and the host
        does not read `obs_h` again until after it — so the batched loop pays
        ONE sync per iteration instead of two. On the CPU target it is a plain
        copy (the env slab is host memory and is about to be overwritten by
        the next `step_batch`, so a snapshot is required, not a pointer)."""
        comptime OBSD = Self.OBS
        comptime if E.ENV_TARGET == "cpu":
            var op = env.obs_ptr()
            for k in range(N_ENVS * OBSD):
                obs_h[k] = op[unsafe_offset=k]
        else:
            var c = ctx.value()
            var obs_view = DeviceBuffer[DT](
                c, env.obs_ptr(), N_ENVS * OBSD, owning=False
            )
            c.enqueue_copy(obs_h.unsafe_ptr(), obs_view)

    def _dl_obs[
        E: BatchedEnv, N_ENVS: Int
    ](
        mut self,
        mut env: E,
        mut obs_h: List[Scalar[DT]],
        ctx: Optional[DeviceContext],
    ) raises:
        """`_stage_obs` + a synchronize — the initial post-reset fill, where
        there is no later sync to piggyback on."""
        self._stage_obs[E, N_ENVS](env, obs_h, ctx)
        comptime if E.ENV_TARGET != "cpu":
            ctx.value().synchronize()
