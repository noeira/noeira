"""`FBCPROnlineAgent` — FB-CPR ONLINE: the reference's own setting.

`docs/BFM_ZERO_SHOT_RL.md` §18.9.1: offline, the style critic `Q_D` sat at
`r/(1−γ)` with `Q_D(s, π)` equal to `Q_D(s, a_data)` to four figures — a
critic of a dataset row carries no action information. §16.2, verbatim
from the paper: "while FB algorithm is offline, FB-CPR is trained fully
online and off-policy". So this agent is the composition §18.3 B2 needs:

    FBOnlineAgent   the rollout: lanes, per-lane z held `z_hold` steps and
                    resampled from sphere ∪ ZBuffer, the device replay ring,
                    and the unchanged `FBTrainer` (A3)
    FBCPRHead       D(s, z) + twin Q_D + the style term, around that trainer
                    (A4), with D's NEGATIVES = the ring's rows under the z
                    they were rolled out under, and its POSITIVES = windows
                    of the expert store with their own window encoding

Per training step (`_train_kernels`, capturable):

    1  sample the ring into `t.bs/ba/bsn/bsp/bz`     (`bz` = the STORED z)
    2  z_neg ← bz                                    (what D's negatives carry)
    3  draw NW expert windows: starts from the valid-start table,
       expanded to SEQ rows each → `head.es` / `head.esn`
    4  head.encode_expert(t)                         (ez = project mean B(s'))
    5  relabel bz: keep `keep_frac`, else the goal / expert / uniform
       mixture (`z_relabel3_kernel`), project
    6  head.step(t)   = D update on (es, ez) vs (bs, z_neg); r_D on
                        (bs, bz); Q_D twins; the actor term; the FB step
    7  push bz into the ZBuffer                       (lanes resample from it)

That is `fb_cpr/agent.py`'s `update` in order: `update_discriminator` on
`train_z` (stored), `sample_mixed_z` + relabel, then `update_fb`,
`update_critic`, `update_actor` on the relabelled z.

Composition, not a third copy: every rollout / ring / ZBuffer method is
delegated to the owned `FBOnlineAgent` (built with `EXPERT_ROWS = 0`: the
reference's FB batch is the ring alone; the expert set feeds D and the z
mixture ONLY, not the measure loss — A3.5 measured that mixing expert rows
into the FB batch diluted the manifold, §18.7.4).

Expert store: `attach_expert_windows(obs, starts, n_starts)` — the 24-D
observation table (via `obs_at`, the env's producer) and a table of VALID
window starts built from the store's episode index (`start + SEQ` inside
the episode), the same rule `fb_train_cpr_gpu.mojo` uses.

Checkpoint: the FB file (`fb_eval_walker_online.mojo` reads it unchanged)
+ the `.cpr` sidecar.
"""

from std.math import sqrt
from layout import Layout, LayoutTensor
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.core.logger import Logger
from noeira.nn.constants import DT, TPB
from noeira.nn.core.module import Module
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.initializer import Initializer, Xavier
from noeira.nn.core.ptr import mptr
from noeira.nn.random.box_muller import advance_rng_offset_kernel
from noeira.data.resident import IDX_DT
from noeira.data.replay_gpu import (
    _uniform_indices_dev_kernel,
    _incr_offset_kernel,
)

from ..data.n_step_replay import GPUNStepBuffer
from ..training.episode_tracker import EpisodeTracker
from ..training.driver_offpolicy import OffPolicyAgentGpu
from .online import FBOnlineAgent
from .cpr import FBCPRHead
from .kernels import (
    gather_rows_into_kernel,
    gather_rows_kernel,
    gather_idx_kernel,
    expand_windows_kernel,
    z_relabel3_kernel,
    project_sphere_kernel,
    uniform01_dev_kernel,
    gaussian_dev_t,
    scale_t,
    ensure_t,
    _blocks,
)


struct FBCPROnlineAgent[
    FNET: Module,
    BNET: Module,
    ANET: Module,
    DNET: Module,
    QNET: Module,
    OBS: Int,
    ACT: Int,
    D: Int,
    BATCH: Int,
    CAP: Int,
    LANES: Int,
    SEQ: Int,
    ZBUF: Int = 10_000,
    DERIVED_TAIL: Int = 0,
](OffPolicyAgentGpu):
    comptime AGENT_TRAIN_TARGET: StaticString = "gpu"
    comptime AGENT_OBS_DIM: Int = Self.OBS
    # what the expert table and the ring actually store; `OBS - DERIVED_TAIL`
    comptime STORE_OBS: Int = Self.OBS - Self.DERIVED_TAIL
    comptime AGENT_ACT_DIM: Int = Self.ACT
    comptime Base = FBOnlineAgent[
        Self.FNET, Self.BNET, Self.ANET, Self.OBS, Self.ACT, Self.D,
        Self.BATCH, Self.CAP, Self.LANES, Self.ZBUF, 0,
        Self.DERIVED_TAIL,
    ]
    comptime Head = FBCPRHead[
        Self.FNET, Self.BNET, Self.ANET, Self.DNET, Self.QNET,
        Self.OBS, Self.ACT, Self.D, Self.BATCH, Self.SEQ, "gpu",
    ]
    comptime NW: Int = Self.BATCH // Self.SEQ

    var base: Self.Base
    var head: Self.Head
    var ctx: Optional[DeviceContext]
    # ── expert windows (device), attached after `make` ──────────────────
    var exp_obs: Tensor
    var starts_dev: Optional[DeviceBuffer[IDX_DT]]
    var n_starts_dev: Optional[DeviceBuffer[DType.int32]]
    var idx_w: Optional[DeviceBuffer[IDX_DT]]
    var win_start: Optional[DeviceBuffer[IDX_DT]]
    var idx_e: Optional[DeviceBuffer[IDX_DT]]
    var idx_en: Optional[DeviceBuffer[IDX_DT]]
    var _n_starts: Int
    # ── the three-way relabel ───────────────────────────────────────────
    var pick3: Tensor
    var p_goal: Float64
    var p_expert: Float64
    var keep_frac: Float64

    def __init__(out self):
        self.base = Self.Base()
        self.head = Self.Head()
        self.ctx = None
        self.exp_obs = Tensor()
        self.starts_dev = None
        self.n_starts_dev = None
        self.idx_w = None
        self.win_start = None
        self.idx_e = None
        self.idx_en = None
        self._n_starts = 0
        self.pick3 = Tensor()
        self.p_goal = 0.2
        self.p_expert = 0.6
        self.keep_frac = 0.2

    def __init__(out self, *, deinit move: Self):
        self.base = move.base^
        self.head = move.head^
        self.ctx = move.ctx^
        self.exp_obs = move.exp_obs^
        self.starts_dev = move.starts_dev^
        self.n_starts_dev = move.n_starts_dev^
        self.idx_w = move.idx_w^
        self.win_start = move.win_start^
        self.idx_e = move.idx_e^
        self.idx_en = move.idx_en^
        self._n_starts = move._n_starts
        self.pick3 = move.pick3^
        self.p_goal = move.p_goal
        self.p_expert = move.p_expert
        self.keep_frac = move.keep_frac

    @staticmethod
    def make[
        INIT: Initializer = Xavier
    ](
        ctx: DeviceContext,
        *,
        lr: Float64 = 3e-4,
        lr_b: Float64 = 1e-5,
        lr_d: Float64 = 1e-5,
        lr_q: Float64 = 1e-4,
        gamma: Float64 = 0.98,
        tau: Float64 = 0.01,
        tau_q: Float64 = 0.005,
        ortho_weight: Float64 = 100.0,
        max_grad_norm: Float64 = 1.0,
        bc_weight: Float64 = 0.0,
        act_l2_weight: Float64 = 100.0,
        act_l2_margin: Float64 = 0.8,
        reg_coeff: Float64 = 0.01,
        gp_coef: Float64 = 10.0,
        gp_eps: Float64 = 1e-2,
        learning_starts: Int = 10_000,
        action_scale: Float64 = 1.0,
        expl_std: Float64 = 0.2,
        z_hold: Int = 150,
        zbuf_frac: Float64 = 0.5,
        keep_frac: Float64 = 0.2,
        p_goal: Float64 = 0.2,
        p_expert: Float64 = 0.6,
        window_size: Int = 100,
        initial_episode_fill: Float64 = 0.0,
        seed: UInt64 = UInt64(0x5EED_0C),
        normalize_obs: Bool = False,
    ) raises -> Self:
        """The reference's online CPR setting on top of `FBOnlineAgent`'s
        rollout defaults: `bc_weight 0` (CPR is BC's replacement; the FB
        batch has no expert rows to clone anyway), the A3 action hinge
        (100 @ 0.8 — our stand-in for the reference's auxiliary critic's
        action-rate term, §16.1), `relabel_ratio 0.8` = `keep_frac 0.2`,
        mixture 0.2 goal / 0.6 expert / 0.2 uniform, ZBuffer rollouts.
        """
        if p_goal + p_expert > 1.0 or p_goal < 0.0 or p_expert < 0.0:
            raise Error("FBCPROnlineAgent.make: p_goal + p_expert must lie in [0, 1]")
        var a = Self()
        var octx = Optional[DeviceContext](ctx)
        a.ctx = octx
        a.base = Self.Base.make[INIT](
            ctx, lr=lr, lr_b=lr_b, gamma=gamma, tau=tau,
            ortho_weight=ortho_weight, max_grad_norm=max_grad_norm,
            bc_weight=bc_weight, act_l2_weight=act_l2_weight,
            act_l2_margin=act_l2_margin, learning_starts=learning_starts,
            action_scale=action_scale, expl_std=expl_std, z_hold=z_hold,
            zbuf_frac=zbuf_frac, keep_frac=keep_frac, uniform_frac=0.5,
            window_size=window_size, initial_episode_fill=initial_episode_fill,
            seed=seed, normalize_obs=normalize_obs,
        )
        a.head = Self.Head.make[INIT](
            octx, lr_d=lr_d, lr_q=lr_q, gamma=gamma, tau_q=tau_q,
            max_grad_norm=max_grad_norm, reg_coeff=reg_coeff, gp_coef=gp_coef,
            gp_eps=gp_eps, policy_noise=a.base.t.policy_noise,
            noise_clip=a.base.t.noise_clip, seed=seed + 707,
        )
        # Sizes the head's scratch and sets `t.has_pi_extra` — BEFORE any
        # capture. D's negatives carry the stored z (`z_neg`), not `t.bz`.
        a.head.ensure_sized(a.base.t)
        a.head.use_z_neg = True
        a.keep_frac = keep_frac
        a.p_goal = p_goal
        a.p_expert = p_expert
        ensure_t["gpu"](a.pick3, Self.BATCH * 3, octx)
        a.idx_w = ctx.enqueue_create_buffer[IDX_DT](Self.NW)
        a.win_start = ctx.enqueue_create_buffer[IDX_DT](Self.NW)
        a.idx_e = ctx.enqueue_create_buffer[IDX_DT](Self.BATCH)
        a.idx_en = ctx.enqueue_create_buffer[IDX_DT](Self.BATCH)
        ctx.synchronize()
        return a^

    # ── expert windows ───────────────────────────────────────────────────

    def attach_expert_windows(
        mut self,
        var obs: Tensor,
        var starts: DeviceBuffer[IDX_DT],
        n_starts: Int,
    ) raises:
        """`obs` `[n_rows, OBS]` uploaded, in the ENV's observation layout;
        `starts` the valid window starts (`start + SEQ` inside its episode,
        so rows `start..start+SEQ-1` and their next rows never cross an
        episode end). ⚠ Required before the driver: a CPR agent with no
        expert set has nothing for D to certify."""
        if n_starts < Self.NW:
            raise Error("attach_expert_windows: fewer starts than windows per batch")
        if not obs.dev:
            raise Error("attach_expert_windows: obs must be uploaded to device")
        var c = self.ctx.value()
        self.exp_obs = obs^
        # the gathers above write only the stored head of `es` / `esn`; zero
        # their derived tails ONCE so `b` and D are never handed whatever the
        # allocator left there (both filter it out, which is exactly the kind
        # of "harmless" that stops being harmless when a filter widens)
        comptime if Self.DERIVED_TAIL > 0:
            self.head.es.dev.value().enqueue_fill(Scalar[DT](0.0))
            self.head.esn.dev.value().enqueue_fill(Scalar[DT](0.0))
        self.starts_dev = starts^
        var nb = c.enqueue_create_buffer[DType.int32](1)
        nb.enqueue_fill(Int32(n_starts))
        self.n_starts_dev = nb^
        self._n_starts = n_starts
        c.synchronize()

    def has_expert(self) -> Bool:
        return self._n_starts > 0

    def _gather_expert_windows(mut self) raises:
        """NW window starts → BATCH rows of expert `s` and `s'` in
        `head.es` / `head.esn`. Device-offset draws: capture-safe."""
        var c = self.ctx.value()
        var n_lt = LayoutTensor[DType.int32, Layout.row_major(1)](
            self.n_starts_dev.value()
        )
        var off_lt = LayoutTensor[DType.uint64, Layout.row_major(1)](
            self.base.samp_off.value()
        )
        var iw_lt = LayoutTensor[IDX_DT, Layout.row_major(Self.NW)](self.idx_w.value())
        c.enqueue_function[_uniform_indices_dev_kernel[Self.NW]](
            iw_lt, n_lt, self.base._train_seed + 4242, off_lt,
            grid_dim=_blocks(Self.NW), block_dim=TPB,
        )
        c.enqueue_function[_incr_offset_kernel[Self.NW]](
            off_lt, grid_dim=1, block_dim=1,
        )
        c.enqueue_function[gather_idx_kernel[Self.NW]](
            mptr(self.starts_dev.value().unsafe_ptr()),
            mptr(self.idx_w.value().unsafe_ptr()),
            mptr(self.win_start.value().unsafe_ptr()),
            grid_dim=_blocks(Self.NW), block_dim=TPB,
        )
        c.enqueue_function[expand_windows_kernel[Self.NW, Self.SEQ]](
            mptr(self.win_start.value().unsafe_ptr()),
            mptr(self.idx_e.value().unsafe_ptr()),
            mptr(self.idx_en.value().unsafe_ptr()),
            grid_dim=_blocks(Self.BATCH), block_dim=TPB,
        )
        # ⚠ THE EXPERT TABLE IS `STORE_OBS` WIDE, NOT `OBS`. Its only readers
        # are `b` (the window encoding) and the discriminator, whose reference
        # filters are both `state + privileged_state` — so it carries no
        # derived tail (§12.37). Gathering it at `OBS` stride misaligns every
        # row after the first and reads PAST THE BUFFER on the last ones: on
        # the box that produced garbage -> NaN in `es` / `esn` -> NaN through
        # D and B and, one optimizer step later, NaN weights everywhere
        # (§12.38). The destination rows stay `OBS` wide; their tails were
        # zeroed once at attach.
        c.enqueue_function[
            gather_rows_into_kernel[Self.STORE_OBS, Self.OBS, Self.BATCH]
        ](
            mptr(self.exp_obs.dev.value().unsafe_ptr()),
            mptr(self.idx_e.value().unsafe_ptr()),
            mptr(self.head.es.dev.value().unsafe_ptr()),
            grid_dim=_blocks(Self.BATCH * Self.STORE_OBS), block_dim=TPB,
        )
        c.enqueue_function[
            gather_rows_into_kernel[Self.STORE_OBS, Self.OBS, Self.BATCH]
        ](
            mptr(self.exp_obs.dev.value().unsafe_ptr()),
            mptr(self.idx_en.value().unsafe_ptr()),
            mptr(self.head.esn.dev.value().unsafe_ptr()),
            grid_dim=_blocks(Self.BATCH * Self.STORE_OBS), block_dim=TPB,
        )

    def _relabel_z3(mut self) raises:
        """`bz` holds the stored z; keep `keep_frac`, else the goal / expert
        / uniform mixture; project every row. `head.ez` must be current."""
        var c = self.ctx.value()
        self.base.t.embed_sp()
        gaussian_dev_t["gpu", Self.BATCH * Self.D](
            self.base.gauss, self.base._train_seed + 5,
            self.base.rng_dev.value(), self.ctx,
        )
        var off_lt = LayoutTensor[DType.uint64, Layout.row_major(1), MutAnyOrigin](
            mptr(self.base.rng_dev.value().unsafe_ptr())
        )
        comptime NP = Self.BATCH * 3
        c.enqueue_function[uniform01_dev_kernel[NP]](
            mptr(self.pick3.dev.value().unsafe_ptr()),
            self.base._train_seed + 9, off_lt,
            grid_dim=_blocks(NP), block_dim=TPB,
        )
        c.enqueue_function[advance_rng_offset_kernel[2 * NP]](
            off_lt, grid_dim=1, block_dim=1
        )
        c.enqueue_function[z_relabel3_kernel[Self.D, Self.BATCH]](
            mptr(self.base.t.bz.dev.value().unsafe_ptr()),
            mptr(self.base.gauss.dev.value().unsafe_ptr()),
            mptr(self.base.t.b_sp.dev.value().unsafe_ptr()),
            mptr(self.head.ez.dev.value().unsafe_ptr()),
            mptr(self.pick3.dev.value().unsafe_ptr()),
            Scalar[DT](self.keep_frac), Scalar[DT](self.p_goal),
            Scalar[DT](self.p_expert), Int32(Self.BATCH), Int32(Self.BATCH),
            grid_dim=_blocks(Self.BATCH), block_dim=TPB,
        )
        c.enqueue_function[project_sphere_kernel[Self.D, Self.BATCH]](
            mptr(self.base.t.bz.dev.value().unsafe_ptr()),
            Scalar[DT](sqrt(Float64(Self.D))),
            grid_dim=_blocks(Self.BATCH), block_dim=TPB,
        )

    def _train_kernels(mut self) raises:
        """The pure device sequence of the module docstring."""
        if self._n_starts <= 0:
            raise Error("FBCPROnlineAgent: attach_expert_windows before training")
        self.base._sample_batch()
        scale_t["gpu", Self.BATCH * Self.D](
            self.head.z_neg, self.base.t.bz, Scalar[DT](1.0), self.ctx
        )
        self._gather_expert_windows()
        if self.base.obs_ema.enabled:
            # expert rows are normalised with the running statistics and
            # never update them (the reference's eval-mode pass)
            self.base.obs_ema.apply[Self.BATCH](self.head.es)
            self.base.obs_ema.apply[Self.BATCH](self.head.esn)
        self.head.encode_expert(self.base.t)
        self._relabel_z3()
        _ = self.head.step(self.base.t, want_loss=False)
        self.base._push_zbuf()
        comptime lba = Layout.row_major(Self.BATCH * Self.ACT)
        self.base._mean_abs_action_dev.accumulate_gpu_abs_lt[Self.BATCH * Self.ACT](
            self.base.t.ba.lt["gpu", lba]()
        )

    # ── OffPolicyAgentGpu: rollout surface, delegated ────────────────────

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
        self.base.select_action_batched[N_ENVS](
            obs, action, ao_scratch, alp_scratch, step_idx
        )

    def select_greedy_action_batched[
        N_ENVS: Int,
    ](
        mut self,
        ctx: Optional[DeviceContext],
        obs: LayoutTensor[
            DT, Layout.row_major(N_ENVS, Self.AGENT_OBS_DIM), MutAnyOrigin
        ],
        action: LayoutTensor[
            DT, Layout.row_major(N_ENVS, Self.AGENT_ACT_DIM), MutAnyOrigin
        ],
        ao_scratch: LayoutTensor[
            DT, Layout.row_major(N_ENVS, 2 * Self.AGENT_ACT_DIM), MutAnyOrigin
        ],
    ) raises:
        self.base.select_greedy_action_batched[N_ENVS](ctx, obs, action, ao_scratch)

    def select_greedy_action(
        mut self,
        ref obs: List[Scalar[DT]],
        mut action_out: List[Scalar[DT]],
    ) raises:
        self.base.select_greedy_action(obs, action_out)

    def record(
        mut self,
        ref obs: List[Scalar[DT]],
        ref action: List[Scalar[DT]],
        reward: Scalar[DT],
        ref next_obs: List[Scalar[DT]],
        done: Scalar[DT],
    ) raises:
        self.base.record(obs, action, reward, next_obs, done)

    def _replay_add(
        mut self,
        ref obs: List[Scalar[DT]],
        ref action: List[Scalar[DT]],
        reward: Scalar[DT],
        ref next_obs: List[Scalar[DT]],
        done: Scalar[DT],
    ) raises:
        self.base._replay_add(obs, action, reward, next_obs, done)

    def _tracker_ptr(self) -> Pointer[EpisodeTracker, MutAnyOrigin]:
        return self.base._tracker_ptr()

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
        self.base.record_batch_gpu[N_ENVS](
            ctx, prev_obs_dev, action_dev, reward_dev, obs_dev, done_dev
        )

    def set_boundary(mut self, b: Bool):
        """Forwarded to the ring — see `FBOnlineAgent.set_boundary`. The
        driver calls this on the agent it holds, which is this one."""
        self.base.set_boundary(b)

    def set_age(mut self, a: Int):
        """Forwarded — see `FBOnlineAgent.set_age` (docs §12.36)."""
        self.base.set_age(a)

    def attach_tail_spec(mut self, ref spec: List[Int32]) raises:
        """Forwarded — see `FBOnlineAgent.attach_tail_spec`."""
        self.base.attach_tail_spec(spec)

    def set_action_norm(mut self, scale: Float64, clip: Float64):
        """The scaling the DERIVED tail applies to stored actions.

        The G1 stores the actor's RAW output and its PD chain consumes
        `clip(a * 5, +-5)`; the tail must reproduce what the policy actually
        saw, not what the net emitted (docs §12.36)."""
        self.base.act_norm = Scalar[DT](scale)
        self.base.act_clip = Scalar[DT](clip)

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
        raise Error("FBCPROnlineAgent: n-step is meaningless for FB; run NS=1")

    # ── OffPolicyAgentGpu: training surface ──────────────────────────────

    def train_step(mut self, step_idx: Int) raises -> Bool:
        if step_idx < self.base.learning_starts or self.base.size < Self.BATCH:
            return False
        self._train_kernels()
        self.base._update_count += 1
        self.base._total_train_steps += 1
        return True

    def train_device_kernels(mut self) raises:
        """Capture body — see `FBOnlineAgent.train_device_kernels`."""
        self._train_kernels()

    def note_train_update(mut self):
        self.base.note_train_update()

    def learning_starts_count(self) -> Int:
        return self.base.learning_starts

    def total_train_steps(self) -> Int:
        return self.base._total_train_steps

    def replay_size(self) -> Int:
        return self.base.size

    # ── diagnostics ──────────────────────────────────────────────────────

    def flush_metrics_through_logger[
        L: Logger
    ](
        mut self,
        logger: Optional[Pointer[L, MutAnyOrigin]],
        step: Int,
    ) raises:
        self.base.flush_metrics_through_logger[L](logger, step)
        var d_pos = Float64(0)
        var d_neg = Float64(0)
        var r_mean = Float64(0)
        var q_mean = Float64(0)
        var q_loss = Float64(0)
        var q_pi = Float64(0)
        self.head.read_diag(d_pos, d_neg, r_mean, q_mean, q_loss, q_pi)
        if Bool(logger):
            var names = List[String]()
            var vals = List[Float64]()
            names.append(String("disc_expert_loss")); vals.append(d_pos)
            names.append(String("disc_policy_loss")); vals.append(d_neg)
            names.append(String("disc_reward_mean")); vals.append(r_mean)
            names.append(String("mean_q")); vals.append(q_mean)
            names.append(String("critic_loss")); vals.append(q_loss)
            names.append(String("policy_q_mean")); vals.append(q_pi)
            logger.value()[].log_scalars(names, vals, step)
        print(
            "   [cpr] step", step, " D+", d_pos, " D-", d_neg, " r", r_mean,
            " Q", q_mean, " Qloss", q_loss, " Qpi", q_pi,
        )
        # ⚠ Reading it: D+/D- both at log 2 = D cannot separate; both near
        # 0 = D won and r_D saturates. Online, `Qpi > Q` is the sign the
        # actor is actually raising the style value — offline (§18.9.1)
        # they were equal to four figures, which is the null this run
        # exists to break.

    # ── checkpoint ───────────────────────────────────────────────────────

    def save_state(mut self, path: String) raises:
        """FB nets in `FBTrainer`'s own format (the evals load it), D and
        Q_D in the `.cpr` sidecar."""
        self.base.save_state(path)
        self.head.save_sidecar(path)

    def load_state(mut self, path: String) raises:
        self.base.load_state(path)
        self.head.load_sidecar(path)
