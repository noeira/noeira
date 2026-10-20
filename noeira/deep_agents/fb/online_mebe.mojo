"""`FBMebeOnlineAgent` — FB-MEBE online: inverse-density exploration plus a
regularity critic, around the unchanged FB rollout and trainer.

`noeira-docs/FB_MEBE_G1_PLAN.md` §2; Hu, Armengol Urpí, Cheng, Coros,
"Maximum Entropy Behavior Exploration for Sim2Real Zero-Shot RL" (RLJ 2026,
arXiv 2603.25464), Algorithm 1. The composition:

    FBOnlineAgent   the rollout, the device ring and the unchanged FBTrainer,
                    with its non-uniform z source switched to an external
                    pool (`enable_zsrc`)
    the MEBE pool   `BATCH` rows of `z = B(s)`, `s` drawn from the ring with
                    probability `∝ (q(φ(s)) + ε)^-β` (`fb/density.mojo`),
                    refreshed every `refresh_every` env iterations (eager)
    FBRegHead       the twin `Q_reg` on a regularity reward stored per ring
                    row, and its actor term (`fb/reg_head.mojo`)

Both halves are optional and independent (`mebe`, `use_reg`), so one binary
runs the paper's four arms: FB, FB-Reg, FB-MEBE (β = 0), FB-MEBE.

## Per env iteration (eager)

    every `refresh_every` iterations, once the ring holds 10·LANES rows:
      draw NC candidate + NR reference rows uniformly from the ring (host)
      gather their φ and filter columns (device) → host
      validity = every filter column inside its box (fall / outlier filter)
      draw BATCH candidates ∝ (q + ε)^-β over the valid ones (KDE on φ)
      gather those rows → normalise (if on) → B → project → `base.zsrc`
    record: r_reg of each lane's transition into `r_rew` at the ring row the
      base is about to write, then the base's own record

## Per training step (capturable)

    base._sample_batch()            s, a, s', s+, stored z
    br ← r_rew[idx_s]               (the SAME index as s)
    base._relabel_z()               keep `keep_frac`, else uniform /
                                    `zsrc` — the release keeps 0 and draws
                                    80 % from the pool, 20 % uniform
    head.step(t)  or  t.train_step()
    push bz into the ZBuffer        (unused by the lanes once zsrc is on;
                                    kept so the diagnostics stay comparable)

## The regularity reward (agent-side, from the ring's own columns)

    r_reg = − w_acc · ‖(q̇' − q̇) / dt‖²  − w_ar · ‖a − a_prev‖²  + w_env · r_env

over the observation columns `[qd_off, qd_off + qd_n)`. The joint
acceleration is a finite difference at the CONTROL step (the release reads
the simulator's per-substep `joint_acc`, so `w_acc` is not the paper's
2.5e-7 one for one). `a_prev` is per lane and resets to 0 at a done or a
scheduled boundary, as IsaacLab's `_previous_actions` does. `r_env` is the
env's own reward, for an env built to return contact-based terms (feet
slip) that only the simulator sees; 0 for every other env, whose reward is a
task and must NOT leak into an unsupervised critic.

Rows whose successor crosses a reset get r = 0; the sampler never draws
them as `s` (`ring_indices_kernel` remaps boundary rows), so the value is
never read.

## What is NOT here (yet)

  * The release's Fz/Q-regulariser (`q_loss_coef 0.1`, Touati 2023 App. B):
    needs `Cov(B)^-1` on device every step. Plan §1.3 item 4.
  * Termination: the release terminates on any non-foot contact. The walker
    never terminates; the G1 driver resets fallen lanes as a TRUNCATION
    (plan §2), so no terminal mask is needed in the FB target.
"""

from std.math import sqrt
from max.gpu import global_idx
from max.gpu.host import DeviceContext, DeviceBuffer

from noeira.core.logger import Logger
from noeira.nn.constants import DT, TPB
from noeira.nn.core.module import Module
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.call import call_forward
from noeira.nn.core.initializer import Initializer, Xavier
from noeira.nn.core.ptr import mptr
from noeira.data.resident import IDX_DT

from ..data.n_step_replay import GPUNStepBuffer
from ..training.episode_tracker import EpisodeTracker
from ..training.driver_offpolicy import OffPolicyAgentGpu
from .online import FBOnlineAgent
from .reg_head import FBRegHead
from .density import (
    SplitMix64,
    InverseDensityConfig,
    InverseDensityReport,
    inverse_density_draw,
)
from .kernels import (
    gather_rows_kernel,
    gather_rows_into_kernel,
    project_sphere_kernel,
    ensure_t,
    _blocks,
)
from noeira.core.tiles import RowMajor, Tile


# ══════════════════════════════════════════════════════════════════════
# Kernels
# ══════════════════════════════════════════════════════════════════════


def gather_cols_kernel[SRC_W: Int, SDT: DType = DT](
    src: Pointer[Scalar[SDT], MutAnyOrigin],
    idx: Pointer[Scalar[IDX_DT], MutAnyOrigin],
    cols: Pointer[Scalar[DType.int32], MutAnyOrigin],
    dst: Pointer[Scalar[DT], MutAnyOrigin],
    n_rows: Int32,
    n_cols: Int32,
):
    """`dst[i, k] = src[idx[i], cols[k]]` — a handful of columns of many
    rows, for the host-side density (φ and the filter columns only)."""
    var t = Int(global_idx.x)
    var nc = Int(n_cols)
    if t >= Int(n_rows) * nc:
        return
    var i = t // nc
    var k = t % nc
    dst[unsafe_offset=t] = src[
        unsafe_offset=Int(idx[unsafe_offset=i]) * SRC_W
        + Int(cols[unsafe_offset=k])
    ].cast[DT]()


def reg_reward_kernel[OBS_W: Int, ACT: Int, LANES: Int, CAP: Int](
    prev_obs: Pointer[Scalar[DT], MutAnyOrigin],
    next_obs: Pointer[Scalar[DT], MutAnyOrigin],
    act: Pointer[Scalar[DT], MutAnyOrigin],
    done: Pointer[Scalar[DT], MutAnyOrigin],
    env_rew: Pointer[Scalar[DT], MutAnyOrigin],
    prev_act: Pointer[Scalar[DT], MutAnyOrigin],
    r_rew: Pointer[Scalar[DT], MutAnyOrigin],
    pos: Int32,
    bnd: Int32,
    qd_off: Int32,
    qd_n: Int32,
    inv_dt: Scalar[DT],
    w_acc: Scalar[DT],
    w_ar: Scalar[DT],
    w_env: Scalar[DT],
):
    """One thread per lane: the regularity reward of this step's transition,
    written at the ring row the base is about to fill, then `a_prev` moved
    on (or reset at a done / scheduled boundary). See the module header."""
    var l = Int(global_idx.x)
    if l >= LANES:
        return
    var row = (Int(pos) + l) % CAP
    var crosses = (Int(bnd) != 0) or (done[unsafe_offset=l] != Scalar[DT](0.0))
    var r = Scalar[DT](0.0)
    if not crosses:
        var acc = Scalar[DT](0.0)
        for k in range(Int(qd_n)):
            var c = Int(qd_off) + k
            var dq = (
                next_obs[unsafe_offset=l * OBS_W + c]
                - prev_obs[unsafe_offset=l * OBS_W + c]
            ) * inv_dt
            acc += dq * dq
        var ar = Scalar[DT](0.0)
        for j in range(ACT):
            var da = act[unsafe_offset=l * ACT + j] - prev_act[unsafe_offset=l * ACT + j]
            ar += da * da
        r = -w_acc * acc - w_ar * ar + w_env * env_rew[unsafe_offset=l]
    r_rew[unsafe_offset=row] = r
    for j in range(ACT):
        prev_act[unsafe_offset=l * ACT + j] = (
            Scalar[DT](0.0) if crosses else act[unsafe_offset=l * ACT + j]
        )


# ══════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════


struct MebeConfig(Copyable, Movable):
    """The exploration half. `phi_cols`: observation columns of φ (the
    density's space). `filt_cols` / `filt_lo` / `filt_hi`: a state is valid
    for exploration iff every listed column lies in its box (the release:
    `g_z <= -0.93` and |vx| <= 1.8, |vy| <= 1.3). Columns index the STORED
    observation (`< OBS - DERIVED_TAIL`)."""

    var phi_cols: List[Int]
    var filt_cols: List[Int]
    var filt_lo: List[Float64]
    var filt_hi: List[Float64]
    var density: InverseDensityConfig
    var n_cand: Int
    var n_ref: Int
    var refresh_every: Int

    def __init__(out self, var phi_cols: List[Int]):
        self.phi_cols = phi_cols^
        self.filt_cols = List[Int]()
        self.filt_lo = List[Float64]()
        self.filt_hi = List[Float64]()
        self.density = InverseDensityConfig.default()
        self.n_cand = 8192
        self.n_ref = 2048
        self.refresh_every = 50

    def add_filter(mut self, col: Int, lo: Float64, hi: Float64):
        self.filt_cols.append(col)
        self.filt_lo.append(lo)
        self.filt_hi.append(hi)


@fieldwise_init
struct RegRewardConfig(Copyable, Movable):
    """The regularity reward (module header). `qd_n = 0` disables the
    joint-acceleration term; `w_env = 0` ignores the env's reward."""

    var qd_off: Int
    var qd_n: Int
    var dt: Float64
    var w_acc: Float64
    var w_ar: Float64
    var w_env: Float64


# ══════════════════════════════════════════════════════════════════════
# The agent
# ══════════════════════════════════════════════════════════════════════


struct FBMebeOnlineAgent[
    FNET: Module,
    BNET: Module,
    ANET: Module,
    QNET: Module,
    OBS: Int,
    ACT: Int,
    D: Int,
    BATCH: Int,
    CAP: Int,
    LANES: Int,
    ZBUF: Int = 10_000,
    DERIVED_TAIL: Int = 0,
    RING_DT: DType = DT,
](OffPolicyAgentGpu):
    comptime AGENT_TRAIN_TARGET: StaticString = "gpu"
    comptime AGENT_OBS_DIM: Int = Self.OBS
    comptime AGENT_ACT_DIM: Int = Self.ACT
    comptime STORE_OBS: Int = Self.OBS - Self.DERIVED_TAIL
    comptime Base = FBOnlineAgent[
        Self.FNET, Self.BNET, Self.ANET, Self.OBS, Self.ACT, Self.D,
        Self.BATCH, Self.CAP, Self.LANES, Self.ZBUF, 0, Self.DERIVED_TAIL,
        Self.RING_DT,
    ]
    comptime Head = FBRegHead[
        Self.FNET, Self.BNET, Self.ANET, Self.QNET,
        Self.OBS, Self.ACT, Self.D, Self.BATCH, "gpu",
    ]

    var base: Self.Base
    var head: Self.Head
    var ctx: Optional[DeviceContext]
    var use_reg: Bool
    var mebe: Bool
    var cfg: MebeConfig
    var reg: RegRewardConfig
    # ── regularity reward ring + per-lane previous action ─────────────
    var r_rew: Tensor          # CAP
    var _prev_act: Tensor      # LANES * ACT
    # ── MEBE refresh scratch (eager) ───────────────────────────────────
    var _cols_dev: Optional[DeviceBuffer[DType.int32]]
    var _idx_cr: Optional[DeviceBuffer[IDX_DT]]     # NC + NR rows
    var _idx_pick: Optional[DeviceBuffer[IDX_DT]]   # BATCH rows
    var _phi_dev: Tensor                            # (NC + NR) * n_cols
    var _pool_obs: Tensor                           # BATCH * OBS
    var _rng: SplitMix64
    var _iter: Int
    var _n_refresh: Int
    var last_report: InverseDensityReport
    # ── scheduled truncation (see `episode_len` in `make`) ─────────────
    var episode_len: Int
    var _n_record: Int
    # ── ring-row embedding for evals (`embed_rows`), allocated on first use
    var _e_idx: Optional[DeviceBuffer[IDX_DT]]
    var _e_obs: Tensor     # BATCH * OBS
    var _e_b: Tensor       # BATCH * D

    def __init__(out self):
        self.base = Self.Base()
        self.head = Self.Head()
        self.ctx = None
        self.use_reg = False
        self.mebe = False
        self.cfg = MebeConfig(List[Int]())
        self.reg = RegRewardConfig(0, 0, 1.0, 0.0, 0.0, 0.0)
        self.r_rew = Tensor()
        self._prev_act = Tensor()
        self._cols_dev = None
        self._idx_cr = None
        self._idx_pick = None
        self._phi_dev = Tensor()
        self._pool_obs = Tensor()
        self._rng = SplitMix64(UInt64(0x3EBE))
        self._iter = 0
        self._n_refresh = 0
        self.last_report = InverseDensityReport()
        self.episode_len = 0
        self._n_record = 0
        self._e_idx = None
        self._e_obs = Tensor()
        self._e_b = Tensor()

    def __init__(out self, *, deinit move: Self):
        self.base = move.base^
        self.head = move.head^
        self.ctx = move.ctx^
        self.use_reg = move.use_reg
        self.mebe = move.mebe
        self.cfg = move.cfg^
        self.reg = move.reg^
        self.r_rew = move.r_rew^
        self._prev_act = move._prev_act^
        self._cols_dev = move._cols_dev^
        self._idx_cr = move._idx_cr^
        self._idx_pick = move._idx_pick^
        self._phi_dev = move._phi_dev^
        self._pool_obs = move._pool_obs^
        self._rng = move._rng^
        self._iter = move._iter
        self._n_refresh = move._n_refresh
        self.last_report = move.last_report^
        self.episode_len = move.episode_len
        self._n_record = move._n_record
        self._e_idx = move._e_idx^
        self._e_obs = move._e_obs^
        self._e_b = move._e_b^

    @staticmethod
    def make[
        INIT: Initializer = Xavier
    ](
        ctx: DeviceContext,
        var cfg: MebeConfig,
        reg: RegRewardConfig,
        *,
        mebe: Bool = True,
        use_reg: Bool = True,
        reg_coeff: Float64 = 20.0,
        reg_raw: Bool = True,
        lr: Float64 = 3e-4,
        lr_b: Float64 = -1.0,
        lr_q: Float64 = 5e-5,
        gamma: Float64 = 0.98,
        tau: Float64 = 0.01,
        tau_q: Float64 = 0.005,
        ortho_weight: Float64 = 100.0,
        max_grad_norm: Float64 = 1.0,
        act_l2_weight: Float64 = 100.0,
        act_l2_margin: Float64 = 0.8,
        learning_starts: Int = 10_000,
        action_scale: Float64 = 1.0,
        expl_std: Float64 = 0.2,
        z_hold: Int = 100,
        explore_frac: Float64 = 0.8,
        keep_frac: Float64 = 0.0,
        window_size: Int = 100,
        seed: UInt64 = UInt64(0x5EED_0E),
        normalize_obs: Bool = False,
        episode_len: Int = 0,
    ) raises -> Self:
        """The release's mixture: lanes held `z_hold` 100 steps then 80 %
        pool / 20 % sphere; relabel keeps 0 % of the stored z and draws
        80 % pool / 20 % sphere. With `mebe = False` the pool is never
        refreshed AND never switched on, so the base runs its own rule
        (ZBuffer lanes, `B(s+)` relabel at the same 80/20) — the paper's FB
        and FB-Reg arms. `use_reg = False` skips the head entirely.
        ⚠ `act_l2_weight` stays ON by default (the A3 hinge): a λ sweep is
        read against the same actor as every walker arm since §18.7.

        ⚠ `episode_len`: the TIME LIMIT of an env whose lanes all truncate
        together (walker 1000, pendulum 200), or 0. The batched driver hands
        `record_batch_gpu` the env's `terminated` only — a truncation reaches
        the agent as nothing at all, so without this the ring derives the
        last pre-reset row's successor from the POST-reset observation, and
        `a_prev` survives into the next episode. With it, every
        `episode_len`-th record is flagged as a boundary, exactly as the G1
        driver's `set_boundary((s + 1) % T_EPISODE == 0)` does."""
        for k in range(len(cfg.phi_cols)):
            if cfg.phi_cols[k] < 0 or cfg.phi_cols[k] >= Self.STORE_OBS:
                raise Error("FBMebeOnlineAgent: φ column outside the stored obs")
        for k in range(len(cfg.filt_cols)):
            if cfg.filt_cols[k] < 0 or cfg.filt_cols[k] >= Self.STORE_OBS:
                raise Error("FBMebeOnlineAgent: filter column outside the stored obs")
        if mebe and len(cfg.phi_cols) == 0:
            raise Error("FBMebeOnlineAgent: mebe needs at least one φ column")
        if reg.qd_n > 0 and (reg.qd_off < 0 or reg.qd_off + reg.qd_n > Self.STORE_OBS):
            raise Error("FBMebeOnlineAgent: q̇ columns outside the stored obs")
        var a = Self()
        var octx = Optional[DeviceContext](ctx)
        a.ctx = octx
        a.mebe = mebe
        a.use_reg = use_reg
        a.reg = reg.copy()
        a.base = Self.Base.make[INIT](
            ctx, lr=lr, lr_b=lr_b, gamma=gamma, tau=tau,
            ortho_weight=ortho_weight, max_grad_norm=max_grad_norm,
            bc_weight=0.0, act_l2_weight=act_l2_weight,
            act_l2_margin=act_l2_margin, learning_starts=learning_starts,
            action_scale=action_scale, expl_std=expl_std, z_hold=z_hold,
            zbuf_frac=explore_frac, keep_frac=keep_frac,
            uniform_frac=1.0 - explore_frac, window_size=window_size,
            initial_episode_fill=0.0, seed=seed, normalize_obs=normalize_obs,
        )
        if use_reg:
            a.head = Self.Head.make[INIT](
                octx, lr_q=lr_q, gamma=gamma, tau_q=tau_q,
                max_grad_norm=max_grad_norm, reg_coeff=reg_coeff, raw=reg_raw,
                policy_noise=a.base.t.policy_noise,
                noise_clip=a.base.t.noise_clip, seed=seed + 909,
            )
            # sizes `br` and sets the trainer's hook flags — BEFORE capture
            a.head.ensure_sized(a.base.t)
        a.r_rew.ensure_gpu(ctx, Self.CAP)
        a.r_rew.dev.value().enqueue_fill(Scalar[DT](0.0))
        a._prev_act.ensure_gpu(ctx, Self.LANES * Self.ACT)
        a._prev_act.dev.value().enqueue_fill(Scalar[DT](0.0))
        if mebe:
            var ncol = len(cfg.phi_cols) + len(cfg.filt_cols)
            var cd = ctx.enqueue_create_buffer[DType.int32](ncol)
            var ch = ctx.enqueue_create_host_buffer[DType.int32](ncol)
            for k in range(len(cfg.phi_cols)):
                ch[k] = Int32(cfg.phi_cols[k])
            for k in range(len(cfg.filt_cols)):
                ch[len(cfg.phi_cols) + k] = Int32(cfg.filt_cols[k])
            ctx.enqueue_copy(cd, ch)
            a._cols_dev = cd^
            a._idx_cr = ctx.enqueue_create_buffer[IDX_DT](cfg.n_cand + cfg.n_ref)
            a._idx_pick = ctx.enqueue_create_buffer[IDX_DT](Self.BATCH)
            a._phi_dev.ensure_gpu(ctx, (cfg.n_cand + cfg.n_ref) * ncol)
            a._pool_obs.ensure_gpu(ctx, Self.BATCH * Self.OBS)
            # the derived tail of a pooled row is never written by the
            # gather; zero it once (B's filter reads the stored prefix)
            a._pool_obs.dev.value().enqueue_fill(Scalar[DT](0.0))
            a.base.enable_zsrc()
        a._rng = SplitMix64(seed + UInt64(0x3EBE))
        a.episode_len = episode_len
        a.cfg = cfg^
        ctx.synchronize()
        return a^

    # ── the MEBE pool ────────────────────────────────────────────────────

    def refresh_pool(mut self) raises:
        """Redraw the exploration pool `zsrc` from the ring (eager; see the
        module header). Public so a test or a driver can force one."""
        var c = self.ctx.value()
        var size = self.base.size
        if size < 1:
            raise Error("refresh_pool: the ring is empty")
        var nc = self.cfg.n_cand
        var nr = self.cfg.n_ref
        var nphi = len(self.cfg.phi_cols)
        var nf = len(self.cfg.filt_cols)
        var ncol = nphi + nf
        var nrows = nc + nr
        # 1. candidate + reference rows, uniform over the fill (host draw)
        var hidx = c.enqueue_create_host_buffer[IDX_DT](nrows)
        var rows = List[Int](length=nrows, fill=0)
        for i in range(nrows):
            rows[i] = self._rng.below(size)
            hidx[i] = Scalar[IDX_DT](rows[i])
        c.enqueue_copy(self._idx_cr.value(), hidx)
        # 2. gather φ + filter columns, D2H
        c.enqueue_function[gather_cols_kernel[Self.STORE_OBS, Self.RING_DT]](
            self.base.ring_obs_ptr(),
            mptr(self._idx_cr.value().unsafe_ptr()),
            mptr(self._cols_dev.value().unsafe_ptr()),
            mptr(self._phi_dev.dev.value().unsafe_ptr()),
            Int32(nrows), Int32(ncol),
            grid_dim=(nrows * ncol + TPB - 1) // TPB, block_dim=TPB,
        )
        var hphi = c.enqueue_create_host_buffer[DT](nrows * ncol)
        c.enqueue_copy(hphi, self._phi_dev.dev.value())
        c.synchronize()
        # 3. validity + φ arrays
        var cand = List[Float64](length=nc * nphi, fill=0.0)
        var refs = List[Float64](length=nr * nphi, fill=0.0)
        var cv = List[Bool](length=nc, fill=True)
        var rv = List[Bool](length=nr, fill=True)
        for i in range(nrows):
            var ok = True
            for f in range(nf):
                var x = Float64(hphi[i * ncol + nphi + f])
                if x < self.cfg.filt_lo[f] or x > self.cfg.filt_hi[f] or x != x:
                    ok = False
            for k in range(nphi):
                var x = Float64(hphi[i * ncol + k])
                if x != x:
                    ok = False
                if i < nc:
                    cand[i * nphi + k] = x
                else:
                    refs[(i - nc) * nphi + k] = x
            if i < nc:
                cv[i] = ok
            else:
                rv[i - nc] = ok
        # 4. the inverse-density draw
        var picks = List[Int]()
        var rep = InverseDensityReport()
        var dcfg = self.cfg.density.copy()
        inverse_density_draw(
            cand, cv, nc, refs, rv, nr, nphi, Self.BATCH,
            dcfg, self._rng, picks, rep,
        )
        self.last_report = rep^
        var hp = c.enqueue_create_host_buffer[IDX_DT](Self.BATCH)
        for t in range(Self.BATCH):
            hp[t] = Scalar[IDX_DT](rows[picks[t]])
        c.enqueue_copy(self._idx_pick.value(), hp)
        # 5. those rows → (normalise) → B → sphere → zsrc
        c.enqueue_function[
            gather_rows_into_kernel[
                Self.STORE_OBS, Self.OBS, Self.BATCH, Self.RING_DT
            ]
        ](
            self.base.ring_obs_ptr(),
            mptr(self._idx_pick.value().unsafe_ptr()),
            mptr(self._pool_obs.dev.value().unsafe_ptr()),
            grid_dim=_blocks(Self.BATCH * Self.STORE_OBS), block_dim=TPB,
        )
        if self.base.obs_ema.enabled:
            self.base.obs_ema.apply[Self.BATCH](self._pool_obs)
        call_forward["gpu", Self.BATCH](
            self.base.t.bnet.online, TensorRefs[1, MutAnyOrigin](self._pool_obs),
            self.base.zsrc, self.ctx,
        )
        c.enqueue_function[project_sphere_kernel[Self.D, Self.BATCH]](
            mptr(self.base.zsrc.dev.value().unsafe_ptr()),
            Scalar[DT](sqrt(Float64(Self.D))),
            grid_dim=_blocks(Self.BATCH), block_dim=TPB,
        )
        self._n_refresh += 1

    # ── ring access for evals and diagnostics (eager) ──────────────────

    def embed_rows(
        mut self,
        ref rows: List[Int],
        mut raw: List[Scalar[DT]],
        mut b: List[Scalar[DT]],
    ) raises:
        """`B(s)` of ring rows, `BATCH` at a time, with the CURRENT
        normaliser and B: `raw` gets each row's stored (unnormalised)
        `STORE_OBS` columns — what a reward reads — and `b` its `D`-wide
        embedding, unprojected (what `z_from_reward` averages). The ring
        is the reward pool of a data-free arm (plan §2 B4)."""
        var c = self.ctx.value()
        if not self._e_idx:
            self._e_idx = c.enqueue_create_buffer[IDX_DT](Self.BATCH)
            ensure_t["gpu"](self._e_obs, Self.BATCH * Self.OBS, self.ctx)
            # the derived tail of a gathered row is never written; B's
            # filter reads the stored prefix
            self._e_obs.dev.value().enqueue_fill(Scalar[DT](0.0))
            ensure_t["gpu"](self._e_b, Self.BATCH * Self.D, self.ctx)
        var n = len(rows)
        for i in range(n):
            if rows[i] < 0 or rows[i] >= self.base.size:
                raise Error("embed_rows: row outside the ring's fill")
        raw = List[Scalar[DT]](length=n * Self.STORE_OBS, fill=Scalar[DT](0))
        b = List[Scalar[DT]](length=n * Self.D, fill=Scalar[DT](0))
        var hidx = c.enqueue_create_host_buffer[IDX_DT](Self.BATCH)
        var hraw = c.enqueue_create_host_buffer[DT](Self.BATCH * Self.OBS)
        var hb = c.enqueue_create_host_buffer[DT](Self.BATCH * Self.D)
        var done = 0
        while done < n:
            var take = Self.BATCH if n - done >= Self.BATCH else n - done
            for i in range(Self.BATCH):
                hidx[i] = Scalar[IDX_DT](rows[done + (i if i < take else 0)])
            c.enqueue_copy(self._e_idx.value(), hidx)
            c.enqueue_function[
                gather_rows_into_kernel[
                    Self.STORE_OBS, Self.OBS, Self.BATCH, Self.RING_DT
                ]
            ](
                self.base.ring_obs_ptr(),
                mptr(self._e_idx.value().unsafe_ptr()),
                mptr(self._e_obs.dev.value().unsafe_ptr()),
                grid_dim=_blocks(Self.BATCH * Self.STORE_OBS), block_dim=TPB,
            )
            c.enqueue_copy(hraw, self._e_obs.dev.value())
            if self.base.obs_ema.enabled:
                self.base.obs_ema.apply[Self.BATCH](self._e_obs)
            call_forward["gpu", Self.BATCH](
                self.base.t.bnet.online, TensorRefs[1, MutAnyOrigin](self._e_obs),
                self._e_b, self.ctx,
            )
            c.enqueue_copy(hb, self._e_b.dev.value())
            c.synchronize()
            for i in range(take):
                for k in range(Self.STORE_OBS):
                    raw[(done + i) * Self.STORE_OBS + k] = hraw[i * Self.OBS + k]
                for k in range(Self.D):
                    b[(done + i) * Self.D + k] = hb[i * Self.D + k]
            done += take

    def sample_ring_cols(
        mut self, ref cols: List[Int], n: Int, seed: UInt64
    ) raises -> List[Float64]:
        """`n` ring rows drawn uniformly over the fill, `len(cols)` stored
        columns each, row-major on the host — the coverage readouts (plan §4
        early kill checks). Allocates per call; never on a hot path."""
        var c = self.ctx.value()
        var size = self.base.size
        if size < 1:
            raise Error("sample_ring_cols: the ring is empty")
        var nc = len(cols)
        for k in range(nc):
            if cols[k] < 0 or cols[k] >= Self.STORE_OBS:
                raise Error("sample_ring_cols: column outside the stored obs")
        var rng = SplitMix64(seed)
        var hidx = c.enqueue_create_host_buffer[IDX_DT](n)
        for i in range(n):
            hidx[i] = Scalar[IDX_DT](rng.below(size))
        var didx = c.enqueue_create_buffer[IDX_DT](n)
        c.enqueue_copy(didx, hidx)
        var hcols = c.enqueue_create_host_buffer[DType.int32](nc)
        for k in range(nc):
            hcols[k] = Int32(cols[k])
        var dcols = c.enqueue_create_buffer[DType.int32](nc)
        c.enqueue_copy(dcols, hcols)
        var dout = c.enqueue_create_buffer[DT](n * nc)
        c.enqueue_function[gather_cols_kernel[Self.STORE_OBS, Self.RING_DT]](
            self.base.ring_obs_ptr(),
            mptr(didx.unsafe_ptr()), mptr(dcols.unsafe_ptr()),
            mptr(dout.unsafe_ptr()), Int32(n), Int32(nc),
            grid_dim=(n * nc + TPB - 1) // TPB, block_dim=TPB,
        )
        var hout = c.enqueue_create_host_buffer[DT](n * nc)
        c.enqueue_copy(hout, dout)
        c.synchronize()
        var out = List[Float64](length=n * nc, fill=0.0)
        for i in range(n * nc):
            out[i] = Float64(hout[i])
        return out^

    def policy_with_z(
        mut self,
        obs_ptr: Pointer[Scalar[DT], MutAnyOrigin],
        z_ptr: Pointer[Scalar[DT], MutAnyOrigin],
        action_ptr: Pointer[Scalar[DT], MutAnyOrigin],
    ) raises:
        """`FBOnlineAgent.policy_with_z` — greedy under a caller's z."""
        self.base.policy_with_z(obs_ptr, z_ptr, action_ptr)

    # ── the training step ────────────────────────────────────────────────

    def _train_kernels(mut self) raises:
        var c = self.ctx.value()
        self.base._sample_batch()
        if self.use_reg:
            c.enqueue_function[gather_rows_kernel[1, Self.BATCH]](
                mptr(self.r_rew.dev.value().unsafe_ptr()),
                mptr(self.base.idx_s.value().unsafe_ptr()),
                mptr(self.head.br.dev.value().unsafe_ptr()),
                grid_dim=_blocks(Self.BATCH), block_dim=TPB,
            )
        self.base._relabel_z()
        if self.use_reg:
            _ = self.head.step(self.base.t, want_loss=False)
        else:
            _ = self.base.t.train_step(want_loss=False)
        self.base._push_zbuf()
        comptime lba = RowMajor[Self.BATCH * Self.ACT]
        self.base._mean_abs_action_dev.accumulate_gpu_abs_lt[Self.BATCH * Self.ACT](
            self.base.t.ba.tt["gpu", Self.BATCH * Self.ACT]()
        )

    # ── OffPolicyAgentGpu: rollout surface ───────────────────────────────

    def select_action_batched[
        N_ENVS: Int
    ](
        mut self,
        obs: Tile[DT, N_ENVS, Self.AGENT_OBS_DIM],
        action: Tile[DT, N_ENVS, Self.AGENT_ACT_DIM],
        ao_scratch: Tile[DT, N_ENVS, 2 * Self.AGENT_ACT_DIM],
        alp_scratch: Tile[DT, N_ENVS, Self.AGENT_ACT_DIM + 1],
        step_idx: Int,
    ) raises:
        if (
            self.mebe
            and self._iter % self.cfg.refresh_every == 0
            and self.base.size >= 10 * Self.LANES
        ):
            self.refresh_pool()
        self._iter += 1
        self.base.select_action_batched[N_ENVS](
            obs, action, ao_scratch, alp_scratch, step_idx
        )

    def select_greedy_action_batched[
        N_ENVS: Int,
    ](
        mut self,
        ctx: Optional[DeviceContext],
        obs: Tile[DT, N_ENVS, Self.AGENT_OBS_DIM],
        action: Tile[DT, N_ENVS, Self.AGENT_ACT_DIM],
        ao_scratch: Tile[DT, N_ENVS, 2 * Self.AGENT_ACT_DIM],
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
        """The regularity reward FIRST, at the row the base is about to
        write (`base.pos`, `base.bnd_next` as they stand), then the base's
        own record, which advances both.

        ⚠ `prev_obs_dev` / `obs_dev` are `STORE_OBS`-strided, as the base's
        ring store reads them (the G1 driver passes the env's 527-D rows for
        a 928-D packed agent). The q̇ difference strides the same way."""
        comptime assert N_ENVS == Self.LANES, (
            "FBMebeOnlineAgent: the driver's N_ENVS must equal the agent's LANES"
        )
        if self.episode_len > 0 and (self._n_record + 1) % self.episode_len == 0:
            self.base.set_boundary(True)
        self._n_record += 1
        if self.use_reg:
            ctx.enqueue_function[
                reg_reward_kernel[Self.STORE_OBS, Self.ACT, Self.LANES, Self.CAP]
            ](
                mptr(prev_obs_dev.unsafe_ptr()),
                mptr(obs_dev.unsafe_ptr()),
                mptr(action_dev.unsafe_ptr()),
                mptr(done_dev.unsafe_ptr()),
                mptr(reward_dev.unsafe_ptr()),
                mptr(self._prev_act.dev.value().unsafe_ptr()),
                mptr(self.r_rew.dev.value().unsafe_ptr()),
                Int32(self.base.pos),
                Int32(1) if self.base.bnd_next else Int32(0),
                Int32(self.reg.qd_off),
                Int32(self.reg.qd_n),
                Scalar[DT](1.0 / self.reg.dt),
                Scalar[DT](self.reg.w_acc),
                Scalar[DT](self.reg.w_ar),
                Scalar[DT](self.reg.w_env),
                grid_dim=_blocks(Self.LANES), block_dim=TPB,
            )
        self.base.record_batch_gpu[N_ENVS](
            ctx, prev_obs_dev, action_dev, reward_dev, obs_dev, done_dev
        )

    def set_boundary(mut self, b: Bool):
        self.base.set_boundary(b)

    def set_age(mut self, a: Int):
        self.base.set_age(a)

    def attach_tail_spec(mut self, ref spec: List[Int32]) raises:
        self.base.attach_tail_spec(spec)

    def set_action_norm(mut self, scale: Float64, clip: Float64):
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
        raise Error("FBMebeOnlineAgent: n-step is meaningless for FB; run NS=1")

    # ── OffPolicyAgentGpu: training surface ──────────────────────────────

    def train_step(mut self, step_idx: Int) raises -> Bool:
        if step_idx < self.base.learning_starts or self.base.size < Self.BATCH:
            return False
        self._train_kernels()
        self.base._update_count += 1
        self.base._total_train_steps += 1
        return True

    def train_device_kernels(mut self) raises:
        """Capture body. Every branch is on a construction-time field
        (`use_reg`, `zsrc_on`, the head's `reg_coeff`)."""
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
        var names = List[String]()
        var vals = List[Float64]()
        if self.use_reg:
            var r_mean = Float64(0)
            var q_mean = Float64(0)
            var q_loss = Float64(0)
            var q_pi = Float64(0)
            self.head.read_diag(r_mean, q_mean, q_loss, q_pi)
            names.append(String("reg_reward_mean")); vals.append(r_mean)
            names.append(String("reg_q_mean")); vals.append(q_mean)
            names.append(String("reg_critic_loss")); vals.append(q_loss)
            names.append(String("reg_policy_q_mean")); vals.append(q_pi)
            print(
                "   [reg] step", step, " r", r_mean, " Q", q_mean,
                " Qloss", q_loss, " Qpi", q_pi,
            )
        if self.mebe:
            var rp = self.last_report.copy()
            names.append(String("mebe_entropy")); vals.append(rp.entropy)
            names.append(String("mebe_ess_frac")); vals.append(rp.ess_frac)
            names.append(String("mebe_max_w_share")); vals.append(rp.max_w_share)
            names.append(String("mebe_valid_frac"))
            vals.append(Float64(rp.n_valid_cand) / Float64(max(self.cfg.n_cand, 1)))
            names.append(String("mebe_refreshes")); vals.append(Float64(self._n_refresh))
            print(
                "   [mebe] step", step, " H(φ)", rp.entropy, " ess", rp.ess_frac,
                " max w", rp.max_w_share, " valid", rp.n_valid_cand, "/",
                self.cfg.n_cand, " refreshes", self._n_refresh,
            )
        if Bool(logger) and len(names) > 0:
            logger.value()[].log_scalars(names, vals, step)

    # ── checkpoint ───────────────────────────────────────────────────────

    def save_state(mut self, path: String) raises:
        """FB nets in `FBTrainer`'s own format (every FB eval loads it), the
        `.norm` sidecar when normalising, `Q_reg` in the `.reg` sidecar."""
        self.base.save_state(path)
        if self.use_reg:
            self.head.save_sidecar(path)

    def load_state(mut self, path: String) raises:
        self.base.load_state(path)
        if self.use_reg:
            self.head.load_sidecar(path)
