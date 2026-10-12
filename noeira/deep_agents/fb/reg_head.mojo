"""`FBRegHead` — FB-MEBE's regularity critic `Q_reg`, around an `FBTrainer`.

`noeira-docs/FB_MEBE_G1_PLAN.md` §2 B3; FB-MEBE (arXiv 2603.25464) eq. 6 and
appendix F-G. The naturalness prior uses NO data: a twin critic is
TD-trained on a hand-written regularity reward

    r_reg = −2.5e-7·‖q̈‖² − 0.1·‖a_t − a_{t−1}‖² − 0.1·Σ_feet c·‖v_xy‖

and the actor maximises `F(s,a,z)·z + λ·Q_reg`. Plain online FB with no prior
drags its feet on a quadruped (the paper's Fig. 2) and goes bang-bang on the
walker (our A3 run 1); this is the paper's replacement for the motion prior.

The head is the `Q_D` half of `FBCPRHead` (`cpr.mojo` steps 2-4) with the
discriminator removed and the reward supplied by the caller in `br`
(`[BATCH]`, gathered from the replay ring next to `s, a, s'`):

    1  a' = smooth(π̄(s', z)),   y = r_reg + γ · pess(Q̄1, Q̄2)(s', a', z)
       twin MSE vjps; clip; Adam      (Polyak after the FB step)
    2  g_pi_extra = −λ/BATCH · ∂ pess(Q1, Q2)/∂a  at (s, π(s, z), z)
    3  FBTrainer.train_step  (reads g_pi_extra in its actor step)

## Deviations from the released code, each deliberate

  * `Q_reg` is z-CONDITIONED, `Q(s, a, z)` on F's own `[s | a | z]` row,
    where the release's critic is `Q(s, a)`. The release's TD target already
    bootstraps through `π(s', z)` with the batch z, so its critic estimates a
    z-dependent quantity with a z-blind net (an average over the z mixture);
    BFM-Zero's auxiliary critic `Q_R` is z-conditioned for that reason, and
    this keeps one critic layout for both heads.
  * The λ scaling: `FBTrainer.pi_extra_raw` (set here at construction) adds
    the term RAW — the release's `actor_loss −= reg_coeff · Q_critic` with
    `reg_coeff 20`. `raw = False` is BFM-Zero's `λ·|Q_fb|·Q_x`. Under Adam
    only the ratio of the two gradients matters, and that ratio is logged
    (`actor_grad_value` → `actor_grad_total`): calibrate λ on it, do not
    copy the paper's number across a different `F·z` scale.

With `reg_coeff = 0` the actor never reads the head, and the four FB nets are
bit-identical to a plain `FBTrainer` on the same batch (gated:
`tests/deep_agents/test_fb_reg_head.mojo`).

Checkpoint: the FB nets go to `path` in `FBTrainer`'s own layout (every FB
eval reads it unchanged); the `Q_reg` twins to `path + ".reg"`.
"""

from noeira.nn.core.param import walk_params, ParamVisitorRef
from max.gpu.host import DeviceContext, DeviceBuffer
from std.os.path import exists as _path_exists

from noeira.nn.constants import DT
from noeira.nn.core.module import Module
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.call import call_forward, call_vjp
from noeira.nn.core.initializer import Initializer, Xavier
from noeira.nn.optimizer.adam import Adam
from noeira.nn.core.checkpoint import (
    BinaryCheckpointWriter, BinaryCheckpointReader,
    _read_file_bytes, _write_file_bytes,
)

from ..core.online_target_pair import OnlineTargetPair
from .trainer import FBTrainer, FBLosses
from .kernels import (
    ensure_t,
    pack2_t,
    pack3_t,
    axpy_t,
    scale_t,
    pessimism_blend_t,
    pessimism_row_weights_t,
    smooth_action_t,
    slice_cols_t,
    mean_into_t,
    gaussian_t,
    gaussian_dev_t,
    diff_scale_t,
    sq_diff_mean_into_t,
)


struct FBRegHead[
    FNET: Module,
    BNET: Module,
    ANET: Module,
    QNET: Module,
    OBS: Int,
    ACT: Int,
    D: Int,
    BATCH: Int,
    TARGET: StaticString = "cpu",
](Movable & Deinitable):
    # The release's `critic_pessimism_penalty` / `actor_pessimism_penalty`,
    # both 0.5: at an ensemble of two, the twin-min (`cpr.mojo` header).
    comptime CRITIC_PESSIMISM: Float64 = 0.5
    comptime ACTOR_PESSIMISM: Float64 = 0.5
    comptime A_IN: Int = Self.OBS + Self.D             # the actor's [s | z]
    comptime Q_IN: Int = Self.OBS + Self.ACT + Self.D  # [s | a | z], F's layout
    comptime _NA: Int = Self.BATCH * Self.ACT
    comptime Inner = FBTrainer[
        Self.FNET, Self.BNET, Self.ANET, Self.OBS, Self.ACT, Self.D,
        Self.BATCH, Self.TARGET,
    ]

    var q1n: OnlineTargetPair[Self.QNET]
    var q2n: OnlineTargetPair[Self.QNET]
    var opt_q1: Adam
    var opt_q2: Adam
    # the reward of each batch row, written by the caller before `step`
    var br: Tensor
    var a_next: Tensor
    var noise: Tensor
    var ain_t: Tensor
    var pi_t: Tensor
    var qin_t: Tensor
    var qt1: Tensor
    var qt2: Tensor
    var q_target: Tensor
    var qin: Tensor
    var q1: Tensor
    var q2: Tensor
    var cot_q1: Tensor
    var cot_q2: Tensor
    var sink_q: Tensor
    var ain: Tensor
    var pi: Tensor
    var qin_pi: Tensor
    var q_pi: Tensor
    var q_pi2: Tensor
    var q_pi_p: Tensor
    var w_q1: Tensor
    var w_q2: Tensor
    var gx2: Tensor
    var cot_pi: Tensor
    var g_qin: Tensor
    var acc_r: Tensor
    var acc_q: Tensor
    var acc_qloss: Tensor
    var acc_qpi: Tensor
    var ctx: Optional[DeviceContext]
    var gamma: Float64
    var tau_q: Float64
    var reg_coeff: Float64
    var raw: Bool
    var max_grad_norm: Float64
    var policy_noise: Float64
    var noise_clip: Float64
    var steps: Int
    var _rng_seed: UInt64
    var _rng_offset: UInt64
    var _rng_off_dev: Optional[DeviceBuffer[DType.uint64]]
    var _sized: Bool

    def __init__(out self):
        self.q1n = OnlineTargetPair[Self.QNET]()
        self.q2n = OnlineTargetPair[Self.QNET]()
        self.opt_q1 = Adam(lr=Scalar[DT](1e-4))
        self.opt_q2 = Adam(lr=Scalar[DT](1e-4))
        self.br = Tensor()
        self.a_next = Tensor()
        self.noise = Tensor()
        self.ain_t = Tensor()
        self.pi_t = Tensor()
        self.qin_t = Tensor()
        self.qt1 = Tensor()
        self.qt2 = Tensor()
        self.q_target = Tensor()
        self.qin = Tensor()
        self.q1 = Tensor()
        self.q2 = Tensor()
        self.cot_q1 = Tensor()
        self.cot_q2 = Tensor()
        self.sink_q = Tensor()
        self.ain = Tensor()
        self.pi = Tensor()
        self.qin_pi = Tensor()
        self.q_pi = Tensor()
        self.q_pi2 = Tensor()
        self.q_pi_p = Tensor()
        self.w_q1 = Tensor()
        self.w_q2 = Tensor()
        self.gx2 = Tensor()
        self.cot_pi = Tensor()
        self.g_qin = Tensor()
        self.acc_r = Tensor()
        self.acc_q = Tensor()
        self.acc_qloss = Tensor()
        self.acc_qpi = Tensor()
        self.ctx = None
        self.gamma = 0.98
        self.tau_q = 0.005
        self.reg_coeff = 0.0
        self.raw = True
        self.max_grad_norm = 0.0
        self.policy_noise = 0.2
        self.noise_clip = 0.3
        self.steps = 0
        self._rng_seed = UInt64(0x9E6)
        self._rng_offset = UInt64(0)
        self._rng_off_dev = None
        self._sized = False

    def __init__(out self, *, deinit move: Self):
        self.q1n = move.q1n^
        self.q2n = move.q2n^
        self.opt_q1 = move.opt_q1^
        self.opt_q2 = move.opt_q2^
        self.br = move.br^
        self.a_next = move.a_next^
        self.noise = move.noise^
        self.ain_t = move.ain_t^
        self.pi_t = move.pi_t^
        self.qin_t = move.qin_t^
        self.qt1 = move.qt1^
        self.qt2 = move.qt2^
        self.q_target = move.q_target^
        self.qin = move.qin^
        self.q1 = move.q1^
        self.q2 = move.q2^
        self.cot_q1 = move.cot_q1^
        self.cot_q2 = move.cot_q2^
        self.sink_q = move.sink_q^
        self.ain = move.ain^
        self.pi = move.pi^
        self.qin_pi = move.qin_pi^
        self.q_pi = move.q_pi^
        self.q_pi2 = move.q_pi2^
        self.q_pi_p = move.q_pi_p^
        self.w_q1 = move.w_q1^
        self.w_q2 = move.w_q2^
        self.gx2 = move.gx2^
        self.cot_pi = move.cot_pi^
        self.g_qin = move.g_qin^
        self.acc_r = move.acc_r^
        self.acc_q = move.acc_q^
        self.acc_qloss = move.acc_qloss^
        self.acc_qpi = move.acc_qpi^
        self.ctx = move.ctx^
        self.gamma = move.gamma
        self.tau_q = move.tau_q
        self.reg_coeff = move.reg_coeff
        self.raw = move.raw
        self.max_grad_norm = move.max_grad_norm
        self.policy_noise = move.policy_noise
        self.noise_clip = move.noise_clip
        self.steps = move.steps
        self._rng_seed = move._rng_seed
        self._rng_offset = move._rng_offset
        self._rng_off_dev = move._rng_off_dev^
        self._sized = move._sized

    @staticmethod
    def make[
        INIT: Initializer = Xavier
    ](
        ctx: Optional[DeviceContext] = None,
        *,
        lr_q: Float64 = 5e-5,
        gamma: Float64 = 0.98,
        tau_q: Float64 = 0.005,
        max_grad_norm: Float64 = 1.0,
        reg_coeff: Float64 = 20.0,
        raw: Bool = True,
        policy_noise: Float64 = 0.2,
        noise_clip: Float64 = 0.3,
        seed: UInt64 = UInt64(0x5EED),
    ) raises -> Self:
        """The release's Go2 values: `lr_critic 5e-5`, `critic_target_tau
        0.005`, `reg_coeff 20` added raw, `clip_grad_norm 1.0`."""
        comptime assert Self.TARGET == "cpu" or Self.TARGET == "gpu", (
            "FBRegHead: TARGET must be 'cpu' or 'gpu'"
        )
        comptime assert (
            Self.QNET.ARITY == 1 and Self.QNET.IN_DIMS[0] == Self.Q_IN
            and Self.QNET.OUT_DIM == 1
        ), "FBRegHead: QNET must map [OBS + ACT + D] -> 1"
        if Self.TARGET == "gpu" and not ctx:
            raise Error("FBRegHead.make: TARGET='gpu' requires a ctx")
        var s = Self()
        s.ctx = ctx
        s.q1n = OnlineTargetPair[Self.QNET].make[Self.TARGET, INIT](ctx)
        s.q2n = OnlineTargetPair[Self.QNET].make[Self.TARGET, INIT](ctx)
        s.opt_q1 = Adam(lr=Scalar[DT](lr_q))
        s.opt_q2 = Adam(lr=Scalar[DT](lr_q))
        comptime if Self.TARGET == "gpu":
            s.opt_q1.adopt[Self.TARGET, Self.QNET](s.q1n.online, ctx)
            s.opt_q2.adopt[Self.TARGET, Self.QNET](s.q2n.online, ctx)
        s.gamma = gamma
        s.tau_q = tau_q
        s.reg_coeff = reg_coeff
        s.raw = raw
        s.max_grad_norm = max_grad_norm
        s.policy_noise = policy_noise
        s.noise_clip = noise_clip
        s._rng_seed = seed + UInt64(0x9E6)
        return s^

    def _size_once(mut self, mut t: Self.Inner) raises:
        if self._sized:
            return
        comptime T = Self.TARGET
        var c = self.ctx
        t.ensure_sized()
        # Construction-time facts of the captured sequence.
        t.has_pi_extra = self.reg_coeff > 0.0
        t.pi_extra_raw = self.raw
        comptime if T == "gpu":
            var d = c.value()
            var ob = d.enqueue_create_buffer[DType.uint64](1)
            var oh = d.enqueue_create_host_buffer[DType.uint64](1)
            oh[0] = self._rng_offset
            d.enqueue_copy(ob, oh)
            d.synchronize()
            self._rng_off_dev = ob^
        ensure_t[T](self.br, Self.BATCH, c)
        ensure_t[T](self.a_next, Self._NA, c)
        ensure_t[T](self.noise, Self._NA, c)
        ensure_t[T](self.ain_t, Self.BATCH * Self.A_IN, c)
        ensure_t[T](self.pi_t, Self._NA, c)
        ensure_t[T](self.qin_t, Self.BATCH * Self.Q_IN, c)
        ensure_t[T](self.qt1, Self.BATCH, c)
        ensure_t[T](self.qt2, Self.BATCH, c)
        ensure_t[T](self.q_target, Self.BATCH, c)
        ensure_t[T](self.qin, Self.BATCH * Self.Q_IN, c)
        ensure_t[T](self.q1, Self.BATCH, c)
        ensure_t[T](self.q2, Self.BATCH, c)
        ensure_t[T](self.cot_q1, Self.BATCH, c)
        ensure_t[T](self.cot_q2, Self.BATCH, c)
        ensure_t[T](self.sink_q, Self.BATCH * Self.Q_IN, c)
        ensure_t[T](self.ain, Self.BATCH * Self.A_IN, c)
        ensure_t[T](self.pi, Self._NA, c)
        ensure_t[T](self.qin_pi, Self.BATCH * Self.Q_IN, c)
        ensure_t[T](self.q_pi, Self.BATCH, c)
        ensure_t[T](self.q_pi2, Self.BATCH, c)
        ensure_t[T](self.q_pi_p, Self.BATCH, c)
        ensure_t[T](self.w_q1, Self.BATCH, c)
        ensure_t[T](self.w_q2, Self.BATCH, c)
        ensure_t[T](self.gx2, Self._NA, c)
        ensure_t[T](self.cot_pi, Self.BATCH, c)
        ensure_t[T](self.g_qin, Self.BATCH * Self.Q_IN, c)
        ensure_t[T](self.acc_r, 1, c)
        ensure_t[T](self.acc_q, 1, c)
        ensure_t[T](self.acc_qloss, 1, c)
        ensure_t[T](self.acc_qpi, 1, c)
        self._sized = True

    def ensure_sized(mut self, mut t: Self.Inner) raises:
        """Size the head's scratch (and `t`'s batch) so a GPU caller can
        gather straight into `t.bs/ba/bsn/bsp` and `br`. ⚠ Sets
        `t.has_pi_extra` / `t.pi_extra_raw` — call BEFORE any capture."""
        self._size_once(t)

    # ── the step ─────────────────────────────────────────────────────────

    def step(
        mut self, mut t: Self.Inner, want_loss: Bool = True
    ) raises -> FBLosses:
        """One regularity-critic step around `t`'s. Precondition: the caller
        has written `t.bs/ba/bsn/bsp/bz` (z already relabelled) and `br`."""
        comptime T = Self.TARGET
        var c = self.ctx
        self._size_once(t)
        self.steps += 1
        var inv_b = Scalar[DT](1.0 / Float64(Self.BATCH))
        mean_into_t[T, Self.BATCH](self.br, self.acc_r, c)

        # ── 1. TD target and twin update ─────────────────────────────────
        pack2_t[T, Self.OBS, Self.D, Self.BATCH](self.ain_t, t.bsn, t.bz, c)
        call_forward[T, Self.BATCH](
            t.actor.target_net, TensorRefs[1, MutAnyOrigin](self.ain_t),
            self.pi_t, c,
        )
        comptime if T == "gpu":
            gaussian_dev_t[T, Self._NA](
                self.noise, self._rng_seed, self._rng_off_dev.value(), c
            )
        else:
            gaussian_t[T, Self._NA](self.noise, self._rng_seed, self._rng_offset, c)
        self._rng_offset += UInt64(Self._NA + (Self._NA % 2))
        smooth_action_t[T, Self._NA](
            self.a_next, self.pi_t, self.noise,
            Scalar[DT](self.policy_noise), Scalar[DT](self.noise_clip), c,
        )
        pack3_t[T, Self.OBS, Self.ACT, Self.D, Self.BATCH](
            self.qin_t, t.bsn, self.a_next, t.bz, c
        )
        call_forward[T, Self.BATCH](
            self.q1n.target_net, TensorRefs[1, MutAnyOrigin](self.qin_t), self.qt1, c
        )
        call_forward[T, Self.BATCH](
            self.q2n.target_net, TensorRefs[1, MutAnyOrigin](self.qin_t), self.qt2, c
        )
        pessimism_blend_t[T, Self.BATCH](
            self.q_target, self.qt1, self.qt2, Scalar[DT](self.gamma),
            Scalar[DT](Self.CRITIC_PESSIMISM), c,
        )
        axpy_t[T, Self.BATCH](self.q_target, self.br, Scalar[DT](1.0), c)

        pack3_t[T, Self.OBS, Self.ACT, Self.D, Self.BATCH](
            self.qin, t.bs, t.ba, t.bz, c
        )
        self.q1n.online.zero_grad[T](c)
        self.q2n.online.zero_grad[T](c)
        call_forward[T, Self.BATCH](
            self.q1n.online, TensorRefs[1, MutAnyOrigin](self.qin), self.q1, c
        )
        call_forward[T, Self.BATCH](
            self.q2n.online, TensorRefs[1, MutAnyOrigin](self.qin), self.q2, c
        )
        diff_scale_t[T, Self.BATCH](self.cot_q1, self.q1, self.q_target, inv_b, c)
        diff_scale_t[T, Self.BATCH](self.cot_q2, self.q2, self.q_target, inv_b, c)
        sq_diff_mean_into_t[T, Self.BATCH](self.q1, self.q_target, self.acc_qloss, c)
        mean_into_t[T, Self.BATCH](self.q1, self.acc_q, c)
        call_vjp[T, Self.BATCH](
            self.q1n.online, TensorRefs[1, MutAnyOrigin](self.qin), self.cot_q1,
            TensorRefs[1, MutAnyOrigin](self.sink_q), c,
        )
        call_vjp[T, Self.BATCH](
            self.q2n.online, TensorRefs[1, MutAnyOrigin](self.qin), self.cot_q2,
            TensorRefs[1, MutAnyOrigin](self.sink_q), c,
        )
        if self.max_grad_norm > 0.0:
            var mgn = Scalar[DT](self.max_grad_norm)
            self.opt_q1.clip_grads_device[T](self.q1n.online, mgn, c)
            self.opt_q2.clip_grads_device[T](self.q2n.online, mgn, c)
        self.opt_q1.step[T](self.q1n.online, c)
        self.opt_q2.step[T](self.q2n.online, c)

        # ── 2. the actor term: −λ/BATCH · ∂ pess(Q1, Q2)/∂a ──────────────
        if self.reg_coeff > 0.0:
            pack2_t[T, Self.OBS, Self.D, Self.BATCH](self.ain, t.bs, t.bz, c)
            call_forward[T, Self.BATCH](
                t.actor.online, TensorRefs[1, MutAnyOrigin](self.ain), self.pi, c
            )
            pack3_t[T, Self.OBS, Self.ACT, Self.D, Self.BATCH](
                self.qin_pi, t.bs, self.pi, t.bz, c
            )
            call_forward[T, Self.BATCH](
                self.q1n.online, TensorRefs[1, MutAnyOrigin](self.qin_pi), self.q_pi, c
            )
            call_forward[T, Self.BATCH](
                self.q2n.online, TensorRefs[1, MutAnyOrigin](self.qin_pi), self.q_pi2, c
            )
            pessimism_blend_t[T, Self.BATCH](
                self.q_pi_p, self.q_pi, self.q_pi2, Scalar[DT](1.0),
                Scalar[DT](Self.ACTOR_PESSIMISM), c,
            )
            mean_into_t[T, Self.BATCH](self.q_pi_p, self.acc_qpi, c)
            pessimism_row_weights_t[T, Self.BATCH](
                self.w_q1, self.w_q2, self.q_pi, self.q_pi2,
                Scalar[DT](Self.ACTOR_PESSIMISM), c,
            )
            var cq = Scalar[DT](-self.reg_coeff / Float64(Self.BATCH))
            # through Q_reg WITHOUT keeping its parameter grads (already
            # stepped above) — zeroed right after
            scale_t[T, Self.BATCH](self.cot_pi, self.w_q1, cq, c)
            call_vjp[T, Self.BATCH](
                self.q1n.online, TensorRefs[1, MutAnyOrigin](self.qin_pi), self.cot_pi,
                TensorRefs[1, MutAnyOrigin](self.g_qin), c,
            )
            self.q1n.online.zero_grad[T](c)
            slice_cols_t[T, Self.Q_IN, Self.OBS, Self.ACT, Self.BATCH](
                t.g_pi_extra, self.g_qin, c
            )
            scale_t[T, Self.BATCH](self.cot_pi, self.w_q2, cq, c)
            call_vjp[T, Self.BATCH](
                self.q2n.online, TensorRefs[1, MutAnyOrigin](self.qin_pi), self.cot_pi,
                TensorRefs[1, MutAnyOrigin](self.g_qin), c,
            )
            self.q2n.online.zero_grad[T](c)
            slice_cols_t[T, Self.Q_IN, Self.OBS, Self.ACT, Self.BATCH](
                self.gx2, self.g_qin, c
            )
            axpy_t[T, Self.BATCH * Self.ACT](
                t.g_pi_extra, self.gx2, Scalar[DT](1.0), c
            )

        # ── 3. the FB step, unchanged, reading `g_pi_extra` ──────────────
        var fb = t.train_step(want_loss)
        self.q1n.polyak_step[T](Scalar[DT](self.tau_q), c)
        self.q2n.polyak_step[T](Scalar[DT](self.tau_q), c)
        return fb^

    # ── diagnostics ──────────────────────────────────────────────────────

    def read_diag(
        mut self,
        mut r_mean: Float64, mut q_mean: Float64, mut q_loss: Float64,
        mut q_pi: Float64,
    ) raises:
        """The LAST step's terms from the device accumulators every step
        writes. D2Hs four 1-element buffers: FLUSH CADENCE ONLY. `q_pi` is 0
        when `reg_coeff` is 0 (that pass is skipped)."""
        r_mean = 0.0
        q_mean = 0.0
        q_loss = 0.0
        q_pi = 0.0
        if self.steps == 0:
            return
        comptime if Self.TARGET == "gpu":
            var d = self.ctx.value()
            self.acc_r.download(d)
            self.acc_q.download(d)
            self.acc_qloss.download(d)
            self.acc_qpi.download(d)
        r_mean = Float64(self.acc_r.data[0])
        q_mean = Float64(self.acc_q.data[0])
        q_loss = Float64(self.acc_qloss.data[0])
        q_pi = Float64(self.acc_qpi.data[0])

    # ── checkpoint ───────────────────────────────────────────────────────

    def save_sidecar(mut self, path: String) raises:
        """The `Q_reg` twins to `path + ".reg"` (v3 binary)."""
        var w = BinaryCheckpointWriter(save_moments=False)
        w.mode = 0
        walk_params[Self.TARGET](self.q1n.online, w, self.ctx, "qr1")
        walk_params[Self.TARGET](self.q2n.online, w, self.ctx, "qr2")
        w.mode = 1
        var _s1 = ParamVisitorRef.of[type_of(w), Self.TARGET](w)
        self.q1n.online.for_each_state[Self.TARGET](_s1, self.ctx, "qr1")
        var _s2 = ParamVisitorRef.of[type_of(w), Self.TARGET](w)
        self.q2n.online.for_each_state[Self.TARGET](_s2, self.ctx, "qr2")
        _write_file_bytes(path + ".reg", w.content)

    def load_sidecar(mut self, path: String) raises:
        """⚠ A missing sidecar RAISES: a resume that silently restarts `Q_reg`
        from random init would push the actor with a critic that has
        forgotten everything, and nothing in the log would say so."""
        var sp = path + ".reg"
        if not _path_exists(sp):
            raise Error("FBRegHead.load_sidecar: sidecar missing: " + sp)
        var rb = BinaryCheckpointReader(_read_file_bytes(sp))
        rb.mode = 0
        walk_params[Self.TARGET](self.q1n.online, rb, self.ctx, "qr1")
        walk_params[Self.TARGET](self.q2n.online, rb, self.ctx, "qr2")
        rb.mode = 1
        var _c1 = ParamVisitorRef.of[type_of(rb), Self.TARGET](rb)
        self.q1n.online.for_each_state[Self.TARGET](_c1, self.ctx, "qr1")
        var _c2 = ParamVisitorRef.of[type_of(rb), Self.TARGET](rb)
        self.q2n.online.for_each_state[Self.TARGET](_c2, self.ctx, "qr2")
        self.q1n.target_net.polyak_from[Self.TARGET](
            self.q1n.online, Scalar[DT](1.0), self.ctx
        )
        self.q2n.target_net.polyak_from[Self.TARGET](
            self.q2n.online, Scalar[DT](1.0), self.ctx
        )
