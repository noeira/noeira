"""`FBCPRTrainer` — FB-CPR (Meta Motivo / BFM-Zero) on top of `FBTrainer`.

`docs/BFM_ZERO_SHOT_RL.md` §16.2 and §18.3 A4. FB-CPR is FB plus three
things, none of which touch the FB arithmetic:

    D(s, z)   a z-CONDITIONED discriminator (§15.3: never `D(s)`) trained
              with BCE-with-logits to tell EXPERT pairs `(s_e, z_e)` from the
              policy's / dataset's pairs `(s, z)`, plus a WGAN gradient
              penalty on interpolations of BOTH `s` and `z` (coef 10).
    Q_D       a separate twin "style" critic, Bellman-trained on
              `r_D(s, z) = log D − log(1 − D) = clamp(logit)`. It is NOT
              folded into `F`.
    actor     `−Q_fb − reg_coeff · |Q_fb| · Q_D` — the FB actor loss plus the
              style term, scaled by the detached magnitude of the FB term
              (BFM-Zero's `scale_reg`; the same mechanism TD3+BC's adaptive
              scale uses here for BC).

The expert `z` is a SEQUENCE encoding: `project(mean_j B(s'_{t+j}))` over
`SEQ` consecutive next-states of an expert window, repeated onto every
row of the window (`encode_expert`). That is the reference's
`encode_expert`, and it is also the mechanism tracking needs (§16.4).

## Composition, not a second body

`FBTrainer` is OWNED and UNCHANGED: its step runs last, over the same
owned batch (`t.bs / ba / bsn / bsp / bz`), and the style term enters
through the ONE additive hook `FBTrainer.g_pi_extra`. With `reg_coeff = 0`
the FB half is bit-identical to the plain trainer (gated); with `gp_coef`
and `reg_coeff` at the reference's values it is FB-CPR. Every scratch
buffer is owned and sized once, no host read sits on the
`want_loss=False` path, and the RNG offsets live on device — so
`train_device_kernels` is capturable exactly as the FB step is.

## The step

    1  D:   x_pos = [s_e | z_e],  x_neg = [s | z]
            zero;  GP on lerp(x_pos, x_neg, α~U) FIRST (it zeroes the probe
            grads);  BCE(l_pos, 1) + BCE(l_neg, 0) vjps;  Adam
    2  r_D = clamp(D(x_neg), ±16.118)                (post-update D)
       Q_D: a' = π̄(s', z) + noise,  y = r_D + γ · min(Q̄1, Q̄2)(s', a', z)
            twin MSE vjps;  Adam;  (Polyak after the FB step)
    3  g_pi_extra = −reg_coeff/BATCH · ∂Q_D1/∂a at (s, π(s, z), z)
    4  FBTrainer.train_step  (reads g_pi_extra inside its actor step)

⚠ The negatives are the TRAINING rows with the mixture `z` the caller wrote
into `t.bz` (offline: every row is relabelled, the mixture IS the z), and
the positives are expert rows with THEIR OWN window encoding. Offline the
expert rows are a subset of the training rows, so what `D` can learn is
the COUPLING — "is `s` on the trajectory `z` encodes" — which is the
conditional signal §15.3 asks for and the reason `D(s)` alone is worse
than nothing.

Every ensemble read here is the reference's `mean − 0.5·spread`, which at two
members is exactly `min`: `Q_D`'s TD target (`CRITIC_PESSIMISM`), `Q_D`'s actor
gradient (`ACTOR_PESSIMISM`, §12.20) and the FB actor's `Q_fb`
(`actor_pessimism`, §12.19). Both of the latter two went through twin 1 alone
until then — the twins were trained and half-consulted, so one critic had two
readers that disagreed about what it meant, and the actor always got the
optimistic one.

Checkpoint: `save_state(p)` writes the FB nets to `p` in `FBTrainer`'s own
layout (so `fb_eval_walker_online.mojo` loads it unchanged) and `D` +
`Q_D` to the sidecar `p + ".cpr"`.
"""

from noeira.nn.core.param import walk_params, ParamVisitorRef
from max.gpu.host import DeviceContext, DeviceBuffer
from std.math import sqrt
from std.random import random_float64
from std.os.path import exists as _path_exists

from noeira.nn.constants import DT
from noeira.nn.core.module import Module
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.call import call_forward, call_vjp
from noeira.nn.core.initializer import Initializer, Xavier
from noeira.nn.optimizer.adam import Adam
from noeira.nn.core.checkpoint import (
    CheckpointReader, _split_lines,
    BinaryCheckpointWriter, BinaryCheckpointReader,
    _read_file_bytes, _write_file_bytes, _is_v3_header,
)
from noeira.nn.loss.grad_penalty import GradPenalty
from noeira.nn.loss.bce_logits import bce_logits_const_t

from ..core.online_target_pair import OnlineTargetPair
from .trainer import FBTrainer, FBLosses
from .kernels import (
    ensure_t,
    pack2_t,
    pack3_t,
    axpy_t,
    scale_t,
    fill_t,
    pessimism_blend_t,
    pessimism_row_weights_t,
    smooth_action_t,
    slice_cols_t,
    mean_into_t,
    gaussian_t,
    gaussian_dev_t,
    uniform01_dev_t,
    window_mean_t,
    project_sphere_t,
    lerp_rows_t,
    clamp_t,
    diff_scale_t,
    sq_diff_mean_into_t,
)

# `log(eps) − log(1 − eps)` at the reference's `eps = 1e-7`: the reward is
# the logit, clamped to what a probability in `[eps, 1 − eps]` can express.
comptime R_D_CLAMP: Float64 = 16.11809565


struct FBCPRLosses(Movable & Deinitable):
    """`fb` is the inner trainer's report; the rest are the CPR terms. All
    zero when `want_loss=False`."""

    var fb: FBLosses
    var d_pos: Float64      # mean −log σ(D) on expert pairs
    var d_neg: Float64      # mean softplus(D) on training pairs
    var d_gp: Float64       # coef · mean (‖∇D‖ − 1)²
    var r_mean: Float64     # mean r_D over the training rows
    var q_mean: Float64     # mean Q_D1(s, a, z) on the batch
    var q_loss: Float64     # twin-1 TD MSE
    var q_pi: Float64       # mean Q_D1(s, π(s,z), z) — the actor's style value

    def __init__(
        out self, var fb: FBLosses, d_pos: Float64, d_neg: Float64,
        d_gp: Float64, r_mean: Float64, q_mean: Float64, q_loss: Float64,
        q_pi: Float64,
    ):
        self.fb = fb^
        self.d_pos = d_pos
        self.d_neg = d_neg
        self.d_gp = d_gp
        self.r_mean = r_mean
        self.q_mean = q_mean
        self.q_loss = q_loss
        self.q_pi = q_pi

    def __init__(out self, *, deinit move: Self):
        self.fb = move.fb^
        self.d_pos = move.d_pos
        self.d_neg = move.d_neg
        self.d_gp = move.d_gp
        self.r_mean = move.r_mean
        self.q_mean = move.q_mean
        self.q_loss = move.q_loss
        self.q_pi = move.q_pi


struct FBCPRHead[
    FNET: Module,
    BNET: Module,
    ANET: Module,
    DNET: Module,
    QNET: Module,
    OBS: Int,
    ACT: Int,
    D: Int,
    BATCH: Int,
    SEQ: Int,
    TARGET: StaticString = "cpu",
](Movable & Deinitable):
    comptime NW: Int = Self.BATCH // Self.SEQ
    # `critic_pessimism_penalty` (`train.py:652`). At an ensemble of two the
    # reference's uncertainty reduction makes 0.5 exactly the twin-min.
    comptime CRITIC_PESSIMISM: Float64 = 0.5
    # `actor_pessimism_penalty` (`train.py:639`). The actor reads Q_D through
    # the SAME reduction as Q_D's own target — one critic, one meaning.
    comptime ACTOR_PESSIMISM: Float64 = 0.5
    comptime D_IN: Int = Self.OBS + Self.D          # [s | z]
    comptime Q_IN: Int = Self.OBS + Self.ACT + Self.D  # [s | a | z], F's layout
    comptime _NA: Int = Self.BATCH * Self.ACT
    comptime _ND: Int = Self.BATCH * Self.D
    comptime Inner = FBTrainer[
        Self.FNET, Self.BNET, Self.ANET, Self.OBS, Self.ACT, Self.D,
        Self.BATCH, Self.TARGET,
    ]

    var disc: Self.DNET
    var opt_d: Adam
    var qd1: OnlineTargetPair[Self.QNET]
    var qd2: OnlineTargetPair[Self.QNET]
    var opt_q1: Adam
    var opt_q2: Adam
    var gp: GradPenalty[Self.D_IN, Self.BATCH]
    # ── owned expert batch: rows [w·SEQ + j] are window w, offset j ────
    #   es   expert s        [BATCH, OBS]   (D's positives)
    #   esn  expert s'       [BATCH, OBS]   (what the window encoding reads)
    #   ez   window encoding [BATCH, D]     (on the sphere, repeated per row)
    var z_neg: Tensor       # [BATCH, D] the z D's negatives carry when `use_z_neg`
    var use_z_neg: Bool
    var es: Tensor
    var esn: Tensor
    var ez: Tensor
    var b_esn: Tensor
    var x_pos: Tensor
    var x_neg: Tensor
    var x_int: Tensor
    var l_pos: Tensor
    var l_neg: Tensor
    var cot_pos: Tensor
    var cot_neg: Tensor
    var loss_pos: Tensor
    var loss_neg: Tensor
    var alpha: Tensor
    var sink_d: Tensor
    var r_d: Tensor
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
    var acc_dpos: Tensor
    var acc_dneg: Tensor
    var acc_r: Tensor
    var acc_q: Tensor
    var acc_qloss: Tensor
    var acc_qpi: Tensor
    var ctx: Optional[DeviceContext]
    var gamma: Float64
    var tau_q: Float64
    var reg_coeff: Float64
    var gp_coef: Float64
    var max_grad_norm: Float64
    var policy_noise: Float64
    var noise_clip: Float64
    var steps: Int
    var _rng_seed: UInt64
    var _rng_offset: UInt64
    var _rng_off_dev: Optional[DeviceBuffer[DType.uint64]]
    var _sized: Bool

    def __init__(out self):
        self.disc = Self.DNET()
        self.opt_d = Adam(lr=Scalar[DT](1e-5))
        self.qd1 = OnlineTargetPair[Self.QNET]()
        self.qd2 = OnlineTargetPair[Self.QNET]()
        self.opt_q1 = Adam(lr=Scalar[DT](1e-4))
        self.opt_q2 = Adam(lr=Scalar[DT](1e-4))
        self.gp = GradPenalty[Self.D_IN, Self.BATCH]()
        self.z_neg = Tensor()
        self.use_z_neg = False
        self.es = Tensor()
        self.esn = Tensor()
        self.ez = Tensor()
        self.b_esn = Tensor()
        self.x_pos = Tensor()
        self.x_neg = Tensor()
        self.x_int = Tensor()
        self.l_pos = Tensor()
        self.l_neg = Tensor()
        self.cot_pos = Tensor()
        self.cot_neg = Tensor()
        self.loss_pos = Tensor()
        self.loss_neg = Tensor()
        self.alpha = Tensor()
        self.sink_d = Tensor()
        self.r_d = Tensor()
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
        self.acc_dpos = Tensor()
        self.acc_dneg = Tensor()
        self.acc_r = Tensor()
        self.acc_q = Tensor()
        self.acc_qloss = Tensor()
        self.acc_qpi = Tensor()
        self.ctx = None
        self.gamma = 0.98
        self.tau_q = 0.005
        self.reg_coeff = 0.01
        self.gp_coef = 10.0
        self.max_grad_norm = 0.0
        self.policy_noise = 0.2
        self.noise_clip = 0.3
        self.steps = 0
        self._rng_seed = UInt64(0xC9A)
        self._rng_offset = UInt64(0)
        self._rng_off_dev = None
        self._sized = False

    def __init__(out self, *, deinit move: Self):
        self.disc = move.disc^
        self.opt_d = move.opt_d^
        self.qd1 = move.qd1^
        self.qd2 = move.qd2^
        self.opt_q1 = move.opt_q1^
        self.opt_q2 = move.opt_q2^
        self.gp = move.gp^
        self.z_neg = move.z_neg^
        self.use_z_neg = move.use_z_neg
        self.es = move.es^
        self.esn = move.esn^
        self.ez = move.ez^
        self.b_esn = move.b_esn^
        self.x_pos = move.x_pos^
        self.x_neg = move.x_neg^
        self.x_int = move.x_int^
        self.l_pos = move.l_pos^
        self.l_neg = move.l_neg^
        self.cot_pos = move.cot_pos^
        self.cot_neg = move.cot_neg^
        self.loss_pos = move.loss_pos^
        self.loss_neg = move.loss_neg^
        self.alpha = move.alpha^
        self.sink_d = move.sink_d^
        self.r_d = move.r_d^
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
        self.acc_dpos = move.acc_dpos^
        self.acc_dneg = move.acc_dneg^
        self.acc_r = move.acc_r^
        self.acc_q = move.acc_q^
        self.acc_qloss = move.acc_qloss^
        self.acc_qpi = move.acc_qpi^
        self.ctx = move.ctx^
        self.gamma = move.gamma
        self.tau_q = move.tau_q
        self.reg_coeff = move.reg_coeff
        self.gp_coef = move.gp_coef
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
        lr_d: Float64 = 1e-5,
        lr_q: Float64 = 1e-4,
        gamma: Float64 = 0.98,
        tau_q: Float64 = 0.005,
        max_grad_norm: Float64 = 1.0,
        reg_coeff: Float64 = 0.01,
        gp_coef: Float64 = 10.0,
        gp_eps: Float64 = 1e-2,
        policy_noise: Float64 = 0.2,
        noise_clip: Float64 = 0.3,
        seed: UInt64 = UInt64(0x5EED),
    ) raises -> Self:
        """BFM-Zero's released CPR values (`fb_cpr/configs.py`:
        `lr_discriminator 1e-5`, `lr_critic 1e-4`, `critic_target_tau 0.005`,
        `reg_coeff 0.01`, `grad_penalty_discriminator 10`). The head does
        not own the FB trainer: `step(t)` runs around whichever `FBTrainer`
        it is handed — the offline `FBCPRTrainer`'s or the online agent's."""
        comptime assert Self.TARGET == "cpu" or Self.TARGET == "gpu", (
            "FBCPRHead: TARGET must be 'cpu' or 'gpu'"
        )
        comptime assert Self.BATCH % Self.SEQ == 0, (
            "FBCPRHead: BATCH must be a multiple of SEQ (whole expert windows)"
        )
        comptime assert (
            Self.DNET.ARITY == 1 and Self.DNET.IN_DIMS[0] == Self.D_IN
            and Self.DNET.OUT_DIM == 1
        ), (
            "FBCPRHead: DNET must map [OBS + D] -> 1"
        )
        comptime assert (
            Self.QNET.ARITY == 1 and Self.QNET.IN_DIMS[0] == Self.Q_IN
            and Self.QNET.OUT_DIM == 1
        ), (
            "FBCPRHead: QNET must map [OBS + ACT + D] -> 1"
        )
        if Self.TARGET == "gpu" and not ctx:
            raise Error("FBCPRHead.make: TARGET='gpu' requires a ctx")
        var s = Self()
        s.ctx = ctx
        s.disc = Self.DNET.make[Self.TARGET, INIT](ctx)
        s.qd1 = OnlineTargetPair[Self.QNET].make[Self.TARGET, INIT](ctx)
        s.qd2 = OnlineTargetPair[Self.QNET].make[Self.TARGET, INIT](ctx)
        s.opt_d = Adam(lr=Scalar[DT](lr_d))
        s.opt_q1 = Adam(lr=Scalar[DT](lr_q))
        s.opt_q2 = Adam(lr=Scalar[DT](lr_q))
        comptime if Self.TARGET == "gpu":
            # Same precondition as the inner trainer's: a captured Adam must
            # read beta^t from device (`adam.mojo`).
            s.opt_d.adopt[Self.TARGET, Self.DNET](s.disc, ctx)
            s.opt_q1.adopt[Self.TARGET, Self.QNET](s.qd1.online, ctx)
            s.opt_q2.adopt[Self.TARGET, Self.QNET](s.qd2.online, ctx)
        s.gp = GradPenalty[Self.D_IN, Self.BATCH].make[Self.TARGET](
            ctx, gp_eps, 1.0
        )
        s.gamma = gamma
        s.tau_q = tau_q
        s.reg_coeff = reg_coeff
        s.gp_coef = gp_coef
        s.max_grad_norm = max_grad_norm
        s.policy_noise = policy_noise
        s.noise_clip = noise_clip
        s._rng_seed = seed + UInt64(0xC9A)
        return s^

    def _size_once(mut self, mut t: Self.Inner) raises:
        if self._sized:
            return
        comptime T = Self.TARGET
        var c = self.ctx
        t.ensure_sized()
        # The hook is a construction-time fact of the captured sequence.
        t.has_pi_extra = self.reg_coeff > 0.0
        comptime if T == "gpu":
            var d = c.value()
            var ob = d.enqueue_create_buffer[DType.uint64](1)
            var oh = d.enqueue_create_host_buffer[DType.uint64](1)
            oh[0] = self._rng_offset
            d.enqueue_copy(ob, oh)
            d.synchronize()
            self._rng_off_dev = ob^
        comptime NO = Self.BATCH * Self.OBS
        ensure_t[T](self.z_neg, Self._ND, c)
        ensure_t[T](self.es, NO, c)
        ensure_t[T](self.esn, NO, c)
        ensure_t[T](self.ez, Self._ND, c)
        ensure_t[T](self.b_esn, Self._ND, c)
        ensure_t[T](self.x_pos, Self.BATCH * Self.D_IN, c)
        ensure_t[T](self.x_neg, Self.BATCH * Self.D_IN, c)
        ensure_t[T](self.x_int, Self.BATCH * Self.D_IN, c)
        ensure_t[T](self.l_pos, Self.BATCH, c)
        ensure_t[T](self.l_neg, Self.BATCH, c)
        ensure_t[T](self.cot_pos, Self.BATCH, c)
        ensure_t[T](self.cot_neg, Self.BATCH, c)
        ensure_t[T](self.loss_pos, Self.BATCH, c)
        ensure_t[T](self.loss_neg, Self.BATCH, c)
        ensure_t[T](self.alpha, Self.BATCH, c)
        ensure_t[T](self.sink_d, Self.BATCH * Self.D_IN, c)
        ensure_t[T](self.r_d, Self.BATCH, c)
        ensure_t[T](self.a_next, Self._NA, c)
        ensure_t[T](self.noise, Self._NA, c)
        ensure_t[T](self.ain_t, Self.BATCH * Self.D_IN, c)
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
        ensure_t[T](self.ain, Self.BATCH * Self.D_IN, c)
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
        ensure_t[T](self.acc_dpos, 1, c)
        ensure_t[T](self.acc_dneg, 1, c)
        ensure_t[T](self.acc_r, 1, c)
        ensure_t[T](self.acc_q, 1, c)
        ensure_t[T](self.acc_qloss, 1, c)
        ensure_t[T](self.acc_qpi, 1, c)
        self._sized = True

    def ensure_sized(mut self, mut t: Self.Inner) raises:
        """Size the head's scratch (and `t`'s owned batch) so a GPU caller
        can gather straight into `t.bs/ba/bsn/bsp` and `es/esn`.
        ⚠ Sets `t.has_pi_extra` — call BEFORE any capture."""
        self._size_once(t)

    def load_expert(mut self, mut s: Tensor, mut s_next: Tensor) raises:
        """Host convenience: copy an expert batch (`[BATCH, OBS]` each, rows
        grouped in `SEQ`-long windows) into `es` / `esn`."""
        comptime T = Self.TARGET
        if not self._sized:
            raise Error("FBCPRHead.load_expert: call ensure_sized(t) first")
        scale_t[T, Self.BATCH * Self.OBS](self.es, s, Scalar[DT](1.0), self.ctx)
        scale_t[T, Self.BATCH * Self.OBS](self.esn, s_next, Scalar[DT](1.0), self.ctx)

    def encode_expert(mut self, mut t: Self.Inner) raises:
        """`ez[w·SEQ + j] = project(mean_j' B(esn[w·SEQ + j']))` — the expert
        window encoding, through the ONLINE `B` (no gradient: nothing reads
        `B`'s grads before the inner step zeroes them). Call BEFORE building
        the `z` mixture that draws from it, and before `step`."""
        comptime T = Self.TARGET
        var c = self.ctx
        self._size_once(t)
        call_forward[T, Self.BATCH](
            t.bnet.online, TensorRefs[1, MutAnyOrigin](self.esn),
            self.b_esn, c,
        )
        window_mean_t[T, Self.SEQ, Self.D, Self.NW](self.ez, self.b_esn, c)
        project_sphere_t[T, Self.D, Self.BATCH](self.ez, c)

    # ── the step ─────────────────────────────────────────────────────────

    def step(
        mut self, mut t: Self.Inner, want_loss: Bool = True
    ) raises -> FBCPRLosses:
        """One CPR step around `t`'s. Precondition: the caller has written
        `t.bs/ba/bsn/bsp/bz` and `es/esn`, called `encode_expert(t)` (so
        `ez` matches `esn`), and — online — copied the STORED z of the batch
        into `z_neg` with `use_z_neg` set, BEFORE relabelling `t.bz`. D's
        negatives then carry the z the rows were rolled out under (the
        reference's `train_z`), while the reward, the critic and the actor
        read the relabelled `t.bz`. Offline both are `t.bz`."""
        comptime T = Self.TARGET
        var c = self.ctx
        self._size_once(t)
        self.steps += 1
        var inv_b = Scalar[DT](1.0 / Float64(Self.BATCH))

        # ── 1. discriminator ─────────────────────────────────────────────
        pack2_t[T, Self.OBS, Self.D, Self.BATCH](self.x_pos, self.es, self.ez, c)
        if self.use_z_neg:
            pack2_t[T, Self.OBS, Self.D, Self.BATCH](self.x_neg, t.bs, self.z_neg, c)
        else:
            pack2_t[T, Self.OBS, Self.D, Self.BATCH](self.x_neg, t.bs, t.bz, c)
        self.disc.zero_grad[T](c)
        var l_gp = Float64(0)
        if self.gp_coef > 0.0:
            # ⚠ FIRST: `apply` zeroes the probe's parameter grads inside.
            # A BCE vjp before this line is silently dropped (§18.8).
            comptime if T == "gpu":
                uniform01_dev_t[T, Self.BATCH](
                    self.alpha, self._rng_seed + 7, self._rng_off_dev.value(), c
                )
            else:
                for i in range(Self.BATCH):
                    self.alpha.data[i] = Scalar[DT](random_float64())
            lerp_rows_t[T, Self.BATCH, Self.D_IN](
                self.x_int, self.x_pos, self.x_neg, self.alpha, c
            )
            l_gp = self.gp.apply[T, Self.DNET](
                self.disc, self.x_int, self.gp_coef, want_loss
            )
        call_forward[T, Self.BATCH](
            self.disc, TensorRefs[1, MutAnyOrigin](self.x_pos), self.l_pos, c
        )
        bce_logits_const_t[T, Self.BATCH](
            self.l_pos, 1.0, Float64(inv_b), self.cot_pos, self.loss_pos, c
        )
        call_vjp[T, Self.BATCH](
            self.disc, TensorRefs[1, MutAnyOrigin](self.x_pos), self.cot_pos,
            TensorRefs[1, MutAnyOrigin](self.sink_d), c,
        )
        call_forward[T, Self.BATCH](
            self.disc, TensorRefs[1, MutAnyOrigin](self.x_neg), self.l_neg, c
        )
        bce_logits_const_t[T, Self.BATCH](
            self.l_neg, 0.0, Float64(inv_b), self.cot_neg, self.loss_neg, c
        )
        call_vjp[T, Self.BATCH](
            self.disc, TensorRefs[1, MutAnyOrigin](self.x_neg), self.cot_neg,
            TensorRefs[1, MutAnyOrigin](self.sink_d), c,
        )
        mean_into_t[T, Self.BATCH](self.loss_pos, self.acc_dpos, c)
        mean_into_t[T, Self.BATCH](self.loss_neg, self.acc_dneg, c)
        self.opt_d.step[T](self.disc, c)

        # ── 2. style reward from the UPDATED D on (s, z_relabelled) ──────
        pack2_t[T, Self.OBS, Self.D, Self.BATCH](self.x_neg, t.bs, t.bz, c)
        call_forward[T, Self.BATCH](
            self.disc, TensorRefs[1, MutAnyOrigin](self.x_neg), self.l_neg, c
        )
        clamp_t[T, Self.BATCH](
            self.r_d, self.l_neg, Scalar[DT](-R_D_CLAMP), Scalar[DT](R_D_CLAMP), c
        )
        mean_into_t[T, Self.BATCH](self.r_d, self.acc_r, c)

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
            self.qd1.target_net, TensorRefs[1, MutAnyOrigin](self.qin_t), self.qt1, c
        )
        call_forward[T, Self.BATCH](
            self.qd2.target_net, TensorRefs[1, MutAnyOrigin](self.qin_t), self.qt2, c
        )
        # `critic_pessimism_penalty` 0.5, which at an ensemble of two IS the
        # twin-min — the same reduction as the FB target, a DIFFERENT penalty.
        # Q_D is a value, so pessimism belongs here; see `pessimism_blend_t`.
        pessimism_blend_t[T, Self.BATCH](
            self.q_target, self.qt1, self.qt2, Scalar[DT](self.gamma),
            Scalar[DT](Self.CRITIC_PESSIMISM), c,
        )
        axpy_t[T, Self.BATCH](self.q_target, self.r_d, Scalar[DT](1.0), c)

        pack3_t[T, Self.OBS, Self.ACT, Self.D, Self.BATCH](
            self.qin, t.bs, t.ba, t.bz, c
        )
        self.qd1.online.zero_grad[T](c)
        self.qd2.online.zero_grad[T](c)
        call_forward[T, Self.BATCH](
            self.qd1.online, TensorRefs[1, MutAnyOrigin](self.qin), self.q1, c
        )
        call_forward[T, Self.BATCH](
            self.qd2.online, TensorRefs[1, MutAnyOrigin](self.qin), self.q2, c
        )
        # `0.5 · Σ_twins mean (Q − y)²` → cotangent `(Q − y) / BATCH` per twin.
        diff_scale_t[T, Self.BATCH](self.cot_q1, self.q1, self.q_target, inv_b, c)
        diff_scale_t[T, Self.BATCH](self.cot_q2, self.q2, self.q_target, inv_b, c)
        sq_diff_mean_into_t[T, Self.BATCH](self.q1, self.q_target, self.acc_qloss, c)
        mean_into_t[T, Self.BATCH](self.q1, self.acc_q, c)
        call_vjp[T, Self.BATCH](
            self.qd1.online, TensorRefs[1, MutAnyOrigin](self.qin), self.cot_q1,
            TensorRefs[1, MutAnyOrigin](self.sink_q), c,
        )
        call_vjp[T, Self.BATCH](
            self.qd2.online, TensorRefs[1, MutAnyOrigin](self.qin), self.cot_q2,
            TensorRefs[1, MutAnyOrigin](self.sink_q), c,
        )
        if self.max_grad_norm > 0.0:
            var mgn = Scalar[DT](self.max_grad_norm)
            self.opt_q1.clip_grads_device[T](self.qd1.online, mgn, c)
            self.opt_q2.clip_grads_device[T](self.qd2.online, mgn, c)
        self.opt_q1.step[T](self.qd1.online, c)
        self.opt_q2.step[T](self.qd2.online, c)

        # ── 3. the style term for the actor: −reg/BATCH · ∂Q_D/∂a ───────
        # Same weights the inner actor step will forward, so the two π agree.
        # Q_D is the PESSIMISTIC reduction over its twin, which at
        # `ACTOR_PESSIMISM` 0.5 is `min(Q_D1, Q_D2)` (`agent.py:266`). This
        # read Q_D1 ALONE until §12.20 — the twin was Bellman-trained and its
        # own TARGET already used the min, so one critic had two readers that
        # disagreed about what "Q_D" meant, and the actor's was the optimistic
        # one.
        if self.reg_coeff > 0.0:
            pack2_t[T, Self.OBS, Self.D, Self.BATCH](self.ain, t.bs, t.bz, c)
            call_forward[T, Self.BATCH](
                t.actor.online, TensorRefs[1, MutAnyOrigin](self.ain), self.pi, c
            )
            pack3_t[T, Self.OBS, Self.ACT, Self.D, Self.BATCH](
                self.qin_pi, t.bs, self.pi, t.bz, c
            )
            call_forward[T, Self.BATCH](
                self.qd1.online, TensorRefs[1, MutAnyOrigin](self.qin_pi), self.q_pi, c
            )
            call_forward[T, Self.BATCH](
                self.qd2.online, TensorRefs[1, MutAnyOrigin](self.qin_pi), self.q_pi2, c
            )
            pessimism_blend_t[T, Self.BATCH](
                self.q_pi_p, self.q_pi, self.q_pi2, Scalar[DT](1.0),
                Scalar[DT](Self.ACTOR_PESSIMISM), c,
            )
            mean_into_t[T, Self.BATCH](self.q_pi_p, self.acc_qpi, c)
            # per-row share of that reduction: at 0.5 the whole gradient goes
            # to whichever twin is the min on that row
            pessimism_row_weights_t[T, Self.BATCH](
                self.w_q1, self.w_q2, self.q_pi, self.q_pi2,
                Scalar[DT](Self.ACTOR_PESSIMISM), c,
            )
            # the cotangent was a CONSTANT fill; it is that constant scaled by
            # the row's weight now, so no new kernel is needed
            var cq = Scalar[DT](-self.reg_coeff / Float64(Self.BATCH))
            # Through Q_D WITHOUT keeping its parameter grads (already
            # stepped above) — zeroed right after, as F1 is in the inner step.
            scale_t[T, Self.BATCH](self.cot_pi, self.w_q1, cq, c)
            call_vjp[T, Self.BATCH](
                self.qd1.online, TensorRefs[1, MutAnyOrigin](self.qin_pi), self.cot_pi,
                TensorRefs[1, MutAnyOrigin](self.g_qin), c,
            )
            self.qd1.online.zero_grad[T](c)
            slice_cols_t[T, Self.Q_IN, Self.OBS, Self.ACT, Self.BATCH](
                t.g_pi_extra, self.g_qin, c
            )
            scale_t[T, Self.BATCH](self.cot_pi, self.w_q2, cq, c)
            call_vjp[T, Self.BATCH](
                self.qd2.online, TensorRefs[1, MutAnyOrigin](self.qin_pi), self.cot_pi,
                TensorRefs[1, MutAnyOrigin](self.g_qin), c,
            )
            self.qd2.online.zero_grad[T](c)
            slice_cols_t[T, Self.Q_IN, Self.OBS, Self.ACT, Self.BATCH](
                self.gx2, self.g_qin, c
            )
            axpy_t[T, Self.BATCH * Self.ACT](
                t.g_pi_extra, self.gx2, Scalar[DT](1.0), c
            )

        # ── 4. the FB step, unchanged, reading `g_pi_extra` ──────────────
        var fb = t.train_step(want_loss)

        self.qd1.polyak_step[T](Scalar[DT](self.tau_q), c)
        self.qd2.polyak_step[T](Scalar[DT](self.tau_q), c)

        if not want_loss:
            return FBCPRLosses(fb^, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        comptime if T == "gpu":
            var d = c.value()
            self.acc_dpos.download(d)
            self.acc_dneg.download(d)
            self.acc_r.download(d)
            self.acc_q.download(d)
            self.acc_qloss.download(d)
            self.acc_qpi.download(d)
        return FBCPRLosses(
            fb^,
            Float64(self.acc_dpos.data[0]),
            Float64(self.acc_dneg.data[0]),
            l_gp,
            Float64(self.acc_r.data[0]),
            Float64(self.acc_q.data[0]),
            Float64(self.acc_qloss.data[0]),
            Float64(self.acc_qpi.data[0]),
        )

    # ── diagnostics ──────────────────────────────────────────────────────

    def read_diag(
        mut self,
        mut d_pos: Float64, mut d_neg: Float64, mut r_mean: Float64,
        mut q_mean: Float64, mut q_loss: Float64, mut q_pi: Float64,
    ) raises:
        """The LAST step's CPR terms, from the device accumulators every
        step writes (`mean_into_t`, capture-safe). D2Hs six 1-element
        buffers: FLUSH CADENCE ONLY. `q_pi` is 0 when `reg_coeff` is 0
        (that pass is skipped). The gradient-penalty value is NOT here — it
        is only reduced on `want_loss` steps (`GradPenalty.apply`)."""
        d_pos = 0.0
        d_neg = 0.0
        r_mean = 0.0
        q_mean = 0.0
        q_loss = 0.0
        q_pi = 0.0
        if self.steps == 0:
            return
        comptime if Self.TARGET == "gpu":
            var d = self.ctx.value()
            self.acc_dpos.download(d)
            self.acc_dneg.download(d)
            self.acc_r.download(d)
            self.acc_q.download(d)
            self.acc_qloss.download(d)
            self.acc_qpi.download(d)
        d_pos = Float64(self.acc_dpos.data[0])
        d_neg = Float64(self.acc_dneg.data[0])
        r_mean = Float64(self.acc_r.data[0])
        q_mean = Float64(self.acc_q.data[0])
        q_loss = Float64(self.acc_qloss.data[0])
        q_pi = Float64(self.acc_qpi.data[0])

    # ── probes ───────────────────────────────────────────────────────────

    def discriminate[
        N: Int
    ](mut self, mut s: Tensor, mut z: Tensor, mut dst: Tensor) raises:
        """`D` logits for `N` rows of `(s, z)`. GPU: left on device."""
        comptime T = Self.TARGET
        var c = self.ctx
        var x = Tensor()
        pack2_t[T, Self.OBS, Self.D, N](x, s, z, c)
        ensure_t[T](dst, N, c)
        call_forward[T, N](self.disc, TensorRefs[1, MutAnyOrigin](x), dst, c)

    # ── checkpoint ───────────────────────────────────────────────────────

    def save_sidecar(mut self, path: String) raises:
        """`D` and the `Q_D` twins to `path + ".cpr"`. The FB nets go to
        `path` through the trainer's own `save_state`, so every FB eval reads
        the FB file unchanged."""
        var w = BinaryCheckpointWriter(save_moments=False)
        w.mode = 0
        walk_params[Self.TARGET](self.disc, w, self.ctx, "disc")
        walk_params[Self.TARGET](self.qd1.online, w, self.ctx, "qd1")
        walk_params[Self.TARGET](self.qd2.online, w, self.ctx, "qd2")
        w.mode = 1
        var _sref1 = ParamVisitorRef.of[type_of(w), Self.TARGET](w)
        self.disc.for_each_state[Self.TARGET](_sref1, self.ctx, "disc")
        var _sref2 = ParamVisitorRef.of[type_of(w), Self.TARGET](w)
        self.qd1.online.for_each_state[Self.TARGET](_sref2, self.ctx, "qd1")
        var _sref3 = ParamVisitorRef.of[type_of(w), Self.TARGET](w)
        self.qd2.online.for_each_state[Self.TARGET](_sref3, self.ctx, "qd2")
        _write_file_bytes(path + ".cpr", w.content)

    def load_sidecar(mut self, path: String) raises:
        """CPR nets from `path + ".cpr"`. ⚠ A missing sidecar RAISES: a
        resume that silently restarts `D` and `Q_D` from random init would
        train against a reward that has forgotten everything, and nothing
        in the log would say so."""
        var sp = path + ".cpr"
        if not _path_exists(sp):
            raise Error("FBCPRHead.load_sidecar: sidecar missing: " + sp)
        # ⚠ Dispatched on the header, same as `FBTrainer.load_state`: v3
        # binary for new files, v2 text for every sidecar already on disk.
        var bytes = _read_file_bytes(sp)
        if _is_v3_header(bytes):
            var rb = BinaryCheckpointReader(bytes^)
            rb.mode = 0
            walk_params[Self.TARGET](self.disc, rb, self.ctx, "disc")
            walk_params[Self.TARGET](self.qd1.online, rb, self.ctx, "qd1")
            walk_params[Self.TARGET](self.qd2.online, rb, self.ctx, "qd2")
            rb.mode = 1
            var _c1 = ParamVisitorRef.of[type_of(rb), Self.TARGET](rb)
            self.disc.for_each_state[Self.TARGET](_c1, self.ctx, "disc")
            var _c2 = ParamVisitorRef.of[type_of(rb), Self.TARGET](rb)
            self.qd1.online.for_each_state[Self.TARGET](_c2, self.ctx, "qd1")
            var _c3 = ParamVisitorRef.of[type_of(rb), Self.TARGET](rb)
            self.qd2.online.for_each_state[Self.TARGET](_c3, self.ctx, "qd2")
        else:
            var content: String
            with open(sp, "r") as f:
                content = String(f.read())
            var lines = _split_lines(content)
            var body = List[String]()
            for li in range(len(lines)):
                if lines[li].startswith("storage-ckpt"):
                    continue
                body.append(lines[li])
            var r = CheckpointReader(body^)
            r.mode = 0
            walk_params[Self.TARGET](self.disc, r, self.ctx, "disc")
            walk_params[Self.TARGET](self.qd1.online, r, self.ctx, "qd1")
            walk_params[Self.TARGET](self.qd2.online, r, self.ctx, "qd2")
            r.mode = 1
            var _sref4 = ParamVisitorRef.of[type_of(r), Self.TARGET](r)
            self.disc.for_each_state[Self.TARGET](_sref4, self.ctx, "disc")
            var _sref5 = ParamVisitorRef.of[type_of(r), Self.TARGET](r)
            self.qd1.online.for_each_state[Self.TARGET](_sref5, self.ctx, "qd1")
            var _sref6 = ParamVisitorRef.of[type_of(r), Self.TARGET](r)
            self.qd2.online.for_each_state[Self.TARGET](_sref6, self.ctx, "qd2")
        self.qd1.target_net.polyak_from[Self.TARGET](
            self.qd1.online, Scalar[DT](1.0), self.ctx
        )
        self.qd2.target_net.polyak_from[Self.TARGET](
            self.qd2.online, Scalar[DT](1.0), self.ctx
        )


struct FBCPRTrainer[
    FNET: Module,
    BNET: Module,
    ANET: Module,
    DNET: Module,
    QNET: Module,
    OBS: Int,
    ACT: Int,
    D: Int,
    BATCH: Int,
    SEQ: Int,
    TARGET: StaticString = "cpu",
](Movable & Deinitable):
    """The OFFLINE composition: an owned `FBTrainer` + an `FBCPRHead`. The
    online composition (`fb/online_cpr.mojo`) hands the same head the
    `FBOnlineAgent`'s trainer instead."""

    comptime Inner = FBTrainer[
        Self.FNET, Self.BNET, Self.ANET, Self.OBS, Self.ACT, Self.D,
        Self.BATCH, Self.TARGET,
    ]
    comptime Head = FBCPRHead[
        Self.FNET, Self.BNET, Self.ANET, Self.DNET, Self.QNET,
        Self.OBS, Self.ACT, Self.D, Self.BATCH, Self.SEQ, Self.TARGET,
    ]

    var t: Self.Inner
    var head: Self.Head

    def __init__(out self):
        self.t = Self.Inner()
        self.head = Self.Head()

    def __init__(out self, *, deinit move: Self):
        self.t = move.t^
        self.head = move.head^

    @staticmethod
    def make[
        INIT: Initializer = Xavier
    ](
        ctx: Optional[DeviceContext] = None,
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
        bc_weight: Float64 = 1.0,
        act_l2_weight: Float64 = 0.0,
        act_l2_margin: Float64 = 0.0,
        reg_coeff: Float64 = 0.01,
        gp_coef: Float64 = 10.0,
        gp_eps: Float64 = 1e-2,
        seed: UInt64 = UInt64(0x5EED),
    ) raises -> Self:
        """Defaults are the 24-D walker base (`ortho 100`, `lr_b 1e-5`,
        `bc 1.0`, clip 1.0 — §18.7.6) plus the head's CPR values.

        ⚠ `bc_weight` stays ON by default. CPR is the reference's
        replacement for BC, but the measured base carries BC and the first
        arm must vary ONE axis; `--bc 0` is the second arm, not the first.
        """
        if Self.TARGET == "gpu" and not ctx:
            raise Error("FBCPRTrainer.make: TARGET='gpu' requires a ctx")
        var s = Self()
        s.t = Self.Inner.make[INIT](
            lr=lr, gamma=gamma, tau=tau, ortho_weight=ortho_weight, ctx=ctx,
            seed=seed, max_grad_norm=max_grad_norm, bc_weight=bc_weight,
            lr_b=lr_b, act_l2_weight=act_l2_weight, act_l2_margin=act_l2_margin,
        )
        s.head = Self.Head.make[INIT](
            ctx, lr_d=lr_d, lr_q=lr_q, gamma=gamma, tau_q=tau_q,
            max_grad_norm=max_grad_norm, reg_coeff=reg_coeff, gp_coef=gp_coef,
            gp_eps=gp_eps, policy_noise=s.t.policy_noise,
            noise_clip=s.t.noise_clip, seed=seed,
        )
        return s^

    def ensure_sized(mut self) raises:
        self.head.ensure_sized(self.t)

    def load_expert(mut self, mut s: Tensor, mut s_next: Tensor) raises:
        self.head.ensure_sized(self.t)
        self.head.load_expert(s, s_next)

    def encode_expert(mut self) raises:
        self.head.encode_expert(self.t)

    def train_step(mut self, want_loss: Bool = True) raises -> FBCPRLosses:
        return self.head.step(self.t, want_loss)

    def train_device_kernels(mut self) raises:
        """`train_step(want_loss=False)` — the capturable body. Every branch
        is on a field fixed at construction (`gp_coef`, `reg_coeff`,
        `max_grad_norm`), every RNG offset is on device, no D2H."""
        comptime assert Self.TARGET == "gpu", (
            "train_device_kernels is the CUDA-graph capture path (GPU only)"
        )
        _ = self.head.step(self.t, want_loss=False)

    def discriminate[
        N: Int
    ](mut self, mut s: Tensor, mut z: Tensor, mut dst: Tensor) raises:
        self.head.discriminate[N](s, z, dst)

    def save_state(mut self, path: String) raises:
        """FB nets to `path` (the inner trainer's layout, so every FB eval
        reads it unchanged); `D` and the `Q_D` twins to `path + ".cpr"`."""
        self.t.save_state(path)
        self.head.save_sidecar(path)

    def load_state(mut self, path: String) raises:
        self.t.load_state(path)
        self.head.load_sidecar(path)
