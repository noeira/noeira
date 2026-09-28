"""`FBTrainer` — offline Forward-Backward training, CPU or GPU.

**There is no environment in this loop.** FB learns from a frozen dataset, which
is the fact that takes dm_control gap G10 off the critical path
(`docs/BFM_ZERO_SHOT_RL.md` §5.3): collection is a separate phase, evaluation is
a separate phase, and only those two touch an env. So this is not an
`OffPolicyAgent` and does not go through `driver_offpolicy` — it is fed batches
and told to step.

## Networks

Five, not three. Passed as compile-time `Module` params so the caller picks the
architecture:

    BNET : [OBS]            -> D     backward, one instance + target
    FNET : [OBS + ACT + D]  -> D     forward, TWIN + targets (TD3-style min)
    ANET : [OBS + D]        -> ACT   pi_z, deterministic + truncated noise

Inputs are pre-concatenated into one buffer per net rather than routed through
`Concat`: the trainer already owns the batch assembly, and a flat
`[s | a | z]` row keeps the FNET a plain `Sequential` the caller can write in
one line. The slice offsets are `_A_OFF` / `_Z_OFF`, and the actor gradient
reads the action slice back out of FNET's input gradient.

## One step

    1  B(s), B(s+), B(s')                          online, three forwards
    2  a' = pi_z_target(s', z) + clipped noise      target-policy smoothing
    3  Mtarget = gamma · min( Fbar1(s',a',z)·Bbar(s+)^T ,
                              Fbar2(s',a',z)·Bbar(s+)^T )     elementwise
    4  L_FB for each twin, L_ortho on B
    5  backprop; Adam; Polyak

⚠ **`B` is forwarded three times per step and its parameter gradients
ACCUMULATE across the three vjps.** That is the framework's contract (`Linear`
does `grad_w += ...`), and it is what makes the three roles of `B` — the ortho
term's `B(s)`, the measure term's `B(s+)`, the anchor's `B(s')` — sum into one
update. It also means `zero_grad` must run exactly once per step, at the top. A
second zeroing anywhere in the middle silently discards whichever contributions
came before it, and the loss would still descend.

⚠⚠ **`BNET` should end in a normalisation.** Meta Motivo carries
`"b": {"norm": true}`, and a bare `Linear -> ReLU -> Linear` backward net
DIVERGED on walker at d=128: `L_ortho` went POSITIVE and grew 8x (21 -> 172
over 24 k steps) while `|B|` climbed and the measure loss fell without bound.
At the orthonormality optimum `L_ortho` is NEGATIVE, so its SIGN is the health
check — not `|B|` alone, and certainly not the measure loss, which descends in
both the healthy and the diverging case. `nn/primitives/layer_norm.mojo`.

⚠ **The two batches must be INDEPENDENT draws.** `s+` is not `s'`. See
`loss.mojo` on why; `train_step` takes them as separate arguments and cannot
enforce it, so the sampler is where that invariant lives.

## CPU and GPU are ONE body

`TARGET` is a struct parameter (default `"cpu"`, so callers written for M1 are
unaffected). The step is written once: `nn` `Module.forward`/`vjp`,
`PairwiseDot`, Adam and Polyak already dispatch on target, and the ~8 remaining
elementwise/pack operations go through the `*_t` helpers in `kernels.mojo`.
There is deliberately no second GPU trainer struct — a duplicated 400-line step
body is a drift hazard, and the interesting failures here are silent ones.

⚠ Buffers are trainer-owned FIELDS, sized once on the first step. At
`BATCH = 1024` the `[BATCH, BATCH]` matrices are 4 MB each and several are live
at once; allocating them per step would dominate.

⚠⚠ **`want_loss` gates a device sync, not just a print.** Reading a loss value
back from the GPU is a full pipeline stall. `train_step(..., want_loss=False)`
skips both reduction kernels and the readback and returns zeros in `FBLosses`;
the GRADIENTS are identical either way, because the loss value never enters the
update. Log every few hundred steps, not every step.
"""

from noeira.nn.core.param import walk_params, ParamVisitorRef
from max.gpu.host import DeviceContext, DeviceBuffer
from std.math import abs, sqrt
from std.random import random_float64

from noeira.nn.constants import DT
from noeira.nn.core.module import Module
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.tensor_pack import TensorPack
from noeira.nn.core.call import call_forward, call_vjp
from noeira.nn.core.initializer import Initializer, Xavier
from noeira.nn.optimizer.adam import Adam
from noeira.nn.core.checkpoint import (
    CheckpointReader, _split_lines,
    BinaryCheckpointWriter, BinaryCheckpointReader,
    _read_file_bytes, _write_file_bytes, _is_v3_header,
)

from ..core.online_target_pair import OnlineTargetPair
from .loss import (
    FBLossWorkspace,
    fb_measure_loss_into,
    fb_ortho_loss_into,
    fb_measure_loss,
    fb_ortho_loss,
    pairwise_matrix,
)
from .kernels import (
    ensure_t,
    pack3_t,
    pack2_t,
    axpy_t,
    hinge_axpy_t,
    masked_rows_axpy_t,
    scale_t,
    sum3_scaled_t,
    pessimism_blend_t,
    pessimism_row_weights_t,
    scale_rows_t,
    mean_abs_into_t,
    smooth_action_t,
    slice_cols_t,
    mean_sq_t,
    mean_t,
    gaussian_t,
    gaussian_dev_t,
    mean_into_t,
    mean_sq_into_t,
    scale_by_inv_mag_t,
    axpy_by_mag_t,
)


struct FBLosses(Movable & Deinitable):
    """Per-step diagnostics. `measure` and `ortho` are the two published terms;
    `f_norm` and `b_norm` are the collapse detectors — a `B` whose rows shrink
    towards zero is the failure `L_ortho` exists to prevent, and watching the
    loss alone will not show it.

    All zero when `want_loss=False`; that is not an error, it is the caller
    declining to pay for a device sync on this step."""

    var measure: Float64
    var ortho: Float64
    var actor: Float64
    var f_norm: Float64
    var b_norm: Float64

    def __init__(
        out self, measure: Float64, ortho: Float64, actor: Float64,
        f_norm: Float64, b_norm: Float64,
    ):
        self.measure = measure
        self.ortho = ortho
        self.actor = actor
        self.f_norm = f_norm
        self.b_norm = b_norm

    def __init__(out self, *, deinit move: Self):
        self.measure = move.measure
        self.ortho = move.ortho
        self.actor = move.actor
        self.f_norm = move.f_norm
        self.b_norm = move.b_norm


struct FBTrainer[
    FNET: Module,
    BNET: Module,
    ANET: Module,
    OBS: Int,
    ACT: Int,
    D: Int,
    BATCH: Int,
    TARGET: StaticString = "cpu",
](Movable & Deinitable):
    comptime F_IN: Int = Self.OBS + Self.ACT + Self.D
    comptime A_IN: Int = Self.OBS + Self.D
    comptime _A_OFF: Int = Self.OBS
    comptime _Z_OFF: Int = Self.OBS + Self.ACT
    comptime _ND: Int = Self.BATCH * Self.D
    comptime _NN: Int = Self.BATCH * Self.BATCH
    comptime _NA: Int = Self.BATCH * Self.ACT

    var f1: OnlineTargetPair[Self.FNET]
    var f2: OnlineTargetPair[Self.FNET]
    var bnet: OnlineTargetPair[Self.BNET]
    var actor: OnlineTargetPair[Self.ANET]

    var opt_f1: Adam
    var opt_f2: Adam
    var opt_b: Adam
    var opt_actor: Adam

    var ws1: FBLossWorkspace[Self.D, Self.BATCH]
    var ws2: FBLossWorkspace[Self.D, Self.BATCH]
    var wso: FBLossWorkspace[Self.D, Self.BATCH]

    # Owned scratch — sized once, reused every step.
    var b_sn: Tensor
    # ⚠ `b_sp` is NOT part of the FB loss any more (§12.17). It is only
    # `B(s+)` for the z MIXTURE — the reference's `sample_mixed_z`, which
    # embeds a shuffled batch of goals to relabel 20 % of `z`. A separate
    # question from the successor measure, on a deliberately different draw.
    var b_sp: Tensor
    var bt_goal: Tensor
    var pi_t: Tensor
    var a_next: Tensor
    var noise: Tensor
    var fin: Tensor
    var fin_t: Tensor
    var ain: Tensor
    var ain_t: Tensor
    var ft1: Tensor
    var ft2: Tensor
    var mt1: Tensor
    var mt2: Tensor
    var m_target: Tensor
    var f1o: Tensor
    var f2o: Tensor
    var g_f1: Tensor
    var g_f2: Tensor
    var g_bsn1: Tensor
    var g_bsn2: Tensor
    var g_bsn_o: Tensor
    var g_bsn: Tensor
    var acc: Tensor
    # ⚠⚠ Persistent vjp GRAD-INPUT sinks. These were `TensorPack[1]()` locals
    # created inside `train_step` / `_actor_step`, i.e. a device alloc + free on
    # EVERY step. Two consequences, both measured on a 3000-step nsys profile:
    #   * `cuMemAlloc` + `cuMemFree` were 81% of all CUDA API time (27.5 s
    #     against 9.9 s of actual kernels).
    #   * CUDA-graph capture FAILED outright — "graph capturing in progress, no
    #     driver fallback", on a 608 KB request, which is exactly
    #     BATCH * F_IN * 4 = 1024 * 152 * 4. Allocation during capture is
    #     illegal, so a single per-step temporary is enough to block it.
    # Sized once in `_size_once` and reused. `sink` is sized to the LARGEST
    # consumer (F's input, BATCH*F_IN) and reused by B's vjps, whose grad-input
    # is smaller — `ensure_gpu` only grows, so that is safe and saves a buffer.
    # Dedicated: `acc` is reused by the f_norm/b_norm diagnostics later in the
    # step, and the adaptive scale must not share scratch with something that
    # overwrites it on logging steps only — that is the same shape of bug this
    # buffer exists to fix.
    var acc_lam: Tensor
    var acc_mag: Tensor
    var sink: Tensor
    var sink_a: Tensor
    var g_fin_a: Tensor
    # Actor-gradient split, device-resident (no D2H in the step): mean-square
    # of `g_pi` from the VALUE term alone, and after the regularisers were
    # added. `read_actor_grad_split` returns their roots at flush cadence.
    # Exists because run 2 of the online walker collapsed to a null policy
    # and nothing in the log could say whether the penalty was too strong or
    # the value term too weak — it was the latter, and only a probe showed it.
    var acc_gv: Tensor
    var acc_gt: Tensor
    # Per-row BC weight, `[BATCH]`, all ones by default (multiplying by 1.0
    # is exact, so the offline arithmetic is unchanged). `fill_bc_mask(n)`
    # sets the first `n` rows to 1 and the rest to 0 — the shape of a batch
    # whose first rows come from an EXPERT store (clone their actions) and
    # whose remaining rows are the policy's own (do not clone those: BC
    # toward the replay's own actions is circular). §18.7.2, A3.5.
    var bc_mask: Tensor
    # ⚠ An EXTERNAL actor-output gradient, `[BATCH, ACT]`, added to `g_pi`
    # when `has_pi_extra` is set — the slot FB-CPR's style critic plugs into
    # (`fb/cpr.mojo`: `−reg_coeff · ∂Q_D/∂π`). Written by the consumer BEFORE
    # `train_step` on the SAME batch; read here, never zeroed (the consumer
    # overwrites it every step). With BC off the value gradient is raw, so
    # the extra is scaled by `|mean F·z|` (BFM-Zero's `scale_reg`) to keep
    # `reg_coeff` a ratio; with BC on the value gradient is already divided
    # by that magnitude, so the extra is added as-is and the ratio is the
    # same. `has_pi_extra` is set once at construction of the consumer, so
    # the captured kernel sequence is fixed.
    var g_pi_extra: Tensor
    var has_pi_extra: Bool
    # ── the batch itself ────────────────────────────────────────────────
    # Owned, not passed per step. `TensorRefs[N, o]` requires every tensor in
    # a pack to share ONE origin, so a `train_step(s, a, ...)` taking five
    # separately-owned `mut Tensor`s cannot build the packs the nets need —
    # that is why the M1 CPU trainer staged everything through a `TensorPack`
    # and copied. Owning them makes every pack `origin_of(self)` and removes
    # the copy; the GPU gather writes straight into these.
    var bs: Tensor
    var ba: Tensor
    var bsn: Tensor
    var bsp: Tensor
    var bz: Tensor
    # actor path
    var pi: Tensor
    var fin_a: Tensor
    var fo: Tensor
    var g_fa: Tensor
    var g_pi: Tensor
    var rz: Tensor
    var rz2: Tensor
    var rzp: Tensor
    var w_f1: Tensor
    var w_f2: Tensor
    var zw: Tensor
    var fo2: Tensor
    var g_pi2: Tensor

    var ctx: Optional[DeviceContext]
    var gamma: Float64
    var tau: Float64
    var ortho_weight: Float64
    var fb_pessimism: Float64
    var actor_pessimism: Float64
    var policy_noise: Float64
    var noise_clip: Float64
    # ⚠ Global grad-norm clip, 0 = OFF. NOT cosmetic on FB: the measure loss is
    # `E[(F·B+^T - gamma·Fbar·Bbar^T)^2]` with ||B|| pinned at sqrt(d) ~ 11.3,
    # so its scale goes as (||F||·11.3)^2 and a spike to 2.5e3 carries gradients
    # to match. Measured on walker at 1 M rows: stable to ~50 k steps, then
    # excursions to +2559 by 116 k. `F` is the unconstrained half of the pair —
    # `L_ortho` and the LayerNorm bound `B`, nothing bounds `F`.
    var max_grad_norm: Float64
    # ⚠⚠ Behaviour-cloning weight on the actor loss (0 = OFF). NOT optional in
    # the offline setting, which is the only setting this trainer runs in.
    #
    # The actor maximises `F(s, pi(s,z), z)·z` over the box [-1, 1]. `F` is a
    # ReLU MLP and therefore near-linear in `a` across that box, and the
    # maximiser of a linear function on a box is a CORNER. Measured on walker
    # at 200 k steps with `bc_weight = 0`: 95-98% of emitted actions had
    # |a| > 0.99, i.e. bang-bang torque, and the policy scored WORSE than
    # random on all three tasks (significantly so on `walk`, t = -6.0).
    #
    # This is the standard offline-RL extrapolation failure — TD3+BC, CQL and
    # IQL each exist to prevent it — and `docs/BFM_ZERO_SHOT_RL.md` §7 names
    # the FB-specific answer: Meta Motivo is FB-**CPR**, where CPR regularises
    # the policy toward the data distribution with a discriminator. The BC term
    # here is the cheap stand-in for that, not a replacement.
    #
    # ⚠ Too LARGE a weight collapses every `z` onto the data's mean action and
    # destroys the whole point of a z-conditioned policy family. The scaling
    # below follows TD3+BC: the value term is normalised by its own magnitude
    # so `bc_weight` is a ratio rather than an absolute.
    var bc_weight: Float64
    # ⚠⚠ Action-MAGNITUDE penalty on the actor (0 = OFF): `w · mean(pi^2)`.
    # The ONLINE counterpart of `bc_weight`. Measured 2026-09-07 on the first
    # online walker run (`fb_online_walker_gpu.mojo`, bc 0, no penalty):
    # mean|a| 0.82 at the FIRST flush after warmup, 0.88 by 155 k updates,
    # 82-90 % of actions saturated at eval, walk/run WORSE than random. The
    # prediction that on-policy data would correct the corner was wrong: a
    # bang-bang policy generates bang-bang data, F fits it, and the corner
    # stays self-consistent. Nothing in plain FB opposes it; BFM-Zero's
    # auxiliary critic carries an action-rate term for exactly this reason
    # (§16.1 D), and its offline stand-in here is TD3+BC. This is the direct
    # form — penalise the quantity that is climbing.
    var act_l2_weight: Float64
    # ⚠⚠ Hinge margin for the penalty above (0 = plain L2 on the whole box).
    # With `margin > 0` the penalty is `w · mean(relu(|pi| - margin)^2)`:
    # ZERO gradient inside the band, so it opposes the corner and nothing
    # else — the interior is left to the value term.
    #
    # ⚠⚠ THE VALUE TERM MUST STAY RAW FOR THIS TO MEAN ANYTHING. The second
    # online walker run (2026-09-07, plain L2 at 1.0) collapsed to a NULL
    # policy (replay mean|a| 0.19, eval 0.10) and the first reading blamed the
    # penalty. A probe with the margin at 0.999 — never crossed — still held
    # mean|a| at 0.27 against 0.73 with no penalty: what suppressed the actor
    # was the ADAPTIVE SCALE, which had been switched on for any regulariser
    # and divides the value gradient by |mean F·z| (~200 on those runs). That
    # scale is TD3+BC's, and it is correct ONLY with BC, whose target is the
    # data action: "the regulariser wins" then means "sit near the data".
    # With a penalty whose target is zero it means a dead actor. So the scale
    # is applied iff `bc_weight > 0`, and the penalty competes with the raw
    # value gradient — the split is measured, see `acc_gv` / `acc_gt`.
    var act_l2_margin: Float64
    var steps: Int
    var _rng_seed: UInt64
    var _rng_offset: UInt64
    # 1-elem device Philox offset, allocated on the GPU target by `_size_once`.
    # ⚠ The HOST `_rng_offset` above cannot be used under CUDA-graph capture:
    # it is passed BY VALUE into the draw kernel, so a captured step bakes it in
    # and every replay redraws the identical noise. See `gaussian_dev_t`.
    var _rng_off_dev: Optional[DeviceBuffer[DType.uint64]]
    var _sized: Bool

    def __init__(out self):
        self.f1 = OnlineTargetPair[Self.FNET]()
        self.f2 = OnlineTargetPair[Self.FNET]()
        self.bnet = OnlineTargetPair[Self.BNET]()
        self.actor = OnlineTargetPair[Self.ANET]()
        self.opt_f1 = Adam(lr=Scalar[DT](3e-4))
        self.opt_f2 = Adam(lr=Scalar[DT](3e-4))
        self.opt_b = Adam(lr=Scalar[DT](3e-4))
        self.opt_actor = Adam(lr=Scalar[DT](3e-4))
        self.ws1 = FBLossWorkspace[Self.D, Self.BATCH]()
        self.ws2 = FBLossWorkspace[Self.D, Self.BATCH]()
        self.wso = FBLossWorkspace[Self.D, Self.BATCH]()
        self.b_sn = Tensor()
        self.b_sp = Tensor()
        self.bt_goal = Tensor()
        self.pi_t = Tensor()
        self.a_next = Tensor()
        self.noise = Tensor()
        self.fin = Tensor()
        self.fin_t = Tensor()
        self.ain = Tensor()
        self.ain_t = Tensor()
        self.ft1 = Tensor()
        self.ft2 = Tensor()
        self.mt1 = Tensor()
        self.mt2 = Tensor()
        self.m_target = Tensor()
        self.f1o = Tensor()
        self.f2o = Tensor()
        self.g_f1 = Tensor()
        self.g_f2 = Tensor()
        self.g_bsn1 = Tensor()
        self.g_bsn2 = Tensor()
        self.g_bsn_o = Tensor()
        self.g_bsn = Tensor()
        self.acc = Tensor()
        self.acc_lam = Tensor()
        self.acc_mag = Tensor()
        self.sink = Tensor()
        self.sink_a = Tensor()
        self.g_fin_a = Tensor()
        self.acc_gv = Tensor()
        self.acc_gt = Tensor()
        self.bc_mask = Tensor()
        self.g_pi_extra = Tensor()
        self.has_pi_extra = False
        self.bs = Tensor()
        self.ba = Tensor()
        self.bsn = Tensor()
        self.bsp = Tensor()
        self.bz = Tensor()
        self.pi = Tensor()
        self.fin_a = Tensor()
        self.fo = Tensor()
        self.g_fa = Tensor()
        self.g_pi = Tensor()
        self.rz = Tensor()
        self.rz2 = Tensor()
        self.rzp = Tensor()
        self.w_f1 = Tensor()
        self.w_f2 = Tensor()
        self.zw = Tensor()
        self.fo2 = Tensor()
        self.g_pi2 = Tensor()
        self.ctx = None
        self.gamma = 0.98
        self.tau = 0.01
        self.ortho_weight = 1.0
        self.fb_pessimism = 0.0
        self.actor_pessimism = 0.5
        self.policy_noise = 0.2
        self.noise_clip = 0.3
        self.max_grad_norm = 0.0
        self.bc_weight = 0.0
        self.act_l2_weight = 0.0
        self.act_l2_margin = 0.0
        self.steps = 0
        self._rng_seed = UInt64(0x5EED)
        self._rng_offset = UInt64(0)
        self._rng_off_dev = None
        self._sized = False

    def __init__(out self, *, deinit move: Self):
        self.f1 = move.f1^
        self.f2 = move.f2^
        self.bnet = move.bnet^
        self.actor = move.actor^
        self.opt_f1 = move.opt_f1^
        self.opt_f2 = move.opt_f2^
        self.opt_b = move.opt_b^
        self.opt_actor = move.opt_actor^
        self.ws1 = move.ws1^
        self.ws2 = move.ws2^
        self.wso = move.wso^
        self.b_sn = move.b_sn^
        self.b_sp = move.b_sp^
        self.bt_goal = move.bt_goal^
        self.pi_t = move.pi_t^
        self.a_next = move.a_next^
        self.noise = move.noise^
        self.fin = move.fin^
        self.fin_t = move.fin_t^
        self.ain = move.ain^
        self.ain_t = move.ain_t^
        self.ft1 = move.ft1^
        self.ft2 = move.ft2^
        self.mt1 = move.mt1^
        self.mt2 = move.mt2^
        self.m_target = move.m_target^
        self.f1o = move.f1o^
        self.f2o = move.f2o^
        self.g_f1 = move.g_f1^
        self.g_f2 = move.g_f2^
        self.g_bsn1 = move.g_bsn1^
        self.g_bsn2 = move.g_bsn2^
        self.g_bsn_o = move.g_bsn_o^
        self.g_bsn = move.g_bsn^
        self.acc = move.acc^
        self.acc_lam = move.acc_lam^
        self.acc_mag = move.acc_mag^
        self.sink = move.sink^
        self.sink_a = move.sink_a^
        self.g_fin_a = move.g_fin_a^
        self.acc_gv = move.acc_gv^
        self.acc_gt = move.acc_gt^
        self.bc_mask = move.bc_mask^
        self.g_pi_extra = move.g_pi_extra^
        self.has_pi_extra = move.has_pi_extra
        self.bs = move.bs^
        self.ba = move.ba^
        self.bsn = move.bsn^
        self.bsp = move.bsp^
        self.bz = move.bz^
        self.pi = move.pi^
        self.fin_a = move.fin_a^
        self.fo = move.fo^
        self.g_fa = move.g_fa^
        self.g_pi = move.g_pi^
        self.rz = move.rz^
        self.rz2 = move.rz2^
        self.rzp = move.rzp^
        self.w_f1 = move.w_f1^
        self.w_f2 = move.w_f2^
        self.zw = move.zw^
        self.fo2 = move.fo2^
        self.g_pi2 = move.g_pi2^
        self.ctx = move.ctx^
        self.gamma = move.gamma
        self.tau = move.tau
        self.ortho_weight = move.ortho_weight
        self.fb_pessimism = move.fb_pessimism
        self.actor_pessimism = move.actor_pessimism
        self.policy_noise = move.policy_noise
        self.noise_clip = move.noise_clip
        self.max_grad_norm = move.max_grad_norm
        self.bc_weight = move.bc_weight
        self.act_l2_weight = move.act_l2_weight
        self.act_l2_margin = move.act_l2_margin
        self.steps = move.steps
        self._rng_seed = move._rng_seed
        self._rng_offset = move._rng_offset
        self._rng_off_dev = move._rng_off_dev^
        self._sized = move._sized

    @staticmethod
    def make[
        INIT: Initializer = Xavier
    ](
        lr: Float64 = 3e-4,
        gamma: Float64 = 0.98,
        tau: Float64 = 0.01,
        ortho_weight: Float64 = 1.0,
        ctx: Optional[DeviceContext] = None,
        seed: UInt64 = UInt64(0x5EED),
        max_grad_norm: Float64 = 0.0,
        bc_weight: Float64 = 0.0,
        lr_b: Float64 = -1.0,
        act_l2_weight: Float64 = 0.0,
        act_l2_margin: Float64 = 0.0,
    ) raises -> Self:
        """`tau = 0.01` (EMA 0.99), `gamma = 0.98`, Adam 3e-4 — the published
        FB / Meta Motivo settings.

        `ctx` is required when `TARGET == "gpu"` and ignored on CPU.

        ⚠ **`ortho_weight` defaults to 1.0 and BFM-Zero ships 100.**
        `docs/BFM_ZERO_SHOT_RL.md` §16.3: both arXiv 2511.04131's Table 1 and
        the released `fb_cpr/configs.py` carry `ortho_coef = 100`, a factor of
        100 above this default. The default is left at 1.0 because every
        measurement in §13 was taken at 1.0 and silently moving it would
        invalidate them; the sweep (`examples/fb/fb_sweep.sh`) is what decides.

        ⚠ **`lr_b` is SEPARATE, and `-1.0` means "inherit `lr`".** The
        reference trains B at **1e-5** against F's 3e-4 — B is the shared
        representation and F chases it, so a B moving at F's rate is a target
        that will not sit still. One shared `lr` was never a considered choice
        here, it was this function having one argument. A negative sentinel
        rather than 0.0 because 0.0 is a legal (frozen-B) setting.
        """
        comptime assert Self.TARGET == "cpu" or Self.TARGET == "gpu", (
            "FBTrainer: TARGET must be 'cpu' or 'gpu'"
        )
        if Self.TARGET == "gpu" and not ctx:
            raise Error("FBTrainer.make: TARGET='gpu' requires a ctx")
        var t = Self()
        t.ctx = ctx
        t.f1 = OnlineTargetPair[Self.FNET].make[Self.TARGET, INIT](ctx)
        t.f2 = OnlineTargetPair[Self.FNET].make[Self.TARGET, INIT](ctx)
        t.bnet = OnlineTargetPair[Self.BNET].make[Self.TARGET, INIT](ctx)
        t.actor = OnlineTargetPair[Self.ANET].make[Self.TARGET, INIT](ctx)
        var lrb = lr if lr_b < 0.0 else lr_b
        t.opt_f1 = Adam(lr=Scalar[DT](lr))
        t.opt_f2 = Adam(lr=Scalar[DT](lr))
        t.opt_b = Adam(lr=Scalar[DT](lrb))
        t.opt_actor = Adam(lr=Scalar[DT](lr))
        # ⚠ Arena adoption is NOT an optimisation here, it is a PRECONDITION:
        # `adam.mojo` states "Don't capture a non-adopted GPU optimizer" — the
        # non-arena step reads beta^t from the HOST, so a captured replay would
        # apply a frozen bias correction. It also switches `clip_grads_device`
        # off its D2H fallback, which would otherwise sync mid-capture. No-op on
        # CPU, so this stays target-agnostic.
        comptime if Self.TARGET == "gpu":
            t.opt_f1.adopt[Self.TARGET, Self.FNET](t.f1.online, ctx)
            t.opt_f2.adopt[Self.TARGET, Self.FNET](t.f2.online, ctx)
            t.opt_b.adopt[Self.TARGET, Self.BNET](t.bnet.online, ctx)
            t.opt_actor.adopt[Self.TARGET, Self.ANET](t.actor.online, ctx)
        t.gamma = gamma
        t.tau = tau
        t.ortho_weight = ortho_weight
        t.max_grad_norm = max_grad_norm
        t.bc_weight = bc_weight
        t.act_l2_weight = act_l2_weight
        t.act_l2_margin = act_l2_margin
        t._rng_seed = seed
        return t^

    def _size_once(mut self) raises:
        """Allocate every scratch buffer on the first step."""
        if self._sized:
            return
        comptime T = Self.TARGET
        var c = self.ctx
        comptime if T == "gpu":
            var d = c.value()
            var ob = d.enqueue_create_buffer[DType.uint64](1)
            var oh = d.enqueue_create_host_buffer[DType.uint64](1)
            oh[0] = self._rng_offset
            d.enqueue_copy(ob, oh)
            d.synchronize()
            self._rng_off_dev = ob^
        ensure_t[T](self.acc_lam, 1, c)
        ensure_t[T](self.acc_mag, 1, c)
        ensure_t[T](self.acc_gv, 1, c)
        ensure_t[T](self.acc_gt, 1, c)
        ensure_t[T](self.bc_mask, Self.BATCH, c)
        for i in range(Self.BATCH):
            self.bc_mask.data[i] = Scalar[DT](1.0)
        comptime if T == "gpu":
            self.bc_mask.upload_resident(c.value())
        ensure_t[T](self.sink, Self.BATCH * Self.F_IN, c)
        ensure_t[T](self.g_pi_extra, Self._NA, c)
        ensure_t[T](self.sink_a, Self.BATCH * (Self.OBS + Self.D), c)
        ensure_t[T](self.g_fin_a, Self.BATCH * Self.F_IN, c)
        ensure_t[T](self.b_sn, Self._ND, c)
        ensure_t[T](self.b_sp, Self._ND, c)
        ensure_t[T](self.bt_goal, Self._ND, c)
        ensure_t[T](self.pi_t, Self._NA, c)
        ensure_t[T](self.a_next, Self._NA, c)
        ensure_t[T](self.noise, Self._NA, c)
        ensure_t[T](self.fin, Self.BATCH * Self.F_IN, c)
        ensure_t[T](self.fin_t, Self.BATCH * Self.F_IN, c)
        ensure_t[T](self.fin_a, Self.BATCH * Self.F_IN, c)
        ensure_t[T](self.ain, Self.BATCH * Self.A_IN, c)
        ensure_t[T](self.ain_t, Self.BATCH * Self.A_IN, c)
        ensure_t[T](self.ft1, Self._ND, c)
        ensure_t[T](self.ft2, Self._ND, c)
        ensure_t[T](self.mt1, Self._NN, c)
        ensure_t[T](self.mt2, Self._NN, c)
        ensure_t[T](self.m_target, Self._NN, c)
        ensure_t[T](self.f1o, Self._ND, c)
        ensure_t[T](self.f2o, Self._ND, c)
        ensure_t[T](self.g_f1, Self._ND, c)
        ensure_t[T](self.g_f2, Self._ND, c)
        ensure_t[T](self.g_bsn1, Self._ND, c)
        ensure_t[T](self.g_bsn2, Self._ND, c)
        ensure_t[T](self.g_bsn_o, Self._ND, c)
        ensure_t[T](self.g_bsn, Self._ND, c)
        ensure_t[T](self.pi, Self._NA, c)
        ensure_t[T](self.fo, Self._ND, c)
        ensure_t[T](self.g_fa, Self._ND, c)
        ensure_t[T](self.g_pi, Self._NA, c)
        ensure_t[T](self.rz, Self.BATCH, c)
        ensure_t[T](self.rz2, Self.BATCH, c)
        ensure_t[T](self.rzp, Self.BATCH, c)
        ensure_t[T](self.w_f1, Self.BATCH, c)
        ensure_t[T](self.w_f2, Self.BATCH, c)
        ensure_t[T](self.zw, Self._ND, c)
        ensure_t[T](self.fo2, Self._ND, c)
        ensure_t[T](self.g_pi2, Self._NA, c)
        ensure_t[T](self.acc, 1, c)
        ensure_t[T](self.bs, Self.BATCH * Self.OBS, c)
        ensure_t[T](self.ba, Self._NA, c)
        ensure_t[T](self.bsn, Self.BATCH * Self.OBS, c)
        ensure_t[T](self.bsp, Self.BATCH * Self.OBS, c)
        ensure_t[T](self.bz, Self._ND, c)
        self._sized = True

    def fill_bc_mask(mut self, n_ones: Int) raises:
        """Rows `[0, n_ones)` get BC weight 1, the rest 0. Call BEFORE any
        capture: the device buffer is written in place (`upload_resident`),
        so the pointer a captured graph holds stays valid."""
        comptime T = Self.TARGET
        self._size_once()
        for i in range(Self.BATCH):
            self.bc_mask.data[i] = Scalar[DT](1.0 if i < n_ones else 0.0)
        comptime if T == "gpu":
            self.bc_mask.upload_resident(self.ctx.value())

    def ensure_sized(mut self) raises:
        """Allocate the owned batch + scratch without running a step.

        A GPU caller gathers straight into `bs`/`ba`/`bsn`/`bsp`/`bz`, so it
        needs them sized first. It cannot get there via `load_batch` — passing
        `self`'s own fields to a `mut self` method aliases, and Mojo rejects
        it.
        """
        self._size_once()

    def embed_sp(mut self) raises:
        """`b_sp = B(bsp)` over the OWNED batch, for the `z` mixture.

        `train_step` computes this itself, but the mixture needs `B(s+)` BEFORE
        the step that consumes `z`. Exposed as a method rather than letting the
        caller write `backward_embed(t.bsp, t.b_sp)`, which aliases `self`.
        """
        comptime T = Self.TARGET
        self._size_once()
        call_forward[T, Self.BATCH](
            self.bnet.online, TensorRefs[1, MutAnyOrigin](self.bsp),
            self.b_sp, self.ctx,
        )

    # ── the step ─────────────────────────────────────────────────────────

    def load_batch(
        mut self,
        mut s: Tensor,
        mut a: Tensor,
        mut s_next: Tensor,
        mut s_plus: Tensor,
        mut z: Tensor,
    ) raises:
        """Copy a batch into the owned buffers.

        The convenience path. A GPU caller that gathers straight into
        `self.bs` / `self.ba` / ... should skip this and call `train_step`
        directly — this exists so a host-side caller need not know about the
        origin constraint described on the fields.
        """
        comptime T = Self.TARGET
        var c = self.ctx
        self._size_once()
        scale_t[T, Self.BATCH * Self.OBS](self.bs, s, Scalar[DT](1.0), c)
        scale_t[T, Self._NA](self.ba, a, Scalar[DT](1.0), c)
        scale_t[T, Self.BATCH * Self.OBS](self.bsn, s_next, Scalar[DT](1.0), c)
        scale_t[T, Self.BATCH * Self.OBS](self.bsp, s_plus, Scalar[DT](1.0), c)
        scale_t[T, Self._ND](self.bz, z, Scalar[DT](1.0), c)

    def train_step(mut self, want_loss: Bool = True) raises -> FBLosses:
        """One gradient step over the OWNED batch (`bs`/`ba`/`bsn`/`bsp`/`bz`).

        ⚠ `bsp` MUST come from a draw independent of `(bs, ba, bsn)`.
        """
        comptime T = Self.TARGET
        var c = self.ctx
        self._size_once()
        self.steps += 1

        # ── zero once, at the top. See the module docstring: B's parameter
        # gradients accumulate over its three forwards, and a second zeroing
        # in the middle would drop whichever came first, silently.
        self.f1.online.zero_grad[T](c)
        self.f2.online.zero_grad[T](c)
        self.bnet.online.zero_grad[T](c)
        self.actor.online.zero_grad[T](c)

        # ── 1. B forwards (online) ───────────────────────────────────────
        # `B(s)` used to be computed here for the two-tensor ortho term. That
        # term was a collapse objective (see `loss.mojo`'s header); the ortho
        # now runs on `b_sn` alone, so this forward and its vjp are gone.
        call_forward[T, Self.BATCH](
            self.bnet.online, TensorRefs[1, MutAnyOrigin](self.bsn), self.b_sn, c
        )

        # ── 2. a' = pi_target(s', z) + truncated noise ───────────────────
        pack2_t[T, Self.OBS, Self.D, Self.BATCH](
            self.ain_t, self.bsn, self.bz, c
        )
        call_forward[T, Self.BATCH](
            self.actor.target_net, TensorRefs[1, MutAnyOrigin](self.ain_t), self.pi_t, c
        )
        gaussian_dev_t[T, Self._NA](
            self.noise, self._rng_seed, self._rng_off_dev.value(), c
        ) if T == "gpu" else gaussian_t[T, Self._NA](
            self.noise, self._rng_seed, self._rng_offset, c
        )
        # Host mirror kept only so the CPU path and `save_state` stay unchanged;
        # on GPU the authoritative counter is the device buffer, bumped inside
        # `gaussian_dev_t` so the advance is part of any captured sequence.
        self._rng_offset += UInt64(Self._NA + (Self._NA % 2))
        smooth_action_t[T, Self._NA](
            self.a_next, self.pi_t, self.noise,
            Scalar[DT](self.policy_noise), Scalar[DT](self.noise_clip), c,
        )

        # ── 3. the bootstrapped target ───────────────────────────────────
        pack3_t[T, Self.OBS, Self.ACT, Self.D, Self.BATCH](
            self.fin_t, self.bsn, self.a_next, self.bz, c
        )
        call_forward[T, Self.BATCH](
            self.f1.target_net, TensorRefs[1, MutAnyOrigin](self.fin_t), self.ft1, c
        )
        call_forward[T, Self.BATCH](
            self.f2.target_net, TensorRefs[1, MutAnyOrigin](self.fin_t), self.ft2, c
        )
        # `goal = next_obs` (`agent.py:190`): ONE B tensor for the matrix, its
        # diagonal and the ortho. `bsp` — the independent draw — no longer
        # feeds anything and its forward is gone with it.
        call_forward[T, Self.BATCH](
            self.bnet.target_net, TensorRefs[1, MutAnyOrigin](self.bsn), self.bt_goal, c
        )

        # Both target matrices go through the SAME primitive the online path
        # uses. A hand-inlined target would be one edit away from disagreeing
        # with the online one, and that disagreement is invisible in the loss.
        self.ws1.prepare[T](c)
        self.ws1.pd.forward[T, Self.BATCH](
            TensorRefs[2, MutAnyOrigin](self.ft1, self.bt_goal), self.mt1, c
        )
        self.ws1.pd.forward[T, Self.BATCH](
            TensorRefs[2, MutAnyOrigin](self.ft2, self.bt_goal), self.mt2, c
        )
        # ⚠ `fb_pessimism` is 0.0 — the FB target is the ensemble MEAN, not
        # the twin-min. This was `min_scale_t` (i.e. penalty 0.5) until
        # §12.15; the reference sets `fb_pessimism_penalty=0.0` for the
        # measure target and 0.5 only for its three Q critics.
        pessimism_blend_t[T, Self._NN](
            self.m_target, self.mt1, self.mt2, Scalar[DT](self.gamma),
            Scalar[DT](self.fb_pessimism), c,
        )

        # ── 4. online F forwards + losses ────────────────────────────────
        pack3_t[T, Self.OBS, Self.ACT, Self.D, Self.BATCH](
            self.fin, self.bs, self.ba, self.bz, c
        )
        call_forward[T, Self.BATCH](
            self.f1.online, TensorRefs[1, MutAnyOrigin](self.fin), self.f1o, c
        )
        call_forward[T, Self.BATCH](
            self.f2.online, TensorRefs[1, MutAnyOrigin](self.fin), self.f2o, c
        )

        var q1 = Float64(0)
        var a1 = Float64(0)
        var q2 = Float64(0)
        var a2 = Float64(0)
        var l1 = fb_measure_loss_into[T, Self.D, Self.BATCH](
            self.ws1, self.f1o, self.b_sn, self.m_target,
            self.g_f1, self.g_bsn1, q1, a1, want_loss, c,
        )
        var l2 = fb_measure_loss_into[T, Self.D, Self.BATCH](
            self.ws2, self.f2o, self.b_sn, self.m_target,
            self.g_f2, self.g_bsn2, q2, a2, want_loss, c,
        )
        # ortho on the SAME tensor the matrix is built from, as the reference
        # does (`Cov = B @ B.T` on `B(goal)`).
        var l_ortho = fb_ortho_loss_into[T, Self.D, Self.BATCH](
            self.wso, self.b_sn, self.g_bsn_o, want_loss, c,
        )

        # ── 5. backprop ──────────────────────────────────────────────────
        call_vjp[T, Self.BATCH](
            self.f1.online, TensorRefs[1, MutAnyOrigin](self.fin), self.g_f1,
            TensorRefs[1, MutAnyOrigin](self.sink), c,
        )
        call_vjp[T, Self.BATCH](
            self.f2.online, TensorRefs[1, MutAnyOrigin](self.fin), self.g_f2,
            TensorRefs[1, MutAnyOrigin](self.sink), c,
        )

        # B: ONE vjp now. `b_sn` = B(goal) carries everything — the measure
        # residual from each F (off-diagonal AND diagonal) and the ortho term.
        sum3_scaled_t[T, Self._ND](
            self.g_bsn, self.g_bsn1, self.g_bsn2, self.g_bsn_o,
            Scalar[DT](self.ortho_weight), c,
        )

        call_vjp[T, Self.BATCH](
            self.bnet.online, TensorRefs[1, MutAnyOrigin](self.bsn), self.g_bsn,
            TensorRefs[1, MutAnyOrigin](self.sink), c,
        )

        if self.max_grad_norm > 0.0:
            var mgn = Scalar[DT](self.max_grad_norm)
            # ⚠ `clip_grads` D2Hs the norm — illegal inside a capture. The
            # `_device` form keeps it on device over persistent scratch and is
            # identical arithmetic; `read_clip_norm` retrieves it at flush
            # cadence if ever needed.
            self.opt_f1.clip_grads_device[T](self.f1.online, mgn, c)
            self.opt_f2.clip_grads_device[T](self.f2.online, mgn, c)
            self.opt_b.clip_grads_device[T](self.bnet.online, mgn, c)
        self.opt_f1.step[T](self.f1.online, c)
        self.opt_f2.step[T](self.f2.online, c)
        self.opt_b.step[T](self.bnet.online, c)

        # ── actor ────────────────────────────────────────────────────────
        var l_actor = self._actor_step(want_loss)

        # ── Polyak ───────────────────────────────────────────────────────
        self.f1.polyak_step[T](Scalar[DT](self.tau), c)
        self.f2.polyak_step[T](Scalar[DT](self.tau), c)
        self.bnet.polyak_step[T](Scalar[DT](self.tau), c)
        self.actor.polyak_step[T](Scalar[DT](self.tau), c)

        if not want_loss:
            return FBLosses(0.0, 0.0, 0.0, 0.0, 0.0)
        var fn2 = mean_sq_t[T, Self._ND](self.f1o, self.acc, c)
        var bn2 = mean_sq_t[T, Self._ND](self.b_sn, self.acc, c)
        return FBLosses(
            0.5 * (l1 + l2), l_ortho, l_actor,
            sqrt(fn2 * Float64(Self.D)), sqrt(bn2 * Float64(Self.D)),
        )

    def read_grad_norms(
        mut self, mut f1: Float64, mut f2: Float64, mut b: Float64
    ) raises:
        """PRE-clip gradient norms from the last `clip_grads_device`.

        The discriminating diagnostic for "is F running away". §13's defect 3
        was F growing unbounded, and the only symptom in the log was the measure
        loss falling — which is ALSO what F merely learning looks like. The
        pre-clip norm separates them: a converging F clips rarely and its norm
        settles; a diverging F pins against `max_grad_norm` every step.

        ⚠ D2Hs three buffers, so FLUSH CADENCE ONLY — never per step. Returns
        0 when `max_grad_norm <= 0` (nothing ever ran the device clip).
        """
        comptime T = Self.TARGET
        f1 = 0.0
        f2 = 0.0
        b = 0.0
        comptime if T == "gpu":
            if self.max_grad_norm > 0.0:
                var c = self.ctx.value()
                f1 = Float64(self.opt_f1.read_clip_norm(c))
                f2 = Float64(self.opt_f2.read_clip_norm(c))
                b = Float64(self.opt_b.read_clip_norm(c))

    def read_actor_grad_split(
        mut self, mut g_value: Float64, mut g_total: Float64
    ) raises:
        """RMS of the actor's output gradient from the value term alone, and
        after BC / the action penalty were added — the last step's. FLUSH
        CADENCE ONLY (D2H). `g_total / g_value` near 1 means the regulariser
        is negligible; far above 1 means it owns the actor."""
        comptime T = Self.TARGET
        g_value = 0.0
        g_total = 0.0
        if self.steps == 0:
            return
        comptime if T == "gpu":
            self.acc_gv.download(self.ctx.value())
            self.acc_gt.download(self.ctx.value())
        g_value = sqrt(Float64(self.acc_gv.data[0]))
        g_total = sqrt(Float64(self.acc_gt.data[0]))

    def train_device_kernels(mut self) raises:
        """The pure device-kernel train step — the body to hand to
        `maybe_capture_replay`.

        It is exactly `train_step(want_loss=False)`. That path already contains
        no host work: every loss readback is behind `want_loss`, the grad-norm
        clip goes through `clip_grads_device`, Adam runs the adopted-arena
        kernel that reads beta^t from device, and the exploration noise draws
        its Philox offset from a device buffer that a kernel bumps in-sequence.

        ⚠ Do NOT call this with `want_loss=True` semantics, and do not add a
        host read to the `want_loss=False` path. A `synchronize` or D2H inside a
        capture is illegal and fails loudly; a host VALUE baked into a kernel
        argument does not — it silently freezes at its capture-time value. The
        second failure mode is the one that has cost this project a run before.

        ⚠ The kernel SEQUENCE must be identical on every call for the captured
        graph to stay valid. It is, because the only branch is
        `self.max_grad_norm > 0.0` on a field fixed at construction.
        """
        comptime assert Self.TARGET == "gpu", (
            "train_device_kernels is the CUDA-graph capture path (GPU only)"
        )
        _ = self.train_step(want_loss=False)

    def _actor_step(mut self, want_loss: Bool) raises -> Float64:
        """DPG through `F1`: maximise `F(s, pi_z(s,z), z) · z`.

        `z` is both the actor's conditioning input and the direction the value
        is projected onto — that is the whole point of the latent, and it is
        why the actor loss needs no reward.
        """
        comptime T = Self.TARGET
        var c = self.ctx

        pack2_t[T, Self.OBS, Self.D, Self.BATCH](
            self.ain, self.bs, self.bz, c
        )
        call_forward[T, Self.BATCH](
            self.actor.online, TensorRefs[1, MutAnyOrigin](self.ain), self.pi, c
        )
        pack3_t[T, Self.OBS, Self.ACT, Self.D, Self.BATCH](
            self.fin_a, self.bs, self.pi, self.bz, c
        )
        call_forward[T, Self.BATCH](
            self.f1.online, TensorRefs[1, MutAnyOrigin](self.fin_a), self.fo, c
        )
        call_forward[T, Self.BATCH](
            self.f2.online, TensorRefs[1, MutAnyOrigin](self.fin_a), self.fo2, c
        )

        # ⚠⚠ Rowwise F·z and its mean are computed on EVERY step, not only
        # when a loss is wanted. They feed TD3+BC's adaptive scale below, and
        # gating them on `want_loss` made the ACTOR OBJECTIVE depend on the
        # LOGGING CADENCE.
        self.ws1.rd.forward[T, Self.BATCH](
            TensorRefs[2, MutAnyOrigin](self.fo, self.bz), self.rz, c
        )
        self.ws2.rd.forward[T, Self.BATCH](
            TensorRefs[2, MutAnyOrigin](self.fo2, self.bz), self.rz2, c
        )
        # Q_fb is the PESSIMISTIC reduction over the ensemble, which at
        # `actor_pessimism` 0.5 is exactly `min(F1·z, F2·z)` (`agent.py:275`).
        # This went through F1 ALONE until §12.19 — an optimistic value term,
        # and the one place the twin was paid for and not used.
        pessimism_blend_t[T, Self.BATCH](
            self.rzp, self.rz, self.rz2, Scalar[DT](1.0),
            Scalar[DT](self.actor_pessimism), c,
        )
        # per-row d/dF1 and d/dF2 of that same reduction: at penalty 0.5 the
        # whole gradient goes to whichever twin is the min on that row
        pessimism_row_weights_t[T, Self.BATCH](
            self.w_f1, self.w_f2, self.rz, self.rz2,
            Scalar[DT](self.actor_pessimism), c,
        )
        mean_into_t[T, Self.BATCH](self.rzp, self.acc_lam, c)
        # ⚠ TWO DIFFERENT REDUCTIONS OF THE SAME ROWS, AND THEY ARE NOT
        # INTERCHANGEABLE. `acc_lam` is the SIGNED mean — that is the actor
        # loss the reference logs (`-Q_fb.mean()`). `acc_mag` is the mean of
        # ABSOLUTE values — that is the `scale_reg` / TD3+BC weight
        # (`Q_fb.abs().mean()`). We used `|acc_lam|` for the weight until
        # §12.15; Jensen makes that never larger and it collapses toward 0 as
        # Q_fb becomes sign-balanced.
        mean_abs_into_t[T, Self.BATCH](self.rzp, self.acc_mag, c)

        var loss = Float64(0)
        if want_loss:
            comptime if T == "cpu":
                loss = -Float64(self.acc_lam.data[0])
            else:
                self.acc_lam.download(c.value())
                loss = -Float64(self.acc_lam.data[0])

        # dL/dF = -lambda·z / BATCH. TD3+BC's adaptive scale: normalising the
        # value term by its own magnitude makes `bc_weight` a RATIO between the
        # two objectives instead of an absolute that has to be retuned whenever
        # ||F|| moves — and ||F|| moves a lot here.
        #
        # ⚠⚠ This USED to read `abs(loss) if want_loss else 1.0`, so the scale
        # was applied on ONE step in `LOG_EVERY` and skipped on the other 1999 —
        # and under CUDA-graph capture (`want_loss=False` always) it was applied
        # NEVER. `||F||` then grew 2.65x over 214 k steps while `bc_weight`
        # stayed fixed, the value term outgrew the BC term, and `pi_z` went
        # bang-bang: 70-79% of actions at |a| > 0.99 and WORSE than random on
        # all three walker tasks. The dataset is not saturated (mean|a| 0.635,
        # 4.9% at the rail), so this was the policy's own doing.
        #
        # The magnitude now stays on device and the scale kernel reads it, so
        # the normalisation is unconditional AND capture-safe.
        # ⚠ BC ONLY — see `act_l2_margin`. Normalising the value term against
        # an action penalty leaves the actor with no drive at all.
        # The value term's cotangent is `-w_i·z_i / BATCH` — the plain `z`
        # scaled PER ROW by that twin's share of the pessimistic reduction.
        # `zw` holds it so the two scale kernels below stay exactly what they
        # were; writing a row-scaled variant of each would be the same rule in
        # two more places.
        for _twin in range(2):
            if _twin == 0:
                scale_rows_t[T, Self.BATCH, Self.D](self.zw, self.bz, self.w_f1, c)
            else:
                scale_rows_t[T, Self.BATCH, Self.D](self.zw, self.bz, self.w_f2, c)
            if self.bc_weight > 0.0:
                scale_by_inv_mag_t[T, Self._ND](
                    self.g_fa, self.zw, self.acc_mag,
                    Scalar[DT](-1.0 / Float64(Self.BATCH)), Scalar[DT](1e-6), c,
                )
            else:
                scale_t[T, Self._ND](
                    self.g_fa, self.zw,
                    Scalar[DT](-1.0 / Float64(Self.BATCH)), c,
                )
            # ⚠ Through F WITHOUT keeping its parameter grads: the optimizer
            # has already stepped them above, and folding a second,
            # differently-scaled critic gradient into the next step would be
            # silent. The vjp accumulates into params, so they are zeroed
            # right after.
            if _twin == 0:
                call_vjp[T, Self.BATCH](
                    self.f1.online, TensorRefs[1, MutAnyOrigin](self.fin_a),
                    self.g_fa, TensorRefs[1, MutAnyOrigin](self.g_fin_a), c,
                )
                self.f1.online.zero_grad[T](c)
                slice_cols_t[T, Self.F_IN, Self._A_OFF, Self.ACT, Self.BATCH](
                    self.g_pi, self.g_fin_a, c
                )
            else:
                call_vjp[T, Self.BATCH](
                    self.f2.online, TensorRefs[1, MutAnyOrigin](self.fin_a),
                    self.g_fa, TensorRefs[1, MutAnyOrigin](self.g_fin_a), c,
                )
                self.f2.online.zero_grad[T](c)
                slice_cols_t[T, Self.F_IN, Self._A_OFF, Self.ACT, Self.BATCH](
                    self.g_pi2, self.g_fin_a, c
                )
                axpy_t[T, Self._NA](self.g_pi, self.g_pi2, Scalar[DT](1.0), c)
        mean_sq_into_t[T, Self._NA](self.g_pi, self.acc_gv, c)
        # + BC: d/dpi of `bc_weight · mean_i mean_k (pi - a_data)^2`.
        # `axpy` twice rather than a bespoke kernel: g_pi += w·pi, g_pi -= w·a.
        if self.bc_weight > 0.0:
            var w = Scalar[DT](
                self.bc_weight * 2.0
                / (Float64(Self.BATCH) * Float64(Self.ACT))
            )
            masked_rows_axpy_t[T, Self.BATCH, Self.ACT](
                self.g_pi, self.pi, self.bc_mask, w, c
            )
            masked_rows_axpy_t[T, Self.BATCH, Self.ACT](
                self.g_pi, self.ba, self.bc_mask, -w, c
            )
        # + action magnitude: d/dpi of `act_l2_weight · mean(pi^2)` — the BC
        # term with the data action at 0. Same scale form, so the two knobs
        # read on the same axis.
        if self.act_l2_weight > 0.0:
            var w2 = Scalar[DT](
                self.act_l2_weight * 2.0
                / (Float64(Self.BATCH) * Float64(Self.ACT))
            )
            if self.act_l2_margin > 0.0:
                hinge_axpy_t[T, Self._NA](
                    self.g_pi, self.pi, w2, Scalar[DT](self.act_l2_margin), c
                )
            else:
                axpy_t[T, Self._NA](self.g_pi, self.pi, w2, c)
        # + the external term (FB-CPR's `−reg·∂Q_D/∂π`), see `g_pi_extra`.
        # `acc_mag` holds THIS step's `mean |F·z|`, so the scale is not lagged.
        if self.has_pi_extra:
            if self.bc_weight > 0.0:
                axpy_t[T, Self._NA](self.g_pi, self.g_pi_extra, Scalar[DT](1.0), c)
            else:
                axpy_by_mag_t[T, Self._NA](
                    self.g_pi, self.g_pi_extra, self.acc_mag, Scalar[DT](1.0), c
                )
        mean_sq_into_t[T, Self._NA](self.g_pi, self.acc_gt, c)

        call_vjp[T, Self.BATCH](
            self.actor.online, TensorRefs[1, MutAnyOrigin](self.ain), self.g_pi,
            TensorRefs[1, MutAnyOrigin](self.sink_a), c,
        )
        if self.max_grad_norm > 0.0:
            # ⚠ `clip_grads` (D2H) was still here after the other three were
            # moved to `clip_grads_device` — one missed call site is enough to
            # make a capture illegal, and it is a per-step sync besides.
            self.opt_actor.clip_grads_device[T](
                self.actor.online, Scalar[DT](self.max_grad_norm), c
            )
        self.opt_actor.step[T](self.actor.online, c)
        return loss

    # ── checkpoint ───────────────────────────────────────────────────────

    def save_state(mut self, path: String) raises:
        """Write B, both F twins and the actor into ONE `storage-ckpt` file.

        ⚠ Only the ONLINE nets. The targets are EMA copies that re-converge
        within a few thousand Polyak steps, and Adam moments re-warm — neither
        is worth the file size. A resume is therefore not bit-identical, which
        is fine for a 2 M-step run and would not be for a parity gate.

        ⚠⚠ Call this PERIODICALLY, not only at the end. The first version of
        the M2 run script trained for 2 M steps and exited without saving
        anything: hours of GPU time producing a log file and no weights.
        """
        var w = BinaryCheckpointWriter(save_moments=False)
        w.mode = 0
        walk_params[Self.TARGET](self.bnet.online, w, self.ctx, "b")
        walk_params[Self.TARGET](self.f1.online, w, self.ctx, "f1")
        walk_params[Self.TARGET](self.f2.online, w, self.ctx, "f2")
        walk_params[Self.TARGET](self.actor.online, w, self.ctx, "actor")
        w.mode = 1
        var _sref1 = ParamVisitorRef.of[type_of(w), Self.TARGET](w)
        self.bnet.online.for_each_state[Self.TARGET](_sref1, self.ctx, "b")
        var _sref2 = ParamVisitorRef.of[type_of(w), Self.TARGET](w)
        self.f1.online.for_each_state[Self.TARGET](_sref2, self.ctx, "f1")
        var _sref3 = ParamVisitorRef.of[type_of(w), Self.TARGET](w)
        self.f2.online.for_each_state[Self.TARGET](_sref3, self.ctx, "f2")
        var _sref4 = ParamVisitorRef.of[type_of(w), Self.TARGET](w)
        self.actor.online.for_each_state[Self.TARGET](_sref4, self.ctx, "actor")
        _write_file_bytes(path, w.content)

    def load_state(mut self, path: String) raises:
        """Restore the online nets and HARD-COPY them onto the targets.

        Without the hard copy the targets stay at their random init while the
        online nets are trained, and the first bootstrapped target is garbage —
        a resume that silently undoes part of the run it is resuming.
        """
        # ⚠ THE FORMAT IS DISPATCHED ON THE HEADER, both ways forever. v3 is
        # binary (4 B a float); v2 was one DECIMAL FLOAT PER LINE, 23 B a
        # float — a 500 MB file per G1 checkpoint, 921 MB with the sidecar,
        # and ~16.5 s of the step it was written on. New files are v3; every
        # v2 file already on disk still loads, because a run in flight must
        # not be made unresumable by a format change.
        var bytes = _read_file_bytes(path)
        if _is_v3_header(bytes):
            var rb = BinaryCheckpointReader(bytes^)
            rb.mode = 0
            walk_params[Self.TARGET](self.bnet.online, rb, self.ctx, "b")
            walk_params[Self.TARGET](self.f1.online, rb, self.ctx, "f1")
            walk_params[Self.TARGET](self.f2.online, rb, self.ctx, "f2")
            walk_params[Self.TARGET](self.actor.online, rb, self.ctx, "actor")
            rb.mode = 1
            var _b1 = ParamVisitorRef.of[type_of(rb), Self.TARGET](rb)
            self.bnet.online.for_each_state[Self.TARGET](_b1, self.ctx, "b")
            var _b2 = ParamVisitorRef.of[type_of(rb), Self.TARGET](rb)
            self.f1.online.for_each_state[Self.TARGET](_b2, self.ctx, "f1")
            var _b3 = ParamVisitorRef.of[type_of(rb), Self.TARGET](rb)
            self.f2.online.for_each_state[Self.TARGET](_b3, self.ctx, "f2")
            var _b4 = ParamVisitorRef.of[type_of(rb), Self.TARGET](rb)
            self.actor.online.for_each_state[Self.TARGET](_b4, self.ctx, "actor")
        else:
            var content: String
            with open(path, "r") as f:
                content = String(f.read())
            var lines = _split_lines(content)
            # The `storage-ckpt vN` header is not a section; the reader expects
            # the first line to BE one, so strip it.
            var body = List[String]()
            for li in range(len(lines)):
                if lines[li].startswith("storage-ckpt"):
                    continue
                body.append(lines[li])
            var r = CheckpointReader(body^)
            r.mode = 0
            walk_params[Self.TARGET](self.bnet.online, r, self.ctx, "b")
            walk_params[Self.TARGET](self.f1.online, r, self.ctx, "f1")
            walk_params[Self.TARGET](self.f2.online, r, self.ctx, "f2")
            walk_params[Self.TARGET](self.actor.online, r, self.ctx, "actor")
            r.mode = 1
            var _sref5 = ParamVisitorRef.of[type_of(r), Self.TARGET](r)
            self.bnet.online.for_each_state[Self.TARGET](_sref5, self.ctx, "b")
            var _sref6 = ParamVisitorRef.of[type_of(r), Self.TARGET](r)
            self.f1.online.for_each_state[Self.TARGET](_sref6, self.ctx, "f1")
            var _sref7 = ParamVisitorRef.of[type_of(r), Self.TARGET](r)
            self.f2.online.for_each_state[Self.TARGET](_sref7, self.ctx, "f2")
            var _sref8 = ParamVisitorRef.of[type_of(r), Self.TARGET](r)
            self.actor.online.for_each_state[Self.TARGET](_sref8, self.ctx, "actor")
        self.bnet.target_net.polyak_from[Self.TARGET](
            self.bnet.online, Scalar[DT](1.0), self.ctx
        )
        self.f1.target_net.polyak_from[Self.TARGET](
            self.f1.online, Scalar[DT](1.0), self.ctx
        )
        self.f2.target_net.polyak_from[Self.TARGET](
            self.f2.online, Scalar[DT](1.0), self.ctx
        )
        self.actor.target_net.polyak_from[Self.TARGET](
            self.actor.online, Scalar[DT](1.0), self.ctx
        )

    # ── inference ────────────────────────────────────────────────────────

    def act[
        N: Int
    ](mut self, mut s: Tensor, mut z_row: Tensor, mut dst: Tensor) raises:
        """`pi_z(s, z)` for `N` rows — the zero-shot policy.

        ⚠ `z_row` must already be on the radius-sqrt(D) sphere and must hold N
        rows (broadcast a single z yourself). `z_sampler`'s producers all
        guarantee the norm; a `z` assembled by hand does not, and the symptom
        is a policy that acts plausibly and optimises nothing.
        """
        comptime T = Self.TARGET
        var c = self.ctx
        # Staged through a TensorPack so the packed input is a single owned
        # buffer of exactly `N * A_IN`, independent of how the caller sized
        # `s` and `z_row`. `N` here is an INFERENCE batch and differs from the
        # training `BATCH`, so none of the trainer's own scratch is the right
        # width.
        var pack = TensorPack[1]()
        var ain = Tensor()
        pack2_t[T, Self.OBS, Self.D, N](ain, s, z_row, c)
        ensure_t[T](pack[0], N * Self.A_IN, c)
        scale_t[T, N * Self.A_IN](pack[0], ain, Scalar[DT](1.0), c)
        ensure_t[T](dst, N * Self.ACT, c)
        call_forward[T, N](
            self.actor.online, TensorRefs[1, MutAnyOrigin](pack[0]), dst, c
        )

    def forward_f[
        N: Int
    ](
        mut self, mut s: Tensor, mut a: Tensor, mut z: Tensor, mut dst: Tensor
    ) raises:
        """`F(s, a, z)` for `N` rows through the ONLINE `f1`.

        The read-only counterpart of `backward_embed`: `f_norm` in `FBLosses`
        is the norm of exactly this, so an offline probe can reproduce the
        logged quantity on a batch of its choosing.

        ⚠ Same device convention as `backward_embed` — on the GPU target the
        result is left ON DEVICE and the caller must download before reading
        `dst.data`.
        """
        comptime T = Self.TARGET
        var c = self.ctx
        ensure_t[T](dst, N * Self.D, c)
        var fin = Tensor()
        ensure_t[T](fin, N * (Self.OBS + Self.ACT + Self.D), c)
        pack3_t[T, Self.OBS, Self.ACT, Self.D, N](fin, s, a, z, c)
        call_forward[T, N](
            self.f1.online, TensorRefs[1, MutAnyOrigin](fin), dst, c
        )

    def backward_embed[
        N: Int
    ](mut self, mut s: Tensor, mut dst: Tensor) raises:
        """`B(s)` for `N` rows — the input to `z_from_reward`.

        ⚠ On the GPU target the result is left ON DEVICE. Callers that read
        `dst.data` must `dst.download(ctx)` first; without it they read a host
        buffer that was never written — all zeros, silently. Cost a debugging
        pass in `test_fb_cuda_graph_safety.mojo`.
        """
        comptime T = Self.TARGET
        var c = self.ctx
        ensure_t[T](dst, N * Self.D, c)
        var pack = TensorPack[1]()
        ensure_t[T](pack[0], N * Self.OBS, c)
        scale_t[T, N * Self.OBS](pack[0], s, Scalar[DT](1.0), c)
        call_forward[T, N](
            self.bnet.online, TensorRefs[1, MutAnyOrigin](pack[0]), dst, c
        )

