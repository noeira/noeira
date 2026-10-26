"""EfficientZeroV2 continuous K-step unroll — MuZero BPTT + SimSiam + Gaussian π.

The continuous twin of `blocks.mojo::ezv2_unroll_train_step_cpu`. Structurally
identical (forward scan h + K dynamics, reverse scan with the ½ dynamics-grad
carry, SimSiam consistency at k=1..K), with two changes for continuous control:

  * **Dynamics input is ``[z | a]`` with a real action *vector*** (``ACT_DIM``
    continuous dims) instead of a one-hot — the build step copies the action
    vector into the ``ACT_DIM`` slots.
  * **Policy head is a squashed Gaussian.** The prediction output row is
    ``[μ_raw | σ_raw | value]`` (``2·ACT_DIM + BINS``). The policy loss is the
    squashed-Gaussian NLL of the search-selected action
    (`loss_ops_continuous.continuous_policy_loss_and_grad`) over the leading
    ``2·ACT_DIM`` slice; value + reward stay categorical soft-CE.

Batch layout (time-major): ``obs_seq[K+1,B,OBS]``, ``actions[K,B,ACT_DIM]`` (the
transition actions), ``policy_act_tgt[K+1,B,ACT_DIM]`` (the per-position target
actions the policy clones), ``value_tgt[K+1,B]``, ``reward_tgt[K,B]`` (raw).

Storage port: mirrors the storage `efficient_zero_v2/blocks.mojo` (List inputs,
owned-Tensor scratch, TensorRefs forward/vjp, per-net clip_grad_norm + begin_step
+ for_each_param). The persistent device scratch is `EZV2UnrollContScratch`.
"""

from noeira.nn.core.param import walk_params
from std.memory import alloc
from max.gpu import global_idx
from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer

from noeira.nn.constants import DT, TPB
from noeira.nn.core.module import Module
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.call import call_forward, call_vjp
from noeira.nn.optimizer.adam import Adam
from noeira.nn.optimizer.grad_clip import clip_grad_norm

from .loss_ops import consistency_loss_and_grad, consistency_loss_grad_k
from .loss_ops_continuous import (
    continuous_policy_loss_and_grad,
    continuous_policy_loss_grad_k,
)
from .blocks import _ez_accum_latent_k
from .unroll_scratch import EZV2UnrollContScratch
from ..muzero.loss_ops import soft_ce_slice_loss_and_grad
from ..muzero.blocks import (
    _mz_copy_latent_k,
    _mz_softce_slice_k,
    _mz_twohot_k,
    _mz_set_carry_latent_k,
    _mz_accum_half_k,
    _mz_bcopy_k,
)
from ..zero.twohot_targets import mz_two_hot_target_batch
from noeira.core.tiles import RowMajor, Tile


def _a(n: Int) -> Pointer[Scalar[DT], MutAnyOrigin]:
    """Raw host scratch for optional unroll outputs (loss_parts) — function-local;
    the unroll's optional-output params are Optional[Pointer]."""
    return alloc[Scalar[DT]]({count = n}).unsafe_leak().as_unsafe_any_origin()


def ezv2_unroll_train_step_continuous_cpu[
    REP: Module,
    DYN: Module,
    PRED: Module,
    PROJM: Module,
    PREDH: Module,
    B: Int,
    K: Int,
    OBS: Int,
    ACT_DIM: Int,
    LATENT: Int,
    BINS: Int,
](
    mut rep: REP,
    mut dyn: DYN,
    mut pred: PRED,
    mut proj: PROJM,
    mut predh: PREDH,
    mut orep: Adam,
    mut odyn: Adam,
    mut opred: Adam,
    mut oproj: Adam,
    mut opredh: Adam,
    obs_seq: List[Scalar[DT]],
    actions: List[Scalar[DT]],         # [K, B, ACT_DIM]
    policy_act_tgt: List[Scalar[DT]],  # [K+1, B, ACT_DIM]
    value_tgt: List[Scalar[DT]],
    reward_tgt: List[Scalar[DT]],
    v_min: Scalar[DT],
    v_max: Scalar[DT],
    value_coef: Scalar[DT] = Scalar[DT](0.25),
    consistency_coef: Scalar[DT] = Scalar[DT](2.0),
    policy_coef: Scalar[DT] = Scalar[DT](1.0),
    max_action: Scalar[DT] = Scalar[DT](1.0),
    min_std: Scalar[DT] = Scalar[DT](0.1),
    soft_clamp: Scalar[DT] = Scalar[DT](5.0),
    init_std: Scalar[DT] = Scalar[DT](1.0),
    ent_scale: Scalar[DT] = Scalar[DT](5e-3),
    max_grad_norm: Float64 = 0.0,
    cons_mask: Optional[Pointer[Scalar[DT], MutAnyOrigin]] = None,
    loss_parts: Optional[Pointer[Scalar[DT], MutAnyOrigin]] = None,
) raises -> Scalar[DT]:
    """One CPU continuous EZv2 unroll step. Returns the mean total loss. Mutates
    all five nets via their optimizers. ``obs_seq`` is ``[K+1, B, OBS]``.
    ``cons_mask`` is the optional ``[K, B]`` episode-boundary mask zeroing
    consistency terms whose target obs is absorbing padding (``None`` ≡ all
    ones) — see ``blocks.mojo::ezv2_unroll_train_step_cpu``."""
    comptime MU2 = 2 * ACT_DIM
    comptime PRED_OUT = MU2 + BINS
    comptime DYN_IN = LATENT + ACT_DIM
    comptime DYN_OUT = LATENT + BINS
    comptime PROJ = PROJM.OUT_DIM

    # ── scratch (owned storage Tensors; RAII — no manual free) ──
    var obs0_t = Tensor.alloc(B * OBS)
    for i in range(B * OBS):
        obs0_t.data[i] = obs_seq[i]
    var zst = Tensor.alloc((K + 1) * B * LATENT)
    var z_work = Tensor.alloc(B * LATENT)
    var zk_work = Tensor.alloc(B * LATENT)
    var din = Tensor.alloc(B * DYN_IN)
    var dout = Tensor.alloc(B * DYN_OUT)
    var pout = Tensor.alloc(B * PRED_OUT)
    var gpout = Tensor.alloc(B * PRED_OUT)
    var gdout = Tensor.alloc(B * DYN_OUT)
    var gz = Tensor.alloc(B * LATENT)
    var gpin = Tensor.alloc(B * LATENT)
    var gdin = Tensor.alloc(B * DYN_IN)
    var gobs = Tensor.alloc(B * OBS)
    var twv = Tensor.alloc(B * BINS)
    var twr = Tensor.alloc(B * BINS)
    var obsk_t = Tensor.alloc(B * OBS)
    var tstore = Tensor.alloc(K * B * PROJ)
    var ztmp = Tensor.alloc(B * LATENT)
    var projo = Tensor.alloc(B * PROJ)
    var pk = Tensor.alloc(B * PROJ)
    var gpk = Tensor.alloc(B * PROJ)
    var gproj = Tensor.alloc(B * PROJ)
    var gzcons = Tensor.alloc(B * LATENT)
    # continuous policy-head + per-k target Lists (the loss/two-hot primitives
    # are List-based; continuous policy loss reads/writes the [B, 2*ACT_DIM] slice)
    var musig_l = List[Scalar[DT]](length=B * MU2, fill=0)
    var gmusig_l = List[Scalar[DT]](length=B * MU2, fill=0)
    var ptgt_l = List[Scalar[DT]](length=B * ACT_DIM, fill=0)
    var val_tgt_l = List[Scalar[DT]](length=B, fill=0)
    var rew_tgt_l = List[Scalar[DT]](length=B, fill=0)
    var cons_t_l = List[Scalar[DT]](length=B * PROJ, fill=0)

    var gscale = Scalar[DT](1.0) / Scalar[DT]((K + 1) * B)
    var cscale = consistency_coef / Scalar[DT](K * B)
    var pscale = policy_coef / Scalar[DT]((K + 1) * B)

    # ── forward scan: rep then K dynamics steps, store every z ──
    call_forward["cpu", B](rep, TensorRefs[REP.ARITY](obs0_t), z_work, None)
    for i in range(B * LATENT):
        zst.data[i] = z_work.data[i]

    for k in range(K):
        var zoff = k * B * LATENT
        for b in range(B):
            var dib = b * DYN_IN
            var zb = zoff + b * LATENT
            for i in range(LATENT):
                din.data[dib + i] = zst.data[zb + i]
            for a in range(ACT_DIM):
                din.data[dib + LATENT + a] = actions[(k * B + b) * ACT_DIM + a]
        call_forward["cpu", B](dyn, TensorRefs[DYN.ARITY](din), dout, None)
        var znoff = (k + 1) * B * LATENT
        for b in range(B):
            for i in range(LATENT):
                zst.data[znoff + b * LATENT + i] = dout.data[b * DYN_OUT + i]

    # ── target pre-pass: t_k = g_proj(h(obs_k)), detached, k = 1..K ──
    for k in range(1, K + 1):
        for i in range(B * OBS):
            obsk_t.data[i] = obs_seq[k * B * OBS + i]
        call_forward["cpu", B](rep, TensorRefs[REP.ARITY](obsk_t), ztmp, None)
        call_forward["cpu", B](proj, TensorRefs[PROJM.ARITY](ztmp), projo, None)
        for i in range(B * PROJ):
            tstore.data[(k - 1) * B * PROJ + i] = projo.data[i]

    # ── reverse scan ──
    rep.zero_grad["cpu"](None)
    dyn.zero_grad["cpu"](None)
    pred.zero_grad["cpu"](None)
    proj.zero_grad["cpu"](None)
    predh.zero_grad["cpu"](None)

    var loss = Scalar[DT](0.0)
    var l_pol = Scalar[DT](0.0)
    var l_val = Scalar[DT](0.0)
    var l_rew = Scalar[DT](0.0)
    var l_cons = Scalar[DT](0.0)
    for rk in range(K + 1):
        var k = K - rk
        var zoff = k * B * LATENT
        for i in range(B * LATENT):
            zk_work.data[i] = zst.data[zoff + i]

        # (a) prediction head: re-forward for cache, seed grads, vjp → grad z_k
        call_forward["cpu", B](pred, TensorRefs[PRED.ARITY](zk_work), pout, None)
        # policy: squashed-Gaussian NLL over the [0, 2*ACT_DIM) slice.
        for b in range(B):
            for i in range(MU2):
                musig_l[b * MU2 + i] = pout.data[b * PRED_OUT + i]
            for d in range(ACT_DIM):
                ptgt_l[b * ACT_DIM + d] = policy_act_tgt[
                    (k * B + b) * ACT_DIM + d
                ]
        var l_pol_k = policy_coef * continuous_policy_loss_and_grad[
            B, ACT_DIM
        ](
            musig_l, ptgt_l, pscale, gmusig_l,
            max_action, min_std, soft_clamp, init_std, ent_scale,
        )
        loss += l_pol_k
        l_pol += l_pol_k
        for b in range(B):
            for i in range(MU2):
                gpout.data[b * PRED_OUT + i] = gmusig_l[b * MU2 + i]
        # value: categorical soft-CE over [2*ACT_DIM, 2*ACT_DIM+BINS).
        for b in range(B):
            val_tgt_l[b] = value_tgt[k * B + b]
        mz_two_hot_target_batch[B, BINS](
            val_tgt_l, 0, v_min, v_max, twv.data, 0
        )
        var l_val_k = value_coef * soft_ce_slice_loss_and_grad[
            B, PRED_OUT, MU2, BINS
        ](pout.data, twv.data, gscale * value_coef, gpout.data)
        loss += l_val_k
        l_val += l_val_k
        call_vjp["cpu", B](
            pred,
            TensorRefs[PRED.ARITY](zk_work),
            gpout,
            TensorRefs[PRED.ARITY](gpin),
            None,
        )

        # (b) consistency online branch (k >= 1): p_k = h_pred(g_proj(z_k))
        if k >= 1:
            call_forward["cpu", B](proj, TensorRefs[PROJM.ARITY](zk_work), projo, None)
            call_forward["cpu", B](predh, TensorRefs[PREDH.ARITY](projo), pk, None)
            for i in range(B * PROJ):
                cons_t_l[i] = tstore.data[(k - 1) * B * PROJ + i]
            var mk = Optional[Pointer[Scalar[DT], MutAnyOrigin]](None)
            if cons_mask:
                mk = cons_mask.value().unsafe_offset((k - 1) * B)
            var l_cons_k = consistency_loss_and_grad[B, PROJ](
                pk.data, cons_t_l, cscale, gpk.data, mask=mk
            )
            loss += l_cons_k
            l_cons += l_cons_k
            call_vjp["cpu", B](
                predh,
                TensorRefs[PREDH.ARITY](projo),
                gpk,
                TensorRefs[PREDH.ARITY](gproj),
                None,
            )
            call_vjp["cpu", B](
                proj,
                TensorRefs[PROJM.ARITY](zk_work),
                gproj,
                TensorRefs[PROJM.ARITY](gzcons),
                None,
            )
            for i in range(B * LATENT):
                gpin.data[i] += gzcons.data[i]

        # (c) dynamics: carry grad from z_{k+1} + reward head, ½ on hidden input
        if k < K:
            for b in range(B):
                var dib = b * DYN_IN
                var zb = zoff + b * LATENT
                for i in range(LATENT):
                    din.data[dib + i] = zst.data[zb + i]
                for a in range(ACT_DIM):
                    din.data[dib + LATENT + a] = actions[
                        (k * B + b) * ACT_DIM + a
                    ]
            call_forward["cpu", B](dyn, TensorRefs[DYN.ARITY](din), dout, None)
            for b in range(B):
                for i in range(LATENT):
                    gdout.data[b * DYN_OUT + i] = gz.data[b * LATENT + i]
            for b in range(B):
                rew_tgt_l[b] = reward_tgt[k * B + b]
            mz_two_hot_target_batch[B, BINS](
                rew_tgt_l, 0, v_min, v_max, twr.data, 0
            )
            var l_rew_k = soft_ce_slice_loss_and_grad[B, DYN_OUT, LATENT, BINS](
                dout.data, twr.data, gscale, gdout.data
            )
            loss += l_rew_k
            l_rew += l_rew_k
            call_vjp["cpu", B](
                dyn,
                TensorRefs[DYN.ARITY](din),
                gdout,
                TensorRefs[DYN.ARITY](gdin),
                None,
            )
            for b in range(B):
                for i in range(LATENT):
                    gpin.data[b * LATENT + i] += (
                        Scalar[DT](0.5) * gdin.data[b * DYN_IN + i]
                    )

        for i in range(B * LATENT):
            gz.data[i] = gpin.data[i]

    # ── rep: re-forward obs0 (cache clobbered by target pre-pass), then vjp ──
    call_forward["cpu", B](rep, TensorRefs[REP.ARITY](obs0_t), z_work, None)
    call_vjp["cpu", B](
        rep, TensorRefs[REP.ARITY](obs0_t), gz, TensorRefs[REP.ARITY](gobs), None
    )

    # Global grad-norm clip per net (max_grad_norm <= 0 ⇒ no-op), then step.
    _ = clip_grad_norm["cpu", PRED](pred, Scalar[DT](max_grad_norm), None)
    opred.begin_step()
    walk_params["cpu"](pred, opred, None)
    _ = clip_grad_norm["cpu", DYN](dyn, Scalar[DT](max_grad_norm), None)
    odyn.begin_step()
    walk_params["cpu"](dyn, odyn, None)
    _ = clip_grad_norm["cpu", REP](rep, Scalar[DT](max_grad_norm), None)
    orep.begin_step()
    walk_params["cpu"](rep, orep, None)
    _ = clip_grad_norm["cpu", PROJM](proj, Scalar[DT](max_grad_norm), None)
    oproj.begin_step()
    walk_params["cpu"](proj, oproj, None)
    _ = clip_grad_norm["cpu", PREDH](predh, Scalar[DT](max_grad_norm), None)
    opredh.begin_step()
    walk_params["cpu"](predh, opredh, None)

    if loss_parts:
        var lp = loss_parts.value()
        var inv = Scalar[DT](1.0) / Scalar[DT](B)
        lp[unsafe_offset=0] = l_pol * inv   # policy
        lp[unsafe_offset=1] = l_val * inv   # value
        lp[unsafe_offset=2] = l_rew * inv   # reward
        lp[unsafe_offset=3] = l_cons * inv  # consistency
    return loss / Scalar[DT](B)


# ─────────────────────────────────────────────────────────────────────────
# GPU path
# ─────────────────────────────────────────────────────────────────────────


def _ez_build_dyn_in_cont_k[
    B_: Int, LATENT_: Int, ACT_DIM_: Int, DYN_IN_: Int,
](
    din: Tile[DT, B_ * DYN_IN_],
    zk: Tile[DT, B_ * LATENT_],
    act: Tile[DT, B_ * ACT_DIM_],
):
    """Assemble the continuous dynamics input row ``[z_k | a_k]`` per sample —
    the continuous twin of ``_mz_build_dyn_in_k`` (which one-hots a discrete
    index). ``act`` is the action **vector** slab ``[B, ACT_DIM]`` for this
    unroll step; its ``ACT_DIM`` dims drop straight into the trailing slots."""
    var idx = Int(global_idx.x)
    if idx < B_ * DYN_IN_:
        var b = idx // DYN_IN_
        var d = idx % DYN_IN_
        if d < LATENT_:
            din[idx] = rebind[Scalar[DT]](zk[b * LATENT_ + d])
        else:
            din[idx] = rebind[Scalar[DT]](act[b * ACT_DIM_ + (d - LATENT_)])


def ezv2_unroll_train_step_continuous_gpu[
    REP: Module,
    DYN: Module,
    PRED: Module,
    PROJM: Module,
    PREDH: Module,
    B: Int,
    K: Int,
    OBS: Int,
    ACT_DIM: Int,
    LATENT: Int,
    BINS: Int,
](
    ctx: DeviceContext,
    mut scratch: EZV2UnrollContScratch[
        B, K, OBS, ACT_DIM, LATENT, BINS, PROJM.OUT_DIM
    ],
    mut rep: REP,
    mut dyn: DYN,
    mut pred: PRED,
    mut proj: PROJM,
    mut predh: PREDH,
    mut orep: Adam,
    mut odyn: Adam,
    mut opred: Adam,
    mut oproj: Adam,
    mut opredh: Adam,
    obs_seq: List[Scalar[DT]],
    actions: List[Scalar[DT]],         # [K, B, ACT_DIM]
    policy_act_tgt: List[Scalar[DT]],  # [K+1, B, ACT_DIM]
    value_tgt: List[Scalar[DT]],
    reward_tgt: List[Scalar[DT]],
    v_min: Scalar[DT],
    v_max: Scalar[DT],
    value_coef: Scalar[DT] = Scalar[DT](0.25),
    consistency_coef: Scalar[DT] = Scalar[DT](2.0),
    policy_coef: Scalar[DT] = Scalar[DT](1.0),
    max_action: Scalar[DT] = Scalar[DT](1.0),
    min_std: Scalar[DT] = Scalar[DT](0.1),
    soft_clamp: Scalar[DT] = Scalar[DT](5.0),
    init_std: Scalar[DT] = Scalar[DT](1.0),
    ent_scale: Scalar[DT] = Scalar[DT](5e-3),
    max_grad_norm: Float64 = 0.0,
    cons_mask: Optional[Pointer[Scalar[DT], MutAnyOrigin]] = None,
    loss_parts: Optional[Pointer[Scalar[DT], MutAnyOrigin]] = None,
) raises -> Scalar[DT]:
    """GPU continuous EZv2 K-step unroll — device mirror of
    ``ezv2_unroll_train_step_continuous_cpu`` (MuZero BPTT + SimSiam consistency
    + squashed-Gaussian policy NLL). Host time-major batch slabs:
    ``obs_seq[K+1,B,OBS]``, ``actions[K,B,ACT_DIM]`` (action vectors),
    ``policy_act_tgt[K+1,B,ACT_DIM]``, ``value_tgt[K+1,B]``, ``reward_tgt[K,B]``.
    Device + host scratch is the persistent ``EZV2UnrollContScratch`` (allocated
    once in ``make``). Returns the mean total loss."""
    comptime MU2 = 2 * ACT_DIM
    comptime PRED_OUT = MU2 + BINS
    comptime DYN_IN = LATENT + ACT_DIM
    comptime DYN_OUT = LATENT + BINS
    comptime PROJ = PROJM.OUT_DIM

    var octx = Optional[DeviceContext](ctx)
    var gscale = Scalar[DT](1.0) / Scalar[DT]((K + 1) * B)
    var cscale = consistency_coef / Scalar[DT](K * B)
    var pscale = policy_coef / Scalar[DT]((K + 1) * B)

    # ── H2D the host batch slabs (sanctioned list.unsafe_ptr() staging) ──
    ctx.enqueue_copy(scratch.d_obs.dev.value(), obs_seq.unsafe_ptr())
    ctx.enqueue_copy(scratch.d_act.dev.value(), actions.unsafe_ptr())
    ctx.enqueue_copy(scratch.d_pol.dev.value(), policy_act_tgt.unsafe_ptr())
    ctx.enqueue_copy(scratch.d_val.dev.value(), value_tgt.unsafe_ptr())
    ctx.enqueue_copy(scratch.d_rew.dev.value(), reward_tgt.unsafe_ptr())
    if cons_mask:
        ctx.enqueue_copy(scratch.d_cmask.dev.value(), cons_mask.value())
    else:
        ctx.enqueue_copy(
            scratch.d_cmask.dev.value(), scratch.h_cmask_ones.value()
        )

    # ── persistent scratch as owned-Tensor refs (device views via .lt/.lt_at) ──
    ref d_obs = scratch.d_obs
    ref d_act = scratch.d_act
    ref d_pol = scratch.d_pol
    ref d_val = scratch.d_val
    ref d_rew = scratch.d_rew
    ref d_obs_work = scratch.d_obs_work
    ref z_work = scratch.z_work
    ref zk_work = scratch.zk_work
    ref zst = scratch.d_zst
    ref din = scratch.d_din
    ref dout = scratch.d_dout
    ref pout = scratch.d_pout
    ref gpout = scratch.d_gpout
    ref gdout = scratch.d_gdout
    ref gz = scratch.d_gz
    ref gpin = scratch.d_gpin
    ref gdin = scratch.d_gdin
    ref gobs = scratch.d_gobs
    ref twv = scratch.d_twv
    ref twr = scratch.d_twr
    ref loss_d = scratch.d_loss
    ref tstore = scratch.d_tstore
    ref ztmp = scratch.d_ztmp
    ref projo = scratch.d_projo
    ref pk = scratch.d_pk
    ref gpk = scratch.d_gpk
    ref gproj = scratch.d_gproj
    ref gzcons = scratch.d_gzcons
    ref d_cmask = scratch.d_cmask

    # zero the 4 loss-component accumulators (policy|value|reward|consistency)
    var h_loss = scratch.h_loss.value()
    for i in range(4 * B):
        h_loss.unsafe_ptr()[unsafe_offset=i] = Scalar[DT](0.0)
    ctx.enqueue_copy(loss_d.dev.value(), h_loss)

    # device-view layouts (built off the storage Tensors via .lt / .lt_at)
    comptime LB = RowMajor[B]
    comptime LBOBS = RowMajor[B * OBS]
    comptime LBL = RowMajor[B * LATENT]
    comptime LBDI = RowMajor[B * DYN_IN]
    comptime LBDO = RowMajor[B * DYN_OUT]
    comptime LBPO = RowMajor[B * PRED_OUT]
    comptime LBBINS = RowMajor[B * BINS]
    comptime LBAD = RowMajor[B * ACT_DIM]
    comptime LBPROJ = RowMajor[B * PROJ]

    comptime nbDIN = (B * DYN_IN + TPB - 1) // TPB
    comptime nbLAT = (B * LATENT + TPB - 1) // TPB
    comptime nbOBS = (B * OBS + TPB - 1) // TPB
    comptime nbPROJ = (B * PROJ + TPB - 1) // TPB
    comptime nbB = (B + TPB - 1) // TPB
    comptime kBuild = _ez_build_dyn_in_cont_k[B, LATENT, ACT_DIM, DYN_IN]
    comptime kCopyL = _mz_copy_latent_k[B, LATENT, DYN_OUT]
    comptime kPol = continuous_policy_loss_grad_k[B, ACT_DIM, PRED_OUT]
    comptime kValCE = _mz_softce_slice_k[B, PRED_OUT, MU2, BINS]
    comptime kRewCE = _mz_softce_slice_k[B, DYN_OUT, LATENT, BINS]
    comptime kTwoHot = _mz_twohot_k[B, BINS]
    comptime kCarry = _mz_set_carry_latent_k[B, LATENT, DYN_OUT]
    comptime kHalf = _mz_accum_half_k[B, LATENT, DYN_IN]
    comptime kBcopy = _mz_bcopy_k[B * LATENT]
    comptime kBcopyOBS = _mz_bcopy_k[B * OBS]
    comptime kBcopyPROJ = _mz_bcopy_k[B * PROJ]
    comptime kCons = consistency_loss_grad_k[B, PROJ]
    comptime kAccum = _ez_accum_latent_k[B * LATENT]

    # ── forward scan: z0 = h(obs0); z_{k+1} = g(z_k, a_k).latent ──
    ctx.enqueue_function[kBcopyOBS](
        d_obs.tt_at["gpu", B * OBS](0), d_obs_work.tt["gpu", B * OBS](),
        grid_dim=nbOBS, block_dim=TPB,
    )
    call_forward["gpu", B](rep, TensorRefs[REP.ARITY](d_obs_work), z_work, octx)
    ctx.enqueue_function[kBcopy](
        z_work.tt["gpu", B * LATENT](), zst.tt_at["gpu", B * LATENT](0),
        grid_dim=nbLAT, block_dim=TPB,
    )
    for k in range(K):
        ctx.enqueue_function[kBuild](
            din.tt["gpu", B * DYN_IN](),
            zst.tt_at["gpu", B * LATENT](k * B * LATENT),
            d_act.tt_at["gpu", B * ACT_DIM](k * B * ACT_DIM),
            grid_dim=nbDIN, block_dim=TPB,
        )
        call_forward["gpu", B](dyn, TensorRefs[DYN.ARITY](din), dout, octx)
        ctx.enqueue_function[kCopyL](
            zst.tt_at["gpu", B * LATENT]((k + 1) * B * LATENT),
            dout.tt["gpu", B * DYN_OUT](),
            grid_dim=nbLAT, block_dim=TPB,
        )

    # ── target pre-pass: t_k = g_proj(h(obs_k)), detached, k = 1..K ──
    for k in range(1, K + 1):
        ctx.enqueue_function[kBcopyOBS](
            d_obs.tt_at["gpu", B * OBS](k * B * OBS),
            d_obs_work.tt["gpu", B * OBS](),
            grid_dim=nbOBS, block_dim=TPB,
        )
        call_forward["gpu", B](rep, TensorRefs[REP.ARITY](d_obs_work), ztmp, octx)
        call_forward["gpu", B](proj, TensorRefs[PROJM.ARITY](ztmp), projo, octx)
        ctx.enqueue_function[kBcopyPROJ](
            projo.tt["gpu", B * PROJ](),
            tstore.tt_at["gpu", B * PROJ]((k - 1) * B * PROJ),
            grid_dim=nbPROJ, block_dim=TPB,
        )

    # ── reverse scan ──
    rep.zero_grad["gpu"](octx)
    dyn.zero_grad["gpu"](octx)
    pred.zero_grad["gpu"](octx)
    proj.zero_grad["gpu"](octx)
    predh.zero_grad["gpu"](octx)

    for rk in range(K + 1):
        var k = K - rk
        ctx.enqueue_function[kBcopy](
            zst.tt_at["gpu", B * LATENT](k * B * LATENT),
            zk_work.tt["gpu", B * LATENT](),
            grid_dim=nbLAT, block_dim=TPB,
        )

        # (a) prediction head: forward (cache), seed grads, vjp → grad z_k
        call_forward["gpu", B](pred, TensorRefs[PRED.ARITY](zk_work), pout, octx)
        # policy: squashed-Gaussian NLL over the [0, 2*ACT_DIM) slice.
        ctx.enqueue_function[kPol](
            pout.tt["gpu", B * PRED_OUT](),
            d_pol.tt_at["gpu", B * ACT_DIM](k * B * ACT_DIM),
            gpout.tt["gpu", B * PRED_OUT](),
            loss_d.tt_at["gpu", B](0),
            pscale, policy_coef,
            max_action, min_std, soft_clamp, init_std, ent_scale,
            grid_dim=nbB, block_dim=TPB,
        )
        # value: categorical soft-CE over [2*ACT_DIM, 2*ACT_DIM+BINS).
        ctx.enqueue_function[kTwoHot](
            twv.tt["gpu", B * BINS](), d_val.tt_at["gpu", B](k * B),
            v_min, v_max, grid_dim=nbB, block_dim=TPB,
        )
        ctx.enqueue_function[kValCE](
            pout.tt["gpu", B * PRED_OUT](),
            twv.tt["gpu", B * BINS](),
            gpout.tt["gpu", B * PRED_OUT](),
            loss_d.tt_at["gpu", B](B),                # value block
            gscale * value_coef, value_coef,
            grid_dim=nbB, block_dim=TPB,
        )
        call_vjp["gpu", B](
            pred,
            TensorRefs[PRED.ARITY](zk_work), gpout,
            TensorRefs[PRED.ARITY](gpin), octx,
        )

        # (b) consistency online branch (k >= 1): p_k = h_pred(g_proj(z_k))
        if k >= 1:
            call_forward["gpu", B](proj, TensorRefs[PROJM.ARITY](zk_work), projo, octx)
            call_forward["gpu", B](predh, TensorRefs[PREDH.ARITY](projo), pk, octx)
            ctx.enqueue_function[kCons](
                pk.tt["gpu", B * PROJ](),
                tstore.tt_at["gpu", B * PROJ]((k - 1) * B * PROJ),
                gpk.tt["gpu", B * PROJ](),
                loss_d.tt_at["gpu", B](3 * B),            # consistency block
                d_cmask.tt_at["gpu", B]((k - 1) * B),     # boundary mask row k
                cscale, Scalar[DT](1.0),
                grid_dim=nbB, block_dim=TPB,
            )
            call_vjp["gpu", B](
                predh,
                TensorRefs[PREDH.ARITY](projo), gpk,
                TensorRefs[PREDH.ARITY](gproj), octx,
            )
            call_vjp["gpu", B](
                proj,
                TensorRefs[PROJM.ARITY](zk_work), gproj,
                TensorRefs[PROJM.ARITY](gzcons), octx,
            )
            ctx.enqueue_function[kAccum](
                gpin.tt["gpu", B * LATENT](),
                gzcons.tt["gpu", B * LATENT](),
                grid_dim=nbLAT, block_dim=TPB,
            )

        # (c) dynamics: carry grad from z_{k+1} + reward head, ½ on hidden input
        if k < K:
            ctx.enqueue_function[kBuild](
                din.tt["gpu", B * DYN_IN](),
                zst.tt_at["gpu", B * LATENT](k * B * LATENT),
                d_act.tt_at["gpu", B * ACT_DIM](k * B * ACT_DIM),
                grid_dim=nbDIN, block_dim=TPB,
            )
            call_forward["gpu", B](dyn, TensorRefs[DYN.ARITY](din), dout, octx)
            ctx.enqueue_function[kCarry](
                gdout.tt["gpu", B * DYN_OUT](),
                gz.tt["gpu", B * LATENT](),
                grid_dim=nbLAT, block_dim=TPB,
            )
            ctx.enqueue_function[kTwoHot](
                twr.tt["gpu", B * BINS](), d_rew.tt_at["gpu", B](k * B),
                v_min, v_max, grid_dim=nbB, block_dim=TPB,
            )
            ctx.enqueue_function[kRewCE](
                dout.tt["gpu", B * DYN_OUT](),
                twr.tt["gpu", B * BINS](),
                gdout.tt["gpu", B * DYN_OUT](),
                loss_d.tt_at["gpu", B](2 * B),            # reward block
                gscale, Scalar[DT](1.0),
                grid_dim=nbB, block_dim=TPB,
            )
            call_vjp["gpu", B](
                dyn,
                TensorRefs[DYN.ARITY](din), gdout,
                TensorRefs[DYN.ARITY](gdin), octx,
            )
            ctx.enqueue_function[kHalf](
                gpin.tt["gpu", B * LATENT](),
                gdin.tt["gpu", B * DYN_IN](),
                grid_dim=nbLAT, block_dim=TPB,
            )

        # carry ← full grad wrt z_k for the next (k-1) iteration
        ctx.enqueue_function[kBcopy](
            gpin.tt["gpu", B * LATENT](),
            gz.tt["gpu", B * LATENT](),
            grid_dim=nbLAT, block_dim=TPB,
        )

    # ── rep: re-forward obs0 (cache clobbered by target pre-pass), then vjp ──
    ctx.enqueue_function[kBcopyOBS](
        d_obs.tt_at["gpu", B * OBS](0), d_obs_work.tt["gpu", B * OBS](),
        grid_dim=nbOBS, block_dim=TPB,
    )
    call_forward["gpu", B](rep, TensorRefs[REP.ARITY](d_obs_work), z_work, octx)
    call_vjp["gpu", B](
        rep,
        TensorRefs[REP.ARITY](d_obs_work), gz,
        TensorRefs[REP.ARITY](gobs), octx,
    )

    # Global grad-norm clip per net (max_grad_norm <= 0 ⇒ no-op), then step.
    _ = clip_grad_norm["gpu", PRED](pred, Scalar[DT](max_grad_norm), octx)
    opred.begin_step(); walk_params["gpu"](pred, opred, octx)
    _ = clip_grad_norm["gpu", DYN](dyn, Scalar[DT](max_grad_norm), octx)
    odyn.begin_step(); walk_params["gpu"](dyn, odyn, octx)
    _ = clip_grad_norm["gpu", REP](rep, Scalar[DT](max_grad_norm), octx)
    orep.begin_step(); walk_params["gpu"](rep, orep, octx)
    _ = clip_grad_norm["gpu", PROJM](proj, Scalar[DT](max_grad_norm), octx)
    oproj.begin_step(); walk_params["gpu"](proj, oproj, octx)
    _ = clip_grad_norm["gpu", PREDH](predh, Scalar[DT](max_grad_norm), octx)
    opredh.begin_step(); walk_params["gpu"](predh, opredh, octx)

    # ── reduce loss (D2H once) — 4 [B] blocks: policy|value|reward|consistency ──
    ctx.enqueue_copy(h_loss, loss_d.dev.value())
    ctx.synchronize()
    var hp = h_loss.unsafe_ptr()
    var l_pol = Scalar[DT](0.0)
    var l_val = Scalar[DT](0.0)
    var l_rew = Scalar[DT](0.0)
    var l_cons = Scalar[DT](0.0)
    for b in range(B):
        l_pol += hp[unsafe_offset=b]
        l_val += hp[unsafe_offset=B + b]
        l_rew += hp[unsafe_offset=2 * B + b]
        l_cons += hp[unsafe_offset=3 * B + b]
    var inv = Scalar[DT](1.0) / Scalar[DT](B)
    if loss_parts:
        var lp = loss_parts.value()
        lp[unsafe_offset=0] = l_pol * inv
        lp[unsafe_offset=1] = l_val * inv
        lp[unsafe_offset=2] = l_rew * inv
        lp[unsafe_offset=3] = l_cons * inv
    return (l_pol + l_val + l_rew + l_cons) * inv
