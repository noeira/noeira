"""EfficientZeroV2 K-step unroll — MuZero BPTT + SimSiam temporal consistency.

The EZv2 training step is the MuZero unroll (`muzero/blocks.mojo`) with one
addition: a **SimSiam consistency** branch at every dynamics step. For each
unroll position ``k = 1..K`` (no consistency at the root k=0):

    online :  p_k = h_pred(g_proj(z_k))          (z_k = the rolled dynamics latent)
    target :  t_k = sg( g_proj(h(obs_k)) )         (g_proj of the *real* future obs)
    L_G    += −cos(p_k, t_k)                        (stop-grad on t_k)

The consistency gradient flows ``p_k → h_pred → g_proj → z_k`` and is added to
the same per-step latent gradient accumulator ``∂L/∂z_k`` that the policy/value
head feeds — so it then propagates back through the dynamics with the MuZero
½ scaling, exactly like every other contribution to ``z_k``. The target branch
is detached (computed in a pre-pass into ``t_store``, never backpropped).

Cache discipline (the nn re-forward-before-vjp idiom): the target pre-pass runs
``h(obs_k)`` which clobbers the representation net's forward cache, so the final
``rep.vjp`` is preceded by a fresh ``rep.forward(obs0)``. Within the reverse
scan, ``g_proj``/``h_pred`` are re-forwarded on the *online* input immediately
before their ``vjp`` so their caches hold the live (online) activations.

Batch layout is time-major like MuZero, except obs is the **full sequence**
``obs_seq[K+1, B, OBS]`` (``obs_seq[0] == obs0``) so the consistency targets can
encode the real future observations: ``actions[K,B]`` (indices),
``policy_tgt[K+1,B,ACT]``, ``value_tgt[K+1,B]``, ``reward_tgt[K,B]`` (raw).
``cons_mask[K,B]`` (optional) is the reference's ``mask_batch`` applied to the
consistency term only: rows whose ``obs_seq[k]`` is absorbing obs-repeat
padding (the window crossed the episode terminal) are zeroed, so the dynamics
is never trained toward the false "terminal obs is a fixed point" target.

CPU path first (overfit-tested); a GPU branch + CPU↔GPU parity follow.
"""

from noeira.nn.core.param import walk_params
from std.memory import alloc
from std.math import exp, log
from std.time import perf_counter_ns
from layout import TileTensor, row_major
from noeira.core.tiles import RowMajor, Tile
from max.gpu import global_idx
from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer

from noeira.nn.constants import DT, TPB
from noeira.nn.core.module import Module
from noeira.nn.core.tensor import Tensor, TensorImpl
from noeira.nn.primitives.linear import _cast_f2b_kernel, _cast_b2f_kernel
from noeira.nn.core.tensor_refs import TensorRefs, child_refs
from noeira.nn.core.call import call_forward, call_vjp
from noeira.nn.optimizer.adam import Adam
from noeira.nn.optimizer.optimizer import Optimizer
from noeira.nn.optimizer.grad_clip import clip_grad_norm

from .loss_ops import consistency_loss_and_grad, consistency_loss_grad_k
from .unroll_scratch import EZV2UnrollScratch
from ..muzero.loss_ops import soft_ce_slice_loss_and_grad
from ..muzero.blocks import (
    _mz_build_dyn_in_k,
    _mz_copy_latent_k,
    _mz_softce_slice_k,
    _mz_twohot_k,
    _mz_set_carry_latent_k,
    _mz_accum_half_k,
    _mz_bcopy_k,
)
from ..zero.twohot_targets import mz_two_hot_target_batch
from .nets_atari import EZRewardLSTMAtari


def _a(n: Int) -> Pointer[Scalar[DT], MutAnyOrigin]:
    """Raw host scratch for optional unroll outputs (loss_parts) — function-local;
    the unroll's optional-output params are Optional[Pointer]."""
    return alloc[Scalar[DT]]({count = n}).unsafe_leak().as_unsafe_any_origin()


# ── PER kernels (6c-2) ──────────────────────────────────────────────────
def _ez_scale_rows_k[B_: Int, ROW_: Int, OFF_: Int, LEN_: Int, ADT: DType = DT](
    grad: Tile[ADT, B_ * ROW_],
    w: Tile[DT, B_],
):
    """Scale the ``[OFF_, OFF_+LEN_)`` column slice of grad row ``b`` by the
    per-sample importance-sampling weight ``w[b]`` (PER gradient weighting).
    One thread per row."""
    var b = Int(global_idx.x)
    if b < B_:
        var wb = rebind[Scalar[DT]](w[b])
        var base = b * ROW_ + OFF_
        for c in range(LEN_):
            grad[base + c] = (
                rebind[Scalar[ADT]](grad[base + c]).cast[DT]() * wb
            ).cast[ADT]()


def _ez_priority_ce_k[
    B_: Int, ROW_: Int, OFF_: Int, NBINS_: Int, ADT: DType = DT
](
    logits: Tile[ADT, B_ * ROW_],
    target: Tile[DT, B_ * NBINS_],
    out_prio: Tile[DT, B_],
):
    """Per-sample soft-CE of the value-head slice vs the value two-hot target —
    the PER priority signal (root value-prediction error). Writes (does NOT
    accumulate) into ``out_prio[b]``. One thread per row; mirrors the loss half
    of `_mz_softce_slice_k`."""
    var b = Int(global_idx.x)
    if b < B_:
        var base = b * ROW_ + OFF_
        var m = rebind[Scalar[ADT]](logits[base]).cast[DT]()
        for i in range(1, NBINS_):
            var v = rebind[Scalar[ADT]](logits[base + i]).cast[DT]()
            if v > m:
                m = v
        var s = Scalar[DT](0.0)
        for i in range(NBINS_):
            s += exp(rebind[Scalar[ADT]](logits[base + i]).cast[DT]() - m)
        var log_s = log(s)
        var tb = b * NBINS_
        var row_loss = Scalar[DT](0.0)
        for i in range(NBINS_):
            var q = rebind[Scalar[DT]](target[tb + i])
            row_loss += -q * (
                (rebind[Scalar[ADT]](logits[base + i]).cast[DT]() - m) - log_s
            )
        out_prio[b] = row_loss


def ezv2_unroll_train_step_cpu[
    REP: Module,
    DYN: Module,
    PRED: Module,
    PROJM: Module,
    PREDH: Module,
    B: Int,
    K: Int,
    OBS: Int,
    ACT: Int,
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
    actions: List[Scalar[DT]],
    policy_tgt: List[Scalar[DT]],
    value_tgt: List[Scalar[DT]],
    reward_tgt: List[Scalar[DT]],
    v_min: Scalar[DT],
    v_max: Scalar[DT],
    value_coef: Scalar[DT] = Scalar[DT](1.0),
    consistency_coef: Scalar[DT] = Scalar[DT](2.0),
    max_grad_norm: Float64 = 0.0,
    cons_mask: Optional[Pointer[Scalar[DT], MutAnyOrigin]] = None,
    loss_parts: Optional[Pointer[Scalar[DT], MutAnyOrigin]] = None,
) raises -> Scalar[DT]:
    """One CPU EZv2 unroll training step (MuZero BPTT + SimSiam consistency).

    Returns the mean total loss (policy + value + reward + consistency). Mutates
    all five nets via their optimizers. ``obs_seq`` is the time-major
    ``[K+1, B, OBS]`` observation sequence (``obs_seq[0]`` is the root obs).
    ``cons_mask`` is the optional ``[K, B]`` episode-boundary mask (reference
    ``mask_batch``): row ``(k-1, b)`` is 0 when ``obs_seq[k]`` is absorbing
    obs-repeat padding rather than a real future observation, zeroing that
    consistency term (``None`` ≡ all ones, the legacy unmasked behaviour).
    """
    comptime PRED_OUT = ACT + BINS
    comptime DYN_IN = LATENT + ACT
    comptime DYN_OUT = LATENT + BINS
    comptime PROJ = PROJM.OUT_DIM

    # ── scratch (owned storage Tensors; RAII — no manual free) ──
    var obs0_t = Tensor.alloc(B * OBS)    # rep input bridge (copy from obs_seq[0])
    for i in range(B * OBS):
        obs0_t.data[i] = obs_seq[i]
    var zst = Tensor.alloc((K + 1) * B * LATENT)  # stored latents z0..zK
    var z_work = Tensor.alloc(B * LATENT)         # forward output working tile
    var zk_work = Tensor.alloc(B * LATENT)        # reverse-scan zk forward input
    var din = Tensor.alloc(B * DYN_IN)
    var dout = Tensor.alloc(B * DYN_OUT)
    var pout = Tensor.alloc(B * PRED_OUT)
    var gpout = Tensor.alloc(B * PRED_OUT)
    var gdout = Tensor.alloc(B * DYN_OUT)
    var gz = Tensor.alloc(B * LATENT)             # carry: grad wrt z_{k+1}
    var gpin = Tensor.alloc(B * LATENT)           # working grad wrt z_k
    var gdin = Tensor.alloc(B * DYN_IN)
    var gobs = Tensor.alloc(B * OBS)              # grad wrt rep input (discarded)
    var twv = Tensor.alloc(B * BINS)
    var twr = Tensor.alloc(B * BINS)
    # consistency scratch
    var obsk_t = Tensor.alloc(B * OBS)            # target-branch obs_k bridge
    var tstore = Tensor.alloc(K * B * PROJ)       # detached target proj t_1..t_K
    var ztmp = Tensor.alloc(B * LATENT)           # rep(obs_k) for the target branch
    var projo = Tensor.alloc(B * PROJ)            # online g_proj(z_k)
    var pk = Tensor.alloc(B * PROJ)               # online h_pred(projo)
    var gpk = Tensor.alloc(B * PROJ)              # grad wrt p_k
    var gproj = Tensor.alloc(B * PROJ)            # grad wrt projector output
    var gzcons = Tensor.alloc(B * LATENT)         # grad wrt z_k from consistency
    # per-k target slices copied from the raw input pointers into owned Lists
    # (the loss/two-hot primitives are List-based).
    var pol_tgt_l = List[Scalar[DT]](length=B * ACT, fill=0)
    var val_tgt_l = List[Scalar[DT]](length=B, fill=0)
    var rew_tgt_l = List[Scalar[DT]](length=B, fill=0)
    var cons_t_l = List[Scalar[DT]](length=B * PROJ, fill=0)

    var gscale = Scalar[DT](1.0) / Scalar[DT]((K + 1) * B)
    # consistency is summed over K steps (no root term) → 1/K mean.
    var cscale = consistency_coef / Scalar[DT](K * B)

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
            for a in range(ACT):
                din.data[dib + LATENT + a] = Scalar[DT](0.0)
            din.data[dib + LATENT + Int(actions[k * B + b])] = Scalar[DT](1.0)
        call_forward["cpu", B](dyn, TensorRefs[DYN.ARITY](din), dout, None)
        var znoff = (k + 1) * B * LATENT
        for b in range(B):
            for i in range(LATENT):
                zst.data[znoff + b * LATENT + i] = dout.data[b * DYN_OUT + i]

    # ── target pre-pass: t_k = g_proj(h(obs_k)), detached, k = 1..K ──
    # (clobbers rep's cache → rep is re-forwarded before the final rep.vjp)
    for k in range(1, K + 1):
        for i in range(B * OBS):
            obsk_t.data[i] = obs_seq[k * B * OBS + i]
        call_forward["cpu", B](rep, TensorRefs[REP.ARITY](obsk_t), ztmp, None)
        call_forward["cpu", B](proj, TensorRefs[PROJM.ARITY](ztmp), projo, None)
        for i in range(B * PROJ):
            tstore.data[(k - 1) * B * PROJ + i] = projo.data[i]

    # ── reverse scan: accumulate grads + loss ──
    rep.zero_grad["cpu"](None)
    dyn.zero_grad["cpu"](None)
    pred.zero_grad["cpu"](None)
    proj.zero_grad["cpu"](None)
    predh.zero_grad["cpu"](None)

    var loss = Scalar[DT](0.0)
    # per-component loss accumulators (for the optional loss_parts breakdown)
    var l_pol = Scalar[DT](0.0)
    var l_val = Scalar[DT](0.0)
    var l_rew = Scalar[DT](0.0)
    var l_cons = Scalar[DT](0.0)
    for rk in range(K + 1):
        var k = K - rk
        var zoff = k * B * LATENT
        # load z_k into the forward-input working tile
        for i in range(B * LATENT):
            zk_work.data[i] = zst.data[zoff + i]

        # (a) prediction head: re-forward for cache, seed grads, vjp → grad z_k
        call_forward["cpu", B](pred, TensorRefs[PRED.ARITY](zk_work), pout, None)
        for i in range(B * ACT):
            pol_tgt_l[i] = policy_tgt[k * B * ACT + i]
        var l_pol_k = soft_ce_slice_loss_and_grad[B, PRED_OUT, 0, ACT](
            pout.data, pol_tgt_l, gscale, gpout.data
        )
        loss += l_pol_k
        l_pol += l_pol_k
        for b in range(B):
            val_tgt_l[b] = value_tgt[k * B + b]
        mz_two_hot_target_batch[B, BINS](
            val_tgt_l, 0, v_min, v_max, twv.data, 0
        )
        var l_val_k = value_coef * soft_ce_slice_loss_and_grad[
            B, PRED_OUT, ACT, BINS
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
                for a in range(ACT):
                    din.data[dib + LATENT + a] = Scalar[DT](0.0)
                din.data[dib + LATENT + Int(actions[k * B + b])] = Scalar[DT](
                    1.0
                )
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

        # carry ← full grad wrt z_k for the next (k-1) iteration
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
# Value-prefix CPU path (EZ `value_prefix=True`). Separate function rather than
# a comptime branch of the baseline: the data flow differs materially (z'-only
# dynamics + a stateful LSTM reward head whose (h,c) is carried across the K
# unroll steps and reset every HORIZON, and the reward consumes z_{k+1} not the
# fused dyn output). `reward_tgt` is the cumulative value-prefix target ([K,B],
# caller applies `value_prefix_from_rewards` first). The 0.5 recurrent half-grad
# and the gscale/cscale conventions match the baseline exactly; the (h,c)
# recurrent gradient is unscaled (EZ halves only the latent `states` hook).
# ─────────────────────────────────────────────────────────────────────────
def ezv2_unroll_train_step_cpu_vp[
    REP: Module,
    DYNZ: Module,
    PRED: Module,
    PROJM: Module,
    PREDH: Module,
    B: Int,
    K: Int,
    OBS: Int,
    ACT: Int,
    LATENT: Int,
    BINS: Int,
    HIDDEN: Int,
    HORIZON: Int,
](
    mut rep: REP,
    mut dynz: DYNZ,
    mut rew: EZRewardLSTMAtari[BINS],
    mut pred: PRED,
    mut proj: PROJM,
    mut predh: PREDH,
    mut orep: Adam,
    mut odynz: Adam,
    mut orew: Adam,
    mut opred: Adam,
    mut oproj: Adam,
    mut opredh: Adam,
    obs_seq: List[Scalar[DT]],
    actions: List[Scalar[DT]],
    policy_tgt: List[Scalar[DT]],
    value_tgt: List[Scalar[DT]],
    reward_tgt: List[Scalar[DT]],   # cumulative value-prefix targets [K,B]
    v_min: Scalar[DT],
    v_max: Scalar[DT],
    value_coef: Scalar[DT] = Scalar[DT](1.0),
    consistency_coef: Scalar[DT] = Scalar[DT](2.0),
    max_grad_norm: Float64 = 0.0,
    cons_mask: Optional[Pointer[Scalar[DT], MutAnyOrigin]] = None,
    loss_parts: Optional[Pointer[Scalar[DT], MutAnyOrigin]] = None,
) raises -> Scalar[DT]:
    """One CPU EZv2 **value-prefix** unroll training step. Mirrors
    `ezv2_unroll_train_step_cpu` but with the stateful LSTM reward head; see the
    module-level comment above for the data-flow differences. Returns mean total
    loss; mutates all six nets via their optimizers."""
    comptime PRED_OUT = ACT + BINS
    comptime DYN_IN = LATENT + ACT
    comptime HH = B * HIDDEN

    # ── scratch ──
    var obs0_t = Tensor.alloc(B * OBS)
    for i in range(B * OBS):
        obs0_t.data[i] = obs_seq[i]
    var zst = Tensor.alloc((K + 1) * B * LATENT)
    var z_work = Tensor.alloc(B * LATENT)
    var zk_work = Tensor.alloc(B * LATENT)
    var din = Tensor.alloc(B * DYN_IN)
    var dz = Tensor.alloc(B * LATENT)             # dyn_z output z_{k+1}
    var pout = Tensor.alloc(B * PRED_OUT)
    var gpout = Tensor.alloc(B * PRED_OUT)
    var gz = Tensor.alloc(B * LATENT)             # carry grad wrt z_{k+1}
    var gpin = Tensor.alloc(B * LATENT)
    var gdin = Tensor.alloc(B * DYN_IN)
    var gobs = Tensor.alloc(B * OBS)
    var twv = Tensor.alloc(B * BINS)
    var twr = Tensor.alloc(B * BINS)
    # value-prefix (h,c) carry + reward bptt scratch
    var h_store = Tensor.alloc((K + 1) * HH)
    var c_store = Tensor.alloc((K + 1) * HH)
    var hbuf = Tensor.alloc(2 * HH)               # slab0 prev, slab1 out
    var cbuf = Tensor.alloc(2 * HH)
    var cache = Tensor.alloc(B * EZRewardLSTMAtari[BINS].CACHE_SIZE)
    var vp = Tensor.alloc(B * BINS)
    var grad_vp = Tensor.alloc(B * BINS)
    var grad_zp = Tensor.alloc(B * LATENT)
    var dh_carry = Tensor.alloc(HH)
    var dc_carry = Tensor.alloc(HH)
    var dh_prev = Tensor.alloc(HH)
    var dc_prev = Tensor.alloc(HH)
    # consistency scratch
    var obsk_t = Tensor.alloc(B * OBS)
    comptime PROJ = PROJM.OUT_DIM
    var tstore = Tensor.alloc(K * B * PROJ)
    var ztmp = Tensor.alloc(B * LATENT)
    var projo = Tensor.alloc(B * PROJ)
    var pk = Tensor.alloc(B * PROJ)
    var gpk = Tensor.alloc(B * PROJ)
    var gproj = Tensor.alloc(B * PROJ)
    var gzcons = Tensor.alloc(B * LATENT)
    var pol_tgt_l = List[Scalar[DT]](length=B * ACT, fill=0)
    var val_tgt_l = List[Scalar[DT]](length=B, fill=0)
    var rew_tgt_l = List[Scalar[DT]](length=B, fill=0)
    var cons_t_l = List[Scalar[DT]](length=B * PROJ, fill=0)

    var gscale = Scalar[DT](1.0) / Scalar[DT]((K + 1) * B)
    var cscale = consistency_coef / Scalar[DT](K * B)

    # ── forward scan: rep, then K (dyn_z → reward LSTM) steps, store z + (h,c) ──
    call_forward["cpu", B](rep, TensorRefs[REP.ARITY](obs0_t), z_work, None)
    for i in range(B * LATENT):
        zst.data[i] = z_work.data[i]
    # h_store[0]/c_store[0] = 0 (Tensor.alloc zero-fills)
    for k in range(K):
        var zoff = k * B * LATENT
        for b in range(B):
            var dib = b * DYN_IN
            for i in range(LATENT):
                din.data[dib + i] = zst.data[zoff + b * LATENT + i]
            for a in range(ACT):
                din.data[dib + LATENT + a] = Scalar[DT](0.0)
            din.data[dib + LATENT + Int(actions[k * B + b])] = Scalar[DT](1.0)
        call_forward["cpu", B](dynz, TensorRefs[DYNZ.ARITY](din), dz, None)
        var znoff = (k + 1) * B * LATENT
        for i in range(B * LATENT):
            zst.data[znoff + i] = dz.data[i]
        # reward LSTM step: (h,c) prev = h_store[k] → out slab → h_store[k+1]
        for i in range(HH):
            hbuf.data[i] = h_store.data[k * HH + i]
            cbuf.data[i] = c_store.data[k * HH + i]
        rew.reward_step_forward["cpu", B](dz, hbuf, cbuf, cache, vp, None)
        for i in range(HH):
            h_store.data[(k + 1) * HH + i] = hbuf.data[HH + i]
            c_store.data[(k + 1) * HH + i] = cbuf.data[HH + i]
        # horizon reset: zero the state CARRIED into the next step
        if (k + 1) % HORIZON == 0:
            for i in range(HH):
                h_store.data[(k + 1) * HH + i] = Scalar[DT](0.0)
                c_store.data[(k + 1) * HH + i] = Scalar[DT](0.0)

    # ── consistency target pre-pass: t_k = g_proj(h(obs_k)), k=1..K ──
    for k in range(1, K + 1):
        for i in range(B * OBS):
            obsk_t.data[i] = obs_seq[k * B * OBS + i]
        call_forward["cpu", B](rep, TensorRefs[REP.ARITY](obsk_t), ztmp, None)
        call_forward["cpu", B](proj, TensorRefs[PROJM.ARITY](ztmp), projo, None)
        for i in range(B * PROJ):
            tstore.data[(k - 1) * B * PROJ + i] = projo.data[i]

    # ── reverse scan ──
    rep.zero_grad["cpu"](None)
    dynz.zero_grad["cpu"](None)
    rew.zero_grad["cpu"](None)
    pred.zero_grad["cpu"](None)
    proj.zero_grad["cpu"](None)
    predh.zero_grad["cpu"](None)
    # zero the recurrent (h,c) grad carry (no future at the last step)
    for i in range(HH):
        dh_carry.data[i] = Scalar[DT](0.0)
        dc_carry.data[i] = Scalar[DT](0.0)

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

        # (a) prediction head on z_k
        call_forward["cpu", B](pred, TensorRefs[PRED.ARITY](zk_work), pout, None)
        for i in range(B * ACT):
            pol_tgt_l[i] = policy_tgt[k * B * ACT + i]
        var l_pol_k = soft_ce_slice_loss_and_grad[B, PRED_OUT, 0, ACT](
            pout.data, pol_tgt_l, gscale, gpout.data)
        loss += l_pol_k; l_pol += l_pol_k
        for b in range(B):
            val_tgt_l[b] = value_tgt[k * B + b]
        mz_two_hot_target_batch[B, BINS](val_tgt_l, 0, v_min, v_max, twv.data, 0)
        var l_val_k = value_coef * soft_ce_slice_loss_and_grad[
            B, PRED_OUT, ACT, BINS](
            pout.data, twv.data, gscale * value_coef, gpout.data)
        loss += l_val_k; l_val += l_val_k
        call_vjp["cpu", B](
            pred, TensorRefs[PRED.ARITY](zk_work), gpout,
            TensorRefs[PRED.ARITY](gpin), None)

        # (b) consistency online branch (k >= 1)
        if k >= 1:
            call_forward["cpu", B](proj, TensorRefs[PROJM.ARITY](zk_work), projo, None)
            call_forward["cpu", B](predh, TensorRefs[PREDH.ARITY](projo), pk, None)
            for i in range(B * PROJ):
                cons_t_l[i] = tstore.data[(k - 1) * B * PROJ + i]
            var mk = Optional[Pointer[Scalar[DT], MutAnyOrigin]](None)
            if cons_mask:
                mk = cons_mask.value().unsafe_offset((k - 1) * B)
            var l_cons_k = consistency_loss_and_grad[B, PROJ](
                pk.data, cons_t_l, cscale, gpk.data, mask=mk)
            loss += l_cons_k; l_cons += l_cons_k
            call_vjp["cpu", B](
                predh, TensorRefs[PREDH.ARITY](projo), gpk,
                TensorRefs[PREDH.ARITY](gproj), None)
            call_vjp["cpu", B](
                proj, TensorRefs[PROJM.ARITY](zk_work), gproj,
                TensorRefs[PROJM.ARITY](gzcons), None)
            for i in range(B * LATENT):
                gpin.data[i] += gzcons.data[i]

        # (c) reward (value-prefix) + dynamics for step k (produces z_{k+1})
        if k < K:
            var znoff = (k + 1) * B * LATENT
            for i in range(B * LATENT):
                dz.data[i] = zst.data[znoff + i]
            # the reset broke the gradient path into S_{k+1} from step k+1
            if (k + 1) % HORIZON == 0:
                for i in range(HH):
                    dh_carry.data[i] = Scalar[DT](0.0)
                    dc_carry.data[i] = Scalar[DT](0.0)
            # re-forward reward head at step k (repopulate stem/cell/head caches)
            for i in range(HH):
                hbuf.data[i] = h_store.data[k * HH + i]
                cbuf.data[i] = c_store.data[k * HH + i]
            rew.reward_step_forward["cpu", B](dz, hbuf, cbuf, cache, vp, None)
            # value-prefix loss on vp_k
            for b in range(B):
                rew_tgt_l[b] = reward_tgt[k * B + b]
            mz_two_hot_target_batch[B, BINS](rew_tgt_l, 0, v_min, v_max, twr.data, 0)
            var l_rew_k = soft_ce_slice_loss_and_grad[B, BINS, 0, BINS](
                vp.data, twr.data, gscale, grad_vp.data)
            loss += l_rew_k; l_rew += l_rew_k
            # reward BPTT → grad wrt z_{k+1} (grad_zp), grad wrt S_k/C_k (carry)
            rew.reward_step_backward["cpu", B](
                dz, grad_vp, hbuf, cbuf, cache, dh_carry, dc_carry,
                grad_zp, dh_prev, dc_prev, None)
            for i in range(B * LATENT):
                gz.data[i] += grad_zp.data[i]
            for i in range(HH):
                dh_carry.data[i] = dh_prev.data[i]
                dc_carry.data[i] = dc_prev.data[i]
            # dynamics: carry full grad wrt z_{k+1} back through dyn_z, ½ on z_k
            for b in range(B):
                var dib = b * DYN_IN
                for i in range(LATENT):
                    din.data[dib + i] = zst.data[zoff + b * LATENT + i]
                for a in range(ACT):
                    din.data[dib + LATENT + a] = Scalar[DT](0.0)
                din.data[dib + LATENT + Int(actions[k * B + b])] = Scalar[DT](1.0)
            call_forward["cpu", B](dynz, TensorRefs[DYNZ.ARITY](din), dz, None)
            call_vjp["cpu", B](
                dynz, TensorRefs[DYNZ.ARITY](din), gz,
                TensorRefs[DYNZ.ARITY](gdin), None)
            for b in range(B):
                for i in range(LATENT):
                    gpin.data[b * LATENT + i] += (
                        Scalar[DT](0.5) * gdin.data[b * DYN_IN + i])

        for i in range(B * LATENT):
            gz.data[i] = gpin.data[i]

    # ── rep: re-forward obs0 then vjp ──
    call_forward["cpu", B](rep, TensorRefs[REP.ARITY](obs0_t), z_work, None)
    call_vjp["cpu", B](
        rep, TensorRefs[REP.ARITY](obs0_t), gz, TensorRefs[REP.ARITY](gobs), None)

    # ── clip + step all six nets ──
    _ = clip_grad_norm["cpu", PRED](pred, Scalar[DT](max_grad_norm), None)
    opred.begin_step(); walk_params["cpu"](pred, opred, None)
    _ = clip_grad_norm["cpu", EZRewardLSTMAtari[BINS]](
        rew, Scalar[DT](max_grad_norm), None)
    orew.begin_step(); walk_params["cpu"](rew, orew, None)
    _ = clip_grad_norm["cpu", DYNZ](dynz, Scalar[DT](max_grad_norm), None)
    odynz.begin_step(); walk_params["cpu"](dynz, odynz, None)
    _ = clip_grad_norm["cpu", REP](rep, Scalar[DT](max_grad_norm), None)
    orep.begin_step(); walk_params["cpu"](rep, orep, None)
    _ = clip_grad_norm["cpu", PROJM](proj, Scalar[DT](max_grad_norm), None)
    oproj.begin_step(); walk_params["cpu"](proj, oproj, None)
    _ = clip_grad_norm["cpu", PREDH](predh, Scalar[DT](max_grad_norm), None)
    opredh.begin_step(); walk_params["cpu"](predh, opredh, None)

    if loss_parts:
        var lp = loss_parts.value()
        var inv = Scalar[DT](1.0) / Scalar[DT](B)
        lp[unsafe_offset=0] = l_pol * inv
        lp[unsafe_offset=1] = l_val * inv
        lp[unsafe_offset=2] = l_rew * inv
        lp[unsafe_offset=3] = l_cons * inv
    return loss / Scalar[DT](B)


# ─────────────────────────────────────────────────────────────────────────
# GPU path
# ─────────────────────────────────────────────────────────────────────────


def _ez_accum_latent_k[N_: Int, DDT: DType = DT, SDT: DType = DT](
    dst: Tile[DDT, N_],
    src: Tile[SDT, N_],
):
    """`dst[i] += src[i]` — fold the consistency latent-grad into ``gpin``.
    Dual-dtype (fp32 accumulator dst, possibly-bf16 src) so the bf16 BPTT keeps
    the grad CARRY in fp32 (`DDT=DT`) while folding a bf16 net-grad src."""
    var i = Int(global_idx.x)
    if i < N_:
        dst[i] = (
            rebind[Scalar[DDT]](dst[i]).cast[DT]()
            + rebind[Scalar[SDT]](src[i]).cast[DT]()
        ).cast[DDT]()


def ezv2_unroll_train_step_gpu[
    REP: Module,
    DYN: Module,
    PRED: Module,
    PROJM: Module,
    PREDH: Module,
    B: Int,
    K: Int,
    OBS: Int,
    ACT: Int,
    LATENT: Int,
    BINS: Int,
](
    ctx: DeviceContext,
    mut scratch: EZV2UnrollScratch[B, K, OBS, ACT, LATENT, BINS, PROJM.OUT_DIM],
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
    actions: List[Scalar[DT]],
    policy_tgt: List[Scalar[DT]],
    value_tgt: List[Scalar[DT]],
    reward_tgt: List[Scalar[DT]],
    v_min: Scalar[DT],
    v_max: Scalar[DT],
    value_coef: Scalar[DT] = Scalar[DT](1.0),
    consistency_coef: Scalar[DT] = Scalar[DT](2.0),
    max_grad_norm: Float64 = 0.0,
    cons_mask: Optional[Pointer[Scalar[DT], MutAnyOrigin]] = None,
    loss_parts: Optional[Pointer[Scalar[DT], MutAnyOrigin]] = None,
    is_weights: Optional[Pointer[Scalar[DT], MutAnyOrigin]] = None,
    out_prio: Optional[Pointer[Scalar[DT], MutAnyOrigin]] = None,
    obs_on_device: Bool = False,
    phase_ns: Optional[Pointer[Float64, MutAnyOrigin]] = None,
    diag_sync: Bool = False,
) raises -> Scalar[DT]:
    """GPU EZv2 K-step unroll training step — device mirror of
    ``ezv2_unroll_train_step_cpu`` (MuZero BPTT + SimSiam consistency,
    ``cons_mask`` = the optional host ``[K, B]`` episode-boundary mask —
    see the CPU docstring).

    Same **host** time-major batch slabs as the CPU path, with ``obs_seq`` the
    full ``[K+1, B, OBS]`` observation sequence (``obs_seq[0]`` is the root obs;
    the rest feed the consistency targets). H2D-copies them once, runs the
    forward scan (rep + K dynamics), the detached target pre-pass
    (``t_k = g_proj(h(obs_k))``), and the reverse scan (pred/dyn/consistency vjp
    with the ½ dynamics gradient + 1/(K+1) loss weight) entirely on device, then
    steps all five Adam optimizers. Returns the mean total loss (same reduction
    as the CPU path). Device + host scratch is supplied by the caller via a
    persistent ``EZV2UnrollScratch`` (allocated **once** in
    ``EZV2UnrollScratch.make`` and reused every step — the old per-step
    ``enqueue_create_buffer`` exploded disk on NVIDIA)."""
    comptime PRED_OUT = ACT + BINS
    comptime DYN_IN = LATENT + ACT
    comptime DYN_OUT = LATENT + BINS
    comptime PROJ = PROJM.OUT_DIM

    var octx = Optional[DeviceContext](ctx)
    var gscale = Scalar[DT](1.0) / Scalar[DT]((K + 1) * B)
    var cscale = consistency_coef / Scalar[DT](K * B)
    _ = phase_ns; _ = diag_sync   # host-enqueue profiling dropped in storage port

    # ── H2D the host batch slabs (sanctioned list.unsafe_ptr() staging) ──
    # obs_on_device: caller already gathered the [K+1,B,OBS] slab into d_obs.
    if not obs_on_device:
        ctx.enqueue_copy(scratch.d_obs.dev.value(), obs_seq.unsafe_ptr())
    ctx.enqueue_copy(scratch.d_act.dev.value(), actions.unsafe_ptr())
    ctx.enqueue_copy(scratch.d_pol.dev.value(), policy_tgt.unsafe_ptr())
    ctx.enqueue_copy(scratch.d_val.dev.value(), value_tgt.unsafe_ptr())
    ctx.enqueue_copy(scratch.d_rew.dev.value(), reward_tgt.unsafe_ptr())
    # consistency boundary mask (all-ones fallback when caller passes none).
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
    ref d_isw = scratch.d_isw
    ref d_prio = scratch.d_prio

    # PER: H2D the IS weights once (gated by `has_isw` → bit-identical to the
    # unweighted path when `is_weights` is None).
    var has_isw = Bool(is_weights)
    if has_isw:
        ctx.enqueue_copy(d_isw.dev.value(), is_weights.value())

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
    comptime LBACT = RowMajor[B * ACT]
    comptime LBPROJ = RowMajor[B * PROJ]

    comptime nbDIN = (B * DYN_IN + TPB - 1) // TPB
    comptime nbLAT = (B * LATENT + TPB - 1) // TPB
    comptime nbOBS = (B * OBS + TPB - 1) // TPB
    comptime nbPROJ = (B * PROJ + TPB - 1) // TPB
    comptime nbB = (B + TPB - 1) // TPB
    comptime kBuild = _mz_build_dyn_in_k[B, LATENT, ACT, DYN_IN]
    comptime kCopyL = _mz_copy_latent_k[B, LATENT, DYN_OUT]
    comptime kPolCE = _mz_softce_slice_k[B, PRED_OUT, 0, ACT]
    comptime kValCE = _mz_softce_slice_k[B, PRED_OUT, ACT, BINS]
    comptime kRewCE = _mz_softce_slice_k[B, DYN_OUT, LATENT, BINS]
    comptime kTwoHot = _mz_twohot_k[B, BINS]
    comptime kCarry = _mz_set_carry_latent_k[B, LATENT, DYN_OUT]
    comptime kHalf = _mz_accum_half_k[B, LATENT, DYN_IN]
    comptime kBcopy = _mz_bcopy_k[B * LATENT]
    comptime kBcopyOBS = _mz_bcopy_k[B * OBS]
    comptime kBcopyPROJ = _mz_bcopy_k[B * PROJ]
    comptime kCons = consistency_loss_grad_k[B, PROJ]
    comptime kAccum = _ez_accum_latent_k[B * LATENT]
    # PER row-scaling: pred head (whole row), reward slice only (latent slice is
    # the already-weighted carry), consistency (whole row); + value-error prio.
    comptime kScalePred = _ez_scale_rows_k[B, PRED_OUT, 0, PRED_OUT]
    comptime kScaleRew = _ez_scale_rows_k[B, DYN_OUT, LATENT, BINS]
    comptime kScaleCons = _ez_scale_rows_k[B, PROJ, 0, PROJ]
    comptime kPrioCE = _ez_priority_ce_k[B, PRED_OUT, ACT, BINS]

    # ── forward scan: z0 = h(obs0); z_{k+1} = g(z_k, a_k).latent ──
    # obs0 = d_obs[0]; bridge sub-slab → obs work tile → rep.forward → z_work.
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
            d_act.tt_at["gpu", B](k * B),
            grid_dim=nbDIN, block_dim=TPB,
        )
        call_forward["gpu", B](dyn, TensorRefs[DYN.ARITY](din), dout, octx)
        ctx.enqueue_function[kCopyL](
            zst.tt_at["gpu", B * LATENT]((k + 1) * B * LATENT),
            dout.tt["gpu", B * DYN_OUT](),
            grid_dim=nbLAT, block_dim=TPB,
        )

    # ── target pre-pass: t_k = g_proj(h(obs_k)), detached, k = 1..K ──
    # (clobbers rep/proj caches → rep re-forwarded before the final rep.vjp)
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
        # load z_k into the reverse-scan forward-input work tile
        ctx.enqueue_function[kBcopy](
            zst.tt_at["gpu", B * LATENT](k * B * LATENT),
            zk_work.tt["gpu", B * LATENT](),
            grid_dim=nbLAT, block_dim=TPB,
        )

        # (a) prediction head: forward (cache), seed grads, vjp → grad z_k
        call_forward["gpu", B](pred, TensorRefs[PRED.ARITY](zk_work), pout, octx)
        ctx.enqueue_function[kPolCE](
            pout.tt["gpu", B * PRED_OUT](),
            d_pol.tt_at["gpu", B * ACT](k * B * ACT),
            gpout.tt["gpu", B * PRED_OUT](),
            loss_d.tt_at["gpu", B](0),
            gscale, Scalar[DT](1.0),
            grid_dim=nbB, block_dim=TPB,
        )
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
        # PER: value-error priority at the root (k=0), read from value logits.
        if out_prio and k == 0:
            ctx.enqueue_function[kPrioCE](
                pout.tt["gpu", B * PRED_OUT](), twv.tt["gpu", B * BINS](),
                d_prio.tt["gpu", B](), grid_dim=nbB, block_dim=TPB,
            )
        if has_isw:
            ctx.enqueue_function[kScalePred](
                gpout.tt["gpu", B * PRED_OUT](), d_isw.tt["gpu", B](),
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
            if has_isw:
                ctx.enqueue_function[kScaleCons](
                    gpk.tt["gpu", B * PROJ](), d_isw.tt["gpu", B](),
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
                d_act.tt_at["gpu", B](k * B),
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
            # PER: weight ONLY the reward slice of the dyn grad by w_b (the latent
            # slice is the already-weighted carry from z_{k+1}).
            if has_isw:
                ctx.enqueue_function[kScaleRew](
                    gdout.tt["gpu", B * DYN_OUT](), d_isw.tt["gpu", B](),
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

    # ── D2H PER priorities + loss with a SINGLE sync ──
    if out_prio:
        ctx.enqueue_copy(scratch.h_prio.value(), d_prio.dev.value())
    ctx.enqueue_copy(h_loss, loss_d.dev.value())
    ctx.synchronize()
    if out_prio:
        var op = out_prio.value()
        var hpp = scratch.h_prio.value().unsafe_ptr()
        for b in range(B):
            op[unsafe_offset=b] = hpp[unsafe_offset=b]

    # ── reduce loss: 4 contiguous [B] blocks policy|value|reward|consistency ──
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


# ─────────────────────────────────────────────────────────────────────────
# Value-prefix GPU path (EZ `value_prefix=True`). Device mirror of
# `ezv2_unroll_train_step_cpu_vp` — same data flow (z'-only dynamics + stateful
# LSTM reward head, (h,c) carried across K steps and reset every HORIZON). Uses
# the reward head's GPU step API (`reward_step_forward/backward`, both targets
# implemented). For clarity/self-containment this allocates its own device
# buffers per call (the non-VP path reuses an EZV2UnrollScratch); folding these
# into a persistent VP scratch is a perf follow-up (bench on NVIDIA).
# ─────────────────────────────────────────────────────────────────────────
def ezv2_unroll_train_step_gpu_vp[
    REP: Module,
    DYNZ: Module,
    PRED: Module,
    PROJM: Module,
    PREDH: Module,
    B: Int,
    K: Int,
    OBS: Int,
    ACT: Int,
    LATENT: Int,
    BINS: Int,
    HIDDEN: Int,
    HORIZON: Int,
](
    ctx: DeviceContext,
    mut rep: REP,
    mut dynz: DYNZ,
    mut rew: EZRewardLSTMAtari[BINS],
    mut pred: PRED,
    mut proj: PROJM,
    mut predh: PREDH,
    mut orep: Adam,
    mut odynz: Adam,
    mut orew: Adam,
    mut opred: Adam,
    mut oproj: Adam,
    mut opredh: Adam,
    obs_seq: List[Scalar[DT]],
    actions: List[Scalar[DT]],
    policy_tgt: List[Scalar[DT]],
    value_tgt: List[Scalar[DT]],
    reward_tgt: List[Scalar[DT]],   # cumulative value-prefix targets [K,B]
    v_min: Scalar[DT],
    v_max: Scalar[DT],
    value_coef: Scalar[DT] = Scalar[DT](1.0),
    consistency_coef: Scalar[DT] = Scalar[DT](2.0),
    max_grad_norm: Float64 = 0.0,
    cons_mask: Optional[Pointer[Scalar[DT], MutAnyOrigin]] = None,
    loss_parts: Optional[Pointer[Scalar[DT], MutAnyOrigin]] = None,
    is_weights: Optional[Pointer[Scalar[DT], MutAnyOrigin]] = None,
    out_prio: Optional[Pointer[Scalar[DT], MutAnyOrigin]] = None,
    obs_dev: Optional[Pointer[Scalar[DT], MutAnyOrigin]] = None,
) raises -> Scalar[DT]:
    """GPU EZv2 **value-prefix** unroll training step — device mirror of
    `ezv2_unroll_train_step_cpu_vp`. Returns mean total loss; mutates all six
    nets via their optimizers.

    PER (batched driver): `is_weights[B]` row-scales the pred / consistency /
    value-prefix grads before vjp (the latent carry is already weighted);
    `out_prio[B]` receives the root (k=0) value soft-CE for the priority
    writeback. `obs_dev`, when set, is a device obs slab `[(K+1)·B·OBS]` (gathered
    by the replay) used in place of the host `obs_seq` H2D (no copy)."""
    comptime PRED_OUT = ACT + BINS
    comptime DYN_IN = LATENT + ACT
    comptime PROJ = PROJM.OUT_DIM
    comptime HH = B * HIDDEN
    # Activation-flow dtype (bf16 under AMP). rep/dynz/pred/proj/predh must agree;
    # the reward head (EZRewardLSTMAtari) stays fp32 — bridged by dz↔dz_f32 casts.
    comptime ADT = REP.ACT_DT
    comptime assert (
        REP.ACT_DT == DYNZ.ACT_DT and REP.ACT_DT == PRED.ACT_DT
        and REP.ACT_DT == PROJM.ACT_DT and REP.ACT_DT == PREDH.ACT_DT
    ), "EZv2 VP unroll: rep/dynz/pred/proj/predh must share ACT_DT"
    var octx = Optional[DeviceContext](ctx)

    # ── device buffers (owned, lazy) ──
    # obs slab: owned; filled by a D2D copy from the replay's device gather
    # (obs_dev / obs_on_device) or an H2D from the host obs_seq. (A non-owning
    # view of obs_dev would skip the D2D, but that conditional-Tensor pattern
    # ICEs the current toolchain; the D2D is device-bandwidth-only.)
    var d_obs = Tensor.alloc_gpu(ctx, (K + 1) * B * OBS)
    if obs_dev:
        d_obs.copy_from_device(ctx, obs_dev.value(), (K + 1) * B * OBS)
    else:
        ctx.enqueue_copy(d_obs.dev.value(), obs_seq.unsafe_ptr())
    var d_act = Tensor.alloc_gpu(ctx, K * B)
    ctx.enqueue_copy(d_act.dev.value(), actions.unsafe_ptr())
    var d_pol = Tensor.alloc_gpu(ctx, (K + 1) * B * ACT)
    ctx.enqueue_copy(d_pol.dev.value(), policy_tgt.unsafe_ptr())
    var d_val = Tensor.alloc_gpu(ctx, (K + 1) * B)
    ctx.enqueue_copy(d_val.dev.value(), value_tgt.unsafe_ptr())
    var d_rew = Tensor.alloc_gpu(ctx, K * B)
    ctx.enqueue_copy(d_rew.dev.value(), reward_tgt.unsafe_ptr())
    var d_cmask = Tensor.alloc_gpu(ctx, K * B)
    if cons_mask:
        ctx.enqueue_copy(d_cmask.dev.value(), cons_mask.value())
    else:
        d_cmask.dev.value().enqueue_fill(Scalar[DT](1.0))

    # Activation/grad buffers flow at ADT (bf16 under AMP). At ADT=DT,
    # TensorImpl[ADT] ≡ Tensor → byte-identical to the fp32 path.
    var d_obs_work = TensorImpl[ADT].alloc_gpu(ctx, B * OBS)
    var zst = TensorImpl[ADT].alloc_gpu(ctx, (K + 1) * B * LATENT)
    var z_work = TensorImpl[ADT].alloc_gpu(ctx, B * LATENT)
    var zk_work = TensorImpl[ADT].alloc_gpu(ctx, B * LATENT)
    var din = TensorImpl[ADT].alloc_gpu(ctx, B * DYN_IN)
    var dz = TensorImpl[ADT].alloc_gpu(ctx, B * LATENT)
    var dz_f32 = Tensor.alloc_gpu(ctx, B * LATENT)   # fp32 reward-LSTM input
    var pout = TensorImpl[ADT].alloc_gpu(ctx, B * PRED_OUT)
    var gpout = TensorImpl[ADT].alloc_gpu(ctx, B * PRED_OUT)
    # BPTT grad CARRY in fp32 (bf16 accumulation across K steps loses too much
    # precision → weak/unstable bf16 convergence). The per-net vjp stays bf16:
    # `gz_b`/`gpin_b` bridge the carry to/from the bf16 nets.
    var gz = Tensor.alloc_gpu(ctx, B * LATENT)
    var gpin = Tensor.alloc_gpu(ctx, B * LATENT)
    var gz_b = TensorImpl[ADT].alloc_gpu(ctx, B * LATENT)    # bf16 vjp grad_out
    var gpin_b = TensorImpl[ADT].alloc_gpu(ctx, B * LATENT)  # bf16 pred grad_in
    var gdin = TensorImpl[ADT].alloc_gpu(ctx, B * DYN_IN)
    var gobs = TensorImpl[ADT].alloc_gpu(ctx, B * OBS)
    var twv = Tensor.alloc_gpu(ctx, B * BINS)
    var twr = Tensor.alloc_gpu(ctx, B * BINS)
    var vp = Tensor.alloc_gpu(ctx, B * BINS)
    var grad_vp = Tensor.alloc_gpu(ctx, B * BINS)
    var grad_zp = Tensor.alloc_gpu(ctx, B * LATENT)   # fp32 reward-LSTM dz-grad
    var h_store = Tensor.alloc_gpu(ctx, (K + 1) * HH)
    var c_store = Tensor.alloc_gpu(ctx, (K + 1) * HH)
    var hbuf = Tensor.alloc_gpu(ctx, 2 * HH)
    var cbuf = Tensor.alloc_gpu(ctx, 2 * HH)
    var cache = Tensor.alloc_gpu(ctx, B * EZRewardLSTMAtari[BINS].CACHE_SIZE)
    var dh_carry = Tensor.alloc_gpu(ctx, HH)
    var dc_carry = Tensor.alloc_gpu(ctx, HH)
    var dh_prev = Tensor.alloc_gpu(ctx, HH)
    var dc_prev = Tensor.alloc_gpu(ctx, HH)
    var tstore = TensorImpl[ADT].alloc_gpu(ctx, K * B * PROJ)
    var ztmp = TensorImpl[ADT].alloc_gpu(ctx, B * LATENT)
    var projo = TensorImpl[ADT].alloc_gpu(ctx, B * PROJ)
    var pk = TensorImpl[ADT].alloc_gpu(ctx, B * PROJ)
    var gpk = TensorImpl[ADT].alloc_gpu(ctx, B * PROJ)
    var gproj = TensorImpl[ADT].alloc_gpu(ctx, B * PROJ)
    var gzcons = TensorImpl[ADT].alloc_gpu(ctx, B * LATENT)
    var loss_d = Tensor.alloc_gpu(ctx, 4 * B)
    var h_loss = ctx.enqueue_create_host_buffer[DT](4 * B)
    for i in range(4 * B):
        h_loss.unsafe_ptr()[unsafe_offset=i] = Scalar[DT](0.0)
    ctx.enqueue_copy(loss_d.dev.value(), h_loss)
    # PER scratch (gated): IS weights H2D'd once; priority CE + D2H mirror.
    var has_isw = Bool(is_weights)
    var d_isw = Tensor.alloc_gpu(ctx, B)
    if has_isw:
        ctx.enqueue_copy(d_isw.dev.value(), is_weights.value())
    var d_prio = Tensor.alloc_gpu(ctx, B)
    var h_prio = ctx.enqueue_create_host_buffer[DT](B)

    var gscale = Scalar[DT](1.0) / Scalar[DT]((K + 1) * B)
    var cscale = consistency_coef / Scalar[DT](K * B)

    comptime LB = RowMajor[B]
    comptime LBOBS = RowMajor[B * OBS]
    comptime LBL = RowMajor[B * LATENT]
    comptime LBDI = RowMajor[B * DYN_IN]
    comptime LBPO = RowMajor[B * PRED_OUT]
    comptime LBBINS = RowMajor[B * BINS]
    comptime LBPROJ = RowMajor[B * PROJ]
    comptime nbDIN = (B * DYN_IN + TPB - 1) // TPB
    comptime nbLAT = (B * LATENT + TPB - 1) // TPB
    comptime nbOBS = (B * OBS + TPB - 1) // TPB
    comptime nbPROJ = (B * PROJ + TPB - 1) // TPB
    comptime nbB = (B + TPB - 1) // TPB
    # Activation kernels carry ADT; kRewVP/kTwoHot/kBcopyOBS stay DT (they touch
    # the fp32 reward-LSTM output / two-hot targets / fp32 obs source).
    comptime kBuild = _mz_build_dyn_in_k[B, LATENT, ACT, DYN_IN, ADT]
    comptime kPolCE = _mz_softce_slice_k[B, PRED_OUT, 0, ACT, ADT]
    comptime kValCE = _mz_softce_slice_k[B, PRED_OUT, ACT, BINS, ADT]
    comptime kRewVP = _mz_softce_slice_k[B, BINS, 0, BINS]
    comptime kTwoHot = _mz_twohot_k[B, BINS]
    comptime kHalf = _mz_accum_half_k[B, LATENT, DYN_IN, DT, ADT]  # fp32 gpin
    comptime kBcopy = _mz_bcopy_k[B * LATENT, ADT]
    comptime kBcopyLf = _mz_bcopy_k[B * LATENT]   # fp32 carry copy (gpin→gz)
    comptime kBcopyOBS = _mz_bcopy_k[B * OBS]
    comptime kBcopyPROJ = _mz_bcopy_k[B * PROJ, ADT]
    comptime kCons = consistency_loss_grad_k[B, PROJ, ADT]
    comptime kAccumGc = _ez_accum_latent_k[B * LATENT, DT, ADT]  # gpin+=gzcons
    comptime kAccumGr = _ez_accum_latent_k[B * LATENT]           # gz+=grad_zp fp32
    # PER row-scaling: pred head (whole), consistency (whole), value-prefix
    # (whole BINS); + root value-error priority CE.
    comptime kScalePred = _ez_scale_rows_k[B, PRED_OUT, 0, PRED_OUT, ADT]
    comptime kScaleCons = _ez_scale_rows_k[B, PROJ, 0, PROJ, ADT]
    comptime kScaleVP = _ez_scale_rows_k[B, BINS, 0, BINS]
    comptime kPrioCE = _ez_priority_ce_k[B, PRED_OUT, ACT, BINS, ADT]

    # ── forward scan: z0=h(obs0); z_{k+1}=g_z(z_k,a_k); (h,c) LSTM step ──
    # obs→d_obs_work: plain copy at fp32; fp32→bf16 cast under AMP.
    comptime if ADT == DT:
        ctx.enqueue_function[kBcopyOBS](
            d_obs.tt_at["gpu", B * OBS](0), d_obs_work.tt["gpu", B * OBS](),
            grid_dim=nbOBS, block_dim=TPB)
    else:
        ctx.enqueue_function[_cast_f2b_kernel](
            d_obs.tt_at["gpu", B * OBS](0).ptr,
            d_obs_work.tt["gpu", B * OBS]().ptr,
            Int64(B * OBS),
            grid_dim=nbOBS,
            block_dim=TPB,
        )
    call_forward["gpu", B](rep, child_refs[N=REP.ARITY, CADT=ADT](d_obs_work), z_work, octx)
    ctx.enqueue_function[kBcopy](
        z_work.tt["gpu", B * LATENT](), zst.tt_at["gpu", B * LATENT](0),
        grid_dim=nbLAT, block_dim=TPB)
    # h_store[0]/c_store[0] = 0 (alloc_gpu zero-fills)
    for k in range(K):
        ctx.enqueue_function[kBuild](
            din.tt["gpu", B * DYN_IN](), zst.tt_at["gpu", B * LATENT](k * B * LATENT),
            d_act.tt_at["gpu", B](k * B), grid_dim=nbDIN, block_dim=TPB)
        call_forward["gpu", B](dynz, child_refs[N=DYNZ.ARITY, CADT=ADT](din), dz, octx)
        ctx.enqueue_function[kBcopy](
            dz.tt["gpu", B * LATENT](), zst.tt_at["gpu", B * LATENT]((k + 1) * B * LATENT),
            grid_dim=nbLAT, block_dim=TPB)
        # (h,c) prev slab ← h_store[k]; step; h_store[k+1] ← out slab
        ctx.enqueue_copy(
            hbuf.dev.value().create_sub_buffer[DT](0, HH),
            h_store.dev.value().create_sub_buffer[DT](k * HH, HH))
        ctx.enqueue_copy(
            cbuf.dev.value().create_sub_buffer[DT](0, HH),
            c_store.dev.value().create_sub_buffer[DT](k * HH, HH))
        comptime if ADT == DT:
            ref dzf = rebind[Tensor](dz)
            rew.reward_step_forward["gpu", B](dzf, hbuf, cbuf, cache, vp, octx)
        else:
            ctx.enqueue_function[_cast_b2f_kernel](
                dz.tt["gpu", B * LATENT]().ptr,
                dz_f32.tt["gpu", B * LATENT]().ptr,
                Int64(B * LATENT),
                grid_dim=nbLAT,
                block_dim=TPB,
            )
            rew.reward_step_forward["gpu", B](dz_f32, hbuf, cbuf, cache, vp, octx)
        ctx.enqueue_copy(
            h_store.dev.value().create_sub_buffer[DT]((k + 1) * HH, HH),
            hbuf.dev.value().create_sub_buffer[DT](HH, HH))
        ctx.enqueue_copy(
            c_store.dev.value().create_sub_buffer[DT]((k + 1) * HH, HH),
            cbuf.dev.value().create_sub_buffer[DT](HH, HH))
        if (k + 1) % HORIZON == 0:
            h_store.dev.value().create_sub_buffer[DT]((k + 1) * HH, HH).enqueue_fill(
                Scalar[DT](0.0))
            c_store.dev.value().create_sub_buffer[DT]((k + 1) * HH, HH).enqueue_fill(
                Scalar[DT](0.0))

    # ── consistency target pre-pass: t_k = g_proj(h(obs_k)), k=1..K ──
    for k in range(1, K + 1):
        comptime if ADT == DT:
            ctx.enqueue_function[kBcopyOBS](
                d_obs.tt_at["gpu", B * OBS](k * B * OBS),
                d_obs_work.tt["gpu", B * OBS](),
                grid_dim=nbOBS, block_dim=TPB)
        else:
            ctx.enqueue_function[_cast_f2b_kernel](
                d_obs.tt_at["gpu", B * OBS](k * B * OBS).ptr,
                d_obs_work.tt["gpu", B * OBS]().ptr,
                Int64(B * OBS),
                grid_dim=nbOBS,
                block_dim=TPB,
            )
        call_forward["gpu", B](rep, child_refs[N=REP.ARITY, CADT=ADT](d_obs_work), ztmp, octx)
        call_forward["gpu", B](proj, child_refs[N=PROJM.ARITY, CADT=ADT](ztmp), projo, octx)
        ctx.enqueue_function[kBcopyPROJ](
            projo.tt["gpu", B * PROJ](), tstore.tt_at["gpu", B * PROJ]((k - 1) * B * PROJ),
            grid_dim=nbPROJ, block_dim=TPB)

    # ── reverse scan ──
    rep.zero_grad["gpu"](octx)
    dynz.zero_grad["gpu"](octx)
    rew.zero_grad["gpu"](octx)
    pred.zero_grad["gpu"](octx)
    proj.zero_grad["gpu"](octx)
    predh.zero_grad["gpu"](octx)
    dh_carry.dev.value().enqueue_fill(Scalar[DT](0.0))
    dc_carry.dev.value().enqueue_fill(Scalar[DT](0.0))

    for rk in range(K + 1):
        var k = K - rk
        ctx.enqueue_function[kBcopy](
            zst.tt_at["gpu", B * LATENT](k * B * LATENT), zk_work.tt["gpu", B * LATENT](),
            grid_dim=nbLAT, block_dim=TPB)

        # (a) prediction head on z_k
        call_forward["gpu", B](pred, child_refs[N=PRED.ARITY, CADT=ADT](zk_work), pout, octx)
        ctx.enqueue_function[kPolCE](
            pout.tt["gpu", B * PRED_OUT](),
            d_pol.tt_at["gpu", B * ACT](k * B * ACT),
            gpout.tt["gpu", B * PRED_OUT](), loss_d.tt_at["gpu", B](0),
            gscale, Scalar[DT](1.0), grid_dim=nbB, block_dim=TPB)
        ctx.enqueue_function[kTwoHot](
            twv.tt["gpu", B * BINS](), d_val.tt_at["gpu", B](k * B),
            v_min, v_max, grid_dim=nbB, block_dim=TPB)
        ctx.enqueue_function[kValCE](
            pout.tt["gpu", B * PRED_OUT](), twv.tt["gpu", B * BINS](),
            gpout.tt["gpu", B * PRED_OUT](), loss_d.tt_at["gpu", B](B),
            gscale * value_coef, value_coef, grid_dim=nbB, block_dim=TPB)
        # PER: root (k=0) value-error priority, then IS-weight the pred grad.
        if out_prio and k == 0:
            ctx.enqueue_function[kPrioCE](
                pout.tt["gpu", B * PRED_OUT](), twv.tt["gpu", B * BINS](),
                d_prio.tt["gpu", B](), grid_dim=nbB, block_dim=TPB)
        if has_isw:
            ctx.enqueue_function[kScalePred](
                gpout.tt["gpu", B * PRED_OUT](), d_isw.tt["gpu", B](),
                grid_dim=nbB, block_dim=TPB)
        # pred grad_input SEEDS the fp32 carry. fp32 path: vjp → gpin directly.
        # bf16 path: vjp → gpin_b (bf16), cast up to the fp32 carry gpin.
        comptime if ADT == DT:
            call_vjp["gpu", B](
                pred, child_refs[N=PRED.ARITY, CADT=ADT](zk_work), gpout,
                child_refs[N=PRED.ARITY, CADT=ADT](gpin), octx)
        else:
            call_vjp["gpu", B](
                pred, child_refs[N=PRED.ARITY, CADT=ADT](zk_work), gpout,
                child_refs[N=PRED.ARITY, CADT=ADT](gpin_b), octx)
            ctx.enqueue_function[_cast_b2f_kernel](
                gpin_b.tt["gpu", B * LATENT]().ptr,
                gpin.tt["gpu", B * LATENT]().ptr,
                Int64(B * LATENT),
                grid_dim=nbLAT,
                block_dim=TPB,
            )

        # (b) consistency (k >= 1)
        if k >= 1:
            call_forward["gpu", B](proj, child_refs[N=PROJM.ARITY, CADT=ADT](zk_work), projo, octx)
            call_forward["gpu", B](predh, child_refs[N=PREDH.ARITY, CADT=ADT](projo), pk, octx)
            ctx.enqueue_function[kCons](
                pk.tt["gpu", B * PROJ](),
                tstore.tt_at["gpu", B * PROJ]((k - 1) * B * PROJ),
                gpk.tt["gpu", B * PROJ](), loss_d.tt_at["gpu", B](3 * B),
                d_cmask.tt_at["gpu", B]((k - 1) * B),
                cscale, Scalar[DT](1.0), grid_dim=nbB, block_dim=TPB)
            if has_isw:
                ctx.enqueue_function[kScaleCons](
                    gpk.tt["gpu", B * PROJ](), d_isw.tt["gpu", B](),
                    grid_dim=nbB, block_dim=TPB)
            call_vjp["gpu", B](
                predh, child_refs[N=PREDH.ARITY, CADT=ADT](projo), gpk,
                child_refs[N=PREDH.ARITY, CADT=ADT](gproj), octx)
            call_vjp["gpu", B](
                proj, child_refs[N=PROJM.ARITY, CADT=ADT](zk_work), gproj,
                child_refs[N=PROJM.ARITY, CADT=ADT](gzcons), octx)
            ctx.enqueue_function[kAccumGc](
                gpin.tt["gpu", B * LATENT](), gzcons.tt["gpu", B * LATENT](),
                grid_dim=nbLAT, block_dim=TPB)

        # (c) reward (value-prefix) + dynamics for step k (produces z_{k+1})
        if k < K:
            ctx.enqueue_function[kBcopy](
                zst.tt_at["gpu", B * LATENT]((k + 1) * B * LATENT), dz.tt["gpu", B * LATENT](),
                grid_dim=nbLAT, block_dim=TPB)
            if (k + 1) % HORIZON == 0:
                dh_carry.dev.value().enqueue_fill(Scalar[DT](0.0))
                dc_carry.dev.value().enqueue_fill(Scalar[DT](0.0))
            # re-forward reward head at step k (repopulate caches)
            ctx.enqueue_copy(
                hbuf.dev.value().create_sub_buffer[DT](0, HH),
                h_store.dev.value().create_sub_buffer[DT](k * HH, HH))
            ctx.enqueue_copy(
                cbuf.dev.value().create_sub_buffer[DT](0, HH),
                c_store.dev.value().create_sub_buffer[DT](k * HH, HH))
            # reward head is fp32: at fp32 rebind the latent (ADT==DT → same
            # type); under AMP cast bf16 z' → fp32 dz_f32 (reused by the backward
            # below; dz unchanged in between).
            comptime if ADT == DT:
                ref dzd = rebind[Tensor](dz)
                rew.reward_step_forward["gpu", B](
                    dzd, hbuf, cbuf, cache, vp, octx)
            else:
                ctx.enqueue_function[_cast_b2f_kernel](
                    dz.tt["gpu", B * LATENT]().ptr,
                    dz_f32.tt["gpu", B * LATENT]().ptr,
                    Int64(B * LATENT),
                    grid_dim=nbLAT,
                    block_dim=TPB,
                )
                rew.reward_step_forward["gpu", B](
                    dz_f32, hbuf, cbuf, cache, vp, octx)
            ctx.enqueue_function[kTwoHot](
                twr.tt["gpu", B * BINS](), d_rew.tt_at["gpu", B](k * B),
                v_min, v_max, grid_dim=nbB, block_dim=TPB)
            ctx.enqueue_function[kRewVP](
                vp.tt["gpu", B * BINS](), twr.tt["gpu", B * BINS](),
                grad_vp.tt["gpu", B * BINS](), loss_d.tt_at["gpu", B](2 * B),
                gscale, Scalar[DT](1.0), grid_dim=nbB, block_dim=TPB)
            # PER: IS-weight the value-prefix grad (the latent carry gz is the
            # already-weighted contribution from z_{k+1}).
            if has_isw:
                ctx.enqueue_function[kScaleVP](
                    grad_vp.tt["gpu", B * BINS](), d_isw.tt["gpu", B](),
                    grid_dim=nbB, block_dim=TPB)
            comptime if ADT == DT:
                ref dzd2 = rebind[Tensor](dz)
                rew.reward_step_backward["gpu", B](
                    dzd2, grad_vp, hbuf, cbuf, cache, dh_carry, dc_carry,
                    grad_zp, dh_prev, dc_prev, octx)
            else:
                rew.reward_step_backward["gpu", B](
                    dz_f32, grad_vp, hbuf, cbuf, cache, dh_carry, dc_carry,
                    grad_zp, dh_prev, dc_prev, octx)
            # gz (fp32) += grad_zp (fp32) — the carry stays fp32; the reward
            # gradient never round-trips through bf16.
            ctx.enqueue_function[kAccumGr](
                gz.tt["gpu", B * LATENT](), grad_zp.tt["gpu", B * LATENT](),
                grid_dim=nbLAT, block_dim=TPB)
            ctx.enqueue_copy(dh_carry.dev.value(), dh_prev.dev.value())
            ctx.enqueue_copy(dc_carry.dev.value(), dc_prev.dev.value())
            # dynamics: full grad wrt z_{k+1} back through dyn_z, ½ on z_k
            ctx.enqueue_function[kBuild](
                din.tt["gpu", B * DYN_IN](), zst.tt_at["gpu", B * LATENT](k * B * LATENT),
                d_act.tt_at["gpu", B](k * B), grid_dim=nbDIN, block_dim=TPB)
            call_forward["gpu", B](dynz, child_refs[N=DYNZ.ARITY, CADT=ADT](din), dz, octx)
            # dynz.vjp grad_output = the fp32 carry gz → cast to bf16 for the net.
            comptime if ADT == DT:
                ref gza = rebind[TensorImpl[ADT]](gz)
                call_vjp["gpu", B](
                    dynz, child_refs[N=DYNZ.ARITY, CADT=ADT](din), gza,
                    child_refs[N=DYNZ.ARITY, CADT=ADT](gdin), octx)
            else:
                ctx.enqueue_function[_cast_f2b_kernel](
                    gz.tt["gpu", B * LATENT]().ptr,
                    gz_b.tt["gpu", B * LATENT]().ptr,
                    Int64(B * LATENT),
                    grid_dim=nbLAT,
                    block_dim=TPB,
                )
                call_vjp["gpu", B](
                    dynz, child_refs[N=DYNZ.ARITY, CADT=ADT](din), gz_b,
                    child_refs[N=DYNZ.ARITY, CADT=ADT](gdin), octx)
            ctx.enqueue_function[kHalf](
                gpin.tt["gpu", B * LATENT](), gdin.tt["gpu", B * DYN_IN](),
                grid_dim=nbLAT, block_dim=TPB)

        # carry: gz (fp32) ← gpin (fp32). Both fp32 → fp32 bcopy.
        ctx.enqueue_function[kBcopyLf](
            gpin.tt["gpu", B * LATENT](), gz.tt["gpu", B * LATENT](),
            grid_dim=nbLAT, block_dim=TPB)

    # ── rep: re-forward obs0 then vjp ──
    comptime if ADT == DT:
        ctx.enqueue_function[kBcopyOBS](
            d_obs.tt_at["gpu", B * OBS](0), d_obs_work.tt["gpu", B * OBS](),
            grid_dim=nbOBS, block_dim=TPB)
    else:
        ctx.enqueue_function[_cast_f2b_kernel](
            d_obs.tt_at["gpu", B * OBS](0).ptr,
            d_obs_work.tt["gpu", B * OBS]().ptr,
            Int64(B * OBS),
            grid_dim=nbOBS,
            block_dim=TPB,
        )
    call_forward["gpu", B](rep, child_refs[N=REP.ARITY, CADT=ADT](d_obs_work), z_work, octx)
    # rep.vjp grad_output = the fp32 carry gz → cast to bf16 for the net.
    comptime if ADT == DT:
        ref gza = rebind[TensorImpl[ADT]](gz)
        call_vjp["gpu", B](
            rep, child_refs[N=REP.ARITY, CADT=ADT](d_obs_work), gza,
            child_refs[N=REP.ARITY, CADT=ADT](gobs), octx)
    else:
        ctx.enqueue_function[_cast_f2b_kernel](
            gz.tt["gpu", B * LATENT]().ptr,
            gz_b.tt["gpu", B * LATENT]().ptr,
            Int64(B * LATENT),
            grid_dim=nbLAT,
            block_dim=TPB,
        )
        call_vjp["gpu", B](
            rep, child_refs[N=REP.ARITY, CADT=ADT](d_obs_work), gz_b,
            child_refs[N=REP.ARITY, CADT=ADT](gobs), octx)

    # ── clip + step all six nets ──
    _ = clip_grad_norm["gpu", PRED](pred, Scalar[DT](max_grad_norm), octx)
    opred.begin_step(); walk_params["gpu"](pred, opred, octx)
    _ = clip_grad_norm["gpu", EZRewardLSTMAtari[BINS]](
        rew, Scalar[DT](max_grad_norm), octx)
    orew.begin_step(); walk_params["gpu"](rew, orew, octx)
    _ = clip_grad_norm["gpu", DYNZ](dynz, Scalar[DT](max_grad_norm), octx)
    odynz.begin_step(); walk_params["gpu"](dynz, odynz, octx)
    _ = clip_grad_norm["gpu", REP](rep, Scalar[DT](max_grad_norm), octx)
    orep.begin_step(); walk_params["gpu"](rep, orep, octx)
    _ = clip_grad_norm["gpu", PROJM](proj, Scalar[DT](max_grad_norm), octx)
    oproj.begin_step(); walk_params["gpu"](proj, oproj, octx)
    _ = clip_grad_norm["gpu", PREDH](predh, Scalar[DT](max_grad_norm), octx)
    opredh.begin_step(); walk_params["gpu"](predh, opredh, octx)

    # ── D2H loss (+ PER priorities), single sync ──
    if out_prio:
        ctx.enqueue_copy(h_prio, d_prio.dev.value())
    ctx.enqueue_copy(h_loss, loss_d.dev.value())
    ctx.synchronize()
    if out_prio:
        var op = out_prio.value()
        var hpp = h_prio.unsafe_ptr()
        for b in range(B):
            op[unsafe_offset=b] = hpp[unsafe_offset=b]
    var hp = h_loss.unsafe_ptr()
    var l_pol = Scalar[DT](0.0)
    var l_val = Scalar[DT](0.0)
    var l_rew = Scalar[DT](0.0)
    var l_cons = Scalar[DT](0.0)
    for b in range(B):
        l_pol += hp[unsafe_offset=b]; l_val += hp[unsafe_offset=B + b]
        l_rew += hp[unsafe_offset=2 * B + b]; l_cons += hp[unsafe_offset=3 * B + b]
    var inv = Scalar[DT](1.0) / Scalar[DT](B)
    if loss_parts:
        var lp = loss_parts.value()
        lp[unsafe_offset=0] = l_pol * inv; lp[unsafe_offset=1] = l_val * inv
        lp[unsafe_offset=2] = l_rew * inv; lp[unsafe_offset=3] = l_cons * inv
    return (l_pol + l_val + l_rew + l_cons) * inv
