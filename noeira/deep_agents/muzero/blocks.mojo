"""MuZero K-step unroll — the world-model BPTT training step.

One training step over a batch of length-``K`` trajectory windows, structured as
the DreamerV3 ``WMStep`` manual forward-scan / reverse-scan (no monolithic
ComputeGraph — far more tractable through learned dynamics):

  forward scan   z₀ = h(obs₀);  zₖ₊₁ = g(zₖ, aₖ).latent      (store z₀..z_K)
  per position   f(zₖ) → (policy, value);  g(zₖ,aₖ) → reward
  losses         soft-CE(policy, π) + soft-CE(value, twohot(h·v))
                 + soft-CE(reward, twohot(h·r))               (all categorical)
  reverse scan   re-forward each net (to refresh its vjp cache), seed each head's
                 grad slice analytically, run ``Module.vjp``, thread the carry
                 gradient ``∂L/∂zₖ = ∂L_pred/∂zₖ + ½·∂L_dyn/∂zₖ`` back to k−1.

Two MuZero-specific gradients are baked in:
  * **½ scale on the dynamics hidden input** — "scale the gradient at the start
    of the dynamics function by ½" (MuZero appendix). Applied to the latent half
    of ``g``'s input-gradient; compounds naturally across unroll steps.
  * **1/(K+1) per-step loss weight** (legacy parity) folded into ``grad_scale``
    alongside the 1/BATCH mean.

Batch layout is **time-major** so every per-step slice is contiguous (no gather):
``obs0[B,OBS]``, ``actions[K,B]`` (indices), ``policy_tgt[K+1,B,ACT]``,
``value_tgt[K+1,B]`` and ``reward_tgt[K,B]`` (raw scalars; ``h`` + two-hot applied
here). ``v_min/v_max`` are the h-space support shared with the planner + targets.

CPU path first (validated by an overfit test); a GPU branch + CPU↔GPU bit-parity
follows. The min-max latent scaling lives inside the nets (`MZRepNet`/`MZDynNet`
``MinMaxNorm`` tails), so it is already in the autodiff graph — no separate scale.
"""

from noeira.nn.core.param import walk_params
from std.math import exp, log, sqrt
from max.gpu import global_idx
from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer

from noeira.nn.constants import DT, TPB
from noeira.nn.core.module import Module
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.call import call_forward, call_vjp
from noeira.nn.optimizer.adam import Adam
from noeira.nn.optimizer.grad_clip import clip_grad_norm

from .loss_ops import soft_ce_slice_loss_and_grad
from ..zero.twohot_targets import mz_two_hot_target_batch
from noeira.core.tiles import RowMajor, Tile


struct MZScratch[
    B: Int,
    K: Int,
    OBS: Int,
    ACT: Int,
    LATENT: Int,
    BINS: Int,
](Movable & Deinitable):
    """Persistent device + host scratch for `mz_unroll_train_step_gpu`.

    Allocated **once** via `make` and reused every training step — per-step
    allocation in the hot loop explodes disk on NVIDIA and adds latency. Every
    buffer is fully overwritten each step before it is read, so reuse is safe.

    Storage-clean: every device buffer is an owned `Tensor` (RAII — no manual
    free); the device kernels take `.lt`/`.lt_at` views and the net forward/vjp
    take `TensorRefs`. `z_work`/`zk_work` bridge the forward-into-slab cases (the
    storage forward writes a whole `Tensor`, so the rep/pred outputs land in a
    working tile then copy to/from the `zst` latent-history slab)."""

    comptime PRED_OUT = Self.ACT + Self.BINS
    comptime DYN_IN = Self.LATENT + Self.ACT
    comptime DYN_OUT = Self.LATENT + Self.BINS

    var d_obs0: Tensor
    var d_act: Tensor
    var d_pol: Tensor
    var d_val: Tensor
    var d_rew: Tensor
    var zst: Tensor
    var z_work: Tensor   # rep forward output working tile (B*LATENT)
    var zk_work: Tensor  # reverse-scan zk forward input (B*LATENT)
    var din: Tensor
    var dout: Tensor
    var pout: Tensor
    var gpout: Tensor
    var gdout: Tensor
    var gz: Tensor
    var gpin: Tensor
    var gdin: Tensor
    var gobs: Tensor
    var twv: Tensor
    var twr: Tensor
    var loss_d: Tensor
    # PER scratch: per-sample IS weights (H2D) + value-error priorities (D2H).
    var d_isw: Tensor
    var d_prio: Tensor
    var h_zloss: Optional[HostBuffer[DT]]
    var h_loss: Optional[HostBuffer[DT]]
    var h_prio: Optional[HostBuffer[DT]]

    def __init__(out self):
        self.d_obs0 = Tensor(); self.d_act = Tensor(); self.d_pol = Tensor()
        self.d_val = Tensor(); self.d_rew = Tensor(); self.zst = Tensor()
        self.z_work = Tensor(); self.zk_work = Tensor()
        self.din = Tensor(); self.dout = Tensor(); self.pout = Tensor()
        self.gpout = Tensor(); self.gdout = Tensor(); self.gz = Tensor()
        self.gpin = Tensor(); self.gdin = Tensor(); self.gobs = Tensor()
        self.twv = Tensor(); self.twr = Tensor(); self.loss_d = Tensor()
        self.d_isw = Tensor(); self.d_prio = Tensor()
        self.h_zloss = None; self.h_loss = None; self.h_prio = None

    @staticmethod
    def make(ctx: DeviceContext) raises -> Self:
        comptime PO = Self.PRED_OUT
        comptime DI = Self.DYN_IN
        comptime DO = Self.DYN_OUT
        comptime BB = Self.B
        var s = Self()
        s.d_obs0 = Tensor.alloc_gpu(ctx, BB * Self.OBS)
        s.d_act = Tensor.alloc_gpu(ctx, Self.K * BB)
        s.d_pol = Tensor.alloc_gpu(ctx, (Self.K + 1) * BB * Self.ACT)
        s.d_val = Tensor.alloc_gpu(ctx, (Self.K + 1) * BB)
        s.d_rew = Tensor.alloc_gpu(ctx, Self.K * BB)
        s.zst = Tensor.alloc_gpu(ctx, (Self.K + 1) * BB * Self.LATENT)
        s.z_work = Tensor.alloc_gpu(ctx, BB * Self.LATENT)
        s.zk_work = Tensor.alloc_gpu(ctx, BB * Self.LATENT)
        s.din = Tensor.alloc_gpu(ctx, BB * DI)
        s.dout = Tensor.alloc_gpu(ctx, BB * DO)
        s.pout = Tensor.alloc_gpu(ctx, BB * PO)
        s.gpout = Tensor.alloc_gpu(ctx, BB * PO)
        s.gdout = Tensor.alloc_gpu(ctx, BB * DO)
        s.gz = Tensor.alloc_gpu(ctx, BB * Self.LATENT)
        s.gpin = Tensor.alloc_gpu(ctx, BB * Self.LATENT)
        s.gdin = Tensor.alloc_gpu(ctx, BB * DI)
        s.gobs = Tensor.alloc_gpu(ctx, BB * Self.OBS)
        s.twv = Tensor.alloc_gpu(ctx, BB * Self.BINS)
        s.twr = Tensor.alloc_gpu(ctx, BB * Self.BINS)
        # 3 contiguous [B] blocks: policy | value | reward.
        s.loss_d = Tensor.alloc_gpu(ctx, 3 * BB)
        s.d_isw = Tensor.alloc_gpu(ctx, BB)
        s.d_prio = Tensor.alloc_gpu(ctx, BB)
        s.h_zloss = ctx.enqueue_create_host_buffer[DT](3 * BB)
        s.h_loss = ctx.enqueue_create_host_buffer[DT](3 * BB)
        s.h_prio = ctx.enqueue_create_host_buffer[DT](BB)
        ctx.synchronize()
        return s^


def mz_unroll_train_step_cpu[
    REP: Module,
    DYN: Module,
    PRED: Module,
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
    mut orep: Adam,
    mut odyn: Adam,
    mut opred: Adam,
    obs0: List[Scalar[DT]],
    actions: List[Scalar[DT]],
    policy_tgt: List[Scalar[DT]],
    value_tgt: List[Scalar[DT]],
    reward_tgt: List[Scalar[DT]],
    v_min: Scalar[DT],
    v_max: Scalar[DT],
    value_coef: Scalar[DT] = Scalar[DT](1.0),
    max_grad_norm: Float64 = 0.0,
    loss_parts: Optional[Pointer[Scalar[DT], MutAnyOrigin]] = None,
) raises -> Scalar[DT]:
    """One CPU MuZero unroll training step. Returns the mean total loss
    (policy + value + reward, summed over the K+1 / K positions then averaged
    over batch and unroll length). Mutates all three nets via their optimizers.
    """
    comptime PRED_OUT = ACT + BINS
    comptime DYN_IN = LATENT + ACT
    comptime DYN_OUT = LATENT + BINS

    # ── scratch (owned storage Tensors; RAII — no manual free) ──
    var obs0_t = Tensor.alloc(B * OBS)    # rep input bridge (copy from obs0 ptr)
    for i in range(B * OBS):
        obs0_t.data[i] = obs0[i]
    var zst = Tensor.alloc((K + 1) * B * LATENT)  # stored latents z0..zK
    var z_work = Tensor.alloc(B * LATENT)         # forward output working tile
    var zk_work = Tensor.alloc(B * LATENT)        # reverse-scan zk forward input
    var din = Tensor.alloc(B * DYN_IN)
    var dout = Tensor.alloc(B * DYN_OUT)
    var pout = Tensor.alloc(B * PRED_OUT)
    var gpout = Tensor.alloc(B * PRED_OUT)        # grad wrt pred output
    var gdout = Tensor.alloc(B * DYN_OUT)         # grad wrt dyn output
    var gz = Tensor.alloc(B * LATENT)             # carry: grad wrt z_{k+1}
    var gpin = Tensor.alloc(B * LATENT)           # working grad wrt z_k
    var gdin = Tensor.alloc(B * DYN_IN)           # grad wrt dyn input
    var gobs = Tensor.alloc(B * OBS)              # grad wrt rep input (discarded)
    var twv = Tensor.alloc(B * BINS)
    var twr = Tensor.alloc(B * BINS)
    # per-k target slices copied from the raw input pointers into owned Lists
    # (the loss/two-hot primitives are now List-based).
    var pol_tgt_l = List[Scalar[DT]](length=B * ACT, fill=0)
    var val_tgt_l = List[Scalar[DT]](length=B, fill=0)
    var rew_tgt_l = List[Scalar[DT]](length=B, fill=0)

    var gscale = Scalar[DT](1.0) / Scalar[DT]((K + 1) * B)

    # ── forward scan: rep then K dynamics steps, store every z ──
    call_forward["cpu", B](rep, TensorRefs[REP.ARITY](obs0_t), z_work, None)
    for i in range(B * LATENT):
        zst.data[i] = z_work.data[i]

    for k in range(K):
        # build dyn input [z_k | onehot(a_k)]
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
        # store next latent z_{k+1} = dyn_out[:, :LATENT]
        var znoff = (k + 1) * B * LATENT
        for b in range(B):
            for i in range(LATENT):
                zst.data[znoff + b * LATENT + i] = dout.data[b * DYN_OUT + i]

    # ── reverse scan: accumulate grads + loss ──
    rep.zero_grad["cpu"](None)
    dyn.zero_grad["cpu"](None)
    pred.zero_grad["cpu"](None)

    var loss = Scalar[DT](0.0)
    # per-component loss accumulators (for the optional loss_parts breakdown)
    var l_pol = Scalar[DT](0.0)
    var l_val = Scalar[DT](0.0)
    var l_rew = Scalar[DT](0.0)
    for rk in range(K + 1):
        var k = K - rk
        var zoff = k * B * LATENT
        # load z_k into the forward-input working tile
        for i in range(B * LATENT):
            zk_work.data[i] = zst.data[zoff + i]

        # (a) prediction head: re-forward for cache, seed grads, vjp → grad z_k
        call_forward["cpu", B](pred, TensorRefs[PRED.ARITY](zk_work), pout, None)
        # policy slice [0, ACT)
        for i in range(B * ACT):
            pol_tgt_l[i] = policy_tgt[k * B * ACT + i]
        var l_pol_k = soft_ce_slice_loss_and_grad[B, PRED_OUT, 0, ACT](
            pout.data, pol_tgt_l, gscale, gpout.data
        )
        loss += l_pol_k
        l_pol += l_pol_k
        # value slice [ACT, ACT+BINS)
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

        # (b) dynamics: carry grad from z_{k+1} + reward head, ½ on hidden input
        if k < K:
            # rebuild dyn input (mirror forward) for cache
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
            # grad_dyn_out = [ carry(grad z_{k+1}) | reward grad ]
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
            # ∂L/∂z_k += ½ · (grad into dyn's latent input)
            for b in range(B):
                for i in range(LATENT):
                    gpin.data[b * LATENT + i] += (
                        Scalar[DT](0.5) * gdin.data[b * DYN_IN + i]
                    )

        # carry ← full grad wrt z_k for the next (k-1) iteration
        for i in range(B * LATENT):
            gz.data[i] = gpin.data[i]

    # ── rep: grad wrt z_0 (== carry after the loop) → rep params ──
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

    if loss_parts:
        var lp = loss_parts.value()
        var inv = Scalar[DT](1.0) / Scalar[DT](B)
        lp[unsafe_offset=0] = l_pol * inv   # policy
        lp[unsafe_offset=1] = l_val * inv   # value
        lp[unsafe_offset=2] = l_rew * inv   # reward
    return loss / Scalar[DT](B)


# ──────────────────────────────────────────────────────────────────────
# GPU unroll — device kernels (mirror the host loops of the CPU step).
#
# All flat 1-D over `MutAnyOrigin` device pointers (the validated nn GPU
# marshalling idiom — see `dreamerv3/blocks.mojo` `_bcopy`). Concrete `DT`
# everywhere, so no `where dtype.is_floating_point()` guard is needed.
# Bit-for-bit the same arithmetic as `loss_ops` / `twohot_targets` so the
# CPU↔GPU parity test holds within fp32 noise.
# ──────────────────────────────────────────────────────────────────────


def _mz_build_dyn_in_k[
    B_: Int, LATENT_: Int, ACT_: Int, DYN_IN_: Int, ADT: DType = DT,
](
    din: Tile[ADT, B_ * DYN_IN_],
    zk: Tile[ADT, B_ * LATENT_],
    act: Tile[DT, B_],
):
    """Assemble the dynamics input row `[z_k | onehot(a_k)]` per sample.
    `act[b]` is the (float-encoded) action index for this unroll step."""
    var idx = Int(global_idx.x)
    if idx < B_ * DYN_IN_:
        var b = idx // DYN_IN_
        var d = idx % DYN_IN_
        if d < LATENT_:
            din[idx] = rebind[Scalar[ADT]](zk[b * LATENT_ + d])
        else:
            var sel = Int(rebind[Scalar[DT]](act[b]))
            din[idx] = (
                Scalar[ADT](1.0) if (d - LATENT_) == sel else Scalar[ADT](0.0)
            )


def _mz_copy_latent_k[
    B_: Int, LATENT_: Int, DYN_OUT_: Int, ADT: DType = DT,
](
    znext: Tile[ADT, B_ * LATENT_],
    dout: Tile[ADT, B_ * DYN_OUT_],
):
    """Copy the latent half `dout[:, :LATENT]` → `znext` (the next z_{k+1})."""
    var idx = Int(global_idx.x)
    if idx < B_ * LATENT_:
        var b = idx // LATENT_
        var i = idx % LATENT_
        znext[idx] = rebind[Scalar[ADT]](dout[b * DYN_OUT_ + i])


def _mz_softce_slice_k[
    B_: Int, ROW_: Int, OFF_: Int, NBINS_: Int, ADT: DType = DT,
](
    logits: Tile[ADT, B_ * ROW_],
    target: Tile[DT, B_ * NBINS_],
    grad_out: Tile[ADT, B_ * ROW_],
    loss_buf: Tile[DT, B_],
    grad_scale: Scalar[DT],
    loss_coef: Scalar[DT],
):
    """Soft-CE over the `[OFF, OFF+NBINS)` column slice of a `[B, ROW]` logits
    tile vs a `[B, NBINS]` soft target. Writes `grad_scale·(softmax − q)` into
    the same slice of `grad_out` and **accumulates** `loss_coef·(−Σ q·log sm)`
    into `loss_buf[b]`. One thread per row — mirrors
    `soft_ce_slice_loss_and_grad` exactly."""
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
            var log_sm = (
                rebind[Scalar[ADT]](logits[base + i]).cast[DT]() - m
            ) - log_s
            row_loss += -q * log_sm
            var sm = exp(log_sm)
            grad_out[base + i] = (grad_scale * (sm - q)).cast[ADT]()
        loss_buf[b] = rebind[Scalar[DT]](loss_buf[b]) + loss_coef * row_loss


# ── PER kernels (mirror efficient_zero_v2/blocks.mojo) ──────────────────────
def _mz_scale_rows_k[B_: Int, ROW_: Int, OFF_: Int, LEN_: Int](
    grad: Tile[DT, B_ * ROW_],
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
            grad[base + c] = rebind[Scalar[DT]](grad[base + c]) * wb


def _mz_priority_ce_k[B_: Int, ROW_: Int, OFF_: Int, NBINS_: Int](
    logits: Tile[DT, B_ * ROW_],
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
        var m = rebind[Scalar[DT]](logits[base])
        for i in range(1, NBINS_):
            var v = rebind[Scalar[DT]](logits[base + i])
            if v > m:
                m = v
        var s = Scalar[DT](0.0)
        for i in range(NBINS_):
            s += exp(rebind[Scalar[DT]](logits[base + i]) - m)
        var log_s = log(s)
        var tb = b * NBINS_
        var row_loss = Scalar[DT](0.0)
        for i in range(NBINS_):
            var q = rebind[Scalar[DT]](target[tb + i])
            row_loss += -q * ((rebind[Scalar[DT]](logits[base + i]) - m) - log_s)
        out_prio[b] = row_loss


def _mz_twohot_k[
    B_: Int, NBINS_: Int,
](
    tgt: Tile[DT, B_ * NBINS_],
    vals: Tile[DT, B_],
    v_min: Scalar[DT],
    v_max: Scalar[DT],
):
    """Two-hot encode raw scalars: apply `h(x)` then two-hot over `NBINS`
    linear bins in `[v_min, v_max]` (h-space support). Mirrors
    `mz_two_hot_target_batch` (`mz_scalar_transform` + `two_hot_encode`)."""
    var b = Int(global_idx.x)
    if b < B_:
        var x = rebind[Scalar[DT]](vals[b])
        # h(x) = sign(x)·(√(|x|+1) − 1) + ε·x   (ε = 0.001)
        var eps = Scalar[DT](0.001)
        var sgn = Scalar[DT](1.0) if x >= Scalar[DT](0.0) else Scalar[DT](-1.0)
        var ax = x if x >= Scalar[DT](0.0) else -x
        var ht = sgn * (sqrt(ax + Scalar[DT](1.0)) - Scalar[DT](1.0)) + eps * x
        var base = b * NBINS_
        for i in range(NBINS_):
            tgt[base + i] = Scalar[DT](0.0)
        if NBINS_ == 1:
            tgt[base] = Scalar[DT](1.0)
            return
        # clamp into the bin range, then two-hot over linear bins.
        if ht < v_min:
            ht = v_min
        if ht > v_max:
            ht = v_max
        var step = (v_max - v_min) / Scalar[DT](NBINS_ - 1)
        var kf = (ht - v_min) / step
        var k = Int(kf)
        if k >= NBINS_ - 1:
            k = NBINS_ - 2
        if k < 0:
            k = 0
        var bin_low = v_min + step * Scalar[DT](k)
        var bin_high = v_min + step * Scalar[DT](k + 1)
        var width = bin_high - bin_low
        var upper = (bin_high - ht) / width
        tgt[base + k] = upper
        tgt[base + k + 1] = Scalar[DT](1.0) - upper


def _mz_set_carry_latent_k[
    B_: Int, LATENT_: Int, DYN_OUT_: Int, ADT: DType = DT,
](
    gdout: Tile[ADT, B_ * DYN_OUT_],
    gz: Tile[ADT, B_ * LATENT_],
):
    """Seed the latent half of the dynamics output-grad with the carry
    `∂L/∂z_{k+1}` (the reward kernel fills the `[LATENT, LATENT+BINS)` half)."""
    var idx = Int(global_idx.x)
    if idx < B_ * LATENT_:
        var b = idx // LATENT_
        var i = idx % LATENT_
        gdout[b * DYN_OUT_ + i] = rebind[Scalar[ADT]](gz[idx])


def _mz_accum_half_k[
    B_: Int, LATENT_: Int, DYN_IN_: Int, DDT: DType = DT, SDT: DType = DT,
](
    gpin: Tile[DDT, B_ * LATENT_],
    gdin: Tile[SDT, B_ * DYN_IN_],
):
    """`∂L/∂z_k += ½·(grad into dyn's latent input)` — the MuZero ½ dynamics
    hidden-input gradient scaling (appendix). Dual-dtype: fp32 accumulator
    (`gpin`, `DDT=DT`) folds a possibly-bf16 (`SDT`) `gdin` — keeps the bf16
    BPTT grad carry in fp32."""
    var idx = Int(global_idx.x)
    if idx < B_ * LATENT_:
        var b = idx // LATENT_
        var i = idx % LATENT_
        gpin[idx] = (
            rebind[Scalar[DDT]](gpin[idx]).cast[DT]()
            + Scalar[DT](0.5)
            * rebind[Scalar[SDT]](gdin[b * DYN_IN_ + i]).cast[DT]()
        ).cast[DDT]()


def _mz_bcopy_k[N_: Int, ADT: DType = DT](
    src: Tile[ADT, N_],
    dst: Tile[ADT, N_],
):
    """Contiguous device→device copy (carry gz ← gpin)."""
    var i = Int(global_idx.x)
    if i < N_:
        dst[i] = rebind[Scalar[ADT]](src[i])


def _mz_train_prologue_gpu[
    B: Int, K: Int, OBS: Int, ACT: Int, LATENT: Int, BINS: Int,
    obs_on_device: Bool = False,
](
    ctx: DeviceContext,
    mut scratch: MZScratch[B, K, OBS, ACT, LATENT, BINS],
    obs0: List[Scalar[DT]],
    actions: List[Scalar[DT]],
    policy_tgt: List[Scalar[DT]],
    value_tgt: List[Scalar[DT]],
    reward_tgt: List[Scalar[DT]],
    is_weights: Optional[Pointer[Scalar[DT], MutAnyOrigin]] = None,
) raises:
    """EAGER prologue of the unroll step: H2D the host batch slabs into the
    persistent device scratch + zero the loss accumulators. Host work + unsafe_memcpy —
    kept OUT of the CUDA-graph-captured region (a captured H2D would bake the
    host source pointer). Run this each iteration before the captured compute;
    the compute reads the freshly-overwritten ``scratch.d_*`` buffers."""
    # ``obs_on_device``: the caller already filled ``scratch.d_obs0`` on-device
    # so skip the obs H2D and ignore ``obs0``.
    # `.unsafe_ptr()` here is the sanctioned H2D-staging boundary (host List →
    # device buffer); the batch inputs are owned Lists everywhere else.
    comptime if not obs_on_device:
        ctx.enqueue_copy(scratch.d_obs0.dev.value(), obs0.unsafe_ptr())
    ctx.enqueue_copy(scratch.d_act.dev.value(), actions.unsafe_ptr())
    ctx.enqueue_copy(scratch.d_pol.dev.value(), policy_tgt.unsafe_ptr())
    ctx.enqueue_copy(scratch.d_val.dev.value(), value_tgt.unsafe_ptr())
    ctx.enqueue_copy(scratch.d_rew.dev.value(), reward_tgt.unsafe_ptr())

    # ── PER (optional): copy IS weights to device; priorities written at k=0 ──
    if is_weights:
        ctx.enqueue_copy(scratch.d_isw.dev.value(), is_weights.value())

    # zero the 3 loss-component accumulators (policy | value | reward)
    var zloss = scratch.h_zloss.value()
    for i in range(3 * B):
        zloss[i] = Scalar[DT](0.0)
    ctx.enqueue_copy(scratch.loss_d.dev.value(), zloss)


def _mz_train_fwdrev_gpu[
    REP: Module,
    DYN: Module,
    PRED: Module,
    B: Int,
    K: Int,
    OBS: Int,
    ACT: Int,
    LATENT: Int,
    BINS: Int,
](
    ctx: DeviceContext,
    mut rep: REP,
    mut dyn: DYN,
    mut pred: PRED,
    mut scratch: MZScratch[B, K, OBS, ACT, LATENT, BINS],
    v_min: Scalar[DT],
    v_max: Scalar[DT],
    value_coef: Scalar[DT],
    has_isw: Bool,
    want_prio: Bool,
) raises:
    """PURE device-kernel forward + reverse scan of the K-step unroll. Reads the
    device batch slabs (filled by the prologue), runs the forward scan (rep + K
    dynamics) and reverse scan (pred/dyn vjp with the ½ dynamics gradient +
    1/(K+1) loss weight), leaving grads in the three nets' Param grad slabs + the
    loss accumulators + (optionally) the value-error priorities — all on-device.
    NO H2D, NO optimizer step, NO sync ⇒ this is the body that drops into a
    CUDA-graph capture. ``has_isw`` / ``want_prio`` are run-constant booleans
    (fixed for a run) so the enqueued kernel sequence is identical on each
    replay."""
    comptime PRED_OUT = ACT + BINS
    comptime DYN_IN = LATENT + ACT
    comptime DYN_OUT = LATENT + BINS

    var gscale = Scalar[DT](1.0) / Scalar[DT]((K + 1) * B)
    var octx = Optional[DeviceContext](ctx)

    # ── reuse persistent scratch (owned storage Tensors, alloc'd in `make`) ──
    ref d_obs0 = scratch.d_obs0
    ref d_act = scratch.d_act
    ref d_pol = scratch.d_pol
    ref d_val = scratch.d_val
    ref d_rew = scratch.d_rew
    ref zst = scratch.zst
    ref z_work = scratch.z_work
    ref zk_work = scratch.zk_work
    ref din = scratch.din
    ref dout = scratch.dout
    ref pout = scratch.pout
    ref gpout = scratch.gpout
    ref gdout = scratch.gdout
    ref gz = scratch.gz
    ref gpin = scratch.gpin
    ref gdin = scratch.gdin
    ref gobs = scratch.gobs
    ref twv = scratch.twv
    ref twr = scratch.twr
    ref loss_d = scratch.loss_d
    ref d_isw = scratch.d_isw
    ref d_prio = scratch.d_prio

    # device-view layouts (built off the storage Tensors via `.lt` / `.lt_at`)
    comptime LB = RowMajor[B]
    comptime LBL = RowMajor[B * LATENT]
    comptime LBDI = RowMajor[B * DYN_IN]
    comptime LBDO = RowMajor[B * DYN_OUT]
    comptime LBPO = RowMajor[B * PRED_OUT]
    comptime LBBINS = RowMajor[B * BINS]
    comptime LBACT = RowMajor[B * ACT]

    comptime nbDIN = (B * DYN_IN + TPB - 1) // TPB
    comptime nbLAT = (B * LATENT + TPB - 1) // TPB
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
    # PER: scale the whole pred-head grad row; scale ONLY the reward slice of
    # the dyn grad; root value-error priority (mirror efficient_zero_v2).
    comptime kScalePred = _mz_scale_rows_k[B, PRED_OUT, 0, PRED_OUT]
    comptime kScaleRew = _mz_scale_rows_k[B, DYN_OUT, LATENT, BINS]
    comptime kPrioCE = _mz_priority_ce_k[B, PRED_OUT, ACT, BINS]

    # ── forward scan: z0 = h(obs0); z_{k+1} = g(z_k, a_k).latent ──
    call_forward["gpu", B](rep, TensorRefs[REP.ARITY](d_obs0), z_work, octx)
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

    # ── reverse scan ──
    rep.zero_grad["gpu"](octx)
    dyn.zero_grad["gpu"](octx)
    pred.zero_grad["gpu"](octx)

    for rk in range(K + 1):
        var k = K - rk
        # load z_k into the forward-input working tile
        ctx.enqueue_function[kBcopy](
            zst.tt_at["gpu", B * LATENT](k * B * LATENT),
            zk_work.tt["gpu", B * LATENT](),
            grid_dim=nbLAT, block_dim=TPB,
        )

        # (a) prediction head: re-forward (cache), seed grads, vjp → grad z_k
        call_forward["gpu", B](pred, TensorRefs[PRED.ARITY](zk_work), pout, octx)
        # policy slice [0, ACT)
        ctx.enqueue_function[kPolCE](
            pout.tt["gpu", B * PRED_OUT](),
            d_pol.tt_at["gpu", B * ACT](k * B * ACT),
            gpout.tt["gpu", B * PRED_OUT](),
            loss_d.tt_at["gpu", B](0),
            gscale, Scalar[DT](1.0),
            grid_dim=nbB, block_dim=TPB,
        )
        # value slice [ACT, ACT+BINS): two-hot then soft-CE (scaled by coef)
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
        # PER: value-error priority at the root (k=0), read from value logits +
        # value two-hot (both still intact — reads logits, not grads).
        if want_prio and k == 0:
            ctx.enqueue_function[kPrioCE](
                pout.tt["gpu", B * PRED_OUT](), twv.tt["gpu", B * BINS](),
                d_prio.tt["gpu", B](), grid_dim=nbB, block_dim=TPB,
            )
        # PER: weight the whole prediction-head grad row by w_b before vjp.
        if has_isw:
            ctx.enqueue_function[kScalePred](
                gpout.tt["gpu", B * PRED_OUT](), d_isw.tt["gpu", B](),
                grid_dim=nbB, block_dim=TPB,
            )
        call_vjp["gpu", B](
            pred,
            TensorRefs[PRED.ARITY](zk_work),
            gpout,
            TensorRefs[PRED.ARITY](gpin),
            octx,
        )

        # (b) dynamics: carry grad from z_{k+1} + reward head, ½ on hidden input
        if k < K:
            ctx.enqueue_function[kBuild](
                din.tt["gpu", B * DYN_IN](),
                zst.tt_at["gpu", B * LATENT](k * B * LATENT),
                d_act.tt_at["gpu", B](k * B),
                grid_dim=nbDIN, block_dim=TPB,
            )
            call_forward["gpu", B](dyn, TensorRefs[DYN.ARITY](din), dout, octx)
            # grad_dyn_out = [ carry(grad z_{k+1}) | reward grad ]
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
                loss_d.tt_at["gpu", B](2 * B),        # reward block
                gscale, Scalar[DT](1.0),
                grid_dim=nbB, block_dim=TPB,
            )
            # PER: weight ONLY the reward slice of the dyn grad by w_b. The
            # latent slice carries the already-weighted gradient from z_{k+1}
            # (kCarry), so scaling it again would double-weight it.
            if has_isw:
                ctx.enqueue_function[kScaleRew](
                    gdout.tt["gpu", B * DYN_OUT](), d_isw.tt["gpu", B](),
                    grid_dim=nbB, block_dim=TPB,
                )
            call_vjp["gpu", B](
                dyn,
                TensorRefs[DYN.ARITY](din),
                gdout,
                TensorRefs[DYN.ARITY](gdin),
                octx,
            )
            # ∂L/∂z_k += ½ · (grad into dyn's latent input)
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

    # ── rep: grad wrt z_0 (== carry after the loop) → rep params ──
    call_vjp["gpu", B](
        rep, TensorRefs[REP.ARITY](d_obs0), gz, TensorRefs[REP.ARITY](gobs), octx
    )


def _mz_arena_opt_step_gpu[
    REP: Module, DYN: Module, PRED: Module
](
    ctx: DeviceContext,
    mut rep: REP,
    mut dyn: DYN,
    mut pred: PRED,
    mut orep: Adam,
    mut odyn: Adam,
    mut opred: Adam,
    max_grad_norm: Float64,
) raises:
    """CUDA-graph-safe optimizer step for the captured train loop. Per net:
    capture-safe grad-norm clip (`clip_grads_device`, no D2H) → arena grouped
    step (`step_captured`, advances device `β^t` + reads device LR). Requires the
    three optimizers to be `adopt`-ed (arena mode); pure device kernels, no host
    bookkeeping ⇒ replay-safe. Mirrors the per-param `clip_grad_norm` + step the
    non-captured monolithic uses, but on the arena path."""
    var octx = Optional[DeviceContext](ctx)
    var mgn = Scalar[DT](max_grad_norm)
    opred.clip_grads_device["gpu"](pred, mgn, octx)
    opred.step_captured(ctx)
    odyn.clip_grads_device["gpu"](dyn, mgn, octx)
    odyn.step_captured(ctx)
    orep.clip_grads_device["gpu"](rep, mgn, octx)
    orep.step_captured(ctx)


def _mz_train_epilogue_gpu[
    B: Int, K: Int, OBS: Int, ACT: Int, LATENT: Int, BINS: Int,
](
    ctx: DeviceContext,
    mut scratch: MZScratch[B, K, OBS, ACT, LATENT, BINS],
    want_prio: Bool,
    loss_parts: Optional[Pointer[Scalar[DT], MutAnyOrigin]] = None,
    out_prio: Optional[Pointer[Scalar[DT], MutAnyOrigin]] = None,
) raises -> Scalar[DT]:
    """EAGER epilogue: sync, D2H the loss components (+ priorities) and reduce on
    the host → returns the mean total loss (same reduction as the CPU path). Kept
    OUT of the captured region (sync + D2H + host reduce). The loss/prio device
    buffers were written by the captured/replayed compute, so this reads fresh
    values every iteration."""
    # ── D2H PER priorities + loss with a single sync — 3 [B] blocks:
    #    policy | value | reward (loss); + [B] value-error priorities ──
    ctx.synchronize()
    var hloss = scratch.h_loss.value()
    if want_prio:
        ctx.enqueue_copy(scratch.h_prio.value(), scratch.d_prio.dev.value())
    ctx.enqueue_copy(hloss, scratch.loss_d.dev.value())
    ctx.synchronize()
    if out_prio:
        var op = out_prio.value()
        var hpp = scratch.h_prio.value()
        for b in range(B):
            op[unsafe_offset=b] = hpp[b]
    var l_pol = Scalar[DT](0.0)
    var l_val = Scalar[DT](0.0)
    var l_rew = Scalar[DT](0.0)
    for b in range(B):
        l_pol += hloss[b]
        l_val += hloss[B + b]
        l_rew += hloss[2 * B + b]
    var inv = Scalar[DT](1.0) / Scalar[DT](B)
    if loss_parts:
        var lp = loss_parts.value()
        lp[unsafe_offset=0] = l_pol * inv
        lp[unsafe_offset=1] = l_val * inv
        lp[unsafe_offset=2] = l_rew * inv
    return (l_pol + l_val + l_rew) * inv


def mz_unroll_train_step_gpu[
    REP: Module,
    DYN: Module,
    PRED: Module,
    B: Int,
    K: Int,
    OBS: Int,
    ACT: Int,
    LATENT: Int,
    BINS: Int,
    obs_on_device: Bool = False,
](
    ctx: DeviceContext,
    mut rep: REP,
    mut dyn: DYN,
    mut pred: PRED,
    mut orep: Adam,
    mut odyn: Adam,
    mut opred: Adam,
    mut scratch: MZScratch[B, K, OBS, ACT, LATENT, BINS],
    obs0: List[Scalar[DT]],
    actions: List[Scalar[DT]],
    policy_tgt: List[Scalar[DT]],
    value_tgt: List[Scalar[DT]],
    reward_tgt: List[Scalar[DT]],
    v_min: Scalar[DT],
    v_max: Scalar[DT],
    value_coef: Scalar[DT] = Scalar[DT](1.0),
    max_grad_norm: Float64 = 0.0,
    loss_parts: Optional[Pointer[Scalar[DT], MutAnyOrigin]] = None,
    is_weights: Optional[Pointer[Scalar[DT], MutAnyOrigin]] = None,
    out_prio: Optional[Pointer[Scalar[DT], MutAnyOrigin]] = None,
) raises -> Scalar[DT]:
    """GPU MuZero K-step unroll training step — device mirror of
    `mz_unroll_train_step_cpu`.

    PER (optional): when ``is_weights`` is given the per-sample importance
    weights scale the whole prediction-head grad row + the reward slice of the
    dyn grad before each vjp (the latent carry is already weighted, so it is
    NOT re-scaled). When ``out_prio`` is given the root (k=0) value-head soft-CE
    is written per row as the new priority signal. Both are no-ops when ``None``
    → bit-identical to the uniform path. NOTE: the batched GPU self-play driver
    does NOT pass ``out_prio`` here — it prioritizes on the MuZero paper signal
    |ν − z| computed in the replay sampler (`sample_training_batch_per_dev`), so
    this soft-CE path is currently exercised only by callers that opt into it.

    Takes the same **host** time-major batch slabs (raw value/reward scalars;
    `h` + two-hot applied on device), H2D-copies them once, runs the forward
    scan (rep + K dynamics) and reverse scan (pred/dyn vjp with the ½ dynamics
    gradient + 1/(K+1) loss weight) entirely on the device, and steps the three
    Adam optimizers in place. Returns the mean total loss (same reduction as
    the CPU path). Device + host scratch is supplied by a persistent
    `MZScratch` (allocated once in `make`, reused every step) — this
    function performs **no** `enqueue_create_buffer` of its own, which would
    explode disk on NVIDIA when called in the per-step hot loop.
    """
    var has_isw = Bool(is_weights)
    var want_prio = Bool(out_prio)
    var octx = Optional[DeviceContext](ctx)

    _mz_train_prologue_gpu[B, K, OBS, ACT, LATENT, BINS, obs_on_device](
        ctx, scratch, obs0, actions, policy_tgt, value_tgt, reward_tgt,
        is_weights,
    )
    _mz_train_fwdrev_gpu[REP, DYN, PRED, B, K, OBS, ACT, LATENT, BINS](
        ctx, rep, dyn, pred, scratch, v_min, v_max, value_coef,
        has_isw, want_prio,
    )

    # Global grad-norm clip per net (max_grad_norm <= 0 ⇒ no-op), then step.
    # Per-param walk (the universal correctness path); the captured path uses
    # `_mz_arena_opt_step_gpu` instead (arena, capture-safe).
    _ = clip_grad_norm["gpu", PRED](pred, Scalar[DT](max_grad_norm), octx)
    opred.begin_step()
    walk_params["gpu"](pred, opred, octx)
    _ = clip_grad_norm["gpu", DYN](dyn, Scalar[DT](max_grad_norm), octx)
    odyn.begin_step()
    walk_params["gpu"](dyn, odyn, octx)
    _ = clip_grad_norm["gpu", REP](rep, Scalar[DT](max_grad_norm), octx)
    orep.begin_step()
    walk_params["gpu"](rep, orep, octx)

    return _mz_train_epilogue_gpu[B, K, OBS, ACT, LATENT, BINS](
        ctx, scratch, want_prio, loss_parts, out_prio
    )
