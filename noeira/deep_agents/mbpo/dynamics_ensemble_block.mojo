"""DynamicsEnsembleBlock[DynNet, N, NUM_ELITES, IN_DIM, OUT_DIM, BATCH].

The probabilistic-ensemble world model used by MBPO. Owns N independent
dynamics networks + N independent optimisers + one shared GaussianNLLLoss
instance + per-member elite ranking.

STORAGE migration (Stage 5): members are `nn.storage` Modules, optimisers are
storage `Adam` (with decoupled weight decay `wd`), the loss is the storage
`GaussianNLLLoss`, and all member-indexed scratch is owned `nn.storage.Tensor`
(was legacy `Scratch`/`TargetStorage`). Member forward/vjp take `TensorRefs` +
owned `Tensor`s (was `TileTensor`). CPU + GPU.

Why a single block (not N free-standing nets in the trainer)?
  - Lifecycle uniformity: one `make[target, INIT]` builds and inits all
    members + opts + scratch.
  - Elite-ranking state (`elite_indices`) is per-ensemble.
  - Member-indexed scratch (`_mb_pred`, `_mb_grad`) is shared across all
    member calls — one slab per direction, reused per member step.

Conventions:
  - `predict_member`: pure forward through `members[member_idx]` + in-place
    logvar clamp; result split into `out_mu` (PRED cols) + `out_lv` (PRED cols).
  - `train_member_step`: one Gaussian-NLL gradient step on the named member's
    parameters; returns the scalar loss.
  - `eval_member_loss`: forward + loss only, no gradient — holdout scoring.
  - `update_elites`: re-rank by holdout losses (lowest NUM_ELITES are elite).

Fixed logvar bounds `[LOGVAR_MIN, LOGVAR_MAX]` delegate to the storage
GaussianNLLLoss (clamp+split done internally for loss/grad; `predict_member`
keeps its own split/clamp for sampling). The opt-in learnable-bounds path
keeps the bespoke double-softplus soft-clamp NLL grad + per-dim bound Adam.
"""

from std.math import exp as fexp, log as flog, sqrt as fsqrt
from max.gpu import global_idx, thread_idx
from max.gpu.primitives import block
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT, TPB, TPB_REDUCE
from noeira.nn.core.module import Module
from noeira.nn.core.amp import AMPPolicy, NoAMP
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.initializer import Initializer
from noeira.nn.core.call import call_forward, call_vjp
from noeira.nn.loss.gaussian_nll_loss import GaussianNLLLoss
from noeira.nn.optimizer.adam import Adam
from noeira.core.tiles import Tile


def _split_clamp_logvar_kernel[
    BATCH: Int, OUT_DIM: Int, PRED_DIM: Int
](
    pred: Tile[DT, BATCH, OUT_DIM],
    out_mu: Tile[DT, BATCH, PRED_DIM],
    out_lv: Tile[DT, BATCH, PRED_DIM],
    lv_min: Scalar[DT],
    lv_max: Scalar[DT],
):
    """GPU mirror of the CPU split+clamp loop in `predict_member`:
    `out_mu[b,j] = pred[b,j]`, `out_lv[b,j] = clamp(pred[b,PRED+j], lv_min,
    lv_max)`. One thread per (b, j) over BATCH × PRED_DIM."""
    var idx = Int(global_idx.x)
    var total = BATCH * PRED_DIM
    if idx >= total:
        return
    var b = idx // PRED_DIM
    var j = idx % PRED_DIM
    out_mu[b, j] = rebind[Scalar[DT]](pred[b, j])
    var v = rebind[Scalar[DT]](pred[b, PRED_DIM + j])
    if v > lv_max:
        v = lv_max
    elif v < lv_min:
        v = lv_min
    out_lv[b, j] = v


# ──────────────────────────────────────────────────────────────────────
# Learnable per-member/per-dim logvar bounds (MBPO/PETS reference). Faithful
# port: soft DOUBLE-softplus clamp lets gradients flow to BOTH the network
# logvar output AND the bounds, which are Adam-updated with a 0.01 L2 penalty
# that learns the upper bound DOWN from +0.5 to ≈[−1,−2].
# ──────────────────────────────────────────────────────────────────────


def _softplus_dt(x: Scalar[DT]) -> Scalar[DT]:
    """Numerically-stable softplus log(1+exp(x)) with ±20 saturation."""
    if x > Scalar[DT](20.0):
        return x
    elif x > Scalar[DT](-20.0):
        return flog(Scalar[DT](1.0) + fexp(x))
    return Scalar[DT](0.0)


def _sigmoid_dt(x: Scalar[DT]) -> Scalar[DT]:
    """Sigmoid 1/(1+exp(-x)) with ±20 saturation = d softplus(x)/dx."""
    if x > Scalar[DT](20.0):
        return Scalar[DT](1.0)
    elif x > Scalar[DT](-20.0):
        return Scalar[DT](1.0) / (Scalar[DT](1.0) + fexp(-x))
    return Scalar[DT](0.0)


def _soft_clamp_lv(
    raw: Scalar[DT], max_d: Scalar[DT], min_d: Scalar[DT]
) -> Scalar[DT]:
    """Double-softplus soft clamp of `raw` into (min_d, max_d):
    `lv = min_d + softplus((max_d − softplus(max_d − raw)) − min_d)`."""
    var lv_inter = max_d - _softplus_dt(max_d - raw)
    return min_d + _softplus_dt(lv_inter - min_d)


def _dyn_learnable_nll_grad_kernel[
    BATCH: Int, PRED_DIM: Int
](
    pred: Tile[DT, BATCH, 2 * PRED_DIM],
    target: Tile[DT, BATCH, PRED_DIM],
    max_lv: Tile[DT, PRED_DIM],
    min_lv: Tile[DT, PRED_DIM],
    grad_out: Tile[DT, BATCH, 2 * PRED_DIM],
    gmax: Tile[DT, BATCH, PRED_DIM],
    gmin: Tile[DT, BATCH, PRED_DIM],
    ploss: Tile[DT, BATCH],
):
    """Combined Gaussian-NLL forward + grad with soft logvar bounds. One
    thread per batch row. Grads use legacy normalisation `2/(BATCH·PRED)`."""
    var b = Int(global_idx.x)
    if b >= BATCH:
        return
    var inv_norm = Scalar[DT](2.0) / Scalar[DT](BATCH * PRED_DIM)
    var row_loss = Scalar[DT](0.0)
    for d in range(PRED_DIM):
        var mu = rebind[Scalar[DT]](pred[b, d])
        var raw = rebind[Scalar[DT]](pred[b, PRED_DIM + d])
        var y = rebind[Scalar[DT]](target[b, d])
        var max_d = rebind[Scalar[DT]](max_lv[d])
        var min_d = rebind[Scalar[DT]](min_lv[d])
        var a1 = max_d - raw
        var g1 = _sigmoid_dt(a1)
        var lv_inter = max_d - _softplus_dt(a1)
        var a2 = lv_inter - min_d
        var g2 = _sigmoid_dt(a2)
        var lv = min_d + _softplus_dt(a2)
        var inv_var = fexp(-lv)
        var diff = mu - y
        var dsq = diff * diff
        row_loss += Scalar[DT](0.5) * dsq * inv_var + Scalar[DT](0.5) * lv
        var grad_lv = Scalar[DT](0.5) * (Scalar[DT](1.0) - dsq * inv_var) * inv_norm
        grad_out[b, d] = diff * inv_var * inv_norm
        grad_out[b, PRED_DIM + d] = grad_lv * g1 * g2
        gmax[b, d] = grad_lv * g2 * (Scalar[DT](1.0) - g1)
        gmin[b, d] = grad_lv * (Scalar[DT](1.0) - g2)
    ploss[b] = row_loss


def _dyn_bounds_adam_kernel[
    PRED_DIM: Int, BATCH: Int
](
    max_lv: Tile[DT, PRED_DIM],
    min_lv: Tile[DT, PRED_DIM],
    max_m: Tile[DT, PRED_DIM],
    max_v: Tile[DT, PRED_DIM],
    min_m: Tile[DT, PRED_DIM],
    min_v: Tile[DT, PRED_DIM],
    gmax: Tile[DT, BATCH, PRED_DIM],
    gmin: Tile[DT, BATCH, PRED_DIM],
    l2_coef: Scalar[DT],
    lr: Scalar[DT],
    beta1: Scalar[DT],
    beta2: Scalar[DT],
    eps: Scalar[DT],
    bc1: Scalar[DT],
    bc2: Scalar[DT],
):
    """Reduce per-batch bound grads, add the L2 term, then Adam-step each
    per-dim bound. One thread per output dim."""
    var d = Int(global_idx.x)
    if d >= PRED_DIM:
        return
    var one = Scalar[DT](1.0)
    var g_max = Scalar[DT](0.0)
    var g_min = Scalar[DT](0.0)
    for b in range(BATCH):
        g_max += rebind[Scalar[DT]](gmax[b, d])
        g_min += rebind[Scalar[DT]](gmin[b, d])
    g_max += l2_coef
    g_min -= l2_coef
    var m1 = beta1 * rebind[Scalar[DT]](max_m[d]) + (one - beta1) * g_max
    var v1 = beta2 * rebind[Scalar[DT]](max_v[d]) + (one - beta2) * g_max * g_max
    max_m[d] = m1
    max_v[d] = v1
    max_lv[d] = rebind[Scalar[DT]](max_lv[d]) - lr * (m1 / bc1) / (
        fsqrt(v1 / bc2) + eps
    )
    var m2 = beta1 * rebind[Scalar[DT]](min_m[d]) + (one - beta1) * g_min
    var v2 = beta2 * rebind[Scalar[DT]](min_v[d]) + (one - beta2) * g_min * g_min
    min_m[d] = m2
    min_v[d] = v2
    min_lv[d] = rebind[Scalar[DT]](min_lv[d]) - lr * (m2 / bc1) / (
        fsqrt(v2 / bc2) + eps
    )


def _soft_clamp_split_kernel[
    BATCH: Int, OUT_DIM: Int, PRED_DIM: Int
](
    pred: Tile[DT, BATCH, OUT_DIM],
    max_lv: Tile[DT, PRED_DIM],
    min_lv: Tile[DT, PRED_DIM],
    out_mu: Tile[DT, BATCH, PRED_DIM],
    out_lv: Tile[DT, BATCH, PRED_DIM],
):
    """Soft-clamp variant of `_split_clamp_logvar_kernel` using the member's
    learnable per-dim bounds. Used by `predict_member` (→ rollout/eval)."""
    var idx = Int(global_idx.x)
    var total = BATCH * PRED_DIM
    if idx >= total:
        return
    var b = idx // PRED_DIM
    var j = idx % PRED_DIM
    out_mu[b, j] = rebind[Scalar[DT]](pred[b, j])
    var raw = rebind[Scalar[DT]](pred[b, PRED_DIM + j])
    out_lv[b, j] = _soft_clamp_lv(
        raw, rebind[Scalar[DT]](max_lv[j]), rebind[Scalar[DT]](min_lv[j])
    )


def _dyn_mse_only_kernel[
    BATCH: Int, OUT_DIM: Int, PRED_DIM: Int
](
    pred: Tile[DT, BATCH, OUT_DIM],
    target: Tile[DT, BATCH, PRED_DIM],
    ploss: Tile[DT, BATCH],
):
    """Per-row mean-over-dims MSE of the MEAN head only — the reference
    holdout/elite metric (`inc_var_loss=False`, bnn.py:193). Variance head is
    ignored; one thread per batch row."""
    var b = Int(global_idx.x)
    if b >= BATCH:
        return
    var row = Scalar[DT](0.0)
    for d in range(PRED_DIM):
        var diff = rebind[Scalar[DT]](pred[b, d]) - rebind[Scalar[DT]](
            target[b, d]
        )
        row += diff * diff
    ploss[b] = row / Scalar[DT](PRED_DIM)


def _dyn_ploss_reduce_kernel[
    BATCH: Int
](
    ploss: Tile[DT, BATCH],
    acc: Tile[DT, 2],
):
    """Block-reduce the per-row learnable-NLL losses and fold the batch MEAN
    into a device accumulator `acc = [sum_of_means, count]`. Mirrors
    GaussianNLLLoss `_gnll_reduce_kernel` so the dynamics loss scalar is read
    ONCE at flush instead of D2H'd (full `ctx.synchronize()`) per train step."""
    var t = Int(thread_idx.x)
    var my_sum = Scalar[DT](0.0)
    var k = t
    while k < BATCH:
        my_sum += rebind[Scalar[DT]](ploss[k])
        k += TPB_REDUCE
    var total = block.sum[block_size=TPB_REDUCE, broadcast=False](val=my_sum)
    if t == 0:
        acc[0] = rebind[Scalar[DT]](acc[0]) + total[0] / Scalar[DT](BATCH)
        acc[1] = rebind[Scalar[DT]](acc[1]) + Scalar[DT](1.0)


struct DynamicsEnsembleBlock[
    DynNet: Module,
    N: Int,
    NUM_ELITES: Int,
    IN_DIM: Int,
    OUT_DIM: Int,
    BATCH: Int,
    LOGVAR_MIN: Float64 = -10.0,
    LOGVAR_MAX: Float64 = -2.0,
](Movable & Deinitable):
    """N-member probabilistic dynamics ensemble.

    `DynNet.OUT_DIM` MUST equal `OUT_DIM == 2 * PRED_DIM` where
    PRED_DIM = 1 + obs_dim (reward + Δobs)."""

    comptime PRED_DIM: Int = Self.OUT_DIM // 2

    var members: List[Self.DynNet]
    var opts: List[Adam]
    var loss: GaussianNLLLoss[Self.PRED_DIM, Self.LOGVAR_MIN, Self.LOGVAR_MAX]
    var elite_indices: List[Int]

    var _target: StaticString
    var ctx: Optional[DeviceContext]

    # Member-indexed scratch (owned storage Tensors, target-resident).
    var _mb_pred: Tensor  # [BATCH * OUT_DIM]
    var _mb_grad: Tensor  # [BATCH * OUT_DIM]

    # ─── Learnable logvar bounds (opt-in) ──────────────────────────────
    var learnable_bounds: Bool
    var _bnd_lr: Scalar[DT]
    var _bnd_step: Int
    var _max_lv: Tensor      # [N * PRED_DIM]
    var _min_lv: Tensor
    var _max_lv_m: Tensor
    var _max_lv_v: Tensor
    var _min_lv_m: Tensor
    var _min_lv_v: Tensor
    var _bnd_gmax: Tensor    # [BATCH * PRED_DIM]
    var _bnd_gmin: Tensor
    var _bnd_ploss: Tensor   # [BATCH] (staging on GPU)
    # Device accumulator [sum_of_batch_means, count] for the learnable-NLL
    # training loss — reduced on-device each step, read once at flush (avoids
    # the per-step D2H + ctx.synchronize() of the old host readback).
    var _dyn_loss_acc: Tensor  # GPU [2] (+ host mirror)

    def __init__(out self):
        comptime assert Self.OUT_DIM == 2 * Self.PRED_DIM, (
            "DynamicsEnsembleBlock: OUT_DIM must be 2 * PRED_DIM"
        )
        comptime assert Self.NUM_ELITES <= Self.N, (
            "NUM_ELITES must not exceed ensemble size N"
        )
        comptime assert Self.OUT_DIM >= Self.IN_DIM, (
            "DynamicsEnsembleBlock: _mb_pred is reused as grad-input"
            " sink during member.vjp, so OUT_DIM must be >= IN_DIM"
        )
        self.members = List[Self.DynNet]()
        self.opts = List[Adam]()
        self.loss = GaussianNLLLoss[
            Self.PRED_DIM, Self.LOGVAR_MIN, Self.LOGVAR_MAX
        ]()
        self.elite_indices = List[Int]()
        self._target = "cpu"
        self.ctx = None
        self._mb_pred = Tensor()
        self._mb_grad = Tensor()
        self.learnable_bounds = False
        self._bnd_lr = Scalar[DT](1e-3)
        self._bnd_step = 0
        self._max_lv = Tensor()
        self._min_lv = Tensor()
        self._max_lv_m = Tensor()
        self._max_lv_v = Tensor()
        self._min_lv_m = Tensor()
        self._min_lv_v = Tensor()
        self._bnd_gmax = Tensor()
        self._bnd_gmin = Tensor()
        self._bnd_ploss = Tensor()
        self._dyn_loss_acc = Tensor()

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer,
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        """Unified CPU/GPU factory. Each member is initialised independently
        from the host RNG so members differ; the ensemble's variance comes
        from initialisation + bootstrap-sample stochasticity in the outer
        loop. CPU drops `ctx`; GPU requires it."""
        comptime assert (
            target == "cpu" or target == "gpu"
        ), "DynamicsEnsembleBlock: target must be 'cpu' or 'gpu'"
        comptime assert Self.DynNet.IN_DIMS[0] == Self.IN_DIM, (
            "DynNet.IN_DIM must equal IN_DIM"
        )
        comptime assert Self.DynNet.OUT_DIM == Self.OUT_DIM, (
            "DynNet.OUT_DIM must equal OUT_DIM"
        )
        comptime if target == "gpu":
            if not ctx:
                raise Error(
                    "DynamicsEnsembleBlock.make[target='gpu']: ctx required"
                )
        var blk = Self()
        blk._target = target
        blk.ctx = ctx
        for _ in range(Self.N):
            var net = Self.DynNet.make[target, INIT](ctx)
            # PETS/MBPO-reference dynamics weight decay (legacy default 5e-5).
            # Storage Adam: `wd` is decoupled (AdamW) and gated by each Param's
            # APPLY_DECAY (Linear weights carry it; biases don't). `adopt` on
            # GPU engages the arena path (decay_mask drives the per-param gate).
            var opt = Adam(lr=Scalar[DT](1e-3), wd=Scalar[DT](5e-5))
            comptime if target == "gpu":
                opt.adopt[target, M=Self.DynNet](net, ctx)
            blk.members.append(net^)
            blk.opts.append(opt^)
        for i in range(Self.NUM_ELITES):
            blk.elite_indices.append(i)
        comptime if target == "cpu":
            blk.loss = GaussianNLLLoss[
                Self.PRED_DIM, Self.LOGVAR_MIN, Self.LOGVAR_MAX
            ].make_cpu()
        else:
            blk.loss = GaussianNLLLoss[
                Self.PRED_DIM, Self.LOGVAR_MIN, Self.LOGVAR_MAX
            ].make_gpu(ctx.value())

        # Member-indexed scratch + learnable-bound state.
        blk._mb_pred = Tensor.make[target](Self.BATCH * Self.OUT_DIM, ctx)
        blk._mb_grad = Tensor.make[target](Self.BATCH * Self.OUT_DIM, ctx)
        blk._max_lv = Tensor.make[target](Self.N * Self.PRED_DIM, ctx)
        blk._min_lv = Tensor.make[target](Self.N * Self.PRED_DIM, ctx)
        blk._max_lv_m = Tensor.make[target](Self.N * Self.PRED_DIM, ctx)
        blk._max_lv_v = Tensor.make[target](Self.N * Self.PRED_DIM, ctx)
        blk._min_lv_m = Tensor.make[target](Self.N * Self.PRED_DIM, ctx)
        blk._min_lv_v = Tensor.make[target](Self.N * Self.PRED_DIM, ctx)
        blk._bnd_gmax = Tensor.make[target](Self.BATCH * Self.PRED_DIM, ctx)
        blk._bnd_gmin = Tensor.make[target](Self.BATCH * Self.PRED_DIM, ctx)
        # `_bnd_ploss` keeps BOTH a host mirror (for the D2H reduction) and a
        # device buffer; on CPU only the host list is used.
        comptime if target == "cpu":
            blk._bnd_ploss = Tensor.alloc(Self.BATCH)
            blk._dyn_loss_acc = Tensor.alloc(2)
        else:
            blk._bnd_ploss = Tensor.alloc_gpu(ctx.value(), Self.BATCH)
            # ensure a host mirror exists for D2H of the per-row loss.
            blk._bnd_ploss.ensure(Self.BATCH)
            blk._dyn_loss_acc = Tensor.alloc_gpu(ctx.value(), 2)
            blk._dyn_loss_acc.dev.value().enqueue_fill(Scalar[DT](0.0))
            blk._dyn_loss_acc.ensure(2)  # host mirror for the flush-time read
        blk._init_bounds[target]()
        return blk^

    def _init_bounds[target: StaticString](mut self) raises:
        """Init learnable-bound state: max_lv=+0.5, min_lv=−10, moments=0."""
        comptime if target == "cpu":
            for i in range(Self.N * Self.PRED_DIM):
                self._max_lv.data[i] = Scalar[DT](0.5)
                self._min_lv.data[i] = Scalar[DT](-10.0)
        else:
            self._max_lv.dev.value().enqueue_fill(Scalar[DT](0.5))
            self._min_lv.dev.value().enqueue_fill(Scalar[DT](-10.0))
            self._max_lv_m.dev.value().enqueue_fill(Scalar[DT](0.0))
            self._max_lv_v.dev.value().enqueue_fill(Scalar[DT](0.0))
            self._min_lv_m.dev.value().enqueue_fill(Scalar[DT](0.0))
            self._min_lv_v.dev.value().enqueue_fill(Scalar[DT](0.0))

    # ------------------------------------------------------------------
    # Public knobs.
    # ------------------------------------------------------------------

    def enable_learnable_bounds(mut self):
        """Opt into learnable per-member/per-dim logvar bounds."""
        self.learnable_bounds = True

    def set_lr(mut self, lr: Scalar[DT]):
        """Set every member's Adam LR + the bound-Adam LR."""
        self._bnd_lr = lr
        for i in range(Self.N):
            self.opts[i].lr = lr

    def set_weight_decay(mut self, wd: Scalar[DT]):
        """Set every member's decoupled weight decay (`Adam.wd`). The dynamics
        ensemble REQUIRES decay to generalise — without it members overfit
        (train NLL ↓, holdout NLL ↑) and synthetic data becomes OOD garbage."""
        for i in range(Self.N):
            self.opts[i].wd = wd

    def set_max_grad_norm(mut self, threshold: Scalar[DT]):
        """No-op (kept for API compatibility; never invoked on the MBPO path)."""
        pass

    # ------------------------------------------------------------------
    # Predict — forward through one member, split + clamp logvar.
    # ------------------------------------------------------------------

    def predict_member[target: StaticString, POLICY: AMPPolicy = NoAMP](
        mut self,
        member_idx: Int,
        mut in_t: Tensor,
        mut out_mu_t: Tensor,
        mut out_lv_t: Tensor,
        out_off: Int = 0,
    ) raises:
        """Forward `members[member_idx]` on `in_t` (BATCH × IN_DIM). Split the
        BATCH × OUT_DIM output into `out_mu_t` + `out_lv_t` (both BATCH × PRED),
        written starting at element `out_off` in each (so callers can forward
        an elite directly into its slice of a stacked NELITES·BATCH×PRED buffer
        — no fresh alloc + D2D copy per elite). The clamped logvar is what
        callers sample / log."""
        call_forward[target, Self.BATCH, POLICY=POLICY](
            self.members[member_idx],
            TensorRefs[Self.DynNet.ARITY](in_t), self._mb_pred, self.ctx
        )
        comptime if target == "cpu":
            var lv_min = Scalar[DT](Self.LOGVAR_MIN)
            var lv_max = Scalar[DT](Self.LOGVAR_MAX)
            var bo = member_idx * Self.PRED_DIM
            for b in range(Self.BATCH):
                var src = b * Self.OUT_DIM
                var dst = out_off + b * Self.PRED_DIM
                for j in range(Self.PRED_DIM):
                    out_mu_t.data[dst + j] = self._mb_pred.data[src + j]
                    var raw = self._mb_pred.data[src + Self.PRED_DIM + j]
                    if self.learnable_bounds:
                        out_lv_t.data[dst + j] = _soft_clamp_lv(
                            raw,
                            self._max_lv.data[bo + j],
                            self._min_lv.data[bo + j],
                        )
                    else:
                        var v = raw
                        if v > lv_max:
                            v = lv_max
                        elif v < lv_min:
                            v = lv_min
                        out_lv_t.data[dst + j] = v
        else:
            var ctx = self.ctx.value()
            comptime total = Self.BATCH * Self.PRED_DIM
            comptime n_blocks = (total + TPB - 1) // TPB
            var pred_lt = self._mb_pred.tt["gpu", Self.BATCH, Self.OUT_DIM]()
            var mu_lt = out_mu_t.tt_at["gpu", Self.BATCH, Self.PRED_DIM](out_off)
            var lv_lt = out_lv_t.tt_at["gpu", Self.BATCH, Self.PRED_DIM](out_off)
            if self.learnable_bounds:
                var bo = member_idx * Self.PRED_DIM
                var max_lt = self._max_lv.tt_at["gpu", Self.PRED_DIM](bo)
                var min_lt = self._min_lv.tt_at["gpu", Self.PRED_DIM](bo)
                comptime soft_kernel = _soft_clamp_split_kernel[
                    Self.BATCH, Self.OUT_DIM, Self.PRED_DIM
                ]
                ctx.enqueue_function[soft_kernel](
                    pred_lt, max_lt, min_lt, mu_lt, lv_lt,
                    grid_dim=n_blocks, block_dim=TPB,
                )
            else:
                comptime split_kernel = _split_clamp_logvar_kernel[
                    Self.BATCH, Self.OUT_DIM, Self.PRED_DIM
                ]
                ctx.enqueue_function[split_kernel](
                    pred_lt, mu_lt, lv_lt,
                    Scalar[DT](Self.LOGVAR_MIN), Scalar[DT](Self.LOGVAR_MAX),
                    grid_dim=n_blocks, block_dim=TPB,
                )

    # ------------------------------------------------------------------
    # Learnable-bounds helpers (soft-clamp NLL grad + bound Adam step).
    # ------------------------------------------------------------------

    def _nll_grad_learnable[target: StaticString, ACCUMULATE: Bool = False](
        mut self, member_idx: Int, mut mb_target_t: Tensor,
    ) raises -> Scalar[DT]:
        """Soft-clamp Gaussian-NLL forward+grad for one member. Reads the
        already-computed `_mb_pred` (BATCH × OUT_DIM), writes the network grad
        into `_mb_grad` + per-(b,d) bound grads into `_bnd_gmax`/`_bnd_gmin`.
        Returns the scalar NLL (nn convention).

        `ACCUMULATE=True` (GPU training): fold the batch-mean loss into the
        device accumulator `_dyn_loss_acc` via a reduce kernel and return 0.0 —
        NO per-step D2H/`ctx.synchronize()`. `ACCUMULATE=False` (CPU, holdout,
        tests): D2H + host-sum the scalar as before."""
        var bo = member_idx * Self.PRED_DIM
        comptime if target == "cpu":
            var inv_norm = Scalar[DT](2.0) / Scalar[DT](Self.BATCH * Self.PRED_DIM)
            var total = Scalar[DT](0.0)
            for b in range(Self.BATCH):
                var po = b * Self.OUT_DIM
                var to = b * Self.PRED_DIM
                for d in range(Self.PRED_DIM):
                    var mu = self._mb_pred.data[po + d]
                    var raw = self._mb_pred.data[po + Self.PRED_DIM + d]
                    var y = mb_target_t.data[to + d]
                    var max_d = self._max_lv.data[bo + d]
                    var min_d = self._min_lv.data[bo + d]
                    var a1 = max_d - raw
                    var g1 = _sigmoid_dt(a1)
                    var lv_inter = max_d - _softplus_dt(a1)
                    var a2 = lv_inter - min_d
                    var g2 = _sigmoid_dt(a2)
                    var lv = min_d + _softplus_dt(a2)
                    var inv_var = fexp(-lv)
                    var diff = mu - y
                    var dsq = diff * diff
                    total += Scalar[DT](0.5) * dsq * inv_var + Scalar[DT](0.5) * lv
                    var grad_lv = (
                        Scalar[DT](0.5) * (Scalar[DT](1.0) - dsq * inv_var)
                        * inv_norm
                    )
                    self._mb_grad.data[po + d] = diff * inv_var * inv_norm
                    self._mb_grad.data[po + Self.PRED_DIM + d] = grad_lv * g1 * g2
                    self._bnd_gmax.data[to + d] = grad_lv * g2 * (
                        Scalar[DT](1.0) - g1
                    )
                    self._bnd_gmin.data[to + d] = grad_lv * (
                        Scalar[DT](1.0) - g2
                    )
            return total / Scalar[DT](Self.BATCH)
        else:
            var ctx = self.ctx.value()
            var pred_lt = self._mb_pred.tt["gpu", Self.BATCH, Self.OUT_DIM]()
            var tgt_lt = mb_target_t.tt["gpu", Self.BATCH, Self.PRED_DIM]()
            var max_lt = self._max_lv.tt_at["gpu", Self.PRED_DIM](bo)
            var min_lt = self._min_lv.tt_at["gpu", Self.PRED_DIM](bo)
            var grad_lt = self._mb_grad.tt["gpu", Self.BATCH, Self.OUT_DIM]()
            var gmax_lt = self._bnd_gmax.tt["gpu", Self.BATCH, Self.PRED_DIM]()
            var gmin_lt = self._bnd_gmin.tt["gpu", Self.BATCH, Self.PRED_DIM]()
            var ploss_lt = self._bnd_ploss.tt["gpu", Self.BATCH]()
            comptime n_rows = (Self.BATCH + TPB - 1) // TPB
            comptime grad_kernel = _dyn_learnable_nll_grad_kernel[
                Self.BATCH, Self.PRED_DIM
            ]
            ctx.enqueue_function[grad_kernel](
                pred_lt, tgt_lt, max_lt, min_lt,
                grad_lt, gmax_lt, gmin_lt, ploss_lt,
                grid_dim=n_rows, block_dim=TPB,
            )
            comptime if ACCUMULATE:
                # Device reduce → accumulator; read once at flush (no sync).
                ctx.enqueue_function[_dyn_ploss_reduce_kernel[Self.BATCH]](
                    ploss_lt,
                    self._dyn_loss_acc.tt["gpu", 2](),
                    grid_dim=1, block_dim=TPB_REDUCE,
                )
                return Scalar[DT](0.0)
            else:
                self._bnd_ploss.download(ctx)
                var total = Scalar[DT](0.0)
                for b in range(Self.BATCH):
                    total += self._bnd_ploss.data[b]
                return total / Scalar[DT](Self.BATCH)

    def _bounds_step[target: StaticString](mut self, member_idx: Int) raises:
        """Adam-update member `member_idx`'s per-dim bounds from the grads in
        `_bnd_gmax`/`_bnd_gmin` + the 0.01 L2 penalty."""
        self._bnd_step += 1
        var beta1 = Scalar[DT](0.9)
        var beta2 = Scalar[DT](0.999)
        var eps = Scalar[DT](1e-8)
        var l2 = Scalar[DT](0.01)
        var sf = Scalar[DT](self._bnd_step)
        var bc1 = Scalar[DT](1.0) - fexp(sf * flog(beta1))
        var bc2 = Scalar[DT](1.0) - fexp(sf * flog(beta2))
        var bo = member_idx * Self.PRED_DIM
        comptime if target == "cpu":
            for d in range(Self.PRED_DIM):
                var g_max = l2
                var g_min = -l2
                for b in range(Self.BATCH):
                    g_max += self._bnd_gmax.data[b * Self.PRED_DIM + d]
                    g_min += self._bnd_gmin.data[b * Self.PRED_DIM + d]
                var i = bo + d
                var m1 = beta1 * self._max_lv_m.data[i] + (
                    Scalar[DT](1.0) - beta1
                ) * g_max
                var v1 = beta2 * self._max_lv_v.data[i] + (
                    Scalar[DT](1.0) - beta2
                ) * g_max * g_max
                self._max_lv_m.data[i] = m1
                self._max_lv_v.data[i] = v1
                self._max_lv.data[i] = self._max_lv.data[i] - self._bnd_lr * (
                    m1 / bc1
                ) / (fsqrt(v1 / bc2) + eps)
                var m2 = beta1 * self._min_lv_m.data[i] + (
                    Scalar[DT](1.0) - beta1
                ) * g_min
                var v2 = beta2 * self._min_lv_v.data[i] + (
                    Scalar[DT](1.0) - beta2
                ) * g_min * g_min
                self._min_lv_m.data[i] = m2
                self._min_lv_v.data[i] = v2
                self._min_lv.data[i] = self._min_lv.data[i] - self._bnd_lr * (
                    m2 / bc1
                ) / (fsqrt(v2 / bc2) + eps)
        else:
            var ctx = self.ctx.value()
            var max_lt = self._max_lv.tt_at["gpu", Self.PRED_DIM](bo)
            var min_lt = self._min_lv.tt_at["gpu", Self.PRED_DIM](bo)
            var mm_lt = self._max_lv_m.tt_at["gpu", Self.PRED_DIM](bo)
            var mv_lt = self._max_lv_v.tt_at["gpu", Self.PRED_DIM](bo)
            var nm_lt = self._min_lv_m.tt_at["gpu", Self.PRED_DIM](bo)
            var nv_lt = self._min_lv_v.tt_at["gpu", Self.PRED_DIM](bo)
            var gmax_lt = self._bnd_gmax.tt["gpu", Self.BATCH, Self.PRED_DIM]()
            var gmin_lt = self._bnd_gmin.tt["gpu", Self.BATCH, Self.PRED_DIM]()
            comptime n_dim_blocks = (Self.PRED_DIM + TPB - 1) // TPB
            comptime bnd_kernel = _dyn_bounds_adam_kernel[
                Self.PRED_DIM, Self.BATCH
            ]
            ctx.enqueue_function[bnd_kernel](
                max_lt, min_lt, mm_lt, mv_lt, nm_lt, nv_lt,
                gmax_lt, gmin_lt,
                l2, self._bnd_lr, beta1, beta2, eps, bc1, bc2,
                grid_dim=n_dim_blocks, block_dim=TPB,
            )

    # ------------------------------------------------------------------
    # Train member — one Gaussian-NLL gradient step.
    # ------------------------------------------------------------------

    def train_member_step[
        target: StaticString, POLICY: AMPPolicy = NoAMP, ACCUMULATE: Bool = False
    ](
        mut self,
        member_idx: Int,
        mut mb_in_t: Tensor,
        mut mb_target_t: Tensor,
    ) raises -> Scalar[DT]:
        """One Gaussian-NLL gradient step on member `member_idx`. Caller owns
        `mb_in_t` (BATCH × IN_DIM) and `mb_target_t` (BATCH × PRED).

        `ACCUMULATE=True` (GPU training) folds the loss into the device
        accumulator and returns 0.0 — no per-step D2H. Read it at flush via
        `read_dyn_loss_accum`. `ACCUMULATE=False` returns the scalar NLL
        (averaged over BATCH), D2H'ing it (CPU is sync-free anyway)."""
        self.opts[member_idx].zero_grad[target, M=Self.DynNet](
            self.members[member_idx], self.ctx
        )
        call_forward[target, Self.BATCH, POLICY=POLICY](
            self.members[member_idx],
            TensorRefs[Self.DynNet.ARITY](mb_in_t), self._mb_pred, self.ctx
        )
        var loss: Scalar[DT]
        if self.learnable_bounds:
            loss = self._nll_grad_learnable[target, ACCUMULATE](
                member_idx, mb_target_t
            )
        else:
            comptime if target == "gpu" and ACCUMULATE:
                # Device-accumulate (no D2H); the grad is recomputed by `vjp`.
                self.loss.forward_accumulate[target, Self.BATCH](
                    self._mb_pred, mb_target_t, self.ctx
                )
                loss = Scalar[DT](0.0)
            else:
                loss = self.loss.forward[target, Self.BATCH](
                    self._mb_pred, mb_target_t, self.ctx
                )
            self.loss.vjp[target, Self.BATCH](
                self._mb_pred, mb_target_t, self._mb_grad, self.ctx
            )
        # member.vjp consumes `_mb_grad` (grad wrt output) → grad-input sink.
        # `_mb_pred` is reused as the discard sink (OUT_DIM >= IN_DIM asserted).
        call_vjp[target, Self.BATCH, POLICY=POLICY](
            self.members[member_idx],
            TensorRefs[Self.DynNet.ARITY](mb_in_t),
            self._mb_grad,
            TensorRefs[Self.DynNet.ARITY](self._mb_pred),
            self.ctx,
        )
        self.opts[member_idx].step[target, M=Self.DynNet](
            self.members[member_idx], self.ctx
        )
        if self.learnable_bounds:
            self._bounds_step[target](member_idx)
        return loss

    def read_dyn_loss_accum[target: StaticString](mut self) raises -> Scalar[DT]:
        """Mean training NLL accumulated since the last reset (the
        `ACCUMULATE=True` device path). Learnable bounds → our own [sum,count]
        accumulator; fixed bounds → the GaussianNLLLoss device accumulator."""
        if self.learnable_bounds:
            comptime if target == "gpu":
                self._dyn_loss_acc.download(self.ctx.value())
                var s = self._dyn_loss_acc.data[0]
                var n = self._dyn_loss_acc.data[1]
                if n == Scalar[DT](0.0):
                    return Scalar[DT](0.0)
                return s / n
            else:
                return Scalar[DT](0.0)
        return self.loss.read_accum[target](self.ctx)

    def reset_dyn_loss_accum[target: StaticString](mut self) raises:
        if self.learnable_bounds:
            comptime if target == "gpu":
                self._dyn_loss_acc.dev.value().enqueue_fill(Scalar[DT](0.0))
            return
        self.loss.reset_accum[target]()

    # ------------------------------------------------------------------
    # Eval member loss — holdout-set scoring (no gradient).
    # ------------------------------------------------------------------

    def eval_member_loss[
        target: StaticString, POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        member_idx: Int,
        mut mb_in_t: Tensor,
        mut mb_target_t: Tensor,
    ) raises -> Scalar[DT]:
        """Holdout-set forward only. Returns the same NLL as
        `train_member_step` would compute, without mutating member weights."""
        call_forward[target, Self.BATCH, POLICY=POLICY](
            self.members[member_idx],
            TensorRefs[Self.DynNet.ARITY](mb_in_t), self._mb_pred, self.ctx
        )
        if self.learnable_bounds:
            return self._nll_grad_learnable[target](member_idx, mb_target_t)
        return self.loss.forward[target, Self.BATCH](
            self._mb_pred, mb_target_t, self.ctx
        )

    def eval_member_mse[
        target: StaticString, POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        member_idx: Int,
        mut mb_in_t: Tensor,
        mut mb_target_t: Tensor,
    ) raises -> Scalar[DT]:
        """Plain MSE of the MEAN head only — the reference's holdout / elite
        metric (`inc_var_loss=False`). Independent of the (learnable) logvar
        bounds: selecting elites by NLL rewards OVER-confidence (a member can
        lower NLL by shrinking variance), so rollouts then sample near-
        deterministic, biased dynamics. MSE selects the most ACCURATE members,
        which is what `update_elites` / holdout early-stop must rank on."""
        call_forward[target, Self.BATCH, POLICY=POLICY](
            self.members[member_idx],
            TensorRefs[Self.DynNet.ARITY](mb_in_t), self._mb_pred, self.ctx
        )
        comptime if target == "cpu":
            var total = Scalar[DT](0.0)
            for b in range(Self.BATCH):
                var po = b * Self.OUT_DIM
                var to = b * Self.PRED_DIM
                var row = Scalar[DT](0.0)
                for d in range(Self.PRED_DIM):
                    var diff = self._mb_pred.data[po + d] - mb_target_t.data[
                        to + d
                    ]
                    row += diff * diff
                total += row / Scalar[DT](Self.PRED_DIM)
            return total / Scalar[DT](Self.BATCH)
        else:
            var ctx = self.ctx.value()
            var pred_lt = self._mb_pred.tt["gpu", Self.BATCH, Self.OUT_DIM]()
            var tgt_lt = mb_target_t.tt["gpu", Self.BATCH, Self.PRED_DIM]()
            var ploss_lt = self._bnd_ploss.tt["gpu", Self.BATCH]()
            comptime n_rows = (Self.BATCH + TPB - 1) // TPB
            comptime mse_kernel = _dyn_mse_only_kernel[
                Self.BATCH, Self.OUT_DIM, Self.PRED_DIM
            ]
            ctx.enqueue_function[mse_kernel](
                pred_lt, tgt_lt, ploss_lt,
                grid_dim=n_rows, block_dim=TPB,
            )
            self._bnd_ploss.download(ctx)
            var total = Scalar[DT](0.0)
            for b in range(Self.BATCH):
                total += self._bnd_ploss.data[b]
            return total / Scalar[DT](Self.BATCH)

    # ------------------------------------------------------------------
    # Elite ranking.
    # ------------------------------------------------------------------

    def update_elites(mut self, mut holdout_losses: List[Scalar[DT]]):
        """Sort members by ascending holdout loss; keep top-NUM_ELITES."""
        var sorted_idx = List[Int]()
        for i in range(Self.N):
            sorted_idx.append(i)
        for i in range(Self.NUM_ELITES):
            var min_pos = i
            for j in range(i + 1, Self.N):
                if (
                    holdout_losses[sorted_idx[j]]
                    < holdout_losses[sorted_idx[min_pos]]
                ):
                    min_pos = j
            var tmp = sorted_idx[i]
            sorted_idx[i] = sorted_idx[min_pos]
            sorted_idx[min_pos] = tmp
        self.elite_indices.clear()
        for i in range(Self.NUM_ELITES):
            self.elite_indices.append(sorted_idx[i])
