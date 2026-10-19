"""Gradient penalty on a scalar-output network — WGAN-GP / R1 — at FIRST order.

    P(θ) = coef · mean_i ( ‖∇ₓ D(x_i; θ)‖ − target )²      target = 1 (WGAN-GP), 0 (R1)

and its parameter gradient, accumulated into `D`'s param grads exactly like a
loss `vjp` would. `docs/BFM_ZERO_SHOT_RL.md` §18.8 — the piece FB-CPR's
discriminator needs (grad-penalty coef 10 on both `obs` and `z`, §16.2) and
the gap §15.6 lists for AMP-style discriminators.

## Why this is not second-order autodiff, and is still exact

`nn` has hand-written, kernel-fused `vjp`s and no tape: nothing records the
backward pass as differentiable ops, so "differentiate the gradient" would
mean a double-backward rule per primitive (83 of them, LayerNorm's is ugly).
It is not needed. Write `g = ∇ₓD` and `ĝ = g / ‖g‖`. Then

    ‖g‖ = ĝ · ∇ₓD = d/dε D(x + ε ĝ) |₀          (directional derivative)

and — this is the step that makes it exact — `∂‖g‖/∂θ = ĝᵀ ∂g/∂θ` with `ĝ`
held CONSTANT, because `∂‖g‖/∂g = ĝ` (envelope). So

    ∂P/∂θ = 2 coef (‖g‖ − t) · ĝᵀ ∂(∇ₓD)/∂θ
          = ∂/∂θ [ coef ( h(θ) − t )² ]     with  h(θ) = ( D(x + εĝ₀; θ) − D(x − εĝ₀; θ) ) / 2ε,  ĝ₀ = stop_grad(ĝ)

up to the central-difference truncation O(ε²) in `h`. The right-hand side is
a plain scalar loss of three ordinary forward passes and two ordinary
backward passes. Every primitive's existing `vjp` is enough.

⚠ ε is in INPUT units along a UNIT direction. The default 1e-2 assumes
O(1) inputs (standardised obs, z on the √d sphere is O(1) per coordinate).
fp32 roundoff in `h` is ~1e-7/ε, truncation ~ε²·D''' — at 1e-2 both sit
around 1e-4 relative, three decades below what a coef-10 regulariser can
feel. Gated by parameter finite differences and a closed form in
`tests/nn/test_grad_penalty.mojo`; if a consumer's inputs are not O(1),
pass its own `eps`.

## ⚠⚠ THE CALL-ORDER CONTRACT

`apply` needs `∇ₓD` first, and the only way to get an input gradient out of
`nn` is a `vjp`, which ALSO accumulates parameter gradients — of `Σ_i D_i`,
which is not part of the penalty. `apply` therefore zeroes the param grads
right after that probe pass and then accumulates the penalty's. So:

    net.zero_grad(...)              # or not — apply zeroes anyway
    gp.apply(net, x, coef, ...)     # FIRST: leaves ONLY the penalty grads
    ... your loss's vjps ...         # THEN the discriminator loss, on top
    opt.step(net)

A caller that runs its loss `vjp` BEFORE `apply` loses that gradient,
silently — the optimizer steps on the penalty alone and the discriminator
never learns. Nothing raises. Keep `apply` first.

## Batch coupling

`∇ₓD_i` is read off ONE `vjp` with a cotangent of ones on `[B, 1]`, which is
the per-sample gradient only if `D_i` depends on `x_i` alone. True for MLPs,
LayerNorm (per row), attention over a row's own tokens; FALSE for BatchNorm
in training mode, where the probe would return a batch-coupled gradient.
Do not put a BatchNorm in a discriminator regularised this way.
"""

from max.gpu import global_idx
from std.math import sqrt
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT, TPB
from noeira.nn.core.module import Module
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.call import call_forward, call_vjp


# ══════════════════════════════════════════════════════════════════════
# Kernels
# ══════════════════════════════════════════════════════════════════════


def _row_unit_kernel[B: Int, IN: Int](
    g: Pointer[Scalar[DT], MutAnyOrigin],
    ghat: Pointer[Scalar[DT], MutAnyOrigin],
    gnorm: Pointer[Scalar[DT], MutAnyOrigin],
):
    """`ghat[i] = g[i] / ‖g[i]‖`, `gnorm[i] = ‖g[i]‖`; a zero row stays zero
    (then `h = 0` and the θ-gradient cancels exactly, which is the right
    answer where ‖·‖ is not differentiable)."""
    var i = Int(global_idx.x)
    if i >= B:
        return
    var acc: Scalar[DT] = 0.0
    for k in range(IN):
        var v = g[unsafe_offset=i * IN + k]
        acc += v * v
    var n = sqrt(acc)
    gnorm[unsafe_offset=i] = n
    var s = Scalar[DT](0.0) if n < Scalar[DT](1e-12) else Scalar[DT](1.0) / n
    for k in range(IN):
        ghat[unsafe_offset=i * IN + k] = g[unsafe_offset=i * IN + k] * s


def _shift_kernel[N: Int](
    x: Pointer[Scalar[DT], MutAnyOrigin],
    ghat: Pointer[Scalar[DT], MutAnyOrigin],
    xp: Pointer[Scalar[DT], MutAnyOrigin],
    xm: Pointer[Scalar[DT], MutAnyOrigin],
    eps: Scalar[DT],
):
    var t = Int(global_idx.x)
    if t >= N:
        return
    var d = eps * ghat[unsafe_offset=t]
    xp[unsafe_offset=t] = x[unsafe_offset=t] + d
    xm[unsafe_offset=t] = x[unsafe_offset=t] - d


def _cot_kernel[B: Int](
    dp: Pointer[Scalar[DT], MutAnyOrigin],
    dm: Pointer[Scalar[DT], MutAnyOrigin],
    cp: Pointer[Scalar[DT], MutAnyOrigin],
    cm: Pointer[Scalar[DT], MutAnyOrigin],
    sq: Pointer[Scalar[DT], MutAnyOrigin],
    h_out: Pointer[Scalar[DT], MutAnyOrigin],
    inv_2eps: Scalar[DT],
    target: Scalar[DT],
    scale: Scalar[DT],
):
    """`h = (D₊ − D₋)/2ε`; cotangents `±scale·2(h − t)/2ε` with
    `scale = coef / B`; `sq = (h − t)²` for the reported value."""
    var i = Int(global_idx.x)
    if i >= B:
        return
    var h = (dp[unsafe_offset=i] - dm[unsafe_offset=i]) * inv_2eps
    var r = h - target
    var c = scale * Scalar[DT](2.0) * r * inv_2eps
    cp[unsafe_offset=i] = c
    cm[unsafe_offset=i] = -c
    sq[unsafe_offset=i] = r * r
    h_out[unsafe_offset=i] = h


def _ensure_t[target: StaticString](
    mut t: Tensor, n: Int, ctx: Optional[DeviceContext]
) raises:
    """Host storage always (the diagnostics read `.data`); device too on GPU.
    A free function: a method taking `mut self` cannot receive `self.field`."""
    t.ensure(n)
    comptime if target == "gpu":
        t.ensure_gpu(ctx.value(), n)


# ══════════════════════════════════════════════════════════════════════
# The penalty
# ══════════════════════════════════════════════════════════════════════


struct GradPenalty[IN: Int, B: Int](Movable & Deinitable):
    """Owned scratch for one input shape, sized once. `eps` and `target` are
    fields so a consumer sets them at construction and every `apply` agrees.
    """

    var eps: Float64
    var target: Float64
    var ctx: Optional[DeviceContext]
    # probe pass
    var d0: Tensor        # [B, 1]  D(x)
    var ones: Tensor      # [B, 1]  cotangent for the probe vjp
    var g: Tensor         # [B, IN] ∇ₓD
    var ghat: Tensor      # [B, IN]
    var gnorm: Tensor     # [B]     ‖∇ₓD_i‖ — the value the penalty is ON
    # shifted passes
    var xp: Tensor
    var xm: Tensor
    var dp: Tensor
    var dm: Tensor
    var cp: Tensor
    var cm: Tensor
    var sq: Tensor
    var h: Tensor
    var sink: Tensor      # grad-input sink for the shifted vjps
    var _sized: Bool

    def __init__(out self):
        self.eps = 1e-2
        self.target = 1.0
        self.ctx = None
        self.d0 = Tensor()
        self.ones = Tensor()
        self.g = Tensor()
        self.ghat = Tensor()
        self.gnorm = Tensor()
        self.xp = Tensor()
        self.xm = Tensor()
        self.dp = Tensor()
        self.dm = Tensor()
        self.cp = Tensor()
        self.cm = Tensor()
        self.sq = Tensor()
        self.h = Tensor()
        self.sink = Tensor()
        self._sized = False

    @staticmethod
    def make[
        target: StaticString
    ](
        ctx: Optional[DeviceContext] = None,
        eps: Float64 = 1e-2,
        penalty_target: Float64 = 1.0,
    ) raises -> Self:
        """`penalty_target` 1.0 is WGAN-GP's (‖∇‖ − 1)²; 0.0 is the
        zero-centred / R1 form ‖∇‖². `ctx` required on GPU."""
        comptime assert target == "cpu" or target == "gpu", (
            "GradPenalty: target must be 'cpu' or 'gpu'"
        )
        if target == "gpu" and not ctx:
            raise Error("GradPenalty.make[gpu]: ctx required")
        if eps <= 0.0:
            raise Error("GradPenalty.make: eps must be > 0")
        var s = Self()
        s.ctx = ctx
        s.eps = eps
        s.target = penalty_target
        return s^

    def _size_once[target: StaticString](mut self) raises:
        if self._sized:
            return
        comptime N = Self.B * Self.IN
        _ensure_t[target](self.d0, Self.B, self.ctx)
        _ensure_t[target](self.ones, Self.B, self.ctx)
        _ensure_t[target](self.g, N, self.ctx)
        _ensure_t[target](self.ghat, N, self.ctx)
        _ensure_t[target](self.gnorm, Self.B, self.ctx)
        _ensure_t[target](self.xp, N, self.ctx)
        _ensure_t[target](self.xm, N, self.ctx)
        _ensure_t[target](self.dp, Self.B, self.ctx)
        _ensure_t[target](self.dm, Self.B, self.ctx)
        _ensure_t[target](self.cp, Self.B, self.ctx)
        _ensure_t[target](self.cm, Self.B, self.ctx)
        _ensure_t[target](self.sq, Self.B, self.ctx)
        _ensure_t[target](self.h, Self.B, self.ctx)
        _ensure_t[target](self.sink, N, self.ctx)
        for i in range(Self.B):
            self.ones.data[i] = Scalar[DT](1.0)
        comptime if target == "gpu":
            self.ones.upload_resident(self.ctx.value())
        self._sized = True

    def apply[
        target: StaticString, M: Module
    ](
        mut self, mut net: M, mut x: Tensor, coef: Float64,
        want_loss: Bool = True,
    ) raises -> Float64:
        """Accumulate `coef · ∂P/∂θ` into `net`'s param grads. ⚠ FIRST
        contributor — see the module docstring: the probe vjp's param grads
        are zeroed here, so anything accumulated before this call is lost.

        Returns `coef · mean_i (‖∇ₓD_i‖_fd − target)²` when `want_loss`
        (one D2H on GPU), else 0. `gnorm` holds the exact per-row ‖∇ₓD_i‖
        from the probe pass on both paths, `h` the finite-difference one —
        their agreement is the gate's check [1].
        """
        comptime assert M.ARITY == 1, "GradPenalty: the network must take ONE input"
        comptime assert M.OUT_DIM == 1, "GradPenalty: the network must output ONE scalar per row"
        comptime assert M.IN_DIMS[0] == Self.IN, "GradPenalty: IN must match the network's input width"
        comptime N = Self.B * Self.IN
        self._size_once[target]()
        var c = self.ctx

        # ── 1. probe: g = ∇ₓD, via the module's own vjp with cotangent 1 ───
        call_forward[target, Self.B](net, TensorRefs[1, MutAnyOrigin](x), self.d0, c)
        call_vjp[target, Self.B](
            net, TensorRefs[1, MutAnyOrigin](x), self.ones,
            TensorRefs[1, MutAnyOrigin](self.g), c,
        )
        # The probe accumulated ∂(ΣD)/∂θ — not part of the penalty. Drop it.
        net.zero_grad[target](c)

        # ── 2. unit direction, shifted inputs ────────────────────────────
        comptime if target == "cpu":
            for i in range(Self.B):
                var acc = Float64(0)
                for k in range(Self.IN):
                    var v = Float64(self.g.data[i * Self.IN + k])
                    acc += v * v
                var n = sqrt(acc)
                self.gnorm.data[i] = Scalar[DT](n)
                var s = 0.0 if n < 1e-12 else 1.0 / n
                for k in range(Self.IN):
                    self.ghat.data[i * Self.IN + k] = Scalar[DT](
                        Float64(self.g.data[i * Self.IN + k]) * s
                    )
            for t in range(N):
                var d = Scalar[DT](self.eps) * self.ghat.data[t]
                self.xp.data[t] = x.data[t] + d
                self.xm.data[t] = x.data[t] - d
        else:
            var d = c.value()
            d.enqueue_function[_row_unit_kernel[Self.B, Self.IN]](
                self.g.dev.value().unsafe_ptr(),
                self.ghat.dev.value().unsafe_ptr(),
                self.gnorm.dev.value().unsafe_ptr(),
                grid_dim=(Self.B + TPB - 1) // TPB, block_dim=TPB,
            )
            d.enqueue_function[_shift_kernel[N]](
                x.dev.value().unsafe_ptr(),
                self.ghat.dev.value().unsafe_ptr(),
                self.xp.dev.value().unsafe_ptr(),
                self.xm.dev.value().unsafe_ptr(),
                Scalar[DT](self.eps),
                grid_dim=(N + TPB - 1) // TPB, block_dim=TPB,
            )

        # ── 3. D(x+εĝ), D(x−εĝ); cotangents from their difference ─────────
        # ⚠ Order matters: a Sequential caches the LAST forward's activations
        # for its vjp, so each shifted vjp must follow its own forward.
        call_forward[target, Self.B](net, TensorRefs[1, MutAnyOrigin](self.xp), self.dp, c)
        call_forward[target, Self.B](net, TensorRefs[1, MutAnyOrigin](self.xm), self.dm, c)
        var inv_2eps = Scalar[DT](1.0 / (2.0 * self.eps))
        var scale = Scalar[DT](coef / Float64(Self.B))
        comptime if target == "cpu":
            for i in range(Self.B):
                var h = (self.dp.data[i] - self.dm.data[i]) * inv_2eps
                var r = h - Scalar[DT](self.target)
                var cc = scale * Scalar[DT](2.0) * r * inv_2eps
                self.cp.data[i] = cc
                self.cm.data[i] = -cc
                self.sq.data[i] = r * r
                self.h.data[i] = h
        else:
            var d = c.value()
            d.enqueue_function[_cot_kernel[Self.B]](
                self.dp.dev.value().unsafe_ptr(), self.dm.dev.value().unsafe_ptr(),
                self.cp.dev.value().unsafe_ptr(), self.cm.dev.value().unsafe_ptr(),
                self.sq.dev.value().unsafe_ptr(), self.h.dev.value().unsafe_ptr(),
                inv_2eps, Scalar[DT](self.target), scale,
                grid_dim=(Self.B + TPB - 1) // TPB, block_dim=TPB,
            )

        # ── 4. the two backward passes: penalty grads accumulate into θ ───
        # `xm` was forwarded last: its vjp first. Then re-forward `xp`.
        call_vjp[target, Self.B](
            net, TensorRefs[1, MutAnyOrigin](self.xm), self.cm,
            TensorRefs[1, MutAnyOrigin](self.sink), c,
        )
        call_forward[target, Self.B](net, TensorRefs[1, MutAnyOrigin](self.xp), self.dp, c)
        call_vjp[target, Self.B](
            net, TensorRefs[1, MutAnyOrigin](self.xp), self.cp,
            TensorRefs[1, MutAnyOrigin](self.sink), c,
        )

        if not want_loss:
            return 0.0
        comptime if target == "gpu":
            self.sq.download(c.value())
        var s = Float64(0)
        for i in range(Self.B):
            s += Float64(self.sq.data[i])
        return coef * s / Float64(Self.B)

    def read_norms[target: StaticString](mut self) raises:
        """D2H `gnorm` (exact ‖∇ₓD_i‖) and `h` (its finite-difference twin)
        so a caller can compare them. Diagnostics only."""
        comptime if target == "gpu":
            self.gnorm.download(self.ctx.value())
            self.h.download(self.ctx.value())
