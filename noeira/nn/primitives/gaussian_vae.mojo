"""CVAE primitives — the diagonal-Gaussian reparameterization and its KL.

    GaussianReparam[LATENT]      (B, 2L) -> (B, L)    mu + exp(logvar/2) * eps
    GaussianKLStdNormal[LATENT]  (B, 2L) -> (B, 1)    KL( N(mu, sigma) || N(0, I) )

Both take ONE packed `[mu | logvar]` input, which is what
`detr_vae.py:104 latent_proj` produces (`Linear[hidden, 2*latent_dim]`, sliced
into halves) — so the graph needs no split node and the two consumers share one
tensor.

## Reparameterization (`detr_vae.py:16 reparametrize`)

    std = exp(logvar / 2);  z = mu + std * eps,   eps ~ N(0, I) fresh per call

⚠ NOT `RSample`. That leaf is SAC's squashed Gaussian: it applies `tanh`,
scales by an action bound, and returns a log-probability alongside the sample.
A CVAE latent is unsquashed and unbounded, and ACT never needs its density.

`eps` is drawn fresh each forward and CACHED — it is random, so backward cannot
recompute it:

    dz/dmu     = 1
    dz/dlogvar = 0.5 * exp(logvar/2) * eps = 0.5 * (z - mu)

**Inference.** The reference does not sample at test time — it skips the CVAE
encoder entirely and feeds `z = 0` (`detr_vae.py:110`). Here the graph keeps one
shape and a `Scale` node on `z` is set to 0, giving `latent_out_proj(0)` — the
same number by construction. So this leaf has no eval mode: `Scale` is where
that switch lives.

## KL (`policy.py:kl_divergence`)

    klds[b, j]  = -0.5 * (1 + logvar - mu^2 - exp(logvar))
    out[b, 0]   = sum_j klds[b, j]

⚠ **A SUM over the latent dimension, not a mean** — `klds.sum(1).mean(0)`. The
batch mean is the trainer's reduction; a mean here would silently divide the KL
term by `LATENT` (32), so `kl_weight=10` would behave like 0.31 and the latent
would collapse slowly enough to look like ordinary underfitting.

    d/dmu     = mu
    d/dlogvar = 0.5 * (exp(logvar) - 1)
"""

from std.math import exp
from max.gpu import global_idx
from max.gpu.host import DeviceContext
from layout import TileTensor, row_major
from noeira.core.tiles import Tile

from noeira.nn.constants import DT, TPB
from noeira.nn.core.ptr import mptr
from noeira.nn.random.box_muller import (
    advance_rng_offset_kernel,
    box_muller_normal,
    _box_muller_kernel_dev,
)
from ..core.tensor import Tensor, TensorImpl
from ..core.tensor_refs import TensorRefs
from ..core.module import Module
from ..core.param import ParamVisitor
from ..core.initializer import Initializer
from ..core.amp import AMPPolicy, NoAMP


# ══════════════════════════════════════════════════════════════════════════
# GaussianReparam
# ══════════════════════════════════════════════════════════════════════════


def _gr_forward_kernel[
    BATCH: Int, L: Int
](
    x: Tile[DT, BATCH, 2 * L],
    eps: Tile[DT, BATCH, L],
    o: Tile[DT, BATCH, L],
):
    var idx = Int(global_idx.x)
    if idx >= BATCH * L:
        return
    var b = idx // L
    var j = idx % L
    var mu = rebind[Scalar[DT]](x[b, j])
    var lv = rebind[Scalar[DT]](x[b, L + j])
    o[b, j] = mu + exp(lv * Scalar[DT](0.5)) * rebind[Scalar[DT]](eps[b, j])



def _gr_prior_kernel[
    N: Int
](
    eps: Tile[DT, N],
    z: Tile[DT, N],
):
    """z = eps: the prior draw, `GaussianReparam.prior`."""
    var idx = Int(global_idx.x)
    if idx < N:
        z[idx] = rebind[Scalar[DT]](eps[idx])

def _gr_zero_eps_kernel[
    N: Int
](eps: Tile[DT, N]):
    var i = Int(global_idx.x)
    if i < N:
        eps.ptr[unsafe_offset=i] = Scalar[DT](0.0)


def _gr_backward_kernel[
    BATCH: Int, L: Int
](
    go: Tile[DT, BATCH, L],
    x: Tile[DT, BATCH, 2 * L],
    eps: Tile[DT, BATCH, L],
    gi: Tile[DT, BATCH, 2 * L],
):
    var idx = Int(global_idx.x)
    if idx >= BATCH * L:
        return
    var b = idx // L
    var j = idx % L
    var g = rebind[Scalar[DT]](go[b, j])
    var lv = rebind[Scalar[DT]](x[b, L + j])
    gi[b, j] = g  # d/dmu = 1
    gi[b, L + j] = (
        g
        * Scalar[DT](0.5)
        * exp(lv * Scalar[DT](0.5))
        * rebind[Scalar[DT]](eps[b, j])
    )


struct GaussianReparam[LATENT: Int](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = Array[Int, 1](fill=2 * Self.LATENT)
    comptime OUT_DIM: Int = Self.LATENT

    var eps: Tensor
    """[BATCH, LATENT] — the draw used by the last forward. Random, so backward
    cannot recompute it."""
    var noise_seed: UInt64
    var noise_offset: TensorImpl[DType.uint64]
    """GPU Philox offset (1 element, device-resident) — the storage RNG idiom
    `rsample.mojo` uses, so the draw stays CUDA-graph-capture friendly."""
    var prior: Bool
    """When set, `z = eps ~ N(0, I)` regardless of `mu`/`logvar` — a draw
    from the PRIOR, for inference. The reference's rule is `z = 0` (the prior
    MEAN, `detr_vae.py:110`), which on demonstrations whose future timing
    varies across operators returns the flat conditional median (measured
    2026-09-20: one action for all 40 chunk positions). A prior draw commits
    to one style the way a nearest-neighbour chunk commits to one
    demonstration; it is only as good as the KL let the latent stay
    informative."""
    var deterministic: Bool
    """When set, `eps = 0` and `z = mu`. Not the reference's inference path (see
    the module docstring — that is a `Scale` on z); this exists so a gate can
    compare the reparameterization against a reference WITHOUT having to inject
    a matching noise stream."""

    def __init__(out self):
        comptime assert Self.LATENT > 0, "GaussianReparam: LATENT must be > 0"
        self.eps = Tensor()
        self.prior = False
        self.deterministic = False
        self.noise_seed = UInt64(0x5DEECE66D)
        self.noise_offset = TensorImpl[DType.uint64]()

    def __init__(out self, *, deinit move: Self):
        self.eps = move.eps^
        self.prior = move.prior
        self.deterministic = move.deterministic
        self.noise_seed = move.noise_seed
        self.noise_offset = move.noise_offset^

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "GaussianReparam: target must be 'cpu' or 'gpu'"
        )
        var r = Self()
        comptime if target == "gpu":
            var c = ctx.value()
            r.noise_offset.ensure_gpu(c, 1)
            r.noise_offset.dev.value().enqueue_fill(UInt64(0))
        return r^

    def set_attr[ATTR: StaticString](mut self, value: Scalar[DT]):
        """`set_attr["deterministic"](1.0)` -> z = mu (gates only)."""
        comptime if ATTR == "deterministic":
            self.deterministic = value != Scalar[DT](0.0)
        comptime if ATTR == "prior":
            self.prior = value != Scalar[DT](0.0)

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[1, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref x = inputs[0]
        comptime L = Self.LATENT
        comptime if target != "cpu":
            var c = ctx.value()
            out.ensure_gpu(c, B * L)
            self.eps.ensure_gpu(c, B * L)
            comptime NZ = B * L
            comptime nbz = (NZ + TPB - 1) // TPB
            if self.deterministic:
                c.enqueue_function[_gr_zero_eps_kernel[NZ]](
                    self.eps.tt["gpu", NZ](),
                    grid_dim=nbz, block_dim=TPB,
                )
            else:
                c.enqueue_function[_box_muller_kernel_dev[NZ]](
                    self.eps.tt["gpu", NZ](),
                    self.noise_seed,
                    self.noise_offset.tt["gpu", 1](),
                    grid_dim=nbz, block_dim=TPB,
                )
                # Advance by the PAIR-rounded count, matching `rsample` — the
                # Box-Muller kernel consumes two Philox draws per pair.
                c.enqueue_function[
                    advance_rng_offset_kernel[((NZ + 1) // 2) * 2]
                ](
                    self.noise_offset.tt["gpu", 1](),
                    grid_dim=1, block_dim=1,
                )
            if self.prior:
                c.enqueue_function[_gr_prior_kernel[NZ]](
                    self.eps.tt["gpu", NZ](),
                    out.tt["gpu", NZ](),
                    grid_dim=nbz, block_dim=TPB,
                )
                return
            c.enqueue_function[_gr_forward_kernel[B, L]](
                x.tt["gpu", B, 2 * L](),
                self.eps.tt["gpu", B, L](),
                out.tt["gpu", B, L](),
                grid_dim=nbz, block_dim=TPB,
            )
            return

        out.ensure(B * L)
        self.eps.ensure(B * L)
        if self.deterministic:
            for i in range(B * L):
                self.eps.data[i] = Scalar[DT](0.0)
        else:
            box_muller_normal(mptr(self.eps.data), B * L)
        if self.prior:
            for i in range(B * L):
                out.data[i] = self.eps.data[i]
            return
        for b in range(B):
            for j in range(L):
                var e = self.eps.data[b * L + j]
                var mu = x.data[b * 2 * L + j]
                var logvar = x.data[b * 2 * L + L + j]
                out.data[b * L + j] = mu + exp(
                    logvar * Scalar[DT](0.5)
                ) * e

    def vjp[
        target: StaticString,
        B: Int,
        ofi: MutOrigin,
        ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[1, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[1, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        comptime L = Self.LATENT
        ref x = forward_input[0]
        ref gin = grad_inputs[0]
        comptime if target != "cpu":
            var c = ctx.value()
            gin.ensure_gpu(c, B * 2 * L)
            c.enqueue_function[_gr_backward_kernel[B, L]](
                grad_output.tt["gpu", B, L](),
                x.tt["gpu", B, 2 * L](),
                self.eps.tt["gpu", B, L](),
                gin.tt["gpu", B, 2 * L](),
                grid_dim=(B * L + TPB - 1) // TPB, block_dim=TPB,
            )
            return

        gin.ensure(B * 2 * L)
        for b in range(B):
            for j in range(L):
                var g = grad_output.data[b * L + j]
                var logvar = x.data[b * 2 * L + L + j]
                var std = exp(logvar * Scalar[DT](0.5))
                gin.data[b * 2 * L + j] = g  # d/dmu = 1
                gin.data[b * 2 * L + L + j] = (
                    g * Scalar[DT](0.5) * std * self.eps.data[b * L + j]
                )


# ══════════════════════════════════════════════════════════════════════════
# GaussianKLStdNormal
# ══════════════════════════════════════════════════════════════════════════


def _kl_forward_kernel[
    BATCH: Int, L: Int
](
    x: Tile[DT, BATCH, 2 * L],
    o: Tile[DT, BATCH, 1],
):
    var r = Int(global_idx.x)
    if r >= BATCH:
        return
    var s = Scalar[DT](0)
    for j in range(L):
        var mu = rebind[Scalar[DT]](x[r, j])
        var lv = rebind[Scalar[DT]](x[r, L + j])
        s += Scalar[DT](-0.5) * (Scalar[DT](1.0) + lv - mu * mu + -exp(lv))
    o[r, 0] = s


def _kl_backward_kernel[
    BATCH: Int, L: Int
](
    go: Tile[DT, BATCH, 1],
    x: Tile[DT, BATCH, 2 * L],
    gi: Tile[DT, BATCH, 2 * L],
):
    var idx = Int(global_idx.x)
    if idx >= BATCH * L:
        return
    var r = idx // L
    var j = idx % L
    var g = rebind[Scalar[DT]](go[r, 0])
    var mu = rebind[Scalar[DT]](x[r, j])
    var lv = rebind[Scalar[DT]](x[r, L + j])
    gi[r, j] = g * mu
    gi[r, L + j] = g * Scalar[DT](0.5) * (exp(lv) - Scalar[DT](1.0))


struct GaussianKLStdNormal[LATENT: Int](Module):
    comptime ARITY: Int = 1
    comptime IN_DIMS = Array[Int, 1](fill=2 * Self.LATENT)
    comptime OUT_DIM: Int = 1

    def __init__(out self):
        comptime assert Self.LATENT > 0, (
            "GaussianKLStdNormal: LATENT must be > 0"
        )

    def __init__(out self, *, deinit move: Self):
        pass

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "GaussianKLStdNormal: target must be 'cpu' or 'gpu'"
        )
        return Self()

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[1, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref x = inputs[0]
        comptime L = Self.LATENT
        comptime if target == "cpu":
            out.ensure(B)
            for b in range(B):
                var s = Scalar[DT](0)
                for j in range(L):
                    var mu = x.data[b * 2 * L + j]
                    var lv = x.data[b * 2 * L + L + j]
                    s += Scalar[DT](-0.5) * (
                        Scalar[DT](1.0) + lv - mu * mu - exp(lv)
                    )
                out.data[b] = s
        else:
            var c = ctx.value()
            out.ensure_gpu(c, B)
            c.enqueue_function[_kl_forward_kernel[B, L]](
                x.tt["gpu", B, 2 * L](),
                out.tt["gpu", B, 1](),
                grid_dim=(B + TPB - 1) // TPB,
                block_dim=TPB,
            )

    def vjp[
        target: StaticString,
        B: Int,
        ofi: MutOrigin,
        ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[1, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[1, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref x = forward_input[0]
        ref gin = grad_inputs[0]
        comptime L = Self.LATENT
        comptime if target == "cpu":
            gin.ensure(B * 2 * L)
            for b in range(B):
                var g = grad_output.data[b]
                for j in range(L):
                    var mu = x.data[b * 2 * L + j]
                    var lv = x.data[b * 2 * L + L + j]
                    gin.data[b * 2 * L + j] = g * mu
                    gin.data[b * 2 * L + L + j] = (
                        g * Scalar[DT](0.5) * (exp(lv) - Scalar[DT](1.0))
                    )
        else:
            var c = ctx.value()
            gin.ensure_gpu(c, B * 2 * L)
            c.enqueue_function[_kl_backward_kernel[B, L]](
                grad_output.tt["gpu", B, 1](),
                x.tt["gpu", B, 2 * L](),
                gin.tt["gpu", B, 2 * L](),
                grid_dim=(B * L + TPB - 1) // TPB,
                block_dim=TPB,
            )
