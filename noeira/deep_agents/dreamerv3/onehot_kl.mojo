"""OneHotKL[STOCH, CLASSES] — DreamerV3 dynamics/representation KL.

Ports `references/dreamerv3-main/dreamerv3/rssm.py:RSSM.loss` together
with the categorical KL of `embodied/jax/outs.py:Categorical(unimix)` and
the `Agg(sum over STOCH)` wrapper:

    p   = unimix(softmax(logits))   where  unimix(p) = (1-u)·p + u/C
    KL(self‖other)_s = Σ_c p_self[s,c]·(log p_self[s,c] − log p_other[s,c])
    dyn = max(Σ_s KL(sg(post)‖prior)_s, free_nats)     # grad → prior only
    rep = max(Σ_s KL(post‖sg(prior))_s, free_nats)     # grad → post  only

Forward `dyn` and `rep` are numerically identical (stop-gradient is the
identity in forward); they differ only in which side the gradient flows
to. The free-nats `max` gates the gradient: when `Σ_s KL < free_nats` the
loss is the constant `free_nats` and the whole row's gradient is zero.

`grad_prior` receives ONLY the dyn contribution (rep stop-grads the prior);
`grad_post` receives ONLY the rep contribution (dyn stop-grads the post).
Both are WRITTEN (`=`), not accumulated — the caller (RSSM loss graph)
owns any cross-path accumulation.

STORAGE migration: `OneHotKLLoss` is now a plain storage `Module` (forward
over `TensorRefs[2]` → `Tensor`, vjp recomputed from `forward_input`, no
cached pointers). The op is ARITY-2 with OUT_DIM=2 — output `[B,2]` =
`[dyn, rep]`. The asymmetric stop-gradient lives in `vjp`: the `[B,2]`
`grad_output` carries the dyn cotangent in channel 0 (→ grad_prior) and the
rep cotangent in channel 1 (→ grad_post); both softmaxes are RECOMPUTED
from `forward_input` (the same logits the forward saw). The GPU kernels
carry the same math (just `def`, with storage `Tensor.lt` plumbing). The
standalone pointer-based `OneHotKL` struct is kept for the PR-2 fixture
test (`test_dreamer_pr2.mojo`).
"""

from std.math import exp, log
from max.gpu import global_idx
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT, TPB
from noeira.nn.core.tensor import Tensor
from noeira.nn.core.tensor_refs import TensorRefs
from noeira.nn.core.module import Module
from noeira.nn.core.initializer import Initializer
from noeira.nn.core.amp import AMPPolicy, NoAMP
from noeira.core.tiles import Tile


@always_inline
def _mptr[
    o: Origin
](p: Pointer[Scalar[DT], o]) -> Pointer[Scalar[DT], MutAnyOrigin]:
    """Erase a CPU pointer's origin (used only by the standalone `OneHotKL`)."""
    return rebind[Pointer[Scalar[DT], MutAnyOrigin]](p)


# ── OneHotKLLoss GPU kernels. NG=B·STOCH groups of CLASSES lanes; NN=B·SC.
#    fwd1 (per group): softmax+unimix(post/prior) → caches + per-group KL.
#    fwd2 (per batch): Σ_s KL → free-nats clamp → out[B,2] + active[B].
#    bwd (per group): asymmetric dyn→prior / rep→post grads.  ────────────
def _kl_fwd1_kernel[NG: Int, C: Int](
    post: Tile[DT, NG * C],
    prior: Tile[DT, NG * C],
    smpo: Tile[DT, NG * C],
    ppo: Tile[DT, NG * C],
    smpr: Tile[DT, NG * C],
    ppr: Tile[DT, NG * C],
    klpart: Tile[DT, NG],
    u: Scalar[DT],
):
    var g = Int(global_idx.x)
    if g < NG:
        var base = g * C
        var uni = u / Scalar[DT](C)
        var omu = Scalar[DT](1.0) - u
        # post softmax + mix
        var zmax = rebind[Scalar[DT]](post[base])
        for c in range(1, C):
            var v = rebind[Scalar[DT]](post[base + c])
            if v > zmax:
                zmax = v
        var ss: Scalar[DT] = 0.0
        for c in range(C):
            var e = exp(rebind[Scalar[DT]](post[base + c]) - zmax)
            smpo[base + c] = e
            ss += e
        var inv = Scalar[DT](1.0) / ss
        for c in range(C):
            var sm = rebind[Scalar[DT]](smpo[base + c]) * inv
            smpo[base + c] = sm
            ppo[base + c] = omu * sm + uni
        # prior softmax + mix
        var zmx2 = rebind[Scalar[DT]](prior[base])
        for c in range(1, C):
            var v = rebind[Scalar[DT]](prior[base + c])
            if v > zmx2:
                zmx2 = v
        var ss2: Scalar[DT] = 0.0
        for c in range(C):
            var e = exp(rebind[Scalar[DT]](prior[base + c]) - zmx2)
            smpr[base + c] = e
            ss2 += e
        var inv2 = Scalar[DT](1.0) / ss2
        for c in range(C):
            var sm = rebind[Scalar[DT]](smpr[base + c]) * inv2
            smpr[base + c] = sm
            ppr[base + c] = omu * sm + uni
        # group KL
        var kl: Scalar[DT] = 0.0
        for c in range(C):
            var pp = rebind[Scalar[DT]](ppo[base + c])
            kl += pp * (log(pp) - log(rebind[Scalar[DT]](ppr[base + c])))
        klpart[g] = kl


def _kl_fwd2_kernel[B: Int, STOCH: Int](
    klpart: Tile[DT, B * STOCH],
    outb: Tile[DT, B * 2],
    active: Tile[DT, B],
    free_nats: Scalar[DT],
):
    var b = Int(global_idx.x)
    if b < B:
        var ksum: Scalar[DT] = 0.0
        for s in range(STOCH):
            ksum += rebind[Scalar[DT]](klpart[b * STOCH + s])
        var clamped = ksum if ksum > free_nats else free_nats
        outb[b * 2] = clamped
        outb[b * 2 + 1] = clamped
        active[b] = Scalar[DT](1.0) if ksum > free_nats else Scalar[DT](0.0)


def _kl_bwd_kernel[NG: Int, C: Int, STOCH: Int](
    go: Tile[DT, NG // STOCH * 2],
    active: Tile[DT, NG // STOCH],
    smpo: Tile[DT, NG * C],
    ppo: Tile[DT, NG * C],
    smpr: Tile[DT, NG * C],
    ppr: Tile[DT, NG * C],
    g_post: Tile[DT, NG * C],
    g_prior: Tile[DT, NG * C],
    one_m_u: Scalar[DT],
):
    var g = Int(global_idx.x)
    if g < NG:
        var b = g // STOCH
        var base = g * C
        var act = rebind[Scalar[DT]](active[b])
        var wd = rebind[Scalar[DT]](go[b * 2]) * act
        var wr = rebind[Scalar[DT]](go[b * 2 + 1]) * act
        var S: Scalar[DT] = 0.0
        var W: Scalar[DT] = 0.0
        for c in range(C):
            S += (rebind[Scalar[DT]](ppo[base + c]) / rebind[Scalar[DT]](ppr[base + c])) * rebind[Scalar[DT]](smpr[base + c])
            W += rebind[Scalar[DT]](smpo[base + c]) * (
                log(rebind[Scalar[DT]](ppo[base + c])) - log(rebind[Scalar[DT]](ppr[base + c]))
            )
        for j in range(C):
            var a_over_p = rebind[Scalar[DT]](ppo[base + j]) / rebind[Scalar[DT]](ppr[base + j])
            g_prior[base + j] = wd * (-one_m_u * rebind[Scalar[DT]](smpr[base + j]) * (a_over_p - S))
            var lr_diff = log(rebind[Scalar[DT]](ppo[base + j])) - log(rebind[Scalar[DT]](ppr[base + j]))
            g_post[base + j] = wr * (one_m_u * rebind[Scalar[DT]](smpo[base + j]) * (lr_diff - W))


struct OneHotKL[STOCH: Int, CLASSES: Int](Movable & Deinitable):
    comptime GROUP = Self.STOCH * Self.CLASSES

    var unimix: Scalar[DT]
    var free_nats: Scalar[DT]

    # Caches (CPU), lazily grown to BATCH·GROUP / BATCH.
    var sm_post: List[Scalar[DT]]    # pre-mix softmax(post)
    var sm_prior: List[Scalar[DT]]
    var p_post: List[Scalar[DT]]     # unimix-mixed probs
    var p_prior: List[Scalar[DT]]
    var active: List[Scalar[DT]]     # [BATCH] 1.0 if not clamped
    var cache_n: Int

    def __init__(out self):
        self.unimix = Scalar[DT](0.01)
        self.free_nats = Scalar[DT](1.0)
        self.sm_post = List[Scalar[DT]]()
        self.sm_prior = List[Scalar[DT]]()
        self.p_post = List[Scalar[DT]]()
        self.p_prior = List[Scalar[DT]]()
        self.active = List[Scalar[DT]]()
        self.cache_n = 0

    @staticmethod
    def make(
        unimix: Scalar[DT] = Scalar[DT](0.01),
        free_nats: Scalar[DT] = Scalar[DT](1.0),
    ) -> Self:
        var k = Self()
        k.unimix = unimix
        k.free_nats = free_nats
        return k^

    def _ensure_cache(mut self, batch: Int):
        if self.cache_n < batch:
            self.sm_post = List[Scalar[DT]](
                length=batch * Self.GROUP, fill=Scalar[DT](0.0)
            )
            self.sm_prior = List[Scalar[DT]](
                length=batch * Self.GROUP, fill=Scalar[DT](0.0)
            )
            self.p_post = List[Scalar[DT]](
                length=batch * Self.GROUP, fill=Scalar[DT](0.0)
            )
            self.p_prior = List[Scalar[DT]](
                length=batch * Self.GROUP, fill=Scalar[DT](0.0)
            )
            self.active = List[Scalar[DT]](length=batch, fill=Scalar[DT](0.0))
            self.cache_n = batch

    @staticmethod
    def _softmax_mix[
        z_o: Origin[mut=True],
        sm_out_o: Origin[mut=True],
        p_out_o: Origin[mut=True],
    ](
        z: Pointer[Scalar[DT], z_o],
        base: Int,
        u: Scalar[DT],
        mut sm_out: Pointer[Scalar[DT], sm_out_o],
        mut p_out: Pointer[Scalar[DT], p_out_o],
    ):
        """softmax + unimix mix for one (b,s) group of CLASSES lanes."""
        var zmax = z[unsafe_offset=base]
        for c in range(1, Self.CLASSES):
            if z[unsafe_offset=base + c] > zmax:
                zmax = z[unsafe_offset=base + c]
        var ssum: Scalar[DT] = 0.0
        for c in range(Self.CLASSES):
            var e = exp(z[unsafe_offset=base + c] - zmax)
            sm_out[unsafe_offset=base + c] = e
            ssum += e
        var inv = Scalar[DT](1.0) / ssum
        var uni = u / Scalar[DT](Self.CLASSES)
        var one_m_u = Scalar[DT](1.0) - u
        for c in range(Self.CLASSES):
            var sm = sm_out[unsafe_offset=base + c] * inv
            sm_out[unsafe_offset=base + c] = sm
            p_out[unsafe_offset=base + c] = one_m_u * sm + uni

    def forward[
        BATCH: Int,
        post_logits_o: Origin[mut=True],
        prior_logits_o: Origin[mut=True],
        dyn_out_o: Origin[mut=True],
        rep_out_o: Origin[mut=True],
    ](
        mut self,
        post_logits: Pointer[Scalar[DT], post_logits_o],
        prior_logits: Pointer[Scalar[DT], prior_logits_o],
        mut dyn_out: Pointer[Scalar[DT], dyn_out_o],
        mut rep_out: Pointer[Scalar[DT], rep_out_o],
    ) raises:
        self._ensure_cache(BATCH)
        var smpo = _mptr(self.sm_post.unsafe_ptr())
        var smpr = _mptr(self.sm_prior.unsafe_ptr())
        var ppo = _mptr(self.p_post.unsafe_ptr())
        var ppr = _mptr(self.p_prior.unsafe_ptr())
        for b in range(BATCH):
            var kl_sum: Scalar[DT] = 0.0
            for s in range(Self.STOCH):
                var base = (b * Self.STOCH + s) * Self.CLASSES
                Self._softmax_mix(post_logits, base, self.unimix, smpo, ppo)
                Self._softmax_mix(prior_logits, base, self.unimix, smpr, ppr)
                for c in range(Self.CLASSES):
                    var pp = ppo[unsafe_offset=base + c]
                    kl_sum += pp * (log(pp) - log(ppr[unsafe_offset=base + c]))
            var clamped = kl_sum if kl_sum > self.free_nats else self.free_nats
            dyn_out[unsafe_offset=b] = clamped
            rep_out[unsafe_offset=b] = clamped
            self.active[b] = (
                Scalar[DT](1.0) if kl_sum > self.free_nats else Scalar[DT](0.0)
            )

    def backward[
        BATCH: Int,
        d_dyn_o: Origin[mut=True],
        d_rep_o: Origin[mut=True],
        grad_post_o: Origin[mut=True],
        grad_prior_o: Origin[mut=True],
    ](
        mut self,
        d_dyn: Pointer[Scalar[DT], d_dyn_o],
        d_rep: Pointer[Scalar[DT], d_rep_o],
        mut grad_post: Pointer[Scalar[DT], grad_post_o],
        mut grad_prior: Pointer[Scalar[DT], grad_prior_o],
    ) raises:
        var smpo = _mptr(self.sm_post.unsafe_ptr())
        var smpr = _mptr(self.sm_prior.unsafe_ptr())
        var ppo = _mptr(self.p_post.unsafe_ptr())
        var ppr = _mptr(self.p_prior.unsafe_ptr())
        var one_m_u = Scalar[DT](1.0) - self.unimix
        for b in range(BATCH):
            var act = self.active[b]
            var wd = d_dyn[unsafe_offset=b] * act
            var wr = d_rep[unsafe_offset=b] * act
            for s in range(Self.STOCH):
                var base = (b * Self.STOCH + s) * Self.CLASSES
                # ── dyn → grad_prior ──
                # S = Σ_c (p_post[c]/p_prior[c])·sm_prior[c]
                var S: Scalar[DT] = 0.0
                for c in range(Self.CLASSES):
                    S += (ppo[unsafe_offset=base + c] / ppr[unsafe_offset=base + c]) * smpr[unsafe_offset=base + c]
                # ── rep → grad_post ──
                # W = Σ_c sm_post[c]·(log p_post[c] − log p_prior[c])
                var W: Scalar[DT] = 0.0
                for c in range(Self.CLASSES):
                    W += smpo[unsafe_offset=base + c] * (
                        log(ppo[unsafe_offset=base + c]) - log(ppr[unsafe_offset=base + c])
                    )
                for j in range(Self.CLASSES):
                    # prior grad (dyn term).
                    var a_over_p = ppo[unsafe_offset=base + j] / ppr[unsafe_offset=base + j]
                    var gpr = -one_m_u * smpr[unsafe_offset=base + j] * (a_over_p - S)
                    grad_prior[unsafe_offset=base + j] = wd * gpr
                    # post grad (rep term).
                    var lr_diff = log(ppo[unsafe_offset=base + j]) - log(ppr[unsafe_offset=base + j])
                    var gpo = one_m_u * smpo[unsafe_offset=base + j] * (lr_diff - W)
                    grad_post[unsafe_offset=base + j] = wr * gpo


# ──────────────────────────────────────────────────────────────────────
# OneHotKLLoss — OneHotKL wrapped as a storage graph Module (ARITY=2).
#   inputs: post[B, S·C], prior[B, S·C]   output: [B, 2] = [dyn, rep]
# The asymmetric sg (dyn→prior, rep→post) lives INSIDE the op's vjp; the
# ComputeGraph just routes the two output cotangents back to the two inputs
# and accumulates them with everything else (e.g. at the shared new_deter).
# Storage migration: no cached pointers — vjp RECOMPUTES both softmaxes from
# `forward_input` (the same logits the forward saw). The free-bits clamp is
# also recomputed (its `active` gate is re-derived from the recomputed KL).
# ──────────────────────────────────────────────────────────────────────


struct OneHotKLLoss[STOCH: Int, CLASSES: Int](Module):
    comptime SC = Self.STOCH * Self.CLASSES
    comptime ARITY: Int = 2
    comptime IN_DIMS = Array[Int, 2](fill=Self.SC)
    comptime OUT_DIM = 2

    @staticmethod
    def display_label() -> String:
        return String("OneHotKL")

    var unimix: Scalar[DT]
    var free_nats: Scalar[DT]

    def __init__(out self):
        self.unimix = Scalar[DT](0.01)
        self.free_nats = Scalar[DT](1.0)

    @staticmethod
    def make[
        target: StaticString, INIT: Initializer
    ](ctx: Optional[DeviceContext] = None) raises -> Self:
        comptime assert target == "cpu" or target == "gpu", (
            "OneHotKLLoss: target must be 'cpu' or 'gpu'"
        )
        var m = Self()
        m.unimix = Scalar[DT](0.01)
        m.free_nats = Scalar[DT](1.0)
        return m^

    # ── shared CPU softmax+unimix mix for one (b,s) group of CLASSES lanes ──
    @always_inline
    def _softmax_mix_cpu(
        self,
        z: List[Scalar[DT]],
        base: Int,
        mut sm_out: List[Scalar[DT]],
        mut p_out: List[Scalar[DT]],
    ):
        var u = self.unimix
        var zmax = z[base]
        for c in range(1, Self.CLASSES):
            if z[base + c] > zmax:
                zmax = z[base + c]
        var ssum: Scalar[DT] = 0.0
        for c in range(Self.CLASSES):
            var e = exp(z[base + c] - zmax)
            sm_out[base + c] = e
            ssum += e
        var inv = Scalar[DT](1.0) / ssum
        var uni = u / Scalar[DT](Self.CLASSES)
        var one_m_u = Scalar[DT](1.0) - u
        for c in range(Self.CLASSES):
            var sm = sm_out[base + c] * inv
            sm_out[base + c] = sm
            p_out[base + c] = one_m_u * sm + uni

    def forward[
        target: StaticString, B: Int, o: MutOrigin, POLICY: AMPPolicy = NoAMP
    ](
        mut self,
        inputs: TensorRefs[Self.ARITY, o],
        mut out: Tensor,
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref post = inputs[0]
        ref prior = inputs[1]
        comptime if target == "cpu":
            out.ensure(B * 2)
            # scratch softmaxes for one group reused across (b,s) groups
            var smpo = List[Scalar[DT]](length=B * Self.SC, fill=Scalar[DT](0))
            var ppo = List[Scalar[DT]](length=B * Self.SC, fill=Scalar[DT](0))
            var smpr = List[Scalar[DT]](length=B * Self.SC, fill=Scalar[DT](0))
            var ppr = List[Scalar[DT]](length=B * Self.SC, fill=Scalar[DT](0))
            for b in range(B):
                var kl_sum: Scalar[DT] = 0.0
                for s in range(Self.STOCH):
                    var base = (b * Self.STOCH + s) * Self.CLASSES
                    self._softmax_mix_cpu(post.data, base, smpo, ppo)
                    self._softmax_mix_cpu(prior.data, base, smpr, ppr)
                    for c in range(Self.CLASSES):
                        var pp = ppo[base + c]
                        kl_sum += pp * (log(pp) - log(ppr[base + c]))
                var clamped = (
                    kl_sum if kl_sum > self.free_nats else self.free_nats
                )
                out.data[b * 2] = clamped
                out.data[b * 2 + 1] = clamped
        else:
            var c = ctx.value()
            comptime NN = B * Self.SC
            comptime NG = B * Self.STOCH
            out.ensure_gpu(c, B * 2)
            # transient device scratch (recomputed again in vjp, so not cached)
            var smpo = Tensor.alloc_gpu(c, NN)
            var ppo = Tensor.alloc_gpu(c, NN)
            var smpr = Tensor.alloc_gpu(c, NN)
            var ppr = Tensor.alloc_gpu(c, NN)
            var klp = Tensor.alloc_gpu(c, NG)
            var act = Tensor.alloc_gpu(c, B)
            comptime nb1 = (NG + TPB - 1) // TPB
            c.enqueue_function[_kl_fwd1_kernel[NG, Self.CLASSES]](
                post.tt["gpu", NN](),
                prior.tt["gpu", NN](),
                smpo.tt["gpu", NN](),
                ppo.tt["gpu", NN](),
                smpr.tt["gpu", NN](),
                ppr.tt["gpu", NN](),
                klp.tt["gpu", NG](),
                self.unimix,
                grid_dim=nb1, block_dim=TPB,
            )
            comptime nb2 = (B + TPB - 1) // TPB
            c.enqueue_function[_kl_fwd2_kernel[B, Self.STOCH]](
                klp.tt["gpu", NG](),
                out.tt["gpu", B * 2](),
                act.tt["gpu", B](),
                self.free_nats,
                grid_dim=nb2, block_dim=TPB,
            )

    def vjp[
        target: StaticString, B: Int, ofi: MutOrigin, ogi: MutOrigin,
        POLICY: AMPPolicy = NoAMP,
    ](
        mut self,
        forward_input: TensorRefs[Self.ARITY, ofi],
        mut grad_output: Tensor,
        grad_inputs: TensorRefs[Self.ARITY, ogi],
        ctx: Optional[DeviceContext] = None,
    ) raises:
        ref post = forward_input[0]
        ref prior = forward_input[1]
        ref g_post = grad_inputs[0]
        ref g_prior = grad_inputs[1]
        var one_m_u = Scalar[DT](1.0) - self.unimix
        comptime if target == "cpu":
            g_post.ensure(B * Self.SC)
            g_prior.ensure(B * Self.SC)
            # RECOMPUTE both softmaxes from forward_input (identical to fwd).
            var smpo = List[Scalar[DT]](length=B * Self.SC, fill=Scalar[DT](0))
            var ppo = List[Scalar[DT]](length=B * Self.SC, fill=Scalar[DT](0))
            var smpr = List[Scalar[DT]](length=B * Self.SC, fill=Scalar[DT](0))
            var ppr = List[Scalar[DT]](length=B * Self.SC, fill=Scalar[DT](0))
            for b in range(B):
                # re-derive the free-bits gate: active = (Σ_s KL > free_nats).
                var kl_sum: Scalar[DT] = 0.0
                for s in range(Self.STOCH):
                    var base = (b * Self.STOCH + s) * Self.CLASSES
                    self._softmax_mix_cpu(post.data, base, smpo, ppo)
                    self._softmax_mix_cpu(prior.data, base, smpr, ppr)
                    for c in range(Self.CLASSES):
                        var pp = ppo[base + c]
                        kl_sum += pp * (log(pp) - log(ppr[base + c]))
                var act = (
                    Scalar[DT](1.0) if kl_sum > self.free_nats
                    else Scalar[DT](0.0)
                )
                # dyn cotangent (channel 0) → grad_prior; rep (channel 1) → post
                var wd = grad_output.data[b * 2] * act
                var wr = grad_output.data[b * 2 + 1] * act
                for s in range(Self.STOCH):
                    var base = (b * Self.STOCH + s) * Self.CLASSES
                    # S = Σ_c (p_post[c]/p_prior[c])·sm_prior[c]   (dyn → prior)
                    var S: Scalar[DT] = 0.0
                    for c in range(Self.CLASSES):
                        S += (ppo[base + c] / ppr[base + c]) * smpr[base + c]
                    # W = Σ_c sm_post[c]·(log p_post[c]−log p_prior[c]) (rep→post)
                    var W: Scalar[DT] = 0.0
                    for c in range(Self.CLASSES):
                        W += smpo[base + c] * (
                            log(ppo[base + c]) - log(ppr[base + c])
                        )
                    for j in range(Self.CLASSES):
                        var a_over_p = ppo[base + j] / ppr[base + j]
                        var gpr = -one_m_u * smpr[base + j] * (a_over_p - S)
                        g_prior.data[base + j] = wd * gpr
                        var lr_diff = log(ppo[base + j]) - log(ppr[base + j])
                        var gpo = one_m_u * smpo[base + j] * (lr_diff - W)
                        g_post.data[base + j] = wr * gpo
        else:
            var c = ctx.value()
            comptime NN = B * Self.SC
            comptime NG = B * Self.STOCH
            g_post.ensure_gpu(c, NN)
            g_prior.ensure_gpu(c, NN)
            # recompute softmaxes + active (forward scratch was transient).
            var smpo = Tensor.alloc_gpu(c, NN)
            var ppo = Tensor.alloc_gpu(c, NN)
            var smpr = Tensor.alloc_gpu(c, NN)
            var ppr = Tensor.alloc_gpu(c, NN)
            var klp = Tensor.alloc_gpu(c, NG)
            var act = Tensor.alloc_gpu(c, B)
            comptime nb1 = (NG + TPB - 1) // TPB
            c.enqueue_function[_kl_fwd1_kernel[NG, Self.CLASSES]](
                post.tt["gpu", NN](),
                prior.tt["gpu", NN](),
                smpo.tt["gpu", NN](),
                ppo.tt["gpu", NN](),
                smpr.tt["gpu", NN](),
                ppr.tt["gpu", NN](),
                klp.tt["gpu", NG](),
                self.unimix,
                grid_dim=nb1, block_dim=TPB,
            )
            # dummy out for the clamp kernel — we only need `act`.
            var dout = Tensor.alloc_gpu(c, B * 2)
            comptime nb2 = (B + TPB - 1) // TPB
            c.enqueue_function[_kl_fwd2_kernel[B, Self.STOCH]](
                klp.tt["gpu", NG](),
                dout.tt["gpu", B * 2](),
                act.tt["gpu", B](),
                self.free_nats,
                grid_dim=nb2, block_dim=TPB,
            )
            comptime nb = (NG + TPB - 1) // TPB
            c.enqueue_function[_kl_bwd_kernel[NG, Self.CLASSES, Self.STOCH]](
                grad_output.tt["gpu", B * 2](),
                act.tt["gpu", B](),
                smpo.tt["gpu", NN](),
                ppo.tt["gpu", NN](),
                smpr.tt["gpu", NN](),
                ppr.tt["gpu", NN](),
                g_post.tt["gpu", NN](),
                g_prior.tt["gpu", NN](),
                one_m_u,
                grid_dim=nb, block_dim=TPB,
            )
