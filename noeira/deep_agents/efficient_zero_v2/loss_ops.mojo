"""EfficientZeroV2 loss primitives — SimSiam temporal-consistency.

The genuinely-new EZv2 objective over MuZero is the **negative cosine
similarity** between the online predictor output and the stop-gradient target
projection (Chen & He, "Exploring Simple Siamese Representation Learning", 2021;
EZv2 paper Eq. 3 λ_G term):

    cos(p, t) = (p · t) / (‖p‖ · ‖t‖)
    L_G       = − cos(p, sg(t))                    (stop-grad on the target t)
    ∂(−cos)/∂p_i = cos · p_i / ‖p‖²  −  t_i / (‖p‖ · ‖t‖)

Only the **online** branch ``p = h_pred(g_proj(z_k))`` receives a gradient; the
**target** branch ``t = g_proj(h(obs_k))`` is detached (no grad written, the
unroll never backprops it). The analytic gradient above is what makes the
manual reverse-scan in ``blocks.mojo`` trustworthy — it is gradchecked against
finite differences in ``tests/deep_agents/test_ezv2_consistency_gradcheck.mojo``.

``grad_scale`` folds the caller's per-unroll-step weight ``1/K`` (consistency is
summed over the ``k = 1..K`` dynamics steps, not ``K+1`` — there is no
consistency at the root), the ``1/BATCH`` mean, and the loss coefficient
``λ_G`` into the gradient in one place, exactly like ``soft_ce_loss_and_grad``.

``mask`` is the per-row **episode-boundary mask** (EZv2 reference
``mask_batch``): rows whose target obs is absorbing padding (the unroll crossed
the episode's terminal, so ``obs_k`` is the repeated last obs, not a real
future observation) get loss and gradient zeroed. Without it the consistency
objective teaches the dynamics a false fixed point — "terminal obs + absorbing
action → same obs" — precisely at the failure boundary. The reference masks
every unroll loss; here only consistency is masked (policy/value/reward keep
the MuZero absorbing-target convention, which the plain MuZero port converges
with).
"""

from std.math import sqrt
from max.gpu import global_idx

from noeira.nn.constants import DT
from noeira.core.tiles import Tile


comptime _COS_EPS: Scalar[DT] = Scalar[DT](1e-12)


def consistency_loss_and_grad[
    BATCH: Int, DIM: Int,
](
    p: List[Scalar[DT]],
    t: List[Scalar[DT]],
    grad_scale: Scalar[DT],
    mut grad_p: List[Scalar[DT]],
    mask: Optional[Pointer[Scalar[DT], MutAnyOrigin]] = None,
) -> Scalar[DT]:
    """Negative-cosine SimSiam consistency over ``BATCH`` rows of ``DIM``.

    ``p`` is the online predictor output, ``t`` the stop-grad target projection.
    Returns the **summed** loss ``Σ_b m_b·(−cos(p_b, t_b))`` and writes the
    analytic online gradient ``grad_p[b,i] = m_b · grad_scale ·
    (cos·p_i/‖p‖² − t_i/(‖p‖·‖t‖))``. No gradient is produced for ``t`` (it is
    detached). ``mask`` is the optional per-row episode-boundary mask ``m_b``
    (``None`` ≡ all ones).
    """
    var total = Scalar[DT](0.0)
    for b in range(BATCH):
        var m = Scalar[DT](1.0)
        if mask:
            m = mask.value()[unsafe_offset=b]
        var base = b * DIM
        var sum_pp = Scalar[DT](0.0)
        var sum_tt = Scalar[DT](0.0)
        var dot = Scalar[DT](0.0)
        for i in range(DIM):
            var pi = p[base + i]
            var ti = t[base + i]
            sum_pp += pi * pi
            sum_tt += ti * ti
            dot += pi * ti
        var np = sqrt(sum_pp + _COS_EPS)
        var nt = sqrt(sum_tt + _COS_EPS)
        var cos = dot / (np * nt)
        total += m * (-cos)
        # ∂(−cos)/∂p_i = cos·p_i/‖p‖² − t_i/(‖p‖·‖t‖)
        var inv_np2 = Scalar[DT](1.0) / (np * np)
        var inv_npnt = Scalar[DT](1.0) / (np * nt)
        for i in range(DIM):
            var gi = cos * p[base + i] * inv_np2 - t[base + i] * inv_npnt
            grad_p[base + i] = m * grad_scale * gi
    return total


def consistency_loss_grad_k[
    B_: Int, DIM_: Int, ADT: DType = DT,
](
    p: Tile[ADT, B_ * DIM_],
    t: Tile[ADT, B_ * DIM_],
    grad_p: Tile[ADT, B_ * DIM_],
    loss_buf: Tile[DT, B_],
    mask: Tile[DT, B_],
    grad_scale: Scalar[DT],
    loss_coef: Scalar[DT],
):
    """GPU per-row SimSiam negative-cosine consistency — device mirror of
    ``consistency_loss_and_grad``. One thread per row ``b``: computes
    ``cos(p_b, t_b)``, **accumulates** ``mask_b·loss_coef·(−cos)`` into
    ``loss_buf[b]``, and writes ``mask_b·grad_scale·(cos·p_i/‖p‖² −
    t_i/(‖p‖·‖t‖))`` into ``grad_p``. ``t`` is the detached target (no gradient
    produced for it). ``mask`` is the per-row episode-boundary mask (all-ones ≡
    unmasked, bit-identical to the pre-mask op). Math is bit-for-bit the same
    scalar sequence as the CPU op (parity ≈ reduction order only)."""
    var b = Int(global_idx.x)
    if b < B_:
        var m = rebind[Scalar[DT]](mask[b])
        var base = b * DIM_
        var sum_pp = Scalar[DT](0.0)
        var sum_tt = Scalar[DT](0.0)
        var dot = Scalar[DT](0.0)
        for i in range(DIM_):
            var pi = rebind[Scalar[ADT]](p[base + i]).cast[DT]()
            var ti = rebind[Scalar[ADT]](t[base + i]).cast[DT]()
            sum_pp += pi * pi
            sum_tt += ti * ti
            dot += pi * ti
        var np = sqrt(sum_pp + _COS_EPS)
        var nt = sqrt(sum_tt + _COS_EPS)
        var cos = dot / (np * nt)
        loss_buf[b] = rebind[Scalar[DT]](loss_buf[b]) + m * loss_coef * (-cos)
        var inv_np2 = Scalar[DT](1.0) / (np * np)
        var inv_npnt = Scalar[DT](1.0) / (np * nt)
        for i in range(DIM_):
            var pi = rebind[Scalar[ADT]](p[base + i]).cast[DT]()
            var ti = rebind[Scalar[ADT]](t[base + i]).cast[DT]()
            grad_p[base + i] = (
                m * grad_scale * (cos * pi * inv_np2 - ti * inv_npnt)
            ).cast[ADT]()
