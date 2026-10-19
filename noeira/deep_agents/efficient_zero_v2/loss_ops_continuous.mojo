"""EfficientZeroV2 continuous-action policy loss — squashed-Gaussian NLL.

The continuous EZv2 agent's prediction head emits ``[μ_raw | σ_raw | value]``;
the policy is a **tanh-squashed Gaussian** (Dreamer-v3 soft-clamped mean +
softplus std). Training behavior-clones the search-selected action ``a*`` by
minimizing its negative log-probability under the squashed Gaussian, plus an
entropy bonus (EZv2 "simple-π" objective, paper Eq. 8):

    μ_d   = soft_clamp · tanh(μ_raw_d / soft_clamp)
    σ_d   = softplus(σ_raw_d + init_std) + min_std
    c_d   = clamp(a*_d / max_action, ±0.999)
    u*_d  = atanh(c_d)
    η_d   = (u*_d − μ_d) / σ_d
    nlp_d = ½·η_d² + log σ_d + ½·log(2π) + log(1 − c_d²)      (tanh Jacobian)
    H_d   = log σ_d + ½·log(2πe)                               (Gaussian entropy)
    loss  = Σ_d ( nlp_d − ent_scale · H_d )

**Entropy choice (LEAN deviation from legacy):** legacy estimates the *squashed*
entropy by Monte-Carlo (1024 samples, includes the tanh log-det). We use the
closed-form *Gaussian* entropy ``Σ_d log σ_d + const`` — deterministic and
analytically differentiable (so it gradchecks), dropping the tanh correction.
That keeps the σ-inflating pressure (the dominant effect) without MC noise; the
Atari-grade MC estimator is a follow-up.

Analytic gradients (target action ``a*`` is detached):
    ∂loss/∂μ_d     = −η_d / σ_d
    ∂loss/∂σ_d     = (1 − η_d² − ent_scale) / σ_d
    ∂μ_d/∂μ_raw_d  = 1 − (μ_d / soft_clamp)²            (= 1 − tanh²)
    ∂σ_d/∂σ_raw_d  = sigmoid(σ_raw_d + init_std)
Gradchecked vs finite differences in
``tests/deep_agents/test_ezv2_continuous_policy_gradcheck.mojo``.
"""

from std.math import log, exp, tanh, sqrt
from max.gpu import global_idx

from noeira.nn.constants import DT
from noeira.core.tiles import Tile


comptime _LOG2PI: Scalar[DT] = Scalar[DT](1.8378770664093453)   # log(2π)
comptime _LOG2PIE: Scalar[DT] = Scalar[DT](2.8378770664093453)  # log(2πe)


@always_inline
def _softplus(x: Scalar[DT]) -> Scalar[DT]:
    # numerically-stable softplus
    if x > Scalar[DT](20.0):
        return x
    return log(Scalar[DT](1.0) + exp(x))


@always_inline
def _atanh(x: Scalar[DT]) -> Scalar[DT]:
    return Scalar[DT](0.5) * log(
        (Scalar[DT](1.0) + x) / (Scalar[DT](1.0) - x)
    )


def continuous_policy_loss_and_grad[
    BATCH: Int, ACT_DIM: Int,
](
    musig: List[Scalar[DT]],     # [BATCH, 2*ACT_DIM]
    target_act: List[Scalar[DT]],# [BATCH, ACT_DIM]
    grad_scale: Scalar[DT],
    mut grad_musig: List[Scalar[DT]],  # [BATCH, 2*ACT_DIM]
    max_action: Scalar[DT] = Scalar[DT](1.0),
    min_std: Scalar[DT] = Scalar[DT](0.1),
    soft_clamp: Scalar[DT] = Scalar[DT](5.0),
    init_std: Scalar[DT] = Scalar[DT](1.0),
    ent_scale: Scalar[DT] = Scalar[DT](5e-3),
) -> Scalar[DT]:
    """Squashed-Gaussian NLL + Gaussian-entropy bonus over ``BATCH`` rows.

    ``musig`` packs ``[μ_raw | σ_raw]`` (each ``ACT_DIM``) per row; ``target_act``
    is the search-selected action. Returns the **summed** loss and writes
    ``grad_musig[b, :ACT_DIM] = grad_scale·∂loss/∂μ_raw`` and
    ``grad_musig[b, ACT_DIM:] = grad_scale·∂loss/∂σ_raw``.
    """
    var total = Scalar[DT](0.0)
    for b in range(BATCH):
        var mbase = b * 2 * ACT_DIM
        var abase = b * ACT_DIM
        for d in range(ACT_DIM):
            var mu_raw = musig[mbase + d]
            var sig_raw = musig[mbase + ACT_DIM + d]
            # forward parameterization
            var th = tanh(mu_raw / soft_clamp)
            var mu = soft_clamp * th
            var sp = _softplus(sig_raw + init_std)
            var sig = sp + min_std
            # target squashing
            var c = target_act[abase + d] / max_action
            if c > Scalar[DT](0.999):
                c = Scalar[DT](0.999)
            if c < Scalar[DT](-0.999):
                c = Scalar[DT](-0.999)
            var ustar = _atanh(c)
            var eta = (ustar - mu) / sig
            var nlp = (
                Scalar[DT](0.5) * eta * eta
                + log(sig)
                + Scalar[DT](0.5) * _LOG2PI
                + log(Scalar[DT](1.0) - c * c)
            )
            var ent = log(sig) + Scalar[DT](0.5) * _LOG2PIE
            total += nlp - ent_scale * ent
            # analytic grads wrt μ, σ
            var dmu = -eta / sig
            var dsig = (Scalar[DT](1.0) - eta * eta - ent_scale) / sig
            # chain through parameterization
            var dmu_raw = dmu * (Scalar[DT](1.0) - th * th)
            var sig_part = Scalar[DT](1.0) / (
                Scalar[DT](1.0) + exp(-(sig_raw + init_std))
            )  # sigmoid
            var dsig_raw = dsig * sig_part
            grad_musig[mbase + d] = grad_scale * dmu_raw
            grad_musig[mbase + ACT_DIM + d] = grad_scale * dsig_raw
    return total


# ──────────────────────────────────────────────────────────────────────
# GPU device mirror — slice-write into a PRED_OUT-strided grad buffer
# ──────────────────────────────────────────────────────────────────────


def continuous_policy_loss_grad_k[
    B_: Int, ACT_DIM_: Int, PRED_OUT_: Int,
](
    pout: Tile[DT, B_ * PRED_OUT_],
    target_act: Tile[DT, B_ * ACT_DIM_],
    grad_pout: Tile[DT, B_ * PRED_OUT_],
    loss_buf: Tile[DT, B_],
    grad_scale: Scalar[DT],
    loss_coef: Scalar[DT],
    max_action: Scalar[DT],
    min_std: Scalar[DT],
    soft_clamp: Scalar[DT],
    init_std: Scalar[DT],
    ent_scale: Scalar[DT],
):
    """GPU per-row squashed-Gaussian policy NLL — device mirror of
    ``continuous_policy_loss_and_grad``. One thread per row ``b``: reads the
    ``[μ_raw | σ_raw]`` slice (the leading ``2·ACT_DIM`` of the ``PRED_OUT``-wide
    ``pout`` row), **accumulates** ``loss_coef·Σ_d (nlp_d − ent_scale·H_d)`` into
    ``loss_buf[b]``, and writes ``grad_scale·∂loss/∂μ_raw`` /
    ``grad_scale·∂loss/∂σ_raw`` into the same slice of ``grad_pout``. The value
    slice ``[2·ACT_DIM, 2·ACT_DIM+BINS)`` is written by the value soft-CE kernel
    (disjoint), so no zeroing is required. Math is the same scalar sequence as
    the CPU op (parity ≈ reduction order only)."""
    var b = Int(global_idx.x)
    if b < B_:
        var mbase = b * PRED_OUT_       # μ_raw at [mbase + d], σ_raw at [mbase + ACT_DIM_ + d]
        var abase = b * ACT_DIM_
        var row = Scalar[DT](0.0)
        for d in range(ACT_DIM_):
            var mu_raw = rebind[Scalar[DT]](pout[mbase + d])
            var sig_raw = rebind[Scalar[DT]](pout[mbase + ACT_DIM_ + d])
            var th = tanh(mu_raw / soft_clamp)
            var mu = soft_clamp * th
            var sp_in = sig_raw + init_std
            var sp = sp_in if sp_in > Scalar[DT](20.0) else log(
                Scalar[DT](1.0) + exp(sp_in)
            )
            var sig = sp + min_std
            var c = rebind[Scalar[DT]](target_act[abase + d]) / max_action
            if c > Scalar[DT](0.999):
                c = Scalar[DT](0.999)
            if c < Scalar[DT](-0.999):
                c = Scalar[DT](-0.999)
            var ustar = Scalar[DT](0.5) * log(
                (Scalar[DT](1.0) + c) / (Scalar[DT](1.0) - c)
            )
            var eta = (ustar - mu) / sig
            var nlp = (
                Scalar[DT](0.5) * eta * eta
                + log(sig)
                + Scalar[DT](0.5) * _LOG2PI
                + log(Scalar[DT](1.0) - c * c)
            )
            var ent = log(sig) + Scalar[DT](0.5) * _LOG2PIE
            row += nlp - ent_scale * ent
            var dmu = -eta / sig
            var dsig = (Scalar[DT](1.0) - eta * eta - ent_scale) / sig
            var dmu_raw = dmu * (Scalar[DT](1.0) - th * th)
            var sig_part = Scalar[DT](1.0) / (
                Scalar[DT](1.0) + exp(-(sig_raw + init_std))
            )
            var dsig_raw = dsig * sig_part
            grad_pout[mbase + d] = grad_scale * dmu_raw
            grad_pout[mbase + ACT_DIM_ + d] = grad_scale * dsig_raw
        loss_buf[b] = rebind[Scalar[DT]](loss_buf[b]) + loss_coef * row
