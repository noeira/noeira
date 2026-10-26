"""Two-hot encoding for distributional value heads (DreamerV3 / TD-MPC2).

Two-hot encoding converts a scalar `x` into a soft one-hot vector over
`NUM_BINS` evenly-spaced bin values:

  k = floor((x - v_min) / step),   step = (v_max - v_min) / (NUM_BINS - 1)
  target[k]     = (bins[k+1] - x) / step
  target[k + 1] = (x - bins[k])   / step
  target[i]     = 0  otherwise

The scalar value is decoded from logits as
  v = sum_i softmax(logits)_i * bins[i]
and (in DreamerV3's symlog form) `symexp(v)` recovers the real scale.

Storage-surface port of the legacy nn helpers
(`noeira/nn/loss/two_hot.mojo`):
  * scalar helpers stay Array-based (compile-time-sized inline use)
  * batched helpers operate on `Tensor` storages (CPU `.data` loops) instead
    of the legacy raw `Pointer` form; they keep the `_ptr` suffix off
    their names since they no longer take pointers.
"""

from std.math import exp, log
from std.math import abs as math_abs
from max.gpu import global_idx
from max.gpu.host import DeviceContext

from noeira.nn.constants import DT, TPB
from ..core.tensor import Tensor
from ..core.tensor_refs import TensorRefs

# Scalar symlog / symexp live in the canonical shared math helper (DT-only)
# and are re-exported here so `from ...storage.loss.two_hot import symlog`
# call-sites keep working.
from noeira.nn.primitives.ops.symlog_math import symlog, symexp
from noeira.core.tiles import Tile


# ──────────────────────────────────────────────────────────────────────
# Bin construction (compile-time-sized Array).
# ──────────────────────────────────────────────────────────────────────


def compute_bins[
    NUM_BINS: Int,
](v_min: Scalar[DT], v_max: Scalar[DT]) -> Array[Scalar[DT], NUM_BINS]:
    """Evenly-spaced bins in [v_min, v_max]."""
    var bins = Array[Scalar[DT], NUM_BINS](fill=0)
    if NUM_BINS == 1:
        bins[0] = (v_min + v_max) * Scalar[DT](0.5)
        return bins^
    var step = (v_max - v_min) / Scalar[DT](NUM_BINS - 1)
    for i in range(NUM_BINS):
        bins[i] = v_min + step * Scalar[DT](i)
    return bins^


def compute_symlog_bins[
    NUM_BINS: Int,
]() -> Array[Scalar[DT], NUM_BINS]:
    """DreamerV3 default: bins evenly spaced in symlog space, range [-20, 20]."""
    var bins = Array[Scalar[DT], NUM_BINS](fill=0)
    if NUM_BINS == 1:
        bins[0] = Scalar[DT](0.0)
        return bins^
    var step = Scalar[DT](40.0) / Scalar[DT](NUM_BINS - 1)
    for i in range(NUM_BINS):
        bins[i] = Scalar[DT](-20.0) + step * Scalar[DT](i)
    return bins^


def fill_bins[
    NUM_BINS: Int,
](
    v_min: Scalar[DT],
    v_max: Scalar[DT],
    mut bins: Tensor,
):
    """Tensor-form of `compute_bins` (writes the first NUM_BINS of `bins`)."""
    bins.ensure(NUM_BINS)
    if NUM_BINS == 1:
        bins.data[0] = (v_min + v_max) * Scalar[DT](0.5)
        return
    var step = (v_max - v_min) / Scalar[DT](NUM_BINS - 1)
    for i in range(NUM_BINS):
        bins.data[i] = v_min + step * Scalar[DT](i)


def fill_symlog_bins[
    NUM_BINS: Int,
](mut bins: Tensor):
    """Tensor-form of `compute_symlog_bins`."""
    bins.ensure(NUM_BINS)
    if NUM_BINS == 1:
        bins.data[0] = Scalar[DT](0.0)
        return
    var step = Scalar[DT](40.0) / Scalar[DT](NUM_BINS - 1)
    for i in range(NUM_BINS):
        bins.data[i] = Scalar[DT](-20.0) + step * Scalar[DT](i)


# ──────────────────────────────────────────────────────────────────────
# Two-hot encoding (single-sample + batched).
# ──────────────────────────────────────────────────────────────────────


def two_hot_encode[
    NUM_BINS: Int,
](
    x: Scalar[DT],
    bins: Array[Scalar[DT], NUM_BINS],
    mut target: Array[Scalar[DT], NUM_BINS],
):
    """Encode scalar `x` into a two-hot distribution over `bins`."""
    for i in range(NUM_BINS):
        target[i] = Scalar[DT](0.0)

    var v_min = bins[0]
    var v_max = bins[NUM_BINS - 1]
    var x_clamped = x
    if x_clamped < v_min:
        x_clamped = v_min
    if x_clamped > v_max:
        x_clamped = v_max

    if NUM_BINS == 1:
        target[0] = Scalar[DT](1.0)
        return

    var step = (v_max - v_min) / Scalar[DT](NUM_BINS - 1)
    var k_float = (x_clamped - v_min) / step
    var k = Int(k_float)
    if k >= NUM_BINS - 1:
        k = NUM_BINS - 2

    var bin_low = bins[k]
    var bin_high = bins[k + 1]
    var width = bin_high - bin_low

    if math_abs(width) < Scalar[DT](1e-8):
        target[k] = Scalar[DT](1.0)
        return

    var upper_weight = (bin_high - x_clamped) / width
    var lower_weight = Scalar[DT](1.0) - upper_weight
    target[k] = upper_weight
    target[k + 1] = lower_weight


def two_hot_encode_batch[
    BATCH: Int, NUM_BINS: Int,
](
    ref values: Tensor,
    ref bins: Tensor,
    mut targets: Tensor,
):
    """Batch two-hot encode (CPU `.data` loops). Inputs:
        values  shape [BATCH]
        bins    shape [NUM_BINS]    (must be sorted ascending, evenly spaced)
        targets shape [BATCH * NUM_BINS] — written.
    """
    targets.ensure(BATCH * NUM_BINS)
    var v_min = bins.data[0]
    var v_max = bins.data[NUM_BINS - 1]

    for b in range(BATCH):
        var base = b * NUM_BINS
        var x = values.data[b]
        if x < v_min:
            x = v_min
        if x > v_max:
            x = v_max

        for i in range(NUM_BINS):
            targets.data[base + i] = Scalar[DT](0.0)

        if NUM_BINS == 1:
            targets.data[base] = Scalar[DT](1.0)
            continue

        var step = (v_max - v_min) / Scalar[DT](NUM_BINS - 1)
        var k_float = (x - v_min) / step
        var k = Int(k_float)
        if k >= NUM_BINS - 1:
            k = NUM_BINS - 2

        var bin_low = bins.data[k]
        var bin_high = bins.data[k + 1]
        var width = bin_high - bin_low

        if math_abs(width) < Scalar[DT](1e-8):
            targets.data[base + k] = Scalar[DT](1.0)
            continue

        var upper = (bin_high - x) / width
        targets.data[base + k] = upper
        targets.data[base + k + 1] = Scalar[DT](1.0) - upper


def two_hot_encode_symlog_batch[
    BATCH: Int, NUM_BINS: Int,
](
    ref values: Tensor,
    ref bins: Tensor,
    mut targets: Tensor,
):
    """DreamerV3-style: apply `symlog` to each value before encoding
    against symlog-spaced bins."""
    targets.ensure(BATCH * NUM_BINS)
    var v_min = bins.data[0]
    var v_max = bins.data[NUM_BINS - 1]

    for b in range(BATCH):
        var base = b * NUM_BINS
        var x = symlog(values.data[b])
        if x < v_min:
            x = v_min
        if x > v_max:
            x = v_max

        for i in range(NUM_BINS):
            targets.data[base + i] = Scalar[DT](0.0)

        if NUM_BINS == 1:
            targets.data[base] = Scalar[DT](1.0)
            continue

        var step = (v_max - v_min) / Scalar[DT](NUM_BINS - 1)
        var k_float = (x - v_min) / step
        var k = Int(k_float)
        if k >= NUM_BINS - 1:
            k = NUM_BINS - 2

        var bin_low = bins.data[k]
        var bin_high = bins.data[k + 1]
        var width = bin_high - bin_low

        if math_abs(width) < Scalar[DT](1e-8):
            targets.data[base + k] = Scalar[DT](1.0)
            continue

        var upper = (bin_high - x) / width
        targets.data[base + k] = upper
        targets.data[base + k + 1] = Scalar[DT](1.0) - upper


# ──────────────────────────────────────────────────────────────────────
# Decoding: logits → expected value.
# ──────────────────────────────────────────────────────────────────────


def decode_value[
    NUM_BINS: Int,
](
    logits: Array[Scalar[DT], NUM_BINS],
    bins: Array[Scalar[DT], NUM_BINS],
) -> Scalar[DT]:
    """Decode distributional value with symexp: returns
    `symexp(sum_i softmax(logits)_i * bins_i)` — bins live in symlog
    space, so the symexp recovers actual-value scale."""
    var max_val = logits[0]
    for i in range(1, NUM_BINS):
        if logits[i] > max_val:
            max_val = logits[i]

    var sum_exp = Scalar[DT](0.0)
    for i in range(NUM_BINS):
        sum_exp += exp(logits[i] - max_val)

    var value_symlog = Scalar[DT](0.0)
    for i in range(NUM_BINS):
        var prob = exp(logits[i] - max_val) / sum_exp
        value_symlog += prob * bins[i]

    return symexp(value_symlog)


def decode_value_batch[
    BATCH: Int, NUM_BINS: Int,
](
    ref logits: Tensor,
    ref bins: Tensor,
    mut values: Tensor,
):
    """Batch decode distributional values with symexp. `logits` shape
    [BATCH * NUM_BINS]; `bins` shape [NUM_BINS] (symlog space);
    `values` shape [BATCH] (actual-value space, written)."""
    values.ensure(BATCH)
    for b in range(BATCH):
        var base = b * NUM_BINS
        var max_val = logits.data[base]
        for i in range(1, NUM_BINS):
            var lv = logits.data[base + i]
            if lv > max_val:
                max_val = lv

        var sum_exp = Scalar[DT](0.0)
        for i in range(NUM_BINS):
            sum_exp += exp(logits.data[base + i] - max_val)

        var val_symlog = Scalar[DT](0.0)
        for i in range(NUM_BINS):
            var prob = exp(logits.data[base + i] - max_val) / sum_exp
            val_symlog += prob * bins.data[i]

        values.data[b] = symexp(val_symlog)


def decode_value_batch_linear[
    BATCH: Int, NUM_BINS: Int,
](
    ref logits: Tensor,
    ref bins: Tensor,
    mut values: Tensor,
):
    """Like `decode_value_batch` but bins are in actual-value space
    (no symexp). For TD-MPC2 / pre-DreamerV3 distributional heads."""
    values.ensure(BATCH)
    for b in range(BATCH):
        var base = b * NUM_BINS
        var max_val = logits.data[base]
        for i in range(1, NUM_BINS):
            var lv = logits.data[base + i]
            if lv > max_val:
                max_val = lv

        var sum_exp = Scalar[DT](0.0)
        for i in range(NUM_BINS):
            sum_exp += exp(logits.data[base + i] - max_val)

        var val = Scalar[DT](0.0)
        for i in range(NUM_BINS):
            var prob = exp(logits.data[base + i] - max_val) / sum_exp
            val += prob * bins.data[i]

        values.data[b] = val


# ──────────────────────────────────────────────────────────────────────
# Fused soft cross-entropy against a two-hot target  (value / reward heads).
#
# Given logits[B,NUM_BINS], a bin grid, and a per-row target SCALAR, the CE is
#   CE[b] = -(w_lo·logp[lo] + w_hi·logp[hi])
# where (lo,hi) are the two bins bracketing the (optionally symlog'd) target and
# (w_lo,w_hi) the linear-interpolation weights (lo==hi → 0.5/0.5). This is the
# bin-agnostic form shared by TD-MPC2 (linear bins in symlog space) and
# DreamerV3 (symlog-spaced bins) value heads. `SYMLOG=True` compresses the raw
# target before bracketing (both heads do this); pass `SYMLOG=False` for a
# target already in bin space. Backward: grad_logits = grad_out·(softmax − two-
# hot target); the target itself is detached (its grad is the caller's to zero).
#
# Provided in two forms with identical math: Tensor `.data` CPU helpers and
# generic GPU kernels — so the op wrappers (and DreamerV3, on migration) share
# one implementation rather than each carrying a private copy.
# ──────────────────────────────────────────────────────────────────────


def two_hot_ce_loss_batch[
    BATCH: Int, NUM_BINS: Int, SYMLOG: Bool, o: MutOrigin
](
    inputs: TensorRefs[2, o],
    ref bins: Tensor,
    mut out: Tensor,
):
    """CPU fused two-hot soft cross-entropy. `inputs[0]` = logits
    [BATCH*NUM_BINS], `inputs[1]` = targets [BATCH] (raw; symlog'd here iff
    SYMLOG), `bins` [NUM_BINS], `out` [BATCH] (written). The operands arrive
    as a `TensorRefs` (one shared origin) so a graph node can pass its pooled
    inputs + write its pooled output without a §B0 ref/mut aliasing clash."""
    ref logits = inputs[0]
    ref targets = inputs[1]
    out.ensure(BATCH)
    for b in range(BATCH):
        var base = b * NUM_BINS
        var raw = targets.data[b]
        var target = symlog(raw) if SYMLOG else raw
        # bracket via count of bins ≤ target.
        var n_le = 0
        for c in range(NUM_BINS):
            if bins.data[c] <= target:
                n_le += 1
        var below = n_le - 1
        var above = n_le
        if below < 0:
            below = 0
        if below > NUM_BINS - 1:
            below = NUM_BINS - 1
        if above > NUM_BINS - 1:
            above = NUM_BINS - 1
        var w_below: Scalar[DT]
        var w_above: Scalar[DT]
        if below == above:
            w_below = Scalar[DT](0.5)
            w_above = Scalar[DT](0.5)
        else:
            var db = math_abs(bins.data[below] - target)
            var da = math_abs(bins.data[above] - target)
            var tot = db + da
            w_below = da / tot
            w_above = db / tot
        # log-sum-exp over the row.
        var zmax = logits.data[base]
        for c in range(1, NUM_BINS):
            if logits.data[base + c] > zmax:
                zmax = logits.data[base + c]
        var ssum = Scalar[DT](0.0)
        for c in range(NUM_BINS):
            ssum += exp(logits.data[base + c] - zmax)
        var lse = zmax + log(ssum)
        var lp_b = logits.data[base + below] - lse
        var lp_a = logits.data[base + above] - lse
        out.data[b] = -(w_below * lp_b + w_above * lp_a)


def two_hot_ce_backward_batch[
    BATCH: Int, NUM_BINS: Int, SYMLOG: Bool, ofi: MutOrigin, ogi: MutOrigin
](
    forward_input: TensorRefs[2, ofi],
    ref bins: Tensor,
    ref grad_out: Tensor,
    grad_inputs: TensorRefs[2, ogi],
):
    """CPU backward of `two_hot_ce_loss_batch`. `forward_input[0]` = logits,
    `[1]` = targets; writes `grad_inputs[0]` [BATCH*NUM_BINS] =
    grad_out·(softmax − two-hot target) and zeroes `grad_inputs[1]` [BATCH]
    (target detached). TensorRefs operands → §B0-safe from a graph node."""
    ref logits = forward_input[0]
    ref targets = forward_input[1]
    ref grad_logits = grad_inputs[0]
    ref grad_targets = grad_inputs[1]
    grad_logits.ensure(BATCH * NUM_BINS)
    grad_targets.ensure(BATCH)
    for b in range(BATCH):
        grad_targets.data[b] = Scalar[DT](0.0)
    for b in range(BATCH):
        var base = b * NUM_BINS
        var raw = targets.data[b]
        var target = symlog(raw) if SYMLOG else raw
        var up = grad_out.data[b]
        var n_le = 0
        for c in range(NUM_BINS):
            if bins.data[c] <= target:
                n_le += 1
        var below = n_le - 1
        var above = n_le
        if below < 0:
            below = 0
        if below > NUM_BINS - 1:
            below = NUM_BINS - 1
        if above > NUM_BINS - 1:
            above = NUM_BINS - 1
        var w_below: Scalar[DT]
        var w_above: Scalar[DT]
        if below == above:
            w_below = Scalar[DT](0.5)
            w_above = Scalar[DT](0.5)
        else:
            var db = math_abs(bins.data[below] - target)
            var da = math_abs(bins.data[above] - target)
            var tot = db + da
            w_below = da / tot
            w_above = db / tot
        var zmax = logits.data[base]
        for c in range(1, NUM_BINS):
            if logits.data[base + c] > zmax:
                zmax = logits.data[base + c]
        var ssum = Scalar[DT](0.0)
        for c in range(NUM_BINS):
            ssum += exp(logits.data[base + c] - zmax)
        var inv = Scalar[DT](1.0) / ssum
        for c in range(NUM_BINS):
            grad_logits.data[base + c] = up * (
                exp(logits.data[base + c] - zmax) * inv
            )
        grad_logits.data[base + below] = (
            grad_logits.data[base + below] - up * w_below
        )
        grad_logits.data[base + above] = (
            grad_logits.data[base + above] - up * w_above
        )


def two_hot_decode_batch[
    BATCH: Int, NUM_BINS: Int, o: MutOrigin
](
    inputs: TensorRefs[1, o],
    ref bins: Tensor,
    mut out: Tensor,
):
    """CPU symexp decode as a graph node: `inputs[0]` = logits
    [BATCH*NUM_BINS], `out` [BATCH] (written). TensorRefs form of
    `decode_value_batch` (kept separately for eval/callback callers)."""
    ref logits = inputs[0]
    out.ensure(BATCH)
    for b in range(BATCH):
        var base = b * NUM_BINS
        var max_val = logits.data[base]
        for i in range(1, NUM_BINS):
            var lv = logits.data[base + i]
            if lv > max_val:
                max_val = lv
        var sum_exp = Scalar[DT](0.0)
        for i in range(NUM_BINS):
            sum_exp += exp(logits.data[base + i] - max_val)
        var s = Scalar[DT](0.0)
        for i in range(NUM_BINS):
            var prob = exp(logits.data[base + i] - max_val) / sum_exp
            s += prob * bins.data[i]
        out.data[b] = symexp(s)


def decode_value_backward_batch[
    BATCH: Int, NUM_BINS: Int, ofi: MutOrigin, ogi: MutOrigin
](
    forward_input: TensorRefs[1, ofi],
    ref bins: Tensor,
    ref grad_out: Tensor,
    grad_inputs: TensorRefs[1, ogi],
):
    """CPU backward of `two_hot_decode_batch` (symexp decode). For
    s = Σ_c softmax(logits)_c·bins_c and v = symexp(s):
        dv/ds        = exp(|s|)
        d v/d logit_c = dv/ds · softmax_c · (bins_c − s)
    Writes `grad_inputs[0]` [BATCH*NUM_BINS] = grad_out · dv/dlogit."""
    ref logits = forward_input[0]
    ref grad_logits = grad_inputs[0]
    grad_logits.ensure(BATCH * NUM_BINS)
    for b in range(BATCH):
        var base = b * NUM_BINS
        var up = grad_out.data[b]
        var zmax = logits.data[base]
        for c in range(1, NUM_BINS):
            if logits.data[base + c] > zmax:
                zmax = logits.data[base + c]
        var ssum = Scalar[DT](0.0)
        for c in range(NUM_BINS):
            ssum += exp(logits.data[base + c] - zmax)
        var inv = Scalar[DT](1.0) / ssum
        var s = Scalar[DT](0.0)
        for c in range(NUM_BINS):
            s += exp(logits.data[base + c] - zmax) * inv * bins.data[c]
        var a = math_abs(s)
        var dds = exp(a)
        for c in range(NUM_BINS):
            var p = exp(logits.data[base + c] - zmax) * inv
            grad_logits.data[base + c] = up * dds * p * (bins.data[c] - s)


# ── GPU kernels (one thread per batch row) — same math as the CPU helpers. ──


def two_hot_ce_fwd_kernel[
    B: Int, BINS: Int, SYMLOG: Bool
](
    lg: Tile[DT, B * BINS],
    tg: Tile[DT, B],
    bins: Tile[DT, BINS],
    o: Tile[DT, B],
):
    var b = Int(global_idx.x)
    if b < B:
        var base = b * BINS
        var raw = rebind[Scalar[DT]](tg[b])
        var target: Scalar[DT]
        comptime if SYMLOG:
            var sgn = Scalar[DT](1.0) if raw >= Scalar[DT](0.0) else Scalar[DT](
                -1.0
            )
            var av = raw if raw >= Scalar[DT](0.0) else -raw
            target = sgn * log(Scalar[DT](1.0) + av)
        else:
            target = raw
        var n_le = 0
        for c in range(BINS):
            if rebind[Scalar[DT]](bins[c]) <= target:
                n_le += 1
        var below = n_le - 1
        var above = n_le
        if below < 0:
            below = 0
        if below > BINS - 1:
            below = BINS - 1
        if above > BINS - 1:
            above = BINS - 1
        var w_below: Scalar[DT]
        var w_above: Scalar[DT]
        if below == above:
            w_below = Scalar[DT](0.5)
            w_above = Scalar[DT](0.5)
        else:
            var db = rebind[Scalar[DT]](bins[below]) - target
            var da = rebind[Scalar[DT]](bins[above]) - target
            db = db if db >= Scalar[DT](0.0) else -db
            da = da if da >= Scalar[DT](0.0) else -da
            var tot = db + da
            w_below = da / tot
            w_above = db / tot
        var zmax = rebind[Scalar[DT]](lg[base])
        for c in range(1, BINS):
            var v = rebind[Scalar[DT]](lg[base + c])
            if v > zmax:
                zmax = v
        var ssum = Scalar[DT](0.0)
        for c in range(BINS):
            ssum += exp(rebind[Scalar[DT]](lg[base + c]) - zmax)
        var lse = zmax + log(ssum)
        var lp_b = rebind[Scalar[DT]](lg[base + below]) - lse
        var lp_a = rebind[Scalar[DT]](lg[base + above]) - lse
        o[b] = -(w_below * lp_b + w_above * lp_a)


def two_hot_ce_bwd_kernel[
    B: Int, BINS: Int, SYMLOG: Bool
](
    go: Tile[DT, B],
    lg: Tile[DT, B * BINS],
    tg: Tile[DT, B],
    bins: Tile[DT, BINS],
    glg: Tile[DT, B * BINS],
):
    var b = Int(global_idx.x)
    if b < B:
        var base = b * BINS
        var raw = rebind[Scalar[DT]](tg[b])
        var target: Scalar[DT]
        comptime if SYMLOG:
            var sgn = Scalar[DT](1.0) if raw >= Scalar[DT](0.0) else Scalar[DT](
                -1.0
            )
            var av = raw if raw >= Scalar[DT](0.0) else -raw
            target = sgn * log(Scalar[DT](1.0) + av)
        else:
            target = raw
        var up = rebind[Scalar[DT]](go[b])
        var n_le = 0
        for c in range(BINS):
            if rebind[Scalar[DT]](bins[c]) <= target:
                n_le += 1
        var below = n_le - 1
        var above = n_le
        if below < 0:
            below = 0
        if below > BINS - 1:
            below = BINS - 1
        if above > BINS - 1:
            above = BINS - 1
        var w_below: Scalar[DT]
        var w_above: Scalar[DT]
        if below == above:
            w_below = Scalar[DT](0.5)
            w_above = Scalar[DT](0.5)
        else:
            var db = rebind[Scalar[DT]](bins[below]) - target
            var da = rebind[Scalar[DT]](bins[above]) - target
            db = db if db >= Scalar[DT](0.0) else -db
            da = da if da >= Scalar[DT](0.0) else -da
            var tot = db + da
            w_below = da / tot
            w_above = db / tot
        var zmax = rebind[Scalar[DT]](lg[base])
        for c in range(1, BINS):
            var v = rebind[Scalar[DT]](lg[base + c])
            if v > zmax:
                zmax = v
        var ssum = Scalar[DT](0.0)
        for c in range(BINS):
            ssum += exp(rebind[Scalar[DT]](lg[base + c]) - zmax)
        var inv = Scalar[DT](1.0) / ssum
        for c in range(BINS):
            glg[base + c] = up * (exp(rebind[Scalar[DT]](lg[base + c]) - zmax) * inv)
        glg[base + below] = rebind[Scalar[DT]](glg[base + below]) - up * w_below
        glg[base + above] = rebind[Scalar[DT]](glg[base + above]) - up * w_above


def decode_value_fwd_kernel[
    B: Int, BINS: Int
](
    lg: Tile[DT, B * BINS],
    bins: Tile[DT, BINS],
    o: Tile[DT, B],
):
    var b = Int(global_idx.x)
    if b < B:
        var base = b * BINS
        var zmax = rebind[Scalar[DT]](lg[base])
        for c in range(1, BINS):
            var v = rebind[Scalar[DT]](lg[base + c])
            if v > zmax:
                zmax = v
        var ssum = Scalar[DT](0.0)
        for c in range(BINS):
            ssum += exp(rebind[Scalar[DT]](lg[base + c]) - zmax)
        var inv = Scalar[DT](1.0) / ssum
        var s = Scalar[DT](0.0)
        for c in range(BINS):
            s += exp(rebind[Scalar[DT]](lg[base + c]) - zmax) * inv * rebind[
                Scalar[DT]
            ](bins[c])
        var sgn = Scalar[DT](1.0) if s >= Scalar[DT](0.0) else Scalar[DT](-1.0)
        var a = s if s >= Scalar[DT](0.0) else -s
        o[b] = sgn * (exp(a) - Scalar[DT](1.0))


def decode_value_bwd_kernel[
    B: Int, BINS: Int
](
    go: Tile[DT, B],
    lg: Tile[DT, B * BINS],
    bins: Tile[DT, BINS],
    glg: Tile[DT, B * BINS],
):
    var b = Int(global_idx.x)
    if b < B:
        var base = b * BINS
        var up = rebind[Scalar[DT]](go[b])
        var zmax = rebind[Scalar[DT]](lg[base])
        for c in range(1, BINS):
            var v = rebind[Scalar[DT]](lg[base + c])
            if v > zmax:
                zmax = v
        var ssum = Scalar[DT](0.0)
        for c in range(BINS):
            ssum += exp(rebind[Scalar[DT]](lg[base + c]) - zmax)
        var inv = Scalar[DT](1.0) / ssum
        var s = Scalar[DT](0.0)
        for c in range(BINS):
            s += exp(rebind[Scalar[DT]](lg[base + c]) - zmax) * inv * rebind[
                Scalar[DT]
            ](bins[c])
        var a = s if s >= Scalar[DT](0.0) else -s
        var dds = exp(a)
        for c in range(BINS):
            var p = exp(rebind[Scalar[DT]](lg[base + c]) - zmax) * inv
            glg[base + c] = up * dds * p * (rebind[Scalar[DT]](bins[c]) - s)
